"""Training, evaluation, and data plumbing for the student example."""

import os
from pathlib import Path
import tempfile
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
import torch

from energy_net.grid_entities.management.price_curve import (
    ActionBasedPriceStrategy,
)
from for_students.competitive_alternating import CompetitiveAlternatingISOEnv
from for_students.internal.market_support import (
    DEFAULT_GENERATION_COST,
    RecordingPCSEnv,
    STEPS_PER_DAY,
    build_combined_market_action_seen_by_pcs,
)


# CPU time received by the training process, excluding time descheduled while
# other student jobs use the machine.
DEFAULT_TRAINING_SECONDS = 300

# Responsive training is also stopped after this much elapsed wall time.
DEFAULT_WALL_SAFETY_SECONDS = 1_800

# Default random seed used by the learned market participants.
DEFAULT_RANDOM_SEED = 0

# Offset gives the competitor a reproducible exploration stream distinct from
# the newly initialized main ISO.
COMPETITOR_RANDOM_SEED_OFFSET = 2

# Daily market decisions collected for each main-ISO update.
MAIN_ISO_DAYS_PER_TRAINING_PHASE = 16

# Daily market decisions collected for each competitor update.
COMPETITOR_DAYS_PER_TRAINING_PHASE = 2

# Half-hour decisions used to keep the loaded PCS learning during each cycle.
PCS_CONTINUING_UPDATE_STEPS = STEPS_PER_DAY

# PPO learning rate used by the main ISO within the fixed training budget.
MAIN_ISO_LEARNING_RATE = 0.002

# A2C learning rate used by the main ISO within the fixed training budget.
MAIN_A2C_LEARNING_RATE = 0.0007

# Off-policy settings used by TD3 and DDPG comparisons.
MAIN_OFF_POLICY_LEARNING_RATE = 0.001
MAIN_OFF_POLICY_ACTION_NOISE = 0.1

# Conservative learning rate used while pretrained agents keep adapting.
PRETRAINED_AGENT_CONTINUING_LEARNING_RATE = 0.0001

# Number of optimization passes made by the main ISO after each rollout.
MAIN_ISO_OPTIMIZATION_EPOCHS = 10

# Hidden-layer sizes for the main ISO's more capable policy network.
MAIN_ISO_POLICY_LAYERS = (64, 64)

# Hidden layers used by the pretrained competitor policy.
COMPETITOR_POLICY_LAYERS = (64, 64)

# Value used only if a dataset begins with missing consumption before any
# earlier reading exists for causal forward filling.
MISSING_CONSUMPTION_FALLBACK = 0.0

# Saved PCS checkpoint produced once by train_pretrained_pcs.py and reused by
# every competitive experiment.
PRETRAINED_PCS_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "pretrained_pcs.zip"
)

# Saved weak-competitor checkpoint produced once after PCS pretraining and
# loaded as the same reproducible opponent at the start of every experiment.
PRETRAINED_COMPETITOR_ISO_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "pretrained_competitor_iso.zip"
)


class TrainingBudget:
    """Track CPU time fairly while retaining a wall-clock safety ceiling."""

    def __init__(self, cpu_seconds, wall_seconds):
        self.cpu_seconds = cpu_seconds
        self.wall_seconds = wall_seconds
        self.started_cpu = time.process_time()
        self.started_wall = time.monotonic()

    @property
    def used_cpu_seconds(self):
        return time.process_time() - self.started_cpu

    @property
    def used_wall_seconds(self):
        return time.monotonic() - self.started_wall

    @property
    def cpu_exhausted(self):
        return self.used_cpu_seconds >= self.cpu_seconds

    @property
    def wall_exhausted(self):
        return self.used_wall_seconds >= self.wall_seconds

    @property
    def exhausted(self):
        return self.cpu_exhausted or self.wall_exhausted

    def raise_if_wall_exhausted(self):
        if self.wall_exhausted and not self.cpu_exhausted:
            raise TimeoutError(
                "Training exceeded the wall-clock safety limit before "
                "receiving its full CPU-time budget."
            )


class StopTrainingAtBudget(BaseCallback):
    """Stop the current learning phase when the shared budget is exhausted."""

    def __init__(self, budget):
        super().__init__()
        self.budget = budget

    def _on_step(self):
        return not self.budget.exhausted


def write_clean_consumption_and_previous_day_predictions(
    source_actual_csv,
    cleaned_actual_csv,
    predictions_csv,
    previous_day=None,
):
    """Causally fill missing readings and create previous-day predictions."""

    actual = pd.read_csv(
        source_actual_csv,
        index_col=0,
        parse_dates=True,
    )
    actual["Consumption"] = (
        actual["Consumption"]
        .ffill()
        .fillna(MISSING_CONSUMPTION_FALLBACK)
    )
    actual.to_csv(cleaned_actual_csv)

    values = actual["Consumption"].to_numpy(dtype=float)
    history = np.asarray(previous_day if previous_day is not None else [])
    combined = np.concatenate([history, values])
    offset = len(history)
    predictions = [
        combined[offset + index - STEPS_PER_DAY]
        if offset + index >= STEPS_PER_DAY
        else 0.0
        for index in range(len(values))
    ]
    pd.DataFrame(
        {
            "timestamp": actual.index,
            "predicted_consumption": predictions,
        }
    ).to_csv(predictions_csv, index=False)
    return values


def create_recording_pcs_environment(actual_csv, predictions_csv):
    """Create EnergyNet's PCS environment with settlement details retained."""

    return RecordingPCSEnv(
        test_data_file=str(actual_csv),
        predictions_file=str(predictions_csv),
        verbosity=0,
    )


def create_competitive_iso_environment(
    actual_csv,
    predictions_csv,
    pcs_environment,
    pcs_model,
    opponent_iso_model=None,
    generation_cost=DEFAULT_GENERATION_COST,
):
    """Create the shared market used to train either ISO model."""

    return CompetitiveAlternatingISOEnv(
        actual_csv=str(actual_csv),
        predicted_csv=str(predictions_csv),
        pcs_env=pcs_environment,
        pcs_model=pcs_model,
        opponent_iso_model=opponent_iso_model,
        generation_cost=generation_cost,
        iso_verbosity=0,
        pcs_verbosity=0,
    )


def create_main_iso_model(algorithm, environment, seed):
    """Create the stronger ISO learner used in algorithm comparisons."""

    shared_arguments = {
        "policy": "MlpPolicy",
        "env": environment,
        "seed": seed,
        "verbose": 0,
        "policy_kwargs": {
            "net_arch": list(MAIN_ISO_POLICY_LAYERS),
        },
    }
    normalized_algorithm = algorithm.upper()
    if normalized_algorithm == "PPO":
        return PPO(
            **shared_arguments,
            learning_rate=MAIN_ISO_LEARNING_RATE,
            n_steps=MAIN_ISO_DAYS_PER_TRAINING_PHASE,
            batch_size=MAIN_ISO_DAYS_PER_TRAINING_PHASE,
            n_epochs=MAIN_ISO_OPTIMIZATION_EPOCHS,
        )
    if normalized_algorithm == "A2C":
        return A2C(
            **shared_arguments,
            learning_rate=MAIN_A2C_LEARNING_RATE,
            n_steps=MAIN_ISO_DAYS_PER_TRAINING_PHASE,
        )
    if normalized_algorithm == "SAC":
        return SAC(
            **shared_arguments,
            learning_rate=0.0003,
            buffer_size=100_000,
            learning_starts=32,
            batch_size=64,
            train_freq=(1, "step"),
            gradient_steps=1,
        )
    if normalized_algorithm in {"TD3", "DDPG"}:
        action_dimension = int(np.prod(environment.action_space.shape))
        model_class = TD3 if normalized_algorithm == "TD3" else DDPG
        return model_class(
            **shared_arguments,
            learning_rate=MAIN_OFF_POLICY_LEARNING_RATE,
            buffer_size=100_000,
            learning_starts=32,
            batch_size=64,
            train_freq=(1, "step"),
            gradient_steps=1,
            action_noise=NormalActionNoise(
                mean=np.zeros(action_dimension),
                sigma=(
                    MAIN_OFF_POLICY_ACTION_NOISE
                    * np.ones(action_dimension)
                ),
            ),
        )
    raise ValueError(
        f"Unsupported main ISO algorithm: {algorithm}. "
        "Choose PPO, A2C, SAC, TD3, or DDPG."
    )


def evaluate_iso_policy(
    iso_policy,
    pcs_model,
    competitor_iso_model,
    test_actual_csv,
    test_predictions_csv,
    generation_cost=DEFAULT_GENERATION_COST,
):
    """Evaluate one frozen ISO policy once over the chronological test data."""

    test_pcs_environment = create_recording_pcs_environment(
        test_actual_csv,
        test_predictions_csv,
    )
    test_iso_environment = create_competitive_iso_environment(
        test_actual_csv,
        test_predictions_csv,
        test_pcs_environment,
        pcs_model,
        opponent_iso_model=competitor_iso_model,
        generation_cost=generation_cost,
    )
    test_observation, _ = test_iso_environment.reset()
    main_iso_total_profit = 0.0
    main_iso_totals = {
        "retail_energy": 0.0,
        "export_energy": 0.0,
        "dispatch_energy": 0.0,
        "emergency_energy": 0.0,
        "retail_revenue": 0.0,
        "export_margin": 0.0,
        "dispatch_cost": 0.0,
        "emergency_energy_cost": 0.0,
    }
    competitor_iso_totals = {
        "profit": 0.0,
        "retail_energy": 0.0,
        "export_energy": 0.0,
        "dispatch_energy": 0.0,
        "emergency_energy": 0.0,
        "retail_revenue": 0.0,
        "export_margin": 0.0,
        "dispatch_cost": 0.0,
        "emergency_energy_cost": 0.0,
    }
    pcs_total_money = 0.0
    pcs_total_shortages = 0
    pcs_total_consumption = 0.0
    test_days = (
        len(test_iso_environment.pcs_env.test_df)
        // test_iso_environment.T
    )

    for _ in range(test_days):
        prediction = iso_policy.predict(
            test_observation,
            deterministic=True,
        )
        main_iso_action = (
            prediction[0] if isinstance(prediction, tuple) else prediction
        )
        main_iso_action = np.asarray(
            main_iso_action,
            dtype=np.float32,
        ).reshape(-1)
        if main_iso_action.shape != test_iso_environment.action_space.shape:
            raise ValueError(
                "Submitted ISO action must have shape "
                f"{test_iso_environment.action_space.shape}; "
                f"received {main_iso_action.shape}."
            )
        if not np.all(np.isfinite(main_iso_action)):
            raise ValueError("Submitted ISO action contains NaN or infinity.")
        if not test_iso_environment.action_space.contains(main_iso_action):
            raise ValueError(
                "Submitted ISO action contains values outside [0, 1]."
            )

        test_observation, reward, _, _, day_information = (
            test_iso_environment.step(main_iso_action)
        )
        main_iso_total_profit += reward
        for metric_name in main_iso_totals:
            information_name = (
                "controlled_iso_emergency_dispatch_cost"
                if metric_name == "emergency_energy_cost"
                else f"controlled_iso_{metric_name}"
            )
            main_iso_totals[metric_name] += day_information[
                information_name
            ]
        for metric_name in competitor_iso_totals:
            information_name = (
                "opponent_iso_emergency_dispatch_cost"
                if metric_name == "emergency_energy_cost"
                else f"opponent_iso_{metric_name}"
            )
            competitor_iso_totals[metric_name] += day_information[
                information_name
            ]
        pcs_total_money += day_information["pcs_money"]
        pcs_total_shortages += day_information["shortages"]
        pcs_total_consumption += float(
            np.sum(day_information["realized_consumption"])
        )
        test_observation, _ = test_iso_environment.reset()

    return {
        "learning_iso_profit": main_iso_total_profit,
        "average_daily_profit": main_iso_total_profit / test_days,
        "main_iso": {
            "profit": main_iso_total_profit,
            **main_iso_totals,
        },
        "competitor_iso": competitor_iso_totals,
        "pcs": {
            "money": pcs_total_money,
            "net_reward": (
                pcs_total_money
                - pcs_total_shortages
                * test_pcs_environment.shortage_penalty
            ),
            "grid_reliance_periods": pcs_total_shortages,
            "consumption": pcs_total_consumption,
        },
        "test_days": test_days,
    }


def run_experiment(
    training_seconds=DEFAULT_TRAINING_SECONDS,
    seed=DEFAULT_RANDOM_SEED,
    main_algorithm="SAC",
    generation_cost=DEFAULT_GENERATION_COST,
    wall_safety_seconds=DEFAULT_WALL_SAFETY_SECONDS,
):
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    data = root / "tests" / "gym" / "data_for_tests"
    train_actual = data / "synthetic_household_consumption_train.csv"
    test_actual = data / "synthetic_household_consumption_test.csv"
    with tempfile.TemporaryDirectory() as temp_dir:
        # Cleaned data and predictions are generated separately for each
        # chronological partition, so later test readings never affect train.
        cleaned_train_actual = Path(temp_dir) / "cleaned_train_actual.csv"
        cleaned_test_actual = Path(temp_dir) / "cleaned_test_actual.csv"
        train_predictions = Path(temp_dir) / "train_predictions.csv"
        test_predictions = Path(temp_dir) / "test_predictions.csv"
        training_consumption = (
            write_clean_consumption_and_previous_day_predictions(
                train_actual,
                cleaned_train_actual,
                train_predictions,
            )
        )
        write_clean_consumption_and_previous_day_predictions(
            test_actual,
            cleaned_test_actual,
            test_predictions,
            previous_day=training_consumption[-STEPS_PER_DAY:],
        )

        training_pcs_environment = create_recording_pcs_environment(
            cleaned_train_actual,
            train_predictions,
        )
        if not PRETRAINED_PCS_MODEL_PATH.exists():
            raise FileNotFoundError(
                "The reusable PCS model has not been trained. Run "
                "'python -m for_students.training.train_pcs' once."
            )
        pretrained_pcs = SAC.load(str(PRETRAINED_PCS_MODEL_PATH))
        pcs_model = SAC(
            "MlpPolicy",
            training_pcs_environment,
            seed=seed,
            verbose=0,
            learning_rate=PRETRAINED_AGENT_CONTINUING_LEARNING_RATE,
            buffer_size=10_000,
            learning_starts=PCS_CONTINUING_UPDATE_STEPS,
            batch_size=64,
            train_freq=(1, "step"),
            gradient_steps=1,
            policy_kwargs={"net_arch": [64, 64]},
        )
        pcs_model.policy.load_state_dict(pretrained_pcs.policy.state_dict())
        training_iso_environment = create_competitive_iso_environment(
            cleaned_train_actual,
            train_predictions,
            training_pcs_environment,
            pcs_model,
            generation_cost=generation_cost,
        )
        main_iso_model = create_main_iso_model(
            main_algorithm,
            training_iso_environment,
            seed,
        )
        if not PRETRAINED_COMPETITOR_ISO_MODEL_PATH.exists():
            raise FileNotFoundError(
                "The reusable competitor ISO has not been trained. Run "
                "'python -m for_students.training.train_competitor_iso' "
                "after PCS pretraining finishes."
            )
        competitor_iso_model = SAC(
            "MlpPolicy",
            training_iso_environment,
            seed=seed + COMPETITOR_RANDOM_SEED_OFFSET,
            verbose=0,
            learning_rate=PRETRAINED_AGENT_CONTINUING_LEARNING_RATE,
            buffer_size=10_000,
            learning_starts=32,
            batch_size=64,
            train_freq=(1, "step"),
            gradient_steps=1,
            policy_kwargs={
                "net_arch": list(COMPETITOR_POLICY_LAYERS),
            },
        )
        pretrained_competitor = SAC.load(
            str(PRETRAINED_COMPETITOR_ISO_MODEL_PATH),
        )
        competitor_iso_model.policy.load_state_dict(
            pretrained_competitor.policy.state_dict()
        )

        training_budget = TrainingBudget(
            cpu_seconds=training_seconds,
            wall_seconds=wall_safety_seconds,
        )
        while not training_budget.exhausted:
            # The main ISO receives more updates and a larger network.
            training_iso_environment.opponent_iso_model = (
                competitor_iso_model
            )
            main_iso_model.set_env(
                training_iso_environment,
                force_reset=True,
            )
            main_iso_model.learn(
                total_timesteps=MAIN_ISO_DAYS_PER_TRAINING_PHASE,
                reset_num_timesteps=False,
                callback=StopTrainingAtBudget(training_budget),
            )
            if training_budget.exhausted:
                break

            # The pretrained competitor keeps adapting during the experiment.
            training_iso_environment.opponent_iso_model = main_iso_model
            competitor_iso_model.set_env(
                training_iso_environment,
                force_reset=True,
            )
            competitor_iso_model.learn(
                total_timesteps=COMPETITOR_DAYS_PER_TRAINING_PHASE,
                reset_num_timesteps=False,
                callback=StopTrainingAtBudget(training_budget),
            )
            if training_budget.exhausted:
                break

            # The PCS learns against the current best prices offered by both
            # changing ISO policies during the same alternating cycle.
            current_iso_observation, _ = training_iso_environment.reset()
            main_iso_action, _ = main_iso_model.predict(
                current_iso_observation,
                deterministic=True,
            )
            competitor_iso_action, _ = (
                competitor_iso_model.predict(
                    current_iso_observation,
                    deterministic=True,
                )
            )
            combined_market_action = (
                build_combined_market_action_seen_by_pcs(
                    main_iso_action,
                    competitor_iso_action,
                    training_iso_environment.T,
                )
            )
            training_pcs_environment.set_price_strategy(
                ActionBasedPriceStrategy(
                    combined_market_action,
                    use_asymmetric_pricing=True,
                    steps_per_day=training_iso_environment.T,
                )
            )
            pcs_training_start_index = (
                training_iso_environment._next_start_idx
            )
            training_pcs_environment.set_step_index(
                pcs_training_start_index
            )
            pcs_model.set_env(
                training_pcs_environment,
                force_reset=True,
            )
            pcs_model.learn(
                total_timesteps=PCS_CONTINUING_UPDATE_STEPS,
                reset_num_timesteps=False,
                callback=StopTrainingAtBudget(training_budget),
            )
            training_iso_environment.sync_to_pcs(
                pcs_training_start_index + STEPS_PER_DAY
            )

        training_budget.raise_if_wall_exhausted()
        evaluation_pcs_model = SAC.load(
            str(PRETRAINED_PCS_MODEL_PATH),
        )
        evaluation_competitor_model = SAC.load(
            str(PRETRAINED_COMPETITOR_ISO_MODEL_PATH),
        )
        results = evaluate_iso_policy(
            main_iso_model,
            evaluation_pcs_model,
            evaluation_competitor_model,
            cleaned_test_actual,
            test_predictions,
            generation_cost=generation_cost,
        )
        results["training_budget"] = {
            "cpu_seconds": training_budget.used_cpu_seconds,
            "wall_seconds": training_budget.used_wall_seconds,
        }
        return results
