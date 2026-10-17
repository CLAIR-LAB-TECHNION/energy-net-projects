"""Train the reusable competitor ISO checkpoint."""

from math import ceil
from pathlib import Path
import tempfile
import time

from stable_baselines3 import SAC

from for_students.run_competitive_alternating import (
    DEFAULT_RANDOM_SEED,
    PRETRAINED_COMPETITOR_ISO_MODEL_PATH,
    PRETRAINED_PCS_MODEL_PATH,
    COMPETITOR_POLICY_LAYERS,
    create_competitive_iso_environment,
    create_recording_pcs_environment,
    write_clean_consumption_and_previous_day_predictions,
)


# Bounded daily market decisions learned by each ISO during its buildup.
COMPETITOR_PRETRAINING_DAYS_PER_MODEL = 5_000

# Daily decisions learned by each ISO before alternating to its opponent.
COMPETITOR_PRETRAINING_DAYS_PER_PHASE = 32

# The long one-time buildup teaches economically meaningful pricing and
# dispatch before the short competitive experiment begins.
COMPETITOR_PRETRAINING_LEARNING_RATE = 0.0003

# Lower entropy discourages the extreme pricing and dispatch exploration that
# is especially costly in this market.
COMPETITOR_PRETRAINING_ENTROPY_COEFFICIENT = 0.01

# Seed offset used to initialize the second ISO differently.
SECOND_ISO_SEED_OFFSET = 1

# Training-data days used to report whether the saved competitor participates.
COMPETITOR_VALIDATION_DAYS = 30

# Minimum activity required before replacing the reusable competitor.
MINIMUM_VALIDATION_RETAIL_ENERGY = 1_000.0

# A functional competitor cannot rely primarily on emergency procurement.
MAXIMUM_VALIDATION_EMERGENCY_ENERGY_SHARE = 0.50

# Number of alternating cycles between terminal progress updates.
PROGRESS_REPORT_INTERVAL_CYCLES = 10


def train_and_save_pretrained_competitor_iso(
    entropy_coefficient=COMPETITOR_PRETRAINING_ENTROPY_COEFFICIENT,
    output_path=PRETRAINED_COMPETITOR_ISO_MODEL_PATH,
):
    """Alternate two ISOs, then save one reproducible competitor."""

    if not PRETRAINED_PCS_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Train the reusable PCS before training the competitor ISO."
        )

    repository_root = Path(__file__).resolve().parents[2]
    source_training_data = (
        repository_root
        / "tests"
        / "gym"
        / "data_for_tests"
        / "synthetic_household_consumption_train.csv"
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        cleaned_training_data = (
            Path(temporary_directory) / "cleaned_training_data.csv"
        )
        training_predictions = (
            Path(temporary_directory) / "training_predictions.csv"
        )
        write_clean_consumption_and_previous_day_predictions(
            source_training_data,
            cleaned_training_data,
            training_predictions,
        )

        pcs_environment = create_recording_pcs_environment(
            cleaned_training_data,
            training_predictions,
        )
        pcs_model = SAC.load(
            str(PRETRAINED_PCS_MODEL_PATH),
            env=pcs_environment,
        )
        competitive_environment = create_competitive_iso_environment(
            cleaned_training_data,
            training_predictions,
            pcs_environment,
            pcs_model,
        )

        first_iso_model = SAC(
            "MlpPolicy",
            competitive_environment,
            seed=DEFAULT_RANDOM_SEED,
            verbose=0,
            learning_rate=COMPETITOR_PRETRAINING_LEARNING_RATE,
            buffer_size=20_000,
            learning_starts=256,
            batch_size=256,
            train_freq=(1, "step"),
            gradient_steps=1,
            ent_coef=entropy_coefficient,
            policy_kwargs={
                "net_arch": list(COMPETITOR_POLICY_LAYERS),
            },
        )
        second_iso_model = SAC(
            "MlpPolicy",
            competitive_environment,
            seed=DEFAULT_RANDOM_SEED + SECOND_ISO_SEED_OFFSET,
            verbose=0,
            learning_rate=COMPETITOR_PRETRAINING_LEARNING_RATE,
            buffer_size=20_000,
            learning_starts=256,
            batch_size=256,
            train_freq=(1, "step"),
            gradient_steps=1,
            ent_coef=entropy_coefficient,
            policy_kwargs={
                "net_arch": list(COMPETITOR_POLICY_LAYERS),
            },
        )

        number_of_alternating_cycles = ceil(
            COMPETITOR_PRETRAINING_DAYS_PER_MODEL
            / COMPETITOR_PRETRAINING_DAYS_PER_PHASE
        )
        pretraining_started = time.monotonic()
        for cycle_index in range(number_of_alternating_cycles):
            competitive_environment.opponent_iso_model = (
                second_iso_model
            )
            first_iso_model.set_env(
                competitive_environment,
                force_reset=True,
            )
            first_iso_model.learn(
                total_timesteps=COMPETITOR_PRETRAINING_DAYS_PER_PHASE,
                reset_num_timesteps=False,
            )

            competitive_environment.opponent_iso_model = first_iso_model
            second_iso_model.set_env(
                competitive_environment,
                force_reset=True,
            )
            second_iso_model.learn(
                total_timesteps=COMPETITOR_PRETRAINING_DAYS_PER_PHASE,
                reset_num_timesteps=False,
            )

            completed_cycles = cycle_index + 1
            if (
                completed_cycles % PROGRESS_REPORT_INTERVAL_CYCLES == 0
                or completed_cycles == number_of_alternating_cycles
            ):
                elapsed_seconds = time.monotonic() - pretraining_started
                print(
                    "Competitor pretraining: "
                    f"{completed_cycles}/{number_of_alternating_cycles} "
                    f"cycles in {elapsed_seconds:.1f} seconds"
                )

        competitive_environment.opponent_iso_model = second_iso_model
        observation, _ = competitive_environment.reset()
        validation_profit = 0.0
        validation_retail_energy = 0.0
        validation_retail_revenue = 0.0
        validation_dispatch_cost = 0.0
        validation_emergency_energy = 0.0
        validation_emergency_cost = 0.0
        validation_grid_reliance_periods = 0
        for _ in range(COMPETITOR_VALIDATION_DAYS):
            action, _ = first_iso_model.predict(
                observation,
                deterministic=True,
            )
            observation, reward, _, _, information = (
                competitive_environment.step(action)
            )
            validation_profit += reward
            validation_retail_energy += information[
                "controlled_iso_retail_energy"
            ]
            validation_retail_revenue += information[
                "controlled_iso_retail_revenue"
            ]
            validation_dispatch_cost += information[
                "controlled_iso_dispatch_cost"
            ]
            validation_emergency_energy += information[
                "controlled_iso_emergency_energy"
            ]
            validation_emergency_cost += information[
                "controlled_iso_emergency_dispatch_cost"
            ]
            validation_grid_reliance_periods += information["shortages"]
            observation, _ = competitive_environment.reset()

        validation_retail_operating_profit = (
            validation_retail_revenue
            - validation_dispatch_cost
            - validation_emergency_cost
        )
        validation_emergency_energy_share = (
            validation_emergency_energy / validation_retail_energy
            if validation_retail_energy > 0
            else 1.0
        )
        print(f"Validation profit: {validation_profit:.2f}")
        print(f"Validation retail energy: {validation_retail_energy:.2f}")
        print(
            "Validation retail operating profit: "
            f"{validation_retail_operating_profit:.2f}"
        )
        print(
            "Validation emergency-energy share: "
            f"{validation_emergency_energy_share:.2%}"
        )
        print(
            "Validation PCS grid-reliance periods: "
            f"{validation_grid_reliance_periods}"
        )
        if (
            validation_retail_energy < MINIMUM_VALIDATION_RETAIL_ENERGY
            or validation_emergency_energy_share
            > MAXIMUM_VALIDATION_EMERGENCY_ENERGY_SHARE
        ):
            raise RuntimeError(
                "Competitor pretraining did not produce a functional market "
                "opponent; the existing checkpoint was not replaced."
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        first_iso_model.save(str(output_path))

    print(f"Saved competitor ISO model: {output_path}")


if __name__ == "__main__":
    train_and_save_pretrained_competitor_iso()
