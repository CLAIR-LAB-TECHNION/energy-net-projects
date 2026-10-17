"""Train the reusable PCS checkpoint used by competitive experiments."""

from pathlib import Path
import tempfile

from stable_baselines3 import SAC

from for_students.competitive_alternating import STEPS_PER_DAY
from for_students.run_competitive_alternating import (
    DEFAULT_RANDOM_SEED,
    PRETRAINED_PCS_MODEL_PATH,
    create_recording_pcs_environment,
    write_clean_consumption_and_previous_day_predictions,
)


# One-time number of half-hour decisions used to build a functional PCS.
PCS_PRETRAINING_TIMESTEPS = 20_000

# Standard learning rate used during the PCS model's one-time buildup.
PCS_PRETRAINING_LEARNING_RATE = 0.0003

# Hidden layers used by the reusable PCS policy.
PCS_PRETRAINING_POLICY_LAYERS = (64, 64)

# Number of training-data days used to report checkpoint reliability.
PCS_VALIDATION_DAYS = 30


def evaluate_pcs_checkpoint(pcs_model, validation_environment):
    """Measure shortages and transaction money without changing the model."""

    grid_reliance_periods = 0
    transaction_money = 0.0

    for _ in range(PCS_VALIDATION_DAYS):
        observation, _ = validation_environment.reset()
        for _ in range(STEPS_PER_DAY):
            action, _ = pcs_model.predict(
                observation,
                deterministic=True,
            )
            observation, _, _, _, information = (
                validation_environment.step(action)
            )
            grid_reliance_periods += int(information["shortage"])
            transaction_money += information["step_money"]

    validation_half_hours = PCS_VALIDATION_DAYS * STEPS_PER_DAY
    return {
        "grid_reliance_periods": grid_reliance_periods,
        "grid_reliance_rate": (
            grid_reliance_periods / validation_half_hours
        ),
        "transaction_money": transaction_money,
    }


def train_and_save_pretrained_pcs():
    """Train the PCS once, validate it, and save its reusable checkpoint."""

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

        training_environment = create_recording_pcs_environment(
            cleaned_training_data,
            training_predictions,
        )
        pcs_model = SAC(
            "MlpPolicy",
            training_environment,
            seed=DEFAULT_RANDOM_SEED,
            verbose=0,
            learning_rate=PCS_PRETRAINING_LEARNING_RATE,
            buffer_size=PCS_PRETRAINING_TIMESTEPS,
            learning_starts=1_000,
            batch_size=256,
            train_freq=(4, "step"),
            gradient_steps=1,
            policy_kwargs={
                "net_arch": list(PCS_PRETRAINING_POLICY_LAYERS),
            },
        )
        pcs_model.learn(total_timesteps=PCS_PRETRAINING_TIMESTEPS)

        validation_environment = create_recording_pcs_environment(
            cleaned_training_data,
            training_predictions,
        )
        validation_results = evaluate_pcs_checkpoint(
            pcs_model,
            validation_environment,
        )

        PRETRAINED_PCS_MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        pcs_model.save(str(PRETRAINED_PCS_MODEL_PATH))

    print(f"Saved PCS model: {PRETRAINED_PCS_MODEL_PATH}")
    print(
        "Validation grid-reliance periods: "
        f"{validation_results['grid_reliance_periods']} "
        f"({validation_results['grid_reliance_rate']:.2%})"
    )
    print(
        "Validation transaction money: "
        f"{validation_results['transaction_money']:.2f}"
    )


if __name__ == "__main__":
    train_and_save_pretrained_pcs()
