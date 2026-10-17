"""Run the five-minute competitive EnergyNet experiment."""

from for_students.diagnostics.terminal_report import print_experiment_report
from for_students.internal.experiment import (
    COMPETITOR_DAYS_PER_TRAINING_PHASE,
    COMPETITOR_POLICY_LAYERS,
    COMPETITOR_RANDOM_SEED_OFFSET,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TRAINING_SECONDS,
    PRETRAINED_COMPETITOR_ISO_MODEL_PATH,
    PRETRAINED_PCS_MODEL_PATH,
    create_competitive_iso_environment,
    create_recording_pcs_environment,
    evaluate_iso_policy,
    run_experiment,
    write_clean_consumption_and_previous_day_predictions,
)


if __name__ == "__main__":
    print_experiment_report(run_experiment())
