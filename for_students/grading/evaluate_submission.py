"""Evaluate one student ISO submission against the fixed held-out market."""

import argparse
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
from stable_baselines3 import SAC

from for_students.competitive_alternating import STEPS_PER_DAY
from for_students.diagnostics import print_experiment_report
from for_students.internal.submission_worker import IsolatedSubmittedPolicy
from for_students.run_competitive_alternating import (
    PRETRAINED_COMPETITOR_ISO_MODEL_PATH,
    PRETRAINED_PCS_MODEL_PATH,
    create_competitive_iso_environment,
    create_recording_pcs_environment,
    evaluate_iso_policy,
    write_clean_consumption_and_previous_day_predictions,
)


# Fixed seed used whenever the evaluator loads a submitted policy.
EVALUATION_SEED = 10_000

# A submitted ISO must earn strictly positive held-out profit.
MINIMUM_PASSING_PROFIT = 0.0


def evaluate_submission(submission_path):
    """Run the submitted policy once and return its metrics and pass result."""

    random.seed(EVALUATION_SEED)
    np.random.seed(EVALUATION_SEED)

    repository_root = Path(__file__).resolve().parents[2]
    data_directory = (
        repository_root / "tests" / "gym" / "data_for_tests"
    )
    training_source = (
        data_directory / "synthetic_household_consumption_train.csv"
    )
    test_source = (
        data_directory / "synthetic_household_consumption_test.csv"
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_directory = Path(temporary_directory)
        cleaned_training_data = (
            temporary_directory / "cleaned_training_data.csv"
        )
        cleaned_test_data = temporary_directory / "cleaned_test_data.csv"
        training_predictions = (
            temporary_directory / "training_predictions.csv"
        )
        test_predictions = temporary_directory / "test_predictions.csv"

        training_consumption = (
            write_clean_consumption_and_previous_day_predictions(
                training_source,
                cleaned_training_data,
                training_predictions,
            )
        )
        write_clean_consumption_and_previous_day_predictions(
            test_source,
            cleaned_test_data,
            test_predictions,
            previous_day=training_consumption[-STEPS_PER_DAY:],
        )

        pcs_environment = create_recording_pcs_environment(
            cleaned_test_data,
            test_predictions,
        )
        pcs_model = SAC.load(
            str(PRETRAINED_PCS_MODEL_PATH),
            env=pcs_environment,
        )
        market_environment = create_competitive_iso_environment(
            cleaned_test_data,
            test_predictions,
            pcs_environment,
            pcs_model,
        )
        competitor_iso_model = SAC.load(
            str(PRETRAINED_COMPETITOR_ISO_MODEL_PATH),
        )
        market_environment.opponent_iso_model = competitor_iso_model

        with IsolatedSubmittedPolicy(
            submission_path,
            market_environment.observation_space,
            market_environment.action_space,
        ) as submitted_policy:
            results = evaluate_iso_policy(
                submitted_policy,
                pcs_model,
                competitor_iso_model,
                cleaned_test_data,
                test_predictions,
            )

    checks = {
        "positive held-out profit": (
            results["main_iso"]["profit"] > MINIMUM_PASSING_PROFIT
        ),
    }
    return results, checks


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one student ISO submission.",
    )
    parser.add_argument(
        "submission",
        type=Path,
        help="Path to submission.py containing create_policy().",
    )
    arguments = parser.parse_args()

    results, checks = evaluate_submission(arguments.submission.resolve())
    print_experiment_report(results)
    print("\nPASS REQUIREMENT")
    print("----------------")
    for requirement, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {requirement}")

    passed = all(checks.values())
    print(f"\nFINAL RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
