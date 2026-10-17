"""Train and run one submitted ISO model outside the evaluator process."""

import importlib.util
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np

from for_students.internal.experiment import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_TRAINING_SECONDS,
    DEFAULT_WALL_SAFETY_SECONDS,
    train_main_iso_policy,
)
from for_students.internal.market_support import DEFAULT_GENERATION_COST


DEFAULT_EVALUATION_TIMEOUT_SECONDS = 60
SUBMISSION_STARTUP_GRACE_SECONDS = 60
MAX_SUBMISSION_BYTES = 100_000


def _load_model_factory(submission_path):
    submission_path = Path(submission_path)
    if submission_path.suffix.lower() != ".py":
        raise ValueError("The submission must be one Python source file.")
    if submission_path.stat().st_size > MAX_SUBMISSION_BYTES:
        raise ValueError(
            f"The submission exceeds the {MAX_SUBMISSION_BYTES}-byte limit."
        )

    specification = importlib.util.spec_from_file_location(
        "student_iso_submission",
        submission_path,
    )
    if specification is None or specification.loader is None:
        raise ValueError(f"Cannot import submission: {submission_path}")

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    create_model = getattr(module, "create_model", None)
    if not callable(create_model):
        raise ValueError(
            "Submission must define "
            "create_model(training_environment, seed)."
        )
    return create_model


def _run_policy_process(
    connection,
    submission_path,
    training_seconds,
    wall_safety_seconds,
    seed,
    generation_cost,
):
    try:
        create_model = _load_model_factory(submission_path)
        policy, training_budget = train_main_iso_policy(
            training_seconds=training_seconds,
            seed=seed,
            generation_cost=generation_cost,
            wall_safety_seconds=wall_safety_seconds,
            main_model_factory=create_model,
        )
        connection.send(("ready", training_budget))
        while True:
            command, payload = connection.recv()
            if command == "close":
                return
            if command != "predict":
                raise ValueError(f"Unknown worker command: {command}")

            prediction = policy.predict(payload, deterministic=True)
            action = (
                prediction[0]
                if isinstance(prediction, tuple)
                else prediction
            )
            connection.send(
                ("action", np.asarray(action, dtype=np.float32))
            )
    except BaseException as error:
        try:
            connection.send(
                (
                    "error",
                    f"{type(error).__name__}: {error}\n"
                    f"{traceback.format_exc()}",
                )
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class IsolatedSubmittedPolicy:
    """Train submitted code once, then proxy frozen prediction requests."""

    def __init__(
        self,
        submission_path,
        training_seconds=DEFAULT_TRAINING_SECONDS,
        wall_safety_seconds=DEFAULT_WALL_SAFETY_SECONDS,
        seed=DEFAULT_RANDOM_SEED,
        generation_cost=DEFAULT_GENERATION_COST,
        startup_timeout=None,
        evaluation_timeout=DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    ):
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe()
        self._connection = parent_connection
        self._process = context.Process(
            target=_run_policy_process,
            args=(
                child_connection,
                str(Path(submission_path).resolve()),
                training_seconds,
                wall_safety_seconds,
                seed,
                generation_cost,
            ),
        )
        self._process.start()
        child_connection.close()
        if startup_timeout is None:
            startup_timeout = (
                wall_safety_seconds + SUBMISSION_STARTUP_GRACE_SECONDS
            )
        status, training_budget = self._receive(
            startup_timeout,
            "Submission training timed out.",
        )
        if status != "ready":
            self.close()
            raise RuntimeError(
                f"Unexpected submission startup response: {status}"
            )
        self.training_budget = training_budget
        self._deadline = time.monotonic() + evaluation_timeout

    def _receive(self, timeout, timeout_message):
        if timeout <= 0 or not self._connection.poll(timeout):
            self.close()
            raise TimeoutError(timeout_message)

        status, payload = self._connection.recv()
        if status == "error":
            self.close()
            raise RuntimeError(f"Submission failed:\n{payload}")
        return status, payload

    def predict(self, observation, deterministic=True):
        del deterministic
        if not self._process.is_alive():
            raise RuntimeError("Submission process exited unexpectedly.")

        self._connection.send(("predict", observation))
        remaining = self._deadline - time.monotonic()
        status, action = self._receive(
            remaining,
            "Submission evaluation timed out.",
        )
        if status != "action":
            self.close()
            raise RuntimeError(
                f"Unexpected response from submission process: {status}"
            )
        return action, None

    def close(self):
        if not hasattr(self, "_process"):
            return
        if self._process.is_alive():
            try:
                self._connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
            self._process.join(timeout=1)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_object):
        self.close()
