"""Run untrusted submission policies outside the evaluator process."""

import importlib.util
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np


DEFAULT_STARTUP_TIMEOUT_SECONDS = 15
DEFAULT_EVALUATION_TIMEOUT_SECONDS = 60


def _load_policy(submission_path, observation_space, action_space):
    specification = importlib.util.spec_from_file_location(
        "student_iso_submission",
        submission_path,
    )
    if specification is None or specification.loader is None:
        raise ValueError(f"Cannot import submission: {submission_path}")

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    if not hasattr(module, "create_policy"):
        raise ValueError(
            "Submission must define create_policy("
            "observation_space, action_space)."
        )

    policy = module.create_policy(observation_space, action_space)
    if not hasattr(policy, "predict"):
        raise ValueError("create_policy() must return an object with predict().")
    return policy


def _run_policy_process(
    connection,
    submission_path,
    observation_space,
    action_space,
):
    try:
        policy = _load_policy(
            submission_path,
            observation_space,
            action_space,
        )
        connection.send(("ready", None))
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
    """Proxy predict() calls to a spawned submission process."""

    def __init__(
        self,
        submission_path,
        observation_space,
        action_space,
        startup_timeout=DEFAULT_STARTUP_TIMEOUT_SECONDS,
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
                observation_space,
                action_space,
            ),
        )
        self._process.start()
        child_connection.close()
        status, _ = self._receive(
            startup_timeout,
            "Submission initialization timed out.",
        )
        if status != "ready":
            self.close()
            raise RuntimeError(
                f"Unexpected submission startup response: {status}"
            )
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
