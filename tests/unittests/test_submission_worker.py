import numpy as np
import pytest

from for_students.internal import market_support
from for_students.internal.submission_worker import IsolatedSubmittedPolicy


def write_submission(tmp_path, body):
    submission = tmp_path / "submission.py"
    submission.write_text(body, encoding="utf-8")
    return submission


def test_submission_trains_outside_evaluator_process(tmp_path):
    original_resale_price = market_support.EXTERNAL_RESALE_PRICE
    submission = write_submission(
        tmp_path,
        """
import numpy as np
from for_students.internal import market_support

market_support.EXTERNAL_RESALE_PRICE = 999.0

def create_model(training_environment, seed):
    return Model(training_environment.action_space)

class Model:
    def __init__(self, action_space):
        self.action_space = action_space

    def set_env(self, environment, force_reset=True):
        self.environment = environment

    def learn(self, total_timesteps, reset_num_timesteps=False, callback=None):
        return self

    def predict(self, observation, deterministic=True):
        return np.zeros(self.action_space.shape, dtype=np.float32), None
""",
    )

    with IsolatedSubmittedPolicy(
        submission,
        training_seconds=0,
        startup_timeout=30,
        evaluation_timeout=5,
    ) as policy:
        action, _ = policy.predict(np.zeros(48, dtype=np.float32))

    np.testing.assert_array_equal(action, np.zeros(144, dtype=np.float32))
    assert policy.training_budget["cpu_seconds"] >= 0
    assert market_support.EXTERNAL_RESALE_PRICE == original_resale_price


def test_submission_training_timeout_fails(tmp_path):
    submission = write_submission(
        tmp_path,
        """
def create_model(training_environment, seed):
    while True:
        pass
""",
    )

    with pytest.raises(
        TimeoutError,
        match="Submission training timed out",
    ):
        IsolatedSubmittedPolicy(
            submission,
            training_seconds=0,
            startup_timeout=0.2,
        )


def test_old_pretrained_policy_contract_is_rejected(tmp_path):
    submission = write_submission(
        tmp_path,
        """
def create_policy(observation_space, action_space):
    return object()
""",
    )

    with pytest.raises(RuntimeError, match="create_model"):
        IsolatedSubmittedPolicy(
            submission,
            training_seconds=0,
            startup_timeout=30,
        )
