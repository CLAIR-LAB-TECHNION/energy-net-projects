import numpy as np
import pytest
from gymnasium import spaces

from for_students.internal import market_support
from for_students.internal.submission_worker import IsolatedSubmittedPolicy


def write_submission(tmp_path, body):
    submission = tmp_path / "submission.py"
    submission.write_text(body, encoding="utf-8")
    return submission


def test_submission_runs_outside_evaluator_process(tmp_path):
    original_resale_price = market_support.EXTERNAL_RESALE_PRICE
    submission = write_submission(
        tmp_path,
        """
import numpy as np
from for_students.internal import market_support

market_support.EXTERNAL_RESALE_PRICE = 999.0

def create_policy(observation_space, action_space):
    return Policy(action_space)

class Policy:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, deterministic=True):
        return np.zeros(self.action_space.shape, dtype=np.float32), None
""",
    )
    observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(4,),
        dtype=np.float32,
    )
    action_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(3,),
        dtype=np.float32,
    )

    with IsolatedSubmittedPolicy(
        submission,
        observation_space,
        action_space,
        evaluation_timeout=5,
    ) as policy:
        action, _ = policy.predict(np.zeros(4, dtype=np.float32))

    np.testing.assert_array_equal(action, np.zeros(3, dtype=np.float32))
    assert market_support.EXTERNAL_RESALE_PRICE == original_resale_price


def test_submission_prediction_timeout_fails(tmp_path):
    submission = write_submission(
        tmp_path,
        """
def create_policy(observation_space, action_space):
    return Policy()

class Policy:
    def predict(self, observation, deterministic=True):
        while True:
            pass
""",
    )
    observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(4,),
        dtype=np.float32,
    )
    action_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(3,),
        dtype=np.float32,
    )

    with IsolatedSubmittedPolicy(
        submission,
        observation_space,
        action_space,
        evaluation_timeout=0.2,
    ) as policy:
        with pytest.raises(
            TimeoutError,
            match="Submission evaluation timed out",
        ):
            policy.predict(np.zeros(4, dtype=np.float32))
