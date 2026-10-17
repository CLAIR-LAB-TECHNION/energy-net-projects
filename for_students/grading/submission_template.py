"""Copy this file and implement the ISO policy submitted for evaluation."""


def create_policy(observation_space, action_space):
    """Return an object that implements predict(observation, deterministic)."""

    return StudentISOPolicy(observation_space, action_space)


class StudentISOPolicy:
    """Required adapter around any model or algorithm chosen by the student."""

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def predict(self, observation, deterministic=True):
        """Return one finite action in the environment's [0, 1] action space."""

        raise NotImplementedError("Load or implement the submitted ISO policy.")
