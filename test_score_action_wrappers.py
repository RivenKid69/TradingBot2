import numpy as np
import pytest
from gymnasium import Env, spaces

from action_proto import ActionProto, ActionType
from wrappers.action_space import LongOnlyActionWrapper, ScoreActionWrapper


class _DummyScoreEnv(Env):
    metadata: dict[str, object] = {}

    def __init__(self) -> None:
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def step(self, action):  # pragma: no cover - not used in tests
        return self.observation_space.sample(), 0.0, False, False, {}

    def reset(self, *, seed=None, options=None):  # pragma: no cover - not used in tests
        return self.observation_space.sample(), {}


def test_score_wrapper_shapes_and_bounds():
    env = ScoreActionWrapper(_DummyScoreEnv())
    sample = env.action_space.sample()
    assert sample.shape == (1,)
    assert np.all(sample >= 0.0)
    assert np.all(sample <= 1.0)


def test_score_wrapper_rejects_non_finite_actions():
    env = ScoreActionWrapper(_DummyScoreEnv())
    with pytest.raises(ValueError):
        env.action([float("nan")])
    with pytest.raises(ValueError):
        env.action([float("inf")])


def test_long_only_wrapper_clamps_payloads():
    env = LongOnlyActionWrapper(_DummyScoreEnv())

    assert env.action(-0.25) == pytest.approx(0.0)

    array_action = env.action(np.array([-1.0, 0.25, 2.0], dtype=np.float32))
    assert np.all(array_action >= 0.0)
    assert np.all(array_action <= 1.0)

    proto_action = env.action(ActionProto(ActionType.MARKET, volume_frac=-0.7))
    assert isinstance(proto_action, ActionProto)
    assert proto_action.volume_frac == pytest.approx(0.0)
