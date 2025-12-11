import numpy as np
import pytest
pytest.importorskip("torch")
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


def test_long_only_wrapper_maps_payloads():
    """Test that LongOnlyActionWrapper maps [-1, 1] to [0, 1].

    CRITICAL FIX (2025-11-25): The wrapper now MAPS values, not CLAMPS them.
    Mapping formula: output = (input + 1) / 2

    This allows agents to express full range of positions:
    - -1.0 -> 0.0 (full exit)
    - 0.0 -> 0.5 (50% position)
    - 1.0 -> 1.0 (full position)
    """
    env = LongOnlyActionWrapper(_DummyScoreEnv())

    # Test mapping: -0.25 -> (-0.25 + 1) / 2 = 0.375
    assert env.action(-0.25) == pytest.approx(0.375)

    # Test array mapping and output bounds
    array_action = env.action(np.array([-1.0, 0.25, 2.0], dtype=np.float32))
    assert np.all(array_action >= 0.0)
    assert np.all(array_action <= 1.0)
    # -1.0 -> 0.0, 0.25 -> 0.625, 2.0 -> 1.5 -> clipped to 1.0
    np.testing.assert_allclose(array_action, [0.0, 0.625, 1.0], atol=1e-6)

    # Test ActionProto mapping: -0.7 -> (-0.7 + 1) / 2 = 0.15
    proto_action = env.action(ActionProto(ActionType.MARKET, volume_frac=-0.7))
    assert isinstance(proto_action, ActionProto)
    assert proto_action.volume_frac == pytest.approx(0.15)
