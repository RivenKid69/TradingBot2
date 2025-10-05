from dataclasses import replace

import numpy as np
from gymnasium import Env, spaces

from action_proto import ActionProto, ActionType
from wrappers.action_space import LongOnlyActionWrapper


class _DummyEnv(Env):
    metadata: dict[str, object] = {}

    def __init__(self):
        self.action_space = spaces.Dict(
            {
                "price_offset_ticks": spaces.Discrete(3),
                "ttl_steps": spaces.Discrete(2),
                "type": spaces.Discrete(4),
                "volume_frac": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            }
        )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def step(self, action):  # pragma: no cover - not used
        return self.observation_space.sample(), 0.0, False, False, {}

    def reset(self, *, seed=None, options=None):  # pragma: no cover - not used
        return self.observation_space.sample(), {}


def test_long_only_wrapper_clamps_mapping_volume():
    env = LongOnlyActionWrapper(_DummyEnv())
    action = {
        "price_offset_ticks": 0,
        "ttl_steps": 0,
        "type": int(ActionType.MARKET),
        "volume_frac": np.array([-0.25], dtype=np.float32),
    }

    wrapped = env.action(action)

    assert isinstance(wrapped, dict)
    vol = wrapped["volume_frac"]
    assert np.all(vol >= 0.0)
    assert vol.dtype == np.float32


def test_long_only_wrapper_clamps_action_proto():
    env = LongOnlyActionWrapper(_DummyEnv())
    action = ActionProto(action_type=ActionType.MARKET, volume_frac=-0.7)

    wrapped = env.action(action)

    assert isinstance(wrapped, ActionProto)
    assert wrapped.volume_frac == 0.0
    # Ensure other fields are untouched
    assert replace(action, volume_frac=0.0) == wrapped
