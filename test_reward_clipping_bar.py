import math
import os
import sys
import types
from collections import deque

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.getcwd())


if "lob_state_cython" not in sys.modules:
    lob_state_stub = types.ModuleType("lob_state_cython")
    lob_state_stub.N_FEATURES = 1
    sys.modules["lob_state_cython"] = lob_state_stub


if "mediator" not in sys.modules:
    mediator_stub = types.ModuleType("mediator")

    class _Mediator:
        def __init__(self, env):
            self.env = env
            self.calls: list = []
            self.exec = None

        def reset(self):
            self.calls.clear()
            return np.zeros(self.env.observation_space.shape, dtype=np.float32), {}

        def step(self, proto):
            self.calls.append(proto)
            obs = np.zeros(self.env.observation_space.shape, dtype=np.float32)
            return obs, 0.0, False, False, {}

    mediator_stub.Mediator = _Mediator
    sys.modules["mediator"] = mediator_stub


from action_proto import ActionProto, ActionType
from scripts.check_reward_clipping_bar_vs_cython import simulate_bar_reward_path
from trading_patchnew import TradingEnv


class _TestMediator:
    def __init__(self, env: TradingEnv) -> None:
        self.env = env
        self.calls: list[ActionProto] = []
        self._queue: deque[tuple[float, float]] = deque()
        self._penalties: deque[float] = deque()

    def queue(self, *, net_worth: float, penalty: float) -> None:
        self._queue.append((float(net_worth), float(net_worth)))
        self._penalties.append(float(penalty))

    def reset(self):
        self.calls.clear()
        self._queue.clear()
        self._penalties.clear()
        return np.zeros(self.env.observation_space.shape, dtype=np.float32), {}

    def set_market_context(self, **_: object) -> None:
        return None

    def step(self, proto: ActionProto):
        self.calls.append(proto)
        obs = np.zeros(self.env.observation_space.shape, dtype=np.float32)
        if not self._queue:
            return obs, 0.0, False, False, {}
        net_worth, cash = self._queue.popleft()
        penalty = self._penalties.popleft()
        self.env.state.net_worth = net_worth
        self.env.state.cash = cash
        self.env.state.units = 0.0
        return obs, 0.0, False, False, {
            "turnover_penalty": penalty,
            "turnover": 0.0,
        }


def _sample_ratio(rng: np.random.Generator) -> float:
    magnitude = 10.0 ** rng.uniform(-6.0, 6.0)
    ratio = magnitude if rng.random() < 0.5 else 1.0 / (magnitude + 1e-16)
    if rng.random() < 0.15:
        ratio *= -1.0
    return float(ratio)


def _make_frame(n: int) -> pd.DataFrame:
    idx = np.arange(n, dtype=np.int64)
    ones = np.ones(n, dtype=np.float64)
    return pd.DataFrame(
        {
            "open": ones,
            "high": ones,
            "low": ones,
            "close": ones,
            "price": ones,
            "quote_asset_volume": ones,
            "ts_ms": idx,
        }
    )


def test_reward_clip_bar_matches_reference() -> None:
    steps = 512
    rng = np.random.default_rng(1234)
    df = _make_frame(steps)
    env = TradingEnv(df, seed=17)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape

    mediator = _TestMediator(env)
    env._mediator = mediator
    mediator.reset()

    prev_net_worth = float(env.state.net_worth)

    rewards = []

    for idx in range(steps):
        env.state.step_idx = min(idx, len(df) - 1)
        if rng.random() < 0.07:
            prev_net_worth = rng.uniform(-1e-4, 1e-4)

        env.state.net_worth = prev_net_worth
        env.state.cash = prev_net_worth
        env.state.units = 0.0

        denom = max(prev_net_worth, 1e-9)
        ratio_raw = _sample_ratio(rng)
        next_net_worth = ratio_raw * denom
        penalty = float(rng.uniform(0.0, 0.2))

        mediator.queue(net_worth=next_net_worth, penalty=penalty)

        _, reward_env, terminated, truncated, info = env.step(ActionProto(ActionType.HOLD, 0.0))
        assert not terminated and not truncated

        ratio_clipped = float(np.clip(ratio_raw, 1e-4, 10.0))
        reward_ref = math.log(ratio_clipped) - penalty

        assert reward_env == pytest.approx(reward_ref, rel=1e-9, abs=1e-9)
        assert reward_env <= math.log(10.0) + 1e-6
        assert info["ratio_raw"] == pytest.approx(ratio_raw, rel=1e-9, abs=1e-9)
        assert info["ratio_clipped"] == pytest.approx(ratio_clipped, rel=1e-9, abs=1e-9)

        rewards.append(reward_env)

        prev_net_worth = next_net_worth
        env.state.step_idx = min(idx + 1, len(df) - 1)

    rewards_arr = np.asarray(rewards, dtype=np.float64)

    assert np.max(rewards_arr) <= math.log(10.0) + 1e-6
    assert np.percentile(rewards_arr, 95) <= math.log(10.0) + 1e-9
    assert np.percentile(rewards_arr, 99) <= math.log(10.0) + 1e-9
    assert np.percentile(rewards_arr, 99.9) <= math.log(10.0) + 1e-9
    assert float(np.mean(rewards_arr > math.log(10.0))) <= 1e-9

    env.close()

    results = simulate_bar_reward_path(num_steps=256, seed=321)
    ratio_samples = np.asarray(results["ratio_samples"], dtype=np.float64)
    penalties = np.asarray(results["penalties"], dtype=np.float64)
    ref = np.log(np.clip(ratio_samples, 1e-4, 10.0)) - penalties

    assert np.allclose(results["rewards_env"], ref, rtol=1e-9, atol=1e-9)
    assert np.allclose(results["ratio_raw"], ratio_samples, rtol=1e-9, atol=1e-9)
    assert np.allclose(results["ratio_clipped"], np.clip(ratio_samples, 1e-4, 10.0), rtol=1e-9, atol=1e-9)

