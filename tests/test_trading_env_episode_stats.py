from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_patchnew import TradingEnv
from winrate_stats import extract_episode_win_payload


class _MediatorStub:
    def __init__(self, env: TradingEnv) -> None:
        self._env = env
        self.calls: list[object] = []
        self.exec = None
        self._step_calls = 0

    def reset(self) -> None:  # pragma: no cover - trivial
        self.calls.clear()
        self._step_calls = 0

    def step(self, proto):  # pragma: no cover - deterministic stub
        state = self._env.state
        self._step_calls += 1
        next_idx = min(state.step_idx + 1, len(self._env.df) - 1)
        state.step_idx = next_idx
        state.net_worth = self._env.initial_cash * (1.0 + 0.05 * self._step_calls)
        state.cash = state.net_worth
        state.units = 1.0
        self.calls.append(proto)
        obs = np.zeros(self._env.observation_space.shape, dtype=np.float32)
        info = {"equity": state.net_worth, "turnover": 0.0}
        terminated = self._step_calls >= 2
        truncated = False
        return obs, 0.0, terminated, truncated, info


def test_trading_env_episode_stats_payload():
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 103.0],
            "close": [100.5, 102.0, 104.0],
            "high": [101.0, 103.0, 104.5],
            "low": [99.5, 100.0, 102.5],
            "price": [100.5, 102.0, 104.0],
            "quote_asset_volume": [1_000.0, 1_000.0, 1_000.0],
        }
    )
    env = TradingEnv(df, seed=0, enable_shocks=False, flash_prob=0.0)
    env._mediator = _MediatorStub(env)

    env.reset()
    assert env._episode_length == 0
    assert env._episode_return == 0.0

    rewards: list[float] = []

    for _ in range(2):
        _, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
        rewards.append(reward)
        assert truncated is False
        if terminated:
            break

    assert terminated is True
    stats = info.get("episode_stats")
    assert stats is not None
    assert stats["length"] == len(rewards)
    assert stats["reward"] == pytest.approx(sum(rewards))

    expected_win = bool(info.get("equity", env.state.net_worth) > env.initial_cash)
    assert stats["win"] is expected_win

    assert env._episode_length == len(rewards)
    assert env._episode_return == pytest.approx(sum(rewards))

    win_flag, length = extract_episode_win_payload(info)
    assert win_flag is expected_win
    assert length == len(rewards)

