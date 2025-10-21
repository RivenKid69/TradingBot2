import math

import numpy as np
import pandas as pd
import pytest

from action_proto import ActionProto, ActionType
from trading_patchnew import TradingEnv


def _frame_for_signal_only() -> pd.DataFrame:
    """Construct a minimal dataframe for signal-only environment tests."""
    steps = 4
    idx = np.arange(steps, dtype=np.int64)
    base = np.linspace(100.0, 101.5, steps)
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base,
            "price": base,
            "quote_asset_volume": np.full(steps, 10.0),
            "ts_ms": idx * 60_000,
        }
    )


def test_signal_only_observation_includes_signal_position() -> None:
    """signal only — expose position to critic: observation tail carries signal_pos."""
    df = _frame_for_signal_only()
    env = TradingEnv(df, seed=0, reward_signal_only=True, enable_shocks=False, flash_prob=0.0)
    obs, info = env.reset()

    assert obs.shape == env.observation_space.shape
    assert obs[-1] == pytest.approx(0.0)
    assert info.get("signal_pos", 0.0) == pytest.approx(0.0)

    first_obs, first_reward, first_term, first_trunc, first_info = env.step(
        ActionProto(ActionType.MARKET, volume_frac=0.75)
    )

    assert first_reward == pytest.approx(0.0)
    assert not first_term
    assert not first_trunc
    assert first_obs.shape == env.observation_space.shape
    assert first_info["signal_pos"] == pytest.approx(0.75)
    first_tail = first_obs[-3:]
    assert first_tail[-1] == pytest.approx(0.75), f"first_tail={first_tail}"

    step_obs, reward, terminated, truncated, step_info = env.step(
        ActionProto(ActionType.HOLD, volume_frac=0.0)
    )

    expected_reward = math.log(df.loc[1, "close"] / df.loc[0, "close"]) * 0.75
    assert reward == pytest.approx(expected_reward)
    assert not terminated
    assert not truncated
    assert step_obs.shape == env.observation_space.shape
    assert step_info["signal_pos"] == pytest.approx(0.75)
    tail = step_obs[-3:]
    assert tail[-1] == pytest.approx(0.75), f"tail={tail}"
    assert step_info["signal_position_prev"] == pytest.approx(0.75)
    assert env._mediator._last_signal_position == pytest.approx(0.75)
