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
        self._queue: deque[tuple[float, float, float, float]] = deque()

    def queue(
        self,
        *,
        net_worth: float,
        turnover_notional: float,
        fee_total: float,
    ) -> None:
        value = float(net_worth)
        self._queue.append(
            (
                value,
                value,
                float(turnover_notional),
                float(fee_total),
            )
        )

    def reset(self):
        self.calls.clear()
        self._queue.clear()
        return np.zeros(self.env.observation_space.shape, dtype=np.float32), {}

    def set_market_context(self, **_: object) -> None:
        return None

    def step(self, proto: ActionProto):
        self.calls.append(proto)
        obs = np.zeros(self.env.observation_space.shape, dtype=np.float32)
        if not self._queue:
            return obs, 0.0, False, False, {}
        net_worth, cash, turnover_notional, fee_total = self._queue.popleft()
        self.env.state.net_worth = net_worth
        self.env.state.cash = cash
        self.env.state.units = 0.0
        return obs, 0.0, False, False, {
            "executed_notional": turnover_notional,
            "turnover": turnover_notional,
            "fee_total": fee_total,
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


def _as_fraction(value: float) -> float:
    return value / 100.0 if value > 1.0 else value


def test_signal_only_reward_applies_atr_clip() -> None:
    df = pd.DataFrame(
        {
            "open": [100.0, 102.0, 104.0],
            "high": [105.0, 106.0, 108.0],
            "low": [95.0, 100.0, 103.0],
            "close": [100.0, 104.0, 107.0],
            "price": [100.0, 104.0, 107.0],
            "ts_ms": [0, 60_000, 120_000],
        }
    )
    clip_cfg = {"adaptive": True, "atr_window": 2, "hard_cap_pct": 50.0, "multiplier": 1.5}
    env = TradingEnv(df, seed=7, reward_signal_only=True, reward_clip=clip_cfg)
    env.reset()
    assert env._reward_signal_only is True
    env.state.net_worth = 1_000.0
    env.state.cash = 1_000.0
    env.state.units = 0.0

    mediator = _TestMediator(env)
    env._mediator = mediator
    mediator.reset()

    # Manually seed prior signal state to verify ATR-based clipping mechanics.
    env._last_signal_position = 0.5
    env._last_reward_price = df["price"].iloc[1]

    env.state.step_idx = 2
    mediator.queue(net_worth=1_000.0, turnover_notional=0.0, fee_total=0.0)
    _, reward_final, terminated, truncated, info_final = env.step(ActionProto(ActionType.HOLD, 0.0))
    assert not terminated and not truncated
    atr_fraction = info_final["reward_clip_atr_fraction"]
    hard_cap_fraction = _as_fraction(clip_cfg["hard_cap_pct"])
    expected_clip = min(
        hard_cap_fraction,
        clip_cfg["multiplier"] * atr_fraction,
        env.reward_robust_clip_fraction,
    )
    raw_expected = math.log(107.0 / 104.0) * info_final["signal_position_prev"]
    clipped_before_costs = float(np.clip(raw_expected, -expected_clip, expected_clip))
    costs_fraction = float(info_final["reward_costs_fraction"])
    after_costs = clipped_before_costs - costs_fraction
    final_expected = float(np.clip(after_costs, -expected_clip, expected_clip))

    assert info_final["reward_clip_bound_fraction"] == pytest.approx(expected_clip, rel=1e-6)
    assert info_final["reward_clip_atr_fraction"] == pytest.approx(atr_fraction, rel=1e-9)
    assert info_final["reward_raw_fraction"] == pytest.approx(raw_expected, rel=1e-6)
    assert info_final["reward_used_fraction_before_costs"] == pytest.approx(
        clipped_before_costs, rel=1e-6
    )
    assert info_final["reward_used_fraction"] == pytest.approx(after_costs, rel=1e-6)
    assert reward_final == pytest.approx(final_expected, rel=1e-6)


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
    env.turnover_penalty_coef = 0.05

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
        turnover_notional = float(abs(rng.normal()) * denom)
        fee_total = float(rng.uniform(0.0, 0.2))

        mediator.queue(
            net_worth=next_net_worth,
            turnover_notional=turnover_notional,
            fee_total=fee_total,
        )

        _, reward_env, terminated, truncated, info = env.step(ActionProto(ActionType.HOLD, 0.0))
        assert not terminated and not truncated

        prev_equity = max(prev_net_worth, env._reward_equity_floor)
        if not math.isfinite(prev_equity):
            prev_equity = env._reward_equity_floor
        equity = next_net_worth
        if not math.isfinite(equity) or equity <= 0.0:
            equity = max(prev_equity, env._reward_equity_floor)
        ratio_for_log = equity / prev_equity if prev_equity > 0.0 else 1.0
        if not math.isfinite(ratio_for_log) or ratio_for_log <= 0.0:
            ratio_for_log = 1.0
        log_ret = float(
            np.clip(
                math.log(ratio_for_log),
                -env.reward_return_clip,
                env.reward_return_clip,
            )
        )
        ratio_clipped = float(
            np.clip(
                ratio_for_log,
                math.exp(-env.reward_return_clip),
                math.exp(env.reward_return_clip),
            )
        )
        turnover_norm = float(
            np.clip(
                (abs(turnover_notional) / prev_equity) if prev_equity > 0.0 else 0.0,
                0.0,
                env.turnover_norm_cap,
            )
        )
        turnover_penalty = env.turnover_penalty_coef * turnover_norm
        signal_prev = float(info.get("signal_position_prev", 0.0))
        raw_expected = math.log(ratio_for_log) * signal_prev
        bound = float(info["reward_clip_bound_fraction"])
        clipped_before_costs = float(np.clip(raw_expected, -bound, bound))
        after_costs_info = float(info.get("reward_used_fraction", 0.0))
        final_expected = float(np.clip(after_costs_info, -bound, bound))

        assert info["reward_raw_fraction"] == pytest.approx(raw_expected, rel=1e-9, abs=1e-9)
        assert info["reward_used_fraction_before_costs"] == pytest.approx(
            clipped_before_costs, rel=1e-9, abs=1e-9
        )
        assert info["reward"] == pytest.approx(final_expected, rel=1e-9, abs=1e-9)
        assert reward_env == pytest.approx(final_expected, rel=1e-9, abs=1e-9)
        assert info["turnover_notional"] == pytest.approx(abs(turnover_notional), rel=1e-9, abs=1e-9)
        assert info["turnover_norm"] == pytest.approx(turnover_norm, rel=1e-9, abs=1e-9)
        assert info["turnover_penalty"] == pytest.approx(turnover_penalty, rel=1e-9, abs=1e-9)
        assert info["fee_total"] == pytest.approx(fee_total, rel=1e-9, abs=1e-9)
        rewards.append(reward_env)

        prev_net_worth = next_net_worth
        env.state.step_idx = min(idx + 1, len(df) - 1)

    rewards_arr = np.asarray(rewards, dtype=np.float64)

    assert np.max(rewards_arr) <= env.reward_cap + 1e-6
    assert np.percentile(rewards_arr, 95) <= env.reward_cap + 1e-9
    assert np.percentile(rewards_arr, 99) <= env.reward_cap + 1e-9
    assert np.percentile(rewards_arr, 99.9) <= env.reward_cap + 1e-9
    assert float(np.mean(rewards_arr > env.reward_cap)) <= 1e-9

    env.close()

