"""Consistency check between Python BAR reward and Cython semantics."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict

import numpy as np
import pandas as pd

from action_proto import ActionProto, ActionType
from trading_patchnew import TradingEnv


@dataclass
class _QueuedStep:
    """Container for deterministic mediator responses used in simulations."""

    net_worth: float
    cash: float
    units: float
    fee_total: float
    turnover: float


class _QueueMediator:
    """Simple mediator that replays queued responses for the environment."""

    def __init__(self, env: TradingEnv) -> None:
        self.env = env
        self.calls: list[ActionProto] = []
        self._queue: Deque[_QueuedStep] = deque()

    def queue_step(
        self,
        *,
        net_worth: float,
        cash: float,
        units: float,
        fee_total: float,
        turnover: float,
    ) -> None:
        self._queue.append(
            _QueuedStep(
                net_worth=float(net_worth),
                cash=float(cash),
                units=float(units),
                fee_total=float(fee_total),
                turnover=float(turnover),
            )
        )

    def step(self, proto: ActionProto):  # pragma: no cover - exercised indirectly
        self.calls.append(proto)
        obs = np.zeros(self.env.observation_space.shape, dtype=np.float32)
        if not self._queue:
            return obs, 0.0, False, False, {}

        item = self._queue.popleft()
        self.env.state.net_worth = item.net_worth
        self.env.state.cash = item.cash
        self.env.state.units = item.units
        info = {
            "executed_notional": item.turnover,
            "turnover": item.turnover,
            "fee_total": item.fee_total,
        }
        return obs, 0.0, False, False, info

    def reset(self):  # pragma: no cover - trivial
        self.calls.clear()
        self._queue.clear()
        return np.zeros(self.env.observation_space.shape, dtype=np.float32), {}

    def set_market_context(self, **_: object) -> None:  # pragma: no cover - noop
        return None


def _make_dummy_frame(length: int) -> pd.DataFrame:
    ts = np.arange(length, dtype=np.int64)
    ones = np.ones(length, dtype=np.float64)
    return pd.DataFrame(
        {
            "open": ones,
            "high": ones,
            "low": ones,
            "close": ones,
            "price": ones,
            "quote_asset_volume": ones,
            "ts_ms": ts,
        }
    )


def _sample_ratio(rng: np.random.Generator) -> float:
    magnitude = 10.0 ** rng.uniform(-6.0, 6.0)
    if rng.random() < 0.5:
        ratio = magnitude
    else:
        ratio = 1.0 / (magnitude + 1e-16)
    if rng.random() < 0.15:
        ratio *= -1.0
    return float(ratio)


def simulate_bar_reward_path(
    *, num_steps: int = 4096, seed: int = 2024
) -> Dict[str, np.ndarray | float]:
    """Generate a random BAR trajectory and capture reward diagnostics."""

    df = _make_dummy_frame(max(1, num_steps))
    env = TradingEnv(df, seed=seed)
    try:
        env.reset()
        mediator = _QueueMediator(env)
        env._mediator = mediator
        mediator.reset()

        env.turnover_penalty_coef = 0.05

        rng = np.random.default_rng(seed)
        rewards_env: list[float] = []
        fees_collected: list[float] = []
        turnover_samples: list[float] = []
        prev_equity_samples: list[float] = []
        ratios_used: list[float] = []
        info_ratio_raw: list[float] = []
        info_ratio_clipped: list[float] = []

        prev_net_worth = float(env.state.net_worth)

        for step_idx in range(num_steps):
            env.state.step_idx = min(step_idx, len(df) - 1)
            if rng.random() < 0.07:
                prev_net_worth = rng.uniform(-1e-4, 1e-4)

            env.state.net_worth = prev_net_worth
            env.state.cash = prev_net_worth
            env.state.units = 0.0

            prev_equity_value = prev_net_worth if math.isfinite(prev_net_worth) else env._reward_equity_floor
            prev_equity_value = max(prev_equity_value, env._reward_equity_floor)
            denom = prev_equity_value
            ratio = _sample_ratio(rng)
            next_net_worth = ratio * denom
            turnover = float(abs(rng.normal()) * denom)
            fee_total = float(rng.uniform(0.0, 0.2))

            mediator.queue_step(
                net_worth=next_net_worth,
                cash=next_net_worth,
                units=0.0,
                fee_total=fee_total,
                turnover=turnover,
            )

            _, reward_env, terminated, truncated, info = env.step(
                ActionProto(ActionType.HOLD, 0.0)
            )

            if terminated or truncated:
                break

            rewards_env.append(float(reward_env))
            fees_collected.append(fee_total)
            turnover_samples.append(abs(turnover))
            prev_equity_samples.append(prev_equity_value)
            ratios_used.append(ratio)
            info_ratio_raw.append(float(info.get("ratio_raw", 0.0)))
            info_ratio_clipped.append(float(info.get("ratio_clipped", 0.0)))

            prev_net_worth = next_net_worth
            env.state.step_idx = min(step_idx + 1, len(df) - 1)

        rewards_env_arr = np.asarray(rewards_env, dtype=np.float64)
        fees_arr = np.asarray(fees_collected, dtype=np.float64)
        turnover_arr = np.asarray(turnover_samples, dtype=np.float64)
        prev_equity_arr = np.asarray(prev_equity_samples, dtype=np.float64)
        ratios_arr = np.asarray(ratios_used, dtype=np.float64)
        info_ratio_raw_arr = np.asarray(info_ratio_raw, dtype=np.float64)
        info_ratio_clipped_arr = np.asarray(info_ratio_clipped, dtype=np.float64)

        return {
            "rewards_env": rewards_env_arr,
            "ratio_samples": ratios_arr,
            "ratio_raw": info_ratio_raw_arr,
            "ratio_clipped": info_ratio_clipped_arr,
            "fees": fees_arr,
            "turnover": turnover_arr,
            "prev_equity": prev_equity_arr,
            "reward_return_clip": float(env.reward_return_clip),
            "turnover_norm_cap": float(env.turnover_norm_cap),
            "reward_cap": float(env.reward_cap),
            "turnover_penalty_coef": float(env.turnover_penalty_coef),
        }
    finally:
        env.close()


def check_bar_reward_consistency(
    *, num_steps: int = 4096, seed: int = 2024, atol: float = 1e-9, rtol: float = 1e-9
) -> Dict[str, np.ndarray | float]:
    """Validate Python BAR reward against the reference Cython semantics."""

    results = simulate_bar_reward_path(num_steps=num_steps, seed=seed)

    rewards_env = np.asarray(results["rewards_env"], dtype=np.float64)
    ratio_samples = np.asarray(results["ratio_samples"], dtype=np.float64)
    ratio_raw = np.asarray(results["ratio_raw"], dtype=np.float64)
    ratio_clipped_info = np.asarray(results["ratio_clipped"], dtype=np.float64)
    fees = np.asarray(results["fees"], dtype=np.float64)
    turnover = np.asarray(results["turnover"], dtype=np.float64)
    prev_equity = np.asarray(results["prev_equity"], dtype=np.float64)

    reward_return_clip = float(results.get("reward_return_clip", 10.0))
    turnover_norm_cap = float(results.get("turnover_norm_cap", 1.0))
    reward_cap = float(results.get("reward_cap", 10.0))
    turnover_penalty_coef = float(results.get("turnover_penalty_coef", 0.0))

    if ratio_samples.shape != ratio_raw.shape:
        raise AssertionError("ratio_raw info shape mismatch")

    ratio_processed = ratio_samples.copy()
    invalid_mask = ~np.isfinite(ratio_processed) | (ratio_processed <= 0.0)
    ratio_processed[invalid_mask] = 1.0

    ratio_clipped_ref = np.clip(
        ratio_processed,
        math.exp(-reward_return_clip),
        math.exp(reward_return_clip),
    )

    log_ret = np.clip(
        np.log(ratio_processed),
        -reward_return_clip,
        reward_return_clip,
    )
    turnover_norm = np.clip(
        np.divide(
            turnover,
            prev_equity,
            out=np.zeros_like(turnover),
            where=prev_equity > 0.0,
        ),
        0.0,
        turnover_norm_cap,
    )

    rewards_ref = np.clip(
        log_ret - fees - turnover_penalty_coef * turnover_norm,
        -reward_cap,
        reward_cap,
    )

    if not np.allclose(ratio_raw, ratio_processed, rtol=rtol, atol=atol):
        raise AssertionError("Environment reported ratio_raw inconsistent with processed ratios")

    if not np.allclose(ratio_clipped_info, ratio_clipped_ref, rtol=rtol, atol=atol):
        raise AssertionError("Environment reported ratio_clipped inconsistent with reference")

    if not np.allclose(rewards_env, rewards_ref, rtol=rtol, atol=atol):
        raise AssertionError("Environment reward diverges from reference computation")

    max_allowed = reward_cap + 1e-6
    if float(np.max(rewards_env, initial=-math.inf)) > max_allowed:
        raise AssertionError("Reward exceeds configured cap")

    percentiles = {
        "p95": float(np.percentile(rewards_env, 95)),
        "p99": float(np.percentile(rewards_env, 99)),
        "p99_9": float(np.percentile(rewards_env, 99.9)),
    }

    for key, value in percentiles.items():
        if value > reward_cap + 1e-9:
            raise AssertionError(f"{key} percentile {value:.6f} exceeds reward cap {reward_cap:.2f}")

    frac_gt_cap = float(np.mean(rewards_env > reward_cap))
    if frac_gt_cap > 1e-9:
        raise AssertionError("Non-zero fraction of rewards above configured cap")

    results["rewards_ref"] = rewards_ref
    results["ratio_clipped_ref"] = ratio_clipped_ref
    results["percentiles"] = percentiles
    results["frac_gt_cap"] = frac_gt_cap
    results["reward_cap"] = reward_cap

    return results


def main() -> None:  # pragma: no cover - entry point
    stats = check_bar_reward_consistency()
    frac = stats.get("frac_gt_cap", 0.0)
    reward_cap = float(stats.get("reward_cap", 10.0))
    print(
        "Simulated {n} steps — reward mean={mean:.6f}, max={max_val:.6f}, cap={cap:.2f}, frac_gt_cap={frac:.3e}".format(
            n=len(stats["rewards_env"]),
            mean=float(np.mean(stats["rewards_env"])),
            max_val=float(np.max(stats["rewards_env"])),
            cap=reward_cap,
            frac=frac,
        )
    )
    pct = stats["percentiles"]
    print(
        "Percentiles: p95={p95:.6f}, p99={p99:.6f}, p99.9={p99_9:.6f}".format(
            p95=pct["p95"],
            p99=pct["p99"],
            p99_9=pct["p99_9"],
        )
    )


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()

