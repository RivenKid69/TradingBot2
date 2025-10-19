"""Utilities for computing reliable win-rate statistics with confidence intervals.

The helpers in this module intentionally avoid any heavy dependencies so that
they can be reused from both Python and Cython code.  The implementation relies
only on :mod:`math` and :mod:`numpy` which are already part of the project
stack.  Confidence intervals are available via both the Wilson score interval
and the exact Clopper–Pearson interval, giving a robust envelope for
proportions observed in reinforcement-learning experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(slots=True)
class WinRateStats:
    """Summary of win-rate statistics for a batch of episodes."""

    total_episodes: int
    total_wins: int
    win_rate: float
    wilson_low: float
    wilson_high: float
    clopper_pearson_low: float
    clopper_pearson_high: float
    steps_to_win_mean: float
    steps_to_win_median: float
    steps_to_win_min: float
    steps_to_win_max: float


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _wilson_interval(successes: int, total: int, confidence_level: float) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")

    successes = int(successes)
    total = int(total)
    if successes < 0:
        successes = 0
    if successes > total:
        successes = total

    alpha = 1.0 - float(confidence_level)
    alpha = float(min(max(alpha, 0.0), 1.0))
    if alpha <= 0.0:
        alpha = 1e-12

    # Normal approximation z-score.
    from statistics import NormalDist

    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z2 = z * z

    p_hat = successes / total
    denom = 1.0 + z2 / total
    centre = p_hat + z2 / (2.0 * total)
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * total)) / total)
    low = (centre - margin) / denom
    high = (centre + margin) / denom

    return float(max(low, 0.0)), float(min(high, 1.0))


def _clopper_pearson_interval(successes: int, total: int, confidence_level: float) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")

    successes = int(successes)
    total = int(total)
    if successes < 0:
        successes = 0
    if successes > total:
        successes = total

    alpha = 1.0 - float(confidence_level)
    alpha = float(min(max(alpha, 0.0), 1.0))
    if alpha <= 0.0:
        alpha = 1e-12

    def _binom_cdf(k: int, n: int, p: float) -> float:
        if k < 0:
            return 0.0
        if k >= n:
            return 1.0
        if p <= 0.0:
            return 1.0
        if p >= 1.0:
            return 0.0

        log1m_p = math.log1p(-p)
        log_p = math.log(p)
        lgamma_n1 = math.lgamma(n + 1)
        max_log = -math.inf
        sum_exp = 0.0
        for i in range(0, k + 1):
            log_term = (
                lgamma_n1
                - math.lgamma(i + 1)
                - math.lgamma(n - i + 1)
                + i * log_p
                + (n - i) * log1m_p
            )
            if log_term > max_log:
                if max_log == -math.inf:
                    sum_exp = 1.0
                else:
                    sum_exp = math.exp(max_log - log_term) * sum_exp + 1.0
                max_log = log_term
            else:
                sum_exp += math.exp(log_term - max_log)

        if not math.isfinite(max_log):
            return 0.0
        log_sum = max_log + math.log(sum_exp)
        cdf = math.exp(log_sum)
        if not math.isfinite(cdf):
            return 1.0
        return float(min(max(cdf, 0.0), 1.0))

    def _binom_sf(k: int, n: int, p: float) -> float:
        complement = 1.0 - _binom_cdf(k, n, p)
        if complement < 0.0:
            return 0.0
        if complement > 1.0:
            return 1.0
        return complement

    if successes == 0:
        lower = 0.0
    else:
        target = alpha / 2.0
        lo = 0.0
        hi = 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            tail = _binom_sf(successes - 1, total, mid)
            if abs(hi - lo) < 1e-12:
                break
            if tail > target:
                hi = mid
            else:
                lo = mid
        lower = hi

    if successes == total:
        upper = 1.0
    else:
        target = alpha / 2.0
        lo = 0.0
        hi = 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            cdf = _binom_cdf(successes, total, mid)
            if abs(hi - lo) < 1e-12:
                break
            if cdf > target:
                lo = mid
            else:
                hi = mid
        upper = hi

    return float(max(lower, 0.0)), float(min(upper, 1.0))


def compute_win_rate_stats(
    wins: Sequence[bool],
    episode_lengths: Sequence[int],
    *,
    confidence_level: float = 0.95,
) -> WinRateStats | None:
    """Compute win-rate statistics for aligned win flags and episode lengths.

    Parameters
    ----------
    wins:
        Iterable of booleans indicating whether each episode ended in a win.
    episode_lengths:
        Iterable of the number of environment steps taken in each episode.
    confidence_level:
        Confidence level for Wilson and Clopper–Pearson intervals.
    """

    if len(wins) == 0 or len(episode_lengths) == 0:
        return None
    if len(wins) != len(episode_lengths):
        raise ValueError("'wins' and 'episode_lengths' must have identical length")

    wins_arr = np.asarray(wins, dtype=bool)
    lengths_arr = np.asarray(episode_lengths, dtype=float)

    mask = np.isfinite(lengths_arr)
    if not np.all(mask):
        wins_arr = wins_arr[mask]
        lengths_arr = lengths_arr[mask]

    total = int(wins_arr.size)
    if total == 0:
        return None

    total_wins = int(wins_arr.sum())
    win_rate = _safe_divide(total_wins, total)

    wilson_low, wilson_high = _wilson_interval(total_wins, total, confidence_level)
    cp_low, cp_high = _clopper_pearson_interval(total_wins, total, confidence_level)

    win_lengths = lengths_arr[wins_arr]
    if win_lengths.size > 0:
        steps_mean = float(np.mean(win_lengths))
        steps_median = float(np.median(win_lengths))
        steps_min = float(np.min(win_lengths))
        steps_max = float(np.max(win_lengths))
    else:
        steps_mean = steps_median = steps_min = steps_max = float("nan")

    return WinRateStats(
        total_episodes=total,
        total_wins=total_wins,
        win_rate=float(win_rate),
        wilson_low=wilson_low,
        wilson_high=wilson_high,
        clopper_pearson_low=cp_low,
        clopper_pearson_high=cp_high,
        steps_to_win_mean=steps_mean,
        steps_to_win_median=steps_median,
        steps_to_win_min=steps_min,
        steps_to_win_max=steps_max,
    )


class WinRateAccumulator:
    """Incrementally collect win outcomes and episode lengths."""

    def __init__(self, *, confidence_level: float = 0.95) -> None:
        self._wins: list[bool] = []
        self._lengths: list[int] = []
        self._confidence_level = float(confidence_level)

    def add_episode(self, win: bool, length: int | float | None) -> None:
        if length is None:
            return
        try:
            steps = int(length)
        except (TypeError, ValueError):
            return
        if steps < 0:
            return
        self._wins.append(bool(win))
        self._lengths.append(steps)

    def summary(self) -> WinRateStats | None:
        return compute_win_rate_stats(
            self._wins,
            self._lengths,
            confidence_level=self._confidence_level,
        )

    def reset(self) -> None:
        self._wins.clear()
        self._lengths.clear()


def extract_episode_win_payload(info: Mapping[str, object] | None) -> tuple[bool | None, int | None]:
    """Extract win flag and length from a typical Gym episode info mapping."""

    if not info:
        return None, None

    candidate: Mapping[str, object] | None = None
    if "episode" in info and isinstance(info["episode"], Mapping):
        candidate = info["episode"]  # Monitor / VecMonitor
    elif "episode_info" in info and isinstance(info["episode_info"], Mapping):
        candidate = info["episode_info"]
    elif "episode_stats" in info and isinstance(info["episode_stats"], Mapping):
        candidate = info["episode_stats"]

    if candidate is None:
        return None, None

    reward_value = candidate.get("reward")
    if reward_value is None:
        reward_value = candidate.get("r")

    length_value = candidate.get("length")
    if length_value is None:
        length_value = candidate.get("l")
    if length_value is None:
        length_value = candidate.get("steps")

    win_flag = candidate.get("win")
    if win_flag is None:
        win_flag = candidate.get("is_success")

    win_bool: bool | None
    if isinstance(win_flag, (bool, np.bool_, int, float)):
        win_bool = bool(win_flag)
    elif reward_value is not None:
        try:
            win_bool = float(reward_value) > 0.0
        except (TypeError, ValueError):
            win_bool = None
    else:
        win_bool = None

    length_int: int | None
    if length_value is None:
        length_int = None
    else:
        try:
            length_int = int(length_value)
        except (TypeError, ValueError):
            length_int = None

    return win_bool, length_int

