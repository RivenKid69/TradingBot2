# -*- coding: utf-8 -*-
"""
research/bootstrap.py
=====================

Block-bootstrap confidence intervals for backtest performance statistics (P1 #8).

A single point estimate of the Sharpe ratio (or CAGR / max-drawdown) hides its
sampling uncertainty. For serially-dependent return series the i.i.d. bootstrap is
invalid; we use the **stationary bootstrap** (Politis & Romano, 1994) — resample
**blocks** of random (geometric) length so the dependence structure is preserved —
and the **circular block bootstrap** (Politis & Romano, 1992) as an alternative.

Outputs per statistic: point estimate, bootstrap mean, standard error, a
two-sided percentile confidence interval, and a one-sided bootstrap p-value
(P[stat ≤ 0]) — i.e. "is the edge statistically distinguishable from zero after
accounting for autocorrelation?".

Reference: Politis & Romano (1994), "The Stationary Bootstrap", JASA 89(428).
Pure-NumPy; no SciPy required.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


def _as_array(returns: Sequence[float]) -> np.ndarray:
    r = np.asarray(list(returns), dtype="float64")
    return r[np.isfinite(r)]


def sharpe(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    if returns.size < 2:
        return 0.0
    sd = float(np.std(returns, ddof=1))
    if sd <= 1e-15:
        return 0.0
    return float(np.mean(returns) / sd * np.sqrt(periods_per_year))


def cagr(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    if returns.size == 0:
        return 0.0
    growth = float(np.prod(1.0 + returns))
    if growth <= 0:
        return -1.0
    years = returns.size / periods_per_year
    return float(growth ** (1.0 / max(years, 1e-9)) - 1.0)


def max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    eq = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(-dd.min())   # positive number = depth


# ---------------------------------------------------------------------------
# Block resampling
# ---------------------------------------------------------------------------
def _stationary_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis–Romano stationary bootstrap indices (geometric block lengths)."""
    p = 1.0 / max(1.0, float(mean_block))
    idx = np.empty(n, dtype="int64")
    i = int(rng.integers(0, n))
    for t in range(n):
        idx[t] = i
        if rng.random() < p:
            i = int(rng.integers(0, n))      # start a new block
        else:
            i = (i + 1) % n                  # continue (circular)
    return idx


def _circular_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Fixed-length circular block bootstrap indices."""
    block = max(1, int(block))
    out: List[int] = []
    while len(out) < n:
        start = int(rng.integers(0, n))
        out.extend((start + k) % n for k in range(block))
    return np.asarray(out[:n], dtype="int64")


def block_bootstrap(
    returns: Sequence[float],
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int = 2000,
    mean_block: Optional[float] = None,
    method: str = "stationary",
    seed: int = 12345,
) -> Dict[str, float]:
    """Bootstrap a scalar ``statistic`` of a return series with block resampling.

    ``mean_block`` defaults to the Politis–White (2004) rule-of-thumb ~ n**(1/3).
    Returns point/mean/se/ci_low/ci_high/p_value/n_boot.
    """
    r = _as_array(returns)
    n = r.size
    point = float(statistic(r)) if n else 0.0
    if n < 4:
        return {"point": point, "mean": point, "se": 0.0, "ci_low": point,
                "ci_high": point, "p_value": float("nan"), "n_boot": 0}
    if mean_block is None:
        mean_block = max(2.0, float(n) ** (1.0 / 3.0))
    rng = np.random.default_rng(seed)
    stats = np.empty(int(n_boot), dtype="float64")
    for b in range(int(n_boot)):
        if method == "circular":
            idx = _circular_indices(n, int(round(mean_block)), rng)
        else:
            idx = _stationary_indices(n, mean_block, rng)
        stats[b] = statistic(r[idx])
    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        return {"point": point, "mean": point, "se": 0.0, "ci_low": point,
                "ci_high": point, "p_value": float("nan"), "n_boot": 0}
    return {
        "point": point,
        "mean": float(np.mean(stats)),
        "se": float(np.std(stats, ddof=1)),
        "ci_low": float(np.quantile(stats, 0.025)),
        "ci_high": float(np.quantile(stats, 0.975)),
        "p_value": float(np.mean(stats <= 0.0)),   # one-sided P[stat ≤ 0]
        "n_boot": int(stats.size),
        "mean_block": float(mean_block),
        "method": method,
    }


def bootstrap_report(
    returns: Sequence[float],
    *,
    periods_per_year: float = 252.0,
    n_boot: int = 2000,
    mean_block: Optional[float] = None,
    method: str = "stationary",
    seed: int = 12345,
) -> Dict[str, Dict[str, float]]:
    """Block-bootstrap CIs for Sharpe, CAGR and max-drawdown of a return series."""
    return {
        "sharpe": block_bootstrap(
            returns, lambda x: sharpe(x, periods_per_year),
            n_boot=n_boot, mean_block=mean_block, method=method, seed=seed),
        "cagr": block_bootstrap(
            returns, lambda x: cagr(x, periods_per_year),
            n_boot=n_boot, mean_block=mean_block, method=method, seed=seed + 1),
        "max_drawdown": block_bootstrap(
            returns, max_drawdown,
            n_boot=n_boot, mean_block=mean_block, method=method, seed=seed + 2),
    }


__all__ = [
    "block_bootstrap", "bootstrap_report",
    "sharpe", "cagr", "max_drawdown",
]
