# -*- coding: utf-8 -*-
"""
service_backtest_validation.py
==============================

Анти-оверфит и «Trust Report» (Stage A9) — слой доверия профи (P0-ценность).

* **Deflated Sharpe Ratio** (Bailey & López de Prado 2014) + Probabilistic Sharpe —
  поправка Sharpe на число испытаний, длину выборки, skew/kurtosis.
* **PBO** (Probability of Backtest Overfitting) через Combinatorial Symmetric
  Cross-Validation (Bailey, Borwein, López de Prado, Zhu 2015).
* **Purged & embargoed K-fold** (López de Prado) — устранение лика на границах.
* **Multiple-testing haircut** Sharpe.
* **IS/OOS degradation**.
* ``trust_report`` — сводный JSON по любому бэктесту (acceptance).

Слой ``service_``. Зависит от ``scipy.stats`` (норм. распределение).
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.stats import norm as _norm

    def _cdf(z: float) -> float:
        return float(_norm.cdf(z))

    def _ppf(p: float) -> float:
        return float(_norm.ppf(p))

except Exception:  # pragma: no cover - scipy есть в окружении

    def _cdf(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def _ppf(p: float) -> float:  # грубое приближение (fallback)
        # Beasley-Springer/Moro не нужен — scipy доступен; оставляем заглушку
        from statistics import NormalDist

        return float(NormalDist().inv_cdf(min(max(p, 1e-9), 1 - 1e-9)))


from core_xs_results import compute_metrics

_EULER = 0.5772156649015329


# ---------------------------------------------------------------------------
# Sharpe / moments
# ---------------------------------------------------------------------------
def _clean(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype="float64")
    return a[np.isfinite(a)]


def _moments(x: np.ndarray) -> Tuple[float, float, float, float]:
    if len(x) == 0:
        return 0.0, 0.0, 0.0, 3.0
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if sd == 0:
        return mu, 0.0, 0.0, 3.0
    z = (x - mu) / sd
    return mu, sd, float(np.mean(z**3)), float(np.mean(z**4))


def sharpe_per_obs(returns: Sequence[float]) -> float:
    mu, sd, _, _ = _moments(_clean(returns))
    return mu / sd if sd > 0 else float("nan")


def _psr_denom(sr: float, skew: float, kurt: float) -> float:
    return math.sqrt(max(1e-12, 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2))


def probabilistic_sharpe_ratio(returns: Sequence[float], sr_benchmark: float = 0.0) -> float:
    """PSR: P(истинный SR > sr_benchmark) с учётом skew/kurtosis и длины выборки."""
    x = _clean(returns)
    T = len(x)
    if T < 2:
        return float("nan")
    mu, sd, skew, kurt = _moments(x)
    if sd == 0:
        return float("nan")
    sr = mu / sd
    z = (sr - sr_benchmark) * math.sqrt(T - 1) / _psr_denom(sr, skew, kurt)
    return _cdf(z)


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Ожидаемый максимум Sharpe под null для N независимых испытаний."""
    if n_trials <= 1 or sr_std <= 0:
        return 0.0
    e = math.e
    return sr_std * (
        (1.0 - _EULER) * _ppf(1.0 - 1.0 / n_trials) + _EULER * _ppf(1.0 - 1.0 / (n_trials * e))
    )


def deflated_sharpe_ratio(
    returns: Sequence[float],
    *,
    n_trials: int = 1,
    trial_sharpes: Optional[Sequence[float]] = None,
    sr_std: Optional[float] = None,
) -> float:
    """DSR = PSR относительно ожидаемого максимума Sharpe под null (поправка на N испытаний)."""
    x = _clean(returns)
    T = len(x)
    if T < 2:
        return float("nan")
    mu, sd, skew, kurt = _moments(x)
    if sd == 0:
        return float("nan")
    sr = mu / sd
    if sr_std is None:
        if trial_sharpes is not None and len(trial_sharpes) > 1:
            sr_std = float(np.std(np.asarray(trial_sharpes, dtype="float64"), ddof=1))
        else:
            sr_std = 1.0 / math.sqrt(T)  # масштаб null-дисперсии оценки SR
    sr0 = expected_max_sharpe(n_trials, sr_std)
    z = (sr - sr0) * math.sqrt(T - 1) / _psr_denom(sr, skew, kurt)
    return _cdf(z)


def sharpe_haircut(observed_sr: float, n_trials: int, sr_std: float) -> float:
    """Multiple-testing haircut: observed SR − ожидаемый максимум под null."""
    return max(0.0, observed_sr - expected_max_sharpe(n_trials, sr_std))


# ---------------------------------------------------------------------------
# PBO via CSCV
# ---------------------------------------------------------------------------
def _col_sharpe(M: np.ndarray) -> np.ndarray:
    mu = M.mean(axis=0)
    sd = M.std(axis=0, ddof=0)
    out = np.where(sd > 0, mu / np.where(sd > 0, sd, 1.0), -np.inf)
    return out


def pbo_cscv(perf_matrix: Any, *, n_splits: int = 8) -> Dict[str, Any]:
    """Probability of Backtest Overfitting через combinatorial symmetric CV.

    ``perf_matrix``: (T_obs × N_strategies) per-period performance кандидатов.
    PBO = доля комбинаций, где выбранная по IS лучшая стратегия попадает ниже медианы
    в OOS (логит λ ≤ 0).
    """
    M = np.asarray(perf_matrix, dtype="float64")
    if M.ndim != 2:
        raise ValueError("perf_matrix must be 2D (T_obs, N_strategies)")
    T, N = M.shape
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even")
    if N < 2 or n_splits < 2 or T < n_splits:
        return {"pbo": float("nan"), "n_combos": 0, "mean_logit": float("nan"), "lambdas": []}

    groups = np.array_split(np.arange(T), n_splits)
    half = n_splits // 2
    lambdas: List[float] = []
    for combo in combinations(range(n_splits), half):
        is_rows = np.concatenate([groups[g] for g in combo])
        oos_rows = np.concatenate([groups[g] for g in range(n_splits) if g not in combo])
        is_sr = _col_sharpe(M[is_rows])
        oos_sr = _col_sharpe(M[oos_rows])
        best = int(np.argmax(is_sr))
        # относительный OOS-ранг выбранной стратегии (1..N) → (0,1)
        order = oos_sr.argsort()  # ascending
        pos = int(np.where(order == best)[0][0]) + 1
        rank = pos / (N + 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        lambdas.append(math.log(rank / (1.0 - rank)))

    lam = np.asarray(lambdas, dtype="float64")
    return {
        "pbo": float(np.mean(lam <= 0.0)),
        "mean_logit": float(lam.mean()),
        "n_combos": int(len(lam)),
        "lambdas": lam.tolist(),
    }


# ---------------------------------------------------------------------------
# Purged & embargoed K-fold
# ---------------------------------------------------------------------------
def purged_kfold_indices(
    n_obs: int,
    n_splits: int,
    *,
    purge: int = 0,
    embargo: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Purged K-fold: train исключает purge до и embargo после каждого test-фолда."""
    folds = np.array_split(np.arange(n_obs), n_splits)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for f in folds:
        t0, t1 = int(f[0]), int(f[-1])
        lo, hi = t0 - purge, t1 + embargo
        train = np.array([i for i in range(n_obs) if i < lo or i > hi], dtype="int64")
        out.append((train, f.astype("int64")))
    return out


# ---------------------------------------------------------------------------
# IS / OOS degradation
# ---------------------------------------------------------------------------
def is_oos_degradation(
    is_returns: Sequence[float],
    oos_returns: Sequence[float],
    *,
    periods_per_year: float = 252.0,
) -> Dict[str, float]:
    is_sr = sharpe_per_obs(is_returns) * math.sqrt(periods_per_year)
    oos_sr = sharpe_per_obs(oos_returns) * math.sqrt(periods_per_year)
    ratio = (
        (oos_sr / is_sr)
        if (is_sr not in (0.0,) and np.isfinite(is_sr) and is_sr != 0)
        else float("nan")
    )
    return {
        "is_sharpe": float(is_sr),
        "oos_sharpe": float(oos_sr),
        "degradation_ratio": float(ratio),
    }


# ---------------------------------------------------------------------------
# Trust report (JSON)
# ---------------------------------------------------------------------------
def _verdict(dsr: float, pbo: Optional[float]) -> str:
    if not np.isfinite(dsr):
        return "insufficient_data"
    if dsr >= 0.95 and (pbo is None or pbo <= 0.1):
        return "strong"
    if dsr >= 0.75 and (pbo is None or pbo <= 0.3):
        return "moderate"
    if dsr >= 0.5:
        return "weak"
    return "likely_overfit"


def trust_report(
    returns: Sequence[float],
    *,
    n_trials: int = 1,
    trial_performance: Any = None,  # (T × N) матрица для PBO
    trial_sharpes: Optional[Sequence[float]] = None,
    periods_per_year: float = 252.0,
    capacity: Optional[Dict[str, Any]] = None,
    cscv_splits: int = 8,
    bootstrap: bool = True,
    bootstrap_n: int = 2000,
) -> Dict[str, Any]:
    """Сводный Trust Report (JSON-сериализуемый) по бэктесту.

    P1 #8: добавлены block-bootstrap CI (Politis–Romano stationary bootstrap) на
    Sharpe/CAGR/maxDD + one-sided p-value, и PBO через CSCV когда передан
    ``trial_performance`` (T×N матрица OOS-путей вариантов)."""
    metrics = compute_metrics(returns, periods_per_year=periods_per_year)
    psr = probabilistic_sharpe_ratio(returns, 0.0)
    dsr = deflated_sharpe_ratio(returns, n_trials=n_trials, trial_sharpes=trial_sharpes)

    report: Dict[str, Any] = {
        "n_obs": int(metrics["n_periods"]),
        "sharpe_annual": metrics["sharpe"],
        "total_return": metrics["total_return"],
        "max_drawdown": metrics["max_drawdown"],
        "probabilistic_sharpe": float(psr),
        "deflated_sharpe": float(dsr),
        "n_trials": int(n_trials),
        "pbo": None,
        "capacity": capacity,
    }
    if trial_performance is not None:
        try:
            report["pbo"] = pbo_cscv(trial_performance, n_splits=cscv_splits)["pbo"]
        except Exception:
            report["pbo"] = None
    if bootstrap:
        try:
            from research.bootstrap import bootstrap_report as _bsr

            report["bootstrap"] = _bsr(
                returns, periods_per_year=periods_per_year, n_boot=int(bootstrap_n)
            )
            sh = report["bootstrap"].get("sharpe", {})
            # the edge is "bootstrap-significant" if the 95% CI for Sharpe excludes 0
            report["sharpe_ci_excludes_zero"] = bool(sh.get("ci_low", 0.0) > 0.0)
            report["sharpe_bootstrap_pvalue"] = sh.get("p_value")
        except Exception:
            report["bootstrap"] = None
    report["verdict"] = _verdict(dsr, report["pbo"])
    return report


__all__ = [
    "sharpe_per_obs",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "sharpe_haircut",
    "pbo_cscv",
    "purged_kfold_indices",
    "is_oos_degradation",
    "trust_report",
]
