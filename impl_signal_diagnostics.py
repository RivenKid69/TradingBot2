# -*- coding: utf-8 -*-
"""
impl_signal_diagnostics.py
==========================

Диагностика качества сигналов (Stage A4): Information Coefficient (IC), IC-decay,
quantile spread, turnover, авто-корреляция. Всё — на панельных Series с MultiIndex
``(ts_ms, symbol)``. Без внешних зависимостей (Spearman = Pearson по рангам).

Эти метрики питают Signal Lab (Pro UI) и отбор сигналов в AlphaModel (A6).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from core_portfolio import SYMBOL_LEVEL, TS_LEVEL


def _xs_corr(a: pd.Series, b: pd.Series, method: str = "spearman") -> float:
    """Кросс-секционная корреляция двух серий (по одной дате)."""
    df = pd.concat([a, b], axis=1)
    df.columns = ["a", "b"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 2:
        return np.nan
    x = df["a"].astype("float64")
    y = df["b"].astype("float64")
    if method == "spearman":
        x = x.rank()
        y = y.rank()
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return np.nan
    return float(np.corrcoef(x.to_numpy(), y.to_numpy())[0, 1])


def _align(signal: pd.Series, fwd: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"sig": signal, "fwd": fwd})
    return df.replace([np.inf, -np.inf], np.nan)


def information_coefficient(
    signal: pd.Series,
    forward_returns: pd.Series,
    *,
    method: str = "spearman",
) -> Dict[str, object]:
    """IC = средняя кросс-секционная корреляция сигнала с forward-доходностью.

    Возвращает: ``ic_series`` (по датам), ``ic_mean``, ``ic_std``, ``ic_ir``
    (=mean/std, «информационное отношение»), ``hit_rate`` (доля IC>0), ``n_periods``.
    """
    df = _align(signal, forward_returns)
    if len(df) == 0:
        return _empty_ic()
    ics = df.groupby(level=TS_LEVEL).apply(lambda g: _xs_corr(g["sig"], g["fwd"], method))
    ics = ics.dropna()
    if len(ics) == 0:
        return _empty_ic()
    mean = float(ics.mean())
    std = float(ics.std(ddof=0))
    return {
        "ic_series": ics,
        "ic_mean": mean,
        "ic_std": std,
        "ic_ir": float(mean / std) if std > 0 else np.nan,
        "hit_rate": float((ics > 0).mean()),
        "n_periods": int(len(ics)),
        "method": method,
    }


def _empty_ic() -> Dict[str, object]:
    return {
        "ic_series": pd.Series(dtype="float64"),
        "ic_mean": np.nan,
        "ic_std": np.nan,
        "ic_ir": np.nan,
        "hit_rate": np.nan,
        "n_periods": 0,
        "method": "spearman",
    }


def ic_decay(
    signal: pd.Series,
    forward_returns_by_horizon: Dict[int, pd.Series],
    *,
    method: str = "spearman",
) -> Dict[int, float]:
    """IC при разных горизонтах: ``{horizon -> ic_mean}``."""
    out: Dict[int, float] = {}
    for h, fwd in forward_returns_by_horizon.items():
        out[int(h)] = float(information_coefficient(signal, fwd, method=method)["ic_mean"])
    return out


def quantile_spread(
    signal: pd.Series,
    forward_returns: pd.Series,
    *,
    n_quantiles: int = 5,
) -> Dict[str, object]:
    """Средняя доходность top-минус-bottom квантиля по сигналу.

    Положительный spread → сигнал предсказывает доходность в нужную сторону.
    """
    df = _align(signal, forward_returns).dropna()
    if len(df) == 0:
        return {"spread": np.nan, "top_mean": np.nan, "bottom_mean": np.nan, "n_periods": 0}

    def _per_ts(g: pd.DataFrame) -> Optional[pd.Series]:
        if len(g) < n_quantiles:
            return None
        try:
            q = pd.qcut(g["sig"].rank(method="first"), n_quantiles, labels=False)
        except ValueError:
            return None
        top = g["fwd"][q == n_quantiles - 1].mean()
        bot = g["fwd"][q == 0].mean()
        return pd.Series({"top": top, "bottom": bot})

    parts = []
    for _ts, g in df.groupby(level=TS_LEVEL):
        r = _per_ts(g)
        if r is not None:
            parts.append(r)
    if not parts:
        return {"spread": np.nan, "top_mean": np.nan, "bottom_mean": np.nan, "n_periods": 0}
    res = pd.DataFrame(parts)
    top_mean = float(res["top"].mean())
    bottom_mean = float(res["bottom"].mean())
    return {
        "spread": top_mean - bottom_mean,
        "top_mean": top_mean,
        "bottom_mean": bottom_mean,
        "n_periods": int(len(res)),
    }


def turnover(signal: pd.Series) -> Dict[str, object]:
    """Оборот сигнала: средняя |Δ| нормированного ранга между соседними датами.

    0 — ранжирование не меняется; ~0.33 — случайное; высокое → быстрый/шумный сигнал.
    """
    s = signal.replace([np.inf, -np.inf], np.nan)
    ranks = s.groupby(level=TS_LEVEL, group_keys=False).apply(lambda g: g.rank(pct=True))
    wide = ranks.unstack(SYMBOL_LEVEL).sort_index()
    if wide.shape[0] < 2:
        return {"turnover_mean": np.nan, "n_periods": 0}
    diffs = wide.diff().abs().mean(axis=1)
    diffs = diffs.dropna()
    if len(diffs) == 0:
        return {"turnover_mean": np.nan, "n_periods": 0}
    return {"turnover_mean": float(diffs.mean()), "n_periods": int(len(diffs))}


def signal_autocorr(signal: pd.Series, *, lag: int = 1) -> float:
    """Средняя по датам кросс-секционная авто-корреляция рангов (rank_t vs rank_{t-lag})."""
    s = signal.replace([np.inf, -np.inf], np.nan)
    ranks = s.groupby(level=TS_LEVEL, group_keys=False).apply(lambda g: g.rank())
    wide = ranks.unstack(SYMBOL_LEVEL).sort_index()
    if wide.shape[0] <= lag:
        return np.nan
    corrs = []
    idx = list(wide.index)
    for i in range(lag, len(idx)):
        a = wide.iloc[i]
        b = wide.iloc[i - lag]
        pair = pd.concat([a, b], axis=1).dropna()
        if len(pair) >= 2 and float(pair.iloc[:, 0].std()) > 0 and float(pair.iloc[:, 1].std()) > 0:
            corrs.append(float(np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1]))
    return float(np.mean(corrs)) if corrs else np.nan


def signal_report(
    signal: pd.Series,
    forward_returns: pd.Series,
    *,
    method: str = "spearman",
    n_quantiles: int = 5,
) -> Dict[str, object]:
    """Сводный отчёт по сигналу (для Signal Lab / отбора сигналов)."""
    ic = information_coefficient(signal, forward_returns, method=method)
    qs = quantile_spread(signal, forward_returns, n_quantiles=n_quantiles)
    to = turnover(signal)
    return {
        "ic_mean": ic["ic_mean"],
        "ic_ir": ic["ic_ir"],
        "ic_hit_rate": ic["hit_rate"],
        "n_periods": ic["n_periods"],
        "quantile_spread": qs["spread"],
        "turnover": to["turnover_mean"],
        "autocorr": signal_autocorr(signal),
    }


__all__ = [
    "information_coefficient",
    "ic_decay",
    "quantile_spread",
    "turnover",
    "signal_autocorr",
    "signal_report",
]
