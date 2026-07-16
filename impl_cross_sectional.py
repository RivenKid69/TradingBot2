# -*- coding: utf-8 -*-
"""
impl_cross_sectional.py
=======================

Поперечные (cross-sectional) преобразования сигналов (Stage A4). Работают на «срезе»
(``pd.Series`` index=symbol) и применяются по каждой дате панели через ``apply_cs``.

Функции:
  * ``rank``       — кросс-секционный ранг ([0,1] или центрированный [-0.5,0.5]);
  * ``zscore``     — стандартизация (mean≈0, std≈1), NaN- и constant-устойчивая;
  * ``winsorize``  — обрезка хвостов (по перцентилям или σ);
  * ``neutralize`` — OLS-остаток сигнала по факторам (sector/beta/size) — убирает
                     случайные экспозиции; категориальные факторы → dummies;
  * ``decay``      — сглаживание по времени (EWMA с half-life) внутри символа.

``run_pipeline`` применяет цепочку трансформов к «сырому» сигналу (MultiIndex Series),
беря факторы для neutralize из факторной панели. Слой ``impl_`` (зависит от
``core_portfolio``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from core_portfolio import PANEL_INDEX_NAMES, SYMBOL_LEVEL, TS_LEVEL

Step = Union[str, Tuple[str, Dict[str, Any]], Dict[str, Any]]


# ---------------------------------------------------------------------------
# Pure cross-section transforms (Series index=symbol -> Series)
# ---------------------------------------------------------------------------
def rank(s: pd.Series, *, pct: bool = True, center: bool = False, ascending: bool = True) -> pd.Series:
    """Кросс-секционный ранг. ``pct`` → [0,1]; ``center`` → вычесть 0.5 → [-0.5,0.5]."""
    r = s.rank(pct=pct, ascending=ascending)
    if pct and center:
        r = r - 0.5
    return r


def zscore(s: pd.Series, *, ddof: int = 0) -> pd.Series:
    """Стандартизация по срезу. Constant/NaN-устойчива (std=0 → нули)."""
    x = pd.to_numeric(s, errors="coerce").astype("float64")
    finite = x[np.isfinite(x)]
    if len(finite) == 0:
        return pd.Series(np.nan, index=s.index)
    m = float(finite.mean())
    sd = float(finite.std(ddof=ddof))
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    if not np.isfinite(sd) or sd == 0.0:
        out[np.isfinite(x)] = 0.0
        return out
    out = (x - m) / sd
    return out


def winsorize(
    s: pd.Series,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
    method: str = "quantile",
    n_std: float = 3.0,
) -> pd.Series:
    """Обрезка хвостов: ``method='quantile'`` (по перцентилям) или ``'std'`` (±n_std)."""
    x = pd.to_numeric(s, errors="coerce").astype("float64")
    finite = x[np.isfinite(x)]
    if len(finite) == 0:
        return x
    if method == "quantile":
        lo = float(np.nanquantile(finite, lower))
        hi = float(np.nanquantile(finite, upper))
    elif method == "std":
        m = float(finite.mean())
        sd = float(finite.std())
        lo, hi = m - n_std * sd, m + n_std * sd
    else:
        raise ValueError(f"winsorize: unknown method {method!r}")
    return x.clip(lo, hi)


def _design_matrix(factors: pd.DataFrame) -> pd.DataFrame:
    """Построить числовую дизайн-матрицу: категориальные → dummies, числовые как есть."""
    if isinstance(factors, pd.Series):
        factors = factors.to_frame()
    parts: List[pd.DataFrame] = []
    for c in factors.columns:
        col = factors[c]
        if col.dtype == object or isinstance(col.dtype, pd.CategoricalDtype):
            dummies = pd.get_dummies(col, prefix=str(c), drop_first=True, dtype="float64")
            parts.append(dummies)
        else:
            parts.append(pd.to_numeric(col, errors="coerce").astype("float64").to_frame(name=c))
    if not parts:
        return pd.DataFrame(index=factors.index)
    return pd.concat(parts, axis=1)


def neutralize(values: pd.Series, factors: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """OLS-остаток ``values`` по ``factors`` (+intercept). Остаток ортогонален факторам.

    При нехватке точек (rows ≤ regressors) возвращает входные значения без изменений.
    """
    design = _design_matrix(pd.DataFrame(factors))
    df = pd.concat([values.rename("__y"), design], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    n_reg = design.shape[1] + 1  # +intercept
    if len(df) <= n_reg:
        return values.copy()
    y = df["__y"].to_numpy(dtype="float64")
    X = df.drop(columns="__y").to_numpy(dtype="float64")
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    out.loc[df.index] = resid
    return out


# ---------------------------------------------------------------------------
# Panel appliers
# ---------------------------------------------------------------------------
def apply_cs(series: pd.Series, func) -> pd.Series:
    """Применить кросс-секционную ``func`` к каждой дате (groupby по ts)."""
    if len(series) == 0:
        return series
    return series.groupby(level=TS_LEVEL, group_keys=False).apply(func)


def decay(series: pd.Series, *, halflife: float = 5.0) -> pd.Series:
    """EWMA-сглаживание по времени внутри каждого символа (half-life)."""
    if len(series) == 0:
        return series
    return series.groupby(level=SYMBOL_LEVEL, group_keys=False).apply(
        lambda g: g.ewm(halflife=halflife).mean()
    )


_CS_FUNCS = {"rank": rank, "zscore": zscore, "winsorize": winsorize}


def _parse_step(step: Step) -> Tuple[str, Dict[str, Any]]:
    if isinstance(step, str):
        return step, {}
    if isinstance(step, tuple):
        name, kw = step
        return name, dict(kw or {})
    if isinstance(step, dict):
        kw = dict(step)
        name = kw.pop("op", None) or kw.pop("name", None)
        if name is None:
            raise ValueError(f"transform step dict must have 'op': {step!r}")
        return name, kw
    raise TypeError(f"unsupported transform step: {step!r}")


def _neutralize_panel(values: pd.Series, factor_panel: Optional[pd.DataFrame], by: Sequence[str]) -> pd.Series:
    by = [c for c in (by or []) if factor_panel is not None and c in factor_panel.columns]
    if not by or factor_panel is None:
        return values
    F = factor_panel.loc[:, by]
    combined = pd.concat([values.rename("__y"), F], axis=1)

    def _f(g: pd.DataFrame) -> pd.Series:
        return neutralize(g["__y"], g.drop(columns="__y"))

    return combined.groupby(level=TS_LEVEL, group_keys=False).apply(_f)


def run_pipeline(
    raw: pd.Series,
    steps: Sequence[Step],
    *,
    factor_panel: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Применить цепочку трансформов к «сырому» сигналу (MultiIndex Series)."""
    out = raw
    for step in steps or []:
        name, kw = _parse_step(step)
        if name == "neutralize":
            out = _neutralize_panel(out, factor_panel, kw.get("by", []))
        elif name == "decay":
            out = decay(out, **kw)
        elif name in _CS_FUNCS:
            func = _CS_FUNCS[name]
            out = apply_cs(out, lambda s, _f=func, _kw=kw: _f(s, **_kw))
        else:
            raise ValueError(f"run_pipeline: unknown transform {name!r}")
    return out


__all__ = [
    "rank",
    "zscore",
    "winsorize",
    "neutralize",
    "apply_cs",
    "decay",
    "run_pipeline",
]
