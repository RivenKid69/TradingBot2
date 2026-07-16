# -*- coding: utf-8 -*-
"""
service_alpha.py
================

Alpha-модели (Stage A6): комбинация панели сигналов в ожидаемую доходность **μ** по
юниверсу. Реализации контракта ``core_portfolio.AlphaModel`` (``fit`` / ``predict``):

* ``EqualWeightAlpha``  — среднее нормированных сигналов (baseline);
* ``ICWeightedAlpha``   — вес сигнала ∝ его (знаковый) Information Coefficient;
* ``RidgeAlpha``        — ridge-регрессия forward-доходностей на сигналы (L2);
* ``GBMAlpha``          — gradient boosting (опц., ленивый импорт sklearn).

``fit`` обучается на переданном окне (purge/rolling организует cross-sectional бэктест
A8 — он подаёт сюда уже очищенный train-срез). ``predict`` принимает срез сигналов на
дату (DataFrame index=symbol, columns=signals) и возвращает μ (Series index=symbol).
``predict_panel`` применяет predict по всем датам. Слой ``service_``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core_portfolio import PANEL_INDEX_NAMES, SYMBOL_LEVEL, TS_LEVEL, Panel
from impl_signal_diagnostics import information_coefficient

logger = logging.getLogger(__name__)

_RETURN_COL_CANDIDATES = ("fwd_return", "return", "ret", "forward_return")


def _as_return_series(forward_returns: Any) -> pd.Series:
    if isinstance(forward_returns, pd.Series):
        return forward_returns
    if isinstance(forward_returns, pd.DataFrame):
        for c in _RETURN_COL_CANDIDATES:
            if c in forward_returns.columns:
                return forward_returns[c]
        return forward_returns.iloc[:, 0]
    raise TypeError(f"unsupported forward_returns type: {type(forward_returns)!r}")


class BaseAlphaModel(ABC):
    """Базовая alpha-модель. Реализуйте ``predict``; ``fit`` опционален."""

    signal_names: List[str] = []

    def fit(self, signals: Panel, forward_returns: Any) -> "BaseAlphaModel":
        self.signal_names = list(signals.columns)
        return self

    @abstractmethod
    def predict(self, signals_t: pd.DataFrame) -> pd.Series:
        """μ на дату: Series index=symbol."""
        raise NotImplementedError

    def predict_panel(self, signals: Panel) -> pd.Series:
        """Применить predict по всем датам → μ-панель (MultiIndex Series)."""
        if len(signals) == 0:
            return pd.Series(dtype="float64", index=signals.index)
        parts: List[pd.Series] = []
        for ts, g in signals.groupby(level=TS_LEVEL):
            cs = g.droplevel(TS_LEVEL)
            mu = self.predict(cs)
            idx = pd.MultiIndex.from_arrays(
                [np.full(len(mu), int(ts), dtype="int64"), np.asarray(mu.index, dtype=object)],
                names=PANEL_INDEX_NAMES,
            )
            parts.append(pd.Series(mu.to_numpy(dtype="float64"), index=idx))
        out = pd.concat(parts).sort_index()
        out.name = "mu"
        return out

    def _align(self, signals_t: pd.DataFrame) -> pd.DataFrame:
        cols = self.signal_names or list(signals_t.columns)
        return signals_t.reindex(columns=cols)


class EqualWeightAlpha(BaseAlphaModel):
    """μ = среднее сигналов (предполагаются нормированными)."""

    def predict(self, signals_t: pd.DataFrame) -> pd.Series:
        X = self._align(signals_t).astype("float64")
        return X.mean(axis=1, skipna=True).rename("mu")


class ICWeightedAlpha(BaseAlphaModel):
    """μ = Σ wᵢ·signalᵢ, где wᵢ ∝ знаковый IC сигнала (анти-сигналы инвертируются)."""

    def __init__(self, *, method: str = "spearman", min_abs_ic: float = 0.0) -> None:
        self.method = method
        self.min_abs_ic = float(min_abs_ic)
        self.weights: Dict[str, float] = {}
        self.ic: Dict[str, float] = {}

    def fit(self, signals: Panel, forward_returns: Any) -> "ICWeightedAlpha":
        self.signal_names = list(signals.columns)
        fwd = _as_return_series(forward_returns)
        ic: Dict[str, float] = {}
        for col in self.signal_names:
            res = information_coefficient(signals[col], fwd, method=self.method)
            v = res["ic_mean"]
            ic[col] = float(v) if (v is not None and np.isfinite(v)) else 0.0
        self.ic = ic
        # обнулить слабые сигналы; нормировать по Σ|IC|
        eff = {k: (v if abs(v) >= self.min_abs_ic else 0.0) for k, v in ic.items()}
        denom = sum(abs(v) for v in eff.values())
        if denom <= 0:
            n = max(1, len(self.signal_names))
            self.weights = {k: 1.0 / n for k in self.signal_names}
        else:
            self.weights = {k: v / denom for k, v in eff.items()}
        return self

    def predict(self, signals_t: pd.DataFrame) -> pd.Series:
        X = self._align(signals_t).astype("float64").fillna(0.0)
        w = pd.Series({c: self.weights.get(c, 0.0) for c in X.columns})
        mu = X.mul(w, axis=1).sum(axis=1)
        return mu.rename("mu")


class RidgeAlpha(BaseAlphaModel):
    """Ridge-регрессия forward-доходностей на сигналы (pooled cross-section)."""

    def __init__(self, *, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.coef_: Optional[np.ndarray] = None
        self._x_mean: Optional[np.ndarray] = None
        self._y_mean: float = 0.0

    def fit(self, signals: Panel, forward_returns: Any) -> "RidgeAlpha":
        self.signal_names = list(signals.columns)
        fwd = _as_return_series(forward_returns)
        df = signals.copy()
        df["__y"] = fwd
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) == 0:
            self.coef_ = np.zeros(len(self.signal_names))
            self._x_mean = np.zeros(len(self.signal_names))
            self._y_mean = 0.0
            return self
        X = df[self.signal_names].to_numpy(dtype="float64")
        y = df["__y"].to_numpy(dtype="float64")
        self._x_mean = X.mean(axis=0)
        self._y_mean = float(y.mean())
        Xc = X - self._x_mean
        yc = y - self._y_mean
        p = Xc.shape[1]
        A = Xc.T @ Xc + self.alpha * np.eye(p)
        self.coef_ = np.linalg.solve(A, Xc.T @ yc)
        return self

    def predict(self, signals_t: pd.DataFrame) -> pd.Series:
        X = self._align(signals_t).astype("float64").fillna(0.0)
        if self.coef_ is None:
            return pd.Series(0.0, index=X.index, name="mu")
        Xc = X.to_numpy(dtype="float64") - self._x_mean
        mu = Xc @ self.coef_ + self._y_mean
        return pd.Series(mu, index=X.index, name="mu")


class GBMAlpha(BaseAlphaModel):
    """Gradient boosting alpha (опционально; требует sklearn)."""

    def __init__(self, **gbm_kwargs: Any) -> None:
        self._kwargs = gbm_kwargs
        self._model = None

    def fit(self, signals: Panel, forward_returns: Any) -> "GBMAlpha":
        # Use sklearn if available; otherwise fall back to the pure-NumPy GBRT so GBM
        # alpha works without sklearn (P2 #22). Same fit/predict API either way.
        try:
            from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
        except Exception:
            from impl_gbrt import GradientBoostingRegressor

        self.signal_names = list(signals.columns)
        fwd = _as_return_series(forward_returns)
        df = signals.copy()
        df["__y"] = fwd
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        self._model = GradientBoostingRegressor(**self._kwargs)
        if len(df):
            self._model.fit(df[self.signal_names].to_numpy(), df["__y"].to_numpy())
        return self

    def predict(self, signals_t: pd.DataFrame) -> pd.Series:
        X = self._align(signals_t).astype("float64").fillna(0.0)
        if self._model is None:
            return pd.Series(0.0, index=X.index, name="mu")
        mu = self._model.predict(X.to_numpy())
        return pd.Series(mu, index=X.index, name="mu")


__all__ = [
    "BaseAlphaModel",
    "EqualWeightAlpha",
    "ICWeightedAlpha",
    "RidgeAlpha",
    "GBMAlpha",
]
