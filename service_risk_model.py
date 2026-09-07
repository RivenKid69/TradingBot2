# -*- coding: utf-8 -*-
"""
service_risk_model.py
=====================

Факторная риск-модель → ковариация активов Σ (Stage A5). Реализации контракта
``core_portfolio.RiskModel``:

* ``FactorRiskModel`` (BARRA-style, fundamental): даны факторные экспозиции **B**
  (per-symbol loadings); факторные доходности **f_t** оцениваются кросс-секционной
  регрессией доходностей на B каждый период; **F** = ковариация факторных доходностей
  (sample / Ledoit-Wolf / EWMA), **D** = идиосинкратическая дисперсия остатков;
  ``Σ = B F Bᵀ + diag(D)``.
* ``StatRiskModel`` (statistical, baseline): Ledoit-Wolf shrinkage сэмпловой ковариации
  + PCA-разложение на ``n_factors`` статфакторов (для exposures/attribution).

Σ гарантированно симметрична и PSD (shrinkage + eigenvalue-clipping). Экспозиции B
интегрируются с ``services.portfolio_constraints.FactorTiltValidator`` (для A7-ограничений
и attribution).

Вход ``fit`` принимает либо «широкие» доходности (index=ts, columns=symbol), либо
панель/Series с MultiIndex ``(ts_ms, symbol)``. Слой ``service_``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from core_portfolio import SYMBOL_LEVEL, TS_LEVEL

logger = logging.getLogger(__name__)

_RETURN_COL_CANDIDATES = ("return", "ret", "fwd_return", "log_return", "close_ret")


# ---------------------------------------------------------------------------
# Linear-algebra helpers
# ---------------------------------------------------------------------------
def to_wide_returns(returns: Any, *, value_col: Optional[str] = None) -> pd.DataFrame:
    """Привести доходности к «широкому» виду (index=ts, columns=symbol)."""
    if isinstance(returns, pd.Series):
        if isinstance(returns.index, pd.MultiIndex):
            return returns.unstack(SYMBOL_LEVEL).sort_index()
        raise ValueError("Series returns must have MultiIndex (ts_ms, symbol)")
    if isinstance(returns, pd.DataFrame):
        if isinstance(returns.index, pd.MultiIndex):
            col = value_col
            if col is None:
                for c in _RETURN_COL_CANDIDATES:
                    if c in returns.columns:
                        col = c
                        break
                if col is None:
                    col = returns.columns[0]
            return returns[col].unstack(SYMBOL_LEVEL).sort_index()
        return returns.sort_index()
    raise TypeError(f"unsupported returns type: {type(returns)!r}")


def nearest_psd(mat: np.ndarray, *, eps: float = 0.0) -> np.ndarray:
    """Ближайшая симметричная PSD-матрица (clip отрицательных собственных значений)."""
    sym = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, eps, None)
    out = (vecs * vals) @ vecs.T
    return 0.5 * (out + out.T)


def ledoit_wolf_identity(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage сэмпловой ковариации к масштабированной единице.

    ``X`` — T×N (демин по столбцам). Возвращает (Σ_shrunk, δ), Σ всегда PSD.
    """
    X = np.asarray(X, dtype="float64")
    T, N = X.shape
    if T < 2:
        # недостаточно данных — диагональ из дисперсий (или нули)
        d = np.var(X, axis=0, ddof=0) if T > 0 else np.zeros(N)
        return np.diag(d), 1.0
    S = (X.T @ X) / T
    m = float(np.trace(S) / N)  # средняя дисперсия
    F = m * np.eye(N)
    d2 = float(np.sum((S - F) ** 2))  # ||S - mI||_F^2
    if d2 <= 0:
        return S, 0.0
    # b̄^2 = (1/T^2) Σ_t || x_t x_t' - S ||_F^2  (векторизовано)
    norm_sq = np.einsum("tj,tj->t", X, X)  # ||x_t||^2
    quad = np.einsum("tj,jk,tk->t", X, S, X)  # x_t' S x_t
    b_bar2 = float(np.sum(norm_sq**2 - 2.0 * quad) + T * np.sum(S**2)) / (T**2)
    b2 = max(0.0, min(b_bar2, d2))
    delta = b2 / d2
    sigma = delta * F + (1.0 - delta) * S
    return nearest_psd(sigma), float(delta)


def _estimate_factor_cov(
    F_ret: np.ndarray, method: str, *, ewma_halflife: float = 20.0
) -> np.ndarray:
    """Ковариация факторных доходностей: 'sample' | 'ledoit_wolf' | 'ewma'."""
    X = np.asarray(F_ret, dtype="float64")
    X = X[np.all(np.isfinite(X), axis=1)]  # полные строки
    if X.shape[0] == 0:
        p = F_ret.shape[1]
        return np.zeros((p, p))
    mu = X.mean(axis=0)
    Xd = X - mu
    if method == "sample":
        return nearest_psd((Xd.T @ Xd) / Xd.shape[0])
    if method == "ledoit_wolf":
        return ledoit_wolf_identity(Xd)[0]
    if method == "ewma":
        T = Xd.shape[0]
        lam = 0.5 ** (1.0 / float(ewma_halflife))
        w = lam ** np.arange(T - 1, -1, -1)
        w = w / w.sum()
        cov = np.einsum("t,ti,tj->ij", w, Xd, Xd)
        return nearest_psd(cov)
    raise ValueError(f"unknown factor cov method: {method!r}")


# ---------------------------------------------------------------------------
# Factor risk model (fundamental / BARRA-style)
# ---------------------------------------------------------------------------
class FactorRiskModel:
    """Фундаментальная факторная модель: Σ = B F Bᵀ + diag(D)."""

    def __init__(
        self,
        exposures: Union[pd.DataFrame, Mapping[str, Mapping[str, float]]],
        *,
        factor_cov_method: str = "ledoit_wolf",
        ewma_halflife: float = 20.0,
        return_col: Optional[str] = None,
        specific_var_floor: float = 0.0,
    ) -> None:
        self._B_full = self._coerce_exposures(exposures)
        self.factors: List[str] = list(self._B_full.columns)
        self.factor_cov_method = factor_cov_method
        self.ewma_halflife = float(ewma_halflife)
        self.return_col = return_col
        self.specific_var_floor = float(specific_var_floor)

        # интеграция с FactorTiltValidator (для A7-ограничений / attribution)
        self._tilt = self._build_tilt_validator(self._B_full)

        self._symbols: List[str] = []
        self._B: Optional[np.ndarray] = None
        self._F: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None
        self._factor_returns: Optional[pd.DataFrame] = None
        self._fitted = False

    @staticmethod
    def _coerce_exposures(exposures: Any) -> pd.DataFrame:
        if isinstance(exposures, pd.DataFrame):
            return exposures.astype("float64")
        # mapping symbol -> {factor: loading}
        return pd.DataFrame.from_dict(dict(exposures), orient="index").astype("float64")

    @staticmethod
    def _build_tilt_validator(B: pd.DataFrame):
        try:
            from services.portfolio_constraints import FactorTiltValidator

            v = FactorTiltValidator()
            for sym, row in B.iterrows():
                v.set_factor_loadings(str(sym), {str(f): float(x) for f, x in row.items()})
            return v
        except Exception as exc:  # pragma: no cover - degrade gracefully
            logger.debug("FactorTiltValidator unavailable: %s", exc)
            return None

    @property
    def tilt_validator(self):
        """Доступ к заполненному FactorTiltValidator (для A7)."""
        return self._tilt

    # ---- RiskModel contract ----
    def fit(self, returns: Any) -> "FactorRiskModel":
        wide = to_wide_returns(returns, value_col=self.return_col)
        syms = [s for s in wide.columns if s in self._B_full.index]
        if not syms:
            raise ValueError("FactorRiskModel.fit: no overlap between returns and exposures")
        wide = wide[syms]
        B = self._B_full.reindex(syms).astype("float64")
        Bmat = B.to_numpy()
        R = wide.to_numpy(dtype="float64")
        T, N = R.shape
        P = Bmat.shape[1]

        # факторные доходности кросс-секционной регрессией каждого периода
        fr = np.full((T, P), np.nan)
        for t in range(T):
            r = R[t]
            mask = np.isfinite(r)
            if int(mask.sum()) <= P:
                continue
            f, *_ = np.linalg.lstsq(Bmat[mask], r[mask], rcond=None)
            fr[t] = f

        # остатки и идиосинкратическая дисперсия
        U = R - fr @ Bmat.T
        D = np.nanvar(U, axis=0, ddof=0)
        D = np.where(np.isfinite(D), D, 0.0)
        D = np.clip(D, self.specific_var_floor, None)

        Fcov = _estimate_factor_cov(fr, self.factor_cov_method, ewma_halflife=self.ewma_halflife)

        self._symbols = list(syms)
        self._B = Bmat
        self._F = Fcov
        self._D = D
        self._factor_returns = pd.DataFrame(fr, index=wide.index, columns=self.factors)
        self._fitted = True
        return self

    def _check(self) -> None:
        if not self._fitted:
            raise RuntimeError("FactorRiskModel is not fitted; call fit() first")

    def exposures(self, asof_ms: Optional[int] = None) -> pd.DataFrame:
        self._check()
        return pd.DataFrame(self._B, index=self._symbols, columns=self.factors)

    def factor_cov(self, asof_ms: Optional[int] = None) -> pd.DataFrame:
        self._check()
        return pd.DataFrame(self._F, index=self.factors, columns=self.factors)

    def specific_var(self, asof_ms: Optional[int] = None) -> pd.Series:
        self._check()
        return pd.Series(self._D, index=self._symbols, name="specific_var")

    def cov(self, asof_ms: Optional[int] = None) -> pd.DataFrame:
        self._check()
        sigma = self._B @ self._F @ self._B.T + np.diag(self._D)
        sigma = nearest_psd(sigma)
        return pd.DataFrame(sigma, index=self._symbols, columns=self._symbols)

    @property
    def factor_returns(self) -> pd.DataFrame:
        self._check()
        return self._factor_returns


# ---------------------------------------------------------------------------
# Statistical risk model (Ledoit-Wolf + PCA factors) — baseline
# ---------------------------------------------------------------------------
class StatRiskModel:
    """Статистическая риск-модель: LW-shrinkage + PCA-факторы (baseline)."""

    def __init__(
        self,
        *,
        method: str = "ledoit_wolf",  # 'ledoit_wolf' | 'sample'
        n_factors: Optional[int] = None,
        return_col: Optional[str] = None,
        specific_var_floor: float = 0.0,
    ) -> None:
        self.method = method
        self.n_factors = n_factors
        self.return_col = return_col
        self.specific_var_floor = float(specific_var_floor)

        self._symbols: List[str] = []
        self._sigma: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None
        self._F: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, returns: Any) -> "StatRiskModel":
        wide = to_wide_returns(returns, value_col=self.return_col)
        wide = wide.dropna(axis=1, how="all")
        R = np.array(wide.to_numpy(dtype="float64"), copy=True)  # writable
        # заполнить точечные NaN средним по столбцу (демин делает их нейтральными)
        col_mean = np.nanmean(np.where(np.isfinite(R), R, np.nan), axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        inds = np.where(~np.isfinite(R))
        R[inds] = np.take(col_mean, inds[1])
        Xd = R - R.mean(axis=0)
        N = Xd.shape[1]

        if self.method == "sample":
            sigma = nearest_psd((Xd.T @ Xd) / max(1, Xd.shape[0]))
        elif self.method == "ledoit_wolf":
            sigma, _ = ledoit_wolf_identity(Xd)
        else:
            raise ValueError(f"StatRiskModel: unknown method {self.method!r}")

        # PCA-разложение Σ на статфакторы
        k = self.n_factors if self.n_factors is not None else N
        k = max(1, min(int(k), N))
        vals, vecs = np.linalg.eigh(sigma)  # по возрастанию
        idx = np.argsort(vals)[::-1][:k]  # top-k
        lam = np.clip(vals[idx], 0.0, None)
        V = vecs[:, idx]
        B = V * np.sqrt(lam)  # N×k, exposures*sqrt(eig)
        Fk = np.eye(k)  # факторы ортонормированы
        common_diag = np.sum(B**2, axis=1)
        D = np.clip(np.diag(sigma) - common_diag, self.specific_var_floor, None)

        self._symbols = list(wide.columns)
        self._sigma = sigma
        self._B = B
        self._F = Fk
        self._D = D
        self._fitted = True
        return self

    def _check(self) -> None:
        if not self._fitted:
            raise RuntimeError("StatRiskModel is not fitted; call fit() first")

    def exposures(self, asof_ms: Optional[int] = None) -> pd.DataFrame:
        self._check()
        cols = [f"pc{i+1}" for i in range(self._B.shape[1])]
        return pd.DataFrame(self._B, index=self._symbols, columns=cols)

    def factor_cov(self, asof_ms: Optional[int] = None) -> pd.DataFrame:
        self._check()
        cols = [f"pc{i+1}" for i in range(self._F.shape[0])]
        return pd.DataFrame(self._F, index=cols, columns=cols)

    def specific_var(self, asof_ms: Optional[int] = None) -> pd.Series:
        self._check()
        return pd.Series(self._D, index=self._symbols, name="specific_var")

    def cov(self, asof_ms: Optional[int] = None) -> pd.DataFrame:
        self._check()
        sigma = self._B @ self._F @ self._B.T + np.diag(self._D)
        sigma = nearest_psd(sigma)
        return pd.DataFrame(sigma, index=self._symbols, columns=self._symbols)

    @property
    def full_cov(self) -> pd.DataFrame:
        """LW/sample Σ до PCA-аппроксимации (диагностика)."""
        self._check()
        return pd.DataFrame(self._sigma, index=self._symbols, columns=self._symbols)


__all__ = [
    "to_wide_returns",
    "nearest_psd",
    "ledoit_wolf_identity",
    "FactorRiskModel",
    "StatRiskModel",
]
