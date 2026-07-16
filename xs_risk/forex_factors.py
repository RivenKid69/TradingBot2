# -*- coding: utf-8 -*-
"""
xs_risk/forex_factors.py
========================

Forex-факторные экспозиции (Stage B4) для факторной риск-модели Σ = B F Bᵀ + D.
Зеркало crypto/equity/futures-факторов, но **минимальный набор непрерывных факторов**
(FX-юниверс мал — G10 ~9 пар; не раздуваем one-hot, чтобы оптимизатор/регрессия не
вырождались):

  * **usd_beta**  — бета к доллару (USD-индекс / равновзвеш. прокси корзины пар);
  * **carry**     — нагрузка на carry (дифференциал ставок, BYO-скор);
  * **value**     — value/PPP-нагрузка (BYO-скор).
  * **bloc**      — опц. блок (G10/EM/commodity) one-hot, по умолчанию выключен (мал юниверс).

ПАКЕТ — ``xs_risk`` (НЕ ``risk``).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from core_portfolio import SYMBOL_LEVEL


def returns_wide_from_panel(panel: pd.DataFrame, *, price_col: str = "close") -> pd.DataFrame:
    """Панель → широкие доходности (index=ts, cols=symbol)."""
    price = panel[price_col].unstack(SYMBOL_LEVEL).sort_index()
    return price.pct_change()


def _usd_returns(returns_wide: pd.DataFrame, usd_symbol: Optional[str]) -> pd.Series:
    if usd_symbol and usd_symbol in returns_wide.columns:
        return returns_wide[usd_symbol].astype("float64")
    return returns_wide.mean(axis=1).astype("float64")


def usd_beta(returns_wide: pd.DataFrame, *, usd_symbol: Optional[str] = None) -> pd.Series:
    """β каждой пары к доллару (USD-индекс или равновзвеш. прокси): cov/var."""
    m = _usd_returns(returns_wide, usd_symbol)
    var_m = float(np.nanvar(m.to_numpy()))
    out: Dict[str, float] = {}
    for s in returns_wide.columns:
        r = returns_wide[s].astype("float64")
        df = pd.concat([r, m], axis=1).dropna()
        if len(df) < 3 or var_m <= 0:
            out[s] = 1.0
        else:
            cov = float(np.cov(df.iloc[:, 0], df.iloc[:, 1], ddof=0)[0, 1])
            out[s] = cov / var_m
    return pd.Series(out, name="usd_beta")


def _standardize(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return ((s - s.mean()) / (sd or 1.0)).fillna(0.0)


def build_forex_exposures(
    returns_wide: pd.DataFrame,
    *,
    carries: Optional[Dict[str, float]] = None,
    values: Optional[Dict[str, float]] = None,
    blocs: Optional[Dict[str, str]] = None,
    usd_symbol: Optional[str] = None,
) -> pd.DataFrame:
    """Построить факторные экспозиции B (index=symbol, cols=[usd_beta, carry?, value?, bloc_*?])."""
    symbols = list(returns_wide.columns)
    cols: Dict[str, pd.Series] = {}

    cols["usd_beta"] = usd_beta(returns_wide, usd_symbol=usd_symbol).reindex(symbols)

    if carries:
        cols["carry"] = _standardize(pd.Series({s: float(carries.get(s, np.nan)) for s in symbols}))
    if values:
        cols["value"] = _standardize(pd.Series({s: float(values.get(s, np.nan)) for s in symbols}))

    B = pd.DataFrame(cols, index=symbols)

    if blocs:
        bl = pd.Series({s: str(blocs.get(s, "OTHER")) for s in symbols}, index=symbols)
        dummies = pd.get_dummies(bl, prefix="bloc", drop_first=True, dtype="float64")
        B = pd.concat([B, dummies], axis=1)

    return B.astype("float64").fillna(0.0)


def build_forex_risk_model(
    returns_wide: pd.DataFrame,
    *,
    carries: Optional[Dict[str, float]] = None,
    values: Optional[Dict[str, float]] = None,
    blocs: Optional[Dict[str, str]] = None,
    usd_symbol: Optional[str] = None,
    factor_cov_method: str = "ledoit_wolf",
):
    """Удобный конструктор FactorRiskModel с forex-экспозициями (USD-beta/carry/value)."""
    from service_risk_model import FactorRiskModel

    B = build_forex_exposures(
        returns_wide, carries=carries, values=values, blocs=blocs, usd_symbol=usd_symbol,
    )
    return FactorRiskModel(B, factor_cov_method=factor_cov_method)


__all__ = [
    "returns_wide_from_panel",
    "usd_beta",
    "build_forex_exposures",
    "build_forex_risk_model",
]
