# -*- coding: utf-8 -*-
"""
xs_risk/options_factors.py
==========================

Options vol-факторные экспозиции (Stage B5) для факторной риск-модели Σ = B F Bᵀ + D —
**риск-вью по vol-факторам** (level/skew/term), дополняющий greeks-нейтральный
конструктор. Опираемся на изменения implied vol (а не цены): обычный риск опционного
портфеля факторизуется по сдвигам поверхности волатильности.

Факторы:
  * **vol_level_beta** — бета изменений IV символа к vol-индексу (VIX/DVOL/равновзвеш. прокси);
  * **skew**           — нагрузка на skew (BYO-скор: put−call IV);
  * **term**           — нагрузка на term-structure (BYO-скор: front−back IV).

Производит ``exposures`` DataFrame (index=symbol, cols=factors) для
``service_risk_model.FactorRiskModel``. ПАКЕТ — ``xs_risk`` (НЕ ``risk``).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from core_portfolio import SYMBOL_LEVEL


def iv_changes_wide_from_panel(panel: pd.DataFrame, *, iv_col: str = "iv") -> pd.DataFrame:
    """Панель → широкие изменения IV (index=ts, cols=symbol)."""
    iv = panel[iv_col].unstack(SYMBOL_LEVEL).sort_index()
    return iv.diff()


def _vol_index_changes(iv_changes_wide: pd.DataFrame, vol_index_symbol: Optional[str]) -> pd.Series:
    if vol_index_symbol and vol_index_symbol in iv_changes_wide.columns:
        return iv_changes_wide[vol_index_symbol].astype("float64")
    return iv_changes_wide.mean(axis=1).astype("float64")


def vol_level_beta(
    iv_changes_wide: pd.DataFrame, *, vol_index_symbol: Optional[str] = None
) -> pd.Series:
    """β изменений IV символа к vol-индексу: cov/var."""
    m = _vol_index_changes(iv_changes_wide, vol_index_symbol)
    var_m = float(np.nanvar(m.to_numpy()))
    out: Dict[str, float] = {}
    for s in iv_changes_wide.columns:
        r = iv_changes_wide[s].astype("float64")
        df = pd.concat([r, m], axis=1).dropna()
        if len(df) < 3 or var_m <= 0:
            out[s] = 1.0
        else:
            cov = float(np.cov(df.iloc[:, 0], df.iloc[:, 1], ddof=0)[0, 1])
            out[s] = cov / var_m
    return pd.Series(out, name="vol_level_beta")


def _standardize(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return ((s - s.mean()) / (sd or 1.0)).fillna(0.0)


def build_options_exposures(
    iv_changes_wide: pd.DataFrame,
    *,
    skews: Optional[Dict[str, float]] = None,
    terms: Optional[Dict[str, float]] = None,
    vol_index_symbol: Optional[str] = None,
) -> pd.DataFrame:
    """Построить vol-факторные экспозиции B (index=symbol, cols=[vol_level_beta, skew?, term?])."""
    symbols = list(iv_changes_wide.columns)
    cols: Dict[str, pd.Series] = {}
    cols["vol_level_beta"] = vol_level_beta(
        iv_changes_wide, vol_index_symbol=vol_index_symbol
    ).reindex(symbols)
    if skews:
        cols["skew"] = _standardize(pd.Series({s: float(skews.get(s, np.nan)) for s in symbols}))
    if terms:
        cols["term"] = _standardize(pd.Series({s: float(terms.get(s, np.nan)) for s in symbols}))
    B = pd.DataFrame(cols, index=symbols)
    return B.astype("float64").fillna(0.0)


def build_options_risk_model(
    iv_changes_wide: pd.DataFrame,
    *,
    skews: Optional[Dict[str, float]] = None,
    terms: Optional[Dict[str, float]] = None,
    vol_index_symbol: Optional[str] = None,
    factor_cov_method: str = "ledoit_wolf",
):
    """Удобный конструктор FactorRiskModel с vol-факторными экспозициями."""
    from service_risk_model import FactorRiskModel

    B = build_options_exposures(
        iv_changes_wide, skews=skews, terms=terms, vol_index_symbol=vol_index_symbol
    )
    return FactorRiskModel(B, factor_cov_method=factor_cov_method)


__all__ = [
    "iv_changes_wide_from_panel",
    "vol_level_beta",
    "build_options_exposures",
    "build_options_risk_model",
]
