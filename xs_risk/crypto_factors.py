# -*- coding: utf-8 -*-
"""
risk/crypto_factors.py
======================

Крипто-факторные экспозиции (Stage B1) для факторной риск-модели Σ = B F Bᵀ + D:

  * **btc_beta** — чувствительность к BTC (рыночный фактор крипты);
  * **size**     — log(mcap) (размер);
  * **sector**   — сектор (L1 / DeFi / Meme / ...) как one-hot dummies.

Производит ``exposures`` DataFrame (index=symbol, cols=factors) для
``service_risk_model.FactorRiskModel``. Плагин на готовый движок.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core_portfolio import SYMBOL_LEVEL


def returns_wide_from_panel(panel: pd.DataFrame, *, price_col: str = "close") -> pd.DataFrame:
    """Панель → широкие доходности (index=ts, cols=symbol)."""
    price = panel[price_col].unstack(SYMBOL_LEVEL).sort_index()
    return price.pct_change()


def btc_beta(returns_wide: pd.DataFrame, *, btc_symbol: str = "BTC") -> pd.Series:
    """β каждого символа к BTC: cov(r_i, r_btc) / var(r_btc)."""
    if btc_symbol not in returns_wide.columns:
        # нет BTC — beta=1 как нейтральная заглушка
        return pd.Series(1.0, index=returns_wide.columns, name="btc_beta")
    b = returns_wide[btc_symbol].astype("float64")
    var_b = float(np.nanvar(b.to_numpy()))
    out = {}
    for s in returns_wide.columns:
        r = returns_wide[s].astype("float64")
        df = pd.concat([r, b], axis=1).dropna()
        if len(df) < 3 or var_b <= 0:
            out[s] = 1.0
        else:
            cov = float(np.cov(df.iloc[:, 0], df.iloc[:, 1], ddof=0)[0, 1])
            out[s] = cov / var_b
    return pd.Series(out, name="btc_beta")


def build_crypto_exposures(
    returns_wide: pd.DataFrame,
    *,
    sectors: Optional[Dict[str, str]] = None,
    mcaps: Optional[Dict[str, float]] = None,
    btc_symbol: str = "BTC",
) -> pd.DataFrame:
    """Построить факторные экспозиции B (index=symbol, cols=[btc_beta, size, sector_*])."""
    symbols = list(returns_wide.columns)
    cols: Dict[str, pd.Series] = {}

    cols["btc_beta"] = btc_beta(returns_wide, btc_symbol=btc_symbol).reindex(symbols)

    if mcaps:
        mc = pd.Series({s: float(mcaps.get(s, np.nan)) for s in symbols}).clip(lower=1e-9)
        size = np.log(mc)
        size = (size - size.mean()) / (size.std(ddof=0) or 1.0)  # стандартизуем
        cols["size"] = size.fillna(0.0)

    B = pd.DataFrame(cols, index=symbols)

    if sectors:
        sec = pd.Series({s: str(sectors.get(s, "OTHER")) for s in symbols}, index=symbols)
        dummies = pd.get_dummies(sec, prefix="sector", drop_first=True, dtype="float64")
        B = pd.concat([B, dummies], axis=1)

    return B.astype("float64").fillna(0.0)


def build_crypto_risk_model(
    returns_wide: pd.DataFrame,
    *,
    sectors: Optional[Dict[str, str]] = None,
    mcaps: Optional[Dict[str, float]] = None,
    btc_symbol: str = "BTC",
    factor_cov_method: str = "ledoit_wolf",
):
    """Удобный конструктор FactorRiskModel с крипто-экспозициями."""
    from service_risk_model import FactorRiskModel

    B = build_crypto_exposures(returns_wide, sectors=sectors, mcaps=mcaps, btc_symbol=btc_symbol)
    return FactorRiskModel(B, factor_cov_method=factor_cov_method)


__all__ = [
    "returns_wide_from_panel",
    "btc_beta",
    "build_crypto_exposures",
    "build_crypto_risk_model",
]
