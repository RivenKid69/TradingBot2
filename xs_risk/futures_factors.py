# -*- coding: utf-8 -*-
"""
xs_risk/futures_factors.py
==========================

Futures-факторные экспозиции (Stage B3) для факторной риск-модели Σ = B F Bᵀ + D.
Зеркало crypto/equity-факторов; «сектор» для фьючерсов = **класс актива**
(equity-index / rates / energy / metals / ag / FX) — диверсифицированный CTA-портфель
естественно факторизуется по asset-class.

Факторы:
  * **market_beta**  — бета к диверсифицированному basket'у (равновзвеш. прокси) или индексу;
  * **vol**          — реализованная волатильность (риск-фактор; CTA сайзит обратно к vol);
  * **asset_class**  — класс актива как one-hot dummies (equity_index/rates/energy/...).

Производит ``exposures`` DataFrame (index=symbol, cols=factors) для
``service_risk_model.FactorRiskModel``. ПАКЕТ — ``xs_risk`` (НЕ ``risk``).
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


def _market_returns(returns_wide: pd.DataFrame, market_symbol: Optional[str]) -> pd.Series:
    if market_symbol and market_symbol in returns_wide.columns:
        return returns_wide[market_symbol].astype("float64")
    return returns_wide.mean(axis=1).astype("float64")


def market_beta(returns_wide: pd.DataFrame, *, market_symbol: Optional[str] = None) -> pd.Series:
    """β каждого инструмента к basket'у: cov(r_i, r_m) / var(r_m)."""
    m = _market_returns(returns_wide, market_symbol)
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
    return pd.Series(out, name="market_beta")


def _standardize(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return ((s - s.mean()) / (sd or 1.0)).fillna(0.0)


def build_futures_exposures(
    returns_wide: pd.DataFrame,
    *,
    asset_classes: Optional[Dict[str, str]] = None,
    market_symbol: Optional[str] = None,
    vol_lookback: int = 60,
) -> pd.DataFrame:
    """Построить факторные экспозиции B (index=symbol, cols=[market_beta, vol, ac_*])."""
    symbols = list(returns_wide.columns)
    cols: Dict[str, pd.Series] = {}

    cols["market_beta"] = market_beta(returns_wide, market_symbol=market_symbol).reindex(symbols)

    # vol-фактор: реализованная волатильность за lookback (стандартизуем)
    if len(returns_wide) >= 2:
        k = min(int(vol_lookback), len(returns_wide))
        vol = returns_wide.tail(k).std(ddof=0)
        cols["vol"] = _standardize(vol.reindex(symbols))

    B = pd.DataFrame(cols, index=symbols)

    if asset_classes:
        ac = pd.Series({s: str(asset_classes.get(s, "OTHER")) for s in symbols}, index=symbols)
        dummies = pd.get_dummies(ac, prefix="ac", drop_first=True, dtype="float64")
        B = pd.concat([B, dummies], axis=1)

    return B.astype("float64").fillna(0.0)


def build_futures_risk_model(
    returns_wide: pd.DataFrame,
    *,
    asset_classes: Optional[Dict[str, str]] = None,
    market_symbol: Optional[str] = None,
    vol_lookback: int = 60,
    factor_cov_method: str = "ledoit_wolf",
):
    """Удобный конструктор FactorRiskModel с futures-экспозициями (asset-class)."""
    from service_risk_model import FactorRiskModel

    B = build_futures_exposures(
        returns_wide,
        asset_classes=asset_classes,
        market_symbol=market_symbol,
        vol_lookback=vol_lookback,
    )
    return FactorRiskModel(B, factor_cov_method=factor_cov_method)


__all__ = [
    "returns_wide_from_panel",
    "market_beta",
    "build_futures_exposures",
    "build_futures_risk_model",
]
