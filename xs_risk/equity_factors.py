# -*- coding: utf-8 -*-
"""
xs_risk/equity_factors.py
=========================

Equity-факторные экспозиции (Stage B2, Barra-lite) для факторной риск-модели
Σ = B F Bᵀ + D:

  * **market_beta** — чувствительность к рынку (бета к индексу/равновзвешенному прокси);
  * **size**        — log(market_cap) (размер);
  * **value**       — value-нагрузка (book-to-price / earnings yield, BYO-скор);
  * **momentum**    — трейлинг-доходность (12-1 прокси из доходностей);
  * **sector**      — сектор (GICS-подобный) как one-hot dummies.

Производит ``exposures`` DataFrame (index=symbol, cols=factors) для
``service_risk_model.FactorRiskModel``. Плагин на готовый движок (зеркало
``xs_risk/crypto_factors.py``). ВНИМАНИЕ: пакет называется ``xs_risk`` (НЕ ``risk``),
чтобы не шадовить существующий top-level ``risk.py``.
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


def _market_returns(returns_wide: pd.DataFrame, market_symbol: Optional[str]) -> pd.Series:
    """Рыночные доходности: явный индекс если задан/присутствует, иначе равновзвешенный прокси."""
    if market_symbol and market_symbol in returns_wide.columns:
        return returns_wide[market_symbol].astype("float64")
    return returns_wide.mean(axis=1).astype("float64")


def market_beta(returns_wide: pd.DataFrame, *, market_symbol: Optional[str] = None) -> pd.Series:
    """β каждого символа к рынку: cov(r_i, r_m) / var(r_m)."""
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


def _value_from_fundamentals(
    symbols: List[str],
    mcaps: Optional[Dict[str, float]],
    earnings: Optional[Dict[str, float]],
    book: Optional[Dict[str, float]],
) -> Optional[pd.Series]:
    """BARRA-style VALUE built from fundamentals (P2 #17): composite of E/P and B/P.

    earnings/book are per-name fundamentals (e.g. net income TTM, book value); market
    cap is the price denominator → E/P + B/P, z-scored cross-sectionally and averaged.
    """
    parts: List[pd.Series] = []
    if mcaps and earnings:
        mc = pd.Series({s: float(mcaps.get(s, np.nan)) for s in symbols}).replace(0.0, np.nan)
        e = pd.Series({s: float(earnings.get(s, np.nan)) for s in symbols})
        parts.append(_standardize(e / mc))  # earnings yield E/P
    if mcaps and book:
        mc = pd.Series({s: float(mcaps.get(s, np.nan)) for s in symbols}).replace(0.0, np.nan)
        b = pd.Series({s: float(book.get(s, np.nan)) for s in symbols})
        parts.append(_standardize(b / mc))  # book-to-price B/P
    if not parts:
        return None
    return _standardize(pd.concat(parts, axis=1).mean(axis=1))


def build_equity_exposures(
    returns_wide: pd.DataFrame,
    *,
    sectors: Optional[Dict[str, str]] = None,
    mcaps: Optional[Dict[str, float]] = None,
    values: Optional[Dict[str, float]] = None,
    earnings: Optional[Dict[str, float]] = None,  # P2 #17: build value from fundamentals
    book: Optional[Dict[str, float]] = None,
    roe: Optional[Dict[str, float]] = None,  # P2 #17: quality factor
    market_symbol: Optional[str] = None,
    momentum_lookback: int = 60,
    vol_lookback: int = 60,
) -> pd.DataFrame:
    """Построить BARRA-lite экспозиции B (index=symbol):
    [market_beta, size, value, momentum, quality, low_vol, sector_*].

    VALUE и QUALITY теперь строятся ИЗ ФУНДАМЕНТАЛОВ (earnings/book/roe), а не только
    из готового BYO-скора; LOW_VOL — из трейлинг-волатильности доходностей.
    """
    symbols = list(returns_wide.columns)
    cols: Dict[str, pd.Series] = {}

    cols["market_beta"] = market_beta(returns_wide, market_symbol=market_symbol).reindex(symbols)

    if mcaps:
        mc = pd.Series({s: float(mcaps.get(s, np.nan)) for s in symbols}).clip(lower=1e-9)
        cols["size"] = _standardize(np.log(mc))

    # VALUE: prefer fundamentals (E/P, B/P); fall back to a BYO value score.
    val = _value_from_fundamentals(symbols, mcaps, earnings, book)
    if val is None and values:
        val = _standardize(pd.Series({s: float(values.get(s, np.nan)) for s in symbols}))
    if val is not None:
        cols["value"] = val

    # QUALITY: ROE z-scored (P2 #17).
    if roe:
        cols["quality"] = _standardize(pd.Series({s: float(roe.get(s, np.nan)) for s in symbols}))

    # momentum-фактор: трейлинг-доходность за lookback (из доходностей), стандартизуем
    if len(returns_wide) >= 2:
        k = min(int(momentum_lookback), len(returns_wide))
        trail = (1.0 + returns_wide.tail(k).fillna(0.0)).prod(axis=0) - 1.0
        cols["momentum"] = _standardize(trail.reindex(symbols))
        # LOW_VOL: negative trailing volatility (low-risk anomaly), standardized.
        kv = min(int(vol_lookback), len(returns_wide))
        vol = returns_wide.tail(kv).std(ddof=0).reindex(symbols)
        cols["low_vol"] = _standardize(-vol)

    B = pd.DataFrame(cols, index=symbols)

    if sectors:
        sec = pd.Series({s: str(sectors.get(s, "OTHER")) for s in symbols}, index=symbols)
        dummies = pd.get_dummies(sec, prefix="sector", drop_first=True, dtype="float64")
        B = pd.concat([B, dummies], axis=1)

    return B.astype("float64").fillna(0.0)


def build_equity_risk_model(
    returns_wide: pd.DataFrame,
    *,
    sectors: Optional[Dict[str, str]] = None,
    mcaps: Optional[Dict[str, float]] = None,
    values: Optional[Dict[str, float]] = None,
    market_symbol: Optional[str] = None,
    momentum_lookback: int = 60,
    factor_cov_method: str = "ledoit_wolf",
):
    """Удобный конструктор FactorRiskModel с equity-экспозициями (Barra-lite)."""
    from service_risk_model import FactorRiskModel

    B = build_equity_exposures(
        returns_wide,
        sectors=sectors,
        mcaps=mcaps,
        values=values,
        market_symbol=market_symbol,
        momentum_lookback=momentum_lookback,
    )
    return FactorRiskModel(B, factor_cov_method=factor_cov_method)


__all__ = [
    "returns_wide_from_panel",
    "market_beta",
    "build_equity_exposures",
    "build_equity_risk_model",
]
