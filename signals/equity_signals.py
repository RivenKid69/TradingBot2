# -*- coding: utf-8 -*-
"""
signals/equity_signals.py
=========================

Equity-специфичная библиотека cross-sectional сигналов (Stage B2) — плагины на готовый
движок (Part A), по образцу ``signals/crypto_signals.py``. Каждый сигнал — ``BaseSignal``;
при отсутствии нужной колонки в панели возвращает NaN-серию (нейтрально), что даёт
**BYO-слот**: пользователь подаёт колонку (фундаментал — ``earnings`` / ``book_value`` /
``fcf`` / ``roe`` / ``accruals`` / ``market_cap``), и сигнал «оживает».

ВАЖНО (honest-note): бесплатный фундаментал (yfinance) — снимок «сейчас», НЕ настоящий
point-in-time, а бесплатные списки SP500/NDX — survivorship-biased. Поэтому фундаментальные
сигналы помечены ``pit_quality='approx'`` в UI/логах; для честного бэктеста подайте BYO
PIT-фундаментал (Sharadar/Compustat) через ``ParquetFundamentals`` + ``asof_join``.

Сигналы:
  * ``EquityMomentum``   — 12-1 momentum (доходность за lookback с пропуском skip);
  * ``EarningsYield``    — E/P (earnings / price; высокая = «дёшево» → лонг);
  * ``BookToPrice``      — B/P (book value / price; высокая = value → лонг);
  * ``FCFYield``         — FCF / price (свободный денежный поток к цене → лонг);
  * ``ReturnOnEquity``   — ROE (качество: прибыльность капитала → лонг);
  * ``Accruals``         — −accruals (низкие начисления = качество прибыли → лонг);
  * ``LowVolatility``    — −realized-vol (low-vol аномалия → лонг низковолатильных);
  * ``EquitySize``       — −log(market_cap) (small-cap премия).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from core_portfolio import SYMBOL_LEVEL, Panel
from service_signals import BaseSignal


def _nan_series(panel: Panel, name: str) -> pd.Series:
    return pd.Series(np.nan, index=panel.index, name=name)


def _yield_or_ratio(panel: Panel, name: str, *, yield_col: str, num_col: str, price_col: str) -> pd.Series:
    """Готовая yield-колонка если есть; иначе ``num_col / price_col``; иначе NaN (BYO)."""
    if yield_col in panel.columns:
        return panel[yield_col].astype("float64").rename(name)
    if num_col in panel.columns and price_col in panel.columns:
        num = panel[num_col].astype("float64")
        px = panel[price_col].astype("float64").replace(0.0, np.nan)
        return (num / px).rename(name)
    return _nan_series(panel, name)


class EquityMomentum(BaseSignal):
    """12-1 momentum: price[t-skip] / price[t-lookback] − 1 (по символу).

    Классика: на дневных барах ``lookback=252, skip=21`` (год минус последний месяц).
    """

    def __init__(self, name: str = "eq_mom", *, lookback: int = 252, skip: int = 21, price_col: str = "close") -> None:
        self.name = name
        self.lookback = int(lookback)
        self.skip = int(skip)
        self.price_col = price_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.price_col not in panel.columns:
            return _nan_series(panel, self.name)
        g = panel[self.price_col].astype("float64").groupby(level=SYMBOL_LEVEL, group_keys=False)
        return (g.shift(self.skip) / g.shift(self.lookback) - 1.0).rename(self.name)


class EarningsYield(BaseSignal):
    """E/P: earnings yield (готовая ``ep`` колонка или ``earnings``/``close``)."""

    def __init__(self, name: str = "earnings_yield", *, yield_col: str = "ep",
                 earnings_col: str = "earnings", price_col: str = "close") -> None:
        self.name = name
        self.yield_col = yield_col
        self.earnings_col = earnings_col
        self.price_col = price_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        return _yield_or_ratio(panel, self.name, yield_col=self.yield_col,
                               num_col=self.earnings_col, price_col=self.price_col)


class BookToPrice(BaseSignal):
    """B/P: book-to-price (готовая ``bp`` колонка или ``book_value``/``close``)."""

    def __init__(self, name: str = "book_to_price", *, yield_col: str = "bp",
                 book_col: str = "book_value", price_col: str = "close") -> None:
        self.name = name
        self.yield_col = yield_col
        self.book_col = book_col
        self.price_col = price_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        return _yield_or_ratio(panel, self.name, yield_col=self.yield_col,
                               num_col=self.book_col, price_col=self.price_col)


class FCFYield(BaseSignal):
    """FCF yield: free-cash-flow / price (готовая ``fcf_yield`` или ``fcf``/``close``)."""

    def __init__(self, name: str = "fcf_yield", *, yield_col: str = "fcf_yield",
                 fcf_col: str = "fcf", price_col: str = "close") -> None:
        self.name = name
        self.yield_col = yield_col
        self.fcf_col = fcf_col
        self.price_col = price_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        return _yield_or_ratio(panel, self.name, yield_col=self.yield_col,
                               num_col=self.fcf_col, price_col=self.price_col)


class ReturnOnEquity(BaseSignal):
    """Quality: ROE (return on equity) как есть — нормирует трансформ."""

    def __init__(self, name: str = "roe", *, roe_col: str = "roe") -> None:
        self.name = name
        self.roe_col = roe_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.roe_col not in panel.columns:
            return _nan_series(panel, self.name)
        return panel[self.roe_col].astype("float64").rename(self.name)


class Accruals(BaseSignal):
    """Quality: −accruals (низкие начисления → качество прибыли → лонг)."""

    def __init__(self, name: str = "accruals", *, accruals_col: str = "accruals") -> None:
        self.name = name
        self.accruals_col = accruals_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.accruals_col not in panel.columns:
            return _nan_series(panel, self.name)
        return (-panel[self.accruals_col].astype("float64")).rename(self.name)


class LowVolatility(BaseSignal):
    """Low-vol аномалия: −(rolling std дневных доходностей за ``window``)."""

    def __init__(self, name: str = "low_vol", *, window: int = 60, price_col: str = "close") -> None:
        self.name = name
        self.window = int(window)
        self.price_col = price_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.price_col not in panel.columns:
            return _nan_series(panel, self.name)
        g = panel[self.price_col].astype("float64").groupby(level=SYMBOL_LEVEL, group_keys=False)
        ret = g.pct_change()
        vol = ret.groupby(level=SYMBOL_LEVEL, group_keys=False).rolling(self.window).std()
        # rolling добавляет лишний уровень индекса — выровняем обратно к панели
        vol = vol.reset_index(level=0, drop=True) if isinstance(vol.index, pd.MultiIndex) and vol.index.nlevels > 2 else vol
        return (-vol).reindex(panel.index).rename(self.name)


class EquitySize(BaseSignal):
    """Size-фактор: −log(market_cap) (small-cap премия). Сырое значение, нормирует трансформ."""

    def __init__(self, name: str = "eq_size", *, mcap_col: str = "market_cap") -> None:
        self.name = name
        self.mcap_col = mcap_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.mcap_col not in panel.columns:
            return _nan_series(panel, self.name)
        mc = panel[self.mcap_col].astype("float64").clip(lower=1e-9)
        return (-np.log(mc)).rename(self.name)


_KINDS = {
    "equity_momentum": EquityMomentum,
    "earnings_yield": EarningsYield,
    "book_to_price": BookToPrice,
    "fcf_yield": FCFYield,
    "roe": ReturnOnEquity,
    "accruals": Accruals,
    "low_vol": LowVolatility,
    "equity_size": EquitySize,
}


def build_equity_signal(kind: str, name: str, **kwargs: Any) -> BaseSignal:
    """Фабрика equity-сигнала по строковому ``kind``."""
    if kind not in _KINDS:
        raise ValueError(f"unknown equity signal kind: {kind!r}")
    return _KINDS[kind](name=name, **kwargs)


EQUITY_SIGNAL_KINDS = tuple(_KINDS.keys())

__all__ = [
    "EquityMomentum", "EarningsYield", "BookToPrice", "FCFYield",
    "ReturnOnEquity", "Accruals", "LowVolatility", "EquitySize",
    "build_equity_signal", "EQUITY_SIGNAL_KINDS",
]
