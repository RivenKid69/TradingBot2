# -*- coding: utf-8 -*-
"""
signals/futures_signals.py
==========================

Futures (CTA-style) cross-sectional сигналы (Stage B3) — плагины на готовый движок,
по образцу crypto/equity. Работают на **back-adjusted непрерывных** сериях
(``impl_continuous_futures``). Каждый сигнал — ``BaseSignal``; при отсутствии нужной
колонки возвращает NaN-серию (нейтрально) = **BYO-слот**.

Сигналы:
  * ``Trend``        — time-series momentum (доходность за lookback; trend 50/100/200),
                       опц. нормировка на реализованную волатильность (vol-target);
  * ``Carry``        — roll-yield / carry перпов-фьючерсов (готовая ``carry``/``roll_yield``
                       колонка ИЛИ из ``front``/``back`` контрактов; backwardation → лонг);
  * ``FuturesValue`` — долгосрочный mean-reversion (−доходность за длинный lookback);
  * ``RealizedVolInv`` — обратная реализованная волатильность (для vol-target сайзинга).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from core_portfolio import SYMBOL_LEVEL, Panel
from service_signals import BaseSignal


def _nan_series(panel: Panel, name: str) -> pd.Series:
    return pd.Series(np.nan, index=panel.index, name=name)


def _grp(panel: Panel, col: str):
    return panel[col].astype("float64").groupby(level=SYMBOL_LEVEL, group_keys=False)


def _realized_vol(panel: Panel, price_col: str, window: int) -> pd.Series:
    ret = _grp(panel, price_col).pct_change()
    vol = ret.groupby(level=SYMBOL_LEVEL, group_keys=False).rolling(int(window)).std()
    if isinstance(vol.index, pd.MultiIndex) and vol.index.nlevels > 2:
        vol = vol.reset_index(level=0, drop=True)
    return vol.reindex(panel.index)


class Trend(BaseSignal):
    """Time-series momentum: price[t] / price[t-lookback] − 1 (опц. /реализованная vol)."""

    def __init__(self, name: str = "trend", *, lookback: int = 100, price_col: str = "close",
                 vol_normalize: bool = False, vol_window: int = 60) -> None:
        self.name = name
        self.lookback = int(lookback)
        self.price_col = price_col
        self.vol_normalize = bool(vol_normalize)
        self.vol_window = int(vol_window)

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.price_col not in panel.columns:
            return _nan_series(panel, self.name)
        g = _grp(panel, self.price_col)
        trend = g.shift(0) / g.shift(self.lookback) - 1.0
        if self.vol_normalize:
            vol = _realized_vol(panel, self.price_col, self.vol_window).replace(0.0, np.nan)
            trend = trend / vol
        return trend.rename(self.name)


class Carry(BaseSignal):
    """Carry / roll-yield: готовая ``carry``/``roll_yield`` колонка ИЛИ ``(front-back)/back``.

    Backwardation (front<back? нет — backwardation = передний дороже дальнего → положит. carry)
    → лонг. Знак: положительный carry = лонг. Если есть готовая колонка — берём как есть.
    """

    def __init__(self, name: str = "carry", *, carry_col: str = "carry",
                 roll_yield_col: str = "roll_yield", front_col: str = "front",
                 back_col: str = "back") -> None:
        self.name = name
        self.carry_col = carry_col
        self.roll_yield_col = roll_yield_col
        self.front_col = front_col
        self.back_col = back_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        for col in (self.carry_col, self.roll_yield_col):
            if col in panel.columns:
                return panel[col].astype("float64").rename(self.name)
        if self.front_col in panel.columns and self.back_col in panel.columns:
            front = panel[self.front_col].astype("float64")
            back = panel[self.back_col].astype("float64").replace(0.0, np.nan)
            return ((front - back) / back).rename(self.name)
        return _nan_series(panel, self.name)


class FuturesValue(BaseSignal):
    """Value (long-horizon mean-reversion): −(price[t] / price[t-lookback] − 1)."""

    def __init__(self, name: str = "fut_value", *, lookback: int = 1000, price_col: str = "close") -> None:
        self.name = name
        self.lookback = int(lookback)
        self.price_col = price_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.price_col not in panel.columns:
            return _nan_series(panel, self.name)
        g = _grp(panel, self.price_col)
        return (-(g.shift(0) / g.shift(self.lookback) - 1.0)).rename(self.name)


class RealizedVolInv(BaseSignal):
    """Обратная реализованная волатильность (vol-target сайзинг): 1/σ (сырое, нормирует трансформ)."""

    def __init__(self, name: str = "inv_vol", *, window: int = 60, price_col: str = "close") -> None:
        self.name = name
        self.window = int(window)
        self.price_col = price_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.price_col not in panel.columns:
            return _nan_series(panel, self.name)
        vol = _realized_vol(panel, self.price_col, self.window)
        return (1.0 / vol.replace(0.0, np.nan)).rename(self.name)


_KINDS = {
    "trend": Trend,
    "carry": Carry,
    "futures_value": FuturesValue,
    "inv_vol": RealizedVolInv,
}


def build_futures_signal(kind: str, name: str, **kwargs: Any) -> BaseSignal:
    """Фабрика futures-сигнала по строковому ``kind``."""
    if kind not in _KINDS:
        raise ValueError(f"unknown futures signal kind: {kind!r}")
    return _KINDS[kind](name=name, **kwargs)


FUTURES_SIGNAL_KINDS = tuple(_KINDS.keys())

__all__ = [
    "Trend", "Carry", "FuturesValue", "RealizedVolInv",
    "build_futures_signal", "FUTURES_SIGNAL_KINDS",
]
