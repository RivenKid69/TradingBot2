# -*- coding: utf-8 -*-
"""
signals/forex_signals.py
========================

Forex (G10/EM) cross-sectional сигналы (Stage B4) — плагины на готовый движок, по образцу
crypto/equity/futures. Каждый сигнал — ``BaseSignal``; при отсутствии нужной колонки
возвращает NaN-серию (нейтрально) = **BYO-слот**.

ВАЖНО: kind'ы с префиксом ``fx_`` — чтобы не коллидировать с futures (`carry`/`trend`/`value`).

Сигналы:
  * ``FXCarry``        — carry = дифференциал ставок (готовая ``rate_diff``/``carry`` колонка ИЛИ
                         ``rate_base − rate_quote``; высокодоходная валюта → лонг);
  * ``FXMomentum``     — трендовый momentum FX-курса (доходность за lookback);
  * ``FXValue``        — value/PPP (готовая ``ppp``/``reer_gap`` колонка [недооценка → лонг] ИЛИ
                         прокси = −долгосрочная доходность, mean-reversion к справедливому курсу);
  * ``TermsOfTrade``   — terms-of-trade для сырьевых валют (BYO колонка ``terms_of_trade``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core_portfolio import SYMBOL_LEVEL, Panel
from service_signals import BaseSignal


def _nan_series(panel: Panel, name: str) -> pd.Series:
    return pd.Series(np.nan, index=panel.index, name=name)


def _grp(panel: Panel, col: str):
    return panel[col].astype("float64").groupby(level=SYMBOL_LEVEL, group_keys=False)


class FXCarry(BaseSignal):
    """Carry: дифференциал процентных ставок (готовая колонка ИЛИ rate_base − rate_quote)."""

    def __init__(
        self,
        name: str = "fx_carry",
        *,
        rate_diff_col: str = "rate_diff",
        carry_col: str = "carry",
        rate_base_col: str = "rate_base",
        rate_quote_col: str = "rate_quote",
    ) -> None:
        self.name = name
        self.rate_diff_col = rate_diff_col
        self.carry_col = carry_col
        self.rate_base_col = rate_base_col
        self.rate_quote_col = rate_quote_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        for col in (self.rate_diff_col, self.carry_col):
            if col in panel.columns:
                return panel[col].astype("float64").rename(self.name)
        if self.rate_base_col in panel.columns and self.rate_quote_col in panel.columns:
            diff = panel[self.rate_base_col].astype("float64") - panel[self.rate_quote_col].astype(
                "float64"
            )
            return diff.rename(self.name)
        return _nan_series(panel, self.name)


class FXMomentum(BaseSignal):
    """Трендовый momentum FX-курса: price[t] / price[t-lookback] − 1."""

    def __init__(
        self, name: str = "fx_mom", *, lookback: int = 90, price_col: str = "close"
    ) -> None:
        self.name = name
        self.lookback = int(lookback)
        self.price_col = price_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.price_col not in panel.columns:
            return _nan_series(panel, self.name)
        g = _grp(panel, self.price_col)
        return (g.shift(0) / g.shift(self.lookback) - 1.0).rename(self.name)


class FXValue(BaseSignal):
    """Value/PPP: готовая ``ppp``/``reer_gap`` колонка (недооценка→лонг) ИЛИ прокси −long-return."""

    def __init__(
        self,
        name: str = "fx_value",
        *,
        ppp_col: str = "ppp",
        reer_col: str = "reer_gap",
        lookback: int = 500,
        price_col: str = "close",
    ) -> None:
        self.name = name
        self.ppp_col = ppp_col
        self.reer_col = reer_col
        self.lookback = int(lookback)
        self.price_col = price_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        for col in (self.ppp_col, self.reer_col):
            if col in panel.columns:
                return panel[col].astype("float64").rename(self.name)
        if self.price_col in panel.columns:
            g = _grp(panel, self.price_col)
            return (-(g.shift(0) / g.shift(self.lookback) - 1.0)).rename(self.name)
        return _nan_series(panel, self.name)


class TermsOfTrade(BaseSignal):
    """Terms-of-trade (сырьевые валюты): BYO колонка ``terms_of_trade`` как есть."""

    def __init__(self, name: str = "terms_of_trade", *, terms_col: str = "terms_of_trade") -> None:
        self.name = name
        self.terms_col = terms_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.terms_col not in panel.columns:
            return _nan_series(panel, self.name)
        return panel[self.terms_col].astype("float64").rename(self.name)


_KINDS = {
    "fx_carry": FXCarry,
    "fx_momentum": FXMomentum,
    "fx_value": FXValue,
    "terms_of_trade": TermsOfTrade,
}


def build_forex_signal(kind: str, name: str, **kwargs: Any) -> BaseSignal:
    """Фабрика forex-сигнала по строковому ``kind``."""
    if kind not in _KINDS:
        raise ValueError(f"unknown forex signal kind: {kind!r}")
    return _KINDS[kind](name=name, **kwargs)


FOREX_SIGNAL_KINDS = tuple(_KINDS.keys())

__all__ = [
    "FXCarry",
    "FXMomentum",
    "FXValue",
    "TermsOfTrade",
    "build_forex_signal",
    "FOREX_SIGNAL_KINDS",
]
