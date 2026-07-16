# -*- coding: utf-8 -*-
"""
signals/crypto_signals.py
=========================

Крипто-специфичная библиотека cross-sectional сигналов (Stage B1) — плагины на готовый
движок (Part A). Каждый сигнал — ``BaseSignal``; при отсутствии нужной колонки в панели
возвращает NaN-серию (нейтрально), что даёт **BYO-слот**: пользователь подаёт колонку
(``funding_rate`` / ``basis`` / ``mcap`` / on-chain), и сигнал «оживает».

Сигналы:
  * ``CryptoMomentum``      — доходность за lookback с пропуском skip (close);
  * ``ShortTermReversal``   — краткосрочный разворот (−недавняя доходность);
  * ``FundingCarry``        — carry по funding rate перпетуалов (−funding);
  * ``Basis``               — базис spot-perp (−basis: контанго → шорт перпа);
  * ``Size``                — size-фактор по market cap (−log mcap: small-cap премия);
  * ``OnChain``             — произвольная on-chain колонка (BYO).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from core_portfolio import SYMBOL_LEVEL, Panel
from service_signals import BaseSignal


def _nan_series(panel: Panel, name: str) -> pd.Series:
    return pd.Series(np.nan, index=panel.index, name=name)


class CryptoMomentum(BaseSignal):
    """Momentum: price[t-skip] / price[t-lookback] − 1 (по символу)."""

    def __init__(self, name: str = "crypto_mom", *, lookback: int = 90, skip: int = 7, price_col: str = "close") -> None:
        self.name = name
        self.lookback = int(lookback)
        self.skip = int(skip)
        self.price_col = price_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.price_col not in panel.columns:
            return _nan_series(panel, self.name)
        g = panel[self.price_col].astype("float64").groupby(level=SYMBOL_LEVEL, group_keys=False)
        return (g.shift(self.skip) / g.shift(self.lookback) - 1.0).rename(self.name)


class ShortTermReversal(BaseSignal):
    """Краткосрочный разворот: −(price[t] / price[t-window] − 1)."""

    def __init__(self, name: str = "reversal", *, window: int = 5, price_col: str = "close") -> None:
        self.name = name
        self.window = int(window)
        self.price_col = price_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.price_col not in panel.columns:
            return _nan_series(panel, self.name)
        g = panel[self.price_col].astype("float64").groupby(level=SYMBOL_LEVEL, group_keys=False)
        return (-(g.shift(0) / g.shift(self.window) - 1.0)).rename(self.name)


class FundingCarry(BaseSignal):
    """Funding carry: −funding_rate (положительный funding → лонги платят → шорт)."""

    def __init__(self, name: str = "funding_carry", *, funding_col: str = "funding_rate") -> None:
        self.name = name
        self.funding_col = funding_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.funding_col not in panel.columns:
            return _nan_series(panel, self.name)
        return (-panel[self.funding_col].astype("float64")).rename(self.name)


class Basis(BaseSignal):
    """Базис spot-perp: −basis (контанго perp>spot → шорт перпа).

    Берёт колонку ``basis_col`` если есть; иначе считает из ``perp_col``/``spot_col``.
    """

    def __init__(self, name: str = "basis", *, basis_col: str = "basis",
                 perp_col: str = "perp_close", spot_col: str = "close") -> None:
        self.name = name
        self.basis_col = basis_col
        self.perp_col = perp_col
        self.spot_col = spot_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.basis_col in panel.columns:
            return (-panel[self.basis_col].astype("float64")).rename(self.name)
        if self.perp_col in panel.columns and self.spot_col in panel.columns:
            basis = panel[self.perp_col].astype("float64") / panel[self.spot_col].astype("float64") - 1.0
            return (-basis).rename(self.name)
        return _nan_series(panel, self.name)


class Size(BaseSignal):
    """Size-фактор: −log(mcap) (small-cap премия). Сырое значение, нормирует трансформ."""

    def __init__(self, name: str = "size", *, mcap_col: str = "mcap") -> None:
        self.name = name
        self.mcap_col = mcap_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.mcap_col not in panel.columns:
            return _nan_series(panel, self.name)
        mc = panel[self.mcap_col].astype("float64").clip(lower=1e-9)
        return (-np.log(mc)).rename(self.name)


class OnChain(BaseSignal):
    """Произвольный on-chain сигнал (BYO): значение колонки ``column`` как есть."""

    def __init__(self, name: str = "onchain", *, column: str = "onchain") -> None:
        self.name = name
        self.column = column

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.column not in panel.columns:
            return _nan_series(panel, self.name)
        return panel[self.column].astype("float64").rename(self.name)


_KINDS = {
    "crypto_momentum": CryptoMomentum,
    "reversal": ShortTermReversal,
    "funding_carry": FundingCarry,
    "basis": Basis,
    "size": Size,
    "onchain": OnChain,
}


def build_crypto_signal(kind: str, name: str, **kwargs: Any) -> BaseSignal:
    """Фабрика крипто-сигнала по строковому ``kind``."""
    if kind not in _KINDS:
        raise ValueError(f"unknown crypto signal kind: {kind!r}")
    return _KINDS[kind](name=name, **kwargs)


CRYPTO_SIGNAL_KINDS = tuple(_KINDS.keys())

__all__ = [
    "CryptoMomentum", "ShortTermReversal", "FundingCarry", "Basis", "Size", "OnChain",
    "build_crypto_signal", "CRYPTO_SIGNAL_KINDS",
]
