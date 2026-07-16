# -*- coding: utf-8 -*-
"""
signals/options_signals.py
==========================

Options cross-sectional сигналы (Stage B5) — оценивают привлекательность vol-структур по
символам/андерлаям; результат = альфа по ногам для **greeks-space оптимизатора**
(``service_options_portfolio``), а НЕ directional веса. Каждый сигнал — ``BaseSignal``;
graceful к отсутствующим колонкам = **BYO-слот** (опционные данные обычно платные/Deribit).

Сигналы:
  * ``VolRiskPremium`` — VRP = IV − realized vol (богатая implied vol → продавать vol);
  * ``Skew``           — risk-reversal: IV(put 25Δ) − IV(call 25Δ) (готовая ``skew`` ИЛИ из колонок);
  * ``Dispersion``     — index IV − средняя single-name IV (готовая ``dispersion``);
  * ``TermStructure``  — calendar: IV(front) − IV(back) (готовая ``term_slope`` ИЛИ из колонок).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core_portfolio import Panel
from service_signals import BaseSignal


def _nan_series(panel: Panel, name: str) -> pd.Series:
    return pd.Series(np.nan, index=panel.index, name=name)


class VolRiskPremium(BaseSignal):
    """VRP = IV − realized vol (готовая ``vrp`` колонка ИЛИ ``iv − realized_vol``)."""

    def __init__(self, name: str = "vrp", *, vrp_col: str = "vrp",
                 iv_col: str = "iv", rv_col: str = "realized_vol") -> None:
        self.name = name
        self.vrp_col = vrp_col
        self.iv_col = iv_col
        self.rv_col = rv_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.vrp_col in panel.columns:
            return panel[self.vrp_col].astype("float64").rename(self.name)
        if self.iv_col in panel.columns and self.rv_col in panel.columns:
            return (panel[self.iv_col].astype("float64") - panel[self.rv_col].astype("float64")).rename(self.name)
        return _nan_series(panel, self.name)


class Skew(BaseSignal):
    """Skew / risk-reversal: IV(put 25Δ) − IV(call 25Δ) (готовая ``skew`` ИЛИ из колонок)."""

    def __init__(self, name: str = "skew", *, skew_col: str = "skew",
                 put_iv_col: str = "iv_put_25", call_iv_col: str = "iv_call_25") -> None:
        self.name = name
        self.skew_col = skew_col
        self.put_iv_col = put_iv_col
        self.call_iv_col = call_iv_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.skew_col in panel.columns:
            return panel[self.skew_col].astype("float64").rename(self.name)
        if self.put_iv_col in panel.columns and self.call_iv_col in panel.columns:
            return (panel[self.put_iv_col].astype("float64") - panel[self.call_iv_col].astype("float64")).rename(self.name)
        return _nan_series(panel, self.name)


class Dispersion(BaseSignal):
    """Dispersion: index IV − средняя single-name IV (готовая ``dispersion`` колонка, BYO)."""

    def __init__(self, name: str = "dispersion", *, dispersion_col: str = "dispersion") -> None:
        self.name = name
        self.dispersion_col = dispersion_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.dispersion_col not in panel.columns:
            return _nan_series(panel, self.name)
        return panel[self.dispersion_col].astype("float64").rename(self.name)


class TermStructure(BaseSignal):
    """Term-structure: IV(front) − IV(back) (готовая ``term_slope`` ИЛИ ``iv_front − iv_back``)."""

    def __init__(self, name: str = "term_structure", *, slope_col: str = "term_slope",
                 front_col: str = "iv_front", back_col: str = "iv_back") -> None:
        self.name = name
        self.slope_col = slope_col
        self.front_col = front_col
        self.back_col = back_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.slope_col in panel.columns:
            return panel[self.slope_col].astype("float64").rename(self.name)
        if self.front_col in panel.columns and self.back_col in panel.columns:
            return (panel[self.front_col].astype("float64") - panel[self.back_col].astype("float64")).rename(self.name)
        return _nan_series(panel, self.name)


_KINDS = {
    "vrp": VolRiskPremium,
    "skew": Skew,
    "dispersion": Dispersion,
    "term_structure": TermStructure,
}


def build_options_signal(kind: str, name: str, **kwargs: Any) -> BaseSignal:
    """Фабрика options-сигнала по строковому ``kind``."""
    if kind not in _KINDS:
        raise ValueError(f"unknown options signal kind: {kind!r}")
    return _KINDS[kind](name=name, **kwargs)


OPTIONS_SIGNAL_KINDS = tuple(_KINDS.keys())

__all__ = [
    "VolRiskPremium", "Skew", "Dispersion", "TermStructure",
    "build_options_signal", "OPTIONS_SIGNAL_KINDS",
]
