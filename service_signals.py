# -*- coding: utf-8 -*-
"""
service_signals.py
==================

Каркас сигналов cross-sectional контура (Stage A4): базовый ``Signal`` ABC,
несколько **asset-agnostic** примитивов и реестр ``SignalLibrary`` с пайплайном
``compute → transform → (neutralize) → signal panel``.

Конкретные библиотеки сигналов по классам активов (momentum/value/carry/funding…)
добавляются в Part B (``signals/<asset>_signals.py``) — здесь только инфраструктура и
минимальные универсальные примитивы (``ColumnSignal``, ``MomentumSignal``,
``FunctionSignal``).

Каждый сигнал реализует контракт ``core_portfolio.Signal`` (``name`` +
``compute(panel, asof_ms) -> Series``). Для эффективности базовый класс считает весь
панельный сигнал сразу (``compute_panel``), а ``compute`` отдаёт срез на дату.

Слой ``service_`` (зависит от ``core_``/``impl_``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

from core_portfolio import PANEL_INDEX_NAMES, SYMBOL_LEVEL, TS_LEVEL, Panel
from impl_cross_sectional import Step, run_pipeline


# ---------------------------------------------------------------------------
# Signal base + primitives
# ---------------------------------------------------------------------------
class BaseSignal(ABC):
    """Базовый cross-sectional сигнал. Реализуйте ``compute_panel``."""

    name: str = "signal"

    @abstractmethod
    def compute_panel(self, panel: Panel) -> pd.Series:
        """Сырой сигнал по всей панели: MultiIndex (ts_ms, symbol) Series."""
        raise NotImplementedError

    def compute(self, panel: Panel, asof_ms: int) -> pd.Series:
        """Срез сигнала на дату ``asof_ms`` (контракт core_portfolio.Signal)."""
        full = self.compute_panel(panel)
        try:
            return full.xs(int(asof_ms), level=TS_LEVEL)
        except KeyError:
            return pd.Series(dtype="float64")


class ColumnSignal(BaseSignal):
    """Сигнал = готовая колонка панели (например, заранее посчитанная фича)."""

    def __init__(self, name: str, column: Optional[str] = None) -> None:
        self.name = name
        self.column = column or name

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.column not in panel.columns:
            raise ValueError(f"ColumnSignal: panel missing column '{self.column}'")
        return panel[self.column].rename(self.name)


class MomentumSignal(BaseSignal):
    """Универсальный momentum: доходность за ``lookback`` баров с пропуском ``skip``.

    ``signal[t] = price[t-skip] / price[t-lookback] - 1`` по каждому символу.
    (Классический 12-1 momentum: ``lookback=12, skip=1`` на месячных барах.)
    """

    def __init__(
        self,
        name: str = "momentum",
        *,
        lookback: int = 60,
        skip: int = 0,
        price_col: str = "close",
    ) -> None:
        self.name = name
        self.lookback = int(lookback)
        self.skip = int(skip)
        self.price_col = price_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.price_col not in panel.columns:
            raise ValueError(f"MomentumSignal: panel missing '{self.price_col}'")
        g = panel[self.price_col].astype("float64").groupby(level=SYMBOL_LEVEL, group_keys=False)
        num = g.shift(self.skip)
        den = g.shift(self.lookback)
        mom = num / den - 1.0
        return mom.rename(self.name)


class FunctionSignal(BaseSignal):
    """Сигнал из произвольной функции ``fn(panel) -> Series`` (BYO-сигналы)."""

    def __init__(self, name: str, fn: Callable[[Panel], pd.Series]) -> None:
        self.name = name
        self._fn = fn

    def compute_panel(self, panel: Panel) -> pd.Series:
        out = self._fn(panel)
        return out.rename(self.name)


# ---------------------------------------------------------------------------
# Signal spec + library
# ---------------------------------------------------------------------------
@dataclass
class SignalSpec:
    """Сигнал + цепочка трансформов + нейтрализация."""

    signal: BaseSignal
    transforms: List[Step] = field(default_factory=list)
    neutralize_by: List[str] = field(default_factory=list)
    name: Optional[str] = None

    @property
    def output_name(self) -> str:
        return self.name or self.signal.name


class SignalLibrary:
    """Реестр сигналов: считает панель сигналов (колонка на сигнал) с трансформами."""

    def __init__(self) -> None:
        self._specs: List[SignalSpec] = []

    def register(
        self,
        signal: BaseSignal,
        *,
        transforms: Optional[Sequence[Step]] = None,
        neutralize_by: Optional[Sequence[str]] = None,
        name: Optional[str] = None,
    ) -> "SignalLibrary":
        self._specs.append(
            SignalSpec(
                signal=signal,
                transforms=list(transforms or []),
                neutralize_by=list(neutralize_by or []),
                name=name,
            )
        )
        return self

    def register_spec(self, spec: SignalSpec) -> "SignalLibrary":
        self._specs.append(spec)
        return self

    @property
    def names(self) -> List[str]:
        return [s.output_name for s in self._specs]

    def compute(self, panel: Panel) -> Panel:
        """Панель сигналов: DataFrame с MultiIndex (ts_ms, symbol), колонки = сигналы."""
        cols: Dict[str, pd.Series] = {}
        for spec in self._specs:
            raw = spec.signal.compute_panel(panel)
            steps: List[Step] = list(spec.transforms)
            if spec.neutralize_by:
                steps.append(("neutralize", {"by": list(spec.neutralize_by)}))
            out = run_pipeline(raw, steps, factor_panel=panel)
            cols[spec.output_name] = out
        if not cols:
            result = pd.DataFrame(index=panel.index)
        else:
            result = pd.DataFrame(cols)
        result.index.names = PANEL_INDEX_NAMES
        return result


__all__ = [
    "BaseSignal",
    "ColumnSignal",
    "MomentumSignal",
    "FunctionSignal",
    "SignalSpec",
    "SignalLibrary",
]
