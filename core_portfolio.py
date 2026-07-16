# -*- coding: utf-8 -*-
"""
core_portfolio.py
=================

Базовые контракты (Protocols) и типы для cross-sectional парадигмы:

    «десятки сигналов → риск-модель → портфельная оптимизация по всему юниверсу»

Слой ``core_`` — без тяжёлых зависимостей (только pandas/numpy/typing/dataclasses).
Реализации живут в ``impl_*`` / ``service_*``. См. ``CROSS_SECTIONAL_PLATFORM_DESIGN.md``.

Этот модуль НИЧЕГО не включает в существующем single-instrument режиме — он лишь
описывает интерфейсы нового (аддитивного) контура. Stage A1 из
``CROSS_SECTIONAL_BUILD_ROADMAP.md``.

Ключевая структура данных — **Panel**: ``pandas.DataFrame`` с MultiIndex
``(ts_ms, symbol)``, где колонки = фичи/сигналы. Единица работы cross-sectional
стратегии — вектор **target weights** (``pandas.Series`` с index = symbol) на дату
ребаланса.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Run modes (аддитивно к core_config.CommonRunConfig.mode)
# ---------------------------------------------------------------------------
#: Текущее поведение MVP (per-symbol стратегия / RL-агент на инструмент).
MODE_SINGLE_INSTRUMENT = "single_instrument"
#: Новый контур: universe-wide signals → risk model → portfolio optimization.
MODE_CROSS_SECTIONAL = "cross_sectional"
VALID_RUN_MODES = (MODE_SINGLE_INSTRUMENT, MODE_CROSS_SECTIONAL)

# ---------------------------------------------------------------------------
# Canonical panel index
# ---------------------------------------------------------------------------
#: Имена уровней MultiIndex панели. ts_ms — int64 миллисекунды UTC.
PANEL_INDEX_NAMES = ("ts_ms", "symbol")
TS_LEVEL = "ts_ms"
SYMBOL_LEVEL = "symbol"

# ---------------------------------------------------------------------------
# Type aliases (документирующие; рантайм-проверки делает validate_panel)
# ---------------------------------------------------------------------------
#: Panel: DataFrame с MultiIndex (ts_ms, symbol), columns = features/signals.
Panel = pd.DataFrame
#: Вектор целевых весов: Series, index = symbol, value = вес (доля капитала).
TargetWeights = pd.Series
#: Ожидаемые доходности μ: Series, index = symbol.
ExpectedReturns = pd.Series
#: Ковариация активов Σ: DataFrame, index == columns == symbols.
CovMatrix = pd.DataFrame


# ---------------------------------------------------------------------------
# Rebalance event
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RebalanceEvent:
    """Одно событие ребаланса: целевые веса по юниверсу на дату ``ts_ms``.

    target_weights — Series (index=symbol). В CCEA это набор Intent'ов:
    Cloud отдаёт целевые экспозиции, Agent локально превращает их в ордера.
    """

    ts_ms: int
    target_weights: pd.Series
    meta: Dict[str, Any] = field(default_factory=dict)

    def symbols(self) -> Sequence[str]:
        return list(self.target_weights.index)

    def gross(self) -> float:
        return float(np.abs(self.target_weights.to_numpy(dtype=float)).sum())

    def net(self) -> float:
        return float(self.target_weights.to_numpy(dtype=float).sum())


# ---------------------------------------------------------------------------
# Protocols (контракты слоёв пайплайна)
# ---------------------------------------------------------------------------
@runtime_checkable
class UniverseProvider(Protocol):
    """Point-in-time состав торгуемого юниверса (без survivorship bias)."""

    def constituents(self, asof_ms: int) -> Sequence[str]:
        """Список символов, входящих в юниверс на дату ``asof_ms``."""
        ...

    def is_tradable(self, symbol: str, asof_ms: int) -> bool:
        """Был ли ``symbol`` торгуемым на дату ``asof_ms`` (не делистнут и т.п.)."""
        ...


@runtime_checkable
class Signal(Protocol):
    """Cross-sectional сигнал: на дату t отдаёт вектор по символам.

    Нормализация/нейтрализация выполняется отдельным слоем
    (``impl_cross_sectional``), сам сигнал возвращает «сырое» значение.
    """

    name: str

    def compute(self, panel: Panel, asof_ms: int) -> pd.Series:
        """Series (index=symbol) сырого значения сигнала на ``asof_ms``."""
        ...


@runtime_checkable
class AlphaModel(Protocol):
    """Комбинирует набор сигналов в ожидаемую доходность μ по юниверсу."""

    def fit(self, signals: Panel, forward_returns: Panel) -> None:
        ...

    def predict(self, signals_t: pd.DataFrame) -> pd.Series:
        """μ: Series (index=symbol) ожидаемых доходностей на дату t."""
        ...


@runtime_checkable
class RiskModel(Protocol):
    """Факторная риск-модель → ковариация активов Σ = B F Bᵀ + diag(D)."""

    def fit(self, returns: Panel) -> None:
        ...

    def exposures(self, asof_ms: int) -> pd.DataFrame:
        """B: factor exposures, index=symbol, columns=factors."""
        ...

    def factor_cov(self, asof_ms: int) -> pd.DataFrame:
        """F: ковариация факторов, index==columns==factors."""
        ...

    def specific_var(self, asof_ms: int) -> pd.Series:
        """D: идиосинкратическая дисперсия, index=symbol."""
        ...

    def cov(self, asof_ms: int) -> CovMatrix:
        """Σ: ковариация активов, index==columns==symbols (PSD)."""
        ...


@runtime_checkable
class PortfolioConstructor(Protocol):
    """μ + Σ + ограничения + текущие веса → целевые веса w*."""

    def solve(
        self,
        mu: pd.Series,
        cov: CovMatrix,
        current_w: pd.Series,
        constraints: Any = None,
        tcost_model: Any = None,
    ) -> pd.Series:
        ...


@runtime_checkable
class CrossSectionalStrategy(Protocol):
    """Полная cross-sectional стратегия: на дату t отдаёт целевые веса."""

    def target_weights(self, asof_ms: int) -> pd.Series:
        ...


# ---------------------------------------------------------------------------
# Panel helpers / validation
# ---------------------------------------------------------------------------
def is_panel(obj: Any) -> bool:
    """True, если ``obj`` — корректная Panel (MultiIndex (ts_ms, symbol))."""
    if not isinstance(obj, pd.DataFrame):
        return False
    idx = obj.index
    if not isinstance(idx, pd.MultiIndex) or idx.nlevels != 2:
        return False
    return tuple(idx.names) == PANEL_INDEX_NAMES


def validate_panel(panel: Any, *, allow_empty: bool = True) -> None:
    """Бросает ``ValueError``/``TypeError``, если ``panel`` нарушает контракт Panel."""
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(f"panel must be a pandas DataFrame, got {type(panel)!r}")
    idx = panel.index
    if not isinstance(idx, pd.MultiIndex) or idx.nlevels != 2:
        raise ValueError("panel must have a 2-level MultiIndex")
    if tuple(idx.names) != PANEL_INDEX_NAMES:
        raise ValueError(
            f"panel index names must be {PANEL_INDEX_NAMES}, got {tuple(idx.names)!r}"
        )
    if len(panel) == 0:
        if allow_empty:
            return
        raise ValueError("panel is empty")
    ts_values = idx.get_level_values(TS_LEVEL)
    if not pd.api.types.is_integer_dtype(ts_values.dtype):
        raise ValueError(f"{TS_LEVEL} level must be integer (ms), got {ts_values.dtype}")
    if idx.has_duplicates:
        raise ValueError("panel index has duplicate (ts_ms, symbol) entries")


def empty_panel(columns: Optional[Sequence[str]] = None) -> Panel:
    """Пустая, но структурно валидная Panel."""
    idx = pd.MultiIndex.from_arrays(
        [np.array([], dtype="int64"), np.array([], dtype=object)],
        names=PANEL_INDEX_NAMES,
    )
    return pd.DataFrame(index=idx, columns=list(columns or []))


def panel_symbols(panel: Panel) -> Sequence[str]:
    """Отсортированный уникальный список символов в панели."""
    return sorted(set(panel.index.get_level_values(SYMBOL_LEVEL)))


def panel_timestamps(panel: Panel) -> Sequence[int]:
    """Отсортированный уникальный список ts_ms в панели."""
    return sorted(set(int(t) for t in panel.index.get_level_values(TS_LEVEL)))


def cross_section(panel: Panel, asof_ms: int) -> pd.DataFrame:
    """Срез панели на дату ``asof_ms``: DataFrame с index=symbol, columns=features.

    Возвращает пустой DataFrame с теми же колонками, если даты нет.
    """
    try:
        slc = panel.xs(int(asof_ms), level=TS_LEVEL, drop_level=True)
    except KeyError:
        return pd.DataFrame(columns=panel.columns)
    if isinstance(slc, pd.Series):  # single row edge-case
        slc = slc.to_frame().T
    return slc


__all__ = [
    # modes
    "MODE_SINGLE_INSTRUMENT",
    "MODE_CROSS_SECTIONAL",
    "VALID_RUN_MODES",
    # index constants
    "PANEL_INDEX_NAMES",
    "TS_LEVEL",
    "SYMBOL_LEVEL",
    # aliases
    "Panel",
    "TargetWeights",
    "ExpectedReturns",
    "CovMatrix",
    # event
    "RebalanceEvent",
    # protocols
    "UniverseProvider",
    "Signal",
    "AlphaModel",
    "RiskModel",
    "PortfolioConstructor",
    "CrossSectionalStrategy",
    # helpers
    "is_panel",
    "validate_panel",
    "empty_panel",
    "panel_symbols",
    "panel_timestamps",
    "cross_section",
]
