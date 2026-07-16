# -*- coding: utf-8 -*-
"""
core_xs_data.py
===============

Контракты слоя сборки данных cross-sectional контура (Stage D0): провенанс колонок и
honest **Data-Quality** отчёт. База для `service_xs_data.DataAssembler` и (позже, D7)
для Data-Trust gate.

Принцип: каждая колонка панели несёт **происхождение** (источник/вендор/`pit_quality`),
а отчёт агрегирует покрытие/пропуски/устаревание/survivorship — то, что про-кванты
смотрят ПЕРЕД сигналами (look-ahead/PIT-дисциплина отличает институционал от любителя).

Слой ``core_`` (без зависимостей от impl/service).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# pit_quality значения (согласованы с impl_data_sources.VALID_PIT_QUALITY)
PIT_TRUE = "true"      # настоящий point-in-time (backtest честный)
PIT_APPROX = "approx"  # приблизительный (лаг/прокси/допущения)
PIT_NONE = "none"      # снимок/синтетика (НЕ backtest-safe)
VALID_PIT = (PIT_TRUE, PIT_APPROX, PIT_NONE)

_PIT_RANK = {PIT_TRUE: 2, PIT_APPROX: 1, PIT_NONE: 0}


@dataclass(frozen=True)
class ColumnProvenance:
    """Происхождение одной колонки панели."""

    column: str
    source: str                      # имя источника/обогатителя
    vendor: str                      # вендор ('binance'/'yahoo'/'byo'/'synthetic'/…)
    pit_quality: str = PIT_TRUE      # true | approx | none
    free: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if self.pit_quality not in VALID_PIT:
            raise ValueError(f"pit_quality must be one of {VALID_PIT}, got {self.pit_quality!r}")

    def to_dict(self) -> Dict[str, object]:
        return {
            "column": self.column, "source": self.source, "vendor": self.vendor,
            "pit_quality": self.pit_quality, "free": self.free, "notes": self.notes,
        }


@dataclass
class DataQualityReport:
    """Агрегированное качество собранной панели (honest, для UI/логов/gate)."""

    n_rows: int
    n_symbols: int
    first_ts_ms: Optional[int]
    last_ts_ms: Optional[int]
    columns: List[ColumnProvenance] = field(default_factory=list)
    coverage: Dict[str, float] = field(default_factory=dict)          # column → доля non-NaN
    per_symbol_coverage: Dict[str, float] = field(default_factory=dict)
    staleness_ms: Optional[int] = None                                # now − last_ts
    survivorship_biased: Optional[bool] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def worst_pit(self) -> str:
        if not self.columns:
            return PIT_TRUE
        return min((c.pit_quality for c in self.columns), key=lambda p: _PIT_RANK[p])

    @property
    def min_coverage(self) -> float:
        return min(self.coverage.values()) if self.coverage else 1.0

    def verdict(self) -> str:
        """ok | warn | poor — грубая сводка качества (D7 расширит до Data-Trust)."""
        if self.min_coverage < 0.2 or self.n_rows == 0:
            return "poor"
        if (self.worst_pit == PIT_NONE) or self.min_coverage < 0.5 or self.survivorship_biased:
            return "warn"
        return "ok"

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_rows": int(self.n_rows),
            "n_symbols": int(self.n_symbols),
            "first_ts_ms": self.first_ts_ms,
            "last_ts_ms": self.last_ts_ms,
            "columns": [c.to_dict() for c in self.columns],
            "coverage": {k: float(v) for k, v in self.coverage.items()},
            "per_symbol_coverage": {k: float(v) for k, v in self.per_symbol_coverage.items()},
            "staleness_ms": self.staleness_ms,
            "survivorship_biased": self.survivorship_biased,
            "worst_pit": self.worst_pit,
            "min_coverage": float(self.min_coverage),
            "verdict": self.verdict(),
            "warnings": list(self.warnings),
        }


__all__ = [
    "PIT_TRUE", "PIT_APPROX", "PIT_NONE", "VALID_PIT",
    "ColumnProvenance", "DataQualityReport",
]
