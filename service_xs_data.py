# -*- coding: utf-8 -*-
"""
service_xs_data.py
==================

Unified data-assembly слой (Stage D0): один оркестратор
``price source + enrichment sources → собранная панель + honest DataQualityReport``.

Про-пайплайн: ``prices → enrichment (funding/fundamentals/rates/IV) joined POINT-IN-TIME
с publish-lag → data-quality``. D0 даёт каркас; конкретные вендорные обогатители
(binance funding, yahoo PIT-фундаментал, oanda rates, deribit IV) добавляются в D1-D5 как
``Enricher``-плагины. Каждая колонка несёт провенанс/``pit_quality`` → честный отчёт.

Слой ``service_`` (зависит от core_/impl_). Сетевых вызовов нет на импорте; источники и
обогатители подаются как объекты (DI) — тестируется без сети.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from core_portfolio import Panel, SYMBOL_LEVEL, TS_LEVEL
from core_xs_data import (
    PIT_APPROX, PIT_NONE, PIT_TRUE, ColumnProvenance, DataQualityReport,
)
from impl_data_sources import DataSourceMeta, PriceSource
from impl_data_cache import ParquetCache
from impl_panel import PanelBuilder

logger = logging.getLogger(__name__)

PRICE_COLS = ("open", "high", "low", "close", "volume")


# ---------------------------------------------------------------------------
# Enricher protocol + generic implementations
# ---------------------------------------------------------------------------
@runtime_checkable
class Enricher(Protocol):
    meta: DataSourceMeta

    def columns(self) -> List[str]:
        """Колонки, которые добавляет обогатитель."""
        ...

    def enrich(self, panel: Panel) -> Panel:
        """Вернуть панель с добавленными колонками (PIT-безопасно)."""
        ...


class FunctionEnricher:
    """DI/тестовый обогатитель из произвольной функции ``fn(panel)->panel``."""

    def __init__(self, fn: Callable[[Panel], Panel], *, columns: Sequence[str],
                 meta: Optional[DataSourceMeta] = None, name: str = "fn") -> None:
        self._fn = fn
        self._cols = list(columns)
        self.meta = meta or DataSourceMeta(name=name, vendor="byo", kind="enrich", pit_quality=PIT_TRUE)

    def columns(self) -> List[str]:
        return list(self._cols)

    def enrich(self, panel: Panel) -> Panel:
        return self._fn(panel)


class ColumnMapEnricher:
    """Статическая карта ``{symbol -> скаляр}`` → бродкаст колонкой (например mcap).

    Без истории → ``pit_quality='approx'`` по умолчанию (значение «как сейчас»).
    """

    def __init__(self, values: Mapping[str, float], column: str, *,
                 vendor: str = "byo", pit_quality: str = PIT_APPROX,
                 name: Optional[str] = None, notes: str = "") -> None:
        self.values = {str(k): float(v) for k, v in (values or {}).items()}
        self.column = column
        self.meta = DataSourceMeta(
            name=name or f"static:{column}", vendor=vendor, kind="enrich",
            pit_quality=pit_quality, notes=notes or "Static per-symbol value (no history).",
        )

    def columns(self) -> List[str]:
        return [self.column]

    def enrich(self, panel: Panel) -> Panel:
        out = panel.copy()
        syms = out.index.get_level_values(SYMBOL_LEVEL)
        out[self.column] = [self.values.get(str(s), np.nan) for s in syms]
        return out


class AsofEnricher:
    """PIT-обогатитель: длинный кадр ``(publish_ts, symbol, <values>)`` → as-of join.

    ``long_provider(symbols) -> DataFrame`` отдаёт записи с публикационным таймстемпом;
    join строго ``publish_ts + publish_lag_ms <= ts`` (анти-look-ahead через
    ``PanelBuilder.asof_join``). Это рабочая лошадка для funding/fundamentals/rates.
    """

    def __init__(self, long_provider: Callable[[Sequence[str]], pd.DataFrame], *,
                 columns: Sequence[str], publish_ts_col: str = "publish_ts",
                 publish_lag_ms: int = 0, meta: Optional[DataSourceMeta] = None,
                 name: str = "asof", vendor: str = "byo", pit_quality: str = PIT_TRUE) -> None:
        self._provider = long_provider
        self._cols = list(columns)
        self.publish_ts_col = publish_ts_col
        self.publish_lag_ms = int(publish_lag_ms)
        self.meta = meta or DataSourceMeta(
            name=name, vendor=vendor, kind="enrich", pit_quality=pit_quality,
            notes=f"As-of joined with publish_lag={publish_lag_ms}ms (PIT-safe).",
        )

    def columns(self) -> List[str]:
        return list(self._cols)

    def enrich(self, panel: Panel) -> Panel:
        symbols = list(pd.unique(panel.index.get_level_values(SYMBOL_LEVEL)))
        long = self._provider(symbols)
        if long is None or len(long) == 0:
            logger.warning("AsofEnricher '%s': empty provider → columns stay NaN", self.meta.name)
            out = panel.copy()
            for c in self._cols:
                if c not in out.columns:
                    out[c] = np.nan
            return out
        return PanelBuilder.asof_join(
            panel, long, value_cols=self._cols, ts_col=self.publish_ts_col,
            symbol_col="symbol", publish_lag_ms=self.publish_lag_ms,
        )


# ---------------------------------------------------------------------------
# Quality report builder
# ---------------------------------------------------------------------------
def build_quality_report(
    panel: Panel,
    provenance: Sequence[ColumnProvenance],
    *,
    price_col: str = "close",
    now_ms: Optional[int] = None,
    survivorship_biased: Optional[bool] = None,
) -> DataQualityReport:
    """Собрать honest DataQualityReport из панели + провенанса колонок."""
    if panel is None or len(panel) == 0:
        return DataQualityReport(0, 0, None, None, list(provenance), survivorship_biased=survivorship_biased,
                                 warnings=["empty panel"])
    ts = panel.index.get_level_values(TS_LEVEL)
    syms = panel.index.get_level_values(SYMBOL_LEVEL)
    first_ts, last_ts = int(ts.min()), int(ts.max())
    coverage = {c: float(panel[c].notna().mean()) for c in panel.columns}
    # per-symbol покрытие по цене
    per_sym: Dict[str, float] = {}
    if price_col in panel.columns:
        g = panel[price_col].groupby(level=SYMBOL_LEVEL).apply(lambda s: float(s.notna().mean()))
        per_sym = {str(k): float(v) for k, v in g.items()}
    staleness = (int(now_ms) - last_ts) if now_ms is not None else None

    warnings: List[str] = []
    for p in provenance:
        if p.pit_quality == PIT_NONE:
            warnings.append(f"column '{p.column}' is pit_quality=none ({p.source}) — НЕ backtest-safe")
    low = [c for c, v in coverage.items() if v < 0.5]
    if low:
        warnings.append(f"low coverage (<50%): {', '.join(sorted(low))}")
    if survivorship_biased:
        warnings.append("universe survivorship-biased")

    return DataQualityReport(
        n_rows=int(len(panel)), n_symbols=int(pd.unique(syms).size),
        first_ts_ms=first_ts, last_ts_ms=last_ts,
        columns=list(provenance), coverage=coverage, per_symbol_coverage=per_sym,
        staleness_ms=staleness, survivorship_biased=survivorship_biased, warnings=warnings,
    )


# ---------------------------------------------------------------------------
# DataAssembler
# ---------------------------------------------------------------------------
@dataclass
class AssembleResult:
    panel: Panel
    report: DataQualityReport


class DataAssembler:
    """Оркестратор: price source (+ кэш) → enrichers → (Panel, DataQualityReport)."""

    def __init__(
        self,
        price_source: PriceSource,
        *,
        enrichers: Optional[Sequence[Enricher]] = None,
        cache: Optional[ParquetCache] = None,
        cache_ttl_ms: Optional[int] = None,
    ) -> None:
        self.price_source = price_source
        self.enrichers = list(enrichers or [])
        self.cache = cache
        self.cache_ttl_ms = cache_ttl_ms

    def _load_prices(self, symbols: Sequence[str], timeframe: str, *,
                     start_ms, end_ms, limit, now_ms) -> Dict[str, pd.DataFrame]:
        vendor = getattr(self.price_source.meta, "vendor", "na")
        frames: Dict[str, pd.DataFrame] = {}
        missing: List[str] = []
        for sym in symbols:
            if self.cache is not None:
                cached = self.cache.get(vendor, sym, timeframe, ttl_ms=self.cache_ttl_ms, now_ms=now_ms)
                if cached is not None and len(cached):
                    frames[sym] = cached
                    continue
            missing.append(sym)
        if missing:
            fetched = self.price_source.get_bars(missing, timeframe, start_ms=start_ms, end_ms=end_ms, limit=limit)
            for sym, df in fetched.items():
                frames[sym] = df
                if self.cache is not None:
                    self.cache.put(vendor, sym, timeframe, df)
        return frames

    def assemble(
        self,
        symbols: Sequence[str],
        timeframe: str,
        *,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = 1000,
        now_ms: Optional[int] = None,
        price_col: str = "close",
        survivorship_biased: Optional[bool] = None,
    ) -> AssembleResult:
        frames = self._load_prices(symbols, timeframe, start_ms=start_ms, end_ms=end_ms,
                                   limit=limit, now_ms=now_ms)
        panel = PanelBuilder.from_frames(frames)

        prov: List[ColumnProvenance] = []
        pm = self.price_source.meta
        for c in panel.columns:
            if c in PRICE_COLS:
                prov.append(ColumnProvenance(c, pm.name, pm.vendor, pm.pit_quality, pm.free,
                                             "OHLCV from price source."))

        for e in self.enrichers:
            try:
                panel = e.enrich(panel)
            except Exception as exc:  # pragma: no cover - вендорный сбой не валит сборку
                logger.warning("enricher '%s' failed: %s", getattr(e.meta, "name", e), exc)
                continue
            em = e.meta
            for c in e.columns():
                if c in panel.columns:
                    prov.append(ColumnProvenance(c, em.name, em.vendor, em.pit_quality, em.free, em.notes))

        report = build_quality_report(panel, prov, price_col=price_col, now_ms=now_ms,
                                      survivorship_biased=survivorship_biased)
        return AssembleResult(panel=panel, report=report)


__all__ = [
    "Enricher", "FunctionEnricher", "ColumnMapEnricher", "AsofEnricher",
    "build_quality_report", "AssembleResult", "DataAssembler",
]
