# -*- coding: utf-8 -*-
"""
loaders/equity_enrich.py
========================

Equity free-обогатители (Stage D2) — «оживляют» value/quality сигналы и дают **честный
PIT-фундаментал** (то, что отличает институционал от любителя):

  * ``PITFundamentalsEnricher`` — фундаментал (earnings/book_value/fcf/roe) через as-of
    join с **publish-lag** (анти-look-ahead). Качество = качество источника:
      − BYO PIT parquet (``ParquetFundamentals``, есть ``publish_ts``) → ``pit_quality='true'``
        (backtest ЧЕСТНЫЙ);
      − free yfinance (``FreeFundamentals``) — СНИМОК «сейчас», ``pit_quality='none'``
        (НЕ backtest-safe, только live-screening; громко помечается в отчёте/UI).
  * ``TotalReturnEnricher`` — total-return цена (реинвест дивидендов + сплиты) через
    детерминированный ``total_return_index`` → колонка ``tr_close`` (free yahoo dividends →
    ``pit_quality='approx'``; BYO → можно ``true``).
  * ``EarningsEnricher`` — флаг ``has_earnings_soon`` (earnings в ближайшие N дней по
    анонсированному календарю; ``pit_quality='approx'``).

Провайдеры — DI (дефолт = yahoo/parquet, тесты = фейки без сети; сбои → graceful NaN).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core_portfolio import Panel, SYMBOL_LEVEL, TS_LEVEL
from core_xs_data import PIT_APPROX, PIT_NONE, PIT_TRUE
from impl_data_sources import DataSourceMeta, total_return_index
from impl_panel import PanelBuilder
from service_xs_data import AsofEnricher

logger = logging.getLogger(__name__)

DEFAULT_FUND_FIELDS = ("earnings", "book_value", "fcf", "roe")


# ---------------------------------------------------------------------------
# PIT fundamentals (headline)
# ---------------------------------------------------------------------------
class PITFundamentalsEnricher(AsofEnricher):
    """Фундаментал as-of (publish-lag) → PIT-true (BYO parquet) или none (free снимок)."""

    def __init__(self, source: Any, *, fields: Sequence[str] = DEFAULT_FUND_FIELDS,
                 publish_lag_ms: int = 0) -> None:
        self._source = source
        self._fields = list(fields)
        pit = getattr(getattr(source, "meta", None), "pit_quality", PIT_NONE)
        vendor = getattr(getattr(source, "meta", None), "vendor", "byo")
        notes = ("PIT fundamentals (publish_ts, backtest-safe)." if pit == PIT_TRUE
                 else "Snapshot fundamentals — NOT backtest-safe (live-screening only).")
        super().__init__(
            self._long_provider, columns=self._fields, publish_ts_col="publish_ts",
            publish_lag_ms=publish_lag_ms,
            meta=DataSourceMeta(name="fundamentals", vendor=vendor, kind="enrich",
                                pit_quality=pit, notes=notes),
        )

    def _long_provider(self, symbols: Sequence[str]) -> pd.DataFrame:
        try:
            df = self._source.get_fundamentals(symbols, self._fields)
        except Exception as exc:  # pragma: no cover - сеть/файл
            logger.warning("PITFundamentalsEnricher: source failed: %s", exc)
            return pd.DataFrame(columns=["publish_ts", "symbol"] + self._fields)
        return df if df is not None else pd.DataFrame(columns=["publish_ts", "symbol"] + self._fields)


def make_pit_fundamentals_enricher(
    *,
    parquet_path: Optional[str] = None,
    source: Any = None,
    fields: Sequence[str] = DEFAULT_FUND_FIELDS,
    publish_lag_ms: int = 0,
):
    """Фабрика: BYO parquet (PIT-true) если задан путь/источник, иначе free снимок (PIT-none)."""
    if source is None:
        if parquet_path:
            from impl_data_sources import ParquetFundamentals
            source = ParquetFundamentals(parquet_path)
        else:
            from impl_data_sources import FreeFundamentals
            source = FreeFundamentals()  # снимок, pit=none, громкое предупреждение
    return PITFundamentalsEnricher(source, fields=fields, publish_lag_ms=publish_lag_ms)


# ---------------------------------------------------------------------------
# Total return
# ---------------------------------------------------------------------------
class TotalReturnEnricher:
    """total-return цена (реинвест дивидендов + сплиты) → колонка ``out_col`` (default tr_close)."""

    def __init__(self, *, actions_fn: Optional[Callable[[str], Tuple[Dict[int, float], Dict[int, float]]]] = None,
                 close_col: str = "close", out_col: str = "tr_close",
                 vendor: str = "yahoo", pit_quality: str = PIT_APPROX) -> None:
        self._actions_fn = actions_fn or _default_yahoo_actions
        self.close_col = close_col
        self.out_col = out_col
        self.meta = DataSourceMeta(name="total_return", vendor=vendor, kind="enrich",
                                   pit_quality=pit_quality,
                                   notes="Total-return (reinvest dividends + splits) via total_return_index.")

    def columns(self) -> List[str]:
        return [self.out_col]

    def enrich(self, panel: Panel) -> Panel:
        out = panel.copy()
        if self.close_col not in out.columns:
            out[self.out_col] = np.nan
            return out
        tr = pd.Series(np.nan, index=out.index, dtype="float64")
        for sym, g in out.groupby(level=SYMBOL_LEVEL, sort=False):
            ts = g.index.get_level_values(TS_LEVEL)
            close = pd.Series(g[self.close_col].astype("float64").to_numpy(), index=ts)
            try:
                divs, splits = self._actions_fn(sym)
            except Exception as exc:  # pragma: no cover
                logger.warning("TotalReturnEnricher actions(%s) failed: %s", sym, exc)
                divs, splits = {}, {}
            tri = total_return_index(close, dividends=divs, splits=splits)
            tr.loc[g.index] = tri.to_numpy()
        out[self.out_col] = tr
        return out


def _ex_to_ms(ex: Any) -> Optional[int]:
    try:
        from impl_panel import normalize_ts_ms
        return int(normalize_ts_ms(pd.Series([ex]))[0])
    except Exception:
        return None


def _default_yahoo_actions(symbol: str) -> Tuple[Dict[int, float], Dict[int, float]]:  # pragma: no cover - сеть
    """Free дивиденды/сплиты через adapters/yahoo (graceful → пусто). ex_date(str)→ms."""
    try:
        from adapters.yahoo.corporate_actions import YahooCorporateActionsAdapter
        from adapters.models import ExchangeVendor
        ca = YahooCorporateActionsAdapter(vendor=ExchangeVendor.YAHOO)
        dmap: Dict[int, float] = {}
        for d in (ca.get_dividends(symbol) or []):
            ts = _ex_to_ms(getattr(d, "ex_date", None)); amt = getattr(d, "amount", None)
            if ts is not None and amt is not None:
                dmap[ts] = float(amt)
        smap: Dict[int, float] = {}
        try:
            for s in (ca.get_splits(symbol) or []):
                ts = _ex_to_ms(getattr(s, "ex_date", None)); af = getattr(s, "adjustment_factor", None)
                if ts is not None and af is not None:
                    smap[ts] = float(af)
        except Exception:
            pass
        return dmap, smap
    except Exception as exc:
        logger.warning("yahoo actions(%s) failed: %s", symbol, exc)
        return {}, {}


# ---------------------------------------------------------------------------
# Earnings flag
# ---------------------------------------------------------------------------
class EarningsEnricher:
    """Флаг ``has_earnings_soon``: earnings в ближайшие ``window_days`` (анонс. календарь)."""

    def __init__(self, *, dates_fn: Optional[Callable[[str], Sequence[int]]] = None,
                 window_days: int = 5, out_col: str = "has_earnings_soon",
                 vendor: str = "yahoo") -> None:
        self._dates_fn = dates_fn or _default_yahoo_earnings_dates
        self.window_ms = int(window_days) * 86_400_000
        self.out_col = out_col
        self.meta = DataSourceMeta(name="earnings", vendor=vendor, kind="enrich",
                                   pit_quality=PIT_APPROX,
                                   notes="Earnings-soon flag from announced calendar (timing approx).")

    def columns(self) -> List[str]:
        return [self.out_col]

    def enrich(self, panel: Panel) -> Panel:
        out = panel.copy()
        flag = pd.Series(0.0, index=out.index, dtype="float64")
        for sym, g in out.groupby(level=SYMBOL_LEVEL, sort=False):
            try:
                dates = sorted(int(d) for d in (self._dates_fn(sym) or []))
            except Exception as exc:  # pragma: no cover
                logger.warning("earnings dates(%s) failed: %s", sym, exc)
                dates = []
            if not dates:
                continue
            ts = g.index.get_level_values(TS_LEVEL).to_numpy()
            darr = np.array(dates, dtype="int64")
            vals = np.zeros(len(ts), dtype="float64")
            for i, t in enumerate(ts):
                # earnings в (t, t+window] — известно заранее из календаря (PIT-approx)
                if np.any((darr > t) & (darr <= t + self.window_ms)):
                    vals[i] = 1.0
            flag.loc[g.index] = vals
        out[self.out_col] = flag
        return out


def _default_yahoo_earnings_dates(symbol: str) -> List[int]:  # pragma: no cover - сеть
    try:
        from adapters.yahoo.earnings import YahooEarningsAdapter
        from adapters.models import ExchangeVendor
        ye = YahooEarningsAdapter(vendor=ExchangeVendor.YAHOO)
        hist = ye.get_earnings_history(symbol) or []
        out = []
        for e in hist:
            ts = (getattr(e, "timestamp_ms", None) or getattr(e, "date_ms", None)
                  or _ex_to_ms(getattr(e, "date", None) or getattr(e, "earnings_date", None)))
            if ts is not None:
                out.append(int(ts))
        return out
    except Exception as exc:
        logger.warning("yahoo earnings(%s) failed: %s", symbol, exc)
        return []


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
EQUITY_ENRICHERS = ("total_return", "pit_fundamentals", "edgar_fundamentals", "earnings")


def build_equity_enricher(name: str, cfg: Any) -> Optional[Any]:
    """Сконструировать equity-обогатитель по имени (из cfg). None если данных нет."""
    if name == "total_return":
        return TotalReturnEnricher()
    if name == "pit_fundamentals":
        path = getattr(cfg, "fundamentals_path", None)
        fields = getattr(cfg, "fundamentals_fields", None) or DEFAULT_FUND_FIELDS
        lag_days = int(getattr(cfg, "fundamentals_publish_lag_days", 0) or 0)
        return make_pit_fundamentals_enricher(
            parquet_path=path, fields=fields, publish_lag_ms=lag_days * 86_400_000)
    if name == "edgar_fundamentals":
        # Настоящий бесплатный PIT-фундаментал из SEC EDGAR (filing dates → pit=true).
        # Опциональный кэш-parquet ускоряет повторные прогоны и работает офлайн.
        from services.edgar_fundamentals import EdgarFundamentals
        cache = getattr(cfg, "fundamentals_path", None) or "data/fundamentals_edgar/edgar_pit.parquet"
        fields = getattr(cfg, "fundamentals_fields", None) or DEFAULT_FUND_FIELDS
        lag_days = int(getattr(cfg, "fundamentals_publish_lag_days", 0) or 0)
        src = EdgarFundamentals(cache_path=cache)
        return make_pit_fundamentals_enricher(
            source=src, fields=fields, publish_lag_ms=lag_days * 86_400_000)
    if name == "earnings":
        return EarningsEnricher()
    return None


__all__ = [
    "PITFundamentalsEnricher", "TotalReturnEnricher", "EarningsEnricher",
    "make_pit_fundamentals_enricher", "DEFAULT_FUND_FIELDS",
    "EQUITY_ENRICHERS", "build_equity_enricher",
]
