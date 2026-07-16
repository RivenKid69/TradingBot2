# -*- coding: utf-8 -*-
"""
loaders/crypto_enrich.py
========================

Crypto free-обогатители (Stage D1) — «оживляют» BYO-сигналы funding_carry/basis/size на
бесплатных Binance-данных:

  * ``FundingEnricher`` — historical funding rate (Binance USDT-M perp,
    ``get_funding_rate_history``) → колонка ``funding_rate``, **PIT** as-of (funding
    наблюдаем в момент fundingTime → publish_lag=0, ``pit_quality='true'``);
  * ``BasisEnricher``  — spot-vs-perp базис → ``basis = perp_close/spot_close − 1``
    (оба close наблюдаемы на баре → PIT-true);
  * ``MarketCapEnricher`` — market cap → ``mcap`` (статич. снимок/coingecko free →
    ``pit_quality='approx'``; для истории подайте BYO history_fn).

Все провайдеры — DI (по умолчанию реальный Binance-адаптер, в тестах — фейки без сети).
Сбои сети мягко логируются (пустой провайдер → колонка NaN, сигнал нейтрален).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from core_portfolio import Panel, SYMBOL_LEVEL
from core_xs_data import PIT_APPROX, PIT_TRUE
from impl_data_sources import DataSourceMeta
from impl_panel import PanelBuilder
from service_xs_data import AsofEnricher, ColumnMapEnricher

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# default Binance providers (lazy, graceful)
# ---------------------------------------------------------------------------
def _binance_futures_adapter(config: Optional[Mapping[str, Any]] = None):
    from adapters.binance.futures_market_data import BinanceFuturesMarketDataAdapter
    from adapters.models import ExchangeVendor

    return BinanceFuturesMarketDataAdapter(vendor=ExchangeVendor.BINANCE, config=dict(config or {}))


def default_funding_history(symbol: str, *, limit: int = 1000) -> List[Any]:
    """Реальная история funding с Binance (graceful → [] при сбое)."""
    try:
        return _binance_futures_adapter().get_funding_rate_history(symbol, limit=limit) or []
    except Exception as exc:  # pragma: no cover - сеть/окружение
        logger.warning("default_funding_history(%s) failed: %s", symbol, exc)
        return []


def default_perp_closes(symbols: Sequence[str], timeframe: str, *, limit: int = 1000) -> pd.DataFrame:
    """Перп-close с Binance (long df[publish_ts, symbol, perp_close]); graceful."""
    rows = []
    try:
        adapter = _binance_futures_adapter()
    except Exception as exc:  # pragma: no cover
        logger.warning("perp adapter init failed: %s", exc)
        return pd.DataFrame(columns=["publish_ts", "symbol", "perp_close"])
    for s in symbols:
        try:
            bars = adapter.get_bars(s, timeframe, limit=limit) or []
        except Exception as exc:  # pragma: no cover
            logger.warning("perp get_bars(%s) failed: %s", s, exc)
            continue
        for b in bars:
            ts = getattr(b, "ts", None) or getattr(b, "timestamp", None)
            close = getattr(b, "close", None)
            if ts is not None and close is not None:
                rows.append({"publish_ts": int(ts), "symbol": s, "perp_close": float(close)})
    return pd.DataFrame(rows, columns=["publish_ts", "symbol", "perp_close"])


# ---------------------------------------------------------------------------
# Enrichers
# ---------------------------------------------------------------------------
class FundingEnricher(AsofEnricher):
    """funding_rate из истории Binance (PIT as-of, publish_lag=0)."""

    def __init__(self, *, history_fn: Optional[Callable[..., List[Any]]] = None,
                 vendor: str = "binance", publish_lag_ms: int = 0, limit: int = 1000) -> None:
        self._history_fn = history_fn or default_funding_history
        self._limit = int(limit)
        super().__init__(
            self._long_provider, columns=["funding_rate"], publish_ts_col="publish_ts",
            publish_lag_ms=publish_lag_ms,
            meta=DataSourceMeta(name="binance:funding", vendor=vendor, kind="enrich",
                                pit_quality=PIT_TRUE, notes="Historical funding (observable at fundingTime)."),
        )

    def _long_provider(self, symbols: Sequence[str]) -> pd.DataFrame:
        rows = []
        for s in symbols:
            try:
                hist = self._history_fn(s, limit=self._limit)
            except TypeError:
                hist = self._history_fn(s)
            for fp in hist or []:
                ts = getattr(fp, "timestamp_ms", None)
                rate = getattr(fp, "funding_rate", None)
                if ts is not None and rate is not None:
                    rows.append({"publish_ts": int(ts), "symbol": s, "funding_rate": float(rate)})
        return pd.DataFrame(rows, columns=["publish_ts", "symbol", "funding_rate"])


class BasisEnricher:
    """basis = perp_close / spot_close − 1 (оба наблюдаемы на баре → PIT-true)."""

    def __init__(self, *, perp_provider: Optional[Callable[..., pd.DataFrame]] = None,
                 timeframe: str = "1d", spot_col: str = "close", vendor: str = "binance",
                 limit: int = 1000) -> None:
        self._perp_provider = perp_provider or default_perp_closes
        self.timeframe = timeframe
        self.spot_col = spot_col
        self._limit = int(limit)
        self.meta = DataSourceMeta(name="binance:basis", vendor=vendor, kind="enrich",
                                   pit_quality=PIT_TRUE, notes="Spot-perp basis (both closes observable).")

    def columns(self) -> List[str]:
        return ["basis"]

    def enrich(self, panel: Panel) -> Panel:
        symbols = list(pd.unique(panel.index.get_level_values(SYMBOL_LEVEL)))
        try:
            long = self._perp_provider(symbols, self.timeframe, limit=self._limit)
        except TypeError:
            long = self._perp_provider(symbols, self.timeframe)
        if long is None or len(long) == 0 or self.spot_col not in panel.columns:
            out = panel.copy(); out["basis"] = np.nan
            return out
        joined = PanelBuilder.asof_join(panel, long, value_cols=["perp_close"],
                                        ts_col="publish_ts", symbol_col="symbol", publish_lag_ms=0)
        joined["basis"] = joined["perp_close"].astype("float64") / joined[self.spot_col].astype("float64") - 1.0
        return joined.drop(columns=["perp_close"], errors="ignore")


class MarketCapEnricher:
    """mcap: статич. снимок (pit=approx) ИЛИ PIT-история через history_fn (publish_ts)."""

    def __init__(self, mcaps: Optional[Mapping[str, float]] = None, *,
                 history_fn: Optional[Callable[[Sequence[str]], pd.DataFrame]] = None,
                 vendor: str = "static", publish_lag_ms: int = 0) -> None:
        self._mcaps = mcaps
        self._history_fn = history_fn
        if history_fn is not None:
            self._impl = AsofEnricher(history_fn, columns=["mcap"], publish_ts_col="publish_ts",
                                      publish_lag_ms=publish_lag_ms,
                                      meta=DataSourceMeta(name="mcap:history", vendor=vendor, kind="enrich",
                                                          pit_quality=PIT_TRUE, notes="Historical market cap (PIT)."))
        else:
            self._impl = ColumnMapEnricher(mcaps or {}, "mcap", vendor=vendor, pit_quality=PIT_APPROX,
                                           name="mcap:snapshot", notes="Market cap snapshot (no history).")
        self.meta = self._impl.meta

    def columns(self) -> List[str]:
        return ["mcap"]

    def enrich(self, panel: Panel) -> Panel:
        return self._impl.enrich(panel)


# Имена для реестра build_enrichers (D1).
CRYPTO_ENRICHERS = ("funding", "basis", "mcap")


def build_crypto_enricher(name: str, cfg: Any) -> Optional[Any]:
    """Сконструировать crypto-обогатитель по имени (из cfg). None если данных нет."""
    if name == "funding":
        return FundingEnricher()
    if name == "basis":
        return BasisEnricher(timeframe=getattr(cfg.data, "timeframe", "1d"))
    if name == "mcap":
        if getattr(cfg, "mcaps", None):
            return MarketCapEnricher(cfg.mcaps)
        return None
    return None


__all__ = [
    "FundingEnricher", "BasisEnricher", "MarketCapEnricher",
    "default_funding_history", "default_perp_closes",
    "CRYPTO_ENRICHERS", "build_crypto_enricher",
]
