# -*- coding: utf-8 -*-
"""
loaders/forex_enrich.py
=======================

Forex free-обогатители (Stage D3) — «оживляют» fx_carry на бесплатных/практик-данных:

  * ``RateDiffEnricher`` — дифференциал процентных ставок: парсит пару base/quote
    (``EURUSD``/``EUR_USD``/``EUR/USD`` → EUR, USD), берёт ставки по валютам и пишет
    ``rate_base``/``rate_quote``/``rate_diff = rate_base − rate_quote``. Два режима:
      − **static snapshot** (карта ``{currency: rate}``, напр. policy rates G10) →
        ``pit_quality='approx'`` (без истории);
      − **history** (``history_fn(currencies) -> long[publish_ts, currency, rate]``) →
        as-of join с publish-lag → ``pit_quality='true'`` (PIT).
  * PPP/reer/terms-of-trade — BYO-колонки (honest, не free).

``oanda_price_source`` — тонкая обёртка (OANDA practice через registry; нужны
``OANDA_API_KEY``/``OANDA_ACCOUNT_ID``). Цены тянутся стандартным free-путём (D0).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core_portfolio import Panel, SYMBOL_LEVEL
from core_xs_data import PIT_APPROX, PIT_TRUE
from impl_data_sources import AdapterPriceSource, DataSourceMeta
from impl_panel import PanelBuilder

logger = logging.getLogger(__name__)


def oanda_price_source(**kwargs: Any) -> AdapterPriceSource:
    """Free прайс-источник OANDA practice (нужны OANDA_API_KEY/ACCOUNT_ID)."""
    return AdapterPriceSource(vendor="oanda", pit_quality=PIT_TRUE, name="free:oanda", **kwargs)


def parse_pair(symbol: str) -> Tuple[str, str]:
    """``EURUSD``/``EUR_USD``/``EUR/USD`` → ('EUR','USD'). Fallback: (symbol, '')."""
    s = str(symbol).replace("_", "").replace("/", "").upper()
    if len(s) >= 6:
        return s[:3], s[3:6]
    return s, ""


class RateDiffEnricher:
    """rate_base/rate_quote/rate_diff из ставок по валютам (static approx ИЛИ history PIT)."""

    def __init__(self, rates: Optional[Mapping[str, float]] = None, *,
                 history_fn: Optional[Callable[[Sequence[str]], pd.DataFrame]] = None,
                 publish_lag_ms: int = 0, vendor: str = "static") -> None:
        self._rates = {str(k).upper(): float(v) for k, v in (rates or {}).items()}
        self._history_fn = history_fn
        self._publish_lag_ms = int(publish_lag_ms)
        pit = PIT_TRUE if history_fn is not None else PIT_APPROX
        notes = ("Interest-rate differential (PIT history)." if history_fn is not None
                 else "Policy-rate snapshot (no history → approx).")
        self.meta = DataSourceMeta(name="rates", vendor=vendor, kind="enrich",
                                   pit_quality=pit, notes=notes)

    def columns(self) -> List[str]:
        return ["rate_base", "rate_quote", "rate_diff"]

    def enrich(self, panel: Panel) -> Panel:
        symbols = list(pd.unique(panel.index.get_level_values(SYMBOL_LEVEL)))
        pairs = {s: parse_pair(s) for s in symbols}
        if self._history_fn is not None:
            return self._enrich_history(panel, pairs)
        return self._enrich_static(panel, pairs)

    def _enrich_static(self, panel: Panel, pairs: Dict[str, Tuple[str, str]]) -> Panel:
        out = panel.copy()
        syms = out.index.get_level_values(SYMBOL_LEVEL)
        rb = np.array([self._rates.get(pairs[str(s)][0], np.nan) for s in syms], dtype="float64")
        rq = np.array([self._rates.get(pairs[str(s)][1], np.nan) for s in syms], dtype="float64")
        out["rate_base"] = rb
        out["rate_quote"] = rq
        out["rate_diff"] = rb - rq
        return out

    def _enrich_history(self, panel: Panel, pairs: Dict[str, Tuple[str, str]]) -> Panel:
        currencies = sorted({c for bq in pairs.values() for c in bq if c})
        try:
            rates_long = self._history_fn(currencies)
        except Exception as exc:  # pragma: no cover
            logger.warning("RateDiffEnricher history_fn failed: %s", exc)
            rates_long = pd.DataFrame(columns=["publish_ts", "currency", "rate"])
        if rates_long is None or len(rates_long) == 0:
            out = panel.copy()
            for c in self.columns():
                out[c] = np.nan
            return out
        base_rows, quote_rows = [], []
        for s, (b, q) in pairs.items():
            for _, r in rates_long[rates_long["currency"] == b].iterrows():
                base_rows.append({"publish_ts": int(r["publish_ts"]), "symbol": s, "rate_base": float(r["rate"])})
            for _, r in rates_long[rates_long["currency"] == q].iterrows():
                quote_rows.append({"publish_ts": int(r["publish_ts"]), "symbol": s, "rate_quote": float(r["rate"])})
        out = panel
        if base_rows:
            out = PanelBuilder.asof_join(out, pd.DataFrame(base_rows), value_cols=["rate_base"],
                                         ts_col="publish_ts", symbol_col="symbol", publish_lag_ms=self._publish_lag_ms)
        else:
            out = out.copy(); out["rate_base"] = np.nan
        if quote_rows:
            out = PanelBuilder.asof_join(out, pd.DataFrame(quote_rows), value_cols=["rate_quote"],
                                         ts_col="publish_ts", symbol_col="symbol", publish_lag_ms=self._publish_lag_ms)
        else:
            out["rate_quote"] = np.nan
        out["rate_diff"] = out["rate_base"].astype("float64") - out["rate_quote"].astype("float64")
        return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
FOREX_ENRICHERS = ("rate_diff",)


def build_forex_enricher(name: str, cfg: Any) -> Optional[Any]:
    """Сконструировать forex-обогатитель по имени (из cfg). None если ставок нет."""
    if name == "rate_diff":
        rates = getattr(cfg, "policy_rates", None)
        if rates:
            return RateDiffEnricher(rates)
        return None
    return None


__all__ = [
    "oanda_price_source", "parse_pair", "RateDiffEnricher",
    "FOREX_ENRICHERS", "build_forex_enricher",
]
