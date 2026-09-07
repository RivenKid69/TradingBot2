# -*- coding: utf-8 -*-
"""
loaders/futures_enrich.py
=========================

Futures free/BYO data (Stage D4) — диверсифицированный CTA на бесплатных continuous-прокси
или точных roll-accurate сериях:

  * ``ContinuousProxySource`` — PriceSource поверх yahoo (``ES=F``/``NQ=F``/…): УЖЕ back-
    adjusted continuous прокси НЕИЗВЕСТНЫМ методом → ``pit_quality='approx'`` (честно).
    Транслирует cross-sectional символы (ES/NQ/CL) в yahoo-тикеры и обратно.
  * ``build_roll_accurate_panel`` — BYO контракты → точная back-adjusted серия через
    ``impl_continuous_futures.build_continuous_panel(method=ratio|diff)`` → ``pit='true'``
    + continuous-meta (метод/n_rolls) для UI-индикатора.
  * ``CarryEnricher`` — front/back контракты → ``front``/``back``/``carry``/``roll_yield``
    (оба наблюдаемы → PIT-true). BYO-провайдер (free continuous-прокси не несёт contango).
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

# CME-символ → yahoo continuous-прокси тикер.
DEFAULT_CME_YAHOO_MAP = {
    "ES": "ES=F",
    "NQ": "NQ=F",
    "YM": "YM=F",
    "RTY": "RTY=F",
    "ZN": "ZN=F",
    "ZB": "ZB=F",
    "ZF": "ZF=F",
    "ZT": "ZT=F",
    "CL": "CL=F",
    "NG": "NG=F",
    "RB": "RB=F",
    "HO": "HO=F",
    "GC": "GC=F",
    "SI": "SI=F",
    "HG": "HG=F",
    "6E": "6E=F",
    "6J": "6J=F",
    "6B": "6B=F",
    "6A": "6A=F",
    "ZC": "ZC=F",
    "ZS": "ZS=F",
    "ZW": "ZW=F",
}


class ContinuousProxySource:
    """Free continuous-прокси (yahoo ES=F…) как PriceSource. pit_quality=approx (honest)."""

    def __init__(
        self,
        *,
        vendor: str = "yahoo",
        ticker_map: Optional[Mapping[str, str]] = None,
        inner: Any = None,
    ) -> None:
        self._map = dict(ticker_map or DEFAULT_CME_YAHOO_MAP)
        self._inner = inner or AdapterPriceSource(
            vendor=vendor, pit_quality=PIT_APPROX, name=f"free:{vendor}:continuous"
        )
        self.meta = DataSourceMeta(
            name="continuous-proxy",
            vendor=vendor,
            kind="price",
            pit_quality=PIT_APPROX,
            notes="Back-adjusted continuous proxy (yahoo, unknown roll method).",
        )

    def _ticker(self, symbol: str) -> str:
        return self._map.get(symbol, symbol)

    def get_bars(
        self,
        symbols: Sequence[str],
        timeframe: str,
        *,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            tk = self._ticker(sym)
            try:
                fetched = self._inner.get_bars(
                    [tk], timeframe, start_ms=start_ms, end_ms=end_ms, limit=limit
                )
            except Exception as exc:  # pragma: no cover - сеть
                logger.warning("ContinuousProxySource get_bars(%s→%s) failed: %s", sym, tk, exc)
                continue
            frame = fetched.get(tk)
            if frame is not None and len(frame):
                frame = frame.copy()
                frame["symbol"] = sym  # вернуть исходный cross-sectional символ
                out[sym] = frame
        return out


def build_roll_accurate_panel(
    contract_map: Dict[str, Sequence[Tuple[int, pd.Series]]],
    *,
    method: str = "ratio",
) -> Tuple[Panel, Dict[str, Any]]:
    """BYO контракты → точная back-adjusted непрерывная панель (reuse impl_continuous_futures)."""
    from impl_continuous_futures import build_continuous_panel

    return build_continuous_panel(contract_map, method=method, pit_quality=PIT_TRUE)


class CarryEnricher:
    """front/back → front/back/carry/roll_yield (carry = (front−back)/back). BYO-провайдер, PIT-true."""

    def __init__(
        self,
        *,
        fb_provider: Callable[[Sequence[str]], pd.DataFrame],
        publish_lag_ms: int = 0,
        vendor: str = "byo",
    ) -> None:
        self._fb_provider = fb_provider
        self._publish_lag_ms = int(publish_lag_ms)
        self.meta = DataSourceMeta(
            name="carry",
            vendor=vendor,
            kind="enrich",
            pit_quality=PIT_TRUE,
            notes="Front/back basis → carry/roll-yield (both observable).",
        )

    def columns(self) -> List[str]:
        return ["front", "back", "carry", "roll_yield"]

    def enrich(self, panel: Panel) -> Panel:
        symbols = list(pd.unique(panel.index.get_level_values(SYMBOL_LEVEL)))
        try:
            long = self._fb_provider(symbols)
        except Exception as exc:  # pragma: no cover
            logger.warning("CarryEnricher fb_provider failed: %s", exc)
            long = None
        if long is None or len(long) == 0:
            out = panel.copy()
            for c in self.columns():
                out[c] = np.nan
            return out
        out = PanelBuilder.asof_join(
            panel,
            long,
            value_cols=["front", "back"],
            ts_col="publish_ts",
            symbol_col="symbol",
            publish_lag_ms=self._publish_lag_ms,
        )
        back = out["back"].astype("float64").replace(0.0, np.nan)
        out["carry"] = (out["front"].astype("float64") - back) / back
        out["roll_yield"] = out["carry"]
        return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
FUTURES_ENRICHERS = ("carry",)


def build_futures_enricher(name: str, cfg: Any) -> Optional[Any]:
    """Сконструировать futures-обогатитель по имени. carry = BYO-only (free прокси без contango)."""
    if name == "carry":
        # carry требует front/back контрактов (BYO) — без провайдера честно пропускаем
        return None
    return None


__all__ = [
    "DEFAULT_CME_YAHOO_MAP",
    "ContinuousProxySource",
    "build_roll_accurate_panel",
    "CarryEnricher",
    "FUTURES_ENRICHERS",
    "build_futures_enricher",
]
