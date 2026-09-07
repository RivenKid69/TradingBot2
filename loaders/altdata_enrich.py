# -*- coding: utf-8 -*-
"""
loaders/altdata_enrich.py
=========================

Alt-data обогатители пайплайна (P2): **COT** (positioning крупных спекулянтов, CFTC) и
**economic calendar** (флаг близкого high-impact события). «Оживляют» сигналы ``cot`` и
календарные фильтры. PIT-честно: COT публикуется с лагом (release ~пятница для отчёта
по вторнику) → publish_lag; календарь — анонсированные даты (PIT-approx, как earnings).

DI/BYO: данные подаются провайдером/parquet (default — существующий
``data/forex/calendar/economic_calendar.parquet``; COT — BYO long parquet). Нет данных →
сигнал нейтрален. Слой loaders (интеграция).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from core_portfolio import SYMBOL_LEVEL, TS_LEVEL, Panel
from core_xs_data import PIT_APPROX, PIT_TRUE
from impl_data_sources import DataSourceMeta
from impl_panel import PanelBuilder
from service_xs_data import AsofEnricher

logger = logging.getLogger(__name__)

_DEFAULT_COT_PATH = "data/cot/cot_positioning.parquet"
_DEFAULT_CAL_PATH = "data/forex/calendar/economic_calendar.parquet"


# ---------------------------------------------------------------------------
# COT positioning (as-of, publish-lag → PIT)
# ---------------------------------------------------------------------------
class COTEnricher(AsofEnricher):
    """COT net-positioning как колонка ``cot_net`` через as-of join с publish-lag."""

    def __init__(
        self,
        provider: Callable[[Sequence[str]], pd.DataFrame],
        *,
        value_col: str = "cot_net",
        publish_lag_days: int = 3,
        pit_quality: str = PIT_TRUE,
    ) -> None:
        # ВАЖНО: не использовать имя ``self._provider`` — оно занято AsofEnricher
        # (иначе self._long вызовет сам себя → бесконечная рекурсия).
        self._cot_provider = provider
        super().__init__(
            self._long,
            columns=[value_col],
            publish_ts_col="publish_ts",
            publish_lag_ms=int(publish_lag_days) * 86_400_000,
            meta=DataSourceMeta(
                name="cot",
                vendor="cftc",
                kind="enrich",
                pit_quality=pit_quality,
                notes="CFTC COT net positioning (publish-lag → PIT).",
            ),
        )

    def _long(self, symbols: Sequence[str]) -> pd.DataFrame:
        try:
            df = self._cot_provider(symbols)
        except Exception as exc:  # pragma: no cover
            logger.warning("COTEnricher provider failed: %s", exc)
            df = None
        cols = ["publish_ts", "symbol", "cot_net"]
        return df if (df is not None and len(df)) else pd.DataFrame(columns=cols)


def _parquet_cot_provider(
    path: str, *, symbol_col: str = "symbol", value_col: str = "cot_net", ts_col: str = "publish_ts"
):
    def provider(symbols: Sequence[str]) -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame(columns=["publish_ts", "symbol", "cot_net"])
        df = pd.read_parquet(path)
        out = pd.DataFrame(
            {
                "publish_ts": (
                    PanelBuilder.normalize_ts(df[ts_col])
                    if hasattr(PanelBuilder, "normalize_ts")
                    else pd.to_datetime(df[ts_col]).astype("int64") // 10**6
                ),
                "symbol": df[symbol_col].astype(str),
                "cot_net": df[value_col].astype("float64"),
            }
        )
        return out[out["symbol"].isin(list(symbols))].reset_index(drop=True)

    return provider


def make_cot_enricher(
    *,
    path: Optional[str] = None,
    provider: Optional[Callable[[Sequence[str]], pd.DataFrame]] = None,
    publish_lag_days: int = 3,
) -> COTEnricher:
    prov = provider or _parquet_cot_provider(path or _DEFAULT_COT_PATH)
    return COTEnricher(prov, publish_lag_days=publish_lag_days)


# ---------------------------------------------------------------------------
# Economic calendar (high-impact-soon flag, PIT-approx)
# ---------------------------------------------------------------------------
class EconCalendarEnricher:
    """Флаг ``high_impact_soon``: high-impact событие по валюте символа в ближайшие
    ``window_days`` (анонсированный календарь, PIT-approx)."""

    def __init__(
        self,
        events: pd.DataFrame,
        *,
        currency_map: Optional[Dict[str, str]] = None,
        window_days: int = 2,
        out_col: str = "high_impact_soon",
    ) -> None:
        self.events = events
        self.currency_map = currency_map or {}
        self.window_ms = int(window_days) * 86_400_000
        self.out_col = out_col
        self.meta = DataSourceMeta(
            name="econ_calendar",
            vendor="calendar",
            kind="enrich",
            pit_quality=PIT_APPROX,
            notes="High-impact event proximity flag (announced calendar).",
        )

    def columns(self) -> List[str]:
        return [self.out_col]

    def _ccy(self, symbol: str) -> str:
        if symbol in self.currency_map:
            return self.currency_map[symbol]
        # forex-пары EUR_USD/EURUSD → берём quote (вторую валюту); иначе сам символ
        s = symbol.replace("/", "_")
        if "_" in s:
            return s.split("_")[1]
        if len(s) == 6:
            return s[3:]
        return s

    def enrich(self, panel: Panel) -> Panel:
        out = panel.copy()
        if self.events is None or not len(self.events):
            out[self.out_col] = 0.0
            return out
        ev = self.events
        # robust → ms независимо от разрешения datetime (pandas 3.0: ns/ms/us)
        ev_ts = pd.Series(
            pd.to_datetime(ev["timestamp"], errors="coerce")
            .values.astype("datetime64[ms]")
            .astype("int64"),
            index=ev.index,
        )
        impact = ev.get("impact", pd.Series(["High"] * len(ev))).astype(str).str.lower()
        ccy = ev.get("currency", pd.Series([""] * len(ev))).astype(str)
        high = impact.str.contains("high")
        by_ccy: Dict[str, np.ndarray] = {}
        for c in ccy[high].unique():
            by_ccy[c] = np.sort(ev_ts[high & (ccy == c)].to_numpy())
        flag = pd.Series(0.0, index=out.index)
        for sym, g in out.groupby(level=SYMBOL_LEVEL, sort=False):
            arr = by_ccy.get(self._ccy(str(sym)))
            if arr is None or not len(arr):
                continue
            ts = g.index.get_level_values(TS_LEVEL).to_numpy()
            vals = np.zeros(len(ts))
            for i, t in enumerate(ts):
                if np.any((arr > t) & (arr <= t + self.window_ms)):
                    vals[i] = 1.0
            flag.loc[g.index] = vals
        out[self.out_col] = flag
        return out


def make_econ_calendar_enricher(
    *,
    path: Optional[str] = None,
    events: Optional[pd.DataFrame] = None,
    currency_map: Optional[Dict[str, str]] = None,
    window_days: int = 2,
) -> EconCalendarEnricher:
    if events is None:
        p = path or _DEFAULT_CAL_PATH
        events = pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()
    return EconCalendarEnricher(events, currency_map=currency_map, window_days=window_days)


ALTDATA_ENRICHERS = ("cot", "econ_calendar")


def build_altdata_enricher(name: str, cfg: Any) -> Optional[Any]:
    if name == "cot":
        path = getattr(cfg.data, "cot_path", None) if hasattr(cfg, "data") else None
        return make_cot_enricher(path=path)
    if name == "econ_calendar":
        path = getattr(cfg.data, "calendar_path", None) if hasattr(cfg, "data") else None
        return make_econ_calendar_enricher(path=path)
    return None


__all__ = [
    "COTEnricher",
    "EconCalendarEnricher",
    "make_cot_enricher",
    "make_econ_calendar_enricher",
    "ALTDATA_ENRICHERS",
    "build_altdata_enricher",
]
