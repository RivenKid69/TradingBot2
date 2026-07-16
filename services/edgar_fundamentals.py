# -*- coding: utf-8 -*-
"""
services/edgar_fundamentals.py
==============================

**Настоящий point-in-time фундаментал из SEC EDGAR** (бесплатно, без ключей) —
закрывает equity-PIT блокер без покупки Sharadar/Compustat.

Почему это PIT-true (а не «снимок сейчас»): каждый факт XBRL в EDGAR несёт дату
ПОДАЧИ отчёта (``filed``). Значение становится публично известным ровно в этот
момент → ``publish_ts = filed`` даёт честный as-of join без look-ahead. Это та же
семантика, что у платных вендоров; интерфейс ``get_fundamentals`` идентичен
``impl_data_sources.ParquetFundamentals`` (drop-in: можно заменить источник, не
трогая пайплайн).

Покрытие/история у бесплатного EDGAR уже, чем у Sharadar, но качество PIT — такое же.
Купленный датасет подключается через тот же ``fundamentals_path`` parquet.

Выход (long-кадр для ``AsofEnricher``):
    publish_ts(ms) | symbol | earnings(EPS) | book_value(BVPS) | fcf(per share) | roe

Сеть инкапсулирована (DI): ``facts_fn``/``tickers_fn`` подменяются в тестах фейками.
SEC требует User-Agent (env ``SEC_EDGAR_USER_AGENT``) и лимит ~10 rps.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from core_xs_data import PIT_TRUE
from impl_data_sources import DataSourceMeta

logger = logging.getLogger(__name__)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
DEFAULT_UA = os.environ.get(
    "SEC_EDGAR_USER_AGENT", "RivenQuant research (contact: research@example.com)"
)

# XBRL-концепты → внутренние per-share / ratio поля.
# (namespace, concept, unit_hint)
_EPS_CONCEPTS = [("us-gaap", "EarningsPerShareDiluted", "USD/shares"),
                 ("us-gaap", "EarningsPerShareBasic", "USD/shares")]
_EQUITY_CONCEPTS = [("us-gaap", "StockholdersEquity", "USD"),
                    ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "USD")]
_SHARES_CONCEPTS = [("dei", "EntityCommonStockSharesOutstanding", "shares"),
                    ("us-gaap", "CommonStockSharesOutstanding", "shares"),
                    ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares")]
_NI_CONCEPTS = [("us-gaap", "NetIncomeLoss", "USD")]
_CFO_CONCEPTS = [("us-gaap", "NetCashProvidedByUsedInOperatingActivities", "USD"),
                 ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations", "USD")]
_CAPEX_CONCEPTS = [("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment", "USD")]


# ---------------------------------------------------------------------------
# Network (injectable)
# ---------------------------------------------------------------------------
def _http_json(url: str, *, ua: str = DEFAULT_UA, timeout: float = 20.0,
               retries: int = 3) -> Any:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua,
                                                       "Accept-Encoding": "gzip, deflate"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                enc = resp.headers.get("Content-Encoding", "")
                if "gzip" in enc:
                    import gzip
                    raw = gzip.decompress(raw)
                return json.loads(raw.decode("utf-8"))
        except Exception as exc:  # pragma: no cover - сеть
            last = exc
            time.sleep(0.4 * (i + 1))
    raise RuntimeError(f"SEC request failed: {url}: {last}")


def default_tickers_fn(ua: str = DEFAULT_UA) -> Dict[str, int]:
    """{TICKER -> CIK} из SEC company_tickers.json."""
    data = _http_json(SEC_TICKERS_URL, ua=ua)
    out: Dict[str, int] = {}
    items = data.values() if isinstance(data, dict) else data
    for it in items:
        try:
            out[str(it["ticker"]).upper()] = int(it["cik_str"])
        except Exception:
            continue
    return out


def default_facts_fn(cik: int, ua: str = DEFAULT_UA) -> Dict[str, Any]:
    """companyfacts JSON по CIK (с rate-limit паузой)."""
    time.sleep(0.15)  # SEC ~10 rps
    return _http_json(SEC_FACTS_URL.format(cik=int(cik)), ua=ua)


# ---------------------------------------------------------------------------
# XBRL parsing
# ---------------------------------------------------------------------------
def _facts_for(facts_json: Mapping[str, Any], concepts) -> List[Dict[str, Any]]:
    """Все наблюдения для первого доступного концепта из списка (с filed/end/accn/val)."""
    facts = facts_json.get("facts", {})
    for ns, concept, unit_hint in concepts:
        node = facts.get(ns, {}).get(concept)
        if not node:
            continue
        units = node.get("units", {})
        # выбрать подходящий unit (по hint, иначе первый)
        unit_key = unit_hint if unit_hint in units else (next(iter(units), None))
        if unit_key is None:
            continue
        obs = []
        for o in units[unit_key]:
            filed = o.get("filed")
            val = o.get("val")
            if filed is None or val is None:
                continue
            obs.append({"filed": filed, "end": o.get("end"), "accn": o.get("accn"),
                        "form": o.get("form"), "fp": o.get("fp"), "val": float(val)})
        if obs:
            return obs
    return []


def _to_ms(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    try:
        return int(pd.Timestamp(date_str, tz="UTC").timestamp() * 1000)
    except Exception:
        return None


def build_symbol_fundamentals(symbol: str, facts_json: Mapping[str, Any]) -> pd.DataFrame:
    """Из companyfacts одного эмитента → long-кадр (publish_ts, symbol, earnings, book_value, fcf, roe).

    Группировка по accession (один отчёт = один publish-момент): берём последнее
    по периоду значение каждого концепта внутри accession, дату подачи = filed.
    """
    buckets: Dict[str, Dict[str, Any]] = {}

    def _ingest(field: str, concepts) -> None:
        for o in _facts_for(facts_json, concepts):
            accn = o["accn"] or f"{o['filed']}:{o['end']}"
            b = buckets.setdefault(accn, {"filed": o["filed"]})
            # последнее по 'end' значение в пределах accession
            prev_end = b.get(f"_{field}_end")
            if prev_end is None or (o["end"] or "") >= prev_end:
                b[field] = o["val"]
                b[f"_{field}_end"] = o["end"]
            b["filed"] = max(b.get("filed", o["filed"]), o["filed"])

    _ingest("eps", _EPS_CONCEPTS)
    _ingest("equity", _EQUITY_CONCEPTS)
    _ingest("shares", _SHARES_CONCEPTS)
    _ingest("ni", _NI_CONCEPTS)
    _ingest("cfo", _CFO_CONCEPTS)
    _ingest("capex", _CAPEX_CONCEPTS)

    rows: List[Dict[str, Any]] = []
    for b in buckets.values():
        pub = _to_ms(b.get("filed"))
        if pub is None:
            continue
        eps = b.get("eps")
        equity = b.get("equity")
        shares = b.get("shares")
        ni = b.get("ni")
        cfo = b.get("cfo")
        capex = b.get("capex")
        bvps = (equity / shares) if (equity is not None and shares not in (None, 0)) else None
        roe = (ni / equity) if (ni is not None and equity not in (None, 0)) else None
        fcf_ps = ((cfo - capex) / shares) if (cfo is not None and capex is not None
                                              and shares not in (None, 0)) else None
        if all(v is None for v in (eps, bvps, fcf_ps, roe)):
            continue
        rows.append({"publish_ts": pub, "symbol": symbol,
                     "earnings": eps, "book_value": bvps, "fcf": fcf_ps, "roe": roe})
    if not rows:
        return pd.DataFrame(columns=["publish_ts", "symbol", "earnings", "book_value", "fcf", "roe"])
    df = pd.DataFrame(rows).sort_values("publish_ts").reset_index(drop=True)
    return df


def build_pit_fundamentals_frame(
    symbols: Sequence[str],
    *,
    tickers_fn: Optional[Callable[[], Dict[str, int]]] = None,
    facts_fn: Optional[Callable[[int], Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Собрать PIT long-кадр по списку тикеров (через SEC или DI-фейки)."""
    tickers_fn = tickers_fn or default_tickers_fn
    facts_fn = facts_fn or default_facts_fn
    cik_map = tickers_fn()
    frames: List[pd.DataFrame] = []
    for sym in symbols:
        cik = cik_map.get(str(sym).upper())
        if cik is None:
            logger.warning("EDGAR: no CIK for %s", sym)
            continue
        try:
            facts = facts_fn(cik)
        except Exception as exc:
            logger.warning("EDGAR: companyfacts failed for %s (CIK %s): %s", sym, cik, exc)
            continue
        df = build_symbol_fundamentals(str(sym).upper(), facts)
        if len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["publish_ts", "symbol", "earnings", "book_value", "fcf", "roe"])
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "publish_ts"]).reset_index(drop=True)


def write_pit_parquet(symbols: Sequence[str], out_path: str, **kwargs: Any) -> str:
    """Скачать PIT-фундаментал и записать parquet для слота ``fundamentals_path``."""
    df = build_pit_fundamentals_frame(symbols, **kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("EDGAR PIT fundamentals written: %s (%d rows, %d symbols)",
                out_path, len(df), df["symbol"].nunique() if len(df) else 0)
    return out_path


# ---------------------------------------------------------------------------
# Drop-in source (тот же интерфейс, что ParquetFundamentals) — pit_quality=true
# ---------------------------------------------------------------------------
class EdgarFundamentals:
    """PIT-фундаментал из SEC EDGAR. ``get_fundamentals(symbols, fields)`` →
    long-кадр (publish_ts, symbol, fields). ``meta.pit_quality = true`` (filing dates)."""

    def __init__(self, *, tickers_fn: Optional[Callable[[], Dict[str, int]]] = None,
                 facts_fn: Optional[Callable[[int], Dict[str, Any]]] = None,
                 cache_path: Optional[str] = None) -> None:
        self._tickers_fn = tickers_fn
        self._facts_fn = facts_fn
        self._cache_path = cache_path
        self.meta = DataSourceMeta(
            name="edgar_fundamentals", vendor="sec_edgar", kind="enrich",
            pit_quality=PIT_TRUE, free=True,
            notes="SEC EDGAR XBRL companyfacts; publish_ts = filing date (genuine PIT).",
        )

    def get_fundamentals(self, symbols: Sequence[str],
                         fields: Optional[Sequence[str]] = None) -> pd.DataFrame:
        if self._cache_path and os.path.exists(self._cache_path):
            df = pd.read_parquet(self._cache_path)
            df = df[df["symbol"].isin([str(s).upper() for s in symbols])]
        else:
            df = build_pit_fundamentals_frame(symbols, tickers_fn=self._tickers_fn,
                                              facts_fn=self._facts_fn)
            if self._cache_path and len(df):
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(self._cache_path)), exist_ok=True)
                    df.to_parquet(self._cache_path, index=False)
                except Exception:
                    pass
        keep = ["publish_ts", "symbol"]
        if fields:
            keep += [f for f in fields if f in df.columns]
        else:
            keep += [c for c in df.columns if c not in ("publish_ts", "symbol")]
        return df[keep].reset_index(drop=True)


__all__ = [
    "EdgarFundamentals", "build_pit_fundamentals_frame", "build_symbol_fundamentals",
    "write_pit_parquet", "default_tickers_fn", "default_facts_fn",
]
