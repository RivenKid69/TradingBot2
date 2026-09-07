# -*- coding: utf-8 -*-
"""Платные/интрадей-фиды: минутные бары и тиковый бэкфилл (P2-M, §5.25).

Закрывает гэп «нет минутных/тиковых платных фидов»: адаптеры Polygon/Alpaca/
Binance/OANDA уже умеют минутные бары через ``get_bars``, но не было единой
точки — матрицы вендоров (кто что умеет, какие ключи нужны, что реально
entitled), скачивания в стандартный layout и MVP-поверхности.

Принципы (как в проф. дата-пайплайнах):
* **Честная entitlement-матрица**: вендор показывается доступным только когда
  его адаптер импортируется И ключи заданы; «paid» помечен явно; то, чего
  адаптер не умеет (исторические тики у Polygon/Alpaca в этой сборке), не
  выдаётся за возможность.
* **Единый layout**: parquet ``data/minute/{vendor}/{SYMBOL}_{tf}.parquet`` со
  схемой ``timestamp(sec) / open / high / low / close / volume / symbol`` —
  той же, что пишет ``scripts/download_stock_data.py`` (совместимо с
  загрузчиками); рядом ``.manifest.json`` (vendor/range/rows/sha256) для QC и
  lineage.
* **Тики — только настоящие**: исторический тиковый бэкфилл реализован там,
  где у вендора есть публичный исторический endpoint — Binance aggTrades
  (``/api/v3/aggTrades``, настоящие агрегированные сделки). Equity/options-тики
  требуют платных планов и в матрице честно помечены unavailable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = os.path.join("data", "minute")
DEFAULT_TICKS_DIR = os.path.join("data", "ticks")

# Матрица вендоров интрадей-данных. capability-поля описывают ТОЛЬКО то, что
# реализовано в этой сборке (адаптер/эндпоинт существует), не маркетинг вендора.
VENDOR_CAPS: Dict[str, Dict[str, Any]] = {
    "binance": {
        "title": "Binance (crypto)",
        "asset_classes": ["crypto"],
        "bars": ["1m", "5m", "15m", "1h"],
        "ticks": "history",  # aggTrades — публичный исторический endpoint
        "paid": False,
        "key_envs": [],  # публичные klines/aggTrades без ключа
        "notes": "Минутки и тиковый бэкфилл (aggTrades) бесплатны.",
    },
    "polygon": {
        "title": "Polygon.io (US equities/options)",
        "asset_classes": ["equity", "options"],
        "bars": ["1m", "5m", "15m", "1h"],
        "ticks": "unavailable",  # исторических trades нет в адаптере этой сборки
        "paid": True,
        "key_envs": ["POLYGON_API_KEY"],
        "notes": "Минутки требуют платного плана (free: 5 req/min, EOD).",
    },
    "alpaca": {
        "title": "Alpaca Data (US equities)",
        "asset_classes": ["equity"],
        "bars": ["1m", "5m", "15m", "1h"],
        "ticks": "unavailable",
        "paid": True,
        "key_envs": ["ALPACA_API_KEY", "ALPACA_API_SECRET"],
        "notes": "IEX-фид бесплатно; полный SIP — платная подписка Alpaca.",
    },
    "oanda": {
        "title": "OANDA (forex)",
        "asset_classes": ["forex"],
        "bars": ["1m", "5m", "15m", "1h"],
        "ticks": "unavailable",
        "paid": False,
        "key_envs": ["OANDA_API_KEY", "OANDA_ACCOUNT_ID"],
        "notes": "Минутки через practice/live API OANDA.",
    },
    "dukascopy": {
        "title": "Dukascopy (forex/metals)",
        "asset_classes": ["forex"],
        "bars": ["1m", "5m", "15m", "1h"],
        "ticks": "history",  # публичный bi5 tick-feed (агрегируется в бары)
        "paid": False,
        "key_envs": [],  # публичный фид без авторизации
        "notes": "Бесплатный публичный tick-feed (bi5), история с 2003, без ключей.",
    },
}


def _looks_placeholder(v: Optional[str]) -> bool:
    if not v:
        return True
    return "test" in v or "YOUR" in v or "$" in v


def vendor_status() -> List[Dict[str, Any]]:
    """Честная entitlement-матрица: адаптер importable + ключи заданы."""
    out: List[Dict[str, Any]] = []
    for vendor, caps in VENDOR_CAPS.items():
        keys_present = all(not _looks_placeholder(os.getenv(k)) for k in caps["key_envs"])
        adapter_ok, adapter_err = True, None
        try:
            from adapters.registry import get_registry, AdapterType
            from adapters.models import ExchangeVendor

            reg = get_registry().get_registration(ExchangeVendor(vendor), AdapterType.MARKET_DATA)
            adapter_ok = reg is not None
            if not adapter_ok:
                adapter_err = "MARKET_DATA адаптер не зарегистрирован"
        except Exception as exc:
            adapter_ok, adapter_err = False, str(exc)
        out.append(
            {
                "vendor": vendor,
                **{
                    k: caps[k] for k in ("title", "asset_classes", "bars", "ticks", "paid", "notes")
                },
                "key_envs": caps["key_envs"],
                "keys_present": keys_present,
                "adapter_available": adapter_ok,
                "adapter_error": adapter_err,
                "ready": adapter_ok and keys_present,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Минутные бары → parquet (единый layout)
# ---------------------------------------------------------------------------


@dataclass
class DownloadResult:
    vendor: str
    symbol: str
    timeframe: str
    path: Optional[str] = None
    manifest_path: Optional[str] = None
    rows: int = 0
    ok: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _vendor_config(vendor: str) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if vendor == "polygon":
        cfg["api_key"] = os.getenv("POLYGON_API_KEY", "")
    elif vendor == "alpaca":
        cfg["api_key"] = os.getenv("ALPACA_API_KEY", "")
        cfg["api_secret"] = os.getenv("ALPACA_API_SECRET", "")
    elif vendor == "oanda":
        cfg["api_key"] = os.getenv("OANDA_API_KEY", "")
        cfg["account_id"] = os.getenv("OANDA_ACCOUNT_ID", "")
    return cfg


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(parquet_path: str, meta: Dict[str, Any]) -> str:
    mpath = parquet_path + ".manifest.json"
    meta = dict(meta)
    meta["sha256"] = _sha256_file(parquet_path)
    meta["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return mpath


def download_minute_bars(
    vendor: str,
    symbols: List[str],
    *,
    timeframe: str = "1m",
    start_ts_ms: int,
    end_ts_ms: int,
    out_dir: str = DEFAULT_OUT_DIR,
    adapter: Any = None,
    chunk_ms: int = 24 * 3600 * 1000,
    limit_per_call: int = 1000,
) -> List[DownloadResult]:
    """Скачать минутные бары в стандартный parquet-layout.

    Пагинация по временным окнам ``chunk_ms`` поверх ``get_bars`` адаптера
    (адаптеры сами ограничены limit-ом на вызов). ``adapter`` инжектится в
    тестах; в проде создаётся из registry по vendor + ключам из env.
    """
    import pandas as pd

    caps = VENDOR_CAPS.get(vendor)
    if caps is None:
        return [
            DownloadResult(vendor, s, timeframe, ok=False, error=f"неизвестный вендор {vendor!r}")
            for s in symbols
        ]
    if timeframe not in caps["bars"]:
        return [
            DownloadResult(
                vendor,
                s,
                timeframe,
                ok=False,
                error=f"{vendor} не поддерживает таймфрейм {timeframe}",
            )
            for s in symbols
        ]

    if adapter is None:
        from adapters.registry import create_market_data_adapter

        adapter = create_market_data_adapter(vendor, _vendor_config(vendor))
        adapter.connect()

    os.makedirs(os.path.join(out_dir, vendor), exist_ok=True)
    results: List[DownloadResult] = []

    for symbol in symbols:
        res = DownloadResult(vendor, symbol, timeframe)
        try:
            rows: List[Dict[str, Any]] = []
            cursor = int(start_ts_ms)
            while cursor < end_ts_ms:
                window_end = min(cursor + chunk_ms, end_ts_ms)
                bars = adapter.get_bars(
                    symbol,
                    timeframe,
                    limit=limit_per_call,
                    start_ts=cursor,
                    end_ts=window_end,
                )
                for b in bars or []:
                    # core_models.Bar несёт объём как volume_base/volume_quote;
                    # некоторые адаптеры могут отдавать просто .volume — берём
                    # первое доступное, честный 0.0 если объёма нет.
                    vol = getattr(b, "volume_base", None)
                    if vol is None:
                        vol = getattr(b, "volume", None)
                    if vol is None:
                        vol = getattr(b, "volume_quote", None)
                    rows.append(
                        {
                            # схема scripts/download_stock_data.py: секунды, OHLCV
                            "timestamp": int(b.ts) // 1000,
                            "open": float(b.open),
                            "high": float(b.high),
                            "low": float(b.low),
                            "close": float(b.close),
                            "volume": float(vol) if vol is not None else 0.0,
                            "symbol": symbol,
                        }
                    )
                cursor = window_end
            if not rows:
                res.error = "вендор не вернул данных за диапазон"
                results.append(res)
                continue
            df = pd.DataFrame(rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            safe_sym = symbol.replace("/", "_").replace("^", "")
            path = os.path.join(out_dir, vendor, f"{safe_sym}_{timeframe}.parquet")
            df.to_parquet(path, index=False)
            res.path = path
            res.rows = int(len(df))
            res.manifest_path = _write_manifest(
                path,
                {
                    "vendor": vendor,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "start_ts_ms": int(start_ts_ms),
                    "end_ts_ms": int(end_ts_ms),
                    "rows": res.rows,
                    "kind": "bars",
                },
            )
            res.ok = True
        except Exception as exc:
            res.error = str(exc)
            logger.warning("premium-data[%s/%s]: %s", vendor, symbol, exc)
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Тиковый бэкфилл — Binance aggTrades (настоящие исторические сделки)
# ---------------------------------------------------------------------------


def download_binance_agg_trades(
    symbols: List[str],
    *,
    start_ts_ms: int,
    end_ts_ms: int,
    out_dir: str = DEFAULT_TICKS_DIR,
    base_url: str = "https://api.binance.com",
    max_requests_per_symbol: int = 2000,
    fetch_fn: Any = None,
) -> List[DownloadResult]:
    """Исторические тики (aggTrades) Binance → parquet.

    Пагинация по ``fromId`` (канонический способ Binance): первый запрос по
    startTime, далее fromId=последний+1 до end_ts. ``fetch_fn(params)->list``
    инжектится в тестах; в проде — публичный REST без ключа.
    """
    import pandas as pd

    if fetch_fn is None:
        import requests

        def fetch_fn(params: Dict[str, Any]) -> List[Dict[str, Any]]:  # type: ignore[misc]
            r = requests.get(f"{base_url}/api/v3/aggTrades", params=params, timeout=30)
            r.raise_for_status()
            return r.json()

    os.makedirs(os.path.join(out_dir, "binance"), exist_ok=True)
    results: List[DownloadResult] = []

    for symbol in symbols:
        res = DownloadResult("binance", symbol, "tick")
        try:
            rows: List[Dict[str, Any]] = []
            params: Dict[str, Any] = {
                "symbol": symbol,
                "limit": 1000,
                "startTime": int(start_ts_ms),
                "endTime": min(int(start_ts_ms) + 3600_000, int(end_ts_ms)),
            }
            from_id: Optional[int] = None
            for _ in range(max_requests_per_symbol):
                if from_id is not None:
                    params = {"symbol": symbol, "limit": 1000, "fromId": from_id}
                batch = fetch_fn(params)
                if not batch:
                    break
                reached_end = False
                for t in batch:
                    ts = int(t["T"])
                    if ts > end_ts_ms:
                        reached_end = True
                        break
                    rows.append(
                        {
                            "ts_ms": ts,
                            "price": float(t["p"]),
                            "qty": float(t["q"]),
                            "agg_id": int(t["a"]),
                            "is_buyer_maker": bool(t["m"]),
                            "symbol": symbol,
                        }
                    )
                if reached_end or int(batch[-1]["T"]) >= end_ts_ms:
                    break
                # canonical Binance pagination: continue strictly by fromId
                from_id = int(batch[-1]["a"]) + 1
                if from_id is not None and len(batch) < 1000 and "fromId" in params:
                    break  # fromId-страница короче лимита → история исчерпана
            if not rows:
                res.error = "aggTrades не вернул сделок за диапазон"
                results.append(res)
                continue
            df = pd.DataFrame(rows).drop_duplicates(subset=["agg_id"]).sort_values("agg_id")
            path = os.path.join(out_dir, "binance", f"{symbol}_ticks.parquet")
            df.to_parquet(path, index=False)
            res.path, res.rows = path, int(len(df))
            res.manifest_path = _write_manifest(
                path,
                {
                    "vendor": "binance",
                    "symbol": symbol,
                    "timeframe": "tick",
                    "start_ts_ms": int(start_ts_ms),
                    "end_ts_ms": int(end_ts_ms),
                    "rows": res.rows,
                    "kind": "agg_trades",
                },
            )
            res.ok = True
        except Exception as exc:
            res.error = str(exc)
            logger.warning("premium-data[binance-ticks/%s]: %s", symbol, exc)
        results.append(res)
    return results


__all__ = [
    "DEFAULT_OUT_DIR",
    "DEFAULT_TICKS_DIR",
    "VENDOR_CAPS",
    "DownloadResult",
    "download_binance_agg_trades",
    "download_minute_bars",
    "vendor_status",
]
