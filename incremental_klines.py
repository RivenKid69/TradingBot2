# -*- coding: utf-8 -*-
"""
incremental_klines.py
--------------------------------------------------------------
Incrementally *synchronise* CLOSED 1h Binance candles per symbol.

The previous implementation only appended the latest closed bar. In case the
script was run a handful of times before the training pipeline, the resulting
dataset contained only a couple of candles. This module now backfills every
missing candle since the beginning of Binance history (or since the last
recorded bar) and writes them to ``data/candles/{SYMBOL}.csv`` with the
original 12 Binance fields plus ``symbol``.

CLI:
  python incremental_klines.py --symbols BTCUSDT,ETHUSDT [--close-lag-ms 2000]
  python incremental_klines.py                       # load symbols from data/universe/symbols.json
"""
from __future__ import annotations

import os
import csv
import time
import argparse
import json
from typing import List, Optional
import requests
import clock

BASE = "https://api.binance.com/api/v3/klines"
OUT_DIR = os.path.join("data", "candles")
os.makedirs(OUT_DIR, exist_ok=True)

INTERVAL_MS = 3_600_000  # 1h
MAX_BATCH = 1000

HEADER = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_asset_volume","number_of_trades",
    "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore","symbol"
]

SYMBOL_MAX_ERRORS = 3
SYMBOL_RETRY_BACKOFF = 1.0


def _read_last_ts(path: str) -> Optional[int]:
    """Read the last numeric open_time from an existing CSV (skip header/blank lines).
    Returns the last open_time (as int) or None if file missing/empty.
    """
    if not os.path.exists(path):
        return None
    last_ts: Optional[int] = None
    try:
        with open(path, "r", newline="") as f:
            r = csv.reader(f)
            for row in r:
                if not row:
                    continue
                try:
                    ts = int(row[0])
                except Exception:
                    continue  # skip header or malformed
                else:
                    last_ts = ts
    except Exception:
        return None
    return last_ts


def _get_with_retry(url: str, *, params: dict, retries: int = 3, backoff: float = 0.5):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(backoff * attempt)


def _header_needed(path: str) -> bool:
    """Return ``True`` if the CSV is missing and header should be written."""
    return not os.path.exists(path)


def _rows_to_write(
    data: list,
    *,
    symbol: str,
    last_ts: Optional[int],
    max_close_ts: int,
    allow_existing: bool = False,
) -> list[tuple[int, list[str]]]:
    rows: list[tuple[int, list[str]]] = []
    for row in data:
        try:
            open_time = int(row[0])
            close_time = int(row[6])
        except Exception:
            continue
        if close_time > max_close_ts:
            break
        if not allow_existing and last_ts is not None and open_time <= last_ts:
            continue
        values = list(map(str, row + [symbol.upper()]))
        rows.append((open_time, values))
    rows.sort(key=lambda item: item[0])
    return rows


def _read_first_ts(path: str) -> Optional[int]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", newline="") as f:
            r = csv.reader(f)
            for row in r:
                if not row:
                    continue
                try:
                    return int(row[0])
                except Exception:
                    continue
    except Exception:
        return None
    return None


def _fetch_earliest_open_time(symbol: str) -> Optional[int]:
    params = {
        "symbol": symbol.upper(),
        "interval": "1h",
        "startTime": 0,
        "limit": 1,
    }
    try:
        response = _get_with_retry(BASE, params=params)
    except Exception:
        return None
    data = response.json()
    if not isinstance(data, list) or not data:
        return None
    try:
        return int(data[0][0])
    except Exception:
        return None


def sync_symbol(symbol: str, close_lag_ms: int, *, out_dir: Optional[str] = None) -> int:
    """Backfill and append all CLOSED 1h candles for ``symbol``.

    Parameters
    ----------
    symbol : str
        Trading pair symbol, e.g. ``BTCUSDT``.
    close_lag_ms : int
        Allowed lag in milliseconds when verifying the freshness of the latest bar.
    out_dir : str
        Directory where the CSV file is stored. Defaults to ``data/candles``.

    Returns
    -------
    int
        Number of rows appended to the CSV.
    """

    if out_dir is None:
        out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{symbol.upper()}.csv")
    last_ts = _read_last_ts(path)

    now_ms = clock.now_ms()
    max_close_ts = now_ms - int(close_lag_ms)
    if max_close_ts <= 0:
        return 0
    max_open_time = max_close_ts - INTERVAL_MS
    rebuild_mode = False
    earliest_ts: Optional[int] = None
    if last_ts is None:
        cursor = 0
    else:
        cursor = last_ts + INTERVAL_MS
        earliest_ts = _read_first_ts(path)

    if cursor > max_open_time:
        if last_ts is None:
            return 0
        earliest_exchange_ts = _fetch_earliest_open_time(symbol)
        if (
            earliest_ts is not None
            and earliest_exchange_ts is not None
            and earliest_ts > earliest_exchange_ts
        ):
            rebuild_mode = True
            cursor = earliest_exchange_ts
            last_ts = None
        else:
            return 0

    appended = 0
    header_pending = _header_needed(path)
    if rebuild_mode:
        header_pending = True

    write_mode = "a"

    while cursor <= max_open_time:
        remaining = max_open_time - cursor
        steps = int(remaining // INTERVAL_MS) + 1
        limit = min(MAX_BATCH, steps)
        params = {
            "symbol": symbol.upper(),
            "interval": "1h",
            "startTime": cursor,
            "limit": max(limit, 1),
        }
        response = _get_with_retry(BASE, params=params)
        data = response.json()
        if not isinstance(data, list) or not data:
            break

        rows = _rows_to_write(
            data,
            symbol=symbol,
            last_ts=last_ts,
            max_close_ts=max_close_ts,
            allow_existing=rebuild_mode,
        )
        if rows:
            if rebuild_mode and header_pending:
                write_mode = "w"
            with open(path, write_mode, newline="") as f:
                w = csv.writer(f)
                if header_pending:
                    w.writerow(HEADER)
                    header_pending = False
                for open_time, values in rows:
                    w.writerow(values)
                    last_ts = open_time
                    appended += 1
            write_mode = "a"
            cursor = last_ts + INTERVAL_MS if last_ts is not None else cursor + INTERVAL_MS
        else:
            last_open = int(data[-1][0])
            cursor = max(last_open + INTERVAL_MS, (last_ts or last_open) + INTERVAL_MS)
            if last_ts is not None and cursor <= last_ts:
                break
        if cursor > max_open_time:
            break

    return appended


def run_many(symbols: List[str], close_lag_ms: int) -> int:
    appended = 0
    skipped: List[str] = []
    for sym in symbols:
        errors = 0
        while True:
            try:
                appended += sync_symbol(sym, close_lag_ms)
                break
            except Exception as e:
                errors += 1
                print(
                    f"[WARN] {sym}: {e} (attempt {errors}/{SYMBOL_MAX_ERRORS})"
                )
                if errors >= SYMBOL_MAX_ERRORS:
                    print(
                        f"[WARN] {sym}: skipping after {SYMBOL_MAX_ERRORS} errors"
                    )
                    skipped.append(sym)
                    break
                time.sleep(SYMBOL_RETRY_BACKOFF * errors)
    if skipped:
        skipped_names = ", ".join(sorted(set(skipped)))
        print(f"[WARN] Skipped symbols: {skipped_names}")
    print(f"\u2713 Appended {appended} closed bars.")
    return appended


def _parse_symbols(s: str) -> List[str]:
    return [x.strip().upper() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Incrementally synchronise CLOSED 1h candles from Binance."
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated symbols or leave empty to use data/universe/symbols.json",
    )
    parser.add_argument(
        "--close-lag-ms",
        type=int,
        default=2000,
        help="Allowed lag in ms when verifying if the fetched bar is closed",
    )
    args = parser.parse_args()
    if args.symbols:
        symbols = _parse_symbols(args.symbols)
    else:
        json_path = os.path.join("data", "universe", "symbols.json")
        with open(json_path, "r") as f:
            symbols = [s.strip().upper() for s in json.load(f)]
    run_many(symbols, args.close_lag_ms)


if __name__ == "__main__":
    main()
