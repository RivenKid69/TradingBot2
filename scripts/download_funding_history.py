#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/download_funding_history.py
Download historical funding rate data from Binance Futures.

Downloads funding rate history and optionally mark price data
for crypto perpetual futures contracts.

Usage:
    # Download funding rates for a single symbol
    python scripts/download_funding_history.py --symbol BTCUSDT --start 2024-01-01

    # Download multiple symbols
    python scripts/download_funding_history.py --symbols BTCUSDT ETHUSDT --start 2024-01-01

    # Download with mark price data
    python scripts/download_funding_history.py --symbol BTCUSDT --start 2024-01-01 --with-mark

    # Download all top futures pairs
    python scripts/download_funding_history.py --popular --start 2024-01-01

References:
- Binance Funding Rate API: https://binance-docs.github.io/apidocs/futures/en/#get-funding-rate-history
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from binance_public import BinancePublicClient
from utils_time import parse_time_to_ms

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Popular perpetual futures pairs
POPULAR_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "LINKUSDT",
    "MATICUSDT",
    "LTCUSDT",
    "NEARUSDT",
    "UNIUSDT",
    "ATOMUSDT",
]


def fetch_funding_history(
    client: BinancePublicClient,
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    sleep_ms: int = 350,
) -> List[Dict[str, Any]]:
    """
    Fetch funding rate history from Binance.

    Args:
        client: Binance public client
        symbol: Futures symbol
        start_ms: Start timestamp (ms)
        end_ms: End timestamp (ms)
        limit: Records per request
        sleep_ms: Sleep between requests

    Returns:
        List of funding rate records
    """
    all_records: List[Dict[str, Any]] = []
    current_ms = start_ms

    logger.info(f"Fetching funding rates for {symbol}...")

    while current_ms < end_ms:
        try:
            batch = client.get_funding(
                symbol=symbol, start_ms=current_ms, end_ms=end_ms, limit=limit
            )
        except Exception as e:
            logger.error(f"Error fetching funding rates: {e}")
            # Advance by one funding period (8 hours)
            current_ms += 8 * 60 * 60 * 1000
            time.sleep(sleep_ms / 1000.0)
            continue

        if not batch:
            # No data, advance by one funding period
            current_ms += 8 * 60 * 60 * 1000
            time.sleep(sleep_ms / 1000.0)
            continue

        all_records.extend(batch)
        last_ts = int(batch[-1]["fundingTime"])
        current_ms = max(current_ms + 1, last_ts + 1)

        if len(all_records) % 100 == 0:
            logger.info(f"  Downloaded {len(all_records)} funding records...")

        time.sleep(sleep_ms / 1000.0)

    logger.info(f"Downloaded {len(all_records)} funding records for {symbol}")
    return all_records


def fetch_mark_price_history(
    client: BinancePublicClient,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1500,
    sleep_ms: int = 350,
) -> List[List[Any]]:
    """
    Fetch mark price klines from Binance.

    Args:
        client: Binance public client
        symbol: Futures symbol
        interval: Kline interval (e.g., "1h", "4h")
        start_ms: Start timestamp (ms)
        end_ms: End timestamp (ms)
        limit: Records per request
        sleep_ms: Sleep between requests

    Returns:
        List of mark price klines
    """
    all_klines: List[List[Any]] = []
    current_ms = start_ms

    logger.info(f"Fetching mark price klines for {symbol} ({interval})...")

    while current_ms < end_ms:
        try:
            batch = client.get_mark_klines(
                symbol=symbol, interval=interval, start_ms=current_ms, end_ms=end_ms, limit=limit
            )
        except Exception as e:
            logger.error(f"Error fetching mark klines: {e}")
            current_ms += 60_000
            time.sleep(sleep_ms / 1000.0)
            continue

        if not batch:
            current_ms += 60_000
            time.sleep(sleep_ms / 1000.0)
            continue

        all_klines.extend(batch)
        last_close = int(batch[-1][6])  # close_time
        current_ms = max(current_ms + 1, last_close + 1)

        if len(all_klines) % 500 == 0:
            logger.info(f"  Downloaded {len(all_klines)} mark price klines...")

        time.sleep(sleep_ms / 1000.0)

    logger.info(f"Downloaded {len(all_klines)} mark price klines for {symbol}")
    return all_klines


def funding_to_dataframe(
    records: List[Dict[str, Any]],
    symbol: str,
) -> pd.DataFrame:
    """
    Convert funding rate records to DataFrame.

    Args:
        records: Raw API records
        symbol: Symbol name

    Returns:
        DataFrame with columns: ts_ms, symbol, funding_rate, mark_price
    """
    if not records:
        return pd.DataFrame(columns=["ts_ms", "symbol", "funding_rate", "mark_price"])

    df = pd.DataFrame(records)
    df["ts_ms"] = df["fundingTime"].astype("int64")
    df["symbol"] = symbol.upper()
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")

    # Mark price may not always be present
    if "markPrice" in df.columns:
        df["mark_price"] = pd.to_numeric(df["markPrice"], errors="coerce")
    else:
        df["mark_price"] = 0.0

    result = df[["ts_ms", "symbol", "funding_rate", "mark_price"]].copy()
    result = result.sort_values(["symbol", "ts_ms"]).reset_index(drop=True)

    return result


def mark_klines_to_dataframe(
    klines: List[List[Any]],
    symbol: str,
) -> pd.DataFrame:
    """
    Convert mark price klines to DataFrame.

    Args:
        klines: Raw kline data
        symbol: Symbol name

    Returns:
        DataFrame with OHLCV columns
    """
    if not klines:
        return pd.DataFrame(
            columns=["ts_ms", "symbol", "mark_open", "mark_high", "mark_low", "mark_close"]
        )

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "ignore",
        "close_time",
        "ignore2",
        "ignore3",
        "ignore4",
        "ignore5",
        "ignore6",
    ]
    df = pd.DataFrame(klines, columns=cols[: len(klines[0])])
    df["ts_ms"] = df["open_time"].astype("int64")
    df["symbol"] = symbol.upper()

    result = df[["ts_ms", "symbol", "open", "high", "low", "close"]].copy()
    result = result.rename(
        columns={"open": "mark_open", "high": "mark_high", "low": "mark_low", "close": "mark_close"}
    )

    for col in ["mark_open", "mark_high", "mark_low", "mark_close"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    return result.sort_values(["symbol", "ts_ms"]).reset_index(drop=True)


def download_symbol_data(
    client: BinancePublicClient,
    symbol: str,
    start_ms: int,
    end_ms: int,
    out_dir: str,
    with_mark: bool = False,
    mark_interval: str = "4h",
    limit: int = 1000,
    sleep_ms: int = 350,
) -> Dict[str, int]:
    """
    Download funding and optionally mark price data for a symbol.

    Args:
        client: Binance client
        symbol: Futures symbol
        start_ms: Start timestamp
        end_ms: End timestamp
        out_dir: Output directory
        with_mark: Also download mark price klines
        mark_interval: Mark price kline interval
        limit: Records per request
        sleep_ms: Sleep between requests

    Returns:
        Dict with counts: funding_count, mark_count
    """
    result = {"funding_count": 0, "mark_count": 0}

    # Download funding rates
    funding_records = fetch_funding_history(client, symbol, start_ms, end_ms, limit, sleep_ms)
    funding_df = funding_to_dataframe(funding_records, symbol)

    if not funding_df.empty:
        funding_path = os.path.join(out_dir, f"{symbol}_funding.parquet")
        funding_df.to_parquet(funding_path, index=False)
        logger.info(f"Wrote {len(funding_df)} funding records: {funding_path}")
        result["funding_count"] = len(funding_df)

    # Download mark price if requested
    if with_mark:
        mark_klines = fetch_mark_price_history(
            client, symbol, mark_interval, start_ms, end_ms, 1500, sleep_ms
        )
        mark_df = mark_klines_to_dataframe(mark_klines, symbol)

        if not mark_df.empty:
            mark_path = os.path.join(out_dir, f"{symbol}_mark_{mark_interval}.parquet")
            mark_df.to_parquet(mark_path, index=False)
            logger.info(f"Wrote {len(mark_df)} mark price records: {mark_path}")
            result["mark_count"] = len(mark_df)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Download Binance Futures funding rate history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download single symbol
    python scripts/download_funding_history.py --symbol BTCUSDT --start 2024-01-01

    # Download multiple symbols
    python scripts/download_funding_history.py --symbols BTCUSDT ETHUSDT --start 2024-01-01

    # Download popular pairs with mark price
    python scripts/download_funding_history.py --popular --start 2024-01-01 --with-mark
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbol", type=str, help="Single futures symbol (e.g., BTCUSDT)")
    group.add_argument("--symbols", nargs="+", type=str, help="Multiple futures symbols")
    group.add_argument("--popular", action="store_true", help="Download popular futures pairs")

    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD or unix ms)")
    parser.add_argument("--end", default=None, help="End date (default: now)")
    parser.add_argument(
        "--out-dir", default="data/futures", help="Output directory (default: data/futures)"
    )
    parser.add_argument("--with-mark", action="store_true", help="Also download mark price klines")
    parser.add_argument(
        "--mark-interval", default="4h", help="Mark price kline interval (default: 4h)"
    )
    parser.add_argument(
        "--limit", type=int, default=1000, help="Records per request (default: 1000)"
    )
    parser.add_argument(
        "--sleep-ms", type=int, default=350, help="Sleep between requests in ms (default: 350)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse timestamps
    start_ms = parse_time_to_ms(args.start)
    if args.end:
        end_ms = parse_time_to_ms(args.end)
    else:
        end_ms = int(time.time() * 1000)

    # Determine symbols
    if args.symbol:
        symbols = [args.symbol.upper()]
    elif args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = POPULAR_SYMBOLS

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Initialize client
    client = BinancePublicClient()

    # Download data for each symbol
    total_funding = 0
    total_mark = 0

    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {symbol}")
        logger.info(f"{'='*50}")

        try:
            result = download_symbol_data(
                client=client,
                symbol=symbol,
                start_ms=start_ms,
                end_ms=end_ms,
                out_dir=args.out_dir,
                with_mark=args.with_mark,
                mark_interval=args.mark_interval,
                limit=args.limit,
                sleep_ms=args.sleep_ms,
            )
            total_funding += result["funding_count"]
            total_mark += result["mark_count"]
        except Exception as e:
            logger.error(f"Failed to download {symbol}: {e}")
            continue

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("DOWNLOAD COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"Total symbols: {len(symbols)}")
    logger.info(f"Total funding records: {total_funding}")
    if args.with_mark:
        logger.info(f"Total mark price records: {total_mark}")
    logger.info(f"Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
