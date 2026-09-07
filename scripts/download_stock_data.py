# -*- coding: utf-8 -*-
"""
scripts/download_stock_data.py
Download historical stock data from Alpaca/Polygon/Yahoo Finance for training.

This script downloads OHLCV data for US equities and indices, preparing it for
the training pipeline. Supports multiple data providers with auto-detection.

Features:
- Multi-symbol parallel downloads
- Auto-detection: Yahoo Finance for indices (^VIX), Alpaca/Polygon for stocks
- Automatic rate limiting
- Resume capability (skip already downloaded symbols)
- Market hours filtering (regular + extended)
- NYSE holiday calendar
- Output compatible with existing training pipeline

Providers:
- Alpaca (default): Free IEX feed or paid SIP feed for stocks
- Polygon: Paid provider with comprehensive market data
- Yahoo Finance: Free indices data (VIX, DXY, Treasury yields)

Usage:
    # Download VIX (auto-uses Yahoo Finance)
    python scripts/download_stock_data.py --symbols ^VIX --start 2020-01-01

    # Download VIX with convenience flag
    python scripts/download_stock_data.py --vix --start 2020-01-01

    # Download macro indicators (VIX, DXY, Treasury yields)
    python scripts/download_stock_data.py --macro --start 2020-01-01

    # Download popular stocks using Alpaca
    python scripts/download_stock_data.py --symbols AAPL MSFT GOOGL --provider alpaca

    # Download from file
    python scripts/download_stock_data.py --symbols-file data/universe/sp500.txt

    # Use Polygon provider
    python scripts/download_stock_data.py --symbols AAPL --provider polygon --api-key YOUR_KEY

    # Custom date range
    python scripts/download_stock_data.py --symbols AAPL --start 2020-01-01 --end 2024-01-01

Output:
    data/raw_stocks/VIX.parquet      # ^VIX sanitized to VIX
    data/raw_stocks/AAPL.parquet
    data/raw_stocks/DX-Y_NYB.parquet  # DX-Y.NYB sanitized

Author: Sultan Suyunbayev
Date: 2025-11-28
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.models import ExchangeVendor
from adapters.registry import create_market_data_adapter
from core_models import Bar

logger = logging.getLogger(__name__)


# =========================
# Configuration
# =========================


@dataclass
class DownloadConfig:
    """Configuration for stock data download."""

    # Data provider
    provider: str = "alpaca"  # "alpaca" or "polygon"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None

    # Symbols
    symbols: List[str] = field(default_factory=list)
    symbols_file: Optional[str] = None

    # Date range
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None  # YYYY-MM-DD
    lookback_days: int = 365 * 3  # Default: 3 years

    # Timeframe
    timeframe: str = "1h"  # 1m, 5m, 15m, 1h, 4h, 1d

    # Output
    output_dir: str = "data/raw_stocks"
    output_format: str = "parquet"  # parquet or feather

    # Processing
    include_extended_hours: bool = False
    filter_market_hours: bool = True
    resample_to: Optional[str] = None  # e.g., "4h" to resample 1h to 4h
    adjustment: str = "all"
    corporate_actions: bool = False

    # Rate limiting
    max_workers: int = 4
    rate_limit_delay: float = 0.25  # seconds between requests

    # Resume capability
    skip_existing: bool = True

    # Data feed (Alpaca-specific)
    feed: str = "iex"  # "iex" (free) or "sip" (paid)


# =========================
# NYSE Calendar
# =========================


class NYSECalendar:
    """
    NYSE market calendar with holidays and trading hours.

    Market hours (Eastern Time):
    - Pre-market: 4:00 AM - 9:30 AM
    - Regular: 9:30 AM - 4:00 PM
    - After-hours: 4:00 PM - 8:00 PM
    """

    # US market holidays (approximate - some may vary by year)
    HOLIDAYS = {
        # 2023
        datetime(2023, 1, 2),  # New Year's Day (observed)
        datetime(2023, 1, 16),  # MLK Day
        datetime(2023, 2, 20),  # Presidents Day
        datetime(2023, 4, 7),  # Good Friday
        datetime(2023, 5, 29),  # Memorial Day
        datetime(2023, 6, 19),  # Juneteenth
        datetime(2023, 7, 4),  # Independence Day
        datetime(2023, 9, 4),  # Labor Day
        datetime(2023, 11, 23),  # Thanksgiving
        datetime(2023, 12, 25),  # Christmas
        # 2024
        datetime(2024, 1, 1),  # New Year's Day
        datetime(2024, 1, 15),  # MLK Day
        datetime(2024, 2, 19),  # Presidents Day
        datetime(2024, 3, 29),  # Good Friday
        datetime(2024, 5, 27),  # Memorial Day
        datetime(2024, 6, 19),  # Juneteenth
        datetime(2024, 7, 4),  # Independence Day
        datetime(2024, 9, 2),  # Labor Day
        datetime(2024, 11, 28),  # Thanksgiving
        datetime(2024, 12, 25),  # Christmas
        # 2025
        datetime(2025, 1, 1),  # New Year's Day
        datetime(2025, 1, 20),  # MLK Day
        datetime(2025, 2, 17),  # Presidents Day
        datetime(2025, 4, 18),  # Good Friday
        datetime(2025, 5, 26),  # Memorial Day
        datetime(2025, 6, 19),  # Juneteenth
        datetime(2025, 7, 4),  # Independence Day
        datetime(2025, 9, 1),  # Labor Day
        datetime(2025, 11, 27),  # Thanksgiving
        datetime(2025, 12, 25),  # Christmas
    }

    @classmethod
    def is_holiday(cls, dt: datetime) -> bool:
        """Check if date is a market holiday."""
        date_only = datetime(dt.year, dt.month, dt.day)
        return date_only in cls.HOLIDAYS

    @classmethod
    def is_weekend(cls, dt: datetime) -> bool:
        """Check if date is a weekend."""
        return dt.weekday() >= 5  # Saturday = 5, Sunday = 6

    @classmethod
    def is_trading_day(cls, dt: datetime) -> bool:
        """Check if date is a trading day."""
        return not cls.is_weekend(dt) and not cls.is_holiday(dt)

    @classmethod
    def is_market_hours(
        cls,
        dt: datetime,
        include_extended: bool = False,
    ) -> bool:
        """
        Check if timestamp is during market hours.

        Args:
            dt: Datetime in UTC
            include_extended: Include pre-market and after-hours

        Returns:
            True if within market hours
        """
        if not cls.is_trading_day(dt):
            return False

        # Convert to Eastern Time (naive approach - ideally use pytz)
        # UTC-5 for EST, UTC-4 for EDT (March-November)
        # Simplified: assume EST
        et_hour = (dt.hour - 5) % 24  # Rough EST conversion

        if include_extended:
            # 4:00 AM - 8:00 PM ET
            return 4 <= et_hour < 20
        else:
            # 9:30 AM - 4:00 PM ET
            return (et_hour == 9 and dt.minute >= 30) or (10 <= et_hour < 16)


# =========================
# Data Download Functions
# =========================


def download_symbol_alpaca(
    symbol: str,
    config: DownloadConfig,
) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
    """
    Download historical data for a single symbol from Alpaca.

    Args:
        symbol: Stock symbol (e.g., "AAPL")
        config: Download configuration

    Returns:
        (symbol, dataframe, error_message)
    """
    try:
        from adapters.alpaca.market_data import AlpacaMarketDataAdapter

        # Create adapter
        adapter_config = {
            "api_key": config.api_key or os.environ.get("ALPACA_API_KEY", ""),
            "api_secret": config.api_secret or os.environ.get("ALPACA_API_SECRET", ""),
            "feed": config.feed,
            "adjustment": config.adjustment,
        }

        adapter = AlpacaMarketDataAdapter(
            vendor=ExchangeVendor.ALPACA,
            config=adapter_config,
        )

        # Calculate date range
        end_dt = datetime.now(timezone.utc)
        if config.end_date:
            end_dt = datetime.strptime(config.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        if config.start_date:
            start_dt = datetime.strptime(config.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            start_dt = end_dt - timedelta(days=config.lookback_days)

        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)

        logger.info(f"Downloading {symbol}: {start_dt.date()} to {end_dt.date()}")

        # Download bars
        bars = adapter.get_bars(
            symbol=symbol,
            timeframe=config.timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=50000,  # Max limit per request
        )

        if not bars:
            return symbol, None, "No data returned"

        # Convert to DataFrame
        records = []
        for bar in bars:
            records.append(
                {
                    "timestamp": bar.ts // 1000,  # Convert to seconds
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume_base),
                    "vwap": float(bar.vwap) if bar.vwap else None,
                    "trades": bar.trades if bar.trades else None,
                }
            )

        df = pd.DataFrame(records)

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Add symbol column
        df["symbol"] = symbol

        # Filter market hours if requested
        if config.filter_market_hours:
            df = _filter_market_hours(df, config.include_extended_hours)

        # Resample if requested
        if config.resample_to:
            df = _resample_bars(df, config.resample_to)

        logger.info(f"Downloaded {symbol}: {len(df)} bars")
        return symbol, df, None

    except ImportError as e:
        return symbol, None, f"Alpaca SDK not installed: {e}"
    except Exception as e:
        logger.exception(f"Error downloading {symbol}")
        return symbol, None, str(e)


def download_symbol_yahoo(
    symbol: str,
    config: DownloadConfig,
) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
    """
    Download historical data for a single symbol from Yahoo Finance.

    Best for: VIX (^VIX), DXY, Treasury yields, and other indices.

    Args:
        symbol: Yahoo symbol (e.g., "^VIX", "DX-Y.NYB")
        config: Download configuration

    Returns:
        (symbol, dataframe, error_message)
    """
    try:
        from adapters.yahoo.market_data import YahooMarketDataAdapter

        adapter = YahooMarketDataAdapter()

        # Calculate date range
        end_dt = datetime.now(timezone.utc)
        if config.end_date:
            end_dt = datetime.strptime(config.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        if config.start_date:
            start_dt = datetime.strptime(config.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            start_dt = end_dt - timedelta(days=config.lookback_days)

        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)

        logger.info(
            f"Downloading {symbol} from Yahoo Finance: {start_dt.date()} to {end_dt.date()}"
        )

        # Download bars
        bars = adapter.get_bars(
            symbol=symbol,
            timeframe=config.timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=50000,
        )

        if not bars:
            return symbol, None, "No data returned"

        # Convert to DataFrame
        records = []
        for bar in bars:
            records.append(
                {
                    "timestamp": bar.ts // 1000,  # Convert to seconds
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume_base),
                }
            )

        df = pd.DataFrame(records)

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Add symbol column (sanitize for filename)
        df["symbol"] = symbol

        # Note: Yahoo indices don't have typical market hours, skip filtering
        # But VIX is based on SPX options which trade during market hours

        # Resample if requested
        if config.resample_to:
            df = _resample_bars(df, config.resample_to)

        logger.info(f"Downloaded {symbol} from Yahoo: {len(df)} bars")
        return symbol, df, None

    except ImportError as e:
        return symbol, None, f"yfinance not installed: {e}"
    except Exception as e:
        logger.exception(f"Error downloading {symbol} from Yahoo")
        return symbol, None, str(e)


def download_symbol_polygon(
    symbol: str,
    config: DownloadConfig,
) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
    """
    Download historical data for a single symbol from Polygon.

    Args:
        symbol: Stock symbol (e.g., "AAPL")
        config: Download configuration

    Returns:
        (symbol, dataframe, error_message)
    """
    try:
        import requests

        api_key = config.api_key or os.environ.get("POLYGON_API_KEY", "")
        if not api_key:
            return symbol, None, "Polygon API key required"

        # Calculate date range
        end_dt = datetime.now(timezone.utc)
        if config.end_date:
            end_dt = datetime.strptime(config.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        if config.start_date:
            start_dt = datetime.strptime(config.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            start_dt = end_dt - timedelta(days=config.lookback_days)

        # Map timeframe to Polygon multiplier/timespan
        tf_map = {
            "1m": (1, "minute"),
            "5m": (5, "minute"),
            "15m": (15, "minute"),
            "1h": (1, "hour"),
            "4h": (4, "hour"),
            "1d": (1, "day"),
        }

        multiplier, timespan = tf_map.get(config.timeframe, (1, "hour"))

        logger.info(f"Downloading {symbol} from Polygon: {start_dt.date()} to {end_dt.date()}")

        # Polygon API endpoint
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/"
            f"{multiplier}/{timespan}/"
            f"{start_dt.strftime('%Y-%m-%d')}/{end_dt.strftime('%Y-%m-%d')}"
        )

        params = {
            "apiKey": api_key,
            "adjusted": "false" if config.adjustment == "raw" else "true",
            "sort": "asc",
            "limit": 50000,
        }

        all_results = []

        while url:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("results"):
                all_results.extend(data["results"])

            # Handle pagination
            url = data.get("next_url")
            if url:
                params = {"apiKey": api_key}
                time.sleep(0.12)  # Polygon rate limit: 5 calls/minute for free tier

        if not all_results:
            return symbol, None, "No data returned"

        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        df = df.rename(
            columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "vw": "vwap",
                "n": "trades",
            }
        )

        # Polygon returns timestamp in milliseconds
        df["timestamp"] = df["timestamp"] // 1000

        # Add symbol
        df["symbol"] = symbol

        # Select columns
        df = df[["timestamp", "open", "high", "low", "close", "volume", "vwap", "trades", "symbol"]]

        # Filter market hours if requested
        if config.filter_market_hours:
            df = _filter_market_hours(df, config.include_extended_hours)

        # Resample if requested
        if config.resample_to:
            df = _resample_bars(df, config.resample_to)

        logger.info(f"Downloaded {symbol} from Polygon: {len(df)} bars")
        return symbol, df, None

    except ImportError:
        return symbol, None, "requests library not installed"
    except Exception as e:
        logger.exception(f"Error downloading {symbol} from Polygon")
        return symbol, None, str(e)


def _filter_market_hours(
    df: pd.DataFrame,
    include_extended: bool = False,
) -> pd.DataFrame:
    """Filter DataFrame to market hours only."""
    if df.empty:
        return df

    # Convert timestamp to datetime
    df["_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    # Filter
    mask = df["_dt"].apply(lambda x: NYSECalendar.is_market_hours(x, include_extended))

    df = df[mask].drop(columns=["_dt"]).reset_index(drop=True)
    return df


def _resample_bars(
    df: pd.DataFrame,
    target_timeframe: str,
) -> pd.DataFrame:
    """
    Resample bars to a larger timeframe.

    Args:
        df: DataFrame with OHLCV data
        target_timeframe: Target timeframe (e.g., "4h", "1d")

    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df

    # Map timeframe to pandas offset
    tf_map = {
        "1h": "1H",
        "4h": "4H",
        "1d": "1D",
    }

    offset = tf_map.get(target_timeframe, target_timeframe.upper())

    # Set timestamp as index
    df = df.copy()
    df["_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("_dt")

    # Resample OHLCV
    resampled = (
        df.resample(offset)
        .agg(
            {
                "timestamp": "first",  # Start of bar
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "vwap": "mean",  # Approximate
                "trades": "sum",
                "symbol": "first",
            }
        )
        .dropna()
    )

    return resampled.reset_index(drop=True)


# =========================
# Main Download Functions
# =========================


def sanitize_filename(symbol: str) -> str:
    """
    Sanitize symbol for use as filename.

    Handles special characters in Yahoo symbols like ^VIX, DX-Y.NYB, GC=F
    """
    # Replace special chars with underscores
    replacements = {
        "^": "",  # ^VIX -> VIX
        "=": "_",  # GC=F -> GC_F
        ".": "_",  # DX-Y.NYB -> DX-Y_NYB
        " ": "_",
    }
    result = symbol
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def download_all_symbols(config: DownloadConfig) -> Dict[str, Any]:
    """
    Download data for all configured symbols.

    Args:
        config: Download configuration

    Returns:
        Summary dict with success/failure counts
    """
    # Load symbols
    symbols = list(config.symbols)

    if config.symbols_file:
        with open(config.symbols_file, "r") as f:
            for line in f:
                sym = line.strip().upper()
                if sym and not sym.startswith("#"):
                    symbols.append(sym)

    # Remove duplicates
    symbols = list(dict.fromkeys(symbols))

    if not symbols:
        logger.error("No symbols specified")
        return {"success": 0, "failed": 0, "skipped": 0}

    logger.info(f"Downloading {len(symbols)} symbols using {config.provider}")

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select download function
    download_funcs = {
        "alpaca": download_symbol_alpaca,
        "polygon": download_symbol_polygon,
        "yahoo": download_symbol_yahoo,
    }

    default_func = download_funcs.get(config.provider.lower())
    if not default_func:
        raise ValueError(f"Unknown provider: {config.provider}")

    def get_download_func(symbol: str):
        """Auto-select provider based on symbol type."""
        # Yahoo symbols: ^VIX, DX-Y.NYB, futures (=F), etc.
        if symbol.startswith("^") or "=" in symbol or "-Y." in symbol:
            return download_symbol_yahoo
        return default_func

    # Track results
    results = {
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "errors": {},
    }

    # Download with parallel execution
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {}

        for symbol in symbols:
            # Check if already exists (sanitize filename for special chars)
            ext = "parquet" if config.output_format == "parquet" else "feather"
            safe_symbol = sanitize_filename(symbol)
            output_path = output_dir / f"{safe_symbol}.{ext}"

            if config.skip_existing and output_path.exists():
                logger.info(f"Skipping {symbol} (already exists)")
                results["skipped"] += 1
                continue

            # Submit download task (auto-select provider for indices)
            download_func = get_download_func(symbol)
            future = executor.submit(download_func, symbol, config)
            futures[future] = symbol

            # Rate limiting
            time.sleep(config.rate_limit_delay)

        # Collect results
        for future in as_completed(futures):
            symbol = futures[future]

            try:
                sym, df, error = future.result()

                if error:
                    logger.error(f"Failed to download {symbol}: {error}")
                    results["failed"] += 1
                    results["errors"][symbol] = error
                elif df is not None and not df.empty:
                    # Save to file (sanitize filename for special chars)
                    ext = "parquet" if config.output_format == "parquet" else "feather"
                    safe_symbol = sanitize_filename(symbol)
                    output_path = output_dir / f"{safe_symbol}.{ext}"

                    if config.output_format == "parquet":
                        df.to_parquet(output_path, index=False)
                    else:
                        df.to_feather(output_path)

                    logger.info(f"Saved {symbol}: {len(df)} bars -> {output_path}")
                    results["success"] += 1
                else:
                    logger.warning(f"No data for {symbol}")
                    results["failed"] += 1
                    results["errors"][symbol] = "No data"

            except Exception as e:
                logger.exception(f"Error processing {symbol}")
                results["failed"] += 1
                results["errors"][symbol] = str(e)

    # Download corporate actions if requested
    if config.corporate_actions:
        logger.info("Downloading corporate actions (splits and dividends) using Yahoo...")
        try:
            from adapters.yahoo.corporate_actions import YahooCorporateActionsAdapter
            from adapters.models import ExchangeVendor

            ca_adapter = YahooCorporateActionsAdapter(vendor=ExchangeVendor.YAHOO)

            for symbol in symbols:
                if symbol.startswith("^") or "=" in symbol or "-Y." in symbol:
                    continue

                logger.info(f"Fetching corporate actions for {symbol}...")
                try:
                    start_str = config.start_date
                    end_str = config.end_date

                    splits = ca_adapter.get_splits(symbol, start_date=start_str, end_date=end_str)
                    dividends = ca_adapter.get_dividends(
                        symbol, start_date=start_str, end_date=end_str
                    )

                    ca_records = []
                    for s in splits:
                        ca_records.append(
                            {
                                "date": s.ex_date,
                                "type": "split",
                                "factor": float(s.adjustment_factor),
                            }
                        )
                    for d in dividends:
                        ca_records.append(
                            {
                                "date": d.ex_date,
                                "type": "dividend",
                                "amount": float(d.amount),
                            }
                        )

                    if ca_records:
                        ca_df = pd.DataFrame(ca_records)
                        ca_df = ca_df.sort_values("date").reset_index(drop=True)

                        safe_symbol = sanitize_filename(symbol)
                        ca_path = output_dir / f"{safe_symbol}_corporate_actions.parquet"
                        ca_df.to_parquet(ca_path, index=False)
                        logger.info(
                            f"Saved {len(ca_df)} corporate actions for {symbol} to {ca_path}"
                        )
                    else:
                        logger.info(f"No corporate actions found for {symbol}")
                except Exception as ex:
                    logger.warning(f"Failed to fetch corporate actions for {symbol}: {ex}")
        except Exception as ex:
            logger.error(f"Failed to initialize corporate actions adapter: {ex}")

    return results


# =========================
# CLI Entry Point
# =========================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download historical stock data for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download AAPL and MSFT using Alpaca
  python scripts/download_stock_data.py --symbols AAPL MSFT --provider alpaca

  # Download from file
  python scripts/download_stock_data.py --symbols-file data/universe/sp500.txt

  # Use Polygon with custom date range
  python scripts/download_stock_data.py --symbols AAPL --provider polygon \\
      --api-key YOUR_KEY --start 2020-01-01 --end 2024-01-01

  # Download hourly data, resample to 4h
  python scripts/download_stock_data.py --symbols AAPL --timeframe 1h --resample 4h
        """,
    )

    # Symbols
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=[],
        help="Stock symbols to download (e.g., AAPL MSFT GOOGL)",
    )
    parser.add_argument(
        "--symbols-file",
        help="File with symbols (one per line)",
    )
    parser.add_argument(
        "--popular",
        action="store_true",
        help="Download popular tech stocks (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA)",
    )
    parser.add_argument(
        "--vix",
        action="store_true",
        help="Download VIX and related volatility indices (^VIX, ^VXN)",
    )
    parser.add_argument(
        "--macro",
        action="store_true",
        help="Download macro indicators (^VIX, DX-Y.NYB, ^TNX, ^TYX)",
    )

    # Provider
    parser.add_argument(
        "--provider",
        choices=["alpaca", "polygon", "yahoo"],
        default="alpaca",
        help="Data provider: alpaca (default), polygon, or yahoo (for indices)",
    )
    parser.add_argument(
        "--api-key",
        help="API key (or set ALPACA_API_KEY / POLYGON_API_KEY env var)",
    )
    parser.add_argument(
        "--api-secret",
        help="API secret (Alpaca only, or set ALPACA_API_SECRET env var)",
    )
    parser.add_argument(
        "--feed",
        choices=["iex", "sip"],
        default="iex",
        help="Alpaca data feed: iex (free) or sip (paid)",
    )

    # Date range
    parser.add_argument(
        "--start",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365 * 3,
        help="Days of history to download (default: 1095 = 3 years)",
    )

    # Timeframe
    parser.add_argument(
        "--timeframe",
        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
        default="1h",
        help="Bar timeframe (default: 1h)",
    )
    parser.add_argument(
        "--resample",
        choices=["1h", "4h", "1d"],
        help="Resample to larger timeframe after download",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="data/raw_stocks",
        help="Output directory (default: data/raw_stocks)",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "feather"],
        default="parquet",
        help="Output format (default: parquet)",
    )

    # Processing
    parser.add_argument(
        "--include-extended",
        action="store_true",
        help="Include pre-market and after-hours data",
    )
    parser.add_argument(
        "--no-filter-hours",
        action="store_true",
        help="Don't filter by market hours",
    )
    parser.add_argument(
        "--adjustment",
        choices=["all", "split", "raw"],
        default="all",
        help="Data adjustment type: all (splits+dividends), split (splits only), or raw (unadjusted)",
    )
    parser.add_argument(
        "--corporate-actions",
        action="store_true",
        help="Download corporate actions (dividends and splits) to JSON/Parquet file",
    )
    parser.add_argument(
        "--rollover",
        choices=["volume", "open_interest", "expiry"],
        default="volume",
        help="Rollover rule for continuous futures contracts (Volume, OI, Expiry)",
    )
    parser.add_argument(
        "--adjust",
        choices=["panama", "ratio", "raw"],
        default="panama",
        help="Rollover adjustment method (Panama, Ratio, Raw)",
    )

    # Execution
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel downloads (default: 4)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download even if file exists",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Build config
    symbols = list(args.symbols)
    if args.popular:
        symbols.extend(["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"])
    if args.vix:
        symbols.extend(["^VIX", "^VXN"])
    if args.macro:
        symbols.extend(["^VIX", "DX-Y.NYB", "^TNX", "^TYX"])

    config = DownloadConfig(
        provider=args.provider,
        api_key=args.api_key,
        api_secret=args.api_secret,
        symbols=symbols,
        symbols_file=args.symbols_file,
        start_date=args.start,
        end_date=args.end,
        lookback_days=args.lookback_days,
        timeframe=args.timeframe,
        resample_to=args.resample,
        output_dir=args.output_dir,
        output_format=args.format,
        include_extended_hours=args.include_extended,
        filter_market_hours=not args.no_filter_hours,
        max_workers=args.workers,
        skip_existing=not args.no_skip_existing,
        feed=args.feed,
        adjustment=args.adjustment,
        corporate_actions=args.corporate_actions,
    )

    # Run download
    print(f"\n{'='*60}")
    print("Stock Data Downloader")
    print(f"{'='*60}")
    print(f"Provider: {config.provider}")
    print(f"Symbols: {len(config.symbols) + (1 if config.symbols_file else 0)} source(s)")
    print(f"Timeframe: {config.timeframe}")
    print(f"Output: {config.output_dir}")
    print(f"{'='*60}\n")

    try:
        results = download_all_symbols(config)

        print(f"\n{'='*60}")
        print("Download Summary")
        print(f"{'='*60}")
        print(f"Success: {results['success']}")
        print(f"Failed:  {results['failed']}")
        print(f"Skipped: {results['skipped']}")

        if results["errors"]:
            print(f"\nErrors:")
            for symbol, error in list(results["errors"].items())[:10]:
                print(f"  {symbol}: {error}")
            if len(results["errors"]) > 10:
                print(f"  ... and {len(results['errors']) - 10} more")

        print(f"{'='*60}\n")

        return 0 if results["failed"] == 0 else 1

    except Exception as e:
        logger.exception("Download failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
