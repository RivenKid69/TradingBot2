# -*- coding: utf-8 -*-
"""
scripts/download_forex_data.py
Download historical Forex data from OANDA for training.

This script downloads OHLCV data for forex pairs, preparing it for
the training pipeline. Features bid/ask spread data and session filtering.

Features:
- Multi-pair parallel downloads
- Bid/Ask/Mid price candles for accurate spread modeling
- Session filtering (Sydney/Tokyo/London/NY)
- Automatic rate limiting (120 req/s OANDA limit)
- Resume capability (skip already downloaded pairs)
- Weekend gap filtering (Forex market closed Sat-Sun)
- Output compatible with existing training pipeline

Providers:
- OANDA (default): REST v20 API with streaming support
- Dukascopy (future): High-frequency tick data

Usage:
    # Download major pairs (EUR/USD, GBP/USD, USD/JPY, etc.)
    python scripts/download_forex_data.py --majors --start 2020-01-01

    # Download specific pairs
    python scripts/download_forex_data.py --pairs EUR_USD GBP_USD USD_JPY --start 2020-01-01

    # Download all supported pairs
    python scripts/download_forex_data.py --all --start 2020-01-01 --timeframe 4h

    # Download with custom output directory
    python scripts/download_forex_data.py --pairs EUR_USD --output data/forex/raw/

Output:
    data/raw_forex/EUR_USD.parquet
    data/raw_forex/GBP_USD.parquet
    data/raw_forex/USD_JPY.parquet

Columns in output:
    - timestamp: Unix timestamp in seconds
    - open, high, low, close: Mid prices (for OHLC analysis)
    - open_bid, high_bid, low_bid, close_bid: Bid prices
    - open_ask, high_ask, low_ask, close_ask: Ask prices
    - volume: Tick volume (OANDA doesn't provide real volume)
    - spread_pips: Close ask - close bid in pips
    - symbol: Currency pair
    - session: Active session at bar time

Author: Sultan Suyunbayev
Date: 2025-11-30
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
from datetime import datetime, timedelta, timezone, time as dt_time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.models import ExchangeVendor

logger = logging.getLogger(__name__)


# =========================
# Configuration
# =========================


@dataclass
class ForexDownloadConfig:
    """Configuration for forex data download."""

    # Data provider
    provider: str = "oanda"
    api_key: Optional[str] = None
    account_id: Optional[str] = None
    practice: bool = True  # Use practice/demo environment

    # Pairs to download
    pairs: List[str] = field(default_factory=list)
    pairs_file: Optional[str] = None

    # Date range
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None  # YYYY-MM-DD
    lookback_days: int = 365 * 3  # Default: 3 years

    # Timeframe
    timeframe: str = "1h"  # 5s, 1m, 5m, 15m, 1h, 4h, 1d

    # Output
    output_dir: str = "data/raw_forex"
    output_format: str = "parquet"  # parquet or feather

    # Processing
    filter_weekends: bool = True  # Remove weekend data (market closed)
    include_spread: bool = True  # Include bid/ask prices
    add_session_labels: bool = True  # Label bars with active session
    resample_to: Optional[str] = None  # e.g., "4h" to resample 1h to 4h
    price_type: str = "mid"  # mid, bid, ask, bid_ask

    # Rate limiting
    max_workers: int = 2  # Lower for OANDA due to rate limits
    rate_limit_delay: float = 0.5  # seconds between requests

    # Resume capability
    skip_existing: bool = True


# =========================
# Currency Pair Definitions
# =========================

# Major pairs (highest liquidity)
MAJOR_PAIRS = [
    "EUR_USD",
    "USD_JPY",
    "GBP_USD",
    "USD_CHF",
    "AUD_USD",
    "USD_CAD",
    "NZD_USD",
]

# Minor pairs (cross pairs without USD)
MINOR_PAIRS = [
    "EUR_GBP",
    "EUR_CHF",
    "EUR_JPY",
    "GBP_CHF",
    "GBP_JPY",
    "CHF_JPY",
    "AUD_JPY",
    "AUD_NZD",
    "NZD_JPY",
]

# Exotic pairs (emerging market currencies)
EXOTIC_PAIRS = [
    "USD_TRY",
    "USD_ZAR",
    "USD_MXN",
    "EUR_TRY",
    "EUR_ZAR",
]

# All supported pairs
ALL_PAIRS = MAJOR_PAIRS + MINOR_PAIRS + EXOTIC_PAIRS

# Pip sizes for different currency pairs
PIP_SIZES = {
    # JPY pairs have pip = 0.01
    "USD_JPY": 0.01,
    "EUR_JPY": 0.01,
    "GBP_JPY": 0.01,
    "CHF_JPY": 0.01,
    "AUD_JPY": 0.01,
    "NZD_JPY": 0.01,
    "CAD_JPY": 0.01,
    # All other pairs have pip = 0.0001
}
DEFAULT_PIP_SIZE = 0.0001


# =========================
# Forex Trading Calendar
# =========================


class ForexCalendar:
    """
    Forex market calendar with trading sessions.

    Market hours (24/5):
    - Opens: Sunday 5:00 PM ET (21:00 UTC summer, 22:00 UTC winter)
    - Closes: Friday 5:00 PM ET
    - Weekend: Saturday 5:00 PM - Sunday 5:00 PM ET
    """

    # Session times in UTC
    SESSIONS = {
        "sydney": (dt_time(21, 0), dt_time(6, 0)),  # 21:00-06:00 UTC
        "tokyo": (dt_time(0, 0), dt_time(9, 0)),  # 00:00-09:00 UTC
        "london": (dt_time(7, 0), dt_time(16, 0)),  # 07:00-16:00 UTC
        "new_york": (dt_time(12, 0), dt_time(21, 0)),  # 12:00-21:00 UTC
    }

    @classmethod
    def is_weekend(cls, dt: datetime) -> bool:
        """
        Check if datetime is during forex weekend closure.

        Forex closes Friday 5pm ET (21:00/22:00 UTC) and
        opens Sunday 5pm ET.

        Args:
            dt: Datetime in UTC

        Returns:
            True if market is closed for weekend
        """
        weekday = dt.weekday()

        # Saturday is always closed
        if weekday == 5:
            return True

        # Sunday before ~21:00 UTC is closed
        if weekday == 6 and dt.hour < 21:
            return True

        # Friday after ~21:00 UTC is closed
        if weekday == 4 and dt.hour >= 21:
            return True

        return False

    @classmethod
    def get_active_session(cls, dt: datetime) -> str:
        """
        Get the primary active trading session for a given time.

        Args:
            dt: Datetime in UTC

        Returns:
            Session name: "sydney", "tokyo", "london", "new_york", or "overlap"
        """
        t = dt.time()

        # Check for overlaps first (highest liquidity)
        # London/NY overlap: 12:00-16:00 UTC
        if dt_time(12, 0) <= t < dt_time(16, 0):
            return "london_ny_overlap"

        # Tokyo/London overlap: 07:00-09:00 UTC
        if dt_time(7, 0) <= t < dt_time(9, 0):
            return "tokyo_london_overlap"

        # Individual sessions
        if dt_time(7, 0) <= t < dt_time(16, 0):
            return "london"
        if dt_time(12, 0) <= t < dt_time(21, 0):
            return "new_york"
        if dt_time(0, 0) <= t < dt_time(9, 0):
            return "tokyo"
        if t >= dt_time(21, 0) or t < dt_time(6, 0):
            return "sydney"

        return "low_liquidity"

    @classmethod
    def is_trading_hours(cls, dt: datetime) -> bool:
        """Check if forex market is open."""
        return not cls.is_weekend(dt)


def get_pip_size(pair: str) -> float:
    """Get pip size for a currency pair."""
    return PIP_SIZES.get(pair, DEFAULT_PIP_SIZE)


# =========================
# Data Download Functions
# =========================


def download_pair_oanda(
    pair: str,
    config: ForexDownloadConfig,
) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
    """
    Download historical data for a single currency pair from OANDA.

    Args:
        pair: Currency pair (e.g., "EUR_USD")
        config: Download configuration

    Returns:
        (pair, dataframe, error_message)
    """
    try:
        from adapters.oanda.market_data import OandaMarketDataAdapter

        # Create adapter
        adapter_config = {
            "api_key": config.api_key or os.environ.get("OANDA_API_KEY", ""),
            "account_id": config.account_id or os.environ.get("OANDA_ACCOUNT_ID", ""),
            "practice": config.practice,
            "price_type": config.price_type,
        }

        adapter = OandaMarketDataAdapter(
            vendor=ExchangeVendor.OANDA,
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

        logger.info(f"Downloading {pair}: {start_dt.date()} to {end_dt.date()}")

        # Download bars in chunks (OANDA max 5000 per request)
        all_records = []
        current_start = start_ts
        chunk_size = 5000

        while current_start < end_ts:
            # Apply rate limiting
            time.sleep(config.rate_limit_delay)

            bars = adapter.get_bars(
                symbol=pair,
                timeframe=config.timeframe,
                start_ts=current_start,
                end_ts=end_ts,
                limit=chunk_size,
            )

            if not bars:
                break

            for bar in bars:
                record = {
                    "timestamp": bar.ts // 1000,  # Convert to seconds
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume_base),  # Tick volume
                }

                # Calculate spread in pips (stored in volume_quote)
                if bar.volume_quote is not None:
                    record["spread_pips"] = float(bar.volume_quote)
                else:
                    record["spread_pips"] = np.nan

                if config.price_type == "bid_ask":
                    record.update(
                        {
                            "open_bid": float(bar.bid_open) if bar.bid_open is not None else np.nan,
                            "high_bid": float(bar.bid_high) if bar.bid_high is not None else np.nan,
                            "low_bid": float(bar.bid_low) if bar.bid_low is not None else np.nan,
                            "close_bid": (
                                float(bar.bid_close) if bar.bid_close is not None else np.nan
                            ),
                            "open_ask": float(bar.ask_open) if bar.ask_open is not None else np.nan,
                            "high_ask": float(bar.ask_high) if bar.ask_high is not None else np.nan,
                            "low_ask": float(bar.ask_low) if bar.ask_low is not None else np.nan,
                            "close_ask": (
                                float(bar.ask_close) if bar.ask_close is not None else np.nan
                            ),
                        }
                    )

                all_records.append(record)

            # Move to next chunk
            if bars:
                current_start = bars[-1].ts + 1
            else:
                break

            logger.debug(f"{pair}: Downloaded {len(all_records)} bars so far")

        if not all_records:
            return pair, None, "No data returned"

        # Convert to DataFrame
        df = pd.DataFrame(all_records)

        # Sort by timestamp
        df = (
            df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        )

        # Add symbol column
        df["symbol"] = pair

        # Add session labels if requested
        if config.add_session_labels:
            df["session"] = df["timestamp"].apply(
                lambda ts: ForexCalendar.get_active_session(
                    datetime.fromtimestamp(ts, tz=timezone.utc)
                )
            )

        # Filter weekends if requested
        if config.filter_weekends:
            df = _filter_weekends(df)

        # Resample if requested
        if config.resample_to:
            df = _resample_bars(df, config.resample_to)

        logger.info(f"Downloaded {pair}: {len(df)} bars")
        return pair, df, None

    except ImportError as e:
        return pair, None, f"OANDA adapter not available: {e}"
    except Exception as e:
        logger.exception(f"Error downloading {pair}")
        return pair, None, str(e)


def download_pair_dukascopy(
    pair: str,
    config: ForexDownloadConfig,
) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
    """
    Download a currency pair from Dukascopy's free public tick feed (bi5).

    No credentials required. Bars are aggregated from ticks by the adapter and
    written in the SAME schema as the OANDA path (timestamp/open/high/low/close/
    volume/spread_pips [+ bid/ask when price_type=bid_ask]), so downstream
    features/targets are provider-agnostic.
    """
    try:
        from adapters.dukascopy.market_data import DukascopyMarketDataAdapter

        adapter = DukascopyMarketDataAdapter(
            vendor=ExchangeVendor.DUKASCOPY,
            config={"max_hours": int(getattr(config, "dukascopy_max_hours", 0)) or 24 * 45},
        )
        adapter.connect()

        end_dt = datetime.now(timezone.utc)
        if config.end_date:
            end_dt = datetime.strptime(config.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if config.start_date:
            start_dt = datetime.strptime(config.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            start_dt = end_dt - timedelta(days=config.lookback_days)

        logger.info(f"Downloading {pair} (Dukascopy): {start_dt.date()} to {end_dt.date()}")
        bars = adapter.get_bars(
            symbol=pair,
            timeframe=config.timeframe,
            start_ts=int(start_dt.timestamp() * 1000),
            end_ts=int(end_dt.timestamp() * 1000),
            limit=10_000_000,  # keep the full range; adapter caps via max_hours
        )
        if not bars:
            return pair, None, "No data returned (weekend/holiday or out-of-range?)"

        records = []
        for bar in bars:
            record = {
                "timestamp": bar.ts // 1000,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume_base),
                "spread_pips": float(bar.volume_quote) if bar.volume_quote is not None else np.nan,
            }
            if config.price_type == "bid_ask":
                record.update(
                    {
                        "open_bid": float(bar.bid_open) if bar.bid_open is not None else np.nan,
                        "high_bid": float(bar.bid_high) if bar.bid_high is not None else np.nan,
                        "low_bid": float(bar.bid_low) if bar.bid_low is not None else np.nan,
                        "close_bid": float(bar.bid_close) if bar.bid_close is not None else np.nan,
                        "open_ask": float(bar.ask_open) if bar.ask_open is not None else np.nan,
                        "high_ask": float(bar.ask_high) if bar.ask_high is not None else np.nan,
                        "low_ask": float(bar.ask_low) if bar.ask_low is not None else np.nan,
                        "close_ask": float(bar.ask_close) if bar.ask_close is not None else np.nan,
                    }
                )
            records.append(record)

        df = (
            pd.DataFrame(records)
            .sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"])
            .reset_index(drop=True)
        )
        df["symbol"] = pair
        if config.add_session_labels:
            df["session"] = df["timestamp"].apply(
                lambda ts: ForexCalendar.get_active_session(
                    datetime.fromtimestamp(ts, tz=timezone.utc)
                )
            )
        if config.filter_weekends:
            df = _filter_weekends(df)
        if config.resample_to:
            df = _resample_bars(df, config.resample_to)
        logger.info(f"Downloaded {pair} (Dukascopy): {len(df)} bars")
        return pair, df, None

    except ImportError as e:
        return pair, None, f"Dukascopy adapter not available: {e}"
    except Exception as e:
        logger.exception(f"Error downloading {pair} from Dukascopy")
        return pair, None, str(e)


def _download_pair(pair: str, config: ForexDownloadConfig):
    """Dispatch to the configured forex data provider."""
    if str(getattr(config, "provider", "oanda")).lower() == "dukascopy":
        return download_pair_dukascopy(pair, config)
    return download_pair_oanda(pair, config)


def _filter_weekends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove data points during forex weekend closure.

    Args:
        df: DataFrame with timestamp column

    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df

    # Convert timestamp to datetime for weekend check
    def is_weekend(ts: int) -> bool:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return ForexCalendar.is_weekend(dt)

    mask = ~df["timestamp"].apply(is_weekend)
    return df[mask].reset_index(drop=True)


def _resample_bars(df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
    """
    Resample bars to a larger timeframe.

    Args:
        df: DataFrame with OHLCV columns
        target_timeframe: Target timeframe (e.g., "4h", "1d")

    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df

    # Map timeframe to pandas frequency
    freq_map = {
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "1d": "1D",
        "1w": "1W",
    }

    freq = freq_map.get(target_timeframe.lower())
    if freq is None:
        logger.warning(f"Unknown target timeframe: {target_timeframe}, skipping resample")
        return df

    # Convert to datetime index
    df_copy = df.copy()
    df_copy["datetime"] = pd.to_datetime(df_copy["timestamp"], unit="s", utc=True)
    df_copy = df_copy.set_index("datetime")

    # Define aggregation rules
    agg_dict = {
        "timestamp": "first",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # Handle optional columns
    if "spread_pips" in df_copy.columns:
        agg_dict["spread_pips"] = "mean"
    if "symbol" in df_copy.columns:
        agg_dict["symbol"] = "first"
    if "session" in df_copy.columns:
        agg_dict["session"] = "first"  # Session at bar open
    for prefix in ["bid", "ask"]:
        if f"open_{prefix}" in df_copy.columns:
            agg_dict[f"open_{prefix}"] = "first"
            agg_dict[f"high_{prefix}"] = "max"
            agg_dict[f"low_{prefix}"] = "min"
            agg_dict[f"close_{prefix}"] = "last"

    # Resample
    resampled = df_copy.resample(freq).agg(agg_dict)

    # Drop rows with NaN (incomplete bars)
    resampled = resampled.dropna(subset=["close"])

    return resampled.reset_index(drop=True)


# =========================
# File I/O
# =========================


def save_dataframe(
    df: pd.DataFrame,
    pair: str,
    config: ForexDownloadConfig,
) -> str:
    """
    Save DataFrame to file.

    Args:
        df: Data to save
        pair: Currency pair name
        config: Configuration

    Returns:
        Path to saved file
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize pair name for filename
    safe_pair = pair.replace("/", "_").replace("\\", "_")

    if config.output_format == "parquet":
        filepath = output_dir / f"{safe_pair}.parquet"
        df.to_parquet(filepath, index=False)
    elif config.output_format == "feather":
        filepath = output_dir / f"{safe_pair}.feather"
        df.to_feather(filepath)
    else:
        filepath = output_dir / f"{safe_pair}.csv"
        df.to_csv(filepath, index=False)

    return str(filepath)


def load_pairs_from_file(filepath: str) -> List[str]:
    """
    Load currency pairs from a text file (one per line).

    Args:
        filepath: Path to file

    Returns:
        List of currency pairs
    """
    with open(filepath, "r") as f:
        pairs = [
            line.strip().upper().replace("/", "_")
            for line in f
            if line.strip() and not line.startswith("#")
        ]
    return pairs


# =========================
# Main Download Runner
# =========================


def download_all_pairs(config: ForexDownloadConfig) -> Dict[str, Any]:
    """
    Download data for all configured pairs.

    Args:
        config: Download configuration

    Returns:
        Summary dict with success/failed pairs
    """
    # Determine pairs to download
    pairs = config.pairs.copy() if config.pairs else []

    if config.pairs_file:
        pairs.extend(load_pairs_from_file(config.pairs_file))

    # If no pairs specified, use majors
    if not pairs:
        pairs = MAJOR_PAIRS

    # Normalize pair names
    pairs = [p.upper().replace("/", "_") for p in pairs]
    pairs = list(set(pairs))  # Remove duplicates

    logger.info(f"Will download {len(pairs)} pairs: {pairs}")

    # Check for existing files if skip_existing is enabled
    output_dir = Path(config.output_dir)
    if config.skip_existing:
        existing = set()
        for ext in ["parquet", "feather", "csv"]:
            for f in output_dir.glob(f"*.{ext}"):
                existing.add(f.stem)

        pairs_to_download = [p for p in pairs if p not in existing]
        skipped = len(pairs) - len(pairs_to_download)
        if skipped > 0:
            logger.info(f"Skipping {skipped} existing pairs")
        pairs = pairs_to_download

    if not pairs:
        logger.info("No pairs to download (all exist)")
        return {"success": [], "failed": [], "skipped": skipped}

    # Download pairs (sequential due to OANDA rate limits)
    success = []
    failed = []

    for pair in pairs:
        pair, df, error = _download_pair(pair, config)

        if error:
            logger.error(f"Failed to download {pair}: {error}")
            failed.append((pair, error))
        elif df is not None and not df.empty:
            filepath = save_dataframe(df, pair, config)
            logger.info(f"Saved {pair} to {filepath}")
            success.append(pair)
        else:
            logger.warning(f"No data for {pair}")
            failed.append((pair, "No data"))

    # Summary
    summary = {
        "success": success,
        "failed": failed,
        "total_downloaded": len(success),
        "total_failed": len(failed),
    }

    logger.info(f"Download complete: {len(success)} success, {len(failed)} failed")

    return summary


# =========================
# CLI
# =========================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download historical forex data from OANDA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download major pairs
  python scripts/download_forex_data.py --majors --start 2020-01-01

  # Download specific pairs
  python scripts/download_forex_data.py --pairs EUR_USD GBP_USD --start 2020-01-01

  # Download with 4-hour timeframe
  python scripts/download_forex_data.py --all --start 2020-01-01 --timeframe 4h
        """,
    )

    # Pair selection
    pair_group = parser.add_mutually_exclusive_group()
    pair_group.add_argument(
        "--pairs",
        nargs="+",
        help="Specific pairs to download (e.g., EUR_USD GBP_USD)",
    )
    pair_group.add_argument(
        "--pairs-file",
        help="File containing pairs (one per line)",
    )
    pair_group.add_argument(
        "--majors",
        action="store_true",
        help="Download major pairs only (EUR_USD, GBP_USD, USD_JPY, etc.)",
    )
    pair_group.add_argument(
        "--minors",
        action="store_true",
        help="Download minor (cross) pairs",
    )
    pair_group.add_argument(
        "--all",
        action="store_true",
        help="Download all supported pairs",
    )

    # Date range
    parser.add_argument(
        "--start",
        dest="start_date",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        dest="end_date",
        help="End date (YYYY-MM-DD), defaults to today",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365 * 3,
        help="Lookback period in days if start not specified (default: 1095)",
    )

    # Timeframe
    parser.add_argument(
        "--timeframe",
        default="1h",
        help="Candle timeframe: 1m, 5m, 15m, 1h, 4h, 1d (default: 1h)",
    )
    parser.add_argument(
        "--resample",
        dest="resample_to",
        help="Resample to larger timeframe after download (e.g., 4h)",
    )

    # Output
    parser.add_argument(
        "--output",
        dest="output_dir",
        default="data/raw_forex",
        help="Output directory (default: data/raw_forex)",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=["parquet", "feather", "csv"],
        default="parquet",
        help="Output format (default: parquet)",
    )

    # Processing options
    parser.add_argument(
        "--include-weekends",
        action="store_true",
        dest="include_weekends",
        help="Include weekend data (normally filtered out)",
    )
    parser.add_argument(
        "--no-spread",
        action="store_true",
        help="Don't include bid/ask spread data",
    )
    parser.add_argument(
        "--no-session-labels",
        action="store_true",
        help="Don't add session labels to bars",
    )
    parser.add_argument(
        "--price-type",
        choices=["mid", "bid", "ask", "bid_ask"],
        default="mid",
        help="Price type to download: mid, bid, ask, or bid_ask (dual feed)",
    )

    # Provider config
    parser.add_argument(
        "--provider",
        choices=["oanda", "dukascopy"],
        default="oanda",
        help="Forex data provider: oanda (needs keys) or dukascopy (free public bi5 tick feed)",
    )
    parser.add_argument(
        "--api-key",
        help="OANDA API key (or set OANDA_API_KEY env var)",
    )
    parser.add_argument(
        "--account-id",
        help="OANDA account ID (or set OANDA_ACCOUNT_ID env var)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live (not practice) API endpoint",
    )

    # Misc
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if files exist",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
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

    # Build configuration
    config = ForexDownloadConfig(
        provider=args.provider,
        api_key=args.api_key,
        account_id=args.account_id,
        practice=not args.live,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_days=args.lookback_days,
        timeframe=args.timeframe,
        output_dir=args.output_dir,
        output_format=args.output_format,
        filter_weekends=not args.include_weekends,
        include_spread=not args.no_spread,
        add_session_labels=not args.no_session_labels,
        resample_to=args.resample_to,
        skip_existing=not args.force,
        price_type=args.price_type,
    )

    # Determine pairs
    if args.pairs:
        config.pairs = args.pairs
    elif args.pairs_file:
        config.pairs_file = args.pairs_file
    elif args.majors:
        config.pairs = MAJOR_PAIRS
    elif args.minors:
        config.pairs = MINOR_PAIRS
    elif args.all:
        config.pairs = ALL_PAIRS
    else:
        # Default to majors
        config.pairs = MAJOR_PAIRS

    # Run download
    try:
        summary = download_all_pairs(config)

        if summary["failed"]:
            logger.error(f"Failed pairs: {summary['failed']}")
            return 1

        return 0

    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Download failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
