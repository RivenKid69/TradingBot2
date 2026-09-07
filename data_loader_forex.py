# -*- coding: utf-8 -*-
"""
data_loader_forex.py
Forex-specific data loader for training and backtesting.

This module extends data_loader_multi_asset.py with forex-specific features:
- OANDA adapter integration
- Swap rate data loading and merging
- Economic calendar event proximity
- Interest rate differential features
- Session labeling (Sydney/Tokyo/London/NY)
- Weekend gap filtering

Usage:
    from data_loader_forex import load_forex_data, load_from_oanda

    # Load from local files
    forex_dfs, obs_shapes = load_forex_data(
        paths=["data/raw_forex/EUR_USD.parquet"],
        timeframe="4h",
        merge_swaps=True,
        merge_calendar=True,
    )

    # Load directly from OANDA
    forex_dfs, obs_shapes = load_from_oanda(
        pairs=["EUR_USD", "GBP_USD"],
        timeframe="4h",
        start_date="2023-01-01",
        end_date="2024-12-31",
    )

    # Load with interest rate features
    forex_dfs, obs_shapes = load_forex_data(
        paths=["data/raw_forex/*.parquet"],
        merge_rates=True,
        rate_dir="data/forex/rates/",
    )

Author: Sultan Suyunbayev
Date: 2025-11-30
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, time as dt_time, timedelta, timezone
from enum import Enum
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Default data directories
DEFAULT_FOREX_DIR = "data/raw_forex"
DEFAULT_SWAP_DIR = "data/forex/swaps"
DEFAULT_RATE_DIR = "data/forex/rates"
DEFAULT_CALENDAR_DIR = "data/forex/calendar"


# Forex trading sessions (UTC)
FOREX_SESSIONS = {
    "sydney": (dt_time(21, 0), dt_time(6, 0)),
    "tokyo": (dt_time(0, 0), dt_time(9, 0)),
    "london": (dt_time(7, 0), dt_time(16, 0)),
    "new_york": (dt_time(12, 0), dt_time(21, 0)),
}

# Session liquidity factors (relative to average)
SESSION_LIQUIDITY = {
    "sydney": 0.6,
    "tokyo": 0.8,
    "london": 1.3,
    "new_york": 1.2,
    "london_ny_overlap": 1.5,
    "tokyo_london_overlap": 1.0,
    "low_liquidity": 0.4,
}


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_forex_data(
    paths: Union[str, List[str]],
    timeframe: str = "4h",
    filter_weekends: bool = True,
    merge_swaps: bool = False,
    merge_rates: bool = False,
    merge_calendar: bool = False,
    swap_dir: Optional[str] = None,
    rate_dir: Optional[str] = None,
    calendar_dir: Optional[str] = None,
    add_session_features: bool = True,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Tuple[int, ...]]]:
    """
    Load forex data from files with optional feature merging.

    Args:
        paths: Path or list of paths (supports glob patterns)
        timeframe: Timeframe string (e.g., "4h")
        filter_weekends: Remove weekend bars (market closed)
        merge_swaps: Merge swap rate data
        merge_rates: Merge interest rate differential data
        merge_calendar: Merge economic calendar proximity
        swap_dir: Directory with swap rate files
        rate_dir: Directory with interest rate files
        calendar_dir: Directory with calendar files
        add_session_features: Add session indicator features
        start_date: Filter start date (YYYY-MM-DD)
        end_date: Filter end date (YYYY-MM-DD)

    Returns:
        (dict of pair -> DataFrame, dict of pair -> obs_shape)
    """
    # Resolve paths
    if isinstance(paths, str):
        paths = [paths]

    all_files = []
    for path in paths:
        if "*" in path or "?" in path:
            all_files.extend(glob(path))
        else:
            all_files.append(path)

    if not all_files:
        logger.warning(f"No files found matching paths: {paths}")
        return {}, {}

    dataframes = {}
    obs_shapes = {}

    for filepath in all_files:
        try:
            pair, df = _load_single_forex_file(
                filepath,
                timeframe=timeframe,
                filter_weekends=filter_weekends,
                start_date=start_date,
                end_date=end_date,
            )

            if df is None or df.empty:
                continue

            # Add session features
            if add_session_features:
                df = _add_session_features(df)

            # Merge swap rates
            if merge_swaps:
                df = _merge_swap_rates(df, pair, swap_dir or DEFAULT_SWAP_DIR)

            # Merge interest rate differentials
            if merge_rates:
                df = _merge_interest_rates(df, pair, rate_dir or DEFAULT_RATE_DIR)

            # Merge economic calendar proximity
            if merge_calendar:
                df = _merge_calendar_proximity(df, pair, calendar_dir or DEFAULT_CALENDAR_DIR)

            # Calculate observation shape
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            obs_shape = (len(numeric_cols),)

            dataframes[pair] = df
            obs_shapes[pair] = obs_shape

            logger.info(f"Loaded {pair}: {len(df)} bars, {len(numeric_cols)} features")

        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            continue

    return dataframes, obs_shapes


def _load_single_forex_file(
    filepath: str,
    timeframe: str,
    filter_weekends: bool,
    start_date: Optional[str],
    end_date: Optional[str],
) -> Tuple[str, Optional[pd.DataFrame]]:
    """Load a single forex data file."""
    path = Path(filepath)

    if not path.exists():
        logger.warning(f"File not found: {filepath}")
        return "", None

    # Load based on format
    if path.suffix == ".parquet":
        df = pd.read_parquet(filepath)
    elif path.suffix == ".feather":
        df = pd.read_feather(filepath)
    elif path.suffix == ".csv":
        df = pd.read_csv(filepath)
    else:
        logger.warning(f"Unsupported format: {path.suffix}")
        return "", None

    # Extract pair from filename or column
    if "symbol" in df.columns:
        pair = df["symbol"].iloc[0]
    else:
        pair = path.stem.replace("_data", "").replace("-", "_").upper()

    # Ensure timestamp column
    if "timestamp" not in df.columns:
        ts_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
        if ts_cols:
            df = df.rename(columns={ts_cols[0]: "timestamp"})
        else:
            logger.warning(f"No timestamp column in {filepath}")
            return pair, None

    # Convert timestamp
    if df["timestamp"].dtype == "object":
        df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**9
    elif df["timestamp"].max() > 10_000_000_000:
        df["timestamp"] = df["timestamp"] // 1000

    df["timestamp"] = df["timestamp"].astype("int64")

    # Filter date range
    if start_date:
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        ).timestamp())
        df = df[df["timestamp"] >= start_ts]

    if end_date:
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        ).timestamp())
        df = df[df["timestamp"] <= end_ts]

    # Filter weekends
    if filter_weekends:
        df = _filter_forex_weekends(df)

    # Sort and deduplicate
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    return pair, df


def _filter_forex_weekends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove data points during forex weekend closure.

    Forex closes Friday 5pm ET (~21:00-22:00 UTC depending on DST)
    and opens Sunday 5pm ET.
    """
    if df.empty:
        return df

    def is_weekend(ts: int) -> bool:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
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

    mask = ~df["timestamp"].apply(is_weekend)
    return df[mask].reset_index(drop=True)


# =============================================================================
# SESSION FEATURES
# =============================================================================

def _add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trading session indicator features.

    Features added:
    - session: Active session name
    - session_liquidity: Session liquidity factor
    - is_overlap: Whether in session overlap period
    - One-hot encoded session columns
    """
    if df.empty or "timestamp" not in df.columns:
        return df

    df = df.copy()

    sessions = []
    liquidity = []
    is_overlap = []

    for ts in df["timestamp"]:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        session = _get_active_session(dt)
        sessions.append(session)
        liquidity.append(SESSION_LIQUIDITY.get(session, 0.5))
        is_overlap.append("overlap" in session)

    df["session"] = sessions
    df["session_liquidity"] = liquidity
    df["is_session_overlap"] = is_overlap

    # One-hot encode sessions
    session_dummies = pd.get_dummies(df["session"], prefix="session")
    df = pd.concat([df, session_dummies], axis=1)

    return df


def _get_active_session(dt: datetime) -> str:
    """Get the primary active trading session."""
    t = dt.time()

    # Check overlaps first (highest liquidity)
    if dt_time(12, 0) <= t < dt_time(16, 0):
        return "london_ny_overlap"
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


# =============================================================================
# SWAP RATE INTEGRATION
# =============================================================================

def _merge_swap_rates(
    df: pd.DataFrame,
    pair: str,
    swap_dir: str,
) -> pd.DataFrame:
    """
    Merge swap rate data into price DataFrame.

    Adds columns:
    - long_swap: Daily swap for long positions (pips)
    - short_swap: Daily swap for short positions (pips)
    """
    swap_file = Path(swap_dir) / f"{pair}_swaps.parquet"

    if not swap_file.exists():
        swap_file = Path(swap_dir) / f"{pair}_swaps.csv"

    if not swap_file.exists():
        logger.debug(f"No swap data found for {pair}")
        df["long_swap"] = 0.0
        df["short_swap"] = 0.0
        return df

    try:
        if str(swap_file).endswith(".parquet"):
            swaps = pd.read_parquet(swap_file)
        else:
            swaps = pd.read_csv(swap_file)

        # Convert dates to timestamps
        if "date" in swaps.columns:
            swaps["swap_date"] = pd.to_datetime(swaps["date"]).dt.date

        # Add date column to main df for merging
        df = df.copy()
        df["bar_date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date

        # Merge on date
        swaps_subset = swaps[["swap_date", "long_swap", "short_swap"]].copy() if "swap_date" in swaps.columns else swaps[["date", "long_swap", "short_swap"]].copy()

        if "swap_date" in swaps_subset.columns:
            df = df.merge(
                swaps_subset,
                left_on="bar_date",
                right_on="swap_date",
                how="left",
            )
            df = df.drop(columns=["swap_date", "bar_date"], errors="ignore")
        else:
            swaps_subset["date"] = pd.to_datetime(swaps_subset["date"]).dt.date
            df = df.merge(
                swaps_subset,
                left_on="bar_date",
                right_on="date",
                how="left",
            )
            df = df.drop(columns=["date", "bar_date"], errors="ignore")

        # Fill missing swaps with 0
        df["long_swap"] = df["long_swap"].fillna(0.0)
        df["short_swap"] = df["short_swap"].fillna(0.0)

        logger.debug(f"Merged swap rates for {pair}")

    except Exception as e:
        logger.warning(f"Error merging swaps for {pair}: {e}")
        df["long_swap"] = 0.0
        df["short_swap"] = 0.0

    return df


# =============================================================================
# INTEREST RATE INTEGRATION
# =============================================================================

def _merge_interest_rates(
    df: pd.DataFrame,
    pair: str,
    rate_dir: str,
) -> pd.DataFrame:
    """
    Merge interest rate differential data.

    Adds columns:
    - rate_differential: Base rate - Quote rate (annual %)
    - base_rate: Base currency interest rate
    - quote_rate: Quote currency interest rate
    """
    # Parse pair to get currencies
    parts = pair.split("_")
    if len(parts) != 2:
        logger.debug(f"Cannot parse pair: {pair}")
        df["rate_differential"] = 0.0
        return df

    base_ccy, quote_ccy = parts

    # Load rate files
    base_rate = _load_rate_series(base_ccy, rate_dir)
    quote_rate = _load_rate_series(quote_ccy, rate_dir)

    if base_rate is None or quote_rate is None:
        df["rate_differential"] = 0.0
        df["base_rate"] = np.nan
        df["quote_rate"] = np.nan
        return df

    df = df.copy()
    # Use datetime64 for merge_asof compatibility (not date objects)
    df["bar_datetime"] = pd.to_datetime(df["timestamp"], unit="s").dt.normalize()

    # Merge base rate
    base_rate_df = base_rate.reset_index()
    base_rate_df.columns = ["rate_datetime", "base_rate"]
    base_rate_df["rate_datetime"] = pd.to_datetime(base_rate_df["rate_datetime"]).dt.normalize()

    df = pd.merge_asof(
        df.sort_values("bar_datetime"),
        base_rate_df.sort_values("rate_datetime"),
        left_on="bar_datetime",
        right_on="rate_datetime",
        direction="backward",
    )
    df = df.drop(columns=["rate_datetime"], errors="ignore")

    # Merge quote rate
    quote_rate_df = quote_rate.reset_index()
    quote_rate_df.columns = ["rate_datetime", "quote_rate"]
    quote_rate_df["rate_datetime"] = pd.to_datetime(quote_rate_df["rate_datetime"]).dt.normalize()

    df = pd.merge_asof(
        df.sort_values("bar_datetime"),
        quote_rate_df.sort_values("rate_datetime"),
        left_on="bar_datetime",
        right_on="rate_datetime",
        direction="backward",
    )
    df = df.drop(columns=["rate_datetime", "bar_datetime"], errors="ignore")

    # Calculate differential
    df["rate_differential"] = df["base_rate"].fillna(0) - df["quote_rate"].fillna(0)

    # Resort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.debug(f"Merged interest rates for {pair}")
    return df


def _load_rate_series(currency: str, rate_dir: str) -> Optional[pd.Series]:
    """Load interest rate series for a currency."""
    rate_file = Path(rate_dir) / f"{currency}_rates.parquet"

    if not rate_file.exists():
        rate_file = Path(rate_dir) / f"{currency}_rates.csv"

    if not rate_file.exists():
        return None

    try:
        if str(rate_file).endswith(".parquet"):
            df = pd.read_parquet(rate_file)
        else:
            df = pd.read_csv(rate_file)

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        return df["rate"]

    except Exception as e:
        logger.warning(f"Error loading rates for {currency}: {e}")
        return None


# =============================================================================
# ECONOMIC CALENDAR INTEGRATION
# =============================================================================

def _merge_calendar_proximity(
    df: pd.DataFrame,
    pair: str,
    calendar_dir: str,
) -> pd.DataFrame:
    """
    Add economic calendar event proximity features.

    Adds columns:
    - hours_to_next_event: Hours until next high-impact event
    - hours_since_last_event: Hours since last high-impact event
    - is_event_window: Whether within ±2 hours of event
    """
    calendar_file = Path(calendar_dir) / "economic_calendar.parquet"

    if not calendar_file.exists():
        calendar_file = Path(calendar_dir) / "economic_calendar.csv"

    if not calendar_file.exists():
        df["hours_to_next_event"] = 999.0
        df["hours_since_last_event"] = 999.0
        df["is_event_window"] = False
        return df

    try:
        if str(calendar_file).endswith(".parquet"):
            calendar = pd.read_parquet(calendar_file)
        else:
            calendar = pd.read_csv(calendar_file)

        # Filter to relevant currencies
        parts = pair.split("_")
        relevant_currencies = set(parts)
        calendar = calendar[calendar["currency"].isin(relevant_currencies)]

        # Filter to high-impact events
        if "impact" in calendar.columns:
            calendar = calendar[calendar["impact"] == "high"]

        if calendar.empty:
            df["hours_to_next_event"] = 999.0
            df["hours_since_last_event"] = 999.0
            df["is_event_window"] = False
            return df

        # Convert datetime
        if "datetime" in calendar.columns:
            calendar["event_ts"] = pd.to_datetime(calendar["datetime"]).astype("int64") // 10**9
        elif "date" in calendar.columns and "time" in calendar.columns:
            calendar["event_ts"] = pd.to_datetime(
                calendar["date"] + " " + calendar["time"]
            ).astype("int64") // 10**9

        event_timestamps = calendar["event_ts"].sort_values().values

        # Calculate proximity for each bar
        df = df.copy()
        hours_to_next = []
        hours_since_last = []
        is_window = []

        for ts in df["timestamp"]:
            # Find next event
            next_events = event_timestamps[event_timestamps > ts]
            if len(next_events) > 0:
                hours_next = (next_events[0] - ts) / 3600
            else:
                hours_next = 999.0

            # Find last event
            prev_events = event_timestamps[event_timestamps <= ts]
            if len(prev_events) > 0:
                hours_prev = (ts - prev_events[-1]) / 3600
            else:
                hours_prev = 999.0

            hours_to_next.append(hours_next)
            hours_since_last.append(hours_prev)
            is_window.append(hours_next <= 2 or hours_prev <= 2)

        df["hours_to_next_event"] = hours_to_next
        df["hours_since_last_event"] = hours_since_last
        df["is_event_window"] = is_window

        logger.debug(f"Merged calendar proximity for {pair}")

    except Exception as e:
        logger.warning(f"Error merging calendar for {pair}: {e}")
        df["hours_to_next_event"] = 999.0
        df["hours_since_last_event"] = 999.0
        df["is_event_window"] = False

    return df


# =============================================================================
# OANDA ADAPTER INTEGRATION
# =============================================================================

def load_from_oanda(
    pairs: List[str],
    timeframe: str = "4h",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
    account_id: Optional[str] = None,
    practice: bool = True,
    merge_swaps: bool = False,
    merge_rates: bool = False,
    merge_calendar: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Tuple[int, ...]]]:
    """
    Load forex data directly from OANDA API.

    Args:
        pairs: List of currency pairs (e.g., ["EUR_USD", "GBP_USD"])
        timeframe: Candle timeframe
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_key: OANDA API key
        account_id: OANDA account ID
        practice: Use practice environment
        merge_swaps: Include swap rate data
        merge_rates: Include interest rate data
        merge_calendar: Include calendar data

    Returns:
        (dict of pair -> DataFrame, dict of pair -> obs_shape)
    """
    try:
        from adapters.oanda.market_data import OandaMarketDataAdapter
        from adapters.models import ExchangeVendor
    except ImportError:
        logger.error("OANDA adapter not available")
        return {}, {}

    api_key = api_key or os.environ.get("OANDA_API_KEY")
    account_id = account_id or os.environ.get("OANDA_ACCOUNT_ID")

    if not api_key or not account_id:
        logger.error("OANDA API key and account ID required")
        return {}, {}

    adapter = OandaMarketDataAdapter(
        vendor=ExchangeVendor.OANDA,
        config={
            "api_key": api_key,
            "account_id": account_id,
            "practice": practice,
        },
    )

    # Calculate timestamps
    end_dt = datetime.now(timezone.utc)
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start_dt = end_dt - timedelta(days=365)

    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    dataframes = {}
    obs_shapes = {}

    for pair in pairs:
        try:
            logger.info(f"Fetching {pair} from OANDA")

            bars = adapter.get_bars(
                symbol=pair,
                timeframe=timeframe,
                start_ts=start_ts,
                end_ts=end_ts,
                limit=50000,
            )

            if not bars:
                logger.warning(f"No data for {pair}")
                continue

            # Convert to DataFrame
            records = []
            for bar in bars:
                records.append({
                    "timestamp": bar.ts // 1000,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume_base),
                })

            df = pd.DataFrame(records)
            df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

            # Add session features
            df = _add_session_features(df)

            # Merge auxiliary data if requested
            if merge_swaps:
                df = _merge_swap_rates(df, pair, DEFAULT_SWAP_DIR)
            if merge_rates:
                df = _merge_interest_rates(df, pair, DEFAULT_RATE_DIR)
            if merge_calendar:
                df = _merge_calendar_proximity(df, pair, DEFAULT_CALENDAR_DIR)

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            obs_shapes[pair] = (len(numeric_cols),)
            dataframes[pair] = df

            logger.info(f"Loaded {pair}: {len(df)} bars")

        except Exception as e:
            logger.error(f"Error fetching {pair}: {e}")
            continue

    return dataframes, obs_shapes


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def list_available_pairs(data_dir: str = DEFAULT_FOREX_DIR) -> List[str]:
    """List available forex pairs in data directory."""
    pairs = []

    for ext in ["parquet", "feather", "csv"]:
        for path in Path(data_dir).glob(f"*.{ext}"):
            pair = path.stem.upper()
            if pair not in pairs:
                pairs.append(pair)

    return sorted(pairs)


def get_pair_info(pair: str, data_dir: str = DEFAULT_FOREX_DIR) -> Optional[Dict[str, Any]]:
    """Get information about a forex pair's data file."""
    for ext in ["parquet", "feather", "csv"]:
        path = Path(data_dir) / f"{pair}.{ext}"
        if path.exists():
            try:
                if ext == "parquet":
                    df = pd.read_parquet(path)
                elif ext == "feather":
                    df = pd.read_feather(path)
                else:
                    df = pd.read_csv(path)

                ts_min = df["timestamp"].min()
                ts_max = df["timestamp"].max()

                return {
                    "pair": pair,
                    "filepath": str(path),
                    "format": ext,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "start_date": datetime.fromtimestamp(ts_min).strftime("%Y-%m-%d"),
                    "end_date": datetime.fromtimestamp(ts_max).strftime("%Y-%m-%d"),
                }
            except Exception as e:
                logger.warning(f"Error reading {path}: {e}")
                continue

    return None


# =============================================================================
# CLI INTEGRATION
# =============================================================================

def main():
    """CLI for testing data loader."""
    import argparse

    parser = argparse.ArgumentParser(description="Forex data loader utility")
    parser.add_argument("--list", action="store_true", help="List available pairs")
    parser.add_argument("--info", help="Show info for a pair")
    parser.add_argument("--load", nargs="+", help="Load and show stats for pairs")
    parser.add_argument("--dir", default=DEFAULT_FOREX_DIR, help="Data directory")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.list:
        pairs = list_available_pairs(args.dir)
        print(f"Available pairs in {args.dir}:")
        for pair in pairs:
            print(f"  {pair}")

    elif args.info:
        info = get_pair_info(args.info.upper(), args.dir)
        if info:
            for k, v in info.items():
                print(f"  {k}: {v}")
        else:
            print(f"No data found for {args.info}")

    elif args.load:
        paths = [f"{args.dir}/{p.upper()}.parquet" for p in args.load]
        dfs, shapes = load_forex_data(paths)
        for pair, df in dfs.items():
            print(f"{pair}: {len(df)} bars, shape={shapes[pair]}")
            print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
