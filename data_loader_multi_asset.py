# -*- coding: utf-8 -*-
"""
data_loader_multi_asset.py
------------------------------------------------------------------
Unified data loader for multi-asset support (crypto + stocks).

This module extends fetch_all_data_patch.py with:
- Support for stock data from Alpaca and Polygon
- Trading hours awareness for equities
- Asset-class specific preprocessing
- Unified data format for TradingEnv

Usage:
    from data_loader_multi_asset import load_multi_asset_data, AssetClass

    # Load crypto data
    crypto_dfs, crypto_obs = load_multi_asset_data(
        paths=["data/processed/BTCUSDT.feather"],
        asset_class=AssetClass.CRYPTO,
    )

    # Load stock data
    stock_dfs, stock_obs = load_multi_asset_data(
        paths=["data/stocks/AAPL.parquet"],
        asset_class=AssetClass.EQUITY,
    )

    # Or via adapter
    stock_dfs, stock_obs = load_from_adapter(
        vendor="alpaca",
        symbols=["AAPL", "MSFT", "GOOGL"],
        timeframe="4h",
        start_date="2023-01-01",
        end_date="2024-12-31",
    )
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class AssetClass(str, Enum):
    """Asset class for data loading."""

    CRYPTO = "crypto"
    EQUITY = "equity"
    FOREX = "forex"


class DataVendor(str, Enum):
    """Data source vendor."""

    BINANCE = "binance"
    ALPACA = "alpaca"
    POLYGON = "polygon"
    CSV = "csv"
    FEATHER = "feather"
    PARQUET = "parquet"
    # Forex vendors (Phase 1)
    OANDA = "oanda"
    IG = "ig"
    DUKASCOPY = "dukascopy"


# =============================================================================
# TIMEFRAME UTILITIES
# =============================================================================


def timeframe_to_seconds(tf: str) -> int:
    """Convert timeframe string to seconds."""
    tf_lower = tf.lower().strip()

    mappings = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
        "1w": 604800,
    }

    if tf_lower in mappings:
        return mappings[tf_lower]

    # Parse numeric format (e.g., "15min", "4hour")
    import re

    match = re.match(r"(\d+)(m|min|h|hour|d|day|w|week)", tf_lower)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        unit_map = {
            "m": 60,
            "min": 60,
            "h": 3600,
            "hour": 3600,
            "d": 86400,
            "day": 86400,
            "w": 604800,
            "week": 604800,
        }
        return value * unit_map.get(unit, 60)

    # Default to 4h
    logger.warning(f"Unknown timeframe '{tf}', defaulting to 4h")
    return 14400


def align_timestamp(ts: int, timeframe_seconds: int) -> int:
    """Align timestamp to timeframe boundary (floor)."""
    return (ts // timeframe_seconds) * timeframe_seconds


# =============================================================================
# FEAR & GREED LOADER (CRYPTO)
# =============================================================================

FNG_PATH = os.path.join("data", "fear_greed.csv")


def load_fear_greed(
    timeframe_seconds: int = 14400,
    path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load Fear & Greed index data for crypto.

    Args:
        timeframe_seconds: Timeframe for alignment
        path: Path to fear_greed.csv

    Returns:
        DataFrame with timestamp, fear_greed_value, fear_greed_value_norm
    """
    fng_path = path or FNG_PATH

    if not os.path.exists(fng_path):
        return pd.DataFrame(columns=["timestamp", "fear_greed_value"])

    fng = pd.read_csv(fng_path)

    # Find timestamp and value columns
    cols = {c.lower(): c for c in fng.columns}
    ts_col = (
        "timestamp"
        if "timestamp" in cols
        else next((c for c in fng.columns if "time" in c.lower()), "timestamp")
    )
    val_col = (
        "fear_greed_value"
        if "fear_greed_value" in fng.columns
        else ("value" if "value" in fng.columns else None)
    )

    fng = fng.rename(columns={ts_col: "timestamp"})
    if val_col and val_col != "fear_greed_value":
        fng = fng.rename(columns={val_col: "fear_greed_value"})

    # Normalize timestamp to seconds
    if fng["timestamp"].max() > 10_000_000_000:
        fng["timestamp"] = (fng["timestamp"] // 1000).astype("int64")
    else:
        fng["timestamp"] = fng["timestamp"].astype("int64")

    fng = fng[["timestamp", "fear_greed_value"]].copy()
    fng["fear_greed_value_norm"] = fng["fear_greed_value"].astype(float) / 100.0

    # Align to timeframe
    fng["timestamp"] = fng["timestamp"].apply(lambda x: align_timestamp(x, timeframe_seconds))
    fng = fng.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    return fng


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================


def load_from_file(
    path: Union[str, Path],
    asset_class: AssetClass = AssetClass.CRYPTO,
    timeframe: str = "4h",
) -> pd.DataFrame:
    """
    Load OHLCV data from file (CSV, Feather, Parquet).

    Args:
        path: Path to data file
        asset_class: Asset class (crypto/equity)
        timeframe: Timeframe for alignment

    Returns:
        DataFrame with standardized columns
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".feather":
        df = pd.read_feather(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Extract symbol from filename if not in data
    symbol = path.stem.upper()
    if "symbol" not in df.columns:
        df["symbol"] = symbol

    # Standardize columns
    df = _standardize_columns(df, asset_class, timeframe)

    return df


def _standardize_columns(
    df: pd.DataFrame,
    asset_class: AssetClass,
    timeframe: str = "4h",
) -> pd.DataFrame:
    """
    Standardize DataFrame columns for TradingEnv.

    Args:
        df: Raw DataFrame
        asset_class: Asset class
        timeframe: Timeframe for alignment

    Returns:
        Standardized DataFrame
    """
    df = df.copy()
    timeframe_seconds = timeframe_to_seconds(timeframe)

    # Column name mapping for different sources
    column_aliases = {
        "timestamp": ["timestamp", "ts", "time", "date", "datetime", "close_time", "t"],
        "open": ["open", "o", "Open"],
        "high": ["high", "h", "High"],
        "low": ["low", "l", "Low"],
        "close": ["close", "c", "Close"],
        "volume": ["volume", "v", "Volume", "vol"],
    }

    # Rename columns using aliases
    for target, aliases in column_aliases.items():
        if target not in df.columns:
            for alias in aliases:
                if alias in df.columns:
                    df = df.rename(columns={alias: target})
                    break

    # Handle timestamp
    if "timestamp" not in df.columns:
        if "open_time" in df.columns:
            ts = pd.to_numeric(df["open_time"], errors="coerce")
            if ts.max() > 10_000_000_000:
                ts = ts // 1000
            df["timestamp"] = ts + timeframe_seconds
        elif "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"]).astype("int64") // 10**9
        else:
            raise ValueError("No timestamp column found")
    else:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        if ts.max() > 10_000_000_000:
            ts = ts // 1000
        df["timestamp"] = ts.astype("int64")

    # Align timestamp to timeframe
    df["timestamp"] = df["timestamp"].apply(lambda x: align_timestamp(int(x), timeframe_seconds))

    # Ensure OHLCV columns exist
    required_float = ["open", "high", "low", "close", "volume"]
    for col in required_float:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = df[col].astype(float)

    # Add optional columns with defaults
    optional_columns = {
        "quote_asset_volume": lambda: df["close"] * df["volume"],
        "number_of_trades": lambda: 0,
        "taker_buy_base_asset_volume": lambda: 0.0,
        "taker_buy_quote_asset_volume": lambda: 0.0,
        "vwap": lambda: (df["high"] + df["low"] + df["close"]) / 3,
    }

    for col, default_fn in optional_columns.items():
        if col not in df.columns:
            df[col] = default_fn()

    # Asset-class specific handling
    if asset_class == AssetClass.EQUITY:
        # Add trading hours flag
        if "is_trading_hours" not in df.columns:
            df["is_trading_hours"] = True  # Assume data is already filtered

        # No Fear & Greed for stocks (use VIX or similar)
        if "fear_greed_value" not in df.columns:
            df["fear_greed_value"] = np.nan

    # Remove duplicates and sort
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    # Standardize column order
    base_cols = [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    base_cols = [c for c in base_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in base_cols]
    df = df[base_cols + other_cols]

    return df


def load_from_adapter(
    vendor: Union[str, DataVendor],
    symbols: Sequence[str],
    timeframe: str = "4h",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    adjust_corporate_actions: bool = True,
    add_corp_features: bool = False,
    add_stock_features: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    """
    Load data from market data adapter.

    Args:
        vendor: Data vendor (alpaca, polygon, binance)
        symbols: List of symbols to load
        timeframe: Bar timeframe
        start_date: Start date (ISO format or YYYY-MM-DD)
        end_date: End date
        config: Adapter configuration
        adjust_corporate_actions: Apply split/dividend adjustments (equity only)
        add_corp_features: Add corporate action features (equity only)
        add_stock_features: Add VIX, sector momentum, relative strength features (equity only)

    Returns:
        (all_dfs_dict, all_obs_dict) tuple
    """
    if isinstance(vendor, str):
        vendor = DataVendor(vendor.lower())

    config = config or {}

    # Determine asset class from vendor
    asset_class = AssetClass.CRYPTO
    if vendor in (DataVendor.ALPACA, DataVendor.POLYGON):
        asset_class = AssetClass.EQUITY
    elif vendor in (DataVendor.OANDA, DataVendor.IG, DataVendor.DUKASCOPY):
        asset_class = AssetClass.FOREX

    # Parse dates
    start_ts = None
    end_ts = None
    if start_date:
        start_dt = pd.Timestamp(start_date, tz="UTC")
        start_ts = int(start_dt.timestamp() * 1000)
    if end_date:
        end_dt = pd.Timestamp(end_date, tz="UTC")
        end_ts = int(end_dt.timestamp() * 1000)

    # Create adapter and fetch data
    all_dfs: Dict[str, pd.DataFrame] = {}
    all_obs: Dict[str, np.ndarray] = {}

    # Load benchmark data for equities (VIX, SPY, QQQ) - done once before the loop
    vix_df: Optional[pd.DataFrame] = None
    spy_df: Optional[pd.DataFrame] = None
    qqq_df: Optional[pd.DataFrame] = None

    if asset_class == AssetClass.EQUITY and add_stock_features:
        vix_df = _load_benchmark_data("VIX", None, timeframe)
        spy_df = _load_benchmark_data("SPY", None, timeframe)
        qqq_df = _load_benchmark_data("QQQ", None, timeframe)

    try:
        from adapters.registry import create_market_data_adapter

        adapter = create_market_data_adapter(vendor.value, config)

        for symbol in symbols:
            logger.info(f"Loading {symbol} from {vendor.value}")

            bars = adapter.get_bars(
                symbol,
                timeframe,
                limit=50000,
                start_ts=start_ts,
                end_ts=end_ts,
            )

            if not bars:
                logger.warning(f"No data returned for {symbol}")
                continue

            # Convert bars to DataFrame
            rows = []
            for bar in bars:
                rows.append(
                    {
                        "timestamp": bar.ts // 1000 if bar.ts > 10_000_000_000 else bar.ts,
                        "symbol": symbol.upper(),
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume_base),
                        "quote_asset_volume": float(bar.volume_quote or 0),
                        "number_of_trades": bar.trades or 0,
                        "taker_buy_base_asset_volume": 0.0,
                        "taker_buy_quote_asset_volume": 0.0,
                        "vwap": float(bar.vwap) if bar.vwap else None,
                    }
                )

            df = pd.DataFrame(rows)
            df = _standardize_columns(df, asset_class, timeframe)

            # Apply corporate actions adjustments for equities
            if asset_class == AssetClass.EQUITY:
                if adjust_corporate_actions:
                    df = apply_split_adjustment(df, symbol.upper())
                if add_corp_features:
                    df = add_corporate_action_features(df, symbol.upper())
                # Add VIX, sector momentum, relative strength features
                if add_stock_features:
                    df = _add_stock_features(
                        df, symbol.upper(), vix_df=vix_df, spy_df=spy_df, qqq_df=qqq_df
                    )

            # Apply forex-specific processing (Phase 9)
            if asset_class == AssetClass.FOREX:
                df = _add_forex_features(df, symbol.upper())

            all_dfs[symbol.upper()] = df

    except ImportError as e:
        logger.error(f"Adapter not available: {e}")
        raise

    return all_dfs, all_obs


# =============================================================================
# BENCHMARK DATA LOADING (EQUITY)
# =============================================================================

# Default paths for benchmark data files
DEFAULT_BENCHMARK_PATHS = {
    "VIX": [
        "data/raw_stocks/VIX.parquet",
        "data/stocks/VIX.parquet",
        "data/raw_stocks/^VIX.parquet",
        "data/stocks/^VIX.parquet",
    ],
    "SPY": [
        "data/raw_stocks/SPY.parquet",
        "data/stocks/SPY.parquet",
    ],
    "QQQ": [
        "data/raw_stocks/QQQ.parquet",
        "data/stocks/QQQ.parquet",
    ],
}


def _load_benchmark_data(
    symbol: str,
    explicit_path: Optional[str] = None,
    timeframe: str = "4h",
) -> Optional[pd.DataFrame]:
    """
    Load benchmark data (VIX, SPY, QQQ) for equity feature enrichment.

    This function tries to load benchmark data from:
    1. Explicit path if provided
    2. Default paths for the symbol

    Args:
        symbol: Benchmark symbol (VIX, SPY, QQQ)
        explicit_path: Explicit path to data file (optional)
        timeframe: Timeframe for alignment

    Returns:
        DataFrame with benchmark data or None if not found
    """
    paths_to_try = []

    if explicit_path:
        paths_to_try.append(explicit_path)

    # Add default paths
    default_paths = DEFAULT_BENCHMARK_PATHS.get(symbol.upper(), [])
    paths_to_try.extend(default_paths)

    for path in paths_to_try:
        if os.path.exists(path):
            try:
                df = load_from_file(path, AssetClass.EQUITY, timeframe)
                if df is not None and not df.empty:
                    logger.debug(f"Loaded benchmark {symbol} from {path}: {len(df)} rows")
                    return df
            except Exception as e:
                logger.debug(f"Failed to load {symbol} from {path}: {e}")
                continue

    # Try to load via Yahoo adapter as fallback
    try:
        from adapters.yahoo.market_data import YahooMarketDataAdapter

        # Map symbol to Yahoo format
        yahoo_symbol = symbol.upper()
        if symbol.upper() == "VIX":
            yahoo_symbol = "^VIX"

        adapter = YahooMarketDataAdapter()
        bars = adapter.get_bars(yahoo_symbol, timeframe="1d", limit=365)

        if bars:
            rows = []
            for bar in bars:
                rows.append(
                    {
                        "timestamp": bar.ts // 1000 if bar.ts > 10_000_000_000 else bar.ts,
                        "symbol": symbol.upper(),
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume_base),
                    }
                )
            df = pd.DataFrame(rows)
            if not df.empty:
                logger.debug(f"Loaded benchmark {symbol} via Yahoo adapter: {len(df)} rows")
                return df

    except ImportError:
        logger.debug("Yahoo adapter not available for benchmark loading")
    except Exception as e:
        logger.debug(f"Failed to load {symbol} via Yahoo: {e}")

    logger.warning(f"Could not load benchmark data for {symbol}")
    return None


def _add_stock_features(
    df: pd.DataFrame,
    symbol: str,
    vix_df: Optional[pd.DataFrame] = None,
    spy_df: Optional[pd.DataFrame] = None,
    qqq_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Add stock-specific features (VIX, sector momentum, relative strength) to DataFrame.

    This function enriches equity DataFrames with:
    - VIX features (vix_normalized, vix_regime)
    - Market regime indicator
    - Relative strength vs SPY and QQQ
    - Sector momentum

    Args:
        df: DataFrame with stock data
        symbol: Stock symbol
        vix_df: VIX benchmark data (optional)
        spy_df: SPY benchmark data (optional)
        qqq_df: QQQ benchmark data (optional)

    Returns:
        DataFrame with stock features added
    """
    try:
        from services.sector_momentum import enrich_dataframe_with_all_stock_features

        df = enrich_dataframe_with_all_stock_features(
            df=df,
            symbol=symbol,
            spy_df=spy_df,
            qqq_df=qqq_df,
            vix_df=vix_df,
        )
        logger.debug(f"Added stock features to {symbol}")

    except ImportError as e:
        logger.warning(f"Could not import sector_momentum service: {e}")
        # Fallback: add features directly from stock_features module
        try:
            from stock_features import add_stock_features_to_dataframe

            df = add_stock_features_to_dataframe(
                df=df,
                symbol=symbol,
                spy_df=spy_df,
                qqq_df=qqq_df,
                vix_df=vix_df,
            )
            logger.debug(f"Added stock features to {symbol} (via fallback)")
        except ImportError:
            logger.warning("stock_features module not available, skipping stock features")
    except Exception as e:
        logger.warning(f"Error adding stock features to {symbol}: {e}")

    return df


def _add_forex_features(
    df: pd.DataFrame,
    pair: str,
) -> pd.DataFrame:
    """
    Add forex-specific features to DataFrame.

    This function enriches forex DataFrames with:
    - Session indicators (Sydney/Tokyo/London/NY)
    - Session liquidity factors
    - Session overlap flags

    For more comprehensive forex features (swap rates, interest rate differentials,
    economic calendar proximity), use data_loader_forex.py directly.

    Args:
        df: DataFrame with forex data
        pair: Currency pair (e.g., "EUR_USD")

    Returns:
        DataFrame with forex features added
    """
    try:
        from data_loader_forex import _add_session_features

        df = _add_session_features(df)
        logger.debug(f"Added forex session features to {pair}")

    except ImportError as e:
        logger.warning(f"Could not import forex session features: {e}")
        # Fallback: add basic session indicators
        if "timestamp" in df.columns and len(df) > 0:
            from datetime import datetime, timezone

            def get_session(ts: int) -> str:
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                hour = dt.hour
                if 12 <= hour < 16:
                    return "london_ny_overlap"
                if 7 <= hour < 9:
                    return "tokyo_london_overlap"
                if 7 <= hour < 16:
                    return "london"
                if 12 <= hour < 21:
                    return "new_york"
                if 0 <= hour < 9:
                    return "tokyo"
                if hour >= 21 or hour < 6:
                    return "sydney"
                return "low_liquidity"

            df = df.copy()
            df["session"] = df["timestamp"].apply(get_session)
            df["session_liquidity"] = (
                df["session"]
                .map(
                    {
                        "london_ny_overlap": 1.5,
                        "tokyo_london_overlap": 1.0,
                        "london": 1.3,
                        "new_york": 1.2,
                        "tokyo": 0.8,
                        "sydney": 0.6,
                        "low_liquidity": 0.4,
                    }
                )
                .fillna(0.5)
            )
            df["is_session_overlap"] = df["session"].str.contains("overlap", na=False)

    except Exception as e:
        logger.warning(f"Error adding forex features to {pair}: {e}")

    return df


def load_multi_asset_data(
    paths: Sequence[Union[str, Path]],
    asset_class: AssetClass = AssetClass.CRYPTO,
    timeframe: str = "4h",
    merge_fear_greed: bool = True,
    synthetic_fraction: float = 0.0,
    seed: int = 42,
    adjust_corporate_actions: bool = True,
    add_corp_features: bool = False,
    add_stock_features: bool = True,
    vix_path: Optional[str] = None,
    spy_path: Optional[str] = None,
    qqq_path: Optional[str] = None,
    # Forex-specific options (Phase 9)
    filter_weekends: bool = True,
    add_session_features: bool = True,
    merge_swap_rates: bool = False,
    merge_interest_rates: bool = False,
    merge_calendar: bool = False,
    swap_dir: Optional[str] = None,
    rate_dir: Optional[str] = None,
    calendar_dir: Optional[str] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    """
    Load data from multiple files with multi-asset support.

    This is the main entry point for loading training data,
    compatible with fetch_all_data_patch.load_all_data().

    Args:
        paths: List of file paths (feather, parquet, csv)
        asset_class: Asset class (crypto/equity/forex)
        timeframe: Bar timeframe
        merge_fear_greed: Merge Fear & Greed data (crypto only)
        synthetic_fraction: Fraction of synthetic data (unused)
        seed: Random seed (unused)
        adjust_corporate_actions: Apply split/dividend adjustments (equity only)
        add_corp_features: Add corporate action features like gap_pct, days_to_earnings (equity only)
        add_stock_features: Add VIX, sector momentum, relative strength features (equity only)
        vix_path: Path to VIX data file (optional, will try default locations)
        spy_path: Path to SPY data file (optional, will try default locations)
        qqq_path: Path to QQQ data file (optional, will try default locations)
        # Forex-specific options (Phase 9)
        filter_weekends: Filter out forex weekend gaps (forex only)
        add_session_features: Add session indicators Sydney/Tokyo/London/NY (forex only)
        merge_swap_rates: Merge swap/rollover rate data (forex only)
        merge_interest_rates: Merge interest rate differential data (forex only)
        merge_calendar: Merge economic calendar proximity (forex only)
        swap_dir: Directory with swap rate files (forex only)
        rate_dir: Directory with interest rate files (forex only)
        calendar_dir: Directory with calendar files (forex only)

    Returns:
        (all_dfs_dict, all_obs_dict) tuple
    """
    all_dfs: Dict[str, pd.DataFrame] = {}
    all_obs: Dict[str, np.ndarray] = {}

    timeframe_seconds = timeframe_to_seconds(timeframe)

    # Load Fear & Greed for crypto
    fng = pd.DataFrame()
    if asset_class == AssetClass.CRYPTO and merge_fear_greed:
        fng = load_fear_greed(timeframe_seconds)

    # Load benchmark data for equities (VIX, SPY, QQQ)
    vix_df: Optional[pd.DataFrame] = None
    spy_df: Optional[pd.DataFrame] = None
    qqq_df: Optional[pd.DataFrame] = None

    if asset_class == AssetClass.EQUITY and add_stock_features:
        vix_df = _load_benchmark_data("VIX", vix_path, timeframe)
        spy_df = _load_benchmark_data("SPY", spy_path, timeframe)
        qqq_df = _load_benchmark_data("QQQ", qqq_path, timeframe)

        if vix_df is not None:
            logger.info(f"Loaded VIX benchmark data: {len(vix_df)} rows")
        if spy_df is not None:
            logger.info(f"Loaded SPY benchmark data: {len(spy_df)} rows")
        if qqq_df is not None:
            logger.info(f"Loaded QQQ benchmark data: {len(qqq_df)} rows")

    # Forex: use specialized forex loader (Phase 9)
    if asset_class == AssetClass.FOREX:
        try:
            from data_loader_forex import load_forex_data as _load_forex_data

            forex_dfs, forex_obs = _load_forex_data(
                paths=list(paths),
                timeframe=timeframe,
                filter_weekends=filter_weekends,
                merge_swaps=merge_swap_rates,
                merge_rates=merge_interest_rates,
                merge_calendar=merge_calendar,
                swap_dir=swap_dir,
                rate_dir=rate_dir,
                calendar_dir=calendar_dir,
                add_session_features=add_session_features,
            )

            logger.info(f"Loaded {len(forex_dfs)} forex pairs via specialized loader")
            return forex_dfs, {k: np.array(v) for k, v in forex_obs.items()}

        except ImportError as e:
            logger.warning(f"Forex loader not available, falling back to generic: {e}")
        except Exception as e:
            logger.warning(f"Forex loader failed, falling back to generic: {e}")

    for path in paths:
        try:
            df = load_from_file(path, asset_class, timeframe)
            symbol = df["symbol"].iloc[0] if "symbol" in df.columns else Path(path).stem

            # Merge Fear & Greed for crypto
            if not fng.empty and asset_class == AssetClass.CRYPTO:
                df = _merge_fear_greed(df, fng)

            # Apply corporate actions adjustments for equities
            if asset_class == AssetClass.EQUITY:
                if adjust_corporate_actions:
                    df = apply_split_adjustment(df, symbol)
                if add_corp_features:
                    df = add_corporate_action_features(df, symbol)
                # Add VIX, sector momentum, relative strength features
                if add_stock_features:
                    df = _add_stock_features(
                        df, symbol, vix_df=vix_df, spy_df=spy_df, qqq_df=qqq_df
                    )

            all_dfs[symbol] = df
            logger.debug(f"Loaded {symbol}: {len(df)} rows")

        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            continue

    if not all_dfs:
        raise ValueError("No data files were successfully loaded")

    logger.info(f"Loaded {len(all_dfs)} symbols: {list(all_dfs.keys())}")
    return all_dfs, all_obs


def _merge_fear_greed(df: pd.DataFrame, fng: pd.DataFrame) -> pd.DataFrame:
    """Merge Fear & Greed data into DataFrame."""
    if fng.empty:
        return df

    # Preserve original if exists
    orig_col = None
    if "fear_greed_value" in df.columns:
        orig_col = "fear_greed_value_orig"
        df = df.rename(columns={"fear_greed_value": orig_col})

    # Merge with backward fill
    fng_sorted = fng.sort_values("timestamp")[["timestamp", "fear_greed_value"]].copy()
    df = pd.merge_asof(
        df.sort_values("timestamp"), fng_sorted, on="timestamp", direction="backward"
    )

    if "fear_greed_value" in df.columns:
        df["fear_greed_value"] = df["fear_greed_value"].ffill()

    # Restore original if merge failed
    if "fear_greed_value" not in df.columns and orig_col:
        df["fear_greed_value"] = df[orig_col]
    elif "fear_greed_value" in df.columns and orig_col:
        df["fear_greed_value"] = df["fear_greed_value"].fillna(df[orig_col])

    if orig_col and orig_col in df.columns:
        df = df.drop(columns=[orig_col])

    return df


# =============================================================================
# TRADING HOURS FILTERING (EQUITY)
# =============================================================================


def filter_trading_hours(
    df: pd.DataFrame,
    include_extended: bool = False,
    timezone_str: str = "America/New_York",
) -> pd.DataFrame:
    """
    Filter DataFrame to trading hours only (for equities).

    Args:
        df: DataFrame with timestamp column
        include_extended: Include pre-market (4:00-9:30) and after-hours (16:00-20:00)
        timezone_str: Market timezone

    Returns:
        Filtered DataFrame
    """
    try:
        import pytz

        tz = pytz.timezone(timezone_str)
    except ImportError:
        logger.warning("pytz not installed - cannot filter trading hours")
        return df

    df = df.copy()

    # Convert timestamp to datetime
    df["_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["_dt_local"] = df["_dt"].dt.tz_convert(tz)

    # Extract time components
    df["_hour"] = df["_dt_local"].dt.hour
    df["_minute"] = df["_dt_local"].dt.minute
    df["_weekday"] = df["_dt_local"].dt.weekday  # 0=Monday, 6=Sunday

    # Filter weekdays only
    mask = df["_weekday"] < 5  # Monday-Friday

    # Time of day filter
    time_minutes = df["_hour"] * 60 + df["_minute"]

    if include_extended:
        # 4:00 - 20:00 ET
        mask &= (time_minutes >= 240) & (time_minutes < 1200)
    else:
        # 9:30 - 16:00 ET
        mask &= (time_minutes >= 570) & (time_minutes < 960)

    # Clean up temp columns
    df = df.drop(columns=["_dt", "_dt_local", "_hour", "_minute", "_weekday"])

    return df[mask].copy()


# =============================================================================
# DATA VALIDATION
# =============================================================================


def validate_data(
    df: pd.DataFrame,
    asset_class: AssetClass = AssetClass.CRYPTO,
    min_rows: int = 100,
) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame for training.

    Args:
        df: DataFrame to validate
        asset_class: Asset class
        min_rows: Minimum required rows

    Returns:
        (is_valid, list of error messages)
    """
    errors: List[str] = []

    # Check row count
    if len(df) < min_rows:
        errors.append(f"Insufficient data: {len(df)} rows (min: {min_rows})")

    # Check required columns
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Check for NaN in critical columns
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                errors.append(f"Column '{col}' has {nan_count} NaN values")

    # Check OHLC validity
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        invalid_ohlc = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        )
        invalid_count = invalid_ohlc.sum()
        if invalid_count > 0:
            errors.append(f"{invalid_count} rows have invalid OHLC (high < low, etc.)")

    # Check timestamp ordering
    if "timestamp" in df.columns:
        if not df["timestamp"].is_monotonic_increasing:
            errors.append("Timestamps are not monotonically increasing")

    # Check for gaps in time series
    if "timestamp" in df.columns and len(df) > 1:
        diffs = df["timestamp"].diff().dropna()
        expected_diff = diffs.mode().iloc[0] if len(diffs) > 0 else None
        if expected_diff:
            gaps = (diffs > expected_diff * 2).sum()
            if gaps > len(df) * 0.1:  # More than 10% gaps
                errors.append(f"Time series has {gaps} gaps (>10% of data)")

    is_valid = len(errors) == 0
    return is_valid, errors


# =============================================================================
# CORPORATE ACTIONS INTEGRATION (Phase 7)
# =============================================================================


def apply_split_adjustment(
    df: pd.DataFrame,
    symbol: str,
    use_adjusted_close: bool = True,
) -> pd.DataFrame:
    """
    Apply stock split adjustments to OHLCV data.

    For equities, it's recommended to use split-adjusted prices
    to ensure price continuity across splits.

    Args:
        df: DataFrame with OHLCV data
        symbol: Stock symbol
        use_adjusted_close: If True, uses 'adjusted_close' column if present
                          (common in Yahoo/Polygon data)

    Returns:
        DataFrame with split-adjusted prices
    """
    df = df.copy()

    # Check if data already has adjusted prices
    if use_adjusted_close and "adjusted_close" in df.columns:
        # Compute adjustment factor from close vs adjusted_close
        adj_factor = df["adjusted_close"] / df["close"]
        adj_factor = adj_factor.fillna(1.0).replace([np.inf, -np.inf], 1.0)

        # Apply to OHLC
        df["open"] = df["open"] * adj_factor
        df["high"] = df["high"] * adj_factor
        df["low"] = df["low"] * adj_factor
        df["close"] = df["adjusted_close"]

        # Volume is inverse-adjusted
        df["volume"] = df["volume"] / adj_factor

        logger.debug(f"Applied adjustment from adjusted_close for {symbol}")
        return df

    # Use corporate actions service
    try:
        from services.corporate_actions import get_service

        service = get_service()
        df = service.adjust_prices(
            df,
            symbol,
            adjust_splits=True,
            adjust_dividends=False,
        )

        # Replace original columns with adjusted
        for col in ["open", "high", "low", "close"]:
            adj_col = f"{col}_adjusted"
            if adj_col in df.columns:
                df[col] = df[adj_col]
                df.drop(columns=[adj_col], inplace=True)

        if "volume_adjusted" in df.columns:
            df["volume"] = df["volume_adjusted"]
            df.drop(columns=["volume_adjusted"], inplace=True)

        logger.debug(f"Applied split adjustment via service for {symbol}")

    except ImportError:
        logger.warning("corporate_actions service not available, skipping adjustment")
    except Exception as e:
        logger.warning(f"Error applying split adjustment for {symbol}: {e}")

    return df


def add_corporate_action_features(
    df: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """
    Add corporate action-related features to DataFrame.

    Features added:
    - days_to_ex_div: Days until next ex-dividend date
    - days_to_earnings: Days until next earnings
    - gap_pct: Pre-market gap percentage
    - gap_direction: Direction of gap (1/-1/0)
    - gap_magnitude: Categorized gap size

    Args:
        df: DataFrame with OHLCV data
        symbol: Stock symbol

    Returns:
        DataFrame with additional features
    """
    df = df.copy()

    try:
        from services.corporate_actions import get_service

        service = get_service()

        # Add gap features (doesn't require external data)
        df = service.compute_gap_features(df)

        # Get date column
        if "timestamp" in df.columns:
            dates = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%d")
        elif "date" in df.columns:
            dates = pd.Series(df["date"]).astype(str)
        else:
            logger.warning("No date column found, skipping calendar features")
            return df

        # Add earnings and dividend features row by row (can be slow for large datasets)
        # For performance, consider batch fetching
        days_to_ex_div = []
        days_to_earnings = []

        # Get all corporate actions upfront
        all_divs = service.get_dividends(symbol)
        all_earnings = service.get_earnings(symbol)

        div_dates = sorted([d["ex_date"] for d in all_divs if d.get("ex_date")])
        earn_dates = sorted([e["report_date"] for e in all_earnings if e.get("report_date")])

        for date in dates:
            # Days to next dividend
            future_divs = [d for d in div_dates if d > date]
            if future_divs:
                next_div = future_divs[0]
                days = (datetime.fromisoformat(next_div) - datetime.fromisoformat(date)).days
                days_to_ex_div.append(min(days, 90))  # Cap at 90
            else:
                days_to_ex_div.append(90)

            # Days to next earnings
            future_earns = [e for e in earn_dates if e > date]
            if future_earns:
                next_earn = future_earns[0]
                days = (datetime.fromisoformat(next_earn) - datetime.fromisoformat(date)).days
                days_to_earnings.append(min(days, 90))
            else:
                days_to_earnings.append(90)

        df["days_to_ex_div"] = days_to_ex_div
        df["days_to_earnings"] = days_to_earnings

        logger.debug(f"Added corporate action features for {symbol}")

    except ImportError:
        logger.warning("corporate_actions service not available")
    except Exception as e:
        logger.warning(f"Error adding corporate action features for {symbol}: {e}")

    return df


def load_equity_data_adjusted(
    paths: Sequence[Union[str, Path]],
    timeframe: str = "4h",
    apply_adjustments: bool = True,
    add_corp_features: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    """
    Load equity data with split adjustments and corporate action features.

    This is the recommended function for loading stock data for backtesting.

    Args:
        paths: List of data file paths
        timeframe: Bar timeframe
        apply_adjustments: Apply split/dividend adjustments
        add_corp_features: Add corporate action features (gap, earnings, etc.)

    Returns:
        (all_dfs_dict, all_obs_dict) tuple
    """
    # Load base data
    all_dfs, all_obs = load_multi_asset_data(
        paths=paths,
        asset_class=AssetClass.EQUITY,
        timeframe=timeframe,
        merge_fear_greed=False,
    )

    # Apply adjustments and features
    for symbol, df in all_dfs.items():
        if apply_adjustments:
            df = apply_split_adjustment(df, symbol)

        if add_corp_features:
            df = add_corporate_action_features(df, symbol)

        all_dfs[symbol] = df

    return all_dfs, all_obs


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def load_crypto_data(
    paths: Sequence[Union[str, Path]],
    timeframe: str = "4h",
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    """Load crypto data from files."""
    return load_multi_asset_data(
        paths=paths,
        asset_class=AssetClass.CRYPTO,
        timeframe=timeframe,
        merge_fear_greed=True,
    )


def load_stock_data(
    paths: Sequence[Union[str, Path]],
    timeframe: str = "4h",
    adjust_corporate_actions: bool = True,
    add_corp_features: bool = False,
    add_stock_features: bool = True,
    vix_path: Optional[str] = None,
    spy_path: Optional[str] = None,
    qqq_path: Optional[str] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    """
    Load stock data from files.

    Args:
        paths: List of file paths
        timeframe: Bar timeframe
        adjust_corporate_actions: Apply split/dividend adjustments
        add_corp_features: Add gap, earnings, dividend features
        add_stock_features: Add VIX, sector momentum, relative strength features
        vix_path: Path to VIX data file (optional)
        spy_path: Path to SPY data file (optional)
        qqq_path: Path to QQQ data file (optional)

    Returns:
        (all_dfs_dict, all_obs_dict) tuple
    """
    return load_multi_asset_data(
        paths=paths,
        asset_class=AssetClass.EQUITY,
        timeframe=timeframe,
        merge_fear_greed=False,
        adjust_corporate_actions=adjust_corporate_actions,
        add_corp_features=add_corp_features,
        add_stock_features=add_stock_features,
        vix_path=vix_path,
        spy_path=spy_path,
        qqq_path=qqq_path,
    )


def load_alpaca_data(
    symbols: Sequence[str],
    timeframe: str = "4h",
    start_date: str = "2023-01-01",
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    adjust_corporate_actions: bool = True,
    add_corp_features: bool = False,
    add_stock_features: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    """
    Load stock data from Alpaca API.

    Args:
        symbols: List of stock symbols
        timeframe: Bar timeframe
        start_date: Start date
        end_date: End date (default: today)
        api_key: Alpaca API key (or env ALPACA_API_KEY)
        api_secret: Alpaca API secret
        adjust_corporate_actions: Apply split/dividend adjustments
        add_corp_features: Add gap, earnings, dividend features
        add_stock_features: Add VIX, sector momentum, relative strength features

    Returns:
        (all_dfs_dict, all_obs_dict) tuple
    """
    config = {}
    if api_key:
        config["api_key"] = api_key
    if api_secret:
        config["api_secret"] = api_secret

    return load_from_adapter(
        vendor=DataVendor.ALPACA,
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        config=config,
        adjust_corporate_actions=adjust_corporate_actions,
        add_corp_features=add_corp_features,
        add_stock_features=add_stock_features,
    )


def load_polygon_data(
    symbols: Sequence[str],
    timeframe: str = "4h",
    start_date: str = "2023-01-01",
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
    adjust_corporate_actions: bool = True,
    add_corp_features: bool = False,
    add_stock_features: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    """
    Load stock data from Polygon.io API.

    Args:
        symbols: List of stock symbols
        timeframe: Bar timeframe
        start_date: Start date
        end_date: End date (default: today)
        api_key: Polygon API key (or env POLYGON_API_KEY)
        adjust_corporate_actions: Apply split/dividend adjustments
        add_corp_features: Add gap, earnings, dividend features
        add_stock_features: Add VIX, sector momentum, relative strength features

    Returns:
        (all_dfs_dict, all_obs_dict) tuple
    """
    config = {}
    if api_key:
        config["api_key"] = api_key

    return load_from_adapter(
        vendor=DataVendor.POLYGON,
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        config=config,
        adjust_corporate_actions=adjust_corporate_actions,
        add_corp_features=add_corp_features,
        add_stock_features=add_stock_features,
    )
