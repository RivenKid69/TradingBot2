# -*- coding: utf-8 -*-
"""
services/earnings_calendar.py
Service for pre-loading and managing earnings calendar data.

Provides:
- Bulk earnings date pre-loading for multiple symbols
- Earnings calendar lookup by date range
- Integration with feature pipeline for earnings features
- Disk-based caching with TTL

Critical for ML Features:
- days_to_earnings: Volatility typically increases before earnings
- post_earnings_drift: PEAD anomaly
- earnings_surprise: For momentum/reversal strategies

Architecture:
    Yahoo Finance Adapter → Corporate Actions Service → Earnings Calendar → Features

Usage:
    from services.earnings_calendar import (
        preload_earnings,
        get_earnings_dates,
        add_earnings_to_df,
    )

    # Pre-load earnings for all symbols in universe
    preload_earnings(symbols=["AAPL", "MSFT", "GOOGL"])

    # Get earnings dates for a symbol
    dates = get_earnings_dates("AAPL", year=2024)

    # Add earnings features to DataFrame
    df = add_earnings_to_df(df, symbol="AAPL")

References:
    - Post-Earnings Announcement Drift (PEAD) - Ball & Brown (1968)
    - Earnings Momentum - Chan, Jegadeesh, Lakonishok (1996)
    - Implied Volatility Crush - options market behavior

Author: Sultan Suyunbayev
Date: 2025-11-28
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_CACHE_DIR = Path("data/cache/earnings")
DEFAULT_TTL_HOURS = 24

# Earnings blackout period (common corporate policy)
EARNINGS_BLACKOUT_DAYS = 14

# Post-earnings drift period
POST_EARNINGS_DRIFT_DAYS = 3


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EarningsCalendarConfig:
    """Configuration for earnings calendar service."""

    cache_dir: Path = DEFAULT_CACHE_DIR
    cache_ttl_hours: int = DEFAULT_TTL_HOURS
    blackout_days: int = EARNINGS_BLACKOUT_DAYS
    drift_days: int = POST_EARNINGS_DRIFT_DAYS
    vendor: str = "yahoo"  # Data vendor


# =============================================================================
# Main Service
# =============================================================================


class EarningsCalendarService:
    """
    Service for managing earnings calendar data.

    Provides bulk loading, caching, and feature engineering
    for earnings dates and surprises.

    Attributes:
        config: Service configuration
        _cache: In-memory cache
    """

    def __init__(
        self,
        config: Optional[EarningsCalendarConfig] = None,
    ) -> None:
        """
        Initialize earnings calendar service.

        Args:
            config: Service configuration
        """
        self.config = config or EarningsCalendarConfig()

        # Ensure cache directory exists
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache: symbol -> list of earnings events
        self._earnings_cache: Dict[str, List[Dict]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

    # =========================================================================
    # Cache Management
    # =========================================================================

    def _get_cache_path(self, symbol: str) -> Path:
        """Get cache file path for a symbol."""
        return self.config.cache_dir / f"{symbol.upper()}_earnings.json"

    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cache is still valid."""
        # Check in-memory timestamp
        if symbol in self._cache_timestamps:
            age = datetime.now() - self._cache_timestamps[symbol]
            if age.total_seconds() < self.config.cache_ttl_hours * 3600:
                return True

        # Check file cache
        cache_path = self._get_cache_path(symbol)
        if cache_path.exists():
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            age = datetime.now() - mtime
            if age.total_seconds() < self.config.cache_ttl_hours * 3600:
                return True

        return False

    def _load_from_cache(self, symbol: str) -> Optional[List[Dict]]:
        """Load earnings from cache."""
        symbol = symbol.upper()

        # Try in-memory cache
        if symbol in self._earnings_cache:
            return self._earnings_cache[symbol]

        # Try file cache
        cache_path = self._get_cache_path(symbol)
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                self._earnings_cache[symbol] = data
                self._cache_timestamps[symbol] = datetime.now()
                return data
            except Exception as e:
                logger.warning(f"Error loading earnings cache for {symbol}: {e}")

        return None

    def _save_to_cache(self, symbol: str, earnings: List[Dict]) -> None:
        """Save earnings to cache."""
        symbol = symbol.upper()

        # Update in-memory cache
        self._earnings_cache[symbol] = earnings
        self._cache_timestamps[symbol] = datetime.now()

        # Save to file
        cache_path = self._get_cache_path(symbol)
        try:
            with open(cache_path, "w") as f:
                json.dump(earnings, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Error saving earnings cache for {symbol}: {e}")

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear cached data."""
        if symbol:
            symbol = symbol.upper()
            cache_path = self._get_cache_path(symbol)
            if cache_path.exists():
                cache_path.unlink()
            self._earnings_cache.pop(symbol, None)
            self._cache_timestamps.pop(symbol, None)
        else:
            for f in self.config.cache_dir.glob("*_earnings.json"):
                f.unlink()
            self._earnings_cache.clear()
            self._cache_timestamps.clear()

    # =========================================================================
    # Data Fetching
    # =========================================================================

    def _fetch_earnings(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict]:
        """
        Fetch earnings from corporate actions service.

        Args:
            symbol: Stock symbol
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            List of earnings event dicts
        """
        from services.corporate_actions import get_service

        service = get_service()
        return service.get_earnings(symbol, start_date, end_date)

    def get_earnings(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> List[Dict]:
        """
        Get earnings history for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date filter
            end_date: End date filter
            force_refresh: Bypass cache

        Returns:
            List of earnings events
        """
        symbol = symbol.upper()

        # Check cache
        if not force_refresh and self._is_cache_valid(symbol):
            cached = self._load_from_cache(symbol)
            if cached is not None:
                # Apply date filters
                result = cached
                if start_date:
                    result = [e for e in result if e.get("report_date", "") >= start_date]
                if end_date:
                    result = [e for e in result if e.get("report_date", "") <= end_date]
                return result

        # Fetch from service
        try:
            earnings = self._fetch_earnings(symbol, start_date, end_date)
            self._save_to_cache(symbol, earnings)
            return earnings
        except Exception as e:
            logger.error(f"Error fetching earnings for {symbol}: {e}")
            return []

    def get_earnings_dates(
        self,
        symbol: str,
        year: Optional[int] = None,
        force_refresh: bool = False,
    ) -> List[str]:
        """
        Get earnings dates for a symbol.

        Args:
            symbol: Stock symbol
            year: Filter by year
            force_refresh: Bypass cache

        Returns:
            List of earnings dates (ISO format)
        """
        earnings = self.get_earnings(symbol, force_refresh=force_refresh)

        dates = [e["report_date"] for e in earnings if "report_date" in e]

        if year:
            dates = [d for d in dates if d.startswith(str(year))]

        return sorted(dates)

    def get_next_earnings(
        self,
        symbol: str,
        as_of_date: str,
        force_refresh: bool = False,
    ) -> Optional[Dict]:
        """
        Get next earnings event after as_of_date.

        Args:
            symbol: Stock symbol
            as_of_date: Reference date (ISO format)
            force_refresh: Bypass cache

        Returns:
            Next earnings event dict or None
        """
        earnings = self.get_earnings(symbol, force_refresh=force_refresh)

        future = [e for e in earnings if e.get("report_date", "") > as_of_date]

        if not future:
            return None

        # Return earliest future earnings
        return min(future, key=lambda e: e["report_date"])

    def get_last_earnings(
        self,
        symbol: str,
        as_of_date: str,
        force_refresh: bool = False,
    ) -> Optional[Dict]:
        """
        Get most recent earnings event before as_of_date.

        Args:
            symbol: Stock symbol
            as_of_date: Reference date (ISO format)
            force_refresh: Bypass cache

        Returns:
            Last earnings event dict or None
        """
        earnings = self.get_earnings(symbol, force_refresh=force_refresh)

        past = [e for e in earnings if e.get("report_date", "") <= as_of_date]

        if not past:
            return None

        # Return most recent past earnings
        return max(past, key=lambda e: e["report_date"])

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    def preload_earnings(
        self,
        symbols: Sequence[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Dict[str, int]:
        """
        Pre-load earnings data for multiple symbols.

        Use this to warm up cache before batch processing.

        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            force_refresh: Bypass existing cache

        Returns:
            Dict mapping symbol to number of earnings loaded
        """
        logger.info(f"Preloading earnings for {len(symbols)} symbols...")

        result = {}
        for symbol in symbols:
            try:
                earnings = self.get_earnings(
                    symbol,
                    start_date=start_date,
                    end_date=end_date,
                    force_refresh=force_refresh,
                )
                result[symbol.upper()] = len(earnings)
            except Exception as e:
                logger.warning(f"Error preloading {symbol}: {e}")
                result[symbol.upper()] = 0

        loaded = sum(1 for v in result.values() if v > 0)
        logger.info(f"Preloaded earnings: {loaded}/{len(symbols)} symbols successful")
        return result

    def get_upcoming_earnings(
        self,
        symbols: Sequence[str],
        days_ahead: int = 30,
    ) -> Dict[str, List[Dict]]:
        """
        Get upcoming earnings for multiple symbols.

        Args:
            symbols: List of symbols
            days_ahead: Days to look ahead

        Returns:
            Dict mapping symbol to list of upcoming earnings
        """
        today = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        result = {}
        for symbol in symbols:
            earnings = self.get_earnings(symbol, start_date=today, end_date=end_date)
            # Filter to only upcoming (no actuals yet)
            upcoming = [e for e in earnings if not e.get("eps_actual")]
            if upcoming:
                result[symbol.upper()] = upcoming

        return result

    # =========================================================================
    # Feature Engineering
    # =========================================================================

    def compute_earnings_features(
        self,
        symbol: str,
        as_of_date: str,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute earnings-related features for a given date.

        Features:
        - days_until_earnings: Days until next earnings (0-90)
        - days_since_earnings: Days since last earnings
        - last_earnings_surprise: EPS surprise percentage
        - in_earnings_blackout: 1 if within blackout period
        - post_earnings: 1 if within drift period after earnings

        Args:
            symbol: Stock symbol
            as_of_date: Reference date

        Returns:
            Dict of feature values
        """
        features = {
            "days_until_earnings": 90.0,  # Default: far from earnings
            "days_since_earnings": None,
            "last_earnings_surprise": 0.0,
            "in_earnings_blackout": 0,
            "post_earnings": 0,
        }

        try:
            as_of_dt = datetime.fromisoformat(as_of_date)

            # Get next earnings
            next_earnings = self.get_next_earnings(symbol, as_of_date, force_refresh)
            if next_earnings:
                next_dt = datetime.fromisoformat(next_earnings["report_date"])
                days_until = (next_dt - as_of_dt).days
                features["days_until_earnings"] = min(float(days_until), 90.0)

                # In blackout if within blackout_days before earnings
                if days_until <= self.config.blackout_days:
                    features["in_earnings_blackout"] = 1

            # Get last earnings
            last_earnings = self.get_last_earnings(symbol, as_of_date, force_refresh)
            if last_earnings:
                last_dt = datetime.fromisoformat(last_earnings["report_date"])
                days_since = (as_of_dt - last_dt).days
                features["days_since_earnings"] = float(days_since)

                # Get surprise
                surprise = last_earnings.get("surprise_pct")
                if surprise is not None:
                    features["last_earnings_surprise"] = float(surprise)

                # Post earnings drift period
                if days_since <= self.config.drift_days:
                    features["post_earnings"] = 1

        except Exception as e:
            logger.warning(f"Error computing earnings features for {symbol}: {e}")

        return features

    def add_earnings_to_df(
        self,
        df: pd.DataFrame,
        symbol: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Add earnings features to a DataFrame.

        Features added:
        - days_until_earnings
        - days_since_earnings
        - last_earnings_surprise
        - in_earnings_blackout
        - post_earnings

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            force_refresh: Bypass cache

        Returns:
            DataFrame with earnings features
        """
        df = df.copy()

        # Initialize columns
        df["days_until_earnings"] = 90.0
        df["days_since_earnings"] = np.nan
        df["last_earnings_surprise"] = 0.0
        df["in_earnings_blackout"] = 0
        df["post_earnings"] = 0

        # Get date column
        if "timestamp" in df.columns:
            df["_date"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%d")
        elif "date" in df.columns:
            df["_date"] = df["date"].astype(str)
        else:
            return df

        # Compute features for each row
        # Note: This is slow for large DataFrames - consider vectorization for production
        for i, row in df.iterrows():
            try:
                features = self.compute_earnings_features(
                    symbol,
                    row["_date"],
                    force_refresh=force_refresh if i == 0 else False,
                )
                df.at[i, "days_until_earnings"] = features["days_until_earnings"]
                if features["days_since_earnings"] is not None:
                    df.at[i, "days_since_earnings"] = features["days_since_earnings"]
                df.at[i, "last_earnings_surprise"] = features["last_earnings_surprise"]
                df.at[i, "in_earnings_blackout"] = features["in_earnings_blackout"]
                df.at[i, "post_earnings"] = features["post_earnings"]
            except Exception:
                continue

        df.drop(columns=["_date"], inplace=True)
        return df

    def add_earnings_to_df_vectorized(
        self,
        df: pd.DataFrame,
        symbol: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Add earnings features to DataFrame using vectorized operations.

        This is much faster than row-by-row computation for large DataFrames.

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            force_refresh: Bypass cache

        Returns:
            DataFrame with earnings features
        """
        df = df.copy()

        # Get date column
        if "timestamp" in df.columns:
            dates = pd.to_datetime(df["timestamp"], unit="s")
        elif "date" in df.columns:
            dates = pd.to_datetime(df["date"])
        else:
            return df

        # Get all earnings dates
        earnings = self.get_earnings(symbol, force_refresh=force_refresh)
        if not earnings:
            df["days_until_earnings"] = 90.0
            df["days_since_earnings"] = np.nan
            df["last_earnings_surprise"] = 0.0
            df["in_earnings_blackout"] = 0
            df["post_earnings"] = 0
            return df

        # Build earnings date lookup
        earnings_dates = sorted(
            [datetime.fromisoformat(e["report_date"]) for e in earnings if "report_date" in e]
        )
        surprise_by_date = {
            e["report_date"]: e.get("surprise_pct", 0.0) for e in earnings if "report_date" in e
        }

        # Initialize columns
        n = len(df)
        days_until = np.full(n, 90.0)
        days_since = np.full(n, np.nan)
        surprise = np.zeros(n)
        blackout = np.zeros(n, dtype=int)
        post = np.zeros(n, dtype=int)

        # Vectorized computation using searchsorted
        dates_arr = dates.values.astype("datetime64[D]")
        earnings_arr = np.array([d.date() for d in earnings_dates], dtype="datetime64[D]")

        for i, date in enumerate(dates.dt.date):
            date_np = np.datetime64(date)

            # Find next earnings (first date > current)
            next_idx = np.searchsorted(earnings_arr, date_np, side="right")
            if next_idx < len(earnings_arr):
                next_date = pd.Timestamp(earnings_arr[next_idx])
                delta = (next_date.date() - date).days
                days_until[i] = min(delta, 90.0)
                if delta <= self.config.blackout_days:
                    blackout[i] = 1

            # Find last earnings (last date <= current)
            if next_idx > 0:
                prev_idx = next_idx - 1
                prev_date = pd.Timestamp(earnings_arr[prev_idx])
                delta = (date - prev_date.date()).days
                days_since[i] = delta

                # Get surprise
                prev_date_str = prev_date.strftime("%Y-%m-%d")
                if prev_date_str in surprise_by_date:
                    s = surprise_by_date[prev_date_str]
                    if s is not None:
                        surprise[i] = float(s)

                # Post earnings drift
                if delta <= self.config.drift_days:
                    post[i] = 1

        df["days_until_earnings"] = days_until
        df["days_since_earnings"] = days_since
        df["last_earnings_surprise"] = surprise
        df["in_earnings_blackout"] = blackout
        df["post_earnings"] = post

        return df


# =============================================================================
# Module-level convenience functions
# =============================================================================

_default_service: Optional[EarningsCalendarService] = None


def get_service() -> EarningsCalendarService:
    """Get default EarningsCalendarService instance."""
    global _default_service
    if _default_service is None:
        _default_service = EarningsCalendarService()
    return _default_service


def preload_earnings(
    symbols: Sequence[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force_refresh: bool = False,
) -> Dict[str, int]:
    """Convenience function to preload earnings for multiple symbols."""
    return get_service().preload_earnings(symbols, start_date, end_date, force_refresh)


def get_earnings_dates(
    symbol: str,
    year: Optional[int] = None,
    force_refresh: bool = False,
) -> List[str]:
    """Convenience function to get earnings dates for a symbol."""
    return get_service().get_earnings_dates(symbol, year, force_refresh)


def get_upcoming_earnings(
    symbols: Sequence[str],
    days_ahead: int = 30,
) -> Dict[str, List[Dict]]:
    """Convenience function to get upcoming earnings for multiple symbols."""
    return get_service().get_upcoming_earnings(symbols, days_ahead)


def add_earnings_to_df(
    df: pd.DataFrame,
    symbol: str,
    force_refresh: bool = False,
    vectorized: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to add earnings features to DataFrame.

    Args:
        df: DataFrame with OHLCV data
        symbol: Stock symbol
        force_refresh: Bypass cache
        vectorized: Use vectorized computation (faster for large DataFrames)

    Returns:
        DataFrame with earnings features
    """
    service = get_service()
    if vectorized:
        return service.add_earnings_to_df_vectorized(df, symbol, force_refresh)
    return service.add_earnings_to_df(df, symbol, force_refresh)


def compute_earnings_features(
    symbol: str,
    as_of_date: str,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Convenience function to compute earnings features for a date."""
    return get_service().compute_earnings_features(symbol, as_of_date, force_refresh)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "EarningsCalendarService",
    "EarningsCalendarConfig",
    "get_service",
    "preload_earnings",
    "get_earnings_dates",
    "get_upcoming_earnings",
    "add_earnings_to_df",
    "compute_earnings_features",
]
