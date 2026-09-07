# -*- coding: utf-8 -*-
"""
services/macro_data.py
Service for downloading, caching, and providing macro economic indicators.

Provides:
- VIX (CBOE Volatility Index) - fear indicator
- DXY (US Dollar Index) - currency strength
- Treasury yields (10Y, 30Y) - rate environment
- Real yield proxy - inflation-adjusted rates

Critical for stock trading:
- VIX inversely correlated with equity returns
- DXY affects multinational earnings
- Treasury yields compete with equity risk premium

Architecture:
    Yahoo Finance Adapter → Macro Data Service → Features Pipeline → obs_builder

Usage:
    from services.macro_data import MacroDataService

    service = MacroDataService()

    # Get VIX history
    vix_df = service.get_vix(start_date="2024-01-01")

    # Merge all macro data into trading DataFrame
    df = service.add_macro_features(df, as_of_date="2024-01-15")

    # Pre-load for batch processing
    service.preload_all(start_date="2023-01-01", end_date="2024-12-31")

Best Practices:
    - Cache data to avoid rate limits (TTL configurable)
    - Handle missing data gracefully (forward fill)
    - Normalize indicators for ML input

References:
    - CBOE VIX Methodology
    - Bloomberg macro data handling
    - Fed FRED economic data standards

Author: Sultan Suyunbayev
Date: 2025-11-28
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Yahoo Finance symbols for macro indicators
MACRO_SYMBOLS = {
    "vix": "^VIX",  # CBOE Volatility Index
    "vix_3m": "^VIX3M",  # CBOE 3-Month Volatility
    "dxy": "DX-Y.NYB",  # US Dollar Index
    "treasury_10y": "^TNX",  # 10-Year Treasury Yield
    "treasury_30y": "^TYX",  # 30-Year Treasury Yield
    "treasury_5y": "^FVX",  # 5-Year Treasury Yield
    "treasury_3m": "^IRX",  # 3-Month Treasury Bill
    "gold": "GC=F",  # Gold Futures
    "oil": "CL=F",  # Crude Oil Futures
    "sp500": "^GSPC",  # S&P 500
}

# Default cache directory
DEFAULT_CACHE_DIR = Path("data/cache/macro")

# VIX regime thresholds (based on historical distribution)
VIX_REGIMES = {
    "low": 15.0,  # Below 15: complacency
    "normal_low": 20.0,  # 15-20: low normal
    "normal_high": 25.0,  # 20-25: high normal
    "elevated": 30.0,  # 25-30: elevated
    "high": 40.0,  # 30-40: high volatility
    # Above 40: extreme fear
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MacroDataConfig:
    """Configuration for MacroDataService."""
    cache_dir: Path = DEFAULT_CACHE_DIR
    cache_ttl_hours: int = 6  # Cache validity in hours (macro data is less volatile)
    default_lookback_days: int = 365 * 3  # Default 3 years of history
    vix_normalization_center: float = 20.0  # VIX center for normalization
    vix_normalization_scale: float = 10.0  # VIX scale for tanh normalization
    dxy_normalization_center: float = 100.0  # DXY center
    dxy_normalization_scale: float = 10.0  # DXY scale
    treasury_normalization_center: float = 3.5  # Treasury yield center
    treasury_normalization_scale: float = 2.0  # Treasury yield scale
    fill_missing_method: str = "ffill"  # Forward fill for missing data


# =============================================================================
# Main Service
# =============================================================================

class MacroDataService:
    """
    Service for managing macroeconomic indicator data.

    Provides caching, batch operations, and feature engineering
    for VIX, DXY, Treasury yields, and other macro indicators.

    Attributes:
        config: Service configuration
        _adapter: Yahoo Finance adapter
        _cache: In-memory cache
    """

    def __init__(
        self,
        config: Optional[MacroDataConfig] = None,
    ) -> None:
        """
        Initialize macro data service.

        Args:
            config: Service configuration
        """
        self.config = config or MacroDataConfig()

        # Ensure cache directory exists
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded adapter
        self._adapter = None

        # In-memory cache: symbol -> DataFrame
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

    @property
    def adapter(self):
        """Get or create the Yahoo market data adapter."""
        if self._adapter is None:
            from adapters.yahoo.market_data import YahooMarketDataAdapter
            self._adapter = YahooMarketDataAdapter()
        return self._adapter

    # =========================================================================
    # Cache Management
    # =========================================================================

    def _get_cache_path(self, indicator: str) -> Path:
        """Get cache file path for an indicator."""
        return self.config.cache_dir / f"{indicator}_history.parquet"

    def _is_cache_valid(self, indicator: str) -> bool:
        """Check if cache is still valid."""
        # Check in-memory timestamp
        if indicator in self._cache_timestamps:
            age = datetime.now() - self._cache_timestamps[indicator]
            if age.total_seconds() < self.config.cache_ttl_hours * 3600:
                return True

        # Check file cache
        cache_path = self._get_cache_path(indicator)
        if cache_path.exists():
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            age = datetime.now() - mtime
            if age.total_seconds() < self.config.cache_ttl_hours * 3600:
                return True

        return False

    def _load_from_cache(self, indicator: str) -> Optional[pd.DataFrame]:
        """Load data from cache."""
        # Try in-memory cache first
        if indicator in self._data_cache:
            return self._data_cache[indicator].copy()

        # Try file cache
        cache_path = self._get_cache_path(indicator)
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                self._data_cache[indicator] = df
                self._cache_timestamps[indicator] = datetime.now()
                return df.copy()
            except Exception as e:
                logger.warning(f"Error loading cache for {indicator}: {e}")

        return None

    def _save_to_cache(self, indicator: str, df: pd.DataFrame) -> None:
        """Save data to cache."""
        # Update in-memory cache
        self._data_cache[indicator] = df.copy()
        self._cache_timestamps[indicator] = datetime.now()

        # Save to file
        cache_path = self._get_cache_path(indicator)
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            logger.warning(f"Error saving cache for {indicator}: {e}")

    def clear_cache(self, indicator: Optional[str] = None) -> None:
        """
        Clear cached data.

        Args:
            indicator: Specific indicator to clear, or None for all
        """
        if indicator:
            cache_path = self._get_cache_path(indicator)
            if cache_path.exists():
                cache_path.unlink()
            self._data_cache.pop(indicator, None)
            self._cache_timestamps.pop(indicator, None)
        else:
            for f in self.config.cache_dir.glob("*.parquet"):
                f.unlink()
            self._data_cache.clear()
            self._cache_timestamps.clear()

    # =========================================================================
    # Data Fetching
    # =========================================================================

    def _fetch_indicator(
        self,
        indicator: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch indicator data with caching.

        Args:
            indicator: Indicator name (vix, dxy, treasury_10y, etc.)
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            force_refresh: Bypass cache

        Returns:
            DataFrame with columns: timestamp, symbol, value
        """
        # Validate indicator
        if indicator not in MACRO_SYMBOLS:
            raise ValueError(f"Unknown indicator: {indicator}. Valid: {list(MACRO_SYMBOLS.keys())}")

        symbol = MACRO_SYMBOLS[indicator]

        # Check cache
        if not force_refresh and self._is_cache_valid(indicator):
            cached = self._load_from_cache(indicator)
            if cached is not None:
                # Apply date filters
                if start_date:
                    cached = cached[cached["date"] >= start_date]
                if end_date:
                    cached = cached[cached["date"] <= end_date]
                return cached

        # Fetch from adapter
        logger.info(f"Fetching {indicator} ({symbol}) from Yahoo Finance...")

        # Calculate date range
        if not end_date:
            end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        else:
            end_ts = int(datetime.fromisoformat(end_date).timestamp() * 1000)

        if not start_date:
            # Default to 3 years of history
            start_ts = end_ts - (self.config.default_lookback_days * 24 * 3600 * 1000)
        else:
            start_ts = int(datetime.fromisoformat(start_date).timestamp() * 1000)

        try:
            bars = self.adapter.get_bars(
                symbol,
                timeframe="1d",
                start_ts=start_ts,
                end_ts=end_ts,
            )

            if not bars:
                logger.warning(f"No data returned for {indicator}")
                return pd.DataFrame(columns=["timestamp", "date", "symbol", "value"])

            # Convert to DataFrame
            data = []
            for bar in bars:
                data.append({
                    "timestamp": bar.ts // 1000,  # Convert to seconds
                    "date": datetime.fromtimestamp(bar.ts // 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                    "symbol": indicator,
                    "value": float(bar.close),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "volume": float(bar.volume_base) if bar.volume_base else 0,
                })

            df = pd.DataFrame(data)
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Cache result
            self._save_to_cache(indicator, df)

            logger.info(f"Fetched {len(df)} rows for {indicator}")
            return df

        except Exception as e:
            logger.error(f"Error fetching {indicator}: {e}")
            return pd.DataFrame(columns=["timestamp", "date", "symbol", "value"])

    def get_vix(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Get VIX (CBOE Volatility Index) data.

        The VIX measures implied volatility of S&P 500 options and is
        often called the "fear index".

        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            force_refresh: Bypass cache

        Returns:
            DataFrame with VIX values
        """
        return self._fetch_indicator("vix", start_date, end_date, force_refresh)

    def get_dxy(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Get DXY (US Dollar Index) data.

        The DXY measures the value of USD against a basket of
        major currencies (EUR, JPY, GBP, CAD, SEK, CHF).

        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            force_refresh: Bypass cache

        Returns:
            DataFrame with DXY values
        """
        return self._fetch_indicator("dxy", start_date, end_date, force_refresh)

    def get_treasury_yields(
        self,
        tenors: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Get Treasury yield data for multiple tenors.

        Available tenors: 3m, 5y, 10y, 30y

        Args:
            tenors: List of tenors (default: ["10y", "30y"])
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            force_refresh: Bypass cache

        Returns:
            DataFrame with yield values for each tenor
        """
        if tenors is None:
            tenors = ["10y", "30y"]

        tenor_map = {
            "3m": "treasury_3m",
            "5y": "treasury_5y",
            "10y": "treasury_10y",
            "30y": "treasury_30y",
        }

        dfs = []
        for tenor in tenors:
            indicator = tenor_map.get(tenor)
            if not indicator:
                logger.warning(f"Unknown tenor: {tenor}")
                continue

            df = self._fetch_indicator(indicator, start_date, end_date, force_refresh)
            if not df.empty:
                df = df[["date", "value"]].rename(columns={"value": f"treasury_{tenor}"})
                dfs.append(df)

        if not dfs:
            return pd.DataFrame(columns=["date"])

        # Merge all tenors
        result = dfs[0]
        for df in dfs[1:]:
            result = result.merge(df, on="date", how="outer")

        return result.sort_values("date").reset_index(drop=True)

    def get_all_macro(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Get all macro indicators merged into a single DataFrame.

        Includes: VIX, DXY, Treasury 10Y, Treasury 30Y

        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            force_refresh: Bypass cache

        Returns:
            DataFrame with all macro indicators
        """
        indicators = ["vix", "dxy", "treasury_10y", "treasury_30y"]

        dfs = []
        for indicator in indicators:
            df = self._fetch_indicator(indicator, start_date, end_date, force_refresh)
            if not df.empty:
                df = df[["date", "value"]].rename(columns={"value": indicator})
                dfs.append(df)

        if not dfs:
            return pd.DataFrame(columns=["date"])

        # Merge all indicators
        result = dfs[0]
        for df in dfs[1:]:
            result = result.merge(df, on="date", how="outer")

        # Sort and forward fill missing values
        result = result.sort_values("date").reset_index(drop=True)
        if self.config.fill_missing_method == "ffill":
            result = result.ffill()

        return result

    # =========================================================================
    # Feature Engineering
    # =========================================================================

    def compute_vix_regime(self, vix_value: float) -> float:
        """
        Compute VIX regime indicator (0-1 scale).

        0.0 = Low volatility (complacency)
        0.5 = Normal volatility
        1.0 = Extreme volatility (fear)

        Args:
            vix_value: Current VIX value

        Returns:
            Regime indicator (0-1)
        """
        if vix_value <= VIX_REGIMES["low"]:
            return 0.0
        elif vix_value <= VIX_REGIMES["normal_low"]:
            return 0.25
        elif vix_value <= VIX_REGIMES["normal_high"]:
            return 0.5
        elif vix_value <= VIX_REGIMES["elevated"]:
            return 0.75
        else:
            return 1.0

    def normalize_vix(self, vix_value: float) -> float:
        """
        Normalize VIX value using tanh transformation.

        Centers at VIX=20, scales to approximately [-1, 1].

        Args:
            vix_value: Raw VIX value

        Returns:
            Normalized VIX (approximately [-1, 1])
        """
        return float(np.tanh(
            (vix_value - self.config.vix_normalization_center)
            / self.config.vix_normalization_scale
        ))

    def normalize_dxy(self, dxy_value: float) -> float:
        """
        Normalize DXY value using tanh transformation.

        Centers at DXY=100, scales to approximately [-1, 1].

        Args:
            dxy_value: Raw DXY value

        Returns:
            Normalized DXY (approximately [-1, 1])
        """
        return float(np.tanh(
            (dxy_value - self.config.dxy_normalization_center)
            / self.config.dxy_normalization_scale
        ))

    def normalize_treasury(self, yield_value: float) -> float:
        """
        Normalize Treasury yield using tanh transformation.

        Centers at 3.5%, scales to approximately [-1, 1].

        Args:
            yield_value: Raw yield value (percentage)

        Returns:
            Normalized yield (approximately [-1, 1])
        """
        return float(np.tanh(
            (yield_value - self.config.treasury_normalization_center)
            / self.config.treasury_normalization_scale
        ))

    def compute_real_yield_proxy(
        self,
        treasury_10y: float,
        vix_value: float,
    ) -> float:
        """
        Compute real yield proxy.

        Real yield ≈ nominal yield - inflation expectation
        We use VIX as a crude proxy for uncertainty/inflation expectation.

        Args:
            treasury_10y: 10-year Treasury yield
            vix_value: VIX value

        Returns:
            Real yield proxy
        """
        # Crude approximation: assume VIX/10 as inflation proxy
        # High VIX → higher inflation expectation → lower real yield
        inflation_proxy = vix_value / 10.0
        return treasury_10y - inflation_proxy

    def add_macro_features(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Add macro features to a trading DataFrame.

        Features added:
        - vix_value: Raw VIX value
        - vix_normalized: Normalized VIX (-1 to 1)
        - vix_regime: VIX regime indicator (0 to 1)
        - dxy_value: Raw DXY value
        - dxy_normalized: Normalized DXY (-1 to 1)
        - treasury_10y_yield: 10-year Treasury yield
        - treasury_10y_normalized: Normalized yield
        - real_yield_proxy: Estimated real yield

        Args:
            df: DataFrame with OHLCV data (must have timestamp or date column)
            symbol: Stock symbol (for logging)
            force_refresh: Bypass cache

        Returns:
            DataFrame with macro features added
        """
        df = df.copy()

        # Get date column
        if "timestamp" in df.columns:
            df["_date"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%d")
        elif "date" in df.columns:
            df["_date"] = df["date"].astype(str)
        else:
            logger.warning("No timestamp or date column found, cannot add macro features")
            return df

        # Get date range
        dates_list = df["_date"].tolist()
        if not dates_list:
            df.drop(columns=["_date"], inplace=True)
            return df

        start_date = min(dates_list)
        end_date = max(dates_list)

        # Fetch macro data
        try:
            macro_df = self.get_all_macro(
                start_date=start_date,
                end_date=end_date,
                force_refresh=force_refresh,
            )
        except Exception as e:
            logger.warning(f"Failed to fetch macro data: {e}")
            df.drop(columns=["_date"], inplace=True)
            return df

        if macro_df.empty:
            logger.warning("No macro data available for date range")
            df.drop(columns=["_date"], inplace=True)
            return df

        # Merge macro data
        df = df.merge(macro_df, left_on="_date", right_on="date", how="left")

        # Forward fill missing values
        macro_cols = ["vix", "dxy", "treasury_10y", "treasury_30y"]
        for col in macro_cols:
            if col in df.columns:
                df[col] = df[col].ffill()

        # Add raw values
        if "vix" in df.columns:
            df["vix_value"] = df["vix"]
        else:
            df["vix_value"] = np.nan

        if "dxy" in df.columns:
            df["dxy_value"] = df["dxy"]
        else:
            df["dxy_value"] = np.nan

        if "treasury_10y" in df.columns:
            df["treasury_10y_yield"] = df["treasury_10y"]
        else:
            df["treasury_10y_yield"] = np.nan

        # Add normalized features
        df["vix_normalized"] = df["vix_value"].apply(
            lambda x: self.normalize_vix(x) if pd.notna(x) else 0.0
        )
        df["vix_regime"] = df["vix_value"].apply(
            lambda x: self.compute_vix_regime(x) if pd.notna(x) else 0.5
        )
        df["dxy_normalized"] = df["dxy_value"].apply(
            lambda x: self.normalize_dxy(x) if pd.notna(x) else 0.0
        )
        df["treasury_10y_normalized"] = df["treasury_10y_yield"].apply(
            lambda x: self.normalize_treasury(x) if pd.notna(x) else 0.0
        )

        # Add real yield proxy
        def compute_real_yield(row):
            if pd.notna(row.get("treasury_10y_yield")) and pd.notna(row.get("vix_value")):
                return self.compute_real_yield_proxy(
                    row["treasury_10y_yield"],
                    row["vix_value"],
                )
            return np.nan

        df["real_yield_proxy"] = df.apply(compute_real_yield, axis=1)

        # Clean up
        df.drop(columns=["_date"], errors="ignore", inplace=True)
        if "date" in df.columns and "date_y" not in df.columns:
            pass  # Keep original date column
        else:
            df.drop(columns=["date"], errors="ignore", inplace=True)

        # Drop intermediate columns
        for col in macro_cols:
            if col in df.columns:
                df.drop(columns=[col], errors="ignore", inplace=True)

        logger.info(f"Added macro features to {len(df)} rows")
        return df

    def preload_all(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> None:
        """
        Preload all macro indicators to cache.

        Use this to warm up cache before batch processing.

        Args:
            start_date: Start date
            end_date: End date
            force_refresh: Bypass existing cache
        """
        logger.info(f"Preloading macro data from {start_date} to {end_date or 'now'}")

        indicators = ["vix", "vix_3m", "dxy", "treasury_10y", "treasury_30y"]
        for indicator in indicators:
            try:
                self._fetch_indicator(indicator, start_date, end_date, force_refresh)
            except Exception as e:
                logger.warning(f"Error preloading {indicator}: {e}")

        logger.info("Macro data preload complete")

    def get_latest_values(self) -> Dict[str, float]:
        """
        Get latest values for all macro indicators.

        Returns:
            Dict mapping indicator to latest value
        """
        result = {}

        for indicator in ["vix", "dxy", "treasury_10y", "treasury_30y"]:
            try:
                df = self._fetch_indicator(indicator)
                if not df.empty:
                    result[indicator] = float(df["value"].iloc[-1])
            except Exception as e:
                logger.warning(f"Error getting latest {indicator}: {e}")

        return result


# =============================================================================
# Module-level convenience functions
# =============================================================================

_default_service: Optional[MacroDataService] = None


def get_service() -> MacroDataService:
    """Get default MacroDataService instance."""
    global _default_service
    if _default_service is None:
        _default_service = MacroDataService()
    return _default_service


def get_vix(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Convenience function to get VIX data."""
    return get_service().get_vix(start_date, end_date, force_refresh)


def get_dxy(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Convenience function to get DXY data."""
    return get_service().get_dxy(start_date, end_date, force_refresh)


def get_treasury_yields(
    tenors: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Convenience function to get Treasury yield data."""
    return get_service().get_treasury_yields(tenors, start_date, end_date, force_refresh)


def add_macro_features(
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Convenience function to add macro features to DataFrame."""
    return get_service().add_macro_features(df, symbol, force_refresh)
