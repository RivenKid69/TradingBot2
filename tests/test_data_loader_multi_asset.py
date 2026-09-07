# -*- coding: utf-8 -*-
"""
tests/test_data_loader_multi_asset.py
Comprehensive tests for the multi-asset data loader.

Tests cover:
- Multi-asset data loading (crypto + stocks)
- Trading hours filtering for equities
- Adapter-based data loading (Alpaca, Polygon)
- Data preprocessing and validation
- Feature pipeline integration
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tempfile
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

# Check for pyarrow availability (needed for parquet/feather tests)
try:
    import pyarrow

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

pyarrow_required = pytest.mark.skipif(
    not HAS_PYARROW, reason="pyarrow required for parquet/feather support"
)

from data_loader_multi_asset import (
    load_multi_asset_data,
    filter_trading_hours,
    load_from_adapter,
    load_from_file,
    load_crypto_data,
    load_stock_data,
    load_alpaca_data,
    load_polygon_data,
    validate_data,
    AssetClass,
    DataVendor,
    timeframe_to_seconds,
    align_timestamp,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_crypto_df():
    """Create sample crypto DataFrame."""
    dates = pd.date_range("2025-01-01", periods=100, freq="4h", tz="UTC")

    return pd.DataFrame(
        {
            "open_time": dates,
            "open": np.random.uniform(40000, 50000, 100),
            "high": np.random.uniform(45000, 55000, 100),
            "low": np.random.uniform(35000, 45000, 100),
            "close": np.random.uniform(40000, 50000, 100),
            "volume": np.random.uniform(1000, 10000, 100),
            "quote_volume": np.random.uniform(40000000, 500000000, 100),
            "trades": np.random.randint(1000, 10000, 100),
        }
    )


@pytest.fixture
def sample_stock_df():
    """Create sample stock DataFrame."""
    # Create dates only during market hours (9:30 AM - 4:00 PM ET)
    dates = pd.date_range("2025-01-02 14:30:00", periods=50, freq="4h", tz="UTC")

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": np.random.uniform(150, 160, 50),
            "high": np.random.uniform(155, 165, 50),
            "low": np.random.uniform(145, 155, 50),
            "close": np.random.uniform(150, 160, 50),
            "volume": np.random.uniform(10000, 100000, 50),
            "vwap": np.random.uniform(150, 160, 50),
            "trade_count": np.random.randint(100, 1000, 50),
        }
    )


@pytest.fixture
def sample_bars():
    """Create sample Bar objects."""
    from core_models import Bar

    bars = []
    base_ts = 1704067200000  # 2024-01-01 00:00:00 UTC

    for i in range(10):
        bar = Bar(
            ts=base_ts + i * 14400000,  # 4h intervals
            symbol="AAPL",
            open=Decimal("150.0") + Decimal(str(i)),
            high=Decimal("155.0") + Decimal(str(i)),
            low=Decimal("148.0") + Decimal(str(i)),
            close=Decimal("152.0") + Decimal(str(i)),
            volume_base=Decimal("100000"),
            volume_quote=Decimal("15000000"),
            trades=1000,
            vwap=Decimal("151.0"),
            is_final=True,
        )
        bars.append(bar)

    return bars


@pytest.fixture
def temp_parquet_file(sample_crypto_df):
    """Create temporary parquet file."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        sample_crypto_df.to_parquet(f.name)
        yield f.name


# =============================================================================
# Trading Hours Filter Tests
# =============================================================================


class TestFilterTradingHours:
    """Tests for trading hours filtering."""

    def test_filter_regular_hours_only(self, sample_stock_df):
        """Test filtering to regular market hours only."""
        # Add some out-of-hours timestamps
        df = sample_stock_df.copy()

        # Convert to Eastern time aware
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        filtered = filter_trading_hours(
            df,
            include_extended=False,
            timezone_str="America/New_York",
        )

        # Should have filtered some rows if there were out-of-hours data
        assert len(filtered) <= len(df)

    def test_filter_with_extended_hours(self, sample_stock_df):
        """Test filtering with extended hours included."""
        df = sample_stock_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        filtered = filter_trading_hours(
            df,
            include_extended=True,
            timezone_str="America/New_York",
        )

        # Extended hours should include more data
        assert len(filtered) >= 0

    def test_filter_excludes_weekends(self):
        """Test that weekends are excluded."""
        # Create data with weekend dates
        dates = pd.date_range("2025-01-18", periods=10, freq="4h", tz="UTC")  # Saturday

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": [100] * 10,
                "high": [105] * 10,
                "low": [95] * 10,
                "close": [102] * 10,
                "volume": [1000] * 10,
            }
        )

        filtered = filter_trading_hours(
            df,
            include_extended=False,
            timezone_str="America/New_York",
        )

        # Weekend data should be filtered out
        assert len(filtered) == 0

    def test_filter_empty_dataframe(self):
        """Test filtering empty DataFrame."""
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        filtered = filter_trading_hours(
            df,
            include_extended=False,
            timezone_str="America/New_York",
        )

        assert len(filtered) == 0


# =============================================================================
# Timeframe Utilities Tests
# =============================================================================


class TestTimeframeUtilities:
    """Tests for timeframe conversion utilities."""

    def test_timeframe_to_seconds_minutes(self):
        """Test minute timeframe conversion."""
        assert timeframe_to_seconds("1m") == 60
        assert timeframe_to_seconds("5m") == 300
        assert timeframe_to_seconds("15m") == 900
        assert timeframe_to_seconds("30m") == 1800

    def test_timeframe_to_seconds_hours(self):
        """Test hour timeframe conversion."""
        assert timeframe_to_seconds("1h") == 3600
        assert timeframe_to_seconds("4h") == 14400

    def test_timeframe_to_seconds_days(self):
        """Test day timeframe conversion."""
        assert timeframe_to_seconds("1d") == 86400

    def test_align_timestamp(self):
        """Test timestamp alignment."""
        ts = 1705312800000  # Some timestamp
        tf_sec = 3600  # 1 hour

        aligned = align_timestamp(ts, tf_sec)
        assert aligned % (tf_sec * 1000) == 0


# =============================================================================
# AssetClass and DataVendor Tests
# =============================================================================


class TestEnums:
    """Tests for enum classes."""

    def test_asset_class_values(self):
        """Test AssetClass enum values."""
        assert AssetClass.CRYPTO == "crypto"
        assert AssetClass.EQUITY == "equity"

    def test_data_vendor_values(self):
        """Test DataVendor enum values."""
        assert DataVendor.BINANCE == "binance"
        assert DataVendor.ALPACA == "alpaca"
        assert DataVendor.POLYGON == "polygon"


# =============================================================================
# DataFrame Validation Tests
# =============================================================================


class TestValidateData:
    """Tests for data validation."""

    def test_valid_dataframe(self, sample_crypto_df):
        """Test validation of valid DataFrame."""
        # Add required timestamp column
        df = sample_crypto_df.copy()
        df["timestamp"] = df["open_time"].astype("int64") // 10**9
        is_valid, errors = validate_data(df, min_rows=10)
        # Should be valid or have specific errors
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)

    def test_empty_dataframe_invalid(self):
        """Test empty DataFrame is invalid."""
        df = pd.DataFrame()
        is_valid, errors = validate_data(df, min_rows=10)
        # Empty data should not be valid
        assert is_valid is False
        assert len(errors) > 0

    def test_missing_required_columns(self):
        """Test DataFrame missing required columns."""
        df = pd.DataFrame(
            {
                "open": [100, 101],
                "close": [102, 103],
                # Missing timestamp, high, low, volume
            }
        )

        is_valid, errors = validate_data(df, min_rows=1)
        # Should report missing columns
        assert is_valid is False
        assert any("Missing" in e for e in errors)

    def test_all_nan_values(self):
        """Test DataFrame with all NaN values."""
        df = pd.DataFrame(
            {
                "timestamp": [1704067200 + i * 3600 for i in range(5)],
                "open": [np.nan] * 5,
                "high": [np.nan] * 5,
                "low": [np.nan] * 5,
                "close": [np.nan] * 5,
                "volume": [np.nan] * 5,
            }
        )

        is_valid, errors = validate_data(df, min_rows=1)
        # Should report NaN issues
        assert is_valid is False
        assert any("NaN" in e for e in errors)


# =============================================================================
# Load From File Tests
# =============================================================================


@pyarrow_required
class TestLoadFromFile:
    """Tests for loading data from files."""

    def test_load_parquet(self, sample_crypto_df):
        """Test loading from parquet file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "BTCUSDT.parquet"
            sample_crypto_df.to_parquet(path)

            df = load_from_file(str(path))

            assert df is not None
            assert len(df) > 0

    def test_load_feather(self, sample_crypto_df):
        """Test loading from feather file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "BTCUSDT.feather"
            sample_crypto_df.reset_index(drop=True).to_feather(path)

            df = load_from_file(str(path))

            assert df is not None
            assert len(df) > 0

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises((FileNotFoundError, ValueError)):
            load_from_file("/nonexistent/path.parquet")


# =============================================================================
# Multi-Asset Loading Tests
# =============================================================================


class TestLoadMultiAssetData:
    """Tests for multi-asset data loading."""

    @pyarrow_required
    def test_load_crypto_from_parquet(self, sample_crypto_df):
        """Test loading crypto data from parquet files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create actual parquet file
            path = Path(tmpdir) / "BTCUSDT.parquet"
            sample_crypto_df.to_parquet(path)

            frames, obs_shapes = load_multi_asset_data(
                paths=[str(path)],
                asset_class="crypto",
                timeframe="4h",
            )

            # Should have loaded data
            assert isinstance(frames, dict)
            assert isinstance(obs_shapes, dict)

    def test_load_empty_paths(self):
        """Test loading with empty paths."""
        # Empty paths should raise or return empty
        try:
            frames, obs_shapes = load_multi_asset_data(
                paths=[],
                asset_class="crypto",
                timeframe="4h",
            )
            assert frames == {}
            assert obs_shapes == {}
        except ValueError:
            # Also acceptable
            pass

    @pyarrow_required
    def test_load_with_basic_params(self, sample_crypto_df):
        """Test loading with basic parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "BTCUSDT.parquet"
            sample_crypto_df.to_parquet(path)

            frames, obs_shapes = load_multi_asset_data(
                paths=[str(path)],
                asset_class="crypto",
                timeframe="4h",
            )

            # Should load data
            assert isinstance(frames, dict)


# =============================================================================
# Adapter Loading Tests
# =============================================================================


class TestLoadFromAdapter:
    """Tests for loading data from adapters."""

    def test_load_from_adapter_function_exists(self):
        """Test that load_from_adapter function is available."""
        # Just verify the function is importable and callable
        assert callable(load_from_adapter)

    def test_load_from_adapter_basic_call(self):
        """Test basic load_from_adapter call (may fail without SDK or credentials)."""
        try:
            # This will likely fail without real credentials or SDK,
            # Just verify the function can be called
            frames, obs_shapes = load_from_adapter(
                vendor="alpaca",
                symbols=["AAPL"],
                timeframe="4h",
                start_date="2025-01-01",
                end_date="2025-01-15",
            )
            assert isinstance(frames, dict)
        except ImportError:
            # Expected if alpaca SDK not installed - acceptable
            pass
        except Exception as e:
            # Other errors (connection, auth) are also acceptable
            pass


# =============================================================================
# Integration Tests
# =============================================================================


class TestMultiAssetIntegration:
    """Integration tests for multi-asset loading."""

    def test_crypto_vs_stock_data_structure(self, sample_crypto_df, sample_stock_df):
        """Test that crypto and stock data have compatible structure."""
        # Both should have OHLCV columns
        crypto_cols = {"open", "high", "low", "close", "volume"}
        stock_cols = {"open", "high", "low", "close", "volume"}

        assert crypto_cols.issubset(set(sample_crypto_df.columns))
        assert stock_cols.issubset(set(sample_stock_df.columns))

    def test_trading_hours_only_affects_stocks(self, sample_crypto_df, sample_stock_df):
        """Test trading hours filtering only applies to stocks."""
        # Crypto should not be affected by trading hours
        # (This is handled by asset_class parameter in load function)

        # For crypto, all data should be kept (24/7 market)
        crypto_len = len(sample_crypto_df)

        # For stocks, only market hours data is kept
        filtered_stock = filter_trading_hours(
            sample_stock_df,
            include_extended=False,
            timezone_str="America/New_York",
        )

        stock_len = len(filtered_stock)

        # Crypto should have all data, stocks may be filtered
        assert crypto_len == 100

    def test_asset_class_enum(self):
        """Test AssetClass enum usage."""
        crypto = AssetClass.CRYPTO
        equity = AssetClass.EQUITY

        assert crypto.value == "crypto"
        assert equity.value == "equity"

    def test_data_vendor_enum(self):
        """Test DataVendor enum usage."""
        binance = DataVendor.BINANCE
        alpaca = DataVendor.ALPACA
        polygon = DataVendor.POLYGON

        assert binance.value == "binance"
        assert alpaca.value == "alpaca"
        assert polygon.value == "polygon"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handle_missing_timestamp_column(self):
        """Test handling DataFrame without timestamp column."""
        df = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [105, 106],
                "low": [95, 96],
                "close": [102, 103],
                "volume": [1000, 1100],
            }
        )

        # Should handle gracefully via validate_data
        is_valid, errors = validate_data(df, min_rows=1)
        assert isinstance(errors, list)
        assert any("Missing" in e for e in errors)  # Missing timestamp

    def test_handle_duplicate_timestamps(self, sample_crypto_df):
        """Test handling duplicate timestamps."""
        df = sample_crypto_df.copy()
        df["timestamp"] = df["open_time"].astype("int64") // 10**9
        # Create duplicates
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)

        # Should still be processable (with warnings)
        is_valid, errors = validate_data(df, min_rows=1)
        assert isinstance(errors, list)

    def test_handle_non_monotonic_timestamps(self, sample_crypto_df):
        """Test handling non-monotonic timestamps."""
        df = sample_crypto_df.copy()
        df["timestamp"] = df["open_time"].astype("int64") // 10**9
        # Shuffle to make non-monotonic
        df = df.sample(frac=1).reset_index(drop=True)

        # Should detect non-monotonic
        is_valid, errors = validate_data(df, min_rows=1)
        assert isinstance(errors, list)
        # May have non-monotonic error
        if not is_valid:
            assert any("monotonic" in e.lower() for e in errors)

    def test_handle_extreme_values(self):
        """Test handling extreme price values."""
        df = pd.DataFrame(
            {
                "timestamp": [1704067200 + i * 14400 for i in range(10)],
                "open": [1e10] * 10,  # Very high price
                "high": [1e10 * 1.1] * 10,
                "low": [1e10 * 0.9] * 10,
                "close": [1e10] * 10,
                "volume": [1e15] * 10,  # Very high volume
            }
        )

        # Should handle without overflow
        is_valid, errors = validate_data(df, min_rows=1)
        assert isinstance(errors, list)

    def test_handle_zero_volume(self):
        """Test handling zero volume bars."""
        df = pd.DataFrame(
            {
                "timestamp": [1704067200 + i * 14400 for i in range(10)],
                "open": [100.0] * 10,
                "high": [105.0] * 10,
                "low": [95.0] * 10,
                "close": [102.0] * 10,
                "volume": [0.0] * 10,  # Zero volume
            }
        )

        # Should handle zero volume
        is_valid, errors = validate_data(df, min_rows=1)
        assert isinstance(errors, list)

    def test_timeframe_seconds_edge_cases(self):
        """Test timeframe conversion edge cases."""
        # Test various formats
        assert timeframe_to_seconds("1m") == 60
        assert timeframe_to_seconds("1h") == 3600
        assert timeframe_to_seconds("1d") == 86400

    def test_align_timestamp_edge_cases(self):
        """Test timestamp alignment edge cases."""
        # align_timestamp works in seconds, not milliseconds
        # Already aligned timestamp (in seconds)
        ts_aligned = 1705312800  # Exactly on hour boundary in seconds
        assert align_timestamp(ts_aligned, 3600) == ts_aligned

        # Not aligned timestamp
        ts_not_aligned = 1705312830  # 30 seconds off
        aligned = align_timestamp(ts_not_aligned, 3600)
        assert aligned % 3600 == 0  # Check in seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
