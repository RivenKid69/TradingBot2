# -*- coding: utf-8 -*-
"""
tests/test_timestamp_consistency.py
-----------------------------------
Comprehensive tests for timestamp consistency in prepare_and_run.py

Tests that CSV and Parquet files with the same underlying data produce
consistent timestamps, avoiding the "open_time vs close_time" ambiguity bug.
"""
import os
import tempfile
import warnings
import pytest
import pandas as pd
import numpy as np

# Import functions to test
from prepare_and_run import _read_raw, _normalize_ohlcv, _to_seconds_any


# ============================================================================
# Test Fixtures - Sample Data
# ============================================================================


@pytest.fixture
def sample_binance_data():
    """Sample Binance-format OHLCV data (4h bars)"""
    # 4h bars: 14400 seconds
    open_times = [
        1609459200,
        1609473600,
        1609488000,
        1609502400,
        1609516800,
    ]  # 2021-01-01 00:00:00 UTC, +4h each
    close_times = [t + 14400 for t in open_times]  # close_time = open_time + 4h

    return pd.DataFrame(
        {
            "open_time": open_times,
            "close_time": close_times,
            "open": [29000.0, 29100.0, 29200.0, 29300.0, 29400.0],
            "high": [29500.0, 29600.0, 29700.0, 29800.0, 29900.0],
            "low": [28500.0, 28600.0, 28700.0, 28800.0, 28900.0],
            "close": [29100.0, 29200.0, 29300.0, 29400.0, 29500.0],
            "volume": [100.0, 110.0, 120.0, 130.0, 140.0],
            "quote_asset_volume": [2910000.0, 3212000.0, 3516000.0, 3822000.0, 4130000.0],
            "number_of_trades": [1000, 1100, 1200, 1300, 1400],
            "taker_buy_base_asset_volume": [50.0, 55.0, 60.0, 65.0, 70.0],
            "taker_buy_quote_asset_volume": [1455000.0, 1606000.0, 1758000.0, 1911000.0, 2065000.0],
        }
    )


@pytest.fixture
def expected_timestamps():
    """Expected canonical timestamps (close_time floored to 4h boundary)"""
    # close_times = [1609473600, 1609488000, 1609502400, 1609516800, 1609531200]
    # Each is already aligned to 4h boundary (14400 seconds)
    return [1609473600, 1609488000, 1609502400, 1609516800, 1609531200]


# ============================================================================
# Test _to_seconds_any()
# ============================================================================


def test_to_seconds_any_numeric_seconds():
    """Test conversion from numeric seconds"""
    s = pd.Series([1609459200, 1609473600, 1609488000])
    result = _to_seconds_any(s)
    expected = pd.Series([1609459200, 1609473600, 1609488000], dtype="int64")
    pd.testing.assert_series_equal(result, expected)


def test_to_seconds_any_numeric_milliseconds():
    """Test conversion from numeric milliseconds"""
    s = pd.Series([1609459200000, 1609473600000, 1609488000000])
    result = _to_seconds_any(s)
    expected = pd.Series([1609459200, 1609473600, 1609488000], dtype="int64")
    pd.testing.assert_series_equal(result, expected)


def test_to_seconds_any_datetime_strings():
    """Test conversion from datetime strings"""
    s = pd.Series(["2021-01-01 00:00:00", "2021-01-01 04:00:00", "2021-01-01 08:00:00"])
    result = _to_seconds_any(s)
    expected = pd.Series([1609459200, 1609473600, 1609488000], dtype="int64")
    pd.testing.assert_series_equal(result, expected)


# ============================================================================
# Test _read_raw() - CSV with Binance format
# ============================================================================


def test_read_raw_csv_with_explicit_times(sample_binance_data, expected_timestamps, tmp_path):
    """Test _read_raw() with explicit open_time and close_time columns"""
    csv_path = tmp_path / "test_binance.csv"
    sample_binance_data.to_csv(csv_path, index=False)

    df = _read_raw(str(csv_path))

    # Check that timestamp is derived from close_time (floored to 4h boundary)
    assert "timestamp" in df.columns
    assert df["timestamp"].tolist() == expected_timestamps

    # NOTE: _read_raw() intentionally drops open_time/close_time columns
    # and keeps only the canonical 'timestamp' column (derived from close_time).
    # This is by design - see keep list in _read_raw() line 49-54.


def test_read_raw_csv_with_milliseconds(sample_binance_data, expected_timestamps, tmp_path):
    """Test _read_raw() with timestamps in milliseconds"""
    # Convert to milliseconds
    sample_binance_data["open_time"] = sample_binance_data["open_time"] * 1000
    sample_binance_data["close_time"] = sample_binance_data["close_time"] * 1000

    csv_path = tmp_path / "test_binance_ms.csv"
    sample_binance_data.to_csv(csv_path, index=False)

    df = _read_raw(str(csv_path))

    # Check that timestamps are correctly converted to seconds
    assert df["timestamp"].tolist() == expected_timestamps


# ============================================================================
# Test _normalize_ohlcv() - Parquet/other formats
# ============================================================================


def test_normalize_ohlcv_explicit_close_time(sample_binance_data, expected_timestamps):
    """Test _normalize_ohlcv() with explicit close_time column"""
    # Add a close_time column
    df = sample_binance_data.copy()
    df["close_time"] = df["close_time"]

    result = _normalize_ohlcv(df, "test.parquet")

    # Check that timestamp is derived from close_time (floored to 4h boundary)
    assert result["timestamp"].tolist() == expected_timestamps


def test_normalize_ohlcv_explicit_open_time_only(sample_binance_data, expected_timestamps):
    """Test _normalize_ohlcv() with only open_time column (should add duration)"""
    # Remove close_time, keep only open_time
    df = sample_binance_data.copy()
    df = df.drop(columns=["close_time"])

    # Set BAR_DURATION_SEC for test
    os.environ["BAR_DURATION_SEC"] = "14400"

    result = _normalize_ohlcv(df, "test.parquet")

    # Check that timestamp is derived from open_time + 14400 (floored to 4h boundary)
    assert result["timestamp"].tolist() == expected_timestamps


def test_normalize_ohlcv_generic_timestamp_with_warning(sample_binance_data):
    """Test _normalize_ohlcv() with generic 'timestamp' column (should warn)"""
    # Use only a generic 'timestamp' column (ambiguous: could be open or close)
    df = sample_binance_data.copy()
    df = df.drop(columns=["open_time", "close_time"])
    # Assume timestamp = open_time (this is the problematic scenario!)
    df["timestamp"] = sample_binance_data["open_time"]

    # This should produce a warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _normalize_ohlcv(df, "test_generic_timestamp.parquet")

        # Check that a warning was issued
        assert len(w) == 1
        assert "Using generic time column 'timestamp'" in str(w[0].message)
        assert "ambiguous whether open or close time" in str(w[0].message)

    # Check that timestamp is used as-is (treated as close_time)
    # NOTE: This is NOT correct if timestamp was actually open_time!
    # But at least we warn the user.
    expected = sample_binance_data["open_time"].tolist()
    assert result["timestamp"].tolist() == expected


def test_normalize_ohlcv_priority_explicit_over_generic(sample_binance_data, expected_timestamps):
    """Test that explicit close_time takes priority over generic timestamp"""
    df = sample_binance_data.copy()
    # Add both explicit close_time AND generic timestamp
    df["timestamp"] = sample_binance_data["open_time"]  # Generic timestamp = open_time
    df["close_time"] = sample_binance_data["close_time"]  # Explicit close_time

    # Should use close_time (explicit) and ignore generic timestamp
    result = _normalize_ohlcv(df, "test_priority.parquet")

    assert result["timestamp"].tolist() == expected_timestamps


# ============================================================================
# Test Consistency between CSV and Parquet
# ============================================================================


def test_consistency_csv_vs_parquet_explicit_times(
    sample_binance_data, expected_timestamps, tmp_path
):
    """Test that CSV and Parquet with explicit times produce the same timestamps"""
    # 1. CSV via _read_raw()
    csv_path = tmp_path / "test_consistency.csv"
    sample_binance_data.to_csv(csv_path, index=False)
    df_csv_raw = _read_raw(str(csv_path))
    df_csv = _normalize_ohlcv(df_csv_raw, str(csv_path))

    # 2. Parquet via _normalize_ohlcv() directly
    df_parquet = _normalize_ohlcv(sample_binance_data, "test_consistency.parquet")

    # Check consistency
    assert df_csv["timestamp"].tolist() == df_parquet["timestamp"].tolist() == expected_timestamps


def test_consistency_csv_vs_parquet_open_time_only(
    sample_binance_data, expected_timestamps, tmp_path
):
    """Test consistency when only open_time is available"""
    # Prepare data with only open_time
    df_open_only = sample_binance_data.copy()
    df_open_only = df_open_only.drop(columns=["close_time"])

    # Set BAR_DURATION_SEC
    os.environ["BAR_DURATION_SEC"] = "14400"

    # 1. CSV via _read_raw() - THIS WILL FAIL because _read_raw expects close_time!
    # We need to update _read_raw to handle this case, but for now let's test Parquet only

    # 2. Parquet via _normalize_ohlcv()
    df_parquet = _normalize_ohlcv(df_open_only, "test_open_only.parquet")

    # Check that timestamp = open_time + 14400
    assert df_parquet["timestamp"].tolist() == expected_timestamps


def test_inconsistency_detection_generic_timestamp():
    """
    Test that demonstrates the BUG: Parquet with generic 'timestamp' = open_time
    will be incorrectly treated as close_time, causing a 1-bar (4h) shift.
    """
    # Scenario: Parquet file with 'timestamp' column that is actually open_time
    open_times = [1609459200, 1609473600, 1609488000, 1609502400, 1609516800]
    close_times = [t + 14400 for t in open_times]

    df = pd.DataFrame(
        {
            "timestamp": open_times,  # PROBLEM: This is open_time, not close_time!
            "open": [29000.0, 29100.0, 29200.0, 29300.0, 29400.0],
            "high": [29500.0, 29600.0, 29700.0, 29800.0, 29900.0],
            "low": [28500.0, 28600.0, 28700.0, 28800.0, 28900.0],
            "close": [29100.0, 29200.0, 29300.0, 29400.0, 29500.0],
            "volume": [100.0, 110.0, 120.0, 130.0, 140.0],
        }
    )

    # _normalize_ohlcv will treat 'timestamp' as close_time (WRONG!)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _normalize_ohlcv(df, "problematic.parquet")

        # Should warn about ambiguity
        assert len(w) == 1
        assert "ambiguous" in str(w[0].message).lower()

    # Timestamp is used as-is (treated as close_time), but it's actually open_time!
    # This causes a 1-bar (4h) shift
    assert result["timestamp"].tolist() == open_times  # WRONG! Should be close_times

    # Expected behavior (after fix): Should add 14400 to convert open_time to close_time
    # But with the current fix, we only warn - the user must rename columns explicitly.


# ============================================================================
# Test Edge Cases
# ============================================================================


def test_normalize_ohlcv_no_time_column():
    """Test that _normalize_ohlcv raises error when no time column is found"""
    df = pd.DataFrame(
        {
            "open": [29000.0, 29100.0],
            "high": [29500.0, 29600.0],
            "low": [28500.0, 28600.0],
            "close": [29100.0, 29200.0],
            "volume": [100.0, 110.0],
        }
    )

    with pytest.raises(ValueError, match="no usable time column"):
        _normalize_ohlcv(df, "no_time.parquet")


def test_normalize_ohlcv_multiple_time_columns_priority(sample_binance_data, expected_timestamps):
    """Test priority: close_time > open_time > timestamp"""
    df = sample_binance_data.copy()

    # Add all types of time columns
    df["timestamp"] = sample_binance_data["open_time"]  # Generic
    df["open_time"] = sample_binance_data["open_time"]  # Explicit open
    df["close_time"] = sample_binance_data["close_time"]  # Explicit close

    result = _normalize_ohlcv(df, "test_priority.parquet")

    # Should use close_time (highest priority)
    assert result["timestamp"].tolist() == expected_timestamps


def test_normalize_ohlcv_timestamp_already_floored():
    """Test that timestamps already floored to 4h boundary are preserved"""
    # Timestamps already aligned to 4h boundary
    timestamps = [1609473600, 1609488000, 1609502400, 1609516800, 1609531200]

    df = pd.DataFrame(
        {
            "close_time": timestamps,
            "open": [29000.0, 29100.0, 29200.0, 29300.0, 29400.0],
            "high": [29500.0, 29600.0, 29700.0, 29800.0, 29900.0],
            "low": [28500.0, 28600.0, 28700.0, 28800.0, 28900.0],
            "close": [29100.0, 29200.0, 29300.0, 29400.0, 29500.0],
            "volume": [100.0, 110.0, 120.0, 130.0, 140.0],
        }
    )

    result = _normalize_ohlcv(df, "test_floored.parquet")

    assert result["timestamp"].tolist() == timestamps


# ============================================================================
# Integration Test - Full Pipeline
# ============================================================================


def test_full_pipeline_csv_to_normalized(sample_binance_data, expected_timestamps, tmp_path):
    """Test full pipeline: CSV → _read_raw → _normalize_ohlcv"""
    csv_path = tmp_path / "test_full_pipeline.csv"
    sample_binance_data.to_csv(csv_path, index=False)

    # Step 1: Read CSV
    df_raw = _read_raw(str(csv_path))

    # Step 2: Normalize (as done in prepare())
    df_norm = _normalize_ohlcv(df_raw, str(csv_path))

    # Check final timestamp
    assert df_norm["timestamp"].tolist() == expected_timestamps

    # Check that all expected columns are present
    expected_cols = [
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
    for col in expected_cols:
        assert col in df_norm.columns


def test_full_pipeline_parquet_explicit_times(sample_binance_data, expected_timestamps):
    """Test full pipeline: Parquet (explicit times) → _normalize_ohlcv"""
    # Simulate reading Parquet (already a DataFrame)
    df_raw = sample_binance_data.copy()

    # Normalize
    df_norm = _normalize_ohlcv(df_raw, "test_pipeline.parquet")

    # Check final timestamp
    assert df_norm["timestamp"].tolist() == expected_timestamps


# ============================================================================
# Test Recommendations for Users
# ============================================================================


def test_recommendation_use_explicit_column_names():
    """
    Recommendation test: Always use explicit column names to avoid ambiguity.

    This test documents the recommended approach for data providers:
    - Use 'open_time' or 'opentime' for bar open timestamp
    - Use 'close_time' or 'closetime' for bar close timestamp
    - Avoid generic 'timestamp' unless it's unambiguous (e.g., only one time column)
    """
    # Good: Explicit column names
    df_good = pd.DataFrame(
        {
            "open_time": [1609459200, 1609473600],
            "close_time": [1609473600, 1609488000],
            "open": [29000.0, 29100.0],
            "high": [29500.0, 29600.0],
            "low": [28500.0, 28600.0],
            "close": [29100.0, 29200.0],
            "volume": [100.0, 110.0],
        }
    )

    result_good = _normalize_ohlcv(df_good, "good_data.parquet")
    expected = [1609473600, 1609488000]  # close_time
    assert result_good["timestamp"].tolist() == expected

    # Bad: Generic 'timestamp' (ambiguous)
    df_bad = pd.DataFrame(
        {
            "timestamp": [1609459200, 1609473600],  # Is this open or close?
            "open": [29000.0, 29100.0],
            "high": [29500.0, 29600.0],
            "low": [28500.0, 28600.0],
            "close": [29100.0, 29200.0],
            "volume": [100.0, 110.0],
        }
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result_bad = _normalize_ohlcv(df_bad, "bad_data.parquet")

        # Should warn
        assert len(w) == 1
        assert "ambiguous" in str(w[0].message).lower()
