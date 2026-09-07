#!/usr/bin/env python3
"""
test_timestamp_consistency_fix.py
==================================================================
Comprehensive test suite for timestamp consistency fix (2025-11-25).

ISSUE: Timestamp inconsistency between CSV and Parquet data sources
- CSV (_read_raw): Used floor(close_time / 14400) * 14400 → OPEN TIME
- Parquet (_normalize_ohlcv): Used close_time directly → CLOSE TIME
- Result: 4-hour offset preventing data merge

FIX: Unified timestamp semantics to CLOSE TIME consistently
- CSV now uses close_time directly (no floor division)
- Parquet continues using close_time
- Both sources now aligned for successful merge/concat

References:
- Issue reported: 2025-11-25
- Fixed in: prepare_and_run.py _read_raw()
- Test files: test_reported_issues.py
"""
import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from prepare_and_run import _read_raw, _normalize_ohlcv


class TestTimestampConsistency:
    """Test that timestamp semantics are consistent across data sources."""

    def test_csv_uses_close_time_not_floor(self):
        """Test that CSV path uses close_time directly (CLOSE TIME), not floor division."""
        # Create CSV with Binance 4h candle (00:00-04:00)
        # close_time = 14399 (03:59:59 - Binance convention: end - 1ms)
        csv_temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
        csv_data = """open_time,close_time,open,high,low,close,volume,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume
0,14399,100.0,105.0,99.0,102.0,1000.0,102000.0,500,600.0,61200.0
14400,28799,102.0,107.0,101.0,105.0,1100.0,115500.0,550,650.0,68250.0
"""
        csv_temp.write(csv_data)
        csv_temp.close()

        try:
            df = _read_raw(csv_temp.name)

            # After fix: timestamp should be close_time (CLOSE TIME)
            # OLD (WRONG): timestamp = (14399 // 14400) * 14400 = 0 (OPEN TIME)
            # NEW (CORRECT): timestamp = 14399 (CLOSE TIME)
            assert (
                df["timestamp"].iloc[0] == 14399
            ), f"CSV timestamp should be close_time (14399), got {df['timestamp'].iloc[0]}"

            # Second bar
            assert (
                df["timestamp"].iloc[1] == 28799
            ), f"CSV timestamp should be close_time (28799), got {df['timestamp'].iloc[1]}"

        finally:
            os.unlink(csv_temp.name)

    def test_parquet_close_time_unchanged(self):
        """Test that Parquet path with close_time column remains unchanged."""
        # Parquet with explicit close_time
        df_parquet = pd.DataFrame(
            {
                "close_time": [14399, 28799],
                "open": [100.0, 102.0],
                "high": [105.0, 107.0],
                "low": [99.0, 101.0],
                "close": [102.0, 105.0],
                "volume": [1000.0, 1100.0],
            }
        )

        df_normalized = _normalize_ohlcv(df_parquet, "test.parquet")

        # Parquet should use close_time directly (CLOSE TIME)
        assert (
            df_normalized["timestamp"].iloc[0] == 14399
        ), f"Parquet timestamp should be close_time (14399), got {df_normalized['timestamp'].iloc[0]}"
        assert df_normalized["timestamp"].iloc[1] == 28799

    def test_parquet_open_time_consistency(self):
        """Test that Parquet path with open_time adds bar duration."""
        # Parquet with explicit open_time
        df_parquet = pd.DataFrame(
            {
                "open_time": [0, 14400],
                "open": [100.0, 102.0],
                "high": [105.0, 107.0],
                "low": [99.0, 101.0],
                "close": [102.0, 105.0],
                "volume": [1000.0, 1100.0],
            }
        )

        df_normalized = _normalize_ohlcv(df_parquet, "test.parquet")

        # Parquet with open_time: timestamp = open_time + 14400 (CLOSE TIME)
        # First bar: 0 + 14400 = 14400
        # NOTE: This differs from Binance close_time by 1 second (14400 vs 14399)
        # but is consistent within Parquet processing
        assert (
            df_normalized["timestamp"].iloc[0] == 14400
        ), f"Parquet timestamp should be open_time + 14400 (14400), got {df_normalized['timestamp'].iloc[0]}"
        assert df_normalized["timestamp"].iloc[1] == 28800

    def test_csv_parquet_timestamp_alignment(self):
        """Test that CSV and Parquet timestamps are now aligned (within 1 second tolerance)."""
        # CSV data
        csv_temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
        csv_data = """open_time,close_time,open,high,low,close,volume,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume
0,14399,100.0,105.0,99.0,102.0,1000.0,102000.0,500,600.0,61200.0
"""
        csv_temp.write(csv_data)
        csv_temp.close()

        # Parquet data (close_time)
        df_parquet_close = pd.DataFrame(
            {
                "close_time": [14399],
                "open": [100.0],
                "high": [105.0],
                "low": [99.0],
                "close": [102.0],
                "volume": [1000.0],
            }
        )

        # Parquet data (open_time)
        df_parquet_open = pd.DataFrame(
            {
                "open_time": [0],
                "open": [100.0],
                "high": [105.0],
                "low": [99.0],
                "close": [102.0],
                "volume": [1000.0],
            }
        )

        try:
            df_csv = _read_raw(csv_temp.name)
            df_parquet_close_norm = _normalize_ohlcv(df_parquet_close, "test_close.parquet")
            df_parquet_open_norm = _normalize_ohlcv(df_parquet_open, "test_open.parquet")

            ts_csv = df_csv["timestamp"].iloc[0]
            ts_parquet_close = df_parquet_close_norm["timestamp"].iloc[0]
            ts_parquet_open = df_parquet_open_norm["timestamp"].iloc[0]

            # CSV and Parquet(close_time) should be identical
            assert (
                ts_csv == ts_parquet_close
            ), f"CSV and Parquet(close_time) timestamps should match: {ts_csv} vs {ts_parquet_close}"

            # CSV and Parquet(open_time) should be within 1 second
            # (Binance close_time is end-1ms, open_time+duration is exact end)
            diff = abs(ts_csv - ts_parquet_open)
            assert (
                diff <= 1
            ), f"CSV and Parquet(open_time) timestamps should be within 1 second: diff={diff}"

            # OLD BEHAVIOR (before fix): diff would be 14399 seconds (4 hours)
            # NEW BEHAVIOR (after fix): diff is 0-1 seconds (aligned)

        finally:
            os.unlink(csv_temp.name)

    def test_merge_asof_works_after_fix(self):
        """Test that merge_asof works correctly after timestamp fix."""
        # Main data (CSV)
        csv_temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
        csv_data = """open_time,close_time,open,high,low,close,volume,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume
0,14399,100.0,105.0,99.0,102.0,1000.0,102000.0,500,600.0,61200.0
14400,28799,102.0,107.0,101.0,105.0,1100.0,115500.0,550,650.0,68250.0
"""
        csv_temp.write(csv_data)
        csv_temp.close()

        # Additional data (Parquet - e.g., Fear & Greed)
        df_additional = pd.DataFrame(
            {
                "timestamp": [14399, 28799],
                "fear_greed_value": [50, 60],
            }
        )

        try:
            df_main = _read_raw(csv_temp.name)

            # This should work without errors after fix
            df_merged = pd.merge_asof(
                df_main.sort_values("timestamp"),
                df_additional.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )

            # Check that merge was successful
            assert (
                "fear_greed_value" in df_merged.columns
            ), "merge_asof should successfully add additional columns"

            # Check that values are correctly aligned
            assert (
                df_merged["fear_greed_value"].iloc[0] == 50
            ), f"First row should have fear_greed=50, got {df_merged['fear_greed_value'].iloc[0]}"
            assert df_merged["fear_greed_value"].iloc[1] == 60

        finally:
            os.unlink(csv_temp.name)

    def test_no_4hour_offset_in_features(self):
        """Test that features from different sources don't have 4-hour temporal misalignment."""
        # Before fix: CSV timestamp would be 4 hours earlier than Parquet
        # This would cause features to be misaligned when merging

        csv_temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
        csv_data = """open_time,close_time,open,high,low,close,volume,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume
0,14399,100.0,105.0,99.0,102.0,1000.0,102000.0,500,600.0,61200.0
"""
        csv_temp.write(csv_data)
        csv_temp.close()

        df_parquet = pd.DataFrame(
            {
                "close_time": [14399],
                "open": [100.0],
                "high": [105.0],
                "low": [99.0],
                "close": [102.0],
                "volume": [1000.0],
                "rsi_14": [55.0],  # Additional feature from Parquet
            }
        )

        try:
            df_csv = _read_raw(csv_temp.name)
            df_parquet_norm = _normalize_ohlcv(df_parquet, "test.parquet")

            # Add RSI feature to Parquet
            df_parquet_norm["rsi_14"] = [55.0]

            # Concat (simulating combined dataset)
            df_combined = pd.concat(
                [df_csv[["timestamp", "close"]], df_parquet_norm[["timestamp", "rsi_14"]]]
            )

            # Group by timestamp to check if rows align
            grouped = df_combined.groupby("timestamp").size()

            # After fix: should have 2 rows with same timestamp (aligned)
            # Before fix: would have 2 different timestamps (4-hour offset)
            assert (
                len(grouped) == 1
            ), f"Should have 1 unique timestamp after fix, got {len(grouped)}"

        finally:
            os.unlink(csv_temp.name)


class TestBackwardCompatibility:
    """Test that changes don't break existing functionality."""

    def test_multiple_bars_processing(self):
        """Test that processing multiple bars works correctly."""
        csv_temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
        csv_data = """open_time,close_time,open,high,low,close,volume,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume
0,14399,100.0,105.0,99.0,102.0,1000.0,102000.0,500,600.0,61200.0
14400,28799,102.0,107.0,101.0,105.0,1100.0,115500.0,550,650.0,68250.0
28800,43199,105.0,110.0,104.0,108.0,1200.0,129600.0,600,700.0,75600.0
"""
        csv_temp.write(csv_data)
        csv_temp.close()

        try:
            df = _read_raw(csv_temp.name)

            # Check that all timestamps are correct
            assert df["timestamp"].iloc[0] == 14399
            assert df["timestamp"].iloc[1] == 28799
            assert df["timestamp"].iloc[2] == 43199

            # Check that timestamps are monotonic increasing
            assert df[
                "timestamp"
            ].is_monotonic_increasing, "Timestamps should be monotonic increasing"

            # Check that no duplicates
            assert df["timestamp"].nunique() == len(df), "All timestamps should be unique"

        finally:
            os.unlink(csv_temp.name)

    def test_ms_to_seconds_conversion(self):
        """Test that millisecond timestamps are correctly converted to seconds."""
        csv_temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
        # close_time in milliseconds (realistic values > 10 billion)
        # Example: 1609459199000 ms = 1609459199 s = 2021-01-01 03:59:59 UTC
        csv_data = """open_time,close_time,open,high,low,close,volume,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume
1609444800000,1609459199000,100.0,105.0,99.0,102.0,1000.0,102000.0,500,600.0,61200.0
"""
        csv_temp.write(csv_data)
        csv_temp.close()

        try:
            df = _read_raw(csv_temp.name)

            # Should convert to seconds
            # 1609459199000 ms / 1000 = 1609459199 s
            assert (
                df["timestamp"].iloc[0] == 1609459199
            ), f"Milliseconds should be converted to seconds: expected 1609459199, got {df['timestamp'].iloc[0]}"

        finally:
            os.unlink(csv_temp.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
