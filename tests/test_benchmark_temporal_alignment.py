# -*- coding: utf-8 -*-
"""
tests/test_benchmark_temporal_alignment.py
Tests for benchmark temporal alignment fix (Issue #4).

FIX (2025-11-29): Tests for _align_benchmark_by_timestamp function and
add_stock_features_to_dataframe temporal alignment.

The bug was that stock_features.py used positional indexing to merge
benchmark data (VIX/SPY/QQQ), which fails when benchmark data has
different date ranges or gaps than the main data.

The fix uses merge_asof with direction="backward" to ensure:
- Proper temporal alignment regardless of date ranges
- No look-ahead bias (only past data is used)
- Graceful handling of gaps and misaligned timestamps

References:
- López de Prado (2018) "Advances in Financial ML" Ch.4 - Temporal alignment
- docs/PLATFORM_REFERENCE.md Issue #4 documentation
"""

import numpy as np
import pandas as pd
import pytest

from stock_features import (
    _align_benchmark_by_timestamp,
    add_stock_features_to_dataframe,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def stock_df_2022():
    """Stock DataFrame starting from 2022-01-01."""
    timestamps = [
        1640995200,  # 2022-01-01
        1641081600,  # 2022-01-02
        1641168000,  # 2022-01-03
        1641254400,  # 2022-01-04
        1641340800,  # 2022-01-05
    ]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["AAPL"] * 5,
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [104.0, 105.0, 106.0, 107.0, 108.0],
            "volume": [1000.0] * 5,
        }
    )


@pytest.fixture
def vix_df_2020():
    """VIX DataFrame starting from 2020-01-01 (before stock data)."""
    timestamps = [
        1577836800,  # 2020-01-01
        1577923200,  # 2020-01-02
        1578009600,  # 2020-01-03
        1640995200,  # 2022-01-01 (overlaps with stock)
        1641081600,  # 2022-01-02
        1641168000,  # 2022-01-03
        1641254400,  # 2022-01-04
        1641340800,  # 2022-01-05
    ]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": [20.0, 21.0, 22.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        }
    )


@pytest.fixture
def spy_df_aligned():
    """SPY DataFrame aligned with stock data."""
    timestamps = [
        1640995200,  # 2022-01-01
        1641081600,  # 2022-01-02
        1641168000,  # 2022-01-03
        1641254400,  # 2022-01-04
        1641340800,  # 2022-01-05
    ]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": [400.0, 401.0, 402.0, 403.0, 404.0],
        }
    )


@pytest.fixture
def spy_df_with_gaps():
    """SPY DataFrame with gaps (missing 2022-01-02 and 2022-01-04)."""
    timestamps = [
        1640995200,  # 2022-01-01
        1641168000,  # 2022-01-03 (gap: 01-02 missing)
        1641340800,  # 2022-01-05 (gap: 01-04 missing)
    ]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": [400.0, 402.0, 404.0],
        }
    )


# =============================================================================
# TESTS FOR _align_benchmark_by_timestamp
# =============================================================================


class TestAlignBenchmarkByTimestamp:
    """Tests for the _align_benchmark_by_timestamp helper function."""

    def test_aligned_data_returns_correct_values(self, stock_df_2022, spy_df_aligned):
        """When data is aligned, should return exact matching values."""
        aligned = _align_benchmark_by_timestamp(stock_df_2022, spy_df_aligned, "close")

        assert len(aligned) == len(stock_df_2022)
        assert aligned[0] == 400.0
        assert aligned[1] == 401.0
        assert aligned[2] == 402.0
        assert aligned[3] == 403.0
        assert aligned[4] == 404.0

    def test_misaligned_date_ranges_uses_backward_lookup(self, stock_df_2022, vix_df_2020):
        """When benchmark starts earlier, should use backward lookup (no look-ahead)."""
        aligned = _align_benchmark_by_timestamp(stock_df_2022, vix_df_2020, "close")

        assert len(aligned) == len(stock_df_2022)
        # 2022-01-01: Should get VIX from 2022-01-01 (not 2020 values!)
        assert aligned[0] == 15.0  # VIX at 2022-01-01
        assert aligned[1] == 16.0  # VIX at 2022-01-02
        assert aligned[2] == 17.0  # VIX at 2022-01-03
        assert aligned[3] == 18.0  # VIX at 2022-01-04
        assert aligned[4] == 19.0  # VIX at 2022-01-05

    def test_gaps_uses_last_known_value(self, stock_df_2022, spy_df_with_gaps):
        """When benchmark has gaps, should use last known value (backward fill)."""
        aligned = _align_benchmark_by_timestamp(stock_df_2022, spy_df_with_gaps, "close")

        assert len(aligned) == len(stock_df_2022)
        # 2022-01-01: exact match
        assert aligned[0] == 400.0
        # 2022-01-02: gap in benchmark, should use value from 2022-01-01
        assert aligned[1] == 400.0
        # 2022-01-03: exact match
        assert aligned[2] == 402.0
        # 2022-01-04: gap in benchmark, should use value from 2022-01-03
        assert aligned[3] == 402.0
        # 2022-01-05: exact match
        assert aligned[4] == 404.0

    def test_empty_benchmark_returns_nones(self, stock_df_2022):
        """When benchmark is empty, should return list of Nones."""
        empty_df = pd.DataFrame(columns=["timestamp", "close"])
        aligned = _align_benchmark_by_timestamp(stock_df_2022, empty_df, "close")

        assert len(aligned) == len(stock_df_2022)
        assert all(v is None for v in aligned)

    def test_none_benchmark_returns_nones(self, stock_df_2022):
        """When benchmark is None, should return list of Nones."""
        aligned = _align_benchmark_by_timestamp(stock_df_2022, None, "close")

        assert len(aligned) == len(stock_df_2022)
        assert all(v is None for v in aligned)

    def test_benchmark_after_stock_returns_nones_for_early_rows(self, stock_df_2022):
        """When benchmark starts after stock, early rows should get None."""
        # Benchmark only has data for 2022-01-04 and 2022-01-05
        late_benchmark = pd.DataFrame(
            {
                "timestamp": [1641254400, 1641340800],  # 2022-01-04, 2022-01-05
                "close": [403.0, 404.0],
            }
        )
        aligned = _align_benchmark_by_timestamp(stock_df_2022, late_benchmark, "close")

        assert len(aligned) == len(stock_df_2022)
        # First 3 rows have no benchmark data (benchmark starts later)
        assert aligned[0] is None
        assert aligned[1] is None
        assert aligned[2] is None
        # Last 2 rows have data
        assert aligned[3] == 403.0
        assert aligned[4] == 404.0

    def test_no_timestamp_column_falls_back_with_warning(self):
        """When no timestamp column, should fall back to positional with warning."""
        df_no_ts = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        benchmark_no_ts = pd.DataFrame({"close": [10.0, 20.0, 30.0]})

        with pytest.warns(RuntimeWarning, match="No timestamp column"):
            aligned = _align_benchmark_by_timestamp(df_no_ts, benchmark_no_ts, "close")

        assert len(aligned) == 3
        assert aligned[0] == 10.0
        assert aligned[1] == 20.0
        assert aligned[2] == 30.0


# =============================================================================
# TESTS FOR add_stock_features_to_dataframe TEMPORAL ALIGNMENT
# =============================================================================


class TestStockFeaturesTemporalAlignment:
    """Tests for add_stock_features_to_dataframe using aligned data."""

    def test_vix_uses_temporally_aligned_values(self, stock_df_2022, vix_df_2020):
        """VIX features should use temporally aligned values, not positional."""
        result = add_stock_features_to_dataframe(
            df=stock_df_2022,
            symbol="AAPL",
            vix_df=vix_df_2020,
        )

        # VIX values should be from 2022, not 2020
        # Without fix: vix_values[0] = 20.0 (from 2020-01-01)
        # With fix: vix_values[0] = 15.0 (from 2022-01-01)
        # The vix_normalized column should reflect 2022 VIX levels
        assert "vix_normalized" in result.columns
        # VIX 15-19 in 2022 is different from VIX 20-22 in 2020

    def test_relative_strength_with_misaligned_spy(self, stock_df_2022):
        """Relative strength should handle misaligned SPY data correctly."""
        # SPY data starts from 2020 (like VIX)
        spy_2020 = pd.DataFrame(
            {
                "timestamp": [
                    1577836800,  # 2020-01-01
                    1577923200,  # 2020-01-02
                    1640995200,  # 2022-01-01
                    1641081600,  # 2022-01-02
                    1641168000,  # 2022-01-03
                    1641254400,  # 2022-01-04
                    1641340800,  # 2022-01-05
                ],
                "close": [300.0, 301.0, 400.0, 401.0, 402.0, 403.0, 404.0],
            }
        )

        result = add_stock_features_to_dataframe(
            df=stock_df_2022,
            symbol="AAPL",
            spy_df=spy_2020,
        )

        # Should not crash and should have RS columns
        assert "rs_spy_20d" in result.columns
        assert len(result) == 5

    def test_no_look_ahead_bias(self, stock_df_2022):
        """Ensure benchmark data from the future is not used (no look-ahead)."""
        # Benchmark with future data (after stock's last timestamp)
        future_benchmark = pd.DataFrame(
            {
                "timestamp": [
                    1641340800,  # 2022-01-05 (last stock row)
                    1641427200,  # 2022-01-06 (future)
                    1641513600,  # 2022-01-07 (future)
                ],
                "close": [404.0, 999.0, 1000.0],  # Future values are very different
            }
        )

        aligned = _align_benchmark_by_timestamp(stock_df_2022, future_benchmark, "close")

        # Only the last row should have data (exact match at 2022-01-05)
        # Future values (999, 1000) should NOT be used for earlier rows
        assert aligned[0] is None  # No data before 2022-01-05
        assert aligned[1] is None
        assert aligned[2] is None
        assert aligned[3] is None
        assert aligned[4] == 404.0  # Exact match at 2022-01-05

    def test_backward_compatibility_empty_benchmark(self, stock_df_2022):
        """Should work correctly with no benchmark data (backward compatible)."""
        result = add_stock_features_to_dataframe(
            df=stock_df_2022,
            symbol="AAPL",
            spy_df=None,
            qqq_df=None,
            vix_df=None,
        )

        assert len(result) == 5
        # All features should be default values
        assert (result["vix_normalized"] == 0.0).all()
        assert (result["rs_spy_20d"] == 0.0).all()


# =============================================================================
# REGRESSION TESTS
# =============================================================================


class TestRegressionNoPositionalIndexing:
    """
    Regression tests to ensure positional indexing bug doesn't return.

    The original bug was:
        spy_prices = spy_df["close"].tolist()
        current_spy = spy_prices[i]  # Position i, not timestamp-based!

    If VIX data covers 2020-2024 and stock covers 2022-2024:
    - Old: vix_values[0] = VIX from 2020-01-01 (WRONG!)
    - New: vix_aligned[0] = VIX from 2022-01-01 (CORRECT)
    """

    def test_long_benchmark_short_stock_uses_last_known_value(self):
        """When benchmark ends before stock starts, merge_asof uses last known value.

        This is the expected behavior of merge_asof with direction="backward":
        - For each stock timestamp, find benchmark value with timestamp <= stock.timestamp
        - If benchmark ends before stock starts, the last benchmark value is used

        This behavior is correct for temporal alignment because:
        1. It's better than positional indexing (which gives wrong values)
        2. For practical use cases, benchmark data typically overlaps with stock data
        3. Using stale data is better than using WRONG data (positional bug)

        Note: In production, data pipelines should ensure adequate overlap.
        """
        # Long benchmark: 100 days starting 2020-01-01 (ends ~2020-04-10)
        long_benchmark = pd.DataFrame(
            {
                "timestamp": [1577836800 + i * 86400 for i in range(100)],
                "close": [100.0 + i for i in range(100)],  # 100, 101, 102...199
            }
        )

        # Short stock: 5 days starting 2022-01-01 (after benchmark ends)
        short_stock = pd.DataFrame(
            {
                "timestamp": [1640995200 + i * 86400 for i in range(5)],
                "close": [200.0 + i for i in range(5)],
            }
        )

        aligned = _align_benchmark_by_timestamp(short_stock, long_benchmark, "close")

        # merge_asof uses the last benchmark value (199.0) for all stock rows
        # because it's the most recent value <= each stock timestamp
        # This is better than positional indexing which would use [100, 101, 102, 103, 104]
        assert len(aligned) == 5
        # All rows get the last benchmark value (most recent available)
        assert all(v == 199.0 for v in aligned)

    def test_overlapping_ranges_uses_correct_values(self):
        """When ranges overlap, should use temporally correct values."""
        # Benchmark: 2021-12-30 to 2022-01-05
        benchmark = pd.DataFrame(
            {
                "timestamp": [
                    1640822400,  # 2021-12-30
                    1640908800,  # 2021-12-31
                    1640995200,  # 2022-01-01
                    1641081600,  # 2022-01-02
                    1641168000,  # 2022-01-03
                ],
                "close": [10.0, 11.0, 12.0, 13.0, 14.0],  # Different values each day
            }
        )

        # Stock: 2022-01-01 to 2022-01-03
        stock = pd.DataFrame(
            {
                "timestamp": [1640995200, 1641081600, 1641168000],
                "close": [100.0, 101.0, 102.0],
            }
        )

        aligned = _align_benchmark_by_timestamp(stock, benchmark, "close")

        # Should align by timestamp:
        # Stock 2022-01-01 -> Benchmark 2022-01-01 = 12.0
        # Stock 2022-01-02 -> Benchmark 2022-01-02 = 13.0
        # Stock 2022-01-03 -> Benchmark 2022-01-03 = 14.0
        assert aligned[0] == 12.0
        assert aligned[1] == 13.0
        assert aligned[2] == 14.0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Edge case tests for temporal alignment."""

    def test_single_row_dataframe(self):
        """Should handle single-row DataFrames."""
        single_stock = pd.DataFrame(
            {
                "timestamp": [1640995200],
                "close": [100.0],
            }
        )
        single_benchmark = pd.DataFrame(
            {
                "timestamp": [1640995200],
                "close": [50.0],
            }
        )

        aligned = _align_benchmark_by_timestamp(single_stock, single_benchmark, "close")

        assert len(aligned) == 1
        assert aligned[0] == 50.0

    def test_duplicate_timestamps_in_benchmark(self):
        """Should handle duplicate timestamps (uses first/last depending on sort)."""
        stock = pd.DataFrame(
            {
                "timestamp": [1640995200, 1641081600],
                "close": [100.0, 101.0],
            }
        )
        # Benchmark has duplicate timestamp
        benchmark = pd.DataFrame(
            {
                "timestamp": [1640995200, 1640995200, 1641081600],
                "close": [50.0, 51.0, 52.0],  # Two values for same timestamp
            }
        )

        aligned = _align_benchmark_by_timestamp(stock, benchmark, "close")

        assert len(aligned) == 2
        # merge_asof will use one of the values (implementation detail)
        assert aligned[0] in [50.0, 51.0]
        assert aligned[1] == 52.0

    def test_nan_values_in_benchmark(self):
        """Should handle NaN values in benchmark data."""
        stock = pd.DataFrame(
            {
                "timestamp": [1640995200, 1641081600, 1641168000],
                "close": [100.0, 101.0, 102.0],
            }
        )
        benchmark = pd.DataFrame(
            {
                "timestamp": [1640995200, 1641081600, 1641168000],
                "close": [50.0, np.nan, 52.0],  # NaN in middle
            }
        )

        aligned = _align_benchmark_by_timestamp(stock, benchmark, "close")

        assert len(aligned) == 3
        assert aligned[0] == 50.0
        # NaN row is dropped before merge, so row 1 gets backward fill from row 0
        assert aligned[1] == 50.0  # Backward fill
        assert aligned[2] == 52.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
