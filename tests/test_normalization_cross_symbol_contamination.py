# -*- coding: utf-8 -*-
"""
Test for normalization cross-symbol contamination fix.

Verifies that when normalizing features across multiple symbols, the shift()
operation is applied per-symbol to prevent the last row of Symbol1 from
contaminating the first row of Symbol2.
"""
import pandas as pd
import numpy as np
import pytest

from features_pipeline import FeaturePipeline


def test_fit_per_symbol_shift_no_contamination():
    """
    Verify that shift() is applied per-symbol during fit() to prevent
    cross-symbol contamination.
    """
    # Create test data for two symbols with distinct values
    # Symbol1: close prices [100, 110, 120]
    # Symbol2: close prices [200, 210, 220]
    df1 = pd.DataFrame(
        {
            "timestamp": [0, 60000, 120000],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "close": [100.0, 110.0, 120.0],
            "volume": [1000.0, 1100.0, 1200.0],
        }
    )

    df2 = pd.DataFrame(
        {
            "timestamp": [0, 60000, 120000],
            "symbol": ["ETHUSDT", "ETHUSDT", "ETHUSDT"],
            "close": [200.0, 210.0, 220.0],
            "volume": [2000.0, 2100.0, 2200.0],
        }
    )

    dfs = {"BTCUSDT": df1, "ETHUSDT": df2}

    # Fit the pipeline
    pipe = FeaturePipeline()
    pipe.fit(dfs)

    # After per-symbol shift, the close values should be:
    # BTCUSDT: [NaN, 100, 110] (shift within symbol)
    # ETHUSDT: [NaN, 200, 210] (shift within symbol)
    #
    # If shift was applied AFTER concat (buggy behavior), we would get:
    # BTCUSDT: [NaN, 100, 110]
    # ETHUSDT: [120, 200, 210]  # ❌ First value contaminated by last BTC value!

    # The mean should exclude NaN values
    # Expected: mean([100, 110, 200, 210]) = 155.0
    # Buggy:    mean([100, 110, 120, 200, 210]) = 148.0

    close_stats = pipe.stats.get("close")
    assert close_stats is not None, "close column should have stats"

    # With per-symbol shift, we should get mean = 155.0
    expected_mean = (100.0 + 110.0 + 200.0 + 210.0) / 4.0
    assert abs(close_stats["mean"] - expected_mean) < 1e-6, (
        f"Expected mean={expected_mean}, got {close_stats['mean']}. "
        "This suggests cross-symbol contamination during shift."
    )


def test_transform_per_symbol_shift_no_contamination():
    """
    Verify that shift() is applied per-symbol during transform_df() when
    a 'symbol' column exists.
    """
    # Create test data
    df = pd.DataFrame(
        {
            "timestamp": [0, 60000, 120000, 0, 60000, 120000],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT", "ETHUSDT"],
            "close": [100.0, 110.0, 120.0, 200.0, 210.0, 220.0],
            "volume": [1000.0, 1100.0, 1200.0, 2000.0, 2100.0, 2200.0],
        }
    )

    # Fit pipeline (this will also apply per-symbol shift internally)
    pipe = FeaturePipeline()
    train_dfs = {
        "BTCUSDT": df[df["symbol"] == "BTCUSDT"].copy(),
        "ETHUSDT": df[df["symbol"] == "ETHUSDT"].copy(),
    }
    pipe.fit(train_dfs)

    # Transform the combined dataframe
    transformed = pipe.transform_df(df)

    # Check that the first row of ETHUSDT (index 3) does NOT contain
    # contaminated data from the last row of BTCUSDT
    # After per-symbol shift:
    # - Row 3 (first ETHUSDT) should have NaN for shifted close
    # - If buggy (global shift), row 3 would have close=120.0 (last BTCUSDT value)

    assert pd.isna(transformed.loc[0, "close"]), "First row of BTCUSDT should have NaN after shift"
    assert pd.isna(
        transformed.loc[3, "close"]
    ), "First row of ETHUSDT should have NaN after shift, not contaminated by BTCUSDT"

    # Check that subsequent rows have correct shifted values
    assert (
        abs(transformed.loc[1, "close"] - 100.0) < 1e-6
    ), "Second row of BTCUSDT should have shifted close=100.0"
    assert (
        abs(transformed.loc[4, "close"] - 200.0) < 1e-6
    ), "Second row of ETHUSDT should have shifted close=200.0"


def test_transform_single_symbol_no_symbol_column():
    """
    Verify that transform_df() works correctly when there's no 'symbol' column
    (single symbol case).
    """
    # Create test data without symbol column
    df = pd.DataFrame(
        {
            "timestamp": [0, 60000, 120000],
            "close": [100.0, 110.0, 120.0],
            "volume": [1000.0, 1100.0, 1200.0],
        }
    )

    # Fit pipeline
    pipe = FeaturePipeline()
    pipe.fit({"BTCUSDT": df})

    # Transform
    transformed = pipe.transform_df(df)

    # First row should be NaN after shift
    assert pd.isna(transformed.loc[0, "close"]), "First row should have NaN after shift"

    # Subsequent rows should have shifted values
    assert abs(transformed.loc[1, "close"] - 100.0) < 1e-6
    assert abs(transformed.loc[2, "close"] - 110.0) < 1e-6


def test_fit_statistics_correctness():
    """
    Verify that computed statistics are correct with per-symbol shift.
    """
    # Create data where cross-symbol contamination would be obvious
    df1 = pd.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "symbol": ["SYM1", "SYM1", "SYM1"],
            "close": [10.0, 20.0, 30.0],  # After shift: [NaN, 10, 20]
            "volume": [100.0, 200.0, 300.0],
        }
    )

    df2 = pd.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "symbol": ["SYM2", "SYM2", "SYM2"],
            "close": [1000.0, 2000.0, 3000.0],  # After shift: [NaN, 1000, 2000]
            "volume": [10000.0, 20000.0, 30000.0],
        }
    )

    dfs = {"SYM1": df1, "SYM2": df2}

    pipe = FeaturePipeline()
    pipe.fit(dfs)

    # With per-symbol shift:
    # close values (excluding NaN): [10, 20, 1000, 2000]
    # mean = (10 + 20 + 1000 + 2000) / 4 = 757.5
    # std = sqrt(sum((x - mean)^2) / n) with ddof=0

    close_stats = pipe.stats.get("close")
    assert close_stats is not None

    expected_mean = (10.0 + 20.0 + 1000.0 + 2000.0) / 4.0
    values = np.array([10.0, 20.0, 1000.0, 2000.0])
    expected_std = np.std(values, ddof=0)

    assert (
        abs(close_stats["mean"] - expected_mean) < 1e-6
    ), f"Expected mean={expected_mean}, got {close_stats['mean']}"
    assert (
        abs(close_stats["std"] - expected_std) < 1e-6
    ), f"Expected std={expected_std}, got {close_stats['std']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
