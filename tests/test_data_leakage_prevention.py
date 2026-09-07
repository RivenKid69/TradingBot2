# -*- coding: utf-8 -*-
"""
test_data_leakage_prevention.py
==================================================================
Comprehensive test suite for data leakage prevention in features_pipeline.py

CRITICAL FIX (2025-11-23): Ensures all feature columns (prices, volumes, technical
indicators) are shifted by 1 period to prevent look-ahead bias.

Test Coverage:
1. All feature columns shifted correctly (not just close)
2. Metadata columns NOT shifted (timestamp, symbol)
3. Target columns NOT shifted (labels)
4. Technical indicators temporally aligned with prices
5. No future information leakage
6. Consistent shifting in fit() and transform_df()
7. Multi-symbol shifting (per-symbol grouping)
"""
import numpy as np
import pandas as pd
import pytest
from features_pipeline import FeaturePipeline, _columns_to_shift


# ==============================================================================
# Test 1: _columns_to_shift() correctly identifies feature columns
# ==============================================================================


def test_columns_to_shift_basic():
    """Test that _columns_to_shift() correctly identifies feature columns."""
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "close": [100.0, 101.0, 102.0],
            "rsi_14": [50.0, 55.0, 60.0],
            "target": [0.01, 0.02, 0.03],
        }
    )

    cols = _columns_to_shift(df)

    # Should include feature columns (close, rsi_14)
    assert "close" in cols, "close should be shifted"
    assert "rsi_14" in cols, "rsi_14 should be shifted"

    # Should exclude metadata and targets
    assert "timestamp" not in cols, "timestamp should NOT be shifted (metadata)"
    assert "symbol" not in cols, "symbol should NOT be shifted (metadata)"
    assert "target" not in cols, "target should NOT be shifted (label)"


def test_columns_to_shift_all_technical_indicators():
    """Test that ALL technical indicators are identified for shifting."""
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "close": [100.0, 101.0, 102.0],
            "volume": [1000.0, 1100.0, 1200.0],
            # Technical indicators that MUST be shifted
            "rsi_14": [50.0, 55.0, 60.0],
            "macd": [0.5, 0.6, 0.7],
            "macd_signal": [0.4, 0.5, 0.6],
            "bb_upper": [105.0, 106.0, 107.0],
            "bb_lower": [95.0, 96.0, 97.0],
            "atr_14": [2.0, 2.1, 2.2],
            "adx_14": [25.0, 26.0, 27.0],
            "ema_20": [99.0, 100.0, 101.0],
            # Metadata - should NOT be shifted
            "wf_role": ["train", "train", "val"],
            # Target - should NOT be shifted
            "target": [0.01, 0.02, 0.03],
        }
    )

    cols = _columns_to_shift(df)

    # All feature columns should be included
    expected_features = [
        "close",
        "volume",
        "rsi_14",
        "macd",
        "macd_signal",
        "bb_upper",
        "bb_lower",
        "atr_14",
        "adx_14",
        "ema_20",
    ]
    for feature in expected_features:
        assert feature in cols, f"{feature} should be shifted (technical indicator)"

    # Metadata and targets should be excluded
    assert "timestamp" not in cols
    assert "wf_role" not in cols
    assert "target" not in cols


def test_columns_to_shift_excludes_normalized():
    """Test that already-normalized columns (_z suffix) are excluded."""
    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0],
            "close_z": [0.0, 0.5, 1.0],  # Already normalized - skip
            "rsi_14": [50.0, 55.0, 60.0],
            "rsi_14_z": [0.0, 0.3, 0.6],  # Already normalized - skip
        }
    )

    cols = _columns_to_shift(df)

    # Original columns should be included
    assert "close" in cols
    assert "rsi_14" in cols

    # Normalized columns should be excluded
    assert "close_z" not in cols, "close_z should NOT be shifted (will be recomputed)"
    assert "rsi_14_z" not in cols, "rsi_14_z should NOT be shifted (will be recomputed)"


# ==============================================================================
# Test 2: fit() shifts ALL feature columns (not just close)
# ==============================================================================


def test_fit_shifts_all_features():
    """Test that fit() shifts ALL feature columns, not just close."""
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3, 4, 5],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "rsi_14": [50.0, 55.0, 60.0, 65.0, 70.0],
            "macd": [0.5, 0.6, 0.7, 0.8, 0.9],
        }
    )

    pipe = FeaturePipeline()
    pipe.fit({"BTCUSDT": df})

    # Check that stats were computed on SHIFTED data
    # Original data: close = [100, 101, 102, 103, 104]
    # Shifted data: close = [NaN, 100, 101, 102, 103]
    # Mean should be ~101.5 (mean of [100, 101, 102, 103])
    assert "close" in pipe.stats
    close_mean = pipe.stats["close"]["mean"]
    # Allow small tolerance for floating point
    assert (
        101.0 <= close_mean <= 102.0
    ), f"close mean {close_mean} suggests data was NOT shifted (expected ~101.5)"

    # Check RSI also computed on shifted data
    assert "rsi_14" in pipe.stats
    rsi_mean = pipe.stats["rsi_14"]["mean"]
    # Original: [50, 55, 60, 65, 70] → mean = 60
    # Shifted: [NaN, 50, 55, 60, 65] → mean = 57.5
    assert (
        56.0 <= rsi_mean <= 59.0
    ), f"rsi_14 mean {rsi_mean} suggests data was NOT shifted (expected ~57.5)"


def test_fit_first_row_becomes_nan():
    """Test that first row becomes NaN after shift (correct behavior)."""
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "close": [100.0, 101.0, 102.0],
            "rsi_14": [50.0, 55.0, 60.0],
        }
    )

    pipe = FeaturePipeline()
    # We can't directly inspect shifted data in fit(), but we can verify
    # that stats exclude first row (which becomes NaN after shift)

    # Stats should be computed on 2 rows (rows 1-2 after shift)
    pipe.fit({"BTCUSDT": df})

    # If all rows were used, close mean would be 101.0
    # If first row excluded (NaN after shift), mean should be 101.5
    close_mean = pipe.stats["close"]["mean"]
    # Mean of [100, 101] = 100.5 (2nd and 3rd original rows → 1st and 2nd after shift)
    assert (
        100.0 <= close_mean <= 101.0
    ), f"close mean {close_mean} suggests first row was NOT excluded"


# ==============================================================================
# Test 3: transform_df() shifts ALL feature columns (consistency with fit)
# ==============================================================================


def test_transform_shifts_all_features():
    """Test that transform_df() shifts ALL feature columns."""
    train_df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3, 4, 5],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "rsi_14": [50.0, 55.0, 60.0, 65.0, 70.0],
        }
    )

    test_df = pd.DataFrame(
        {
            "timestamp": [6, 7, 8],
            "close": [105.0, 106.0, 107.0],
            "rsi_14": [75.0, 80.0, 85.0],
        }
    )

    pipe = FeaturePipeline()
    pipe.fit({"BTCUSDT": train_df})
    transformed = pipe.transform_df(test_df)

    # First row should have NaN after shift
    assert pd.isna(transformed["close"].iloc[0]), "First close should be NaN after shift"
    assert pd.isna(transformed["rsi_14"].iloc[0]), "First rsi_14 should be NaN after shift"

    # Second row should have values from first row (shift by 1)
    assert transformed["close"].iloc[1] == 105.0, "close[1] should be original close[0]"
    assert transformed["rsi_14"].iloc[1] == 75.0, "rsi_14[1] should be original rsi_14[0]"

    # Third row should have values from second row
    assert transformed["close"].iloc[2] == 106.0, "close[2] should be original close[1]"
    assert transformed["rsi_14"].iloc[2] == 80.0, "rsi_14[2] should be original rsi_14[1]"


def test_transform_metadata_not_shifted():
    """Test that metadata columns (timestamp, symbol) are NOT shifted."""
    train_df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "close": [100.0, 101.0, 102.0],
        }
    )

    test_df = pd.DataFrame(
        {
            "timestamp": [4, 5, 6],
            "close": [103.0, 104.0, 105.0],
        }
    )

    pipe = FeaturePipeline()
    pipe.fit({"BTCUSDT": train_df})
    transformed = pipe.transform_df(test_df)

    # Timestamp should NOT be shifted
    assert transformed["timestamp"].iloc[0] == 4, "timestamp should NOT be shifted"
    assert transformed["timestamp"].iloc[1] == 5
    assert transformed["timestamp"].iloc[2] == 6

    # But close SHOULD be shifted
    assert pd.isna(transformed["close"].iloc[0]), "close should be shifted"
    assert transformed["close"].iloc[1] == 103.0


# ==============================================================================
# Test 4: No data leakage - future information NOT accessible
# ==============================================================================


def test_no_data_leakage_temporal_alignment():
    """
    CRITICAL TEST: Verify that technical indicators at time t reflect
    information available BEFORE time t (no future leakage).
    """
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3, 4, 5],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            # Indicator calculated on close[t-13:t] (assumes 14-period RSI)
            # Original: rsi[t] based on close[t-13:t]
            # After shift: rsi[t] based on close[t-14:t-1] (correct - no future info)
            "rsi_14": [50.0, 55.0, 60.0, 65.0, 70.0],
        }
    )

    pipe = FeaturePipeline()
    pipe.fit({"BTCUSDT": df})
    transformed = pipe.transform_df(df.copy())

    # At time t=3 (index 2 after shift → index 1 originally):
    # - close[2] should be 102.0 (original close[1])
    # - rsi_14[2] should be 55.0 (original rsi_14[1])
    # - Both reflect information up to time t=2 (original), NOT t=3
    assert transformed["close"].iloc[2] == 101.0, "close[2] should be from t=1"
    assert transformed["rsi_14"].iloc[2] == 55.0, "rsi_14[2] should be from t=1"

    # CRITICAL: rsi_14[2] should NOT be 60.0 (which would reflect close up to t=2)
    # It should be 55.0 (reflecting close up to t=1)
    # This ensures decision at t=2 uses only information available BEFORE t=2
    assert (
        transformed["rsi_14"].iloc[2] != 60.0
    ), "DATA LEAKAGE DETECTED! rsi_14[2] reflects future information"


def test_no_leakage_indicators_consistent_with_prices():
    """
    Test that indicators and prices are temporally aligned (both shifted by 1).
    """
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3, 4, 5],
            "close": [100.0, 105.0, 110.0, 115.0, 120.0],  # Trending up
            "rsi_14": [50.0, 60.0, 70.0, 80.0, 90.0],  # RSI increasing with price
            "ema_20": [98.0, 103.0, 108.0, 113.0, 118.0],  # EMA lagging close
        }
    )

    pipe = FeaturePipeline()
    pipe.fit({"BTCUSDT": df})
    transformed = pipe.transform_df(df.copy())

    # Check that indicators are shifted together with prices
    # At index 2 (original time t=2):
    # - close[2] = 105.0 (from t=1)
    # - rsi_14[2] = 60.0 (from t=1, consistent with close=105)
    # - ema_20[2] = 103.0 (from t=1, lagging close=105)
    assert transformed["close"].iloc[2] == 105.0
    assert transformed["rsi_14"].iloc[2] == 60.0, "RSI should match shifted close"
    assert transformed["ema_20"].iloc[2] == 103.0, "EMA should match shifted close"

    # Verify temporal consistency: RSI and EMA reflect same time period as close
    # NOT future values (rsi_14=70, ema_20=108)
    assert transformed["rsi_14"].iloc[2] != 70.0, "RSI should NOT reflect future"
    assert transformed["ema_20"].iloc[2] != 108.0, "EMA should NOT reflect future"


# ==============================================================================
# Test 5: Multi-symbol shifting (per-symbol grouping)
# ==============================================================================


def test_multi_symbol_shift_no_contamination():
    """Test that shifting respects symbol boundaries (no cross-symbol contamination)."""
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3, 4, 5, 6],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT", "ETHUSDT"],
            "close": [100.0, 101.0, 102.0, 200.0, 201.0, 202.0],
            "rsi_14": [50.0, 55.0, 60.0, 40.0, 45.0, 50.0],
        }
    )

    pipe = FeaturePipeline()
    pipe.fit({"multi": df})
    transformed = pipe.transform_df(df.copy())

    # First row of each symbol should be NaN after shift
    assert pd.isna(transformed["close"].iloc[0]), "BTCUSDT first close should be NaN"
    assert pd.isna(transformed["close"].iloc[3]), "ETHUSDT first close should be NaN"

    # Second row of BTCUSDT should have first row values
    assert transformed["close"].iloc[1] == 100.0
    assert transformed["rsi_14"].iloc[1] == 50.0

    # Second row of ETHUSDT should have first ETHUSDT row values (NOT last BTCUSDT!)
    assert transformed["close"].iloc[4] == 200.0, "No cross-symbol contamination"
    assert transformed["rsi_14"].iloc[4] == 40.0, "No cross-symbol contamination"

    # CRITICAL: ETHUSDT[4] should NOT be 102.0 (last BTCUSDT value)
    assert transformed["close"].iloc[4] != 102.0, "CROSS-SYMBOL CONTAMINATION DETECTED!"


# ==============================================================================
# Test 6: Consistency between fit() and transform_df()
# ==============================================================================


def test_fit_transform_consistency():
    """Test that fit() and transform_df() use identical shifting logic."""
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3, 4, 5],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "rsi_14": [50.0, 55.0, 60.0, 65.0, 70.0],
        }
    )

    pipe = FeaturePipeline()
    pipe.fit({"BTCUSDT": df})

    # Transform same data used for fitting
    transformed = pipe.transform_df(df.copy())

    # Check that z-scores are reasonable (not huge outliers)
    # If shifting logic differs between fit/transform, z-scores will be garbage
    close_z = transformed["close_z"].dropna()
    assert close_z.abs().max() < 5.0, "Large z-scores suggest fit/transform shifting logic mismatch"

    rsi_z = transformed["rsi_14_z"].dropna()
    assert rsi_z.abs().max() < 5.0, "Large z-scores suggest fit/transform shifting logic mismatch"


# ==============================================================================
# Test 7: Edge cases
# ==============================================================================


def test_single_row_dataframe():
    """Test that single-row DataFrame is handled correctly (all NaN after shift)."""
    df = pd.DataFrame(
        {
            "timestamp": [1],
            "close": [100.0],
            "rsi_14": [50.0],
        }
    )

    # Single row → after shift all NaN → fit succeeds but marks columns as all-NaN
    pipe = FeaturePipeline()
    pipe.fit({"BTCUSDT": df})

    # Check that columns are marked as all-NaN
    assert "close" in pipe.stats
    assert pipe.stats["close"].get(
        "is_all_nan", False
    ), "close should be marked as all-NaN after shift"
    assert "rsi_14" in pipe.stats
    assert pipe.stats["rsi_14"].get(
        "is_all_nan", False
    ), "rsi_14 should be marked as all-NaN after shift"


def test_empty_dataframe():
    """Test that empty DataFrame is handled correctly."""
    df = pd.DataFrame(
        {
            "timestamp": [],
            "close": [],
            "rsi_14": [],
        }
    )

    pipe = FeaturePipeline()
    with pytest.raises(ValueError, match="No rows available"):
        pipe.fit({"BTCUSDT": df})


def test_only_metadata_columns():
    """Test DataFrame with only metadata columns (no features)."""
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
        }
    )

    pipe = FeaturePipeline()
    # Should not crash, but stats will be empty
    pipe.fit({"BTCUSDT": df})
    assert len(pipe.stats) == 0, "No stats for metadata-only DataFrame"


def test_close_orig_preserved():
    """
    Test that close_orig column (if present) prevents double-shifting.

    IMPORTANT: close_orig is a marker that data has already been shifted.
    When present, ALL features should remain unshifted to prevent double-shift.
    """
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "close": [100.0, 101.0, 102.0],
            "close_orig": [100.0, 101.0, 102.0],  # Marker that shift already applied
            "rsi_14": [50.0, 55.0, 60.0],
        }
    )

    pipe = FeaturePipeline()
    pipe.fit({"BTCUSDT": df})
    transformed = pipe.transform_df(df.copy())

    # When close_orig present, NO features should be shifted
    # (data already went through shift during previous pipeline pass)
    assert not pd.isna(
        transformed["close"].iloc[0]
    ), "close should NOT be shifted when close_orig present"
    assert transformed["close"].iloc[0] == 100.0, "close should preserve original values"

    assert not pd.isna(
        transformed["rsi_14"].iloc[0]
    ), "rsi_14 should NOT be shifted when close_orig present"
    assert transformed["rsi_14"].iloc[0] == 50.0, "rsi_14 should preserve original values"


# ==============================================================================
# Test 8: Integration test with real-world scenario
# ==============================================================================


def test_integration_realistic_features():
    """
    Integration test with realistic feature set (prices + many indicators).
    """
    np.random.seed(42)
    n = 100

    df = pd.DataFrame(
        {
            "timestamp": range(1, n + 1),
            "symbol": ["BTCUSDT"] * n,
            "open": 100.0 + np.cumsum(np.random.randn(n) * 0.5),
            "high": 101.0 + np.cumsum(np.random.randn(n) * 0.5),
            "low": 99.0 + np.cumsum(np.random.randn(n) * 0.5),
            "close": 100.0 + np.cumsum(np.random.randn(n) * 0.5),
            "volume": 1000.0 + np.random.rand(n) * 100,
            # Technical indicators
            "rsi_14": 50.0 + np.random.randn(n) * 10,
            "macd": np.random.randn(n) * 0.5,
            "macd_signal": np.random.randn(n) * 0.5,
            "bb_upper": 102.0 + np.cumsum(np.random.randn(n) * 0.5),
            "bb_lower": 98.0 + np.cumsum(np.random.randn(n) * 0.5),
            "atr_14": 2.0 + np.random.rand(n),
            "ema_20": 100.0 + np.cumsum(np.random.randn(n) * 0.5),
            "sma_50": 100.0 + np.cumsum(np.random.randn(n) * 0.3),
            # Target (should NOT be shifted)
            "target": np.random.randn(n) * 0.01,
        }
    )

    # Split train/test
    train_df = df.iloc[:80].copy()
    test_df = df.iloc[80:].copy()

    # Fit on train
    pipe = FeaturePipeline()
    pipe.fit({"BTCUSDT": train_df})

    # Transform test
    transformed = pipe.transform_df(test_df)

    # Verify ALL feature columns were shifted
    feature_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "rsi_14",
        "macd",
        "macd_signal",
        "bb_upper",
        "bb_lower",
        "atr_14",
        "ema_20",
        "sma_50",
    ]

    for col in feature_cols:
        # First row should be NaN after shift
        assert pd.isna(transformed[col].iloc[0]), f"{col} first row should be NaN after shift"

        # Second row should equal first row of original (shift by 1)
        assert (
            transformed[col].iloc[1] == test_df[col].iloc[0]
        ), f"{col} not properly shifted (temporal misalignment)"

    # Verify metadata NOT shifted
    assert (
        transformed["timestamp"].iloc[0] == test_df["timestamp"].iloc[0]
    ), "timestamp should NOT be shifted"
    assert (
        transformed["symbol"].iloc[0] == test_df["symbol"].iloc[0]
    ), "symbol should NOT be shifted"

    # Verify target NOT shifted (if present in stats - it shouldn't be)
    if "target" in transformed.columns:
        # Target should be preserved without shifting
        assert (
            transformed["target"].iloc[0] == test_df["target"].iloc[0]
        ), "target should NOT be shifted"


# ==============================================================================
# Test 9: Backwards compatibility
# ==============================================================================


def test_backwards_compatibility_old_behavior():
    """
    Test that old behavior (shift only close) is NO LONGER supported.
    This test documents the breaking change.
    """
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "close": [100.0, 101.0, 102.0],
            "rsi_14": [50.0, 55.0, 60.0],
        }
    )

    pipe = FeaturePipeline()
    pipe.fit({"BTCUSDT": df})
    transformed = pipe.transform_df(df.copy())

    # NEW BEHAVIOR: Both close AND rsi_14 are shifted
    assert pd.isna(transformed["close"].iloc[0])
    assert pd.isna(
        transformed["rsi_14"].iloc[0]
    ), "NEW BEHAVIOR: rsi_14 is now shifted (breaking change from old code)"

    # OLD BEHAVIOR (BUG): Only close was shifted, rsi_14 was NOT
    # This test would FAIL with old code:
    # assert transformed["rsi_14"].iloc[0] == 50.0  # OLD BUG

    # Document breaking change
    # Models trained with old pipeline MUST be retrained!


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
