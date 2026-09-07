#!/usr/bin/env python3
"""
test_close_orig_enhancement.py
==================================================================
Comprehensive test suite for close_orig preservation enhancement (2025-11-25).

ISSUE: Original unshifted close prices were lost after feature pipeline transformation
- After transform_df(), close is shifted by 1 period (correct for data leakage prevention)
- But original close price $P_t$ was not preserved
- Downstream analysis and debugging needed access to original prices

ENHANCEMENT: Added preserve_close_orig parameter to FeaturePipeline
- When True, creates 'close_orig' column with unshifted prices before shifting
- Default False (not needed for standard ML pipeline)
- Useful for post-training analysis, debugging, and comparison

References:
- Issue reported: 2025-11-25
- Enhanced in: features_pipeline.py FeaturePipeline.__init__(), transform_df()
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

from features_pipeline import FeaturePipeline


class TestCloseOrigPreservation:
    """Test that close_orig is correctly preserved when requested."""

    def test_close_orig_created_when_enabled(self):
        """Test that close_orig is created when preserve_close_orig=True."""
        df = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000, 4000, 5000],
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            }
        )

        # Enable close_orig preservation
        pipe = FeaturePipeline(preserve_close_orig=True)
        pipe.fit({"test": df})
        df_transformed = pipe.transform_df(df.copy())

        # close_orig should exist
        assert (
            "close_orig" in df_transformed.columns
        ), "close_orig should be created when preserve_close_orig=True"

        # close_orig should contain original unshifted values
        assert np.allclose(
            df_transformed["close_orig"].values, [100.0, 101.0, 102.0, 103.0, 104.0]
        ), "close_orig should contain original unshifted close prices"

        # close should be shifted
        shifted_expected = [np.nan, 100.0, 101.0, 102.0, 103.0]
        assert np.allclose(
            df_transformed["close"].values[~np.isnan(df_transformed["close"].values)],
            np.array(shifted_expected)[~np.isnan(shifted_expected)],
        ), "close should be shifted by 1 period"

    def test_close_orig_not_created_when_disabled(self):
        """Test that close_orig is NOT created when preserve_close_orig=False (explicit)."""
        df = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000],
                "close": [100.0, 101.0, 102.0],
            }
        )

        # Explicit disable (preserve_close_orig=False)
        # NOTE (2025-11-25): Default changed to True for TradingEnv reward calculation
        pipe = FeaturePipeline(preserve_close_orig=False)
        pipe.fit({"test": df})
        df_transformed = pipe.transform_df(df.copy())

        # close_orig should NOT exist when explicitly disabled
        assert (
            "close_orig" not in df_transformed.columns
        ), "close_orig should NOT be created when preserve_close_orig=False"

    def test_close_orig_preserved_across_multiple_transforms(self):
        """Test that close_orig prevents repeated shifting when strict_idempotency=False."""
        df = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000],
                "close": [100.0, 101.0, 102.0],
            }
        )

        # Use strict_idempotency=False to allow repeated transforms
        # (for testing purposes - in production, strict=True is recommended)
        pipe = FeaturePipeline(preserve_close_orig=True, strict_idempotency=False)
        pipe.fit({"test": df})

        # First transform
        df_transformed_1 = pipe.transform_df(df.copy())

        # Second transform on already-transformed data
        # With strict_idempotency=False and close_orig present, should not double-shift
        df_transformed_2 = pipe.transform_df(df_transformed_1.copy())

        # close_orig should remain unchanged
        assert np.allclose(
            df_transformed_1["close_orig"].values, df_transformed_2["close_orig"].values
        ), "close_orig should remain unchanged across multiple transforms"

        # close should also remain unchanged (no double-shift)
        assert np.allclose(
            df_transformed_1["close"].values[~np.isnan(df_transformed_1["close"].values)],
            df_transformed_2["close"].values[~np.isnan(df_transformed_2["close"].values)],
        ), "close should not be double-shifted when close_orig exists"

    def test_close_orig_per_symbol_shift(self):
        """Test that close_orig works correctly with per-symbol shifting."""
        df = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000, 1000, 2000, 3000],
                "symbol": ["BTC", "BTC", "BTC", "ETH", "ETH", "ETH"],
                "close": [100.0, 110.0, 105.0, 200.0, 210.0, 205.0],
            }
        )

        pipe = FeaturePipeline(preserve_close_orig=True)
        pipe.fit({"test": df})
        df_transformed = pipe.transform_df(df.copy())

        # Check close_orig for BTC
        btc_rows = df_transformed[df_transformed["symbol"] == "BTC"]
        assert np.allclose(
            btc_rows["close_orig"].values, [100.0, 110.0, 105.0]
        ), "close_orig should preserve original BTC prices"

        # Check close_orig for ETH
        eth_rows = df_transformed[df_transformed["symbol"] == "ETH"]
        assert np.allclose(
            eth_rows["close_orig"].values, [200.0, 210.0, 205.0]
        ), "close_orig should preserve original ETH prices"

        # Check that close is shifted per-symbol
        assert pd.isna(btc_rows["close"].iloc[0]), "First BTC row should have NaN close"
        assert pd.isna(eth_rows["close"].iloc[0]), "First ETH row should have NaN close"

        assert np.allclose(
            btc_rows["close"].values[1:], [100.0, 110.0]
        ), "BTC close should be shifted"
        assert np.allclose(
            eth_rows["close"].values[1:], [200.0, 210.0]
        ), "ETH close should be shifted"

    def test_close_orig_with_analysis_use_case(self):
        """Test that close_orig enables post-training analysis."""
        df = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000, 4000, 5000],
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            }
        )

        pipe = FeaturePipeline(preserve_close_orig=True)
        pipe.fit({"test": df})
        df_transformed = pipe.transform_df(df.copy())

        # Use case: Calculate actual returns using close_orig
        # At t=1, we have close[t=1]=100 (shifted from t=0)
        # and close_orig[t=1]=101 (actual close at t=1)
        # Actual return[t=1] = (close_orig[t=1] - close[t=1]) / close[t=1]
        #                     = (101 - 100) / 100 = 0.01 (1%)

        # Calculate returns
        df_transformed["actual_return"] = (
            df_transformed["close_orig"] - df_transformed["close"]
        ) / df_transformed["close"]

        # Check that returns are calculated correctly
        expected_returns = [
            np.nan,  # First row has NaN close (shifted)
            0.01,  # (101-100)/100 = 0.01
            0.0099,  # (102-101)/101 ≈ 0.0099
            0.0098,  # (103-102)/102 ≈ 0.0098
            0.0097,  # (104-103)/103 ≈ 0.0097
        ]

        for i in range(1, len(expected_returns)):
            assert (
                abs(df_transformed["actual_return"].iloc[i] - expected_returns[i]) < 0.0001
            ), f"Return at index {i} should be ~{expected_returns[i]:.4f}, got {df_transformed['actual_return'].iloc[i]:.4f}"


class TestSaveLoadPreserveCloseOrig:
    """Test that preserve_close_orig flag is saved and loaded correctly."""

    def test_save_load_with_preserve_close_orig(self):
        """Test that preserve_close_orig=True is saved and loaded."""
        df = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000],
                "close": [100.0, 101.0, 102.0],
            }
        )

        pipe = FeaturePipeline(preserve_close_orig=True)
        pipe.fit({"test": df})

        # Save
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            pipe.save(temp_path)

            # Load
            pipe_loaded = FeaturePipeline.load(temp_path)

            # Check that preserve_close_orig was preserved
            assert (
                pipe_loaded.preserve_close_orig is True
            ), "preserve_close_orig should be loaded as True"

            # Test that loaded pipeline creates close_orig
            df_transformed = pipe_loaded.transform_df(df.copy())
            assert (
                "close_orig" in df_transformed.columns
            ), "Loaded pipeline should create close_orig when preserve_close_orig=True"

        finally:
            os.unlink(temp_path)

    def test_save_load_without_preserve_close_orig(self):
        """Test that preserve_close_orig=False (default) is saved and loaded."""
        df = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000],
                "close": [100.0, 101.0, 102.0],
            }
        )

        pipe = FeaturePipeline(preserve_close_orig=False)
        pipe.fit({"test": df})

        # Save
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            pipe.save(temp_path)

            # Load
            pipe_loaded = FeaturePipeline.load(temp_path)

            # Check that preserve_close_orig was preserved
            assert (
                pipe_loaded.preserve_close_orig is False
            ), "preserve_close_orig should be loaded as False"

            # Test that loaded pipeline does NOT create close_orig
            df_transformed = pipe_loaded.transform_df(df.copy())
            assert (
                "close_orig" not in df_transformed.columns
            ), "Loaded pipeline should NOT create close_orig when preserve_close_orig=False"

        finally:
            os.unlink(temp_path)

    def test_backward_compatibility_legacy_artifacts(self):
        """Test that legacy artifacts (without preserve_close_orig) still work."""
        # Simulate legacy artifact (JSON with stats but no preserve_close_orig in config)
        legacy_json = {
            "stats": {"close": {"mean": 100.0, "std": 5.0, "is_constant": False}},
            "metadata": {},
            "config": {
                "enable_winsorization": True,
                "winsorize_percentiles": [1.0, 99.0],
                "strict_idempotency": True,
                # NO "preserve_close_orig" key (legacy)
            },
        }

        import json

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(legacy_json, f)
            temp_path = f.name

        try:
            # Load legacy artifact
            pipe = FeaturePipeline.load(temp_path)

            # FIX (2025-11-25): Changed default from False to True to match constructor
            # Legacy artifacts now default to True for correct TradingEnv reward calculation
            assert (
                pipe.preserve_close_orig is True
            ), "Legacy artifacts should default preserve_close_orig to True (fixed 2025-11-25)"

            # Should work without errors
            df = pd.DataFrame(
                {
                    "timestamp": [1000, 2000, 3000],
                    "close": [100.0, 110.0, 90.0],
                }
            )

            df_transformed = pipe.transform_df(df.copy())
            assert (
                "close_z" in df_transformed.columns
            ), "Legacy pipeline should still normalize features"

            # With new default, close_orig should now be created
            assert (
                "close_orig" in df_transformed.columns
            ), "Legacy artifacts with new default should create close_orig"

        finally:
            os.unlink(temp_path)


class TestDocumentationAndUseCases:
    """Test that documentation examples work correctly."""

    def test_use_case_post_training_analysis(self):
        """Test documented use case: post-training analysis."""
        # Scenario: User wants to analyze model predictions vs actual prices
        # Need both shifted features (for model input) and original prices (for comparison)

        df = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000, 4000, 5000],
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "rsi_14": [50.0, 55.0, 60.0, 65.0, 70.0],
            }
        )

        # Enable close_orig for analysis
        pipe = FeaturePipeline(preserve_close_orig=True)
        pipe.fit({"test": df})
        df_transformed = pipe.transform_df(df.copy())

        # Model would use shifted features
        # At t=1: close=100 (from t=0), rsi=50 (from t=0) → model predicts for t=1
        # Actual outcome at t=1: close_orig=101

        # Calculate prediction error
        # (In real use case, model.predict() would be used here)
        df_transformed["predicted_close"] = df_transformed["close"] * 1.01  # Dummy prediction
        df_transformed["error"] = (
            df_transformed["close_orig"] - df_transformed["predicted_close"]
        ).abs()

        # Should be able to analyze errors
        assert "error" in df_transformed.columns
        assert (
            not df_transformed["error"].dropna().empty
        ), "Should be able to calculate prediction errors"

    def test_use_case_debugging_data_leakage(self):
        """Test use case: debugging data leakage."""
        # Scenario: User suspects data leakage and wants to verify temporal alignment

        df = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000],
                "close": [100.0, 101.0, 102.0],
                "rsi_14": [50.0, 55.0, 60.0],
            }
        )

        pipe = FeaturePipeline(preserve_close_orig=True)
        pipe.fit({"test": df})
        df_transformed = pipe.transform_df(df.copy())

        # Verify temporal alignment
        # At t=1, model sees close[t-1] and rsi[t-1], predicts for t=1 (close_orig)
        # close[t=1]=100 should be < close_orig[t=1]=101 (price increased)

        t1_close = df_transformed["close"].iloc[1]  # 100 (from t=0)
        t1_close_orig = df_transformed["close_orig"].iloc[1]  # 101 (actual t=1)
        t1_rsi = df_transformed["rsi_14"].iloc[1]  # 50 (from t=0)

        assert (
            t1_close < t1_close_orig
        ), "Shifted close should be from previous period (less than current)"

        # This verification would catch data leakage if close was not shifted
        # (close would equal close_orig, indicating no shift applied)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
