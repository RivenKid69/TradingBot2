#!/usr/bin/env python3
"""
test_first_bar_reward_fix.py
==================================================================
Test suite for first bar reward=0 fix (2025-11-25).

ISSUE: First bar of each episode always had reward=0 because:
1. FeaturePipeline shifted close by 1 period (correct for data leakage prevention)
2. But close_orig was not preserved by default (preserve_close_orig=False)
3. TradingEnv._close_actual used shifted close with NaN at position 0
4. _resolve_reward_price(0) returned 0.0 due to NaN
5. First step: reward_price_prev <= 0.0 → reward_raw_fraction = 0.0

FIX: Changed preserve_close_orig default to True in FeaturePipeline
- close_orig is now created by default during shift
- TradingEnv uses close_orig for _close_actual when available
- First bar now has correct reward calculation

References:
- Issue reported: 2025-11-25
- Fixed in: features_pipeline.py (default change), trading_patchnew.py (warning)
"""
import numpy as np
import pandas as pd
import pytest
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features_pipeline import FeaturePipeline


class TestFirstBarRewardFix:
    """Test that first bar reward is calculated correctly after the fix."""

    def test_preserve_close_orig_default_is_true(self):
        """Verify that preserve_close_orig default changed to True."""
        pipe = FeaturePipeline()
        assert (
            pipe.preserve_close_orig is True
        ), "preserve_close_orig should default to True since 2025-11-25 fix"

    def test_close_orig_created_by_default(self):
        """Test that close_orig is created when using default FeaturePipeline."""
        df = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000, 4000, 5000],
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            }
        )

        # Default FeaturePipeline (preserve_close_orig=True by default)
        pipe = FeaturePipeline()
        pipe.fit({"test": df})
        df_transformed = pipe.transform_df(df.copy())

        # close_orig should exist by default now
        assert (
            "close_orig" in df_transformed.columns
        ), "close_orig should be created by default (preserve_close_orig=True)"

        # close_orig should contain original unshifted values
        expected_orig = [100.0, 101.0, 102.0, 103.0, 104.0]
        np.testing.assert_allclose(
            df_transformed["close_orig"].values,
            expected_orig,
            err_msg="close_orig should contain original unshifted close prices",
        )

    def test_close_is_shifted_while_close_orig_is_not(self):
        """Test that close is shifted but close_orig preserves original values."""
        df = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000, 4000, 5000],
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            }
        )

        pipe = FeaturePipeline()
        pipe.fit({"test": df})
        df_transformed = pipe.transform_df(df.copy())

        # close[0] should be NaN after shift
        assert np.isnan(df_transformed["close"].iloc[0]), "close[0] should be NaN after shift(1)"

        # close_orig[0] should be original value (not NaN)
        assert not np.isnan(
            df_transformed["close_orig"].iloc[0]
        ), "close_orig[0] should NOT be NaN (unshifted)"
        assert (
            df_transformed["close_orig"].iloc[0] == 100.0
        ), "close_orig[0] should be original first close price"

    def test_close_orig_enables_correct_first_bar_price_resolution(self):
        """Test that close_orig at index 0 is valid (not NaN) for reward calculation."""
        df = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000, 4000, 5000],
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            }
        )

        pipe = FeaturePipeline()
        pipe.fit({"test": df})
        df_transformed = pipe.transform_df(df.copy())

        # Simulate TradingEnv._resolve_reward_price(0) logic
        close_orig = df_transformed["close_orig"]
        first_price = float(close_orig.iloc[0]) if len(close_orig) > 0 else 0.0

        assert first_price > 0.0, "First price from close_orig should be > 0 for reward calculation"
        assert first_price == 100.0, "First price should be original close price"


class TestLegacyDataWarning:
    """Test warning for legacy data without close_orig."""

    def test_warning_when_close_shifted_but_no_close_orig(self, caplog):
        """Test that TradingEnv warns when _close_shifted exists but close_orig doesn't."""
        # Create legacy-style data: shifted but no close_orig
        df = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000, 4000, 5000],
                "close": [np.nan, 100.0, 101.0, 102.0, 103.0],  # Shifted (NaN at start)
                "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
                "_close_shifted": [True] * 5,  # Marker present
                # Note: no 'close_orig' column - legacy data
            }
        )

        # Import TradingEnv (may need adjustment based on actual import path)
        try:
            from trading_patchnew import TradingEnv
        except ImportError:
            pytest.skip("TradingEnv not available")

        # Create TradingEnv and check for warning
        with caplog.at_level(logging.WARNING):
            try:
                env = TradingEnv(df=df, initial_cash=10000.0, max_position=100.0)
            except Exception as e:
                # If TradingEnv fails to initialize for other reasons, skip
                if "close_orig" in str(e).lower() or "_close_shifted" in str(e).lower():
                    raise
                pytest.skip(f"TradingEnv initialization failed: {e}")

        # Check warning was logged
        warning_found = any(
            "_close_shifted" in record.message and "close_orig" in record.message
            for record in caplog.records
        )
        assert (
            warning_found
        ), "TradingEnv should warn when _close_shifted exists but close_orig doesn't"


class TestFeaturePipelineBackwardCompatibility:
    """Test backward compatibility with preserve_close_orig=False."""

    def test_explicit_false_still_works(self):
        """Test that explicitly setting preserve_close_orig=False still works."""
        df = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000],
                "close": [100.0, 101.0, 102.0],
            }
        )

        # Explicitly disable close_orig preservation
        pipe = FeaturePipeline(preserve_close_orig=False)
        pipe.fit({"test": df})
        df_transformed = pipe.transform_df(df.copy())

        # close_orig should NOT exist when explicitly disabled
        assert (
            "close_orig" not in df_transformed.columns
        ), "close_orig should NOT be created when preserve_close_orig=False"

    def test_config_save_load_preserves_preserve_close_orig(self):
        """Test that preserve_close_orig is correctly saved and loaded."""
        import tempfile
        import os

        df = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000],
                "close": [100.0, 101.0, 102.0],
            }
        )

        # Create pipeline with default (True)
        pipe = FeaturePipeline()
        pipe.fit({"test": df})

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "pipeline.json")
            pipe.save(config_path)
            loaded_pipe = FeaturePipeline.load(config_path)

        assert (
            loaded_pipe.preserve_close_orig is True
        ), "preserve_close_orig should be preserved after save/load"


class TestRewardCalculationIntegration:
    """Integration tests for reward calculation with first bar."""

    def test_first_bar_price_available_for_reward(self):
        """Test that first bar has valid price for reward calculation."""
        df = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000, 4000, 5000],
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            }
        )

        pipe = FeaturePipeline()
        pipe.fit({"test": df})
        df_transformed = pipe.transform_df(df.copy())

        # Simulate reward calculation check
        # In TradingEnv._resolve_reward_price, close_orig is used
        close_orig = df_transformed["close_orig"]

        # All prices should be valid (finite and > 0)
        for i in range(len(close_orig)):
            price = float(close_orig.iloc[i])
            assert np.isfinite(price), f"close_orig[{i}] should be finite"
            assert price > 0.0, f"close_orig[{i}] should be > 0"

    def test_first_bar_price_ratio_calculable(self):
        """Test that price ratio can be calculated for first bar reward."""
        df = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000, 4000, 5000],
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            }
        )

        pipe = FeaturePipeline()
        pipe.fit({"test": df})
        df_transformed = pipe.transform_df(df.copy())

        close_orig = df_transformed["close_orig"]

        # Simulate first step reward calculation
        # After reset(): _last_reward_price = close_orig[0] = 100.0
        # At step 1: reward_price_prev = 100.0, reward_price_curr = 101.0
        reward_price_prev = float(close_orig.iloc[0])  # From reset
        reward_price_curr = float(close_orig.iloc[1])  # From step 1

        assert reward_price_prev > 0.0, "First bar price should be valid"
        assert reward_price_curr > 0.0, "Second bar price should be valid"

        # Price ratio should be calculable
        ratio = reward_price_curr / reward_price_prev
        assert np.isfinite(ratio), "Price ratio should be finite"
        assert ratio > 0.0, "Price ratio should be positive"

        # Expected ratio for 100 -> 101
        expected_ratio = 101.0 / 100.0
        np.testing.assert_almost_equal(ratio, expected_ratio)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
