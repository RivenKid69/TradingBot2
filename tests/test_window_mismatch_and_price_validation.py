"""
Comprehensive tests for two critical fixes:

1. Window Mismatch Fix (transformers.py):
   - Validates that window conversion from minutes to bars emits warnings
   - Tests for non-divisible windows (e.g., 1000 minutes with 240-minute bars)
   - Verifies actual window lengths match expected after conversion

2. Price Validation Fix (feature_pipe.py, transformers.py):
   - Validates protection against non-positive prices in log-return calculations
   - Tests NaN/inf/zero/negative price handling
   - Ensures no -inf/NaN injection into training targets

References:
- CRITICAL_FIXES_COMPLETE_REPORT.md
- NUMERICAL_ISSUES_FIX_SUMMARY.md
"""

import math
import warnings
import numpy as np
import pandas as pd
import pytest

from transformers import FeatureSpec, OnlineFeatureTransformer
from feature_pipe import FeaturePipe
from core_models import Bar


# ============================================================================
# Test Suite 1: Window Mismatch Validation
# ============================================================================


class TestWindowMismatchValidation:
    """Test window conversion validation from minutes to bars."""

    def test_divisible_windows_no_warning(self):
        """Divisible windows should not emit warnings."""
        # For 4h bars (240 minutes), these windows are perfectly divisible
        divisible_windows = [240, 480, 720, 1200, 1440, 5040, 10080, 12000]

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            spec = FeatureSpec(lookbacks_prices=divisible_windows, bar_duration_minutes=240)

        # Verify conversion is correct
        expected_bars = [1, 2, 3, 5, 6, 21, 42, 50]
        assert spec.lookbacks_prices == expected_bars
        assert spec._lookbacks_prices_minutes == divisible_windows

    def test_non_divisible_window_emits_warning(self):
        """Non-divisible window should emit UserWarning."""
        # 1000 minutes is not divisible by 240 (4h)
        # 1000 // 240 = 4 bars = 960 minutes (40 minutes discrepancy!)
        non_divisible_windows = [1000]

        with pytest.warns(UserWarning, match="not divisible"):
            spec = FeatureSpec(lookbacks_prices=non_divisible_windows, bar_duration_minutes=240)

        # Verify actual window is 960 minutes (4 bars), not 1000
        assert spec.lookbacks_prices == [4]  # 4 bars
        assert spec._lookbacks_prices_minutes == [1000]  # Original request

        # Calculate actual window in minutes
        actual_minutes = spec.lookbacks_prices[0] * 240
        assert actual_minutes == 960  # Not 1000!

    def test_warning_message_format(self):
        """Verify warning message contains all critical information."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec = FeatureSpec(lookbacks_prices=[1000], bar_duration_minutes=240)

            assert len(w) == 1
            warning_msg = str(w[0].message)

            # Check warning contains key information
            assert "1000" in warning_msg  # Requested window
            assert "960" in warning_msg  # Actual window
            assert "4" in warning_msg  # Number of bars
            assert "240" in warning_msg  # Bar duration
            assert "4.00%" in warning_msg  # Discrepancy percentage (40/1000 = 4%)

    def test_multiple_windows_mixed_divisibility(self):
        """Test mixed divisible and non-divisible windows."""
        mixed_windows = [240, 1000, 720, 1500, 1440]
        # Divisible: 240, 720, 1440
        # Non-divisible: 1000, 1500

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec = FeatureSpec(lookbacks_prices=mixed_windows, bar_duration_minutes=240)

            # Should emit 2 warnings (for 1000 and 1500)
            assert len(w) == 2
            warning_messages = [str(warning.message) for warning in w]
            assert any("1000" in msg for msg in warning_messages)
            assert any("1500" in msg for msg in warning_messages)

    def test_all_window_types_validation(self):
        """Test validation applies to all window types (not just lookbacks_prices)."""
        # All these should emit warnings for non-divisible windows
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec = FeatureSpec(
                lookbacks_prices=[1000],  # Non-divisible
                yang_zhang_windows=[2500],  # Non-divisible (2500 // 240 = 10 bars = 2400 min)
                parkinson_windows=[3000],  # Divisible (3000 // 240 = 12.5 → 12 bars = 2880 min)
                garch_windows=[11000],  # Non-divisible (11000 // 240 = 45.8 → 45 bars = 10800 min)
                taker_buy_ratio_windows=[500],  # Non-divisible (500 // 240 = 2 bars = 480 min)
                taker_buy_ratio_momentum=[350],  # Non-divisible (350 // 240 = 1 bar = 240 min)
                cvd_windows=[1500],  # Non-divisible (1500 // 240 = 6 bars = 1440 min)
                bar_duration_minutes=240,
            )

            # Should have 7 warnings (one for each non-divisible window)
            assert len(w) == 7

    def test_extreme_discrepancy_warning(self):
        """Test warning for extreme discrepancies (e.g., 239 min → 0 bars → 240 min)."""
        # 239 minutes with 240-minute bars
        # 239 // 240 = 0 → max(1, 0) = 1 bar = 240 minutes
        # Discrepancy: 1 minute (but percentage is tiny)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec = FeatureSpec(lookbacks_prices=[239], bar_duration_minutes=240)

            assert len(w) == 1
            assert spec.lookbacks_prices == [1]  # 1 bar minimum
            assert spec._lookbacks_prices_minutes == [239]

            # Verify actual window is 240 minutes (not 239)
            actual_minutes = spec.lookbacks_prices[0] * 240
            assert actual_minutes == 240

    def test_1m_timeframe_no_warnings(self):
        """For 1m timeframe, all minute-based windows should be divisible."""
        # For 1m bars (bar_duration_minutes=1), any minute-based window is divisible
        windows = [5, 10, 15, 30, 60, 120, 240, 1440]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            spec = FeatureSpec(lookbacks_prices=windows, bar_duration_minutes=1)

        # All windows should convert exactly
        assert spec.lookbacks_prices == windows
        assert spec._lookbacks_prices_minutes == windows


# ============================================================================
# Test Suite 2: Price Validation for Log Returns
# ============================================================================


class TestPriceValidationLogReturns:
    """Test price validation for safe log-return computation."""

    def test_valid_prices_no_nan(self):
        """Valid positive prices should produce finite log returns."""
        df = pd.DataFrame(
            {
                "symbol": ["BTC"] * 10,
                "price": [100.0, 105.0, 110.0, 108.0, 112.0, 115.0, 113.0, 118.0, 120.0, 122.0],
                "ts_ms": list(range(10)),
            }
        )

        spec = FeatureSpec(lookbacks_prices=[1], bar_duration_minutes=1)
        pipe = FeaturePipe(spec=spec, price_col="price")
        targets = pipe.make_targets(df)

        # All targets should be finite (no NaN, no inf)
        assert targets.notna().sum() == 9  # Last one is NaN (no future price)
        assert np.isfinite(targets[:-1]).all()

    def test_zero_price_produces_nan(self):
        """Zero prices should produce NaN in targets."""
        df = pd.DataFrame(
            {
                "symbol": ["BTC"] * 5,
                "price": [100.0, 0.0, 110.0, 120.0, 130.0],  # Zero price at index 1
                "ts_ms": list(range(5)),
            }
        )

        spec = FeatureSpec(lookbacks_prices=[1], bar_duration_minutes=1)
        pipe = FeaturePipe(spec=spec, price_col="price")
        targets = pipe.make_targets(df)

        # Target at index 0 should be NaN (future_price is 0)
        # Target at index 1 should be NaN (current price is 0)
        assert pd.isna(targets.iloc[0])  # ln(0 / 100) → NaN
        assert pd.isna(targets.iloc[1])  # ln(110 / 0) → NaN

        # Other targets should be finite
        assert np.isfinite(targets.iloc[2])  # ln(120 / 110) → finite
        assert np.isfinite(targets.iloc[3])  # ln(130 / 120) → finite

    def test_negative_price_produces_nan(self):
        """Negative prices should produce NaN in targets."""
        df = pd.DataFrame(
            {
                "symbol": ["BTC"] * 5,
                "price": [100.0, -50.0, 110.0, 120.0, 130.0],  # Negative price at index 1
                "ts_ms": list(range(5)),
            }
        )

        spec = FeatureSpec(lookbacks_prices=[1], bar_duration_minutes=1)
        pipe = FeaturePipe(spec=spec, price_col="price")
        targets = pipe.make_targets(df)

        # Targets involving negative price should be NaN
        assert pd.isna(targets.iloc[0])  # ln(-50 / 100) → NaN
        assert pd.isna(targets.iloc[1])  # ln(110 / -50) → NaN

        # Other targets should be finite
        assert np.isfinite(targets.iloc[2])
        assert np.isfinite(targets.iloc[3])

    def test_nan_price_produces_nan(self):
        """NaN prices should produce NaN in targets (not crash)."""
        df = pd.DataFrame(
            {
                "symbol": ["BTC"] * 5,
                "price": [100.0, np.nan, 110.0, 120.0, 130.0],  # NaN price at index 1
                "ts_ms": list(range(5)),
            }
        )

        spec = FeatureSpec(lookbacks_prices=[1], bar_duration_minutes=1)
        pipe = FeaturePipe(spec=spec, price_col="price")
        targets = pipe.make_targets(df)

        # Targets involving NaN price should be NaN
        assert pd.isna(targets.iloc[0])  # ln(NaN / 100) → NaN
        assert pd.isna(targets.iloc[1])  # ln(110 / NaN) → NaN

        # Other targets should be finite
        assert np.isfinite(targets.iloc[2])
        assert np.isfinite(targets.iloc[3])

    def test_inf_price_produces_nan(self):
        """Inf prices should produce NaN in targets."""
        df = pd.DataFrame(
            {
                "symbol": ["BTC"] * 5,
                "price": [100.0, np.inf, 110.0, 120.0, 130.0],  # Inf price at index 1
                "ts_ms": list(range(5)),
            }
        )

        spec = FeatureSpec(lookbacks_prices=[1], bar_duration_minutes=1)
        pipe = FeaturePipe(spec=spec, price_col="price")
        targets = pipe.make_targets(df)

        # Targets involving inf price should be NaN
        assert pd.isna(targets.iloc[0])  # ln(inf / 100) → NaN
        assert pd.isna(targets.iloc[1])  # ln(110 / inf) → NaN

        # Other targets should be finite
        assert np.isfinite(targets.iloc[2])
        assert np.isfinite(targets.iloc[3])

    def test_no_inf_in_targets(self):
        """Ensure no -inf values in targets (critical for training stability)."""
        # Create scenario that WOULD produce -inf without validation:
        # ln(0 / 100) = ln(0) = -inf
        df = pd.DataFrame(
            {
                "symbol": ["BTC"] * 5,
                "price": [100.0, 0.0, 110.0, 120.0, 130.0],
                "ts_ms": list(range(5)),
            }
        )

        spec = FeatureSpec(lookbacks_prices=[1], bar_duration_minutes=1)
        pipe = FeaturePipe(spec=spec, price_col="price")
        targets = pipe.make_targets(df)

        # CRITICAL: No -inf values should exist (all should be NaN instead)
        assert not np.any(np.isinf(targets))

        # Verify NaN count is correct
        nan_count = targets.isna().sum()
        assert nan_count >= 2  # At least 2 NaN (zero price + last row)


# ============================================================================
# Test Suite 3: Online Transformer Price Validation
# ============================================================================


class TestOnlineTransformerPriceValidation:
    """Test price validation in OnlineFeatureTransformer.update()."""

    def test_valid_prices_produce_finite_returns(self):
        """Valid prices should produce finite log returns."""
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)
        transformer = OnlineFeatureTransformer(spec)

        # Update with two valid prices
        feats1 = transformer.update(symbol="BTC", ts_ms=1000, close=100.0)
        feats2 = transformer.update(symbol="BTC", ts_ms=2000, close=105.0)

        # Second update should have ret_4h feature
        assert "ret_4h" in feats2
        ret = feats2["ret_4h"]

        # Should be finite log return: ln(105/100) = 0.04879...
        assert math.isfinite(ret)
        assert abs(ret - math.log(105.0 / 100.0)) < 1e-6

    def test_zero_old_price_produces_nan(self):
        """Zero old price should produce NaN (not 0.0)."""
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)
        transformer = OnlineFeatureTransformer(spec)

        # Update with zero price first
        feats1 = transformer.update(symbol="BTC", ts_ms=1000, close=0.0)
        feats2 = transformer.update(symbol="BTC", ts_ms=2000, close=105.0)

        # Should produce NaN (not 0.0, not -inf)
        assert "ret_4h" in feats2
        assert math.isnan(feats2["ret_4h"])

    def test_zero_current_price_produces_nan(self):
        """Zero current price should produce NaN."""
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)
        transformer = OnlineFeatureTransformer(spec)

        # Update with valid price first
        feats1 = transformer.update(symbol="BTC", ts_ms=1000, close=100.0)
        # Then update with zero current price
        feats2 = transformer.update(symbol="BTC", ts_ms=2000, close=0.0)

        # Should produce NaN
        assert "ret_4h" in feats2
        assert math.isnan(feats2["ret_4h"])

    def test_negative_prices_produce_nan(self):
        """Negative prices should produce NaN."""
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)
        transformer = OnlineFeatureTransformer(spec)

        # Test negative old price
        feats1 = transformer.update(symbol="BTC", ts_ms=1000, close=-100.0)
        feats2 = transformer.update(symbol="BTC", ts_ms=2000, close=105.0)
        assert "ret_4h" in feats2
        assert math.isnan(feats2["ret_4h"])

        # Reset and test negative current price
        transformer = OnlineFeatureTransformer(spec)
        feats1 = transformer.update(symbol="BTC", ts_ms=1000, close=100.0)
        feats2 = transformer.update(symbol="BTC", ts_ms=2000, close=-105.0)
        assert "ret_4h" in feats2
        assert math.isnan(feats2["ret_4h"])

    def test_no_inf_in_online_returns(self):
        """Ensure no -inf values in online returns."""
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)
        transformer = OnlineFeatureTransformer(spec)

        # Scenario that would produce -inf: ln(0 / 100) = -inf
        feats1 = transformer.update(symbol="BTC", ts_ms=1000, close=100.0)
        feats2 = transformer.update(symbol="BTC", ts_ms=2000, close=0.0)

        # Should be NaN, not -inf
        assert "ret_4h" in feats2
        assert math.isnan(feats2["ret_4h"])
        assert not math.isinf(feats2["ret_4h"])


# ============================================================================
# Integration Test: Full Pipeline
# ============================================================================


class TestFullPipelineIntegration:
    """Integration test: window validation + price validation."""

    def test_full_pipeline_with_edge_cases(self):
        """Test full pipeline with both non-divisible windows and invalid prices."""
        # Use non-divisible windows (should emit warnings)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec = FeatureSpec(
                lookbacks_prices=[1000],  # Non-divisible: 1000 // 240 = 4 bars = 960 min
                bar_duration_minutes=240,
            )

            # Should emit 1 warning for non-divisible window
            assert len(w) >= 1

        # Create DataFrame with edge case prices
        df = pd.DataFrame(
            {
                "symbol": ["BTC"] * 10,
                "price": [
                    100.0,  # Valid
                    0.0,  # Zero (invalid for log)
                    110.0,  # Valid
                    -50.0,  # Negative (invalid)
                    120.0,  # Valid
                    np.nan,  # NaN
                    130.0,  # Valid
                    np.inf,  # Inf (invalid)
                    140.0,  # Valid
                    150.0,  # Valid
                ],
                "ts_ms": list(range(10)),
            }
        )

        pipe = FeaturePipe(spec=spec, price_col="price")
        targets = pipe.make_targets(df)

        # Verify no -inf in targets (critical!)
        assert not np.any(np.isinf(targets))

        # Verify NaN for invalid price pairs
        # Target calculation: ln(price[i+1] / price[i])
        assert pd.isna(targets.iloc[0])  # ln(0 / 100) → NaN (future_price is 0)
        assert pd.isna(targets.iloc[1])  # ln(110 / 0) → NaN (current price is 0)
        assert pd.isna(targets.iloc[2])  # ln(-50 / 110) → NaN (future_price is negative)
        assert pd.isna(targets.iloc[3])  # ln(120 / -50) → NaN (current price is negative)
        assert pd.isna(targets.iloc[4])  # ln(NaN / 120) → NaN (future_price is NaN)
        assert pd.isna(targets.iloc[5])  # ln(130 / NaN) → NaN (current price is NaN)
        assert pd.isna(targets.iloc[6])  # ln(inf / 130) → NaN (future_price is inf)
        assert pd.isna(targets.iloc[7])  # ln(140 / inf) → NaN (current price is inf)

        # Verify finite for valid price pairs
        assert np.isfinite(targets.iloc[8])  # ln(150 / 140) → finite (both valid)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
