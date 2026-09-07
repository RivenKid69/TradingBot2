"""
Comprehensive tests for Bollinger Bands Position Symmetric Clipping Fix (2025-11-23)

Tests verify that bb_position uses symmetric [-1.0, 1.0] range instead of
asymmetric [-1.0, 2.0] range for unbiased neural network training.

CRITICAL BUG FIXED:
- Old range: [-1.0, 2.0] → creates training distribution bias (model sees +2.0 but never -2.0)
- New range: [-1.0, 1.0] → symmetric, zero-centered, follows ML best practices

Research support:
- Goodfellow et al. (2016): "Deep Learning" - symmetric inputs
- Ioffe & Szegedy (2015): "Batch Normalization" - symmetric distributions
- Lopez de Prado (2018): "Advances in Financial ML" - unbiased features
"""

import pytest
import numpy as np

try:
    from obs_builder import build_observation_vector

    HAVE_OBS_BUILDER = True
except ImportError:
    HAVE_OBS_BUILDER = False
    pytest.skip("obs_builder (Cython module) not available", allow_module_level=True)


class TestBBPositionSymmetricClipping:
    """Tests for Bollinger Bands position symmetric clipping fix"""

    def setup_method(self):
        """Setup common test parameters"""
        self.price = 100.0
        self.prev_price = 99.0
        self.bb_lower = 95.0
        self.bb_upper = 105.0
        self.bb_width = self.bb_upper - self.bb_lower  # = 10.0

        # External features (21 elements for 4h timeframe)
        self.norm_cols = np.zeros(21, dtype=np.float32)
        self.norm_cols_validity = np.ones(21, dtype=np.uint8)

        # Output buffer (83 features without validity flags, 104 with validity flags)
        # Feature layout:
        # 0-2: price, log_volume_norm, rel_volume
        # 3-4: ma5, ma5_valid
        # 5-6: ma20, ma20_valid
        # 7-8: rsi, rsi_valid
        # 9-10: macd, macd_valid
        # 11-12: macd_signal, macd_signal_valid
        # 13-14: momentum, momentum_valid
        # 15-16: atr, atr_valid
        # 17-18: cci, cci_valid
        # 19-20: obv, obv_valid
        # 21: ret_bar
        # 22: vol_proxy
        # 23: cash_ratio
        # 24: position_value_ratio
        # 25: vol_imbalance
        # 26: trade_intensity
        # 27: realized_spread
        # 28: agent_fill_ratio
        # 29: price_momentum
        # 30: bb_squeeze
        # 31: trend_strength
        # 32: bb_position ← TARGET FEATURE
        # 33: bb_width
        # 34: event_importance
        # 35: time_since_event
        # 36: risk_off_flag
        # 37-38: fear_greed, has_fear_greed
        # 39-59: external features (21)
        # 60-62: token metadata (num_tokens_norm, token_id_norm, padding)
        # 63-83: external validity flags (21) if enabled
        self.out_features = np.zeros(104, dtype=np.float32)
        self.bb_position_idx = 32

    def test_price_at_middle_returns_neutral(self):
        """
        Price at middle of BB → bb_position = 0.5 (neutral)
        """
        price = (self.bb_lower + self.bb_upper) / 2.0  # = 100.0
        bb_position_expected = 0.5  # (100 - 95) / 10 = 0.5

        build_observation_vector(
            price=price,
            prev_price=self.prev_price,
            log_volume_norm=0.0,
            rel_volume=0.0,
            ma5=100.0,
            ma20=100.0,
            rsi14=50.0,
            macd=0.0,
            macd_signal=0.0,
            momentum=0.0,
            atr=2.0,
            cci=0.0,
            obv=0.0,
            bb_lower=self.bb_lower,
            bb_upper=self.bb_upper,
            is_high_importance=0.0,
            time_since_event=0.0,
            fear_greed_value=50.0,
            has_fear_greed=True,
            risk_off_flag=False,
            cash=10000.0,
            units=0.0,
            last_vol_imbalance=0.0,
            last_trade_intensity=0.0,
            last_realized_spread=0.0,
            last_agent_fill_ratio=0.0,
            token_id=0,
            max_num_tokens=1,
            num_tokens=1,
            norm_cols_values=self.norm_cols,
            norm_cols_validity=self.norm_cols_validity,
            enable_validity_flags=True,
            out_features=self.out_features,
        )

        bb_position = self.out_features[self.bb_position_idx]
        assert (
            abs(bb_position - bb_position_expected) < 0.01
        ), f"Price at middle should give bb_position = {bb_position_expected:.2f}, got {bb_position:.4f}"

    def test_price_at_upper_band_returns_one(self):
        """
        Price at upper band → bb_position = 1.0
        """
        price = self.bb_upper  # = 105.0
        bb_position_expected = 1.0  # (105 - 95) / 10 = 1.0

        build_observation_vector(
            price=price,
            prev_price=self.prev_price,
            log_volume_norm=0.0,
            rel_volume=0.0,
            ma5=100.0,
            ma20=100.0,
            rsi14=50.0,
            macd=0.0,
            macd_signal=0.0,
            momentum=0.0,
            atr=2.0,
            cci=0.0,
            obv=0.0,
            bb_lower=self.bb_lower,
            bb_upper=self.bb_upper,
            is_high_importance=0.0,
            time_since_event=0.0,
            fear_greed_value=50.0,
            has_fear_greed=True,
            risk_off_flag=False,
            cash=10000.0,
            units=0.0,
            last_vol_imbalance=0.0,
            last_trade_intensity=0.0,
            last_realized_spread=0.0,
            last_agent_fill_ratio=0.0,
            token_id=0,
            max_num_tokens=1,
            num_tokens=1,
            norm_cols_values=self.norm_cols,
            norm_cols_validity=self.norm_cols_validity,
            enable_validity_flags=True,
            out_features=self.out_features,
        )

        bb_position = self.out_features[self.bb_position_idx]
        assert (
            abs(bb_position - bb_position_expected) < 0.01
        ), f"Price at upper band should give bb_position = {bb_position_expected:.2f}, got {bb_position:.4f}"

    def test_price_at_lower_band_returns_zero(self):
        """
        Price at lower band → bb_position = 0.0
        """
        price = self.bb_lower  # = 95.0
        bb_position_expected = 0.0  # (95 - 95) / 10 = 0.0

        build_observation_vector(
            price=price,
            prev_price=self.prev_price,
            log_volume_norm=0.0,
            rel_volume=0.0,
            ma5=100.0,
            ma20=100.0,
            rsi14=50.0,
            macd=0.0,
            macd_signal=0.0,
            momentum=0.0,
            atr=2.0,
            cci=0.0,
            obv=0.0,
            bb_lower=self.bb_lower,
            bb_upper=self.bb_upper,
            is_high_importance=0.0,
            time_since_event=0.0,
            fear_greed_value=50.0,
            has_fear_greed=True,
            risk_off_flag=False,
            cash=10000.0,
            units=0.0,
            last_vol_imbalance=0.0,
            last_trade_intensity=0.0,
            last_realized_spread=0.0,
            last_agent_fill_ratio=0.0,
            token_id=0,
            max_num_tokens=1,
            num_tokens=1,
            norm_cols_values=self.norm_cols,
            norm_cols_validity=self.norm_cols_validity,
            enable_validity_flags=True,
            out_features=self.out_features,
        )

        bb_position = self.out_features[self.bb_position_idx]
        assert (
            abs(bb_position - bb_position_expected) < 0.01
        ), f"Price at lower band should give bb_position = {bb_position_expected:.2f}, got {bb_position:.4f}"

    def test_price_above_upper_band_clips_to_one(self):
        """
        CRITICAL: Price above upper band → bb_position clips to 1.0 (NEW behavior)

        OLD behavior: price at upper + 1*width → bb_position = 2.0
        NEW behavior: price at upper + 1*width → bb_position = 1.0 (clipped)
        """
        # Price at upper band + 1 * width
        price = self.bb_upper + self.bb_width  # = 105 + 10 = 115
        # Unclipped: (115 - 95) / 10 = 2.0
        # NEW clipped: min(2.0, 1.0) = 1.0

        build_observation_vector(
            price=price,
            prev_price=self.prev_price,
            log_volume_norm=0.0,
            rel_volume=0.0,
            ma5=100.0,
            ma20=100.0,
            rsi14=50.0,
            macd=0.0,
            macd_signal=0.0,
            momentum=0.0,
            atr=2.0,
            cci=0.0,
            obv=0.0,
            bb_lower=self.bb_lower,
            bb_upper=self.bb_upper,
            is_high_importance=0.0,
            time_since_event=0.0,
            fear_greed_value=50.0,
            has_fear_greed=True,
            risk_off_flag=False,
            cash=10000.0,
            units=0.0,
            last_vol_imbalance=0.0,
            last_trade_intensity=0.0,
            last_realized_spread=0.0,
            last_agent_fill_ratio=0.0,
            token_id=0,
            max_num_tokens=1,
            num_tokens=1,
            norm_cols_values=self.norm_cols,
            norm_cols_validity=self.norm_cols_validity,
            enable_validity_flags=True,
            out_features=self.out_features,
        )

        bb_position = self.out_features[self.bb_position_idx]

        # NEW BEHAVIOR: Should clip to 1.0 (not 2.0)
        assert abs(bb_position - 1.0) < 0.01, (
            f"Extreme bullish breakout should clip to 1.0 (NEW), got {bb_position:.4f}. "
            f"OLD behavior would give 2.0."
        )

        # Verify it's NOT the old behavior
        assert (
            abs(bb_position - 2.0) > 0.5
        ), f"bb_position should NOT be 2.0 (old behavior), got {bb_position:.4f}"

    def test_price_below_lower_band_clips_to_minus_one(self):
        """
        CRITICAL: Price below lower band → bb_position clips to -1.0 (symmetric)

        NEW behavior: symmetric clipping [-1.0, 1.0]
        OLD behavior: also clipped to -1.0 (but asymmetric with upper bound)
        """
        # Price at lower band - 1 * width
        price = self.bb_lower - self.bb_width  # = 95 - 10 = 85
        # Unclipped: (85 - 95) / 10 = -1.0

        build_observation_vector(
            price=price,
            prev_price=self.prev_price,
            log_volume_norm=0.0,
            rel_volume=0.0,
            ma5=100.0,
            ma20=100.0,
            rsi14=50.0,
            macd=0.0,
            macd_signal=0.0,
            momentum=0.0,
            atr=2.0,
            cci=0.0,
            obv=0.0,
            bb_lower=self.bb_lower,
            bb_upper=self.bb_upper,
            is_high_importance=0.0,
            time_since_event=0.0,
            fear_greed_value=50.0,
            has_fear_greed=True,
            risk_off_flag=False,
            cash=10000.0,
            units=0.0,
            last_vol_imbalance=0.0,
            last_trade_intensity=0.0,
            last_realized_spread=0.0,
            last_agent_fill_ratio=0.0,
            token_id=0,
            max_num_tokens=1,
            num_tokens=1,
            norm_cols_values=self.norm_cols,
            norm_cols_validity=self.norm_cols_validity,
            enable_validity_flags=True,
            out_features=self.out_features,
        )

        bb_position = self.out_features[self.bb_position_idx]

        # Should clip to -1.0 (symmetric extreme)
        assert (
            abs(bb_position - (-1.0)) < 0.01
        ), f"Extreme bearish breakout should clip to -1.0, got {bb_position:.4f}"

    def test_symmetric_range_property(self):
        """
        CORE PROPERTY: Range should be symmetric [-1.0, 1.0]

        Tests that extreme bullish and bearish breakouts have symmetric magnitudes
        """
        # Extreme bullish: price = upper + 2*width
        price_bullish = self.bb_upper + 2 * self.bb_width  # = 125
        # Unclipped: (125 - 95) / 10 = 3.0
        # NEW clipped: 1.0

        build_observation_vector(
            price=price_bullish,
            prev_price=self.prev_price,
            log_volume_norm=0.0,
            rel_volume=0.0,
            ma5=100.0,
            ma20=100.0,
            rsi14=50.0,
            macd=0.0,
            macd_signal=0.0,
            momentum=0.0,
            atr=2.0,
            cci=0.0,
            obv=0.0,
            bb_lower=self.bb_lower,
            bb_upper=self.bb_upper,
            is_high_importance=0.0,
            time_since_event=0.0,
            fear_greed_value=50.0,
            has_fear_greed=True,
            risk_off_flag=False,
            cash=10000.0,
            units=0.0,
            last_vol_imbalance=0.0,
            last_trade_intensity=0.0,
            last_realized_spread=0.0,
            last_agent_fill_ratio=0.0,
            token_id=0,
            max_num_tokens=1,
            num_tokens=1,
            norm_cols_values=self.norm_cols,
            norm_cols_validity=self.norm_cols_validity,
            enable_validity_flags=True,
            out_features=self.out_features,
        )

        bb_pos_bullish = self.out_features[self.bb_position_idx]

        # Extreme bearish: price = lower - 2*width
        price_bearish = self.bb_lower - 2 * self.bb_width  # = 75
        # Unclipped: (75 - 95) / 10 = -2.0
        # NEW clipped: -1.0

        build_observation_vector(
            price=price_bearish,
            prev_price=self.prev_price,
            log_volume_norm=0.0,
            rel_volume=0.0,
            ma5=100.0,
            ma20=100.0,
            rsi14=50.0,
            macd=0.0,
            macd_signal=0.0,
            momentum=0.0,
            atr=2.0,
            cci=0.0,
            obv=0.0,
            bb_lower=self.bb_lower,
            bb_upper=self.bb_upper,
            is_high_importance=0.0,
            time_since_event=0.0,
            fear_greed_value=50.0,
            has_fear_greed=True,
            risk_off_flag=False,
            cash=10000.0,
            units=0.0,
            last_vol_imbalance=0.0,
            last_trade_intensity=0.0,
            last_realized_spread=0.0,
            last_agent_fill_ratio=0.0,
            token_id=0,
            max_num_tokens=1,
            num_tokens=1,
            norm_cols_values=self.norm_cols,
            norm_cols_validity=self.norm_cols_validity,
            enable_validity_flags=True,
            out_features=self.out_features,
        )

        bb_pos_bearish = self.out_features[self.bb_position_idx]

        # CRITICAL: Symmetric property
        assert (
            abs(bb_pos_bullish - 1.0) < 0.01
        ), f"Bullish extreme should be 1.0, got {bb_pos_bullish:.4f}"
        assert (
            abs(bb_pos_bearish - (-1.0)) < 0.01
        ), f"Bearish extreme should be -1.0, got {bb_pos_bearish:.4f}"

        # Magnitudes should be equal (symmetric)
        assert (
            abs(abs(bb_pos_bullish) - abs(bb_pos_bearish)) < 0.01
        ), f"Extremes should be symmetric: |{bb_pos_bullish:.4f}| vs |{bb_pos_bearish:.4f}|"

        # OLD BUG: bullish would be 2.0, bearish -1.0 → asymmetric (2x bias)
        # NEW FIX: both are 1.0 and -1.0 → symmetric (no bias)

    def test_no_value_above_one(self):
        """
        NEW BEHAVIOR: No bb_position value should ever exceed 1.0

        OLD BUG: Values could reach 2.0 for extreme bullish breakouts
        NEW FIX: Clipped to 1.0 maximum
        """
        # Test multiple extreme bullish prices
        extreme_prices = [
            self.bb_upper + 1 * self.bb_width,  # +1 width above
            self.bb_upper + 2 * self.bb_width,  # +2 widths above
            self.bb_upper + 10 * self.bb_width,  # +10 widths above (extreme)
        ]

        for price in extreme_prices:
            build_observation_vector(
                price=price,
                prev_price=self.prev_price,
                log_volume_norm=0.0,
                rel_volume=0.0,
                ma5=100.0,
                ma20=100.0,
                rsi14=50.0,
                macd=0.0,
                macd_signal=0.0,
                momentum=0.0,
                atr=2.0,
                cci=0.0,
                obv=0.0,
                bb_lower=self.bb_lower,
                bb_upper=self.bb_upper,
                is_high_importance=0.0,
                time_since_event=0.0,
                fear_greed_value=50.0,
                has_fear_greed=True,
                risk_off_flag=False,
                cash=10000.0,
                units=0.0,
                last_vol_imbalance=0.0,
                last_trade_intensity=0.0,
                last_realized_spread=0.0,
                last_agent_fill_ratio=0.0,
                token_id=0,
                max_num_tokens=1,
                num_tokens=1,
                norm_cols_values=self.norm_cols,
                norm_cols_validity=self.norm_cols_validity,
                enable_validity_flags=True,
                out_features=self.out_features,
            )

            bb_position = self.out_features[self.bb_position_idx]

            # CRITICAL: Must be <= 1.0 (NEW behavior)
            assert bb_position <= 1.0, (
                f"bb_position should be <= 1.0 for price={price:.1f}, got {bb_position:.4f}. "
                f"OLD BUG: would allow values up to 2.0"
            )

            # Should actually be exactly 1.0 for all extreme cases
            assert (
                abs(bb_position - 1.0) < 0.01
            ), f"Extreme bullish should clip to exactly 1.0, got {bb_position:.4f}"

    def test_no_value_below_minus_one(self):
        """
        Symmetric property: No bb_position value should ever go below -1.0
        """
        # Test multiple extreme bearish prices
        extreme_prices = [
            self.bb_lower - 1 * self.bb_width,  # -1 width below
            self.bb_lower - 2 * self.bb_width,  # -2 widths below
            self.bb_lower - 10 * self.bb_width,  # -10 widths below (extreme)
        ]

        for price in extreme_prices:
            build_observation_vector(
                price=price,
                prev_price=self.prev_price,
                log_volume_norm=0.0,
                rel_volume=0.0,
                ma5=100.0,
                ma20=100.0,
                rsi14=50.0,
                macd=0.0,
                macd_signal=0.0,
                momentum=0.0,
                atr=2.0,
                cci=0.0,
                obv=0.0,
                bb_lower=self.bb_lower,
                bb_upper=self.bb_upper,
                is_high_importance=0.0,
                time_since_event=0.0,
                fear_greed_value=50.0,
                has_fear_greed=True,
                risk_off_flag=False,
                cash=10000.0,
                units=0.0,
                last_vol_imbalance=0.0,
                last_trade_intensity=0.0,
                last_realized_spread=0.0,
                last_agent_fill_ratio=0.0,
                token_id=0,
                max_num_tokens=1,
                num_tokens=1,
                norm_cols_values=self.norm_cols,
                norm_cols_validity=self.norm_cols_validity,
                enable_validity_flags=True,
                out_features=self.out_features,
            )

            bb_position = self.out_features[self.bb_position_idx]

            # CRITICAL: Must be >= -1.0
            assert (
                bb_position >= -1.0
            ), f"bb_position should be >= -1.0 for price={price:.1f}, got {bb_position:.4f}"

            # Should actually be exactly -1.0 for all extreme cases
            assert (
                abs(bb_position - (-1.0)) < 0.01
            ), f"Extreme bearish should clip to exactly -1.0, got {bb_position:.4f}"

    def test_nan_bands_returns_neutral_fallback(self):
        """
        Edge case: NaN Bollinger Bands → bb_position = 0.5 (neutral fallback)
        """
        build_observation_vector(
            price=100.0,
            prev_price=self.prev_price,
            log_volume_norm=0.0,
            rel_volume=0.0,
            ma5=100.0,
            ma20=100.0,
            rsi14=50.0,
            macd=0.0,
            macd_signal=0.0,
            momentum=0.0,
            atr=2.0,
            cci=0.0,
            obv=0.0,
            bb_lower=np.nan,  # NaN bands
            bb_upper=np.nan,
            is_high_importance=0.0,
            time_since_event=0.0,
            fear_greed_value=50.0,
            has_fear_greed=True,
            risk_off_flag=False,
            cash=10000.0,
            units=0.0,
            last_vol_imbalance=0.0,
            last_trade_intensity=0.0,
            last_realized_spread=0.0,
            last_agent_fill_ratio=0.0,
            token_id=0,
            max_num_tokens=1,
            num_tokens=1,
            norm_cols_values=self.norm_cols,
            norm_cols_validity=self.norm_cols_validity,
            enable_validity_flags=True,
            out_features=self.out_features,
        )

        bb_position = self.out_features[self.bb_position_idx]

        # Should return neutral fallback 0.5
        assert (
            abs(bb_position - 0.5) < 0.01
        ), f"NaN bands should give neutral fallback 0.5, got {bb_position:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
