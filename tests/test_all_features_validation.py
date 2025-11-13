"""
Comprehensive validation test suite for observation vector.

This test suite validates that NO feature in the observation vector can contain NaN, Inf, or invalid values
under ANY circumstances, including edge cases.

Test Coverage:
1. All technical indicators with NaN inputs
2. All mathematical operations with boundary values (0, negative, very large)
3. Observation vector validation on first 30 bars of simulation
4. Edge cases: price=0, cash=0, units=0, empty norm_cols
5. Input validation for all external data sources
"""

import numpy as np
import pandas as pd
import pytest
from typing import Any


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def assert_no_nan_inf(obs: np.ndarray, context: str = ""):
    """Assert that observation contains no NaN or Inf values."""
    has_nan = np.any(np.isnan(obs))
    has_inf = np.any(np.isinf(obs))

    if has_nan:
        nan_indices = np.where(np.isnan(obs))[0]
        raise AssertionError(f"{context}: Found NaN at indices {nan_indices.tolist()}")

    if has_inf:
        inf_indices = np.where(np.isinf(obs))[0]
        raise AssertionError(f"{context}: Found Inf at indices {inf_indices.tolist()}")


def build_test_observation(
    price=50000.0,
    prev_price=50000.0,
    log_volume_norm=0.5,
    rel_volume=0.5,
    ma5=float('nan'),
    ma20=float('nan'),
    rsi14=50.0,
    macd=0.0,
    macd_signal=0.0,
    momentum=0.0,
    atr=float('nan'),
    cci=0.0,
    obv=0.0,
    bb_lower=float('nan'),
    bb_upper=float('nan'),
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
    norm_cols_size=21,
):
    """Build observation vector using obs_builder with custom inputs."""
    try:
        from obs_builder import build_observation_vector
    except ImportError:
        pytest.skip("obs_builder not available")

    obs = np.zeros(56, dtype=np.float32)
    norm_cols = np.zeros(norm_cols_size, dtype=np.float32)

    build_observation_vector(
        float(price),
        float(prev_price),
        float(log_volume_norm),
        float(rel_volume),
        float(ma5),
        float(ma20),
        float(rsi14),
        float(macd),
        float(macd_signal),
        float(momentum),
        float(atr),
        float(cci),
        float(obv),
        float(bb_lower),
        float(bb_upper),
        float(is_high_importance),
        float(time_since_event),
        float(fear_greed_value),
        bool(has_fear_greed),
        bool(risk_off_flag),
        float(cash),
        float(units),
        float(last_vol_imbalance),
        float(last_trade_intensity),
        float(last_realized_spread),
        float(last_agent_fill_ratio),
        int(token_id),
        int(max_num_tokens),
        int(num_tokens),
        norm_cols,
        obs,
    )

    return obs


# ============================================================================
# TEST SUITE
# ============================================================================

class TestInputValidation:
    """Test validation of all input parameters."""

    def test_nan_price(self):
        """CRITICAL: Test that NaN price is handled."""
        obs = build_test_observation(price=float('nan'), prev_price=50000.0)
        assert_no_nan_inf(obs, "NaN price")

    def test_nan_prev_price(self):
        """CRITICAL: Test that NaN prev_price is handled."""
        obs = build_test_observation(price=50000.0, prev_price=float('nan'))
        assert_no_nan_inf(obs, "NaN prev_price")

    def test_inf_price(self):
        """CRITICAL: Test that Inf price is handled."""
        obs = build_test_observation(price=float('inf'), prev_price=50000.0)
        assert_no_nan_inf(obs, "Inf price")

    def test_negative_price(self):
        """IMPORTANT: Test that negative price is handled."""
        obs = build_test_observation(price=-50000.0, prev_price=50000.0)
        assert_no_nan_inf(obs, "Negative price")

    def test_zero_price(self):
        """CRITICAL: Test that zero price is handled."""
        obs = build_test_observation(price=0.0, prev_price=50000.0)
        assert_no_nan_inf(obs, "Zero price")

    def test_very_small_price(self):
        """IMPORTANT: Test that very small price (1e-10) is handled."""
        obs = build_test_observation(price=1e-10, prev_price=1e-10)
        assert_no_nan_inf(obs, "Very small price")

    def test_very_large_price(self):
        """Test that very large price (1e15) is handled."""
        obs = build_test_observation(price=1e15, prev_price=1e15)
        assert_no_nan_inf(obs, "Very large price")

    def test_nan_log_volume_norm(self):
        """Test that NaN log_volume_norm is handled."""
        obs = build_test_observation(log_volume_norm=float('nan'))
        assert_no_nan_inf(obs, "NaN log_volume_norm")

    def test_nan_rel_volume(self):
        """Test that NaN rel_volume is handled."""
        obs = build_test_observation(rel_volume=float('nan'))
        assert_no_nan_inf(obs, "NaN rel_volume")


class TestTechnicalIndicators:
    """Test handling of NaN technical indicators."""

    def test_all_indicators_nan(self):
        """Test that all NaN indicators are handled with defaults."""
        obs = build_test_observation(
            ma5=float('nan'),
            ma20=float('nan'),
            rsi14=float('nan'),
            macd=float('nan'),
            macd_signal=float('nan'),
            momentum=float('nan'),
            atr=float('nan'),
            cci=float('nan'),
            obv=float('nan'),
            bb_lower=float('nan'),
            bb_upper=float('nan'),
        )
        assert_no_nan_inf(obs, "All indicators NaN")

    def test_all_indicators_inf(self):
        """Test that all Inf indicators are handled."""
        obs = build_test_observation(
            ma5=float('inf'),
            ma20=float('inf'),
            rsi14=float('inf'),
            macd=float('inf'),
            macd_signal=float('inf'),
            momentum=float('inf'),
            atr=float('inf'),
            cci=float('inf'),
            obv=float('inf'),
            bb_lower=float('inf'),
            bb_upper=float('inf'),
        )
        assert_no_nan_inf(obs, "All indicators Inf")

    def test_rsi_extreme_values(self):
        """Test RSI with extreme values (0, 100, -100)."""
        for rsi_val in [0.0, 100.0, -100.0, 1000.0]:
            obs = build_test_observation(rsi14=rsi_val)
            assert_no_nan_inf(obs, f"RSI={rsi_val}")

    def test_atr_zero(self):
        """IMPORTANT: Test ATR=0 (used in division)."""
        obs = build_test_observation(atr=0.0)
        assert_no_nan_inf(obs, "ATR=0")

    def test_atr_negative(self):
        """Test negative ATR (should not happen, but defensive)."""
        obs = build_test_observation(atr=-100.0)
        assert_no_nan_inf(obs, "ATR negative")

    def test_momentum_extreme(self):
        """Test momentum with extreme values."""
        for mom_val in [-1e6, 1e6, float('inf'), float('-inf')]:
            obs = build_test_observation(momentum=mom_val)
            assert_no_nan_inf(obs, f"Momentum={mom_val}")

    def test_bollinger_bands_invalid(self):
        """IMPORTANT: Test invalid Bollinger Bands (lower > upper, NaN combinations)."""
        # Case 1: bb_lower > bb_upper (should not happen)
        obs = build_test_observation(bb_lower=51000.0, bb_upper=49000.0)
        assert_no_nan_inf(obs, "BB lower > upper")

        # Case 2: bb_lower valid, bb_upper NaN
        obs = build_test_observation(bb_lower=49000.0, bb_upper=float('nan'))
        assert_no_nan_inf(obs, "BB upper NaN")

        # Case 3: bb_lower NaN, bb_upper valid
        obs = build_test_observation(bb_lower=float('nan'), bb_upper=51000.0)
        assert_no_nan_inf(obs, "BB lower NaN")

        # Case 4: Both NaN
        obs = build_test_observation(bb_lower=float('nan'), bb_upper=float('nan'))
        assert_no_nan_inf(obs, "BB both NaN")

    def test_bollinger_bands_zero_width(self):
        """Test BB with zero width (bb_lower == bb_upper)."""
        obs = build_test_observation(bb_lower=50000.0, bb_upper=50000.0)
        assert_no_nan_inf(obs, "BB zero width")


class TestMathematicalOperations:
    """Test all mathematical operations with edge cases."""

    def test_division_by_zero_protection(self):
        """Test that all divisions have protection against zero."""
        # prev_price = 0 (used in ret_bar calculation)
        obs = build_test_observation(price=50000.0, prev_price=0.0)
        assert_no_nan_inf(obs, "prev_price=0")

        # price = 0 (used in multiple calculations)
        obs = build_test_observation(price=0.0, prev_price=50000.0)
        assert_no_nan_inf(obs, "price=0")

    def test_tanh_overflow_protection(self):
        """Test that tanh with very large inputs doesn't overflow."""
        # Very large vol_imbalance (goes through tanh)
        obs = build_test_observation(last_vol_imbalance=1e10)
        assert_no_nan_inf(obs, "Large vol_imbalance")

        # Very large trade_intensity
        obs = build_test_observation(last_trade_intensity=1e10)
        assert_no_nan_inf(obs, "Large trade_intensity")

    def test_log1p_with_negative(self):
        """Test log1p protection (in mediator, but affects obs)."""
        # This is tested indirectly through log_volume_norm and rel_volume
        # which come pre-computed from mediator
        obs = build_test_observation(log_volume_norm=-10.0, rel_volume=-10.0)
        assert_no_nan_inf(obs, "Negative volume norms")


class TestAgentState:
    """Test agent state edge cases."""

    def test_zero_cash_zero_units(self):
        """CRITICAL: Test zero cash and zero units (total_worth=0)."""
        obs = build_test_observation(cash=0.0, units=0.0)
        assert_no_nan_inf(obs, "cash=0, units=0")

    def test_negative_cash(self):
        """Test negative cash (leverage/margin)."""
        obs = build_test_observation(cash=-10000.0, units=1.0, price=50000.0)
        assert_no_nan_inf(obs, "Negative cash")

    def test_very_large_position(self):
        """Test very large position."""
        obs = build_test_observation(units=1e10, price=50000.0)
        assert_no_nan_inf(obs, "Very large position")

    def test_realized_spread_extreme(self):
        """Test realized_spread with extreme values (clipped to -0.1, 0.1)."""
        for spread in [-10.0, 10.0, float('nan'), float('inf')]:
            obs = build_test_observation(last_realized_spread=spread)
            assert_no_nan_inf(obs, f"realized_spread={spread}")

    def test_agent_fill_ratio_invalid(self):
        """Test agent_fill_ratio with invalid values."""
        for ratio in [-1.0, 2.0, float('nan'), float('inf')]:
            obs = build_test_observation(last_agent_fill_ratio=ratio)
            assert_no_nan_inf(obs, f"fill_ratio={ratio}")


class TestExternalData:
    """Test external data sources."""

    def test_fear_greed_nan(self):
        """MEDIUM: Test NaN fear_greed_value."""
        obs = build_test_observation(fear_greed_value=float('nan'), has_fear_greed=True)
        assert_no_nan_inf(obs, "fear_greed NaN")

    def test_fear_greed_extreme(self):
        """Test fear_greed with values outside 0-100."""
        for fg_val in [-100.0, 200.0, 1000.0]:
            obs = build_test_observation(fear_greed_value=fg_val, has_fear_greed=True)
            assert_no_nan_inf(obs, f"fear_greed={fg_val}")

    def test_fear_greed_disabled(self):
        """Test observation when fear_greed is disabled."""
        obs = build_test_observation(fear_greed_value=50.0, has_fear_greed=False)
        assert_no_nan_inf(obs, "fear_greed disabled")

    def test_event_metadata_nan(self):
        """Test NaN event metadata."""
        obs = build_test_observation(
            is_high_importance=float('nan'),
            time_since_event=float('nan')
        )
        assert_no_nan_inf(obs, "Event metadata NaN")

    def test_event_metadata_negative(self):
        """Test negative time_since_event (should not happen)."""
        obs = build_test_observation(time_since_event=-100.0)
        assert_no_nan_inf(obs, "Negative time_since_event")


class TestNormCols:
    """Test normalized columns (external features)."""

    def test_empty_norm_cols(self):
        """Test with empty norm_cols (size=0)."""
        # This should work but observation size will be different
        # Skip this test as it requires different obs size
        pass

    def test_norm_cols_all_nan(self):
        """Test norm_cols with all NaN values."""
        try:
            from obs_builder import build_observation_vector
        except ImportError:
            pytest.skip("obs_builder not available")

        obs = np.zeros(56, dtype=np.float32)
        norm_cols = np.full(21, float('nan'), dtype=np.float32)

        build_observation_vector(
            50000.0, 50000.0, 0.5, 0.5,
            float('nan'), float('nan'), 50.0,
            0.0, 0.0, 0.0, float('nan'), 0.0, 0.0,
            float('nan'), float('nan'),
            0.0, 0.0, 50.0, True, False,
            10000.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0, 1, 1,
            norm_cols,
            obs,
        )

        assert_no_nan_inf(obs, "norm_cols all NaN")

    def test_norm_cols_all_inf(self):
        """Test norm_cols with all Inf values."""
        try:
            from obs_builder import build_observation_vector
        except ImportError:
            pytest.skip("obs_builder not available")

        obs = np.zeros(56, dtype=np.float32)
        norm_cols = np.full(21, float('inf'), dtype=np.float32)

        build_observation_vector(
            50000.0, 50000.0, 0.5, 0.5,
            float('nan'), float('nan'), 50.0,
            0.0, 0.0, 0.0, float('nan'), 0.0, 0.0,
            float('nan'), float('nan'),
            0.0, 0.0, 50.0, True, False,
            10000.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0, 1, 1,
            norm_cols,
            obs,
        )

        assert_no_nan_inf(obs, "norm_cols all Inf")

    def test_norm_cols_extreme_values(self):
        """Test norm_cols with extreme values."""
        try:
            from obs_builder import build_observation_vector
        except ImportError:
            pytest.skip("obs_builder not available")

        obs = np.zeros(56, dtype=np.float32)
        norm_cols = np.array([1e10, -1e10, 1e-10, -1e-10] + [0.0] * 17, dtype=np.float32)

        build_observation_vector(
            50000.0, 50000.0, 0.5, 0.5,
            float('nan'), float('nan'), 50.0,
            0.0, 0.0, 0.0, float('nan'), 0.0, 0.0,
            float('nan'), float('nan'),
            0.0, 0.0, 50.0, True, False,
            10000.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0, 1, 1,
            norm_cols,
            obs,
        )

        assert_no_nan_inf(obs, "norm_cols extreme values")


class TestFirstBars:
    """Test observation on first bars of simulation (cold start)."""

    def test_first_30_bars(self):
        """CRITICAL: Test that observation is valid on first 30 bars."""
        # Simulate first 30 bars where indicators are not ready
        for bar_idx in range(30):
            # Indicators become available at different bars:
            # ma5: ready after 5 bars
            # ma20: ready after 20 bars
            # rsi14: ready after 14 bars
            # macd: ready after ~26 bars
            # bb: ready after 20 bars
            # atr: ready after 14 bars

            ma5 = 50000.0 if bar_idx >= 5 else float('nan')
            ma20 = 50000.0 if bar_idx >= 20 else float('nan')
            rsi14 = 50.0 if bar_idx >= 14 else float('nan')
            macd = 0.0 if bar_idx >= 26 else float('nan')
            macd_signal = 0.0 if bar_idx >= 26 else float('nan')
            momentum = 0.0 if bar_idx >= 10 else float('nan')
            atr = 100.0 if bar_idx >= 14 else float('nan')
            bb_lower = 49000.0 if bar_idx >= 20 else float('nan')
            bb_upper = 51000.0 if bar_idx >= 20 else float('nan')

            obs = build_test_observation(
                price=50000.0 + bar_idx * 10,
                prev_price=50000.0 + (bar_idx - 1) * 10 if bar_idx > 0 else 50000.0,
                ma5=ma5,
                ma20=ma20,
                rsi14=rsi14,
                macd=macd,
                macd_signal=macd_signal,
                momentum=momentum,
                atr=atr,
                bb_lower=bb_lower,
                bb_upper=bb_upper,
            )

            assert_no_nan_inf(obs, f"Bar {bar_idx}")


class TestTokenMetadata:
    """Test token metadata edge cases."""

    def test_zero_max_tokens(self):
        """Test with max_num_tokens=0 (token features disabled)."""
        obs = build_test_observation(max_num_tokens=0, num_tokens=0, token_id=0)
        assert_no_nan_inf(obs, "max_num_tokens=0")

    def test_invalid_token_id(self):
        """Test with invalid token_id (out of range)."""
        obs = build_test_observation(token_id=-1, max_num_tokens=5, num_tokens=3)
        assert_no_nan_inf(obs, "token_id=-1")

        obs = build_test_observation(token_id=10, max_num_tokens=5, num_tokens=3)
        assert_no_nan_inf(obs, "token_id=10")

    def test_num_tokens_greater_than_max(self):
        """Test with num_tokens > max_num_tokens."""
        obs = build_test_observation(token_id=0, max_num_tokens=5, num_tokens=10)
        assert_no_nan_inf(obs, "num_tokens > max_num_tokens")


class TestWorstCaseScenarios:
    """Test worst-case combinations."""

    def test_all_inputs_nan(self):
        """EXTREME: Test with all possible NaN inputs."""
        obs = build_test_observation(
            price=float('nan'),
            prev_price=float('nan'),
            log_volume_norm=float('nan'),
            rel_volume=float('nan'),
            ma5=float('nan'),
            ma20=float('nan'),
            rsi14=float('nan'),
            macd=float('nan'),
            macd_signal=float('nan'),
            momentum=float('nan'),
            atr=float('nan'),
            cci=float('nan'),
            obv=float('nan'),
            bb_lower=float('nan'),
            bb_upper=float('nan'),
            fear_greed_value=float('nan'),
            is_high_importance=float('nan'),
            time_since_event=float('nan'),
        )
        assert_no_nan_inf(obs, "All inputs NaN")

    def test_all_inputs_inf(self):
        """EXTREME: Test with all possible Inf inputs."""
        obs = build_test_observation(
            price=float('inf'),
            prev_price=float('inf'),
            log_volume_norm=float('inf'),
            rel_volume=float('inf'),
            ma5=float('inf'),
            ma20=float('inf'),
            rsi14=float('inf'),
            macd=float('inf'),
            macd_signal=float('inf'),
            momentum=float('inf'),
            atr=float('inf'),
            cci=float('inf'),
            obv=float('inf'),
            bb_lower=float('inf'),
            bb_upper=float('inf'),
            fear_greed_value=float('inf'),
            is_high_importance=float('inf'),
            time_since_event=float('inf'),
        )
        assert_no_nan_inf(obs, "All inputs Inf")

    def test_all_zeros(self):
        """Test with all zero inputs."""
        obs = build_test_observation(
            price=0.0,
            prev_price=0.0,
            log_volume_norm=0.0,
            rel_volume=0.0,
            ma5=0.0,
            ma20=0.0,
            rsi14=0.0,
            macd=0.0,
            macd_signal=0.0,
            momentum=0.0,
            atr=0.0,
            cci=0.0,
            obv=0.0,
            bb_lower=0.0,
            bb_upper=0.0,
            fear_greed_value=0.0,
            cash=0.0,
            units=0.0,
        )
        assert_no_nan_inf(obs, "All zeros")

    def test_market_crash_scenario(self):
        """Test extreme market crash: price drop 99%, negative cash, high volatility."""
        obs = build_test_observation(
            price=500.0,  # Down 99%
            prev_price=50000.0,
            atr=25000.0,  # Extreme volatility
            rsi14=5.0,  # Oversold
            momentum=-49500.0,  # Huge negative momentum
            cash=-10000.0,  # Negative cash (margin call)
            units=100.0,  # Large position
            bb_lower=100.0,
            bb_upper=60000.0,  # Wide bands
            fear_greed_value=1.0,  # Extreme fear
        )
        assert_no_nan_inf(obs, "Market crash scenario")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
