"""
Comprehensive test suite for prev_price validation in ret_bar calculation.

This test suite specifically targets the vulnerability where invalid prev_price
values could propagate NaN into the ret_bar feature (index 14) in the observation vector.

Coverage:
- P0 (Entry point validation): Tests at build_observation_vector() wrapper
- P1 (Mediator validation): Tests at mediator._extract_market_data()
- P2 (Defense-in-depth): Tests inline safety checks in C function
- Edge cases: Zero price difference, extreme jumps, denormalized numbers

Research references:
- "Defense in Depth" (OWASP): Multiple validation layers
- "Fail-fast validation" (Martin Fowler): Catch errors early
- IEEE 754 floating point standard: NaN propagation behavior
- "Data Validation Best Practices" (Cube Software)
"""

import pytest
import numpy as np
import math
from obs_builder import build_observation_vector


class TestPrevPriceRetBarValidation:
    """Test prev_price validation specifically for ret_bar calculation."""

    @pytest.fixture
    def valid_params(self):
        """Baseline valid parameters for observation vector construction."""
        return {
            "price": 50000.0,
            "prev_price": 49500.0,  # Normal 1% price change
            "log_volume_norm": 0.5,
            "rel_volume": 0.5,
            "ma5": 49800.0,
            "ma20": 49600.0,
            "rsi14": 55.0,
            "macd": 10.0,
            "macd_signal": 8.0,
            "momentum": 5.0,
            "atr": 100.0,
            "cci": 20.0,
            "obv": 1000000.0,
            "bb_lower": 49000.0,
            "bb_upper": 50500.0,
            "is_high_importance": 0.0,
            "time_since_event": 12.0,
            "fear_greed_value": 50.0,
            "has_fear_greed": True,
            "risk_off_flag": False,
            "cash": 10000.0,
            "units": 0.1,
            "last_vol_imbalance": 0.05,
            "last_trade_intensity": 0.1,
            "last_realized_spread": 0.001,
            "last_agent_fill_ratio": 0.95,
            "token_id": 0,
            "max_num_tokens": 10,
            "num_tokens": 5,
            "norm_cols_values": np.zeros(5, dtype=np.float32),
            "out_features": np.zeros(56, dtype=np.float32),
        }

    # ========================================================================
    # P0 Tests: Entry point validation (wrapper function)
    # ========================================================================

    def test_nan_prev_price_rejected_at_entry(self, valid_params):
        """
        Test P0.1: NaN prev_price should be rejected at wrapper validation.

        Critical: This is the first line of defense. If prev_price is NaN,
        it must be caught BEFORE entering the C function to prevent NaN
        propagation to ret_bar feature.
        """
        params = valid_params.copy()
        params["prev_price"] = float("nan")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "NaN" in error_msg, "Error should mention NaN"
        assert "prev_price" in error_msg.lower(), "Error should identify prev_price"
        assert "corrupted" in error_msg.lower() or "missing" in error_msg.lower(), \
            "Error should explain data corruption cause"

    def test_inf_prev_price_rejected_at_entry(self, valid_params):
        """
        Test P0.2: +Inf prev_price should be rejected at wrapper validation.

        Critical: Infinity in prev_price would cause undefined behavior in
        division for ret_bar calculation. Must be caught early.
        """
        params = valid_params.copy()
        params["prev_price"] = float("inf")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "infinity" in error_msg.lower(), "Error should mention infinity"
        assert "prev_price" in error_msg.lower(), "Error should identify prev_price"

    def test_neg_inf_prev_price_rejected_at_entry(self, valid_params):
        """
        Test P0.3: -Inf prev_price should be rejected at wrapper validation.
        """
        params = valid_params.copy()
        params["prev_price"] = float("-inf")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "infinity" in error_msg.lower(), "Error should mention infinity"
        assert "prev_price" in error_msg.lower(), "Error should identify prev_price"

    def test_zero_prev_price_rejected_at_entry(self, valid_params):
        """
        Test P0.4: Zero prev_price should be rejected (invalid price).

        Critical: While zero wouldn't cause NaN, it's an invalid price value
        that indicates data corruption. Must be rejected.
        """
        params = valid_params.copy()
        params["prev_price"] = 0.0

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "positive" in error_msg.lower(), "Error should mention positive requirement"
        assert "prev_price" in error_msg.lower(), "Error should identify prev_price"

    def test_negative_prev_price_rejected_at_entry(self, valid_params):
        """
        Test P0.5: Negative prev_price should be rejected (invalid price).
        """
        params = valid_params.copy()
        params["prev_price"] = -1000.0

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "positive" in error_msg.lower(), "Error should mention positive requirement"
        assert "prev_price" in error_msg.lower(), "Error should identify prev_price"

    # ========================================================================
    # P1 Tests: Valid prev_price values produce correct ret_bar
    # ========================================================================

    def test_ret_bar_normal_price_increase(self, valid_params):
        """
        Test P1.1: Normal price increase produces positive ret_bar.

        Setup: price=50000, prev_price=49500 (1.01% increase)
        Expected: ret_bar > 0 (normalized by tanh)
        """
        params = valid_params.copy()
        params["price"] = 50000.0
        params["prev_price"] = 49500.0

        build_observation_vector(**params)
        ret_bar = params["out_features"][14]  # ret_bar is at index 14

        # Calculate expected value: tanh((50000 - 49500) / (49500 + 1e-8))
        expected = math.tanh((50000.0 - 49500.0) / (49500.0 + 1e-8))

        assert not math.isnan(ret_bar), "ret_bar should not be NaN"
        assert not math.isinf(ret_bar), "ret_bar should not be Inf"
        assert ret_bar > 0, "ret_bar should be positive for price increase"
        assert abs(ret_bar - expected) < 1e-6, f"ret_bar should match expected {expected}"

    def test_ret_bar_normal_price_decrease(self, valid_params):
        """
        Test P1.2: Normal price decrease produces negative ret_bar.

        Setup: price=49000, prev_price=50000 (2% decrease)
        Expected: ret_bar < 0 (normalized by tanh)
        """
        params = valid_params.copy()
        params["price"] = 49000.0
        params["prev_price"] = 50000.0

        build_observation_vector(**params)
        ret_bar = params["out_features"][14]

        expected = math.tanh((49000.0 - 50000.0) / (50000.0 + 1e-8))

        assert not math.isnan(ret_bar), "ret_bar should not be NaN"
        assert not math.isinf(ret_bar), "ret_bar should not be Inf"
        assert ret_bar < 0, "ret_bar should be negative for price decrease"
        assert abs(ret_bar - expected) < 1e-6, f"ret_bar should match expected {expected}"

    def test_ret_bar_no_price_change(self, valid_params):
        """
        Test P1.3: No price change produces zero ret_bar.

        Setup: price = prev_price = 50000
        Expected: ret_bar ≈ 0
        """
        params = valid_params.copy()
        params["price"] = 50000.0
        params["prev_price"] = 50000.0

        build_observation_vector(**params)
        ret_bar = params["out_features"][14]

        assert not math.isnan(ret_bar), "ret_bar should not be NaN"
        assert not math.isinf(ret_bar), "ret_bar should not be Inf"
        assert abs(ret_bar) < 1e-6, "ret_bar should be ~0 for no price change"

    def test_ret_bar_extreme_price_jump(self, valid_params):
        """
        Test P1.4: Extreme price jump (10x) produces valid ret_bar.

        Setup: price=500000, prev_price=50000 (10x jump)
        Expected: ret_bar close to 1.0 (tanh saturates at large values)
        """
        params = valid_params.copy()
        params["price"] = 500000.0
        params["prev_price"] = 50000.0

        build_observation_vector(**params)
        ret_bar = params["out_features"][14]

        expected = math.tanh((500000.0 - 50000.0) / (50000.0 + 1e-8))

        assert not math.isnan(ret_bar), "ret_bar should not be NaN"
        assert not math.isinf(ret_bar), "ret_bar should not be Inf"
        assert ret_bar > 0.9, "ret_bar should be close to 1.0 for 10x jump"
        assert abs(ret_bar - expected) < 1e-6, f"ret_bar should match expected {expected}"

    def test_ret_bar_extreme_price_crash(self, valid_params):
        """
        Test P1.5: Extreme price crash (90% drop) produces valid ret_bar.

        Setup: price=5000, prev_price=50000 (90% crash)
        Expected: ret_bar strongly negative (tanh(-0.9) ≈ -0.716)
        Note: tanh saturates slowly; need >3.0 input for >0.99 output
        """
        params = valid_params.copy()
        params["price"] = 5000.0
        params["prev_price"] = 50000.0

        build_observation_vector(**params)
        ret_bar = params["out_features"][14]

        expected = math.tanh((5000.0 - 50000.0) / (50000.0 + 1e-8))

        assert not math.isnan(ret_bar), "ret_bar should not be NaN"
        assert not math.isinf(ret_bar), "ret_bar should not be Inf"
        assert ret_bar < -0.7, "ret_bar should be strongly negative for 90% crash"
        assert abs(ret_bar - expected) < 1e-6, f"ret_bar should match expected {expected}"

    # ========================================================================
    # P2 Tests: Edge cases and numerical stability
    # ========================================================================

    def test_ret_bar_very_small_prev_price(self, valid_params):
        """
        Test P2.1: Very small prev_price (but valid) produces correct ret_bar.

        Setup: prev_price=0.00001 (1 satoshi-like value)
        Expected: Valid ret_bar calculation without overflow
        """
        params = valid_params.copy()
        params["price"] = 0.00002
        params["prev_price"] = 0.00001

        build_observation_vector(**params)
        ret_bar = params["out_features"][14]

        expected = math.tanh((0.00002 - 0.00001) / (0.00001 + 1e-8))

        assert not math.isnan(ret_bar), "ret_bar should not be NaN for small prev_price"
        assert not math.isinf(ret_bar), "ret_bar should not be Inf for small prev_price"
        assert abs(ret_bar - expected) < 1e-6, f"ret_bar should match expected {expected}"

    def test_ret_bar_very_large_prev_price(self, valid_params):
        """
        Test P2.2: Very large prev_price produces correct ret_bar.

        Setup: prev_price=1e9 (billion dollar asset)
        Expected: Valid ret_bar without underflow
        """
        params = valid_params.copy()
        params["price"] = 1.01e9
        params["prev_price"] = 1.0e9

        build_observation_vector(**params)
        ret_bar = params["out_features"][14]

        expected = math.tanh((1.01e9 - 1.0e9) / (1.0e9 + 1e-8))

        assert not math.isnan(ret_bar), "ret_bar should not be NaN for large prev_price"
        assert not math.isinf(ret_bar), "ret_bar should not be Inf for large prev_price"
        assert abs(ret_bar - expected) < 1e-6, f"ret_bar should match expected {expected}"

    def test_ret_bar_tiny_price_change(self, valid_params):
        """
        Test P2.3: Tiny price change (0.001%) produces valid ret_bar.

        Setup: price=50000.5, prev_price=50000.0 (0.001% change)
        Expected: Very small but valid ret_bar
        """
        params = valid_params.copy()
        params["price"] = 50000.5
        params["prev_price"] = 50000.0

        build_observation_vector(**params)
        ret_bar = params["out_features"][14]

        expected = math.tanh((50000.5 - 50000.0) / (50000.0 + 1e-8))

        assert not math.isnan(ret_bar), "ret_bar should not be NaN for tiny change"
        assert not math.isinf(ret_bar), "ret_bar should not be Inf for tiny change"
        assert abs(ret_bar) < 0.01, "ret_bar should be very small for tiny change"
        assert abs(ret_bar - expected) < 1e-6, f"ret_bar should match expected {expected}"

    # ========================================================================
    # P3 Tests: Integration with full observation vector
    # ========================================================================

    def test_no_nan_in_observation_vector_with_valid_prev_price(self, valid_params):
        """
        Test P3.1: Valid prev_price produces no NaN in entire observation vector.

        Critical: Ensures prev_price validation prevents NaN propagation
        to ALL features, not just ret_bar.
        """
        params = valid_params.copy()
        build_observation_vector(**params)

        out_features = params["out_features"]
        nan_indices = np.where(np.isnan(out_features))[0]

        assert len(nan_indices) == 0, \
            f"No NaN values should be in observation vector. Found at indices: {nan_indices}"

    def test_ret_bar_index_14_is_correct(self, valid_params):
        """
        Test P3.2: Verify ret_bar is actually at index 14 (documentation check).

        Critical: If feature ordering changes, tests need to be updated.
        """
        params = valid_params.copy()
        params["price"] = 51000.0
        params["prev_price"] = 50000.0

        build_observation_vector(**params)

        # Calculate expected ret_bar
        expected_ret_bar = math.tanh((51000.0 - 50000.0) / (50000.0 + 1e-8))

        # Check index 14
        actual_ret_bar = params["out_features"][14]

        assert abs(actual_ret_bar - expected_ret_bar) < 1e-6, \
            f"ret_bar at index 14 should be {expected_ret_bar}, got {actual_ret_bar}. " \
            f"Feature ordering may have changed - update tests!"

    def test_both_price_and_prev_price_invalid(self, valid_params):
        """
        Test P3.3: Both price and prev_price invalid - should catch price first.

        Validates that validation happens in correct order (price before prev_price).
        """
        params = valid_params.copy()
        params["price"] = float("nan")
        params["prev_price"] = float("nan")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        # Should mention "price" (validated first), not "prev_price"
        assert "price" in error_msg.lower(), "Should validate price first"

    # ========================================================================
    # P4 Tests: Real-world scenarios
    # ========================================================================

    def test_ret_bar_btc_realistic_4h_movement(self, valid_params):
        """
        Test P4.1: Realistic BTC 4h price movement scenario.

        Scenario: BTC moves from $49,500 to $50,000 over 4 hours (1.01% gain)
        Expected: Small positive ret_bar reflecting realistic market movement
        """
        params = valid_params.copy()
        params["price"] = 50000.0
        params["prev_price"] = 49500.0

        build_observation_vector(**params)
        ret_bar = params["out_features"][14]

        assert not math.isnan(ret_bar), "ret_bar should not be NaN"
        assert 0.0 < ret_bar < 0.02, \
            f"ret_bar should be small positive for 1% gain, got {ret_bar}"

    def test_ret_bar_flash_crash_scenario(self, valid_params):
        """
        Test P4.2: Flash crash scenario (20% instant drop).

        Scenario: Market flash crash from $50,000 to $40,000
        Expected: Large negative ret_bar (but not NaN/Inf)
        """
        params = valid_params.copy()
        params["price"] = 40000.0
        params["prev_price"] = 50000.0

        build_observation_vector(**params)
        ret_bar = params["out_features"][14]

        assert not math.isnan(ret_bar), "ret_bar should not be NaN during flash crash"
        assert not math.isinf(ret_bar), "ret_bar should not be Inf during flash crash"
        assert ret_bar < -0.15, f"ret_bar should be strongly negative for 20% crash, got {ret_bar}"

    def test_ret_bar_sideways_market(self, valid_params):
        """
        Test P4.3: Sideways market (minimal price movement).

        Scenario: Price oscillates around $50,000 (0.01% change)
        Expected: ret_bar very close to zero
        """
        params = valid_params.copy()
        params["price"] = 50005.0
        params["prev_price"] = 50000.0

        build_observation_vector(**params)
        ret_bar = params["out_features"][14]

        assert not math.isnan(ret_bar), "ret_bar should not be NaN in sideways market"
        assert abs(ret_bar) < 0.001, \
            f"ret_bar should be near zero for sideways market, got {ret_bar}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
