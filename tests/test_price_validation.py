"""
Comprehensive test suite for price validation in obs_builder.

This test suite validates that price inputs are properly validated before
being used in observation vector construction, preventing NaN/Inf/negative
values from corrupting the entire observation.

Test coverage:
1. Valid price inputs (positive, finite values)
2. NaN price detection and rejection
3. Infinity price detection and rejection
4. Zero price detection and rejection
5. Negative price detection and rejection
6. prev_price validation (same checks as price)
7. Edge cases: very small positive prices, very large prices
8. Integration: full observation vector construction with validated prices

References:
- Issue: "Нет валидации входного параметра price (obs_builder.pyx:88, 135, 139, 182, 195, 206, 216, 224, 231)"
- Research: Financial data validation best practices (Cube Software, OMSCS ML Trading)
"""

import numpy as np
import pytest
import math


# Try to import the compiled Cython module
try:
    from obs_builder import build_observation_vector
    HAS_OBS_BUILDER = True
except ImportError:
    HAS_OBS_BUILDER = False
    pytest.skip("obs_builder module not available (needs compilation)", allow_module_level=True)


class TestPriceValidation:
    """Test suite for price validation in build_observation_vector."""

    def setup_method(self):
        """Set up test fixtures with valid default parameters."""
        self.valid_params = {
            "price": 50000.0,
            "prev_price": 49500.0,
            "log_volume_norm": 0.5,
            "rel_volume": 0.3,
            "ma5": 50100.0,
            "ma20": 49900.0,
            "rsi14": 55.0,
            "macd": 10.0,
            "macd_signal": 8.0,
            "momentum": 15.0,
            "atr": 500.0,
            "cci": 20.0,
            "obv": 10000.0,
            "bb_lower": 49000.0,
            "bb_upper": 51000.0,
            "is_high_importance": 0.0,
            "time_since_event": 1.0,
            "fear_greed_value": 50.0,
            "has_fear_greed": True,
            "risk_off_flag": False,
            "cash": 10000.0,
            "units": 0.5,
            "last_vol_imbalance": 0.1,
            "last_trade_intensity": 5.0,
            "last_realized_spread": 0.001,
            "last_agent_fill_ratio": 0.95,
            "token_id": 0,
            "max_num_tokens": 1,
            "num_tokens": 1,
            "norm_cols_values": np.zeros(21, dtype=np.float32),
            "out_features": np.zeros(56, dtype=np.float32),
        }

    def test_valid_price_inputs(self):
        """Test 1: Valid price inputs should succeed without errors."""
        params = self.valid_params.copy()

        # Should not raise any exception
        build_observation_vector(**params)

        # Observation should be populated (not all zeros)
        obs = params["out_features"]
        assert not np.allclose(obs, 0.0), "Observation should be populated"
        assert obs[0] == pytest.approx(50000.0), "First feature should be price"

        # No NaN or Inf in output
        assert not np.any(np.isnan(obs)), "No NaN values in observation"
        assert not np.any(np.isinf(obs)), "No Inf values in observation"

    def test_nan_price_raises_error(self):
        """Test 2: NaN price should raise ValueError with diagnostic message."""
        params = self.valid_params.copy()
        params["price"] = float("nan")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "NaN" in error_msg, "Error should mention NaN"
        assert "price" in error_msg.lower(), "Error should mention price parameter"
        assert "corrupted" in error_msg.lower() or "missing" in error_msg.lower(), \
            "Error should explain data corruption"

    def test_nan_prev_price_raises_error(self):
        """Test 3: NaN prev_price should raise ValueError."""
        params = self.valid_params.copy()
        params["prev_price"] = float("nan")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "NaN" in error_msg, "Error should mention NaN"
        assert "prev_price" in error_msg.lower(), "Error should mention prev_price parameter"

    def test_positive_infinity_price_raises_error(self):
        """Test 4: +Inf price should raise ValueError."""
        params = self.valid_params.copy()
        params["price"] = float("inf")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "infinity" in error_msg.lower(), "Error should mention infinity"
        assert "price" in error_msg.lower(), "Error should mention price parameter"
        assert "overflow" in error_msg.lower(), "Error should explain arithmetic overflow"

    def test_negative_infinity_price_raises_error(self):
        """Test 5: -Inf price should raise ValueError."""
        params = self.valid_params.copy()
        params["price"] = float("-inf")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "infinity" in error_msg.lower(), "Error should mention infinity"
        assert "price" in error_msg.lower(), "Error should mention price parameter"

    def test_positive_infinity_prev_price_raises_error(self):
        """Test 6: +Inf prev_price should raise ValueError."""
        params = self.valid_params.copy()
        params["prev_price"] = float("inf")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "infinity" in error_msg.lower(), "Error should mention infinity"
        assert "prev_price" in error_msg.lower(), "Error should mention prev_price"

    def test_negative_infinity_prev_price_raises_error(self):
        """Test 7: -Inf prev_price should raise ValueError."""
        params = self.valid_params.copy()
        params["prev_price"] = float("-inf")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "infinity" in error_msg.lower(), "Error should mention infinity"
        assert "prev_price" in error_msg.lower(), "Error should mention prev_price"

    def test_zero_price_raises_error(self):
        """Test 8: Zero price should raise ValueError."""
        params = self.valid_params.copy()
        params["price"] = 0.0

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "positive" in error_msg.lower(), "Error should mention positive requirement"
        assert "price" in error_msg.lower(), "Error should mention price parameter"
        assert "0" in error_msg or "zero" in error_msg.lower(), "Error should mention zero value"

    def test_zero_prev_price_raises_error(self):
        """Test 9: Zero prev_price should raise ValueError."""
        params = self.valid_params.copy()
        params["prev_price"] = 0.0

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "positive" in error_msg.lower(), "Error should mention positive requirement"
        assert "prev_price" in error_msg.lower(), "Error should mention prev_price"

    def test_negative_price_raises_error(self):
        """Test 10: Negative price should raise ValueError."""
        params = self.valid_params.copy()
        params["price"] = -1000.0

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "positive" in error_msg.lower(), "Error should mention positive requirement"
        assert "price" in error_msg.lower(), "Error should mention price parameter"

    def test_negative_prev_price_raises_error(self):
        """Test 11: Negative prev_price should raise ValueError."""
        params = self.valid_params.copy()
        params["prev_price"] = -500.0

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "positive" in error_msg.lower(), "Error should mention positive requirement"
        assert "prev_price" in error_msg.lower(), "Error should mention prev_price"

    def test_very_small_positive_price_succeeds(self):
        """Test 12: Very small but positive prices should succeed."""
        params = self.valid_params.copy()
        params["price"] = 0.000001  # 1 satoshi in BTC terms
        params["prev_price"] = 0.0000009

        # Should not raise
        build_observation_vector(**params)

        obs = params["out_features"]
        assert obs[0] == pytest.approx(0.000001), "Price should be preserved"
        assert not np.any(np.isnan(obs)), "No NaN in observation"
        assert not np.any(np.isinf(obs)), "No Inf in observation"

    def test_very_large_price_succeeds(self):
        """Test 13: Very large prices should succeed if finite."""
        params = self.valid_params.copy()
        params["price"] = 1e9  # 1 billion
        params["prev_price"] = 9.99e8

        # Should not raise
        build_observation_vector(**params)

        obs = params["out_features"]
        assert obs[0] == pytest.approx(1e9), "Large price should be preserved"
        assert not np.any(np.isnan(obs)), "No NaN in observation"
        assert not np.any(np.isinf(obs)), "No Inf in observation"

    def test_price_equals_prev_price_succeeds(self):
        """Test 14: price == prev_price (no movement) should succeed."""
        params = self.valid_params.copy()
        params["price"] = 50000.0
        params["prev_price"] = 50000.0

        # Should not raise
        build_observation_vector(**params)

        obs = params["out_features"]
        assert not np.any(np.isnan(obs)), "No NaN in observation"
        assert not np.any(np.isinf(obs)), "No Inf in observation"

    def test_multiple_invalid_prices(self):
        """Test 15: Both price and prev_price invalid - should catch first one."""
        params = self.valid_params.copy()
        params["price"] = float("nan")
        params["prev_price"] = float("nan")

        # Should raise for price first (validated before prev_price)
        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        # Should mention "price" (not "prev_price") as it's validated first
        assert "price" in error_msg.lower(), "Should fail on price validation first"

    def test_integration_full_observation_vector(self):
        """Test 16: Integration test - full observation with validated prices."""
        params = self.valid_params.copy()

        # Set up realistic market data
        params["price"] = 51234.56
        params["prev_price"] = 51000.00
        params["log_volume_norm"] = 0.75
        params["rel_volume"] = 0.82
        params["ma5"] = 51100.0
        params["ma20"] = 50800.0
        params["rsi14"] = 62.5
        params["macd"] = 15.5
        params["macd_signal"] = 12.3
        params["momentum"] = 20.0
        params["atr"] = 450.0
        params["cci"] = 15.0
        params["obv"] = 15000.0
        params["bb_lower"] = 50200.0
        params["bb_upper"] = 52000.0

        # Add some norm_cols data (CVD, GARCH, etc.)
        params["norm_cols_values"] = np.array([
            0.5,   # cvd_24h
            0.3,   # cvd_7d
            0.025, # yang_zhang_48h
            0.030, # yang_zhang_7d
            0.028, # garch_200h
            0.032, # garch_14d
            0.001, # ret_12h
            0.002, # ret_24h
            0.0005, # ret_4h
            50000.0, # sma_12000
            0.035, # yang_zhang_30d
            0.022, # parkinson_48h
            0.028, # parkinson_7d
            0.040, # garch_30d
            0.52,  # taker_buy_ratio
            0.51,  # taker_buy_ratio_sma_24h
            0.50,  # taker_buy_ratio_sma_8h
            0.53,  # taker_buy_ratio_sma_16h
            0.01,  # taker_buy_ratio_momentum_4h
            0.02,  # taker_buy_ratio_momentum_8h
            0.015, # taker_buy_ratio_momentum_12h
        ], dtype=np.float32)

        # Execute
        build_observation_vector(**params)

        obs = params["out_features"]

        # Validate observation properties
        assert obs.shape == (56,), f"Expected shape (56,), got {obs.shape}"
        assert obs[0] == pytest.approx(51234.56), "First feature should be price"

        # Check that observation is well-formed
        non_zero_count = np.count_nonzero(obs)
        assert non_zero_count >= 30, f"Expected >=30 non-zero features, got {non_zero_count}"

        # CRITICAL: No NaN or Inf anywhere in the observation
        nan_count = np.sum(np.isnan(obs))
        inf_count = np.sum(np.isinf(obs))
        assert nan_count == 0, f"Found {nan_count} NaN values in observation: {np.where(np.isnan(obs))}"
        assert inf_count == 0, f"Found {inf_count} Inf values in observation: {np.where(np.isinf(obs))}"

        # Verify all derived computations worked correctly
        # These are the 15+ computations that depend on price being valid
        assert all(math.isfinite(float(x)) for x in obs), "All features must be finite"

    def test_edge_case_price_jump(self):
        """Test 17: Large price jump (prev_price << price) should succeed."""
        params = self.valid_params.copy()
        params["prev_price"] = 10000.0
        params["price"] = 50000.0  # 5x price jump

        # Should not raise (large but valid jump)
        build_observation_vector(**params)

        obs = params["out_features"]
        assert not np.any(np.isnan(obs)), "No NaN despite large price jump"
        assert not np.any(np.isinf(obs)), "No Inf despite large price jump"

    def test_edge_case_price_crash(self):
        """Test 18: Large price crash (prev_price >> price) should succeed."""
        params = self.valid_params.copy()
        params["prev_price"] = 50000.0
        params["price"] = 10000.0  # 80% crash

        # Should not raise (large but valid drop)
        build_observation_vector(**params)

        obs = params["out_features"]
        assert not np.any(np.isnan(obs)), "No NaN despite large price crash"
        assert not np.any(np.isinf(obs)), "No Inf despite large price crash"


class TestPriceValidationErrorMessages:
    """Test suite specifically for error message quality and diagnostics."""

    def setup_method(self):
        """Set up minimal valid parameters."""
        self.valid_params = {
            "price": 50000.0,
            "prev_price": 49500.0,
            "log_volume_norm": 0.0,
            "rel_volume": 0.0,
            "ma5": float("nan"),
            "ma20": float("nan"),
            "rsi14": 50.0,
            "macd": 0.0,
            "macd_signal": 0.0,
            "momentum": 0.0,
            "atr": 0.0,
            "cci": 0.0,
            "obv": 0.0,
            "bb_lower": float("nan"),
            "bb_upper": float("nan"),
            "is_high_importance": 0.0,
            "time_since_event": 0.0,
            "fear_greed_value": 50.0,
            "has_fear_greed": False,
            "risk_off_flag": False,
            "cash": 10000.0,
            "units": 0.0,
            "last_vol_imbalance": 0.0,
            "last_trade_intensity": 0.0,
            "last_realized_spread": 0.0,
            "last_agent_fill_ratio": 1.0,
            "token_id": 0,
            "max_num_tokens": 1,
            "num_tokens": 1,
            "norm_cols_values": np.zeros(21, dtype=np.float32),
            "out_features": np.zeros(56, dtype=np.float32),
        }

    def test_error_message_contains_diagnostic_info(self):
        """Test 19: Error messages should contain actionable diagnostic info."""
        params = self.valid_params.copy()
        params["price"] = float("nan")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)

        # Should contain these diagnostic keywords
        assert any(keyword in error_msg.lower() for keyword in
                  ["data", "corrupted", "missing", "integrity", "check"]), \
            "Error should provide diagnostic context"

    def test_error_message_for_zero_price_is_clear(self):
        """Test 20: Zero price error should clearly explain the issue."""
        params = self.valid_params.copy()
        params["price"] = 0.0

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)

        # Should mention that zero is invalid and why
        assert "positive" in error_msg.lower(), "Should explain positive requirement"
        assert any(keyword in error_msg.lower() for keyword in
                  ["invalid", "error", "zero"]), \
            "Should clearly identify the problem"


class TestPortfolioValidation:
    """Test suite for cash and units validation."""

    def setup_method(self):
        """Set up test fixtures with valid default parameters."""
        self.valid_params = {
            "price": 50000.0,
            "prev_price": 49500.0,
            "log_volume_norm": 0.5,
            "rel_volume": 0.3,
            "ma5": 50100.0,
            "ma20": 49900.0,
            "rsi14": 55.0,
            "macd": 10.0,
            "macd_signal": 8.0,
            "momentum": 15.0,
            "atr": 500.0,
            "cci": 20.0,
            "obv": 10000.0,
            "bb_lower": 49000.0,
            "bb_upper": 51000.0,
            "is_high_importance": 0.0,
            "time_since_event": 1.0,
            "fear_greed_value": 50.0,
            "has_fear_greed": True,
            "risk_off_flag": False,
            "cash": 10000.0,
            "units": 0.5,
            "last_vol_imbalance": 0.1,
            "last_trade_intensity": 5.0,
            "last_realized_spread": 0.001,
            "last_agent_fill_ratio": 0.95,
            "token_id": 0,
            "max_num_tokens": 1,
            "num_tokens": 1,
            "norm_cols_values": np.zeros(21, dtype=np.float32),
            "out_features": np.zeros(56, dtype=np.float32),
        }

    def test_nan_cash_raises_error(self):
        """Test 21: NaN cash should raise ValueError."""
        params = self.valid_params.copy()
        params["cash"] = float("nan")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "cash" in error_msg.lower(), "Error should mention cash parameter"
        assert "NaN" in error_msg, "Error should mention NaN"

    def test_inf_cash_raises_error(self):
        """Test 22: +Inf cash should raise ValueError."""
        params = self.valid_params.copy()
        params["cash"] = float("inf")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "cash" in error_msg.lower(), "Error should mention cash"
        assert "infinity" in error_msg.lower(), "Error should mention infinity"

    def test_nan_units_raises_error(self):
        """Test 23: NaN units should raise ValueError."""
        params = self.valid_params.copy()
        params["units"] = float("nan")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "units" in error_msg.lower(), "Error should mention units parameter"
        assert "NaN" in error_msg, "Error should mention NaN"

    def test_inf_units_raises_error(self):
        """Test 24: +Inf units should raise ValueError."""
        params = self.valid_params.copy()
        params["units"] = float("inf")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "units" in error_msg.lower(), "Error should mention units"
        assert "infinity" in error_msg.lower(), "Error should mention infinity"

    def test_zero_cash_succeeds(self):
        """Test 25: Zero cash is valid (no money in account)."""
        params = self.valid_params.copy()
        params["cash"] = 0.0

        # Should not raise
        build_observation_vector(**params)

        obs = params["out_features"]
        assert not np.any(np.isnan(obs)), "No NaN in observation"
        assert not np.any(np.isinf(obs)), "No Inf in observation"

    def test_zero_units_succeeds(self):
        """Test 26: Zero units is valid (no position)."""
        params = self.valid_params.copy()
        params["units"] = 0.0

        # Should not raise
        build_observation_vector(**params)

        obs = params["out_features"]
        assert not np.any(np.isnan(obs)), "No NaN in observation"
        assert not np.any(np.isinf(obs)), "No Inf in observation"

    def test_negative_cash_succeeds(self):
        """Test 27: Negative cash is valid (margin debt, short position)."""
        params = self.valid_params.copy()
        params["cash"] = -5000.0  # Margin debt

        # Should not raise
        build_observation_vector(**params)

        obs = params["out_features"]
        assert not np.any(np.isnan(obs)), "No NaN in observation"
        assert not np.any(np.isinf(obs)), "No Inf in observation"

    def test_negative_units_succeeds(self):
        """Test 28: Negative units is valid (short position)."""
        params = self.valid_params.copy()
        params["units"] = -0.5  # Short position

        # Should not raise
        build_observation_vector(**params)

        obs = params["out_features"]
        assert not np.any(np.isnan(obs)), "No NaN in observation"
        assert not np.any(np.isinf(obs)), "No Inf in observation"

    def test_both_cash_and_units_zero_succeeds(self):
        """Test 29: Both cash and units = 0 (empty portfolio)."""
        params = self.valid_params.copy()
        params["cash"] = 0.0
        params["units"] = 0.0

        # Should not raise
        build_observation_vector(**params)

        obs = params["out_features"]
        assert not np.any(np.isnan(obs)), "No NaN in observation"
        assert not np.any(np.isinf(obs)), "No Inf in observation"

    def test_large_cash_values_succeed(self):
        """Test 30: Very large cash values should work."""
        params = self.valid_params.copy()
        params["cash"] = 1e9  # 1 billion

        # Should not raise
        build_observation_vector(**params)

        obs = params["out_features"]
        assert not np.any(np.isnan(obs)), "No NaN in observation"
        assert not np.any(np.isinf(obs)), "No Inf in observation"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
