"""
Comprehensive test suite for volume metric validation in obs_builder.

This test suite validates that volume-derived inputs (log_volume_norm, rel_volume)
are properly validated before being used in observation vector construction,
preventing NaN/Inf values from corrupting the observation.

Test coverage:
1. Valid volume metric inputs (finite values in typical range [-1, 1])
2. NaN detection and rejection for both metrics
3. Infinity detection and rejection
4. Zero values (valid - no volume)
5. Negative values (edge case - should be allowed as finite)
6. Integration: full observation vector construction with validated volume metrics
7. Defense-in-depth: mediator.py computation validation

References:
- Issue: "Нет валидации log_volume_norm и rel_volume (obs_builder.pyx:90, 92)"
- Research: Defense in Depth (OWASP), Data validation best practices (Cube Software)
- Pattern: Based on test_price_validation.py for consistency
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


class TestVolumeMetricValidation:
    """Test suite for volume metric validation in build_observation_vector."""

    def setup_method(self):
        """Set up test fixtures with valid default parameters."""
        self.valid_params = {
            "price": 50000.0,
            "prev_price": 49500.0,
            "log_volume_norm": 0.5,  # Typical normalized volume
            "rel_volume": 0.3,        # Typical relative volume
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

    # ========================================================================
    # P0 Tests: Valid inputs should succeed
    # ========================================================================

    def test_valid_volume_metric_inputs(self):
        """Test 1: Valid volume metric inputs should succeed without errors."""
        params = self.valid_params.copy()

        # Should not raise any exception
        build_observation_vector(**params)

        # Observation should be populated
        obs = params["out_features"]
        assert not np.allclose(obs, 0.0), "Observation should be populated"

        # Volume metrics should be at indices 1 and 2 (after price at index 0)
        assert obs[1] == pytest.approx(0.5), "log_volume_norm should be at index 1"
        assert obs[2] == pytest.approx(0.3), "rel_volume should be at index 2"

        # No NaN or Inf in output
        assert not np.any(np.isnan(obs)), "No NaN values in observation"
        assert not np.any(np.isinf(obs)), "No Inf values in observation"

    def test_zero_volume_metrics_succeed(self):
        """Test 2: Zero volume metrics are valid (no volume in bar)."""
        params = self.valid_params.copy()
        params["log_volume_norm"] = 0.0
        params["rel_volume"] = 0.0

        # Should not raise
        build_observation_vector(**params)

        obs = params["out_features"]
        assert obs[1] == pytest.approx(0.0), "Zero log_volume_norm is valid"
        assert obs[2] == pytest.approx(0.0), "Zero rel_volume is valid"
        assert not np.any(np.isnan(obs)), "No NaN in observation"
        assert not np.any(np.isinf(obs)), "No Inf in observation"

    def test_typical_tanh_range_volume_metrics(self):
        """Test 3: Volume metrics in typical tanh range [-1, 1] should succeed."""
        params = self.valid_params.copy()
        params["log_volume_norm"] = 0.95  # High volume
        params["rel_volume"] = -0.1       # Slight negative (edge case)

        # Should not raise
        build_observation_vector(**params)

        obs = params["out_features"]
        assert obs[1] == pytest.approx(0.95)
        assert obs[2] == pytest.approx(-0.1)
        assert not np.any(np.isnan(obs)), "No NaN in observation"
        assert not np.any(np.isinf(obs)), "No Inf in observation"

    # ========================================================================
    # P1 Tests: NaN detection and rejection
    # ========================================================================

    def test_nan_log_volume_norm_raises_error(self):
        """Test 4: NaN log_volume_norm should raise ValueError with diagnostic message."""
        params = self.valid_params.copy()
        params["log_volume_norm"] = float("nan")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "NaN" in error_msg, "Error should mention NaN"
        assert "log_volume_norm" in error_msg.lower(), "Error should mention log_volume_norm"
        assert "corrupted" in error_msg.lower() or "volume" in error_msg.lower(), \
            "Error should explain data corruption"

    def test_nan_rel_volume_raises_error(self):
        """Test 5: NaN rel_volume should raise ValueError."""
        params = self.valid_params.copy()
        params["rel_volume"] = float("nan")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "NaN" in error_msg, "Error should mention NaN"
        assert "rel_volume" in error_msg.lower(), "Error should mention rel_volume parameter"

    def test_both_volume_metrics_nan_raises_error(self):
        """Test 6: Both volume metrics NaN - should catch first one (log_volume_norm)."""
        params = self.valid_params.copy()
        params["log_volume_norm"] = float("nan")
        params["rel_volume"] = float("nan")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        # Should mention log_volume_norm (validated first)
        assert "log_volume_norm" in error_msg.lower(), "Should validate log_volume_norm first"

    # ========================================================================
    # P2 Tests: Infinity detection and rejection
    # ========================================================================

    def test_positive_infinity_log_volume_norm_raises_error(self):
        """Test 7: +Inf log_volume_norm should raise ValueError."""
        params = self.valid_params.copy()
        params["log_volume_norm"] = float("inf")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "infinity" in error_msg.lower(), "Error should mention infinity"
        assert "log_volume_norm" in error_msg.lower(), "Error should mention log_volume_norm"
        assert "overflow" in error_msg.lower(), "Error should explain overflow"

    def test_negative_infinity_log_volume_norm_raises_error(self):
        """Test 8: -Inf log_volume_norm should raise ValueError."""
        params = self.valid_params.copy()
        params["log_volume_norm"] = float("-inf")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "infinity" in error_msg.lower(), "Error should mention infinity"
        assert "log_volume_norm" in error_msg.lower(), "Error should mention log_volume_norm"

    def test_positive_infinity_rel_volume_raises_error(self):
        """Test 9: +Inf rel_volume should raise ValueError."""
        params = self.valid_params.copy()
        params["rel_volume"] = float("inf")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "infinity" in error_msg.lower(), "Error should mention infinity"
        assert "rel_volume" in error_msg.lower(), "Error should mention rel_volume"

    def test_negative_infinity_rel_volume_raises_error(self):
        """Test 10: -Inf rel_volume should raise ValueError."""
        params = self.valid_params.copy()
        params["rel_volume"] = float("-inf")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "infinity" in error_msg.lower(), "Error should mention infinity"
        assert "rel_volume" in error_msg.lower(), "Error should mention rel_volume"

    # ========================================================================
    # P3 Tests: Edge cases
    # ========================================================================

    def test_very_small_positive_volume_metrics_succeed(self):
        """Test 11: Very small but positive volume metrics should succeed."""
        params = self.valid_params.copy()
        params["log_volume_norm"] = 0.000001
        params["rel_volume"] = 0.0000005

        # Should not raise
        build_observation_vector(**params)

        obs = params["out_features"]
        assert obs[1] == pytest.approx(0.000001), "Small log_volume_norm should be preserved"
        assert obs[2] == pytest.approx(0.0000005), "Small rel_volume should be preserved"
        assert not np.any(np.isnan(obs)), "No NaN in observation"
        assert not np.any(np.isinf(obs)), "No Inf in observation"

    def test_negative_volume_metrics_succeed(self):
        """Test 12: Negative volume metrics should succeed (edge case, but valid)."""
        params = self.valid_params.copy()
        params["log_volume_norm"] = -0.25  # Theoretical edge case
        params["rel_volume"] = -0.1

        # Should not raise (finite values are allowed)
        build_observation_vector(**params)

        obs = params["out_features"]
        assert obs[1] == pytest.approx(-0.25), "Negative log_volume_norm is finite"
        assert obs[2] == pytest.approx(-0.1), "Negative rel_volume is finite"
        assert not np.any(np.isnan(obs)), "No NaN in observation"
        assert not np.any(np.isinf(obs)), "No Inf in observation"

    def test_near_tanh_saturation_volume_metrics_succeed(self):
        """Test 13: Volume metrics near tanh saturation (±0.99) should succeed."""
        params = self.valid_params.copy()
        params["log_volume_norm"] = 0.99  # Near saturation (very high volume)
        params["rel_volume"] = -0.99      # Near negative saturation (edge case)

        # Should not raise
        build_observation_vector(**params)

        obs = params["out_features"]
        assert obs[1] == pytest.approx(0.99), "Near-saturation log_volume_norm is valid"
        assert obs[2] == pytest.approx(-0.99), "Near-saturation rel_volume is valid"
        assert not np.any(np.isnan(obs)), "No NaN in observation"
        assert not np.any(np.isinf(obs)), "No Inf in observation"

    def test_one_valid_one_invalid_volume_metric(self):
        """Test 14: One valid, one invalid - should catch invalid one."""
        params = self.valid_params.copy()
        params["log_volume_norm"] = 0.5  # Valid
        params["rel_volume"] = float("nan")  # Invalid

        # Should raise for rel_volume (validated second)
        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)
        assert "rel_volume" in error_msg.lower(), "Should catch rel_volume validation"

    # ========================================================================
    # P4 Tests: Integration with full observation vector
    # ========================================================================

    def test_integration_full_observation_with_valid_volume_metrics(self):
        """Test 15: Integration test - full observation with validated volume metrics."""
        params = self.valid_params.copy()

        # Set up realistic market data with volume
        params["price"] = 51234.56
        params["prev_price"] = 51000.00
        params["log_volume_norm"] = 0.75  # High volume (normalized)
        params["rel_volume"] = 0.82       # High relative volume
        params["ma5"] = 51100.0
        params["ma20"] = 50800.0
        params["rsi14"] = 62.5

        # Execute
        build_observation_vector(**params)

        obs = params["out_features"]

        # Validate observation properties
        assert obs.shape == (56,), f"Expected shape (56,), got {obs.shape}"
        assert obs[0] == pytest.approx(51234.56), "Price at index 0"
        assert obs[1] == pytest.approx(0.75), "log_volume_norm at index 1"
        assert obs[2] == pytest.approx(0.82), "rel_volume at index 2"

        # CRITICAL: No NaN or Inf anywhere in the observation
        nan_count = np.sum(np.isnan(obs))
        inf_count = np.sum(np.isinf(obs))
        assert nan_count == 0, f"Found {nan_count} NaN values in observation"
        assert inf_count == 0, f"Found {inf_count} Inf values in observation"

        # Verify all features are finite
        assert all(math.isfinite(float(x)) for x in obs), "All features must be finite"

    def test_no_nan_propagation_from_volume_metrics(self):
        """Test 16: NaN in volume metrics must not propagate to observation vector."""
        params = self.valid_params.copy()
        params["log_volume_norm"] = float("nan")

        # CRITICAL: System must raise ValueError, not silently write NaN to obs[1]
        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        # Verify error message is informative
        error_msg = str(exc_info.value)
        assert "log_volume_norm" in error_msg.lower(), \
            "Error must identify log_volume_norm parameter"
        assert "NaN" in error_msg or "nan" in error_msg.lower(), \
            "Error must mention NaN"

    def test_fail_fast_not_silent_failure_volume_metrics(self):
        """
        Test 17: CRITICAL - System fails loudly for invalid volume metrics.

        This test verifies fail-fast philosophy for volume metrics:
        - Invalid log_volume_norm → ValueError raised (fail loudly)
        - NOT silent obs[1] = NaN (silent corruption)

        Anti-pattern we're preventing:
        - Silent NaN propagation that corrupts observation
        - Model training on corrupted volume signals
        - Impossible debugging due to no error signal
        """
        params = self.valid_params.copy()
        params["rel_volume"] = float("nan")

        # CRITICAL: System must raise ValueError, not return silent NaN
        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        # Verify error message is informative
        error_msg = str(exc_info.value)
        assert "rel_volume" in error_msg.lower(), \
            "Error must identify rel_volume parameter"
        assert "NaN" in error_msg or "nan" in error_msg.lower(), \
            "Error must mention NaN"

    # ========================================================================
    # P5 Tests: Real-world scenarios
    # ========================================================================

    def test_realistic_high_volume_scenario(self):
        """
        Test 18: Realistic high volume 4h bar scenario.

        Scenario: BTC 4h bar with very high volume (normalized to 0.9)
        Expected: Valid observation with no NaN/Inf
        """
        params = self.valid_params.copy()
        params["price"] = 50000.0
        params["prev_price"] = 49500.0
        params["log_volume_norm"] = 0.9  # Very high volume bar
        params["rel_volume"] = 0.85

        build_observation_vector(**params)
        obs = params["out_features"]

        assert not np.any(np.isnan(obs)), "No NaN for high volume bar"
        assert not np.any(np.isinf(obs)), "No Inf for high volume bar"
        assert obs[1] == pytest.approx(0.9), "log_volume_norm preserved"
        assert obs[2] == pytest.approx(0.85), "rel_volume preserved"

    def test_realistic_low_volume_scenario(self):
        """
        Test 19: Realistic low volume scenario (weekend/off-hours).

        Scenario: Low volume period with near-zero normalized volume
        Expected: Valid observation with volume metrics near 0
        """
        params = self.valid_params.copy()
        params["log_volume_norm"] = 0.05  # Very low volume
        params["rel_volume"] = 0.02

        build_observation_vector(**params)
        obs = params["out_features"]

        assert not np.any(np.isnan(obs)), "No NaN for low volume bar"
        assert not np.any(np.isinf(obs)), "No Inf for low volume bar"
        assert obs[1] == pytest.approx(0.05), "Low log_volume_norm is valid"
        assert obs[2] == pytest.approx(0.02), "Low rel_volume is valid"

    def test_realistic_no_volume_scenario(self):
        """
        Test 20: Realistic no volume scenario (market halt, missing data).

        Scenario: Bar with zero volume (missing data or market halt)
        Expected: Valid observation with volume metrics = 0.0
        """
        params = self.valid_params.copy()
        params["log_volume_norm"] = 0.0
        params["rel_volume"] = 0.0

        build_observation_vector(**params)
        obs = params["out_features"]

        assert not np.any(np.isnan(obs)), "No NaN for zero volume bar"
        assert not np.any(np.isinf(obs)), "No Inf for zero volume bar"
        assert obs[1] == pytest.approx(0.0), "Zero log_volume_norm is valid"
        assert obs[2] == pytest.approx(0.0), "Zero rel_volume is valid"


class TestVolumeMetricErrorMessages:
    """Test suite specifically for volume metric error message quality."""

    def setup_method(self):
        """Set up minimal valid parameters."""
        self.valid_params = {
            "price": 50000.0,
            "prev_price": 49500.0,
            "log_volume_norm": 0.5,
            "rel_volume": 0.3,
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
        """Test 21: Error messages should contain actionable diagnostic info."""
        params = self.valid_params.copy()
        params["log_volume_norm"] = float("nan")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)

        # Should contain diagnostic keywords
        assert any(keyword in error_msg.lower() for keyword in
                  ["volume", "corrupted", "data", "calculation", "check"]), \
            "Error should provide diagnostic context"

    def test_error_message_for_infinity_is_clear(self):
        """Test 22: Infinity error should clearly explain the issue."""
        params = self.valid_params.copy()
        params["rel_volume"] = float("inf")

        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(**params)

        error_msg = str(exc_info.value)

        # Should mention that infinity is invalid and why
        assert "infinity" in error_msg.lower(), "Should explain infinity issue"
        assert any(keyword in error_msg.lower() for keyword in
                  ["overflow", "finite", "invalid"]), \
            "Should clearly identify the problem"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
