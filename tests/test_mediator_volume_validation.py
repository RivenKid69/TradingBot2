"""
Test suite for mediator._extract_market_data() volume validation (P1 layer).

This test suite validates the P1 defense layer - computation validation in mediator.py.
It ensures that volume metric calculations are validated after computation to detect
numerical overflow/underflow that could produce NaN/Inf.

Test coverage:
1. Valid volume computations produce correct results
2. Extreme volume values that could cause overflow are detected
3. Integration: mediator → obs_builder full pipeline
4. Edge cases: zero volume, missing data, etc.

Defense-in-depth layers tested:
- P0: _get_safe_float() returns defaults for invalid raw data (tested implicitly)
- P1: _extract_market_data() validates computation results (tested here)
- P2: obs_builder validates inputs before writing (tested in test_volume_validation.py)

References:
- Addresses critical gap: P1 layer was not tested in initial implementation
- Complements test_volume_validation.py which tests P2 layer
"""

import pytest
import numpy as np
import math
from unittest.mock import Mock, MagicMock

# Try to import Mediator - skip tests if dependencies are missing
try:
    from mediator import Mediator
    HAS_MEDIATOR = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_MEDIATOR = False
    MEDIATOR_IMPORT_ERROR = str(e)
    pytest.skip(
        f"Mediator not available (missing dependencies: {MEDIATOR_IMPORT_ERROR}). "
        f"These tests verify P1 validation layer but require full mediator import.",
        allow_module_level=True
    )


class TestMediatorVolumeValidationP1:
    """Test P1 validation layer - computation validation in mediator._extract_market_data()."""

    def setup_method(self):
        """Set up test fixtures with mock environment."""
        # Create minimal mock environment
        self.mock_env = Mock()
        self.mock_env.max_abs_position = 1.0
        self.mock_env.max_notional = 1e12
        self.mock_env.max_drawdown_pct = 1.0
        self.mock_env.intrabar_dd_pct = 0.3
        self.mock_env.dd_window = 500
        self.mock_env.bankruptcy_cash_th = -1e12

        # Create mediator instance
        self.mediator = Mediator(self.mock_env, event_level=0)

        # Mock state for _extract_market_data() calls
        self.mock_state = Mock()
        self.mock_state.units = 0.0
        self.mock_state.cash = 10000.0

    # ========================================================================
    # P1 Tests: Valid computations produce correct results
    # ========================================================================

    def test_valid_volume_computation_produces_correct_log_volume_norm(self):
        """Test P1.1: Valid quote_volume produces correct log_volume_norm."""
        # Create row with valid volume data
        row = {"volume": 1000.0, "quote_asset_volume": 50000.0}

        result = self.mediator._extract_market_data(row, self.mock_state, 50000.0, 49500.0)

        # Verify computation: tanh(log1p(50000.0 / 240e6))
        expected = np.tanh(np.log1p(50000.0 / 240e6))
        assert math.isfinite(result["log_volume_norm"]), "log_volume_norm must be finite"
        assert abs(result["log_volume_norm"] - expected) < 1e-9, \
            f"Expected {expected}, got {result['log_volume_norm']}"

    def test_valid_volume_computation_produces_correct_rel_volume(self):
        """Test P1.2: Valid volume produces correct rel_volume."""
        row = {"volume": 1000.0, "quote_asset_volume": 50000.0}

        result = self.mediator._extract_market_data(row, self.mock_state, 50000.0, 49500.0)

        # Verify computation: tanh(log1p(1000.0 / 24000.0))
        expected = np.tanh(np.log1p(1000.0 / 24000.0))
        assert math.isfinite(result["rel_volume"]), "rel_volume must be finite"
        assert abs(result["rel_volume"] - expected) < 1e-9, \
            f"Expected {expected}, got {result['rel_volume']}"

    def test_zero_volume_produces_zero_metrics(self):
        """Test P1.3: Zero volume produces zero log_volume_norm and rel_volume."""
        row = {"volume": 0.0, "quote_asset_volume": 0.0}

        result = self.mediator._extract_market_data(row, self.mock_state, 50000.0, 49500.0)

        assert result["log_volume_norm"] == 0.0, "Zero quote_volume should produce 0.0"
        assert result["rel_volume"] == 0.0, "Zero volume should produce 0.0"

    def test_missing_volume_data_uses_defaults(self):
        """Test P1.4: Missing volume data uses safe defaults (P0 layer)."""
        row = {}  # No volume data

        # Should not raise - _get_safe_float returns default 1.0
        result = self.mediator._extract_market_data(row, self.mock_state, 50000.0, 49500.0)

        # Default volume=1.0, quote_volume=1.0
        expected_log = np.tanh(np.log1p(1.0 / 240e6))
        expected_rel = np.tanh(np.log1p(1.0 / 24000.0))

        assert abs(result["log_volume_norm"] - expected_log) < 1e-9
        assert abs(result["rel_volume"] - expected_rel) < 1e-9

    # ========================================================================
    # P1 Tests: Extreme values that could cause numerical issues
    # ========================================================================

    def test_very_large_quote_volume_remains_finite(self):
        """Test P1.5: Very large quote_volume still produces finite result."""
        # Large but valid volume (1 trillion dollars)
        row = {"volume": 1e9, "quote_asset_volume": 1e12}

        result = self.mediator._extract_market_data(row, self.mock_state, 50000.0, 49500.0)

        # tanh always returns [-1, 1], so should be finite
        assert math.isfinite(result["log_volume_norm"]), \
            "Even very large volumes should produce finite log_volume_norm"
        assert -1.0 <= result["log_volume_norm"] <= 1.0, \
            "log_volume_norm should be in tanh range [-1, 1]"

    def test_very_large_volume_remains_finite(self):
        """Test P1.6: Very large volume still produces finite result."""
        row = {"volume": 1e9, "quote_asset_volume": 1e12}

        result = self.mediator._extract_market_data(row, self.mock_state, 50000.0, 49500.0)

        assert math.isfinite(result["rel_volume"]), \
            "Even very large volumes should produce finite rel_volume"
        assert -1.0 <= result["rel_volume"] <= 1.0, \
            "rel_volume should be in tanh range [-1, 1]"

    def test_very_small_positive_volume_remains_finite(self):
        """Test P1.7: Very small positive volumes produce valid results."""
        row = {"volume": 0.0001, "quote_asset_volume": 0.001}

        result = self.mediator._extract_market_data(row, self.mock_state, 50000.0, 49500.0)

        assert math.isfinite(result["log_volume_norm"])
        assert math.isfinite(result["rel_volume"])
        # Small volumes should produce values close to 0
        assert abs(result["log_volume_norm"]) < 0.01
        assert abs(result["rel_volume"]) < 0.01

    # ========================================================================
    # P1 Tests: Edge cases with NaN/Inf in raw data (P0 should catch these)
    # ========================================================================

    def test_nan_raw_volume_handled_by_p0_layer(self):
        """Test P1.8: NaN in raw volume is caught by P0 (_get_safe_float)."""
        row = {"volume": float("nan"), "quote_asset_volume": 1000.0}

        # Should not raise - _get_safe_float returns default 1.0 for NaN
        result = self.mediator._extract_market_data(row, self.mock_state, 50000.0, 49500.0)

        # rel_volume computed from default volume=1.0
        assert math.isfinite(result["rel_volume"])
        assert math.isfinite(result["log_volume_norm"])

    def test_inf_raw_quote_volume_handled_by_p0_layer(self):
        """Test P1.9: Inf in raw quote_volume is caught by P0."""
        row = {"volume": 1000.0, "quote_asset_volume": float("inf")}

        # Should not raise - _get_safe_float returns default 1.0 for Inf
        result = self.mediator._extract_market_data(row, self.mock_state, 50000.0, 49500.0)

        # log_volume_norm computed from default quote_volume=1.0
        assert math.isfinite(result["log_volume_norm"])
        assert math.isfinite(result["rel_volume"])

    # ========================================================================
    # P1 Tests: Price validation (already exists, verify it works)
    # ========================================================================

    def test_invalid_mark_price_raises_valueerror(self):
        """Test P1.10: Invalid mark_price is caught by price validation."""
        row = {"volume": 1000.0, "quote_asset_volume": 50000.0}

        with pytest.raises(ValueError) as exc_info:
            self.mediator._extract_market_data(row, self.mock_state, float("nan"), 49500.0)

        assert "mark_price" in str(exc_info.value).lower()

    def test_invalid_prev_price_raises_valueerror(self):
        """Test P1.11: Invalid prev_price is caught by price validation."""
        row = {"volume": 1000.0, "quote_asset_volume": 50000.0}

        with pytest.raises(ValueError) as exc_info:
            self.mediator._extract_market_data(row, self.mock_state, 50000.0, 0.0)

        assert "prev_price" in str(exc_info.value).lower()

    # ========================================================================
    # P1 Tests: Return value structure
    # ========================================================================

    def test_returns_dict_with_all_required_keys(self):
        """Test P1.12: _extract_market_data returns dict with all keys."""
        row = {"volume": 1000.0, "quote_asset_volume": 50000.0}

        result = self.mediator._extract_market_data(row, self.mock_state, 50000.0, 49500.0)

        assert isinstance(result, dict)
        assert "price" in result
        assert "prev_price" in result
        assert "log_volume_norm" in result
        assert "rel_volume" in result
        assert len(result) == 4, f"Expected 4 keys, got {len(result)}"

    def test_returns_finite_values_for_all_fields(self):
        """Test P1.13: All returned values must be finite."""
        row = {"volume": 1000.0, "quote_asset_volume": 50000.0}

        result = self.mediator._extract_market_data(row, self.mock_state, 50000.0, 49500.0)

        for key, value in result.items():
            assert math.isfinite(value), f"{key} = {value} is not finite"

    # ========================================================================
    # P1 Tests: Integration with realistic data
    # ========================================================================

    def test_realistic_btc_4h_bar_data(self):
        """Test P1.14: Realistic BTC 4h bar produces valid metrics."""
        # Realistic BTC 4h bar: ~$500M quote volume, ~10 BTC volume
        row = {
            "volume": 10.0,
            "quote_asset_volume": 500_000_000.0,
        }

        result = self.mediator._extract_market_data(row, self.mock_state, 50000.0, 49500.0)

        # All values should be finite and in reasonable range
        assert math.isfinite(result["log_volume_norm"])
        assert math.isfinite(result["rel_volume"])

        # Volume metrics should be in tanh range
        assert -1.0 <= result["log_volume_norm"] <= 1.0
        assert -1.0 <= result["rel_volume"] <= 1.0

        # With 500M quote volume, should be relatively high (>0.1)
        assert result["log_volume_norm"] > 0.1, \
            f"Expected high volume metric, got {result['log_volume_norm']}"

    def test_realistic_low_volume_4h_bar(self):
        """Test P1.15: Low volume 4h bar produces small metrics."""
        # Low volume bar: $1M quote volume, 0.02 BTC volume
        row = {
            "volume": 0.02,
            "quote_asset_volume": 1_000_000.0,
        }

        result = self.mediator._extract_market_data(row, self.mock_state, 50000.0, 49500.0)

        # Low volume should produce small but positive metrics
        assert 0.0 < result["log_volume_norm"] < 0.1, \
            f"Expected small log_volume_norm, got {result['log_volume_norm']}"
        assert 0.0 < result["rel_volume"] < 0.01, \
            f"Expected small rel_volume, got {result['rel_volume']}"


class TestMediatorVolumeIntegration:
    """Integration tests: mediator → obs_builder full pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_env = Mock()
        self.mock_env.max_abs_position = 1.0
        self.mock_env.max_notional = 1e12
        self.mock_env.max_drawdown_pct = 1.0
        self.mock_env.intrabar_dd_pct = 0.3
        self.mock_env.dd_window = 500
        self.mock_env.bankruptcy_cash_th = -1e12

        self.mediator = Mediator(self.mock_env, event_level=0)
        self.mock_state = Mock()
        self.mock_state.units = 0.0
        self.mock_state.cash = 10000.0

    def test_integration_mediator_to_obs_builder(self):
        """Test INT.1: Full pipeline from mediator to obs_builder works."""
        try:
            from obs_builder import build_observation_vector
        except ImportError:
            pytest.skip("obs_builder not available")

        # Extract market data using mediator
        row = {"volume": 1000.0, "quote_asset_volume": 50_000_000.0}
        market_data = self.mediator._extract_market_data(
            row, self.mock_state, 50000.0, 49500.0
        )

        # Pass to obs_builder
        out_features = np.zeros(56, dtype=np.float32)
        norm_cols = np.zeros(21, dtype=np.float32)

        # Should not raise - both P1 and P2 validation pass
        build_observation_vector(
            market_data["price"],
            market_data["prev_price"],
            market_data["log_volume_norm"],
            market_data["rel_volume"],
            50100.0,  # ma5
            49900.0,  # ma20
            55.0,     # rsi14
            10.0,     # macd
            8.0,      # macd_signal
            15.0,     # momentum
            500.0,    # atr
            20.0,     # cci
            10000.0,  # obv
            49000.0,  # bb_lower
            51000.0,  # bb_upper
            0.0,      # is_high_importance
            1.0,      # time_since_event
            50.0,     # fear_greed_value
            True,     # has_fear_greed
            False,    # risk_off_flag
            10000.0,  # cash
            0.5,      # units
            0.1,      # last_vol_imbalance
            5.0,      # last_trade_intensity
            0.001,    # last_realized_spread
            0.95,     # last_agent_fill_ratio
            0,        # token_id
            1,        # max_num_tokens
            1,        # num_tokens
            norm_cols,
            out_features,
        )

        # Verify observation is valid
        assert not np.any(np.isnan(out_features)), "Observation contains NaN"
        assert not np.any(np.isinf(out_features)), "Observation contains Inf"

        # Verify volume metrics are in observation
        assert out_features[1] == pytest.approx(market_data["log_volume_norm"])
        assert out_features[2] == pytest.approx(market_data["rel_volume"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
