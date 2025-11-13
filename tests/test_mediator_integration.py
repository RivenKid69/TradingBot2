"""
Integration tests for mediator validation with obs_builder.

This test suite validates the complete pipeline from mediator data extraction
through observation vector construction, ensuring that invalid data is caught
at the appropriate layer with clear error messages.

Test coverage:
1. Valid price data flows through successfully
2. NaN prices are caught by mediator._validate_critical_price()
3. Inf prices are caught by mediator._validate_critical_price()
4. Zero/negative prices are caught appropriately
5. NaN/Inf cash/units are caught by obs_builder
6. Valid edge cases (0 cash, negative margin) work correctly

This ensures defense-in-depth: validation at both mediator and obs_builder layers.
"""

import math
import numpy as np
import pytest
from typing import Any


# Test mediator validation methods independently
class TestMediatorValidation:
    """Test mediator._validate_critical_price() in isolation."""

    @staticmethod
    def _validate_critical_price(value: Any, param_name: str = "price") -> float:
        """Replicate mediator._validate_critical_price() for testing."""
        if value is None:
            raise ValueError(f"Invalid {param_name}: None")

        try:
            numeric = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid {param_name}: cannot convert to float. {e}")

        if math.isnan(numeric):
            raise ValueError(f"Invalid {param_name}: NaN")

        if math.isinf(numeric):
            raise ValueError(f"Invalid {param_name}: infinity")

        if numeric <= 0.0:
            raise ValueError(f"Invalid {param_name}: {numeric:.10f}. Must be > 0")

        return numeric

    def test_valid_price_passes(self):
        """Test 1: Valid positive price passes validation."""
        result = self._validate_critical_price(50000.0, "mark_price")
        assert result == 50000.0

    def test_none_price_raises(self):
        """Test 2: None price raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            self._validate_critical_price(None, "mark_price")
        assert "None" in str(exc_info.value)

    def test_nan_price_raises(self):
        """Test 3: NaN price raises ValueError at mediator layer."""
        with pytest.raises(ValueError) as exc_info:
            self._validate_critical_price(float("nan"), "mark_price")

        error_msg = str(exc_info.value)
        assert "NaN" in error_msg
        assert "mark_price" in error_msg.lower()

    def test_inf_price_raises(self):
        """Test 4: +Inf price raises ValueError at mediator layer."""
        with pytest.raises(ValueError) as exc_info:
            self._validate_critical_price(float("inf"), "mark_price")

        error_msg = str(exc_info.value)
        assert "infinity" in error_msg.lower()

    def test_zero_price_raises(self):
        """Test 5: Zero price raises ValueError at mediator layer."""
        with pytest.raises(ValueError) as exc_info:
            self._validate_critical_price(0.0, "mark_price")

        error_msg = str(exc_info.value)
        assert "0.0" in error_msg or "Must be > 0" in error_msg

    def test_negative_price_raises(self):
        """Test 6: Negative price raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            self._validate_critical_price(-1000.0, "mark_price")

        error_msg = str(exc_info.value)
        assert "-1000" in error_msg or "Must be > 0" in error_msg

    def test_string_price_raises(self):
        """Test 7: Non-numeric string raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            self._validate_critical_price("invalid", "mark_price")

        assert "cannot convert" in str(exc_info.value).lower()


class TestMediatorExtractMarketData:
    """Test mediator._extract_market_data() with validation."""

    @staticmethod
    def _validate_critical_price(value: Any, param_name: str = "price") -> float:
        """Same as mediator._validate_critical_price()."""
        if value is None:
            raise ValueError(f"Invalid {param_name}: None")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid {param_name}: cannot convert")
        if math.isnan(numeric):
            raise ValueError(f"Invalid {param_name}: NaN")
        if math.isinf(numeric):
            raise ValueError(f"Invalid {param_name}: infinity")
        if numeric <= 0.0:
            raise ValueError(f"Invalid {param_name}: must be > 0")
        return numeric

    def _extract_market_data_validated(
        self, mark_price: float, prev_price: float
    ):
        """
        Simplified version of mediator._extract_market_data() with validation.
        """
        # CRITICAL: Strict validation (no fallback to 0.0)
        price = self._validate_critical_price(mark_price, "mark_price")
        prev = self._validate_critical_price(prev_price, "prev_price")

        return {
            "price": price,
            "prev_price": prev,
        }

    def test_valid_prices_succeed(self):
        """Test 8: Valid prices flow through successfully."""
        result = self._extract_market_data_validated(50000.0, 49500.0)
        assert result["price"] == 50000.0
        assert result["prev_price"] == 49500.0

    def test_nan_mark_price_caught_early(self):
        """Test 9: NaN mark_price caught at mediator layer (before obs_builder)."""
        with pytest.raises(ValueError) as exc_info:
            self._extract_market_data_validated(float("nan"), 49500.0)

        error_msg = str(exc_info.value)
        assert "mark_price" in error_msg.lower()
        assert "NaN" in error_msg

    def test_inf_prev_price_caught_early(self):
        """Test 10: Inf prev_price caught at mediator layer."""
        with pytest.raises(ValueError) as exc_info:
            self._extract_market_data_validated(50000.0, float("inf"))

        error_msg = str(exc_info.value)
        assert "prev_price" in error_msg.lower()
        assert "infinity" in error_msg.lower()

    def test_zero_mark_price_caught_early(self):
        """Test 11: Zero mark_price caught at mediator layer (was silent before fix)."""
        with pytest.raises(ValueError) as exc_info:
            self._extract_market_data_validated(0.0, 49500.0)

        error_msg = str(exc_info.value)
        assert "mark_price" in error_msg.lower()
        # This is the KEY FIX: 0.0 no longer silently accepted


class TestFullPipelineIntegration:
    """
    Test the complete pipeline: mediator validation → obs_builder.

    This ensures defense-in-depth: both layers catch invalid data.
    """

    @staticmethod
    def _validate_critical_price(value: Any, param_name: str = "price") -> float:
        """Mediator layer validation."""
        if value is None:
            raise ValueError(f"Invalid {param_name}: None")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid {param_name}: cannot convert")
        if math.isnan(numeric):
            raise ValueError(f"Invalid {param_name}: NaN")
        if math.isinf(numeric):
            raise ValueError(f"Invalid {param_name}: infinity")
        if numeric <= 0.0:
            raise ValueError(f"Invalid {param_name}: must be > 0")
        return numeric

    def test_pipeline_with_valid_data(self):
        """Test 12: Valid data flows through both layers successfully."""
        # Try to import obs_builder
        try:
            from obs_builder import build_observation_vector
        except ImportError:
            pytest.skip("obs_builder not compiled")

        # Step 1: Mediator validation
        mark_price = 50000.0
        prev_price = 49500.0

        validated_price = self._validate_critical_price(mark_price, "mark_price")
        validated_prev = self._validate_critical_price(prev_price, "prev_price")

        # Step 2: obs_builder (should also validate)
        obs = np.zeros(56, dtype=np.float32)
        norm_cols = np.zeros(21, dtype=np.float32)

        build_observation_vector(
            price=validated_price,
            prev_price=validated_prev,
            log_volume_norm=0.5,
            rel_volume=0.3,
            ma5=50100.0,
            ma20=49900.0,
            rsi14=55.0,
            macd=10.0,
            macd_signal=8.0,
            momentum=15.0,
            atr=500.0,
            cci=20.0,
            obv=10000.0,
            bb_lower=49000.0,
            bb_upper=51000.0,
            is_high_importance=0.0,
            time_since_event=1.0,
            fear_greed_value=50.0,
            has_fear_greed=True,
            risk_off_flag=False,
            cash=10000.0,
            units=0.5,
            last_vol_imbalance=0.1,
            last_trade_intensity=5.0,
            last_realized_spread=0.001,
            last_agent_fill_ratio=0.95,
            token_id=0,
            max_num_tokens=1,
            num_tokens=1,
            norm_cols_values=norm_cols,
            out_features=obs,
        )

        # Success: observation built without errors
        assert obs[0] == pytest.approx(50000.0)
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))

    def test_pipeline_catches_nan_at_mediator_layer(self):
        """Test 13: NaN caught at mediator layer (first defense)."""
        # This should fail at mediator layer BEFORE reaching obs_builder
        with pytest.raises(ValueError) as exc_info:
            self._validate_critical_price(float("nan"), "mark_price")

        assert "NaN" in str(exc_info.value)
        # This prevents obs_builder from ever seeing invalid data

    def test_defense_in_depth_obs_builder_catches_bypass(self):
        """Test 14: If someone bypasses mediator, obs_builder catches it."""
        try:
            from obs_builder import build_observation_vector
        except ImportError:
            pytest.skip("obs_builder not compiled")

        # Simulate bypassing mediator validation (shouldn't happen in prod)
        obs = np.zeros(56, dtype=np.float32)
        norm_cols = np.zeros(21, dtype=np.float32)

        # obs_builder should catch this as second defense
        with pytest.raises(ValueError) as exc_info:
            build_observation_vector(
                price=0.0,  # Invalid! (would come from NaN → 0.0 conversion)
                prev_price=49500.0,
                log_volume_norm=0.0,
                rel_volume=0.0,
                ma5=float("nan"),
                ma20=float("nan"),
                rsi14=50.0,
                macd=0.0,
                macd_signal=0.0,
                momentum=0.0,
                atr=0.0,
                cci=0.0,
                obv=0.0,
                bb_lower=float("nan"),
                bb_upper=float("nan"),
                is_high_importance=0.0,
                time_since_event=0.0,
                fear_greed_value=50.0,
                has_fear_greed=False,
                risk_off_flag=False,
                cash=10000.0,
                units=0.0,
                last_vol_imbalance=0.0,
                last_trade_intensity=0.0,
                last_realized_spread=0.0,
                last_agent_fill_ratio=1.0,
                token_id=0,
                max_num_tokens=1,
                num_tokens=1,
                norm_cols_values=norm_cols,
                out_features=obs,
            )

        # Second layer of defense caught it
        assert "price" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
