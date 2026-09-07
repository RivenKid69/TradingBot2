"""
Unit tests for validity flags implementation (Issue #2 COMPLETE FIX).

Tests the new methods:
- Mediator._get_safe_float_with_validity()
- Mediator._extract_norm_cols() with validity tracking

Ensures model can distinguish missing data from zero values.
"""

import math
import numpy as np
import pytest
from unittest.mock import MagicMock


def test_get_safe_float_with_validity_valid_value():
    """Test that valid values return (value, True)."""
    from mediator import Mediator

    row = {"cvd_24h": 1.5, "garch_14d": -0.5, "ret_12h": 0.0}

    # Positive value
    value, is_valid = Mediator._get_safe_float_with_validity(row, "cvd_24h", default=0.0)
    assert value == 1.5, "Should return actual value"
    assert is_valid is True, "Should be marked as valid"

    # Negative value
    value, is_valid = Mediator._get_safe_float_with_validity(row, "garch_14d", default=0.0)
    assert value == -0.5, "Should return negative value"
    assert is_valid is True, "Negative values are valid"

    # Zero value (CRITICAL: should be valid!)
    value, is_valid = Mediator._get_safe_float_with_validity(row, "ret_12h", default=0.0)
    assert value == 0.0, "Should return zero"
    assert is_valid is True, "Zero is a valid value!"


def test_get_safe_float_with_validity_nan_handling():
    """Test that NaN values return (default, False)."""
    from mediator import Mediator

    row = {"nan_feature": float("nan"), "inf_feature": float("inf"), "neg_inf": float("-inf")}

    # NaN
    value, is_valid = Mediator._get_safe_float_with_validity(row, "nan_feature", default=0.0)
    assert value == 0.0, "NaN should fallback to default"
    assert is_valid is False, "NaN should be marked as invalid"

    # Positive Infinity
    value, is_valid = Mediator._get_safe_float_with_validity(row, "inf_feature", default=0.0)
    assert value == 0.0, "Inf should fallback to default"
    assert is_valid is False, "Inf should be marked as invalid"

    # Negative Infinity
    value, is_valid = Mediator._get_safe_float_with_validity(row, "neg_inf", default=0.0)
    assert value == 0.0, "Neg Inf should fallback to default"
    assert is_valid is False, "Neg Inf should be marked as invalid"


def test_get_safe_float_with_validity_none_handling():
    """Test that None values return (default, False)."""
    from mediator import Mediator

    row = {"none_feature": None, "present_feature": 1.0}

    # None value
    value, is_valid = Mediator._get_safe_float_with_validity(row, "none_feature", default=0.0)
    assert value == 0.0, "None should fallback to default"
    assert is_valid is False, "None should be marked as invalid"

    # Missing key (returns None from dict.get())
    value, is_valid = Mediator._get_safe_float_with_validity(row, "missing_key", default=0.0)
    assert value == 0.0, "Missing key should fallback to default"
    assert is_valid is False, "Missing key should be marked as invalid"

    # None row
    value, is_valid = Mediator._get_safe_float_with_validity(None, "any_key", default=0.0)
    assert value == 0.0, "None row should fallback to default"
    assert is_valid is False, "None row should be marked as invalid"


def test_get_safe_float_with_validity_range_validation():
    """Test that range validation sets is_valid=False correctly."""
    from mediator import Mediator

    row = {"value": 150.0}

    # Within range
    value, is_valid = Mediator._get_safe_float_with_validity(
        row, "value", default=0.0, min_value=0.0, max_value=200.0
    )
    assert value == 150.0, "Value within range should be returned"
    assert is_valid is True, "Value within range should be valid"

    # Below minimum
    value, is_valid = Mediator._get_safe_float_with_validity(
        row, "value", default=0.0, min_value=200.0
    )
    assert value == 0.0, "Value below min should fallback to default"
    assert is_valid is False, "Value below min should be invalid"

    # Above maximum
    value, is_valid = Mediator._get_safe_float_with_validity(
        row, "value", default=0.0, max_value=100.0
    )
    assert value == 0.0, "Value above max should fallback to default"
    assert is_valid is False, "Value above max should be invalid"

    # Edge case: exactly at boundary
    value, is_valid = Mediator._get_safe_float_with_validity(
        row, "value", default=0.0, min_value=150.0, max_value=150.0
    )
    assert value == 150.0, "Value at boundary should be returned"
    assert is_valid is True, "Value at boundary should be valid (inclusive)"


def test_get_safe_float_with_validity_type_conversion_error():
    """Test that type conversion errors return (default, False)."""
    from mediator import Mediator

    row = {"string_value": "not_a_number", "dict_value": {"nested": 1.0}}

    # String that can't convert to float
    value, is_valid = Mediator._get_safe_float_with_validity(row, "string_value", default=0.0)
    assert value == 0.0, "Conversion error should fallback to default"
    assert is_valid is False, "Conversion error should be marked as invalid"

    # Dict that can't convert to float
    value, is_valid = Mediator._get_safe_float_with_validity(row, "dict_value", default=0.0)
    assert value == 0.0, "Dict should fallback to default"
    assert is_valid is False, "Dict should be marked as invalid"


def test_get_safe_float_with_validity_semantic_distinction():
    """
    CRITICAL TEST: Verify that validity flags enable semantic distinction.

    This is the core fix for Issue #2. Model can now distinguish:
    - Zero value (value=0.0, is_valid=True) = "balanced volume", "no movement", etc.
    - Missing data (value=0.0, is_valid=False) = "data unavailable"
    """
    from mediator import Mediator

    # Scenario 1: CVD is genuinely zero (balanced buy/sell)
    row_zero = {"cvd_24h": 0.0}
    value_zero, is_valid_zero = Mediator._get_safe_float_with_validity(row_zero, "cvd_24h", 0.0)

    # Scenario 2: CVD is missing (NaN)
    row_nan = {"cvd_24h": float("nan")}
    value_nan, is_valid_nan = Mediator._get_safe_float_with_validity(row_nan, "cvd_24h", 0.0)

    # VALUES are the same (both 0.0)
    assert value_zero == value_nan == 0.0, "Both scenarios have value=0.0"

    # But VALIDITY is different!
    assert is_valid_zero is True, "Zero value should be VALID"
    assert is_valid_nan is False, "Missing data should be INVALID"

    # This enables model to learn:
    # - (0.0, True) → "safe to trade based on balanced volume"
    # - (0.0, False) → "data missing, reduce position or wait"


def test_extract_norm_cols_returns_tuple():
    """Test that _extract_norm_cols now returns (values, validity) tuple."""
    from mediator import Mediator

    mediator = Mediator.__new__(Mediator)

    row = {"cvd_24h": 1.0, "cvd_7d": float("nan"), "yang_zhang_48h": 0.5}

    result = mediator._extract_norm_cols(row)

    # Should return tuple
    assert isinstance(result, tuple), "_extract_norm_cols should return tuple"
    assert len(result) == 2, "Tuple should have 2 elements (values, validity)"

    values, validity = result

    # Check shapes
    assert isinstance(values, np.ndarray), "values should be ndarray"
    assert isinstance(validity, np.ndarray), "validity should be ndarray"
    assert values.shape == (21,), "values should have shape (21,)"
    assert validity.shape == (21,), "validity should have shape (21,)"
    assert values.dtype == np.float32, "values should be float32"
    assert validity.dtype == bool, "validity should be bool"


def test_extract_norm_cols_validity_tracking():
    """Test that _extract_norm_cols correctly tracks validity for each feature."""
    from mediator import Mediator

    mediator = Mediator.__new__(Mediator)

    # Mix of valid, NaN, and None values
    row = {
        "cvd_24h": 1.5,  # [0] - valid
        "cvd_7d": float("nan"),  # [1] - invalid (NaN)
        "yang_zhang_48h": 0.8,  # [2] - valid
        "yang_zhang_7d": None,  # [3] - invalid (None)
        "garch_200h": float("inf"),  # [4] - invalid (Inf)
        "garch_14d": 0.5,  # [5] - valid
        "ret_12h": 0.0,  # [6] - valid (zero is valid!)
        "ret_24h": -0.1,  # [7] - valid (negative is valid!)
        # Rest default to 0.0 with validity depending on presence
    }

    values, validity = mediator._extract_norm_cols(row)

    # Verify valid features
    assert abs(values[0] - 1.5) < 1e-6, "cvd_24h value should be 1.5"
    assert validity[0] == True, "cvd_24h should be valid"

    assert abs(values[2] - 0.8) < 1e-6, "yang_zhang_48h value should be 0.8"
    assert validity[2] == True, "yang_zhang_48h should be valid"

    assert abs(values[5] - 0.5) < 1e-6, "garch_14d value should be 0.5"
    assert validity[5] == True, "garch_14d should be valid"

    # CRITICAL: Zero is a valid value!
    assert values[6] == 0.0, "ret_12h should be 0.0"
    assert validity[6] == True, "ret_12h (zero) should be VALID"

    # Negative is valid
    assert abs(values[7] - (-0.1)) < 1e-6, "ret_24h should be -0.1"
    assert validity[7] == True, "ret_24h (negative) should be valid"

    # Verify invalid features (NaN/None/Inf → 0.0 with is_valid=False)
    assert values[1] == 0.0, "cvd_7d (NaN) should fallback to 0.0"
    assert validity[1] == False, "cvd_7d (NaN) should be INVALID"

    assert values[3] == 0.0, "yang_zhang_7d (None) should fallback to 0.0"
    assert validity[3] == False, "yang_zhang_7d (None) should be INVALID"

    assert values[4] == 0.0, "garch_200h (Inf) should fallback to 0.0"
    assert validity[4] == False, "garch_200h (Inf) should be INVALID"


def test_extract_norm_cols_all_valid():
    """Test scenario where all features are valid."""
    from mediator import Mediator

    mediator = Mediator.__new__(Mediator)

    # All 21 features present and valid
    row = {
        "cvd_24h": 0.1,
        "cvd_7d": 0.2,
        "yang_zhang_48h": 0.3,
        "yang_zhang_7d": 0.4,
        "garch_200h": 0.5,
        "garch_14d": 0.6,
        "ret_12h": 0.01,
        "ret_24h": 0.02,
        "ret_4h": 0.005,
        "sma_12000": 50000.0,
        "yang_zhang_30d": 0.7,
        "parkinson_48h": 0.8,
        "parkinson_7d": 0.9,
        "garch_30d": 0.35,
        "taker_buy_ratio": 0.52,
        "taker_buy_ratio_sma_24h": 0.51,
        "taker_buy_ratio_sma_8h": 0.50,
        "taker_buy_ratio_sma_16h": 0.505,
        "taker_buy_ratio_momentum_4h": 0.01,
        "taker_buy_ratio_momentum_8h": 0.02,
        "taker_buy_ratio_momentum_12h": 0.015,
    }

    values, validity = mediator._extract_norm_cols(row)

    # All should be valid
    assert np.all(validity), "All features should be valid when data is present"
    assert np.all(np.isfinite(values)), "All values should be finite"

    # Spot check some values
    assert abs(values[0] - 0.1) < 1e-6, "cvd_24h should be 0.1"
    assert abs(values[9] - 50000.0) < 1e-3, "sma_12000 should be 50000.0"


def test_extract_norm_cols_all_missing():
    """Test scenario where all features are missing (NaN)."""
    from mediator import Mediator

    mediator = Mediator.__new__(Mediator)

    # All features are NaN (e.g., during cold start or data outage)
    row = {f"feature_{i}": float("nan") for i in range(30)}  # Fake row with NaN

    values, validity = mediator._extract_norm_cols(row)

    # All should be invalid (features not in row)
    assert np.all(~validity), "All features should be invalid when data is missing"

    # All values should be 0.0 (default fallback)
    assert np.all(values == 0.0), "All missing features should fallback to 0.0"


def test_extract_norm_cols_partial_missing():
    """Test realistic scenario with some missing features."""
    from mediator import Mediator

    mediator = Mediator.__new__(Mediator)

    # Realistic scenario: Some features present, some missing
    # E.g., during GARCH warmup period (first 200h), GARCH features are NaN
    row = {
        "cvd_24h": 100.0,  # Valid
        "cvd_7d": 500.0,  # Valid
        "yang_zhang_48h": 0.015,  # Valid
        "yang_zhang_7d": 0.018,  # Valid
        "garch_200h": float("nan"),  # Missing (warmup)
        "garch_14d": float("nan"),  # Missing (warmup)
        "garch_30d": float("nan"),  # Missing (warmup)
        "ret_12h": 0.002,  # Valid
        "ret_24h": 0.005,  # Valid
        "ret_4h": 0.001,  # Valid
        "taker_buy_ratio": 0.55,  # Valid
        # Other features missing
    }

    values, validity = mediator._extract_norm_cols(row)

    # Check valid features
    assert validity[0] == True, "cvd_24h should be valid"
    assert validity[1] == True, "cvd_7d should be valid"
    assert validity[2] == True, "yang_zhang_48h should be valid"
    assert validity[6] == True, "ret_12h should be valid"
    assert validity[14] == True, "taker_buy_ratio should be valid"

    # Check missing features (GARCH during warmup)
    assert validity[4] == False, "garch_200h should be invalid (warmup)"
    assert validity[5] == False, "garch_14d should be invalid (warmup)"
    assert validity[13] == False, "garch_30d should be invalid (warmup)"

    # Values should still be finite (0.0 for missing)
    assert np.all(np.isfinite(values)), "All values should be finite"


def test_backward_compatibility_with_old_code():
    """
    Test that old code expecting single array can still work (temporarily).

    This test documents the breaking change and how to handle it.
    """
    from mediator import Mediator

    mediator = Mediator.__new__(Mediator)

    row = {"cvd_24h": 1.0, "garch_14d": 0.5}

    # New API returns tuple
    result = mediator._extract_norm_cols(row)
    assert isinstance(result, tuple), "New API returns tuple"

    # Old code expecting single array would need to be updated:
    # OLD: norm_cols = mediator._extract_norm_cols(row)
    # NEW: norm_cols_values, norm_cols_validity = mediator._extract_norm_cols(row)

    # Unpack tuple
    values, validity = result

    # Old code could temporarily use just values (ignore validity)
    # But this defeats the purpose of the fix!
    assert isinstance(values, np.ndarray), "Values can be used as before"
    assert values.shape == (21,), "Shape is the same"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
