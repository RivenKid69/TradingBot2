#!/usr/bin/env python3
"""
Comprehensive tests for Bollinger Bands validation in obs_builder.

This test suite verifies the fix for incomplete Bollinger Bands validation:
- BOTH bb_lower and bb_upper are validated (not just bb_lower)
- Finitude is checked (not just NaN, but also Inf)
- Logical consistency is enforced (bb_upper >= bb_lower)
- Derived features (bb_position, bb_width, bb_squeeze) never become NaN
- Defense-in-depth validation prevents all edge cases

Test coverage:
1. ✓ Both bb_lower and bb_upper are NaN (early bars)
2. ✓ Only bb_lower is NaN, bb_upper is valid (asymmetric)
3. ✓ Only bb_upper is NaN, bb_lower is valid (CRITICAL - the original bug)
4. ✓ Both bands are Inf
5. ✓ Only bb_lower is Inf
6. ✓ Only bb_upper is Inf
7. ✓ bb_upper < bb_lower (invalid ordering)
8. ✓ bb_width = 0 (zero volatility)
9. ✓ Both bands valid and consistent (happy path)
10. ✓ Extreme values but still valid

Research references:
- "Bollinger Bands" (John Bollinger): Upper band >= Lower band by definition
- "Defense in Depth" (OWASP): Multiple validation layers
- "Data Validation Best Practices" (Cube Software): Validate all inputs
- "Fail-fast validation" (Martin Fowler): Catch errors early
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from obs_builder import build_observation_vector
    HAVE_OBS_BUILDER = True
except ImportError:
    HAVE_OBS_BUILDER = False
    pytest.skip("obs_builder not available, skipping tests", allow_module_level=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_valid_inputs(**overrides):
    """
    Create a valid set of inputs for build_observation_vector.

    All inputs have safe defaults. Use overrides to test specific edge cases.
    """
    defaults = {
        "price": 50000.0,
        "prev_price": 49900.0,
        "log_volume_norm": 0.5,
        "rel_volume": 0.3,
        "ma5": 50100.0,
        "ma20": 50050.0,
        "rsi14": 55.0,
        "macd": 10.0,
        "macd_signal": 8.0,
        "momentum": 5.0,
        "atr": 500.0,
        "cci": 25.0,
        "obv": 10000.0,
        "bb_lower": 49500.0,  # Valid lower band
        "bb_upper": 50500.0,  # Valid upper band
        "is_high_importance": 0.0,
        "time_since_event": 2.0,
        "fear_greed_value": 60.0,
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
    }

    defaults.update(overrides)
    return defaults


def build_obs_with_inputs(**kwargs):
    """
    Build observation vector with given inputs.

    Returns:
        np.ndarray: Observation vector (56 features)
    """
    inputs = create_valid_inputs(**kwargs)

    # Create norm_cols and output array
    norm_cols = np.zeros(21, dtype=np.float32)
    obs = np.zeros(56, dtype=np.float32)

    build_observation_vector(
        float(inputs["price"]),
        float(inputs["prev_price"]),
        float(inputs["log_volume_norm"]),
        float(inputs["rel_volume"]),
        float(inputs["ma5"]),
        float(inputs["ma20"]),
        float(inputs["rsi14"]),
        float(inputs["macd"]),
        float(inputs["macd_signal"]),
        float(inputs["momentum"]),
        float(inputs["atr"]),
        float(inputs["cci"]),
        float(inputs["obv"]),
        float(inputs["bb_lower"]),
        float(inputs["bb_upper"]),
        float(inputs["is_high_importance"]),
        float(inputs["time_since_event"]),
        float(inputs["fear_greed_value"]),
        bool(inputs["has_fear_greed"]),
        bool(inputs["risk_off_flag"]),
        float(inputs["cash"]),
        float(inputs["units"]),
        float(inputs["last_vol_imbalance"]),
        float(inputs["last_trade_intensity"]),
        float(inputs["last_realized_spread"]),
        float(inputs["last_agent_fill_ratio"]),
        int(inputs["token_id"]),
        int(inputs["max_num_tokens"]),
        int(inputs["num_tokens"]),
        norm_cols,
        obs,
    )

    return obs


def assert_no_nan_or_inf(obs, test_name=""):
    """Assert that observation has no NaN or Inf values."""
    has_nan = np.any(np.isnan(obs))
    has_inf = np.any(np.isinf(obs))

    if has_nan:
        nan_indices = np.where(np.isnan(obs))[0]
        raise AssertionError(
            f"{test_name}: Observation contains NaN at indices {nan_indices.tolist()}"
        )

    if has_inf:
        inf_indices = np.where(np.isinf(obs))[0]
        raise AssertionError(
            f"{test_name}: Observation contains Inf at indices {inf_indices.tolist()}"
        )


def get_bb_features(obs):
    """
    Extract Bollinger Bands related features from observation.

    Feature layout (56-feature observation):
    - bb_squeeze: index 23 (derived from bb_width, measures volatility regime)
    - bb_position: index 25 (price position within bands)
    - bb_width: index 26 (normalized band width)

    Returns:
        dict: {feature_name: value}
    """
    return {
        "bb_squeeze": obs[23],
        "bb_position": obs[25],
        "bb_width": obs[26],
    }


# =============================================================================
# TEST CASES
# =============================================================================

def test_both_bands_nan():
    """Test 1: Both bb_lower and bb_upper are NaN (early bars, typical scenario)."""
    obs = build_obs_with_inputs(
        bb_lower=float('nan'),
        bb_upper=float('nan'),
    )

    assert_no_nan_or_inf(obs, "Test 1: Both bands NaN")

    bb_features = get_bb_features(obs)

    # Expected defaults when bands not available
    assert bb_features["bb_squeeze"] == 0.0, "bb_squeeze should default to 0.0"
    assert bb_features["bb_position"] == 0.5, "bb_position should default to 0.5 (middle)"
    assert bb_features["bb_width"] == 0.0, "bb_width should default to 0.0"

    print("✓ Test 1 passed: Both bands NaN → safe defaults")


def test_only_lower_nan():
    """Test 2: Only bb_lower is NaN, bb_upper is valid (asymmetric case)."""
    obs = build_obs_with_inputs(
        bb_lower=float('nan'),
        bb_upper=50500.0,
    )

    assert_no_nan_or_inf(obs, "Test 2: Only bb_lower NaN")

    bb_features = get_bb_features(obs)

    # Should use safe defaults because validation requires BOTH bands
    assert bb_features["bb_squeeze"] == 0.0
    assert bb_features["bb_position"] == 0.5
    assert bb_features["bb_width"] == 0.0

    print("✓ Test 2 passed: Only bb_lower NaN → safe defaults")


def test_only_upper_nan():
    """
    Test 3: Only bb_upper is NaN, bb_lower is valid (CRITICAL - original bug).

    This is the exact scenario that caused the vulnerability:
    - Old code: bb_valid = not isnan(bb_lower) → True
    - Then: bb_squeeze = tanh((NaN - 49500.0) / ...) → NaN
    - Result: NaN propagates to observation vector

    With the fix, this should now use safe defaults.
    """
    obs = build_obs_with_inputs(
        bb_lower=49500.0,
        bb_upper=float('nan'),
    )

    assert_no_nan_or_inf(obs, "Test 3: Only bb_upper NaN (CRITICAL)")

    bb_features = get_bb_features(obs)

    # CRITICAL: Must use safe defaults, NOT attempt calculation with NaN
    assert bb_features["bb_squeeze"] == 0.0, "bb_squeeze must not be NaN!"
    assert bb_features["bb_position"] == 0.5, "bb_position must not be NaN!"
    assert bb_features["bb_width"] == 0.0, "bb_width must not be NaN!"

    print("✓ Test 3 passed: Only bb_upper NaN → safe defaults (CRITICAL FIX VERIFIED)")


def test_both_bands_inf():
    """Test 4: Both bb_lower and bb_upper are Inf (calculation overflow)."""
    obs = build_obs_with_inputs(
        bb_lower=float('inf'),
        bb_upper=float('inf'),
    )

    assert_no_nan_or_inf(obs, "Test 4: Both bands Inf")

    bb_features = get_bb_features(obs)

    # Should use safe defaults because bands are not finite
    assert bb_features["bb_squeeze"] == 0.0
    assert bb_features["bb_position"] == 0.5
    assert bb_features["bb_width"] == 0.0

    print("✓ Test 4 passed: Both bands Inf → safe defaults")


def test_only_lower_inf():
    """Test 5: Only bb_lower is Inf, bb_upper is valid."""
    obs = build_obs_with_inputs(
        bb_lower=float('inf'),
        bb_upper=50500.0,
    )

    assert_no_nan_or_inf(obs, "Test 5: Only bb_lower Inf")

    bb_features = get_bb_features(obs)

    assert bb_features["bb_squeeze"] == 0.0
    assert bb_features["bb_position"] == 0.5
    assert bb_features["bb_width"] == 0.0

    print("✓ Test 5 passed: Only bb_lower Inf → safe defaults")


def test_only_upper_inf():
    """Test 6: Only bb_upper is Inf, bb_lower is valid."""
    obs = build_obs_with_inputs(
        bb_lower=49500.0,
        bb_upper=float('inf'),
    )

    assert_no_nan_or_inf(obs, "Test 6: Only bb_upper Inf")

    bb_features = get_bb_features(obs)

    assert bb_features["bb_squeeze"] == 0.0
    assert bb_features["bb_position"] == 0.5
    assert bb_features["bb_width"] == 0.0

    print("✓ Test 6 passed: Only bb_upper Inf → safe defaults")


def test_invalid_ordering():
    """
    Test 7: bb_upper < bb_lower (invalid ordering, data corruption).

    By definition, Bollinger upper band should be >= lower band.
    If bb_upper < bb_lower, this indicates data corruption and should be caught.
    """
    obs = build_obs_with_inputs(
        bb_lower=50500.0,  # Higher value
        bb_upper=49500.0,  # Lower value (INVALID!)
    )

    assert_no_nan_or_inf(obs, "Test 7: Invalid band ordering")

    bb_features = get_bb_features(obs)

    # Should detect invalid ordering and use safe defaults
    assert bb_features["bb_squeeze"] == 0.0
    assert bb_features["bb_position"] == 0.5
    assert bb_features["bb_width"] == 0.0

    print("✓ Test 7 passed: Invalid ordering (bb_upper < bb_lower) → safe defaults")


def test_zero_width():
    """
    Test 8: bb_width = 0 (zero volatility, all prices identical).

    When volatility is zero, upper and lower bands collapse to the same value.
    This is a valid state but requires careful handling to avoid division by zero.
    """
    price = 50000.0
    obs = build_obs_with_inputs(
        price=price,
        bb_lower=price,  # Same as price
        bb_upper=price,  # Same as price (zero width)
    )

    assert_no_nan_or_inf(obs, "Test 8: Zero width bands")

    bb_features = get_bb_features(obs)

    # bb_squeeze should be 0.0 (tanh(0) = 0)
    assert bb_features["bb_squeeze"] == 0.0

    # bb_position should default to 0.5 (width too small)
    assert bb_features["bb_position"] == 0.5

    # bb_width should be near 0.0
    assert abs(bb_features["bb_width"]) < 0.01

    print("✓ Test 8 passed: Zero width bands → safe handling")


def test_valid_bands():
    """
    Test 9: Both bands valid and consistent (happy path).

    This is the normal case where both bands are finite and properly ordered.
    """
    price = 50000.0
    bb_lower = 49500.0
    bb_upper = 50500.0

    obs = build_obs_with_inputs(
        price=price,
        bb_lower=bb_lower,
        bb_upper=bb_upper,
    )

    assert_no_nan_or_inf(obs, "Test 9: Valid bands")

    bb_features = get_bb_features(obs)

    # bb_squeeze should be non-zero (bands have width)
    assert bb_features["bb_squeeze"] != 0.0, "bb_squeeze should be calculated"

    # bb_position should be calculated (not default)
    # Price at middle: (50000 - 49500) / (50500 - 49500) = 0.5
    assert 0.4 < bb_features["bb_position"] < 0.6, "bb_position should be ~0.5"

    # bb_width should be non-zero
    assert bb_features["bb_width"] > 0.0, "bb_width should be positive"

    print("✓ Test 9 passed: Valid bands → correct calculations")


def test_extreme_but_valid():
    """
    Test 10: Extreme values but still valid (stress test).

    Tests that validation doesn't reject valid but unusual values.
    """
    price = 100000.0
    bb_lower = 95000.0
    bb_upper = 105000.0

    obs = build_obs_with_inputs(
        price=price,
        bb_lower=bb_lower,
        bb_upper=bb_upper,
    )

    assert_no_nan_or_inf(obs, "Test 10: Extreme but valid")

    bb_features = get_bb_features(obs)

    # Should calculate normally
    assert bb_features["bb_squeeze"] != 0.0
    assert bb_features["bb_width"] > 0.0

    print("✓ Test 10 passed: Extreme but valid values → correct calculations")


def test_negative_inf_bands():
    """Test 11: Negative infinity bands (edge case)."""
    obs = build_obs_with_inputs(
        bb_lower=float('-inf'),
        bb_upper=float('-inf'),
    )

    assert_no_nan_or_inf(obs, "Test 11: Negative Inf bands")

    bb_features = get_bb_features(obs)

    assert bb_features["bb_squeeze"] == 0.0
    assert bb_features["bb_position"] == 0.5
    assert bb_features["bb_width"] == 0.0

    print("✓ Test 11 passed: Negative Inf bands → safe defaults")


def test_mixed_inf_nan():
    """Test 12: Mix of Inf and NaN (chaos scenario)."""
    obs = build_obs_with_inputs(
        bb_lower=float('nan'),
        bb_upper=float('inf'),
    )

    assert_no_nan_or_inf(obs, "Test 12: Mixed NaN and Inf")

    bb_features = get_bb_features(obs)

    assert bb_features["bb_squeeze"] == 0.0
    assert bb_features["bb_position"] == 0.5
    assert bb_features["bb_width"] == 0.0

    print("✓ Test 12 passed: Mixed NaN/Inf → safe defaults")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("BOLLINGER BANDS VALIDATION TESTS")
    print("=" * 80)
    print()

    try:
        test_both_bands_nan()
        test_only_lower_nan()
        test_only_upper_nan()  # CRITICAL TEST
        test_both_bands_inf()
        test_only_lower_inf()
        test_only_upper_inf()
        test_invalid_ordering()
        test_zero_width()
        test_valid_bands()
        test_extreme_but_valid()
        test_negative_inf_bands()
        test_mixed_inf_nan()

        print()
        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print()
        print("Summary:")
        print("- Incomplete validation FIXED: Both bb_lower and bb_upper validated")
        print("- Finitude checked: Not just NaN, but also Inf")
        print("- Logical consistency enforced: bb_upper >= bb_lower")
        print("- Defense-in-depth: Multiple validation layers")
        print("- All edge cases covered: NaN, Inf, invalid ordering, zero width")
        print("- No NaN propagation: All features remain finite")

    except AssertionError as e:
        print()
        print("=" * 80)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 80)
        raise
    except Exception as e:
        print()
        print("=" * 80)
        print(f"❌ ERROR: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        raise
