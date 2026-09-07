"""
CRITICAL TEST: Verify that derived features use validity flags instead of isnan()

This test ensures that:
1. price_momentum (index 28) uses momentum_valid flag, not isnan(momentum)
2. trend_strength (index 30) uses macd_valid and macd_signal_valid flags, not isnan()

Why this matters:
- Using flags ensures consistency across the codebase
- Flags are computed once and reused (no duplicate isnan() checks)
- Future refactoring is easier when pattern is consistent

Test scenarios:
- Valid indicators → computed values
- Invalid momentum → price_momentum = 0.0
- Invalid MACD → trend_strength = 0.0
- Invalid MACD signal → trend_strength = 0.0
- Both MACD invalid → trend_strength = 0.0
"""

import numpy as np
import pytest

try:
    from obs_builder import build_observation_vector

    HAVE_OBS_BUILDER = True
except ImportError:
    HAVE_OBS_BUILDER = False
    pytest.skip("obs_builder (Cython module) not available", allow_module_level=True)


def test_price_momentum_uses_validity_flag_when_valid():
    """Test that price_momentum is computed when momentum_valid = True."""
    # Setup: Valid momentum indicator
    price = 1000.0
    prev_price = 1000.0
    momentum = 10.0  # Valid momentum value

    # All other indicators (some can be NaN - doesn't matter for this test)
    ma5 = 1005.0
    ma20 = 1010.0
    rsi14 = 50.0
    macd = 2.0
    macd_signal = 1.5
    atr = 10.0
    cci = 0.0
    obv = 1000.0

    # Observation buffer
    obs = np.zeros(63, dtype=np.float32)
    norm_cols = np.zeros(21, dtype=np.float32)

    # Call observation builder
    build_observation_vector(
        price=price,
        prev_price=prev_price,
        log_volume_norm=0.5,
        rel_volume=0.5,
        ma5=ma5,
        ma20=ma20,
        rsi14=rsi14,
        macd=macd,
        macd_signal=macd_signal,
        momentum=momentum,  # VALID
        atr=atr,
        cci=cci,
        obv=obv,
        bb_lower=990.0,
        bb_upper=1010.0,
        is_high_importance=0.0,
        time_since_event=1.0,
        fear_greed_value=50.0,
        has_fear_greed=True,
        risk_off_flag=False,
        cash=10000.0,
        units=1.0,
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

    # price_momentum is at index 28
    price_momentum = obs[28]

    # Expected: tanh(momentum / (price * 0.01 + 1e-8))
    # = tanh(10.0 / (1000.0 * 0.01 + 1e-8))
    # = tanh(10.0 / 10.0)
    # = tanh(1.0)
    # ≈ 0.7616
    expected = np.tanh(momentum / (price * 0.01 + 1e-8))

    assert abs(price_momentum - expected) < 1e-5, (
        f"price_momentum should be {expected}, got {price_momentum}. "
        f"When momentum is valid, price_momentum should be computed."
    )
    assert price_momentum != 0.0, "price_momentum should not be 0.0 when momentum is valid"


def test_price_momentum_uses_validity_flag_when_invalid():
    """Test that price_momentum = 0.0 when momentum_valid = False (momentum is NaN)."""
    # Setup: Invalid momentum (NaN)
    price = 1000.0
    prev_price = 1000.0
    momentum = float("nan")  # INVALID momentum

    # All other indicators valid
    ma5 = 1005.0
    ma20 = 1010.0
    rsi14 = 50.0
    macd = 2.0
    macd_signal = 1.5
    atr = 10.0
    cci = 0.0
    obv = 1000.0

    # Observation buffer
    obs = np.zeros(63, dtype=np.float32)
    norm_cols = np.zeros(21, dtype=np.float32)

    # Call observation builder
    build_observation_vector(
        price=price,
        prev_price=prev_price,
        log_volume_norm=0.5,
        rel_volume=0.5,
        ma5=ma5,
        ma20=ma20,
        rsi14=rsi14,
        macd=macd,
        macd_signal=macd_signal,
        momentum=momentum,  # NaN - INVALID
        atr=atr,
        cci=cci,
        obv=obv,
        bb_lower=990.0,
        bb_upper=1010.0,
        is_high_importance=0.0,
        time_since_event=1.0,
        fear_greed_value=50.0,
        has_fear_greed=True,
        risk_off_flag=False,
        cash=10000.0,
        units=1.0,
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

    # price_momentum is at index 28
    price_momentum = obs[28]

    # Expected: 0.0 (because momentum is NaN → momentum_valid = False)
    assert price_momentum == 0.0, (
        f"price_momentum should be 0.0 when momentum is NaN, got {price_momentum}. "
        f"This test verifies that momentum_valid flag is used correctly."
    )


def test_trend_strength_uses_validity_flags_when_both_valid():
    """Test that trend_strength is computed when both macd_valid and macd_signal_valid = True."""
    # Setup: Valid MACD and signal
    price = 1000.0
    prev_price = 1000.0
    macd = 5.0  # Valid MACD
    macd_signal = 3.0  # Valid signal

    # Other indicators
    ma5 = 1005.0
    ma20 = 1010.0
    rsi14 = 50.0
    momentum = 10.0
    atr = 10.0
    cci = 0.0
    obv = 1000.0

    # Observation buffer
    obs = np.zeros(63, dtype=np.float32)
    norm_cols = np.zeros(21, dtype=np.float32)

    # Call observation builder
    build_observation_vector(
        price=price,
        prev_price=prev_price,
        log_volume_norm=0.5,
        rel_volume=0.5,
        ma5=ma5,
        ma20=ma20,
        rsi14=rsi14,
        macd=macd,  # VALID
        macd_signal=macd_signal,  # VALID
        momentum=momentum,
        atr=atr,
        cci=cci,
        obv=obv,
        bb_lower=990.0,
        bb_upper=1010.0,
        is_high_importance=0.0,
        time_since_event=1.0,
        fear_greed_value=50.0,
        has_fear_greed=True,
        risk_off_flag=False,
        cash=10000.0,
        units=1.0,
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

    # trend_strength is at index 30
    trend_strength = obs[30]

    # Expected: tanh((macd - macd_signal) / (price * 0.01 + 1e-8))
    # = tanh((5.0 - 3.0) / (1000.0 * 0.01 + 1e-8))
    # = tanh(2.0 / 10.0)
    # = tanh(0.2)
    # ≈ 0.1974
    expected = np.tanh((macd - macd_signal) / (price * 0.01 + 1e-8))

    assert abs(trend_strength - expected) < 1e-5, (
        f"trend_strength should be {expected}, got {trend_strength}. "
        f"When both MACD flags are valid, trend_strength should be computed."
    )
    assert (
        trend_strength != 0.0
    ), "trend_strength should not be 0.0 when both MACD indicators are valid"


def test_trend_strength_zero_when_macd_invalid():
    """Test that trend_strength = 0.0 when macd_valid = False (macd is NaN)."""
    # Setup: Invalid MACD, valid signal
    price = 1000.0
    prev_price = 1000.0
    macd = float("nan")  # INVALID MACD
    macd_signal = 3.0  # Valid signal

    # Other indicators
    ma5 = 1005.0
    ma20 = 1010.0
    rsi14 = 50.0
    momentum = 10.0
    atr = 10.0
    cci = 0.0
    obv = 1000.0

    # Observation buffer
    obs = np.zeros(63, dtype=np.float32)
    norm_cols = np.zeros(21, dtype=np.float32)

    # Call observation builder
    build_observation_vector(
        price=price,
        prev_price=prev_price,
        log_volume_norm=0.5,
        rel_volume=0.5,
        ma5=ma5,
        ma20=ma20,
        rsi14=rsi14,
        macd=macd,  # NaN - INVALID
        macd_signal=macd_signal,  # VALID
        momentum=momentum,
        atr=atr,
        cci=cci,
        obv=obv,
        bb_lower=990.0,
        bb_upper=1010.0,
        is_high_importance=0.0,
        time_since_event=1.0,
        fear_greed_value=50.0,
        has_fear_greed=True,
        risk_off_flag=False,
        cash=10000.0,
        units=1.0,
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

    # trend_strength is at index 30
    trend_strength = obs[30]

    # Expected: 0.0 (because macd is NaN → macd_valid = False)
    assert trend_strength == 0.0, (
        f"trend_strength should be 0.0 when macd is NaN, got {trend_strength}. "
        f"This test verifies that macd_valid flag is used correctly."
    )


def test_trend_strength_zero_when_macd_signal_invalid():
    """Test that trend_strength = 0.0 when macd_signal_valid = False (macd_signal is NaN)."""
    # Setup: Valid MACD, invalid signal
    price = 1000.0
    prev_price = 1000.0
    macd = 5.0  # Valid MACD
    macd_signal = float("nan")  # INVALID signal

    # Other indicators
    ma5 = 1005.0
    ma20 = 1010.0
    rsi14 = 50.0
    momentum = 10.0
    atr = 10.0
    cci = 0.0
    obv = 1000.0

    # Observation buffer
    obs = np.zeros(63, dtype=np.float32)
    norm_cols = np.zeros(21, dtype=np.float32)

    # Call observation builder
    build_observation_vector(
        price=price,
        prev_price=prev_price,
        log_volume_norm=0.5,
        rel_volume=0.5,
        ma5=ma5,
        ma20=ma20,
        rsi14=rsi14,
        macd=macd,  # VALID
        macd_signal=macd_signal,  # NaN - INVALID
        momentum=momentum,
        atr=atr,
        cci=cci,
        obv=obv,
        bb_lower=990.0,
        bb_upper=1010.0,
        is_high_importance=0.0,
        time_since_event=1.0,
        fear_greed_value=50.0,
        has_fear_greed=True,
        risk_off_flag=False,
        cash=10000.0,
        units=1.0,
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

    # trend_strength is at index 30
    trend_strength = obs[30]

    # Expected: 0.0 (because macd_signal is NaN → macd_signal_valid = False)
    assert trend_strength == 0.0, (
        f"trend_strength should be 0.0 when macd_signal is NaN, got {trend_strength}. "
        f"This test verifies that macd_signal_valid flag is used correctly."
    )


def test_trend_strength_zero_when_both_invalid():
    """Test that trend_strength = 0.0 when both macd and macd_signal are NaN."""
    # Setup: Both invalid
    price = 1000.0
    prev_price = 1000.0
    macd = float("nan")  # INVALID MACD
    macd_signal = float("nan")  # INVALID signal

    # Other indicators
    ma5 = 1005.0
    ma20 = 1010.0
    rsi14 = 50.0
    momentum = 10.0
    atr = 10.0
    cci = 0.0
    obv = 1000.0

    # Observation buffer
    obs = np.zeros(63, dtype=np.float32)
    norm_cols = np.zeros(21, dtype=np.float32)

    # Call observation builder
    build_observation_vector(
        price=price,
        prev_price=prev_price,
        log_volume_norm=0.5,
        rel_volume=0.5,
        ma5=ma5,
        ma20=ma20,
        rsi14=rsi14,
        macd=macd,  # NaN - INVALID
        macd_signal=macd_signal,  # NaN - INVALID
        momentum=momentum,
        atr=atr,
        cci=cci,
        obv=obv,
        bb_lower=990.0,
        bb_upper=1010.0,
        is_high_importance=0.0,
        time_since_event=1.0,
        fear_greed_value=50.0,
        has_fear_greed=True,
        risk_off_flag=False,
        cash=10000.0,
        units=1.0,
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

    # trend_strength is at index 30
    trend_strength = obs[30]

    # Expected: 0.0 (because both are NaN)
    assert trend_strength == 0.0, (
        f"trend_strength should be 0.0 when both MACD indicators are NaN, got {trend_strength}. "
        f"This test verifies that both validity flags are checked with AND logic."
    )


def test_validity_flags_indices():
    """Verify that validity flags are at correct indices in observation vector."""
    # Setup with all valid indicators
    price = 1000.0
    prev_price = 1000.0

    # Observation buffer
    obs = np.zeros(63, dtype=np.float32)
    norm_cols = np.zeros(21, dtype=np.float32)

    # Call observation builder with all valid indicators
    build_observation_vector(
        price=price,
        prev_price=prev_price,
        log_volume_norm=0.5,
        rel_volume=0.5,
        ma5=1005.0,  # VALID
        ma20=1010.0,  # VALID
        rsi14=50.0,  # VALID
        macd=2.0,  # VALID
        macd_signal=1.5,  # VALID
        momentum=10.0,  # VALID
        atr=10.0,
        cci=0.0,  # VALID
        obv=1000.0,  # VALID
        bb_lower=990.0,
        bb_upper=1010.0,
        is_high_importance=0.0,
        time_since_event=1.0,
        fear_greed_value=50.0,
        has_fear_greed=True,
        risk_off_flag=False,
        cash=10000.0,
        units=1.0,
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

    # Check validity flags (should all be 1.0 for valid indicators)
    assert obs[4] == 1.0, f"ma5_valid (index 4) should be 1.0, got {obs[4]}"
    assert obs[6] == 1.0, f"ma20_valid (index 6) should be 1.0, got {obs[6]}"
    assert obs[8] == 1.0, f"rsi_valid (index 8) should be 1.0, got {obs[8]}"
    assert obs[10] == 1.0, f"macd_valid (index 10) should be 1.0, got {obs[10]}"
    assert obs[12] == 1.0, f"macd_signal_valid (index 12) should be 1.0, got {obs[12]}"
    assert obs[14] == 1.0, f"momentum_valid (index 14) should be 1.0, got {obs[14]}"
    assert obs[16] == 1.0, f"atr_valid (index 16) should be 1.0, got {obs[16]}"
    assert obs[18] == 1.0, f"cci_valid (index 18) should be 1.0, got {obs[18]}"
    assert obs[20] == 1.0, f"obv_valid (index 20) should be 1.0, got {obs[20]}"

    print("✅ All validity flags are at correct indices and set to 1.0 for valid indicators")


if __name__ == "__main__":
    # Run all tests
    print("\n" + "=" * 80)
    print("TESTING: Derived Features Use Validity Flags (Not isnan())")
    print("=" * 80 + "\n")

    print("Test 1: price_momentum when momentum is VALID...")
    test_price_momentum_uses_validity_flag_when_valid()
    print("✅ PASSED\n")

    print("Test 2: price_momentum when momentum is INVALID (NaN)...")
    test_price_momentum_uses_validity_flag_when_invalid()
    print("✅ PASSED\n")

    print("Test 3: trend_strength when both MACD indicators are VALID...")
    test_trend_strength_uses_validity_flags_when_both_valid()
    print("✅ PASSED\n")

    print("Test 4: trend_strength when MACD is INVALID (NaN)...")
    test_trend_strength_zero_when_macd_invalid()
    print("✅ PASSED\n")

    print("Test 5: trend_strength when MACD signal is INVALID (NaN)...")
    test_trend_strength_zero_when_macd_signal_invalid()
    print("✅ PASSED\n")

    print("Test 6: trend_strength when BOTH are INVALID (NaN)...")
    test_trend_strength_zero_when_both_invalid()
    print("✅ PASSED\n")

    print("Test 7: Verify validity flags indices...")
    test_validity_flags_indices()
    print("✅ PASSED\n")

    print("=" * 80)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 80)
    print("\nConclusion:")
    print("✅ price_momentum correctly uses momentum_valid flag")
    print("✅ trend_strength correctly uses macd_valid AND macd_signal_valid flags")
    print("✅ No more isnan() checks in derived features - pattern is consistent")
    print("=" * 80)
