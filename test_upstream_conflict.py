#!/usr/bin/env python3
"""
Test to verify the conflict between _coerce_finite and price validation.

This test demonstrates that:
1. mediator._coerce_finite() converts NaN → 0.0 (silent fallback)
2. obs_builder._validate_price() rejects 0.0 (explicit error)
3. This creates a conflict where NaN prices now raise errors instead of being silently converted
"""

import math
import numpy as np

# Import the validation logic
try:
    from obs_builder import build_observation_vector
    HAS_OBS_BUILDER = True
except ImportError:
    HAS_OBS_BUILDER = False
    print("❌ obs_builder not available, cannot run test")
    exit(1)


def _coerce_finite(value, default=0.0):
    """Replicate mediator._coerce_finite() logic."""
    if value is None:
        return float(default)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return numeric


def test_upstream_conflict():
    """
    Test the conflict between _coerce_finite and price validation.
    """
    print("=" * 80)
    print("TEST: Upstream conflict between _coerce_finite and price validation")
    print("=" * 80)
    print()

    # Simulate what happens in mediator.py
    print("SCENARIO: mark_price = NaN (missing data)")
    print("-" * 80)

    mark_price = float("nan")
    print(f"1. Input: mark_price = {mark_price}")

    # Step 1: _coerce_finite converts NaN → 0.0
    coerced_price = _coerce_finite(mark_price, default=0.0)
    print(f"2. After _coerce_finite(): price = {coerced_price} (NaN silently converted to 0.0)")

    # Step 2: Try to build observation with coerced price
    print(f"3. Calling build_observation_vector(price={coerced_price}, ...)")

    obs = np.zeros(56, dtype=np.float32)
    norm_cols = np.zeros(21, dtype=np.float32)

    try:
        build_observation_vector(
            price=coerced_price,  # 0.0 from NaN
            prev_price=50000.0,
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
        print("   ✓ build_observation_vector() succeeded")
        print()
        print("❌ TEST FAILED: Expected ValueError but got success!")
        print("   This means price validation is NOT working correctly.")

    except ValueError as e:
        print(f"   ✗ build_observation_vector() raised ValueError")
        print(f"   Error message: {str(e)[:100]}...")
        print()
        print("✅ TEST PASSED: Price validation correctly rejects 0.0 from NaN")
        print()
        print("CONCLUSION:")
        print("-" * 80)
        print("• Old behavior: NaN → 0.0 → observation built with invalid data (SILENT FAILURE)")
        print("• New behavior: NaN → 0.0 → ValueError (EXPLICIT FAILURE)")
        print()
        print("⚠️  THIS IS A BREAKING CHANGE")
        print("   Code that relied on silent fallback to 0.0 will now fail explicitly.")
        print()
        print("✅ BUT THIS IS CORRECT BEHAVIOR:")
        print("   NaN prices indicate data corruption and should be caught early.")
        print("   The fix exposes existing data quality issues that were previously hidden.")


def test_positive_infinity_scenario():
    """Test with +Inf input."""
    print()
    print("=" * 80)
    print("SCENARIO: mark_price = +Inf (arithmetic overflow)")
    print("-" * 80)

    mark_price = float("inf")
    print(f"1. Input: mark_price = {mark_price}")

    coerced_price = _coerce_finite(mark_price, default=0.0)
    print(f"2. After _coerce_finite(): price = {coerced_price}")

    obs = np.zeros(56, dtype=np.float32)
    norm_cols = np.zeros(21, dtype=np.float32)

    try:
        build_observation_vector(
            price=coerced_price,
            prev_price=50000.0,
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
        print("❌ Expected ValueError but got success")
    except ValueError as e:
        print(f"✅ Correctly rejected: {str(e)[:80]}...")


if __name__ == "__main__":
    test_upstream_conflict()
    test_positive_infinity_scenario()

    print()
    print("=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    print("The _coerce_finite() function in mediator.py should be updated to:")
    print("1. NOT use 0.0 as fallback for price/prev_price (use None or raise error)")
    print("2. Handle NaN/Inf prices explicitly BEFORE calling build_observation_vector()")
    print("3. Log warnings when invalid prices are detected")
    print()
    print("This way the error is caught at the data ingestion layer,")
    print("not at the observation construction layer.")
