"""
Simple standalone test to verify RSI edge case logic fix.
Tests the core RSI calculation logic without requiring full dependencies.
"""

import math


def calculate_rsi_old_buggy(avg_gain, avg_loss):
    """
    OLD BUGGY VERSION: Returns NaN when avg_loss = 0.
    This is the buggy code that was in transformers.py before the fix.
    """
    if avg_loss > 0.0:
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
    else:
        return float("nan")  # BUG: Returns NaN instead of 100


def calculate_rsi_fixed(avg_gain, avg_loss):
    """
    FIXED VERSION: Handles all edge cases correctly.
    This is the fixed code now in transformers.py.
    """
    if avg_loss == 0.0 and avg_gain > 0.0:
        # Pure uptrend: RS = infinity → RSI = 100
        return 100.0
    elif avg_gain == 0.0 and avg_loss > 0.0:
        # Pure downtrend: RS = 0 → RSI = 0
        return 0.0
    elif avg_gain == 0.0 and avg_loss == 0.0:
        # No price movement: neutral RSI
        return 50.0
    else:
        # Normal case: both avg_gain and avg_loss > 0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))


def test_case(name, avg_gain, avg_loss, expected_rsi):
    """Run a single test case and print results."""
    old_result = calculate_rsi_old_buggy(avg_gain, avg_loss)
    new_result = calculate_rsi_fixed(avg_gain, avg_loss)

    old_ok = (math.isnan(old_result) and math.isnan(expected_rsi)) or (abs(old_result - expected_rsi) < 0.01)
    new_ok = (math.isnan(new_result) and math.isnan(expected_rsi)) or (abs(new_result - expected_rsi) < 0.01)

    print(f"\n{name}:")
    print(f"  avg_gain={avg_gain}, avg_loss={avg_loss}")
    print(f"  Expected RSI: {expected_rsi}")
    print(f"  Old (buggy):  {old_result:.2f} {'✓' if old_ok else '✗ WRONG'}")
    print(f"  New (fixed):  {new_result:.2f} {'✓' if new_ok else '✗ WRONG'}")

    if not old_ok and new_ok:
        print(f"  🔧 BUG FIXED!")

    return new_ok


if __name__ == "__main__":
    print("=" * 70)
    print("RSI Edge Case Logic Verification")
    print("=" * 70)

    all_passed = True

    # Test 1: Pure uptrend (avg_loss = 0) - THE CRITICAL BUG
    print("\n" + "─" * 70)
    print("TEST 1: CRITICAL BUG - Pure uptrend (4 bars rising)")
    print("─" * 70)
    all_passed &= test_case(
        "Pure uptrend (avg_loss=0)",
        avg_gain=100.0,
        avg_loss=0.0,
        expected_rsi=100.0
    )

    # Test 2: Pure downtrend (avg_gain = 0)
    print("\n" + "─" * 70)
    print("TEST 2: Pure downtrend")
    print("─" * 70)
    all_passed &= test_case(
        "Pure downtrend (avg_gain=0)",
        avg_gain=0.0,
        avg_loss=50.0,
        expected_rsi=0.0
    )

    # Test 3: No movement (both = 0)
    print("\n" + "─" * 70)
    print("TEST 3: No price movement")
    print("─" * 70)
    all_passed &= test_case(
        "No movement (both=0)",
        avg_gain=0.0,
        avg_loss=0.0,
        expected_rsi=50.0
    )

    # Test 4: Normal case (both > 0)
    print("\n" + "─" * 70)
    print("TEST 4: Normal case with mixed movements")
    print("─" * 70)
    # RS = 92.8 / 14.3 = 6.49
    # RSI = 100 - (100 / (1 + 6.49)) = 100 - 13.35 = 86.65
    all_passed &= test_case(
        "Mixed movements",
        avg_gain=92.8,
        avg_loss=14.3,
        expected_rsi=86.65
    )

    # Test 5: Another normal case
    print("\n" + "─" * 70)
    print("TEST 5: Balanced movements")
    print("─" * 70)
    # RS = 50 / 50 = 1.0
    # RSI = 100 - (100 / 2) = 50
    all_passed &= test_case(
        "Balanced (avg_gain=avg_loss)",
        avg_gain=50.0,
        avg_loss=50.0,
        expected_rsi=50.0
    )

    # Test 6: Heavily oversold
    print("\n" + "─" * 70)
    print("TEST 6: Heavily oversold")
    print("─" * 70)
    # RS = 10 / 90 = 0.111
    # RSI = 100 - (100 / 1.111) = 100 - 90.0 = 10.0
    all_passed &= test_case(
        "Oversold",
        avg_gain=10.0,
        avg_loss=90.0,
        expected_rsi=10.0
    )

    # Test 7: Heavily overbought
    print("\n" + "─" * 70)
    print("TEST 7: Heavily overbought")
    print("─" * 70)
    # RS = 90 / 10 = 9.0
    # RSI = 100 - (100 / 10) = 100 - 10 = 90.0
    all_passed &= test_case(
        "Overbought",
        avg_gain=90.0,
        avg_loss=10.0,
        expected_rsi=90.0
    )

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED! RSI bug is FIXED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 70)
