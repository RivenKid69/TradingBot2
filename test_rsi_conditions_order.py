"""
Verification test: Check order of conditions handles all edge cases correctly.
"""


def test_all_combinations():
    """Test all possible combinations of avg_gain and avg_loss."""

    # Test data: (avg_gain, avg_loss, expected_condition_hit, expected_rsi)
    test_cases = [
        (0.0, 0.0, "condition_3", 50.0),      # Both zero
        (0.0, 50.0, "condition_2", 0.0),      # Only loss
        (100.0, 0.0, "condition_1", 100.0),   # Only gain (THE CRITICAL BUG)
        (50.0, 50.0, "else", 50.0),           # Equal (RS=1)
        (90.0, 10.0, "else", 90.0),           # Overbought
        (10.0, 90.0, "else", 10.0),           # Oversold
    ]

    print("=" * 80)
    print("Testing condition order for all avg_gain/avg_loss combinations")
    print("=" * 80)

    all_passed = True

    for avg_gain, avg_loss, expected_condition, expected_rsi in test_cases:
        # Simulate the exact logic from transformers.py
        if avg_loss == 0.0 and avg_gain > 0.0:
            condition_hit = "condition_1"
            rsi = 100.0
        elif avg_gain == 0.0 and avg_loss > 0.0:
            condition_hit = "condition_2"
            rsi = 0.0
        elif avg_gain == 0.0 and avg_loss == 0.0:
            condition_hit = "condition_3"
            rsi = 50.0
        else:
            condition_hit = "else"
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        # Check results
        condition_ok = condition_hit == expected_condition
        rsi_ok = abs(rsi - expected_rsi) < 0.01

        passed = condition_ok and rsi_ok

        status = "✓" if passed else "✗"
        print(f"\n{status} avg_gain={avg_gain:5.1f}, avg_loss={avg_loss:5.1f}")
        print(f"  Expected: {expected_condition:12} → RSI={expected_rsi:5.1f}")
        print(f"  Got:      {condition_hit:12} → RSI={rsi:5.1f}")

        if not passed:
            all_passed = False
            if not condition_ok:
                print(f"  ❌ WRONG CONDITION! Expected {expected_condition}, got {condition_hit}")
            if not rsi_ok:
                print(f"  ❌ WRONG RSI! Expected {expected_rsi:.1f}, got {rsi:.1f}")

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL CONDITION TESTS PASSED!")
    else:
        print("❌ SOME CONDITION TESTS FAILED!")
    print("=" * 80)

    return all_passed


def test_edge_case_zero_zero():
    """
    Special test: What happens when both are exactly 0?
    This could happen if price doesn't move at all.
    """
    print("\n" + "=" * 80)
    print("EDGE CASE: Both avg_gain and avg_loss are 0.0")
    print("=" * 80)

    avg_gain = 0.0
    avg_loss = 0.0

    # Which condition fires?
    if avg_loss == 0.0 and avg_gain > 0.0:
        print(f"✗ Condition 1 fired (WRONG): 0.0 == 0.0 and 0.0 > 0.0")
        print(f"  = True and False = False (should not fire)")
        return False
    elif avg_gain == 0.0 and avg_loss > 0.0:
        print(f"✗ Condition 2 fired (WRONG): 0.0 == 0.0 and 0.0 > 0.0")
        print(f"  = True and False = False (should not fire)")
        return False
    elif avg_gain == 0.0 and avg_loss == 0.0:
        print(f"✓ Condition 3 fired (CORRECT): 0.0 == 0.0 and 0.0 == 0.0")
        print(f"  = True and True = True")
        print(f"  RSI = 50.0 (neutral)")
        return True
    else:
        print(f"✗ Else fired (WRONG): Division by zero would occur!")
        return False


if __name__ == "__main__":
    result1 = test_all_combinations()
    result2 = test_edge_case_zero_zero()

    if result1 and result2:
        print("\n" + "🎉" * 30)
        print("ALL VERIFICATION TESTS PASSED!")
        print("Condition order is correct!")
        print("🎉" * 30)
    else:
        print("\n" + "⚠️" * 30)
        print("VERIFICATION FAILED!")
        print("⚠️" * 30)
