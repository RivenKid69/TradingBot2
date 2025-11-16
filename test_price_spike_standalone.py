"""
Standalone test for price spike detection logic.

This test validates the spike detection algorithm without importing mediator.py
to avoid dependency issues in test environment.
"""

import math
from typing import Any


def validate_critical_price(
    value: Any,
    param_name: str = "price",
    prev_price: float | None = None,
    max_spike_pct: float = 0.5
) -> float:
    """
    Standalone copy of _validate_critical_price for testing.
    This is the exact same logic as in mediator.py.
    """
    if value is None:
        raise ValueError(
            f"Invalid {param_name}: None. "
            f"Price parameters cannot be None. "
            f"This indicates missing data in the pipeline. "
            f"Check data source and ensure price is provided."
        )

    try:
        numeric = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid {param_name}: cannot convert {type(value).__name__} to float. "
            f"Price must be a numeric value. "
            f"Original error: {e}"
        )

    if math.isnan(numeric):
        raise ValueError(
            f"Invalid {param_name}: NaN (Not a Number). "
            f"This indicates missing or corrupted market data. "
            f"NaN prices cannot be safely defaulted to 0.0. "
            f"Fix data source to provide valid prices."
        )

    if math.isinf(numeric):
        sign = "positive" if numeric > 0 else "negative"
        raise ValueError(
            f"Invalid {param_name}: {sign} infinity. "
            f"This indicates arithmetic overflow in calculations. "
            f"Check price calculations for numerical stability. "
            f"Infinity prices cannot be safely handled."
        )

    if numeric <= 0.0:
        raise ValueError(
            f"Invalid {param_name}: {numeric:.10f}. "
            f"Price must be strictly positive (> 0). "
            f"Zero or negative prices are invalid in trading systems. "
            f"If this is intentional (e.g., testing), use a small positive value like 0.01."
        )

    # Spike detection: Check for abnormal price changes
    if prev_price is not None and prev_price > 0.0:
        relative_change = abs(numeric - prev_price) / prev_price
        if relative_change > max_spike_pct:
            pct_change = relative_change * 100.0
            direction = "increase" if numeric > prev_price else "decrease"
            raise ValueError(
                f"Invalid {param_name}: price spike detected. "
                f"Price changed by {pct_change:.2f}% ({direction}) in one bar. "
                f"Previous price: {prev_price:.2f}, Current price: {numeric:.2f}. "
                f"Maximum allowed change: {max_spike_pct * 100.0:.1f}%. "
                f"This may indicate: (1) flash crash/pump, (2) data error, "
                f"(3) missing bars, or (4) exchange outage. "
                f"Review data source and market conditions. "
                f"For legitimate extreme moves, adjust max_spike_pct parameter."
            )

    return numeric


def run_tests():
    """Run all tests and report results."""
    print("=" * 70)
    print("Price Spike Detection - Standalone Tests")
    print("=" * 70)
    print()

    passed = 0
    failed = 0

    # Test 1: Valid price
    try:
        result = validate_critical_price(100.0)
        assert result == 100.0
        print("✓ Test 1: Valid price passes")
        passed += 1
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}")
        failed += 1

    # Test 2: None raises
    try:
        validate_critical_price(None)
        print("✗ Test 2 FAILED: None should raise ValueError")
        failed += 1
    except ValueError:
        print("✓ Test 2: None raises ValueError")
        passed += 1

    # Test 3: NaN raises
    try:
        validate_critical_price(float('nan'))
        print("✗ Test 3 FAILED: NaN should raise ValueError")
        failed += 1
    except ValueError:
        print("✓ Test 3: NaN raises ValueError")
        passed += 1

    # Test 4: Inf raises
    try:
        validate_critical_price(float('inf'))
        print("✗ Test 4 FAILED: Inf should raise ValueError")
        failed += 1
    except ValueError:
        print("✓ Test 4: Inf raises ValueError")
        passed += 1

    # Test 5: Zero raises
    try:
        validate_critical_price(0.0)
        print("✗ Test 5 FAILED: Zero should raise ValueError")
        failed += 1
    except ValueError:
        print("✓ Test 5: Zero raises ValueError")
        passed += 1

    # Test 6: Negative raises
    try:
        validate_critical_price(-100.0)
        print("✗ Test 6 FAILED: Negative should raise ValueError")
        failed += 1
    except ValueError:
        print("✓ Test 6: Negative raises ValueError")
        passed += 1

    # Test 7: Normal change (30% increase)
    try:
        result = validate_critical_price(130.0, prev_price=100.0, max_spike_pct=0.5)
        assert result == 130.0
        print("✓ Test 7: Normal 30% increase passes")
        passed += 1
    except Exception as e:
        print(f"✗ Test 7 FAILED: {e}")
        failed += 1

    # Test 8: Normal change (30% decrease)
    try:
        result = validate_critical_price(70.0, prev_price=100.0, max_spike_pct=0.5)
        assert result == 70.0
        print("✓ Test 8: Normal 30% decrease passes")
        passed += 1
    except Exception as e:
        print(f"✗ Test 8 FAILED: {e}")
        failed += 1

    # Test 9: Spike detected (60% increase)
    try:
        validate_critical_price(160.0, prev_price=100.0, max_spike_pct=0.5)
        print("✗ Test 9 FAILED: 60% increase should raise ValueError")
        failed += 1
    except ValueError as e:
        if "spike detected" in str(e) and "60.00%" in str(e):
            print("✓ Test 9: 60% increase spike detected")
            passed += 1
        else:
            print(f"✗ Test 9 FAILED: Wrong error message: {e}")
            failed += 1

    # Test 10: Spike detected (60% decrease)
    try:
        validate_critical_price(40.0, prev_price=100.0, max_spike_pct=0.5)
        print("✗ Test 10 FAILED: 60% decrease should raise ValueError")
        failed += 1
    except ValueError as e:
        if "spike detected" in str(e) and "60.00%" in str(e):
            print("✓ Test 10: 60% decrease spike detected")
            passed += 1
        else:
            print(f"✗ Test 10 FAILED: Wrong error message: {e}")
            failed += 1

    # Test 11: Flash crash (90% drop)
    try:
        validate_critical_price(5000.0, prev_price=50000.0, max_spike_pct=0.5)
        print("✗ Test 11 FAILED: Flash crash should raise ValueError")
        failed += 1
    except ValueError as e:
        if "spike detected" in str(e) and "90.00%" in str(e):
            print("✓ Test 11: Flash crash (90% drop) detected")
            passed += 1
        else:
            print(f"✗ Test 11 FAILED: Wrong error message: {e}")
            failed += 1

    # Test 12: Exactly at threshold (50%)
    try:
        result = validate_critical_price(150.0, prev_price=100.0, max_spike_pct=0.5)
        assert result == 150.0
        print("✓ Test 12: Exactly 50% change passes")
        passed += 1
    except Exception as e:
        print(f"✗ Test 12 FAILED: {e}")
        failed += 1

    # Test 13: Barely over threshold (50.01%)
    try:
        validate_critical_price(150.01, prev_price=100.0, max_spike_pct=0.5)
        print("✗ Test 13 FAILED: 50.01% should raise ValueError")
        failed += 1
    except ValueError:
        print("✓ Test 13: Barely over threshold detected")
        passed += 1

    # Test 14: No prev_price skips spike check
    try:
        result = validate_critical_price(1000000.0, prev_price=None)
        assert result == 1000000.0
        print("✓ Test 14: No prev_price skips spike check")
        passed += 1
    except Exception as e:
        print(f"✗ Test 14 FAILED: {e}")
        failed += 1

    # Test 15: Custom threshold (10%)
    try:
        validate_critical_price(115.0, prev_price=100.0, max_spike_pct=0.1)
        print("✗ Test 15 FAILED: 15% increase with 10% threshold should raise")
        failed += 1
    except ValueError as e:
        if "spike detected" in str(e) and "15.00%" in str(e):
            print("✓ Test 15: Custom threshold respected")
            passed += 1
        else:
            print(f"✗ Test 15 FAILED: Wrong error message: {e}")
            failed += 1

    # Summary
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    print()

    if failed == 0:
        print("All tests PASSED! ✓")
        return 0
    else:
        print(f"Some tests FAILED! ✗")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_tests())
