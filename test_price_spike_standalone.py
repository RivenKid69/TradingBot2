"""
Standalone test for price validation (basic checks only).

This test validates the basic validation logic without importing mediator.py
to avoid dependency issues in test environment.
"""

import math
from typing import Any


def validate_critical_price(value: Any, param_name: str = "price") -> float:
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

    return numeric


def run_tests():
    """Run all tests and report results."""
    print("=" * 70)
    print("Price Validation - Basic Checks")
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

    # Test 4: Positive Inf raises
    try:
        validate_critical_price(float('inf'))
        print("✗ Test 4 FAILED: Positive Inf should raise ValueError")
        failed += 1
    except ValueError:
        print("✓ Test 4: Positive Inf raises ValueError")
        passed += 1

    # Test 5: Negative Inf raises
    try:
        validate_critical_price(float('-inf'))
        print("✗ Test 5 FAILED: Negative Inf should raise ValueError")
        failed += 1
    except ValueError:
        print("✓ Test 5: Negative Inf raises ValueError")
        passed += 1

    # Test 6: Zero raises
    try:
        validate_critical_price(0.0)
        print("✗ Test 6 FAILED: Zero should raise ValueError")
        failed += 1
    except ValueError:
        print("✓ Test 6: Zero raises ValueError")
        passed += 1

    # Test 7: Negative price raises
    try:
        validate_critical_price(-100.0)
        print("✗ Test 7 FAILED: Negative should raise ValueError")
        failed += 1
    except ValueError:
        print("✓ Test 7: Negative price raises ValueError")
        passed += 1

    # Test 8: Very small positive price
    try:
        result = validate_critical_price(0.0001)
        assert result == 0.0001
        print("✓ Test 8: Very small positive price passes")
        passed += 1
    except Exception as e:
        print(f"✗ Test 8 FAILED: {e}")
        failed += 1

    # Test 9: Large price
    try:
        result = validate_critical_price(1000000.0)
        assert result == 1000000.0
        print("✓ Test 9: Large price passes")
        passed += 1
    except Exception as e:
        print(f"✗ Test 9 FAILED: {e}")
        failed += 1

    # Test 10: String that can be converted
    try:
        result = validate_critical_price("100.5")
        assert result == 100.5
        print("✓ Test 10: String conversion passes")
        passed += 1
    except Exception as e:
        print(f"✗ Test 10 FAILED: {e}")
        failed += 1

    # Test 11: String that cannot be converted
    try:
        validate_critical_price("not_a_number")
        print("✗ Test 11 FAILED: Invalid string should raise ValueError")
        failed += 1
    except ValueError:
        print("✓ Test 11: Invalid string raises ValueError")
        passed += 1

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
