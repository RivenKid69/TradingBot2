"""
Simple test script for price validation (no pytest required).
"""

import sys
from pathlib import Path

# Add project root to path
base = Path(__file__).resolve().parent
if str(base) not in sys.path:
    sys.path.append(str(base))

from mediator import Mediator


def test_basic_validation():
    """Test basic price validation."""
    print("Testing basic validation...")

    # Valid price
    try:
        result = Mediator._validate_critical_price(100.0, "test_price")
        assert result == 100.0
        print("  ✓ Valid price passes")
    except Exception as e:
        print(f"  ✗ Valid price failed: {e}")
        return False

    # None should raise
    try:
        Mediator._validate_critical_price(None, "test_price")
        print("  ✗ None should raise ValueError")
        return False
    except ValueError as e:
        assert "None" in str(e)
        print("  ✓ None raises ValueError")

    # NaN should raise
    try:
        Mediator._validate_critical_price(float('nan'), "test_price")
        print("  ✗ NaN should raise ValueError")
        return False
    except ValueError as e:
        assert "NaN" in str(e)
        print("  ✓ NaN raises ValueError")

    # Inf should raise
    try:
        Mediator._validate_critical_price(float('inf'), "test_price")
        print("  ✗ Inf should raise ValueError")
        return False
    except ValueError as e:
        assert "infinity" in str(e)
        print("  ✓ Inf raises ValueError")

    # Zero should raise
    try:
        Mediator._validate_critical_price(0.0, "test_price")
        print("  ✗ Zero should raise ValueError")
        return False
    except ValueError as e:
        assert "positive" in str(e)
        print("  ✓ Zero raises ValueError")

    # Negative should raise
    try:
        Mediator._validate_critical_price(-100.0, "test_price")
        print("  ✗ Negative should raise ValueError")
        return False
    except ValueError as e:
        assert "positive" in str(e)
        print("  ✓ Negative raises ValueError")

    print("  All basic validation tests passed!\n")
    return True


def test_spike_detection():
    """Test spike detection functionality."""
    print("Testing spike detection...")

    # Normal change (30% increase, within 50% threshold)
    try:
        result = Mediator._validate_critical_price(
            130.0, "test_price", prev_price=100.0, max_spike_pct=0.5
        )
        assert result == 130.0
        print("  ✓ 30% increase within threshold passes")
    except Exception as e:
        print(f"  ✗ Normal change failed: {e}")
        return False

    # Normal change (30% decrease, within 50% threshold)
    try:
        result = Mediator._validate_critical_price(
            70.0, "test_price", prev_price=100.0, max_spike_pct=0.5
        )
        assert result == 70.0
        print("  ✓ 30% decrease within threshold passes")
    except Exception as e:
        print(f"  ✗ Normal decrease failed: {e}")
        return False

    # Spike detected (60% increase, exceeds 50% threshold)
    try:
        Mediator._validate_critical_price(
            160.0, "test_price", prev_price=100.0, max_spike_pct=0.5
        )
        print("  ✗ 60% increase should raise ValueError")
        return False
    except ValueError as e:
        assert "spike detected" in str(e)
        assert "60.00%" in str(e)
        assert "increase" in str(e)
        print("  ✓ 60% increase detected as spike")

    # Spike detected (60% decrease, exceeds 50% threshold)
    try:
        Mediator._validate_critical_price(
            40.0, "test_price", prev_price=100.0, max_spike_pct=0.5
        )
        print("  ✗ 60% decrease should raise ValueError")
        return False
    except ValueError as e:
        assert "spike detected" in str(e)
        assert "60.00%" in str(e)
        assert "decrease" in str(e)
        print("  ✓ 60% decrease detected as spike")

    # Flash crash (90% drop)
    try:
        Mediator._validate_critical_price(
            5000.0, "test_price", prev_price=50000.0, max_spike_pct=0.5
        )
        print("  ✗ Flash crash should raise ValueError")
        return False
    except ValueError as e:
        assert "spike detected" in str(e)
        assert "90.00%" in str(e)
        print("  ✓ Flash crash (90% drop) detected")

    # Exactly at threshold (50% change)
    try:
        result = Mediator._validate_critical_price(
            150.0, "test_price", prev_price=100.0, max_spike_pct=0.5
        )
        assert result == 150.0
        print("  ✓ Exactly 50% change passes")
    except Exception as e:
        print(f"  ✗ Exactly at threshold failed: {e}")
        return False

    # No prev_price skips spike check
    try:
        result = Mediator._validate_critical_price(
            1000000.0, "test_price", prev_price=None
        )
        assert result == 1000000.0
        print("  ✓ No prev_price skips spike check")
    except Exception as e:
        print(f"  ✗ No prev_price failed: {e}")
        return False

    print("  All spike detection tests passed!\n")
    return True


def test_real_world_scenarios():
    """Test real-world cryptocurrency scenarios."""
    print("Testing real-world scenarios...")

    # Normal 4h volatility (10% increase)
    try:
        result = Mediator._validate_critical_price(
            55000.0, "test_price", prev_price=50000.0, max_spike_pct=0.5
        )
        assert result == 55000.0
        print("  ✓ Normal 4h volatility (10% increase) passes")
    except Exception as e:
        print(f"  ✗ Normal volatility failed: {e}")
        return False

    # Extreme but acceptable (40% increase)
    try:
        result = Mediator._validate_critical_price(
            70000.0, "test_price", prev_price=50000.0, max_spike_pct=0.5
        )
        assert result == 70000.0
        print("  ✓ Extreme volatility (40% increase) passes")
    except Exception as e:
        print(f"  ✗ Extreme volatility failed: {e}")
        return False

    # Binance flash crash scenario (87% drop) should be rejected
    try:
        Mediator._validate_critical_price(
            8000.0, "test_price", prev_price=60000.0, max_spike_pct=0.5
        )
        print("  ✗ Binance flash crash should raise ValueError")
        return False
    except ValueError as e:
        assert "spike detected" in str(e)
        print("  ✓ Binance flash crash (87% drop) detected")

    # Stablecoin minor fluctuation (0.5%)
    try:
        result = Mediator._validate_critical_price(
            1.005, "test_price", prev_price=1.0, max_spike_pct=0.5
        )
        assert abs(result - 1.005) < 0.0001
        print("  ✓ Stablecoin minor fluctuation passes")
    except Exception as e:
        print(f"  ✗ Stablecoin fluctuation failed: {e}")
        return False

    print("  All real-world scenario tests passed!\n")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("Price Validation Tests")
    print("=" * 70)
    print()

    results = []
    results.append(("Basic Validation", test_basic_validation()))
    results.append(("Spike Detection", test_spike_detection()))
    results.append(("Real-World Scenarios", test_real_world_scenarios()))

    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}: {status}")

    all_passed = all(passed for _, passed in results)
    print()
    if all_passed:
        print("All tests PASSED! ✓")
        return 0
    else:
        print("Some tests FAILED! ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
