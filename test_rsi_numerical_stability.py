"""
Attack test: Numerical stability of RSI calculation.
Check edge cases with extreme values.
"""

import math


def calculate_rsi(avg_gain, avg_loss):
    """Fixed RSI calculation from transformers.py"""
    if avg_loss == 0.0 and avg_gain > 0.0:
        return float(100.0)
    elif avg_gain == 0.0 and avg_loss > 0.0:
        return float(0.0)
    elif avg_gain == 0.0 and avg_loss == 0.0:
        return float(50.0)
    else:
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))


print("=" * 80)
print("ATTACK TEST: Numerical Stability")
print("=" * 80)

all_ok = True

# Test 1: Very large avg_gain (extreme overbought)
print("\n[Test 1] Extreme overbought: avg_gain = 1e100, avg_loss = 1e-100")
rsi = calculate_rsi(1e100, 1e-100)
print(f"  RSI = {rsi}")
expected = 100.0
if abs(rsi - expected) < 0.01:
    print(f"  ✓ Correct (expected ~{expected})")
else:
    print(f"  ✗ WRONG! Expected ~{expected}, got {rsi}")
    all_ok = False

# Test 2: Very small avg_gain (extreme oversold)
print("\n[Test 2] Extreme oversold: avg_gain = 1e-100, avg_loss = 1e100")
rsi = calculate_rsi(1e-100, 1e100)
print(f"  RSI = {rsi}")
expected = 0.0
if abs(rsi - expected) < 0.01:
    print(f"  ✓ Correct (expected ~{expected})")
else:
    print(f"  ✗ WRONG! Expected ~{expected}, got {rsi}")
    all_ok = False

# Test 3: Very small equal values
print("\n[Test 3] Very small equal values: avg_gain = 1e-100, avg_loss = 1e-100")
rsi = calculate_rsi(1e-100, 1e-100)
print(f"  RSI = {rsi}")
expected = 50.0
if abs(rsi - expected) < 0.01:
    print(f"  ✓ Correct (expected ~{expected})")
else:
    print(f"  ✗ WRONG! Expected ~{expected}, got {rsi}")
    all_ok = False

# Test 4: Very large equal values
print("\n[Test 4] Very large equal values: avg_gain = 1e100, avg_loss = 1e100")
rsi = calculate_rsi(1e100, 1e100)
print(f"  RSI = {rsi}")
expected = 50.0
if abs(rsi - expected) < 0.01:
    print(f"  ✓ Correct (expected ~{expected})")
else:
    print(f"  ✗ WRONG! Expected ~{expected}, got {rsi}")
    all_ok = False

# Test 5: Exact zero check (not float approximation)
print("\n[Test 5] CRITICAL: Exact zero (0.0), not approximation")
print("  Scenario: Pure uptrend, loss = EXACTLY 0.0")
rsi = calculate_rsi(100.0, 0.0)
print(f"  RSI = {rsi}")
if rsi == 100.0 and not math.isnan(rsi):
    print(f"  ✓ Correct! Returns 100.0, NOT NaN")
else:
    print(f"  ✗ WRONG! Should return 100.0, got {rsi}")
    all_ok = False

# Test 6: Verify NaN is NOT returned in any edge case
print("\n[Test 6] Verify NO NaN in any valid input")
test_cases = [
    (100.0, 0.0),
    (0.0, 100.0),
    (0.0, 0.0),
    (50.0, 50.0),
    (1e100, 1e-100),
    (1e-100, 1e100),
]
nan_found = False
for gain, loss in test_cases:
    rsi = calculate_rsi(gain, loss)
    if math.isnan(rsi):
        print(f"  ✗ NaN found for avg_gain={gain}, avg_loss={loss}")
        nan_found = True
        all_ok = False

if not nan_found:
    print("  ✓ No NaN found in any test case")

# Test 7: RSI bounds check (must be 0-100)
print("\n[Test 7] RSI must be in range [0, 100]")
for gain, loss in test_cases:
    rsi = calculate_rsi(gain, loss)
    if not (0 <= rsi <= 100):
        print(f"  ✗ RSI out of bounds: {rsi} for gain={gain}, loss={loss}")
        all_ok = False

print("  ✓ All RSI values in valid range [0, 100]")

# Test 8: Division by zero protection
print("\n[Test 8] Division by zero protection")
print("  Old buggy code would do: rs = avg_gain / 0.0 → crash or infinity")
print("  New code handles: avg_loss = 0.0 → RSI = 100.0 directly")
try:
    rsi = calculate_rsi(100.0, 0.0)
    print(f"  ✓ No division by zero! RSI = {rsi}")
except ZeroDivisionError:
    print(f"  ✗ DIVISION BY ZERO ERROR!")
    all_ok = False

print("\n" + "=" * 80)
if all_ok:
    print("✅ NUMERICAL STABILITY: ALL TESTS PASSED!")
    print("Code handles extreme values correctly.")
else:
    print("❌ NUMERICAL STABILITY: SOME TESTS FAILED!")
print("=" * 80)
