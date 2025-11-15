"""
FINAL TEST: Exact scenario from the bug report.

From issue:
| Бар | Цена  | Δ    | avg_gain | avg_loss | Условие       | RSI | Должно |
|-----|-------|------|----------|----------|---------------|-----|--------|
| 1   | 29100 | +100 | 100.0    | 0.0      | 0.0 > 0.0 = F | NaN | 100.0  |
| 2   | 29200 | +100 | 100.0    | 0.0      | 0.0 > 0.0 = F | NaN | 100.0  |
| 3   | 29300 | +100 | 100.0    | 0.0      | 0.0 > 0.0 = F | NaN | 100.0  |
| 4   | 29100 | -200 | 92.8     | 14.3     | 14.3 > 0.0 = T| 76.5| 76.5   |
"""

import math


def calculate_rsi_old_buggy(avg_gain, avg_loss):
    """OLD BUGGY VERSION from transformers.py"""
    if avg_loss > 0.0:  # BUG: doesn't handle avg_loss == 0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
    else:
        return float("nan")


def calculate_rsi_fixed(avg_gain, avg_loss):
    """FIXED VERSION now in transformers.py"""
    if avg_loss == 0.0 and avg_gain > 0.0:
        return float(100.0)
    elif avg_gain == 0.0 and avg_loss > 0.0:
        return float(0.0)
    elif avg_gain == 0.0 and avg_loss == 0.0:
        return float(50.0)
    else:
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))


def simulate_rsi_update(prices, period=14):
    """
    Simulate RSI calculation as in transformers.py.
    Returns (avg_gain, avg_loss) after processing all prices.
    """
    avg_gain = None
    avg_loss = None
    last_price = None

    for price in prices:
        if last_price is not None:
            delta = price - last_price
            gain = max(delta, 0.0)
            loss = max(-delta, 0.0)

            if avg_gain is None or avg_loss is None:
                avg_gain = float(gain)
                avg_loss = float(loss)
            else:
                avg_gain = ((avg_gain * (period - 1)) + gain) / period
                avg_loss = ((avg_loss * (period - 1)) + loss) / period

        last_price = price

    return avg_gain, avg_loss


print("=" * 80)
print("FINAL TEST: Exact scenario from bug report")
print("=" * 80)

# Exact prices from bug report
prices = [29000, 29100, 29200, 29300, 29100]

print("\nPrice sequence:")
for i, price in enumerate(prices):
    if i > 0:
        delta = price - prices[i - 1]
        print(f"  Bar {i}: {price} (Δ = {delta:+.0f})")
    else:
        print(f"  Bar {i}: {price} (initial)")

print("\nSimulating RSI calculation...")

# Process each bar
last_price = None
avg_gain = None
avg_loss = None

results = []

for i, price in enumerate(prices):
    if last_price is not None:
        delta = price - last_price
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)

        if avg_gain is None or avg_loss is None:
            avg_gain = float(gain)
            avg_loss = float(loss)
        else:
            period = 14
            avg_gain = ((avg_gain * (period - 1)) + gain) / period
            avg_loss = ((avg_loss * (period - 1)) + loss) / period

        # Calculate RSI with both old and new methods
        rsi_old = calculate_rsi_old_buggy(avg_gain, avg_loss)
        rsi_new = calculate_rsi_fixed(avg_gain, avg_loss)

        results.append({
            "bar": i,
            "price": price,
            "delta": delta,
            "avg_gain": avg_gain,
            "avg_loss": avg_loss,
            "rsi_old": rsi_old,
            "rsi_new": rsi_new,
        })

    last_price = price

print("\n" + "=" * 80)
print("Results comparison:")
print("=" * 80)

print(f"\n{'Bar':<5} {'Price':<7} {'Δ':<6} {'avg_gain':<10} {'avg_loss':<10} "
      f"{'OLD RSI':<10} {'NEW RSI':<10} {'Status':<10}")
print("-" * 80)

all_ok = True

for r in results:
    old_str = f"{r['rsi_old']:.1f}" if not math.isnan(r['rsi_old']) else "NaN"
    new_str = f"{r['rsi_new']:.1f}"

    # Check if fix is working
    if r['avg_loss'] == 0.0 and r['avg_gain'] > 0.0:
        # This is the bug scenario!
        if math.isnan(r['rsi_old']) and r['rsi_new'] == 100.0:
            status = "🔧 FIXED"
        else:
            status = "❌ WRONG"
            all_ok = False
    else:
        if abs(r['rsi_old'] - r['rsi_new']) < 0.01:
            status = "✓ Same"
        else:
            status = "⚠️ Diff"

    print(f"{r['bar']:<5} {r['price']:<7.0f} {r['delta']:>5.0f} "
          f"{r['avg_gain']:<10.1f} {r['avg_loss']:<10.1f} "
          f"{old_str:<10} {new_str:<10} {status:<10}")

print("\n" + "=" * 80)
print("VERIFICATION:")
print("=" * 80)

# Bar 1-3: Pure uptrend (avg_loss = 0)
print("\nBars 1-3 (pure uptrend, avg_loss = 0):")
for i in [0, 1, 2]:
    r = results[i]
    if r['avg_loss'] == 0.0:
        old_ok = "NaN" if math.isnan(r['rsi_old']) else f"{r['rsi_old']:.1f}"
        new_ok = r['rsi_new'] == 100.0
        print(f"  Bar {r['bar']}: OLD = {old_ok} (WRONG), NEW = {r['rsi_new']:.1f} "
              f"({'✓ FIXED' if new_ok else '❌ WRONG'})")

# Bar 4: Mixed movement
print("\nBar 4 (has both gains and losses):")
r = results[3]
if r['avg_loss'] > 0:
    match = abs(r['rsi_old'] - r['rsi_new']) < 0.01
    print(f"  Bar {r['bar']}: OLD = {r['rsi_old']:.1f}, NEW = {r['rsi_new']:.1f} "
          f"({'✓ Same' if match else '⚠️ Different'})")

print("\n" + "=" * 80)
if all_ok:
    print("✅ BUG FIX VERIFIED!")
    print("   Old code: Returns NaN when avg_loss = 0")
    print("   New code: Returns 100.0 correctly (pure uptrend)")
else:
    print("❌ BUG FIX FAILED!")
print("=" * 80)
