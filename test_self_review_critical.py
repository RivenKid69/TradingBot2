#!/usr/bin/env python3
"""Critical self-review: Check for maxlen deque bug."""

from collections import deque


def test_momentum_maxlen_issue():
    """Test if maxlen is sufficient for momentum calculation."""
    print("=" * 70)
    print("CRITICAL ISSUE CHECK: maxlen for momentum")
    print("=" * 70)

    # Simulate the current code
    taker_buy_ratio_windows = [2, 4, 6]  # SMA windows (in bars)
    taker_buy_ratio_momentum = [1, 2, 3, 6]  # Momentum windows (in bars)

    all_windows = []
    all_windows.extend(taker_buy_ratio_windows)
    all_windows.extend(taker_buy_ratio_momentum)

    maxlen = max(all_windows)  # Current code
    print(f"\nCurrent maxlen calculation: max({all_windows}) = {maxlen}")

    # Test momentum calculation with window=6
    window = 6
    ratio_list = deque(maxlen=maxlen)

    # Add 7 values
    for i in range(7):
        ratio_list.append(0.5 + i * 0.01)
        print(f"Added value {i}: {0.5 + i * 0.01:.3f}, deque length: {len(ratio_list)}")

    print(f"\nDeque contents: {list(ratio_list)}")
    print(f"Deque length: {len(ratio_list)}")
    print(f"maxlen: {ratio_list.maxlen}")

    # Try to calculate momentum with window=6
    print(f"\n🔍 Attempting momentum calculation with window={window}:")
    print(f"   Need: window + 1 = {window + 1} elements")
    print(f"   Have: {len(ratio_list)} elements")

    if len(ratio_list) >= window + 1:
        try:
            current = ratio_list[-1]
            past = ratio_list[-(window + 1)]
            momentum = (current - past) / past if past > 0 else 0.0
            print(f"   ✅ SUCCESS: current={current:.3f}, past={past:.3f}, momentum={momentum:.3f}")
        except IndexError as e:
            print(f"   ❌ ERROR: {e}")
            return False
    else:
        print(f"   ❌ INSUFFICIENT DATA: {len(ratio_list)} < {window + 1}")
        return False

    # Now test with maxlen=6 (current code)
    print(f"\n🔍 Testing with maxlen={maxlen} (current code):")
    ratio_list_limited = deque(maxlen=maxlen)
    for i in range(7):
        ratio_list_limited.append(0.5 + i * 0.01)

    print(f"   After adding 7 values to deque with maxlen={maxlen}:")
    print(f"   Deque length: {len(ratio_list_limited)}")
    print(f"   Deque contents: {list(ratio_list_limited)}")
    print(f"   ⚠️  OLDEST VALUE WAS DROPPED! (0.500 is missing)")

    if len(ratio_list_limited) >= window + 1:
        try:
            current = ratio_list_limited[-1]
            past = ratio_list_limited[-(window + 1)]
            momentum = (current - past) / past if past > 0 else 0.0
            print(f"   ✅ Calculation succeeded: current={current:.3f}, past={past:.3f}")
            print(f"   BUT IS IT CORRECT? We wanted to compare with 6 bars ago!")
        except IndexError as e:
            print(f"   ❌ INDEX ERROR: {e}")
            return False
    else:
        print(f"   ❌ CONDITION FAILS: {len(ratio_list_limited)} >= {window + 1}")
        return False

    print("\n" + "=" * 70)
    print("🔴 CRITICAL BUG CONFIRMED!")
    print("=" * 70)
    print("\nPROBLEM:")
    print(f"  - For momentum with window={window}, we need {window + 1} elements")
    print(f"  - Current maxlen = {maxlen} (max of all windows)")
    print(f"  - When we add {window + 1} elements to deque, oldest is dropped!")
    print(f"  - We CAN'T access element -(window+1) reliably")
    print("\nSOLUTION:")
    print(f"  maxlen should be: max(all_windows) + 1 = {maxlen + 1}")
    print(f"  OR: separate logic for momentum windows")
    print("=" * 70)

    return False


def test_roc_extreme_values():
    """Test ROC calculation with extreme values."""
    print("\n" + "=" * 70)
    print("CRITICAL ISSUE CHECK: ROC with small 'past' values")
    print("=" * 70)

    test_cases = [
        (1e-10, 0.5, "past below threshold (1e-10)"),
        (1e-9, 0.5, "past = 1e-9 (just above threshold)"),
        (1e-8, 0.5, "past = 1e-8"),
        (0.001, 0.5, "past = 0.001"),
        (0.01, 0.5, "past = 0.01"),
    ]

    threshold = 1e-10  # Current threshold in code
    has_issues = False

    for past, current, description in test_cases:
        if abs(past) > threshold:
            momentum = (current - past) / past
            extreme = abs(momentum) > 100  # More than 10000% change
            status = "⚠️  EXTREME" if extreme else "✅ OK"
            print(f"\n{description}:")
            print(f"  past={past:.2e}, current={current:.3f}")
            print(f"  momentum = ({current:.3f} - {past:.2e}) / {past:.2e} = {momentum:.2e}")
            print(f"  {status}")
            if extreme:
                has_issues = True
        else:
            print(f"\n{description}:")
            print(f"  past={past:.2e} <= threshold, using fallback")
            print(f"  ✅ OK (fallback prevents extreme values)")

    if has_issues:
        print("\n" + "=" * 70)
        print("🔴 POTENTIAL ISSUE WITH ROC CALCULATION")
        print("=" * 70)
        print("\nPROBLEM:")
        print("  - threshold=1e-10 is too small")
        print("  - Values like 1e-9, 1e-8 can cause extreme ROC")
        print("  - ROC can be > 10^7 (unrealistic)")
        print("\nSOLUTION:")
        print("  - Increase threshold to 0.01 (1%) or 0.001 (0.1%)")
        print("  - For taker_buy_ratio in [0,1], 0.01 is reasonable")
        print("  - OR: add clipping to ROC (e.g., clip to [-10, 10])")
        print("=" * 70)

    return not has_issues


if __name__ == "__main__":
    print("\n🔍 CRITICAL SELF-REVIEW OF TAKER_BUY_RATIO ANALYSIS\n")

    issue1 = not test_momentum_maxlen_issue()
    issue2 = not test_roc_extreme_values()

    print("\n\n" + "=" * 70)
    print(" SELF-REVIEW RESULTS")
    print("=" * 70)

    if issue1 or issue2:
        print("\n❌ CRITICAL ISSUES FOUND IN ORIGINAL ANALYSIS!")
        print(f"\n  Issue 1 (maxlen bug): {'FOUND' if issue1 else 'OK'}")
        print(f"  Issue 2 (ROC extremes): {'FOUND' if issue2 else 'OK'}")
        print("\n⚠️  MY ORIGINAL ANALYSIS WAS INCOMPLETE!")
        print("These bugs existed BEFORE my changes, but I MISSED them!")
        print("\nNEXT STEPS:")
        print("  1. Fix maxlen calculation")
        print("  2. Improve ROC threshold")
        print("  3. Add tests for these edge cases")
    else:
        print("\n✅ No critical issues found in self-review")

    print("=" * 70 + "\n")
