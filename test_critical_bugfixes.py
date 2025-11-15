#!/usr/bin/env python3
"""Test critical bug fixes: maxlen and ROC threshold."""

import sys
from transformers import FeatureSpec, OnlineFeatureTransformer


def test_maxlen_fix():
    """Test that maxlen is now sufficient for momentum calculation."""
    print("=" * 70)
    print("TEST: maxlen fix for momentum")
    print("=" * 70)

    spec = FeatureSpec(
        lookbacks_prices=[1],
        taker_buy_ratio_windows=[2, 4, 6],  # SMA windows
        taker_buy_ratio_momentum=[1, 2, 3, 6],  # Momentum windows (max = 6)
        bar_duration_minutes=240,
    )

    transformer = OnlineFeatureTransformer(spec)

    # Add 7 bars (need 7 for momentum window=6)
    print("\nAdding 7 bars to test momentum_24h (window=6 bars):")
    for i in range(7):
        feats = transformer.update(
            symbol="BTCUSDT",
            ts_ms=1000 + i * 1000,
            close=100.0 + i,
            volume=100.0,
            taker_buy_base=50.0 + i,
        )
        ratio = feats.get("taker_buy_ratio", float("nan"))
        momentum_24h = feats.get("taker_buy_ratio_momentum_24h", float("nan"))
        print(f"  Bar {i}: ratio={ratio:.3f}, momentum_24h={'NaN' if str(momentum_24h) == 'nan' else f'{momentum_24h:.3f}'}")

    # Check final bar
    final_ratio = feats["taker_buy_ratio"]
    final_momentum = feats.get("taker_buy_ratio_momentum_24h", float("nan"))

    print(f"\n✅ Final bar (bar 6):")
    print(f"   taker_buy_ratio: {final_ratio:.3f}")
    print(f"   taker_buy_ratio_momentum_24h: {final_momentum:.6f if str(final_momentum) != 'nan' else 'NaN'}")

    # Should NOT be NaN on bar 6 (have 7 bars total, enough for window=6)
    if str(final_momentum) == "nan":
        print("\n❌ FAIL: momentum_24h is still NaN! maxlen fix didn't work")
        return False
    else:
        print("\n✅ PASS: momentum_24h calculated successfully")
        print(f"   (Expected: value calculated from 7 bars of data)")
        return True


def test_roc_threshold_fix():
    """Test that ROC threshold prevents extreme values."""
    print("\n" + "=" * 70)
    print("TEST: ROC threshold fix for extreme values")
    print("=" * 70)

    spec = FeatureSpec(
        lookbacks_prices=[1],
        taker_buy_ratio_windows=[2],
        taker_buy_ratio_momentum=[1],
        bar_duration_minutes=240,
    )

    transformer = OnlineFeatureTransformer(spec)

    test_cases = [
        # (past_ratio, current_ratio, description)
        (0.001, 0.5, "Extreme jump from 0.1% to 50%"),
        (0.005, 0.5, "Large jump from 0.5% to 50%"),
        (0.01, 0.5, "Jump from 1% to 50% (at threshold)"),
        (0.05, 0.5, "Jump from 5% to 50% (above threshold)"),
        (0.5, 0.001, "Extreme drop from 50% to 0.1%"),
        (0.5, 0.05, "Drop from 50% to 5%"),
    ]

    all_passed = True
    print("\nTesting various taker_buy_ratio transitions:\n")

    for past_tbb, current_tbb, description in test_cases:
        # Reset transformer
        transformer = OnlineFeatureTransformer(spec)

        # Add bar with past ratio
        transformer.update(
            symbol="BTCUSDT",
            ts_ms=1000,
            close=100.0,
            volume=1.0,  # volume=1 for easy ratio calculation
            taker_buy_base=past_tbb,  # ratio = past_tbb / 1.0 = past_tbb
        )

        # Add bar with current ratio
        feats = transformer.update(
            symbol="BTCUSDT",
            ts_ms=2000,
            close=101.0,
            volume=1.0,
            taker_buy_base=current_tbb,  # ratio = current_tbb
        )

        momentum = feats.get("taker_buy_ratio_momentum_4h", float("nan"))

        # Check if momentum is reasonable (not extreme)
        is_reasonable = abs(momentum) <= 100  # Max 10000% change
        status = "✅" if is_reasonable else "❌"

        print(f"{status} {description}:")
        print(f"     Past ratio: {past_tbb:.3f}, Current ratio: {current_tbb:.3f}")
        print(f"     Momentum: {momentum:.3f}")

        if past_tbb < 0.01 or current_tbb < 0.01:
            # Should use fallback (clamped to -1, 0, or +1)
            expected_fallback = abs(momentum) <= 1.0
            if not expected_fallback:
                print(f"     ⚠️  Expected fallback value (|momentum| <= 1), got {momentum:.3f}")
                all_passed = False
        else:
            # Should use ROC formula
            expected_roc = (current_tbb - past_tbb) / past_tbb
            if abs(momentum - expected_roc) > 0.01:
                print(f"     ⚠️  Expected ROC {expected_roc:.3f}, got {momentum:.3f}")
                all_passed = False

        if not is_reasonable:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ PASS: All ROC values are reasonable")
        print("   - Threshold 0.01 prevents extreme values")
        print("   - Fallback logic works correctly")
    else:
        print("❌ FAIL: Some ROC values are still extreme")
    print("=" * 70)

    return all_passed


def main():
    """Run all critical bug fix tests."""
    print("\n" + "=" * 70)
    print(" CRITICAL BUG FIX VERIFICATION")
    print("=" * 70 + "\n")

    results = []

    try:
        results.append(("maxlen fix", test_maxlen_fix()))
    except Exception as e:
        print(f"❌ maxlen test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("maxlen fix", False))

    try:
        results.append(("ROC threshold fix", test_roc_threshold_fix()))
    except Exception as e:
        print(f"❌ ROC threshold test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("ROC threshold fix", False))

    print("\n" + "=" * 70)
    print(" FINAL RESULTS")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30} {status}")
        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("✅ ALL CRITICAL BUG FIXES VERIFIED")
        print("\nFixed bugs:")
        print("1. maxlen calculation now accounts for momentum (window + 1)")
        print("2. ROC threshold increased to 0.01 to prevent extreme values")
    else:
        print("❌ SOME FIXES FAILED VERIFICATION")
        print("\nPlease review failed tests above.")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
