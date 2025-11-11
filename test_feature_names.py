#!/usr/bin/env python3
"""Test script to verify feature name generation for 4h interval."""

import sys
from transformers import _format_window_name, FeatureSpec

def test_format_window_name():
    """Test _format_window_name() function."""
    print("Testing _format_window_name() function:")
    print("-" * 50)

    test_cases = [
        # (input_minutes, expected_output, description)
        (240, "4h", "4 hours"),
        (720, "12h", "12 hours"),
        (1440, "24h", "24 hours"),
        (2880, "48h", "48 hours"),
        (10080, "7d", "7 days"),
        (20160, "14d", "14 days"),
        (43200, "30d", "30 days"),
        (480, "8h", "8 hours"),
        (960, "16h", "16 hours"),
    ]

    all_passed = True
    for minutes, expected, description in test_cases:
        result = _format_window_name(minutes)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"{status} _format_window_name({minutes:5d}) = '{result:4s}' (expected '{expected:4s}') - {description}")

    print()
    return all_passed

def test_feature_names():
    """Test that feature names match between transformers and mediator."""
    print("Testing feature name consistency:")
    print("-" * 50)

    # Expected feature names from mediator.py _extract_norm_cols()
    expected_features = [
        # Returns
        "ret_4h", "ret_12h", "ret_24h",
        # GARCH
        "garch_7d", "garch_14d", "garch_30d",
        # Yang-Zhang
        "yang_zhang_48h", "yang_zhang_7d", "yang_zhang_30d",
        # Parkinson
        "parkinson_48h", "parkinson_7d",
        # CVD
        "cvd_24h", "cvd_7d",
        # Taker Buy Ratio SMA
        "taker_buy_ratio_sma_8h", "taker_buy_ratio_sma_16h", "taker_buy_ratio_sma_24h",
        # Taker Buy Ratio Momentum
        "taker_buy_ratio_momentum_4h", "taker_buy_ratio_momentum_8h", "taker_buy_ratio_momentum_12h",
    ]

    # Create FeatureSpec with 4h interval parameters
    spec = FeatureSpec(
        lookbacks_prices=[240, 720, 1440],  # 4h, 12h, 24h
        garch_windows=[10080, 20160, 43200],  # 7d, 14d, 30d
        yang_zhang_windows=[2880, 10080, 43200],  # 48h, 7d, 30d
        parkinson_windows=[2880, 10080],  # 48h, 7d
        cvd_windows=[1440, 10080],  # 24h, 7d
        taker_buy_ratio_windows=[480, 960, 1440],  # 8h, 16h, 24h
        taker_buy_ratio_momentum=[240, 480, 720],  # 4h, 8h, 12h
    )

    # Generate feature names from transformers.py logic
    generated_features = []

    # Returns
    for lb in spec.lookbacks_prices:
        generated_features.append(f"ret_{_format_window_name(lb)}")

    # GARCH
    for window in spec.garch_windows:
        generated_features.append(f"garch_{_format_window_name(window)}")

    # Yang-Zhang
    for window in spec.yang_zhang_windows:
        generated_features.append(f"yang_zhang_{_format_window_name(window)}")

    # Parkinson
    for window in spec.parkinson_windows:
        generated_features.append(f"parkinson_{_format_window_name(window)}")

    # CVD
    for window in spec.cvd_windows:
        generated_features.append(f"cvd_{_format_window_name(window)}")

    # Taker Buy Ratio SMA
    for window in spec.taker_buy_ratio_windows:
        generated_features.append(f"taker_buy_ratio_sma_{_format_window_name(window)}")

    # Taker Buy Ratio Momentum
    for window in spec.taker_buy_ratio_momentum:
        generated_features.append(f"taker_buy_ratio_momentum_{_format_window_name(window)}")

    # Compare
    print("\nExpected features (from mediator.py):")
    for feat in expected_features:
        print(f"  - {feat}")

    print("\nGenerated features (from transformers.py):")
    for feat in generated_features:
        print(f"  - {feat}")

    print("\nComparison:")
    all_match = True

    # Check if all expected features are generated
    for feat in expected_features:
        if feat in generated_features:
            print(f"✓ {feat}")
        else:
            print(f"✗ {feat} - MISSING in generated features")
            all_match = False

    # Check for unexpected features
    for feat in generated_features:
        if feat not in expected_features:
            print(f"✗ {feat} - UNEXPECTED (not in mediator)")
            all_match = False

    print()
    return all_match

def main():
    """Run all tests."""
    print("=" * 50)
    print("Feature Name Generation Tests for 4h Interval")
    print("=" * 50)
    print()

    test1_passed = test_format_window_name()
    print()

    test2_passed = test_feature_names()
    print()

    print("=" * 50)
    if test1_passed and test2_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 50)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 50)
        return 1

if __name__ == "__main__":
    sys.exit(main())
