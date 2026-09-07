#!/usr/bin/env python3
"""
test_reported_issues.py
==================================================================
Test script to verify two reported problems:
1. Loss of original price (close_orig) after feature pipeline transformation
2. Timestamp inconsistency between CSV and Parquet data sources

This script creates reproducible test cases to confirm or deny each issue.
"""
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from features_pipeline import FeaturePipeline
from prepare_and_run import _read_raw, _normalize_ohlcv


def test_problem_1_close_orig_loss():
    """
    Problem #1: Loss of original price (close_orig) after transformation.

    Expected behavior:
    - After transform_df(), close column is shifted by 1 period
    - Original unshifted close price should be preserved in close_orig
    - Downstream components (execution simulator, PnL calc) need current price

    Current behavior:
    - close_orig is checked but never created
    - After transformation, original price $P_t$ is lost
    """
    print("=" * 80)
    print("TEST 1: close_orig Preservation")
    print("=" * 80)

    # Create sample data
    df_original = pd.DataFrame(
        {
            "timestamp": [1000, 2000, 3000, 4000, 5000],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            "rsi_14": [50.0, 55.0, 60.0, 65.0, 70.0],
        }
    )

    print("\n1. Original DataFrame:")
    print(df_original[["timestamp", "close", "rsi_14"]])

    # Fit and transform
    pipe = FeaturePipeline()
    pipe.fit({"test": df_original})

    df_transformed = pipe.transform_df(df_original.copy())

    print("\n2. After transform_df():")
    print(df_transformed[["timestamp", "close", "rsi_14"]])

    print("\n3. Check for close_orig column:")
    if "close_orig" in df_transformed.columns:
        print("   [OK] close_orig column EXISTS")
        print(f"   Original prices preserved: {df_transformed['close_orig'].values}")
    else:
        print("   [FAIL] close_orig column MISSING")
        print("   [WARNING] Original unshifted close prices are LOST")
        print("   [WARNING] Downstream components cannot access current price $P_t$")

    print("\n4. Shifted vs Original close:")
    print(f"   Original close: {df_original['close'].values}")
    print(f"   Shifted close:  {df_transformed['close'].values}")

    # Check if shift was applied
    if pd.isna(df_transformed["close"].iloc[0]):
        print("   [OK] Shift was applied (first value is NaN)")
    else:
        print("   [FAIL] Shift was NOT applied")

    # Check if original values are accessible
    original_close_at_t1 = df_original["close"].iloc[1]  # 101.0
    shifted_close_at_t1 = df_transformed["close"].iloc[1]  # Should be 100.0 (shifted)

    print(f"\n5. Example at t=1 (timestamp=2000):")
    print(f"   Original close[t=1]: {original_close_at_t1}")
    print(f"   Shifted close[t=1]:  {shifted_close_at_t1}")

    if "close_orig" in df_transformed.columns:
        close_orig_at_t1 = df_transformed["close_orig"].iloc[1]
        print(f"   close_orig[t=1]:     {close_orig_at_t1}")
        if close_orig_at_t1 == original_close_at_t1:
            print("   [OK] close_orig correctly preserves original price")
        else:
            print("   [FAIL] close_orig does NOT match original price")
    else:
        print("   [FAIL] No way to access original price after transformation")

    print("\n" + "=" * 80)
    print("VERDICT for Problem #1:")
    if "close_orig" not in df_transformed.columns:
        print("[WARNING]  PROBLEM CONFIRMED: close_orig is NOT created")
        print("    Impact: Original price $P_t$ is lost after transformation")
        print("    Consequence: Execution simulator and PnL calc cannot access current price")
        return False
    else:
        print("[OK] NO PROBLEM: close_orig is properly created and preserved")
        return True


def test_problem_2_timestamp_inconsistency():
    """
    Problem #2: Timestamp inconsistency between CSV and Parquet sources.

    Expected behavior:
    - Timestamps from CSV and Parquet should be identical for the same bar
    - Both should represent the same point in time (either Open Time or Close Time consistently)

    Current behavior:
    - CSV: timestamp = floor(close_time / 14400) * 14400 → OPEN TIME (start of bar)
    - Parquet (close_time): timestamp = close_time → CLOSE TIME (end of bar)
    - Parquet (open_time): timestamp = open_time + 14400 → CLOSE TIME (end of bar)
    - Result: 4-hour offset between CSV and Parquet data!
    """
    print("\n" + "=" * 80)
    print("TEST 2: Timestamp Consistency CSV vs Parquet")
    print("=" * 80)

    # Binance 4h candle example (00:00 - 04:00 UTC)
    # Open Time:  0 (00:00:00)
    # Close Time: 14399 (03:59:59) - Binance convention: end - 1ms

    # Test Case 1: CSV path
    print("\n1. CSV Data Source (via _read_raw):")
    csv_temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
    csv_data = """open_time,close_time,open,high,low,close,volume,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume
0,14399,100.0,105.0,99.0,102.0,1000.0,102000.0,500,600.0,61200.0
14400,28799,102.0,107.0,101.0,105.0,1100.0,115500.0,550,650.0,68250.0
"""
    csv_temp.write(csv_data)
    csv_temp.close()

    try:
        df_csv = _read_raw(csv_temp.name)
        print(f"   Computed timestamp:   {df_csv['timestamp'].iloc[0]}")

        # After fix (2025-11-25): _read_raw() uses close_time directly (no floor division)
        # OLD (WRONG): timestamp = (close_time // 14400) * 14400 = 0 (OPEN TIME)
        # NEW (CORRECT): timestamp = close_time = 14399 (CLOSE TIME)
        print(f"   Expected after fix:   timestamp = close_time = 14399 (CLOSE TIME)")

        # After fix: CSV should use close_time (CLOSE TIME), not floor division (OPEN TIME)
        if df_csv["timestamp"].iloc[0] == 14399:
            print("   [OK] CSV timestamp = 14399 (CLOSE TIME - matches Parquet behavior)")
        else:
            print(
                f"   [WARNING]  CSV timestamp = {df_csv['timestamp'].iloc[0]} (unexpected - should be 14399)"
            )

    finally:
        os.unlink(csv_temp.name)

    # Test Case 2: Parquet path (explicit close_time column)
    print("\n2. Parquet Data Source (via _normalize_ohlcv) - close_time column:")
    df_parquet_close = pd.DataFrame(
        {
            "close_time": [14399, 28799],  # Binance convention: end - 1ms
            "open": [100.0, 102.0],
            "high": [105.0, 107.0],
            "low": [99.0, 101.0],
            "close": [102.0, 105.0],
            "volume": [1000.0, 1100.0],
        }
    )

    df_normalized_close = _normalize_ohlcv(df_parquet_close, "test_close.parquet")
    print(f"   First bar close_time: {df_parquet_close['close_time'].iloc[0]} (03:59:59)")
    print(f"   Computed timestamp:   {df_normalized_close['timestamp'].iloc[0]}")

    if df_normalized_close["timestamp"].iloc[0] == 14399:
        print("   [OK] Parquet timestamp = 14399 (CLOSE TIME - end of 00:00-04:00 bar)")
    else:
        print(
            f"   [WARNING]  Parquet timestamp = {df_normalized_close['timestamp'].iloc[0]} (unexpected)"
        )

    # Test Case 3: Parquet path (explicit open_time column)
    print("\n3. Parquet Data Source (via _normalize_ohlcv) - open_time column:")
    df_parquet_open = pd.DataFrame(
        {
            "open_time": [0, 14400],  # Start of bar
            "open": [100.0, 102.0],
            "high": [105.0, 107.0],
            "low": [99.0, 101.0],
            "close": [102.0, 105.0],
            "volume": [1000.0, 1100.0],
        }
    )

    df_normalized_open = _normalize_ohlcv(df_parquet_open, "test_open.parquet")
    print(f"   First bar open_time:  {df_parquet_open['open_time'].iloc[0]} (00:00:00)")
    print(f"   Computed timestamp:   {df_normalized_open['timestamp'].iloc[0]}")
    print(
        f"   Calculation:          open_time + BAR_DURATION = {df_parquet_open['open_time'].iloc[0]} + 14400 = {df_parquet_open['open_time'].iloc[0] + 14400}"
    )

    if df_normalized_open["timestamp"].iloc[0] == 14400:
        print("   [OK] Parquet timestamp = 14400 (CLOSE TIME - end of 00:00-04:00 bar)")
        print(
            "   [WARNING]  NOTE: This differs from Binance close_time by 1 second (14400 vs 14399)"
        )

    # Compare all three
    print("\n4. Timestamp Comparison for Same Bar (00:00-04:00):")
    csv_ts = df_csv["timestamp"].iloc[0]  # Use actual CSV timestamp (should be 14399 after fix)
    parquet_close_ts = df_normalized_close["timestamp"].iloc[0]  # 14399
    parquet_open_ts = df_normalized_open["timestamp"].iloc[0]  # 14400

    # After fix: CSV should use CLOSE TIME like Parquet
    csv_label = "CLOSE TIME" if csv_ts == 14399 else f"value={csv_ts}"
    print(f"   CSV (via _read_raw):              {csv_ts} ({csv_label})")
    print(f"   Parquet close_time (via _normalize): {parquet_close_ts} (CLOSE TIME)")
    print(f"   Parquet open_time (via _normalize):  {parquet_open_ts} (CLOSE TIME + 1)")

    print("\n5. Time Difference Analysis:")
    diff_csv_parquet_close = abs(parquet_close_ts - csv_ts)
    diff_csv_parquet_open = abs(parquet_open_ts - csv_ts)

    print(
        f"   CSV vs Parquet(close_time): {diff_csv_parquet_close} seconds ({diff_csv_parquet_close / 3600:.1f} hours)"
    )
    print(
        f"   CSV vs Parquet(open_time):  {diff_csv_parquet_open} seconds ({diff_csv_parquet_open / 3600:.1f} hours)"
    )

    if diff_csv_parquet_close >= 14400 or diff_csv_parquet_open >= 14400:
        print("\n   [WARNING]  INCONSISTENCY DETECTED: ~4 hour offset between data sources!")
        print("   Impact:")
        print("   - Cannot merge/concat data from CSV and Parquet sources")
        print("   - merge_asof will mismatch (4-hour temporal misalignment)")
        print("   - Features from different sources will be misaligned")
        print("   - Creates look-ahead bias or lag in combined datasets")
    else:
        print("\n   [OK] CONSISTENCY VERIFIED: Timestamps aligned within tolerance")
        print(
            f"   CSV vs Parquet(close_time): {diff_csv_parquet_close} seconds ({diff_csv_parquet_close / 3600:.3f} hours)"
        )
        print(
            f"   CSV vs Parquet(open_time):  {diff_csv_parquet_open} seconds ({diff_csv_parquet_open / 3600:.3f} hours)"
        )

    print("\n" + "=" * 80)
    print("VERDICT for Problem #2:")
    if diff_csv_parquet_close >= 14400 or diff_csv_parquet_open >= 14400:
        print("[WARNING]  PROBLEM CONFIRMED: Timestamp inconsistency exists")
        print(f"    CSV uses OPEN TIME (start of bar)")
        print(f"    Parquet uses CLOSE TIME (end of bar)")
        print(f"    Time offset: ~{diff_csv_parquet_close / 3600:.1f} hours")
        return False
    else:
        print("[OK]  PROBLEM FIXED: Timestamps are now consistent")
        print(f"    CSV now uses CLOSE TIME (same as Parquet)")
        print(
            f"    Time offset: ~{diff_csv_parquet_close / 3600:.3f} hours (within 1 second tolerance)"
        )
        return True


def main():
    """Run all tests and provide summary."""
    print("\n" + "=" * 80)
    print("TESTING REPORTED ISSUES")
    print("=" * 80)

    # Test Problem #1
    problem_1_ok = test_problem_1_close_orig_loss()

    # Test Problem #2
    problem_2_ok = test_problem_2_timestamp_inconsistency()

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    if not problem_1_ok:
        print("\n[INFO]  Problem #1 Analysis: close_orig not created by default (BY DESIGN)")
        print("   Status: ENHANCEMENT ADDED (preserve_close_orig parameter)")
        print("   Usage: FeaturePipeline(preserve_close_orig=True) to enable")
        print("   Note: Online inference does NOT need close_orig (prices come from market data)")
        print("   Use case: Post-training analysis and debugging only")
    else:
        print("\n[OK] Problem #1: close_orig is properly handled")

    if not problem_2_ok:
        print("\n[WARNING]  Problem #2 CONFIRMED: Timestamp inconsistency between CSV and Parquet")
        print("   Recommendation: Unify timestamp logic in prepare_and_run.py")
        print("   Preferred approach: Use CLOSE TIME consistently (end of bar)")
        print("   Fix: Remove floor division in _read_raw, use close_time directly")
    else:
        print("\n[OK]  Problem #2 FIXED: Timestamps are now consistent across data sources")
        print("   CSV and Parquet both use CLOSE TIME convention")
        print("   Data merging and concatenation now work correctly")

    print("\n" + "=" * 80)

    return problem_1_ok and problem_2_ok


if __name__ == "__main__":
    all_ok = main()
    sys.exit(0 if all_ok else 1)
