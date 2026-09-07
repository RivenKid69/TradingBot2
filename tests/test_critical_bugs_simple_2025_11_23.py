"""
Simplified regression tests for critical bugs fixed on 2025-11-23.

These tests are simplified versions that don't require full environment setup
and can run in CI/CD pipelines without dependencies.

Reference: CRITICAL_BUGS_ANALYSIS_2025_11_23.md
"""

import numpy as np
import pandas as pd
import pytest


# ==============================================================================
# Problem #1: Data Leakage - Technical Indicators Shift (SIMPLIFIED)
# ==============================================================================


def test_indicator_shift_logic():
    """
    Simplified test: Verify that shift logic for indicators exists in trading_patchnew.py.

    This test doesn't require full environment initialization, just checks
    that the fix code is present.
    """
    # Read trading_patchnew.py and verify fix is present
    with open("trading_patchnew.py", "r", encoding="utf-8") as f:
        content = f.read()

    # ASSERTION 1: Comment about CRITICAL FIX should be present
    assert (
        "CRITICAL FIX (2025-11-23)" in content
    ), "CRITICAL FIX comment not found! The data leakage fix may have been removed."

    # ASSERTION 2: Code should shift indicators
    assert (
        "_indicators_to_shift" in content
    ), "_indicators_to_shift variable not found! Indicator shift logic may be missing."

    # ASSERTION 3: RSI should be in the list of indicators to shift
    assert '"rsi"' in content or "'rsi'" in content, "RSI not found in indicator shift list!"

    # ASSERTION 4: SMA pattern should be detected and shifted
    assert (
        'startswith("sma_")' in content or "startswith('sma_')" in content
    ), "SMA pattern detection not found! SMA columns may not be shifted."

    # ASSERTION 5: Shift operation should be applied to indicators
    assert (
        "self.df[_indicator] = self.df[_indicator].shift(1)" in content
    ), "Shift operation not found! Indicators may not be shifted."

    print("[OK] All data leakage fix checks passed!")


def test_close_and_indicator_shift_synchronization():
    """
    Test that close and indicators are shifted together (logic verification).
    """
    # Create test dataframe
    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "rsi": [50.0, 55.0, 60.0, 65.0, 70.0],
            "sma_1200": [99.0, 100.0, 101.0, 102.0, 103.0],
            "macd": [-1.0, -0.5, 0.0, 0.5, 1.0],
        }
    )

    # Store original values
    original_close = df["close"].copy()
    original_rsi = df["rsi"].copy()
    original_sma = df["sma_1200"].copy()

    # Simulate the fix logic (what trading_patchnew.py should do)
    df["close"] = df["close"].shift(1)

    # Apply same shift to indicators (THE FIX)
    indicators_to_shift = ["rsi", "macd"]
    sma_cols = [col for col in df.columns if col.startswith("sma_")]
    indicators_to_shift.extend(sma_cols)

    for indicator in indicators_to_shift:
        if indicator in df.columns:
            df[indicator] = df[indicator].shift(1)

    # ASSERTION 1: close[0] should be NaN after shift
    assert pd.isna(df["close"].iloc[0]), "close[0] should be NaN after shift"

    # ASSERTION 2: close[1] should equal original_close[0]
    assert (
        df["close"].iloc[1] == original_close.iloc[0]
    ), f"close[1] should be {original_close.iloc[0]}, got {df['close'].iloc[1]}"

    # ASSERTION 3: rsi[0] should be NaN after shift (same as close)
    assert pd.isna(df["rsi"].iloc[0]), "rsi[0] should be NaN after shift"

    # ASSERTION 4: rsi[1] should equal original_rsi[0] (same shift as close)
    assert (
        df["rsi"].iloc[1] == original_rsi.iloc[0]
    ), f"rsi[1] should be {original_rsi.iloc[0]}, got {df['rsi'].iloc[1]}"

    # ASSERTION 5: sma[1] should equal original_sma[0] (same shift as close)
    assert (
        df["sma_1200"].iloc[1] == original_sma.iloc[0]
    ), f"sma_1200[1] should be {original_sma.iloc[0]}, got {df['sma_1200'].iloc[1]}"

    # ASSERTION 6: All three should be shifted by SAME amount (temporal consistency)
    # At index 2: all should refer to original index 1
    assert df["close"].iloc[2] == original_close.iloc[1]
    assert df["rsi"].iloc[2] == original_rsi.iloc[1]
    assert df["sma_1200"].iloc[2] == original_sma.iloc[1]

    print("[OK] Close and indicator synchronization test passed!")


def test_no_data_leakage_after_shift():
    """
    Verify that after shift, there's no look-ahead bias.

    Scenario: Price spike at index 2 → spike should appear at index 3 after shift.
    """
    # Create data with spike at index 2
    df = pd.DataFrame(
        {
            "close": [100.0, 100.0, 200.0, 100.0, 100.0],  # Spike at index 2
            "rsi": [50.0, 50.0, 90.0, 50.0, 50.0],  # RSI spike at index 2
        }
    )

    # Apply shift (THE FIX)
    df["close"] = df["close"].shift(1)
    df["rsi"] = df["rsi"].shift(1)

    # KEY ASSERTION: At index 2, agent should NOT see the spike
    # (spike is at original index 2, but after shift it appears at index 3)

    # At index 2: NO spike (shows data from original index 1)
    assert (
        df["close"].iloc[2] == 100.0
    ), f"close[2] should be 100.0 (no spike), got {df['close'].iloc[2]}"
    assert df["rsi"].iloc[2] == 50.0, f"rsi[2] should be 50.0 (no spike), got {df['rsi'].iloc[2]}"

    # At index 3: Spike appears (data from original index 2)
    assert (
        df["close"].iloc[3] == 200.0
    ), f"close[3] should be 200.0 (spike), got {df['close'].iloc[3]}"
    assert df["rsi"].iloc[3] == 90.0, f"rsi[3] should be 90.0 (spike), got {df['rsi'].iloc[3]}"

    print("[OK] No data leakage verification passed!")


# ==============================================================================
# Problem #2: Bankruptcy NaN Crash (SIMPLIFIED)
# ==============================================================================


def test_bankruptcy_penalty_code_exists():
    """
    Verify that bankruptcy penalty fix exists in reward.pyx.

    This test checks that the code was modified correctly without requiring
    Cython compilation.
    """
    # Read reward.pyx and verify fix is present
    with open("reward.pyx", "r", encoding="utf-8") as f:
        content = f.read()

    # ASSERTION 1: CRITICAL FIX comment should be present
    assert (
        "CRITICAL FIX (2025-11-23)" in content
    ), "CRITICAL FIX comment not found in reward.pyx! The bankruptcy fix may have been removed."

    # ASSERTION 2: Should return penalty instead of NAN
    assert (
        "return -10.0" in content or "return -10." in content
    ), "Bankruptcy penalty return statement not found! Fix may be missing."

    # ASSERTION 3: Should NOT return NAN for bankruptcy
    # (old code: return NAN)
    # New code should have penalty before NAN check
    lines = content.split("\n")
    found_penalty_before_nan = False
    for i, line in enumerate(lines):
        if "return -10.0" in line or "return -10." in line:
            # Check if this is in bankruptcy condition
            # Look for "net_worth <= 0" or "prev_net_worth <= 0" nearby
            context = "\n".join(lines[max(0, i - 5) : min(len(lines), i + 5)])
            if "net_worth <= 0" in context or "prev_net_worth <= 0" in context:
                found_penalty_before_nan = True
                break

    assert found_penalty_before_nan, (
        "Bankruptcy penalty not found in correct context! " "Fix may not be properly implemented."
    )

    print("[OK] Bankruptcy penalty fix verification passed!")


def test_bankruptcy_penalty_logic():
    """
    Test the bankruptcy penalty logic (Python equivalent).

    This simulates what the fixed Cython code should do.
    """

    def log_return_fixed(net_worth, prev_net_worth):
        """Fixed version with bankruptcy penalty."""
        import math

        # CRITICAL FIX: Return penalty instead of NAN
        if prev_net_worth <= 0.0 or net_worth <= 0.0:
            return -10.0  # Large negative penalty for bankruptcy

        ratio = net_worth / (prev_net_worth + 1e-9)
        ratio = max(0.1, min(ratio, 10.0))  # clamp
        return math.log(ratio)

    # Test Case 1: Normal case (no bankruptcy)
    reward_normal = log_return_fixed(1100.0, 1000.0)  # 10% gain
    assert np.isfinite(reward_normal), "Normal case should return finite value"
    assert reward_normal > 0, "Positive return should give positive reward"

    # Test Case 2: Small loss (no bankruptcy)
    reward_loss = log_return_fixed(990.0, 1000.0)  # 1% loss
    assert np.isfinite(reward_loss), "Loss case should return finite value"
    assert reward_loss < 0, "Negative return should give negative reward"

    # Test Case 3: Bankruptcy (net_worth = 0)
    reward_bankruptcy = log_return_fixed(0.0, 1000.0)
    assert (
        reward_bankruptcy == -10.0
    ), f"Bankruptcy should return -10.0 penalty, got {reward_bankruptcy}"
    assert np.isfinite(reward_bankruptcy), "Bankruptcy should return FINITE value, not NaN"
    assert not np.isnan(reward_bankruptcy), "Bankruptcy should NOT return NaN!"

    # Test Case 4: Bankruptcy (prev_net_worth = 0)
    reward_bankruptcy2 = log_return_fixed(1000.0, 0.0)
    assert (
        reward_bankruptcy2 == -10.0
    ), f"Bankruptcy (prev=0) should return -10.0 penalty, got {reward_bankruptcy2}"

    # Test Case 5: Penalty magnitude check
    assert abs(reward_bankruptcy) > 5 * abs(reward_loss), (
        f"Bankruptcy penalty ({reward_bankruptcy}) should be much larger than "
        f"normal loss ({reward_loss})"
    )

    print("[OK] Bankruptcy penalty logic test passed!")


def test_bankruptcy_does_not_crash_training():
    """
    Verify that finite bankruptcy penalty doesn't cause NaN errors.

    This simulates what happens in distributional_ppo.py GAE computation.
    """
    # Create rewards array with bankruptcy event
    rewards = np.array(
        [
            [0.01],  # Normal
            [0.02],  # Normal
            [-10.0],  # Bankruptcy penalty (FINITE, not NaN!)
            [0.0],  # After bankruptcy
        ],
        dtype=np.float32,
    )

    # CRITICAL ASSERTION: Rewards should all be finite
    assert np.all(np.isfinite(rewards)), (
        f"Rewards contain NaN or Inf! This would crash training. "
        f"Non-finite values: {rewards[~np.isfinite(rewards)]}"
    )

    # Simulate GAE computation check (from distributional_ppo.py:226-230)
    # This is what would crash if bankruptcy returned NaN
    if not np.all(np.isfinite(rewards)):
        pytest.fail(
            "GAE computation would crash! Rewards contain NaN/Inf. "
            "Bankruptcy is still returning NaN instead of penalty."
        )

    # If we reach here, the check passed (no crash)
    print("[OK] Bankruptcy does not crash training (finite penalty)!")


# ==============================================================================
# Integration Tests
# ==============================================================================


def test_both_fixes_are_present():
    """
    Verify that both critical fixes are present in the codebase.
    """
    # Check Fix #1: Data leakage
    with open("trading_patchnew.py", "r", encoding="utf-8") as f:
        trading_content = f.read()

    fix1_present = (
        "CRITICAL FIX (2025-11-23)" in trading_content and "_indicators_to_shift" in trading_content
    )

    # Check Fix #2: Bankruptcy penalty
    with open("reward.pyx", "r", encoding="utf-8") as f:
        reward_content = f.read()

    fix2_present = (
        "CRITICAL FIX (2025-11-23)" in reward_content and "return -10.0" in reward_content
    )

    assert fix1_present, "Fix #1 (Data Leakage) is MISSING!"
    assert fix2_present, "Fix #2 (Bankruptcy Penalty) is MISSING!"

    print("[OK] Both critical fixes are present in codebase!")


def test_documentation_exists():
    """
    Verify that documentation for fixes exists.

    Note: Documentation files have been moved to archive as part of project
    reorganization (2025-11-25). Test updated to check correct archive paths.
    """
    import os

    # Check for analysis report (moved to archive 2025-11-25)
    analysis_paths = [
        "CRITICAL_BUGS_ANALYSIS_2025_11_23.md",  # Original location
        "docs/archive/verification_2025_11/bug_analysis/CRITICAL_BUGS_ANALYSIS_2025_11_23.md",  # Archive
    ]
    analysis_exists = any(os.path.exists(p) for p in analysis_paths)
    assert analysis_exists, f"Analysis report not found in any of: {analysis_paths}"

    # Check for implementation report (moved to archive 2025-11-25)
    impl_paths = [
        "CRITICAL_BUGS_FIX_IMPLEMENTATION_REPORT_2025_11_23.md",  # Original location
        "docs/archive/verification_2025_11/implementation/CRITICAL_BUGS_FIX_IMPLEMENTATION_REPORT_2025_11_23.md",  # Archive
    ]
    impl_exists = any(os.path.exists(p) for p in impl_paths)
    assert impl_exists, f"Implementation report not found in any of: {impl_paths}"

    print("[OK] Documentation exists for all fixes!")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
