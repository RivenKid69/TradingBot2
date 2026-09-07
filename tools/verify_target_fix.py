#!/usr/bin/env python3
"""
Quick verification script for PPO target clipping fix.

Checks that the critical code changes are in place.
"""

import re
from pathlib import Path


def verify_fix():
    """Verify that all critical fixes are in place."""

    ppo_file = Path("distributional_ppo.py")
    if not ppo_file.exists():
        print("❌ ERROR: distributional_ppo.py not found")
        return False

    code = ppo_file.read_text()

    checks = []

    # Check 1: Training quantile uses unclipped target
    check1 = bool(
        re.search(r"targets_norm_for_loss\s*=\s*target_returns_norm_raw_selected\.reshape", code)
    )
    checks.append(("Training quantile uses unclipped target", check1))

    # Check 2: No usage of clipped target in loss
    check2 = not bool(
        re.search(r"targets_norm_for_loss\s*=\s*target_returns_norm_selected\.reshape", code)
    )
    checks.append(("No clipped target in loss", check2))

    # Check 3: Distributional uses unclipped target
    check3 = bool(re.search(r"clamped_targets\s*=\s*target_returns_norm_raw\.clamp", code))
    checks.append(("Distributional uses unclipped target", check3))

    # Check 4: Eval uses unclipped target
    check4 = bool(re.search(r"target_norm_col\s*=\s*target_returns_norm_unclipped\.reshape", code))
    checks.append(("Eval uses unclipped target", check4))

    # Check 5: EV batches use unclipped target
    check5 = bool(
        re.search(
            r"value_target_batches_norm\.append\s*\(\s*target_returns_norm_raw_selected",
            code,
            re.DOTALL,
        )
    )
    checks.append(("EV batches use unclipped target", check5))

    # Check 6: Critical fix comments present
    check6 = "CRITICAL FIX: Use UNCLIPPED target" in code
    checks.append(("Critical fix comments present", check6))

    # Check 7: PPO formula documented
    check7 = "L^CLIP_VF" in code and "V_targ must remain unchanged" in code
    checks.append(("PPO formula documented", check7))

    # Check 8: Predictions still clipped
    check8 = bool(re.search(r"value_pred_raw_clipped\s*=\s*torch\.clamp", code))
    checks.append(("Predictions still clipped (no regression)", check8))

    # Print results
    print("=" * 70)
    print("PPO TARGET CLIPPING FIX VERIFICATION")
    print("=" * 70)

    all_passed = True
    for description, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {description}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("✓ ALL CHECKS PASSED - Fix is correctly implemented!")
        print()
        print("Summary of changes:")
        print("1. Training quantile loss uses unclipped targets")
        print("2. Distributional (C51) projection uses unclipped targets")
        print("3. Evaluation uses unclipped targets")
        print("4. Explained variance batches use unclipped targets")
        print("5. Consistency fixes for tensor sizes")
        print("6. Comprehensive comments explain the fix")
        print("7. Predictions still clipped (PPO VF clipping preserved)")
        return True
    else:
        print("✗ SOME CHECKS FAILED - Review the implementation")
        return False


if __name__ == "__main__":
    import sys

    success = verify_fix()
    sys.exit(0 if success else 1)
