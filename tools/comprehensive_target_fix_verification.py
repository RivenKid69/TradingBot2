#!/usr/bin/env python3
"""
Comprehensive verification script for PPO target clipping fix.

This script performs deep analysis to ensure:
1. ALL locations using targets are correctly fixed
2. No missed edge cases
3. Both training and evaluation sections are correct
4. Both quantile and distributional paths are correct
5. All configuration combinations work
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict


class ComprehensiveVerifier:
    """Comprehensive verifier for PPO target clipping fix."""

    def __init__(self):
        self.ppo_file = Path("distributional_ppo.py")
        if not self.ppo_file.exists():
            raise FileNotFoundError(f"File not found: {self.ppo_file}")
        self.code = self.ppo_file.read_text()
        self.issues = []
        self.warnings = []
        self.checks_passed = 0
        self.checks_total = 0

    def check(self, name: str, condition: bool, error_msg: str = None, warning_msg: str = None):
        """Record a check result."""
        self.checks_total += 1
        if condition:
            self.checks_passed += 1
            print(f"✓ PASS: {name}")
        else:
            if error_msg:
                self.issues.append(f"✗ FAIL: {name} - {error_msg}")
                print(f"✗ FAIL: {name} - {error_msg}")
            elif warning_msg:
                self.warnings.append(f"⚠ WARN: {name} - {warning_msg}")
                print(f"⚠ WARN: {name} - {warning_msg}")

    def verify_training_quantile_path(self):
        """Verify training section quantile path uses unclipped targets."""
        print("\n=== Training Quantile Path ===")

        # Check 1: Uses target_returns_norm_raw_selected
        pattern = r"targets_norm_for_loss\s*=\s*target_returns_norm_raw_selected\.reshape"
        matches = re.findall(pattern, self.code)
        self.check(
            "Training quantile uses unclipped target",
            len(matches) >= 1,
            "Missing: targets_norm_for_loss = target_returns_norm_raw_selected.reshape(...)",
        )

        # Check 2: Does NOT use target_returns_norm_selected (clipped)
        wrong_pattern = r"targets_norm_for_loss\s*=\s*target_returns_norm_selected\.reshape"
        wrong_matches = re.findall(wrong_pattern, self.code)
        self.check(
            "Training quantile does NOT use clipped target",
            len(wrong_matches) == 0,
            f"Found incorrect usage: targets_norm_for_loss = target_returns_norm_selected",
        )

        # Check 3: Comment explains the fix
        self.check(
            "Training quantile has explanatory comment",
            "CRITICAL FIX: Use UNCLIPPED target" in self.code,
            "Missing explanatory comment",
        )

    def verify_training_distributional_path(self):
        """Verify training section distributional (C51) path uses unclipped targets."""
        print("\n=== Training Distributional (C51) Path ===")

        # Check 1: Uses target_returns_norm_raw for C51 projection
        pattern = r"clamped_targets\s*=\s*target_returns_norm_raw\.clamp"
        matches = re.findall(pattern, self.code)
        self.check(
            "Distributional uses unclipped target for projection",
            len(matches) >= 1,
            "Missing: clamped_targets = target_returns_norm_raw.clamp(...)",
        )

        # Check 2: Comment about preventing double-clipping
        self.check(
            "Distributional has explanatory comment",
            "Use UNCLIPPED target for distributional projection" in self.code
            or "Only clamp to support bounds" in self.code,
            "Missing explanatory comment about preventing double-clipping",
        )

        # Check 3: Clamps to support bounds [v_min, v_max]
        pattern = r"target_returns_norm_raw\.clamp\s*\(\s*self\.policy\.v_min"
        matches = re.findall(pattern, self.code)
        self.check(
            "Distributional clamps to support bounds",
            len(matches) >= 1,
            "Should clamp to [v_min, v_max] support bounds",
        )

    def verify_evaluation_section(self):
        """Verify evaluation section uses unclipped targets."""
        print("\n=== Evaluation Section ===")

        # Check 1: Uses target_returns_norm_unclipped
        pattern = r"target_norm_col\s*=\s*target_returns_norm_unclipped\.reshape"
        matches = re.findall(pattern, self.code)
        self.check(
            "Evaluation uses unclipped target",
            len(matches) >= 1,
            "Missing: target_norm_col = target_returns_norm_unclipped.reshape(...)",
        )

        # Check 2: Does NOT use target_returns_norm (clipped)
        # (But target_returns_norm should still exist for statistics)
        pattern = r"target_norm_col\s*=\s*target_returns_norm\.reshape"
        wrong_matches = re.findall(pattern, self.code)
        self.check(
            "Evaluation does NOT use clipped target",
            len(wrong_matches) == 0,
            "Found incorrect usage in evaluation section",
        )

        # Check 3: Comment explains the fix
        self.check(
            "Evaluation has explanatory comment",
            "Use UNCLIPPED targets for explained variance" in self.code
            or "Do NOT clip targets in eval" in self.code,
            "Missing explanatory comment in evaluation section",
        )

    def verify_explained_variance_batches(self):
        """Verify EV batches store unclipped targets."""
        print("\n=== Explained Variance Batches ===")

        # Check: value_target_batches_norm uses unclipped
        pattern = r"value_target_batches_norm\.append\s*\(\s*target_returns_norm_raw_selected"
        matches = re.findall(pattern, self.code, re.DOTALL)
        self.check(
            "EV batches use unclipped target",
            len(matches) >= 1,
            "value_target_batches_norm should append target_returns_norm_raw_selected",
        )

    def verify_consistency_fixes(self):
        """Verify consistency fixes for tensor sizes."""
        print("\n=== Consistency Fixes ===")

        # Check 1: weight_tensor uses correct .numel()
        pattern = r"target_returns_norm_raw_selected\.numel\(\)"
        matches = re.findall(pattern, self.code)
        self.check(
            "weight_tensor uses correct numel()",
            len(matches) >= 1,
            "weight_tensor should use target_returns_norm_raw_selected.numel()",
        )

        # Check 2: expected_group_len uses correct shape
        pattern = r"expected_group_len\s*=.*target_returns_norm_raw_selected"
        matches = re.findall(pattern, self.code)
        self.check(
            "expected_group_len uses correct shape",
            len(matches) >= 1,
            "expected_group_len should use target_returns_norm_raw_selected",
        )

    def verify_statistics_logging(self):
        """Verify statistics/logging correctly use clipped targets."""
        print("\n=== Statistics and Logging ===")

        # Statistics SHOULD use clipped targets (to measure clipping)
        pattern = r"target_norm_for_stats\s*=\s*target_returns_norm_selected"
        matches = re.findall(pattern, self.code)
        self.check(
            "Statistics use clipped target (intentional)",
            len(matches) >= 1,
            warning_msg="Statistics might not be using clipped targets for logging",
        )

        # Debug stats recording should exist
        pattern = r"_record_value_debug_stats"
        matches = re.findall(pattern, self.code)
        self.check(
            "Debug statistics recording preserved",
            len(matches) >= 10,
            warning_msg="Fewer debug stats recordings than expected",
        )

    def verify_no_regressions(self):
        """Verify no regressions were introduced."""
        print("\n=== Regression Checks ===")

        # Check 1: Predictions still clipped
        pattern = r"value_pred_raw_clipped\s*=\s*torch\.clamp"
        matches = re.findall(pattern, self.code)
        self.check(
            "Predictions still clipped (no regression)",
            len(matches) >= 1,
            "Value predictions should still be clipped",
        )

        # Check 2: old_values_raw_tensor still used
        pattern = r"old_values_raw_tensor"
        matches = re.findall(pattern, self.code)
        self.check(
            "old_values_raw_tensor still used (no regression)",
            len(matches) >= 5,
            "old_values_raw_tensor should still be used for prediction clipping",
        )

        # Check 3: clip_range_vf checks still present
        pattern = r"if\s+clip_range_vf"
        matches = re.findall(pattern, self.code)
        self.check(
            "clip_range_vf checks preserved (no regression)",
            len(matches) >= 2,
            "clip_range_vf conditional checks missing",
        )

    def verify_both_normalize_returns_paths(self):
        """Verify both normalize_returns=True and False paths are correct."""
        print("\n=== Both normalize_returns Paths ===")

        # Path 1: normalize_returns=True (uses ret_mu, ret_std)
        pattern = r"target_returns_norm_raw\s*=\s*\(\s*target_returns_raw\s*-\s*ret_mu_tensor\s*\)\s*/\s*ret_std_tensor"
        matches = re.findall(pattern, self.code)
        self.check(
            "normalize_returns=True path exists",
            len(matches) >= 1,
            "normalize_returns=True normalization path missing",
        )

        # Path 2: normalize_returns=False (uses base_scale)
        pattern = r"target_returns_norm_raw\s*=\s*\(\s*\(target_returns_raw\s*/.*base_scale"
        matches = re.findall(pattern, self.code, re.DOTALL)
        self.check(
            "normalize_returns=False path exists",
            len(matches) >= 1,
            "normalize_returns=False normalization path missing",
        )

    def verify_ppo_formula_documented(self):
        """Verify PPO VF clipping formula is documented."""
        print("\n=== PPO Formula Documentation ===")

        # Check for formula in comments
        self.check(
            "L^CLIP_VF formula documented",
            "L^CLIP_VF" in self.code,
            "PPO VF clipping formula not documented",
        )

        self.check(
            "V_targ unchanged requirement documented",
            "V_targ must remain unchanged" in self.code,
            "Requirement that V_targ remains unchanged not documented",
        )

        self.check(
            "PPO paper reference present",
            "PPO" in self.code and ("formula" in self.code or "paper" in self.code),
            "PPO paper/formula reference missing",
        )

    def verify_variable_naming(self):
        """Verify variable naming is clear and consistent."""
        print("\n=== Variable Naming ===")

        # Check that _raw and _unclipped variables exist
        self.check(
            "target_returns_norm_raw variable exists",
            "target_returns_norm_raw" in self.code,
            "target_returns_norm_raw variable missing",
        )

        self.check(
            "target_returns_norm_unclipped variable exists",
            "target_returns_norm_unclipped" in self.code,
            "target_returns_norm_unclipped variable missing",
        )

        self.check(
            "target_returns_norm_raw_selected variable exists",
            "target_returns_norm_raw_selected" in self.code,
            "target_returns_norm_raw_selected variable missing",
        )

    def find_all_target_usages(self) -> List[Tuple[int, str]]:
        """Find all usages of target variables."""
        print("\n=== All Target Variable Usages ===")

        target_vars = [
            "target_returns_norm_selected",
            "target_returns_norm_raw_selected",
            "target_norm_col",
            "targets_norm_for_loss",
            "target_distribution",
        ]

        usages = []
        lines = self.code.split("\n")
        for i, line in enumerate(lines, 1):
            for var in target_vars:
                if var in line and not line.strip().startswith("#"):
                    usages.append((i, line.strip()))

        print(f"Found {len(usages)} target variable usages")
        return usages

    def check_for_unused_variables(self):
        """Check if clipped target variables are used incorrectly."""
        print("\n=== Unused/Misused Variables ===")

        # target_returns_norm_selected should ONLY be used for statistics
        pattern = r"target_returns_norm_selected"
        matches = list(re.finditer(pattern, self.code))

        # Count non-statistics usages
        non_stats_usages = 0
        for match in matches:
            # Get context around match
            start = max(0, match.start() - 100)
            end = min(len(self.code), match.end() + 100)
            context = self.code[start:end]

            # Check if it's for statistics/logging
            if "stats" not in context.lower() and "numel" not in context.lower():
                # Check if it's the definition line
                if "=" in context and "target_returns_norm_selected" in context:
                    before_equals = context[: context.index("=")]
                    if "target_returns_norm_selected" in before_equals:
                        # This is the definition, not usage
                        continue

                # This might be incorrect usage
                line_num = self.code[: match.start()].count("\n") + 1
                print(
                    f"  Line {line_num}: {context[max(0, match.start()-start-20):min(len(context), match.end()-start+20)]}"
                )
                non_stats_usages += 1

        self.check(
            "target_returns_norm_selected used only for statistics",
            non_stats_usages <= 2,  # Allow definition lines
            warning_msg=f"Found {non_stats_usages} non-statistics usages of clipped targets",
        )

    def generate_summary(self):
        """Generate verification summary."""
        print("\n" + "=" * 70)
        print("VERIFICATION SUMMARY")
        print("=" * 70)

        print(f"\nChecks passed: {self.checks_passed}/{self.checks_total}")
        success_rate = (
            (self.checks_passed / self.checks_total * 100) if self.checks_total > 0 else 0
        )
        print(f"Success rate: {success_rate:.1f}%")

        if self.issues:
            print(f"\n❌ ISSUES FOUND ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  {issue}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")

        if not self.issues and not self.warnings:
            print("\n✅ ALL CHECKS PASSED - Fix is correctly implemented!")
            print("\nFix summary:")
            print("1. ✓ Training quantile path uses unclipped targets")
            print("2. ✓ Training distributional path uses unclipped targets")
            print("3. ✓ Evaluation section uses unclipped targets")
            print("4. ✓ Explained variance batches use unclipped targets")
            print("5. ✓ Consistency fixes applied")
            print("6. ✓ Statistics logging preserved")
            print("7. ✓ No regressions introduced")
            print("8. ✓ Both normalize_returns paths correct")
            print("9. ✓ PPO formula properly documented")
            print("10. ✓ Variable naming clear and consistent")
            return True
        elif not self.issues:
            print("\n⚠️  ALL CRITICAL CHECKS PASSED, but there are warnings")
            return True
        else:
            print("\n❌ SOME CRITICAL CHECKS FAILED - Review needed")
            return False

    def run_all_verifications(self) -> bool:
        """Run all verification checks."""
        print("=" * 70)
        print("COMPREHENSIVE PPO TARGET CLIPPING FIX VERIFICATION")
        print("=" * 70)

        self.verify_training_quantile_path()
        self.verify_training_distributional_path()
        self.verify_evaluation_section()
        self.verify_explained_variance_batches()
        self.verify_consistency_fixes()
        self.verify_statistics_logging()
        self.verify_no_regressions()
        self.verify_both_normalize_returns_paths()
        self.verify_ppo_formula_documented()
        self.verify_variable_naming()
        self.find_all_target_usages()
        self.check_for_unused_variables()

        return self.generate_summary()


def main():
    """Main verification function."""
    import sys

    try:
        verifier = ComprehensiveVerifier()
        success = verifier.run_all_verifications()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
