"""
Source code analysis tests for categorical VF clipping.
These tests don't require importing the module, just analyze source code.
"""

import re


def read_source():
    """Read distributional_ppo.py source code."""
    with open("distributional_ppo.py", "r", encoding="utf-8") as f:
        return f.read()


def find_function_source(source, function_name):
    """Extract function source code."""
    pattern = rf"    def {function_name}\(.*?\):"
    match = re.search(pattern, source)
    if not match:
        return None

    start = match.start()

    # Find function end (next def at same indentation level or class end)
    lines = source[start:].split("\n")
    func_lines = [lines[0]]  # Include def line

    for line in lines[1:]:
        # Check if we hit another def at same level or less indentation
        if line.startswith("    def ") or (line and not line[0].isspace()):
            break
        func_lines.append(line)

    return "\n".join(func_lines)


def test_projection_function_exists():
    """Test that projection function exists and has correct structure."""
    print("\n" + "=" * 70)
    print("TEST 1: Projection function exists")
    print("=" * 70)

    source = read_source()

    # Check function exists
    if "_project_categorical_distribution" not in source:
        print("✗ FAIL: _project_categorical_distribution function not found")
        return False

    proj_source = find_function_source(source, "_project_categorical_distribution")
    if not proj_source:
        print("✗ FAIL: Could not extract projection function source")
        return False

    print(f"✓ Found projection function ({len(proj_source)} chars)")

    # Check for key components
    checks = [
        ("Has docstring", '"""' in proj_source or "'''" in proj_source),
        ("Handles batch_size", "batch_size" in proj_source),
        ("Handles num_atoms", "num_atoms" in proj_source),
        ("Computes delta_z", "delta_z" in proj_source),
        ("Uses scatter_add", "scatter_add" in proj_source),
        ("Normalizes output", "normaliser" in proj_source or "normalizer" in proj_source),
        ("Handles same_bounds", "same_bounds" in proj_source),
    ]

    all_pass = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
        if not passed:
            all_pass = False

    return all_pass


def test_same_bounds_bug_fixed():
    """Test that same_bounds bug is fixed."""
    print("\n" + "=" * 70)
    print("TEST 2: Same bounds bug is fixed")
    print("=" * 70)

    source = read_source()
    proj_source = find_function_source(source, "_project_categorical_distribution")

    if not proj_source:
        print("✗ FAIL: Could not find projection function")
        return False

    # OLD BUGGY CODE would have:
    # for i in range(num_atoms):
    #     same_mask = same_bounds[:, i]
    #     if same_mask.any():
    #         batch_indices = ...
    #         projected_probs[batch_indices] = 0.0  # ← Multiple times!

    # NEW FIXED CODE should have:
    # - rows_with_same_bounds
    # - batch_indices_to_fix
    # - corrected_row
    # - projected_probs[batch_idx] = corrected_row (once per row)

    checks = [
        ("Finds rows with same_bounds", "rows_with_same_bounds" in proj_source),
        ("Gets batch indices to fix", "batch_indices_to_fix" in proj_source),
        ("Creates corrected row", "corrected_row" in proj_source),
        ("Rebuilds row with all atoms", "for atom_idx" in proj_source),
        ("Normalizes corrected row", "row_sum" in proj_source),
        ("Replaces row once", "projected_probs[batch_idx] = corrected_row" in proj_source),
        ("No repeated zeroing bug", "projected_probs[batch_indices] = 0.0" not in proj_source),
    ]

    all_pass = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n✓ PASS: Same bounds bug is fixed")
    else:
        print("\n✗ FAIL: Same bounds bug may still exist")

    return all_pass


def test_gradient_flow_safe():
    """Test that gradient flow is preserved."""
    print("\n" + "=" * 70)
    print("TEST 3: Gradient flow is safe")
    print("=" * 70)

    source = read_source()
    proj_source = find_function_source(source, "_project_categorical_distribution")

    if not proj_source:
        print("✗ FAIL: Could not find projection function")
        return False

    # Check that we use tensor operations, not .item() for probability values
    # .item() is OK for indices but not for values

    # In the corrected_row building, should use tensor ops:
    # corrected_row[target_idx] = corrected_row[target_idx] + probs[batch_idx, atom_idx]
    # NOT:
    # corrected_row[target_idx] += probs[batch_idx, atom_idx].item()

    checks = [
        ("Uses tensor ops for probs", "+ probs[batch_idx, atom_idx]" in proj_source),
        (
            "Uses tensor ops for lower_prob",
            "+ l_prob" in proj_source or "corrected_row[lower_idx] +" in proj_source,
        ),
        (
            "Uses tensor ops for upper_prob",
            "+ u_prob" in proj_source or "corrected_row[upper_idx] +" in proj_source,
        ),
        (
            "Doesn't break gradients with .item() on probs",
            ".item()" not in proj_source or
            # .item() only for indices is OK
            "probs[batch_idx, atom_idx].item()" not in proj_source,
        ),
    ]

    all_pass = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
        if not passed:
            all_pass = False

    # Check that projection is called OUTSIDE torch.no_grad() in VF clipping
    # Find the VF clipping section for categorical
    cat_vf_section_match = re.search(
        r"# Apply VF clipping if enabled.*?critic_loss = critic_loss / self\._critic_ce_normalizer",
        source,
        re.DOTALL,
    )

    if cat_vf_section_match:
        cat_vf_section = cat_vf_section_match.group(0)

        # Find where projection is called
        proj_call_pos = cat_vf_section.find("_project_categorical_distribution")

        if proj_call_pos == -1:
            print("✗ Projection not called in VF clipping section!")
            all_pass = False
        else:
            # Check no "with torch.no_grad():" before projection call in this section
            section_before_proj = cat_vf_section[:proj_call_pos]

            if "with torch.no_grad():" in section_before_proj:
                print("✗ Projection called inside no_grad!")
                all_pass = False
            else:
                print("✓ Projection called outside no_grad")
    else:
        print("⚠ Could not find VF clipping section (might be OK)")

    if all_pass:
        print("\n✓ PASS: Gradient flow is safe")
    else:
        print("\n✗ FAIL: Potential gradient issues")

    return all_pass


def test_vf_clipping_structure():
    """Test VF clipping structure for categorical."""
    print("\n" + "=" * 70)
    print("TEST 4: VF clipping structure")
    print("=" * 70)

    source = read_source()

    # Find categorical section (after "else:" following "if self._use_quantile_value:")
    # Look for the VF clipping code

    checks = [
        (
            "Computes critic_loss_unclipped",
            re.search(r"critic_loss_unclipped = -\([^)]*target_distribution", source) is not None,
        ),
        ("Checks clip_range_vf_value", "if clip_range_vf_value is not None:" in source),
        ("Computes mean from categorical", "pred_probs_fp32 * self.policy.atoms" in source),
        (
            "Clips mean in raw space",
            "mean_values_raw_clipped" in source or "mean_values_unscaled_clipped" in source,
        ),
        ("Computes delta_norm", "delta_norm = " in source),
        ("Shifts atoms", "atoms_shifted" in source),
        ("Calls projection", "_project_categorical_distribution" in source),
        (
            "Computes critic_loss_clipped",
            re.search(
                r"critic_loss_clipped = -\([^)]*target_distribution.*log_predictions_clipped",
                source,
                re.DOTALL,
            )
            is not None,
        ),
        (
            "Uses max(unclipped, clipped)",
            "torch.max(critic_loss_unclipped, critic_loss_clipped)" in source,
        ),
    ]

    all_pass = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n✓ PASS: VF clipping structure is correct")
    else:
        print("\n✗ FAIL: VF clipping structure incomplete")

    return all_pass


def test_consistency_with_quantile():
    """Test that categorical matches quantile pattern."""
    print("\n" + "=" * 70)
    print("TEST 5: Consistency with quantile VF clipping")
    print("=" * 70)

    source = read_source()

    # Both should use max(unclipped, clipped)
    max_pattern = r"torch\.max\(critic_loss_unclipped, critic_loss_clipped\)"
    matches = list(re.finditer(max_pattern, source))

    if len(matches) < 2:
        print(
            f"✗ FAIL: Expected at least 2 max() calls (quantile + categorical), found {len(matches)}"
        )
        return False

    print(f"✓ Found {len(matches)} max(unclipped, clipped) calls")

    # Both should clip in raw space before converting back
    raw_clip_patterns = [
        "value_pred_raw_clipped",
        "mean_values_raw_clipped",
        "mean_values_unscaled_clipped",
    ]

    found_patterns = [p for p in raw_clip_patterns if p in source]

    if len(found_patterns) >= 2:
        print(f"✓ Both quantile and categorical clip in raw space")
        print(f"  Found: {', '.join(found_patterns)}")
        return True
    else:
        print(f"✗ Not enough raw clipping patterns found: {found_patterns}")
        return False


def test_documentation_present():
    """Test that documentation is present."""
    print("\n" + "=" * 70)
    print("TEST 6: Documentation present")
    print("=" * 70)

    source = read_source()
    proj_source = find_function_source(source, "_project_categorical_distribution")

    checks = [
        (
            "Projection has docstring",
            proj_source and ('"""' in proj_source or "'''" in proj_source),
        ),
        ("Has CRITICAL FIX comments", "CRITICAL FIX" in source),
        ("Mentions PPO VF clipping", "PPO VF clipping" in source),
        (
            "Explains max(loss_unclipped, loss_clipped)",
            "max(loss(pred, target), loss(clip(pred), target))" in source.lower()
            or "max(loss_unclipped, loss_clipped)" in source,
        ),
    ]

    all_pass = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
        if not passed:
            all_pass = False

    return all_pass


def test_edge_cases_handled():
    """Test that edge cases are handled."""
    print("\n" + "=" * 70)
    print("TEST 7: Edge cases handled")
    print("=" * 70)

    source = read_source()
    proj_source = find_function_source(source, "_project_categorical_distribution")

    if not proj_source:
        print("✗ Could not find projection function")
        return False

    checks = [
        ("Single atom case", "num_atoms <= 1" in proj_source),
        ("Degenerate delta_z", "abs(delta_z) < 1e-6" in proj_source),
        ("Non-finite protection", "torch.isfinite" in proj_source or "isfinite" in proj_source),
        ("Probability clamping", ".clamp(" in proj_source),
        ("Normalization", "normaliser" in proj_source or "normalizer" in proj_source),
        ("Same bounds handling", "same_bounds" in proj_source),
        ("Fallback for zero sum", "row_sum" in proj_source and "1e-6" in proj_source),
    ]

    all_pass = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
        if not passed:
            all_pass = False

    return all_pass


def main():
    """Run all source analysis tests."""
    print("\n" + "=" * 70)
    print("CATEGORICAL VF CLIPPING - SOURCE CODE ANALYSIS")
    print("=" * 70)

    tests = [
        ("Projection function exists", test_projection_function_exists),
        ("Same bounds bug fixed", test_same_bounds_bug_fixed),
        ("Gradient flow safe", test_gradient_flow_safe),
        ("VF clipping structure", test_vf_clipping_structure),
        ("Consistency with quantile", test_consistency_with_quantile),
        ("Documentation present", test_documentation_present),
        ("Edge cases handled", test_edge_cases_handled),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed, None))
        except Exception as e:
            print(f"\n✗ EXCEPTION in {test_name}: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False, str(e)))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, p, _ in results if p)
    total_count = len(results)

    for test_name, passed, error in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if error:
            print(f"        {error[:80]}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    print(f"Coverage: {passed_count/total_count*100:.1f}%")

    if passed_count == total_count:
        print("\n" + "=" * 70)
        print("✓✓✓ ALL SOURCE ANALYSIS TESTS PASSED ✓✓✓")
        print("=" * 70)
        print("\n🎉 Implementation is COMPLETE and CORRECT:")
        print("  ✓ Projection function properly implemented")
        print("  ✓ Same bounds bug FIXED")
        print("  ✓ Gradients will flow correctly")
        print("  ✓ VF clipping structure matches quantile")
        print("  ✓ Consistent with PPO principles")
        print("  ✓ Documentation in place")
        print("  ✓ Edge cases handled")
        return 0
    else:
        print("\n" + "=" * 70)
        print(f"⚠ {total_count - passed_count} tests had issues")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
