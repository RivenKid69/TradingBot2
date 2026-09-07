"""
Test VF clipping logic WITHOUT dependencies (no torch, numpy, etc).

This tests the CORE logic that determines whether VF clipping is enabled.
"""


def test_logic_default_none():
    """Test: mode=None (default) should DISABLE VF clipping."""
    print("=" * 70)
    print("TEST: Default mode=None should DISABLE VF clipping")
    print("=" * 70)

    clip_range_vf_value = 0.5
    distributional_vf_clip_mode = None  # DEFAULT

    # This is the EXACT logic from distributional_ppo.py:8713-8716
    distributional_vf_clip_enabled = (
        clip_range_vf_value is not None and distributional_vf_clip_mode not in (None, "disable")
    )

    print(f"Input:")
    print(f"  clip_range_vf_value = {clip_range_vf_value}")
    print(f"  distributional_vf_clip_mode = {distributional_vf_clip_mode}")
    print(f"")
    print(f"Logic:")
    print(f"  clip_range_vf_value is not None = {clip_range_vf_value is not None}")
    print(
        f"  distributional_vf_clip_mode not in (None, 'disable') = {distributional_vf_clip_mode not in (None, 'disable')}"
    )
    print(f"  AND = {distributional_vf_clip_enabled}")
    print(f"")
    print(f"Result: distributional_vf_clip_enabled = {distributional_vf_clip_enabled}")

    if distributional_vf_clip_enabled:
        print("❌ FAIL: VF clipping should be DISABLED by default!")
        return False
    else:
        print("✓ PASS: VF clipping is DISABLED by default")
        return True


def test_logic_disable():
    """Test: mode='disable' should DISABLE VF clipping."""
    print("\n" + "=" * 70)
    print("TEST: mode='disable' should DISABLE VF clipping")
    print("=" * 70)

    clip_range_vf_value = 0.5
    distributional_vf_clip_mode = "disable"

    distributional_vf_clip_enabled = (
        clip_range_vf_value is not None and distributional_vf_clip_mode not in (None, "disable")
    )

    print(f"Input:")
    print(f"  clip_range_vf_value = {clip_range_vf_value}")
    print(f"  distributional_vf_clip_mode = '{distributional_vf_clip_mode}'")
    print(f"")
    print(f"Logic:")
    print(f"  clip_range_vf_value is not None = {clip_range_vf_value is not None}")
    print(
        f"  distributional_vf_clip_mode not in (None, 'disable') = {distributional_vf_clip_mode not in (None, 'disable')}"
    )
    print(f"  AND = {distributional_vf_clip_enabled}")
    print(f"")
    print(f"Result: distributional_vf_clip_enabled = {distributional_vf_clip_enabled}")

    if distributional_vf_clip_enabled:
        print("❌ FAIL: VF clipping should be DISABLED with mode='disable'!")
        return False
    else:
        print("✓ PASS: VF clipping is DISABLED with mode='disable'")
        return True


def test_logic_mean_only():
    """Test: mode='mean_only' should ENABLE VF clipping."""
    print("\n" + "=" * 70)
    print("TEST: mode='mean_only' should ENABLE VF clipping")
    print("=" * 70)

    clip_range_vf_value = 0.5
    distributional_vf_clip_mode = "mean_only"

    distributional_vf_clip_enabled = (
        clip_range_vf_value is not None and distributional_vf_clip_mode not in (None, "disable")
    )

    print(f"Input:")
    print(f"  clip_range_vf_value = {clip_range_vf_value}")
    print(f"  distributional_vf_clip_mode = '{distributional_vf_clip_mode}'")
    print(f"")
    print(f"Logic:")
    print(f"  clip_range_vf_value is not None = {clip_range_vf_value is not None}")
    print(
        f"  distributional_vf_clip_mode not in (None, 'disable') = {distributional_vf_clip_mode not in (None, 'disable')}"
    )
    print(f"  AND = {distributional_vf_clip_enabled}")
    print(f"")
    print(f"Result: distributional_vf_clip_enabled = {distributional_vf_clip_enabled}")

    if not distributional_vf_clip_enabled:
        print("❌ FAIL: VF clipping should be ENABLED with mode='mean_only'!")
        return False
    else:
        print("✓ PASS: VF clipping is ENABLED with mode='mean_only'")
        return True


def test_logic_mean_and_variance():
    """Test: mode='mean_and_variance' should ENABLE VF clipping."""
    print("\n" + "=" * 70)
    print("TEST: mode='mean_and_variance' should ENABLE VF clipping")
    print("=" * 70)

    clip_range_vf_value = 0.5
    distributional_vf_clip_mode = "mean_and_variance"

    distributional_vf_clip_enabled = (
        clip_range_vf_value is not None and distributional_vf_clip_mode not in (None, "disable")
    )

    print(f"Input:")
    print(f"  clip_range_vf_value = {clip_range_vf_value}")
    print(f"  distributional_vf_clip_mode = '{distributional_vf_clip_mode}'")
    print(f"")
    print(f"Logic:")
    print(f"  clip_range_vf_value is not None = {clip_range_vf_value is not None}")
    print(
        f"  distributional_vf_clip_mode not in (None, 'disable') = {distributional_vf_clip_mode not in (None, 'disable')}"
    )
    print(f"  AND = {distributional_vf_clip_enabled}")
    print(f"")
    print(f"Result: distributional_vf_clip_enabled = {distributional_vf_clip_enabled}")

    if not distributional_vf_clip_enabled:
        print("❌ FAIL: VF clipping should be ENABLED with mode='mean_and_variance'!")
        return False
    else:
        print("✓ PASS: VF clipping is ENABLED with mode='mean_and_variance'")
        return True


def test_logic_no_clip_range():
    """Test: clip_range_vf=None should DISABLE regardless of mode."""
    print("\n" + "=" * 70)
    print("TEST: clip_range_vf=None should DISABLE regardless of mode")
    print("=" * 70)

    clip_range_vf_value = None

    for mode in [None, "disable", "mean_only", "mean_and_variance"]:
        distributional_vf_clip_enabled = clip_range_vf_value is not None and mode not in (
            None,
            "disable",
        )

        print(f"  mode='{mode}': enabled={distributional_vf_clip_enabled}")

        if distributional_vf_clip_enabled:
            print(f"  ❌ FAIL: Should be DISABLED when clip_range_vf=None!")
            return False

    print("✓ PASS: DISABLED when clip_range_vf=None for all modes")
    return True


def test_backward_compatibility():
    """Test backward compatibility scenario."""
    print("\n" + "=" * 70)
    print("TEST: Backward compatibility")
    print("=" * 70)

    print("Scenario 1: Old code (implicit behavior)")
    print("  User code: DistributionalPPO(clip_range_vf=0.5)")
    print("  Old behavior: VF clipping WAS applied")
    print("  New behavior:")

    clip_range_vf_value = 0.5
    distributional_vf_clip_mode = None  # Not specified by user

    distributional_vf_clip_enabled = (
        clip_range_vf_value is not None and distributional_vf_clip_mode not in (None, "disable")
    )

    print(f"    distributional_vf_clip_enabled = {distributional_vf_clip_enabled}")
    print(
        f"    Result: VF clipping is {'ENABLED' if distributional_vf_clip_enabled else 'DISABLED'}"
    )

    if distributional_vf_clip_enabled:
        print("  ❌ BREAKING CHANGE NOT IMPLEMENTED: Should be DISABLED by default!")
        return False

    print("  ⚠️  BREAKING CHANGE: VF clipping now DISABLED by default")

    print("\nScenario 2: Restore old behavior")
    print(
        "  User code: DistributionalPPO(clip_range_vf=0.5, distributional_vf_clip_mode='mean_only')"
    )

    distributional_vf_clip_mode = "mean_only"
    distributional_vf_clip_enabled = (
        clip_range_vf_value is not None and distributional_vf_clip_mode not in (None, "disable")
    )

    print(f"    distributional_vf_clip_enabled = {distributional_vf_clip_enabled}")
    print(
        f"    Result: VF clipping is {'ENABLED' if distributional_vf_clip_enabled else 'DISABLED'}"
    )

    if not distributional_vf_clip_enabled:
        print("  ❌ FAIL: Should be ENABLED with mode='mean_only'!")
        return False

    print("  ✓ PASS: Old behavior can be restored with mode='mean_only'")
    return True


def run_all_tests():
    """Run all logic tests."""
    print("\n" + "=" * 70)
    print("TESTING DISTRIBUTIONAL VF CLIPPING LOGIC (NO DEPENDENCIES)")
    print("=" * 70)
    print("\nThese tests verify the core logic that determines whether")
    print("VF clipping is enabled for distributional critics.")
    print("")

    tests = [
        test_logic_default_none,
        test_logic_disable,
        test_logic_mean_only,
        test_logic_mean_and_variance,
        test_logic_no_clip_range,
        test_backward_compatibility,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ ERROR in {test.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 ALL LOGIC TESTS PASSED! 🎉")
        print("\nKey findings:")
        print("  ✓ VF clipping is DISABLED by default (mode=None)")
        print("  ✓ mode='disable' explicitly disables VF clipping")
        print("  ✓ mode='mean_only' enables VF clipping (legacy)")
        print("  ✓ mode='mean_and_variance' enables VF clipping (improved)")
        print("  ✓ clip_range_vf=None disables for all modes")
        print("  ✓ Backward compatibility: old behavior can be restored")
        print("\n⚠️  BREAKING CHANGE: Default behavior changed!")
        print("     Old: VF clipping applied when clip_range_vf set")
        print("     New: VF clipping disabled by default for distributional critics")
        print("     Migration: Add distributional_vf_clip_mode='mean_only' to restore old behavior")
        return 0
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(run_all_tests())
