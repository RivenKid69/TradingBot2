#!/usr/bin/env python3
"""
Comprehensive audit of obs_builder.pyx feature indices.
Verifies that feature_idx increments correctly to reach 63 total features.
"""

import re
from pathlib import Path


def audit_obs_builder():
    """Parse obs_builder.pyx and track feature_idx increments."""

    obs_builder_path = Path("/home/user/AI-Powered Quantitative Research Platform/obs_builder.pyx")

    if not obs_builder_path.exists():
        print(f"❌ File not found: {obs_builder_path}")
        return False

    with open(obs_builder_path, "r") as f:
        lines = f.readlines()

    print("=" * 80)
    print("AUDIT: obs_builder.pyx Feature Index Tracking")
    print("=" * 80)
    print()

    # Track feature assignments
    feature_assignments = []
    current_idx = 0
    in_function = False

    for line_num, line in enumerate(lines, 1):
        # Check if we're in build_observation_vector function
        if "def build_observation_vector" in line or "cpdef void build_observation_vector" in line:
            in_function = True
            print(f"Line {line_num}: Found build_observation_vector function")
            continue

        if not in_function:
            continue

        # Check for feature_idx initialization
        if "feature_idx = 0" in line:
            print(f"Line {line_num}: feature_idx initialized to 0")
            current_idx = 0
            continue

        # Check for out_features assignments
        if "out_features[feature_idx]" in line and "=" in line:
            # Extract comment if exists
            comment = ""
            if "#" in line:
                comment = line.split("#", 1)[1].strip()

            feature_assignments.append(
                {"line": line_num, "index": current_idx, "code": line.strip(), "comment": comment}
            )

        # Check for feature_idx increments
        if "feature_idx += 1" in line or "feature_idx = feature_idx + 1" in line:
            current_idx += 1

    # Display feature assignments
    print(f"\nTotal feature assignments found: {len(feature_assignments)}")
    print(f"Final feature_idx value: {current_idx}")
    print()

    if current_idx == 63:
        print("✅ CORRECT: feature_idx reaches 63 (expected for 63-feature system)")
    else:
        print(f"❌ ERROR: feature_idx reaches {current_idx}, expected 63")
        return False

    # Check for critical features
    print("\n" + "=" * 80)
    print("Verifying Critical Features")
    print("=" * 80)

    # Search for ATR validity flag
    atr_valid_found = False
    vol_proxy_checks_atr = False

    for line_num, line in enumerate(lines, 1):
        if "atr_valid = not isnan(atr)" in line:
            print(f"Line {line_num}: ✅ Found atr_valid flag assignment")
            atr_valid_found = True

        if "if atr_valid:" in line and "vol_proxy" in lines[line_num]:
            print(f"Line {line_num}: ✅ Found vol_proxy checking atr_valid")
            vol_proxy_checks_atr = True

    if not atr_valid_found:
        print("❌ ERROR: atr_valid flag not found!")
        return False

    if not vol_proxy_checks_atr:
        print("❌ WARNING: vol_proxy may not be checking atr_valid flag")

    # Display feature index mapping (first 30 to see the pattern)
    print("\n" + "=" * 80)
    print("Feature Index Mapping (first 30 features)")
    print("=" * 80)

    for i, assignment in enumerate(feature_assignments[:30]):
        comment = f" // {assignment['comment']}" if assignment["comment"] else ""
        print(
            f"[{assignment['index']:2d}] Line {assignment['line']:4d}: {assignment['code'][:60]}{comment}"
        )

    if len(feature_assignments) > 30:
        print(f"... ({len(feature_assignments) - 30} more features)")

    return True


def check_feature_config():
    """Verify feature_config.py has correct size."""
    print("\n" + "=" * 80)
    print("AUDIT: feature_config.py")
    print("=" * 80)

    config_path = Path("/home/user/AI-Powered Quantitative Research Platform/feature_config.py")

    with open(config_path, "r") as f:
        content = f.read()

    # Check indicators-related blocks (ma5 + ma20 + indicators + bb_context = 2+2+14+2 = 20 total)
    # NOTE: Structure was updated to split the old "indicators (20)" block into granular blocks
    has_ma5 = '"ma5"' in content and '"size": 2' in content
    has_ma20 = '"ma20"' in content and '"size": 2' in content
    has_indicators = '"indicators"' in content and '"size": 14' in content
    has_bb_context = '"bb_context"' in content and '"size": 2' in content

    if has_ma5 and has_ma20 and has_indicators and has_bb_context:
        print("OK: Indicators structure = ma5(2) + ma20(2) + indicators(14) + bb_context(2) = 20")
    else:
        print("ERROR: Indicators structure not found or incorrect")
        print(
            f"  ma5: {has_ma5}, ma20: {has_ma20}, indicators: {has_indicators}, bb_context: {has_bb_context}"
        )
        return False

    # Compute total from layout
    import sys

    sys.path.insert(0, "/home/user/AI-Powered Quantitative Research Platform")

    try:
        from feature_config import N_FEATURES, FEATURES_LAYOUT

        print(f"\nN_FEATURES = {N_FEATURES}")

        if N_FEATURES == 63:
            print("✅ N_FEATURES = 63 (correct)")
        else:
            print(f"❌ ERROR: N_FEATURES = {N_FEATURES}, expected 63")
            return False

        print("\nFeature blocks:")
        total = 0
        for block in FEATURES_LAYOUT:
            name = block["name"]
            size = block["size"]
            print(f"  {name:15s}: {size:2d}")
            total += size

        print(f"  {'TOTAL':15s}: {total:2d}")

        if total == 63:
            print("✅ Total = 63 (correct)")
        else:
            print(f"❌ ERROR: Total = {total}, expected 63")
            return False

    except Exception as e:
        print(f"❌ ERROR importing feature_config: {e}")
        return False

    return True


def check_documentation():
    """Verify documentation is consistent."""
    print("\n" + "=" * 80)
    print("AUDIT: Documentation Consistency")
    print("=" * 80)

    checks = []

    # Check FEATURE_MAPPING_63.md
    mapping_path = Path(
        "/home/user/AI-Powered Quantitative Research Platform/FEATURE_MAPPING_63.md"
    )
    if mapping_path.exists():
        with open(mapping_path, "r") as f:
            content = f.read()

        if "63 features" in content.lower():
            checks.append(("FEATURE_MAPPING_63.md mentions 63 features", True))
        else:
            checks.append(("FEATURE_MAPPING_63.md mentions 63 features", False))

        if "atr_valid" in content.lower() or "index 16" in content.lower():
            checks.append(("FEATURE_MAPPING_63.md documents atr_valid", True))
        else:
            checks.append(("FEATURE_MAPPING_63.md documents atr_valid", False))
    else:
        checks.append(("FEATURE_MAPPING_63.md exists", False))

    # Check MIGRATION_GUIDE
    migration_path = Path(
        "/home/user/AI-Powered Quantitative Research Platform/MIGRATION_GUIDE_62_TO_63.md"
    )
    if migration_path.exists():
        with open(migration_path, "r") as f:
            content = f.read()

        if "62" in content and "63" in content:
            checks.append(("MIGRATION_GUIDE_62_TO_63.md mentions migration", True))
        else:
            checks.append(("MIGRATION_GUIDE_62_TO_63.md mentions migration", False))
    else:
        checks.append(("MIGRATION_GUIDE_62_TO_63.md exists", False))

    # Display results
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")

    return all(passed for _, passed in checks)


def main():
    """Run all audits."""
    print("\n" + "=" * 80)
    print(" COMPREHENSIVE AUDIT: 62 → 63 Features Migration")
    print("=" * 80)
    print()

    results = []

    # Audit 1: obs_builder.pyx
    try:
        result = audit_obs_builder()
        results.append(("obs_builder.pyx", result))
    except Exception as e:
        print(f"❌ ERROR auditing obs_builder.pyx: {e}")
        import traceback

        traceback.print_exc()
        results.append(("obs_builder.pyx", False))

    # Audit 2: feature_config.py
    try:
        result = check_feature_config()
        results.append(("feature_config.py", result))
    except Exception as e:
        print(f"❌ ERROR auditing feature_config.py: {e}")
        import traceback

        traceback.print_exc()
        results.append(("feature_config.py", False))

    # Audit 3: Documentation
    try:
        result = check_documentation()
        results.append(("Documentation", result))
    except Exception as e:
        print(f"❌ ERROR auditing documentation: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Documentation", False))

    # Summary
    print("\n" + "=" * 80)
    print(" AUDIT SUMMARY")
    print("=" * 80)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n🎉 ALL AUDITS PASSED!")
        print("✅ Migration 62 → 63 is CORRECT")
        return 0
    else:
        print("\n⚠️  SOME AUDITS FAILED!")
        print("Please review the errors above.")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
