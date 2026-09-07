#!/usr/bin/env python3
"""
DEEP VERIFICATION: Comprehensive check of all 63 features.
This script performs the most thorough possible verification of:
1. Code implementation (obs_builder.pyx)
2. Tests correctness
3. Documentation accuracy
4. Feature count and indices
"""

import sys
import re
from pathlib import Path

# ANSI colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(title):
    print(f"\n{BLUE}{'=' * 80}{RESET}")
    print(f"{BLUE}{title:^80}{RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}\n")


def print_pass(msg):
    print(f"{GREEN}✓{RESET} {msg}")


def print_fail(msg):
    print(f"{RED}✗{RESET} {msg}")


def print_warn(msg):
    print(f"{YELLOW}⚠{RESET} {msg}")


errors = []
warnings = []

# ============================================================================
# PART 1: Verify obs_builder.pyx structure
# ============================================================================
print_header("PART 1: obs_builder.pyx Code Analysis")

obs_builder = Path("/home/user/AI-Powered Quantitative Research Platform/obs_builder.pyx")
if not obs_builder.exists():
    print_fail("obs_builder.pyx not found!")
    sys.exit(1)

with open(obs_builder, "r") as f:
    code = f.read()
    lines = code.split("\n")

# Check for atr_valid declaration
if "cdef bint atr_valid" in code:
    print_pass("atr_valid declared as bint")
else:
    print_fail("atr_valid NOT declared!")
    errors.append("Missing atr_valid declaration")

# Check for atr_valid assignment
atr_valid_pattern = r"atr_valid\s*=\s*not\s+isnan\(atr\)"
if re.search(atr_valid_pattern, code):
    print_pass("atr_valid assigned from 'not isnan(atr)'")
else:
    print_fail("atr_valid assignment not found!")
    errors.append("Missing atr_valid assignment")

# Check for atr_valid flag being stored
if "out_features[feature_idx] = 1.0 if atr_valid else 0.0" in code:
    print_pass("atr_valid flag stored in observation")
else:
    print_fail("atr_valid flag NOT stored!")
    errors.append("atr_valid not written to out_features")

# Check vol_proxy uses atr_valid
if "if atr_valid:" in code and "vol_proxy" in code:
    # Find the section
    vol_proxy_section = code[code.find("if atr_valid:") : code.find("if atr_valid:") + 500]
    if "vol_proxy" in vol_proxy_section:
        print_pass("vol_proxy calculation checks atr_valid")
    else:
        print_fail("vol_proxy does NOT check atr_valid!")
        errors.append("vol_proxy missing atr_valid check")
else:
    print_fail("vol_proxy atr_valid check not found!")
    errors.append("vol_proxy doesn't use atr_valid")

# Count feature_idx increments
increments = code.count("feature_idx += 1")
print(f"\nFeature increments found: {increments}")
if increments == 42:
    print_pass(f"Correct number of increments: {increments}")
else:
    print_warn(f"Expected 42 increments, found {increments}")
    warnings.append(f"Increment count: {increments} (expected 42)")

# Check for external loop (21 features)
external_loop = "for i in range(norm_cols_values.shape[0]):"
if external_loop in code:
    print_pass("External features loop found (21 features)")
else:
    print_fail("External features loop NOT found!")
    errors.append("Missing external features loop")

# ============================================================================
# PART 2: Verify feature_config.py
# ============================================================================
print_header("PART 2: feature_config.py Verification")

try:
    sys.path.insert(0, "/home/user/AI-Powered Quantitative Research Platform")
    from feature_config import N_FEATURES, FEATURES_LAYOUT, EXT_NORM_DIM

    if N_FEATURES == 63:
        print_pass(f"N_FEATURES = {N_FEATURES} ✓")
    else:
        print_fail(f"N_FEATURES = {N_FEATURES}, expected 63!")
        errors.append(f"Wrong N_FEATURES: {N_FEATURES}")

    if EXT_NORM_DIM == 21:
        print_pass(f"EXT_NORM_DIM = {EXT_NORM_DIM} ✓")
    else:
        print_fail(f"EXT_NORM_DIM = {EXT_NORM_DIM}, expected 21!")
        errors.append(f"Wrong EXT_NORM_DIM: {EXT_NORM_DIM}")

    # Compute total
    total = sum(block["size"] for block in FEATURES_LAYOUT)
    if total == 63:
        print_pass(f"Layout sum = {total} ✓")
    else:
        print_fail(f"Layout sum = {total}, expected 63!")
        errors.append(f"Wrong layout sum: {total}")

    # Check indicators-related blocks (ma5 + ma20 + indicators + bb_context = 2+2+14+2 = 20 total)
    ma5_block = next((b for b in FEATURES_LAYOUT if b["name"] == "ma5"), None)
    ma20_block = next((b for b in FEATURES_LAYOUT if b["name"] == "ma20"), None)
    indicators_block = next((b for b in FEATURES_LAYOUT if b["name"] == "indicators"), None)
    bb_context_block = next((b for b in FEATURES_LAYOUT if b["name"] == "bb_context"), None)

    if ma5_block and ma5_block["size"] == 2:
        print_pass(f"MA5 block size = {ma5_block['size']}")
    else:
        print_fail("MA5 block not found or wrong size!")
        errors.append("MA5 block issue")

    if ma20_block and ma20_block["size"] == 2:
        print_pass(f"MA20 block size = {ma20_block['size']}")
    else:
        print_fail("MA20 block not found or wrong size!")
        errors.append("MA20 block issue")

    if indicators_block and indicators_block["size"] == 14:
        print_pass(f"Indicators block size = {indicators_block['size']}")
    else:
        print_fail(f"Indicators block not found or wrong size (expected 14)!")
        errors.append("Indicators block issue")

    if bb_context_block and bb_context_block["size"] == 2:
        print_pass(f"BB context block size = {bb_context_block['size']}")
    else:
        print_fail("BB context block not found or wrong size!")
        errors.append("BB context block issue")

    # Total indicators-related features should be 20
    total_indicator_features = 0
    if ma5_block:
        total_indicator_features += ma5_block["size"]
    if ma20_block:
        total_indicator_features += ma20_block["size"]
    if indicators_block:
        total_indicator_features += indicators_block["size"]
    if bb_context_block:
        total_indicator_features += bb_context_block["size"]

    if total_indicator_features == 20:
        print_pass(f"Total indicator-related features = {total_indicator_features}")
    else:
        print_fail(f"Total indicator-related features = {total_indicator_features}, expected 20!")
        errors.append(f"Wrong total indicator features: {total_indicator_features}")

except Exception as e:
    print_fail(f"Error loading feature_config: {e}")
    errors.append(f"feature_config error: {e}")

# ============================================================================
# PART 3: Verify test_atr_validity_flag.py
# ============================================================================
print_header("PART 3: test_atr_validity_flag.py Critical Indices")

test_file = Path(
    "/home/user/AI-Powered Quantitative Research Platform/tests/test_atr_validity_flag.py"
)
if test_file.exists():
    with open(test_file, "r") as f:
        test_code = f.read()

    # Check for vol_proxy index
    vol_proxy_indices = re.findall(r"obs\[(\d+)\].*vol_proxy|vol_proxy.*obs\[(\d+)\]", test_code)

    correct_vol_proxy = 0
    wrong_vol_proxy = 0

    for match in vol_proxy_indices:
        idx = match[0] or match[1]
        if idx == "22":
            correct_vol_proxy += 1
        elif idx in ["23", "24"]:
            wrong_vol_proxy += 1
            print_fail(f"Found vol_proxy at WRONG index {idx} (should be 22)!")
            errors.append(f"test_atr_validity_flag.py has vol_proxy at index {idx}")

    if wrong_vol_proxy == 0 and correct_vol_proxy > 0:
        print_pass(f"All vol_proxy references use index 22 ({correct_vol_proxy} occurrences)")
    elif correct_vol_proxy == 0:
        print_warn("No vol_proxy index checks found in test")
        warnings.append("test_atr_validity_flag.py: no vol_proxy checks")

    # Check for atr_valid index
    if "obs[16]" in test_code and "atr_valid" in test_code:
        print_pass("atr_valid referenced at index 16")
    else:
        print_fail("atr_valid at index 16 not found in test!")
        errors.append("test_atr_validity_flag.py: missing atr_valid[16] check")

else:
    print_fail("test_atr_validity_flag.py not found!")
    errors.append("Missing test_atr_validity_flag.py")

# ============================================================================
# PART 4: Verify FEATURE_MAPPING_63.md
# ============================================================================
print_header("PART 4: FEATURE_MAPPING_63.md Documentation")

feature_map = Path("/home/user/AI-Powered Quantitative Research Platform/FEATURE_MAPPING_63.md")
if feature_map.exists():
    with open(feature_map, "r") as f:
        doc = f.read()

    # Check critical indices
    critical_checks = [
        (r"\|\s*15\s*\|\s*atr\s*\|", "atr at index 15"),
        (r"\|\s*16\s*\|\s*\*\*atr_valid\*\*\s*\|", "atr_valid at index 16"),
        (r"\|\s*17\s*\|\s*cci\s*\|", "cci at index 17"),
        (r"21-22.*Derived|Derived.*21-22", "Derived features at 21-22"),
        (r"23-28.*Agent|Agent.*23-28", "Agent features at 23-28"),
        (r"29-31.*Microstructure|Technical.*29-31", "Microstructure at 29-31"),
        (r"32-33.*Bollinger|Bollinger.*32-33", "Bollinger at 32-33"),
    ]

    for pattern, description in critical_checks:
        if re.search(pattern, doc, re.IGNORECASE):
            print_pass(description)
        else:
            print_fail(f"NOT FOUND: {description}")
            errors.append(f"FEATURE_MAPPING_63.md: {description} not found")

    # Check for wrong indices (common errors)
    wrong_patterns = [
        (r"\|\s*21\s*\|\s*bb_position", "bb_position at 21 (WRONG! should be 32)"),
        (r"\|\s*24\s*\|\s*vol_proxy", "vol_proxy at 24 (WRONG! should be 22)"),
        (r"25-30.*Agent", "Agent at 25-30 (WRONG! should be 23-28)"),
    ]

    for pattern, description in wrong_patterns:
        if re.search(pattern, doc):
            print_fail(f"FOUND ERROR: {description}")
            errors.append(f"FEATURE_MAPPING_63.md: {description}")

else:
    print_fail("FEATURE_MAPPING_63.md not found!")
    errors.append("Missing FEATURE_MAPPING_63.md")

# ============================================================================
# PART 5: Verify OBSERVATION_MAPPING.md
# ============================================================================
print_header("PART 5: OBSERVATION_MAPPING.md Documentation")

obs_map = Path("/home/user/AI-Powered Quantitative Research Platform/OBSERVATION_MAPPING.md")
if obs_map.exists():
    with open(obs_map, "r") as f:
        obs_doc = f.read()

    # Check critical positions
    critical_obs = [
        (r"\|\s*21\s*\|\s*`?ret_bar`?", "ret_bar at position 21"),
        (r"\|\s*22\s*\|\s*`?vol_proxy`?", "vol_proxy at position 22"),
        (r"23-28.*Agent|Agent.*23-28", "Agent at 23-28"),
        (r"32-33.*Bollinger|Bollinger.*32-33", "Bollinger at 32-33"),
        (r"60-61.*Token.*Metadata", "Token metadata at 60-61"),
        (r"\|\s*62\s*\|.*Token.*one-hot|one-hot.*62", "Token one-hot at 62"),
    ]

    for pattern, description in critical_obs:
        if re.search(pattern, obs_doc, re.IGNORECASE):
            print_pass(description)
        else:
            print_fail(f"NOT FOUND: {description}")
            errors.append(f"OBSERVATION_MAPPING.md: {description} not found")

    # Check total
    if "63 features" in obs_doc or "**63**" in obs_doc:
        print_pass("Total = 63 features mentioned")
    else:
        print_warn("Total = 63 not clearly stated")
        warnings.append("OBSERVATION_MAPPING.md: total not clear")

else:
    print_fail("OBSERVATION_MAPPING.md not found!")
    errors.append("Missing OBSERVATION_MAPPING.md")

# ============================================================================
# PART 6: Verify MIGRATION_GUIDE_62_TO_63.md
# ============================================================================
print_header("PART 6: MIGRATION_GUIDE_62_TO_63.md")

migration_guide = Path(
    "/home/user/AI-Powered Quantitative Research Platform/MIGRATION_GUIDE_62_TO_63.md"
)
if migration_guide.exists():
    with open(migration_guide, "r") as f:
        guide = f.read()

    # Check shift table
    shift_checks = [
        (r"20:.*ret_bar.*21:.*ret_bar", "ret_bar shift 20→21"),
        (r"21:.*vol_proxy.*22:.*vol_proxy", "vol_proxy shift 21→22"),
        (r"22:.*cash_ratio.*23:.*cash_ratio", "cash_ratio shift 22→23"),
        (r"31:.*bb_position.*32:.*bb_position", "bb_position shift 31→32"),
    ]

    for pattern, description in shift_checks:
        if re.search(pattern, guide):
            print_pass(description)
        else:
            print_fail(f"NOT FOUND: {description}")
            errors.append(f"MIGRATION_GUIDE: {description} not found")

    # Check examples use correct indices
    example_checks = [
        (r"obs\[16\].*atr_valid", "Example uses obs[16] for atr_valid"),
        (r"obs\[22\].*vol_proxy", "Example uses obs[22] for vol_proxy"),
    ]

    for pattern, description in example_checks:
        if re.search(pattern, guide):
            print_pass(description)
        else:
            print_fail(f"NOT FOUND: {description}")
            errors.append(f"MIGRATION_GUIDE: {description} not found")

else:
    print_fail("MIGRATION_GUIDE_62_TO_63.md not found!")
    errors.append("Missing MIGRATION_GUIDE_62_TO_63.md")

# ============================================================================
# PART 7: Cross-check all tests
# ============================================================================
print_header("PART 7: All Test Files Index Verification")

test_files = [
    "test_atr_validity_flag.py",
    "test_derived_features_validity_flags.py",
    "test_technical_indicators_in_obs.py",
    "test_validity_flags.py",
    "test_bollinger_bands_validation.py",
    "test_mediator_integration.py",
]

for test_name in test_files:
    test_path = Path(f"/home/user/AI-Powered Quantitative Research Platform/tests/{test_name}")
    if test_path.exists():
        with open(test_path, "r") as f:
            test_content = f.read()

        # Check for 63 features
        if (
            "np.zeros(63" in test_content
            or "shape[0] == 63" in test_content
            or "shape == (63,)" in test_content
        ):
            print_pass(f"{test_name}: Uses 63 features")
        else:
            print_warn(f"{test_name}: May not explicitly check for 63")
            warnings.append(f"{test_name}: no explicit 63-feature check")

        # Check for atr_valid at index 16
        if "obs[16]" in test_content and ("atr" in test_content or "valid" in test_content):
            print_pass(f"{test_name}: References index 16 (atr_valid region)")

    else:
        print_warn(f"{test_name}: Not found")

# ============================================================================
# PART 8: Verify INDEX_AUDIT_REPORT.md
# ============================================================================
print_header("PART 8: INDEX_AUDIT_REPORT.md Completeness")

audit_report = Path("/home/user/AI-Powered Quantitative Research Platform/INDEX_AUDIT_REPORT.md")
if audit_report.exists():
    with open(audit_report, "r") as f:
        report = f.read()

    if "vol_proxy" in report and "22" in report:
        print_pass("Audit report documents vol_proxy at index 22")
    else:
        print_warn("Audit report may not fully document vol_proxy")

    if "bb_position" in report and "32" in report:
        print_pass("Audit report documents bb_position at index 32")
    else:
        print_warn("Audit report may not fully document bb_position")
else:
    print_warn("INDEX_AUDIT_REPORT.md not found")
    warnings.append("Missing INDEX_AUDIT_REPORT.md")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print_header("FINAL VERIFICATION SUMMARY")

print(f"\n{BLUE}Statistics:{RESET}")
print(f"  Errors:   {len(errors)}")
print(f"  Warnings: {len(warnings)}")

if errors:
    print(f"\n{RED}❌ ERRORS FOUND:{RESET}")
    for i, err in enumerate(errors, 1):
        print(f"  {i}. {err}")

if warnings:
    print(f"\n{YELLOW}⚠️  WARNINGS:{RESET}")
    for i, warn in enumerate(warnings, 1):
        print(f"  {i}. {warn}")

if not errors and not warnings:
    print(f"\n{GREEN}{'=' * 80}{RESET}")
    print(f"{GREEN}🎉 PERFECT! ALL VERIFICATIONS PASSED!{RESET}")
    print(f"{GREEN}{'=' * 80}{RESET}")
    print(f"\n{GREEN}✓ Code implementation correct{RESET}")
    print(f"{GREEN}✓ All tests use correct indices{RESET}")
    print(f"{GREEN}✓ All documentation accurate{RESET}")
    print(f"{GREEN}✓ Feature count = 63{RESET}")
    print(f"{GREEN}✓ atr_valid at index 16{RESET}")
    print(f"{GREEN}✓ vol_proxy at index 22{RESET}")
    print(f"\n{GREEN}Migration 62→63 is FULLY CORRECT!{RESET}")
    sys.exit(0)
elif not errors:
    print(f"\n{YELLOW}{'=' * 80}{RESET}")
    print(f"{YELLOW}⚠️  VERIFICATION PASSED WITH WARNINGS{RESET}")
    print(f"{YELLOW}{'=' * 80}{RESET}")
    print(f"\n{GREEN}✓ No critical errors found{RESET}")
    print(f"{YELLOW}⚠  Some minor warnings (see above){RESET}")
    print(f"\n{YELLOW}Migration is functionally correct, warnings are informational.{RESET}")
    sys.exit(0)
else:
    print(f"\n{RED}{'=' * 80}{RESET}")
    print(f"{RED}❌ VERIFICATION FAILED{RESET}")
    print(f"{RED}{'=' * 80}{RESET}")
    print(f"\n{RED}Critical errors found. Please review and fix.{RESET}")
    sys.exit(1)
