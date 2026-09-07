#!/usr/bin/env python3
"""
Documentation Reorganization Script

This script reorganizes the AI-Powered Quantitative Research Platform documentation by moving files
from the root directory to appropriate subdirectories.

Usage:
    python scripts/reorganize_docs.py [--dry-run] [--verbose]

Options:
    --dry-run    Show what would be moved without actually moving files
    --verbose    Show detailed output
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List
import argparse


# File categorization rules
CATEGORIZATION = {
    # Bug reports
    "docs/reports/bugs": [
        "BUG_REPORT_*.md",
        "BUG_FIX_*.md",
        "BUG_FIXES_*.md",
        "BUG_LOCALIZATION_*.md",
        "*_BUG_FIX_*.md",
        "RSI_BUG_*.md",
        "CRITICAL_BUGS_*.md",
        "PARKINSON_ERROR_*.md",
    ],
    # Audit reports
    "docs/reports/audits": [
        "AUDIT_*.md",
        "*_AUDIT_*.md",
        "DEEP_AUDIT_*.md",
        "SEASONALITY_AUDIT_*.md",
        "FEATURE_AUDIT_*.md",
        "DOCUMENTATION_AUDIT_*.md",
        "MA5_AUDIT_*.md",
        "MA20_INDICATOR_AUDIT_*.md",
        "TWIN_CRITICS_*_AUDIT_*.md",
    ],
    # Integration reports
    "docs/reports/integration": [
        "INTEGRATION_*.md",
        "*_INTEGRATION_*.md",
        "API_FIX_*.md",
        "MIGRATION_*.md",
        "PYDANTIC_V2_MIGRATION_*.md",
    ],
    # Feature reports
    "docs/reports/features": [
        "FEATURE_MAPPING_*.md",
        "FEATURE_ADAPTATION_*.md",
        "CURRENT_FEATURE_*.md",
        "FULL_FEATURES_*.md",
        "OBSERVATION_MAPPING.md",
        "GARCH_*.md",
        "TAKER_BUY_RATIO_*.md",
        "TBR_MOMENTUM_*.md",
    ],
    # Analysis reports
    "docs/reports/analysis": [
        "ANALYSIS_*.md",
        "NORMALIZATION_ANALYSIS.md",
        "KL_DIVERGENCE_ANALYSIS.md",
        "PARKINSON_ANALYSIS_*.md",
        "LAGRANGIAN_GRADIENT_FLOW_ANALYSIS.md",
        "VF_VARIANCE_DEEP_ANALYSIS.md",
        "VF_CLIPPING_ANALYSIS_*.md",
        "TRAINING_METRICS_ANALYSIS.md",
        "TRAINING_PIPELINE_ANALYSIS.md",
        "CODEBASE_STRUCTURE_ANALYSIS.md",
        "PROJECT_STRUCTURE_ANALYSIS.md",
        "SIZE_ANALYSIS.md",
        "DETAILED_FEATURE_CORRUPTION_ANALYSIS.md",
    ],
    # Fix reports
    "docs/reports/fixes": [
        "*_FIX.md",
        "*_FIX_*.md",
        "*_FIXES_*.md",
        "DISTRIBUTIONAL_VF_*.md",
        "CATEGORICAL_VF_*.md",
        "PER_QUANTILE_*.md",
        "COMPREHENSIVE_DDOF_*.md",
        "DDOF_FIX_*.md",
        "QUANTILE_*.md",
        "CVAR_LAGRANGIAN_*.md",
        "HPO_DATA_LEAKAGE_*.md",
        "FORWARD_LOOKING_BIAS_*.md",
        "ADVANTAGE_STD_FLOOR_*.md",
        "FINAL_SUMMARY_NAN_*.md",
        "NORMALIZATION_RECOMMENDATIONS.md",
        "DATASET_FIX_*.md",
        "SEASONALITY_FIXES.md",
        "YANG_ZHANG_FIX_*.md",
        "TORCH_LOAD_SECURITY_*.md",
        "ISSUE8_FIX_*.md",
        "CRITICAL_FIX_*.md",
        "FIXES_SUMMARY.md",
        "METRICS_FIXES_*.md",
        "DOCS_LOG_RATIO_FIX.md",
    ],
    # Test and verification reports
    "docs/reports/tests": [
        "TEST_*.md",
        "VERIFICATION_*.md",
        "*_VERIFICATION_*.md",
        "COMPREHENSIVE_TEST_*.md",
        "DEEP_VALIDATION_*.md",
        "DEEP_VERIFICATION_*.md",
        "FINAL_VERIFICATION_*.md",
        "PARKINSON_TESTING_*.md",
        "SELF_AUDIT_VERIFICATION.md",
    ],
    # UPGD and VGS reports
    "docs/reports/upgd_vgs": [
        "UPGD_*.md",
        "VGS_*.md",
        "vgs_param_*.md",
    ],
    # Twin Critics reports
    "docs/reports/twin_critics": [
        "TWIN_CRITICS_*.md",
    ],
    # Self-review and critical analysis
    "docs/reports/self_review": [
        "SELF_REVIEW_*.md",
        "SELF_AUDIT_*.md",
        "MA5_CRITICAL_*.md",
        "CRITICAL_REVIEW.md",
        "VERDICT_*.md",
    ],
    # Summary documents
    "docs/reports/summaries": [
        "*_SUMMARY.md",
        "CHANGES_SUMMARY.md",
        "DEEP_AUDIT_FIXES_SUMMARY.md",
    ],
    # Archive - deprecated or old documents
    "docs/archive/deprecated": [
        "DOCUMENTATION_AUDIT_2025-11-11.md",  # Specific old audit
    ],
}

# Files that should stay in root
KEEP_IN_ROOT = {
    "README.md",
    "docs/PLATFORM_REFERENCE.md",
    "docs/PLATFORM_REFERENCE.md",  # lowercase version
    "ARCHITECTURE.md",
    "ARCHITECTURE_DIAGRAM.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "BUILD_INSTRUCTIONS.md",
    "QUICK_START_REFERENCE.md",
    "FILE_REFERENCE.md",
    "DOCS_INDEX.md",
    "COMPILATION_REPORT.md",
    "BUG_09_QUICK_REFERENCE.md",
    "VERIFICATION_INSTRUCTIONS.md",
    "DOCUMENTATION_INDEX.md",
}


def match_pattern(filename: str, pattern: str) -> bool:
    """Check if filename matches a glob-like pattern."""
    import fnmatch

    return fnmatch.fnmatch(filename, pattern)


def categorize_file(filename: str) -> str | None:
    """Determine the destination directory for a file."""
    if filename in KEEP_IN_ROOT:
        return None

    for dest_dir, patterns in CATEGORIZATION.items():
        for pattern in patterns:
            if match_pattern(filename, pattern):
                return dest_dir

    # Files not matching any pattern go to archive
    return "docs/archive/uncategorized"


def reorganize_docs(root_dir: Path, dry_run: bool = False, verbose: bool = False):
    """Reorganize documentation files."""
    moved_files = []
    skipped_files = []

    # Get all markdown files in root
    md_files = list(root_dir.glob("*.md"))

    print(f"Found {len(md_files)} markdown files in root directory\n")

    for md_file in md_files:
        filename = md_file.name
        dest_dir = categorize_file(filename)

        if dest_dir is None:
            if verbose:
                print(f"KEEP: {filename} (staying in root)")
            continue

        dest_path = root_dir / dest_dir / filename

        # Create destination directory if it doesn't exist
        if not dry_run:
            dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file already exists in destination
        if dest_path.exists() and not dry_run:
            print(f"WARNING: {filename} already exists in {dest_dir}")
            skipped_files.append((filename, dest_dir, "already exists"))
            continue

        if dry_run:
            print(f"WOULD MOVE: {filename} -> {dest_dir}/")
            moved_files.append((filename, dest_dir))
        else:
            try:
                shutil.move(str(md_file), str(dest_path))
                print(f"MOVED: {filename} -> {dest_dir}/")
                moved_files.append((filename, dest_dir))
            except Exception as e:
                print(f"ERROR: Failed to move {filename}: {e}")
                skipped_files.append((filename, dest_dir, str(e)))

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {len(md_files)}")
    print(f"Files moved: {len(moved_files)}")
    print(f"Files skipped: {len(skipped_files)}")
    print(f"Files kept in root: {len([f for f in md_files if f.name in KEEP_IN_ROOT])}")

    if verbose and skipped_files:
        print("\n" + "=" * 80)
        print("SKIPPED FILES")
        print("=" * 80)
        for filename, dest_dir, reason in skipped_files:
            print(f"  {filename} -> {dest_dir}/ (reason: {reason})")

    if dry_run:
        print("\nDRY RUN MODE - No files were actually moved")
        print("   Run without --dry-run to perform the actual reorganization")


def create_readme_files(root_dir: Path, dry_run: bool = False):
    """Create README.md files for each documentation category."""

    readme_content = {
        "docs/reports/bugs": """# Bug Reports

This directory contains bug reports and their fixes.

## Categories
- Bug reports (BUG_REPORT_*.md)
- Bug fixes (BUG_FIX_*.md)
- Critical bugs
- Component-specific bugs

See [DOCS_INDEX.md](../../DOCS_INDEX.md) for the main documentation index.
""",
        "docs/reports/audits": """# Audit Reports

This directory contains audit reports for various system components.

## Categories
- Feature audits
- System audits
- Documentation audits
- Deep audits

See [DOCS_INDEX.md](../../DOCS_INDEX.md) for the main documentation index.
""",
        "docs/reports/integration": """# Integration & Migration Reports

This directory contains integration status reports and migration guides.

## Categories
- Integration reports
- Migration guides
- API fixes
- Pydantic migration

See [DOCS_INDEX.md](../../DOCS_INDEX.md) for the main documentation index.
""",
        "docs/reports/features": """# Feature Reports

This directory contains feature documentation, mappings, and analysis.

## Categories
- Feature mappings (56, 62, 63 features)
- Feature adaptations
- Component-specific features (GARCH, TBR, etc.)

See [DOCS_INDEX.md](../../DOCS_INDEX.md) for the main documentation index.
""",
        "docs/reports/analysis": """# Analysis Reports

This directory contains in-depth analysis reports.

## Categories
- Data analysis
- Algorithm analysis
- System analysis
- Performance analysis

See [DOCS_INDEX.md](../../DOCS_INDEX.md) for the main documentation index.
""",
        "docs/reports/fixes": """# Fix Reports

This directory contains detailed reports of bug fixes and improvements.

## Categories
- PPO & Value function fixes
- Statistical fixes
- Quantile & loss fixes
- Optimization fixes
- Data fixes

See [DOCS_INDEX.md](../../DOCS_INDEX.md) for the main documentation index.
""",
        "docs/reports/tests": """# Test & Verification Reports

This directory contains test coverage and verification reports.

## Categories
- Test coverage reports
- Verification reports
- Deep validation
- Component testing

See [DOCS_INDEX.md](../../DOCS_INDEX.md) for the main documentation index.
""",
        "docs/reports/upgd_vgs": """# UPGD & VGS Reports

This directory contains reports related to UPGD optimizer and VGS (Variance Gradient Scaling).

## Categories
- UPGD test reports
- VGS analysis
- UPGD-VGS integration

See [DOCS_INDEX.md](../../DOCS_INDEX.md) for the main documentation index.
""",
        "docs/reports/twin_critics": """# Twin Critics Reports

This directory contains reports related to the Twin Critics architecture.

## Categories
- Twin Critics audit
- Integration reports
- Configuration

See [DOCS_INDEX.md](../../DOCS_INDEX.md) for the main documentation index.
""",
        "docs/reports/self_review": """# Self-Review & Critical Analysis

This directory contains self-review reports and critical analyses.

## Categories
- Self-review reports
- Critical audits
- Verdicts and decisions

See [DOCS_INDEX.md](../../DOCS_INDEX.md) for the main documentation index.
""",
        "docs/reports/summaries": """# Summary Documents

This directory contains various summary documents.

## Categories
- Change summaries
- Fix summaries
- Audit summaries

See [DOCS_INDEX.md](../../DOCS_INDEX.md) for the main documentation index.
""",
        "docs/archive/uncategorized": """# Uncategorized Archive

This directory contains documentation that doesn't fit into other categories.

Files here should be reviewed and either:
1. Properly categorized
2. Merged with existing documentation
3. Deleted if obsolete

See [DOCS_INDEX.md](../../DOCS_INDEX.md) for the main documentation index.
""",
        "docs/archive/deprecated": """# Deprecated Documentation

This directory contains deprecated or outdated documentation.

These files are kept for historical reference but should not be used for current development.

See [DOCS_INDEX.md](../../DOCS_INDEX.md) for the main documentation index.
""",
    }

    for dir_path, content in readme_content.items():
        full_path = root_dir / dir_path / "README.md"

        if dry_run:
            print(f"WOULD CREATE: {dir_path}/README.md")
        else:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"CREATED: {dir_path}/README.md")


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize AI-Powered Quantitative Research Platform documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without actually moving files",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument(
        "--create-readmes", action="store_true", help="Create README.md files in subdirectories"
    )

    args = parser.parse_args()

    root_dir = Path(__file__).parent.parent.resolve()

    print(f"AI-Powered Quantitative Research Platform Documentation Reorganization")
    print(f"Root directory: {root_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'ACTUAL EXECUTION'}\n")

    if args.create_readmes:
        print("\n" + "=" * 80)
        print("Creating README files")
        print("=" * 80 + "\n")
        create_readme_files(root_dir, args.dry_run)

    print("\n" + "=" * 80)
    print("Reorganizing documentation files")
    print("=" * 80 + "\n")
    reorganize_docs(root_dir, args.dry_run, args.verbose)

    print("\nDone!")
    print("\nNext steps:")
    print("1. Review the moved files")
    print("2. Update docs/PLATFORM_REFERENCE.md with new documentation structure")
    print("3. Update any hardcoded paths in scripts")
    print("4. Commit the changes")


if __name__ == "__main__":
    main()
