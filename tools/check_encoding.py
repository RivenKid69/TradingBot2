#!/usr/bin/env python3
"""
Check encoding issues in markdown and YAML files.

This script checks for problematic Unicode characters that should be normalized.
Exit code 1 if issues found (for CI/CD).

Based on CommonMark and YAML best practices.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple


# Characters to check
# Em-dash and en-dash are ordinary punctuation and are used throughout the
# Russian-language documentation; they render correctly everywhere, so they are
# not listed here. What remains are the characters that look like something they
# are not, or that break parsers and diffs.
PROBLEMATIC_CHARS = {
    "\u2011": "non-breaking hyphen (looks like -, breaks search; use -)",
    "\u00a0": "non-breaking space (looks like a space; use a space)",
    "\u200b": "zero-width space (invisible)",
    "\ufeff": "BOM (should be removed)",
}


def check_file(filepath: Path) -> Dict[str, List[Tuple[int, str]]]:
    """
    Check file for problematic characters.

    Returns:
        Dict mapping character_name -> list of (line_number, line_content)
    """
    issues = {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            for char, name in PROBLEMATIC_CHARS.items():
                if char in line:
                    if name not in issues:
                        issues[name] = []
                    # Truncate long lines
                    line_preview = line.strip()
                    if len(line_preview) > 80:
                        line_preview = line_preview[:77] + "..."
                    issues[name].append((line_num, line_preview))

    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)

    return issues


def main():
    # Find all markdown and YAML files
    root = Path(".")
    patterns = ["*.md", "*.yaml", "*.yml"]
    exclude_dirs = {".git", "node_modules", "venv", "__pycache__", ".pytest_cache"}

    files = []
    for pattern in patterns:
        for f in root.rglob(pattern):
            if not any(part in exclude_dirs for part in f.parts):
                files.append(f)

    if not files:
        print("No files found to check")
        return 0

    # Check all files
    total_issues = 0
    files_with_issues = []

    for filepath in sorted(files):
        issues = check_file(filepath)

        if issues:
            files_with_issues.append(filepath)
            print(f"\n{filepath}:")

            for char_name, occurrences in issues.items():
                count = len(occurrences)
                total_issues += count
                print(f"  [ISSUE] {char_name}: {count} occurrence(s)")

                # Show first 3 occurrences
                for line_num, line_preview in occurrences[:3]:
                    print(f"    Line {line_num}: {line_preview}")

                if count > 3:
                    print(f"    ... and {count - 3} more")

    # Summary
    print("\n" + "=" * 60)
    if total_issues > 0:
        print(f"[ERROR] Found {total_issues} encoding issue(s) in {len(files_with_issues)} file(s)")
        print("\nTo fix these issues, run:")
        print("  python tools/normalize_encoding.py")
        print("=" * 60)
        return 1
    else:
        print(f"[OK] No encoding issues found in {len(files)} file(s)")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(main())
