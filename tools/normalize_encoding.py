#!/usr/bin/env python3
"""
Normalize encoding in markdown and YAML files.

This script fixes common encoding issues:
- Replaces em-dash (—) with double hyphen (--)
- Replaces non-breaking hyphen (‑) with regular hyphen (-)
- Replaces non-breaking space with regular space
- Removes zero-width spaces
- Removes BOM if present

Based on best practices:
- CommonMark spec recommends ASCII-safe markdown
- YAML spec requires UTF-8 without BOM
- Better compatibility with various editors and CI tools
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple


# Normalization rules
REPLACEMENTS = {
    "\u2014": "--",  # Em-dash → double hyphen
    "\u2013": "-",  # En-dash → single hyphen
    "\u2011": "-",  # Non-breaking hyphen → regular hyphen
    "\u00a0": " ",  # Non-breaking space → regular space
    "\u200b": "",  # Zero-width space → removed
    "\ufeff": "",  # BOM → removed
}


def normalize_content(content: str) -> Tuple[str, int]:
    """
    Normalize Unicode characters in content.

    Returns:
        Tuple of (normalized_content, change_count)
    """
    original = content
    changes = 0

    for old_char, new_char in REPLACEMENTS.items():
        count = content.count(old_char)
        if count > 0:
            content = content.replace(old_char, new_char)
            changes += count

    return content, changes


def normalize_file(filepath: Path, dry_run: bool = False) -> Tuple[bool, int]:
    """
    Normalize a single file.

    Returns:
        Tuple of (was_modified, change_count)
    """
    try:
        # Read with UTF-8 encoding
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Normalize
        normalized, changes = normalize_content(content)

        if changes == 0:
            return False, 0

        if not dry_run:
            # Write back
            with open(filepath, "w", encoding="utf-8", newline="\n") as f:
                f.write(normalized)

        return True, changes

    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return False, 0


def find_files(root: Path, patterns: List[str]) -> List[Path]:
    """Find all files matching patterns."""
    files = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))

    # Exclude common directories
    exclude_dirs = {".git", "node_modules", "venv", "__pycache__", ".pytest_cache"}

    return [f for f in files if not any(part in exclude_dirs for part in f.parts)]


def main():
    parser = argparse.ArgumentParser(description="Normalize encoding in markdown and YAML files")
    parser.add_argument(
        "paths", nargs="*", help="Files or directories to process (default: current directory)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be changed without modifying files"
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["*.md", "*.yaml", "*.yml"],
        help="File patterns to process (default: *.md *.yaml *.yml)",
    )

    args = parser.parse_args()

    # Determine files to process
    if not args.paths:
        args.paths = ["."]

    all_files = []
    for path_str in args.paths:
        path = Path(path_str)

        if path.is_file():
            all_files.append(path)
        elif path.is_dir():
            all_files.extend(find_files(path, args.patterns))
        else:
            print(f"Warning: {path} does not exist", file=sys.stderr)

    if not all_files:
        print("No files found to process")
        return 1

    # Process files
    total_files = 0
    total_changes = 0
    modified_files = []

    for filepath in sorted(all_files):
        was_modified, changes = normalize_file(filepath, args.dry_run)

        if was_modified:
            total_files += 1
            total_changes += changes
            modified_files.append((filepath, changes))

            status = "[DRY RUN]" if args.dry_run else "[MODIFIED]"
            print(f"{status} {filepath}: {changes} changes")

    # Summary
    print("\n" + "=" * 60)
    if args.dry_run:
        print(f"Dry run completed:")
    else:
        print(f"Normalization completed:")
    print(f"  Files modified: {total_files}")
    print(f"  Total changes: {total_changes}")
    print("=" * 60)

    if modified_files and not args.dry_run:
        print("\nModified files:")
        for filepath, changes in modified_files:
            print(f"  - {filepath}: {changes} changes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
