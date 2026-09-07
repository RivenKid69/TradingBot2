#!/usr/bin/env python3
"""
Fix files with broken encoding (non-UTF-8).

This script attempts to detect and convert files with incorrect encoding
(e.g., Windows-1251, CP1252, ISO-8859-1) to UTF-8.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional


# Common encodings to try
ENCODINGS_TO_TRY = [
    "utf-8",
    "windows-1251",  # Cyrillic (Russian)
    "cp1252",  # Windows Western European
    "iso-8859-1",  # Latin-1
    "cp866",  # DOS Cyrillic
    "koi8-r",  # KOI8-R (Russian)
]


def detect_and_convert(filepath: Path, dry_run: bool = False) -> bool:
    """
    Detect file encoding and convert to UTF-8.

    Returns:
        True if conversion was successful
    """
    print(f"Processing: {filepath}")

    # Try to detect encoding
    content = None
    detected_encoding = None

    for encoding in ENCODINGS_TO_TRY:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                content = f.read()
            detected_encoding = encoding
            print(f"  [OK] Successfully read with {encoding}")
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if content is None:
        print(f"  [ERROR] Failed to decode file with any known encoding")
        return False

    # If already UTF-8, check if it's valid
    if detected_encoding == "utf-8":
        print(f"  [INFO] File is already UTF-8")
        return True

    # Convert to UTF-8
    if not dry_run:
        try:
            with open(filepath, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
            print(f"  [OK] Converted from {detected_encoding} to UTF-8")
            return True
        except Exception as e:
            print(f"  [ERROR] Failed to write UTF-8: {e}")
            return False
    else:
        print(f"  [DRY RUN] Would convert from {detected_encoding} to UTF-8")
        return True


def main():
    parser = argparse.ArgumentParser(description="Fix files with broken encoding (non-UTF-8)")
    parser.add_argument("files", nargs="+", help="Files to fix")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without modifying files"
    )

    args = parser.parse_args()

    success_count = 0
    fail_count = 0

    for file_path_str in args.files:
        file_path = Path(file_path_str)

        if not file_path.exists():
            print(f"File not found: {file_path}", file=sys.stderr)
            fail_count += 1
            continue

        if detect_and_convert(file_path, args.dry_run):
            success_count += 1
        else:
            fail_count += 1

        print()

    # Summary
    print("=" * 60)
    print(f"Conversion completed:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print("=" * 60)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
