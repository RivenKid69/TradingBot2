# -*- coding: utf-8 -*-
"""
CCEA Design Doc SHA Verification.

Verifies that the Design Doc snapshot file matches the SHA256 hash
recorded in the rendered markdown document.

This is a critical guardrail to ensure Design Doc changes are intentional
and properly tracked (WI-TRACE-01).

Usage:
    python -m ccea.guardrails.design_doc_check
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

# ============================================================================
# Constants
# ============================================================================

# Primary path (root location) - actual location of Design Doc
PRIMARY_SNAPSHOT_PATH = Path("Design Doc CCEA Cloud.txt")

# Legacy paths (for backwards compatibility)
LEGACY_SNAPSHOT_PATH = Path("docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt")
LEGACY_RENDERED_PATH = Path("docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.md")

# Default to primary path, fall back to legacy
DEFAULT_SNAPSHOT_PATH = PRIMARY_SNAPSHOT_PATH
DEFAULT_RENDERED_PATH = LEGACY_RENDERED_PATH

# Regex to extract SHA256 from rendered document
SHA256_PATTERN = re.compile(r'\*\*SHA256\*\*:\s*([a-fA-F0-9]{64})')


# ============================================================================
# SHA Verification Functions
# ============================================================================

def compute_sha256(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        Lowercase hex digest of SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest().lower()


def extract_recorded_sha256(rendered_path: Path) -> Optional[str]:
    """
    Extract recorded SHA256 from the rendered markdown document.

    Args:
        rendered_path: Path to rendered markdown file

    Returns:
        Recorded SHA256 hash or None if not found
    """
    try:
        with open(rendered_path, "r", encoding="utf-8") as f:
            content = f.read()

        match = SHA256_PATTERN.search(content)
        if match:
            return match.group(1).lower()
        return None
    except FileNotFoundError:
        return None


def find_design_doc_path() -> Optional[Path]:
    """
    Find the Design Doc snapshot file.

    Searches in multiple locations for compatibility.

    Returns:
        Path to Design Doc or None if not found
    """
    # Try paths in order of preference
    candidates = [
        PRIMARY_SNAPSHOT_PATH,
        LEGACY_SNAPSHOT_PATH,
        Path("Design Doc CCEA Cloud.txt"),
        Path("docs/Design Doc CCEA Cloud.txt"),
        Path("design_doc.txt"),
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def verify_design_doc_sha(
    snapshot_path: Optional[Path] = None,
    rendered_path: Path = DEFAULT_RENDERED_PATH,
) -> Tuple[bool, str]:
    """
    Verify that Design Doc snapshot matches recorded SHA256.

    Args:
        snapshot_path: Path to snapshot .txt file (auto-detected if None)
        rendered_path: Path to rendered .md file

    Returns:
        Tuple of (passed, message)
    """
    # Auto-detect snapshot path if not provided
    if snapshot_path is None:
        snapshot_path = find_design_doc_path()
        if snapshot_path is None:
            return False, (
                f"Design Doc not found. Searched:\n"
                f"  - {PRIMARY_SNAPSHOT_PATH}\n"
                f"  - {LEGACY_SNAPSHOT_PATH}\n"
                "Please ensure Design Doc is in one of these locations."
            )

    # Check snapshot exists
    if not snapshot_path.exists():
        return False, f"Snapshot file not found: {snapshot_path}"

    # Check rendered doc exists
    if not rendered_path.exists():
        return False, f"Rendered document not found: {rendered_path}"

    # Compute actual SHA256
    actual_sha256 = compute_sha256(snapshot_path)

    # Extract recorded SHA256
    recorded_sha256 = extract_recorded_sha256(rendered_path)

    if recorded_sha256 is None:
        return False, f"No SHA256 found in {rendered_path}. Add '**SHA256**: <hash>' line."

    # Compare
    if actual_sha256 == recorded_sha256:
        return True, f"SHA256 match: {actual_sha256}"
    else:
        return False, (
            f"SHA256 mismatch!\n"
            f"  Snapshot: {actual_sha256}\n"
            f"  Recorded: {recorded_sha256}\n"
            f"  If the snapshot was intentionally updated, update the SHA256 in {rendered_path}"
        )


# ============================================================================
# CLI Interface
# ============================================================================

def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CCEA Design Doc SHA Verification"
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Path to Design Doc snapshot (.txt) (auto-detects if omitted)",
    )
    parser.add_argument(
        "--rendered",
        type=Path,
        default=DEFAULT_RENDERED_PATH,
        help="Path to rendered Design Doc (.md)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update SHA256 in rendered doc (for authorized updates only)",
    )

    args = parser.parse_args()

    snapshot_path = args.snapshot if args.snapshot is not None else find_design_doc_path()

    print(f"Checking Design Doc SHA256...")
    print(f"  Snapshot: {snapshot_path}")
    print(f"  Rendered: {args.rendered}")

    if args.update:
        # Update mode: compute new SHA and update rendered doc
        if snapshot_path is None or not snapshot_path.exists():
            print(f"\n[FAIL] Snapshot not found: {snapshot_path}")
            return 1

        new_sha = compute_sha256(snapshot_path)

        if not args.rendered.exists():
            print(f"\n[FAIL] Rendered doc not found: {args.rendered}")
            return 1

        with open(args.rendered, "r", encoding="utf-8") as f:
            content = f.read()

        old_sha = extract_recorded_sha256(args.rendered)

        if old_sha:
            content = content.replace(old_sha, new_sha)
        else:
            # Insert SHA after Version/Date/Status lines
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("> **Status**"):
                    lines.insert(i + 1, f"> **SHA256**: {new_sha}")
                    break
            content = "\n".join(lines)

        with open(args.rendered, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"\n[UPDATED] SHA256 updated to: {new_sha}")
        return 0

    # Verify mode
    passed, message = verify_design_doc_sha(snapshot_path, args.rendered)

    if passed:
        print(f"\n[PASS] {message}")
        return 0
    else:
        print(f"\n[FAIL] {message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
