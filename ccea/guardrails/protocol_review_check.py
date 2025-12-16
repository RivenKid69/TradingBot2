# -*- coding: utf-8 -*-
"""
Protocol Change Review Gate - CI Enforcement.

GDPR Phase 2 Implementation: Ensures new protocol changes have security review approval.

This module:
    - Detects protocol changes in PRs
    - Verifies security review approval exists in journal
    - Fails CI if approval is missing

Design Doc Reference:
    - docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1043 (security review)
    - docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L960 (change management)
    - docs/compliance/PROTOCOL_CHANGE_REVIEW.md

Usage:
    python -m ccea.guardrails.protocol_review_check

Exit codes:
    0 - Check passed (no protocol changes, or approval found)
    1 - Check failed (protocol changes without approval)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# Constants
# ============================================================================

# Journal file location
JOURNAL_PATH = Path("docs/compliance/protocol_change_journal.json")

# Files that constitute protocol changes
PROTOCOL_PATHS = [
    "ccea/models/protocol.py",
    "ccea/protocol/",
    "docs/schemas/protocol_messages.schema.json",
    "docs/schemas/artifact_manifest.schema.json",
    "packages/cloud/control_plane/services/command_service.py",
]

# Patterns that indicate protocol changes
PROTOCOL_PATTERNS = [
    "class.*Command.*BaseCommand",
    "class.*Message.*BaseMessage",
    r"CommandType\.",
    r"MessageType\.",
    r"TelemetryLevel\.",
]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ProtocolChange:
    """Detected protocol change."""
    file_path: str
    change_type: str  # added, modified, deleted
    patterns_matched: List[str]


@dataclass
class JournalEntry:
    """Protocol change journal entry."""
    change_id: str
    review_id: str
    timestamp: datetime
    change_type: str
    approval_status: str
    commit_hash: Optional[str]
    conditions_met: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JournalEntry":
        """Create from dictionary."""
        return cls(
            change_id=data.get("change_id", ""),
            review_id=data.get("review_id", ""),
            timestamp=datetime.fromisoformat(data.get("timestamp", "").replace("Z", "+00:00")),
            change_type=data.get("change_type", ""),
            approval_status=data.get("approval_status", ""),
            commit_hash=data.get("commit_hash"),
            conditions_met=data.get("conditions_met", False),
        )


@dataclass
class CheckResult:
    """Result of protocol review check."""
    passed: bool
    message: str
    protocol_changes: List[ProtocolChange]
    approval_found: Optional[JournalEntry]


# ============================================================================
# Detection Functions
# ============================================================================

def get_changed_files(base_branch: str = "origin/main") -> List[Tuple[str, str]]:
    """
    Get files changed in current branch compared to base.

    Returns:
        List of (file_path, change_type) tuples
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-status", f"{base_branch}...HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            # Fallback: compare to HEAD~1
            result = subprocess.run(
                ["git", "diff", "--name-status", "HEAD~1"],
                capture_output=True,
                text=True,
                check=False,
            )

        changes = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                status = parts[0]
                file_path = parts[-1]
                change_type = {
                    "A": "added",
                    "M": "modified",
                    "D": "deleted",
                    "R": "renamed",
                }.get(status[0], "modified")
                changes.append((file_path, change_type))

        return changes
    except Exception:
        return []


def is_protocol_file(file_path: str) -> bool:
    """Check if file is a protocol-related file."""
    for protocol_path in PROTOCOL_PATHS:
        if file_path.startswith(protocol_path) or file_path == protocol_path:
            return True
    return False


def get_file_content_patterns(file_path: str) -> List[str]:
    """
    Check if file content matches protocol patterns.

    Returns:
        List of matched patterns
    """
    import re

    matches = []

    try:
        # Get diff content
        result = subprocess.run(
            ["git", "diff", "origin/main...HEAD", "--", file_path],
            capture_output=True,
            text=True,
            check=False,
        )

        diff_content = result.stdout

        for pattern in PROTOCOL_PATTERNS:
            if re.search(pattern, diff_content):
                matches.append(pattern)

    except Exception:
        pass

    return matches


def detect_protocol_changes() -> List[ProtocolChange]:
    """
    Detect protocol changes in the current branch.

    Returns:
        List of detected protocol changes
    """
    changes = []
    file_changes = get_changed_files()

    for file_path, change_type in file_changes:
        is_protocol = is_protocol_file(file_path)
        patterns = get_file_content_patterns(file_path) if not is_protocol else []

        if is_protocol or patterns:
            changes.append(ProtocolChange(
                file_path=file_path,
                change_type=change_type,
                patterns_matched=patterns if patterns else ["protocol_path"],
            ))

    return changes


# ============================================================================
# Journal Functions
# ============================================================================

def load_journal() -> List[JournalEntry]:
    """Load protocol change journal."""
    if not JOURNAL_PATH.exists():
        return []

    try:
        with open(JOURNAL_PATH, "r") as f:
            data = json.load(f)
            return [JournalEntry.from_dict(entry) for entry in data]
    except Exception:
        return []


def get_current_commit() -> Optional[str]:
    """Get current HEAD commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def find_approval_for_commit(
    journal: List[JournalEntry],
    commit_hash: str,
) -> Optional[JournalEntry]:
    """Find approval entry for a commit."""
    for entry in journal:
        if entry.commit_hash == commit_hash:
            return entry
    return None


def find_recent_approval(
    journal: List[JournalEntry],
    max_age_hours: int = 48,
) -> Optional[JournalEntry]:
    """Find recent approval that might apply to this PR."""
    from datetime import timedelta, timezone

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max_age_hours)

    recent = [
        entry for entry in journal
        if entry.timestamp.replace(tzinfo=timezone.utc) > cutoff
        and entry.approval_status in ("APPROVED", "APPROVED_WITH_CONDITIONS")
    ]

    if recent:
        # Return most recent
        return max(recent, key=lambda e: e.timestamp)

    return None


# ============================================================================
# Check Function
# ============================================================================

def check_protocol_changes() -> CheckResult:
    """
    Check if protocol changes have security review approval.

    Returns:
        CheckResult with pass/fail status and details
    """
    # Detect protocol changes
    protocol_changes = detect_protocol_changes()

    if not protocol_changes:
        return CheckResult(
            passed=True,
            message="No protocol changes detected",
            protocol_changes=[],
            approval_found=None,
        )

    # Protocol changes detected - check for journal entry
    journal = load_journal()
    current_commit = get_current_commit()

    # First: look for approval matching current commit
    if current_commit:
        approval = find_approval_for_commit(journal, current_commit)
        if approval:
            if approval.approval_status == "APPROVED":
                return CheckResult(
                    passed=True,
                    message=f"Protocol change approved: {approval.change_id}",
                    protocol_changes=protocol_changes,
                    approval_found=approval,
                )
            elif approval.approval_status == "APPROVED_WITH_CONDITIONS":
                if approval.conditions_met:
                    return CheckResult(
                        passed=True,
                        message=f"Protocol change approved with conditions (met): {approval.change_id}",
                        protocol_changes=protocol_changes,
                        approval_found=approval,
                    )
                else:
                    return CheckResult(
                        passed=False,
                        message=(
                            f"Protocol change {approval.change_id} has unmet conditions. "
                            "All conditions must be satisfied before merge."
                        ),
                        protocol_changes=protocol_changes,
                        approval_found=approval,
                    )
            elif approval.approval_status == "REJECTED":
                return CheckResult(
                    passed=False,
                    message=f"Protocol change {approval.change_id} was rejected.",
                    protocol_changes=protocol_changes,
                    approval_found=approval,
                )

    # Second: check for recent approval that might apply
    recent_approval = find_recent_approval(journal)
    if recent_approval:
        return CheckResult(
            passed=True,
            message=(
                f"Recent protocol change approval found: {recent_approval.change_id}. "
                "Note: Update journal with commit hash after merge."
            ),
            protocol_changes=protocol_changes,
            approval_found=recent_approval,
        )

    # No approval found
    changed_files = "\n".join(f"  - {c.file_path} ({c.change_type})" for c in protocol_changes)

    return CheckResult(
        passed=False,
        message=(
            "Protocol change detected but no security review approval found.\n\n"
            f"Files changed:\n{changed_files}\n\n"
            "Required actions:\n"
            "  1. Complete security review checklist (docs/compliance/PROTOCOL_CHANGE_REVIEW.md)\n"
            "  2. Request review from security team (@security-team)\n"
            "  3. Add journal entry with approval to docs/compliance/protocol_change_journal.json\n"
            "  4. Re-run CI\n"
        ),
        protocol_changes=protocol_changes,
        approval_found=None,
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> int:
    """Main entry point for CI."""
    print("=" * 60)
    print("PROTOCOL CHANGE REVIEW CHECK")
    print("=" * 60)

    result = check_protocol_changes()

    if result.passed:
        print(f"\n[PASS] {result.message}")
        if result.protocol_changes:
            print("\nProtocol changes detected (approved):")
            for change in result.protocol_changes:
                print(f"  - {change.file_path}")
        return 0
    else:
        print(f"\n[FAIL] {result.message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
