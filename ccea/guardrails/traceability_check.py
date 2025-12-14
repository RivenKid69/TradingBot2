# -*- coding: utf-8 -*-
"""
CCEA Traceability Matrix Check.

Validates that all DONE statuses in the traceability matrix have
corresponding Artifact/Check references.

This is a critical guardrail to ensure traceability is truthful (WI-TRACE-02).

Usage:
    python -m ccea.guardrails.traceability_check
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ============================================================================
# Constants
# ============================================================================

DEFAULT_MATRIX_PATH = Path("docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md")

# Statuses that require evidence
STATUSES_REQUIRING_EVIDENCE = {"DONE"}

# Statuses that don't require evidence
STATUSES_WITHOUT_EVIDENCE = {"PLANNED", "IN_PROGRESS", "N/A"}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TraceabilityViolation:
    """Represents a traceability violation."""
    line_number: int
    row_content: str
    requirement_id: str
    status: str
    reason: str

    def __str__(self) -> str:
        return (
            f"Line {self.line_number}: {self.requirement_id} is {self.status} "
            f"but {self.reason}"
        )


@dataclass
class TraceabilityCheckResult:
    """Result of traceability check."""
    violations: List[TraceabilityViolation] = field(default_factory=list)
    rows_checked: int = 0
    done_count: int = 0
    planned_count: int = 0
    passed: bool = True

    def add_violation(self, violation: TraceabilityViolation) -> None:
        self.violations.append(violation)
        self.passed = False


# ============================================================================
# Traceability Parsing and Validation
# ============================================================================

def parse_markdown_table_row(line: str) -> Optional[List[str]]:
    """
    Parse a markdown table row into cells.

    Args:
        line: Markdown table line

    Returns:
        List of cell values or None if not a table row
    """
    line = line.strip()
    if not line.startswith("|") or not line.endswith("|"):
        return None

    # Skip separator rows (e.g., |---|---|---|)
    # Check if all cells contain only dashes, colons, and spaces
    cells_content = line[1:-1]  # Remove leading/trailing |
    if re.match(r'^[\s\-:|]+$', cells_content):
        return None

    # Split by | and strip whitespace
    cells = [cell.strip() for cell in line.split("|")]
    # Remove empty first and last elements (from leading/trailing |)
    cells = cells[1:-1]

    return cells if cells else None


def check_done_has_evidence(cells: List[str], table_headers: List[str]) -> Optional[str]:
    """
    Check if a DONE status row has evidence.

    Args:
        cells: Table row cells
        table_headers: Column headers

    Returns:
        Violation reason or None if valid
    """
    # Find status column (usually "Status")
    status_col = None
    artifact_col = None
    check_col = None

    for i, header in enumerate(table_headers):
        header_lower = header.lower()
        if "status" in header_lower:
            status_col = i
        elif "artifact" in header_lower or "check" in header_lower:
            if artifact_col is None:
                artifact_col = i
            else:
                check_col = i

    if status_col is None or status_col >= len(cells):
        return None  # Can't determine status

    status = cells[status_col].upper().strip()

    if status not in STATUSES_REQUIRING_EVIDENCE:
        return None  # Not DONE, no check needed

    # For DONE status, we need artifact or check evidence
    has_artifact = False
    has_check = False

    # Check if Artifact/Check column has content
    if artifact_col is not None and artifact_col < len(cells):
        artifact_val = cells[artifact_col].strip()
        if artifact_val and artifact_val != "-" and artifact_val != "N/A":
            has_artifact = True

    if check_col is not None and check_col < len(cells):
        check_val = cells[check_col].strip()
        if check_val and check_val != "-" and check_val != "N/A":
            has_check = True

    # Also check for inline evidence in any column (file references, test names, etc.)
    evidence_patterns = [
        r'\.py\b',      # Python files
        r'\.json\b',    # JSON files
        r'\.md\b',      # Markdown files
        r'\.yml\b',     # YAML files (CI workflows)
        r'\.yaml\b',    # YAML files
        r'\.toml\b',    # TOML config files
        r'\.ini\b',     # INI config files
        r'test_\w+',    # Test names
        r'docs/',       # Documentation paths
        r'ccea/',       # CCEA module paths
    ]

    for cell in cells:
        for pattern in evidence_patterns:
            if re.search(pattern, cell, re.IGNORECASE):
                has_artifact = True
                break

    if not has_artifact and not has_check:
        return "missing Artifact/Check evidence"

    return None


def validate_traceability_matrix(matrix_path: Path) -> TraceabilityCheckResult:
    """
    Validate traceability matrix for DONE entries without evidence.

    Args:
        matrix_path: Path to traceability matrix markdown file

    Returns:
        TraceabilityCheckResult with any violations
    """
    result = TraceabilityCheckResult()

    if not matrix_path.exists():
        result.add_violation(TraceabilityViolation(
            line_number=0,
            row_content="",
            requirement_id="N/A",
            status="N/A",
            reason=f"Matrix file not found: {matrix_path}",
        ))
        return result

    with open(matrix_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_headers: List[str] = []
    in_table = False

    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Detect table start (header row)
        if stripped.startswith("|") and not in_table:
            cells = parse_markdown_table_row(stripped)
            if cells:
                current_headers = cells
                in_table = True
                continue

        # Skip separator rows
        if re.match(r'^\|[\s\-:]+\|$', stripped):
            continue

        # Parse table data row
        if in_table and stripped.startswith("|"):
            cells = parse_markdown_table_row(stripped)
            if cells:
                result.rows_checked += 1

                # Extract requirement ID (usually first column)
                req_id = cells[0] if cells else "Unknown"

                # Check for status column
                for i, header in enumerate(current_headers):
                    if "status" in header.lower() and i < len(cells):
                        status = cells[i].upper().strip()
                        if status == "DONE":
                            result.done_count += 1
                        elif status == "PLANNED":
                            result.planned_count += 1
                        break

                # Check DONE entries for evidence
                violation_reason = check_done_has_evidence(cells, current_headers)
                if violation_reason:
                    status = "DONE"
                    for i, header in enumerate(current_headers):
                        if "status" in header.lower() and i < len(cells):
                            status = cells[i].strip()
                            break

                    result.add_violation(TraceabilityViolation(
                        line_number=line_num,
                        row_content=stripped[:100] + "..." if len(stripped) > 100 else stripped,
                        requirement_id=req_id,
                        status=status,
                        reason=violation_reason,
                    ))
        elif not stripped.startswith("|") and in_table:
            # End of table
            in_table = False
            current_headers = []

    return result


# ============================================================================
# CLI Interface
# ============================================================================

def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CCEA Traceability Matrix Check"
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=DEFAULT_MATRIX_PATH,
        help="Path to traceability matrix markdown file",
    )
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        default=True,
        help="Exit with error code on violations",
    )

    args = parser.parse_args()

    print(f"Checking traceability matrix: {args.matrix}")

    result = validate_traceability_matrix(args.matrix)

    print(f"\nRows checked: {result.rows_checked}")
    print(f"DONE entries: {result.done_count}")
    print(f"PLANNED entries: {result.planned_count}")

    if result.violations:
        print(f"\nViolations ({len(result.violations)}):")
        for v in result.violations:
            print(f"  {v}")

    if result.passed:
        print("\n[PASS] Traceability matrix check PASSED")
        return 0
    else:
        print("\n[FAIL] Traceability matrix check FAILED")
        print("\nTo fix: Add Artifact/Check references for each DONE item,")
        print("or change status to PLANNED if not yet implemented.")
        if args.fail_on_violation:
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
