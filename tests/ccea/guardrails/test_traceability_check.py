# -*- coding: utf-8 -*-
"""
Tests for CCEA Traceability Matrix Check (WI-TRACE-02).

Ensures that all DONE statuses have corresponding Artifact/Check references.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ccea.guardrails.traceability_check import (
    TraceabilityCheckResult,
    TraceabilityViolation,
    check_done_has_evidence,
    parse_markdown_table_row,
    validate_traceability_matrix,
)


class TestParseMarkdownTableRow:
    """Tests for markdown table row parsing."""

    def test_parse_valid_row(self) -> None:
        """Test parsing a valid table row."""
        line = "| Cell1 | Cell2 | Cell3 |"
        cells = parse_markdown_table_row(line)

        assert cells == ["Cell1", "Cell2", "Cell3"]

    def test_parse_row_with_whitespace(self) -> None:
        """Test parsing row with extra whitespace."""
        line = "|  Cell1  |  Cell2  |  Cell3  |"
        cells = parse_markdown_table_row(line)

        assert cells == ["Cell1", "Cell2", "Cell3"]

    def test_parse_separator_row(self) -> None:
        """Test that separator rows are skipped."""
        line = "|-------|-------|-------|"
        cells = parse_markdown_table_row(line)

        assert cells is None

    def test_parse_separator_with_colons(self) -> None:
        """Test separator row with alignment markers."""
        line = "|:------|:------:|------:|"
        cells = parse_markdown_table_row(line)

        assert cells is None

    def test_parse_non_table_line(self) -> None:
        """Test non-table lines return None."""
        line = "This is not a table"
        cells = parse_markdown_table_row(line)

        assert cells is None

    def test_parse_empty_cells(self) -> None:
        """Test row with empty cells."""
        line = "| Cell1 | | Cell3 |"
        cells = parse_markdown_table_row(line)

        assert cells == ["Cell1", "", "Cell3"]


class TestCheckDoneHasEvidence:
    """Tests for DONE status evidence checking."""

    def test_done_with_artifact_evidence(self) -> None:
        """Test DONE with artifact reference passes."""
        headers = ["Requirement", "Status", "Artifact"]
        cells = ["REQ-001", "DONE", "docs/schema.json"]

        reason = check_done_has_evidence(cells, headers)

        assert reason is None  # No violation

    def test_done_with_test_evidence(self) -> None:
        """Test DONE with test reference passes."""
        headers = ["Requirement", "Status", "Artifact"]
        cells = ["REQ-001", "DONE", "test_schema_check.py"]

        reason = check_done_has_evidence(cells, headers)

        assert reason is None  # No violation

    def test_done_without_evidence(self) -> None:
        """Test DONE without evidence fails."""
        headers = ["Requirement", "Status", "Artifact"]
        cells = ["REQ-001", "DONE", "-"]

        reason = check_done_has_evidence(cells, headers)

        assert reason is not None
        assert "evidence" in reason.lower()

    def test_planned_no_evidence_needed(self) -> None:
        """Test PLANNED status doesn't need evidence."""
        headers = ["Requirement", "Status", "Artifact"]
        cells = ["REQ-001", "PLANNED", "-"]

        reason = check_done_has_evidence(cells, headers)

        assert reason is None  # No violation for PLANNED

    def test_done_with_ccea_path(self) -> None:
        """Test DONE with ccea/ module path passes."""
        headers = ["Requirement", "Status", "Artifact"]
        cells = ["REQ-001", "DONE", "ccea/guardrails/check.py"]

        reason = check_done_has_evidence(cells, headers)

        assert reason is None

    def test_done_with_docs_path(self) -> None:
        """Test DONE with docs/ path passes."""
        headers = ["Requirement", "Status", "Artifact"]
        cells = ["REQ-001", "DONE", "docs/design/CCEA.md"]

        reason = check_done_has_evidence(cells, headers)

        assert reason is None

    def test_done_na_is_valid(self) -> None:
        """Test DONE with N/A still requires evidence in other columns."""
        headers = ["Requirement", "Status", "Artifact", "Test"]
        cells = ["REQ-001", "DONE", "N/A", "test_check.py"]

        reason = check_done_has_evidence(cells, headers)

        assert reason is None


class TestValidateTraceabilityMatrix:
    """Tests for full matrix validation."""

    def test_validate_valid_matrix(self, tmp_path: Path) -> None:
        """Test validation of a valid matrix."""
        matrix_content = """# Traceability Matrix

| Requirement | Status | Artifact |
|-------------|--------|----------|
| REQ-001 | DONE | docs/schema.json |
| REQ-002 | DONE | test_check.py |
| REQ-003 | PLANNED | - |
"""
        matrix_file = tmp_path / "matrix.md"
        matrix_file.write_text(matrix_content, encoding="utf-8")

        result = validate_traceability_matrix(matrix_file)

        assert result.passed is True
        assert len(result.violations) == 0
        assert result.done_count == 2
        assert result.planned_count == 1

    def test_validate_matrix_with_violations(self, tmp_path: Path) -> None:
        """Test validation catches DONE without evidence."""
        matrix_content = """# Traceability Matrix

| Requirement | Status | Artifact |
|-------------|--------|----------|
| REQ-001 | DONE | - |
| REQ-002 | DONE | N/A |
"""
        matrix_file = tmp_path / "matrix.md"
        matrix_file.write_text(matrix_content, encoding="utf-8")

        result = validate_traceability_matrix(matrix_file)

        assert result.passed is False
        assert len(result.violations) == 2

    def test_validate_matrix_not_found(self, tmp_path: Path) -> None:
        """Test validation when matrix file is missing."""
        matrix_file = tmp_path / "nonexistent.md"

        result = validate_traceability_matrix(matrix_file)

        assert result.passed is False
        assert len(result.violations) == 1
        assert "not found" in str(result.violations[0]).lower()

    def test_validate_multiple_tables(self, tmp_path: Path) -> None:
        """Test validation handles multiple tables."""
        matrix_content = """# Matrix

## Table 1
| Requirement | Status | Artifact |
|-------------|--------|----------|
| REQ-001 | DONE | docs/schema.json |

## Table 2
| Item | Status | Test |
|------|--------|------|
| ITM-001 | DONE | test_item.py |
"""
        matrix_file = tmp_path / "matrix.md"
        matrix_file.write_text(matrix_content, encoding="utf-8")

        result = validate_traceability_matrix(matrix_file)

        assert result.passed is True
        assert result.rows_checked >= 2


class TestRealTraceabilityMatrix:
    """Integration tests against real traceability matrix."""

    def test_real_matrix_valid(self) -> None:
        """Test that real traceability matrix passes validation."""
        matrix_path = Path("docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md")

        if not matrix_path.exists():
            pytest.skip("Real traceability matrix not available")

        result = validate_traceability_matrix(matrix_path)

        # Print violations for debugging if any
        if not result.passed:
            for v in result.violations:
                print(f"Violation: {v}")

        assert result.passed, f"Traceability matrix has {len(result.violations)} violations"


class TestTraceabilityViolation:
    """Tests for TraceabilityViolation data class."""

    def test_violation_str_format(self) -> None:
        """Test violation string representation."""
        violation = TraceabilityViolation(
            line_number=42,
            row_content="| REQ-001 | DONE | - |",
            requirement_id="REQ-001",
            status="DONE",
            reason="missing evidence",
        )

        result = str(violation)

        assert "42" in result
        assert "REQ-001" in result
        assert "DONE" in result
        assert "evidence" in result.lower()
