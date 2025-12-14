# -*- coding: utf-8 -*-
"""
Tests for CCEA Design Doc SHA Verification (WI-TRACE-01).

Ensures that Design Doc changes are tracked and SHA256 verification works correctly.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

from ccea.guardrails.design_doc_check import (
    compute_sha256,
    extract_recorded_sha256,
    verify_design_doc_sha,
)


class TestComputeSha256:
    """Tests for SHA256 computation."""

    def test_compute_sha256_basic(self, tmp_path: Path) -> None:
        """Test SHA256 computation for a basic file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!", encoding="utf-8")

        sha256 = compute_sha256(test_file)

        # Verify against expected hash
        expected = hashlib.sha256(b"Hello, World!").hexdigest()
        assert sha256 == expected

    def test_compute_sha256_empty_file(self, tmp_path: Path) -> None:
        """Test SHA256 computation for empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("", encoding="utf-8")

        sha256 = compute_sha256(test_file)

        expected = hashlib.sha256(b"").hexdigest()
        assert sha256 == expected

    def test_compute_sha256_binary_content(self, tmp_path: Path) -> None:
        """Test SHA256 computation for binary content."""
        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03")

        sha256 = compute_sha256(test_file)

        expected = hashlib.sha256(b"\x00\x01\x02\x03").hexdigest()
        assert sha256 == expected

    def test_compute_sha256_multiline(self, tmp_path: Path) -> None:
        """Test SHA256 computation for multiline content."""
        content = "Line 1\nLine 2\nLine 3\n"
        test_file = tmp_path / "multiline.txt"
        test_file.write_text(content, encoding="utf-8")

        sha256 = compute_sha256(test_file)

        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert sha256 == expected


class TestExtractRecordedSha256:
    """Tests for extracting SHA256 from markdown documents."""

    def test_extract_sha256_standard_format(self, tmp_path: Path) -> None:
        """Test extraction of SHA256 from standard format."""
        md_content = """> **Version**: 1.0.0
> **Date**: 2025-12-15
> **Status**: APPROVED
> **SHA256**: abcd1234567890abcdef1234567890abcdef1234567890abcdef1234567890ab

## Content here
"""
        test_file = tmp_path / "doc.md"
        test_file.write_text(md_content, encoding="utf-8")

        sha256 = extract_recorded_sha256(test_file)

        assert sha256 == "abcd1234567890abcdef1234567890abcdef1234567890abcdef1234567890ab"

    def test_extract_sha256_not_found(self, tmp_path: Path) -> None:
        """Test extraction when SHA256 is not present."""
        md_content = """> **Version**: 1.0.0
> **Date**: 2025-12-15

## Content here
"""
        test_file = tmp_path / "doc.md"
        test_file.write_text(md_content, encoding="utf-8")

        sha256 = extract_recorded_sha256(test_file)

        assert sha256 is None

    def test_extract_sha256_file_not_found(self, tmp_path: Path) -> None:
        """Test extraction when file doesn't exist."""
        test_file = tmp_path / "nonexistent.md"

        sha256 = extract_recorded_sha256(test_file)

        assert sha256 is None

    def test_extract_sha256_case_insensitive(self, tmp_path: Path) -> None:
        """Test SHA256 is normalized to lowercase."""
        md_content = """> **SHA256**: ABCD1234567890ABCDEF1234567890ABCDEF1234567890ABCDEF1234567890AB
"""
        test_file = tmp_path / "doc.md"
        test_file.write_text(md_content, encoding="utf-8")

        sha256 = extract_recorded_sha256(test_file)

        assert sha256 == "abcd1234567890abcdef1234567890abcdef1234567890abcdef1234567890ab"


class TestVerifyDesignDocSha:
    """Tests for full SHA verification workflow."""

    def test_verify_sha_match(self, tmp_path: Path) -> None:
        """Test verification when SHA matches."""
        # Create snapshot
        snapshot = tmp_path / "snapshot.txt"
        snapshot.write_text("Test content", encoding="utf-8")

        # Compute actual SHA
        actual_sha = compute_sha256(snapshot)

        # Create rendered doc with matching SHA
        rendered = tmp_path / "rendered.md"
        rendered.write_text(f"> **SHA256**: {actual_sha}\n\n# Content", encoding="utf-8")

        passed, message = verify_design_doc_sha(snapshot, rendered)

        assert passed is True
        assert actual_sha in message

    def test_verify_sha_mismatch(self, tmp_path: Path) -> None:
        """Test verification when SHA doesn't match."""
        # Create snapshot
        snapshot = tmp_path / "snapshot.txt"
        snapshot.write_text("Test content", encoding="utf-8")

        # Create rendered doc with wrong SHA
        wrong_sha = "0" * 64
        rendered = tmp_path / "rendered.md"
        rendered.write_text(f"> **SHA256**: {wrong_sha}\n\n# Content", encoding="utf-8")

        passed, message = verify_design_doc_sha(snapshot, rendered)

        assert passed is False
        assert "mismatch" in message.lower()

    def test_verify_snapshot_not_found(self, tmp_path: Path) -> None:
        """Test verification when snapshot is missing."""
        snapshot = tmp_path / "nonexistent.txt"
        rendered = tmp_path / "rendered.md"
        rendered.write_text("> **SHA256**: abc", encoding="utf-8")

        passed, message = verify_design_doc_sha(snapshot, rendered)

        assert passed is False
        assert "not found" in message.lower()

    def test_verify_rendered_not_found(self, tmp_path: Path) -> None:
        """Test verification when rendered doc is missing."""
        snapshot = tmp_path / "snapshot.txt"
        snapshot.write_text("Content", encoding="utf-8")
        rendered = tmp_path / "nonexistent.md"

        passed, message = verify_design_doc_sha(snapshot, rendered)

        assert passed is False
        assert "not found" in message.lower()

    def test_verify_no_sha_in_rendered(self, tmp_path: Path) -> None:
        """Test verification when rendered doc has no SHA256."""
        snapshot = tmp_path / "snapshot.txt"
        snapshot.write_text("Content", encoding="utf-8")
        rendered = tmp_path / "rendered.md"
        rendered.write_text("# No SHA here", encoding="utf-8")

        passed, message = verify_design_doc_sha(snapshot, rendered)

        assert passed is False
        assert "no sha256 found" in message.lower()


class TestRealDesignDoc:
    """Integration tests against real Design Doc files."""

    def test_real_design_doc_sha_matches(self) -> None:
        """Test that real Design Doc snapshot matches recorded SHA."""
        snapshot_path = Path("docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt")
        rendered_path = Path("docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.md")

        if not snapshot_path.exists() or not rendered_path.exists():
            pytest.skip("Real Design Doc files not available")

        passed, message = verify_design_doc_sha(snapshot_path, rendered_path)

        assert passed, f"Design Doc SHA verification failed: {message}"
