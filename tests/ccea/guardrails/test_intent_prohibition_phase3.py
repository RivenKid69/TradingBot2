# -*- coding: utf-8 -*-
"""
Tests for CCEA Intent Prohibition (Phase 3 fixes).

Specifically tests WI-CI-02 fix:
- CLI output is ASCII-safe (no Unicode symbols like checkmark/cross)
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from ccea.guardrails.intent_prohibition import (
    IntentProhibitionResult,
    main as intent_main,
    check_cloud_package_for_intents,
    check_python_source_for_intent_injection,
)


class TestAsciiSafeOutput:
    """Tests for ASCII-safe CLI output (WI-CI-02)."""

    def test_pass_message_is_ascii(self, tmp_path: Path) -> None:
        """Test that PASS message uses ASCII characters."""
        # Create valid cloud directory
        cloud_dir = tmp_path / "packages" / "cloud"
        cloud_dir.mkdir(parents=True)
        (cloud_dir / "safe.py").write_text("import os\n", encoding="utf-8")

        # Capture stdout
        captured = io.StringIO()
        with patch.object(sys, 'stdout', captured):
            with patch.object(sys, 'argv', ['intent_prohibition', '--cloud-path', str(cloud_dir)]):
                try:
                    exit_code = intent_main()
                except SystemExit as e:
                    exit_code = e.code

        output = captured.getvalue()

        # Verify ASCII-safe
        assert output.isascii(), f"Output contains non-ASCII: {output!r}"
        assert "[PASS]" in output or "PASSED" in output

    def test_fail_message_is_ascii(self, tmp_path: Path) -> None:
        """Test that FAIL message uses ASCII characters."""
        # Create cloud directory with violation
        cloud_dir = tmp_path / "packages" / "cloud"
        cloud_dir.mkdir(parents=True)
        # File with intent field
        (cloud_dir / "bad.py").write_text(
            'signal = {"side": "BUY"}\n',
            encoding="utf-8"
        )

        # Capture stdout
        captured = io.StringIO()
        with patch.object(sys, 'stdout', captured):
            with patch.object(sys, 'argv', ['intent_prohibition', '--cloud-path', str(cloud_dir)]):
                try:
                    exit_code = intent_main()
                except SystemExit as e:
                    exit_code = e.code

        output = captured.getvalue()

        # Verify ASCII-safe
        assert output.isascii(), f"Output contains non-ASCII: {output!r}"
        assert "[FAIL]" in output or "FAILED" in output

    def test_no_unicode_checkmark(self, tmp_path: Path) -> None:
        """Test that Unicode checkmark is not used."""
        cloud_dir = tmp_path / "packages" / "cloud"
        cloud_dir.mkdir(parents=True)
        (cloud_dir / "safe.py").write_text("x = 1\n", encoding="utf-8")

        captured = io.StringIO()
        with patch.object(sys, 'stdout', captured):
            with patch.object(sys, 'argv', ['intent_prohibition', '--cloud-path', str(cloud_dir)]):
                try:
                    intent_main()
                except SystemExit:
                    pass

        output = captured.getvalue()
        assert "\u2713" not in output  # checkmark
        assert "\u2714" not in output  # heavy checkmark

    def test_no_unicode_cross(self, tmp_path: Path) -> None:
        """Test that Unicode cross is not used."""
        cloud_dir = tmp_path / "packages" / "cloud"
        cloud_dir.mkdir(parents=True)
        (cloud_dir / "bad.py").write_text('intent = "test"\n', encoding="utf-8")

        captured = io.StringIO()
        with patch.object(sys, 'stdout', captured):
            with patch.object(sys, 'argv', ['intent_prohibition', '--cloud-path', str(cloud_dir)]):
                try:
                    intent_main()
                except SystemExit:
                    pass

        output = captured.getvalue()
        assert "\u2717" not in output  # ballot x
        assert "\u2718" not in output  # heavy ballot x


class TestIntentProhibition:
    """General tests for intent prohibition functionality."""

    def test_detects_target_position_pattern(self, tmp_path: Path) -> None:
        """Test detection of target_position pattern in code (prohibited intent)."""
        cloud_dir = tmp_path / "packages" / "cloud"
        cloud_dir.mkdir(parents=True)
        # Create code with prohibited order-like JSON pattern
        (cloud_dir / "service.py").write_text(
            'data = {"target_position": 100, "symbol": "AAPL"}\n',
            encoding="utf-8"
        )

        result = check_cloud_package_for_intents(cloud_dir)

        # This should be detected as a prohibited pattern
        assert result.passed is False
        assert len(result.violations) >= 1

    def test_detects_side_pattern(self, tmp_path: Path) -> None:
        """Test detection of side field in JSON-like payload."""
        cloud_dir = tmp_path / "packages" / "cloud"
        cloud_dir.mkdir(parents=True)
        (cloud_dir / "service.py").write_text(
            'order = {"side": "BUY", "qty": 100}\n',
            encoding="utf-8"
        )

        result = check_cloud_package_for_intents(cloud_dir)

        assert result.passed is False

    def test_allows_safe_code(self, tmp_path: Path) -> None:
        """Test that safe code passes."""
        cloud_dir = tmp_path / "packages" / "cloud"
        cloud_dir.mkdir(parents=True)
        (cloud_dir / "service.py").write_text(
            'def analyze(data):\n    return data.mean()\n',
            encoding="utf-8"
        )

        result = check_cloud_package_for_intents(cloud_dir)

        assert result.passed is True

    def test_allows_aggregated_metrics(self, tmp_path: Path) -> None:
        """Test that aggregated metrics pass (not order-like)."""
        cloud_dir = tmp_path / "packages" / "cloud"
        cloud_dir.mkdir(parents=True)
        (cloud_dir / "metrics.py").write_text(
            'stats = {"total_return": 0.15, "sharpe_ratio": 1.2}\n',
            encoding="utf-8"
        )

        result = check_cloud_package_for_intents(cloud_dir)

        assert result.passed is True


class TestIntentProhibitionResult:
    """Tests for IntentProhibitionResult data class."""

    def test_result_starts_passed(self) -> None:
        """Test that result starts as passed."""
        result = IntentProhibitionResult()
        assert result.passed is True

    def test_result_tracks_violations(self) -> None:
        """Test that violations are tracked."""
        result = IntentProhibitionResult()
        assert len(result.violations) == 0
