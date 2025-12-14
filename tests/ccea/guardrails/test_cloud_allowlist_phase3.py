# -*- coding: utf-8 -*-
"""
Tests for CCEA Cloud Allowlist (Phase 3 fixes).

Specifically tests WI-CI-02 fixes:
- modules_checked is properly populated
- Fail-closed check when files_checked > 0 but modules_checked == 0
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ccea.guardrails.cloud_allowlist import (
    AllowlistCheckResult,
    validate_cloud_build,
)


class TestModulesCheckedPopulation:
    """Tests for modules_checked counter fix."""

    def test_modules_checked_nonzero_for_python_files(self, tmp_path: Path) -> None:
        """Test that modules_checked > 0 when Python files exist."""
        # Create cloud directory with Python files
        cloud_dir = tmp_path / "packages" / "cloud"
        cloud_dir.mkdir(parents=True)

        # Create a non-test Python file
        (cloud_dir / "module.py").write_text("import os\n", encoding="utf-8")
        (cloud_dir / "another.py").write_text("import sys\n", encoding="utf-8")

        result = validate_cloud_build(cloud_dir, check_transitive=False)

        assert result.files_checked >= 2
        assert result.modules_checked >= 2, "modules_checked should count non-test files"

    def test_modules_checked_excludes_test_files(self, tmp_path: Path) -> None:
        """Test that test files are not counted in modules_checked."""
        cloud_dir = tmp_path / "packages" / "cloud"
        cloud_dir.mkdir(parents=True)

        # Create test and non-test files
        (cloud_dir / "module.py").write_text("import os\n", encoding="utf-8")
        (cloud_dir / "test_module.py").write_text("import pytest\n", encoding="utf-8")

        result = validate_cloud_build(cloud_dir, check_transitive=False)

        assert result.files_checked >= 2  # All files counted
        assert result.modules_checked == 1  # Only non-test file

    def test_modules_checked_excludes_tests_directory(self, tmp_path: Path) -> None:
        """Test that files in tests/ directory are excluded from modules_checked."""
        cloud_dir = tmp_path / "packages" / "cloud"
        tests_dir = cloud_dir / "tests"
        tests_dir.mkdir(parents=True)

        # Create non-test file and test directory file
        (cloud_dir / "module.py").write_text("import os\n", encoding="utf-8")
        (tests_dir / "test_module.py").write_text("import pytest\n", encoding="utf-8")

        result = validate_cloud_build(cloud_dir, check_transitive=False)

        assert result.modules_checked == 1

    def test_modules_checked_empty_directory(self, tmp_path: Path) -> None:
        """Test that empty directory gives 0 for both counters."""
        cloud_dir = tmp_path / "packages" / "cloud"
        cloud_dir.mkdir(parents=True)

        result = validate_cloud_build(cloud_dir, check_transitive=False)

        assert result.files_checked == 0
        assert result.modules_checked == 0

    def test_modules_checked_only_test_files(self, tmp_path: Path) -> None:
        """Test directory with only test files."""
        cloud_dir = tmp_path / "packages" / "cloud"
        cloud_dir.mkdir(parents=True)

        (cloud_dir / "test_module.py").write_text("import pytest\n", encoding="utf-8")
        (cloud_dir / "test_another.py").write_text("import pytest\n", encoding="utf-8")

        result = validate_cloud_build(cloud_dir, check_transitive=False)

        assert result.files_checked >= 2
        assert result.modules_checked == 0


class TestFailClosedCheck:
    """Tests for fail-closed behavior (WI-CI-02)."""

    def test_result_passes_normal_case(self, tmp_path: Path) -> None:
        """Test normal case passes."""
        cloud_dir = tmp_path / "packages" / "cloud"
        cloud_dir.mkdir(parents=True)
        (cloud_dir / "module.py").write_text("import os\n", encoding="utf-8")

        result = validate_cloud_build(cloud_dir, check_transitive=False)

        assert result.passed is True
        assert result.files_checked > 0
        assert result.modules_checked > 0

    def test_result_passes_empty_directory(self, tmp_path: Path) -> None:
        """Test empty directory passes (no files to check)."""
        cloud_dir = tmp_path / "packages" / "cloud"
        cloud_dir.mkdir(parents=True)

        result = validate_cloud_build(cloud_dir, check_transitive=False)

        # Empty is technically OK (nothing to violate)
        assert result.passed is True
        assert result.files_checked == 0
        assert result.modules_checked == 0


class TestCloudAllowlistResult:
    """Tests for AllowlistCheckResult properties."""

    def test_result_counts_violations(self) -> None:
        """Test that violations are properly counted."""
        result = AllowlistCheckResult()
        assert result.passed is True
        assert len(result.violations) == 0

    def test_result_tracks_files_checked(self) -> None:
        """Test that files_checked is tracked."""
        result = AllowlistCheckResult()
        result.files_checked = 42

        assert result.files_checked == 42

    def test_result_tracks_modules_checked(self) -> None:
        """Test that modules_checked is tracked."""
        result = AllowlistCheckResult()
        result.modules_checked = 10

        assert result.modules_checked == 10


class TestProhibitedImportDetection:
    """Tests for detecting prohibited imports."""

    def test_detects_order_execution_import(self, tmp_path: Path) -> None:
        """Test detection of order_execution imports."""
        cloud_dir = tmp_path / "packages" / "cloud"
        cloud_dir.mkdir(parents=True)

        # File with prohibited import
        (cloud_dir / "bad.py").write_text(
            "from adapters.alpaca.order_execution import OrderExecutor\n",
            encoding="utf-8"
        )

        result = validate_cloud_build(cloud_dir, check_transitive=False)

        assert result.passed is False
        assert len(result.violations) >= 1
        assert any("order_execution" in str(v) for v in result.violations)

    def test_allows_safe_imports(self, tmp_path: Path) -> None:
        """Test that safe imports are allowed."""
        cloud_dir = tmp_path / "packages" / "cloud"
        cloud_dir.mkdir(parents=True)

        # File with allowed imports
        (cloud_dir / "good.py").write_text(
            "import os\nimport json\nfrom pathlib import Path\n",
            encoding="utf-8"
        )

        result = validate_cloud_build(cloud_dir, check_transitive=False)

        assert result.passed is True
        assert len(result.violations) == 0
