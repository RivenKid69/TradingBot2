# -*- coding: utf-8 -*-
"""
Tests for CCEA Import Boundary Check.

Tests verify that:
1. Cloud build doesn't import trading/execution modules
2. Agent modules are correctly classified
3. Zone classification works correctly
"""

import tempfile
from pathlib import Path

import pytest

from ccea.guardrails.import_check import (
    PROHIBITED_IN_CLOUD,
    ZoneType,
    check_cloud_imports,
    extract_imports_from_file,
    get_zone_for_module,
    matches_pattern,
)


class TestZoneClassification:
    """Tests for module zone classification."""

    def test_core_modules_are_shared(self):
        """Core modules should be in SHARED zone."""
        assert get_zone_for_module("core_models") == ZoneType.SHARED
        assert get_zone_for_module("core_config") == ZoneType.SHARED
        assert get_zone_for_module("core_events") == ZoneType.SHARED

    def test_impl_modules_are_shared(self):
        """Implementation modules should be in SHARED zone."""
        assert get_zone_for_module("impl_slippage") == ZoneType.SHARED
        assert get_zone_for_module("impl_fees") == ZoneType.SHARED
        assert get_zone_for_module("impl_latency") == ZoneType.SHARED

    def test_order_execution_is_agent_only(self):
        """Order execution modules should be in AGENT zone."""
        assert get_zone_for_module("adapters.alpaca.order_execution") == ZoneType.AGENT
        assert get_zone_for_module("adapters.oanda.order_execution") == ZoneType.AGENT
        assert get_zone_for_module("adapters.ib.order_execution") == ZoneType.AGENT

    def test_market_data_is_shared(self):
        """Market data modules should be in SHARED zone."""
        assert get_zone_for_module("adapters.binance.market_data") == ZoneType.SHARED
        assert get_zone_for_module("adapters.alpaca.market_data") == ZoneType.SHARED

    def test_execution_providers_is_agent(self):
        """Execution providers should be in AGENT zone."""
        assert get_zone_for_module("execution_providers") == ZoneType.AGENT
        assert get_zone_for_module("execution_providers_l3") == ZoneType.AGENT

    def test_cloud_modules_are_cloud(self):
        """Cloud-specific modules should be in CLOUD zone."""
        assert get_zone_for_module("app") == ZoneType.CLOUD
        assert get_zone_for_module("service_backtest") == ZoneType.CLOUD
        assert get_zone_for_module("service_train") == ZoneType.CLOUD


class TestPatternMatching:
    """Tests for pattern matching."""

    def test_exact_match(self):
        """Test exact module match."""
        assert matches_pattern("core_models", ["core_*"])
        assert matches_pattern("impl_fees", ["impl_*"])

    def test_wildcard_match(self):
        """Test wildcard pattern matching."""
        assert matches_pattern(
            "adapters.alpaca.order_execution",
            ["adapters.*.order_execution"]
        )
        assert matches_pattern(
            "adapters.oanda.order_execution",
            ["adapters.*.order_execution"]
        )

    def test_no_match(self):
        """Test non-matching patterns."""
        assert not matches_pattern("core_models", ["impl_*"])
        assert not matches_pattern(
            "adapters.alpaca.market_data",
            ["adapters.*.order_execution"]
        )


class TestImportExtraction:
    """Tests for import extraction from files."""

    def test_extract_simple_import(self):
        """Test extraction of simple imports."""
        code = "import os\nimport sys\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            imports = extract_imports_from_file(Path(f.name))

        assert ("os", 1) in imports
        assert ("sys", 2) in imports

    def test_extract_from_import(self):
        """Test extraction of from...import statements."""
        code = "from pathlib import Path\nfrom typing import List, Dict\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            imports = extract_imports_from_file(Path(f.name))

        assert ("pathlib", 1) in imports
        assert ("typing", 2) in imports

    def test_extract_nested_import(self):
        """Test extraction of nested module imports."""
        code = "from adapters.alpaca.order_execution import submit_order\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            imports = extract_imports_from_file(Path(f.name))

        assert ("adapters.alpaca.order_execution", 1) in imports


class TestCloudImportCheck:
    """Tests for Cloud import boundary checking."""

    def test_clean_cloud_code_passes(self):
        """Clean cloud code without trading imports should pass."""
        code = """
import os
from pathlib import Path
from core_models import Order
from impl_fees import calculate_fee
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "cloud_module.py").write_text(code)

            result = check_cloud_imports(tmppath)

            assert result.passed
            assert len(result.violations) == 0

    def test_trading_import_fails(self):
        """Code importing trading modules should fail."""
        code = """
from adapters.alpaca.order_execution import submit_order
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "bad_cloud.py").write_text(code)

            result = check_cloud_imports(tmppath)

            assert not result.passed
            assert len(result.violations) > 0
            assert "order_execution" in result.violations[0].module_imported

    def test_execution_providers_import_fails(self):
        """Code importing execution_providers should fail."""
        code = """
from execution_providers import get_provider
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "bad_cloud.py").write_text(code)

            result = check_cloud_imports(tmppath)

            assert not result.passed
            assert any("execution_providers" in v.module_imported for v in result.violations)

    def test_test_files_are_skipped(self):
        """Test files should be skipped."""
        code = """
from adapters.alpaca.order_execution import submit_order
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "test_trading.py").write_text(code)

            result = check_cloud_imports(tmppath)

            # Test files should be skipped
            assert result.passed


class TestProhibitedInCloudList:
    """Tests for the prohibited imports list."""

    def test_order_execution_in_prohibited(self):
        """Order execution modules should be in prohibited list."""
        patterns = PROHIBITED_IN_CLOUD
        assert any("order_execution" in p for p in patterns)

    def test_execution_providers_in_prohibited(self):
        """Execution providers should be in prohibited list."""
        patterns = PROHIBITED_IN_CLOUD
        assert "execution_providers" in patterns

    def test_service_signal_runner_in_prohibited(self):
        """Service signal runner should be in prohibited list."""
        patterns = PROHIBITED_IN_CLOUD
        assert "service_signal_runner" in patterns


class TestMultipleViolations:
    """Tests for multiple violation detection."""

    def test_multiple_violations_detected(self):
        """Multiple violations should all be detected."""
        code = """
from adapters.alpaca.order_execution import submit_order
from adapters.oanda.order_execution import place_order
from execution_providers import get_provider
from service_signal_runner import run
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "very_bad_cloud.py").write_text(code)

            result = check_cloud_imports(tmppath)

            assert not result.passed
            # Should have multiple violations
            assert len(result.violations) >= 3


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_file(self):
        """Empty file should pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "empty.py").write_text("")

            result = check_cloud_imports(tmppath)

            assert result.passed

    def test_syntax_error_file_handled(self):
        """Files with syntax errors should be handled gracefully."""
        code = "def broken(\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "broken.py").write_text(code)

            result = check_cloud_imports(tmppath)

            # Should not crash, just skip the file
            assert result.files_checked == 1

    def test_empty_directory(self):
        """Empty directory should pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            result = check_cloud_imports(tmppath)

            assert result.passed
            assert result.files_checked == 0
