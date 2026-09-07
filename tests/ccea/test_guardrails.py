# -*- coding: utf-8 -*-
"""
Tests for CCEA guardrails.

Phase 2 Implementation: Tests for import checking, cloud allowlist,
and build artifact verification.
"""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path


class TestImportCheck:
    """Tests for import boundary checking."""

    def test_check_cloud_imports_clean(self):
        """Test clean cloud directory passes."""
        from ccea.guardrails.import_check import check_cloud_imports

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create clean Python file
            (tmppath / "clean_module.py").write_text(
                """
import numpy as np
import pandas as pd
from packages.shared.contracts import OrderIntent
"""
            )

            result = check_cloud_imports(tmppath)
            assert result.passed is True
            assert len(result.violations) == 0

    def test_check_cloud_imports_violation(self):
        """Test that order_execution import is caught."""
        from ccea.guardrails.import_check import check_cloud_imports

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create file with prohibited import
            (tmppath / "bad_module.py").write_text(
                """
import numpy as np
from adapters.alpaca.order_execution import submit_order
"""
            )

            result = check_cloud_imports(tmppath)
            assert result.passed is False
            assert any("order_execution" in v.module_imported for v in result.violations)

    def test_get_zone_for_module(self):
        """Test zone classification."""
        from ccea.guardrails.import_check import get_zone_for_module, ZoneType

        # Agent modules
        assert get_zone_for_module("adapters.alpaca.order_execution") == ZoneType.AGENT
        assert get_zone_for_module("execution_providers") == ZoneType.AGENT
        assert get_zone_for_module("service_signal_runner") == ZoneType.AGENT

        # Shared modules
        assert get_zone_for_module("adapters.alpaca.market_data") == ZoneType.SHARED
        assert get_zone_for_module("core_models") == ZoneType.SHARED

        # Cloud modules
        assert get_zone_for_module("app") == ZoneType.CLOUD
        assert get_zone_for_module("service_backtest") == ZoneType.CLOUD


class TestCloudAllowlist:
    """Tests for cloud dependency allowlist."""

    def test_stdlib_allowed(self):
        """Test that stdlib modules are allowed."""
        from ccea.guardrails.cloud_allowlist import is_cloud_allowed

        assert is_cloud_allowed("os") is True
        assert is_cloud_allowed("sys") is True
        assert is_cloud_allowed("json") is True
        assert is_cloud_allowed("pathlib") is True
        assert is_cloud_allowed("typing") is True
        assert is_cloud_allowed("collections.abc") is True

    def test_allowed_third_party(self):
        """Test that allowed third-party packages pass."""
        from ccea.guardrails.cloud_allowlist import is_cloud_allowed

        assert is_cloud_allowed("numpy") is True
        assert is_cloud_allowed("pandas") is True
        assert is_cloud_allowed("torch") is True
        assert is_cloud_allowed("pydantic") is True

    def test_prohibited_packages(self):
        """Test that prohibited packages are rejected."""
        from ccea.guardrails.cloud_allowlist import is_prohibited_package

        assert is_prohibited_package("ib_insync") is True
        assert is_prohibited_package("alpaca_trade_api") is True

    def test_prohibited_internal(self):
        """Test that trading modules are prohibited."""
        from ccea.guardrails.cloud_allowlist import is_prohibited_internal

        assert is_prohibited_internal("adapters.alpaca.order_execution") is True
        assert is_prohibited_internal("packages.agent.vault") is True
        assert is_prohibited_internal("execution_providers") is True
        assert is_prohibited_internal("service_signal_runner") is True

        # Data modules should be allowed
        assert is_prohibited_internal("adapters.alpaca.market_data") is False
        assert is_prohibited_internal("packages.shared.contracts") is False

    def test_validate_cloud_build(self):
        """Test cloud build validation."""
        from ccea.guardrails.cloud_allowlist import validate_cloud_build

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create clean module
            (tmppath / "clean.py").write_text(
                """
import numpy as np
from packages.shared.contracts import OrderIntent
"""
            )

            result = validate_cloud_build(tmppath, check_transitive=False)
            assert result.passed is True

    def test_validate_cloud_build_violation(self):
        """Test cloud build validation catches violations."""
        from ccea.guardrails.cloud_allowlist import validate_cloud_build

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create module with prohibited import
            (tmppath / "bad.py").write_text(
                """
from packages.agent.vault import LocalVault
"""
            )

            result = validate_cloud_build(tmppath, check_transitive=False)
            assert result.passed is False


class TestTransitiveDependencyChecker:
    """Tests for transitive dependency checking."""

    def test_transitive_checker_build_graph(self):
        """Test building import graph."""
        from ccea.guardrails.cloud_allowlist import TransitiveDependencyChecker

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create module hierarchy
            (tmppath / "module_a.py").write_text(
                """
from module_b import foo
"""
            )
            (tmppath / "module_b.py").write_text(
                """
import numpy
"""
            )

            checker = TransitiveDependencyChecker(tmppath)
            checker.build_graph()

            assert "module_a" in checker.import_graph
            assert "module_b" in checker.import_graph["module_a"]


class TestBuildArtifactCheck:
    """Tests for build artifact verification."""

    def test_scan_directory_clean(self):
        """Test scanning clean directory."""
        from ccea.guardrails.build_artifact_check import scan_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create clean file
            (tmppath / "clean.py").write_text(
                """
def process_data(data):
    return data * 2
"""
            )

            result = scan_directory(tmppath)
            assert result.passed is True

    def test_scan_directory_prohibited_file(self):
        """Test detecting prohibited file names."""
        from ccea.guardrails.build_artifact_check import scan_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create file with prohibited name
            (tmppath / "order_execution.py").write_text(
                """
def submit_order():
    pass
"""
            )

            result = scan_directory(tmppath)
            assert result.passed is False
            assert any("order_execution" in v.file_path for v in result.violations)

    def test_scan_directory_prohibited_code(self):
        """Test detecting prohibited code patterns."""
        from ccea.guardrails.build_artifact_check import scan_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create file with prohibited code
            (tmppath / "trading.py").write_text(
                """
def trade():
    broker.submit_order(symbol="AAPL", qty=100)
"""
            )

            result = scan_directory(tmppath)
            assert result.passed is False
            assert any("submit_order" in str(v) for v in result.violations)

    def test_scan_directory_prohibited_import(self):
        """Test detecting prohibited imports."""
        from ccea.guardrails.build_artifact_check import scan_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create file with prohibited import
            (tmppath / "module.py").write_text(
                """
from packages.agent.vault import LocalVault
from execution_providers import get_provider
"""
            )

            result = scan_directory(tmppath)
            assert result.passed is False

    def test_verify_cloud_manifest(self):
        """Test manifest verification."""
        from ccea.guardrails.build_artifact_check import verify_cloud_manifest

        # Clean manifest
        clean_manifest = {
            "modules": ["packages.shared.contracts", "packages.cloud.research"],
            "dependencies": ["numpy", "pandas"],
            "entry_points": {},
        }

        result = verify_cloud_manifest(clean_manifest)
        assert result.passed is True

    def test_verify_cloud_manifest_violation(self):
        """Test manifest with prohibited content."""
        from ccea.guardrails.build_artifact_check import verify_cloud_manifest

        # Manifest with prohibited module
        bad_manifest = {
            "modules": ["order_execution", "packages.cloud.research"],
            "dependencies": ["numpy", "ib_insync"],  # ib_insync prohibited!
        }

        result = verify_cloud_manifest(bad_manifest)
        assert result.passed is False


class TestZoneMapping:
    """Tests for adapter zone mapping."""

    def test_data_only_modules(self):
        """Test data-only module classification."""
        from packages.shared.adapters.zone_mapping import DATA_ONLY_MODULES

        assert "adapters.alpaca.market_data" in DATA_ONLY_MODULES
        assert "adapters.binance.market_data" in DATA_ONLY_MODULES
        assert "adapters.alpaca.fees" in DATA_ONLY_MODULES
        assert "adapters.polygon.trading_hours" in DATA_ONLY_MODULES

    def test_trading_only_modules(self):
        """Test trading-only module classification."""
        from packages.shared.adapters.zone_mapping import TRADING_ONLY_MODULES

        assert "adapters.alpaca.order_execution" in TRADING_ONLY_MODULES
        assert "adapters.binance.futures_order_execution" in TRADING_ONLY_MODULES
        assert "execution_providers" in TRADING_ONLY_MODULES
        assert "service_signal_runner" in TRADING_ONLY_MODULES

    def test_get_adapter_zone(self):
        """Test getting adapter zone."""
        from packages.shared.adapters.zone_mapping import get_adapter_zone, AdapterZone

        assert get_adapter_zone("adapters.alpaca.market_data") == AdapterZone.SHARED
        assert get_adapter_zone("adapters.alpaca.order_execution") == AdapterZone.AGENT
        assert get_adapter_zone("adapters.binance.fees") == AdapterZone.SHARED

    def test_is_cloud_safe(self):
        """Test cloud safety check."""
        from packages.shared.adapters.zone_mapping import is_cloud_safe

        assert is_cloud_safe("adapters.alpaca.market_data") is True
        assert is_cloud_safe("adapters.binance.fees") is True
        assert is_cloud_safe("adapters.alpaca.order_execution") is False
        assert is_cloud_safe("execution_providers") is False

    def test_is_agent_only(self):
        """Test agent-only check."""
        from packages.shared.adapters.zone_mapping import is_agent_only

        assert is_agent_only("adapters.alpaca.order_execution") is True
        assert is_agent_only("execution_providers") is True
        assert is_agent_only("adapters.alpaca.market_data") is False

    def test_validate_cloud_imports(self):
        """Test cloud import validation."""
        from packages.shared.adapters.zone_mapping import validate_cloud_imports

        # Clean imports
        clean_imports = {
            "adapters.alpaca.market_data",
            "adapters.binance.fees",
            "numpy",
        }
        violations = validate_cloud_imports(clean_imports)
        assert len(violations) == 0

        # Imports with violations
        bad_imports = {
            "adapters.alpaca.market_data",
            "adapters.alpaca.order_execution",  # Violation!
            "execution_providers",  # Violation!
        }
        violations = validate_cloud_imports(bad_imports)
        assert len(violations) == 2

    def test_no_zone_overlap(self):
        """Test that no module is in both zones."""
        from packages.shared.adapters.zone_mapping import validate_zone_separation

        issues = validate_zone_separation()
        assert "overlapping_modules" not in issues or len(issues["overlapping_modules"]) == 0
