# -*- coding: utf-8 -*-
"""
Phase 1 Architecture Tests - Canonical Stack Verification.

Tests verify that:
1. Canonical stacks (packages/) are production-ready
2. Legacy stacks (ccea/agent, ccea/control_plane) emit deprecation warnings
3. Build artifacts contain only appropriate modules
4. Zone separation is enforced
5. No double paths in CI/Makefile

Work Item: WI-DEDRIFT-01
Requirement: Phase 1 - Canonization and drift elimination
"""

from __future__ import annotations

import ast
import importlib
import os
import re
import sys
import warnings
from pathlib import Path
from typing import List, Set

import pytest


class TestCanonicalAgentStack:
    """Tests verifying packages/agent is the canonical Agent implementation."""

    def test_packages_agent_is_importable(self):
        """packages.agent must be importable."""
        import packages.agent
        assert hasattr(packages.agent, "__version__")
        assert packages.agent.ZONE == "agent"

    def test_packages_agent_has_required_components(self):
        """packages.agent must declare required components."""
        import packages.agent
        required = [
            "LocalVault",
            "PolicyFirewall",
            "LiveExecutionEngine",
            "ApprovalManager",
        ]
        for component in required:
            assert component in packages.agent.AGENT_COMPONENTS

    def test_packages_agent_version_is_2_or_higher(self):
        """Canonical agent version should be 2.0.0+."""
        import packages.agent
        major = int(packages.agent.__version__.split(".")[0])
        assert major >= 2, "Canonical agent should be version 2.0.0+"


class TestDeprecatedAgentStack:
    """Tests verifying ccea.agent emits deprecation warnings."""

    def test_ccea_agent_emits_deprecation_warning(self):
        """ccea.agent import must emit DeprecationWarning."""
        # Clear any cached import
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("ccea.agent"):
                del sys.modules[mod_name]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import ccea.agent
            # Reload to ensure warning is emitted
            importlib.reload(ccea.agent)

            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) > 0, (
                "ccea.agent should emit DeprecationWarning"
            )
            assert "deprecated" in str(deprecation_warnings[0].message).lower()


class TestCanonicalCloudStack:
    """Tests verifying packages/cloud is the canonical Cloud implementation."""

    def test_packages_cloud_control_plane_exists(self):
        """packages.cloud.control_plane must exist."""
        control_plane_path = Path(__file__).parent.parent.parent / "packages" / "cloud" / "control_plane"
        assert control_plane_path.exists(), "packages/cloud/control_plane must exist"
        assert (control_plane_path / "app.py").exists(), "FastAPI app must exist"
        assert (control_plane_path / "models.py").exists(), "SQLAlchemy models must exist"

    def test_packages_cloud_has_routers(self):
        """packages.cloud.control_plane must have routers."""
        routers_path = Path(__file__).parent.parent.parent / "packages" / "cloud" / "control_plane" / "routers"
        assert routers_path.exists(), "routers directory must exist"
        assert len(list(routers_path.glob("*.py"))) > 0, "routers must contain modules"

    def test_packages_cloud_has_services(self):
        """packages.cloud.control_plane must have services."""
        services_path = Path(__file__).parent.parent.parent / "packages" / "cloud" / "control_plane" / "services"
        assert services_path.exists(), "services directory must exist"


class TestDeprecatedControlPlaneStack:
    """Tests verifying ccea.control_plane emits deprecation warnings."""

    def test_ccea_control_plane_emits_deprecation_warning(self):
        """ccea.control_plane import must emit DeprecationWarning."""
        # Clear any cached import
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("ccea.control_plane"):
                del sys.modules[mod_name]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import ccea.control_plane
            importlib.reload(ccea.control_plane)

            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) > 0, (
                "ccea.control_plane should emit DeprecationWarning"
            )
            assert "deprecated" in str(deprecation_warnings[0].message).lower()


class TestBuildArtifactZoneSeparation:
    """Tests verifying zone separation in build artifacts."""

    @pytest.fixture
    def build_script_path(self) -> Path:
        """Get path to build script."""
        return Path(__file__).parent.parent.parent / "tools" / "build_zone_distributions.py"

    def test_build_script_exists(self, build_script_path: Path):
        """Zone distribution build script must exist."""
        assert build_script_path.exists()

    def test_cloud_spec_excludes_agent_modules(self, build_script_path: Path):
        """Cloud distribution spec must not include agent modules."""
        content = build_script_path.read_text()

        # Parse the file to find cloud_spec
        tree = ast.parse(content)

        cloud_paths: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "cloud_spec":
                        # Extract include_paths from DistributionSpec call
                        if isinstance(node.value, ast.Call):
                            for kw in node.value.keywords:
                                if kw.arg == "include_paths":
                                    if isinstance(kw.value, ast.List):
                                        for elt in kw.value.elts:
                                            if isinstance(elt, ast.Constant):
                                                cloud_paths.append(elt.value)

        # Cloud should NOT include agent modules
        prohibited = ["packages/agent", "ccea/agent", "ccea/control_plane"]
        for path in cloud_paths:
            for prohibited_path in prohibited:
                assert prohibited_path not in path, (
                    f"Cloud spec should not include {prohibited_path}"
                )

    def test_agent_spec_excludes_cloud_modules(self, build_script_path: Path):
        """Agent distribution spec must not include cloud-specific modules."""
        content = build_script_path.read_text()

        tree = ast.parse(content)

        agent_paths: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "agent_spec":
                        if isinstance(node.value, ast.Call):
                            for kw in node.value.keywords:
                                if kw.arg == "include_paths":
                                    if isinstance(kw.value, ast.List):
                                        for elt in kw.value.elts:
                                            if isinstance(elt, ast.Constant):
                                                agent_paths.append(elt.value)

        # Agent should NOT include cloud-internal modules
        prohibited = ["packages/cloud"]
        for path in agent_paths:
            for prohibited_path in prohibited:
                assert prohibited_path not in path, (
                    f"Agent spec should not include {prohibited_path}"
                )


class TestMakefileNoDoublePaths:
    """Tests verifying Makefile has no double build paths."""

    @pytest.fixture
    def makefile_path(self) -> Path:
        """Get path to Makefile."""
        return Path(__file__).parent.parent.parent / "Makefile"

    def test_makefile_exists(self, makefile_path: Path):
        """Makefile must exist."""
        assert makefile_path.exists()

    def test_makefile_has_zone_targets(self, makefile_path: Path):
        """Makefile must have zone-separated targets."""
        content = makefile_path.read_text()

        assert "dist-cloud" in content, "Makefile must have dist-cloud target"
        assert "dist-agent" in content, "Makefile must have dist-agent target"
        assert "artifact-check-cloud" in content, "Makefile must have artifact-check-cloud target"

    def test_makefile_no_legacy_agent_paths(self, makefile_path: Path):
        """Makefile should not reference legacy agent directly."""
        content = makefile_path.read_text()

        # Check for problematic patterns
        # Note: references to docs/archive are OK
        lines = content.split("\n")
        for i, line in enumerate(lines):
            # Skip comments and doc references
            if line.strip().startswith("#") or "docs/archive" in line:
                continue
            # Check for direct legacy module builds
            if "ccea/agent" in line and "build" in line.lower():
                pytest.fail(f"Line {i+1}: Legacy agent build path found: {line}")


class TestCIWorkflowsNoDoublePaths:
    """Tests verifying CI workflows use canonical paths."""

    @pytest.fixture
    def workflows_dir(self) -> Path:
        """Get path to workflows directory."""
        return Path(__file__).parent.parent.parent / ".github" / "workflows"

    def test_workflows_exist(self, workflows_dir: Path):
        """GitHub workflows directory must exist."""
        assert workflows_dir.exists()

    def test_build_workflow_uses_zone_separation(self, workflows_dir: Path):
        """Build workflow should use zone-separated artifacts."""
        build_workflow = workflows_dir / "build-and-test.yml"
        if not build_workflow.exists():
            pytest.skip("build-and-test.yml not found")

        content = build_workflow.read_text()

        # Should reference zone build commands
        # At minimum, should not have ambiguous paths
        lines = content.split("\n")
        for i, line in enumerate(lines):
            # Check for problematic patterns that could build wrong artifact
            if "pip install" in line and "ccea_agent" in line and "cloud" in line.lower():
                pytest.fail(
                    f"Line {i+1}: Potential cross-zone install in CI: {line}"
                )


class TestDocumentationConsistency:
    """Tests verifying documentation points to canonical implementations."""

    @pytest.fixture
    def architecture_md(self) -> Path:
        """Get path to ARCHITECTURE.md."""
        return Path(__file__).parent.parent.parent / "ARCHITECTURE.md"

    @pytest.fixture
    def readme_md(self) -> Path:
        """Get path to README.md."""
        return Path(__file__).parent.parent.parent / "README.md"

    def test_architecture_references_packages_agent(self, architecture_md: Path):
        """ARCHITECTURE.md must reference packages/agent."""
        content = architecture_md.read_text()
        assert "packages/agent" in content or "packages.agent" in content

    def test_architecture_references_packages_cloud(self, architecture_md: Path):
        """ARCHITECTURE.md must reference packages/cloud."""
        content = architecture_md.read_text()
        assert "packages/cloud" in content or "packages.cloud" in content

    def test_readme_mentions_agent_daemon(self, readme_md: Path):
        """README.md must mention Agent daemon for live trading."""
        content = readme_md.read_text()
        assert "packages.agent.daemon.agentd" in content

    def test_readme_script_live_is_development_only(self, readme_md: Path):
        """README.md must mark script_live.py as development-only."""
        content = readme_md.read_text()
        # Check that script_live.py is mentioned in context of development/testing
        script_live_pattern = r"script_live\.py.*(development|testing|dry-run)"
        assert re.search(script_live_pattern, content, re.IGNORECASE), (
            "README should indicate script_live.py is for development/testing"
        )


class TestLegacyStacksDocumented:
    """Tests verifying legacy stacks are properly documented."""

    @pytest.fixture
    def legacy_docs_path(self) -> Path:
        """Get path to legacy stacks documentation."""
        return Path(__file__).parent.parent.parent / "docs" / "archive" / "LEGACY_STACKS.md"

    def test_legacy_docs_exist(self, legacy_docs_path: Path):
        """Legacy stacks documentation must exist."""
        assert legacy_docs_path.exists(), (
            "docs/archive/LEGACY_STACKS.md must exist"
        )

    def test_legacy_docs_covers_ccea_agent(self, legacy_docs_path: Path):
        """Legacy docs must cover ccea.agent deprecation."""
        content = legacy_docs_path.read_text()
        assert "ccea.agent" in content or "ccea/agent" in content

    def test_legacy_docs_covers_ccea_control_plane(self, legacy_docs_path: Path):
        """Legacy docs must cover ccea.control_plane deprecation."""
        content = legacy_docs_path.read_text()
        assert "ccea.control_plane" in content or "ccea/control_plane" in content

    def test_legacy_docs_has_migration_guide(self, legacy_docs_path: Path):
        """Legacy docs must have migration guide."""
        content = legacy_docs_path.read_text()
        assert "migration" in content.lower()


class TestScriptLiveGuard:
    """Tests verifying script_live.py has production guard."""

    @pytest.fixture
    def script_live_path(self) -> Path:
        """Get path to script_live.py."""
        return Path(__file__).parent.parent.parent / "script_live.py"

    def test_script_live_has_deprecation_warning(self, script_live_path: Path):
        """script_live.py must emit deprecation warning."""
        content = script_live_path.read_text()
        assert "DeprecationWarning" in content
        assert "packages.agent.daemon.agentd" in content

    def test_script_live_checks_production_mode(self, script_live_path: Path):
        """script_live.py must check CCEA_PRODUCTION_MODE."""
        content = script_live_path.read_text()
        assert "CCEA_PRODUCTION_MODE" in content

    def test_script_live_has_override_env_var(self, script_live_path: Path):
        """script_live.py must allow override via env var."""
        content = script_live_path.read_text()
        assert "CCEA_ALLOW_LEGACY_LIVE" in content
