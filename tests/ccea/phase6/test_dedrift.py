# -*- coding: utf-8 -*-
"""
Tests for WI-DEDRIFT-01: Canonical Stack Selection.

Verifies:
- Canonical stacks are packages/agent/* and packages/cloud/control_plane/*
- Non-canonical stacks (ccea/agent/*, ccea/control_plane/*) are deprecated
- Import boundaries enforce single stack usage
"""

import pytest
import warnings
from pathlib import Path
import sys


class TestCanonicalStacksExist:
    """Test canonical stacks exist and are properly structured."""

    @pytest.fixture
    def project_root(self):
        """Get project root path."""
        return Path(__file__).parent.parent.parent.parent

    def test_packages_agent_exists(self, project_root):
        """Test packages/agent/ is the canonical Agent stack."""
        agent_pkg = project_root / "packages" / "agent"
        assert agent_pkg.exists(), "packages/agent/ should be the canonical Agent stack"

    def test_packages_agent_has_daemon(self, project_root):
        """Test packages/agent/daemon/agentd.py exists."""
        agentd = project_root / "packages" / "agent" / "daemon" / "agentd.py"
        assert agentd.exists(), "packages/agent/daemon/agentd.py should exist"

    def test_packages_cloud_control_plane_exists(self, project_root):
        """Test packages/cloud/control_plane/ is the canonical Control Plane stack."""
        cp_pkg = project_root / "packages" / "cloud" / "control_plane"
        assert cp_pkg.exists(), "packages/cloud/control_plane/ should be canonical"

    def test_packages_agent_has_vault(self, project_root):
        """Test packages/agent/vault/ exists."""
        vault = project_root / "packages" / "agent" / "vault"
        assert vault.exists(), "packages/agent/vault/ should exist"

    def test_packages_agent_has_approval(self, project_root):
        """Test packages/agent/approval/ exists."""
        approval = project_root / "packages" / "agent" / "approval"
        assert approval.exists(), "packages/agent/approval/ should exist"


class TestDeprecatedStacksMarked:
    """Test non-canonical stacks are marked as deprecated."""

    @pytest.fixture
    def project_root(self):
        """Get project root path."""
        return Path(__file__).parent.parent.parent.parent

    def test_ccea_agent_deprecated(self, project_root):
        """Test ccea/agent/__init__.py has deprecation warning."""
        init_file = project_root / "ccea" / "agent" / "__init__.py"
        if not init_file.exists():
            pytest.skip("ccea/agent/__init__.py not found")

        content = init_file.read_text(encoding='utf-8')

        # Check for deprecation markers
        has_deprecation = any([
            "deprecated" in content.lower(),
            "DeprecationWarning" in content,
            "warnings.warn" in content,
        ])
        assert has_deprecation, "ccea/agent/ should be marked as deprecated"

    def test_ccea_control_plane_deprecated(self, project_root):
        """Test ccea/control_plane/__init__.py has deprecation warning."""
        init_file = project_root / "ccea" / "control_plane" / "__init__.py"
        if not init_file.exists():
            pytest.skip("ccea/control_plane/__init__.py not found")

        content = init_file.read_text(encoding='utf-8')

        # Check for deprecation markers
        has_deprecation = any([
            "deprecated" in content.lower(),
            "DeprecationWarning" in content,
            "warnings.warn" in content,
        ])
        assert has_deprecation, "ccea/control_plane/ should be marked as deprecated"

    def test_ccea_agent_points_to_canonical(self, project_root):
        """Test ccea/agent/ docstring points to canonical implementation."""
        init_file = project_root / "ccea" / "agent" / "__init__.py"
        if not init_file.exists():
            pytest.skip("ccea/agent/__init__.py not found")

        content = init_file.read_text(encoding='utf-8')

        # Should mention packages.agent
        assert "packages.agent" in content or "packages/agent" in content, \
            "ccea/agent/ should point to packages.agent as canonical"

    def test_ccea_control_plane_points_to_canonical(self, project_root):
        """Test ccea/control_plane/ docstring points to canonical implementation."""
        init_file = project_root / "ccea" / "control_plane" / "__init__.py"
        if not init_file.exists():
            pytest.skip("ccea/control_plane/__init__.py not found")

        content = init_file.read_text(encoding='utf-8')

        # Should mention packages.cloud.control_plane
        assert "packages.cloud.control_plane" in content or "packages/cloud/control_plane" in content, \
            "ccea/control_plane/ should point to packages.cloud.control_plane as canonical"


class TestDeprecationWarningsEmitted:
    """Test deprecation warnings are actually emitted on import."""

    def test_ccea_agent_import_warns(self):
        """Test importing ccea.agent emits deprecation warning."""
        # Clear any cached imports
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith('ccea.agent')]
        for mod in modules_to_remove:
            del sys.modules[mod]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                import ccea.agent  # noqa: F401
            except ImportError:
                pytest.skip("ccea.agent not importable")

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0, "Importing ccea.agent should emit DeprecationWarning"

    def test_ccea_control_plane_import_warns(self):
        """Test importing ccea.control_plane emits deprecation warning."""
        # Clear any cached imports
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith('ccea.control_plane')]
        for mod in modules_to_remove:
            del sys.modules[mod]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                import ccea.control_plane  # noqa: F401
            except ImportError:
                pytest.skip("ccea.control_plane not importable")

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0, "Importing ccea.control_plane should emit DeprecationWarning"


class TestNoNewCodeUsesDeprecated:
    """Test no new code uses deprecated stacks."""

    @pytest.fixture
    def project_root(self):
        """Get project root path."""
        return Path(__file__).parent.parent.parent.parent

    def test_packages_agent_no_import_ccea_agent(self, project_root):
        """Test packages/agent/ doesn't import from ccea.agent."""
        agent_pkg = project_root / "packages" / "agent"
        if not agent_pkg.exists():
            pytest.skip("packages/agent/ not found")

        for py_file in agent_pkg.rglob("*.py"):
            content = py_file.read_text(encoding='utf-8')
            # Check for imports from deprecated module
            has_deprecated_import = (
                "from ccea.agent" in content or
                "import ccea.agent" in content
            )
            assert not has_deprecated_import, \
                f"{py_file} should not import from deprecated ccea.agent"

    def test_packages_cloud_no_import_ccea_control_plane(self, project_root):
        """Test packages/cloud/ doesn't import from ccea.control_plane."""
        cloud_pkg = project_root / "packages" / "cloud"
        if not cloud_pkg.exists():
            pytest.skip("packages/cloud/ not found")

        for py_file in cloud_pkg.rglob("*.py"):
            content = py_file.read_text(encoding='utf-8')
            # Check for imports from deprecated module
            has_deprecated_import = (
                "from ccea.control_plane" in content or
                "import ccea.control_plane" in content
            )
            assert not has_deprecated_import, \
                f"{py_file} should not import from deprecated ccea.control_plane"


class TestStackSeparation:
    """Test Agent and Cloud stacks are properly separated."""

    @pytest.fixture
    def project_root(self):
        """Get project root path."""
        return Path(__file__).parent.parent.parent.parent

    def test_cloud_not_in_agent(self, project_root):
        """Test Cloud control plane code is not in Agent package."""
        agent_pkg = project_root / "packages" / "agent"
        if not agent_pkg.exists():
            pytest.skip("packages/agent/ not found")

        # Agent should not have control_plane module
        cp_in_agent = agent_pkg / "control_plane"
        assert not cp_in_agent.exists(), "Agent package should not contain control_plane"

    def test_agent_not_in_cloud(self, project_root):
        """Test Agent code is not in Cloud package."""
        cloud_pkg = project_root / "packages" / "cloud"
        if not cloud_pkg.exists():
            pytest.skip("packages/cloud/ not found")

        # Cloud should not have agent module
        agent_in_cloud = cloud_pkg / "agent"
        # It's OK to have references, but not full implementation
        if agent_in_cloud.exists():
            # Should be minimal/stub at most
            py_files = list(agent_in_cloud.rglob("*.py"))
            # Exclude __init__.py
            impl_files = [f for f in py_files if f.name != "__init__.py"]
            assert len(impl_files) == 0, "Cloud package should not have Agent implementation"
