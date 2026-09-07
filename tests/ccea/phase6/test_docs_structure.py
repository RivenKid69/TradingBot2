# -*- coding: utf-8 -*-
"""
Tests for CCEA Documentation Structure (WI-DOCS-01, WI-DOCS-02).

Verifies:
- Required CCEA documentation exists
- Documentation structure is correct
- Legacy script_live.py is not recommended in docs
"""

import pytest
from pathlib import Path
import re


class TestCCEADocsExist:
    """Test that required CCEA documentation exists (WI-DOCS-01)."""

    @pytest.fixture
    def docs_root(self):
        """Get docs root path."""
        return Path(__file__).parent.parent.parent.parent / "docs"

    def test_ccea_overview_exists(self, docs_root):
        """Test CCEA_OVERVIEW.md exists."""
        overview = docs_root / "CCEA_OVERVIEW.md"
        assert overview.exists(), "docs/CCEA_OVERVIEW.md is required but missing"

    def test_ccea_overview_content(self, docs_root):
        """Test CCEA_OVERVIEW.md has required sections."""
        overview = docs_root / "CCEA_OVERVIEW.md"
        if not overview.exists():
            pytest.skip("CCEA_OVERVIEW.md not found")

        content = overview.read_text(encoding="utf-8").lower()
        required_topics = [
            "boundary",
            "threat",  # threat model
            "legal",  # legal posture
            "cloud",
            "agent",
        ]
        for topic in required_topics:
            assert topic in content, f"CCEA_OVERVIEW.md should mention '{topic}'"


class TestCloudDocsExist:
    """Test docs/cloud/* structure exists."""

    @pytest.fixture
    def cloud_docs(self):
        """Get cloud docs path."""
        return Path(__file__).parent.parent.parent.parent / "docs" / "cloud"

    def test_cloud_readme_exists(self, cloud_docs):
        """Test docs/cloud/README.md exists."""
        readme = cloud_docs / "README.md"
        assert readme.exists(), "docs/cloud/README.md is required"

    def test_control_plane_api_exists(self, cloud_docs):
        """Test control plane API documentation exists."""
        api_doc = cloud_docs / "CONTROL_PLANE_API.md"
        assert api_doc.exists(), "docs/cloud/CONTROL_PLANE_API.md is required"

    def test_governance_exists(self, cloud_docs):
        """Test governance documentation exists."""
        gov_doc = cloud_docs / "GOVERNANCE.md"
        assert gov_doc.exists(), "docs/cloud/GOVERNANCE.md is required"


class TestAgentDocsExist:
    """Test docs/agent/* structure exists."""

    @pytest.fixture
    def agent_docs(self):
        """Get agent docs path."""
        return Path(__file__).parent.parent.parent.parent / "docs" / "agent"

    def test_agent_readme_exists(self, agent_docs):
        """Test docs/agent/README.md exists."""
        readme = agent_docs / "README.md"
        assert readme.exists(), "docs/agent/README.md is required"

    def test_installation_exists(self, agent_docs):
        """Test installation documentation exists."""
        install = agent_docs / "INSTALLATION.md"
        assert install.exists(), "docs/agent/INSTALLATION.md is required"

    def test_local_vault_exists(self, agent_docs):
        """Test local vault documentation exists."""
        vault = agent_docs / "LOCAL_VAULT.md"
        assert vault.exists(), "docs/agent/LOCAL_VAULT.md is required"

    def test_approvals_exists(self, agent_docs):
        """Test approvals documentation exists."""
        approvals = agent_docs / "APPROVALS.md"
        assert approvals.exists(), "docs/agent/APPROVALS.md is required"

    def test_risk_controls_exists(self, agent_docs):
        """Test risk controls documentation exists."""
        risk = agent_docs / "RISK_CONTROLS.md"
        assert risk.exists(), "docs/agent/RISK_CONTROLS.md is required"

    def test_degraded_modes_exists(self, agent_docs):
        """Test degraded modes documentation exists."""
        degraded = agent_docs / "DEGRADED_MODES.md"
        assert degraded.exists(), "docs/agent/DEGRADED_MODES.md is required"


class TestRunbooksExist:
    """Test docs/runbooks/* structure exists."""

    @pytest.fixture
    def runbooks(self):
        """Get runbooks path."""
        return Path(__file__).parent.parent.parent.parent / "docs" / "runbooks"

    def test_runbooks_readme_exists(self, runbooks):
        """Test docs/runbooks/README.md exists."""
        readme = runbooks / "README.md"
        assert readme.exists(), "docs/runbooks/README.md is required"

    def test_incident_response_exists(self, runbooks):
        """Test incident response runbook exists."""
        incident = runbooks / "INCIDENT_RESPONSE.md"
        assert incident.exists(), "docs/runbooks/INCIDENT_RESPONSE.md is required"

    def test_kill_switch_exists(self, runbooks):
        """Test kill switch runbook exists."""
        kill_switch = runbooks / "KILL_SWITCH.md"
        assert kill_switch.exists(), "docs/runbooks/KILL_SWITCH.md is required"

    def test_recovery_exists(self, runbooks):
        """Test recovery runbook exists."""
        recovery = runbooks / "RECOVERY.md"
        assert recovery.exists(), "docs/runbooks/RECOVERY.md is required"


class TestLegalDocsExist:
    """Test docs/legal/* structure exists."""

    @pytest.fixture
    def legal_docs(self):
        """Get legal docs path."""
        return Path(__file__).parent.parent.parent.parent / "docs" / "legal"

    def test_tos_exists(self, legal_docs):
        """Test TERMS_OF_SERVICE.md exists."""
        tos = legal_docs / "TERMS_OF_SERVICE.md"
        assert tos.exists(), "docs/legal/TERMS_OF_SERVICE.md is required"

    def test_privacy_policy_exists(self, legal_docs):
        """Test PRIVACY_POLICY.md exists."""
        pp = legal_docs / "PRIVACY_POLICY.md"
        assert pp.exists(), "docs/legal/PRIVACY_POLICY.md is required"

    def test_dpa_template_exists(self, legal_docs):
        """Test DPA_TEMPLATE.md exists."""
        dpa = legal_docs / "DPA_TEMPLATE.md"
        assert dpa.exists(), "docs/legal/DPA_TEMPLATE.md is required"

    def test_aup_exists(self, legal_docs):
        """Test AUP.md exists."""
        aup = legal_docs / "AUP.md"
        assert aup.exists(), "docs/legal/AUP.md is required"


class TestScriptLiveNotRecommended:
    """Test script_live.py is not recommended as production path (WI-DOCS-02)."""

    @pytest.fixture
    def project_root(self):
        """Get project root path."""
        return Path(__file__).parent.parent.parent.parent

    def test_readme_has_ccea_warning(self, project_root):
        """Test README mentions script_live.py is not production path."""
        readme = project_root / "README.md"
        if not readme.exists():
            pytest.skip("README.md not found")

        content = readme.read_text(encoding="utf-8")

        # Check for CCEA/Agent workflow mention
        assert "agent" in content.lower(), "README should mention Agent workflow"
        assert "ccea" in content.lower(), "README should mention CCEA"

        # If script_live.py is mentioned, it should be marked as development/legacy
        if "script_live.py" in content:
            # Should have context like "development", "testing", "legacy"
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if "script_live.py" in line.lower():
                    context = "\n".join(lines[max(0, i - 3) : i + 4]).lower()
                    has_warning = any(
                        word in context
                        for word in [
                            "development",
                            "testing",
                            "dry-run",
                            "legacy",
                            "not production",
                        ]
                    )
                    # At least one mention should have warning context
                    if has_warning:
                        return
            # If we get here, check for production Agent mention
            assert (
                "packages.agent" in content.lower() or "agentd" in content.lower()
            ), "README should reference production Agent daemon"

    def test_architecture_has_ccea_note(self, project_root):
        """Test ARCHITECTURE.md mentions CCEA for live trading."""
        arch = project_root / "ARCHITECTURE.md"
        if not arch.exists():
            pytest.skip("ARCHITECTURE.md not found")

        content = arch.read_text(encoding="utf-8")

        # Should mention CCEA or Agent for live trading
        assert (
            "ccea" in content.lower() or "agent zone" in content.lower()
        ), "ARCHITECTURE.md should mention CCEA architecture"

    def test_getting_started_ccea_compliant(self, project_root):
        """Test GETTING_STARTED.md is CCEA-compliant."""
        gs = project_root / "docs" / "GETTING_STARTED.md"
        if not gs.exists():
            pytest.skip("GETTING_STARTED.md not found")

        content = gs.read_text(encoding="utf-8")

        # If mentions live trading, should have CCEA context
        if "live trading" in content.lower() or "live execution" in content.lower():
            has_ccea_context = any(
                term in content.lower()
                for term in ["agent", "ccea", "local execution", "local vault"]
            )
            assert has_ccea_context, "GETTING_STARTED.md should mention Agent for live trading"

    def test_script_live_not_sole_path(self, project_root):
        """Test script_live.py is not the only documented live trading path."""
        docs_dir = project_root / "docs"

        # Find all markdown files mentioning script_live.py
        script_live_mentions = []
        for md_file in docs_dir.rglob("*.md"):
            # Skip archive
            if "archive" in str(md_file).lower():
                continue
            try:
                content = md_file.read_text(encoding="utf-8")
                if "script_live.py" in content:
                    script_live_mentions.append(md_file)
            except Exception:
                continue

        # For each file mentioning script_live.py, check for Agent context
        for md_file in script_live_mentions:
            content = md_file.read_text(encoding="utf-8").lower()
            has_ccea_context = any(
                term in content
                for term in ["agent", "ccea", "development", "testing", "dry-run", "legacy"]
            )
            assert has_ccea_context, f"{md_file} mentions script_live.py without CCEA/Agent context"


class TestArchiveExists:
    """Test docs/archive exists for legacy documentation."""

    @pytest.fixture
    def archive_dir(self):
        """Get archive directory path."""
        return Path(__file__).parent.parent.parent.parent / "docs" / "archive"

    def test_archive_directory_exists(self, archive_dir):
        """Test docs/archive/ directory exists."""
        assert archive_dir.exists(), "docs/archive/ directory should exist for legacy docs"

    def test_archive_has_readme(self, archive_dir):
        """Test archive has README or deprecation note."""
        if not archive_dir.exists():
            pytest.skip("Archive directory not found")

        # Either README.md or deprecated directory should exist
        has_readme = (archive_dir / "README.md").exists()
        has_deprecated = (archive_dir / "deprecated").exists()

        assert (
            has_readme or has_deprecated or any(archive_dir.iterdir())
        ), "Archive directory should have content or README"
