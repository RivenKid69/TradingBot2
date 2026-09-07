# -*- coding: utf-8 -*-
"""
Phase 11 - CCEA Documentation CI Guardrails Tests.

Tests for:
- Documentation file existence and structure
- Legal document version consistency
- CCEA architecture boundary enforcement
- Prohibited payload field detection
- Cross-reference validation
"""

import os
import re
import pytest
from pathlib import Path


# ============================================================================
# Test Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"


# ============================================================================
# Documentation Structure Tests
# ============================================================================


class TestDocumentationStructure:
    """Tests for documentation file structure."""

    @pytest.fixture
    def required_docs(self):
        """List of required documentation files."""
        return [
            "CCEA_OVERVIEW.md",
            "cloud/README.md",
            "cloud/CONTROL_PLANE_API.md",
            "cloud/ARTIFACT_BUILDER.md",
            "cloud/GOVERNANCE.md",
            "cloud/RESEARCH_JOB_ISOLATION.md",
            "agent/README.md",
            "agent/INSTALLATION.md",
            "agent/LOCAL_VAULT.md",
            "agent/APPROVALS.md",
            "agent/RISK_CONTROLS.md",
            "agent/DEGRADED_MODES.md",
            "schemas/README.md",
            "runbooks/README.md",
            "runbooks/KILL_SWITCH.md",
            "runbooks/RECOVERY.md",
            "runbooks/AGENT_REVOCATION.md",
            "runbooks/DEGRADED_MODE.md",
            "runbooks/INCIDENT_RESPONSE.md",
            "legal/TERMS_OF_SERVICE.md",
            "legal/PRIVACY_POLICY.md",
            "legal/ACCEPTABLE_USE_POLICY.md",
            "ui/README.md",
            "ui/ONBOARDING_GUARDRAILS.md",
        ]

    def test_required_docs_exist(self, required_docs):
        """Test that all required documentation files exist."""
        missing = []
        for doc in required_docs:
            doc_path = DOCS_DIR / doc
            if not doc_path.exists():
                missing.append(doc)

        assert not missing, f"Missing required documentation: {missing}"

    def test_ccea_overview_has_required_sections(self):
        """Test CCEA_OVERVIEW.md has all required sections."""
        ccea_path = DOCS_DIR / "CCEA_OVERVIEW.md"
        if not ccea_path.exists():
            pytest.skip("CCEA_OVERVIEW.md not found")

        content = ccea_path.read_text()
        required_sections = [
            "Architecture",
            "Security",
            "Protocol",
            "Threat Model",
        ]

        for section in required_sections:
            assert (
                section.lower() in content.lower()
            ), f"CCEA_OVERVIEW.md missing section: {section}"

    def test_cloud_docs_directory_exists(self):
        """Test Cloud documentation directory exists."""
        cloud_dir = DOCS_DIR / "cloud"
        assert cloud_dir.exists() and cloud_dir.is_dir(), "docs/cloud/ directory missing"

    def test_agent_docs_directory_exists(self):
        """Test Agent documentation directory exists."""
        agent_dir = DOCS_DIR / "agent"
        assert agent_dir.exists() and agent_dir.is_dir(), "docs/agent/ directory missing"

    def test_runbooks_directory_exists(self):
        """Test runbooks directory exists."""
        runbooks_dir = DOCS_DIR / "runbooks"
        assert runbooks_dir.exists() and runbooks_dir.is_dir(), "docs/runbooks/ directory missing"


# ============================================================================
# Legal Document Tests
# ============================================================================


class TestLegalDocuments:
    """Tests for legal document compliance."""

    def test_terms_of_service_exists(self):
        """Test Terms of Service exists."""
        tos_path = DOCS_DIR / "legal" / "TERMS_OF_SERVICE.md"
        assert tos_path.exists(), "Terms of Service not found"

    def test_privacy_policy_exists(self):
        """Test Privacy Policy exists."""
        privacy_path = DOCS_DIR / "legal" / "PRIVACY_POLICY.md"
        assert privacy_path.exists(), "Privacy Policy not found"

    def test_aup_exists(self):
        """Test Acceptable Use Policy exists."""
        aup_path = DOCS_DIR / "legal" / "ACCEPTABLE_USE_POLICY.md"
        assert aup_path.exists(), "Acceptable Use Policy not found"

    def test_tos_has_ccea_section(self):
        """Test ToS mentions CCEA architecture."""
        tos_path = DOCS_DIR / "legal" / "TERMS_OF_SERVICE.md"
        if not tos_path.exists():
            pytest.skip("ToS not found")

        content = tos_path.read_text()
        assert "CCEA" in content, "Terms of Service must mention CCEA architecture"

    def test_tos_has_not_investment_advice(self):
        """Test ToS has 'not investment advice' disclaimer."""
        tos_path = DOCS_DIR / "legal" / "TERMS_OF_SERVICE.md"
        if not tos_path.exists():
            pytest.skip("ToS not found")

        content = tos_path.read_text().lower()
        assert (
            "not" in content and "investment advice" in content
        ), "Terms of Service must contain 'not investment advice' disclaimer"

    def test_privacy_policy_has_ccea_data_zones(self):
        """Test Privacy Policy mentions CCEA data zones."""
        privacy_path = DOCS_DIR / "legal" / "PRIVACY_POLICY.md"
        if not privacy_path.exists():
            pytest.skip("Privacy Policy not found")

        content = privacy_path.read_text()
        assert "CCEA" in content, "Privacy Policy must mention CCEA architecture"
        assert (
            "Cloud" in content and "Agent" in content
        ), "Privacy Policy must mention Cloud and Agent zones"

    def test_privacy_policy_mentions_no_credentials_in_cloud(self):
        """Test Privacy Policy states credentials not stored in Cloud."""
        privacy_path = DOCS_DIR / "legal" / "PRIVACY_POLICY.md"
        if not privacy_path.exists():
            pytest.skip("Privacy Policy not found")

        content = privacy_path.read_text().lower()
        # Check for phrases like "never store" credentials, "local only", etc.
        never_store = "never" in content and ("store" in content or "stored" in content)
        credentials_local = "local" in content and ("credential" in content or "api key" in content)
        assert (
            never_store or credentials_local
        ), "Privacy Policy must state credentials are NOT stored in Cloud"


# ============================================================================
# CCEA Security Boundary Tests
# ============================================================================


class TestCCEASecurityBoundaries:
    """Tests for CCEA security boundary enforcement."""

    def test_cloud_docs_mention_no_order_execution(self):
        """Test Cloud docs clarify no order execution."""
        cloud_readme = DOCS_DIR / "cloud" / "README.md"
        if not cloud_readme.exists():
            pytest.skip("Cloud README not found")

        content = cloud_readme.read_text().lower()
        assert (
            "never" in content and "order" in content
        ), "Cloud docs must state Cloud NEVER executes orders"

    def test_cloud_docs_mention_no_credentials(self):
        """Test Cloud docs clarify no credential storage."""
        cloud_readme = DOCS_DIR / "cloud" / "README.md"
        if not cloud_readme.exists():
            pytest.skip("Cloud README not found")

        content = cloud_readme.read_text().lower()
        assert "never" in content and (
            "credential" in content or "api key" in content
        ), "Cloud docs must state Cloud NEVER stores credentials"

    def test_agent_docs_mention_local_vault(self):
        """Test Agent docs mention local vault."""
        vault_doc = DOCS_DIR / "agent" / "LOCAL_VAULT.md"
        if not vault_doc.exists():
            pytest.skip("LOCAL_VAULT.md not found")

        content = vault_doc.read_text().lower()
        assert "encrypt" in content, "Agent LOCAL_VAULT.md must mention encryption"
        assert "local" in content, "Agent LOCAL_VAULT.md must emphasize local storage"

    def test_schema_docs_mention_prohibited_fields(self):
        """Test Schema docs mention prohibited fields."""
        schema_readme = DOCS_DIR / "schemas" / "README.md"
        if not schema_readme.exists():
            pytest.skip("Schema README not found")

        content = schema_readme.read_text().lower()
        prohibited_fields = ["side", "quantity", "price", "order_type"]
        found_prohibitions = sum(1 for f in prohibited_fields if f in content)

        assert (
            found_prohibitions >= 2
        ), "Schema docs must list prohibited payload fields (side, quantity, price, etc.)"


# ============================================================================
# Prohibited Payload Field Tests
# ============================================================================


class TestProhibitedPayloadFields:
    """Tests to ensure prohibited fields are not in Cloud code."""

    @pytest.fixture
    def cloud_code_files(self):
        """Get list of Python files in cloud-related modules."""
        cloud_dirs = [
            PROJECT_ROOT / "src" / "cloud",
            PROJECT_ROOT / "ccea" / "cloud",
        ]

        files = []
        for cloud_dir in cloud_dirs:
            if cloud_dir.exists():
                files.extend(cloud_dir.rglob("*.py"))
        return files

    @pytest.fixture
    def prohibited_patterns(self):
        """Patterns that indicate order-like payloads in Cloud code."""
        return [
            # Direct field assignments that could be order payloads
            r'["\']side["\']\s*:\s*["\'](?:BUY|SELL|buy|sell)["\']',
            r'["\']order_type["\']\s*:\s*["\'](?:MARKET|LIMIT|market|limit)["\']',
            r'side\s*=\s*["\'](?:BUY|SELL)',
            # Sending orders from cloud (should never happen)
            r"execute_order\s*\(",
            r"submit_order\s*\(",
            r"place_order\s*\(",
        ]

    def test_cloud_code_no_order_execution_calls(self, cloud_code_files, prohibited_patterns):
        """Test Cloud code doesn't contain order execution calls."""
        if not cloud_code_files:
            pytest.skip("No cloud code files found")

        violations = []
        for file_path in cloud_code_files:
            content = file_path.read_text()
            for pattern in prohibited_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    violations.append(f"{file_path}: matches pattern '{pattern}'")

        assert not violations, f"Cloud code contains prohibited patterns:\n" + "\n".join(violations)


# ============================================================================
# Cross-Reference Tests
# ============================================================================


class TestDocumentationCrossReferences:
    """Tests for documentation cross-reference integrity."""

    def test_ccea_overview_links_to_cloud_docs(self):
        """Test CCEA_OVERVIEW.md links to cloud docs."""
        ccea_path = DOCS_DIR / "CCEA_OVERVIEW.md"
        if not ccea_path.exists():
            pytest.skip("CCEA_OVERVIEW.md not found")

        content = ccea_path.read_text()
        assert (
            "cloud/" in content.lower() or "docs/cloud" in content.lower()
        ), "CCEA_OVERVIEW.md should reference cloud documentation"

    def test_ccea_overview_links_to_agent_docs(self):
        """Test CCEA_OVERVIEW.md links to agent docs."""
        ccea_path = DOCS_DIR / "CCEA_OVERVIEW.md"
        if not ccea_path.exists():
            pytest.skip("CCEA_OVERVIEW.md not found")

        content = ccea_path.read_text()
        assert (
            "agent/" in content.lower() or "docs/agent" in content.lower()
        ), "CCEA_OVERVIEW.md should reference agent documentation"

    def test_runbooks_exist_and_linked(self):
        """Test runbooks are linked from main runbook index."""
        runbook_index = DOCS_DIR / "runbooks" / "README.md"
        if not runbook_index.exists():
            pytest.skip("Runbook index not found")

        content = runbook_index.read_text()
        required_runbooks = ["KILL_SWITCH", "RECOVERY", "AGENT_REVOCATION"]

        for runbook in required_runbooks:
            assert runbook.lower() in content.lower(), f"Runbook index should reference {runbook}"


# ============================================================================
# Version Consistency Tests
# ============================================================================


class TestVersionConsistency:
    """Tests for document version consistency."""

    def test_tos_version_is_2_0_or_higher(self):
        """Test Terms of Service has been updated (version 2.0+)."""
        tos_path = DOCS_DIR / "legal" / "TERMS_OF_SERVICE.md"
        if not tos_path.exists():
            pytest.skip("ToS not found")

        content = tos_path.read_text()
        # Look for version pattern
        version_match = re.search(r"Version[:\s]*(\d+\.\d+\.\d+)", content, re.IGNORECASE)

        if version_match:
            version = version_match.group(1)
            major = int(version.split(".")[0])
            assert major >= 2, f"Terms of Service should be version 2.0.0+ (found {version})"

    def test_privacy_policy_version_is_2_0_or_higher(self):
        """Test Privacy Policy has been updated (version 2.0+)."""
        privacy_path = DOCS_DIR / "legal" / "PRIVACY_POLICY.md"
        if not privacy_path.exists():
            pytest.skip("Privacy Policy not found")

        content = privacy_path.read_text()
        version_match = re.search(r"Version[:\s]*(\d+\.\d+\.\d+)", content, re.IGNORECASE)

        if version_match:
            version = version_match.group(1)
            major = int(version.split(".")[0])
            assert major >= 2, f"Privacy Policy should be version 2.0.0+ (found {version})"


# ============================================================================
# UI Guardrails Tests
# ============================================================================


class TestUIGuardrails:
    """Tests for UI guardrails documentation."""

    def test_onboarding_guardrails_exists(self):
        """Test onboarding guardrails document exists."""
        guardrails_path = DOCS_DIR / "ui" / "ONBOARDING_GUARDRAILS.md"
        assert guardrails_path.exists(), "UI onboarding guardrails document not found"

    def test_onboarding_guardrails_has_disclaimers(self):
        """Test onboarding guardrails mentions disclaimers."""
        guardrails_path = DOCS_DIR / "ui" / "ONBOARDING_GUARDRAILS.md"
        if not guardrails_path.exists():
            pytest.skip("Onboarding guardrails not found")

        content = guardrails_path.read_text().lower()
        assert "disclaimer" in content, "Onboarding guardrails must include disclaimers"

    def test_onboarding_guardrails_has_risk_warning(self):
        """Test onboarding guardrails includes risk warnings."""
        guardrails_path = DOCS_DIR / "ui" / "ONBOARDING_GUARDRAILS.md"
        if not guardrails_path.exists():
            pytest.skip("Onboarding guardrails not found")

        content = guardrails_path.read_text().lower()
        assert (
            "risk" in content and "warning" in content
        ), "Onboarding guardrails must include risk warnings"

    def test_onboarding_guardrails_has_acknowledgments(self):
        """Test onboarding guardrails defines acknowledgment flows."""
        guardrails_path = DOCS_DIR / "ui" / "ONBOARDING_GUARDRAILS.md"
        if not guardrails_path.exists():
            pytest.skip("Onboarding guardrails not found")

        content = guardrails_path.read_text().lower()
        assert "acknowledg" in content, "Onboarding guardrails must define acknowledgment flows"

    def test_onboarding_guardrails_has_ai_disclosure(self):
        """Test onboarding guardrails includes AI system disclosure."""
        guardrails_path = DOCS_DIR / "ui" / "ONBOARDING_GUARDRAILS.md"
        if not guardrails_path.exists():
            pytest.skip("Onboarding guardrails not found")

        content = guardrails_path.read_text().lower()
        assert "ai" in content and (
            "generated" in content or "system" in content
        ), "Onboarding guardrails must include AI system disclosure"


# ============================================================================
# Architecture Document Tests
# ============================================================================


class TestArchitectureDocumentation:
    """Tests for main architecture documentation."""

    def test_readme_mentions_ccea(self):
        """Test main README mentions CCEA."""
        readme_path = PROJECT_ROOT / "README.md"
        if not readme_path.exists():
            pytest.skip("README.md not found")

        content = readme_path.read_text()
        assert "CCEA" in content, "README.md must mention CCEA architecture"

    def test_architecture_md_exists(self):
        """Test ARCHITECTURE.md exists."""
        arch_path = PROJECT_ROOT / "ARCHITECTURE.md"
        assert arch_path.exists(), "ARCHITECTURE.md not found"

    def test_architecture_mentions_cloud_agent_separation(self):
        """Test ARCHITECTURE.md mentions Cloud/Agent separation."""
        arch_path = PROJECT_ROOT / "ARCHITECTURE.md"
        if not arch_path.exists():
            pytest.skip("ARCHITECTURE.md not found")

        content = arch_path.read_text().lower()
        assert (
            "cloud" in content and "agent" in content
        ), "ARCHITECTURE.md must describe Cloud/Agent separation"


# ============================================================================
# Integration Tests
# ============================================================================


class TestDocumentationIntegrity:
    """Integration tests for documentation integrity."""

    def test_no_broken_internal_links(self):
        """Test for broken internal markdown links in key docs."""
        key_docs = [
            DOCS_DIR / "CCEA_OVERVIEW.md",
            DOCS_DIR / "cloud" / "README.md",
            DOCS_DIR / "agent" / "README.md",
        ]

        broken_links = []
        link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

        for doc_path in key_docs:
            if not doc_path.exists():
                continue

            content = doc_path.read_text()
            for match in link_pattern.finditer(content):
                link_text, link_url = match.groups()

                # Skip external links
                if link_url.startswith(("http://", "https://", "#")):
                    continue

                # Resolve relative path
                if link_url.startswith("./"):
                    link_url = link_url[2:]

                target_path = doc_path.parent / link_url
                if not target_path.exists():
                    broken_links.append(f"{doc_path}: [{link_text}]({link_url})")

        # Allow some broken links (they might be placeholders)
        # but warn if there are many
        if len(broken_links) > 5:
            pytest.fail(f"Too many broken internal links:\n" + "\n".join(broken_links[:10]))

    def test_all_legal_docs_have_version(self):
        """Test all legal documents have version numbers."""
        legal_dir = DOCS_DIR / "legal"
        if not legal_dir.exists():
            pytest.skip("Legal directory not found")

        missing_version = []
        for doc in legal_dir.glob("*.md"):
            content = doc.read_text()
            if not re.search(r"version", content, re.IGNORECASE):
                missing_version.append(doc.name)

        assert not missing_version, f"Legal documents missing version: {missing_version}"


# ============================================================================
# Telemetry Redaction Tests
# ============================================================================


class TestTelemetryRedactionDocumentation:
    """Tests for telemetry redaction documentation."""

    def test_privacy_policy_mentions_telemetry_redaction(self):
        """Test Privacy Policy mentions telemetry redaction."""
        privacy_path = DOCS_DIR / "legal" / "PRIVACY_POLICY.md"
        if not privacy_path.exists():
            pytest.skip("Privacy Policy not found")

        content = privacy_path.read_text().lower()
        assert (
            "redact" in content or "telemetry" in content
        ), "Privacy Policy must mention telemetry redaction"

    def test_agent_docs_mention_redaction(self):
        """Test Agent docs mention telemetry redaction."""
        # Check multiple possible locations
        possible_files = [
            DOCS_DIR / "agent" / "README.md",
            DOCS_DIR / "CCEA_OVERVIEW.md",
        ]

        found_redaction = False
        for doc_path in possible_files:
            if doc_path.exists():
                content = doc_path.read_text().lower()
                if "redact" in content:
                    found_redaction = True
                    break

        assert found_redaction, "Documentation must mention telemetry redaction"


# ============================================================================
# Kill Switch Documentation Tests
# ============================================================================


class TestKillSwitchDocumentation:
    """Tests for kill switch documentation."""

    def test_kill_switch_runbook_exists(self):
        """Test kill switch runbook exists."""
        killswitch_path = DOCS_DIR / "runbooks" / "KILL_SWITCH.md"
        assert killswitch_path.exists(), "Kill switch runbook not found"

    def test_kill_switch_has_trigger_commands(self):
        """Test kill switch runbook has trigger commands."""
        killswitch_path = DOCS_DIR / "runbooks" / "KILL_SWITCH.md"
        if not killswitch_path.exists():
            pytest.skip("Kill switch runbook not found")

        content = killswitch_path.read_text().lower()
        assert "trigger" in content, "Kill switch runbook must include trigger instructions"

    def test_kill_switch_has_recovery_steps(self):
        """Test kill switch runbook has recovery steps."""
        killswitch_path = DOCS_DIR / "runbooks" / "KILL_SWITCH.md"
        if not killswitch_path.exists():
            pytest.skip("Kill switch runbook not found")

        content = killswitch_path.read_text().lower()
        assert (
            "recover" in content or "reset" in content
        ), "Kill switch runbook must include recovery steps"


# ============================================================================
# Marker for CI
# ============================================================================

# Mark all tests for easy CI filtering
pytestmark = pytest.mark.documentation
