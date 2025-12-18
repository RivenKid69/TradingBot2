# -*- coding: utf-8 -*-
"""
Tests for Legal/Marketing CCEA Alignment (WI-LEGAL-01).

Verifies legal documents correctly reflect CCEA architecture:
- No claims of credential storage in Cloud
- No claims of order execution by Cloud
- No brokerage or custody claims
- Proper CCEA boundary documentation
"""

import pytest
from pathlib import Path
import re


class TestTermsOfServiceCCEA:
    """Test Terms of Service CCEA compliance."""

    @pytest.fixture
    def tos_content(self):
        """Load ToS content."""
        tos_path = Path(__file__).parent.parent.parent.parent / "docs" / "legal" / "TERMS_OF_SERVICE.md"
        if not tos_path.exists():
            pytest.skip("TERMS_OF_SERVICE.md not found")
        return tos_path.read_text(encoding='utf-8')

    def test_tos_mentions_ccea(self, tos_content):
        """Test ToS mentions CCEA architecture."""
        assert "ccea" in tos_content.lower(), "ToS should mention CCEA architecture"

    def test_tos_cloud_no_credentials(self, tos_content):
        """Test ToS states Cloud doesn't store credentials."""
        content_lower = tos_content.lower()
        # Should explicitly state Cloud doesn't store credentials
        has_no_creds_statement = any([
            "never" in content_lower and "credential" in content_lower,
            "never" in content_lower and "api key" in content_lower,
            "cloud" in content_lower and "not store" in content_lower,
        ])
        assert has_no_creds_statement, "ToS should state Cloud doesn't store credentials"

    def test_tos_cloud_no_orders(self, tos_content):
        """Test ToS states Cloud doesn't execute orders."""
        content_lower = tos_content.lower()
        has_no_orders_statement = any([
            "never" in content_lower and "execut" in content_lower and "order" in content_lower,
            "cloud" in content_lower and "lifecycle" in content_lower,
        ])
        assert has_no_orders_statement, "ToS should state Cloud doesn't execute orders"

    def test_tos_mentions_agent(self, tos_content):
        """Test ToS mentions Agent for execution."""
        content_lower = tos_content.lower()
        assert "agent" in content_lower, "ToS should mention Agent for execution"


class TestPrivacyPolicyCCEA:
    """Test Privacy Policy CCEA compliance."""

    @pytest.fixture
    def pp_content(self):
        """Load Privacy Policy content."""
        pp_path = Path(__file__).parent.parent.parent.parent / "docs" / "legal" / "PRIVACY_POLICY.md"
        if not pp_path.exists():
            pytest.skip("PRIVACY_POLICY.md not found")
        return pp_path.read_text(encoding='utf-8')

    def test_pp_mentions_ccea(self, pp_content):
        """Test Privacy Policy mentions CCEA."""
        assert "ccea" in pp_content.lower(), "Privacy Policy should mention CCEA"

    def test_pp_data_zones(self, pp_content):
        """Test Privacy Policy describes data zones."""
        content_lower = pp_content.lower()
        # Should mention Cloud and Agent zones
        has_zones = "cloud" in content_lower and "agent" in content_lower
        assert has_zones, "Privacy Policy should describe Cloud and Agent data zones"

    def test_pp_credentials_local(self, pp_content):
        """Test Privacy Policy states credentials are local."""
        content_lower = pp_content.lower()
        # Should state credentials are in Agent, not Cloud
        has_local_creds = any([
            "local" in content_lower and "vault" in content_lower,
            "agent" in content_lower and "credential" in content_lower,
            "never" in content_lower and "store" in content_lower and "key" in content_lower,
        ])
        assert has_local_creds, "Privacy Policy should state credentials are local"

    def test_pp_telemetry_redaction(self, pp_content):
        """Test Privacy Policy mentions telemetry redaction."""
        content_lower = pp_content.lower()
        has_redaction = "redact" in content_lower
        assert has_redaction, "Privacy Policy should mention telemetry redaction"


class TestDPATemplateCCEA:
    """Test DPA Template CCEA compliance."""

    @pytest.fixture
    def dpa_content(self):
        """Load DPA content."""
        dpa_path = Path(__file__).parent.parent.parent.parent / "docs" / "legal" / "DPA_TEMPLATE.md"
        if not dpa_path.exists():
            pytest.skip("DPA_TEMPLATE.md not found")
        return dpa_path.read_text(encoding='utf-8')

    def test_dpa_ccea_note(self, dpa_content):
        """Test DPA has CCEA architecture note."""
        content_lower = dpa_content.lower()
        has_ccea_note = "ccea" in content_lower or "architecture" in content_lower
        assert has_ccea_note, "DPA should have CCEA architecture note"

    def test_dpa_no_credential_processing(self, dpa_content):
        """Test DPA doesn't claim credential processing."""
        content_lower = dpa_content.lower()
        # Should explicitly disclaim credential processing
        has_disclaimer = any([
            "never" in content_lower and "credential" in content_lower,
            "never" in content_lower and "api key" in content_lower,
            "agent" in content_lower and "local" in content_lower,
        ])
        assert has_disclaimer, "DPA should disclaim Cloud credential processing"


class TestAUPCCEA:
    """Test Acceptable Use Policy CCEA compliance."""

    @pytest.fixture
    def aup_content(self):
        """Load AUP content."""
        aup_path = Path(__file__).parent.parent.parent.parent / "docs" / "legal" / "AUP.md"
        if not aup_path.exists():
            pytest.skip("AUP.md not found")
        return aup_path.read_text(encoding='utf-8')

    def test_aup_mentions_ccea(self, aup_content):
        """Test AUP mentions CCEA architecture."""
        assert "ccea" in aup_content.lower(), "AUP should mention CCEA"

    def test_aup_cloud_compute_abuse(self, aup_content):
        """Test AUP addresses cloud compute abuse."""
        content_lower = aup_content.lower()
        has_abuse_section = any([
            "compute abuse" in content_lower,
            "resource abuse" in content_lower,
            "mining" in content_lower,  # crypto mining prohibition
        ])
        assert has_abuse_section, "AUP should address cloud compute abuse"

    def test_aup_protocol_violations(self, aup_content):
        """Test AUP prohibits CCEA protocol violations."""
        content_lower = aup_content.lower()
        has_protocol_section = any([
            "protocol" in content_lower and "violation" in content_lower,
            "boundary" in content_lower and "violation" in content_lower,
            "circumvent" in content_lower,
        ])
        assert has_protocol_section, "AUP should prohibit CCEA protocol violations"

    def test_aup_credential_upload_prohibited(self, aup_content):
        """Test AUP prohibits uploading credentials to Cloud."""
        content_lower = aup_content.lower()
        # Should prohibit transmitting credentials to Cloud
        has_cred_prohibition = any([
            "credential" in content_lower and "cloud" in content_lower,
            "api key" in content_lower and "prohibited" in content_lower,
            "transmit" in content_lower and "credential" in content_lower,
        ])
        assert has_cred_prohibition, "AUP should prohibit uploading credentials to Cloud"


class TestNoBrokerageClaimsInDocs:
    """Test no documents claim brokerage status."""

    @pytest.fixture
    def project_root(self):
        """Get project root path."""
        return Path(__file__).parent.parent.parent.parent

    def test_no_we_are_broker_claims(self, project_root):
        """Test no documents claim 'we are a broker'."""
        legal_dir = project_root / "docs" / "legal"
        if not legal_dir.exists():
            pytest.skip("docs/legal/ not found")

        broker_pattern = re.compile(r"we\s+are\s+(?:a\s+)?(?:registered\s+)?broker", re.IGNORECASE)

        for md_file in legal_dir.glob("*.md"):
            content = md_file.read_text(encoding='utf-8')
            matches = broker_pattern.findall(content)
            assert len(matches) == 0, f"{md_file.name} claims brokerage status: {matches}"

    def test_software_vendor_positioning(self, project_root):
        """Test documents position as software vendor."""
        canon = project_root / "docs" / "DOCUMENTATION_CANON_DESIGN.md"
        if not canon.exists():
            pytest.skip("DOCUMENTATION_CANON_DESIGN.md not found")

        content = canon.read_text(encoding="utf-8").lower()
        has_vendor_position = any(
            [
                "software/ict" in content,
                "software" in content and "ict" in content,
                "software" in content and "provider" in content,
                "technology" in content and "provider" in content,
            ]
        )
        assert has_vendor_position, "Documentation canon should position as software/ICT provider"

        # Also ensure we explicitly avoid advice/execution positioning in canon narratives.
        must_include = [
            "does not provide investment advice",
            "does not store credentials",
            "does not send live trading instructions",
        ]
        for phrase in must_include:
            assert phrase in content, f"Documentation canon should include: {phrase}"


class TestDORACompliance:
    """Test DORA documentation CCEA compliance."""

    @pytest.fixture
    def dora_content(self):
        """Load DORA content."""
        dora_path = Path(__file__).parent.parent.parent.parent / "docs" / "DORA_OPERATIONAL_RESILIENCE_PLAN.md"
        if not dora_path.exists():
            pytest.skip("DORA_OPERATIONAL_RESILIENCE_PLAN.md not found")
        return dora_path.read_text(encoding='utf-8')

    def test_dora_no_client_key_storage(self, dora_content):
        """Test DORA doc doesn't claim client key storage."""
        # Should not claim we store client API keys
        # Look for patterns that suggest we store client credentials
        problematic_patterns = [
            r"has_data_access:\s*true.*#.*(?:client|api)\s*key",  # old pattern
            r"we\s+store\s+(?:client|api)\s+key",
        ]

        for pattern in problematic_patterns:
            matches = re.findall(pattern, dora_content, re.IGNORECASE)
            # Filter out CCEA notes
            real_issues = [m for m in matches if "CCEA" not in m and "Agent" not in m]
            assert len(real_issues) == 0, f"DORA doc may claim client key storage: {real_issues}"

    def test_dora_brokers_are_client_relationships(self, dora_content):
        """Test DORA doc clarifies broker relationships are client's."""
        content_lower = dora_content.lower()
        # Should clarify Alpaca/Binance are client's relationships, not ours
        has_client_relationship = any([
            "client's" in content_lower and "agent" in content_lower,
            "client" in content_lower and "direct" in content_lower,
            "ccea" in content_lower,
        ])
        assert has_client_relationship, "DORA should clarify broker relationships are client's"
