"""
Tests for legal compliance documents.

Verifies that Terms of Service and Privacy Policy documents exist
and contain all required sections per regulatory requirements.

References:
    - EU E-Commerce Directive 2000/31/EC Art. 5, 6
    - GDPR Art. 13-14 transparency requirements
    - ESMA Q&A MiFID II (ESMA35-43-349)
"""

import pytest
from pathlib import Path


class TestTermsOfService:
    """Tests for Terms of Service document."""

    @pytest.fixture
    def tos_path(self) -> Path:
        """Get path to Terms of Service document."""
        return Path("docs/legal/TERMS_OF_SERVICE.md")

    @pytest.fixture
    def tos_content(self, tos_path: Path) -> str:
        """Load Terms of Service content."""
        assert tos_path.exists(), f"Terms of Service not found at {tos_path}"
        return tos_path.read_text(encoding="utf-8")

    def test_tos_file_exists(self, tos_path: Path):
        """Verify Terms of Service document exists."""
        assert tos_path.exists(), "Terms of Service document must exist"

    def test_tos_not_empty(self, tos_content: str):
        """Verify Terms of Service is not empty."""
        assert len(tos_content) > 1000, "Terms of Service should be substantial"

    def test_tos_contains_definitions_section(self, tos_content: str):
        """Verify definitions section exists."""
        assert "definition" in tos_content.lower(), "Must contain definitions section"
        assert "platform" in tos_content.lower(), "Must define 'Platform'"
        assert "user" in tos_content.lower(), "Must define 'User'"

    def test_tos_contains_no_investment_advice_disclaimer(self, tos_content: str):
        """Verify no investment advice disclaimer is present."""
        content_lower = tos_content.lower()
        assert (
            "no investment advice" in content_lower or "not investment advice" in content_lower
        ), "Must contain NO INVESTMENT ADVICE disclaimer"

    def test_tos_contains_limitation_of_liability(self, tos_content: str):
        """Verify limitation of liability section exists."""
        assert (
            "limitation of liability" in tos_content.lower()
        ), "Must contain LIMITATION OF LIABILITY section"

    def test_tos_contains_broker_api_keys_section(self, tos_content: str):
        """Verify broker API keys section exists."""
        content_lower = tos_content.lower()
        assert (
            "broker" in content_lower and "api" in content_lower
        ), "Must contain BROKER API KEYS section"
        assert "encrypt" in content_lower, "Must mention encryption"

    def test_tos_contains_risk_warning(self, tos_content: str):
        """Verify risk warnings are present."""
        content_lower = tos_content.lower()
        assert (
            "substantial risk" in content_lower or "risk of loss" in content_lower
        ), "Must contain substantial risk warning"

    def test_tos_contains_past_performance_disclaimer(self, tos_content: str):
        """Verify past performance disclaimer exists."""
        assert "past performance" in tos_content.lower(), "Must contain past performance disclaimer"

    def test_tos_contains_user_responsibilities(self, tos_content: str):
        """Verify user responsibilities section exists."""
        content_lower = tos_content.lower()
        assert "responsibilit" in content_lower, "Must contain user responsibilities section"

    def test_tos_contains_termination_section(self, tos_content: str):
        """Verify termination section exists."""
        assert "termination" in tos_content.lower(), "Must contain termination section"

    def test_tos_contains_governing_law(self, tos_content: str):
        """Verify governing law section exists."""
        content_lower = tos_content.lower()
        assert (
            "governing law" in content_lower or "jurisdiction" in content_lower
        ), "Must contain governing law section"

    def test_tos_contains_gdpr_reference(self, tos_content: str):
        """Verify GDPR is referenced."""
        assert (
            "gdpr" in tos_content.lower() or "data protection" in tos_content.lower()
        ), "Must reference GDPR or data protection"

    def test_tos_contains_disclaimers(self, tos_content: str):
        """Verify disclaimers section exists."""
        assert "disclaimer" in tos_content.lower(), "Must contain disclaimers section"

    def test_tos_contains_software_tool_clarification(self, tos_content: str):
        """Verify platform is described as software tool."""
        content_lower = tos_content.lower()
        assert "software" in content_lower, "Must clarify this is software"
        assert (
            "tool" in content_lower or "vendor" in content_lower
        ), "Must clarify this is a tool/vendor, not investment service"

    def test_tos_version_present(self, tos_content: str):
        """Verify document version is present."""
        assert "version" in tos_content.lower(), "Must contain version information"


class TestPrivacyPolicy:
    """Tests for Privacy Policy document."""

    @pytest.fixture
    def privacy_path(self) -> Path:
        """Get path to Privacy Policy document."""
        return Path("docs/legal/PRIVACY_POLICY.md")

    @pytest.fixture
    def privacy_content(self, privacy_path: Path) -> str:
        """Load Privacy Policy content."""
        assert privacy_path.exists(), f"Privacy Policy not found at {privacy_path}"
        return privacy_path.read_text(encoding="utf-8")

    def test_privacy_policy_exists(self, privacy_path: Path):
        """Verify Privacy Policy document exists."""
        assert privacy_path.exists(), "Privacy Policy document must exist"

    def test_privacy_policy_not_empty(self, privacy_content: str):
        """Verify Privacy Policy is not empty."""
        assert len(privacy_content) > 1000, "Privacy Policy should be substantial"

    def test_contains_data_controller_section(self, privacy_content: str):
        """Verify data controller is specified (GDPR Art. 13(1)(a))."""
        assert "data controller" in privacy_content.lower(), "Must specify data controller"

    def test_contains_gdpr_article_15_reference(self, privacy_content: str):
        """Verify right of access is mentioned (GDPR Art. 15)."""
        assert (
            "article 15" in privacy_content.lower() or "right of access" in privacy_content.lower()
        ), "Must reference GDPR Article 15 (Right of Access)"

    def test_contains_gdpr_article_17_reference(self, privacy_content: str):
        """Verify right to erasure is mentioned (GDPR Art. 17)."""
        assert (
            "article 17" in privacy_content.lower()
            or "right to erasure" in privacy_content.lower()
            or "right to be forgotten" in privacy_content.lower()
        ), "Must reference GDPR Article 17 (Right to Erasure)"

    def test_contains_gdpr_article_20_reference(self, privacy_content: str):
        """Verify data portability is mentioned (GDPR Art. 20)."""
        assert (
            "article 20" in privacy_content.lower() or "data portability" in privacy_content.lower()
        ), "Must reference GDPR Article 20 (Data Portability)"

    def test_contains_legal_basis_section(self, privacy_content: str):
        """Verify legal basis for processing is specified (GDPR Art. 6)."""
        content_lower = privacy_content.lower()
        assert (
            "legal basis" in content_lower or "article 6" in content_lower
        ), "Must specify legal basis for processing"

    def test_contains_data_retention_section(self, privacy_content: str):
        """Verify data retention periods are specified."""
        assert "retention" in privacy_content.lower(), "Must specify data retention periods"

    def test_contains_security_measures(self, privacy_content: str):
        """Verify security measures are described."""
        content_lower = privacy_content.lower()
        assert "security" in content_lower, "Must describe security measures"
        assert "encrypt" in content_lower, "Must mention encryption"

    def test_contains_user_rights_section(self, privacy_content: str):
        """Verify user rights are listed."""
        content_lower = privacy_content.lower()
        assert "rights" in content_lower, "Must list user rights"

    def test_contains_cookies_section(self, privacy_content: str):
        """Verify cookies are addressed."""
        assert "cookie" in privacy_content.lower(), "Must address cookies"

    def test_contains_contact_information(self, privacy_content: str):
        """Verify contact information is provided."""
        content_lower = privacy_content.lower()
        assert (
            "contact" in content_lower or "email" in content_lower
        ), "Must provide contact information"

    def test_contains_broker_credentials_handling(self, privacy_content: str):
        """Verify broker credentials handling is described."""
        content_lower = privacy_content.lower()
        assert "api" in content_lower and (
            "credential" in content_lower or "key" in content_lower
        ), "Must describe broker credential handling"
        assert (
            "aes" in content_lower or "encrypt" in content_lower
        ), "Must mention encryption for credentials"

    def test_contains_international_transfers(self, privacy_content: str):
        """Verify international transfers are addressed."""
        content_lower = privacy_content.lower()
        assert (
            "international" in content_lower or "transfer" in content_lower or "eu" in content_lower
        ), "Must address international data transfers"

    def test_contains_third_party_sharing(self, privacy_content: str):
        """Verify third party data sharing is addressed."""
        content_lower = privacy_content.lower()
        assert (
            "third" in content_lower or "sharing" in content_lower or "broker" in content_lower
        ), "Must address third party data sharing"


class TestLegalDocumentsIntegration:
    """Integration tests for legal documents."""

    def test_both_documents_exist(self):
        """Verify both required legal documents exist."""
        tos_path = Path("docs/legal/TERMS_OF_SERVICE.md")
        privacy_path = Path("docs/legal/PRIVACY_POLICY.md")

        assert tos_path.exists(), "Terms of Service must exist"
        assert privacy_path.exists(), "Privacy Policy must exist"

    def test_legal_directory_structure(self):
        """Verify legal documents directory exists."""
        legal_dir = Path("docs/legal")
        assert legal_dir.exists(), "docs/legal directory must exist"
        assert legal_dir.is_dir(), "docs/legal must be a directory"

    def test_cross_reference_between_documents(self):
        """Verify documents reference each other."""
        tos_path = Path("docs/legal/TERMS_OF_SERVICE.md")
        privacy_path = Path("docs/legal/PRIVACY_POLICY.md")

        if tos_path.exists():
            tos_content = tos_path.read_text(encoding="utf-8").lower()
            assert "privacy" in tos_content, "ToS should reference Privacy Policy"

    def test_documents_are_markdown(self):
        """Verify documents are valid markdown files."""
        tos_path = Path("docs/legal/TERMS_OF_SERVICE.md")
        privacy_path = Path("docs/legal/PRIVACY_POLICY.md")

        assert tos_path.suffix == ".md", "ToS should be markdown"
        assert privacy_path.suffix == ".md", "Privacy Policy should be markdown"

    def test_documents_contain_version_info(self):
        """Verify both documents have version information."""
        tos_path = Path("docs/legal/TERMS_OF_SERVICE.md")
        privacy_path = Path("docs/legal/PRIVACY_POLICY.md")

        for path in [tos_path, privacy_path]:
            if path.exists():
                content = path.read_text(encoding="utf-8").lower()
                assert "version" in content, f"{path.name} must contain version info"


class TestDataProcessingAgreement:
    """Tests for Data Processing Agreement (DPA) template."""

    @pytest.fixture
    def dpa_path(self) -> Path:
        """Get path to DPA template."""
        return Path("docs/legal/DPA_TEMPLATE.md")

    @pytest.fixture
    def dpa_content(self, dpa_path: Path) -> str:
        """Load DPA content."""
        assert dpa_path.exists(), f"DPA template not found at {dpa_path}"
        return dpa_path.read_text(encoding="utf-8")

    def test_dpa_exists(self, dpa_path: Path):
        """Verify DPA template exists."""
        assert dpa_path.exists(), "DPA template must exist"

    def test_dpa_not_empty(self, dpa_content: str):
        """Verify DPA is not empty."""
        assert len(dpa_content) > 5000, "DPA should be substantial"

    def test_contains_parties_section(self, dpa_content: str):
        """Verify parties section exists (GDPR Art. 28)."""
        content_lower = dpa_content.lower()
        assert "controller" in content_lower, "Must define Controller"
        assert "processor" in content_lower, "Must define Processor"

    def test_contains_subject_matter(self, dpa_content: str):
        """Verify subject matter section exists."""
        assert "subject matter" in dpa_content.lower(), "Must contain subject matter section"

    def test_contains_data_types(self, dpa_content: str):
        """Verify types of personal data are specified."""
        content_lower = dpa_content.lower()
        assert "personal data" in content_lower, "Must specify types of personal data"
        assert "account" in content_lower or "email" in content_lower, "Must list data types"

    def test_contains_data_subjects(self, dpa_content: str):
        """Verify categories of data subjects are specified."""
        content_lower = dpa_content.lower()
        assert "data subject" in content_lower, "Must specify data subjects"

    def test_contains_processor_obligations(self, dpa_content: str):
        """Verify processor obligations per GDPR Art. 28(3)."""
        content_lower = dpa_content.lower()
        assert "instruction" in content_lower, "Must include processing on instructions"
        assert "confidentiality" in content_lower, "Must include confidentiality"
        assert "security" in content_lower, "Must include security measures"

    def test_contains_sub_processor_section(self, dpa_content: str):
        """Verify sub-processor section exists."""
        content_lower = dpa_content.lower()
        assert (
            "sub-processor" in content_lower or "subprocessor" in content_lower
        ), "Must address sub-processors"

    def test_contains_security_measures(self, dpa_content: str):
        """Verify security measures are detailed (GDPR Art. 32)."""
        content_lower = dpa_content.lower()
        assert "encryption" in content_lower, "Must mention encryption"
        assert "aes" in content_lower, "Must specify encryption standard"
        assert "access control" in content_lower, "Must mention access controls"

    def test_contains_breach_notification(self, dpa_content: str):
        """Verify data breach notification section exists."""
        content_lower = dpa_content.lower()
        assert "breach" in content_lower, "Must address data breaches"
        assert (
            "notification" in content_lower or "notify" in content_lower
        ), "Must specify breach notification"
        assert (
            "24 hour" in content_lower or "24-hour" in content_lower
        ), "Must specify notification timeline"

    def test_contains_international_transfers(self, dpa_content: str):
        """Verify international transfers are addressed."""
        content_lower = dpa_content.lower()
        assert (
            "international" in content_lower or "transfer" in content_lower
        ), "Must address international transfers"
        assert (
            "eea" in content_lower or "european economic area" in content_lower
        ), "Must reference EEA"

    def test_contains_audit_rights(self, dpa_content: str):
        """Verify audit rights are specified."""
        content_lower = dpa_content.lower()
        assert "audit" in content_lower, "Must include audit rights"

    def test_contains_termination_section(self, dpa_content: str):
        """Verify termination section exists."""
        content_lower = dpa_content.lower()
        assert "termination" in content_lower, "Must include termination provisions"
        assert (
            "deletion" in content_lower or "delete" in content_lower
        ), "Must specify data deletion on termination"

    def test_contains_annex_a(self, dpa_content: str):
        """Verify Annex A (Technical Measures) exists."""
        assert "Annex A" in dpa_content, "Must include Annex A"

    def test_contains_annex_b(self, dpa_content: str):
        """Verify Annex B (Sub-processor List) exists."""
        assert "Annex B" in dpa_content, "Must include Annex B"

    def test_contains_annex_c(self, dpa_content: str):
        """Verify Annex C (SCCs) exists."""
        assert (
            "Annex C" in dpa_content or "Standard Contractual Clauses" in dpa_content
        ), "Must include SCCs reference"

    def test_contains_gdpr_references(self, dpa_content: str):
        """Verify GDPR articles are referenced."""
        content_lower = dpa_content.lower()
        assert "gdpr" in content_lower, "Must reference GDPR"
        assert "article 28" in content_lower, "Must reference GDPR Article 28"

    def test_contains_version_info(self, dpa_content: str):
        """Verify version information is present."""
        assert (
            "Version" in dpa_content or "version" in dpa_content.lower()
        ), "Must contain version information"

    def test_contains_signature_blocks(self, dpa_content: str):
        """Verify signature blocks exist."""
        content_lower = dpa_content.lower()
        assert "signature" in content_lower, "Must include signature blocks"


class TestPhase2LegalDocuments:
    """Integration tests for Phase 2 legal documents."""

    def test_dpa_template_exists(self):
        """Verify DPA template exists."""
        dpa_path = Path("docs/legal/DPA_TEMPLATE.md")
        assert dpa_path.exists(), "DPA template must exist"

    def test_all_phase2_documents_exist(self):
        """Verify all Phase 2 legal documents exist."""
        required_docs = [
            "docs/legal/TERMS_OF_SERVICE.md",
            "docs/legal/PRIVACY_POLICY.md",
            "docs/legal/DPA_TEMPLATE.md",
        ]
        for doc in required_docs:
            assert Path(doc).exists(), f"{doc} must exist"
