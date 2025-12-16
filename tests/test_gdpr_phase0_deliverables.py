"""
Tests for GDPR Phase 0 Deliverables - Scope, Data Map, and Role Model.

This test module validates the completeness and correctness of Phase 0 deliverables
as defined in docs/compliance/GDPR_CCEA_IMPLEMENTATION_PLAN.md.

Phase 0 DoD (Definition of Done):
1. A RoPA-lite table exists with columns: system, data category, purpose,
   lawful basis, retention, residency, access roles, subprocessors.
2. A Cloud<->Agent data flow diagram exists and labels Cloud telemetry levels
   (AGGREGATED/DETAILED_NON_SENSITIVE/RAW_ORDER_EVENTS), and documents RAW gating
   (enterprise-only, explicit opt-in) and "telemetry stays local" option.
3. Every listed data store/log stream has: owner, retention, lawful basis,
   and residency=EU (no blanks).

References:
    - GDPR Regulation (EU) 2016/679 Article 30 (Records of Processing Activities)
    - docs/compliance/GDPR_CCEA_IMPLEMENTATION_PLAN.md (Phase 0)
    - docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt
"""

import os
import re
import pytest
from pathlib import Path
from typing import List, Set, Dict, Any


# Path to compliance documents
COMPLIANCE_DIR = Path(__file__).parent.parent / "docs" / "compliance"
GDPR_SCOPE_MEMO_PATH = COMPLIANCE_DIR / "GDPR_RISK_SCOPE_MEMO.md"
GDPR_IMPLEMENTATION_PLAN_PATH = COMPLIANCE_DIR / "GDPR_CCEA_IMPLEMENTATION_PLAN.md"


class TestPhase0DocumentExistence:
    """Test that all required Phase 0 documents exist."""

    def test_gdpr_risk_scope_memo_exists(self):
        """GDPR_RISK_SCOPE_MEMO.md must exist as per Phase 0 deliverables."""
        assert GDPR_SCOPE_MEMO_PATH.exists(), (
            f"GDPR_RISK_SCOPE_MEMO.md not found at {GDPR_SCOPE_MEMO_PATH}. "
            "This is a required Phase 0 deliverable."
        )

    def test_gdpr_implementation_plan_exists(self):
        """GDPR_CCEA_IMPLEMENTATION_PLAN.md must exist."""
        assert GDPR_IMPLEMENTATION_PLAN_PATH.exists(), (
            f"GDPR_CCEA_IMPLEMENTATION_PLAN.md not found at {GDPR_IMPLEMENTATION_PLAN_PATH}"
        )

    def test_gdpr_scope_memo_not_empty(self):
        """GDPR_RISK_SCOPE_MEMO.md must have substantial content."""
        content = GDPR_SCOPE_MEMO_PATH.read_text()
        # Should be at least 10KB (comprehensive document)
        assert len(content) > 10000, (
            f"GDPR_RISK_SCOPE_MEMO.md is too short ({len(content)} bytes). "
            "Phase 0 requires a comprehensive data map document."
        )


class TestGDPRScopeMemoStructure:
    """Test the structure and completeness of GDPR_RISK_SCOPE_MEMO.md."""

    @pytest.fixture
    def memo_content(self) -> str:
        """Load the memo content."""
        return GDPR_SCOPE_MEMO_PATH.read_text()

    def test_has_executive_summary(self, memo_content: str):
        """Document must have an executive summary section."""
        assert "Executive Summary" in memo_content or "executive summary" in memo_content.lower(), (
            "Missing Executive Summary section"
        )

    def test_has_platform_zones_section(self, memo_content: str):
        """Document must describe Cloud and Agent zones."""
        assert "CLOUD ZONE" in memo_content or "Cloud Zone" in memo_content, (
            "Missing Cloud Zone description"
        )
        assert "AGENT ZONE" in memo_content or "Agent Zone" in memo_content, (
            "Missing Agent Zone description"
        )

    def test_has_controller_processor_section(self, memo_content: str):
        """Document must define Controller vs Processor roles."""
        assert "Controller" in memo_content and "Processor" in memo_content, (
            "Missing Controller vs Processor role definitions"
        )

    def test_has_data_flow_diagram(self, memo_content: str):
        """Document must contain a data flow diagram (ASCII art or reference)."""
        # Check for ASCII diagram indicators
        has_diagram = any([
            "Cloud↔Agent" in memo_content,
            "Cloud→Agent" in memo_content,
            "Agent→Cloud" in memo_content,
            "┌" in memo_content,  # ASCII box drawing
            "├" in memo_content,
            "│" in memo_content,
            "Data Flow" in memo_content,
        ])
        assert has_diagram, "Missing Cloud<->Agent data flow diagram"

    def test_has_ropa_table(self, memo_content: str):
        """Document must contain a RoPA-lite table."""
        assert "RoPA" in memo_content or "Records of Processing" in memo_content, (
            "Missing RoPA (Records of Processing Activities) section"
        )

    def test_has_subprocessor_section(self, memo_content: str):
        """Document must list subprocessors."""
        assert "Subprocessor" in memo_content or "subprocessor" in memo_content, (
            "Missing Subprocessor register section"
        )


class TestRoPATableCompleteness:
    """Test that the RoPA-lite table has all required columns."""

    @pytest.fixture
    def memo_content(self) -> str:
        """Load the memo content."""
        return GDPR_SCOPE_MEMO_PATH.read_text()

    def test_ropa_has_system_column(self, memo_content: str):
        """RoPA table must have 'System' column."""
        # Check for table header with System
        assert re.search(r"\|\s*System\s*\|", memo_content, re.IGNORECASE), (
            "RoPA table missing 'System' column"
        )

    def test_ropa_has_data_category_column(self, memo_content: str):
        """RoPA table must have 'Data Category' column."""
        assert re.search(r"\|\s*Data\s*Category\s*\|", memo_content, re.IGNORECASE), (
            "RoPA table missing 'Data Category' column"
        )

    def test_ropa_has_purpose_column(self, memo_content: str):
        """RoPA table must have 'Purpose' column."""
        assert re.search(r"\|\s*Purpose\s*\|", memo_content, re.IGNORECASE), (
            "RoPA table missing 'Purpose' column"
        )

    def test_ropa_has_lawful_basis_column(self, memo_content: str):
        """RoPA table must have 'Lawful Basis' column."""
        assert re.search(r"\|\s*Lawful\s*Basis\s*\|", memo_content, re.IGNORECASE), (
            "RoPA table missing 'Lawful Basis' column"
        )

    def test_ropa_has_retention_column(self, memo_content: str):
        """RoPA table must have 'Retention' column."""
        assert re.search(r"\|\s*Retention\s*\|", memo_content, re.IGNORECASE), (
            "RoPA table missing 'Retention' column"
        )

    def test_ropa_has_residency_column(self, memo_content: str):
        """RoPA table must have 'Residency' column."""
        assert re.search(r"\|\s*Residency\s*\|", memo_content, re.IGNORECASE), (
            "RoPA table missing 'Residency' column"
        )

    def test_ropa_has_access_roles_column(self, memo_content: str):
        """RoPA table must have 'Access Roles' column."""
        assert re.search(r"\|\s*Access\s*Roles\s*\|", memo_content, re.IGNORECASE), (
            "RoPA table missing 'Access Roles' column"
        )

    def test_ropa_has_subprocessors_column(self, memo_content: str):
        """RoPA table must have 'Subprocessors' column."""
        assert re.search(r"\|\s*Subprocessors\s*\|", memo_content, re.IGNORECASE), (
            "RoPA table missing 'Subprocessors' column"
        )

    def test_ropa_has_owner_column(self, memo_content: str):
        """RoPA table must have 'Owner' column per DoD requirement."""
        assert re.search(r"\|\s*Owner\s*\|", memo_content, re.IGNORECASE), (
            "RoPA table missing 'Owner' column (required by Phase 0 DoD)"
        )


class TestTelemetryLevelsDocumentation:
    """Test that all telemetry sensitivity levels are documented."""

    @pytest.fixture
    def memo_content(self) -> str:
        """Load the memo content."""
        return GDPR_SCOPE_MEMO_PATH.read_text()

    def test_aggregated_level_documented(self, memo_content: str):
        """AGGREGATED telemetry level must be documented."""
        assert "AGGREGATED" in memo_content, (
            "Missing documentation for AGGREGATED telemetry level"
        )

    def test_detailed_non_sensitive_level_documented(self, memo_content: str):
        """DETAILED_NON_SENSITIVE telemetry level must be documented."""
        assert "DETAILED_NON_SENSITIVE" in memo_content, (
            "Missing documentation for DETAILED_NON_SENSITIVE telemetry level"
        )

    def test_raw_order_events_level_documented(self, memo_content: str):
        """RAW_ORDER_EVENTS telemetry level must be documented."""
        assert "RAW_ORDER_EVENTS" in memo_content, (
            "Missing documentation for RAW_ORDER_EVENTS telemetry level"
        )

    def test_aggregated_is_default(self, memo_content: str):
        """AGGREGATED must be marked as default level."""
        # Check for indicators that AGGREGATED is default
        has_default_indicator = any([
            "AGGREGATED (Default" in memo_content,
            "AGGREGATED (default" in memo_content,
            "default: AGGREGATED" in memo_content.lower(),
            "Default for all tiers" in memo_content,
        ])
        assert has_default_indicator, (
            "AGGREGATED telemetry level must be marked as default"
        )

    def test_raw_enterprise_only(self, memo_content: str):
        """RAW_ORDER_EVENTS must be marked as enterprise-only."""
        # Check for enterprise-only indicators near RAW_ORDER_EVENTS
        has_enterprise_indicator = any([
            "RAW_ORDER_EVENTS" in memo_content and "enterprise" in memo_content.lower(),
            "Enterprise-only" in memo_content,
            "enterprise-only" in memo_content,
        ])
        assert has_enterprise_indicator, (
            "RAW_ORDER_EVENTS must be marked as enterprise-only"
        )

    def test_raw_explicit_opt_in_required(self, memo_content: str):
        """RAW_ORDER_EVENTS must require explicit opt-in."""
        has_opt_in_indicator = any([
            "explicit opt-in" in memo_content.lower(),
            "explicit per-workspace opt-in" in memo_content,
        ])
        assert has_opt_in_indicator, (
            "RAW_ORDER_EVENTS must require explicit opt-in (GDPR consent requirement)"
        )

    def test_telemetry_stays_local_option(self, memo_content: str):
        """Enterprise 'telemetry stays local' option must be documented."""
        has_local_option = any([
            "telemetry stays local" in memo_content.lower(),
            "local-only mode" in memo_content.lower(),
            "local export" in memo_content.lower(),
        ])
        assert has_local_option, (
            "Missing documentation for enterprise 'telemetry stays local' option"
        )


class TestEUDataResidency:
    """Test that EU-only data residency is properly documented."""

    @pytest.fixture
    def memo_content(self) -> str:
        """Load the memo content."""
        return GDPR_SCOPE_MEMO_PATH.read_text()

    def test_eu_only_residency_stated(self, memo_content: str):
        """EU-only data residency must be explicitly stated."""
        has_eu_only = any([
            "EU-only" in memo_content,
            "eu-only" in memo_content,
            "EU (eu-central-1)" in memo_content,
            "eu-central-1" in memo_content,
            "eu-west-1" in memo_content,
        ])
        assert has_eu_only, "EU-only data residency must be explicitly stated"

    def test_all_residency_values_are_eu(self, memo_content: str):
        """All residency values in RoPA table must be EU regions."""
        # Extract residency values from table rows
        # Look for | EU ... | or | eu- patterns
        residency_pattern = r"\|\s*EU\s*\(.*?\)\s*\|"
        eu_residencies = re.findall(residency_pattern, memo_content)

        # Also check for any non-EU residency which would be a violation
        non_eu_pattern = r"\|\s*(us-|ap-|sa-|me-|af-)"
        non_eu_matches = re.findall(non_eu_pattern, memo_content)

        assert len(non_eu_matches) == 0, (
            f"Found non-EU residency values in document: {non_eu_matches}. "
            "All data must reside in EU per GDPR_CCEA_IMPLEMENTATION_PLAN.md"
        )

    def test_subprocessors_eu_only(self, memo_content: str):
        """All subprocessors must be in EU regions."""
        # Check that subprocessor section mentions EU regions
        subprocessor_section = memo_content[memo_content.find("Subprocessor"):] if "Subprocessor" in memo_content else ""

        # Should mention EU regions
        has_eu_regions = any([
            "eu-central-1" in subprocessor_section.lower(),
            "eu-west-1" in subprocessor_section.lower(),
            "EU (Germany)" in subprocessor_section,
            "EU (Ireland)" in subprocessor_section,
            "(EU)" in subprocessor_section,
        ])
        assert has_eu_regions, "Subprocessors must be documented with EU regions"


class TestDataStoreCompleteness:
    """Test that every data store has required attributes (no blanks)."""

    @pytest.fixture
    def memo_content(self) -> str:
        """Load the memo content."""
        return GDPR_SCOPE_MEMO_PATH.read_text()

    def _extract_ropa_rows(self, content: str) -> List[str]:
        """Extract RoPA table rows from content."""
        # Find markdown table rows (lines starting with |)
        lines = content.split('\n')
        table_rows = []
        in_table = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('|') and '---' not in stripped:
                # Skip header separator rows
                if not re.match(r'\|\s*[-:]+\s*\|', stripped):
                    table_rows.append(stripped)
                    in_table = True
            elif in_table and not stripped.startswith('|'):
                # End of table
                in_table = False

        return table_rows

    def test_no_empty_cells_in_ropa(self, memo_content: str):
        """RoPA table must not have empty cells for required fields."""
        rows = self._extract_ropa_rows(memo_content)

        # Skip header row
        data_rows = [r for r in rows if not any(h in r.lower() for h in ['system', 'data category', 'purpose'])]

        empty_cell_pattern = r'\|\s*\|'
        rows_with_empty = []

        for row in data_rows:
            if re.search(empty_cell_pattern, row):
                rows_with_empty.append(row[:100])  # First 100 chars for debugging

        assert len(rows_with_empty) == 0, (
            f"Found {len(rows_with_empty)} rows with empty cells in RoPA table. "
            f"Examples: {rows_with_empty[:3]}"
        )

    def test_every_store_has_retention(self, memo_content: str):
        """Every data store must have a retention period defined."""
        # Check for retention periods in various formats
        retention_patterns = [
            r"Account lifetime",
            r"\d+ days",
            r"\d+ hours",
            r"\d+ years",
            r"Customer-defined",
            r"indefinite",
        ]

        # Count how many data stores we have (approximate by counting PostgreSQL entries)
        postgres_entries = len(re.findall(r"PostgreSQL", memo_content))

        # Count retention mentions
        retention_mentions = sum(
            len(re.findall(pattern, memo_content))
            for pattern in retention_patterns
        )

        # Should have at least as many retention mentions as data stores
        assert retention_mentions >= postgres_entries, (
            f"Not all data stores have retention periods. "
            f"Found {postgres_entries} stores but only {retention_mentions} retention mentions."
        )


class TestControllerProcessorRoles:
    """Test that Controller vs Processor roles are properly defined."""

    @pytest.fixture
    def memo_content(self) -> str:
        """Load the memo content."""
        return GDPR_SCOPE_MEMO_PATH.read_text()

    def test_has_role_determination_framework(self, memo_content: str):
        """Document must explain how roles are determined."""
        has_framework = any([
            "Role Determination" in memo_content,
            "Art. 4(7)" in memo_content,  # Controller definition
            "Art. 4(8)" in memo_content,  # Processor definition
        ])
        assert has_framework, (
            "Missing Controller/Processor role determination framework"
        )

    def test_user_data_role_defined(self, memo_content: str):
        """User account data role (Controller/Processor) must be defined."""
        has_user_role = any([
            "User Account Data" in memo_content and "Controller" in memo_content,
            "User Identity" in memo_content,
        ])
        assert has_user_role, "User account data role not defined"

    def test_telemetry_data_role_defined(self, memo_content: str):
        """Telemetry data role must be defined."""
        has_telemetry_role = any([
            "Telemetry" in memo_content and ("Processor" in memo_content or "Controller" in memo_content),
        ])
        assert has_telemetry_role, "Telemetry data role not defined"

    def test_strategy_data_role_defined(self, memo_content: str):
        """Strategy/model data role must be defined."""
        has_strategy_role = any([
            "Strategy" in memo_content and "Processor" in memo_content,
            "Strategy Assets" in memo_content,
        ])
        assert has_strategy_role, "Strategy data role not defined"

    def test_audit_log_role_defined(self, memo_content: str):
        """Audit log data role must be defined."""
        has_audit_role = any([
            "Audit" in memo_content and "Controller" in memo_content,
        ])
        assert has_audit_role, "Audit log data role not defined"


class TestArchitecturalInvariants:
    """Test that CCEA architectural invariants are documented."""

    @pytest.fixture
    def memo_content(self) -> str:
        """Load the memo content."""
        return GDPR_SCOPE_MEMO_PATH.read_text()

    def test_cloud_never_receives_secrets(self, memo_content: str):
        """Document must state Cloud never receives secrets."""
        has_secrets_invariant = any([
            "Cloud never receives" in memo_content and "secret" in memo_content.lower(),
            "Cloud NEVER receives" in memo_content,
            "never receives" in memo_content.lower() and "API key" in memo_content,
        ])
        assert has_secrets_invariant, (
            "Missing invariant: Cloud never receives broker credentials/API keys"
        )

    def test_no_order_payloads_in_commands(self, memo_content: str):
        """Document must state no order-like payloads in Cloud->Agent commands."""
        has_no_order_invariant = any([
            "order-like payloads" in memo_content.lower(),
            "side/qty/price" in memo_content.lower() or "side, quantity, price" in memo_content.lower(),
            "FORBIDDEN COMMAND" in memo_content,
        ])
        assert has_no_order_invariant, (
            "Missing invariant: No order-like payloads in Cloud->Agent commands"
        )

    def test_mandatory_redaction(self, memo_content: str):
        """Document must state redaction is mandatory and cannot be disabled."""
        has_redaction_invariant = any([
            "mandatory redaction" in memo_content.lower(),
            "cannot be disabled" in memo_content.lower(),
            "MANDATORY REDACTION" in memo_content,
        ])
        assert has_redaction_invariant, (
            "Missing invariant: Redaction is mandatory and cannot be disabled"
        )


class TestCommandProtocolDocumentation:
    """Test that the command protocol is properly documented."""

    @pytest.fixture
    def memo_content(self) -> str:
        """Load the memo content."""
        return GDPR_SCOPE_MEMO_PATH.read_text()

    def test_allowed_commands_listed(self, memo_content: str):
        """Allowed command types must be listed."""
        expected_commands = [
            "REQUEST_START_RUN",
            "REQUEST_STOP_RUN",
            "REQUEST_PAUSE_RUN",
            "REQUEST_UPGRADE_ARTIFACT",
        ]
        for cmd in expected_commands:
            assert cmd in memo_content, f"Missing allowed command type: {cmd}"

    def test_forbidden_commands_listed(self, memo_content: str):
        """Forbidden command types must be listed."""
        expected_forbidden = [
            "PLACE_ORDER",
            "SUBMIT_ORDER",
            "EXECUTE_SIGNAL",
        ]
        for cmd in expected_forbidden:
            assert cmd in memo_content, f"Missing forbidden command type: {cmd}"


class TestDSARBoundaries:
    """Test that DSAR (Data Subject Access Request) boundaries are documented."""

    @pytest.fixture
    def memo_content(self) -> str:
        """Load the memo content."""
        return GDPR_SCOPE_MEMO_PATH.read_text()

    def test_dsar_scope_documented(self, memo_content: str):
        """DSAR scope must be documented."""
        has_dsar_scope = any([
            "DSAR" in memo_content,
            "Data Subject" in memo_content,
            "Art. 15" in memo_content or "Art. 17" in memo_content,
        ])
        assert has_dsar_scope, "DSAR scope not documented"

    def test_cloud_vs_agent_dsar_boundary(self, memo_content: str):
        """Cloud vs Agent data boundary for DSAR must be clear."""
        has_boundary = any([
            "Cloud-controlled" in memo_content,
            "Agent-controlled" in memo_content,
            "OUT OF SCOPE" in memo_content,
            "IN SCOPE" in memo_content,
        ])
        assert has_boundary, (
            "Cloud vs Agent data boundary for DSAR not clearly defined"
        )


class TestGDPRArticleReferences:
    """Test that relevant GDPR articles are referenced."""

    @pytest.fixture
    def memo_content(self) -> str:
        """Load the memo content."""
        return GDPR_SCOPE_MEMO_PATH.read_text()

    def test_references_art_5_principles(self, memo_content: str):
        """Article 5 (Data Protection Principles) must be referenced."""
        assert "Art. 5" in memo_content or "Article 5" in memo_content, (
            "Missing reference to GDPR Article 5 (Principles)"
        )

    def test_references_art_6_lawful_basis(self, memo_content: str):
        """Article 6 (Lawful Basis) must be referenced."""
        assert "Art. 6" in memo_content or "Article 6" in memo_content, (
            "Missing reference to GDPR Article 6 (Lawful Basis)"
        )

    def test_references_art_25_privacy_by_design(self, memo_content: str):
        """Article 25 (Privacy by Design) must be referenced."""
        has_art_25 = any([
            "Art. 25" in memo_content,
            "Article 25" in memo_content,
            "Privacy by Design" in memo_content,
        ])
        assert has_art_25, (
            "Missing reference to GDPR Article 25 (Privacy by Design)"
        )

    def test_references_art_30_ropa(self, memo_content: str):
        """Article 30 (Records of Processing Activities) must be referenced."""
        has_art_30 = any([
            "Art. 30" in memo_content,
            "Article 30" in memo_content,
            "Records of Processing" in memo_content,
        ])
        assert has_art_30, (
            "Missing reference to GDPR Article 30 (Records of Processing Activities)"
        )

    def test_references_art_32_security(self, memo_content: str):
        """Article 32 (Security) must be referenced."""
        has_art_32 = any([
            "Art. 32" in memo_content,
            "Article 32" in memo_content,
        ])
        assert has_art_32, (
            "Missing reference to GDPR Article 32 (Security of Processing)"
        )


class TestPhase0ChecklistCompleteness:
    """Test that all Phase 0 DoD items are satisfied."""

    @pytest.fixture
    def memo_content(self) -> str:
        """Load the memo content."""
        return GDPR_SCOPE_MEMO_PATH.read_text()

    def test_dod_checklist_exists(self, memo_content: str):
        """Compliance checklist section must exist."""
        has_checklist = any([
            "Compliance Checklist" in memo_content,
            "Phase 0 DoD" in memo_content,
            "Definition of Done" in memo_content,
        ])
        assert has_checklist, "Missing Phase 0 DoD compliance checklist section"

    def test_all_dod_items_marked_complete(self, memo_content: str):
        """All DoD items should be marked as complete (checkmark)."""
        # Count checkmarks in checklist section
        checklist_section = memo_content[memo_content.find("Checklist"):] if "Checklist" in memo_content else memo_content

        # Count completion indicators
        checkmarks = len(re.findall(r"[✅✓]", checklist_section))
        incomplete = len(re.findall(r"[❌☐]", checklist_section))

        assert checkmarks > 0, "No completion checkmarks found in DoD checklist"
        assert incomplete == 0, f"Found {incomplete} incomplete items in DoD checklist"


class TestDataCategoryDefinitions:
    """Test that data categories are properly defined."""

    @pytest.fixture
    def memo_content(self) -> str:
        """Load the memo content."""
        return GDPR_SCOPE_MEMO_PATH.read_text()

    def test_has_data_category_appendix(self, memo_content: str):
        """Document should have a data category definitions appendix."""
        has_appendix = any([
            "Data Category Definitions" in memo_content,
            "Appendix A" in memo_content,
            "Category ID" in memo_content,
        ])
        assert has_appendix, "Missing data category definitions (Appendix)"

    def test_has_lawful_basis_reference(self, memo_content: str):
        """Document should have lawful basis reference table."""
        has_lb_reference = any([
            "Lawful Basis Reference" in memo_content,
            "Appendix B" in memo_content,
            "LB-CONTRACT" in memo_content,
        ])
        assert has_lb_reference, "Missing lawful basis reference (Appendix)"

    def test_has_retention_reference(self, memo_content: str):
        """Document should have retention period reference table."""
        has_ret_reference = any([
            "Retention Period Reference" in memo_content,
            "Retention Code" in memo_content,
            "RET-" in memo_content,
        ])
        assert has_ret_reference, "Missing retention period reference (Appendix)"


class TestRiskAssessment:
    """Test that risk assessment is included."""

    @pytest.fixture
    def memo_content(self) -> str:
        """Load the memo content."""
        return GDPR_SCOPE_MEMO_PATH.read_text()

    def test_has_risk_assessment_section(self, memo_content: str):
        """Document should include privacy risk assessment."""
        has_risk = any([
            "Risk Assessment" in memo_content,
            "Risk Matrix" in memo_content,
            "Privacy Risk" in memo_content,
        ])
        assert has_risk, "Missing risk assessment section"

    def test_has_dpia_reference(self, memo_content: str):
        """Document should reference DPIA requirement assessment."""
        has_dpia = any([
            "DPIA" in memo_content,
            "Data Protection Impact Assessment" in memo_content,
            "Article 35" in memo_content,
        ])
        assert has_dpia, "Missing DPIA requirement assessment"


class TestIntegrationWithDesignDoc:
    """Test alignment with CCEA Design Doc."""

    @pytest.fixture
    def memo_content(self) -> str:
        """Load the memo content."""
        return GDPR_SCOPE_MEMO_PATH.read_text()

    def test_references_design_doc(self, memo_content: str):
        """Document must reference the CCEA Design Doc."""
        has_design_doc_ref = any([
            "Design_Doc_CCEA_Cloud" in memo_content,
            "CCEA_CLOUD" in memo_content,
            "Design Doc" in memo_content,
        ])
        assert has_design_doc_ref, (
            "Missing reference to CCEA Design Doc "
            "(docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt)"
        )

    def test_design_doc_line_references(self, memo_content: str):
        """Document should include specific Design Doc line references."""
        # Should have at least some line number references
        has_line_refs = re.search(r"#L\d+|line \d+|§\d+", memo_content) is not None
        # Or section references
        has_section_refs = re.search(r"§\d+|Section \d+|\d+\.\d+", memo_content) is not None

        assert has_line_refs or has_section_refs, (
            "Document should include specific Design Doc references (line numbers or sections)"
        )


# Validation helper class for programmatic checks
class Phase0Validator:
    """Validator for Phase 0 deliverables - can be used programmatically."""

    def __init__(self, memo_path: Path = GDPR_SCOPE_MEMO_PATH):
        self.memo_path = memo_path
        self.content = memo_path.read_text() if memo_path.exists() else ""

    def validate_all(self) -> Dict[str, Any]:
        """Run all validations and return results."""
        results = {
            "document_exists": self.memo_path.exists(),
            "has_ropa_table": "RoPA" in self.content,
            "has_data_flow_diagram": "Data Flow" in self.content or "┌" in self.content,
            "has_controller_processor_roles": "Controller" in self.content and "Processor" in self.content,
            "has_telemetry_levels": all([
                "AGGREGATED" in self.content,
                "DETAILED_NON_SENSITIVE" in self.content,
                "RAW_ORDER_EVENTS" in self.content,
            ]),
            "has_eu_residency": "EU" in self.content and "eu-central-1" in self.content,
            "has_subprocessors": "Subprocessor" in self.content,
        }
        results["all_passed"] = all(results.values())
        return results


if __name__ == "__main__":
    # Quick validation when run directly
    validator = Phase0Validator()
    results = validator.validate_all()

    print("Phase 0 Deliverables Validation Results:")
    print("-" * 50)
    for key, value in results.items():
        status = "PASS" if value else "FAIL"
        print(f"  {key}: {status}")
    print("-" * 50)
    print(f"Overall: {'PASS' if results['all_passed'] else 'FAIL'}")
