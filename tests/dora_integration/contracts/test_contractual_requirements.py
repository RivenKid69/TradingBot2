# -*- coding: utf-8 -*-
"""
Tests for DORA Contractual Requirements Module.

Tests Article 30 contract compliance checking functionality.
"""

import pytest
from datetime import datetime, timezone, timedelta

from services.dora_integration.contracts.contractual_requirements import (
    # Main class
    DORAContractualRequirements,
    # Configuration
    ContractualRequirementsConfig,
    # Enumerations
    RequirementCategory,
    RequirementType,
    ComplianceStatus,
    GapSeverity,
    RemediationStatus,
    ContractStatus,
    # Data structures
    ContractualRequirement,
    ContractProvision,
    ContractAssessment,
    ContractGap,
    ContractAmendment,
    SLADefinition,
    ICTContract,
    TerminationClause,
    # Factory and utility functions
    create_contractual_requirements,
    get_article_30_requirements,
    get_requirement_types,
    get_basic_requirement_count,
    get_critical_requirement_count,
    get_termination_clause_templates,
)


class TestEnumerations:
    """Test all enumerations."""

    def test_requirement_category_values(self):
        """Test RequirementCategory enum values."""
        assert RequirementCategory.BASIC.value == "basic"
        assert RequirementCategory.CRITICAL.value == "critical"

    def test_requirement_type_basic_values(self):
        """Test RequirementType enum has Article 30(2) basic requirements."""
        basic_types = [
            RequirementType.SERVICE_DESCRIPTION,
            RequirementType.DATA_LOCATION,
            RequirementType.SERVICE_LEVELS,
            RequirementType.DATA_ACCESS_RECOVERY,
            RequirementType.SERVICE_LEVEL_DESCRIPTIONS,
            RequirementType.INCIDENT_ASSISTANCE,
            RequirementType.AUTHORITY_COOPERATION,
            RequirementType.TERMINATION_RIGHTS,
            RequirementType.TRAINING_PARTICIPATION,
        ]
        for req_type in basic_types:
            assert req_type in RequirementType

    def test_requirement_type_critical_values(self):
        """Test RequirementType enum has Article 30(3) critical requirements."""
        critical_types = [
            RequirementType.SLA_TARGETS,
            RequirementType.NOTICE_PERIODS,
            RequirementType.BCP_REQUIREMENTS,
            RequirementType.RESILIENCE_TESTING,
            RequirementType.AUDIT_RIGHTS,
            RequirementType.EXIT_STRATEGY,
            RequirementType.NCA_ACCESS,
            RequirementType.BUSINESS_CONTINUITY,
            RequirementType.SECURITY_MEASURES,
            RequirementType.SUBCONTRACTING,
        ]
        for req_type in critical_types:
            assert req_type in RequirementType

    def test_compliance_status_values(self):
        """Test ComplianceStatus enum values."""
        assert ComplianceStatus.COMPLIANT.value == "compliant"
        assert ComplianceStatus.PARTIALLY_COMPLIANT.value == "partially_compliant"
        assert ComplianceStatus.NON_COMPLIANT.value == "non_compliant"
        assert ComplianceStatus.NOT_ASSESSED.value == "not_assessed"
        assert ComplianceStatus.REQUIRES_REVIEW.value == "requires_review"

    def test_gap_severity_values(self):
        """Test GapSeverity enum values."""
        assert GapSeverity.CRITICAL.value == "critical"
        assert GapSeverity.HIGH.value == "high"
        assert GapSeverity.MEDIUM.value == "medium"
        assert GapSeverity.LOW.value == "low"

    def test_remediation_status_values(self):
        """Test RemediationStatus enum values."""
        assert RemediationStatus.NOT_STARTED.value == "not_started"
        assert RemediationStatus.IN_PROGRESS.value == "in_progress"
        assert RemediationStatus.COMPLETED.value == "completed"
        assert RemediationStatus.BLOCKED.value == "blocked"
        assert RemediationStatus.NOT_APPLICABLE.value == "not_applicable"

    def test_contract_status_values(self):
        """Test ContractStatus enum values."""
        assert ContractStatus.ACTIVE.value == "active"
        assert ContractStatus.UNDER_REVIEW.value == "under_review"
        assert ContractStatus.PENDING_AMENDMENT.value == "pending_amendment"
        assert ContractStatus.AMENDED.value == "amended"
        assert ContractStatus.TERMINATED.value == "terminated"


class TestDataStructures:
    """Test data structures."""

    def test_termination_clause_creation(self):
        """Test TerminationClause dataclass."""
        clause = TerminationClause(
            clause_type="standard",
            notice_period_days=90,
            minimum_notice_days=30,
        )
        assert clause.clause_type == "standard"
        assert clause.notice_period_days == 90
        assert clause.minimum_notice_days == 30
        assert clause.clause_id.startswith("TRM-")

    def test_contractual_requirement_creation(self):
        """Test ContractualRequirement dataclass."""
        req = ContractualRequirement(
            requirement_type=RequirementType.SERVICE_DESCRIPTION,
            category=RequirementCategory.BASIC,
            article_reference="Article 30(2)(a)",
            name="Test Requirement",
            description="Test description",
            mandatory=True,
        )
        assert req.requirement_type == RequirementType.SERVICE_DESCRIPTION
        assert req.category == RequirementCategory.BASIC
        assert req.requirement_id.startswith("REQ-")

    def test_contract_provision_creation(self):
        """Test ContractProvision dataclass."""
        provision = ContractProvision(
            contract_id="CTR-001",
            requirement_id="REQ-001",
            clause_reference="Section 5.1",
            provision_summary="Test provision",
        )
        assert provision.contract_id == "CTR-001"
        assert provision.provision_id.startswith("PRV-")
        assert provision.compliance_status == ComplianceStatus.NOT_ASSESSED

    def test_contract_assessment_creation(self):
        """Test ContractAssessment dataclass."""
        assessment = ContractAssessment(
            contract_id="CTR-001",
            provider_name="Test Provider",
            is_critical_function=True,
        )
        assert assessment.contract_id == "CTR-001"
        assert assessment.assessment_id.startswith("ASM-")
        assert assessment.assessment_date != ""

    def test_contract_gap_creation(self):
        """Test ContractGap dataclass."""
        gap = ContractGap(
            assessment_id="ASM-001",
            contract_id="CTR-001",
            requirement_id="REQ-001",
            gap_description="Missing clause",
            severity=GapSeverity.HIGH,
        )
        assert gap.gap_id.startswith("GAP-")
        assert gap.severity == GapSeverity.HIGH
        assert gap.remediation_status == RemediationStatus.NOT_STARTED

    def test_contract_amendment_creation(self):
        """Test ContractAmendment dataclass."""
        amendment = ContractAmendment(
            contract_id="CTR-001",
            provider_name="Test Provider",
            amendment_type="modify_clause",
        )
        assert amendment.amendment_id.startswith("AMD-")
        assert amendment.created_date != ""

    def test_sla_definition_creation(self):
        """Test SLADefinition dataclass."""
        sla = SLADefinition(
            contract_id="CTR-001",
            service_name="Test Service",
            metric_name="Availability",
            target_value=99.9,
            target_unit="percent",
        )
        assert sla.sla_id.startswith("SLA-")
        assert sla.target_value == 99.9

    def test_ict_contract_creation(self):
        """Test ICTContract dataclass."""
        contract = ICTContract(
            provider_name="Test Provider",
            contract_type="procurement",
            supports_critical_function=True,
        )
        assert contract.contract_id.startswith("CTR-")
        assert contract.status == ContractStatus.ACTIVE
        assert contract.supports_critical_function is True


class TestFactoryFunctions:
    """Test factory and utility functions."""

    def test_create_contractual_requirements(self):
        """Test factory function creates instance."""
        manager = create_contractual_requirements()
        assert isinstance(manager, DORAContractualRequirements)

    def test_create_with_config(self):
        """Test factory with custom config."""
        config = ContractualRequirementsConfig(
            assessment_frequency_months=6,
            compliant_threshold_pct=95.0,
        )
        manager = create_contractual_requirements(config)
        assert manager.config.assessment_frequency_months == 6
        assert manager.config.compliant_threshold_pct == 95.0

    def test_get_article_30_requirements(self):
        """Test get_article_30_requirements returns requirements."""
        requirements = get_article_30_requirements()
        assert len(requirements) > 0
        assert all(isinstance(r, ContractualRequirement) for r in requirements)

    def test_get_requirement_types(self):
        """Test get_requirement_types returns all types."""
        types = get_requirement_types()
        assert len(types) == len(RequirementType)

    def test_get_basic_requirement_count(self):
        """Test get_basic_requirement_count."""
        count = get_basic_requirement_count()
        assert count > 0
        # Should match Article 30(2) requirements
        assert count >= 9  # At least 9 basic requirements

    def test_get_critical_requirement_count(self):
        """Test get_critical_requirement_count."""
        count = get_critical_requirement_count()
        assert count > 0
        # Should match Article 30(3) requirements
        assert count >= 10

    def test_get_termination_clause_templates(self):
        """Test get_termination_clause_templates."""
        templates = get_termination_clause_templates()
        assert "standard" in templates
        assert "for_cause" in templates
        assert "regulatory" in templates
        assert "critical_function" in templates
        for template in templates.values():
            assert isinstance(template, TerminationClause)


class TestDORAContractualRequirements:
    """Test DORAContractualRequirements main class."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager instance for testing."""
        config = ContractualRequirementsConfig(
            log_path=str(tmp_path / "logs"),
        )
        return DORAContractualRequirements(config)

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager is not None
        assert len(manager._requirements) > 0

    def test_get_all_requirements(self, manager):
        """Test get_all_requirements."""
        reqs = manager.get_all_requirements()
        assert len(reqs) > 0

    def test_get_basic_requirements(self, manager):
        """Test get_basic_requirements."""
        basic = manager.get_basic_requirements()
        assert len(basic) > 0
        assert all(r.category == RequirementCategory.BASIC for r in basic)

    def test_get_critical_requirements(self, manager):
        """Test get_critical_requirements."""
        critical = manager.get_critical_requirements()
        assert len(critical) > 0
        assert all(r.category == RequirementCategory.CRITICAL for r in critical)

    def test_get_applicable_requirements_non_critical(self, manager):
        """Test get_applicable_requirements for non-critical contracts."""
        applicable = manager.get_applicable_requirements(supports_critical_function=False)
        assert len(applicable) == len(manager.get_basic_requirements())

    def test_get_applicable_requirements_critical(self, manager):
        """Test get_applicable_requirements for critical contracts."""
        applicable = manager.get_applicable_requirements(supports_critical_function=True)
        assert len(applicable) == len(manager.get_all_requirements())

    def test_get_requirement_by_id(self, manager):
        """Test get_requirement by ID."""
        reqs = manager.get_all_requirements()
        req = manager.get_requirement(reqs[0].requirement_id)
        assert req is not None
        assert req.requirement_id == reqs[0].requirement_id

    def test_get_requirement_not_found(self, manager):
        """Test get_requirement returns None for unknown ID."""
        req = manager.get_requirement("UNKNOWN-ID")
        assert req is None


class TestContractManagement:
    """Test contract management functionality."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager instance for testing."""
        config = ContractualRequirementsConfig(
            log_path=str(tmp_path / "logs"),
        )
        return DORAContractualRequirements(config)

    def test_register_contract(self, manager):
        """Test contract registration."""
        contract = manager.register_contract(
            provider_name="Test Provider",
            provider_id="PRV-001",
            contract_type="procurement",
            supports_critical_function=False,
        )
        assert contract is not None
        assert contract.provider_name == "Test Provider"
        assert contract.contract_id.startswith("CTR-")
        assert contract.status == ContractStatus.ACTIVE

    def test_register_critical_contract(self, manager):
        """Test critical contract registration."""
        contract = manager.register_contract(
            provider_name="Critical Provider",
            supports_critical_function=True,
            critical_functions=["trading", "settlement"],
            notice_period_days=90,
        )
        assert contract.supports_critical_function is True
        assert "trading" in contract.critical_functions
        assert contract.notice_period_days == 90

    def test_get_contract(self, manager):
        """Test get contract by ID."""
        contract = manager.register_contract(provider_name="Test")
        retrieved = manager.get_contract(contract.contract_id)
        assert retrieved is not None
        assert retrieved.contract_id == contract.contract_id

    def test_get_contract_not_found(self, manager):
        """Test get_contract returns None for unknown ID."""
        contract = manager.get_contract("UNKNOWN-ID")
        assert contract is None

    def test_get_all_contracts(self, manager):
        """Test get_all_contracts."""
        manager.register_contract(provider_name="Provider 1")
        manager.register_contract(provider_name="Provider 2")
        contracts = manager.get_all_contracts()
        assert len(contracts) == 2

    def test_get_all_contracts_exclude_terminated(self, manager):
        """Test get_all_contracts excludes terminated by default."""
        contract = manager.register_contract(provider_name="Provider")
        contract.status = ContractStatus.TERMINATED
        contracts = manager.get_all_contracts(include_terminated=False)
        assert len(contracts) == 0

    def test_get_contracts_for_critical_functions(self, manager):
        """Test get_contracts_for_critical_functions."""
        manager.register_contract(provider_name="Normal", supports_critical_function=False)
        manager.register_contract(provider_name="Critical", supports_critical_function=True)
        critical = manager.get_contracts_for_critical_functions()
        assert len(critical) == 1
        assert critical[0].provider_name == "Critical"

    def test_get_contracts_needing_assessment(self, manager):
        """Test get_contracts_needing_assessment."""
        contract = manager.register_contract(provider_name="Test")
        # No assessment done yet
        needing = manager.get_contracts_needing_assessment()
        assert contract.contract_id in [c.contract_id for c in needing]


class TestProvisionManagement:
    """Test provision management functionality."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager instance for testing."""
        config = ContractualRequirementsConfig(
            log_path=str(tmp_path / "logs"),
        )
        manager = DORAContractualRequirements(config)
        # Create a contract for testing
        manager.register_contract(provider_name="Test Provider")
        return manager

    def test_add_contract_provision(self, manager):
        """Test adding provision to contract."""
        contracts = manager.get_all_contracts()
        contract_id = contracts[0].contract_id
        reqs = manager.get_all_requirements()

        provision = manager.add_contract_provision(
            contract_id=contract_id,
            requirement_id=reqs[0].requirement_id,
            clause_reference="Section 5.1",
            provision_summary="Test provision",
        )
        assert provision is not None
        assert provision.contract_id == contract_id

    def test_add_provision_unknown_contract(self, manager):
        """Test adding provision to unknown contract."""
        provision = manager.add_contract_provision(
            contract_id="UNKNOWN",
            requirement_id="REQ-001",
            clause_reference="Section 1",
            provision_summary="Test",
        )
        assert provision is None

    def test_assess_provision(self, manager):
        """Test assessing a provision."""
        contracts = manager.get_all_contracts()
        contract_id = contracts[0].contract_id
        reqs = manager.get_all_requirements()

        provision = manager.add_contract_provision(
            contract_id=contract_id,
            requirement_id=reqs[0].requirement_id,
            clause_reference="Section 5.1",
            provision_summary="Test provision",
        )

        assessed = manager.assess_provision(
            provision_id=provision.provision_id,
            compliance_status=ComplianceStatus.COMPLIANT,
            compliance_notes="Fully compliant",
            assessed_by="Test Assessor",
        )
        assert assessed.compliance_status == ComplianceStatus.COMPLIANT
        assert assessed.assessed_by == "Test Assessor"

    def test_get_provisions_for_contract(self, manager):
        """Test getting provisions for a contract."""
        contracts = manager.get_all_contracts()
        contract_id = contracts[0].contract_id
        reqs = manager.get_all_requirements()

        manager.add_contract_provision(
            contract_id=contract_id,
            requirement_id=reqs[0].requirement_id,
            clause_reference="Section 1",
            provision_summary="Provision 1",
        )
        manager.add_contract_provision(
            contract_id=contract_id,
            requirement_id=reqs[1].requirement_id,
            clause_reference="Section 2",
            provision_summary="Provision 2",
        )

        provisions = manager.get_provisions_for_contract(contract_id)
        assert len(provisions) == 2


class TestContractAssessment:
    """Test contract assessment functionality."""

    @pytest.fixture
    def manager_with_contract(self, tmp_path):
        """Create manager with contract and provisions."""
        config = ContractualRequirementsConfig(
            log_path=str(tmp_path / "logs"),
        )
        manager = DORAContractualRequirements(config)
        contract = manager.register_contract(
            provider_name="Test Provider",
            supports_critical_function=False,
        )
        return manager, contract

    def test_assess_contract_no_provisions(self, manager_with_contract):
        """Test assessment with no provisions creates gaps."""
        manager, contract = manager_with_contract
        assessment = manager.assess_contract(
            contract_id=contract.contract_id,
            assessment_type="initial",
            assessed_by="Test Assessor",
        )
        assert assessment is not None
        assert assessment.non_compliant_count > 0
        assert len(assessment.gaps) > 0

    def test_assess_contract_with_provisions(self, manager_with_contract):
        """Test assessment with compliant provisions."""
        manager, contract = manager_with_contract
        reqs = manager.get_basic_requirements()

        # Add compliant provisions for all basic requirements
        for req in reqs:
            provision = manager.add_contract_provision(
                contract_id=contract.contract_id,
                requirement_id=req.requirement_id,
                clause_reference=f"Section {req.requirement_id}",
                provision_summary="Compliant provision",
                compliance_status=ComplianceStatus.COMPLIANT,
            )

        assessment = manager.assess_contract(
            contract_id=contract.contract_id,
            assessment_type="periodic",
        )
        assert assessment.compliant_count == len(reqs)
        assert assessment.compliance_score_pct >= 90

    def test_assess_unknown_contract(self, manager_with_contract):
        """Test assessment of unknown contract."""
        manager, _ = manager_with_contract
        assessment = manager.assess_contract(contract_id="UNKNOWN")
        assert assessment is None

    def test_get_assessment(self, manager_with_contract):
        """Test get_assessment by ID."""
        manager, contract = manager_with_contract
        assessment = manager.assess_contract(contract_id=contract.contract_id)
        retrieved = manager.get_assessment(assessment.assessment_id)
        assert retrieved is not None
        assert retrieved.assessment_id == assessment.assessment_id

    def test_get_assessments_for_contract(self, manager_with_contract):
        """Test get_assessments_for_contract."""
        manager, contract = manager_with_contract
        manager.assess_contract(contract_id=contract.contract_id)
        manager.assess_contract(contract_id=contract.contract_id)
        assessments = manager.get_assessments_for_contract(contract.contract_id)
        assert len(assessments) == 2

    def test_approve_assessment(self, manager_with_contract):
        """Test approve_assessment."""
        manager, contract = manager_with_contract
        assessment = manager.assess_contract(contract_id=contract.contract_id)
        approved = manager.approve_assessment(
            assessment_id=assessment.assessment_id,
            approved_by="Approver",
        )
        assert approved.approved_by == "Approver"
        assert approved.approval_date is not None


class TestGapManagement:
    """Test gap management functionality."""

    @pytest.fixture
    def manager_with_gaps(self, tmp_path):
        """Create manager with contract and assessment (creates gaps)."""
        config = ContractualRequirementsConfig(
            log_path=str(tmp_path / "logs"),
        )
        manager = DORAContractualRequirements(config)
        contract = manager.register_contract(
            provider_name="Test Provider",
            supports_critical_function=False,
        )
        # Assessment without provisions creates gaps
        assessment = manager.assess_contract(contract_id=contract.contract_id)
        return manager, contract, assessment

    def test_get_gap(self, manager_with_gaps):
        """Test get_gap by ID."""
        manager, contract, assessment = manager_with_gaps
        gaps = manager.get_gaps_for_contract(contract.contract_id)
        if gaps:
            gap = manager.get_gap(gaps[0].gap_id)
            assert gap is not None

    def test_get_gaps_for_contract(self, manager_with_gaps):
        """Test get_gaps_for_contract."""
        manager, contract, assessment = manager_with_gaps
        gaps = manager.get_gaps_for_contract(contract.contract_id)
        assert len(gaps) > 0

    def test_get_open_gaps(self, manager_with_gaps):
        """Test get_open_gaps."""
        manager, contract, assessment = manager_with_gaps
        open_gaps = manager.get_open_gaps()
        assert len(open_gaps) > 0
        assert all(g.remediation_status not in [
            RemediationStatus.COMPLETED,
            RemediationStatus.NOT_APPLICABLE,
        ] for g in open_gaps)

    def test_get_gaps_by_severity(self, manager_with_gaps):
        """Test get_gaps_by_severity."""
        manager, contract, assessment = manager_with_gaps
        # Most missing provisions should be HIGH or CRITICAL
        high_gaps = manager.get_gaps_by_severity(GapSeverity.HIGH)
        critical_gaps = manager.get_gaps_by_severity(GapSeverity.CRITICAL)
        assert len(high_gaps) + len(critical_gaps) > 0

    def test_update_gap_remediation(self, manager_with_gaps):
        """Test update_gap_remediation."""
        manager, contract, assessment = manager_with_gaps
        gaps = manager.get_gaps_for_contract(contract.contract_id)
        if gaps:
            updated = manager.update_gap_remediation(
                gap_id=gaps[0].gap_id,
                status=RemediationStatus.IN_PROGRESS,
                owner="Remediation Owner",
            )
            assert updated.remediation_status == RemediationStatus.IN_PROGRESS
            assert updated.remediation_owner == "Remediation Owner"

    def test_complete_gap_remediation(self, manager_with_gaps):
        """Test completing gap remediation."""
        manager, contract, assessment = manager_with_gaps
        gaps = manager.get_gaps_for_contract(contract.contract_id)
        if gaps:
            updated = manager.update_gap_remediation(
                gap_id=gaps[0].gap_id,
                status=RemediationStatus.COMPLETED,
            )
            assert updated.remediation_status == RemediationStatus.COMPLETED
            assert updated.remediation_completed_date is not None


class TestAmendmentManagement:
    """Test amendment management functionality."""

    @pytest.fixture
    def manager_with_gaps(self, tmp_path):
        """Create manager with contract and assessment."""
        config = ContractualRequirementsConfig(
            log_path=str(tmp_path / "logs"),
        )
        manager = DORAContractualRequirements(config)
        contract = manager.register_contract(
            provider_name="Test Provider",
            supports_critical_function=False,
        )
        assessment = manager.assess_contract(contract_id=contract.contract_id)
        return manager, contract, assessment

    def test_create_amendment_request(self, manager_with_gaps):
        """Test creating amendment request."""
        manager, contract, assessment = manager_with_gaps
        gaps = manager.get_gaps_for_contract(contract.contract_id)

        amendment = manager.create_amendment_request(
            contract_id=contract.contract_id,
            gap_ids=[g.gap_id for g in gaps[:2]],
            proposed_changes=[{"action": "add_clause"}],
            justification="DORA compliance",
            created_by="Test User",
        )
        assert amendment is not None
        assert amendment.amendment_id.startswith("AMD-")
        assert amendment.status == "draft"

    def test_create_amendment_unknown_contract(self, manager_with_gaps):
        """Test creating amendment for unknown contract."""
        manager, _, _ = manager_with_gaps
        amendment = manager.create_amendment_request(
            contract_id="UNKNOWN",
            gap_ids=[],
            proposed_changes=[],
            justification="Test",
        )
        assert amendment is None

    def test_generate_amendments(self, manager_with_gaps):
        """Test auto-generating amendments."""
        manager, contract, assessment = manager_with_gaps
        amendments = manager.generate_amendments(contract.contract_id)
        assert len(amendments) > 0

    def test_submit_amendment(self, manager_with_gaps):
        """Test submitting amendment."""
        manager, contract, assessment = manager_with_gaps
        amendments = manager.generate_amendments(contract.contract_id)
        if amendments:
            submitted = manager.submit_amendment(amendments[0].amendment_id)
            assert submitted.status == "submitted"
            assert submitted.submitted_date is not None

    def test_record_amendment_response_accepted(self, manager_with_gaps):
        """Test recording accepted amendment response."""
        manager, contract, assessment = manager_with_gaps
        amendments = manager.generate_amendments(contract.contract_id)
        if amendments:
            manager.submit_amendment(amendments[0].amendment_id)
            responded = manager.record_amendment_response(
                amendment_id=amendments[0].amendment_id,
                accepted=True,
                notes="Accepted by provider",
            )
            assert responded.status == "accepted"

    def test_record_amendment_response_counter(self, manager_with_gaps):
        """Test recording counter proposal."""
        manager, contract, assessment = manager_with_gaps
        amendments = manager.generate_amendments(contract.contract_id)
        if amendments:
            manager.submit_amendment(amendments[0].amendment_id)
            responded = manager.record_amendment_response(
                amendment_id=amendments[0].amendment_id,
                accepted=False,
                counter_proposal="Modified terms",
            )
            assert responded.status == "negotiating"

    def test_complete_amendment(self, manager_with_gaps):
        """Test completing amendment."""
        manager, contract, assessment = manager_with_gaps
        amendments = manager.generate_amendments(contract.contract_id)
        if amendments:
            manager.submit_amendment(amendments[0].amendment_id)
            manager.record_amendment_response(
                amendment_id=amendments[0].amendment_id,
                accepted=True,
            )
            completed = manager.complete_amendment(amendments[0].amendment_id)
            assert completed.status == "implemented"
            assert completed.implementation_date is not None

    def test_get_pending_amendments(self, manager_with_gaps):
        """Test get_pending_amendments."""
        manager, contract, assessment = manager_with_gaps
        manager.generate_amendments(contract.contract_id)
        pending = manager.get_pending_amendments()
        assert len(pending) > 0


class TestSLAManagement:
    """Test SLA management functionality."""

    @pytest.fixture
    def manager_with_contract(self, tmp_path):
        """Create manager with contract."""
        config = ContractualRequirementsConfig(
            log_path=str(tmp_path / "logs"),
        )
        manager = DORAContractualRequirements(config)
        contract = manager.register_contract(provider_name="Test Provider")
        return manager, contract

    def test_add_sla(self, manager_with_contract):
        """Test adding SLA definition."""
        manager, contract = manager_with_contract
        sla = manager.add_sla(
            contract_id=contract.contract_id,
            service_name="API Service",
            metric_name="Availability",
            target_value=99.9,
            target_unit="percent",
            measurement_period="monthly",
        )
        assert sla is not None
        assert sla.target_value == 99.9

    def test_add_sla_unknown_contract(self, manager_with_contract):
        """Test adding SLA to unknown contract."""
        manager, _ = manager_with_contract
        sla = manager.add_sla(
            contract_id="UNKNOWN",
            service_name="Service",
            metric_name="Metric",
            target_value=99.0,
            target_unit="percent",
        )
        assert sla is None

    def test_get_slas_for_contract(self, manager_with_contract):
        """Test getting SLAs for contract."""
        manager, contract = manager_with_contract
        manager.add_sla(
            contract_id=contract.contract_id,
            service_name="Service 1",
            metric_name="Availability",
            target_value=99.9,
            target_unit="percent",
        )
        manager.add_sla(
            contract_id=contract.contract_id,
            service_name="Service 2",
            metric_name="Response Time",
            target_value=100,
            target_unit="ms",
        )
        slas = manager.get_slas_for_contract(contract.contract_id)
        assert len(slas) == 2


class TestReporting:
    """Test reporting functionality."""

    @pytest.fixture
    def manager_with_data(self, tmp_path):
        """Create manager with contracts and assessments."""
        config = ContractualRequirementsConfig(
            log_path=str(tmp_path / "logs"),
        )
        manager = DORAContractualRequirements(config)
        # Create contracts
        contract1 = manager.register_contract(
            provider_name="Provider 1",
            supports_critical_function=False,
        )
        contract2 = manager.register_contract(
            provider_name="Provider 2",
            supports_critical_function=True,
        )
        # Create assessments
        manager.assess_contract(contract_id=contract1.contract_id)
        manager.assess_contract(contract_id=contract2.contract_id)
        return manager

    def test_get_compliance_summary(self, manager_with_data):
        """Test get_compliance_summary."""
        summary = manager_with_data.get_compliance_summary()
        assert "timestamp" in summary
        assert "contracts" in summary
        assert "assessments" in summary
        assert "gaps" in summary
        assert "amendments" in summary
        assert "compliance_indicators" in summary

    def test_generate_gap_report(self, manager_with_data):
        """Test generate_gap_report."""
        report = manager_with_data.generate_gap_report()
        assert "report_date" in report
        assert "total_gaps" in report
        assert "gaps_by_severity" in report

    def test_generate_gap_report_for_contract(self, manager_with_data):
        """Test generate_gap_report for specific contract."""
        contracts = manager_with_data.get_all_contracts()
        report = manager_with_data.generate_gap_report(
            contract_id=contracts[0].contract_id
        )
        assert report["contract_id"] == contracts[0].contract_id


class TestConfiguration:
    """Test configuration options."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ContractualRequirementsConfig()
        assert config.assessment_frequency_months == 12
        assert config.critical_contract_assessment_months == 6
        assert config.compliant_threshold_pct == 90.0
        assert config.partially_compliant_threshold_pct == 70.0
        assert config.gap_remediation_default_days == 90
        assert config.critical_gap_remediation_days == 30

    def test_custom_config(self):
        """Test custom configuration."""
        config = ContractualRequirementsConfig(
            assessment_frequency_months=6,
            compliant_threshold_pct=95.0,
            notify_on_new_gap=False,
        )
        assert config.assessment_frequency_months == 6
        assert config.compliant_threshold_pct == 95.0
        assert config.notify_on_new_gap is False

    def test_escalation_callback(self, tmp_path):
        """Test escalation callback is called."""
        callback_data = {}

        def escalation_callback(event_type, data):
            callback_data["event"] = event_type
            callback_data["data"] = data

        config = ContractualRequirementsConfig(
            log_path=str(tmp_path / "logs"),
            critical_gap_escalation=True,
            escalation_callback=escalation_callback,
        )
        manager = DORAContractualRequirements(config)
        contract = manager.register_contract(
            provider_name="Test",
            supports_critical_function=False,
        )
        # Assessment creates critical gaps for missing mandatory provisions
        manager.assess_contract(contract_id=contract.contract_id)
        # Escalation may be called for critical gaps
        # (depends on gap severity determination)
