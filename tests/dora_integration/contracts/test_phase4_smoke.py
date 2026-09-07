# -*- coding: utf-8 -*-
"""
Smoke Tests for DORA Integration Layer Phase 4.

Comprehensive integration tests for Contracts & SLA Layer.
Validates that all modules work together correctly.
"""

import pytest
from datetime import datetime, timezone

from services.dora_integration.contracts import (
    # Contractual Requirements
    DORAContractualRequirements,
    ContractualRequirementsConfig,
    RequirementCategory,
    RequirementType,
    ComplianceStatus,
    GapSeverity,
    ContractStatus,
    create_contractual_requirements,
    get_article_30_requirements,
    get_basic_requirement_count,
    get_critical_requirement_count,
    # SLA Guardrails
    SLAGuardrails,
    SLAGuardrailsConfig,
    SLATier,
    CapacityStatus,
    ApprovalStatus,
    create_sla_guardrails,
    get_sla_tier_definitions,
    get_sla_tiers,
    # Exit Strategies
    DORAExitStrategies,
    ExitStrategiesConfig,
    ExitTrigger,
    ExitPhase,
    ExitPlanStatus,
    TransitionType,
    ReadinessLevel,
    create_exit_strategies,
    get_exit_triggers,
    get_exit_phases,
)


class TestPhase4Imports:
    """Test all Phase 4 imports work correctly."""

    def test_contractual_requirements_imports(self):
        """Test contractual requirements imports."""
        assert DORAContractualRequirements is not None
        assert ContractualRequirementsConfig is not None
        assert RequirementCategory is not None
        assert RequirementType is not None
        assert ComplianceStatus is not None
        assert GapSeverity is not None
        assert ContractStatus is not None
        assert create_contractual_requirements is not None

    def test_sla_guardrails_imports(self):
        """Test SLA guardrails imports."""
        assert SLAGuardrails is not None
        assert SLAGuardrailsConfig is not None
        assert SLATier is not None
        assert CapacityStatus is not None
        assert ApprovalStatus is not None
        assert create_sla_guardrails is not None

    def test_exit_strategies_imports(self):
        """Test exit strategies imports."""
        assert DORAExitStrategies is not None
        assert ExitStrategiesConfig is not None
        assert ExitTrigger is not None
        assert ExitPhase is not None
        assert ExitPlanStatus is not None
        assert TransitionType is not None
        assert ReadinessLevel is not None
        assert create_exit_strategies is not None


class TestPhase4FactoryFunctions:
    """Test all Phase 4 factory functions."""

    def test_create_contractual_requirements(self, tmp_path):
        """Test creating contractual requirements manager."""
        config = ContractualRequirementsConfig(log_path=str(tmp_path / "logs"))
        manager = create_contractual_requirements(config)
        assert isinstance(manager, DORAContractualRequirements)

    def test_create_sla_guardrails(self):
        """Test creating SLA guardrails."""
        guardrails = create_sla_guardrails()
        assert isinstance(guardrails, SLAGuardrails)

    def test_create_exit_strategies(self, tmp_path):
        """Test creating exit strategies manager."""
        config = ExitStrategiesConfig(log_path=str(tmp_path / "logs"))
        manager = create_exit_strategies(config)
        assert isinstance(manager, DORAExitStrategies)


class TestPhase4Utilities:
    """Test utility functions."""

    def test_article_30_requirements(self):
        """Test Article 30 requirements retrieval."""
        requirements = get_article_30_requirements()
        assert len(requirements) > 0
        basic_count = get_basic_requirement_count()
        critical_count = get_critical_requirement_count()
        assert basic_count + critical_count == len(requirements)

    def test_sla_tiers(self):
        """Test SLA tier retrieval."""
        definitions = get_sla_tier_definitions()
        assert len(definitions) == 4
        tiers = get_sla_tiers()
        assert len(tiers) == 4

    def test_exit_triggers_phases(self):
        """Test exit triggers and phases."""
        triggers = get_exit_triggers()
        phases = get_exit_phases()
        assert len(triggers) == len(ExitTrigger)
        assert len(phases) == len(ExitPhase)


class TestPhase4IntegrationWorkflow:
    """Test integrated workflow across all Phase 4 modules."""

    @pytest.fixture
    def full_setup(self, tmp_path):
        """Create all Phase 4 managers."""
        contract_config = ContractualRequirementsConfig(
            log_path=str(tmp_path / "contract_logs")
        )
        exit_config = ExitStrategiesConfig(
            log_path=str(tmp_path / "exit_logs")
        )
        return {
            "contracts": DORAContractualRequirements(contract_config),
            "sla": SLAGuardrails(),
            "exit": DORAExitStrategies(exit_config),
        }

    def test_provider_onboarding_workflow(self, full_setup):
        """
        Test complete provider onboarding workflow:
        1. Register contract
        2. Assess compliance
        3. Request SLA commitment
        4. Create exit plan
        """
        contracts = full_setup["contracts"]
        sla = full_setup["sla"]
        exit_mgr = full_setup["exit"]

        # Step 1: Register contract
        contract = contracts.register_contract(
            provider_name="Cloud Provider X",
            provider_id="PRV-001",
            contract_type="procurement",
            supports_critical_function=True,
            critical_functions=["market_data", "order_execution"],
            notice_period_days=90,
        )
        assert contract.contract_id.startswith("CTR-")
        assert contract.supports_critical_function is True

        # Step 2: Assess contract compliance
        assessment = contracts.assess_contract(
            contract_id=contract.contract_id,
            assessment_type="initial",
            assessed_by="Compliance Team",
        )
        assert assessment is not None
        # Will have gaps for missing provisions
        assert assessment.overall_compliance in [
            ComplianceStatus.NON_COMPLIANT,
            ComplianceStatus.PARTIALLY_COMPLIANT,
        ]

        # Step 3: Request SLA commitment
        sla_request = sla.request_sla_commitment(
            client_id="CLT-001",
            client_name="Financial Institution A",
            requested_tier=SLATier.PROFESSIONAL,
            requested_by="Sales Team",
            services_in_scope=["market_data", "order_execution"],
            is_critical_function=True,
        )
        assert sla_request.request_id.startswith("SLA-")
        # Should be pending (Professional available by default)
        assert sla_request.approval_status == ApprovalStatus.PENDING

        # Step 4: Create exit plan
        exit_plan = exit_mgr.create_exit_plan(
            provider_id="PRV-001",
            provider_name="Cloud Provider X",
            services=["market_data", "order_execution"],
            is_critical=True,
            critical_functions=["market_data", "order_execution"],
            created_by="Risk Manager",
        )
        assert exit_plan.plan_id.startswith("EXP-")
        assert exit_plan.is_critical_provider is True
        # Should have default tasks
        tasks = exit_mgr.get_tasks_for_plan(exit_plan.plan_id)
        assert len(tasks) > 0

    def test_compliance_remediation_workflow(self, full_setup):
        """
        Test compliance remediation workflow:
        1. Assess contract and identify gaps
        2. Generate amendments
        3. Track remediation
        """
        contracts = full_setup["contracts"]

        # Register and assess contract
        contract = contracts.register_contract(
            provider_name="Provider Y",
            supports_critical_function=False,
        )
        assessment = contracts.assess_contract(contract.contract_id)

        # Get gaps
        gaps = contracts.get_gaps_for_contract(contract.contract_id)
        assert len(gaps) > 0

        # Generate amendments
        amendments = contracts.generate_amendments(contract.contract_id)
        assert len(amendments) > 0

    def test_sla_approval_workflow(self, full_setup):
        """
        Test SLA approval workflow:
        1. Validate capacity
        2. Request commitment
        3. Approve commitment
        """
        sla = full_setup["sla"]

        # Step 1: Validate capacity
        validation = sla.validate_capacity_for_tier(
            tier=SLATier.STANDARD,
            validated_by="Platform Engineer",
        )
        assert validation.status == CapacityStatus.VALIDATED

        # Step 2: Request commitment
        request = sla.request_sla_commitment(
            client_id="CLT-002",
            client_name="Bank B",
            requested_tier=SLATier.STANDARD,
            requested_by="Sales",
            services_in_scope=["api_access"],
        )
        assert request.approval_status == ApprovalStatus.PENDING

        # Step 3: Approve
        approved = sla.approve_commitment(
            request_id=request.request_id,
            approved_by="Engineering Lead",
        )
        assert approved.approval_status == ApprovalStatus.APPROVED

    def test_exit_readiness_workflow(self, full_setup):
        """
        Test exit readiness workflow:
        1. Create plan
        2. Add alternatives
        3. Assess readiness
        4. Approve plan
        """
        exit_mgr = full_setup["exit"]

        # Step 1: Create plan
        plan = exit_mgr.create_exit_plan(
            provider_id="PRV-003",
            provider_name="Provider Z",
            services=["storage"],
        )

        # Step 2: Add and evaluate alternatives
        alt = exit_mgr.add_alternative_provider(
            plan_id=plan.plan_id,
            provider_name="Alternative Storage Co",
            capability_match_pct=85.0,
        )
        exit_mgr.evaluate_alternative(
            alternative_id=alt.alternative_id,
            technical_score=80.0,
            commercial_score=75.0,
            compliance_score=90.0,
        )
        exit_mgr.qualify_alternative(alt.alternative_id)
        exit_mgr.select_target_provider(plan.plan_id, alt.alternative_id)

        # Add costs
        exit_mgr.add_cost_estimate(
            plan_id=plan.plan_id,
            category="transition",
            amount_eur=25000.0,
            description="Migration costs",
        )

        # Add risks
        exit_mgr.add_exit_risk(
            plan_id=plan.plan_id,
            risk_name="Service disruption",
            description="Risk of service disruption during transition",
            category="operational",
            likelihood=2,
            impact=3,
        )

        # Step 3: Assess readiness
        assessment = exit_mgr.assess_exit_readiness(
            plan_id=plan.plan_id,
            assessed_by="Risk Manager",
        )
        assert assessment.overall_score > 0

        # Step 4: Approve plan
        approved = exit_mgr.approve_exit_plan(
            plan_id=plan.plan_id,
            approved_by="Risk Committee",
        )
        assert approved.status == ExitPlanStatus.APPROVED


class TestPhase4DataModels:
    """Test data model consistency across Phase 4."""

    def test_all_ids_have_prefixes(self):
        """Test all IDs use standard prefixes."""
        from services.dora_integration.contracts.contractual_requirements import (
            ICTContract, ContractAssessment, ContractGap, ContractAmendment,
        )
        from services.dora_integration.contracts.sla_guardrails import (
            CapacityValidation, SLACommitmentRequest,
        )
        from services.dora_integration.contracts.exit_strategies import (
            ExitPlan, AlternativeProvider, TransitionTask,
        )

        # Test ID prefixes
        contract = ICTContract(provider_name="Test")
        assert contract.contract_id.startswith("CTR-")

        assessment = ContractAssessment(contract_id="CTR-001", provider_name="Test")
        assert assessment.assessment_id.startswith("ASM-")

        gap = ContractGap(assessment_id="ASM-001", contract_id="CTR-001", requirement_id="REQ-001")
        assert gap.gap_id.startswith("GAP-")

        validation = CapacityValidation()
        assert validation.validation_id.startswith("VAL-")

        request = SLACommitmentRequest()
        assert request.request_id.startswith("SLA-")

        plan = ExitPlan()
        assert plan.plan_id.startswith("EXP-")

        alt = AlternativeProvider()
        assert alt.alternative_id.startswith("ALT-")

        task = TransitionTask()
        assert task.task_id.startswith("TSK-")


class TestPhase4DORArticleCompliance:
    """Test DORA Article compliance coverage."""

    def test_article_30_coverage(self):
        """Test Article 30 requirements coverage."""
        requirements = get_article_30_requirements()

        # Check Article 30(2) basic requirements are covered
        basic_reqs = [r for r in requirements if r.category == RequirementCategory.BASIC]
        assert len(basic_reqs) >= 9  # Art. 30(2)(a)-(i)

        # Check Article 30(3) critical requirements are covered
        critical_reqs = [r for r in requirements if r.category == RequirementCategory.CRITICAL]
        assert len(critical_reqs) >= 10  # Art. 30(3) additional requirements

    def test_article_30_2_e_sla_coverage(self):
        """Test Article 30(2)(e) SLA coverage."""
        tiers = get_sla_tier_definitions()

        # Verify all tiers have required SLA elements
        for tier, defn in tiers.items():
            assert defn.availability_target_pct > 0
            assert defn.rto_hours > 0
            assert defn.rpo_minutes > 0
            assert defn.incident_response_critical_minutes > 0

    def test_article_28_8_exit_strategy_coverage(self):
        """Test Article 28(8) exit strategy coverage."""
        triggers = get_exit_triggers()
        phases = get_exit_phases()

        # Verify exit triggers include key scenarios
        trigger_values = [t.value for t in triggers]
        assert "provider_failure" in trigger_values
        assert "provider_insolvency" in trigger_values
        assert "security_breach" in trigger_values

        # Verify exit phases cover full lifecycle
        phase_values = [p.value for p in phases]
        assert "planning" in phase_values
        assert "transition" in phase_values
        assert "parallel_run" in phase_values
        assert "cutover" in phase_values
        assert "validation" in phase_values


class TestPhase4ReportingConsistency:
    """Test reporting consistency across Phase 4 modules."""

    @pytest.fixture
    def managers(self, tmp_path):
        """Create all managers with data."""
        contract_config = ContractualRequirementsConfig(
            log_path=str(tmp_path / "contract_logs")
        )
        exit_config = ExitStrategiesConfig(
            log_path=str(tmp_path / "exit_logs")
        )
        contracts = DORAContractualRequirements(contract_config)
        sla = SLAGuardrails()
        exit_mgr = DORAExitStrategies(exit_config)

        # Add some data
        contracts.register_contract("Provider", supports_critical_function=True)
        sla.request_sla_commitment("C1", "Client", SLATier.STANDARD, "Sales", ["api"])
        exit_mgr.create_exit_plan("P1", "Provider", ["service"])

        return contracts, sla, exit_mgr

    def test_all_reports_have_timestamp(self, managers):
        """Test all reports include timestamp."""
        contracts, sla, exit_mgr = managers

        contract_summary = contracts.get_compliance_summary()
        assert "timestamp" in contract_summary

        sla_report = sla.generate_capacity_report()
        assert "report_date" in sla_report

        exit_status = exit_mgr.get_exit_strategy_status()
        assert "timestamp" in exit_status

    def test_all_reports_have_counts(self, managers):
        """Test all reports include relevant counts."""
        contracts, sla, exit_mgr = managers

        contract_summary = contracts.get_compliance_summary()
        assert "contracts" in contract_summary
        assert "total" in contract_summary["contracts"]

        sla_report = sla.generate_capacity_report()
        assert "total_validations" in sla_report
        assert "total_requests" in sla_report

        exit_status = exit_mgr.get_exit_strategy_status()
        assert "plans" in exit_status
        assert "total" in exit_status["plans"]


class TestPhase4ThreadSafety:
    """Test thread safety of Phase 4 operations."""

    def test_concurrent_contract_registration(self, tmp_path):
        """Test concurrent contract registration."""
        import threading

        config = ContractualRequirementsConfig(log_path=str(tmp_path / "logs"))
        manager = DORAContractualRequirements(config)

        contracts = []

        def register():
            contract = manager.register_contract(f"Provider-{threading.current_thread().name}")
            contracts.append(contract)

        threads = [threading.Thread(target=register, name=f"T{i}") for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(contracts) == 10
        # All contracts should have unique IDs
        ids = [c.contract_id for c in contracts]
        assert len(set(ids)) == 10

    def test_concurrent_sla_requests(self):
        """Test concurrent SLA requests."""
        import threading

        guardrails = SLAGuardrails()
        requests = []

        def request_sla(idx):
            req = guardrails.request_sla_commitment(
                f"C{idx}", f"Client {idx}", SLATier.STANDARD, "Sales", ["api"]
            )
            requests.append(req)

        threads = [threading.Thread(target=request_sla, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(requests) == 10
        ids = [r.request_id for r in requests]
        assert len(set(ids)) == 10
