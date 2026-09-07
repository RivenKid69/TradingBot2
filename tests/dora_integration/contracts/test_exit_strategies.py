# -*- coding: utf-8 -*-
"""
Tests for DORA Exit Strategies Module.

Tests Article 28(8) exit planning and management functionality.
"""

import pytest
from datetime import datetime, timezone, timedelta

from services.dora_integration.contracts.exit_strategies import (
    # Main class
    DORAExitStrategies,
    # Configuration
    ExitStrategiesConfig,
    # Enumerations
    ExitTrigger,
    ExitPhase,
    ExitPlanStatus,
    TransitionType,
    ReadinessLevel,
    AlternativeProviderStatus,
    RiskLevel,
    # Data structures
    AlternativeProvider,
    DataMigrationPlan,
    TransitionTask,
    ExitRisk,
    ExitCostEstimate,
    ExitPlan,
    ExitExecution,
    ExitReadinessAssessment,
    # Factory and utility functions
    create_exit_strategies,
    get_exit_triggers,
    get_exit_phases,
    get_transition_types,
)


class TestEnumerations:
    """Test all enumerations."""

    def test_exit_trigger_values(self):
        """Test ExitTrigger enum values."""
        assert ExitTrigger.PLANNED_TERMINATION.value == "planned_termination"
        assert ExitTrigger.CONTRACT_EXPIRY.value == "contract_expiry"
        assert ExitTrigger.PROVIDER_FAILURE.value == "provider_failure"
        assert ExitTrigger.PROVIDER_INSOLVENCY.value == "provider_insolvency"
        assert ExitTrigger.SECURITY_BREACH.value == "security_breach"
        assert ExitTrigger.REGULATORY_ACTION.value == "regulatory_action"
        assert ExitTrigger.PERFORMANCE_FAILURE.value == "performance_failure"
        assert ExitTrigger.STRATEGIC_CHANGE.value == "strategic_change"
        assert ExitTrigger.CONCENTRATION_RISK.value == "concentration_risk"
        assert ExitTrigger.COST_OPTIMIZATION.value == "cost_optimization"

    def test_exit_phase_values(self):
        """Test ExitPhase enum values."""
        assert ExitPhase.PLANNING.value == "planning"
        assert ExitPhase.NOTIFICATION.value == "notification"
        assert ExitPhase.TRANSITION.value == "transition"
        assert ExitPhase.PARALLEL_RUN.value == "parallel_run"
        assert ExitPhase.CUTOVER.value == "cutover"
        assert ExitPhase.VALIDATION.value == "validation"
        assert ExitPhase.CLEANUP.value == "cleanup"
        assert ExitPhase.COMPLETED.value == "completed"

    def test_exit_plan_status_values(self):
        """Test ExitPlanStatus enum values."""
        assert ExitPlanStatus.DRAFT.value == "draft"
        assert ExitPlanStatus.APPROVED.value == "approved"
        assert ExitPlanStatus.ACTIVE.value == "active"
        assert ExitPlanStatus.EXECUTING.value == "executing"
        assert ExitPlanStatus.COMPLETED.value == "completed"
        assert ExitPlanStatus.ABANDONED.value == "abandoned"

    def test_transition_type_values(self):
        """Test TransitionType enum values."""
        assert TransitionType.TO_ALTERNATIVE_PROVIDER.value == "to_alternative_provider"
        assert TransitionType.IN_HOUSE.value == "in_house"
        assert TransitionType.WIND_DOWN.value == "wind_down"
        assert TransitionType.HYBRID.value == "hybrid"

    def test_readiness_level_values(self):
        """Test ReadinessLevel enum values."""
        assert ReadinessLevel.NOT_READY.value == "not_ready"
        assert ReadinessLevel.PARTIALLY_READY.value == "partially_ready"
        assert ReadinessLevel.READY.value == "ready"
        assert ReadinessLevel.TESTED.value == "tested"

    def test_alternative_provider_status_values(self):
        """Test AlternativeProviderStatus enum values."""
        assert AlternativeProviderStatus.IDENTIFIED.value == "identified"
        assert AlternativeProviderStatus.EVALUATED.value == "evaluated"
        assert AlternativeProviderStatus.QUALIFIED.value == "qualified"
        assert AlternativeProviderStatus.CONTRACTED.value == "contracted"
        assert AlternativeProviderStatus.READY.value == "ready"

    def test_risk_level_values(self):
        """Test RiskLevel enum values."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"


class TestDataStructures:
    """Test data structures."""

    def test_alternative_provider_creation(self):
        """Test AlternativeProvider dataclass."""
        alt = AlternativeProvider(
            provider_name="Alternative Corp",
            provider_country="DE",
            capability_match_pct=85.0,
        )
        assert alt.alternative_id.startswith("ALT-")
        assert alt.provider_name == "Alternative Corp"
        assert alt.status == AlternativeProviderStatus.IDENTIFIED

    def test_alternative_provider_score_calculation(self):
        """Test AlternativeProvider score calculation."""
        alt = AlternativeProvider(
            provider_name="Test",
            technical_fit_score=80.0,
            commercial_fit_score=70.0,
            compliance_fit_score=90.0,
        )
        # Overall = 80*0.4 + 70*0.3 + 90*0.3 = 32 + 21 + 27 = 80
        assert alt.overall_score == 80.0

    def test_data_migration_plan_creation(self):
        """Test DataMigrationPlan dataclass."""
        migration = DataMigrationPlan(
            data_type="transactional",
            data_classification="confidential",
            estimated_data_volume_gb=500.0,
        )
        assert migration.migration_id.startswith("MIG-")
        assert migration.data_type == "transactional"

    def test_transition_task_creation(self):
        """Test TransitionTask dataclass."""
        task = TransitionTask(
            task_name="Setup environment",
            phase=ExitPhase.TRANSITION,
            responsible_party="alternative",
        )
        assert task.task_id.startswith("TSK-")
        assert task.phase == ExitPhase.TRANSITION

    def test_exit_risk_creation(self):
        """Test ExitRisk dataclass."""
        risk = ExitRisk(
            risk_name="Data loss during migration",
            description="Risk of data loss",
            category="technical",
            likelihood=3,
            impact=4,
        )
        assert risk.risk_id.startswith("ERK-")
        # Score = 3*4 = 12, which is HIGH (9-16)
        assert risk.risk_level == RiskLevel.HIGH

    def test_exit_risk_level_calculation(self):
        """Test ExitRisk level calculation."""
        low = ExitRisk(likelihood=1, impact=2)  # Score = 2
        assert low.risk_level == RiskLevel.LOW

        medium = ExitRisk(likelihood=3, impact=3)  # Score = 9
        assert medium.risk_level == RiskLevel.MEDIUM

        high = ExitRisk(likelihood=4, impact=4)  # Score = 16
        assert high.risk_level == RiskLevel.HIGH

        critical = ExitRisk(likelihood=5, impact=5)  # Score = 25
        assert critical.risk_level == RiskLevel.CRITICAL

    def test_exit_cost_estimate_creation(self):
        """Test ExitCostEstimate dataclass."""
        cost = ExitCostEstimate(
            cost_category="transition",
            estimated_amount_eur=50000.0,
            description="Integration costs",
        )
        assert cost.cost_id.startswith("CST-")
        assert cost.estimated_amount_eur == 50000.0

    def test_exit_plan_creation(self):
        """Test ExitPlan dataclass."""
        plan = ExitPlan(
            provider_id="PRV-001",
            provider_name="Cloud Provider",
            is_critical_provider=True,
        )
        assert plan.plan_id.startswith("EXP-")
        assert plan.status == ExitPlanStatus.DRAFT
        assert plan.is_critical_provider is True

    def test_exit_execution_creation(self):
        """Test ExitExecution dataclass."""
        execution = ExitExecution(
            plan_id="EXP-001",
            provider_name="Provider",
            trigger=ExitTrigger.PLANNED_TERMINATION,
        )
        assert execution.execution_id.startswith("EXE-")
        assert execution.trigger == ExitTrigger.PLANNED_TERMINATION

    def test_exit_readiness_assessment_creation(self):
        """Test ExitReadinessAssessment dataclass."""
        assessment = ExitReadinessAssessment(
            plan_id="EXP-001",
            assessed_by="Assessor",
        )
        assert assessment.assessment_id.startswith("ERA-")
        assert assessment.readiness_level == ReadinessLevel.NOT_READY


class TestFactoryFunctions:
    """Test factory and utility functions."""

    def test_create_exit_strategies(self, tmp_path):
        """Test factory function creates instance."""
        config = ExitStrategiesConfig(log_path=str(tmp_path / "logs"))
        manager = create_exit_strategies(config)
        assert isinstance(manager, DORAExitStrategies)

    def test_create_with_default_config(self, tmp_path):
        """Test factory with default config."""
        manager = create_exit_strategies()
        assert manager is not None

    def test_get_exit_triggers(self):
        """Test get_exit_triggers returns all triggers."""
        triggers = get_exit_triggers()
        assert len(triggers) == len(ExitTrigger)

    def test_get_exit_phases(self):
        """Test get_exit_phases returns all phases."""
        phases = get_exit_phases()
        assert len(phases) == len(ExitPhase)

    def test_get_transition_types(self):
        """Test get_transition_types returns all types."""
        types = get_transition_types()
        assert len(types) == len(TransitionType)


class TestDORAExitStrategies:
    """Test DORAExitStrategies main class."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager instance for testing."""
        config = ExitStrategiesConfig(log_path=str(tmp_path / "logs"))
        return DORAExitStrategies(config)

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager is not None
        assert len(manager._plans) == 0


class TestExitPlanManagement:
    """Test exit plan management functionality."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager instance for testing."""
        config = ExitStrategiesConfig(log_path=str(tmp_path / "logs"))
        return DORAExitStrategies(config)

    def test_create_exit_plan(self, manager):
        """Test creating exit plan."""
        plan = manager.create_exit_plan(
            provider_id="PRV-001",
            provider_name="Cloud Provider",
            services=["api", "data"],
            is_critical=False,
            created_by="Risk Manager",
        )
        assert plan is not None
        assert plan.provider_name == "Cloud Provider"
        assert len(plan.services_in_scope) == 2
        assert plan.status == ExitPlanStatus.DRAFT

    def test_create_critical_exit_plan(self, manager):
        """Test creating exit plan for critical provider."""
        plan = manager.create_exit_plan(
            provider_id="PRV-002",
            provider_name="Critical Provider",
            services=["trading"],
            is_critical=True,
            critical_functions=["order_execution"],
        )
        assert plan.is_critical_provider is True
        assert "order_execution" in plan.critical_functions_affected
        # Critical should have longer duration by default
        assert plan.total_duration_days == manager.config.critical_transition_days

    def test_create_plan_with_custom_duration(self, manager):
        """Test creating plan with custom duration."""
        plan = manager.create_exit_plan(
            provider_id="PRV-003",
            provider_name="Provider",
            services=["service"],
            total_duration_days=120,
        )
        assert plan.total_duration_days == 120

    def test_create_plan_creates_default_tasks(self, manager):
        """Test creating plan creates default tasks."""
        plan = manager.create_exit_plan(
            provider_id="PRV-004",
            provider_name="Provider",
            services=["service"],
        )
        tasks = manager.get_tasks_for_plan(plan.plan_id)
        assert len(tasks) > 0
        # Should have tasks for multiple phases
        phases = set(t.phase for t in tasks)
        assert ExitPhase.PLANNING in phases
        assert ExitPhase.TRANSITION in phases
        assert ExitPhase.CUTOVER in phases

    def test_get_exit_plan(self, manager):
        """Test getting exit plan by ID."""
        plan = manager.create_exit_plan(
            provider_id="PRV-001",
            provider_name="Provider",
            services=["service"],
        )
        retrieved = manager.get_exit_plan(plan.plan_id)
        assert retrieved is not None
        assert retrieved.plan_id == plan.plan_id

    def test_get_exit_plan_not_found(self, manager):
        """Test getting non-existent plan."""
        plan = manager.get_exit_plan("UNKNOWN")
        assert plan is None

    def test_get_exit_plan_for_provider(self, manager):
        """Test getting exit plan by provider ID."""
        plan = manager.create_exit_plan(
            provider_id="PRV-001",
            provider_name="Provider",
            services=["service"],
        )
        retrieved = manager.get_exit_plan_for_provider("PRV-001")
        assert retrieved is not None
        assert retrieved.plan_id == plan.plan_id

    def test_get_all_exit_plans(self, manager):
        """Test getting all exit plans."""
        manager.create_exit_plan("PRV-001", "Provider 1", ["s1"])
        manager.create_exit_plan("PRV-002", "Provider 2", ["s2"])
        plans = manager.get_all_exit_plans()
        assert len(plans) == 2

    def test_get_critical_exit_plans(self, manager):
        """Test getting critical exit plans only."""
        manager.create_exit_plan("PRV-001", "Normal", ["s1"], is_critical=False)
        manager.create_exit_plan("PRV-002", "Critical", ["s2"], is_critical=True)
        critical = manager.get_critical_exit_plans()
        assert len(critical) == 1
        assert critical[0].provider_name == "Critical"

    def test_update_exit_plan(self, manager):
        """Test updating exit plan."""
        plan = manager.create_exit_plan("PRV-001", "Provider", ["s1"])
        updated = manager.update_exit_plan(
            plan_id=plan.plan_id,
            max_acceptable_downtime_hours=8,
        )
        assert updated.max_acceptable_downtime_hours == 8
        # Version should increment
        assert updated.plan_version != "1.0"

    def test_approve_exit_plan(self, manager):
        """Test approving exit plan."""
        plan = manager.create_exit_plan("PRV-001", "Provider", ["s1"])
        approved = manager.approve_exit_plan(
            plan_id=plan.plan_id,
            approved_by="Risk Committee",
        )
        assert approved.status == ExitPlanStatus.APPROVED
        assert approved.approved_by == "Risk Committee"
        assert approved.approved_date is not None

    def test_get_plans_needing_review(self, manager):
        """Test getting plans due for review."""
        plan = manager.create_exit_plan("PRV-001", "Provider", ["s1"])
        # Set next review date to the past
        plan.next_review_date = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        needing_review = manager.get_plans_needing_review()
        assert plan.plan_id in [p.plan_id for p in needing_review]


class TestAlternativeProviderManagement:
    """Test alternative provider management."""

    @pytest.fixture
    def manager_with_plan(self, tmp_path):
        """Create manager with exit plan."""
        config = ExitStrategiesConfig(log_path=str(tmp_path / "logs"))
        manager = DORAExitStrategies(config)
        plan = manager.create_exit_plan("PRV-001", "Provider", ["service"])
        return manager, plan

    def test_add_alternative_provider(self, manager_with_plan):
        """Test adding alternative provider."""
        manager, plan = manager_with_plan
        alt = manager.add_alternative_provider(
            plan_id=plan.plan_id,
            provider_name="Alternative Corp",
            provider_country="DE",
            capability_match_pct=90.0,
        )
        assert alt is not None
        assert alt.provider_name == "Alternative Corp"

    def test_add_alternative_unknown_plan(self, manager_with_plan):
        """Test adding alternative to unknown plan."""
        manager, _ = manager_with_plan
        alt = manager.add_alternative_provider(
            plan_id="UNKNOWN",
            provider_name="Alt",
        )
        assert alt is None

    def test_evaluate_alternative(self, manager_with_plan):
        """Test evaluating alternative provider."""
        manager, plan = manager_with_plan
        alt = manager.add_alternative_provider(
            plan_id=plan.plan_id,
            provider_name="Alternative",
        )
        evaluated = manager.evaluate_alternative(
            alternative_id=alt.alternative_id,
            technical_score=85.0,
            commercial_score=80.0,
            compliance_score=90.0,
            evaluation_notes="Strong technical fit",
            pros=["Good API", "24/7 support"],
            cons=["Higher cost"],
        )
        assert evaluated.status == AlternativeProviderStatus.EVALUATED
        assert evaluated.overall_score > 0
        assert len(evaluated.pros) == 2
        assert len(evaluated.cons) == 1

    def test_qualify_alternative(self, manager_with_plan):
        """Test qualifying alternative provider."""
        manager, plan = manager_with_plan
        alt = manager.add_alternative_provider(
            plan_id=plan.plan_id,
            provider_name="Alternative",
        )
        manager.evaluate_alternative(alt.alternative_id, 80.0, 80.0, 80.0)
        qualified = manager.qualify_alternative(alt.alternative_id)
        assert qualified.status == AlternativeProviderStatus.QUALIFIED

    def test_get_alternatives_for_plan(self, manager_with_plan):
        """Test getting alternatives for plan."""
        manager, plan = manager_with_plan
        manager.add_alternative_provider(plan.plan_id, "Alt 1")
        manager.add_alternative_provider(plan.plan_id, "Alt 2")
        alternatives = manager.get_alternatives_for_plan(plan.plan_id)
        assert len(alternatives) == 2

    def test_get_best_alternative(self, manager_with_plan):
        """Test getting best-scored alternative."""
        manager, plan = manager_with_plan
        alt1 = manager.add_alternative_provider(plan.plan_id, "Alt 1")
        alt2 = manager.add_alternative_provider(plan.plan_id, "Alt 2")
        manager.evaluate_alternative(alt1.alternative_id, 70.0, 70.0, 70.0)
        manager.evaluate_alternative(alt2.alternative_id, 90.0, 90.0, 90.0)
        best = manager.get_best_alternative(plan.plan_id)
        assert best.alternative_id == alt2.alternative_id

    def test_select_target_provider(self, manager_with_plan):
        """Test selecting target provider."""
        manager, plan = manager_with_plan
        alt = manager.add_alternative_provider(plan.plan_id, "Target Provider")
        updated = manager.select_target_provider(
            plan_id=plan.plan_id,
            alternative_id=alt.alternative_id,
        )
        assert updated.target_provider_id == alt.alternative_id
        assert updated.target_provider_name == "Target Provider"


class TestTaskManagement:
    """Test transition task management."""

    @pytest.fixture
    def manager_with_plan(self, tmp_path):
        """Create manager with exit plan."""
        config = ExitStrategiesConfig(log_path=str(tmp_path / "logs"))
        manager = DORAExitStrategies(config)
        plan = manager.create_exit_plan("PRV-001", "Provider", ["service"])
        return manager, plan

    def test_add_transition_task(self, manager_with_plan):
        """Test adding transition task."""
        manager, plan = manager_with_plan
        task = manager.add_transition_task(
            plan_id=plan.plan_id,
            task_name="Custom Task",
            description="Do something",
            phase=ExitPhase.TRANSITION,
            owner="Team Lead",
        )
        assert task is not None
        assert task.task_name == "Custom Task"

    def test_update_task_status(self, manager_with_plan):
        """Test updating task status."""
        manager, plan = manager_with_plan
        tasks = manager.get_tasks_for_plan(plan.plan_id)
        task = tasks[0]
        updated = manager.update_task_status(
            task_id=task.task_id,
            status="in_progress",
            completion_pct=50.0,
        )
        assert updated.status == "in_progress"
        assert updated.completion_pct == 50.0
        assert updated.actual_start_date is not None

    def test_complete_task(self, manager_with_plan):
        """Test completing task."""
        manager, plan = manager_with_plan
        tasks = manager.get_tasks_for_plan(plan.plan_id)
        task = tasks[0]
        completed = manager.update_task_status(
            task_id=task.task_id,
            status="completed",
        )
        assert completed.status == "completed"
        assert completed.completion_pct == 100.0
        assert completed.actual_end_date is not None

    def test_get_tasks_by_phase(self, manager_with_plan):
        """Test getting tasks by phase."""
        manager, plan = manager_with_plan
        planning_tasks = manager.get_tasks_by_phase(plan.plan_id, ExitPhase.PLANNING)
        assert len(planning_tasks) > 0
        assert all(t.phase == ExitPhase.PLANNING for t in planning_tasks)


class TestDataMigrationPlanning:
    """Test data migration planning."""

    @pytest.fixture
    def manager_with_plan(self, tmp_path):
        """Create manager with exit plan."""
        config = ExitStrategiesConfig(log_path=str(tmp_path / "logs"))
        manager = DORAExitStrategies(config)
        plan = manager.create_exit_plan("PRV-001", "Provider", ["service"])
        return manager, plan

    def test_add_data_migration(self, manager_with_plan):
        """Test adding data migration plan."""
        manager, plan = manager_with_plan
        migration = manager.add_data_migration(
            plan_id=plan.plan_id,
            data_type="transactional",
            data_classification="confidential",
            estimated_volume_gb=500.0,
            migration_method="api_sync",
            estimated_hours=48,
        )
        assert migration is not None
        assert migration.data_type == "transactional"
        assert migration.estimated_data_volume_gb == 500.0

    def test_get_migrations_for_plan(self, manager_with_plan):
        """Test getting migrations for plan."""
        manager, plan = manager_with_plan
        manager.add_data_migration(
            plan.plan_id, "transactional", "confidential", 100.0, "export_import", 24
        )
        manager.add_data_migration(plan.plan_id, "historical", "internal", 1000.0, "api_sync", 72)
        migrations = manager.get_migrations_for_plan(plan.plan_id)
        assert len(migrations) == 2


class TestRiskManagement:
    """Test exit risk management."""

    @pytest.fixture
    def manager_with_plan(self, tmp_path):
        """Create manager with exit plan."""
        config = ExitStrategiesConfig(log_path=str(tmp_path / "logs"))
        manager = DORAExitStrategies(config)
        plan = manager.create_exit_plan("PRV-001", "Provider", ["service"])
        return manager, plan

    def test_add_exit_risk(self, manager_with_plan):
        """Test adding exit risk."""
        manager, plan = manager_with_plan
        risk = manager.add_exit_risk(
            plan_id=plan.plan_id,
            risk_name="Data loss",
            description="Risk of data loss during migration",
            category="technical",
            likelihood=3,
            impact=4,
            mitigation_strategy="Backup before migration",
        )
        assert risk is not None
        assert risk.risk_level == RiskLevel.HIGH

    def test_get_risks_for_plan(self, manager_with_plan):
        """Test getting risks for plan."""
        manager, plan = manager_with_plan
        manager.add_exit_risk(plan.plan_id, "Risk 1", "Desc", "technical", 2, 2)
        manager.add_exit_risk(plan.plan_id, "Risk 2", "Desc", "operational", 3, 3)
        risks = manager.get_risks_for_plan(plan.plan_id)
        assert len(risks) == 2

    def test_get_high_risks(self, manager_with_plan):
        """Test getting high/critical risks."""
        manager, plan = manager_with_plan
        manager.add_exit_risk(plan.plan_id, "Low", "D", "c", 1, 1)  # Low
        manager.add_exit_risk(plan.plan_id, "High", "D", "c", 4, 4)  # High
        manager.add_exit_risk(plan.plan_id, "Critical", "D", "c", 5, 5)  # Critical
        high_risks = manager.get_high_risks(plan.plan_id)
        assert len(high_risks) == 2


class TestCostEstimation:
    """Test exit cost estimation."""

    @pytest.fixture
    def manager_with_plan(self, tmp_path):
        """Create manager with exit plan."""
        config = ExitStrategiesConfig(log_path=str(tmp_path / "logs"))
        manager = DORAExitStrategies(config)
        plan = manager.create_exit_plan("PRV-001", "Provider", ["service"])
        return manager, plan

    def test_add_cost_estimate(self, manager_with_plan):
        """Test adding cost estimate."""
        manager, plan = manager_with_plan
        cost = manager.add_cost_estimate(
            plan_id=plan.plan_id,
            category="transition",
            amount_eur=50000.0,
            description="Integration costs",
            confidence="medium",
        )
        assert cost is not None
        assert cost.estimated_amount_eur == 50000.0

    def test_cost_updates_plan_total(self, manager_with_plan):
        """Test cost updates plan total with contingency."""
        manager, plan = manager_with_plan
        manager.add_cost_estimate(plan.plan_id, "transition", 100000.0, "Transition")
        manager.add_cost_estimate(plan.plan_id, "resources", 50000.0, "Resources")
        updated_plan = manager.get_exit_plan(plan.plan_id)
        # Total = 150000 + 20% contingency = 180000
        assert updated_plan.total_estimated_cost_eur == 180000.0

    def test_get_costs_for_plan(self, manager_with_plan):
        """Test getting costs for plan."""
        manager, plan = manager_with_plan
        manager.add_cost_estimate(plan.plan_id, "transition", 10000.0, "C1")
        manager.add_cost_estimate(plan.plan_id, "termination", 5000.0, "C2")
        costs = manager.get_costs_for_plan(plan.plan_id)
        assert len(costs) == 2


class TestExitReadinessAssessment:
    """Test exit readiness assessment."""

    @pytest.fixture
    def manager_with_full_plan(self, tmp_path):
        """Create manager with comprehensive exit plan."""
        config = ExitStrategiesConfig(log_path=str(tmp_path / "logs"))
        manager = DORAExitStrategies(config)
        plan = manager.create_exit_plan("PRV-001", "Provider", ["service"])
        # Add alternatives
        alt = manager.add_alternative_provider(plan.plan_id, "Alternative")
        manager.evaluate_alternative(alt.alternative_id, 80.0, 80.0, 80.0)
        manager.qualify_alternative(alt.alternative_id)
        manager.select_target_provider(plan.plan_id, alt.alternative_id)
        # Add costs
        manager.add_cost_estimate(plan.plan_id, "transition", 50000.0, "Cost")
        # Add risks
        manager.add_exit_risk(plan.plan_id, "Risk", "Desc", "tech", 2, 2)
        return manager, plan

    def test_assess_exit_readiness(self, manager_with_full_plan):
        """Test assessing exit readiness."""
        manager, plan = manager_with_full_plan
        assessment = manager.assess_exit_readiness(
            plan_id=plan.plan_id,
            assessed_by="Risk Manager",
        )
        assert assessment is not None
        assert assessment.overall_score > 0
        assert assessment.readiness_level in ReadinessLevel

    def test_assessment_categories(self, manager_with_full_plan):
        """Test assessment has all categories."""
        manager, plan = manager_with_full_plan
        assessment = manager.assess_exit_readiness(plan.plan_id)
        assert assessment.documentation_score >= 0
        assert assessment.alternatives_score >= 0
        assert assessment.resources_score >= 0
        assert assessment.testing_score >= 0
        assert assessment.communication_score >= 0

    def test_assessment_generates_recommendations(self, manager_with_full_plan):
        """Test assessment generates recommendations for gaps."""
        manager, plan = manager_with_full_plan
        assessment = manager.assess_exit_readiness(plan.plan_id)
        # Should have some gaps and recommendations
        assert len(assessment.gaps) >= 0
        assert len(assessment.recommendations) >= 0

    def test_assessment_updates_plan(self, manager_with_full_plan):
        """Test assessment updates plan readiness."""
        manager, plan = manager_with_full_plan
        manager.assess_exit_readiness(plan.plan_id)
        updated_plan = manager.get_exit_plan(plan.plan_id)
        assert updated_plan.last_readiness_assessment is not None
        assert updated_plan.readiness_score_pct > 0


class TestExitExecution:
    """Test exit execution functionality."""

    @pytest.fixture
    def manager_with_approved_plan(self, tmp_path):
        """Create manager with approved exit plan."""
        config = ExitStrategiesConfig(log_path=str(tmp_path / "logs"))
        manager = DORAExitStrategies(config)
        plan = manager.create_exit_plan("PRV-001", "Provider", ["service"])
        manager.approve_exit_plan(plan.plan_id, "Approver")
        return manager, plan

    def test_trigger_exit(self, manager_with_approved_plan):
        """Test triggering exit execution."""
        manager, plan = manager_with_approved_plan
        execution = manager.trigger_exit(
            plan_id=plan.plan_id,
            trigger=ExitTrigger.PLANNED_TERMINATION,
            trigger_reason="Contract expiry",
            triggered_by="Exit Manager",
        )
        assert execution is not None
        assert execution.trigger == ExitTrigger.PLANNED_TERMINATION
        assert execution.status == "initiated"

    def test_trigger_exit_unapproved_fails(self, tmp_path):
        """Test triggering exit on unapproved plan fails."""
        config = ExitStrategiesConfig(log_path=str(tmp_path / "logs"))
        manager = DORAExitStrategies(config)
        plan = manager.create_exit_plan("PRV-001", "Provider", ["service"])
        # Don't approve
        execution = manager.trigger_exit(plan.plan_id, ExitTrigger.PLANNED_TERMINATION, "Reason")
        assert execution is None

    def test_update_execution_progress(self, manager_with_approved_plan):
        """Test updating execution progress."""
        manager, plan = manager_with_approved_plan
        execution = manager.trigger_exit(plan.plan_id, ExitTrigger.PLANNED_TERMINATION, "Reason")
        updated = manager.update_execution_progress(
            execution_id=execution.execution_id,
            phase=ExitPhase.TRANSITION,
            tasks_completed=5,
        )
        assert updated.current_phase == ExitPhase.TRANSITION
        assert updated.tasks_completed == 5
        assert updated.overall_progress_pct > 0

    def test_complete_exit(self, manager_with_approved_plan):
        """Test completing exit."""
        manager, plan = manager_with_approved_plan
        execution = manager.trigger_exit(plan.plan_id, ExitTrigger.PLANNED_TERMINATION, "Reason")
        completed = manager.complete_exit(execution.execution_id)
        assert completed.status == "completed"
        assert completed.current_phase == ExitPhase.COMPLETED
        # Plan should also be marked completed
        updated_plan = manager.get_exit_plan(plan.plan_id)
        assert updated_plan.status == ExitPlanStatus.COMPLETED

    def test_get_active_executions(self, manager_with_approved_plan):
        """Test getting active executions."""
        manager, plan = manager_with_approved_plan
        execution = manager.trigger_exit(plan.plan_id, ExitTrigger.PLANNED_TERMINATION, "Reason")
        active = manager.get_active_executions()
        assert execution.execution_id in [e.execution_id for e in active]


class TestExitTesting:
    """Test exit plan testing functionality."""

    @pytest.fixture
    def manager_with_plan(self, tmp_path):
        """Create manager with exit plan."""
        config = ExitStrategiesConfig(log_path=str(tmp_path / "logs"))
        manager = DORAExitStrategies(config)
        plan = manager.create_exit_plan("PRV-001", "Provider", ["service"])
        return manager, plan

    def test_record_exit_test(self, manager_with_plan):
        """Test recording exit test."""
        manager, plan = manager_with_plan
        test_date = datetime.now(timezone.utc).isoformat()
        result = manager.record_exit_test(
            plan_id=plan.plan_id,
            test_type="tabletop",
            test_date=test_date,
            success=True,
            findings=["Good coordination", "Need more docs"],
            tested_by="Test Manager",
        )
        assert result.last_test_date == test_date
        assert len(result.test_results) == 1
        assert result.test_results[0]["success"] is True


class TestReporting:
    """Test reporting functionality."""

    @pytest.fixture
    def manager_with_data(self, tmp_path):
        """Create manager with various data."""
        config = ExitStrategiesConfig(log_path=str(tmp_path / "logs"))
        manager = DORAExitStrategies(config)
        # Create multiple plans
        manager.create_exit_plan("PRV-001", "Normal", ["s1"], is_critical=False)
        manager.create_exit_plan("PRV-002", "Critical", ["s2"], is_critical=True)
        return manager

    def test_get_exit_strategy_status(self, manager_with_data):
        """Test get_exit_strategy_status."""
        status = manager_with_data.get_exit_strategy_status()
        assert "timestamp" in status
        assert "plans" in status
        assert "alternatives" in status
        assert "executions" in status
        assert "compliance_indicators" in status

    def test_status_plan_counts(self, manager_with_data):
        """Test status shows correct plan counts."""
        status = manager_with_data.get_exit_strategy_status()
        assert status["plans"]["total"] == 2
        assert status["plans"]["critical"] == 1


class TestConfiguration:
    """Test configuration options."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExitStrategiesConfig()
        assert config.require_exit_plan_for_critical is True
        assert config.exit_plan_review_frequency_months == 12
        assert config.critical_plan_review_months == 6
        assert config.min_alternatives_for_critical == 2
        assert config.default_transition_days == 90
        assert config.critical_transition_days == 180
        assert config.default_contingency_pct == 20.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExitStrategiesConfig(
            exit_plan_review_frequency_months=6,
            default_contingency_pct=25.0,
            test_frequency_months=6,
        )
        assert config.exit_plan_review_frequency_months == 6
        assert config.default_contingency_pct == 25.0
        assert config.test_frequency_months == 6

    def test_notification_callback(self, tmp_path):
        """Test notification callback."""
        callback_data = {}

        def notification_callback(event_type, data):
            callback_data["event"] = event_type
            callback_data["data"] = data

        config = ExitStrategiesConfig(
            log_path=str(tmp_path / "logs"),
            notification_callback=notification_callback,
        )
        manager = DORAExitStrategies(config)
        plan = manager.create_exit_plan("PRV-001", "Provider", ["service"])
        manager.approve_exit_plan(plan.plan_id, "Approver")
        manager.trigger_exit(plan.plan_id, ExitTrigger.PLANNED_TERMINATION, "Test")
        assert callback_data.get("event") == "exit_triggered"
