# -*- coding: utf-8 -*-
"""
Tests for DORA Exit Strategies Module (Article 28(8)).

Comprehensive test coverage for:
- Enumerations (ExitTrigger, ExitPhase, etc.)
- Data structures (ExitPlan, AlternativeProvider, etc.)
- ExitStrategiesConfig
- DORAExitStrategies
- Factory functions

References:
    - Article 28(8) DORA (EU 2022/2554)
    - Article 30(3)(f) DORA
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch
import threading

from services.dora.exit_strategies import (
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
    # Configuration
    ExitStrategiesConfig,
    # Main class
    DORAExitStrategies,
    # Factory functions
    create_exit_strategies,
    get_exit_triggers,
    get_exit_phases,
    get_transition_types,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def config():
    """Create test configuration."""
    return ExitStrategiesConfig(
        default_transition_days=90,
        critical_transition_days=180,
        min_alternatives_for_critical=2,
        log_all_events=False,
    )


@pytest.fixture
def manager(config):
    """Create test manager instance."""
    return DORAExitStrategies(config=config)


@pytest.fixture
def sample_plan(manager):
    """Create and return sample exit plan."""
    plan = manager.create_exit_plan(
        provider_id="PRV-001",
        provider_name="Cloud Provider AG",
        services=["compute", "storage", "database"],
        is_critical=True,
        critical_functions=["trading", "risk_management"],
    )
    return plan


@pytest.fixture
def plan_with_alternatives(manager, sample_plan):
    """Create plan with alternative providers."""
    manager.add_alternative_provider(
        plan_id=sample_plan.plan_id,
        provider_name="Alternative Cloud A",
        provider_country="DE",
        capability_match_pct=90.0,
    )
    manager.add_alternative_provider(
        plan_id=sample_plan.plan_id,
        provider_name="Alternative Cloud B",
        provider_country="FR",
        capability_match_pct=85.0,
    )
    return sample_plan


# =============================================================================
# Enumeration Tests
# =============================================================================

class TestEnumerations:
    """Test all enumeration classes."""

    def test_exit_trigger_values(self):
        """Test ExitTrigger enumeration values."""
        assert ExitTrigger.PLANNED_TERMINATION.value == "planned_termination"
        assert ExitTrigger.PROVIDER_FAILURE.value == "provider_failure"
        assert ExitTrigger.SECURITY_BREACH.value == "security_breach"
        assert len(ExitTrigger) == 10

    def test_exit_phase_values(self):
        """Test ExitPhase enumeration values."""
        assert ExitPhase.PLANNING.value == "planning"
        assert ExitPhase.TRANSITION.value == "transition"
        assert ExitPhase.CUTOVER.value == "cutover"
        assert ExitPhase.COMPLETED.value == "completed"
        assert len(ExitPhase) == 8

    def test_exit_plan_status_values(self):
        """Test ExitPlanStatus enumeration values."""
        assert ExitPlanStatus.DRAFT.value == "draft"
        assert ExitPlanStatus.APPROVED.value == "approved"
        assert ExitPlanStatus.EXECUTING.value == "executing"

    def test_transition_type_values(self):
        """Test TransitionType enumeration values."""
        assert TransitionType.TO_ALTERNATIVE_PROVIDER.value == "to_alternative_provider"
        assert TransitionType.IN_HOUSE.value == "in_house"
        assert TransitionType.WIND_DOWN.value == "wind_down"

    def test_readiness_level_values(self):
        """Test ReadinessLevel enumeration values."""
        assert ReadinessLevel.NOT_READY.value == "not_ready"
        assert ReadinessLevel.READY.value == "ready"
        assert ReadinessLevel.TESTED.value == "tested"

    def test_alternative_provider_status_values(self):
        """Test AlternativeProviderStatus enumeration values."""
        assert AlternativeProviderStatus.IDENTIFIED.value == "identified"
        assert AlternativeProviderStatus.QUALIFIED.value == "qualified"


# =============================================================================
# Data Structure Tests
# =============================================================================

class TestAlternativeProvider:
    """Test AlternativeProvider data structure."""

    def test_alternative_creation(self):
        """Test alternative provider creation."""
        alt = AlternativeProvider(
            provider_name="Test Provider",
            provider_country="DE",
        )
        assert alt.alternative_id.startswith("ALT-")
        assert alt.status == AlternativeProviderStatus.IDENTIFIED

    def test_alternative_score_calculation(self):
        """Test overall score calculation."""
        alt = AlternativeProvider(
            provider_name="Scored Provider",
            technical_fit_score=80,
            commercial_fit_score=70,
            compliance_fit_score=90,
        )
        # 80*0.4 + 70*0.3 + 90*0.3 = 32 + 21 + 27 = 80
        assert alt.overall_score == 80.0


class TestDataMigrationPlan:
    """Test DataMigrationPlan data structure."""

    def test_migration_creation(self):
        """Test data migration plan creation."""
        migration = DataMigrationPlan(
            data_type="transactional",
            data_classification="confidential",
            estimated_data_volume_gb=500,
        )
        assert migration.migration_id.startswith("MIG-")
        assert migration.estimated_data_volume_gb == 500


class TestTransitionTask:
    """Test TransitionTask data structure."""

    def test_task_creation(self):
        """Test transition task creation."""
        task = TransitionTask(
            task_name="Setup alternative environment",
            phase=ExitPhase.TRANSITION,
        )
        assert task.task_id.startswith("TSK-")
        assert task.phase == ExitPhase.TRANSITION


class TestExitRisk:
    """Test ExitRisk data structure."""

    def test_exit_risk_creation(self):
        """Test exit risk creation."""
        risk = ExitRisk(
            risk_name="Data Loss Risk",
            description="Risk of data loss during migration",
            likelihood=4,
            impact=5,
        )
        assert risk.risk_id.startswith("ERK-")
        assert risk.risk_level == RiskLevel.CRITICAL  # 4*5=20 > 16

    def test_risk_level_calculation(self):
        """Test risk level calculation."""
        low_risk = ExitRisk(likelihood=1, impact=2)  # 2
        assert low_risk.risk_level == RiskLevel.LOW

        medium_risk = ExitRisk(likelihood=3, impact=3)  # 9
        assert medium_risk.risk_level == RiskLevel.MEDIUM

        high_risk = ExitRisk(likelihood=4, impact=4)  # 16
        assert high_risk.risk_level == RiskLevel.HIGH


class TestExitPlan:
    """Test ExitPlan data structure."""

    def test_exit_plan_creation(self):
        """Test exit plan creation."""
        plan = ExitPlan(
            provider_name="Test Provider",
            services_in_scope=["compute", "storage"],
        )
        assert plan.plan_id.startswith("EXP-")
        assert plan.status == ExitPlanStatus.DRAFT
        assert plan.created_date is not None


class TestExitExecution:
    """Test ExitExecution data structure."""

    def test_execution_creation(self):
        """Test exit execution creation."""
        execution = ExitExecution(
            plan_id="EXP-12345678",
            provider_name="Test Provider",
            trigger=ExitTrigger.PLANNED_TERMINATION,
        )
        assert execution.execution_id.startswith("EXE-")
        assert execution.triggered_date is not None


# =============================================================================
# Configuration Tests
# =============================================================================

class TestExitStrategiesConfig:
    """Test ExitStrategiesConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExitStrategiesConfig()
        assert config.default_transition_days == 90
        assert config.critical_transition_days == 180
        assert config.min_alternatives_for_critical == 2
        assert config.test_frequency_months == 12

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ExitStrategiesConfig(
            default_transition_days=60,
            min_alternatives_for_critical=3,
        )
        assert config.default_transition_days == 60
        assert config.min_alternatives_for_critical == 3


# =============================================================================
# Exit Plan Management Tests
# =============================================================================

class TestExitPlanManagement:
    """Test exit plan management functionality."""

    def test_create_exit_plan(self, manager):
        """Test creating exit plan."""
        plan = manager.create_exit_plan(
            provider_id="PRV-001",
            provider_name="Test Provider",
            services=["compute"],
            is_critical=False,
        )
        assert plan is not None
        assert plan.provider_name == "Test Provider"
        assert plan.total_duration_days == 90  # default_transition_days

    def test_create_critical_exit_plan(self, manager):
        """Test creating critical exit plan."""
        plan = manager.create_exit_plan(
            provider_id="PRV-002",
            provider_name="Critical Provider",
            services=["trading"],
            is_critical=True,
        )
        assert plan.is_critical_provider is True
        assert plan.total_duration_days == 180  # critical_transition_days

    def test_create_exit_plan_with_default_tasks(self, manager):
        """Test that default tasks are created."""
        plan = manager.create_exit_plan(
            provider_id="PRV-001",
            provider_name="Test Provider",
            services=["compute"],
        )
        tasks = manager.get_tasks_for_plan(plan.plan_id)
        assert len(tasks) > 10  # Default tasks created

    def test_get_exit_plan(self, manager, sample_plan):
        """Test getting exit plan by ID."""
        retrieved = manager.get_exit_plan(sample_plan.plan_id)
        assert retrieved is not None
        assert retrieved.plan_id == sample_plan.plan_id

    def test_get_exit_plan_for_provider(self, manager, sample_plan):
        """Test getting exit plan for provider."""
        retrieved = manager.get_exit_plan_for_provider("PRV-001")
        assert retrieved is not None
        assert retrieved.provider_id == "PRV-001"

    def test_approve_exit_plan(self, manager, sample_plan):
        """Test approving exit plan."""
        approved = manager.approve_exit_plan(
            sample_plan.plan_id,
            approved_by="Risk Manager",
        )
        assert approved.status == ExitPlanStatus.APPROVED
        assert approved.approved_by == "Risk Manager"

    def test_get_critical_exit_plans(self, manager):
        """Test getting critical exit plans."""
        manager.create_exit_plan(
            provider_id="PRV-CRIT",
            provider_name="Critical",
            services=["trading"],
            is_critical=True,
        )
        manager.create_exit_plan(
            provider_id="PRV-STD",
            provider_name="Standard",
            services=["email"],
            is_critical=False,
        )

        critical = manager.get_critical_exit_plans()
        assert len(critical) == 1
        assert critical[0].provider_name == "Critical"


# =============================================================================
# Alternative Provider Tests
# =============================================================================

class TestAlternativeProviderManagement:
    """Test alternative provider management."""

    def test_add_alternative_provider(self, manager, sample_plan):
        """Test adding alternative provider."""
        alt = manager.add_alternative_provider(
            plan_id=sample_plan.plan_id,
            provider_name="Alternative Cloud",
            provider_country="DE",
            capability_match_pct=85.0,
        )
        assert alt is not None
        assert alt.provider_name == "Alternative Cloud"

    def test_evaluate_alternative(self, manager, sample_plan):
        """Test evaluating alternative provider."""
        alt = manager.add_alternative_provider(
            plan_id=sample_plan.plan_id,
            provider_name="To Evaluate",
            provider_country="FR",
        )

        evaluated = manager.evaluate_alternative(
            alt.alternative_id,
            technical_score=85,
            commercial_score=80,
            compliance_score=90,
            evaluation_notes="Good fit overall",
        )
        assert evaluated.status == AlternativeProviderStatus.EVALUATED
        assert evaluated.overall_score == 85 * 0.4 + 80 * 0.3 + 90 * 0.3

    def test_qualify_alternative(self, manager, sample_plan):
        """Test qualifying alternative provider."""
        alt = manager.add_alternative_provider(
            plan_id=sample_plan.plan_id,
            provider_name="To Qualify",
        )

        qualified = manager.qualify_alternative(alt.alternative_id)
        assert qualified.status == AlternativeProviderStatus.QUALIFIED

    def test_get_alternatives_for_plan(self, manager, plan_with_alternatives):
        """Test getting alternatives for plan."""
        alternatives = manager.get_alternatives_for_plan(plan_with_alternatives.plan_id)
        assert len(alternatives) == 2

    def test_get_best_alternative(self, manager, sample_plan):
        """Test getting best-scored alternative."""
        manager.add_alternative_provider(
            plan_id=sample_plan.plan_id,
            provider_name="Lower Score",
            capability_match_pct=70.0,
        )
        alt2 = manager.add_alternative_provider(
            plan_id=sample_plan.plan_id,
            provider_name="Higher Score",
            capability_match_pct=90.0,
        )

        # Evaluate to set scores
        manager.evaluate_alternative(
            alt2.alternative_id,
            technical_score=90,
            commercial_score=85,
            compliance_score=95,
        )

        best = manager.get_best_alternative(sample_plan.plan_id)
        assert best.provider_name == "Higher Score"

    def test_select_target_provider(self, manager, sample_plan):
        """Test selecting target provider."""
        alt = manager.add_alternative_provider(
            plan_id=sample_plan.plan_id,
            provider_name="Selected Provider",
        )

        updated = manager.select_target_provider(
            sample_plan.plan_id,
            alt.alternative_id,
        )
        assert updated.target_provider_name == "Selected Provider"


# =============================================================================
# Task Management Tests
# =============================================================================

class TestTaskManagement:
    """Test task management functionality."""

    def test_add_transition_task(self, manager, sample_plan):
        """Test adding transition task."""
        task = manager.add_transition_task(
            plan_id=sample_plan.plan_id,
            task_name="Custom Task",
            description="Custom task description",
            phase=ExitPhase.TRANSITION,
        )
        assert task is not None
        assert task.task_name == "Custom Task"

    def test_update_task_status(self, manager, sample_plan):
        """Test updating task status."""
        tasks = manager.get_tasks_for_plan(sample_plan.plan_id)
        task = tasks[0]

        updated = manager.update_task_status(
            task.task_id,
            status="in_progress",
            completion_pct=50.0,
        )
        assert updated.status == "in_progress"
        assert updated.completion_pct == 50.0
        assert updated.actual_start_date is not None

    def test_complete_task(self, manager, sample_plan):
        """Test completing task."""
        tasks = manager.get_tasks_for_plan(sample_plan.plan_id)
        task = tasks[0]

        completed = manager.update_task_status(
            task.task_id,
            status="completed",
        )
        assert completed.completion_pct == 100.0
        assert completed.actual_end_date is not None

    def test_get_tasks_by_phase(self, manager, sample_plan):
        """Test getting tasks by phase."""
        planning_tasks = manager.get_tasks_by_phase(
            sample_plan.plan_id,
            ExitPhase.PLANNING,
        )
        assert len(planning_tasks) > 0
        assert all(t.phase == ExitPhase.PLANNING for t in planning_tasks)


# =============================================================================
# Risk and Cost Tests
# =============================================================================

class TestRiskAndCostManagement:
    """Test risk and cost management."""

    def test_add_exit_risk(self, manager, sample_plan):
        """Test adding exit risk."""
        risk = manager.add_exit_risk(
            plan_id=sample_plan.plan_id,
            risk_name="Data Migration Risk",
            description="Risk of data loss during migration",
            category="technical",
            likelihood=4,
            impact=4,
        )
        assert risk is not None
        assert risk.risk_level == RiskLevel.HIGH

    def test_get_high_risks(self, manager, sample_plan):
        """Test getting high risks."""
        manager.add_exit_risk(
            plan_id=sample_plan.plan_id,
            risk_name="High Risk",
            description="Critical risk",
            category="technical",
            likelihood=5,
            impact=5,
        )
        manager.add_exit_risk(
            plan_id=sample_plan.plan_id,
            risk_name="Low Risk",
            description="Minor risk",
            category="operational",
            likelihood=1,
            impact=2,
        )

        high_risks = manager.get_high_risks(sample_plan.plan_id)
        assert len(high_risks) == 1

    def test_add_cost_estimate(self, manager, sample_plan):
        """Test adding cost estimate."""
        cost = manager.add_cost_estimate(
            plan_id=sample_plan.plan_id,
            category="transition",
            amount_eur=50000,
            description="Professional services for migration",
        )
        assert cost is not None
        assert cost.estimated_amount_eur == 50000

    def test_total_cost_calculation(self, manager, sample_plan):
        """Test total cost calculation with contingency."""
        manager.add_cost_estimate(
            plan_id=sample_plan.plan_id,
            category="transition",
            amount_eur=100000,
            description="Migration costs",
        )
        manager.add_cost_estimate(
            plan_id=sample_plan.plan_id,
            category="parallel_run",
            amount_eur=50000,
            description="Parallel run costs",
        )

        plan = manager.get_exit_plan(sample_plan.plan_id)
        # 150000 + 20% contingency = 180000
        assert plan.total_estimated_cost_eur == 180000


# =============================================================================
# Exit Readiness Tests
# =============================================================================

class TestExitReadiness:
    """Test exit readiness assessment."""

    def test_assess_exit_readiness(self, manager, sample_plan):
        """Test basic readiness assessment."""
        assessment = manager.assess_exit_readiness(
            sample_plan.plan_id,
            assessed_by="Compliance Officer",
        )
        assert assessment is not None
        assert "documentation_score" in assessment.__dict__
        assert "alternatives_score" in assessment.__dict__

    def test_readiness_level_determination(self, manager, sample_plan):
        """Test readiness level determination."""
        # Add alternatives to improve score
        manager.add_alternative_provider(
            plan_id=sample_plan.plan_id,
            provider_name="Alt 1",
        )
        manager.add_alternative_provider(
            plan_id=sample_plan.plan_id,
            provider_name="Alt 2",
        )

        assessment = manager.assess_exit_readiness(sample_plan.plan_id)
        assert assessment.readiness_level in ReadinessLevel


# =============================================================================
# Exit Execution Tests
# =============================================================================

class TestExitExecution:
    """Test exit execution functionality."""

    def test_trigger_exit(self, manager, sample_plan):
        """Test triggering exit execution."""
        # Approve plan first
        manager.approve_exit_plan(sample_plan.plan_id, "Manager")

        execution = manager.trigger_exit(
            plan_id=sample_plan.plan_id,
            trigger=ExitTrigger.PLANNED_TERMINATION,
            trigger_reason="Contract expiry",
            triggered_by="Operations",
        )
        assert execution is not None
        assert execution.status == "initiated"
        assert execution.trigger == ExitTrigger.PLANNED_TERMINATION

    def test_trigger_unapproved_plan_fails(self, manager, sample_plan):
        """Test that triggering unapproved plan fails."""
        execution = manager.trigger_exit(
            plan_id=sample_plan.plan_id,
            trigger=ExitTrigger.PLANNED_TERMINATION,
            trigger_reason="Test",
        )
        assert execution is None

    def test_update_execution_progress(self, manager, sample_plan):
        """Test updating execution progress."""
        manager.approve_exit_plan(sample_plan.plan_id, "Manager")
        execution = manager.trigger_exit(
            plan_id=sample_plan.plan_id,
            trigger=ExitTrigger.PLANNED_TERMINATION,
            trigger_reason="Test",
        )

        updated = manager.update_execution_progress(
            execution.execution_id,
            phase=ExitPhase.TRANSITION,
            tasks_completed=5,
        )
        assert updated.current_phase == ExitPhase.TRANSITION
        assert updated.tasks_completed == 5

    def test_complete_exit(self, manager, sample_plan):
        """Test completing exit."""
        manager.approve_exit_plan(sample_plan.plan_id, "Manager")
        execution = manager.trigger_exit(
            plan_id=sample_plan.plan_id,
            trigger=ExitTrigger.PLANNED_TERMINATION,
            trigger_reason="Test",
        )

        completed = manager.complete_exit(execution.execution_id)
        assert completed.status == "completed"
        assert completed.current_phase == ExitPhase.COMPLETED

        # Plan should also be marked completed
        plan = manager.get_exit_plan(sample_plan.plan_id)
        assert plan.status == ExitPlanStatus.COMPLETED


# =============================================================================
# Testing and Reporting Tests
# =============================================================================

class TestTestingAndReporting:
    """Test testing and reporting functionality."""

    def test_record_exit_test(self, manager, sample_plan):
        """Test recording exit plan test."""
        plan = manager.record_exit_test(
            plan_id=sample_plan.plan_id,
            test_type="tabletop",
            test_date=datetime.now(timezone.utc).isoformat(),
            success=True,
            findings=["Process works well"],
            tested_by="Test Team",
        )
        assert plan.last_test_date is not None
        assert len(plan.test_results) == 1

    def test_get_exit_strategy_status(self, manager, sample_plan):
        """Test getting exit strategy status."""
        status = manager.get_exit_strategy_status()
        assert "plans" in status
        assert "alternatives" in status
        assert "executions" in status
        assert "compliance_indicators" in status


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_exit_strategies(self):
        """Test factory function."""
        manager = create_exit_strategies()
        assert manager is not None
        assert isinstance(manager, DORAExitStrategies)

    def test_create_with_config(self):
        """Test factory function with config."""
        config = ExitStrategiesConfig(default_transition_days=60)
        manager = create_exit_strategies(config=config)
        assert manager.config.default_transition_days == 60

    def test_get_exit_triggers(self):
        """Test getting exit triggers."""
        triggers = get_exit_triggers()
        assert len(triggers) == 10
        assert ExitTrigger.PLANNED_TERMINATION in triggers

    def test_get_exit_phases(self):
        """Test getting exit phases."""
        phases = get_exit_phases()
        assert len(phases) == 8
        assert ExitPhase.PLANNING in phases

    def test_get_transition_types(self):
        """Test getting transition types."""
        types = get_transition_types()
        assert len(types) == 4
        assert TransitionType.TO_ALTERNATIVE_PROVIDER in types


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_exit_lifecycle(self, manager):
        """Test full exit plan lifecycle."""
        # 1. Create plan
        plan = manager.create_exit_plan(
            provider_id="PRV-FULL",
            provider_name="Full Lifecycle Provider",
            services=["compute", "storage"],
            is_critical=True,
        )

        # 2. Add alternatives
        alt1 = manager.add_alternative_provider(
            plan_id=plan.plan_id,
            provider_name="Primary Alternative",
            provider_country="DE",
        )
        manager.add_alternative_provider(
            plan_id=plan.plan_id,
            provider_name="Backup Alternative",
            provider_country="FR",
        )

        # 3. Evaluate and select
        manager.evaluate_alternative(
            alt1.alternative_id,
            technical_score=90,
            commercial_score=85,
            compliance_score=95,
        )
        manager.qualify_alternative(alt1.alternative_id)
        manager.select_target_provider(plan.plan_id, alt1.alternative_id)

        # 4. Add risks and costs
        manager.add_exit_risk(
            plan_id=plan.plan_id,
            risk_name="Data Migration Risk",
            description="Risk during data transfer",
            category="technical",
            likelihood=3,
            impact=4,
        )
        manager.add_cost_estimate(
            plan_id=plan.plan_id,
            category="transition",
            amount_eur=100000,
            description="Migration costs",
        )

        # 5. Assess readiness
        assessment = manager.assess_exit_readiness(plan.plan_id)
        assert assessment.alternatives_score > 0

        # 6. Approve plan
        manager.approve_exit_plan(plan.plan_id, "Risk Manager")

        # 7. Test plan
        manager.record_exit_test(
            plan_id=plan.plan_id,
            test_type="simulation",
            test_date=datetime.now(timezone.utc).isoformat(),
            success=True,
        )

        # 8. Trigger execution
        execution = manager.trigger_exit(
            plan_id=plan.plan_id,
            trigger=ExitTrigger.PLANNED_TERMINATION,
            trigger_reason="Contract renewal not pursued",
        )
        assert execution is not None

        # 9. Progress through phases
        manager.update_execution_progress(
            execution.execution_id,
            phase=ExitPhase.TRANSITION,
            tasks_completed=10,
        )

        # 10. Complete exit
        completed = manager.complete_exit(execution.execution_id)
        assert completed.status == "completed"

        # Verify final state
        final_plan = manager.get_exit_plan(plan.plan_id)
        assert final_plan.status == ExitPlanStatus.COMPLETED
