# -*- coding: utf-8 -*-
"""
Tests for MiFID II Phase 6 Business Continuity Plan Module (RTS 6 Article 3).

Comprehensive tests covering:
- ScenarioCategory, ImpactLevel, LikelihoodLevel, RecoveryStatus, AlertLevel enums
- EmergencyContact, RecoveryStep, RecoveryProcedure data classes
- BCPScenario functionality and risk scoring
- BCPIncident tracking
- BusinessContinuityPlan lifecycle and management
- Factory functions and standard scenario templates

Coverage target: 100%
"""

import json
import pytest
import time
import tempfile
import os
from datetime import date, timedelta
from decimal import Decimal

from services.compliance.bcp import (
    # Enums
    ScenarioCategory,
    ImpactLevel,
    LikelihoodLevel,
    RecoveryStatus,
    AlertLevel,
    # Data classes
    EmergencyContact,
    RecoveryStep,
    RecoveryProcedure,
    BCPScenario,
    BCPIncident,
    BusinessContinuityPlan,
    # Factory functions
    create_business_continuity_plan,
    load_bcp_from_file,
    save_bcp_to_file,
    # Templates
    get_standard_bcp_scenarios,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestScenarioCategory:
    """Tests for ScenarioCategory enum."""

    def test_all_categories_defined(self):
        """Verify all BCP categories exist."""
        assert ScenarioCategory.SYSTEM_FAILURE.value == "system_failure"
        assert ScenarioCategory.NETWORK_FAILURE.value == "network_failure"
        assert ScenarioCategory.DATA_CENTER.value == "data_center"
        assert ScenarioCategory.MARKET_DATA.value == "market_data"
        assert ScenarioCategory.EXCHANGE_CONNECTIVITY.value == "exchange_connectivity"
        assert ScenarioCategory.CYBERSECURITY.value == "cybersecurity"
        assert ScenarioCategory.PERSONNEL.value == "personnel"
        assert ScenarioCategory.THIRD_PARTY.value == "third_party"
        assert ScenarioCategory.NATURAL_DISASTER.value == "natural_disaster"
        assert ScenarioCategory.PANDEMIC.value == "pandemic"
        assert ScenarioCategory.REGULATORY.value == "regulatory"
        assert ScenarioCategory.ALGORITHM_MALFUNCTION.value == "algorithm_malfunction"
        assert ScenarioCategory.MARKET_DISRUPTION.value == "market_disruption"

    def test_category_count(self):
        """Test expected number of categories."""
        assert len(ScenarioCategory) == 13

    def test_category_from_value(self):
        """Test enum creation from string value."""
        cat = ScenarioCategory("system_failure")
        assert cat == ScenarioCategory.SYSTEM_FAILURE


class TestImpactLevel:
    """Tests for ImpactLevel enum."""

    def test_all_levels_defined(self):
        """Verify all impact levels exist."""
        assert ImpactLevel.CRITICAL.value == "critical"
        assert ImpactLevel.HIGH.value == "high"
        assert ImpactLevel.MEDIUM.value == "medium"
        assert ImpactLevel.LOW.value == "low"
        assert ImpactLevel.MINIMAL.value == "minimal"

    def test_level_count(self):
        """Test expected number of levels."""
        assert len(ImpactLevel) == 5


class TestLikelihoodLevel:
    """Tests for LikelihoodLevel enum."""

    def test_all_levels_defined(self):
        """Verify all likelihood levels exist."""
        assert LikelihoodLevel.VERY_HIGH.value == "very_high"
        assert LikelihoodLevel.HIGH.value == "high"
        assert LikelihoodLevel.MEDIUM.value == "medium"
        assert LikelihoodLevel.LOW.value == "low"
        assert LikelihoodLevel.VERY_LOW.value == "very_low"

    def test_level_count(self):
        """Test expected number of levels."""
        assert len(LikelihoodLevel) == 5


class TestRecoveryStatus:
    """Tests for RecoveryStatus enum."""

    def test_all_statuses_defined(self):
        """Verify all recovery statuses exist."""
        assert RecoveryStatus.NOT_TESTED.value == "not_tested"
        assert RecoveryStatus.TESTED_PASS.value == "tested_pass"
        assert RecoveryStatus.TESTED_FAIL.value == "tested_fail"
        assert RecoveryStatus.PARTIALLY_TESTED.value == "partially_tested"
        assert RecoveryStatus.IN_PROGRESS.value == "in_progress"
        assert RecoveryStatus.COMPLETED.value == "completed"


class TestAlertLevel:
    """Tests for AlertLevel enum."""

    def test_all_levels_defined(self):
        """Verify all alert levels exist."""
        assert AlertLevel.GREEN.value == "green"
        assert AlertLevel.AMBER.value == "amber"
        assert AlertLevel.RED.value == "red"
        assert AlertLevel.BLACK.value == "black"


# =============================================================================
# EmergencyContact Tests
# =============================================================================


class TestEmergencyContact:
    """Tests for EmergencyContact data class."""

    def test_contact_creation_defaults(self):
        """Test contact creation with defaults."""
        contact = EmergencyContact()
        assert contact.contact_id
        assert contact.name == ""
        assert contact.role == ""
        assert contact.available_hours == "24/7"
        assert contact.is_primary is False
        assert contact.categories == []

    def test_contact_creation_with_values(self):
        """Test contact with all values."""
        contact = EmergencyContact(
            name="John Smith",
            role="Head of IT",
            email="john.smith@firm.com",
            phone_primary="+1-555-123-4567",
            phone_secondary="+1-555-987-6543",
            available_hours="Business hours",
            is_primary=True,
            categories=[ScenarioCategory.SYSTEM_FAILURE, ScenarioCategory.NETWORK_FAILURE],
            notes="Primary contact for IT issues",
        )
        assert contact.name == "John Smith"
        assert contact.is_primary is True
        assert len(contact.categories) == 2

    def test_contact_to_dict(self):
        """Test contact serialization."""
        contact = EmergencyContact(
            name="Jane Doe",
            role="Compliance Officer",
            email="jane.doe@firm.com",
            is_primary=True,
            categories=[ScenarioCategory.REGULATORY],
        )
        data = contact.to_dict()
        assert data["name"] == "Jane Doe"
        assert data["is_primary"] is True
        assert data["categories"] == ["regulatory"]

    def test_contact_from_dict(self):
        """Test contact deserialization."""
        data = {
            "contact_id": "contact-123",
            "name": "Bob Wilson",
            "role": "Trading Manager",
            "email": "bob@firm.com",
            "phone_primary": "+1-555-111-2222",
            "is_primary": False,
            "categories": ["system_failure", "algorithm_malfunction"],
        }
        contact = EmergencyContact.from_dict(data)
        assert contact.contact_id == "contact-123"
        assert contact.name == "Bob Wilson"
        assert ScenarioCategory.SYSTEM_FAILURE in contact.categories


# =============================================================================
# RecoveryStep Tests
# =============================================================================


class TestRecoveryStep:
    """Tests for RecoveryStep data class."""

    def test_step_creation_defaults(self):
        """Test step creation with defaults."""
        step = RecoveryStep()
        assert step.step_number == 0
        assert step.action == ""
        assert step.expected_duration_minutes == 0

    def test_step_creation_with_values(self):
        """Test step with all values."""
        step = RecoveryStep(
            step_number=1,
            action="Switch to backup server",
            responsible_role="IT Operations",
            expected_duration_minutes=5,
            success_criteria="Backup server responds to health check",
            fallback_action="Contact vendor support",
        )
        assert step.step_number == 1
        assert step.expected_duration_minutes == 5
        assert step.success_criteria == "Backup server responds to health check"

    def test_step_to_dict(self):
        """Test step serialization."""
        step = RecoveryStep(
            step_number=2,
            action="Verify connectivity",
            expected_duration_minutes=3,
        )
        data = step.to_dict()
        assert data["step_number"] == 2
        assert data["expected_duration_minutes"] == 3

    def test_step_from_dict(self):
        """Test step deserialization."""
        data = {
            "step_number": 3,
            "action": "Resume trading",
            "responsible_role": "Trading Desk",
            "expected_duration_minutes": 2,
        }
        step = RecoveryStep.from_dict(data)
        assert step.step_number == 3
        assert step.action == "Resume trading"


# =============================================================================
# RecoveryProcedure Tests
# =============================================================================


class TestRecoveryProcedure:
    """Tests for RecoveryProcedure data class."""

    def test_procedure_creation_defaults(self):
        """Test procedure creation with defaults."""
        procedure = RecoveryProcedure()
        assert procedure.procedure_id
        assert procedure.name == ""
        assert procedure.steps == []
        assert procedure.status == RecoveryStatus.NOT_TESTED

    def test_procedure_creation_with_values(self):
        """Test procedure with all values."""
        procedure = RecoveryProcedure(
            name="System Failover",
            description="Failover to backup system",
            recovery_time_objective_minutes=15,
            recovery_point_objective_minutes=1,
            owner="IT Manager",
            steps=[
                RecoveryStep(step_number=1, action="Stop primary", expected_duration_minutes=2),
                RecoveryStep(step_number=2, action="Start backup", expected_duration_minutes=5),
            ],
        )
        assert procedure.name == "System Failover"
        assert procedure.recovery_time_objective_minutes == 15
        assert len(procedure.steps) == 2

    def test_total_estimated_duration(self):
        """Test total duration calculation."""
        procedure = RecoveryProcedure(
            steps=[
                RecoveryStep(step_number=1, expected_duration_minutes=5),
                RecoveryStep(step_number=2, expected_duration_minutes=10),
                RecoveryStep(step_number=3, expected_duration_minutes=3),
            ],
        )
        assert procedure.total_estimated_duration() == 18

    def test_meets_rto_true(self):
        """Test meets_rto when within RTO."""
        procedure = RecoveryProcedure(
            recovery_time_objective_minutes=20,
            steps=[
                RecoveryStep(step_number=1, expected_duration_minutes=5),
                RecoveryStep(step_number=2, expected_duration_minutes=10),
            ],
        )
        assert procedure.meets_rto() is True

    def test_meets_rto_false(self):
        """Test meets_rto when exceeds RTO."""
        procedure = RecoveryProcedure(
            recovery_time_objective_minutes=10,
            steps=[
                RecoveryStep(step_number=1, expected_duration_minutes=5),
                RecoveryStep(step_number=2, expected_duration_minutes=10),
            ],
        )
        assert procedure.meets_rto() is False

    def test_needs_testing_no_test(self):
        """Test needs_testing when never tested."""
        procedure = RecoveryProcedure()
        assert procedure.needs_testing() is True

    def test_needs_testing_recent(self):
        """Test needs_testing when recently tested."""
        procedure = RecoveryProcedure(
            last_test_date=date.today() - timedelta(days=30),
        )
        assert procedure.needs_testing() is False

    def test_needs_testing_overdue(self):
        """Test needs_testing when test overdue."""
        procedure = RecoveryProcedure(
            last_test_date=date.today() - timedelta(days=400),
        )
        assert procedure.needs_testing() is True

    def test_procedure_to_dict(self):
        """Test procedure serialization."""
        procedure = RecoveryProcedure(
            name="Test Procedure",
            recovery_time_objective_minutes=15,
            status=RecoveryStatus.TESTED_PASS,
            last_test_date=date(2024, 6, 1),
        )
        data = procedure.to_dict()
        assert data["name"] == "Test Procedure"
        assert data["status"] == "tested_pass"
        assert data["last_test_date"] == "2024-06-01"

    def test_procedure_from_dict(self):
        """Test procedure deserialization."""
        data = {
            "procedure_id": "proc-123",
            "name": "Failover",
            "recovery_time_objective_minutes": 10,
            "status": "tested_fail",
            "steps": [
                {"step_number": 1, "action": "Step 1", "expected_duration_minutes": 5},
            ],
        }
        procedure = RecoveryProcedure.from_dict(data)
        assert procedure.procedure_id == "proc-123"
        assert procedure.status == RecoveryStatus.TESTED_FAIL
        assert len(procedure.steps) == 1


# =============================================================================
# BCPScenario Tests
# =============================================================================


class TestBCPScenario:
    """Tests for BCPScenario data class."""

    def test_scenario_creation_defaults(self):
        """Test scenario creation with defaults."""
        scenario = BCPScenario()
        assert scenario.scenario_id
        assert scenario.category == ScenarioCategory.SYSTEM_FAILURE
        assert scenario.impact == ImpactLevel.MEDIUM
        assert scenario.likelihood == LikelihoodLevel.MEDIUM
        assert scenario.drill_frequency_days == 365

    def test_scenario_creation_with_values(self):
        """Test scenario with all values."""
        scenario = BCPScenario(
            name="Primary System Failure",
            description="Complete failure of primary trading system",
            category=ScenarioCategory.SYSTEM_FAILURE,
            impact=ImpactLevel.CRITICAL,
            likelihood=LikelihoodLevel.MEDIUM,
            immediate_actions=["Activate kill switch", "Notify compliance"],
            responsible_roles=["Head of IT", "Trading Manager"],
            trading_halt_threshold="Immediate on detection",
            owner="IT Manager",
        )
        assert scenario.name == "Primary System Failure"
        assert scenario.impact == ImpactLevel.CRITICAL
        assert len(scenario.immediate_actions) == 2

    def test_risk_score_calculation(self):
        """Test risk score is calculated correctly."""
        # Critical (5) * High (4) = 20
        scenario = BCPScenario(
            impact=ImpactLevel.CRITICAL,
            likelihood=LikelihoodLevel.HIGH,
        )
        assert scenario.risk_score == 20.0

        # Low (2) * Very Low (1) = 2
        scenario2 = BCPScenario(
            impact=ImpactLevel.LOW,
            likelihood=LikelihoodLevel.VERY_LOW,
        )
        assert scenario2.risk_score == 2.0

        # Medium (3) * Medium (3) = 9
        scenario3 = BCPScenario(
            impact=ImpactLevel.MEDIUM,
            likelihood=LikelihoodLevel.MEDIUM,
        )
        assert scenario3.risk_score == 9.0

    def test_needs_drill_never_tested(self):
        """Test needs_drill when never tested."""
        scenario = BCPScenario()
        assert scenario.needs_drill() is True

    def test_needs_drill_recent(self):
        """Test needs_drill when recently tested."""
        scenario = BCPScenario(
            last_drill_date=date.today() - timedelta(days=30),
        )
        assert scenario.needs_drill() is False

    def test_needs_drill_overdue(self):
        """Test needs_drill when overdue."""
        scenario = BCPScenario(
            last_drill_date=date.today() - timedelta(days=400),
            drill_frequency_days=365,
        )
        assert scenario.needs_drill() is True

    def test_record_drill_success(self):
        """Test recording a successful drill."""
        scenario = BCPScenario()
        scenario.recovery_procedure = RecoveryProcedure(name="Test")

        scenario.record_drill("Test passed successfully")

        assert scenario.last_drill_date == date.today()
        assert scenario.recovery_procedure.status == RecoveryStatus.TESTED_PASS
        assert scenario.recovery_procedure.last_test_result == "Test passed successfully"

    def test_record_drill_failure(self):
        """Test recording a failed drill."""
        scenario = BCPScenario()
        scenario.recovery_procedure = RecoveryProcedure(name="Test")

        scenario.record_drill("Test failed - timeout exceeded")

        assert scenario.recovery_procedure.status == RecoveryStatus.TESTED_FAIL

    def test_scenario_to_dict(self):
        """Test scenario serialization."""
        scenario = BCPScenario(
            name="Test Scenario",
            category=ScenarioCategory.NETWORK_FAILURE,
            impact=ImpactLevel.HIGH,
            likelihood=LikelihoodLevel.LOW,
        )
        data = scenario.to_dict()
        assert data["name"] == "Test Scenario"
        assert data["category"] == "network_failure"
        assert data["impact"] == "high"
        assert data["likelihood"] == "low"
        assert data["risk_score"] == 8.0  # HIGH(4) * LOW(2)

    def test_scenario_from_dict(self):
        """Test scenario deserialization."""
        data = {
            "scenario_id": "bcp-123",
            "name": "Data Center Failure",
            "category": "data_center",
            "impact": "critical",
            "likelihood": "very_low",
            "immediate_actions": ["Activate DR site"],
        }
        scenario = BCPScenario.from_dict(data)
        assert scenario.scenario_id == "bcp-123"
        assert scenario.category == ScenarioCategory.DATA_CENTER
        assert scenario.impact == ImpactLevel.CRITICAL
        assert len(scenario.immediate_actions) == 1


# =============================================================================
# BCPIncident Tests
# =============================================================================


class TestBCPIncident:
    """Tests for BCPIncident data class."""

    def test_incident_creation_defaults(self):
        """Test incident creation with defaults."""
        incident = BCPIncident()
        assert incident.incident_id
        assert incident.alert_level == AlertLevel.GREEN
        assert incident.trading_halted is False
        assert incident.is_resolved is False

    def test_incident_creation_with_values(self):
        """Test incident with all values."""
        incident = BCPIncident(
            scenario_id="bcp-sys-001",
            alert_level=AlertLevel.RED,
            description="Primary system went offline",
            trading_halted=True,
            reported_by="IT Operations",
        )
        assert incident.scenario_id == "bcp-sys-001"
        assert incident.alert_level == AlertLevel.RED
        assert incident.trading_halted is True

    def test_is_resolved_false(self):
        """Test is_resolved when not resolved."""
        incident = BCPIncident()
        assert incident.is_resolved is False

    def test_is_resolved_true(self):
        """Test is_resolved when resolved."""
        incident = BCPIncident(
            timestamp_end_ns=time.time_ns(),
        )
        assert incident.is_resolved is True

    def test_resolve_incident(self):
        """Test resolving an incident."""
        incident = BCPIncident(
            description="Test incident",
        )
        time.sleep(0.01)  # Small delay to ensure measurable duration

        incident.resolve("IT Manager", "Hardware failure")

        assert incident.is_resolved is True
        assert incident.resolved_by == "IT Manager"
        assert incident.root_cause == "Hardware failure"
        assert incident.timestamp_end_ns is not None
        assert incident.recovery_duration_minutes is not None

    def test_incident_to_dict(self):
        """Test incident serialization."""
        incident = BCPIncident(
            scenario_id="bcp-001",
            alert_level=AlertLevel.AMBER,
            description="Test incident",
            reported_by="Operator",
        )
        data = incident.to_dict()
        assert data["scenario_id"] == "bcp-001"
        assert data["alert_level"] == "amber"
        assert data["reported_by"] == "Operator"

    def test_incident_from_dict(self):
        """Test incident deserialization."""
        data = {
            "incident_id": "inc-123",
            "scenario_id": "bcp-002",
            "alert_level": "red",
            "description": "Major outage",
            "trading_halted": True,
        }
        incident = BCPIncident.from_dict(data)
        assert incident.incident_id == "inc-123"
        assert incident.alert_level == AlertLevel.RED
        assert incident.trading_halted is True


# =============================================================================
# BusinessContinuityPlan Tests
# =============================================================================


class TestBusinessContinuityPlan:
    """Tests for BusinessContinuityPlan data class."""

    def test_plan_creation_defaults(self):
        """Test plan creation with defaults."""
        plan = BusinessContinuityPlan()
        assert plan.plan_id
        assert plan.version == "1.0.0"
        assert plan.current_alert_level == AlertLevel.GREEN
        assert plan.max_disruptions_before_halt == 3
        assert plan.effective_date == date.today()
        assert plan.review_date == date.today() + timedelta(days=365)

    def test_plan_creation_with_values(self):
        """Test plan with all values."""
        plan = BusinessContinuityPlan(
            version="2.0.0",
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            approved_by="CEO",
            max_disruptions_before_halt=5,
        )
        assert plan.version == "2.0.0"
        assert plan.firm_lei == "5493001KJTIIGC8Y1R12"
        assert plan.approved_by == "CEO"

    def test_add_scenario(self):
        """Test adding scenario."""
        plan = BusinessContinuityPlan()
        scenario = BCPScenario(name="Test Scenario")
        plan.add_scenario(scenario)

        assert len(plan.scenarios) == 1
        assert plan.scenarios[0].created_date == date.today()

    def test_get_scenario_found(self):
        """Test getting scenario by ID."""
        plan = BusinessContinuityPlan()
        scenario = BCPScenario(name="Test")
        plan.add_scenario(scenario)

        result = plan.get_scenario(scenario.scenario_id)
        assert result is not None
        assert result.name == "Test"

    def test_get_scenario_not_found(self):
        """Test getting non-existent scenario."""
        plan = BusinessContinuityPlan()
        result = plan.get_scenario("nonexistent")
        assert result is None

    def test_get_scenarios_by_category(self):
        """Test filtering scenarios by category."""
        plan = BusinessContinuityPlan()
        plan.add_scenario(BCPScenario(
            name="System 1",
            category=ScenarioCategory.SYSTEM_FAILURE,
        ))
        plan.add_scenario(BCPScenario(
            name="Network 1",
            category=ScenarioCategory.NETWORK_FAILURE,
        ))

        system_scenarios = plan.get_scenarios_by_category(ScenarioCategory.SYSTEM_FAILURE)
        assert len(system_scenarios) == 1
        assert system_scenarios[0].name == "System 1"

    def test_get_high_risk_scenarios(self):
        """Test getting high-risk scenarios."""
        plan = BusinessContinuityPlan()
        plan.add_scenario(BCPScenario(
            name="High Risk",
            impact=ImpactLevel.CRITICAL,
            likelihood=LikelihoodLevel.HIGH,  # Score = 20
        ))
        plan.add_scenario(BCPScenario(
            name="Low Risk",
            impact=ImpactLevel.LOW,
            likelihood=LikelihoodLevel.LOW,  # Score = 4
        ))

        high_risk = plan.get_high_risk_scenarios(min_score=15.0)
        assert len(high_risk) == 1
        assert high_risk[0].name == "High Risk"

    def test_get_scenarios_needing_drill(self):
        """Test getting scenarios needing drill."""
        plan = BusinessContinuityPlan()
        plan.add_scenario(BCPScenario(
            name="Never Tested",
        ))
        plan.add_scenario(BCPScenario(
            name="Recently Tested",
            last_drill_date=date.today() - timedelta(days=30),
        ))

        needing_drill = plan.get_scenarios_needing_drill()
        assert len(needing_drill) == 1
        assert needing_drill[0].name == "Never Tested"

    def test_add_contact(self):
        """Test adding contact."""
        plan = BusinessContinuityPlan()
        contact = EmergencyContact(name="John Doe", is_primary=True)
        plan.add_contact(contact)

        assert len(plan.contacts) == 1

    def test_get_primary_contacts(self):
        """Test getting primary contacts."""
        plan = BusinessContinuityPlan()
        plan.add_contact(EmergencyContact(name="Primary", is_primary=True))
        plan.add_contact(EmergencyContact(name="Secondary", is_primary=False))

        primaries = plan.get_primary_contacts()
        assert len(primaries) == 1
        assert primaries[0].name == "Primary"

    def test_get_contacts_for_category(self):
        """Test getting contacts for category."""
        plan = BusinessContinuityPlan()
        plan.add_contact(EmergencyContact(
            name="IT Contact",
            categories=[ScenarioCategory.SYSTEM_FAILURE],
        ))
        plan.add_contact(EmergencyContact(
            name="Compliance Contact",
            categories=[ScenarioCategory.REGULATORY],
        ))

        it_contacts = plan.get_contacts_for_category(ScenarioCategory.SYSTEM_FAILURE)
        assert len(it_contacts) == 1
        assert it_contacts[0].name == "IT Contact"

    def test_start_incident(self):
        """Test starting an incident."""
        plan = BusinessContinuityPlan()
        incident = plan.start_incident(
            scenario_id="bcp-001",
            description="System failure detected",
            reported_by="Monitoring",
            alert_level=AlertLevel.RED,
        )

        assert incident.scenario_id == "bcp-001"
        assert incident.alert_level == AlertLevel.RED
        assert len(plan.incidents) == 1
        assert plan.current_alert_level == AlertLevel.RED

    def test_resolve_incident(self):
        """Test resolving an incident."""
        plan = BusinessContinuityPlan()
        incident = plan.start_incident(
            scenario_id="bcp-001",
            description="Test incident",
            reported_by="Operator",
        )

        result = plan.resolve_incident(
            incident.incident_id,
            resolved_by="IT Manager",
            root_cause="Hardware failure",
            lessons_learned="Need better monitoring",
        )

        assert result is True
        assert plan.incidents[0].is_resolved is True
        assert plan.incidents[0].root_cause == "Hardware failure"
        assert plan.current_alert_level == AlertLevel.GREEN  # Reset when no active

    def test_resolve_incident_not_found(self):
        """Test resolving non-existent incident."""
        plan = BusinessContinuityPlan()
        result = plan.resolve_incident("nonexistent", "Manager")
        assert result is False

    def test_get_active_incidents(self):
        """Test getting active incidents."""
        plan = BusinessContinuityPlan()
        plan.start_incident("bcp-001", "Incident 1", "Operator")
        plan.start_incident("bcp-002", "Incident 2", "Operator")

        active = plan.get_active_incidents()
        assert len(active) == 2

        # Resolve one
        plan.resolve_incident(plan.incidents[0].incident_id, "Manager")
        active = plan.get_active_incidents()
        assert len(active) == 1

    def test_recent_disruption_count(self):
        """Test counting recent disruptions."""
        plan = BusinessContinuityPlan(disruption_window_hours=24)
        plan.start_incident("bcp-001", "Incident 1", "Operator")
        plan.start_incident("bcp-002", "Incident 2", "Operator")

        count = plan.recent_disruption_count()
        assert count == 2

    def test_should_halt_trading_false(self):
        """Test should_halt_trading when below threshold."""
        plan = BusinessContinuityPlan(max_disruptions_before_halt=5)
        plan.start_incident("bcp-001", "Incident", "Operator")

        should_halt, reason = plan.should_halt_trading()
        assert should_halt is False
        assert reason == ""

    def test_should_halt_trading_true(self):
        """Test should_halt_trading when at threshold."""
        plan = BusinessContinuityPlan(max_disruptions_before_halt=2)
        plan.start_incident("bcp-001", "Incident 1", "Operator")
        plan.start_incident("bcp-002", "Incident 2", "Operator")

        should_halt, reason = plan.should_halt_trading()
        assert should_halt is True
        assert "threshold exceeded" in reason.lower()

    def test_get_plan_summary(self):
        """Test plan summary generation."""
        plan = BusinessContinuityPlan(firm_lei="5493001KJTIIGC8Y1R12")
        plan.add_scenario(BCPScenario(
            category=ScenarioCategory.SYSTEM_FAILURE,
            impact=ImpactLevel.HIGH,
        ))
        plan.add_contact(EmergencyContact(name="Contact", is_primary=True))

        summary = plan.get_plan_summary()
        assert summary["total_scenarios"] == 1
        assert summary["total_contacts"] == 1
        assert summary["primary_contacts"] == 1
        assert summary["current_alert_level"] == "green"

    def test_get_drill_schedule(self):
        """Test drill schedule generation."""
        plan = BusinessContinuityPlan()
        plan.add_scenario(BCPScenario(
            name="Tested",
            last_drill_date=date.today() - timedelta(days=30),
            next_drill_due=date.today() + timedelta(days=335),
        ))
        plan.add_scenario(BCPScenario(
            name="Overdue",
            last_drill_date=None,
        ))

        schedule = plan.get_drill_schedule()
        assert len(schedule) == 2
        # Overdue should be first
        assert schedule[0]["overdue"] is True

    def test_generate_bcp_document(self):
        """Test BCP document generation."""
        plan = BusinessContinuityPlan(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            approved_by="CEO",
        )
        plan.add_scenario(BCPScenario(
            name="System Failure",
            category=ScenarioCategory.SYSTEM_FAILURE,
        ))
        plan.add_contact(EmergencyContact(
            name="IT Manager",
            role="Head of IT",
            is_primary=True,
        ))

        doc = plan.generate_bcp_document()
        assert "BUSINESS CONTINUITY PLAN" in doc
        assert "Test Firm" in doc
        assert "System Failure" in doc
        assert "IT Manager" in doc

    def test_validate_missing_lei(self):
        """Test validation catches missing LEI."""
        plan = BusinessContinuityPlan()
        errors = plan.validate()
        assert "Missing firm_lei" in errors

    def test_validate_missing_approval(self):
        """Test validation catches missing approval."""
        plan = BusinessContinuityPlan(firm_lei="5493001KJTIIGC8Y1R12")
        errors = plan.validate()
        assert any("Missing approved_by" in error for error in errors)

    def test_validate_missing_categories(self):
        """Test validation catches missing critical categories."""
        plan = BusinessContinuityPlan(
            firm_lei="5493001KJTIIGC8Y1R12",
            approved_by="CEO",
        )
        plan.add_contact(EmergencyContact(is_primary=True))

        errors = plan.validate()
        # Should require system_failure, network_failure, market_data, algorithm_malfunction
        assert any("system_failure" in e for e in errors)

    def test_validate_no_primary_contact(self):
        """Test validation catches missing primary contact."""
        plan = BusinessContinuityPlan(
            firm_lei="5493001KJTIIGC8Y1R12",
            approved_by="CEO",
        )
        errors = plan.validate()
        assert "No primary emergency contact defined" in errors

    def test_is_valid(self):
        """Test is_valid method."""
        plan = BusinessContinuityPlan()
        assert plan.is_valid() is False

    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization/deserialization roundtrip."""
        plan = BusinessContinuityPlan(
            version="2.0.0",
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
        )
        plan.add_scenario(BCPScenario(name="Test Scenario"))
        plan.add_contact(EmergencyContact(name="Test Contact"))

        data = plan.to_dict()
        restored = BusinessContinuityPlan.from_dict(data)

        assert restored.version == "2.0.0"
        assert restored.firm_lei == "5493001KJTIIGC8Y1R12"
        assert len(restored.scenarios) == 1
        assert len(restored.contacts) == 1

    def test_to_json_from_json_roundtrip(self):
        """Test JSON serialization roundtrip."""
        plan = BusinessContinuityPlan(firm_lei="5493001KJTIIGC8Y1R12")

        json_str = plan.to_json()
        restored = BusinessContinuityPlan.from_json(json_str)

        assert restored.firm_lei == "5493001KJTIIGC8Y1R12"


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_bcp_defaults(self):
        """Test creating BCP with defaults."""
        plan = create_business_continuity_plan(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
        )

        assert plan.firm_lei == "5493001KJTIIGC8Y1R12"
        assert plan.firm_name == "Test Firm"

    def test_create_bcp_with_standard_scenarios(self):
        """Test creating BCP with standard scenarios."""
        plan = create_business_continuity_plan(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            include_standard_scenarios=True,
        )

        # Should have scenarios from template
        assert len(plan.scenarios) > 0

        # Should have scenarios from different categories
        categories = {s.category for s in plan.scenarios}
        assert ScenarioCategory.SYSTEM_FAILURE in categories
        assert ScenarioCategory.ALGORITHM_MALFUNCTION in categories

    def test_create_bcp_without_scenarios(self):
        """Test creating BCP without standard scenarios."""
        plan = create_business_continuity_plan(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            include_standard_scenarios=False,
        )

        assert len(plan.scenarios) == 0


# =============================================================================
# Template Tests
# =============================================================================


class TestStandardScenarios:
    """Tests for standard BCP scenarios template."""

    def test_template_has_scenarios(self):
        """Test template returns scenarios."""
        scenarios = get_standard_bcp_scenarios()
        assert len(scenarios) > 0

    def test_template_covers_critical_categories(self):
        """Test template covers critical categories."""
        scenarios = get_standard_bcp_scenarios()
        categories = {s.category for s in scenarios}

        critical = {
            ScenarioCategory.SYSTEM_FAILURE,
            ScenarioCategory.ALGORITHM_MALFUNCTION,
            ScenarioCategory.MARKET_DATA,
            ScenarioCategory.NETWORK_FAILURE,
        }

        assert critical.issubset(categories)

    def test_template_scenarios_have_recovery_procedures(self):
        """Test scenarios have recovery procedures."""
        scenarios = get_standard_bcp_scenarios()

        # At least some should have procedures
        with_procedures = [s for s in scenarios if s.recovery_procedure is not None]
        assert len(with_procedures) > 0

    def test_template_scenarios_have_immediate_actions(self):
        """Test scenarios have immediate actions."""
        scenarios = get_standard_bcp_scenarios()

        for scenario in scenarios:
            assert len(scenario.immediate_actions) > 0, \
                f"Scenario {scenario.name} missing immediate actions"

    def test_template_scenario_ids_unique(self):
        """Test all scenario IDs are unique."""
        scenarios = get_standard_bcp_scenarios()
        ids = [s.scenario_id for s in scenarios]
        assert len(ids) == len(set(ids)), "Duplicate scenario IDs found"


# =============================================================================
# File I/O Tests
# =============================================================================


class TestFileIO:
    """Tests for file I/O operations."""

    def test_save_and_load_bcp(self):
        """Test saving and loading BCP from file."""
        plan = create_business_continuity_plan(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            include_standard_scenarios=False,
        )
        plan.add_scenario(BCPScenario(name="Test Scenario"))

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            file_path = f.name

        try:
            save_bcp_to_file(plan, file_path)
            loaded = load_bcp_from_file(file_path)

            assert loaded.firm_lei == "5493001KJTIIGC8Y1R12"
            assert len(loaded.scenarios) == 1
            assert loaded.scenarios[0].name == "Test Scenario"
        finally:
            os.unlink(file_path)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for BCP workflow."""

    def test_incident_response_workflow(self):
        """Test complete incident response workflow."""
        plan = create_business_continuity_plan(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            approved_by="CEO",
        )

        # Get a system failure scenario
        sys_scenarios = plan.get_scenarios_by_category(ScenarioCategory.SYSTEM_FAILURE)
        assert len(sys_scenarios) > 0
        scenario = sys_scenarios[0]

        # Start incident
        incident = plan.start_incident(
            scenario_id=scenario.scenario_id,
            description="Primary trading server unresponsive",
            reported_by="Monitoring System",
            alert_level=AlertLevel.RED,
        )

        assert plan.current_alert_level == AlertLevel.RED
        assert incident.trading_halted is False

        # Escalate - mark trading halted
        incident.trading_halted = True

        # Resolve incident
        plan.resolve_incident(
            incident.incident_id,
            resolved_by="IT Manager",
            root_cause="Memory exhaustion",
            lessons_learned="Need memory monitoring alerts",
        )

        assert plan.current_alert_level == AlertLevel.GREEN
        assert incident.is_resolved is True
        assert incident.recovery_duration_minutes is not None

    def test_drill_recording_workflow(self):
        """Test drill recording workflow."""
        plan = create_business_continuity_plan(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
        )

        # Get scenarios needing drill
        needing_drill = plan.get_scenarios_needing_drill()
        assert len(needing_drill) > 0

        scenario = needing_drill[0]
        initial_count = len(needing_drill)

        # Record drill
        scenario.record_drill("Drill completed successfully - RTO met")

        # Verify drill recorded
        assert scenario.last_drill_date == date.today()
        assert scenario.needs_drill() is False

        # Check fewer scenarios need drill now
        still_needing = plan.get_scenarios_needing_drill()
        assert len(still_needing) == initial_count - 1

    def test_halt_threshold_workflow(self):
        """Test trading halt threshold workflow."""
        plan = create_business_continuity_plan(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
        )
        plan.max_disruptions_before_halt = 3

        # Create incidents up to threshold
        for i in range(3):
            plan.start_incident(
                scenario_id=f"bcp-00{i}",
                description=f"Incident {i + 1}",
                reported_by="System",
            )

        # Check halt threshold
        should_halt, reason = plan.should_halt_trading()
        assert should_halt is True
        assert "3 incidents" in reason
