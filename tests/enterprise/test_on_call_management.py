# -*- coding: utf-8 -*-
"""
Comprehensive tests for 24/7 On-Call Management Service.

Tests on-call management per DORA Art. 30(2)(f) requirements.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from services.enterprise.on_call_management import (
    # Enums
    OnCallTier,
    EscalationLevel,
    ShiftType,
    IncidentPriority,
    # Data structures
    OnCallEngineer,
    OnCallShift,
    OnCallSchedule,
    EscalationPolicy,
    IncidentAssignment,
    OnCallMetrics,
    OnCallConfig,
    # Service
    OnCallManagementService,
    # Factory
    create_on_call_management,
)


# =============================================================================
# OnCallEngineer Tests
# =============================================================================


class TestOnCallEngineer:
    """Tests for OnCallEngineer dataclass."""

    def test_create_engineer(self) -> None:
        """Test creating an on-call engineer."""
        engineer = OnCallEngineer(
            engineer_id="eng-1",
            name="John Doe",
            email="john@example.com",
            phone="+1234567890",
            timezone="UTC",
            escalation_level=EscalationLevel.L1,
        )
        assert engineer.name == "John Doe"
        assert engineer.escalation_level == EscalationLevel.L1

    def test_can_take_shift_available(self) -> None:
        """Test can_take_shift for available engineer."""
        engineer = OnCallEngineer(
            engineer_id="eng-1",
            name="John Doe",
            email="john@example.com",
            phone="+1234567890",
            timezone="UTC",
            escalation_level=EscalationLevel.L1,
            is_available=True,
            current_hours_this_week=20.0,
            max_hours_per_week=40,
        )
        assert engineer.can_take_shift is True

    def test_can_take_shift_unavailable(self) -> None:
        """Test can_take_shift for unavailable engineer."""
        engineer = OnCallEngineer(
            engineer_id="eng-1",
            name="John Doe",
            email="john@example.com",
            phone="+1234567890",
            timezone="UTC",
            escalation_level=EscalationLevel.L1,
            is_available=False,
        )
        assert engineer.can_take_shift is False

    def test_can_take_shift_max_hours(self) -> None:
        """Test can_take_shift when max hours reached."""
        engineer = OnCallEngineer(
            engineer_id="eng-1",
            name="John Doe",
            email="john@example.com",
            phone="+1234567890",
            timezone="UTC",
            escalation_level=EscalationLevel.L1,
            is_available=True,
            current_hours_this_week=40.0,
            max_hours_per_week=40,
        )
        assert engineer.can_take_shift is False


# =============================================================================
# OnCallShift Tests
# =============================================================================


class TestOnCallShift:
    """Tests for OnCallShift dataclass."""

    def test_create_shift(self) -> None:
        """Test creating an on-call shift."""
        start = datetime.utcnow()
        end = start + timedelta(hours=8)
        shift = OnCallShift(
            shift_id="shift-1",
            engineer_id="eng-1",
            shift_type=ShiftType.PRIMARY,
            start_time=start,
            end_time=end,
            timezone="UTC",
        )
        assert shift.shift_type == ShiftType.PRIMARY
        assert shift.duration_hours == 8.0

    def test_is_active_true(self) -> None:
        """Test is_active for active shift."""
        start = datetime.utcnow() - timedelta(hours=2)
        end = datetime.utcnow() + timedelta(hours=6)
        shift = OnCallShift(
            shift_id="shift-1",
            engineer_id="eng-1",
            shift_type=ShiftType.PRIMARY,
            start_time=start,
            end_time=end,
            timezone="UTC",
        )
        assert shift.is_active is True

    def test_is_active_false_past(self) -> None:
        """Test is_active for past shift."""
        start = datetime.utcnow() - timedelta(days=1)
        end = datetime.utcnow() - timedelta(hours=16)
        shift = OnCallShift(
            shift_id="shift-1",
            engineer_id="eng-1",
            shift_type=ShiftType.PRIMARY,
            start_time=start,
            end_time=end,
            timezone="UTC",
        )
        assert shift.is_active is False

    def test_is_active_false_future(self) -> None:
        """Test is_active for future shift."""
        start = datetime.utcnow() + timedelta(days=1)
        end = start + timedelta(hours=8)
        shift = OnCallShift(
            shift_id="shift-1",
            engineer_id="eng-1",
            shift_type=ShiftType.PRIMARY,
            start_time=start,
            end_time=end,
            timezone="UTC",
        )
        assert shift.is_active is False


# =============================================================================
# OnCallSchedule Tests
# =============================================================================


class TestOnCallSchedule:
    """Tests for OnCallSchedule dataclass."""

    def test_create_schedule(self) -> None:
        """Test creating an on-call schedule."""
        schedule = OnCallSchedule(
            schedule_id="sched-1",
            name="24/7 Coverage",
            tier=OnCallTier.FULL,
            rotation_period_days=7,
        )
        assert schedule.name == "24/7 Coverage"
        assert schedule.tier == OnCallTier.FULL

    def test_get_current_on_call(self) -> None:
        """Test getting current on-call shifts."""
        schedule = OnCallSchedule(
            schedule_id="sched-1",
            name="Test",
            tier=OnCallTier.FULL,
            rotation_period_days=7,
        )
        # Add active shift
        active_shift = OnCallShift(
            shift_id="shift-1",
            engineer_id="eng-1",
            shift_type=ShiftType.PRIMARY,
            start_time=datetime.utcnow() - timedelta(hours=2),
            end_time=datetime.utcnow() + timedelta(hours=6),
            timezone="UTC",
        )
        schedule.shifts.append(active_shift)

        # Add inactive shift
        past_shift = OnCallShift(
            shift_id="shift-2",
            engineer_id="eng-2",
            shift_type=ShiftType.PRIMARY,
            start_time=datetime.utcnow() - timedelta(days=1),
            end_time=datetime.utcnow() - timedelta(hours=16),
            timezone="UTC",
        )
        schedule.shifts.append(past_shift)

        current = schedule.get_current_on_call()
        assert len(current) == 1
        assert current[0].shift_id == "shift-1"

    def test_get_primary_on_call(self) -> None:
        """Test getting primary on-call."""
        schedule = OnCallSchedule(
            schedule_id="sched-1",
            name="Test",
            tier=OnCallTier.FULL,
            rotation_period_days=7,
        )
        primary_shift = OnCallShift(
            shift_id="shift-1",
            engineer_id="eng-1",
            shift_type=ShiftType.PRIMARY,
            start_time=datetime.utcnow() - timedelta(hours=2),
            end_time=datetime.utcnow() + timedelta(hours=6),
            timezone="UTC",
        )
        secondary_shift = OnCallShift(
            shift_id="shift-2",
            engineer_id="eng-2",
            shift_type=ShiftType.SECONDARY,
            start_time=datetime.utcnow() - timedelta(hours=2),
            end_time=datetime.utcnow() + timedelta(hours=6),
            timezone="UTC",
        )
        schedule.shifts.extend([primary_shift, secondary_shift])

        primary = schedule.get_primary_on_call()
        assert primary is not None
        assert primary.shift_type == ShiftType.PRIMARY


# =============================================================================
# EscalationPolicy Tests
# =============================================================================


class TestEscalationPolicy:
    """Tests for EscalationPolicy dataclass."""

    def test_create_policy(self) -> None:
        """Test creating an escalation policy."""
        policy = EscalationPolicy(
            policy_id="policy-1",
            name="Standard Escalation",
            description="Standard escalation policy",
            levels=[
                {"level": "l1", "timeout_minutes": 15},
                {"level": "l2", "timeout_minutes": 30},
            ],
        )
        assert policy.name == "Standard Escalation"
        assert len(policy.levels) == 2

    def test_get_timeout_minutes(self) -> None:
        """Test getting timeout for escalation level."""
        policy = EscalationPolicy(
            policy_id="policy-1",
            name="Test",
            description="Test",
            levels=[
                {"level": "l1", "timeout_minutes": 15},
                {"level": "l2", "timeout_minutes": 30},
            ],
        )
        assert policy.get_timeout_minutes(EscalationLevel.L1) == 15
        assert policy.get_timeout_minutes(EscalationLevel.L2) == 30

    def test_get_timeout_minutes_default(self) -> None:
        """Test getting default timeout for unknown level."""
        policy = EscalationPolicy(
            policy_id="policy-1",
            name="Test",
            description="Test",
            levels=[{"level": "l1", "timeout_minutes": 15}],
        )
        # L3 not configured, should return default
        assert policy.get_timeout_minutes(EscalationLevel.L3) == 30


# =============================================================================
# IncidentAssignment Tests
# =============================================================================


class TestIncidentAssignment:
    """Tests for IncidentAssignment dataclass."""

    def test_create_assignment(self) -> None:
        """Test creating an incident assignment."""
        assignment = IncidentAssignment(
            assignment_id="assign-1",
            incident_id="inc-1",
            engineer_id="eng-1",
            priority=IncidentPriority.P1,
            escalation_level=EscalationLevel.L1,
            assigned_at=datetime.utcnow(),
        )
        assert assignment.priority == IncidentPriority.P1
        assert assignment.acknowledged_at is None

    def test_acknowledge(self) -> None:
        """Test acknowledging an incident."""
        assignment = IncidentAssignment(
            assignment_id="assign-1",
            incident_id="inc-1",
            engineer_id="eng-1",
            priority=IncidentPriority.P1,
            escalation_level=EscalationLevel.L1,
            assigned_at=datetime.utcnow() - timedelta(minutes=5),
        )
        assignment.acknowledge()
        assert assignment.acknowledged_at is not None
        assert assignment.response_time_seconds is not None
        assert assignment.response_time_seconds >= 300  # At least 5 minutes

    def test_resolve(self) -> None:
        """Test resolving an incident."""
        assignment = IncidentAssignment(
            assignment_id="assign-1",
            incident_id="inc-1",
            engineer_id="eng-1",
            priority=IncidentPriority.P2,
            escalation_level=EscalationLevel.L1,
            assigned_at=datetime.utcnow() - timedelta(hours=1),
        )
        assignment.resolve()
        assert assignment.resolved_at is not None
        assert assignment.resolution_time_seconds is not None

    def test_escalate(self) -> None:
        """Test escalating an incident."""
        assignment = IncidentAssignment(
            assignment_id="assign-1",
            incident_id="inc-1",
            engineer_id="eng-1",
            priority=IncidentPriority.P1,
            escalation_level=EscalationLevel.L1,
            assigned_at=datetime.utcnow(),
        )
        assignment.escalate("eng-2")
        assert assignment.escalated_at is not None
        assert assignment.escalated_to == "eng-2"


# =============================================================================
# OnCallMetrics Tests
# =============================================================================


class TestOnCallMetrics:
    """Tests for OnCallMetrics dataclass."""

    def test_create_metrics(self) -> None:
        """Test creating on-call metrics."""
        metrics = OnCallMetrics(
            period_start=datetime.utcnow() - timedelta(days=30),
            period_end=datetime.utcnow(),
            total_incidents=100,
            acknowledged_within_sla=95,
            resolved_within_sla=90,
            escalations=5,
            avg_response_time_seconds=120.0,
            avg_resolution_time_seconds=3600.0,
            p95_response_time_seconds=300.0,
            sla_compliance_percent=95.0,
        )
        assert metrics.total_incidents == 100
        assert metrics.sla_compliance_percent == 95.0

    def test_acknowledgment_rate(self) -> None:
        """Test acknowledgment rate calculation."""
        metrics = OnCallMetrics(
            period_start=datetime.utcnow() - timedelta(days=30),
            period_end=datetime.utcnow(),
            total_incidents=100,
            acknowledged_within_sla=80,
            resolved_within_sla=75,
            escalations=10,
            avg_response_time_seconds=150.0,
            avg_resolution_time_seconds=4000.0,
            p95_response_time_seconds=400.0,
            sla_compliance_percent=80.0,
        )
        assert metrics.acknowledgment_rate == 80.0

    def test_acknowledgment_rate_no_incidents(self) -> None:
        """Test acknowledgment rate with no incidents."""
        metrics = OnCallMetrics(
            period_start=datetime.utcnow() - timedelta(days=30),
            period_end=datetime.utcnow(),
            total_incidents=0,
            acknowledged_within_sla=0,
            resolved_within_sla=0,
            escalations=0,
            avg_response_time_seconds=0,
            avg_resolution_time_seconds=0,
            p95_response_time_seconds=0,
            sla_compliance_percent=100.0,
        )
        assert metrics.acknowledgment_rate == 100.0


# =============================================================================
# OnCallManagementService Tests
# =============================================================================


class TestOnCallManagementService:
    """Tests for OnCallManagementService."""

    def test_create_service_default_config(self) -> None:
        """Test creating service with default config."""
        service = OnCallManagementService()
        assert service.config.minimum_engineers == 4
        assert service.config.auto_escalation_enabled is True

    def test_create_service_custom_config(self) -> None:
        """Test creating service with custom config."""
        config = OnCallConfig(
            minimum_engineers=6,
            rotation_period_days=14,
            auto_escalation_enabled=False,
        )
        service = OnCallManagementService(config)
        assert service.config.minimum_engineers == 6

    def test_default_policy_initialized(self) -> None:
        """Test that default policy is initialized."""
        service = OnCallManagementService()
        policy = service.get_policy("default")
        assert policy is not None
        assert policy.name == "Default Escalation Policy"

    def test_add_engineer(self) -> None:
        """Test adding an engineer."""
        service = OnCallManagementService()
        engineer = service.add_engineer(
            name="John Doe",
            email="john@example.com",
            phone="+1234567890",
            timezone="UTC",
            escalation_level=EscalationLevel.L1,
        )
        assert engineer.name == "John Doe"
        assert engineer.escalation_level == EscalationLevel.L1

    def test_get_engineer(self) -> None:
        """Test getting engineer by ID."""
        service = OnCallManagementService()
        engineer = service.add_engineer("John", "john@example.com", "+1", "UTC", EscalationLevel.L1)
        retrieved = service.get_engineer(engineer.engineer_id)
        assert retrieved is not None
        assert retrieved.name == "John"

    def test_get_engineer_not_found(self) -> None:
        """Test getting non-existent engineer."""
        service = OnCallManagementService()
        assert service.get_engineer("nonexistent") is None

    def test_list_engineers(self) -> None:
        """Test listing engineers."""
        service = OnCallManagementService()
        service.add_engineer("Eng 1", "e1@ex.com", "+1", "UTC", EscalationLevel.L1)
        service.add_engineer("Eng 2", "e2@ex.com", "+2", "UTC", EscalationLevel.L2)

        engineers = service.list_engineers()
        assert len(engineers) == 2

    def test_list_engineers_available_only(self) -> None:
        """Test listing only available engineers."""
        service = OnCallManagementService()
        eng1 = service.add_engineer("Eng 1", "e1@ex.com", "+1", "UTC", EscalationLevel.L1)
        service.add_engineer("Eng 2", "e2@ex.com", "+2", "UTC", EscalationLevel.L2)
        service.set_availability(eng1.engineer_id, False)

        available = service.list_engineers(available_only=True)
        assert len(available) == 1

    def test_list_engineers_by_level(self) -> None:
        """Test listing engineers by level."""
        service = OnCallManagementService()
        service.add_engineer("Eng 1", "e1@ex.com", "+1", "UTC", EscalationLevel.L1)
        service.add_engineer("Eng 2", "e2@ex.com", "+2", "UTC", EscalationLevel.L1)
        service.add_engineer("Eng 3", "e3@ex.com", "+3", "UTC", EscalationLevel.L2)

        l1_engineers = service.list_engineers(level=EscalationLevel.L1)
        assert len(l1_engineers) == 2

    def test_set_availability(self) -> None:
        """Test setting engineer availability."""
        service = OnCallManagementService()
        engineer = service.add_engineer("John", "j@ex.com", "+1", "UTC", EscalationLevel.L1)

        result = service.set_availability(engineer.engineer_id, False)
        assert result is True
        assert engineer.is_available is False

    def test_set_availability_not_found(self) -> None:
        """Test setting availability for non-existent engineer."""
        service = OnCallManagementService()
        result = service.set_availability("nonexistent", False)
        assert result is False

    def test_get_team_capacity(self) -> None:
        """Test getting team capacity."""
        service = OnCallManagementService()
        for i in range(4):
            service.add_engineer(f"Eng {i}", f"e{i}@ex.com", f"+{i}", "UTC", EscalationLevel.L1)

        capacity = service.get_team_capacity()
        assert capacity["total_engineers"] == 4
        assert capacity["available_engineers"] == 4
        assert capacity["is_adequate"] is True

    def test_get_team_capacity_inadequate(self) -> None:
        """Test team capacity when inadequate."""
        service = OnCallManagementService()
        # Only add 2 engineers (minimum is 4)
        for i in range(2):
            service.add_engineer(f"Eng {i}", f"e{i}@ex.com", f"+{i}", "UTC", EscalationLevel.L1)

        capacity = service.get_team_capacity()
        assert capacity["is_adequate"] is False

    def test_create_schedule(self) -> None:
        """Test creating a schedule."""
        service = OnCallManagementService()
        schedule = service.create_schedule(
            name="24/7 Coverage",
            tier=OnCallTier.FULL,
        )
        assert schedule.name == "24/7 Coverage"
        assert schedule.tier == OnCallTier.FULL

    def test_get_schedule(self) -> None:
        """Test getting schedule by ID."""
        service = OnCallManagementService()
        schedule = service.create_schedule("Test", OnCallTier.FULL)

        retrieved = service.get_schedule(schedule.schedule_id)
        assert retrieved is not None

    def test_add_shift(self) -> None:
        """Test adding a shift to schedule."""
        service = OnCallManagementService()
        engineer = service.add_engineer("John", "j@ex.com", "+1", "UTC", EscalationLevel.L1)
        schedule = service.create_schedule("Test", OnCallTier.FULL)

        shift = service.add_shift(
            schedule_id=schedule.schedule_id,
            engineer_id=engineer.engineer_id,
            shift_type=ShiftType.PRIMARY,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(hours=8),
        )
        assert shift.shift_type == ShiftType.PRIMARY
        assert len(schedule.shifts) == 1

    def test_add_shift_schedule_not_found(self) -> None:
        """Test adding shift to non-existent schedule."""
        service = OnCallManagementService()
        engineer = service.add_engineer("John", "j@ex.com", "+1", "UTC", EscalationLevel.L1)

        with pytest.raises(ValueError, match="Schedule not found"):
            service.add_shift(
                "nonexistent",
                engineer.engineer_id,
                ShiftType.PRIMARY,
                datetime.utcnow(),
                datetime.utcnow() + timedelta(hours=8),
            )

    def test_add_shift_engineer_not_found(self) -> None:
        """Test adding shift with non-existent engineer."""
        service = OnCallManagementService()
        schedule = service.create_schedule("Test", OnCallTier.FULL)

        with pytest.raises(ValueError, match="Engineer not found"):
            service.add_shift(
                schedule.schedule_id,
                "nonexistent",
                ShiftType.PRIMARY,
                datetime.utcnow(),
                datetime.utcnow() + timedelta(hours=8),
            )

    def test_get_current_on_call(self) -> None:
        """Test getting current on-call engineers."""
        service = OnCallManagementService()
        engineer = service.add_engineer("John", "j@ex.com", "+1", "UTC", EscalationLevel.L1)
        schedule = service.create_schedule("Test", OnCallTier.FULL)

        service.add_shift(
            schedule.schedule_id,
            engineer.engineer_id,
            ShiftType.PRIMARY,
            datetime.utcnow() - timedelta(hours=2),
            datetime.utcnow() + timedelta(hours=6),
        )

        on_call = service.get_current_on_call()
        assert len(on_call) == 1
        assert on_call[0].name == "John"

    def test_generate_rotation(self) -> None:
        """Test generating rotation."""
        service = OnCallManagementService()
        for i in range(4):
            service.add_engineer(f"Eng {i}", f"e{i}@ex.com", f"+{i}", "UTC", EscalationLevel.L1)

        schedule = service.create_schedule("Test", OnCallTier.FULL)

        shifts = service.generate_rotation(
            schedule_id=schedule.schedule_id,
            start_date=datetime.utcnow(),
            weeks=4,
        )
        # Should have 8 shifts (2 per week - primary and secondary)
        assert len(shifts) == 8

    def test_generate_rotation_insufficient_engineers(self) -> None:
        """Test generating rotation with insufficient engineers."""
        service = OnCallManagementService()
        # Only add 2 engineers
        for i in range(2):
            service.add_engineer(f"Eng {i}", f"e{i}@ex.com", f"+{i}", "UTC", EscalationLevel.L1)

        schedule = service.create_schedule("Test", OnCallTier.FULL)

        with pytest.raises(ValueError, match="Not enough engineers"):
            service.generate_rotation(schedule.schedule_id, datetime.utcnow())

    def test_create_policy(self) -> None:
        """Test creating an escalation policy."""
        service = OnCallManagementService()
        policy = service.create_policy(
            name="Custom Policy",
            description="Custom escalation",
            levels=[{"level": "l1", "timeout_minutes": 10}],
        )
        assert policy.name == "Custom Policy"

    def test_assign_incident(self) -> None:
        """Test assigning an incident."""
        service = OnCallManagementService()
        engineer = service.add_engineer("John", "j@ex.com", "+1", "UTC", EscalationLevel.L1)
        schedule = service.create_schedule("Test", OnCallTier.FULL)
        service.add_shift(
            schedule.schedule_id,
            engineer.engineer_id,
            ShiftType.PRIMARY,
            datetime.utcnow() - timedelta(hours=2),
            datetime.utcnow() + timedelta(hours=6),
        )

        assignment = service.assign_incident("inc-1", IncidentPriority.P1)
        assert assignment.incident_id == "inc-1"
        assert assignment.priority == IncidentPriority.P1

    def test_assign_incident_no_on_call(self) -> None:
        """Test assigning incident when no one on-call."""
        service = OnCallManagementService()

        with pytest.raises(ValueError, match="No engineers currently on-call"):
            service.assign_incident("inc-1", IncidentPriority.P1)

    def test_acknowledge_incident(self) -> None:
        """Test acknowledging an incident."""
        service = OnCallManagementService()
        engineer = service.add_engineer("John", "j@ex.com", "+1", "UTC", EscalationLevel.L1)
        schedule = service.create_schedule("Test", OnCallTier.FULL)
        service.add_shift(
            schedule.schedule_id,
            engineer.engineer_id,
            ShiftType.PRIMARY,
            datetime.utcnow() - timedelta(hours=2),
            datetime.utcnow() + timedelta(hours=6),
        )

        assignment = service.assign_incident("inc-1", IncidentPriority.P1)
        result = service.acknowledge_incident(assignment.assignment_id)
        assert result is True
        assert assignment.acknowledged_at is not None

    def test_resolve_incident(self) -> None:
        """Test resolving an incident."""
        service = OnCallManagementService()
        engineer = service.add_engineer("John", "j@ex.com", "+1", "UTC", EscalationLevel.L1)
        schedule = service.create_schedule("Test", OnCallTier.FULL)
        service.add_shift(
            schedule.schedule_id,
            engineer.engineer_id,
            ShiftType.PRIMARY,
            datetime.utcnow() - timedelta(hours=2),
            datetime.utcnow() + timedelta(hours=6),
        )

        assignment = service.assign_incident("inc-1", IncidentPriority.P2)
        result = service.resolve_incident(assignment.assignment_id)
        assert result is True
        assert assignment.resolved_at is not None

    def test_escalate_incident(self) -> None:
        """Test escalating an incident."""
        service = OnCallManagementService()
        eng1 = service.add_engineer("Eng 1", "e1@ex.com", "+1", "UTC", EscalationLevel.L1)
        eng2 = service.add_engineer("Eng 2", "e2@ex.com", "+2", "UTC", EscalationLevel.L2)
        schedule = service.create_schedule("Test", OnCallTier.FULL)
        service.add_shift(
            schedule.schedule_id,
            eng1.engineer_id,
            ShiftType.PRIMARY,
            datetime.utcnow() - timedelta(hours=2),
            datetime.utcnow() + timedelta(hours=6),
        )

        assignment = service.assign_incident("inc-1", IncidentPriority.P1)
        result = service.escalate_incident(assignment.assignment_id, eng2.engineer_id)
        assert result is True
        assert assignment.escalation_level == EscalationLevel.L2

    def test_get_sla_target(self) -> None:
        """Test getting SLA target for priority."""
        service = OnCallManagementService()
        assert service.get_sla_target(IncidentPriority.P1) == 300  # 5 minutes
        assert service.get_sla_target(IncidentPriority.P2) == 900  # 15 minutes
        assert service.get_sla_target(IncidentPriority.P3) == 3600  # 1 hour
        assert service.get_sla_target(IncidentPriority.P4) == 86400  # 24 hours

    def test_calculate_metrics(self) -> None:
        """Test calculating metrics."""
        service = OnCallManagementService()
        engineer = service.add_engineer("John", "j@ex.com", "+1", "UTC", EscalationLevel.L1)
        schedule = service.create_schedule("Test", OnCallTier.FULL)
        service.add_shift(
            schedule.schedule_id,
            engineer.engineer_id,
            ShiftType.PRIMARY,
            datetime.utcnow() - timedelta(days=15),
            datetime.utcnow() + timedelta(days=15),
        )

        # Create and acknowledge some incidents
        for i in range(5):
            assignment = service.assign_incident(f"inc-{i}", IncidentPriority.P2)
            assignment.acknowledge()
            assignment.resolve()

        metrics = service.calculate_metrics(
            period_start=datetime.utcnow() - timedelta(days=30),
            period_end=datetime.utcnow(),
        )
        assert metrics.total_incidents == 5
        assert metrics.sla_compliance_percent >= 0

    def test_calculate_metrics_no_incidents(self) -> None:
        """Test calculating metrics with no incidents."""
        service = OnCallManagementService()

        metrics = service.calculate_metrics(
            period_start=datetime.utcnow() - timedelta(days=30),
            period_end=datetime.utcnow(),
        )
        assert metrics.total_incidents == 0
        assert metrics.sla_compliance_percent == 100.0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_on_call_management_default(self) -> None:
        """Test creating service with factory function."""
        service = create_on_call_management()
        assert isinstance(service, OnCallManagementService)
        assert service.config.minimum_engineers == 4

    def test_create_on_call_management_custom(self) -> None:
        """Test creating service with custom options."""
        service = create_on_call_management(
            minimum_engineers=6,
            auto_escalation_enabled=False,
        )
        assert service.config.minimum_engineers == 6
        assert service.config.auto_escalation_enabled is False


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_on_call_tier_values(self) -> None:
        """Test all on-call tier values."""
        assert OnCallTier.BASIC.value == "basic"
        assert OnCallTier.EXTENDED.value == "extended"
        assert OnCallTier.FULL.value == "full"

    def test_escalation_level_values(self) -> None:
        """Test all escalation level values."""
        assert EscalationLevel.L1.value == "l1"
        assert EscalationLevel.L2.value == "l2"
        assert EscalationLevel.L3.value == "l3"
        assert EscalationLevel.L4.value == "l4"

    def test_shift_type_values(self) -> None:
        """Test all shift type values."""
        assert ShiftType.PRIMARY.value == "primary"
        assert ShiftType.SECONDARY.value == "secondary"
        assert ShiftType.SHADOW.value == "shadow"

    def test_incident_priority_values(self) -> None:
        """Test all incident priority values."""
        assert IncidentPriority.P1.value == "p1"
        assert IncidentPriority.P2.value == "p2"
        assert IncidentPriority.P3.value == "p3"
        assert IncidentPriority.P4.value == "p4"
