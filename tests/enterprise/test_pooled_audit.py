# -*- coding: utf-8 -*-
"""
Comprehensive tests for Pooled Audit Coordination Service.

Tests pooled audit coordination per DORA Art. 30(4) requirements.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from services.enterprise.pooled_audit_coordination import (
    # Enums
    AuditCoordinationStatus,
    ParticipantRole,
    CostAllocationMethod,
    # Data structures
    AuditParticipant,
    AuditSchedule,
    CostAllocation,
    AuditFinding,
    AuditCoordinationPlan,
    CoordinationConfig,
    # Service
    PooledAuditCoordinationService,
    # Factory
    create_pooled_audit_coordination,
)


# =============================================================================
# AuditParticipant Tests
# =============================================================================


class TestAuditParticipant:
    """Tests for AuditParticipant dataclass."""

    def test_create_participant(self) -> None:
        """Test creating an audit participant."""
        participant = AuditParticipant(
            participant_id="part-1",
            client_id="client-1",
            client_name="Bank ABC",
            role=ParticipantRole.PARTICIPANT,
            contact_name="John Doe",
            contact_email="john@bank.com",
            joined_at=datetime.utcnow(),
        )
        assert participant.client_name == "Bank ABC"
        assert participant.role == ParticipantRole.PARTICIPANT

    def test_participant_defaults(self) -> None:
        """Test participant default values."""
        participant = AuditParticipant(
            participant_id="part-1",
            client_id="client-1",
            client_name="Test",
            role=ParticipantRole.PARTICIPANT,
            contact_name="Test",
            contact_email="test@test.com",
            joined_at=datetime.utcnow(),
        )
        assert participant.status == "active"
        assert participant.cost_share_percent == 0.0
        assert participant.has_signed_agreement is False


# =============================================================================
# CostAllocation Tests
# =============================================================================


class TestCostAllocation:
    """Tests for CostAllocation dataclass."""

    def test_create_allocation(self) -> None:
        """Test creating cost allocation."""
        allocation = CostAllocation(
            allocation_id="alloc-1",
            audit_id="audit-1",
            method=CostAllocationMethod.EQUAL,
            total_cost=100000.0,
            currency="EUR",
        )
        assert allocation.total_cost == 100000.0
        assert allocation.finalized is False

    def test_calculate_equal(self) -> None:
        """Test equal cost allocation calculation."""
        allocation = CostAllocation(
            allocation_id="alloc-1",
            audit_id="audit-1",
            method=CostAllocationMethod.EQUAL,
            total_cost=90000.0,
        )

        participants = [
            AuditParticipant(
                f"part-{i}",
                f"client-{i}",
                f"Client {i}",
                ParticipantRole.PARTICIPANT,
                "Contact",
                "email@test.com",
                datetime.utcnow(),
            )
            for i in range(3)
        ]

        allocation.calculate_equal(participants)

        assert len(allocation.participant_allocations) == 3
        for p in participants:
            assert allocation.participant_allocations[p.participant_id] == 30000.0
            assert p.cost_share_amount == 30000.0


# =============================================================================
# AuditCoordinationPlan Tests
# =============================================================================


class TestAuditCoordinationPlan:
    """Tests for AuditCoordinationPlan dataclass."""

    def test_create_plan(self) -> None:
        """Test creating an audit coordination plan."""
        plan = AuditCoordinationPlan(
            plan_id="plan-1",
            title="Q1 2025 Pooled SOC2 Audit",
            description="Joint SOC2 audit for financial clients",
            audit_type="SOC2",
            status=AuditCoordinationStatus.PROPOSED,
            proposed_by="compliance@provider.com",
            proposed_at=datetime.utcnow(),
            minimum_participants=3,
            maximum_participants=10,
            estimated_cost=150000.0,
        )
        assert plan.title == "Q1 2025 Pooled SOC2 Audit"
        assert plan.audit_type == "SOC2"

    def test_participant_count(self) -> None:
        """Test participant count property."""
        plan = AuditCoordinationPlan(
            plan_id="plan-1",
            title="Test",
            description="Test",
            audit_type="SOC2",
            status=AuditCoordinationStatus.RECRUITING,
            proposed_by="user",
            proposed_at=datetime.utcnow(),
            minimum_participants=3,
            maximum_participants=10,
            estimated_cost=100000.0,
        )

        # Add active participants
        for i in range(3):
            plan.participants.append(
                AuditParticipant(
                    f"part-{i}",
                    f"client-{i}",
                    f"Client {i}",
                    ParticipantRole.PARTICIPANT,
                    "Contact",
                    "email@test.com",
                    datetime.utcnow(),
                )
            )

        # Add withdrawn participant
        withdrawn = AuditParticipant(
            "part-w",
            "client-w",
            "Withdrawn Client",
            ParticipantRole.PARTICIPANT,
            "Contact",
            "email@test.com",
            datetime.utcnow(),
        )
        withdrawn.status = "withdrawn"
        plan.participants.append(withdrawn)

        assert plan.participant_count == 3

    def test_is_viable(self) -> None:
        """Test is_viable property."""
        plan = AuditCoordinationPlan(
            plan_id="plan-1",
            title="Test",
            description="Test",
            audit_type="SOC2",
            status=AuditCoordinationStatus.RECRUITING,
            proposed_by="user",
            proposed_at=datetime.utcnow(),
            minimum_participants=3,
            maximum_participants=10,
            estimated_cost=100000.0,
        )

        assert plan.is_viable is False  # No participants

        for i in range(3):
            plan.participants.append(
                AuditParticipant(
                    f"part-{i}",
                    f"client-{i}",
                    f"Client {i}",
                    ParticipantRole.PARTICIPANT,
                    "Contact",
                    "email@test.com",
                    datetime.utcnow(),
                )
            )

        assert plan.is_viable is True

    def test_add_participant(self) -> None:
        """Test adding participant to plan."""
        plan = AuditCoordinationPlan(
            plan_id="plan-1",
            title="Test",
            description="Test",
            audit_type="SOC2",
            status=AuditCoordinationStatus.RECRUITING,
            proposed_by="user",
            proposed_at=datetime.utcnow(),
            minimum_participants=1,
            maximum_participants=2,
            estimated_cost=100000.0,
        )

        p1 = AuditParticipant(
            "p1", "c1", "C1", ParticipantRole.PARTICIPANT, "N", "e@e.com", datetime.utcnow()
        )
        p2 = AuditParticipant(
            "p2", "c2", "C2", ParticipantRole.PARTICIPANT, "N", "e@e.com", datetime.utcnow()
        )
        p3 = AuditParticipant(
            "p3", "c3", "C3", ParticipantRole.PARTICIPANT, "N", "e@e.com", datetime.utcnow()
        )

        assert plan.add_participant(p1) is True
        assert plan.add_participant(p2) is True
        assert plan.add_participant(p3) is False  # Max reached


# =============================================================================
# PooledAuditCoordinationService Tests
# =============================================================================


class TestPooledAuditCoordinationService:
    """Tests for PooledAuditCoordinationService."""

    def test_create_service_default_config(self) -> None:
        """Test creating service with default config."""
        service = PooledAuditCoordinationService()
        assert service.config.minimum_participants_default == 3
        assert service.config.maximum_participants_default == 10

    def test_create_service_custom_config(self) -> None:
        """Test creating service with custom config."""
        config = CoordinationConfig(
            minimum_participants_default=5,
            maximum_participants_default=15,
        )
        service = PooledAuditCoordinationService(config)
        assert service.config.minimum_participants_default == 5

    def test_create_plan(self) -> None:
        """Test creating a pooled audit plan."""
        service = PooledAuditCoordinationService()
        plan = service.create_plan(
            title="2025 SOC2 Pooled Audit",
            description="Annual SOC2 audit coordination",
            audit_type="SOC2",
            proposed_by="compliance@provider.com",
            estimated_cost=150000.0,
            scope_areas=["Security", "Availability"],
        )
        assert plan.title == "2025 SOC2 Pooled Audit"
        assert plan.status == AuditCoordinationStatus.PROPOSED

    def test_get_plan(self) -> None:
        """Test getting plan by ID."""
        service = PooledAuditCoordinationService()
        plan = service.create_plan("Test", "Test", "SOC2", "user", 100000.0, ["Security"])

        retrieved = service.get_plan(plan.plan_id)
        assert retrieved is not None
        assert retrieved.plan_id == plan.plan_id

    def test_get_plan_not_found(self) -> None:
        """Test getting non-existent plan."""
        service = PooledAuditCoordinationService()
        assert service.get_plan("nonexistent") is None

    def test_list_plans(self) -> None:
        """Test listing plans."""
        service = PooledAuditCoordinationService()
        service.create_plan("Plan 1", "Desc", "SOC2", "user", 100000.0, [])
        service.create_plan("Plan 2", "Desc", "ISO27001", "user", 80000.0, [])

        plans = service.list_plans()
        assert len(plans) == 2

    def test_list_plans_by_status(self) -> None:
        """Test listing plans by status."""
        service = PooledAuditCoordinationService()
        plan1 = service.create_plan("Plan 1", "Desc", "SOC2", "user", 100000.0, [])
        service.create_plan("Plan 2", "Desc", "SOC2", "user", 80000.0, [])

        service.update_plan_status(plan1.plan_id, AuditCoordinationStatus.RECRUITING)

        recruiting = service.list_plans(status=AuditCoordinationStatus.RECRUITING)
        assert len(recruiting) == 1

    def test_list_plans_by_type(self) -> None:
        """Test listing plans by audit type."""
        service = PooledAuditCoordinationService()
        service.create_plan("Plan 1", "Desc", "SOC2", "user", 100000.0, [])
        service.create_plan("Plan 2", "Desc", "ISO27001", "user", 80000.0, [])

        soc2_plans = service.list_plans(audit_type="SOC2")
        assert len(soc2_plans) == 1

    def test_update_plan_status(self) -> None:
        """Test updating plan status."""
        service = PooledAuditCoordinationService()
        plan = service.create_plan("Test", "Desc", "SOC2", "user", 100000.0, [])

        updated = service.update_plan_status(plan.plan_id, AuditCoordinationStatus.PLANNING)
        assert updated is not None
        assert updated.status == AuditCoordinationStatus.PLANNING

    def test_update_plan_status_in_progress(self) -> None:
        """Test updating status to IN_PROGRESS sets actual_start."""
        service = PooledAuditCoordinationService()
        plan = service.create_plan("Test", "Desc", "SOC2", "user", 100000.0, [])

        updated = service.update_plan_status(plan.plan_id, AuditCoordinationStatus.IN_PROGRESS)
        assert updated is not None
        assert updated.actual_start is not None

    def test_update_plan_status_completed(self) -> None:
        """Test updating status to COMPLETED sets actual_end."""
        service = PooledAuditCoordinationService()
        plan = service.create_plan("Test", "Desc", "SOC2", "user", 100000.0, [])

        updated = service.update_plan_status(plan.plan_id, AuditCoordinationStatus.COMPLETED)
        assert updated is not None
        assert updated.actual_end is not None

    def test_add_participant(self) -> None:
        """Test adding participant to plan."""
        service = PooledAuditCoordinationService()
        plan = service.create_plan("Test", "Desc", "SOC2", "user", 100000.0, [])

        participant = service.add_participant(
            plan_id=plan.plan_id,
            client_id="client-1",
            client_name="Bank ABC",
            contact_name="John Doe",
            contact_email="john@bank.com",
        )
        assert participant.client_name == "Bank ABC"
        assert len(plan.participants) == 1

    def test_add_participant_plan_not_found(self) -> None:
        """Test adding participant to non-existent plan."""
        service = PooledAuditCoordinationService()

        with pytest.raises(ValueError, match="Plan not found"):
            service.add_participant(
                "nonexistent",
                "client-1",
                "Client",
                "Contact",
                "email@test.com",
            )

    def test_add_participant_max_reached(self) -> None:
        """Test adding participant when max reached."""
        service = PooledAuditCoordinationService()
        plan = service.create_plan(
            "Test",
            "Desc",
            "SOC2",
            "user",
            100000.0,
            [],
            maximum_participants=1,
        )

        service.add_participant(plan.plan_id, "c1", "C1", "N", "e@e.com")

        with pytest.raises(ValueError, match="Maximum participants"):
            service.add_participant(plan.plan_id, "c2", "C2", "N", "e@e.com")

    def test_remove_participant(self) -> None:
        """Test removing participant from plan."""
        service = PooledAuditCoordinationService()
        plan = service.create_plan("Test", "Desc", "SOC2", "user", 100000.0, [])
        participant = service.add_participant(plan.plan_id, "c1", "C1", "N", "e@e.com")

        result = service.remove_participant(plan.plan_id, participant.participant_id)
        assert result is True
        assert participant.status == "withdrawn"

    def test_sign_agreement(self) -> None:
        """Test recording agreement signature."""
        service = PooledAuditCoordinationService()
        plan = service.create_plan("Test", "Desc", "SOC2", "user", 100000.0, [])
        participant = service.add_participant(plan.plan_id, "c1", "C1", "N", "e@e.com")

        result = service.sign_agreement(plan.plan_id, participant.participant_id)
        assert result is True
        assert participant.has_signed_agreement is True
        assert participant.agreement_signed_at is not None

    def test_get_participants(self) -> None:
        """Test getting participants for plan."""
        service = PooledAuditCoordinationService()
        plan = service.create_plan("Test", "Desc", "SOC2", "user", 100000.0, [])

        service.add_participant(plan.plan_id, "c1", "C1", "N", "e@e.com")
        p2 = service.add_participant(plan.plan_id, "c2", "C2", "N", "e@e.com")
        service.remove_participant(plan.plan_id, p2.participant_id)

        all_participants = service.get_participants(plan.plan_id, active_only=False)
        active_only = service.get_participants(plan.plan_id, active_only=True)

        assert len(all_participants) == 2
        assert len(active_only) == 1

    def test_create_cost_allocation(self) -> None:
        """Test creating cost allocation."""
        service = PooledAuditCoordinationService()
        plan = service.create_plan("Test", "Desc", "SOC2", "user", 100000.0, [])

        for i in range(3):
            service.add_participant(plan.plan_id, f"c{i}", f"C{i}", "N", "e@e.com")

        allocation = service.create_cost_allocation(
            plan_id=plan.plan_id,
            total_cost=90000.0,
            method=CostAllocationMethod.EQUAL,
        )
        assert allocation.total_cost == 90000.0
        assert len(allocation.participant_allocations) == 3

    def test_create_cost_allocation_plan_not_found(self) -> None:
        """Test creating allocation for non-existent plan."""
        service = PooledAuditCoordinationService()

        with pytest.raises(ValueError, match="Plan not found"):
            service.create_cost_allocation("nonexistent", 100000.0)

    def test_finalize_costs(self) -> None:
        """Test finalizing cost allocation."""
        service = PooledAuditCoordinationService()
        plan = service.create_plan("Test", "Desc", "SOC2", "user", 100000.0, [])
        service.add_participant(plan.plan_id, "c1", "C1", "N", "e@e.com")
        service.create_cost_allocation(plan.plan_id, 100000.0)

        result = service.finalize_costs(plan.plan_id)
        assert result is True
        assert plan.cost_allocation is not None
        assert plan.cost_allocation.finalized is True

    def test_add_schedule(self) -> None:
        """Test adding schedule to plan."""
        service = PooledAuditCoordinationService()
        plan = service.create_plan("Test", "Desc", "SOC2", "user", 100000.0, [])

        schedule = service.add_schedule(
            plan_id=plan.plan_id,
            phase="fieldwork",
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow() + timedelta(weeks=2),
            activities=["Document review", "Interviews"],
        )
        assert schedule.phase == "fieldwork"
        assert len(plan.schedules) == 1

    def test_get_schedules(self) -> None:
        """Test getting schedules for plan."""
        service = PooledAuditCoordinationService()
        plan = service.create_plan("Test", "Desc", "SOC2", "user", 100000.0, [])

        service.add_schedule(
            plan.plan_id, "planning", datetime.utcnow(), datetime.utcnow() + timedelta(weeks=1), []
        )
        service.add_schedule(
            plan.plan_id, "fieldwork", datetime.utcnow(), datetime.utcnow() + timedelta(weeks=2), []
        )

        schedules = service.get_schedules(plan.plan_id)
        assert len(schedules) == 2

    def test_add_finding(self) -> None:
        """Test adding finding to plan."""
        service = PooledAuditCoordinationService()
        plan = service.create_plan("Test", "Desc", "SOC2", "user", 100000.0, [])

        finding = service.add_finding(
            plan_id=plan.plan_id,
            title="Access Control Gap",
            description="Missing MFA for admin accounts",
            severity="high",
        )
        assert finding.title == "Access Control Gap"
        assert finding.severity == "high"
        assert len(plan.findings) == 1

    def test_get_findings(self) -> None:
        """Test getting findings for plan."""
        service = PooledAuditCoordinationService()
        plan = service.create_plan("Test", "Desc", "SOC2", "user", 100000.0, [])

        service.add_finding(plan.plan_id, "Finding 1", "Desc", "high")
        service.add_finding(plan.plan_id, "Finding 2", "Desc", "medium")

        findings = service.get_findings(plan.plan_id)
        assert len(findings) == 2

    def test_get_findings_by_status(self) -> None:
        """Test getting findings by status."""
        service = PooledAuditCoordinationService()
        plan = service.create_plan("Test", "Desc", "SOC2", "user", 100000.0, [])

        f1 = service.add_finding(plan.plan_id, "Finding 1", "Desc", "high")
        service.add_finding(plan.plan_id, "Finding 2", "Desc", "medium")
        f1.status = "remediated"

        open_findings = service.get_findings(plan.plan_id, status="open")
        assert len(open_findings) == 1

    def test_issue_report(self) -> None:
        """Test issuing audit report."""
        service = PooledAuditCoordinationService()
        plan = service.create_plan("Test", "Desc", "SOC2", "user", 100000.0, [])

        result = service.issue_report(plan.plan_id, "/reports/audit_2025.pdf")
        assert result is True
        assert plan.report_issued_at is not None
        assert plan.report_path == "/reports/audit_2025.pdf"
        assert plan.status == AuditCoordinationStatus.COMPLETED

    def test_get_coordination_summary(self) -> None:
        """Test getting coordination summary."""
        service = PooledAuditCoordinationService()
        plan = service.create_plan(
            "Test Audit",
            "Desc",
            "SOC2",
            "user",
            100000.0,
            [],
            minimum_participants=3,
        )

        for i in range(3):
            p = service.add_participant(plan.plan_id, f"c{i}", f"C{i}", "N", "e@e.com")
            service.sign_agreement(plan.plan_id, p.participant_id)

        service.create_cost_allocation(plan.plan_id, 90000.0)
        service.add_finding(plan.plan_id, "Finding", "Desc", "high")

        summary = service.get_coordination_summary(plan.plan_id)

        assert summary["title"] == "Test Audit"
        assert summary["participant_count"] == 3
        assert summary["is_viable"] is True
        assert summary["all_agreements_signed"] is True
        assert summary["total_findings"] == 1


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_pooled_audit_coordination_default(self) -> None:
        """Test creating service with factory function."""
        service = create_pooled_audit_coordination()
        assert isinstance(service, PooledAuditCoordinationService)

    def test_create_pooled_audit_coordination_custom(self) -> None:
        """Test creating service with custom options."""
        service = create_pooled_audit_coordination(
            minimum_participants=5,
            maximum_participants=20,
        )
        assert service.config.minimum_participants_default == 5
        assert service.config.maximum_participants_default == 20


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_coordination_status_values(self) -> None:
        """Test all coordination status values."""
        assert AuditCoordinationStatus.PROPOSED.value == "proposed"
        assert AuditCoordinationStatus.RECRUITING.value == "recruiting"
        assert AuditCoordinationStatus.PLANNING.value == "planning"
        assert AuditCoordinationStatus.SCHEDULED.value == "scheduled"
        assert AuditCoordinationStatus.IN_PROGRESS.value == "in_progress"
        assert AuditCoordinationStatus.COMPLETED.value == "completed"

    def test_participant_role_values(self) -> None:
        """Test all participant role values."""
        assert ParticipantRole.LEAD.value == "lead"
        assert ParticipantRole.CO_LEAD.value == "co_lead"
        assert ParticipantRole.PARTICIPANT.value == "participant"
        assert ParticipantRole.OBSERVER.value == "observer"

    def test_cost_allocation_method_values(self) -> None:
        """Test all cost allocation method values."""
        assert CostAllocationMethod.EQUAL.value == "equal"
        assert CostAllocationMethod.PROPORTIONAL_AUM.value == "proportional_aum"
        assert CostAllocationMethod.PROPORTIONAL_USAGE.value == "proportional_usage"
        assert CostAllocationMethod.FIXED_PLUS_VARIABLE.value == "fixed_plus_variable"
