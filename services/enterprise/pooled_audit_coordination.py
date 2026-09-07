# -*- coding: utf-8 -*-
"""
Pooled Audit Coordination Service.

DORA Phase 3 Block 3.12: Implement pooled audit coordination

Provides multi-client audit coordination per DORA Art. 30(4):
- Joint audit planning and scheduling
- Cost allocation among participants
- Shared audit report management
- Coordination of findings remediation

DORA References:
    - Art. 30(4): Pooled audits and joint testing arrangements
    - Art. 30(3)(e): Audit and access rights
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4


# =============================================================================
# Enums
# =============================================================================


class AuditCoordinationStatus(Enum):
    """Pooled audit coordination status."""

    PROPOSED = "proposed"
    RECRUITING = "recruiting"  # Recruiting participants
    PLANNING = "planning"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ParticipantRole(Enum):
    """Participant role in pooled audit."""

    LEAD = "lead"  # Lead organizer
    CO_LEAD = "co_lead"  # Co-organizer
    PARTICIPANT = "participant"  # Standard participant
    OBSERVER = "observer"  # Observer only


class CostAllocationMethod(Enum):
    """Cost allocation methods."""

    EQUAL = "equal"  # Split equally
    PROPORTIONAL_AUM = "proportional_aum"  # By assets under management
    PROPORTIONAL_USAGE = "proportional_usage"  # By platform usage
    FIXED_PLUS_VARIABLE = "fixed_plus_variable"  # Fixed base + variable


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class AuditParticipant:
    """Participant in pooled audit."""

    participant_id: str
    client_id: str
    client_name: str
    role: ParticipantRole
    contact_name: str
    contact_email: str
    joined_at: datetime
    status: str = "active"  # active, withdrawn
    cost_share_percent: float = 0.0
    cost_share_amount: float = 0.0
    scope_requirements: list[str] = field(default_factory=list)
    has_signed_agreement: bool = False
    agreement_signed_at: datetime | None = None


@dataclass
class AuditSchedule:
    """Pooled audit schedule."""

    schedule_id: str
    audit_id: str
    phase: str  # planning, fieldwork, review, reporting
    start_date: datetime
    end_date: datetime
    location: str | None = None
    is_remote: bool = True
    activities: list[str] = field(default_factory=list)
    assigned_auditors: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class CostAllocation:
    """Cost allocation for pooled audit."""

    allocation_id: str
    audit_id: str
    method: CostAllocationMethod
    total_cost: float
    currency: str = "EUR"
    participant_allocations: dict[str, float] = field(
        default_factory=dict
    )  # participant_id -> amount
    calculated_at: datetime = field(default_factory=datetime.utcnow)
    finalized: bool = False
    finalized_at: datetime | None = None

    def calculate_equal(self, participants: list[AuditParticipant]) -> None:
        """Calculate equal cost allocation."""
        if not participants:
            return
        per_participant = self.total_cost / len(participants)
        for p in participants:
            self.participant_allocations[p.participant_id] = per_participant
            p.cost_share_amount = per_participant
            p.cost_share_percent = 100.0 / len(participants)


@dataclass
class AuditFinding:
    """Finding from pooled audit."""

    finding_id: str
    audit_id: str
    title: str
    description: str
    severity: str  # critical, high, medium, low
    affected_participants: list[str]  # All or specific
    identified_at: datetime
    status: str = "open"
    remediation_owner: str = "provider"  # provider or participant
    remediation_due: datetime | None = None
    remediated_at: datetime | None = None


@dataclass
class AuditCoordinationPlan:
    """Pooled audit coordination plan."""

    plan_id: str
    title: str
    description: str
    audit_type: str  # SOC2, ISO27001, DORA_compliance, custom
    status: AuditCoordinationStatus
    proposed_by: str
    proposed_at: datetime
    minimum_participants: int
    maximum_participants: int
    estimated_cost: float
    currency: str = "EUR"
    audit_firm: str | None = None
    lead_auditor: str | None = None
    planned_start: datetime | None = None
    planned_end: datetime | None = None
    actual_start: datetime | None = None
    actual_end: datetime | None = None
    scope_areas: list[str] = field(default_factory=list)
    participants: list[AuditParticipant] = field(default_factory=list)
    schedules: list[AuditSchedule] = field(default_factory=list)
    cost_allocation: CostAllocation | None = None
    findings: list[AuditFinding] = field(default_factory=list)
    report_issued_at: datetime | None = None
    report_path: str | None = None

    @property
    def participant_count(self) -> int:
        """Get current participant count."""
        return len([p for p in self.participants if p.status == "active"])

    @property
    def is_viable(self) -> bool:
        """Check if audit has minimum participants."""
        return self.participant_count >= self.minimum_participants

    def add_participant(self, participant: AuditParticipant) -> bool:
        """Add participant to audit."""
        if self.participant_count >= self.maximum_participants:
            return False
        self.participants.append(participant)
        return True


@dataclass
class CoordinationConfig:
    """Pooled audit coordination configuration."""

    minimum_participants_default: int = 3
    maximum_participants_default: int = 10
    lead_time_days: int = 60
    default_cost_method: CostAllocationMethod = CostAllocationMethod.EQUAL
    require_signed_agreements: bool = True
    auto_calculate_costs: bool = True


# =============================================================================
# Main Service Class
# =============================================================================


class PooledAuditCoordinationService:
    """
    Pooled Audit Coordination Service.

    Coordinates multi-client audits per DORA Art. 30(4).
    """

    def __init__(self, config: CoordinationConfig | None = None) -> None:
        """Initialize pooled audit coordination service."""
        self.config = config or CoordinationConfig()
        self._plans: dict[str, AuditCoordinationPlan] = {}

    # =========================================================================
    # Plan Management
    # =========================================================================

    def create_plan(
        self,
        title: str,
        description: str,
        audit_type: str,
        proposed_by: str,
        estimated_cost: float,
        scope_areas: list[str],
        minimum_participants: int | None = None,
        maximum_participants: int | None = None,
        planned_start: datetime | None = None,
        planned_end: datetime | None = None,
    ) -> AuditCoordinationPlan:
        """Create a new pooled audit plan."""
        plan = AuditCoordinationPlan(
            plan_id=str(uuid4()),
            title=title,
            description=description,
            audit_type=audit_type,
            status=AuditCoordinationStatus.PROPOSED,
            proposed_by=proposed_by,
            proposed_at=datetime.utcnow(),
            minimum_participants=minimum_participants or self.config.minimum_participants_default,
            maximum_participants=maximum_participants or self.config.maximum_participants_default,
            estimated_cost=estimated_cost,
            scope_areas=scope_areas,
            planned_start=planned_start,
            planned_end=planned_end,
        )
        self._plans[plan.plan_id] = plan
        return plan

    def get_plan(self, plan_id: str) -> AuditCoordinationPlan | None:
        """Get plan by ID."""
        return self._plans.get(plan_id)

    def list_plans(
        self,
        status: AuditCoordinationStatus | None = None,
        audit_type: str | None = None,
    ) -> list[AuditCoordinationPlan]:
        """List plans with optional filters."""
        plans = list(self._plans.values())

        if status:
            plans = [p for p in plans if p.status == status]
        if audit_type:
            plans = [p for p in plans if p.audit_type == audit_type]

        return plans

    def update_plan_status(
        self,
        plan_id: str,
        status: AuditCoordinationStatus,
    ) -> AuditCoordinationPlan | None:
        """Update plan status."""
        plan = self._plans.get(plan_id)
        if not plan:
            return None

        plan.status = status

        if status == AuditCoordinationStatus.IN_PROGRESS:
            plan.actual_start = datetime.utcnow()
        elif status == AuditCoordinationStatus.COMPLETED:
            plan.actual_end = datetime.utcnow()

        return plan

    # =========================================================================
    # Participant Management
    # =========================================================================

    def add_participant(
        self,
        plan_id: str,
        client_id: str,
        client_name: str,
        contact_name: str,
        contact_email: str,
        role: ParticipantRole = ParticipantRole.PARTICIPANT,
        scope_requirements: list[str] | None = None,
    ) -> AuditParticipant:
        """Add participant to a plan."""
        plan = self._plans.get(plan_id)
        if not plan:
            raise ValueError(f"Plan not found: {plan_id}")

        participant = AuditParticipant(
            participant_id=str(uuid4()),
            client_id=client_id,
            client_name=client_name,
            role=role,
            contact_name=contact_name,
            contact_email=contact_email,
            joined_at=datetime.utcnow(),
            scope_requirements=scope_requirements or [],
        )

        if not plan.add_participant(participant):
            raise ValueError("Maximum participants reached")

        # Recalculate costs if auto-calculate is enabled
        if self.config.auto_calculate_costs and plan.cost_allocation:
            self._recalculate_costs(plan)

        return participant

    def remove_participant(self, plan_id: str, participant_id: str) -> bool:
        """Remove participant from a plan."""
        plan = self._plans.get(plan_id)
        if not plan:
            return False

        for p in plan.participants:
            if p.participant_id == participant_id:
                p.status = "withdrawn"
                if self.config.auto_calculate_costs and plan.cost_allocation:
                    self._recalculate_costs(plan)
                return True

        return False

    def sign_agreement(self, plan_id: str, participant_id: str) -> bool:
        """Record that participant signed agreement."""
        plan = self._plans.get(plan_id)
        if not plan:
            return False

        for p in plan.participants:
            if p.participant_id == participant_id:
                p.has_signed_agreement = True
                p.agreement_signed_at = datetime.utcnow()
                return True

        return False

    def get_participants(self, plan_id: str, active_only: bool = True) -> list[AuditParticipant]:
        """Get participants for a plan."""
        plan = self._plans.get(plan_id)
        if not plan:
            return []

        participants = plan.participants
        if active_only:
            participants = [p for p in participants if p.status == "active"]

        return participants

    # =========================================================================
    # Cost Management
    # =========================================================================

    def create_cost_allocation(
        self,
        plan_id: str,
        total_cost: float,
        method: CostAllocationMethod | None = None,
        currency: str = "EUR",
    ) -> CostAllocation:
        """Create cost allocation for a plan."""
        plan = self._plans.get(plan_id)
        if not plan:
            raise ValueError(f"Plan not found: {plan_id}")

        allocation = CostAllocation(
            allocation_id=str(uuid4()),
            audit_id=plan_id,
            method=method or self.config.default_cost_method,
            total_cost=total_cost,
            currency=currency,
        )

        # Calculate based on method
        active_participants = [p for p in plan.participants if p.status == "active"]
        if allocation.method == CostAllocationMethod.EQUAL:
            allocation.calculate_equal(active_participants)

        plan.cost_allocation = allocation
        return allocation

    def _recalculate_costs(self, plan: AuditCoordinationPlan) -> None:
        """Recalculate cost allocation for a plan."""
        if not plan.cost_allocation:
            return

        active_participants = [p for p in plan.participants if p.status == "active"]
        if plan.cost_allocation.method == CostAllocationMethod.EQUAL:
            plan.cost_allocation.calculate_equal(active_participants)

    def finalize_costs(self, plan_id: str) -> bool:
        """Finalize cost allocation."""
        plan = self._plans.get(plan_id)
        if not plan or not plan.cost_allocation:
            return False

        plan.cost_allocation.finalized = True
        plan.cost_allocation.finalized_at = datetime.utcnow()
        return True

    # =========================================================================
    # Schedule Management
    # =========================================================================

    def add_schedule(
        self,
        plan_id: str,
        phase: str,
        start_date: datetime,
        end_date: datetime,
        activities: list[str],
        is_remote: bool = True,
        location: str | None = None,
    ) -> AuditSchedule:
        """Add schedule to a plan."""
        plan = self._plans.get(plan_id)
        if not plan:
            raise ValueError(f"Plan not found: {plan_id}")

        schedule = AuditSchedule(
            schedule_id=str(uuid4()),
            audit_id=plan_id,
            phase=phase,
            start_date=start_date,
            end_date=end_date,
            activities=activities,
            is_remote=is_remote,
            location=location,
        )
        plan.schedules.append(schedule)
        return schedule

    def get_schedules(self, plan_id: str) -> list[AuditSchedule]:
        """Get schedules for a plan."""
        plan = self._plans.get(plan_id)
        if not plan:
            return []
        return plan.schedules

    # =========================================================================
    # Findings Management
    # =========================================================================

    def add_finding(
        self,
        plan_id: str,
        title: str,
        description: str,
        severity: str,
        affected_participants: list[str] | None = None,
        remediation_owner: str = "provider",
    ) -> AuditFinding:
        """Add finding to a plan."""
        plan = self._plans.get(plan_id)
        if not plan:
            raise ValueError(f"Plan not found: {plan_id}")

        finding = AuditFinding(
            finding_id=str(uuid4()),
            audit_id=plan_id,
            title=title,
            description=description,
            severity=severity,
            affected_participants=affected_participants or ["all"],
            identified_at=datetime.utcnow(),
            remediation_owner=remediation_owner,
        )
        plan.findings.append(finding)
        return finding

    def get_findings(
        self,
        plan_id: str,
        status: str | None = None,
    ) -> list[AuditFinding]:
        """Get findings for a plan."""
        plan = self._plans.get(plan_id)
        if not plan:
            return []

        findings = plan.findings
        if status:
            findings = [f for f in findings if f.status == status]

        return findings

    # =========================================================================
    # Reporting
    # =========================================================================

    def issue_report(
        self,
        plan_id: str,
        report_path: str,
    ) -> bool:
        """Mark audit report as issued."""
        plan = self._plans.get(plan_id)
        if not plan:
            return False

        plan.report_issued_at = datetime.utcnow()
        plan.report_path = report_path
        plan.status = AuditCoordinationStatus.COMPLETED
        plan.actual_end = datetime.utcnow()
        return True

    def get_coordination_summary(self, plan_id: str) -> dict[str, Any]:
        """Get summary of pooled audit coordination."""
        plan = self._plans.get(plan_id)
        if not plan:
            return {}

        active_participants = [p for p in plan.participants if p.status == "active"]
        signed_agreements = sum(1 for p in active_participants if p.has_signed_agreement)
        open_findings = sum(1 for f in plan.findings if f.status == "open")

        return {
            "plan_id": plan.plan_id,
            "title": plan.title,
            "audit_type": plan.audit_type,
            "status": plan.status.value,
            "participant_count": len(active_participants),
            "minimum_participants": plan.minimum_participants,
            "is_viable": plan.is_viable,
            "signed_agreements": signed_agreements,
            "all_agreements_signed": signed_agreements == len(active_participants),
            "estimated_cost": plan.estimated_cost,
            "cost_finalized": plan.cost_allocation.finalized if plan.cost_allocation else False,
            "schedule_phases": len(plan.schedules),
            "total_findings": len(plan.findings),
            "open_findings": open_findings,
            "report_issued": plan.report_issued_at is not None,
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_pooled_audit_coordination(
    minimum_participants: int = 3,
    maximum_participants: int = 10,
    **kwargs: Any,
) -> PooledAuditCoordinationService:
    """Create pooled audit coordination service instance."""
    config = CoordinationConfig(
        minimum_participants_default=minimum_participants,
        maximum_participants_default=maximum_participants,
        **kwargs,
    )
    return PooledAuditCoordinationService(config)
