# -*- coding: utf-8 -*-
"""
24/7 On-Call Management Service.

DORA Phase 3 Block 3.10: 24/7 on-call (Option C: 4+ engineers)

Provides enterprise on-call management capabilities:
- 24/7 engineer rotation scheduling
- Escalation policy management
- Incident assignment and tracking
- On-call metrics and SLA compliance

DORA References:
    - Art. 30(2)(f): Incident assistance obligations
    - Art. 11: Response and recovery requirements
    - Art. 17: ICT incident management
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


class OnCallTier(Enum):
    """On-call coverage tiers."""

    BASIC = "basic"  # Business hours only
    EXTENDED = "extended"  # Extended hours (8am-10pm)
    FULL = "full"  # 24/7 coverage


class EscalationLevel(Enum):
    """Escalation levels."""

    L1 = "l1"  # First responder
    L2 = "l2"  # Senior engineer
    L3 = "l3"  # Expert/Lead
    L4 = "l4"  # Management


class ShiftType(Enum):
    """On-call shift types."""

    PRIMARY = "primary"  # Primary on-call
    SECONDARY = "secondary"  # Backup on-call
    SHADOW = "shadow"  # Training/shadow


class IncidentPriority(Enum):
    """Incident priority levels."""

    P1 = "p1"  # Critical - immediate response
    P2 = "p2"  # High - 15 minute response
    P3 = "p3"  # Medium - 1 hour response
    P4 = "p4"  # Low - next business day


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class OnCallEngineer:
    """On-call engineer profile."""

    engineer_id: str
    name: str
    email: str
    phone: str
    timezone: str
    escalation_level: EscalationLevel
    skills: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    max_hours_per_week: int = 40
    current_hours_this_week: float = 0.0
    is_available: bool = True
    last_on_call: datetime | None = None

    @property
    def can_take_shift(self) -> bool:
        """Check if engineer can take more shifts."""
        return self.is_available and self.current_hours_this_week < self.max_hours_per_week


@dataclass
class OnCallShift:
    """On-call shift assignment."""

    shift_id: str
    engineer_id: str
    shift_type: ShiftType
    start_time: datetime
    end_time: datetime
    timezone: str
    notes: str = ""

    @property
    def duration_hours(self) -> float:
        """Calculate shift duration."""
        return (self.end_time - self.start_time).total_seconds() / 3600

    @property
    def is_active(self) -> bool:
        """Check if shift is currently active."""
        now = datetime.utcnow()
        return self.start_time <= now <= self.end_time


@dataclass
class OnCallSchedule:
    """On-call rotation schedule."""

    schedule_id: str
    name: str
    tier: OnCallTier
    rotation_period_days: int
    shifts: list[OnCallShift] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    effective_from: datetime = field(default_factory=datetime.utcnow)

    def get_current_on_call(self) -> list[OnCallShift]:
        """Get currently active shifts."""
        return [s for s in self.shifts if s.is_active]

    def get_primary_on_call(self) -> OnCallShift | None:
        """Get current primary on-call."""
        active = self.get_current_on_call()
        primaries = [s for s in active if s.shift_type == ShiftType.PRIMARY]
        return primaries[0] if primaries else None


@dataclass
class EscalationPolicy:
    """Escalation policy configuration."""

    policy_id: str
    name: str
    description: str
    levels: list[dict[str, Any]]  # Level configs with timeout
    default_priority: IncidentPriority = IncidentPriority.P3
    auto_escalate: bool = True
    notify_on_escalation: bool = True

    def get_timeout_minutes(self, level: EscalationLevel) -> int:
        """Get timeout for escalation level."""
        for lvl in self.levels:
            if lvl.get("level") == level.value:
                return lvl.get("timeout_minutes", 30)
        return 30  # Default


@dataclass
class IncidentAssignment:
    """Incident assignment to on-call engineer."""

    assignment_id: str
    incident_id: str
    engineer_id: str
    priority: IncidentPriority
    escalation_level: EscalationLevel
    assigned_at: datetime
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None
    escalated_at: datetime | None = None
    escalated_to: str | None = None
    response_time_seconds: float | None = None
    resolution_time_seconds: float | None = None

    def acknowledge(self) -> None:
        """Acknowledge the incident."""
        self.acknowledged_at = datetime.utcnow()
        self.response_time_seconds = (self.acknowledged_at - self.assigned_at).total_seconds()

    def resolve(self) -> None:
        """Mark incident as resolved."""
        self.resolved_at = datetime.utcnow()
        self.resolution_time_seconds = (self.resolved_at - self.assigned_at).total_seconds()

    def escalate(self, to_engineer_id: str) -> None:
        """Escalate to another engineer."""
        self.escalated_at = datetime.utcnow()
        self.escalated_to = to_engineer_id


@dataclass
class OnCallMetrics:
    """On-call performance metrics."""

    period_start: datetime
    period_end: datetime
    total_incidents: int
    acknowledged_within_sla: int
    resolved_within_sla: int
    escalations: int
    avg_response_time_seconds: float
    avg_resolution_time_seconds: float
    p95_response_time_seconds: float
    sla_compliance_percent: float

    @property
    def acknowledgment_rate(self) -> float:
        """Calculate acknowledgment rate."""
        if self.total_incidents == 0:
            return 100.0
        return (self.acknowledged_within_sla / self.total_incidents) * 100


@dataclass
class OnCallConfig:
    """On-call management configuration."""

    minimum_engineers: int = 4  # DORA requirement for 24/7
    rotation_period_days: int = 7
    max_consecutive_days: int = 7
    response_sla_p1_seconds: int = 300  # 5 minutes
    response_sla_p2_seconds: int = 900  # 15 minutes
    response_sla_p3_seconds: int = 3600  # 1 hour
    response_sla_p4_seconds: int = 86400  # 24 hours
    auto_escalation_enabled: bool = True


# =============================================================================
# Main Service Class
# =============================================================================


class OnCallManagementService:
    """
    24/7 On-Call Management Service.

    Provides enterprise on-call management per DORA Art. 30(2)(f).
    """

    def __init__(self, config: OnCallConfig | None = None) -> None:
        """Initialize on-call management service."""
        self.config = config or OnCallConfig()
        self._engineers: dict[str, OnCallEngineer] = {}
        self._schedules: dict[str, OnCallSchedule] = {}
        self._policies: dict[str, EscalationPolicy] = {}
        self._assignments: dict[str, IncidentAssignment] = {}
        self._initialize_default_policy()

    def _initialize_default_policy(self) -> None:
        """Initialize default escalation policy."""
        policy = EscalationPolicy(
            policy_id="default",
            name="Default Escalation Policy",
            description="Standard escalation for all incidents",
            levels=[
                {"level": "l1", "timeout_minutes": 15, "notify": ["primary_on_call"]},
                {"level": "l2", "timeout_minutes": 30, "notify": ["secondary_on_call", "team_lead"]},
                {"level": "l3", "timeout_minutes": 60, "notify": ["engineering_manager"]},
                {"level": "l4", "timeout_minutes": 120, "notify": ["vp_engineering", "cto"]},
            ],
        )
        self._policies[policy.policy_id] = policy

    # =========================================================================
    # Engineer Management
    # =========================================================================

    def add_engineer(
        self,
        name: str,
        email: str,
        phone: str,
        timezone: str,
        escalation_level: EscalationLevel,
        skills: list[str] | None = None,
    ) -> OnCallEngineer:
        """Add an on-call engineer."""
        engineer = OnCallEngineer(
            engineer_id=str(uuid4()),
            name=name,
            email=email,
            phone=phone,
            timezone=timezone,
            escalation_level=escalation_level,
            skills=skills or [],
        )
        self._engineers[engineer.engineer_id] = engineer
        return engineer

    def get_engineer(self, engineer_id: str) -> OnCallEngineer | None:
        """Get engineer by ID."""
        return self._engineers.get(engineer_id)

    def list_engineers(
        self,
        available_only: bool = False,
        level: EscalationLevel | None = None,
    ) -> list[OnCallEngineer]:
        """List engineers with optional filters."""
        engineers = list(self._engineers.values())
        if available_only:
            engineers = [e for e in engineers if e.is_available]
        if level:
            engineers = [e for e in engineers if e.escalation_level == level]
        return engineers

    def set_availability(self, engineer_id: str, is_available: bool) -> bool:
        """Set engineer availability."""
        engineer = self._engineers.get(engineer_id)
        if not engineer:
            return False
        engineer.is_available = is_available
        return True

    def get_team_capacity(self) -> dict[str, Any]:
        """Get current team capacity."""
        total = len(self._engineers)
        available = sum(1 for e in self._engineers.values() if e.is_available)

        return {
            "total_engineers": total,
            "available_engineers": available,
            "minimum_required": self.config.minimum_engineers,
            "is_adequate": available >= self.config.minimum_engineers,
            "capacity_percent": (available / self.config.minimum_engineers * 100) if self.config.minimum_engineers > 0 else 0,
        }

    # =========================================================================
    # Schedule Management
    # =========================================================================

    def create_schedule(
        self,
        name: str,
        tier: OnCallTier,
        rotation_period_days: int | None = None,
    ) -> OnCallSchedule:
        """Create an on-call schedule."""
        schedule = OnCallSchedule(
            schedule_id=str(uuid4()),
            name=name,
            tier=tier,
            rotation_period_days=rotation_period_days or self.config.rotation_period_days,
        )
        self._schedules[schedule.schedule_id] = schedule
        return schedule

    def get_schedule(self, schedule_id: str) -> OnCallSchedule | None:
        """Get schedule by ID."""
        return self._schedules.get(schedule_id)

    def add_shift(
        self,
        schedule_id: str,
        engineer_id: str,
        shift_type: ShiftType,
        start_time: datetime,
        end_time: datetime,
        timezone: str = "UTC",
    ) -> OnCallShift:
        """Add a shift to a schedule."""
        schedule = self._schedules.get(schedule_id)
        if not schedule:
            raise ValueError(f"Schedule not found: {schedule_id}")

        engineer = self._engineers.get(engineer_id)
        if not engineer:
            raise ValueError(f"Engineer not found: {engineer_id}")

        shift = OnCallShift(
            shift_id=str(uuid4()),
            engineer_id=engineer_id,
            shift_type=shift_type,
            start_time=start_time,
            end_time=end_time,
            timezone=timezone,
        )
        schedule.shifts.append(shift)

        # Update engineer hours
        engineer.current_hours_this_week += shift.duration_hours

        return shift

    def get_current_on_call(self, schedule_id: str | None = None) -> list[OnCallEngineer]:
        """Get currently on-call engineers."""
        if schedule_id:
            schedules = [self._schedules.get(schedule_id)]
        else:
            schedules = list(self._schedules.values())

        on_call_ids = set()
        for schedule in schedules:
            if schedule:
                for shift in schedule.get_current_on_call():
                    on_call_ids.add(shift.engineer_id)

        return [self._engineers[eid] for eid in on_call_ids if eid in self._engineers]

    def generate_rotation(
        self,
        schedule_id: str,
        start_date: datetime,
        weeks: int = 4,
    ) -> list[OnCallShift]:
        """Generate rotation for a schedule."""
        schedule = self._schedules.get(schedule_id)
        if not schedule:
            raise ValueError(f"Schedule not found: {schedule_id}")

        available_engineers = [e for e in self._engineers.values() if e.can_take_shift]
        if len(available_engineers) < self.config.minimum_engineers:
            raise ValueError(
                f"Not enough engineers. Need {self.config.minimum_engineers}, have {len(available_engineers)}"
            )

        shifts: list[OnCallShift] = []
        current_date = start_date

        for week in range(weeks):
            # Rotate through engineers
            primary_idx = week % len(available_engineers)
            secondary_idx = (week + 1) % len(available_engineers)

            week_start = current_date + timedelta(weeks=week)
            week_end = week_start + timedelta(days=7)

            # Primary shift
            primary_shift = OnCallShift(
                shift_id=str(uuid4()),
                engineer_id=available_engineers[primary_idx].engineer_id,
                shift_type=ShiftType.PRIMARY,
                start_time=week_start,
                end_time=week_end,
                timezone="UTC",
            )
            shifts.append(primary_shift)
            schedule.shifts.append(primary_shift)

            # Secondary shift
            secondary_shift = OnCallShift(
                shift_id=str(uuid4()),
                engineer_id=available_engineers[secondary_idx].engineer_id,
                shift_type=ShiftType.SECONDARY,
                start_time=week_start,
                end_time=week_end,
                timezone="UTC",
            )
            shifts.append(secondary_shift)
            schedule.shifts.append(secondary_shift)

        return shifts

    # =========================================================================
    # Escalation Policies
    # =========================================================================

    def create_policy(
        self,
        name: str,
        description: str,
        levels: list[dict[str, Any]],
    ) -> EscalationPolicy:
        """Create an escalation policy."""
        policy = EscalationPolicy(
            policy_id=str(uuid4()),
            name=name,
            description=description,
            levels=levels,
        )
        self._policies[policy.policy_id] = policy
        return policy

    def get_policy(self, policy_id: str) -> EscalationPolicy | None:
        """Get policy by ID."""
        return self._policies.get(policy_id)

    # =========================================================================
    # Incident Assignment
    # =========================================================================

    def assign_incident(
        self,
        incident_id: str,
        priority: IncidentPriority,
        schedule_id: str | None = None,
    ) -> IncidentAssignment:
        """Assign incident to on-call engineer."""
        # Get current primary on-call
        on_call = self.get_current_on_call(schedule_id)
        if not on_call:
            raise ValueError("No engineers currently on-call")

        # Find primary on-call
        primary = on_call[0]

        assignment = IncidentAssignment(
            assignment_id=str(uuid4()),
            incident_id=incident_id,
            engineer_id=primary.engineer_id,
            priority=priority,
            escalation_level=EscalationLevel.L1,
            assigned_at=datetime.utcnow(),
        )
        self._assignments[assignment.assignment_id] = assignment
        return assignment

    def get_assignment(self, assignment_id: str) -> IncidentAssignment | None:
        """Get assignment by ID."""
        return self._assignments.get(assignment_id)

    def acknowledge_incident(self, assignment_id: str) -> bool:
        """Acknowledge an incident."""
        assignment = self._assignments.get(assignment_id)
        if not assignment or assignment.acknowledged_at:
            return False
        assignment.acknowledge()
        return True

    def resolve_incident(self, assignment_id: str) -> bool:
        """Resolve an incident."""
        assignment = self._assignments.get(assignment_id)
        if not assignment or assignment.resolved_at:
            return False
        assignment.resolve()
        return True

    def escalate_incident(self, assignment_id: str, to_engineer_id: str) -> bool:
        """Escalate an incident."""
        assignment = self._assignments.get(assignment_id)
        if not assignment:
            return False

        to_engineer = self._engineers.get(to_engineer_id)
        if not to_engineer:
            return False

        assignment.escalate(to_engineer_id)

        # Update escalation level
        level_order = [EscalationLevel.L1, EscalationLevel.L2, EscalationLevel.L3, EscalationLevel.L4]
        current_idx = level_order.index(assignment.escalation_level)
        if current_idx < len(level_order) - 1:
            assignment.escalation_level = level_order[current_idx + 1]

        return True

    def get_sla_target(self, priority: IncidentPriority) -> int:
        """Get SLA target in seconds for priority."""
        sla_map = {
            IncidentPriority.P1: self.config.response_sla_p1_seconds,
            IncidentPriority.P2: self.config.response_sla_p2_seconds,
            IncidentPriority.P3: self.config.response_sla_p3_seconds,
            IncidentPriority.P4: self.config.response_sla_p4_seconds,
        }
        return sla_map.get(priority, self.config.response_sla_p3_seconds)

    # =========================================================================
    # Metrics
    # =========================================================================

    def calculate_metrics(
        self,
        period_start: datetime,
        period_end: datetime,
    ) -> OnCallMetrics:
        """Calculate on-call metrics for a period."""
        assignments = [
            a for a in self._assignments.values()
            if period_start <= a.assigned_at <= period_end
        ]

        if not assignments:
            return OnCallMetrics(
                period_start=period_start,
                period_end=period_end,
                total_incidents=0,
                acknowledged_within_sla=0,
                resolved_within_sla=0,
                escalations=0,
                avg_response_time_seconds=0,
                avg_resolution_time_seconds=0,
                p95_response_time_seconds=0,
                sla_compliance_percent=100.0,
            )

        response_times = [
            a.response_time_seconds for a in assignments
            if a.response_time_seconds is not None
        ]
        resolution_times = [
            a.resolution_time_seconds for a in assignments
            if a.resolution_time_seconds is not None
        ]

        acknowledged_within_sla = sum(
            1 for a in assignments
            if a.response_time_seconds is not None
            and a.response_time_seconds <= self.get_sla_target(a.priority)
        )

        resolved_within_sla = sum(
            1 for a in assignments
            if a.resolution_time_seconds is not None
        )

        escalations = sum(1 for a in assignments if a.escalated_at is not None)

        sorted_response = sorted(response_times) if response_times else [0]
        p95_idx = int(len(sorted_response) * 0.95)

        return OnCallMetrics(
            period_start=period_start,
            period_end=period_end,
            total_incidents=len(assignments),
            acknowledged_within_sla=acknowledged_within_sla,
            resolved_within_sla=resolved_within_sla,
            escalations=escalations,
            avg_response_time_seconds=sum(response_times) / len(response_times) if response_times else 0,
            avg_resolution_time_seconds=sum(resolution_times) / len(resolution_times) if resolution_times else 0,
            p95_response_time_seconds=sorted_response[p95_idx] if sorted_response else 0,
            sla_compliance_percent=(acknowledged_within_sla / len(assignments) * 100) if assignments else 100,
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_on_call_management(
    minimum_engineers: int = 4,
    auto_escalation_enabled: bool = True,
    **kwargs: Any,
) -> OnCallManagementService:
    """Create on-call management service instance."""
    config = OnCallConfig(
        minimum_engineers=minimum_engineers,
        auto_escalation_enabled=auto_escalation_enabled,
        **kwargs,
    )
    return OnCallManagementService(config)
