# -*- coding: utf-8 -*-
"""
On-Call Rotation System (Block 2.10).

Implements formal on-call rotation management:
- Option B: Business hours coverage (9x5)
- Option C: Extended coverage (24/7)
- Escalation paths
- Incident assignment

DORA References:
    - Article 11: Response and Recovery
    - Article 14: Communication
    - Article 17: ICT Incident Management
    - RTS CDR 2024/1774: Incident response requirements
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class OnCallTier(Enum):
    """On-call tier options."""

    OPTION_A = "option_a"  # No formal rotation (not recommended)
    OPTION_B = "option_b"  # Business hours (9x5) - 2 engineers
    OPTION_C = "option_c"  # 24/7 coverage - 4+ engineers


class RotationSchedule(Enum):
    """Rotation schedule types."""

    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    DAILY = "daily"
    CUSTOM = "custom"


class EscalationPath(Enum):
    """Escalation path levels."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    MANAGER = "manager"
    DIRECTOR = "director"
    EXECUTIVE = "executive"


class IncidentPriority(Enum):
    """Incident priority levels."""

    P1 = "P1"  # Critical - Immediate response
    P2 = "P2"  # High - 15 min response
    P3 = "P3"  # Medium - 1 hour response
    P4 = "P4"  # Low - Next business day


@dataclass
class OnCallEngineer:
    """On-call engineer record."""

    engineer_id: str = ""
    name: str = ""
    email: str = ""
    phone: str = ""
    slack_handle: str = ""
    team: str = ""
    skill_level: str = "senior"  # junior, mid, senior, lead
    certifications: List[str] = field(default_factory=list)
    max_consecutive_days: int = 7
    is_available: bool = True
    timezone: str = "UTC"

    def __post_init__(self):
        if not self.engineer_id:
            self.engineer_id = f"ENG-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class OnCallShift:
    """On-call shift assignment."""

    shift_id: str = ""
    engineer_id: str = ""
    engineer_name: str = ""
    escalation_path: EscalationPath = EscalationPath.PRIMARY

    # Schedule
    start_time: str = ""
    end_time: str = ""
    is_weekend: bool = False
    is_holiday: bool = False

    # Status
    is_active: bool = True
    handoff_notes: str = ""
    incidents_handled: int = 0

    def __post_init__(self):
        if not self.shift_id:
            self.shift_id = f"SHIFT-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class EscalationRule:
    """Escalation rule configuration."""

    rule_id: str = ""
    priority: IncidentPriority = IncidentPriority.P2
    escalation_path: List[EscalationPath] = field(default_factory=list)
    initial_response_minutes: int = 15
    escalation_intervals_minutes: List[int] = field(default_factory=lambda: [15, 30, 60])

    def __post_init__(self):
        if not self.rule_id:
            self.rule_id = f"ESCRULE-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class OnCallIncident:
    """Incident assigned to on-call."""

    incident_id: str = ""
    title: str = ""
    description: str = ""
    priority: IncidentPriority = IncidentPriority.P2

    # Assignment
    assigned_engineer_id: str = ""
    assigned_at: str = ""
    escalation_level: int = 0

    # Timing
    created_at: str = ""
    acknowledged_at: str = ""
    resolved_at: str = ""

    # SLA
    response_sla_minutes: int = 15
    response_met: bool = False
    resolution_sla_minutes: int = 240
    resolution_met: bool = False

    def __post_init__(self):
        if not self.incident_id:
            self.incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class OnCallRotationConfig:
    """Configuration for OnCallRotationManager."""

    tier: OnCallTier = OnCallTier.OPTION_B
    rotation_schedule: RotationSchedule = RotationSchedule.WEEKLY
    business_hours_start: int = 9  # 9 AM
    business_hours_end: int = 18  # 6 PM
    business_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri
    default_response_sla: Dict[str, int] = field(
        default_factory=lambda: {
            "P1": 5,
            "P2": 15,
            "P3": 60,
            "P4": 480,  # 8 hours
        }
    )
    log_all_events: bool = True
    log_path: str = "logs/core/oncall"
    alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# Response time requirements by tier
TIER_REQUIREMENTS = {
    OnCallTier.OPTION_A: {
        "description": "No formal rotation",
        "coverage": "None",
        "min_engineers": 0,
        "response_time": "Best effort",
        "recommended": False,
    },
    OnCallTier.OPTION_B: {
        "description": "Business hours coverage",
        "coverage": "9x5",
        "min_engineers": 2,
        "response_time": "Within 15 minutes during business hours",
        "recommended": True,
    },
    OnCallTier.OPTION_C: {
        "description": "24/7 coverage",
        "coverage": "24/7",
        "min_engineers": 4,
        "response_time": "Within 15 minutes anytime",
        "recommended": True,
    },
}


class OnCallRotationManager:
    """On-Call Rotation Manager."""

    def __init__(self, config: Optional[OnCallRotationConfig] = None):
        self.config = config or OnCallRotationConfig()
        self._engineers: Dict[str, OnCallEngineer] = {}
        self._shifts: Dict[str, OnCallShift] = {}
        self._incidents: Dict[str, OnCallIncident] = {}
        self._escalation_rules: Dict[str, EscalationRule] = {}
        self._lock = threading.RLock()
        self._init_default_escalation_rules()
        logger.info(f"OnCallRotationManager initialized with tier: {self.config.tier.value}")

    def _init_default_escalation_rules(self) -> None:
        """Initialize default escalation rules."""
        for priority in IncidentPriority:
            response_time = self.config.default_response_sla.get(priority.value, 60)
            rule = EscalationRule(
                priority=priority,
                escalation_path=[
                    EscalationPath.PRIMARY,
                    EscalationPath.SECONDARY,
                    EscalationPath.MANAGER,
                ],
                initial_response_minutes=response_time,
            )
            self._escalation_rules[rule.rule_id] = rule

    def register_engineer(
        self,
        name: str,
        email: str,
        phone: str = "",
        slack_handle: str = "",
        team: str = "",
        skill_level: str = "senior",
    ) -> OnCallEngineer:
        """Register an on-call engineer."""
        engineer = OnCallEngineer(
            name=name,
            email=email,
            phone=phone,
            slack_handle=slack_handle,
            team=team,
            skill_level=skill_level,
        )
        with self._lock:
            self._engineers[engineer.engineer_id] = engineer
        return engineer

    def create_shift(
        self,
        engineer_id: str,
        start_time: str,
        end_time: str,
        escalation_path: EscalationPath = EscalationPath.PRIMARY,
    ) -> Optional[OnCallShift]:
        """Create an on-call shift."""
        with self._lock:
            if engineer_id not in self._engineers:
                return None

            engineer = self._engineers[engineer_id]

            shift = OnCallShift(
                engineer_id=engineer_id,
                engineer_name=engineer.name,
                start_time=start_time,
                end_time=end_time,
                escalation_path=escalation_path,
            )
            self._shifts[shift.shift_id] = shift

        return shift

    def get_current_oncall(
        self, escalation_path: EscalationPath = EscalationPath.PRIMARY
    ) -> Optional[OnCallEngineer]:
        """Get currently on-call engineer."""
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            active_shifts = [
                s
                for s in self._shifts.values()
                if s.is_active
                and s.escalation_path == escalation_path
                and s.start_time <= now <= s.end_time
            ]

            if not active_shifts:
                return None

            shift = active_shifts[0]
            return self._engineers.get(shift.engineer_id)

    def assign_incident(
        self,
        title: str,
        description: str,
        priority: IncidentPriority,
    ) -> OnCallIncident:
        """Assign incident to on-call engineer."""
        oncall = self.get_current_oncall(EscalationPath.PRIMARY)

        incident = OnCallIncident(
            title=title,
            description=description,
            priority=priority,
            assigned_engineer_id=oncall.engineer_id if oncall else "",
            assigned_at=datetime.now(timezone.utc).isoformat(),
            response_sla_minutes=self.config.default_response_sla.get(priority.value, 60),
        )

        with self._lock:
            self._incidents[incident.incident_id] = incident

        # Alert on-call
        if oncall and self.config.alert_callback:
            self.config.alert_callback(
                "incident_assigned",
                {
                    "incident_id": incident.incident_id,
                    "engineer": oncall.name,
                    "priority": priority.value,
                },
            )

        return incident

    def acknowledge_incident(self, incident_id: str, engineer_id: str) -> Optional[OnCallIncident]:
        """Acknowledge an incident."""
        with self._lock:
            if incident_id not in self._incidents:
                return None

            incident = self._incidents[incident_id]
            incident.acknowledged_at = datetime.now(timezone.utc).isoformat()

            # Check SLA
            created = datetime.fromisoformat(incident.created_at.replace("Z", "+00:00"))
            acknowledged = datetime.fromisoformat(incident.acknowledged_at.replace("Z", "+00:00"))
            response_minutes = (acknowledged - created).total_seconds() / 60
            incident.response_met = response_minutes <= incident.response_sla_minutes

        return incident

    def resolve_incident(
        self, incident_id: str, resolution_notes: str = ""
    ) -> Optional[OnCallIncident]:
        """Resolve an incident."""
        with self._lock:
            if incident_id not in self._incidents:
                return None

            incident = self._incidents[incident_id]
            incident.resolved_at = datetime.now(timezone.utc).isoformat()

            # Check resolution SLA
            created = datetime.fromisoformat(incident.created_at.replace("Z", "+00:00"))
            resolved = datetime.fromisoformat(incident.resolved_at.replace("Z", "+00:00"))
            resolution_minutes = (resolved - created).total_seconds() / 60
            incident.resolution_met = resolution_minutes <= incident.resolution_sla_minutes

        return incident

    def escalate_incident(self, incident_id: str) -> Optional[OnCallIncident]:
        """Escalate an incident."""
        with self._lock:
            if incident_id not in self._incidents:
                return None

            incident = self._incidents[incident_id]
            incident.escalation_level += 1

            # Get next escalation path
            paths = [
                EscalationPath.PRIMARY,
                EscalationPath.SECONDARY,
                EscalationPath.MANAGER,
                EscalationPath.DIRECTOR,
            ]
            if incident.escalation_level < len(paths):
                next_path = paths[incident.escalation_level]
                next_oncall = self.get_current_oncall(next_path)

                if next_oncall:
                    incident.assigned_engineer_id = next_oncall.engineer_id

        return incident

    def check_coverage_compliance(self) -> Dict[str, Any]:
        """Check if on-call coverage meets tier requirements."""
        tier_req = TIER_REQUIREMENTS[self.config.tier]

        with self._lock:
            engineer_count = len([e for e in self._engineers.values() if e.is_available])
            current_oncall = self.get_current_oncall()

        min_required = tier_req["min_engineers"]
        has_current_coverage = current_oncall is not None

        compliant = engineer_count >= min_required and (
            has_current_coverage or self.config.tier == OnCallTier.OPTION_A
        )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tier": self.config.tier.value,
            "tier_description": tier_req["description"],
            "compliant": compliant,
            "engineers": {
                "available": engineer_count,
                "required": min_required,
                "meets_requirement": engineer_count >= min_required,
            },
            "current_coverage": {
                "has_oncall": has_current_coverage,
                "engineer": current_oncall.name if current_oncall else None,
            },
            "coverage_type": tier_req["coverage"],
            "response_time_requirement": tier_req["response_time"],
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get on-call rotation summary."""
        compliance = self.check_coverage_compliance()

        with self._lock:
            incidents = list(self._incidents.values())
            recent = [
                i
                for i in incidents
                if i.created_at > (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            ]

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tier": self.config.tier.value,
            "compliance": compliance,
            "engineers": len(self._engineers),
            "active_shifts": len([s for s in self._shifts.values() if s.is_active]),
            "incidents_30d": {
                "total": len(recent),
                "response_sla_met": sum(1 for i in recent if i.response_met),
                "resolution_sla_met": sum(1 for i in recent if i.resolution_met),
            },
            "dora_compliance": {
                "article_11": "compliant" if compliance["compliant"] else "non_compliant",
                "article_17": "compliant" if compliance["compliant"] else "non_compliant",
            },
        }


def create_oncall_rotation_manager(
    config: Optional[OnCallRotationConfig] = None,
) -> OnCallRotationManager:
    """Create an OnCallRotationManager instance."""
    return OnCallRotationManager(config=config)
