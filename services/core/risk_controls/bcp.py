# -*- coding: utf-8 -*-
"""
MiFID II Business Continuity Plan - RTS 6 Article 3 Compliance.

This module implements business continuity requirements for algorithmic trading
per MiFID II RTS 6 Article 3:

"Investment firms engaged in algorithmic trading shall have effective business
continuity arrangements in place to deal with failures of its trading systems
including a pre-determined threshold on the number of trading disruptions
beyond which the firm shall cease trading."

Key Requirements:
    - Document all potential disruption scenarios
    - Define recovery procedures for each scenario
    - Set recovery time objectives (RTOs)
    - Maintain emergency contacts and escalation procedures
    - Test BCP at least annually
    - Integrate with kill switch procedures

References:
    - RTS 6 (Regulation 2017/589) Article 3: Business Continuity Arrangements
    - MiFID II Article 17: Algorithmic Trading
    - ESMA Guidelines on algorithmic trading (ESMA/2015/1464)
    - ISO 22301: Business Continuity Management Systems
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from typing import (
    Optional,
    List,
    Dict,
    Any,
    Tuple,
    Callable,
    Sequence,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class ScenarioCategory(Enum):
    """
    BCP scenario categories per RTS 6 Article 3.
    """

    SYSTEM_FAILURE = "system_failure"
    """Trading system hardware or software failure."""

    NETWORK_FAILURE = "network_failure"
    """Network connectivity issues."""

    DATA_CENTER = "data_center"
    """Data center or infrastructure failure."""

    MARKET_DATA = "market_data"
    """Market data feed failure or degradation."""

    EXCHANGE_CONNECTIVITY = "exchange_connectivity"
    """Loss of connection to trading venues."""

    CYBERSECURITY = "cybersecurity"
    """Cybersecurity incidents."""

    PERSONNEL = "personnel"
    """Key personnel unavailability."""

    THIRD_PARTY = "third_party"
    """Third-party service provider failure."""

    NATURAL_DISASTER = "natural_disaster"
    """Natural disaster affecting operations."""

    PANDEMIC = "pandemic"
    """Pandemic or health emergency."""

    REGULATORY = "regulatory"
    """Regulatory-mandated trading halt."""

    ALGORITHM_MALFUNCTION = "algorithm_malfunction"
    """Algorithm malfunction or erroneous behavior."""

    MARKET_DISRUPTION = "market_disruption"
    """Market-wide disruption or circuit breaker."""


class ImpactLevel(Enum):
    """Impact level of a disruption scenario."""

    CRITICAL = "critical"
    """Complete inability to trade (RTO < 15 minutes)."""

    HIGH = "high"
    """Major impact on trading operations (RTO < 1 hour)."""

    MEDIUM = "medium"
    """Moderate impact, degraded operations (RTO < 4 hours)."""

    LOW = "low"
    """Minor impact, workaround available (RTO < 8 hours)."""

    MINIMAL = "minimal"
    """Negligible impact (RTO < 24 hours)."""


class LikelihoodLevel(Enum):
    """Likelihood of a scenario occurring."""

    VERY_HIGH = "very_high"
    """Expected to occur multiple times per year."""

    HIGH = "high"
    """Expected to occur at least once per year."""

    MEDIUM = "medium"
    """Expected to occur at least once every 2-3 years."""

    LOW = "low"
    """Expected to occur once every 3-5 years."""

    VERY_LOW = "very_low"
    """Expected to occur less than once every 5 years."""


class RecoveryStatus(Enum):
    """Status of a recovery procedure."""

    NOT_TESTED = "not_tested"
    """Procedure has not been tested."""

    TESTED_PASS = "tested_pass"
    """Procedure tested successfully."""

    TESTED_FAIL = "tested_fail"
    """Procedure tested with issues found."""

    PARTIALLY_TESTED = "partially_tested"
    """Procedure partially tested."""

    IN_PROGRESS = "in_progress"
    """Recovery currently in progress."""

    COMPLETED = "completed"
    """Recovery completed."""


class AlertLevel(Enum):
    """BCP alert levels for escalation."""

    GREEN = "green"
    """Normal operations."""

    AMBER = "amber"
    """Potential issue, monitoring closely."""

    RED = "red"
    """Incident in progress, recovery initiated."""

    BLACK = "black"
    """Critical incident, trading halted."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EmergencyContact:
    """
    Emergency contact for BCP.

    Per RTS 6 Article 12, compliance staff must maintain contact
    with individuals who can execute emergency procedures.
    """

    contact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    role: str = ""
    email: str = ""
    phone_primary: str = ""
    phone_secondary: str = ""
    available_hours: str = "24/7"
    is_primary: bool = False
    categories: List[ScenarioCategory] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contact_id": self.contact_id,
            "name": self.name,
            "role": self.role,
            "email": self.email,
            "phone_primary": self.phone_primary,
            "phone_secondary": self.phone_secondary,
            "available_hours": self.available_hours,
            "is_primary": self.is_primary,
            "categories": [c.value for c in self.categories],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmergencyContact":
        """Create from dictionary."""
        return cls(
            contact_id=data.get("contact_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            role=data.get("role", ""),
            email=data.get("email", ""),
            phone_primary=data.get("phone_primary", ""),
            phone_secondary=data.get("phone_secondary", ""),
            available_hours=data.get("available_hours", "24/7"),
            is_primary=data.get("is_primary", False),
            categories=[ScenarioCategory(c) for c in data.get("categories", [])],
            notes=data.get("notes", ""),
        )


@dataclass
class RecoveryStep:
    """
    A step in a recovery procedure.
    """

    step_number: int = 0
    action: str = ""
    responsible_role: str = ""
    expected_duration_minutes: int = 0
    success_criteria: str = ""
    fallback_action: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_number": self.step_number,
            "action": self.action,
            "responsible_role": self.responsible_role,
            "expected_duration_minutes": self.expected_duration_minutes,
            "success_criteria": self.success_criteria,
            "fallback_action": self.fallback_action,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecoveryStep":
        """Create from dictionary."""
        return cls(
            step_number=data.get("step_number", 0),
            action=data.get("action", ""),
            responsible_role=data.get("responsible_role", ""),
            expected_duration_minutes=data.get("expected_duration_minutes", 0),
            success_criteria=data.get("success_criteria", ""),
            fallback_action=data.get("fallback_action", ""),
            notes=data.get("notes", ""),
        )


@dataclass
class RecoveryProcedure:
    """
    Recovery procedure for a BCP scenario.
    """

    procedure_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    steps: List[RecoveryStep] = field(default_factory=list)
    recovery_time_objective_minutes: int = 0
    recovery_point_objective_minutes: int = 0
    dependencies: List[str] = field(default_factory=list)
    status: RecoveryStatus = RecoveryStatus.NOT_TESTED
    last_test_date: Optional[date] = None
    last_test_result: str = ""
    owner: str = ""
    notes: str = ""

    def total_estimated_duration(self) -> int:
        """Calculate total estimated recovery duration in minutes."""
        return sum(step.expected_duration_minutes for step in self.steps)

    def meets_rto(self) -> bool:
        """Check if procedure can meet RTO."""
        return self.total_estimated_duration() <= self.recovery_time_objective_minutes

    def needs_testing(self, days_threshold: int = 365) -> bool:
        """Check if procedure needs testing."""
        if self.last_test_date is None:
            return True
        days_since_test = (date.today() - self.last_test_date).days
        return days_since_test >= days_threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "procedure_id": self.procedure_id,
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "recovery_time_objective_minutes": self.recovery_time_objective_minutes,
            "recovery_point_objective_minutes": self.recovery_point_objective_minutes,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "last_test_date": self.last_test_date.isoformat() if self.last_test_date else None,
            "last_test_result": self.last_test_result,
            "owner": self.owner,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecoveryProcedure":
        """Create from dictionary."""
        return cls(
            procedure_id=data.get("procedure_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            steps=[RecoveryStep.from_dict(s) for s in data.get("steps", [])],
            recovery_time_objective_minutes=data.get("recovery_time_objective_minutes", 0),
            recovery_point_objective_minutes=data.get("recovery_point_objective_minutes", 0),
            dependencies=data.get("dependencies", []),
            status=RecoveryStatus(data.get("status", "not_tested")),
            last_test_date=date.fromisoformat(data["last_test_date"]) if data.get("last_test_date") else None,
            last_test_result=data.get("last_test_result", ""),
            owner=data.get("owner", ""),
            notes=data.get("notes", ""),
        )


@dataclass
class BCPScenario:
    """
    Business continuity scenario.

    Per RTS 6 Article 3, firms must identify and plan for
    potential disruption scenarios.
    """

    scenario_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: ScenarioCategory = ScenarioCategory.SYSTEM_FAILURE

    # Risk assessment
    impact: ImpactLevel = ImpactLevel.MEDIUM
    likelihood: LikelihoodLevel = LikelihoodLevel.MEDIUM
    risk_score: float = 0.0  # Calculated from impact * likelihood

    # Response
    immediate_actions: List[str] = field(default_factory=list)
    recovery_procedure: Optional[RecoveryProcedure] = None
    responsible_roles: List[str] = field(default_factory=list)

    # Thresholds (per RTS 6 Article 3)
    trading_halt_threshold: str = ""
    resume_trading_criteria: str = ""

    # Testing
    last_drill_date: Optional[date] = None
    next_drill_due: Optional[date] = None
    drill_frequency_days: int = 365

    # Metadata
    created_date: Optional[date] = None
    last_modified: Optional[date] = None
    owner: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        """Calculate risk score after initialization."""
        self._calculate_risk_score()

    def _calculate_risk_score(self) -> None:
        """Calculate risk score from impact and likelihood."""
        impact_scores = {
            ImpactLevel.CRITICAL: 5,
            ImpactLevel.HIGH: 4,
            ImpactLevel.MEDIUM: 3,
            ImpactLevel.LOW: 2,
            ImpactLevel.MINIMAL: 1,
        }
        likelihood_scores = {
            LikelihoodLevel.VERY_HIGH: 5,
            LikelihoodLevel.HIGH: 4,
            LikelihoodLevel.MEDIUM: 3,
            LikelihoodLevel.LOW: 2,
            LikelihoodLevel.VERY_LOW: 1,
        }
        self.risk_score = impact_scores[self.impact] * likelihood_scores[self.likelihood]

    def needs_drill(self) -> bool:
        """Check if scenario needs a drill/test."""
        if self.last_drill_date is None:
            return True
        days_since_drill = (date.today() - self.last_drill_date).days
        return days_since_drill >= self.drill_frequency_days

    def record_drill(self, result: str = "") -> None:
        """Record a drill/test execution."""
        self.last_drill_date = date.today()
        self.next_drill_due = date.today() + timedelta(days=self.drill_frequency_days)
        if self.recovery_procedure:
            self.recovery_procedure.last_test_date = date.today()
            self.recovery_procedure.last_test_result = result
            self.recovery_procedure.status = (
                RecoveryStatus.TESTED_PASS if "success" in result.lower() or "pass" in result.lower()
                else RecoveryStatus.TESTED_FAIL if "fail" in result.lower()
                else RecoveryStatus.PARTIALLY_TESTED
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "impact": self.impact.value,
            "likelihood": self.likelihood.value,
            "risk_score": self.risk_score,
            "immediate_actions": self.immediate_actions,
            "recovery_procedure": self.recovery_procedure.to_dict() if self.recovery_procedure else None,
            "responsible_roles": self.responsible_roles,
            "trading_halt_threshold": self.trading_halt_threshold,
            "resume_trading_criteria": self.resume_trading_criteria,
            "last_drill_date": self.last_drill_date.isoformat() if self.last_drill_date else None,
            "next_drill_due": self.next_drill_due.isoformat() if self.next_drill_due else None,
            "drill_frequency_days": self.drill_frequency_days,
            "created_date": self.created_date.isoformat() if self.created_date else None,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "owner": self.owner,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BCPScenario":
        """Create from dictionary."""
        scenario = cls(
            scenario_id=data.get("scenario_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            category=ScenarioCategory(data.get("category", "system_failure")),
            impact=ImpactLevel(data.get("impact", "medium")),
            likelihood=LikelihoodLevel(data.get("likelihood", "medium")),
            immediate_actions=data.get("immediate_actions", []),
            responsible_roles=data.get("responsible_roles", []),
            trading_halt_threshold=data.get("trading_halt_threshold", ""),
            resume_trading_criteria=data.get("resume_trading_criteria", ""),
            drill_frequency_days=data.get("drill_frequency_days", 365),
            owner=data.get("owner", ""),
            notes=data.get("notes", ""),
        )

        # Parse optional fields
        if data.get("recovery_procedure"):
            scenario.recovery_procedure = RecoveryProcedure.from_dict(data["recovery_procedure"])
        if data.get("last_drill_date"):
            scenario.last_drill_date = date.fromisoformat(data["last_drill_date"])
        if data.get("next_drill_due"):
            scenario.next_drill_due = date.fromisoformat(data["next_drill_due"])
        if data.get("created_date"):
            scenario.created_date = date.fromisoformat(data["created_date"])
        if data.get("last_modified"):
            scenario.last_modified = date.fromisoformat(data["last_modified"])
        if data.get("risk_score"):
            scenario.risk_score = data["risk_score"]

        return scenario


@dataclass
class BCPIncident:
    """
    A recorded BCP incident.
    """

    incident_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scenario_id: str = ""
    timestamp_start_ns: int = field(default_factory=time.time_ns)
    timestamp_end_ns: Optional[int] = None
    alert_level: AlertLevel = AlertLevel.GREEN
    description: str = ""
    root_cause: str = ""
    impact_description: str = ""
    actions_taken: List[str] = field(default_factory=list)
    recovery_duration_minutes: Optional[int] = None
    trading_halted: bool = False
    trading_halt_duration_minutes: Optional[int] = None
    lessons_learned: str = ""
    follow_up_actions: List[str] = field(default_factory=list)
    reported_by: str = ""
    resolved_by: str = ""

    @property
    def is_resolved(self) -> bool:
        """Check if incident is resolved."""
        return self.timestamp_end_ns is not None

    def resolve(self, resolved_by: str, root_cause: str = "") -> None:
        """Mark incident as resolved."""
        self.timestamp_end_ns = time.time_ns()
        self.resolved_by = resolved_by
        if root_cause:
            self.root_cause = root_cause

        # Calculate recovery duration
        if self.timestamp_start_ns and self.timestamp_end_ns:
            duration_ns = self.timestamp_end_ns - self.timestamp_start_ns
            self.recovery_duration_minutes = int(duration_ns / 1e9 / 60)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "incident_id": self.incident_id,
            "scenario_id": self.scenario_id,
            "timestamp_start_ns": self.timestamp_start_ns,
            "timestamp_end_ns": self.timestamp_end_ns,
            "alert_level": self.alert_level.value,
            "description": self.description,
            "root_cause": self.root_cause,
            "impact_description": self.impact_description,
            "actions_taken": self.actions_taken,
            "recovery_duration_minutes": self.recovery_duration_minutes,
            "trading_halted": self.trading_halted,
            "trading_halt_duration_minutes": self.trading_halt_duration_minutes,
            "lessons_learned": self.lessons_learned,
            "follow_up_actions": self.follow_up_actions,
            "reported_by": self.reported_by,
            "resolved_by": self.resolved_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BCPIncident":
        """Create from dictionary."""
        return cls(
            incident_id=data.get("incident_id", str(uuid.uuid4())),
            scenario_id=data.get("scenario_id", ""),
            timestamp_start_ns=data.get("timestamp_start_ns", time.time_ns()),
            timestamp_end_ns=data.get("timestamp_end_ns"),
            alert_level=AlertLevel(data.get("alert_level", "green")),
            description=data.get("description", ""),
            root_cause=data.get("root_cause", ""),
            impact_description=data.get("impact_description", ""),
            actions_taken=data.get("actions_taken", []),
            recovery_duration_minutes=data.get("recovery_duration_minutes"),
            trading_halted=data.get("trading_halted", False),
            trading_halt_duration_minutes=data.get("trading_halt_duration_minutes"),
            lessons_learned=data.get("lessons_learned", ""),
            follow_up_actions=data.get("follow_up_actions", []),
            reported_by=data.get("reported_by", ""),
            resolved_by=data.get("resolved_by", ""),
        )


@dataclass
class BusinessContinuityPlan:
    """
    RTS 6 Article 3 Business Continuity Plan.

    Comprehensive BCP for algorithmic trading operations.

    Per RTS 6 Article 3:
    "Investment firms shall have effective business continuity arrangements
    in place to deal with failures of its trading systems."

    Attributes:
        plan_id: Unique identifier
        version: Plan version
        effective_date: When plan became effective
        firm_lei: Firm's LEI
        firm_name: Firm's legal name
        scenarios: List of BCP scenarios
        contacts: Emergency contacts
        incidents: Historical incidents
        current_alert_level: Current alert level
    """

    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: str = "1.0.0"
    effective_date: Optional[date] = None
    review_date: Optional[date] = None
    approved_by: str = ""

    # Firm identification
    firm_lei: str = ""
    firm_name: str = ""

    # Plan content
    scenarios: List[BCPScenario] = field(default_factory=list)
    contacts: List[EmergencyContact] = field(default_factory=list)
    escalation_matrix: Dict[str, List[str]] = field(default_factory=dict)

    # Incident tracking
    incidents: List[BCPIncident] = field(default_factory=list)
    current_alert_level: AlertLevel = AlertLevel.GREEN

    # Thresholds (per RTS 6 Article 3)
    max_disruptions_before_halt: int = 3
    disruption_window_hours: int = 24

    # Timestamps
    created_timestamp_ns: int = field(default_factory=time.time_ns)
    modified_timestamp_ns: int = field(default_factory=time.time_ns)

    def __post_init__(self) -> None:
        """Set default dates if not provided."""
        if self.effective_date is None:
            self.effective_date = date.today()
        if self.review_date is None:
            self.review_date = date.today() + timedelta(days=365)

    # =========================================================================
    # Scenario Management
    # =========================================================================

    def add_scenario(self, scenario: BCPScenario) -> None:
        """Add a scenario to the plan."""
        scenario.created_date = date.today()
        self.scenarios.append(scenario)
        self.modified_timestamp_ns = time.time_ns()

    def get_scenario(self, scenario_id: str) -> Optional[BCPScenario]:
        """Get scenario by ID."""
        for s in self.scenarios:
            if s.scenario_id == scenario_id:
                return s
        return None

    def get_scenarios_by_category(self, category: ScenarioCategory) -> List[BCPScenario]:
        """Get scenarios by category."""
        return [s for s in self.scenarios if s.category == category]

    def get_high_risk_scenarios(self, min_score: float = 15.0) -> List[BCPScenario]:
        """Get scenarios with risk score above threshold."""
        return [s for s in self.scenarios if s.risk_score >= min_score]

    def get_scenarios_needing_drill(self) -> List[BCPScenario]:
        """Get scenarios that need a drill/test."""
        return [s for s in self.scenarios if s.needs_drill()]

    # =========================================================================
    # Contact Management
    # =========================================================================

    def add_contact(self, contact: EmergencyContact) -> None:
        """Add emergency contact."""
        self.contacts.append(contact)
        self.modified_timestamp_ns = time.time_ns()

    def get_primary_contacts(self) -> List[EmergencyContact]:
        """Get primary emergency contacts."""
        return [c for c in self.contacts if c.is_primary]

    def get_contacts_for_category(self, category: ScenarioCategory) -> List[EmergencyContact]:
        """Get contacts responsible for a scenario category."""
        return [c for c in self.contacts if category in c.categories]

    # =========================================================================
    # Incident Management
    # =========================================================================

    def start_incident(
        self,
        scenario_id: str,
        description: str,
        reported_by: str,
        alert_level: AlertLevel = AlertLevel.AMBER,
    ) -> BCPIncident:
        """
        Start a new incident.

        Args:
            scenario_id: Related scenario ID.
            description: Incident description.
            reported_by: Person reporting the incident.
            alert_level: Initial alert level.

        Returns:
            New BCPIncident instance.
        """
        incident = BCPIncident(
            scenario_id=scenario_id,
            description=description,
            reported_by=reported_by,
            alert_level=alert_level,
        )
        self.incidents.append(incident)
        self.current_alert_level = alert_level
        self.modified_timestamp_ns = time.time_ns()

        logger.warning(
            f"BCP Incident started: {incident.incident_id} "
            f"- Alert Level: {alert_level.value}"
        )

        return incident

    def resolve_incident(
        self,
        incident_id: str,
        resolved_by: str,
        root_cause: str = "",
        lessons_learned: str = "",
    ) -> bool:
        """
        Resolve an incident.

        Args:
            incident_id: Incident to resolve.
            resolved_by: Person resolving.
            root_cause: Root cause analysis.
            lessons_learned: Lessons learned.

        Returns:
            True if resolved successfully.
        """
        for incident in self.incidents:
            if incident.incident_id == incident_id:
                incident.resolve(resolved_by, root_cause)
                incident.lessons_learned = lessons_learned

                # Reset alert level if no other active incidents
                active_incidents = [i for i in self.incidents if not i.is_resolved]
                if not active_incidents:
                    self.current_alert_level = AlertLevel.GREEN

                self.modified_timestamp_ns = time.time_ns()
                logger.info(f"BCP Incident resolved: {incident_id}")
                return True
        return False

    def get_active_incidents(self) -> List[BCPIncident]:
        """Get all active (unresolved) incidents."""
        return [i for i in self.incidents if not i.is_resolved]

    def recent_disruption_count(self) -> int:
        """
        Count disruptions in the configured window.

        Per RTS 6 Article 3: "pre-determined threshold on the number of
        trading disruptions beyond which the firm shall cease trading"
        """
        if not self.disruption_window_hours:
            return 0

        cutoff_ns = time.time_ns() - (self.disruption_window_hours * 3600 * 1e9)
        recent = [
            i for i in self.incidents
            if i.timestamp_start_ns >= cutoff_ns
        ]
        return len(recent)

    def should_halt_trading(self) -> Tuple[bool, str]:
        """
        Check if trading should be halted based on disruption count.

        Returns:
            Tuple of (should_halt, reason).
        """
        count = self.recent_disruption_count()
        if count >= self.max_disruptions_before_halt:
            return (
                True,
                f"Disruption threshold exceeded: {count} incidents in "
                f"last {self.disruption_window_hours} hours (max: {self.max_disruptions_before_halt})"
            )
        return (False, "")

    # =========================================================================
    # Metrics and Reporting
    # =========================================================================

    def get_plan_summary(self) -> Dict[str, Any]:
        """Get summary of the BCP."""
        scenarios_by_category = {}
        for cat in ScenarioCategory:
            scenarios_by_category[cat.value] = len(self.get_scenarios_by_category(cat))

        scenarios_by_impact = {}
        for impact in ImpactLevel:
            scenarios_by_impact[impact.value] = len([
                s for s in self.scenarios if s.impact == impact
            ])

        needing_drill = self.get_scenarios_needing_drill()
        high_risk = self.get_high_risk_scenarios()
        active_incidents = self.get_active_incidents()

        return {
            "plan_id": self.plan_id,
            "version": self.version,
            "effective_date": self.effective_date.isoformat() if self.effective_date else None,
            "review_date": self.review_date.isoformat() if self.review_date else None,
            "current_alert_level": self.current_alert_level.value,
            "total_scenarios": len(self.scenarios),
            "scenarios_by_category": scenarios_by_category,
            "scenarios_by_impact": scenarios_by_impact,
            "high_risk_scenarios": len(high_risk),
            "scenarios_needing_drill": len(needing_drill),
            "total_contacts": len(self.contacts),
            "primary_contacts": len(self.get_primary_contacts()),
            "total_incidents": len(self.incidents),
            "active_incidents": len(active_incidents),
            "recent_disruptions": self.recent_disruption_count(),
            "max_disruptions_threshold": self.max_disruptions_before_halt,
        }

    def get_drill_schedule(self) -> List[Dict[str, Any]]:
        """Get drill schedule for all scenarios."""
        schedule = []
        for scenario in self.scenarios:
            schedule.append({
                "scenario_id": scenario.scenario_id,
                "name": scenario.name,
                "category": scenario.category.value,
                "last_drill": scenario.last_drill_date.isoformat() if scenario.last_drill_date else None,
                "next_due": scenario.next_drill_due.isoformat() if scenario.next_drill_due else None,
                "overdue": scenario.needs_drill(),
                "risk_score": scenario.risk_score,
            })
        return sorted(schedule, key=lambda x: (not x["overdue"], x.get("next_due") or "9999"))

    def generate_bcp_document(self) -> str:
        """
        Generate formal BCP document.

        Returns:
            Formatted BCP document text.
        """
        summary = self.get_plan_summary()

        doc = f"""
================================================================================
BUSINESS CONTINUITY PLAN
MiFID II RTS 6 Article 3 Compliance
================================================================================

DOCUMENT CONTROL
----------------
Plan ID:            {self.plan_id}
Version:            {self.version}
Effective Date:     {self.effective_date}
Review Date:        {self.review_date}
Approved By:        {self.approved_by or 'Pending Approval'}

FIRM INFORMATION
----------------
Legal Name:         {self.firm_name}
LEI:                {self.firm_lei}
Current Alert:      {self.current_alert_level.value.upper()}

================================================================================
SECTION 1: SCOPE AND OBJECTIVES
================================================================================

1.1 Purpose
-----------
This Business Continuity Plan (BCP) establishes procedures to maintain
algorithmic trading operations during disruptions, per MiFID II RTS 6 Article 3.

1.2 Objectives
--------------
- Minimize trading disruption during incidents
- Protect firm and market integrity
- Support regulatory compliance requirements
- Enable rapid recovery to normal operations

1.3 Trading Halt Threshold
--------------------------
Per RTS 6 Article 3, trading will be halted after:
  {self.max_disruptions_before_halt} disruptions within {self.disruption_window_hours} hours

================================================================================
SECTION 2: EMERGENCY CONTACTS
================================================================================

"""
        # Add contacts
        for i, contact in enumerate(self.contacts, 1):
            primary_tag = " [PRIMARY]" if contact.is_primary else ""
            doc += f"""
{i}. {contact.name}{primary_tag}
   Role:          {contact.role}
   Phone:         {contact.phone_primary}
   Secondary:     {contact.phone_secondary}
   Email:         {contact.email}
   Available:     {contact.available_hours}
"""

        doc += """
================================================================================
SECTION 3: SCENARIO REGISTRY
================================================================================

"""
        # Add scenarios by category
        for category in ScenarioCategory:
            cat_scenarios = self.get_scenarios_by_category(category)
            if not cat_scenarios:
                continue

            doc += f"\n3.{list(ScenarioCategory).index(category) + 1} {category.value.upper().replace('_', ' ')}\n"
            doc += "-" * 60 + "\n"

            for scenario in cat_scenarios:
                doc += f"""
Scenario: {scenario.name} [{scenario.scenario_id}]
  Impact:      {scenario.impact.value.upper()}
  Likelihood:  {scenario.likelihood.value.upper()}
  Risk Score:  {scenario.risk_score:.0f}

  Description:
    {scenario.description}

  Immediate Actions:
"""
                for action in scenario.immediate_actions:
                    doc += f"    - {action}\n"

                if scenario.recovery_procedure:
                    doc += f"""
  Recovery Procedure:
    RTO: {scenario.recovery_procedure.recovery_time_objective_minutes} minutes
    RPO: {scenario.recovery_procedure.recovery_point_objective_minutes} minutes
    Steps:
"""
                    for step in scenario.recovery_procedure.steps:
                        doc += f"      {step.step_number}. {step.action}\n"

                doc += f"""
  Trading Halt Criteria:
    {scenario.trading_halt_threshold or 'As per general threshold'}

  Resume Trading Criteria:
    {scenario.resume_trading_criteria or 'Incident resolved and systems verified'}

"""

        doc += f"""
================================================================================
SECTION 4: ESCALATION MATRIX
================================================================================

Alert Levels:
  GREEN  - Normal operations
  AMBER  - Potential issue, monitoring
  RED    - Incident in progress
  BLACK  - Critical, trading halted

Current Status: {self.current_alert_level.value.upper()}

================================================================================
SECTION 5: TESTING AND DRILLS
================================================================================

Annual Drill Requirement: Yes (per RTS 6)
Scenarios Needing Drill: {summary['scenarios_needing_drill']}

Drill Schedule:
"""
        for entry in self.get_drill_schedule()[:10]:
            status = "OVERDUE" if entry["overdue"] else "Scheduled"
            doc += f"  - [{status}] {entry['name']}: Next due {entry.get('next_due', 'Not scheduled')}\n"

        doc += f"""
================================================================================
SECTION 6: INCIDENT HISTORY
================================================================================

Total Incidents:     {summary['total_incidents']}
Active Incidents:    {summary['active_incidents']}
Recent Disruptions:  {summary['recent_disruptions']} (in last {self.disruption_window_hours}h)

"""
        # Add recent incidents
        recent_incidents = sorted(
            self.incidents,
            key=lambda x: x.timestamp_start_ns,
            reverse=True
        )[:5]

        for incident in recent_incidents:
            status = "RESOLVED" if incident.is_resolved else "ACTIVE"
            doc += f"""
[{status}] {incident.incident_id}
  Started:  {datetime.fromtimestamp(incident.timestamp_start_ns / 1e9).isoformat()}
  Level:    {incident.alert_level.value.upper()}
  Desc:     {incident.description[:60]}...
"""

        doc += """
================================================================================
SECTION 7: DOCUMENT APPROVAL
================================================================================

This Business Continuity Plan has been reviewed and approved in accordance
with MiFID II RTS 6 Article 3 requirements.

"""
        doc += f"Prepared By:        ___________________\n"
        doc += f"Reviewed By:        ___________________\n"
        doc += f"Approved By:        {self.approved_by or '___________________'}\n"
        doc += f"Date:               {date.today().isoformat()}\n"

        doc += """
================================================================================
END OF BUSINESS CONTINUITY PLAN
================================================================================
"""
        return doc

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self) -> List[str]:
        """
        Validate BCP completeness.

        Returns:
            List of validation errors.
        """
        errors = []

        # Required fields
        if not self.firm_lei:
            errors.append("Missing firm_lei")

        if not self.approved_by:
            errors.append("Missing approved_by (senior management sign-off required)")

        # Must have scenarios for critical categories
        required_categories = {
            ScenarioCategory.SYSTEM_FAILURE,
            ScenarioCategory.NETWORK_FAILURE,
            ScenarioCategory.MARKET_DATA,
            ScenarioCategory.ALGORITHM_MALFUNCTION,
        }

        for cat in required_categories:
            if not self.get_scenarios_by_category(cat):
                errors.append(f"Missing scenarios for critical category: {cat.value}")

        # Must have at least one primary contact
        if not self.get_primary_contacts():
            errors.append("No primary emergency contact defined")

        # Check for scenarios needing drill
        overdue_drills = self.get_scenarios_needing_drill()
        if overdue_drills:
            errors.append(f"{len(overdue_drills)} scenarios have overdue drills")

        # Check if review is due
        if self.review_date and date.today() > self.review_date:
            errors.append("BCP review is overdue")

        return errors

    def is_valid(self) -> bool:
        """Check if BCP is valid."""
        return len(self.validate()) == 0

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "version": self.version,
            "effective_date": self.effective_date.isoformat() if self.effective_date else None,
            "review_date": self.review_date.isoformat() if self.review_date else None,
            "approved_by": self.approved_by,
            "firm_lei": self.firm_lei,
            "firm_name": self.firm_name,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "contacts": [c.to_dict() for c in self.contacts],
            "escalation_matrix": self.escalation_matrix,
            "incidents": [i.to_dict() for i in self.incidents],
            "current_alert_level": self.current_alert_level.value,
            "max_disruptions_before_halt": self.max_disruptions_before_halt,
            "disruption_window_hours": self.disruption_window_hours,
            "created_timestamp_ns": self.created_timestamp_ns,
            "modified_timestamp_ns": self.modified_timestamp_ns,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BusinessContinuityPlan":
        """Create from dictionary."""
        plan = cls(
            plan_id=data.get("plan_id", str(uuid.uuid4())),
            version=data.get("version", "1.0.0"),
            approved_by=data.get("approved_by", ""),
            firm_lei=data.get("firm_lei", ""),
            firm_name=data.get("firm_name", ""),
            escalation_matrix=data.get("escalation_matrix", {}),
            current_alert_level=AlertLevel(data.get("current_alert_level", "green")),
            max_disruptions_before_halt=data.get("max_disruptions_before_halt", 3),
            disruption_window_hours=data.get("disruption_window_hours", 24),
            created_timestamp_ns=data.get("created_timestamp_ns", time.time_ns()),
            modified_timestamp_ns=data.get("modified_timestamp_ns", time.time_ns()),
        )

        # Parse dates
        if data.get("effective_date"):
            plan.effective_date = date.fromisoformat(data["effective_date"])
        if data.get("review_date"):
            plan.review_date = date.fromisoformat(data["review_date"])

        # Parse collections
        plan.scenarios = [BCPScenario.from_dict(s) for s in data.get("scenarios", [])]
        plan.contacts = [EmergencyContact.from_dict(c) for c in data.get("contacts", [])]
        plan.incidents = [BCPIncident.from_dict(i) for i in data.get("incidents", [])]

        return plan

    @classmethod
    def from_json(cls, json_str: str) -> "BusinessContinuityPlan":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# Standard BCP Scenarios
# =============================================================================


def get_standard_bcp_scenarios() -> List[BCPScenario]:
    """
    Get standard BCP scenarios for algorithmic trading.

    These scenarios cover the most common disruption types
    per RTS 6 Article 3 requirements.

    Returns:
        List of pre-defined BCPScenario instances.
    """
    scenarios = []

    # =========================================================================
    # SYSTEM FAILURE SCENARIOS
    # =========================================================================

    scenarios.append(BCPScenario(
        scenario_id="BCP-SYS-001",
        name="Primary Trading System Failure",
        description="Complete failure of the primary trading system due to hardware, "
                    "software, or operating system issues.",
        category=ScenarioCategory.SYSTEM_FAILURE,
        impact=ImpactLevel.CRITICAL,
        likelihood=LikelihoodLevel.MEDIUM,
        immediate_actions=[
            "Activate kill switch to cancel all pending orders",
            "Notify trading desk and compliance",
            "Initiate failover to backup system",
            "Alert IT support for root cause analysis",
        ],
        recovery_procedure=RecoveryProcedure(
            name="System Failover Procedure",
            description="Failover to backup trading system",
            recovery_time_objective_minutes=15,
            recovery_point_objective_minutes=1,
            steps=[
                RecoveryStep(1, "Confirm primary system failure", "IT Operations", 2, "System status confirmed"),
                RecoveryStep(2, "Activate kill switch", "Trading Desk", 1, "All orders cancelled"),
                RecoveryStep(3, "Switch to backup system", "IT Operations", 5, "Backup system active"),
                RecoveryStep(4, "Verify connectivity to venues", "IT Operations", 3, "All venues connected"),
                RecoveryStep(5, "Resume trading with reduced limits", "Trading Desk", 2, "Trading resumed"),
                RecoveryStep(6, "Monitor system stability", "IT Operations", 60, "No issues for 1 hour"),
            ],
        ),
        responsible_roles=["Head of IT", "Trading Desk Manager", "Compliance Officer"],
        trading_halt_threshold="Immediate upon detection of system failure",
        resume_trading_criteria="Backup system stable for 15 minutes, all venues connected",
    ))

    scenarios.append(BCPScenario(
        scenario_id="BCP-SYS-002",
        name="Algorithm Malfunction",
        description="Trading algorithm exhibits erroneous behavior such as excessive "
                    "ordering, incorrect pricing, or unintended positions.",
        category=ScenarioCategory.ALGORITHM_MALFUNCTION,
        impact=ImpactLevel.CRITICAL,
        likelihood=LikelihoodLevel.HIGH,
        immediate_actions=[
            "Activate kill switch for affected algorithm",
            "Cancel all pending orders from the algorithm",
            "Disable algorithm in production",
            "Assess position impact and risk exposure",
            "Notify compliance and risk management",
        ],
        recovery_procedure=RecoveryProcedure(
            name="Algorithm Incident Response",
            description="Response to algorithm malfunction",
            recovery_time_objective_minutes=10,
            recovery_point_objective_minutes=0,
            steps=[
                RecoveryStep(1, "Trigger algorithm-specific kill switch", "Trading Desk", 1, "Algorithm stopped"),
                RecoveryStep(2, "Cancel all outstanding orders", "Trading System", 1, "Orders cancelled"),
                RecoveryStep(3, "Assess current positions", "Risk Management", 5, "Positions documented"),
                RecoveryStep(4, "Determine root cause", "Quant Team", 30, "Root cause identified"),
                RecoveryStep(5, "Implement fix and test", "Quant Team", 120, "Fix validated"),
                RecoveryStep(6, "Obtain approval for restart", "Compliance", 15, "Approval granted"),
            ],
        ),
        responsible_roles=["Quant Developer", "Risk Manager", "Trading Desk Manager"],
        trading_halt_threshold="Immediate when abnormal behavior detected",
        resume_trading_criteria="Root cause identified, fix tested, compliance approval",
    ))

    # =========================================================================
    # MARKET DATA SCENARIOS
    # =========================================================================

    scenarios.append(BCPScenario(
        scenario_id="BCP-MKT-001",
        name="Market Data Feed Failure",
        description="Loss of market data from primary data vendor affecting "
                    "ability to price and execute orders.",
        category=ScenarioCategory.MARKET_DATA,
        impact=ImpactLevel.HIGH,
        likelihood=LikelihoodLevel.HIGH,
        immediate_actions=[
            "Switch to backup market data feed",
            "Reduce position limits until resolved",
            "Alert trading desk of degraded mode",
            "Contact data vendor for status",
        ],
        recovery_procedure=RecoveryProcedure(
            name="Market Data Failover",
            description="Switch to backup market data",
            recovery_time_objective_minutes=5,
            recovery_point_objective_minutes=0,
            steps=[
                RecoveryStep(1, "Detect data feed failure", "Market Data System", 1, "Failure confirmed"),
                RecoveryStep(2, "Switch to backup feed", "IT Operations", 2, "Backup active"),
                RecoveryStep(3, "Verify data quality", "Trading Desk", 2, "Data quality confirmed"),
                RecoveryStep(4, "Resume normal operations", "Trading Desk", 1, "Normal mode"),
            ],
        ),
        responsible_roles=["Market Data Manager", "IT Operations", "Trading Desk"],
        trading_halt_threshold="If no backup feed available",
        resume_trading_criteria="Reliable market data restored",
    ))

    scenarios.append(BCPScenario(
        scenario_id="BCP-MKT-002",
        name="Market Data Stale/Delayed",
        description="Market data becoming stale or significantly delayed, "
                    "leading to incorrect pricing decisions.",
        category=ScenarioCategory.MARKET_DATA,
        impact=ImpactLevel.HIGH,
        likelihood=LikelihoodLevel.MEDIUM,
        immediate_actions=[
            "Activate stale data alerts",
            "Widen spreads or pause market making",
            "Switch to backup data source if available",
            "Reduce aggressive order types",
        ],
        recovery_procedure=RecoveryProcedure(
            name="Stale Data Response",
            description="Response to stale market data",
            recovery_time_objective_minutes=3,
            recovery_point_objective_minutes=0,
            steps=[
                RecoveryStep(1, "Detect stale data condition", "Market Data System", 1, "Staleness confirmed"),
                RecoveryStep(2, "Enable conservative trading mode", "Trading System", 1, "Mode activated"),
                RecoveryStep(3, "Attempt data feed refresh", "IT Operations", 2, "Feed status updated"),
            ],
        ),
        responsible_roles=["Market Data Manager", "Trading Desk"],
        trading_halt_threshold="Data stale for more than 5 seconds during market hours",
        resume_trading_criteria="Fresh data confirmed from reliable source",
    ))

    # =========================================================================
    # NETWORK SCENARIOS
    # =========================================================================

    scenarios.append(BCPScenario(
        scenario_id="BCP-NET-001",
        name="Network Connectivity Loss",
        description="Loss of network connectivity to trading venues, "
                    "preventing order submission and management.",
        category=ScenarioCategory.NETWORK_FAILURE,
        impact=ImpactLevel.CRITICAL,
        likelihood=LikelihoodLevel.MEDIUM,
        immediate_actions=[
            "Activate backup network connection",
            "Notify venues of potential orphan orders",
            "Contact network provider",
            "Prepare for manual order management",
        ],
        recovery_procedure=RecoveryProcedure(
            name="Network Failover",
            description="Switch to backup network",
            recovery_time_objective_minutes=5,
            recovery_point_objective_minutes=0,
            steps=[
                RecoveryStep(1, "Detect network failure", "Network Monitoring", 1, "Failure confirmed"),
                RecoveryStep(2, "Activate backup connection", "IT Operations", 2, "Backup active"),
                RecoveryStep(3, "Verify venue connectivity", "Trading System", 2, "All venues connected"),
                RecoveryStep(4, "Reconcile order status", "Trading Desk", 5, "Orders reconciled"),
            ],
        ),
        responsible_roles=["Network Manager", "IT Operations", "Trading Desk"],
        trading_halt_threshold="No connectivity to any venue for 30 seconds",
        resume_trading_criteria="Stable connection to all venues",
    ))

    # =========================================================================
    # EXCHANGE CONNECTIVITY SCENARIOS
    # =========================================================================

    scenarios.append(BCPScenario(
        scenario_id="BCP-EXC-001",
        name="Exchange Connection Lost",
        description="Loss of connection to one or more trading venues.",
        category=ScenarioCategory.EXCHANGE_CONNECTIVITY,
        impact=ImpactLevel.HIGH,
        likelihood=LikelihoodLevel.MEDIUM,
        immediate_actions=[
            "Attempt reconnection",
            "Route orders to alternative venues if available",
            "Track orphan orders on affected venue",
            "Contact exchange technical support",
        ],
        recovery_procedure=RecoveryProcedure(
            name="Exchange Reconnection",
            description="Reconnect to trading venue",
            recovery_time_objective_minutes=10,
            recovery_point_objective_minutes=0,
            steps=[
                RecoveryStep(1, "Detect disconnection", "Connection Monitor", 1, "Disconnection confirmed"),
                RecoveryStep(2, "Attempt automatic reconnection", "Trading System", 3, "Reconnection status"),
                RecoveryStep(3, "Verify order status on venue", "Trading Desk", 5, "Orders verified"),
                RecoveryStep(4, "Resume normal routing", "Trading System", 1, "Normal operations"),
            ],
        ),
        responsible_roles=["Trading Technology", "Trading Desk"],
        trading_halt_threshold="N/A - route to alternative venues",
        resume_trading_criteria="Stable connection re-established",
    ))

    # =========================================================================
    # CYBERSECURITY SCENARIOS
    # =========================================================================

    scenarios.append(BCPScenario(
        scenario_id="BCP-SEC-001",
        name="Cybersecurity Incident",
        description="Suspected or confirmed cybersecurity breach affecting "
                    "trading systems or data.",
        category=ScenarioCategory.CYBERSECURITY,
        impact=ImpactLevel.CRITICAL,
        likelihood=LikelihoodLevel.LOW,
        immediate_actions=[
            "Isolate affected systems immediately",
            "Activate kill switch",
            "Notify Information Security team",
            "Preserve evidence for forensic analysis",
            "Notify regulators if required",
        ],
        recovery_procedure=RecoveryProcedure(
            name="Cybersecurity Incident Response",
            description="Response to security breach",
            recovery_time_objective_minutes=60,
            recovery_point_objective_minutes=60,
            steps=[
                RecoveryStep(1, "Isolate affected systems", "IT Security", 5, "Systems isolated"),
                RecoveryStep(2, "Stop all trading activity", "Trading Desk", 2, "Trading stopped"),
                RecoveryStep(3, "Conduct forensic analysis", "IT Security", 240, "Analysis complete"),
                RecoveryStep(4, "Remediate vulnerabilities", "IT Security", 480, "Vulnerabilities fixed"),
                RecoveryStep(5, "Restore from clean backup", "IT Operations", 120, "Systems restored"),
                RecoveryStep(6, "Security verification", "IT Security", 60, "Security verified"),
            ],
        ),
        responsible_roles=["CISO", "IT Security", "Compliance Officer", "Legal"],
        trading_halt_threshold="Immediate upon suspected breach",
        resume_trading_criteria="Full security audit complete, regulator clearance if required",
    ))

    # =========================================================================
    # MARKET DISRUPTION SCENARIOS
    # =========================================================================

    scenarios.append(BCPScenario(
        scenario_id="BCP-MRK-001",
        name="Market Circuit Breaker Triggered",
        description="Exchange-wide or market-wide circuit breaker halts trading.",
        category=ScenarioCategory.MARKET_DISRUPTION,
        impact=ImpactLevel.MEDIUM,
        likelihood=LikelihoodLevel.MEDIUM,
        immediate_actions=[
            "Acknowledge circuit breaker status",
            "Review open positions and exposure",
            "Prepare for market reopening",
            "Monitor official announcements",
        ],
        recovery_procedure=RecoveryProcedure(
            name="Circuit Breaker Response",
            description="Response to market circuit breaker",
            recovery_time_objective_minutes=60,
            recovery_point_objective_minutes=0,
            steps=[
                RecoveryStep(1, "Confirm circuit breaker activation", "Trading Desk", 1, "Status confirmed"),
                RecoveryStep(2, "Review position risk", "Risk Management", 10, "Risk assessed"),
                RecoveryStep(3, "Prepare for reopening", "Trading Desk", 15, "Ready for reopening"),
                RecoveryStep(4, "Resume trading on reopening", "Trading System", 1, "Trading resumed"),
            ],
        ),
        responsible_roles=["Trading Desk", "Risk Management"],
        trading_halt_threshold="Per exchange circuit breaker rules",
        resume_trading_criteria="Market reopening confirmed by exchange",
    ))

    return scenarios


# =============================================================================
# Factory Functions
# =============================================================================


def create_business_continuity_plan(
    firm_lei: str,
    firm_name: str,
    approved_by: str = "",
    include_standard_scenarios: bool = True,
) -> BusinessContinuityPlan:
    """
    Factory function to create a new Business Continuity Plan.

    Args:
        firm_lei: Firm's Legal Entity Identifier.
        firm_name: Firm's legal name.
        approved_by: Senior manager approving the plan.
        include_standard_scenarios: Whether to include standard scenarios.

    Returns:
        New BusinessContinuityPlan instance.
    """
    plan = BusinessContinuityPlan(
        firm_lei=firm_lei,
        firm_name=firm_name,
        approved_by=approved_by,
    )

    if include_standard_scenarios:
        for scenario in get_standard_bcp_scenarios():
            plan.add_scenario(scenario)

    return plan


def load_bcp_from_file(file_path: str) -> BusinessContinuityPlan:
    """
    Load BCP from JSON file.

    Args:
        file_path: Path to JSON file.

    Returns:
        Loaded BusinessContinuityPlan.
    """
    with open(file_path, "r") as f:
        return BusinessContinuityPlan.from_json(f.read())


def save_bcp_to_file(plan: BusinessContinuityPlan, file_path: str) -> None:
    """
    Save BCP to JSON file.

    Args:
        plan: Plan to save.
        file_path: Path to output file.
    """
    with open(file_path, "w") as f:
        f.write(plan.to_json())


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "ScenarioCategory",
    "ImpactLevel",
    "LikelihoodLevel",
    "RecoveryStatus",
    "AlertLevel",
    # Data classes
    "EmergencyContact",
    "RecoveryStep",
    "RecoveryProcedure",
    "BCPScenario",
    "BCPIncident",
    "BusinessContinuityPlan",
    # Factory functions
    "create_business_continuity_plan",
    "load_bcp_from_file",
    "save_bcp_to_file",
    # Templates
    "get_standard_bcp_scenarios",
]
