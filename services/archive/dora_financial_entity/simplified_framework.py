"""
DORA Article 16 - Simplified ICT Risk Management Framework Module.

This module implements the simplified ICT risk management framework as specified in
DORA Article 16, providing proportionate requirements for smaller or less complex
financial entities that meet specific criteria.

DORA Article 16 Requirements:
- Simplified ICT risk management framework
- Proportionate requirements based on size and complexity
- Essential controls and documentation
- Reduced testing requirements
- Streamlined governance
- Basic incident management
- Simplified third-party management

Reference: Regulation (EU) 2022/2554, Article 16
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable
from pathlib import Path
import threading
import json
import uuid


class EntitySize(Enum):
    """Entity size classification for proportionality."""

    MICRO = "micro"
    SMALL = "small"
    MEDIUM = "medium"


class SimplifiedControlCategory(Enum):
    """Simplified control categories."""

    ACCESS_CONTROL = "access_control"
    DATA_PROTECTION = "data_protection"
    NETWORK_SECURITY = "network_security"
    SYSTEM_SECURITY = "system_security"
    BACKUP = "backup"
    INCIDENT_RESPONSE = "incident_response"
    AWARENESS = "awareness"


class ControlStatus(Enum):
    """Status of simplified controls."""

    NOT_IMPLEMENTED = "not_implemented"
    PARTIALLY_IMPLEMENTED = "partially_implemented"
    FULLY_IMPLEMENTED = "fully_implemented"
    NOT_APPLICABLE = "not_applicable"


class RiskLevel(Enum):
    """Simplified risk levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IncidentPriority(Enum):
    """Simplified incident priorities."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"


class IncidentStatus(Enum):
    """Incident status."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TestType(Enum):
    """Simplified testing types."""

    BACKUP_RESTORATION = "backup_restoration"
    INCIDENT_RESPONSE = "incident_response"
    ACCESS_REVIEW = "access_review"
    BASIC_VULNERABILITY = "basic_vulnerability"


@dataclass
class EligibilityCriteria:
    """
    Eligibility criteria for simplified framework per Article 16(1).

    Defines criteria for entities to qualify for the simplified framework.
    """

    criteria_id: str
    entity_name: str
    entity_type: str
    entity_size: EntitySize
    total_assets_eur: float
    annual_revenue_eur: float
    number_of_employees: int
    ict_spend_eur: float
    critical_functions_count: int
    third_party_providers_count: int
    cross_border_operations: bool
    significant_cyber_risk: bool
    part_of_group: bool
    group_uses_simplified: bool
    assessment_date: datetime
    eligible: bool
    eligibility_notes: str
    next_review_date: datetime
    assessed_by: str
    approved_by: str | None = None
    approval_date: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class SimplifiedControl:
    """
    Simplified control per Article 16(2).

    Essential controls for simplified framework compliance.
    """

    control_id: str
    name: str
    description: str
    category: SimplifiedControlCategory
    requirement_reference: str  # Article reference
    implementation_guidance: str
    evidence_required: list[str]
    status: ControlStatus
    implementation_notes: str | None = None
    evidence_location: str | None = None
    last_reviewed: datetime | None = None
    reviewer: str | None = None
    owner: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class SimplifiedRiskAssessment:
    """
    Simplified risk assessment per Article 16(3).

    Basic risk assessment for simplified framework entities.
    """

    assessment_id: str
    assessment_date: datetime
    assessor: str
    asset_name: str
    asset_type: str
    asset_description: str
    threat_description: str
    vulnerability_description: str
    existing_controls: list[str]
    risk_level: RiskLevel
    impact_description: str
    likelihood_description: str
    risk_treatment: str  # accept, mitigate, transfer, avoid
    treatment_actions: list[str]
    residual_risk: RiskLevel
    risk_owner: str
    review_date: datetime
    status: str = "active"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class SimplifiedIncident:
    """
    Simplified incident record per Article 16(4).

    Basic incident management for simplified framework entities.
    """

    incident_id: str
    title: str
    description: str
    priority: IncidentPriority
    status: IncidentStatus
    detected_date: datetime
    detected_by: str
    affected_systems: list[str]
    affected_services: list[str]
    impact_description: str
    initial_response: str
    resolution_actions: list[str] = field(default_factory=list)
    resolved_date: datetime | None = None
    resolved_by: str | None = None
    root_cause: str | None = None
    lessons_learned: str | None = None
    regulatory_notification_required: bool = False
    regulatory_notification_sent: bool = False
    notification_date: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class SimplifiedBackup:
    """
    Simplified backup record per Article 16(5).

    Basic backup management for simplified framework entities.
    """

    backup_id: str
    system_name: str
    backup_type: str  # full, incremental
    backup_date: datetime
    backup_location: str
    retention_days: int
    encryption_enabled: bool
    size_gb: float
    performed_by: str
    verification_date: datetime | None = None
    verification_result: str | None = None
    notes: str | None = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SimplifiedThirdParty:
    """
    Simplified third-party provider record per Article 16(6).

    Basic third-party management for simplified framework entities.
    """

    provider_id: str
    name: str
    service_description: str
    service_criticality: str  # critical, important, standard
    contract_start_date: datetime
    contract_end_date: datetime | None
    contact_name: str
    contact_email: str
    contact_phone: str
    data_processing: bool
    data_location: str | None
    security_assessment_date: datetime | None = None
    security_assessment_result: str | None = None
    last_review_date: datetime | None = None
    next_review_date: datetime | None = None
    exit_strategy: str | None = None
    status: str = "active"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class SimplifiedTest:
    """
    Simplified testing record per Article 16(7).

    Basic testing for simplified framework entities.
    """

    test_id: str
    test_type: TestType
    test_name: str
    description: str
    scheduled_date: datetime
    actual_date: datetime | None = None
    performed_by: str | None = None
    scope: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    passed: bool | None = None
    notes: str | None = None
    follow_up_required: bool = False
    follow_up_actions: list[str] = field(default_factory=list)
    status: str = "scheduled"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class SimplifiedAwarenessTraining:
    """
    Simplified awareness training record per Article 16(8).

    Basic security awareness for simplified framework entities.
    """

    training_id: str
    title: str
    description: str
    target_audience: str
    training_date: datetime
    duration_minutes: int
    trainer: str
    topics_covered: list[str]
    attendees: list[str]
    completion_rate: float
    assessment_passed: bool | None = None
    materials_location: str | None = None
    next_training_date: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AnnualReview:
    """
    Annual review of simplified framework per Article 16(9).

    Documents annual review of the simplified framework.
    """

    review_id: str
    review_year: int
    review_date: datetime
    reviewer: str
    eligibility_confirmed: bool
    controls_status: dict[str, str]  # control_id -> status
    risks_reviewed: int
    incidents_count: int
    tests_performed: int
    tests_passed: int
    third_parties_reviewed: int
    training_sessions: int
    key_findings: list[str]
    improvement_actions: list[str]
    management_approval: bool
    next_review_date: datetime
    approved_by: str | None = None
    approval_date: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)


# Essential controls required for simplified framework
ESSENTIAL_CONTROLS = [
    {
        "category": SimplifiedControlCategory.ACCESS_CONTROL,
        "name": "User Access Management",
        "description": "Manage user accounts and access rights",
        "requirement_reference": "Article 16(2)(a)",
        "implementation_guidance": "Implement unique user IDs, password policies, and regular access reviews",
        "evidence_required": [
            "User account list",
            "Access review records",
            "Password policy document",
        ],
    },
    {
        "category": SimplifiedControlCategory.ACCESS_CONTROL,
        "name": "Privileged Access Control",
        "description": "Control and monitor privileged access",
        "requirement_reference": "Article 16(2)(a)",
        "implementation_guidance": "Restrict admin access, use separate admin accounts, log privileged activities",
        "evidence_required": ["Privileged account list", "Activity logs", "Approval records"],
    },
    {
        "category": SimplifiedControlCategory.DATA_PROTECTION,
        "name": "Data Encryption",
        "description": "Encrypt sensitive data at rest and in transit",
        "requirement_reference": "Article 16(2)(b)",
        "implementation_guidance": "Use TLS for data in transit, encrypt sensitive data at rest",
        "evidence_required": ["Encryption configuration", "Certificate records"],
    },
    {
        "category": SimplifiedControlCategory.DATA_PROTECTION,
        "name": "Data Backup",
        "description": "Regular backup of critical data",
        "requirement_reference": "Article 16(2)(b)",
        "implementation_guidance": "Daily backups of critical data, monthly restoration tests",
        "evidence_required": ["Backup logs", "Restoration test records"],
    },
    {
        "category": SimplifiedControlCategory.NETWORK_SECURITY,
        "name": "Firewall Protection",
        "description": "Network perimeter protection",
        "requirement_reference": "Article 16(2)(c)",
        "implementation_guidance": "Configure firewalls to restrict unauthorized access",
        "evidence_required": ["Firewall configuration", "Rule review records"],
    },
    {
        "category": SimplifiedControlCategory.NETWORK_SECURITY,
        "name": "Malware Protection",
        "description": "Protection against malicious software",
        "requirement_reference": "Article 16(2)(c)",
        "implementation_guidance": "Deploy and maintain anti-malware on all systems",
        "evidence_required": ["Anti-malware deployment records", "Update logs"],
    },
    {
        "category": SimplifiedControlCategory.SYSTEM_SECURITY,
        "name": "Patch Management",
        "description": "Timely application of security patches",
        "requirement_reference": "Article 16(2)(d)",
        "implementation_guidance": "Apply critical patches within 30 days, regular patching schedule",
        "evidence_required": ["Patch logs", "Vulnerability reports"],
    },
    {
        "category": SimplifiedControlCategory.SYSTEM_SECURITY,
        "name": "Secure Configuration",
        "description": "Secure system configurations",
        "requirement_reference": "Article 16(2)(d)",
        "implementation_guidance": "Remove unnecessary services, change default credentials",
        "evidence_required": ["Configuration standards", "Configuration review records"],
    },
    {
        "category": SimplifiedControlCategory.INCIDENT_RESPONSE,
        "name": "Incident Response Procedure",
        "description": "Basic incident response capability",
        "requirement_reference": "Article 16(2)(e)",
        "implementation_guidance": "Document and test basic incident response procedures",
        "evidence_required": ["Incident response procedure", "Incident logs"],
    },
    {
        "category": SimplifiedControlCategory.AWARENESS,
        "name": "Security Awareness Training",
        "description": "Annual security awareness for all staff",
        "requirement_reference": "Article 16(2)(f)",
        "implementation_guidance": "Annual training covering phishing, passwords, data protection",
        "evidence_required": ["Training records", "Attendance lists"],
    },
]


class DORASimplifiedFramework:
    """
    DORA Article 16 Simplified ICT Risk Management Framework.

    Implements a proportionate framework for smaller or less complex
    financial entities that meet specific eligibility criteria.

    Per Article 16 Requirements:
    - Eligibility assessment
    - Essential controls
    - Simplified risk assessment
    - Basic incident management
    - Simplified backup and recovery
    - Basic third-party management
    - Proportionate testing
    - Annual reviews
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize DORA Simplified Framework.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._lock = threading.RLock()

        # Eligibility
        self._eligibility: EligibilityCriteria | None = None

        # Controls
        self._controls: dict[str, SimplifiedControl] = {}

        # Risk assessments
        self._risk_assessments: dict[str, SimplifiedRiskAssessment] = {}

        # Incidents
        self._incidents: dict[str, SimplifiedIncident] = {}

        # Backups
        self._backups: dict[str, SimplifiedBackup] = {}

        # Third parties
        self._third_parties: dict[str, SimplifiedThirdParty] = {}

        # Tests
        self._tests: dict[str, SimplifiedTest] = {}

        # Training
        self._training_sessions: dict[str, SimplifiedAwarenessTraining] = {}

        # Annual reviews
        self._annual_reviews: dict[str, AnnualReview] = {}

        # Event log
        self._event_log: list[dict] = []

        # Initialize essential controls
        self._initialize_essential_controls()

    def _log_event(self, event_type: str, details: dict) -> None:
        """Log framework events."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details,
        }
        self._event_log.append(event)

    def _initialize_essential_controls(self) -> None:
        """Initialize essential controls from template."""
        for control_def in ESSENTIAL_CONTROLS:
            control_id = f"SC-{uuid.uuid4().hex[:8].upper()}"
            control = SimplifiedControl(
                control_id=control_id,
                name=control_def["name"],
                description=control_def["description"],
                category=control_def["category"],
                requirement_reference=control_def["requirement_reference"],
                implementation_guidance=control_def["implementation_guidance"],
                evidence_required=control_def["evidence_required"],
                status=ControlStatus.NOT_IMPLEMENTED,
            )
            self._controls[control_id] = control

    # =========================================================================
    # Eligibility Assessment (Article 16(1))
    # =========================================================================

    def assess_eligibility(
        self,
        entity_name: str,
        entity_type: str,
        entity_size: EntitySize,
        total_assets_eur: float,
        annual_revenue_eur: float,
        number_of_employees: int,
        ict_spend_eur: float,
        critical_functions_count: int,
        third_party_providers_count: int,
        cross_border_operations: bool,
        significant_cyber_risk: bool,
        part_of_group: bool,
        group_uses_simplified: bool,
        assessed_by: str,
        next_review_date: datetime,
    ) -> EligibilityCriteria:
        """
        Assess eligibility for simplified framework per Article 16(1).

        Args:
            entity_name: Entity name
            entity_type: Type of financial entity
            entity_size: Size classification
            total_assets_eur: Total assets in EUR
            annual_revenue_eur: Annual revenue in EUR
            number_of_employees: Number of employees
            ict_spend_eur: Annual ICT spend in EUR
            critical_functions_count: Number of critical functions
            third_party_providers_count: Number of ICT third-party providers
            cross_border_operations: Whether entity operates cross-border
            significant_cyber_risk: Whether entity has significant cyber risk
            part_of_group: Whether entity is part of a group
            group_uses_simplified: Whether group uses simplified framework
            assessed_by: Person conducting assessment
            next_review_date: Next review date

        Returns:
            EligibilityCriteria instance
        """
        with self._lock:
            criteria_id = f"EC-{uuid.uuid4().hex[:8].upper()}"

            # Determine eligibility based on Article 16 criteria
            eligible = self._evaluate_eligibility(
                entity_size=entity_size,
                total_assets_eur=total_assets_eur,
                number_of_employees=number_of_employees,
                critical_functions_count=critical_functions_count,
                cross_border_operations=cross_border_operations,
                significant_cyber_risk=significant_cyber_risk,
                part_of_group=part_of_group,
                group_uses_simplified=group_uses_simplified,
            )

            eligibility_notes = self._generate_eligibility_notes(
                eligible=eligible,
                entity_size=entity_size,
                significant_cyber_risk=significant_cyber_risk,
                cross_border_operations=cross_border_operations,
            )

            eligibility = EligibilityCriteria(
                criteria_id=criteria_id,
                entity_name=entity_name,
                entity_type=entity_type,
                entity_size=entity_size,
                total_assets_eur=total_assets_eur,
                annual_revenue_eur=annual_revenue_eur,
                number_of_employees=number_of_employees,
                ict_spend_eur=ict_spend_eur,
                critical_functions_count=critical_functions_count,
                third_party_providers_count=third_party_providers_count,
                cross_border_operations=cross_border_operations,
                significant_cyber_risk=significant_cyber_risk,
                part_of_group=part_of_group,
                group_uses_simplified=group_uses_simplified,
                assessment_date=datetime.now(),
                eligible=eligible,
                eligibility_notes=eligibility_notes,
                next_review_date=next_review_date,
                assessed_by=assessed_by,
            )

            self._eligibility = eligibility

            self._log_event(
                "eligibility_assessed",
                {"criteria_id": criteria_id, "entity_name": entity_name, "eligible": eligible},
            )

            return eligibility

    def _evaluate_eligibility(
        self,
        entity_size: EntitySize,
        total_assets_eur: float,
        number_of_employees: int,
        critical_functions_count: int,
        cross_border_operations: bool,
        significant_cyber_risk: bool,
        part_of_group: bool,
        group_uses_simplified: bool,
    ) -> bool:
        """Evaluate eligibility based on Article 16 criteria."""
        # Basic size criteria
        if entity_size not in {EntitySize.MICRO, EntitySize.SMALL}:
            return False

        # Micro enterprise criteria (per EU definition)
        if entity_size == EntitySize.MICRO:
            if number_of_employees > 10 or total_assets_eur > 2_000_000:
                return False

        # Small enterprise criteria
        if entity_size == EntitySize.SMALL:
            if number_of_employees > 50 or total_assets_eur > 10_000_000:
                return False

        # Exclusion criteria
        if significant_cyber_risk:
            return False

        if cross_border_operations and critical_functions_count > 2:
            return False

        # Group considerations
        if part_of_group and not group_uses_simplified:
            return False

        return True

    def _generate_eligibility_notes(
        self,
        eligible: bool,
        entity_size: EntitySize,
        significant_cyber_risk: bool,
        cross_border_operations: bool,
    ) -> str:
        """Generate eligibility assessment notes."""
        notes = []

        if eligible:
            notes.append(
                f"Entity qualifies as {entity_size.value} enterprise for simplified framework."
            )
            notes.append("All eligibility criteria met per Article 16(1).")
        else:
            notes.append("Entity does not qualify for simplified framework.")
            if significant_cyber_risk:
                notes.append("Exclusion: Significant cyber risk profile identified.")
            if cross_border_operations:
                notes.append("Cross-border operations may require full framework.")

        return " ".join(notes)

    def get_eligibility(self) -> EligibilityCriteria | None:
        """Get current eligibility assessment."""
        with self._lock:
            return self._eligibility

    def approve_eligibility(self, approved_by: str) -> bool:
        """Approve eligibility assessment."""
        with self._lock:
            if not self._eligibility:
                return False

            self._eligibility.approved_by = approved_by
            self._eligibility.approval_date = datetime.now()
            self._eligibility.updated_at = datetime.now()

            self._log_event(
                "eligibility_approved",
                {"criteria_id": self._eligibility.criteria_id, "approved_by": approved_by},
            )

            return True

    def is_eligible(self) -> bool:
        """Check if entity is eligible for simplified framework."""
        with self._lock:
            return self._eligibility is not None and self._eligibility.eligible

    # =========================================================================
    # Simplified Controls (Article 16(2))
    # =========================================================================

    def get_all_controls(self) -> list[SimplifiedControl]:
        """Get all essential controls."""
        with self._lock:
            return list(self._controls.values())

    def get_control(self, control_id: str) -> SimplifiedControl | None:
        """Get control by ID."""
        with self._lock:
            return self._controls.get(control_id)

    def get_controls_by_category(
        self, category: SimplifiedControlCategory
    ) -> list[SimplifiedControl]:
        """Get controls by category."""
        with self._lock:
            return [c for c in self._controls.values() if c.category == category]

    def update_control_status(
        self,
        control_id: str,
        status: ControlStatus,
        implementation_notes: str | None = None,
        evidence_location: str | None = None,
        reviewer: str | None = None,
    ) -> bool:
        """Update control implementation status."""
        with self._lock:
            if control_id not in self._controls:
                return False

            control = self._controls[control_id]
            control.status = status
            control.last_reviewed = datetime.now()
            control.updated_at = datetime.now()

            if implementation_notes:
                control.implementation_notes = implementation_notes
            if evidence_location:
                control.evidence_location = evidence_location
            if reviewer:
                control.reviewer = reviewer

            self._log_event(
                "control_status_updated", {"control_id": control_id, "status": status.value}
            )

            return True

    def get_control_compliance_summary(self) -> dict:
        """Get summary of control compliance."""
        with self._lock:
            status_counts = {status.value: 0 for status in ControlStatus}
            for control in self._controls.values():
                status_counts[control.status.value] += 1

            total = len(self._controls)
            implemented = status_counts[ControlStatus.FULLY_IMPLEMENTED.value]
            partial = status_counts[ControlStatus.PARTIALLY_IMPLEMENTED.value]

            return {
                "total_controls": total,
                "status_counts": status_counts,
                "compliance_rate": implemented / total if total > 0 else 0,
                "partial_rate": partial / total if total > 0 else 0,
                "controls_needing_attention": [
                    c.control_id
                    for c in self._controls.values()
                    if c.status == ControlStatus.NOT_IMPLEMENTED
                ],
            }

    # =========================================================================
    # Simplified Risk Assessment (Article 16(3))
    # =========================================================================

    def create_risk_assessment(
        self,
        assessor: str,
        asset_name: str,
        asset_type: str,
        asset_description: str,
        threat_description: str,
        vulnerability_description: str,
        existing_controls: list[str],
        risk_level: RiskLevel,
        impact_description: str,
        likelihood_description: str,
        risk_treatment: str,
        treatment_actions: list[str],
        residual_risk: RiskLevel,
        risk_owner: str,
        review_date: datetime,
        **kwargs,
    ) -> SimplifiedRiskAssessment:
        """
        Create simplified risk assessment per Article 16(3).

        Args:
            assessor: Person conducting assessment
            asset_name: Name of asset at risk
            asset_type: Type of asset
            asset_description: Description of asset
            threat_description: Description of threat
            vulnerability_description: Description of vulnerability
            existing_controls: Existing controls
            risk_level: Assessed risk level
            impact_description: Description of potential impact
            likelihood_description: Description of likelihood
            risk_treatment: Treatment approach
            treatment_actions: Treatment actions
            residual_risk: Residual risk after treatment
            risk_owner: Person responsible for risk
            review_date: Next review date
            **kwargs: Additional attributes

        Returns:
            Created SimplifiedRiskAssessment instance
        """
        with self._lock:
            assessment_id = f"RA-{uuid.uuid4().hex[:8].upper()}"

            assessment = SimplifiedRiskAssessment(
                assessment_id=assessment_id,
                assessment_date=datetime.now(),
                assessor=assessor,
                asset_name=asset_name,
                asset_type=asset_type,
                asset_description=asset_description,
                threat_description=threat_description,
                vulnerability_description=vulnerability_description,
                existing_controls=existing_controls,
                risk_level=risk_level,
                impact_description=impact_description,
                likelihood_description=likelihood_description,
                risk_treatment=risk_treatment,
                treatment_actions=treatment_actions,
                residual_risk=residual_risk,
                risk_owner=risk_owner,
                review_date=review_date,
                **kwargs,
            )

            self._risk_assessments[assessment_id] = assessment

            self._log_event(
                "risk_assessment_created",
                {
                    "assessment_id": assessment_id,
                    "asset_name": asset_name,
                    "risk_level": risk_level.value,
                },
            )

            return assessment

    def get_risk_assessment(self, assessment_id: str) -> SimplifiedRiskAssessment | None:
        """Get risk assessment by ID."""
        with self._lock:
            return self._risk_assessments.get(assessment_id)

    def get_risks_by_level(self, risk_level: RiskLevel) -> list[SimplifiedRiskAssessment]:
        """Get risk assessments by level."""
        with self._lock:
            return [
                r
                for r in self._risk_assessments.values()
                if r.risk_level == risk_level and r.status == "active"
            ]

    def get_high_risks(self) -> list[SimplifiedRiskAssessment]:
        """Get high-level risks."""
        return self.get_risks_by_level(RiskLevel.HIGH)

    # =========================================================================
    # Simplified Incident Management (Article 16(4))
    # =========================================================================

    def report_incident(
        self,
        title: str,
        description: str,
        priority: IncidentPriority,
        detected_by: str,
        affected_systems: list[str],
        affected_services: list[str],
        impact_description: str,
        initial_response: str,
        **kwargs,
    ) -> SimplifiedIncident:
        """
        Report incident per Article 16(4).

        Args:
            title: Incident title
            description: Incident description
            priority: Incident priority
            detected_by: Person who detected
            affected_systems: Affected systems
            affected_services: Affected services
            impact_description: Description of impact
            initial_response: Initial response taken
            **kwargs: Additional attributes

        Returns:
            Created SimplifiedIncident instance
        """
        with self._lock:
            incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"

            # Determine if regulatory notification is required
            regulatory_required = priority == IncidentPriority.CRITICAL

            incident = SimplifiedIncident(
                incident_id=incident_id,
                title=title,
                description=description,
                priority=priority,
                status=IncidentStatus.OPEN,
                detected_date=datetime.now(),
                detected_by=detected_by,
                affected_systems=affected_systems,
                affected_services=affected_services,
                impact_description=impact_description,
                initial_response=initial_response,
                regulatory_notification_required=regulatory_required,
                **kwargs,
            )

            self._incidents[incident_id] = incident

            self._log_event(
                "incident_reported",
                {"incident_id": incident_id, "title": title, "priority": priority.value},
            )

            return incident

    def get_incident(self, incident_id: str) -> SimplifiedIncident | None:
        """Get incident by ID."""
        with self._lock:
            return self._incidents.get(incident_id)

    def update_incident_status(
        self,
        incident_id: str,
        status: IncidentStatus,
        resolution_actions: list[str] | None = None,
        resolved_by: str | None = None,
    ) -> bool:
        """Update incident status."""
        with self._lock:
            if incident_id not in self._incidents:
                return False

            incident = self._incidents[incident_id]
            incident.status = status
            incident.updated_at = datetime.now()

            if resolution_actions:
                incident.resolution_actions.extend(resolution_actions)

            if status == IncidentStatus.RESOLVED:
                incident.resolved_date = datetime.now()
                if resolved_by:
                    incident.resolved_by = resolved_by

            self._log_event(
                "incident_status_updated", {"incident_id": incident_id, "status": status.value}
            )

            return True

    def close_incident(self, incident_id: str, root_cause: str, lessons_learned: str) -> bool:
        """Close incident with root cause and lessons."""
        with self._lock:
            if incident_id not in self._incidents:
                return False

            incident = self._incidents[incident_id]
            incident.status = IncidentStatus.CLOSED
            incident.root_cause = root_cause
            incident.lessons_learned = lessons_learned
            incident.updated_at = datetime.now()

            self._log_event("incident_closed", {"incident_id": incident_id})

            return True

    def get_open_incidents(self) -> list[SimplifiedIncident]:
        """Get open incidents."""
        with self._lock:
            return [
                i
                for i in self._incidents.values()
                if i.status in {IncidentStatus.OPEN, IncidentStatus.IN_PROGRESS}
            ]

    # =========================================================================
    # Simplified Backup Management (Article 16(5))
    # =========================================================================

    def record_backup(
        self,
        system_name: str,
        backup_type: str,
        backup_location: str,
        retention_days: int,
        encryption_enabled: bool,
        size_gb: float,
        performed_by: str,
        notes: str | None = None,
    ) -> SimplifiedBackup:
        """
        Record backup per Article 16(5).

        Args:
            system_name: System backed up
            backup_type: Type of backup
            backup_location: Backup location
            retention_days: Retention period
            encryption_enabled: Whether encrypted
            size_gb: Backup size in GB
            performed_by: Person who performed backup
            notes: Additional notes

        Returns:
            Created SimplifiedBackup instance
        """
        with self._lock:
            backup_id = f"BK-{uuid.uuid4().hex[:8].upper()}"

            backup = SimplifiedBackup(
                backup_id=backup_id,
                system_name=system_name,
                backup_type=backup_type,
                backup_date=datetime.now(),
                backup_location=backup_location,
                retention_days=retention_days,
                encryption_enabled=encryption_enabled,
                size_gb=size_gb,
                performed_by=performed_by,
                notes=notes,
            )

            self._backups[backup_id] = backup

            self._log_event(
                "backup_recorded",
                {"backup_id": backup_id, "system_name": system_name, "backup_type": backup_type},
            )

            return backup

    def get_backup(self, backup_id: str) -> SimplifiedBackup | None:
        """Get backup by ID."""
        with self._lock:
            return self._backups.get(backup_id)

    def verify_backup(self, backup_id: str, verification_result: str) -> bool:
        """Record backup verification."""
        with self._lock:
            if backup_id not in self._backups:
                return False

            backup = self._backups[backup_id]
            backup.verification_date = datetime.now()
            backup.verification_result = verification_result

            self._log_event(
                "backup_verified", {"backup_id": backup_id, "result": verification_result}
            )

            return True

    def get_recent_backups(self, system_name: str, days: int = 30) -> list[SimplifiedBackup]:
        """Get recent backups for system."""
        with self._lock:
            cutoff = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff = cutoff.replace(day=cutoff.day - days) if cutoff.day > days else cutoff

            return [
                b
                for b in self._backups.values()
                if b.system_name == system_name and b.backup_date >= cutoff
            ]

    # =========================================================================
    # Simplified Third-Party Management (Article 16(6))
    # =========================================================================

    def register_third_party(
        self,
        name: str,
        service_description: str,
        service_criticality: str,
        contract_start_date: datetime,
        contact_name: str,
        contact_email: str,
        contact_phone: str,
        data_processing: bool,
        contract_end_date: datetime | None = None,
        data_location: str | None = None,
        exit_strategy: str | None = None,
        **kwargs,
    ) -> SimplifiedThirdParty:
        """
        Register third-party provider per Article 16(6).

        Args:
            name: Provider name
            service_description: Service description
            service_criticality: Criticality level
            contract_start_date: Contract start date
            contact_name: Contact person name
            contact_email: Contact email
            contact_phone: Contact phone
            data_processing: Whether provider processes data
            contract_end_date: Contract end date
            data_location: Data location
            exit_strategy: Exit strategy
            **kwargs: Additional attributes

        Returns:
            Created SimplifiedThirdParty instance
        """
        with self._lock:
            provider_id = f"TP-{uuid.uuid4().hex[:8].upper()}"

            provider = SimplifiedThirdParty(
                provider_id=provider_id,
                name=name,
                service_description=service_description,
                service_criticality=service_criticality,
                contract_start_date=contract_start_date,
                contract_end_date=contract_end_date,
                contact_name=contact_name,
                contact_email=contact_email,
                contact_phone=contact_phone,
                data_processing=data_processing,
                data_location=data_location,
                exit_strategy=exit_strategy,
                **kwargs,
            )

            self._third_parties[provider_id] = provider

            self._log_event(
                "third_party_registered",
                {"provider_id": provider_id, "name": name, "criticality": service_criticality},
            )

            return provider

    def get_third_party(self, provider_id: str) -> SimplifiedThirdParty | None:
        """Get third-party provider by ID."""
        with self._lock:
            return self._third_parties.get(provider_id)

    def update_third_party_review(
        self,
        provider_id: str,
        security_assessment_result: str | None = None,
        next_review_date: datetime | None = None,
    ) -> bool:
        """Update third-party review status."""
        with self._lock:
            if provider_id not in self._third_parties:
                return False

            provider = self._third_parties[provider_id]
            provider.last_review_date = datetime.now()
            provider.updated_at = datetime.now()

            if security_assessment_result:
                provider.security_assessment_date = datetime.now()
                provider.security_assessment_result = security_assessment_result

            if next_review_date:
                provider.next_review_date = next_review_date

            return True

    def get_critical_third_parties(self) -> list[SimplifiedThirdParty]:
        """Get critical third-party providers."""
        with self._lock:
            return [
                p
                for p in self._third_parties.values()
                if p.service_criticality == "critical" and p.status == "active"
            ]

    # =========================================================================
    # Simplified Testing (Article 16(7))
    # =========================================================================

    def schedule_test(
        self,
        test_type: TestType,
        test_name: str,
        description: str,
        scheduled_date: datetime,
        scope: list[str],
        **kwargs,
    ) -> SimplifiedTest:
        """
        Schedule simplified test per Article 16(7).

        Args:
            test_type: Type of test
            test_name: Test name
            description: Test description
            scheduled_date: Scheduled date
            scope: Test scope
            **kwargs: Additional attributes

        Returns:
            Created SimplifiedTest instance
        """
        with self._lock:
            test_id = f"ST-{uuid.uuid4().hex[:8].upper()}"

            test = SimplifiedTest(
                test_id=test_id,
                test_type=test_type,
                test_name=test_name,
                description=description,
                scheduled_date=scheduled_date,
                scope=scope,
                **kwargs,
            )

            self._tests[test_id] = test

            self._log_event(
                "test_scheduled",
                {
                    "test_id": test_id,
                    "test_type": test_type.value,
                    "scheduled_date": scheduled_date.isoformat(),
                },
            )

            return test

    def get_test(self, test_id: str) -> SimplifiedTest | None:
        """Get test by ID."""
        with self._lock:
            return self._tests.get(test_id)

    def complete_test(
        self,
        test_id: str,
        performed_by: str,
        findings: list[str],
        recommendations: list[str],
        passed: bool,
        notes: str | None = None,
        follow_up_actions: list[str] | None = None,
    ) -> bool:
        """Complete test with results."""
        with self._lock:
            if test_id not in self._tests:
                return False

            test = self._tests[test_id]
            test.actual_date = datetime.now()
            test.performed_by = performed_by
            test.findings = findings
            test.recommendations = recommendations
            test.passed = passed
            test.notes = notes
            test.status = "completed"
            test.updated_at = datetime.now()

            if follow_up_actions:
                test.follow_up_required = True
                test.follow_up_actions = follow_up_actions

            self._log_event("test_completed", {"test_id": test_id, "passed": passed})

            return True

    def get_scheduled_tests(self) -> list[SimplifiedTest]:
        """Get scheduled tests."""
        with self._lock:
            return [t for t in self._tests.values() if t.status == "scheduled"]

    # =========================================================================
    # Simplified Awareness Training (Article 16(8))
    # =========================================================================

    def record_training(
        self,
        title: str,
        description: str,
        target_audience: str,
        training_date: datetime,
        duration_minutes: int,
        trainer: str,
        topics_covered: list[str],
        attendees: list[str],
        completion_rate: float,
        assessment_passed: bool | None = None,
        materials_location: str | None = None,
        next_training_date: datetime | None = None,
    ) -> SimplifiedAwarenessTraining:
        """
        Record awareness training per Article 16(8).

        Args:
            title: Training title
            description: Training description
            target_audience: Target audience
            training_date: Training date
            duration_minutes: Duration in minutes
            trainer: Trainer name
            topics_covered: Topics covered
            attendees: List of attendees
            completion_rate: Completion rate
            assessment_passed: Whether assessment was passed
            materials_location: Location of materials
            next_training_date: Next training date

        Returns:
            Created SimplifiedAwarenessTraining instance
        """
        with self._lock:
            training_id = f"TR-{uuid.uuid4().hex[:8].upper()}"

            training = SimplifiedAwarenessTraining(
                training_id=training_id,
                title=title,
                description=description,
                target_audience=target_audience,
                training_date=training_date,
                duration_minutes=duration_minutes,
                trainer=trainer,
                topics_covered=topics_covered,
                attendees=attendees,
                completion_rate=completion_rate,
                assessment_passed=assessment_passed,
                materials_location=materials_location,
                next_training_date=next_training_date,
            )

            self._training_sessions[training_id] = training

            self._log_event(
                "training_recorded",
                {"training_id": training_id, "title": title, "attendees_count": len(attendees)},
            )

            return training

    def get_training(self, training_id: str) -> SimplifiedAwarenessTraining | None:
        """Get training by ID."""
        with self._lock:
            return self._training_sessions.get(training_id)

    def get_training_history(self, year: int | None = None) -> list[SimplifiedAwarenessTraining]:
        """Get training history."""
        with self._lock:
            if year:
                return [t for t in self._training_sessions.values() if t.training_date.year == year]
            return list(self._training_sessions.values())

    # =========================================================================
    # Annual Review (Article 16(9))
    # =========================================================================

    def create_annual_review(
        self,
        review_year: int,
        reviewer: str,
        eligibility_confirmed: bool,
        key_findings: list[str],
        improvement_actions: list[str],
        next_review_date: datetime,
    ) -> AnnualReview:
        """
        Create annual review per Article 16(9).

        Args:
            review_year: Year being reviewed
            reviewer: Person conducting review
            eligibility_confirmed: Whether eligibility confirmed
            key_findings: Key findings
            improvement_actions: Improvement actions
            next_review_date: Next review date

        Returns:
            Created AnnualReview instance
        """
        with self._lock:
            review_id = f"AR-{uuid.uuid4().hex[:8].upper()}"

            # Gather metrics
            controls_status = {c.control_id: c.status.value for c in self._controls.values()}

            incidents_count = len(
                [i for i in self._incidents.values() if i.detected_date.year == review_year]
            )

            tests = [t for t in self._tests.values() if t.scheduled_date.year == review_year]
            tests_performed = len([t for t in tests if t.status == "completed"])
            tests_passed = len([t for t in tests if t.passed])

            third_parties_reviewed = len(
                [
                    p
                    for p in self._third_parties.values()
                    if p.last_review_date and p.last_review_date.year == review_year
                ]
            )

            training_sessions = len(
                [t for t in self._training_sessions.values() if t.training_date.year == review_year]
            )

            review = AnnualReview(
                review_id=review_id,
                review_year=review_year,
                review_date=datetime.now(),
                reviewer=reviewer,
                eligibility_confirmed=eligibility_confirmed,
                controls_status=controls_status,
                risks_reviewed=len(self._risk_assessments),
                incidents_count=incidents_count,
                tests_performed=tests_performed,
                tests_passed=tests_passed,
                third_parties_reviewed=third_parties_reviewed,
                training_sessions=training_sessions,
                key_findings=key_findings,
                improvement_actions=improvement_actions,
                management_approval=False,
                next_review_date=next_review_date,
            )

            self._annual_reviews[review_id] = review

            self._log_event(
                "annual_review_created", {"review_id": review_id, "review_year": review_year}
            )

            return review

    def get_annual_review(self, review_id: str) -> AnnualReview | None:
        """Get annual review by ID."""
        with self._lock:
            return self._annual_reviews.get(review_id)

    def approve_annual_review(self, review_id: str, approved_by: str) -> bool:
        """Approve annual review."""
        with self._lock:
            if review_id not in self._annual_reviews:
                return False

            review = self._annual_reviews[review_id]
            review.management_approval = True
            review.approved_by = approved_by
            review.approval_date = datetime.now()

            self._log_event(
                "annual_review_approved", {"review_id": review_id, "approved_by": approved_by}
            )

            return True

    # =========================================================================
    # Dashboard and Metrics
    # =========================================================================

    def get_framework_metrics(self) -> dict:
        """Get simplified framework metrics."""
        with self._lock:
            control_compliance = self.get_control_compliance_summary()

            return {
                "eligibility": {
                    "assessed": self._eligibility is not None,
                    "eligible": self.is_eligible(),
                    "approved": (
                        self._eligibility.approved_by is not None if self._eligibility else False
                    ),
                },
                "controls": control_compliance,
                "risks": {
                    "total": len(self._risk_assessments),
                    "high": len(self.get_high_risks()),
                    "by_level": self._count_by_enum(self._risk_assessments.values(), "risk_level"),
                },
                "incidents": {
                    "total": len(self._incidents),
                    "open": len(self.get_open_incidents()),
                    "by_priority": self._count_by_enum(self._incidents.values(), "priority"),
                },
                "backups": {
                    "total": len(self._backups),
                    "verified": len([b for b in self._backups.values() if b.verification_date]),
                },
                "third_parties": {
                    "total": len(self._third_parties),
                    "critical": len(self.get_critical_third_parties()),
                    "active": len(
                        [p for p in self._third_parties.values() if p.status == "active"]
                    ),
                },
                "testing": {
                    "scheduled": len(self.get_scheduled_tests()),
                    "completed": len([t for t in self._tests.values() if t.status == "completed"]),
                    "pass_rate": self._calculate_test_pass_rate(),
                },
                "training": {
                    "sessions": len(self._training_sessions),
                    "this_year": len(
                        [
                            t
                            for t in self._training_sessions.values()
                            if t.training_date.year == datetime.now().year
                        ]
                    ),
                },
                "annual_reviews": {
                    "total": len(self._annual_reviews),
                    "approved": len(
                        [r for r in self._annual_reviews.values() if r.management_approval]
                    ),
                },
            }

    def _count_by_enum(self, items: Any, attr: str) -> dict[str, int]:
        """Count items by enum attribute."""
        counts: dict[str, int] = {}
        for item in items:
            value = getattr(item, attr)
            key = value.value if hasattr(value, "value") else str(value)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _calculate_test_pass_rate(self) -> float:
        """Calculate test pass rate."""
        completed = [t for t in self._tests.values() if t.status == "completed"]
        if not completed:
            return 0.0
        passed = len([t for t in completed if t.passed])
        return passed / len(completed)

    def generate_compliance_report(self) -> dict:
        """Generate simplified framework compliance report."""
        with self._lock:
            metrics = self.get_framework_metrics()

            return {
                "report_date": datetime.now().isoformat(),
                "framework_type": "simplified",
                "eligibility_status": metrics["eligibility"],
                "control_compliance": metrics["controls"],
                "risk_management": metrics["risks"],
                "incident_management": metrics["incidents"],
                "backup_status": metrics["backups"],
                "third_party_management": metrics["third_parties"],
                "testing_status": metrics["testing"],
                "training_status": metrics["training"],
                "annual_review_status": metrics["annual_reviews"],
                "overall_compliance_score": self._calculate_overall_compliance(),
            }

    def _calculate_overall_compliance(self) -> float:
        """Calculate overall compliance score."""
        scores = []

        # Control compliance (40% weight)
        control_summary = self.get_control_compliance_summary()
        scores.append(control_summary["compliance_rate"] * 0.4)

        # Risk management (15% weight)
        if self._risk_assessments:
            risk_score = 1.0 - (len(self.get_high_risks()) / len(self._risk_assessments))
            scores.append(risk_score * 0.15)

        # Incident management (15% weight)
        if self._incidents:
            closed = len([i for i in self._incidents.values() if i.status == IncidentStatus.CLOSED])
            scores.append((closed / len(self._incidents)) * 0.15)

        # Testing (15% weight)
        scores.append(self._calculate_test_pass_rate() * 0.15)

        # Training (15% weight)
        current_year = datetime.now().year
        training_this_year = len(
            [t for t in self._training_sessions.values() if t.training_date.year == current_year]
        )
        scores.append(min(training_this_year / 4, 1.0) * 0.15)  # Expect 4 sessions/year

        return sum(scores) if scores else 0.0

    # =========================================================================
    # Export and Persistence
    # =========================================================================

    def export_to_file(self, filepath: Path) -> bool:
        """Export framework data to JSONL file."""
        try:
            with open(filepath, "w") as f:
                for event in self._event_log:
                    f.write(json.dumps(event) + "\n")
            return True
        except Exception:
            return False

    def get_event_log(self) -> list[dict]:
        """Get event log."""
        with self._lock:
            return list(self._event_log)


# =============================================================================
# Factory Function
# =============================================================================


def create_dora_simplified_framework(config: dict | None = None) -> DORASimplifiedFramework:
    """
    Factory function to create DORA Simplified Framework.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured DORASimplifiedFramework instance
    """
    return DORASimplifiedFramework(config=config)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "EntitySize",
    "SimplifiedControlCategory",
    "ControlStatus",
    "RiskLevel",
    "IncidentPriority",
    "IncidentStatus",
    "TestType",
    # Dataclasses
    "EligibilityCriteria",
    "SimplifiedControl",
    "SimplifiedRiskAssessment",
    "SimplifiedIncident",
    "SimplifiedBackup",
    "SimplifiedThirdParty",
    "SimplifiedTest",
    "SimplifiedAwarenessTraining",
    "AnnualReview",
    # Constants
    "ESSENTIAL_CONTROLS",
    # Main class
    "DORASimplifiedFramework",
    # Factory
    "create_dora_simplified_framework",
]
