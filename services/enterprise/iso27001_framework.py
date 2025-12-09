# -*- coding: utf-8 -*-
"""
ISO 27001 Certification Framework.

DORA Phase 3 Block 3.14: ISO 27001 certification (start)

Provides ISO 27001:2022 certification management:
- Annex A control implementation tracking
- Information Security Management System (ISMS) framework
- Risk assessment methodology
- Certification audit preparation

DORA References:
    - Art. 28(5): Information security standards
    - Art. 6: ICT risk management framework
    - Art. 9: Protection and prevention
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


# =============================================================================
# Enums
# =============================================================================


class ISO27001Domain(Enum):
    """ISO 27001:2022 Annex A domains."""

    A5_ORGANIZATIONAL = "A.5"  # Organizational controls
    A6_PEOPLE = "A.6"  # People controls
    A7_PHYSICAL = "A.7"  # Physical controls
    A8_TECHNOLOGICAL = "A.8"  # Technological controls


class ControlObjective(Enum):
    """Control objective status."""

    NOT_STARTED = "not_started"
    DOCUMENTED = "documented"
    IMPLEMENTED = "implemented"
    OPERATING = "operating"
    OPTIMIZING = "optimizing"


class ImplementationStatus(Enum):
    """Control implementation status."""

    NOT_APPLICABLE = "not_applicable"
    NOT_IMPLEMENTED = "not_implemented"
    PARTIALLY_IMPLEMENTED = "partially_implemented"
    FULLY_IMPLEMENTED = "fully_implemented"
    NEEDS_IMPROVEMENT = "needs_improvement"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class ISO27001Control:
    """ISO 27001:2022 control definition."""

    control_id: str
    reference: str  # e.g., A.5.1
    domain: ISO27001Domain
    name: str
    description: str
    objective: str
    implementation_status: ImplementationStatus = ImplementationStatus.NOT_IMPLEMENTED
    control_objective: ControlObjective = ControlObjective.NOT_STARTED
    owner: str = ""
    implementation_notes: str = ""
    evidence_references: list[str] = field(default_factory=list)
    dora_mapping: list[str] = field(default_factory=list)
    last_review: datetime | None = None
    next_review: datetime | None = None

    @property
    def is_implemented(self) -> bool:
        """Check if control is implemented."""
        return self.implementation_status in (
            ImplementationStatus.FULLY_IMPLEMENTED,
            ImplementationStatus.PARTIALLY_IMPLEMENTED,
        )


@dataclass
class ControlImplementation:
    """Control implementation details."""

    implementation_id: str
    control_id: str
    description: str
    procedures: list[str]
    technologies: list[str]
    responsible_parties: list[str]
    implementation_date: datetime | None = None
    effectiveness_rating: int = 0  # 1-5 scale
    gaps_identified: list[str] = field(default_factory=list)


@dataclass
class RiskAssessment:
    """Information security risk assessment."""

    assessment_id: str
    title: str
    description: str
    asset: str
    threat: str
    vulnerability: str
    likelihood: int  # 1-5
    impact: int  # 1-5
    inherent_risk: int = 0  # likelihood * impact
    controls: list[str] = field(default_factory=list)  # Control IDs
    residual_likelihood: int = 0
    residual_impact: int = 0
    residual_risk: int = 0
    risk_treatment: str = ""  # accept, mitigate, transfer, avoid
    assessed_at: datetime = field(default_factory=datetime.utcnow)
    assessed_by: str = ""

    def calculate_risks(self) -> None:
        """Calculate inherent and residual risks."""
        self.inherent_risk = self.likelihood * self.impact
        self.residual_risk = self.residual_likelihood * self.residual_impact


@dataclass
class ISO27001Audit:
    """ISO 27001 audit record."""

    audit_id: str
    audit_type: str  # internal, surveillance, certification, recertification
    auditor: str
    audit_date: datetime
    scope: list[str]
    findings: list[dict[str, Any]] = field(default_factory=list)
    nonconformities: int = 0
    observations: int = 0
    recommendations: list[str] = field(default_factory=list)
    overall_result: str = ""  # pass, conditional, fail


@dataclass
class CertificationStatus:
    """ISO 27001 certification status."""

    is_certified: bool = False
    certification_body: str = ""
    certificate_number: str = ""
    initial_certification_date: datetime | None = None
    current_certificate_valid_from: datetime | None = None
    current_certificate_valid_until: datetime | None = None
    next_surveillance_audit: datetime | None = None
    scope_description: str = ""


@dataclass
class ISO27001Config:
    """ISO 27001 framework configuration."""

    organization_name: str
    isms_scope: str
    certification_body: str = ""
    review_frequency_days: int = 365
    risk_assessment_frequency_days: int = 365
    internal_audit_frequency_days: int = 365


# =============================================================================
# ISO 27001:2022 Control Library
# =============================================================================


def get_iso27001_control_library() -> list[dict[str, Any]]:
    """Get ISO 27001:2022 Annex A control library with DORA mappings."""
    return [
        # A.5 - Organizational Controls
        {
            "reference": "A.5.1",
            "domain": ISO27001Domain.A5_ORGANIZATIONAL,
            "name": "Policies for information security",
            "description": "Information security policy and topic-specific policies shall be defined, approved by management, published, communicated to and acknowledged by relevant personnel and relevant interested parties, and reviewed at planned intervals and if significant changes occur.",
            "objective": "Management direction for information security",
            "dora_mapping": ["Art. 5", "Art. 6"],
        },
        {
            "reference": "A.5.2",
            "domain": ISO27001Domain.A5_ORGANIZATIONAL,
            "name": "Information security roles and responsibilities",
            "description": "Information security roles and responsibilities shall be defined and allocated according to the organization needs.",
            "objective": "Organization of information security",
            "dora_mapping": ["Art. 5(2)", "Art. 5(3)"],
        },
        {
            "reference": "A.5.3",
            "domain": ISO27001Domain.A5_ORGANIZATIONAL,
            "name": "Segregation of duties",
            "description": "Conflicting duties and conflicting areas of responsibility shall be segregated.",
            "objective": "Organization of information security",
            "dora_mapping": ["Art. 9(3)"],
        },
        {
            "reference": "A.5.7",
            "domain": ISO27001Domain.A5_ORGANIZATIONAL,
            "name": "Threat intelligence",
            "description": "Information relating to information security threats shall be collected and analysed to produce threat intelligence.",
            "objective": "Information gathering",
            "dora_mapping": ["Art. 10", "Art. 45"],
        },
        {
            "reference": "A.5.8",
            "domain": ISO27001Domain.A5_ORGANIZATIONAL,
            "name": "Information security in project management",
            "description": "Information security shall be integrated into project management.",
            "objective": "Project security",
            "dora_mapping": ["Art. 7"],
        },
        {
            "reference": "A.5.23",
            "domain": ISO27001Domain.A5_ORGANIZATIONAL,
            "name": "Information security for use of cloud services",
            "description": "Processes for acquisition, use, management and exit from cloud services shall be established in accordance with the organization's information security requirements.",
            "objective": "Cloud security",
            "dora_mapping": ["Art. 28", "Art. 30"],
        },
        {
            "reference": "A.5.24",
            "domain": ISO27001Domain.A5_ORGANIZATIONAL,
            "name": "Information security incident management planning and preparation",
            "description": "The organization shall plan and prepare for managing information security incidents by defining, establishing and communicating information security incident management processes, roles and responsibilities.",
            "objective": "Incident management",
            "dora_mapping": ["Art. 17", "Art. 11"],
        },
        {
            "reference": "A.5.29",
            "domain": ISO27001Domain.A5_ORGANIZATIONAL,
            "name": "Information security during disruption",
            "description": "The organization shall plan how to maintain information security at an appropriate level during disruption.",
            "objective": "Business continuity",
            "dora_mapping": ["Art. 11", "Art. 15"],
        },
        {
            "reference": "A.5.30",
            "domain": ISO27001Domain.A5_ORGANIZATIONAL,
            "name": "ICT readiness for business continuity",
            "description": "ICT readiness shall be planned, implemented, maintained and tested based on business continuity objectives and ICT continuity requirements.",
            "objective": "ICT continuity",
            "dora_mapping": ["Art. 11", "Art. 12", "Art. 15"],
        },
        # A.6 - People Controls
        {
            "reference": "A.6.1",
            "domain": ISO27001Domain.A6_PEOPLE,
            "name": "Screening",
            "description": "Background verification checks on all candidates to become personnel shall be carried out prior to joining the organization and on an ongoing basis taking into consideration applicable laws, regulations and ethics and be proportional to the business requirements, the classification of the information to be accessed and the perceived risks.",
            "objective": "Human resource security",
            "dora_mapping": ["Art. 9(3)"],
        },
        {
            "reference": "A.6.3",
            "domain": ISO27001Domain.A6_PEOPLE,
            "name": "Information security awareness, education and training",
            "description": "Personnel of the organization and relevant interested parties shall receive appropriate information security awareness, education and training and regular updates of the organization's information security policy, topic-specific policies and procedures, as relevant for their job function.",
            "objective": "Security awareness",
            "dora_mapping": ["Art. 13(6)", "Art. 30(2)(i)"],
        },
        # A.7 - Physical Controls
        {
            "reference": "A.7.1",
            "domain": ISO27001Domain.A7_PHYSICAL,
            "name": "Physical security perimeters",
            "description": "Security perimeters shall be defined and used to protect areas that contain information and other associated assets.",
            "objective": "Physical security",
            "dora_mapping": ["Art. 9(2)"],
        },
        {
            "reference": "A.7.9",
            "domain": ISO27001Domain.A7_PHYSICAL,
            "name": "Security of assets off-premises",
            "description": "Off-site assets shall be protected.",
            "objective": "Asset protection",
            "dora_mapping": ["Art. 9"],
        },
        # A.8 - Technological Controls
        {
            "reference": "A.8.1",
            "domain": ISO27001Domain.A8_TECHNOLOGICAL,
            "name": "User endpoint devices",
            "description": "Information stored on, processed by or accessible via user endpoint devices shall be protected.",
            "objective": "Endpoint security",
            "dora_mapping": ["Art. 9(4)"],
        },
        {
            "reference": "A.8.5",
            "domain": ISO27001Domain.A8_TECHNOLOGICAL,
            "name": "Secure authentication",
            "description": "Secure authentication technologies and procedures shall be implemented based on information access restrictions and the topic-specific policy on access control.",
            "objective": "Access control",
            "dora_mapping": ["Art. 9(3)", "Art. 9(4)"],
        },
        {
            "reference": "A.8.8",
            "domain": ISO27001Domain.A8_TECHNOLOGICAL,
            "name": "Management of technical vulnerabilities",
            "description": "Information about technical vulnerabilities of information systems in use shall be obtained, the organization's exposure to such vulnerabilities shall be evaluated and appropriate measures shall be taken.",
            "objective": "Vulnerability management",
            "dora_mapping": ["Art. 8", "Art. 24"],
        },
        {
            "reference": "A.8.13",
            "domain": ISO27001Domain.A8_TECHNOLOGICAL,
            "name": "Information backup",
            "description": "Backup copies of information, software and systems shall be maintained and regularly tested in accordance with the agreed topic-specific policy on backup.",
            "objective": "Data backup",
            "dora_mapping": ["Art. 12"],
        },
        {
            "reference": "A.8.14",
            "domain": ISO27001Domain.A8_TECHNOLOGICAL,
            "name": "Redundancy of information processing facilities",
            "description": "Information processing facilities shall be implemented with redundancy sufficient to meet availability requirements.",
            "objective": "System availability",
            "dora_mapping": ["Art. 11", "Art. 15"],
        },
        {
            "reference": "A.8.15",
            "domain": ISO27001Domain.A8_TECHNOLOGICAL,
            "name": "Logging",
            "description": "Logs that record activities, exceptions, faults and other relevant events shall be produced, stored, protected and analysed.",
            "objective": "Logging and monitoring",
            "dora_mapping": ["Art. 10"],
        },
        {
            "reference": "A.8.16",
            "domain": ISO27001Domain.A8_TECHNOLOGICAL,
            "name": "Monitoring activities",
            "description": "Networks, systems and applications shall be monitored for anomalous behaviour and appropriate actions taken to evaluate potential information security incidents.",
            "objective": "Security monitoring",
            "dora_mapping": ["Art. 10"],
        },
        {
            "reference": "A.8.24",
            "domain": ISO27001Domain.A8_TECHNOLOGICAL,
            "name": "Use of cryptography",
            "description": "Rules for the effective use of cryptography, including cryptographic key management, shall be defined and implemented.",
            "objective": "Encryption",
            "dora_mapping": ["Art. 9(4)"],
        },
    ]


# =============================================================================
# Main Service Class
# =============================================================================


class ISO27001FrameworkService:
    """
    ISO 27001 Certification Framework Service.

    Manages ISO 27001:2022 certification per ISMS requirements.
    """

    def __init__(self, config: ISO27001Config) -> None:
        """Initialize ISO 27001 framework service."""
        self.config = config
        self._controls: dict[str, ISO27001Control] = {}
        self._implementations: dict[str, ControlImplementation] = {}
        self._risk_assessments: dict[str, RiskAssessment] = {}
        self._audits: dict[str, ISO27001Audit] = {}
        self.certification_status = CertificationStatus()
        self._initialize_controls()

    def _initialize_controls(self) -> None:
        """Initialize ISO 27001 controls from library."""
        for ctrl_def in get_iso27001_control_library():
            control = ISO27001Control(
                control_id=str(uuid4()),
                reference=ctrl_def["reference"],
                domain=ctrl_def["domain"],
                name=ctrl_def["name"],
                description=ctrl_def["description"],
                objective=ctrl_def["objective"],
                dora_mapping=ctrl_def.get("dora_mapping", []),
            )
            self._controls[control.control_id] = control

    # =========================================================================
    # Control Management
    # =========================================================================

    def get_control(self, control_id: str) -> ISO27001Control | None:
        """Get control by ID."""
        return self._controls.get(control_id)

    def get_control_by_reference(self, reference: str) -> ISO27001Control | None:
        """Get control by reference (e.g., A.5.1)."""
        for control in self._controls.values():
            if control.reference == reference:
                return control
        return None

    def list_controls(
        self,
        domain: ISO27001Domain | None = None,
        status: ImplementationStatus | None = None,
    ) -> list[ISO27001Control]:
        """List controls with optional filters."""
        controls = list(self._controls.values())

        if domain:
            controls = [c for c in controls if c.domain == domain]
        if status:
            controls = [c for c in controls if c.implementation_status == status]

        return controls

    def update_control_status(
        self,
        control_id: str,
        implementation_status: ImplementationStatus,
        notes: str = "",
    ) -> ISO27001Control | None:
        """Update control implementation status."""
        control = self._controls.get(control_id)
        if not control:
            return None

        control.implementation_status = implementation_status
        if notes:
            control.implementation_notes = notes
        control.last_review = datetime.utcnow()

        return control

    def assign_control_owner(self, control_id: str, owner: str) -> bool:
        """Assign owner to a control."""
        control = self._controls.get(control_id)
        if not control:
            return False
        control.owner = owner
        return True

    # =========================================================================
    # Implementation Management
    # =========================================================================

    def add_implementation(
        self,
        control_id: str,
        description: str,
        procedures: list[str],
        technologies: list[str],
        responsible_parties: list[str],
    ) -> ControlImplementation:
        """Add implementation details for a control."""
        if control_id not in self._controls:
            raise ValueError(f"Control not found: {control_id}")

        implementation = ControlImplementation(
            implementation_id=str(uuid4()),
            control_id=control_id,
            description=description,
            procedures=procedures,
            technologies=technologies,
            responsible_parties=responsible_parties,
            implementation_date=datetime.utcnow(),
        )
        self._implementations[implementation.implementation_id] = implementation
        return implementation

    def get_implementation(self, implementation_id: str) -> ControlImplementation | None:
        """Get implementation by ID."""
        return self._implementations.get(implementation_id)

    def list_implementations(self, control_id: str | None = None) -> list[ControlImplementation]:
        """List implementations with optional control filter."""
        implementations = list(self._implementations.values())
        if control_id:
            implementations = [i for i in implementations if i.control_id == control_id]
        return implementations

    # =========================================================================
    # Risk Assessment
    # =========================================================================

    def create_risk_assessment(
        self,
        title: str,
        description: str,
        asset: str,
        threat: str,
        vulnerability: str,
        likelihood: int,
        impact: int,
        assessed_by: str,
    ) -> RiskAssessment:
        """Create a risk assessment."""
        assessment = RiskAssessment(
            assessment_id=str(uuid4()),
            title=title,
            description=description,
            asset=asset,
            threat=threat,
            vulnerability=vulnerability,
            likelihood=likelihood,
            impact=impact,
            assessed_by=assessed_by,
        )
        assessment.calculate_risks()
        self._risk_assessments[assessment.assessment_id] = assessment
        return assessment

    def get_risk_assessment(self, assessment_id: str) -> RiskAssessment | None:
        """Get risk assessment by ID."""
        return self._risk_assessments.get(assessment_id)

    def list_risk_assessments(self, min_risk: int | None = None) -> list[RiskAssessment]:
        """List risk assessments with optional minimum risk filter."""
        assessments = list(self._risk_assessments.values())
        if min_risk:
            assessments = [a for a in assessments if a.inherent_risk >= min_risk]
        return sorted(assessments, key=lambda a: a.inherent_risk, reverse=True)

    def apply_controls_to_risk(
        self,
        assessment_id: str,
        control_ids: list[str],
        residual_likelihood: int,
        residual_impact: int,
        treatment: str,
    ) -> RiskAssessment | None:
        """Apply controls to reduce risk."""
        assessment = self._risk_assessments.get(assessment_id)
        if not assessment:
            return None

        assessment.controls = control_ids
        assessment.residual_likelihood = residual_likelihood
        assessment.residual_impact = residual_impact
        assessment.risk_treatment = treatment
        assessment.calculate_risks()
        return assessment

    # =========================================================================
    # Audit Management
    # =========================================================================

    def create_audit(
        self,
        audit_type: str,
        auditor: str,
        audit_date: datetime,
        scope: list[str],
    ) -> ISO27001Audit:
        """Create an audit record."""
        audit = ISO27001Audit(
            audit_id=str(uuid4()),
            audit_type=audit_type,
            auditor=auditor,
            audit_date=audit_date,
            scope=scope,
        )
        self._audits[audit.audit_id] = audit
        return audit

    def get_audit(self, audit_id: str) -> ISO27001Audit | None:
        """Get audit by ID."""
        return self._audits.get(audit_id)

    def add_audit_finding(
        self,
        audit_id: str,
        finding_type: str,  # nonconformity, observation, opportunity
        description: str,
        control_reference: str,
        severity: str,
    ) -> bool:
        """Add finding to an audit."""
        audit = self._audits.get(audit_id)
        if not audit:
            return False

        audit.findings.append({
            "type": finding_type,
            "description": description,
            "control_reference": control_reference,
            "severity": severity,
            "identified_at": datetime.utcnow().isoformat(),
        })

        if finding_type == "nonconformity":
            audit.nonconformities += 1
        elif finding_type == "observation":
            audit.observations += 1

        return True

    # =========================================================================
    # Compliance Status
    # =========================================================================

    def get_compliance_status(self) -> dict[str, Any]:
        """Get overall ISO 27001 compliance status."""
        controls = list(self._controls.values())
        total = len(controls)

        by_status = {}
        for status in ImplementationStatus:
            count = sum(1 for c in controls if c.implementation_status == status)
            by_status[status.value] = count

        by_domain = {}
        for domain in ISO27001Domain:
            domain_controls = [c for c in controls if c.domain == domain]
            implemented = sum(1 for c in domain_controls if c.is_implemented)
            by_domain[domain.value] = {
                "total": len(domain_controls),
                "implemented": implemented,
                "compliance_percent": (implemented / len(domain_controls) * 100) if domain_controls else 0,
            }

        implemented_controls = sum(1 for c in controls if c.is_implemented)
        high_risks = sum(1 for r in self._risk_assessments.values() if r.inherent_risk >= 15)

        return {
            "total_controls": total,
            "implemented_controls": implemented_controls,
            "overall_compliance_percent": (implemented_controls / total * 100) if total > 0 else 0,
            "by_status": by_status,
            "by_domain": by_domain,
            "high_risks": high_risks,
            "total_risk_assessments": len(self._risk_assessments),
            "certification_status": {
                "is_certified": self.certification_status.is_certified,
                "certificate_valid_until": (
                    self.certification_status.current_certificate_valid_until.isoformat()
                    if self.certification_status.current_certificate_valid_until
                    else None
                ),
            },
        }

    def get_dora_mapping_report(self) -> dict[str, list[str]]:
        """Get DORA to ISO 27001 control mapping report."""
        mapping: dict[str, list[str]] = {}

        for control in self._controls.values():
            for dora_ref in control.dora_mapping:
                if dora_ref not in mapping:
                    mapping[dora_ref] = []
                mapping[dora_ref].append(f"{control.reference}: {control.name}")

        return mapping


# =============================================================================
# Factory Functions
# =============================================================================


def create_iso27001_framework(
    organization_name: str,
    isms_scope: str,
    certification_body: str = "",
    **kwargs: Any,
) -> ISO27001FrameworkService:
    """Create ISO 27001 framework service instance."""
    config = ISO27001Config(
        organization_name=organization_name,
        isms_scope=isms_scope,
        certification_body=certification_body,
        **kwargs,
    )
    return ISO27001FrameworkService(config)
