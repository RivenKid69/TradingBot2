# -*- coding: utf-8 -*-
"""
SOC2 Type II Certification Framework.

DORA Phase 3 Block 3.11: Complete SOC2 Type II certification

Provides SOC2 Type II certification management:
- Trust Services Criteria (TSC) control mapping
- Evidence collection and management
- Audit finding tracking
- Certification lifecycle management

DORA References:
    - Art. 28(5): Information security standards
    - Art. 30(3)(e): Audit rights and access
    - Supports client audit requirements

SOC2 Trust Service Principles:
    - Security (Common Criteria)
    - Availability
    - Processing Integrity
    - Confidentiality
    - Privacy
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


class SOC2TrustPrinciple(Enum):
    """SOC2 Trust Service Principles."""

    SECURITY = "security"  # CC - Common Criteria
    AVAILABILITY = "availability"  # A
    PROCESSING_INTEGRITY = "processing_integrity"  # PI
    CONFIDENTIALITY = "confidentiality"  # C
    PRIVACY = "privacy"  # P


class ControlStatus(Enum):
    """Control implementation status."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    TESTED = "tested"
    EFFECTIVE = "effective"
    NOT_APPLICABLE = "not_applicable"


class EvidenceType(Enum):
    """Evidence types for SOC2 controls."""

    POLICY = "policy"
    PROCEDURE = "procedure"
    SCREENSHOT = "screenshot"
    LOG = "log"
    CONFIGURATION = "configuration"
    INTERVIEW = "interview"
    OBSERVATION = "observation"
    REPORT = "report"
    CERTIFICATE = "certificate"


class AuditStatus(Enum):
    """Audit status."""

    PLANNING = "planning"
    FIELDWORK = "fieldwork"
    REVIEW = "review"
    COMPLETED = "completed"
    REPORT_ISSUED = "report_issued"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SOC2Control:
    """SOC2 control definition."""

    control_id: str
    trust_principle: SOC2TrustPrinciple
    criteria_id: str  # e.g., CC1.1, A1.2
    name: str
    description: str
    status: ControlStatus = ControlStatus.NOT_STARTED
    owner: str = ""
    implementation_notes: str = ""
    test_procedures: list[str] = field(default_factory=list)
    dora_mapping: list[str] = field(default_factory=list)  # DORA article references
    last_tested: datetime | None = None
    next_test_due: datetime | None = None

    @property
    def is_effective(self) -> bool:
        """Check if control is effective."""
        return self.status in (ControlStatus.TESTED, ControlStatus.EFFECTIVE)


@dataclass
class ControlEvidence:
    """Evidence for a SOC2 control."""

    evidence_id: str
    control_id: str
    evidence_type: EvidenceType
    title: str
    description: str
    file_path: str | None = None
    collected_at: datetime = field(default_factory=datetime.utcnow)
    collected_by: str = ""
    valid_from: datetime = field(default_factory=datetime.utcnow)
    valid_until: datetime | None = None
    is_current: bool = True

    def expire(self) -> None:
        """Mark evidence as expired."""
        self.is_current = False


@dataclass
class AuditFinding:
    """Audit finding from SOC2 examination."""

    finding_id: str
    audit_id: str
    control_id: str
    title: str
    description: str
    severity: str  # critical, high, medium, low
    identified_at: datetime
    identified_by: str
    status: str = "open"  # open, in_remediation, remediated, accepted
    remediation_plan: str | None = None
    remediation_due: datetime | None = None
    remediated_at: datetime | None = None
    exception_approved: bool = False
    exception_reason: str | None = None

    def start_remediation(self, plan: str, due_date: datetime) -> None:
        """Start remediation."""
        self.status = "in_remediation"
        self.remediation_plan = plan
        self.remediation_due = due_date

    def complete_remediation(self) -> None:
        """Complete remediation."""
        self.status = "remediated"
        self.remediated_at = datetime.utcnow()


@dataclass
class RemediationItem:
    """Remediation action item."""

    item_id: str
    finding_id: str
    title: str
    description: str
    assignee: str
    due_date: datetime
    status: str = "pending"  # pending, in_progress, completed
    completed_at: datetime | None = None


@dataclass
class SOC2Report:
    """SOC2 examination report."""

    report_id: str
    report_type: str  # Type I or Type II
    audit_period_start: datetime
    audit_period_end: datetime
    auditor_firm: str
    lead_auditor: str
    opinion: str  # unqualified, qualified, adverse, disclaimer
    issued_at: datetime | None = None
    trust_principles_covered: list[SOC2TrustPrinciple] = field(default_factory=list)
    total_controls: int = 0
    controls_tested: int = 0
    findings_count: int = 0
    exceptions_count: int = 0


@dataclass
class SOC2Config:
    """SOC2 certification service configuration."""

    organization_name: str
    system_description: str
    audit_firm: str = ""
    evidence_retention_years: int = 7
    control_test_frequency_days: int = 90
    auto_remind_before_days: int = 14


# =============================================================================
# SOC2 Control Library
# =============================================================================


def get_soc2_control_library() -> list[dict[str, Any]]:
    """Get standard SOC2 control library with DORA mappings."""
    return [
        # Security - Common Criteria (CC)
        {
            "criteria_id": "CC1.1",
            "trust_principle": SOC2TrustPrinciple.SECURITY,
            "name": "COSO Principle 1: Integrity and Ethics",
            "description": "The entity demonstrates a commitment to integrity and ethical values",
            "dora_mapping": ["Art. 5", "Art. 13(6)"],
        },
        {
            "criteria_id": "CC1.2",
            "trust_principle": SOC2TrustPrinciple.SECURITY,
            "name": "COSO Principle 2: Board Independence",
            "description": "The board of directors demonstrates independence from management",
            "dora_mapping": ["Art. 5(2)"],
        },
        {
            "criteria_id": "CC2.1",
            "trust_principle": SOC2TrustPrinciple.SECURITY,
            "name": "COSO Principle 13: Quality Information",
            "description": "The entity obtains or generates relevant, quality information",
            "dora_mapping": ["Art. 6", "Art. 8"],
        },
        {
            "criteria_id": "CC3.1",
            "trust_principle": SOC2TrustPrinciple.SECURITY,
            "name": "COSO Principle 6: Risk Assessment",
            "description": "The entity specifies objectives with sufficient clarity",
            "dora_mapping": ["Art. 6", "Art. 8"],
        },
        {
            "criteria_id": "CC5.1",
            "trust_principle": SOC2TrustPrinciple.SECURITY,
            "name": "COSO Principle 10: Control Activities",
            "description": "The entity selects and develops control activities",
            "dora_mapping": ["Art. 9", "Art. 10"],
        },
        {
            "criteria_id": "CC6.1",
            "trust_principle": SOC2TrustPrinciple.SECURITY,
            "name": "Logical and Physical Access Controls",
            "description": "Logical access security software, infrastructure, and architectures",
            "dora_mapping": ["Art. 9(3)", "Art. 9(4)"],
        },
        {
            "criteria_id": "CC6.6",
            "trust_principle": SOC2TrustPrinciple.SECURITY,
            "name": "Threat Detection",
            "description": "Security events are identified and evaluated",
            "dora_mapping": ["Art. 10"],
        },
        {
            "criteria_id": "CC6.7",
            "trust_principle": SOC2TrustPrinciple.SECURITY,
            "name": "Incident Response",
            "description": "Procedures exist to respond to security breaches",
            "dora_mapping": ["Art. 11", "Art. 17"],
        },
        {
            "criteria_id": "CC7.1",
            "trust_principle": SOC2TrustPrinciple.SECURITY,
            "name": "System Monitoring",
            "description": "System components are monitored to detect security events",
            "dora_mapping": ["Art. 10"],
        },
        {
            "criteria_id": "CC7.4",
            "trust_principle": SOC2TrustPrinciple.SECURITY,
            "name": "Incident Recovery",
            "description": "The entity recovers and resumes operations",
            "dora_mapping": ["Art. 11", "Art. 12"],
        },
        {
            "criteria_id": "CC8.1",
            "trust_principle": SOC2TrustPrinciple.SECURITY,
            "name": "Change Management",
            "description": "Changes to infrastructure are authorized and documented",
            "dora_mapping": ["Art. 7"],
        },
        {
            "criteria_id": "CC9.1",
            "trust_principle": SOC2TrustPrinciple.SECURITY,
            "name": "Risk Mitigation",
            "description": "The entity identifies, selects, and develops risk mitigation activities",
            "dora_mapping": ["Art. 6", "Art. 28"],
        },
        # Availability
        {
            "criteria_id": "A1.1",
            "trust_principle": SOC2TrustPrinciple.AVAILABILITY,
            "name": "System Availability Commitments",
            "description": "Current processing capacity and usage are maintained",
            "dora_mapping": ["Art. 7", "Art. 30(2)(e)"],
        },
        {
            "criteria_id": "A1.2",
            "trust_principle": SOC2TrustPrinciple.AVAILABILITY,
            "name": "Environmental Protections",
            "description": "Environmental protections for critical systems",
            "dora_mapping": ["Art. 9(2)"],
        },
        {
            "criteria_id": "A1.3",
            "trust_principle": SOC2TrustPrinciple.AVAILABILITY,
            "name": "Recovery and Continuity",
            "description": "Recovery procedures support system availability commitments",
            "dora_mapping": ["Art. 11", "Art. 12", "Art. 15"],
        },
        # Confidentiality
        {
            "criteria_id": "C1.1",
            "trust_principle": SOC2TrustPrinciple.CONFIDENTIALITY,
            "name": "Confidential Information Identification",
            "description": "Confidential information is identified and protected",
            "dora_mapping": ["Art. 9(4)", "Art. 30(2)(c)"],
        },
        {
            "criteria_id": "C1.2",
            "trust_principle": SOC2TrustPrinciple.CONFIDENTIALITY,
            "name": "Confidential Information Disposal",
            "description": "Confidential information is disposed of properly",
            "dora_mapping": ["Art. 30(2)(d)"],
        },
        # Processing Integrity
        {
            "criteria_id": "PI1.1",
            "trust_principle": SOC2TrustPrinciple.PROCESSING_INTEGRITY,
            "name": "Processing Completeness and Accuracy",
            "description": "System inputs are complete, accurate, and valid",
            "dora_mapping": ["Art. 7"],
        },
    ]


# =============================================================================
# Main Service Class
# =============================================================================


class SOC2CertificationService:
    """
    SOC2 Type II Certification Service.

    Manages SOC2 certification lifecycle per AICPA standards.
    """

    def __init__(self, config: SOC2Config) -> None:
        """Initialize SOC2 certification service."""
        self.config = config
        self._controls: dict[str, SOC2Control] = {}
        self._evidence: dict[str, ControlEvidence] = {}
        self._findings: dict[str, AuditFinding] = {}
        self._remediation_items: dict[str, RemediationItem] = {}
        self._reports: dict[str, SOC2Report] = {}
        self._initialize_controls()

    def _initialize_controls(self) -> None:
        """Initialize SOC2 controls from library."""
        for ctrl_def in get_soc2_control_library():
            control = SOC2Control(
                control_id=str(uuid4()),
                trust_principle=ctrl_def["trust_principle"],
                criteria_id=ctrl_def["criteria_id"],
                name=ctrl_def["name"],
                description=ctrl_def["description"],
                dora_mapping=ctrl_def.get("dora_mapping", []),
            )
            self._controls[control.control_id] = control

    # =========================================================================
    # Control Management
    # =========================================================================

    def get_control(self, control_id: str) -> SOC2Control | None:
        """Get control by ID."""
        return self._controls.get(control_id)

    def get_control_by_criteria(self, criteria_id: str) -> SOC2Control | None:
        """Get control by criteria ID (e.g., CC6.1)."""
        for control in self._controls.values():
            if control.criteria_id == criteria_id:
                return control
        return None

    def list_controls(
        self,
        trust_principle: SOC2TrustPrinciple | None = None,
        status: ControlStatus | None = None,
    ) -> list[SOC2Control]:
        """List controls with optional filters."""
        controls = list(self._controls.values())

        if trust_principle:
            controls = [c for c in controls if c.trust_principle == trust_principle]
        if status:
            controls = [c for c in controls if c.status == status]

        return controls

    def update_control_status(
        self,
        control_id: str,
        status: ControlStatus,
        notes: str = "",
    ) -> SOC2Control | None:
        """Update control implementation status."""
        control = self._controls.get(control_id)
        if not control:
            return None

        control.status = status
        if notes:
            control.implementation_notes = notes
        if status in (ControlStatus.TESTED, ControlStatus.EFFECTIVE):
            control.last_tested = datetime.utcnow()
            control.next_test_due = datetime.utcnow() + timedelta(
                days=self.config.control_test_frequency_days
            )

        return control

    def assign_control_owner(self, control_id: str, owner: str) -> bool:
        """Assign owner to a control."""
        control = self._controls.get(control_id)
        if not control:
            return False
        control.owner = owner
        return True

    # =========================================================================
    # Evidence Management
    # =========================================================================

    def add_evidence(
        self,
        control_id: str,
        evidence_type: EvidenceType,
        title: str,
        description: str,
        collected_by: str,
        file_path: str | None = None,
        valid_until: datetime | None = None,
    ) -> ControlEvidence:
        """Add evidence for a control."""
        if control_id not in self._controls:
            raise ValueError(f"Control not found: {control_id}")

        evidence = ControlEvidence(
            evidence_id=str(uuid4()),
            control_id=control_id,
            evidence_type=evidence_type,
            title=title,
            description=description,
            file_path=file_path,
            collected_by=collected_by,
            valid_until=valid_until,
        )
        self._evidence[evidence.evidence_id] = evidence
        return evidence

    def get_evidence(self, evidence_id: str) -> ControlEvidence | None:
        """Get evidence by ID."""
        return self._evidence.get(evidence_id)

    def list_evidence(
        self,
        control_id: str | None = None,
        current_only: bool = True,
    ) -> list[ControlEvidence]:
        """List evidence with optional filters."""
        evidence = list(self._evidence.values())

        if control_id:
            evidence = [e for e in evidence if e.control_id == control_id]
        if current_only:
            evidence = [e for e in evidence if e.is_current]

        return evidence

    def get_evidence_coverage(self) -> dict[str, Any]:
        """Get evidence coverage statistics."""
        total_controls = len(self._controls)
        controls_with_evidence = len(
            set(e.control_id for e in self._evidence.values() if e.is_current)
        )

        return {
            "total_controls": total_controls,
            "controls_with_evidence": controls_with_evidence,
            "coverage_percent": (
                (controls_with_evidence / total_controls * 100) if total_controls > 0 else 0
            ),
            "evidence_count": len([e for e in self._evidence.values() if e.is_current]),
        }

    # =========================================================================
    # Audit and Findings
    # =========================================================================

    def create_audit_report(
        self,
        report_type: str,
        audit_period_start: datetime,
        audit_period_end: datetime,
        auditor_firm: str,
        lead_auditor: str,
        trust_principles: list[SOC2TrustPrinciple] | None = None,
        opinion: str = "pending",
    ) -> SOC2Report:
        """Create a new SOC2 audit report."""
        report = SOC2Report(
            report_id=str(uuid4()),
            report_type=report_type,
            audit_period_start=audit_period_start,
            audit_period_end=audit_period_end,
            auditor_firm=auditor_firm,
            lead_auditor=lead_auditor,
            opinion=opinion,
            trust_principles_covered=trust_principles or list(SOC2TrustPrinciple),
        )
        self._reports[report.report_id] = report
        return report

    def get_report(self, report_id: str) -> SOC2Report | None:
        """Get audit report by ID."""
        return self._reports.get(report_id)

    def add_finding(
        self,
        audit_id: str,
        control_id: str,
        title: str,
        description: str,
        severity: str,
        identified_by: str,
    ) -> AuditFinding:
        """Add an audit finding."""
        finding = AuditFinding(
            finding_id=str(uuid4()),
            audit_id=audit_id,
            control_id=control_id,
            title=title,
            description=description,
            severity=severity,
            identified_at=datetime.utcnow(),
            identified_by=identified_by,
        )
        self._findings[finding.finding_id] = finding

        # Update report if exists
        report = self._reports.get(audit_id)
        if report:
            report.findings_count += 1

        return finding

    def get_finding(self, finding_id: str) -> AuditFinding | None:
        """Get finding by ID."""
        return self._findings.get(finding_id)

    def list_findings(
        self,
        audit_id: str | None = None,
        status: str | None = None,
    ) -> list[AuditFinding]:
        """List findings with optional filters."""
        findings = list(self._findings.values())

        if audit_id:
            findings = [f for f in findings if f.audit_id == audit_id]
        if status:
            findings = [f for f in findings if f.status == status]

        return findings

    # =========================================================================
    # Compliance Status
    # =========================================================================

    def get_compliance_status(self) -> dict[str, Any]:
        """Get overall SOC2 compliance status."""
        controls = list(self._controls.values())
        total = len(controls)

        by_status = {}
        for status in ControlStatus:
            count = sum(1 for c in controls if c.status == status)
            by_status[status.value] = count

        by_principle = {}
        for principle in SOC2TrustPrinciple:
            principle_controls = [c for c in controls if c.trust_principle == principle]
            effective = sum(1 for c in principle_controls if c.is_effective)
            by_principle[principle.value] = {
                "total": len(principle_controls),
                "effective": effective,
                "compliance_percent": (
                    (effective / len(principle_controls) * 100) if principle_controls else 0
                ),
            }

        effective_controls = sum(1 for c in controls if c.is_effective)
        open_findings = sum(1 for f in self._findings.values() if f.status == "open")

        return {
            "total_controls": total,
            "effective_controls": effective_controls,
            "overall_compliance_percent": (effective_controls / total * 100) if total > 0 else 0,
            "by_status": by_status,
            "by_principle": by_principle,
            "open_findings": open_findings,
            "evidence_coverage": self.get_evidence_coverage()["coverage_percent"],
        }

    def get_dora_mapping_report(self) -> dict[str, list[str]]:
        """Get DORA to SOC2 control mapping report."""
        mapping: dict[str, list[str]] = {}

        for control in self._controls.values():
            for dora_ref in control.dora_mapping:
                if dora_ref not in mapping:
                    mapping[dora_ref] = []
                mapping[dora_ref].append(f"{control.criteria_id}: {control.name}")

        return mapping


# =============================================================================
# Factory Functions
# =============================================================================


def create_soc2_certification(
    organization_name: str,
    system_description: str,
    audit_firm: str = "",
    **kwargs: Any,
) -> SOC2CertificationService:
    """Create SOC2 certification service instance."""
    config = SOC2Config(
        organization_name=organization_name,
        system_description=system_description,
        audit_firm=audit_firm,
        **kwargs,
    )
    return SOC2CertificationService(config)
