# -*- coding: utf-8 -*-
"""
TLPT Cooperation Service for Enterprise Clients.

DORA Phase 3 Block 3.5: TLPT cooperation procedures

Provides ICT provider support for client TLPT engagements per DORA Art. 26:
- Cooperation request handling
- Secure access provisioning
- Documentation preparation
- Finding coordination

DORA References:
    - Art. 26: Threat-led penetration testing (TLPT)
    - Art. 27: Requirements for testers
    - RTS on TLPT (CDR 2024/XXXX): Technical standards

Note: As ICT third-party provider, we support client TLPT testing
but do not conduct TLPT ourselves (that's client's obligation).
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


class TLPTCooperationType(Enum):
    """Types of TLPT cooperation requests."""

    INFORMATION_REQUEST = "information_request"  # Request for system info
    ACCESS_PROVISIONING = "access_provisioning"  # Request for test access
    ENVIRONMENT_SETUP = "environment_setup"  # Request to set up test env
    FINDING_REVIEW = "finding_review"  # Review findings together
    REMEDIATION_SUPPORT = "remediation_support"  # Support remediation
    ATTESTATION = "attestation"  # Provide attestation


class TLPTPhase(Enum):
    """TLPT engagement phases per TIBER-EU framework."""

    PREPARATION = "preparation"  # Scoping and planning
    TESTING = "testing"  # Active testing phase
    CLOSURE = "closure"  # Findings and remediation


class DocumentationType(Enum):
    """Types of documentation for TLPT support."""

    ARCHITECTURE_DIAGRAM = "architecture_diagram"
    DATA_FLOW_DIAGRAM = "data_flow_diagram"
    API_DOCUMENTATION = "api_documentation"
    SECURITY_CONTROLS = "security_controls"
    NETWORK_TOPOLOGY = "network_topology"
    ACCESS_MATRIX = "access_matrix"
    INCIDENT_RESPONSE = "incident_response"
    BUSINESS_CONTINUITY = "business_continuity"
    COMPLIANCE_EVIDENCE = "compliance_evidence"


class AccessLevel(Enum):
    """Access levels for TLPT testers."""

    READ_ONLY = "read_only"  # View only
    LIMITED = "limited"  # Limited interaction
    STANDARD = "standard"  # Normal user access
    ELEVATED = "elevated"  # Admin-like access
    FULL = "full"  # Complete access for testing


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class TLPTCooperationRequest:
    """Request for TLPT cooperation from client."""

    request_id: str
    client_id: str
    client_name: str
    cooperation_type: TLPTCooperationType
    tlpt_phase: TLPTPhase
    description: str
    requested_at: datetime
    requested_by: str
    due_date: datetime
    priority: str = "normal"  # low, normal, high, critical
    status: str = "pending"  # pending, in_progress, completed, rejected
    nca_reference: str | None = None  # NCA engagement reference
    tester_organization: str | None = None
    scope_systems: list[str] = field(default_factory=list)
    notes: str = ""
    completed_at: datetime | None = None
    completed_by: str | None = None

    def approve(self, approver: str) -> None:
        """Approve the request."""
        self.status = "in_progress"
        self.notes = f"Approved by {approver} at {datetime.utcnow().isoformat()}"

    def complete(self, user: str) -> None:
        """Mark request as completed."""
        self.status = "completed"
        self.completed_at = datetime.utcnow()
        self.completed_by = user

    def reject(self, reason: str) -> None:
        """Reject the request."""
        self.status = "rejected"
        self.notes = f"Rejected: {reason}"


@dataclass
class TLPTDocumentation:
    """Documentation prepared for TLPT support."""

    doc_id: str
    request_id: str
    doc_type: DocumentationType
    title: str
    description: str
    version: str
    created_at: datetime
    created_by: str
    classification: str = "CONFIDENTIAL"
    file_path: str | None = None
    content_hash: str | None = None
    expiry_date: datetime | None = None
    access_granted_to: list[str] = field(default_factory=list)

    def grant_access(self, entity: str) -> None:
        """Grant access to an entity."""
        if entity not in self.access_granted_to:
            self.access_granted_to.append(entity)

    def revoke_access(self, entity: str) -> None:
        """Revoke access from an entity."""
        if entity in self.access_granted_to:
            self.access_granted_to.remove(entity)


@dataclass
class TLPTAccessGrant:
    """Access grant for TLPT testers."""

    grant_id: str
    request_id: str
    client_id: str
    tester_id: str
    tester_organization: str
    access_level: AccessLevel
    systems: list[str]
    environments: list[str]  # production, staging, test
    granted_at: datetime
    granted_by: str
    valid_from: datetime
    valid_until: datetime
    ip_restrictions: list[str] = field(default_factory=list)
    time_restrictions: str | None = None  # e.g., "09:00-18:00 CET"
    revoked: bool = False
    revoked_at: datetime | None = None
    revoked_by: str | None = None
    revocation_reason: str | None = None

    @property
    def is_active(self) -> bool:
        """Check if access grant is currently active."""
        now = datetime.utcnow()
        return not self.revoked and self.valid_from <= now <= self.valid_until

    def revoke(self, user: str, reason: str) -> None:
        """Revoke the access grant."""
        self.revoked = True
        self.revoked_at = datetime.utcnow()
        self.revoked_by = user
        self.revocation_reason = reason


@dataclass
class TLPTFinding:
    """TLPT finding reported by testers."""

    finding_id: str
    request_id: str
    client_id: str
    title: str
    description: str
    severity: str  # critical, high, medium, low, informational
    affected_systems: list[str]
    attack_technique: str  # MITRE ATT&CK reference
    evidence: str
    reported_at: datetime
    reported_by: str
    status: str = "open"  # open, acknowledged, in_remediation, resolved, accepted_risk
    remediation_plan: str | None = None
    remediation_due: datetime | None = None
    resolved_at: datetime | None = None
    our_responsibility: bool = False  # True if we need to fix
    provider_response: str | None = None

    def acknowledge(self, is_our_responsibility: bool, response: str) -> None:
        """Acknowledge the finding."""
        self.status = "acknowledged"
        self.our_responsibility = is_our_responsibility
        self.provider_response = response

    def start_remediation(self, plan: str, due_date: datetime) -> None:
        """Start remediation."""
        self.status = "in_remediation"
        self.remediation_plan = plan
        self.remediation_due = due_date

    def resolve(self) -> None:
        """Mark finding as resolved."""
        self.status = "resolved"
        self.resolved_at = datetime.utcnow()


@dataclass
class TLPTCooperationReport:
    """Summary report of TLPT cooperation engagement."""

    report_id: str
    client_id: str
    client_name: str
    engagement_reference: str
    tlpt_start: datetime
    tlpt_end: datetime
    requests_handled: int
    documentation_provided: int
    access_grants_issued: int
    findings_reported: int
    findings_our_responsibility: int
    remediation_completed: int
    remediation_pending: int
    cooperation_rating: str  # excellent, good, satisfactory, needs_improvement
    lessons_learned: list[str]
    improvements_identified: list[str]
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"


@dataclass
class TLPTConfig:
    """TLPT cooperation service configuration."""

    max_access_duration_days: int = 30
    default_access_level: AccessLevel = AccessLevel.LIMITED
    require_nca_reference: bool = True
    auto_revoke_on_expiry: bool = True
    documentation_retention_days: int = 2555  # 7 years
    require_dual_approval: bool = True
    allowed_environments: list[str] = field(default_factory=lambda: ["staging", "test"])


# =============================================================================
# Main Service Class
# =============================================================================


class TLPTCooperationService:
    """
    TLPT Cooperation Service.

    Supports client TLPT engagements per DORA Art. 26-27.
    """

    def __init__(self, config: TLPTConfig | None = None) -> None:
        """Initialize TLPT cooperation service."""
        self.config = config or TLPTConfig()
        self._requests: dict[str, TLPTCooperationRequest] = {}
        self._documentation: dict[str, TLPTDocumentation] = {}
        self._access_grants: dict[str, TLPTAccessGrant] = {}
        self._findings: dict[str, TLPTFinding] = {}
        self._reports: dict[str, TLPTCooperationReport] = {}

    # =========================================================================
    # Request Management
    # =========================================================================

    def create_request(
        self,
        client_id: str,
        client_name: str,
        cooperation_type: TLPTCooperationType,
        tlpt_phase: TLPTPhase,
        description: str,
        requested_by: str,
        due_date: datetime,
        nca_reference: str | None = None,
        tester_organization: str | None = None,
        scope_systems: list[str] | None = None,
        priority: str = "normal",
    ) -> TLPTCooperationRequest:
        """Create a new TLPT cooperation request."""
        if self.config.require_nca_reference and not nca_reference:
            raise ValueError("NCA reference is required for TLPT cooperation")

        request = TLPTCooperationRequest(
            request_id=str(uuid4()),
            client_id=client_id,
            client_name=client_name,
            cooperation_type=cooperation_type,
            tlpt_phase=tlpt_phase,
            description=description,
            requested_at=datetime.utcnow(),
            requested_by=requested_by,
            due_date=due_date,
            priority=priority,
            nca_reference=nca_reference,
            tester_organization=tester_organization,
            scope_systems=scope_systems or [],
        )
        self._requests[request.request_id] = request
        return request

    def get_request(self, request_id: str) -> TLPTCooperationRequest | None:
        """Get request by ID."""
        return self._requests.get(request_id)

    def list_requests(
        self,
        client_id: str | None = None,
        status: str | None = None,
        phase: TLPTPhase | None = None,
    ) -> list[TLPTCooperationRequest]:
        """List requests with optional filters."""
        requests = list(self._requests.values())

        if client_id:
            requests = [r for r in requests if r.client_id == client_id]
        if status:
            requests = [r for r in requests if r.status == status]
        if phase:
            requests = [r for r in requests if r.tlpt_phase == phase]

        return requests

    def approve_request(self, request_id: str, approver: str) -> bool:
        """Approve a cooperation request."""
        request = self._requests.get(request_id)
        if not request or request.status != "pending":
            return False
        request.approve(approver)
        return True

    def complete_request(self, request_id: str, user: str) -> bool:
        """Mark a request as completed."""
        request = self._requests.get(request_id)
        if not request or request.status != "in_progress":
            return False
        request.complete(user)
        return True

    # =========================================================================
    # Documentation Management
    # =========================================================================

    def provide_documentation(
        self,
        request_id: str,
        doc_type: DocumentationType,
        title: str,
        description: str,
        created_by: str,
        file_path: str | None = None,
        version: str = "1.0",
    ) -> TLPTDocumentation:
        """Provide documentation for TLPT support."""
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        doc = TLPTDocumentation(
            doc_id=str(uuid4()),
            request_id=request_id,
            doc_type=doc_type,
            title=title,
            description=description,
            version=version,
            created_at=datetime.utcnow(),
            created_by=created_by,
            file_path=file_path,
            expiry_date=datetime.utcnow()
            + timedelta(days=self.config.documentation_retention_days),
        )
        self._documentation[doc.doc_id] = doc
        return doc

    def get_documentation(self, doc_id: str) -> TLPTDocumentation | None:
        """Get documentation by ID."""
        return self._documentation.get(doc_id)

    def list_documentation(self, request_id: str | None = None) -> list[TLPTDocumentation]:
        """List documentation with optional request filter."""
        docs = list(self._documentation.values())
        if request_id:
            docs = [d for d in docs if d.request_id == request_id]
        return docs

    # =========================================================================
    # Access Management
    # =========================================================================

    def grant_access(
        self,
        request_id: str,
        tester_id: str,
        tester_organization: str,
        access_level: AccessLevel,
        systems: list[str],
        environments: list[str],
        granted_by: str,
        duration_days: int | None = None,
        ip_restrictions: list[str] | None = None,
    ) -> TLPTAccessGrant:
        """Grant access for TLPT testers."""
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        # Validate environments
        for env in environments:
            if env not in self.config.allowed_environments:
                raise ValueError(f"Environment not allowed: {env}")

        # Validate duration
        duration = duration_days or self.config.max_access_duration_days
        if duration > self.config.max_access_duration_days:
            duration = self.config.max_access_duration_days

        valid_from = datetime.utcnow()
        valid_until = valid_from + timedelta(days=duration)

        grant = TLPTAccessGrant(
            grant_id=str(uuid4()),
            request_id=request_id,
            client_id=request.client_id,
            tester_id=tester_id,
            tester_organization=tester_organization,
            access_level=access_level,
            systems=systems,
            environments=environments,
            granted_at=datetime.utcnow(),
            granted_by=granted_by,
            valid_from=valid_from,
            valid_until=valid_until,
            ip_restrictions=ip_restrictions or [],
        )
        self._access_grants[grant.grant_id] = grant
        return grant

    def get_access_grant(self, grant_id: str) -> TLPTAccessGrant | None:
        """Get access grant by ID."""
        return self._access_grants.get(grant_id)

    def list_access_grants(
        self,
        client_id: str | None = None,
        active_only: bool = False,
    ) -> list[TLPTAccessGrant]:
        """List access grants with optional filters."""
        grants = list(self._access_grants.values())

        if client_id:
            grants = [g for g in grants if g.client_id == client_id]
        if active_only:
            grants = [g for g in grants if g.is_active]

        return grants

    def revoke_access(self, grant_id: str, user: str, reason: str) -> bool:
        """Revoke an access grant."""
        grant = self._access_grants.get(grant_id)
        if not grant or grant.revoked:
            return False
        grant.revoke(user, reason)
        return True

    def revoke_expired_access(self) -> int:
        """Revoke all expired access grants."""
        if not self.config.auto_revoke_on_expiry:
            return 0

        count = 0
        now = datetime.utcnow()
        for grant in self._access_grants.values():
            if not grant.revoked and grant.valid_until < now:
                grant.revoke("system", "Access period expired")
                count += 1
        return count

    # =========================================================================
    # Finding Management
    # =========================================================================

    def record_finding(
        self,
        request_id: str,
        title: str,
        description: str,
        severity: str,
        affected_systems: list[str],
        attack_technique: str,
        evidence: str,
        reported_by: str,
    ) -> TLPTFinding:
        """Record a TLPT finding."""
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        finding = TLPTFinding(
            finding_id=str(uuid4()),
            request_id=request_id,
            client_id=request.client_id,
            title=title,
            description=description,
            severity=severity,
            affected_systems=affected_systems,
            attack_technique=attack_technique,
            evidence=evidence,
            reported_at=datetime.utcnow(),
            reported_by=reported_by,
        )
        self._findings[finding.finding_id] = finding
        return finding

    def get_finding(self, finding_id: str) -> TLPTFinding | None:
        """Get finding by ID."""
        return self._findings.get(finding_id)

    def list_findings(
        self,
        request_id: str | None = None,
        client_id: str | None = None,
        our_responsibility: bool | None = None,
    ) -> list[TLPTFinding]:
        """List findings with optional filters."""
        findings = list(self._findings.values())

        if request_id:
            findings = [f for f in findings if f.request_id == request_id]
        if client_id:
            findings = [f for f in findings if f.client_id == client_id]
        if our_responsibility is not None:
            findings = [f for f in findings if f.our_responsibility == our_responsibility]

        return findings

    def acknowledge_finding(
        self,
        finding_id: str,
        is_our_responsibility: bool,
        response: str,
    ) -> bool:
        """Acknowledge a finding."""
        finding = self._findings.get(finding_id)
        if not finding or finding.status != "open":
            return False
        finding.acknowledge(is_our_responsibility, response)
        return True

    # =========================================================================
    # Reporting
    # =========================================================================

    def generate_cooperation_report(
        self,
        client_id: str,
        client_name: str,
        engagement_reference: str,
        tlpt_start: datetime,
        tlpt_end: datetime,
        created_by: str,
    ) -> TLPTCooperationReport:
        """Generate a TLPT cooperation summary report."""
        requests = self.list_requests(client_id=client_id)
        findings = self.list_findings(client_id=client_id)
        access_grants = self.list_access_grants(client_id=client_id)

        our_findings = [f for f in findings if f.our_responsibility]
        remediated = [f for f in our_findings if f.status == "resolved"]

        report = TLPTCooperationReport(
            report_id=str(uuid4()),
            client_id=client_id,
            client_name=client_name,
            engagement_reference=engagement_reference,
            tlpt_start=tlpt_start,
            tlpt_end=tlpt_end,
            requests_handled=sum(1 for r in requests if r.status == "completed"),
            documentation_provided=len(self.list_documentation()),
            access_grants_issued=len(access_grants),
            findings_reported=len(findings),
            findings_our_responsibility=len(our_findings),
            remediation_completed=len(remediated),
            remediation_pending=len(our_findings) - len(remediated),
            cooperation_rating="good",  # Would be assessed
            lessons_learned=[],
            improvements_identified=[],
            created_by=created_by,
        )
        self._reports[report.report_id] = report
        return report

    def get_report(self, report_id: str) -> TLPTCooperationReport | None:
        """Get report by ID."""
        return self._reports.get(report_id)


# =============================================================================
# Factory Functions
# =============================================================================


def create_tlpt_cooperation(
    require_nca_reference: bool = True,
    allowed_environments: list[str] | None = None,
    **kwargs: Any,
) -> TLPTCooperationService:
    """Create TLPT cooperation service instance."""
    config = TLPTConfig(
        require_nca_reference=require_nca_reference,
        allowed_environments=allowed_environments or ["staging", "test"],
        **kwargs,
    )
    return TLPTCooperationService(config)
