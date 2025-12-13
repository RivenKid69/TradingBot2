# -*- coding: utf-8 -*-
"""
MiFID II Conformance Certification - RTS 6 Article 5 Compliance.

This module manages conformance certificates for algorithmic trading systems
per MiFID II RTS 6 requirements:

"Investment firms shall not deploy an algorithm until the algorithm
has undergone certification or validation by the compliance function
or another appropriate function."

Key Responsibilities:
    - Issue conformance certificates after successful testing
    - Track certificate validity and expiry
    - Manage certificate revocation
    - Generate deployment approval documentation
    - Support NCA audit requirements

References:
    - RTS 6 (Regulation 2017/589) Article 5: Testing conformance
    - RTS 6 Article 7: Deployment of algorithms
    - ESMA Guidelines on algorithmic trading (ESMA/2015/1464)
    - FCA Market Watch 67: Algorithmic trading compliance
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import (
    Optional,
    List,
    Dict,
    Any,
    Tuple,
    Callable,
    Sequence,
)

from services.algo_integration.conformance_testing import (
    ConformanceTestSuite,
    ConformanceSuiteStatus,
    TestCategory,
    TestResult,
    TestEnvironment,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class CertificateStatus(Enum):
    """Status of a conformance certificate."""

    DRAFT = "draft"
    """Certificate being prepared."""

    PENDING_APPROVAL = "pending_approval"
    """Awaiting compliance approval."""

    APPROVED = "approved"
    """Certificate approved and valid."""

    EXPIRED = "expired"
    """Certificate has expired."""

    REVOKED = "revoked"
    """Certificate revoked (issue discovered)."""

    SUPERSEDED = "superseded"
    """Superseded by newer certificate."""


class CertificateType(Enum):
    """Type of conformance certificate."""

    INITIAL = "initial"
    """Initial deployment certification."""

    UPDATE = "update"
    """Certification for algorithm update."""

    RECERTIFICATION = "recertification"
    """Periodic recertification."""

    EMERGENCY = "emergency"
    """Emergency certification (expedited)."""


class DeploymentApproval(Enum):
    """Deployment approval status."""

    NOT_REQUESTED = "not_requested"
    """Deployment approval not requested."""

    PENDING = "pending"
    """Deployment approval pending."""

    APPROVED = "approved"
    """Deployment approved."""

    REJECTED = "rejected"
    """Deployment rejected."""

    CONDITIONAL = "conditional"
    """Deployment approved with conditions."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CertificateCondition:
    """
    Condition attached to a certificate.

    Conditions may be required before or during deployment.
    """

    condition_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    requirement: str = ""
    deadline: Optional[date] = None
    satisfied: bool = False
    satisfied_date: Optional[date] = None
    satisfied_by: str = ""
    notes: str = ""

    def is_overdue(self) -> bool:
        """Check if condition is overdue."""
        if self.satisfied:
            return False
        if self.deadline is None:
            return False
        return date.today() > self.deadline

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condition_id": self.condition_id,
            "description": self.description,
            "requirement": self.requirement,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "satisfied": self.satisfied,
            "satisfied_date": self.satisfied_date.isoformat() if self.satisfied_date else None,
            "satisfied_by": self.satisfied_by,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CertificateCondition":
        """Create from dictionary."""
        return cls(
            condition_id=data.get("condition_id", str(uuid.uuid4())),
            description=data.get("description", ""),
            requirement=data.get("requirement", ""),
            deadline=date.fromisoformat(data["deadline"]) if data.get("deadline") else None,
            satisfied=data.get("satisfied", False),
            satisfied_date=date.fromisoformat(data["satisfied_date"]) if data.get("satisfied_date") else None,
            satisfied_by=data.get("satisfied_by", ""),
            notes=data.get("notes", ""),
        )


@dataclass
class ConformanceCertificate:
    """
    RTS 6 Article 5 Conformance Certificate.

    Official certification that an algorithm has passed conformance
    testing and is approved for deployment.

    Per RTS 6: "Investment firms shall not deploy an algorithm until
    the algorithm has undergone certification or validation."

    Attributes:
        certificate_id: Unique identifier
        certificate_number: Human-readable certificate number
        certificate_type: Type of certification
        algorithm_id: Algorithm identifier
        algorithm_version: Algorithm version
        algorithm_name: Algorithm name
        firm_lei: Firm's LEI
        firm_name: Firm's legal name
        test_suite_id: ID of conformance test suite
        test_environment: Environment where tests were run
        test_results_summary: Summary of test results
        status: Certificate status
        conditions: Attached conditions
        issued_date: Date certificate issued
        valid_from: Start of validity period
        valid_until: End of validity period
        issued_by: Person who issued certificate
        approved_by: Person who approved certificate
        revoked_by: Person who revoked (if applicable)
        revocation_reason: Reason for revocation
    """

    certificate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    certificate_number: str = ""
    certificate_type: CertificateType = CertificateType.INITIAL

    # Algorithm identification
    algorithm_id: str = ""
    algorithm_version: str = ""
    algorithm_name: str = ""

    # Firm identification
    firm_lei: str = ""
    firm_name: str = ""

    # Test results
    test_suite_id: str = ""
    test_environment: TestEnvironment = TestEnvironment.UAT
    test_results_summary: Dict[str, Any] = field(default_factory=dict)
    test_pass_rate: float = 0.0
    critical_tests_passed: int = 0
    critical_tests_total: int = 0

    # Status
    status: CertificateStatus = CertificateStatus.DRAFT

    # Conditions
    conditions: List[CertificateCondition] = field(default_factory=list)

    # Validity
    issued_date: Optional[date] = None
    valid_from: Optional[date] = None
    valid_until: Optional[date] = None
    validity_days: int = 365  # Default 1 year validity

    # Approvals
    issued_by: str = ""
    approved_by: str = ""
    approval_date: Optional[date] = None
    approval_notes: str = ""

    # Revocation
    revoked_by: str = ""
    revocation_date: Optional[date] = None
    revocation_reason: str = ""

    # Metadata
    created_timestamp_ns: int = field(default_factory=time.time_ns)
    modified_timestamp_ns: int = field(default_factory=time.time_ns)

    def __post_init__(self) -> None:
        """Initialize certificate number if not set."""
        if not self.certificate_number:
            # Generate certificate number: CERT-YYYYMMDD-XXXX
            today = date.today().strftime("%Y%m%d")
            short_id = self.certificate_id[:4].upper()
            self.certificate_number = f"CERT-{today}-{short_id}"

    # =========================================================================
    # Status Management
    # =========================================================================

    def is_valid(self) -> bool:
        """Check if certificate is currently valid."""
        if self.status != CertificateStatus.APPROVED:
            return False

        if self.valid_until and date.today() > self.valid_until:
            return False

        if self.valid_from and date.today() < self.valid_from:
            return False

        return True

    def is_expired(self) -> bool:
        """Check if certificate has expired."""
        if self.valid_until and date.today() > self.valid_until:
            return True
        return False

    def days_until_expiry(self) -> Optional[int]:
        """Get days until expiry."""
        if self.valid_until is None:
            return None
        delta = self.valid_until - date.today()
        return delta.days

    def submit_for_approval(self, issued_by: str) -> None:
        """Submit certificate for approval."""
        self.status = CertificateStatus.PENDING_APPROVAL
        self.issued_by = issued_by
        self.issued_date = date.today()
        self.modified_timestamp_ns = time.time_ns()
        logger.info(f"Certificate {self.certificate_number} submitted for approval")

    def approve(self, approved_by: str, notes: str = "") -> None:
        """Approve the certificate."""
        self.status = CertificateStatus.APPROVED
        self.approved_by = approved_by
        self.approval_date = date.today()
        self.approval_notes = notes

        # Set validity period
        if self.valid_from is None:
            self.valid_from = date.today()
        if self.valid_until is None:
            self.valid_until = self.valid_from + timedelta(days=self.validity_days)

        self.modified_timestamp_ns = time.time_ns()
        logger.info(
            f"Certificate {self.certificate_number} approved by {approved_by}, "
            f"valid until {self.valid_until}"
        )

    def reject(self, rejected_by: str, reason: str) -> None:
        """Reject the certificate."""
        self.status = CertificateStatus.DRAFT
        self.approval_notes = f"Rejected by {rejected_by}: {reason}"
        self.modified_timestamp_ns = time.time_ns()
        logger.warning(f"Certificate {self.certificate_number} rejected: {reason}")

    def revoke(self, revoked_by: str, reason: str) -> None:
        """Revoke the certificate."""
        self.status = CertificateStatus.REVOKED
        self.revoked_by = revoked_by
        self.revocation_date = date.today()
        self.revocation_reason = reason
        self.modified_timestamp_ns = time.time_ns()
        logger.warning(f"Certificate {self.certificate_number} REVOKED: {reason}")

    def supersede(self, new_certificate_id: str) -> None:
        """Mark certificate as superseded."""
        self.status = CertificateStatus.SUPERSEDED
        self.approval_notes = f"Superseded by certificate {new_certificate_id}"
        self.modified_timestamp_ns = time.time_ns()
        logger.info(f"Certificate {self.certificate_number} superseded")

    def expire(self) -> None:
        """Mark certificate as expired."""
        self.status = CertificateStatus.EXPIRED
        self.modified_timestamp_ns = time.time_ns()
        logger.info(f"Certificate {self.certificate_number} expired")

    # =========================================================================
    # Conditions
    # =========================================================================

    def add_condition(self, condition: CertificateCondition) -> None:
        """Add a condition to the certificate."""
        self.conditions.append(condition)
        self.modified_timestamp_ns = time.time_ns()

    def get_unsatisfied_conditions(self) -> List[CertificateCondition]:
        """Get all unsatisfied conditions."""
        return [c for c in self.conditions if not c.satisfied]

    def get_overdue_conditions(self) -> List[CertificateCondition]:
        """Get all overdue conditions."""
        return [c for c in self.conditions if c.is_overdue()]

    def satisfy_condition(self, condition_id: str, satisfied_by: str, notes: str = "") -> bool:
        """Mark a condition as satisfied."""
        for condition in self.conditions:
            if condition.condition_id == condition_id:
                condition.satisfied = True
                condition.satisfied_date = date.today()
                condition.satisfied_by = satisfied_by
                condition.notes = notes
                self.modified_timestamp_ns = time.time_ns()
                logger.info(f"Condition {condition_id} satisfied")
                return True
        return False

    def all_conditions_satisfied(self) -> bool:
        """Check if all conditions are satisfied."""
        return all(c.satisfied for c in self.conditions)

    # =========================================================================
    # Hash and Verification
    # =========================================================================

    def compute_hash(self) -> str:
        """Compute certificate hash for integrity verification."""
        hash_data = {
            "certificate_id": self.certificate_id,
            "certificate_number": self.certificate_number,
            "algorithm_id": self.algorithm_id,
            "algorithm_version": self.algorithm_version,
            "firm_lei": self.firm_lei,
            "test_suite_id": self.test_suite_id,
            "test_pass_rate": self.test_pass_rate,
            "status": self.status.value,
            "issued_date": self.issued_date.isoformat() if self.issued_date else "",
            "valid_from": self.valid_from.isoformat() if self.valid_from else "",
            "valid_until": self.valid_until.isoformat() if self.valid_until else "",
            "approved_by": self.approved_by,
        }

        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()

    # =========================================================================
    # Reporting
    # =========================================================================

    def generate_certificate_document(self) -> str:
        """
        Generate formal certificate document.

        Returns text representation of the conformance certificate.
        """
        doc = f"""
================================================================================
                    CONFORMANCE CERTIFICATE
                    RTS 6 Article 5 Compliance
================================================================================

CERTIFICATE NUMBER:     {self.certificate_number}
CERTIFICATE TYPE:       {self.certificate_type.value.upper()}
STATUS:                 {self.status.value.upper()}

--------------------------------------------------------------------------------
                         FIRM IDENTIFICATION
--------------------------------------------------------------------------------
Firm Name:              {self.firm_name}
Legal Entity Identifier: {self.firm_lei}

--------------------------------------------------------------------------------
                       ALGORITHM IDENTIFICATION
--------------------------------------------------------------------------------
Algorithm ID:           {self.algorithm_id}
Algorithm Version:      {self.algorithm_version}
Algorithm Name:         {self.algorithm_name}

--------------------------------------------------------------------------------
                         TEST RESULTS SUMMARY
--------------------------------------------------------------------------------
Test Suite ID:          {self.test_suite_id}
Test Environment:       {self.test_environment.value}
Pass Rate:              {self.test_pass_rate:.1f}%
Critical Tests:         {self.critical_tests_passed}/{self.critical_tests_total} passed

--------------------------------------------------------------------------------
                              VALIDITY
--------------------------------------------------------------------------------
Issued Date:            {self.issued_date or 'N/A'}
Valid From:             {self.valid_from or 'N/A'}
Valid Until:            {self.valid_until or 'N/A'}
"""

        if self.conditions:
            doc += """
--------------------------------------------------------------------------------
                             CONDITIONS
--------------------------------------------------------------------------------
"""
            for i, cond in enumerate(self.conditions, 1):
                status = "SATISFIED" if cond.satisfied else "PENDING"
                doc += f"""
{i}. [{status}] {cond.description}
   Requirement: {cond.requirement}
   Deadline: {cond.deadline or 'N/A'}
"""

        doc += f"""
--------------------------------------------------------------------------------
                             APPROVALS
--------------------------------------------------------------------------------
Issued By:              {self.issued_by}
Approved By:            {self.approved_by or 'N/A'}
Approval Date:          {self.approval_date or 'N/A'}
Approval Notes:         {self.approval_notes or 'N/A'}

"""

        if self.status == CertificateStatus.REVOKED:
            doc += f"""
--------------------------------------------------------------------------------
                             REVOCATION
--------------------------------------------------------------------------------
Revoked By:             {self.revoked_by}
Revocation Date:        {self.revocation_date}
Reason:                 {self.revocation_reason}

"""

        doc += f"""
--------------------------------------------------------------------------------
                          CERTIFICATE HASH
--------------------------------------------------------------------------------
Hash:                   {self.compute_hash()}

================================================================================
This certificate confirms that the above-identified algorithm has successfully
completed conformance testing per MiFID II RTS 6 Article 5 requirements and
is approved for deployment in the specified environment.

This certificate is valid only for the version specified above. Any material
changes to the algorithm require recertification.
================================================================================
"""
        return doc

    def generate_deployment_approval(self) -> str:
        """Generate deployment approval document."""
        if self.status != CertificateStatus.APPROVED:
            return "ERROR: Certificate not approved. Deployment not authorized."

        if not self.is_valid():
            return "ERROR: Certificate not valid. Deployment not authorized."

        unsatisfied = self.get_unsatisfied_conditions()
        if unsatisfied:
            return f"ERROR: {len(unsatisfied)} condition(s) not satisfied. Deployment not authorized."

        return f"""
================================================================================
                    DEPLOYMENT APPROVAL
                    RTS 6 Article 7 Compliance
================================================================================

DEPLOYMENT AUTHORIZED

Certificate Number:     {self.certificate_number}
Algorithm:             {self.algorithm_id} v{self.algorithm_version}
Firm:                  {self.firm_name} ({self.firm_lei})

This deployment approval confirms that:

1. The algorithm has successfully completed conformance testing
2. All critical tests passed ({self.critical_tests_passed}/{self.critical_tests_total})
3. The certificate is valid and approved
4. All conditions (if any) have been satisfied

AUTHORIZATION DETAILS
---------------------
Authorized By:         {self.approved_by}
Authorization Date:    {date.today()}
Valid Until:           {self.valid_until}

DEPLOYMENT REQUIREMENTS
-----------------------
- Deploy only the certified version ({self.algorithm_version})
- Monitor algorithm behavior post-deployment
- Report any anomalies to compliance immediately
- Re-certify before any material changes

================================================================================
Certificate Hash: {self.compute_hash()}
================================================================================
"""

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "certificate_id": self.certificate_id,
            "certificate_number": self.certificate_number,
            "certificate_type": self.certificate_type.value,
            "algorithm_id": self.algorithm_id,
            "algorithm_version": self.algorithm_version,
            "algorithm_name": self.algorithm_name,
            "firm_lei": self.firm_lei,
            "firm_name": self.firm_name,
            "test_suite_id": self.test_suite_id,
            "test_environment": self.test_environment.value,
            "test_results_summary": self.test_results_summary,
            "test_pass_rate": self.test_pass_rate,
            "critical_tests_passed": self.critical_tests_passed,
            "critical_tests_total": self.critical_tests_total,
            "status": self.status.value,
            "conditions": [c.to_dict() for c in self.conditions],
            "issued_date": self.issued_date.isoformat() if self.issued_date else None,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "validity_days": self.validity_days,
            "issued_by": self.issued_by,
            "approved_by": self.approved_by,
            "approval_date": self.approval_date.isoformat() if self.approval_date else None,
            "approval_notes": self.approval_notes,
            "revoked_by": self.revoked_by,
            "revocation_date": self.revocation_date.isoformat() if self.revocation_date else None,
            "revocation_reason": self.revocation_reason,
            "created_timestamp_ns": self.created_timestamp_ns,
            "modified_timestamp_ns": self.modified_timestamp_ns,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConformanceCertificate":
        """Create from dictionary."""
        cert = cls(
            certificate_id=data.get("certificate_id", str(uuid.uuid4())),
            certificate_number=data.get("certificate_number", ""),
            certificate_type=CertificateType(data.get("certificate_type", "initial")),
            algorithm_id=data.get("algorithm_id", ""),
            algorithm_version=data.get("algorithm_version", ""),
            algorithm_name=data.get("algorithm_name", ""),
            firm_lei=data.get("firm_lei", ""),
            firm_name=data.get("firm_name", ""),
            test_suite_id=data.get("test_suite_id", ""),
            test_environment=TestEnvironment(data.get("test_environment", "uat")),
            test_results_summary=data.get("test_results_summary", {}),
            test_pass_rate=data.get("test_pass_rate", 0.0),
            critical_tests_passed=data.get("critical_tests_passed", 0),
            critical_tests_total=data.get("critical_tests_total", 0),
            status=CertificateStatus(data.get("status", "draft")),
            validity_days=data.get("validity_days", 365),
            issued_by=data.get("issued_by", ""),
            approved_by=data.get("approved_by", ""),
            approval_notes=data.get("approval_notes", ""),
            revoked_by=data.get("revoked_by", ""),
            revocation_reason=data.get("revocation_reason", ""),
            created_timestamp_ns=data.get("created_timestamp_ns", time.time_ns()),
            modified_timestamp_ns=data.get("modified_timestamp_ns", time.time_ns()),
        )

        # Parse dates
        if data.get("issued_date"):
            cert.issued_date = date.fromisoformat(data["issued_date"])
        if data.get("valid_from"):
            cert.valid_from = date.fromisoformat(data["valid_from"])
        if data.get("valid_until"):
            cert.valid_until = date.fromisoformat(data["valid_until"])
        if data.get("approval_date"):
            cert.approval_date = date.fromisoformat(data["approval_date"])
        if data.get("revocation_date"):
            cert.revocation_date = date.fromisoformat(data["revocation_date"])

        # Parse conditions
        cert.conditions = [
            CertificateCondition.from_dict(c)
            for c in data.get("conditions", [])
        ]

        return cert

    @classmethod
    def from_json(cls, json_str: str) -> "ConformanceCertificate":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# Certificate Manager
# =============================================================================


class CertificateManager:
    """
    Manager for conformance certificates.

    Handles certificate lifecycle including:
    - Creating certificates from test suites
    - Managing certificate storage
    - Tracking validity and expiry
    - Supporting NCA audits
    """

    def __init__(
        self,
        storage_path: str = "state/compliance/certificates",
        default_validity_days: int = 365,
        expiry_warning_days: int = 30,
    ):
        """
        Initialize certificate manager.

        Args:
            storage_path: Path for certificate storage.
            default_validity_days: Default certificate validity in days.
            expiry_warning_days: Days before expiry to warn.
        """
        self.storage_path = Path(storage_path)
        self.default_validity_days = default_validity_days
        self.expiry_warning_days = expiry_warning_days
        self._certificates: Dict[str, ConformanceCertificate] = {}
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        """Ensure storage directory exists."""
        self.storage_path.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Certificate Creation
    # =========================================================================

    def create_certificate_from_suite(
        self,
        suite: ConformanceTestSuite,
        firm_lei: str,
        firm_name: str,
        algorithm_name: str = "",
        certificate_type: CertificateType = CertificateType.INITIAL,
        issued_by: str = "",
    ) -> Optional[ConformanceCertificate]:
        """
        Create a certificate from a completed test suite.

        Args:
            suite: Completed conformance test suite.
            firm_lei: Firm's LEI.
            firm_name: Firm's legal name.
            algorithm_name: Algorithm name.
            certificate_type: Type of certification.
            issued_by: Person issuing certificate.

        Returns:
            ConformanceCertificate if suite passed, None otherwise.
        """
        if not suite.can_certify():
            logger.warning(
                f"Cannot create certificate: suite {suite.suite_id} did not pass "
                f"(status: {suite.status.value})"
            )
            return None

        metrics = suite.get_metrics()

        cert = ConformanceCertificate(
            certificate_type=certificate_type,
            algorithm_id=suite.algorithm_id,
            algorithm_version=suite.algorithm_version,
            algorithm_name=algorithm_name or suite.algorithm_id,
            firm_lei=firm_lei,
            firm_name=firm_name,
            test_suite_id=suite.suite_id,
            test_environment=suite.environment,
            test_results_summary=metrics,
            test_pass_rate=metrics["pass_rate_pct"],
            critical_tests_passed=metrics["critical_passed"],
            critical_tests_total=metrics["critical_total"],
            validity_days=self.default_validity_days,
            issued_by=issued_by,
        )

        # Add warning condition if there were warnings
        if metrics["warnings"] > 0:
            cert.add_condition(CertificateCondition(
                description="Review test warnings before deployment",
                requirement=f"Review and document resolution of {metrics['warnings']} test warnings",
                deadline=date.today() + timedelta(days=14),
            ))

        self._certificates[cert.certificate_id] = cert
        logger.info(f"Certificate {cert.certificate_number} created for {suite.algorithm_id}")

        return cert

    # =========================================================================
    # Certificate Management
    # =========================================================================

    def get_certificate(self, certificate_id: str) -> Optional[ConformanceCertificate]:
        """Get certificate by ID."""
        return self._certificates.get(certificate_id)

    def get_certificate_by_number(self, certificate_number: str) -> Optional[ConformanceCertificate]:
        """Get certificate by certificate number."""
        for cert in self._certificates.values():
            if cert.certificate_number == certificate_number:
                return cert
        return None

    def get_certificates_for_algorithm(
        self,
        algorithm_id: str,
        include_invalid: bool = False,
    ) -> List[ConformanceCertificate]:
        """Get all certificates for an algorithm."""
        certs = [
            c for c in self._certificates.values()
            if c.algorithm_id == algorithm_id
        ]

        if not include_invalid:
            certs = [c for c in certs if c.is_valid()]

        return sorted(certs, key=lambda c: c.created_timestamp_ns, reverse=True)

    def get_valid_certificate(self, algorithm_id: str, version: str) -> Optional[ConformanceCertificate]:
        """Get valid certificate for specific algorithm version."""
        for cert in self._certificates.values():
            if (
                cert.algorithm_id == algorithm_id
                and cert.algorithm_version == version
                and cert.is_valid()
            ):
                return cert
        return None

    def get_expiring_certificates(self, days: Optional[int] = None) -> List[ConformanceCertificate]:
        """Get certificates expiring within specified days."""
        if days is None:
            days = self.expiry_warning_days

        expiring = []
        for cert in self._certificates.values():
            if cert.status == CertificateStatus.APPROVED:
                remaining = cert.days_until_expiry()
                if remaining is not None and 0 < remaining <= days:
                    expiring.append(cert)

        return sorted(expiring, key=lambda c: c.valid_until or date.max)

    def get_all_certificates(
        self,
        status: Optional[CertificateStatus] = None,
    ) -> List[ConformanceCertificate]:
        """Get all certificates, optionally filtered by status."""
        certs = list(self._certificates.values())
        if status:
            certs = [c for c in certs if c.status == status]
        return sorted(certs, key=lambda c: c.created_timestamp_ns, reverse=True)

    # =========================================================================
    # Certificate Actions
    # =========================================================================

    def submit_for_approval(self, certificate_id: str, issued_by: str) -> bool:
        """Submit certificate for approval."""
        cert = self.get_certificate(certificate_id)
        if cert:
            cert.submit_for_approval(issued_by)
            return True
        return False

    def approve_certificate(
        self,
        certificate_id: str,
        approved_by: str,
        notes: str = "",
    ) -> bool:
        """Approve a certificate."""
        cert = self.get_certificate(certificate_id)
        if cert and cert.status == CertificateStatus.PENDING_APPROVAL:
            cert.approve(approved_by, notes)
            return True
        return False

    def revoke_certificate(
        self,
        certificate_id: str,
        revoked_by: str,
        reason: str,
    ) -> bool:
        """Revoke a certificate."""
        cert = self.get_certificate(certificate_id)
        if cert and cert.status == CertificateStatus.APPROVED:
            cert.revoke(revoked_by, reason)
            return True
        return False

    def check_and_expire_certificates(self) -> List[str]:
        """Check and expire any certificates past their validity."""
        expired = []
        for cert in self._certificates.values():
            if cert.status == CertificateStatus.APPROVED and cert.is_expired():
                cert.expire()
                expired.append(cert.certificate_number)

        if expired:
            logger.info(f"Expired {len(expired)} certificates")

        return expired

    # =========================================================================
    # Deployment Authorization
    # =========================================================================

    def can_deploy(self, algorithm_id: str, version: str) -> Tuple[bool, str]:
        """
        Check if algorithm can be deployed.

        Args:
            algorithm_id: Algorithm identifier.
            version: Algorithm version.

        Returns:
            Tuple of (authorized, reason).
        """
        cert = self.get_valid_certificate(algorithm_id, version)

        if cert is None:
            return False, "No valid certificate found for this algorithm version"

        if not cert.is_valid():
            return False, f"Certificate {cert.certificate_number} is not valid"

        unsatisfied = cert.get_unsatisfied_conditions()
        if unsatisfied:
            return False, f"{len(unsatisfied)} condition(s) not satisfied"

        return True, f"Deployment authorized per certificate {cert.certificate_number}"

    # =========================================================================
    # Persistence
    # =========================================================================

    def save_certificate(self, certificate: ConformanceCertificate) -> None:
        """Save certificate to storage."""
        file_path = self.storage_path / f"{certificate.certificate_id}.json"
        with open(file_path, "w") as f:
            f.write(certificate.to_json())
        logger.debug(f"Saved certificate {certificate.certificate_number}")

    def load_certificate(self, certificate_id: str) -> Optional[ConformanceCertificate]:
        """Load certificate from storage."""
        file_path = self.storage_path / f"{certificate_id}.json"
        if file_path.exists():
            with open(file_path, "r") as f:
                cert = ConformanceCertificate.from_json(f.read())
                self._certificates[cert.certificate_id] = cert
                return cert
        return None

    def load_all_certificates(self) -> int:
        """Load all certificates from storage."""
        count = 0
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    cert = ConformanceCertificate.from_json(f.read())
                    self._certificates[cert.certificate_id] = cert
                    count += 1
            except Exception as e:
                logger.error(f"Failed to load certificate {file_path}: {e}")

        logger.info(f"Loaded {count} certificates")
        return count

    def save_all_certificates(self) -> int:
        """Save all certificates to storage."""
        count = 0
        for cert in self._certificates.values():
            try:
                self.save_certificate(cert)
                count += 1
            except Exception as e:
                logger.error(f"Failed to save certificate {cert.certificate_number}: {e}")

        return count

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_certificate_summary(self) -> Dict[str, Any]:
        """Get summary of all certificates."""
        status_counts = {status: 0 for status in CertificateStatus}
        type_counts = {cert_type: 0 for cert_type in CertificateType}

        for cert in self._certificates.values():
            status_counts[cert.status] += 1
            type_counts[cert.certificate_type] += 1

        expiring = self.get_expiring_certificates()

        return {
            "total_certificates": len(self._certificates),
            "status_counts": {k.value: v for k, v in status_counts.items()},
            "type_counts": {k.value: v for k, v in type_counts.items()},
            "expiring_within_30_days": len(expiring),
            "expiring_certificates": [
                {
                    "number": c.certificate_number,
                    "algorithm": c.algorithm_id,
                    "expires": c.valid_until.isoformat() if c.valid_until else None,
                }
                for c in expiring[:5]  # Top 5
            ],
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_certificate(
    algorithm_id: str,
    algorithm_version: str,
    firm_lei: str,
    firm_name: str,
    test_suite_id: str,
    test_pass_rate: float,
    critical_passed: int,
    critical_total: int,
    certificate_type: CertificateType = CertificateType.INITIAL,
    validity_days: int = 365,
) -> ConformanceCertificate:
    """
    Create a new conformance certificate.

    Args:
        algorithm_id: Algorithm identifier.
        algorithm_version: Algorithm version.
        firm_lei: Firm's LEI.
        firm_name: Firm's legal name.
        test_suite_id: Test suite ID.
        test_pass_rate: Pass rate percentage.
        critical_passed: Number of critical tests passed.
        critical_total: Total critical tests.
        certificate_type: Type of certification.
        validity_days: Days of validity.

    Returns:
        New ConformanceCertificate instance.
    """
    return ConformanceCertificate(
        certificate_type=certificate_type,
        algorithm_id=algorithm_id,
        algorithm_version=algorithm_version,
        firm_lei=firm_lei,
        firm_name=firm_name,
        test_suite_id=test_suite_id,
        test_pass_rate=test_pass_rate,
        critical_tests_passed=critical_passed,
        critical_tests_total=critical_total,
        validity_days=validity_days,
    )


def create_certificate_manager(
    storage_path: str = "state/compliance/certificates",
    default_validity_days: int = 365,
) -> CertificateManager:
    """
    Create a certificate manager.

    Args:
        storage_path: Path for certificate storage.
        default_validity_days: Default validity period.

    Returns:
        CertificateManager instance.
    """
    return CertificateManager(
        storage_path=storage_path,
        default_validity_days=default_validity_days,
    )


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "CertificateStatus",
    "CertificateType",
    "DeploymentApproval",
    # Data classes
    "CertificateCondition",
    "ConformanceCertificate",
    # Manager
    "CertificateManager",
    # Factory functions
    "create_certificate",
    "create_certificate_manager",
]
