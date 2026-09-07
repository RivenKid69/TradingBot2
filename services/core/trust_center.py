# -*- coding: utf-8 -*-
"""
Trust Center Platform (Block 2.12).

Implements trust center for transparency:
- Security documentation
- Compliance certifications
- Security posture reporting
- Self-service access for clients

DORA References:
    - Article 28: Third-Party ICT Risk (transparency requirements)
    - Article 30: Contractual Arrangements
    - RTS CDR 2024/1774: Documentation requirements
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


class DocumentType(Enum):
    """Trust center document types."""

    SOC2_REPORT = "soc2_report"
    ISO27001_CERT = "iso27001_cert"
    PENTEST_REPORT = "pentest_report"
    SECURITY_WHITEPAPER = "security_whitepaper"
    PRIVACY_POLICY = "privacy_policy"
    DPA = "dpa"  # Data Processing Agreement
    SUBPROCESSOR_LIST = "subprocessor_list"
    INCIDENT_REPORT = "incident_report"
    SLA_DOCUMENT = "sla_document"
    DORA_COMPLIANCE = "dora_compliance"


class AccessLevel(Enum):
    """Document access levels."""

    PUBLIC = "public"
    NDA_REQUIRED = "nda_required"
    CUSTOMER_ONLY = "customer_only"
    ENTERPRISE_ONLY = "enterprise_only"
    INTERNAL = "internal"


class CertificationType(Enum):
    """Certification types."""

    SOC2_TYPE_I = "soc2_type_i"
    SOC2_TYPE_II = "soc2_type_ii"
    ISO_27001 = "iso_27001"
    ISO_27017 = "iso_27017"
    ISO_27018 = "iso_27018"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    DORA = "dora"


class ComplianceStatus(Enum):
    """Compliance status."""

    COMPLIANT = "compliant"
    IN_PROGRESS = "in_progress"
    PLANNED = "planned"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class TrustDocument:
    """Trust center document."""

    document_id: str = ""
    title: str = ""
    description: str = ""
    document_type: DocumentType = DocumentType.SECURITY_WHITEPAPER

    # Access
    access_level: AccessLevel = AccessLevel.PUBLIC
    requires_nda: bool = False

    # Version
    version: str = "1.0"
    effective_date: str = ""
    expiry_date: str = ""

    # Storage
    file_path: str = ""
    file_size_kb: int = 0
    file_hash: str = ""

    # Metadata
    created_at: str = ""
    updated_at: str = ""
    created_by: str = ""

    # Stats
    download_count: int = 0
    last_downloaded: str = ""

    def __post_init__(self):
        if not self.document_id:
            self.document_id = f"DOC-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class CertificationRecord:
    """Certification record."""

    cert_id: str = ""
    certification_type: CertificationType = CertificationType.SOC2_TYPE_II
    name: str = ""
    description: str = ""

    # Validity
    issued_date: str = ""
    expiry_date: str = ""
    is_valid: bool = True

    # Issuer
    issuing_body: str = ""
    auditor: str = ""
    certificate_number: str = ""

    # Scope
    scope_description: str = ""
    services_covered: List[str] = field(default_factory=list)

    # Documents
    certificate_document_id: str = ""
    report_document_id: str = ""

    def __post_init__(self):
        if not self.cert_id:
            self.cert_id = f"CERT-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class SecurityPosture:
    """Security posture summary."""

    posture_id: str = ""
    generated_at: str = ""

    # Overview
    overall_rating: str = "A"  # A, B, C, D, F
    security_score: int = 95

    # Certifications
    active_certifications: List[str] = field(default_factory=list)
    pending_certifications: List[str] = field(default_factory=list)

    # Controls
    controls_implemented: int = 0
    controls_total: int = 0
    control_coverage_percent: float = 0.0

    # Recent activity
    last_pentest_date: str = ""
    last_audit_date: str = ""
    last_incident_date: str = ""
    days_since_incident: int = 0

    # DORA specific
    dora_compliance_status: str = "compliant"
    dora_articles_covered: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.posture_id:
            self.posture_id = f"POST-{uuid.uuid4().hex[:8].upper()}"
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()


@dataclass
class TrustCenterConfig:
    """Configuration for TrustCenterPlatform."""

    organization_name: str = "Quantitative Research Platform"
    support_email: str = "security@platform.com"
    nda_required_default: bool = False
    log_all_access: bool = True
    log_path: str = "logs/core/trust_center"
    alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


class TrustCenterPlatform:
    """Trust Center Platform."""

    def __init__(self, config: Optional[TrustCenterConfig] = None):
        self.config = config or TrustCenterConfig()
        self._documents: Dict[str, TrustDocument] = {}
        self._certifications: Dict[str, CertificationRecord] = {}
        self._access_log: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._init_default_content()
        logger.info("TrustCenterPlatform initialized")

    def _init_default_content(self) -> None:
        """Initialize default trust center content."""
        # Add default documents
        self.add_document(
            title="Security Overview",
            description="Overview of our security practices and controls",
            document_type=DocumentType.SECURITY_WHITEPAPER,
            access_level=AccessLevel.PUBLIC,
        )

        self.add_document(
            title="DORA Compliance Statement",
            description="Our compliance with DORA regulations",
            document_type=DocumentType.DORA_COMPLIANCE,
            access_level=AccessLevel.CUSTOMER_ONLY,
        )

        self.add_document(
            title="Sub-processor List",
            description="List of our sub-processors per DORA Article 30",
            document_type=DocumentType.SUBPROCESSOR_LIST,
            access_level=AccessLevel.CUSTOMER_ONLY,
        )

    def add_document(
        self,
        title: str,
        description: str,
        document_type: DocumentType,
        access_level: AccessLevel = AccessLevel.PUBLIC,
        file_path: str = "",
        version: str = "1.0",
    ) -> TrustDocument:
        """Add a document to the trust center."""
        doc = TrustDocument(
            title=title,
            description=description,
            document_type=document_type,
            access_level=access_level,
            file_path=file_path,
            version=version,
            requires_nda=access_level == AccessLevel.NDA_REQUIRED,
        )
        with self._lock:
            self._documents[doc.document_id] = doc
        return doc

    def add_certification(
        self,
        certification_type: CertificationType,
        name: str,
        issued_date: str,
        expiry_date: str,
        issuing_body: str,
        scope_description: str = "",
        services_covered: Optional[List[str]] = None,
    ) -> CertificationRecord:
        """Add a certification record."""
        cert = CertificationRecord(
            certification_type=certification_type,
            name=name,
            issued_date=issued_date,
            expiry_date=expiry_date,
            issuing_body=issuing_body,
            scope_description=scope_description,
            services_covered=services_covered or [],
        )

        # Check validity
        if cert.expiry_date:
            expiry_str = cert.expiry_date.replace("Z", "+00:00")
            expiry = datetime.fromisoformat(expiry_str)
            # If no timezone info, assume UTC
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            cert.is_valid = expiry > datetime.now(timezone.utc)

        with self._lock:
            self._certifications[cert.cert_id] = cert

        return cert

    def get_document(
        self,
        document_id: str,
        accessor_id: str = "",
        access_level: AccessLevel = AccessLevel.PUBLIC,
    ) -> Optional[TrustDocument]:
        """Get a document with access control."""
        with self._lock:
            if document_id not in self._documents:
                return None

            doc = self._documents[document_id]

            # Check access level
            access_hierarchy = [
                AccessLevel.PUBLIC,
                AccessLevel.NDA_REQUIRED,
                AccessLevel.CUSTOMER_ONLY,
                AccessLevel.ENTERPRISE_ONLY,
                AccessLevel.INTERNAL,
            ]

            if access_hierarchy.index(access_level) < access_hierarchy.index(doc.access_level):
                self._log_access(document_id, accessor_id, "denied")
                return None

            # Log access
            doc.download_count += 1
            doc.last_downloaded = datetime.now(timezone.utc).isoformat()
            self._log_access(document_id, accessor_id, "granted")

        return doc

    def _log_access(self, document_id: str, accessor_id: str, result: str) -> None:
        """Log document access."""
        log_entry = {
            "document_id": document_id,
            "accessor_id": accessor_id,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._access_log.append(log_entry)

    def get_public_documents(self) -> List[TrustDocument]:
        """Get all public documents."""
        with self._lock:
            return [d for d in self._documents.values() if d.access_level == AccessLevel.PUBLIC]

    def get_certifications(self, valid_only: bool = True) -> List[CertificationRecord]:
        """Get certifications."""
        with self._lock:
            certs = list(self._certifications.values())
            if valid_only:
                certs = [c for c in certs if c.is_valid]
            return certs

    def get_security_posture(self) -> SecurityPosture:
        """Generate current security posture."""
        with self._lock:
            certs = list(self._certifications.values())
            docs = list(self._documents.values())

        active_certs = [c.certification_type.value for c in certs if c.is_valid]

        posture = SecurityPosture(
            overall_rating="A" if len(active_certs) >= 2 else "B",
            security_score=95 if len(active_certs) >= 2 else 85,
            active_certifications=active_certs,
            pending_certifications=["iso_27001"] if "iso_27001" not in active_certs else [],
            controls_implemented=150,
            controls_total=160,
            control_coverage_percent=93.75,
            dora_compliance_status="compliant",
            dora_articles_covered=[
                "Article 5",
                "Article 6",
                "Article 9",
                "Article 10",
                "Article 11",
                "Article 12",
                "Article 17",
                "Article 28",
                "Article 30",
            ],
        )

        return posture

    def get_trust_center_summary(self) -> Dict[str, Any]:
        """Get trust center summary."""
        posture = self.get_security_posture()
        certs = self.get_certifications()
        docs = self.get_public_documents()

        return {
            "organization": self.config.organization_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "security_posture": asdict(posture),
            "certifications": {
                "active": len(certs),
                "list": [
                    {"type": c.certification_type.value, "expiry": c.expiry_date} for c in certs
                ],
            },
            "documents": {
                "public": len(docs),
                "types": list(set(d.document_type.value for d in docs)),
            },
            "support_contact": self.config.support_email,
        }

    def export_for_client(self, client_id: str, access_level: AccessLevel) -> Dict[str, Any]:
        """Export trust center data for a client."""
        with self._lock:
            docs = [
                asdict(d)
                for d in self._documents.values()
                if self._can_access(d.access_level, access_level)
            ]
            certs = [asdict(c) for c in self._certifications.values() if c.is_valid]

        return {
            "export_date": datetime.now(timezone.utc).isoformat(),
            "client_id": client_id,
            "access_level": access_level.value,
            "security_posture": asdict(self.get_security_posture()),
            "certifications": certs,
            "documents": docs,
            "dora_compliance": {
                "status": "compliant",
                "articles_covered": [
                    "Article 28 - Third-Party ICT Risk",
                    "Article 30 - Contractual Arrangements",
                ],
            },
        }

    def _can_access(self, doc_level: AccessLevel, user_level: AccessLevel) -> bool:
        """Check if user can access document."""
        hierarchy = [
            AccessLevel.PUBLIC,
            AccessLevel.NDA_REQUIRED,
            AccessLevel.CUSTOMER_ONLY,
            AccessLevel.ENTERPRISE_ONLY,
            AccessLevel.INTERNAL,
        ]
        return hierarchy.index(user_level) >= hierarchy.index(doc_level)


def create_trust_center(
    config: Optional[TrustCenterConfig] = None,
) -> TrustCenterPlatform:
    """Create a TrustCenterPlatform instance."""
    return TrustCenterPlatform(config=config)
