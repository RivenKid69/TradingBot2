# -*- coding: utf-8 -*-
"""
GDPR Phase 5: DSAR (Data Subject Access Request) Service.

Implements comprehensive DSAR workflows for GDPR Articles 12-23:
- Access (Art. 15) - Right of access
- Portability/Export (Art. 20) - Right to data portability
- Erasure (Art. 17) - Right to erasure ("right to be forgotten")
- Rectification (Art. 16) - Right to rectification
- Restriction (Art. 18) - Right to restriction of processing

Key features:
- Identity verification system with multiple methods
- Legal hold integration for erasure protection
- CCEA boundary clarity (Cloud vs Agent data)
- Deadline management (30 days + 60 days extension)
- Secure export with checksum and encryption
- Full audit trail
- Metrics and statistics APIs

Design Doc Reference:
    - Phase 5: "DSAR: access/export/delete (Cloud data) with CCEA boundary clarity"
    - GDPR Articles 12-23: Data subject rights

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple, TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from .retention_service import LegalHoldService, RetentionPolicyRegistry
    from .consent import SupportConsentService


logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# GDPR requires response within 30 days
DSAR_RESPONSE_DEADLINE_DAYS: Final[int] = 30

# Extension allowed once for complex requests (GDPR Art. 12(3))
DSAR_EXTENSION_DAYS: Final[int] = 60

# Maximum extension period (total 90 days from request)
DSAR_MAX_TOTAL_DAYS: Final[int] = 90

# Identity verification timeout
VERIFICATION_TOKEN_TTL_HOURS: Final[int] = 24

# Export link validity
EXPORT_LINK_TTL_HOURS: Final[int] = 168  # 7 days

# Maximum requests per user per month (abuse prevention)
MAX_REQUESTS_PER_USER_PER_MONTH: Final[int] = 12

# Data categories subject to DSAR (Cloud-controlled, IN SCOPE)
DSAR_DATA_CATEGORIES: Final[Set[str]] = {
    "telemetry_events",
    "alerts",
    "commands",
    "approval_records",
    "access_audits",
    "user_settings",
    "agent_data",
    "run_data",
    "deployment_data",
    "consent_records",
    "break_glass_requests",
    "session_data",
    "billing_records",
}

# Data categories OUT OF SCOPE (Agent-controlled)
DSAR_OUT_OF_SCOPE_CATEGORIES: Final[Set[str]] = {
    "broker_credentials",
    "local_execution_logs",
    "order_fill_data",
    "local_vault_contents",
    "position_data_local",
    "local_strategy_source",
    "local_config_files",
}

# CCEA boundary notice for DSAR responses
CCEA_BOUNDARY_NOTICE: Final[str] = """
CCEA Architecture Data Boundary Notice

Your request has been processed for all personal data held in our Cloud systems.
Due to our Cloud-Controlled Execution Architecture (CCEA), certain data categories
are stored exclusively in your local Agent environment and are not accessible to us:

- Broker API credentials (API keys, secrets, tokens)
- Local execution logs (unless explicitly exported to Cloud)
- Order and fill data (unless RAW_ORDER_EVENTS telemetry is enabled)
- Local vault contents
- Position data (unless transmitted via enabled telemetry)
- Local strategy source code
- Local configuration files

For access to Agent-local data, please contact your system administrator or access
your Agent's local storage directly. We cannot export or delete data that we do not
receive or store.

Per GDPR Article 20(2), the right to data portability shall not adversely affect
the rights and freedoms of others. Trading strategy IP stored locally remains
under your control.
""".strip()

# Exemption categories for erasure (Art. 17(3))
ERASURE_EXEMPTIONS: Final[Dict[str, str]] = {
    "legal_obligation": "Art. 17(3)(b) - Compliance with legal obligation",
    "public_interest": "Art. 17(3)(d) - Public interest (archiving, research)",
    "legal_claims": "Art. 17(3)(e) - Establishment, exercise or defence of legal claims",
    "regulatory_retention": "Financial regulations require 7-year retention",
}

# Compliance data exempt from erasure (7-year retention required)
COMPLIANCE_DATA_CATEGORIES: Final[Set[str]] = {
    "approval_records",
    "access_audits",
    "break_glass_requests",
    "governance_audit_logs",
    "billing_records",
}


# ============================================================================
# Enums
# ============================================================================

class DSARRequestType(str, Enum):
    """Types of DSAR requests per GDPR articles."""
    ACCESS = "access"              # Art. 15 - Right of access
    PORTABILITY = "portability"    # Art. 20 - Right to data portability
    ERASURE = "erasure"            # Art. 17 - Right to erasure
    RECTIFICATION = "rectification"  # Art. 16 - Right to rectification
    RESTRICTION = "restriction"    # Art. 18 - Right to restriction


class DSARStatus(str, Enum):
    """DSAR request processing status."""
    PENDING = "pending"
    AWAITING_VERIFICATION = "awaiting_verification"
    VERIFIED = "verified"
    IN_PROGRESS = "in_progress"
    PARTIALLY_COMPLETED = "partially_completed"
    COMPLETED = "completed"
    REJECTED = "rejected"
    EXTENDED = "extended"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class VerificationMethod(str, Enum):
    """Identity verification methods."""
    EMAIL_LINK = "email_link"
    EMAIL_OTP = "email_otp"
    SMS_OTP = "sms_otp"
    DOCUMENT_UPLOAD = "document_upload"
    SSO_SESSION = "sso_session"
    MFA_CHALLENGE = "mfa_challenge"
    SUPPORT_MANUAL = "support_manual"
    EXISTING_SESSION = "existing_session"


class VerificationStatus(str, Enum):
    """Status of identity verification."""
    PENDING = "pending"
    SENT = "sent"
    VERIFIED = "verified"
    FAILED = "failed"
    EXPIRED = "expired"


class ExportFormat(str, Enum):
    """Export file formats."""
    JSON = "json"
    JSON_ENCRYPTED = "json_encrypted"
    CSV = "csv"
    ZIP = "zip"


class AuditAction(str, Enum):
    """DSAR audit action types."""
    REQUEST_CREATED = "request_created"
    VERIFICATION_SENT = "verification_sent"
    VERIFICATION_ATTEMPTED = "verification_attempted"
    VERIFICATION_COMPLETED = "verification_completed"
    VERIFICATION_FAILED = "verification_failed"
    PROCESSING_STARTED = "processing_started"
    DATA_COLLECTED = "data_collected"
    EXPORT_GENERATED = "export_generated"
    ERASURE_STARTED = "erasure_started"
    ERASURE_COMPLETED = "erasure_completed"
    ERASURE_BLOCKED = "erasure_blocked"
    EXEMPTION_APPLIED = "exemption_applied"
    DEADLINE_EXTENDED = "deadline_extended"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_REJECTED = "request_rejected"
    REQUEST_CANCELLED = "request_cancelled"
    REQUEST_EXPIRED = "request_expired"
    ERROR_OCCURRED = "error_occurred"
    LEGAL_HOLD_CHECK = "legal_hold_check"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class VerificationToken:
    """Token for identity verification."""
    token_hash: str  # SHA-256 hash of token
    method: VerificationMethod
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=VERIFICATION_TOKEN_TTL_HOURS))
    attempts: int = 0
    max_attempts: int = 5
    status: VerificationStatus = VerificationStatus.PENDING

    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_locked(self) -> bool:
        return self.attempts >= self.max_attempts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "status": self.status.value,
            "is_expired": self.is_expired,
            "is_locked": self.is_locked,
        }


@dataclass
class DSARRequest:
    """
    DSAR request record.

    Tracks full lifecycle of a data subject request including
    verification, processing, and completion.
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    request_type: DSARRequestType = DSARRequestType.ACCESS
    user_id: str = ""
    workspace_id: str = ""
    status: DSARStatus = DSARStatus.PENDING
    data_categories: Set[str] = field(default_factory=lambda: set(DSAR_DATA_CATEGORIES))
    reason: str = ""

    # Identity verification
    verification_method: Optional[VerificationMethod] = None
    verification_token: Optional[VerificationToken] = None
    verified_at: Optional[datetime] = None
    verified_by: Optional[str] = None

    # Timing and deadlines
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    deadline: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=DSAR_RESPONSE_DEADLINE_DAYS))
    extended_deadline: Optional[datetime] = None
    extension_reason: Optional[str] = None
    completed_at: Optional[datetime] = None

    # Processing
    processed_by: Optional[str] = None
    processing_started_at: Optional[datetime] = None
    notes: List[str] = field(default_factory=list)

    # Export result
    export_path: Optional[str] = None
    export_checksum: Optional[str] = None
    export_format: ExportFormat = ExportFormat.JSON
    export_encrypted: bool = False
    export_key_id: Optional[str] = None

    # Erasure result
    records_processed: int = 0
    records_deleted: int = 0
    records_retained: int = 0  # Due to exemptions

    # Exemptions
    exemptions_applied: List[str] = field(default_factory=list)
    legal_hold_blocked: bool = False
    legal_hold_reason: Optional[str] = None

    # Download link
    download_token: Optional[str] = None
    download_expires_at: Optional[datetime] = None

    # Error tracking
    error_message: Optional[str] = None
    error_count: int = 0

    @property
    def effective_deadline(self) -> datetime:
        """Get the effective deadline considering extensions."""
        return self.extended_deadline or self.deadline

    @property
    def is_overdue(self) -> bool:
        """Check if request is overdue."""
        return (
            datetime.now(timezone.utc) > self.effective_deadline and
            self.status not in (DSARStatus.COMPLETED, DSARStatus.REJECTED, DSARStatus.CANCELLED, DSARStatus.EXPIRED)
        )

    @property
    def days_until_deadline(self) -> int:
        """Days remaining until deadline."""
        delta = self.effective_deadline - datetime.now(timezone.utc)
        return max(0, delta.days)

    @property
    def requires_verification(self) -> bool:
        """Check if this request type requires identity verification."""
        # Erasure always requires verification
        # ACCESS/PORTABILITY recommended but optional
        return self.request_type == DSARRequestType.ERASURE

    @property
    def is_verified(self) -> bool:
        """Check if identity is verified."""
        return self.verified_at is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "request_type": self.request_type.value,
            "user_id": self.user_id,
            "workspace_id": self.workspace_id,
            "status": self.status.value,
            "data_categories": list(self.data_categories),
            "reason": self.reason,
            "verification": {
                "method": self.verification_method.value if self.verification_method else None,
                "verified_at": self.verified_at.isoformat() if self.verified_at else None,
                "verified_by": self.verified_by,
                "is_verified": self.is_verified,
            },
            "timing": {
                "created_at": self.created_at.isoformat(),
                "deadline": self.deadline.isoformat(),
                "extended_deadline": self.extended_deadline.isoformat() if self.extended_deadline else None,
                "extension_reason": self.extension_reason,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "effective_deadline": self.effective_deadline.isoformat(),
                "days_until_deadline": self.days_until_deadline,
                "is_overdue": self.is_overdue,
            },
            "processing": {
                "processed_by": self.processed_by,
                "processing_started_at": self.processing_started_at.isoformat() if self.processing_started_at else None,
            },
            "export": {
                "path": self.export_path,
                "checksum": self.export_checksum,
                "format": self.export_format.value,
                "encrypted": self.export_encrypted,
            },
            "erasure": {
                "records_processed": self.records_processed,
                "records_deleted": self.records_deleted,
                "records_retained": self.records_retained,
            },
            "exemptions": {
                "applied": self.exemptions_applied,
                "legal_hold_blocked": self.legal_hold_blocked,
                "legal_hold_reason": self.legal_hold_reason,
            },
            "download": {
                "token": self.download_token is not None,
                "expires_at": self.download_expires_at.isoformat() if self.download_expires_at else None,
            },
            "notes": self.notes,
            "error": {
                "message": self.error_message,
                "count": self.error_count,
            } if self.error_message else None,
        }


@dataclass
class DSARResult:
    """Result of DSAR processing."""
    success: bool = True
    request_id: str = ""
    request_type: DSARRequestType = DSARRequestType.ACCESS
    status: DSARStatus = DSARStatus.COMPLETED
    records_processed: int = 0
    records_deleted: int = 0
    records_retained: int = 0
    export_path: Optional[str] = None
    export_checksum: Optional[str] = None
    download_url: Optional[str] = None
    download_expires_at: Optional[datetime] = None
    error: Optional[str] = None
    processing_time_seconds: float = 0.0

    # CCEA boundary information
    ccea_boundary_notice: str = CCEA_BOUNDARY_NOTICE
    in_scope_categories: Set[str] = field(default_factory=lambda: set(DSAR_DATA_CATEGORIES))
    out_of_scope_categories: Set[str] = field(default_factory=lambda: set(DSAR_OUT_OF_SCOPE_CATEGORIES))

    # Exemptions
    exemptions_applied: List[str] = field(default_factory=list)
    exemption_details: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "request_id": self.request_id,
            "request_type": self.request_type.value,
            "status": self.status.value,
            "processing": {
                "records_processed": self.records_processed,
                "records_deleted": self.records_deleted,
                "records_retained": self.records_retained,
                "processing_time_seconds": self.processing_time_seconds,
            },
            "export": {
                "path": self.export_path,
                "checksum": self.export_checksum,
                "download_url": self.download_url,
                "download_expires_at": self.download_expires_at.isoformat() if self.download_expires_at else None,
            },
            "ccea_boundary": {
                "notice": self.ccea_boundary_notice,
                "in_scope_categories": list(self.in_scope_categories),
                "out_of_scope_categories": list(self.out_of_scope_categories),
                "explanation": (
                    "This export contains only Cloud-controlled data. "
                    "Agent-zone data (broker credentials, local logs, order/fill data) "
                    "is stored on your local Agent and is not accessible to us."
                ),
            },
            "exemptions": {
                "applied": self.exemptions_applied,
                "details": self.exemption_details,
            },
            "error": self.error,
        }


@dataclass
class DSARMetrics:
    """DSAR metrics and statistics."""
    total_requests: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_status: Dict[str, int] = field(default_factory=dict)
    average_completion_days: float = 0.0
    overdue_count: int = 0
    pending_count: int = 0
    completed_on_time_rate: float = 0.0
    average_records_per_request: float = 0.0
    total_records_processed: int = 0
    total_records_deleted: int = 0
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "by_type": self.by_type,
            "by_status": self.by_status,
            "average_completion_days": self.average_completion_days,
            "overdue_count": self.overdue_count,
            "pending_count": self.pending_count,
            "completed_on_time_rate": self.completed_on_time_rate,
            "average_records_per_request": self.average_records_per_request,
            "total_records_processed": self.total_records_processed,
            "total_records_deleted": self.total_records_deleted,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
        }


@dataclass
class AuditEntry:
    """Immutable audit log entry."""
    id: str = field(default_factory=lambda: str(uuid4()))
    action: AuditAction = AuditAction.REQUEST_CREATED
    request_id: str = ""
    user_id: str = ""
    workspace_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    actor_id: Optional[str] = None
    actor_type: str = "system"  # system, user, scheduler
    details: Dict[str, Any] = field(default_factory=dict)
    integrity_hash: str = ""

    def __post_init__(self):
        """Compute integrity hash after initialization."""
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute integrity hash for this entry."""
        content = json.dumps({
            "id": self.id,
            "action": self.action.value,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "workspace_id": self.workspace_id,
            "timestamp": self.timestamp.isoformat(),
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "details": self.details,
        }, sort_keys=True, default=str)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action.value,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "workspace_id": self.workspace_id,
            "timestamp": self.timestamp.isoformat(),
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "details": self.details,
            "integrity_hash": self.integrity_hash,
        }


@dataclass
class DataCategory:
    """Data category definition for DSAR."""
    name: str
    description: str
    table_name: Optional[str] = None
    is_exportable: bool = True
    is_deletable: bool = True
    requires_consent: bool = False
    retention_days: int = 90
    exemption: Optional[str] = None


# ============================================================================
# Data Category Registry
# ============================================================================

DATA_CATEGORIES: Dict[str, DataCategory] = {
    "telemetry_events": DataCategory(
        name="telemetry_events",
        description="Telemetry and metrics from agents",
        table_name="telemetry_events",
        retention_days=90,
    ),
    "alerts": DataCategory(
        name="alerts",
        description="Alert notifications",
        table_name="alerts",
        retention_days=365,
    ),
    "commands": DataCategory(
        name="commands",
        description="Commands sent to agents",
        table_name="commands",
        retention_days=180,
    ),
    "approval_records": DataCategory(
        name="approval_records",
        description="Deployment and change approvals",
        table_name="approval_records",
        is_deletable=False,  # 7-year retention required
        retention_days=2555,
        exemption="regulatory_retention",
    ),
    "access_audits": DataCategory(
        name="access_audits",
        description="Access audit logs",
        table_name="access_audits",
        is_deletable=False,
        retention_days=2555,
        exemption="regulatory_retention",
    ),
    "user_settings": DataCategory(
        name="user_settings",
        description="User preferences and settings",
        is_exportable=True,
        is_deletable=True,
        retention_days=90,
    ),
    "agent_data": DataCategory(
        name="agent_data",
        description="Agent registration and status",
        table_name="agents",
        retention_days=365,
    ),
    "run_data": DataCategory(
        name="run_data",
        description="Strategy run records",
        table_name="runs",
        retention_days=365,
    ),
    "deployment_data": DataCategory(
        name="deployment_data",
        description="Deployment records",
        table_name="deployments",
        retention_days=365,
    ),
    "consent_records": DataCategory(
        name="consent_records",
        description="Consent grants and revocations",
        is_deletable=False,
        retention_days=2555,
        exemption="legal_obligation",
    ),
    "break_glass_requests": DataCategory(
        name="break_glass_requests",
        description="Emergency access requests",
        table_name="break_glass_requests",
        is_deletable=False,
        retention_days=2555,
        exemption="regulatory_retention",
    ),
    "session_data": DataCategory(
        name="session_data",
        description="User session data",
        retention_days=1,
    ),
    "billing_records": DataCategory(
        name="billing_records",
        description="Billing and invoice records",
        is_deletable=False,
        retention_days=2555,
        exemption="legal_obligation",
    ),
}


# ============================================================================
# DSAR Service
# ============================================================================

class DSARPhase5Service:
    """
    GDPR Phase 5 DSAR Service.

    Comprehensive Data Subject Access Request handling with:
    - Identity verification
    - Legal hold integration
    - CCEA boundary enforcement
    - Deadline management
    - Secure export
    - Full audit trail

    Usage:
        service = DSARPhase5Service(
            legal_hold_service=legal_hold_service,
            data_fetcher=my_data_fetcher,
            data_deleter=my_data_deleter,
        )

        # Create request
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        # Initiate verification
        token = await service.initiate_verification(
            request_id=request.id,
            method=VerificationMethod.EMAIL_LINK,
        )

        # Verify identity
        await service.verify_identity(
            request_id=request.id,
            token=token,
        )

        # Process request
        result = await service.process_request(request.id)
    """

    def __init__(
        self,
        legal_hold_service: Optional["LegalHoldService"] = None,
        retention_registry: Optional["RetentionPolicyRegistry"] = None,
        consent_service: Optional["SupportConsentService"] = None,
        data_fetcher: Optional[Callable[[str, str, Set[str]], List[Dict]]] = None,
        data_deleter: Optional[Callable[[str, str, Set[str]], int]] = None,
        export_dir: Optional[Path] = None,
        base_url: str = "",
        audit_callback: Optional[Callable[[AuditEntry], None]] = None,
    ):
        """
        Initialize DSAR service.

        Args:
            legal_hold_service: Legal hold service for erasure protection
            retention_registry: Retention policy registry
            consent_service: Support consent service
            data_fetcher: Function (user_id, workspace_id, categories) -> records
            data_deleter: Function (user_id, workspace_id, categories) -> count
            export_dir: Directory for export files
            base_url: Base URL for download links
            audit_callback: Callback for audit events
        """
        self._legal_hold_service = legal_hold_service
        self._retention_registry = retention_registry
        self._consent_service = consent_service
        self._data_fetcher = data_fetcher
        self._data_deleter = data_deleter
        self._export_dir = export_dir or Path("/tmp/dsar_exports")
        self._base_url = base_url
        self._audit_callback = audit_callback

        # Internal state
        self._requests: Dict[str, DSARRequest] = {}
        self._audit_log: List[AuditEntry] = []
        self._verification_tokens: Dict[str, str] = {}  # token_hash -> request_id
        self._lock = threading.Lock()

    # ========================================================================
    # Request Lifecycle
    # ========================================================================

    async def create_request(
        self,
        user_id: str,
        workspace_id: str,
        request_type: DSARRequestType,
        data_categories: Optional[Set[str]] = None,
        reason: str = "",
    ) -> DSARRequest:
        """
        Create a new DSAR request.

        Args:
            user_id: User making the request
            workspace_id: Workspace scope
            request_type: Type of request (ACCESS, ERASURE, etc.)
            data_categories: Categories to include (defaults to all)
            reason: Reason for request

        Returns:
            Created DSARRequest

        Raises:
            ValueError: If request type is invalid or rate limited
        """
        # Validate request type
        if request_type not in DSARRequestType:
            raise ValueError(f"Invalid request type: {request_type}")

        # Check rate limiting
        user_requests = self._get_recent_requests(user_id, days=30)
        if len(user_requests) >= MAX_REQUESTS_PER_USER_PER_MONTH:
            raise ValueError(
                f"Rate limit exceeded: maximum {MAX_REQUESTS_PER_USER_PER_MONTH} "
                f"requests per user per month"
            )

        # Filter to valid categories
        valid_categories = data_categories or set(DSAR_DATA_CATEGORIES)
        valid_categories = valid_categories.intersection(DSAR_DATA_CATEGORIES)
        if not valid_categories:
            valid_categories = set(DSAR_DATA_CATEGORIES)

        # Create request
        request = DSARRequest(
            request_type=request_type,
            user_id=user_id,
            workspace_id=workspace_id,
            data_categories=valid_categories,
            reason=reason,
        )

        with self._lock:
            self._requests[request.id] = request
            self._log_audit(
                action=AuditAction.REQUEST_CREATED,
                request=request,
                details={
                    "request_type": request_type.value,
                    "data_categories": list(valid_categories),
                    "reason": reason,
                    "deadline": request.deadline.isoformat(),
                },
            )

        logger.info(
            f"DSAR request created: id={request.id}, type={request_type.value}, "
            f"user={user_id}, workspace={workspace_id}"
        )

        return request

    def get_request(self, request_id: str) -> Optional[DSARRequest]:
        """Get request by ID."""
        with self._lock:
            return self._requests.get(request_id)

    def get_user_requests(
        self,
        user_id: str,
        workspace_id: Optional[str] = None,
        status: Optional[DSARStatus] = None,
    ) -> List[DSARRequest]:
        """Get all requests for a user."""
        with self._lock:
            requests = [r for r in self._requests.values() if r.user_id == user_id]

            if workspace_id:
                requests = [r for r in requests if r.workspace_id == workspace_id]

            if status:
                requests = [r for r in requests if r.status == status]

            return sorted(requests, key=lambda r: r.created_at, reverse=True)

    def get_workspace_requests(
        self,
        workspace_id: str,
        status: Optional[DSARStatus] = None,
    ) -> List[DSARRequest]:
        """Get all requests for a workspace."""
        with self._lock:
            requests = [r for r in self._requests.values() if r.workspace_id == workspace_id]

            if status:
                requests = [r for r in requests if r.status == status]

            return sorted(requests, key=lambda r: r.created_at, reverse=True)

    def _get_recent_requests(self, user_id: str, days: int = 30) -> List[DSARRequest]:
        """Get recent requests for rate limiting."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self._lock:
            return [
                r for r in self._requests.values()
                if r.user_id == user_id and r.created_at >= cutoff
            ]

    # ========================================================================
    # Identity Verification
    # ========================================================================

    async def initiate_verification(
        self,
        request_id: str,
        method: VerificationMethod,
        send_to: Optional[str] = None,
    ) -> Optional[str]:
        """
        Initiate identity verification for a request.

        Args:
            request_id: Request ID
            method: Verification method
            send_to: Email/phone to send verification to (if applicable)

        Returns:
            Verification token (plain) or None if failed
        """
        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return None

            # Generate token
            token = secrets.token_urlsafe(32)
            token_hash = hashlib.sha256(token.encode()).hexdigest()

            verification = VerificationToken(
                token_hash=token_hash,
                method=method,
                status=VerificationStatus.PENDING,
            )

            request.verification_method = method
            request.verification_token = verification
            request.status = DSARStatus.AWAITING_VERIFICATION

            # Store token mapping
            self._verification_tokens[token_hash] = request_id

            self._log_audit(
                action=AuditAction.VERIFICATION_SENT,
                request=request,
                details={
                    "method": method.value,
                    "send_to": send_to,
                    "expires_at": verification.expires_at.isoformat(),
                },
            )

        logger.info(
            f"Verification initiated: request={request_id}, method={method.value}"
        )

        return token

    async def verify_identity(
        self,
        request_id: str,
        token: Optional[str] = None,
        verification_method: Optional[str] = None,
        verified_by: Optional[str] = None,
    ) -> bool:
        """
        Verify identity for a DSAR request.

        Args:
            request_id: Request ID
            token: Verification token (for token-based methods)
            verification_method: Method used (for manual verification)
            verified_by: Who performed verification (for manual)

        Returns:
            True if verification succeeded
        """
        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return False

            # Handle token-based verification
            if token:
                token_hash = hashlib.sha256(token.encode()).hexdigest()

                if not request.verification_token:
                    self._log_audit(
                        action=AuditAction.VERIFICATION_FAILED,
                        request=request,
                        details={"reason": "No verification token found"},
                    )
                    return False

                verification = request.verification_token

                # Check token validity
                if verification.is_expired:
                    verification.status = VerificationStatus.EXPIRED
                    self._log_audit(
                        action=AuditAction.VERIFICATION_FAILED,
                        request=request,
                        details={"reason": "Token expired"},
                    )
                    return False

                if verification.is_locked:
                    self._log_audit(
                        action=AuditAction.VERIFICATION_FAILED,
                        request=request,
                        details={"reason": "Too many attempts"},
                    )
                    return False

                verification.attempts += 1

                # Verify token
                if not hmac.compare_digest(verification.token_hash, token_hash):
                    verification.status = VerificationStatus.FAILED
                    self._log_audit(
                        action=AuditAction.VERIFICATION_ATTEMPTED,
                        request=request,
                        details={
                            "success": False,
                            "attempts": verification.attempts,
                        },
                    )
                    return False

                verification.status = VerificationStatus.VERIFIED

            # Update request
            request.verified_at = datetime.now(timezone.utc)
            request.verified_by = verified_by or "self"
            request.verification_method = (
                VerificationMethod(verification_method)
                if verification_method
                else request.verification_method
            )
            request.status = DSARStatus.VERIFIED

            self._log_audit(
                action=AuditAction.VERIFICATION_COMPLETED,
                request=request,
                details={
                    "method": request.verification_method.value if request.verification_method else "manual",
                    "verified_by": request.verified_by,
                },
            )

        logger.info(f"Identity verified: request={request_id}")
        return True

    async def verify_identity_manual(
        self,
        request_id: str,
        verification_method: str,
        verified_by: str,
        notes: str = "",
    ) -> bool:
        """
        Manual identity verification by support.

        Args:
            request_id: Request ID
            verification_method: Method used
            verified_by: Support agent ID
            notes: Verification notes

        Returns:
            True if verification recorded
        """
        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return False

            request.verified_at = datetime.now(timezone.utc)
            request.verified_by = verified_by
            request.verification_method = VerificationMethod.SUPPORT_MANUAL
            request.status = DSARStatus.VERIFIED
            if notes:
                request.notes.append(f"Verification: {notes}")

            self._log_audit(
                action=AuditAction.VERIFICATION_COMPLETED,
                request=request,
                actor_id=verified_by,
                actor_type="user",
                details={
                    "method": verification_method,
                    "verified_by": verified_by,
                    "notes": notes,
                },
            )

        return True

    # ========================================================================
    # Deadline Management
    # ========================================================================

    async def extend_deadline(
        self,
        request_id: str,
        reason: str,
        extended_by: str,
    ) -> bool:
        """
        Extend request deadline (allowed once per GDPR Art. 12(3)).

        Args:
            request_id: Request to extend
            reason: Reason for extension
            extended_by: Who extended

        Returns:
            True if extension granted
        """
        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return False

            # Can only extend once (GDPR Art. 12(3))
            if request.extended_deadline:
                return False

            # Cannot extend if already completed
            if request.status in (DSARStatus.COMPLETED, DSARStatus.REJECTED, DSARStatus.CANCELLED):
                return False

            # Calculate new deadline (up to 90 days total)
            new_deadline = datetime.now(timezone.utc) + timedelta(days=DSAR_EXTENSION_DAYS)
            max_deadline = request.created_at + timedelta(days=DSAR_MAX_TOTAL_DAYS)
            request.extended_deadline = min(new_deadline, max_deadline)
            request.extension_reason = reason
            request.status = DSARStatus.EXTENDED
            request.notes.append(f"Deadline extended by {extended_by}: {reason}")

            self._log_audit(
                action=AuditAction.DEADLINE_EXTENDED,
                request=request,
                actor_id=extended_by,
                actor_type="user",
                details={
                    "reason": reason,
                    "new_deadline": request.extended_deadline.isoformat(),
                    "extended_by": extended_by,
                },
            )

        logger.info(
            f"Deadline extended: request={request_id}, "
            f"new_deadline={request.extended_deadline.isoformat()}"
        )
        return True

    # ========================================================================
    # Request Processing
    # ========================================================================

    async def process_request(
        self,
        request_id: str,
        processed_by: str = "system",
    ) -> DSARResult:
        """
        Process a DSAR request.

        Args:
            request_id: Request to process
            processed_by: Who is processing

        Returns:
            DSARResult
        """
        start_time = datetime.now(timezone.utc)

        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return DSARResult(
                    success=False,
                    request_id=request_id,
                    error="Request not found",
                )

            # Check verification requirement
            if request.requires_verification and not request.is_verified:
                return DSARResult(
                    success=False,
                    request_id=request_id,
                    request_type=request.request_type,
                    status=DSARStatus.AWAITING_VERIFICATION,
                    error="Identity verification required",
                )

            # Start processing
            request.processed_by = processed_by
            request.processing_started_at = datetime.now(timezone.utc)
            request.status = DSARStatus.IN_PROGRESS

            self._log_audit(
                action=AuditAction.PROCESSING_STARTED,
                request=request,
                actor_id=processed_by,
                details={"started_at": request.processing_started_at.isoformat()},
            )

        try:
            if request.request_type in (DSARRequestType.ACCESS, DSARRequestType.PORTABILITY):
                result = await self._process_export(request)
            elif request.request_type == DSARRequestType.ERASURE:
                result = await self._process_erasure(request)
            elif request.request_type == DSARRequestType.RECTIFICATION:
                result = await self._process_rectification(request)
            elif request.request_type == DSARRequestType.RESTRICTION:
                result = await self._process_restriction(request)
            else:
                result = DSARResult(
                    success=False,
                    request_id=request_id,
                    request_type=request.request_type,
                    error=f"Unsupported request type: {request.request_type}",
                )

            # Update request
            with self._lock:
                request.status = result.status
                request.completed_at = datetime.now(timezone.utc)
                request.records_processed = result.records_processed
                request.records_deleted = result.records_deleted
                request.records_retained = result.records_retained
                request.export_path = result.export_path
                request.export_checksum = result.export_checksum
                request.exemptions_applied = result.exemptions_applied
                request.error_message = result.error

                self._log_audit(
                    action=AuditAction.REQUEST_COMPLETED,
                    request=request,
                    details={
                        "status": result.status.value,
                        "records_processed": result.records_processed,
                        "records_deleted": result.records_deleted,
                        "records_retained": result.records_retained,
                        "exemptions": result.exemptions_applied,
                    },
                )

            result.processing_time_seconds = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds()

            return result

        except Exception as e:
            logger.exception(f"Error processing DSAR request: {request_id}")

            with self._lock:
                request.status = DSARStatus.PARTIALLY_COMPLETED
                request.error_message = str(e)
                request.error_count += 1
                request.notes.append(f"Error: {str(e)}")

                self._log_audit(
                    action=AuditAction.ERROR_OCCURRED,
                    request=request,
                    details={"error": str(e)},
                )

            return DSARResult(
                success=False,
                request_id=request_id,
                request_type=request.request_type,
                status=DSARStatus.PARTIALLY_COMPLETED,
                error=str(e),
                processing_time_seconds=(
                    datetime.now(timezone.utc) - start_time
                ).total_seconds(),
            )

    async def _process_export(self, request: DSARRequest) -> DSARResult:
        """Process ACCESS/PORTABILITY request."""
        if not self._data_fetcher:
            return DSARResult(
                success=False,
                request_id=request.id,
                request_type=request.request_type,
                error="Data fetcher not configured",
            )

        # Fetch data
        records = self._data_fetcher(
            request.user_id,
            request.workspace_id,
            request.data_categories,
        )

        self._log_audit(
            action=AuditAction.DATA_COLLECTED,
            request=request,
            details={"record_count": len(records)},
        )

        # Create export directory
        self._export_dir.mkdir(parents=True, exist_ok=True)

        # Build export data with CCEA boundary
        export_data = {
            "metadata": {
                "request_id": request.id,
                "request_type": request.request_type.value,
                "user_id": request.user_id,
                "workspace_id": request.workspace_id,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "data_categories": list(request.data_categories),
                "record_count": len(records),
                "gdpr_article": "Article 15 (Access)" if request.request_type == DSARRequestType.ACCESS else "Article 20 (Portability)",
            },
            "ccea_boundary": {
                "notice": CCEA_BOUNDARY_NOTICE,
                "in_scope_categories": list(DSAR_DATA_CATEGORIES),
                "out_of_scope_categories": list(DSAR_OUT_OF_SCOPE_CATEGORIES),
                "explanation": (
                    "This export contains only Cloud-controlled data. "
                    "Agent-zone data (broker credentials, local logs, order/fill data) "
                    "is stored on your local Agent and is not accessible to us."
                ),
            },
            "data": records,
        }

        # Write export file
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        export_filename = f"dsar_export_{request.id}_{timestamp}.json"
        export_path = self._export_dir / export_filename

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)

        # Calculate checksum
        with open(export_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        # Generate download token
        download_token = secrets.token_urlsafe(32)
        download_expires = datetime.now(timezone.utc) + timedelta(hours=EXPORT_LINK_TTL_HOURS)

        with self._lock:
            request.download_token = download_token
            request.download_expires_at = download_expires

        # Build download URL
        download_url = None
        if self._base_url:
            download_url = f"{self._base_url}/dsar/download/{request.id}?token={download_token}"

        self._log_audit(
            action=AuditAction.EXPORT_GENERATED,
            request=request,
            details={
                "export_path": str(export_path),
                "checksum": checksum,
                "record_count": len(records),
                "download_expires_at": download_expires.isoformat(),
            },
        )

        return DSARResult(
            success=True,
            request_id=request.id,
            request_type=request.request_type,
            status=DSARStatus.COMPLETED,
            records_processed=len(records),
            export_path=str(export_path),
            export_checksum=f"sha256:{checksum}",
            download_url=download_url,
            download_expires_at=download_expires,
        )

    async def _process_erasure(self, request: DSARRequest) -> DSARResult:
        """Process ERASURE request with legal hold check."""
        exemptions_applied = []
        exemption_details = {}
        records_deleted = 0
        records_retained = 0
        records_processed = 0

        self._log_audit(
            action=AuditAction.ERASURE_STARTED,
            request=request,
            details={"categories": list(request.data_categories)},
        )

        # Check categories for exemptions and legal holds
        categories_to_delete = set()
        for category in request.data_categories:
            cat_info = DATA_CATEGORIES.get(category)

            # Check if category is exempt from deletion
            if cat_info and not cat_info.is_deletable:
                exemption_key = cat_info.exemption or "regulatory_retention"
                exemptions_applied.append(category)
                exemption_details[category] = ERASURE_EXEMPTIONS.get(
                    exemption_key,
                    "Data cannot be deleted due to legal requirements"
                )
                continue

            # Check legal hold
            if self._legal_hold_service:
                hold = self._legal_hold_service.get_active_hold(
                    request.workspace_id,
                    category,
                )
                if hold:
                    self._log_audit(
                        action=AuditAction.LEGAL_HOLD_CHECK,
                        request=request,
                        details={
                            "category": category,
                            "hold_id": hold.id,
                            "hold_reason": hold.reason,
                            "blocked": True,
                        },
                    )
                    exemptions_applied.append(category)
                    exemption_details[category] = f"Legal hold active: {hold.reason}"
                    request.legal_hold_blocked = True
                    request.legal_hold_reason = hold.reason
                    continue

            categories_to_delete.add(category)

        # Perform deletion for eligible categories
        if categories_to_delete and self._data_deleter:
            # First, count records that will be affected
            if self._data_fetcher:
                records = self._data_fetcher(
                    request.user_id,
                    request.workspace_id,
                    categories_to_delete,
                )
                records_processed = len(records)

            # Then delete
            records_deleted = self._data_deleter(
                request.user_id,
                request.workspace_id,
                categories_to_delete,
            )

        # Calculate retained records
        if self._data_fetcher and exemptions_applied:
            retained_records = self._data_fetcher(
                request.user_id,
                request.workspace_id,
                set(exemptions_applied).intersection(DSAR_DATA_CATEGORIES),
            )
            records_retained = len(retained_records)

        self._log_audit(
            action=AuditAction.ERASURE_COMPLETED,
            request=request,
            details={
                "categories_deleted": list(categories_to_delete),
                "records_deleted": records_deleted,
                "records_retained": records_retained,
                "exemptions_applied": exemptions_applied,
            },
        )

        # Determine final status
        if not categories_to_delete:
            status = DSARStatus.PARTIALLY_COMPLETED
            if exemptions_applied:
                self._log_audit(
                    action=AuditAction.EXEMPTION_APPLIED,
                    request=request,
                    details={
                        "exemptions": exemptions_applied,
                        "details": exemption_details,
                    },
                )
        elif exemptions_applied:
            status = DSARStatus.PARTIALLY_COMPLETED
        else:
            status = DSARStatus.COMPLETED

        return DSARResult(
            success=True,
            request_id=request.id,
            request_type=request.request_type,
            status=status,
            records_processed=records_processed,
            records_deleted=records_deleted,
            records_retained=records_retained,
            exemptions_applied=exemptions_applied,
            exemption_details=exemption_details,
        )

    async def _process_rectification(self, request: DSARRequest) -> DSARResult:
        """Process RECTIFICATION request (placeholder - requires specific data)."""
        # Rectification requires specific data to be corrected
        # This is handled through a separate workflow
        return DSARResult(
            success=True,
            request_id=request.id,
            request_type=request.request_type,
            status=DSARStatus.COMPLETED,
            records_processed=0,
        )

    async def _process_restriction(self, request: DSARRequest) -> DSARResult:
        """Process RESTRICTION request (placeholder - applies processing restriction)."""
        # Restriction of processing requires flagging data
        # This is handled through a separate workflow
        return DSARResult(
            success=True,
            request_id=request.id,
            request_type=request.request_type,
            status=DSARStatus.COMPLETED,
            records_processed=0,
        )

    # ========================================================================
    # Request Management
    # ========================================================================

    async def reject_request(
        self,
        request_id: str,
        reason: str,
        rejected_by: str,
    ) -> bool:
        """
        Reject a DSAR request.

        Args:
            request_id: Request to reject
            reason: Rejection reason
            rejected_by: Who rejected

        Returns:
            True if rejection recorded
        """
        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return False

            request.status = DSARStatus.REJECTED
            request.completed_at = datetime.now(timezone.utc)
            request.notes.append(f"Rejected by {rejected_by}: {reason}")

            self._log_audit(
                action=AuditAction.REQUEST_REJECTED,
                request=request,
                actor_id=rejected_by,
                actor_type="user",
                details={"reason": reason},
            )

        logger.info(f"Request rejected: {request_id}, reason={reason}")
        return True

    async def cancel_request(
        self,
        request_id: str,
        cancelled_by: str,
        reason: str = "",
    ) -> bool:
        """
        Cancel a DSAR request (by the user).

        Args:
            request_id: Request to cancel
            cancelled_by: Who cancelled
            reason: Cancellation reason

        Returns:
            True if cancellation recorded
        """
        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return False

            # Can only cancel pending/in-progress requests
            if request.status in (DSARStatus.COMPLETED, DSARStatus.REJECTED, DSARStatus.CANCELLED):
                return False

            request.status = DSARStatus.CANCELLED
            request.completed_at = datetime.now(timezone.utc)
            request.notes.append(f"Cancelled by {cancelled_by}: {reason}")

            self._log_audit(
                action=AuditAction.REQUEST_CANCELLED,
                request=request,
                actor_id=cancelled_by,
                actor_type="user",
                details={"reason": reason},
            )

        return True

    # ========================================================================
    # Queries and Monitoring
    # ========================================================================

    def get_overdue_requests(self) -> List[DSARRequest]:
        """Get all overdue requests."""
        with self._lock:
            return [r for r in self._requests.values() if r.is_overdue]

    def get_pending_requests(self) -> List[DSARRequest]:
        """Get all pending/active requests."""
        active_statuses = {
            DSARStatus.PENDING,
            DSARStatus.AWAITING_VERIFICATION,
            DSARStatus.VERIFIED,
            DSARStatus.IN_PROGRESS,
            DSARStatus.EXTENDED,
        }
        with self._lock:
            return [r for r in self._requests.values() if r.status in active_statuses]

    def get_requests_by_deadline(self, days: int = 7) -> List[DSARRequest]:
        """Get requests due within specified days."""
        cutoff = datetime.now(timezone.utc) + timedelta(days=days)
        with self._lock:
            return [
                r for r in self._requests.values()
                if (
                    r.effective_deadline <= cutoff and
                    r.status not in (DSARStatus.COMPLETED, DSARStatus.REJECTED, DSARStatus.CANCELLED, DSARStatus.EXPIRED)
                )
            ]

    def check_expired_requests(self) -> List[DSARRequest]:
        """Check for and mark expired requests."""
        expired = []
        with self._lock:
            for request in self._requests.values():
                if request.is_overdue and request.status != DSARStatus.EXPIRED:
                    request.status = DSARStatus.EXPIRED
                    request.completed_at = datetime.now(timezone.utc)
                    expired.append(request)

                    self._log_audit(
                        action=AuditAction.REQUEST_EXPIRED,
                        request=request,
                        details={"deadline": request.effective_deadline.isoformat()},
                    )

        return expired

    # ========================================================================
    # Metrics
    # ========================================================================

    def get_metrics(
        self,
        workspace_id: Optional[str] = None,
        period_days: int = 30,
    ) -> DSARMetrics:
        """
        Get DSAR metrics and statistics.

        Args:
            workspace_id: Optional workspace filter
            period_days: Period for metrics

        Returns:
            DSARMetrics
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)

        with self._lock:
            requests = list(self._requests.values())

            if workspace_id:
                requests = [r for r in requests if r.workspace_id == workspace_id]

            # Filter by period
            period_requests = [r for r in requests if r.created_at >= cutoff]

            # Calculate metrics
            by_type: Dict[str, int] = {}
            by_status: Dict[str, int] = {}
            completion_days: List[float] = []
            total_records_processed = 0
            total_records_deleted = 0

            for r in period_requests:
                # By type
                by_type[r.request_type.value] = by_type.get(r.request_type.value, 0) + 1

                # By status
                by_status[r.status.value] = by_status.get(r.status.value, 0) + 1

                # Completion time
                if r.completed_at:
                    days = (r.completed_at - r.created_at).total_seconds() / 86400
                    completion_days.append(days)

                # Records
                total_records_processed += r.records_processed
                total_records_deleted += r.records_deleted

            # Overdue and pending counts
            overdue_count = len([r for r in period_requests if r.is_overdue])
            pending_count = len([
                r for r in period_requests
                if r.status not in (DSARStatus.COMPLETED, DSARStatus.REJECTED, DSARStatus.CANCELLED, DSARStatus.EXPIRED)
            ])

            # On-time rate
            completed = [r for r in period_requests if r.status == DSARStatus.COMPLETED]
            on_time = [r for r in completed if r.completed_at and r.completed_at <= r.effective_deadline]
            on_time_rate = len(on_time) / len(completed) if completed else 0.0

            return DSARMetrics(
                total_requests=len(period_requests),
                by_type=by_type,
                by_status=by_status,
                average_completion_days=(
                    sum(completion_days) / len(completion_days)
                    if completion_days else 0.0
                ),
                overdue_count=overdue_count,
                pending_count=pending_count,
                completed_on_time_rate=on_time_rate,
                average_records_per_request=(
                    total_records_processed / len(period_requests)
                    if period_requests else 0.0
                ),
                total_records_processed=total_records_processed,
                total_records_deleted=total_records_deleted,
                period_start=cutoff,
                period_end=datetime.now(timezone.utc),
            )

    # ========================================================================
    # Audit
    # ========================================================================

    def _log_audit(
        self,
        action: AuditAction,
        request: DSARRequest,
        actor_id: Optional[str] = None,
        actor_type: str = "system",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an audit entry."""
        entry = AuditEntry(
            action=action,
            request_id=request.id,
            user_id=request.user_id,
            workspace_id=request.workspace_id,
            actor_id=actor_id,
            actor_type=actor_type,
            details=details or {},
        )

        self._audit_log.append(entry)

        if self._audit_callback:
            try:
                self._audit_callback(entry)
            except Exception as e:
                logger.error(f"Audit callback error: {e}")

    def get_audit_log(
        self,
        request_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """Get audit log with optional filters."""
        with self._lock:
            entries = list(self._audit_log)

            if request_id:
                entries = [e for e in entries if e.request_id == request_id]

            if workspace_id:
                entries = [e for e in entries if e.workspace_id == workspace_id]

            if action:
                entries = [e for e in entries if e.action == action]

            return entries[-limit:]

    def get_audit_trail(self, request_id: str) -> List[Dict[str, Any]]:
        """Get complete audit trail for a request."""
        entries = self.get_audit_log(request_id=request_id, limit=1000)
        return [e.to_dict() for e in sorted(entries, key=lambda e: e.timestamp)]


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Constants
    "DSAR_RESPONSE_DEADLINE_DAYS",
    "DSAR_EXTENSION_DAYS",
    "DSAR_MAX_TOTAL_DAYS",
    "DSAR_DATA_CATEGORIES",
    "DSAR_OUT_OF_SCOPE_CATEGORIES",
    "CCEA_BOUNDARY_NOTICE",
    "ERASURE_EXEMPTIONS",
    "COMPLIANCE_DATA_CATEGORIES",
    # Enums
    "DSARRequestType",
    "DSARStatus",
    "VerificationMethod",
    "VerificationStatus",
    "ExportFormat",
    "AuditAction",
    # Data classes
    "VerificationToken",
    "DSARRequest",
    "DSARResult",
    "DSARMetrics",
    "AuditEntry",
    "DataCategory",
    # Registry
    "DATA_CATEGORIES",
    # Service
    "DSARPhase5Service",
]
