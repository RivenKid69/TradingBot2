# -*- coding: utf-8 -*-
"""
GDPR Phase 6: Enhanced Break-Glass Access Service.

Implements comprehensive emergency access controls:
- Incident-only access
- Mandatory reason (minimum 20 characters)
- Scope-limited access
- Time-bounded (max 24 hours)
- Full audit trail
- Integration with RBAC and Access Audit

Design Doc Reference:
    - Phase 6: "Break-glass: incident-only, reason required, scope limited, time bounded"
    - Design Doc 14.4: "Break-glass with reason and auditable event"

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, Final, List, Optional, Set, TYPE_CHECKING
from uuid import uuid4

from .access_audit import (
    AccessAuditService,
    AuditAction,
    AuditEntry,
    AuditPrincipal,
    AuditResult,
)
from .rbac_service import (
    Principal,
    ResourceType,
    SensitivityLevel,
)


logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Maximum duration for break-glass access (hours)
MAX_BREAK_GLASS_DURATION_HOURS: Final[int] = 24

# Default duration (hours)
DEFAULT_BREAK_GLASS_DURATION_HOURS: Final[int] = 4

# Minimum duration (minutes)
MIN_BREAK_GLASS_DURATION_MINUTES: Final[int] = 15

# Cooldown period between requests (minutes)
BREAK_GLASS_COOLDOWN_MINUTES: Final[int] = 5

# Minimum reason length
MIN_REASON_LENGTH: Final[int] = 20

# Maximum active requests per user
MAX_ACTIVE_REQUESTS_PER_USER: Final[int] = 3

# Access token validity (same as request duration)
ACCESS_TOKEN_PREFIX: Final[str] = "bg_"


# ============================================================================
# Enums
# ============================================================================


class BreakGlassReasonType(str, Enum):
    """Pre-defined break-glass reason categories."""

    INCIDENT_RESPONSE = "incident_response"
    SECURITY_INVESTIGATION = "security_investigation"
    COMPLIANCE_AUDIT = "compliance_audit"
    DATA_RECOVERY = "data_recovery"
    SYSTEM_FAILURE = "system_failure"
    CUSTOMER_EMERGENCY = "customer_emergency"
    PRODUCTION_DEBUGGING = "production_debugging"
    REGULATORY_REQUEST = "regulatory_request"
    OTHER = "other"


class BreakGlassScope(str, Enum):
    """Scope of break-glass access."""

    TELEMETRY_READ = "telemetry_read"
    TELEMETRY_RAW_READ = "telemetry_raw_read"  # RAW_ORDER_EVENTS
    AUDIT_READ = "audit_read"
    CONFIG_READ = "config_read"
    CONFIG_WRITE = "config_write"
    AGENT_READ = "agent_read"
    AGENT_ADMIN = "agent_admin"
    DEPLOYMENT_READ = "deployment_read"
    DEPLOYMENT_ADMIN = "deployment_admin"
    USER_READ = "user_read"
    DATA_EXPORT = "data_export"
    ADMIN_ACCESS = "admin_access"


class BreakGlassStatus(str, Enum):
    """Status of break-glass request."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


# ============================================================================
# Scope Permissions Mapping
# ============================================================================

# Map break-glass scopes to RBAC resource:action pairs
SCOPE_PERMISSIONS: Dict[BreakGlassScope, List[tuple]] = {
    BreakGlassScope.TELEMETRY_READ: [
        (ResourceType.TELEMETRY.value, "read"),
        (ResourceType.ALERT.value, "read"),
    ],
    BreakGlassScope.TELEMETRY_RAW_READ: [
        (ResourceType.TELEMETRY.value, "read"),
        (ResourceType.TELEMETRY.value, "read_raw"),
    ],
    BreakGlassScope.AUDIT_READ: [
        (ResourceType.AUDIT_LOG.value, "read"),
        (ResourceType.ACCESS_AUDIT.value, "read"),
    ],
    BreakGlassScope.CONFIG_READ: [
        (ResourceType.CONFIG_BLOB.value, "read"),
        (ResourceType.GOVERNANCE_POLICY.value, "read"),
    ],
    BreakGlassScope.CONFIG_WRITE: [
        (ResourceType.CONFIG_BLOB.value, "read"),
        (ResourceType.CONFIG_BLOB.value, "write"),
    ],
    BreakGlassScope.AGENT_READ: [
        (ResourceType.AGENT.value, "read"),
        (ResourceType.DEPLOYMENT.value, "read"),
        (ResourceType.RUN.value, "read"),
    ],
    BreakGlassScope.AGENT_ADMIN: [
        (ResourceType.AGENT.value, "*"),
        (ResourceType.DEPLOYMENT.value, "*"),
        (ResourceType.RUN.value, "*"),
    ],
    BreakGlassScope.DEPLOYMENT_READ: [
        (ResourceType.DEPLOYMENT.value, "read"),
        (ResourceType.RUN.value, "read"),
    ],
    BreakGlassScope.DEPLOYMENT_ADMIN: [
        (ResourceType.DEPLOYMENT.value, "*"),
        (ResourceType.RUN.value, "*"),
    ],
    BreakGlassScope.USER_READ: [
        (ResourceType.USER.value, "read"),
    ],
    BreakGlassScope.DATA_EXPORT: [
        (ResourceType.TELEMETRY.value, "export"),
        (ResourceType.AUDIT_LOG.value, "export"),
    ],
    BreakGlassScope.ADMIN_ACCESS: [
        ("*", "*"),
    ],
}

# Scopes that require additional approval
ELEVATED_SCOPES: Set[BreakGlassScope] = {
    BreakGlassScope.TELEMETRY_RAW_READ,
    BreakGlassScope.CONFIG_WRITE,
    BreakGlassScope.AGENT_ADMIN,
    BreakGlassScope.DEPLOYMENT_ADMIN,
    BreakGlassScope.ADMIN_ACCESS,
}


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class BreakGlassRequest:
    """
    Break-glass access request.

    Represents a request for emergency elevated access.
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""

    # Requester
    requester_id: str = ""
    requester_email: str = ""
    requester_organization_id: str = ""

    # Request details
    reason: str = ""
    reason_type: BreakGlassReasonType = BreakGlassReasonType.OTHER
    scopes: Set[BreakGlassScope] = field(default_factory=set)
    resource_ids: List[str] = field(default_factory=list)  # Optional specific resources
    ticket_id: Optional[str] = None

    # Duration
    duration_hours: int = DEFAULT_BREAK_GLASS_DURATION_HOURS

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    approved_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None

    # Status
    status: BreakGlassStatus = BreakGlassStatus.PENDING

    # Approval
    approved_by: Optional[str] = None
    approver_email: Optional[str] = None
    approval_notes: str = ""
    denial_reason: Optional[str] = None

    # Access tracking
    access_token_hash: Optional[str] = None
    access_count: int = 0
    last_access_at: Optional[datetime] = None
    accessed_resources: List[str] = field(default_factory=list)

    # Evidence
    evidence_hash: str = ""

    def __post_init__(self):
        """Validate and compute evidence hash."""
        # Enforce duration limits
        self.duration_hours = max(
            MIN_BREAK_GLASS_DURATION_MINUTES // 60,
            min(self.duration_hours, MAX_BREAK_GLASS_DURATION_HOURS),
        )

        # Compute evidence hash
        self._compute_evidence_hash()

    def _compute_evidence_hash(self) -> None:
        """Compute evidence hash for this request."""
        evidence_data = {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "requester_id": self.requester_id,
            "reason": self.reason,
            "reason_type": self.reason_type.value,
            "scopes": sorted([s.value for s in self.scopes]),
            "created_at": self.created_at.isoformat(),
            "duration_hours": self.duration_hours,
        }
        content = json.dumps(evidence_data, sort_keys=True)
        self.evidence_hash = f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    @property
    def is_active(self) -> bool:
        """Check if request is currently active."""
        if self.status != BreakGlassStatus.ACTIVE:
            return False
        if self.revoked_at:
            return False
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        return True

    @property
    def is_pending(self) -> bool:
        """Check if request is pending approval."""
        return self.status == BreakGlassStatus.PENDING

    @property
    def is_expired(self) -> bool:
        """Check if request has expired."""
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return True
        return self.status == BreakGlassStatus.EXPIRED

    @property
    def requires_elevated_approval(self) -> bool:
        """Check if request requires additional approval."""
        return bool(self.scopes.intersection(ELEVATED_SCOPES))

    @property
    def time_remaining_seconds(self) -> int:
        """Get remaining time in seconds."""
        if not self.expires_at:
            return 0
        remaining = (self.expires_at - datetime.now(timezone.utc)).total_seconds()
        return max(0, int(remaining))

    def has_scope(self, scope: BreakGlassScope) -> bool:
        """Check if request has a specific scope."""
        return scope in self.scopes

    def has_permission(self, resource_type: str, action: str) -> bool:
        """Check if request grants permission for resource:action."""
        for scope in self.scopes:
            permissions = SCOPE_PERMISSIONS.get(scope, [])
            for perm_resource, perm_action in permissions:
                if perm_resource == "*" or perm_resource == resource_type:
                    if perm_action == "*" or perm_action == action:
                        return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "requester": {
                "id": self.requester_id,
                "email": self.requester_email,
                "organization_id": self.requester_organization_id,
            },
            "request": {
                "reason": self.reason,
                "reason_type": self.reason_type.value,
                "scopes": [s.value for s in self.scopes],
                "resource_ids": self.resource_ids,
                "ticket_id": self.ticket_id,
                "duration_hours": self.duration_hours,
            },
            "timing": {
                "created_at": self.created_at.isoformat(),
                "approved_at": self.approved_at.isoformat() if self.approved_at else None,
                "expires_at": self.expires_at.isoformat() if self.expires_at else None,
                "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
                "time_remaining_seconds": self.time_remaining_seconds,
            },
            "status": self.status.value,
            "approval": {
                "approved_by": self.approved_by,
                "approver_email": self.approver_email,
                "approval_notes": self.approval_notes,
                "denial_reason": self.denial_reason,
            },
            "access": {
                "access_count": self.access_count,
                "last_access_at": self.last_access_at.isoformat() if self.last_access_at else None,
                "accessed_resources": self.accessed_resources,
            },
            "evidence_hash": self.evidence_hash,
            "is_active": self.is_active,
            "is_expired": self.is_expired,
            "requires_elevated_approval": self.requires_elevated_approval,
        }


@dataclass
class BreakGlassResult:
    """Result of a break-glass operation."""

    success: bool = True
    request_id: str = ""
    access_token: Optional[str] = None  # Only returned on approval
    expires_at: Optional[datetime] = None
    error: Optional[str] = None
    requires_approval: bool = False
    requires_elevated_approval: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "request_id": self.request_id,
            "has_token": self.access_token is not None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "error": self.error,
            "requires_approval": self.requires_approval,
            "requires_elevated_approval": self.requires_elevated_approval,
        }


@dataclass
class BreakGlassStats:
    """Break-glass usage statistics."""

    total_requests: int = 0
    pending_count: int = 0
    active_count: int = 0
    approved_count: int = 0
    denied_count: int = 0
    revoked_count: int = 0
    expired_count: int = 0
    by_reason_type: Dict[str, int] = field(default_factory=dict)
    by_scope: Dict[str, int] = field(default_factory=dict)
    avg_duration_hours: float = 0.0
    total_access_count: int = 0
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "pending_count": self.pending_count,
            "active_count": self.active_count,
            "approved_count": self.approved_count,
            "denied_count": self.denied_count,
            "revoked_count": self.revoked_count,
            "expired_count": self.expired_count,
            "by_reason_type": self.by_reason_type,
            "by_scope": self.by_scope,
            "avg_duration_hours": self.avg_duration_hours,
            "total_access_count": self.total_access_count,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
        }


# ============================================================================
# Break-Glass Service
# ============================================================================


class BreakGlassPhase6Service:
    """
    Enhanced Break-Glass Access Service.

    Provides controlled emergency access with:
    - Mandatory approval workflow
    - Scope-based access control
    - Time-bounded access
    - Full audit trail
    - Integration with RBAC

    Usage:
        service = BreakGlassPhase6Service(
            audit_service=audit_service,
            approvers={"admin@example.com"},
        )

        # Create request
        result = service.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            workspace_id="ws-456",
            reason="Investigating production incident #12345",
            reason_type=BreakGlassReasonType.INCIDENT_RESPONSE,
            scopes={BreakGlassScope.TELEMETRY_READ},
            ticket_id="INC-12345",
        )

        # Approve (by authorized approver)
        result = service.approve_request(
            request_id=result.request_id,
            approver_id="admin-789",
            approver_email="admin@example.com",
            notes="Approved for incident response",
        )

        # Check access
        if service.has_access(result.access_token, BreakGlassScope.TELEMETRY_READ):
            # Access granted
            pass
    """

    def __init__(
        self,
        audit_service: Optional[AccessAuditService] = None,
        approvers: Optional[Set[str]] = None,
        elevated_approvers: Optional[Set[str]] = None,
        require_approval: bool = True,
        on_request: Optional[Callable[[BreakGlassRequest], None]] = None,
        on_access: Optional[Callable[[BreakGlassRequest, BreakGlassScope], None]] = None,
    ):
        """
        Initialize Break-Glass Service.

        Args:
            audit_service: Access audit service for logging
            approvers: Set of user IDs/emails who can approve requests
            elevated_approvers: Set of user IDs/emails who can approve elevated requests
            require_approval: Whether approval is required (set False only for testing)
            on_request: Callback when request is created
            on_access: Callback when access is used
        """
        self._audit_service = audit_service
        self._approvers = approvers or set()
        self._elevated_approvers = elevated_approvers or self._approvers
        self._require_approval = require_approval
        self._on_request = on_request
        self._on_access = on_access

        # Storage
        self._requests: Dict[str, BreakGlassRequest] = {}
        self._tokens: Dict[str, str] = {}  # token_hash -> request_id
        self._user_requests: Dict[str, List[str]] = {}  # user_id -> request_ids

        self._lock = threading.Lock()

    # ========================================================================
    # Request Management
    # ========================================================================

    def create_request(
        self,
        requester_id: str,
        requester_email: str,
        workspace_id: str,
        reason: str,
        reason_type: BreakGlassReasonType,
        scopes: Set[BreakGlassScope],
        requester_organization_id: str = "",
        resource_ids: Optional[List[str]] = None,
        ticket_id: Optional[str] = None,
        duration_hours: int = DEFAULT_BREAK_GLASS_DURATION_HOURS,
    ) -> BreakGlassResult:
        """
        Create a break-glass request.

        Args:
            requester_id: Who is requesting
            requester_email: Requester's email
            workspace_id: Target workspace
            reason: Detailed reason (min 20 characters)
            reason_type: Category of reason
            scopes: Access scopes needed
            requester_organization_id: Requester's organization
            resource_ids: Optional specific resources
            ticket_id: Related incident/ticket ID
            duration_hours: Duration of access

        Returns:
            BreakGlassResult

        Raises:
            ValueError: If validation fails
        """
        # Validate reason length
        if len(reason.strip()) < MIN_REASON_LENGTH:
            raise ValueError(
                f"Reason must be at least {MIN_REASON_LENGTH} characters. "
                f"Provided: {len(reason.strip())} characters"
            )

        # Validate scopes
        if not scopes:
            raise ValueError("At least one scope must be requested")

        # Check cooldown
        with self._lock:
            user_req_ids = self._user_requests.get(requester_id, [])
            for req_id in user_req_ids:
                req = self._requests.get(req_id)
                if req and not req.is_expired:
                    elapsed = (datetime.now(timezone.utc) - req.created_at).total_seconds()
                    if elapsed < BREAK_GLASS_COOLDOWN_MINUTES * 60:
                        raise ValueError(
                            f"Cooldown in effect. Wait {BREAK_GLASS_COOLDOWN_MINUTES} minutes "
                            f"between requests"
                        )

            # Check max active requests
            active_count = sum(
                1
                for req_id in user_req_ids
                if self._requests.get(req_id) and self._requests[req_id].is_active
            )
            if active_count >= MAX_ACTIVE_REQUESTS_PER_USER:
                raise ValueError(
                    f"Maximum {MAX_ACTIVE_REQUESTS_PER_USER} active break-glass requests "
                    f"per user"
                )

        # Create request
        request = BreakGlassRequest(
            workspace_id=workspace_id,
            requester_id=requester_id,
            requester_email=requester_email,
            requester_organization_id=requester_organization_id,
            reason=reason.strip(),
            reason_type=reason_type,
            scopes=scopes,
            resource_ids=resource_ids or [],
            ticket_id=ticket_id,
            duration_hours=duration_hours,
        )

        with self._lock:
            self._requests[request.id] = request

            # Track by user
            if requester_id not in self._user_requests:
                self._user_requests[requester_id] = []
            self._user_requests[requester_id].append(request.id)

        # Audit
        self._audit_request_created(request)

        # Callback
        if self._on_request:
            try:
                self._on_request(request)
            except Exception as e:
                logger.error(f"on_request callback error: {e}")

        logger.info(
            f"Break-glass request created: {request.id} by {requester_email} "
            f"for {workspace_id}, scopes={[s.value for s in scopes]}"
        )

        return BreakGlassResult(
            success=True,
            request_id=request.id,
            requires_approval=self._require_approval,
            requires_elevated_approval=request.requires_elevated_approval,
        )

    def approve_request(
        self,
        request_id: str,
        approver_id: str,
        approver_email: str,
        notes: str = "",
    ) -> BreakGlassResult:
        """
        Approve a break-glass request.

        Args:
            request_id: Request to approve
            approver_id: Approver user ID
            approver_email: Approver email
            notes: Approval notes

        Returns:
            BreakGlassResult with access token
        """
        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return BreakGlassResult(
                    success=False,
                    request_id=request_id,
                    error="Request not found",
                )

            # Check if already processed
            if request.status != BreakGlassStatus.PENDING:
                return BreakGlassResult(
                    success=False,
                    request_id=request_id,
                    error=f"Request already {request.status.value}",
                )

            # Check authorization
            if self._require_approval:
                # Self-approval not allowed
                if approver_id == request.requester_id:
                    return BreakGlassResult(
                        success=False,
                        request_id=request_id,
                        error="Self-approval not allowed",
                    )

                # Check if approver is authorized
                is_authorized = approver_id in self._approvers or approver_email in self._approvers

                # Elevated scopes require elevated approvers
                if request.requires_elevated_approval:
                    is_authorized = (
                        approver_id in self._elevated_approvers
                        or approver_email in self._elevated_approvers
                    )
                    if not is_authorized:
                        return BreakGlassResult(
                            success=False,
                            request_id=request_id,
                            error="Elevated approval required for requested scopes",
                        )

                if not is_authorized:
                    return BreakGlassResult(
                        success=False,
                        request_id=request_id,
                        error="Not authorized to approve",
                    )

            # Approve
            now = datetime.now(timezone.utc)
            request.status = BreakGlassStatus.ACTIVE
            request.approved_by = approver_id
            request.approver_email = approver_email
            request.approved_at = now
            request.approval_notes = notes
            request.expires_at = now + timedelta(hours=request.duration_hours)

            # Generate access token
            token = f"{ACCESS_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            request.access_token_hash = token_hash
            self._tokens[token_hash] = request_id

        # Audit
        self._audit_request_approved(request, approver_id, approver_email)

        logger.info(
            f"Break-glass request approved: {request_id} by {approver_email}, "
            f"expires_at={request.expires_at.isoformat()}"
        )

        return BreakGlassResult(
            success=True,
            request_id=request_id,
            access_token=token,
            expires_at=request.expires_at,
        )

    def deny_request(
        self,
        request_id: str,
        denier_id: str,
        denier_email: str,
        reason: str,
    ) -> BreakGlassResult:
        """
        Deny a break-glass request.

        Args:
            request_id: Request to deny
            denier_id: User denying
            denier_email: Denier email
            reason: Denial reason

        Returns:
            BreakGlassResult
        """
        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return BreakGlassResult(
                    success=False,
                    request_id=request_id,
                    error="Request not found",
                )

            if request.status != BreakGlassStatus.PENDING:
                return BreakGlassResult(
                    success=False,
                    request_id=request_id,
                    error=f"Request already {request.status.value}",
                )

            request.status = BreakGlassStatus.DENIED
            request.denial_reason = reason

        # Audit
        self._audit_request_denied(request, denier_id, denier_email, reason)

        logger.info(f"Break-glass request denied: {request_id} by {denier_email}")

        return BreakGlassResult(
            success=True,
            request_id=request_id,
        )

    def revoke_request(
        self,
        request_id: str,
        revoker_id: str,
        revoker_email: str,
        reason: str = "",
    ) -> BreakGlassResult:
        """
        Revoke an active break-glass request.

        Args:
            request_id: Request to revoke
            revoker_id: User revoking
            revoker_email: Revoker email
            reason: Revocation reason

        Returns:
            BreakGlassResult
        """
        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return BreakGlassResult(
                    success=False,
                    request_id=request_id,
                    error="Request not found",
                )

            if not request.is_active:
                return BreakGlassResult(
                    success=False,
                    request_id=request_id,
                    error="Request is not active",
                )

            request.status = BreakGlassStatus.REVOKED
            request.revoked_at = datetime.now(timezone.utc)

            # Invalidate token
            if request.access_token_hash:
                self._tokens.pop(request.access_token_hash, None)

        # Audit
        self._audit_request_revoked(request, revoker_id, revoker_email, reason)

        logger.info(f"Break-glass request revoked: {request_id} by {revoker_email}")

        return BreakGlassResult(
            success=True,
            request_id=request_id,
        )

    def cancel_request(
        self,
        request_id: str,
        canceller_id: str,
    ) -> BreakGlassResult:
        """
        Cancel a pending break-glass request (by requester).

        Args:
            request_id: Request to cancel
            canceller_id: User cancelling

        Returns:
            BreakGlassResult
        """
        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return BreakGlassResult(
                    success=False,
                    request_id=request_id,
                    error="Request not found",
                )

            # Only requester can cancel
            if request.requester_id != canceller_id:
                return BreakGlassResult(
                    success=False,
                    request_id=request_id,
                    error="Only requester can cancel",
                )

            if request.status != BreakGlassStatus.PENDING:
                return BreakGlassResult(
                    success=False,
                    request_id=request_id,
                    error=f"Cannot cancel: request is {request.status.value}",
                )

            request.status = BreakGlassStatus.CANCELLED

        logger.info(f"Break-glass request cancelled: {request_id}")

        return BreakGlassResult(
            success=True,
            request_id=request_id,
        )

    # ========================================================================
    # Access Control
    # ========================================================================

    def has_access(
        self,
        token: str,
        scope: BreakGlassScope,
        resource_id: Optional[str] = None,
    ) -> bool:
        """
        Check if token has access to scope.

        Args:
            token: Access token
            scope: Required scope
            resource_id: Optional specific resource

        Returns:
            True if access granted
        """
        request = self.validate_token(token)
        if not request:
            return False

        if not request.is_active:
            return False

        if not request.has_scope(scope):
            return False

        # Check specific resource if specified
        if resource_id and request.resource_ids:
            if resource_id not in request.resource_ids:
                return False

        # Record access
        with self._lock:
            request.access_count += 1
            request.last_access_at = datetime.now(timezone.utc)
            if resource_id and resource_id not in request.accessed_resources:
                request.accessed_resources.append(resource_id)

        # Audit
        self._audit_access_used(request, scope, resource_id)

        # Callback
        if self._on_access:
            try:
                self._on_access(request, scope)
            except Exception as e:
                logger.error(f"on_access callback error: {e}")

        return True

    def has_permission(
        self,
        token: str,
        resource_type: str,
        action: str,
    ) -> bool:
        """
        Check if token grants permission for resource:action.

        Args:
            token: Access token
            resource_type: Resource type
            action: Action

        Returns:
            True if permission granted
        """
        request = self.validate_token(token)
        if not request or not request.is_active:
            return False
        return request.has_permission(resource_type, action)

    def validate_token(self, token: str) -> Optional[BreakGlassRequest]:
        """
        Validate access token and return request.

        Args:
            token: Access token

        Returns:
            BreakGlassRequest if valid, None otherwise
        """
        if not token or not token.startswith(ACCESS_TOKEN_PREFIX):
            return None

        token_hash = hashlib.sha256(token.encode()).hexdigest()

        with self._lock:
            request_id = self._tokens.get(token_hash)
            if not request_id:
                return None

            request = self._requests.get(request_id)
            if not request:
                return None

            # Check expiry
            if request.is_expired:
                request.status = BreakGlassStatus.EXPIRED
                self._tokens.pop(token_hash, None)
                return None

            return request

    def get_request_for_principal(self, principal_id: str) -> Optional[BreakGlassRequest]:
        """Get active break-glass request for a principal."""
        with self._lock:
            request_ids = self._user_requests.get(principal_id, [])
            for request_id in reversed(request_ids):  # Most recent first
                request = self._requests.get(request_id)
                if request and request.is_active:
                    return request
            return None

    # ========================================================================
    # Queries
    # ========================================================================

    def get_request(self, request_id: str) -> Optional[BreakGlassRequest]:
        """Get request by ID."""
        with self._lock:
            return self._requests.get(request_id)

    def get_pending_requests(
        self,
        workspace_id: Optional[str] = None,
    ) -> List[BreakGlassRequest]:
        """Get pending requests awaiting approval."""
        with self._lock:
            pending = [r for r in self._requests.values() if r.status == BreakGlassStatus.PENDING]
            if workspace_id:
                pending = [r for r in pending if r.workspace_id == workspace_id]
            return sorted(pending, key=lambda r: r.created_at, reverse=True)

    def get_active_requests(
        self,
        workspace_id: Optional[str] = None,
    ) -> List[BreakGlassRequest]:
        """Get active break-glass requests."""
        with self._lock:
            active = [r for r in self._requests.values() if r.is_active]
            if workspace_id:
                active = [r for r in active if r.workspace_id == workspace_id]
            return sorted(active, key=lambda r: r.expires_at or r.created_at)

    def get_requests_by_user(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List[BreakGlassRequest]:
        """Get requests by user."""
        with self._lock:
            request_ids = self._user_requests.get(user_id, [])
            requests = [self._requests[rid] for rid in request_ids if rid in self._requests]
            return sorted(requests, key=lambda r: r.created_at, reverse=True)[:limit]

    # ========================================================================
    # Maintenance
    # ========================================================================

    def cleanup_expired(self) -> int:
        """
        Clean up expired requests and tokens.

        Returns:
            Number of expired items cleaned up
        """
        count = 0
        now = datetime.now(timezone.utc)

        with self._lock:
            # Find expired active requests
            for request in self._requests.values():
                if (
                    request.status == BreakGlassStatus.ACTIVE
                    and request.expires_at
                    and now > request.expires_at
                ):
                    request.status = BreakGlassStatus.EXPIRED
                    if request.access_token_hash:
                        self._tokens.pop(request.access_token_hash, None)
                    count += 1

                    # Audit
                    self._audit_request_expired(request)

        if count > 0:
            logger.info(f"Cleaned up {count} expired break-glass requests")

        return count

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_stats(
        self,
        workspace_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> BreakGlassStats:
        """
        Get break-glass usage statistics.

        Args:
            workspace_id: Optional workspace filter
            start_time: Start of period
            end_time: End of period

        Returns:
            BreakGlassStats
        """
        with self._lock:
            requests = list(self._requests.values())

            if workspace_id:
                requests = [r for r in requests if r.workspace_id == workspace_id]

            if start_time:
                requests = [r for r in requests if r.created_at >= start_time]

            if end_time:
                requests = [r for r in requests if r.created_at <= end_time]

            stats = BreakGlassStats(
                total_requests=len(requests),
                period_start=start_time,
                period_end=end_time or datetime.now(timezone.utc),
            )

            total_duration = 0
            approved_count = 0

            for r in requests:
                # By status
                if r.status == BreakGlassStatus.PENDING:
                    stats.pending_count += 1
                elif r.is_active:
                    stats.active_count += 1
                elif r.status == BreakGlassStatus.APPROVED or r.approved_at:
                    stats.approved_count += 1
                    approved_count += 1
                    total_duration += r.duration_hours
                elif r.status == BreakGlassStatus.DENIED:
                    stats.denied_count += 1
                elif r.status == BreakGlassStatus.REVOKED:
                    stats.revoked_count += 1
                elif r.status == BreakGlassStatus.EXPIRED:
                    stats.expired_count += 1

                # By reason type
                reason_key = r.reason_type.value
                stats.by_reason_type[reason_key] = stats.by_reason_type.get(reason_key, 0) + 1

                # By scope
                for scope in r.scopes:
                    scope_key = scope.value
                    stats.by_scope[scope_key] = stats.by_scope.get(scope_key, 0) + 1

                # Total access
                stats.total_access_count += r.access_count

            if approved_count > 0:
                stats.avg_duration_hours = total_duration / approved_count

            return stats

    # ========================================================================
    # Export
    # ========================================================================

    def export_requests(
        self,
        workspace_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Export break-glass requests for evidence pack."""
        with self._lock:
            requests = [r for r in self._requests.values() if r.workspace_id == workspace_id]

            if start_time:
                requests = [r for r in requests if r.created_at >= start_time]

            if end_time:
                requests = [r for r in requests if r.created_at <= end_time]

            # Sort by created_at
            requests.sort(key=lambda r: r.created_at)

            return [r.to_dict() for r in requests]

    # ========================================================================
    # Audit Logging
    # ========================================================================

    def _audit_request_created(self, request: BreakGlassRequest) -> None:
        """Audit break-glass request creation."""
        if not self._audit_service:
            return

        self._audit_service.log_access(
            workspace_id=request.workspace_id,
            principal=AuditPrincipal(
                id=request.requester_id,
                type="user",
                email=request.requester_email,
                organization_id=request.requester_organization_id,
            ),
            action=AuditAction.BREAK_GLASS_REQUEST,
            resource_type=ResourceType.BREAK_GLASS.value,
            resource_id=request.id,
            result=AuditResult.SUCCESS,
            sensitivity_level=SensitivityLevel.CRITICAL,
            reason=request.reason,
            ticket_id=request.ticket_id,
            context={
                "reason_type": request.reason_type.value,
                "scopes": [s.value for s in request.scopes],
                "duration_hours": request.duration_hours,
                "evidence_hash": request.evidence_hash,
            },
        )

    def _audit_request_approved(
        self,
        request: BreakGlassRequest,
        approver_id: str,
        approver_email: str,
    ) -> None:
        """Audit break-glass request approval."""
        if not self._audit_service:
            return

        self._audit_service.log_access(
            workspace_id=request.workspace_id,
            principal=AuditPrincipal(
                id=approver_id,
                type="user",
                email=approver_email,
            ),
            action=AuditAction.BREAK_GLASS_APPROVE,
            resource_type=ResourceType.BREAK_GLASS.value,
            resource_id=request.id,
            result=AuditResult.SUCCESS,
            sensitivity_level=SensitivityLevel.CRITICAL,
            context={
                "requester_id": request.requester_id,
                "scopes": [s.value for s in request.scopes],
                "expires_at": request.expires_at.isoformat() if request.expires_at else None,
                "approval_notes": request.approval_notes,
            },
        )

    def _audit_request_denied(
        self,
        request: BreakGlassRequest,
        denier_id: str,
        denier_email: str,
        reason: str,
    ) -> None:
        """Audit break-glass request denial."""
        if not self._audit_service:
            return

        self._audit_service.log_access(
            workspace_id=request.workspace_id,
            principal=AuditPrincipal(
                id=denier_id,
                type="user",
                email=denier_email,
            ),
            action=AuditAction.BREAK_GLASS_DENY,
            resource_type=ResourceType.BREAK_GLASS.value,
            resource_id=request.id,
            result=AuditResult.SUCCESS,
            sensitivity_level=SensitivityLevel.CRITICAL,
            context={
                "requester_id": request.requester_id,
                "denial_reason": reason,
            },
        )

    def _audit_request_revoked(
        self,
        request: BreakGlassRequest,
        revoker_id: str,
        revoker_email: str,
        reason: str,
    ) -> None:
        """Audit break-glass request revocation."""
        if not self._audit_service:
            return

        self._audit_service.log_access(
            workspace_id=request.workspace_id,
            principal=AuditPrincipal(
                id=revoker_id,
                type="user",
                email=revoker_email,
            ),
            action=AuditAction.BREAK_GLASS_REVOKE,
            resource_type=ResourceType.BREAK_GLASS.value,
            resource_id=request.id,
            result=AuditResult.SUCCESS,
            sensitivity_level=SensitivityLevel.CRITICAL,
            context={
                "requester_id": request.requester_id,
                "revocation_reason": reason,
                "access_count": request.access_count,
            },
        )

    def _audit_request_expired(self, request: BreakGlassRequest) -> None:
        """Audit break-glass request expiration."""
        if not self._audit_service:
            return

        self._audit_service.log_access(
            workspace_id=request.workspace_id,
            principal=AuditPrincipal(id="system", type="system"),
            action=AuditAction.BREAK_GLASS_EXPIRE,
            resource_type=ResourceType.BREAK_GLASS.value,
            resource_id=request.id,
            result=AuditResult.SUCCESS,
            sensitivity_level=SensitivityLevel.CRITICAL,
            context={
                "requester_id": request.requester_id,
                "access_count": request.access_count,
                "duration_hours": request.duration_hours,
            },
        )

    def _audit_access_used(
        self,
        request: BreakGlassRequest,
        scope: BreakGlassScope,
        resource_id: Optional[str],
    ) -> None:
        """Audit break-glass access usage."""
        if not self._audit_service:
            return

        self._audit_service.log_access(
            workspace_id=request.workspace_id,
            principal=AuditPrincipal(
                id=request.requester_id,
                type="user",
                email=request.requester_email,
            ),
            action=AuditAction.BREAK_GLASS_ACCESS,
            resource_type=scope.value,
            resource_id=resource_id,
            result=AuditResult.SUCCESS,
            authorization_method="break_glass",
            break_glass_id=request.id,
            sensitivity_level=SensitivityLevel.CRITICAL,
            reason=request.reason,
            ticket_id=request.ticket_id,
            context={
                "access_count": request.access_count,
                "time_remaining_seconds": request.time_remaining_seconds,
            },
        )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Constants
    "MAX_BREAK_GLASS_DURATION_HOURS",
    "DEFAULT_BREAK_GLASS_DURATION_HOURS",
    "MIN_REASON_LENGTH",
    "BREAK_GLASS_COOLDOWN_MINUTES",
    # Enums
    "BreakGlassReasonType",
    "BreakGlassScope",
    "BreakGlassStatus",
    # Data classes
    "BreakGlassRequest",
    "BreakGlassResult",
    "BreakGlassStats",
    # Service
    "BreakGlassPhase6Service",
]
