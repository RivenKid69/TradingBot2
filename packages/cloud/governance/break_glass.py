# -*- coding: utf-8 -*-
"""
Break Glass Access Controller.

CCEA Phase 8 Implementation.

This module provides emergency access controls:
    - Break-glass access for emergencies
    - Mandatory reason and audit
    - Time-limited elevated privileges
    - Full audit trail

Design Doc Reference:
    - Phase 8 (13.3): "Break-glass only with reason and audit event"
    - Emergency access procedures

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, Final, List, Optional, Set
from uuid import uuid4


# ============================================================================
# Constants
# ============================================================================

# Maximum duration for break-glass access (hours)
MAX_BREAK_GLASS_DURATION_HOURS: Final[int] = 24

# Default duration
DEFAULT_BREAK_GLASS_DURATION_HOURS: Final[int] = 4

# Cool-down period between requests (minutes)
BREAK_GLASS_COOLDOWN_MINUTES: Final[int] = 5

# Minimum reason length
MIN_REASON_LENGTH: Final[int] = 20


class BreakGlassReason(Enum):
    """Pre-defined break-glass reasons."""

    INCIDENT_RESPONSE = "incident_response"
    SECURITY_INVESTIGATION = "security_investigation"
    COMPLIANCE_AUDIT = "compliance_audit"
    DATA_RECOVERY = "data_recovery"
    SYSTEM_FAILURE = "system_failure"
    CUSTOMER_EMERGENCY = "customer_emergency"
    OTHER = "other"


class BreakGlassScope(Enum):
    """Scope of break-glass access."""

    TELEMETRY_READ = auto()  # Read telemetry data
    AUDIT_READ = auto()  # Read audit logs
    CONFIG_READ = auto()  # Read configuration
    ADMIN_ACCESS = auto()  # Full admin access
    DATA_EXPORT = auto()  # Export data


@dataclass
class BreakGlassRequest:
    """
    Break-glass access request.

    Attributes:
        id: Request identifier
        requester_id: Who is requesting
        reason: Detailed reason
        reason_type: Pre-defined reason category
        scope: What access is needed
        workspace_id: Target workspace
        duration_hours: How long access is needed
        approved: Whether request was approved
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    requester_id: str = ""
    requester_email: str = ""
    reason: str = ""
    reason_type: BreakGlassReason = BreakGlassReason.OTHER
    scope: Set[BreakGlassScope] = field(default_factory=lambda: {BreakGlassScope.TELEMETRY_READ})
    workspace_id: str = ""

    # Timing
    duration_hours: int = DEFAULT_BREAK_GLASS_DURATION_HOURS
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    approved_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None

    # Approval
    approved: bool = False
    approved_by: Optional[str] = None
    approval_notes: str = ""

    # Tracking
    access_count: int = 0
    last_access_at: Optional[datetime] = None
    evidence_hash: str = ""

    def __post_init__(self):
        """Validate request."""
        # Enforce duration limits
        if self.duration_hours > MAX_BREAK_GLASS_DURATION_HOURS:
            self.duration_hours = MAX_BREAK_GLASS_DURATION_HOURS

        # Generate evidence hash
        evidence = f"{self.id}:{self.requester_id}:{self.reason}:{self.created_at.isoformat()}"
        self.evidence_hash = hashlib.sha256(evidence.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "requester_id": self.requester_id,
            "requester_email": self.requester_email,
            "reason": self.reason,
            "reason_type": self.reason_type.value,
            "scope": [s.name for s in self.scope],
            "workspace_id": self.workspace_id,
            "duration_hours": self.duration_hours,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "approved": self.approved,
            "approved_by": self.approved_by,
            "access_count": self.access_count,
            "last_access_at": self.last_access_at.isoformat() if self.last_access_at else None,
            "evidence_hash": self.evidence_hash,
        }

    @property
    def is_active(self) -> bool:
        """Check if request is currently active."""
        if not self.approved or self.revoked_at:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True


@dataclass
class BreakGlassResult:
    """Result of break-glass operation."""

    success: bool = True
    request_id: str = ""
    access_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "request_id": self.request_id,
            "has_token": self.access_token is not None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "error": self.error,
        }


class BreakGlassController:
    """
    Break-glass access controller.

    Provides controlled emergency access with full audit trail.

    Usage:
        controller = BreakGlassController(approvers=["admin@example.com"])

        # Create request
        request = controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="Investigating production incident #12345",
            reason_type=BreakGlassReason.INCIDENT_RESPONSE,
            workspace_id="workspace-456",
            scope={BreakGlassScope.TELEMETRY_READ},
        )

        # Approve request
        result = controller.approve_request(
            request_id=request.id,
            approved_by="admin@example.com",
        )

        # Check access
        if controller.has_access(request.id, BreakGlassScope.TELEMETRY_READ):
            access_data()

    SECURITY:
        - All requests logged with evidence hash
        - Time-limited access
        - Mandatory reason with minimum length
        - Approval required
    """

    def __init__(
        self,
        approvers: Optional[Set[str]] = None,
        require_approval: bool = True,
        on_access: Optional[Callable[[BreakGlassRequest, BreakGlassScope], None]] = None,
    ):
        """
        Initialize controller.

        Args:
            approvers: Set of users who can approve requests
            require_approval: Whether requests need approval
            on_access: Callback when access is used
        """
        self._approvers = approvers or set()
        self._require_approval = require_approval
        self._on_access = on_access
        self._requests: Dict[str, BreakGlassRequest] = {}
        self._tokens: Dict[str, str] = {}  # token -> request_id
        self._audit_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def create_request(
        self,
        requester_id: str,
        requester_email: str,
        reason: str,
        reason_type: BreakGlassReason,
        workspace_id: str,
        scope: Set[BreakGlassScope],
        duration_hours: int = DEFAULT_BREAK_GLASS_DURATION_HOURS,
    ) -> BreakGlassRequest:
        """
        Create a break-glass request.

        Args:
            requester_id: Who is requesting
            requester_email: Requester's email
            reason: Detailed reason
            reason_type: Category of reason
            workspace_id: Target workspace
            scope: Access scope needed
            duration_hours: Duration of access

        Returns:
            Created request

        Raises:
            ValueError: If reason is too short or cooldown in effect
        """
        # Validate reason length
        if len(reason) < MIN_REASON_LENGTH:
            raise ValueError(f"Reason must be at least {MIN_REASON_LENGTH} characters")

        # Check cooldown
        with self._lock:
            recent = [
                r
                for r in self._requests.values()
                if r.requester_id == requester_id
                and (datetime.utcnow() - r.created_at).total_seconds()
                < BREAK_GLASS_COOLDOWN_MINUTES * 60
            ]
            if recent:
                raise ValueError(
                    f"Cooldown in effect. Wait {BREAK_GLASS_COOLDOWN_MINUTES} minutes between requests"
                )

        request = BreakGlassRequest(
            requester_id=requester_id,
            requester_email=requester_email,
            reason=reason,
            reason_type=reason_type,
            workspace_id=workspace_id,
            scope=scope,
            duration_hours=duration_hours,
        )

        with self._lock:
            self._requests[request.id] = request
            self._log_audit("request_created", request)

        return request

    def approve_request(
        self,
        request_id: str,
        approved_by: str,
        notes: str = "",
    ) -> BreakGlassResult:
        """
        Approve a break-glass request.

        Args:
            request_id: Request to approve
            approved_by: Approver identifier
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

            # Check if approver is authorized
            if self._approvers and approved_by not in self._approvers:
                return BreakGlassResult(
                    success=False,
                    request_id=request_id,
                    error="Not authorized to approve",
                )

            # Self-approval not allowed
            if approved_by == request.requester_id:
                return BreakGlassResult(
                    success=False,
                    request_id=request_id,
                    error="Self-approval not allowed",
                )

            # Approve
            request.approved = True
            request.approved_by = approved_by
            request.approved_at = datetime.utcnow()
            request.approval_notes = notes
            request.expires_at = datetime.utcnow() + timedelta(hours=request.duration_hours)

            # Generate access token
            token = secrets.token_urlsafe(32)
            self._tokens[token] = request_id

            self._log_audit("request_approved", request, {"approved_by": approved_by})

            return BreakGlassResult(
                success=True,
                request_id=request_id,
                access_token=token,
                expires_at=request.expires_at,
            )

    def auto_approve_request(self, request_id: str) -> BreakGlassResult:
        """
        Auto-approve request (when approval not required).

        Args:
            request_id: Request to approve

        Returns:
            BreakGlassResult
        """
        if self._require_approval:
            return BreakGlassResult(
                success=False,
                request_id=request_id,
                error="Manual approval required",
            )

        return self.approve_request(request_id, approved_by="system_auto")

    def revoke_request(
        self,
        request_id: str,
        revoked_by: str,
        reason: str = "",
    ) -> bool:
        """
        Revoke an active break-glass request.

        Args:
            request_id: Request to revoke
            revoked_by: Who is revoking
            reason: Reason for revocation

        Returns:
            True if revoked
        """
        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return False

            request.revoked_at = datetime.utcnow()

            # Invalidate token
            tokens_to_remove = [t for t, r in self._tokens.items() if r == request_id]
            for token in tokens_to_remove:
                del self._tokens[token]

            self._log_audit(
                "request_revoked",
                request,
                {
                    "revoked_by": revoked_by,
                    "reason": reason,
                },
            )

            return True

    def has_access(
        self,
        request_id: str,
        scope: BreakGlassScope,
    ) -> bool:
        """
        Check if request has access to scope.

        Args:
            request_id: Request identifier
            scope: Access scope to check

        Returns:
            True if access granted
        """
        with self._lock:
            request = self._requests.get(request_id)
            if not request or not request.is_active:
                return False

            if scope not in request.scope:
                return False

            # Record access
            request.access_count += 1
            request.last_access_at = datetime.utcnow()

            self._log_audit("access_used", request, {"scope": scope.name})

            if self._on_access:
                self._on_access(request, scope)

            return True

    def validate_token(self, token: str) -> Optional[str]:
        """
        Validate access token and return request ID.

        Args:
            token: Access token to validate

        Returns:
            Request ID if valid, None otherwise
        """
        with self._lock:
            request_id = self._tokens.get(token)
            if not request_id:
                return None

            request = self._requests.get(request_id)
            if not request or not request.is_active:
                return None

            return request_id

    def get_request(self, request_id: str) -> Optional[BreakGlassRequest]:
        """Get request by ID."""
        with self._lock:
            return self._requests.get(request_id)

    def get_active_requests(self, workspace_id: Optional[str] = None) -> List[BreakGlassRequest]:
        """Get all active requests."""
        with self._lock:
            active = [r for r in self._requests.values() if r.is_active]
            if workspace_id:
                active = [r for r in active if r.workspace_id == workspace_id]
            return active

    def get_pending_requests(self) -> List[BreakGlassRequest]:
        """Get requests awaiting approval."""
        with self._lock:
            return [r for r in self._requests.values() if not r.approved and not r.revoked_at]

    def cleanup_expired(self) -> int:
        """
        Clean up expired tokens.

        Returns:
            Number of tokens cleaned up
        """
        with self._lock:
            expired_tokens = []
            for token, request_id in self._tokens.items():
                request = self._requests.get(request_id)
                if not request or not request.is_active:
                    expired_tokens.append(token)

            for token in expired_tokens:
                del self._tokens[token]

            return len(expired_tokens)

    def _log_audit(
        self,
        action: str,
        request: BreakGlassRequest,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log audit event."""
        entry = {
            "action": action,
            "request_id": request.id,
            "requester_id": request.requester_id,
            "workspace_id": request.workspace_id,
            "reason_type": request.reason_type.value,
            "evidence_hash": request.evidence_hash,
            "timestamp": datetime.utcnow().isoformat(),
            **(extra or {}),
        }
        self._audit_log.append(entry)

    def get_audit_log(
        self,
        request_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get audit log."""
        with self._lock:
            log = list(self._audit_log)

            if request_id:
                log = [e for e in log if e.get("request_id") == request_id]
            if workspace_id:
                log = [e for e in log if e.get("workspace_id") == workspace_id]

            return log
