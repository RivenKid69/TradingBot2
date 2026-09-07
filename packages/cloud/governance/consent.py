# -*- coding: utf-8 -*-
"""
Support Data Access Consent Service.

GDPR Phase 1 Implementation.

This service handles support-with-consent requirements:
    - Explicit consent for support data access
    - Time-limited consent with expiry
    - Immediate revocation capability
    - Complete audit trail

Design Doc Reference:
    - Phase 1: "Transparency + legal artifacts aligned to CCEA"
    - Support-with-consent enforcement

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, Final, List, Optional, Set
from uuid import UUID, uuid4


# ============================================================================
# Constants
# ============================================================================

# Default consent duration (hours)
DEFAULT_CONSENT_DURATION_HOURS: Final[int] = 72

# Maximum consent duration (days)
MAX_CONSENT_DURATION_DAYS: Final[int] = 30

# Consent request expiry (days) - how long a pending request stays valid
CONSENT_REQUEST_EXPIRY_DAYS: Final[int] = 7


class ConsentType(str, Enum):
    """Types of support consent."""

    SUPPORT_DATA_ACCESS = "support_data_access"
    SUPPORT_DATA_EXPORT = "support_data_export"
    SUPPORT_LOG_ACCESS = "support_log_access"
    RAW_TELEMETRY_ACCESS = "raw_telemetry_access"  # Enterprise only


class ConsentStatus(str, Enum):
    """Consent record status."""

    PENDING = "pending"  # Awaiting customer approval
    ACTIVE = "active"  # Consent granted and valid
    EXPIRED = "expired"  # Consent expired naturally
    REVOKED = "revoked"  # Customer revoked consent
    DENIED = "denied"  # Customer denied request


class ConsentScope(str, Enum):
    """Data scope for consent."""

    TELEMETRY = "telemetry"  # Telemetry data access
    LOGS = "logs"  # Application logs
    CONFIG = "config"  # Workspace configuration
    STRATEGY_METADATA = "strategy_metadata"  # Strategy names, versions (not code)
    COMMANDS = "commands"  # Command history
    AUDIT = "audit"  # Audit logs (where user is subject)
    FULL = "full"  # All above combined


# Valid scopes per consent type
VALID_SCOPES: Dict[ConsentType, Set[ConsentScope]] = {
    ConsentType.SUPPORT_DATA_ACCESS: {
        ConsentScope.TELEMETRY,
        ConsentScope.LOGS,
        ConsentScope.CONFIG,
        ConsentScope.STRATEGY_METADATA,
        ConsentScope.COMMANDS,
        ConsentScope.AUDIT,
        ConsentScope.FULL,
    },
    ConsentType.SUPPORT_DATA_EXPORT: {
        ConsentScope.TELEMETRY,
        ConsentScope.LOGS,
        ConsentScope.CONFIG,
        ConsentScope.FULL,
    },
    ConsentType.SUPPORT_LOG_ACCESS: {
        ConsentScope.LOGS,
    },
    ConsentType.RAW_TELEMETRY_ACCESS: {
        ConsentScope.TELEMETRY,  # RAW telemetry requires enterprise + explicit consent
    },
}


@dataclass
class ConsentRequest:
    """
    Request for support data access consent.

    Created when support agent needs data access.
    Awaits customer approval before becoming active.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    consent_type: ConsentType = ConsentType.SUPPORT_DATA_ACCESS

    # Requestor (support agent)
    support_agent_id: str = ""
    support_ticket_id: str = ""

    # Customer context
    user_id: str = ""  # Customer user who will approve
    workspace_id: str = ""  # Workspace scope
    organization_id: str = ""  # Organization context

    # Scope
    scopes: Set[ConsentScope] = field(default_factory=lambda: {ConsentScope.FULL})
    purpose: str = ""  # Why access is needed

    # Duration
    requested_duration_hours: int = DEFAULT_CONSENT_DURATION_HOURS

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(
        default_factory=lambda: datetime.utcnow() + timedelta(days=CONSENT_REQUEST_EXPIRY_DAYS)
    )

    # Status
    status: ConsentStatus = ConsentStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "consent_type": self.consent_type.value,
            "support_agent_id": self.support_agent_id,
            "support_ticket_id": self.support_ticket_id,
            "user_id": self.user_id,
            "workspace_id": self.workspace_id,
            "organization_id": self.organization_id,
            "scopes": [s.value for s in self.scopes],
            "purpose": self.purpose,
            "requested_duration_hours": self.requested_duration_hours,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "status": self.status.value,
        }

    @property
    def is_expired(self) -> bool:
        """Check if request is expired."""
        return datetime.utcnow() > self.expires_at


@dataclass
class ConsentRecord:
    """
    Active consent record.

    Created when customer approves a consent request.
    Contains all required fields for audit compliance.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    consent_type: ConsentType = ConsentType.SUPPORT_DATA_ACCESS
    request_id: str = ""  # Original request ID

    # Customer context
    user_id: str = ""  # Customer user who granted consent
    workspace_id: str = ""  # Workspace scope
    organization_id: str = ""  # Organization context
    granted_by_email: str = ""  # Email of granting user

    # Support context
    support_agent_id: str = ""
    support_ticket_id: str = ""

    # Scope
    scopes: Set[ConsentScope] = field(default_factory=lambda: {ConsentScope.FULL})
    purpose: str = ""

    # Timing
    granted_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(
        default_factory=lambda: datetime.utcnow() + timedelta(hours=DEFAULT_CONSENT_DURATION_HOURS)
    )

    # Status
    status: ConsentStatus = ConsentStatus.ACTIVE

    # Revocation (if revoked)
    revoked_at: Optional[datetime] = None
    revoked_by: Optional[str] = None
    revocation_reason: Optional[str] = None

    # Audit trail
    access_count: int = 0  # Number of data accesses under this consent
    last_access_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "consent_type": self.consent_type.value,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "workspace_id": self.workspace_id,
            "organization_id": self.organization_id,
            "granted_by_email": self.granted_by_email,
            "support_agent_id": self.support_agent_id,
            "support_ticket_id": self.support_ticket_id,
            "scopes": [s.value for s in self.scopes],
            "purpose": self.purpose,
            "granted_at": self.granted_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "status": self.status.value,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "revoked_by": self.revoked_by,
            "revocation_reason": self.revocation_reason,
            "access_count": self.access_count,
            "last_access_at": self.last_access_at.isoformat() if self.last_access_at else None,
        }

    @property
    def is_active(self) -> bool:
        """Check if consent is currently active."""
        if self.status != ConsentStatus.ACTIVE:
            return False
        if datetime.utcnow() > self.expires_at:
            return False
        if self.revoked_at is not None:
            return False
        return True

    @property
    def is_expired(self) -> bool:
        """Check if consent has expired."""
        return datetime.utcnow() > self.expires_at


@dataclass
class ConsentVerificationResult:
    """Result of consent verification check."""

    has_consent: bool = False
    consent_id: Optional[str] = None
    consent_type: Optional[ConsentType] = None
    scopes: Set[ConsentScope] = field(default_factory=set)
    expires_at: Optional[datetime] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "has_consent": self.has_consent,
            "consent_id": self.consent_id,
            "consent_type": self.consent_type.value if self.consent_type else None,
            "scopes": [s.value for s in self.scopes],
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "error": self.error,
        }


class SupportConsentService:
    """
    Service for managing support data access consent.

    Implements support-with-consent requirements from GDPR Phase 1:
        - Explicit consent required for support data access
        - Consent is time-limited with expiry
        - Consent can be revoked at any time
        - All consent actions are auditable

    Usage:
        consent_service = SupportConsentService()

        # Support agent creates consent request
        request = consent_service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-12345",
            user_id="user_123",
            workspace_id="ws_456",
            scopes={ConsentScope.TELEMETRY, ConsentScope.LOGS},
            purpose="Debug performance issue",
        )

        # Customer approves (would be via UI/email link)
        consent = consent_service.approve_request(
            request_id=request.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
        )

        # Before accessing data, verify consent
        result = consent_service.verify_consent(
            support_agent_id="agent_001",
            workspace_id="ws_456",
            required_scope=ConsentScope.TELEMETRY,
        )

        if result.has_consent:
            # Access data
            consent_service.record_access(result.consent_id)
        else:
            raise AccessDenied(result.error)

        # Customer can revoke at any time
        consent_service.revoke_consent(
            consent_id=consent.id,
            revoked_by="user_123",
            reason="No longer needed",
        )
    """

    def __init__(self):
        """Initialize consent service."""
        self._requests: Dict[str, ConsentRequest] = {}
        self._consents: Dict[str, ConsentRecord] = {}
        self._audit_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def create_request(
        self,
        support_agent_id: str,
        support_ticket_id: str,
        user_id: str,
        workspace_id: str,
        organization_id: str = "",
        consent_type: ConsentType = ConsentType.SUPPORT_DATA_ACCESS,
        scopes: Optional[Set[ConsentScope]] = None,
        purpose: str = "",
        duration_hours: int = DEFAULT_CONSENT_DURATION_HOURS,
    ) -> ConsentRequest:
        """
        Create a consent request for support data access.

        Args:
            support_agent_id: ID of support agent requesting access
            support_ticket_id: Associated support ticket
            user_id: Customer user ID who will approve
            workspace_id: Workspace scope
            organization_id: Organization context
            consent_type: Type of consent requested
            scopes: Data scopes requested
            purpose: Reason for access
            duration_hours: Requested duration

        Returns:
            Created ConsentRequest

        Raises:
            ValueError: If scopes are invalid for consent type
        """
        # Validate scopes
        scopes = scopes or {ConsentScope.FULL}
        valid_scopes = VALID_SCOPES.get(consent_type, set())
        invalid_scopes = scopes - valid_scopes
        if invalid_scopes:
            raise ValueError(
                f"Invalid scopes {invalid_scopes} for consent type {consent_type.value}. "
                f"Valid scopes: {valid_scopes}"
            )

        # Enforce maximum duration
        max_hours = MAX_CONSENT_DURATION_DAYS * 24
        if duration_hours > max_hours:
            duration_hours = max_hours

        request = ConsentRequest(
            consent_type=consent_type,
            support_agent_id=support_agent_id,
            support_ticket_id=support_ticket_id,
            user_id=user_id,
            workspace_id=workspace_id,
            organization_id=organization_id,
            scopes=scopes,
            purpose=purpose,
            requested_duration_hours=duration_hours,
        )

        with self._lock:
            self._requests[request.id] = request
            self._log_audit("consent_request_created", request=request)

        return request

    def get_request(self, request_id: str) -> Optional[ConsentRequest]:
        """Get consent request by ID."""
        with self._lock:
            return self._requests.get(request_id)

    def get_pending_requests(self, user_id: str) -> List[ConsentRequest]:
        """Get all pending consent requests for a user."""
        with self._lock:
            return [
                r
                for r in self._requests.values()
                if r.user_id == user_id and r.status == ConsentStatus.PENDING and not r.is_expired
            ]

    def approve_request(
        self,
        request_id: str,
        approved_by: str,
        approved_by_email: str,
        duration_hours: Optional[int] = None,
    ) -> Optional[ConsentRecord]:
        """
        Approve a consent request.

        Args:
            request_id: Request to approve
            approved_by: User ID approving
            approved_by_email: Email of approving user
            duration_hours: Override requested duration

        Returns:
            Created ConsentRecord or None if request not found/invalid
        """
        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return None

            if request.status != ConsentStatus.PENDING:
                return None

            if request.is_expired:
                request.status = ConsentStatus.EXPIRED
                return None

            # Calculate expiry
            duration = duration_hours or request.requested_duration_hours
            max_hours = MAX_CONSENT_DURATION_DAYS * 24
            duration = min(duration, max_hours)

            # Create consent record
            consent = ConsentRecord(
                consent_type=request.consent_type,
                request_id=request.id,
                user_id=request.user_id,
                workspace_id=request.workspace_id,
                organization_id=request.organization_id,
                granted_by_email=approved_by_email,
                support_agent_id=request.support_agent_id,
                support_ticket_id=request.support_ticket_id,
                scopes=request.scopes,
                purpose=request.purpose,
                expires_at=datetime.utcnow() + timedelta(hours=duration),
                status=ConsentStatus.ACTIVE,
            )

            # Update request status
            request.status = ConsentStatus.ACTIVE

            # Store consent
            self._consents[consent.id] = consent

            self._log_audit(
                "consent_granted",
                consent=consent,
                extra={"approved_by": approved_by, "duration_hours": duration},
            )

            return consent

    def deny_request(
        self,
        request_id: str,
        denied_by: str,
        reason: str = "",
    ) -> bool:
        """
        Deny a consent request.

        Args:
            request_id: Request to deny
            denied_by: User ID denying
            reason: Reason for denial

        Returns:
            True if denial recorded
        """
        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return False

            if request.status != ConsentStatus.PENDING:
                return False

            request.status = ConsentStatus.DENIED

            self._log_audit(
                "consent_denied",
                request=request,
                extra={"denied_by": denied_by, "reason": reason},
            )

            return True

    def get_consent(self, consent_id: str) -> Optional[ConsentRecord]:
        """Get consent record by ID."""
        with self._lock:
            consent = self._consents.get(consent_id)
            if consent:
                self._update_consent_status(consent)
            return consent

    def get_active_consents(self, workspace_id: str) -> List[ConsentRecord]:
        """Get all active consents for a workspace."""
        with self._lock:
            result = []
            for consent in self._consents.values():
                self._update_consent_status(consent)
                if consent.workspace_id == workspace_id and consent.is_active:
                    result.append(consent)
            return result

    def get_user_consents(self, user_id: str) -> List[ConsentRecord]:
        """Get all consents (active and historical) for a user."""
        with self._lock:
            result = []
            for consent in self._consents.values():
                self._update_consent_status(consent)
                if consent.user_id == user_id:
                    result.append(consent)
            return result

    def verify_consent(
        self,
        support_agent_id: str,
        workspace_id: str,
        required_scope: ConsentScope,
    ) -> ConsentVerificationResult:
        """
        Verify if valid consent exists for data access.

        This method should be called BEFORE any support data access.

        Args:
            support_agent_id: Support agent requesting access
            workspace_id: Workspace being accessed
            required_scope: Scope of data being accessed

        Returns:
            ConsentVerificationResult indicating if access is allowed
        """
        with self._lock:
            # Find active consent for this agent and workspace
            for consent in self._consents.values():
                self._update_consent_status(consent)

                if not consent.is_active:
                    continue

                if consent.workspace_id != workspace_id:
                    continue

                if consent.support_agent_id != support_agent_id:
                    continue

                # Check scope
                if required_scope not in consent.scopes and ConsentScope.FULL not in consent.scopes:
                    continue

                # Found valid consent
                return ConsentVerificationResult(
                    has_consent=True,
                    consent_id=consent.id,
                    consent_type=consent.consent_type,
                    scopes=consent.scopes,
                    expires_at=consent.expires_at,
                )

            # No valid consent found
            return ConsentVerificationResult(
                has_consent=False,
                error=f"No active consent found for agent {support_agent_id} on workspace {workspace_id} with scope {required_scope.value}",
            )

    def record_access(self, consent_id: str) -> bool:
        """
        Record a data access under consent.

        Should be called each time data is accessed under a consent.

        Args:
            consent_id: Consent under which access occurred

        Returns:
            True if access recorded
        """
        with self._lock:
            consent = self._consents.get(consent_id)
            if not consent:
                return False

            self._update_consent_status(consent)
            if not consent.is_active:
                return False

            consent.access_count += 1
            consent.last_access_at = datetime.utcnow()

            self._log_audit(
                "consent_access_recorded",
                consent=consent,
                extra={"access_count": consent.access_count},
            )

            return True

    def revoke_consent(
        self,
        consent_id: str,
        revoked_by: str,
        reason: str = "",
    ) -> bool:
        """
        Revoke a consent (immediate effect).

        Customers can revoke consent at any time.

        Args:
            consent_id: Consent to revoke
            revoked_by: User ID revoking
            reason: Reason for revocation

        Returns:
            True if revocation recorded
        """
        with self._lock:
            consent = self._consents.get(consent_id)
            if not consent:
                return False

            if consent.status == ConsentStatus.REVOKED:
                return False  # Already revoked

            consent.status = ConsentStatus.REVOKED
            consent.revoked_at = datetime.utcnow()
            consent.revoked_by = revoked_by
            consent.revocation_reason = reason

            self._log_audit(
                "consent_revoked",
                consent=consent,
                extra={"revoked_by": revoked_by, "reason": reason},
            )

            return True

    def _update_consent_status(self, consent: ConsentRecord) -> None:
        """Update consent status if expired."""
        if consent.status == ConsentStatus.ACTIVE and consent.is_expired:
            consent.status = ConsentStatus.EXPIRED
            self._log_audit("consent_expired", consent=consent)

    def get_audit_log(
        self,
        consent_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get consent audit log.

        Args:
            consent_id: Filter by consent ID
            workspace_id: Filter by workspace ID

        Returns:
            List of audit log entries
        """
        with self._lock:
            result = list(self._audit_log)

            if consent_id:
                result = [e for e in result if e.get("consent_id") == consent_id]

            if workspace_id:
                result = [e for e in result if e.get("workspace_id") == workspace_id]

            return result

    def _log_audit(
        self,
        action: str,
        request: Optional[ConsentRequest] = None,
        consent: Optional[ConsentRecord] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log audit event."""
        entry = {
            "action": action,
            "timestamp": datetime.utcnow().isoformat(),
            **(extra or {}),
        }

        if request:
            entry.update(
                {
                    "request_id": request.id,
                    "consent_type": request.consent_type.value,
                    "user_id": request.user_id,
                    "workspace_id": request.workspace_id,
                    "support_agent_id": request.support_agent_id,
                    "support_ticket_id": request.support_ticket_id,
                }
            )

        if consent:
            entry.update(
                {
                    "consent_id": consent.id,
                    "request_id": consent.request_id,
                    "consent_type": consent.consent_type.value,
                    "user_id": consent.user_id,
                    "workspace_id": consent.workspace_id,
                    "support_agent_id": consent.support_agent_id,
                    "support_ticket_id": consent.support_ticket_id,
                }
            )

        self._audit_log.append(entry)

    def cleanup_expired(self) -> int:
        """
        Clean up expired requests and update consent statuses.

        Should be run periodically (e.g., hourly).

        Returns:
            Number of items cleaned up
        """
        count = 0
        with self._lock:
            # Update expired consents
            for consent in self._consents.values():
                if consent.status == ConsentStatus.ACTIVE and consent.is_expired:
                    consent.status = ConsentStatus.EXPIRED
                    self._log_audit("consent_expired", consent=consent)
                    count += 1

            # Clean up old expired requests (older than 30 days)
            cutoff = datetime.utcnow() - timedelta(days=30)
            expired_request_ids = [
                r.id for r in self._requests.values() if r.is_expired and r.created_at < cutoff
            ]
            for rid in expired_request_ids:
                del self._requests[rid]
                count += 1

        return count


# Module-level service instance (optional, for convenience)
_default_service: Optional[SupportConsentService] = None


def get_consent_service() -> SupportConsentService:
    """Get or create default consent service instance."""
    global _default_service
    if _default_service is None:
        _default_service = SupportConsentService()
    return _default_service
