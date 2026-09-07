# -*- coding: utf-8 -*-
"""
GDPR Phase 6: Access Audit Service.

Implements comprehensive audit logging for all access to sensitive data:
- Every access attributable to principal and request_id
- Immutable audit entries with integrity hashes
- Query and export capabilities
- Tamper detection via hash chain

Design Doc Reference:
    - Phase 6: "Access audit log: who accessed what/when"
    - Design Doc 14.4: "Audit log of access"

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple
from uuid import uuid4

from .rbac_service import (
    AccessContext,
    AccessDecision,
    Principal,
    ResourceType,
    SensitivityLevel,
)


logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Audit log retention (7 years for compliance)
AUDIT_LOG_RETENTION_DAYS: Final[int] = 2555  # ~7 years

# Maximum entries per query
MAX_QUERY_RESULTS: Final[int] = 10000

# Hash chain verification batch size
HASH_CHAIN_BATCH_SIZE: Final[int] = 1000


# ============================================================================
# Enums
# ============================================================================


class AuditAction(str, Enum):
    """Audit action types."""

    # CRUD operations
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"

    # Special operations
    APPROVE = "approve"
    REJECT = "reject"
    EXPORT = "export"
    IMPORT = "import"
    EXECUTE = "execute"

    # Access operations
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    MFA_CHALLENGE = "mfa_challenge"
    MFA_VERIFIED = "mfa_verified"
    MFA_FAILED = "mfa_failed"

    # Break-glass operations
    BREAK_GLASS_REQUEST = "break_glass_request"
    BREAK_GLASS_APPROVE = "break_glass_approve"
    BREAK_GLASS_DENY = "break_glass_deny"
    BREAK_GLASS_REVOKE = "break_glass_revoke"
    BREAK_GLASS_ACCESS = "break_glass_access"
    BREAK_GLASS_EXPIRE = "break_glass_expire"

    # DSAR operations
    DSAR_REQUEST = "dsar_request"
    DSAR_PROCESS = "dsar_process"
    DSAR_EXPORT = "dsar_export"
    DSAR_DELETE = "dsar_delete"

    # Change operations
    CHANGE_REQUEST = "change_request"
    CHANGE_APPROVE = "change_approve"
    CHANGE_REJECT = "change_reject"
    CHANGE_DEPLOY = "change_deploy"
    CHANGE_ROLLBACK = "change_rollback"

    # Trust operations
    AGENT_ENROLL = "agent_enroll"
    AGENT_REVOKE = "agent_revoke"
    AGENT_SUSPEND = "agent_suspend"

    # Admin operations
    ROLE_ASSIGN = "role_assign"
    ROLE_REVOKE = "role_revoke"
    PERMISSION_GRANT = "permission_grant"
    PERMISSION_REVOKE = "permission_revoke"

    # Data operations
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"
    BULK_OPERATION = "bulk_operation"


class AuditResult(str, Enum):
    """Result of audited operation."""

    SUCCESS = "success"
    DENIED = "denied"
    ERROR = "error"
    PARTIAL = "partial"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class AuditPrincipal:
    """
    Principal information for audit entry.
    """

    id: str = ""
    type: str = "user"  # user, agent, system, service, anonymous
    email: Optional[str] = None
    organization_id: str = ""
    roles: List[str] = field(default_factory=list)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    mfa_verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "email": self.email,
            "organization_id": self.organization_id,
            "roles": self.roles,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "session_id": self.session_id,
            "mfa_verified": self.mfa_verified,
        }

    @classmethod
    def from_principal(cls, principal: Principal) -> "AuditPrincipal":
        """Create from RBAC Principal."""
        return cls(
            id=principal.id,
            type=principal.type,
            email=principal.email,
            organization_id=principal.organization_id,
            roles=list(principal.roles),
            ip_address=principal.ip_address,
            user_agent=principal.user_agent,
            session_id=principal.session_id,
            mfa_verified=principal.mfa_verified,
        )


@dataclass
class AuditEntry:
    """
    Immutable audit log entry.

    Every sensitive access creates an audit entry that includes:
    - Who (principal)
    - What (action, resource)
    - When (timestamp)
    - Why (reason, request_id)
    - Result (success/denied/error)
    - Integrity (hash)
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Context
    workspace_id: str = ""
    request_id: str = ""
    correlation_id: Optional[str] = None

    # Principal
    principal: AuditPrincipal = field(default_factory=AuditPrincipal)

    # Action
    action: AuditAction = AuditAction.READ
    resource_type: str = ""
    resource_id: Optional[str] = None
    resource_name: Optional[str] = None
    operation: Optional[str] = None  # HTTP method + path or operation name
    data_categories: List[str] = field(default_factory=list)
    sensitivity_level: SensitivityLevel = SensitivityLevel.STANDARD

    # Authorization
    authorization_method: str = "rbac"  # rbac, break_glass, api_key, denied
    role_used: Optional[str] = None
    permissions_checked: List[str] = field(default_factory=list)
    break_glass_id: Optional[str] = None
    consent_id: Optional[str] = None

    # Context/Reason
    reason: Optional[str] = None
    ticket_id: Optional[str] = None
    related_request_ids: List[str] = field(default_factory=list)

    # Result
    result: AuditResult = AuditResult.SUCCESS
    records_accessed: int = 0
    error_message: Optional[str] = None

    # Additional context
    context: Dict[str, Any] = field(default_factory=dict)

    # Integrity
    integrity_hash: str = ""
    previous_hash: Optional[str] = None  # For hash chain

    def __post_init__(self):
        """Compute integrity hash after initialization."""
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute integrity hash for this entry."""
        content = json.dumps(
            {
                "id": self.id,
                "timestamp": self.timestamp.isoformat(),
                "workspace_id": self.workspace_id,
                "request_id": self.request_id,
                "principal": self.principal.to_dict(),
                "action": self.action.value,
                "resource_type": self.resource_type,
                "resource_id": self.resource_id,
                "result": self.result.value,
                "previous_hash": self.previous_hash,
            },
            sort_keys=True,
            default=str,
        )
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "workspace_id": self.workspace_id,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "principal": self.principal.to_dict(),
            "action": {
                "type": self.action.value,
                "resource_type": self.resource_type,
                "resource_id": self.resource_id,
                "resource_name": self.resource_name,
                "operation": self.operation,
                "data_categories": self.data_categories,
                "sensitivity_level": self.sensitivity_level.value,
            },
            "authorization": {
                "method": self.authorization_method,
                "role_used": self.role_used,
                "permissions_checked": self.permissions_checked,
                "break_glass_id": self.break_glass_id,
                "consent_id": self.consent_id,
            },
            "context": {
                "reason": self.reason,
                "ticket_id": self.ticket_id,
                "related_request_ids": self.related_request_ids,
                "extra": self.context,
            },
            "result": {
                "status": self.result.value,
                "records_accessed": self.records_accessed,
                "error_message": self.error_message,
            },
            "integrity": {
                "hash": self.integrity_hash,
                "previous_hash": self.previous_hash,
            },
        }


@dataclass
class AuditQuery:
    """Query parameters for audit log search."""

    workspace_id: Optional[str] = None
    principal_id: Optional[str] = None
    principal_type: Optional[str] = None
    action: Optional[AuditAction] = None
    actions: Optional[List[AuditAction]] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    result: Optional[AuditResult] = None
    sensitivity_level: Optional[SensitivityLevel] = None
    break_glass_only: bool = False
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    ticket_id: Optional[str] = None
    ip_address: Optional[str] = None
    limit: int = 100
    offset: int = 0


@dataclass
class AuditStats:
    """Audit log statistics."""

    total_entries: int = 0
    by_action: Dict[str, int] = field(default_factory=dict)
    by_result: Dict[str, int] = field(default_factory=dict)
    by_resource_type: Dict[str, int] = field(default_factory=dict)
    by_principal_type: Dict[str, int] = field(default_factory=dict)
    by_sensitivity: Dict[str, int] = field(default_factory=dict)
    break_glass_count: int = 0
    denied_count: int = 0
    error_count: int = 0
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_entries": self.total_entries,
            "by_action": self.by_action,
            "by_result": self.by_result,
            "by_resource_type": self.by_resource_type,
            "by_principal_type": self.by_principal_type,
            "by_sensitivity": self.by_sensitivity,
            "break_glass_count": self.break_glass_count,
            "denied_count": self.denied_count,
            "error_count": self.error_count,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
        }


@dataclass
class AuditExport:
    """Audit log export."""

    export_id: str = field(default_factory=lambda: str(uuid4()))
    exported_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exported_by: str = ""
    workspace_id: str = ""
    query: Optional[AuditQuery] = None
    entry_count: int = 0
    entries: List[Dict[str, Any]] = field(default_factory=list)
    checksum: str = ""
    format: str = "json"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "export_metadata": {
                "export_id": self.export_id,
                "exported_at": self.exported_at.isoformat(),
                "exported_by": self.exported_by,
                "workspace_id": self.workspace_id,
                "export_type": "access_audit",
                "entry_count": self.entry_count,
                "format": self.format,
            },
            "integrity": {
                "checksum": self.checksum,
            },
            "data": self.entries,
        }


# ============================================================================
# Access Audit Service
# ============================================================================


class AccessAuditService:
    """
    Access Audit Service.

    Provides comprehensive audit logging for all sensitive data access:
    - Immutable audit entries
    - Hash chain for integrity
    - Query and filtering
    - Export capabilities
    - Statistics and reporting

    Usage:
        service = AccessAuditService()

        # Log access
        entry = service.log_access(
            workspace_id="ws-123",
            principal=AuditPrincipal(id="user-456", type="user"),
            action=AuditAction.READ,
            resource_type="telemetry",
            resource_id="telem-789",
            result=AuditResult.SUCCESS,
        )

        # Query logs
        entries = service.query(AuditQuery(
            workspace_id="ws-123",
            action=AuditAction.READ,
            start_time=datetime.now() - timedelta(days=7),
        ))

        # Export
        export = service.export(
            workspace_id="ws-123",
            exported_by="admin-123",
            query=query,
        )
    """

    def __init__(
        self,
        storage_callback: Optional[Callable[[AuditEntry], None]] = None,
        alert_callback: Optional[Callable[[AuditEntry], None]] = None,
        enable_hash_chain: bool = True,
    ):
        """
        Initialize Access Audit Service.

        Args:
            storage_callback: Callback for persistent storage
            alert_callback: Callback for security alerts
            enable_hash_chain: Enable hash chain for integrity
        """
        self._storage_callback = storage_callback
        self._alert_callback = alert_callback
        self._enable_hash_chain = enable_hash_chain

        # In-memory storage (use storage_callback for persistence)
        self._entries: List[AuditEntry] = []
        self._entries_by_workspace: Dict[str, List[AuditEntry]] = defaultdict(list)
        self._entries_by_request: Dict[str, List[AuditEntry]] = defaultdict(list)

        # Hash chain
        self._last_hash: Dict[str, str] = {}  # workspace_id -> last hash

        self._lock = threading.Lock()

    # ========================================================================
    # Logging
    # ========================================================================

    def log_access(
        self,
        workspace_id: str,
        principal: AuditPrincipal,
        action: AuditAction,
        resource_type: str,
        result: AuditResult,
        resource_id: Optional[str] = None,
        resource_name: Optional[str] = None,
        operation: Optional[str] = None,
        request_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        data_categories: Optional[List[str]] = None,
        sensitivity_level: SensitivityLevel = SensitivityLevel.STANDARD,
        authorization_method: str = "rbac",
        role_used: Optional[str] = None,
        permissions_checked: Optional[List[str]] = None,
        break_glass_id: Optional[str] = None,
        consent_id: Optional[str] = None,
        reason: Optional[str] = None,
        ticket_id: Optional[str] = None,
        related_request_ids: Optional[List[str]] = None,
        records_accessed: int = 0,
        error_message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """
        Log an access event.

        Args:
            workspace_id: Workspace context
            principal: Who performed the action
            action: What action was performed
            resource_type: Type of resource accessed
            result: Result of the action
            resource_id: Specific resource ID
            resource_name: Human-readable resource name
            operation: HTTP operation or command
            request_id: Request tracking ID
            correlation_id: Correlation ID for related operations
            data_categories: Data categories accessed
            sensitivity_level: Sensitivity of data
            authorization_method: How access was authorized
            role_used: Role that granted access
            permissions_checked: Permissions that were checked
            break_glass_id: Break-glass request ID if applicable
            consent_id: Consent record ID if applicable
            reason: Reason for access
            ticket_id: Related ticket/incident ID
            related_request_ids: Related request IDs
            records_accessed: Number of records accessed
            error_message: Error message if failed
            context: Additional context

        Returns:
            Created AuditEntry
        """
        with self._lock:
            # Get previous hash for chain
            previous_hash = None
            if self._enable_hash_chain:
                previous_hash = self._last_hash.get(workspace_id)

            entry = AuditEntry(
                workspace_id=workspace_id,
                request_id=request_id or str(uuid4()),
                correlation_id=correlation_id,
                principal=principal,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                resource_name=resource_name,
                operation=operation,
                data_categories=data_categories or [],
                sensitivity_level=sensitivity_level,
                authorization_method=authorization_method,
                role_used=role_used,
                permissions_checked=permissions_checked or [],
                break_glass_id=break_glass_id,
                consent_id=consent_id,
                reason=reason,
                ticket_id=ticket_id,
                related_request_ids=related_request_ids or [],
                result=result,
                records_accessed=records_accessed,
                error_message=error_message,
                context=context or {},
                previous_hash=previous_hash,
            )

            # Store entry
            self._entries.append(entry)
            self._entries_by_workspace[workspace_id].append(entry)
            self._entries_by_request[entry.request_id].append(entry)

            # Update hash chain
            if self._enable_hash_chain:
                self._last_hash[workspace_id] = entry.integrity_hash

            # Persist if callback provided
            if self._storage_callback:
                try:
                    self._storage_callback(entry)
                except Exception as e:
                    logger.error(f"Failed to persist audit entry: {e}")

            # Alert on suspicious activity
            self._check_alerts(entry)

            logger.debug(
                f"Audit logged: {action.value} {resource_type} "
                f"by {principal.id} -> {result.value}"
            )

            return entry

    def log_from_rbac(
        self,
        context: AccessContext,
        decision: AccessDecision,
        records_accessed: int = 0,
        resource_name: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> AuditEntry:
        """
        Log access from RBAC check result.

        Args:
            context: Access context from RBAC
            decision: Access decision from RBAC
            records_accessed: Number of records accessed
            resource_name: Human-readable resource name
            operation: Operation performed

        Returns:
            Created AuditEntry
        """
        result = AuditResult.SUCCESS if decision.granted else AuditResult.DENIED

        return self.log_access(
            workspace_id=context.workspace_id,
            principal=AuditPrincipal.from_principal(context.principal),
            action=self._scope_to_action(context.scope),
            resource_type=context.resource_type,
            resource_id=context.resource_id,
            resource_name=resource_name,
            operation=operation,
            request_id=context.request_id,
            sensitivity_level=decision.sensitivity_level,
            authorization_method=decision.method,
            role_used=decision.role_used,
            permissions_checked=decision.permissions_checked,
            break_glass_id=context.principal.break_glass_id,
            reason=context.reason,
            ticket_id=context.ticket_id,
            result=result,
            records_accessed=records_accessed if decision.granted else 0,
            error_message=decision.reason if not decision.granted else None,
        )

    def _scope_to_action(self, scope: str) -> AuditAction:
        """Convert RBAC scope to audit action."""
        scope_map = {
            "read": AuditAction.READ,
            "write": AuditAction.UPDATE,
            "create": AuditAction.CREATE,
            "delete": AuditAction.DELETE,
            "admin": AuditAction.UPDATE,
            "approve": AuditAction.APPROVE,
            "export": AuditAction.EXPORT,
            "execute": AuditAction.EXECUTE,
        }
        return scope_map.get(scope.lower(), AuditAction.DATA_ACCESS)

    # ========================================================================
    # Querying
    # ========================================================================

    def query(self, query: AuditQuery) -> List[AuditEntry]:
        """
        Query audit log entries.

        Args:
            query: Query parameters

        Returns:
            List of matching AuditEntry objects
        """
        with self._lock:
            # Start with all entries or workspace-specific
            if query.workspace_id:
                entries = list(self._entries_by_workspace.get(query.workspace_id, []))
            else:
                entries = list(self._entries)

            # Apply filters
            if query.principal_id:
                entries = [e for e in entries if e.principal.id == query.principal_id]

            if query.principal_type:
                entries = [e for e in entries if e.principal.type == query.principal_type]

            if query.action:
                entries = [e for e in entries if e.action == query.action]

            if query.actions:
                entries = [e for e in entries if e.action in query.actions]

            if query.resource_type:
                entries = [e for e in entries if e.resource_type == query.resource_type]

            if query.resource_id:
                entries = [e for e in entries if e.resource_id == query.resource_id]

            if query.result:
                entries = [e for e in entries if e.result == query.result]

            if query.sensitivity_level:
                entries = [e for e in entries if e.sensitivity_level == query.sensitivity_level]

            if query.break_glass_only:
                entries = [e for e in entries if e.break_glass_id is not None]

            if query.start_time:
                entries = [e for e in entries if e.timestamp >= query.start_time]

            if query.end_time:
                entries = [e for e in entries if e.timestamp <= query.end_time]

            if query.request_id:
                entries = [e for e in entries if e.request_id == query.request_id]

            if query.correlation_id:
                entries = [e for e in entries if e.correlation_id == query.correlation_id]

            if query.ticket_id:
                entries = [e for e in entries if e.ticket_id == query.ticket_id]

            if query.ip_address:
                entries = [e for e in entries if e.principal.ip_address == query.ip_address]

            # Sort by timestamp descending
            entries.sort(key=lambda e: e.timestamp, reverse=True)

            # Apply pagination
            total = len(entries)
            entries = entries[query.offset : query.offset + min(query.limit, MAX_QUERY_RESULTS)]

            return entries

    def get_entry(self, entry_id: str) -> Optional[AuditEntry]:
        """Get a specific audit entry by ID."""
        with self._lock:
            for entry in self._entries:
                if entry.id == entry_id:
                    return entry
            return None

    def get_by_request_id(self, request_id: str) -> List[AuditEntry]:
        """Get all entries for a request ID."""
        with self._lock:
            return list(self._entries_by_request.get(request_id, []))

    def get_recent(
        self,
        workspace_id: str,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """Get recent entries for a workspace."""
        return self.query(
            AuditQuery(
                workspace_id=workspace_id,
                limit=limit,
            )
        )

    def get_by_principal(
        self,
        principal_id: str,
        workspace_id: Optional[str] = None,
        days: int = 30,
    ) -> List[AuditEntry]:
        """Get entries for a specific principal."""
        return self.query(
            AuditQuery(
                workspace_id=workspace_id,
                principal_id=principal_id,
                start_time=datetime.now(timezone.utc) - timedelta(days=days),
            )
        )

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_stats(
        self,
        workspace_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> AuditStats:
        """
        Get audit statistics.

        Args:
            workspace_id: Optional workspace filter
            start_time: Start of period
            end_time: End of period

        Returns:
            AuditStats
        """
        query = AuditQuery(
            workspace_id=workspace_id,
            start_time=start_time,
            end_time=end_time,
            limit=MAX_QUERY_RESULTS,
        )
        entries = self.query(query)

        stats = AuditStats(
            total_entries=len(entries),
            period_start=start_time,
            period_end=end_time or datetime.now(timezone.utc),
        )

        for entry in entries:
            # By action
            action_key = entry.action.value
            stats.by_action[action_key] = stats.by_action.get(action_key, 0) + 1

            # By result
            result_key = entry.result.value
            stats.by_result[result_key] = stats.by_result.get(result_key, 0) + 1

            # By resource type
            rt_key = entry.resource_type
            stats.by_resource_type[rt_key] = stats.by_resource_type.get(rt_key, 0) + 1

            # By principal type
            pt_key = entry.principal.type
            stats.by_principal_type[pt_key] = stats.by_principal_type.get(pt_key, 0) + 1

            # By sensitivity
            sens_key = entry.sensitivity_level.value
            stats.by_sensitivity[sens_key] = stats.by_sensitivity.get(sens_key, 0) + 1

            # Counts
            if entry.break_glass_id:
                stats.break_glass_count += 1
            if entry.result == AuditResult.DENIED:
                stats.denied_count += 1
            if entry.result == AuditResult.ERROR:
                stats.error_count += 1

        return stats

    # ========================================================================
    # Export
    # ========================================================================

    def export(
        self,
        workspace_id: str,
        exported_by: str,
        query: Optional[AuditQuery] = None,
        format: str = "json",
    ) -> AuditExport:
        """
        Export audit log entries.

        Args:
            workspace_id: Workspace to export
            exported_by: User performing export
            query: Optional query filter
            format: Export format

        Returns:
            AuditExport
        """
        if query is None:
            query = AuditQuery(workspace_id=workspace_id)
        else:
            query.workspace_id = workspace_id

        entries = self.query(query)
        entry_dicts = [e.to_dict() for e in entries]

        # Compute checksum
        content = json.dumps(entry_dicts, sort_keys=True, default=str)
        checksum = f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

        export = AuditExport(
            exported_by=exported_by,
            workspace_id=workspace_id,
            query=query,
            entry_count=len(entries),
            entries=entry_dicts,
            checksum=checksum,
            format=format,
        )

        # Log the export itself
        self.log_access(
            workspace_id=workspace_id,
            principal=AuditPrincipal(id=exported_by, type="user"),
            action=AuditAction.DATA_EXPORT,
            resource_type="access_audit",
            result=AuditResult.SUCCESS,
            records_accessed=len(entries),
            context={"export_id": export.export_id},
        )

        return export

    def export_to_file(
        self,
        workspace_id: str,
        exported_by: str,
        output_path: Path,
        query: Optional[AuditQuery] = None,
    ) -> AuditExport:
        """Export audit log to file."""
        export = self.export(workspace_id, exported_by, query)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export.to_dict(), f, indent=2, default=str, ensure_ascii=False)

        return export

    # ========================================================================
    # Integrity Verification
    # ========================================================================

    def verify_hash_chain(
        self,
        workspace_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Verify hash chain integrity.

        Args:
            workspace_id: Workspace to verify
            start_time: Start of verification period
            end_time: End of verification period

        Returns:
            (is_valid, list of error messages)
        """
        query = AuditQuery(
            workspace_id=workspace_id,
            start_time=start_time,
            end_time=end_time,
            limit=MAX_QUERY_RESULTS,
        )
        entries = self.query(query)

        # Sort by timestamp ascending for chain verification
        entries.sort(key=lambda e: e.timestamp)

        errors = []
        previous_hash = None

        for entry in entries:
            # Verify integrity hash
            expected_hash = entry._compute_hash()
            if entry.integrity_hash != expected_hash:
                errors.append(f"Entry {entry.id}: integrity hash mismatch")

            # Verify chain (if enabled and entry has previous_hash)
            if self._enable_hash_chain and entry.previous_hash:
                if previous_hash and entry.previous_hash != previous_hash:
                    errors.append(f"Entry {entry.id}: chain hash mismatch")

            previous_hash = entry.integrity_hash

        return len(errors) == 0, errors

    # ========================================================================
    # Alerts
    # ========================================================================

    def _check_alerts(self, entry: AuditEntry) -> None:
        """Check if entry should trigger security alert."""
        should_alert = False
        alert_reason = ""

        # Alert on denied access to critical resources
        if entry.result == AuditResult.DENIED and entry.sensitivity_level in (
            SensitivityLevel.CRITICAL,
            SensitivityLevel.RESTRICTED,
        ):
            should_alert = True
            alert_reason = f"Denied access to {entry.sensitivity_level.value} resource"

        # Alert on break-glass usage
        if entry.break_glass_id:
            should_alert = True
            alert_reason = f"Break-glass access: {entry.break_glass_id}"

        # Alert on bulk operations
        if entry.action == AuditAction.BULK_OPERATION:
            should_alert = True
            alert_reason = f"Bulk operation: {entry.records_accessed} records"

        # Alert on errors
        if entry.result == AuditResult.ERROR:
            should_alert = True
            alert_reason = f"Access error: {entry.error_message}"

        if should_alert and self._alert_callback:
            try:
                self._alert_callback(entry)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    # ========================================================================
    # Cleanup
    # ========================================================================

    def cleanup_expired(
        self,
        workspace_id: Optional[str] = None,
        retention_days: int = AUDIT_LOG_RETENTION_DAYS,
    ) -> int:
        """
        Clean up entries older than retention period.

        Note: Audit logs typically should NOT be deleted even after retention
        period unless required by policy. This method is for controlled cleanup.

        Args:
            workspace_id: Optional workspace filter
            retention_days: Retention period in days

        Returns:
            Number of entries removed
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        removed = 0

        with self._lock:
            # Filter entries to keep
            if workspace_id:
                original = self._entries_by_workspace.get(workspace_id, [])
                kept = [e for e in original if e.timestamp >= cutoff]
                removed = len(original) - len(kept)
                self._entries_by_workspace[workspace_id] = kept
            else:
                original = len(self._entries)
                self._entries = [e for e in self._entries if e.timestamp >= cutoff]
                removed = original - len(self._entries)

                # Update workspace index
                for ws_id in self._entries_by_workspace:
                    self._entries_by_workspace[ws_id] = [
                        e for e in self._entries_by_workspace[ws_id] if e.timestamp >= cutoff
                    ]

        if removed > 0:
            logger.info(f"Cleaned up {removed} expired audit entries")

        return removed


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Constants
    "AUDIT_LOG_RETENTION_DAYS",
    "MAX_QUERY_RESULTS",
    # Enums
    "AuditAction",
    "AuditResult",
    # Data classes
    "AuditPrincipal",
    "AuditEntry",
    "AuditQuery",
    "AuditStats",
    "AuditExport",
    # Service
    "AccessAuditService",
]
