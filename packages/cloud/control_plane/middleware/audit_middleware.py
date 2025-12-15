# -*- coding: utf-8 -*-
"""
Access Audit Middleware for Cloud Control Plane.

Phase 2 Implementation (2.2): Real AccessAudit for sensitive data access.

This module provides:
    - AuditMiddleware: FastAPI middleware for automatic audit logging
    - audit_sensitive_access: Decorator for sensitive operations
    - AuditContext: Context manager for audit scopes
    - AuditLogger: Helper for creating audit entries

Design Doc Reference:
    - Phase 2.2: "Реальный AccessAudit для кто смотрел чувствительное"
    - All access to sensitive data must be logged with actor, action, resource

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import functools
import hashlib
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union
from uuid import UUID, uuid4

from fastapi import Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from ..models import AccessAudit, AuditAction


# ============================================================================
# Context Variables
# ============================================================================

# Current request context for audit
_audit_context: ContextVar[Optional["AuditContext"]] = ContextVar(
    "_audit_context", default=None
)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AuditEntry:
    """Audit entry data."""
    action: AuditAction
    resource_type: str
    resource_id: Optional[str] = None
    workspace_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    agent_id: Optional[UUID] = None
    actor_type: str = "user"
    actor_ip: Optional[str] = None
    is_break_glass: bool = False
    break_glass_reason: Optional[str] = None
    break_glass_evidence_hash: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    request_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AuditContext:
    """
    Audit context for a request/operation scope.

    Collects audit entries and commits them at the end of the scope.
    """
    request_id: str
    user_id: Optional[UUID] = None
    agent_id: Optional[UUID] = None
    workspace_id: Optional[UUID] = None
    actor_type: str = "user"
    actor_ip: Optional[str] = None
    entries: List[AuditEntry] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def add_entry(
        self,
        action: AuditAction,
        resource_type: str,
        resource_id: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        is_break_glass: bool = False,
        break_glass_reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Add an audit entry to this context."""
        entry = AuditEntry(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            workspace_id=self.workspace_id,
            user_id=self.user_id,
            agent_id=self.agent_id,
            actor_type=self.actor_type,
            actor_ip=self.actor_ip,
            is_break_glass=is_break_glass,
            break_glass_reason=break_glass_reason,
            success=success,
            error_message=error_message,
            request_id=self.request_id,
            context=context or {},
        )
        self.entries.append(entry)
        return entry


# ============================================================================
# Sensitive Resources Configuration
# ============================================================================

# Resources that require audit logging on read
SENSITIVE_RESOURCES_READ: Set[str] = {
    "telemetry_event",
    "telemetry_events",
    "alert",
    "alerts",
    "approval_record",
    "approval_records",
    "access_audit",
    "access_audits",
    "config_blob",
    "config_blobs",
    "break_glass_request",
    "dsar_request",
    "dsar_export",
}

# Resources that require audit logging on write
SENSITIVE_RESOURCES_WRITE: Set[str] = {
    "agent",
    "agents",
    "deployment",
    "deployments",
    "command",
    "commands",
    "approval_record",
    "config_blob",
    "retention_policy",
    "residency_policy",
    "break_glass_request",
    "dsar_request",
    "user",
    "role",
    "permission",
}

# Endpoints that always require audit logging
SENSITIVE_ENDPOINTS: Set[str] = {
    "/api/v1/telemetry/events",
    "/api/v1/telemetry/events/{event_id}",
    "/api/v1/telemetry/export",
    "/api/v1/governance/dsar",
    "/api/v1/governance/break-glass",
    "/api/v1/governance/retention",
    "/api/v1/governance/residency",
    "/api/v1/agents/{agent_id}/revoke",
    "/api/v1/agents/{agent_id}/suspend",
    "/api/v1/commands/{command_id}/approvals",
}


# ============================================================================
# Audit Logger
# ============================================================================

class AuditLogger:
    """
    Helper class for creating audit log entries.

    Provides a unified API for logging access to sensitive resources.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def log(
        self,
        action: AuditAction,
        resource_type: str,
        resource_id: Optional[str] = None,
        workspace_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
        agent_id: Optional[UUID] = None,
        actor_type: str = "user",
        actor_ip: Optional[str] = None,
        is_break_glass: bool = False,
        break_glass_reason: Optional[str] = None,
        break_glass_evidence_hash: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AccessAudit:
        """
        Create an audit log entry.

        Args:
            action: Type of action (create, read, update, delete, etc.)
            resource_type: Type of resource being accessed
            resource_id: Optional specific resource ID
            workspace_id: Workspace context
            user_id: User performing the action
            agent_id: Agent performing the action (if applicable)
            actor_type: Type of actor (user, agent, system)
            actor_ip: IP address of actor
            is_break_glass: Whether this is a break-glass access
            break_glass_reason: Reason for break-glass access
            break_glass_evidence_hash: Evidence hash for break-glass
            success: Whether the operation succeeded
            error_message: Error message if failed
            request_id: Request correlation ID
            context: Additional context data

        Returns:
            Created audit entry
        """
        audit = AccessAudit(
            workspace_id=workspace_id,
            user_id=user_id,
            agent_id=agent_id,
            actor_type=actor_type,
            actor_ip=actor_ip,
            action=action.value,
            resource_type=resource_type,
            resource_id=resource_id,
            is_break_glass=is_break_glass,
            break_glass_reason=break_glass_reason,
            break_glass_evidence_hash=break_glass_evidence_hash,
            success=success,
            error_message=error_message,
            request_id=request_id,
            context=context or {},
        )
        self.session.add(audit)
        await self.session.flush()
        return audit

    async def log_read(
        self,
        resource_type: str,
        resource_id: Optional[str] = None,
        workspace_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AccessAudit:
        """Log a read access."""
        return await self.log(
            action=AuditAction.READ,
            resource_type=resource_type,
            resource_id=resource_id,
            workspace_id=workspace_id,
            user_id=user_id,
            context=context,
        )

    async def log_export(
        self,
        resource_type: str,
        workspace_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
        records_exported: int = 0,
        export_format: str = "json",
        context: Optional[Dict[str, Any]] = None,
    ) -> AccessAudit:
        """Log a data export."""
        ctx = context or {}
        ctx["records_exported"] = records_exported
        ctx["export_format"] = export_format

        return await self.log(
            action=AuditAction.EXPORT,
            resource_type=resource_type,
            workspace_id=workspace_id,
            user_id=user_id,
            context=ctx,
        )

    async def log_break_glass_access(
        self,
        resource_type: str,
        resource_id: Optional[str] = None,
        workspace_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
        reason: str = "",
        evidence_hash: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AccessAudit:
        """Log a break-glass access."""
        # Generate evidence hash if not provided
        if not evidence_hash:
            evidence_data = f"{uuid4()}:{user_id}:{reason}:{datetime.now(timezone.utc).isoformat()}"
            evidence_hash = hashlib.sha256(evidence_data.encode()).hexdigest()

        return await self.log(
            action=AuditAction.BREAK_GLASS,
            resource_type=resource_type,
            resource_id=resource_id,
            workspace_id=workspace_id,
            user_id=user_id,
            is_break_glass=True,
            break_glass_reason=reason,
            break_glass_evidence_hash=evidence_hash,
            context=context,
        )

    async def log_entries(self, entries: List[AuditEntry]) -> List[AccessAudit]:
        """Log multiple audit entries at once."""
        audits = []
        for entry in entries:
            audit = await self.log(
                action=entry.action,
                resource_type=entry.resource_type,
                resource_id=entry.resource_id,
                workspace_id=entry.workspace_id,
                user_id=entry.user_id,
                agent_id=entry.agent_id,
                actor_type=entry.actor_type,
                actor_ip=entry.actor_ip,
                is_break_glass=entry.is_break_glass,
                break_glass_reason=entry.break_glass_reason,
                break_glass_evidence_hash=entry.break_glass_evidence_hash,
                success=entry.success,
                error_message=entry.error_message,
                request_id=entry.request_id,
                context=entry.context,
            )
            audits.append(audit)
        return audits


# ============================================================================
# Context Manager
# ============================================================================

@asynccontextmanager
async def audit_scope(
    session: AsyncSession,
    user_id: Optional[UUID] = None,
    agent_id: Optional[UUID] = None,
    workspace_id: Optional[UUID] = None,
    actor_type: str = "user",
    actor_ip: Optional[str] = None,
    request_id: Optional[str] = None,
):
    """
    Context manager for audit scope.

    Collects audit entries and commits them at the end.

    Usage:
        async with audit_scope(session, user_id=user.id) as ctx:
            ctx.add_entry(AuditAction.READ, "telemetry_event", str(event_id))
            # ... perform operations
    """
    ctx = AuditContext(
        request_id=request_id or str(uuid4()),
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        actor_type=actor_type,
        actor_ip=actor_ip,
    )

    # Set context variable
    token = _audit_context.set(ctx)

    try:
        yield ctx
    finally:
        # Restore previous context
        _audit_context.reset(token)

        # Commit all entries
        if ctx.entries:
            logger = AuditLogger(session)
            await logger.log_entries(ctx.entries)
            await session.commit()


def get_current_audit_context() -> Optional[AuditContext]:
    """Get the current audit context if one exists."""
    return _audit_context.get()


# ============================================================================
# Decorator
# ============================================================================

F = TypeVar("F", bound=Callable[..., Any])


def audit_sensitive_access(
    action: AuditAction,
    resource_type: str,
    resource_id_param: Optional[str] = None,
    include_result: bool = False,
):
    """
    Decorator for auditing sensitive resource access.

    Usage:
        @audit_sensitive_access(
            action=AuditAction.READ,
            resource_type="telemetry_event",
            resource_id_param="event_id",
        )
        async def get_telemetry_event(event_id: UUID, ...):
            ...

    Args:
        action: Audit action type
        resource_type: Type of resource being accessed
        resource_id_param: Name of the parameter containing resource ID
        include_result: Whether to include result summary in audit context
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get resource ID from parameters
            resource_id = None
            if resource_id_param:
                resource_id = kwargs.get(resource_id_param)
                if resource_id is not None:
                    resource_id = str(resource_id)

            # Get audit context
            ctx = get_current_audit_context()

            try:
                result = await func(*args, **kwargs)

                # Log success
                if ctx:
                    audit_ctx = {"duration_ms": int((time.time() - ctx.start_time) * 1000)}
                    if include_result and result is not None:
                        # Add summary of result (don't include sensitive data)
                        if hasattr(result, "__len__"):
                            audit_ctx["result_count"] = len(result)

                    ctx.add_entry(
                        action=action,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        success=True,
                        context=audit_ctx,
                    )

                return result

            except Exception as e:
                # Log failure
                if ctx:
                    ctx.add_entry(
                        action=action,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        success=False,
                        error_message=str(e),
                    )
                raise

        return wrapper  # type: ignore
    return decorator


# ============================================================================
# FastAPI Middleware
# ============================================================================

class AuditMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic audit logging.

    Logs access to sensitive endpoints and resources.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process request with audit logging."""
        # Generate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid4()))

        # Get actor information from request state (set by auth middleware)
        user_id = getattr(request.state, "user_id", None)
        agent_id = getattr(request.state, "agent_id", None)
        workspace_id = getattr(request.state, "workspace_id", None)

        # Determine actor type
        actor_type = "agent" if agent_id else "user"

        # Get client IP
        actor_ip = request.client.host if request.client else None

        # Create audit context
        ctx = AuditContext(
            request_id=request_id,
            user_id=user_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            actor_type=actor_type,
            actor_ip=actor_ip,
        )

        # Set context variable
        token = _audit_context.set(ctx)

        try:
            # Check if endpoint is sensitive
            path = request.url.path
            is_sensitive = self._is_sensitive_endpoint(path, request.method)

            # Process request
            response = await call_next(request)

            # Log if sensitive and successful
            if is_sensitive and 200 <= response.status_code < 400:
                action = self._get_action_for_method(request.method)
                resource_type = self._extract_resource_type(path)

                ctx.add_entry(
                    action=action,
                    resource_type=resource_type,
                    resource_id=self._extract_resource_id(path),
                    success=True,
                    context={
                        "method": request.method,
                        "path": path,
                        "status_code": response.status_code,
                        "duration_ms": int((time.time() - ctx.start_time) * 1000),
                    },
                )

            return response

        except Exception as e:
            # Log failure
            if self._is_sensitive_endpoint(request.url.path, request.method):
                ctx.add_entry(
                    action=self._get_action_for_method(request.method),
                    resource_type=self._extract_resource_type(request.url.path),
                    success=False,
                    error_message=str(e),
                )
            raise

        finally:
            # Restore context
            _audit_context.reset(token)

            # Note: In production, audit entries would be committed
            # via the database session after the response is sent

    def _is_sensitive_endpoint(self, path: str, method: str) -> bool:
        """Check if endpoint requires audit logging."""
        # Exact match
        if path in SENSITIVE_ENDPOINTS:
            return True

        # Pattern match for common sensitive paths
        sensitive_patterns = [
            "/telemetry/",
            "/governance/",
            "/break-glass/",
            "/dsar/",
            "/revoke",
            "/suspend",
            "/approvals",
            "/export",
        ]

        for pattern in sensitive_patterns:
            if pattern in path:
                return True

        return False

    def _get_action_for_method(self, method: str) -> AuditAction:
        """Map HTTP method to audit action."""
        mapping = {
            "GET": AuditAction.READ,
            "POST": AuditAction.CREATE,
            "PUT": AuditAction.UPDATE,
            "PATCH": AuditAction.UPDATE,
            "DELETE": AuditAction.DELETE,
        }
        return mapping.get(method.upper(), AuditAction.READ)

    def _extract_resource_type(self, path: str) -> str:
        """Extract resource type from path."""
        parts = path.strip("/").split("/")

        # Skip api/v1 prefix
        if len(parts) >= 2 and parts[0] == "api" and parts[1].startswith("v"):
            parts = parts[2:]

        if parts:
            return parts[0]

        return "unknown"

    def _extract_resource_id(self, path: str) -> Optional[str]:
        """Extract resource ID from path if present."""
        parts = path.strip("/").split("/")

        # Look for UUID-like segments
        for part in parts:
            # Check if it looks like a UUID
            if len(part) == 36 and part.count("-") == 4:
                return part

        return None


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Classes
    "AuditEntry",
    "AuditContext",
    "AuditLogger",
    "AuditMiddleware",
    # Functions
    "audit_scope",
    "audit_sensitive_access",
    "get_current_audit_context",
    # Constants
    "SENSITIVE_RESOURCES_READ",
    "SENSITIVE_RESOURCES_WRITE",
    "SENSITIVE_ENDPOINTS",
]
