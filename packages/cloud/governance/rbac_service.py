# -*- coding: utf-8 -*-
"""
GDPR Phase 6: Role-Based Access Control (RBAC) Service.

Implements comprehensive RBAC with:
- Granular scopes (READ, WRITE, ADMIN, SUPPORT, AUDIT, BREAK_GLASS)
- Resource-level permissions
- Workspace isolation
- Permission caching
- Full audit trail

Design Doc Reference:
    - Phase 6: "RBAC inside workspace (read vs admin vs support scopes)"
    - Design Doc 14.4: "Access control: RBAC in workspace"

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Callable, Dict, Final, FrozenSet, List, Optional, Set, Tuple
from uuid import UUID, uuid4


logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Cache TTL for permission checks
PERMISSION_CACHE_TTL_SECONDS: Final[int] = 300  # 5 minutes

# Maximum roles per user
MAX_ROLES_PER_USER: Final[int] = 20

# Maximum permissions per role
MAX_PERMISSIONS_PER_ROLE: Final[int] = 100


# ============================================================================
# Enums
# ============================================================================


class Scope(str, Enum):
    """Access scopes for RBAC."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    SUPPORT = "support"
    AUDIT = "audit"
    BREAK_GLASS = "break_glass"
    APPROVE = "approve"
    EXPORT = "export"
    EXECUTE = "execute"


class ResourceType(str, Enum):
    """Resource types for permission checks."""

    STRATEGY = "strategy"
    STRATEGY_VERSION = "strategy_version"
    BUILD = "build"
    ARTIFACT = "artifact"
    DEPLOYMENT = "deployment"
    RUN = "run"
    AGENT = "agent"
    COMMAND = "command"
    APPROVAL = "approval"
    CONFIG_BLOB = "config_blob"
    TELEMETRY = "telemetry"
    ALERT = "alert"
    AUDIT_LOG = "audit_log"
    ACCESS_AUDIT = "access_audit"
    BREAK_GLASS = "break_glass"
    DSAR = "dsar"
    CONSENT = "consent"
    LEGAL_HOLD = "legal_hold"
    RETENTION_POLICY = "retention_policy"
    GOVERNANCE_POLICY = "governance_policy"
    USER = "user"
    ROLE = "role"
    WORKSPACE = "workspace"
    ORGANIZATION = "organization"


class SensitivityLevel(str, Enum):
    """Data sensitivity levels."""

    STANDARD = "standard"
    SENSITIVE = "sensitive"
    CRITICAL = "critical"
    RESTRICTED = "restricted"


# ============================================================================
# Sensitivity Classification
# ============================================================================

RESOURCE_SENSITIVITY: Dict[ResourceType, SensitivityLevel] = {
    ResourceType.STRATEGY: SensitivityLevel.STANDARD,
    ResourceType.STRATEGY_VERSION: SensitivityLevel.STANDARD,
    ResourceType.BUILD: SensitivityLevel.STANDARD,
    ResourceType.ARTIFACT: SensitivityLevel.SENSITIVE,
    ResourceType.DEPLOYMENT: SensitivityLevel.SENSITIVE,
    ResourceType.RUN: SensitivityLevel.SENSITIVE,
    ResourceType.AGENT: SensitivityLevel.SENSITIVE,
    ResourceType.COMMAND: SensitivityLevel.SENSITIVE,
    ResourceType.APPROVAL: SensitivityLevel.CRITICAL,
    ResourceType.CONFIG_BLOB: SensitivityLevel.SENSITIVE,
    ResourceType.TELEMETRY: SensitivityLevel.SENSITIVE,
    ResourceType.ALERT: SensitivityLevel.STANDARD,
    ResourceType.AUDIT_LOG: SensitivityLevel.CRITICAL,
    ResourceType.ACCESS_AUDIT: SensitivityLevel.CRITICAL,
    ResourceType.BREAK_GLASS: SensitivityLevel.CRITICAL,
    ResourceType.DSAR: SensitivityLevel.CRITICAL,
    ResourceType.CONSENT: SensitivityLevel.SENSITIVE,
    ResourceType.LEGAL_HOLD: SensitivityLevel.CRITICAL,
    ResourceType.RETENTION_POLICY: SensitivityLevel.SENSITIVE,
    ResourceType.GOVERNANCE_POLICY: SensitivityLevel.SENSITIVE,
    ResourceType.USER: SensitivityLevel.SENSITIVE,
    ResourceType.ROLE: SensitivityLevel.SENSITIVE,
    ResourceType.WORKSPACE: SensitivityLevel.STANDARD,
    ResourceType.ORGANIZATION: SensitivityLevel.SENSITIVE,
}

# Resources that require audit logging on access
AUDIT_REQUIRED_RESOURCES: FrozenSet[ResourceType] = frozenset(
    {
        ResourceType.APPROVAL,
        ResourceType.AUDIT_LOG,
        ResourceType.ACCESS_AUDIT,
        ResourceType.BREAK_GLASS,
        ResourceType.DSAR,
        ResourceType.TELEMETRY,
        ResourceType.CONFIG_BLOB,
        ResourceType.AGENT,
        ResourceType.USER,
        ResourceType.LEGAL_HOLD,
    }
)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass(frozen=True)
class Permission:
    """
    Immutable permission definition.

    Format: resource:scope or resource:scope:constraint
    Examples:
        - strategy:read
        - deployment:write
        - telemetry:read:aggregated_only
        - *:read (all resources, read only)
        - strategy:* (strategy, all scopes)
    """

    resource: str  # ResourceType.value or "*"
    scope: str  # Scope.value or "*"
    constraint: Optional[str] = None

    def __str__(self) -> str:
        base = f"{self.resource}:{self.scope}"
        return f"{base}:{self.constraint}" if self.constraint else base

    @classmethod
    def from_string(cls, perm_str: str) -> "Permission":
        """Parse permission from string."""
        parts = perm_str.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid permission format: {perm_str}")
        resource = parts[0]
        scope = parts[1]
        constraint = parts[2] if len(parts) > 2 else None
        return cls(resource=resource, scope=scope, constraint=constraint)

    def matches(self, resource: str, scope: str) -> bool:
        """Check if this permission matches the requested resource and scope."""
        resource_match = self.resource == "*" or self.resource == resource
        scope_match = self.scope == "*" or self.scope == scope
        return resource_match and scope_match


@dataclass
class Role:
    """
    Role definition with permissions.

    Roles can be:
    - Organization-wide (workspace_id is None)
    - Workspace-specific (workspace_id is set)
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    organization_id: str = ""
    workspace_id: Optional[str] = None  # None = org-wide
    permissions: Set[Permission] = field(default_factory=set)
    is_system_role: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None

    def has_permission(self, resource: str, scope: str) -> bool:
        """Check if role has the specified permission."""
        return any(p.matches(resource, scope) for p in self.permissions)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "permissions": [str(p) for p in self.permissions],
            "is_system_role": self.is_system_role,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
        }


@dataclass
class Principal:
    """
    Principal (user or service) for access control.
    """

    id: str = ""
    type: str = "user"  # user, agent, system, service
    email: Optional[str] = None
    organization_id: str = ""
    roles: List[str] = field(default_factory=list)  # Role IDs
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    mfa_verified: bool = False
    break_glass_id: Optional[str] = None  # Active break-glass request

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
            "break_glass_id": self.break_glass_id,
        }


@dataclass
class AccessContext:
    """
    Context for access check.
    """

    principal: Principal
    workspace_id: str = ""
    resource_type: str = ""
    resource_id: Optional[str] = None
    scope: str = ""
    request_id: str = field(default_factory=lambda: str(uuid4()))
    reason: Optional[str] = None
    ticket_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "principal": self.principal.to_dict(),
            "workspace_id": self.workspace_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "scope": self.scope,
            "request_id": self.request_id,
            "reason": self.reason,
            "ticket_id": self.ticket_id,
        }


@dataclass
class AccessDecision:
    """
    Result of access check.
    """

    granted: bool = False
    reason: str = ""
    method: str = "rbac"  # rbac, break_glass, api_key, denied
    role_used: Optional[str] = None
    permissions_checked: List[str] = field(default_factory=list)
    requires_mfa: bool = False
    requires_break_glass: bool = False
    sensitivity_level: SensitivityLevel = SensitivityLevel.STANDARD
    audit_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "granted": self.granted,
            "reason": self.reason,
            "method": self.method,
            "role_used": self.role_used,
            "permissions_checked": self.permissions_checked,
            "requires_mfa": self.requires_mfa,
            "requires_break_glass": self.requires_break_glass,
            "sensitivity_level": self.sensitivity_level.value,
            "audit_required": self.audit_required,
        }


# ============================================================================
# Default Roles
# ============================================================================


def _create_default_roles(organization_id: str) -> Dict[str, Role]:
    """Create default system roles for an organization."""
    return {
        "owner": Role(
            name="owner",
            description="Full access to all resources",
            organization_id=organization_id,
            permissions={Permission("*", "*")},
            is_system_role=True,
        ),
        "admin": Role(
            name="admin",
            description="Administrative access (cannot transfer ownership)",
            organization_id=organization_id,
            permissions={
                Permission("*", "read"),
                Permission("*", "write"),
                Permission("*", "delete"),
                Permission("*", "admin"),
                Permission("*", "approve"),
                Permission("*", "export"),
                Permission("break_glass", "approve"),
            },
            is_system_role=True,
        ),
        "developer": Role(
            name="developer",
            description="Development team access",
            organization_id=organization_id,
            permissions={
                Permission("strategy", "*"),
                Permission("strategy_version", "*"),
                Permission("build", "*"),
                Permission("artifact", "read"),
                Permission("deployment", "read"),
                Permission("deployment", "write"),
                Permission("run", "*"),
                Permission("agent", "read"),
                Permission("command", "read"),
                Permission("command", "write"),
                Permission("config_blob", "read"),
                Permission("config_blob", "write"),
                Permission("telemetry", "read"),
                Permission("alert", "read"),
            },
            is_system_role=True,
        ),
        "viewer": Role(
            name="viewer",
            description="Read-only access to non-sensitive resources",
            organization_id=organization_id,
            permissions={
                Permission("strategy", "read"),
                Permission("strategy_version", "read"),
                Permission("build", "read"),
                Permission("deployment", "read"),
                Permission("run", "read"),
                Permission("alert", "read"),
            },
            is_system_role=True,
        ),
        "support": Role(
            name="support",
            description="Time-limited support access",
            organization_id=organization_id,
            permissions={
                Permission("telemetry", "read"),
                Permission("alert", "read"),
                Permission("agent", "read"),
                Permission("run", "read"),
                Permission("deployment", "read"),
            },
            is_system_role=True,
        ),
        "auditor": Role(
            name="auditor",
            description="Compliance audit access",
            organization_id=organization_id,
            permissions={
                Permission("audit_log", "read"),
                Permission("access_audit", "read"),
                Permission("approval", "read"),
                Permission("break_glass", "read"),
                Permission("dsar", "read"),
                Permission("consent", "read"),
                Permission("legal_hold", "read"),
                Permission("retention_policy", "read"),
                Permission("governance_policy", "read"),
                Permission("*", "export", "audit_only"),
            },
            is_system_role=True,
        ),
        "break_glass_approver": Role(
            name="break_glass_approver",
            description="Can approve break-glass requests",
            organization_id=organization_id,
            permissions={
                Permission("break_glass", "read"),
                Permission("break_glass", "approve"),
            },
            is_system_role=True,
        ),
        "dpo": Role(
            name="dpo",
            description="Data Protection Officer access",
            organization_id=organization_id,
            permissions={
                Permission("dsar", "*"),
                Permission("consent", "*"),
                Permission("legal_hold", "*"),
                Permission("retention_policy", "*"),
                Permission("governance_policy", "*"),
                Permission("audit_log", "read"),
                Permission("access_audit", "read"),
                Permission("user", "read"),
            },
            is_system_role=True,
        ),
    }


# ============================================================================
# RBAC Service
# ============================================================================


class RBACService:
    """
    Role-Based Access Control Service.

    Provides:
    - Permission checking
    - Role management
    - Workspace isolation
    - Permission caching
    - Audit integration

    Usage:
        rbac = RBACService()

        # Check permission
        context = AccessContext(
            principal=Principal(id="user-123", roles=["developer"]),
            workspace_id="ws-456",
            resource_type="strategy",
            scope="read",
        )
        decision = rbac.check_access(context)

        if decision.granted:
            # Access allowed
            pass
    """

    def __init__(
        self,
        audit_callback: Optional[Callable[[AccessContext, AccessDecision], None]] = None,
        cache_enabled: bool = True,
    ):
        """
        Initialize RBAC service.

        Args:
            audit_callback: Callback for audit logging
            cache_enabled: Enable permission caching
        """
        self._audit_callback = audit_callback
        self._cache_enabled = cache_enabled

        # Storage
        self._roles: Dict[str, Role] = {}  # role_id -> Role
        self._user_roles: Dict[str, Set[str]] = {}  # user_id -> set of role_ids
        self._workspace_roles: Dict[str, Set[str]] = {}  # workspace_id -> set of role_ids
        self._org_default_roles: Dict[str, Dict[str, Role]] = {}  # org_id -> default roles

        # Cache
        self._permission_cache: Dict[str, Tuple[AccessDecision, datetime]] = {}
        self._lock = threading.Lock()

    # ========================================================================
    # Role Management
    # ========================================================================

    def create_role(
        self,
        name: str,
        description: str,
        organization_id: str,
        permissions: Set[Permission],
        workspace_id: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> Role:
        """
        Create a new role.

        Args:
            name: Role name
            description: Role description
            organization_id: Organization ID
            permissions: Set of permissions
            workspace_id: Optional workspace scope
            created_by: Creator user ID

        Returns:
            Created Role

        Raises:
            ValueError: If role name already exists or too many permissions
        """
        if len(permissions) > MAX_PERMISSIONS_PER_ROLE:
            raise ValueError(f"Too many permissions (max {MAX_PERMISSIONS_PER_ROLE})")

        with self._lock:
            # Check for duplicate name
            for existing in self._roles.values():
                if (
                    existing.name == name
                    and existing.organization_id == organization_id
                    and existing.workspace_id == workspace_id
                ):
                    raise ValueError(f"Role '{name}' already exists")

            role = Role(
                name=name,
                description=description,
                organization_id=organization_id,
                workspace_id=workspace_id,
                permissions=permissions,
                is_system_role=False,
                created_by=created_by,
            )

            self._roles[role.id] = role

            # Add to workspace index
            if workspace_id:
                if workspace_id not in self._workspace_roles:
                    self._workspace_roles[workspace_id] = set()
                self._workspace_roles[workspace_id].add(role.id)

            logger.info(f"Role created: {role.id} ({name})")
            return role

    def get_role(self, role_id: str) -> Optional[Role]:
        """Get role by ID."""
        with self._lock:
            return self._roles.get(role_id)

    def get_role_by_name(
        self,
        name: str,
        organization_id: str,
        workspace_id: Optional[str] = None,
    ) -> Optional[Role]:
        """Get role by name."""
        with self._lock:
            # Check workspace-specific roles first
            if workspace_id:
                for role in self._roles.values():
                    if (
                        role.name == name
                        and role.organization_id == organization_id
                        and role.workspace_id == workspace_id
                    ):
                        return role

            # Then check org-wide roles
            for role in self._roles.values():
                if (
                    role.name == name
                    and role.organization_id == organization_id
                    and role.workspace_id is None
                ):
                    return role

            # Finally check default roles
            if organization_id in self._org_default_roles:
                return self._org_default_roles[organization_id].get(name)

            return None

    def list_roles(
        self,
        organization_id: str,
        workspace_id: Optional[str] = None,
        include_system: bool = True,
    ) -> List[Role]:
        """List roles for an organization/workspace."""
        with self._lock:
            roles = []

            # Custom roles
            for role in self._roles.values():
                if role.organization_id != organization_id:
                    continue
                if workspace_id and role.workspace_id and role.workspace_id != workspace_id:
                    continue
                roles.append(role)

            # Default system roles
            if include_system and organization_id in self._org_default_roles:
                roles.extend(self._org_default_roles[organization_id].values())

            return roles

    def update_role(
        self,
        role_id: str,
        permissions: Optional[Set[Permission]] = None,
        description: Optional[str] = None,
    ) -> Optional[Role]:
        """Update a role."""
        with self._lock:
            role = self._roles.get(role_id)
            if not role:
                return None

            if role.is_system_role:
                raise ValueError("Cannot modify system roles")

            if permissions is not None:
                if len(permissions) > MAX_PERMISSIONS_PER_ROLE:
                    raise ValueError(f"Too many permissions (max {MAX_PERMISSIONS_PER_ROLE})")
                role.permissions = permissions

            if description is not None:
                role.description = description

            role.updated_at = datetime.now(timezone.utc)

            # Invalidate cache
            self._invalidate_cache_for_role(role_id)

            return role

    def delete_role(self, role_id: str) -> bool:
        """Delete a role."""
        with self._lock:
            role = self._roles.get(role_id)
            if not role:
                return False

            if role.is_system_role:
                raise ValueError("Cannot delete system roles")

            del self._roles[role_id]

            # Remove from workspace index
            if role.workspace_id and role.workspace_id in self._workspace_roles:
                self._workspace_roles[role.workspace_id].discard(role_id)

            # Invalidate cache
            self._invalidate_cache_for_role(role_id)

            return True

    def initialize_organization(self, organization_id: str) -> None:
        """Initialize default roles for an organization."""
        with self._lock:
            if organization_id not in self._org_default_roles:
                self._org_default_roles[organization_id] = _create_default_roles(organization_id)
                logger.info(f"Default roles initialized for organization: {organization_id}")

    # ========================================================================
    # User-Role Assignment
    # ========================================================================

    def assign_role(
        self,
        user_id: str,
        role_id: str,
    ) -> bool:
        """
        Assign a role to a user.

        Args:
            user_id: User ID
            role_id: Role ID

        Returns:
            True if assigned
        """
        with self._lock:
            # Check role exists
            if role_id not in self._roles:
                # Check default roles
                found = False
                for default_roles in self._org_default_roles.values():
                    if role_id in [r.id for r in default_roles.values()]:
                        found = True
                        break
                if not found:
                    return False

            if user_id not in self._user_roles:
                self._user_roles[user_id] = set()

            if len(self._user_roles[user_id]) >= MAX_ROLES_PER_USER:
                raise ValueError(f"User has too many roles (max {MAX_ROLES_PER_USER})")

            self._user_roles[user_id].add(role_id)

            # Invalidate cache for user
            self._invalidate_cache_for_user(user_id)

            return True

    def assign_role_by_name(
        self,
        user_id: str,
        role_name: str,
        organization_id: str,
        workspace_id: Optional[str] = None,
    ) -> bool:
        """Assign a role to a user by role name."""
        role = self.get_role_by_name(role_name, organization_id, workspace_id)
        if not role:
            return False
        return self.assign_role(user_id, role.id)

    def revoke_role(self, user_id: str, role_id: str) -> bool:
        """Revoke a role from a user."""
        with self._lock:
            if user_id not in self._user_roles:
                return False

            if role_id not in self._user_roles[user_id]:
                return False

            self._user_roles[user_id].discard(role_id)

            # Invalidate cache for user
            self._invalidate_cache_for_user(user_id)

            return True

    def get_user_roles(self, user_id: str) -> List[Role]:
        """Get all roles assigned to a user."""
        with self._lock:
            role_ids = self._user_roles.get(user_id, set())
            roles = []

            for role_id in role_ids:
                if role_id in self._roles:
                    roles.append(self._roles[role_id])
                else:
                    # Check default roles
                    for default_roles in self._org_default_roles.values():
                        for role in default_roles.values():
                            if role.id == role_id:
                                roles.append(role)
                                break

            return roles

    def get_user_permissions(
        self,
        user_id: str,
        workspace_id: Optional[str] = None,
    ) -> Set[Permission]:
        """Get all permissions for a user (union of all roles)."""
        roles = self.get_user_roles(user_id)
        permissions = set()

        for role in roles:
            # Skip workspace-specific roles that don't match
            if workspace_id and role.workspace_id and role.workspace_id != workspace_id:
                continue
            permissions.update(role.permissions)

        return permissions

    # ========================================================================
    # Permission Checking
    # ========================================================================

    def check_access(self, context: AccessContext) -> AccessDecision:
        """
        Check if access is granted for the given context.

        Args:
            context: Access context with principal, resource, and scope

        Returns:
            AccessDecision with grant/deny and details
        """
        # Try cache first
        cache_key = self._get_cache_key(context)
        if self._cache_enabled:
            cached = self._get_cached_decision(cache_key)
            if cached:
                return cached

        # Determine sensitivity and audit requirements
        resource_type = context.resource_type
        try:
            rt = ResourceType(resource_type)
            sensitivity = RESOURCE_SENSITIVITY.get(rt, SensitivityLevel.STANDARD)
            audit_required = rt in AUDIT_REQUIRED_RESOURCES
        except ValueError:
            sensitivity = SensitivityLevel.STANDARD
            audit_required = False

        # Build base decision
        decision = AccessDecision(
            sensitivity_level=sensitivity,
            audit_required=audit_required,
            permissions_checked=[f"{resource_type}:{context.scope}"],
        )

        # Check break-glass access first
        if context.principal.break_glass_id:
            decision = self._check_break_glass_access(context, decision)
            if decision.granted:
                self._cache_decision(cache_key, decision)
                self._audit_access(context, decision)
                return decision

        # Check MFA requirements for sensitive resources
        if sensitivity in (SensitivityLevel.CRITICAL, SensitivityLevel.RESTRICTED):
            if not context.principal.mfa_verified:
                decision.granted = False
                decision.reason = "MFA required for sensitive resource"
                decision.requires_mfa = True
                self._audit_access(context, decision)
                return decision

        # Check RBAC permissions
        decision = self._check_rbac_access(context, decision)

        # Cache and audit
        self._cache_decision(cache_key, decision)
        self._audit_access(context, decision)

        return decision

    def _check_rbac_access(
        self,
        context: AccessContext,
        decision: AccessDecision,
    ) -> AccessDecision:
        """Check access using RBAC."""
        principal = context.principal

        # Get user roles
        roles = self.get_user_roles(principal.id)

        # Also include roles from principal object (may be set by auth layer)
        for role_name in principal.roles:
            role = self.get_role_by_name(
                role_name,
                principal.organization_id,
                context.workspace_id,
            )
            if role and role not in roles:
                roles.append(role)

        if not roles:
            decision.granted = False
            decision.reason = "No roles assigned"
            decision.method = "denied"
            return decision

        # Check each role for the required permission
        for role in roles:
            # Skip workspace-specific roles that don't match
            if context.workspace_id and role.workspace_id:
                if role.workspace_id != context.workspace_id:
                    continue

            if role.has_permission(context.resource_type, context.scope):
                decision.granted = True
                decision.reason = "Permission granted by role"
                decision.method = "rbac"
                decision.role_used = role.name
                return decision

        decision.granted = False
        decision.reason = "No role grants required permission"
        decision.method = "denied"
        return decision

    def _check_break_glass_access(
        self,
        context: AccessContext,
        decision: AccessDecision,
    ) -> AccessDecision:
        """Check access using break-glass."""
        # Break-glass provides elevated access to specific scopes
        # The actual scope validation is done by BreakGlassService
        decision.granted = True
        decision.reason = f"Break-glass access: {context.principal.break_glass_id}"
        decision.method = "break_glass"
        decision.requires_break_glass = True
        return decision

    def has_permission(
        self,
        principal: Principal,
        resource_type: str,
        scope: str,
        workspace_id: Optional[str] = None,
    ) -> bool:
        """
        Simple permission check.

        Args:
            principal: Principal making the request
            resource_type: Resource type
            scope: Required scope
            workspace_id: Workspace context

        Returns:
            True if permission granted
        """
        context = AccessContext(
            principal=principal,
            workspace_id=workspace_id or "",
            resource_type=resource_type,
            scope=scope,
        )
        decision = self.check_access(context)
        return decision.granted

    def require_permission(
        self,
        principal: Principal,
        resource_type: str,
        scope: str,
        workspace_id: Optional[str] = None,
    ) -> AccessDecision:
        """
        Check permission and raise if denied.

        Raises:
            PermissionError: If access denied
        """
        context = AccessContext(
            principal=principal,
            workspace_id=workspace_id or "",
            resource_type=resource_type,
            scope=scope,
        )
        decision = self.check_access(context)

        if not decision.granted:
            raise PermissionError(f"Access denied: {resource_type}:{scope} - {decision.reason}")

        return decision

    # ========================================================================
    # Caching
    # ========================================================================

    def _get_cache_key(self, context: AccessContext) -> str:
        """Generate cache key for access context."""
        key_data = (
            f"{context.principal.id}:"
            f"{context.workspace_id}:"
            f"{context.resource_type}:"
            f"{context.scope}:"
            f"{context.principal.break_glass_id or ''}"
        )
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached_decision(self, cache_key: str) -> Optional[AccessDecision]:
        """Get cached decision if valid."""
        with self._lock:
            if cache_key not in self._permission_cache:
                return None

            decision, cached_at = self._permission_cache[cache_key]

            # Check TTL
            if (
                datetime.now(timezone.utc) - cached_at
            ).total_seconds() > PERMISSION_CACHE_TTL_SECONDS:
                del self._permission_cache[cache_key]
                return None

            return decision

    def _cache_decision(self, cache_key: str, decision: AccessDecision) -> None:
        """Cache access decision."""
        if not self._cache_enabled:
            return

        with self._lock:
            self._permission_cache[cache_key] = (decision, datetime.now(timezone.utc))

    def _invalidate_cache_for_user(self, user_id: str) -> None:
        """Invalidate cache entries for a user."""
        # Simple approach: clear entire cache
        # In production, use prefix-based invalidation
        self._permission_cache.clear()

    def _invalidate_cache_for_role(self, role_id: str) -> None:
        """Invalidate cache entries related to a role."""
        # Simple approach: clear entire cache
        self._permission_cache.clear()

    def clear_cache(self) -> None:
        """Clear the entire permission cache."""
        with self._lock:
            self._permission_cache.clear()

    # ========================================================================
    # Audit
    # ========================================================================

    def _audit_access(self, context: AccessContext, decision: AccessDecision) -> None:
        """Log access check to audit callback."""
        if self._audit_callback and decision.audit_required:
            try:
                self._audit_callback(context, decision)
            except Exception as e:
                logger.error(f"Audit callback error: {e}")

    # ========================================================================
    # Export
    # ========================================================================

    def export_roles(
        self,
        organization_id: str,
        workspace_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Export roles for an organization."""
        roles = self.list_roles(organization_id, workspace_id)
        return [r.to_dict() for r in roles]

    def export_user_role_assignments(
        self,
        organization_id: str,
    ) -> List[Dict[str, Any]]:
        """Export user-role assignments."""
        with self._lock:
            assignments = []
            for user_id, role_ids in self._user_roles.items():
                for role_id in role_ids:
                    role = self._roles.get(role_id)
                    if role and role.organization_id == organization_id:
                        assignments.append(
                            {
                                "user_id": user_id,
                                "role_id": role_id,
                                "role_name": role.name,
                            }
                        )
            return assignments

    def get_snapshot(
        self,
        organization_id: str,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get a point-in-time snapshot of RBAC configuration.

        For evidence pack and audit purposes.
        """
        snapshot_time = datetime.now(timezone.utc)
        roles = self.export_roles(organization_id, workspace_id)
        assignments = self.export_user_role_assignments(organization_id)

        snapshot = {
            "snapshot_id": str(uuid4()),
            "timestamp": snapshot_time.isoformat(),
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "roles": roles,
            "assignments": assignments,
            "role_count": len(roles),
            "assignment_count": len(assignments),
        }

        # Compute integrity hash
        content = json.dumps(snapshot, sort_keys=True, default=str)
        snapshot["integrity_hash"] = f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

        return snapshot


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Constants
    "PERMISSION_CACHE_TTL_SECONDS",
    "MAX_ROLES_PER_USER",
    "MAX_PERMISSIONS_PER_ROLE",
    "RESOURCE_SENSITIVITY",
    "AUDIT_REQUIRED_RESOURCES",
    # Enums
    "Scope",
    "ResourceType",
    "SensitivityLevel",
    # Data classes
    "Permission",
    "Role",
    "Principal",
    "AccessContext",
    "AccessDecision",
    # Service
    "RBACService",
]
