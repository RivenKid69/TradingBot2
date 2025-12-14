# -*- coding: utf-8 -*-
"""
FastAPI Dependencies for Cloud Control Plane.

CLOUD ZONE ONLY.

Provides dependency injection for:
- Authentication (JWT)
- Authorization (RBAC)
- Tenant context
- Database sessions
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Optional, Set
from uuid import UUID

import jwt
from fastapi import Depends, Header, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .database import TenantContext, get_session
from .models import Agent, Permission, Role, TrustState, User, Workspace


# JWT Configuration
JWT_SECRET = os.getenv("CCEA_JWT_SECRET", "dev-secret-change-in-production")
JWT_ALGORITHM = os.getenv("CCEA_JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_HOURS = int(os.getenv("CCEA_JWT_EXPIRATION_HOURS", "24"))

# Security scheme
security = HTTPBearer(auto_error=False)


class TokenData(BaseModel):
    """JWT token payload."""

    user_id: UUID
    email: str
    workspace_id: Optional[UUID] = None
    org_id: Optional[UUID] = None
    permissions: List[str] = Field(default_factory=list)
    exp: datetime
    iat: datetime
    jti: Optional[str] = None  # JWT ID for revocation


class AgentTokenData(BaseModel):
    """Agent authentication token payload."""

    agent_id: UUID
    workspace_id: UUID
    org_id: UUID
    capabilities: List[str] = Field(default_factory=list)
    exp: datetime
    iat: datetime
    jti: Optional[str] = None


class CurrentUser(BaseModel):
    """Current authenticated user context."""

    id: UUID
    email: str
    workspace_id: Optional[UUID] = None
    org_id: Optional[UUID] = None
    permissions: Set[str] = Field(default_factory=set)
    is_superuser: bool = False

    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        if self.is_superuser:
            return True
        return permission in self.permissions

    def has_any_permission(self, permissions: List[str]) -> bool:
        """Check if user has any of the specified permissions."""
        if self.is_superuser:
            return True
        return bool(self.permissions.intersection(permissions))

    def has_all_permissions(self, permissions: List[str]) -> bool:
        """Check if user has all specified permissions."""
        if self.is_superuser:
            return True
        return set(permissions).issubset(self.permissions)


class CurrentAgent(BaseModel):
    """Current authenticated agent context."""

    id: UUID
    workspace_id: UUID
    org_id: UUID
    capabilities: List[str] = Field(default_factory=list)
    trust_state: TrustState = TrustState.ENROLLED

    def has_capability(self, capability: str) -> bool:
        """Check if agent has a specific capability."""
        return capability in self.capabilities


def create_access_token(
    user_id: UUID,
    email: str,
    workspace_id: Optional[UUID] = None,
    org_id: Optional[UUID] = None,
    permissions: Optional[List[str]] = None,
    expires_delta: Optional[timedelta] = None,
    is_superuser: bool = False,
) -> str:
    """
    Create JWT access token for user.

    Args:
        user_id: User ID
        email: User email
        workspace_id: Current workspace ID
        org_id: Organization ID
        permissions: List of permission names
        expires_delta: Token expiration delta

    Returns:
        Encoded JWT token
    """
    now = datetime.now(timezone.utc)
    expires = now + (expires_delta or timedelta(hours=JWT_EXPIRATION_HOURS))

    payload = {
        "sub": str(user_id),
        "email": email,
        "workspace_id": str(workspace_id) if workspace_id else None,
        "org_id": str(org_id) if org_id else None,
        "permissions": permissions or [],
        "is_superuser": is_superuser,
        "exp": expires,
        "iat": now,
    }

    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_agent_token(
    agent_id: UUID,
    workspace_id: UUID,
    org_id: UUID,
    capabilities: Optional[List[str]] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create JWT token for agent authentication.

    Args:
        agent_id: Agent ID
        workspace_id: Workspace ID
        org_id: Organization ID
        capabilities: Agent capabilities
        expires_delta: Token expiration delta

    Returns:
        Encoded JWT token
    """
    now = datetime.now(timezone.utc)
    expires = now + (expires_delta or timedelta(hours=JWT_EXPIRATION_HOURS))

    payload = {
        "sub": str(agent_id),
        "type": "agent",
        "workspace_id": str(workspace_id),
        "org_id": str(org_id),
        "capabilities": capabilities or [],
        "exp": expires,
        "iat": now,
    }

    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate JWT token.

    Args:
        token: JWT token string

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials], Depends(security)
    ] = None,
) -> CurrentUser:
    """
    Get current authenticated user from JWT token.

    Args:
        credentials: HTTP Bearer credentials

    Returns:
        Current user context

    Raises:
        HTTPException: If not authenticated or invalid token
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(credentials.credentials)

    # Check if this is an agent token
    if payload.get("type") == "agent":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Agent token not allowed for user endpoints",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return CurrentUser(
        id=UUID(payload["sub"]),
        email=payload["email"],
        workspace_id=UUID(payload["workspace_id"]) if payload.get("workspace_id") else None,
        org_id=UUID(payload["org_id"]) if payload.get("org_id") else None,
        permissions=set(payload.get("permissions", [])),
        is_superuser=payload.get("is_superuser", False),
    )


async def get_current_agent(
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials], Depends(security)
    ] = None,
) -> CurrentAgent:
    """
    Get current authenticated agent from JWT token.

    Args:
        credentials: HTTP Bearer credentials

    Returns:
        Current agent context

    Raises:
        HTTPException: If not authenticated or invalid token
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(credentials.credentials)

    # Check if this is an agent token
    if payload.get("type") != "agent":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User token not allowed for agent endpoints",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return CurrentAgent(
        id=UUID(payload["sub"]),
        workspace_id=UUID(payload["workspace_id"]),
        org_id=UUID(payload["org_id"]),
        capabilities=payload.get("capabilities", []),
        trust_state=TrustState.ENROLLED,  # Will be verified against DB
    )


async def get_optional_user(
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials], Depends(security)
    ] = None,
) -> Optional[CurrentUser]:
    """
    Get current user if authenticated, None otherwise.

    Args:
        credentials: HTTP Bearer credentials

    Returns:
        Current user context or None
    """
    if credentials is None:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def require_permission(permission: str):
    """
    Dependency to require a specific permission.

    Usage:
        @app.get("/admin")
        async def admin_endpoint(
            user: CurrentUser = Depends(require_permission("admin:read"))
        ):
            ...
    """

    async def _require_permission(
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
    ) -> CurrentUser:
        if not current_user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission}",
            )
        return current_user

    return _require_permission


def require_any_permission(permissions: List[str]):
    """
    Dependency to require any of the specified permissions.

    Usage:
        @app.get("/items")
        async def list_items(
            user: CurrentUser = Depends(require_any_permission(["items:read", "admin"]))
        ):
            ...
    """

    async def _require_any_permission(
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
    ) -> CurrentUser:
        if not current_user.has_any_permission(permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of these permissions required: {', '.join(permissions)}",
            )
        return current_user

    return _require_any_permission


async def get_workspace_session(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> AsyncSession:
    """
    Get database session with tenant context from current user.

    Args:
        current_user: Current authenticated user

    Yields:
        Database session with tenant context
    """
    if current_user.workspace_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No workspace selected",
        )

    tenant_context = TenantContext(current_user.workspace_id)
    async with get_session(tenant_context) as session:
        yield session


async def get_admin_session() -> AsyncSession:
    """
    Get database session without tenant context (for admin operations).

    Yields:
        Database session without tenant filtering
    """
    async with get_session() as session:
        yield session


# Type aliases for FastAPI dependencies
UserDep = Annotated[CurrentUser, Depends(get_current_user)]
AgentDep = Annotated[CurrentAgent, Depends(get_current_agent)]
OptionalUserDep = Annotated[Optional[CurrentUser], Depends(get_optional_user)]
WorkspaceSessionDep = Annotated[AsyncSession, Depends(get_workspace_session)]
AdminSessionDep = Annotated[AsyncSession, Depends(get_admin_session)]


# Pagination
class PaginationParams(BaseModel):
    """Common pagination parameters."""

    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")

    @property
    def offset(self) -> int:
        """Calculate offset for database query."""
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        """Get limit for database query."""
        return self.page_size


def get_pagination(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
) -> PaginationParams:
    """Get pagination parameters from query."""
    return PaginationParams(page=page, page_size=page_size)


PaginationDep = Annotated[PaginationParams, Depends(get_pagination)]
