# -*- coding: utf-8 -*-
"""
Authentication Router.

CLOUD ZONE ONLY.

Provides JWT-based authentication endpoints.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..database import get_session
from ..dependencies import (
    JWT_EXPIRATION_HOURS,
    UserDep,
    create_access_token,
    create_agent_token,
)
from ..models import Agent, AgentEnrollmentToken, Role, TrustState, User

router = APIRouter()


# Request/Response models
class LoginRequest(BaseModel):
    """User login request."""

    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    """User login response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = JWT_EXPIRATION_HOURS * 3600
    user_id: UUID
    email: str
    workspace_id: Optional[UUID] = None


class RefreshRequest(BaseModel):
    """Token refresh request."""

    refresh_token: str


class RefreshResponse(BaseModel):
    """Token refresh response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = JWT_EXPIRATION_HOURS * 3600


class AgentEnrollRequest(BaseModel):
    """Agent enrollment request."""

    enrollment_token: str
    agent_name: str = Field(..., min_length=1, max_length=255)
    public_key: str = Field(..., min_length=64, max_length=4096)
    agent_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+.*$")
    capabilities: list[str] = Field(default_factory=list)
    attestation: Optional[str] = None


class AgentEnrollResponse(BaseModel):
    """Agent enrollment response."""

    agent_id: UUID
    access_token: str
    token_type: str = "bearer"
    workspace_id: UUID
    org_id: UUID


class AgentHeartbeatRequest(BaseModel):
    """Agent heartbeat request."""

    agent_version: str
    current_state: str
    last_run_id: Optional[UUID] = None
    health_metrics: dict = Field(default_factory=dict)


class AgentHeartbeatResponse(BaseModel):
    """Agent heartbeat response."""

    server_time: datetime
    trust_state: str
    pending_commands: int = 0
    next_heartbeat_sec: int = 60


def hash_password(password: str) -> str:
    """Hash password with SHA256 (for demo; use bcrypt in production)."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash."""
    return hash_password(password) == hashed


@router.post(
    "/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    summary="User login",
    description="Authenticate user and return JWT token.",
)
async def login(request: LoginRequest) -> LoginResponse:
    """
    Authenticate user with email and password.

    Returns JWT access token on success.
    """
    async with get_session() as session:
        # Find user by email with eager loading of roles and permissions
        result = await session.execute(
            select(User)
            .where(User.email == request.email, User.is_active == True)
            .options(selectinload(User.roles).selectinload(Role.permissions))
        )
        user = result.scalar_one_or_none()

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        # Verify password
        if not verify_password(request.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        # Get user permissions
        permissions = []
        for role in user.roles:
            for perm in role.permissions:
                if perm.name not in permissions:
                    permissions.append(perm.name)

        # Create access token
        token = create_access_token(
            user_id=user.id,
            email=user.email,
            workspace_id=user.default_workspace_id,
            org_id=user.organization_id,
            permissions=permissions,
        )

        # Update last login
        user.last_login = datetime.now(timezone.utc)
        await session.commit()

        return LoginResponse(
            access_token=token,
            user_id=user.id,
            email=user.email,
            workspace_id=user.default_workspace_id,
        )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="User logout",
    description="Invalidate current session.",
)
async def logout(current_user: UserDep) -> None:
    """
    Logout current user.

    In a production system, this would invalidate the JWT
    by adding it to a blocklist or rotating refresh tokens.
    """
    # TODO: Add JWT to blocklist for revocation
    pass


@router.get(
    "/me",
    response_model=dict,
    summary="Get current user",
    description="Get information about the current authenticated user.",
)
async def get_me(current_user: UserDep) -> dict:
    """Get current user information."""
    return {
        "id": str(current_user.id),
        "email": current_user.email,
        "workspace_id": str(current_user.workspace_id) if current_user.workspace_id else None,
        "org_id": str(current_user.org_id) if current_user.org_id else None,
        "permissions": list(current_user.permissions),
        "is_superuser": current_user.is_superuser,
    }


@router.post(
    "/agent/enroll",
    response_model=AgentEnrollResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll agent",
    description="Enroll a new agent using an enrollment token.",
)
async def enroll_agent(request: AgentEnrollRequest) -> AgentEnrollResponse:
    """
    Enroll a new agent.

    The agent must provide a valid enrollment token obtained
    from the workspace admin. The token is single-use and expires.
    """
    async with get_session() as session:
        # Hash the provided token
        token_hash = hashlib.sha256(request.enrollment_token.encode()).hexdigest()

        # Find enrollment token
        result = await session.execute(
            select(AgentEnrollmentToken).where(
                AgentEnrollmentToken.token_hash == token_hash,
                AgentEnrollmentToken.is_used == False,
                AgentEnrollmentToken.expires_at > datetime.now(timezone.utc),
            )
        )
        enrollment_token = result.scalar_one_or_none()

        if enrollment_token is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired enrollment token",
            )

        # Create agent
        agent = Agent(
            name=request.agent_name,
            workspace_id=enrollment_token.workspace_id,
            public_key=request.public_key,
            agent_version=request.agent_version,
            capabilities=request.capabilities,
            trust_state=TrustState.ENROLLED.value,
            attestation=request.attestation,
            last_seen_at=datetime.now(timezone.utc),
        )
        session.add(agent)

        # Mark token as used
        enrollment_token.is_used = True
        enrollment_token.used_at = datetime.now(timezone.utc)
        enrollment_token.agent_id = agent.id

        await session.commit()
        await session.refresh(agent)

        # Get workspace to find org_id
        from ..models import Workspace

        ws_result = await session.execute(
            select(Workspace).where(Workspace.id == agent.workspace_id)
        )
        workspace = ws_result.scalar_one()

        # Create agent token
        token = create_agent_token(
            agent_id=agent.id,
            workspace_id=agent.workspace_id,
            org_id=workspace.organization_id,
            capabilities=agent.capabilities,
        )

        return AgentEnrollResponse(
            agent_id=agent.id,
            access_token=token,
            workspace_id=agent.workspace_id,
            org_id=workspace.organization_id,
        )


@router.post(
    "/agent/heartbeat",
    response_model=AgentHeartbeatResponse,
    summary="Agent heartbeat",
    description="Report agent status and receive pending commands count.",
)
async def agent_heartbeat(
    request: AgentHeartbeatRequest,
) -> AgentHeartbeatResponse:
    """
    Agent heartbeat endpoint.

    Updates agent's last_seen timestamp and returns server time
    and any pending commands count.
    """
    # Note: In production, this would use AgentDep for authentication
    # For now, we return a simple response

    return AgentHeartbeatResponse(
        server_time=datetime.now(timezone.utc),
        trust_state=TrustState.ENROLLED.value,
        pending_commands=0,
        next_heartbeat_sec=60,
    )
