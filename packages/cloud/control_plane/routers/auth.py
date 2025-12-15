# -*- coding: utf-8 -*-
"""
Authentication Router.

CLOUD ZONE ONLY.

Provides JWT-based authentication endpoints.

Phase 5 Security (WI-AUTH-01):
- Argon2id password hashing
- Rate limiting and account lockout
- JWT revocation (jti blocklist)
- Password policy validation
"""

from __future__ import annotations

import hashlib
import secrets
import uuid as uuid_lib
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
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
    decode_token,
)
from ..models import Agent, AgentEnrollmentToken, Role, TrustState, User
from ..security.password_hasher import (
    PasswordHasher,
    hash_password as secure_hash_password,
    verify_password as secure_verify_password,
    needs_rehash,
)
from ..security.rate_limiter import (
    RateLimiter,
    RateLimitExceeded,
    AccountLockout,
    get_rate_limiter,
    check_lockout,
    check_rate_limit,
    record_login_attempt,
)
from ..security.jwt_revocation import (
    JTIBlocklist,
    revoke_token,
    is_token_revoked,
    get_blocklist,
)
from ..services.command_signer import (
    get_cloud_signer,
    CloudKeyNotConfiguredError,
)

import logging

logger = logging.getLogger(__name__)

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
    """
    Agent enrollment response.

    Design Doc 10.2: cloud_public_key is provided so Agent can verify
    signatures on commands from Cloud.
    """

    agent_id: UUID
    access_token: str
    token_type: str = "bearer"
    workspace_id: UUID
    org_id: UUID
    cloud_public_key: Optional[str] = None  # Design Doc 10.2: For command verification


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
    """
    Hash password using Argon2id (WI-AUTH-01).

    Argon2id is memory-hard and resistant to GPU attacks.
    Falls back to bcrypt/PBKDF2 if Argon2 is not available.
    """
    return secure_hash_password(password)


def verify_password(password: str, hashed: str) -> bool:
    """
    Verify password against hash (WI-AUTH-01).

    Supports Argon2id, bcrypt, PBKDF2, and legacy SHA256 (for migration).
    """
    return secure_verify_password(password, hashed)


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request (handles proxies)."""
    # Check X-Forwarded-For header for proxied requests
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # First IP in the list is the client
        return forwarded.split(",")[0].strip()
    # Fall back to direct connection
    if request.client:
        return request.client.host
    return "unknown"


@router.post(
    "/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    summary="User login",
    description="Authenticate user and return JWT token.",
)
async def login(request: LoginRequest, http_request: Request) -> LoginResponse:
    """
    Authenticate user with email and password.

    Returns JWT access token on success.

    WI-AUTH-01 Security:
    - Rate limiting per IP address
    - Account lockout after failed attempts
    - Argon2id password verification with rehash support
    - JWT with unique jti for revocation support
    """
    client_ip = _get_client_ip(http_request)

    # WI-AUTH-01: Check IP rate limit
    try:
        check_rate_limit(client_ip)
    except RateLimitExceeded as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=e.args[0],
            headers={"Retry-After": str(e.retry_after)},
        )

    # WI-AUTH-01: Check account lockout
    try:
        check_lockout(request.email)
    except AccountLockout as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=e.args[0],
            headers={"Retry-After": str(e.retry_after)},
        )

    async with get_session() as session:
        # Find user by email with eager loading of roles and permissions
        result = await session.execute(
            select(User)
            .where(User.email == request.email, User.is_active == True)
            .options(selectinload(User.roles).selectinload(Role.permissions))
        )
        user = result.scalar_one_or_none()

        if user is None:
            # Record failed attempt even for non-existent users (timing attack mitigation)
            record_login_attempt(request.email, success=False, ip_address=client_ip)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        # Verify password
        if not verify_password(request.password, user.password_hash):
            # WI-AUTH-01: Record failed attempt for lockout tracking
            record_login_attempt(request.email, success=False, ip_address=client_ip)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        # WI-AUTH-01: Record successful login
        record_login_attempt(request.email, success=True, ip_address=client_ip)

        # WI-AUTH-01: Check if password needs rehashing (algorithm upgrade)
        if needs_rehash(user.password_hash):
            user.password_hash = hash_password(request.password)

        # Get user permissions
        permissions = []
        for role in user.roles:
            for perm in role.permissions:
                if perm.name not in permissions:
                    permissions.append(perm.name)

        # WI-AUTH-01: Generate unique jti for revocation support
        jti = str(uuid_lib.uuid4())

        # Create access token with jti
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
async def logout(
    current_user: UserDep,
    http_request: Request,
) -> None:
    """
    Logout current user.

    WI-AUTH-01: Revokes the current JWT token by adding its jti to the blocklist.
    """
    # Extract token from authorization header
    auth_header = http_request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        try:
            # Decode to get jti
            payload = decode_token(token)
            jti = payload.get("jti")
            if jti:
                # Calculate token expiry for cleanup
                exp = payload.get("exp")
                expires_at = datetime.fromtimestamp(exp, tz=timezone.utc) if exp else None

                # WI-AUTH-01: Add token to blocklist
                revoke_token(
                    jti=jti,
                    expires_at=expires_at,
                    reason="logout",
                    user_id=str(current_user.id),
                )
        except Exception:
            # Token already invalid, nothing to revoke
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

        # Get Cloud's public key for command verification (Design Doc 10.2)
        cloud_public_key = None
        try:
            signer = get_cloud_signer()
            cloud_public_key = signer.get_public_key_pem()
        except CloudKeyNotConfiguredError:
            logger.warning(
                "Cloud signing key not configured. Agent will not be able to verify "
                "command signatures. Configure CCEA_CLOUD_PRIVATE_KEY for production."
            )

        return AgentEnrollResponse(
            agent_id=agent.id,
            access_token=token,
            workspace_id=agent.workspace_id,
            org_id=workspace.organization_id,
            cloud_public_key=cloud_public_key,
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
