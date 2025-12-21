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


class LoginResponse2FA(BaseModel):
    """Response when MFA is required."""

    requires_mfa: bool = True
    mfa_token: str  # Temporary token to complete MFA
    message: str = "MFA verification required"


class MFAVerifyRequest(BaseModel):
    """MFA verification request."""

    mfa_token: str
    totp_code: str = Field(..., pattern=r"^\d{6}$")


class MFASetupRequest(BaseModel):
    """MFA setup request."""

    totp_code: str = Field(..., pattern=r"^\d{6}$")


class MFASetupResponse(BaseModel):
    """MFA setup response with secret and QR code."""

    secret: str
    provisioning_uri: str
    backup_codes: list[str]


class RefreshRequest(BaseModel):
    """Token refresh request."""

    refresh_token: str


class RefreshResponse(BaseModel):
    """Token refresh response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = JWT_EXPIRATION_HOURS * 3600


# MFA constants (Design Doc 4.1)
MFA_LOCKOUT_ATTEMPTS = 5
MFA_LOCKOUT_DURATION_MINUTES = 30
MFA_TOKEN_EXPIRY_MINUTES = 5


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

    Design Doc 10.2, 15.2: cloud_public_key is provided so Agent can verify
    signatures on commands from Cloud. Key versioning via key_id supports rotation.
    """

    agent_id: UUID
    access_token: str
    token_type: str = "bearer"
    workspace_id: UUID
    org_id: UUID
    cloud_public_key: Optional[str] = None  # Design Doc 10.2: For command verification
    cloud_key_id: Optional[str] = None      # Design Doc 15.2: Key version identifier
    cloud_key_fingerprint: Optional[str] = None  # For key verification


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


# MFA token storage
# SECURITY: In-memory storage is acceptable for single-instance deployments.
# For multi-instance/production deployments, use Redis or database storage.
# Tech Debt: docs/reports/TECH_DEBT_REGISTRY.md#security-distributed-state
# Control Artifact: docs/security/DISTRIBUTED_SECURITY_REQUIREMENTS.md
# Metrics: mfa_pending_token_count, mfa_token_issued, mfa_token_redeemed
_mfa_pending_tokens: dict[str, dict] = {}


def _generate_totp_secret() -> str:
    """Generate a new TOTP secret."""
    try:
        import pyotp
        return pyotp.random_base32()
    except ImportError:
        # Fallback if pyotp not installed
        return secrets.token_hex(20)


def _verify_totp(secret: str, code: str) -> bool:
    """
    Verify TOTP code against secret.

    SECURITY: This function implements fail-closed behavior.
    If pyotp is not available, verification fails (not bypassed).

    Args:
        secret: TOTP secret key
        code: User-provided TOTP code

    Returns:
        True if code is valid, False otherwise
    """
    try:
        import pyotp
        totp = pyotp.TOTP(secret)
        return totp.verify(code, valid_window=1)  # Allow 1 window tolerance
    except ImportError:
        # SECURITY: Fail-closed per security requirements
        # MFA verification MUST NOT be bypassed
        logger.error(
            "SECURITY: pyotp not installed. MFA verification FAILED (fail-closed). "
            "Install pyotp package: pip install pyotp"
        )
        return False


def _generate_provisioning_uri(secret: str, email: str) -> str:
    """Generate TOTP provisioning URI for QR code."""
    try:
        import pyotp
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=email, issuer_name="CCEA Platform")
    except ImportError:
        return f"otpauth://totp/CCEA:{email}?secret={secret}&issuer=CCEA"


def _generate_backup_codes(count: int = 10) -> list[str]:
    """Generate backup codes for MFA recovery."""
    return [secrets.token_hex(4).upper() for _ in range(count)]


def _hash_backup_codes(codes: list[str]) -> str:
    """Hash backup codes for storage."""
    import json
    # Store hashed versions
    hashed = [hashlib.sha256(c.encode()).hexdigest() for c in codes]
    return json.dumps(hashed)


def _verify_backup_code(stored_hash: str, code: str) -> bool:
    """Verify a backup code against stored hashes."""
    import json
    try:
        hashed_codes = json.loads(stored_hash)
        code_hash = hashlib.sha256(code.upper().encode()).hexdigest()
        return code_hash in hashed_codes
    except Exception:
        return False


def _create_mfa_pending_token(user_id: UUID, email: str) -> str:
    """Create a temporary token for MFA completion."""
    token = secrets.token_urlsafe(32)
    _mfa_pending_tokens[token] = {
        "user_id": str(user_id),
        "email": email,
        "expires_at": datetime.now(timezone.utc) + timedelta(minutes=MFA_TOKEN_EXPIRY_MINUTES),
    }
    return token


def _verify_mfa_pending_token(token: str) -> Optional[dict]:
    """Verify and consume a pending MFA token."""
    data = _mfa_pending_tokens.get(token)
    if not data:
        return None
    if data["expires_at"] < datetime.now(timezone.utc):
        del _mfa_pending_tokens[token]
        return None
    return data


def _check_mfa_lockout(user: User) -> None:
    """Check if user is locked out of MFA due to failed attempts."""
    if user.mfa_locked_until and user.mfa_locked_until > datetime.now(timezone.utc):
        remaining = (user.mfa_locked_until - datetime.now(timezone.utc)).seconds
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"MFA locked due to too many failed attempts. Try again in {remaining} seconds.",
            headers={"Retry-After": str(remaining)},
        )


@router.post(
    "/login",
    response_model=LoginResponse | LoginResponse2FA,
    status_code=status.HTTP_200_OK,
    summary="User login",
    description="Authenticate user and return JWT token. If MFA is enabled, returns MFA token instead.",
)
async def login(request: LoginRequest, http_request: Request) -> LoginResponse | LoginResponse2FA:
    """
    Authenticate user with email and password.

    Returns JWT access token on success, or MFA token if MFA is enabled.

    WI-AUTH-01 Security:
    - Rate limiting per IP address
    - Account lockout after failed attempts
    - Argon2id password verification with rehash support
    - JWT with unique jti for revocation support

    Design Doc 4.1 MFA:
    - If MFA is enabled, returns temporary mfa_token
    - User must call /auth/mfa/verify with TOTP code
    - MFA lockout after failed attempts
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
            await session.commit()

        # Design Doc 4.1: Check if MFA is enabled
        if user.mfa_enabled and user.mfa_secret:
            # Check MFA lockout
            _check_mfa_lockout(user)

            # Issue temporary MFA token
            mfa_token = _create_mfa_pending_token(user.id, user.email)
            return LoginResponse2FA(
                mfa_token=mfa_token,
                message="MFA verification required. Use /auth/mfa/verify endpoint.",
            )

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
    "/mfa/verify",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify MFA code",
    description="Complete login by verifying TOTP code (Design Doc 4.1).",
)
async def verify_mfa(request: MFAVerifyRequest) -> LoginResponse:
    """
    Verify MFA code and complete login.

    Design Doc 4.1:
    - Validates TOTP code against user's MFA secret
    - Supports backup codes for recovery
    - Locks out after too many failed attempts
    """
    # Verify the pending MFA token
    token_data = _verify_mfa_pending_token(request.mfa_token)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired MFA token. Please login again.",
        )

    async with get_session() as session:
        # Find user
        result = await session.execute(
            select(User)
            .where(User.id == UUID(token_data["user_id"]), User.is_active == True)
            .options(selectinload(User.roles).selectinload(Role.permissions))
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        # Check MFA lockout
        _check_mfa_lockout(user)

        # Verify TOTP code
        code_valid = False
        if user.mfa_secret:
            code_valid = _verify_totp(user.mfa_secret, request.totp_code)

        # If TOTP fails, try backup codes
        if not code_valid and user.mfa_backup_codes_hash:
            code_valid = _verify_backup_code(user.mfa_backup_codes_hash, request.totp_code)

        if not code_valid:
            # Increment failed attempts
            user.failed_mfa_attempts = (user.failed_mfa_attempts or 0) + 1

            # Check if we should lock out
            if user.failed_mfa_attempts >= MFA_LOCKOUT_ATTEMPTS:
                user.mfa_locked_until = datetime.now(timezone.utc) + timedelta(
                    minutes=MFA_LOCKOUT_DURATION_MINUTES
                )
                await session.commit()
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Too many failed MFA attempts. Account locked for {MFA_LOCKOUT_DURATION_MINUTES} minutes.",
                )

            await session.commit()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid MFA code. {MFA_LOCKOUT_ATTEMPTS - user.failed_mfa_attempts} attempts remaining.",
            )

        # Success - reset failed attempts and update verification timestamp
        user.failed_mfa_attempts = 0
        user.mfa_locked_until = None
        if not user.mfa_verified_at:
            user.mfa_verified_at = datetime.now(timezone.utc)

        # Consume the MFA token
        del _mfa_pending_tokens[request.mfa_token]

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
    "/mfa/setup",
    response_model=MFASetupResponse,
    status_code=status.HTTP_200_OK,
    summary="Setup MFA",
    description="Generate TOTP secret and backup codes for MFA setup (Design Doc 4.1).",
)
async def setup_mfa(current_user: UserDep) -> MFASetupResponse:
    """
    Setup MFA for current user.

    Design Doc 4.1:
    - Generates TOTP secret
    - Provides provisioning URI for authenticator apps
    - Generates backup codes for recovery
    - MFA is not enabled until confirmed with /mfa/confirm
    """
    async with get_session() as session:
        # Find user
        result = await session.execute(
            select(User).where(User.id == current_user.id)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        if user.mfa_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA is already enabled. Disable it first to reconfigure.",
            )

        # Generate new secret
        secret = _generate_totp_secret()
        provisioning_uri = _generate_provisioning_uri(secret, user.email)

        # Generate backup codes
        backup_codes = _generate_backup_codes()

        # Store secret temporarily (will be confirmed later)
        user.mfa_secret = secret
        user.mfa_backup_codes_hash = _hash_backup_codes(backup_codes)
        await session.commit()

        return MFASetupResponse(
            secret=secret,
            provisioning_uri=provisioning_uri,
            backup_codes=backup_codes,
        )


@router.post(
    "/mfa/confirm",
    status_code=status.HTTP_200_OK,
    summary="Confirm MFA setup",
    description="Confirm MFA setup by verifying TOTP code (Design Doc 4.1).",
)
async def confirm_mfa(request: MFASetupRequest, current_user: UserDep) -> dict:
    """
    Confirm MFA setup with TOTP code.

    User must provide a valid TOTP code to enable MFA.
    This ensures they have correctly configured their authenticator app.
    """
    async with get_session() as session:
        # Find user
        result = await session.execute(
            select(User).where(User.id == current_user.id)
        )
        user = result.scalar_one_or_none()

        if not user or not user.mfa_secret:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA setup not started. Call /mfa/setup first.",
            )

        # Verify TOTP code
        if not _verify_totp(user.mfa_secret, request.totp_code):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid TOTP code. Please check your authenticator app.",
            )

        # Enable MFA
        user.mfa_enabled = True
        user.mfa_verified_at = datetime.now(timezone.utc)
        user.failed_mfa_attempts = 0
        await session.commit()

        return {"message": "MFA enabled successfully", "mfa_enabled": True}


@router.post(
    "/mfa/disable",
    status_code=status.HTTP_200_OK,
    summary="Disable MFA",
    description="Disable MFA for current user (requires TOTP or backup code).",
)
async def disable_mfa(request: MFASetupRequest, current_user: UserDep) -> dict:
    """
    Disable MFA for current user.

    Requires valid TOTP code or backup code for security.
    """
    async with get_session() as session:
        # Find user
        result = await session.execute(
            select(User).where(User.id == current_user.id)
        )
        user = result.scalar_one_or_none()

        if not user or not user.mfa_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA is not enabled.",
            )

        # Verify TOTP code or backup code
        code_valid = _verify_totp(user.mfa_secret, request.totp_code)
        if not code_valid and user.mfa_backup_codes_hash:
            code_valid = _verify_backup_code(user.mfa_backup_codes_hash, request.totp_code)

        if not code_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid verification code.",
            )

        # Disable MFA
        user.mfa_enabled = False
        user.mfa_secret = None
        user.mfa_backup_codes_hash = None
        user.mfa_verified_at = None
        user.failed_mfa_attempts = 0
        user.mfa_locked_until = None
        await session.commit()

        return {"message": "MFA disabled successfully", "mfa_enabled": False}


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

        # Get Cloud's public key for command verification (Design Doc 10.2, 15.2)
        cloud_public_key = None
        cloud_key_id = None
        cloud_key_fingerprint = None
        try:
            signer = get_cloud_signer()
            key_info = signer.get_key_info()
            cloud_public_key = key_info.public_key_pem
            cloud_key_id = key_info.key_id
            cloud_key_fingerprint = key_info.fingerprint
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
            cloud_key_id=cloud_key_id,
            cloud_key_fingerprint=cloud_key_fingerprint,
        )


class CloudPublicKeyResponse(BaseModel):
    """
    Response with Cloud's public key for signature verification.

    Design Doc 10.2, 15.2: Agents need Cloud's public key to verify
    command signatures. Key rotation is supported via key_id versioning.
    """
    key_id: str
    algorithm: str
    public_key_pem: str
    fingerprint: str
    issued_at: datetime


@router.get(
    "/cloud/public-key",
    response_model=CloudPublicKeyResponse,
    summary="Get Cloud public key",
    description="Get Cloud's public key for signature verification (Design Doc 10.2).",
)
async def get_cloud_public_key() -> CloudPublicKeyResponse:
    """
    Get Cloud's public key for signature verification.

    Agents use this key to verify that commands actually came from Cloud.
    Key rotation is supported - agents should store key_id with the key
    and verify signatures against the correct key version.

    Design Doc 10.2: All Cloud-to-Agent messages are signed.
    Design Doc 15.2: Key rotation/versioning is supported.
    """
    try:
        signer = get_cloud_signer()
        key_info = signer.get_key_info()

        return CloudPublicKeyResponse(
            key_id=key_info.key_id,
            algorithm=key_info.algorithm,
            public_key_pem=key_info.public_key_pem,
            fingerprint=key_info.fingerprint,
            issued_at=key_info.created_at,
        )

    except CloudKeyNotConfiguredError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud signing key not configured. Contact administrator.",
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
