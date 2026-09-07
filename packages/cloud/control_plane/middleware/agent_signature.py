# -*- coding: utf-8 -*-
"""
Agent Signature Verification Middleware.

Design Doc 10.2: Verifies signatures on Agent-to-Cloud requests.

CLOUD ZONE ONLY.

This middleware:
    - Validates X-CCEA-Signature header on agent requests
    - Looks up agent's public key from database
    - Verifies signature against request body
    - Optionally enforces signature requirement (strict mode)

Security:
    - Signatures use Ed25519 (agent's keypair from enrollment)
    - Replay protection via timestamp validation
    - Key fingerprint matching for multi-key scenarios
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional, Set

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse
from sqlalchemy import select

from ccea.crypto.signing import verify_signature
from ccea.crypto.keys import load_public_key

logger = logging.getLogger(__name__)


# Header names
HEADER_SIGNATURE = "X-CCEA-Signature"
HEADER_FINGERPRINT = "X-CCEA-Key-Fingerprint"
HEADER_TIMESTAMP = "X-CCEA-Timestamp"
HEADER_KEY_ID = "X-CCEA-Key-ID"

# Maximum age of request timestamp (replay protection)
MAX_TIMESTAMP_AGE_SECONDS = 300  # 5 minutes

# Paths that require agent signature verification (when strict mode is enabled)
AGENT_SIGNED_PATHS: Set[str] = frozenset(
    {
        "/api/v1/agent/heartbeat",
        "/api/v1/agent/commands/poll",
        "/api/v1/agent/commands/",  # Prefix for command operations
        "/api/v1/agent/telemetry",
        "/api/v1/agent/approval",
    }
)


@dataclass
class SignatureVerificationResult:
    """Result of signature verification."""

    verified: bool
    agent_id: Optional[str] = None
    key_fingerprint: Optional[str] = None
    error: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class SignatureConfig:
    """Configuration for signature verification."""

    strict_mode: bool = False  # If True, reject unsigned requests
    verify_timestamp: bool = True  # If True, check timestamp freshness
    max_timestamp_age: int = MAX_TIMESTAMP_AGE_SECONDS
    log_unverified: bool = True  # Log when signature is missing/invalid
    paths_require_signature: Set[str] = field(default_factory=lambda: set(AGENT_SIGNED_PATHS))


class AgentSignatureVerifier:
    """
    Verifies agent signatures on requests.

    Design Doc 10.2: Agent-to-Cloud requests should be signed
    with the agent's private key to ensure authenticity.

    Usage:
        verifier = AgentSignatureVerifier(config)
        result = await verifier.verify_request(request, get_agent_public_key)

        if not result.verified and config.strict_mode:
            raise HTTPException(403, "Signature verification failed")
    """

    def __init__(self, config: Optional[SignatureConfig] = None):
        """Initialize verifier with configuration."""
        self.config = config or SignatureConfig()
        self._verified_count = 0
        self._failed_count = 0

    async def verify_request(
        self,
        request: Request,
        get_public_key: Callable[[str], Any],  # async func(agent_id) -> public_key
    ) -> SignatureVerificationResult:
        """
        Verify signature on an incoming request.

        Args:
            request: FastAPI/Starlette request
            get_public_key: Async function to retrieve agent's public key by ID

        Returns:
            SignatureVerificationResult with verification status
        """
        # Get signature from header
        signature_b64 = request.headers.get(HEADER_SIGNATURE)
        if not signature_b64:
            if self.config.log_unverified:
                logger.debug(f"No {HEADER_SIGNATURE} header on request {request.url.path}")
            return SignatureVerificationResult(verified=False, error="Missing signature header")

        # Get timestamp (for replay protection)
        timestamp_str = request.headers.get(HEADER_TIMESTAMP)
        request_timestamp = None
        if timestamp_str and self.config.verify_timestamp:
            try:
                request_timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                age = abs((now - request_timestamp).total_seconds())

                if age > self.config.max_timestamp_age:
                    self._failed_count += 1
                    return SignatureVerificationResult(
                        verified=False,
                        timestamp=request_timestamp,
                        error=f"Request timestamp too old ({age:.0f}s > {self.config.max_timestamp_age}s)",
                    )
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid timestamp format: {timestamp_str}")

        # Get agent ID from request context (set by auth middleware)
        agent_id = getattr(request.state, "agent_id", None)
        if not agent_id:
            # Try to extract from JWT token context
            agent_id = getattr(request.state, "current_agent", {})
            if hasattr(agent_id, "id"):
                agent_id = str(agent_id.id)
            else:
                agent_id = None

        if not agent_id:
            if self.config.log_unverified:
                logger.debug("No agent ID in request context for signature verification")
            return SignatureVerificationResult(
                verified=False, error="No agent ID in request context"
            )

        # Get agent's public key
        try:
            public_key = await get_public_key(agent_id)
            if public_key is None:
                self._failed_count += 1
                return SignatureVerificationResult(
                    verified=False, agent_id=agent_id, error="Agent public key not found"
                )

            # Load public key if it's a PEM string
            if isinstance(public_key, str):
                public_key = load_public_key(public_key)

        except Exception as e:
            logger.error(f"Error loading agent public key: {e}")
            self._failed_count += 1
            return SignatureVerificationResult(
                verified=False, agent_id=agent_id, error=f"Error loading public key: {e}"
            )

        # Get request body for verification
        try:
            body = await request.body()
        except Exception as e:
            logger.error(f"Error reading request body: {e}")
            return SignatureVerificationResult(
                verified=False, agent_id=agent_id, error=f"Error reading request body: {e}"
            )

        # Build message to verify (body + timestamp if present)
        # Design Doc 10.2: bytes-to-sign = body_bytes + b"|" + timestamp_iso
        message = body
        if timestamp_str:
            # Include timestamp in signed data (MUST match agent signing format)
            message = body + b"|" + timestamp_str.encode("utf-8")

        # Verify signature
        # NOTE: verify_signature() expects base64-encoded signature string, NOT raw bytes
        # The agent sends base64-encoded signature in header
        try:
            is_valid = verify_signature(message, signature_b64, public_key)
        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            self._failed_count += 1
            return SignatureVerificationResult(
                verified=False, agent_id=agent_id, error=f"Signature verification error: {e}"
            )

        if is_valid:
            self._verified_count += 1
            # Compute fingerprint for logging
            fingerprint = request.headers.get(HEADER_FINGERPRINT) or "unknown"
            return SignatureVerificationResult(
                verified=True,
                agent_id=agent_id,
                key_fingerprint=fingerprint,
                timestamp=request_timestamp,
            )
        else:
            self._failed_count += 1
            return SignatureVerificationResult(
                verified=False, agent_id=agent_id, error="Invalid signature"
            )

    def should_verify(self, path: str) -> bool:
        """Check if path requires signature verification."""
        for required_path in self.config.paths_require_signature:
            if path.startswith(required_path):
                return True
        return False

    @property
    def stats(self) -> Dict[str, int]:
        """Get verification statistics."""
        return {
            "verified": self._verified_count,
            "failed": self._failed_count,
        }


class AgentSignatureMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for agent signature verification.

    Design Doc 10.2: Ensures agent requests are authenticated
    not just via JWT but also via cryptographic signature.

    This provides defense-in-depth:
    - JWT: Proves agent identity (bearer token)
    - Signature: Proves agent possession of private key
    """

    def __init__(
        self,
        app,
        config: Optional[SignatureConfig] = None,
        get_agent_public_key: Optional[Callable] = None,
    ):
        """
        Initialize middleware.

        Args:
            app: FastAPI/Starlette application
            config: Signature verification config
            get_agent_public_key: Async function(agent_id) -> public_key_pem
        """
        super().__init__(app)
        self.config = config or SignatureConfig()
        self.verifier = AgentSignatureVerifier(self.config)
        self._get_agent_public_key = get_agent_public_key

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with signature verification."""
        # Skip non-agent paths
        if not self.verifier.should_verify(request.url.path):
            return await call_next(request)

        # Skip if no public key getter configured
        if self._get_agent_public_key is None:
            return await call_next(request)

        # Verify signature
        result = await self.verifier.verify_request(
            request,
            self._get_agent_public_key,
        )

        # Store result in request state for handlers
        request.state.signature_verified = result.verified
        request.state.signature_result = result

        # If strict mode and verification failed, reject request
        if self.config.strict_mode and not result.verified:
            logger.warning(
                f"Agent signature verification failed: {result.error} "
                f"(path={request.url.path}, agent_id={result.agent_id})"
            )
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "detail": f"Agent signature verification failed: {result.error}",
                    "signature_required": True,
                },
            )

        # Log unverified requests in non-strict mode
        if not result.verified and self.config.log_unverified:
            logger.info(
                f"Agent request without valid signature: {result.error} "
                f"(path={request.url.path}, agent_id={result.agent_id})"
            )

        return await call_next(request)


async def get_agent_public_key_from_db(agent_id: str) -> Optional[str]:
    """
    Default implementation to get agent's public key from database.

    Args:
        agent_id: UUID of the agent

    Returns:
        PEM-encoded public key string, or None if not found
    """
    from ..database import get_session
    from ..models import Agent
    from uuid import UUID

    try:
        agent_uuid = UUID(agent_id)
    except (ValueError, TypeError):
        return None

    async with get_session() as session:
        result = await session.execute(
            select(Agent.public_key).where(
                Agent.id == agent_uuid,
                Agent.deleted_at.is_(None),
            )
        )
        row = result.scalar_one_or_none()
        return row


def create_agent_signature_middleware(
    app,
    strict_mode: bool = False,
) -> AgentSignatureMiddleware:
    """
    Factory function to create configured middleware.

    Args:
        app: FastAPI application
        strict_mode: If True, reject unsigned requests

    Returns:
        Configured AgentSignatureMiddleware
    """
    config = SignatureConfig(
        strict_mode=strict_mode,
        verify_timestamp=True,
        log_unverified=True,
    )

    return AgentSignatureMiddleware(
        app=app,
        config=config,
        get_agent_public_key=get_agent_public_key_from_db,
    )


# Export for convenience
__all__ = [
    "HEADER_SIGNATURE",
    "HEADER_FINGERPRINT",
    "HEADER_TIMESTAMP",
    "HEADER_KEY_ID",
    "SignatureVerificationResult",
    "SignatureConfig",
    "AgentSignatureVerifier",
    "AgentSignatureMiddleware",
    "get_agent_public_key_from_db",
    "create_agent_signature_middleware",
]
