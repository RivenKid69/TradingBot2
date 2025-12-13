# -*- coding: utf-8 -*-
"""
CCEA Enrollment Models.

Models for agent enrollment process.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

# ============================================================================
# Patterns
# ============================================================================

AGENT_ID_PATTERN = re.compile(r"^agent_[a-zA-Z0-9]{16,32}$")
TOKEN_PATTERN = re.compile(r"^enroll_[a-zA-Z0-9]{32,64}$")


# ============================================================================
# Enums
# ============================================================================

class TrustState(str, Enum):
    """Agent trust states."""
    PENDING = "PENDING"
    ENROLLED = "ENROLLED"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"


class AgentCapability(str, Enum):
    """Agent capabilities."""
    LIVE_TRADING = "LIVE_TRADING"
    PAPER_TRADING = "PAPER_TRADING"
    BACKTEST = "BACKTEST"
    DATA_COLLECTION = "DATA_COLLECTION"


# ============================================================================
# Models
# ============================================================================

class EnrollmentToken(BaseModel):
    """
    Enrollment token with TTL.

    Used to initiate agent enrollment securely.
    """
    token_id: str = Field(
        ...,
        pattern=r"^enroll_[a-zA-Z0-9]{32,64}$",
        description="Enrollment token ID"
    )
    workspace_id: str = Field(..., description="Target workspace")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    created_by: str = Field(..., description="User who created the token")
    max_uses: int = Field(1, ge=1, description="Maximum number of uses")
    uses_count: int = Field(0, ge=0, description="Current use count")
    capabilities: List[AgentCapability] = Field(
        default_factory=lambda: [AgentCapability.PAPER_TRADING]
    )
    revoked: bool = False
    revoked_at: Optional[datetime] = None
    revoked_by: Optional[str] = None

    model_config = {"extra": "forbid"}

    @classmethod
    def create(
        cls,
        token_id: str,
        workspace_id: str,
        created_by: str,
        ttl_hours: int = 24,
        max_uses: int = 1,
        capabilities: Optional[List[AgentCapability]] = None,
    ) -> "EnrollmentToken":
        """Create a new enrollment token with TTL."""
        now = datetime.utcnow()
        return cls(
            token_id=token_id,
            workspace_id=workspace_id,
            created_at=now,
            expires_at=now + timedelta(hours=ttl_hours),
            created_by=created_by,
            max_uses=max_uses,
            capabilities=capabilities or [AgentCapability.PAPER_TRADING],
        )

    def is_valid(self) -> bool:
        """Check if token is still valid."""
        if self.revoked:
            return False
        if datetime.utcnow() > self.expires_at:
            return False
        if self.uses_count >= self.max_uses:
            return False
        return True


class EnrollmentRequest(BaseModel):
    """
    Agent enrollment request.

    Sent by agent to register with cloud.
    Agent generates keypair locally and sends public key.
    """
    token: str = Field(
        ...,
        pattern=r"^enroll_[a-zA-Z0-9]{32,64}$",
        description="Enrollment token"
    )
    public_key: str = Field(
        ...,
        description="Agent's public key (PEM format)"
    )
    public_key_algorithm: str = Field(
        "ed25519",
        description="Public key algorithm"
    )
    agent_version: str = Field(..., description="Agent software version")
    hostname: Optional[str] = Field(None, description="Agent hostname (optional)")
    capabilities: List[AgentCapability] = Field(
        default_factory=lambda: [AgentCapability.PAPER_TRADING]
    )
    # Metadata for audit
    requested_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"extra": "forbid"}


class EnrollmentResponse(BaseModel):
    """
    Enrollment response from cloud.

    Contains assigned agent_id and configuration.
    """
    success: bool
    agent_id: Optional[str] = Field(
        None,
        pattern=r"^agent_[a-zA-Z0-9]{16,32}$",
        description="Assigned agent ID"
    )
    workspace_id: Optional[str] = None
    enrolled_at: Optional[datetime] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    # Configuration
    heartbeat_interval_seconds: int = Field(30, ge=5, le=300)
    command_poll_interval_seconds: int = Field(5, ge=1, le=60)
    supported_schema_versions: List[str] = Field(
        default_factory=lambda: ["1.0.0"]
    )
    # Endpoints
    heartbeat_endpoint: Optional[str] = None
    commands_endpoint: Optional[str] = None
    telemetry_endpoint: Optional[str] = None

    model_config = {"extra": "forbid"}

    @classmethod
    def success_response(
        cls,
        agent_id: str,
        workspace_id: str,
        heartbeat_endpoint: str,
        commands_endpoint: str,
        telemetry_endpoint: str,
    ) -> "EnrollmentResponse":
        """Create a successful enrollment response."""
        return cls(
            success=True,
            agent_id=agent_id,
            workspace_id=workspace_id,
            enrolled_at=datetime.utcnow(),
            heartbeat_endpoint=heartbeat_endpoint,
            commands_endpoint=commands_endpoint,
            telemetry_endpoint=telemetry_endpoint,
        )

    @classmethod
    def error_response(
        cls,
        error_code: str,
        error_message: str,
    ) -> "EnrollmentResponse":
        """Create an error enrollment response."""
        return cls(
            success=False,
            error_code=error_code,
            error_message=error_message,
        )


class AgentInfo(BaseModel):
    """
    Agent information stored in cloud.

    Contains agent metadata and trust state.
    """
    agent_id: str = Field(
        ...,
        pattern=r"^agent_[a-zA-Z0-9]{16,32}$"
    )
    workspace_id: str
    public_key: str = Field(..., description="Agent's public key (PEM)")
    public_key_algorithm: str = "ed25519"
    agent_version: str
    hostname: Optional[str] = None
    capabilities: List[AgentCapability] = Field(default_factory=list)
    trust_state: TrustState = TrustState.ENROLLED
    enrolled_at: datetime
    enrolled_by_token: str
    last_seen: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    revoked_by: Optional[str] = None
    revoke_reason: Optional[str] = None

    model_config = {"extra": "forbid"}

    def is_active(self) -> bool:
        """Check if agent is active and enrolled."""
        return self.trust_state == TrustState.ENROLLED

    def revoke(self, revoked_by: str, reason: str) -> None:
        """Revoke agent trust."""
        self.trust_state = TrustState.REVOKED
        self.revoked_at = datetime.utcnow()
        self.revoked_by = revoked_by
        self.revoke_reason = reason
