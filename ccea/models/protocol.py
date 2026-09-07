# -*- coding: utf-8 -*-
"""
CCEA Protocol Models.

Pydantic models for Cloud-Agent protocol messages.
Implements JSON schema constraints from protocol_messages.schema.json.

SECURITY: Order-like fields are PROHIBITED in all messages.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# ============================================================================
# Type Patterns (from JSON schema)
# ============================================================================

IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{16,64}$")
AGENT_ID_PATTERN = re.compile(r"^agent_[a-zA-Z0-9]{16,32}$")
DIGEST_PATTERN = re.compile(r"^sha256:[a-f0-9]{64}$")


# ============================================================================
# Enums
# ============================================================================


class SignatureAlgorithm(str, Enum):
    """Supported signature algorithms."""

    ED25519 = "ed25519"
    ECDSA_P256 = "ecdsa-p256"
    RSA_PSS = "rsa-pss"


class CommandStatus(str, Enum):
    """Command processing status."""

    PENDING = "PENDING"
    RECEIVED = "RECEIVED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ApprovalStatus(str, Enum):
    """Approval result status."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    TIMEOUT = "TIMEOUT"


class RunState(str, Enum):
    """Run state machine states."""

    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    HALTED = "HALTED"
    STOPPED = "STOPPED"


class DeploymentState(str, Enum):
    """Deployment state machine states."""

    CREATED = "CREATED"
    PENDING = "PENDING"
    ENROLLED = "ENROLLED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    REVOKED = "REVOKED"
    TERMINATED = "TERMINATED"


class ChangeClass(str, Enum):
    """Change classification for approval requirements."""

    TRADING_IMPACTING = "TRADING_IMPACTING"
    NON_TRADING_IMPACTING = "NON_TRADING_IMPACTING"


class TelemetryLevel(str, Enum):
    """Telemetry detail levels."""

    AGGREGATED = "AGGREGATED"
    DETAILED_NON_SENSITIVE = "DETAILED_NON_SENSITIVE"
    RAW_ORDER_EVENTS = "RAW_ORDER_EVENTS"


class CommandType(str, Enum):
    """Allowed command types (Design Doc E2)."""

    REQUEST_START_RUN = "REQUEST_START_RUN"
    REQUEST_STOP_RUN = "REQUEST_STOP_RUN"
    REQUEST_PAUSE_RUN = "REQUEST_PAUSE_RUN"
    REQUEST_UPGRADE_ARTIFACT = "REQUEST_UPGRADE_ARTIFACT"
    REQUEST_UPDATE_CONFIG = "REQUEST_UPDATE_CONFIG"
    REQUEST_ROTATE_AGENT_SESSION = "REQUEST_ROTATE_AGENT_SESSION"
    REQUEST_EXPORT_LOGS = "REQUEST_EXPORT_LOGS"


class MessageType(str, Enum):
    """Allowed message types."""

    HEARTBEAT = "HEARTBEAT"
    POLL_COMMANDS = "POLL_COMMANDS"
    COMMAND_BATCH = "COMMAND_BATCH"
    COMMAND_ACK = "COMMAND_ACK"
    COMMAND_APPROVAL = "COMMAND_APPROVAL"
    COMMAND_RESULT = "COMMAND_RESULT"
    TELEMETRY = "TELEMETRY"


# ============================================================================
# Base Types
# ============================================================================


class IdempotencyKey(str):
    """Idempotency key with validation."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: str) -> "IdempotencyKey":
        if not IDEMPOTENCY_KEY_PATTERN.match(v):
            raise ValueError(
                f"Invalid idempotency key format. Must match pattern: {IDEMPOTENCY_KEY_PATTERN.pattern}"
            )
        return cls(v)


class AgentId(str):
    """Agent identifier with validation."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: str) -> "AgentId":
        if not AGENT_ID_PATTERN.match(v):
            raise ValueError(
                f"Invalid agent_id format. Must match pattern: {AGENT_ID_PATTERN.pattern}"
            )
        return cls(v)


class Digest(str):
    """SHA256 digest with validation."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: str) -> "Digest":
        if not DIGEST_PATTERN.match(v):
            raise ValueError(f"Invalid digest format. Must match pattern: {DIGEST_PATTERN.pattern}")
        return cls(v)


class Signature(BaseModel):
    """Cryptographic signature."""

    algorithm: SignatureAlgorithm
    value: str = Field(..., description="Base64-encoded signature")
    key_id: Optional[str] = Field(None, description="Key identifier")

    model_config = {"extra": "forbid"}


# ============================================================================
# Prohibited Fields Validator
# ============================================================================

PROHIBITED_FIELDS = frozenset(
    {
        "side",
        "quantity",
        "qty",
        "price",
        "order_type",
        "target_position",
        "execute_order",
        "place_order",
        "submit_order",
        "intent",
        "signal",
    }
)


def check_no_order_fields(values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validator to ensure no order-like fields exist.

    SECURITY: This is a critical guardrail.
    """
    for field in PROHIBITED_FIELDS:
        if field in values:
            raise ValueError(
                f"PROHIBITED: Field '{field}' is not allowed in protocol messages. "
                "Order-like payloads are forbidden per Design Doc 10.5/19.2."
            )
    return values


# ============================================================================
# State Models
# ============================================================================


class AgentState(BaseModel):
    """Agent state information in heartbeat."""

    deployment_state: Optional[DeploymentState] = None
    run_state: Optional[RunState] = None
    agent_version: Optional[str] = None
    uptime_seconds: Optional[int] = Field(None, ge=0)
    last_command_id: Optional[str] = None

    model_config = {"extra": "forbid"}


class AgentHealth(BaseModel):
    """Agent health metrics."""

    cpu_percent: Optional[float] = Field(None, ge=0, le=100)
    memory_percent: Optional[float] = Field(None, ge=0, le=100)
    disk_percent: Optional[float] = Field(None, ge=0, le=100)
    broker_connected: Optional[bool] = None
    data_feed_ok: Optional[bool] = None

    model_config = {"extra": "forbid"}


class VersionNegotiation(BaseModel):
    """
    Version negotiation for Cloud-Agent protocol compatibility.

    Uses min_supported/max_supported per Design Doc.
    """

    min_supported: str = Field(
        ...,
        pattern=r"^\d+\.\d+\.\d+$",
        description="Minimum supported schema version (MAJOR.MINOR.PATCH)",
    )
    max_supported: str = Field(
        ...,
        pattern=r"^\d+\.\d+\.\d+$",
        description="Maximum supported schema version (MAJOR.MINOR.PATCH)",
    )
    preferred: Optional[str] = Field(
        None, pattern=r"^\d+\.\d+\.\d+$", description="Preferred schema version if within range"
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_version_range(self) -> "VersionNegotiation":
        """Ensure max_supported >= min_supported."""
        from ccea.protocol.schema_versioning import SchemaVersion

        min_ver = SchemaVersion.parse(self.min_supported)
        max_ver = SchemaVersion.parse(self.max_supported)
        if max_ver < min_ver:
            raise ValueError(
                f"max_supported ({self.max_supported}) must be >= "
                f"min_supported ({self.min_supported})"
            )
        return self


# ============================================================================
# Base Message Model
# ============================================================================


class BaseMessage(BaseModel):
    """Base model for all messages."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    signature: Optional[Signature] = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def validate_no_order_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        return check_no_order_fields(values)


class BaseCommand(BaseMessage):
    """Base model for all commands."""

    command_type: CommandType
    idempotency_key: str = Field(..., pattern=r"^[a-zA-Z0-9_-]{16,64}$")

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, v: str) -> str:
        if not IDEMPOTENCY_KEY_PATTERN.match(v):
            raise ValueError("Invalid idempotency key format")
        return v


# ============================================================================
# Agent -> Cloud Messages
# ============================================================================


class HeartbeatMessage(BaseMessage):
    """
    Heartbeat message from agent to cloud.

    Design Doc: HEARTBEAT includes agent state, health metrics, and capabilities.
    Capabilities advertise what features the agent supports.
    """

    message_type: Literal[MessageType.HEARTBEAT] = MessageType.HEARTBEAT
    agent_id: str = Field(..., pattern=r"^agent_[a-zA-Z0-9]{16,32}$")
    state: AgentState
    health: Optional[AgentHealth] = None
    capabilities: Optional[List[str]] = Field(
        None,
        description="Agent capabilities: e.g., 'live_trading', 'paper_trading', 'backtesting', 'gpu'",
    )


class PollCommandsMessage(BaseMessage):
    """Poll commands request from agent."""

    message_type: Literal[MessageType.POLL_COMMANDS] = MessageType.POLL_COMMANDS
    agent_id: str = Field(..., pattern=r"^agent_[a-zA-Z0-9]{16,32}$")
    last_command_id: Optional[str] = None
    version_negotiation: Optional[VersionNegotiation] = Field(
        None, description="Version negotiation using min_supported/max_supported per Design Doc"
    )


class CommandAckMessage(BaseMessage):
    """Command acknowledgment from agent."""

    message_type: Literal[MessageType.COMMAND_ACK] = MessageType.COMMAND_ACK
    agent_id: str = Field(..., pattern=r"^agent_[a-zA-Z0-9]{16,32}$")
    command_id: str
    idempotency_key: str = Field(..., pattern=r"^[a-zA-Z0-9_-]{16,64}$")
    status: CommandStatus
    message: Optional[str] = None


class CommandApprovalMessage(BaseMessage):
    """Command approval from agent."""

    message_type: Literal[MessageType.COMMAND_APPROVAL] = MessageType.COMMAND_APPROVAL
    agent_id: str = Field(..., pattern=r"^agent_[a-zA-Z0-9]{16,32}$")
    command_id: str
    approval_status: ApprovalStatus
    approver: Optional[str] = None
    evidence_hash: Optional[str] = Field(None, pattern=r"^sha256:[a-f0-9]{64}$")
    reason: Optional[str] = None


class CommandResultMessage(BaseMessage):
    """Command result from agent."""

    message_type: Literal[MessageType.COMMAND_RESULT] = MessageType.COMMAND_RESULT
    agent_id: str = Field(..., pattern=r"^agent_[a-zA-Z0-9]{16,32}$")
    command_id: str
    idempotency_key: Optional[str] = Field(None, pattern=r"^[a-zA-Z0-9_-]{16,64}$")
    status: CommandStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, str]] = None


class TelemetryEvent(BaseModel):
    """Single telemetry event."""

    event_type: str
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None

    model_config = {"extra": "forbid"}


class TelemetryMessage(BaseMessage):
    """Telemetry message from agent."""

    message_type: Literal[MessageType.TELEMETRY] = MessageType.TELEMETRY
    agent_id: str = Field(..., pattern=r"^agent_[a-zA-Z0-9]{16,32}$")
    level: TelemetryLevel
    events: List[TelemetryEvent]
    redaction_applied: Literal[True] = Field(
        True, description="MUST always be true - redaction is mandatory"
    )


# ============================================================================
# Cloud -> Agent Commands
# ============================================================================


class RequestStartRunCommand(BaseCommand):
    """Request to start a run (TRADING_IMPACTING)."""

    command_type: Literal[CommandType.REQUEST_START_RUN] = CommandType.REQUEST_START_RUN
    deployment_id: str
    artifact_digest: str = Field(..., pattern=r"^sha256:[a-f0-9]{64}$")
    config_digest: Optional[str] = Field(None, pattern=r"^sha256:[a-f0-9]{64}$")
    requires_approval: Literal[True] = True
    change_class: Literal[ChangeClass.TRADING_IMPACTING] = ChangeClass.TRADING_IMPACTING


class RequestStopRunCommand(BaseCommand):
    """Request to stop a run (safety - no approval needed)."""

    command_type: Literal[CommandType.REQUEST_STOP_RUN] = CommandType.REQUEST_STOP_RUN
    deployment_id: str
    reason: Optional[str] = None
    requires_approval: Literal[False] = False


class RequestPauseRunCommand(BaseCommand):
    """Request to pause a run (safety - no approval needed)."""

    command_type: Literal[CommandType.REQUEST_PAUSE_RUN] = CommandType.REQUEST_PAUSE_RUN
    deployment_id: str
    reason: Optional[str] = None
    requires_approval: Literal[False] = False


class RequestUpgradeArtifactCommand(BaseCommand):
    """Request to upgrade artifact (TRADING_IMPACTING)."""

    command_type: Literal[CommandType.REQUEST_UPGRADE_ARTIFACT] = (
        CommandType.REQUEST_UPGRADE_ARTIFACT
    )
    deployment_id: str
    new_artifact_digest: str = Field(..., pattern=r"^sha256:[a-f0-9]{64}$")
    old_artifact_digest: Optional[str] = Field(None, pattern=r"^sha256:[a-f0-9]{64}$")
    requires_approval: Literal[True] = True
    change_class: Literal[ChangeClass.TRADING_IMPACTING] = ChangeClass.TRADING_IMPACTING


class RequestUpdateConfigCommand(BaseCommand):
    """Request to update config."""

    command_type: Literal[CommandType.REQUEST_UPDATE_CONFIG] = CommandType.REQUEST_UPDATE_CONFIG
    deployment_id: str
    new_config_digest: str = Field(..., pattern=r"^sha256:[a-f0-9]{64}$")
    old_config_digest: Optional[str] = Field(None, pattern=r"^sha256:[a-f0-9]{64}$")
    change_class: ChangeClass
    requires_approval: bool = Field(..., description="True if change_class is TRADING_IMPACTING")

    @model_validator(mode="after")
    def validate_approval_requirement(self) -> "RequestUpdateConfigCommand":
        if self.change_class == ChangeClass.TRADING_IMPACTING and not self.requires_approval:
            raise ValueError("TRADING_IMPACTING changes must require approval")
        return self


class RequestRotateAgentSessionCommand(BaseCommand):
    """Request to rotate agent session."""

    command_type: Literal[CommandType.REQUEST_ROTATE_AGENT_SESSION] = (
        CommandType.REQUEST_ROTATE_AGENT_SESSION
    )
    agent_id: str = Field(..., pattern=r"^agent_[a-zA-Z0-9]{16,32}$")
    reason: Optional[str] = None
    requires_approval: Literal[True] = True


class ExportConfig(BaseModel):
    """Log export configuration."""

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    log_types: Optional[List[Literal["audit", "telemetry", "errors", "commands"]]] = None
    redaction_level: Literal["FULL", "STANDARD"] = "FULL"

    model_config = {"extra": "forbid"}


class RequestExportLogsCommand(BaseCommand):
    """Request to export logs (DATA_SENSITIVE)."""

    command_type: Literal[CommandType.REQUEST_EXPORT_LOGS] = CommandType.REQUEST_EXPORT_LOGS
    agent_id: str = Field(..., pattern=r"^agent_[a-zA-Z0-9]{16,32}$")
    export_config: ExportConfig
    requires_approval: Literal[True] = True
    break_glass_reason: Optional[str] = Field(None, description="Required reason for data export")


# ============================================================================
# Cloud -> Agent Batch Message
# ============================================================================

Command = Union[
    RequestStartRunCommand,
    RequestStopRunCommand,
    RequestPauseRunCommand,
    RequestUpgradeArtifactCommand,
    RequestUpdateConfigCommand,
    RequestRotateAgentSessionCommand,
    RequestExportLogsCommand,
]


class CommandBatchMessage(BaseMessage):
    """Batch of commands from cloud to agent."""

    message_type: Literal[MessageType.COMMAND_BATCH] = MessageType.COMMAND_BATCH
    commands: List[Command]


# ============================================================================
# Union of all messages
# ============================================================================

Message = Union[
    HeartbeatMessage,
    PollCommandsMessage,
    CommandBatchMessage,
    CommandAckMessage,
    CommandApprovalMessage,
    CommandResultMessage,
    TelemetryMessage,
]
