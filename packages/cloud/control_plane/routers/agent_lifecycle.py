# -*- coding: utf-8 -*-
"""
Agent Lifecycle Router - Authenticated endpoints for agent operations.

Phase 7 (WI-CLOUD-02): Implements agent-auth lifecycle endpoints.
Phase 10: Protocol versioning negotiation (Design Doc 10.3/10.4)

CLOUD ZONE ONLY.

Provides authenticated endpoints for:
- Version negotiation (Design Doc 10.3/10.4)
- Heartbeat with accurate pending command count
- Command polling (long-poll ready)
- Command acknowledgement
- Command result submission
- Approval submission (for local approval mode)

Security:
- All endpoints require AgentDep authentication
- Agent must be enrolled (trust_state = ENROLLED)
- Commands are workspace-scoped
- Version negotiation MUST complete before commands are accepted

References:
- Design Doc: Agent polling and command delivery
- Design Doc 10.3/10.4: Protocol version negotiation
- WI-CLOUD-02: Implement agent-auth lifecycle endpoints
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..dependencies import AgentDep, CurrentAgent
from ..models import (
    Agent,
    ApprovalRecord,
    Command,
    CommandStatus,
    TrustState,
)
from ..services.agent_service import (
    AgentService,
    AgentNotFoundError,
    AgentNotEnrolledError,
    HeartbeatRequest as ServiceHeartbeatRequest,
)
from ..services.command_service import (
    CommandService,
    CommandAckRequest,
    CommandNotFoundError,
    CommandResultRequest,
    CommandStateError,
)
from ..services.command_signer import (
    get_cloud_signer,
    CloudKeyNotConfiguredError,
)
from ccea.protocol.schema_versioning import (
    SchemaVersionNegotiator,
    VersionNegotiationRequest,
    NegotiationStatus,
    CURRENT_SCHEMA_VERSION,
    MIN_SUPPORTED_VERSION,
    MAX_SUPPORTED_VERSION,
)


router = APIRouter()


# ============================================================================
# Version Negotiator Singleton
# ============================================================================

# Cloud-side version negotiator (Design Doc 10.3/10.4)
_version_negotiator = SchemaVersionNegotiator(
    min_supported=MIN_SUPPORTED_VERSION,
    max_supported=MAX_SUPPORTED_VERSION,
    prefer_latest=True,
)

# Store negotiated versions per agent
_negotiated_versions: Dict[str, str] = {}


# ============================================================================
# Request/Response Models
# ============================================================================

# ----- Version Negotiation Models (Design Doc 10.3/10.4) -----

class VersionNegotiateRequest(BaseModel):
    """
    Version negotiation request from Agent.

    Design Doc 10.3/10.4: Agent sends supported version range,
    Cloud responds with selected version or error.
    """
    min_supported: str = Field(
        ...,
        pattern=r"^\d+\.\d+\.\d+$",
        description="Agent's minimum supported protocol version"
    )
    max_supported: str = Field(
        ...,
        pattern=r"^\d+\.\d+\.\d+$",
        description="Agent's maximum supported protocol version"
    )
    preferred: Optional[str] = Field(
        None,
        pattern=r"^\d+\.\d+\.\d+$",
        description="Agent's preferred version (optional)"
    )

    @field_validator("max_supported")
    @classmethod
    def validate_version_range(cls, v: str, info) -> str:
        """Ensure max_supported >= min_supported."""
        min_ver_str = info.data.get("min_supported")
        if min_ver_str:
            # Parse and compare
            min_parts = [int(x) for x in min_ver_str.split(".")]
            max_parts = [int(x) for x in v.split(".")]
            if max_parts < min_parts:
                raise ValueError(
                    f"max_supported ({v}) must be >= min_supported ({min_ver_str})"
                )
        return v


class VersionNegotiateResponse(BaseModel):
    """
    Version negotiation response to Agent.

    Design Doc 10.3/10.4: Cloud returns selected version or error.
    Agent MUST verify version before proceeding.
    """
    status: str = Field(..., description="PENDING, SUCCESS, FAILED, TIMEOUT")
    selected_version: Optional[str] = Field(None, description="Negotiated protocol version")
    cloud_min: str = Field(..., description="Cloud's minimum supported version")
    cloud_max: str = Field(..., description="Cloud's maximum supported version")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    negotiated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentHeartbeatRequest(BaseModel):
    """Agent heartbeat request."""
    agent_version: str = Field(..., min_length=1, max_length=50)
    current_state: str = Field(..., min_length=1, max_length=50)
    last_run_id: Optional[UUID] = None
    health_metrics: Dict[str, Any] = Field(default_factory=dict)


class AgentHeartbeatResponse(BaseModel):
    """Agent heartbeat response."""
    server_time: datetime
    trust_state: str
    pending_commands: int
    next_heartbeat_sec: int


class CommandPollResponse(BaseModel):
    """Response from command polling."""
    commands: List["PendingCommand"]
    has_more: bool
    poll_again_after_sec: int
    # Design Doc 10.3/10.4: Include negotiated version in response
    protocol_version: Optional[str] = Field(None, description="Negotiated protocol version")


class PendingCommand(BaseModel):
    """
    Pending command for agent.

    Design Doc 10.2: All commands MUST have signature field.
    Agent MUST verify signature before processing.
    """
    id: UUID
    status: str
    idempotency_key: str
    command_type: str
    payload_ref: str
    change_class: str
    requires_approval: bool
    deployment_id: Optional[UUID] = None
    run_id: Optional[UUID] = None
    expires_at: Optional[datetime] = None
    created_at: datetime
    timestamp: Optional[datetime] = None  # For replay protection
    signature: Optional[Dict[str, Any]] = None  # Design Doc 10.2: Command signature


class CommandAckResponse(BaseModel):
    """Response from command acknowledgement."""
    command_id: UUID
    status: str
    acknowledged_at: datetime


class CommandResultSubmission(BaseModel):
    """Command execution result submission."""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class CommandResultResponse(BaseModel):
    """Response from result submission."""
    command_id: UUID
    status: str
    executed_at: Optional[datetime] = None


class LocalApprovalRequest(BaseModel):
    """
    Local approval request from agent.

    Design Doc 6.2, 12.2: Structured approval evidence with immutable blob references.
    """
    approved: bool
    evidence_hash: Optional[str] = Field(None, max_length=128)
    attestation: Optional[Dict[str, Any]] = None
    diff_summary: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    # Design Doc 6.2, 12.2: Immutable blob references
    config_blob_digest: Optional[str] = Field(
        None, max_length=128, description="ConfigBlob digest at approval time"
    )
    manifest_digest: Optional[str] = Field(
        None, max_length=128, description="Artifact manifest digest at approval time"
    )
    previous_state_digest: Optional[str] = Field(
        None, max_length=128, description="Previous configuration state digest"
    )
    new_state_digest: Optional[str] = Field(
        None, max_length=128, description="New configuration state digest"
    )


class LocalApprovalResponse(BaseModel):
    """Response from local approval."""
    command_id: UUID
    status: str
    approved: bool


# ============================================================================
# Agent Verification
# ============================================================================

async def verify_agent_enrolled(
    session: AsyncSession,
    agent: CurrentAgent,
) -> Agent:
    """
    Verify agent exists and is enrolled.

    Args:
        session: Database session
        agent: Current agent from token

    Returns:
        Agent from database

    Raises:
        HTTPException: If agent not found or not enrolled
    """
    result = await session.execute(
        select(Agent).where(
            Agent.id == agent.id,
            Agent.workspace_id == agent.workspace_id,
            Agent.deleted_at.is_(None),
        )
    )
    db_agent = result.scalar_one_or_none()

    if db_agent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )

    if db_agent.trust_state == TrustState.REVOKED.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Agent has been revoked",
        )

    if db_agent.trust_state == TrustState.SUSPENDED.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Agent is suspended",
        )

    if db_agent.trust_state != TrustState.ENROLLED.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Agent not enrolled. Current state: {db_agent.trust_state}",
        )

    return db_agent


async def verify_version_negotiated(
    agent_id: str,
    x_protocol_version: Optional[str] = Header(None, alias="X-Protocol-Version"),
) -> str:
    """
    Verify version has been negotiated for this agent.

    Design Doc 10.3/10.4: Version MUST be negotiated before commands.
    Returns negotiated version or raises 428 Precondition Required.

    Args:
        agent_id: Agent identifier
        x_protocol_version: Protocol version header from agent

    Returns:
        Negotiated version string

    Raises:
        HTTPException 428: If version not negotiated
        HTTPException 400: If version mismatch
    """
    negotiated = _negotiated_versions.get(agent_id)

    if not negotiated:
        raise HTTPException(
            status_code=status.HTTP_428_PRECONDITION_REQUIRED,
            detail={
                "error": "version_not_negotiated",
                "message": "Protocol version negotiation required. Call POST /negotiate-version first.",
                "cloud_min": MIN_SUPPORTED_VERSION,
                "cloud_max": MAX_SUPPORTED_VERSION,
            }
        )

    # Verify header matches negotiated version if provided
    if x_protocol_version and x_protocol_version != negotiated:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "version_mismatch",
                "message": f"Protocol version mismatch. Header: {x_protocol_version}, Negotiated: {negotiated}",
                "negotiated_version": negotiated,
            }
        )

    return negotiated


# ============================================================================
# Version Negotiation Endpoint (Design Doc 10.3/10.4)
# ============================================================================

@router.post(
    "/negotiate-version",
    response_model=VersionNegotiateResponse,
    summary="Negotiate protocol version",
    description="Negotiate protocol version between Agent and Cloud. MUST be called before polling commands.",
)
async def negotiate_version(
    request: VersionNegotiateRequest,
    current_agent: AgentDep,
) -> VersionNegotiateResponse:
    """
    Negotiate protocol version with Agent.

    Design Doc 10.3/10.4:
    - Agent sends supported version range
    - Cloud responds with selected version or error
    - Negotiation MUST complete before any commands are accepted
    - Selected version is cached for subsequent requests

    Security:
    - Version negotiation is MANDATORY before command operations
    - Mismatch in versions results in negotiation failure
    - Agent MUST verify returned version is within its supported range
    """
    async with get_session() as session:
        # Verify agent is enrolled
        await verify_agent_enrolled(session, current_agent)

        # Create internal negotiation request
        agent_id_str = f"agent_{str(current_agent.id).replace('-', '')[:24]}"
        internal_request = VersionNegotiationRequest(
            agent_id=agent_id_str,
            min_supported=request.min_supported,
            max_supported=request.max_supported,
            preferred=request.preferred,
        )

        # Perform negotiation
        result = _version_negotiator.negotiate(internal_request)

        # Cache successful negotiation
        if result.is_success() and result.selected_version:
            _negotiated_versions[str(current_agent.id)] = str(result.selected_version)

            logger.info(
                "Protocol version negotiated",
                extra={
                    "agent_id": str(current_agent.id),
                    "selected_version": str(result.selected_version),
                    "agent_range": f"[{request.min_supported}-{request.max_supported}]",
                    "cloud_range": f"[{MIN_SUPPORTED_VERSION}-{MAX_SUPPORTED_VERSION}]",
                }
            )

        return VersionNegotiateResponse(
            status=result.status.value,
            selected_version=str(result.selected_version) if result.selected_version else None,
            cloud_min=MIN_SUPPORTED_VERSION,
            cloud_max=MAX_SUPPORTED_VERSION,
            error_message=result.error_message,
            negotiated_at=result.negotiated_at,
        )


@router.get(
    "/version-info",
    response_model=Dict[str, Any],
    summary="Get protocol version info",
    description="Get Cloud protocol version information.",
)
async def get_version_info() -> Dict[str, Any]:
    """
    Get Cloud protocol version information.

    Returns Cloud's supported version range for client discovery.
    No authentication required for version discovery.
    """
    return {
        "current_version": CURRENT_SCHEMA_VERSION,
        "min_supported": MIN_SUPPORTED_VERSION,
        "max_supported": MAX_SUPPORTED_VERSION,
        "documentation": "https://docs.ccea.io/protocol-versions",
    }


# ============================================================================
# Heartbeat Endpoint
# ============================================================================

@router.post(
    "/heartbeat",
    response_model=AgentHeartbeatResponse,
    summary="Agent heartbeat",
    description="Report agent status and receive pending commands count.",
)
async def agent_heartbeat(
    request: AgentHeartbeatRequest,
    current_agent: AgentDep,
) -> AgentHeartbeatResponse:
    """
    Agent heartbeat endpoint.

    Updates agent's last_seen timestamp and returns:
    - Server time (for clock sync)
    - Current trust state
    - Number of pending commands
    - Recommended next heartbeat interval

    WI-CLOUD-02: Proper implementation with AgentDep auth and DB update.
    """
    async with get_session() as session:
        # Verify agent
        db_agent = await verify_agent_enrolled(session, current_agent)

        # Use agent service for heartbeat
        service = AgentService(session)
        try:
            result = await service.heartbeat(
                ServiceHeartbeatRequest(
                    agent_id=current_agent.id,
                    agent_version=request.agent_version,
                    current_state=request.current_state,
                    health_metrics=request.health_metrics,
                    last_run_id=request.last_run_id,
                )
            )

            return AgentHeartbeatResponse(
                server_time=result.server_time,
                trust_state=result.trust_state.value,
                pending_commands=result.pending_commands,
                next_heartbeat_sec=result.next_heartbeat_sec,
            )

        except AgentNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found",
            )
        except AgentNotEnrolledError as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(e),
            )


# ============================================================================
# Command Polling Endpoint
# ============================================================================

@router.get(
    "/commands/poll",
    response_model=CommandPollResponse,
    summary="Poll for commands",
    description="Long-poll for pending commands to execute.",
)
async def poll_commands(
    current_agent: AgentDep,
    limit: int = 10,
) -> CommandPollResponse:
    """
    Poll for pending commands.

    Returns commands in order of creation (FIFO).
    Commands requiring approval are excluded until approved.

    Design Doc 10.2: All commands MUST be signed by Cloud.
    Agent MUST verify signature before processing.

    WI-CLOUD-02: Implements command polling with accurate results.
    """
    async with get_session() as session:
        # Verify agent
        await verify_agent_enrolled(session, current_agent)

        # Use command service for polling
        service = CommandService(session)
        result = await service.poll_commands(
            agent_id=current_agent.id,
            workspace_id=current_agent.workspace_id,
            limit=min(limit, 50),  # Cap at 50
        )

        # Get command signer for signing commands (Design Doc 10.2)
        try:
            signer = get_cloud_signer()
        except CloudKeyNotConfiguredError:
            logger.error("Cloud signing key not configured - commands will be unsigned")
            signer = None

        # Convert to response format with signatures
        commands = []
        for cmd in result.commands:
            # Build command data dict for signing
            cmd_data = {
                "id": str(cmd.id),
                "status": cmd.status,
                "idempotency_key": cmd.idempotency_key,
                "command_type": cmd.command_type,
                "payload_ref": cmd.payload_ref,
                "change_class": cmd.change_class,
                "requires_approval": cmd.requires_approval,
                "deployment_id": str(cmd.deployment_id) if cmd.deployment_id else None,
                "run_id": str(cmd.run_id) if cmd.run_id else None,
                "expires_at": cmd.expires_at.isoformat() if cmd.expires_at else None,
                "created_at": cmd.created_at.isoformat() if cmd.created_at else None,
            }

            # Sign the command (Design Doc 10.2)
            if signer:
                signed_data = signer.sign_command(cmd_data)
                signature = signed_data.get("signature")
                timestamp_str = signed_data.get("timestamp")
                timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else None
            else:
                signature = None
                timestamp = datetime.now(timezone.utc)

            commands.append(
                PendingCommand(
                    id=cmd.id,
                    status=cmd.status,
                    idempotency_key=cmd.idempotency_key,
                    command_type=cmd.command_type,
                    payload_ref=cmd.payload_ref,
                    change_class=cmd.change_class,
                    requires_approval=cmd.requires_approval,
                    deployment_id=cmd.deployment_id,
                    run_id=cmd.run_id,
                    expires_at=cmd.expires_at,
                    created_at=cmd.created_at,
                    timestamp=timestamp,
                    signature=signature,
                )
            )

        # Get negotiated version for this agent (Design Doc 10.3/10.4)
        negotiated_version = _negotiated_versions.get(str(current_agent.id))

        return CommandPollResponse(
            commands=commands,
            has_more=result.has_more,
            poll_again_after_sec=result.poll_again_after_sec,
            protocol_version=negotiated_version,
        )


# ============================================================================
# Command Acknowledgement Endpoint
# ============================================================================

@router.post(
    "/commands/{command_id}/ack",
    response_model=CommandAckResponse,
    summary="Acknowledge command",
    description="Acknowledge receipt of a command.",
)
async def acknowledge_command(
    command_id: UUID,
    current_agent: AgentDep,
) -> CommandAckResponse:
    """
    Acknowledge receipt of a command.

    Agent calls this after receiving command via poll.
    Transitions command to ACKNOWLEDGED state.

    WI-CLOUD-02: Command acknowledgement for delivery confirmation.
    """
    async with get_session() as session:
        # Verify agent
        await verify_agent_enrolled(session, current_agent)

        # Use command service
        service = CommandService(session)
        try:
            command = await service.acknowledge_command(
                CommandAckRequest(
                    command_id=command_id,
                    agent_id=current_agent.id,
                )
            )

            return CommandAckResponse(
                command_id=command.id,
                status=command.status,
                acknowledged_at=command.acknowledged_at,
            )

        except CommandNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Command {command_id} not found",
            )
        except CommandStateError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )


# ============================================================================
# Command Result Submission Endpoint
# ============================================================================

@router.post(
    "/commands/{command_id}/result",
    response_model=CommandResultResponse,
    summary="Submit command result",
    description="Submit execution result for a command.",
)
async def submit_command_result(
    command_id: UUID,
    request: CommandResultSubmission,
    current_agent: AgentDep,
) -> CommandResultResponse:
    """
    Submit execution result for a command.

    Agent calls this after executing a command.
    Transitions command to EXECUTED or FAILED state.

    WI-CLOUD-02: Command result for execution tracking.
    """
    async with get_session() as session:
        # Verify agent
        await verify_agent_enrolled(session, current_agent)

        # Use command service
        service = CommandService(session)
        try:
            command = await service.submit_result(
                CommandResultRequest(
                    command_id=command_id,
                    agent_id=current_agent.id,
                    success=request.success,
                    result=request.result,
                    error_message=request.error_message,
                )
            )

            return CommandResultResponse(
                command_id=command.id,
                status=command.status,
                executed_at=command.executed_at,
            )

        except CommandNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Command {command_id} not found",
            )
        except CommandStateError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )


# ============================================================================
# Local Approval Endpoint
# ============================================================================

@router.post(
    "/commands/{command_id}/approval",
    response_model=LocalApprovalResponse,
    summary="Submit local approval",
    description="Submit local approval decision for a command.",
)
async def submit_local_approval(
    command_id: UUID,
    request: LocalApprovalRequest,
    current_agent: AgentDep,
) -> LocalApprovalResponse:
    """
    Submit local approval for a command.

    Agent calls this when local approval is required (TRADING_IMPACTING).
    Transitions command from PENDING_APPROVAL to APPROVED or REJECTED.

    WI-CLOUD-02: Local approval workflow for trading-impacting changes.

    Design Doc reference:
    - All TRADING_IMPACTING changes require local approval by default
    - Approval evidence hash is recorded for audit
    """
    async with get_session() as session:
        # Verify agent
        db_agent = await verify_agent_enrolled(session, current_agent)

        # Get command
        result = await session.execute(
            select(Command).where(
                Command.id == command_id,
                Command.agent_id == current_agent.id,
            )
        )
        command = result.scalar_one_or_none()

        if command is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Command {command_id} not found",
            )

        # Validate state
        if command.status != CommandStatus.PENDING_APPROVAL.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Command not pending approval. Current status: {command.status}",
            )

        # Check expiration
        if command.expires_at:
            expires = command.expires_at
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) > expires:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Command has expired",
                )

        # Create approval record (Design Doc 6.2, 12.2)
        now = datetime.now(timezone.utc)
        approval = ApprovalRecord(
            workspace_id=command.workspace_id,
            command_id=command_id,
            approved=request.approved,
            approved_by=f"local:{current_agent.id}",  # Mark as local approval
            evidence_hash=request.evidence_hash,
            attestation=request.attestation,
            reason=request.reason,
            diff_summary=request.diff_summary,
            # Design Doc 6.2, 12.2: Immutable blob references
            config_blob_digest=request.config_blob_digest,
            manifest_digest=request.manifest_digest,
            previous_state_digest=request.previous_state_digest,
            new_state_digest=request.new_state_digest,
        )
        session.add(approval)

        # Update command status
        if request.approved:
            command.status = CommandStatus.APPROVED.value
        else:
            command.status = CommandStatus.REJECTED.value

        command.updated_at = now
        await session.commit()

        return LocalApprovalResponse(
            command_id=command.id,
            status=command.status,
            approved=request.approved,
        )


# ============================================================================
# Agent Status Endpoint
# ============================================================================

@router.get(
    "/status",
    response_model=Dict[str, Any],
    summary="Get agent status",
    description="Get current agent status and configuration.",
)
async def get_agent_status(
    current_agent: AgentDep,
) -> Dict[str, Any]:
    """
    Get current agent status.

    Returns agent configuration and trust state.
    """
    async with get_session() as session:
        db_agent = await verify_agent_enrolled(session, current_agent)

        return {
            "agent_id": str(db_agent.id),
            "name": db_agent.name,
            "workspace_id": str(db_agent.workspace_id),
            "agent_version": db_agent.agent_version,
            "trust_state": db_agent.trust_state,
            "capabilities": db_agent.capabilities or [],
            "last_heartbeat_at": db_agent.last_heartbeat_at.isoformat() if db_agent.last_heartbeat_at else None,
            "health_status": db_agent.health_status,
        }


# Update forward reference for Pydantic
CommandPollResponse.model_rebuild()
