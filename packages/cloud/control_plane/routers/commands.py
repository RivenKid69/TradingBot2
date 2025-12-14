# -*- coding: utf-8 -*-
"""
Commands Router.

CLOUD ZONE ONLY.

Provides CRUD endpoints for command management and approval workflow.
Commands are sent from Cloud to Agent with idempotency and approval support.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from ..database import get_session
from ..dependencies import PaginationDep, UserDep
from ..models import (
    Agent,
    ApprovalRecord,
    ChangeClass,
    Command,
    CommandStatus,
    Deployment,
    Run,
    TrustState,
    Workspace,
)

router = APIRouter()


# ============================================================================
# State Transition Maps
# ============================================================================

# Valid state transitions for commands
COMMAND_STATE_TRANSITIONS: Dict[CommandStatus, List[CommandStatus]] = {
    CommandStatus.PENDING: [
        CommandStatus.PENDING_APPROVAL,  # If requires approval
        CommandStatus.SENT,              # Direct send
        CommandStatus.EXPIRED,           # Timeout
    ],
    CommandStatus.PENDING_APPROVAL: [
        CommandStatus.APPROVED,
        CommandStatus.REJECTED,
        CommandStatus.EXPIRED,
    ],
    CommandStatus.APPROVED: [
        CommandStatus.SENT,
        CommandStatus.EXPIRED,
    ],
    CommandStatus.SENT: [
        CommandStatus.ACKNOWLEDGED,
        CommandStatus.FAILED,
        CommandStatus.EXPIRED,
    ],
    CommandStatus.ACKNOWLEDGED: [
        CommandStatus.EXECUTED,
        CommandStatus.FAILED,
        CommandStatus.EXPIRED,
    ],
    # Terminal states
    CommandStatus.REJECTED: [],
    CommandStatus.EXECUTED: [],
    CommandStatus.FAILED: [],
    CommandStatus.EXPIRED: [],
}


# ============================================================================
# Request/Response Models
# ============================================================================

class ApprovalCreate(BaseModel):
    """Create approval request."""

    approved: bool
    reason: Optional[str] = None
    evidence_hash: Optional[str] = None
    attestation: Optional[dict] = None
    diff_summary: Optional[dict] = None


class ApprovalResponse(BaseModel):
    """Approval response."""

    id: UUID
    command_id: UUID
    approved: bool
    approved_by: str
    evidence_hash: Optional[str]
    attestation: Optional[dict]
    reason: Optional[str]
    diff_summary: Optional[dict]
    created_at: datetime


class CommandCreate(BaseModel):
    """Create command request."""

    agent_id: UUID
    deployment_id: Optional[UUID] = None
    run_id: Optional[UUID] = None
    command_type: str = Field(..., min_length=1, max_length=100)
    payload_ref: str = Field(..., min_length=1, max_length=128)  # sha256 digest
    change_class: ChangeClass = ChangeClass.OPERATIONAL
    requires_approval: bool = False
    expires_at: Optional[datetime] = None


class CommandUpdate(BaseModel):
    """Update command request (limited fields)."""

    expires_at: Optional[datetime] = None
    result: Optional[dict] = None
    error_message: Optional[str] = None


class CommandTransition(BaseModel):
    """Command state transition request."""

    new_status: CommandStatus
    result: Optional[dict] = None
    error_message: Optional[str] = None


class CommandResponse(BaseModel):
    """Command response."""

    id: UUID
    workspace_id: UUID
    idempotency_key: str
    agent_id: UUID
    deployment_id: Optional[UUID]
    run_id: Optional[UUID]
    command_type: str
    payload_ref: str
    change_class: ChangeClass
    requires_approval: bool
    status: CommandStatus
    sent_at: Optional[datetime]
    acknowledged_at: Optional[datetime]
    executed_at: Optional[datetime]
    expires_at: Optional[datetime]
    result: Optional[dict]
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime
    approvals: List[ApprovalResponse] = Field(default_factory=list)
    is_expired: bool = False


class CommandListResponse(BaseModel):
    """Paginated command list response."""

    items: List[CommandResponse]
    total: int
    page: int
    page_size: int


# ============================================================================
# Helper Functions
# ============================================================================

def _generate_idempotency_key() -> str:
    """Generate a unique idempotency key."""
    import secrets
    return secrets.token_urlsafe(24)


def _is_expired(command: Command) -> bool:
    """Check if command has expired."""
    if command.expires_at is None:
        return False
    # Handle both naive and aware datetimes (SQLite returns naive)
    expires = command.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) > expires


def _approval_to_response(approval: ApprovalRecord) -> ApprovalResponse:
    """Convert ApprovalRecord model to response."""
    return ApprovalResponse(
        id=approval.id,
        command_id=approval.command_id,
        approved=approval.approved,
        approved_by=approval.approved_by,
        evidence_hash=approval.evidence_hash,
        attestation=approval.attestation,
        reason=approval.reason,
        diff_summary=approval.diff_summary,
        created_at=approval.created_at,
    )


def _command_to_response(command: Command) -> CommandResponse:
    """Convert Command model to response."""
    return CommandResponse(
        id=command.id,
        workspace_id=command.workspace_id,
        idempotency_key=command.idempotency_key,
        agent_id=command.agent_id,
        deployment_id=command.deployment_id,
        run_id=command.run_id,
        command_type=command.command_type,
        payload_ref=command.payload_ref,
        change_class=ChangeClass(command.change_class),
        requires_approval=command.requires_approval,
        status=CommandStatus(command.status),
        sent_at=command.sent_at,
        acknowledged_at=command.acknowledged_at,
        executed_at=command.executed_at,
        expires_at=command.expires_at,
        result=command.result,
        error_message=command.error_message,
        created_at=command.created_at,
        updated_at=command.updated_at,
        approvals=[
            _approval_to_response(a) for a in command.approvals
        ] if command.approvals else [],
        is_expired=_is_expired(command),
    )


async def _verify_workspace_access(
    session,
    workspace_id: UUID,
    current_user,
) -> Workspace:
    """Verify user has access to the workspace."""
    result = await session.execute(
        select(Workspace).where(
            Workspace.id == workspace_id,
            Workspace.is_active == True,
        )
    )
    workspace = result.scalar_one_or_none()

    if workspace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found",
        )

    # Check organization access
    if not current_user.is_superuser:
        if current_user.org_id != workspace.organization_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this workspace",
            )

    return workspace


# ============================================================================
# Command Endpoints
# ============================================================================

@router.get(
    "",
    response_model=CommandListResponse,
    summary="List commands",
    description="List commands with optional filters.",
)
async def list_commands(
    current_user: UserDep,
    pagination: PaginationDep,
    workspace_id: Optional[UUID] = None,
    agent_id: Optional[UUID] = None,
    deployment_id: Optional[UUID] = None,
    run_id: Optional[UUID] = None,
    command_status: Optional[CommandStatus] = None,
    command_type: Optional[str] = None,
    change_class: Optional[ChangeClass] = None,
    requires_approval: Optional[bool] = None,
) -> CommandListResponse:
    """
    List commands.

    Filters available:
    - workspace_id: Filter by workspace
    - agent_id: Filter by target agent
    - deployment_id: Filter by deployment
    - run_id: Filter by run
    - command_status: Filter by status
    - command_type: Filter by command type
    - change_class: Filter by change classification
    - requires_approval: Filter by approval requirement
    """
    async with get_session() as session:
        # Build base query
        query = select(Command).options(selectinload(Command.approvals))
        count_query = select(func.count(Command.id))

        # Apply workspace filter
        if current_user.is_superuser:
            if workspace_id:
                query = query.where(Command.workspace_id == workspace_id)
                count_query = count_query.where(Command.workspace_id == workspace_id)
        else:
            # Non-superusers need workspace access through organization
            if current_user.org_id is None:
                return CommandListResponse(
                    items=[],
                    total=0,
                    page=pagination.page,
                    page_size=pagination.page_size,
                )

            # Get workspace IDs for user's organization
            ws_result = await session.execute(
                select(Workspace.id).where(
                    Workspace.organization_id == current_user.org_id,
                    Workspace.is_active == True,
                )
            )
            workspace_ids = [row[0] for row in ws_result.fetchall()]

            if not workspace_ids:
                return CommandListResponse(
                    items=[],
                    total=0,
                    page=pagination.page,
                    page_size=pagination.page_size,
                )

            if workspace_id and workspace_id in workspace_ids:
                query = query.where(Command.workspace_id == workspace_id)
                count_query = count_query.where(Command.workspace_id == workspace_id)
            else:
                query = query.where(Command.workspace_id.in_(workspace_ids))
                count_query = count_query.where(Command.workspace_id.in_(workspace_ids))

        # Apply additional filters
        if agent_id:
            query = query.where(Command.agent_id == agent_id)
            count_query = count_query.where(Command.agent_id == agent_id)

        if deployment_id:
            query = query.where(Command.deployment_id == deployment_id)
            count_query = count_query.where(Command.deployment_id == deployment_id)

        if run_id:
            query = query.where(Command.run_id == run_id)
            count_query = count_query.where(Command.run_id == run_id)

        if command_status:
            query = query.where(Command.status == command_status.value)
            count_query = count_query.where(Command.status == command_status.value)

        if command_type:
            query = query.where(Command.command_type == command_type)
            count_query = count_query.where(Command.command_type == command_type)

        if change_class:
            query = query.where(Command.change_class == change_class.value)
            count_query = count_query.where(Command.change_class == change_class.value)

        if requires_approval is not None:
            query = query.where(Command.requires_approval == requires_approval)
            count_query = count_query.where(Command.requires_approval == requires_approval)

        # Get total count
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Get commands
        query = (
            query.offset(pagination.offset)
            .limit(pagination.limit)
            .order_by(Command.created_at.desc())
        )
        result = await session.execute(query)
        commands = result.scalars().unique().all()

        items = [_command_to_response(cmd) for cmd in commands]

        return CommandListResponse(
            items=items,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
        )


@router.post(
    "",
    response_model=CommandResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create command",
    description="Create a new command for an agent.",
)
async def create_command(
    request: CommandCreate,
    workspace_id: UUID,
    current_user: UserDep,
    idempotency_key: Optional[str] = None,
) -> CommandResponse:
    """
    Create a new command.

    The command will be created in PENDING status.
    If requires_approval is True, it will need to be approved before sending.
    """
    async with get_session() as session:
        # Verify workspace access
        workspace = await _verify_workspace_access(session, workspace_id, current_user)

        # Check permission
        if not current_user.is_superuser and not current_user.has_permission("command:create"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission required: command:create",
            )

        # Verify agent exists and is enrolled
        agent_result = await session.execute(
            select(Agent).where(
                Agent.id == request.agent_id,
                Agent.workspace_id == workspace_id,
                Agent.deleted_at == None,
            )
        )
        agent = agent_result.scalar_one_or_none()

        if agent is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found in this workspace",
            )

        if agent.trust_state != TrustState.ENROLLED.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Agent is not enrolled. Current state: {agent.trust_state}",
            )

        # Verify deployment if provided
        if request.deployment_id:
            deploy_result = await session.execute(
                select(Deployment).where(
                    Deployment.id == request.deployment_id,
                    Deployment.workspace_id == workspace_id,
                    Deployment.deleted_at == None,
                )
            )
            if deploy_result.scalar_one_or_none() is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Deployment not found in this workspace",
                )

        # Verify run if provided
        if request.run_id:
            run_result = await session.execute(
                select(Run).where(
                    Run.id == request.run_id,
                    Run.workspace_id == workspace_id,
                )
            )
            if run_result.scalar_one_or_none() is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Run not found in this workspace",
                )

        # Generate or use provided idempotency key
        idem_key = idempotency_key or _generate_idempotency_key()

        # Check idempotency key uniqueness
        existing = await session.execute(
            select(Command).where(
                Command.workspace_id == workspace_id,
                Command.idempotency_key == idem_key,
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Command with idempotency key '{idem_key}' already exists",
            )

        # Determine initial status
        initial_status = CommandStatus.PENDING

        command = Command(
            workspace_id=workspace_id,
            idempotency_key=idem_key,
            agent_id=request.agent_id,
            deployment_id=request.deployment_id,
            run_id=request.run_id,
            command_type=request.command_type,
            payload_ref=request.payload_ref,
            change_class=request.change_class.value,
            requires_approval=request.requires_approval,
            status=initial_status.value,
            expires_at=request.expires_at,
        )
        session.add(command)
        await session.commit()
        await session.refresh(command)

        # Reload with approvals
        result = await session.execute(
            select(Command)
            .options(selectinload(Command.approvals))
            .where(Command.id == command.id)
        )
        command = result.scalar_one()

        return _command_to_response(command)


@router.get(
    "/{command_id}",
    response_model=CommandResponse,
    summary="Get command",
    description="Get command by ID.",
)
async def get_command(
    command_id: UUID,
    current_user: UserDep,
) -> CommandResponse:
    """Get command by ID."""
    async with get_session() as session:
        result = await session.execute(
            select(Command)
            .options(selectinload(Command.approvals))
            .where(Command.id == command_id)
        )
        command = result.scalar_one_or_none()

        if command is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Command not found",
            )

        # Verify access
        await _verify_workspace_access(session, command.workspace_id, current_user)

        return _command_to_response(command)


@router.patch(
    "/{command_id}",
    response_model=CommandResponse,
    summary="Update command",
    description="Update command fields (limited).",
)
async def update_command(
    command_id: UUID,
    request: CommandUpdate,
    current_user: UserDep,
) -> CommandResponse:
    """
    Update command.

    Only certain fields can be updated: expires_at, result, error_message.
    """
    async with get_session() as session:
        result = await session.execute(
            select(Command)
            .options(selectinload(Command.approvals))
            .where(Command.id == command_id)
        )
        command = result.scalar_one_or_none()

        if command is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Command not found",
            )

        # Verify access
        await _verify_workspace_access(session, command.workspace_id, current_user)

        # Check permission
        if not current_user.is_superuser and not current_user.has_permission("command:write"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission required: command:write",
            )

        # Check if command is in terminal state
        terminal_states = [
            CommandStatus.EXECUTED.value,
            CommandStatus.FAILED.value,
            CommandStatus.EXPIRED.value,
            CommandStatus.REJECTED.value,
        ]
        if command.status in terminal_states:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot update command in terminal state: {command.status}",
            )

        # Update fields
        if request.expires_at is not None:
            command.expires_at = request.expires_at

        if request.result is not None:
            command.result = request.result

        if request.error_message is not None:
            command.error_message = request.error_message

        command.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(command)

        # Reload with approvals
        result = await session.execute(
            select(Command)
            .options(selectinload(Command.approvals))
            .where(Command.id == command.id)
        )
        command = result.scalar_one()

        return _command_to_response(command)


@router.post(
    "/{command_id}/transition",
    response_model=CommandResponse,
    summary="Transition command state",
    description="Transition command to a new state.",
)
async def transition_command(
    command_id: UUID,
    request: CommandTransition,
    current_user: UserDep,
) -> CommandResponse:
    """
    Transition command to a new state.

    Valid transitions are enforced by state machine.
    """
    async with get_session() as session:
        result = await session.execute(
            select(Command)
            .options(selectinload(Command.approvals))
            .where(Command.id == command_id)
        )
        command = result.scalar_one_or_none()

        if command is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Command not found",
            )

        # Verify access
        await _verify_workspace_access(session, command.workspace_id, current_user)

        # Check permission
        if not current_user.is_superuser and not current_user.has_permission("command:transition"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission required: command:transition",
            )

        # Check expiration
        if _is_expired(command) and request.new_status != CommandStatus.EXPIRED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Command has expired. Can only transition to EXPIRED state.",
            )

        # Validate state transition
        current_status = CommandStatus(command.status)
        allowed_transitions = COMMAND_STATE_TRANSITIONS.get(current_status, [])

        if request.new_status not in allowed_transitions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid transition from {current_status.value} to {request.new_status.value}. "
                       f"Allowed: {[s.value for s in allowed_transitions]}",
            )

        # Special validation for PENDING_APPROVAL
        if (
            current_status == CommandStatus.PENDING
            and request.new_status == CommandStatus.PENDING_APPROVAL
            and not command.requires_approval
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Command does not require approval",
            )

        # Special validation for SENT (must be approved if requires approval)
        if (
            request.new_status == CommandStatus.SENT
            and command.requires_approval
            and current_status != CommandStatus.APPROVED
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Command requires approval before sending",
            )

        # Update state
        now = datetime.now(timezone.utc)
        command.status = request.new_status.value

        # Set timing fields based on state
        if request.new_status == CommandStatus.SENT:
            command.sent_at = now
        elif request.new_status == CommandStatus.ACKNOWLEDGED:
            command.acknowledged_at = now
        elif request.new_status == CommandStatus.EXECUTED:
            command.executed_at = now
            if request.result is not None:
                command.result = request.result

        # Set error message for failed states
        if request.new_status == CommandStatus.FAILED:
            if request.error_message:
                command.error_message = request.error_message

        command.updated_at = now
        await session.commit()
        await session.refresh(command)

        # Reload with approvals
        result = await session.execute(
            select(Command)
            .options(selectinload(Command.approvals))
            .where(Command.id == command.id)
        )
        command = result.scalar_one()

        return _command_to_response(command)


# ============================================================================
# Approval Endpoints
# ============================================================================

@router.get(
    "/{command_id}/approvals",
    response_model=List[ApprovalResponse],
    summary="List approvals",
    description="List all approvals for a command.",
)
async def list_approvals(
    command_id: UUID,
    current_user: UserDep,
) -> List[ApprovalResponse]:
    """List all approvals for a command."""
    async with get_session() as session:
        # Get command first
        cmd_result = await session.execute(
            select(Command).where(Command.id == command_id)
        )
        command = cmd_result.scalar_one_or_none()

        if command is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Command not found",
            )

        # Verify access
        await _verify_workspace_access(session, command.workspace_id, current_user)

        # Get approvals
        result = await session.execute(
            select(ApprovalRecord)
            .where(ApprovalRecord.command_id == command_id)
            .order_by(ApprovalRecord.created_at.desc())
        )
        approvals = result.scalars().all()

        return [_approval_to_response(a) for a in approvals]


@router.post(
    "/{command_id}/approvals",
    response_model=ApprovalResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create approval",
    description="Create an approval decision for a command.",
)
async def create_approval(
    command_id: UUID,
    request: ApprovalCreate,
    current_user: UserDep,
) -> ApprovalResponse:
    """
    Create an approval decision for a command.

    This will also transition the command state to APPROVED or REJECTED.
    """
    async with get_session() as session:
        # Get command with lock for update
        result = await session.execute(
            select(Command)
            .options(selectinload(Command.approvals))
            .where(Command.id == command_id)
        )
        command = result.scalar_one_or_none()

        if command is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Command not found",
            )

        # Verify access
        await _verify_workspace_access(session, command.workspace_id, current_user)

        # Check permission
        if not current_user.is_superuser and not current_user.has_permission("command:approve"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission required: command:approve",
            )

        # Validate command state
        if command.status != CommandStatus.PENDING_APPROVAL.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Command is not pending approval. Current status: {command.status}",
            )

        # Check expiration
        if _is_expired(command):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Command has expired",
            )

        # Get approver identity
        approver = str(current_user.id)

        # Create approval record
        approval = ApprovalRecord(
            workspace_id=command.workspace_id,
            command_id=command_id,
            approved=request.approved,
            approved_by=approver,
            evidence_hash=request.evidence_hash,
            attestation=request.attestation,
            reason=request.reason,
            diff_summary=request.diff_summary,
        )
        session.add(approval)

        # Transition command state
        now = datetime.now(timezone.utc)
        if request.approved:
            command.status = CommandStatus.APPROVED.value
        else:
            command.status = CommandStatus.REJECTED.value

        command.updated_at = now
        await session.commit()
        await session.refresh(approval)

        return _approval_to_response(approval)


@router.get(
    "/{command_id}/approvals/{approval_id}",
    response_model=ApprovalResponse,
    summary="Get approval",
    description="Get a specific approval record.",
)
async def get_approval(
    command_id: UUID,
    approval_id: UUID,
    current_user: UserDep,
) -> ApprovalResponse:
    """Get a specific approval record."""
    async with get_session() as session:
        # Get command first
        cmd_result = await session.execute(
            select(Command).where(Command.id == command_id)
        )
        command = cmd_result.scalar_one_or_none()

        if command is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Command not found",
            )

        # Verify access
        await _verify_workspace_access(session, command.workspace_id, current_user)

        # Get approval
        result = await session.execute(
            select(ApprovalRecord).where(
                ApprovalRecord.id == approval_id,
                ApprovalRecord.command_id == command_id,
            )
        )
        approval = result.scalar_one_or_none()

        if approval is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Approval not found",
            )

        return _approval_to_response(approval)


# ============================================================================
# Batch Operations
# ============================================================================

@router.post(
    "/expire-stale",
    summary="Expire stale commands",
    description="Mark all expired commands as EXPIRED.",
)
async def expire_stale_commands(
    workspace_id: UUID,
    current_user: UserDep,
) -> Dict[str, int]:
    """
    Expire all stale commands in a workspace.

    Returns count of expired commands.
    """
    async with get_session() as session:
        # Verify workspace access
        await _verify_workspace_access(session, workspace_id, current_user)

        # Check permission
        if not current_user.is_superuser and not current_user.has_permission("command:expire"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission required: command:expire",
            )

        # Find and expire stale commands
        now = datetime.now(timezone.utc)
        non_terminal_states = [
            CommandStatus.PENDING.value,
            CommandStatus.PENDING_APPROVAL.value,
            CommandStatus.APPROVED.value,
            CommandStatus.SENT.value,
            CommandStatus.ACKNOWLEDGED.value,
        ]

        result = await session.execute(
            select(Command).where(
                Command.workspace_id == workspace_id,
                Command.status.in_(non_terminal_states),
                Command.expires_at != None,
                Command.expires_at < now,
            )
        )
        stale_commands = result.scalars().all()

        count = 0
        for cmd in stale_commands:
            cmd.status = CommandStatus.EXPIRED.value
            cmd.updated_at = now
            count += 1

        await session.commit()

        return {"expired_count": count}
