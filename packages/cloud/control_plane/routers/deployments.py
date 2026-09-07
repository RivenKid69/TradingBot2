# -*- coding: utf-8 -*-
"""
Deployments Router.

CLOUD ZONE ONLY.

Provides CRUD endpoints for deployment and run management.
Implements the deployment lifecycle from artifact to running instance.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from ..database import get_session
from ..dependencies import PaginationDep, UserDep
from ..models import (
    Agent,
    Artifact,
    ConfigBlob,
    Deployment,
    DeploymentState,
    Run,
    RunState,
    TrustState,
    Workspace,
)

router = APIRouter()


# =============================================================================
# Request/Response Models - Deployment
# =============================================================================


class DeploymentCreate(BaseModel):
    """Create deployment request."""

    workspace_id: UUID
    agent_id: UUID
    artifact_id: UUID
    config_blob_id: Optional[UUID] = None
    desired_state: str = Field(
        default="DEPLOYED",
        pattern=r"^(DEPLOYED|SUSPENDED)$",
    )
    metadata: dict = Field(default_factory=dict)


class DeploymentUpdate(BaseModel):
    """Update deployment request."""

    config_blob_id: Optional[UUID] = None
    desired_state: Optional[str] = Field(
        None,
        pattern=r"^(DEPLOYED|SUSPENDED|RETIRED)$",
    )
    metadata: Optional[dict] = None


class DeploymentResponse(BaseModel):
    """Deployment response."""

    id: UUID
    workspace_id: UUID
    agent_id: UUID
    artifact_id: UUID
    config_blob_id: Optional[UUID]
    state: str
    desired_state: str
    state_changed_at: Optional[datetime]
    metadata: dict
    created_at: datetime
    updated_at: datetime
    run_count: int = 0
    active_run_id: Optional[UUID] = None


class DeploymentListResponse(BaseModel):
    """Paginated deployment list response."""

    items: List[DeploymentResponse]
    total: int
    page: int
    page_size: int


class DeploymentStateTransition(BaseModel):
    """State transition request."""

    target_state: str = Field(
        ...,
        pattern=r"^(pending_approval|approved|deploying|deployed|suspended|failed|retired)$",
    )
    reason: Optional[str] = Field(None, max_length=500)


# =============================================================================
# Request/Response Models - Run
# =============================================================================


class RunCreate(BaseModel):
    """Create run request."""

    is_paper_trading: bool = True
    metadata: dict = Field(default_factory=dict)


class RunUpdate(BaseModel):
    """Update run request."""

    error_message: Optional[str] = Field(None, max_length=2000)
    error_code: Optional[str] = Field(None, max_length=100)
    metrics_summary: Optional[dict] = None


class RunResponse(BaseModel):
    """Run response."""

    id: UUID
    deployment_id: UUID
    workspace_id: UUID
    state: str
    state_changed_at: Optional[datetime]
    started_at: Optional[datetime]
    stopped_at: Optional[datetime]
    is_paper_trading: bool
    error_message: Optional[str]
    error_code: Optional[str]
    metrics_summary: Optional[dict]
    created_at: datetime
    updated_at: datetime


class RunListResponse(BaseModel):
    """Paginated run list response."""

    items: List[RunResponse]
    total: int
    page: int
    page_size: int


class RunStateTransition(BaseModel):
    """Run state transition request."""

    target_state: str = Field(
        ...,
        pattern=r"^(pending_approval|approved|starting|running|paused|stopping|stopped|failed|completed)$",
    )
    reason: Optional[str] = Field(None, max_length=500)


# =============================================================================
# Helper Functions
# =============================================================================


async def _get_run_count(session, deployment_id: UUID) -> int:
    """Get the count of runs for a deployment."""
    result = await session.execute(
        select(func.count(Run.id)).where(Run.deployment_id == deployment_id)
    )
    return result.scalar() or 0


async def _get_active_run(session, deployment_id: UUID) -> Optional[UUID]:
    """Get the ID of the active run for a deployment."""
    result = await session.execute(
        select(Run.id).where(
            Run.deployment_id == deployment_id,
            Run.state.in_(
                [
                    RunState.STARTING.value,
                    RunState.RUNNING.value,
                    RunState.PAUSED.value,
                ]
            ),
        )
    )
    run_id = result.scalar_one_or_none()
    return run_id


async def _verify_workspace_access(
    session, workspace_id: UUID, current_user, require_permission: Optional[str] = None
) -> Workspace:
    """Verify workspace exists and user has access."""
    ws_result = await session.execute(
        select(Workspace).where(
            Workspace.id == workspace_id,
            Workspace.is_active == True,
        )
    )
    workspace = ws_result.scalar_one_or_none()

    if workspace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found",
        )

    if not current_user.is_superuser:
        if current_user.org_id != workspace.organization_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this workspace",
            )
        if require_permission and not current_user.has_permission(require_permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {require_permission}",
            )

    return workspace


def _deployment_to_response(
    deployment: Deployment,
    run_count: int = 0,
    active_run_id: Optional[UUID] = None,
) -> DeploymentResponse:
    """Convert Deployment model to response."""
    return DeploymentResponse(
        id=deployment.id,
        workspace_id=deployment.workspace_id,
        agent_id=deployment.agent_id,
        artifact_id=deployment.artifact_id,
        config_blob_id=deployment.config_blob_id,
        state=deployment.state.value if hasattr(deployment.state, "value") else deployment.state,
        desired_state=deployment.desired_state,
        state_changed_at=deployment.state_changed_at,
        metadata=deployment.extra_metadata or {},
        created_at=deployment.created_at,
        updated_at=deployment.updated_at,
        run_count=run_count,
        active_run_id=active_run_id,
    )


def _run_to_response(run: Run) -> RunResponse:
    """Convert Run model to response."""
    return RunResponse(
        id=run.id,
        deployment_id=run.deployment_id,
        workspace_id=run.workspace_id,
        state=run.state.value if hasattr(run.state, "value") else run.state,
        state_changed_at=run.state_changed_at,
        started_at=run.started_at,
        stopped_at=run.stopped_at,
        is_paper_trading=run.is_paper_trading,
        error_message=run.error_message,
        error_code=run.error_code,
        metrics_summary=run.metrics_summary,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


# Valid state transitions for deployments
DEPLOYMENT_STATE_TRANSITIONS = {
    DeploymentState.CREATED: [DeploymentState.PENDING_APPROVAL, DeploymentState.APPROVED],
    DeploymentState.PENDING_APPROVAL: [DeploymentState.APPROVED, DeploymentState.FAILED],
    DeploymentState.APPROVED: [DeploymentState.DEPLOYING, DeploymentState.FAILED],
    DeploymentState.DEPLOYING: [DeploymentState.DEPLOYED, DeploymentState.FAILED],
    DeploymentState.DEPLOYED: [
        DeploymentState.SUSPENDED,
        DeploymentState.RETIRED,
        DeploymentState.FAILED,
    ],
    DeploymentState.SUSPENDED: [DeploymentState.DEPLOYED, DeploymentState.RETIRED],
    DeploymentState.FAILED: [DeploymentState.PENDING_APPROVAL, DeploymentState.RETIRED],
    DeploymentState.RETIRED: [],  # Terminal state
}

# Valid state transitions for runs
RUN_STATE_TRANSITIONS = {
    RunState.CREATED: [RunState.PENDING_APPROVAL, RunState.APPROVED],
    RunState.PENDING_APPROVAL: [RunState.APPROVED, RunState.FAILED],
    RunState.APPROVED: [RunState.STARTING, RunState.FAILED],
    RunState.STARTING: [RunState.RUNNING, RunState.FAILED],
    RunState.RUNNING: [RunState.PAUSED, RunState.STOPPING, RunState.FAILED, RunState.COMPLETED],
    RunState.PAUSED: [RunState.RUNNING, RunState.STOPPING, RunState.FAILED],
    RunState.STOPPING: [RunState.STOPPED, RunState.FAILED],
    RunState.STOPPED: [RunState.STARTING],  # Can restart
    RunState.FAILED: [],  # Terminal state
    RunState.COMPLETED: [],  # Terminal state
}


def _validate_deployment_state_transition(
    current_state: DeploymentState, target_state: DeploymentState
) -> bool:
    """Validate deployment state transition."""
    valid_targets = DEPLOYMENT_STATE_TRANSITIONS.get(current_state, [])
    return target_state in valid_targets


def _validate_run_state_transition(current_state: RunState, target_state: RunState) -> bool:
    """Validate run state transition."""
    valid_targets = RUN_STATE_TRANSITIONS.get(current_state, [])
    return target_state in valid_targets


# =============================================================================
# Deployment Endpoints
# =============================================================================


@router.get(
    "",
    response_model=DeploymentListResponse,
    summary="List deployments",
    description="List deployments in a workspace.",
)
async def list_deployments(
    current_user: UserDep,
    pagination: PaginationDep,
    workspace_id: Optional[UUID] = None,
    agent_id: Optional[UUID] = None,
    state: Optional[str] = None,
    include_retired: bool = False,
) -> DeploymentListResponse:
    """
    List deployments.

    Users see deployments in their organization's workspaces.
    Superusers can see all or filter by workspace.
    """
    async with get_session() as session:
        # Build base query
        query = select(Deployment)
        count_query = select(func.count(Deployment.id))

        # Exclude soft-deleted
        query = query.where(Deployment.deleted_at.is_(None))
        count_query = count_query.where(Deployment.deleted_at.is_(None))

        # Apply workspace filter
        if workspace_id:
            await _verify_workspace_access(session, workspace_id, current_user)
            query = query.where(Deployment.workspace_id == workspace_id)
            count_query = count_query.where(Deployment.workspace_id == workspace_id)
        elif not current_user.is_superuser:
            # Get all workspaces in user's org
            ws_ids_q = select(Workspace.id).where(Workspace.organization_id == current_user.org_id)
            query = query.where(Deployment.workspace_id.in_(ws_ids_q))
            count_query = count_query.where(Deployment.workspace_id.in_(ws_ids_q))

        # Apply agent filter
        if agent_id:
            query = query.where(Deployment.agent_id == agent_id)
            count_query = count_query.where(Deployment.agent_id == agent_id)

        # Apply state filter
        if state:
            query = query.where(Deployment.state == state)
            count_query = count_query.where(Deployment.state == state)

        # Exclude retired unless requested
        if not include_retired:
            query = query.where(Deployment.state != DeploymentState.RETIRED.value)
            count_query = count_query.where(Deployment.state != DeploymentState.RETIRED.value)

        # Get total count
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Get deployments
        query = (
            query.offset(pagination.offset)
            .limit(pagination.limit)
            .order_by(Deployment.created_at.desc())
        )
        result = await session.execute(query)
        deployments = result.scalars().all()

        # Build response with counts
        items = []
        for deployment in deployments:
            run_count = await _get_run_count(session, deployment.id)
            active_run_id = await _get_active_run(session, deployment.id)
            items.append(_deployment_to_response(deployment, run_count, active_run_id))

        return DeploymentListResponse(
            items=items,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
        )


@router.post(
    "",
    response_model=DeploymentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create deployment",
    description="Create a new deployment.",
)
async def create_deployment(
    request: DeploymentCreate,
    current_user: UserDep,
) -> DeploymentResponse:
    """
    Create a new deployment.

    Requires deployment:create permission or superuser.
    """
    async with get_session() as session:
        # Verify workspace access
        await _verify_workspace_access(
            session, request.workspace_id, current_user, "deployment:create"
        )

        # Verify agent exists and is in the same workspace
        agent_result = await session.execute(
            select(Agent).where(
                Agent.id == request.agent_id,
                Agent.workspace_id == request.workspace_id,
            )
        )
        agent = agent_result.scalar_one_or_none()

        if agent is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found in this workspace",
            )

        # Verify agent is enrolled
        if agent.trust_state != TrustState.ENROLLED.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Agent is not enrolled (current state: {agent.trust_state})",
            )

        # Verify artifact exists and is in the same workspace
        artifact_result = await session.execute(
            select(Artifact).where(
                Artifact.id == request.artifact_id,
                Artifact.workspace_id == request.workspace_id,
            )
        )
        artifact = artifact_result.scalar_one_or_none()

        if artifact is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Artifact not found in this workspace",
            )

        # Verify config blob if provided
        if request.config_blob_id:
            config_result = await session.execute(
                select(ConfigBlob).where(
                    ConfigBlob.id == request.config_blob_id,
                    ConfigBlob.workspace_id == request.workspace_id,
                )
            )
            config = config_result.scalar_one_or_none()

            if config is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Config blob not found in this workspace",
                )

        # Create deployment
        deployment = Deployment(
            workspace_id=request.workspace_id,
            agent_id=request.agent_id,
            artifact_id=request.artifact_id,
            config_blob_id=request.config_blob_id,
            state=DeploymentState.CREATED,
            desired_state=request.desired_state,
            extra_metadata=request.metadata,
        )
        session.add(deployment)
        await session.commit()
        await session.refresh(deployment)

        return _deployment_to_response(deployment, 0, None)


@router.get(
    "/{deployment_id}",
    response_model=DeploymentResponse,
    summary="Get deployment",
    description="Get deployment by ID.",
)
async def get_deployment(
    deployment_id: UUID,
    current_user: UserDep,
) -> DeploymentResponse:
    """Get deployment by ID."""
    async with get_session() as session:
        result = await session.execute(
            select(Deployment).where(
                Deployment.id == deployment_id,
                Deployment.deleted_at.is_(None),
            )
        )
        deployment = result.scalar_one_or_none()

        if deployment is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Deployment not found",
            )

        # Verify workspace access
        await _verify_workspace_access(session, deployment.workspace_id, current_user)

        run_count = await _get_run_count(session, deployment.id)
        active_run_id = await _get_active_run(session, deployment.id)

        return _deployment_to_response(deployment, run_count, active_run_id)


@router.patch(
    "/{deployment_id}",
    response_model=DeploymentResponse,
    summary="Update deployment",
    description="Update deployment settings.",
)
async def update_deployment(
    deployment_id: UUID,
    request: DeploymentUpdate,
    current_user: UserDep,
) -> DeploymentResponse:
    """
    Update deployment.

    Requires deployment:write permission or superuser.
    """
    async with get_session() as session:
        result = await session.execute(
            select(Deployment).where(
                Deployment.id == deployment_id,
                Deployment.deleted_at.is_(None),
            )
        )
        deployment = result.scalar_one_or_none()

        if deployment is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Deployment not found",
            )

        # Verify workspace access
        await _verify_workspace_access(
            session, deployment.workspace_id, current_user, "deployment:write"
        )

        # Update fields
        if request.config_blob_id is not None:
            # Verify config blob
            config_result = await session.execute(
                select(ConfigBlob).where(
                    ConfigBlob.id == request.config_blob_id,
                    ConfigBlob.workspace_id == deployment.workspace_id,
                )
            )
            if config_result.scalar_one_or_none() is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Config blob not found in this workspace",
                )
            deployment.config_blob_id = request.config_blob_id

        if request.desired_state is not None:
            deployment.desired_state = request.desired_state

        if request.metadata is not None:
            deployment.extra_metadata = request.metadata

        deployment.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(deployment)

        run_count = await _get_run_count(session, deployment.id)
        active_run_id = await _get_active_run(session, deployment.id)

        return _deployment_to_response(deployment, run_count, active_run_id)


@router.post(
    "/{deployment_id}/transition",
    response_model=DeploymentResponse,
    summary="Transition deployment state",
    description="Transition deployment to a new state.",
)
async def transition_deployment_state(
    deployment_id: UUID,
    request: DeploymentStateTransition,
    current_user: UserDep,
) -> DeploymentResponse:
    """
    Transition deployment state.

    Requires deployment:write permission or superuser.
    Validates state transitions according to the deployment state machine.
    """
    async with get_session() as session:
        result = await session.execute(
            select(Deployment).where(
                Deployment.id == deployment_id,
                Deployment.deleted_at.is_(None),
            )
        )
        deployment = result.scalar_one_or_none()

        if deployment is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Deployment not found",
            )

        # Verify workspace access
        await _verify_workspace_access(
            session, deployment.workspace_id, current_user, "deployment:write"
        )

        # Get current and target states
        current_state = (
            DeploymentState(deployment.state)
            if isinstance(deployment.state, str)
            else deployment.state
        )
        target_state = DeploymentState(request.target_state)

        # Validate transition
        if not _validate_deployment_state_transition(current_state, target_state):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid state transition from {current_state.value} to {target_state.value}",
            )

        # Update state
        deployment.state = target_state
        deployment.state_changed_at = datetime.now(timezone.utc)
        deployment.updated_at = datetime.now(timezone.utc)

        await session.commit()
        await session.refresh(deployment)

        run_count = await _get_run_count(session, deployment.id)
        active_run_id = await _get_active_run(session, deployment.id)

        return _deployment_to_response(deployment, run_count, active_run_id)


@router.delete(
    "/{deployment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    summary="Delete deployment",
    description="Soft delete deployment.",
)
async def delete_deployment(
    deployment_id: UUID,
    current_user: UserDep,
) -> None:
    """
    Soft delete deployment.

    Requires deployment:delete permission or superuser.
    Cannot delete deployments with active runs.
    """
    async with get_session() as session:
        result = await session.execute(
            select(Deployment).where(
                Deployment.id == deployment_id,
                Deployment.deleted_at.is_(None),
            )
        )
        deployment = result.scalar_one_or_none()

        if deployment is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Deployment not found",
            )

        # Verify workspace access
        await _verify_workspace_access(
            session, deployment.workspace_id, current_user, "deployment:delete"
        )

        # Check for active runs
        active_run_id = await _get_active_run(session, deployment.id)
        if active_run_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete deployment with active runs",
            )

        deployment.deleted_at = datetime.now(timezone.utc)
        deployment.updated_at = datetime.now(timezone.utc)
        await session.commit()


# =============================================================================
# Run Endpoints
# =============================================================================


@router.get(
    "/{deployment_id}/runs",
    response_model=RunListResponse,
    summary="List runs",
    description="List runs for a deployment.",
)
async def list_runs(
    deployment_id: UUID,
    current_user: UserDep,
    pagination: PaginationDep,
    state: Optional[str] = None,
) -> RunListResponse:
    """
    List runs for a deployment.
    """
    async with get_session() as session:
        # Verify deployment exists
        deploy_result = await session.execute(
            select(Deployment).where(
                Deployment.id == deployment_id,
                Deployment.deleted_at.is_(None),
            )
        )
        deployment = deploy_result.scalar_one_or_none()

        if deployment is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Deployment not found",
            )

        # Verify workspace access
        await _verify_workspace_access(session, deployment.workspace_id, current_user)

        # Build query
        query = select(Run).where(Run.deployment_id == deployment_id)
        count_query = select(func.count(Run.id)).where(Run.deployment_id == deployment_id)

        if state:
            query = query.where(Run.state == state)
            count_query = count_query.where(Run.state == state)

        # Get total
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Get runs
        query = (
            query.offset(pagination.offset).limit(pagination.limit).order_by(Run.created_at.desc())
        )
        result = await session.execute(query)
        runs = result.scalars().all()

        items = [_run_to_response(run) for run in runs]

        return RunListResponse(
            items=items,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
        )


@router.post(
    "/{deployment_id}/runs",
    response_model=RunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create run",
    description="Create a new run for a deployment.",
)
async def create_run(
    deployment_id: UUID,
    request: RunCreate,
    current_user: UserDep,
) -> RunResponse:
    """
    Create a new run.

    Requires run:create permission or superuser.
    Deployment must be in DEPLOYED state.
    """
    async with get_session() as session:
        # Verify deployment exists
        deploy_result = await session.execute(
            select(Deployment).where(
                Deployment.id == deployment_id,
                Deployment.deleted_at.is_(None),
            )
        )
        deployment = deploy_result.scalar_one_or_none()

        if deployment is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Deployment not found",
            )

        # Verify workspace access
        await _verify_workspace_access(session, deployment.workspace_id, current_user, "run:create")

        # Verify deployment is in correct state
        current_state = (
            DeploymentState(deployment.state)
            if isinstance(deployment.state, str)
            else deployment.state
        )
        if current_state != DeploymentState.DEPLOYED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot create run for deployment in state {current_state.value}",
            )

        # Check for existing active runs
        active_run_id = await _get_active_run(session, deployment.id)
        if active_run_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Deployment already has an active run",
            )

        # Create run
        run = Run(
            deployment_id=deployment_id,
            workspace_id=deployment.workspace_id,
            state=RunState.CREATED,
            is_paper_trading=request.is_paper_trading,
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)

        return _run_to_response(run)


@router.get(
    "/{deployment_id}/runs/{run_id}",
    response_model=RunResponse,
    summary="Get run",
    description="Get run by ID.",
)
async def get_run(
    deployment_id: UUID,
    run_id: UUID,
    current_user: UserDep,
) -> RunResponse:
    """Get run by ID."""
    async with get_session() as session:
        result = await session.execute(
            select(Run).where(
                Run.id == run_id,
                Run.deployment_id == deployment_id,
            )
        )
        run = result.scalar_one_or_none()

        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Run not found",
            )

        # Verify workspace access
        await _verify_workspace_access(session, run.workspace_id, current_user)

        return _run_to_response(run)


@router.patch(
    "/{deployment_id}/runs/{run_id}",
    response_model=RunResponse,
    summary="Update run",
    description="Update run metadata.",
)
async def update_run(
    deployment_id: UUID,
    run_id: UUID,
    request: RunUpdate,
    current_user: UserDep,
) -> RunResponse:
    """
    Update run.

    Requires run:write permission or superuser.
    """
    async with get_session() as session:
        result = await session.execute(
            select(Run).where(
                Run.id == run_id,
                Run.deployment_id == deployment_id,
            )
        )
        run = result.scalar_one_or_none()

        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Run not found",
            )

        # Verify workspace access
        await _verify_workspace_access(session, run.workspace_id, current_user, "run:write")

        # Update fields
        if request.error_message is not None:
            run.error_message = request.error_message
        if request.error_code is not None:
            run.error_code = request.error_code
        if request.metrics_summary is not None:
            run.metrics_summary = request.metrics_summary

        run.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(run)

        return _run_to_response(run)


@router.post(
    "/{deployment_id}/runs/{run_id}/transition",
    response_model=RunResponse,
    summary="Transition run state",
    description="Transition run to a new state.",
)
async def transition_run_state(
    deployment_id: UUID,
    run_id: UUID,
    request: RunStateTransition,
    current_user: UserDep,
) -> RunResponse:
    """
    Transition run state.

    Requires run:write permission or superuser.
    Validates state transitions according to the run state machine.
    """
    async with get_session() as session:
        result = await session.execute(
            select(Run).where(
                Run.id == run_id,
                Run.deployment_id == deployment_id,
            )
        )
        run = result.scalar_one_or_none()

        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Run not found",
            )

        # Verify workspace access
        await _verify_workspace_access(session, run.workspace_id, current_user, "run:write")

        # Get current and target states
        current_state = RunState(run.state) if isinstance(run.state, str) else run.state
        target_state = RunState(request.target_state)

        # Validate transition
        if not _validate_run_state_transition(current_state, target_state):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid state transition from {current_state.value} to {target_state.value}",
            )

        # Update state
        run.state = target_state
        run.state_changed_at = datetime.now(timezone.utc)
        run.updated_at = datetime.now(timezone.utc)

        # Update timing fields based on state
        if target_state == RunState.RUNNING and run.started_at is None:
            run.started_at = datetime.now(timezone.utc)
        elif target_state in [RunState.STOPPED, RunState.FAILED, RunState.COMPLETED]:
            run.stopped_at = datetime.now(timezone.utc)

        await session.commit()
        await session.refresh(run)

        return _run_to_response(run)
