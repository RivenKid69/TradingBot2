# -*- coding: utf-8 -*-
"""
Agents Router.

CLOUD ZONE ONLY.

Provides endpoints for agent management, enrollment tokens, and trust management.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from ..database import TenantContext, get_session
from ..dependencies import PaginationDep, UserDep
from ..models import Agent, AgentEnrollmentToken, Deployment, DeploymentState, TrustState, Workspace

router = APIRouter()


# Request/Response models
class EnrollmentTokenCreate(BaseModel):
    """Create enrollment token request."""

    workspace_id: UUID
    expires_in_hours: int = Field(default=24, ge=1, le=720)  # Max 30 days
    description: Optional[str] = Field(None, max_length=500)


class EnrollmentTokenResponse(BaseModel):
    """Enrollment token response."""

    id: UUID
    token: str  # Raw token (only shown once!)
    workspace_id: UUID
    expires_at: datetime
    description: Optional[str]
    is_used: bool
    created_at: datetime


class EnrollmentTokenListItem(BaseModel):
    """Enrollment token list item (without raw token)."""

    id: UUID
    workspace_id: UUID
    expires_at: datetime
    description: Optional[str]
    is_used: bool
    used_at: Optional[datetime]
    agent_id: Optional[UUID]
    created_at: datetime


class EnrollmentTokenListResponse(BaseModel):
    """Paginated enrollment token list response."""

    items: List[EnrollmentTokenListItem]
    total: int
    page: int
    page_size: int


class AgentUpdate(BaseModel):
    """Update agent request."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    trust_state: Optional[TrustState] = None
    capabilities: Optional[List[str]] = None


class AgentResponse(BaseModel):
    """Agent response."""

    id: UUID
    name: str
    workspace_id: UUID
    trust_state: TrustState
    public_key: str
    agent_version: str
    capabilities: List[str]
    last_seen: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    active_deployments: int = 0


class AgentListResponse(BaseModel):
    """Paginated agent list response."""

    items: List[AgentResponse]
    total: int
    page: int
    page_size: int


class AgentRevokeRequest(BaseModel):
    """Revoke agent request."""

    reason: str = Field(..., min_length=1, max_length=500)


async def _get_agent_deployment_count(session, agent_id: UUID) -> int:
    """Get active deployment count for an agent."""
    # Use lowercase values matching DeploymentState enum values
    active_states = [
        DeploymentState.DEPLOYING.value,  # "deploying"
        DeploymentState.DEPLOYED.value,    # "deployed"
    ]
    count_q = select(func.count(Deployment.id)).where(
        Deployment.agent_id == agent_id,
        Deployment.state.in_(active_states),
    )
    result = await session.execute(count_q)
    return result.scalar() or 0


def _agent_to_response(agent: Agent, deployment_count: int = 0) -> AgentResponse:
    """Convert Agent model to response."""
    return AgentResponse(
        id=agent.id,
        name=agent.name,
        workspace_id=agent.workspace_id,
        trust_state=agent.trust_state,
        public_key=agent.public_key,
        agent_version=agent.agent_version,
        capabilities=agent.capabilities or [],
        last_seen=agent.last_seen_at,
        created_at=agent.created_at,
        updated_at=agent.updated_at,
        active_deployments=deployment_count,
    )


# Enrollment Token endpoints
@router.post(
    "/enrollment-tokens",
    response_model=EnrollmentTokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create enrollment token",
    description="Create a new enrollment token for agent enrollment.",
)
async def create_enrollment_token(
    request: EnrollmentTokenCreate,
    current_user: UserDep,
) -> EnrollmentTokenResponse:
    """
    Create enrollment token.

    Requires agent:enroll permission or superuser.
    Returns raw token - only shown once!
    """
    async with get_session() as session:
        # Verify workspace access
        ws_result = await session.execute(
            select(Workspace).where(
                Workspace.id == request.workspace_id,
                Workspace.is_active == True,
            )
        )
        workspace = ws_result.scalar_one_or_none()

        if workspace is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace not found",
            )

        # Check access
        if not current_user.is_superuser:
            if current_user.org_id != workspace.organization_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this workspace",
                )
            if not current_user.has_permission("agent:enroll"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission required: agent:enroll",
                )

        # Generate token
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()

        enrollment_token = AgentEnrollmentToken(
            workspace_id=request.workspace_id,
            token_hash=token_hash,
            expires_at=datetime.now(timezone.utc)
            + timedelta(hours=request.expires_in_hours),
            created_by_user_id=current_user.id,
            name=request.description or f"Token {token_hash[:8]}",
        )
        session.add(enrollment_token)
        await session.commit()
        await session.refresh(enrollment_token)

        return EnrollmentTokenResponse(
            id=enrollment_token.id,
            token=raw_token,  # Only time raw token is exposed!
            workspace_id=enrollment_token.workspace_id,
            expires_at=enrollment_token.expires_at,
            description=enrollment_token.name,  # Model uses 'name' field
            is_used=enrollment_token.is_used,
            created_at=enrollment_token.created_at,
        )


@router.get(
    "/enrollment-tokens",
    response_model=EnrollmentTokenListResponse,
    summary="List enrollment tokens",
    description="List enrollment tokens for a workspace.",
)
async def list_enrollment_tokens(
    current_user: UserDep,
    pagination: PaginationDep,
    workspace_id: Optional[UUID] = None,
    include_used: bool = False,
) -> EnrollmentTokenListResponse:
    """
    List enrollment tokens.

    Requires agent:enroll permission or superuser.
    """
    async with get_session() as session:
        # Build query
        query = select(AgentEnrollmentToken)
        count_query = select(func.count(AgentEnrollmentToken.id))

        if workspace_id:
            # Verify workspace access
            ws_result = await session.execute(
                select(Workspace).where(Workspace.id == workspace_id)
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
                        detail="Access denied",
                    )

            query = query.where(AgentEnrollmentToken.workspace_id == workspace_id)
            count_query = count_query.where(
                AgentEnrollmentToken.workspace_id == workspace_id
            )
        elif not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="workspace_id required for non-superusers",
            )

        if not include_used:
            query = query.where(AgentEnrollmentToken.is_used == False)
            count_query = count_query.where(AgentEnrollmentToken.is_used == False)

        # Get total
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Get tokens
        query = (
            query.offset(pagination.offset)
            .limit(pagination.limit)
            .order_by(AgentEnrollmentToken.created_at.desc())
        )
        result = await session.execute(query)
        tokens = result.scalars().all()

        items = [
            EnrollmentTokenListItem(
                id=token.id,
                workspace_id=token.workspace_id,
                expires_at=token.expires_at,
                description=token.name,  # Model uses 'name' field
                is_used=token.is_used,
                used_at=token.used_at,
                agent_id=token.used_by_agent_id,  # Model uses 'used_by_agent_id'
                created_at=token.created_at,
            )
            for token in tokens
        ]

        return EnrollmentTokenListResponse(
            items=items,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
        )


@router.delete(
    "/enrollment-tokens/{token_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete enrollment token",
    description="Delete an unused enrollment token.",
)
async def delete_enrollment_token(
    token_id: UUID,
    current_user: UserDep,
) -> None:
    """
    Delete enrollment token.

    Only unused tokens can be deleted.
    """
    async with get_session() as session:
        result = await session.execute(
            select(AgentEnrollmentToken).where(AgentEnrollmentToken.id == token_id)
        )
        token = result.scalar_one_or_none()

        if token is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Enrollment token not found",
            )

        if token.is_used:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete used enrollment token",
            )

        # Check workspace access
        ws_result = await session.execute(
            select(Workspace).where(Workspace.id == token.workspace_id)
        )
        workspace = ws_result.scalar_one_or_none()

        if not current_user.is_superuser:
            if workspace is None or current_user.org_id != workspace.organization_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied",
                )

        await session.delete(token)
        await session.commit()


# Agent management endpoints
@router.get(
    "",
    response_model=AgentListResponse,
    summary="List agents",
    description="List agents in a workspace.",
)
async def list_agents(
    current_user: UserDep,
    pagination: PaginationDep,
    workspace_id: Optional[UUID] = None,
    trust_state: Optional[TrustState] = None,
) -> AgentListResponse:
    """
    List agents.

    Users can list agents in their organization's workspaces.
    """
    async with get_session() as session:
        # Build query
        query = select(Agent)
        count_query = select(func.count(Agent.id))

        if workspace_id:
            # Verify workspace access
            ws_result = await session.execute(
                select(Workspace).where(Workspace.id == workspace_id)
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
                        detail="Access denied",
                    )

            query = query.where(Agent.workspace_id == workspace_id)
            count_query = count_query.where(Agent.workspace_id == workspace_id)
        elif not current_user.is_superuser:
            # Get all workspaces in user's org
            ws_ids_q = select(Workspace.id).where(
                Workspace.organization_id == current_user.org_id
            )
            query = query.where(Agent.workspace_id.in_(ws_ids_q))
            count_query = count_query.where(Agent.workspace_id.in_(ws_ids_q))

        if trust_state:
            query = query.where(Agent.trust_state == trust_state)
            count_query = count_query.where(Agent.trust_state == trust_state)

        # Get total
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Get agents
        query = (
            query.offset(pagination.offset)
            .limit(pagination.limit)
            .order_by(Agent.created_at.desc())
        )
        result = await session.execute(query)
        agents = result.scalars().all()

        items = []
        for agent in agents:
            deploy_count = await _get_agent_deployment_count(session, agent.id)
            items.append(_agent_to_response(agent, deploy_count))

        return AgentListResponse(
            items=items,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
        )


@router.get(
    "/{agent_id}",
    response_model=AgentResponse,
    summary="Get agent",
    description="Get agent by ID.",
)
async def get_agent(
    agent_id: UUID,
    current_user: UserDep,
) -> AgentResponse:
    """
    Get agent by ID.
    """
    async with get_session() as session:
        result = await session.execute(select(Agent).where(Agent.id == agent_id))
        agent = result.scalar_one_or_none()

        if agent is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found",
            )

        # Check workspace access
        ws_result = await session.execute(
            select(Workspace).where(Workspace.id == agent.workspace_id)
        )
        workspace = ws_result.scalar_one_or_none()

        if not current_user.is_superuser:
            if workspace is None or current_user.org_id != workspace.organization_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied",
                )

        deploy_count = await _get_agent_deployment_count(session, agent.id)
        return _agent_to_response(agent, deploy_count)


@router.patch(
    "/{agent_id}",
    response_model=AgentResponse,
    summary="Update agent",
    description="Update agent settings.",
)
async def update_agent(
    agent_id: UUID,
    request: AgentUpdate,
    current_user: UserDep,
) -> AgentResponse:
    """
    Update agent.

    Requires agent:write permission or superuser.
    """
    async with get_session() as session:
        result = await session.execute(select(Agent).where(Agent.id == agent_id))
        agent = result.scalar_one_or_none()

        if agent is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found",
            )

        # Check workspace access
        ws_result = await session.execute(
            select(Workspace).where(Workspace.id == agent.workspace_id)
        )
        workspace = ws_result.scalar_one_or_none()

        if not current_user.is_superuser:
            if workspace is None or current_user.org_id != workspace.organization_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied",
                )
            if not current_user.has_permission("agent:write"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission required: agent:write",
                )

        # Update fields
        if request.name is not None:
            agent.name = request.name

        if request.trust_state is not None:
            # Trust state changes require special permission
            if not current_user.is_superuser and not current_user.has_permission(
                "agent:trust"
            ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission required: agent:trust",
                )
            agent.trust_state = request.trust_state

        if request.capabilities is not None:
            agent.capabilities = request.capabilities

        agent.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(agent)

        deploy_count = await _get_agent_deployment_count(session, agent.id)
        return _agent_to_response(agent, deploy_count)


@router.post(
    "/{agent_id}/revoke",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke agent",
    description="Revoke agent trust, preventing it from operating.",
)
async def revoke_agent(
    agent_id: UUID,
    request: AgentRevokeRequest,
    current_user: UserDep,
) -> None:
    """
    Revoke agent trust.

    Requires agent:trust permission or superuser.
    Sets trust_state to REVOKED.
    """
    async with get_session() as session:
        result = await session.execute(select(Agent).where(Agent.id == agent_id))
        agent = result.scalar_one_or_none()

        if agent is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found",
            )

        # Check workspace access
        ws_result = await session.execute(
            select(Workspace).where(Workspace.id == agent.workspace_id)
        )
        workspace = ws_result.scalar_one_or_none()

        if not current_user.is_superuser:
            if workspace is None or current_user.org_id != workspace.organization_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied",
                )
            if not current_user.has_permission("agent:trust"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission required: agent:trust",
                )

        agent.trust_state = TrustState.REVOKED
        agent.revocation_reason = request.reason
        agent.updated_at = datetime.now(timezone.utc)
        await session.commit()


@router.post(
    "/{agent_id}/reinstate",
    response_model=AgentResponse,
    summary="Reinstate agent",
    description="Reinstate a revoked or suspended agent.",
)
async def reinstate_agent(
    agent_id: UUID,
    current_user: UserDep,
) -> AgentResponse:
    """
    Reinstate agent.

    Requires agent:trust permission or superuser.
    Changes trust_state from REVOKED/SUSPENDED to ENROLLED.
    """
    async with get_session() as session:
        result = await session.execute(select(Agent).where(Agent.id == agent_id))
        agent = result.scalar_one_or_none()

        if agent is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found",
            )

        if agent.trust_state == TrustState.ENROLLED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Agent is already enrolled",
            )

        # Check workspace access
        ws_result = await session.execute(
            select(Workspace).where(Workspace.id == agent.workspace_id)
        )
        workspace = ws_result.scalar_one_or_none()

        if not current_user.is_superuser:
            if workspace is None or current_user.org_id != workspace.organization_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied",
                )
            if not current_user.has_permission("agent:trust"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission required: agent:trust",
                )

        agent.trust_state = TrustState.ENROLLED
        agent.revocation_reason = None
        agent.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(agent)

        deploy_count = await _get_agent_deployment_count(session, agent.id)
        return _agent_to_response(agent, deploy_count)
