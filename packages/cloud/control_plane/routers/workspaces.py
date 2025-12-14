# -*- coding: utf-8 -*-
"""
Workspaces Router.

CLOUD ZONE ONLY.

Provides CRUD endpoints for workspace management.
Workspaces are the primary tenant boundary in the system.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from ..database import get_session
from ..dependencies import PaginationDep, UserDep
from ..models import Agent, Deployment, Organization, Workspace

router = APIRouter()


# Request/Response models
class WorkspaceCreate(BaseModel):
    """Create workspace request."""

    name: str = Field(..., min_length=1, max_length=255)
    display_name: Optional[str] = Field(None, max_length=255)
    organization_id: UUID
    settings: dict = Field(default_factory=dict)


class WorkspaceUpdate(BaseModel):
    """Update workspace request."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    display_name: Optional[str] = Field(None, max_length=255)
    settings: Optional[dict] = None
    is_active: Optional[bool] = None


class WorkspaceResponse(BaseModel):
    """Workspace response."""

    id: UUID
    name: str
    display_name: Optional[str]
    organization_id: UUID
    settings: dict
    is_active: bool
    created_at: datetime
    updated_at: datetime
    agent_count: int = 0
    deployment_count: int = 0


class WorkspaceListResponse(BaseModel):
    """Paginated workspace list response."""

    items: List[WorkspaceResponse]
    total: int
    page: int
    page_size: int


async def _get_workspace_stats(session, workspace_id: UUID) -> tuple[int, int]:
    """Get agent and deployment counts for a workspace."""
    agent_count_q = select(func.count(Agent.id)).where(
        Agent.workspace_id == workspace_id
    )
    agent_result = await session.execute(agent_count_q)
    agent_count = agent_result.scalar() or 0

    deploy_count_q = select(func.count(Deployment.id)).where(
        Deployment.workspace_id == workspace_id
    )
    deploy_result = await session.execute(deploy_count_q)
    deploy_count = deploy_result.scalar() or 0

    return agent_count, deploy_count


@router.get(
    "",
    response_model=WorkspaceListResponse,
    summary="List workspaces",
    description="List workspaces accessible to the user.",
)
async def list_workspaces(
    current_user: UserDep,
    pagination: PaginationDep,
    organization_id: Optional[UUID] = None,
) -> WorkspaceListResponse:
    """
    List workspaces.

    Users see workspaces in their organization.
    Superusers can see all or filter by organization.
    """
    async with get_session() as session:
        # Build base query
        query = select(Workspace).where(Workspace.is_active == True)
        count_query = select(func.count(Workspace.id)).where(Workspace.is_active == True)

        # Apply organization filter
        if current_user.is_superuser:
            if organization_id:
                query = query.where(Workspace.organization_id == organization_id)
                count_query = count_query.where(
                    Workspace.organization_id == organization_id
                )
        else:
            # Non-superusers can only see their organization's workspaces
            if current_user.org_id is None:
                return WorkspaceListResponse(
                    items=[],
                    total=0,
                    page=pagination.page,
                    page_size=pagination.page_size,
                )
            query = query.where(Workspace.organization_id == current_user.org_id)
            count_query = count_query.where(
                Workspace.organization_id == current_user.org_id
            )

        # Get total count
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Get workspaces
        query = (
            query.offset(pagination.offset)
            .limit(pagination.limit)
            .order_by(Workspace.created_at.desc())
        )
        result = await session.execute(query)
        workspaces = result.scalars().all()

        items = []
        for ws in workspaces:
            agent_count, deploy_count = await _get_workspace_stats(session, ws.id)
            items.append(
                WorkspaceResponse(
                    id=ws.id,
                    name=ws.name,
                    display_name=ws.display_name,
                    organization_id=ws.organization_id,
                    settings=ws.settings or {},
                    is_active=ws.is_active,
                    created_at=ws.created_at,
                    updated_at=ws.updated_at,
                    agent_count=agent_count,
                    deployment_count=deploy_count,
                )
            )

        return WorkspaceListResponse(
            items=items,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
        )


@router.post(
    "",
    response_model=WorkspaceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create workspace",
    description="Create a new workspace in an organization.",
)
async def create_workspace(
    request: WorkspaceCreate,
    current_user: UserDep,
) -> WorkspaceResponse:
    """
    Create a new workspace.

    Requires workspace:create permission or superuser.
    """
    async with get_session() as session:
        # Check organization access
        if not current_user.is_superuser:
            if current_user.org_id != request.organization_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cannot create workspace in another organization",
                )
            if not current_user.has_permission("workspace:create"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission required: workspace:create",
                )

        # Verify organization exists
        org_result = await session.execute(
            select(Organization).where(
                Organization.id == request.organization_id,
                Organization.is_active == True,
            )
        )
        org = org_result.scalar_one_or_none()
        if org is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found",
            )

        # Check name uniqueness within organization
        existing = await session.execute(
            select(Workspace).where(
                Workspace.organization_id == request.organization_id,
                Workspace.name == request.name,
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Workspace with name '{request.name}' already exists in this organization",
            )

        workspace = Workspace(
            name=request.name,
            display_name=request.display_name,
            organization_id=request.organization_id,
            settings=request.settings,
        )
        session.add(workspace)
        await session.commit()
        await session.refresh(workspace)

        return WorkspaceResponse(
            id=workspace.id,
            name=workspace.name,
            display_name=workspace.display_name,
            organization_id=workspace.organization_id,
            settings=workspace.settings or {},
            is_active=workspace.is_active,
            created_at=workspace.created_at,
            updated_at=workspace.updated_at,
            agent_count=0,
            deployment_count=0,
        )


@router.get(
    "/{workspace_id}",
    response_model=WorkspaceResponse,
    summary="Get workspace",
    description="Get workspace by ID.",
)
async def get_workspace(
    workspace_id: UUID,
    current_user: UserDep,
) -> WorkspaceResponse:
    """
    Get workspace by ID.

    Users can only view workspaces in their organization.
    """
    async with get_session() as session:
        result = await session.execute(
            select(Workspace).where(Workspace.id == workspace_id)
        )
        workspace = result.scalar_one_or_none()

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

        agent_count, deploy_count = await _get_workspace_stats(session, workspace.id)

        return WorkspaceResponse(
            id=workspace.id,
            name=workspace.name,
            display_name=workspace.display_name,
            organization_id=workspace.organization_id,
            settings=workspace.settings or {},
            is_active=workspace.is_active,
            created_at=workspace.created_at,
            updated_at=workspace.updated_at,
            agent_count=agent_count,
            deployment_count=deploy_count,
        )


@router.patch(
    "/{workspace_id}",
    response_model=WorkspaceResponse,
    summary="Update workspace",
    description="Update workspace settings.",
)
async def update_workspace(
    workspace_id: UUID,
    request: WorkspaceUpdate,
    current_user: UserDep,
) -> WorkspaceResponse:
    """
    Update workspace.

    Requires workspace:write permission or superuser.
    """
    async with get_session() as session:
        result = await session.execute(
            select(Workspace).where(Workspace.id == workspace_id)
        )
        workspace = result.scalar_one_or_none()

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
            if not current_user.has_permission("workspace:write"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission required: workspace:write",
                )

        # Update fields
        if request.name is not None:
            # Check name uniqueness within organization
            existing = await session.execute(
                select(Workspace).where(
                    Workspace.organization_id == workspace.organization_id,
                    Workspace.name == request.name,
                    Workspace.id != workspace_id,
                )
            )
            if existing.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Workspace with name '{request.name}' already exists in this organization",
                )
            workspace.name = request.name

        if request.display_name is not None:
            workspace.display_name = request.display_name
        if request.settings is not None:
            workspace.settings = request.settings
        if request.is_active is not None:
            workspace.is_active = request.is_active

        workspace.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(workspace)

        agent_count, deploy_count = await _get_workspace_stats(session, workspace.id)

        return WorkspaceResponse(
            id=workspace.id,
            name=workspace.name,
            display_name=workspace.display_name,
            organization_id=workspace.organization_id,
            settings=workspace.settings or {},
            is_active=workspace.is_active,
            created_at=workspace.created_at,
            updated_at=workspace.updated_at,
            agent_count=agent_count,
            deployment_count=deploy_count,
        )


@router.delete(
    "/{workspace_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete workspace",
    description="Soft delete workspace.",
)
async def delete_workspace(
    workspace_id: UUID,
    current_user: UserDep,
) -> None:
    """
    Soft delete workspace.

    Requires workspace:delete permission or superuser.
    Sets is_active=False.
    """
    async with get_session() as session:
        result = await session.execute(
            select(Workspace).where(Workspace.id == workspace_id)
        )
        workspace = result.scalar_one_or_none()

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
            if not current_user.has_permission("workspace:delete"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission required: workspace:delete",
                )

        workspace.is_active = False
        workspace.updated_at = datetime.now(timezone.utc)
        await session.commit()
