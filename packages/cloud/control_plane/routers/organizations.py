# -*- coding: utf-8 -*-
"""
Organizations Router.

CLOUD ZONE ONLY.

Provides CRUD endpoints for organization management.
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
from ..dependencies import PaginationDep, UserDep, require_permission
from ..models import Organization, Workspace

router = APIRouter()


# Request/Response models
class OrganizationCreate(BaseModel):
    """Create organization request."""

    name: str = Field(..., min_length=1, max_length=255)
    display_name: Optional[str] = Field(None, max_length=255)
    settings: dict = Field(default_factory=dict)


class OrganizationUpdate(BaseModel):
    """Update organization request."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    display_name: Optional[str] = Field(None, max_length=255)
    settings: Optional[dict] = None
    is_active: Optional[bool] = None


class OrganizationResponse(BaseModel):
    """Organization response."""

    id: UUID
    name: str
    display_name: Optional[str]
    settings: dict
    is_active: bool
    created_at: datetime
    updated_at: datetime
    workspace_count: int = 0


class OrganizationListResponse(BaseModel):
    """Paginated organization list response."""

    items: List[OrganizationResponse]
    total: int
    page: int
    page_size: int


@router.get(
    "",
    response_model=OrganizationListResponse,
    summary="List organizations",
    description="List all organizations (superuser only).",
)
async def list_organizations(
    current_user: UserDep,
    pagination: PaginationDep,
) -> OrganizationListResponse:
    """
    List all organizations.

    Requires superuser privileges.
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superuser privileges required",
        )

    async with get_session() as session:
        # Count total
        count_query = select(func.count(Organization.id)).where(Organization.is_active == True)
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Get organizations with workspace count
        query = (
            select(Organization)
            .where(Organization.is_active == True)
            .offset(pagination.offset)
            .limit(pagination.limit)
            .order_by(Organization.created_at.desc())
        )
        result = await session.execute(query)
        organizations = result.scalars().all()

        # Get workspace counts
        items = []
        for org in organizations:
            ws_count_query = select(func.count(Workspace.id)).where(
                Workspace.organization_id == org.id
            )
            ws_count_result = await session.execute(ws_count_query)
            ws_count = ws_count_result.scalar() or 0

            items.append(
                OrganizationResponse(
                    id=org.id,
                    name=org.name,
                    display_name=org.display_name,
                    settings=org.settings or {},
                    is_active=org.is_active,
                    created_at=org.created_at,
                    updated_at=org.updated_at,
                    workspace_count=ws_count,
                )
            )

        return OrganizationListResponse(
            items=items,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
        )


@router.post(
    "",
    response_model=OrganizationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create organization",
    description="Create a new organization (superuser only).",
)
async def create_organization(
    request: OrganizationCreate,
    current_user: UserDep,
) -> OrganizationResponse:
    """
    Create a new organization.

    Requires superuser privileges.
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superuser privileges required",
        )

    async with get_session() as session:
        # Check name uniqueness
        existing = await session.execute(
            select(Organization).where(Organization.name == request.name)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Organization with name '{request.name}' already exists",
            )

        org = Organization(
            name=request.name,
            display_name=request.display_name,
            settings=request.settings,
        )
        session.add(org)
        await session.commit()
        await session.refresh(org)

        return OrganizationResponse(
            id=org.id,
            name=org.name,
            display_name=org.display_name,
            settings=org.settings or {},
            is_active=org.is_active,
            created_at=org.created_at,
            updated_at=org.updated_at,
            workspace_count=0,
        )


@router.get(
    "/{org_id}",
    response_model=OrganizationResponse,
    summary="Get organization",
    description="Get organization by ID.",
)
async def get_organization(
    org_id: UUID,
    current_user: UserDep,
) -> OrganizationResponse:
    """
    Get organization by ID.

    Users can only view their own organization unless superuser.
    """
    async with get_session() as session:
        result = await session.execute(select(Organization).where(Organization.id == org_id))
        org = result.scalar_one_or_none()

        if org is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found",
            )

        # Check access
        if not current_user.is_superuser and current_user.org_id != org_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this organization",
            )

        # Get workspace count
        ws_count_query = select(func.count(Workspace.id)).where(Workspace.organization_id == org.id)
        ws_count_result = await session.execute(ws_count_query)
        ws_count = ws_count_result.scalar() or 0

        return OrganizationResponse(
            id=org.id,
            name=org.name,
            display_name=org.display_name,
            settings=org.settings or {},
            is_active=org.is_active,
            created_at=org.created_at,
            updated_at=org.updated_at,
            workspace_count=ws_count,
        )


@router.patch(
    "/{org_id}",
    response_model=OrganizationResponse,
    summary="Update organization",
    description="Update organization settings.",
)
async def update_organization(
    org_id: UUID,
    request: OrganizationUpdate,
    current_user: UserDep,
) -> OrganizationResponse:
    """
    Update organization.

    Requires org:write permission or superuser.
    """
    async with get_session() as session:
        result = await session.execute(select(Organization).where(Organization.id == org_id))
        org = result.scalar_one_or_none()

        if org is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found",
            )

        # Check access
        if not current_user.is_superuser:
            if current_user.org_id != org_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this organization",
                )
            if not current_user.has_permission("org:write"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission required: org:write",
                )

        # Update fields
        if request.name is not None:
            # Check name uniqueness
            existing = await session.execute(
                select(Organization).where(
                    Organization.name == request.name,
                    Organization.id != org_id,
                )
            )
            if existing.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Organization with name '{request.name}' already exists",
                )
            org.name = request.name

        if request.display_name is not None:
            org.display_name = request.display_name
        if request.settings is not None:
            org.settings = request.settings
        if request.is_active is not None:
            org.is_active = request.is_active

        org.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(org)

        # Get workspace count
        ws_count_query = select(func.count(Workspace.id)).where(Workspace.organization_id == org.id)
        ws_count_result = await session.execute(ws_count_query)
        ws_count = ws_count_result.scalar() or 0

        return OrganizationResponse(
            id=org.id,
            name=org.name,
            display_name=org.display_name,
            settings=org.settings or {},
            is_active=org.is_active,
            created_at=org.created_at,
            updated_at=org.updated_at,
            workspace_count=ws_count,
        )


@router.delete(
    "/{org_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete organization",
    description="Soft delete organization (superuser only).",
)
async def delete_organization(
    org_id: UUID,
    current_user: UserDep,
) -> None:
    """
    Soft delete organization.

    Requires superuser privileges. Sets is_active=False.
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superuser privileges required",
        )

    async with get_session() as session:
        result = await session.execute(select(Organization).where(Organization.id == org_id))
        org = result.scalar_one_or_none()

        if org is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found",
            )

        org.is_active = False
        org.updated_at = datetime.now(timezone.utc)
        await session.commit()
