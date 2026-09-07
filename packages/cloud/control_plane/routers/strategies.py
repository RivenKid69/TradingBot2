# -*- coding: utf-8 -*-
"""
Strategies Router.

CLOUD ZONE ONLY.

Provides CRUD endpoints for strategy, version, build, and artifact management.
Implements the full strategy lifecycle from code to deployment-ready artifacts.
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
    Artifact,
    Build,
    ChangeClass,
    Strategy,
    StrategyVersion,
    Workspace,
)

router = APIRouter()


# =============================================================================
# Request/Response Models - Strategy
# =============================================================================


class StrategyCreate(BaseModel):
    """Create strategy request."""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    workspace_id: UUID
    git_repo: Optional[str] = Field(None, max_length=500)
    tags: List[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class StrategyUpdate(BaseModel):
    """Update strategy request."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    git_repo: Optional[str] = Field(None, max_length=500)
    tags: Optional[List[str]] = None
    metadata: Optional[dict] = None
    is_active: Optional[bool] = None


class StrategyResponse(BaseModel):
    """Strategy response."""

    id: UUID
    name: str
    description: Optional[str]
    workspace_id: UUID
    git_repo: Optional[str]
    tags: List[str]
    metadata: dict
    is_active: bool
    created_at: datetime
    updated_at: datetime
    version_count: int = 0
    latest_version: Optional[str] = None


class StrategyListResponse(BaseModel):
    """Paginated strategy list response."""

    items: List[StrategyResponse]
    total: int
    page: int
    page_size: int


# =============================================================================
# Request/Response Models - StrategyVersion
# =============================================================================


class StrategyVersionCreate(BaseModel):
    """Create strategy version request."""

    version: str = Field(..., min_length=1, max_length=50, pattern=r"^\d+\.\d+\.\d+.*$")
    git_sha: str = Field(..., min_length=40, max_length=40)
    git_tag: Optional[str] = Field(None, max_length=100)
    changelog: Optional[str] = Field(None, max_length=5000)
    metadata: dict = Field(default_factory=dict)


class StrategyVersionUpdate(BaseModel):
    """Update strategy version request."""

    is_deprecated: Optional[bool] = None
    changelog: Optional[str] = Field(None, max_length=5000)
    metadata: Optional[dict] = None


class StrategyVersionResponse(BaseModel):
    """Strategy version response."""

    id: UUID
    strategy_id: UUID
    workspace_id: UUID
    version: str
    git_sha: str
    git_tag: Optional[str]
    changelog: Optional[str]
    is_latest: bool
    is_deprecated: bool
    metadata: dict
    created_at: datetime
    updated_at: datetime
    build_count: int = 0


class StrategyVersionListResponse(BaseModel):
    """Paginated strategy version list response."""

    items: List[StrategyVersionResponse]
    total: int
    page: int
    page_size: int


# =============================================================================
# Request/Response Models - Build
# =============================================================================


class BuildCreate(BaseModel):
    """Create build request."""

    builder_id: Optional[str] = Field(None, max_length=255)
    ci_job_id: Optional[str] = Field(None, max_length=255)
    ci_pipeline_url: Optional[str] = Field(None, max_length=500)
    metadata: dict = Field(default_factory=dict)


class BuildUpdate(BaseModel):
    """Update build request."""

    status: Optional[str] = Field(None, pattern=r"^(pending|building|success|failed|cancelled)$")
    logs_url: Optional[str] = Field(None, max_length=500)
    finished_at: Optional[datetime] = None
    metadata: Optional[dict] = None


class BuildResponse(BaseModel):
    """Build response."""

    id: UUID
    strategy_version_id: UUID
    workspace_id: UUID
    build_number: int
    builder_id: Optional[str]
    ci_job_id: Optional[str]
    ci_pipeline_url: Optional[str]
    status: str
    started_at: datetime
    finished_at: Optional[datetime]
    logs_url: Optional[str]
    metadata: dict
    created_at: datetime
    updated_at: datetime
    artifact_count: int = 0


class BuildListResponse(BaseModel):
    """Paginated build list response."""

    items: List[BuildResponse]
    total: int
    page: int
    page_size: int


# =============================================================================
# Request/Response Models - Artifact
# =============================================================================


class ArtifactCreate(BaseModel):
    """Create artifact request."""

    name: str = Field(..., min_length=1, max_length=255)
    format: str = Field(..., min_length=1, max_length=50)
    digest: str = Field(..., min_length=64, max_length=64)  # SHA256
    size_bytes: int = Field(..., ge=0)
    registry_url: str = Field(..., min_length=1, max_length=500)
    signature_ref: Optional[str] = Field(None, max_length=500)
    sbom_ref: Optional[str] = Field(None, max_length=500)
    provenance: dict = Field(default_factory=dict)
    change_class: str = Field(
        default="OPERATIONAL",
        pattern=r"^(OPERATIONAL|TRADING_IMPACTING|SECURITY_SENSITIVE|DATA_SENSITIVE)$",
    )
    metadata: dict = Field(default_factory=dict)


class ArtifactResponse(BaseModel):
    """Artifact response."""

    id: UUID
    build_id: UUID
    workspace_id: UUID
    name: str
    format: str
    digest: str
    size_bytes: int
    registry_url: str
    signature_ref: Optional[str]
    sbom_ref: Optional[str]
    provenance: dict
    change_class: str
    is_signed: bool
    metadata: dict
    created_at: datetime
    updated_at: datetime


class ArtifactListResponse(BaseModel):
    """Paginated artifact list response."""

    items: List[ArtifactResponse]
    total: int
    page: int
    page_size: int


# =============================================================================
# Helper Functions
# =============================================================================


async def _get_version_count(session, strategy_id: UUID) -> int:
    """Get the count of versions for a strategy."""
    result = await session.execute(
        select(func.count(StrategyVersion.id)).where(StrategyVersion.strategy_id == strategy_id)
    )
    return result.scalar() or 0


async def _get_latest_version(session, strategy_id: UUID) -> Optional[str]:
    """Get the latest version string for a strategy."""
    result = await session.execute(
        select(StrategyVersion.version).where(
            StrategyVersion.strategy_id == strategy_id,
            StrategyVersion.is_latest == True,
        )
    )
    version = result.scalar_one_or_none()
    return version


async def _get_build_count(session, version_id: UUID) -> int:
    """Get the count of builds for a version."""
    result = await session.execute(
        select(func.count(Build.id)).where(Build.strategy_version_id == version_id)
    )
    return result.scalar() or 0


async def _get_artifact_count(session, build_id: UUID) -> int:
    """Get the count of artifacts for a build."""
    result = await session.execute(
        select(func.count(Artifact.id)).where(Artifact.build_id == build_id)
    )
    return result.scalar() or 0


async def _get_next_build_number(session, version_id: UUID) -> int:
    """Get the next build number for a version."""
    result = await session.execute(
        select(func.max(Build.build_number)).where(Build.strategy_version_id == version_id)
    )
    max_num = result.scalar()
    return (max_num or 0) + 1


def _strategy_to_response(
    strategy: Strategy,
    version_count: int = 0,
    latest_version: Optional[str] = None,
) -> StrategyResponse:
    """Convert Strategy model to response."""
    return StrategyResponse(
        id=strategy.id,
        name=strategy.name,
        description=strategy.description,
        workspace_id=strategy.workspace_id,
        git_repo=strategy.git_repo,
        tags=strategy.tags or [],
        metadata=strategy.extra_metadata or {},
        is_active=strategy.is_active,
        created_at=strategy.created_at,
        updated_at=strategy.updated_at,
        version_count=version_count,
        latest_version=latest_version,
    )


def _version_to_response(version: StrategyVersion, build_count: int = 0) -> StrategyVersionResponse:
    """Convert StrategyVersion model to response."""
    return StrategyVersionResponse(
        id=version.id,
        strategy_id=version.strategy_id,
        workspace_id=version.workspace_id,
        version=version.version,
        git_sha=version.git_sha,
        git_tag=version.git_tag,
        changelog=version.changelog,
        is_latest=version.is_latest,
        is_deprecated=version.is_deprecated,
        metadata=version.extra_metadata or {},
        created_at=version.created_at,
        updated_at=version.updated_at,
        build_count=build_count,
    )


def _build_to_response(build: Build, artifact_count: int = 0) -> BuildResponse:
    """Convert Build model to response."""
    return BuildResponse(
        id=build.id,
        strategy_version_id=build.strategy_version_id,
        workspace_id=build.workspace_id,
        build_number=build.build_number,
        builder_id=build.builder_id,
        ci_job_id=build.ci_job_id,
        ci_pipeline_url=build.ci_pipeline_url,
        status=build.status,
        started_at=build.started_at,
        finished_at=build.finished_at,
        logs_url=build.logs_url,
        metadata=build.extra_metadata or {},
        created_at=build.created_at,
        updated_at=build.updated_at,
        artifact_count=artifact_count,
    )


def _artifact_to_response(artifact: Artifact) -> ArtifactResponse:
    """Convert Artifact model to response."""
    return ArtifactResponse(
        id=artifact.id,
        build_id=artifact.build_id,
        workspace_id=artifact.workspace_id,
        name=artifact.name,
        format=artifact.format,
        digest=artifact.digest,
        size_bytes=artifact.size_bytes,
        registry_url=artifact.registry_url,
        signature_ref=artifact.signature_ref,
        sbom_ref=artifact.sbom_ref,
        provenance=artifact.provenance or {},
        change_class=(
            artifact.change_class.value
            if hasattr(artifact.change_class, "value")
            else artifact.change_class
        ),
        is_signed=artifact.is_signed,
        metadata=artifact.extra_metadata or {},
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
    )


# =============================================================================
# Strategy Endpoints
# =============================================================================


@router.get(
    "",
    response_model=StrategyListResponse,
    summary="List strategies",
    description="List strategies in the workspace.",
)
async def list_strategies(
    current_user: UserDep,
    pagination: PaginationDep,
    workspace_id: Optional[UUID] = None,
) -> StrategyListResponse:
    """
    List strategies.

    Users see strategies in their workspace.
    Superusers can see all or filter by workspace.
    """
    async with get_session() as session:
        # Build base query
        query = select(Strategy).where(Strategy.is_active == True)
        count_query = select(func.count(Strategy.id)).where(Strategy.is_active == True)

        # Apply workspace filter
        if current_user.is_superuser:
            if workspace_id:
                query = query.where(Strategy.workspace_id == workspace_id)
                count_query = count_query.where(Strategy.workspace_id == workspace_id)
        else:
            if current_user.workspace_id is None:
                return StrategyListResponse(
                    items=[],
                    total=0,
                    page=pagination.page,
                    page_size=pagination.page_size,
                )
            query = query.where(Strategy.workspace_id == current_user.workspace_id)
            count_query = count_query.where(Strategy.workspace_id == current_user.workspace_id)

        # Get total count
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Get strategies
        query = (
            query.offset(pagination.offset)
            .limit(pagination.limit)
            .order_by(Strategy.created_at.desc())
        )
        result = await session.execute(query)
        strategies = result.scalars().all()

        # Build response with counts
        items = []
        for strat in strategies:
            version_count = await _get_version_count(session, strat.id)
            latest_version = await _get_latest_version(session, strat.id)
            items.append(_strategy_to_response(strat, version_count, latest_version))

        return StrategyListResponse(
            items=items,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
        )


@router.post(
    "",
    response_model=StrategyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create strategy",
    description="Create a new strategy.",
)
async def create_strategy(
    request: StrategyCreate,
    current_user: UserDep,
) -> StrategyResponse:
    """
    Create a new strategy.

    Requires strategy:create permission or superuser.
    """
    async with get_session() as session:
        # Check workspace access
        if not current_user.is_superuser:
            if current_user.workspace_id != request.workspace_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cannot create strategy in another workspace",
                )
            if not current_user.has_permission("strategy:create"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission required: strategy:create",
                )

        # Verify workspace exists
        ws_result = await session.execute(
            select(Workspace).where(
                Workspace.id == request.workspace_id,
                Workspace.is_active == True,
            )
        )
        if ws_result.scalar_one_or_none() is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace not found",
            )

        # Check name uniqueness within workspace
        existing = await session.execute(
            select(Strategy).where(
                Strategy.workspace_id == request.workspace_id,
                Strategy.name == request.name,
                Strategy.is_active == True,
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Strategy with name '{request.name}' already exists in this workspace",
            )

        strategy = Strategy(
            name=request.name,
            description=request.description,
            workspace_id=request.workspace_id,
            git_repo=request.git_repo,
            tags=request.tags,
            extra_metadata=request.metadata,
        )
        session.add(strategy)
        await session.commit()
        await session.refresh(strategy)

        return _strategy_to_response(strategy, 0, None)


@router.get(
    "/{strategy_id}",
    response_model=StrategyResponse,
    summary="Get strategy",
    description="Get strategy by ID.",
)
async def get_strategy(
    strategy_id: UUID,
    current_user: UserDep,
) -> StrategyResponse:
    """Get strategy by ID."""
    async with get_session() as session:
        result = await session.execute(select(Strategy).where(Strategy.id == strategy_id))
        strategy = result.scalar_one_or_none()

        if strategy is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy not found",
            )

        # Check access
        if not current_user.is_superuser:
            if current_user.workspace_id != strategy.workspace_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this strategy",
                )

        version_count = await _get_version_count(session, strategy.id)
        latest_version = await _get_latest_version(session, strategy.id)

        return _strategy_to_response(strategy, version_count, latest_version)


@router.patch(
    "/{strategy_id}",
    response_model=StrategyResponse,
    summary="Update strategy",
    description="Update strategy metadata.",
)
async def update_strategy(
    strategy_id: UUID,
    request: StrategyUpdate,
    current_user: UserDep,
) -> StrategyResponse:
    """Update strategy."""
    async with get_session() as session:
        result = await session.execute(select(Strategy).where(Strategy.id == strategy_id))
        strategy = result.scalar_one_or_none()

        if strategy is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy not found",
            )

        # Check access
        if not current_user.is_superuser:
            if current_user.workspace_id != strategy.workspace_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this strategy",
                )
            if not current_user.has_permission("strategy:write"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission required: strategy:write",
                )

        # Update fields
        if request.name is not None:
            # Check uniqueness
            existing = await session.execute(
                select(Strategy).where(
                    Strategy.workspace_id == strategy.workspace_id,
                    Strategy.name == request.name,
                    Strategy.id != strategy_id,
                    Strategy.is_active == True,
                )
            )
            if existing.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Strategy with name '{request.name}' already exists",
                )
            strategy.name = request.name

        if request.description is not None:
            strategy.description = request.description
        if request.git_repo is not None:
            strategy.git_repo = request.git_repo
        if request.tags is not None:
            strategy.tags = request.tags
        if request.metadata is not None:
            strategy.extra_metadata = request.metadata
        if request.is_active is not None:
            strategy.is_active = request.is_active

        strategy.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(strategy)

        version_count = await _get_version_count(session, strategy.id)
        latest_version = await _get_latest_version(session, strategy.id)

        return _strategy_to_response(strategy, version_count, latest_version)


@router.delete(
    "/{strategy_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete strategy",
    description="Soft delete strategy.",
)
async def delete_strategy(
    strategy_id: UUID,
    current_user: UserDep,
) -> None:
    """Soft delete strategy."""
    async with get_session() as session:
        result = await session.execute(select(Strategy).where(Strategy.id == strategy_id))
        strategy = result.scalar_one_or_none()

        if strategy is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy not found",
            )

        # Check access
        if not current_user.is_superuser:
            if current_user.workspace_id != strategy.workspace_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this strategy",
                )
            if not current_user.has_permission("strategy:delete"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission required: strategy:delete",
                )

        strategy.is_active = False
        strategy.updated_at = datetime.now(timezone.utc)
        await session.commit()


# =============================================================================
# Strategy Version Endpoints
# =============================================================================


@router.get(
    "/{strategy_id}/versions",
    response_model=StrategyVersionListResponse,
    summary="List strategy versions",
    description="List all versions of a strategy.",
)
async def list_strategy_versions(
    strategy_id: UUID,
    current_user: UserDep,
    pagination: PaginationDep,
    include_deprecated: bool = False,
) -> StrategyVersionListResponse:
    """List versions for a strategy."""
    async with get_session() as session:
        # Verify strategy exists and access
        strat_result = await session.execute(select(Strategy).where(Strategy.id == strategy_id))
        strategy = strat_result.scalar_one_or_none()

        if strategy is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy not found",
            )

        if not current_user.is_superuser:
            if current_user.workspace_id != strategy.workspace_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this strategy",
                )

        # Build query
        query = select(StrategyVersion).where(StrategyVersion.strategy_id == strategy_id)
        count_query = select(func.count(StrategyVersion.id)).where(
            StrategyVersion.strategy_id == strategy_id
        )

        if not include_deprecated:
            query = query.where(StrategyVersion.is_deprecated == False)
            count_query = count_query.where(StrategyVersion.is_deprecated == False)

        # Get total
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Get versions
        query = (
            query.offset(pagination.offset)
            .limit(pagination.limit)
            .order_by(StrategyVersion.created_at.desc())
        )
        result = await session.execute(query)
        versions = result.scalars().all()

        items = []
        for version in versions:
            build_count = await _get_build_count(session, version.id)
            items.append(_version_to_response(version, build_count))

        return StrategyVersionListResponse(
            items=items,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
        )


@router.post(
    "/{strategy_id}/versions",
    response_model=StrategyVersionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create strategy version",
    description="Create a new version for a strategy.",
)
async def create_strategy_version(
    strategy_id: UUID,
    request: StrategyVersionCreate,
    current_user: UserDep,
) -> StrategyVersionResponse:
    """Create a new strategy version."""
    async with get_session() as session:
        # Verify strategy exists
        strat_result = await session.execute(
            select(Strategy).where(
                Strategy.id == strategy_id,
                Strategy.is_active == True,
            )
        )
        strategy = strat_result.scalar_one_or_none()

        if strategy is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy not found",
            )

        # Check access
        if not current_user.is_superuser:
            if current_user.workspace_id != strategy.workspace_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this strategy",
                )
            if not current_user.has_permission("strategy:write"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission required: strategy:write",
                )

        # Check version uniqueness
        existing = await session.execute(
            select(StrategyVersion).where(
                StrategyVersion.strategy_id == strategy_id,
                StrategyVersion.version == request.version,
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Version '{request.version}' already exists for this strategy",
            )

        # Set previous latest to not latest
        await session.execute(
            select(StrategyVersion).where(
                StrategyVersion.strategy_id == strategy_id,
                StrategyVersion.is_latest == True,
            )
        )
        prev_latest = await session.execute(
            select(StrategyVersion).where(
                StrategyVersion.strategy_id == strategy_id,
                StrategyVersion.is_latest == True,
            )
        )
        prev_version = prev_latest.scalar_one_or_none()
        if prev_version:
            prev_version.is_latest = False

        # Create new version
        version = StrategyVersion(
            strategy_id=strategy_id,
            workspace_id=strategy.workspace_id,
            version=request.version,
            git_sha=request.git_sha,
            git_tag=request.git_tag,
            changelog=request.changelog,
            extra_metadata=request.metadata,
            is_latest=True,
            is_deprecated=False,
        )
        session.add(version)
        await session.commit()
        await session.refresh(version)

        return _version_to_response(version, 0)


@router.get(
    "/{strategy_id}/versions/{version_id}",
    response_model=StrategyVersionResponse,
    summary="Get strategy version",
    description="Get a specific strategy version.",
)
async def get_strategy_version(
    strategy_id: UUID,
    version_id: UUID,
    current_user: UserDep,
) -> StrategyVersionResponse:
    """Get strategy version by ID."""
    async with get_session() as session:
        result = await session.execute(
            select(StrategyVersion).where(
                StrategyVersion.id == version_id,
                StrategyVersion.strategy_id == strategy_id,
            )
        )
        version = result.scalar_one_or_none()

        if version is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy version not found",
            )

        # Check access
        if not current_user.is_superuser:
            if current_user.workspace_id != version.workspace_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this version",
                )

        build_count = await _get_build_count(session, version.id)
        return _version_to_response(version, build_count)


@router.patch(
    "/{strategy_id}/versions/{version_id}",
    response_model=StrategyVersionResponse,
    summary="Update strategy version",
    description="Update a strategy version (deprecate, update changelog).",
)
async def update_strategy_version(
    strategy_id: UUID,
    version_id: UUID,
    request: StrategyVersionUpdate,
    current_user: UserDep,
) -> StrategyVersionResponse:
    """Update strategy version."""
    async with get_session() as session:
        result = await session.execute(
            select(StrategyVersion).where(
                StrategyVersion.id == version_id,
                StrategyVersion.strategy_id == strategy_id,
            )
        )
        version = result.scalar_one_or_none()

        if version is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy version not found",
            )

        # Check access
        if not current_user.is_superuser:
            if current_user.workspace_id != version.workspace_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this version",
                )
            if not current_user.has_permission("strategy:write"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission required: strategy:write",
                )

        # Update fields
        if request.is_deprecated is not None:
            version.is_deprecated = request.is_deprecated
        if request.changelog is not None:
            version.changelog = request.changelog
        if request.metadata is not None:
            version.extra_metadata = request.metadata

        version.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(version)

        build_count = await _get_build_count(session, version.id)
        return _version_to_response(version, build_count)


# =============================================================================
# Build Endpoints
# =============================================================================


@router.get(
    "/{strategy_id}/versions/{version_id}/builds",
    response_model=BuildListResponse,
    summary="List builds",
    description="List all builds for a strategy version.",
)
async def list_builds(
    strategy_id: UUID,
    version_id: UUID,
    current_user: UserDep,
    pagination: PaginationDep,
    status_filter: Optional[str] = None,
) -> BuildListResponse:
    """List builds for a version."""
    async with get_session() as session:
        # Verify version exists
        ver_result = await session.execute(
            select(StrategyVersion).where(
                StrategyVersion.id == version_id,
                StrategyVersion.strategy_id == strategy_id,
            )
        )
        version = ver_result.scalar_one_or_none()

        if version is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy version not found",
            )

        if not current_user.is_superuser:
            if current_user.workspace_id != version.workspace_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied",
                )

        # Build query
        query = select(Build).where(Build.strategy_version_id == version_id)
        count_query = select(func.count(Build.id)).where(Build.strategy_version_id == version_id)

        if status_filter:
            query = query.where(Build.status == status_filter)
            count_query = count_query.where(Build.status == status_filter)

        # Get total
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Get builds
        query = (
            query.offset(pagination.offset)
            .limit(pagination.limit)
            .order_by(Build.build_number.desc())
        )
        result = await session.execute(query)
        builds = result.scalars().all()

        items = []
        for build in builds:
            artifact_count = await _get_artifact_count(session, build.id)
            items.append(_build_to_response(build, artifact_count))

        return BuildListResponse(
            items=items,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
        )


@router.post(
    "/{strategy_id}/versions/{version_id}/builds",
    response_model=BuildResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create build",
    description="Create a new build for a strategy version.",
)
async def create_build(
    strategy_id: UUID,
    version_id: UUID,
    request: BuildCreate,
    current_user: UserDep,
) -> BuildResponse:
    """Create a new build."""
    async with get_session() as session:
        # Verify version exists
        ver_result = await session.execute(
            select(StrategyVersion).where(
                StrategyVersion.id == version_id,
                StrategyVersion.strategy_id == strategy_id,
            )
        )
        version = ver_result.scalar_one_or_none()

        if version is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy version not found",
            )

        if version.is_deprecated:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot build deprecated version",
            )

        # Check access
        if not current_user.is_superuser:
            if current_user.workspace_id != version.workspace_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied",
                )
            if not current_user.has_permission("build:create"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission required: build:create",
                )

        # Get next build number
        build_number = await _get_next_build_number(session, version_id)

        build = Build(
            strategy_version_id=version_id,
            workspace_id=version.workspace_id,
            build_number=build_number,
            builder_id=request.builder_id,
            ci_job_id=request.ci_job_id,
            ci_pipeline_url=request.ci_pipeline_url,
            status="pending",
            started_at=datetime.now(timezone.utc),
            extra_metadata=request.metadata,
        )
        session.add(build)
        await session.commit()
        await session.refresh(build)

        return _build_to_response(build, 0)


@router.get(
    "/{strategy_id}/versions/{version_id}/builds/{build_id}",
    response_model=BuildResponse,
    summary="Get build",
    description="Get a specific build.",
)
async def get_build(
    strategy_id: UUID,
    version_id: UUID,
    build_id: UUID,
    current_user: UserDep,
) -> BuildResponse:
    """Get build by ID."""
    async with get_session() as session:
        result = await session.execute(
            select(Build).where(
                Build.id == build_id,
                Build.strategy_version_id == version_id,
            )
        )
        build = result.scalar_one_or_none()

        if build is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Build not found",
            )

        if not current_user.is_superuser:
            if current_user.workspace_id != build.workspace_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied",
                )

        artifact_count = await _get_artifact_count(session, build.id)
        return _build_to_response(build, artifact_count)


@router.patch(
    "/{strategy_id}/versions/{version_id}/builds/{build_id}",
    response_model=BuildResponse,
    summary="Update build",
    description="Update build status and metadata.",
)
async def update_build(
    strategy_id: UUID,
    version_id: UUID,
    build_id: UUID,
    request: BuildUpdate,
    current_user: UserDep,
) -> BuildResponse:
    """Update build."""
    async with get_session() as session:
        result = await session.execute(
            select(Build).where(
                Build.id == build_id,
                Build.strategy_version_id == version_id,
            )
        )
        build = result.scalar_one_or_none()

        if build is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Build not found",
            )

        if not current_user.is_superuser:
            if current_user.workspace_id != build.workspace_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied",
                )
            if not current_user.has_permission("build:write"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission required: build:write",
                )

        # Update fields
        if request.status is not None:
            build.status = request.status
        if request.logs_url is not None:
            build.logs_url = request.logs_url
        if request.finished_at is not None:
            build.finished_at = request.finished_at
        if request.metadata is not None:
            build.extra_metadata = request.metadata

        build.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(build)

        artifact_count = await _get_artifact_count(session, build.id)
        return _build_to_response(build, artifact_count)


# =============================================================================
# Artifact Endpoints
# =============================================================================


@router.get(
    "/{strategy_id}/versions/{version_id}/builds/{build_id}/artifacts",
    response_model=ArtifactListResponse,
    summary="List artifacts",
    description="List all artifacts for a build.",
)
async def list_artifacts(
    strategy_id: UUID,
    version_id: UUID,
    build_id: UUID,
    current_user: UserDep,
    pagination: PaginationDep,
) -> ArtifactListResponse:
    """List artifacts for a build."""
    async with get_session() as session:
        # Verify build exists
        build_result = await session.execute(
            select(Build).where(
                Build.id == build_id,
                Build.strategy_version_id == version_id,
            )
        )
        build = build_result.scalar_one_or_none()

        if build is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Build not found",
            )

        if not current_user.is_superuser:
            if current_user.workspace_id != build.workspace_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied",
                )

        # Build query
        query = select(Artifact).where(Artifact.build_id == build_id)
        count_query = select(func.count(Artifact.id)).where(Artifact.build_id == build_id)

        # Get total
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Get artifacts
        query = (
            query.offset(pagination.offset)
            .limit(pagination.limit)
            .order_by(Artifact.created_at.desc())
        )
        result = await session.execute(query)
        artifacts = result.scalars().all()

        items = [_artifact_to_response(a) for a in artifacts]

        return ArtifactListResponse(
            items=items,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
        )


@router.post(
    "/{strategy_id}/versions/{version_id}/builds/{build_id}/artifacts",
    response_model=ArtifactResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create artifact",
    description="Register a new artifact for a build.",
)
async def create_artifact(
    strategy_id: UUID,
    version_id: UUID,
    build_id: UUID,
    request: ArtifactCreate,
    current_user: UserDep,
) -> ArtifactResponse:
    """Create a new artifact."""
    async with get_session() as session:
        # Verify build exists
        build_result = await session.execute(
            select(Build).where(
                Build.id == build_id,
                Build.strategy_version_id == version_id,
            )
        )
        build = build_result.scalar_one_or_none()

        if build is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Build not found",
            )

        # Check access
        if not current_user.is_superuser:
            if current_user.workspace_id != build.workspace_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied",
                )
            if not current_user.has_permission("artifact:create"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission required: artifact:create",
                )

        # Check digest uniqueness within build
        existing = await session.execute(
            select(Artifact).where(
                Artifact.build_id == build_id,
                Artifact.digest == request.digest,
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Artifact with this digest already exists for this build",
            )

        # Map change_class string to enum
        change_class_enum = ChangeClass[request.change_class]

        artifact = Artifact(
            build_id=build_id,
            workspace_id=build.workspace_id,
            name=request.name,
            format=request.format,
            digest=request.digest,
            size_bytes=request.size_bytes,
            registry_url=request.registry_url,
            signature_ref=request.signature_ref,
            sbom_ref=request.sbom_ref,
            provenance=request.provenance,
            change_class=change_class_enum,
            is_signed=request.signature_ref is not None,
            extra_metadata=request.metadata,
        )
        session.add(artifact)
        await session.commit()
        await session.refresh(artifact)

        return _artifact_to_response(artifact)


@router.get(
    "/{strategy_id}/versions/{version_id}/builds/{build_id}/artifacts/{artifact_id}",
    response_model=ArtifactResponse,
    summary="Get artifact",
    description="Get a specific artifact.",
)
async def get_artifact(
    strategy_id: UUID,
    version_id: UUID,
    build_id: UUID,
    artifact_id: UUID,
    current_user: UserDep,
) -> ArtifactResponse:
    """Get artifact by ID."""
    async with get_session() as session:
        result = await session.execute(
            select(Artifact).where(
                Artifact.id == artifact_id,
                Artifact.build_id == build_id,
            )
        )
        artifact = result.scalar_one_or_none()

        if artifact is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Artifact not found",
            )

        if not current_user.is_superuser:
            if current_user.workspace_id != artifact.workspace_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied",
                )

        return _artifact_to_response(artifact)
