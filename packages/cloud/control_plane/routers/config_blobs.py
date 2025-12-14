# -*- coding: utf-8 -*-
"""
Config Blobs Router.

CLOUD ZONE ONLY.

Provides endpoints for immutable configuration blob management.

Key Concepts:
    - Content-addressable storage via SHA256 digest
    - Immutable: once created, cannot be modified (create new blob for changes)
    - Cloud stores only desired state (not secrets)
    - JSONB for queryability of content
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from ..database import get_session
from ..dependencies import PaginationDep, UserDep
from ..models import ConfigBlob, Workspace

router = APIRouter()


# ============================================================================
# Constants
# ============================================================================

# Valid config types
VALID_CONFIG_TYPES = frozenset([
    "strategy",
    "risk",
    "execution",
    "environment",
    "feature_flags",
    "model",
    "data",
    "alert",
    "monitoring",
    "custom",
])

# Maximum content size (10 MB)
MAX_CONTENT_SIZE_BYTES = 10 * 1024 * 1024


# ============================================================================
# Helper Functions
# ============================================================================

def _compute_digest(content: Dict[str, Any]) -> str:
    """Compute SHA256 digest of content."""
    # Serialize with sorted keys for deterministic hashing
    content_str = json.dumps(content, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(content_str.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _compute_size_bytes(content: Dict[str, Any]) -> int:
    """Compute size of content in bytes."""
    content_str = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return len(content_str.encode("utf-8"))


async def _verify_workspace_access(
    session,
    workspace_id: UUID,
    current_user,
) -> bool:
    """
    Verify that the user has access to the workspace.

    Returns True if access is granted.
    Raises HTTPException if access is denied.
    """
    # Superusers have access to all workspaces
    if current_user.is_superuser:
        return True

    # Check if workspace exists and belongs to user's organization
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

    # Verify organization match
    if current_user.org_id != workspace.organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this workspace",
        )

    return True


def _config_blob_to_response(blob: ConfigBlob) -> "ConfigBlobResponse":
    """Convert ConfigBlob model to response."""
    return ConfigBlobResponse(
        id=blob.id,
        workspace_id=blob.workspace_id,
        digest=blob.digest,
        content=blob.content,
        size_bytes=blob.size_bytes,
        config_type=blob.config_type,
        schema_version=blob.schema_version,
        created_by_user_id=blob.created_by_user_id,
        created_at=blob.created_at,
    )


# ============================================================================
# Request/Response Models
# ============================================================================

class ConfigBlobCreate(BaseModel):
    """Create config blob request."""

    workspace_id: UUID
    content: Dict[str, Any] = Field(..., description="Configuration content (JSONB)")
    config_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Config type: strategy, risk, execution, etc.",
    )
    schema_version: str = Field(
        default="1.0.0",
        max_length=20,
        description="Schema version",
    )


class ConfigBlobResponse(BaseModel):
    """Config blob response."""

    id: UUID
    workspace_id: UUID
    digest: str
    content: Dict[str, Any]
    size_bytes: int
    config_type: str
    schema_version: str
    created_by_user_id: Optional[UUID]
    created_at: datetime


class ConfigBlobSummaryResponse(BaseModel):
    """Config blob summary response (without content)."""

    id: UUID
    workspace_id: UUID
    digest: str
    size_bytes: int
    config_type: str
    schema_version: str
    created_by_user_id: Optional[UUID]
    created_at: datetime


class ConfigBlobListResponse(BaseModel):
    """Paginated config blob list response."""

    items: List[ConfigBlobSummaryResponse]
    total: int
    page: int
    page_size: int


class DigestLookupRequest(BaseModel):
    """Digest lookup request."""

    workspace_id: UUID
    digest: str = Field(..., min_length=1, max_length=128)


# ============================================================================
# Endpoints
# ============================================================================

@router.get(
    "",
    response_model=ConfigBlobListResponse,
    summary="List config blobs",
    description="List config blobs with optional filters.",
)
async def list_config_blobs(
    current_user: UserDep,
    pagination: PaginationDep,
    workspace_id: Optional[UUID] = None,
    config_type: Optional[str] = None,
    schema_version: Optional[str] = None,
) -> ConfigBlobListResponse:
    """
    List config blobs.

    Users see blobs in their organization's workspaces.
    Superusers can see all or filter by workspace.
    """
    async with get_session() as session:
        # Build base query
        query = select(ConfigBlob)
        count_query = select(func.count(ConfigBlob.id))

        # Apply workspace filter
        if current_user.is_superuser:
            if workspace_id:
                query = query.where(ConfigBlob.workspace_id == workspace_id)
                count_query = count_query.where(ConfigBlob.workspace_id == workspace_id)
        else:
            # Non-superusers must specify workspace or see nothing
            if workspace_id:
                await _verify_workspace_access(session, workspace_id, current_user)
                query = query.where(ConfigBlob.workspace_id == workspace_id)
                count_query = count_query.where(ConfigBlob.workspace_id == workspace_id)
            else:
                # Get all workspaces in user's organization
                ws_result = await session.execute(
                    select(Workspace.id).where(
                        Workspace.organization_id == current_user.org_id,
                        Workspace.is_active == True,
                    )
                )
                workspace_ids = [row[0] for row in ws_result.fetchall()]
                if not workspace_ids:
                    return ConfigBlobListResponse(
                        items=[],
                        total=0,
                        page=pagination.page,
                        page_size=pagination.page_size,
                    )
                query = query.where(ConfigBlob.workspace_id.in_(workspace_ids))
                count_query = count_query.where(ConfigBlob.workspace_id.in_(workspace_ids))

        # Apply config_type filter
        if config_type:
            query = query.where(ConfigBlob.config_type == config_type)
            count_query = count_query.where(ConfigBlob.config_type == config_type)

        # Apply schema_version filter
        if schema_version:
            query = query.where(ConfigBlob.schema_version == schema_version)
            count_query = count_query.where(ConfigBlob.schema_version == schema_version)

        # Get total count
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Get blobs with pagination
        query = (
            query.offset(pagination.offset)
            .limit(pagination.limit)
            .order_by(ConfigBlob.created_at.desc())
        )
        result = await session.execute(query)
        blobs = result.scalars().all()

        items = [
            ConfigBlobSummaryResponse(
                id=blob.id,
                workspace_id=blob.workspace_id,
                digest=blob.digest,
                size_bytes=blob.size_bytes,
                config_type=blob.config_type,
                schema_version=blob.schema_version,
                created_by_user_id=blob.created_by_user_id,
                created_at=blob.created_at,
            )
            for blob in blobs
        ]

        return ConfigBlobListResponse(
            items=items,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
        )


@router.post(
    "",
    response_model=ConfigBlobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create config blob",
    description="Create a new immutable config blob.",
)
async def create_config_blob(
    request: ConfigBlobCreate,
    current_user: UserDep,
) -> ConfigBlobResponse:
    """
    Create a new config blob.

    Config blobs are immutable and content-addressable.
    If a blob with the same digest already exists, returns the existing blob.

    Requires config:create permission or superuser.
    """
    async with get_session() as session:
        # Verify workspace access
        await _verify_workspace_access(session, request.workspace_id, current_user)

        # Check permission
        if not current_user.is_superuser and not current_user.has_permission("config:create"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission required: config:create",
            )

        # Validate config_type
        if request.config_type not in VALID_CONFIG_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid config_type. Must be one of: {', '.join(sorted(VALID_CONFIG_TYPES))}",
            )

        # Compute digest and size
        digest = _compute_digest(request.content)
        size_bytes = _compute_size_bytes(request.content)

        # Check size limit
        if size_bytes > MAX_CONTENT_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Content too large. Maximum size is {MAX_CONTENT_SIZE_BYTES} bytes",
            )

        # Check if blob with same digest already exists in workspace
        existing_result = await session.execute(
            select(ConfigBlob).where(
                ConfigBlob.workspace_id == request.workspace_id,
                ConfigBlob.digest == digest,
            )
        )
        existing_blob = existing_result.scalar_one_or_none()

        if existing_blob:
            # Return existing blob (immutable, content-addressable)
            return _config_blob_to_response(existing_blob)

        # Create new blob
        blob = ConfigBlob(
            workspace_id=request.workspace_id,
            digest=digest,
            content=request.content,
            size_bytes=size_bytes,
            config_type=request.config_type,
            schema_version=request.schema_version,
            created_by_user_id=current_user.id,
        )
        session.add(blob)
        await session.commit()
        await session.refresh(blob)

        return _config_blob_to_response(blob)


@router.get(
    "/{config_blob_id}",
    response_model=ConfigBlobResponse,
    summary="Get config blob",
    description="Get config blob by ID.",
)
async def get_config_blob(
    config_blob_id: UUID,
    current_user: UserDep,
) -> ConfigBlobResponse:
    """
    Get config blob by ID.

    Requires config:read permission or superuser.
    """
    async with get_session() as session:
        result = await session.execute(
            select(ConfigBlob).where(ConfigBlob.id == config_blob_id)
        )
        blob = result.scalar_one_or_none()

        if blob is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Config blob not found",
            )

        # Verify workspace access
        await _verify_workspace_access(session, blob.workspace_id, current_user)

        # Check permission
        if not current_user.is_superuser and not current_user.has_permission("config:read"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission required: config:read",
            )

        return _config_blob_to_response(blob)


@router.post(
    "/lookup-by-digest",
    response_model=ConfigBlobResponse,
    summary="Lookup config blob by digest",
    description="Find config blob by its content digest.",
)
async def lookup_config_blob_by_digest(
    request: DigestLookupRequest,
    current_user: UserDep,
) -> ConfigBlobResponse:
    """
    Lookup config blob by digest.

    This is the primary way to retrieve a config blob by its content address.

    Requires config:read permission or superuser.
    """
    async with get_session() as session:
        # Verify workspace access
        await _verify_workspace_access(session, request.workspace_id, current_user)

        # Check permission
        if not current_user.is_superuser and not current_user.has_permission("config:read"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission required: config:read",
            )

        # Lookup by digest
        result = await session.execute(
            select(ConfigBlob).where(
                ConfigBlob.workspace_id == request.workspace_id,
                ConfigBlob.digest == request.digest,
            )
        )
        blob = result.scalar_one_or_none()

        if blob is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Config blob not found with given digest",
            )

        return _config_blob_to_response(blob)


@router.get(
    "/by-digest/{digest}",
    response_model=ConfigBlobResponse,
    summary="Get config blob by digest",
    description="Get config blob by digest (workspace determined from user context).",
)
async def get_config_blob_by_digest(
    digest: str,
    current_user: UserDep,
    workspace_id: Optional[UUID] = None,
) -> ConfigBlobResponse:
    """
    Get config blob by digest.

    If workspace_id is not provided, searches all workspaces accessible to the user.
    For non-superusers, searches only their organization's workspaces.

    Requires config:read permission or superuser.
    """
    async with get_session() as session:
        # Check permission
        if not current_user.is_superuser and not current_user.has_permission("config:read"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission required: config:read",
            )

        if workspace_id:
            # Verify workspace access
            await _verify_workspace_access(session, workspace_id, current_user)

            result = await session.execute(
                select(ConfigBlob).where(
                    ConfigBlob.workspace_id == workspace_id,
                    ConfigBlob.digest == digest,
                )
            )
        elif current_user.is_superuser:
            # Superuser can search all workspaces
            result = await session.execute(
                select(ConfigBlob).where(ConfigBlob.digest == digest).limit(1)
            )
        else:
            # Get all workspaces in user's organization
            ws_result = await session.execute(
                select(Workspace.id).where(
                    Workspace.organization_id == current_user.org_id,
                    Workspace.is_active == True,
                )
            )
            workspace_ids = [row[0] for row in ws_result.fetchall()]
            if not workspace_ids:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Config blob not found with given digest",
                )

            result = await session.execute(
                select(ConfigBlob).where(
                    ConfigBlob.workspace_id.in_(workspace_ids),
                    ConfigBlob.digest == digest,
                ).limit(1)
            )

        blob = result.scalar_one_or_none()

        if blob is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Config blob not found with given digest",
            )

        return _config_blob_to_response(blob)


@router.post(
    "/verify-digest",
    response_model=Dict[str, Any],
    summary="Verify content digest",
    description="Compute digest for given content without storing.",
)
async def verify_digest(
    content: Dict[str, Any],
    current_user: UserDep,
) -> Dict[str, Any]:
    """
    Verify/compute digest for given content.

    This endpoint computes the digest for the given content
    without storing it. Useful for verifying content integrity
    or checking if a blob already exists.

    No permissions required beyond being authenticated.
    """
    digest = _compute_digest(content)
    size_bytes = _compute_size_bytes(content)

    return {
        "digest": digest,
        "size_bytes": size_bytes,
        "exceeds_max_size": size_bytes > MAX_CONTENT_SIZE_BYTES,
    }


@router.get(
    "/types/list",
    response_model=List[str],
    summary="List valid config types",
    description="Get list of valid config types.",
)
async def list_config_types(
    current_user: UserDep,
) -> List[str]:
    """
    List valid config types.

    Returns list of allowed config_type values.
    """
    return sorted(VALID_CONFIG_TYPES)
