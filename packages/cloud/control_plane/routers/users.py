# -*- coding: utf-8 -*-
"""
Users Router.

CLOUD ZONE ONLY.

Provides CRUD endpoints for user management.

Phase 5 Security (WI-AUTH-01):
- Argon2id password hashing
- Password policy validation
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from ..database import get_session
from ..dependencies import PaginationDep, UserDep
from ..models import Organization, Role, User, Workspace
from ..security.password_hasher import hash_password as secure_hash_password
from ..security.password_policy import validate_password, DEFAULT_PASSWORD_POLICY

router = APIRouter()


def hash_password(password: str) -> str:
    """
    Hash password using Argon2id (WI-AUTH-01).

    Argon2id is memory-hard and resistant to GPU attacks.
    """
    return secure_hash_password(password)


# Request/Response models
class UserCreate(BaseModel):
    """Create user request."""

    email: EmailStr
    password: str = Field(..., min_length=8)
    display_name: Optional[str] = Field(None, max_length=255)
    organization_id: UUID
    default_workspace_id: Optional[UUID] = None
    role_ids: List[UUID] = Field(default_factory=list)

    @field_validator("password")
    @classmethod
    def validate_password_policy(cls, v: str, info) -> str:
        """WI-AUTH-01: Validate password against policy."""
        # Get email from values if available for context check
        email = info.data.get("email") if hasattr(info, "data") else None
        result = validate_password(v, email=email)
        if not result.valid:
            messages = [viol.message for viol in result.violations]
            raise ValueError(f"Password policy violation: {'; '.join(messages)}")
        return v


class UserUpdate(BaseModel):
    """Update user request."""

    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=8)
    display_name: Optional[str] = Field(None, max_length=255)
    default_workspace_id: Optional[UUID] = None
    is_active: Optional[bool] = None
    role_ids: Optional[List[UUID]] = None

    @field_validator("password")
    @classmethod
    def validate_password_policy(cls, v: Optional[str], info) -> Optional[str]:
        """WI-AUTH-01: Validate password against policy if provided."""
        if v is None:
            return v
        email = info.data.get("email") if hasattr(info, "data") else None
        result = validate_password(v, email=email)
        if not result.valid:
            messages = [viol.message for viol in result.violations]
            raise ValueError(f"Password policy violation: {'; '.join(messages)}")
        return v


class RoleResponse(BaseModel):
    """Role response."""

    id: UUID
    name: str
    description: Optional[str]


class UserResponse(BaseModel):
    """User response."""

    id: UUID
    email: str
    display_name: Optional[str]
    organization_id: UUID
    default_workspace_id: Optional[UUID]
    is_active: bool
    last_login: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    roles: List[RoleResponse] = Field(default_factory=list)


class UserListResponse(BaseModel):
    """Paginated user list response."""

    items: List[UserResponse]
    total: int
    page: int
    page_size: int


def _user_to_response(user: User) -> UserResponse:
    """Convert User model to response."""
    return UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        organization_id=user.organization_id,
        default_workspace_id=user.default_workspace_id,
        is_active=user.is_active,
        last_login=user.last_login,
        created_at=user.created_at,
        updated_at=user.updated_at,
        roles=[
            RoleResponse(
                id=role.id,
                name=role.name,
                description=role.description,
            )
            for role in user.roles
        ],
    )


@router.get(
    "",
    response_model=UserListResponse,
    summary="List users",
    description="List users in the organization.",
)
async def list_users(
    current_user: UserDep,
    pagination: PaginationDep,
    organization_id: Optional[UUID] = None,
) -> UserListResponse:
    """
    List users.

    Users see users in their organization.
    Superusers can see all or filter by organization.
    """
    async with get_session() as session:
        # Build base query
        query = select(User).options(selectinload(User.roles))
        count_query = select(func.count(User.id))

        # Apply organization filter
        if current_user.is_superuser:
            if organization_id:
                query = query.where(User.organization_id == organization_id)
                count_query = count_query.where(User.organization_id == organization_id)
        else:
            if current_user.org_id is None:
                return UserListResponse(
                    items=[],
                    total=0,
                    page=pagination.page,
                    page_size=pagination.page_size,
                )
            query = query.where(User.organization_id == current_user.org_id)
            count_query = count_query.where(User.organization_id == current_user.org_id)

        # Get total count
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Get users
        query = (
            query.offset(pagination.offset).limit(pagination.limit).order_by(User.created_at.desc())
        )
        result = await session.execute(query)
        users = result.scalars().unique().all()

        items = [_user_to_response(user) for user in users]

        return UserListResponse(
            items=items,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
        )


@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create user",
    description="Create a new user.",
)
async def create_user(
    request: UserCreate,
    current_user: UserDep,
) -> UserResponse:
    """
    Create a new user.

    Requires user:create permission or superuser.
    """
    async with get_session() as session:
        # Check organization access
        if not current_user.is_superuser:
            if current_user.org_id != request.organization_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cannot create user in another organization",
                )
            if not current_user.has_permission("user:create"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission required: user:create",
                )

        # Verify organization exists
        org_result = await session.execute(
            select(Organization).where(
                Organization.id == request.organization_id,
                Organization.is_active == True,
            )
        )
        if org_result.scalar_one_or_none() is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found",
            )

        # Check email uniqueness
        existing = await session.execute(select(User).where(User.email == request.email))
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"User with email '{request.email}' already exists",
            )

        # Verify workspace if provided
        if request.default_workspace_id:
            ws_result = await session.execute(
                select(Workspace).where(
                    Workspace.id == request.default_workspace_id,
                    Workspace.organization_id == request.organization_id,
                    Workspace.is_active == True,
                )
            )
            if ws_result.scalar_one_or_none() is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Workspace not found or not in the same organization",
                )

        # Get roles
        roles = []
        if request.role_ids:
            roles_result = await session.execute(
                select(Role).where(
                    Role.id.in_(request.role_ids),
                    Role.organization_id == request.organization_id,
                )
            )
            roles = list(roles_result.scalars().all())
            if len(roles) != len(request.role_ids):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="One or more roles not found",
                )

        user = User(
            email=request.email,
            password_hash=hash_password(request.password),
            display_name=request.display_name,
            organization_id=request.organization_id,
            default_workspace_id=request.default_workspace_id,
        )
        user.roles = roles
        session.add(user)
        await session.commit()
        await session.refresh(user)

        # Reload with roles
        result = await session.execute(
            select(User).options(selectinload(User.roles)).where(User.id == user.id)
        )
        user = result.scalar_one()

        return _user_to_response(user)


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get user",
    description="Get user by ID.",
)
async def get_user(
    user_id: UUID,
    current_user: UserDep,
) -> UserResponse:
    """
    Get user by ID.

    Users can view themselves or users in their organization with permission.
    """
    async with get_session() as session:
        result = await session.execute(
            select(User).options(selectinload(User.roles)).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        # Check access
        if not current_user.is_superuser:
            if current_user.id != user_id:
                if current_user.org_id != user.organization_id:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied",
                    )
                if not current_user.has_permission("user:read"):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Permission required: user:read",
                    )

        return _user_to_response(user)


@router.patch(
    "/{user_id}",
    response_model=UserResponse,
    summary="Update user",
    description="Update user profile.",
)
async def update_user(
    user_id: UUID,
    request: UserUpdate,
    current_user: UserDep,
) -> UserResponse:
    """
    Update user.

    Users can update themselves or others with user:write permission.
    """
    async with get_session() as session:
        result = await session.execute(
            select(User).options(selectinload(User.roles)).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        # Check access
        is_self = current_user.id == user_id
        if not current_user.is_superuser:
            if not is_self:
                if current_user.org_id != user.organization_id:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied",
                    )
                if not current_user.has_permission("user:write"):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Permission required: user:write",
                    )

        # Update fields
        if request.email is not None:
            # Check email uniqueness
            existing = await session.execute(
                select(User).where(
                    User.email == request.email,
                    User.id != user_id,
                )
            )
            if existing.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"User with email '{request.email}' already exists",
                )
            user.email = request.email

        if request.password is not None:
            user.password_hash = hash_password(request.password)

        if request.display_name is not None:
            user.display_name = request.display_name

        if request.default_workspace_id is not None:
            # Verify workspace
            ws_result = await session.execute(
                select(Workspace).where(
                    Workspace.id == request.default_workspace_id,
                    Workspace.organization_id == user.organization_id,
                    Workspace.is_active == True,
                )
            )
            if ws_result.scalar_one_or_none() is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Workspace not found or not in the same organization",
                )
            user.default_workspace_id = request.default_workspace_id

        if request.is_active is not None:
            # Only superusers or users with permission can deactivate
            if not current_user.is_superuser and not current_user.has_permission("user:write"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission required to change user status",
                )
            user.is_active = request.is_active

        if request.role_ids is not None:
            # Only superusers or users with permission can change roles
            if not current_user.is_superuser and not current_user.has_permission(
                "user:assign_role"
            ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission required: user:assign_role",
                )

            if request.role_ids:
                roles_result = await session.execute(
                    select(Role).where(
                        Role.id.in_(request.role_ids),
                        Role.organization_id == user.organization_id,
                    )
                )
                roles = list(roles_result.scalars().all())
                if len(roles) != len(request.role_ids):
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="One or more roles not found",
                    )
                user.roles = roles
            else:
                user.roles = []

        user.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(user)

        # Reload with roles
        result = await session.execute(
            select(User).options(selectinload(User.roles)).where(User.id == user.id)
        )
        user = result.scalar_one()

        return _user_to_response(user)


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    summary="Delete user",
    description="Soft delete user.",
)
async def delete_user(
    user_id: UUID,
    current_user: UserDep,
) -> None:
    """
    Soft delete user.

    Requires user:delete permission or superuser.
    Cannot delete yourself.
    """
    if current_user.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete yourself",
        )

    async with get_session() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        # Check access
        if not current_user.is_superuser:
            if current_user.org_id != user.organization_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied",
                )
            if not current_user.has_permission("user:delete"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission required: user:delete",
                )

        user.is_active = False
        user.updated_at = datetime.now(timezone.utc)
        await session.commit()
