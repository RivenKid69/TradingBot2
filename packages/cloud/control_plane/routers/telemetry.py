# -*- coding: utf-8 -*-
"""
Telemetry Router.

CLOUD ZONE ONLY.

Provides endpoints for telemetry events and alerts management.

Key Concepts:
    - TelemetryEvent: Events from agents with redacted payloads
    - Alert: System-generated alerts with acknowledgment workflow
    - TelemetryLevel: Controls verbosity (aggregated/detailed/raw)
    - Workspace-scoped with proper tenant isolation
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, select

from ..database import get_session
from ..dependencies import PaginationDep, UserDep
from ..models import (
    Agent,
    Alert,
    AlertSeverity,
    TelemetryEvent,
    TelemetryLevel,
    Workspace,
)

router = APIRouter()


# ============================================================================
# Constants
# ============================================================================

# Valid telemetry levels
VALID_TELEMETRY_LEVELS = frozenset(
    [
        TelemetryLevel.AGGREGATED.value,
        TelemetryLevel.DETAILED_NON_SENSITIVE.value,
        TelemetryLevel.RAW_ORDER_EVENTS.value,
    ]
)

# Valid alert severities
VALID_ALERT_SEVERITIES = frozenset(
    [
        AlertSeverity.INFO.value,
        AlertSeverity.WARNING.value,
        AlertSeverity.ERROR.value,
        AlertSeverity.CRITICAL.value,
    ]
)

# Default redaction version
DEFAULT_REDACTION_VERSION = "1.0.0"


# ============================================================================
# Helper Functions
# ============================================================================


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


async def _verify_agent_access(
    session,
    agent_id: UUID,
    current_user,
) -> Agent:
    """
    Verify access to an agent and return the agent if accessible.
    """
    result = await session.execute(select(Agent).where(Agent.id == agent_id))
    agent = result.scalar_one_or_none()

    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )

    # Verify workspace access
    await _verify_workspace_access(session, agent.workspace_id, current_user)

    return agent


# ============================================================================
# Telemetry Event Request/Response Models
# ============================================================================


class TelemetryEventCreate(BaseModel):
    """Create telemetry event request."""

    agent_id: UUID
    run_id: Optional[UUID] = None
    event_type: str = Field(..., min_length=1, max_length=100)
    event_timestamp: datetime
    telemetry_level: str = Field(default=TelemetryLevel.AGGREGATED.value, max_length=50)
    payload: Dict[str, Any] = Field(..., description="Redacted telemetry payload")
    redaction_applied: bool = Field(default=True, description="Confirm redaction was applied")
    redaction_version: Optional[str] = Field(default=DEFAULT_REDACTION_VERSION, max_length=20)


class TelemetryEventResponse(BaseModel):
    """Telemetry event response."""

    id: UUID
    workspace_id: UUID
    agent_id: UUID
    run_id: Optional[UUID]
    event_type: str
    event_timestamp: datetime
    telemetry_level: str
    payload: Dict[str, Any]
    redaction_applied: bool
    redaction_version: Optional[str]
    created_at: datetime


class TelemetryEventSummaryResponse(BaseModel):
    """Telemetry event summary response (without payload)."""

    id: UUID
    workspace_id: UUID
    agent_id: UUID
    run_id: Optional[UUID]
    event_type: str
    event_timestamp: datetime
    telemetry_level: str
    redaction_applied: bool
    created_at: datetime


class TelemetryEventListResponse(BaseModel):
    """Paginated telemetry event list response."""

    items: List[TelemetryEventSummaryResponse]
    total: int
    page: int
    page_size: int


class TelemetryEventBatchCreate(BaseModel):
    """Batch create telemetry events request."""

    events: List[TelemetryEventCreate] = Field(..., min_length=1, max_length=100)


class TelemetryEventBatchResponse(BaseModel):
    """Batch create telemetry events response."""

    created_count: int
    event_ids: List[UUID]


# ============================================================================
# Alert Request/Response Models
# ============================================================================


class AlertCreate(BaseModel):
    """Create alert request."""

    workspace_id: UUID
    agent_id: Optional[UUID] = None
    run_id: Optional[UUID] = None
    alert_type: str = Field(..., min_length=1, max_length=100)
    severity: str = Field(default=AlertSeverity.INFO.value, max_length=20)
    title: str = Field(..., min_length=1, max_length=255)
    message: str = Field(..., min_length=1)
    context: Optional[Dict[str, Any]] = None


class AlertResponse(BaseModel):
    """Alert response."""

    id: UUID
    workspace_id: UUID
    agent_id: Optional[UUID]
    run_id: Optional[UUID]
    alert_type: str
    severity: str
    title: str
    message: str
    acknowledged: bool
    acknowledged_by: Optional[str]
    acknowledged_at: Optional[datetime]
    resolved: bool
    resolved_at: Optional[datetime]
    context: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class AlertSummaryResponse(BaseModel):
    """Alert summary response (without context)."""

    id: UUID
    workspace_id: UUID
    agent_id: Optional[UUID]
    alert_type: str
    severity: str
    title: str
    acknowledged: bool
    resolved: bool
    created_at: datetime


class AlertListResponse(BaseModel):
    """Paginated alert list response."""

    items: List[AlertSummaryResponse]
    total: int
    page: int
    page_size: int


class AlertAcknowledgeRequest(BaseModel):
    """Acknowledge alert request."""

    acknowledged_by: str = Field(..., min_length=1, max_length=255)


class AlertResolveRequest(BaseModel):
    """Resolve alert request."""

    resolved_by: Optional[str] = Field(None, max_length=255)


class AlertStatsResponse(BaseModel):
    """Alert statistics response."""

    total: int
    by_severity: Dict[str, int]
    unacknowledged: int
    unresolved: int


# ============================================================================
# Telemetry Event Endpoints
# ============================================================================


@router.get(
    "/events",
    response_model=TelemetryEventListResponse,
    summary="List telemetry events",
    description="List telemetry events with optional filters.",
)
async def list_telemetry_events(
    current_user: UserDep,
    pagination: PaginationDep,
    workspace_id: Optional[UUID] = None,
    agent_id: Optional[UUID] = None,
    run_id: Optional[UUID] = None,
    event_type: Optional[str] = None,
    telemetry_level: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> TelemetryEventListResponse:
    """
    List telemetry events.

    Users see events from their organization's workspaces.
    Superusers can see all or filter by workspace.

    Requires telemetry:read permission.
    """
    async with get_session() as session:
        # Check permission
        if not current_user.is_superuser and not current_user.has_permission("telemetry:read"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission required: telemetry:read",
            )

        # Build base query
        query = select(TelemetryEvent)
        count_query = select(func.count(TelemetryEvent.id))

        # Apply workspace filter
        if current_user.is_superuser:
            if workspace_id:
                query = query.where(TelemetryEvent.workspace_id == workspace_id)
                count_query = count_query.where(TelemetryEvent.workspace_id == workspace_id)
        else:
            # Non-superusers must specify workspace or see their org's workspaces
            if workspace_id:
                await _verify_workspace_access(session, workspace_id, current_user)
                query = query.where(TelemetryEvent.workspace_id == workspace_id)
                count_query = count_query.where(TelemetryEvent.workspace_id == workspace_id)
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
                    return TelemetryEventListResponse(
                        items=[],
                        total=0,
                        page=pagination.page,
                        page_size=pagination.page_size,
                    )
                query = query.where(TelemetryEvent.workspace_id.in_(workspace_ids))
                count_query = count_query.where(TelemetryEvent.workspace_id.in_(workspace_ids))

        # Apply additional filters
        if agent_id:
            query = query.where(TelemetryEvent.agent_id == agent_id)
            count_query = count_query.where(TelemetryEvent.agent_id == agent_id)

        if run_id:
            query = query.where(TelemetryEvent.run_id == run_id)
            count_query = count_query.where(TelemetryEvent.run_id == run_id)

        if event_type:
            query = query.where(TelemetryEvent.event_type == event_type)
            count_query = count_query.where(TelemetryEvent.event_type == event_type)

        if telemetry_level:
            query = query.where(TelemetryEvent.telemetry_level == telemetry_level)
            count_query = count_query.where(TelemetryEvent.telemetry_level == telemetry_level)

        if start_time:
            query = query.where(TelemetryEvent.event_timestamp >= start_time)
            count_query = count_query.where(TelemetryEvent.event_timestamp >= start_time)

        if end_time:
            query = query.where(TelemetryEvent.event_timestamp <= end_time)
            count_query = count_query.where(TelemetryEvent.event_timestamp <= end_time)

        # Get total count
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Get events with pagination
        query = (
            query.offset(pagination.offset)
            .limit(pagination.limit)
            .order_by(TelemetryEvent.event_timestamp.desc())
        )
        result = await session.execute(query)
        events = result.scalars().all()

        items = [
            TelemetryEventSummaryResponse(
                id=event.id,
                workspace_id=event.workspace_id,
                agent_id=event.agent_id,
                run_id=event.run_id,
                event_type=event.event_type,
                event_timestamp=event.event_timestamp,
                telemetry_level=event.telemetry_level,
                redaction_applied=event.redaction_applied,
                created_at=event.created_at,
            )
            for event in events
        ]

        return TelemetryEventListResponse(
            items=items,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
        )


@router.post(
    "/events",
    response_model=TelemetryEventResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create telemetry event",
    description="Create a new telemetry event.",
)
async def create_telemetry_event(
    request: TelemetryEventCreate,
    current_user: UserDep,
) -> TelemetryEventResponse:
    """
    Create a new telemetry event.

    Requires telemetry:write permission or agent context.
    Redaction must be confirmed for events with sensitive data.
    """
    async with get_session() as session:
        # Verify agent access (which also verifies workspace access)
        agent = await _verify_agent_access(session, request.agent_id, current_user)

        # Check permission
        if not current_user.is_superuser and not current_user.has_permission("telemetry:write"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission required: telemetry:write",
            )

        # Validate telemetry level
        if request.telemetry_level not in VALID_TELEMETRY_LEVELS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid telemetry_level. Must be one of: {', '.join(sorted(VALID_TELEMETRY_LEVELS))}",
            )

        # RAW_ORDER_EVENTS requires explicit opt-in and enterprise tier
        if request.telemetry_level == TelemetryLevel.RAW_ORDER_EVENTS.value:
            if not current_user.is_superuser:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="RAW_ORDER_EVENTS telemetry level requires enterprise tier",
                )

        # Enforce redaction for non-aggregated events
        if (
            request.telemetry_level != TelemetryLevel.AGGREGATED.value
            and not request.redaction_applied
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="redaction_applied must be True for non-aggregated telemetry levels",
            )

        # Create event
        event = TelemetryEvent(
            workspace_id=agent.workspace_id,
            agent_id=request.agent_id,
            run_id=request.run_id,
            event_type=request.event_type,
            event_timestamp=request.event_timestamp,
            telemetry_level=request.telemetry_level,
            payload=request.payload,
            redaction_applied=request.redaction_applied,
            redaction_version=request.redaction_version,
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)

        return TelemetryEventResponse(
            id=event.id,
            workspace_id=event.workspace_id,
            agent_id=event.agent_id,
            run_id=event.run_id,
            event_type=event.event_type,
            event_timestamp=event.event_timestamp,
            telemetry_level=event.telemetry_level,
            payload=event.payload,
            redaction_applied=event.redaction_applied,
            redaction_version=event.redaction_version,
            created_at=event.created_at,
        )


@router.post(
    "/events/batch",
    response_model=TelemetryEventBatchResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create telemetry events in batch",
    description="Create multiple telemetry events in a single request.",
)
async def create_telemetry_events_batch(
    request: TelemetryEventBatchCreate,
    current_user: UserDep,
) -> TelemetryEventBatchResponse:
    """
    Create telemetry events in batch.

    More efficient than creating events one by one.
    All events must belong to the same workspace.

    Requires telemetry:write permission.
    """
    async with get_session() as session:
        # Check permission
        if not current_user.is_superuser and not current_user.has_permission("telemetry:write"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission required: telemetry:write",
            )

        # Validate all events and collect agent info
        agent_ids = set(e.agent_id for e in request.events)
        agents_result = await session.execute(select(Agent).where(Agent.id.in_(agent_ids)))
        agents = {a.id: a for a in agents_result.scalars().all()}

        # Verify all agents exist and belong to same workspace
        workspace_ids = set()
        for event_req in request.events:
            if event_req.agent_id not in agents:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent {event_req.agent_id} not found",
                )
            agent = agents[event_req.agent_id]
            workspace_ids.add(agent.workspace_id)

        # All events must be in the same workspace for batch efficiency
        if len(workspace_ids) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="All events in batch must belong to the same workspace",
            )

        # Verify workspace access
        workspace_id = workspace_ids.pop()
        await _verify_workspace_access(session, workspace_id, current_user)

        # Validate telemetry levels
        for event_req in request.events:
            if event_req.telemetry_level not in VALID_TELEMETRY_LEVELS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid telemetry_level: {event_req.telemetry_level}",
                )
            if event_req.telemetry_level == TelemetryLevel.RAW_ORDER_EVENTS.value:
                if not current_user.is_superuser:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="RAW_ORDER_EVENTS telemetry level requires enterprise tier",
                    )

        # Create all events
        events = []
        for event_req in request.events:
            event = TelemetryEvent(
                workspace_id=workspace_id,
                agent_id=event_req.agent_id,
                run_id=event_req.run_id,
                event_type=event_req.event_type,
                event_timestamp=event_req.event_timestamp,
                telemetry_level=event_req.telemetry_level,
                payload=event_req.payload,
                redaction_applied=event_req.redaction_applied,
                redaction_version=event_req.redaction_version,
            )
            session.add(event)
            events.append(event)

        await session.commit()

        # Refresh events to get their assigned IDs
        event_ids = []
        for event in events:
            await session.refresh(event)
            event_ids.append(event.id)

        return TelemetryEventBatchResponse(
            created_count=len(event_ids),
            event_ids=event_ids,
        )


@router.get(
    "/events/{event_id}",
    response_model=TelemetryEventResponse,
    summary="Get telemetry event",
    description="Get telemetry event by ID.",
)
async def get_telemetry_event(
    event_id: UUID,
    current_user: UserDep,
) -> TelemetryEventResponse:
    """
    Get telemetry event by ID.

    Requires telemetry:read permission.
    """
    async with get_session() as session:
        # Check permission
        if not current_user.is_superuser and not current_user.has_permission("telemetry:read"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission required: telemetry:read",
            )

        result = await session.execute(select(TelemetryEvent).where(TelemetryEvent.id == event_id))
        event = result.scalar_one_or_none()

        if event is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Telemetry event not found",
            )

        # Verify workspace access
        await _verify_workspace_access(session, event.workspace_id, current_user)

        return TelemetryEventResponse(
            id=event.id,
            workspace_id=event.workspace_id,
            agent_id=event.agent_id,
            run_id=event.run_id,
            event_type=event.event_type,
            event_timestamp=event.event_timestamp,
            telemetry_level=event.telemetry_level,
            payload=event.payload,
            redaction_applied=event.redaction_applied,
            redaction_version=event.redaction_version,
            created_at=event.created_at,
        )


@router.get(
    "/events/types/list",
    response_model=List[str],
    summary="List telemetry levels",
    description="Get list of valid telemetry levels.",
)
async def list_telemetry_levels(
    current_user: UserDep,
) -> List[str]:
    """
    List valid telemetry levels.

    Returns list of allowed telemetry_level values.
    """
    return sorted(VALID_TELEMETRY_LEVELS)


# ============================================================================
# Alert Endpoints
# ============================================================================


@router.get(
    "/alerts",
    response_model=AlertListResponse,
    summary="List alerts",
    description="List alerts with optional filters.",
)
async def list_alerts(
    current_user: UserDep,
    pagination: PaginationDep,
    workspace_id: Optional[UUID] = None,
    agent_id: Optional[UUID] = None,
    alert_type: Optional[str] = None,
    severity: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    resolved: Optional[bool] = None,
) -> AlertListResponse:
    """
    List alerts.

    Users see alerts from their organization's workspaces.
    Superusers can see all or filter by workspace.

    Requires alert:read permission.
    """
    async with get_session() as session:
        # Check permission
        if not current_user.is_superuser and not current_user.has_permission("alert:read"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission required: alert:read",
            )

        # Build base query
        query = select(Alert)
        count_query = select(func.count(Alert.id))

        # Apply workspace filter
        if current_user.is_superuser:
            if workspace_id:
                query = query.where(Alert.workspace_id == workspace_id)
                count_query = count_query.where(Alert.workspace_id == workspace_id)
        else:
            if workspace_id:
                await _verify_workspace_access(session, workspace_id, current_user)
                query = query.where(Alert.workspace_id == workspace_id)
                count_query = count_query.where(Alert.workspace_id == workspace_id)
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
                    return AlertListResponse(
                        items=[],
                        total=0,
                        page=pagination.page,
                        page_size=pagination.page_size,
                    )
                query = query.where(Alert.workspace_id.in_(workspace_ids))
                count_query = count_query.where(Alert.workspace_id.in_(workspace_ids))

        # Apply additional filters
        if agent_id:
            query = query.where(Alert.agent_id == agent_id)
            count_query = count_query.where(Alert.agent_id == agent_id)

        if alert_type:
            query = query.where(Alert.alert_type == alert_type)
            count_query = count_query.where(Alert.alert_type == alert_type)

        if severity:
            query = query.where(Alert.severity == severity)
            count_query = count_query.where(Alert.severity == severity)

        if acknowledged is not None:
            query = query.where(Alert.acknowledged == acknowledged)
            count_query = count_query.where(Alert.acknowledged == acknowledged)

        if resolved is not None:
            query = query.where(Alert.resolved == resolved)
            count_query = count_query.where(Alert.resolved == resolved)

        # Get total count
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Get alerts with pagination
        query = (
            query.offset(pagination.offset)
            .limit(pagination.limit)
            .order_by(Alert.created_at.desc())
        )
        result = await session.execute(query)
        alerts = result.scalars().all()

        items = [
            AlertSummaryResponse(
                id=alert.id,
                workspace_id=alert.workspace_id,
                agent_id=alert.agent_id,
                alert_type=alert.alert_type,
                severity=alert.severity,
                title=alert.title,
                acknowledged=alert.acknowledged,
                resolved=alert.resolved,
                created_at=alert.created_at,
            )
            for alert in alerts
        ]

        return AlertListResponse(
            items=items,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
        )


@router.post(
    "/alerts",
    response_model=AlertResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create alert",
    description="Create a new alert.",
)
async def create_alert(
    request: AlertCreate,
    current_user: UserDep,
) -> AlertResponse:
    """
    Create a new alert.

    Requires alert:create permission or superuser.
    """
    async with get_session() as session:
        # Verify workspace access
        await _verify_workspace_access(session, request.workspace_id, current_user)

        # Check permission
        if not current_user.is_superuser and not current_user.has_permission("alert:create"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission required: alert:create",
            )

        # Validate severity
        if request.severity not in VALID_ALERT_SEVERITIES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid severity. Must be one of: {', '.join(sorted(VALID_ALERT_SEVERITIES))}",
            )

        # Verify agent exists if provided
        if request.agent_id:
            agent_result = await session.execute(
                select(Agent).where(
                    Agent.id == request.agent_id,
                    Agent.workspace_id == request.workspace_id,
                )
            )
            if agent_result.scalar_one_or_none() is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Agent not found in this workspace",
                )

        # Create alert
        alert = Alert(
            workspace_id=request.workspace_id,
            agent_id=request.agent_id,
            run_id=request.run_id,
            alert_type=request.alert_type,
            severity=request.severity,
            title=request.title,
            message=request.message,
            context=request.context,
        )
        session.add(alert)
        await session.commit()
        await session.refresh(alert)

        return AlertResponse(
            id=alert.id,
            workspace_id=alert.workspace_id,
            agent_id=alert.agent_id,
            run_id=alert.run_id,
            alert_type=alert.alert_type,
            severity=alert.severity,
            title=alert.title,
            message=alert.message,
            acknowledged=alert.acknowledged,
            acknowledged_by=alert.acknowledged_by,
            acknowledged_at=alert.acknowledged_at,
            resolved=alert.resolved,
            resolved_at=alert.resolved_at,
            context=alert.context,
            created_at=alert.created_at,
            updated_at=alert.updated_at,
        )


# NOTE: Static routes must be defined BEFORE dynamic path parameter routes
# to avoid FastAPI matching /alerts/stats as /alerts/{alert_id}


@router.get(
    "/alerts/stats",
    response_model=AlertStatsResponse,
    summary="Get alert statistics",
    description="Get aggregated alert statistics.",
)
async def get_alert_stats(
    current_user: UserDep,
    workspace_id: Optional[UUID] = None,
) -> AlertStatsResponse:
    """
    Get alert statistics.

    Returns total count, count by severity, unacknowledged and unresolved counts.

    Requires alert:read permission.
    """
    async with get_session() as session:
        # Check permission
        if not current_user.is_superuser and not current_user.has_permission("alert:read"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission required: alert:read",
            )

        # Determine workspace scope
        if current_user.is_superuser:
            workspace_filter = Alert.workspace_id == workspace_id if workspace_id else True
        else:
            if workspace_id:
                await _verify_workspace_access(session, workspace_id, current_user)
                workspace_filter = Alert.workspace_id == workspace_id
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
                    return AlertStatsResponse(
                        total=0,
                        by_severity={},
                        unacknowledged=0,
                        unresolved=0,
                    )
                workspace_filter = Alert.workspace_id.in_(workspace_ids)

        # Total count
        total_result = await session.execute(select(func.count(Alert.id)).where(workspace_filter))
        total = total_result.scalar() or 0

        # Count by severity
        severity_result = await session.execute(
            select(Alert.severity, func.count(Alert.id))
            .where(workspace_filter)
            .group_by(Alert.severity)
        )
        by_severity = {row[0]: row[1] for row in severity_result.fetchall()}

        # Unacknowledged count
        unack_result = await session.execute(
            select(func.count(Alert.id)).where(and_(workspace_filter, Alert.acknowledged == False))
        )
        unacknowledged = unack_result.scalar() or 0

        # Unresolved count
        unres_result = await session.execute(
            select(func.count(Alert.id)).where(and_(workspace_filter, Alert.resolved == False))
        )
        unresolved = unres_result.scalar() or 0

        return AlertStatsResponse(
            total=total,
            by_severity=by_severity,
            unacknowledged=unacknowledged,
            unresolved=unresolved,
        )


@router.get(
    "/alerts/severities/list",
    response_model=List[str],
    summary="List alert severities",
    description="Get list of valid alert severities.",
)
async def list_alert_severities(
    current_user: UserDep,
) -> List[str]:
    """
    List valid alert severities.

    Returns list of allowed severity values.
    """
    return sorted(VALID_ALERT_SEVERITIES)


@router.get(
    "/alerts/{alert_id}",
    response_model=AlertResponse,
    summary="Get alert",
    description="Get alert by ID.",
)
async def get_alert(
    alert_id: UUID,
    current_user: UserDep,
) -> AlertResponse:
    """
    Get alert by ID.

    Requires alert:read permission.
    """
    async with get_session() as session:
        # Check permission
        if not current_user.is_superuser and not current_user.has_permission("alert:read"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission required: alert:read",
            )

        result = await session.execute(select(Alert).where(Alert.id == alert_id))
        alert = result.scalar_one_or_none()

        if alert is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found",
            )

        # Verify workspace access
        await _verify_workspace_access(session, alert.workspace_id, current_user)

        return AlertResponse(
            id=alert.id,
            workspace_id=alert.workspace_id,
            agent_id=alert.agent_id,
            run_id=alert.run_id,
            alert_type=alert.alert_type,
            severity=alert.severity,
            title=alert.title,
            message=alert.message,
            acknowledged=alert.acknowledged,
            acknowledged_by=alert.acknowledged_by,
            acknowledged_at=alert.acknowledged_at,
            resolved=alert.resolved,
            resolved_at=alert.resolved_at,
            context=alert.context,
            created_at=alert.created_at,
            updated_at=alert.updated_at,
        )


@router.post(
    "/alerts/{alert_id}/acknowledge",
    response_model=AlertResponse,
    summary="Acknowledge alert",
    description="Acknowledge an alert.",
)
async def acknowledge_alert(
    alert_id: UUID,
    request: AlertAcknowledgeRequest,
    current_user: UserDep,
) -> AlertResponse:
    """
    Acknowledge an alert.

    Requires alert:acknowledge permission.
    """
    async with get_session() as session:
        # Check permission
        if not current_user.is_superuser and not current_user.has_permission("alert:acknowledge"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission required: alert:acknowledge",
            )

        result = await session.execute(select(Alert).where(Alert.id == alert_id))
        alert = result.scalar_one_or_none()

        if alert is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found",
            )

        # Verify workspace access
        await _verify_workspace_access(session, alert.workspace_id, current_user)

        if alert.acknowledged:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Alert is already acknowledged",
            )

        # Acknowledge
        alert.acknowledged = True
        alert.acknowledged_by = request.acknowledged_by
        alert.acknowledged_at = datetime.now(timezone.utc)
        alert.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(alert)

        return AlertResponse(
            id=alert.id,
            workspace_id=alert.workspace_id,
            agent_id=alert.agent_id,
            run_id=alert.run_id,
            alert_type=alert.alert_type,
            severity=alert.severity,
            title=alert.title,
            message=alert.message,
            acknowledged=alert.acknowledged,
            acknowledged_by=alert.acknowledged_by,
            acknowledged_at=alert.acknowledged_at,
            resolved=alert.resolved,
            resolved_at=alert.resolved_at,
            context=alert.context,
            created_at=alert.created_at,
            updated_at=alert.updated_at,
        )


@router.post(
    "/alerts/{alert_id}/resolve",
    response_model=AlertResponse,
    summary="Resolve alert",
    description="Mark an alert as resolved.",
)
async def resolve_alert(
    alert_id: UUID,
    request: AlertResolveRequest,
    current_user: UserDep,
) -> AlertResponse:
    """
    Resolve an alert.

    Requires alert:resolve permission.
    """
    async with get_session() as session:
        # Check permission
        if not current_user.is_superuser and not current_user.has_permission("alert:resolve"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission required: alert:resolve",
            )

        result = await session.execute(select(Alert).where(Alert.id == alert_id))
        alert = result.scalar_one_or_none()

        if alert is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found",
            )

        # Verify workspace access
        await _verify_workspace_access(session, alert.workspace_id, current_user)

        if alert.resolved:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Alert is already resolved",
            )

        # Resolve
        alert.resolved = True
        alert.resolved_at = datetime.now(timezone.utc)
        alert.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(alert)

        return AlertResponse(
            id=alert.id,
            workspace_id=alert.workspace_id,
            agent_id=alert.agent_id,
            run_id=alert.run_id,
            alert_type=alert.alert_type,
            severity=alert.severity,
            title=alert.title,
            message=alert.message,
            acknowledged=alert.acknowledged,
            acknowledged_by=alert.acknowledged_by,
            acknowledged_at=alert.acknowledged_at,
            resolved=alert.resolved,
            resolved_at=alert.resolved_at,
            context=alert.context,
            created_at=alert.created_at,
            updated_at=alert.updated_at,
        )
