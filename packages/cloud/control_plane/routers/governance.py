# -*- coding: utf-8 -*-
"""
Governance Router.

CCEA Phase 8 Implementation.

Provides REST API endpoints for governance features:
    - DSAR (Data Subject Access Requests)
    - Data Retention policies
    - Data Residency management
    - Break-glass access
    - Health monitoring
    - Alert management

Design Doc Reference:
    - Phase 8: "Telemetry + Privacy/GDPR + Residency + Access Controls"

CLOUD ZONE ONLY.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ..dependencies import UserDep
from ...governance import (
    DSARService,
    DSARRequest,
    DSARRequestType,
    DSARStatus,
    DataResidencyManager,
    DataRegion,
    ResidencyMode,
    RetentionService,
    RetentionAction,
    BreakGlassController,
    BreakGlassReason,
    BreakGlassScope,
    HealthMonitorService,
    RunStatus,
    AlertRulesEngine,
    AlertSeverity,
)


router = APIRouter()

# Initialize services (in production, use dependency injection)
_dsar_service = DSARService()
_residency_manager = DataResidencyManager()
_retention_service = RetentionService()
_break_glass_controller = BreakGlassController()
_health_monitor = HealthMonitorService()
_alert_engine = AlertRulesEngine()


# ============================================================================
# DSAR Endpoints
# ============================================================================

class DSARCreateRequest(BaseModel):
    """Create DSAR request."""
    workspace_id: str
    request_type: str = Field(..., description="ACCESS, PORTABILITY, or ERASURE")
    data_categories: Optional[List[str]] = None
    reason: str = ""


class DSARResponse(BaseModel):
    """DSAR response."""
    id: str
    request_type: str
    user_id: str
    workspace_id: str
    status: str
    created_at: datetime
    deadline: datetime
    extended_deadline: Optional[datetime]
    completed_at: Optional[datetime]


@router.post(
    "/dsar/requests",
    response_model=DSARResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create DSAR request",
    description="Create a new Data Subject Access Request.",
)
async def create_dsar_request(
    request: DSARCreateRequest,
    current_user: UserDep,
) -> DSARResponse:
    """Create DSAR request."""
    try:
        request_type = DSARRequestType[request.request_type.upper()]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request type. Must be one of: ACCESS, PORTABILITY, ERASURE",
        )

    dsar = _dsar_service.create_request(
        user_id=str(current_user.id),
        workspace_id=request.workspace_id,
        request_type=request_type,
        data_categories=set(request.data_categories) if request.data_categories else None,
        reason=request.reason,
    )

    return DSARResponse(
        id=dsar.id,
        request_type=dsar.request_type.name,
        user_id=dsar.user_id,
        workspace_id=dsar.workspace_id,
        status=dsar.status.value,
        created_at=dsar.created_at,
        deadline=dsar.deadline,
        extended_deadline=dsar.extended_deadline,
        completed_at=dsar.completed_at,
    )


@router.get(
    "/dsar/requests/{request_id}",
    response_model=DSARResponse,
    summary="Get DSAR request",
)
async def get_dsar_request(
    request_id: str,
    current_user: UserDep,
) -> DSARResponse:
    """Get DSAR request by ID."""
    dsar = _dsar_service.get_request(request_id)
    if not dsar:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="DSAR request not found",
        )

    # Check access
    if dsar.user_id != str(current_user.id) and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    return DSARResponse(
        id=dsar.id,
        request_type=dsar.request_type.name,
        user_id=dsar.user_id,
        workspace_id=dsar.workspace_id,
        status=dsar.status.value,
        created_at=dsar.created_at,
        deadline=dsar.deadline,
        extended_deadline=dsar.extended_deadline,
        completed_at=dsar.completed_at,
    )


@router.get(
    "/dsar/requests",
    response_model=List[DSARResponse],
    summary="List DSAR requests",
)
async def list_dsar_requests(
    current_user: UserDep,
) -> List[DSARResponse]:
    """List DSAR requests for current user."""
    requests = _dsar_service.get_user_requests(str(current_user.id))
    return [
        DSARResponse(
            id=r.id,
            request_type=r.request_type.name,
            user_id=r.user_id,
            workspace_id=r.workspace_id,
            status=r.status.value,
            created_at=r.created_at,
            deadline=r.deadline,
            extended_deadline=r.extended_deadline,
            completed_at=r.completed_at,
        )
        for r in requests
    ]


# ============================================================================
# Data Residency Endpoints
# ============================================================================

class ResidencyPolicyRequest(BaseModel):
    """Create residency policy request."""
    workspace_id: str
    country_code: str
    mode: Optional[str] = None


class ResidencyPolicyResponse(BaseModel):
    """Residency policy response."""
    id: str
    workspace_id: str
    primary_region: str
    allowed_regions: List[str]
    mode: str
    gdpr_compliant: bool
    requires_eu_only: bool
    telemetry_local: bool


@router.post(
    "/residency/policies",
    response_model=ResidencyPolicyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create residency policy",
)
async def create_residency_policy(
    request: ResidencyPolicyRequest,
    current_user: UserDep,
) -> ResidencyPolicyResponse:
    """Create data residency policy for workspace."""
    policy = _residency_manager.create_policy_for_country(
        workspace_id=request.workspace_id,
        country_code=request.country_code,
    )

    return ResidencyPolicyResponse(
        id=policy.id,
        workspace_id=policy.workspace_id,
        primary_region=policy.primary_region.value,
        allowed_regions=[r.value for r in policy.allowed_regions],
        mode=policy.mode.name,
        gdpr_compliant=policy.gdpr_compliant,
        requires_eu_only=policy.requires_eu_only,
        telemetry_local=policy.telemetry_local,
    )


@router.get(
    "/residency/policies/{workspace_id}",
    response_model=ResidencyPolicyResponse,
    summary="Get residency policy",
)
async def get_residency_policy(
    workspace_id: str,
    current_user: UserDep,
) -> ResidencyPolicyResponse:
    """Get residency policy for workspace."""
    policy = _residency_manager.get_policy(workspace_id)
    if not policy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Residency policy not found",
        )

    return ResidencyPolicyResponse(
        id=policy.id,
        workspace_id=policy.workspace_id,
        primary_region=policy.primary_region.value,
        allowed_regions=[r.value for r in policy.allowed_regions],
        mode=policy.mode.name,
        gdpr_compliant=policy.gdpr_compliant,
        requires_eu_only=policy.requires_eu_only,
        telemetry_local=policy.telemetry_local,
    )


# ============================================================================
# Retention Endpoints
# ============================================================================

class RetentionPolicyRequest(BaseModel):
    """Create retention policy request."""
    workspace_id: str
    data_type: str
    retention_days: int = 90
    action: str = "DELETE"


class RetentionPolicyResponse(BaseModel):
    """Retention policy response."""
    id: str
    workspace_id: str
    data_type: str
    retention_days: int
    action: str
    enabled: bool
    legal_hold: bool


@router.post(
    "/retention/policies",
    response_model=RetentionPolicyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create retention policy",
)
async def create_retention_policy(
    request: RetentionPolicyRequest,
    current_user: UserDep,
) -> RetentionPolicyResponse:
    """Create data retention policy."""
    try:
        action = RetentionAction[request.action.upper()]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid action. Must be one of: DELETE, ARCHIVE, ANONYMIZE, AGGREGATE",
        )

    policy = _retention_service.create_policy(
        workspace_id=request.workspace_id,
        data_type=request.data_type,
        retention_days=request.retention_days,
        action=action,
    )

    return RetentionPolicyResponse(
        id=policy.id,
        workspace_id=policy.workspace_id,
        data_type=policy.data_type,
        retention_days=policy.retention_days,
        action=policy.action.name,
        enabled=policy.enabled,
        legal_hold=policy.legal_hold,
    )


@router.get(
    "/retention/policies/{workspace_id}",
    response_model=List[RetentionPolicyResponse],
    summary="List retention policies",
)
async def list_retention_policies(
    workspace_id: str,
    current_user: UserDep,
) -> List[RetentionPolicyResponse]:
    """List retention policies for workspace."""
    policies = _retention_service.get_workspace_policies(workspace_id)
    return [
        RetentionPolicyResponse(
            id=p.id,
            workspace_id=p.workspace_id,
            data_type=p.data_type,
            retention_days=p.retention_days,
            action=p.action.name,
            enabled=p.enabled,
            legal_hold=p.legal_hold,
        )
        for p in policies
    ]


@router.post(
    "/retention/legal-hold",
    status_code=status.HTTP_200_OK,
    summary="Set legal hold",
)
async def set_legal_hold(
    workspace_id: str,
    data_type: str,
    reason: str,
    current_user: UserDep,
) -> Dict[str, Any]:
    """Set legal hold on data type."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only superusers can set legal holds",
        )

    success = _retention_service.set_legal_hold(
        workspace_id=workspace_id,
        data_type=data_type,
        reason=reason,
    )

    return {"success": success}


# ============================================================================
# Break Glass Endpoints
# ============================================================================

class BreakGlassCreateRequest(BaseModel):
    """Create break-glass request."""
    workspace_id: str
    reason: str = Field(..., min_length=20)
    reason_type: str = "OTHER"
    scope: List[str] = ["TELEMETRY_READ"]
    duration_hours: int = 4


class BreakGlassResponse(BaseModel):
    """Break-glass response."""
    id: str
    requester_id: str
    workspace_id: str
    reason_type: str
    status: str
    approved: bool
    expires_at: Optional[datetime]
    access_token: Optional[str] = None


@router.post(
    "/break-glass/requests",
    response_model=BreakGlassResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create break-glass request",
)
async def create_break_glass_request(
    request: BreakGlassCreateRequest,
    current_user: UserDep,
) -> BreakGlassResponse:
    """Create break-glass access request."""
    try:
        reason_type = BreakGlassReason[request.reason_type.upper()]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid reason type",
        )

    try:
        scope = {BreakGlassScope[s.upper()] for s in request.scope}
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid scope",
        )

    try:
        bg_request = _break_glass_controller.create_request(
            requester_id=str(current_user.id),
            requester_email=current_user.email,
            reason=request.reason,
            reason_type=reason_type,
            workspace_id=request.workspace_id,
            scope=scope,
            duration_hours=request.duration_hours,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return BreakGlassResponse(
        id=bg_request.id,
        requester_id=bg_request.requester_id,
        workspace_id=bg_request.workspace_id,
        reason_type=bg_request.reason_type.value,
        status="pending_approval",
        approved=bg_request.approved,
        expires_at=bg_request.expires_at,
    )


@router.post(
    "/break-glass/requests/{request_id}/approve",
    response_model=BreakGlassResponse,
    summary="Approve break-glass request",
)
async def approve_break_glass_request(
    request_id: str,
    current_user: UserDep,
) -> BreakGlassResponse:
    """Approve a break-glass request."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only superusers can approve break-glass requests",
        )

    result = _break_glass_controller.approve_request(
        request_id=request_id,
        approved_by=str(current_user.id),
    )

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error,
        )

    bg_request = _break_glass_controller.get_request(request_id)
    return BreakGlassResponse(
        id=bg_request.id,
        requester_id=bg_request.requester_id,
        workspace_id=bg_request.workspace_id,
        reason_type=bg_request.reason_type.value,
        status="approved",
        approved=bg_request.approved,
        expires_at=bg_request.expires_at,
        access_token=result.access_token,
    )


# ============================================================================
# Health Monitor Endpoints
# ============================================================================

class AgentHealthResponse(BaseModel):
    """Agent health response."""
    agent_id: str
    workspace_id: str
    status: str
    health: str
    run_status: str
    last_heartbeat: Optional[datetime]
    halt_reason: Optional[str]
    cpu_usage: float
    memory_usage: float


class HealthDashboardResponse(BaseModel):
    """Health dashboard response."""
    workspace_id: str
    total_agents: int
    online_agents: int
    offline_agents: int
    healthy_count: int
    warning_count: int
    critical_count: int
    running_count: int
    halted_count: int
    halted_agents: List[Dict[str, Any]]


@router.get(
    "/health/dashboard/{workspace_id}",
    response_model=HealthDashboardResponse,
    summary="Get health dashboard",
)
async def get_health_dashboard(
    workspace_id: str,
    current_user: UserDep,
) -> HealthDashboardResponse:
    """Get health dashboard for workspace."""
    dashboard = _health_monitor.get_dashboard(workspace_id)

    return HealthDashboardResponse(
        workspace_id=dashboard.workspace_id,
        total_agents=dashboard.total_agents,
        online_agents=dashboard.online_agents,
        offline_agents=dashboard.offline_agents,
        healthy_count=dashboard.healthy_count,
        warning_count=dashboard.warning_count,
        critical_count=dashboard.critical_count,
        running_count=dashboard.running_count,
        halted_count=dashboard.halted_count,
        halted_agents=dashboard.halted_agents,
    )


@router.get(
    "/health/agents/{agent_id}",
    response_model=AgentHealthResponse,
    summary="Get agent health",
)
async def get_agent_health(
    agent_id: str,
    current_user: UserDep,
) -> AgentHealthResponse:
    """Get health for specific agent."""
    health = _health_monitor.get_agent_health(agent_id)
    if not health:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )

    return AgentHealthResponse(
        agent_id=health.agent_id,
        workspace_id=health.workspace_id,
        status=health.status.value,
        health=health.health.value,
        run_status=health.run_status.value,
        last_heartbeat=health.last_heartbeat,
        halt_reason=health.halt_reason,
        cpu_usage=health.cpu_usage,
        memory_usage=health.memory_usage,
    )


@router.get(
    "/health/issues/{workspace_id}",
    response_model=List[Dict[str, Any]],
    summary="Get health issues",
)
async def get_health_issues(
    workspace_id: str,
    current_user: UserDep,
) -> List[Dict[str, Any]]:
    """Get current health issues for workspace."""
    return _health_monitor.get_health_issues(workspace_id)


# ============================================================================
# Alert Endpoints
# ============================================================================

class AlertTriggerResponse(BaseModel):
    """Alert trigger response."""
    id: str
    rule_name: str
    alert_type: str
    severity: str
    message: str
    workspace_id: str
    agent_id: Optional[str]
    timestamp: datetime
    acknowledged: bool
    resolved: bool


class AlertStatsResponse(BaseModel):
    """Alert statistics response."""
    total: int
    unresolved: int
    unacknowledged: int
    by_severity: Dict[str, int]


@router.get(
    "/alerts/triggers",
    response_model=List[AlertTriggerResponse],
    summary="List alert triggers",
)
async def list_alert_triggers(
    current_user: UserDep,
    workspace_id: Optional[str] = None,
    severity: Optional[str] = None,
    unresolved_only: bool = False,
) -> List[AlertTriggerResponse]:
    """List alert triggers."""
    sev = None
    if severity:
        try:
            sev = AlertSeverity[severity.upper()]
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid severity",
            )

    triggers = _alert_engine.get_triggers(
        workspace_id=workspace_id,
        severity=sev,
        unresolved_only=unresolved_only,
    )

    return [
        AlertTriggerResponse(
            id=t.id,
            rule_name=t.rule_name,
            alert_type=t.alert_type,
            severity=t.severity.value,
            message=t.message,
            workspace_id=t.workspace_id,
            agent_id=t.agent_id,
            timestamp=t.timestamp,
            acknowledged=t.acknowledged,
            resolved=t.resolved,
        )
        for t in triggers
    ]


@router.get(
    "/alerts/stats",
    response_model=AlertStatsResponse,
    summary="Get alert statistics",
)
async def get_alert_stats(
    current_user: UserDep,
    workspace_id: Optional[str] = None,
) -> AlertStatsResponse:
    """Get alert statistics."""
    stats = _alert_engine.get_trigger_stats(workspace_id)
    return AlertStatsResponse(**stats)


@router.post(
    "/alerts/triggers/{trigger_id}/acknowledge",
    status_code=status.HTTP_200_OK,
    summary="Acknowledge alert",
)
async def acknowledge_alert(
    trigger_id: str,
    current_user: UserDep,
) -> Dict[str, Any]:
    """Acknowledge an alert trigger."""
    success = _alert_engine.acknowledge_trigger(
        trigger_id=trigger_id,
        acknowledged_by=str(current_user.id),
    )
    return {"success": success}


@router.post(
    "/alerts/triggers/{trigger_id}/resolve",
    status_code=status.HTTP_200_OK,
    summary="Resolve alert",
)
async def resolve_alert(
    trigger_id: str,
    current_user: UserDep,
) -> Dict[str, Any]:
    """Resolve an alert trigger."""
    success = _alert_engine.resolve_trigger(trigger_id)
    return {"success": success}
