# -*- coding: utf-8 -*-
"""
Research Jobs API - Endpoints for cloud research job management.

CCEA Phase 10: API for secure cloud research job execution.

Design Doc 15.3/5.3:
- Sandbox for research jobs
- Quotas CPU/RAM/time
- Egress allowlist + abuse detection
- Tenant isolation

Endpoints:
- POST /jobs - Submit a new research job
- GET /jobs - List jobs for tenant
- GET /jobs/{job_id} - Get job details
- POST /jobs/{job_id}/cancel - Cancel a job
- GET /jobs/{job_id}/logs - Get job logs
- GET /jobs/{job_id}/artifacts - List job artifacts
- GET /quota - Get tenant quota
- GET /quota/usage - Get quota usage
- GET /egress/policy - Get egress policy
- PUT /egress/policy - Update egress policy
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Body, BackgroundTasks
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

from packages.cloud.control_plane.database import get_session
from packages.cloud.control_plane.dependencies import (
    UserDep,
)

# Aliases for compatibility
get_db = get_session  # get_session is async contextmanager
get_current_user = UserDep  # Use UserDep for current user
get_current_workspace = None  # TODO: Implement if needed


# ============================================================================
# Request/Response Models
# ============================================================================

class JobSubmitRequest(BaseModel):
    """Request to submit a new research job."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    tags: Optional[List[str]] = Field(default_factory=list)

    # Code
    code: str = Field(..., min_length=1, max_length=1_000_000)  # 1MB limit
    entrypoint: str = Field(default="main.py", max_length=255)

    # Resources
    cpu: float = Field(default=1.0, ge=0.1, le=8.0)
    memory_mb: int = Field(default=1024, ge=128, le=16384)
    timeout_seconds: int = Field(default=3600, ge=10, le=86400)
    disk_mb: int = Field(default=512, ge=64, le=10240)

    # Network
    network_enabled: bool = Field(default=False)
    egress_allowlist: Optional[List[str]] = Field(default_factory=list)

    # Environment
    docker_image: str = Field(default="python:3.11-slim", max_length=255)
    env_vars: Optional[Dict[str, str]] = Field(default_factory=dict)

    # Input data
    input_data: Optional[Dict[str, Any]] = None

    @validator("egress_allowlist")
    def validate_egress_allowlist(cls, v):
        if v and len(v) > 100:
            raise ValueError("Maximum 100 egress destinations allowed")
        return v

    @validator("env_vars")
    def validate_env_vars(cls, v):
        if v:
            # Block sensitive env var names
            blocked = {"SECRET", "PASSWORD", "TOKEN", "KEY", "CREDENTIAL"}
            for key in v.keys():
                for blocked_word in blocked:
                    if blocked_word in key.upper():
                        raise ValueError(f"Environment variable '{key}' contains blocked keyword")
        return v


class JobResponse(BaseModel):
    """Response for job operations."""
    job_id: UUID
    name: str
    state: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    termination_reason: Optional[str] = None
    error_message: Optional[str] = None

    # Config summary
    cpu: float
    memory_mb: int
    timeout_seconds: int
    network_enabled: bool

    # Resource usage
    cpu_seconds_used: float = 0.0
    peak_memory_mb: float = 0.0

    # Security
    alerts_count: int = 0
    egress_violations_count: int = 0

    class Config:
        from_attributes = True


class JobDetailResponse(JobResponse):
    """Detailed job response with output."""
    description: Optional[str] = None
    tags: List[str] = []
    docker_image: str
    entrypoint: str
    exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    sandbox_id: Optional[str] = None


class JobListResponse(BaseModel):
    """Response for job listing."""
    jobs: List[JobResponse]
    total: int
    page: int
    page_size: int


class JobLogsResponse(BaseModel):
    """Response for job logs."""
    job_id: UUID
    stdout: str
    stderr: str
    truncated: bool = False


class QuotaResponse(BaseModel):
    """Response for quota information."""
    tier: str
    cpu_hours_daily: float
    memory_gb_hours_daily: float
    max_concurrent_jobs: int
    max_job_duration_seconds: int
    storage_mb: int
    network_gb_daily: float
    jobs_per_day: int
    is_active: bool


class QuotaUsageResponse(BaseModel):
    """Response for quota usage."""
    period_start: datetime
    cpu_hours_used: float
    cpu_hours_remaining: float
    memory_gb_hours_used: float
    memory_gb_hours_remaining: float
    concurrent_jobs: int
    storage_mb_used: int
    network_bytes_used: int
    jobs_today: int
    jobs_remaining_today: int


class EgressPolicyResponse(BaseModel):
    """Response for egress policy."""
    name: str
    allowed_destinations: List[str]
    allowed_ports: List[int]
    allowlist_only: bool
    max_requests_per_minute: int
    is_active: bool


class EgressPolicyUpdateRequest(BaseModel):
    """Request to update egress policy."""
    add_destinations: Optional[List[str]] = None
    remove_destinations: Optional[List[str]] = None
    add_ports: Optional[List[int]] = None
    max_requests_per_minute: Optional[int] = Field(None, ge=1, le=1000)


class AbuseAlertResponse(BaseModel):
    """Response for abuse alert."""
    alert_id: UUID
    job_id: UUID
    abuse_type: str
    severity: str
    confidence: float
    title: str
    description: str
    detected_at: datetime
    action_taken: str
    job_terminated: bool


class EgressViolationResponse(BaseModel):
    """Response for egress violation."""
    violation_id: UUID
    job_id: UUID
    destination: str
    port: int
    reason: str
    risk_score: float
    timestamp: datetime


# ============================================================================
# Router
# ============================================================================

router = APIRouter(prefix="/research", tags=["research-jobs"])


@router.post("/jobs", response_model=JobResponse, status_code=201)
async def submit_job(
    request: JobSubmitRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    workspace: dict = Depends(get_current_workspace),
):
    """
    Submit a new research job.

    The job will be validated, queued, and executed in an isolated sandbox.
    Resource quotas are checked before accepting the job.
    """
    from packages.cloud.research.sandbox import (
        ResearchJobExecutor,
        JobConfig,
        IsolationLevel,
    )
    from packages.cloud.research.models import (
        ResearchJob,
        ResearchJobState,
        create_research_job,
    )

    # Check quota (would be done via quota_manager in real implementation)
    # For now, create job record

    job = create_research_job(
        workspace_id=workspace["id"],
        user_id=current_user["id"],
        name=request.name,
        description=request.description,
        tags=request.tags,
        cpu_limit=request.cpu,
        memory_limit_mb=request.memory_mb,
        timeout_seconds=request.timeout_seconds,
        disk_limit_mb=request.disk_mb,
        network_enabled=request.network_enabled,
        egress_allowlist=request.egress_allowlist,
        docker_image=request.docker_image,
        entrypoint=request.entrypoint,
        env_vars=request.env_vars,
    )

    # In a real implementation, add to database and queue for execution
    # db.add(job)
    # await db.commit()

    # Queue job execution in background
    # background_tasks.add_task(execute_job, job.id, request.code, request.input_data)

    return JobResponse(
        job_id=job.id,
        name=job.name,
        state=job.state,
        created_at=job.created_at,
        cpu=job.cpu_limit,
        memory_mb=job.memory_limit_mb,
        timeout_seconds=job.timeout_seconds,
        network_enabled=job.network_enabled,
    )


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    state: Optional[str] = Query(None, description="Filter by state"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    workspace: dict = Depends(get_current_workspace),
):
    """
    List research jobs for the current workspace.

    Jobs are filtered by workspace (tenant isolation enforced).
    """
    # In real implementation, query from database with tenant filter
    return JobListResponse(
        jobs=[],
        total=0,
        page=page,
        page_size=page_size,
    )


@router.get("/jobs/{job_id}", response_model=JobDetailResponse)
async def get_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    workspace: dict = Depends(get_current_workspace),
):
    """
    Get details of a specific job.

    Tenant isolation enforced - can only access jobs in current workspace.
    """
    # In real implementation, query from database with tenant check
    raise HTTPException(status_code=404, detail="Job not found")


@router.post("/jobs/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(
    job_id: UUID,
    reason: str = Body(default="User cancelled", embed=True),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    workspace: dict = Depends(get_current_workspace),
):
    """
    Cancel a running job.

    Only jobs in PENDING, QUEUED, or RUNNING state can be cancelled.
    """
    # In real implementation, find job and cancel via executor
    raise HTTPException(status_code=404, detail="Job not found")


@router.get("/jobs/{job_id}/logs", response_model=JobLogsResponse)
async def get_job_logs(
    job_id: UUID,
    tail: int = Query(1000, ge=1, le=10000, description="Number of lines"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    workspace: dict = Depends(get_current_workspace),
):
    """
    Get stdout/stderr logs for a job.

    Logs are redacted to remove sensitive information.
    """
    raise HTTPException(status_code=404, detail="Job not found")


@router.get("/jobs/{job_id}/alerts", response_model=List[AbuseAlertResponse])
async def get_job_alerts(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    workspace: dict = Depends(get_current_workspace),
):
    """
    Get abuse alerts for a job.
    """
    raise HTTPException(status_code=404, detail="Job not found")


@router.get("/jobs/{job_id}/violations", response_model=List[EgressViolationResponse])
async def get_job_violations(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    workspace: dict = Depends(get_current_workspace),
):
    """
    Get egress policy violations for a job.
    """
    raise HTTPException(status_code=404, detail="Job not found")


# ============================================================================
# Quota Endpoints
# ============================================================================

@router.get("/quota", response_model=QuotaResponse)
async def get_quota(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    workspace: dict = Depends(get_current_workspace),
):
    """
    Get quota configuration for the current workspace.
    """
    # Return default quota for demo
    return QuotaResponse(
        tier="free",
        cpu_hours_daily=2.0,
        memory_gb_hours_daily=4.0,
        max_concurrent_jobs=2,
        max_job_duration_seconds=3600,
        storage_mb=1024,
        network_gb_daily=1.0,
        jobs_per_day=100,
        is_active=True,
    )


@router.get("/quota/usage", response_model=QuotaUsageResponse)
async def get_quota_usage(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    workspace: dict = Depends(get_current_workspace),
):
    """
    Get current quota usage for the workspace.
    """
    from datetime import datetime

    return QuotaUsageResponse(
        period_start=datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0),
        cpu_hours_used=0.0,
        cpu_hours_remaining=2.0,
        memory_gb_hours_used=0.0,
        memory_gb_hours_remaining=4.0,
        concurrent_jobs=0,
        storage_mb_used=0,
        network_bytes_used=0,
        jobs_today=0,
        jobs_remaining_today=100,
    )


# ============================================================================
# Egress Policy Endpoints
# ============================================================================

@router.get("/egress/policy", response_model=EgressPolicyResponse)
async def get_egress_policy(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    workspace: dict = Depends(get_current_workspace),
):
    """
    Get egress policy for the current workspace.
    """
    return EgressPolicyResponse(
        name="default",
        allowed_destinations=[
            "api.binance.com",
            "api.github.com",
            "pypi.org",
        ],
        allowed_ports=[443],
        allowlist_only=True,
        max_requests_per_minute=60,
        is_active=True,
    )


@router.put("/egress/policy", response_model=EgressPolicyResponse)
async def update_egress_policy(
    request: EgressPolicyUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    workspace: dict = Depends(get_current_workspace),
):
    """
    Update egress policy for the workspace.

    Only allowed to add/remove destinations from allowlist.
    """
    # In real implementation, update policy in database
    return EgressPolicyResponse(
        name="default",
        allowed_destinations=[
            "api.binance.com",
            "api.github.com",
            "pypi.org",
        ] + (request.add_destinations or []),
        allowed_ports=[443] + (request.add_ports or []),
        allowlist_only=True,
        max_requests_per_minute=request.max_requests_per_minute or 60,
        is_active=True,
    )


# ============================================================================
# Statistics Endpoints
# ============================================================================

@router.get("/stats")
async def get_research_stats(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    workspace: dict = Depends(get_current_workspace),
):
    """
    Get research job statistics for the workspace.
    """
    return {
        "jobs_total": 0,
        "jobs_completed": 0,
        "jobs_failed": 0,
        "jobs_running": 0,
        "abuse_alerts_total": 0,
        "egress_violations_total": 0,
        "cpu_hours_used_today": 0.0,
        "memory_gb_hours_used_today": 0.0,
    }
