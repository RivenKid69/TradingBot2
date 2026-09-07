# -*- coding: utf-8 -*-
"""
Research Execution Service - Orchestrates isolated research job execution.

CCEA Phase 10: Design Doc 15.3 compliance.

Design Doc Requirements:
- Sandbox for research jobs (container/VM)
- Quotas CPU/RAM/time
- Egress allowlist + abuse detection
- Tenant isolation at job execution level
- Audit logging for all operations

This service connects the REST API to the sandbox execution layer,
providing:
1. Job submission with quota validation
2. Execution in isolated containers/VMs
3. Real-time status tracking
4. Audit trail for all operations
5. Resource usage accounting

Security:
- All jobs run in isolated environments
- Network egress is strictly controlled
- Resource limits are enforced at OS level
- Abuse detection with automatic termination
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class ExecutionMode(str, Enum):
    """Execution isolation mode."""
    PROCESS = "process"  # Process isolation (development)
    CONTAINER = "container"  # Docker container (default)
    GVISOR = "gvisor"  # gVisor sandbox (high security)
    MICROVM = "microvm"  # Firecracker microVM (enterprise)


class JobPriority(str, Enum):
    """Job execution priority."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


# ============================================================================
# Service Configuration
# ============================================================================

@dataclass
class ExecutionServiceConfig:
    """Configuration for research execution service."""
    # Execution
    default_mode: ExecutionMode = ExecutionMode.CONTAINER
    max_concurrent_jobs: int = 10
    job_queue_size: int = 100

    # Timeouts
    default_timeout_seconds: int = 3600
    max_timeout_seconds: int = 86400

    # Resources
    default_cpu: float = 1.0
    max_cpu: float = 8.0
    default_memory_mb: int = 1024
    max_memory_mb: int = 16384
    default_disk_mb: int = 512
    max_disk_mb: int = 10240

    # Network
    default_network_enabled: bool = False
    egress_allowlist_limit: int = 100

    # Docker
    docker_image: str = "python:3.11-slim"
    docker_runtime: str = "runc"  # runc, runsc (gVisor), kata

    # Audit
    audit_enabled: bool = True
    audit_log_path: Optional[Path] = None

    # Abuse detection
    abuse_detection_enabled: bool = True
    terminate_on_abuse: bool = True


# ============================================================================
# Audit Events
# ============================================================================

@dataclass
class AuditEvent:
    """Audit event for research job operations."""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: str = ""
    tenant_id: str = ""
    job_id: str = ""
    user_id: str = ""
    action: str = ""
    resource_type: str = "research_job"
    resource_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "tenant_id": self.tenant_id,
            "job_id": self.job_id,
            "user_id": self.user_id,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "success": self.success,
            "error_message": self.error_message,
        }


# ============================================================================
# Job Request/Result
# ============================================================================

@dataclass
class JobSubmitRequest:
    """Request to submit a research job."""
    tenant_id: str
    user_id: str
    workspace_id: str
    name: str
    code: str
    entrypoint: str = "main.py"
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Resources
    cpu: float = 1.0
    memory_mb: int = 1024
    timeout_seconds: int = 3600
    disk_mb: int = 512

    # Network
    network_enabled: bool = False
    egress_allowlist: List[str] = field(default_factory=list)

    # Docker
    docker_image: str = "python:3.11-slim"
    env_vars: Dict[str, str] = field(default_factory=dict)

    # Input
    input_data: Optional[Dict[str, Any]] = None
    files: Optional[Dict[str, bytes]] = None

    # Priority
    priority: JobPriority = JobPriority.NORMAL

    # Execution mode
    execution_mode: Optional[ExecutionMode] = None


@dataclass
class JobExecutionResult:
    """Result of job execution."""
    job_id: str
    tenant_id: str
    success: bool
    state: str
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    termination_reason: Optional[str] = None
    error_message: Optional[str] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    queue_time_seconds: float = 0.0

    # Resources
    cpu_seconds_used: float = 0.0
    memory_mb_seconds_used: float = 0.0
    peak_memory_mb: float = 0.0
    network_bytes_in: int = 0
    network_bytes_out: int = 0

    # Security
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    egress_violations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "tenant_id": self.tenant_id,
            "success": self.success,
            "state": self.state,
            "exit_code": self.exit_code,
            "termination_reason": self.termination_reason,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "cpu_seconds_used": self.cpu_seconds_used,
            "peak_memory_mb": self.peak_memory_mb,
            "alerts_count": len(self.alerts),
            "egress_violations_count": len(self.egress_violations),
        }


# ============================================================================
# Research Execution Service
# ============================================================================

class ResearchExecutionService:
    """
    Service for executing research jobs in isolated sandboxes.

    Design Doc 15.3 compliance:
    - Container/VM isolation
    - Resource quotas
    - Egress control
    - Abuse detection
    - Audit logging

    Usage:
        service = ResearchExecutionService(config, session)

        # Submit job
        result = await service.submit_job(request)

        # Get job status
        status = await service.get_job_status(job_id)

        # Cancel job
        await service.cancel_job(job_id)
    """

    def __init__(
        self,
        config: Optional[ExecutionServiceConfig] = None,
        session: Optional[AsyncSession] = None,
        on_job_complete: Optional[Callable[[JobExecutionResult], None]] = None,
        on_audit_event: Optional[Callable[[AuditEvent], None]] = None,
    ):
        """
        Initialize execution service.

        Args:
            config: Service configuration
            session: Database session
            on_job_complete: Callback for job completion
            on_audit_event: Callback for audit events
        """
        self.config = config or ExecutionServiceConfig()
        self._session = session
        self._on_job_complete = on_job_complete
        self._on_audit_event = on_audit_event

        # Job tracking
        self._active_jobs: Dict[str, Dict[str, Any]] = {}
        self._audit_events: List[AuditEvent] = []

        # Import sandbox components
        try:
            from packages.cloud.research.sandbox.job_executor import (
                ResearchJobExecutor,
                JobConfig,
                JobResult,
            )
            from packages.cloud.research.sandbox.cloud_sandbox import IsolationLevel
            from packages.cloud.research.sandbox.resource_quota import ResourceQuotaManager
            from packages.cloud.research.sandbox.egress_firewall import EgressFirewall
            from packages.cloud.research.sandbox.abuse_detector import AbuseDetector

            # Initialize executor
            self._quota_manager = ResourceQuotaManager()
            self._firewall = EgressFirewall(on_violation=self._on_egress_violation)
            self._abuse_detector = AbuseDetector(
                on_alert=self._on_abuse_alert,
                on_terminate=self._on_abuse_terminate,
            )

            self._executor = ResearchJobExecutor(
                quota_manager=self._quota_manager,
                firewall=self._firewall,
                abuse_detector=self._abuse_detector,
                max_workers=self.config.max_concurrent_jobs,
                on_job_complete=self._handle_job_complete,
            )

            self._sandbox_available = True
            logger.info("Research execution service initialized with sandbox support")

        except ImportError as e:
            logger.warning(f"Sandbox components not available: {e}")
            self._executor = None
            self._quota_manager = None
            self._firewall = None
            self._abuse_detector = None
            self._sandbox_available = False

    @property
    def sandbox_available(self) -> bool:
        """Check if sandbox execution is available."""
        return self._sandbox_available

    async def submit_job(self, request: JobSubmitRequest) -> JobExecutionResult:
        """
        Submit a research job for execution.

        Args:
            request: Job submission request

        Returns:
            JobExecutionResult (may be partial if async execution)
        """
        job_id = str(uuid4())
        now = datetime.now(timezone.utc)

        # Audit: Job submission
        self._emit_audit(AuditEvent(
            event_type="job.submit",
            tenant_id=request.tenant_id,
            job_id=job_id,
            user_id=request.user_id,
            action="submit",
            resource_id=job_id,
            details={
                "name": request.name,
                "cpu": request.cpu,
                "memory_mb": request.memory_mb,
                "timeout_seconds": request.timeout_seconds,
                "network_enabled": request.network_enabled,
                "execution_mode": (request.execution_mode or self.config.default_mode).value,
            },
        ))

        # Validate request
        validation_error = self._validate_request(request)
        if validation_error:
            self._emit_audit(AuditEvent(
                event_type="job.validation_failed",
                tenant_id=request.tenant_id,
                job_id=job_id,
                user_id=request.user_id,
                action="validate",
                resource_id=job_id,
                success=False,
                error_message=validation_error,
            ))
            return JobExecutionResult(
                job_id=job_id,
                tenant_id=request.tenant_id,
                success=False,
                state="FAILED",
                error_message=validation_error,
            )

        # Check if sandbox is available
        if not self._sandbox_available:
            logger.warning("Sandbox not available, returning mock result")
            return JobExecutionResult(
                job_id=job_id,
                tenant_id=request.tenant_id,
                success=False,
                state="FAILED",
                error_message="Sandbox execution not available in this environment",
            )

        try:
            # Map execution mode to isolation level
            from packages.cloud.research.sandbox.cloud_sandbox import IsolationLevel
            from packages.cloud.research.sandbox.job_executor import JobConfig

            mode_map = {
                ExecutionMode.PROCESS: IsolationLevel.PROCESS,
                ExecutionMode.CONTAINER: IsolationLevel.CONTAINER,
                ExecutionMode.GVISOR: IsolationLevel.GVISOR,
                ExecutionMode.MICROVM: IsolationLevel.MICROVM,
            }

            isolation_level = mode_map.get(
                request.execution_mode or self.config.default_mode,
                IsolationLevel.CONTAINER,
            )

            # Create job config
            job_config = JobConfig(
                job_id=job_id,
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                cpu=min(request.cpu, self.config.max_cpu),
                memory_mb=min(request.memory_mb, self.config.max_memory_mb),
                timeout_seconds=min(request.timeout_seconds, self.config.max_timeout_seconds),
                disk_mb=min(request.disk_mb, self.config.max_disk_mb),
                isolation_level=isolation_level,
                network_enabled=request.network_enabled,
                egress_allowlist=request.egress_allowlist[:self.config.egress_allowlist_limit],
                docker_image=request.docker_image or self.config.docker_image,
                entrypoint=request.entrypoint,
                env_vars=request.env_vars,
                name=request.name,
                description=request.description,
                tags=request.tags,
            )

            # Track job
            self._active_jobs[job_id] = {
                "config": job_config,
                "request": request,
                "submitted_at": now,
                "state": "QUEUED",
            }

            # Execute job
            result = self._executor.execute(
                config=job_config,
                code=request.code,
                input_data=request.input_data,
                files=request.files,
                wait=True,  # Synchronous for now
            )

            # Convert to our result format
            execution_result = JobExecutionResult(
                job_id=job_id,
                tenant_id=request.tenant_id,
                success=result.success,
                state=result.state.name,
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                termination_reason=result.termination_reason.name if result.termination_reason else None,
                error_message="; ".join(result.errors) if result.errors else None,
                started_at=now,
                completed_at=datetime.now(timezone.utc),
                duration_seconds=result.duration_seconds,
                queue_time_seconds=result.queue_time_seconds,
                cpu_seconds_used=result.cpu_seconds_used,
                memory_mb_seconds_used=result.memory_mb_seconds_used,
                peak_memory_mb=result.peak_memory_mb,
                alerts=result.alerts,
                egress_violations=result.egress_violations,
            )

            # Audit: Job completed
            self._emit_audit(AuditEvent(
                event_type="job.completed",
                tenant_id=request.tenant_id,
                job_id=job_id,
                user_id=request.user_id,
                action="complete",
                resource_id=job_id,
                success=result.success,
                details={
                    "exit_code": result.exit_code,
                    "duration_seconds": result.duration_seconds,
                    "termination_reason": execution_result.termination_reason,
                    "alerts_count": len(result.alerts),
                    "egress_violations_count": len(result.egress_violations),
                },
            ))

            # Cleanup tracking
            self._active_jobs.pop(job_id, None)

            return execution_result

        except Exception as e:
            logger.exception(f"Job execution failed: {e}")

            # Audit: Job failed
            self._emit_audit(AuditEvent(
                event_type="job.error",
                tenant_id=request.tenant_id,
                job_id=job_id,
                user_id=request.user_id,
                action="execute",
                resource_id=job_id,
                success=False,
                error_message=str(e),
            ))

            return JobExecutionResult(
                job_id=job_id,
                tenant_id=request.tenant_id,
                success=False,
                state="FAILED",
                error_message=str(e),
            )

    async def cancel_job(
        self,
        job_id: str,
        tenant_id: str,
        user_id: str,
        reason: str = "User cancelled",
    ) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: Job identifier
            tenant_id: Tenant identifier (for authorization)
            user_id: User who is cancelling
            reason: Cancellation reason

        Returns:
            True if cancelled successfully
        """
        # Audit: Cancel attempt
        self._emit_audit(AuditEvent(
            event_type="job.cancel",
            tenant_id=tenant_id,
            job_id=job_id,
            user_id=user_id,
            action="cancel",
            resource_id=job_id,
            details={"reason": reason},
        ))

        # Check if job exists and belongs to tenant
        job_info = self._active_jobs.get(job_id)
        if not job_info:
            return False

        if job_info.get("request", {}).tenant_id != tenant_id:
            logger.warning(f"Unauthorized cancel attempt: job {job_id} by tenant {tenant_id}")
            return False

        # Cancel via executor
        if self._executor:
            success = self._executor.cancel(job_id, reason)

            if success:
                self._active_jobs.pop(job_id, None)

            return success

        return False

    async def get_job_status(self, job_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job status.

        Args:
            job_id: Job identifier
            tenant_id: Tenant identifier (for authorization)

        Returns:
            Job status dict or None if not found
        """
        job_info = self._active_jobs.get(job_id)
        if not job_info:
            return None

        # Verify tenant
        if hasattr(job_info.get("request"), "tenant_id"):
            if job_info["request"].tenant_id != tenant_id:
                return None

        if self._executor:
            job = self._executor.get_job(job_id)
            if job:
                return job.to_dict()

        return {
            "job_id": job_id,
            "state": job_info.get("state", "UNKNOWN"),
            "submitted_at": job_info.get("submitted_at", datetime.now(timezone.utc)).isoformat(),
        }

    async def list_jobs(
        self,
        tenant_id: str,
        state: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List jobs for a tenant.

        Args:
            tenant_id: Tenant identifier
            state: Optional state filter
            limit: Maximum results

        Returns:
            List of job status dicts
        """
        results = []

        if self._executor:
            jobs = self._executor.get_jobs(tenant_id=tenant_id, limit=limit)
            for job in jobs:
                if state is None or job.state.name == state:
                    results.append(job.to_dict())

        return results[:limit]

    def get_executor_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        if self._executor:
            return self._executor.get_stats()
        return {"sandbox_available": False}

    def _validate_request(self, request: JobSubmitRequest) -> Optional[str]:
        """Validate job submission request."""
        # Check code length
        if len(request.code) > 1_000_000:  # 1MB limit
            return "Code exceeds maximum size (1MB)"

        # Check resources
        if request.cpu > self.config.max_cpu:
            return f"CPU limit ({request.cpu}) exceeds maximum ({self.config.max_cpu})"

        if request.memory_mb > self.config.max_memory_mb:
            return f"Memory limit ({request.memory_mb}MB) exceeds maximum ({self.config.max_memory_mb}MB)"

        if request.timeout_seconds > self.config.max_timeout_seconds:
            return f"Timeout ({request.timeout_seconds}s) exceeds maximum ({self.config.max_timeout_seconds}s)"

        # Check egress allowlist
        if len(request.egress_allowlist) > self.config.egress_allowlist_limit:
            return f"Egress allowlist size ({len(request.egress_allowlist)}) exceeds limit ({self.config.egress_allowlist_limit})"

        return None

    def _handle_job_complete(self, result) -> None:
        """Handle job completion from executor."""
        if self._on_job_complete:
            try:
                # Convert to our result type
                execution_result = JobExecutionResult(
                    job_id=result.job_id,
                    tenant_id=result.tenant_id,
                    success=result.success,
                    state=result.state.name if hasattr(result.state, 'name') else str(result.state),
                    exit_code=result.exit_code,
                    duration_seconds=result.duration_seconds,
                )
                self._on_job_complete(execution_result)
            except Exception:
                pass

    def _on_egress_violation(self, violation) -> None:
        """Handle egress violation."""
        logger.warning(
            f"Egress violation: {violation.destination}:{violation.port} "
            f"for job {violation.job_id}"
        )

        self._emit_audit(AuditEvent(
            event_type="security.egress_violation",
            tenant_id=violation.tenant_id,
            job_id=violation.job_id,
            action="egress_violation",
            resource_id=violation.job_id,
            details={
                "destination": violation.destination,
                "port": violation.port,
                "protocol": violation.protocol.value if hasattr(violation.protocol, 'value') else str(violation.protocol),
                "reason": violation.reason,
                "risk_score": violation.risk_score,
            },
        ))

    def _on_abuse_alert(self, alert) -> None:
        """Handle abuse alert."""
        logger.warning(
            f"Abuse alert: {alert.abuse_type.name} for job {alert.job_id}: {alert.title}"
        )

        self._emit_audit(AuditEvent(
            event_type="security.abuse_alert",
            tenant_id=alert.tenant_id,
            job_id=alert.job_id,
            action="abuse_detected",
            resource_id=alert.job_id,
            details={
                "abuse_type": alert.abuse_type.name,
                "severity": alert.severity.value if hasattr(alert.severity, 'value') else str(alert.severity),
                "confidence": alert.confidence,
                "title": alert.title,
                "recommendation": alert.recommendation,
            },
        ))

    def _on_abuse_terminate(self, job_id: str, reason: str) -> bool:
        """Handle abuse termination request."""
        logger.error(f"Terminating job {job_id} for abuse: {reason}")

        job_info = self._active_jobs.get(job_id)
        tenant_id = ""
        if job_info and hasattr(job_info.get("request"), "tenant_id"):
            tenant_id = job_info["request"].tenant_id

        self._emit_audit(AuditEvent(
            event_type="security.abuse_terminate",
            tenant_id=tenant_id,
            job_id=job_id,
            action="terminate",
            resource_id=job_id,
            details={"reason": reason},
        ))

        return True

    def _emit_audit(self, event: AuditEvent) -> None:
        """Emit audit event."""
        if not self.config.audit_enabled:
            return

        self._audit_events.append(event)

        # Trim audit events
        if len(self._audit_events) > 10000:
            self._audit_events = self._audit_events[-10000:]

        # Log
        logger.info(
            f"AUDIT: {event.event_type} tenant={event.tenant_id} "
            f"job={event.job_id} action={event.action} success={event.success}"
        )

        # Callback
        if self._on_audit_event:
            try:
                self._on_audit_event(event)
            except Exception:
                pass

    def get_audit_events(
        self,
        tenant_id: Optional[str] = None,
        job_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get audit events with filters."""
        events = self._audit_events

        if tenant_id:
            events = [e for e in events if e.tenant_id == tenant_id]
        if job_id:
            events = [e for e in events if e.job_id == job_id]
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return [e.to_dict() for e in events[-limit:]]


# ============================================================================
# Factory
# ============================================================================

def create_execution_service(
    session: Optional[AsyncSession] = None,
    config: Optional[ExecutionServiceConfig] = None,
) -> ResearchExecutionService:
    """
    Create a research execution service.

    Args:
        session: Optional database session
        config: Optional configuration

    Returns:
        Configured ResearchExecutionService
    """
    return ResearchExecutionService(
        config=config,
        session=session,
    )
