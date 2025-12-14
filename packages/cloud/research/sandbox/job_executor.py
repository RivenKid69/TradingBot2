# -*- coding: utf-8 -*-
"""
Research Job Executor - Orchestrator for cloud research jobs.

CCEA Phase 10: Complete integration of isolation components.

Design Doc 15.3/5.3:
- Sandbox for research jobs (container/VM)
- Quotas CPU/RAM/time
- Egress allowlist + abuse detection
- Tenant isolation at job execution level

This module orchestrates all isolation components:
- CloudResearchSandbox for execution isolation
- EgressFirewall for network control
- AbuseDetector for abuse prevention
- ResourceQuotaManager for quota enforcement
- TenantJobIsolation for tenant boundaries

Execution Flow:
1. Validate job request
2. Check tenant quota
3. Create isolated job context
4. Configure sandbox with egress/abuse policies
5. Execute job in sandbox
6. Monitor for abuse
7. Update quota usage
8. Cleanup and return results

Reference:
- AWS Lambda Execution Model
- Google Cloud Run
- Kubernetes Job Controller
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple
from uuid import uuid4

from packages.cloud.research.sandbox.cloud_sandbox import (
    CloudResearchSandbox,
    CloudSandboxConfig,
    CloudSandboxResult,
    CloudSandboxState,
    IsolationLevel,
)
from packages.cloud.research.sandbox.egress_firewall import (
    EgressFirewall,
    EgressPolicy,
    EgressViolation,
    EgressProtocol,
)
from packages.cloud.research.sandbox.abuse_detector import (
    AbuseDetector,
    AbuseDetectorConfig,
    AbuseAlert,
    AbuseType,
    JobMetrics,
)
from packages.cloud.research.sandbox.resource_quota import (
    ResourceQuotaManager,
    TenantQuota,
    QuotaUsage,
    QuotaCheckResult,
    QuotaExceededError,
    QuotaTier,
)
from packages.cloud.research.sandbox.tenant_isolation import (
    TenantJobIsolation,
    IsolatedJobContext,
    TenantBoundaryViolation,
)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_JOB_TIMEOUT: Final[int] = 3600  # 1 hour
MAX_JOB_TIMEOUT: Final[int] = 86400  # 24 hours
DEFAULT_CPU: Final[float] = 1.0
DEFAULT_MEMORY_MB: Final[int] = 1024

# Logging
logger = logging.getLogger(__name__)


class JobState(Enum):
    """Job lifecycle state."""
    PENDING = auto()
    VALIDATING = auto()
    QUEUED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TERMINATED = auto()
    CANCELLED = auto()


class JobTerminationReason(Enum):
    """Reason for job termination."""
    COMPLETED = auto()
    TIMEOUT = auto()
    OOM = auto()
    ABUSE_DETECTED = auto()
    QUOTA_EXCEEDED = auto()
    USER_CANCELLED = auto()
    SYSTEM_ERROR = auto()
    EGRESS_VIOLATION = auto()
    TENANT_VIOLATION = auto()


@dataclass
class JobConfig:
    """
    Configuration for a research job.
    """
    # Identity
    job_id: str = field(default_factory=lambda: str(uuid4()))
    tenant_id: str = ""
    user_id: str = ""

    # Resource requirements
    cpu: float = DEFAULT_CPU
    memory_mb: int = DEFAULT_MEMORY_MB
    timeout_seconds: int = DEFAULT_JOB_TIMEOUT
    disk_mb: int = 512

    # Isolation
    isolation_level: IsolationLevel = IsolationLevel.CONTAINER

    # Network
    network_enabled: bool = False
    egress_allowlist: List[str] = field(default_factory=list)

    # Execution
    docker_image: str = "python:3.11-slim"
    entrypoint: str = "main.py"
    env_vars: Dict[str, str] = field(default_factory=dict)

    # Metadata
    name: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "cpu": self.cpu,
            "memory_mb": self.memory_mb,
            "timeout_seconds": self.timeout_seconds,
            "disk_mb": self.disk_mb,
            "isolation_level": self.isolation_level.name,
            "network_enabled": self.network_enabled,
            "docker_image": self.docker_image,
            "name": self.name,
        }


@dataclass
class ResearchJob:
    """
    Research job instance.
    """
    config: JobConfig

    # State
    state: JobState = JobState.PENDING
    state_changed_at: datetime = field(default_factory=datetime.utcnow)

    # Execution
    sandbox_id: str = ""
    context_token: str = ""

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results
    result: Optional[CloudSandboxResult] = None
    termination_reason: Optional[JobTerminationReason] = None
    error_message: str = ""

    # Alerts
    alerts: List[AbuseAlert] = field(default_factory=list)
    egress_violations: List[EgressViolation] = field(default_factory=list)

    # Resource usage
    cpu_seconds_used: float = 0.0
    memory_mb_seconds_used: float = 0.0
    network_bytes_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.config.job_id,
            "tenant_id": self.config.tenant_id,
            "state": self.state.name,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "termination_reason": self.termination_reason.name if self.termination_reason else None,
            "error_message": self.error_message,
            "alerts_count": len(self.alerts),
            "config": self.config.to_dict(),
        }


@dataclass
class JobResult:
    """
    Complete result of job execution.
    """
    job_id: str = ""
    tenant_id: str = ""

    # Status
    success: bool = False
    state: JobState = JobState.COMPLETED
    termination_reason: Optional[JobTerminationReason] = None

    # Output
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0

    # Timing
    duration_seconds: float = 0.0
    queue_time_seconds: float = 0.0

    # Resource usage
    cpu_seconds_used: float = 0.0
    memory_mb_seconds_used: float = 0.0
    peak_memory_mb: float = 0.0
    network_bytes_used: int = 0

    # Security
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    egress_violations: List[Dict[str, Any]] = field(default_factory=list)

    # Errors
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "tenant_id": self.tenant_id,
            "success": self.success,
            "state": self.state.name,
            "termination_reason": self.termination_reason.name if self.termination_reason else None,
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
            "cpu_seconds_used": self.cpu_seconds_used,
            "peak_memory_mb": self.peak_memory_mb,
            "alerts_count": len(self.alerts),
            "errors": self.errors,
        }


class ResearchJobExecutor:
    """
    Orchestrator for cloud research job execution.

    Integrates all isolation components to provide secure,
    resource-limited execution of user research code.

    Security Features:
    - Quota enforcement before job start
    - Sandbox isolation (container/VM)
    - Network egress control
    - Abuse detection and termination
    - Tenant boundary enforcement

    Usage:
        executor = ResearchJobExecutor()

        # Configure tenant
        executor.quota_manager.set_quota(tenant_id, QuotaTier.PREMIUM)
        executor.firewall.create_policy(tenant_id, allowlist=["api.github.com"])

        # Submit job
        config = JobConfig(
            tenant_id=tenant_id,
            cpu=2.0,
            memory_mb=4096,
            timeout_seconds=3600,
        )

        result = executor.execute(config, code)

        # Or async
        job = await executor.submit(config, code)
        result = await executor.wait(job.config.job_id)
    """

    def __init__(
        self,
        quota_manager: Optional[ResourceQuotaManager] = None,
        firewall: Optional[EgressFirewall] = None,
        abuse_detector: Optional[AbuseDetector] = None,
        tenant_isolation: Optional[TenantJobIsolation] = None,
        max_workers: int = 10,
        on_job_complete: Optional[Callable[[JobResult], None]] = None,
        on_alert: Optional[Callable[[AbuseAlert], None]] = None,
    ):
        """
        Initialize executor.

        Args:
            quota_manager: Resource quota manager
            firewall: Egress firewall
            abuse_detector: Abuse detector
            tenant_isolation: Tenant isolation manager
            max_workers: Maximum concurrent jobs
            on_job_complete: Callback for job completion
            on_alert: Callback for abuse alerts
        """
        # Components
        self.quota_manager = quota_manager or ResourceQuotaManager()
        self.firewall = firewall or EgressFirewall(
            on_violation=self._on_egress_violation,
        )
        self.abuse_detector = abuse_detector or AbuseDetector(
            on_alert=self._on_abuse_alert,
            on_terminate=self._terminate_job_for_abuse,
        )
        self.tenant_isolation = tenant_isolation or TenantJobIsolation(
            on_violation=self._on_tenant_violation,
        )

        # Callbacks
        self._on_job_complete = on_job_complete
        self._on_alert = on_alert

        # State
        self._lock = threading.RLock()
        self._jobs: Dict[str, ResearchJob] = {}
        self._sandboxes: Dict[str, CloudResearchSandbox] = {}

        # Execution
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: Dict[str, Future] = {}

        # Shutdown flag
        self._shutdown = False

        # Statistics
        self._stats = {
            "jobs_submitted": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "jobs_terminated_abuse": 0,
            "jobs_terminated_quota": 0,
        }

    def execute(
        self,
        config: JobConfig,
        code: str,
        input_data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, bytes]] = None,
        wait: bool = True,
    ) -> JobResult:
        """
        Execute a research job synchronously.

        Args:
            config: Job configuration
            code: Python code to execute
            input_data: Input data for the job
            files: Additional files
            wait: Whether to wait for completion

        Returns:
            JobResult with execution outcome
        """
        # Create job
        job = self._create_job(config)

        # Validate and check quota
        validation_error = self._validate_job(job)
        if validation_error:
            return self._create_error_result(job, validation_error)

        # Execute
        try:
            result = self._execute_job(job, code, input_data, files)
            return result

        except Exception as e:
            logger.exception(f"Job execution failed: {e}")
            return self._create_error_result(job, str(e))

        finally:
            self._cleanup_job(job)

    async def submit(
        self,
        config: JobConfig,
        code: str,
        input_data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, bytes]] = None,
    ) -> ResearchJob:
        """
        Submit a job for asynchronous execution.

        Args:
            config: Job configuration
            code: Python code to execute
            input_data: Input data
            files: Additional files

        Returns:
            ResearchJob instance (use wait() to get result)
        """
        job = self._create_job(config)

        # Validate
        validation_error = self._validate_job(job)
        if validation_error:
            job.state = JobState.FAILED
            job.error_message = validation_error
            return job

        # Submit to executor
        job.state = JobState.QUEUED
        job.state_changed_at = datetime.utcnow()

        future = self._executor.submit(
            self._execute_job, job, code, input_data, files
        )

        with self._lock:
            self._futures[job.config.job_id] = future

        return job

    async def wait(
        self,
        job_id: str,
        timeout: Optional[float] = None,
    ) -> JobResult:
        """
        Wait for job completion.

        Args:
            job_id: Job identifier
            timeout: Maximum wait time

        Returns:
            JobResult
        """
        with self._lock:
            future = self._futures.get(job_id)
            job = self._jobs.get(job_id)

        if not future or not job:
            return JobResult(
                job_id=job_id,
                success=False,
                errors=["Job not found"],
            )

        try:
            result = future.result(timeout=timeout)
            return result

        except Exception as e:
            return self._create_error_result(job, str(e))

        finally:
            self._cleanup_job(job)

    def cancel(self, job_id: str, reason: str = "User cancelled") -> bool:
        """
        Cancel a running job.

        Args:
            job_id: Job identifier
            reason: Cancellation reason

        Returns:
            True if cancelled
        """
        with self._lock:
            job = self._jobs.get(job_id)
            sandbox = self._sandboxes.get(job_id)

            if not job:
                return False

            # Update state
            job.state = JobState.CANCELLED
            job.termination_reason = JobTerminationReason.USER_CANCELLED
            job.error_message = reason
            job.completed_at = datetime.utcnow()

            # Terminate sandbox
            if sandbox:
                sandbox.terminate(reason)

            return True

    def get_job(self, job_id: str) -> Optional[ResearchJob]:
        """Get job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_jobs(
        self,
        tenant_id: Optional[str] = None,
        state: Optional[JobState] = None,
        limit: int = 100,
    ) -> List[ResearchJob]:
        """Get jobs with filters."""
        with self._lock:
            jobs = list(self._jobs.values())

        if tenant_id:
            jobs = [j for j in jobs if j.config.tenant_id == tenant_id]
        if state:
            jobs = [j for j in jobs if j.state == state]

        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def _create_job(self, config: JobConfig) -> ResearchJob:
        """Create a new job instance."""
        job = ResearchJob(config=config)

        with self._lock:
            self._jobs[config.job_id] = job
            self._stats["jobs_submitted"] += 1

        return job

    def _validate_job(self, job: ResearchJob) -> Optional[str]:
        """
        Validate job and check quota.

        Returns error message or None if valid.
        """
        job.state = JobState.VALIDATING
        config = job.config

        # Validate tenant
        if not config.tenant_id:
            return "Tenant ID required"

        # Check quota
        quota_result = self.quota_manager.check_quota(
            tenant_id=config.tenant_id,
            cpu=config.cpu,
            memory_mb=config.memory_mb,
            duration_seconds=config.timeout_seconds,
            storage_mb=config.disk_mb,
        )

        if not quota_result.allowed:
            self._stats["jobs_terminated_quota"] += 1
            return f"Quota exceeded: {quota_result.message}"

        # Validate timeout
        if config.timeout_seconds > MAX_JOB_TIMEOUT:
            return f"Timeout exceeds maximum: {MAX_JOB_TIMEOUT}s"

        return None

    def _execute_job(
        self,
        job: ResearchJob,
        code: str,
        input_data: Optional[Dict[str, Any]],
        files: Optional[Dict[str, bytes]],
    ) -> JobResult:
        """Execute job in sandbox with all protections."""
        config = job.config
        result = JobResult(
            job_id=config.job_id,
            tenant_id=config.tenant_id,
        )

        start_time = time.time()

        try:
            # Create isolated job context
            job_context = self.tenant_isolation.create_job_context(
                tenant_id=config.tenant_id,
                job_id=config.job_id,
                max_disk_mb=config.disk_mb,
                can_access_network=config.network_enabled,
            )
            job.context_token = job_context.context_token

            # Create sandbox config
            sandbox_config = CloudSandboxConfig(
                sandbox_id=str(uuid4()),
                tenant_id=config.tenant_id,
                job_id=config.job_id,
                isolation_level=config.isolation_level,
                cpu_limit=config.cpu,
                memory_limit_mb=config.memory_mb,
                timeout_seconds=config.timeout_seconds,
                disk_quota_mb=config.disk_mb,
                network_enabled=config.network_enabled,
                egress_allowlist=config.egress_allowlist,
                docker_image=config.docker_image,
                env_vars=config.env_vars,
                readonly_rootfs=True,
                drop_capabilities=True,
            )

            job.sandbox_id = sandbox_config.sandbox_id

            # Create and configure sandbox
            sandbox = CloudResearchSandbox(
                config=sandbox_config,
                on_state_change=lambda s: self._on_sandbox_state_change(job, s),
            )

            with self._lock:
                self._sandboxes[config.job_id] = sandbox

            # Configure egress policy if network enabled
            if config.network_enabled:
                self.firewall.create_policy(
                    tenant_id=config.tenant_id,
                    allowlist=config.egress_allowlist,
                    include_default_allowlist=True,
                )

            # Start abuse monitoring
            self.abuse_detector.start_monitoring(
                tenant_id=config.tenant_id,
                job_id=config.job_id,
                sandbox_id=sandbox_config.sandbox_id,
            )

            # Reserve quota
            self.quota_manager.start_job(
                tenant_id=config.tenant_id,
                job_id=config.job_id,
                cpu=config.cpu,
                memory_mb=config.memory_mb,
            )

            # Update job state
            job.state = JobState.RUNNING
            job.started_at = datetime.utcnow()

            # Execute in sandbox
            with self.tenant_isolation.job_scope(job_context):
                sandbox_result = sandbox.execute(
                    code=code,
                    entrypoint=config.entrypoint,
                    input_data=input_data,
                    files=files,
                )

            # Process result
            job.result = sandbox_result
            job.completed_at = datetime.utcnow()

            # Determine termination reason
            if sandbox_result.killed_by_timeout:
                job.termination_reason = JobTerminationReason.TIMEOUT
                job.state = JobState.TERMINATED
            elif sandbox_result.killed_by_oom:
                job.termination_reason = JobTerminationReason.OOM
                job.state = JobState.TERMINATED
            elif sandbox_result.killed_by_abuse:
                job.termination_reason = JobTerminationReason.ABUSE_DETECTED
                job.state = JobState.TERMINATED
            elif sandbox_result.success:
                job.termination_reason = JobTerminationReason.COMPLETED
                job.state = JobState.COMPLETED
            else:
                job.termination_reason = JobTerminationReason.SYSTEM_ERROR
                job.state = JobState.FAILED

            # Build result
            result.success = sandbox_result.success
            result.state = job.state
            result.termination_reason = job.termination_reason
            result.stdout = sandbox_result.stdout
            result.stderr = sandbox_result.stderr
            result.exit_code = sandbox_result.exit_code
            result.duration_seconds = sandbox_result.metrics.wall_time_seconds
            result.cpu_seconds_used = sandbox_result.metrics.cpu_time_seconds
            result.peak_memory_mb = sandbox_result.metrics.peak_memory_mb
            result.alerts = [a.to_dict() for a in job.alerts]
            result.egress_violations = [v.to_dict() for v in job.egress_violations]
            result.errors = sandbox_result.errors

            # Update stats
            if result.success:
                self._stats["jobs_completed"] += 1
            else:
                self._stats["jobs_failed"] += 1

        except TenantBoundaryViolation as e:
            job.state = JobState.FAILED
            job.termination_reason = JobTerminationReason.TENANT_VIOLATION
            job.error_message = str(e)
            result.success = False
            result.errors.append(str(e))
            self._stats["jobs_failed"] += 1

        except QuotaExceededError as e:
            job.state = JobState.FAILED
            job.termination_reason = JobTerminationReason.QUOTA_EXCEEDED
            job.error_message = str(e)
            result.success = False
            result.errors.append(str(e))
            self._stats["jobs_terminated_quota"] += 1

        except Exception as e:
            job.state = JobState.FAILED
            job.termination_reason = JobTerminationReason.SYSTEM_ERROR
            job.error_message = str(e)
            result.success = False
            result.errors.append(str(e))
            self._stats["jobs_failed"] += 1
            logger.exception(f"Job {config.job_id} failed: {e}")

        finally:
            # Calculate usage
            duration = time.time() - start_time
            result.duration_seconds = duration
            job.cpu_seconds_used = config.cpu * duration
            job.memory_mb_seconds_used = config.memory_mb * duration

            # Update quota usage
            self.quota_manager.update_usage(
                tenant_id=config.tenant_id,
                job_id=config.job_id,
                cpu_seconds=job.cpu_seconds_used,
                memory_mb_seconds=job.memory_mb_seconds_used,
            )

            # End quota tracking
            self.quota_manager.end_job(config.tenant_id, config.job_id)

            # Stop abuse monitoring
            self.abuse_detector.stop_monitoring(config.job_id)

            # Trigger callback
            if self._on_job_complete:
                try:
                    self._on_job_complete(result)
                except Exception:
                    pass

        return result

    def _cleanup_job(self, job: ResearchJob) -> None:
        """Cleanup job resources."""
        config = job.config

        with self._lock:
            # Remove sandbox
            sandbox = self._sandboxes.pop(config.job_id, None)
            if sandbox:
                try:
                    sandbox.terminate("Cleanup")
                except Exception:
                    pass

            # Remove future
            self._futures.pop(config.job_id, None)

        # Cleanup tenant isolation
        try:
            self.tenant_isolation.cleanup_job(config.tenant_id, config.job_id)
        except Exception:
            pass

    def _create_error_result(self, job: ResearchJob, error: str) -> JobResult:
        """Create error result."""
        return JobResult(
            job_id=job.config.job_id,
            tenant_id=job.config.tenant_id,
            success=False,
            state=JobState.FAILED,
            termination_reason=JobTerminationReason.SYSTEM_ERROR,
            errors=[error],
        )

    def _on_sandbox_state_change(
        self,
        job: ResearchJob,
        state: CloudSandboxState,
    ) -> None:
        """Handle sandbox state changes."""
        logger.debug(f"Job {job.config.job_id} sandbox state: {state.name}")

    def _on_egress_violation(self, violation: EgressViolation) -> None:
        """Handle egress violations."""
        with self._lock:
            job = self._jobs.get(violation.job_id)
            if job:
                job.egress_violations.append(violation)

        logger.warning(
            f"Egress violation: {violation.destination}:{violation.port} "
            f"for job {violation.job_id}"
        )

    def _on_abuse_alert(self, alert: AbuseAlert) -> None:
        """Handle abuse alerts."""
        with self._lock:
            job = self._jobs.get(alert.job_id)
            if job:
                job.alerts.append(alert)

        logger.warning(
            f"Abuse alert: {alert.abuse_type.name} for job {alert.job_id}: {alert.title}"
        )

        if self._on_alert:
            try:
                self._on_alert(alert)
            except Exception:
                pass

    def _terminate_job_for_abuse(self, job_id: str, reason: str) -> bool:
        """Terminate job due to abuse."""
        with self._lock:
            job = self._jobs.get(job_id)
            sandbox = self._sandboxes.get(job_id)

            if not job:
                return False

            job.state = JobState.TERMINATED
            job.termination_reason = JobTerminationReason.ABUSE_DETECTED
            job.error_message = reason
            job.completed_at = datetime.utcnow()

            self._stats["jobs_terminated_abuse"] += 1

            if sandbox:
                sandbox.terminate(reason)

            return True

    def _on_tenant_violation(self, violation) -> None:
        """Handle tenant boundary violations."""
        logger.warning(
            f"Tenant violation: {violation.violation_type.name} "
            f"for tenant {violation.tenant_id}, job {violation.job_id}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        with self._lock:
            return {
                **self._stats,
                "active_jobs": len([j for j in self._jobs.values() if j.state == JobState.RUNNING]),
                "total_jobs": len(self._jobs),
                "quota_stats": self.quota_manager.get_stats(),
                "firewall_stats": self.firewall.get_stats(),
                "abuse_stats": self.abuse_detector.get_stats(),
                "isolation_stats": self.tenant_isolation.get_stats(),
            }

    def shutdown(self, wait: bool = True, timeout: float = 60.0) -> None:
        """
        Shutdown executor.

        Args:
            wait: Whether to wait for running jobs
            timeout: Maximum wait time
        """
        self._shutdown = True

        # Cancel pending jobs
        with self._lock:
            for job_id in list(self._jobs.keys()):
                self.cancel(job_id, "Executor shutdown")

        # Shutdown thread pool
        self._executor.shutdown(wait=wait)


def create_research_executor(
    max_workers: int = 10,
    on_job_complete: Optional[Callable[[JobResult], None]] = None,
    on_alert: Optional[Callable[[AbuseAlert], None]] = None,
) -> ResearchJobExecutor:
    """
    Create a research job executor with default configuration.

    Args:
        max_workers: Maximum concurrent jobs
        on_job_complete: Job completion callback
        on_alert: Abuse alert callback

    Returns:
        Configured ResearchJobExecutor
    """
    return ResearchJobExecutor(
        max_workers=max_workers,
        on_job_complete=on_job_complete,
        on_alert=on_alert,
    )
