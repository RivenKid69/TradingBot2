# -*- coding: utf-8 -*-
"""
Job Monitor - CLOUD ZONE ONLY.

Job monitoring, health checks, and metrics collection.

Design Doc Reference:
    - Section 4.1: "Telemetry Ingestion & Monitoring"
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Set
from uuid import UUID

from .tasks import TaskState, TaskType

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

HEALTH_CHECK_INTERVAL: Final[int] = 30  # seconds
STALE_JOB_THRESHOLD: Final[int] = 3600  # 1 hour
METRICS_WINDOW: Final[int] = 3600  # 1 hour for rate calculations


# ============================================================================
# Enums
# ============================================================================


class JobStatus(str, Enum):
    """Overall job system status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class JobMetrics:
    """Metrics for job monitoring."""
    # Counts
    total_jobs: int = 0
    pending_jobs: int = 0
    running_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    cancelled_jobs: int = 0

    # By type
    jobs_by_type: Dict[str, int] = field(default_factory=dict)
    running_by_type: Dict[str, int] = field(default_factory=dict)

    # Timing
    avg_wait_time_seconds: float = 0.0
    avg_execution_time_seconds: float = 0.0
    p95_wait_time_seconds: float = 0.0
    p95_execution_time_seconds: float = 0.0

    # Rates (per hour)
    submission_rate: float = 0.0
    completion_rate: float = 0.0
    failure_rate: float = 0.0

    # Resources
    total_cpu_hours: float = 0.0
    total_gpu_hours: float = 0.0

    # Timestamp
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_jobs": self.total_jobs,
            "pending_jobs": self.pending_jobs,
            "running_jobs": self.running_jobs,
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
            "cancelled_jobs": self.cancelled_jobs,
            "jobs_by_type": self.jobs_by_type,
            "running_by_type": self.running_by_type,
            "avg_wait_time_seconds": self.avg_wait_time_seconds,
            "avg_execution_time_seconds": self.avg_execution_time_seconds,
            "p95_wait_time_seconds": self.p95_wait_time_seconds,
            "p95_execution_time_seconds": self.p95_execution_time_seconds,
            "submission_rate": self.submission_rate,
            "completion_rate": self.completion_rate,
            "failure_rate": self.failure_rate,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class JobAlert:
    """Alert for job system issues."""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    job_id: Optional[str] = None
    workspace_id: Optional[UUID] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "job_id": self.job_id,
            "workspace_id": str(self.workspace_id) if self.workspace_id else None,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class HealthStatus:
    """Health status of job system."""
    status: JobStatus
    checks: Dict[str, bool]
    metrics: JobMetrics
    alerts: List[JobAlert]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "checks": self.checks,
            "metrics": self.metrics.to_dict(),
            "alerts": [a.to_dict() for a in self.alerts[:10]],  # Latest 10
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# Job Monitor
# ============================================================================


class JobMonitor:
    """
    Job system monitor.

    Tracks:
    - Job metrics and statistics
    - System health
    - Alerts and anomalies

    SECURITY: Respects workspace isolation in metrics.
    """

    def __init__(self, scheduler=None):
        self._scheduler = scheduler
        self._alerts: List[JobAlert] = []
        self._metrics_history: List[JobMetrics] = []
        self._alert_counter = 0
        self._running = False

        # Event tracking for rate calculations
        self._submission_times: List[datetime] = []
        self._completion_times: List[datetime] = []
        self._failure_times: List[datetime] = []

        logger.info("JobMonitor initialized")

    # ========================================================================
    # Metrics Collection
    # ========================================================================

    async def collect_metrics(self, workspace_id: Optional[UUID] = None) -> JobMetrics:
        """
        Collect current job metrics.

        Args:
            workspace_id: Optional filter by workspace

        Returns:
            JobMetrics with current statistics
        """
        metrics = JobMetrics()

        if not self._scheduler:
            return metrics

        # Get jobs from scheduler
        jobs = await self._scheduler.list_jobs(workspace_id=workspace_id, limit=10000)

        wait_times = []
        exec_times = []

        for job in jobs:
            metrics.total_jobs += 1

            # Count by state
            if job.state == TaskState.PENDING:
                metrics.pending_jobs += 1
            elif job.state in (TaskState.STARTED, TaskState.RUNNING):
                metrics.running_jobs += 1
            elif job.state == TaskState.SUCCESS:
                metrics.completed_jobs += 1
            elif job.state == TaskState.FAILURE:
                metrics.failed_jobs += 1
            elif job.state == TaskState.REVOKED:
                metrics.cancelled_jobs += 1

            # Count by type
            type_key = job.task_type.value
            metrics.jobs_by_type[type_key] = metrics.jobs_by_type.get(type_key, 0) + 1
            if job.state in (TaskState.STARTED, TaskState.RUNNING):
                metrics.running_by_type[type_key] = metrics.running_by_type.get(type_key, 0) + 1

            # Calculate timing
            if job.started_at and job.created_at:
                wait_time = (job.started_at - job.created_at).total_seconds()
                wait_times.append(wait_time)

            if job.completed_at and job.started_at:
                exec_time = (job.completed_at - job.started_at).total_seconds()
                exec_times.append(exec_time)

        # Calculate averages
        if wait_times:
            metrics.avg_wait_time_seconds = sum(wait_times) / len(wait_times)
            sorted_wait = sorted(wait_times)
            p95_idx = int(len(sorted_wait) * 0.95)
            metrics.p95_wait_time_seconds = sorted_wait[min(p95_idx, len(sorted_wait) - 1)]

        if exec_times:
            metrics.avg_execution_time_seconds = sum(exec_times) / len(exec_times)
            sorted_exec = sorted(exec_times)
            p95_idx = int(len(sorted_exec) * 0.95)
            metrics.p95_execution_time_seconds = sorted_exec[min(p95_idx, len(sorted_exec) - 1)]

        # Calculate rates
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=METRICS_WINDOW)

        self._submission_times = [t for t in self._submission_times if t > window_start]
        self._completion_times = [t for t in self._completion_times if t > window_start]
        self._failure_times = [t for t in self._failure_times if t > window_start]

        hours = METRICS_WINDOW / 3600
        metrics.submission_rate = len(self._submission_times) / hours
        metrics.completion_rate = len(self._completion_times) / hours
        metrics.failure_rate = len(self._failure_times) / hours

        # Store in history
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > 1000:
            self._metrics_history = self._metrics_history[-500:]

        return metrics

    async def get_workspace_metrics(self, workspace_id: UUID) -> JobMetrics:
        """Get metrics for specific workspace."""
        return await self.collect_metrics(workspace_id=workspace_id)

    # ========================================================================
    # Health Checks
    # ========================================================================

    async def check_health(self) -> HealthStatus:
        """
        Perform health check of job system.

        Returns:
            HealthStatus with check results
        """
        checks = {
            "scheduler_running": False,
            "queue_healthy": False,
            "no_stale_jobs": False,
            "failure_rate_ok": False,
            "workers_available": True,  # Assume available
        }

        alerts = []
        metrics = await self.collect_metrics()

        # Check scheduler
        if self._scheduler and self._scheduler._running:
            checks["scheduler_running"] = True
        else:
            alerts.append(self._create_alert(
                AlertSeverity.ERROR,
                "Scheduler Not Running",
                "Job scheduler is not running",
            ))

        # Check queue health
        if metrics.pending_jobs < 1000:  # Arbitrary threshold
            checks["queue_healthy"] = True
        else:
            alerts.append(self._create_alert(
                AlertSeverity.WARNING,
                "Queue Backlog",
                f"Large queue backlog: {metrics.pending_jobs} pending jobs",
            ))

        # Check for stale jobs
        stale_count = await self._count_stale_jobs()
        if stale_count == 0:
            checks["no_stale_jobs"] = True
        else:
            alerts.append(self._create_alert(
                AlertSeverity.WARNING,
                "Stale Jobs Detected",
                f"{stale_count} jobs appear stale (running > {STALE_JOB_THRESHOLD}s)",
            ))

        # Check failure rate
        if metrics.failure_rate < 10:  # Less than 10 failures/hour
            checks["failure_rate_ok"] = True
        else:
            alerts.append(self._create_alert(
                AlertSeverity.WARNING,
                "High Failure Rate",
                f"Job failure rate: {metrics.failure_rate:.1f}/hour",
            ))

        # Determine overall status
        failed_checks = [k for k, v in checks.items() if not v]
        if not failed_checks:
            status = JobStatus.HEALTHY
        elif len(failed_checks) <= 2:
            status = JobStatus.DEGRADED
        else:
            status = JobStatus.UNHEALTHY

        return HealthStatus(
            status=status,
            checks=checks,
            metrics=metrics,
            alerts=alerts,
        )

    async def _count_stale_jobs(self) -> int:
        """Count jobs that appear stale."""
        if not self._scheduler:
            return 0

        count = 0
        now = datetime.now(timezone.utc)
        jobs = await self._scheduler.list_jobs(state=TaskState.RUNNING, limit=1000)

        for job in jobs:
            if job.started_at:
                elapsed = (now - job.started_at).total_seconds()
                if elapsed > STALE_JOB_THRESHOLD:
                    count += 1

        return count

    # ========================================================================
    # Alerts
    # ========================================================================

    def _create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        job_id: Optional[str] = None,
        workspace_id: Optional[UUID] = None,
    ) -> JobAlert:
        """Create a new alert."""
        self._alert_counter += 1
        alert = JobAlert(
            alert_id=f"alert_{self._alert_counter}",
            severity=severity,
            title=title,
            message=message,
            job_id=job_id,
            workspace_id=workspace_id,
        )
        self._alerts.append(alert)

        # Keep only recent alerts
        if len(self._alerts) > 1000:
            self._alerts = self._alerts[-500:]

        logger.warning(f"Job alert: [{severity.value}] {title}: {message}")
        return alert

    async def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        workspace_id: Optional[UUID] = None,
        limit: int = 100,
    ) -> List[JobAlert]:
        """Get recent alerts."""
        alerts = []
        for alert in reversed(self._alerts):
            if severity and alert.severity != severity:
                continue
            if workspace_id and alert.workspace_id != workspace_id:
                continue
            alerts.append(alert)
            if len(alerts) >= limit:
                break
        return alerts

    async def resolve_alert(self, alert_id: str) -> bool:
        """Mark alert as resolved."""
        for alert in self._alerts:
            if alert.alert_id == alert_id and not alert.resolved_at:
                alert.resolved_at = datetime.now(timezone.utc)
                return True
        return False

    # ========================================================================
    # Event Recording
    # ========================================================================

    def record_submission(self) -> None:
        """Record job submission event."""
        self._submission_times.append(datetime.now(timezone.utc))

    def record_completion(self) -> None:
        """Record job completion event."""
        self._completion_times.append(datetime.now(timezone.utc))

    def record_failure(self) -> None:
        """Record job failure event."""
        self._failure_times.append(datetime.now(timezone.utc))

    # ========================================================================
    # Monitoring Loop
    # ========================================================================

    async def start(self) -> None:
        """Start monitoring loop."""
        if self._running:
            return

        self._running = True
        logger.info("JobMonitor started")

        while self._running:
            try:
                await self.check_health()
            except Exception as e:
                logger.exception(f"Monitor error: {e}")

            await asyncio.sleep(HEALTH_CHECK_INTERVAL)

    async def stop(self) -> None:
        """Stop monitoring loop."""
        self._running = False
        logger.info("JobMonitor stopped")
