# -*- coding: utf-8 -*-
"""
Job Scheduler - CLOUD ZONE ONLY.

Job scheduling, prioritization, and queue management.

Design Doc Reference:
    - Section 4.1: "Training Service (RL/ML jobs)"
    - Resource quotas and fair scheduling per workspace
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from typing import Any, Dict, Final, List, Optional, Set, Tuple
from uuid import UUID

from .tasks import BaseTask, TaskConfig, TaskResult, TaskState, TaskType

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

MAX_CONCURRENT_JOBS_PER_WORKSPACE: Final[int] = 10
MAX_QUEUED_JOBS_PER_WORKSPACE: Final[int] = 50
DEFAULT_JOB_TTL_HOURS: Final[int] = 24


# ============================================================================
# Priority
# ============================================================================


class JobPriority(IntEnum):
    """Job priority levels (lower value = higher priority)."""
    CRITICAL = 1
    HIGH = 3
    NORMAL = 5
    LOW = 7
    BACKGROUND = 9


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class JobConfig:
    """Job configuration for scheduling."""
    workspace_id: UUID
    user_id: UUID
    task_type: TaskType
    task_config: Dict[str, Any]
    priority: JobPriority = JobPriority.NORMAL
    schedule_at: Optional[datetime] = None  # None = immediately
    ttl_hours: int = DEFAULT_JOB_TTL_HOURS
    dependencies: List[str] = field(default_factory=list)  # Job IDs to wait for
    tags: List[str] = field(default_factory=list)


@dataclass(order=True)
class ScheduledJob:
    """
    A scheduled job in the queue.

    Ordering: priority (asc), scheduled_at (asc)
    """
    # Fields used for ordering
    priority: int = field(compare=True)
    scheduled_at: datetime = field(compare=True, default_factory=lambda: datetime.now(timezone.utc))

    # Non-ordering fields
    job_id: str = field(compare=False, default_factory=lambda: str(uuid.uuid4()))
    workspace_id: UUID = field(compare=False, default_factory=uuid.uuid4)
    user_id: UUID = field(compare=False, default_factory=uuid.uuid4)
    task_type: TaskType = field(compare=False, default=TaskType.BACKTEST)
    task_config: Dict[str, Any] = field(compare=False, default_factory=dict)

    state: TaskState = field(compare=False, default=TaskState.PENDING)
    created_at: datetime = field(compare=False, default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = field(compare=False, default=None)
    completed_at: Optional[datetime] = field(compare=False, default=None)
    expires_at: datetime = field(compare=False, default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=DEFAULT_JOB_TTL_HOURS))

    dependencies: List[str] = field(compare=False, default_factory=list)
    tags: List[str] = field(compare=False, default_factory=list)

    result: Optional[TaskResult] = field(compare=False, default=None)
    error: Optional[str] = field(compare=False, default=None)
    retry_count: int = field(compare=False, default=0)
    max_retries: int = field(compare=False, default=3)

    def is_expired(self) -> bool:
        """Check if job has expired."""
        return datetime.now(timezone.utc) > self.expires_at

    def is_ready(self) -> bool:
        """Check if job is ready to run."""
        if self.state != TaskState.PENDING:
            return False
        if self.is_expired():
            return False
        if self.scheduled_at and datetime.now(timezone.utc) < self.scheduled_at:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "workspace_id": str(self.workspace_id),
            "user_id": str(self.user_id),
            "task_type": self.task_type.value,
            "priority": self.priority,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "expires_at": self.expires_at.isoformat(),
            "tags": self.tags,
            "error": self.error,
            "retry_count": self.retry_count,
        }


@dataclass
class WorkspaceQuota:
    """Resource quota per workspace."""
    max_concurrent_jobs: int = MAX_CONCURRENT_JOBS_PER_WORKSPACE
    max_queued_jobs: int = MAX_QUEUED_JOBS_PER_WORKSPACE
    max_training_hours_per_day: int = 24
    max_backtest_hours_per_day: int = 48
    current_running: int = 0
    current_queued: int = 0
    training_hours_used: float = 0.0
    backtest_hours_used: float = 0.0


# ============================================================================
# Job Scheduler
# ============================================================================


class JobScheduler:
    """
    Job scheduler with priority queue and fair scheduling.

    Features:
    - Priority-based scheduling
    - Per-workspace resource quotas
    - Job dependencies
    - Automatic retry on failure
    - Job TTL and expiration

    SECURITY: Enforces workspace isolation.
    """

    def __init__(
        self,
        max_concurrent_global: int = 100,
        default_quota: Optional[WorkspaceQuota] = None,
    ):
        self._max_concurrent_global = max_concurrent_global
        self._default_quota = default_quota or WorkspaceQuota()

        # Job storage
        self._jobs: Dict[str, ScheduledJob] = {}
        self._job_queue: List[ScheduledJob] = []  # Heap queue

        # Workspace tracking
        self._workspace_jobs: Dict[UUID, Set[str]] = {}
        self._workspace_quotas: Dict[UUID, WorkspaceQuota] = {}
        self._workspace_running: Dict[UUID, Set[str]] = {}

        # Dependencies
        self._waiting_for: Dict[str, Set[str]] = {}  # job_id -> set of dependency job_ids
        self._dependents: Dict[str, Set[str]] = {}  # job_id -> set of dependent job_ids

        self._lock = asyncio.Lock()
        self._running = False

        logger.info("JobScheduler initialized")

    # ========================================================================
    # Job Submission
    # ========================================================================

    async def submit_job(self, config: JobConfig) -> ScheduledJob:
        """
        Submit a job for scheduling.

        Args:
            config: Job configuration

        Returns:
            ScheduledJob instance

        Raises:
            ValueError: If quota exceeded or invalid configuration
        """
        async with self._lock:
            # Check workspace quota
            quota = self._get_quota(config.workspace_id)
            workspace_jobs = self._workspace_jobs.get(config.workspace_id, set())

            if len(workspace_jobs) >= quota.max_queued_jobs:
                raise ValueError(f"Maximum queued jobs ({quota.max_queued_jobs}) exceeded for workspace")

            # Create scheduled job
            job = ScheduledJob(
                priority=config.priority.value,
                scheduled_at=config.schedule_at or datetime.now(timezone.utc),
                workspace_id=config.workspace_id,
                user_id=config.user_id,
                task_type=config.task_type,
                task_config=config.task_config,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=config.ttl_hours),
                dependencies=config.dependencies,
                tags=config.tags,
            )

            # Store job
            self._jobs[job.job_id] = job

            # Track by workspace
            if config.workspace_id not in self._workspace_jobs:
                self._workspace_jobs[config.workspace_id] = set()
            self._workspace_jobs[config.workspace_id].add(job.job_id)

            # Handle dependencies
            if config.dependencies:
                self._waiting_for[job.job_id] = set(config.dependencies)
                for dep_id in config.dependencies:
                    if dep_id not in self._dependents:
                        self._dependents[dep_id] = set()
                    self._dependents[dep_id].add(job.job_id)
            else:
                # No dependencies - add to queue
                heapq.heappush(self._job_queue, job)

            quota.current_queued += 1

        logger.info(f"Job {job.job_id} submitted (type={config.task_type.value}, priority={config.priority.name})")
        return job

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False

            if job.state in (TaskState.SUCCESS, TaskState.FAILURE, TaskState.REVOKED):
                return False  # Already terminal

            job.state = TaskState.REVOKED
            job.completed_at = datetime.now(timezone.utc)

            # Notify dependents
            await self._notify_dependents(job_id, success=False)

        logger.info(f"Job {job_id} cancelled")
        return True

    # ========================================================================
    # Job Retrieval
    # ========================================================================

    async def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Get job by ID."""
        return self._jobs.get(job_id)

    async def list_jobs(
        self,
        workspace_id: Optional[UUID] = None,
        state: Optional[TaskState] = None,
        task_type: Optional[TaskType] = None,
        limit: int = 100,
    ) -> List[ScheduledJob]:
        """List jobs with optional filters."""
        jobs = []
        for job in self._jobs.values():
            if workspace_id and job.workspace_id != workspace_id:
                continue
            if state and job.state != state:
                continue
            if task_type and job.task_type != task_type:
                continue
            jobs.append(job)
            if len(jobs) >= limit:
                break

        # Sort by created_at desc
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs

    async def get_queue_position(self, job_id: str) -> Optional[int]:
        """Get job's position in queue (0-indexed)."""
        for i, job in enumerate(sorted(self._job_queue)):
            if job.job_id == job_id:
                return i
        return None

    # ========================================================================
    # Scheduling Loop
    # ========================================================================

    async def start(self) -> None:
        """Start the scheduler loop."""
        if self._running:
            return

        self._running = True
        logger.info("JobScheduler started")

        while self._running:
            try:
                await self._process_queue()
            except Exception as e:
                logger.exception(f"Scheduler error: {e}")

            await asyncio.sleep(1)  # Check every second

    async def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False
        logger.info("JobScheduler stopped")

    async def _process_queue(self) -> None:
        """Process jobs from queue."""
        async with self._lock:
            # Count global running
            global_running = sum(
                len(jobs) for jobs in self._workspace_running.values()
            )

            if global_running >= self._max_concurrent_global:
                return  # At capacity

            # Get next ready job
            while self._job_queue:
                # Peek at top job
                if not self._job_queue:
                    break

                job = self._job_queue[0]

                if job.is_expired():
                    heapq.heappop(self._job_queue)
                    job.state = TaskState.FAILURE
                    job.error = "Job expired"
                    continue

                if not job.is_ready():
                    break  # No ready jobs

                # Check workspace quota
                quota = self._get_quota(job.workspace_id)
                running = self._workspace_running.get(job.workspace_id, set())

                if len(running) >= quota.max_concurrent_jobs:
                    # Skip this workspace, try next job
                    # Note: This is simplified - production would need fair scheduling
                    break

                # Check dependencies
                if job.job_id in self._waiting_for and self._waiting_for[job.job_id]:
                    break  # Still waiting

                # Pop and execute
                heapq.heappop(self._job_queue)
                await self._execute_job(job)
                quota.current_running += 1
                quota.current_queued -= 1

                global_running += 1
                if global_running >= self._max_concurrent_global:
                    break

    async def _execute_job(self, job: ScheduledJob) -> None:
        """Execute a job (spawn task)."""
        job.state = TaskState.STARTED
        job.started_at = datetime.now(timezone.utc)

        # Track running
        if job.workspace_id not in self._workspace_running:
            self._workspace_running[job.workspace_id] = set()
        self._workspace_running[job.workspace_id].add(job.job_id)

        logger.info(f"Executing job {job.job_id}")

        # Spawn execution (in production, this would send to Celery)
        asyncio.create_task(self._run_job(job))

    async def _run_job(self, job: ScheduledJob) -> None:
        """Run job and handle completion."""
        try:
            from .tasks import get_task_class, TaskConfig

            # Create task instance
            task_class = get_task_class(job.task_type)
            config = TaskConfig(
                workspace_id=job.workspace_id,
                user_id=job.user_id,
                **job.task_config
            )
            task = task_class(config)

            # Execute
            result = await task.run()
            job.result = result
            job.state = result.state

            if result.state == TaskState.SUCCESS:
                logger.info(f"Job {job.job_id} completed successfully")
            else:
                logger.warning(f"Job {job.job_id} failed: {result.error}")
                job.error = result.error

        except Exception as e:
            job.state = TaskState.FAILURE
            job.error = str(e)
            logger.exception(f"Job {job.job_id} error: {e}")

        finally:
            job.completed_at = datetime.now(timezone.utc)

            async with self._lock:
                # Remove from running
                if job.workspace_id in self._workspace_running:
                    self._workspace_running[job.workspace_id].discard(job.job_id)

                quota = self._get_quota(job.workspace_id)
                quota.current_running = max(0, quota.current_running - 1)

                # Notify dependents
                success = job.state == TaskState.SUCCESS
                await self._notify_dependents(job.job_id, success)

    async def _notify_dependents(self, job_id: str, success: bool) -> None:
        """Notify dependent jobs that dependency completed."""
        dependents = self._dependents.pop(job_id, set())

        for dep_job_id in dependents:
            if dep_job_id not in self._waiting_for:
                continue

            self._waiting_for[dep_job_id].discard(job_id)

            if not success:
                # Dependency failed - fail dependent
                dep_job = self._jobs.get(dep_job_id)
                if dep_job:
                    dep_job.state = TaskState.FAILURE
                    dep_job.error = f"Dependency {job_id} failed"
                self._waiting_for.pop(dep_job_id, None)

            elif not self._waiting_for[dep_job_id]:
                # All dependencies satisfied - add to queue
                self._waiting_for.pop(dep_job_id, None)
                dep_job = self._jobs.get(dep_job_id)
                if dep_job and dep_job.state == TaskState.PENDING:
                    heapq.heappush(self._job_queue, dep_job)

    # ========================================================================
    # Quota Management
    # ========================================================================

    def _get_quota(self, workspace_id: UUID) -> WorkspaceQuota:
        """Get quota for workspace."""
        if workspace_id not in self._workspace_quotas:
            self._workspace_quotas[workspace_id] = WorkspaceQuota(
                max_concurrent_jobs=self._default_quota.max_concurrent_jobs,
                max_queued_jobs=self._default_quota.max_queued_jobs,
            )
        return self._workspace_quotas[workspace_id]

    async def set_quota(
        self,
        workspace_id: UUID,
        max_concurrent: Optional[int] = None,
        max_queued: Optional[int] = None,
    ) -> WorkspaceQuota:
        """Set quota for workspace."""
        quota = self._get_quota(workspace_id)
        if max_concurrent is not None:
            quota.max_concurrent_jobs = max_concurrent
        if max_queued is not None:
            quota.max_queued_jobs = max_queued
        return quota

    # ========================================================================
    # Cleanup
    # ========================================================================

    async def cleanup_expired(self) -> int:
        """Remove expired jobs."""
        async with self._lock:
            expired = []
            for job_id, job in list(self._jobs.items()):
                if job.is_expired() and job.state in (TaskState.PENDING, TaskState.SUCCESS, TaskState.FAILURE):
                    expired.append(job_id)

            for job_id in expired:
                job = self._jobs.pop(job_id, None)
                if job:
                    if job.workspace_id in self._workspace_jobs:
                        self._workspace_jobs[job.workspace_id].discard(job_id)

            if expired:
                logger.info(f"Cleaned up {len(expired)} expired jobs")

            return len(expired)
