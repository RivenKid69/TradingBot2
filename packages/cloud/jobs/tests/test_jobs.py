# -*- coding: utf-8 -*-
"""
Tests for Jobs Package.

Tests for:
- JobScheduler
- JobMonitor
- Task definitions
"""

import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from packages.cloud.jobs import (
    JobScheduler,
    JobPriority,
    JobConfig,
    ScheduledJob,
    JobMonitor,
    JobStatus,
    JobMetrics,
)
from packages.cloud.jobs.tasks import (
    TrainingTask,
    BacktestTask,
    ResearchTask,
    ArtifactBuildTask,
    TaskType,
    TaskState,
)


# ============================================================================
# JobScheduler Tests
# ============================================================================


class TestJobScheduler:
    """Tests for JobScheduler."""

    @pytest.fixture
    def scheduler(self):
        """Create JobScheduler instance."""
        return JobScheduler()

    @pytest.fixture
    def workspace_id(self):
        """Create test workspace ID."""
        return uuid4()

    @pytest.fixture
    def user_id(self):
        """Create test user ID."""
        return uuid4()

    @pytest.mark.asyncio
    async def test_start_stop(self, scheduler):
        """Test starting and stopping scheduler."""
        import asyncio

        # Start scheduler in background task
        task = asyncio.create_task(scheduler.start())
        await asyncio.sleep(0.1)  # Give it time to start
        assert scheduler._running is True

        await scheduler.stop()
        assert scheduler._running is False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_submit_job(self, scheduler, workspace_id, user_id):
        """Test submitting a job."""
        config = JobConfig(
            workspace_id=workspace_id,
            user_id=user_id,
            task_type=TaskType.RESEARCH,
            task_config={},
        )

        job = await scheduler.submit_job(config)

        assert job is not None
        assert job.workspace_id == workspace_id
        assert job.state == TaskState.PENDING

    @pytest.mark.asyncio
    async def test_submit_job_with_priority(self, scheduler, workspace_id, user_id):
        """Test submitting job with priority."""
        config = JobConfig(
            workspace_id=workspace_id,
            user_id=user_id,
            task_type=TaskType.TRAINING,
            task_config={},
            priority=JobPriority.HIGH,
        )

        job = await scheduler.submit_job(config)

        assert job.priority == JobPriority.HIGH.value

    @pytest.mark.asyncio
    async def test_get_job(self, scheduler, workspace_id, user_id):
        """Test getting job by ID."""
        config = JobConfig(
            workspace_id=workspace_id,
            user_id=user_id,
            task_type=TaskType.BACKTEST,
            task_config={},
        )
        submitted = await scheduler.submit_job(config)

        job = await scheduler.get_job(submitted.job_id)

        assert job is not None
        assert job.job_id == submitted.job_id

    @pytest.mark.asyncio
    async def test_cancel_job(self, scheduler, workspace_id, user_id):
        """Test cancelling a job."""
        config = JobConfig(
            workspace_id=workspace_id,
            user_id=user_id,
            task_type=TaskType.RESEARCH,
            task_config={},
        )
        job = await scheduler.submit_job(config)

        cancelled = await scheduler.cancel_job(job.job_id)

        assert cancelled is True
        updated = await scheduler.get_job(job.job_id)
        assert updated.state == TaskState.REVOKED

    @pytest.mark.asyncio
    async def test_list_jobs_by_workspace(self, scheduler):
        """Test listing jobs by workspace."""
        ws1 = uuid4()
        ws2 = uuid4()
        user = uuid4()

        await scheduler.submit_job(
            JobConfig(workspace_id=ws1, user_id=user, task_type=TaskType.RESEARCH, task_config={})
        )
        await scheduler.submit_job(
            JobConfig(workspace_id=ws1, user_id=user, task_type=TaskType.TRAINING, task_config={})
        )
        await scheduler.submit_job(
            JobConfig(workspace_id=ws2, user_id=user, task_type=TaskType.BACKTEST, task_config={})
        )

        jobs_ws1 = await scheduler.list_jobs(workspace_id=ws1)

        assert len(jobs_ws1) == 2

    @pytest.mark.asyncio
    async def test_list_jobs_by_state(self, scheduler, workspace_id, user_id):
        """Test listing jobs by state."""
        config = JobConfig(
            workspace_id=workspace_id,
            user_id=user_id,
            task_type=TaskType.RESEARCH,
            task_config={},
        )
        await scheduler.submit_job(config)

        pending = await scheduler.list_jobs(state=TaskState.PENDING)

        assert len(pending) >= 1

    @pytest.mark.asyncio
    async def test_job_dependencies(self, scheduler, workspace_id, user_id):
        """Test job with dependencies."""
        job1 = await scheduler.submit_job(
            JobConfig(
                workspace_id=workspace_id,
                user_id=user_id,
                task_type=TaskType.TRAINING,
                task_config={},
            )
        )

        job2 = await scheduler.submit_job(
            JobConfig(
                workspace_id=workspace_id,
                user_id=user_id,
                task_type=TaskType.BACKTEST,
                task_config={},
                dependencies=[job1.job_id],
            )
        )

        assert job1.job_id in job2.dependencies


# ============================================================================
# JobMonitor Tests
# ============================================================================


class TestJobMonitor:
    """Tests for JobMonitor."""

    @pytest.fixture
    def monitor(self):
        """Create JobMonitor instance."""
        return JobMonitor()

    @pytest.fixture
    def scheduler(self):
        """Create JobScheduler for monitor."""
        return JobScheduler()

    @pytest.mark.asyncio
    async def test_collect_metrics(self, monitor):
        """Test collecting job metrics."""
        metrics = await monitor.collect_metrics()

        assert metrics is not None
        assert isinstance(metrics, JobMetrics)

    @pytest.mark.asyncio
    async def test_check_health(self, monitor):
        """Test health check."""
        health = await monitor.check_health()

        assert health is not None
        assert health.status in [JobStatus.HEALTHY, JobStatus.DEGRADED, JobStatus.UNHEALTHY]

    @pytest.mark.asyncio
    async def test_record_events(self, monitor):
        """Test recording submission/completion/failure events."""
        monitor.record_submission()
        monitor.record_completion()
        monitor.record_failure()

        metrics = await monitor.collect_metrics()

        # Rates should reflect recorded events
        assert metrics.submission_rate >= 0
        assert metrics.completion_rate >= 0
        assert metrics.failure_rate >= 0

    @pytest.mark.asyncio
    async def test_get_alerts(self, monitor):
        """Test getting alerts."""
        alerts = await monitor.get_alerts()

        assert isinstance(alerts, list)

    @pytest.mark.asyncio
    async def test_resolve_alert(self, monitor):
        """Test resolving an alert."""
        # Create an alert via health check
        await monitor.check_health()

        alerts = await monitor.get_alerts(limit=1)
        if alerts:
            resolved = await monitor.resolve_alert(alerts[0].alert_id)
            assert resolved is True


# ============================================================================
# Task Tests
# ============================================================================


class TestTasks:
    """Tests for task definitions."""

    def test_training_task_creation(self):
        """Test TrainingTask creation."""
        from packages.cloud.jobs.tasks import TrainingConfig

        config = TrainingConfig(
            workspace_id=uuid4(),
            user_id=uuid4(),
            model_type="ppo",
        )
        task = TrainingTask(config)

        assert task.task_type == TaskType.TRAINING
        assert task.config.model_type == "ppo"

    def test_backtest_task_creation(self):
        """Test BacktestTask creation."""
        from packages.cloud.jobs.tasks import BacktestConfig

        config = BacktestConfig(
            workspace_id=uuid4(),
            user_id=uuid4(),
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        task = BacktestTask(config)

        assert task.task_type == TaskType.BACKTEST

    def test_research_task_creation(self):
        """Test ResearchTask creation."""
        from packages.cloud.jobs.tasks import ResearchConfig

        config = ResearchConfig(
            workspace_id=uuid4(),
            user_id=uuid4(),
            notebook_id="research/analysis.ipynb",
        )
        task = ResearchTask(config)

        assert task.task_type == TaskType.RESEARCH

    def test_artifact_build_task_creation(self):
        """Test ArtifactBuildTask creation."""
        from packages.cloud.jobs.tasks import ArtifactBuildConfig

        config = ArtifactBuildConfig(
            workspace_id=uuid4(),
            user_id=uuid4(),
            strategy_version="1.0.0",
        )
        task = ArtifactBuildTask(config)

        assert task.task_type == TaskType.ARTIFACT_BUILD


class TestJobConfig:
    """Tests for JobConfig."""

    def test_default_config(self):
        """Test default JobConfig."""
        config = JobConfig(
            workspace_id=uuid4(),
            user_id=uuid4(),
            task_type=TaskType.RESEARCH,
            task_config={},
        )

        assert config.priority == JobPriority.NORMAL

    def test_config_with_priority(self):
        """Test JobConfig with priority."""
        config = JobConfig(
            workspace_id=uuid4(),
            user_id=uuid4(),
            task_type=TaskType.TRAINING,
            task_config={},
            priority=JobPriority.CRITICAL,
        )

        assert config.priority == JobPriority.CRITICAL

    def test_config_with_dependencies(self):
        """Test JobConfig with dependencies."""
        config = JobConfig(
            workspace_id=uuid4(),
            user_id=uuid4(),
            task_type=TaskType.BACKTEST,
            task_config={},
            dependencies=["job-1", "job-2"],
        )

        assert len(config.dependencies) == 2


class TestScheduledJob:
    """Tests for ScheduledJob."""

    def test_job_state_transitions(self):
        """Test job state transitions."""
        job = ScheduledJob(
            priority=JobPriority.NORMAL.value,
            workspace_id=uuid4(),
            task_type=TaskType.RESEARCH,
        )

        assert job.state == TaskState.PENDING

        job.state = TaskState.STARTED
        job.started_at = datetime.now(timezone.utc)
        assert job.state == TaskState.STARTED

        job.state = TaskState.SUCCESS
        job.completed_at = datetime.now(timezone.utc)
        assert job.state == TaskState.SUCCESS

    def test_job_duration_calculation(self):
        """Test job duration calculation."""
        job = ScheduledJob(
            priority=JobPriority.NORMAL.value,
            workspace_id=uuid4(),
            task_type=TaskType.TRAINING,
        )
        job.started_at = datetime.now(timezone.utc) - timedelta(hours=1)
        job.completed_at = datetime.now(timezone.utc)

        duration = (job.completed_at - job.started_at).total_seconds()

        assert duration >= 3600  # At least 1 hour

    def test_job_is_expired(self):
        """Test job expiration check."""
        job = ScheduledJob(
            priority=JobPriority.NORMAL.value,
            workspace_id=uuid4(),
            task_type=TaskType.RESEARCH,
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        assert job.is_expired() is True
