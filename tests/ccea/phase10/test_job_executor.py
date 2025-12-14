# -*- coding: utf-8 -*-
"""
Tests for ResearchJobExecutor.

CCEA Phase 10: Complete integration of isolation components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from packages.cloud.research.sandbox.job_executor import (
    ResearchJobExecutor,
    ResearchJob,
    JobConfig,
    JobResult,
    JobState,
    JobTerminationReason,
    create_research_executor,
    DEFAULT_JOB_TIMEOUT,
    MAX_JOB_TIMEOUT,
    DEFAULT_CPU,
    DEFAULT_MEMORY_MB,
)
from packages.cloud.research.sandbox.cloud_sandbox import IsolationLevel
from packages.cloud.research.sandbox.resource_quota import (
    ResourceQuotaManager,
    QuotaTier,
)
from packages.cloud.research.sandbox.abuse_detector import AbuseAlert, AbuseType


class TestJobConfig:
    """Tests for JobConfig."""

    def test_default_config(self):
        """Test default config values."""
        config = JobConfig(tenant_id="tenant-123")

        assert config.tenant_id == "tenant-123"
        assert config.cpu == DEFAULT_CPU
        assert config.memory_mb == DEFAULT_MEMORY_MB
        assert config.timeout_seconds == DEFAULT_JOB_TIMEOUT
        assert config.isolation_level == IsolationLevel.CONTAINER
        assert config.network_enabled is False

    def test_custom_config(self):
        """Test custom config values."""
        config = JobConfig(
            tenant_id="tenant-123",
            cpu=4.0,
            memory_mb=8192,
            timeout_seconds=7200,
            network_enabled=True,
            egress_allowlist=["api.github.com"],
        )

        assert config.cpu == 4.0
        assert config.memory_mb == 8192
        assert config.timeout_seconds == 7200
        assert config.network_enabled is True
        assert "api.github.com" in config.egress_allowlist

    def test_config_to_dict(self):
        """Test config serialization."""
        config = JobConfig(
            tenant_id="tenant-123",
            user_id="user-456",
            name="Test Job",
        )
        data = config.to_dict()

        assert data["tenant_id"] == "tenant-123"
        assert data["user_id"] == "user-456"
        assert data["name"] == "Test Job"
        assert "job_id" in data


class TestResearchJob:
    """Tests for ResearchJob."""

    def test_job_creation(self):
        """Test job creation."""
        config = JobConfig(tenant_id="tenant-123")
        job = ResearchJob(config=config)

        assert job.config == config
        assert job.state == JobState.PENDING
        assert job.started_at is None
        assert job.completed_at is None

    def test_job_to_dict(self):
        """Test job serialization."""
        config = JobConfig(tenant_id="tenant-123")
        job = ResearchJob(config=config)
        data = job.to_dict()

        assert data["tenant_id"] == "tenant-123"
        assert data["state"] == "PENDING"
        assert "created_at" in data


class TestJobResult:
    """Tests for JobResult."""

    def test_success_result(self):
        """Test successful result."""
        result = JobResult(
            job_id="job-123",
            tenant_id="tenant-456",
            success=True,
            exit_code=0,
            stdout="Hello, World!",
        )

        assert result.success is True
        assert result.exit_code == 0
        assert result.stdout == "Hello, World!"

    def test_failure_result(self):
        """Test failure result."""
        result = JobResult(
            job_id="job-123",
            success=False,
            termination_reason=JobTerminationReason.TIMEOUT,
            errors=["Execution timed out"],
        )

        assert result.success is False
        assert result.termination_reason == JobTerminationReason.TIMEOUT

    def test_result_to_dict(self):
        """Test result serialization."""
        result = JobResult(
            job_id="job-123",
            tenant_id="tenant-456",
            success=True,
            cpu_seconds_used=100.0,
            peak_memory_mb=512.0,
        )
        data = result.to_dict()

        assert data["job_id"] == "job-123"
        assert data["success"] is True
        assert data["cpu_seconds_used"] == 100.0


class TestJobState:
    """Tests for JobState enum."""

    def test_all_states_exist(self):
        """Test all states exist."""
        assert JobState.PENDING
        assert JobState.VALIDATING
        assert JobState.QUEUED
        assert JobState.STARTING
        assert JobState.RUNNING
        assert JobState.STOPPING
        assert JobState.COMPLETED
        assert JobState.FAILED
        assert JobState.TERMINATED
        assert JobState.CANCELLED


class TestJobTerminationReason:
    """Tests for JobTerminationReason enum."""

    def test_all_reasons_exist(self):
        """Test all termination reasons exist."""
        assert JobTerminationReason.COMPLETED
        assert JobTerminationReason.TIMEOUT
        assert JobTerminationReason.OOM
        assert JobTerminationReason.ABUSE_DETECTED
        assert JobTerminationReason.QUOTA_EXCEEDED
        assert JobTerminationReason.USER_CANCELLED
        assert JobTerminationReason.SYSTEM_ERROR
        assert JobTerminationReason.EGRESS_VIOLATION
        assert JobTerminationReason.TENANT_VIOLATION


class TestResearchJobExecutor:
    """Tests for ResearchJobExecutor."""

    def test_executor_creation(self):
        """Test executor creation."""
        executor = ResearchJobExecutor()

        stats = executor.get_stats()
        assert stats["jobs_submitted"] == 0
        assert stats["jobs_completed"] == 0

    def test_executor_with_components(self):
        """Test executor with custom components."""
        quota_manager = ResourceQuotaManager()

        executor = ResearchJobExecutor(quota_manager=quota_manager)

        assert executor.quota_manager is quota_manager

    def test_executor_get_job(self):
        """Test getting job by ID."""
        executor = ResearchJobExecutor()
        config = JobConfig(tenant_id="tenant-123")

        # Submit without executing to create job
        job = executor._create_job(config)

        result = executor.get_job(config.job_id)

        assert result is not None
        assert result.config.job_id == config.job_id

    def test_executor_get_job_nonexistent(self):
        """Test getting non-existent job."""
        executor = ResearchJobExecutor()

        result = executor.get_job("nonexistent")

        assert result is None

    def test_executor_get_jobs(self):
        """Test getting jobs list."""
        executor = ResearchJobExecutor()
        executor._create_job(JobConfig(tenant_id="tenant-123"))
        executor._create_job(JobConfig(tenant_id="tenant-123"))
        executor._create_job(JobConfig(tenant_id="tenant-other"))

        # All jobs
        all_jobs = executor.get_jobs()
        assert len(all_jobs) == 3

        # Filter by tenant
        tenant_jobs = executor.get_jobs(tenant_id="tenant-123")
        assert len(tenant_jobs) == 2

    def test_executor_cancel_job(self):
        """Test cancelling a job."""
        executor = ResearchJobExecutor()
        config = JobConfig(tenant_id="tenant-123")
        job = executor._create_job(config)

        result = executor.cancel(config.job_id, "Test cancellation")

        assert result is True
        assert job.state == JobState.CANCELLED
        assert job.termination_reason == JobTerminationReason.USER_CANCELLED

    def test_executor_cancel_nonexistent(self):
        """Test cancelling non-existent job."""
        executor = ResearchJobExecutor()

        result = executor.cancel("nonexistent")

        assert result is False

    def test_validate_job_no_tenant(self):
        """Test validation fails without tenant."""
        executor = ResearchJobExecutor()
        config = JobConfig(tenant_id="")  # Empty tenant
        job = ResearchJob(config=config)

        error = executor._validate_job(job)

        assert error is not None
        assert "Tenant ID required" in error

    def test_validate_job_timeout_exceeds_max(self):
        """Test validation fails for excessive timeout."""
        executor = ResearchJobExecutor()
        executor.quota_manager.set_quota("tenant-123", QuotaTier.ENTERPRISE)

        config = JobConfig(
            tenant_id="tenant-123",
            timeout_seconds=MAX_JOB_TIMEOUT + 1,  # Exceeds max
        )
        job = ResearchJob(config=config)

        error = executor._validate_job(job)

        assert error is not None
        assert "exceeds maximum" in error

    def test_validate_job_quota_exceeded(self):
        """Test validation fails for quota exceeded."""
        executor = ResearchJobExecutor()
        executor.quota_manager.set_quota("tenant-123", QuotaTier.FREE)  # 2 CPU hours

        config = JobConfig(
            tenant_id="tenant-123",
            cpu=10.0,  # High CPU
            timeout_seconds=36000,  # Long duration = exceeds quota
        )
        job = ResearchJob(config=config)

        error = executor._validate_job(job)

        assert error is not None
        assert "Quota exceeded" in error

    def test_validate_job_success(self):
        """Test validation succeeds for valid job."""
        executor = ResearchJobExecutor()
        executor.quota_manager.set_quota("tenant-123", QuotaTier.PREMIUM)

        config = JobConfig(
            tenant_id="tenant-123",
            cpu=1.0,
            memory_mb=1024,
            timeout_seconds=1800,
        )
        job = ResearchJob(config=config)

        error = executor._validate_job(job)

        assert error is None

    def test_create_error_result(self):
        """Test creating error result."""
        executor = ResearchJobExecutor()
        config = JobConfig(tenant_id="tenant-123")
        job = ResearchJob(config=config)

        result = executor._create_error_result(job, "Test error")

        assert result.success is False
        assert result.job_id == config.job_id
        assert "Test error" in result.errors

    def test_on_job_complete_callback(self):
        """Test job completion callback."""
        results = []

        def on_complete(result):
            results.append(result)

        executor = ResearchJobExecutor(on_job_complete=on_complete)

        # Mock the execution to simulate completion
        executor.quota_manager.set_quota("tenant-123", QuotaTier.FREE)
        config = JobConfig(tenant_id="tenant-123")
        job = executor._create_job(config)

        # Trigger callback directly (simulating completion)
        if executor._on_job_complete:
            executor._on_job_complete(JobResult(job_id=config.job_id, success=True))

        assert len(results) == 1

    def test_on_alert_callback(self):
        """Test abuse alert callback."""
        alerts = []

        def on_alert(alert):
            alerts.append(alert)

        executor = ResearchJobExecutor(on_alert=on_alert)

        # Trigger alert callback directly
        alert = AbuseAlert(
            job_id="job-123",
            tenant_id="tenant-123",
            abuse_type=AbuseType.CRYPTOCURRENCY_MINING,
            title="Test alert",
            description="Test",
        )

        executor._on_abuse_alert(alert)

        assert len(alerts) == 1

    def test_terminate_job_for_abuse(self):
        """Test terminating job for abuse."""
        executor = ResearchJobExecutor()
        config = JobConfig(tenant_id="tenant-123")
        job = executor._create_job(config)

        result = executor._terminate_job_for_abuse(config.job_id, "Mining detected")

        assert result is True
        assert job.state == JobState.TERMINATED
        assert job.termination_reason == JobTerminationReason.ABUSE_DETECTED

    def test_terminate_job_nonexistent(self):
        """Test terminating non-existent job."""
        executor = ResearchJobExecutor()

        result = executor._terminate_job_for_abuse("nonexistent", "Test")

        assert result is False

    def test_get_stats(self):
        """Test getting statistics."""
        executor = ResearchJobExecutor()
        executor._create_job(JobConfig(tenant_id="tenant-123"))

        stats = executor.get_stats()

        assert stats["jobs_submitted"] == 1
        assert "quota_stats" in stats
        assert "firewall_stats" in stats
        assert "abuse_stats" in stats
        assert "isolation_stats" in stats

    def test_shutdown(self):
        """Test executor shutdown."""
        executor = ResearchJobExecutor()

        executor.shutdown(wait=False)

        assert executor._shutdown is True


class TestCreateResearchExecutor:
    """Tests for create_research_executor factory."""

    def test_create_with_defaults(self):
        """Test creating with defaults."""
        executor = create_research_executor()

        assert executor is not None
        assert executor._executor is not None

    def test_create_with_max_workers(self):
        """Test creating with custom max workers."""
        executor = create_research_executor(max_workers=5)

        assert executor is not None

    def test_create_with_callbacks(self):
        """Test creating with callbacks."""
        def on_complete(result):
            pass

        def on_alert(alert):
            pass

        executor = create_research_executor(
            on_job_complete=on_complete,
            on_alert=on_alert,
        )

        assert executor._on_job_complete is not None
        assert executor._on_alert is not None


class TestExecutorIntegration:
    """Integration tests for executor with mocked sandbox."""

    def test_execute_validates_before_run(self):
        """Test execute validates job before running."""
        executor = ResearchJobExecutor()

        config = JobConfig(tenant_id="")  # Invalid - no tenant

        result = executor.execute(config, "print('hello')")

        assert result.success is False
        assert "Tenant ID required" in result.errors

    def test_execute_checks_quota(self):
        """Test execute checks quota."""
        executor = ResearchJobExecutor()
        # Don't set quota - will use defaults which may fail

        config = JobConfig(
            tenant_id="tenant-no-quota",
            cpu=100.0,
            timeout_seconds=3600,
        )

        result = executor.execute(config, "print('hello')")

        assert result.success is False

    def test_job_state_transitions(self):
        """Test job goes through state transitions."""
        executor = ResearchJobExecutor()
        config = JobConfig(tenant_id="tenant-123")
        job = executor._create_job(config)

        # Initial state
        assert job.state == JobState.PENDING

        # After validation starts
        job.state = JobState.VALIDATING
        assert job.state == JobState.VALIDATING

        # After queued
        job.state = JobState.QUEUED
        assert job.state == JobState.QUEUED

    def test_cleanup_removes_resources(self):
        """Test cleanup removes job resources."""
        executor = ResearchJobExecutor()
        config = JobConfig(tenant_id="tenant-123")
        job = executor._create_job(config)

        executor._cleanup_job(job)

        # Future should be removed
        assert config.job_id not in executor._futures


class TestExecutorConstants:
    """Tests for executor constants."""

    def test_default_timeout(self):
        """Test default timeout value."""
        assert DEFAULT_JOB_TIMEOUT == 3600  # 1 hour

    def test_max_timeout(self):
        """Test max timeout value."""
        assert MAX_JOB_TIMEOUT == 86400  # 24 hours

    def test_default_cpu(self):
        """Test default CPU value."""
        assert DEFAULT_CPU == 1.0

    def test_default_memory(self):
        """Test default memory value."""
        assert DEFAULT_MEMORY_MB == 1024
