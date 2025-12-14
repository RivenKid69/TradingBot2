# -*- coding: utf-8 -*-
"""
Tests for CloudResearchSandbox.

CCEA Phase 10: Cloud research job isolation.
"""

import pytest
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from packages.cloud.research.sandbox.cloud_sandbox import (
    CloudResearchSandbox,
    CloudSandboxConfig,
    CloudSandboxState,
    CloudSandboxResult,
    CloudSandboxMetrics,
    IsolationLevel,
    create_cloud_sandbox,
    DEFAULT_CPU_LIMIT,
    DEFAULT_MEMORY_MB,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_CPU_LIMIT,
    MAX_MEMORY_MB,
    MAX_TIMEOUT_SECONDS,
    MIN_MEMORY_MB,
)


class TestCloudSandboxConfig:
    """Tests for CloudSandboxConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CloudSandboxConfig()

        assert config.isolation_level == IsolationLevel.CONTAINER
        assert config.cpu_limit == DEFAULT_CPU_LIMIT
        assert config.memory_limit_mb == DEFAULT_MEMORY_MB
        assert config.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
        assert config.network_enabled is False
        assert config.readonly_rootfs is True
        assert config.drop_capabilities is True

    def test_config_with_tenant(self):
        """Test configuration with tenant ID."""
        config = CloudSandboxConfig(
            tenant_id="tenant-123",
            job_id="job-456",
            cpu_limit=2.0,
            memory_limit_mb=2048,
        )

        assert config.tenant_id == "tenant-123"
        assert config.job_id == "job-456"
        assert config.cpu_limit == 2.0
        assert config.memory_limit_mb == 2048

    def test_config_validation_valid(self):
        """Test validation with valid config."""
        config = CloudSandboxConfig(
            tenant_id="tenant-123",
            cpu_limit=2.0,
            memory_limit_mb=2048,
            timeout_seconds=1800,
        )

        errors = config.validate()
        assert len(errors) == 0

    def test_config_validation_missing_tenant(self):
        """Test validation fails without tenant_id for isolated execution."""
        config = CloudSandboxConfig(
            isolation_level=IsolationLevel.CONTAINER,
            tenant_id="",  # Missing
        )

        errors = config.validate()
        assert any("tenant_id" in e for e in errors)

    def test_config_validation_cpu_exceeds_max(self):
        """Test validation fails when CPU exceeds maximum."""
        config = CloudSandboxConfig(
            tenant_id="tenant-123",
            cpu_limit=MAX_CPU_LIMIT + 1,
        )

        errors = config.validate()
        assert any("cpu_limit" in e for e in errors)

    def test_config_validation_memory_exceeds_max(self):
        """Test validation fails when memory exceeds maximum."""
        config = CloudSandboxConfig(
            tenant_id="tenant-123",
            memory_limit_mb=MAX_MEMORY_MB + 1,
        )

        errors = config.validate()
        assert any("memory_limit_mb" in e for e in errors)

    def test_config_validation_memory_below_min(self):
        """Test validation fails when memory below minimum."""
        config = CloudSandboxConfig(
            tenant_id="tenant-123",
            memory_limit_mb=MIN_MEMORY_MB - 1,
        )

        errors = config.validate()
        assert any("memory_limit_mb" in e for e in errors)

    def test_config_validation_timeout_exceeds_max(self):
        """Test validation fails when timeout exceeds maximum."""
        config = CloudSandboxConfig(
            tenant_id="tenant-123",
            timeout_seconds=MAX_TIMEOUT_SECONDS + 1,
        )

        errors = config.validate()
        assert any("timeout_seconds" in e for e in errors)

    def test_config_to_dict(self):
        """Test config serialization."""
        config = CloudSandboxConfig(
            tenant_id="tenant-123",
            job_id="job-456",
            cpu_limit=2.0,
        )

        data = config.to_dict()

        assert data["tenant_id"] == "tenant-123"
        assert data["job_id"] == "job-456"
        assert data["cpu_limit"] == 2.0
        assert "isolation_level" in data


class TestCloudSandboxMetrics:
    """Tests for CloudSandboxMetrics."""

    def test_default_metrics(self):
        """Test default metrics values."""
        metrics = CloudSandboxMetrics()

        assert metrics.wall_time_seconds == 0.0
        assert metrics.cpu_time_seconds == 0.0
        assert metrics.peak_memory_mb == 0.0
        assert metrics.oom_kills == 0

    def test_metrics_with_values(self):
        """Test metrics with custom values."""
        metrics = CloudSandboxMetrics(
            sandbox_id="sandbox-123",
            tenant_id="tenant-123",
            wall_time_seconds=60.0,
            peak_memory_mb=512.0,
        )

        assert metrics.sandbox_id == "sandbox-123"
        assert metrics.wall_time_seconds == 60.0
        assert metrics.peak_memory_mb == 512.0

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = CloudSandboxMetrics(
            sandbox_id="sandbox-123",
            wall_time_seconds=60.0,
        )

        data = metrics.to_dict()

        assert data["sandbox_id"] == "sandbox-123"
        assert data["wall_time_seconds"] == 60.0


class TestCloudSandboxResult:
    """Tests for CloudSandboxResult."""

    def test_default_result(self):
        """Test default result values."""
        result = CloudSandboxResult()

        assert result.success is False
        assert result.exit_code == 0
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.killed_by_timeout is False
        assert result.killed_by_oom is False
        assert result.killed_by_abuse is False

    def test_successful_result(self):
        """Test successful result."""
        result = CloudSandboxResult(
            sandbox_id="sandbox-123",
            job_id="job-456",
            success=True,
            exit_code=0,
            stdout="Hello World",
        )

        assert result.success is True
        assert result.stdout == "Hello World"

    def test_timeout_result(self):
        """Test timeout result."""
        result = CloudSandboxResult(
            success=False,
            killed_by_timeout=True,
            termination_reason="Timeout after 3600s",
        )

        assert result.killed_by_timeout is True
        assert "Timeout" in result.termination_reason

    def test_oom_result(self):
        """Test OOM result."""
        result = CloudSandboxResult(
            success=False,
            killed_by_oom=True,
            exit_code=137,
        )

        assert result.killed_by_oom is True
        assert result.exit_code == 137

    def test_result_to_dict(self):
        """Test result serialization."""
        result = CloudSandboxResult(
            sandbox_id="sandbox-123",
            success=True,
            stdout="output",
        )

        data = result.to_dict()

        assert data["sandbox_id"] == "sandbox-123"
        assert data["success"] is True


class TestCloudResearchSandbox:
    """Tests for CloudResearchSandbox."""

    def test_sandbox_creation(self):
        """Test sandbox creation."""
        config = CloudSandboxConfig(
            tenant_id="tenant-123",
            job_id="job-456",
            isolation_level=IsolationLevel.NONE,  # For testing
        )

        sandbox = CloudResearchSandbox(config)

        assert sandbox.sandbox_id == config.sandbox_id
        assert sandbox.state == CloudSandboxState.CREATED
        assert sandbox.is_running is False

    def test_sandbox_creation_with_callbacks(self):
        """Test sandbox creation with callbacks."""
        state_changes = []

        def on_state_change(state):
            state_changes.append(state)

        config = CloudSandboxConfig(
            tenant_id="tenant-123",
            isolation_level=IsolationLevel.NONE,
        )

        sandbox = CloudResearchSandbox(
            config,
            on_state_change=on_state_change,
        )

        assert sandbox.state == CloudSandboxState.CREATED

    def test_sandbox_initialize(self):
        """Test sandbox initialization."""
        config = CloudSandboxConfig(
            tenant_id="tenant-123",
            isolation_level=IsolationLevel.NONE,
        )

        sandbox = CloudResearchSandbox(config)
        result = sandbox.initialize()

        assert result is True
        assert sandbox.state == CloudSandboxState.READY

    def test_sandbox_execute_simple_code(self):
        """Test executing simple code."""
        config = CloudSandboxConfig(
            tenant_id="tenant-123",
            job_id="job-456",
            isolation_level=IsolationLevel.NONE,
            timeout_seconds=30,
        )

        sandbox = CloudResearchSandbox(config)

        code = """
print("Hello from sandbox")
"""

        result = sandbox.execute(code)

        assert result.success is True
        assert result.state == CloudSandboxState.STOPPED

    def test_sandbox_execute_with_error(self):
        """Test executing code that raises an error."""
        config = CloudSandboxConfig(
            tenant_id="tenant-123",
            isolation_level=IsolationLevel.NONE,
        )

        sandbox = CloudResearchSandbox(config)

        code = """
raise ValueError("Test error")
"""

        result = sandbox.execute(code)

        # May succeed or fail depending on isolation
        assert result.state in (CloudSandboxState.STOPPED, CloudSandboxState.FAILED)

    def test_sandbox_execute_with_input_data(self):
        """Test executing with input data."""
        config = CloudSandboxConfig(
            tenant_id="tenant-123",
            isolation_level=IsolationLevel.NONE,
        )

        sandbox = CloudResearchSandbox(config)

        code = """
import json
with open('input.json') as f:
    data = json.load(f)
print(f"Received: {data}")
"""

        result = sandbox.execute(
            code,
            input_data={"key": "value"},
        )

        # Check execution completed
        assert result.state in (CloudSandboxState.STOPPED, CloudSandboxState.FAILED)

    def test_sandbox_get_status(self):
        """Test getting sandbox status."""
        config = CloudSandboxConfig(
            tenant_id="tenant-123",
            job_id="job-456",
            isolation_level=IsolationLevel.NONE,
        )

        sandbox = CloudResearchSandbox(config)
        status = sandbox.get_status()

        assert status["sandbox_id"] == config.sandbox_id
        assert status["job_id"] == "job-456"
        assert status["tenant_id"] == "tenant-123"
        assert "state" in status
        assert "config" in status

    def test_sandbox_terminate(self):
        """Test sandbox termination."""
        config = CloudSandboxConfig(
            tenant_id="tenant-123",
            isolation_level=IsolationLevel.NONE,
        )

        sandbox = CloudResearchSandbox(config)
        sandbox.initialize()

        result = sandbox.terminate("Test termination")

        assert result is True
        assert sandbox.state == CloudSandboxState.TERMINATED

    def test_sandbox_metrics(self):
        """Test sandbox metrics."""
        config = CloudSandboxConfig(
            tenant_id="tenant-123",
            isolation_level=IsolationLevel.NONE,
        )

        sandbox = CloudResearchSandbox(config)
        metrics = sandbox.metrics

        assert metrics.sandbox_id == config.sandbox_id

    def test_safe_env_filtering(self):
        """Test environment variable filtering."""
        config = CloudSandboxConfig(
            tenant_id="tenant-123",
            isolation_level=IsolationLevel.NONE,
            env_vars={
                "ALLOWED_VAR": "value",
                "SECRET_KEY": "should_be_blocked",
                "AWS_ACCESS_KEY_ID": "should_be_blocked",
            },
        )

        sandbox = CloudResearchSandbox(config)
        safe_env = sandbox._get_safe_env()

        # Safe vars are included, secrets are blocked
        assert "ALLOWED_VAR" in safe_env  # Non-secret custom var is kept
        assert "SECRET_KEY" not in safe_env  # Contains 'SECRET' keyword
        assert "AWS_ACCESS_KEY_ID" not in safe_env  # AWS credential
        assert "PATH" in safe_env
        assert "LANG" in safe_env


class TestCreateCloudSandbox:
    """Tests for create_cloud_sandbox factory function."""

    def test_create_with_defaults(self):
        """Test creating sandbox with defaults."""
        sandbox = create_cloud_sandbox(
            tenant_id="tenant-123",
            job_id="job-456",
        )

        assert sandbox.config.tenant_id == "tenant-123"
        assert sandbox.config.job_id == "job-456"
        assert sandbox.config.isolation_level == IsolationLevel.CONTAINER

    def test_create_with_process_isolation(self):
        """Test creating sandbox with process isolation."""
        sandbox = create_cloud_sandbox(
            tenant_id="tenant-123",
            job_id="job-456",
            isolation_level="process",
        )

        assert sandbox.config.isolation_level == IsolationLevel.PROCESS

    def test_create_with_custom_resources(self):
        """Test creating sandbox with custom resources."""
        sandbox = create_cloud_sandbox(
            tenant_id="tenant-123",
            job_id="job-456",
            cpu_limit=4.0,
            memory_limit_mb=4096,
            timeout_seconds=7200,
        )

        assert sandbox.config.cpu_limit == 4.0
        assert sandbox.config.memory_limit_mb == 4096
        assert sandbox.config.timeout_seconds == 7200

    def test_create_with_network(self):
        """Test creating sandbox with network enabled."""
        sandbox = create_cloud_sandbox(
            tenant_id="tenant-123",
            job_id="job-456",
            network_enabled=True,
            egress_allowlist=["api.github.com"],
        )

        assert sandbox.config.network_enabled is True
        assert "api.github.com" in sandbox.config.egress_allowlist


class TestIsolationLevel:
    """Tests for IsolationLevel enum."""

    def test_isolation_levels(self):
        """Test all isolation levels exist."""
        assert IsolationLevel.NONE
        assert IsolationLevel.PROCESS
        assert IsolationLevel.CONTAINER
        assert IsolationLevel.GVISOR
        assert IsolationLevel.MICROVM

    def test_isolation_level_comparison(self):
        """Test isolation level can be compared."""
        assert IsolationLevel.NONE != IsolationLevel.CONTAINER
        assert IsolationLevel.CONTAINER == IsolationLevel.CONTAINER


class TestSandboxSecurity:
    """Security-focused tests for CloudResearchSandbox."""

    def test_config_requires_tenant_for_isolation(self):
        """Test that isolated execution requires tenant ID."""
        config = CloudSandboxConfig(
            tenant_id="",
            isolation_level=IsolationLevel.CONTAINER,
        )

        errors = config.validate()
        assert len(errors) > 0

    def test_readonly_rootfs_default(self):
        """Test read-only rootfs is enabled by default."""
        config = CloudSandboxConfig(tenant_id="tenant-123")
        assert config.readonly_rootfs is True

    def test_capabilities_dropped_default(self):
        """Test capabilities are dropped by default."""
        config = CloudSandboxConfig(tenant_id="tenant-123")
        assert config.drop_capabilities is True

    def test_network_disabled_default(self):
        """Test network is disabled by default."""
        config = CloudSandboxConfig(tenant_id="tenant-123")
        assert config.network_enabled is False

    def test_no_new_privileges_default(self):
        """Test no-new-privileges is enabled by default."""
        config = CloudSandboxConfig(tenant_id="tenant-123")
        assert config.no_new_privileges is True

    def test_env_var_security_check(self):
        """Test that sensitive env vars are blocked."""
        sandbox = CloudResearchSandbox(
            CloudSandboxConfig(
                tenant_id="tenant-123",
                isolation_level=IsolationLevel.NONE,
            )
        )

        # These should be blocked
        assert sandbox._is_safe_env_var("LD_PRELOAD") is False
        assert sandbox._is_safe_env_var("AWS_SECRET_ACCESS_KEY") is False
        assert sandbox._is_safe_env_var("GOOGLE_APPLICATION_CREDENTIALS") is False
        assert sandbox._is_safe_env_var("MY_PASSWORD") is False
        assert sandbox._is_safe_env_var("API_TOKEN") is False

        # These should be allowed
        assert sandbox._is_safe_env_var("MY_VAR") is True
        assert sandbox._is_safe_env_var("DEBUG") is True


class TestSandboxOutputTruncation:
    """Tests for output truncation."""

    def test_truncate_short_output(self):
        """Test short output is not truncated."""
        sandbox = CloudResearchSandbox(
            CloudSandboxConfig(
                tenant_id="tenant-123",
                isolation_level=IsolationLevel.NONE,
                max_output_bytes=1000,
            )
        )

        output = "short output"
        result = sandbox._truncate_output(output)

        assert result == output
        assert "truncated" not in result

    def test_truncate_long_output(self):
        """Test long output is truncated."""
        sandbox = CloudResearchSandbox(
            CloudSandboxConfig(
                tenant_id="tenant-123",
                isolation_level=IsolationLevel.NONE,
                max_output_bytes=100,
            )
        )

        output = "x" * 500
        result = sandbox._truncate_output(output)

        assert len(result) < len(output)
        assert "truncated" in result
