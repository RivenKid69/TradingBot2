# -*- coding: utf-8 -*-
"""
Tests for TenantJobIsolation.

CCEA Phase 10: Tenant isolation at job execution level.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from packages.cloud.research.sandbox.tenant_isolation import (
    TenantJobIsolation,
    TenantNamespace,
    IsolatedJobContext,
    ViolationRecord,
    TenantBoundaryViolation,
    IsolationViolationType,
    create_tenant_isolation,
    SAFE_ID_CHARS,
)


class TestTenantNamespace:
    """Tests for TenantNamespace."""

    def test_namespace_creation(self):
        """Test namespace default values."""
        namespace = TenantNamespace(tenant_id="tenant-123")

        assert namespace.tenant_id == "tenant-123"
        assert namespace.is_active is True
        assert len(namespace.active_jobs) == 0

    def test_namespace_to_dict(self):
        """Test namespace serialization."""
        namespace = TenantNamespace(
            tenant_id="tenant-123",
            max_disk_mb=5120,
        )
        data = namespace.to_dict()

        assert data["tenant_id"] == "tenant-123"
        assert data["max_disk_mb"] == 5120


class TestIsolatedJobContext:
    """Tests for IsolatedJobContext."""

    def test_context_creation(self):
        """Test context default values."""
        context = IsolatedJobContext(
            tenant_id="tenant-123",
            job_id="job-456",
        )

        assert context.tenant_id == "tenant-123"
        assert context.job_id == "job-456"
        assert context.is_active is True
        assert context.can_read_data is True
        assert context.can_write_output is True
        assert context.can_access_network is False

    def test_context_verify_path_allowed(self):
        """Test path verification - allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()

            context = IsolatedJobContext(
                tenant_id="tenant-123",
                job_id="job-456",
                workspace_path=workspace,
                allowed_paths={workspace.resolve()},
            )

            # Path inside allowed
            test_file = workspace / "test.py"
            test_file.touch()
            assert context.verify_path(test_file) is True

    def test_context_verify_path_denied(self):
        """Test path verification - denied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()

            context = IsolatedJobContext(
                tenant_id="tenant-123",
                job_id="job-456",
                workspace_path=workspace,
                allowed_paths={workspace.resolve()},
            )

            # Path outside allowed
            outside = Path(tmpdir) / "outside"
            outside.mkdir()
            test_file = outside / "test.py"
            test_file.touch()
            assert context.verify_path(test_file) is False

    def test_context_to_dict(self):
        """Test context serialization."""
        context = IsolatedJobContext(
            tenant_id="tenant-123",
            job_id="job-456",
            max_disk_mb=2048,
        )
        data = context.to_dict()

        assert data["tenant_id"] == "tenant-123"
        assert data["job_id"] == "job-456"
        assert data["max_disk_mb"] == 2048


class TestViolationRecord:
    """Tests for ViolationRecord."""

    def test_violation_creation(self):
        """Test violation record creation."""
        record = ViolationRecord(
            violation_type=IsolationViolationType.PATH_TRAVERSAL,
            tenant_id="tenant-123",
            job_id="job-456",
            attempted_resource="/etc/passwd",
        )

        assert record.violation_type == IsolationViolationType.PATH_TRAVERSAL
        assert record.tenant_id == "tenant-123"
        assert record.blocked is True

    def test_violation_to_dict(self):
        """Test violation serialization."""
        record = ViolationRecord(
            violation_type=IsolationViolationType.CROSS_TENANT_ACCESS,
            tenant_id="tenant-123",
        )
        data = record.to_dict()

        assert data["violation_type"] == "CROSS_TENANT_ACCESS"
        assert "violation_id" in data


class TestTenantJobIsolation:
    """Tests for TenantJobIsolation."""

    def test_isolation_creation(self):
        """Test isolation manager creation."""
        isolation = TenantJobIsolation()

        stats = isolation.get_stats()
        assert stats["namespaces_created"] == 0
        assert stats["active_namespaces"] == 0

    def test_create_tenant_namespace(self):
        """Test creating tenant namespace."""
        isolation = TenantJobIsolation()

        namespace = isolation.create_tenant_namespace("tenant-123")

        assert namespace.tenant_id == "tenant-123"
        assert namespace.base_path.exists()
        assert namespace.data_path.exists()
        assert namespace.jobs_path.exists()

    def test_create_tenant_namespace_idempotent(self):
        """Test creating same namespace twice returns same."""
        isolation = TenantJobIsolation()

        ns1 = isolation.create_tenant_namespace("tenant-123")
        ns2 = isolation.create_tenant_namespace("tenant-123")

        assert ns1 is ns2

    def test_create_tenant_namespace_invalid_id(self):
        """Test creating namespace with invalid ID."""
        isolation = TenantJobIsolation()

        with pytest.raises(TenantBoundaryViolation) as exc:
            isolation.create_tenant_namespace("../invalid")

        assert exc.value.violation_type == IsolationViolationType.INVALID_TENANT_ID

    def test_delete_tenant_namespace(self):
        """Test deleting tenant namespace."""
        isolation = TenantJobIsolation()
        isolation.create_tenant_namespace("tenant-123")

        result = isolation.delete_tenant_namespace("tenant-123")

        assert result is True
        assert isolation.get_namespace("tenant-123") is None

    def test_delete_tenant_namespace_with_active_jobs(self):
        """Test cannot delete namespace with active jobs."""
        isolation = TenantJobIsolation()
        isolation.create_tenant_namespace("tenant-123")
        isolation.create_job_context("tenant-123", "job-456")

        result = isolation.delete_tenant_namespace("tenant-123", force=False)

        assert result is False  # Cannot delete with active jobs

    def test_delete_tenant_namespace_force(self):
        """Test force delete namespace with active jobs."""
        isolation = TenantJobIsolation()
        isolation.create_tenant_namespace("tenant-123")
        isolation.create_job_context("tenant-123", "job-456")

        result = isolation.delete_tenant_namespace("tenant-123", force=True)

        assert result is True

    def test_get_namespace(self):
        """Test getting namespace."""
        isolation = TenantJobIsolation()
        isolation.create_tenant_namespace("tenant-123")

        namespace = isolation.get_namespace("tenant-123")

        assert namespace is not None
        assert namespace.tenant_id == "tenant-123"

    def test_get_namespace_nonexistent(self):
        """Test getting non-existent namespace."""
        isolation = TenantJobIsolation()

        namespace = isolation.get_namespace("nonexistent")

        assert namespace is None

    def test_create_job_context(self):
        """Test creating job context."""
        isolation = TenantJobIsolation()

        context = isolation.create_job_context("tenant-123", "job-456")

        assert context.tenant_id == "tenant-123"
        assert context.job_id == "job-456"
        assert context.workspace_path.exists()
        assert context.output_path.exists()
        assert context.context_token != ""

    def test_create_job_context_auto_creates_namespace(self):
        """Test job context auto-creates namespace."""
        isolation = TenantJobIsolation()

        # No namespace yet
        assert isolation.get_namespace("tenant-123") is None

        context = isolation.create_job_context("tenant-123", "job-456")

        # Namespace auto-created
        assert isolation.get_namespace("tenant-123") is not None

    def test_create_job_context_invalid_tenant(self):
        """Test creating context with invalid tenant ID."""
        isolation = TenantJobIsolation()

        with pytest.raises(TenantBoundaryViolation) as exc:
            isolation.create_job_context("/etc/passwd", "job-456")

        assert exc.value.violation_type == IsolationViolationType.INVALID_TENANT_ID

    def test_create_job_context_invalid_job(self):
        """Test creating context with invalid job ID."""
        isolation = TenantJobIsolation()

        with pytest.raises(TenantBoundaryViolation) as exc:
            isolation.create_job_context("tenant-123", "../../../etc/passwd")

        assert exc.value.violation_type == IsolationViolationType.INVALID_JOB_ID

    def test_create_job_context_exceeds_concurrent_limit(self):
        """Test concurrent job limit."""
        isolation = TenantJobIsolation()
        isolation.create_tenant_namespace("tenant-123", max_concurrent_jobs=2)

        # Create 2 jobs (at limit)
        isolation.create_job_context("tenant-123", "job-1")
        isolation.create_job_context("tenant-123", "job-2")

        # Third job should fail
        with pytest.raises(TenantBoundaryViolation) as exc:
            isolation.create_job_context("tenant-123", "job-3")

        assert exc.value.violation_type == IsolationViolationType.PERMISSION_DENIED

    def test_get_job_context(self):
        """Test getting job context."""
        isolation = TenantJobIsolation()
        isolation.create_job_context("tenant-123", "job-456")

        context = isolation.get_job_context("job-456")

        assert context is not None
        assert context.job_id == "job-456"

    def test_get_job_context_nonexistent(self):
        """Test getting non-existent context."""
        isolation = TenantJobIsolation()

        context = isolation.get_job_context("nonexistent")

        assert context is None

    def test_verify_job_context_valid(self):
        """Test verifying valid job context."""
        isolation = TenantJobIsolation()
        context = isolation.create_job_context("tenant-123", "job-456")

        result = isolation.verify_job_context("tenant-123", "job-456", context.context_token)

        assert result is True

    def test_verify_job_context_invalid_token(self):
        """Test verifying invalid token."""
        isolation = TenantJobIsolation()
        isolation.create_job_context("tenant-123", "job-456")

        result = isolation.verify_job_context("tenant-123", "job-456", "invalid-token")

        assert result is False

    def test_verify_job_context_wrong_tenant(self):
        """Test verifying with wrong tenant."""
        isolation = TenantJobIsolation()
        context = isolation.create_job_context("tenant-123", "job-456")

        result = isolation.verify_job_context("other-tenant", "job-456", context.context_token)

        assert result is False

    def test_check_path_access_allowed(self):
        """Test path access - allowed."""
        isolation = TenantJobIsolation()
        context = isolation.create_job_context("tenant-123", "job-456")

        # Create a file in workspace
        test_file = context.workspace_path / "test.py"
        test_file.touch()

        result = isolation.check_path_access("job-456", test_file)

        assert result is True

    def test_check_path_access_denied_outside(self):
        """Test path access - denied (outside allowed)."""
        isolation = TenantJobIsolation()
        isolation.create_job_context("tenant-123", "job-456")

        result = isolation.check_path_access("job-456", Path("/etc/passwd"))

        assert result is False

    def test_check_path_access_denied_traversal(self):
        """Test path access - denied (path traversal)."""
        isolation = TenantJobIsolation()
        context = isolation.create_job_context("tenant-123", "job-456")

        traversal_path = context.workspace_path / ".." / ".." / "etc" / "passwd"

        result = isolation.check_path_access("job-456", traversal_path)

        assert result is False

    def test_check_path_access_denied_proc(self):
        """Test path access - denied (/proc)."""
        isolation = TenantJobIsolation()
        isolation.create_job_context("tenant-123", "job-456")

        result = isolation.check_path_access("job-456", Path("/proc/self/environ"))

        assert result is False

    def test_check_path_access_write_to_data_denied(self):
        """Test write access to data path is denied."""
        isolation = TenantJobIsolation()
        context = isolation.create_job_context("tenant-123", "job-456")

        # Try to write to data path
        data_file = context.data_path / "data.csv"

        result = isolation.check_path_access("job-456", data_file, write=True)

        assert result is False

    def test_check_cross_tenant_access_same(self):
        """Test cross-tenant check - same tenant."""
        isolation = TenantJobIsolation()

        result = isolation.check_cross_tenant_access(
            "tenant-123", "job-456", "tenant-123", "resource"
        )

        assert result is True

    def test_check_cross_tenant_access_different(self):
        """Test cross-tenant check - different tenants."""
        isolation = TenantJobIsolation()

        result = isolation.check_cross_tenant_access(
            "tenant-123", "job-456", "tenant-other", "resource"
        )

        assert result is False

    def test_job_scope_context_manager(self):
        """Test job scope context manager."""
        isolation = TenantJobIsolation()
        context = isolation.create_job_context("tenant-123", "job-456")

        assert context.is_active is True

        with isolation.job_scope(context) as ctx:
            assert ctx.is_active is True
            assert ctx.job_id == "job-456"

        # After scope, context is inactive
        assert context.is_active is False

    def test_job_scope_inactive_context_raises(self):
        """Test job scope with inactive context raises."""
        isolation = TenantJobIsolation()
        context = isolation.create_job_context("tenant-123", "job-456")
        context.is_active = False

        with pytest.raises(TenantBoundaryViolation):
            with isolation.job_scope(context):
                pass

    def test_cleanup_job(self):
        """Test job cleanup."""
        isolation = TenantJobIsolation()
        context = isolation.create_job_context("tenant-123", "job-456")

        assert isolation.get_job_context("job-456") is not None

        result = isolation.cleanup_job("tenant-123", "job-456")

        assert result is True
        assert isolation.get_job_context("job-456") is None

    def test_cleanup_job_wrong_tenant(self):
        """Test cleanup with wrong tenant."""
        isolation = TenantJobIsolation()
        isolation.create_job_context("tenant-123", "job-456")

        result = isolation.cleanup_job("wrong-tenant", "job-456")

        assert result is False

    def test_cleanup_job_nonexistent(self):
        """Test cleanup non-existent job."""
        isolation = TenantJobIsolation()

        result = isolation.cleanup_job("tenant-123", "nonexistent")

        assert result is False

    def test_violation_callback(self):
        """Test violation callback is invoked."""
        violations = []

        def on_violation(record):
            violations.append(record)

        isolation = TenantJobIsolation(on_violation=on_violation)
        isolation.create_job_context("tenant-123", "job-456")

        # Trigger violation - /etc/passwd is caught by PATH_TRAVERSAL check
        # because it starts with /etc which is in the blocked paths check
        isolation.check_path_access("job-456", Path("/etc/passwd"))

        assert len(violations) >= 1
        # PATH_TRAVERSAL or UNAUTHORIZED_RESOURCE depending on check order
        assert violations[0].violation_type in (
            IsolationViolationType.UNAUTHORIZED_RESOURCE,
            IsolationViolationType.PATH_TRAVERSAL,
        )

    def test_get_violations(self):
        """Test getting violation history."""
        isolation = TenantJobIsolation()
        isolation.create_job_context("tenant-123", "job-456")

        # Trigger violations
        isolation.check_path_access("job-456", Path("/etc/passwd"))
        isolation.check_path_access("job-456", Path("/proc/self"))

        violations = isolation.get_violations()

        assert len(violations) >= 2

    def test_get_violations_filtered(self):
        """Test getting violations with filters."""
        isolation = TenantJobIsolation()
        isolation.create_job_context("tenant-123", "job-456")
        isolation.create_job_context("tenant-other", "job-789")

        isolation.check_path_access("job-456", Path("/etc/passwd"))
        isolation.check_path_access("job-789", Path("/etc/shadow"))

        violations = isolation.get_violations(tenant_id="tenant-123")

        assert all(v.tenant_id == "tenant-123" for v in violations)

    def test_get_stats(self):
        """Test getting statistics."""
        isolation = TenantJobIsolation()
        isolation.create_tenant_namespace("tenant-123")
        isolation.create_job_context("tenant-123", "job-456")
        isolation.check_path_access("job-456", Path("/etc/passwd"))

        stats = isolation.get_stats()

        assert stats["namespaces_created"] == 1
        assert stats["contexts_created"] == 1
        assert stats["violations_blocked"] >= 1
        assert stats["path_checks"] >= 1

    def test_get_tenant_jobs(self):
        """Test getting tenant's active jobs."""
        isolation = TenantJobIsolation()
        isolation.create_job_context("tenant-123", "job-1")
        isolation.create_job_context("tenant-123", "job-2")

        jobs = isolation.get_tenant_jobs("tenant-123")

        assert len(jobs) == 2
        assert "job-1" in jobs
        assert "job-2" in jobs


class TestIsolationViolationType:
    """Tests for IsolationViolationType enum."""

    def test_all_types_exist(self):
        """Test all violation types exist."""
        assert IsolationViolationType.PATH_TRAVERSAL
        assert IsolationViolationType.CROSS_TENANT_ACCESS
        assert IsolationViolationType.UNAUTHORIZED_RESOURCE
        assert IsolationViolationType.NAMESPACE_ESCAPE
        assert IsolationViolationType.INVALID_TENANT_ID
        assert IsolationViolationType.INVALID_JOB_ID
        assert IsolationViolationType.TOKEN_INVALID
        assert IsolationViolationType.PERMISSION_DENIED


class TestCreateTenantIsolation:
    """Tests for create_tenant_isolation factory."""

    def test_create_with_defaults(self):
        """Test creating with defaults."""
        isolation = create_tenant_isolation()

        assert isolation is not None

    def test_create_with_base_path(self):
        """Test creating with custom base path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "tenants"

            isolation = create_tenant_isolation(base_path=base_path)

            assert isolation._base_path == base_path

    def test_create_with_callback(self):
        """Test creating with violation callback."""
        violations = []

        def on_violation(record):
            violations.append(record)

        isolation = create_tenant_isolation(on_violation=on_violation)

        assert isolation._on_violation is not None


class TestSafeIdChars:
    """Tests for ID validation."""

    def test_safe_chars_alphanumeric(self):
        """Test alphanumeric chars are safe."""
        for char in "abcdefghijklmnopqrstuvwxyz":
            assert char in SAFE_ID_CHARS

        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            assert char in SAFE_ID_CHARS

        for char in "0123456789":
            assert char in SAFE_ID_CHARS

    def test_safe_chars_special(self):
        """Test allowed special chars."""
        assert "-" in SAFE_ID_CHARS
        assert "_" in SAFE_ID_CHARS

    def test_unsafe_chars_path(self):
        """Test path chars are unsafe."""
        assert "/" not in SAFE_ID_CHARS
        assert "\\" not in SAFE_ID_CHARS
        assert "." not in SAFE_ID_CHARS
