# -*- coding: utf-8 -*-
"""
Tests for Sandbox Permissions Enforcer - Phase 4.4A

Tests cover:
- Permission computation from manifest + policy
- Network access checks
- Filesystem access checks
- Resource limit enforcement
- Violation logging
"""

import pytest

from packages.agent.daemon.artifact_manager import ArtifactManifest
from packages.agent.daemon.sandbox_enforcer import (
    SandboxPermissionsEnforcer,
    EnforcedPermissions,
    PermissionViolationType,
)
from packages.agent.policy.layered_policy import LayeredPolicyEngine, PolicySource


class TestEnforcedPermissions:
    """Test EnforcedPermissions dataclass."""

    def test_default_values(self):
        """Test default restrictive values."""
        perms = EnforcedPermissions()

        assert perms.network_enabled is False
        assert perms.egress_allowlist == []
        assert perms.filesystem_readonly is True
        assert perms.max_memory_mb == 256
        assert perms.max_cpu_percent == 50.0

    def test_to_dict(self):
        """Test serialization."""
        perms = EnforcedPermissions(
            network_enabled=True,
            egress_allowlist=["example.com"],
            max_memory_mb=512,
        )

        data = perms.to_dict()

        assert data["network_enabled"] is True
        assert "example.com" in data["egress_allowlist"]
        assert data["max_memory_mb"] == 512

    def test_to_sandbox_config(self):
        """Test conversion to SandboxConfig."""
        perms = EnforcedPermissions(
            network_enabled=True,
            max_memory_mb=512,
            max_cpu_percent=75.0,
            max_execution_time_seconds=600,
        )

        config = perms.to_sandbox_config()

        assert config.network_enabled is True
        assert config.memory_limit_mb == 512
        assert config.cpu_limit == 0.75  # 75% = 0.75
        assert config.timeout_seconds == 600


class TestSandboxPermissionsEnforcer:
    """Test SandboxPermissionsEnforcer operations."""

    @pytest.fixture
    def policy_engine(self):
        """Create policy engine."""
        return LayeredPolicyEngine()

    @pytest.fixture
    def enforcer(self, policy_engine):
        """Create enforcer with policy engine."""
        return SandboxPermissionsEnforcer(policy_engine=policy_engine)

    def test_compute_permissions_basic(self, enforcer):
        """Test computing permissions from manifest."""
        manifest = ArtifactManifest(
            name="test",
            network_allowed=True,
            egress_allowlist=["api.example.com"],
            max_memory_mb=256,
        )

        perms = enforcer.compute_permissions(manifest, "artifact-1")

        assert perms.network_enabled is True
        assert "api.example.com" in perms.egress_allowlist
        assert perms.max_memory_mb == 256

    def test_local_policy_restricts_manifest(self, policy_engine, enforcer):
        """Test local policy can restrict manifest permissions."""
        # Manifest allows 512MB
        manifest = ArtifactManifest(max_memory_mb=512)

        # Local policy restricts to 256MB
        policy_engine.set_local_policy({"max_memory_mb": 256})

        perms = enforcer.compute_permissions(manifest, "artifact-1")

        # Should use minimum (most restrictive)
        assert perms.max_memory_mb == 256

    def test_local_policy_disables_network(self, policy_engine, enforcer):
        """Test local policy can disable network even if manifest allows."""
        manifest = ArtifactManifest(network_allowed=True)
        policy_engine.set_local_policy({"network_enabled": False})

        perms = enforcer.compute_permissions(manifest, "artifact-1")

        assert perms.network_enabled is False

    def test_egress_allowlist_intersection(self, policy_engine, enforcer):
        """Test egress allowlist is intersection of manifest and local."""
        manifest = ArtifactManifest(
            network_allowed=True,
            egress_allowlist=["a.com", "b.com", "c.com"],
        )
        policy_engine.set_local_policy({
            "egress_allowlist": ["b.com", "c.com", "d.com"],
        })

        perms = enforcer.compute_permissions(manifest, "artifact-1")

        # Intersection: b.com, c.com
        assert set(perms.egress_allowlist) == {"b.com", "c.com"}


class TestNetworkAccessChecks:
    """Test network access permission checks."""

    @pytest.fixture
    def enforcer(self):
        """Create enforcer."""
        return SandboxPermissionsEnforcer()

    def test_network_disabled_blocks_all(self, enforcer):
        """Test network disabled blocks all hosts."""
        manifest = ArtifactManifest(network_allowed=False)
        enforcer.compute_permissions(manifest, "artifact-1")

        allowed, violation = enforcer.can_access_network("example.com")

        assert allowed is False
        assert violation is not None
        assert violation.violation_type == PermissionViolationType.NETWORK_BLOCKED

    def test_network_enabled_no_allowlist_allows_all(self, enforcer):
        """Test network enabled with empty allowlist allows all."""
        manifest = ArtifactManifest(
            network_allowed=True,
            egress_allowlist=[],
        )
        enforcer.compute_permissions(manifest, "artifact-1")

        allowed, violation = enforcer.can_access_network("any.example.com")

        assert allowed is True
        assert violation is None

    def test_egress_allowlist_enforced(self, enforcer):
        """Test egress allowlist is enforced."""
        manifest = ArtifactManifest(
            network_allowed=True,
            egress_allowlist=["api.binance.com", "api.coinbase.com"],
        )
        enforcer.compute_permissions(manifest, "artifact-1")

        # Allowed host
        allowed1, _ = enforcer.can_access_network("api.binance.com")
        assert allowed1 is True

        # Blocked host
        allowed2, violation = enforcer.can_access_network("malicious.com")
        assert allowed2 is False
        assert violation.violation_type == PermissionViolationType.NETWORK_EGRESS_DENIED


class TestFilesystemAccessChecks:
    """Test filesystem access permission checks."""

    @pytest.fixture
    def enforcer(self):
        """Create enforcer."""
        return SandboxPermissionsEnforcer()

    def test_readonly_blocks_writes(self, enforcer):
        """Test readonly filesystem blocks writes."""
        manifest = ArtifactManifest(filesystem_readonly=True)
        enforcer.compute_permissions(manifest, "artifact-1")

        allowed, violation = enforcer.can_write_file("/tmp/test.txt")

        assert allowed is False
        assert violation.violation_type == PermissionViolationType.FILESYSTEM_WRITE_DENIED

    def test_allowed_paths_enforced(self, enforcer):
        """Test allowed paths are enforced for writes."""
        manifest = ArtifactManifest(
            filesystem_readonly=False,
            allowed_paths=["/tmp", "/var/data"],
        )
        enforcer.compute_permissions(manifest, "artifact-1")

        # Allowed path
        allowed1, _ = enforcer.can_write_file("/tmp/test.txt")
        assert allowed1 is True

        # Blocked path
        allowed2, violation = enforcer.can_write_file("/etc/passwd")
        assert allowed2 is False
        assert violation.violation_type == PermissionViolationType.FILESYSTEM_PATH_DENIED


class TestResourceLimitChecks:
    """Test resource limit enforcement."""

    @pytest.fixture
    def enforcer(self):
        """Create enforcer."""
        return SandboxPermissionsEnforcer()

    def test_memory_limit_enforced(self, enforcer):
        """Test memory limit enforcement."""
        manifest = ArtifactManifest(max_memory_mb=256)
        enforcer.compute_permissions(manifest, "artifact-1")

        # Within limit
        within, _ = enforcer.check_resource_usage(memory_mb=200)
        assert within is True

        # Exceeds limit
        exceeds, violation = enforcer.check_resource_usage(memory_mb=300)
        assert exceeds is False
        assert violation.violation_type == PermissionViolationType.MEMORY_EXCEEDED

    def test_cpu_limit_enforced(self, enforcer):
        """Test CPU limit enforcement."""
        manifest = ArtifactManifest(max_cpu_percent=50.0)
        enforcer.compute_permissions(manifest, "artifact-1")

        # Within limit
        within, _ = enforcer.check_resource_usage(cpu_percent=40)
        assert within is True

        # Exceeds limit
        exceeds, violation = enforcer.check_resource_usage(cpu_percent=60)
        assert exceeds is False
        assert violation.violation_type == PermissionViolationType.CPU_EXCEEDED

    def test_execution_time_enforced(self, enforcer):
        """Test execution time limit enforcement."""
        manifest = ArtifactManifest(max_execution_time_seconds=300)
        enforcer.compute_permissions(manifest, "artifact-1")

        # Within limit
        within, _ = enforcer.check_resource_usage(execution_time_seconds=200)
        assert within is True

        # Exceeds limit
        exceeds, violation = enforcer.check_resource_usage(execution_time_seconds=400)
        assert exceeds is False
        assert violation.violation_type == PermissionViolationType.EXECUTION_TIME_EXCEEDED


class TestViolationLogging:
    """Test violation logging and audit."""

    def test_violations_logged(self):
        """Test violations are logged."""
        violations_received = []

        def on_violation(v):
            violations_received.append(v)

        enforcer = SandboxPermissionsEnforcer(on_violation=on_violation)
        manifest = ArtifactManifest(network_allowed=False)
        enforcer.compute_permissions(manifest, "artifact-1")

        enforcer.can_access_network("example.com")

        assert len(violations_received) == 1

    def test_get_violations_history(self):
        """Test getting violation history."""
        enforcer = SandboxPermissionsEnforcer()
        manifest = ArtifactManifest(network_allowed=False)
        enforcer.compute_permissions(manifest, "artifact-1")

        enforcer.can_access_network("host1.com")
        enforcer.can_access_network("host2.com")

        violations = enforcer.get_violations()

        assert len(violations) >= 2

    def test_get_status(self):
        """Test getting enforcer status."""
        enforcer = SandboxPermissionsEnforcer()
        manifest = ArtifactManifest(max_memory_mb=512)
        enforcer.compute_permissions(manifest, "artifact-1")

        status = enforcer.get_status()

        assert status["has_permissions"] is True
        assert status["artifact_id"] == "artifact-1"
        assert status["current_permissions"]["max_memory_mb"] == 512
