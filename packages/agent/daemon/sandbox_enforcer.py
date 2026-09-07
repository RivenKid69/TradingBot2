# -*- coding: utf-8 -*-
"""
Sandbox Permissions Enforcer - Enforce manifest permissions in sandbox.

AGENT ZONE ONLY.

Extracts permissions from artifact manifest and enforces them in sandbox:
- Network egress allowlist
- Filesystem restrictions
- Resource limits (CPU, memory)
- Execution time limits

Design Doc Phase 4.4A:
- Sandbox permissions enforcement from manifest
- Manifest defines maximum permissions
- Local policy can further restrict
- Cloud cannot expand permissions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, Final, List, Optional, Set, Tuple

from packages.agent.daemon.artifact_manager import ArtifactManifest
from packages.agent.daemon.sandbox import Sandbox, SandboxConfig, SandboxType
from packages.agent.policy.layered_policy import LayeredPolicyEngine, PolicySource


# Default restrictive limits
DEFAULT_MAX_MEMORY_MB: Final[int] = 256
DEFAULT_MAX_CPU_PERCENT: Final[float] = 50.0
DEFAULT_MAX_EXECUTION_TIME: Final[int] = 300  # 5 minutes


class PermissionViolationType(Enum):
    """Types of permission violations."""

    NETWORK_BLOCKED = auto()
    NETWORK_EGRESS_DENIED = auto()
    FILESYSTEM_WRITE_DENIED = auto()
    FILESYSTEM_PATH_DENIED = auto()
    MEMORY_EXCEEDED = auto()
    CPU_EXCEEDED = auto()
    EXECUTION_TIME_EXCEEDED = auto()
    RESOURCE_DENIED = auto()


@dataclass
class PermissionViolation:
    """
    Record of a permission violation.
    """

    violation_id: str = ""
    violation_type: PermissionViolationType = PermissionViolationType.RESOURCE_DENIED
    artifact_id: str = ""
    strategy_id: str = ""
    details: str = ""
    attempted_resource: str = ""
    manifest_allows: bool = False
    policy_allows: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "violation_id": self.violation_id,
            "violation_type": self.violation_type.name,
            "artifact_id": self.artifact_id,
            "strategy_id": self.strategy_id,
            "details": self.details,
            "attempted_resource": self.attempted_resource,
            "manifest_allows": self.manifest_allows,
            "policy_allows": self.policy_allows,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EnforcedPermissions:
    """
    Final enforced permissions after layering.

    Combines manifest permissions with local policy.
    Local policy can RESTRICT, never EXPAND.
    """

    # Network
    network_enabled: bool = False
    egress_allowlist: List[str] = field(default_factory=list)

    # Filesystem
    filesystem_readonly: bool = True
    allowed_paths: List[str] = field(default_factory=list)

    # Resources
    max_memory_mb: int = DEFAULT_MAX_MEMORY_MB
    max_cpu_percent: float = DEFAULT_MAX_CPU_PERCENT
    max_execution_time_seconds: int = DEFAULT_MAX_EXECUTION_TIME

    # Source tracking
    network_source: PolicySource = PolicySource.DEFAULT
    memory_source: PolicySource = PolicySource.DEFAULT
    cpu_source: PolicySource = PolicySource.DEFAULT

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "network_enabled": self.network_enabled,
            "egress_allowlist": self.egress_allowlist,
            "filesystem_readonly": self.filesystem_readonly,
            "allowed_paths": self.allowed_paths,
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_percent": self.max_cpu_percent,
            "max_execution_time_seconds": self.max_execution_time_seconds,
        }

    def to_sandbox_config(self) -> SandboxConfig:
        """Convert to SandboxConfig."""
        return SandboxConfig(
            sandbox_type=SandboxType.PROCESS,
            cpu_limit=self.max_cpu_percent / 100.0,
            memory_limit_mb=self.max_memory_mb,
            timeout_seconds=self.max_execution_time_seconds,
            network_enabled=self.network_enabled,
            egress_allowlist=self.egress_allowlist,
            readonly_fs=self.filesystem_readonly,
            allowed_paths=[__import__("pathlib").Path(p) for p in self.allowed_paths],
        )


class SandboxPermissionsEnforcer:
    """
    Enforces sandbox permissions from manifest + local policy.

    SECURITY CRITICAL:
    - Manifest defines MAXIMUM permissions artifact can request
    - Local policy can RESTRICT but never EXPAND
    - Cloud cannot modify sandbox permissions
    - All violations are logged for audit

    Permission Layering:
    1. Start with DEFAULT (most restrictive)
    2. Apply MANIFEST permissions (artifact's request)
    3. Apply LOCAL policy (user's restrictions) - can only RESTRICT
    4. Never apply CLOUD policy to sandbox permissions

    Usage:
        enforcer = SandboxPermissionsEnforcer(policy_engine)

        # Load manifest permissions
        permissions = enforcer.compute_permissions(manifest)

        # Create sandbox with enforced permissions
        sandbox_config = permissions.to_sandbox_config()
        sandbox = Sandbox(sandbox_config)

        # Check specific permission
        if enforcer.can_access_network("api.exchange.com"):
            # Allow network call
    """

    def __init__(
        self,
        policy_engine: Optional[LayeredPolicyEngine] = None,
        on_violation: Optional[callable] = None,
    ):
        """
        Initialize enforcer.

        Args:
            policy_engine: Policy engine for local policy
            on_violation: Callback when violation occurs
        """
        self._policy_engine = policy_engine or LayeredPolicyEngine()
        self._on_violation = on_violation

        # Current enforced permissions
        self._current_permissions: Optional[EnforcedPermissions] = None
        self._current_manifest: Optional[ArtifactManifest] = None
        self._artifact_id: str = ""

        # Violation log
        self._violations: List[PermissionViolation] = []

    def compute_permissions(
        self,
        manifest: ArtifactManifest,
        artifact_id: str = "",
    ) -> EnforcedPermissions:
        """
        Compute enforced permissions from manifest + local policy.

        Args:
            manifest: Artifact manifest with permissions
            artifact_id: Artifact identifier

        Returns:
            EnforcedPermissions with final enforced values
        """
        self._current_manifest = manifest
        self._artifact_id = artifact_id

        permissions = EnforcedPermissions()

        # 1. Network permissions
        # Manifest can enable network if it wants, local policy can disable
        # DEFAULT source should not restrict manifest (only LOCAL/CLOUD can restrict)
        manifest_network = manifest.network_allowed
        local_network, source = self._policy_engine.get_policy("network_enabled")

        # If source is DEFAULT, don't let it restrict manifest
        # Only LOCAL or CLOUD policies should be able to restrict
        if source == PolicySource.DEFAULT:
            local_network = True  # Don't let system default restrict manifest

        if local_network is None:
            local_network = True  # No local restriction

        # Final: manifest AND local must both allow
        permissions.network_enabled = manifest_network and local_network
        permissions.network_source = PolicySource.MANIFEST if manifest_network else source

        # 2. Egress allowlist
        # Intersection of manifest and local allowlist (most restrictive)
        manifest_egress = set(manifest.egress_allowlist) if manifest.egress_allowlist else set()
        local_egress, _ = self._policy_engine.get_policy("egress_allowlist")
        local_egress = set(local_egress) if local_egress else set()

        if manifest_egress and local_egress:
            # Intersection - only hosts in both lists
            permissions.egress_allowlist = list(manifest_egress & local_egress)
        elif manifest_egress:
            permissions.egress_allowlist = list(manifest_egress)
        elif local_egress:
            permissions.egress_allowlist = list(local_egress)
        # else: empty = no egress allowed

        # 3. Filesystem
        # Manifest can request read-write, local policy can force read-only
        # DEFAULT source should not restrict manifest (only LOCAL/CLOUD can restrict)
        manifest_readonly = manifest.filesystem_readonly
        local_readonly, source = self._policy_engine.get_policy("filesystem_readonly")

        # If source is DEFAULT, don't let it restrict manifest
        if source == PolicySource.DEFAULT:
            local_readonly = False  # Don't let system default override manifest

        if local_readonly is None:
            local_readonly = False  # No local restriction

        # Final: read-only if EITHER requires it (most restrictive)
        permissions.filesystem_readonly = manifest_readonly or local_readonly

        # 4. Allowed paths
        # Intersection of manifest and local (most restrictive)
        manifest_paths = set(manifest.allowed_paths) if manifest.allowed_paths else set()
        local_paths, _ = self._policy_engine.get_policy("allowed_paths")
        local_paths = set(local_paths) if local_paths else set()

        if manifest_paths and local_paths:
            permissions.allowed_paths = list(manifest_paths & local_paths)
        elif manifest_paths:
            permissions.allowed_paths = list(manifest_paths)
        elif local_paths:
            permissions.allowed_paths = list(local_paths)

        # 5. Memory limit
        # Take MINIMUM of manifest and local (most restrictive)
        manifest_memory = manifest.max_memory_mb
        local_memory, source = self._policy_engine.get_policy("max_memory_mb")

        if local_memory is not None:
            permissions.max_memory_mb = min(manifest_memory, local_memory)
            permissions.memory_source = (
                source if local_memory < manifest_memory else PolicySource.MANIFEST
            )
        else:
            permissions.max_memory_mb = manifest_memory
            permissions.memory_source = PolicySource.MANIFEST

        # 6. CPU limit
        manifest_cpu = manifest.max_cpu_percent
        local_cpu, source = self._policy_engine.get_policy("max_cpu_percent")

        if local_cpu is not None:
            permissions.max_cpu_percent = min(manifest_cpu, local_cpu)
            permissions.cpu_source = source if local_cpu < manifest_cpu else PolicySource.MANIFEST
        else:
            permissions.max_cpu_percent = manifest_cpu

        # 7. Execution time
        manifest_time = manifest.max_execution_time_seconds
        local_time, _ = self._policy_engine.get_policy("max_execution_time")

        if local_time is not None:
            permissions.max_execution_time_seconds = min(manifest_time, local_time)
        else:
            permissions.max_execution_time_seconds = manifest_time

        self._current_permissions = permissions
        return permissions

    def can_access_network(self, host: str = "") -> Tuple[bool, Optional[PermissionViolation]]:
        """
        Check if network access is allowed.

        Args:
            host: Target host (empty for general check)

        Returns:
            Tuple of (allowed, violation if denied)
        """
        if not self._current_permissions:
            return False, self._create_violation(
                PermissionViolationType.NETWORK_BLOCKED,
                "No permissions computed",
                host,
            )

        if not self._current_permissions.network_enabled:
            return False, self._create_violation(
                PermissionViolationType.NETWORK_BLOCKED,
                "Network disabled by permissions",
                host,
            )

        if host and self._current_permissions.egress_allowlist:
            # Check if host is in allowlist
            allowed = any(
                host.endswith(allowed_host) or host == allowed_host
                for allowed_host in self._current_permissions.egress_allowlist
            )
            if not allowed:
                return False, self._create_violation(
                    PermissionViolationType.NETWORK_EGRESS_DENIED,
                    f"Host '{host}' not in egress allowlist",
                    host,
                )

        return True, None

    def can_write_file(self, path: str) -> Tuple[bool, Optional[PermissionViolation]]:
        """
        Check if file write is allowed.

        Args:
            path: Target file path

        Returns:
            Tuple of (allowed, violation if denied)
        """
        if not self._current_permissions:
            return False, self._create_violation(
                PermissionViolationType.FILESYSTEM_WRITE_DENIED,
                "No permissions computed",
                path,
            )

        if self._current_permissions.filesystem_readonly:
            return False, self._create_violation(
                PermissionViolationType.FILESYSTEM_WRITE_DENIED,
                "Filesystem is read-only",
                path,
            )

        if self._current_permissions.allowed_paths:
            # Check if path is within allowed paths
            allowed = any(
                path.startswith(allowed_path)
                for allowed_path in self._current_permissions.allowed_paths
            )
            if not allowed:
                return False, self._create_violation(
                    PermissionViolationType.FILESYSTEM_PATH_DENIED,
                    f"Path '{path}' not in allowed paths",
                    path,
                )

        return True, None

    def check_resource_usage(
        self,
        memory_mb: float = 0,
        cpu_percent: float = 0,
        execution_time_seconds: float = 0,
    ) -> Tuple[bool, Optional[PermissionViolation]]:
        """
        Check if resource usage is within limits.

        Args:
            memory_mb: Current memory usage
            cpu_percent: Current CPU usage
            execution_time_seconds: Current execution time

        Returns:
            Tuple of (within_limits, violation if exceeded)
        """
        if not self._current_permissions:
            return False, self._create_violation(
                PermissionViolationType.RESOURCE_DENIED,
                "No permissions computed",
            )

        if memory_mb > self._current_permissions.max_memory_mb:
            return False, self._create_violation(
                PermissionViolationType.MEMORY_EXCEEDED,
                f"Memory {memory_mb}MB exceeds limit {self._current_permissions.max_memory_mb}MB",
                f"memory:{memory_mb}MB",
            )

        if cpu_percent > self._current_permissions.max_cpu_percent:
            return False, self._create_violation(
                PermissionViolationType.CPU_EXCEEDED,
                f"CPU {cpu_percent}% exceeds limit {self._current_permissions.max_cpu_percent}%",
                f"cpu:{cpu_percent}%",
            )

        if execution_time_seconds > self._current_permissions.max_execution_time_seconds:
            return False, self._create_violation(
                PermissionViolationType.EXECUTION_TIME_EXCEEDED,
                f"Execution time {execution_time_seconds}s exceeds limit {self._current_permissions.max_execution_time_seconds}s",
                f"time:{execution_time_seconds}s",
            )

        return True, None

    def _create_violation(
        self,
        violation_type: PermissionViolationType,
        details: str,
        resource: str = "",
    ) -> PermissionViolation:
        """Create and log a violation."""
        from uuid import uuid4

        violation = PermissionViolation(
            violation_id=str(uuid4()),
            violation_type=violation_type,
            artifact_id=self._artifact_id,
            details=details,
            attempted_resource=resource,
            manifest_allows=self._check_manifest_allows(violation_type, resource),
            policy_allows=self._check_policy_allows(violation_type, resource),
        )

        self._violations.append(violation)

        # Callback
        if self._on_violation:
            self._on_violation(violation)

        return violation

    def _check_manifest_allows(
        self,
        violation_type: PermissionViolationType,
        resource: str,
    ) -> bool:
        """Check if manifest would allow this."""
        if not self._current_manifest:
            return False

        if violation_type == PermissionViolationType.NETWORK_BLOCKED:
            return self._current_manifest.network_allowed

        if violation_type == PermissionViolationType.NETWORK_EGRESS_DENIED:
            return resource in self._current_manifest.egress_allowlist

        if violation_type in (
            PermissionViolationType.FILESYSTEM_WRITE_DENIED,
            PermissionViolationType.FILESYSTEM_PATH_DENIED,
        ):
            return not self._current_manifest.filesystem_readonly

        return True

    def _check_policy_allows(
        self,
        violation_type: PermissionViolationType,
        resource: str,
    ) -> bool:
        """Check if local policy would allow this."""
        if violation_type == PermissionViolationType.NETWORK_BLOCKED:
            value, _ = self._policy_engine.get_policy("network_enabled")
            return value if value is not None else True

        if violation_type in (PermissionViolationType.FILESYSTEM_WRITE_DENIED,):
            value, _ = self._policy_engine.get_policy("filesystem_readonly")
            return not (value if value is not None else True)

        return True

    def get_violations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent violations."""
        return [v.to_dict() for v in self._violations[-limit:]]

    def get_current_permissions(self) -> Optional[EnforcedPermissions]:
        """Get current enforced permissions."""
        return self._current_permissions

    def create_sandbox(
        self,
        manifest: ArtifactManifest,
        artifact_id: str = "",
    ) -> Sandbox:
        """
        Create a sandbox with enforced permissions.

        Args:
            manifest: Artifact manifest
            artifact_id: Artifact identifier

        Returns:
            Configured Sandbox
        """
        permissions = self.compute_permissions(manifest, artifact_id)
        config = permissions.to_sandbox_config()
        return Sandbox(config, on_error=self._handle_sandbox_error)

    def _handle_sandbox_error(self, error: str) -> None:
        """Handle sandbox error."""
        self._create_violation(
            PermissionViolationType.RESOURCE_DENIED,
            f"Sandbox error: {error}",
        )

    def get_status(self) -> Dict[str, Any]:
        """Get enforcer status."""
        return {
            "has_permissions": self._current_permissions is not None,
            "artifact_id": self._artifact_id,
            "violation_count": len(self._violations),
            "current_permissions": (
                self._current_permissions.to_dict() if self._current_permissions else None
            ),
        }
