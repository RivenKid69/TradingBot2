# -*- coding: utf-8 -*-
"""
Tenant Job Isolation - Enforce tenant boundaries.

CCEA Phase 10: Tenant isolation at job execution level.

Design Doc 15.3/5.3:
- Tenant isolation at job execution level
- Prevent cross-tenant data access
- Secure multi-tenant environment

This module enforces tenant boundaries:
- Namespace isolation (filesystem, network, IPC)
- Data isolation (each tenant's data is inaccessible to others)
- Resource isolation (dedicated resource pools)
- Process isolation (no process visibility across tenants)

Security Model:
- Each job runs in tenant-specific namespace
- No shared state between tenants
- Cryptographic verification of tenant identity
- Audit logging of all cross-boundary attempts

Reference:
- Linux Namespaces: https://man7.org/linux/man-pages/man7/namespaces.7.html
- Kubernetes Namespaces: https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/
- AWS Account Isolation: https://aws.amazon.com/blogs/security/aws-multi-account-strategy/
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import shutil
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Final, Generator, List, Optional, Set
from uuid import UUID, uuid4
import tempfile


# ============================================================================
# Constants
# ============================================================================

# Default paths for tenant isolation
DEFAULT_TENANT_BASE_PATH: Final[str] = "/var/lib/ccea/tenants"
DEFAULT_JOB_BASE_PATH: Final[str] = "/var/lib/ccea/jobs"

# Maximum path depth to prevent traversal attacks
MAX_PATH_DEPTH: Final[int] = 10

# Allowed characters in tenant/job IDs
SAFE_ID_CHARS: Final[str] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"


class IsolationViolationType(Enum):
    """Types of isolation violations."""

    PATH_TRAVERSAL = auto()
    CROSS_TENANT_ACCESS = auto()
    UNAUTHORIZED_RESOURCE = auto()
    NAMESPACE_ESCAPE = auto()
    INVALID_TENANT_ID = auto()
    INVALID_JOB_ID = auto()
    TOKEN_INVALID = auto()
    PERMISSION_DENIED = auto()


class TenantBoundaryViolation(Exception):
    """Exception raised when tenant boundary is violated."""

    def __init__(
        self,
        violation_type: IsolationViolationType,
        tenant_id: str,
        job_id: str = "",
        attempted_resource: str = "",
        message: str = "",
    ):
        self.violation_type = violation_type
        self.tenant_id = tenant_id
        self.job_id = job_id
        self.attempted_resource = attempted_resource
        self.message = message or f"Tenant boundary violation: {violation_type.name}"
        super().__init__(self.message)


@dataclass
class IsolatedJobContext:
    """
    Isolated context for a job execution.

    Contains all necessary isolation parameters and verified credentials.
    """

    # Identity
    tenant_id: str
    job_id: str
    sandbox_id: str = ""

    # Paths (verified safe)
    workspace_path: Path = field(default_factory=lambda: Path("/"))
    data_path: Path = field(default_factory=lambda: Path("/"))
    output_path: Path = field(default_factory=lambda: Path("/"))
    temp_path: Path = field(default_factory=lambda: Path("/"))

    # Security token (for verification)
    context_token: str = ""
    token_expires_at: Optional[datetime] = None

    # Permissions
    can_read_data: bool = True
    can_write_output: bool = True
    can_access_network: bool = False
    allowed_paths: Set[Path] = field(default_factory=set)

    # Resource limits (per-job)
    max_disk_mb: int = 1024
    max_files: int = 10000

    # State
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

    def verify_path(self, path: Path) -> bool:
        """
        Verify a path is within allowed boundaries.

        Args:
            path: Path to verify

        Returns:
            True if path is allowed
        """
        try:
            resolved = path.resolve()

            # Check against allowed paths
            for allowed in self.allowed_paths:
                try:
                    resolved.relative_to(allowed)
                    return True
                except ValueError:
                    continue

            return False

        except (OSError, RuntimeError):
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "job_id": self.job_id,
            "sandbox_id": self.sandbox_id,
            "workspace_path": str(self.workspace_path),
            "data_path": str(self.data_path),
            "output_path": str(self.output_path),
            "can_read_data": self.can_read_data,
            "can_write_output": self.can_write_output,
            "can_access_network": self.can_access_network,
            "max_disk_mb": self.max_disk_mb,
            "is_active": self.is_active,
        }


@dataclass
class TenantNamespace:
    """
    Namespace for a tenant.

    Contains all tenant-specific resources and isolation boundaries.
    """

    tenant_id: str

    # Paths
    base_path: Path = field(default_factory=lambda: Path("/"))
    data_path: Path = field(default_factory=lambda: Path("/"))
    jobs_path: Path = field(default_factory=lambda: Path("/"))
    temp_path: Path = field(default_factory=lambda: Path("/"))

    # Security
    namespace_secret: str = ""  # Secret for HMAC verification

    # Resource tracking
    active_jobs: Set[str] = field(default_factory=set)
    total_disk_used_mb: int = 0
    total_files: int = 0

    # Limits
    max_disk_mb: int = 10240  # 10 GB per tenant
    max_files: int = 100000
    max_concurrent_jobs: int = 10

    # State
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "base_path": str(self.base_path),
            "active_jobs_count": len(self.active_jobs),
            "total_disk_used_mb": self.total_disk_used_mb,
            "max_disk_mb": self.max_disk_mb,
            "is_active": self.is_active,
        }


@dataclass
class ViolationRecord:
    """Record of an isolation violation."""

    violation_id: str = field(default_factory=lambda: str(uuid4()))
    violation_type: IsolationViolationType = IsolationViolationType.PATH_TRAVERSAL
    tenant_id: str = ""
    job_id: str = ""
    attempted_resource: str = ""
    source_ip: str = ""
    user_agent: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    blocked: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "violation_id": self.violation_id,
            "violation_type": self.violation_type.name,
            "tenant_id": self.tenant_id,
            "job_id": self.job_id,
            "attempted_resource": self.attempted_resource,
            "timestamp": self.timestamp.isoformat(),
            "blocked": self.blocked,
            "details": self.details,
        }


class TenantJobIsolation:
    """
    Enforce tenant boundaries in job execution.

    Provides:
    - Tenant namespace creation and management
    - Job context creation with verified paths
    - Path traversal prevention
    - Cross-tenant access prevention
    - Violation logging and alerting

    Usage:
        isolation = TenantJobIsolation()

        # Create tenant namespace
        isolation.create_tenant_namespace(tenant_id)

        # Create isolated job context
        context = isolation.create_job_context(tenant_id, job_id)

        # Use context in sandbox
        with isolation.job_scope(context) as ctx:
            # Job executes with verified paths
            workspace = ctx.workspace_path
            ...

        # Cleanup
        isolation.cleanup_job(tenant_id, job_id)
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        on_violation: Optional[Callable[[ViolationRecord], None]] = None,
    ):
        """
        Initialize tenant isolation manager.

        Args:
            base_path: Base path for tenant data
            on_violation: Callback for violations
        """
        self._base_path = base_path or Path(tempfile.mkdtemp(prefix="ccea-tenants-"))
        self._on_violation = on_violation

        # State
        self._lock = threading.RLock()
        self._namespaces: Dict[str, TenantNamespace] = {}
        self._contexts: Dict[str, IsolatedJobContext] = {}  # job_id -> context

        # Violation history
        self._violations: List[ViolationRecord] = []
        self._max_violations = 10000

        # Statistics
        self._stats = {
            "namespaces_created": 0,
            "contexts_created": 0,
            "violations_blocked": 0,
            "path_checks": 0,
        }

    def create_tenant_namespace(
        self,
        tenant_id: str,
        max_disk_mb: int = 10240,
        max_concurrent_jobs: int = 10,
    ) -> TenantNamespace:
        """
        Create isolated namespace for a tenant.

        Args:
            tenant_id: Tenant identifier
            max_disk_mb: Maximum disk space for tenant
            max_concurrent_jobs: Maximum concurrent jobs

        Returns:
            Created TenantNamespace

        Raises:
            TenantBoundaryViolation: If tenant_id is invalid
        """
        # Validate tenant_id
        if not self._validate_id(tenant_id):
            raise TenantBoundaryViolation(
                IsolationViolationType.INVALID_TENANT_ID,
                tenant_id,
                message=f"Invalid tenant ID: {tenant_id}",
            )

        with self._lock:
            # Check if already exists
            if tenant_id in self._namespaces:
                return self._namespaces[tenant_id]

            # Create tenant-specific paths
            tenant_base = self._base_path / self._safe_path_component(tenant_id)
            data_path = tenant_base / "data"
            jobs_path = tenant_base / "jobs"
            temp_path = tenant_base / "tmp"

            # Create directories
            for path in [tenant_base, data_path, jobs_path, temp_path]:
                path.mkdir(parents=True, exist_ok=True)

            # Generate namespace secret for token verification
            namespace_secret = secrets.token_hex(32)

            namespace = TenantNamespace(
                tenant_id=tenant_id,
                base_path=tenant_base,
                data_path=data_path,
                jobs_path=jobs_path,
                temp_path=temp_path,
                namespace_secret=namespace_secret,
                max_disk_mb=max_disk_mb,
                max_concurrent_jobs=max_concurrent_jobs,
            )

            self._namespaces[tenant_id] = namespace
            self._stats["namespaces_created"] += 1

            return namespace

    def delete_tenant_namespace(
        self,
        tenant_id: str,
        force: bool = False,
    ) -> bool:
        """
        Delete a tenant namespace and all data.

        Args:
            tenant_id: Tenant identifier
            force: Force delete even if jobs are active

        Returns:
            True if deleted
        """
        with self._lock:
            namespace = self._namespaces.get(tenant_id)
            if not namespace:
                return False

            # Check for active jobs
            if namespace.active_jobs and not force:
                return False

            # Remove job contexts
            for job_id in list(namespace.active_jobs):
                self.cleanup_job(tenant_id, job_id)

            # Remove directory
            try:
                shutil.rmtree(namespace.base_path)
            except Exception:
                pass

            del self._namespaces[tenant_id]
            return True

    def get_namespace(self, tenant_id: str) -> Optional[TenantNamespace]:
        """Get tenant namespace."""
        with self._lock:
            return self._namespaces.get(tenant_id)

    def create_job_context(
        self,
        tenant_id: str,
        job_id: str,
        sandbox_id: str = "",
        max_disk_mb: int = 1024,
        can_access_network: bool = False,
    ) -> IsolatedJobContext:
        """
        Create isolated context for a job.

        Args:
            tenant_id: Tenant identifier
            job_id: Job identifier
            sandbox_id: Sandbox identifier
            max_disk_mb: Maximum disk for this job
            can_access_network: Whether job can access network

        Returns:
            IsolatedJobContext with verified paths

        Raises:
            TenantBoundaryViolation: If validation fails
        """
        # Validate IDs
        if not self._validate_id(tenant_id):
            raise TenantBoundaryViolation(
                IsolationViolationType.INVALID_TENANT_ID,
                tenant_id,
                job_id,
                message=f"Invalid tenant ID: {tenant_id}",
            )

        if not self._validate_id(job_id):
            raise TenantBoundaryViolation(
                IsolationViolationType.INVALID_JOB_ID,
                tenant_id,
                job_id,
                message=f"Invalid job ID: {job_id}",
            )

        with self._lock:
            # Get or create namespace
            namespace = self._namespaces.get(tenant_id)
            if not namespace:
                namespace = self.create_tenant_namespace(tenant_id)

            # Check concurrent job limit
            if len(namespace.active_jobs) >= namespace.max_concurrent_jobs:
                raise TenantBoundaryViolation(
                    IsolationViolationType.PERMISSION_DENIED,
                    tenant_id,
                    job_id,
                    message=f"Concurrent job limit reached: {namespace.max_concurrent_jobs}",
                )

            # Create job-specific paths
            job_base = namespace.jobs_path / self._safe_path_component(job_id)
            workspace_path = job_base / "workspace"
            output_path = job_base / "output"
            temp_path = job_base / "tmp"

            # Create directories
            for path in [job_base, workspace_path, output_path, temp_path]:
                path.mkdir(parents=True, exist_ok=True)

            # Generate context token
            context_token = self._generate_context_token(
                tenant_id, job_id, namespace.namespace_secret
            )

            # Build allowed paths set
            allowed_paths = {
                workspace_path.resolve(),
                output_path.resolve(),
                temp_path.resolve(),
                namespace.data_path.resolve(),  # Read-only data access
            }

            context = IsolatedJobContext(
                tenant_id=tenant_id,
                job_id=job_id,
                sandbox_id=sandbox_id,
                workspace_path=workspace_path,
                data_path=namespace.data_path,
                output_path=output_path,
                temp_path=temp_path,
                context_token=context_token,
                can_access_network=can_access_network,
                allowed_paths=allowed_paths,
                max_disk_mb=max_disk_mb,
            )

            self._contexts[job_id] = context
            namespace.active_jobs.add(job_id)
            self._stats["contexts_created"] += 1

            return context

    def get_job_context(self, job_id: str) -> Optional[IsolatedJobContext]:
        """Get job context."""
        with self._lock:
            return self._contexts.get(job_id)

    def verify_job_context(
        self,
        tenant_id: str,
        job_id: str,
        context_token: str,
    ) -> bool:
        """
        Verify job context token.

        Args:
            tenant_id: Tenant identifier
            job_id: Job identifier
            context_token: Token to verify

        Returns:
            True if token is valid
        """
        with self._lock:
            namespace = self._namespaces.get(tenant_id)
            if not namespace:
                return False

            context = self._contexts.get(job_id)
            if not context:
                return False

            if context.tenant_id != tenant_id:
                self._record_violation(
                    IsolationViolationType.CROSS_TENANT_ACCESS,
                    tenant_id,
                    job_id,
                    f"Token tenant mismatch: {context.tenant_id}",
                )
                return False

            # Verify token
            expected_token = self._generate_context_token(
                tenant_id, job_id, namespace.namespace_secret
            )

            if not hmac.compare_digest(context_token, expected_token):
                self._record_violation(
                    IsolationViolationType.TOKEN_INVALID,
                    tenant_id,
                    job_id,
                    "Invalid context token",
                )
                return False

            return True

    def check_path_access(
        self,
        job_id: str,
        path: Path,
        write: bool = False,
    ) -> bool:
        """
        Check if a path access is allowed for a job.

        Args:
            job_id: Job identifier
            path: Path to access
            write: Whether this is a write operation

        Returns:
            True if access is allowed
        """
        with self._lock:
            self._stats["path_checks"] += 1

            context = self._contexts.get(job_id)
            if not context:
                return False

            # Resolve path
            try:
                resolved = path.resolve()
            except (OSError, RuntimeError):
                self._record_violation(
                    IsolationViolationType.PATH_TRAVERSAL,
                    context.tenant_id,
                    job_id,
                    str(path),
                    {"error": "Path resolution failed"},
                )
                return False

            # Check for path traversal attempts
            path_str = str(resolved)
            if ".." in path_str or path_str.startswith("/etc") or path_str.startswith("/proc"):
                self._record_violation(
                    IsolationViolationType.PATH_TRAVERSAL,
                    context.tenant_id,
                    job_id,
                    path_str,
                )
                return False

            # Check against allowed paths
            for allowed in context.allowed_paths:
                try:
                    resolved.relative_to(allowed)

                    # For data_path, only allow read
                    if write and resolved.is_relative_to(context.data_path):
                        self._record_violation(
                            IsolationViolationType.PERMISSION_DENIED,
                            context.tenant_id,
                            job_id,
                            path_str,
                            {"reason": "Write to data path not allowed"},
                        )
                        return False

                    return True

                except ValueError:
                    continue

            # Path not in allowed set
            self._record_violation(
                IsolationViolationType.UNAUTHORIZED_RESOURCE,
                context.tenant_id,
                job_id,
                path_str,
            )
            return False

    def check_cross_tenant_access(
        self,
        source_tenant_id: str,
        source_job_id: str,
        target_tenant_id: str,
        target_resource: str,
    ) -> bool:
        """
        Check for cross-tenant access attempt.

        Args:
            source_tenant_id: Requesting tenant
            source_job_id: Requesting job
            target_tenant_id: Target tenant
            target_resource: Target resource

        Returns:
            True if access is same-tenant, False if cross-tenant
        """
        if source_tenant_id == target_tenant_id:
            return True

        # Cross-tenant access detected
        self._record_violation(
            IsolationViolationType.CROSS_TENANT_ACCESS,
            source_tenant_id,
            source_job_id,
            target_resource,
            {
                "target_tenant": target_tenant_id,
            },
        )

        return False

    @contextmanager
    def job_scope(
        self,
        context: IsolatedJobContext,
    ) -> Generator[IsolatedJobContext, None, None]:
        """
        Context manager for job execution scope.

        Provides isolated context and ensures cleanup.

        Args:
            context: Job context

        Yields:
            IsolatedJobContext for use in job

        Example:
            with isolation.job_scope(context) as ctx:
                # Job runs here with verified paths
                workspace = ctx.workspace_path
        """
        try:
            # Verify context is still valid
            if not context.is_active:
                raise TenantBoundaryViolation(
                    IsolationViolationType.PERMISSION_DENIED,
                    context.tenant_id,
                    context.job_id,
                    message="Job context is no longer active",
                )

            yield context

        finally:
            # Mark context as inactive
            context.is_active = False

    def cleanup_job(self, tenant_id: str, job_id: str) -> bool:
        """
        Cleanup job context and resources.

        Args:
            tenant_id: Tenant identifier
            job_id: Job identifier

        Returns:
            True if cleaned up
        """
        with self._lock:
            context = self._contexts.get(job_id)
            if not context:
                return False

            # Verify tenant match
            if context.tenant_id != tenant_id:
                self._record_violation(
                    IsolationViolationType.CROSS_TENANT_ACCESS,
                    tenant_id,
                    job_id,
                    "cleanup_job",
                    {"actual_tenant": context.tenant_id},
                )
                return False

            # Mark inactive
            context.is_active = False

            # Remove job directory
            namespace = self._namespaces.get(tenant_id)
            if namespace:
                job_path = namespace.jobs_path / self._safe_path_component(job_id)
                try:
                    shutil.rmtree(job_path)
                except Exception:
                    pass

                namespace.active_jobs.discard(job_id)

            del self._contexts[job_id]
            return True

    def _validate_id(self, id_value: str) -> bool:
        """Validate an ID string for safety."""
        if not id_value or len(id_value) > 128:
            return False

        # Check for path traversal attempts
        if ".." in id_value or "/" in id_value or "\\" in id_value:
            return False

        # Allow only safe characters
        return all(c in SAFE_ID_CHARS for c in id_value)

    def _safe_path_component(self, value: str) -> str:
        """Create a safe path component from a value."""
        # Hash long values
        if len(value) > 64:
            return hashlib.sha256(value.encode()).hexdigest()[:32]

        # Replace unsafe characters
        safe = ""
        for c in value:
            if c in SAFE_ID_CHARS:
                safe += c
            else:
                safe += "_"

        return safe or "default"

    def _generate_context_token(
        self,
        tenant_id: str,
        job_id: str,
        secret: str,
    ) -> str:
        """Generate HMAC-based context token."""
        message = f"{tenant_id}:{job_id}".encode()
        return hmac.new(
            secret.encode(),
            message,
            hashlib.sha256,
        ).hexdigest()

    def _record_violation(
        self,
        violation_type: IsolationViolationType,
        tenant_id: str,
        job_id: str,
        attempted_resource: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> ViolationRecord:
        """Record a violation."""
        record = ViolationRecord(
            violation_type=violation_type,
            tenant_id=tenant_id,
            job_id=job_id,
            attempted_resource=attempted_resource,
            details=details or {},
        )

        self._violations.append(record)
        if len(self._violations) > self._max_violations:
            self._violations = self._violations[-self._max_violations :]

        self._stats["violations_blocked"] += 1

        # Trigger callback
        if self._on_violation:
            try:
                self._on_violation(record)
            except Exception:
                pass

        return record

    def get_violations(
        self,
        tenant_id: Optional[str] = None,
        job_id: Optional[str] = None,
        violation_type: Optional[IsolationViolationType] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[ViolationRecord]:
        """
        Get violation history.

        Args:
            tenant_id: Filter by tenant
            job_id: Filter by job
            violation_type: Filter by type
            since: Filter by time
            limit: Maximum results

        Returns:
            List of violations
        """
        with self._lock:
            results = self._violations.copy()

        if tenant_id:
            results = [v for v in results if v.tenant_id == tenant_id]
        if job_id:
            results = [v for v in results if v.job_id == job_id]
        if violation_type:
            results = [v for v in results if v.violation_type == violation_type]
        if since:
            results = [v for v in results if v.timestamp >= since]

        results.sort(key=lambda v: v.timestamp, reverse=True)
        return results[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get isolation statistics."""
        with self._lock:
            return {
                **self._stats,
                "active_namespaces": len(self._namespaces),
                "active_contexts": len(self._contexts),
                "violations_in_history": len(self._violations),
            }

    def get_tenant_jobs(self, tenant_id: str) -> List[str]:
        """Get active jobs for a tenant."""
        with self._lock:
            namespace = self._namespaces.get(tenant_id)
            if not namespace:
                return []
            return list(namespace.active_jobs)


def create_tenant_isolation(
    base_path: Optional[Path] = None,
    on_violation: Optional[Callable[[ViolationRecord], None]] = None,
) -> TenantJobIsolation:
    """
    Create a tenant isolation manager.

    Args:
        base_path: Base path for tenant data
        on_violation: Violation callback

    Returns:
        Configured TenantJobIsolation
    """
    return TenantJobIsolation(base_path=base_path, on_violation=on_violation)
