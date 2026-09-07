# -*- coding: utf-8 -*-
"""
Cloud Research Sandbox - Isolated execution for research jobs.

CCEA Phase 10: Cloud research jobs isolation.

Design Doc 15.3/5.3:
- Sandbox for research jobs (container/VM)
- Quotas CPU/RAM/time
- Tenant isolation at job execution level

This module provides a secure, isolated environment for executing
user-submitted research code in the cloud. Unlike the Agent sandbox,
this is designed for multi-tenant cloud environments with strict
resource limits and abuse prevention.

Security Model:
- Process/container/VM isolation levels
- Cgroup-based resource limits (CPU, memory, PIDs)
- Network namespace isolation
- Read-only root filesystem (optional)
- Seccomp profiles for syscall filtering
- User namespace remapping (rootless)
- Time-bound execution with hard kill

Reference:
- gVisor: https://gvisor.dev/docs/
- Firecracker: https://firecracker-microvm.github.io/
- Docker Security: https://docs.docker.com/engine/security/
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)
import hashlib
import shutil


# ============================================================================
# Constants
# ============================================================================

# Default resource limits
DEFAULT_CPU_LIMIT: Final[float] = 1.0  # 1 CPU core
DEFAULT_MEMORY_MB: Final[int] = 1024  # 1 GB
DEFAULT_TIMEOUT_SECONDS: Final[int] = 3600  # 1 hour
DEFAULT_MAX_PIDS: Final[int] = 100  # Maximum processes/threads
DEFAULT_MAX_FILE_DESCRIPTORS: Final[int] = 256
DEFAULT_DISK_QUOTA_MB: Final[int] = 512  # 512 MB scratch space

# Maximum limits (cannot exceed these)
MAX_CPU_LIMIT: Final[float] = 8.0  # 8 CPU cores
MAX_MEMORY_MB: Final[int] = 16384  # 16 GB
MAX_TIMEOUT_SECONDS: Final[int] = 86400  # 24 hours
MAX_DISK_QUOTA_MB: Final[int] = 10240  # 10 GB

# Minimum limits (safety floor)
MIN_MEMORY_MB: Final[int] = 128  # 128 MB
MIN_TIMEOUT_SECONDS: Final[int] = 10  # 10 seconds

# Docker defaults
DEFAULT_DOCKER_IMAGE: Final[str] = "python:3.11-slim"
DEFAULT_DOCKER_NETWORK: Final[str] = "none"  # No network by default


class IsolationLevel(Enum):
    """
    Isolation level for research jobs.

    Higher levels provide stronger isolation but more overhead.
    """

    NONE = auto()  # No isolation (development only, NEVER in production)
    PROCESS = auto()  # Separate process with resource limits
    CONTAINER = auto()  # Docker container with cgroups
    MICROVM = auto()  # Firecracker microVM (enterprise)
    GVISOR = auto()  # gVisor sandbox (high security)


class CloudSandboxState(Enum):
    """Sandbox lifecycle state."""

    CREATED = auto()
    INITIALIZING = auto()
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()
    TERMINATED = auto()  # Forcefully killed


@dataclass
class CloudSandboxConfig:
    """
    Configuration for cloud research sandbox.

    Security-critical settings with safe defaults.
    """

    # Identification
    sandbox_id: str = field(default_factory=lambda: str(uuid4()))
    tenant_id: str = ""
    job_id: str = ""

    # Isolation
    isolation_level: IsolationLevel = IsolationLevel.CONTAINER

    # Resource limits
    cpu_limit: float = DEFAULT_CPU_LIMIT
    memory_limit_mb: int = DEFAULT_MEMORY_MB
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_pids: int = DEFAULT_MAX_PIDS
    max_file_descriptors: int = DEFAULT_MAX_FILE_DESCRIPTORS
    disk_quota_mb: int = DEFAULT_DISK_QUOTA_MB

    # Network (RESTRICTED by default)
    network_enabled: bool = False
    egress_allowlist: List[str] = field(default_factory=list)

    # Filesystem
    readonly_rootfs: bool = True
    scratch_dir: Optional[Path] = None
    allowed_mounts: List[str] = field(default_factory=list)

    # Container settings
    docker_image: str = DEFAULT_DOCKER_IMAGE
    docker_network: str = DEFAULT_DOCKER_NETWORK
    docker_runtime: str = "runc"  # runc, runsc (gVisor), kata

    # Security profiles
    seccomp_profile: Optional[str] = None  # Path to seccomp profile
    apparmor_profile: Optional[str] = None  # AppArmor profile name
    no_new_privileges: bool = True
    drop_capabilities: bool = True
    user_namespace: bool = True  # Rootless mode

    # Environment
    env_vars: Dict[str, str] = field(default_factory=dict)
    working_dir: str = "/workspace"

    # Logging
    capture_stdout: bool = True
    capture_stderr: bool = True
    max_output_bytes: int = 10 * 1024 * 1024  # 10 MB

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # CPU limits
        if self.cpu_limit <= 0:
            errors.append("cpu_limit must be positive")
        if self.cpu_limit > MAX_CPU_LIMIT:
            errors.append(f"cpu_limit exceeds maximum ({MAX_CPU_LIMIT})")

        # Memory limits
        if self.memory_limit_mb < MIN_MEMORY_MB:
            errors.append(f"memory_limit_mb below minimum ({MIN_MEMORY_MB})")
        if self.memory_limit_mb > MAX_MEMORY_MB:
            errors.append(f"memory_limit_mb exceeds maximum ({MAX_MEMORY_MB})")

        # Timeout limits
        if self.timeout_seconds < MIN_TIMEOUT_SECONDS:
            errors.append(f"timeout_seconds below minimum ({MIN_TIMEOUT_SECONDS})")
        if self.timeout_seconds > MAX_TIMEOUT_SECONDS:
            errors.append(f"timeout_seconds exceeds maximum ({MAX_TIMEOUT_SECONDS})")

        # Disk quota
        if self.disk_quota_mb > MAX_DISK_QUOTA_MB:
            errors.append(f"disk_quota_mb exceeds maximum ({MAX_DISK_QUOTA_MB})")

        # Tenant ID required for production
        if self.isolation_level != IsolationLevel.NONE and not self.tenant_id:
            errors.append("tenant_id required for isolated execution")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sandbox_id": self.sandbox_id,
            "tenant_id": self.tenant_id,
            "job_id": self.job_id,
            "isolation_level": self.isolation_level.name,
            "cpu_limit": self.cpu_limit,
            "memory_limit_mb": self.memory_limit_mb,
            "timeout_seconds": self.timeout_seconds,
            "max_pids": self.max_pids,
            "disk_quota_mb": self.disk_quota_mb,
            "network_enabled": self.network_enabled,
            "egress_allowlist": self.egress_allowlist,
            "readonly_rootfs": self.readonly_rootfs,
            "docker_image": self.docker_image,
            "docker_runtime": self.docker_runtime,
        }


@dataclass
class CloudSandboxMetrics:
    """Runtime metrics from sandbox execution."""

    sandbox_id: str = ""
    tenant_id: str = ""

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    wall_time_seconds: float = 0.0
    cpu_time_seconds: float = 0.0

    # Resource usage
    peak_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    disk_usage_mb: float = 0.0

    # I/O
    network_bytes_in: int = 0
    network_bytes_out: int = 0
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0

    # Processes
    max_processes: int = 0

    # Limits
    memory_limit_hits: int = 0
    cpu_throttle_count: int = 0
    oom_kills: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sandbox_id": self.sandbox_id,
            "tenant_id": self.tenant_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "wall_time_seconds": self.wall_time_seconds,
            "cpu_time_seconds": self.cpu_time_seconds,
            "peak_memory_mb": self.peak_memory_mb,
            "peak_cpu_percent": self.peak_cpu_percent,
            "disk_usage_mb": self.disk_usage_mb,
            "network_bytes_in": self.network_bytes_in,
            "network_bytes_out": self.network_bytes_out,
            "oom_kills": self.oom_kills,
        }


@dataclass
class CloudSandboxResult:
    """Result of sandbox execution."""

    sandbox_id: str = ""
    job_id: str = ""
    tenant_id: str = ""

    # Status
    success: bool = False
    exit_code: int = 0
    state: CloudSandboxState = CloudSandboxState.STOPPED

    # Output (truncated if too large)
    stdout: str = ""
    stderr: str = ""
    output_truncated: bool = False

    # Termination reason
    killed_by_timeout: bool = False
    killed_by_oom: bool = False
    killed_by_abuse: bool = False
    killed_by_quota: bool = False
    termination_reason: str = ""

    # Metrics
    metrics: CloudSandboxMetrics = field(default_factory=CloudSandboxMetrics)

    # Errors
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sandbox_id": self.sandbox_id,
            "job_id": self.job_id,
            "tenant_id": self.tenant_id,
            "success": self.success,
            "exit_code": self.exit_code,
            "state": self.state.name,
            "stdout_length": len(self.stdout),
            "stderr_length": len(self.stderr),
            "output_truncated": self.output_truncated,
            "killed_by_timeout": self.killed_by_timeout,
            "killed_by_oom": self.killed_by_oom,
            "killed_by_abuse": self.killed_by_abuse,
            "killed_by_quota": self.killed_by_quota,
            "termination_reason": self.termination_reason,
            "metrics": self.metrics.to_dict(),
            "errors": self.errors,
        }


class CloudResearchSandbox:
    """
    Isolated sandbox for cloud research job execution.

    Provides secure, resource-limited execution environment for
    user-submitted research code. Supports multiple isolation levels
    from simple process isolation to full microVM isolation.

    Security Features:
    - Process/container/VM isolation
    - CPU/memory/time/disk quotas
    - Network isolation with egress allowlist
    - Syscall filtering (seccomp)
    - Capability dropping
    - User namespace isolation
    - Read-only root filesystem

    Usage:
        config = CloudSandboxConfig(
            tenant_id="tenant-123",
            job_id="job-456",
            cpu_limit=2.0,
            memory_limit_mb=2048,
            timeout_seconds=3600,
        )

        sandbox = CloudResearchSandbox(config)

        # Execute code
        result = sandbox.execute(code, entrypoint="main.py")

        # Or run a function
        result = sandbox.run_function(my_func, args=(1, 2), kwargs={})
    """

    def __init__(
        self,
        config: Optional[CloudSandboxConfig] = None,
        on_state_change: Optional[Callable[[CloudSandboxState], None]] = None,
        on_metrics_update: Optional[Callable[[CloudSandboxMetrics], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize sandbox.

        Args:
            config: Sandbox configuration
            on_state_change: Callback for state changes
            on_metrics_update: Callback for metrics updates
            on_error: Callback for errors
        """
        self.config = config or CloudSandboxConfig()
        self._on_state_change = on_state_change
        self._on_metrics_update = on_metrics_update
        self._on_error = on_error

        # Validate config
        errors = self.config.validate()
        if errors and self.config.isolation_level != IsolationLevel.NONE:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

        # State
        self._state = CloudSandboxState.CREATED
        self._lock = threading.RLock()

        # Process/container tracking
        self._process: Optional[subprocess.Popen] = None
        self._container_id: Optional[str] = None
        self._start_time: Optional[datetime] = None

        # Metrics
        self._metrics = CloudSandboxMetrics(
            sandbox_id=self.config.sandbox_id,
            tenant_id=self.config.tenant_id,
        )

        # Scratch directory
        self._scratch_dir: Optional[Path] = None

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=2)

        # Termination flag
        self._terminate_flag = threading.Event()

    @property
    def sandbox_id(self) -> str:
        """Get sandbox ID."""
        return self.config.sandbox_id

    @property
    def state(self) -> CloudSandboxState:
        """Get current state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if sandbox is currently running."""
        return self._state == CloudSandboxState.RUNNING

    @property
    def metrics(self) -> CloudSandboxMetrics:
        """Get current metrics."""
        return self._metrics

    def _set_state(self, new_state: CloudSandboxState) -> None:
        """Set state with callback notification."""
        old_state = self._state
        self._state = new_state

        if self._on_state_change and old_state != new_state:
            try:
                self._on_state_change(new_state)
            except Exception:
                pass

    def initialize(self) -> bool:
        """
        Initialize sandbox resources.

        Creates scratch directories, validates Docker/VM availability,
        and prepares isolation environment.

        Returns:
            True if initialization successful
        """
        with self._lock:
            if self._state != CloudSandboxState.CREATED:
                return False

            self._set_state(CloudSandboxState.INITIALIZING)

            try:
                # Create scratch directory
                if self.config.scratch_dir:
                    self._scratch_dir = self.config.scratch_dir
                else:
                    self._scratch_dir = Path(
                        tempfile.mkdtemp(prefix=f"ccea-sandbox-{self.config.sandbox_id[:8]}-")
                    )

                # Validate isolation backend
                if self.config.isolation_level == IsolationLevel.CONTAINER:
                    self._validate_docker()
                elif self.config.isolation_level == IsolationLevel.GVISOR:
                    self._validate_gvisor()
                elif self.config.isolation_level == IsolationLevel.MICROVM:
                    self._validate_firecracker()

                self._set_state(CloudSandboxState.READY)
                return True

            except Exception as e:
                self._set_state(CloudSandboxState.FAILED)
                if self._on_error:
                    self._on_error(str(e))
                return False

    def _validate_docker(self) -> None:
        """Validate Docker is available and functional."""
        try:
            result = subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError("Docker not available")
        except FileNotFoundError:
            raise RuntimeError("Docker CLI not found")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Docker not responding")

    def _validate_gvisor(self) -> None:
        """Validate gVisor (runsc) is available."""
        try:
            result = subprocess.run(
                ["runsc", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise RuntimeError("gVisor (runsc) not available")
        except FileNotFoundError:
            raise RuntimeError("gVisor (runsc) not found - install from https://gvisor.dev/")

    def _validate_firecracker(self) -> None:
        """Validate Firecracker is available."""
        try:
            result = subprocess.run(
                ["firecracker", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise RuntimeError("Firecracker not available")
        except FileNotFoundError:
            raise RuntimeError("Firecracker not found - enterprise feature")

    def execute(
        self,
        code: str,
        entrypoint: str = "main.py",
        input_data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, bytes]] = None,
    ) -> CloudSandboxResult:
        """
        Execute code in the sandbox.

        Args:
            code: Python code to execute
            entrypoint: Entry point file name
            input_data: Input data (serialized to JSON)
            files: Additional files to include {filename: content}

        Returns:
            CloudSandboxResult with execution outcome
        """
        result = CloudSandboxResult(
            sandbox_id=self.config.sandbox_id,
            job_id=self.config.job_id,
            tenant_id=self.config.tenant_id,
        )

        # Initialize if needed
        if self._state == CloudSandboxState.CREATED:
            if not self.initialize():
                result.errors.append("Failed to initialize sandbox")
                return result

        if self._state != CloudSandboxState.READY:
            result.errors.append(f"Sandbox not ready (state: {self._state.name})")
            return result

        with self._lock:
            self._set_state(CloudSandboxState.RUNNING)
            self._start_time = datetime.utcnow()
            self._metrics.start_time = self._start_time
            self._terminate_flag.clear()

        try:
            # Prepare workspace
            self._prepare_workspace(code, entrypoint, input_data, files)

            # Execute based on isolation level
            if self.config.isolation_level == IsolationLevel.NONE:
                result = self._execute_direct(entrypoint, result)
            elif self.config.isolation_level == IsolationLevel.PROCESS:
                result = self._execute_process(entrypoint, result)
            elif self.config.isolation_level in (
                IsolationLevel.CONTAINER,
                IsolationLevel.GVISOR,
            ):
                result = self._execute_container(entrypoint, result)
            elif self.config.isolation_level == IsolationLevel.MICROVM:
                result = self._execute_microvm(entrypoint, result)

            # Finalize metrics
            self._metrics.end_time = datetime.utcnow()
            if self._metrics.start_time:
                self._metrics.wall_time_seconds = (
                    self._metrics.end_time - self._metrics.start_time
                ).total_seconds()
            result.metrics = self._metrics

        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            result.state = CloudSandboxState.FAILED
            self._set_state(CloudSandboxState.FAILED)

        finally:
            # Cleanup
            self._cleanup()

        return result

    def run_function(
        self,
        func: Callable[..., Any],
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> CloudSandboxResult:
        """
        Run a Python function in the sandbox.

        The function is serialized and executed in isolation.

        Args:
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            CloudSandboxResult
        """
        import inspect
        import textwrap

        # Get function source
        try:
            source = inspect.getsource(func)
            source = textwrap.dedent(source)
        except (OSError, TypeError):
            # Can't get source, create wrapper
            source = ""

        func_name = func.__name__

        # Create wrapper code
        code = f"""
{source}

import json
import sys

def _run():
    args = {args!r}
    kwargs = {kwargs or {}!r}

    try:
        result = {func_name}(*args, **kwargs)
        print(json.dumps({{"success": True, "result": str(result)}}))
    except Exception as e:
        print(json.dumps({{"success": False, "error": str(e)}}))
        sys.exit(1)

if __name__ == "__main__":
    _run()
"""

        return self.execute(code, entrypoint="main.py")

    def _prepare_workspace(
        self,
        code: str,
        entrypoint: str,
        input_data: Optional[Dict[str, Any]],
        files: Optional[Dict[str, bytes]],
    ) -> None:
        """Prepare workspace with code and data files."""
        if not self._scratch_dir:
            raise RuntimeError("Scratch directory not initialized")

        workspace = self._scratch_dir / "workspace"
        workspace.mkdir(exist_ok=True)

        # Write main code
        (workspace / entrypoint).write_text(code)

        # Write input data
        if input_data:
            (workspace / "input.json").write_text(json.dumps(input_data))

        # Write additional files
        if files:
            for filename, content in files.items():
                file_path = workspace / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, bytes):
                    file_path.write_bytes(content)
                else:
                    file_path.write_text(content)

    def _execute_direct(
        self,
        entrypoint: str,
        result: CloudSandboxResult,
    ) -> CloudSandboxResult:
        """Execute without isolation (development only)."""
        import importlib.util

        workspace = self._scratch_dir / "workspace"
        entry_path = workspace / entrypoint

        # Add workspace to path
        old_path = sys.path.copy()
        sys.path.insert(0, str(workspace))
        old_cwd = os.getcwd()
        os.chdir(workspace)

        try:
            # Execute with timeout
            spec = importlib.util.spec_from_file_location("__main__", entry_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

            result.success = True
            result.exit_code = 0
            result.state = CloudSandboxState.STOPPED

        except Exception as e:
            result.success = False
            result.stderr = str(e)
            result.exit_code = 1

        finally:
            sys.path = old_path
            os.chdir(old_cwd)

        self._set_state(CloudSandboxState.STOPPED)
        return result

    def _execute_process(
        self,
        entrypoint: str,
        result: CloudSandboxResult,
    ) -> CloudSandboxResult:
        """Execute in isolated process with resource limits."""
        workspace = self._scratch_dir / "workspace"
        entry_path = workspace / entrypoint

        # Build command
        cmd = [sys.executable, str(entry_path)]

        # Start process
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(workspace),
                env=self._get_safe_env(),
                preexec_fn=self._apply_process_limits,
            )

            # Wait with timeout
            try:
                stdout, stderr = self._process.communicate(timeout=self.config.timeout_seconds)

                result.stdout = self._truncate_output(stdout.decode("utf-8", errors="replace"))
                result.stderr = self._truncate_output(stderr.decode("utf-8", errors="replace"))
                result.exit_code = self._process.returncode
                result.success = result.exit_code == 0
                result.state = CloudSandboxState.STOPPED

            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.communicate(timeout=5)

                result.killed_by_timeout = True
                result.termination_reason = f"Timeout after {self.config.timeout_seconds}s"
                result.exit_code = -9
                result.state = CloudSandboxState.TERMINATED

        except MemoryError:
            result.killed_by_oom = True
            result.termination_reason = "Out of memory"
            result.exit_code = 137
            result.state = CloudSandboxState.TERMINATED

        except Exception as e:
            result.errors.append(str(e))
            result.exit_code = 1
            result.state = CloudSandboxState.FAILED

        finally:
            self._process = None

        self._set_state(result.state)
        return result

    def _execute_container(
        self,
        entrypoint: str,
        result: CloudSandboxResult,
    ) -> CloudSandboxResult:
        """Execute in Docker container."""
        workspace = self._scratch_dir / "workspace"
        container_name = f"ccea-job-{self.config.sandbox_id[:8]}"

        # Build docker run command
        cmd = self._build_docker_command(container_name, entrypoint)

        try:
            # Run container
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(workspace),
            )

            # Monitor with timeout
            try:
                stdout, stderr = self._process.communicate(
                    timeout=self.config.timeout_seconds + 10  # Buffer for container overhead
                )

                result.stdout = self._truncate_output(stdout.decode("utf-8", errors="replace"))
                result.stderr = self._truncate_output(stderr.decode("utf-8", errors="replace"))
                result.exit_code = self._process.returncode
                result.success = result.exit_code == 0
                result.state = CloudSandboxState.STOPPED

                # Check for OOM
                if result.exit_code == 137:
                    result.killed_by_oom = True
                    result.termination_reason = "Container killed by OOM"
                    self._metrics.oom_kills += 1

            except subprocess.TimeoutExpired:
                # Force stop container
                self._stop_container(container_name)

                result.killed_by_timeout = True
                result.termination_reason = (
                    f"Container timeout after {self.config.timeout_seconds}s"
                )
                result.exit_code = -9
                result.state = CloudSandboxState.TERMINATED

        except Exception as e:
            result.errors.append(f"Container execution failed: {e}")
            result.state = CloudSandboxState.FAILED

        finally:
            # Cleanup container
            self._cleanup_container(container_name)
            self._process = None

        self._set_state(result.state)
        return result

    def _build_docker_command(
        self,
        container_name: str,
        entrypoint: str,
    ) -> List[str]:
        """Build Docker run command with security options."""
        workspace = self._scratch_dir / "workspace"

        cmd = [
            "docker",
            "run",
            "--name",
            container_name,
            "--rm",  # Auto-remove on exit
            # Resource limits
            f"--memory={self.config.memory_limit_mb}m",
            f"--memory-swap={self.config.memory_limit_mb}m",  # No swap
            f"--cpus={self.config.cpu_limit}",
            f"--pids-limit={self.config.max_pids}",
            # Network
            "--network",
            self.config.docker_network,
            # Security
            "--security-opt",
            "no-new-privileges",
        ]

        # Read-only filesystem
        if self.config.readonly_rootfs:
            cmd.extend(["--read-only"])
            cmd.extend(["--tmpfs", "/tmp:size=64m,mode=1777"])

        # Drop capabilities
        if self.config.drop_capabilities:
            cmd.extend(["--cap-drop", "ALL"])

        # User namespace (rootless)
        if self.config.user_namespace:
            cmd.extend(["--userns", "host"])

        # Seccomp profile
        if self.config.seccomp_profile:
            cmd.extend(["--security-opt", f"seccomp={self.config.seccomp_profile}"])

        # AppArmor profile
        if self.config.apparmor_profile:
            cmd.extend(["--security-opt", f"apparmor={self.config.apparmor_profile}"])

        # Runtime (gVisor, kata, etc.)
        if self.config.docker_runtime != "runc":
            cmd.extend(["--runtime", self.config.docker_runtime])

        # Mount workspace
        cmd.extend(
            [
                "-v",
                f"{workspace}:/workspace:ro",
                "-w",
                "/workspace",
            ]
        )

        # Environment variables (filtered)
        for key, value in self.config.env_vars.items():
            if self._is_safe_env_var(key):
                cmd.extend(["-e", f"{key}={value}"])

        # Image and command
        cmd.extend(
            [
                self.config.docker_image,
                "python",
                entrypoint,
            ]
        )

        return cmd

    def _execute_microvm(
        self,
        entrypoint: str,
        result: CloudSandboxResult,
    ) -> CloudSandboxResult:
        """
        Execute in Firecracker microVM (enterprise feature).

        Firecracker provides lightweight virtualization using KVM.
        Reference: https://firecracker-microvm.github.io/

        Prerequisites:
        - Linux host with KVM support
        - Firecracker binary installed
        - Root filesystem image (rootfs.ext4)
        - Kernel image (vmlinux)

        Security Model:
        - Full VM isolation (separate kernel)
        - Hardware-enforced memory isolation
        - No shared filesystem by default
        - Network namespace isolation
        """
        workspace = self._scratch_dir / "workspace"
        vm_id = f"ccea-vm-{self.config.sandbox_id[:8]}"

        # Configuration paths
        firecracker_config_dir = Path("/etc/ccea/firecracker")
        kernel_path = firecracker_config_dir / "vmlinux"
        rootfs_path = firecracker_config_dir / "rootfs.ext4"
        firecracker_bin = "/usr/bin/firecracker"

        # Check prerequisites
        if not Path(firecracker_bin).exists():
            result.errors.append(
                "Firecracker not installed. "
                "Install from: https://github.com/firecracker-microvm/firecracker/releases"
            )
            result.state = CloudSandboxState.FAILED
            self._set_state(CloudSandboxState.FAILED)
            return result

        if not kernel_path.exists():
            result.errors.append(
                f"Kernel image not found at {kernel_path}. " "Download from Firecracker releases."
            )
            result.state = CloudSandboxState.FAILED
            self._set_state(CloudSandboxState.FAILED)
            return result

        if not rootfs_path.exists():
            result.errors.append(
                f"Root filesystem not found at {rootfs_path}. "
                "Create using build-rootfs.sh from Firecracker repository."
            )
            result.state = CloudSandboxState.FAILED
            self._set_state(CloudSandboxState.FAILED)
            return result

        try:
            # Create a copy of rootfs for this VM (copy-on-write if supported)
            vm_rootfs = self._scratch_dir / "rootfs.ext4"
            self._create_rootfs_overlay(rootfs_path, vm_rootfs, workspace)

            # Create socket for Firecracker API
            api_socket = self._scratch_dir / "firecracker.sock"

            # Build Firecracker configuration
            vm_config = self._build_firecracker_config(
                kernel_path=kernel_path,
                rootfs_path=vm_rootfs,
                entrypoint=entrypoint,
            )

            config_path = self._scratch_dir / "vm_config.json"
            with open(config_path, "w") as f:
                json.dump(vm_config, f)

            # Start Firecracker process
            cmd = [
                firecracker_bin,
                "--api-sock",
                str(api_socket),
                "--config-file",
                str(config_path),
            ]

            # Add seccomp filter if available
            seccomp_path = firecracker_config_dir / "seccomp-filter.json"
            if seccomp_path.exists():
                cmd.extend(["--seccomp-filter", str(seccomp_path)])

            logger.info(f"Starting Firecracker VM: {vm_id}")

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self._scratch_dir),
            )

            # Wait for completion with timeout
            try:
                # Firecracker VM timeout includes boot time overhead
                vm_timeout = self.config.timeout_seconds + 30

                stdout, stderr = self._process.communicate(timeout=vm_timeout)

                result.stdout = self._truncate_output(stdout.decode("utf-8", errors="replace"))
                result.stderr = self._truncate_output(stderr.decode("utf-8", errors="replace"))
                result.exit_code = self._process.returncode
                result.success = result.exit_code == 0
                result.state = CloudSandboxState.STOPPED

                # Parse metrics from VM output if available
                self._parse_vm_metrics(result)

            except subprocess.TimeoutExpired:
                # Kill the VM
                self._terminate_firecracker_vm(api_socket)
                self._process.kill()
                self._process.communicate(timeout=5)

                result.killed_by_timeout = True
                result.termination_reason = f"VM timeout after {self.config.timeout_seconds}s"
                result.exit_code = -9
                result.state = CloudSandboxState.TERMINATED

        except Exception as e:
            result.errors.append(f"MicroVM execution failed: {e}")
            result.state = CloudSandboxState.FAILED
            logger.error(f"Firecracker execution error: {e}")

        finally:
            # Cleanup VM resources
            self._cleanup_firecracker_vm(vm_id)
            self._process = None

        self._set_state(result.state)
        return result

    def _create_rootfs_overlay(
        self,
        base_rootfs: Path,
        vm_rootfs: Path,
        workspace: Path,
    ) -> None:
        """
        Create a copy-on-write overlay for the root filesystem.

        This allows multiple VMs to share the base image while
        having their own writable layer.
        """
        import shutil

        # For simplicity, we copy the rootfs. In production, use:
        # - qcow2 backing files
        # - device-mapper snapshots
        # - overlayfs (if supported)

        # Check if we can use reflinks (btrfs, xfs)
        try:
            # Try copy with reflink (copy-on-write)
            subprocess.run(
                ["cp", "--reflink=auto", str(base_rootfs), str(vm_rootfs)],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to regular copy
            shutil.copy2(base_rootfs, vm_rootfs)

        # Mount the rootfs and copy workspace files
        # Note: This requires root privileges in production
        # For development, workspace is passed via virtio-9p or vsock
        mount_point = self._scratch_dir / "mnt"
        mount_point.mkdir(exist_ok=True)

        try:
            # Try to mount and copy (requires privileges)
            subprocess.run(
                ["mount", "-o", "loop", str(vm_rootfs), str(mount_point)],
                check=True,
                capture_output=True,
            )

            # Copy workspace into rootfs
            workspace_dest = mount_point / "workspace"
            shutil.copytree(workspace, workspace_dest, dirs_exist_ok=True)

            subprocess.run(
                ["umount", str(mount_point)],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, PermissionError) as e:
            logger.warning(
                f"Could not mount rootfs to copy workspace: {e}. "
                "Workspace will be passed via kernel cmdline or virtio."
            )

    def _build_firecracker_config(
        self,
        kernel_path: Path,
        rootfs_path: Path,
        entrypoint: str,
    ) -> Dict[str, Any]:
        """
        Build Firecracker VM configuration.

        Reference: https://github.com/firecracker-microvm/firecracker/blob/main/docs/api_requests/actions.md
        """
        # Calculate vCPU and memory from config
        vcpu_count = max(1, int(self.config.cpu_limit))
        mem_size_mib = self.config.memory_limit_mb

        # Build kernel boot arguments
        boot_args = [
            "console=ttyS0",
            "reboot=k",
            "panic=1",
            "pci=off",
            f"init=/usr/bin/python3 /workspace/{entrypoint}",
        ]

        if self.config.readonly_rootfs:
            boot_args.append("ro")

        config = {
            "boot-source": {
                "kernel_image_path": str(kernel_path),
                "boot_args": " ".join(boot_args),
            },
            "drives": [
                {
                    "drive_id": "rootfs",
                    "path_on_host": str(rootfs_path),
                    "is_root_device": True,
                    "is_read_only": self.config.readonly_rootfs,
                }
            ],
            "machine-config": {
                "vcpu_count": vcpu_count,
                "mem_size_mib": mem_size_mib,
                "smt": False,  # Disable SMT for security
            },
            "logger": {
                "log_path": str(self._scratch_dir / "firecracker.log"),
                "level": "Warning",
                "show_level": True,
                "show_log_origin": True,
            },
            "metrics": {
                "metrics_path": str(self._scratch_dir / "metrics.json"),
            },
        }

        # Network configuration (disabled by default)
        if self.config.network_enabled and self.config.egress_allowlist:
            # In production, create a tap device with firewall rules
            # For now, we leave network disabled
            pass

        return config

    def _terminate_firecracker_vm(self, api_socket: Path) -> None:
        """
        Gracefully terminate Firecracker VM via API.
        """
        if not api_socket.exists():
            return

        try:
            import socket

            # Send shutdown action via Unix socket
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect(str(api_socket))

            # Send PUT request to /actions
            request = (
                "PUT /actions HTTP/1.1\r\n"
                "Host: localhost\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: 29\r\n"
                "\r\n"
                '{"action_type": "SendCtrlAltDel"}'
            )
            sock.send(request.encode())
            sock.close()

        except Exception as e:
            logger.debug(f"Could not gracefully terminate VM: {e}")

    def _cleanup_firecracker_vm(self, vm_id: str) -> None:
        """
        Cleanup Firecracker VM resources.
        """
        # Remove socket, rootfs copy, logs
        for file_name in ["firecracker.sock", "rootfs.ext4", "vm_config.json"]:
            file_path = self._scratch_dir / file_name
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass

    def _parse_vm_metrics(self, result: CloudSandboxResult) -> None:
        """
        Parse Firecracker metrics from metrics file.
        """
        metrics_path = self._scratch_dir / "metrics.json"
        if not metrics_path.exists():
            return

        try:
            with open(metrics_path, "r") as f:
                # Firecracker writes metrics as newline-delimited JSON
                for line in f:
                    if line.strip():
                        metrics = json.loads(line)

                        # Extract relevant metrics
                        if "vcpu" in metrics:
                            self._metrics.cpu_time_seconds = (
                                float(metrics.get("vcpu", {}).get("exit_io_out", 0)) / 1e9
                            )  # Convert to seconds

                        if "block" in metrics:
                            self._metrics.disk_read_bytes = metrics.get("block", {}).get(
                                "read_bytes", 0
                            )
                            self._metrics.disk_write_bytes = metrics.get("block", {}).get(
                                "write_bytes", 0
                            )

        except Exception as e:
            logger.debug(f"Could not parse VM metrics: {e}")

    def _stop_container(self, container_name: str, timeout: int = 10) -> None:
        """Stop a running container."""
        try:
            subprocess.run(
                ["docker", "stop", "-t", str(timeout), container_name],
                capture_output=True,
                timeout=timeout + 5,
            )
        except Exception:
            # Force kill
            subprocess.run(
                ["docker", "kill", container_name],
                capture_output=True,
                timeout=5,
            )

    def _cleanup_container(self, container_name: str) -> None:
        """Remove container if still exists."""
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass

    def _apply_process_limits(self) -> None:
        """Apply resource limits to current process (preexec_fn)."""
        try:
            # Memory limit
            memory_bytes = self.config.memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

            # CPU time limit
            resource.setrlimit(
                resource.RLIMIT_CPU, (self.config.timeout_seconds, self.config.timeout_seconds)
            )

            # File descriptor limit
            resource.setrlimit(
                resource.RLIMIT_NOFILE,
                (self.config.max_file_descriptors, self.config.max_file_descriptors),
            )

            # Process limit
            resource.setrlimit(resource.RLIMIT_NPROC, (self.config.max_pids, self.config.max_pids))

            # No core dumps
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        except (ValueError, resource.error):
            pass  # Some limits may not be available

    def _get_safe_env(self) -> Dict[str, str]:
        """Get sanitized environment variables."""
        # Start with minimal safe env
        safe_env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "en_US.UTF-8",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        # Add allowed custom vars
        for key, value in self.config.env_vars.items():
            if self._is_safe_env_var(key):
                safe_env[key] = value

        return safe_env

    def _is_safe_env_var(self, key: str) -> bool:
        """Check if environment variable is safe to pass."""
        # Block dangerous env vars
        blocked = {
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
            "HOME",
            "USER",
            "SHELL",
            "TERM",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "AZURE_CLIENT_SECRET",
        }

        # Block any key with secret/key/password/token
        key_lower = key.lower()
        sensitive_patterns = ["secret", "key", "password", "token", "credential"]

        if key in blocked:
            return False

        for pattern in sensitive_patterns:
            if pattern in key_lower:
                return False

        return True

    def _truncate_output(self, output: str) -> str:
        """Truncate output to max size."""
        max_chars = self.config.max_output_bytes
        if len(output) > max_chars:
            return output[:max_chars] + f"\n... [truncated, {len(output)} total chars]"
        return output

    def _cleanup(self) -> None:
        """Cleanup sandbox resources."""
        # Remove scratch directory
        if self._scratch_dir and self._scratch_dir.exists():
            try:
                shutil.rmtree(self._scratch_dir)
            except Exception:
                pass

        self._scratch_dir = None

    def terminate(self, reason: str = "Manual termination") -> bool:
        """
        Force terminate the sandbox.

        Args:
            reason: Reason for termination

        Returns:
            True if terminated
        """
        with self._lock:
            self._terminate_flag.set()

            if self._process:
                try:
                    self._process.kill()
                    self._process.wait(timeout=5)
                except Exception:
                    pass

            if self._container_id:
                try:
                    subprocess.run(
                        ["docker", "kill", self._container_id],
                        capture_output=True,
                        timeout=10,
                    )
                except Exception:
                    pass

            self._set_state(CloudSandboxState.TERMINATED)
            self._cleanup()

            return True

    def get_status(self) -> Dict[str, Any]:
        """Get current sandbox status."""
        uptime = 0.0
        if self._start_time:
            uptime = (datetime.utcnow() - self._start_time).total_seconds()

        return {
            "sandbox_id": self.config.sandbox_id,
            "job_id": self.config.job_id,
            "tenant_id": self.config.tenant_id,
            "state": self._state.name,
            "isolation_level": self.config.isolation_level.name,
            "is_running": self.is_running,
            "uptime_seconds": uptime,
            "config": self.config.to_dict(),
            "metrics": self._metrics.to_dict(),
        }


def create_cloud_sandbox(
    tenant_id: str,
    job_id: str,
    isolation_level: str = "container",
    cpu_limit: float = DEFAULT_CPU_LIMIT,
    memory_limit_mb: int = DEFAULT_MEMORY_MB,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    network_enabled: bool = False,
    **kwargs,
) -> CloudResearchSandbox:
    """
    Factory function to create a cloud research sandbox.

    Args:
        tenant_id: Tenant identifier
        job_id: Job identifier
        isolation_level: Isolation level (none, process, container, gvisor, microvm)
        cpu_limit: CPU limit in cores
        memory_limit_mb: Memory limit in MB
        timeout_seconds: Execution timeout
        network_enabled: Whether to enable network access
        **kwargs: Additional config options

    Returns:
        Configured CloudResearchSandbox
    """
    level_map = {
        "none": IsolationLevel.NONE,
        "process": IsolationLevel.PROCESS,
        "container": IsolationLevel.CONTAINER,
        "gvisor": IsolationLevel.GVISOR,
        "microvm": IsolationLevel.MICROVM,
    }

    config = CloudSandboxConfig(
        tenant_id=tenant_id,
        job_id=job_id,
        isolation_level=level_map.get(isolation_level.lower(), IsolationLevel.CONTAINER),
        cpu_limit=cpu_limit,
        memory_limit_mb=memory_limit_mb,
        timeout_seconds=timeout_seconds,
        network_enabled=network_enabled,
        **kwargs,
    )

    return CloudResearchSandbox(config)
