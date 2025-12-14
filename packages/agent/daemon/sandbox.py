# -*- coding: utf-8 -*-
"""
Strategy Sandbox - Process/container isolation.

Design Doc Phase 5:
- Sandbox/изоляция стратегии
- Базово: стратегия в отдельном процессе/контейнере + лимиты CPU/RAM
- Enterprise: ro-fs, egress allowlist, deny-by-default outbound network

CCEA Phase 5 Component.

Platform Notes:
- Linux/macOS: Full resource limits via `resource` module (RLIMIT_*)
- Windows: Limited support via psutil/Job Objects (graceful degradation)
"""

from __future__ import annotations

import json
import multiprocessing
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Tuple
from uuid import uuid4

# Platform detection
IS_WINDOWS = platform.system() == "Windows"
IS_POSIX = os.name == "posix"

# Conditional import for POSIX resource module
# On Windows, this module is not available
_resource_module = None
if IS_POSIX:
    try:
        import resource as _resource_module
    except ImportError:
        _resource_module = None


# Constants
DEFAULT_CPU_LIMIT: Final[float] = 1.0  # 1 CPU core
DEFAULT_MEMORY_MB: Final[int] = 512  # 512 MB
DEFAULT_TIMEOUT_SECONDS: Final[int] = 3600  # 1 hour


class SandboxType(Enum):
    """Types of sandbox isolation."""
    PROCESS = auto()  # Separate process with resource limits
    CONTAINER = auto()  # Docker container
    NONE = auto()  # No isolation (for development)


class SandboxState(Enum):
    """Sandbox state."""
    CREATED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()


@dataclass
class SandboxConfig:
    """
    Sandbox configuration.
    """
    # Sandbox type
    sandbox_type: SandboxType = SandboxType.PROCESS

    # Resource limits
    cpu_limit: float = DEFAULT_CPU_LIMIT  # CPU cores
    memory_limit_mb: int = DEFAULT_MEMORY_MB
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    # Network (Enterprise)
    network_enabled: bool = True
    egress_allowlist: List[str] = field(default_factory=list)  # Empty = allow all
    deny_outbound_by_default: bool = False  # Enterprise only

    # Filesystem (Enterprise)
    readonly_fs: bool = False  # Enterprise only
    allowed_paths: List[Path] = field(default_factory=list)

    # Container settings (if using Docker)
    docker_image: Optional[str] = None
    docker_network: str = "none"  # none, host, bridge
    docker_volumes: Dict[str, str] = field(default_factory=dict)

    # Process settings
    working_dir: Optional[Path] = None
    env_vars: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sandbox_type": self.sandbox_type.name,
            "cpu_limit": self.cpu_limit,
            "memory_limit_mb": self.memory_limit_mb,
            "timeout_seconds": self.timeout_seconds,
            "network_enabled": self.network_enabled,
            "readonly_fs": self.readonly_fs,
        }


@dataclass
class SandboxMetrics:
    """
    Sandbox resource metrics.
    """
    sandbox_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    uptime_seconds: float = 0.0
    restart_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sandbox_id": self.sandbox_id,
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "memory_percent": self.memory_percent,
            "uptime_seconds": self.uptime_seconds,
            "restart_count": self.restart_count,
        }


@dataclass
class SandboxResult:
    """
    Result from sandbox execution.
    """
    sandbox_id: str = ""
    success: bool = False
    exit_code: int = 0
    output: str = ""
    error: str = ""
    duration_seconds: float = 0.0
    killed_by_timeout: bool = False
    killed_by_memory: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sandbox_id": self.sandbox_id,
            "success": self.success,
            "exit_code": self.exit_code,
            "output": self.output[:1000] if self.output else "",  # Truncate
            "error": self.error[:1000] if self.error else "",
            "duration_seconds": self.duration_seconds,
            "killed_by_timeout": self.killed_by_timeout,
            "killed_by_memory": self.killed_by_memory,
        }


class Sandbox:
    """
    Strategy sandbox for isolated execution.

    Design Doc Phase 5:
    - Isolates strategy in separate process/container
    - Enforces CPU/RAM limits
    - Enterprise: read-only filesystem, egress allowlist

    Usage:
        sandbox = Sandbox(config)

        # Start sandbox
        sandbox.start()

        # Execute in sandbox
        result = sandbox.execute(strategy_fn, args)

        # Or run continuously
        sandbox.run_strategy_loop(strategy, market_data_fn)

        # Stop
        sandbox.stop()
    """

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize sandbox.

        Args:
            config: Sandbox configuration
            on_error: Error callback
        """
        self.config = config or SandboxConfig()
        self._on_error = on_error

        # State
        self._sandbox_id = str(uuid4())
        self._state = SandboxState.CREATED
        self._process: Optional[multiprocessing.Process] = None
        self._container_id: Optional[str] = None
        self._start_time: Optional[datetime] = None
        self._restart_count = 0

        # Communication
        self._input_queue: Optional[multiprocessing.Queue] = None
        self._output_queue: Optional[multiprocessing.Queue] = None
        self._stop_event: Optional[multiprocessing.Event] = None

        # Metrics
        self._metrics = SandboxMetrics(sandbox_id=self._sandbox_id)

        # Lock
        self._lock = threading.RLock()

    @property
    def sandbox_id(self) -> str:
        """Get sandbox ID."""
        return self._sandbox_id

    @property
    def state(self) -> SandboxState:
        """Get current state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if sandbox is running."""
        return self._state == SandboxState.RUNNING

    @property
    def metrics(self) -> SandboxMetrics:
        """Get current metrics."""
        return self._metrics

    def start(self) -> bool:
        """
        Start the sandbox.

        Returns:
            True if started successfully
        """
        with self._lock:
            if self._state == SandboxState.RUNNING:
                return True

            self._state = SandboxState.STARTING

            try:
                if self.config.sandbox_type == SandboxType.CONTAINER:
                    self._start_container()
                elif self.config.sandbox_type == SandboxType.PROCESS:
                    self._start_process()
                else:
                    # No isolation
                    pass

                self._state = SandboxState.RUNNING
                self._start_time = datetime.utcnow()
                return True

            except Exception as e:
                self._state = SandboxState.FAILED
                if self._on_error:
                    self._on_error(str(e))
                return False

    def stop(self, timeout: float = 10.0) -> bool:
        """
        Stop the sandbox.

        Args:
            timeout: Timeout for graceful shutdown

        Returns:
            True if stopped successfully
        """
        with self._lock:
            if self._state in (SandboxState.STOPPED, SandboxState.CREATED):
                return True

            self._state = SandboxState.STOPPING

            try:
                if self.config.sandbox_type == SandboxType.CONTAINER:
                    self._stop_container(timeout)
                elif self.config.sandbox_type == SandboxType.PROCESS:
                    self._stop_process(timeout)

                self._state = SandboxState.STOPPED
                return True

            except Exception as e:
                if self._on_error:
                    self._on_error(str(e))
                return False

    def execute(
        self,
        func: Callable[..., Any],
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> SandboxResult:
        """
        Execute a function in the sandbox.

        Args:
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            timeout: Timeout in seconds

        Returns:
            SandboxResult
        """
        if self.config.sandbox_type == SandboxType.NONE:
            # No isolation, execute directly
            return self._execute_direct(func, args, kwargs, timeout)

        timeout = timeout or self.config.timeout_seconds
        result = SandboxResult(sandbox_id=self._sandbox_id)
        start_time = time.time()

        try:
            # Create communication queues
            result_queue: multiprocessing.Queue = multiprocessing.Queue()

            # Start worker process
            process = multiprocessing.Process(
                target=self._sandbox_worker,
                args=(func, args, kwargs or {}, result_queue),
            )

            # Apply resource limits before start
            process.start()

            # Wait for result
            try:
                worker_result = result_queue.get(timeout=timeout)
                result.success = worker_result.get("success", False)
                result.output = worker_result.get("output", "")
                result.error = worker_result.get("error", "")
                result.exit_code = worker_result.get("exit_code", 0)
            except Exception:
                # Timeout or error
                result.killed_by_timeout = True
                result.error = f"Execution timed out after {timeout}s"
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()

            process.join(timeout=5)
            result.exit_code = process.exitcode or 0

        except Exception as e:
            result.error = str(e)
            result.success = False

        result.duration_seconds = time.time() - start_time
        return result

    def _execute_direct(
        self,
        func: Callable[..., Any],
        args: Tuple,
        kwargs: Optional[Dict[str, Any]],
        timeout: Optional[float],
    ) -> SandboxResult:
        """Execute without isolation."""
        result = SandboxResult(sandbox_id=self._sandbox_id)
        start_time = time.time()

        try:
            output = func(*args, **(kwargs or {}))
            result.success = True
            result.output = str(output) if output else ""
        except Exception as e:
            result.success = False
            result.error = str(e)

        result.duration_seconds = time.time() - start_time
        return result

    @staticmethod
    def _sandbox_worker(
        func: Callable[..., Any],
        args: Tuple,
        kwargs: Dict[str, Any],
        result_queue: multiprocessing.Queue,
    ) -> None:
        """Worker function that runs in sandboxed process."""
        result = {
            "success": False,
            "output": "",
            "error": "",
            "exit_code": 0,
        }

        try:
            # Apply resource limits
            Sandbox._apply_resource_limits()

            # Execute function
            output = func(*args, **kwargs)
            result["success"] = True
            result["output"] = str(output) if output else ""

        except MemoryError:
            result["error"] = "Memory limit exceeded"
            result["exit_code"] = 137
        except Exception as e:
            result["error"] = str(e)
            result["exit_code"] = 1

        try:
            result_queue.put(result)
        except Exception:
            pass

    @staticmethod
    def _apply_resource_limits(
        memory_mb: int = DEFAULT_MEMORY_MB,
        cpu_time_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """
        Apply resource limits to current process.

        Platform behavior:
        - POSIX (Linux/macOS): Uses resource module (RLIMIT_*)
        - Windows: Graceful degradation, limits via psutil if available
        """
        if IS_POSIX and _resource_module is not None:
            Sandbox._apply_posix_resource_limits(memory_mb, cpu_time_seconds)
        elif IS_WINDOWS:
            Sandbox._apply_windows_resource_limits(memory_mb, cpu_time_seconds)
        # else: No resource limits available, continue without them

    @staticmethod
    def _apply_posix_resource_limits(
        memory_mb: int = DEFAULT_MEMORY_MB,
        cpu_time_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Apply POSIX resource limits via resource module."""
        if _resource_module is None:
            return

        try:
            # Memory limit (soft and hard)
            memory_bytes = memory_mb * 1024 * 1024
            _resource_module.setrlimit(_resource_module.RLIMIT_AS, (memory_bytes, memory_bytes))

            # CPU time limit
            _resource_module.setrlimit(_resource_module.RLIMIT_CPU, (cpu_time_seconds, cpu_time_seconds))

            # Core dump disabled
            _resource_module.setrlimit(_resource_module.RLIMIT_CORE, (0, 0))

            # Limit number of processes/threads (not available on macOS)
            try:
                _resource_module.setrlimit(_resource_module.RLIMIT_NPROC, (100, 100))
            except (AttributeError, ValueError):
                pass  # RLIMIT_NPROC not available on all POSIX systems

            # Limit file descriptors
            _resource_module.setrlimit(_resource_module.RLIMIT_NOFILE, (256, 256))

        except (ValueError, OSError, AttributeError):
            pass  # Some limits may not be available on all platforms

    @staticmethod
    def _apply_windows_resource_limits(
        memory_mb: int = DEFAULT_MEMORY_MB,
        cpu_time_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """
        Apply Windows resource limits.

        On Windows, true process resource limits require Job Objects.
        This is a best-effort implementation using psutil where available.
        For production Windows isolation, consider using containers or
        Windows Job Objects directly via ctypes/pywin32.
        """
        try:
            import psutil
            # On Windows, we can't set hard limits like on POSIX,
            # but we can monitor and self-terminate if exceeded.
            # This is handled at a higher level (execute() method timeout).
            # Here we just set process priority to below-normal as a soft constraint.
            proc = psutil.Process()
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except (ImportError, AttributeError, psutil.Error):
            pass  # psutil not available or error setting priority

    def _start_process(self) -> None:
        """Start process-based sandbox."""
        self._input_queue = multiprocessing.Queue()
        self._output_queue = multiprocessing.Queue()
        self._stop_event = multiprocessing.Event()

    def _stop_process(self, timeout: float) -> None:
        """Stop process-based sandbox."""
        if self._stop_event:
            self._stop_event.set()

        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=5)

    def _start_container(self) -> None:
        """Start Docker container sandbox."""
        if not self.config.docker_image:
            raise ValueError("Docker image not specified")

        # Build docker run command
        cmd = [
            "docker", "run",
            "-d",  # Detached
            "--name", f"ccea-sandbox-{self._sandbox_id[:8]}",
            "--rm",  # Auto remove
            f"--memory={self.config.memory_limit_mb}m",
            f"--cpus={self.config.cpu_limit}",
            "--network", self.config.docker_network,
        ]

        # Read-only filesystem
        if self.config.readonly_fs:
            cmd.append("--read-only")
            # Add tmpfs for writable directories
            cmd.extend(["--tmpfs", "/tmp:size=64m"])

        # Volume mounts
        for host_path, container_path in self.config.docker_volumes.items():
            cmd.extend(["-v", f"{host_path}:{container_path}"])

        # Environment variables
        for key, value in self.config.env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Image and command
        cmd.append(self.config.docker_image)

        # Start container
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")

        self._container_id = result.stdout.strip()

    def _stop_container(self, timeout: float) -> None:
        """Stop Docker container sandbox."""
        if not self._container_id:
            return

        # Stop container
        subprocess.run(
            ["docker", "stop", "-t", str(int(timeout)), self._container_id],
            capture_output=True,
            timeout=timeout + 10,
        )

        self._container_id = None

    def get_status(self) -> Dict[str, Any]:
        """
        Get sandbox status.

        Returns:
            Status dictionary
        """
        uptime = 0.0
        if self._start_time:
            uptime = (datetime.utcnow() - self._start_time).total_seconds()

        return {
            "sandbox_id": self._sandbox_id,
            "state": self._state.name,
            "sandbox_type": self.config.sandbox_type.name,
            "is_running": self.is_running,
            "uptime_seconds": uptime,
            "restart_count": self._restart_count,
            "config": self.config.to_dict(),
            "container_id": self._container_id,
        }

    def update_metrics(self) -> SandboxMetrics:
        """
        Update and return current metrics.

        Returns:
            Updated metrics
        """
        self._metrics.timestamp = datetime.utcnow()

        if self._start_time:
            self._metrics.uptime_seconds = (
                datetime.utcnow() - self._start_time
            ).total_seconds()

        self._metrics.restart_count = self._restart_count

        # Get process metrics if using process isolation
        if self._process and self._process.is_alive():
            try:
                import psutil
                proc = psutil.Process(self._process.pid)
                self._metrics.cpu_percent = proc.cpu_percent()
                mem_info = proc.memory_info()
                self._metrics.memory_mb = mem_info.rss / (1024 * 1024)
            except Exception:
                pass

        # Get container metrics if using container isolation
        if self._container_id:
            try:
                result = subprocess.run(
                    ["docker", "stats", "--no-stream", "--format",
                     "{{.CPUPerc}},{{.MemUsage}}", self._container_id],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    if len(parts) >= 2:
                        self._metrics.cpu_percent = float(parts[0].rstrip("%"))
                        # Parse memory (e.g., "123MiB / 512MiB")
                        mem_parts = parts[1].split("/")
                        if mem_parts:
                            mem_str = mem_parts[0].strip()
                            if "MiB" in mem_str:
                                self._metrics.memory_mb = float(mem_str.replace("MiB", ""))
                            elif "GiB" in mem_str:
                                self._metrics.memory_mb = float(mem_str.replace("GiB", "")) * 1024
            except Exception:
                pass

        return self._metrics


def create_sandbox(
    sandbox_type: str = "process",
    cpu_limit: float = DEFAULT_CPU_LIMIT,
    memory_limit_mb: int = DEFAULT_MEMORY_MB,
    **kwargs,
) -> Sandbox:
    """
    Factory function to create sandbox.

    Args:
        sandbox_type: Type of sandbox (process, container, none)
        cpu_limit: CPU limit in cores
        memory_limit_mb: Memory limit in MB
        **kwargs: Additional config options

    Returns:
        Configured Sandbox
    """
    type_map = {
        "process": SandboxType.PROCESS,
        "container": SandboxType.CONTAINER,
        "none": SandboxType.NONE,
    }

    config = SandboxConfig(
        sandbox_type=type_map.get(sandbox_type.lower(), SandboxType.PROCESS),
        cpu_limit=cpu_limit,
        memory_limit_mb=memory_limit_mb,
        **kwargs,
    )

    return Sandbox(config)
