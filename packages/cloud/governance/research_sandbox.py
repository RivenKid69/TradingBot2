"""
Research Sandbox Service - Phase 7 GDPR Implementation

Implements Design Doc Section 15.3 Cloud execution isolation controls:
- Sandbox isolation levels
- CPU/RAM quotas
- Egress allowlist
- Abuse detection (mining, botnet, scanning)

Reference: docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L927-L935
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4


# =============================================================================
# Enums and Constants
# =============================================================================

class IsolationLevel(str, Enum):
    """Sandbox isolation levels."""
    CONTAINER = "container"       # Standard container isolation
    CONTAINER_HARDENED = "container_hardened"  # Container with additional restrictions
    FIRECRACKER = "firecracker"   # microVM isolation
    KATA = "kata"                 # Kata containers
    VM = "vm"                     # Full VM isolation
    WASM = "wasm"                 # WebAssembly sandbox


class JobStatus(str, Enum):
    """Research job status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    TERMINATED = "terminated"
    QUOTA_EXCEEDED = "quota_exceeded"
    ABUSE_DETECTED = "abuse_detected"


class AbuseType(str, Enum):
    """Types of detected abuse."""
    CRYPTO_MINING = "crypto_mining"
    BOTNET_ACTIVITY = "botnet_activity"
    NETWORK_SCANNING = "network_scanning"
    DATA_EXFILTRATION = "data_exfiltration"
    RESOURCE_ABUSE = "resource_abuse"
    CREDENTIAL_STUFFING = "credential_stuffing"
    MALWARE_ACTIVITY = "malware_activity"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    EXCESSIVE_EGRESS = "excessive_egress"
    SUSPICIOUS_PROCESS = "suspicious_process"


class AbuseAction(str, Enum):
    """Actions taken on abuse detection."""
    ALERT = "alert"
    THROTTLE = "throttle"
    TERMINATE = "terminate"
    BLOCK = "block"
    QUARANTINE = "quarantine"


class SandboxEventType(str, Enum):
    """Types of sandbox events."""
    JOB_CREATED = "job_created"
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_TERMINATED = "job_terminated"
    QUOTA_EXCEEDED = "quota_exceeded"
    EGRESS_BLOCKED = "egress_blocked"
    EGRESS_ALLOWED = "egress_allowed"
    ABUSE_DETECTED = "abuse_detected"
    ABUSE_ACTION_TAKEN = "abuse_action_taken"
    POLICY_UPDATED = "policy_updated"
    RESOURCE_WARNING = "resource_warning"
    SANDBOX_CREATED = "sandbox_created"
    SANDBOX_DESTROYED = "sandbox_destroyed"


class QuotaType(str, Enum):
    """Types of resource quotas."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    GPU = "gpu"
    NETWORK_EGRESS = "network_egress"
    RUNTIME = "runtime"
    PROCESSES = "processes"
    FILE_DESCRIPTORS = "file_descriptors"


# Constants
DEFAULT_CPU_LIMIT = 2.0
DEFAULT_MEMORY_LIMIT_MB = 4096
DEFAULT_DISK_LIMIT_MB = 10240
DEFAULT_RUNTIME_SECONDS = 3600  # 1 hour
MAX_RUNTIME_SECONDS = 86400  # 24 hours
DEFAULT_MAX_PROCESSES = 100
DEFAULT_MAX_FILE_DESCRIPTORS = 1024
ABUSE_DETECTION_INTERVAL_SECONDS = 30
CRYPTO_MINING_CPU_THRESHOLD = 0.9  # 90% sustained CPU
EGRESS_RATE_LIMIT_BYTES_PER_SEC = 10 * 1024 * 1024  # 10 MB/s


# Known crypto mining patterns
CRYPTO_MINING_PATTERNS = [
    r"stratum\+tcp://",
    r"xmr\.",
    r"monero",
    r"nicehash",
    r"pool\.(supportxmr|hashvault|minexmr)",
    r"coinhive",
    r"--(donate-level|user|pass|algo|url).*pool",
]

# Known C2 patterns
C2_PATTERNS = [
    r"\.onion$",
    r"tor2web",
    r"\.bit$",
    r"ngrok\.io",
    r"serveo\.net",
    r"pagekite\.me",
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ResourceQuota:
    """Resource quota configuration."""
    quota_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    cpu_limit: float = DEFAULT_CPU_LIMIT
    cpu_request: float = 0.5
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB
    memory_request_mb: int = 512
    disk_limit_mb: int = DEFAULT_DISK_LIMIT_MB
    gpu_limit: int = 0
    max_runtime_seconds: int = DEFAULT_RUNTIME_SECONDS
    max_processes: int = DEFAULT_MAX_PROCESSES
    max_file_descriptors: int = DEFAULT_MAX_FILE_DESCRIPTORS
    max_egress_bytes_per_sec: int = EGRESS_RATE_LIMIT_BYTES_PER_SEC
    max_concurrent_jobs: int = 5
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quota_id": self.quota_id,
            "workspace_id": self.workspace_id,
            "cpu_limit": self.cpu_limit,
            "cpu_request": self.cpu_request,
            "memory_limit_mb": self.memory_limit_mb,
            "memory_request_mb": self.memory_request_mb,
            "disk_limit_mb": self.disk_limit_mb,
            "gpu_limit": self.gpu_limit,
            "max_runtime_seconds": self.max_runtime_seconds,
            "max_processes": self.max_processes,
            "max_file_descriptors": self.max_file_descriptors,
            "max_egress_bytes_per_sec": self.max_egress_bytes_per_sec,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class SandboxConfig:
    """Sandbox isolation configuration."""
    config_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    isolation_level: IsolationLevel = IsolationLevel.CONTAINER_HARDENED
    seccomp_profile: str = "runtime/default"
    apparmor_profile: str = "runtime/default"
    capabilities_drop: List[str] = field(default_factory=lambda: [
        "ALL",
    ])
    capabilities_add: List[str] = field(default_factory=list)
    read_only_rootfs: bool = True
    no_new_privileges: bool = True
    user_namespace: bool = True
    privileged: bool = False
    allowed_syscalls: Set[str] = field(default_factory=set)
    blocked_syscalls: Set[str] = field(default_factory=lambda: {
        "mount", "umount", "ptrace", "kexec_load", "reboot",
        "swapon", "swapoff", "syslog", "init_module", "delete_module",
    })
    network_mode: str = "none"  # none, host, bridge, custom
    volumes_read_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "workspace_id": self.workspace_id,
            "isolation_level": self.isolation_level.value,
            "seccomp_profile": self.seccomp_profile,
            "apparmor_profile": self.apparmor_profile,
            "capabilities_drop": self.capabilities_drop,
            "capabilities_add": self.capabilities_add,
            "read_only_rootfs": self.read_only_rootfs,
            "no_new_privileges": self.no_new_privileges,
            "user_namespace": self.user_namespace,
            "privileged": self.privileged,
            "allowed_syscalls": list(self.allowed_syscalls),
            "blocked_syscalls": list(self.blocked_syscalls),
            "network_mode": self.network_mode,
            "volumes_read_only": self.volumes_read_only,
        }


@dataclass
class EgressRule:
    """A rule in the egress policy."""
    rule_id: str = field(default_factory=lambda: str(uuid4()))
    domain: Optional[str] = None        # Domain pattern (e.g., "*.polygon.io")
    ip_cidr: Optional[str] = None       # IP/CIDR (e.g., "10.0.0.0/8")
    port: Optional[int] = None          # Specific port (None = all)
    ports: Set[int] = field(default_factory=lambda: {443, 80})
    protocol: str = "tcp"               # tcp, udp, any
    allow: bool = True
    description: str = ""

    def matches(self, target_domain: Optional[str], target_ip: Optional[str], target_port: int) -> bool:
        """Check if this rule matches the target."""
        # Check domain
        if self.domain and target_domain:
            if self.domain.startswith("*."):
                pattern = self.domain[2:]
                if not (target_domain.endswith(pattern) or target_domain == pattern.lstrip(".")):
                    return False
            elif target_domain != self.domain:
                return False

        # Check IP/CIDR
        if self.ip_cidr and target_ip:
            # Simplified CIDR matching (production would use proper CIDR math)
            if "/" in self.ip_cidr:
                network = self.ip_cidr.split("/")[0]
                if not target_ip.startswith(network.rsplit(".", 1)[0]):
                    return False
            elif target_ip != self.ip_cidr:
                return False

        # Check port
        if self.port and target_port != self.port:
            return False
        if self.ports and target_port not in self.ports:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "domain": self.domain,
            "ip_cidr": self.ip_cidr,
            "port": self.port,
            "ports": list(self.ports),
            "protocol": self.protocol,
            "allow": self.allow,
            "description": self.description,
        }


@dataclass
class EgressPolicy:
    """Egress policy for a workspace."""
    policy_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    deny_all_by_default: bool = True
    rules: List[EgressRule] = field(default_factory=list)
    log_all_connections: bool = True
    rate_limit_bytes_per_sec: int = EGRESS_RATE_LIMIT_BYTES_PER_SEC
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_allowed(
        self,
        domain: Optional[str],
        ip: Optional[str],
        port: int,
    ) -> Tuple[bool, Optional[EgressRule]]:
        """Check if egress is allowed."""
        # Check each rule in order
        for rule in self.rules:
            if rule.matches(domain, ip, port):
                return rule.allow, rule

        # Default action
        return not self.deny_all_by_default, None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "workspace_id": self.workspace_id,
            "deny_all_by_default": self.deny_all_by_default,
            "rules": [r.to_dict() for r in self.rules],
            "log_all_connections": self.log_all_connections,
            "rate_limit_bytes_per_sec": self.rate_limit_bytes_per_sec,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class ResourceUsage:
    """Current resource usage for a job."""
    cpu_percent: float = 0.0
    memory_used_mb: int = 0
    memory_percent: float = 0.0
    disk_used_mb: int = 0
    disk_percent: float = 0.0
    network_rx_bytes: int = 0
    network_tx_bytes: int = 0
    process_count: int = 0
    file_descriptor_count: int = 0
    runtime_seconds: float = 0.0
    sampled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_percent": self.memory_percent,
            "disk_used_mb": self.disk_used_mb,
            "disk_percent": self.disk_percent,
            "network_rx_bytes": self.network_rx_bytes,
            "network_tx_bytes": self.network_tx_bytes,
            "process_count": self.process_count,
            "file_descriptor_count": self.file_descriptor_count,
            "runtime_seconds": self.runtime_seconds,
            "sampled_at": self.sampled_at.isoformat(),
        }


@dataclass
class ResearchJob:
    """A research job running in a sandbox."""
    job_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    user_id: str = ""
    name: str = ""
    description: str = ""
    image_digest: str = ""
    command: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    quota: Optional[ResourceQuota] = None
    sandbox_config: Optional[SandboxConfig] = None
    current_usage: Optional[ResourceUsage] = None
    peak_usage: Optional[ResourceUsage] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    output_artifacts: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)

    def get_runtime_seconds(self) -> float:
        """Get total runtime in seconds."""
        if not self.started_at:
            return 0.0
        end_time = self.completed_at or datetime.now(timezone.utc)
        return (end_time - self.started_at).total_seconds()

    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status == JobStatus.RUNNING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "image_digest": self.image_digest,
            "command": self.command,
            "environment": {k: "***" for k in self.environment},  # Redact values
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "quota": self.quota.to_dict() if self.quota else None,
            "sandbox_config": self.sandbox_config.to_dict() if self.sandbox_config else None,
            "current_usage": self.current_usage.to_dict() if self.current_usage else None,
            "peak_usage": self.peak_usage.to_dict() if self.peak_usage else None,
            "exit_code": self.exit_code,
            "error": self.error,
            "output_artifacts": self.output_artifacts,
            "labels": self.labels,
        }


@dataclass
class AbuseIndicator:
    """An indicator of potential abuse."""
    indicator_id: str = field(default_factory=lambda: str(uuid4()))
    abuse_type: AbuseType = AbuseType.RESOURCE_ABUSE
    pattern: str = ""
    description: str = ""
    severity: float = 0.5  # 0.0 - 1.0
    threshold: Optional[float] = None
    duration_seconds: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "indicator_id": self.indicator_id,
            "abuse_type": self.abuse_type.value,
            "pattern": self.pattern,
            "description": self.description,
            "severity": self.severity,
            "threshold": self.threshold,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class AbuseEvent:
    """A detected abuse event."""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    job_id: str = ""
    workspace_id: str = ""
    abuse_type: AbuseType = AbuseType.RESOURCE_ABUSE
    confidence: float = 0.0
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    indicators: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    action_taken: AbuseAction = AbuseAction.ALERT
    action_taken_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    evidence_hash: str = ""

    def __post_init__(self) -> None:
        if not self.evidence_hash:
            self.evidence_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps({
            "event_id": self.event_id,
            "job_id": self.job_id,
            "workspace_id": self.workspace_id,
            "abuse_type": self.abuse_type.value,
            "detected_at": self.detected_at.isoformat(),
            "indicators": self.indicators,
        }, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "job_id": self.job_id,
            "workspace_id": self.workspace_id,
            "abuse_type": self.abuse_type.value,
            "confidence": self.confidence,
            "detected_at": self.detected_at.isoformat(),
            "indicators": self.indicators,
            "evidence": self.evidence,
            "action_taken": self.action_taken.value,
            "action_taken_at": self.action_taken_at.isoformat() if self.action_taken_at else None,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "evidence_hash": self.evidence_hash,
        }


@dataclass
class EgressLogEntry:
    """Log entry for egress connection."""
    entry_id: str = field(default_factory=lambda: str(uuid4()))
    job_id: str = ""
    workspace_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    destination_domain: Optional[str] = None
    destination_ip: Optional[str] = None
    destination_port: int = 0
    protocol: str = "tcp"
    bytes_sent: int = 0
    bytes_received: int = 0
    allowed: bool = True
    rule_id: Optional[str] = None
    blocked_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "job_id": self.job_id,
            "workspace_id": self.workspace_id,
            "timestamp": self.timestamp.isoformat(),
            "destination_domain": self.destination_domain,
            "destination_ip": self.destination_ip,
            "destination_port": self.destination_port,
            "protocol": self.protocol,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "allowed": self.allowed,
            "rule_id": self.rule_id,
            "blocked_reason": self.blocked_reason,
        }


@dataclass
class SandboxEvent:
    """Event in the sandbox audit log."""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    job_id: Optional[str] = None
    event_type: SandboxEventType = SandboxEventType.JOB_CREATED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    actor_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    result: str = "success"
    integrity_hash: str = ""

    def __post_init__(self) -> None:
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps({
            "event_id": self.event_id,
            "workspace_id": self.workspace_id,
            "job_id": self.job_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "actor_id": self.actor_id,
            "result": self.result,
        }, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "workspace_id": self.workspace_id,
            "job_id": self.job_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "actor_id": self.actor_id,
            "details": self.details,
            "result": self.result,
            "integrity_hash": self.integrity_hash,
        }


@dataclass
class SandboxStats:
    """Statistics for sandbox usage."""
    workspace_id: str = ""
    total_jobs: int = 0
    jobs_running: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    jobs_terminated: int = 0
    total_runtime_hours: float = 0.0
    avg_cpu_utilization: float = 0.0
    avg_memory_utilization: float = 0.0
    abuse_events_30d: int = 0
    egress_blocked_30d: int = 0
    quota_exceeded_30d: int = 0
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "total_jobs": self.total_jobs,
            "jobs_running": self.jobs_running,
            "jobs_completed": self.jobs_completed,
            "jobs_failed": self.jobs_failed,
            "jobs_terminated": self.jobs_terminated,
            "total_runtime_hours": self.total_runtime_hours,
            "avg_cpu_utilization": self.avg_cpu_utilization,
            "avg_memory_utilization": self.avg_memory_utilization,
            "abuse_events_30d": self.abuse_events_30d,
            "egress_blocked_30d": self.egress_blocked_30d,
            "quota_exceeded_30d": self.quota_exceeded_30d,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


# =============================================================================
# Research Sandbox Service
# =============================================================================

class ResearchSandboxService:
    """
    Research Sandbox Service.

    Manages:
    - Sandbox isolation configuration
    - Resource quotas
    - Egress allowlist
    - Abuse detection

    Reference: Design Doc Section 15.3
    """

    def __init__(
        self,
        default_isolation: IsolationLevel = IsolationLevel.CONTAINER_HARDENED,
        default_quota: Optional[ResourceQuota] = None,
        on_event: Optional[Callable[[SandboxEvent], None]] = None,
        on_abuse_detected: Optional[Callable[[AbuseEvent], None]] = None,
        on_quota_exceeded: Optional[Callable[[ResearchJob], None]] = None,
    ):
        """
        Initialize Research Sandbox Service.

        Args:
            default_isolation: Default isolation level
            default_quota: Default resource quota
            on_event: Callback for sandbox events
            on_abuse_detected: Callback for abuse detection
            on_quota_exceeded: Callback for quota exceeded
        """
        self._lock = threading.RLock()  # Re-entrant lock for nested calls

        # Configuration
        self._default_isolation = default_isolation
        self._default_quota = default_quota or ResourceQuota()

        # Callbacks
        self._on_event = on_event
        self._on_abuse_detected = on_abuse_detected
        self._on_quota_exceeded = on_quota_exceeded

        # Compile abuse detection patterns
        self._crypto_patterns = [re.compile(p, re.IGNORECASE) for p in CRYPTO_MINING_PATTERNS]
        self._c2_patterns = [re.compile(p, re.IGNORECASE) for p in C2_PATTERNS]

        # Storage
        self._quotas: Dict[str, ResourceQuota] = {}  # workspace_id -> quota
        self._sandbox_configs: Dict[str, SandboxConfig] = {}  # workspace_id -> config
        self._egress_policies: Dict[str, EgressPolicy] = {}  # workspace_id -> policy
        self._jobs: Dict[str, ResearchJob] = {}  # job_id -> job
        self._abuse_events: List[AbuseEvent] = []
        self._egress_logs: List[EgressLogEntry] = []
        self._events: List[SandboxEvent] = []

    # =========================================================================
    # Quota Management
    # =========================================================================

    def set_quota(
        self,
        workspace_id: str,
        quota: ResourceQuota,
        set_by: str,
    ) -> ResourceQuota:
        """Set resource quota for a workspace."""
        quota.workspace_id = workspace_id
        quota.updated_at = datetime.now(timezone.utc)

        with self._lock:
            self._quotas[workspace_id] = quota

        # Log event
        event = SandboxEvent(
            workspace_id=workspace_id,
            event_type=SandboxEventType.POLICY_UPDATED,
            actor_id=set_by,
            details={
                "policy_type": "quota",
                "cpu_limit": quota.cpu_limit,
                "memory_limit_mb": quota.memory_limit_mb,
                "max_runtime_seconds": quota.max_runtime_seconds,
            },
        )
        self._log_event(event)

        return quota

    def get_quota(self, workspace_id: str) -> ResourceQuota:
        """Get resource quota for a workspace."""
        with self._lock:
            return self._quotas.get(workspace_id, self._default_quota)

    def check_quota(
        self,
        workspace_id: str,
        usage: ResourceUsage,
    ) -> Tuple[bool, List[str]]:
        """Check if resource usage is within quota."""
        quota = self.get_quota(workspace_id)
        violations = []

        if usage.cpu_percent > 100:
            violations.append(f"CPU usage {usage.cpu_percent:.1f}% exceeds limit")

        if usage.memory_used_mb > quota.memory_limit_mb:
            violations.append(f"Memory usage {usage.memory_used_mb}MB exceeds limit {quota.memory_limit_mb}MB")

        if usage.disk_used_mb > quota.disk_limit_mb:
            violations.append(f"Disk usage {usage.disk_used_mb}MB exceeds limit {quota.disk_limit_mb}MB")

        if usage.process_count > quota.max_processes:
            violations.append(f"Process count {usage.process_count} exceeds limit {quota.max_processes}")

        if usage.runtime_seconds > quota.max_runtime_seconds:
            violations.append(f"Runtime {usage.runtime_seconds:.0f}s exceeds limit {quota.max_runtime_seconds}s")

        return len(violations) == 0, violations

    # =========================================================================
    # Sandbox Configuration
    # =========================================================================

    def set_sandbox_config(
        self,
        workspace_id: str,
        config: SandboxConfig,
        set_by: str,
    ) -> SandboxConfig:
        """Set sandbox configuration for a workspace."""
        config.workspace_id = workspace_id

        with self._lock:
            self._sandbox_configs[workspace_id] = config

        # Log event
        event = SandboxEvent(
            workspace_id=workspace_id,
            event_type=SandboxEventType.POLICY_UPDATED,
            actor_id=set_by,
            details={
                "policy_type": "sandbox_config",
                "isolation_level": config.isolation_level.value,
                "read_only_rootfs": config.read_only_rootfs,
                "no_new_privileges": config.no_new_privileges,
            },
        )
        self._log_event(event)

        return config

    def get_sandbox_config(self, workspace_id: str) -> SandboxConfig:
        """Get sandbox configuration for a workspace."""
        with self._lock:
            config = self._sandbox_configs.get(workspace_id)
            if config:
                return config

        # Return default config
        return SandboxConfig(
            workspace_id=workspace_id,
            isolation_level=self._default_isolation,
        )

    # =========================================================================
    # Egress Policy
    # =========================================================================

    def set_egress_policy(
        self,
        workspace_id: str,
        policy: EgressPolicy,
        set_by: str,
    ) -> EgressPolicy:
        """Set egress policy for a workspace."""
        policy.workspace_id = workspace_id
        policy.updated_at = datetime.now(timezone.utc)

        with self._lock:
            self._egress_policies[workspace_id] = policy

        # Log event
        event = SandboxEvent(
            workspace_id=workspace_id,
            event_type=SandboxEventType.POLICY_UPDATED,
            actor_id=set_by,
            details={
                "policy_type": "egress",
                "deny_all_by_default": policy.deny_all_by_default,
                "rules_count": len(policy.rules),
            },
        )
        self._log_event(event)

        return policy

    def get_egress_policy(self, workspace_id: str) -> EgressPolicy:
        """Get egress policy for a workspace."""
        with self._lock:
            policy = self._egress_policies.get(workspace_id)
            if policy:
                return policy

        # Return default (deny all)
        return EgressPolicy(
            workspace_id=workspace_id,
            deny_all_by_default=True,
        )

    def add_egress_rule(
        self,
        workspace_id: str,
        rule: EgressRule,
        added_by: str,
    ) -> EgressPolicy:
        """Add an egress rule to a workspace policy."""
        with self._lock:
            policy = self._egress_policies.get(workspace_id)
            if not policy:
                policy = EgressPolicy(workspace_id=workspace_id)
                self._egress_policies[workspace_id] = policy

            policy.rules.append(rule)
            policy.updated_at = datetime.now(timezone.utc)

        # Log event
        event = SandboxEvent(
            workspace_id=workspace_id,
            event_type=SandboxEventType.POLICY_UPDATED,
            actor_id=added_by,
            details={
                "action": "add_egress_rule",
                "rule_id": rule.rule_id,
                "domain": rule.domain,
                "allow": rule.allow,
            },
        )
        self._log_event(event)

        return policy

    def check_egress(
        self,
        job_id: str,
        workspace_id: str,
        domain: Optional[str],
        ip: Optional[str],
        port: int,
        bytes_sent: int = 0,
        bytes_received: int = 0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if egress connection is allowed.

        Returns: (allowed, blocked_reason)
        """
        policy = self.get_egress_policy(workspace_id)
        allowed, matched_rule = policy.is_allowed(domain, ip, port)

        # Check for malicious patterns
        if allowed and domain:
            # Check C2 patterns
            for pattern in self._c2_patterns:
                if pattern.search(domain):
                    allowed = False
                    blocked_reason = f"Blocked: matches known C2 pattern"
                    break

            # Check crypto mining patterns
            for pattern in self._crypto_patterns:
                if pattern.search(domain):
                    allowed = False
                    blocked_reason = f"Blocked: matches crypto mining pattern"
                    break

        blocked_reason = None
        if not allowed:
            blocked_reason = "Denied by default policy" if not matched_rule else "Denied by rule"

        # Log egress attempt
        log_entry = EgressLogEntry(
            job_id=job_id,
            workspace_id=workspace_id,
            destination_domain=domain,
            destination_ip=ip,
            destination_port=port,
            bytes_sent=bytes_sent,
            bytes_received=bytes_received,
            allowed=allowed,
            rule_id=matched_rule.rule_id if matched_rule else None,
            blocked_reason=blocked_reason,
        )

        with self._lock:
            self._egress_logs.append(log_entry)

        # Log event
        event_type = SandboxEventType.EGRESS_ALLOWED if allowed else SandboxEventType.EGRESS_BLOCKED
        event = SandboxEvent(
            workspace_id=workspace_id,
            job_id=job_id,
            event_type=event_type,
            actor_id="system",
            details={
                "domain": domain,
                "ip": ip,
                "port": port,
            },
            result="success" if allowed else "blocked",
        )
        self._log_event(event)

        return allowed, blocked_reason

    # =========================================================================
    # Job Management
    # =========================================================================

    def create_job(
        self,
        workspace_id: str,
        user_id: str,
        name: str,
        image_digest: str,
        command: List[str],
        environment: Optional[Dict[str, str]] = None,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
    ) -> ResearchJob:
        """Create a new research job."""
        quota = self.get_quota(workspace_id)
        sandbox_config = self.get_sandbox_config(workspace_id)

        # Check concurrent job limit
        running_jobs = self.list_jobs(
            workspace_id=workspace_id,
            status=JobStatus.RUNNING,
        )
        if len(running_jobs) >= quota.max_concurrent_jobs:
            raise ValueError(f"Concurrent job limit ({quota.max_concurrent_jobs}) exceeded")

        job = ResearchJob(
            workspace_id=workspace_id,
            user_id=user_id,
            name=name,
            description=description,
            image_digest=image_digest,
            command=command,
            environment=environment or {},
            quota=quota,
            sandbox_config=sandbox_config,
            labels=labels or {},
        )

        with self._lock:
            self._jobs[job.job_id] = job

        # Log event
        event = SandboxEvent(
            workspace_id=workspace_id,
            job_id=job.job_id,
            event_type=SandboxEventType.JOB_CREATED,
            actor_id=user_id,
            details={
                "name": name,
                "image_digest": image_digest,
                "isolation_level": sandbox_config.isolation_level.value,
            },
        )
        self._log_event(event)

        return job

    def start_job(
        self,
        job_id: str,
    ) -> ResearchJob:
        """Start a job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")

            if job.status != JobStatus.PENDING:
                raise ValueError(f"Job is in {job.status.value} status, cannot start")

            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)

        # Log event
        event = SandboxEvent(
            workspace_id=job.workspace_id,
            job_id=job_id,
            event_type=SandboxEventType.JOB_STARTED,
            actor_id="system",
            details={"name": job.name},
        )
        self._log_event(event)

        return job

    def update_job_usage(
        self,
        job_id: str,
        usage: ResourceUsage,
    ) -> ResearchJob:
        """Update resource usage for a job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")

            job.current_usage = usage

            # Update peak usage
            if not job.peak_usage:
                job.peak_usage = usage
            else:
                if usage.cpu_percent > job.peak_usage.cpu_percent:
                    job.peak_usage.cpu_percent = usage.cpu_percent
                if usage.memory_used_mb > job.peak_usage.memory_used_mb:
                    job.peak_usage.memory_used_mb = usage.memory_used_mb
                if usage.disk_used_mb > job.peak_usage.disk_used_mb:
                    job.peak_usage.disk_used_mb = usage.disk_used_mb

        # Check quota
        quota_ok, violations = self.check_quota(job.workspace_id, usage)
        if not quota_ok:
            # Log warning
            event = SandboxEvent(
                workspace_id=job.workspace_id,
                job_id=job_id,
                event_type=SandboxEventType.RESOURCE_WARNING,
                actor_id="system",
                details={"violations": violations},
            )
            self._log_event(event)

        # Check for abuse patterns in usage
        self._detect_abuse_from_usage(job, usage)

        return job

    def complete_job(
        self,
        job_id: str,
        exit_code: int,
        output_artifacts: Optional[List[str]] = None,
    ) -> ResearchJob:
        """Mark a job as completed."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")

            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.exit_code = exit_code
            job.output_artifacts = output_artifacts or []

        # Log event
        event = SandboxEvent(
            workspace_id=job.workspace_id,
            job_id=job_id,
            event_type=SandboxEventType.JOB_COMPLETED,
            actor_id="system",
            details={
                "exit_code": exit_code,
                "runtime_seconds": job.get_runtime_seconds(),
                "artifacts_count": len(job.output_artifacts),
            },
        )
        self._log_event(event)

        return job

    def fail_job(
        self,
        job_id: str,
        error: str,
        exit_code: Optional[int] = None,
    ) -> ResearchJob:
        """Mark a job as failed."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")

            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            job.error = error
            job.exit_code = exit_code

        # Log event
        event = SandboxEvent(
            workspace_id=job.workspace_id,
            job_id=job_id,
            event_type=SandboxEventType.JOB_FAILED,
            actor_id="system",
            details={
                "error": error,
                "exit_code": exit_code,
            },
            result="failure",
        )
        self._log_event(event)

        return job

    def terminate_job(
        self,
        job_id: str,
        terminated_by: str,
        reason: str,
    ) -> ResearchJob:
        """Terminate a running job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")

            if job.status != JobStatus.RUNNING:
                raise ValueError(f"Job is in {job.status.value} status, cannot terminate")

            job.status = JobStatus.TERMINATED
            job.completed_at = datetime.now(timezone.utc)
            job.error = reason

        # Log event
        event = SandboxEvent(
            workspace_id=job.workspace_id,
            job_id=job_id,
            event_type=SandboxEventType.JOB_TERMINATED,
            actor_id=terminated_by,
            details={"reason": reason},
        )
        self._log_event(event)

        return job

    def handle_quota_exceeded(
        self,
        job_id: str,
        violations: List[str],
    ) -> ResearchJob:
        """Handle quota exceeded for a job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")

            job.status = JobStatus.QUOTA_EXCEEDED
            job.completed_at = datetime.now(timezone.utc)
            job.error = "; ".join(violations)

        # Log event
        event = SandboxEvent(
            workspace_id=job.workspace_id,
            job_id=job_id,
            event_type=SandboxEventType.QUOTA_EXCEEDED,
            actor_id="system",
            details={"violations": violations},
            result="terminated",
        )
        self._log_event(event)

        if self._on_quota_exceeded:
            try:
                self._on_quota_exceeded(job)
            except Exception:
                pass

        return job

    def get_job(self, job_id: str) -> Optional[ResearchJob]:
        """Get job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(
        self,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 100,
    ) -> List[ResearchJob]:
        """List jobs with filtering."""
        with self._lock:
            jobs = list(self._jobs.values())

        if workspace_id:
            jobs = [j for j in jobs if j.workspace_id == workspace_id]
        if user_id:
            jobs = [j for j in jobs if j.user_id == user_id]
        if status:
            jobs = [j for j in jobs if j.status == status]

        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    # =========================================================================
    # Abuse Detection
    # =========================================================================

    def _detect_abuse_from_usage(
        self,
        job: ResearchJob,
        usage: ResourceUsage,
    ) -> Optional[AbuseEvent]:
        """Detect abuse patterns from resource usage."""
        indicators = []
        confidence = 0.0

        # Check for crypto mining (sustained high CPU)
        if usage.cpu_percent >= CRYPTO_MINING_CPU_THRESHOLD * 100:
            indicators.append("Sustained high CPU usage (>90%)")
            confidence += 0.4

        # Check for excessive network egress
        if usage.network_tx_bytes > EGRESS_RATE_LIMIT_BYTES_PER_SEC * 60:  # 1 minute sample
            indicators.append("High network egress rate")
            confidence += 0.3

        # Check for too many processes (potential fork bomb)
        if job.quota and usage.process_count > job.quota.max_processes * 0.9:
            indicators.append("Near process limit")
            confidence += 0.2

        if not indicators:
            return None

        # Only create abuse event if confidence is high enough
        if confidence < 0.5:
            return None

        abuse_type = AbuseType.CRYPTO_MINING if "CPU" in indicators[0] else AbuseType.RESOURCE_ABUSE

        abuse_event = AbuseEvent(
            job_id=job.job_id,
            workspace_id=job.workspace_id,
            abuse_type=abuse_type,
            confidence=min(confidence, 1.0),
            indicators=indicators,
            evidence={
                "cpu_percent": usage.cpu_percent,
                "network_tx_bytes": usage.network_tx_bytes,
                "process_count": usage.process_count,
            },
        )

        # Determine action
        if confidence >= 0.8:
            abuse_event.action_taken = AbuseAction.TERMINATE
            self.terminate_job(job.job_id, "system", f"Abuse detected: {abuse_type.value}")
        elif confidence >= 0.6:
            abuse_event.action_taken = AbuseAction.THROTTLE
        else:
            abuse_event.action_taken = AbuseAction.ALERT

        abuse_event.action_taken_at = datetime.now(timezone.utc)

        with self._lock:
            self._abuse_events.append(abuse_event)

        # Log event
        event = SandboxEvent(
            workspace_id=job.workspace_id,
            job_id=job.job_id,
            event_type=SandboxEventType.ABUSE_DETECTED,
            actor_id="system",
            details={
                "abuse_type": abuse_type.value,
                "confidence": confidence,
                "action": abuse_event.action_taken.value,
            },
        )
        self._log_event(event)

        if self._on_abuse_detected:
            try:
                self._on_abuse_detected(abuse_event)
            except Exception:
                pass

        return abuse_event

    def detect_abuse(
        self,
        job_id: str,
        abuse_type: AbuseType,
        indicators: List[str],
        evidence: Dict[str, Any],
        confidence: float,
        action: AbuseAction = AbuseAction.ALERT,
    ) -> AbuseEvent:
        """Manually record an abuse detection."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        abuse_event = AbuseEvent(
            job_id=job_id,
            workspace_id=job.workspace_id,
            abuse_type=abuse_type,
            confidence=confidence,
            indicators=indicators,
            evidence=evidence,
            action_taken=action,
            action_taken_at=datetime.now(timezone.utc),
        )

        with self._lock:
            self._abuse_events.append(abuse_event)

            # Update job status if action is terminate
            if action == AbuseAction.TERMINATE:
                job.status = JobStatus.ABUSE_DETECTED
                job.completed_at = datetime.now(timezone.utc)
                job.error = f"Abuse detected: {abuse_type.value}"

        # Log event
        event = SandboxEvent(
            workspace_id=job.workspace_id,
            job_id=job_id,
            event_type=SandboxEventType.ABUSE_DETECTED,
            actor_id="system",
            details={
                "abuse_type": abuse_type.value,
                "confidence": confidence,
                "action": action.value,
            },
        )
        self._log_event(event)

        if self._on_abuse_detected:
            try:
                self._on_abuse_detected(abuse_event)
            except Exception:
                pass

        return abuse_event

    def resolve_abuse_event(
        self,
        event_id: str,
        resolved_by: str,
    ) -> Optional[AbuseEvent]:
        """Resolve an abuse event."""
        with self._lock:
            for event in self._abuse_events:
                if event.event_id == event_id:
                    event.resolved = True
                    event.resolved_at = datetime.now(timezone.utc)
                    event.resolved_by = resolved_by
                    return event
        return None

    def get_abuse_events(
        self,
        workspace_id: Optional[str] = None,
        job_id: Optional[str] = None,
        abuse_type: Optional[AbuseType] = None,
        unresolved_only: bool = False,
        limit: int = 100,
    ) -> List[AbuseEvent]:
        """Get abuse events with filtering."""
        with self._lock:
            events = list(self._abuse_events)

        if workspace_id:
            events = [e for e in events if e.workspace_id == workspace_id]
        if job_id:
            events = [e for e in events if e.job_id == job_id]
        if abuse_type:
            events = [e for e in events if e.abuse_type == abuse_type]
        if unresolved_only:
            events = [e for e in events if not e.resolved]

        events.sort(key=lambda e: e.detected_at, reverse=True)
        return events[:limit]

    # =========================================================================
    # Statistics and Export
    # =========================================================================

    def get_stats(self, workspace_id: str) -> SandboxStats:
        """Get sandbox statistics for a workspace."""
        jobs = self.list_jobs(workspace_id=workspace_id, limit=10000)

        running = len([j for j in jobs if j.status == JobStatus.RUNNING])
        completed = len([j for j in jobs if j.status == JobStatus.COMPLETED])
        failed = len([j for j in jobs if j.status == JobStatus.FAILED])
        terminated = len([j for j in jobs if j.status in (JobStatus.TERMINATED, JobStatus.ABUSE_DETECTED)])

        total_runtime = sum(j.get_runtime_seconds() for j in jobs if j.completed_at)

        # Calculate average utilization from completed jobs
        cpu_samples = [j.peak_usage.cpu_percent for j in jobs if j.peak_usage]
        mem_samples = [j.peak_usage.memory_percent for j in jobs if j.peak_usage]

        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
        avg_mem = sum(mem_samples) / len(mem_samples) if mem_samples else 0.0

        # 30-day metrics
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)

        with self._lock:
            abuse_events_30d = len([e for e in self._abuse_events
                                   if e.workspace_id == workspace_id
                                   and e.detected_at >= cutoff])

            egress_blocked_30d = len([e for e in self._egress_logs
                                     if e.workspace_id == workspace_id
                                     and e.timestamp >= cutoff
                                     and not e.allowed])

        quota_exceeded_30d = len([j for j in jobs
                                  if j.status == JobStatus.QUOTA_EXCEEDED
                                  and j.completed_at and j.completed_at >= cutoff])

        return SandboxStats(
            workspace_id=workspace_id,
            total_jobs=len(jobs),
            jobs_running=running,
            jobs_completed=completed,
            jobs_failed=failed,
            jobs_terminated=terminated,
            total_runtime_hours=total_runtime / 3600,
            avg_cpu_utilization=avg_cpu,
            avg_memory_utilization=avg_mem,
            abuse_events_30d=abuse_events_30d,
            egress_blocked_30d=egress_blocked_30d,
            quota_exceeded_30d=quota_exceeded_30d,
        )

    def _log_event(self, event: SandboxEvent) -> None:
        """Log a sandbox event."""
        with self._lock:
            self._events.append(event)

        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                pass

    def get_events(
        self,
        workspace_id: Optional[str] = None,
        job_id: Optional[str] = None,
        event_type: Optional[SandboxEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[SandboxEvent]:
        """Get sandbox events."""
        with self._lock:
            events = list(self._events)

        if workspace_id:
            events = [e for e in events if e.workspace_id == workspace_id]
        if job_id:
            events = [e for e in events if e.job_id == job_id]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def get_egress_logs(
        self,
        workspace_id: Optional[str] = None,
        job_id: Optional[str] = None,
        blocked_only: bool = False,
        limit: int = 1000,
    ) -> List[EgressLogEntry]:
        """Get egress logs."""
        with self._lock:
            logs = list(self._egress_logs)

        if workspace_id:
            logs = [l for l in logs if l.workspace_id == workspace_id]
        if job_id:
            logs = [l for l in logs if l.job_id == job_id]
        if blocked_only:
            logs = [l for l in logs if not l.allowed]

        logs.sort(key=lambda l: l.timestamp, reverse=True)
        return logs[:limit]

    def export_sandbox_records(
        self,
        workspace_id: str,
        include_events: bool = True,
        include_abuse: bool = True,
        include_egress_logs: bool = True,
    ) -> Dict[str, Any]:
        """Export sandbox records for evidence pack."""
        quota = self.get_quota(workspace_id)
        sandbox_config = self.get_sandbox_config(workspace_id)
        egress_policy = self.get_egress_policy(workspace_id)
        jobs = self.list_jobs(workspace_id=workspace_id, limit=10000)

        export_data: Dict[str, Any] = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "workspace_id": workspace_id,
            "quota": quota.to_dict(),
            "sandbox_config": sandbox_config.to_dict(),
            "egress_policy": egress_policy.to_dict(),
            "jobs": [j.to_dict() for j in jobs],
            "stats": self.get_stats(workspace_id).to_dict(),
        }

        if include_events:
            events = self.get_events(workspace_id=workspace_id, limit=10000)
            export_data["events"] = [e.to_dict() for e in events]

        if include_abuse:
            abuse_events = self.get_abuse_events(workspace_id=workspace_id, limit=10000)
            export_data["abuse_events"] = [e.to_dict() for e in abuse_events]

        if include_egress_logs:
            egress_logs = self.get_egress_logs(workspace_id=workspace_id, limit=10000)
            export_data["egress_logs"] = [l.to_dict() for l in egress_logs]

        # Compute hash
        content = json.dumps(export_data, sort_keys=True, default=str)
        export_data["integrity_hash"] = f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

        return export_data
