# -*- coding: utf-8 -*-
"""
Abuse Detector - Detect mining/scanning/botnet patterns.

CCEA Phase 10: Abuse detection for cloud research jobs.

Design Doc 15.3/5.3:
- Запрет майнинга/сканирования/ботнета (abuse detection)
- Cloud execution должен быть защищен от "compute abuse"

This module detects various abuse patterns:
- Cryptocurrency mining (high CPU, specific process patterns)
- Port/network scanning (connection patterns)
- Botnet behavior (C2 communication patterns)
- Resource exhaustion attacks
- Data exfiltration attempts

Detection Methods:
- Process pattern matching
- Resource usage anomaly detection
- Network behavior analysis
- Syscall monitoring
- Code/binary analysis

Reference:
- AWS GuardDuty: https://docs.aws.amazon.com/guardduty/
- GCP Cloud IDS: https://cloud.google.com/ids/docs
- Falco: https://falco.org/docs/
"""

from __future__ import annotations

import hashlib
import math
import re
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple
from uuid import uuid4


# ============================================================================
# Constants
# ============================================================================

# Mining detection thresholds
MINING_CPU_THRESHOLD: Final[float] = 80.0  # % CPU for sustained period
MINING_DURATION_THRESHOLD: Final[int] = 60  # seconds of high CPU
MINING_PROCESS_KEYWORDS: Final[Set[str]] = {
    "xmrig", "minerd", "minergate", "cgminer", "bfgminer",
    "ethminer", "claymore", "phoenixminer", "gminer", "lolminer",
    "t-rex", "nbminer", "cpuminer", "cudaminer", "ccminer",
    "nicehash", "nanominer", "excavator", "bminer", "srbminer",
}

# Mining pool domains/IPs
MINING_POOL_PATTERNS: Final[List[str]] = [
    r".*pool\..*mining.*",
    r".*mining.*pool.*",
    r".*\.nicehash\.com",
    r".*\.nanopool\.org",
    r".*\.2miners\.com",
    r".*\.f2pool\.com",
    r".*\.antpool\.com",
    r".*\.ethermine\.org",
    r".*\.hiveon\.com",
    r".*stratum\+tcp://.*",
    r".*\.xmrpool\..*",
    r".*\.moneroocean\..*",
]

# Scanning detection thresholds
SCAN_CONNECTIONS_THRESHOLD: Final[int] = 50  # unique IPs in short window
SCAN_PORTS_THRESHOLD: Final[int] = 20  # unique ports per IP
SCAN_WINDOW_SECONDS: Final[int] = 60

# Botnet detection patterns
C2_COMMUNICATION_PATTERNS: Final[List[str]] = [
    r"^[a-f0-9]{64}$",  # Long hex strings (C2 commands)
    r"^[A-Za-z0-9+/]{100,}={0,2}$",  # Long base64 (encoded commands)
    r".*\.(onion|i2p)$",  # Tor/I2P domains
]

# Data exfiltration patterns
EXFIL_PATTERNS: Final[List[str]] = [
    r"\.env$",
    r"\.pem$",
    r"\.key$",
    r"id_rsa",
    r"id_ed25519",
    r"credentials",
    r"aws_access_key",
    r"api_key",
]

# Resource exhaustion thresholds
MEMORY_SPIKE_THRESHOLD: Final[float] = 90.0  # % memory usage
DISK_WRITE_THRESHOLD: Final[int] = 500 * 1024 * 1024  # 500 MB/min
PROCESS_SPAWN_THRESHOLD: Final[int] = 50  # processes per minute

# Alert cooldown (prevent alert storms)
ALERT_COOLDOWN_SECONDS: Final[int] = 300  # 5 minutes per alert type


class AbuseType(Enum):
    """Type of detected abuse."""
    CRYPTOCURRENCY_MINING = auto()
    PORT_SCANNING = auto()
    NETWORK_SCANNING = auto()
    BOTNET_C2 = auto()
    DATA_EXFILTRATION = auto()
    RESOURCE_EXHAUSTION = auto()
    PROCESS_INJECTION = auto()
    PRIVILEGE_ESCALATION = auto()
    DENIAL_OF_SERVICE = auto()
    MALWARE_EXECUTION = auto()
    CRYPTOJACKING = auto()
    REVERSE_SHELL = auto()


class AlertSeverity(Enum):
    """Alert severity level."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class AbuseAlert:
    """
    Alert for detected abuse.
    """
    alert_id: str = field(default_factory=lambda: str(uuid4()))
    tenant_id: str = ""
    job_id: str = ""
    sandbox_id: str = ""

    # Detection
    abuse_type: AbuseType = AbuseType.CRYPTOCURRENCY_MINING
    severity: AlertSeverity = AlertSeverity.MEDIUM
    confidence: float = 0.0  # 0-100

    # Details
    title: str = ""
    description: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    # Detection metadata
    detector_name: str = ""
    detection_method: str = ""

    # Timing
    detected_at: datetime = field(default_factory=datetime.utcnow)
    duration_seconds: float = 0.0

    # Action
    action_taken: str = ""
    job_terminated: bool = False

    # Risk score (0-100)
    risk_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "tenant_id": self.tenant_id,
            "job_id": self.job_id,
            "sandbox_id": self.sandbox_id,
            "abuse_type": self.abuse_type.name,
            "severity": self.severity.name,
            "confidence": self.confidence,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
            "detector_name": self.detector_name,
            "detected_at": self.detected_at.isoformat(),
            "action_taken": self.action_taken,
            "job_terminated": self.job_terminated,
            "risk_score": self.risk_score,
        }


@dataclass
class AbuseDetectorConfig:
    """
    Configuration for abuse detector.
    """
    # Mining detection
    mining_detection_enabled: bool = True
    mining_cpu_threshold: float = MINING_CPU_THRESHOLD
    mining_duration_threshold: int = MINING_DURATION_THRESHOLD

    # Scanning detection
    scanning_detection_enabled: bool = True
    scan_connections_threshold: int = SCAN_CONNECTIONS_THRESHOLD
    scan_ports_threshold: int = SCAN_PORTS_THRESHOLD

    # Botnet detection
    botnet_detection_enabled: bool = True

    # Exfiltration detection
    exfiltration_detection_enabled: bool = True

    # Resource exhaustion
    resource_exhaustion_enabled: bool = True
    memory_spike_threshold: float = MEMORY_SPIKE_THRESHOLD
    disk_write_threshold: int = DISK_WRITE_THRESHOLD

    # Alert settings
    alert_cooldown_seconds: int = ALERT_COOLDOWN_SECONDS
    auto_terminate_on_critical: bool = True

    # Monitoring interval
    monitoring_interval_seconds: float = 5.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mining_detection_enabled": self.mining_detection_enabled,
            "scanning_detection_enabled": self.scanning_detection_enabled,
            "botnet_detection_enabled": self.botnet_detection_enabled,
            "exfiltration_detection_enabled": self.exfiltration_detection_enabled,
            "resource_exhaustion_enabled": self.resource_exhaustion_enabled,
            "auto_terminate_on_critical": self.auto_terminate_on_critical,
        }


@dataclass
class JobMetrics:
    """
    Real-time metrics from a running job.
    """
    job_id: str = ""
    sandbox_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # CPU metrics
    cpu_percent: float = 0.0
    cpu_time_user: float = 0.0
    cpu_time_system: float = 0.0

    # Memory metrics
    memory_percent: float = 0.0
    memory_rss_mb: float = 0.0
    memory_vms_mb: float = 0.0

    # Process metrics
    process_count: int = 0
    thread_count: int = 0

    # I/O metrics
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0

    # Network metrics
    open_connections: int = 0
    unique_destinations: int = 0
    unique_ports: int = 0

    # Process info
    process_names: List[str] = field(default_factory=list)
    process_cmdlines: List[str] = field(default_factory=list)


class AbuseDetector:
    """
    Detect abuse patterns in cloud research jobs.

    Monitors job execution for signs of:
    - Cryptocurrency mining
    - Port/network scanning
    - Botnet C2 communication
    - Data exfiltration
    - Resource exhaustion attacks

    Usage:
        detector = AbuseDetector(
            on_alert=handle_alert,
            on_terminate=terminate_job,
        )

        # Start monitoring a job
        detector.start_monitoring(tenant_id, job_id, sandbox_id)

        # Feed metrics periodically
        detector.process_metrics(metrics)

        # Check network activity
        detector.check_network_activity(destination, port)

        # Check code/binary
        detector.analyze_code(code)

        # Stop monitoring
        detector.stop_monitoring(job_id)
    """

    def __init__(
        self,
        config: Optional[AbuseDetectorConfig] = None,
        on_alert: Optional[Callable[[AbuseAlert], None]] = None,
        on_terminate: Optional[Callable[[str, str], bool]] = None,
    ):
        """
        Initialize detector.

        Args:
            config: Detector configuration
            on_alert: Callback for alerts
            on_terminate: Callback to terminate job (job_id, reason) -> success
        """
        self.config = config or AbuseDetectorConfig()
        self._on_alert = on_alert
        self._on_terminate = on_terminate

        # State
        self._lock = threading.RLock()
        self._monitoring: Dict[str, Dict[str, Any]] = {}  # job_id -> state

        # Alert history
        self._alerts: List[AbuseAlert] = []
        self._max_alerts = 10000

        # Alert cooldown tracking
        self._last_alert_time: Dict[str, datetime] = {}  # (job_id, abuse_type) -> time

        # Compiled patterns
        self._mining_pool_patterns = [
            re.compile(p, re.IGNORECASE) for p in MINING_POOL_PATTERNS
        ]
        self._c2_patterns = [
            re.compile(p) for p in C2_COMMUNICATION_PATTERNS
        ]
        self._exfil_patterns = [
            re.compile(p, re.IGNORECASE) for p in EXFIL_PATTERNS
        ]

        # Statistics
        self._stats = {
            "jobs_monitored": 0,
            "alerts_generated": 0,
            "jobs_terminated": 0,
            "mining_detected": 0,
            "scanning_detected": 0,
            "botnet_detected": 0,
        }

    def start_monitoring(
        self,
        tenant_id: str,
        job_id: str,
        sandbox_id: str,
    ) -> None:
        """
        Start monitoring a job for abuse.

        Args:
            tenant_id: Tenant identifier
            job_id: Job identifier
            sandbox_id: Sandbox identifier
        """
        with self._lock:
            self._monitoring[job_id] = {
                "tenant_id": tenant_id,
                "sandbox_id": sandbox_id,
                "start_time": datetime.utcnow(),
                "metrics_history": [],
                "network_history": [],
                "high_cpu_start": None,
                "unique_destinations": set(),
                "unique_ports": defaultdict(set),  # dest -> ports
                "alerts_raised": set(),
            }
            self._stats["jobs_monitored"] += 1

    def stop_monitoring(self, job_id: str) -> None:
        """Stop monitoring a job."""
        with self._lock:
            if job_id in self._monitoring:
                del self._monitoring[job_id]

    def is_monitoring(self, job_id: str) -> bool:
        """Check if job is being monitored."""
        with self._lock:
            return job_id in self._monitoring

    def process_metrics(self, metrics: JobMetrics) -> List[AbuseAlert]:
        """
        Process metrics and detect abuse.

        Args:
            metrics: Current job metrics

        Returns:
            List of generated alerts
        """
        alerts = []

        with self._lock:
            state = self._monitoring.get(metrics.job_id)
            if not state:
                return alerts

            # Store metrics
            state["metrics_history"].append(metrics)

            # Trim history (keep last 100 samples)
            if len(state["metrics_history"]) > 100:
                state["metrics_history"] = state["metrics_history"][-100:]

            # Run detectors
            if self.config.mining_detection_enabled:
                alert = self._detect_mining(metrics, state)
                if alert:
                    alerts.append(alert)

            if self.config.resource_exhaustion_enabled:
                alert = self._detect_resource_exhaustion(metrics, state)
                if alert:
                    alerts.append(alert)

            if self.config.scanning_detection_enabled:
                alert = self._detect_scanning(metrics, state)
                if alert:
                    alerts.append(alert)

            # Check process names for malware
            alert = self._detect_malware_processes(metrics, state)
            if alert:
                alerts.append(alert)

        return alerts

    def check_network_activity(
        self,
        job_id: str,
        destination: str,
        port: int,
        bytes_count: int = 0,
        payload: Optional[bytes] = None,
    ) -> Optional[AbuseAlert]:
        """
        Check network activity for abuse patterns.

        Args:
            job_id: Job identifier
            destination: Destination IP/domain
            port: Destination port
            bytes_count: Data size
            payload: Raw payload (for pattern matching)

        Returns:
            Alert if abuse detected
        """
        with self._lock:
            state = self._monitoring.get(job_id)
            if not state:
                return None

            # Track unique destinations
            state["unique_destinations"].add(destination)
            state["unique_ports"][destination].add(port)

            # Record network activity
            state["network_history"].append({
                "timestamp": datetime.utcnow(),
                "destination": destination,
                "port": port,
                "bytes": bytes_count,
            })

            # Trim history
            if len(state["network_history"]) > 1000:
                state["network_history"] = state["network_history"][-1000:]

            # Check for mining pool connection
            if self.config.mining_detection_enabled:
                alert = self._detect_mining_pool(destination, state)
                if alert:
                    return alert

            # Check for C2 patterns
            if self.config.botnet_detection_enabled:
                alert = self._detect_c2_communication(
                    destination, port, payload, state
                )
                if alert:
                    return alert

            # Check for scanning behavior
            if self.config.scanning_detection_enabled:
                alert = self._detect_network_scanning(state)
                if alert:
                    return alert

        return None

    def analyze_code(
        self,
        job_id: str,
        code: str,
        filename: str = "",
    ) -> List[AbuseAlert]:
        """
        Analyze code for suspicious patterns.

        Args:
            job_id: Job identifier
            code: Code content
            filename: Filename

        Returns:
            List of alerts
        """
        alerts = []

        with self._lock:
            state = self._monitoring.get(job_id)
            if not state:
                return alerts

            # Check for mining code patterns
            if self.config.mining_detection_enabled:
                alert = self._detect_mining_code(code, filename, state)
                if alert:
                    alerts.append(alert)

            # Check for exfiltration attempts
            if self.config.exfiltration_detection_enabled:
                alert = self._detect_exfiltration_code(code, filename, state)
                if alert:
                    alerts.append(alert)

            # Check for reverse shell patterns
            alert = self._detect_reverse_shell_code(code, filename, state)
            if alert:
                alerts.append(alert)

        return alerts

    def _detect_mining(
        self,
        metrics: JobMetrics,
        state: Dict[str, Any],
    ) -> Optional[AbuseAlert]:
        """Detect cryptocurrency mining by CPU pattern."""
        if metrics.cpu_percent < self.config.mining_cpu_threshold:
            # CPU dropped below threshold, reset timer
            state["high_cpu_start"] = None
            return None

        # High CPU detected
        now = datetime.utcnow()

        if state["high_cpu_start"] is None:
            state["high_cpu_start"] = now
            return None

        # Check duration
        duration = (now - state["high_cpu_start"]).total_seconds()
        if duration < self.config.mining_duration_threshold:
            return None

        # Sustained high CPU - potential mining
        return self._create_alert(
            state=state,
            abuse_type=AbuseType.CRYPTOCURRENCY_MINING,
            severity=AlertSeverity.HIGH,
            confidence=70.0,
            title="Potential cryptocurrency mining detected",
            description=f"Sustained high CPU usage ({metrics.cpu_percent:.1f}%) "
                       f"for {duration:.0f} seconds",
            evidence={
                "cpu_percent": metrics.cpu_percent,
                "duration_seconds": duration,
                "process_names": metrics.process_names,
            },
            detector_name="mining_cpu_detector",
            risk_score=75.0,
        )

    def _detect_mining_pool(
        self,
        destination: str,
        state: Dict[str, Any],
    ) -> Optional[AbuseAlert]:
        """Detect connection to mining pool."""
        for pattern in self._mining_pool_patterns:
            if pattern.match(destination):
                return self._create_alert(
                    state=state,
                    abuse_type=AbuseType.CRYPTOCURRENCY_MINING,
                    severity=AlertSeverity.CRITICAL,
                    confidence=95.0,
                    title="Mining pool connection detected",
                    description=f"Connection attempt to suspected mining pool: {destination}",
                    evidence={
                        "destination": destination,
                        "matched_pattern": pattern.pattern,
                    },
                    detector_name="mining_pool_detector",
                    risk_score=95.0,
                )
        return None

    def _detect_mining_code(
        self,
        code: str,
        filename: str,
        state: Dict[str, Any],
    ) -> Optional[AbuseAlert]:
        """Detect mining code patterns."""
        code_lower = code.lower()

        # Check for mining keywords
        found_keywords = []
        for keyword in MINING_PROCESS_KEYWORDS:
            if keyword in code_lower:
                found_keywords.append(keyword)

        # Check for stratum protocol
        if "stratum+tcp" in code_lower or "stratum+ssl" in code_lower:
            found_keywords.append("stratum_protocol")

        # Check for hashrate/mining function names
        mining_functions = [
            "def mine(", "def hash_block(", "def proof_of_work(",
            "cryptonight", "randomx", "ethash", "kawpow",
        ]
        for func in mining_functions:
            if func in code_lower:
                found_keywords.append(func)

        if len(found_keywords) >= 2:
            return self._create_alert(
                state=state,
                abuse_type=AbuseType.CRYPTOCURRENCY_MINING,
                severity=AlertSeverity.HIGH,
                confidence=80.0,
                title="Mining code patterns detected",
                description=f"Code contains mining-related patterns: {', '.join(found_keywords[:5])}",
                evidence={
                    "filename": filename,
                    "keywords_found": found_keywords,
                },
                detector_name="mining_code_detector",
                risk_score=80.0,
            )
        return None

    def _detect_malware_processes(
        self,
        metrics: JobMetrics,
        state: Dict[str, Any],
    ) -> Optional[AbuseAlert]:
        """Detect known malware process names."""
        found_malware = []

        for name in metrics.process_names:
            name_lower = name.lower()
            for keyword in MINING_PROCESS_KEYWORDS:
                if keyword in name_lower:
                    found_malware.append(name)
                    break

        if found_malware:
            return self._create_alert(
                state=state,
                abuse_type=AbuseType.MALWARE_EXECUTION,
                severity=AlertSeverity.CRITICAL,
                confidence=90.0,
                title="Malware process detected",
                description=f"Known malware processes found: {', '.join(found_malware)}",
                evidence={
                    "malware_processes": found_malware,
                    "all_processes": metrics.process_names,
                },
                detector_name="malware_process_detector",
                risk_score=95.0,
            )
        return None

    def _detect_scanning(
        self,
        metrics: JobMetrics,
        state: Dict[str, Any],
    ) -> Optional[AbuseAlert]:
        """Detect port/network scanning."""
        # Check unique destinations
        if metrics.unique_destinations > self.config.scan_connections_threshold:
            return self._create_alert(
                state=state,
                abuse_type=AbuseType.NETWORK_SCANNING,
                severity=AlertSeverity.HIGH,
                confidence=85.0,
                title="Network scanning detected",
                description=f"High number of unique destinations: {metrics.unique_destinations}",
                evidence={
                    "unique_destinations": metrics.unique_destinations,
                    "threshold": self.config.scan_connections_threshold,
                },
                detector_name="network_scanning_detector",
                risk_score=80.0,
            )

        # Check port scanning (many ports per destination)
        for dest, ports in state["unique_ports"].items():
            if len(ports) > self.config.scan_ports_threshold:
                return self._create_alert(
                    state=state,
                    abuse_type=AbuseType.PORT_SCANNING,
                    severity=AlertSeverity.HIGH,
                    confidence=85.0,
                    title="Port scanning detected",
                    description=f"Port scan detected on {dest}: {len(ports)} unique ports",
                    evidence={
                        "destination": dest,
                        "ports_scanned": len(ports),
                        "sample_ports": list(ports)[:20],
                    },
                    detector_name="port_scanning_detector",
                    risk_score=80.0,
                )

        return None

    def _detect_network_scanning(
        self,
        state: Dict[str, Any],
    ) -> Optional[AbuseAlert]:
        """Detect network scanning based on connection history."""
        history = state["network_history"]
        if len(history) < 10:
            return None

        # Check connection rate in last minute
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=SCAN_WINDOW_SECONDS)
        recent = [h for h in history if h["timestamp"] > window_start]

        unique_recent_dests = set(h["destination"] for h in recent)

        if len(unique_recent_dests) > self.config.scan_connections_threshold:
            return self._create_alert(
                state=state,
                abuse_type=AbuseType.NETWORK_SCANNING,
                severity=AlertSeverity.HIGH,
                confidence=80.0,
                title="Rapid network scanning detected",
                description=f"High connection rate: {len(unique_recent_dests)} unique "
                           f"destinations in {SCAN_WINDOW_SECONDS}s",
                evidence={
                    "connections_in_window": len(recent),
                    "unique_destinations": len(unique_recent_dests),
                    "window_seconds": SCAN_WINDOW_SECONDS,
                },
                detector_name="rapid_scanning_detector",
                risk_score=80.0,
            )

        return None

    def _detect_c2_communication(
        self,
        destination: str,
        port: int,
        payload: Optional[bytes],
        state: Dict[str, Any],
    ) -> Optional[AbuseAlert]:
        """Detect botnet C2 communication patterns."""
        # Check for Tor/I2P
        if destination.endswith(".onion") or destination.endswith(".i2p"):
            return self._create_alert(
                state=state,
                abuse_type=AbuseType.BOTNET_C2,
                severity=AlertSeverity.CRITICAL,
                confidence=90.0,
                title="Anonymous network connection detected",
                description=f"Connection to Tor/I2P network: {destination}",
                evidence={
                    "destination": destination,
                    "port": port,
                },
                detector_name="anonymous_network_detector",
                risk_score=90.0,
            )

        # Check payload patterns
        if payload:
            try:
                payload_str = payload.decode("utf-8", errors="ignore")
                for pattern in self._c2_patterns:
                    if pattern.match(payload_str):
                        return self._create_alert(
                            state=state,
                            abuse_type=AbuseType.BOTNET_C2,
                            severity=AlertSeverity.HIGH,
                            confidence=70.0,
                            title="Suspicious C2-like payload detected",
                            description="Network payload matches known C2 patterns",
                            evidence={
                                "destination": destination,
                                "port": port,
                                "pattern_matched": pattern.pattern,
                            },
                            detector_name="c2_payload_detector",
                            risk_score=75.0,
                        )
            except Exception:
                pass

        return None

    def _detect_resource_exhaustion(
        self,
        metrics: JobMetrics,
        state: Dict[str, Any],
    ) -> Optional[AbuseAlert]:
        """Detect resource exhaustion attacks."""
        # Memory exhaustion
        if metrics.memory_percent > self.config.memory_spike_threshold:
            return self._create_alert(
                state=state,
                abuse_type=AbuseType.RESOURCE_EXHAUSTION,
                severity=AlertSeverity.MEDIUM,
                confidence=60.0,
                title="Memory exhaustion detected",
                description=f"Memory usage at {metrics.memory_percent:.1f}%",
                evidence={
                    "memory_percent": metrics.memory_percent,
                    "memory_rss_mb": metrics.memory_rss_mb,
                    "threshold": self.config.memory_spike_threshold,
                },
                detector_name="memory_exhaustion_detector",
                risk_score=50.0,
            )

        # Fork bomb detection (rapid process spawning)
        if metrics.process_count > PROCESS_SPAWN_THRESHOLD:
            return self._create_alert(
                state=state,
                abuse_type=AbuseType.RESOURCE_EXHAUSTION,
                severity=AlertSeverity.HIGH,
                confidence=80.0,
                title="Process fork bomb detected",
                description=f"Excessive process spawning: {metrics.process_count} processes",
                evidence={
                    "process_count": metrics.process_count,
                    "threshold": PROCESS_SPAWN_THRESHOLD,
                },
                detector_name="fork_bomb_detector",
                risk_score=85.0,
            )

        # Disk exhaustion
        history = state["metrics_history"]
        if len(history) >= 2:
            disk_write_rate = (
                metrics.disk_write_bytes - history[-2].disk_write_bytes
            ) / self.config.monitoring_interval_seconds * 60

            if disk_write_rate > self.config.disk_write_threshold:
                return self._create_alert(
                    state=state,
                    abuse_type=AbuseType.RESOURCE_EXHAUSTION,
                    severity=AlertSeverity.MEDIUM,
                    confidence=70.0,
                    title="Excessive disk write detected",
                    description=f"Disk write rate: {disk_write_rate / 1024 / 1024:.1f} MB/min",
                    evidence={
                        "disk_write_rate_mb_min": disk_write_rate / 1024 / 1024,
                        "threshold_mb_min": self.config.disk_write_threshold / 1024 / 1024,
                    },
                    detector_name="disk_exhaustion_detector",
                    risk_score=60.0,
                )

        return None

    def _detect_exfiltration_code(
        self,
        code: str,
        filename: str,
        state: Dict[str, Any],
    ) -> Optional[AbuseAlert]:
        """Detect data exfiltration attempts in code."""
        found_patterns = []

        for pattern in self._exfil_patterns:
            if pattern.search(code):
                found_patterns.append(pattern.pattern)

        # Check for sensitive file operations
        sensitive_ops = [
            "open('/etc/passwd'",
            "open('/etc/shadow'",
            "os.environ",
            "subprocess.call",
            "subprocess.run",
            "os.system(",
            "eval(",
            "exec(",
            "pickle.loads",
            "__import__",
        ]

        for op in sensitive_ops:
            if op in code:
                found_patterns.append(op)

        if len(found_patterns) >= 2:
            return self._create_alert(
                state=state,
                abuse_type=AbuseType.DATA_EXFILTRATION,
                severity=AlertSeverity.HIGH,
                confidence=70.0,
                title="Potential data exfiltration code detected",
                description=f"Code contains suspicious patterns: {', '.join(found_patterns[:5])}",
                evidence={
                    "filename": filename,
                    "patterns_found": found_patterns,
                },
                detector_name="exfiltration_code_detector",
                risk_score=75.0,
            )

        return None

    def _detect_reverse_shell_code(
        self,
        code: str,
        filename: str,
        state: Dict[str, Any],
    ) -> Optional[AbuseAlert]:
        """Detect reverse shell patterns in code."""
        shell_patterns = [
            "socket.socket",
            "os.dup2",
            "subprocess.call(['/bin/sh'",
            "subprocess.call(['/bin/bash'",
            "pty.spawn",
            "/dev/tcp/",
            "nc -e",
            "bash -i",
            "python -c 'import socket",
        ]

        found = []
        code_lower = code.lower()

        for pattern in shell_patterns:
            if pattern.lower() in code_lower:
                found.append(pattern)

        # Need multiple patterns for high confidence
        if len(found) >= 2:
            return self._create_alert(
                state=state,
                abuse_type=AbuseType.REVERSE_SHELL,
                severity=AlertSeverity.CRITICAL,
                confidence=85.0,
                title="Reverse shell code detected",
                description=f"Code contains reverse shell patterns: {', '.join(found)}",
                evidence={
                    "filename": filename,
                    "patterns_found": found,
                },
                detector_name="reverse_shell_detector",
                risk_score=95.0,
            )

        return None

    def _create_alert(
        self,
        state: Dict[str, Any],
        abuse_type: AbuseType,
        severity: AlertSeverity,
        confidence: float,
        title: str,
        description: str,
        evidence: Dict[str, Any],
        detector_name: str,
        risk_score: float,
    ) -> Optional[AbuseAlert]:
        """Create and record an alert."""
        # Check cooldown
        cooldown_key = f"{state.get('tenant_id', '')}:{abuse_type.name}"
        now = datetime.utcnow()

        if cooldown_key in self._last_alert_time:
            elapsed = (now - self._last_alert_time[cooldown_key]).total_seconds()
            if elapsed < self.config.alert_cooldown_seconds:
                return None

        # Check if already raised
        alert_key = (abuse_type, detector_name)
        if alert_key in state.get("alerts_raised", set()):
            return None

        # Create alert
        alert = AbuseAlert(
            tenant_id=state.get("tenant_id", ""),
            job_id=state.get("job_id", ""),
            sandbox_id=state.get("sandbox_id", ""),
            abuse_type=abuse_type,
            severity=severity,
            confidence=confidence,
            title=title,
            description=description,
            evidence=evidence,
            detector_name=detector_name,
            risk_score=risk_score,
        )

        # Record alert
        self._alerts.append(alert)
        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[-self._max_alerts:]

        self._last_alert_time[cooldown_key] = now
        state.setdefault("alerts_raised", set()).add(alert_key)

        # Update stats
        self._stats["alerts_generated"] += 1
        if abuse_type == AbuseType.CRYPTOCURRENCY_MINING:
            self._stats["mining_detected"] += 1
        elif abuse_type in (AbuseType.PORT_SCANNING, AbuseType.NETWORK_SCANNING):
            self._stats["scanning_detected"] += 1
        elif abuse_type == AbuseType.BOTNET_C2:
            self._stats["botnet_detected"] += 1

        # Trigger callback
        if self._on_alert:
            try:
                self._on_alert(alert)
            except Exception:
                pass

        # Auto-terminate on critical
        if (
            severity == AlertSeverity.CRITICAL
            and self.config.auto_terminate_on_critical
            and self._on_terminate
        ):
            try:
                job_id = state.get("job_id", "")
                if job_id:
                    success = self._on_terminate(job_id, title)
                    if success:
                        alert.job_terminated = True
                        alert.action_taken = "Job terminated"
                        self._stats["jobs_terminated"] += 1
            except Exception:
                pass

        return alert

    def get_alerts(
        self,
        tenant_id: Optional[str] = None,
        job_id: Optional[str] = None,
        abuse_type: Optional[AbuseType] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AbuseAlert]:
        """
        Get alert history.

        Args:
            tenant_id: Filter by tenant
            job_id: Filter by job
            abuse_type: Filter by abuse type
            since: Filter by time
            limit: Maximum results

        Returns:
            List of alerts
        """
        with self._lock:
            results = self._alerts.copy()

        if tenant_id:
            results = [a for a in results if a.tenant_id == tenant_id]
        if job_id:
            results = [a for a in results if a.job_id == job_id]
        if abuse_type:
            results = [a for a in results if a.abuse_type == abuse_type]
        if since:
            results = [a for a in results if a.detected_at >= since]

        results.sort(key=lambda a: a.detected_at, reverse=True)
        return results[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        with self._lock:
            return {
                **self._stats,
                "active_monitoring": len(self._monitoring),
                "alerts_in_history": len(self._alerts),
            }

    def get_risk_score(self, job_id: str) -> float:
        """
        Get aggregate risk score for a job.

        Args:
            job_id: Job identifier

        Returns:
            Risk score (0-100)
        """
        alerts = self.get_alerts(job_id=job_id)
        if not alerts:
            return 0.0

        # Use max risk score with decay for older alerts
        max_score = 0.0
        now = datetime.utcnow()

        for alert in alerts:
            age = (now - alert.detected_at).total_seconds()
            decay = math.exp(-age / 3600)  # 1-hour half-life
            adjusted_score = alert.risk_score * decay
            max_score = max(max_score, adjusted_score)

        return min(100.0, max_score)


def create_strict_detector() -> AbuseDetector:
    """
    Create a strict abuse detector with aggressive thresholds.

    Returns:
        Configured AbuseDetector
    """
    config = AbuseDetectorConfig(
        mining_cpu_threshold=70.0,
        mining_duration_threshold=30,
        scan_connections_threshold=30,
        scan_ports_threshold=10,
        auto_terminate_on_critical=True,
    )
    return AbuseDetector(config)


def create_permissive_detector() -> AbuseDetector:
    """
    Create a more permissive detector (for development/testing).

    Returns:
        Configured AbuseDetector
    """
    config = AbuseDetectorConfig(
        mining_cpu_threshold=95.0,
        mining_duration_threshold=300,
        scan_connections_threshold=100,
        scan_ports_threshold=50,
        auto_terminate_on_critical=False,
    )
    return AbuseDetector(config)
