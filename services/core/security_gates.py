# -*- coding: utf-8 -*-
"""
CI/CD Security Gates (Block 2.6).

Implements SAST/DAST security gates for CI/CD pipelines:
- Static Application Security Testing (SAST)
- Dynamic Application Security Testing (DAST)
- Dependency vulnerability scanning
- Security policy enforcement

DORA References:
    - Article 9: Protection and Prevention
    - Article 24: Digital Operational Resilience Testing
    - RTS CDR 2024/1774: ICT Risk Management Framework
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ScanType(Enum):
    """Security scan types."""
    SAST = "sast"
    DAST = "dast"
    SCA = "sca"  # Software Composition Analysis
    SECRET_SCAN = "secret_scan"
    CONTAINER_SCAN = "container_scan"
    IAC_SCAN = "iac_scan"  # Infrastructure as Code


class ScanStatus(Enum):
    """Scan status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class VulnerabilitySeverity(Enum):
    """Vulnerability severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class GateDecision(Enum):
    """Gate decision outcomes."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    MANUAL_REVIEW = "manual_review"


@dataclass
class SecurityScanResult:
    """Security scan result."""
    scan_id: str = ""
    scan_type: ScanType = ScanType.SAST
    status: ScanStatus = ScanStatus.COMPLETED

    # Context
    pipeline_id: str = ""
    commit_sha: str = ""
    branch: str = ""

    # Timing
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0

    # Results
    vulnerabilities_found: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    # Findings
    findings: List[Dict[str, Any]] = field(default_factory=list)

    # Tool info
    tool_name: str = ""
    tool_version: str = ""

    def __post_init__(self):
        if not self.scan_id:
            self.scan_id = f"SCAN-{uuid.uuid4().hex[:8].upper()}"
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc).isoformat()


@dataclass
class SecurityGate:
    """Security gate definition."""
    gate_id: str = ""
    name: str = ""
    description: str = ""

    # Required scans
    required_scans: List[ScanType] = field(default_factory=list)

    # Thresholds (fail if exceeded)
    max_critical: int = 0
    max_high: int = 0
    max_medium: int = 10
    max_low: int = 50

    # Policy
    block_on_critical: bool = True
    block_on_high: bool = True
    allow_exceptions: bool = True
    exception_approvers: List[str] = field(default_factory=list)

    # Status
    is_active: bool = True

    def __post_init__(self):
        if not self.gate_id:
            self.gate_id = f"GATE-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class GatePolicy:
    """Security gate policy."""
    policy_id: str = ""
    name: str = ""
    gates: List[SecurityGate] = field(default_factory=list)
    applies_to_branches: List[str] = field(default_factory=list)
    is_active: bool = True

    def __post_init__(self):
        if not self.policy_id:
            self.policy_id = f"SPOL-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class SecurityGatesConfig:
    """Configuration for SecurityGatesManager."""
    default_max_critical: int = 0
    default_max_high: int = 0
    enabled_scan_types: List[ScanType] = field(default_factory=lambda: [
        ScanType.SAST, ScanType.SCA, ScanType.SECRET_SCAN
    ])
    log_all_events: bool = True
    log_path: str = "logs/core/security_gates"
    alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


class SecurityGatesManager:
    """Security Gates Manager for CI/CD pipelines."""

    def __init__(self, config: Optional[SecurityGatesConfig] = None):
        self.config = config or SecurityGatesConfig()
        self._gates: Dict[str, SecurityGate] = {}
        self._policies: Dict[str, GatePolicy] = {}
        self._scans: Dict[str, SecurityScanResult] = {}
        self._lock = threading.RLock()
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)
        self._init_default_gates()
        logger.info("SecurityGatesManager initialized")

    def _init_default_gates(self) -> None:
        """Initialize default security gates."""
        self.create_gate(
            name="Production Security Gate",
            required_scans=[ScanType.SAST, ScanType.SCA, ScanType.SECRET_SCAN],
            max_critical=0,
            max_high=0,
            block_on_critical=True,
        )

    def create_gate(
        self,
        name: str,
        required_scans: List[ScanType],
        max_critical: int = 0,
        max_high: int = 0,
        max_medium: int = 10,
        block_on_critical: bool = True,
        block_on_high: bool = True,
    ) -> SecurityGate:
        """Create a security gate."""
        gate = SecurityGate(
            name=name,
            required_scans=required_scans,
            max_critical=max_critical,
            max_high=max_high,
            max_medium=max_medium,
            block_on_critical=block_on_critical,
            block_on_high=block_on_high,
        )
        with self._lock:
            self._gates[gate.gate_id] = gate
        return gate

    def record_scan(
        self,
        scan_type: ScanType,
        pipeline_id: str,
        commit_sha: str,
        findings: List[Dict[str, Any]],
        tool_name: str = "",
    ) -> SecurityScanResult:
        """Record a security scan result."""
        critical = sum(1 for f in findings if f.get("severity") == "critical")
        high = sum(1 for f in findings if f.get("severity") == "high")
        medium = sum(1 for f in findings if f.get("severity") == "medium")
        low = sum(1 for f in findings if f.get("severity") == "low")

        scan = SecurityScanResult(
            scan_type=scan_type,
            pipeline_id=pipeline_id,
            commit_sha=commit_sha,
            vulnerabilities_found=len(findings),
            critical_count=critical,
            high_count=high,
            medium_count=medium,
            low_count=low,
            findings=findings,
            tool_name=tool_name,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

        with self._lock:
            self._scans[scan.scan_id] = scan

        return scan

    def evaluate_gate(
        self,
        gate_id: str,
        scan_results: List[SecurityScanResult],
    ) -> Dict[str, Any]:
        """Evaluate a security gate against scan results."""
        with self._lock:
            if gate_id not in self._gates:
                return {"decision": GateDecision.FAIL.value, "error": "Gate not found"}
            gate = self._gates[gate_id]

        total_critical = sum(s.critical_count for s in scan_results)
        total_high = sum(s.high_count for s in scan_results)
        total_medium = sum(s.medium_count for s in scan_results)

        failures = []
        if total_critical > gate.max_critical:
            failures.append(f"Critical: {total_critical} > {gate.max_critical}")
        if total_high > gate.max_high:
            failures.append(f"High: {total_high} > {gate.max_high}")
        if total_medium > gate.max_medium:
            failures.append(f"Medium: {total_medium} > {gate.max_medium}")

        if failures:
            if total_critical > 0 and gate.block_on_critical:
                decision = GateDecision.FAIL
            elif total_high > 0 and gate.block_on_high:
                decision = GateDecision.FAIL
            else:
                decision = GateDecision.WARN
        else:
            decision = GateDecision.PASS

        return {
            "gate_id": gate_id,
            "gate_name": gate.name,
            "decision": decision.value,
            "summary": {
                "critical": total_critical,
                "high": total_high,
                "medium": total_medium,
            },
            "thresholds": {
                "max_critical": gate.max_critical,
                "max_high": gate.max_high,
                "max_medium": gate.max_medium,
            },
            "failures": failures,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_scan_summary(self, pipeline_id: str) -> Dict[str, Any]:
        """Get scan summary for a pipeline."""
        with self._lock:
            scans = [s for s in self._scans.values() if s.pipeline_id == pipeline_id]

        return {
            "pipeline_id": pipeline_id,
            "total_scans": len(scans),
            "vulnerabilities": {
                "critical": sum(s.critical_count for s in scans),
                "high": sum(s.high_count for s in scans),
                "medium": sum(s.medium_count for s in scans),
                "low": sum(s.low_count for s in scans),
            },
            "scan_types": [s.scan_type.value for s in scans],
        }


def create_security_gates_manager(
    config: Optional[SecurityGatesConfig] = None,
) -> SecurityGatesManager:
    """Create a SecurityGatesManager instance."""
    return SecurityGatesManager(config=config)
