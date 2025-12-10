# -*- coding: utf-8 -*-
"""
DORA Detection Module (Article 10).

Regulation (EU) 2022/2554 Article 10 defines detection requirements:
    - Mechanisms to promptly detect anomalous activities
    - Including network performance issues and ICT-related incidents
    - Identify potential single points of failure
    - Multiple layers of control
    - Detection thresholds and criteria for ICT-related incidents

This module provides:
    1. AnomalyDetector - Anomalous activity detection
    2. IncidentDetector - ICT incident detection
    3. PerformanceMonitor - Network and system performance monitoring
    4. SinglePointOfFailureDetector - SPOF identification
    5. DORADetection - Main detection orchestrator

References:
    - Article 10 DORA: https://www.digital-operational-resilience-act.com/Article_10.html
    - RTS on ICT Risk Management: CDR 2024/1774
    - NIST CSF 2.0 - Detect Function
    - MITRE ATT&CK Framework
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
from typing import Any, Callable, Dict, List, Optional, Tuple
import statistics

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class AnomalyType(Enum):
    """Anomaly types per DORA Article 10."""
    NETWORK_TRAFFIC = "network_traffic"
    SYSTEM_PERFORMANCE = "system_performance"
    USER_BEHAVIOR = "user_behavior"
    DATA_ACCESS = "data_access"
    AUTHENTICATION = "authentication"
    PROCESS_EXECUTION = "process_execution"
    FILE_ACTIVITY = "file_activity"
    CONFIGURATION_CHANGE = "configuration_change"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"
    ESCALATED = "escalated"


class DetectionMethod(Enum):
    """Detection methods."""
    THRESHOLD_BASED = "threshold_based"
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"
    RULE_BASED = "rule_based"
    SIGNATURE_BASED = "signature_based"
    BEHAVIORAL = "behavioral"
    CORRELATION = "correlation"


class IncidentCategory(Enum):
    """ICT incident categories."""
    SECURITY = "security"
    AVAILABILITY = "availability"
    PERFORMANCE = "performance"
    DATA_INTEGRITY = "data_integrity"
    CONFIGURATION = "configuration"
    THIRD_PARTY = "third_party"
    NETWORK = "network"
    APPLICATION = "application"


class MonitoringStatus(Enum):
    """Monitoring system status."""
    ACTIVE = "active"
    PAUSED = "paused"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"
    FAILED = "failed"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DetectionRule:
    """
    Detection rule per DORA Article 10.

    Defines criteria for detecting anomalies and incidents.
    """
    rule_id: str = ""
    name: str = ""
    description: str = ""
    category: IncidentCategory = IncidentCategory.SECURITY

    # Detection method
    method: DetectionMethod = DetectionMethod.THRESHOLD_BASED
    detection_logic: str = ""

    # Thresholds
    threshold_value: float = 0.0
    threshold_operator: str = ""  # "gt", "lt", "eq", "gte", "lte", "between"
    threshold_upper: Optional[float] = None

    # Time window
    time_window_seconds: int = 300
    aggregation_method: str = ""  # "count", "sum", "avg", "max", "min"

    # Severity mapping
    severity_mapping: Dict[str, AlertSeverity] = field(default_factory=dict)
    default_severity: AlertSeverity = AlertSeverity.MEDIUM

    # Correlation
    correlate_with_rules: List[str] = field(default_factory=list)
    correlation_window_seconds: int = 600

    # Response
    auto_response_actions: List[str] = field(default_factory=list)
    escalation_path: List[str] = field(default_factory=list)

    # Status
    is_active: bool = True
    last_triggered: Optional[str] = None
    trigger_count: int = 0

    # Metadata
    created_date: str = ""
    created_by: str = ""
    mitre_attack_mapping: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.rule_id:
            self.rule_id = f"RULE-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_date:
            self.created_date = datetime.now(timezone.utc).isoformat()


@dataclass
class DetectionAlert:
    """
    Detection alert.

    Documents detected anomalies or potential incidents.
    """
    alert_id: str = ""
    rule_id: str = ""
    rule_name: str = ""

    # Classification
    severity: AlertSeverity = AlertSeverity.MEDIUM
    category: IncidentCategory = IncidentCategory.SECURITY
    anomaly_type: AnomalyType = AnomalyType.SYSTEM_PERFORMANCE

    # Details
    title: str = ""
    description: str = ""
    detection_method: DetectionMethod = DetectionMethod.THRESHOLD_BASED

    # Context
    source_system: str = ""
    source_ip: str = ""
    target_system: str = ""
    target_ip: str = ""
    user_id: str = ""

    # Metrics
    observed_value: float = 0.0
    threshold_value: float = 0.0
    deviation_percentage: float = 0.0
    confidence_score: float = 0.0

    # Status
    status: AlertStatus = AlertStatus.NEW
    acknowledged_by: str = ""
    acknowledged_time: Optional[str] = None
    resolved_by: str = ""
    resolved_time: Optional[str] = None

    # Correlation
    correlated_alerts: List[str] = field(default_factory=list)
    incident_id: str = ""

    # Evidence
    raw_data: Dict[str, Any] = field(default_factory=dict)
    indicators: List[str] = field(default_factory=list)

    # Timestamps
    detection_time: str = ""
    event_time: str = ""

    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = f"ALERT-{uuid.uuid4().hex[:8].upper()}"
        if not self.detection_time:
            self.detection_time = datetime.now(timezone.utc).isoformat()
        if not self.event_time:
            self.event_time = self.detection_time


@dataclass
class PerformanceMetric:
    """
    Performance metric for monitoring.
    """
    metric_id: str = ""
    metric_name: str = ""
    system_id: str = ""
    component: str = ""

    # Values
    current_value: float = 0.0
    baseline_value: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    avg_value: float = 0.0

    # Thresholds
    warning_threshold: float = 70.0
    critical_threshold: float = 90.0

    # Status
    is_anomalous: bool = False
    anomaly_score: float = 0.0

    # Timestamp
    timestamp: str = ""

    def __post_init__(self):
        if not self.metric_id:
            self.metric_id = f"PERF-{uuid.uuid4().hex[:8].upper()}"
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class SinglePointOfFailure:
    """
    Single point of failure identification.

    Per DORA Article 10(2), entities must identify potential SPOFs.
    """
    spof_id: str = ""
    name: str = ""
    description: str = ""

    # Component details
    system_id: str = ""
    component_type: str = ""  # "hardware", "software", "network", "personnel"

    # Impact
    affected_functions: List[str] = field(default_factory=list)
    affected_systems: List[str] = field(default_factory=list)
    failure_impact: str = ""
    criticality: str = ""  # "critical", "high", "medium", "low"

    # Redundancy
    redundancy_exists: bool = False
    redundancy_type: str = ""  # "active-active", "active-passive", "none"
    failover_time_minutes: float = 0.0

    # Mitigation
    mitigation_plan: str = ""
    mitigation_status: str = ""  # "implemented", "planned", "none"
    target_resolution_date: Optional[str] = None

    # Assessment
    last_assessed: str = ""
    assessed_by: str = ""

    def __post_init__(self):
        if not self.spof_id:
            self.spof_id = f"SPOF-{uuid.uuid4().hex[:8].upper()}"
        if not self.last_assessed:
            self.last_assessed = datetime.now(timezone.utc).isoformat()


@dataclass
class DetectionCapability:
    """
    Detection capability assessment.
    """
    capability_id: str = ""
    name: str = ""
    description: str = ""
    detection_layer: str = ""  # "network", "endpoint", "application", "data"

    # Coverage
    threat_categories_covered: List[str] = field(default_factory=list)
    systems_covered: List[str] = field(default_factory=list)
    coverage_percentage: float = 0.0

    # Effectiveness
    detection_rate_percentage: float = 0.0
    false_positive_rate_percentage: float = 0.0
    mean_time_to_detect_minutes: float = 0.0

    # Status
    status: MonitoringStatus = MonitoringStatus.ACTIVE
    last_verified: Optional[str] = None

    def __post_init__(self):
        if not self.capability_id:
            self.capability_id = f"CAP-{uuid.uuid4().hex[:8].upper()}"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DetectionConfig:
    """Configuration for DORA Detection module."""
    # Alert thresholds
    default_alert_threshold: float = 80.0
    critical_alert_threshold: float = 95.0

    # Time windows
    default_time_window_seconds: int = 300
    correlation_window_seconds: int = 600

    # Statistical detection
    statistical_baseline_samples: int = 100
    statistical_deviation_threshold: float = 3.0  # Standard deviations

    # Rate limiting
    max_alerts_per_minute: int = 100
    alert_deduplication_window_seconds: int = 60

    # Retention
    alert_retention_days: int = 90
    metric_retention_hours: int = 24

    # Performance monitoring
    performance_check_interval_seconds: int = 60

    # Logging
    log_all_events: bool = True
    log_path: str = "logs/dora/detection"

    # Callbacks
    alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Main Detection Class
# =============================================================================

class DORADetection:
    """
    DORA Article 10 compliant Detection module.

    Manages:
    - Detection rules and thresholds
    - Anomaly detection
    - Performance monitoring
    - Alert management
    - Single point of failure identification

    Usage:
        config = DetectionConfig()
        detection = DORADetection(config)

        # Create detection rule
        rule = detection.create_detection_rule(
            name="High CPU Usage",
            category=IncidentCategory.PERFORMANCE,
            threshold_value=90.0,
        )

        # Monitor metric
        alert = detection.check_metric(
            metric_name="cpu_usage",
            current_value=95.0,
            system_id="trading-server-01",
        )
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        """Initialize Detection module."""
        self.config = config or DetectionConfig()

        # Data stores
        self._rules: Dict[str, DetectionRule] = {}
        self._alerts: List[DetectionAlert] = []
        self._metrics: Dict[str, List[PerformanceMetric]] = {}  # system_id -> metrics
        self._spofs: Dict[str, SinglePointOfFailure] = {}
        self._capabilities: Dict[str, DetectionCapability] = {}

        # Baselines for statistical detection
        self._baselines: Dict[str, List[float]] = {}  # metric_key -> values

        # Thread safety
        self._lock = threading.RLock()

        # Rate limiting
        self._alert_timestamps: List[str] = []

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("DORADetection initialized")

    # =========================================================================
    # Detection Rules
    # =========================================================================

    def create_detection_rule(
        self,
        name: str,
        category: IncidentCategory,
        threshold_value: float,
        description: str = "",
        method: DetectionMethod = DetectionMethod.THRESHOLD_BASED,
        threshold_operator: str = "gt",
        time_window_seconds: Optional[int] = None,
        default_severity: AlertSeverity = AlertSeverity.MEDIUM,
        auto_response_actions: Optional[List[str]] = None,
        mitre_attack_mapping: Optional[List[str]] = None,
    ) -> DetectionRule:
        """
        Create a detection rule.

        Per Article 10, entities must have mechanisms to detect anomalies.

        Args:
            name: Rule name
            category: Incident category
            threshold_value: Detection threshold
            description: Rule description
            method: Detection method
            threshold_operator: Threshold operator
            time_window_seconds: Time window for aggregation
            default_severity: Default alert severity
            auto_response_actions: Automatic response actions
            mitre_attack_mapping: MITRE ATT&CK technique IDs

        Returns:
            Created DetectionRule
        """
        rule = DetectionRule(
            name=name,
            category=category,
            threshold_value=threshold_value,
            description=description,
            method=method,
            threshold_operator=threshold_operator,
            time_window_seconds=time_window_seconds or self.config.default_time_window_seconds,
            default_severity=default_severity,
            auto_response_actions=auto_response_actions or [],
            mitre_attack_mapping=mitre_attack_mapping or [],
        )

        with self._lock:
            self._rules[rule.rule_id] = rule

        self._log_event("rule_created", {
            "rule_id": rule.rule_id,
            "name": name,
            "threshold": threshold_value,
        })

        return rule

    def get_detection_rule(self, rule_id: str) -> Optional[DetectionRule]:
        """Get detection rule by ID."""
        with self._lock:
            return self._rules.get(rule_id)

    def get_active_rules(self) -> List[DetectionRule]:
        """Get all active detection rules."""
        with self._lock:
            return [r for r in self._rules.values() if r.is_active]

    def update_rule_status(
        self,
        rule_id: str,
        is_active: bool,
    ) -> Optional[DetectionRule]:
        """Enable or disable a rule."""
        with self._lock:
            if rule_id not in self._rules:
                return None

            self._rules[rule_id].is_active = is_active
            return self._rules[rule_id]

    # =========================================================================
    # Metric Monitoring and Anomaly Detection
    # =========================================================================

    def check_metric(
        self,
        metric_name: str,
        current_value: float,
        system_id: str,
        component: str = "",
    ) -> Optional[DetectionAlert]:
        """
        Check a metric against detection rules.

        Args:
            metric_name: Metric name
            current_value: Current metric value
            system_id: System identifier
            component: Component name

        Returns:
            DetectionAlert if threshold exceeded, None otherwise
        """
        metric_key = f"{system_id}:{metric_name}"

        # Update baseline for statistical detection
        self._update_baseline(metric_key, current_value)

        # Store metric
        metric = PerformanceMetric(
            metric_name=metric_name,
            system_id=system_id,
            component=component,
            current_value=current_value,
            baseline_value=self._get_baseline_average(metric_key),
        )

        with self._lock:
            if system_id not in self._metrics:
                self._metrics[system_id] = []
            self._metrics[system_id].append(metric)

            # Cleanup old metrics
            cutoff = (
                datetime.now(timezone.utc) - timedelta(hours=self.config.metric_retention_hours)
            ).isoformat()
            self._metrics[system_id] = [
                m for m in self._metrics[system_id] if m.timestamp > cutoff
            ]

        # Check against rules
        for rule in self._rules.values():
            if not rule.is_active:
                continue

            if self._evaluate_rule(rule, current_value, metric_key):
                alert = self._create_alert(
                    rule=rule,
                    observed_value=current_value,
                    source_system=system_id,
                    metric_name=metric_name,
                )
                return alert

        # Check for statistical anomaly
        anomaly_score = self._calculate_anomaly_score(metric_key, current_value)
        if anomaly_score > self.config.statistical_deviation_threshold:
            alert = self._create_anomaly_alert(
                metric_name=metric_name,
                current_value=current_value,
                anomaly_score=anomaly_score,
                system_id=system_id,
            )
            return alert

        return None

    def _evaluate_rule(
        self,
        rule: DetectionRule,
        value: float,
        metric_key: str,
    ) -> bool:
        """Evaluate a detection rule against a value."""
        threshold = rule.threshold_value
        operator = rule.threshold_operator

        if operator == "gt":
            return value > threshold
        elif operator == "gte":
            return value >= threshold
        elif operator == "lt":
            return value < threshold
        elif operator == "lte":
            return value <= threshold
        elif operator == "eq":
            return value == threshold
        elif operator == "between" and rule.threshold_upper is not None:
            return threshold <= value <= rule.threshold_upper

        return False

    def _update_baseline(self, metric_key: str, value: float) -> None:
        """Update baseline for a metric."""
        with self._lock:
            if metric_key not in self._baselines:
                self._baselines[metric_key] = []

            self._baselines[metric_key].append(value)

            # Keep only recent samples
            max_samples = self.config.statistical_baseline_samples
            if len(self._baselines[metric_key]) > max_samples:
                self._baselines[metric_key] = self._baselines[metric_key][-max_samples:]

    def _get_baseline_average(self, metric_key: str) -> float:
        """Get baseline average for a metric."""
        with self._lock:
            values = self._baselines.get(metric_key, [])
            return statistics.mean(values) if values else 0.0

    def _calculate_anomaly_score(self, metric_key: str, value: float) -> float:
        """Calculate anomaly score using z-score."""
        with self._lock:
            values = self._baselines.get(metric_key, [])

            if len(values) < 10:
                return 0.0

            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 1.0

            if stdev == 0:
                return 0.0

            z_score = abs((value - mean) / stdev)
            return z_score

    # =========================================================================
    # Alert Management
    # =========================================================================

    def _create_alert(
        self,
        rule: DetectionRule,
        observed_value: float,
        source_system: str,
        metric_name: str,
    ) -> Optional[DetectionAlert]:
        """Create an alert from a triggered rule."""
        # Rate limiting
        if not self._check_rate_limit():
            logger.warning("Alert rate limit exceeded")
            return None

        # Deduplication
        if self._is_duplicate_alert(rule.rule_id, source_system):
            return None

        deviation = 0.0
        if rule.threshold_value > 0:
            deviation = ((observed_value - rule.threshold_value) / rule.threshold_value) * 100

        alert = DetectionAlert(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.default_severity,
            category=rule.category,
            title=f"{rule.name}: {metric_name} exceeded threshold",
            description=f"Value {observed_value} exceeded threshold {rule.threshold_value}",
            detection_method=rule.method,
            source_system=source_system,
            observed_value=observed_value,
            threshold_value=rule.threshold_value,
            deviation_percentage=deviation,
        )

        with self._lock:
            self._alerts.append(alert)
            rule.last_triggered = alert.detection_time
            rule.trigger_count += 1

        # Execute auto-response actions
        if rule.auto_response_actions:
            self._execute_auto_response(alert, rule.auto_response_actions)

        # Callback
        if self.config.alert_callback:
            try:
                self.config.alert_callback("detection_alert", asdict(alert))
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        self._log_event("alert_created", {
            "alert_id": alert.alert_id,
            "rule_id": rule.rule_id,
            "severity": alert.severity.value,
        })

        logger.warning(f"Alert: {alert.title} ({alert.severity.value})")
        return alert

    def _create_anomaly_alert(
        self,
        metric_name: str,
        current_value: float,
        anomaly_score: float,
        system_id: str,
    ) -> DetectionAlert:
        """Create an alert for statistical anomaly."""
        severity = AlertSeverity.MEDIUM
        if anomaly_score > 5:
            severity = AlertSeverity.HIGH
        elif anomaly_score > 7:
            severity = AlertSeverity.CRITICAL

        alert = DetectionAlert(
            rule_id="STATISTICAL",
            rule_name="Statistical Anomaly Detection",
            severity=severity,
            category=IncidentCategory.PERFORMANCE,
            anomaly_type=AnomalyType.SYSTEM_PERFORMANCE,
            title=f"Anomaly detected: {metric_name}",
            description=f"Statistical anomaly detected with z-score {anomaly_score:.2f}",
            detection_method=DetectionMethod.STATISTICAL,
            source_system=system_id,
            observed_value=current_value,
            confidence_score=min(1.0, anomaly_score / 10),
        )

        with self._lock:
            self._alerts.append(alert)

        if self.config.alert_callback:
            try:
                self.config.alert_callback("anomaly_alert", asdict(alert))
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        return alert

    def _check_rate_limit(self) -> bool:
        """Check if alert rate limit allows new alert."""
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(minutes=1)).isoformat()

        with self._lock:
            self._alert_timestamps = [t for t in self._alert_timestamps if t > cutoff]

            if len(self._alert_timestamps) >= self.config.max_alerts_per_minute:
                return False

            self._alert_timestamps.append(now.isoformat())
            return True

    def _is_duplicate_alert(self, rule_id: str, source_system: str) -> bool:
        """Check for duplicate alert within deduplication window."""
        cutoff = (
            datetime.now(timezone.utc) -
            timedelta(seconds=self.config.alert_deduplication_window_seconds)
        ).isoformat()

        with self._lock:
            for alert in reversed(self._alerts[-100:]):
                if alert.detection_time < cutoff:
                    break
                if alert.rule_id == rule_id and alert.source_system == source_system:
                    return True

        return False

    def _execute_auto_response(
        self,
        alert: DetectionAlert,
        actions: List[str],
    ) -> None:
        """Execute automatic response actions."""
        for action in actions:
            logger.info(f"Executing auto-response: {action} for alert {alert.alert_id}")
            # In production, this would trigger actual response actions

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
    ) -> Optional[DetectionAlert]:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.status = AlertStatus.ACKNOWLEDGED
                    alert.acknowledged_by = acknowledged_by
                    alert.acknowledged_time = datetime.now(timezone.utc).isoformat()
                    return alert
        return None

    def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str,
        resolution_notes: str = "",
    ) -> Optional[DetectionAlert]:
        """Resolve an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.status = AlertStatus.RESOLVED
                    alert.resolved_by = resolved_by
                    alert.resolved_time = datetime.now(timezone.utc).isoformat()
                    return alert
        return None

    def get_alerts(
        self,
        status: Optional[AlertStatus] = None,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100,
    ) -> List[DetectionAlert]:
        """Get alerts with optional filters."""
        with self._lock:
            alerts = list(self._alerts)

            if status:
                alerts = [a for a in alerts if a.status == status]

            if severity:
                alerts = [a for a in alerts if a.severity == severity]

            return alerts[-limit:]

    def get_open_alerts(self) -> List[DetectionAlert]:
        """Get all open (unresolved) alerts."""
        with self._lock:
            return [
                a for a in self._alerts
                if a.status in (AlertStatus.NEW, AlertStatus.ACKNOWLEDGED, AlertStatus.INVESTIGATING)
            ]

    def get_critical_alerts(self) -> List[DetectionAlert]:
        """Get critical and high severity open alerts."""
        with self._lock:
            return [
                a for a in self._alerts
                if a.severity in (AlertSeverity.CRITICAL, AlertSeverity.HIGH)
                and a.status not in (AlertStatus.RESOLVED, AlertStatus.FALSE_POSITIVE)
            ]

    # =========================================================================
    # Single Point of Failure Detection (Article 10(2))
    # =========================================================================

    def identify_spof(
        self,
        name: str,
        system_id: str,
        component_type: str,
        description: str = "",
        affected_functions: Optional[List[str]] = None,
        affected_systems: Optional[List[str]] = None,
        failure_impact: str = "",
        criticality: str = "medium",
        redundancy_exists: bool = False,
        redundancy_type: str = "none",
        mitigation_plan: str = "",
    ) -> SinglePointOfFailure:
        """
        Identify a single point of failure.

        Per Article 10(2), entities must identify potential SPOFs.

        Args:
            name: SPOF name
            system_id: System ID
            component_type: Component type
            description: Description
            affected_functions: Affected business functions
            affected_systems: Affected systems
            failure_impact: Impact description
            criticality: Criticality level
            redundancy_exists: Whether redundancy exists
            redundancy_type: Type of redundancy
            mitigation_plan: Mitigation plan

        Returns:
            Identified SinglePointOfFailure
        """
        spof = SinglePointOfFailure(
            name=name,
            system_id=system_id,
            component_type=component_type,
            description=description,
            affected_functions=affected_functions or [],
            affected_systems=affected_systems or [],
            failure_impact=failure_impact,
            criticality=criticality,
            redundancy_exists=redundancy_exists,
            redundancy_type=redundancy_type,
            mitigation_plan=mitigation_plan,
            mitigation_status="planned" if mitigation_plan else "none",
        )

        with self._lock:
            self._spofs[spof.spof_id] = spof

        self._log_event("spof_identified", {
            "spof_id": spof.spof_id,
            "name": name,
            "criticality": criticality,
        })

        logger.info(f"SPOF identified: {name} ({criticality})")
        return spof

    def get_spof(self, spof_id: str) -> Optional[SinglePointOfFailure]:
        """Get SPOF by ID."""
        with self._lock:
            return self._spofs.get(spof_id)

    def get_unmitigated_spofs(self) -> List[SinglePointOfFailure]:
        """Get SPOFs without mitigation."""
        with self._lock:
            return [
                s for s in self._spofs.values()
                if s.mitigation_status != "implemented" and not s.redundancy_exists
            ]

    def get_critical_spofs(self) -> List[SinglePointOfFailure]:
        """Get critical SPOFs."""
        with self._lock:
            return [s for s in self._spofs.values() if s.criticality == "critical"]

    def update_spof_mitigation(
        self,
        spof_id: str,
        mitigation_status: str,
        mitigation_plan: str = "",
    ) -> Optional[SinglePointOfFailure]:
        """Update SPOF mitigation status."""
        with self._lock:
            if spof_id not in self._spofs:
                return None

            spof = self._spofs[spof_id]
            spof.mitigation_status = mitigation_status
            if mitigation_plan:
                spof.mitigation_plan = mitigation_plan

            return spof

    # =========================================================================
    # Detection Capabilities
    # =========================================================================

    def register_detection_capability(
        self,
        name: str,
        detection_layer: str,
        description: str = "",
        threat_categories_covered: Optional[List[str]] = None,
        systems_covered: Optional[List[str]] = None,
        coverage_percentage: float = 0.0,
    ) -> DetectionCapability:
        """Register a detection capability."""
        capability = DetectionCapability(
            name=name,
            detection_layer=detection_layer,
            description=description,
            threat_categories_covered=threat_categories_covered or [],
            systems_covered=systems_covered or [],
            coverage_percentage=coverage_percentage,
        )

        with self._lock:
            self._capabilities[capability.capability_id] = capability

        return capability

    def get_detection_capabilities(self) -> List[DetectionCapability]:
        """Get all detection capabilities."""
        with self._lock:
            return list(self._capabilities.values())

    def assess_detection_coverage(self) -> Dict[str, Any]:
        """Assess overall detection coverage."""
        with self._lock:
            capabilities = list(self._capabilities.values())

            if not capabilities:
                return {"coverage": 0.0, "gaps": ["No detection capabilities registered"]}

            layers_covered = set(c.detection_layer for c in capabilities)
            avg_coverage = sum(c.coverage_percentage for c in capabilities) / len(capabilities)

            gaps = []
            required_layers = {"network", "endpoint", "application", "data"}
            missing_layers = required_layers - layers_covered
            if missing_layers:
                gaps.append(f"Missing detection layers: {', '.join(missing_layers)}")

            return {
                "total_capabilities": len(capabilities),
                "layers_covered": list(layers_covered),
                "average_coverage_percentage": avg_coverage,
                "gaps": gaps,
            }

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_detection_summary(self) -> Dict[str, Any]:
        """Get detection status summary."""
        with self._lock:
            open_alerts = self.get_open_alerts()
            critical_alerts = self.get_critical_alerts()
            unmitigated_spofs = self.get_unmitigated_spofs()
            critical_spofs = self.get_critical_spofs()

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "rules": {
                    "total": len(self._rules),
                    "active": sum(1 for r in self._rules.values() if r.is_active),
                },
                "alerts": {
                    "total": len(self._alerts),
                    "open": len(open_alerts),
                    "critical_high": len(critical_alerts),
                    "by_severity": {
                        s.value: sum(1 for a in self._alerts if a.severity == s)
                        for s in AlertSeverity
                    },
                    "by_status": {
                        s.value: sum(1 for a in self._alerts if a.status == s)
                        for s in AlertStatus
                    },
                },
                "spofs": {
                    "total": len(self._spofs),
                    "unmitigated": len(unmitigated_spofs),
                    "critical": len(critical_spofs),
                },
                "capabilities": self.assess_detection_coverage(),
                "metrics_monitored": len(self._metrics),
            }

    def export_detection_report(self) -> Dict[str, Any]:
        """Export complete detection report."""
        with self._lock:
            return {
                "export_date": datetime.now(timezone.utc).isoformat(),
                "article_reference": "Article 10",
                "summary": self.get_detection_summary(),
                "rules": [asdict(r) for r in self._rules.values()],
                "alerts": [asdict(a) for a in self._alerts[-1000:]],
                "spofs": [asdict(s) for s in self._spofs.values()],
                "capabilities": [asdict(c) for c in self._capabilities.values()],
            }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event."""
        if not self.config.log_all_events:
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"detection_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_detection(
    config: Optional[DetectionConfig] = None,
) -> DORADetection:
    """
    Create a DORADetection instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORADetection instance
    """
    return DORADetection(config=config)
