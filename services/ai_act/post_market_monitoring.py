# -*- coding: utf-8 -*-
"""
AI Act Post-Market Monitoring System (Article 72).

EU AI Act Article 72 requires providers of high-risk AI systems to establish
a post-market monitoring system to collect, document and analyze relevant data.

This module provides:
1. PostMarketMonitoringSystem - Main monitoring orchestrator
2. PerformanceMonitor - Performance drift detection
3. IncidentTracker - Incident tracking and reporting
4. FeedbackCollector - Deployer/user feedback collection
5. ComplianceReporter - Periodic compliance reporting

Article 72 Requirements Implemented:
    - (1) Post-market monitoring system establishment
    - (2) Active and systematic data collection
    - (3) Performance monitoring against declared specifications
    - (4) Identification of corrective actions
    - (5) Proportionate to nature and risk of AI system

Related Articles:
    - Article 73: Reporting of serious incidents
    - Article 74: Market surveillance and control

References:
    - Article 72: https://artificialintelligenceact.eu/article/72/
    - Article 73: https://artificialintelligenceact.eu/article/73/
"""

from __future__ import annotations

import json
import logging
import statistics
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class MonitoringMetricType(Enum):
    """Types of metrics monitored post-market."""

    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"
    FAIRNESS = "fairness"
    DRIFT = "drift"
    SAFETY = "safety"
    USER_SATISFACTION = "user_satisfaction"
    CUSTOM = "custom"


class DriftType(Enum):
    """Types of performance drift."""

    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    MODEL_DRIFT = "model_drift"
    PREDICTION_DRIFT = "prediction_drift"
    FEATURE_DRIFT = "feature_drift"


class DriftSeverity(Enum):
    """Severity of detected drift."""

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class IncidentSeverity(Enum):
    """Severity of incidents per Article 73."""

    MINOR = 1
    MODERATE = 2
    MAJOR = 3
    SERIOUS = 4  # Requires reporting per Article 73


class IncidentStatus(Enum):
    """Status of incident handling."""

    REPORTED = "reported"
    INVESTIGATING = "investigating"
    ROOT_CAUSE_IDENTIFIED = "root_cause_identified"
    CORRECTIVE_ACTION = "corrective_action"
    RESOLVED = "resolved"
    CLOSED = "closed"


class FeedbackType(Enum):
    """Types of feedback collected."""

    BUG_REPORT = "bug_report"
    PERFORMANCE_ISSUE = "performance_issue"
    FEATURE_REQUEST = "feature_request"
    SAFETY_CONCERN = "safety_concern"
    GENERAL_FEEDBACK = "general_feedback"
    INCIDENT_REPORT = "incident_report"


class AlertPriority(Enum):
    """Priority of monitoring alerts."""

    INFO = 1
    WARNING = 2
    HIGH = 3
    CRITICAL = 4


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class MonitoringMetric:
    """
    Monitored metric definition.

    Represents a metric being tracked post-market.
    """

    metric_id: str
    name: str
    metric_type: MonitoringMetricType
    description: str

    # Baseline (from pre-market/deployment)
    baseline_value: float = 0.0
    baseline_std: float = 0.0
    baseline_samples: int = 0

    # Thresholds
    warning_threshold: float = 0.0
    critical_threshold: float = 0.0
    threshold_direction: str = "below"  # "below", "above", "deviation"

    # Current state
    current_value: Optional[float] = None
    current_std: Optional[float] = None
    current_samples: int = 0
    last_updated: str = ""

    # Drift detection
    drift_score: float = 0.0
    drift_severity: DriftSeverity = DriftSeverity.NONE

    # Metadata
    unit: str = ""
    collection_frequency: str = "hourly"  # hourly, daily, weekly
    created_at: str = ""

    def __post_init__(self):
        if not self.metric_id:
            self.metric_id = f"PMON-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class DriftDetectionResult:
    """
    Result of drift detection analysis.

    Documents detected performance drift.
    """

    detection_id: str
    metric_id: str
    drift_type: DriftType
    severity: DriftSeverity

    # Detection details
    detection_time: str = ""
    baseline_value: float = 0.0
    current_value: float = 0.0
    deviation: float = 0.0
    deviation_percent: float = 0.0

    # Statistical analysis
    p_value: Optional[float] = None
    test_statistic: Optional[float] = None
    test_method: str = ""

    # Impact assessment
    affected_functionality: List[str] = field(default_factory=list)
    potential_impact: str = ""

    # Actions
    recommended_actions: List[str] = field(default_factory=list)
    action_taken: str = ""
    action_date: Optional[str] = None

    # Status
    acknowledged: bool = False
    acknowledged_by: str = ""
    resolved: bool = False
    resolution_date: Optional[str] = None

    def __post_init__(self):
        if not self.detection_id:
            self.detection_id = f"DRIFT-{uuid.uuid4().hex[:8].upper()}"
        if not self.detection_time:
            self.detection_time = datetime.now(timezone.utc).isoformat()


@dataclass
class Incident:
    """
    Incident record per Article 73.

    Documents incidents and malfunctioning requiring investigation.
    """

    incident_id: str
    severity: IncidentSeverity
    title: str
    description: str

    # Classification
    incident_type: str = ""  # malfunction, safety_event, data_breach, etc.
    affected_component: str = ""

    # Timing
    occurred_at: str = ""
    detected_at: str = ""
    reported_at: str = ""

    # Impact
    affected_users: int = 0
    affected_operations: List[str] = field(default_factory=list)
    financial_impact: Optional[float] = None
    safety_impact: str = ""

    # Investigation
    status: IncidentStatus = IncidentStatus.REPORTED
    root_cause: str = ""
    contributing_factors: List[str] = field(default_factory=list)
    investigation_notes: str = ""

    # Corrective actions
    immediate_actions: List[str] = field(default_factory=list)
    corrective_actions: List[str] = field(default_factory=list)
    preventive_actions: List[str] = field(default_factory=list)

    # Regulatory reporting (Article 73)
    requires_regulatory_report: bool = False
    regulatory_report_submitted: bool = False
    regulatory_report_date: Optional[str] = None
    regulatory_authority: str = ""

    # Resolution
    resolved_at: Optional[str] = None
    resolution_details: str = ""
    lessons_learned: str = ""

    # Metadata
    reporter: str = ""
    assignee: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.incident_id:
            self.incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.occurred_at:
            self.occurred_at = self.created_at
        if not self.detected_at:
            self.detected_at = self.created_at


@dataclass
class Feedback:
    """
    User/deployer feedback record.

    Collects feedback from system users and deployers.
    """

    feedback_id: str
    feedback_type: FeedbackType
    subject: str
    description: str

    # Source
    source_type: str = ""  # deployer, user, operator, automated
    source_id: str = ""
    source_contact: str = ""

    # Details
    affected_functionality: str = ""
    frequency: str = ""  # one_time, intermittent, persistent
    reproducible: Optional[bool] = None
    steps_to_reproduce: List[str] = field(default_factory=list)

    # Priority
    priority: AlertPriority = AlertPriority.INFO
    urgency: str = "low"  # low, medium, high, immediate

    # Response
    acknowledged: bool = False
    acknowledged_at: Optional[str] = None
    response: str = ""
    response_date: Optional[str] = None

    # Resolution
    resolved: bool = False
    resolution: str = ""
    resolved_at: Optional[str] = None

    # Related items
    related_incidents: List[str] = field(default_factory=list)
    related_changes: List[str] = field(default_factory=list)

    # Metadata
    submitted_at: str = ""
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.feedback_id:
            self.feedback_id = f"FB-{uuid.uuid4().hex[:8].upper()}"
        if not self.submitted_at:
            self.submitted_at = datetime.now(timezone.utc).isoformat()


@dataclass
class MonitoringAlert:
    """
    Monitoring alert record.

    Documents alerts generated by the monitoring system.
    """

    alert_id: str
    priority: AlertPriority
    title: str
    description: str

    # Source
    source_metric: str = ""
    source_component: str = ""

    # Details
    trigger_value: float = 0.0
    threshold_value: float = 0.0
    deviation: float = 0.0

    # Timing
    triggered_at: str = ""
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None

    # Status
    is_active: bool = True
    acknowledged: bool = False
    acknowledged_by: str = ""

    # Actions
    auto_action_taken: str = ""
    manual_action_required: bool = False

    # Escalation
    escalated: bool = False
    escalation_level: int = 0
    escalation_time: Optional[str] = None

    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = f"ALERT-{uuid.uuid4().hex[:8].upper()}"
        if not self.triggered_at:
            self.triggered_at = datetime.now(timezone.utc).isoformat()


@dataclass
class PeriodicReport:
    """
    Periodic monitoring report.

    Generated at regular intervals per Article 72.
    """

    report_id: str
    report_type: str  # daily, weekly, monthly, quarterly
    period_start: str
    period_end: str

    # Summary metrics
    total_predictions: int = 0
    total_incidents: int = 0
    serious_incidents: int = 0
    total_feedback: int = 0

    # Performance summary
    average_accuracy: float = 0.0
    average_latency: float = 0.0
    availability_percent: float = 0.0
    error_rate: float = 0.0

    # Drift summary
    drift_detections: int = 0
    critical_drifts: int = 0

    # Alerts summary
    total_alerts: int = 0
    critical_alerts: int = 0
    unresolved_alerts: int = 0

    # Compliance status
    compliance_score: float = 0.0
    compliance_issues: List[str] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    corrective_actions_needed: List[str] = field(default_factory=list)

    # Metadata
    generated_at: str = ""
    generated_by: str = ""

    def __post_init__(self):
        if not self.report_id:
            self.report_id = f"REP-{uuid.uuid4().hex[:8].upper()}"
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PostMarketConfig:
    """
    Configuration for Post-Market Monitoring System.
    """

    # Monitoring settings
    enable_performance_monitoring: bool = True
    enable_drift_detection: bool = True
    enable_incident_tracking: bool = True
    enable_feedback_collection: bool = True

    # Thresholds
    drift_warning_threshold: float = 0.1  # 10% deviation
    drift_critical_threshold: float = 0.2  # 20% deviation
    error_rate_warning: float = 0.01  # 1%
    error_rate_critical: float = 0.05  # 5%

    # Drift detection
    drift_detection_window: int = 100  # samples
    drift_check_frequency_hours: int = 1
    statistical_significance: float = 0.05

    # Alerting
    enable_alerting: bool = True
    alert_escalation_minutes: int = 30
    max_escalation_levels: int = 3

    # Reporting
    daily_report_enabled: bool = True
    weekly_report_enabled: bool = True
    monthly_report_enabled: bool = True

    # Retention
    metrics_retention_days: int = 365
    incidents_retention_days: int = 1825  # 5 years
    alerts_retention_days: int = 90

    # Regulatory
    serious_incident_report_hours: int = 72  # Per Article 73

    # Paths
    log_path: str = "logs/ai_act/post_market"
    reports_path: str = "reports/post_market"

    # Callbacks
    alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    incident_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Performance Monitor
# =============================================================================


class PerformanceMonitor:
    """
    Performance monitoring component.

    Tracks performance metrics and detects drift.
    """

    def __init__(self, config: PostMarketConfig):
        """Initialize performance monitor."""
        self.config = config
        self._metrics: Dict[str, MonitoringMetric] = {}
        self._history: Dict[str, List[Tuple[str, float]]] = {}  # metric_id -> [(timestamp, value)]
        self._lock = threading.RLock()

    def register_metric(
        self,
        name: str,
        metric_type: MonitoringMetricType,
        baseline_value: float,
        baseline_std: float = 0.0,
        warning_threshold: float = 0.0,
        critical_threshold: float = 0.0,
        threshold_direction: str = "below",
        unit: str = "",
    ) -> MonitoringMetric:
        """
        Register a metric for monitoring.

        Args:
            name: Metric name
            metric_type: Type of metric
            baseline_value: Baseline value from pre-market
            baseline_std: Baseline standard deviation
            warning_threshold: Warning threshold
            critical_threshold: Critical threshold
            threshold_direction: Direction for threshold comparison
            unit: Measurement unit

        Returns:
            Created MonitoringMetric
        """
        metric = MonitoringMetric(
            metric_id="",
            name=name,
            metric_type=metric_type,
            description=f"Post-market monitoring metric: {name}",
            baseline_value=baseline_value,
            baseline_std=baseline_std,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            threshold_direction=threshold_direction,
            unit=unit,
        )

        with self._lock:
            self._metrics[metric.metric_id] = metric
            self._history[metric.metric_id] = []

        return metric

    def record_value(
        self,
        metric_id: str,
        value: float,
        timestamp: Optional[str] = None,
    ) -> MonitoringMetric:
        """
        Record a metric value.

        Args:
            metric_id: Metric ID
            value: Observed value
            timestamp: Optional timestamp

        Returns:
            Updated MonitoringMetric
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        with self._lock:
            if metric_id not in self._metrics:
                raise ValueError(f"Metric {metric_id} not found")

            metric = self._metrics[metric_id]
            history = self._history[metric_id]

            # Record value
            history.append((ts, value))

            # Trim history
            max_history = self.config.drift_detection_window * 10
            if len(history) > max_history:
                history.pop(0)

            # Update current statistics
            recent_values = [v for _, v in history[-self.config.drift_detection_window :]]
            metric.current_value = value
            metric.current_samples = len(recent_values)
            metric.last_updated = ts

            if len(recent_values) > 1:
                metric.current_std = statistics.stdev(recent_values)

        return metric

    def detect_drift(
        self,
        metric_id: str,
    ) -> Optional[DriftDetectionResult]:
        """
        Detect performance drift for a metric.

        Args:
            metric_id: Metric ID

        Returns:
            DriftDetectionResult if drift detected, None otherwise
        """
        with self._lock:
            if metric_id not in self._metrics:
                raise ValueError(f"Metric {metric_id} not found")

            metric = self._metrics[metric_id]
            history = self._history[metric_id]

        if len(history) < self.config.drift_detection_window:
            return None

        # Get recent values
        recent_values = [v for _, v in history[-self.config.drift_detection_window :]]
        current_mean = statistics.mean(recent_values)
        current_std = statistics.stdev(recent_values) if len(recent_values) > 1 else 0

        # Calculate deviation
        baseline = metric.baseline_value
        deviation = abs(current_mean - baseline)
        deviation_percent = deviation / abs(baseline) if abs(baseline) > 1e-8 else deviation

        # Determine severity
        if deviation_percent >= self.config.drift_critical_threshold:
            severity = DriftSeverity.CRITICAL
        elif deviation_percent >= self.config.drift_warning_threshold:
            severity = DriftSeverity.MEDIUM
        elif deviation_percent >= self.config.drift_warning_threshold / 2:
            severity = DriftSeverity.LOW
        else:
            severity = DriftSeverity.NONE

        # Update metric
        metric.drift_score = deviation_percent
        metric.drift_severity = severity

        if severity == DriftSeverity.NONE:
            return None

        # Create detection result
        result = DriftDetectionResult(
            detection_id="",
            metric_id=metric_id,
            drift_type=DriftType.MODEL_DRIFT,
            severity=severity,
            baseline_value=baseline,
            current_value=current_mean,
            deviation=deviation,
            deviation_percent=deviation_percent * 100,
            test_method="mean_comparison",
            recommended_actions=self._get_recommended_actions(severity),
        )

        return result

    def _get_recommended_actions(self, severity: DriftSeverity) -> List[str]:
        """Get recommended actions based on drift severity."""
        if severity == DriftSeverity.CRITICAL:
            return [
                "Immediately investigate root cause",
                "Consider pausing AI system operation",
                "Notify stakeholders",
                "Prepare incident report",
            ]
        elif severity == DriftSeverity.HIGH:
            return [
                "Investigate root cause within 24 hours",
                "Monitor closely for further degradation",
                "Prepare mitigation plan",
            ]
        elif severity == DriftSeverity.MEDIUM:
            return [
                "Schedule investigation within 1 week",
                "Monitor trend",
                "Document for periodic review",
            ]
        else:
            return [
                "Document for periodic review",
                "Continue monitoring",
            ]

    def get_metric(self, metric_id: str) -> Optional[MonitoringMetric]:
        """Get metric by ID."""
        with self._lock:
            return self._metrics.get(metric_id)

    def get_all_metrics(self) -> List[MonitoringMetric]:
        """Get all registered metrics."""
        with self._lock:
            return list(self._metrics.values())

    def get_metrics_with_drift(self) -> List[MonitoringMetric]:
        """Get metrics that have detected drift."""
        with self._lock:
            return [m for m in self._metrics.values() if m.drift_severity != DriftSeverity.NONE]


# =============================================================================
# Incident Tracker
# =============================================================================


class IncidentTracker:
    """
    Incident tracking and reporting per Article 73.

    Tracks incidents and manages regulatory reporting requirements.
    """

    def __init__(self, config: PostMarketConfig):
        """Initialize incident tracker."""
        self.config = config
        self._incidents: Dict[str, Incident] = {}
        self._lock = threading.RLock()

    def report_incident(
        self,
        severity: IncidentSeverity,
        title: str,
        description: str,
        incident_type: str = "",
        affected_component: str = "",
        reporter: str = "",
    ) -> Incident:
        """
        Report a new incident.

        Args:
            severity: Incident severity
            title: Incident title
            description: Incident description
            incident_type: Type of incident
            affected_component: Affected component
            reporter: Who reported the incident

        Returns:
            Created Incident
        """
        incident = Incident(
            incident_id="",
            severity=severity,
            title=title,
            description=description,
            incident_type=incident_type,
            affected_component=affected_component,
            reporter=reporter,
            reported_at=datetime.now(timezone.utc).isoformat(),
        )

        # Check if regulatory reporting required (Article 73)
        if severity == IncidentSeverity.SERIOUS:
            incident.requires_regulatory_report = True

        with self._lock:
            self._incidents[incident.incident_id] = incident

        # Trigger callback
        if self.config.incident_callback:
            try:
                self.config.incident_callback("incident_reported", asdict(incident))
            except Exception as e:
                logger.error(f"Incident callback failed: {e}")

        logger.info(f"Incident reported: {incident.incident_id} - {title}")
        return incident

    def update_incident_status(
        self,
        incident_id: str,
        status: IncidentStatus,
        notes: str = "",
    ) -> Incident:
        """Update incident status."""
        with self._lock:
            if incident_id not in self._incidents:
                raise ValueError(f"Incident {incident_id} not found")

            incident = self._incidents[incident_id]
            incident.status = status
            if notes:
                incident.investigation_notes += (
                    f"\n[{datetime.now(timezone.utc).isoformat()}] {notes}"
                )

            if status == IncidentStatus.RESOLVED:
                incident.resolved_at = datetime.now(timezone.utc).isoformat()

        return incident

    def set_root_cause(
        self,
        incident_id: str,
        root_cause: str,
        contributing_factors: Optional[List[str]] = None,
    ) -> Incident:
        """Set root cause for an incident."""
        with self._lock:
            if incident_id not in self._incidents:
                raise ValueError(f"Incident {incident_id} not found")

            incident = self._incidents[incident_id]
            incident.root_cause = root_cause
            incident.contributing_factors = contributing_factors or []
            incident.status = IncidentStatus.ROOT_CAUSE_IDENTIFIED

        return incident

    def add_corrective_action(
        self,
        incident_id: str,
        action: str,
        action_type: str = "corrective",
    ) -> Incident:
        """Add corrective/preventive action to incident."""
        with self._lock:
            if incident_id not in self._incidents:
                raise ValueError(f"Incident {incident_id} not found")

            incident = self._incidents[incident_id]
            if action_type == "immediate":
                incident.immediate_actions.append(action)
            elif action_type == "preventive":
                incident.preventive_actions.append(action)
            else:
                incident.corrective_actions.append(action)
                incident.status = IncidentStatus.CORRECTIVE_ACTION

        return incident

    def submit_regulatory_report(
        self,
        incident_id: str,
        authority: str,
    ) -> Incident:
        """Mark regulatory report as submitted per Article 73."""
        with self._lock:
            if incident_id not in self._incidents:
                raise ValueError(f"Incident {incident_id} not found")

            incident = self._incidents[incident_id]
            incident.regulatory_report_submitted = True
            incident.regulatory_report_date = datetime.now(timezone.utc).isoformat()
            incident.regulatory_authority = authority

        logger.info(f"Regulatory report submitted for incident: {incident_id}")
        return incident

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID."""
        with self._lock:
            return self._incidents.get(incident_id)

    def get_open_incidents(self) -> List[Incident]:
        """Get all open incidents."""
        with self._lock:
            return [
                i
                for i in self._incidents.values()
                if i.status not in (IncidentStatus.RESOLVED, IncidentStatus.CLOSED)
            ]

    def get_serious_incidents(self) -> List[Incident]:
        """Get serious incidents requiring regulatory reporting."""
        with self._lock:
            return [i for i in self._incidents.values() if i.severity == IncidentSeverity.SERIOUS]

    def get_pending_regulatory_reports(self) -> List[Incident]:
        """Get incidents with pending regulatory reports."""
        with self._lock:
            return [
                i
                for i in self._incidents.values()
                if i.requires_regulatory_report and not i.regulatory_report_submitted
            ]


# =============================================================================
# Feedback Collector
# =============================================================================


class FeedbackCollector:
    """
    User/deployer feedback collection.

    Collects and manages feedback from system users.
    """

    def __init__(self, config: PostMarketConfig):
        """Initialize feedback collector."""
        self.config = config
        self._feedback: Dict[str, Feedback] = {}
        self._lock = threading.RLock()

    def submit_feedback(
        self,
        feedback_type: FeedbackType,
        subject: str,
        description: str,
        source_type: str = "",
        source_id: str = "",
        priority: AlertPriority = AlertPriority.INFO,
    ) -> Feedback:
        """
        Submit new feedback.

        Args:
            feedback_type: Type of feedback
            subject: Feedback subject
            description: Detailed description
            source_type: Type of source
            source_id: Source identifier
            priority: Feedback priority

        Returns:
            Created Feedback
        """
        feedback = Feedback(
            feedback_id="",
            feedback_type=feedback_type,
            subject=subject,
            description=description,
            source_type=source_type,
            source_id=source_id,
            priority=priority,
        )

        with self._lock:
            self._feedback[feedback.feedback_id] = feedback

        return feedback

    def acknowledge_feedback(
        self,
        feedback_id: str,
        response: str = "",
    ) -> Feedback:
        """Acknowledge feedback."""
        with self._lock:
            if feedback_id not in self._feedback:
                raise ValueError(f"Feedback {feedback_id} not found")

            fb = self._feedback[feedback_id]
            fb.acknowledged = True
            fb.acknowledged_at = datetime.now(timezone.utc).isoformat()
            if response:
                fb.response = response
                fb.response_date = fb.acknowledged_at

        return fb

    def resolve_feedback(
        self,
        feedback_id: str,
        resolution: str,
    ) -> Feedback:
        """Resolve feedback."""
        with self._lock:
            if feedback_id not in self._feedback:
                raise ValueError(f"Feedback {feedback_id} not found")

            fb = self._feedback[feedback_id]
            fb.resolved = True
            fb.resolution = resolution
            fb.resolved_at = datetime.now(timezone.utc).isoformat()

        return fb

    def link_to_incident(
        self,
        feedback_id: str,
        incident_id: str,
    ) -> Feedback:
        """Link feedback to an incident."""
        with self._lock:
            if feedback_id not in self._feedback:
                raise ValueError(f"Feedback {feedback_id} not found")

            fb = self._feedback[feedback_id]
            if incident_id not in fb.related_incidents:
                fb.related_incidents.append(incident_id)

        return fb

    def get_feedback(self, feedback_id: str) -> Optional[Feedback]:
        """Get feedback by ID."""
        with self._lock:
            return self._feedback.get(feedback_id)

    def get_unacknowledged_feedback(self) -> List[Feedback]:
        """Get unacknowledged feedback."""
        with self._lock:
            return [f for f in self._feedback.values() if not f.acknowledged]

    def get_feedback_by_type(self, feedback_type: FeedbackType) -> List[Feedback]:
        """Get feedback by type."""
        with self._lock:
            return [f for f in self._feedback.values() if f.feedback_type == feedback_type]


# =============================================================================
# Main Post-Market Monitoring System
# =============================================================================


class PostMarketMonitoringSystem:
    """
    AI Act compliant Post-Market Monitoring System per Article 72.

    Provides comprehensive post-market monitoring including:
    - Performance metrics tracking
    - Drift detection
    - Incident tracking and reporting
    - Feedback collection
    - Periodic compliance reporting

    Usage:
        config = PostMarketConfig(
            enable_drift_detection=True,
            drift_warning_threshold=0.1,
        )
        pmms = PostMarketMonitoringSystem(config)

        # Register a metric
        metric = pmms.register_metric(
            name="Accuracy",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )

        # Record values
        pmms.record_metric_value(metric.metric_id, 0.93)

        # Check for drift
        drift = pmms.check_drift(metric.metric_id)

        # Report an incident
        incident = pmms.report_incident(
            severity=IncidentSeverity.MAJOR,
            title="Model Performance Degradation",
            description="Accuracy dropped below threshold",
        )
    """

    def __init__(self, config: Optional[PostMarketConfig] = None):
        """Initialize post-market monitoring system."""
        self.config = config or PostMarketConfig()

        # Components
        self._performance_monitor = PerformanceMonitor(self.config)
        self._incident_tracker = IncidentTracker(self.config)
        self._feedback_collector = FeedbackCollector(self.config)

        # Alerts
        self._alerts: Dict[str, MonitoringAlert] = {}
        self._drift_detections: Dict[str, DriftDetectionResult] = {}
        self._reports: Dict[str, PeriodicReport] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        self._reports_path = Path(self.config.reports_path)
        self._reports_path.mkdir(parents=True, exist_ok=True)

        logger.info("PostMarketMonitoringSystem initialized")

    # =========================================================================
    # Performance Monitoring
    # =========================================================================

    def register_metric(
        self,
        name: str,
        metric_type: MonitoringMetricType,
        baseline_value: float,
        baseline_std: float = 0.0,
        warning_threshold: Optional[float] = None,
        critical_threshold: Optional[float] = None,
        unit: str = "",
    ) -> MonitoringMetric:
        """Register a metric for monitoring."""
        warning = warning_threshold or baseline_value * (1 - self.config.drift_warning_threshold)
        critical = critical_threshold or baseline_value * (1 - self.config.drift_critical_threshold)

        metric = self._performance_monitor.register_metric(
            name=name,
            metric_type=metric_type,
            baseline_value=baseline_value,
            baseline_std=baseline_std,
            warning_threshold=warning,
            critical_threshold=critical,
            unit=unit,
        )

        self._log_event(
            "metric_registered",
            {
                "metric_id": metric.metric_id,
                "name": name,
                "baseline": baseline_value,
            },
        )

        return metric

    def record_metric_value(
        self,
        metric_id: str,
        value: float,
        check_drift: bool = True,
    ) -> Tuple[MonitoringMetric, Optional[DriftDetectionResult]]:
        """
        Record a metric value and optionally check for drift.

        Args:
            metric_id: Metric ID
            value: Observed value
            check_drift: Whether to check for drift

        Returns:
            Tuple of (updated metric, drift detection if any)
        """
        metric = self._performance_monitor.record_value(metric_id, value)

        drift = None
        if check_drift and self.config.enable_drift_detection:
            drift = self._performance_monitor.detect_drift(metric_id)

            if drift:
                with self._lock:
                    self._drift_detections[drift.detection_id] = drift

                # Generate alert if needed
                if drift.severity.value >= DriftSeverity.MEDIUM.value:
                    self._generate_alert(
                        priority=(
                            AlertPriority.HIGH
                            if drift.severity == DriftSeverity.CRITICAL
                            else AlertPriority.WARNING
                        ),
                        title=f"Performance drift detected: {metric.name}",
                        description=f"Deviation: {drift.deviation_percent:.1f}%",
                        source_metric=metric_id,
                    )

                self._log_event("drift_detected", asdict(drift))

        return metric, drift

    def check_drift(self, metric_id: str) -> Optional[DriftDetectionResult]:
        """Check for drift on a metric."""
        return self._performance_monitor.detect_drift(metric_id)

    def get_metric(self, metric_id: str) -> Optional[MonitoringMetric]:
        """Get metric by ID."""
        return self._performance_monitor.get_metric(metric_id)

    def get_all_metrics(self) -> List[MonitoringMetric]:
        """Get all registered metrics."""
        return self._performance_monitor.get_all_metrics()

    def get_metrics_with_drift(self) -> List[MonitoringMetric]:
        """Get metrics that have detected drift."""
        return self._performance_monitor.get_metrics_with_drift()

    # =========================================================================
    # Incident Management
    # =========================================================================

    def report_incident(
        self,
        severity: IncidentSeverity,
        title: str,
        description: str,
        incident_type: str = "",
        affected_component: str = "",
        reporter: str = "",
    ) -> Incident:
        """Report a new incident."""
        incident = self._incident_tracker.report_incident(
            severity=severity,
            title=title,
            description=description,
            incident_type=incident_type,
            affected_component=affected_component,
            reporter=reporter,
        )

        # Generate alert for serious incidents
        if severity.value >= IncidentSeverity.MAJOR.value:
            self._generate_alert(
                priority=(
                    AlertPriority.CRITICAL
                    if severity == IncidentSeverity.SERIOUS
                    else AlertPriority.HIGH
                ),
                title=f"Incident: {title}",
                description=description,
                source_component=affected_component,
            )

        self._log_event("incident_reported", asdict(incident))
        return incident

    def update_incident(
        self,
        incident_id: str,
        status: IncidentStatus,
        notes: str = "",
    ) -> Incident:
        """Update incident status."""
        incident = self._incident_tracker.update_incident_status(incident_id, status, notes)
        self._log_event(
            "incident_updated",
            {
                "incident_id": incident_id,
                "status": status.value,
            },
        )
        return incident

    def set_incident_root_cause(
        self,
        incident_id: str,
        root_cause: str,
        contributing_factors: Optional[List[str]] = None,
    ) -> Incident:
        """Set root cause for incident."""
        return self._incident_tracker.set_root_cause(incident_id, root_cause, contributing_factors)

    def add_incident_action(
        self,
        incident_id: str,
        action: str,
        action_type: str = "corrective",
    ) -> Incident:
        """Add action to incident."""
        return self._incident_tracker.add_corrective_action(incident_id, action, action_type)

    def submit_regulatory_report(
        self,
        incident_id: str,
        authority: str,
    ) -> Incident:
        """Submit regulatory report for serious incident."""
        incident = self._incident_tracker.submit_regulatory_report(incident_id, authority)
        self._log_event(
            "regulatory_report_submitted",
            {
                "incident_id": incident_id,
                "authority": authority,
            },
        )
        return incident

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID."""
        return self._incident_tracker.get_incident(incident_id)

    def get_open_incidents(self) -> List[Incident]:
        """Get all open incidents."""
        return self._incident_tracker.get_open_incidents()

    def get_serious_incidents(self) -> List[Incident]:
        """Get serious incidents."""
        return self._incident_tracker.get_serious_incidents()

    def get_pending_regulatory_reports(self) -> List[Incident]:
        """Get incidents pending regulatory reporting."""
        return self._incident_tracker.get_pending_regulatory_reports()

    # =========================================================================
    # Feedback Management
    # =========================================================================

    def submit_feedback(
        self,
        feedback_type: FeedbackType,
        subject: str,
        description: str,
        source_type: str = "",
        source_id: str = "",
        priority: AlertPriority = AlertPriority.INFO,
    ) -> Feedback:
        """Submit user/deployer feedback."""
        feedback = self._feedback_collector.submit_feedback(
            feedback_type=feedback_type,
            subject=subject,
            description=description,
            source_type=source_type,
            source_id=source_id,
            priority=priority,
        )

        # Generate alert for safety concerns
        if feedback_type == FeedbackType.SAFETY_CONCERN:
            self._generate_alert(
                priority=AlertPriority.HIGH,
                title=f"Safety concern reported: {subject}",
                description=description,
            )

        self._log_event("feedback_submitted", asdict(feedback))
        return feedback

    def acknowledge_feedback(
        self,
        feedback_id: str,
        response: str = "",
    ) -> Feedback:
        """Acknowledge feedback."""
        return self._feedback_collector.acknowledge_feedback(feedback_id, response)

    def resolve_feedback(
        self,
        feedback_id: str,
        resolution: str,
    ) -> Feedback:
        """Resolve feedback."""
        return self._feedback_collector.resolve_feedback(feedback_id, resolution)

    def get_feedback(self, feedback_id: str) -> Optional[Feedback]:
        """Get feedback by ID."""
        return self._feedback_collector.get_feedback(feedback_id)

    def get_unacknowledged_feedback(self) -> List[Feedback]:
        """Get unacknowledged feedback."""
        return self._feedback_collector.get_unacknowledged_feedback()

    # =========================================================================
    # Alerting
    # =========================================================================

    def _generate_alert(
        self,
        priority: AlertPriority,
        title: str,
        description: str,
        source_metric: str = "",
        source_component: str = "",
    ) -> MonitoringAlert:
        """Generate a monitoring alert."""
        alert = MonitoringAlert(
            alert_id="",
            priority=priority,
            title=title,
            description=description,
            source_metric=source_metric,
            source_component=source_component,
        )

        with self._lock:
            self._alerts[alert.alert_id] = alert

        # Trigger callback
        if self.config.alert_callback:
            try:
                self.config.alert_callback("alert", asdict(alert))
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        self._log_event("alert_generated", asdict(alert))
        return alert

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
    ) -> MonitoringAlert:
        """Acknowledge an alert."""
        with self._lock:
            if alert_id not in self._alerts:
                raise ValueError(f"Alert {alert_id} not found")

            alert = self._alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now(timezone.utc).isoformat()

        return alert

    def resolve_alert(self, alert_id: str) -> MonitoringAlert:
        """Resolve an alert."""
        with self._lock:
            if alert_id not in self._alerts:
                raise ValueError(f"Alert {alert_id} not found")

            alert = self._alerts[alert_id]
            alert.is_active = False
            alert.resolved_at = datetime.now(timezone.utc).isoformat()

        return alert

    def get_alert(self, alert_id: str) -> Optional[MonitoringAlert]:
        """Get alert by ID."""
        with self._lock:
            return self._alerts.get(alert_id)

    def get_active_alerts(self) -> List[MonitoringAlert]:
        """Get all active alerts."""
        with self._lock:
            return [a for a in self._alerts.values() if a.is_active]

    # =========================================================================
    # Reporting
    # =========================================================================

    def generate_periodic_report(
        self,
        report_type: str = "daily",
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> PeriodicReport:
        """
        Generate a periodic monitoring report.

        Args:
            report_type: Type of report (daily, weekly, monthly)
            period_start: Period start (ISO format)
            period_end: Period end (ISO format)

        Returns:
            Generated PeriodicReport
        """
        now = datetime.now(timezone.utc)

        if not period_end:
            period_end = now.isoformat()

        if not period_start:
            if report_type == "daily":
                period_start = (now - timedelta(days=1)).isoformat()
            elif report_type == "weekly":
                period_start = (now - timedelta(weeks=1)).isoformat()
            elif report_type == "monthly":
                period_start = (now - timedelta(days=30)).isoformat()
            else:
                period_start = (now - timedelta(days=1)).isoformat()

        # Gather statistics
        metrics = self.get_all_metrics()
        incidents = self._incident_tracker._incidents
        feedback = self._feedback_collector._feedback
        alerts = self._alerts
        drifts = self._drift_detections

        # Calculate metrics
        accuracy_values = [
            m.current_value
            for m in metrics
            if m.metric_type == MonitoringMetricType.ACCURACY and m.current_value is not None
        ]
        latency_values = [
            m.current_value
            for m in metrics
            if m.metric_type == MonitoringMetricType.LATENCY and m.current_value is not None
        ]

        report = PeriodicReport(
            report_id="",
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            total_incidents=len(incidents),
            serious_incidents=len(
                [i for i in incidents.values() if i.severity == IncidentSeverity.SERIOUS]
            ),
            total_feedback=len(feedback),
            average_accuracy=statistics.mean(accuracy_values) if accuracy_values else 0.0,
            average_latency=statistics.mean(latency_values) if latency_values else 0.0,
            drift_detections=len(drifts),
            critical_drifts=len(
                [d for d in drifts.values() if d.severity == DriftSeverity.CRITICAL]
            ),
            total_alerts=len(alerts),
            critical_alerts=len(
                [a for a in alerts.values() if a.priority == AlertPriority.CRITICAL]
            ),
            unresolved_alerts=len([a for a in alerts.values() if a.is_active]),
        )

        # Calculate compliance score
        report.compliance_score = self._calculate_compliance_score(report)
        report.compliance_issues = self._identify_compliance_issues(report)
        report.recommendations = self._generate_recommendations(report)

        with self._lock:
            self._reports[report.report_id] = report

        # Save report
        self._save_report(report)

        self._log_event(
            "report_generated",
            {
                "report_id": report.report_id,
                "type": report_type,
            },
        )

        return report

    def _calculate_compliance_score(self, report: PeriodicReport) -> float:
        """Calculate compliance score."""
        score = 100.0

        # Deduct for serious incidents
        score -= min(report.serious_incidents * 10, 30)

        # Deduct for critical drifts
        score -= min(report.critical_drifts * 5, 20)

        # Deduct for unresolved alerts
        score -= min(report.unresolved_alerts * 2, 20)

        # Deduct for critical alerts
        score -= min(report.critical_alerts * 5, 20)

        return max(0, score)

    def _identify_compliance_issues(self, report: PeriodicReport) -> List[str]:
        """Identify compliance issues."""
        issues = []

        if report.serious_incidents > 0:
            issues.append(f"{report.serious_incidents} serious incident(s) occurred")

        pending_reports = self.get_pending_regulatory_reports()
        if pending_reports:
            issues.append(f"{len(pending_reports)} regulatory report(s) pending")

        if report.critical_drifts > 0:
            issues.append(f"{report.critical_drifts} critical drift(s) detected")

        if report.unresolved_alerts > 5:
            issues.append(f"{report.unresolved_alerts} unresolved alerts")

        return issues

    def _generate_recommendations(self, report: PeriodicReport) -> List[str]:
        """Generate recommendations based on report."""
        recommendations = []

        if report.serious_incidents > 0:
            recommendations.append("Review and address all serious incidents immediately")

        pending_reports = self.get_pending_regulatory_reports()
        if pending_reports:
            recommendations.append(
                f"Submit {len(pending_reports)} pending regulatory report(s) within required timeframe"
            )

        if report.critical_drifts > 0:
            recommendations.append("Investigate root causes of critical performance drifts")

        if report.compliance_score < 80:
            recommendations.append("Conduct comprehensive compliance review")

        if report.unresolved_alerts > 10:
            recommendations.append("Allocate resources to address backlog of alerts")

        return recommendations

    def _save_report(self, report: PeriodicReport) -> None:
        """Save report to file."""
        filename = f"report_{report.report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self._reports_path / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(asdict(report), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    def get_report(self, report_id: str) -> Optional[PeriodicReport]:
        """Get report by ID."""
        with self._lock:
            return self._reports.get(report_id)

    # =========================================================================
    # Compliance Status
    # =========================================================================

    def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall post-market monitoring compliance status."""
        metrics = self.get_all_metrics()
        metrics_with_drift = self.get_metrics_with_drift()
        open_incidents = self.get_open_incidents()
        serious_incidents = self.get_serious_incidents()
        pending_reports = self.get_pending_regulatory_reports()
        active_alerts = self.get_active_alerts()
        unack_feedback = self.get_unacknowledged_feedback()

        with self._lock:
            total_drifts = len(self._drift_detections)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "article_reference": "Article 72",
            "metrics": {
                "total_registered": len(metrics),
                "with_drift": len(metrics_with_drift),
            },
            "drift": {
                "total_detections": total_drifts,
                "critical": len(
                    [
                        d
                        for d in self._drift_detections.values()
                        if d.severity == DriftSeverity.CRITICAL
                    ]
                ),
            },
            "incidents": {
                "open": len(open_incidents),
                "serious": len(serious_incidents),
                "pending_regulatory_reports": len(pending_reports),
            },
            "alerts": {
                "active": len(active_alerts),
                "critical": len([a for a in active_alerts if a.priority == AlertPriority.CRITICAL]),
            },
            "feedback": {
                "unacknowledged": len(unack_feedback),
            },
            "compliance_issues": self._get_current_compliance_issues(
                pending_reports, open_incidents, active_alerts
            ),
            "is_compliant": len(pending_reports) == 0
            and len([a for a in active_alerts if a.priority == AlertPriority.CRITICAL]) == 0,
        }

    def _get_current_compliance_issues(
        self,
        pending_reports: List[Incident],
        open_incidents: List[Incident],
        active_alerts: List[MonitoringAlert],
    ) -> List[str]:
        """Get current compliance issues."""
        issues = []

        if pending_reports:
            for inc in pending_reports:
                # Check if deadline passed
                reported = datetime.fromisoformat(inc.reported_at.replace("Z", "+00:00"))
                deadline = reported + timedelta(hours=self.config.serious_incident_report_hours)
                if datetime.now(timezone.utc) > deadline:
                    issues.append(f"Overdue regulatory report for incident {inc.incident_id}")

        critical_alerts = [a for a in active_alerts if a.priority == AlertPriority.CRITICAL]
        if critical_alerts:
            issues.append(f"{len(critical_alerts)} critical alert(s) require immediate attention")

        return issues

    # =========================================================================
    # Export
    # =========================================================================

    def export_monitoring_data(self) -> Dict[str, Any]:
        """Export all monitoring data."""
        with self._lock:
            return {
                "export_date": datetime.now(timezone.utc).isoformat(),
                "metrics": [asdict(m) for m in self._performance_monitor.get_all_metrics()],
                "drift_detections": [asdict(d) for d in self._drift_detections.values()],
                "incidents": [asdict(i) for i in self._incident_tracker._incidents.values()],
                "feedback": [asdict(f) for f in self._feedback_collector._feedback.values()],
                "alerts": [asdict(a) for a in self._alerts.values()],
                "reports": [asdict(r) for r in self._reports.values()],
                "compliance_status": self.get_compliance_status(),
            }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log event to file."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"pmm_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log PMM event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================


def create_post_market_monitoring(
    config: Optional[Union[Dict[str, Any], PostMarketConfig]] = None,
) -> PostMarketMonitoringSystem:
    """
    Create a PostMarketMonitoringSystem instance.

    Args:
        config: Optional configuration

    Returns:
        Configured PostMarketMonitoringSystem instance
    """
    if config is None:
        pmm_config = PostMarketConfig()
    elif isinstance(config, PostMarketConfig):
        pmm_config = config
    else:
        pmm_config = PostMarketConfig(
            enable_performance_monitoring=config.get("enable_performance_monitoring", True),
            enable_drift_detection=config.get("enable_drift_detection", True),
            enable_incident_tracking=config.get("enable_incident_tracking", True),
            enable_feedback_collection=config.get("enable_feedback_collection", True),
            drift_warning_threshold=config.get("drift_warning_threshold", 0.1),
            drift_critical_threshold=config.get("drift_critical_threshold", 0.2),
            error_rate_warning=config.get("error_rate_warning", 0.01),
            error_rate_critical=config.get("error_rate_critical", 0.05),
            drift_detection_window=config.get("drift_detection_window", 100),
            enable_alerting=config.get("enable_alerting", True),
            daily_report_enabled=config.get("daily_report_enabled", True),
            weekly_report_enabled=config.get("weekly_report_enabled", True),
            monthly_report_enabled=config.get("monthly_report_enabled", True),
            serious_incident_report_hours=config.get("serious_incident_report_hours", 72),
            log_path=config.get("log_path", "logs/ai_act/post_market"),
            reports_path=config.get("reports_path", "reports/post_market"),
        )

    return PostMarketMonitoringSystem(pmm_config)


def get_default_monitoring_metrics() -> List[Dict[str, Any]]:
    """
    Get default monitoring metrics for trading AI systems.

    Returns:
        List of metric definitions
    """
    return [
        {
            "name": "Model Accuracy",
            "metric_type": MonitoringMetricType.ACCURACY,
            "description": "Overall model prediction accuracy",
            "baseline_value": 0.65,
            "unit": "ratio",
        },
        {
            "name": "Inference Latency",
            "metric_type": MonitoringMetricType.LATENCY,
            "description": "Model inference latency",
            "baseline_value": 50.0,
            "unit": "ms",
        },
        {
            "name": "Error Rate",
            "metric_type": MonitoringMetricType.ERROR_RATE,
            "description": "System error rate",
            "baseline_value": 0.001,
            "unit": "ratio",
        },
        {
            "name": "System Availability",
            "metric_type": MonitoringMetricType.AVAILABILITY,
            "description": "System uptime availability",
            "baseline_value": 0.999,
            "unit": "ratio",
        },
        {
            "name": "Prediction Drift",
            "metric_type": MonitoringMetricType.DRIFT,
            "description": "Model prediction distribution drift",
            "baseline_value": 0.0,
            "unit": "score",
        },
    ]
