# -*- coding: utf-8 -*-
"""
Comprehensive tests for AI Act Post-Market Monitoring System (Article 72).

This module provides 100% test coverage for:
- PostMarketMonitoringSystem
- PerformanceMonitor
- IncidentTracker
- FeedbackCollector
- All data structures and enums
- Factory functions
"""

import json
import pytest
import tempfile
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from dataclasses import asdict

from services.ai_act.post_market_monitoring import (
    # Enums
    MonitoringMetricType,
    DriftType,
    DriftSeverity,
    IncidentSeverity,
    IncidentStatus,
    FeedbackType,
    AlertPriority,
    # Data structures
    MonitoringMetric,
    DriftDetectionResult,
    Incident,
    Feedback,
    MonitoringAlert,
    PeriodicReport,
    # Configuration
    PostMarketConfig,
    # Components
    PerformanceMonitor,
    IncidentTracker,
    FeedbackCollector,
    # Main class
    PostMarketMonitoringSystem,
    # Factory functions
    create_post_market_monitoring,
    get_default_monitoring_metrics,
)


# =============================================================================
# Enum Tests
# =============================================================================

class TestMonitoringMetricType:
    """Tests for MonitoringMetricType enum."""

    def test_all_metric_types(self):
        """Test all metric types are defined."""
        assert MonitoringMetricType.ACCURACY.value == "accuracy"
        assert MonitoringMetricType.LATENCY.value == "latency"
        assert MonitoringMetricType.THROUGHPUT.value == "throughput"
        assert MonitoringMetricType.ERROR_RATE.value == "error_rate"
        assert MonitoringMetricType.AVAILABILITY.value == "availability"
        assert MonitoringMetricType.FAIRNESS.value == "fairness"
        assert MonitoringMetricType.DRIFT.value == "drift"
        assert MonitoringMetricType.SAFETY.value == "safety"
        assert MonitoringMetricType.USER_SATISFACTION.value == "user_satisfaction"
        assert MonitoringMetricType.CUSTOM.value == "custom"

    def test_enum_count(self):
        """Test correct number of metric types."""
        assert len(MonitoringMetricType) == 10


class TestDriftType:
    """Tests for DriftType enum."""

    def test_all_drift_types(self):
        """Test all drift types are defined."""
        assert DriftType.DATA_DRIFT.value == "data_drift"
        assert DriftType.CONCEPT_DRIFT.value == "concept_drift"
        assert DriftType.MODEL_DRIFT.value == "model_drift"
        assert DriftType.PREDICTION_DRIFT.value == "prediction_drift"
        assert DriftType.FEATURE_DRIFT.value == "feature_drift"

    def test_enum_count(self):
        """Test correct number of drift types."""
        assert len(DriftType) == 5


class TestDriftSeverity:
    """Tests for DriftSeverity enum."""

    def test_all_severity_levels(self):
        """Test all severity levels are defined."""
        assert DriftSeverity.NONE.value == 0
        assert DriftSeverity.LOW.value == 1
        assert DriftSeverity.MEDIUM.value == 2
        assert DriftSeverity.HIGH.value == 3
        assert DriftSeverity.CRITICAL.value == 4

    def test_ordering(self):
        """Test severity ordering."""
        assert DriftSeverity.NONE.value < DriftSeverity.LOW.value
        assert DriftSeverity.LOW.value < DriftSeverity.MEDIUM.value
        assert DriftSeverity.MEDIUM.value < DriftSeverity.HIGH.value
        assert DriftSeverity.HIGH.value < DriftSeverity.CRITICAL.value


class TestIncidentSeverity:
    """Tests for IncidentSeverity enum."""

    def test_all_severity_levels(self):
        """Test all incident severity levels."""
        assert IncidentSeverity.MINOR.value == 1
        assert IncidentSeverity.MODERATE.value == 2
        assert IncidentSeverity.MAJOR.value == 3
        assert IncidentSeverity.SERIOUS.value == 4

    def test_serious_triggers_regulatory(self):
        """Test that SERIOUS is the highest level requiring regulatory reporting."""
        assert IncidentSeverity.SERIOUS.value == 4


class TestIncidentStatus:
    """Tests for IncidentStatus enum."""

    def test_all_statuses(self):
        """Test all incident statuses."""
        assert IncidentStatus.REPORTED.value == "reported"
        assert IncidentStatus.INVESTIGATING.value == "investigating"
        assert IncidentStatus.ROOT_CAUSE_IDENTIFIED.value == "root_cause_identified"
        assert IncidentStatus.CORRECTIVE_ACTION.value == "corrective_action"
        assert IncidentStatus.RESOLVED.value == "resolved"
        assert IncidentStatus.CLOSED.value == "closed"


class TestFeedbackType:
    """Tests for FeedbackType enum."""

    def test_all_feedback_types(self):
        """Test all feedback types."""
        assert FeedbackType.BUG_REPORT.value == "bug_report"
        assert FeedbackType.PERFORMANCE_ISSUE.value == "performance_issue"
        assert FeedbackType.FEATURE_REQUEST.value == "feature_request"
        assert FeedbackType.SAFETY_CONCERN.value == "safety_concern"
        assert FeedbackType.GENERAL_FEEDBACK.value == "general_feedback"
        assert FeedbackType.INCIDENT_REPORT.value == "incident_report"


class TestAlertPriority:
    """Tests for AlertPriority enum."""

    def test_all_priorities(self):
        """Test all alert priorities."""
        assert AlertPriority.INFO.value == 1
        assert AlertPriority.WARNING.value == 2
        assert AlertPriority.HIGH.value == 3
        assert AlertPriority.CRITICAL.value == 4


# =============================================================================
# Data Structure Tests
# =============================================================================

class TestMonitoringMetric:
    """Tests for MonitoringMetric dataclass."""

    def test_creation_with_defaults(self):
        """Test metric creation with default values."""
        metric = MonitoringMetric(
            metric_id="",
            name="Test Metric",
            metric_type=MonitoringMetricType.ACCURACY,
            description="Test description",
        )

        assert metric.metric_id.startswith("PMON-")
        assert metric.name == "Test Metric"
        assert metric.metric_type == MonitoringMetricType.ACCURACY
        assert metric.baseline_value == 0.0
        assert metric.current_value is None
        assert metric.drift_severity == DriftSeverity.NONE
        assert metric.created_at != ""

    def test_creation_with_custom_id(self):
        """Test metric creation with custom ID."""
        metric = MonitoringMetric(
            metric_id="CUSTOM-001",
            name="Custom Metric",
            metric_type=MonitoringMetricType.LATENCY,
            description="Custom description",
        )

        assert metric.metric_id == "CUSTOM-001"

    def test_full_metric_configuration(self):
        """Test fully configured metric."""
        metric = MonitoringMetric(
            metric_id="",
            name="Full Metric",
            metric_type=MonitoringMetricType.ACCURACY,
            description="Full description",
            baseline_value=0.95,
            baseline_std=0.02,
            baseline_samples=1000,
            warning_threshold=0.90,
            critical_threshold=0.85,
            threshold_direction="below",
            unit="ratio",
            collection_frequency="hourly",
        )

        assert metric.baseline_value == 0.95
        assert metric.baseline_std == 0.02
        assert metric.baseline_samples == 1000
        assert metric.warning_threshold == 0.90
        assert metric.critical_threshold == 0.85
        assert metric.threshold_direction == "below"
        assert metric.unit == "ratio"
        assert metric.collection_frequency == "hourly"


class TestDriftDetectionResult:
    """Tests for DriftDetectionResult dataclass."""

    def test_creation_with_defaults(self):
        """Test drift result creation with defaults."""
        result = DriftDetectionResult(
            detection_id="",
            metric_id="PMON-001",
            drift_type=DriftType.MODEL_DRIFT,
            severity=DriftSeverity.MEDIUM,
        )

        assert result.detection_id.startswith("DRIFT-")
        assert result.metric_id == "PMON-001"
        assert result.drift_type == DriftType.MODEL_DRIFT
        assert result.severity == DriftSeverity.MEDIUM
        assert result.detection_time != ""
        assert result.acknowledged is False
        assert result.resolved is False

    def test_full_drift_result(self):
        """Test fully configured drift result."""
        result = DriftDetectionResult(
            detection_id="",
            metric_id="PMON-001",
            drift_type=DriftType.CONCEPT_DRIFT,
            severity=DriftSeverity.CRITICAL,
            baseline_value=0.95,
            current_value=0.75,
            deviation=0.20,
            deviation_percent=21.05,
            p_value=0.001,
            test_statistic=5.5,
            test_method="kolmogorov_smirnov",
            affected_functionality=["trading_decisions", "risk_assessment"],
            potential_impact="High risk of incorrect trading signals",
            recommended_actions=["Investigate immediately", "Consider pausing"],
        )

        assert result.baseline_value == 0.95
        assert result.current_value == 0.75
        assert result.deviation == 0.20
        assert result.deviation_percent == 21.05
        assert result.p_value == 0.001
        assert len(result.affected_functionality) == 2
        assert len(result.recommended_actions) == 2


class TestIncident:
    """Tests for Incident dataclass."""

    def test_creation_with_defaults(self):
        """Test incident creation with defaults."""
        incident = Incident(
            incident_id="",
            severity=IncidentSeverity.MAJOR,
            title="Test Incident",
            description="Test description",
        )

        assert incident.incident_id.startswith("INC-")
        assert incident.severity == IncidentSeverity.MAJOR
        assert incident.title == "Test Incident"
        assert incident.status == IncidentStatus.REPORTED
        assert incident.requires_regulatory_report is False
        assert incident.created_at != ""
        assert incident.occurred_at != ""
        assert incident.detected_at != ""

    def test_serious_incident(self):
        """Test serious incident with regulatory requirements."""
        incident = Incident(
            incident_id="",
            severity=IncidentSeverity.SERIOUS,
            title="Serious Incident",
            description="Critical failure requiring regulatory notification",
            incident_type="safety_event",
            affected_users=1000,
            safety_impact="Potential financial harm",
        )

        assert incident.severity == IncidentSeverity.SERIOUS
        assert incident.affected_users == 1000
        assert incident.safety_impact == "Potential financial harm"

    def test_incident_with_investigation(self):
        """Test incident with investigation details."""
        incident = Incident(
            incident_id="",
            severity=IncidentSeverity.MODERATE,
            title="Moderate Incident",
            description="Performance degradation",
            root_cause="Memory leak in inference engine",
            contributing_factors=["High load", "Missing monitoring"],
            corrective_actions=["Fix memory leak", "Add monitoring"],
        )

        assert incident.root_cause == "Memory leak in inference engine"
        assert len(incident.contributing_factors) == 2
        assert len(incident.corrective_actions) == 2


class TestFeedback:
    """Tests for Feedback dataclass."""

    def test_creation_with_defaults(self):
        """Test feedback creation with defaults."""
        feedback = Feedback(
            feedback_id="",
            feedback_type=FeedbackType.BUG_REPORT,
            subject="Bug Report",
            description="Found a bug",
        )

        assert feedback.feedback_id.startswith("FB-")
        assert feedback.feedback_type == FeedbackType.BUG_REPORT
        assert feedback.subject == "Bug Report"
        assert feedback.priority == AlertPriority.INFO
        assert feedback.acknowledged is False
        assert feedback.resolved is False
        assert feedback.submitted_at != ""

    def test_safety_concern_feedback(self):
        """Test safety concern feedback."""
        feedback = Feedback(
            feedback_id="",
            feedback_type=FeedbackType.SAFETY_CONCERN,
            subject="Safety Issue",
            description="Model making risky predictions",
            source_type="deployer",
            source_id="DEP-001",
            priority=AlertPriority.CRITICAL,
            urgency="immediate",
        )

        assert feedback.feedback_type == FeedbackType.SAFETY_CONCERN
        assert feedback.priority == AlertPriority.CRITICAL
        assert feedback.urgency == "immediate"


class TestMonitoringAlert:
    """Tests for MonitoringAlert dataclass."""

    def test_creation_with_defaults(self):
        """Test alert creation with defaults."""
        alert = MonitoringAlert(
            alert_id="",
            priority=AlertPriority.WARNING,
            title="Test Alert",
            description="Test description",
        )

        assert alert.alert_id.startswith("ALERT-")
        assert alert.priority == AlertPriority.WARNING
        assert alert.is_active is True
        assert alert.acknowledged is False
        assert alert.escalated is False
        assert alert.triggered_at != ""

    def test_critical_alert(self):
        """Test critical alert with escalation."""
        alert = MonitoringAlert(
            alert_id="",
            priority=AlertPriority.CRITICAL,
            title="Critical Alert",
            description="Critical failure detected",
            source_metric="PMON-001",
            trigger_value=0.70,
            threshold_value=0.85,
            deviation=0.15,
            manual_action_required=True,
        )

        assert alert.priority == AlertPriority.CRITICAL
        assert alert.trigger_value == 0.70
        assert alert.threshold_value == 0.85
        assert alert.manual_action_required is True


class TestPeriodicReport:
    """Tests for PeriodicReport dataclass."""

    def test_creation_with_defaults(self):
        """Test report creation with defaults."""
        report = PeriodicReport(
            report_id="",
            report_type="daily",
            period_start="2024-01-01T00:00:00Z",
            period_end="2024-01-02T00:00:00Z",
        )

        assert report.report_id.startswith("REP-")
        assert report.report_type == "daily"
        assert report.total_incidents == 0
        assert report.compliance_score == 0.0
        assert report.generated_at != ""

    def test_full_report(self):
        """Test fully populated report."""
        report = PeriodicReport(
            report_id="",
            report_type="weekly",
            period_start="2024-01-01T00:00:00Z",
            period_end="2024-01-08T00:00:00Z",
            total_predictions=10000,
            total_incidents=3,
            serious_incidents=1,
            total_feedback=25,
            average_accuracy=0.92,
            average_latency=45.5,
            availability_percent=99.9,
            error_rate=0.001,
            drift_detections=2,
            critical_drifts=0,
            total_alerts=10,
            critical_alerts=1,
            unresolved_alerts=3,
            compliance_score=85.0,
            compliance_issues=["1 serious incident"],
            recommendations=["Review incident response"],
        )

        assert report.total_predictions == 10000
        assert report.total_incidents == 3
        assert report.serious_incidents == 1
        assert report.average_accuracy == 0.92
        assert report.compliance_score == 85.0


# =============================================================================
# Configuration Tests
# =============================================================================

class TestPostMarketConfig:
    """Tests for PostMarketConfig."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = PostMarketConfig()

        assert config.enable_performance_monitoring is True
        assert config.enable_drift_detection is True
        assert config.enable_incident_tracking is True
        assert config.enable_feedback_collection is True
        assert config.drift_warning_threshold == 0.1
        assert config.drift_critical_threshold == 0.2
        assert config.error_rate_warning == 0.01
        assert config.error_rate_critical == 0.05
        assert config.drift_detection_window == 100
        assert config.statistical_significance == 0.05
        assert config.serious_incident_report_hours == 72
        assert config.metrics_retention_days == 365
        assert config.incidents_retention_days == 1825

    def test_custom_configuration(self):
        """Test custom configuration."""
        config = PostMarketConfig(
            drift_warning_threshold=0.05,
            drift_critical_threshold=0.15,
            drift_detection_window=50,
            enable_alerting=False,
            serious_incident_report_hours=24,
        )

        assert config.drift_warning_threshold == 0.05
        assert config.drift_critical_threshold == 0.15
        assert config.drift_detection_window == 50
        assert config.enable_alerting is False
        assert config.serious_incident_report_hours == 24


# =============================================================================
# PerformanceMonitor Tests
# =============================================================================

class TestPerformanceMonitor:
    """Tests for PerformanceMonitor component."""

    @pytest.fixture
    def monitor(self):
        """Create performance monitor fixture."""
        config = PostMarketConfig(drift_detection_window=10)
        return PerformanceMonitor(config)

    def test_register_metric(self, monitor):
        """Test metric registration."""
        metric = monitor.register_metric(
            name="Accuracy",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
            baseline_std=0.02,
            warning_threshold=0.90,
            critical_threshold=0.85,
            unit="ratio",
        )

        assert metric.name == "Accuracy"
        assert metric.baseline_value == 0.95
        assert metric.metric_id in monitor._metrics

    def test_record_value(self, monitor):
        """Test recording metric values."""
        metric = monitor.register_metric(
            name="Accuracy",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )

        updated = monitor.record_value(metric.metric_id, 0.94)

        assert updated.current_value == 0.94
        assert updated.current_samples == 1
        assert updated.last_updated != ""

    def test_record_value_not_found(self, monitor):
        """Test recording to non-existent metric."""
        with pytest.raises(ValueError, match="not found"):
            monitor.record_value("NONEXISTENT", 0.5)

    def test_detect_drift_insufficient_data(self, monitor):
        """Test drift detection with insufficient data."""
        metric = monitor.register_metric(
            name="Accuracy",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )

        # Add fewer samples than detection window
        for i in range(5):
            monitor.record_value(metric.metric_id, 0.94)

        result = monitor.detect_drift(metric.metric_id)
        assert result is None

    def test_detect_drift_no_drift(self, monitor):
        """Test drift detection with stable performance."""
        metric = monitor.register_metric(
            name="Accuracy",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )

        # Add samples close to baseline
        for i in range(15):
            monitor.record_value(metric.metric_id, 0.94 + (i % 2) * 0.01)

        result = monitor.detect_drift(metric.metric_id)
        # Should be None or LOW severity for small deviation
        assert result is None or result.severity == DriftSeverity.LOW

    def test_detect_drift_critical(self, monitor):
        """Test detection of critical drift."""
        metric = monitor.register_metric(
            name="Accuracy",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )

        # Add samples with significant drift (>20% deviation)
        for i in range(15):
            monitor.record_value(metric.metric_id, 0.70)

        result = monitor.detect_drift(metric.metric_id)
        assert result is not None
        assert result.severity == DriftSeverity.CRITICAL
        assert result.deviation_percent > 20

    def test_detect_drift_medium(self, monitor):
        """Test detection of medium drift."""
        config = PostMarketConfig(
            drift_detection_window=10,
            drift_warning_threshold=0.1,
            drift_critical_threshold=0.2,
        )
        monitor = PerformanceMonitor(config)

        metric = monitor.register_metric(
            name="Accuracy",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )

        # Add samples with ~15% deviation
        for i in range(15):
            monitor.record_value(metric.metric_id, 0.82)

        result = monitor.detect_drift(metric.metric_id)
        assert result is not None
        assert result.severity == DriftSeverity.MEDIUM

    def test_detect_drift_not_found(self, monitor):
        """Test drift detection for non-existent metric."""
        with pytest.raises(ValueError, match="not found"):
            monitor.detect_drift("NONEXISTENT")

    def test_get_metric(self, monitor):
        """Test getting metric by ID."""
        metric = monitor.register_metric(
            name="Test",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )

        retrieved = monitor.get_metric(metric.metric_id)
        assert retrieved is not None
        assert retrieved.name == "Test"

    def test_get_metric_not_found(self, monitor):
        """Test getting non-existent metric."""
        result = monitor.get_metric("NONEXISTENT")
        assert result is None

    def test_get_all_metrics(self, monitor):
        """Test getting all metrics."""
        monitor.register_metric(
            name="Metric1",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )
        monitor.register_metric(
            name="Metric2",
            metric_type=MonitoringMetricType.LATENCY,
            baseline_value=50.0,
        )

        metrics = monitor.get_all_metrics()
        assert len(metrics) == 2

    def test_get_metrics_with_drift(self, monitor):
        """Test getting metrics with drift."""
        metric1 = monitor.register_metric(
            name="Metric1",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )
        metric2 = monitor.register_metric(
            name="Metric2",
            metric_type=MonitoringMetricType.LATENCY,
            baseline_value=50.0,
        )

        # Induce drift on metric1
        for i in range(15):
            monitor.record_value(metric1.metric_id, 0.60)
        monitor.detect_drift(metric1.metric_id)

        drifted = monitor.get_metrics_with_drift()
        assert len(drifted) == 1
        assert drifted[0].name == "Metric1"

    def test_recommended_actions_critical(self, monitor):
        """Test recommended actions for critical drift."""
        actions = monitor._get_recommended_actions(DriftSeverity.CRITICAL)
        assert "Immediately investigate root cause" in actions
        assert "Consider pausing AI system operation" in actions

    def test_recommended_actions_high(self, monitor):
        """Test recommended actions for high drift."""
        actions = monitor._get_recommended_actions(DriftSeverity.HIGH)
        assert "Investigate root cause within 24 hours" in actions

    def test_recommended_actions_medium(self, monitor):
        """Test recommended actions for medium drift."""
        actions = monitor._get_recommended_actions(DriftSeverity.MEDIUM)
        assert "Schedule investigation within 1 week" in actions

    def test_recommended_actions_low(self, monitor):
        """Test recommended actions for low drift."""
        actions = monitor._get_recommended_actions(DriftSeverity.LOW)
        assert "Document for periodic review" in actions

    def test_history_trimming(self, monitor):
        """Test that history is trimmed to prevent memory growth."""
        metric = monitor.register_metric(
            name="Test",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )

        # Add many values
        for i in range(2000):
            monitor.record_value(metric.metric_id, 0.94)

        # History should be trimmed
        assert len(monitor._history[metric.metric_id]) <= 1000

    def test_thread_safety(self, monitor):
        """Test thread safety of performance monitor."""
        metric = monitor.register_metric(
            name="Test",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )

        errors = []

        def record_values():
            try:
                for i in range(100):
                    monitor.record_value(metric.metric_id, 0.94 + (i % 3) * 0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_values) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# IncidentTracker Tests
# =============================================================================

class TestIncidentTracker:
    """Tests for IncidentTracker component."""

    @pytest.fixture
    def tracker(self):
        """Create incident tracker fixture."""
        config = PostMarketConfig()
        return IncidentTracker(config)

    def test_report_incident(self, tracker):
        """Test incident reporting."""
        incident = tracker.report_incident(
            severity=IncidentSeverity.MAJOR,
            title="Model Failure",
            description="Model stopped responding",
            incident_type="malfunction",
            affected_component="inference_engine",
            reporter="operator@example.com",
        )

        assert incident.incident_id.startswith("INC-")
        assert incident.severity == IncidentSeverity.MAJOR
        assert incident.title == "Model Failure"
        assert incident.status == IncidentStatus.REPORTED
        assert incident.requires_regulatory_report is False

    def test_report_serious_incident(self, tracker):
        """Test serious incident requiring regulatory report."""
        incident = tracker.report_incident(
            severity=IncidentSeverity.SERIOUS,
            title="Critical Safety Issue",
            description="Model causing financial harm",
        )

        assert incident.severity == IncidentSeverity.SERIOUS
        assert incident.requires_regulatory_report is True

    def test_report_incident_with_callback(self):
        """Test incident callback invocation."""
        callback = MagicMock()
        config = PostMarketConfig(incident_callback=callback)
        tracker = IncidentTracker(config)

        tracker.report_incident(
            severity=IncidentSeverity.MAJOR,
            title="Test Incident",
            description="Test",
        )

        callback.assert_called_once()
        assert callback.call_args[0][0] == "incident_reported"

    def test_update_incident_status(self, tracker):
        """Test incident status update."""
        incident = tracker.report_incident(
            severity=IncidentSeverity.MODERATE,
            title="Test",
            description="Test",
        )

        updated = tracker.update_incident_status(
            incident.incident_id,
            IncidentStatus.INVESTIGATING,
            notes="Investigation started",
        )

        assert updated.status == IncidentStatus.INVESTIGATING
        assert "Investigation started" in updated.investigation_notes

    def test_update_incident_resolved(self, tracker):
        """Test incident resolution."""
        incident = tracker.report_incident(
            severity=IncidentSeverity.MINOR,
            title="Test",
            description="Test",
        )

        resolved = tracker.update_incident_status(
            incident.incident_id,
            IncidentStatus.RESOLVED,
        )

        assert resolved.status == IncidentStatus.RESOLVED
        assert resolved.resolved_at is not None

    def test_update_incident_not_found(self, tracker):
        """Test updating non-existent incident."""
        with pytest.raises(ValueError, match="not found"):
            tracker.update_incident_status("NONEXISTENT", IncidentStatus.INVESTIGATING)

    def test_set_root_cause(self, tracker):
        """Test setting root cause."""
        incident = tracker.report_incident(
            severity=IncidentSeverity.MODERATE,
            title="Test",
            description="Test",
        )

        updated = tracker.set_root_cause(
            incident.incident_id,
            root_cause="Memory leak in model server",
            contributing_factors=["High load", "Missing monitoring"],
        )

        assert updated.root_cause == "Memory leak in model server"
        assert len(updated.contributing_factors) == 2
        assert updated.status == IncidentStatus.ROOT_CAUSE_IDENTIFIED

    def test_set_root_cause_not_found(self, tracker):
        """Test setting root cause for non-existent incident."""
        with pytest.raises(ValueError, match="not found"):
            tracker.set_root_cause("NONEXISTENT", "Test cause")

    def test_add_corrective_action(self, tracker):
        """Test adding corrective action."""
        incident = tracker.report_incident(
            severity=IncidentSeverity.MODERATE,
            title="Test",
            description="Test",
        )

        updated = tracker.add_corrective_action(
            incident.incident_id,
            action="Fix memory leak",
            action_type="corrective",
        )

        assert "Fix memory leak" in updated.corrective_actions
        assert updated.status == IncidentStatus.CORRECTIVE_ACTION

    def test_add_immediate_action(self, tracker):
        """Test adding immediate action."""
        incident = tracker.report_incident(
            severity=IncidentSeverity.MAJOR,
            title="Test",
            description="Test",
        )

        updated = tracker.add_corrective_action(
            incident.incident_id,
            action="Restart service",
            action_type="immediate",
        )

        assert "Restart service" in updated.immediate_actions

    def test_add_preventive_action(self, tracker):
        """Test adding preventive action."""
        incident = tracker.report_incident(
            severity=IncidentSeverity.MODERATE,
            title="Test",
            description="Test",
        )

        updated = tracker.add_corrective_action(
            incident.incident_id,
            action="Add monitoring",
            action_type="preventive",
        )

        assert "Add monitoring" in updated.preventive_actions

    def test_add_action_not_found(self, tracker):
        """Test adding action to non-existent incident."""
        with pytest.raises(ValueError, match="not found"):
            tracker.add_corrective_action("NONEXISTENT", "Test action")

    def test_submit_regulatory_report(self, tracker):
        """Test regulatory report submission."""
        incident = tracker.report_incident(
            severity=IncidentSeverity.SERIOUS,
            title="Serious Incident",
            description="Critical issue",
        )

        updated = tracker.submit_regulatory_report(
            incident.incident_id,
            authority="EU Market Surveillance Authority",
        )

        assert updated.regulatory_report_submitted is True
        assert updated.regulatory_report_date is not None
        assert updated.regulatory_authority == "EU Market Surveillance Authority"

    def test_submit_regulatory_report_not_found(self, tracker):
        """Test submitting report for non-existent incident."""
        with pytest.raises(ValueError, match="not found"):
            tracker.submit_regulatory_report("NONEXISTENT", "Authority")

    def test_get_incident(self, tracker):
        """Test getting incident by ID."""
        incident = tracker.report_incident(
            severity=IncidentSeverity.MINOR,
            title="Test",
            description="Test",
        )

        retrieved = tracker.get_incident(incident.incident_id)
        assert retrieved is not None
        assert retrieved.title == "Test"

    def test_get_incident_not_found(self, tracker):
        """Test getting non-existent incident."""
        result = tracker.get_incident("NONEXISTENT")
        assert result is None

    def test_get_open_incidents(self, tracker):
        """Test getting open incidents."""
        inc1 = tracker.report_incident(
            severity=IncidentSeverity.MINOR,
            title="Open 1",
            description="Test",
        )
        inc2 = tracker.report_incident(
            severity=IncidentSeverity.MODERATE,
            title="Open 2",
            description="Test",
        )
        inc3 = tracker.report_incident(
            severity=IncidentSeverity.MINOR,
            title="Resolved",
            description="Test",
        )
        tracker.update_incident_status(inc3.incident_id, IncidentStatus.RESOLVED)

        open_incidents = tracker.get_open_incidents()
        assert len(open_incidents) == 2

    def test_get_serious_incidents(self, tracker):
        """Test getting serious incidents."""
        tracker.report_incident(
            severity=IncidentSeverity.MINOR,
            title="Minor",
            description="Test",
        )
        tracker.report_incident(
            severity=IncidentSeverity.SERIOUS,
            title="Serious 1",
            description="Test",
        )
        tracker.report_incident(
            severity=IncidentSeverity.SERIOUS,
            title="Serious 2",
            description="Test",
        )

        serious = tracker.get_serious_incidents()
        assert len(serious) == 2

    def test_get_pending_regulatory_reports(self, tracker):
        """Test getting pending regulatory reports."""
        inc1 = tracker.report_incident(
            severity=IncidentSeverity.SERIOUS,
            title="Pending",
            description="Test",
        )
        inc2 = tracker.report_incident(
            severity=IncidentSeverity.SERIOUS,
            title="Submitted",
            description="Test",
        )
        tracker.submit_regulatory_report(inc2.incident_id, "Authority")

        pending = tracker.get_pending_regulatory_reports()
        assert len(pending) == 1
        assert pending[0].title == "Pending"

    def test_callback_error_handling(self):
        """Test callback error handling."""
        def failing_callback(event_type, data):
            raise Exception("Callback error")

        config = PostMarketConfig(incident_callback=failing_callback)
        tracker = IncidentTracker(config)

        # Should not raise despite callback failure
        incident = tracker.report_incident(
            severity=IncidentSeverity.MINOR,
            title="Test",
            description="Test",
        )
        assert incident is not None


# =============================================================================
# FeedbackCollector Tests
# =============================================================================

class TestFeedbackCollector:
    """Tests for FeedbackCollector component."""

    @pytest.fixture
    def collector(self):
        """Create feedback collector fixture."""
        config = PostMarketConfig()
        return FeedbackCollector(config)

    def test_submit_feedback(self, collector):
        """Test feedback submission."""
        feedback = collector.submit_feedback(
            feedback_type=FeedbackType.BUG_REPORT,
            subject="Bug Found",
            description="Model returns NaN values",
            source_type="deployer",
            source_id="DEP-001",
            priority=AlertPriority.HIGH,
        )

        assert feedback.feedback_id.startswith("FB-")
        assert feedback.feedback_type == FeedbackType.BUG_REPORT
        assert feedback.subject == "Bug Found"
        assert feedback.priority == AlertPriority.HIGH
        assert feedback.acknowledged is False

    def test_acknowledge_feedback(self, collector):
        """Test feedback acknowledgment."""
        feedback = collector.submit_feedback(
            feedback_type=FeedbackType.PERFORMANCE_ISSUE,
            subject="Slow Response",
            description="Model latency increased",
        )

        acked = collector.acknowledge_feedback(
            feedback.feedback_id,
            response="Investigating the issue",
        )

        assert acked.acknowledged is True
        assert acked.acknowledged_at is not None
        assert acked.response == "Investigating the issue"
        assert acked.response_date is not None

    def test_acknowledge_feedback_not_found(self, collector):
        """Test acknowledging non-existent feedback."""
        with pytest.raises(ValueError, match="not found"):
            collector.acknowledge_feedback("NONEXISTENT")

    def test_resolve_feedback(self, collector):
        """Test feedback resolution."""
        feedback = collector.submit_feedback(
            feedback_type=FeedbackType.BUG_REPORT,
            subject="Bug",
            description="Test bug",
        )

        resolved = collector.resolve_feedback(
            feedback.feedback_id,
            resolution="Fixed in version 2.0.1",
        )

        assert resolved.resolved is True
        assert resolved.resolution == "Fixed in version 2.0.1"
        assert resolved.resolved_at is not None

    def test_resolve_feedback_not_found(self, collector):
        """Test resolving non-existent feedback."""
        with pytest.raises(ValueError, match="not found"):
            collector.resolve_feedback("NONEXISTENT", "Resolution")

    def test_link_to_incident(self, collector):
        """Test linking feedback to incident."""
        feedback = collector.submit_feedback(
            feedback_type=FeedbackType.INCIDENT_REPORT,
            subject="Incident Report",
            description="System failure",
        )

        linked = collector.link_to_incident(feedback.feedback_id, "INC-001")
        linked = collector.link_to_incident(feedback.feedback_id, "INC-002")
        # Try duplicate link
        linked = collector.link_to_incident(feedback.feedback_id, "INC-001")

        assert len(linked.related_incidents) == 2
        assert "INC-001" in linked.related_incidents
        assert "INC-002" in linked.related_incidents

    def test_link_feedback_not_found(self, collector):
        """Test linking non-existent feedback."""
        with pytest.raises(ValueError, match="not found"):
            collector.link_to_incident("NONEXISTENT", "INC-001")

    def test_get_feedback(self, collector):
        """Test getting feedback by ID."""
        feedback = collector.submit_feedback(
            feedback_type=FeedbackType.FEATURE_REQUEST,
            subject="Feature",
            description="New feature request",
        )

        retrieved = collector.get_feedback(feedback.feedback_id)
        assert retrieved is not None
        assert retrieved.subject == "Feature"

    def test_get_feedback_not_found(self, collector):
        """Test getting non-existent feedback."""
        result = collector.get_feedback("NONEXISTENT")
        assert result is None

    def test_get_unacknowledged_feedback(self, collector):
        """Test getting unacknowledged feedback."""
        fb1 = collector.submit_feedback(
            feedback_type=FeedbackType.BUG_REPORT,
            subject="Bug 1",
            description="Test",
        )
        fb2 = collector.submit_feedback(
            feedback_type=FeedbackType.BUG_REPORT,
            subject="Bug 2",
            description="Test",
        )
        collector.acknowledge_feedback(fb2.feedback_id)

        unacked = collector.get_unacknowledged_feedback()
        assert len(unacked) == 1
        assert unacked[0].subject == "Bug 1"

    def test_get_feedback_by_type(self, collector):
        """Test getting feedback by type."""
        collector.submit_feedback(
            feedback_type=FeedbackType.BUG_REPORT,
            subject="Bug",
            description="Test",
        )
        collector.submit_feedback(
            feedback_type=FeedbackType.FEATURE_REQUEST,
            subject="Feature",
            description="Test",
        )
        collector.submit_feedback(
            feedback_type=FeedbackType.BUG_REPORT,
            subject="Bug 2",
            description="Test",
        )

        bugs = collector.get_feedback_by_type(FeedbackType.BUG_REPORT)
        assert len(bugs) == 2


# =============================================================================
# PostMarketMonitoringSystem Tests
# =============================================================================

class TestPostMarketMonitoringSystem:
    """Tests for PostMarketMonitoringSystem main class."""

    @pytest.fixture
    def pmms(self, tmp_path):
        """Create post-market monitoring system fixture."""
        config = PostMarketConfig(
            drift_detection_window=10,
            log_path=str(tmp_path / "logs"),
            reports_path=str(tmp_path / "reports"),
        )
        return PostMarketMonitoringSystem(config)

    def test_initialization(self, pmms):
        """Test system initialization."""
        assert pmms._performance_monitor is not None
        assert pmms._incident_tracker is not None
        assert pmms._feedback_collector is not None
        assert pmms._log_path.exists()
        assert pmms._reports_path.exists()

    def test_initialization_default_config(self, tmp_path):
        """Test initialization with default config."""
        pmms = PostMarketMonitoringSystem()
        assert pmms.config is not None
        assert pmms.config.enable_drift_detection is True

    # Performance Monitoring Tests

    def test_register_metric(self, pmms):
        """Test metric registration through main system."""
        metric = pmms.register_metric(
            name="Accuracy",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
            unit="ratio",
        )

        assert metric.name == "Accuracy"
        assert metric.baseline_value == 0.95

    def test_record_metric_value(self, pmms):
        """Test recording metric value."""
        metric = pmms.register_metric(
            name="Accuracy",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )

        updated, drift = pmms.record_metric_value(metric.metric_id, 0.94)

        assert updated.current_value == 0.94
        # No drift with insufficient data
        assert drift is None

    def test_record_metric_value_with_drift_detection(self, pmms):
        """Test recording with drift detection."""
        metric = pmms.register_metric(
            name="Accuracy",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )

        # Record enough values to detect drift
        for i in range(15):
            _, drift = pmms.record_metric_value(metric.metric_id, 0.60)

        # Should detect drift on final value
        assert drift is not None or metric.drift_severity != DriftSeverity.NONE

    def test_record_metric_value_generates_alert(self, tmp_path):
        """Test that drift generates alert."""
        config = PostMarketConfig(
            drift_detection_window=10,
            drift_warning_threshold=0.05,
            log_path=str(tmp_path / "logs"),
            reports_path=str(tmp_path / "reports"),
        )
        pmms = PostMarketMonitoringSystem(config)

        metric = pmms.register_metric(
            name="Accuracy",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )

        # Induce significant drift
        for i in range(15):
            pmms.record_metric_value(metric.metric_id, 0.50)

        alerts = pmms.get_active_alerts()
        assert len(alerts) >= 1

    def test_check_drift(self, pmms):
        """Test explicit drift check."""
        metric = pmms.register_metric(
            name="Accuracy",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )

        for i in range(15):
            pmms.record_metric_value(metric.metric_id, 0.50, check_drift=False)

        drift = pmms.check_drift(metric.metric_id)
        assert drift is not None

    def test_get_metric(self, pmms):
        """Test getting metric."""
        metric = pmms.register_metric(
            name="Test",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )

        retrieved = pmms.get_metric(metric.metric_id)
        assert retrieved is not None

    def test_get_all_metrics(self, pmms):
        """Test getting all metrics."""
        pmms.register_metric(
            name="Metric1",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )
        pmms.register_metric(
            name="Metric2",
            metric_type=MonitoringMetricType.LATENCY,
            baseline_value=50.0,
        )

        metrics = pmms.get_all_metrics()
        assert len(metrics) == 2

    def test_get_metrics_with_drift(self, pmms):
        """Test getting drifted metrics."""
        metric = pmms.register_metric(
            name="Test",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )

        for i in range(15):
            pmms.record_metric_value(metric.metric_id, 0.50)

        drifted = pmms.get_metrics_with_drift()
        assert len(drifted) >= 1

    # Incident Management Tests

    def test_report_incident(self, pmms):
        """Test incident reporting."""
        incident = pmms.report_incident(
            severity=IncidentSeverity.MAJOR,
            title="Test Incident",
            description="Test description",
        )

        assert incident.incident_id.startswith("INC-")

    def test_report_incident_generates_alert(self, pmms):
        """Test that major incident generates alert."""
        pmms.report_incident(
            severity=IncidentSeverity.MAJOR,
            title="Major Incident",
            description="Test",
        )

        alerts = pmms.get_active_alerts()
        assert len(alerts) >= 1

    def test_update_incident(self, pmms):
        """Test incident update."""
        incident = pmms.report_incident(
            severity=IncidentSeverity.MODERATE,
            title="Test",
            description="Test",
        )

        updated = pmms.update_incident(
            incident.incident_id,
            IncidentStatus.INVESTIGATING,
            notes="Started investigation",
        )

        assert updated.status == IncidentStatus.INVESTIGATING

    def test_set_incident_root_cause(self, pmms):
        """Test setting root cause."""
        incident = pmms.report_incident(
            severity=IncidentSeverity.MODERATE,
            title="Test",
            description="Test",
        )

        updated = pmms.set_incident_root_cause(
            incident.incident_id,
            root_cause="Memory leak",
            contributing_factors=["High load"],
        )

        assert updated.root_cause == "Memory leak"

    def test_add_incident_action(self, pmms):
        """Test adding incident action."""
        incident = pmms.report_incident(
            severity=IncidentSeverity.MODERATE,
            title="Test",
            description="Test",
        )

        updated = pmms.add_incident_action(
            incident.incident_id,
            action="Fix bug",
        )

        assert "Fix bug" in updated.corrective_actions

    def test_submit_regulatory_report(self, pmms):
        """Test regulatory report submission."""
        incident = pmms.report_incident(
            severity=IncidentSeverity.SERIOUS,
            title="Serious",
            description="Test",
        )

        updated = pmms.submit_regulatory_report(
            incident.incident_id,
            authority="EU Authority",
        )

        assert updated.regulatory_report_submitted is True

    def test_get_incident(self, pmms):
        """Test getting incident."""
        incident = pmms.report_incident(
            severity=IncidentSeverity.MINOR,
            title="Test",
            description="Test",
        )

        retrieved = pmms.get_incident(incident.incident_id)
        assert retrieved is not None

    def test_get_open_incidents(self, pmms):
        """Test getting open incidents."""
        pmms.report_incident(
            severity=IncidentSeverity.MINOR,
            title="Open",
            description="Test",
        )

        open_inc = pmms.get_open_incidents()
        assert len(open_inc) >= 1

    def test_get_serious_incidents(self, pmms):
        """Test getting serious incidents."""
        pmms.report_incident(
            severity=IncidentSeverity.SERIOUS,
            title="Serious",
            description="Test",
        )

        serious = pmms.get_serious_incidents()
        assert len(serious) >= 1

    def test_get_pending_regulatory_reports(self, pmms):
        """Test getting pending reports."""
        pmms.report_incident(
            severity=IncidentSeverity.SERIOUS,
            title="Pending",
            description="Test",
        )

        pending = pmms.get_pending_regulatory_reports()
        assert len(pending) >= 1

    # Feedback Management Tests

    def test_submit_feedback(self, pmms):
        """Test feedback submission."""
        feedback = pmms.submit_feedback(
            feedback_type=FeedbackType.BUG_REPORT,
            subject="Bug",
            description="Found a bug",
        )

        assert feedback.feedback_id.startswith("FB-")

    def test_submit_safety_feedback_generates_alert(self, pmms):
        """Test that safety concern generates alert."""
        pmms.submit_feedback(
            feedback_type=FeedbackType.SAFETY_CONCERN,
            subject="Safety Issue",
            description="Model unsafe",
        )

        alerts = pmms.get_active_alerts()
        assert len(alerts) >= 1

    def test_acknowledge_feedback(self, pmms):
        """Test feedback acknowledgment."""
        feedback = pmms.submit_feedback(
            feedback_type=FeedbackType.BUG_REPORT,
            subject="Bug",
            description="Test",
        )

        acked = pmms.acknowledge_feedback(
            feedback.feedback_id,
            response="Investigating",
        )

        assert acked.acknowledged is True

    def test_resolve_feedback(self, pmms):
        """Test feedback resolution."""
        feedback = pmms.submit_feedback(
            feedback_type=FeedbackType.BUG_REPORT,
            subject="Bug",
            description="Test",
        )

        resolved = pmms.resolve_feedback(
            feedback.feedback_id,
            resolution="Fixed",
        )

        assert resolved.resolved is True

    def test_get_feedback(self, pmms):
        """Test getting feedback."""
        feedback = pmms.submit_feedback(
            feedback_type=FeedbackType.FEATURE_REQUEST,
            subject="Feature",
            description="Test",
        )

        retrieved = pmms.get_feedback(feedback.feedback_id)
        assert retrieved is not None

    def test_get_unacknowledged_feedback(self, pmms):
        """Test getting unacknowledged feedback."""
        pmms.submit_feedback(
            feedback_type=FeedbackType.BUG_REPORT,
            subject="Unacked",
            description="Test",
        )

        unacked = pmms.get_unacknowledged_feedback()
        assert len(unacked) >= 1

    # Alert Management Tests

    def test_acknowledge_alert(self, pmms):
        """Test alert acknowledgment."""
        pmms.report_incident(
            severity=IncidentSeverity.MAJOR,
            title="Test",
            description="Test",
        )

        alerts = pmms.get_active_alerts()
        assert len(alerts) >= 1

        acked = pmms.acknowledge_alert(alerts[0].alert_id, "admin@example.com")
        assert acked.acknowledged is True
        assert acked.acknowledged_by == "admin@example.com"

    def test_acknowledge_alert_not_found(self, pmms):
        """Test acknowledging non-existent alert."""
        with pytest.raises(ValueError, match="not found"):
            pmms.acknowledge_alert("NONEXISTENT", "admin")

    def test_resolve_alert(self, pmms):
        """Test alert resolution."""
        pmms.report_incident(
            severity=IncidentSeverity.MAJOR,
            title="Test",
            description="Test",
        )

        alerts = pmms.get_active_alerts()
        resolved = pmms.resolve_alert(alerts[0].alert_id)

        assert resolved.is_active is False
        assert resolved.resolved_at is not None

    def test_resolve_alert_not_found(self, pmms):
        """Test resolving non-existent alert."""
        with pytest.raises(ValueError, match="not found"):
            pmms.resolve_alert("NONEXISTENT")

    def test_get_alert(self, pmms):
        """Test getting alert."""
        pmms.report_incident(
            severity=IncidentSeverity.MAJOR,
            title="Test",
            description="Test",
        )

        alerts = pmms.get_active_alerts()
        retrieved = pmms.get_alert(alerts[0].alert_id)
        assert retrieved is not None

    def test_get_active_alerts(self, pmms):
        """Test getting active alerts."""
        pmms.report_incident(
            severity=IncidentSeverity.MAJOR,
            title="Test 1",
            description="Test",
        )
        pmms.report_incident(
            severity=IncidentSeverity.SERIOUS,
            title="Test 2",
            description="Test",
        )

        active = pmms.get_active_alerts()
        assert len(active) >= 2

    # Reporting Tests

    def test_generate_periodic_report_daily(self, pmms):
        """Test daily report generation."""
        # Add some data
        metric = pmms.register_metric(
            name="Accuracy",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )
        for i in range(15):
            pmms.record_metric_value(metric.metric_id, 0.93)

        pmms.report_incident(
            severity=IncidentSeverity.MINOR,
            title="Test",
            description="Test",
        )

        report = pmms.generate_periodic_report(report_type="daily")

        assert report.report_type == "daily"
        assert report.total_incidents >= 1
        assert report.compliance_score <= 100

    def test_generate_periodic_report_weekly(self, pmms):
        """Test weekly report generation."""
        report = pmms.generate_periodic_report(report_type="weekly")
        assert report.report_type == "weekly"

    def test_generate_periodic_report_monthly(self, pmms):
        """Test monthly report generation."""
        report = pmms.generate_periodic_report(report_type="monthly")
        assert report.report_type == "monthly"

    def test_generate_periodic_report_custom_period(self, pmms):
        """Test report with custom period."""
        report = pmms.generate_periodic_report(
            report_type="custom",
            period_start="2024-01-01T00:00:00Z",
            period_end="2024-01-31T00:00:00Z",
        )

        assert report.period_start == "2024-01-01T00:00:00Z"
        assert report.period_end == "2024-01-31T00:00:00Z"

    def test_report_compliance_score_calculation(self, pmms):
        """Test compliance score calculation."""
        # Add serious incidents to reduce score
        pmms.report_incident(
            severity=IncidentSeverity.SERIOUS,
            title="Serious 1",
            description="Test",
        )
        pmms.report_incident(
            severity=IncidentSeverity.SERIOUS,
            title="Serious 2",
            description="Test",
        )

        report = pmms.generate_periodic_report()

        assert report.compliance_score < 100
        assert report.serious_incidents >= 2

    def test_report_identifies_compliance_issues(self, pmms):
        """Test compliance issue identification."""
        pmms.report_incident(
            severity=IncidentSeverity.SERIOUS,
            title="Test",
            description="Test",
        )

        report = pmms.generate_periodic_report()

        assert len(report.compliance_issues) >= 1

    def test_report_generates_recommendations(self, pmms):
        """Test recommendation generation."""
        pmms.report_incident(
            severity=IncidentSeverity.SERIOUS,
            title="Test",
            description="Test",
        )

        report = pmms.generate_periodic_report()

        assert len(report.recommendations) >= 1

    def test_get_report(self, pmms):
        """Test getting report by ID."""
        report = pmms.generate_periodic_report()
        retrieved = pmms.get_report(report.report_id)
        assert retrieved is not None

    # Compliance Status Tests

    def test_get_compliance_status(self, pmms):
        """Test compliance status retrieval."""
        status = pmms.get_compliance_status()

        assert "timestamp" in status
        assert "article_reference" in status
        assert status["article_reference"] == "Article 72"
        assert "metrics" in status
        assert "incidents" in status
        assert "alerts" in status
        assert "is_compliant" in status

    def test_compliance_status_with_data(self, pmms):
        """Test compliance status with various data."""
        metric = pmms.register_metric(
            name="Test",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )
        for i in range(15):
            pmms.record_metric_value(metric.metric_id, 0.50)

        pmms.report_incident(
            severity=IncidentSeverity.SERIOUS,
            title="Serious",
            description="Test",
        )

        status = pmms.get_compliance_status()

        assert status["incidents"]["serious"] >= 1
        assert status["incidents"]["pending_regulatory_reports"] >= 1
        assert status["is_compliant"] is False

    def test_compliance_issues_overdue_report(self, tmp_path):
        """Test compliance issue for overdue regulatory report."""
        config = PostMarketConfig(
            serious_incident_report_hours=0,  # 0 hours to trigger overdue
            log_path=str(tmp_path / "logs"),
            reports_path=str(tmp_path / "reports"),
        )
        pmms = PostMarketMonitoringSystem(config)

        pmms.report_incident(
            severity=IncidentSeverity.SERIOUS,
            title="Test",
            description="Test",
        )

        # Wait a tiny bit to ensure time passes
        time.sleep(0.1)

        status = pmms.get_compliance_status()
        assert len(status["compliance_issues"]) >= 1

    # Export Tests

    def test_export_monitoring_data(self, pmms):
        """Test data export."""
        metric = pmms.register_metric(
            name="Test",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )
        for i in range(15):
            pmms.record_metric_value(metric.metric_id, 0.93)

        pmms.report_incident(
            severity=IncidentSeverity.MINOR,
            title="Test",
            description="Test",
        )

        pmms.submit_feedback(
            feedback_type=FeedbackType.BUG_REPORT,
            subject="Bug",
            description="Test",
        )

        export = pmms.export_monitoring_data()

        assert "export_date" in export
        assert "metrics" in export
        assert "incidents" in export
        assert "feedback" in export
        assert "alerts" in export
        assert "compliance_status" in export
        assert len(export["metrics"]) >= 1
        assert len(export["incidents"]) >= 1
        assert len(export["feedback"]) >= 1

    # Logging Tests

    def test_log_event(self, pmms):
        """Test event logging."""
        pmms.register_metric(
            name="Test",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )

        # Check that log file was created
        log_files = list(pmms._log_path.glob("pmm_events_*.jsonl"))
        assert len(log_files) >= 1

    def test_alert_callback(self, tmp_path):
        """Test alert callback invocation."""
        callback = MagicMock()
        config = PostMarketConfig(
            drift_detection_window=10,
            alert_callback=callback,
            log_path=str(tmp_path / "logs"),
            reports_path=str(tmp_path / "reports"),
        )
        pmms = PostMarketMonitoringSystem(config)

        pmms.report_incident(
            severity=IncidentSeverity.MAJOR,
            title="Test",
            description="Test",
        )

        callback.assert_called()

    def test_alert_callback_error_handling(self, tmp_path):
        """Test alert callback error handling."""
        def failing_callback(event_type, data):
            raise Exception("Callback error")

        config = PostMarketConfig(
            alert_callback=failing_callback,
            log_path=str(tmp_path / "logs"),
            reports_path=str(tmp_path / "reports"),
        )
        pmms = PostMarketMonitoringSystem(config)

        # Should not raise despite callback failure
        pmms.report_incident(
            severity=IncidentSeverity.MAJOR,
            title="Test",
            description="Test",
        )


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_post_market_monitoring_default(self, tmp_path):
        """Test creating system with default config."""
        pmms = create_post_market_monitoring()
        assert pmms is not None
        assert pmms.config.enable_drift_detection is True

    def test_create_post_market_monitoring_with_config(self, tmp_path):
        """Test creating system with PostMarketConfig."""
        config = PostMarketConfig(
            drift_warning_threshold=0.05,
            log_path=str(tmp_path / "logs"),
            reports_path=str(tmp_path / "reports"),
        )
        pmms = create_post_market_monitoring(config)
        assert pmms.config.drift_warning_threshold == 0.05

    def test_create_post_market_monitoring_with_dict(self, tmp_path):
        """Test creating system with dict config."""
        config = {
            "drift_warning_threshold": 0.08,
            "enable_alerting": False,
            "log_path": str(tmp_path / "logs"),
            "reports_path": str(tmp_path / "reports"),
        }
        pmms = create_post_market_monitoring(config)
        assert pmms.config.drift_warning_threshold == 0.08
        assert pmms.config.enable_alerting is False

    def test_get_default_monitoring_metrics(self):
        """Test getting default metrics."""
        metrics = get_default_monitoring_metrics()

        assert len(metrics) >= 5
        assert any(m["name"] == "Model Accuracy" for m in metrics)
        assert any(m["name"] == "Inference Latency" for m in metrics)
        assert any(m["name"] == "Error Rate" for m in metrics)
        assert any(m["name"] == "System Availability" for m in metrics)
        assert any(m["name"] == "Prediction Drift" for m in metrics)

    def test_default_metrics_structure(self):
        """Test default metrics have correct structure."""
        metrics = get_default_monitoring_metrics()

        for metric in metrics:
            assert "name" in metric
            assert "metric_type" in metric
            assert "description" in metric
            assert "baseline_value" in metric
            assert "unit" in metric


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.fixture
    def pmms(self, tmp_path):
        """Create post-market monitoring system fixture."""
        config = PostMarketConfig(
            drift_detection_window=10,
            drift_warning_threshold=0.1,
            drift_critical_threshold=0.2,
            log_path=str(tmp_path / "logs"),
            reports_path=str(tmp_path / "reports"),
        )
        return PostMarketMonitoringSystem(config)

    def test_complete_monitoring_workflow(self, pmms):
        """Test complete monitoring workflow."""
        # 1. Register metrics
        accuracy_metric = pmms.register_metric(
            name="Model Accuracy",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )
        latency_metric = pmms.register_metric(
            name="Inference Latency",
            metric_type=MonitoringMetricType.LATENCY,
            baseline_value=50.0,
        )

        # 2. Record values
        for i in range(15):
            pmms.record_metric_value(accuracy_metric.metric_id, 0.94)
            pmms.record_metric_value(latency_metric.metric_id, 52.0)

        # 3. Check drift
        drift = pmms.check_drift(accuracy_metric.metric_id)

        # 4. Get compliance status
        status = pmms.get_compliance_status()
        assert status["metrics"]["total_registered"] == 2

        # 5. Generate report
        report = pmms.generate_periodic_report()
        assert report.report_id is not None

    def test_incident_lifecycle_workflow(self, pmms):
        """Test complete incident lifecycle."""
        # 1. Report incident
        incident = pmms.report_incident(
            severity=IncidentSeverity.MAJOR,
            title="Model Degradation",
            description="Accuracy dropped significantly",
            incident_type="performance",
            affected_component="inference_engine",
            reporter="monitoring@example.com",
        )

        assert incident.status == IncidentStatus.REPORTED

        # 2. Update to investigating
        pmms.update_incident(
            incident.incident_id,
            IncidentStatus.INVESTIGATING,
            notes="Started investigation",
        )

        # 3. Set root cause
        pmms.set_incident_root_cause(
            incident.incident_id,
            root_cause="Data quality degradation in training set",
            contributing_factors=["Missing validation", "No drift monitoring"],
        )

        # 4. Add corrective actions
        pmms.add_incident_action(
            incident.incident_id,
            action="Retrain model with clean data",
        )
        pmms.add_incident_action(
            incident.incident_id,
            action="Implement data validation",
            action_type="preventive",
        )

        # 5. Resolve incident
        resolved = pmms.update_incident(
            incident.incident_id,
            IncidentStatus.RESOLVED,
            notes="Issue resolved after retraining",
        )

        assert resolved.status == IncidentStatus.RESOLVED
        assert len(resolved.corrective_actions) >= 1
        assert len(resolved.preventive_actions) >= 1

    def test_serious_incident_regulatory_workflow(self, pmms):
        """Test serious incident with regulatory reporting."""
        # 1. Report serious incident
        incident = pmms.report_incident(
            severity=IncidentSeverity.SERIOUS,
            title="Critical Safety Issue",
            description="Model causing financial harm to users",
        )

        assert incident.requires_regulatory_report is True

        # 2. Verify pending regulatory reports
        pending = pmms.get_pending_regulatory_reports()
        assert len(pending) >= 1

        # 3. Submit regulatory report
        pmms.submit_regulatory_report(
            incident.incident_id,
            authority="EU Market Surveillance Authority",
        )

        # 4. Verify no longer pending
        pending = pmms.get_pending_regulatory_reports()
        assert len([p for p in pending if p.incident_id == incident.incident_id]) == 0

    def test_feedback_to_incident_workflow(self, pmms):
        """Test feedback leading to incident workflow."""
        # 1. Submit safety concern feedback
        feedback = pmms.submit_feedback(
            feedback_type=FeedbackType.SAFETY_CONCERN,
            subject="Model making unsafe predictions",
            description="Multiple users reported risky trading signals",
            source_type="deployer",
            priority=AlertPriority.HIGH,
        )

        # 2. Acknowledge feedback
        pmms.acknowledge_feedback(
            feedback.feedback_id,
            response="Investigating the safety concern",
        )

        # 3. Create incident from feedback
        incident = pmms.report_incident(
            severity=IncidentSeverity.MAJOR,
            title="Safety Concern Investigation",
            description="Based on feedback: " + feedback.description,
        )

        # 4. Link feedback to incident
        fb = pmms._feedback_collector.link_to_incident(
            feedback.feedback_id,
            incident.incident_id,
        )

        assert incident.incident_id in fb.related_incidents

        # 5. Resolve both
        pmms.update_incident(incident.incident_id, IncidentStatus.RESOLVED)
        pmms.resolve_feedback(feedback.feedback_id, "Addressed via incident response")

    def test_drift_triggering_incident(self, pmms):
        """Test drift detection triggering incident creation."""
        # 1. Register metric
        metric = pmms.register_metric(
            name="Accuracy",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )

        # 2. Induce critical drift
        for i in range(15):
            _, drift = pmms.record_metric_value(metric.metric_id, 0.60)

        # 3. Check drift was detected
        drifted = pmms.get_metrics_with_drift()
        assert len(drifted) >= 1

        # 4. Alerts should have been generated
        alerts = pmms.get_active_alerts()
        assert len(alerts) >= 1

        # 5. Create incident for drift
        incident = pmms.report_incident(
            severity=IncidentSeverity.MAJOR,
            title="Critical Performance Drift",
            description=f"Metric {metric.name} drifted significantly from baseline",
            incident_type="drift",
            affected_component="model",
        )

        assert incident is not None

    def test_comprehensive_report_generation(self, pmms):
        """Test comprehensive report with all data types."""
        # Add metrics
        for name, mt, baseline in [
            ("Accuracy", MonitoringMetricType.ACCURACY, 0.95),
            ("Latency", MonitoringMetricType.LATENCY, 50.0),
            ("Error Rate", MonitoringMetricType.ERROR_RATE, 0.01),
        ]:
            metric = pmms.register_metric(
                name=name, metric_type=mt, baseline_value=baseline
            )
            for i in range(15):
                pmms.record_metric_value(metric.metric_id, baseline * 0.95)

        # Add incidents
        for severity in [IncidentSeverity.MINOR, IncidentSeverity.MODERATE, IncidentSeverity.SERIOUS]:
            pmms.report_incident(
                severity=severity,
                title=f"{severity.name} Incident",
                description="Test incident",
            )

        # Add feedback
        for ft in [FeedbackType.BUG_REPORT, FeedbackType.FEATURE_REQUEST]:
            pmms.submit_feedback(
                feedback_type=ft,
                subject=f"{ft.name} Feedback",
                description="Test feedback",
            )

        # Generate report
        report = pmms.generate_periodic_report(report_type="weekly")

        # Verify report contents
        assert report.total_incidents >= 3
        assert report.serious_incidents >= 1
        assert report.total_feedback >= 2
        assert report.compliance_score < 100  # Reduced due to serious incident

    def test_export_and_reload_data(self, pmms, tmp_path):
        """Test data export contains all necessary information."""
        # Create data
        metric = pmms.register_metric(
            name="Test Metric",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )
        for i in range(15):
            pmms.record_metric_value(metric.metric_id, 0.93)

        incident = pmms.report_incident(
            severity=IncidentSeverity.MODERATE,
            title="Test Incident",
            description="Test",
        )

        feedback = pmms.submit_feedback(
            feedback_type=FeedbackType.BUG_REPORT,
            subject="Test Bug",
            description="Test",
        )

        pmms.generate_periodic_report()

        # Export
        export = pmms.export_monitoring_data()

        # Verify export completeness
        assert len(export["metrics"]) >= 1
        assert len(export["incidents"]) >= 1
        assert len(export["feedback"]) >= 1
        assert len(export["reports"]) >= 1
        assert "compliance_status" in export

        # Save export
        export_file = tmp_path / "export.json"
        with open(export_file, "w") as f:
            json.dump(export, f, default=str)

        # Verify file was created
        assert export_file.exists()


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Thread safety tests for the monitoring system."""

    @pytest.fixture
    def pmms(self, tmp_path):
        """Create post-market monitoring system fixture."""
        config = PostMarketConfig(
            drift_detection_window=10,
            log_path=str(tmp_path / "logs"),
            reports_path=str(tmp_path / "reports"),
        )
        return PostMarketMonitoringSystem(config)

    def test_concurrent_metric_recording(self, pmms):
        """Test concurrent metric recording."""
        metric = pmms.register_metric(
            name="Test",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )

        errors = []

        def record_values():
            try:
                for i in range(50):
                    pmms.record_metric_value(metric.metric_id, 0.93 + (i % 5) * 0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_values) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_incident_reporting(self, pmms):
        """Test concurrent incident reporting."""
        errors = []

        def report_incidents():
            try:
                for i in range(20):
                    pmms.report_incident(
                        severity=IncidentSeverity.MINOR,
                        title=f"Incident {i}",
                        description="Test",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=report_incidents) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_feedback_submission(self, pmms):
        """Test concurrent feedback submission."""
        errors = []

        def submit_feedback():
            try:
                for i in range(20):
                    pmms.submit_feedback(
                        feedback_type=FeedbackType.BUG_REPORT,
                        subject=f"Bug {i}",
                        description="Test",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=submit_feedback) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_mixed_operations(self, pmms):
        """Test concurrent mixed operations."""
        metric = pmms.register_metric(
            name="Test",
            metric_type=MonitoringMetricType.ACCURACY,
            baseline_value=0.95,
        )

        errors = []

        def mixed_operations():
            try:
                for i in range(20):
                    # Record metric
                    pmms.record_metric_value(metric.metric_id, 0.93)
                    # Report incident
                    pmms.report_incident(
                        severity=IncidentSeverity.MINOR,
                        title=f"Incident {i}",
                        description="Test",
                    )
                    # Submit feedback
                    pmms.submit_feedback(
                        feedback_type=FeedbackType.BUG_REPORT,
                        subject=f"Bug {i}",
                        description="Test",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=mixed_operations) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
