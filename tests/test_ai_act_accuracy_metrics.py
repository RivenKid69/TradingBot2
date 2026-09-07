# -*- coding: utf-8 -*-
"""
Tests for EU AI Act Article 15 - Accuracy Metrics Module.

Tests the accuracy_metrics.py module which implements accuracy tracking
and declaration requirements for high-risk AI systems.
"""

import math
import pytest
from datetime import datetime, timedelta
from typing import List

from services.ai_act.accuracy_metrics import (
    MetricType,
    MetricStatus,
    AccuracyMetric,
    MetricMeasurement,
    DeclaredAccuracyMetrics,
    AccuracyMonitor,
    create_accuracy_monitor,
    get_default_trading_metrics,
)


# ============================================================================
# MetricType Enum Tests
# ============================================================================


class TestMetricType:
    """Tests for MetricType enum."""

    def test_metric_type_performance_metrics(self):
        """Test performance metric types exist."""
        assert MetricType.SHARPE_RATIO
        assert MetricType.SORTINO_RATIO
        assert MetricType.CALMAR_RATIO
        assert MetricType.MAX_DRAWDOWN

    def test_metric_type_trade_level_metrics(self):
        """Test trade-level metric types exist."""
        assert MetricType.WIN_RATE
        assert MetricType.PROFIT_FACTOR
        assert MetricType.AVERAGE_WIN_LOSS_RATIO

    def test_metric_type_prediction_metrics(self):
        """Test prediction metric types exist."""
        assert MetricType.DIRECTION_ACCURACY
        assert MetricType.MAGNITUDE_ACCURACY
        assert MetricType.SIGNAL_QUALITY

    def test_metric_type_risk_adjusted_metrics(self):
        """Test risk-adjusted metric types exist."""
        assert MetricType.VAR_95
        assert MetricType.CVAR_95
        assert MetricType.INFORMATION_RATIO

    def test_metric_type_stability_metrics(self):
        """Test stability metric types exist."""
        assert MetricType.CONSISTENCY_SCORE
        assert MetricType.REGIME_ADAPTABILITY


# ============================================================================
# MetricStatus Enum Tests
# ============================================================================


class TestMetricStatus:
    """Tests for MetricStatus enum."""

    def test_metric_status_values(self):
        """Test all metric status values exist."""
        assert MetricStatus.EXCEEDS_EXPECTED.value == "exceeds_expected"
        assert MetricStatus.MEETS_EXPECTED.value == "meets_expected"
        assert MetricStatus.MEETS_MINIMUM.value == "meets_minimum"
        assert MetricStatus.BELOW_MINIMUM.value == "below_minimum"
        assert MetricStatus.CRITICAL.value == "critical"
        assert MetricStatus.NOT_MEASURED.value == "not_measured"


# ============================================================================
# AccuracyMetric Tests
# ============================================================================


class TestAccuracyMetric:
    """Tests for AccuracyMetric dataclass."""

    @pytest.fixture
    def sharpe_metric(self):
        """Create a Sharpe Ratio metric (higher is better)."""
        return AccuracyMetric(
            metric_type=MetricType.SHARPE_RATIO,
            name="Sharpe Ratio",
            description="Risk-adjusted return",
            expected_value=1.5,
            minimum_value=0.5,
            critical_value=0.0,
            measurement_method="Rolling calculation",
            measurement_window="252 days",
            unit="ratio",
            higher_is_better=True,
        )

    @pytest.fixture
    def drawdown_metric(self):
        """Create a Max Drawdown metric (lower is better)."""
        return AccuracyMetric(
            metric_type=MetricType.MAX_DRAWDOWN,
            name="Maximum Drawdown",
            description="Maximum peak-to-trough decline",
            expected_value=0.15,
            minimum_value=0.25,
            critical_value=0.35,
            measurement_method="Historical max",
            measurement_window="Since inception",
            unit="percentage",
            higher_is_better=False,
        )

    # ---- evaluate_value tests (higher_is_better=True) ----

    def test_evaluate_value_exceeds_expected_higher_better(self, sharpe_metric):
        """Test evaluation when value exceeds expected (higher is better)."""
        status = sharpe_metric.evaluate_value(2.0)
        assert status == MetricStatus.EXCEEDS_EXPECTED

    def test_evaluate_value_meets_expected_higher_better(self, sharpe_metric):
        """Test evaluation when value meets expected range (higher is better)."""
        # Between midpoint (1.0) and expected (1.5)
        status = sharpe_metric.evaluate_value(1.2)
        assert status == MetricStatus.MEETS_EXPECTED

    def test_evaluate_value_meets_minimum_higher_better(self, sharpe_metric):
        """Test evaluation when value meets minimum (higher is better)."""
        # Above minimum (0.5) but below midpoint (1.0)
        status = sharpe_metric.evaluate_value(0.7)
        assert status == MetricStatus.MEETS_MINIMUM

    def test_evaluate_value_below_minimum_higher_better(self, sharpe_metric):
        """Test evaluation when value is below minimum (higher is better)."""
        status = sharpe_metric.evaluate_value(0.3)
        assert status == MetricStatus.BELOW_MINIMUM

    def test_evaluate_value_critical_higher_better(self, sharpe_metric):
        """Test evaluation when value is critical (higher is better)."""
        status = sharpe_metric.evaluate_value(-0.5)
        assert status == MetricStatus.CRITICAL

    # ---- evaluate_value tests (higher_is_better=False) ----

    def test_evaluate_value_exceeds_expected_lower_better(self, drawdown_metric):
        """Test evaluation when value exceeds expected (lower is better)."""
        status = drawdown_metric.evaluate_value(0.10)  # Better than 0.15
        assert status == MetricStatus.EXCEEDS_EXPECTED

    def test_evaluate_value_meets_expected_lower_better(self, drawdown_metric):
        """Test evaluation when value meets expected range (lower is better)."""
        # Midpoint is 0.20, value between expected (0.15) and midpoint
        status = drawdown_metric.evaluate_value(0.18)
        assert status == MetricStatus.MEETS_EXPECTED

    def test_evaluate_value_meets_minimum_lower_better(self, drawdown_metric):
        """Test evaluation when value meets minimum (lower is better)."""
        # Between midpoint (0.20) and minimum (0.25)
        status = drawdown_metric.evaluate_value(0.22)
        assert status == MetricStatus.MEETS_MINIMUM

    def test_evaluate_value_below_minimum_lower_better(self, drawdown_metric):
        """Test evaluation when value is below minimum (lower is better)."""
        status = drawdown_metric.evaluate_value(0.30)  # Worse than 0.25
        assert status == MetricStatus.BELOW_MINIMUM

    def test_evaluate_value_critical_lower_better(self, drawdown_metric):
        """Test evaluation when value is critical (lower is better)."""
        status = drawdown_metric.evaluate_value(0.40)  # Worse than 0.35
        assert status == MetricStatus.CRITICAL

    def test_evaluate_value_nan(self, sharpe_metric):
        """Test evaluation with NaN value."""
        status = sharpe_metric.evaluate_value(float("nan"))
        assert status == MetricStatus.NOT_MEASURED

    def test_evaluate_value_infinity(self, sharpe_metric):
        """Test evaluation with infinity value."""
        status = sharpe_metric.evaluate_value(float("inf"))
        assert status == MetricStatus.NOT_MEASURED

    def test_evaluate_value_no_critical_threshold(self):
        """Test evaluation when no critical threshold is set."""
        metric = AccuracyMetric(
            metric_type=MetricType.WIN_RATE,
            name="Win Rate",
            description="Percentage of winning trades",
            expected_value=0.55,
            minimum_value=0.45,
            critical_value=None,  # No critical
            unit="percentage",
            higher_is_better=True,
        )
        # Should still classify as below_minimum, not critical
        status = metric.evaluate_value(0.40)
        assert status == MetricStatus.BELOW_MINIMUM

    # ---- get_compliance_text tests ----

    def test_get_compliance_text_exceeds(self, sharpe_metric):
        """Test compliance text for exceeding expected."""
        text = sharpe_metric.get_compliance_text(2.0)
        assert "EXCELLENT" in text
        assert "Exceeds expected" in text

    def test_get_compliance_text_meets_expected(self, sharpe_metric):
        """Test compliance text for meeting expected."""
        text = sharpe_metric.get_compliance_text(1.2)
        assert "GOOD" in text
        assert "expected range" in text

    def test_get_compliance_text_meets_minimum(self, sharpe_metric):
        """Test compliance text for meeting minimum."""
        text = sharpe_metric.get_compliance_text(0.7)
        assert "ACCEPTABLE" in text
        assert "minimum threshold" in text

    def test_get_compliance_text_below_minimum(self, sharpe_metric):
        """Test compliance text for below minimum."""
        text = sharpe_metric.get_compliance_text(0.3)
        assert "WARNING" in text
        assert "Article 15" in text

    def test_get_compliance_text_critical(self, sharpe_metric):
        """Test compliance text for critical."""
        text = sharpe_metric.get_compliance_text(-0.5)
        assert "CRITICAL" in text
        assert "Immediate action" in text

    def test_get_compliance_text_not_measured(self, sharpe_metric):
        """Test compliance text for NaN."""
        text = sharpe_metric.get_compliance_text(float("nan"))
        assert "Not measured" in text or "invalid" in text


# ============================================================================
# MetricMeasurement Tests
# ============================================================================


class TestMetricMeasurement:
    """Tests for MetricMeasurement dataclass."""

    def test_create_measurement_minimal(self):
        """Test creating measurement with minimal fields."""
        measurement = MetricMeasurement(
            metric_type=MetricType.SHARPE_RATIO,
            value=1.5,
        )
        assert measurement.metric_type == MetricType.SHARPE_RATIO
        assert measurement.value == 1.5
        assert measurement.measurement_id  # Should be auto-generated
        assert measurement.timestamp  # Should be auto-generated

    def test_create_measurement_full(self):
        """Test creating measurement with all fields."""
        now = datetime.utcnow()
        measurement = MetricMeasurement(
            metric_type=MetricType.MAX_DRAWDOWN,
            value=0.15,
            timestamp=now,
            measurement_id="test-id",
            window_start=now - timedelta(days=252),
            window_end=now,
            sample_size=252,
            confidence_interval=(0.12, 0.18),
            metadata={"strategy": "momentum"},
        )
        assert measurement.value == 0.15
        assert measurement.measurement_id == "test-id"
        assert measurement.sample_size == 252
        assert measurement.confidence_interval == (0.12, 0.18)
        assert measurement.metadata["strategy"] == "momentum"


# ============================================================================
# DeclaredAccuracyMetrics Tests
# ============================================================================


class TestDeclaredAccuracyMetrics:
    """Tests for DeclaredAccuracyMetrics dataclass."""

    @pytest.fixture
    def declaration(self):
        """Create a sample declaration."""
        return DeclaredAccuracyMetrics(
            system_name="TestSystem",
            system_version="1.0.0",
            metrics=[
                AccuracyMetric(
                    metric_type=MetricType.SHARPE_RATIO,
                    name="Sharpe Ratio",
                    description="Risk-adjusted return",
                    expected_value=1.5,
                    minimum_value=0.5,
                    critical_value=0.0,
                    measurement_method="Rolling calculation",
                    measurement_window="252 days",
                    unit="ratio",
                    higher_is_better=True,
                ),
            ],
            measurement_conditions="Standard market conditions",
            limitations=["Past performance disclaimer", "Liquidity assumption"],
            regulatory_notes="Compliant with Article 15",
        )

    def test_create_declaration(self, declaration):
        """Test creating declaration."""
        assert declaration.system_name == "TestSystem"
        assert declaration.system_version == "1.0.0"
        assert len(declaration.metrics) == 1
        assert declaration.declaration_id  # Auto-generated

    def test_generate_instructions_of_use_section(self, declaration):
        """Test generating instructions of use section."""
        text = declaration.generate_instructions_of_use_section()

        # Check header
        assert "ACCURACY METRICS DECLARATION" in text
        assert "EU AI Act Article 15" in text

        # Check system info
        assert "TestSystem" in text
        assert "1.0.0" in text

        # Check metrics
        assert "Sharpe Ratio" in text
        assert "Expected Value" in text
        assert "Minimum Acceptable" in text
        assert "Critical Threshold" in text

        # Check conditions
        assert "MEASUREMENT CONDITIONS" in text
        assert "Standard market conditions" in text

        # Check limitations
        assert "KNOWN LIMITATIONS" in text
        assert "Past performance" in text

        # Check regulatory notes
        assert "REGULATORY COMPLIANCE" in text
        assert "Compliant with Article 15" in text

    def test_generate_instructions_no_limitations(self):
        """Test generating instructions without limitations."""
        declaration = DeclaredAccuracyMetrics(
            system_name="TestSystem",
            metrics=[],
            limitations=[],
        )
        text = declaration.generate_instructions_of_use_section()
        assert "No specific limitations declared" in text

    def test_generate_instructions_no_critical(self):
        """Test generating instructions without critical threshold."""
        declaration = DeclaredAccuracyMetrics(
            system_name="TestSystem",
            metrics=[
                AccuracyMetric(
                    metric_type=MetricType.WIN_RATE,
                    name="Win Rate",
                    description="Win rate",
                    expected_value=0.55,
                    minimum_value=0.45,
                    critical_value=None,  # No critical
                    unit="percentage",
                ),
            ],
        )
        text = declaration.generate_instructions_of_use_section()
        assert "Win Rate" in text
        # Should not have Critical Threshold line
        assert text.count("Critical Threshold") == 0


# ============================================================================
# AccuracyMonitor Tests
# ============================================================================


class TestAccuracyMonitor:
    """Tests for AccuracyMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create monitor with default metrics."""
        return create_accuracy_monitor(history_limit=100)

    @pytest.fixture
    def alert_capture(self):
        """Capture alerts for testing."""
        alerts = []

        def capture(metric, value, status):
            alerts.append((metric.name, value, status))

        return alerts, capture

    def test_create_monitor_default(self):
        """Test creating monitor with defaults."""
        monitor = create_accuracy_monitor()
        assert monitor.declared_metrics.system_name == "TradingBot2"
        assert len(monitor.declared_metrics.metrics) > 0

    def test_create_monitor_with_callback(self, alert_capture):
        """Test creating monitor with alert callback."""
        alerts, callback = alert_capture
        monitor = create_accuracy_monitor(alert_callback=callback)

        # Record a critical value to trigger alert
        monitor.record_measurement(MetricType.SHARPE_RATIO, -0.5)

        assert len(alerts) > 0
        assert alerts[0][2] == MetricStatus.CRITICAL

    def test_create_monitor_custom_metrics(self):
        """Test creating monitor with custom metrics."""
        custom = [
            AccuracyMetric(
                metric_type=MetricType.WIN_RATE,
                name="Custom Win Rate",
                description="Custom metric",
                expected_value=0.60,
                minimum_value=0.50,
                unit="percentage",
            ),
        ]
        monitor = create_accuracy_monitor(custom_metrics=custom)
        assert len(monitor.declared_metrics.metrics) == 1
        assert monitor.declared_metrics.metrics[0].name == "Custom Win Rate"

    # ---- record_measurement tests ----

    def test_record_measurement_basic(self, monitor):
        """Test basic measurement recording."""
        measurement = monitor.record_measurement(
            MetricType.SHARPE_RATIO,
            1.8,
        )
        assert measurement.metric_type == MetricType.SHARPE_RATIO
        assert measurement.value == 1.8

    def test_record_measurement_with_window(self, monitor):
        """Test measurement recording with time window."""
        now = datetime.utcnow()
        measurement = monitor.record_measurement(
            MetricType.SHARPE_RATIO,
            1.5,
            window_start=now - timedelta(days=252),
            window_end=now,
            sample_size=252,
        )
        assert measurement.window_start is not None
        assert measurement.sample_size == 252

    def test_record_measurement_with_confidence_interval(self, monitor):
        """Test measurement recording with confidence interval."""
        measurement = monitor.record_measurement(
            MetricType.SHARPE_RATIO,
            1.5,
            confidence_interval=(1.2, 1.8),
        )
        assert measurement.confidence_interval == (1.2, 1.8)

    def test_record_measurement_with_metadata(self, monitor):
        """Test measurement recording with metadata."""
        measurement = monitor.record_measurement(
            MetricType.SHARPE_RATIO,
            1.5,
            metadata={"strategy": "momentum", "universe": "crypto"},
        )
        assert measurement.metadata["strategy"] == "momentum"

    def test_record_measurement_triggers_alert_on_critical(self, alert_capture):
        """Test alert triggered on critical value."""
        alerts, callback = alert_capture
        monitor = create_accuracy_monitor(alert_callback=callback)

        monitor.record_measurement(MetricType.SHARPE_RATIO, -0.5)

        assert len(alerts) >= 1
        assert any(status == MetricStatus.CRITICAL for _, _, status in alerts)

    def test_record_measurement_triggers_alert_on_status_change(self, alert_capture):
        """Test alert triggered on status change."""
        alerts, callback = alert_capture
        monitor = create_accuracy_monitor(alert_callback=callback)

        # First measurement - EXCEEDS_EXPECTED
        monitor.record_measurement(MetricType.SHARPE_RATIO, 2.0)
        initial_alerts = len(alerts)

        # Second measurement - BELOW_MINIMUM (status change)
        monitor.record_measurement(MetricType.SHARPE_RATIO, 0.3)

        assert len(alerts) > initial_alerts

    def test_record_measurement_unknown_metric(self, monitor):
        """Test recording for unknown metric type."""
        # Should not raise, just not update status
        measurement = monitor.record_measurement(
            MetricType.REGIME_ADAPTABILITY,  # Not in default metrics
            0.5,
        )
        assert measurement.value == 0.5

    def test_record_measurement_history_limit(self):
        """Test history limit is enforced."""
        monitor = create_accuracy_monitor(history_limit=5)

        # Record 10 measurements
        for i in range(10):
            monitor.record_measurement(MetricType.SHARPE_RATIO, 1.0 + i * 0.1)

        history = monitor.get_measurement_history(MetricType.SHARPE_RATIO)
        assert len(history) == 5
        # Should keep the most recent
        assert history[-1].value == pytest.approx(1.9, rel=0.01)

    # ---- get_current_status tests ----

    def test_get_current_status_initial(self, monitor):
        """Test initial status is NOT_MEASURED."""
        status = monitor.get_current_status(MetricType.SHARPE_RATIO)
        assert status == MetricStatus.NOT_MEASURED

    def test_get_current_status_after_measurement(self, monitor):
        """Test status after measurement."""
        monitor.record_measurement(MetricType.SHARPE_RATIO, 2.0)
        status = monitor.get_current_status(MetricType.SHARPE_RATIO)
        assert status == MetricStatus.EXCEEDS_EXPECTED

    def test_get_current_status_unknown_metric(self, monitor):
        """Test status for unknown metric."""
        status = monitor.get_current_status(MetricType.REGIME_ADAPTABILITY)
        assert status == MetricStatus.NOT_MEASURED

    # ---- get_latest_measurement tests ----

    def test_get_latest_measurement_none(self, monitor):
        """Test getting latest when none exist."""
        latest = monitor.get_latest_measurement(MetricType.SHARPE_RATIO)
        assert latest is None

    def test_get_latest_measurement(self, monitor):
        """Test getting latest measurement."""
        monitor.record_measurement(MetricType.SHARPE_RATIO, 1.0)
        monitor.record_measurement(MetricType.SHARPE_RATIO, 1.5)
        monitor.record_measurement(MetricType.SHARPE_RATIO, 2.0)

        latest = monitor.get_latest_measurement(MetricType.SHARPE_RATIO)
        assert latest is not None
        assert latest.value == 2.0

    # ---- get_measurement_history tests ----

    def test_get_measurement_history_empty(self, monitor):
        """Test getting history when empty."""
        history = monitor.get_measurement_history(MetricType.SHARPE_RATIO)
        assert history == []

    def test_get_measurement_history_with_limit(self, monitor):
        """Test getting history with limit."""
        for i in range(10):
            monitor.record_measurement(MetricType.SHARPE_RATIO, 1.0 + i * 0.1)

        history = monitor.get_measurement_history(MetricType.SHARPE_RATIO, limit=5)
        assert len(history) == 5
        # Should be the last 5
        assert history[0].value == pytest.approx(1.5, rel=0.01)

    def test_get_measurement_history_returns_copy(self, monitor):
        """Test that history returns a copy."""
        monitor.record_measurement(MetricType.SHARPE_RATIO, 1.5)

        history1 = monitor.get_measurement_history(MetricType.SHARPE_RATIO)
        history2 = monitor.get_measurement_history(MetricType.SHARPE_RATIO)

        assert history1 is not history2

    # ---- compute_metric_statistics tests ----

    def test_compute_statistics_empty(self, monitor):
        """Test computing statistics when no data."""
        stats = monitor.compute_metric_statistics(MetricType.SHARPE_RATIO)
        assert stats["count"] == 0
        assert math.isnan(stats["mean"])

    def test_compute_statistics_single_value(self, monitor):
        """Test computing statistics with single value."""
        monitor.record_measurement(MetricType.SHARPE_RATIO, 1.5)

        stats = monitor.compute_metric_statistics(MetricType.SHARPE_RATIO)
        assert stats["count"] == 1
        assert stats["mean"] == 1.5
        assert stats["std"] == 0.0
        assert stats["min"] == 1.5
        assert stats["max"] == 1.5

    def test_compute_statistics_multiple_values(self, monitor):
        """Test computing statistics with multiple values."""
        values = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        for v in values:
            monitor.record_measurement(MetricType.SHARPE_RATIO, v)

        stats = monitor.compute_metric_statistics(MetricType.SHARPE_RATIO)
        assert stats["count"] == 6
        assert stats["mean"] == pytest.approx(1.5, rel=0.01)
        assert stats["min"] == 1.0
        assert stats["max"] == 2.0
        assert stats["std"] > 0

    def test_compute_statistics_with_nan(self, monitor):
        """Test computing statistics with NaN values."""
        monitor.record_measurement(MetricType.SHARPE_RATIO, 1.5)
        monitor.record_measurement(MetricType.SHARPE_RATIO, float("nan"))
        monitor.record_measurement(MetricType.SHARPE_RATIO, 2.0)

        stats = monitor.compute_metric_statistics(MetricType.SHARPE_RATIO)
        # NaN should be filtered out
        assert stats["count"] == 2
        assert stats["mean"] == pytest.approx(1.75, rel=0.01)

    def test_compute_statistics_trend(self, monitor):
        """Test computing statistics trend."""
        # Increasing values should give positive trend
        for i in range(10):
            monitor.record_measurement(MetricType.SHARPE_RATIO, 1.0 + i * 0.1)

        stats = monitor.compute_metric_statistics(MetricType.SHARPE_RATIO)
        assert stats["trend"] > 0

    # ---- generate_compliance_report tests ----

    def test_generate_compliance_report_empty(self, monitor):
        """Test generating report with no measurements."""
        report = monitor.generate_compliance_report()

        assert "ARTICLE 15 ACCURACY COMPLIANCE REPORT" in report
        assert "Total Measurements: 0" in report

    def test_generate_compliance_report_with_data(self, monitor):
        """Test generating report with measurements."""
        monitor.record_measurement(MetricType.SHARPE_RATIO, 2.0)
        monitor.record_measurement(MetricType.MAX_DRAWDOWN, 0.10)

        report = monitor.generate_compliance_report()

        assert "METRIC STATUS SUMMARY" in report
        assert "Sharpe Ratio" in report
        assert "exceeds_expected" in report
        assert "OVERALL COMPLIANCE ASSESSMENT" in report

    def test_generate_compliance_report_critical(self, monitor):
        """Test generating report with critical metric."""
        monitor.record_measurement(MetricType.SHARPE_RATIO, -0.5)

        report = monitor.generate_compliance_report()

        assert "CRITICAL" in report
        assert "Immediate review required" in report

    def test_generate_compliance_report_below_minimum(self, monitor):
        """Test generating report with below minimum metric."""
        monitor.record_measurement(MetricType.SHARPE_RATIO, 0.3)

        report = monitor.generate_compliance_report()

        assert "WARNING" in report

    def test_generate_compliance_report_all_good(self, monitor):
        """Test generating report when all metrics are good."""
        monitor.record_measurement(MetricType.SHARPE_RATIO, 2.0)
        monitor.record_measurement(MetricType.MAX_DRAWDOWN, 0.10)
        monitor.record_measurement(MetricType.WIN_RATE, 0.60)
        monitor.record_measurement(MetricType.DIRECTION_ACCURACY, 0.55)

        report = monitor.generate_compliance_report()

        # Should indicate compliance
        assert "All metrics within acceptable" in report or "compliant" in report.lower()

    # ---- export_for_audit tests ----

    def test_export_for_audit_empty(self, monitor):
        """Test audit export with no data."""
        export = monitor.export_for_audit()

        assert "export_timestamp" in export
        assert "declaration" in export
        assert "declared_metrics" in export
        assert "current_status" in export
        assert "measurement_history" in export
        assert "statistics" in export

    def test_export_for_audit_with_data(self, monitor):
        """Test audit export with measurements."""
        monitor.record_measurement(MetricType.SHARPE_RATIO, 1.5)

        export = monitor.export_for_audit()

        # Check declaration
        assert export["declaration"]["system_name"] == "TradingBot2"

        # Check declared metrics
        assert len(export["declared_metrics"]) > 0

        # Check current status
        assert "SHARPE_RATIO" in export["current_status"]

        # Check measurement history
        assert "SHARPE_RATIO" in export["measurement_history"]
        assert len(export["measurement_history"]["SHARPE_RATIO"]) == 1

    def test_export_for_audit_statistics(self, monitor):
        """Test audit export statistics."""
        monitor.record_measurement(MetricType.SHARPE_RATIO, 1.5)

        export = monitor.export_for_audit()

        assert export["statistics"]["total_measurements"] == 1

    def test_export_for_audit_measurement_format(self, monitor):
        """Test audit export measurement format."""
        now = datetime.utcnow()
        monitor.record_measurement(
            MetricType.SHARPE_RATIO,
            1.5,
            window_start=now - timedelta(days=252),
            window_end=now,
            sample_size=252,
            confidence_interval=(1.2, 1.8),
            metadata={"test": True},
        )

        export = monitor.export_for_audit()
        measurement = export["measurement_history"]["SHARPE_RATIO"][0]

        assert "measurement_id" in measurement
        assert measurement["value"] == 1.5
        assert measurement["timestamp"] is not None
        assert measurement["window_start"] is not None
        assert measurement["window_end"] is not None
        assert measurement["sample_size"] == 252
        assert measurement["confidence_interval"] == (1.2, 1.8)
        assert measurement["metadata"]["test"] is True


# ============================================================================
# get_default_trading_metrics Tests
# ============================================================================


class TestGetDefaultTradingMetrics:
    """Tests for get_default_trading_metrics function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        metrics = get_default_trading_metrics()
        assert isinstance(metrics, list)

    def test_contains_required_metrics(self):
        """Test that default metrics include required ones."""
        metrics = get_default_trading_metrics()
        metric_types = [m.metric_type for m in metrics]

        # Per EU_AI_ACT_INTEGRATION_PLAN.md
        assert MetricType.SHARPE_RATIO in metric_types
        assert MetricType.MAX_DRAWDOWN in metric_types
        assert MetricType.WIN_RATE in metric_types
        assert MetricType.DIRECTION_ACCURACY in metric_types

    def test_sharpe_ratio_thresholds(self):
        """Test Sharpe Ratio has correct thresholds."""
        metrics = get_default_trading_metrics()
        sharpe = next(m for m in metrics if m.metric_type == MetricType.SHARPE_RATIO)

        # Per integration plan: expected > 1.5, minimum > 0.5
        assert sharpe.expected_value == 1.5
        assert sharpe.minimum_value == 0.5
        assert sharpe.higher_is_better is True

    def test_max_drawdown_thresholds(self):
        """Test Max Drawdown has correct thresholds."""
        metrics = get_default_trading_metrics()
        dd = next(m for m in metrics if m.metric_type == MetricType.MAX_DRAWDOWN)

        # Per integration plan: expected < 15%, minimum < 25%
        assert dd.expected_value == 0.15
        assert dd.minimum_value == 0.25
        assert dd.higher_is_better is False  # Lower is better

    def test_win_rate_thresholds(self):
        """Test Win Rate has correct thresholds."""
        metrics = get_default_trading_metrics()
        wr = next(m for m in metrics if m.metric_type == MetricType.WIN_RATE)

        # Per integration plan: expected > 55%, minimum > 45%
        assert wr.expected_value == 0.55
        assert wr.minimum_value == 0.45

    def test_direction_accuracy_thresholds(self):
        """Test Direction Accuracy has correct thresholds."""
        metrics = get_default_trading_metrics()
        da = next(m for m in metrics if m.metric_type == MetricType.DIRECTION_ACCURACY)

        # Per integration plan: expected > 52%, minimum > 50%
        assert da.expected_value == 0.52
        assert da.minimum_value == 0.50

    def test_all_metrics_have_required_fields(self):
        """Test all metrics have required fields populated."""
        metrics = get_default_trading_metrics()

        for metric in metrics:
            assert metric.name, f"{metric.metric_type} missing name"
            assert metric.description, f"{metric.metric_type} missing description"
            assert metric.measurement_method, f"{metric.metric_type} missing method"
            assert metric.unit, f"{metric.metric_type} missing unit"


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Test thread safety of AccuracyMonitor."""

    def test_concurrent_measurements(self):
        """Test concurrent measurement recording."""
        import threading

        monitor = create_accuracy_monitor(history_limit=1000)
        errors = []

        def record_measurements(thread_id):
            try:
                for i in range(100):
                    monitor.record_measurement(
                        MetricType.SHARPE_RATIO,
                        1.0 + thread_id * 0.01 + i * 0.001,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_measurements, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Should have 500 measurements (5 threads * 100 each)
        history = monitor.get_measurement_history(MetricType.SHARPE_RATIO)
        assert len(history) == 500

    def test_concurrent_read_write(self):
        """Test concurrent reading and writing."""
        import threading

        monitor = create_accuracy_monitor()
        errors = []

        def writer():
            try:
                for i in range(100):
                    monitor.record_measurement(MetricType.SHARPE_RATIO, 1.5)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    monitor.get_current_status(MetricType.SHARPE_RATIO)
                    monitor.get_latest_measurement(MetricType.SHARPE_RATIO)
                    monitor.compute_metric_statistics(MetricType.SHARPE_RATIO)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_metric_at_exact_threshold(self):
        """Test value exactly at threshold."""
        metric = AccuracyMetric(
            metric_type=MetricType.SHARPE_RATIO,
            name="Test",
            description="Test",
            expected_value=1.5,
            minimum_value=0.5,
            unit="ratio",
            higher_is_better=True,
        )

        # Exactly at expected
        status = metric.evaluate_value(1.5)
        assert status == MetricStatus.EXCEEDS_EXPECTED

        # Exactly at minimum
        status = metric.evaluate_value(0.5)
        assert status == MetricStatus.MEETS_MINIMUM

    def test_very_large_value(self):
        """Test handling of very large values."""
        metric = AccuracyMetric(
            metric_type=MetricType.SHARPE_RATIO,
            name="Test",
            description="Test",
            expected_value=1.5,
            minimum_value=0.5,
            unit="ratio",
            higher_is_better=True,
        )

        status = metric.evaluate_value(1e10)
        assert status == MetricStatus.EXCEEDS_EXPECTED

    def test_negative_infinity(self):
        """Test handling of negative infinity."""
        metric = AccuracyMetric(
            metric_type=MetricType.SHARPE_RATIO,
            name="Test",
            description="Test",
            expected_value=1.5,
            minimum_value=0.5,
            unit="ratio",
            higher_is_better=True,
        )

        status = metric.evaluate_value(float("-inf"))
        assert status == MetricStatus.NOT_MEASURED

    def test_empty_declaration_metrics(self):
        """Test declaration with no metrics."""
        declaration = DeclaredAccuracyMetrics(
            system_name="Empty",
            metrics=[],
        )
        text = declaration.generate_instructions_of_use_section()
        assert "Empty" in text
