# -*- coding: utf-8 -*-
"""
Tests for MiFID II RTS 6 Article 17 Real-Time Monitoring.

Comprehensive test coverage for:
    - Alert generation (within 5-second RTS 6 requirement)
    - OTR monitoring and thresholds
    - P&L threshold monitoring
    - Position limit monitoring
    - Latency monitoring
    - Clock drift monitoring
    - System health monitoring
    - Alert management and escalation

References:
    - RTS 6 Article 17: Real-time monitoring requirements
"""

import pytest
import asyncio
import time
import threading
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from services.compliance.realtime_monitor import (
    AlertSeverity,
    AlertCategory,
    ComplianceAlert,
    MonitoringThreshold,
    RealTimeMonitorConfig,
    MonitoringMetrics,
    RealTimeMonitor,
    create_realtime_monitor,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def config():
    """Default configuration for testing."""
    return RealTimeMonitorConfig(
        max_alert_delay_seconds=5.0,
        check_interval_seconds=0.1,  # Fast for testing
        otr_warning_threshold=10.0,
        otr_critical_threshold=20.0,
        otr_emergency_threshold=50.0,
        pnl_warning_pct=50.0,
        pnl_critical_pct=75.0,
        latency_warning_ms=50.0,
        latency_critical_ms=100.0,
        persist_alerts=False,
    )


@pytest.fixture
def monitor(config):
    """Create monitor instance."""
    return RealTimeMonitor(config=config)


@pytest.fixture
def mock_alert_callback():
    """Mock alert callback."""
    return Mock()


@pytest.fixture
def mock_kill_switch_callback():
    """Mock kill switch callback."""
    return Mock()


# =============================================================================
# Test ComplianceAlert
# =============================================================================

class TestComplianceAlert:
    """Tests for ComplianceAlert data class."""

    def test_alert_creation(self):
        """Test creating an alert."""
        alert = ComplianceAlert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.ORDER_TO_TRADE_RATIO,
            message="OTR elevated",
            details={"otr": 25.5},
        )

        assert alert.severity == AlertSeverity.WARNING
        assert alert.category == AlertCategory.ORDER_TO_TRADE_RATIO
        assert alert.message == "OTR elevated"
        assert alert.details["otr"] == 25.5
        assert alert.alert_id is not None
        assert alert.timestamp_ns > 0

    def test_alert_to_dict(self):
        """Test alert serialization."""
        alert = ComplianceAlert(
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.PNL_THRESHOLD,
            message="P&L limit breach",
            algorithm_id="ALGO001",
            venue="XLON",
        )

        data = alert.to_dict()

        assert data["severity"] == "critical"
        assert data["category"] == "pnl_threshold"
        assert data["algorithm_id"] == "ALGO001"
        assert data["venue"] == "XLON"


# =============================================================================
# Test MonitoringMetrics
# =============================================================================

class TestMonitoringMetrics:
    """Tests for MonitoringMetrics data class."""

    def test_metrics_creation(self):
        """Test creating metrics snapshot."""
        metrics = MonitoringMetrics(
            order_to_trade_ratio=15.0,
            daily_pnl=Decimal("-5000"),
            daily_pnl_limit=Decimal("10000"),
            position_utilization_pct=75.0,
        )

        assert metrics.order_to_trade_ratio == 15.0
        assert metrics.daily_pnl == Decimal("-5000")
        assert metrics.position_utilization_pct == 75.0

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = MonitoringMetrics(
            order_to_trade_ratio=10.0,
            messages_per_second=50.0,
            avg_latency_ms=25.5,
        )

        data = metrics.to_dict()

        assert data["order_to_trade_ratio"] == 10.0
        assert data["messages_per_second"] == 50.0
        assert data["avg_latency_ms"] == 25.5


# =============================================================================
# Test RealTimeMonitor Initialization
# =============================================================================

class TestMonitorInitialization:
    """Tests for monitor initialization."""

    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor._running is False
        assert monitor._metrics is not None

    def test_configuration_applied(self, config):
        """Test configuration is applied."""
        monitor = RealTimeMonitor(config=config)

        assert monitor._config.max_alert_delay_seconds == 5.0
        assert monitor._config.otr_warning_threshold == 10.0


# =============================================================================
# Test Alert Generation
# =============================================================================

class TestAlertGeneration:
    """Tests for alert generation."""

    def test_generate_alert(self, monitor):
        """Test generating an alert."""
        alert = monitor.generate_alert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RISK_LIMIT,
            message="Risk limit approaching",
            details={"current": 80, "limit": 100},
        )

        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING
        assert alert.category == AlertCategory.RISK_LIMIT

    def test_alert_with_context(self, monitor):
        """Test alert with context information."""
        alert = monitor.generate_alert(
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.POSITION_LIMIT,
            message="Position limit exceeded",
            algorithm_id="ALGO001",
            venue="XLON",
            instrument="ISIN123",
        )

        assert alert.algorithm_id == "ALGO001"
        assert alert.venue == "XLON"
        assert alert.instrument == "ISIN123"

    def test_alert_callback_invoked(self, config, mock_alert_callback):
        """Test alert callback is invoked."""
        monitor = RealTimeMonitor(
            config=config,
            alert_callback=mock_alert_callback,
        )

        monitor.generate_alert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.LATENCY,
            message="High latency",
        )

        mock_alert_callback.assert_called_once()


# =============================================================================
# Test Alert Rate Limiting
# =============================================================================

class TestAlertRateLimiting:
    """Tests for alert rate limiting."""

    def test_rate_limiting(self, config):
        """Test alerts are rate limited."""
        config.max_alerts_per_category_per_minute = 3
        monitor = RealTimeMonitor(config=config)

        alerts = []
        for i in range(5):
            alert = monitor.generate_alert(
                severity=AlertSeverity.WARNING,
                category=AlertCategory.LATENCY,
                message=f"Alert {i}",
            )
            if alert:
                alerts.append(alert)

        # Should have limited to 3
        assert len(alerts) <= 3


# =============================================================================
# Test Alert Deduplication
# =============================================================================

class TestAlertDeduplication:
    """Tests for alert deduplication."""

    def test_deduplication(self, config):
        """Test duplicate alerts are suppressed."""
        config.alert_deduplication_window_seconds = 10.0
        monitor = RealTimeMonitor(config=config)

        # Same message
        alert1 = monitor.generate_alert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.LATENCY,
            message="Same message",
        )
        alert2 = monitor.generate_alert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.LATENCY,
            message="Same message",
        )

        assert alert1 is not None
        assert alert2 is None  # Deduplicated


# =============================================================================
# Test OTR Monitoring
# =============================================================================

class TestOTRMonitoring:
    """Tests for order-to-trade ratio monitoring."""

    def test_update_order_metrics(self, monitor):
        """Test updating order metrics."""
        monitor.update_order_metrics(submitted=100, cancelled=10, filled=80)

        metrics = monitor.get_current_metrics()
        assert metrics.orders_submitted == 100
        assert metrics.orders_cancelled == 10
        assert metrics.orders_filled == 80

    def test_otr_calculation(self, monitor):
        """Test OTR calculation."""
        monitor.update_order_metrics(submitted=100, filled=10)

        metrics = monitor.get_current_metrics()
        assert metrics.order_to_trade_ratio == 10.0


# =============================================================================
# Test P&L Monitoring
# =============================================================================

class TestPnLMonitoring:
    """Tests for P&L monitoring."""

    def test_update_pnl(self, monitor):
        """Test updating P&L metrics."""
        monitor.update_pnl(
            realized_pnl=Decimal("1000"),
            unrealized_pnl=Decimal("-500"),
            daily_pnl=Decimal("500"),
        )

        metrics = monitor.get_current_metrics()
        assert metrics.realized_pnl == Decimal("1000")
        assert metrics.unrealized_pnl == Decimal("-500")
        assert metrics.daily_pnl == Decimal("500")


# =============================================================================
# Test Position Monitoring
# =============================================================================

class TestPositionMonitoring:
    """Tests for position monitoring."""

    def test_update_position(self, monitor):
        """Test updating position metrics."""
        monitor.update_position(
            total_value=Decimal("80000"),
            max_value=Decimal("100000"),
        )

        metrics = monitor.get_current_metrics()
        assert metrics.total_position_value == Decimal("80000")
        assert metrics.max_position_value == Decimal("100000")


# =============================================================================
# Test Latency Monitoring
# =============================================================================

class TestLatencyMonitoring:
    """Tests for latency monitoring."""

    def test_record_latency(self, monitor):
        """Test recording latency."""
        monitor.record_latency(10.0)
        monitor.record_latency(20.0)
        monitor.record_latency(30.0)

        metrics = monitor.get_current_metrics()
        assert metrics.avg_latency_ms == 20.0  # Average of 10, 20, 30

    def test_max_latency_tracked(self, monitor):
        """Test max latency is tracked."""
        monitor.record_latency(10.0)
        monitor.record_latency(50.0)
        monitor.record_latency(30.0)

        metrics = monitor.get_current_metrics()
        assert metrics.max_latency_ms == 50.0


# =============================================================================
# Test Clock Drift Monitoring
# =============================================================================

class TestClockDriftMonitoring:
    """Tests for clock drift monitoring per RTS 25."""

    def test_update_clock_status(self, monitor):
        """Test updating clock status."""
        monitor.update_clock_status(offset_ms=25.5, synced=True)

        metrics = monitor.get_current_metrics()
        assert metrics.clock_offset_ms == 25.5
        assert metrics.clock_synced is True


# =============================================================================
# Test System Health Monitoring
# =============================================================================

class TestSystemHealthMonitoring:
    """Tests for system health monitoring."""

    def test_update_system_metrics(self, monitor):
        """Test updating system metrics."""
        monitor.update_system_metrics(
            cpu_pct=50.0,
            memory_pct=60.0,
            disk_pct=70.0,
        )

        metrics = monitor.get_current_metrics()
        assert metrics.cpu_usage_pct == 50.0
        assert metrics.memory_usage_pct == 60.0
        assert metrics.disk_usage_pct == 70.0


# =============================================================================
# Test Alert Management
# =============================================================================

class TestAlertManagement:
    """Tests for alert management."""

    def test_acknowledge_alert(self, monitor):
        """Test acknowledging an alert."""
        alert = monitor.generate_alert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.LATENCY,
            message="Test alert",
        )

        success = monitor.acknowledge_alert(alert.alert_id, "test_user")

        assert success is True
        alerts = monitor.get_alerts()
        acked = [a for a in alerts if a.alert_id == alert.alert_id]
        assert acked[0].acknowledged is True
        assert acked[0].acknowledged_by == "test_user"

    def test_resolve_alert(self, monitor):
        """Test resolving an alert."""
        alert = monitor.generate_alert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.LATENCY,
            message="Test alert",
        )

        success = monitor.resolve_alert(alert.alert_id)

        assert success is True
        alerts = monitor.get_alerts()
        resolved = [a for a in alerts if a.alert_id == alert.alert_id]
        assert resolved[0].resolved is True

    def test_get_unacknowledged_alerts(self, monitor):
        """Test getting unacknowledged alerts."""
        monitor.generate_alert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.LATENCY,
            message="Alert 1",
        )
        alert2 = monitor.generate_alert(
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.PNL_THRESHOLD,
            message="Alert 2",
        )
        monitor.acknowledge_alert(alert2.alert_id, "user")

        unacked = monitor.get_alerts(unacknowledged_only=True)

        assert len(unacked) == 1
        assert unacked[0].message == "Alert 1"

    def test_filter_alerts_by_severity(self, monitor):
        """Test filtering alerts by severity."""
        monitor.generate_alert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.LATENCY,
            message="Warning",
        )
        monitor.generate_alert(
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.PNL_THRESHOLD,
            message="Critical",
        )

        critical_alerts = monitor.get_alerts(severity=AlertSeverity.CRITICAL)

        assert len(critical_alerts) == 1
        assert critical_alerts[0].severity == AlertSeverity.CRITICAL

    def test_filter_alerts_by_category(self, monitor):
        """Test filtering alerts by category."""
        monitor.generate_alert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.LATENCY,
            message="Latency",
        )
        monitor.generate_alert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.PNL_THRESHOLD,
            message="P&L",
        )

        latency_alerts = monitor.get_alerts(category=AlertCategory.LATENCY)

        assert len(latency_alerts) == 1
        assert latency_alerts[0].category == AlertCategory.LATENCY


# =============================================================================
# Test Escalation and Kill Switch
# =============================================================================

class TestEscalation:
    """Tests for alert escalation."""

    def test_escalation_on_critical(self, config):
        """Test escalation callback on critical alert."""
        escalation_mock = Mock()
        monitor = RealTimeMonitor(
            config=config,
            escalation_callback=escalation_mock,
        )

        monitor.generate_alert(
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK_LIMIT,
            message="Critical alert",
        )

        escalation_mock.assert_called_once()

    def test_kill_switch_on_emergency(self, config, mock_kill_switch_callback):
        """Test kill switch callback on emergency alert."""
        monitor = RealTimeMonitor(
            config=config,
            kill_switch_callback=mock_kill_switch_callback,
        )

        monitor.generate_alert(
            severity=AlertSeverity.EMERGENCY,
            category=AlertCategory.ORDER_TO_TRADE_RATIO,
            message="Emergency OTR breach",
        )

        mock_kill_switch_callback.assert_called_once()


# =============================================================================
# Test Statistics and Reporting
# =============================================================================

class TestStatisticsAndReporting:
    """Tests for statistics and compliance reporting."""

    def test_statistics(self, monitor):
        """Test statistics tracking."""
        monitor.generate_alert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.LATENCY,
            message="Warning 1",
        )
        monitor.generate_alert(
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.PNL_THRESHOLD,
            message="Critical 1",
        )

        stats = monitor.get_statistics()

        assert stats["alerts_generated"] == 2
        assert stats["alerts_by_severity"]["warning"] == 1
        assert stats["alerts_by_severity"]["critical"] == 1

    def test_compliance_report(self, monitor):
        """Test compliance report generation."""
        report = monitor.generate_compliance_report()

        assert report["regulation"] == "RTS 6 Article 17"
        assert report["requirement"] == "Real-Time Monitoring"
        assert "alert_deadline_requirement" in report
        assert report["alert_deadline_requirement"] == "5 seconds"
        assert "configuration" in report
        assert "current_metrics" in report


# =============================================================================
# Test Async Operations
# =============================================================================

class TestAsyncOperations:
    """Tests for async operations."""

    @pytest.mark.asyncio
    async def test_start_stop(self, config):
        """Test starting and stopping monitor."""
        monitor = RealTimeMonitor(config=config)

        await monitor.start()
        assert monitor._running is True

        await asyncio.sleep(0.2)

        await monitor.stop()
        assert monitor._running is False

    @pytest.mark.asyncio
    async def test_async_alert_callback(self, config):
        """Test async alert callback."""
        async_mock = AsyncMock()
        monitor = RealTimeMonitor(
            config=config,
            async_alert_callback=async_mock,
        )

        await monitor.start()

        # Generate alert that triggers async callback
        await monitor._generate_alert_async(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.LATENCY,
            message="Test async",
        )

        await monitor.stop()

        async_mock.assert_called()


# =============================================================================
# Test Sync Operations
# =============================================================================

class TestSyncOperations:
    """Tests for sync (background thread) operations."""

    def test_start_stop_sync(self, config):
        """Test sync start/stop."""
        config.check_interval_seconds = 0.1
        monitor = RealTimeMonitor(config=config)

        monitor.start_sync()
        assert monitor._running is True

        time.sleep(0.3)

        monitor.stop_sync()
        assert monitor._running is False


# =============================================================================
# Test Factory Function
# =============================================================================

class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_with_dict_config(self):
        """Test creating with dictionary config."""
        config = {
            "check_interval_seconds": 5.0,
            "otr_warning_threshold": 30.0,
            "persist_alerts": False,
        }

        monitor = create_realtime_monitor(config=config)

        assert monitor is not None

    def test_create_with_callbacks(self, mock_alert_callback, mock_kill_switch_callback):
        """Test creating with callbacks."""
        monitor = create_realtime_monitor(
            alert_callback=mock_alert_callback,
            kill_switch_callback=mock_kill_switch_callback,
        )

        assert monitor is not None
        assert monitor._alert_callback is mock_alert_callback
        assert monitor._kill_switch_callback is mock_kill_switch_callback


# =============================================================================
# Test RTS 6 Compliance
# =============================================================================

class TestRTS6Compliance:
    """Tests for RTS 6 Article 17 compliance."""

    def test_alert_deadline_default(self, monitor):
        """Test default alert deadline is 5 seconds."""
        assert monitor._config.max_alert_delay_seconds == 5.0

    def test_alert_generated_within_deadline(self, monitor):
        """Test alerts are generated quickly."""
        start = time.time()
        monitor.generate_alert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.LATENCY,
            message="Test",
        )
        elapsed = time.time() - start

        # Alert generation should be very fast
        assert elapsed < 0.1  # Much less than 5 second deadline


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
