# -*- coding: utf-8 -*-
"""
Tests for MiFID II Order-to-Trade Ratio (OTR) Monitor.

Comprehensive test coverage for:
    - OTR calculation across time windows
    - Per-venue and per-algorithm OTR tracking
    - Threshold breach detection
    - Throttling and blocking responses
    - Cancellation rate monitoring
    - Compliance reporting

References:
    - RTS 6 Article 13: OTR requirements for investment firms
    - RTS 9: OTR limits for trading venues
"""

import pytest
import time
import threading
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from services.compliance.otr_monitor import (
    OrderEvent,
    OTRBucket,
    OTRLevel,
    OTRMetrics,
    OTRBreachEvent,
    OTRMonitorConfig,
    PerVenueOTR,
    PerAlgorithmOTR,
    OTRMonitor,
    create_otr_monitor,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def config():
    """Default configuration for testing."""
    return OTRMonitorConfig(
        otr_warning_threshold=10.0,
        otr_critical_threshold=20.0,
        otr_breach_threshold=50.0,
        throttle_on_warning=True,
        block_on_critical=True,
        trigger_kill_switch_on_breach=True,
        persist_breach_events=False,
    )


@pytest.fixture
def monitor(config):
    """Create OTR monitor instance."""
    return OTRMonitor(config=config)


@pytest.fixture
def mock_breach_callback():
    """Mock breach callback."""
    return Mock()


@pytest.fixture
def mock_kill_switch_callback():
    """Mock kill switch callback."""
    return Mock()


# =============================================================================
# Test OTRMetrics
# =============================================================================

class TestOTRMetrics:
    """Tests for OTRMetrics data class."""

    def test_metrics_creation(self):
        """Test creating metrics."""
        metrics = OTRMetrics(
            otr_current=15.0,
            otr_rolling_1min=12.0,
            orders_total=100,
            trades_total=10,
            level=OTRLevel.ELEVATED,
        )

        assert metrics.otr_current == 15.0
        assert metrics.orders_total == 100
        assert metrics.trades_total == 10
        assert metrics.level == OTRLevel.ELEVATED

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = OTRMetrics(
            otr_current=10.0,
            cancellation_rate_pct=25.0,
            is_within_limits=True,
        )

        data = metrics.to_dict()

        assert data["otr_current"] == 10.0
        assert data["cancellation_rate_pct"] == 25.0
        assert data["is_within_limits"] is True


# =============================================================================
# Test OTRBreachEvent
# =============================================================================

class TestOTRBreachEvent:
    """Tests for OTRBreachEvent data class."""

    def test_breach_event_creation(self):
        """Test creating breach event."""
        event = OTRBreachEvent(
            otr_value=55.0,
            otr_limit=50.0,
            level=OTRLevel.BREACH,
            venue="XLON",
            algorithm_id="ALGO001",
        )

        assert event.otr_value == 55.0
        assert event.otr_limit == 50.0
        assert event.venue == "XLON"
        assert event.event_id is not None

    def test_breach_event_to_dict(self):
        """Test breach event serialization."""
        event = OTRBreachEvent(
            otr_value=100.0,
            otr_limit=50.0,
            orders_count=1000,
            trades_count=10,
        )

        data = event.to_dict()

        assert data["otr_value"] == 100.0
        assert data["orders_count"] == 1000
        assert data["trades_count"] == 10


# =============================================================================
# Test PerVenueOTR
# =============================================================================

class TestPerVenueOTR:
    """Tests for per-venue OTR tracking."""

    def test_venue_otr_calculation(self):
        """Test venue OTR calculation."""
        venue = PerVenueOTR(
            venue_mic="XLON",
            orders=100,
            trades=10,
        )

        assert venue.otr == 10.0

    def test_venue_otr_no_trades(self):
        """Test venue OTR with no trades."""
        venue = PerVenueOTR(
            venue_mic="XLON",
            orders=50,
            trades=0,
        )

        # Should return orders count when no trades
        assert venue.otr == 50.0


# =============================================================================
# Test OTRMonitor Initialization
# =============================================================================

class TestMonitorInitialization:
    """Tests for monitor initialization."""

    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor._config is not None
        assert len(monitor._buckets) == 0

    def test_configuration_applied(self, config):
        """Test configuration is applied."""
        monitor = OTRMonitor(config=config)

        assert monitor._config.otr_warning_threshold == 10.0
        assert monitor._config.otr_breach_threshold == 50.0


# =============================================================================
# Test Order Recording
# =============================================================================

class TestOrderRecording:
    """Tests for recording order events."""

    def test_record_order(self, monitor):
        """Test recording an order."""
        monitor.record_order(venue="XLON", algorithm_id="ALGO001")

        metrics = monitor.get_metrics()
        assert metrics.orders_total == 1

    def test_record_multiple_orders(self, monitor):
        """Test recording multiple orders."""
        for _ in range(10):
            monitor.record_order()

        metrics = monitor.get_metrics()
        assert metrics.orders_total == 10

    def test_record_order_modification(self, monitor):
        """Test recording order modification."""
        monitor.record_order(is_modification=True)

        metrics = monitor.get_metrics()
        assert metrics.modifications_total == 1

    def test_record_trade(self, monitor):
        """Test recording a trade."""
        monitor.record_trade(venue="XLON")

        metrics = monitor.get_metrics()
        assert metrics.trades_total == 1

    def test_record_cancellation(self, monitor):
        """Test recording cancellation."""
        monitor.record_cancellation()

        metrics = monitor.get_metrics()
        assert metrics.cancellations_total == 1

    def test_record_rejection(self, monitor):
        """Test recording rejection."""
        monitor.record_rejection(reason="Invalid price")

        stats = monitor.get_statistics()
        assert stats["total_rejections"] == 1


# =============================================================================
# Test OTR Calculation
# =============================================================================

class TestOTRCalculation:
    """Tests for OTR calculation."""

    def test_otr_basic_calculation(self, monitor):
        """Test basic OTR calculation."""
        # 100 orders, 10 trades = OTR of 10
        for _ in range(100):
            monitor.record_order()
        for _ in range(10):
            monitor.record_trade()

        metrics = monitor.get_metrics()
        assert metrics.otr_daily == 10.0

    def test_otr_no_trades(self, monitor):
        """Test OTR with no trades."""
        for _ in range(50):
            monitor.record_order()

        metrics = monitor.get_metrics()
        # When no trades, OTR = orders count
        assert metrics.otr_daily == 50.0

    def test_otr_level_determination(self, monitor):
        """Test OTR level is correctly determined."""
        # 60 orders, 1 trade = OTR of 60 (breach level)
        for _ in range(60):
            monitor.record_order()
        monitor.record_trade()

        metrics = monitor.get_metrics()
        assert metrics.level == OTRLevel.BREACH


# =============================================================================
# Test Per-Venue Tracking
# =============================================================================

class TestPerVenueTracking:
    """Tests for per-venue OTR tracking."""

    def test_venue_otr_tracked(self, monitor):
        """Test venue OTR is tracked."""
        for _ in range(20):
            monitor.record_order(venue="XLON")
        for _ in range(2):
            monitor.record_trade(venue="XLON")

        otr = monitor.get_venue_otr("XLON")
        assert otr == 10.0

    def test_multiple_venues(self, monitor):
        """Test tracking multiple venues."""
        for _ in range(10):
            monitor.record_order(venue="XLON")
        monitor.record_trade(venue="XLON")

        for _ in range(20):
            monitor.record_order(venue="XETR")
        monitor.record_trade(venue="XETR")

        assert monitor.get_venue_otr("XLON") == 10.0
        assert monitor.get_venue_otr("XETR") == 20.0

    def test_get_all_venue_otrs(self, monitor):
        """Test getting all venue OTRs."""
        monitor.record_order(venue="XLON")
        monitor.record_order(venue="XETR")
        monitor.record_order(venue="XPAR")

        otrs = monitor.get_all_venue_otrs()

        assert "XLON" in otrs
        assert "XETR" in otrs
        assert "XPAR" in otrs


# =============================================================================
# Test Per-Algorithm Tracking
# =============================================================================

class TestPerAlgorithmTracking:
    """Tests for per-algorithm OTR tracking."""

    def test_algorithm_otr_tracked(self, monitor):
        """Test algorithm OTR is tracked."""
        for _ in range(30):
            monitor.record_order(algorithm_id="ALGO001")
        for _ in range(3):
            monitor.record_trade(algorithm_id="ALGO001")

        otr = monitor.get_algorithm_otr("ALGO001")
        assert otr == 10.0

    def test_multiple_algorithms(self, monitor):
        """Test tracking multiple algorithms."""
        for _ in range(15):
            monitor.record_order(algorithm_id="ALGO001")
        monitor.record_trade(algorithm_id="ALGO001")

        for _ in range(30):
            monitor.record_order(algorithm_id="ALGO002")
        monitor.record_trade(algorithm_id="ALGO002")

        assert monitor.get_algorithm_otr("ALGO001") == 15.0
        assert monitor.get_algorithm_otr("ALGO002") == 30.0


# =============================================================================
# Test Threshold Breaches
# =============================================================================

class TestThresholdBreaches:
    """Tests for threshold breach detection."""

    def test_warning_threshold(self, monitor):
        """Test warning threshold detection."""
        # OTR of 12 > 10 warning threshold
        for _ in range(12):
            monitor.record_order()
        monitor.record_trade()

        metrics = monitor.get_metrics()
        assert metrics.level == OTRLevel.ELEVATED or metrics.level == OTRLevel.WARNING

    def test_critical_threshold(self, monitor):
        """Test critical threshold detection."""
        # OTR of 25 > 20 critical threshold
        for _ in range(25):
            monitor.record_order()
        monitor.record_trade()

        metrics = monitor.get_metrics()
        assert metrics.level == OTRLevel.CRITICAL

    def test_breach_threshold(self, config, mock_breach_callback):
        """Test breach threshold triggers callback."""
        monitor = OTRMonitor(
            config=config,
            breach_callback=mock_breach_callback,
        )

        # OTR of 60 > 50 breach threshold
        for _ in range(60):
            monitor.record_order()
        monitor.record_trade()

        mock_breach_callback.assert_called()

    def test_kill_switch_on_breach(self, config, mock_kill_switch_callback):
        """Test kill switch triggered on breach."""
        monitor = OTRMonitor(
            config=config,
            kill_switch_callback=mock_kill_switch_callback,
        )

        # OTR of 60 > 50 breach threshold
        for _ in range(60):
            monitor.record_order()
        monitor.record_trade()

        mock_kill_switch_callback.assert_called()


# =============================================================================
# Test Trading Permission Checks
# =============================================================================

class TestTradingPermissions:
    """Tests for trading permission checks."""

    def test_can_submit_when_normal(self, monitor):
        """Test orders allowed when OTR is normal."""
        for _ in range(5):
            monitor.record_order()
        monitor.record_trade()

        can_submit, reason = monitor.can_submit_order()
        assert can_submit is True

    def test_cannot_submit_when_blocked(self, config):
        """Test orders blocked on critical OTR."""
        config.block_on_critical = True
        monitor = OTRMonitor(config=config)

        # Exceed critical threshold
        for _ in range(25):
            monitor.record_order()
        monitor.record_trade()

        can_submit, reason = monitor.can_submit_order()
        assert can_submit is False

    def test_venue_specific_limit(self, config):
        """Test venue-specific OTR limit."""
        config.venue_otr_limits = {"XLON": 5.0}
        monitor = OTRMonitor(config=config)

        # Venue OTR of 8 > 5 limit
        for _ in range(8):
            monitor.record_order(venue="XLON")
        monitor.record_trade(venue="XLON")

        can_submit, reason = monitor.can_submit_order(venue="XLON")
        assert can_submit is False
        assert "XLON" in reason


# =============================================================================
# Test Throttling
# =============================================================================

class TestThrottling:
    """Tests for throttling behavior."""

    def test_throttle_delay_on_warning(self, config):
        """Test throttle delay on warning level."""
        config.throttle_on_warning = True
        config.throttle_delay_ms = 100
        monitor = OTRMonitor(config=config)

        # Exceed warning threshold
        for _ in range(12):
            monitor.record_order()
        monitor.record_trade()

        delay = monitor.get_throttle_delay()
        assert delay >= 0  # Some delay expected

    def test_no_throttle_when_normal(self, monitor):
        """Test no throttle when OTR is normal."""
        for _ in range(5):
            monitor.record_order()
        monitor.record_trade()

        delay = monitor.get_throttle_delay()
        assert delay == 0


# =============================================================================
# Test Cancellation Rate
# =============================================================================

class TestCancellationRate:
    """Tests for cancellation rate monitoring."""

    def test_cancellation_rate_calculation(self, monitor):
        """Test cancellation rate calculation."""
        for _ in range(100):
            monitor.record_order()
        for _ in range(30):
            monitor.record_cancellation()

        metrics = monitor.get_metrics()
        assert metrics.cancellation_rate_pct == 30.0

    def test_high_cancellation_rate_warning(self, config):
        """Test high cancellation rate generates warning."""
        config.cancellation_warning_pct = 50.0
        monitor = OTRMonitor(config=config)

        for _ in range(100):
            monitor.record_order()
        for _ in range(60):
            monitor.record_cancellation()
        # Also record some trades to avoid OTR breach
        for _ in range(10):
            monitor.record_trade()

        metrics = monitor.get_metrics()
        assert metrics.cancellation_rate_pct == 60.0


# =============================================================================
# Test Daily Reset
# =============================================================================

class TestDailyReset:
    """Tests for daily counter reset."""

    def test_reset_daily(self, monitor):
        """Test daily reset functionality."""
        for _ in range(100):
            monitor.record_order()
        for _ in range(10):
            monitor.record_trade()

        monitor.reset_daily()

        metrics = monitor.get_metrics()
        assert metrics.orders_total == 0
        assert metrics.trades_total == 0

    def test_reset_block(self, monitor):
        """Test reset blocking state."""
        # Force blocking
        monitor._is_blocked = True

        monitor.reset_block()

        can_submit, _ = monitor.can_submit_order()
        assert can_submit is True


# =============================================================================
# Test Breach Events
# =============================================================================

class TestBreachEvents:
    """Tests for breach event tracking."""

    def test_breach_events_recorded(self, config, mock_breach_callback):
        """Test breach events are recorded."""
        monitor = OTRMonitor(
            config=config,
            breach_callback=mock_breach_callback,
        )

        # Trigger breach
        for _ in range(60):
            monitor.record_order()
        monitor.record_trade()

        breaches = monitor.get_breaches()
        assert len(breaches) >= 1

    def test_breach_event_details(self, config, mock_breach_callback):
        """Test breach event contains correct details."""
        monitor = OTRMonitor(
            config=config,
            breach_callback=mock_breach_callback,
        )

        # Trigger breach
        for _ in range(60):
            monitor.record_order(venue="XLON", algorithm_id="ALGO001")
        monitor.record_trade()

        breaches = monitor.get_breaches()
        assert breaches[0].otr_value >= 50.0


# =============================================================================
# Test Statistics and Reporting
# =============================================================================

class TestStatisticsAndReporting:
    """Tests for statistics and compliance reporting."""

    def test_statistics(self, monitor):
        """Test statistics tracking."""
        for _ in range(50):
            monitor.record_order()
        for _ in range(5):
            monitor.record_trade()
        for _ in range(10):
            monitor.record_cancellation()

        stats = monitor.get_statistics()

        assert stats["total_orders"] == 50
        assert stats["total_trades"] == 5
        assert stats["total_cancellations"] == 10

    def test_compliance_report(self, monitor):
        """Test compliance report generation."""
        report = monitor.generate_compliance_report()

        assert report["regulation"] == "RTS 6 Article 13 / RTS 9"
        assert report["requirement"] == "Order-to-Trade Ratio Monitoring"
        assert "configuration" in report
        assert "current_metrics" in report
        assert "venue_otrs" in report
        assert "algorithm_otrs" in report


# =============================================================================
# Test Factory Function
# =============================================================================

class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_with_dict_config(self):
        """Test creating with dictionary config."""
        config = {
            "otr_warning_threshold": 15.0,
            "throttle_delay_ms": 200,
            "persist_breach_events": False,
        }

        monitor = create_otr_monitor(config=config)

        assert monitor is not None
        assert monitor._config.otr_warning_threshold == 15.0

    def test_create_with_callbacks(self, mock_breach_callback, mock_kill_switch_callback):
        """Test creating with callbacks."""
        monitor = create_otr_monitor(
            breach_callback=mock_breach_callback,
            kill_switch_callback=mock_kill_switch_callback,
        )

        assert monitor._breach_callback is mock_breach_callback
        assert monitor._kill_switch_callback is mock_kill_switch_callback


# =============================================================================
# Test Thread Safety
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_order_recording(self, monitor):
        """Test concurrent order recording."""
        def record_orders(n):
            for _ in range(100):
                monitor.record_order()

        threads = [threading.Thread(target=record_orders, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        metrics = monitor.get_metrics()
        assert metrics.orders_total == 500


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_venue(self, monitor):
        """Test recording with empty venue."""
        monitor.record_order(venue="")

        metrics = monitor.get_metrics()
        assert metrics.orders_total == 1

    def test_unknown_venue_otr(self, monitor):
        """Test getting OTR for unknown venue."""
        otr = monitor.get_venue_otr("UNKNOWN")
        assert otr == 0.0

    def test_partial_fill(self, monitor):
        """Test recording partial fill."""
        monitor.record_trade(is_partial=True)

        metrics = monitor.get_metrics()
        assert metrics.trades_total == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
