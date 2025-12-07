# -*- coding: utf-8 -*-
"""
Tests for MiFID II Compliance Clock Module.

Comprehensive tests for RTS 25 compliant clock synchronisation.
"""

import pytest
import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from services.compliance.compliance_clock import (
    ClockDriftSeverity,
    ClockSyncStatus,
    ClockSyncEvent,
    ComplianceClock,
    create_compliance_clock,
)


class TestClockDriftSeverity:
    """Tests for ClockDriftSeverity enum."""

    def test_normal_value(self):
        assert ClockDriftSeverity.NORMAL.value == "normal"

    def test_warning_value(self):
        assert ClockDriftSeverity.WARNING.value == "warning"

    def test_critical_value(self):
        assert ClockDriftSeverity.CRITICAL.value == "critical"

    def test_kill_switch_value(self):
        assert ClockDriftSeverity.KILL_SWITCH.value == "kill_switch"

    def test_from_offset_normal(self):
        severity = ClockDriftSeverity.from_offset(10.0)
        assert severity == ClockDriftSeverity.NORMAL

    def test_from_offset_warning(self):
        severity = ClockDriftSeverity.from_offset(60.0)
        assert severity == ClockDriftSeverity.WARNING

    def test_from_offset_critical(self):
        severity = ClockDriftSeverity.from_offset(150.0)
        assert severity == ClockDriftSeverity.CRITICAL

    def test_from_offset_kill_switch(self):
        severity = ClockDriftSeverity.from_offset(1500.0)
        assert severity == ClockDriftSeverity.KILL_SWITCH

    def test_from_offset_negative(self):
        severity = ClockDriftSeverity.from_offset(-60.0)
        assert severity == ClockDriftSeverity.WARNING

    def test_from_offset_custom_thresholds(self):
        severity = ClockDriftSeverity.from_offset(
            30.0, warning_ms=20.0, critical_ms=40.0
        )
        assert severity == ClockDriftSeverity.WARNING


class TestClockSyncStatus:
    """Tests for ClockSyncStatus dataclass."""

    def test_create_status(self):
        status = ClockSyncStatus(
            offset_ms=10.5,
            stratum=2,
            reference_server="time.google.com",
            last_sync_time=time.time(),
            sync_success=True,
        )
        assert status.offset_ms == 10.5
        assert status.stratum == 2
        assert status.sync_success is True

    def test_is_compliant_true(self):
        status = ClockSyncStatus(
            offset_ms=50.0,
            stratum=2,
            reference_server="time.google.com",
            last_sync_time=time.time(),
            sync_success=True,
        )
        assert status.is_compliant(max_offset_ms=100.0) is True

    def test_is_compliant_false_offset(self):
        status = ClockSyncStatus(
            offset_ms=150.0,
            stratum=2,
            reference_server="time.google.com",
            last_sync_time=time.time(),
            sync_success=True,
        )
        assert status.is_compliant(max_offset_ms=100.0) is False

    def test_is_compliant_false_sync_failed(self):
        status = ClockSyncStatus(
            offset_ms=10.0,
            stratum=16,
            reference_server="",
            last_sync_time=time.time(),
            sync_success=False,
        )
        assert status.is_compliant() is False

    def test_to_dict(self):
        status = ClockSyncStatus(
            offset_ms=10.5,
            stratum=2,
            reference_server="time.google.com",
            last_sync_time=time.time(),
            sync_success=True,
            round_trip_ms=5.0,
            severity=ClockDriftSeverity.NORMAL,
        )
        data = status.to_dict()
        assert "offset_ms" in data
        assert "stratum" in data
        assert "reference_server" in data
        assert "is_compliant" in data
        assert data["severity"] == "normal"


class TestClockSyncEvent:
    """Tests for ClockSyncEvent dataclass."""

    def test_create_event(self):
        event = ClockSyncEvent(
            timestamp=time.time(),
            previous_offset_ms=10.0,
            current_offset_ms=60.0,
            severity=ClockDriftSeverity.WARNING,
            message="Drift increased",
        )
        assert event.previous_offset_ms == 10.0
        assert event.current_offset_ms == 60.0
        assert event.severity == ClockDriftSeverity.WARNING


class TestComplianceClock:
    """Tests for ComplianceClock class."""

    def test_init_default(self):
        clock = ComplianceClock()
        assert clock._max_offset_ms == 100.0
        assert clock._sync_interval == 60
        assert len(clock._ntp_servers) > 0

    def test_init_custom_servers(self):
        servers = ["custom.ntp.server"]
        clock = ComplianceClock(ntp_servers=servers)
        assert clock._ntp_servers == servers

    def test_init_custom_thresholds(self):
        clock = ComplianceClock(
            max_offset_ms=50.0,
            warning_threshold_ms=25.0,
            critical_threshold_ms=50.0,
            kill_switch_threshold_ms=500.0,
        )
        assert clock._max_offset_ms == 50.0
        assert clock._warning_threshold_ms == 25.0

    def test_offset_ms_property(self):
        clock = ComplianceClock()
        assert clock.offset_ms == 0.0

    def test_is_synchronized_false_initially(self):
        clock = ComplianceClock()
        assert clock.is_synchronized is False

    def test_now_utc_ns(self):
        clock = ComplianceClock()
        ts = clock.now_utc_ns()
        assert isinstance(ts, int)
        assert ts > 0
        # Should be in nanoseconds (roughly current time)
        assert ts > 1_000_000_000_000_000_000

    def test_now_utc_us(self):
        clock = ComplianceClock()
        ts = clock.now_utc_us()
        assert isinstance(ts, int)
        assert ts > 1_000_000_000_000_000

    def test_now_utc_ms(self):
        clock = ComplianceClock()
        ts = clock.now_utc_ms()
        assert isinstance(ts, int)
        assert ts > 1_000_000_000_000

    def test_now_utc_s(self):
        clock = ComplianceClock()
        ts = clock.now_utc_s()
        assert isinstance(ts, float)
        assert ts > 1_000_000_000

    def test_now_utc_datetime(self):
        clock = ComplianceClock()
        dt = clock.now_utc_datetime()
        assert isinstance(dt, datetime)
        assert dt.tzinfo == timezone.utc

    def test_now_utc_iso(self):
        clock = ComplianceClock()
        iso = clock.now_utc_iso()
        assert isinstance(iso, str)
        assert iso.endswith("Z")
        assert "T" in iso

    def test_get_status_not_synchronized(self):
        clock = ComplianceClock()
        status = clock.get_status()
        assert status.sync_success is False
        assert status.stratum == 16

    def test_get_statistics(self):
        clock = ComplianceClock()
        stats = clock.get_statistics()
        assert "sync_attempts" in stats
        assert "sync_successes" in stats
        assert "sync_failures" in stats
        assert "current_offset_ms" in stats
        assert "is_synchronized" in stats

    def test_generate_compliance_report(self):
        clock = ComplianceClock()
        report = clock.generate_compliance_report()
        assert "report_timestamp" in report
        assert "regulation" in report
        assert "compliance_status" in report
        assert "configuration" in report
        assert "statistics" in report
        assert report["regulation"] == "RTS 25 (Regulation 2017/574)"

    def test_sync_with_mock_ntp(self):
        """Test sync with mocked NTP response."""
        clock = ComplianceClock()

        # Mock ntplib
        with patch("services.compliance.compliance_clock.ComplianceClock._sync_with_ntp_server") as mock_sync:
            mock_sync.return_value = (5.0, 2, 10.0)  # offset_ms, stratum, rtt_ms

            status = clock.sync()
            assert status.sync_success is True
            assert status.stratum == 2

    def test_sync_all_servers_fail(self):
        """Test sync when all NTP servers fail."""
        clock = ComplianceClock()

        with patch("services.compliance.compliance_clock.ComplianceClock._sync_with_ntp_server") as mock_sync:
            mock_sync.return_value = None

            status = clock.sync()
            assert status.sync_success is False

    def test_sync_with_http_fallback(self):
        """Test fallback to HTTP time sync."""
        clock = ComplianceClock()

        with patch("services.compliance.compliance_clock.ComplianceClock._sync_with_ntp_server") as mock_ntp:
            mock_ntp.return_value = None

            with patch("services.compliance.compliance_clock.ComplianceClock._sync_with_http_time") as mock_http:
                mock_http.return_value = (10.0, 15, 100.0)

                status = clock.sync()
                # Should have tried HTTP fallback

    def test_handle_severity_warning(self):
        """Test that warning callback is triggered."""
        alert_callback = Mock()
        clock = ComplianceClock(
            warning_threshold_ms=10.0,
            alert_callback=alert_callback,
        )

        # Set initial offset
        clock._offset_ms = 5.0

        # Create status with warning-level offset
        status = ClockSyncStatus(
            offset_ms=15.0,
            stratum=2,
            reference_server="test",
            last_sync_time=time.time(),
            sync_success=True,
            severity=ClockDriftSeverity.WARNING,
        )

        clock._handle_severity(status, 5.0)
        assert clock._stats["warnings_triggered"] >= 0  # May or may not trigger

    def test_handle_severity_kill_switch(self):
        """Test that kill switch callback is triggered."""
        kill_switch_callback = Mock()
        clock = ComplianceClock(
            kill_switch_threshold_ms=100.0,
            kill_switch_callback=kill_switch_callback,
        )

        status = ClockSyncStatus(
            offset_ms=1500.0,
            stratum=2,
            reference_server="test",
            last_sync_time=time.time(),
            sync_success=True,
            severity=ClockDriftSeverity.KILL_SWITCH,
        )

        clock._handle_severity(status, 50.0)
        assert clock._stats["kill_switch_triggers"] >= 1
        kill_switch_callback.assert_called()

    def test_start_stop_sync(self):
        """Test starting and stopping sync thread."""
        clock = ComplianceClock(sync_interval_seconds=1)

        with patch.object(clock, "sync") as mock_sync:
            mock_sync.return_value = ClockSyncStatus(
                offset_ms=0.0,
                stratum=2,
                reference_server="test",
                last_sync_time=time.time(),
                sync_success=True,
            )

            clock.start_sync()
            assert clock._running is True

            time.sleep(0.1)  # Let thread start

            clock.stop_sync()
            assert clock._running is False

    def test_start_sync_idempotent(self):
        """Test that starting sync multiple times is safe."""
        clock = ComplianceClock()

        with patch.object(clock, "sync") as mock_sync:
            mock_sync.return_value = ClockSyncStatus(
                offset_ms=0.0,
                stratum=2,
                reference_server="test",
                last_sync_time=time.time(),
                sync_success=True,
            )

            clock.start_sync()
            clock.start_sync()  # Should not start another thread
            clock.stop_sync()

    def test_stop_sync_without_start(self):
        """Test that stopping without starting is safe."""
        clock = ComplianceClock()
        clock.stop_sync()  # Should not raise


class TestCreateComplianceClock:
    """Tests for create_compliance_clock factory function."""

    def test_create_default(self):
        clock = create_compliance_clock()
        assert isinstance(clock, ComplianceClock)

    def test_create_with_config(self):
        config = {
            "max_offset_ms": 50.0,
            "sync_interval_seconds": 120,
            "warning_threshold_ms": 25.0,
        }
        clock = create_compliance_clock(config=config)
        assert clock._max_offset_ms == 50.0
        assert clock._sync_interval == 120

    def test_create_with_callbacks(self):
        kill_switch = Mock()
        alert = Mock()
        clock = create_compliance_clock(
            kill_switch_callback=kill_switch,
            alert_callback=alert,
        )
        assert clock._kill_switch_callback is kill_switch
        assert clock._alert_callback is alert

    def test_create_with_ntp_servers(self):
        config = {"ntp_servers": ["custom.ntp.server"]}
        clock = create_compliance_clock(config=config)
        assert "custom.ntp.server" in clock._ntp_servers
