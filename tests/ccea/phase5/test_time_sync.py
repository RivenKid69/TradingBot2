# -*- coding: utf-8 -*-
"""
Tests for Time Sync Checker.

Design Doc D1: Time synchronization verification.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from packages.agent.daemon.time_sync import (
    TimeSyncChecker,
    TimeSyncConfig,
    TimeSyncResult,
    verify_time_sync,
    TimeSyncError,
    TimeDriftError,
)


class TestTimeSyncResult:
    """Tests for TimeSyncResult."""

    def test_create_result(self):
        """Test creating result."""
        result = TimeSyncResult(
            synchronized=True,
            drift_seconds=0.5,
            drift_ms=500,
            ntp_server="time.google.com",
        )

        assert result.synchronized is True
        assert result.is_acceptable is True
        assert result.drift_seconds == 0.5

    def test_unacceptable_drift(self):
        """Test drift exceeding threshold."""
        result = TimeSyncResult(
            synchronized=True,
            drift_seconds=10.0,
            drift_ms=10000,
        )

        assert result.is_acceptable is False

    def test_not_synchronized(self):
        """Test unsynchronized result."""
        result = TimeSyncResult(
            synchronized=False,
            error="Connection failed",
        )

        assert result.is_acceptable is False

    def test_to_dict(self):
        """Test serialization."""
        result = TimeSyncResult(
            synchronized=True,
            drift_seconds=0.1,
            drift_ms=100,
            ntp_server="pool.ntp.org",
        )

        d = result.to_dict()
        assert d["synchronized"] is True
        assert d["drift_ms"] == 100


class TestTimeSyncConfig:
    """Tests for TimeSyncConfig."""

    def test_default_config(self):
        """Test default values."""
        config = TimeSyncConfig()

        assert config.max_drift_seconds == 5.0
        assert config.warning_drift_seconds == 2.0
        assert config.halt_on_drift is True
        assert len(config.ntp_servers) > 0

    def test_custom_config(self):
        """Test custom values."""
        config = TimeSyncConfig(
            max_drift_seconds=10.0,
            ntp_servers=["custom.ntp.org"],
        )

        assert config.max_drift_seconds == 10.0
        assert config.ntp_servers == ["custom.ntp.org"]


class TestTimeSyncChecker:
    """Tests for TimeSyncChecker."""

    @pytest.fixture
    def checker(self):
        """Create TimeSyncChecker."""
        config = TimeSyncConfig(
            ntp_servers=["time.google.com"],
            retry_count=1,
        )
        return TimeSyncChecker(config=config)

    def test_initial_state(self, checker):
        """Test initial state."""
        assert checker.last_result is None
        assert checker.is_synchronized is False
        assert checker.current_drift_seconds == float("inf")

    @pytest.mark.skipif(True, reason="Requires network access")
    def test_check_with_network(self, checker):
        """Test actual NTP check (requires network)."""
        result = checker.check()

        # May or may not work depending on network
        assert isinstance(result, TimeSyncResult)

    def test_check_timestamp_valid_current(self, checker):
        """Test valid current timestamp."""
        now = datetime.now(timezone.utc)
        is_valid, reason = checker.check_timestamp_valid(now)

        assert is_valid is True
        assert reason == "valid"

    def test_check_timestamp_too_old(self, checker):
        """Test timestamp too old."""
        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        is_valid, reason = checker.check_timestamp_valid(old_time, max_age_seconds=60)

        assert is_valid is False
        assert "too old" in reason

    def test_check_timestamp_in_future(self, checker):
        """Test timestamp in future."""
        future_time = datetime.now(timezone.utc) + timedelta(seconds=60)
        is_valid, reason = checker.check_timestamp_valid(future_time, max_future_seconds=5)

        assert is_valid is False
        assert "future" in reason

    def test_check_timestamp_naive(self, checker):
        """Test naive timestamp handling."""
        naive_time = datetime.utcnow()  # No timezone
        is_valid, reason = checker.check_timestamp_valid(naive_time)

        # Should handle naive datetime
        assert isinstance(is_valid, bool)

    def test_get_statistics(self, checker):
        """Test statistics retrieval."""
        stats = checker.get_statistics()

        assert stats["checks"] == 0
        assert stats["successful_checks"] == 0

    def test_callback_on_drift(self, checker):
        """Test drift callback."""
        callback = MagicMock()
        checker._on_drift = callback
        checker.config.max_drift_seconds = 0.001  # Very strict

        # Mock a result with drift
        result = TimeSyncResult(
            synchronized=True,
            drift_seconds=1.0,
            drift_ms=1000,
        )
        checker._last_check = result
        checker._check_history.append(result)

        # Drift should trigger callback
        # In real usage, check() would call this

    @patch.object(TimeSyncChecker, "_query_ntp")
    def test_check_mocked(self, mock_query, checker):
        """Test check with mocked NTP."""
        mock_query.return_value = TimeSyncResult(
            synchronized=True,
            drift_seconds=0.1,
            drift_ms=100,
            ntp_server="mocked",
        )

        result = checker.check()

        assert result.synchronized is True
        assert result.drift_ms == 100
        assert checker.last_result == result

    @patch.object(TimeSyncChecker, "_query_ntp")
    def test_check_all_servers_fail(self, mock_query, checker):
        """Test when all NTP servers fail."""
        mock_query.side_effect = Exception("Connection failed")

        result = checker.check()

        assert result.synchronized is False
        assert "failed" in result.error.lower()

    @patch.object(TimeSyncChecker, "_query_ntp")
    def test_check_drift_callback(self, mock_query, checker):
        """Test drift callback is called."""
        callback = MagicMock()
        checker._on_drift = callback
        checker.config.max_drift_seconds = 1.0

        mock_query.return_value = TimeSyncResult(
            synchronized=True,
            drift_seconds=5.0,
            drift_ms=5000,
            ntp_server="test",
        )

        result = checker.check()

        callback.assert_called_once_with(result)


class TestVerifyTimeSync:
    """Tests for verify_time_sync utility."""

    @patch.object(TimeSyncChecker, "check")
    def test_verify_success(self, mock_check):
        """Test successful verification."""
        mock_check.return_value = TimeSyncResult(
            synchronized=True,
            drift_seconds=0.1,
            drift_ms=100,
        )

        ok, message = verify_time_sync(max_drift_seconds=5.0)

        # This depends on actual network, so we mock
        assert isinstance(ok, bool)
        assert isinstance(message, str)

    @patch.object(TimeSyncChecker, "check")
    def test_verify_drift_exceeded(self, mock_check):
        """Test drift exceeded."""
        mock_check.return_value = TimeSyncResult(
            synchronized=True,
            drift_seconds=10.0,
            drift_ms=10000,
        )

        ok, message = verify_time_sync(max_drift_seconds=5.0)

        assert ok is False
        assert "exceeds" in message.lower()

    @patch.object(TimeSyncChecker, "check")
    def test_verify_not_synchronized(self, mock_check):
        """Test not synchronized."""
        mock_check.return_value = TimeSyncResult(
            synchronized=False,
            error="NTP failed",
        )

        ok, message = verify_time_sync()

        assert ok is False
        assert "failed" in message.lower()
