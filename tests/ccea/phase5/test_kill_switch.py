# -*- coding: utf-8 -*-
"""
Tests for Kill Switch Manager.

Design Doc 9.4: Kill Switch + halt reasons
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from packages.agent.daemon.kill_switch import (
    KillSwitchManager,
    KillSwitchConfig,
    HaltReason,
    HaltReasonType,
    HaltSeverity,
    HaltAction,
    HaltEvent,
)


class TestHaltReason:
    """Tests for HaltReason dataclass."""

    def test_create_halt_reason(self):
        """Test creating a halt reason."""
        reason = HaltReason(
            reason_type=HaltReasonType.MAX_DAILY_LOSS,
            severity=HaltSeverity.CRITICAL,
            message="Daily loss exceeded",
            details={"loss": "10000", "threshold": "5000"},
        )

        assert reason.reason_type == HaltReasonType.MAX_DAILY_LOSS
        assert reason.severity == HaltSeverity.CRITICAL
        assert reason.message == "Daily loss exceeded"
        assert reason.reason_id is not None
        assert reason.acknowledgment_required is True

    def test_halt_reason_to_dict(self):
        """Test serialization to dictionary."""
        reason = HaltReason(
            reason_type=HaltReasonType.BROKER_ERROR_BURST,
            severity=HaltSeverity.HIGH,
            message="Too many broker errors",
            trigger_source="BrokerConnector",
        )

        d = reason.to_dict()
        assert d["reason_type"] == "BROKER_ERROR_BURST"
        assert d["severity"] == "high"
        assert d["message"] == "Too many broker errors"
        assert d["trigger_source"] == "BrokerConnector"

    def test_halt_reason_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "reason_id": "test-123",
            "reason_type": "LATENCY_SPIKE",
            "severity": "medium",
            "message": "High latency detected",
            "details": {"latency_ms": 6000},
            "timestamp": "2025-01-01T00:00:00",
            "trigger_source": "LatencyMonitor",
            "acknowledgment_required": False,
        }

        reason = HaltReason.from_dict(data)
        assert reason.reason_id == "test-123"
        assert reason.reason_type == HaltReasonType.LATENCY_SPIKE
        assert reason.severity == HaltSeverity.MEDIUM
        assert reason.message == "High latency detected"

    def test_telemetry_safe_details_redaction(self):
        """Test that sensitive data is redacted."""
        reason = HaltReason(
            reason_type=HaltReasonType.MANUAL_TRIGGER,
            message="Manual halt",
            details={
                "api_key": "secret123",
                "api_secret": "supersecret",
                "loss": "5000",
                "password": "mypass",
            },
        )

        safe = reason.get_telemetry_safe_details()
        assert safe["api_key"] == "[REDACTED]"
        assert safe["api_secret"] == "[REDACTED]"
        assert safe["password"] == "[REDACTED]"
        assert safe["loss"] == "5000"


class TestHaltEvent:
    """Tests for HaltEvent dataclass."""

    def test_create_halt_event(self):
        """Test creating a halt event."""
        reason = HaltReason(
            reason_type=HaltReasonType.MAX_DRAWDOWN,
            severity=HaltSeverity.CRITICAL,
            message="Drawdown exceeded",
        )

        event = HaltEvent(
            halt_reason=reason,
            action_taken=HaltAction.CANCEL_ORDERS,
            orders_cancelled=5,
        )

        assert event.halt_reason == reason
        assert event.action_taken == HaltAction.CANCEL_ORDERS
        assert event.orders_cancelled == 5
        assert event.evidence_hash != ""

    def test_halt_event_evidence_hash(self):
        """Test evidence hash is computed."""
        reason = HaltReason(
            reason_type=HaltReasonType.ORDER_SPAM,
            message="Order spam detected",
        )

        event1 = HaltEvent(halt_reason=reason, action_taken=HaltAction.HALT_ONLY)
        event2 = HaltEvent(halt_reason=reason, action_taken=HaltAction.HALT_ONLY)

        # Different event IDs = different hashes
        assert event1.evidence_hash != event2.evidence_hash


class TestKillSwitchConfig:
    """Tests for KillSwitchConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = KillSwitchConfig()

        assert config.max_daily_loss_pct == Decimal("0.30")
        assert config.max_drawdown_pct == Decimal("0.50")
        assert config.max_broker_errors_per_minute == 5
        assert config.max_latency_ms == 5000
        assert config.require_manual_reset is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = KillSwitchConfig(
            max_daily_loss_pct=Decimal("0.10"),
            max_orders_per_second=5,
            cooldown_seconds=600,
        )

        assert config.max_daily_loss_pct == Decimal("0.10")
        assert config.max_orders_per_second == 5
        assert config.cooldown_seconds == 600


class TestKillSwitchManager:
    """Tests for KillSwitchManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create KillSwitchManager with temp storage."""
        config = KillSwitchConfig(
            history_file=temp_dir / "halt_history.json",
            cooldown_seconds=0,  # No cooldown for tests
        )
        return KillSwitchManager(config=config)

    def test_initial_state(self, manager):
        """Test initial state is not triggered."""
        assert manager.is_triggered is False
        assert manager.current_halt is None

    def test_trigger_kill_switch(self, manager):
        """Test triggering kill switch."""
        reason = HaltReason(
            reason_type=HaltReasonType.MAX_DAILY_LOSS,
            severity=HaltSeverity.CRITICAL,
            message="Test trigger",
        )

        event = manager.trigger(reason)

        assert manager.is_triggered is True
        assert manager.current_halt is not None
        assert event.halt_reason.message == "Test trigger"
        assert len(manager.halt_history) == 1

    def test_trigger_with_cancel_callback(self, manager):
        """Test trigger calls cancel orders callback."""
        cancel_fn = MagicMock(return_value=5)
        manager._cancel_orders_fn = cancel_fn

        reason = HaltReason(
            reason_type=HaltReasonType.BROKER_ERROR_BURST,
            message="Errors",
        )

        event = manager.trigger(reason, action=HaltAction.CANCEL_ORDERS)

        cancel_fn.assert_called_once()
        assert event.orders_cancelled == 5

    def test_acknowledge_and_reset(self, manager):
        """Test acknowledge and reset flow."""
        reason = HaltReason(
            reason_type=HaltReasonType.MANUAL_TRIGGER,
            message="Manual test",
        )
        manager.trigger(reason)

        # Acknowledge
        assert manager.acknowledge("admin", "APPROVE_RESET_123") is True
        assert manager.current_halt.acknowledged_by == "admin"

        # Reset
        assert manager.reset() is True
        assert manager.is_triggered is False

    def test_invalid_approval_code(self, manager):
        """Test invalid approval code is rejected."""
        reason = HaltReason(reason_type=HaltReasonType.MANUAL_TRIGGER, message="Test")
        manager.trigger(reason)

        # Invalid code
        assert manager.acknowledge("admin", "INVALID_CODE") is False

    def test_check_daily_loss(self, manager):
        """Test daily loss check."""
        # Loss under threshold
        result = manager.check_daily_loss(
            daily_pnl=Decimal("-1000"),
            equity=Decimal("100000"),
        )
        assert result is None

        # Loss over threshold (30%)
        result = manager.check_daily_loss(
            daily_pnl=Decimal("-35000"),
            equity=Decimal("100000"),
        )
        assert result is not None
        assert result.reason_type == HaltReasonType.MAX_DAILY_LOSS

    def test_check_drawdown(self, manager):
        """Test drawdown check."""
        # Drawdown under threshold
        result = manager.check_drawdown(
            current_equity=Decimal("90000"),
            peak_equity=Decimal("100000"),
        )
        assert result is None

        # Drawdown over threshold (50%)
        result = manager.check_drawdown(
            current_equity=Decimal("40000"),
            peak_equity=Decimal("100000"),
        )
        assert result is not None
        assert result.reason_type == HaltReasonType.MAX_DRAWDOWN

    def test_record_error_burst(self, manager):
        """Test error burst detection."""
        # Record multiple errors quickly
        for _ in range(6):
            result = manager.record_error()

        # Should trigger after exceeding per-minute threshold
        assert result is not None
        assert result.reason_type == HaltReasonType.BROKER_ERROR_BURST

    def test_record_latency_spike(self, manager):
        """Test latency spike detection."""
        # Record latency spikes
        for _ in range(3):
            result = manager.record_latency(6000)

        assert result is not None
        assert result.reason_type == HaltReasonType.LATENCY_SPIKE

    def test_record_order_spam(self, manager):
        """Test order spam detection."""
        # Record many orders quickly
        for _ in range(15):
            result = manager.record_order()

        assert result is not None
        assert result.reason_type == HaltReasonType.ORDER_SPAM

    def test_check_data_feed_stale(self, manager):
        """Test data feed staleness check."""
        # Recent data
        result = manager.check_data_feed(datetime.utcnow())
        assert result is None

        # Stale data
        old_time = datetime.utcnow() - timedelta(seconds=60)
        result = manager.check_data_feed(old_time)
        assert result is not None
        assert result.reason_type == HaltReasonType.DATA_FEED_INVALID

    def test_check_position_mismatch(self, manager):
        """Test position mismatch detection."""
        # Small difference
        result = manager.check_position_mismatch(
            local_position=Decimal("100"),
            broker_position=Decimal("100.5"),
            symbol="BTCUSDT",
        )
        assert result is None

        # Large mismatch
        result = manager.check_position_mismatch(
            local_position=Decimal("100"),
            broker_position=Decimal("110"),
            symbol="BTCUSDT",
        )
        assert result is not None
        assert result.reason_type == HaltReasonType.POSITION_MISMATCH

    def test_persistence(self, temp_dir):
        """Test halt history persistence."""
        config = KillSwitchConfig(history_file=temp_dir / "halt_history.json")

        # Create and trigger
        manager1 = KillSwitchManager(config=config)
        reason = HaltReason(reason_type=HaltReasonType.MANUAL_TRIGGER, message="Test")
        manager1.trigger(reason)

        # Create new manager, should load history
        manager2 = KillSwitchManager(config=config)
        assert len(manager2.halt_history) == 1
        assert manager2.halt_history[0].halt_reason.message == "Test"

    def test_export_for_evidence(self, manager):
        """Test evidence export."""
        reason = HaltReason(reason_type=HaltReasonType.MANUAL_TRIGGER, message="Test")
        manager.trigger(reason)

        evidence = manager.export_for_evidence()

        assert "export_timestamp" in evidence
        assert evidence["current_triggered"] is True
        assert evidence["history_count"] == 1

    def test_on_trigger_callback(self, manager):
        """Test on_trigger callback is called."""
        callback = MagicMock()
        manager._on_trigger = callback

        reason = HaltReason(reason_type=HaltReasonType.MANUAL_TRIGGER, message="Test")
        manager.trigger(reason)

        callback.assert_called_once()

    def test_double_trigger_updates_severity(self, manager):
        """Test triggering twice updates to more severe reason."""
        reason1 = HaltReason(
            reason_type=HaltReasonType.LATENCY_SPIKE,
            severity=HaltSeverity.MEDIUM,
            message="Latency warning",
        )
        manager.trigger(reason1)

        reason2 = HaltReason(
            reason_type=HaltReasonType.MAX_DAILY_LOSS,
            severity=HaltSeverity.CRITICAL,
            message="Critical loss",
        )
        manager.trigger(reason2)

        # Should update to more severe
        assert manager.current_halt.halt_reason.severity == HaltSeverity.CRITICAL

    def test_create_halt_reason_factory(self):
        """Test factory method."""
        reason = KillSwitchManager.create_halt_reason(
            reason_type=HaltReasonType.NETWORK_DOWN,
            message="Network lost",
            severity=HaltSeverity.HIGH,
            details={"interface": "eth0"},
            trigger_source="NetworkMonitor",
        )

        assert reason.reason_type == HaltReasonType.NETWORK_DOWN
        assert reason.message == "Network lost"
        assert reason.severity == HaltSeverity.HIGH
