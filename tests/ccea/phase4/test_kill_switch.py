# -*- coding: utf-8 -*-
"""
Tests for Kill Switch Manager - Phase 4.5

Tests cover:
- Halt triggers (loss, drawdown, errors, latency, order spam, data feed, position mismatch)
- Halt actions (cancel orders, flatten)
- Acknowledgment and recovery
- Evidence hash generation
- History persistence
"""

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

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
    """Test HaltReason dataclass."""

    def test_creation_with_defaults(self):
        """Test halt reason creation with defaults."""
        reason = HaltReason()

        assert reason.reason_type == HaltReasonType.MANUAL_TRIGGER
        assert reason.severity == HaltSeverity.CRITICAL
        assert reason.acknowledgment_required is True
        assert reason.reason_id is not None

    def test_creation_with_custom_values(self):
        """Test halt reason creation with custom values."""
        reason = HaltReason(
            reason_type=HaltReasonType.MAX_DAILY_LOSS,
            severity=HaltSeverity.HIGH,
            message="Daily loss exceeded 30%",
            details={"loss_pct": "0.35"},
            trigger_source="test",
        )

        assert reason.reason_type == HaltReasonType.MAX_DAILY_LOSS
        assert reason.severity == HaltSeverity.HIGH
        assert "30%" in reason.message
        assert reason.details["loss_pct"] == "0.35"

    def test_to_dict(self):
        """Test serialization."""
        reason = HaltReason(
            reason_type=HaltReasonType.BROKER_ERROR_BURST,
            message="Too many errors",
        )

        data = reason.to_dict()

        assert data["reason_type"] == "BROKER_ERROR_BURST"
        assert data["message"] == "Too many errors"
        assert "timestamp" in data

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "reason_id": "test-123",
            "reason_type": "LATENCY_SPIKE",
            "severity": "high",
            "message": "High latency",
            "details": {"latency_ms": 5000},
            "timestamp": datetime.utcnow().isoformat(),
            "trigger_source": "test",
            "acknowledgment_required": True,
        }

        reason = HaltReason.from_dict(data)

        assert reason.reason_id == "test-123"
        assert reason.reason_type == HaltReasonType.LATENCY_SPIKE
        assert reason.severity == HaltSeverity.HIGH

    def test_get_telemetry_safe_details(self):
        """Test sensitive data redaction."""
        reason = HaltReason(
            details={
                "message": "public info",
                "api_key": "secret123",
                "password": "secret456",
                "latency": 100,
            }
        )

        safe = reason.get_telemetry_safe_details()

        assert safe["message"] == "public info"
        assert safe["api_key"] == "[REDACTED]"
        assert safe["password"] == "[REDACTED]"
        assert safe["latency"] == 100


class TestHaltEvent:
    """Test HaltEvent dataclass."""

    def test_evidence_hash_computed(self):
        """Test evidence hash is computed automatically."""
        event = HaltEvent()

        assert event.evidence_hash is not None
        assert len(event.evidence_hash) == 64  # SHA256 hex

    def test_to_dict(self):
        """Test serialization."""
        reason = HaltReason(reason_type=HaltReasonType.ORDER_SPAM)
        event = HaltEvent(
            halt_reason=reason,
            action_taken=HaltAction.CANCEL_ORDERS,
            orders_cancelled=5,
        )

        data = event.to_dict()

        assert data["action_taken"] == "cancel_orders"
        assert data["orders_cancelled"] == 5
        assert "evidence_hash" in data


class TestKillSwitchManager:
    """Test KillSwitchManager operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for history."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create kill switch manager."""
        config = KillSwitchConfig(
            max_daily_loss_pct=Decimal("0.30"),
            max_drawdown_pct=Decimal("0.50"),
            max_broker_errors_per_minute=3,
            max_latency_ms=1000,
            max_orders_per_second=5,
            cooldown_seconds=0,  # No cooldown for tests
            require_manual_reset=False,
            history_file=temp_dir / "halt_history.json",
        )
        return KillSwitchManager(config=config)

    def test_initial_state(self, manager):
        """Test initial state is not triggered."""
        assert manager.is_triggered is False
        assert manager.current_halt is None

    def test_trigger_creates_halt_event(self, manager):
        """Test triggering creates halt event."""
        reason = HaltReason(
            reason_type=HaltReasonType.MANUAL_TRIGGER,
            message="Test halt",
        )

        event = manager.trigger(reason)

        assert manager.is_triggered is True
        assert manager.current_halt is not None
        assert event.halt_reason.reason_type == HaltReasonType.MANUAL_TRIGGER

    def test_trigger_with_cancel_orders(self, manager):
        """Test trigger with cancel orders action."""
        orders_cancelled = []

        def cancel_fn():
            orders_cancelled.append(1)
            return 3

        manager._cancel_orders_fn = cancel_fn

        reason = HaltReason(reason_type=HaltReasonType.HARD_CAP_EXCEEDED)
        event = manager.trigger(reason, action=HaltAction.CANCEL_ORDERS)

        assert len(orders_cancelled) == 1
        assert event.orders_cancelled == 3

    def test_trigger_with_flatten(self, manager):
        """Test trigger with flatten action."""
        manager.config.allow_flatten_on_halt = True
        manager._flatten_fn = lambda: True

        reason = HaltReason(reason_type=HaltReasonType.MAX_DAILY_LOSS)
        event = manager.trigger(reason, action=HaltAction.FLATTEN_LOCAL)

        assert event.positions_flattened is True

    def test_second_trigger_updates_severity(self, manager):
        """Test second trigger updates severity if more severe."""
        reason1 = HaltReason(
            reason_type=HaltReasonType.LATENCY_SPIKE,
            severity=HaltSeverity.HIGH,
        )
        reason2 = HaltReason(
            reason_type=HaltReasonType.BROKER_ERROR_BURST,
            severity=HaltSeverity.CRITICAL,
        )

        manager.trigger(reason1)
        event = manager.trigger(reason2)

        # Should keep more severe reason
        assert event.halt_reason.severity == HaltSeverity.CRITICAL

    def test_acknowledge_requires_approval_code(self, manager):
        """Test acknowledgment requires proper approval code."""
        reason = HaltReason(reason_type=HaltReasonType.MANUAL_TRIGGER)
        manager.trigger(reason)

        # Wrong format
        success = manager.acknowledge("admin", "wrong_code")
        assert success is False

        # Correct format
        success = manager.acknowledge("admin", "APPROVE_RESET_12345")
        assert success is True
        assert manager.current_halt.acknowledged_by == "admin"

    def test_reset_after_acknowledgment(self, manager):
        """Test reset after acknowledgment."""
        manager.config.require_manual_reset = True

        reason = HaltReason(reason_type=HaltReasonType.MANUAL_TRIGGER)
        manager.trigger(reason)

        # Cannot reset without acknowledgment
        success = manager.reset()
        assert success is False

        # Acknowledge and reset
        manager.acknowledge("admin", "APPROVE_RESET_99999")
        success = manager.reset()
        assert success is True
        assert manager.is_triggered is False

    def test_check_daily_loss(self, manager):
        """Test daily loss check."""
        # Within limit
        reason = manager.check_daily_loss(
            daily_pnl=Decimal("-2000"),
            equity=Decimal("10000"),
        )
        assert reason is None

        # Exceeds limit (30% loss)
        reason = manager.check_daily_loss(
            daily_pnl=Decimal("-3500"),
            equity=Decimal("10000"),
        )
        assert reason is not None
        assert reason.reason_type == HaltReasonType.MAX_DAILY_LOSS

    def test_check_drawdown(self, manager):
        """Test drawdown check."""
        # Within limit
        reason = manager.check_drawdown(
            current_equity=Decimal("8000"),
            peak_equity=Decimal("10000"),
        )
        assert reason is None

        # Exceeds limit (50% drawdown)
        reason = manager.check_drawdown(
            current_equity=Decimal("4000"),
            peak_equity=Decimal("10000"),
        )
        assert reason is not None
        assert reason.reason_type == HaltReasonType.MAX_DRAWDOWN

    def test_record_error_burst(self, manager):
        """Test error burst detection."""
        # Record errors
        for _ in range(2):
            reason = manager.record_error()
            assert reason is None  # Under threshold

        # Third error triggers
        reason = manager.record_error()
        assert reason is not None
        assert reason.reason_type == HaltReasonType.BROKER_ERROR_BURST

    def test_record_latency_spikes(self, manager):
        """Test latency spike detection."""
        manager.config.max_latency_spikes_in_window = 2

        # First spike
        reason = manager.record_latency(2000)
        assert reason is None

        # Second spike triggers
        reason = manager.record_latency(3000)
        assert reason is not None
        assert reason.reason_type == HaltReasonType.LATENCY_SPIKE

    def test_record_order_spam(self, manager):
        """Test order spam detection."""
        # Record orders quickly
        for _ in range(5):
            reason = manager.record_order()
            assert reason is None

        # Sixth order in same second triggers
        reason = manager.record_order()
        assert reason is not None
        assert reason.reason_type == HaltReasonType.ORDER_SPAM

    def test_check_data_feed(self, manager):
        """Test data feed staleness check."""
        # Recent data
        reason = manager.check_data_feed(datetime.utcnow() - timedelta(seconds=5))
        assert reason is None

        # Stale data (> 30 seconds)
        reason = manager.check_data_feed(datetime.utcnow() - timedelta(seconds=60))
        assert reason is not None
        assert reason.reason_type == HaltReasonType.DATA_FEED_INVALID

    def test_check_position_mismatch(self, manager):
        """Test position mismatch detection."""
        # Within tolerance
        reason = manager.check_position_mismatch(
            local_position=Decimal("1.005"),
            broker_position=Decimal("1.0"),
            symbol="BTCUSDT",
        )
        assert reason is None

        # Exceeds tolerance
        reason = manager.check_position_mismatch(
            local_position=Decimal("1.5"),
            broker_position=Decimal("1.0"),
            symbol="BTCUSDT",
        )
        assert reason is not None
        assert reason.reason_type == HaltReasonType.POSITION_MISMATCH

    def test_halt_history_preserved(self, manager):
        """Test halt history is preserved."""
        reason1 = HaltReason(reason_type=HaltReasonType.MANUAL_TRIGGER, message="First")
        manager.trigger(reason1)
        manager.acknowledge("admin", "APPROVE_RESET_1")
        manager.reset()

        reason2 = HaltReason(reason_type=HaltReasonType.LATENCY_SPIKE, message="Second")
        manager.trigger(reason2)

        history = manager.halt_history
        assert len(history) == 2

    def test_export_for_evidence(self, manager):
        """Test export for evidence pack."""
        reason = HaltReason(reason_type=HaltReasonType.MANUAL_TRIGGER)
        manager.trigger(reason)

        export = manager.export_for_evidence()

        assert export["current_triggered"] is True
        assert export["current_halt"] is not None
        assert "config" in export

    def test_create_halt_reason_factory(self, manager):
        """Test factory method for halt reasons."""
        reason = KillSwitchManager.create_halt_reason(
            reason_type=HaltReasonType.NETWORK_DOWN,
            message="Network connection lost",
            severity=HaltSeverity.HIGH,
            details={"last_ping_ms": 5000},
        )

        assert reason.reason_type == HaltReasonType.NETWORK_DOWN
        assert reason.severity == HaltSeverity.HIGH
        assert reason.details["last_ping_ms"] == 5000

    def test_callback_on_trigger(self, manager):
        """Test callback is called on trigger."""
        callbacks = []

        def on_trigger(reason):
            callbacks.append(reason)

        manager._on_trigger = on_trigger

        reason = HaltReason(reason_type=HaltReasonType.MANUAL_TRIGGER)
        manager.trigger(reason)

        assert len(callbacks) == 1
        assert callbacks[0].reason_type == HaltReasonType.MANUAL_TRIGGER


class TestKillSwitchPersistence:
    """Test kill switch persistence."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_history_persisted_to_disk(self, temp_dir):
        """Test history is persisted to disk."""
        history_file = temp_dir / "halt_history.json"
        config = KillSwitchConfig(
            history_file=history_file,
            cooldown_seconds=0,
            require_manual_reset=False,
        )

        # Create manager and trigger halt
        manager1 = KillSwitchManager(config=config)
        reason = HaltReason(
            reason_type=HaltReasonType.MANUAL_TRIGGER,
            message="Persistent test",
        )
        manager1.trigger(reason)
        manager1.acknowledge("admin", "APPROVE_RESET_1")
        manager1.reset()

        # Create new manager - should load history
        manager2 = KillSwitchManager(config=config)

        assert len(manager2.halt_history) == 1
        assert manager2.halt_history[0].halt_reason.message == "Persistent test"
