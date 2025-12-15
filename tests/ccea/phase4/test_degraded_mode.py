# -*- coding: utf-8 -*-
"""
Tests for Degraded Mode Manager - Phase 4.7

Tests cover:
- Mode transitions (cloud down, network down, data feed invalid)
- Action policies (continue, restrict, close-only, pause, halt)
- Auto-recovery
- Multiple concurrent modes
- Status reporting
"""

from datetime import datetime, timedelta

import pytest

from packages.agent.daemon.degraded_mode import (
    DegradedModeManager,
    DegradedModeConfig,
    DegradedMode,
    DegradedModeAction,
    DegradedModeEvent,
)


class TestDegradedModeEvent:
    """Test DegradedModeEvent dataclass."""

    def test_creation(self):
        """Test event creation."""
        event = DegradedModeEvent(
            mode=DegradedMode.CLOUD_UNREACHABLE,
            action_taken=DegradedModeAction.CONTINUE,
            reason="Cloud connection lost",
        )

        assert event.mode == DegradedMode.CLOUD_UNREACHABLE
        assert event.action_taken == DegradedModeAction.CONTINUE
        assert event.is_active is True  # No exit time

    def test_duration_calculation(self):
        """Test duration calculation."""
        event = DegradedModeEvent(
            entered_at=datetime.utcnow() - timedelta(minutes=5)
        )

        # Duration should be approximately 5 minutes
        assert event.duration_seconds >= 299

    def test_is_active_false_when_exited(self):
        """Test is_active is False when exited."""
        event = DegradedModeEvent(
            entered_at=datetime.utcnow() - timedelta(minutes=5),
            exited_at=datetime.utcnow(),
        )

        assert event.is_active is False

    def test_to_dict(self):
        """Test serialization."""
        event = DegradedModeEvent(
            mode=DegradedMode.DATA_FEED_STALE,
            action_taken=DegradedModeAction.PAUSE,
            reason="Data feed stale for 60s",
            details={"elapsed_seconds": 60},
        )

        data = event.to_dict()

        assert data["mode"] == "DATA_FEED_STALE"
        assert data["action_taken"] == "pause"
        assert data["details"]["elapsed_seconds"] == 60


class TestDegradedModeConfig:
    """Test DegradedModeConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DegradedModeConfig()

        assert config.cloud_timeout_seconds == 60
        assert config.data_stale_threshold_seconds == 30
        assert config.broker_unreachable_action == DegradedModeAction.HALT
        assert config.auto_recover is True

    def test_to_dict(self):
        """Test serialization."""
        config = DegradedModeConfig(
            cloud_timeout_seconds=120,
            cloud_unreachable_action=DegradedModeAction.RESTRICT,
        )

        data = config.to_dict()

        assert data["cloud_timeout_seconds"] == 120
        assert data["cloud_unreachable_action"] == "restrict"


class TestDegradedModeManager:
    """Test DegradedModeManager operations."""

    @pytest.fixture
    def manager(self):
        """Create degraded mode manager."""
        config = DegradedModeConfig(
            cloud_timeout_seconds=60,
            data_stale_threshold_seconds=30,
            broker_timeout_seconds=30,
            latency_threshold_ms=2000,
        )
        return DegradedModeManager(config=config)

    def test_initial_state_normal(self, manager):
        """Test initial state is normal."""
        assert manager.current_mode == DegradedMode.NORMAL
        assert manager.is_degraded is False
        assert manager.can_submit_order() is True
        assert manager.can_close_position() is True
        assert manager.is_halted() is False

    def test_report_cloud_connected(self, manager):
        """Test reporting cloud connected."""
        event = manager.report_cloud_status(connected=True)

        assert event is None  # No mode change
        assert manager.current_mode == DegradedMode.NORMAL

    def test_report_cloud_unreachable(self, manager):
        """Test reporting cloud unreachable."""
        # First report - starts tracking
        manager.report_cloud_status(connected=True)

        # Second report after timeout - triggers mode
        manager._last_cloud_contact = datetime.utcnow() - timedelta(seconds=120)
        event = manager.report_cloud_status(connected=False)

        assert event is not None
        assert manager.current_mode == DegradedMode.CLOUD_UNREACHABLE
        assert manager.is_degraded is True

    def test_cloud_unreachable_continues_by_default(self, manager):
        """Test cloud unreachable allows continue by default."""
        manager._last_cloud_contact = datetime.utcnow() - timedelta(seconds=120)
        manager.report_cloud_status(connected=False)

        # Default action is CONTINUE for cloud unreachable
        assert manager.can_submit_order() is True
        assert manager.is_halted() is False

    def test_report_data_feed_stale(self, manager):
        """Test reporting stale data feed."""
        last_update = datetime.utcnow() - timedelta(seconds=60)
        event = manager.report_data_feed(last_update=last_update)

        assert event is not None
        assert manager.current_mode == DegradedMode.DATA_FEED_STALE
        assert manager.is_paused() is True  # Default action is PAUSE

    def test_report_data_feed_invalid(self, manager):
        """Test reporting invalid data feed."""
        event = manager.report_data_feed(last_update=None, is_valid=False)

        assert event is not None
        assert manager.current_mode == DegradedMode.DATA_FEED_INVALID
        assert manager.is_halted() is True  # Default action is HALT

    def test_report_data_feed_recovery(self, manager):
        """Test data feed recovery."""
        # Enter stale mode
        stale_time = datetime.utcnow() - timedelta(seconds=60)
        manager.report_data_feed(last_update=stale_time)
        assert manager.is_degraded is True

        # Report fresh data
        manager.report_data_feed(last_update=datetime.utcnow())
        assert manager.is_degraded is False
        assert manager.current_mode == DegradedMode.NORMAL

    def test_report_broker_unreachable(self, manager):
        """Test reporting broker unreachable."""
        # Initial contact
        manager.report_broker_status(connected=True)

        # After timeout
        manager._last_broker_contact = datetime.utcnow() - timedelta(seconds=60)
        event = manager.report_broker_status(connected=False)

        assert event is not None
        assert manager.current_mode == DegradedMode.BROKER_UNREACHABLE
        assert manager.is_halted() is True  # Broker unreachable = HALT

    def test_report_high_latency(self, manager):
        """Test reporting high latency."""
        event = manager.report_latency(latency_ms=5000)

        assert event is not None
        assert DegradedMode.HIGH_LATENCY in manager.active_modes
        # High latency action is RESTRICT by default
        assert manager.can_submit_order() is True  # Still allowed

    def test_report_normal_latency_recovery(self, manager):
        """Test latency recovery."""
        # Enter high latency mode
        manager.report_latency(latency_ms=5000)
        assert DegradedMode.HIGH_LATENCY in manager.active_modes

        # Report normal latency
        manager.report_latency(latency_ms=100)
        assert DegradedMode.HIGH_LATENCY not in manager.active_modes

    def test_manual_restrict_mode(self, manager):
        """Test manual restriction mode."""
        event = manager.enter_manual_restrict(reason="Maintenance window")

        assert event.mode == DegradedMode.MANUAL_RESTRICT
        assert manager.is_degraded is True

        # Exit manual restrict
        manager.exit_manual_restrict()
        assert manager.is_degraded is False

    def test_multiple_concurrent_modes(self, manager):
        """Test multiple degraded modes simultaneously."""
        # Cloud unreachable
        manager._last_cloud_contact = datetime.utcnow() - timedelta(seconds=120)
        manager.report_cloud_status(connected=False)

        # High latency
        manager.report_latency(latency_ms=5000)

        # Both modes should be active
        assert DegradedMode.CLOUD_UNREACHABLE in manager.active_modes
        assert DegradedMode.HIGH_LATENCY in manager.active_modes
        assert len(manager.active_modes) == 2

    def test_mode_priority_broker_highest(self, manager):
        """Test broker unreachable has highest priority."""
        # Enter cloud unreachable
        manager._last_cloud_contact = datetime.utcnow() - timedelta(seconds=120)
        manager.report_cloud_status(connected=False)

        # Enter broker unreachable
        manager._last_broker_contact = datetime.utcnow() - timedelta(seconds=60)
        manager.report_broker_status(connected=False)

        # Broker should be the current mode (highest priority)
        assert manager.current_mode == DegradedMode.BROKER_UNREACHABLE

    def test_action_priority_most_restrictive(self, manager):
        """Test most restrictive action is applied."""
        # Cloud unreachable (CONTINUE)
        manager._last_cloud_contact = datetime.utcnow() - timedelta(seconds=120)
        manager.report_cloud_status(connected=False)
        assert manager.current_action == DegradedModeAction.CONTINUE

        # Data feed stale (PAUSE)
        stale_time = datetime.utcnow() - timedelta(seconds=60)
        manager.report_data_feed(last_update=stale_time)

        # Most restrictive action should be PAUSE
        assert manager.current_action == DegradedModeAction.PAUSE

    def test_can_submit_order_restrictions(self, manager):
        """Test order submission restrictions."""
        # CONTINUE - can submit
        assert manager.can_submit_order() is True

        # PAUSE - cannot submit
        stale_time = datetime.utcnow() - timedelta(seconds=60)
        manager.report_data_feed(last_update=stale_time)
        assert manager.can_submit_order() is False

    def test_can_close_position_restrictions(self, manager):
        """Test position closing restrictions."""
        # Default - can close
        assert manager.can_close_position() is True

        # HALT - cannot close
        manager.report_data_feed(last_update=None, is_valid=False)
        assert manager.can_close_position() is False

    def test_callback_on_mode_change(self, manager):
        """Test callback is called on mode change."""
        callbacks = []

        def on_change(mode, action):
            callbacks.append((mode, action))

        manager._on_mode_change = on_change

        # Trigger mode change
        manager.report_data_feed(last_update=None, is_valid=False)

        assert len(callbacks) >= 1
        assert callbacks[0][0] == DegradedMode.DATA_FEED_INVALID

    def test_get_status(self, manager):
        """Test status reporting."""
        # Enter degraded mode
        manager._last_cloud_contact = datetime.utcnow() - timedelta(seconds=120)
        manager.report_cloud_status(connected=False)

        status = manager.get_status()

        assert status["is_degraded"] is True
        assert status["current_mode"] == "CLOUD_UNREACHABLE"
        assert "CLOUD_UNREACHABLE" in status["active_modes"]
        assert "can_submit_order" in status
        assert "is_halted" in status

    def test_get_history(self, manager):
        """Test getting mode history."""
        # Trigger some mode changes
        manager._last_cloud_contact = datetime.utcnow() - timedelta(seconds=120)
        manager.report_cloud_status(connected=False)
        manager.report_cloud_status(connected=True)  # Recovery

        history = manager.get_history()

        assert len(history) >= 1


class TestDegradedModeActions:
    """Test different degraded mode actions."""

    def test_close_only_mode(self):
        """Test CLOSE_ONLY mode."""
        config = DegradedModeConfig(
            data_feed_stale_action=DegradedModeAction.CLOSE_ONLY,
        )
        manager = DegradedModeManager(config=config)

        stale_time = datetime.utcnow() - timedelta(seconds=60)
        manager.report_data_feed(last_update=stale_time)

        # Cannot submit new orders
        assert manager.can_submit_order() is False
        # Can close positions
        assert manager.can_close_position() is True

    def test_restrict_mode(self):
        """Test RESTRICT mode."""
        config = DegradedModeConfig(
            high_latency_action=DegradedModeAction.RESTRICT,
        )
        manager = DegradedModeManager(config=config)

        manager.report_latency(latency_ms=5000)

        # Can still submit (with restrictions)
        assert manager.can_submit_order() is True
        # Can close
        assert manager.can_close_position() is True


class TestSafeDegradedBehavior:
    """Test safe degraded mode behaviors per Design Doc."""

    def test_cloud_down_agent_continues(self):
        """Test agent continues autonomously when cloud is down."""
        config = DegradedModeConfig(
            cloud_unreachable_action=DegradedModeAction.CONTINUE,
        )
        manager = DegradedModeManager(config=config)

        manager._last_cloud_contact = datetime.utcnow() - timedelta(seconds=120)
        manager.report_cloud_status(connected=False)

        # Agent should continue operating
        assert manager.can_submit_order() is True
        assert manager.is_halted() is False

    def test_network_down_restricts(self):
        """Test network degradation restricts but doesn't halt."""
        config = DegradedModeConfig(
            high_latency_action=DegradedModeAction.RESTRICT,
        )
        manager = DegradedModeManager(config=config)

        manager.report_latency(latency_ms=10000)

        # Restricted but not halted
        assert manager.is_degraded is True
        assert manager.is_halted() is False

    def test_data_feed_invalid_halts(self):
        """Test invalid data feed triggers halt."""
        manager = DegradedModeManager()

        manager.report_data_feed(last_update=None, is_valid=False)

        # Must halt - can't trade on invalid data
        assert manager.is_halted() is True
        assert manager.can_submit_order() is False
        assert manager.can_close_position() is False

    def test_broker_unreachable_halts(self):
        """Test broker unreachable triggers halt."""
        manager = DegradedModeManager()

        manager._last_broker_contact = datetime.utcnow() - timedelta(seconds=60)
        manager.report_broker_status(connected=False)

        # Must halt - can't execute without broker
        assert manager.is_halted() is True
