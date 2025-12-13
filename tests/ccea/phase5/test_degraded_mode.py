# -*- coding: utf-8 -*-
"""
Tests for Degraded Mode Manager.

Design Doc Phase 5: Safe degradation handling.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from packages.agent.daemon.degraded_mode import (
    DegradedModeManager,
    DegradedModeConfig,
    DegradedMode,
    DegradedModeAction,
    DegradedModeEvent,
)


class TestDegradedModeEvent:
    """Tests for DegradedModeEvent."""

    def test_create_event(self):
        """Test creating event."""
        event = DegradedModeEvent(
            mode=DegradedMode.CLOUD_UNREACHABLE,
            action_taken=DegradedModeAction.CONTINUE,
            reason="Cloud connection lost",
        )

        assert event.mode == DegradedMode.CLOUD_UNREACHABLE
        assert event.action_taken == DegradedModeAction.CONTINUE
        assert event.is_active is True
        assert event.duration_seconds >= 0

    def test_event_exit(self):
        """Test event exit."""
        event = DegradedModeEvent(
            mode=DegradedMode.DATA_FEED_STALE,
            action_taken=DegradedModeAction.PAUSE,
            reason="Data feed stale",
        )

        event.exited_at = datetime.utcnow()
        assert event.is_active is False

    def test_event_to_dict(self):
        """Test serialization."""
        event = DegradedModeEvent(
            mode=DegradedMode.BROKER_UNREACHABLE,
            action_taken=DegradedModeAction.HALT,
            reason="Broker down",
        )

        d = event.to_dict()
        assert d["mode"] == "BROKER_UNREACHABLE"
        assert d["action_taken"] == "halt"
        assert d["is_active"] is True


class TestDegradedModeConfig:
    """Tests for DegradedModeConfig."""

    def test_default_config(self):
        """Test default values."""
        config = DegradedModeConfig()

        assert config.cloud_timeout_seconds == 60
        assert config.data_stale_threshold_seconds == 30
        assert config.cloud_unreachable_action == DegradedModeAction.CONTINUE
        assert config.broker_unreachable_action == DegradedModeAction.HALT

    def test_custom_config(self):
        """Test custom values."""
        config = DegradedModeConfig(
            cloud_timeout_seconds=120,
            data_feed_stale_action=DegradedModeAction.RESTRICT,
        )

        assert config.cloud_timeout_seconds == 120
        assert config.data_feed_stale_action == DegradedModeAction.RESTRICT


class TestDegradedModeManager:
    """Tests for DegradedModeManager."""

    @pytest.fixture
    def manager(self):
        """Create DegradedModeManager."""
        config = DegradedModeConfig(
            cloud_timeout_seconds=30,
            data_stale_threshold_seconds=10,
        )
        return DegradedModeManager(config=config)

    def test_initial_state(self, manager):
        """Test initial state is normal."""
        assert manager.current_mode == DegradedMode.NORMAL
        assert manager.is_degraded is False
        assert manager.can_submit_order() is True
        assert manager.is_halted() is False

    def test_cloud_unreachable(self, manager):
        """Test cloud unreachable detection."""
        # First report connected
        manager.report_cloud_status(connected=True)
        assert DegradedMode.CLOUD_UNREACHABLE not in manager.active_modes

        # Report disconnected (no timeout yet)
        manager.report_cloud_status(connected=False)
        # Need to wait for timeout, or simulate old timestamp
        manager._last_cloud_contact = datetime.utcnow() - timedelta(seconds=60)
        event = manager.report_cloud_status(connected=False)

        assert DegradedMode.CLOUD_UNREACHABLE in manager.active_modes

    def test_data_feed_stale(self, manager):
        """Test data feed staleness detection."""
        # Fresh data
        manager.report_data_feed(last_update=datetime.utcnow())
        assert DegradedMode.DATA_FEED_STALE not in manager.active_modes

        # Stale data
        old_time = datetime.utcnow() - timedelta(seconds=60)
        manager.report_data_feed(last_update=old_time)

        assert DegradedMode.DATA_FEED_STALE in manager.active_modes

    def test_data_feed_invalid(self, manager):
        """Test invalid data feed detection."""
        manager.report_data_feed(last_update=datetime.utcnow(), is_valid=False)

        assert DegradedMode.DATA_FEED_INVALID in manager.active_modes
        assert manager.is_halted() is True

    def test_broker_unreachable(self, manager):
        """Test broker unreachable detection."""
        manager.report_broker_status(connected=True)
        assert DegradedMode.BROKER_UNREACHABLE not in manager.active_modes

        manager._last_broker_contact = datetime.utcnow() - timedelta(seconds=60)
        manager.report_broker_status(connected=False)

        assert DegradedMode.BROKER_UNREACHABLE in manager.active_modes
        assert manager.is_halted() is True

    def test_high_latency(self, manager):
        """Test high latency detection."""
        manager.report_latency(latency_ms=100)
        assert DegradedMode.HIGH_LATENCY not in manager.active_modes

        manager.report_latency(latency_ms=5000)
        assert DegradedMode.HIGH_LATENCY in manager.active_modes

    def test_manual_restrict(self, manager):
        """Test manual restriction mode."""
        event = manager.enter_manual_restrict("Maintenance mode")

        assert DegradedMode.MANUAL_RESTRICT in manager.active_modes
        assert event.reason == "Maintenance mode"

        # Exit
        manager.exit_manual_restrict()
        assert DegradedMode.MANUAL_RESTRICT not in manager.active_modes

    def test_action_priority(self, manager):
        """Test action priority (most restrictive wins)."""
        # Enter restrict mode
        manager.enter_manual_restrict()
        assert manager.current_action == DegradedModeAction.RESTRICT

        # Enter halt mode (should override)
        manager.report_data_feed(last_update=datetime.utcnow(), is_valid=False)
        assert manager.current_action == DegradedModeAction.HALT

    def test_mode_priority(self, manager):
        """Test mode priority (worst first)."""
        # Enter cloud unreachable
        manager._enter_mode(
            DegradedMode.CLOUD_UNREACHABLE,
            DegradedModeAction.CONTINUE,
            "Cloud down",
        )

        # Enter broker unreachable (higher priority)
        manager._enter_mode(
            DegradedMode.BROKER_UNREACHABLE,
            DegradedModeAction.HALT,
            "Broker down",
        )

        # Current mode should be broker (higher priority)
        assert manager.current_mode == DegradedMode.BROKER_UNREACHABLE

    def test_can_submit_order(self, manager):
        """Test order submission checks."""
        assert manager.can_submit_order() is True

        # Restrict mode still allows
        manager.enter_manual_restrict()
        assert manager.can_submit_order() is True

        # Halt mode blocks
        manager._enter_mode(
            DegradedMode.BROKER_UNREACHABLE,
            DegradedModeAction.HALT,
            "Broker down",
        )
        assert manager.can_submit_order() is False

    def test_can_close_position(self, manager):
        """Test position closing checks."""
        assert manager.can_close_position() is True

        # Even in close-only mode
        manager._enter_mode(
            DegradedMode.HIGH_LATENCY,
            DegradedModeAction.CLOSE_ONLY,
            "High latency",
        )
        assert manager.can_close_position() is True

    def test_callback_on_mode_change(self, manager):
        """Test callback is called on mode change."""
        callback = MagicMock()
        manager._on_mode_change = callback

        manager.enter_manual_restrict("Test")

        callback.assert_called()

    def test_recovery(self, manager):
        """Test mode recovery."""
        # Enter stale mode
        old_time = datetime.utcnow() - timedelta(seconds=60)
        manager.report_data_feed(last_update=old_time)
        assert DegradedMode.DATA_FEED_STALE in manager.active_modes

        # Recover with fresh data
        manager.report_data_feed(last_update=datetime.utcnow())
        assert DegradedMode.DATA_FEED_STALE not in manager.active_modes

    def test_get_status(self, manager):
        """Test status reporting."""
        manager.enter_manual_restrict("Test mode")

        status = manager.get_status()

        assert status["is_degraded"] is True
        assert "MANUAL_RESTRICT" in status["active_modes"]
        assert status["can_submit_order"] is True

    def test_get_history(self, manager):
        """Test history retrieval."""
        manager.enter_manual_restrict("Test 1")
        manager.exit_manual_restrict()
        manager.enter_manual_restrict("Test 2")

        history = manager.get_history()

        assert len(history) == 2

    def test_multiple_modes(self, manager):
        """Test multiple degraded modes simultaneously."""
        manager.enter_manual_restrict("Manual")
        manager._enter_mode(
            DegradedMode.HIGH_LATENCY,
            DegradedModeAction.RESTRICT,
            "Latency",
        )

        assert len(manager.active_modes) == 2
        assert DegradedMode.MANUAL_RESTRICT in manager.active_modes
        assert DegradedMode.HIGH_LATENCY in manager.active_modes
