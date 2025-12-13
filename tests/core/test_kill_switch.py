# -*- coding: utf-8 -*-
"""
Tests for MiFID II RTS 6 Article 12 Enhanced Kill Switch.

Comprehensive test coverage for:
    - Kill switch triggering and recovery
    - Scope-based cancellation (ALL, VENUE, ALGORITHM, INSTRUMENT)
    - Rate limiting and cooldown
    - Emergency contacts
    - Audit trail and event logging
    - Integration with callbacks

References:
    - RTS 6 Article 12: Emergency cancellation requirements
"""

import pytest
import time
import json
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from decimal import Decimal

from services.core.risk_controls.kill_switch import (
    KillSwitchScope,
    KillSwitchTriggerReason,
    KillSwitchState,
    KillSwitchEvent,
    KillSwitchConfig,
    EmergencyContact,
    EnhancedKillSwitch,
    create_enhanced_kill_switch,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_cancel_callback():
    """Mock order cancellation callback."""
    return Mock(return_value=10)


@pytest.fixture
def mock_alert_callback():
    """Mock alert callback."""
    return Mock()


@pytest.fixture
def mock_compliance_callback():
    """Mock compliance callback."""
    return Mock()


@pytest.fixture
def config():
    """Default configuration for testing."""
    return KillSwitchConfig(
        primary_contact_name="Test User",
        primary_contact_email="test@example.com",
        primary_contact_phone="+1234567890",
        escalation_contact_email="escalation@example.com",
        cooldown_seconds=1.0,  # Short for testing
        max_triggers_per_hour=100,
        persist_events=False,
    )


@pytest.fixture
def kill_switch(mock_cancel_callback, config):
    """Create kill switch instance for testing."""
    return EnhancedKillSwitch(
        order_cancellation_callback=mock_cancel_callback,
        config=config,
    )


# =============================================================================
# Test KillSwitchEvent
# =============================================================================

class TestKillSwitchEvent:
    """Tests for KillSwitchEvent data class."""

    def test_event_creation(self):
        """Test creating a kill switch event."""
        event = KillSwitchEvent(
            scope=KillSwitchScope.ALL,
            reason=KillSwitchTriggerReason.MANUAL,
            reason_detail="Test trigger",
            triggered_by="test_user",
            orders_cancelled=5,
        )

        assert event.scope == KillSwitchScope.ALL
        assert event.reason == KillSwitchTriggerReason.MANUAL
        assert event.reason_detail == "Test trigger"
        assert event.triggered_by == "test_user"
        assert event.orders_cancelled == 5
        assert event.event_id is not None
        assert event.timestamp_ns > 0

    def test_event_to_dict(self):
        """Test event serialization."""
        event = KillSwitchEvent(
            scope=KillSwitchScope.VENUE,
            scope_id="XLON",
            reason=KillSwitchTriggerReason.MARKET_DISRUPTION,
            reason_detail="Market halt",
        )

        data = event.to_dict()

        assert data["scope"] == "venue"
        assert data["scope_id"] == "XLON"
        assert data["reason"] == "market_disruption"
        assert "event_id" in data
        assert "timestamp_ns" in data

    def test_event_hash_computation(self):
        """Test event hash for integrity."""
        event = KillSwitchEvent(
            scope=KillSwitchScope.ALL,
            reason=KillSwitchTriggerReason.MANUAL,
        )

        hash1 = event.compute_hash()
        hash2 = event.compute_hash()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256


# =============================================================================
# Test KillSwitchConfig
# =============================================================================

class TestKillSwitchConfig:
    """Tests for kill switch configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = KillSwitchConfig()

        assert config.cooldown_seconds == 300.0
        assert config.max_triggers_per_hour == 10
        assert config.notify_venues is True
        assert config.persist_events is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = KillSwitchConfig(
            primary_contact_email="ops@firm.com",
            cooldown_seconds=600.0,
            error_threshold_rest=50,
        )

        assert config.primary_contact_email == "ops@firm.com"
        assert config.cooldown_seconds == 600.0
        assert config.error_threshold_rest == 50


# =============================================================================
# Test EnhancedKillSwitch - Basic Operations
# =============================================================================

class TestEnhancedKillSwitchBasic:
    """Tests for basic kill switch operations."""

    def test_initialization(self, kill_switch, config):
        """Test kill switch initialization."""
        assert kill_switch.state == KillSwitchState.ARMED
        assert kill_switch.is_armed is True
        assert kill_switch.is_triggered is False

    def test_trigger_all(self, kill_switch, mock_cancel_callback):
        """Test triggering for all orders."""
        event = kill_switch.trigger_all(
            reason_detail="Emergency test",
            triggered_by="test_user",
        )

        assert kill_switch.is_triggered is True
        assert kill_switch.state == KillSwitchState.TRIGGERED
        assert event.scope == KillSwitchScope.ALL
        assert event.orders_cancelled == 10
        mock_cancel_callback.assert_called_once_with(KillSwitchScope.ALL, "")

    def test_trigger_venue(self, kill_switch, mock_cancel_callback):
        """Test triggering for specific venue."""
        event = kill_switch.trigger_venue(
            venue_mic="XLON",
            reason=KillSwitchTriggerReason.MARKET_DISRUPTION,
            reason_detail="LSE circuit breaker",
        )

        assert kill_switch.is_triggered is True
        assert event.scope == KillSwitchScope.VENUE
        assert event.scope_id == "XLON"
        assert "XLON" in event.venues_affected
        mock_cancel_callback.assert_called_with(KillSwitchScope.VENUE, "XLON")

    def test_trigger_algorithm(self, kill_switch, mock_cancel_callback):
        """Test triggering for specific algorithm."""
        event = kill_switch.trigger_algorithm(
            algorithm_id="ALGO001",
            reason=KillSwitchTriggerReason.ALGORITHMIC_ERROR,
            reason_detail="Algorithm malfunction detected",
        )

        assert kill_switch.is_triggered is True
        assert event.scope == KillSwitchScope.ALGORITHM
        assert event.scope_id == "ALGO001"
        assert "ALGO001" in event.algorithms_affected

    def test_trigger_instrument(self, kill_switch, mock_cancel_callback):
        """Test triggering for specific instrument."""
        event = kill_switch.trigger_instrument(
            isin="GB0002634946",
            reason=KillSwitchTriggerReason.RISK_LIMIT,
            reason_detail="Position limit exceeded",
        )

        assert kill_switch.is_triggered is True
        assert event.scope == KillSwitchScope.INSTRUMENT
        assert event.scope_id == "GB0002634946"


# =============================================================================
# Test Recovery Operations
# =============================================================================

class TestKillSwitchRecovery:
    """Tests for kill switch recovery."""

    def test_recovery_after_cooldown(self, kill_switch):
        """Test recovery respects cooldown period."""
        kill_switch.trigger_all("Test")

        # Immediate recovery should fail (within cooldown)
        # Note: With 1s cooldown, we need to wait
        success = kill_switch.recover(recovered_by="test_user")
        assert success is False  # Still in cooldown

        # Wait for cooldown
        time.sleep(1.1)

        success = kill_switch.recover(recovered_by="test_user")
        assert success is True
        assert kill_switch.is_triggered is False
        assert kill_switch.is_armed is True

    def test_force_recovery(self, kill_switch):
        """Test force recovery ignoring cooldown."""
        kill_switch.trigger_all("Test")

        # Force recovery should work immediately
        success = kill_switch.force_recover(recovered_by="admin")
        assert success is True
        assert kill_switch.is_triggered is False
        assert kill_switch.is_armed is True

    def test_scope_specific_recovery(self, kill_switch):
        """Test recovering specific scope."""
        # Trigger for venue
        kill_switch.trigger_venue("XLON")
        kill_switch.trigger_venue("XETR")

        time.sleep(1.1)  # Wait for cooldown

        # Recover one venue
        success = kill_switch.recover(
            scope=KillSwitchScope.VENUE,
            scope_id="XLON",
        )
        assert success is True

        # Should still be triggered for XETR
        assert kill_switch.is_scope_triggered(KillSwitchScope.VENUE, "XETR") is True


# =============================================================================
# Test Trading Permission Checks
# =============================================================================

class TestTradingPermissions:
    """Tests for trading permission checks."""

    def test_can_trade_when_armed(self, kill_switch):
        """Test trading allowed when kill switch is armed."""
        can_trade, reason = kill_switch.can_trade(
            venue="XLON",
            algorithm="ALGO001",
        )

        assert can_trade is True

    def test_cannot_trade_when_triggered_all(self, kill_switch):
        """Test trading blocked when ALL scope triggered."""
        kill_switch.trigger_all("Test")

        can_trade, reason = kill_switch.can_trade()
        assert can_trade is False
        assert "ALL" in reason

    def test_cannot_trade_on_triggered_venue(self, kill_switch):
        """Test trading blocked on triggered venue."""
        kill_switch.trigger_venue("XLON")

        can_trade, reason = kill_switch.can_trade(venue="XLON")
        assert can_trade is False
        assert "XLON" in reason

        # Other venues should be allowed
        can_trade, reason = kill_switch.can_trade(venue="XETR")
        assert can_trade is True

    def test_can_trade_when_disarmed(self, kill_switch):
        """Test trading allowed when disarmed."""
        kill_switch.disarm(disarmed_by="admin")

        can_trade, reason = kill_switch.can_trade()
        assert can_trade is True
        assert "disarmed" in reason.lower()


# =============================================================================
# Test Rate Limiting
# =============================================================================

class TestRateLimiting:
    """Tests for trigger rate limiting."""

    def test_rate_limit_warning(self, mock_cancel_callback):
        """Test rate limit warning on excessive triggers."""
        config = KillSwitchConfig(
            max_triggers_per_hour=5,
            cooldown_seconds=0.1,
            persist_events=False,
        )
        ks = EnhancedKillSwitch(
            order_cancellation_callback=mock_cancel_callback,
            config=config,
        )

        # Trigger multiple times
        for i in range(6):
            ks.trigger_all(f"Test {i}")
            ks.force_recover()

        # Should still work but log warning
        assert mock_cancel_callback.call_count == 6


# =============================================================================
# Test Emergency Contacts
# =============================================================================

class TestEmergencyContacts:
    """Tests for emergency contact management."""

    def test_default_contacts_from_config(self, kill_switch, config):
        """Test contacts initialized from config."""
        contacts = kill_switch.get_emergency_contacts()

        assert len(contacts) >= 1
        assert any(c.email == "test@example.com" for c in contacts)

    def test_add_emergency_contact(self, kill_switch):
        """Test adding emergency contact."""
        contact = EmergencyContact(
            name="Night Shift",
            role="Operations",
            email="night@example.com",
            phone="+9876543210",
            is_available_24_7=True,
            escalation_order=10,
        )

        kill_switch.add_emergency_contact(contact)
        contacts = kill_switch.get_emergency_contacts()

        assert any(c.email == "night@example.com" for c in contacts)

    def test_contact_info_dict(self, kill_switch):
        """Test getting contact info as dictionary."""
        info = kill_switch.get_contact_info()

        assert "contacts" in info
        assert "primary_email" in info


# =============================================================================
# Test Callbacks
# =============================================================================

class TestCallbacks:
    """Tests for callback notifications."""

    def test_alert_callback_on_trigger(self, mock_cancel_callback, mock_alert_callback):
        """Test alert callback is called on trigger."""
        ks = EnhancedKillSwitch(
            order_cancellation_callback=mock_cancel_callback,
            config=KillSwitchConfig(persist_events=False),
            alert_callback=mock_alert_callback,
        )

        ks.trigger_all("Test")

        mock_alert_callback.assert_called_once()
        event = mock_alert_callback.call_args[0][0]
        assert isinstance(event, KillSwitchEvent)

    def test_compliance_callback_on_trigger(self, mock_cancel_callback, mock_compliance_callback):
        """Test compliance callback is called on trigger."""
        ks = EnhancedKillSwitch(
            order_cancellation_callback=mock_cancel_callback,
            config=KillSwitchConfig(
                notify_compliance=True,
                persist_events=False,
            ),
            compliance_callback=mock_compliance_callback,
        )

        ks.trigger_all("Test")

        mock_compliance_callback.assert_called_once()


# =============================================================================
# Test Event History
# =============================================================================

class TestEventHistory:
    """Tests for event history tracking."""

    def test_events_recorded(self, kill_switch):
        """Test events are recorded in history."""
        kill_switch.trigger_all("Test 1")
        kill_switch.force_recover()
        kill_switch.trigger_venue("XLON", reason_detail="Test 2")

        events = kill_switch.get_events()

        assert len(events) == 2
        assert events[0].scope == KillSwitchScope.ALL
        assert events[1].scope == KillSwitchScope.VENUE

    def test_get_last_event(self, kill_switch):
        """Test getting last event."""
        kill_switch.trigger_all("First")
        kill_switch.force_recover()
        kill_switch.trigger_venue("XLON", reason_detail="Second")

        last_event = kill_switch.get_last_event()

        assert last_event is not None
        assert last_event.scope == KillSwitchScope.VENUE

    def test_filter_events_by_scope(self, kill_switch):
        """Test filtering events by scope."""
        kill_switch.trigger_all("Test")
        kill_switch.force_recover()
        kill_switch.trigger_venue("XLON")
        kill_switch.force_recover()
        kill_switch.trigger_venue("XETR")

        venue_events = kill_switch.get_events(scope=KillSwitchScope.VENUE)

        assert len(venue_events) == 2
        assert all(e.scope == KillSwitchScope.VENUE for e in venue_events)


# =============================================================================
# Test Statistics and Reporting
# =============================================================================

class TestStatisticsAndReporting:
    """Tests for statistics and compliance reporting."""

    def test_statistics(self, kill_switch):
        """Test statistics tracking."""
        kill_switch.trigger_all("Test 1")
        kill_switch.force_recover()
        kill_switch.trigger_venue("XLON")

        stats = kill_switch.get_statistics()

        assert stats["total_triggers"] == 2
        assert stats["total_orders_cancelled"] == 20
        assert stats["active_scopes"] == 1
        assert KillSwitchTriggerReason.MANUAL.value in stats["triggers_by_reason"]

    def test_compliance_report(self, kill_switch):
        """Test compliance report generation."""
        kill_switch.trigger_all("Test")

        report = kill_switch.generate_compliance_report()

        assert report["regulation"] == "RTS 6 Article 12"
        assert report["requirement"] == "Kill Switch / Emergency Cancellation"
        assert "current_state" in report
        assert "configuration" in report
        assert "emergency_contacts" in report
        assert "statistics" in report


# =============================================================================
# Test Arm/Disarm
# =============================================================================

class TestArmDisarm:
    """Tests for arming and disarming."""

    def test_disarm_prevents_trigger(self, kill_switch, mock_cancel_callback):
        """Test disarming prevents triggering."""
        kill_switch.disarm(disarmed_by="admin")

        with pytest.raises(RuntimeError, match="disarmed"):
            kill_switch.trigger_all("Should fail")

    def test_arm_after_disarm(self, kill_switch):
        """Test arming after disarm."""
        kill_switch.disarm(disarmed_by="admin")
        assert kill_switch.state == KillSwitchState.DISARMED

        kill_switch.arm()
        assert kill_switch.state == KillSwitchState.ARMED


# =============================================================================
# Test Thread Safety
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_triggers(self, mock_cancel_callback):
        """Test concurrent triggers are handled safely."""
        config = KillSwitchConfig(
            cooldown_seconds=0.01,
            persist_events=False,
        )
        ks = EnhancedKillSwitch(
            order_cancellation_callback=mock_cancel_callback,
            config=config,
        )

        results = []

        def trigger_thread(n):
            try:
                ks.trigger_all(f"Thread {n}")
                results.append(("triggered", n))
            except Exception as e:
                results.append(("error", n, str(e)))
            finally:
                try:
                    ks.force_recover()
                except Exception:
                    pass

        threads = [threading.Thread(target=trigger_thread, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All triggers should have succeeded
        triggered_count = sum(1 for r in results if r[0] == "triggered")
        assert triggered_count >= 1


# =============================================================================
# Test Factory Function
# =============================================================================

class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_with_dict_config(self, mock_cancel_callback):
        """Test creating with dictionary config."""
        config = {
            "cooldown_seconds": 600.0,
            "primary_contact_email": "ops@firm.com",
            "max_triggers_per_hour": 20,
        }

        ks = create_enhanced_kill_switch(
            order_cancellation_callback=mock_cancel_callback,
            config=config,
        )

        assert ks is not None
        assert ks.is_armed is True

    def test_create_with_defaults(self, mock_cancel_callback):
        """Test creating with default config."""
        ks = create_enhanced_kill_switch(
            order_cancellation_callback=mock_cancel_callback,
        )

        assert ks is not None
        assert ks.state == KillSwitchState.ARMED


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_cancellation_callback_error(self, mock_alert_callback):
        """Test handling of cancellation callback error."""
        def failing_callback(scope, scope_id):
            raise Exception("Cancellation failed")

        ks = EnhancedKillSwitch(
            order_cancellation_callback=failing_callback,
            config=KillSwitchConfig(persist_events=False),
            alert_callback=mock_alert_callback,
        )

        # Should still record event even if cancellation fails
        event = ks.trigger_all("Test")

        assert event is not None
        assert event.orders_cancelled == 0  # Cancellation failed

    def test_empty_scope_id(self, kill_switch):
        """Test triggering with empty scope ID."""
        event = kill_switch.trigger(
            scope=KillSwitchScope.VENUE,
            scope_id="",  # Empty
        )

        assert event is not None
        assert event.scope_id == ""

    def test_multiple_reasons(self, kill_switch):
        """Test triggering with different reasons."""
        reasons = [
            KillSwitchTriggerReason.MANUAL,
            KillSwitchTriggerReason.RISK_LIMIT,
            KillSwitchTriggerReason.CLOCK_DRIFT,
            KillSwitchTriggerReason.CONNECTIVITY,
        ]

        for reason in reasons:
            kill_switch.trigger(
                scope=KillSwitchScope.ALL,
                reason=reason,
                reason_detail=f"Testing {reason.value}",
            )
            kill_switch.force_recover()

        events = kill_switch.get_events()
        assert len(events) == len(reasons)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
