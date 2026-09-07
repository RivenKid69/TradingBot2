# -*- coding: utf-8 -*-
"""
tests/test_futures_margin_monitor.py
Comprehensive tests for futures margin monitoring service.

Phase 9: Unified Live Trading (2025-12-02)

Tests cover:
- Margin monitor configuration
- Margin level classification and transitions
- Alert management with cooldowns
- Background monitoring
- Reduction suggestions
- Statistics computation
- Factory functions
- Edge cases and error handling
- Thread safety
"""

import asyncio
import pytest
import statistics
import threading
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

from core_futures import (
    FuturesPosition,
    FuturesType,
    MarginMode,
    PositionSide,
)

from services.futures_margin_monitor import (
    # Constants
    DEFAULT_WARNING_RATIO,
    DEFAULT_DANGER_RATIO,
    DEFAULT_CRITICAL_RATIO,
    DEFAULT_LIQUIDATION_RATIO,
    DEFAULT_MONITOR_INTERVAL_SEC,
    DEFAULT_ALERT_COOLDOWN_SEC,
    # Enums
    MarginLevel,
    MarginAlertType,
    # Data classes
    MarginSnapshot,
    MarginAlert,
    ReductionSuggestion,
    MarginMonitorConfig,
    # Classes
    MarginLevelTracker,
    MarginAlertManager,
    FuturesMarginMonitor,
    # Factory functions
    create_margin_monitor,
    create_crypto_margin_monitor,
    create_cme_margin_monitor,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_margin_provider():
    """Create a mock margin data provider."""
    provider = MagicMock()
    provider.get_account_equity.return_value = Decimal("10000.00")
    provider.get_total_margin_used.return_value = Decimal("5000.00")
    provider.get_available_margin.return_value = Decimal("5000.00")
    provider.get_maintenance_margin.return_value = Decimal("2500.00")
    provider.get_positions.return_value = {
        "BTCUSDT": FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("1.0"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
            margin=Decimal("5000"),
            position_value=Decimal("50000"),
        )
    }
    return provider


@pytest.fixture
def default_config():
    """Create default margin monitor config."""
    return MarginMonitorConfig()


@pytest.fixture
def crypto_config():
    """Create crypto-specific config."""
    return MarginMonitorConfig.for_crypto()


@pytest.fixture
def cme_config():
    """Create CME-specific config."""
    return MarginMonitorConfig.for_cme()


@pytest.fixture
def margin_monitor(mock_margin_provider, default_config):
    """Create a margin monitor with mocked dependencies."""
    return FuturesMarginMonitor(default_config, mock_margin_provider)


@pytest.fixture
def level_tracker(default_config):
    """Create a margin level tracker."""
    return MarginLevelTracker(default_config)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestMarginMonitorConfig:
    """Tests for MarginMonitorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MarginMonitorConfig()
        assert config.warning_ratio == DEFAULT_WARNING_RATIO
        assert config.danger_ratio == DEFAULT_DANGER_RATIO
        assert config.critical_ratio == DEFAULT_CRITICAL_RATIO
        assert config.liquidation_ratio == DEFAULT_LIQUIDATION_RATIO
        assert config.monitor_interval_sec == DEFAULT_MONITOR_INTERVAL_SEC
        assert config.alert_cooldown_sec == DEFAULT_ALERT_COOLDOWN_SEC
        assert config.enable_alerts is True
        assert config.enable_auto_reduction_suggestions is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = MarginMonitorConfig(
            warning_ratio=Decimal("1.8"),
            danger_ratio=Decimal("1.4"),
            monitor_interval_sec=10.0,
        )
        assert config.warning_ratio == Decimal("1.8")
        assert config.danger_ratio == Decimal("1.4")
        assert config.monitor_interval_sec == 10.0

    def test_crypto_config_factory(self):
        """Test crypto-specific config factory."""
        config = MarginMonitorConfig.for_crypto()
        assert config.futures_type == FuturesType.CRYPTO_PERPETUAL
        assert config.warning_ratio == DEFAULT_WARNING_RATIO

    def test_cme_config_factory(self):
        """Test CME-specific config factory with stricter thresholds."""
        config = MarginMonitorConfig.for_cme()
        assert config.futures_type == FuturesType.INDEX_FUTURES
        # CME has stricter thresholds
        assert config.warning_ratio == Decimal("1.3")
        assert config.danger_ratio == Decimal("1.15")
        assert config.critical_ratio == Decimal("1.05")

    def test_from_dict(self):
        """Test config creation from dictionary."""
        d = {
            "warning_ratio": "1.6",
            "danger_ratio": "1.3",
            "monitor_interval_sec": 15.0,
            "futures_type": "INDEX_FUTURES",
        }
        config = MarginMonitorConfig.from_dict(d)
        assert config.warning_ratio == Decimal("1.6")
        assert config.danger_ratio == Decimal("1.3")
        assert config.monitor_interval_sec == 15.0
        assert config.futures_type == FuturesType.INDEX_FUTURES

    def test_from_dict_with_defaults(self):
        """Test config creation with partial dictionary."""
        d = {"warning_ratio": "1.8"}
        config = MarginMonitorConfig.from_dict(d)
        assert config.warning_ratio == Decimal("1.8")
        assert config.danger_ratio == DEFAULT_DANGER_RATIO  # Default


# =============================================================================
# Margin Level Enum Tests
# =============================================================================


class TestMarginLevel:
    """Tests for MarginLevel enum."""

    def test_all_levels(self):
        """Test all margin levels exist."""
        assert MarginLevel.HEALTHY is not None
        assert MarginLevel.WARNING is not None
        assert MarginLevel.DANGER is not None
        assert MarginLevel.CRITICAL is not None
        assert MarginLevel.LIQUIDATION is not None

    def test_level_ordering(self):
        """Test levels have different values."""
        levels = [
            MarginLevel.HEALTHY,
            MarginLevel.WARNING,
            MarginLevel.DANGER,
            MarginLevel.CRITICAL,
            MarginLevel.LIQUIDATION,
        ]
        # Each level should be unique
        assert len(set(levels)) == 5


class TestMarginAlertType:
    """Tests for MarginAlertType enum."""

    def test_all_alert_types(self):
        """Test all alert types exist."""
        assert MarginAlertType.LEVEL_CHANGE is not None
        assert MarginAlertType.APPROACHING_WARNING is not None
        assert MarginAlertType.APPROACHING_DANGER is not None
        assert MarginAlertType.APPROACHING_CRITICAL is not None
        assert MarginAlertType.MARGIN_CALL is not None
        assert MarginAlertType.LIQUIDATION_RISK is not None
        assert MarginAlertType.RECOVERED is not None


# =============================================================================
# Margin Snapshot Tests
# =============================================================================


class TestMarginSnapshot:
    """Tests for MarginSnapshot dataclass."""

    def test_basic_snapshot(self):
        """Test basic snapshot creation."""
        snapshot = MarginSnapshot(
            timestamp_ms=int(time.time() * 1000),
            equity=Decimal("10000"),
            margin_used=Decimal("5000"),
            available_margin=Decimal("5000"),
            maintenance_margin=Decimal("2500"),
            margin_ratio=Decimal("2.0"),
            level=MarginLevel.HEALTHY,
            position_count=1,
            total_position_value=Decimal("50000"),
        )
        assert snapshot.equity == Decimal("10000")
        assert snapshot.margin_ratio == Decimal("2.0")
        assert snapshot.level == MarginLevel.HEALTHY

    def test_utilization_pct(self):
        """Test margin utilization percentage calculation."""
        snapshot = MarginSnapshot(
            timestamp_ms=0,
            equity=Decimal("10000"),
            margin_used=Decimal("5000"),
            available_margin=Decimal("5000"),
            maintenance_margin=Decimal("2500"),
            margin_ratio=Decimal("2.0"),
            level=MarginLevel.HEALTHY,
            position_count=1,
            total_position_value=Decimal("50000"),
        )
        assert snapshot.utilization_pct == Decimal("50")  # 5000/10000 * 100

    def test_utilization_pct_zero_equity(self):
        """Test utilization with zero equity."""
        snapshot = MarginSnapshot(
            timestamp_ms=0,
            equity=Decimal("0"),
            margin_used=Decimal("1000"),
            available_margin=Decimal("0"),
            maintenance_margin=Decimal("500"),
            margin_ratio=Decimal("0"),
            level=MarginLevel.LIQUIDATION,
            position_count=1,
            total_position_value=Decimal("10000"),
        )
        assert snapshot.utilization_pct == Decimal("0")

    def test_buffer_to_liquidation_pct(self):
        """Test buffer to liquidation calculation."""
        snapshot = MarginSnapshot(
            timestamp_ms=0,
            equity=Decimal("10000"),
            margin_used=Decimal("5000"),
            available_margin=Decimal("5000"),
            maintenance_margin=Decimal("2500"),
            margin_ratio=Decimal("2.0"),
            level=MarginLevel.HEALTHY,
            position_count=1,
            total_position_value=Decimal("50000"),
        )
        # margin_ratio - 1 = 1.0 = 100%
        assert snapshot.buffer_to_liquidation_pct == Decimal("100")

    def test_to_dict(self):
        """Test snapshot serialization."""
        snapshot = MarginSnapshot(
            timestamp_ms=1000000,
            equity=Decimal("10000"),
            margin_used=Decimal("5000"),
            available_margin=Decimal("5000"),
            maintenance_margin=Decimal("2500"),
            margin_ratio=Decimal("2.0"),
            level=MarginLevel.HEALTHY,
            position_count=1,
            total_position_value=Decimal("50000"),
        )
        d = snapshot.to_dict()
        assert d["equity"] == 10000.0
        assert d["margin_ratio"] == 2.0
        assert d["level"] == "HEALTHY"
        assert d["utilization_pct"] == 50.0


# =============================================================================
# Margin Alert Tests
# =============================================================================


class TestMarginAlert:
    """Tests for MarginAlert dataclass."""

    def test_basic_alert(self):
        """Test basic alert creation."""
        alert = MarginAlert(
            timestamp_ms=int(time.time() * 1000),
            alert_type=MarginAlertType.MARGIN_CALL,
            level=MarginLevel.CRITICAL,
            previous_level=MarginLevel.DANGER,
            margin_ratio=Decimal("1.08"),
            equity=Decimal("5400"),
            margin_used=Decimal("5000"),
            message="MARGIN CALL: Margin ratio 1.08",
            recommended_action="Reduce positions",
            severity="ERROR",
        )
        assert alert.alert_type == MarginAlertType.MARGIN_CALL
        assert alert.level == MarginLevel.CRITICAL
        assert alert.severity == "ERROR"

    def test_alert_to_dict(self):
        """Test alert serialization."""
        alert = MarginAlert(
            timestamp_ms=1000000,
            alert_type=MarginAlertType.LEVEL_CHANGE,
            level=MarginLevel.WARNING,
            previous_level=MarginLevel.HEALTHY,
            margin_ratio=Decimal("1.4"),
            equity=Decimal("7000"),
            margin_used=Decimal("5000"),
            message="Warning",
        )
        d = alert.to_dict()
        assert d["alert_type"] == "LEVEL_CHANGE"
        assert d["level"] == "WARNING"
        assert d["previous_level"] == "HEALTHY"
        assert d["margin_ratio"] == 1.4

    def test_alert_without_previous_level(self):
        """Test alert with no previous level."""
        alert = MarginAlert(
            timestamp_ms=1000000,
            alert_type=MarginAlertType.LEVEL_CHANGE,
            level=MarginLevel.WARNING,
            previous_level=None,
            margin_ratio=Decimal("1.4"),
            equity=Decimal("7000"),
            margin_used=Decimal("5000"),
            message="Level change",
        )
        d = alert.to_dict()
        assert d["previous_level"] is None


# =============================================================================
# Reduction Suggestion Tests
# =============================================================================


class TestReductionSuggestion:
    """Tests for ReductionSuggestion dataclass."""

    def test_basic_suggestion(self):
        """Test basic suggestion creation."""
        suggestion = ReductionSuggestion(
            symbol="BTCUSDT",
            current_qty=Decimal("1.0"),
            suggested_reduction_qty=Decimal("0.3"),
            suggested_reduction_pct=Decimal("30"),
            reason="Approaching margin call",
            priority=1,
        )
        assert suggestion.symbol == "BTCUSDT"
        assert suggestion.suggested_reduction_pct == Decimal("30")
        assert suggestion.priority == 1

    def test_suggestion_to_dict(self):
        """Test suggestion serialization."""
        suggestion = ReductionSuggestion(
            symbol="ETHUSDT",
            current_qty=Decimal("5.0"),
            suggested_reduction_qty=Decimal("2.5"),
            suggested_reduction_pct=Decimal("50"),
            reason="Margin management",
            priority=2,
        )
        d = suggestion.to_dict()
        assert d["symbol"] == "ETHUSDT"
        assert d["current_qty"] == 5.0
        assert d["suggested_reduction_qty"] == 2.5


# =============================================================================
# Margin Level Tracker Tests
# =============================================================================


class TestMarginLevelTracker:
    """Tests for MarginLevelTracker class."""

    def test_classify_healthy(self, level_tracker):
        """Test classification of healthy margin."""
        level = level_tracker.classify_level(Decimal("2.0"))
        assert level == MarginLevel.HEALTHY

    def test_classify_warning(self, level_tracker):
        """Test classification of warning margin."""
        level = level_tracker.classify_level(Decimal("1.4"))
        assert level == MarginLevel.WARNING

    def test_classify_danger(self, level_tracker):
        """Test classification of danger margin."""
        level = level_tracker.classify_level(Decimal("1.15"))
        assert level == MarginLevel.DANGER

    def test_classify_critical(self, level_tracker):
        """Test classification of critical margin."""
        level = level_tracker.classify_level(Decimal("1.05"))
        assert level == MarginLevel.CRITICAL

    def test_classify_liquidation(self, level_tracker):
        """Test classification of liquidation margin."""
        level = level_tracker.classify_level(Decimal("1.0"))
        assert level == MarginLevel.LIQUIDATION

    def test_classify_below_liquidation(self, level_tracker):
        """Test classification below liquidation threshold."""
        level = level_tracker.classify_level(Decimal("0.9"))
        assert level == MarginLevel.LIQUIDATION

    def test_update_level_no_change(self, level_tracker):
        """Test update with no level change."""
        level_tracker.update_level(Decimal("2.0"))  # Establish HEALTHY
        level, changed = level_tracker.update_level(Decimal("1.8"))  # Still HEALTHY
        assert level == MarginLevel.HEALTHY
        assert changed is False

    def test_update_level_with_change(self, level_tracker):
        """Test update with level change."""
        level_tracker.update_level(Decimal("2.0"))  # HEALTHY
        level, changed = level_tracker.update_level(Decimal("1.4"))  # -> WARNING
        assert level == MarginLevel.WARNING
        assert changed is True

    def test_get_threshold_for_level(self, level_tracker):
        """Test getting threshold for level."""
        assert level_tracker.get_threshold_for_level(MarginLevel.HEALTHY) == DEFAULT_WARNING_RATIO
        assert level_tracker.get_threshold_for_level(MarginLevel.WARNING) == DEFAULT_DANGER_RATIO
        assert level_tracker.get_threshold_for_level(MarginLevel.DANGER) == DEFAULT_CRITICAL_RATIO
        assert (
            level_tracker.get_threshold_for_level(MarginLevel.CRITICAL) == DEFAULT_LIQUIDATION_RATIO
        )

    def test_get_buffer_to_next_level(self, level_tracker):
        """Test getting buffer to next worse level."""
        next_level, buffer = level_tracker.get_buffer_to_next_level(Decimal("1.6"))
        assert next_level == MarginLevel.WARNING
        assert buffer == Decimal("0.1")  # 1.6 - 1.5 (WARNING threshold)


# =============================================================================
# Alert Manager Tests
# =============================================================================


class TestMarginAlertManager:
    """Tests for MarginAlertManager class."""

    def test_should_alert_initial(self, default_config):
        """Test initial alert is allowed."""
        manager = MarginAlertManager(default_config)
        assert manager.should_alert(MarginAlertType.MARGIN_CALL) is True

    def test_should_alert_cooldown(self, default_config):
        """Test alert cooldown."""
        config = MarginMonitorConfig(alert_cooldown_sec=1.0)  # Short cooldown for test
        manager = MarginAlertManager(config)

        alert = MarginAlert(
            timestamp_ms=int(time.time() * 1000),
            alert_type=MarginAlertType.MARGIN_CALL,
            level=MarginLevel.CRITICAL,
            previous_level=MarginLevel.DANGER,
            margin_ratio=Decimal("1.08"),
            equity=Decimal("5400"),
            margin_used=Decimal("5000"),
            message="Test",
        )

        # First alert goes through
        manager.send_alert(alert)
        # Second should be blocked
        assert manager.should_alert(MarginAlertType.MARGIN_CALL) is False

        # Wait for cooldown
        time.sleep(1.1)
        assert manager.should_alert(MarginAlertType.MARGIN_CALL) is True

    def test_send_alert_to_callbacks(self, default_config):
        """Test alert is sent to callbacks."""
        callback_calls = []

        def my_callback(alert_type, level, data):
            callback_calls.append((alert_type, level, data))

        manager = MarginAlertManager(default_config, callbacks=[my_callback])

        alert = MarginAlert(
            timestamp_ms=int(time.time() * 1000),
            alert_type=MarginAlertType.LEVEL_CHANGE,
            level=MarginLevel.WARNING,
            previous_level=None,
            margin_ratio=Decimal("1.4"),
            equity=Decimal("7000"),
            margin_used=Decimal("5000"),
            message="Test",
        )

        manager.send_alert(alert)

        assert len(callback_calls) == 1
        assert callback_calls[0][0] == MarginAlertType.LEVEL_CHANGE
        assert callback_calls[0][1] == MarginLevel.WARNING

    def test_add_callback(self, default_config):
        """Test adding callback after initialization."""
        manager = MarginAlertManager(default_config)
        callback_calls = []

        manager.add_callback(lambda t, l, d: callback_calls.append(t))

        alert = MarginAlert(
            timestamp_ms=int(time.time() * 1000),
            alert_type=MarginAlertType.RECOVERED,
            level=MarginLevel.HEALTHY,
            previous_level=MarginLevel.WARNING,
            margin_ratio=Decimal("1.6"),
            equity=Decimal("8000"),
            margin_used=Decimal("5000"),
            message="Recovered",
        )

        manager.send_alert(alert)
        assert len(callback_calls) == 1

    def test_get_alert_history(self, default_config):
        """Test alert history retrieval."""
        manager = MarginAlertManager(default_config)

        for i in range(5):
            alert = MarginAlert(
                timestamp_ms=int(time.time() * 1000) + i,
                alert_type=MarginAlertType.LEVEL_CHANGE,
                level=MarginLevel.WARNING,
                previous_level=None,
                margin_ratio=Decimal("1.4"),
                equity=Decimal("7000"),
                margin_used=Decimal("5000"),
                message=f"Alert {i}",
            )
            manager._alert_history.append(alert)

        history = manager.get_alert_history(limit=3)
        assert len(history) == 3

    def test_clear_cooldowns(self, default_config):
        """Test clearing cooldowns."""
        config = MarginMonitorConfig(alert_cooldown_sec=3600.0)  # Long cooldown
        manager = MarginAlertManager(config)

        # Send alert
        alert = MarginAlert(
            timestamp_ms=int(time.time() * 1000),
            alert_type=MarginAlertType.MARGIN_CALL,
            level=MarginLevel.CRITICAL,
            previous_level=None,
            margin_ratio=Decimal("1.08"),
            equity=Decimal("5400"),
            margin_used=Decimal("5000"),
            message="Test",
        )
        manager.send_alert(alert)

        # Blocked by cooldown
        assert manager.should_alert(MarginAlertType.MARGIN_CALL) is False

        # Clear cooldowns
        manager.clear_cooldowns()

        # Now allowed
        assert manager.should_alert(MarginAlertType.MARGIN_CALL) is True

    def test_disabled_alerts(self):
        """Test alerts disabled in config."""
        config = MarginMonitorConfig(enable_alerts=False)
        manager = MarginAlertManager(config)
        callback_calls = []

        manager.add_callback(lambda t, l, d: callback_calls.append(t))

        alert = MarginAlert(
            timestamp_ms=int(time.time() * 1000),
            alert_type=MarginAlertType.MARGIN_CALL,
            level=MarginLevel.CRITICAL,
            previous_level=None,
            margin_ratio=Decimal("1.08"),
            equity=Decimal("5400"),
            margin_used=Decimal("5000"),
            message="Test",
        )

        manager.send_alert(alert)

        # Callback should not be called
        assert len(callback_calls) == 0


# =============================================================================
# Futures Margin Monitor Tests
# =============================================================================


class TestFuturesMarginMonitor:
    """Tests for FuturesMarginMonitor class."""

    def test_initialization(self, margin_monitor):
        """Test monitor initialization."""
        assert margin_monitor.get_current_snapshot() is None
        assert margin_monitor.get_current_level() == MarginLevel.HEALTHY
        assert margin_monitor._monitoring is False

    def test_take_snapshot(self, margin_monitor):
        """Test taking margin snapshot."""
        snapshot = margin_monitor.take_snapshot()

        assert snapshot.equity == Decimal("10000")
        assert snapshot.margin_used == Decimal("5000")
        assert snapshot.margin_ratio == Decimal("2.0")  # 10000/5000
        assert snapshot.level == MarginLevel.HEALTHY

    def test_take_snapshot_no_margin(self, default_config):
        """Test snapshot with no margin used."""
        provider = MagicMock()
        provider.get_account_equity.return_value = Decimal("10000")
        provider.get_total_margin_used.return_value = Decimal("0")
        provider.get_available_margin.return_value = Decimal("10000")
        provider.get_maintenance_margin.return_value = Decimal("0")
        provider.get_positions.return_value = {}

        monitor = FuturesMarginMonitor(default_config, provider)
        snapshot = monitor.take_snapshot()

        assert snapshot.margin_ratio == Decimal("999")  # No margin = very healthy
        assert snapshot.level == MarginLevel.HEALTHY

    def test_check_once(self, margin_monitor):
        """Test single monitoring check."""
        snapshot, alert = margin_monitor.check_once()

        assert snapshot is not None
        assert snapshot.level == MarginLevel.HEALTHY
        # No alert for staying healthy
        assert alert is None

    def test_check_once_with_level_change(self, default_config):
        """Test check with level change triggers alert."""
        # Start healthy
        provider = MagicMock()
        provider.get_account_equity.return_value = Decimal("10000")
        provider.get_total_margin_used.return_value = Decimal("5000")
        provider.get_available_margin.return_value = Decimal("5000")
        provider.get_maintenance_margin.return_value = Decimal("2500")
        provider.get_positions.return_value = {}

        monitor = FuturesMarginMonitor(default_config, provider)
        monitor.check_once()  # Establish HEALTHY

        # Now drop to WARNING
        provider.get_account_equity.return_value = Decimal("7000")  # 7000/5000 = 1.4 = WARNING
        snapshot, alert = monitor.check_once()

        assert snapshot.level == MarginLevel.WARNING
        assert alert is not None
        assert alert.level == MarginLevel.WARNING

    def test_check_once_critical_level(self, default_config):
        """Test check at critical level."""
        provider = MagicMock()
        provider.get_account_equity.return_value = Decimal("5400")  # 5400/5000 = 1.08 = CRITICAL
        provider.get_total_margin_used.return_value = Decimal("5000")
        provider.get_available_margin.return_value = Decimal("400")
        provider.get_maintenance_margin.return_value = Decimal("2500")
        provider.get_positions.return_value = {}

        monitor = FuturesMarginMonitor(default_config, provider)
        monitor._previous_level = MarginLevel.DANGER  # Simulate coming from DANGER

        snapshot, alert = monitor.check_once()

        assert snapshot.level == MarginLevel.CRITICAL
        assert alert is not None
        assert alert.alert_type == MarginAlertType.MARGIN_CALL
        assert alert.severity == "ERROR"

    def test_check_once_liquidation_level(self, default_config):
        """Test check at liquidation level."""
        provider = MagicMock()
        provider.get_account_equity.return_value = Decimal("5000")  # 5000/5000 = 1.0 = LIQUIDATION
        provider.get_total_margin_used.return_value = Decimal("5000")
        provider.get_available_margin.return_value = Decimal("0")
        provider.get_maintenance_margin.return_value = Decimal("2500")
        provider.get_positions.return_value = {}

        monitor = FuturesMarginMonitor(default_config, provider)
        monitor._previous_level = MarginLevel.CRITICAL

        snapshot, alert = monitor.check_once()

        assert snapshot.level == MarginLevel.LIQUIDATION
        assert alert is not None
        assert alert.alert_type == MarginAlertType.LIQUIDATION_RISK
        assert alert.severity == "CRITICAL"

    def test_check_once_recovery(self, default_config):
        """Test check showing recovery."""
        provider = MagicMock()
        provider.get_total_margin_used.return_value = Decimal("5000")
        provider.get_available_margin.return_value = Decimal("5000")
        provider.get_maintenance_margin.return_value = Decimal("2500")
        provider.get_positions.return_value = {}

        monitor = FuturesMarginMonitor(default_config, provider)

        # Start at WARNING
        provider.get_account_equity.return_value = Decimal("7000")
        monitor.check_once()
        assert monitor.get_current_level() == MarginLevel.WARNING

        # Recover to HEALTHY
        provider.get_account_equity.return_value = Decimal("10000")
        snapshot, alert = monitor.check_once()

        assert snapshot.level == MarginLevel.HEALTHY
        assert alert is not None
        assert alert.alert_type == MarginAlertType.RECOVERED

    def test_get_current_snapshot(self, margin_monitor):
        """Test getting current snapshot."""
        assert margin_monitor.get_current_snapshot() is None

        margin_monitor.check_once()

        snapshot = margin_monitor.get_current_snapshot()
        assert snapshot is not None
        assert snapshot.equity == Decimal("10000")

    def test_get_current_level(self, margin_monitor):
        """Test getting current level."""
        assert margin_monitor.get_current_level() == MarginLevel.HEALTHY

        margin_monitor.check_once()
        assert margin_monitor.get_current_level() == MarginLevel.HEALTHY

    def test_get_history(self, margin_monitor):
        """Test getting snapshot history."""
        # Multiple checks
        for _ in range(5):
            margin_monitor.check_once()

        history = margin_monitor.get_history()
        assert len(history) == 5

        limited_history = margin_monitor.get_history(limit=3)
        assert len(limited_history) == 3

    def test_get_margin_ratio(self, margin_monitor):
        """Test getting margin ratio."""
        assert margin_monitor.get_margin_ratio() is None

        margin_monitor.check_once()

        ratio = margin_monitor.get_margin_ratio()
        assert ratio == Decimal("2.0")

    def test_is_healthy(self, margin_monitor):
        """Test is_healthy check."""
        margin_monitor.check_once()
        assert margin_monitor.is_healthy() is True

    def test_is_at_risk(self, default_config):
        """Test is_at_risk check."""
        provider = MagicMock()
        provider.get_account_equity.return_value = Decimal("5800")  # 5800/5000 = 1.16 = DANGER
        provider.get_total_margin_used.return_value = Decimal("5000")
        provider.get_available_margin.return_value = Decimal("800")
        provider.get_maintenance_margin.return_value = Decimal("2500")
        provider.get_positions.return_value = {}

        monitor = FuturesMarginMonitor(default_config, provider)
        monitor.check_once()

        assert monitor.is_at_risk() is True

    def test_register_alert_callback(self, margin_monitor):
        """Test registering alert callback."""
        callback_calls = []

        def my_callback(alert_type, level, data):
            callback_calls.append((alert_type, level))

        margin_monitor.register_alert_callback(my_callback)

        # Trigger level change
        margin_monitor._margin_provider.get_account_equity.return_value = Decimal("7000")
        margin_monitor.check_once()  # To WARNING
        margin_monitor._margin_provider.get_account_equity.return_value = Decimal("10000")
        margin_monitor.check_once()  # Back to HEALTHY

        assert len(callback_calls) >= 1


# =============================================================================
# Background Monitoring Tests
# =============================================================================


class TestBackgroundMonitoring:
    """Tests for background monitoring functionality."""

    def test_start_background_monitoring(self, margin_monitor):
        """Test starting background monitoring."""
        margin_monitor.start_background_monitoring()

        assert margin_monitor._monitoring is True

        # Cleanup
        margin_monitor.stop_background_monitoring()
        assert margin_monitor._monitoring is False

    def test_stop_background_monitoring(self, margin_monitor):
        """Test stopping background monitoring."""
        margin_monitor.start_background_monitoring()
        margin_monitor.stop_background_monitoring()

        assert margin_monitor._monitoring is False

    def test_background_monitoring_runs(self, default_config):
        """Test background monitoring executes checks."""
        provider = MagicMock()
        provider.get_account_equity.return_value = Decimal("10000")
        provider.get_total_margin_used.return_value = Decimal("5000")
        provider.get_available_margin.return_value = Decimal("5000")
        provider.get_maintenance_margin.return_value = Decimal("2500")
        provider.get_positions.return_value = {}

        config = MarginMonitorConfig(monitor_interval_sec=0.5)  # Fast for test
        monitor = FuturesMarginMonitor(config, provider)

        monitor.start_background_monitoring()
        time.sleep(1.5)  # Wait for checks
        monitor.stop_background_monitoring()

        # Should have run multiple checks
        history = monitor.get_history()
        assert len(history) >= 2

    def test_double_start_warning(self, margin_monitor):
        """Test warning when starting twice."""
        margin_monitor.start_background_monitoring()
        # Second start should not crash
        margin_monitor.start_background_monitoring()

        assert margin_monitor._monitoring is True

        # Cleanup
        margin_monitor.stop_background_monitoring()


# =============================================================================
# Reduction Suggestions Tests
# =============================================================================


class TestReductionSuggestions:
    """Tests for position reduction suggestions."""

    def test_no_suggestions_healthy(self, margin_monitor):
        """Test no suggestions when healthy."""
        margin_monitor.check_once()

        positions = margin_monitor._margin_provider.get_positions()
        suggestions = margin_monitor.get_reduction_suggestions(positions)

        assert len(suggestions) == 0

    def test_suggestions_when_at_risk(self, default_config):
        """Test suggestions when margin at risk."""
        provider = MagicMock()
        provider.get_account_equity.return_value = Decimal("5800")  # DANGER level
        provider.get_total_margin_used.return_value = Decimal("5000")
        provider.get_available_margin.return_value = Decimal("800")
        provider.get_maintenance_margin.return_value = Decimal("2500")

        position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("1.0"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
            margin=Decimal("5000"),
            position_value=Decimal("50000"),
        )
        provider.get_positions.return_value = {"BTCUSDT": position}

        monitor = FuturesMarginMonitor(default_config, provider)
        monitor.check_once()

        suggestions = monitor.get_reduction_suggestions({"BTCUSDT": position})

        assert len(suggestions) > 0
        assert suggestions[0].symbol == "BTCUSDT"
        assert suggestions[0].priority == 1

    def test_suggestions_disabled(self, default_config):
        """Test suggestions disabled in config."""
        config = MarginMonitorConfig(enable_auto_reduction_suggestions=False)

        provider = MagicMock()
        provider.get_account_equity.return_value = Decimal("5800")
        provider.get_total_margin_used.return_value = Decimal("5000")
        provider.get_available_margin.return_value = Decimal("800")
        provider.get_maintenance_margin.return_value = Decimal("2500")
        provider.get_positions.return_value = {}

        monitor = FuturesMarginMonitor(config, provider)
        monitor.check_once()

        suggestions = monitor.get_reduction_suggestions({})
        assert len(suggestions) == 0


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for margin statistics computation."""

    def test_get_statistics(self, margin_monitor):
        """Test statistics computation."""
        # Generate some history
        for _ in range(10):
            margin_monitor.check_once()

        stats = margin_monitor.get_statistics()

        assert "snapshot_count" in stats
        assert "current_ratio" in stats
        assert "min_ratio" in stats
        assert "max_ratio" in stats
        assert "mean_ratio" in stats
        assert "level_distribution" in stats
        assert "time_at_risk_pct" in stats

        assert stats["snapshot_count"] == 10

    def test_get_statistics_empty(self, margin_monitor):
        """Test statistics with no history."""
        stats = margin_monitor.get_statistics()
        assert stats == {}


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for margin monitor factory functions."""

    def test_create_margin_monitor(self, mock_margin_provider):
        """Test basic factory function."""
        monitor = create_margin_monitor(mock_margin_provider)

        assert monitor is not None
        assert isinstance(monitor, FuturesMarginMonitor)

    def test_create_margin_monitor_with_config(self, mock_margin_provider):
        """Test factory with custom config."""
        config = MarginMonitorConfig(warning_ratio=Decimal("1.8"))
        monitor = create_margin_monitor(mock_margin_provider, config)

        assert monitor._config.warning_ratio == Decimal("1.8")

    def test_create_margin_monitor_with_callbacks(self, mock_margin_provider):
        """Test factory with callbacks."""
        callback_calls = []

        def my_callback(t, l, d):
            callback_calls.append(t)

        monitor = create_margin_monitor(mock_margin_provider, alert_callbacks=[my_callback])

        # Trigger alert
        mock_margin_provider.get_account_equity.return_value = Decimal("7000")
        monitor.check_once()  # HEALTHY -> WARNING
        mock_margin_provider.get_account_equity.return_value = Decimal("10000")
        monitor.check_once()  # Back

        assert len(callback_calls) >= 1

    def test_create_crypto_margin_monitor(self, mock_margin_provider):
        """Test crypto margin monitor factory."""
        monitor = create_crypto_margin_monitor(mock_margin_provider)

        assert monitor._config.futures_type == FuturesType.CRYPTO_PERPETUAL
        assert monitor._config.warning_ratio == DEFAULT_WARNING_RATIO

    def test_create_cme_margin_monitor(self, mock_margin_provider):
        """Test CME margin monitor factory."""
        monitor = create_cme_margin_monitor(mock_margin_provider)

        assert monitor._config.futures_type == FuturesType.INDEX_FUTURES
        # CME has stricter thresholds
        assert monitor._config.warning_ratio == Decimal("1.3")


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_snapshot_with_provider_error(self, default_config):
        """Test snapshot handles provider errors."""
        provider = MagicMock()
        provider.get_account_equity.side_effect = Exception("API Error")

        monitor = FuturesMarginMonitor(default_config, provider)
        snapshot = monitor.take_snapshot()

        # Should return safe default
        assert snapshot.equity == Decimal("0")
        assert snapshot.level == MarginLevel.LIQUIDATION

    def test_callback_error_handling(self, default_config, mock_margin_provider):
        """Test callback errors don't crash monitoring."""

        def bad_callback(t, l, d):
            raise Exception("Callback error")

        monitor = FuturesMarginMonitor(default_config, mock_margin_provider)
        monitor.register_alert_callback(bad_callback)

        # Trigger alert
        mock_margin_provider.get_account_equity.return_value = Decimal("7000")

        # Should not raise
        snapshot, alert = monitor.check_once()
        assert snapshot is not None


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_checks(self, mock_margin_provider, default_config):
        """Test concurrent check_once calls."""
        monitor = FuturesMarginMonitor(default_config, mock_margin_provider)
        results = []
        errors = []

        def check_worker():
            try:
                snapshot, _ = monitor.check_once()
                results.append(snapshot)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=check_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10

    def test_concurrent_snapshot_access(self, mock_margin_provider, default_config):
        """Test concurrent snapshot access."""
        monitor = FuturesMarginMonitor(default_config, mock_margin_provider)
        monitor.check_once()  # Initial snapshot

        errors = []

        def read_snapshot():
            try:
                for _ in range(100):
                    _ = monitor.get_current_snapshot()
                    _ = monitor.get_history()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_snapshot) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_high_margin_ratio(self, default_config):
        """Test handling very high margin ratio."""
        provider = MagicMock()
        provider.get_account_equity.return_value = Decimal("100000")
        provider.get_total_margin_used.return_value = Decimal("100")
        provider.get_available_margin.return_value = Decimal("99900")
        provider.get_maintenance_margin.return_value = Decimal("50")
        provider.get_positions.return_value = {}

        monitor = FuturesMarginMonitor(default_config, provider)
        snapshot = monitor.take_snapshot()

        assert snapshot.margin_ratio == Decimal("1000")  # 100000/100
        assert snapshot.level == MarginLevel.HEALTHY

    def test_zero_positions(self, default_config):
        """Test with no positions."""
        provider = MagicMock()
        provider.get_account_equity.return_value = Decimal("10000")
        provider.get_total_margin_used.return_value = Decimal("0")
        provider.get_available_margin.return_value = Decimal("10000")
        provider.get_maintenance_margin.return_value = Decimal("0")
        provider.get_positions.return_value = {}

        monitor = FuturesMarginMonitor(default_config, provider)
        snapshot = monitor.take_snapshot()

        assert snapshot.position_count == 0
        assert snapshot.total_position_value == Decimal("0")

    def test_negative_pnl_impact(self, default_config):
        """Test position value calculation with position values."""
        provider = MagicMock()
        provider.get_account_equity.return_value = Decimal("10000")
        provider.get_total_margin_used.return_value = Decimal("5000")
        provider.get_available_margin.return_value = Decimal("5000")
        provider.get_maintenance_margin.return_value = Decimal("2500")

        positions = {
            "BTCUSDT": FuturesPosition(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                entry_price=Decimal("50000"),
                qty=Decimal("1.0"),
                leverage=10,
                margin_mode=MarginMode.CROSS,
                margin=Decimal("2500"),
                position_value=Decimal("50000"),
            ),
            "ETHUSDT": FuturesPosition(
                symbol="ETHUSDT",
                side=PositionSide.SHORT,
                entry_price=Decimal("3000"),
                qty=Decimal("-5.0"),  # Short
                leverage=10,
                margin_mode=MarginMode.CROSS,
                margin=Decimal("1500"),
                position_value=Decimal("-15000"),  # Negative for short
            ),
        }
        provider.get_positions.return_value = positions

        monitor = FuturesMarginMonitor(default_config, provider)
        snapshot = monitor.take_snapshot()

        # Should sum absolute values
        assert snapshot.total_position_value == Decimal("65000")  # 50000 + 15000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
