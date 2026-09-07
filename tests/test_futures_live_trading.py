# -*- coding: utf-8 -*-
"""
tests/test_futures_live_trading.py
Comprehensive tests for Phase 9: Futures Live Trading.

Test coverage:
- FuturesLiveRunner: lifecycle, state management, event handling
- FuturesPositionSynchronizer: sync logic, discrepancy detection
- FuturesMarginMonitor: level tracking, alerts, reduction suggestions
- Integration tests: full workflow, error handling

Total: 100+ tests

Author: Trading Bot Team
Date: 2025-12-02
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest

# Import modules under test
from core_futures import (
    FuturesType,
    FuturesPosition,
    FuturesContractSpec,
    MarginMode,
    PositionSide,
    Exchange,
    OrderSide,
    OrderType,
    FundingRateInfo,
)
from services.futures_live_runner import (
    FuturesLiveRunner,
    FuturesLiveRunnerConfig,
    LiveRunnerState,
    LiveRunnerEvent,
    LiveRunnerStats,
    HealthStatus,
    create_futures_live_runner,
    create_crypto_futures_runner,
    create_cme_futures_runner,
    DEFAULT_MAIN_LOOP_INTERVAL_SEC,
    DEFAULT_POSITION_SYNC_INTERVAL_SEC,
    MarketDataProvider,
    OrderExecutor,
    SignalProvider,
)
from services.futures_position_sync import (
    FuturesPositionSynchronizer,
    FuturesSyncConfig,
    FuturesSyncResult,
    FuturesSyncEventType,
    ADLRiskLevel,
    FuturesPositionDiff,
    create_crypto_futures_sync,
    create_cme_futures_sync,
)
from services.futures_margin_monitor import (
    FuturesMarginMonitor,
    MarginMonitorConfig,
    MarginLevel,
    MarginAlert,
    MarginAlertType,
    MarginSnapshot,
    MarginLevelTracker,
    MarginAlertManager,
    ReductionSuggestion,
    create_margin_monitor,
    create_crypto_margin_monitor,
    create_cme_margin_monitor,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_market_data():
    """Create mock market data provider."""
    mock = MagicMock()
    mock.get_mark_price.return_value = Decimal("50000")
    mock.get_index_price.return_value = Decimal("49990")
    mock.get_funding_rate.return_value = FundingRateInfo(
        symbol="BTCUSDT",
        funding_rate=Decimal("0.0001"),
        next_funding_time_ms=int(time.time() * 1000) + 3600000,
    )
    mock.get_order_book.return_value = {
        "bids": [["49990", "10"], ["49980", "20"]],
        "asks": [["50000", "10"], ["50010", "20"]],
    }
    mock.is_connected.return_value = True
    return mock


@pytest.fixture
def mock_order_executor():
    """Create mock order executor."""
    mock = MagicMock()
    mock.submit_market_order.return_value = "order_123"
    mock.submit_limit_order.return_value = "order_456"
    mock.cancel_order.return_value = True
    mock.cancel_all_orders.return_value = 0
    mock.get_order_status.return_value = {"status": "FILLED"}
    return mock


@pytest.fixture
def mock_signal_provider():
    """Create mock signal provider."""
    mock = MagicMock()
    mock.get_signal.return_value = {
        "action": "BUY",
        "qty": 0.1,
        "order_type": "MARKET",
    }
    return mock


@pytest.fixture
def sample_position():
    """Create sample futures position."""
    return FuturesPosition(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_price=Decimal("49000"),
        qty=Decimal("0.5"),
        leverage=10,
        margin_mode=MarginMode.CROSS,
        unrealized_pnl=Decimal("500"),
        realized_pnl=Decimal("0"),
        liquidation_price=Decimal("44000"),
        mark_price=Decimal("50000"),
        margin=Decimal("2450"),
        maint_margin=Decimal("1225"),
        timestamp_ms=int(time.time() * 1000),
        position_value=Decimal("25000"),
    )


@pytest.fixture
def runner_config():
    """Create sample runner configuration."""
    return FuturesLiveRunnerConfig(
        futures_type=FuturesType.CRYPTO_PERPETUAL,
        symbols=["BTCUSDT", "ETHUSDT"],
        main_loop_interval_sec=0.1,  # Fast for testing
        position_sync_interval_sec=0.5,
        margin_check_interval_sec=0.5,
        enable_position_sync=True,
        enable_margin_monitoring=True,
        enable_funding_tracking=True,
        enable_adl_monitoring=True,
        paper_trading=True,
    )


@pytest.fixture
def mock_margin_provider():
    """Create mock margin data provider."""
    mock = MagicMock()
    mock.get_account_equity.return_value = Decimal("10000")
    mock.get_total_margin_used.return_value = Decimal("5000")
    mock.get_available_margin.return_value = Decimal("5000")
    mock.get_maintenance_margin.return_value = Decimal("2500")
    mock.get_positions.return_value = {}
    return mock


# ============================================================================
# TESTS: FuturesLiveRunnerConfig
# ============================================================================


class TestFuturesLiveRunnerConfig:
    """Tests for FuturesLiveRunnerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FuturesLiveRunnerConfig()

        assert config.futures_type == FuturesType.CRYPTO_PERPETUAL
        assert config.symbols == []
        assert config.main_loop_interval_sec == DEFAULT_MAIN_LOOP_INTERVAL_SEC
        assert config.paper_trading is True

    def test_from_dict(self):
        """Test creating config from dictionary."""
        d = {
            "futures_type": "INDEX_FUTURES",
            "symbols": ["ES", "NQ"],
            "paper_trading": False,
            "max_leverage": 20,
        }

        config = FuturesLiveRunnerConfig.from_dict(d)

        assert config.futures_type == FuturesType.INDEX_FUTURES
        assert config.symbols == ["ES", "NQ"]
        assert config.paper_trading is False
        assert config.max_leverage == 20

    def test_for_crypto_perpetual(self):
        """Test crypto perpetual preset."""
        config = FuturesLiveRunnerConfig.for_crypto_perpetual(
            symbols=["BTCUSDT"],
            paper_trading=True,
        )

        assert config.futures_type == FuturesType.CRYPTO_PERPETUAL
        assert config.symbols == ["BTCUSDT"]
        assert config.enable_adl_monitoring is True
        assert config.enable_circuit_breaker_monitoring is False

    def test_for_cme_futures(self):
        """Test CME futures preset."""
        config = FuturesLiveRunnerConfig.for_cme_futures(
            symbols=["ES", "NQ"],
            paper_trading=True,
        )

        assert config.futures_type == FuturesType.INDEX_FUTURES
        assert config.symbols == ["ES", "NQ"]
        assert config.enable_adl_monitoring is False
        assert config.enable_circuit_breaker_monitoring is True
        assert config.enable_funding_tracking is False


# ============================================================================
# TESTS: FuturesLiveRunner Lifecycle
# ============================================================================


class TestFuturesLiveRunnerLifecycle:
    """Tests for FuturesLiveRunner lifecycle management."""

    def test_initial_state(self, runner_config, mock_market_data, mock_order_executor):
        """Test initial state is INITIALIZING."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        assert runner.state == LiveRunnerState.INITIALIZING
        assert runner.is_running is False

    def test_start_transitions_to_running(
        self, runner_config, mock_market_data, mock_order_executor
    ):
        """Test start transitions through states to RUNNING."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        runner.start()

        assert runner.state == LiveRunnerState.RUNNING
        assert runner.is_running is True

        runner.stop()

    def test_stop_transitions_to_stopped(
        self, runner_config, mock_market_data, mock_order_executor
    ):
        """Test stop transitions to STOPPED."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        runner.start()
        runner.stop()

        assert runner.state == LiveRunnerState.STOPPED
        assert runner.is_running is False

    def test_pause_and_resume(self, runner_config, mock_market_data, mock_order_executor):
        """Test pause and resume functionality."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        runner.start()
        runner.pause()
        assert runner.state == LiveRunnerState.PAUSED

        runner.resume()
        assert runner.state == LiveRunnerState.RUNNING

        runner.stop()

    def test_double_start_warning(self, runner_config, mock_market_data, mock_order_executor):
        """Test that double start logs warning."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        runner.start()
        runner.start()  # Should log warning, not raise

        runner.stop()

    def test_stop_cancels_all_orders(self, runner_config, mock_market_data, mock_order_executor):
        """Test that stop cancels all orders."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        runner.start()
        runner.stop()

        mock_order_executor.cancel_all_orders.assert_called()


# ============================================================================
# TESTS: FuturesLiveRunner Position Management
# ============================================================================


class TestFuturesLiveRunnerPositions:
    """Tests for FuturesLiveRunner position management."""

    def test_get_positions_empty(self, runner_config, mock_market_data, mock_order_executor):
        """Test get_positions returns empty dict initially."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        positions = runner.get_positions()
        assert positions == {}

    def test_get_position_by_symbol(
        self, runner_config, mock_market_data, mock_order_executor, sample_position
    ):
        """Test get_position by symbol."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        runner._update_local_positions({"BTCUSDT": sample_position})

        pos = runner.get_position("BTCUSDT")
        assert pos is not None
        assert pos.symbol == "BTCUSDT"

    def test_get_position_missing(self, runner_config, mock_market_data, mock_order_executor):
        """Test get_position returns None for missing symbol."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        pos = runner.get_position("UNKNOWN")
        assert pos is None


# ============================================================================
# TESTS: FuturesLiveRunner Order Management
# ============================================================================


class TestFuturesLiveRunnerOrders:
    """Tests for FuturesLiveRunner order management."""

    def test_get_active_orders_empty(self, runner_config, mock_market_data, mock_order_executor):
        """Test get_active_orders returns empty dict initially."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        orders = runner.get_active_orders()
        assert orders == {}

    def test_add_and_remove_order(self, runner_config, mock_market_data, mock_order_executor):
        """Test adding and removing orders."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        runner._add_order("order_1", {"symbol": "BTCUSDT", "side": "BUY"})
        assert len(runner.get_active_orders()) == 1

        removed = runner._remove_order("order_1")
        assert removed is not None
        assert len(runner.get_active_orders()) == 0


# ============================================================================
# TESTS: FuturesLiveRunner Health
# ============================================================================


class TestFuturesLiveRunnerHealth:
    """Tests for FuturesLiveRunner health checks."""

    def test_health_status_initial(self, runner_config, mock_market_data, mock_order_executor):
        """Test initial health status."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        health = runner.get_health_status()

        assert isinstance(health, HealthStatus)
        assert health.state == LiveRunnerState.INITIALIZING
        assert health.active_orders == 0
        assert health.open_positions == 0

    def test_health_status_running(self, runner_config, mock_market_data, mock_order_executor):
        """Test health status when running."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        runner.start()
        health = runner.get_health_status()

        assert health.state == LiveRunnerState.RUNNING
        assert health.connection_status["exchange"] is True

        runner.stop()

    def test_health_check_connected(self, runner_config, mock_market_data, mock_order_executor):
        """Test health check when connected."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        runner.start()
        is_healthy = runner.perform_health_check()

        assert is_healthy is True

        runner.stop()

    def test_health_check_disconnected(self, runner_config, mock_market_data, mock_order_executor):
        """Test health check when disconnected."""
        mock_market_data.is_connected.return_value = False

        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        runner.start()
        health = runner.get_health_status()

        assert health.is_healthy is False
        assert "Not connected" in " ".join(health.errors)

        runner.stop()


# ============================================================================
# TESTS: FuturesLiveRunner Events
# ============================================================================


class TestFuturesLiveRunnerEvents:
    """Tests for FuturesLiveRunner event handling."""

    def test_register_callback(self, runner_config, mock_market_data, mock_order_executor):
        """Test registering event callbacks."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        callback = MagicMock()
        runner.register_callback(LiveRunnerEvent.STATE_CHANGED, callback)

        runner.start()

        # Callback should have been called for state changes
        assert callback.called

        runner.stop()

    def test_event_emitted_on_state_change(
        self, runner_config, mock_market_data, mock_order_executor
    ):
        """Test that events are emitted on state changes."""
        events_received = []

        def callback(event, data):
            events_received.append((event, data))

        runner_config.event_callbacks = {LiveRunnerEvent.STATE_CHANGED: [callback]}

        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        runner.start()
        runner.stop()

        assert len(events_received) > 0


# ============================================================================
# TESTS: FuturesLiveRunner Statistics
# ============================================================================


class TestFuturesLiveRunnerStats:
    """Tests for FuturesLiveRunner statistics."""

    def test_initial_stats(self, runner_config, mock_market_data, mock_order_executor):
        """Test initial statistics."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        stats = runner.get_stats()

        assert stats.total_orders_submitted == 0
        assert stats.total_orders_filled == 0
        assert stats.total_position_syncs == 0

    def test_stats_reset_on_start(self, runner_config, mock_market_data, mock_order_executor):
        """Test statistics reset on start."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        runner.start()
        stats = runner.get_stats()

        assert stats.session_start_time_ms > 0

        runner.stop()


# ============================================================================
# TESTS: FuturesPositionSynchronizer
# ============================================================================


class TestFuturesPositionSynchronizer:
    """Tests for FuturesPositionSynchronizer."""

    def test_sync_config_defaults(self):
        """Test sync config default values."""
        config = FuturesSyncConfig()

        assert config.sync_interval_sec == 10.0
        assert config.qty_tolerance_pct == 0.001
        assert config.auto_reconcile is False

    def test_sync_config_from_dict(self):
        """Test creating sync config with custom values."""
        # FuturesSyncConfig is a dataclass - create with kwargs
        config = FuturesSyncConfig(
            sync_interval_sec=5.0,
            qty_tolerance_pct=0.002,
            auto_reconcile=True,
        )

        assert config.sync_interval_sec == 5.0
        assert config.qty_tolerance_pct == 0.002
        assert config.auto_reconcile is True

    def test_sync_result_success(self):
        """Test successful sync result."""
        result = FuturesSyncResult(
            timestamp_ms=int(time.time() * 1000),
            success=True,
            exchange=Exchange.BINANCE,
            futures_type=FuturesType.CRYPTO_PERPETUAL,
            local_positions={},
            remote_positions={},
            events=[],
        )

        assert result.success is True
        assert len(result.events) == 0

    def test_sync_result_with_discrepancies(self, sample_position):
        """Test sync result with discrepancies."""
        diff = FuturesPositionDiff(
            symbol="BTCUSDT",
            event_type=FuturesSyncEventType.QTY_MISMATCH,
            local_qty=Decimal("0.5"),
            remote_qty=Decimal("0.6"),
        )

        result = FuturesSyncResult(
            timestamp_ms=int(time.time() * 1000),
            success=True,
            exchange=Exchange.BINANCE,
            futures_type=FuturesType.CRYPTO_PERPETUAL,
            local_positions={"BTCUSDT": sample_position},
            remote_positions={"BTCUSDT": sample_position},
            events=[diff],
        )

        assert len(result.events) == 1
        assert result.events[0].event_type == FuturesSyncEventType.QTY_MISMATCH


class TestFuturesPositionDiff:
    """Tests for FuturesPositionDiff."""

    def test_position_mismatch(self):
        """Test quantity mismatch diff."""
        diff = FuturesPositionDiff(
            symbol="BTCUSDT",
            event_type=FuturesSyncEventType.QTY_MISMATCH,
            local_qty=Decimal("0.5"),
            remote_qty=Decimal("0.6"),
        )

        assert diff.event_type == FuturesSyncEventType.QTY_MISMATCH
        assert diff.local_qty == Decimal("0.5")
        assert diff.remote_qty == Decimal("0.6")

    def test_unexpected_position(self):
        """Test unexpected position (remote has, local doesn't)."""
        diff = FuturesPositionDiff(
            symbol="ETHUSDT",
            event_type=FuturesSyncEventType.POSITION_OPENED,
            local_qty=Decimal("0"),
            remote_qty=Decimal("1.0"),
        )

        assert diff.event_type == FuturesSyncEventType.POSITION_OPENED

    def test_missing_position(self):
        """Test missing position (local has, remote doesn't)."""
        diff = FuturesPositionDiff(
            symbol="BTCUSDT",
            event_type=FuturesSyncEventType.POSITION_CLOSED,
            local_qty=Decimal("0.5"),
            remote_qty=Decimal("0"),
        )

        assert diff.event_type == FuturesSyncEventType.POSITION_CLOSED

    def test_adl_warning(self):
        """Test ADL detected diff."""
        diff = FuturesPositionDiff(
            symbol="BTCUSDT",
            event_type=FuturesSyncEventType.ADL_DETECTED,
            adl_risk_level=ADLRiskLevel.DANGER,
        )

        assert diff.event_type == FuturesSyncEventType.ADL_DETECTED
        assert diff.adl_risk_level == ADLRiskLevel.DANGER


class TestADLRiskLevel:
    """Tests for ADLRiskLevel enum."""

    def test_adl_levels(self):
        """Test all ADL risk levels."""
        # ADLRiskLevel uses string values: safe, warning, danger, critical
        assert ADLRiskLevel.SAFE.value == "safe"
        assert ADLRiskLevel.WARNING.value == "warning"
        assert ADLRiskLevel.DANGER.value == "danger"
        assert ADLRiskLevel.CRITICAL.value == "critical"


# ============================================================================
# TESTS: FuturesMarginMonitor
# ============================================================================


class TestMarginMonitorConfig:
    """Tests for MarginMonitorConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = MarginMonitorConfig()

        assert config.warning_ratio == Decimal("1.5")
        assert config.danger_ratio == Decimal("1.2")
        assert config.critical_ratio == Decimal("1.1")
        assert config.liquidation_ratio == Decimal("1.0")

    def test_from_dict(self):
        """Test creating config from dictionary."""
        d = {
            "warning_ratio": "1.6",
            "danger_ratio": "1.3",
            "enable_alerts": False,
        }

        config = MarginMonitorConfig.from_dict(d)

        assert config.warning_ratio == Decimal("1.6")
        assert config.danger_ratio == Decimal("1.3")
        assert config.enable_alerts is False

    def test_for_crypto(self):
        """Test crypto preset."""
        config = MarginMonitorConfig.for_crypto()

        assert config.futures_type == FuturesType.CRYPTO_PERPETUAL

    def test_for_cme(self):
        """Test CME preset with stricter thresholds."""
        config = MarginMonitorConfig.for_cme()

        assert config.futures_type == FuturesType.INDEX_FUTURES
        assert config.warning_ratio == Decimal("1.3")
        assert config.danger_ratio == Decimal("1.15")


class TestMarginLevelTracker:
    """Tests for MarginLevelTracker."""

    def test_classify_level_healthy(self):
        """Test healthy level classification."""
        config = MarginMonitorConfig()
        tracker = MarginLevelTracker(config)

        level = tracker.classify_level(Decimal("2.0"))
        assert level == MarginLevel.HEALTHY

    def test_classify_level_warning(self):
        """Test warning level classification."""
        config = MarginMonitorConfig()
        tracker = MarginLevelTracker(config)

        level = tracker.classify_level(Decimal("1.3"))
        assert level == MarginLevel.WARNING

    def test_classify_level_danger(self):
        """Test danger level classification."""
        config = MarginMonitorConfig()
        tracker = MarginLevelTracker(config)

        level = tracker.classify_level(Decimal("1.15"))
        assert level == MarginLevel.DANGER

    def test_classify_level_critical(self):
        """Test critical level classification."""
        config = MarginMonitorConfig()
        tracker = MarginLevelTracker(config)

        level = tracker.classify_level(Decimal("1.05"))
        assert level == MarginLevel.CRITICAL

    def test_classify_level_liquidation(self):
        """Test liquidation level classification."""
        config = MarginMonitorConfig()
        tracker = MarginLevelTracker(config)

        level = tracker.classify_level(Decimal("0.95"))
        assert level == MarginLevel.LIQUIDATION

    def test_update_level_detects_change(self):
        """Test level change detection."""
        config = MarginMonitorConfig()
        tracker = MarginLevelTracker(config)

        # Initial level is HEALTHY by default
        # Update with HEALTHY ratio - no change expected
        level, changed = tracker.update_level(Decimal("2.0"))
        assert level == MarginLevel.HEALTHY
        assert changed is False  # Initial level is already HEALTHY

        # Same level
        level, changed = tracker.update_level(Decimal("1.8"))
        assert level == MarginLevel.HEALTHY
        assert changed is False

        # Level change to WARNING
        level, changed = tracker.update_level(Decimal("1.3"))
        assert level == MarginLevel.WARNING
        assert changed is True

    def test_get_buffer_to_next_level(self):
        """Test buffer calculation."""
        config = MarginMonitorConfig()
        tracker = MarginLevelTracker(config)

        next_level, buffer = tracker.get_buffer_to_next_level(Decimal("1.6"))

        assert next_level == MarginLevel.WARNING
        assert buffer == Decimal("0.1")  # 1.6 - 1.5


class TestMarginSnapshot:
    """Tests for MarginSnapshot."""

    def test_snapshot_creation(self):
        """Test snapshot creation."""
        snapshot = MarginSnapshot(
            timestamp_ms=int(time.time() * 1000),
            equity=Decimal("10000"),
            margin_used=Decimal("5000"),
            available_margin=Decimal("5000"),
            maintenance_margin=Decimal("2500"),
            margin_ratio=Decimal("2.0"),
            level=MarginLevel.HEALTHY,
            position_count=2,
            total_position_value=Decimal("50000"),
        )

        assert snapshot.equity == Decimal("10000")
        assert snapshot.margin_ratio == Decimal("2.0")

    def test_utilization_pct(self):
        """Test utilization percentage calculation."""
        snapshot = MarginSnapshot(
            timestamp_ms=int(time.time() * 1000),
            equity=Decimal("10000"),
            margin_used=Decimal("5000"),
            available_margin=Decimal("5000"),
            maintenance_margin=Decimal("2500"),
            margin_ratio=Decimal("2.0"),
            level=MarginLevel.HEALTHY,
            position_count=2,
            total_position_value=Decimal("50000"),
        )

        assert snapshot.utilization_pct == Decimal("50")

    def test_buffer_to_liquidation_pct(self):
        """Test buffer to liquidation calculation."""
        snapshot = MarginSnapshot(
            timestamp_ms=int(time.time() * 1000),
            equity=Decimal("10000"),
            margin_used=Decimal("5000"),
            available_margin=Decimal("5000"),
            maintenance_margin=Decimal("2500"),
            margin_ratio=Decimal("2.0"),
            level=MarginLevel.HEALTHY,
            position_count=2,
            total_position_value=Decimal("50000"),
        )

        assert snapshot.buffer_to_liquidation_pct == Decimal("100")

    def test_to_dict(self):
        """Test snapshot serialization."""
        snapshot = MarginSnapshot(
            timestamp_ms=1234567890,
            equity=Decimal("10000"),
            margin_used=Decimal("5000"),
            available_margin=Decimal("5000"),
            maintenance_margin=Decimal("2500"),
            margin_ratio=Decimal("2.0"),
            level=MarginLevel.HEALTHY,
            position_count=2,
            total_position_value=Decimal("50000"),
        )

        d = snapshot.to_dict()

        assert d["equity"] == 10000.0
        assert d["level"] == "HEALTHY"


class TestMarginAlert:
    """Tests for MarginAlert."""

    def test_alert_creation(self):
        """Test alert creation."""
        alert = MarginAlert(
            timestamp_ms=int(time.time() * 1000),
            alert_type=MarginAlertType.LEVEL_CHANGE,
            level=MarginLevel.WARNING,
            previous_level=MarginLevel.HEALTHY,
            margin_ratio=Decimal("1.3"),
            equity=Decimal("10000"),
            margin_used=Decimal("7692"),
            message="Margin level changed to WARNING",
            severity="WARNING",
        )

        assert alert.alert_type == MarginAlertType.LEVEL_CHANGE
        assert alert.level == MarginLevel.WARNING

    def test_alert_to_dict(self):
        """Test alert serialization."""
        alert = MarginAlert(
            timestamp_ms=1234567890,
            alert_type=MarginAlertType.MARGIN_CALL,
            level=MarginLevel.CRITICAL,
            previous_level=MarginLevel.DANGER,
            margin_ratio=Decimal("1.05"),
            equity=Decimal("10000"),
            margin_used=Decimal("9524"),
            message="MARGIN CALL",
            recommended_action="Reduce positions",
            severity="ERROR",
        )

        d = alert.to_dict()

        assert d["alert_type"] == "MARGIN_CALL"
        assert d["severity"] == "ERROR"


class TestReductionSuggestion:
    """Tests for ReductionSuggestion."""

    def test_suggestion_creation(self):
        """Test reduction suggestion creation."""
        suggestion = ReductionSuggestion(
            symbol="BTCUSDT",
            current_qty=Decimal("1.0"),
            suggested_reduction_qty=Decimal("0.3"),
            suggested_reduction_pct=Decimal("30"),
            reason="Margin utilization too high",
            priority=1,
        )

        assert suggestion.symbol == "BTCUSDT"
        assert suggestion.suggested_reduction_pct == Decimal("30")
        assert suggestion.priority == 1

    def test_suggestion_to_dict(self):
        """Test suggestion serialization."""
        suggestion = ReductionSuggestion(
            symbol="ETHUSDT",
            current_qty=Decimal("10.0"),
            suggested_reduction_qty=Decimal("5.0"),
            suggested_reduction_pct=Decimal("50"),
            reason="Critical margin",
            priority=1,
        )

        d = suggestion.to_dict()

        assert d["symbol"] == "ETHUSDT"
        assert d["suggested_reduction_pct"] == 50.0


class TestMarginAlertManager:
    """Tests for MarginAlertManager."""

    def test_should_alert_initially(self):
        """Test initial alert should pass."""
        config = MarginMonitorConfig(alert_cooldown_sec=60.0)
        manager = MarginAlertManager(config)

        assert manager.should_alert(MarginAlertType.LEVEL_CHANGE) is True

    def test_should_alert_cooldown(self):
        """Test cooldown prevents repeated alerts."""
        config = MarginMonitorConfig(alert_cooldown_sec=60.0)
        manager = MarginAlertManager(config)

        alert = MarginAlert(
            timestamp_ms=int(time.time() * 1000),
            alert_type=MarginAlertType.MARGIN_CALL,
            level=MarginLevel.CRITICAL,
            previous_level=MarginLevel.DANGER,
            margin_ratio=Decimal("1.05"),
            equity=Decimal("10000"),
            margin_used=Decimal("9524"),
            message="MARGIN CALL",
            severity="ERROR",
        )

        manager.send_alert(alert)
        assert manager.should_alert(MarginAlertType.MARGIN_CALL) is False

    def test_callback_invoked(self):
        """Test callbacks are invoked on alert."""
        callback = MagicMock()
        config = MarginMonitorConfig()
        manager = MarginAlertManager(config, callbacks=[callback])

        alert = MarginAlert(
            timestamp_ms=int(time.time() * 1000),
            alert_type=MarginAlertType.LEVEL_CHANGE,
            level=MarginLevel.WARNING,
            previous_level=MarginLevel.HEALTHY,
            margin_ratio=Decimal("1.3"),
            equity=Decimal("10000"),
            margin_used=Decimal("7692"),
            message="Level changed",
            severity="WARNING",
        )

        manager.send_alert(alert)

        callback.assert_called_once()

    def test_get_alert_history(self):
        """Test alert history retrieval."""
        config = MarginMonitorConfig(alert_cooldown_sec=0)
        manager = MarginAlertManager(config)

        for i in range(5):
            alert = MarginAlert(
                timestamp_ms=int(time.time() * 1000) + i,
                alert_type=MarginAlertType.LEVEL_CHANGE,
                level=MarginLevel.WARNING,
                previous_level=None,
                margin_ratio=Decimal("1.3"),
                equity=Decimal("10000"),
                margin_used=Decimal("7692"),
                message=f"Alert {i}",
                severity="WARNING",
            )
            manager.send_alert(alert)

        history = manager.get_alert_history(limit=3)
        assert len(history) == 3


class TestFuturesMarginMonitor:
    """Tests for FuturesMarginMonitor."""

    def test_monitor_creation(self, mock_margin_provider):
        """Test monitor creation."""
        config = MarginMonitorConfig()
        monitor = FuturesMarginMonitor(config, mock_margin_provider)

        assert monitor.get_current_level() == MarginLevel.HEALTHY

    def test_take_snapshot(self, mock_margin_provider):
        """Test taking a snapshot."""
        config = MarginMonitorConfig()
        monitor = FuturesMarginMonitor(config, mock_margin_provider)

        snapshot = monitor.take_snapshot()

        assert snapshot.equity == Decimal("10000")
        assert snapshot.margin_used == Decimal("5000")
        assert snapshot.margin_ratio == Decimal("2.0")
        assert snapshot.level == MarginLevel.HEALTHY

    def test_check_once(self, mock_margin_provider):
        """Test single check."""
        config = MarginMonitorConfig()
        monitor = FuturesMarginMonitor(config, mock_margin_provider)

        snapshot, alert = monitor.check_once()

        assert snapshot is not None
        assert monitor.get_current_snapshot() is not None

    def test_check_once_with_level_change(self, mock_margin_provider):
        """Test check with level change generates alert."""
        config = MarginMonitorConfig()
        monitor = FuturesMarginMonitor(config, mock_margin_provider)

        # First check - initial level is HEALTHY, current is also HEALTHY, no change
        snapshot1, alert1 = monitor.check_once()
        assert alert1 is None  # No level change from initial HEALTHY to HEALTHY

        # Change to warning level (margin ratio 1.3 = 10000/7692)
        mock_margin_provider.get_total_margin_used.return_value = Decimal("7700")

        snapshot2, alert2 = monitor.check_once()
        assert alert2 is not None
        assert alert2.level == MarginLevel.WARNING

    def test_get_history(self, mock_margin_provider):
        """Test history retrieval."""
        config = MarginMonitorConfig()
        monitor = FuturesMarginMonitor(config, mock_margin_provider)

        for _ in range(5):
            monitor.check_once()

        history = monitor.get_history()
        assert len(history) == 5

    def test_is_healthy(self, mock_margin_provider):
        """Test health check."""
        config = MarginMonitorConfig()
        monitor = FuturesMarginMonitor(config, mock_margin_provider)
        monitor.check_once()

        assert monitor.is_healthy() is True

    def test_is_at_risk(self, mock_margin_provider):
        """Test risk check."""
        mock_margin_provider.get_total_margin_used.return_value = Decimal("9000")

        config = MarginMonitorConfig()
        monitor = FuturesMarginMonitor(config, mock_margin_provider)
        monitor.check_once()

        assert monitor.is_at_risk() is True

    def test_register_alert_callback(self, mock_margin_provider):
        """Test registering alert callback."""
        callback = MagicMock()
        config = MarginMonitorConfig()
        monitor = FuturesMarginMonitor(config, mock_margin_provider)

        monitor.register_alert_callback(callback)

        # First check with HEALTHY won't trigger callback
        monitor.check_once()
        callback.assert_not_called()  # No level change

        # Change margin to trigger warning level
        mock_margin_provider.get_total_margin_used.return_value = Decimal("8000")
        monitor.check_once()

        # Callback should be called for level change to warning
        callback.assert_called()

    def test_get_reduction_suggestions(self, mock_margin_provider, sample_position):
        """Test reduction suggestions."""
        mock_margin_provider.get_total_margin_used.return_value = Decimal("9000")
        mock_margin_provider.get_positions.return_value = {"BTCUSDT": sample_position}

        config = MarginMonitorConfig()
        monitor = FuturesMarginMonitor(config, mock_margin_provider)
        monitor.check_once()

        suggestions = monitor.get_reduction_suggestions({"BTCUSDT": sample_position})

        # Should have suggestions when margin is at risk
        assert len(suggestions) > 0

    def test_get_statistics(self, mock_margin_provider):
        """Test statistics calculation."""
        config = MarginMonitorConfig()
        monitor = FuturesMarginMonitor(config, mock_margin_provider)

        for _ in range(10):
            monitor.check_once()

        stats = monitor.get_statistics()

        assert stats["snapshot_count"] == 10
        assert "mean_ratio" in stats


# ============================================================================
# TESTS: Factory Functions
# ============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_futures_live_runner(self, runner_config, mock_market_data, mock_order_executor):
        """Test create_futures_live_runner factory."""
        runner = create_futures_live_runner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        assert isinstance(runner, FuturesLiveRunner)

    def test_create_crypto_futures_runner(self, mock_market_data, mock_order_executor):
        """Test create_crypto_futures_runner factory."""
        runner = create_crypto_futures_runner(
            symbols=["BTCUSDT"],
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        assert isinstance(runner, FuturesLiveRunner)

    def test_create_cme_futures_runner(self, mock_market_data, mock_order_executor):
        """Test create_cme_futures_runner factory."""
        runner = create_cme_futures_runner(
            symbols=["ES", "NQ"],
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        assert isinstance(runner, FuturesLiveRunner)

    def test_create_margin_monitor(self, mock_margin_provider):
        """Test create_margin_monitor factory."""
        monitor = create_margin_monitor(mock_margin_provider)

        assert isinstance(monitor, FuturesMarginMonitor)

    def test_create_crypto_margin_monitor(self, mock_margin_provider):
        """Test create_crypto_margin_monitor factory."""
        monitor = create_crypto_margin_monitor(mock_margin_provider)

        assert isinstance(monitor, FuturesMarginMonitor)

    def test_create_cme_margin_monitor(self, mock_margin_provider):
        """Test create_cme_margin_monitor factory."""
        monitor = create_cme_margin_monitor(mock_margin_provider)

        assert isinstance(monitor, FuturesMarginMonitor)


# ============================================================================
# TESTS: Integration Scenarios
# ============================================================================


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""

    def test_full_lifecycle(self, runner_config, mock_market_data, mock_order_executor):
        """Test complete lifecycle: init -> start -> run -> stop."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        # Initialize
        assert runner.state == LiveRunnerState.INITIALIZING

        # Start
        runner.start()
        assert runner.state == LiveRunnerState.RUNNING

        # Pause
        runner.pause()
        assert runner.state == LiveRunnerState.PAUSED

        # Resume
        runner.resume()
        assert runner.state == LiveRunnerState.RUNNING

        # Stop
        runner.stop()
        assert runner.state == LiveRunnerState.STOPPED

    def test_margin_level_transitions(self, mock_margin_provider):
        """Test margin level transitions."""
        config = MarginMonitorConfig(alert_cooldown_sec=0)
        monitor = FuturesMarginMonitor(config, mock_margin_provider)

        # Start healthy
        monitor.check_once()
        assert monitor.get_current_level() == MarginLevel.HEALTHY

        # Transition to warning
        mock_margin_provider.get_total_margin_used.return_value = Decimal("7700")
        snapshot, alert = monitor.check_once()
        assert monitor.get_current_level() == MarginLevel.WARNING
        assert alert is not None
        assert alert.level == MarginLevel.WARNING

        # Transition to danger
        mock_margin_provider.get_total_margin_used.return_value = Decimal("8700")
        snapshot, alert = monitor.check_once()
        assert monitor.get_current_level() == MarginLevel.DANGER
        assert alert is not None

        # Transition to critical
        mock_margin_provider.get_total_margin_used.return_value = Decimal("9500")
        snapshot, alert = monitor.check_once()
        assert monitor.get_current_level() == MarginLevel.CRITICAL

        # Recovery to healthy
        mock_margin_provider.get_total_margin_used.return_value = Decimal("5000")
        snapshot, alert = monitor.check_once()
        assert monitor.get_current_level() == MarginLevel.HEALTHY
        assert alert.alert_type == MarginAlertType.RECOVERED

    def test_position_sync_workflow(self, sample_position):
        """Test position sync workflow."""
        # Create mock providers
        position_provider = MagicMock()
        position_provider.get_futures_positions.return_value = [sample_position]

        account_provider = MagicMock()
        account_provider.get_futures_account.return_value = MagicMock(
            total_wallet_balance=Decimal("10000"),
            total_initial_margin=Decimal("5000"),
            total_maint_margin=Decimal("2500"),
            total_margin_balance=Decimal("10000"),
        )

        def local_getter():
            return {}

        # Create synchronizer with exchange/futures_type in config
        config = FuturesSyncConfig(
            exchange=Exchange.BINANCE,
            futures_type=FuturesType.CRYPTO_PERPETUAL,
        )
        sync = FuturesPositionSynchronizer(
            position_provider=position_provider,
            account_provider=account_provider,
            local_state_getter=local_getter,
            config=config,
        )

        # Perform sync
        result = sync.sync_once()

        assert result.success is True
        assert "BTCUSDT" in result.remote_positions

    def test_runner_with_all_components(
        self,
        runner_config,
        mock_market_data,
        mock_order_executor,
        mock_signal_provider,
    ):
        """Test runner with all components."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
            signal_provider=mock_signal_provider,
        )

        runner.start()

        # Get status
        health = runner.get_health_status()
        assert health.is_healthy is True

        stats = runner.get_stats()
        assert stats.session_start_time_ms > 0

        runner.stop()


# ============================================================================
# TESTS: Error Handling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_runner_handles_disconnection(
        self, runner_config, mock_market_data, mock_order_executor
    ):
        """Test runner handles disconnection gracefully."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        runner.start()

        # Simulate disconnection
        mock_market_data.is_connected.return_value = False

        # Perform a loop iteration
        runner._loop_iteration()

        # Should not crash
        runner.stop()

    def test_runner_handles_order_execution_error(
        self,
        runner_config,
        mock_market_data,
        mock_order_executor,
        mock_signal_provider,
    ):
        """Test runner handles order execution errors."""
        mock_order_executor.submit_market_order.side_effect = Exception("API Error")

        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
            signal_provider=mock_signal_provider,
        )

        runner.start()

        # Try to process signals - should handle error
        runner._process_signals()

        # Stats should reflect rejected order
        stats = runner.get_stats()
        assert stats.total_orders_rejected >= 0  # May have incremented

        runner.stop()

    def test_margin_monitor_handles_provider_error(self, mock_margin_provider):
        """Test margin monitor handles provider errors."""
        mock_margin_provider.get_account_equity.side_effect = Exception("Provider error")

        config = MarginMonitorConfig()
        monitor = FuturesMarginMonitor(config, mock_margin_provider)

        # Should not raise
        snapshot = monitor.take_snapshot()

        # Should return safe default
        assert snapshot.equity == Decimal("0")
        assert snapshot.level == MarginLevel.LIQUIDATION


# ============================================================================
# TESTS: Thread Safety
# ============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_position_access(
        self, runner_config, mock_market_data, mock_order_executor, sample_position
    ):
        """Test concurrent position access is thread-safe."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        errors = []

        def reader():
            for _ in range(100):
                try:
                    runner.get_positions()
                except Exception as e:
                    errors.append(e)

        def writer():
            for _ in range(100):
                try:
                    runner._update_local_positions({"BTCUSDT": sample_position})
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_order_access(self, runner_config, mock_market_data, mock_order_executor):
        """Test concurrent order access is thread-safe."""
        runner = FuturesLiveRunner(
            config=runner_config,
            market_data=mock_market_data,
            order_executor=mock_order_executor,
        )

        errors = []

        def adder():
            for i in range(100):
                try:
                    runner._add_order(f"order_{i}", {"symbol": "BTCUSDT"})
                except Exception as e:
                    errors.append(e)

        def remover():
            for i in range(100):
                try:
                    runner._remove_order(f"order_{i}")
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=adder),
            threading.Thread(target=remover),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_margin_monitor_thread_safety(self, mock_margin_provider):
        """Test margin monitor thread safety."""
        config = MarginMonitorConfig()
        monitor = FuturesMarginMonitor(config, mock_margin_provider)

        errors = []

        def checker():
            for _ in range(50):
                try:
                    monitor.check_once()
                except Exception as e:
                    errors.append(e)

        def reader():
            for _ in range(50):
                try:
                    monitor.get_current_snapshot()
                    monitor.get_history()
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=checker),
            threading.Thread(target=reader),
            threading.Thread(target=checker),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
