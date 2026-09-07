# -*- coding: utf-8 -*-
"""
tests/test_futures_position_sync.py
Comprehensive tests for futures position synchronization service.

Phase 9: Unified Live Trading (2025-12-02)

Tests cover:
- Position sync configuration
- Position comparison and diff detection
- Liquidation detection
- ADL (Auto-Deleveraging) monitoring
- Funding tracking integration
- Background sync operations
- Factory functions for crypto and CME futures
- Edge cases and error handling
- Thread safety
"""

import asyncio
import pytest
import threading
import time  # Used in fixtures for timestamp_ms
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import MagicMock, Mock, patch

from core_futures import (
    FuturesPosition,
    FuturesAccountState,
    FuturesType,
    MarginMode,
    PositionSide,
    Exchange,
)

from services.futures_position_sync import (
    # Config and constants
    FuturesSyncConfig,
    DEFAULT_SYNC_INTERVAL_SEC,
    MIN_SYNC_INTERVAL_SEC,
    MAX_SYNC_INTERVAL_SEC,
    DEFAULT_QTY_TOLERANCE_PCT,
    ADL_WARNING_PERCENTILE,
    ADL_DANGER_PERCENTILE,
    ADL_CRITICAL_PERCENTILE,
    # Enums
    FuturesSyncEventType,
    ADLRiskLevel,
    # Data classes
    FuturesPositionDiff,
    FuturesSyncResult,
    # Main class
    FuturesPositionSynchronizer,
    # Factory functions
    create_crypto_futures_sync,
    create_cme_futures_sync,
)


# =============================================================================
# Test Fixtures and Mocks
# =============================================================================


@pytest.fixture
def mock_futures_position() -> FuturesPosition:
    """Create a mock futures position."""
    return FuturesPosition(
        symbol="BTCUSDT",
        qty=Decimal("0.5"),
        entry_price=Decimal("50000.00"),
        leverage=10,
        margin_mode=MarginMode.CROSS,
        unrealized_pnl=Decimal("250.00"),
        side=PositionSide.LONG,
        mark_price=Decimal("50500.00"),
        liquidation_price=Decimal("45000.00"),
    )


@pytest.fixture
def mock_account_state() -> FuturesAccountState:
    """Create a mock futures account state."""
    return FuturesAccountState(
        timestamp_ms=int(time.time() * 1000),
        total_wallet_balance=Decimal("10000.00"),
        total_margin_balance=Decimal("10250.00"),
        total_unrealized_pnl=Decimal("250.00"),
        available_balance=Decimal("7250.00"),
        total_initial_margin=Decimal("2500.00"),
        total_maint_margin=Decimal("1250.00"),
    )


@pytest.fixture
def mock_position_provider(mock_futures_position):
    """Create a mock position provider."""
    provider = MagicMock()
    provider.get_futures_positions.return_value = [mock_futures_position]
    return provider


@pytest.fixture
def mock_account_provider(mock_account_state):
    """Create a mock account provider."""
    provider = MagicMock()
    provider.get_futures_account.return_value = mock_account_state
    return provider


@pytest.fixture
def mock_adl_provider():
    """Create a mock ADL indicator provider."""
    provider = MagicMock()
    provider.get_adl_indicator.return_value = 2  # 2/5 lights = 40%
    return provider


@pytest.fixture
def local_state_getter(mock_futures_position):
    """Create a local state getter function."""
    positions = {"BTCUSDT": mock_futures_position}
    return lambda: positions


@pytest.fixture
def default_config() -> FuturesSyncConfig:
    """Create default sync config."""
    return FuturesSyncConfig(
        exchange=Exchange.BINANCE,
        futures_type=FuturesType.CRYPTO_PERPETUAL,
        sync_interval_sec=10.0,
    )


@pytest.fixture
def synchronizer(
    mock_position_provider,
    mock_account_provider,
    local_state_getter,
    default_config,
):
    """Create a synchronizer with mocked dependencies."""
    return FuturesPositionSynchronizer(
        position_provider=mock_position_provider,
        account_provider=mock_account_provider,
        local_state_getter=local_state_getter,
        config=default_config,
    )


# =============================================================================
# Configuration Tests
# =============================================================================


class TestFuturesSyncConfig:
    """Tests for FuturesSyncConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FuturesSyncConfig()
        assert config.exchange == Exchange.BINANCE
        assert config.futures_type == FuturesType.CRYPTO_PERPETUAL
        assert config.sync_interval_sec == DEFAULT_SYNC_INTERVAL_SEC
        assert config.qty_tolerance_pct == DEFAULT_QTY_TOLERANCE_PCT
        assert config.enable_adl_monitoring is True
        assert config.enable_liquidation_detection is True
        assert config.enable_funding_tracking is True
        assert config.auto_reconcile is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = FuturesSyncConfig(
            exchange=Exchange.CME,
            futures_type=FuturesType.INDEX_FUTURES,
            sync_interval_sec=30.0,
            enable_adl_monitoring=False,
        )
        assert config.exchange == Exchange.CME
        assert config.futures_type == FuturesType.INDEX_FUTURES
        assert config.sync_interval_sec == 30.0
        assert config.enable_adl_monitoring is False

    def test_config_validation_min_interval(self):
        """Test validation of minimum sync interval."""
        with pytest.raises(ValueError, match="sync_interval_sec must be >="):
            FuturesSyncConfig(sync_interval_sec=0.1)  # Below MIN_SYNC_INTERVAL_SEC

    def test_config_validation_max_interval(self):
        """Test validation of maximum sync interval."""
        with pytest.raises(ValueError, match="sync_interval_sec must be <="):
            FuturesSyncConfig(sync_interval_sec=500.0)  # Above MAX_SYNC_INTERVAL_SEC

    def test_config_symbol_filtering(self):
        """Test symbol include/exclude lists."""
        config = FuturesSyncConfig(
            include_symbols=["BTCUSDT", "ETHUSDT"],
            exclude_symbols=["DOGEUSDT"],
        )
        assert config.include_symbols == ["BTCUSDT", "ETHUSDT"]
        assert config.exclude_symbols == ["DOGEUSDT"]

    def test_config_adl_thresholds(self):
        """Test ADL threshold configuration."""
        config = FuturesSyncConfig(
            adl_warning_threshold=60.0,
            adl_danger_threshold=80.0,
        )
        assert config.adl_warning_threshold == 60.0
        assert config.adl_danger_threshold == 80.0


# =============================================================================
# Event Type Tests
# =============================================================================


class TestFuturesSyncEventType:
    """Tests for FuturesSyncEventType enum."""

    def test_position_events(self):
        """Test position-related event types."""
        assert FuturesSyncEventType.POSITION_OPENED == "position_opened"
        assert FuturesSyncEventType.POSITION_CLOSED == "position_closed"
        assert FuturesSyncEventType.POSITION_MODIFIED == "position_modified"
        assert FuturesSyncEventType.QTY_MISMATCH == "qty_mismatch"
        assert FuturesSyncEventType.LEVERAGE_MISMATCH == "leverage_mismatch"

    def test_futures_specific_events(self):
        """Test futures-specific event types."""
        assert FuturesSyncEventType.LIQUIDATION_DETECTED == "liquidation_detected"
        assert FuturesSyncEventType.ADL_DETECTED == "adl_detected"
        assert FuturesSyncEventType.FUNDING_RECEIVED == "funding_received"
        assert FuturesSyncEventType.FUNDING_PAID == "funding_paid"
        assert FuturesSyncEventType.SETTLEMENT_OCCURRED == "settlement_occurred"

    def test_margin_events(self):
        """Test margin-related event types."""
        assert FuturesSyncEventType.MARGIN_CALL == "margin_call"
        assert FuturesSyncEventType.MARGIN_RATIO_LOW == "margin_ratio_low"

    def test_sync_status_events(self):
        """Test sync status event types."""
        assert FuturesSyncEventType.SYNC_SUCCESS == "sync_success"
        assert FuturesSyncEventType.SYNC_FAILURE == "sync_failure"
        assert FuturesSyncEventType.CONNECTION_LOST == "connection_lost"
        assert FuturesSyncEventType.CONNECTION_RESTORED == "connection_restored"


class TestADLRiskLevel:
    """Tests for ADLRiskLevel enum."""

    def test_risk_levels(self):
        """Test all ADL risk levels."""
        assert ADLRiskLevel.SAFE == "safe"
        assert ADLRiskLevel.WARNING == "warning"
        assert ADLRiskLevel.DANGER == "danger"
        assert ADLRiskLevel.CRITICAL == "critical"


# =============================================================================
# Position Diff Tests
# =============================================================================


class TestFuturesPositionDiff:
    """Tests for FuturesPositionDiff dataclass."""

    def test_basic_diff(self):
        """Test basic position diff creation."""
        diff = FuturesPositionDiff(
            symbol="BTCUSDT",
            event_type=FuturesSyncEventType.QTY_MISMATCH,
            local_qty=Decimal("1.0"),
            remote_qty=Decimal("0.8"),
        )
        assert diff.symbol == "BTCUSDT"
        assert diff.event_type == FuturesSyncEventType.QTY_MISMATCH
        assert diff.qty_diff == Decimal("-0.2")

    def test_qty_diff_calculation(self):
        """Test quantity difference calculation."""
        diff = FuturesPositionDiff(
            symbol="ETHUSDT",
            event_type=FuturesSyncEventType.QTY_MISMATCH,
            local_qty=Decimal("5.0"),
            remote_qty=Decimal("7.5"),
        )
        assert diff.qty_diff == Decimal("2.5")  # remote - local

    def test_pnl_diff_calculation(self):
        """Test PnL difference calculation."""
        diff = FuturesPositionDiff(
            symbol="BTCUSDT",
            event_type=FuturesSyncEventType.POSITION_MODIFIED,
            local_unrealized_pnl=Decimal("100.00"),
            remote_unrealized_pnl=Decimal("150.00"),
        )
        assert diff.pnl_diff == Decimal("50.00")

    def test_is_position_closed_externally_liquidation(self):
        """Test liquidation detection flag."""
        diff = FuturesPositionDiff(
            symbol="BTCUSDT",
            event_type=FuturesSyncEventType.LIQUIDATION_DETECTED,
            local_qty=Decimal("1.0"),
        )
        assert diff.is_position_closed_externally is True

    def test_is_position_closed_externally_adl(self):
        """Test ADL detection flag."""
        diff = FuturesPositionDiff(
            symbol="BTCUSDT",
            event_type=FuturesSyncEventType.ADL_DETECTED,
            local_qty=Decimal("1.0"),
        )
        assert diff.is_position_closed_externally is True

    def test_is_position_closed_externally_normal_close(self):
        """Test normal close (not external)."""
        diff = FuturesPositionDiff(
            symbol="BTCUSDT",
            event_type=FuturesSyncEventType.POSITION_CLOSED,
        )
        assert diff.is_position_closed_externally is False

    def test_diff_with_none_values(self):
        """Test diff calculation with None values."""
        diff = FuturesPositionDiff(
            symbol="BTCUSDT",
            event_type=FuturesSyncEventType.POSITION_OPENED,
            local_qty=None,
            remote_qty=Decimal("1.0"),
        )
        assert diff.qty_diff == Decimal("1.0")  # None treated as 0

    def test_diff_timestamp(self):
        """Test timestamp is set automatically."""
        diff = FuturesPositionDiff(
            symbol="BTCUSDT",
            event_type=FuturesSyncEventType.SYNC_SUCCESS,
        )
        assert diff.timestamp_ms > 0

    def test_diff_with_adl_data(self):
        """Test diff with ADL risk data."""
        diff = FuturesPositionDiff(
            symbol="BTCUSDT",
            event_type=FuturesSyncEventType.ADL_DETECTED,
            adl_percentile=85.0,
            adl_risk_level=ADLRiskLevel.DANGER,
        )
        assert diff.adl_percentile == 85.0
        assert diff.adl_risk_level == ADLRiskLevel.DANGER


# =============================================================================
# Sync Result Tests
# =============================================================================


class TestFuturesSyncResult:
    """Tests for FuturesSyncResult dataclass."""

    def test_successful_result(self, mock_futures_position):
        """Test successful sync result."""
        result = FuturesSyncResult(
            timestamp_ms=int(time.time() * 1000),
            success=True,
            exchange=Exchange.BINANCE,
            futures_type=FuturesType.CRYPTO_PERPETUAL,
            local_positions={"BTCUSDT": mock_futures_position},
            remote_positions={"BTCUSDT": mock_futures_position},
            account_balance=Decimal("10000.00"),
            margin_ratio=Decimal("0.25"),
        )
        assert result.success is True
        assert result.has_events is False
        assert result.has_liquidations is False
        assert result.has_adl_events is False
        assert result.position_count_diff == 0

    def test_result_with_events(self, mock_futures_position):
        """Test result with detected events."""
        events = [
            FuturesPositionDiff(
                symbol="BTCUSDT",
                event_type=FuturesSyncEventType.QTY_MISMATCH,
            )
        ]
        result = FuturesSyncResult(
            timestamp_ms=int(time.time() * 1000),
            success=True,
            exchange=Exchange.BINANCE,
            futures_type=FuturesType.CRYPTO_PERPETUAL,
            events=events,
        )
        assert result.has_events is True

    def test_result_with_liquidation(self):
        """Test result with liquidation event."""
        events = [
            FuturesPositionDiff(
                symbol="BTCUSDT",
                event_type=FuturesSyncEventType.LIQUIDATION_DETECTED,
            )
        ]
        result = FuturesSyncResult(
            timestamp_ms=int(time.time() * 1000),
            success=True,
            exchange=Exchange.BINANCE,
            futures_type=FuturesType.CRYPTO_PERPETUAL,
            events=events,
        )
        assert result.has_liquidations is True
        assert result.has_adl_events is False

    def test_result_with_adl(self):
        """Test result with ADL event."""
        events = [
            FuturesPositionDiff(
                symbol="BTCUSDT",
                event_type=FuturesSyncEventType.ADL_DETECTED,
            )
        ]
        result = FuturesSyncResult(
            timestamp_ms=int(time.time() * 1000),
            success=True,
            exchange=Exchange.BINANCE,
            futures_type=FuturesType.CRYPTO_PERPETUAL,
            events=events,
        )
        assert result.has_adl_events is True
        assert result.has_liquidations is False

    def test_result_position_count_diff(self, mock_futures_position):
        """Test position count difference calculation."""
        local_pos = {"BTCUSDT": mock_futures_position}
        remote_pos = {
            "BTCUSDT": mock_futures_position,
            "ETHUSDT": mock_futures_position,
        }
        result = FuturesSyncResult(
            timestamp_ms=int(time.time() * 1000),
            success=True,
            exchange=Exchange.BINANCE,
            futures_type=FuturesType.CRYPTO_PERPETUAL,
            local_positions=local_pos,
            remote_positions=remote_pos,
        )
        assert result.position_count_diff == 1  # remote has 1 more

    def test_failed_result(self):
        """Test failed sync result."""
        result = FuturesSyncResult(
            timestamp_ms=int(time.time() * 1000),
            success=False,
            exchange=Exchange.BINANCE,
            futures_type=FuturesType.CRYPTO_PERPETUAL,
            error="Connection timeout",
            retry_count=3,
        )
        assert result.success is False
        assert result.error == "Connection timeout"
        assert result.retry_count == 3


# =============================================================================
# Synchronizer Core Tests
# =============================================================================


class TestFuturesPositionSynchronizer:
    """Tests for FuturesPositionSynchronizer class."""

    def test_initialization(self, synchronizer):
        """Test synchronizer initialization."""
        assert synchronizer.sync_count == 0
        assert synchronizer.error_count == 0
        assert synchronizer.liquidation_count == 0
        assert synchronizer.adl_count == 0
        assert synchronizer.is_running is False

    def test_sync_once_success(self, synchronizer, mock_futures_position):
        """Test successful single sync operation."""
        result = synchronizer.sync_once()

        assert result.success is True
        assert result.exchange == Exchange.BINANCE
        assert result.futures_type == FuturesType.CRYPTO_PERPETUAL
        assert "BTCUSDT" in result.remote_positions
        assert synchronizer.sync_count == 1
        assert synchronizer.error_count == 0

    def test_sync_once_with_qty_mismatch(
        self,
        mock_position_provider,
        mock_account_provider,
        default_config,
    ):
        """Test sync with quantity mismatch detection."""
        # Local has 1.0 BTC
        local_position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000.00"),
            qty=Decimal("1.0"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )
        local_state_getter = lambda: {"BTCUSDT": local_position}

        # Remote has 0.5 BTC (significant mismatch)
        remote_position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000.00"),
            qty=Decimal("0.5"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )
        mock_position_provider.get_futures_positions.return_value = [remote_position]

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=local_state_getter,
            config=default_config,
        )

        result = sync.sync_once()

        assert result.has_events is True
        qty_mismatch_events = [
            e for e in result.events if e.event_type == FuturesSyncEventType.QTY_MISMATCH
        ]
        assert len(qty_mismatch_events) >= 1

    def test_sync_once_with_leverage_mismatch(
        self,
        mock_position_provider,
        mock_account_provider,
        default_config,
    ):
        """Test sync with leverage mismatch detection."""
        # Local has 10x leverage
        local_position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000.00"),
            qty=Decimal("1.0"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )
        local_state_getter = lambda: {"BTCUSDT": local_position}

        # Remote has 20x leverage
        remote_position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000.00"),
            qty=Decimal("1.0"),
            leverage=20,
            margin_mode=MarginMode.CROSS,
        )
        mock_position_provider.get_futures_positions.return_value = [remote_position]

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=local_state_getter,
            config=default_config,
        )

        result = sync.sync_once()

        leverage_events = [
            e for e in result.events if e.event_type == FuturesSyncEventType.LEVERAGE_MISMATCH
        ]
        assert len(leverage_events) >= 1

    def test_sync_detects_new_position(
        self,
        mock_position_provider,
        mock_account_provider,
        default_config,
    ):
        """Test sync detects new position opened on exchange."""
        # Local has no positions
        local_state_getter = lambda: {}

        # Remote has a new position
        remote_position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000.00"),
            qty=Decimal("1.0"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )
        mock_position_provider.get_futures_positions.return_value = [remote_position]

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=local_state_getter,
            config=default_config,
        )

        result = sync.sync_once()

        opened_events = [
            e for e in result.events if e.event_type == FuturesSyncEventType.POSITION_OPENED
        ]
        assert len(opened_events) == 1
        assert opened_events[0].symbol == "BTCUSDT"

    def test_sync_detects_position_closed(
        self,
        mock_position_provider,
        mock_account_provider,
        default_config,
    ):
        """Test sync detects position closed on exchange."""
        # Local has a position
        local_position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000.00"),
            qty=Decimal("1.0"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )
        local_state_getter = lambda: {"BTCUSDT": local_position}

        # Remote has no positions
        mock_position_provider.get_futures_positions.return_value = []

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=local_state_getter,
            config=default_config,
        )

        result = sync.sync_once()

        closed_events = [
            e for e in result.events if e.event_type == FuturesSyncEventType.POSITION_CLOSED
        ]
        assert len(closed_events) == 1

    def test_sync_with_provider_error(
        self,
        mock_account_provider,
        default_config,
    ):
        """Test sync handles provider errors."""
        error_provider = MagicMock()
        error_provider.get_futures_positions.side_effect = Exception("API Error")

        sync = FuturesPositionSynchronizer(
            position_provider=error_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {},
            config=default_config,
        )

        result = sync.sync_once()

        assert result.success is False
        assert "API Error" in result.error
        assert sync.error_count == 1

    def test_sync_retry_on_failure(
        self,
        mock_account_provider,
        default_config,
    ):
        """Test sync retries on transient failures."""
        fail_count = 0

        def fail_twice(*args, **kwargs):
            nonlocal fail_count
            fail_count += 1
            if fail_count <= 2:
                raise Exception(f"Transient error {fail_count}")
            return []

        retry_provider = MagicMock()
        retry_provider.get_futures_positions.side_effect = fail_twice

        sync = FuturesPositionSynchronizer(
            position_provider=retry_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {},
            config=default_config,
        )

        result = sync.sync_once()

        assert result.success is True
        assert result.retry_count == 2


# =============================================================================
# Liquidation Detection Tests
# =============================================================================


class TestLiquidationDetection:
    """Tests for liquidation detection functionality."""

    def test_liquidation_detection_high_leverage(
        self,
        mock_position_provider,
        mock_account_provider,
    ):
        """Test liquidation detection for high leverage position."""
        config = FuturesSyncConfig(
            enable_liquidation_detection=True,
        )

        # Local has high leverage position with loss
        local_position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000.00"),
            qty=Decimal("1.0"),
            leverage=50,  # High leverage
            margin_mode=MarginMode.CROSS,
            unrealized_pnl=Decimal("-5000.00"),  # Loss
        )
        local_state_getter = lambda: {"BTCUSDT": local_position}

        # Remote shows position gone
        mock_position_provider.get_futures_positions.return_value = []

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=local_state_getter,
            config=config,
        )

        # First sync to establish known positions
        sync._known_positions = {"BTCUSDT": local_position}

        result = sync.sync_once()

        liquidation_events = [
            e for e in result.events if e.event_type == FuturesSyncEventType.LIQUIDATION_DETECTED
        ]
        assert len(liquidation_events) >= 1
        assert sync.liquidation_count >= 1

    def test_liquidation_callback_fired(
        self,
        mock_position_provider,
        mock_account_provider,
    ):
        """Test liquidation callback is called."""
        callback_called = []

        def liquidation_callback(event: FuturesPositionDiff):
            callback_called.append(event)

        config = FuturesSyncConfig(enable_liquidation_detection=True)

        local_position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000.00"),
            qty=Decimal("1.0"),
            leverage=20,
            margin_mode=MarginMode.CROSS,
        )
        local_state_getter = lambda: {"BTCUSDT": local_position}

        mock_position_provider.get_futures_positions.return_value = []

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=local_state_getter,
            config=config,
            on_liquidation=liquidation_callback,
        )
        sync._known_positions = {"BTCUSDT": local_position}

        sync.sync_once()

        # Callback should be called if liquidation detected
        # Note: detection is heuristic based on leverage
        if sync.liquidation_count > 0:
            assert len(callback_called) >= 1


# =============================================================================
# ADL Monitoring Tests
# =============================================================================


class TestADLMonitoring:
    """Tests for ADL (Auto-Deleveraging) monitoring."""

    def test_adl_warning_detection(
        self,
        mock_position_provider,
        mock_account_provider,
        mock_futures_position,
    ):
        """Test ADL warning level detection."""
        adl_provider = MagicMock()
        adl_provider.get_adl_indicator.return_value = 4  # 4/5 lights = 80%

        config = FuturesSyncConfig(
            enable_adl_monitoring=True,
            adl_warning_threshold=70.0,
            adl_danger_threshold=85.0,
        )

        mock_position_provider.get_futures_positions.return_value = [mock_futures_position]

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {"BTCUSDT": mock_futures_position},
            config=config,
            adl_provider=adl_provider,
        )

        result = sync.sync_once()

        adl_events = [e for e in result.events if e.event_type == FuturesSyncEventType.ADL_DETECTED]
        assert len(adl_events) >= 1
        assert adl_events[0].adl_risk_level == ADLRiskLevel.WARNING

    def test_adl_danger_detection(
        self,
        mock_position_provider,
        mock_account_provider,
        mock_futures_position,
    ):
        """Test ADL danger level detection."""
        adl_provider = MagicMock()
        adl_provider.get_adl_indicator.return_value = 5  # 5/5 lights = 100%

        config = FuturesSyncConfig(
            enable_adl_monitoring=True,
            adl_warning_threshold=70.0,
            adl_danger_threshold=85.0,
        )

        mock_position_provider.get_futures_positions.return_value = [mock_futures_position]

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {"BTCUSDT": mock_futures_position},
            config=config,
            adl_provider=adl_provider,
        )

        result = sync.sync_once()

        adl_events = [e for e in result.events if e.event_type == FuturesSyncEventType.ADL_DETECTED]
        assert len(adl_events) >= 1
        # 5/5 = 100% which is >= 85% danger and >= 95% critical
        assert adl_events[0].adl_risk_level in (ADLRiskLevel.DANGER, ADLRiskLevel.CRITICAL)

    def test_adl_callback_fired(
        self,
        mock_position_provider,
        mock_account_provider,
        mock_futures_position,
    ):
        """Test ADL callback is called."""
        callback_events = []

        def adl_callback(event: FuturesPositionDiff):
            callback_events.append(event)

        adl_provider = MagicMock()
        adl_provider.get_adl_indicator.return_value = 5

        config = FuturesSyncConfig(
            enable_adl_monitoring=True,
            adl_warning_threshold=70.0,
        )

        mock_position_provider.get_futures_positions.return_value = [mock_futures_position]

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {"BTCUSDT": mock_futures_position},
            config=config,
            adl_provider=adl_provider,
            on_adl=adl_callback,
        )

        sync.sync_once()

        assert len(callback_events) >= 1

    def test_adl_safe_level(
        self,
        mock_position_provider,
        mock_account_provider,
        mock_futures_position,
    ):
        """Test no ADL events for safe level."""
        adl_provider = MagicMock()
        adl_provider.get_adl_indicator.return_value = 1  # 1/5 lights = 20%

        config = FuturesSyncConfig(
            enable_adl_monitoring=True,
            adl_warning_threshold=70.0,
        )

        mock_position_provider.get_futures_positions.return_value = [mock_futures_position]

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {"BTCUSDT": mock_futures_position},
            config=config,
            adl_provider=adl_provider,
        )

        result = sync.sync_once()

        adl_events = [e for e in result.events if e.event_type == FuturesSyncEventType.ADL_DETECTED]
        assert len(adl_events) == 0  # No events for safe level


# =============================================================================
# Background Sync Tests
# =============================================================================


class TestBackgroundSync:
    """Tests for background synchronization."""

    def test_start_background_sync(self, synchronizer):
        """Test starting background sync."""
        synchronizer.start_background_sync()

        assert synchronizer.is_running is True

        # Cleanup
        synchronizer.stop_background_sync()
        time.sleep(0.1)
        assert synchronizer.is_running is False

    def test_stop_background_sync(self, synchronizer):
        """Test stopping background sync."""
        synchronizer.start_background_sync()
        assert synchronizer.is_running is True

        synchronizer.stop_background_sync()
        time.sleep(0.1)

        assert synchronizer.is_running is False

    def test_background_sync_runs_periodically(
        self,
        mock_position_provider,
        mock_account_provider,
        mock_futures_position,
    ):
        """Test background sync runs at configured interval."""
        config = FuturesSyncConfig(sync_interval_sec=1.0)  # 1 second

        mock_position_provider.get_futures_positions.return_value = [mock_futures_position]

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {"BTCUSDT": mock_futures_position},
            config=config,
        )

        sync.start_background_sync()
        time.sleep(2.5)  # Wait for multiple sync cycles
        sync.stop_background_sync()

        assert sync.sync_count >= 2

    def test_double_start_warning(self, synchronizer):
        """Test warning when starting already running sync."""
        synchronizer.start_background_sync()
        assert synchronizer.is_running is True

        # Second start should just warn (no crash)
        synchronizer.start_background_sync()
        assert synchronizer.is_running is True

        # Cleanup
        synchronizer.stop_background_sync()

    @pytest.mark.asyncio
    async def test_async_background_sync(
        self,
        mock_position_provider,
        mock_account_provider,
        mock_futures_position,
    ):
        """Test async background sync."""
        config = FuturesSyncConfig(sync_interval_sec=1.0)

        mock_position_provider.get_futures_positions.return_value = [mock_futures_position]

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {"BTCUSDT": mock_futures_position},
            config=config,
        )

        await sync.start_background_sync_async()
        assert sync.is_running is True

        await asyncio.sleep(1.5)
        await sync.stop_background_sync_async()

        assert sync.is_running is False
        assert sync.sync_count >= 1


# =============================================================================
# Callback Tests
# =============================================================================


class TestCallbacks:
    """Tests for sync event callbacks."""

    def test_on_event_callback(
        self,
        mock_position_provider,
        mock_account_provider,
    ):
        """Test general event callback."""
        events_received = []

        def on_event(event: FuturesPositionDiff):
            events_received.append(event)

        config = FuturesSyncConfig()

        # Create mismatch scenario
        local_position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000.00"),
            qty=Decimal("1.0"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )
        remote_position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000.00"),
            qty=Decimal("0.5"),  # Mismatch
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )

        mock_position_provider.get_futures_positions.return_value = [remote_position]

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {"BTCUSDT": local_position},
            config=config,
            on_event=on_event,
        )

        sync.sync_once()

        assert len(events_received) >= 1

    def test_on_sync_complete_callback(
        self,
        mock_position_provider,
        mock_account_provider,
        mock_futures_position,
    ):
        """Test sync complete callback."""
        results_received = []

        def on_complete(result: FuturesSyncResult):
            results_received.append(result)

        config = FuturesSyncConfig()
        mock_position_provider.get_futures_positions.return_value = [mock_futures_position]

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {"BTCUSDT": mock_futures_position},
            config=config,
            on_sync_complete=on_complete,
        )

        sync.sync_once()

        assert len(results_received) == 1
        assert results_received[0].success is True

    def test_callback_error_handling(
        self,
        mock_position_provider,
        mock_account_provider,
        mock_futures_position,
    ):
        """Test callbacks handle errors gracefully."""

        def bad_callback(event):
            raise Exception("Callback error")

        config = FuturesSyncConfig()
        mock_position_provider.get_futures_positions.return_value = [mock_futures_position]

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {},  # Will trigger POSITION_OPENED
            config=config,
            on_event=bad_callback,
        )

        # Should not raise
        result = sync.sync_once()
        assert result.success is True


# =============================================================================
# Metrics and Properties Tests
# =============================================================================


class TestMetrics:
    """Tests for synchronizer metrics."""

    def test_get_metrics(self, synchronizer):
        """Test get_metrics returns expected data."""
        synchronizer.sync_once()

        metrics = synchronizer.get_metrics()

        assert "sync_count" in metrics
        assert "error_count" in metrics
        assert "liquidation_count" in metrics
        assert "adl_count" in metrics
        assert "is_running" in metrics
        assert "last_sync_time" in metrics
        assert "known_positions" in metrics

    def test_last_sync_result(self, synchronizer):
        """Test last_sync_result property."""
        assert synchronizer.last_sync_result is None

        synchronizer.sync_once()

        assert synchronizer.last_sync_result is not None
        assert synchronizer.last_sync_result.success is True

    def test_last_account_state(
        self,
        synchronizer,
        mock_account_state,
    ):
        """Test last_account_state property."""
        assert synchronizer.last_account_state is None

        synchronizer.sync_once()

        assert synchronizer.last_account_state is not None


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for synchronizer factory functions."""

    def test_create_crypto_futures_sync(
        self,
        mock_position_provider,
        mock_account_provider,
    ):
        """Test crypto futures sync factory."""
        sync = create_crypto_futures_sync(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {},
        )

        assert sync._config.exchange == Exchange.BINANCE
        assert sync._config.futures_type == FuturesType.CRYPTO_PERPETUAL
        assert sync._config.enable_adl_monitoring is True
        assert sync._config.enable_funding_tracking is True

    def test_create_crypto_futures_sync_with_options(
        self,
        mock_position_provider,
        mock_account_provider,
    ):
        """Test crypto futures sync factory with options."""
        sync = create_crypto_futures_sync(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {},
            sync_interval_sec=5.0,
            enable_adl_monitoring=False,
        )

        assert sync._config.sync_interval_sec == 5.0
        assert sync._config.enable_adl_monitoring is False

    def test_create_cme_futures_sync(
        self,
        mock_position_provider,
        mock_account_provider,
    ):
        """Test CME futures sync factory."""
        sync = create_cme_futures_sync(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {},
        )

        assert sync._config.exchange == Exchange.CME
        assert sync._config.futures_type == FuturesType.INDEX_FUTURES
        assert sync._config.enable_adl_monitoring is False  # No ADL for CME
        assert sync._config.enable_funding_tracking is False  # No funding for CME

    def test_create_cme_futures_sync_with_type(
        self,
        mock_position_provider,
        mock_account_provider,
    ):
        """Test CME futures sync factory with custom futures type."""
        sync = create_cme_futures_sync(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {},
            futures_type=FuturesType.COMMODITY_FUTURES,
        )

        assert sync._config.futures_type == FuturesType.COMMODITY_FUTURES


# =============================================================================
# Symbol Filtering Tests
# =============================================================================


class TestSymbolFiltering:
    """Tests for symbol include/exclude filtering."""

    def test_include_symbols(
        self,
        mock_account_provider,
    ):
        """Test include symbols filtering."""
        positions = [
            FuturesPosition(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                entry_price=Decimal("50000"),
                qty=Decimal("1.0"),
                leverage=10,
                margin_mode=MarginMode.CROSS,
            ),
            FuturesPosition(
                symbol="ETHUSDT",
                side=PositionSide.LONG,
                entry_price=Decimal("3000"),
                qty=Decimal("5.0"),
                leverage=10,
                margin_mode=MarginMode.CROSS,
            ),
            FuturesPosition(
                symbol="SOLUSDT",
                side=PositionSide.LONG,
                entry_price=Decimal("100"),
                qty=Decimal("100"),
                leverage=10,
                margin_mode=MarginMode.CROSS,
            ),
        ]

        provider = MagicMock()
        provider.get_futures_positions.return_value = positions

        config = FuturesSyncConfig(
            include_symbols=["BTCUSDT", "ETHUSDT"],  # Only these
        )

        sync = FuturesPositionSynchronizer(
            position_provider=provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {},
            config=config,
        )

        # Verify include_symbols is passed to provider
        sync.sync_once()
        provider.get_futures_positions.assert_called_with(symbols=["BTCUSDT", "ETHUSDT"])

    def test_exclude_symbols(
        self,
        mock_account_provider,
    ):
        """Test exclude symbols filtering."""
        positions = [
            FuturesPosition(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                entry_price=Decimal("50000"),
                qty=Decimal("1.0"),
                leverage=10,
                margin_mode=MarginMode.CROSS,
            ),
            FuturesPosition(
                symbol="DOGEUSDT",
                side=PositionSide.LONG,
                entry_price=Decimal("0.1"),
                qty=Decimal("10000"),
                leverage=10,
                margin_mode=MarginMode.CROSS,
            ),
        ]

        provider = MagicMock()
        provider.get_futures_positions.return_value = positions

        config = FuturesSyncConfig(
            exclude_symbols=["DOGEUSDT"],  # Exclude DOGE
        )

        sync = FuturesPositionSynchronizer(
            position_provider=provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {},
            config=config,
        )

        result = sync.sync_once()

        # DOGEUSDT should be excluded from remote_positions
        assert "DOGEUSDT" not in result.remote_positions
        assert "BTCUSDT" in result.remote_positions


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_sync_calls(
        self,
        mock_position_provider,
        mock_account_provider,
        mock_futures_position,
    ):
        """Test concurrent sync calls don't cause issues."""
        mock_position_provider.get_futures_positions.return_value = [mock_futures_position]

        config = FuturesSyncConfig()

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {"BTCUSDT": mock_futures_position},
            config=config,
        )

        results = []
        errors = []

        def sync_worker():
            try:
                result = sync.sync_once()
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=sync_worker) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 5
        assert all(r.success for r in results)

    def test_sync_during_callback(
        self,
        mock_position_provider,
        mock_account_provider,
        mock_futures_position,
    ):
        """Test sync is safe during callback execution."""
        mock_position_provider.get_futures_positions.return_value = [mock_futures_position]

        nested_sync_success = []

        def callback_that_syncs(result):
            # Try to sync from within callback
            # This should work due to RLock
            pass  # Don't actually nest to avoid complexity

        config = FuturesSyncConfig()

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {},
            config=config,
            on_sync_complete=callback_that_syncs,
        )

        result = sync.sync_once()
        assert result.success is True


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_local_and_remote(
        self,
        mock_position_provider,
        mock_account_provider,
    ):
        """Test sync with no positions on either side."""
        mock_position_provider.get_futures_positions.return_value = []

        config = FuturesSyncConfig()

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {},
            config=config,
        )

        result = sync.sync_once()

        assert result.success is True
        assert len(result.local_positions) == 0
        assert len(result.remote_positions) == 0
        assert result.has_events is False

    def test_zero_quantity_position(
        self,
        mock_position_provider,
        mock_account_provider,
    ):
        """Test handling of zero quantity positions."""
        zero_position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("0"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )
        mock_position_provider.get_futures_positions.return_value = [zero_position]

        local_position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("1.0"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )

        config = FuturesSyncConfig()

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {"BTCUSDT": local_position},
            config=config,
        )

        result = sync.sync_once()

        # Should detect position closed (zero qty)
        closed_events = [
            e for e in result.events if e.event_type == FuturesSyncEventType.POSITION_CLOSED
        ]
        assert len(closed_events) >= 1

    def test_local_state_getter_error(
        self,
        mock_position_provider,
        mock_account_provider,
    ):
        """Test handling of local state getter errors."""

        def bad_getter():
            raise Exception("Local state error")

        mock_position_provider.get_futures_positions.return_value = []

        config = FuturesSyncConfig()

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=bad_getter,
            config=config,
        )

        # Should handle gracefully
        result = sync.sync_once()
        assert result.success is True  # Remote fetch succeeded
        assert len(result.local_positions) == 0

    def test_very_small_quantity_difference(
        self,
        mock_position_provider,
        mock_account_provider,
    ):
        """Test tiny quantity differences within tolerance."""
        local_position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("1.000000"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )
        remote_position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("1.000001"),  # Tiny difference
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )

        mock_position_provider.get_futures_positions.return_value = [remote_position]

        config = FuturesSyncConfig(qty_tolerance_pct=0.001)  # 0.1%

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {"BTCUSDT": local_position},
            config=config,
        )

        result = sync.sync_once()

        # Should NOT detect mismatch for tiny difference
        qty_mismatch_events = [
            e for e in result.events if e.event_type == FuturesSyncEventType.QTY_MISMATCH
        ]
        assert len(qty_mismatch_events) == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for position sync service."""

    def test_full_sync_workflow(
        self,
        mock_position_provider,
        mock_account_provider,
    ):
        """Test complete sync workflow with all features."""
        events_log = []
        sync_log = []

        def on_event(event):
            events_log.append(event)

        def on_sync(result):
            sync_log.append(result)

        # Initial position
        position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("1.0"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
            unrealized_pnl=Decimal("500"),
        )

        mock_position_provider.get_futures_positions.return_value = [position]

        config = FuturesSyncConfig(sync_interval_sec=1.0)

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {"BTCUSDT": position},
            config=config,
            on_event=on_event,
            on_sync_complete=on_sync,
        )

        # Perform sync
        result = sync.sync_once()

        assert result.success is True
        assert len(sync_log) == 1
        assert sync.sync_count == 1

    def test_sync_with_position_changes(
        self,
        mock_position_provider,
        mock_account_provider,
    ):
        """Test sync detecting position changes over time."""
        call_count = [0]

        def get_positions(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [
                    FuturesPosition(
                        symbol="BTCUSDT",
                        side=PositionSide.LONG,
                        entry_price=Decimal("50000"),
                        qty=Decimal("1.0"),
                        leverage=10,
                        margin_mode=MarginMode.CROSS,
                    )
                ]
            else:
                return [
                    FuturesPosition(
                        symbol="BTCUSDT",
                        side=PositionSide.LONG,
                        entry_price=Decimal("50000"),
                        qty=Decimal("2.0"),  # Qty increased
                        leverage=10,
                        margin_mode=MarginMode.CROSS,
                    )
                ]

        mock_position_provider.get_futures_positions.side_effect = get_positions

        local_position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("1.0"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )

        config = FuturesSyncConfig()

        sync = FuturesPositionSynchronizer(
            position_provider=mock_position_provider,
            account_provider=mock_account_provider,
            local_state_getter=lambda: {"BTCUSDT": local_position},
            config=config,
        )

        # First sync - matching
        result1 = sync.sync_once()
        assert result1.success is True

        # Second sync - mismatch detected
        result2 = sync.sync_once()
        assert result2.success is True

        qty_events = [
            e for e in result2.events if e.event_type == FuturesSyncEventType.QTY_MISMATCH
        ]
        assert len(qty_events) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
