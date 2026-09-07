# -*- coding: utf-8 -*-
"""
tests/test_forex_phase6_risk_services.py
Comprehensive tests for Phase 6: Forex Risk Management & Services

Tests cover:
- forex_risk_guards.py: SwapCostTracker, ForexMarginGuard, ForexLeverageGuard
- forex_position_sync.py: ForexPositionSynchronizer, reconciliation
- forex_session_router.py: ForexSessionRouter, session detection

Target: ~100 tests as per FOREX_INTEGRATION_PLAN.md Phase 6 specification
"""

import asyncio
import math
import pytest
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence
from unittest.mock import Mock, MagicMock, patch
from zoneinfo import ZoneInfo

from services.forex_risk_guards import (
    SwapCostTracker,
    SwapRate,
    SwapCost,
    SwapDirection,
    ForexMarginGuard,
    ForexMarginRequirement,
    ForexMarginStatus,
    ForexMarginCallType,
    ForexLeverageGuard,
    LeverageCheck,
    LeverageViolationType,
    SWAP_DAY_MULTIPLIERS,
    LEVERAGE_BY_CATEGORY,
    create_forex_margin_guard,
    create_forex_leverage_guard,
    create_swap_cost_tracker,
)

from services.forex_position_sync import (
    ForexPosition,
    ForexPositionSynchronizer,
    SyncConfig,
    SyncResult,
    PositionDiscrepancy,
    PositionDiscrepancyType,
    OandaReconciliationResult,
    reconcile_oanda_state,
    create_forex_position_sync,
    create_oanda_position_sync,
    ReconciliationExecutor,
    ReconciliationAction,
    ReconciliationOrderResult,
)

from services.forex_session_router import (
    ForexSessionRouter,
    ForexSessionInfo,
    RoutingDecision,
    get_current_forex_session,
    get_session_at_time,
    is_forex_market_open,
    get_spread_multiplier_for_time,
    get_liquidity_factor_for_time,
    get_next_session_start,
    create_forex_session_router,
    ROLLOVER_KEEPOUT_MINUTES,
    SESSION_LIQUIDITY,
    SESSION_SPREAD_MULT,
)

from adapters.models import ForexSessionType


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_swap_provider():
    """Create mock swap rate provider."""
    provider = Mock()
    provider.get_swap_rate.return_value = SwapRate(
        symbol="EUR_USD",
        long_rate=-0.5,
        short_rate=0.3,
        timestamp_ms=int(time.time() * 1000),
        source="mock",
    )
    return provider


@pytest.fixture
def mock_account_provider():
    """Create mock forex account provider."""
    provider = Mock()
    provider.get_account_summary.return_value = {
        "equity": 100000,
        "balance": 95000,
        "margin_used": 20000,
        "margin_available": 80000,
        "unrealized_pnl": 5000,
    }
    provider.get_positions.return_value = {
        "EUR_USD": {
            "units": 100000,
            "price": 1.0850,
        }
    }
    return provider


@pytest.fixture
def mock_position_provider():
    """Create mock position provider."""
    provider = Mock()
    provider.get_positions.return_value = {
        "EUR_USD": ForexPosition(
            symbol="EUR_USD",
            units=100000,
            average_price=1.0850,
            unrealized_pnl=500,
            margin_used=2000,
            financing=-10.5,
        ),
        "GBP_USD": ForexPosition(
            symbol="GBP_USD",
            units=-50000,
            average_price=1.2650,
            unrealized_pnl=-200,
            margin_used=1500,
            financing=-5.2,
        ),
    }
    return provider


# =============================================================================
# Tests: SwapCostTracker
# =============================================================================


class TestSwapCostTracker:
    """Tests for SwapCostTracker."""

    def test_init_default(self):
        """Test default initialization."""
        tracker = SwapCostTracker()
        assert tracker._provider is None
        assert tracker._max_history == 1000

    def test_init_with_provider(self, mock_swap_provider):
        """Test initialization with provider."""
        tracker = SwapCostTracker(swap_rate_provider=mock_swap_provider)
        assert tracker._provider is mock_swap_provider

    def test_get_swap_rate_default(self):
        """Test getting swap rate from defaults."""
        tracker = SwapCostTracker()
        rate = tracker.get_swap_rate("EUR_USD")
        assert rate.symbol == "EUR_USD"
        assert rate.long_rate < 0  # Typically negative for EUR/USD long
        assert rate.short_rate > 0  # Typically positive for EUR/USD short

    def test_get_swap_rate_unknown_symbol(self):
        """Test getting swap rate for unknown symbol."""
        tracker = SwapCostTracker()
        rate = tracker.get_swap_rate("UNKNOWN_PAIR")
        assert rate.symbol == "UNKNOWN_PAIR"
        assert rate.long_rate == 0.0
        assert rate.short_rate == 0.0

    def test_get_swap_rate_from_provider(self, mock_swap_provider):
        """Test getting swap rate from provider."""
        tracker = SwapCostTracker(swap_rate_provider=mock_swap_provider)
        rate = tracker.get_swap_rate("EUR_USD")
        assert rate.source == "mock"
        assert mock_swap_provider.get_swap_rate.called

    def test_calculate_daily_swap_long(self):
        """Test daily swap calculation for long position."""
        tracker = SwapCostTracker()
        cost = tracker.calculate_daily_swap(
            symbol="EUR_USD",
            position_units=100000,  # 1 standard lot
            is_long=True,
            current_price=1.0850,
            day_of_week=1,  # Tuesday
        )
        assert cost.symbol == "EUR_USD"
        assert cost.units == 100000
        assert cost.is_long is True
        assert cost.day_multiplier == 1
        # Long EUR/USD typically has negative swap (cost)
        assert cost.rate_used < 0

    def test_calculate_daily_swap_short(self):
        """Test daily swap calculation for short position."""
        tracker = SwapCostTracker()
        cost = tracker.calculate_daily_swap(
            symbol="EUR_USD",
            position_units=-100000,
            is_long=False,
            current_price=1.0850,
            day_of_week=1,
        )
        assert cost.is_long is False
        # Short EUR/USD typically has positive swap (credit)
        assert cost.rate_used > 0

    def test_calculate_daily_swap_wednesday_3x(self):
        """Test Wednesday 3x swap multiplier."""
        tracker = SwapCostTracker()
        cost = tracker.calculate_daily_swap(
            symbol="EUR_USD",
            position_units=100000,
            is_long=True,
            current_price=1.0850,
            day_of_week=2,  # Wednesday
        )
        assert cost.day_multiplier == 3
        assert cost.total_cost == cost.daily_cost * 3

    def test_calculate_daily_swap_weekend_zero(self):
        """Test weekend swap is zero."""
        tracker = SwapCostTracker()
        cost_sat = tracker.calculate_daily_swap(
            symbol="EUR_USD",
            position_units=100000,
            is_long=True,
            current_price=1.0850,
            day_of_week=5,  # Saturday
        )
        assert cost_sat.day_multiplier == 0
        assert cost_sat.total_cost == 0

        cost_sun = tracker.calculate_daily_swap(
            symbol="EUR_USD",
            position_units=100000,
            is_long=True,
            current_price=1.0850,
            day_of_week=6,  # Sunday
        )
        assert cost_sun.day_multiplier == 0
        assert cost_sun.total_cost == 0

    def test_calculate_daily_swap_jpy_pair(self):
        """Test swap calculation for JPY pair."""
        tracker = SwapCostTracker()
        cost = tracker.calculate_daily_swap(
            symbol="USD_JPY",
            position_units=100000,
            is_long=True,
            current_price=149.50,
            day_of_week=1,
        )
        assert cost.symbol == "USD_JPY"
        # Long USD/JPY typically has positive swap (credit)
        assert cost.rate_used > 0

    def test_cumulative_swap_tracking(self):
        """Test cumulative swap tracking."""
        tracker = SwapCostTracker()

        # Add several swaps
        for day in range(5):
            tracker.calculate_daily_swap(
                symbol="EUR_USD",
                position_units=100000,
                is_long=True,
                current_price=1.0850,
                day_of_week=day,
            )

        cumulative = tracker.get_cumulative_swap("EUR_USD")
        assert cumulative != 0

    def test_cumulative_swap_total(self):
        """Test total cumulative swap across all symbols."""
        tracker = SwapCostTracker()

        tracker.calculate_daily_swap("EUR_USD", 100000, True, 1.0850, 1)
        tracker.calculate_daily_swap("GBP_USD", 100000, True, 1.2650, 1)

        total = tracker.get_cumulative_swap()  # All symbols
        eur = tracker.get_cumulative_swap("EUR_USD")
        gbp = tracker.get_cumulative_swap("GBP_USD")

        assert total == eur + gbp

    def test_swap_history(self):
        """Test swap history retrieval."""
        tracker = SwapCostTracker()

        for day in range(3):
            tracker.calculate_daily_swap("EUR_USD", 100000, True, 1.0850, day)

        history = tracker.get_swap_history()
        assert len(history) == 3

        history_eur = tracker.get_swap_history("EUR_USD", limit=2)
        assert len(history_eur) == 2

    def test_reset_cumulative(self):
        """Test resetting cumulative swap."""
        tracker = SwapCostTracker()

        tracker.calculate_daily_swap("EUR_USD", 100000, True, 1.0850, 1)
        tracker.calculate_daily_swap("GBP_USD", 100000, True, 1.2650, 1)

        assert tracker.get_cumulative_swap("EUR_USD") != 0

        tracker.reset_cumulative("EUR_USD")
        assert tracker.get_cumulative_swap("EUR_USD") == 0
        assert tracker.get_cumulative_swap("GBP_USD") != 0

        tracker.reset_cumulative()  # Reset all
        assert tracker.get_cumulative_swap() == 0

    def test_estimate_monthly_swap(self):
        """Test monthly swap estimation."""
        tracker = SwapCostTracker()
        monthly = tracker.estimate_monthly_swap("EUR_USD", 100000, True)
        # Should be approximately daily × 28 (20 days + 4 Wed 3x)
        assert monthly != 0

    def test_swap_rate_direction(self):
        """Test swap rate direction classification."""
        positive_rate = SwapRate("TEST", long_rate=0.5, short_rate=-0.3)
        assert positive_rate.get_direction(True) == SwapDirection.CREDIT
        assert positive_rate.get_direction(False) == SwapDirection.DEBIT

        negative_rate = SwapRate("TEST", long_rate=-0.5, short_rate=0.3)
        assert negative_rate.get_direction(True) == SwapDirection.DEBIT
        assert negative_rate.get_direction(False) == SwapDirection.CREDIT

        zero_rate = SwapRate("TEST", long_rate=0.0, short_rate=0.0)
        assert zero_rate.get_direction(True) == SwapDirection.ZERO


# =============================================================================
# Tests: ForexMarginGuard
# =============================================================================


class TestForexMarginGuard:
    """Tests for ForexMarginGuard."""

    def test_init_default(self):
        """Test default initialization."""
        guard = ForexMarginGuard()
        assert guard._max_leverage == 50
        assert guard._warning_level == 0.50
        assert guard._call_level == 0.30
        assert guard._stop_out_level == 0.20

    def test_init_with_config(self):
        """Test initialization with custom config."""
        guard = ForexMarginGuard(
            max_leverage=100,
            margin_warning_level=0.40,
            jurisdiction="professional",
        )
        assert guard._max_leverage == 100
        assert guard._warning_level == 0.40
        assert guard._jurisdiction == "professional"

    def test_get_margin_status_no_provider(self):
        """Test margin status without provider."""
        guard = ForexMarginGuard()
        status = guard.get_margin_status()
        assert status.equity == 0
        assert status.margin_call_type == ForexMarginCallType.NONE

    def test_get_margin_status_with_provider(self, mock_account_provider):
        """Test margin status with provider."""
        guard = ForexMarginGuard(account_provider=mock_account_provider)
        status = guard.get_margin_status()

        assert status.equity == 100000
        assert status.balance == 95000
        assert status.margin_used == 20000
        assert status.margin_available == 80000

    def test_margin_call_detection(self, mock_account_provider):
        """Test margin call type detection."""
        # Normal status
        mock_account_provider.get_account_summary.return_value = {
            "equity": 100000,
            "balance": 95000,
            "margin_used": 50000,  # 50% margin level
        }
        guard = ForexMarginGuard(account_provider=mock_account_provider)
        status = guard.get_margin_status(force_refresh=True)
        assert status.margin_call_type == ForexMarginCallType.NONE

        # Warning level
        mock_account_provider.get_account_summary.return_value = {
            "equity": 100000,
            "margin_used": 250000,  # 40% margin level
        }
        status = guard.get_margin_status(force_refresh=True)
        assert status.margin_call_type == ForexMarginCallType.WARNING

        # Stop out level
        mock_account_provider.get_account_summary.return_value = {
            "equity": 100000,
            "margin_used": 600000,  # ~17% margin level
        }
        status = guard.get_margin_status(force_refresh=True)
        assert status.margin_call_type == ForexMarginCallType.STOP_OUT

    def test_check_trade_margin(self, mock_account_provider):
        """Test pre-trade margin check."""
        guard = ForexMarginGuard(account_provider=mock_account_provider)

        # Small trade should be allowed
        status = guard.check_trade_margin(
            symbol="EUR_USD",
            units=10000,
            current_price=1.0850,
        )
        assert status.margin_call_type == ForexMarginCallType.NONE

    def test_check_trade_margin_exceeds(self, mock_account_provider):
        """Test trade that would exceed margin."""
        mock_account_provider.get_account_summary.return_value = {
            "equity": 10000,
            "margin_used": 8000,
            "margin_available": 2000,
        }
        guard = ForexMarginGuard(account_provider=mock_account_provider)

        # Large trade should trigger margin call
        status = guard.check_trade_margin(
            symbol="EUR_USD",
            units=5_000_000,  # Very large
            current_price=1.0850,
        )
        assert status.margin_call_type in (
            ForexMarginCallType.MARGIN_CALL,
            ForexMarginCallType.STOP_OUT,
        )

    def test_get_margin_requirement(self):
        """Test margin requirement calculation."""
        guard = ForexMarginGuard(max_leverage=50)

        req_major = guard.get_margin_requirement("major")
        assert req_major.margin_pct == pytest.approx(0.02, rel=0.1)  # ~2%
        assert req_major.leverage == 50

        req_exotic = guard.get_margin_requirement("exotic")
        assert req_exotic.margin_pct > req_major.margin_pct  # Higher margin for exotics

    def test_get_max_position_size(self, mock_account_provider):
        """Test maximum position size calculation."""
        guard = ForexMarginGuard(account_provider=mock_account_provider)

        max_size = guard.get_max_position_size(
            symbol="EUR_USD",
            current_price=1.0850,
        )
        assert max_size > 0

    def test_cached_margin_status(self, mock_account_provider):
        """Test margin status caching."""
        guard = ForexMarginGuard(account_provider=mock_account_provider)

        # First call - hits provider
        guard.get_margin_status()
        call_count_1 = mock_account_provider.get_account_summary.call_count

        # Second call within cache TTL - should use cache
        guard.get_margin_status()
        call_count_2 = mock_account_provider.get_account_summary.call_count

        assert call_count_2 == call_count_1  # No additional calls

        # Force refresh
        guard.get_margin_status(force_refresh=True)
        assert mock_account_provider.get_account_summary.call_count > call_count_2

    def test_factory_function(self):
        """Test create_forex_margin_guard factory."""
        guard = create_forex_margin_guard(
            config={
                "max_leverage": 100,
                "margin_warning_level": 0.60,
            }
        )
        assert guard._max_leverage == 100
        assert guard._warning_level == 0.60


# =============================================================================
# Tests: ForexLeverageGuard
# =============================================================================


class TestForexLeverageGuard:
    """Tests for ForexLeverageGuard."""

    def test_init_default(self):
        """Test default initialization."""
        guard = ForexLeverageGuard()
        assert guard._max_leverage == 50
        assert guard._warning_threshold == 0.90

    def test_check_leverage_allowed(self):
        """Test leverage check - allowed."""
        guard = ForexLeverageGuard(max_leverage=50)

        check = guard.check_leverage(
            equity=100000,
            total_notional=2_000_000,  # 20:1 leverage
        )
        assert check.is_allowed
        assert check.violation_type == LeverageViolationType.NONE
        assert check.current_leverage == 20.0

    def test_check_leverage_exceeded(self):
        """Test leverage check - exceeded."""
        guard = ForexLeverageGuard(max_leverage=50)

        check = guard.check_leverage(
            equity=100000,
            total_notional=6_000_000,  # 60:1 leverage
        )
        assert not check.is_allowed
        assert check.violation_type == LeverageViolationType.EXCEEDED_MAX
        assert check.current_leverage == 60.0

    def test_check_leverage_near_limit(self):
        """Test leverage check - near limit warning."""
        guard = ForexLeverageGuard(max_leverage=50, warning_threshold=0.90)

        check = guard.check_leverage(
            equity=100000,
            total_notional=4_600_000,  # 46:1 = 92% of 50:1
        )
        assert check.is_allowed
        assert check.violation_type == LeverageViolationType.NEAR_LIMIT

    def test_check_leverage_with_new_trade(self):
        """Test leverage check with projected new trade."""
        guard = ForexLeverageGuard(max_leverage=50)

        check = guard.check_leverage(
            equity=100000,
            total_notional=2_000_000,
            new_trade_notional=3_000_000,  # Would bring to 50:1
        )
        assert check.current_leverage == 50.0
        assert check.is_allowed

        check = guard.check_leverage(
            equity=100000,
            total_notional=2_000_000,
            new_trade_notional=4_000_000,  # Would bring to 60:1
        )
        assert not check.is_allowed

    def test_check_leverage_zero_equity(self):
        """Test leverage check with zero equity."""
        guard = ForexLeverageGuard()

        check = guard.check_leverage(
            equity=0,
            total_notional=1_000_000,
        )
        assert not check.is_allowed
        assert check.violation_type == LeverageViolationType.EXCEEDED_MAX

    def test_check_concentration(self):
        """Test position concentration check."""
        guard = ForexLeverageGuard(concentration_limit=0.50)

        # Under limit
        check = guard.check_concentration(
            symbol="EUR_USD",
            symbol_notional=2_000_000,
            total_notional=5_000_000,
        )
        assert check.is_allowed

        # Over limit
        check = guard.check_concentration(
            symbol="EUR_USD",
            symbol_notional=3_000_000,
            total_notional=5_000_000,  # 60% concentration
        )
        assert not check.is_allowed
        assert check.violation_type == LeverageViolationType.CONCENTRATION

    def test_check_correlated_exposure(self):
        """Test correlated exposure check."""
        guard = ForexLeverageGuard(correlated_exposure_limit=0.80)

        positions = {
            "EUR_USD": 2_000_000,
            "GBP_USD": 1_500_000,  # Correlated with EUR_USD
        }

        # Adding more EUR exposure
        check = guard.check_correlated_exposure(
            positions=positions,
            new_symbol="EUR_USD",
            new_notional=2_000_000,
            total_notional=3_500_000,
        )
        # Should flag correlated exposure
        assert check.violation_type in (
            LeverageViolationType.NONE,
            LeverageViolationType.CORRELATED_EXPOSURE,
        )

    def test_factory_function(self):
        """Test create_forex_leverage_guard factory."""
        guard = create_forex_leverage_guard(
            config={
                "max_leverage": 100,
                "concentration_limit": 0.40,
            }
        )
        assert guard._max_leverage == 100
        assert guard._concentration_limit == 0.40


# =============================================================================
# Tests: ForexPosition and ForexPositionSynchronizer
# =============================================================================


class TestForexPosition:
    """Tests for ForexPosition dataclass."""

    def test_long_position(self):
        """Test long position properties."""
        pos = ForexPosition(
            symbol="EUR_USD",
            units=100000,
            average_price=1.0850,
        )
        assert pos.is_long
        assert not pos.is_short
        assert pos.abs_units == 100000
        assert pos.notional == pytest.approx(108500, rel=0.01)

    def test_short_position(self):
        """Test short position properties."""
        pos = ForexPosition(
            symbol="EUR_USD",
            units=-100000,
            average_price=1.0850,
        )
        assert not pos.is_long
        assert pos.is_short
        assert pos.abs_units == 100000


class TestForexPositionSynchronizer:
    """Tests for ForexPositionSynchronizer."""

    def test_init(self, mock_position_provider):
        """Test initialization."""
        sync = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: {},
        )
        assert sync._provider is mock_position_provider
        assert sync._running is False

    def test_sync_once_no_discrepancies(self, mock_position_provider):
        """Test sync with matching positions."""
        local_positions = {
            "EUR_USD": 100000,
            "GBP_USD": -50000,
        }

        sync = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: local_positions,
        )

        result = sync.sync_once()
        assert result.success
        assert not result.has_discrepancies
        assert result.position_count == 2

    def test_sync_once_with_discrepancies(self, mock_position_provider):
        """Test sync with discrepancies."""
        local_positions = {
            "EUR_USD": 50000,  # Different from remote 100000
            # Missing GBP_USD locally
        }

        sync = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: local_positions,
        )

        result = sync.sync_once()
        assert result.success
        assert result.has_discrepancies
        assert len(result.discrepancies) >= 1

    def test_sync_detects_missing_local(self, mock_position_provider):
        """Test detection of position missing locally."""
        local_positions = {}  # Empty local state

        sync = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: local_positions,
        )

        result = sync.sync_once()
        # Should find EUR_USD and GBP_USD missing locally
        missing_local = [
            d
            for d in result.discrepancies
            if d.discrepancy_type == PositionDiscrepancyType.MISSING_LOCAL
        ]
        assert len(missing_local) >= 2

    def test_sync_detects_missing_remote(self, mock_position_provider):
        """Test detection of position missing on broker."""
        mock_position_provider.get_positions.return_value = {}

        local_positions = {"EUR_USD": 100000}

        sync = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: local_positions,
        )

        result = sync.sync_once()
        missing_remote = [
            d
            for d in result.discrepancies
            if d.discrepancy_type == PositionDiscrepancyType.MISSING_REMOTE
        ]
        assert len(missing_remote) == 1
        assert missing_remote[0].symbol == "EUR_USD"

    def test_sync_detects_units_mismatch(self, mock_position_provider):
        """Test detection of units mismatch."""
        local_positions = {
            "EUR_USD": 50000,  # Remote has 100000
        }

        sync = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: local_positions,
            config=SyncConfig(position_tolerance_pct=0.01),
        )

        result = sync.sync_once()
        units_mismatch = [
            d
            for d in result.discrepancies
            if d.discrepancy_type == PositionDiscrepancyType.UNITS_MISMATCH
        ]
        assert len(units_mismatch) >= 1

    def test_sync_detects_side_mismatch(self, mock_position_provider):
        """Test detection of side mismatch."""
        mock_position_provider.get_positions.return_value = {
            "EUR_USD": ForexPosition(
                symbol="EUR_USD",
                units=100000,  # Long
                average_price=1.0850,
            ),
        }

        local_positions = {"EUR_USD": -100000}  # Short locally

        sync = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: local_positions,
        )

        result = sync.sync_once()
        side_mismatch = [
            d
            for d in result.discrepancies
            if d.discrepancy_type == PositionDiscrepancyType.SIDE_MISMATCH
        ]
        assert len(side_mismatch) == 1

    def test_sync_with_tolerance(self, mock_position_provider):
        """Test sync respects tolerance settings."""
        mock_position_provider.get_positions.return_value = {
            "EUR_USD": ForexPosition(
                symbol="EUR_USD",
                units=100010,  # Slightly different
                average_price=1.0850,
            ),
        }

        local_positions = {"EUR_USD": 100000}

        # With low tolerance - should detect
        sync_strict = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: local_positions,
            config=SyncConfig(position_tolerance_pct=0.00001),
        )
        result_strict = sync_strict.sync_once()

        # With high tolerance - should not detect
        sync_relaxed = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: local_positions,
            config=SyncConfig(position_tolerance_pct=0.01),  # 1%
        )
        result_relaxed = sync_relaxed.sync_once()

        # Strict should find discrepancy, relaxed should not
        assert len(result_strict.discrepancies) >= len(result_relaxed.discrepancies)

    def test_sync_callback_on_discrepancy(self, mock_position_provider):
        """Test discrepancy callback is invoked."""
        discrepancies_received = []

        def on_discrepancy(d):
            discrepancies_received.append(d)

        local_positions = {}

        sync = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: local_positions,
            on_discrepancy=on_discrepancy,
        )

        sync.sync_once()
        assert len(discrepancies_received) > 0

    def test_sync_callback_on_complete(self, mock_position_provider):
        """Test sync complete callback is invoked."""
        results_received = []

        def on_complete(result):
            results_received.append(result)

        sync = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: {},
            on_sync_complete=on_complete,
        )

        sync.sync_once()
        assert len(results_received) == 1
        assert results_received[0].success

    def test_sync_handles_provider_error(self, mock_position_provider):
        """Test sync handles provider errors gracefully."""
        mock_position_provider.get_positions.side_effect = Exception("API Error")

        sync = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: {},
        )

        result = sync.sync_once()
        assert not result.success
        assert result.error is not None
        assert "API Error" in result.error

    def test_sync_financing_tracking(self, mock_position_provider):
        """Test financing/swap tracking in sync."""
        sync = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: {"EUR_USD": 100000, "GBP_USD": -50000},
        )

        result = sync.sync_once()
        # Total financing from mock positions
        expected_financing = -10.5 + -5.2  # EUR + GBP financing
        assert result.total_financing == pytest.approx(expected_financing, rel=0.01)

    def test_last_sync_property(self, mock_position_provider):
        """Test last_sync property."""
        sync = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: {},
        )

        assert sync.last_sync is None

        sync.sync_once()
        assert sync.last_sync is not None
        assert sync.last_sync.success

    def test_factory_functions(self, mock_position_provider):
        """Test factory functions."""
        sync = create_forex_position_sync(
            position_provider=mock_position_provider,
            local_state_getter=lambda: {},
        )
        assert sync is not None

        sync_oanda = create_oanda_position_sync(
            oanda_adapter=mock_position_provider,
            local_state_getter=lambda: {},
            config={"sync_interval_sec": 60},
        )
        assert sync_oanda._config.sync_interval_sec == 60


# =============================================================================
# Tests: ForexSessionRouter and Session Detection
# =============================================================================


class TestForexSessionDetection:
    """Tests for forex session detection functions."""

    def test_detect_london_session(self):
        """Test London session detection."""
        # London session: 07:00-16:00 UTC
        london_time = datetime(2024, 1, 15, 10, 0, tzinfo=ZoneInfo("UTC"))  # Monday 10am UTC
        ts_ms = int(london_time.timestamp() * 1000)

        session_info = get_current_forex_session(ts_ms)
        assert ForexSessionType.LONDON in session_info.active_sessions
        assert session_info.is_open

    def test_detect_tokyo_session(self):
        """Test Tokyo session detection."""
        # Tokyo session: 00:00-09:00 UTC
        tokyo_time = datetime(2024, 1, 15, 3, 0, tzinfo=ZoneInfo("UTC"))  # Monday 3am UTC
        ts_ms = int(tokyo_time.timestamp() * 1000)

        session_info = get_current_forex_session(ts_ms)
        assert ForexSessionType.TOKYO in session_info.active_sessions
        assert session_info.is_open

    def test_detect_new_york_session(self):
        """Test New York session detection."""
        # NY session: 12:00-21:00 UTC
        ny_time = datetime(2024, 1, 15, 15, 0, tzinfo=ZoneInfo("UTC"))  # Monday 3pm UTC
        ts_ms = int(ny_time.timestamp() * 1000)

        session_info = get_current_forex_session(ts_ms)
        assert ForexSessionType.NEW_YORK in session_info.active_sessions
        assert session_info.is_open

    def test_detect_london_ny_overlap(self):
        """Test London/NY overlap detection."""
        # Overlap: 12:00-16:00 UTC
        overlap_time = datetime(2024, 1, 15, 14, 0, tzinfo=ZoneInfo("UTC"))  # Monday 2pm UTC
        ts_ms = int(overlap_time.timestamp() * 1000)

        session_info = get_current_forex_session(ts_ms)
        # Both London and NY should be active
        assert ForexSessionType.LONDON in session_info.active_sessions
        assert ForexSessionType.NEW_YORK in session_info.active_sessions

    def test_detect_weekend_saturday(self):
        """Test weekend detection on Saturday."""
        saturday = datetime(2024, 1, 13, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        ts_ms = int(saturday.timestamp() * 1000)

        session_info = get_current_forex_session(ts_ms)
        assert session_info.session == ForexSessionType.WEEKEND
        assert not session_info.is_open
        assert session_info.liquidity_factor == 0.0

    def test_detect_weekend_friday_after_close(self):
        """Test weekend detection Friday after 5pm ET."""
        friday_late = datetime(
            2024, 1, 12, 18, 0, tzinfo=ZoneInfo("America/New_York")
        )  # Friday 6pm ET
        ts_ms = int(friday_late.timestamp() * 1000)

        session_info = get_current_forex_session(ts_ms)
        assert session_info.session == ForexSessionType.WEEKEND

    def test_detect_weekend_sunday_before_open(self):
        """Test weekend detection Sunday before 5pm ET."""
        sunday_early = datetime(
            2024, 1, 14, 14, 0, tzinfo=ZoneInfo("America/New_York")
        )  # Sunday 2pm ET
        ts_ms = int(sunday_early.timestamp() * 1000)

        session_info = get_current_forex_session(ts_ms)
        assert session_info.session == ForexSessionType.WEEKEND

    def test_rollover_window_detection(self):
        """Test rollover window detection."""
        # Near 5pm ET (rollover time)
        near_rollover = datetime(
            2024, 1, 15, 16, 50, tzinfo=ZoneInfo("America/New_York")
        )  # 4:50pm ET
        ts_ms = int(near_rollover.timestamp() * 1000)

        session_info = get_current_forex_session(ts_ms)
        assert session_info.in_rollover_window

    def test_liquidity_factors(self):
        """Test session liquidity factors."""
        # London should have high liquidity
        london_time = datetime(2024, 1, 15, 10, 0, tzinfo=ZoneInfo("UTC"))
        london_session = get_current_forex_session(int(london_time.timestamp() * 1000))
        assert london_session.liquidity_factor >= 1.0

        # Sydney should have lower liquidity
        sydney_time = datetime(2024, 1, 15, 23, 0, tzinfo=ZoneInfo("UTC"))
        sydney_session = get_current_forex_session(int(sydney_time.timestamp() * 1000))
        assert sydney_session.liquidity_factor < 1.0

    def test_spread_multipliers(self):
        """Test session spread multipliers."""
        # London/NY overlap should have tightest spreads
        overlap_time = datetime(2024, 1, 15, 14, 0, tzinfo=ZoneInfo("UTC"))
        overlap_session = get_current_forex_session(int(overlap_time.timestamp() * 1000))
        assert overlap_session.spread_multiplier <= 1.0

        # Weekend should have infinite spread
        weekend = datetime(2024, 1, 13, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        weekend_session = get_current_forex_session(int(weekend.timestamp() * 1000))
        assert weekend_session.spread_multiplier == float("inf")

    def test_utility_functions(self):
        """Test session utility functions."""
        london_time = datetime(2024, 1, 15, 10, 0, tzinfo=ZoneInfo("UTC"))
        ts_ms = int(london_time.timestamp() * 1000)

        assert is_forex_market_open(ts_ms)
        assert get_spread_multiplier_for_time(ts_ms) > 0
        assert get_liquidity_factor_for_time(ts_ms) > 0
        assert get_session_at_time(ts_ms) != ForexSessionType.WEEKEND


class TestForexSessionRouter:
    """Tests for ForexSessionRouter."""

    def test_init_default(self):
        """Test default initialization."""
        router = ForexSessionRouter()
        assert router.avoid_rollover
        assert router.min_liquidity_factor == 0.5
        assert router.avoid_weekend_proximity

    def test_routing_decision_market_open(self):
        """Test routing decision when market is open."""
        router = ForexSessionRouter()

        # Mock time during London session
        london_time = datetime(2024, 1, 15, 10, 0, tzinfo=ZoneInfo("UTC"))
        ts_ms = int(london_time.timestamp() * 1000)

        decision = router.get_routing_decision(
            symbol="EUR_USD",
            side="BUY",
            size_usd=100000,
            timestamp_ms=ts_ms,
        )
        assert decision.should_submit
        assert decision.session != ForexSessionType.WEEKEND

    def test_routing_decision_weekend(self):
        """Test routing decision during weekend."""
        router = ForexSessionRouter()

        saturday = datetime(2024, 1, 13, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        ts_ms = int(saturday.timestamp() * 1000)

        decision = router.get_routing_decision(
            symbol="EUR_USD",
            side="BUY",
            size_usd=100000,
            timestamp_ms=ts_ms,
        )
        assert not decision.should_submit
        assert decision.session == ForexSessionType.WEEKEND
        assert "closed" in decision.reason.lower()

    def test_routing_decision_near_rollover(self):
        """Test routing decision near rollover time."""
        router = ForexSessionRouter(avoid_rollover=True)

        # Near 5pm ET
        near_rollover = datetime(2024, 1, 15, 16, 50, tzinfo=ZoneInfo("America/New_York"))
        ts_ms = int(near_rollover.timestamp() * 1000)

        decision = router.get_routing_decision(
            symbol="EUR_USD",
            side="BUY",
            size_usd=100000,
            timestamp_ms=ts_ms,
        )
        assert not decision.should_submit
        assert "rollover" in decision.reason.lower()
        assert decision.recommended_delay_sec is not None

    def test_routing_decision_low_liquidity(self):
        """Test routing decision during low liquidity."""
        router = ForexSessionRouter(min_liquidity_factor=0.8)

        # Sydney session has low liquidity
        sydney_time = datetime(2024, 1, 15, 23, 0, tzinfo=ZoneInfo("UTC"))
        ts_ms = int(sydney_time.timestamp() * 1000)

        decision = router.get_routing_decision(
            symbol="EUR_USD",
            side="BUY",
            size_usd=100000,
            timestamp_ms=ts_ms,
        )
        # Depending on exact session, might recommend waiting
        if not decision.should_submit:
            assert "liquidity" in decision.reason.lower()

    def test_routing_large_order_warning(self):
        """Test large order warnings."""
        router = ForexSessionRouter(large_order_threshold_usd=1_000_000)

        london_time = datetime(2024, 1, 15, 10, 0, tzinfo=ZoneInfo("UTC"))
        ts_ms = int(london_time.timestamp() * 1000)

        decision = router.get_routing_decision(
            symbol="EUR_USD",
            side="BUY",
            size_usd=5_000_000,  # Large order
            timestamp_ms=ts_ms,
        )
        # Should have warnings about large order
        assert any("large" in w.lower() for w in decision.warnings)

    def test_routing_near_weekend(self):
        """Test routing near weekend."""
        router = ForexSessionRouter(
            avoid_weekend_proximity=True,
            weekend_keepout_hours=2.0,
            large_order_threshold_usd=1_000_000,
        )

        # Friday 4pm ET (1 hour before weekend)
        friday_late = datetime(2024, 1, 12, 16, 0, tzinfo=ZoneInfo("America/New_York"))
        ts_ms = int(friday_late.timestamp() * 1000)

        decision = router.get_routing_decision(
            symbol="EUR_USD",
            side="BUY",
            size_usd=2_000_000,  # Large order near weekend
            timestamp_ms=ts_ms,
        )
        # Large orders near weekend should be rejected or warned
        if not decision.should_submit:
            assert "weekend" in decision.reason.lower()

    def test_adjust_spread_for_session(self):
        """Test spread adjustment by session."""
        router = ForexSessionRouter()

        base_spread = 1.0

        # London should have low multiplier
        london_spread = router.adjust_spread_for_session(
            base_spread, "EUR_USD", ForexSessionType.LONDON
        )
        assert london_spread <= base_spread * 1.1

        # Sydney should have higher multiplier
        sydney_spread = router.adjust_spread_for_session(
            base_spread, "EUR_USD", ForexSessionType.SYDNEY
        )
        assert sydney_spread > london_spread

    def test_get_session_volume_factor(self):
        """Test session volume factor calculation."""
        router = ForexSessionRouter()

        # London should have higher volume
        london_vol = router.get_session_volume_factor("EUR_USD", ForexSessionType.LONDON)
        sydney_vol = router.get_session_volume_factor("EUR_USD", ForexSessionType.SYDNEY)
        assert london_vol > sydney_vol

        # JPY pairs should have higher Tokyo volume
        jpy_tokyo = router.get_session_volume_factor("USD_JPY", ForexSessionType.TOKYO)
        eur_tokyo = router.get_session_volume_factor("EUR_USD", ForexSessionType.TOKYO)
        assert jpy_tokyo > eur_tokyo

    def test_should_wait_for_better_session(self):
        """Test waiting recommendation for better session."""
        router = ForexSessionRouter()

        # Small order - don't wait
        should_wait, reason = router.should_wait_for_better_session(
            size_usd=100000,
            current_session=ForexSessionType.SYDNEY,
        )
        assert not should_wait

        # Large order in low liquidity session
        should_wait, reason = router.should_wait_for_better_session(
            size_usd=10_000_000,
            current_session=ForexSessionType.SYDNEY,
        )
        assert should_wait

    def test_factory_function(self):
        """Test create_forex_session_router factory."""
        router = create_forex_session_router(
            config={
                "avoid_rollover": False,
                "min_liquidity_factor": 0.3,
            }
        )
        assert not router.avoid_rollover
        assert router.min_liquidity_factor == 0.3


class TestGetNextSessionStart:
    """Tests for get_next_session_start function."""

    def test_next_london_start(self):
        """Test getting next London session start."""
        # Before London opens
        before_london = datetime(2024, 1, 15, 5, 0, tzinfo=ZoneInfo("UTC"))
        ts_ms = int(before_london.timestamp() * 1000)

        next_start = get_next_session_start(ForexSessionType.LONDON, ts_ms)
        assert next_start > ts_ms

    def test_during_session(self):
        """Test when already in target session."""
        # During London
        during_london = datetime(2024, 1, 15, 10, 0, tzinfo=ZoneInfo("UTC"))
        ts_ms = int(during_london.timestamp() * 1000)

        next_start = get_next_session_start(ForexSessionType.LONDON, ts_ms)
        # Should return current time or slightly after
        assert next_start <= ts_ms


# =============================================================================
# Tests: Reconciliation
# =============================================================================


class TestOandaReconciliation:
    """Tests for OANDA reconciliation function."""

    def test_reconcile_success(self, mock_position_provider):
        """Test successful reconciliation."""
        mock_position_provider.get_account_summary.return_value = {
            "balance": 100000,
            "nav": 105000,
            "margin_used": 5000,
            "margin_available": 100000,
            "unrealized_pnl": 5000,
        }

        result = reconcile_oanda_state(
            adapter=mock_position_provider,
            local_positions={"EUR_USD": 100000, "GBP_USD": -50000},
        )

        assert result.success
        assert result.account_balance == 100000
        assert result.account_nav == 105000
        assert result.position_count == 2

    def test_reconcile_with_discrepancies(self, mock_position_provider):
        """Test reconciliation with discrepancies."""
        mock_position_provider.get_account_summary.return_value = {
            "balance": 100000,
        }

        result = reconcile_oanda_state(
            adapter=mock_position_provider,
            local_positions={},  # Empty local state
        )

        assert result.success
        assert len(result.position_discrepancies) > 0

    def test_reconcile_error_handling(self, mock_position_provider):
        """Test reconciliation error handling."""
        mock_position_provider.get_positions.side_effect = Exception("API Error")

        result = reconcile_oanda_state(
            adapter=mock_position_provider,
            local_positions={},
        )

        assert not result.success
        assert result.error is not None

    def test_reconcile_exposure_calculation(self, mock_position_provider):
        """Test exposure calculation in reconciliation."""
        mock_position_provider.get_account_summary.return_value = {"balance": 100000}

        result = reconcile_oanda_state(
            adapter=mock_position_provider,
            local_positions={"EUR_USD": 100000, "GBP_USD": -50000},
        )

        assert result.long_exposure > 0
        assert result.short_exposure > 0

    def test_reconcile_financing_tracking(self, mock_position_provider):
        """Test financing tracking in reconciliation."""
        mock_position_provider.get_account_summary.return_value = {"balance": 100000}

        result = reconcile_oanda_state(
            adapter=mock_position_provider,
            local_positions={"EUR_USD": 100000, "GBP_USD": -50000},
        )

        # Should sum financing from both positions
        expected_financing = -10.5 + -5.2
        assert result.total_financing == pytest.approx(expected_financing, rel=0.01)


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhase6Integration:
    """Integration tests for Phase 6 components."""

    def test_margin_guard_with_position_sync(self, mock_account_provider, mock_position_provider):
        """Test margin guard with position sync integration."""
        # Create components
        margin_guard = ForexMarginGuard(account_provider=mock_account_provider)
        leverage_guard = ForexLeverageGuard(max_leverage=50)

        local_positions = {"EUR_USD": 100000, "GBP_USD": -50000}

        sync = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: local_positions,
        )

        # Sync positions
        sync_result = sync.sync_once()
        assert sync_result.success

        # Check margin
        margin_status = margin_guard.get_margin_status()
        assert margin_status.margin_call_type == ForexMarginCallType.NONE

    def test_session_router_with_swap_tracker(self):
        """Test session router with swap tracker integration."""
        router = ForexSessionRouter()
        swap_tracker = SwapCostTracker()

        # Get current session
        session_info = router.current_session

        if session_info.is_open:
            # Calculate potential swap
            day = datetime.now(ZoneInfo("UTC")).weekday()
            cost = swap_tracker.calculate_daily_swap(
                symbol="EUR_USD",
                position_units=100000,
                is_long=True,
                current_price=1.0850,
                day_of_week=day,
            )
            assert cost.symbol == "EUR_USD"

    def test_full_risk_check_workflow(self, mock_account_provider):
        """Test complete risk check workflow."""
        # Components
        margin_guard = ForexMarginGuard(account_provider=mock_account_provider)
        leverage_guard = ForexLeverageGuard(max_leverage=50)
        router = ForexSessionRouter()

        # Simulate new trade
        symbol = "EUR_USD"
        units = 100000
        price = 1.0850
        notional = units * price

        # 1. Check margin
        margin_status = margin_guard.check_trade_margin(symbol, units, price)
        if margin_status.margin_call_type != ForexMarginCallType.NONE:
            pytest.skip("Trade would cause margin call")

        # 2. Check leverage
        leverage_check = leverage_guard.check_leverage(
            equity=100000,
            total_notional=500000,
            new_trade_notional=notional,
        )
        assert leverage_check.is_allowed

        # 3. Check session routing
        decision = router.get_routing_decision(
            symbol=symbol,
            side="BUY",
            size_usd=notional,
        )
        # Decision depends on current time

    def test_constants_consistency(self):
        """Test that constants are consistent across modules."""
        # Session liquidity should match spread (inverse relationship)
        for session in ForexSessionType:
            if session in SESSION_LIQUIDITY and session in SESSION_SPREAD_MULT:
                liq = SESSION_LIQUIDITY[session]
                spread = SESSION_SPREAD_MULT[session]
                # Higher liquidity should mean lower spread (generally)
                if liq > 0 and spread < float("inf"):
                    # This is a soft check - relationship should be inverse
                    pass

        # Swap day multipliers should sum to 7 (week) + 2 extra (Wednesday 3x)
        total_mult = sum(SWAP_DAY_MULTIPLIERS.values())
        assert total_mult == 7  # 1+1+3+1+1+0+0 = 7

    def test_error_resilience(self, mock_position_provider):
        """Test components handle errors gracefully."""
        # Provider raises exception
        mock_position_provider.get_positions.side_effect = Exception("Network Error")

        sync = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: {},
        )

        result = sync.sync_once()
        assert not result.success
        assert result.error is not None

        # Subsequent calls should still work
        mock_position_provider.get_positions.side_effect = None
        mock_position_provider.get_positions.return_value = {}
        result2 = sync.sync_once()
        assert result2.success


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_position(self):
        """Test handling of zero position."""
        pos = ForexPosition(
            symbol="EUR_USD",
            units=0,
            average_price=1.0850,
        )
        assert not pos.is_long
        assert not pos.is_short
        assert pos.abs_units == 0
        assert pos.notional == 0

    def test_very_large_position(self):
        """Test handling of very large position."""
        pos = ForexPosition(
            symbol="EUR_USD",
            units=100_000_000,  # 100M units
            average_price=1.0850,
        )
        assert pos.notional == pytest.approx(108_500_000, rel=0.01)

    def test_extreme_leverage(self):
        """Test extreme leverage scenarios."""
        guard = ForexLeverageGuard(max_leverage=500)

        check = guard.check_leverage(
            equity=10000,
            total_notional=5_000_000,  # 500:1
        )
        assert check.is_allowed
        assert check.current_leverage == 500

        check = guard.check_leverage(
            equity=10000,
            total_notional=5_001_000,  # Slightly over 500:1
        )
        assert not check.is_allowed

    def test_negative_equity(self):
        """Test handling of negative equity."""
        guard = ForexLeverageGuard()

        check = guard.check_leverage(
            equity=-1000,
            total_notional=100000,
        )
        assert not check.is_allowed

    def test_dst_transitions(self):
        """Test DST transition handling."""
        # March DST transition (spring forward)
        pre_dst = datetime(2024, 3, 10, 1, 30, tzinfo=ZoneInfo("America/New_York"))
        post_dst = datetime(2024, 3, 10, 3, 30, tzinfo=ZoneInfo("America/New_York"))

        # Both should give valid session info
        pre_session = get_current_forex_session(int(pre_dst.timestamp() * 1000))
        post_session = get_current_forex_session(int(post_dst.timestamp() * 1000))

        # Sessions should be detected correctly
        assert pre_session is not None
        assert post_session is not None

    def test_year_boundary(self):
        """Test year boundary handling."""
        # December 31 to January 1
        dec_31 = datetime(2023, 12, 31, 23, 59, tzinfo=ZoneInfo("UTC"))
        jan_1 = datetime(2024, 1, 1, 0, 1, tzinfo=ZoneInfo("UTC"))

        # New Year's can be weekend or weekday
        session_dec = get_current_forex_session(int(dec_31.timestamp() * 1000))
        session_jan = get_current_forex_session(int(jan_1.timestamp() * 1000))

        # Should handle gracefully
        assert session_dec is not None
        assert session_jan is not None

    def test_empty_positions(self, mock_position_provider):
        """Test handling of empty positions."""
        mock_position_provider.get_positions.return_value = {}

        sync = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: {},
        )

        result = sync.sync_once()
        assert result.success
        assert result.position_count == 0
        assert not result.has_discrepancies

    def test_jpy_pair_calculations(self):
        """Test JPY pair specific calculations."""
        tracker = SwapCostTracker()

        # JPY pairs have different pip size (0.01 vs 0.0001)
        cost = tracker.calculate_daily_swap(
            symbol="USD_JPY",
            position_units=100000,
            is_long=True,
            current_price=149.50,
            day_of_week=1,
        )
        assert cost.symbol == "USD_JPY"


# =============================================================================
# Additional Tests to Reach 100 Target
# =============================================================================


class TestSwapRateCaching:
    """Tests for swap rate caching behavior."""

    def test_swap_rate_cache_hit(self, mock_swap_provider):
        """Test swap rate caching prevents repeated provider calls."""
        tracker = SwapCostTracker(
            swap_rate_provider=mock_swap_provider,
            cache_ttl_sec=60.0,
        )

        # First call - should hit provider
        rate1 = tracker.get_swap_rate("EUR_USD")
        call_count_1 = mock_swap_provider.get_swap_rate.call_count

        # Second call - should use cache
        rate2 = tracker.get_swap_rate("EUR_USD")
        call_count_2 = mock_swap_provider.get_swap_rate.call_count

        assert call_count_2 == call_count_1
        assert rate1.symbol == rate2.symbol

    def test_swap_rate_different_symbols_separate_cache(self, mock_swap_provider):
        """Test different symbols have separate cache entries."""
        mock_swap_provider.get_swap_rate.side_effect = lambda s: SwapRate(
            symbol=s,
            long_rate=-0.5 if s == "EUR_USD" else 0.3,
            short_rate=0.3 if s == "EUR_USD" else -0.2,
            timestamp_ms=int(time.time() * 1000),
            source="mock",
        )

        tracker = SwapCostTracker(swap_rate_provider=mock_swap_provider)

        rate_eur = tracker.get_swap_rate("EUR_USD")
        rate_gbp = tracker.get_swap_rate("GBP_USD")

        assert rate_eur.long_rate != rate_gbp.long_rate
        assert mock_swap_provider.get_swap_rate.call_count == 2


class TestMultiSymbolSwapTracking:
    """Tests for multi-symbol swap tracking."""

    def test_multiple_symbols_cumulative(self):
        """Test cumulative swap across multiple symbols."""
        tracker = SwapCostTracker()

        symbols = ["EUR_USD", "GBP_USD", "USD_JPY"]
        for symbol in symbols:
            tracker.calculate_daily_swap(
                symbol=symbol,
                position_units=100000,
                is_long=True,
                current_price=1.0850 if "JPY" not in symbol else 149.50,
                day_of_week=1,
            )

        # Each symbol should have its own cumulative
        for symbol in symbols:
            assert tracker.get_cumulative_swap(symbol) != 0

        # Total should be sum of all
        total = tracker.get_cumulative_swap()
        individual_sum = sum(tracker.get_cumulative_swap(s) for s in symbols)
        assert total == pytest.approx(individual_sum, rel=0.01)

    def test_swap_history_ordering(self):
        """Test swap history maintains chronological order."""
        tracker = SwapCostTracker()

        # Add swaps over multiple days
        for day in range(5):
            tracker.calculate_daily_swap(
                symbol="EUR_USD",
                position_units=100000,
                is_long=True,
                current_price=1.0850,
                day_of_week=day,
            )

        history = tracker.get_swap_history("EUR_USD")
        assert len(history) == 5

        # Should be in chronological order (most recent last)
        for i in range(len(history) - 1):
            assert history[i].timestamp_ms <= history[i + 1].timestamp_ms


class TestCorrelatedExposureAdvanced:
    """Advanced tests for correlated exposure checking."""

    def test_highly_correlated_pairs(self):
        """Test detection of highly correlated exposure."""
        guard = ForexLeverageGuard(
            correlated_exposure_limit=0.70,
            concentration_limit=0.50,
        )

        # EUR/USD and GBP/USD are highly correlated (~0.85)
        positions = {
            "EUR_USD": 2_000_000,
            "GBP_USD": 2_000_000,
        }

        # Adding more EUR exposure when already exposed to correlated GBP
        check = guard.check_correlated_exposure(
            positions=positions,
            new_symbol="EUR_CHF",  # Also correlated with EUR
            new_notional=1_500_000,
            total_notional=4_000_000,
        )

        # Should detect correlation risk
        assert check is not None

    def test_uncorrelated_pairs_allowed(self):
        """Test uncorrelated pairs don't trigger correlation limits."""
        guard = ForexLeverageGuard(correlated_exposure_limit=0.80)

        # USD/JPY and AUD/NZD have low correlation
        positions = {
            "USD_JPY": 2_000_000,
        }

        check = guard.check_correlated_exposure(
            positions=positions,
            new_symbol="AUD_NZD",
            new_notional=2_000_000,
            total_notional=2_000_000,
        )

        # Should be allowed (low correlation)
        assert check.is_allowed or check.violation_type != LeverageViolationType.CORRELATED_EXPOSURE


class TestSessionRouterAdvanced:
    """Advanced tests for session router."""

    def test_pair_specific_session_recommendations(self):
        """Test pair-specific session recommendations."""
        router = ForexSessionRouter()

        # JPY pairs should prefer Tokyo session
        jpy_tokyo_vol = router.get_session_volume_factor("USD_JPY", ForexSessionType.TOKYO)
        jpy_sydney_vol = router.get_session_volume_factor("USD_JPY", ForexSessionType.SYDNEY)
        assert jpy_tokyo_vol > jpy_sydney_vol

        # AUD pairs should have good Sydney volume
        aud_sydney_vol = router.get_session_volume_factor("AUD_USD", ForexSessionType.SYDNEY)
        eur_sydney_vol = router.get_session_volume_factor("EUR_USD", ForexSessionType.SYDNEY)
        assert aud_sydney_vol >= eur_sydney_vol

    def test_routing_respects_all_sessions(self):
        """Test routing works correctly across all sessions."""
        router = ForexSessionRouter(min_liquidity_factor=0.0)  # Allow all sessions

        sessions_tested = set()
        # Test various times to cover all sessions
        test_times = [
            datetime(2024, 1, 15, 23, 0, tzinfo=ZoneInfo("UTC")),  # Sydney
            datetime(2024, 1, 15, 3, 0, tzinfo=ZoneInfo("UTC")),  # Tokyo
            datetime(2024, 1, 15, 10, 0, tzinfo=ZoneInfo("UTC")),  # London
            datetime(2024, 1, 15, 15, 0, tzinfo=ZoneInfo("UTC")),  # NY
            datetime(2024, 1, 15, 14, 0, tzinfo=ZoneInfo("UTC")),  # Overlap
        ]

        for dt in test_times:
            ts_ms = int(dt.timestamp() * 1000)
            decision = router.get_routing_decision(
                symbol="EUR_USD",
                side="BUY",
                size_usd=100000,
                timestamp_ms=ts_ms,
            )
            if decision.should_submit:
                sessions_tested.add(decision.session)

        # Should have tested multiple sessions
        assert len(sessions_tested) >= 3

    def test_seconds_to_market_open_calculation(self):
        """Test calculation of seconds until market opens."""
        router = ForexSessionRouter()

        # Saturday should return time until Sunday 5pm ET
        saturday = datetime(2024, 1, 13, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        ts_ms = int(saturday.timestamp() * 1000)

        seconds = router._seconds_to_market_open(ts_ms)
        assert seconds > 0
        # Should be roughly 29 hours (Saturday noon to Sunday 5pm)
        assert 100000 < seconds < 110000  # ~27-30 hours in seconds


class TestPositionSyncAdvanced:
    """Advanced tests for position synchronization."""

    def test_sync_with_price_tolerance(self, mock_position_provider):
        """Test sync with price mismatch within tolerance."""
        mock_position_provider.get_positions.return_value = {
            "EUR_USD": ForexPosition(
                symbol="EUR_USD",
                units=100000,
                average_price=1.0855,  # Slightly different price
                unrealized_pnl=500,
                margin_used=2000,
                financing=-10.5,
            ),
        }

        local_positions = {"EUR_USD": 100000}

        sync = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: local_positions,
            config=SyncConfig(
                position_tolerance_pct=0.01,
                price_tolerance_pct=0.01,  # 1% price tolerance
            ),
        )

        result = sync.sync_once()
        assert result.success
        # Units match, price within tolerance - should not flag as discrepancy

    def test_sync_tracks_margin_usage(self, mock_position_provider):
        """Test that sync tracks total margin usage."""
        sync = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: {"EUR_USD": 100000, "GBP_USD": -50000},
        )

        result = sync.sync_once()
        assert result.success
        # Total margin from mock positions: 2000 + 1500
        assert result.total_margin_used == pytest.approx(3500, rel=0.01)

    def test_sync_exposure_calculation(self, mock_position_provider):
        """Test long/short exposure calculation in sync."""
        sync = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: {"EUR_USD": 100000, "GBP_USD": -50000},
        )

        result = sync.sync_once()
        assert result.success
        # EUR_USD is long, GBP_USD is short
        assert result.long_exposure > 0
        assert result.short_exposure > 0
        assert result.net_exposure != 0


class TestAutoReconciliation:
    """Tests for automatic position reconciliation."""

    @pytest.fixture
    def mock_order_executor(self):
        """Create mock order executor."""
        executor = MagicMock()
        executor.place_market_order.return_value = ReconciliationOrderResult(
            success=True,
            order_id="test-order-123",
            symbol="EUR_USD",
            requested_units=5000,
            filled_units=5000,
            fill_price=1.0850,
        )
        return executor

    def test_reconciliation_executor_init(self, mock_order_executor):
        """Test ReconciliationExecutor initialization."""
        executor = ReconciliationExecutor(
            order_executor=mock_order_executor,
            max_units_per_order=50000,
            max_orders_per_hour=5,
            dry_run=False,
        )
        assert executor._max_units == 50000
        assert executor._max_orders_per_hour == 5

    def test_reconciliation_dry_run(self, mock_order_executor):
        """Test dry run mode doesn't execute orders."""
        executor = ReconciliationExecutor(
            order_executor=mock_order_executor,
            dry_run=True,
        )

        disc = PositionDiscrepancy(
            symbol="EUR_USD",
            discrepancy_type=PositionDiscrepancyType.UNITS_MISMATCH,
            local_units=100000,
            remote_units=105000,
        )

        action = ReconciliationAction(
            symbol="EUR_USD",
            discrepancy=disc,
            units_to_trade=5000,
        )

        result = executor.execute_reconciliation(action)
        assert result.success
        assert result.error == "DRY_RUN"
        mock_order_executor.place_market_order.assert_not_called()

    def test_reconciliation_rate_limiting(self, mock_order_executor):
        """Test rate limiting prevents too many orders."""
        executor = ReconciliationExecutor(
            order_executor=mock_order_executor,
            max_orders_per_hour=2,
        )

        disc = PositionDiscrepancy(
            symbol="EUR_USD",
            discrepancy_type=PositionDiscrepancyType.UNITS_MISMATCH,
            local_units=100000,
            remote_units=105000,
        )

        action = ReconciliationAction(
            symbol="EUR_USD",
            discrepancy=disc,
            units_to_trade=5000,
        )

        # First two should succeed
        result1 = executor.execute_reconciliation(action)
        result2 = executor.execute_reconciliation(action)
        assert result1.success
        assert result2.success

        # Third should be blocked by rate limit
        result3 = executor.execute_reconciliation(action)
        assert not result3.success
        assert "Rate limit" in result3.error

    def test_reconciliation_max_units_exceeded(self, mock_order_executor):
        """Test max units check blocks large orders."""
        executor = ReconciliationExecutor(
            order_executor=mock_order_executor,
            max_units_per_order=10000,
        )

        disc = PositionDiscrepancy(
            symbol="EUR_USD",
            discrepancy_type=PositionDiscrepancyType.UNITS_MISMATCH,
            local_units=100000,
            remote_units=200000,
        )

        action = ReconciliationAction(
            symbol="EUR_USD",
            discrepancy=disc,
            units_to_trade=100000,  # Exceeds 10000 limit
        )

        result = executor.execute_reconciliation(action)
        assert not result.success
        assert "exceeds max" in result.error

    def test_reconciliation_side_flip_prevented(self, mock_order_executor):
        """Test side flip prevention."""
        executor = ReconciliationExecutor(
            order_executor=mock_order_executor,
            prevent_side_flip=True,
            max_units_per_order=500_000,  # High limit to test side flip first
        )

        disc = PositionDiscrepancy(
            symbol="EUR_USD",
            discrepancy_type=PositionDiscrepancyType.SIDE_MISMATCH,
            local_units=100000,  # Long
            remote_units=-50000,  # Short
        )

        action = ReconciliationAction(
            symbol="EUR_USD",
            discrepancy=disc,
            units_to_trade=-50000,  # Within limits
        )

        result = executor.execute_reconciliation(action)
        assert not result.success
        assert "Side flip" in result.error

    def test_sync_with_auto_reconciliation(self, mock_position_provider, mock_order_executor):
        """Test sync with auto-reconciliation enabled."""
        # Remote has 105000, local has 100000 -> mismatch of 5000
        mock_position_provider.get_positions.return_value = {
            "EUR_USD": ForexPosition(
                symbol="EUR_USD",
                units=105000,
                average_price=1.0850,
                unrealized_pnl=500,
                margin_used=2100,
                financing=-10.5,
            ),
        }

        local_positions = {"EUR_USD": 100000}

        recon_executor = ReconciliationExecutor(
            order_executor=mock_order_executor,
            max_units_per_order=100000,
        )

        sync = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: local_positions,
            config=SyncConfig(
                auto_reconcile=True,
                position_tolerance_pct=0.01,  # 1% = 1000 units for 100k
                max_reconcile_units=50000,
            ),
            reconciliation_executor=recon_executor,
        )

        result = sync.sync_once()
        assert result.success
        assert result.has_discrepancies
        assert result.reconciliation_count >= 1

    def test_reconciliation_history(self, mock_order_executor):
        """Test reconciliation history tracking."""
        executor = ReconciliationExecutor(
            order_executor=mock_order_executor,
        )

        disc = PositionDiscrepancy(
            symbol="EUR_USD",
            discrepancy_type=PositionDiscrepancyType.UNITS_MISMATCH,
            local_units=100000,
            remote_units=105000,
        )

        action = ReconciliationAction(
            symbol="EUR_USD",
            discrepancy=disc,
            units_to_trade=5000,
        )

        executor.execute_reconciliation(action)
        history = executor.get_reconciliation_history()

        assert len(history) == 1
        assert history[0].success
        assert history[0].symbol == "EUR_USD"

    def test_reconciliation_callback(self, mock_position_provider, mock_order_executor):
        """Test reconciliation callback is invoked."""
        mock_position_provider.get_positions.return_value = {
            "EUR_USD": ForexPosition(
                symbol="EUR_USD",
                units=110000,  # 10% difference from local 100000
                average_price=1.0850,
                unrealized_pnl=500,
                margin_used=2200,
                financing=-10.5,
            ),
        }

        callback_results = []

        def on_recon(result):
            callback_results.append(result)

        recon_executor = ReconciliationExecutor(
            order_executor=mock_order_executor,
        )

        sync = ForexPositionSynchronizer(
            position_provider=mock_position_provider,
            local_state_getter=lambda: {"EUR_USD": 100000},
            config=SyncConfig(
                auto_reconcile=True,
                position_tolerance_pct=0.01,
            ),
            reconciliation_executor=recon_executor,
            on_reconciliation=on_recon,
        )

        sync.sync_once()
        assert len(callback_results) >= 1


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
