# -*- coding: utf-8 -*-
"""
test_circuit_breaker.py
Comprehensive tests for CME circuit breaker and price limit simulation.

Test Coverage:
- Equity index circuit breakers (Rule 80B)
- Overnight limit up/down
- Commodity daily price limits with expansion
- Velocity logic (fat-finger protection)
- Multi-product manager
- Factory functions

References:
- CME Rule 80B: https://www.cmegroup.com/education/articles-and-reports/understanding-stock-index-futures-circuit-breakers.html
- CME Velocity Logic: https://www.cmegroup.com/market-data/files/CME_Globex_Velocity_Logic.pdf
"""

import pytest
from decimal import Decimal
from typing import Dict, Any

from impl_circuit_breaker import (
    # Enums
    CircuitBreakerLevel,
    TradingState,
    PriceLimitStatus,
    LimitViolationType,
    # Data classes
    CommodityPriceLimits,
    CircuitBreakerEvent,
    PriceLimitEvent,
    VelocityEvent,
    CircuitBreakerState,
    # Main classes
    CMECircuitBreaker,
    CircuitBreakerManager,
    # Constants
    EQUITY_CB_THRESHOLDS,
    EQUITY_CB_HALT_DURATIONS,
    EQUITY_CB_PRODUCTS,
    OVERNIGHT_LIMITS,
    COMMODITY_LIMITS,
    VELOCITY_THRESHOLDS,
    VELOCITY_PAUSE_MS,
    # Factory functions
    create_circuit_breaker,
    get_circuit_breaker_level,
    is_equity_index_product,
    is_commodity_with_limits,
    get_commodity_limits,
    get_velocity_threshold,
)


# =============================================================================
# Test Constants and Configuration
# =============================================================================


class TestCircuitBreakerConstants:
    """Test circuit breaker constants and thresholds."""

    def test_equity_cb_thresholds(self):
        """Test equity circuit breaker thresholds are correct."""
        assert EQUITY_CB_THRESHOLDS[CircuitBreakerLevel.LEVEL_1] == Decimal("-0.07")
        assert EQUITY_CB_THRESHOLDS[CircuitBreakerLevel.LEVEL_2] == Decimal("-0.13")
        assert EQUITY_CB_THRESHOLDS[CircuitBreakerLevel.LEVEL_3] == Decimal("-0.20")

    def test_equity_cb_halt_durations(self):
        """Test halt durations for each level."""
        # Level 1 & 2: 15 minutes
        assert EQUITY_CB_HALT_DURATIONS[CircuitBreakerLevel.LEVEL_1] == 15 * 60
        assert EQUITY_CB_HALT_DURATIONS[CircuitBreakerLevel.LEVEL_2] == 15 * 60
        # Level 3: Day halt
        assert EQUITY_CB_HALT_DURATIONS[CircuitBreakerLevel.LEVEL_3] is None

    def test_equity_products(self):
        """Test equity index product set."""
        assert "ES" in EQUITY_CB_PRODUCTS
        assert "NQ" in EQUITY_CB_PRODUCTS
        assert "YM" in EQUITY_CB_PRODUCTS
        assert "RTY" in EQUITY_CB_PRODUCTS
        assert "MES" in EQUITY_CB_PRODUCTS  # Micro
        assert "GC" not in EQUITY_CB_PRODUCTS  # Gold is commodity

    def test_overnight_limits(self):
        """Test overnight price limits."""
        assert OVERNIGHT_LIMITS["ES"] == Decimal("0.05")  # ±5%
        assert OVERNIGHT_LIMITS["NQ"] == Decimal("0.05")

    def test_commodity_limits_structure(self):
        """Test commodity limits configuration."""
        gc_limits = COMMODITY_LIMITS["GC"]
        assert gc_limits.initial == Decimal("0.05")  # 5%
        assert gc_limits.expanded_1 == Decimal("0.075")  # 7.5%
        assert gc_limits.expanded_2 == Decimal("0.10")  # 10%

    def test_velocity_thresholds(self):
        """Test velocity logic thresholds."""
        assert VELOCITY_THRESHOLDS["ES"] == 12  # 12 ticks/sec
        assert VELOCITY_THRESHOLDS["CL"] == 50  # 50 ticks/sec

    def test_velocity_pause_duration(self):
        """Test velocity pause is 2 seconds."""
        assert VELOCITY_PAUSE_MS == 2000


# =============================================================================
# Test Circuit Breaker Level Detection
# =============================================================================


class TestCircuitBreakerLevelDetection:
    """Test circuit breaker level detection functions."""

    def test_level_none_for_positive_change(self):
        """Test no circuit breaker for positive price change."""
        level = get_circuit_breaker_level(Decimal("0.05"))
        assert level == CircuitBreakerLevel.NONE

    def test_level_none_for_small_decline(self):
        """Test no circuit breaker for small decline."""
        level = get_circuit_breaker_level(Decimal("-0.03"))
        assert level == CircuitBreakerLevel.NONE

    def test_level_1_at_exactly_7_percent(self):
        """Test Level 1 triggers at exactly -7%."""
        level = get_circuit_breaker_level(Decimal("-0.07"))
        assert level == CircuitBreakerLevel.LEVEL_1

    def test_level_1_at_8_percent(self):
        """Test Level 1 for -8% decline."""
        level = get_circuit_breaker_level(Decimal("-0.08"))
        assert level == CircuitBreakerLevel.LEVEL_1

    def test_level_2_at_13_percent(self):
        """Test Level 2 triggers at -13%."""
        level = get_circuit_breaker_level(Decimal("-0.13"))
        assert level == CircuitBreakerLevel.LEVEL_2

    def test_level_2_at_15_percent(self):
        """Test Level 2 for -15% decline."""
        level = get_circuit_breaker_level(Decimal("-0.15"))
        assert level == CircuitBreakerLevel.LEVEL_2

    def test_level_3_at_20_percent(self):
        """Test Level 3 triggers at -20%."""
        level = get_circuit_breaker_level(Decimal("-0.20"))
        assert level == CircuitBreakerLevel.LEVEL_3

    def test_level_3_at_25_percent(self):
        """Test Level 3 for severe decline."""
        level = get_circuit_breaker_level(Decimal("-0.25"))
        assert level == CircuitBreakerLevel.LEVEL_3


# =============================================================================
# Test CMECircuitBreaker Basic Operations
# =============================================================================


class TestCMECircuitBreakerBasic:
    """Test basic CMECircuitBreaker operations."""

    def test_init_equity_index(self):
        """Test initialization for equity index product."""
        cb = CMECircuitBreaker("ES")
        assert cb.symbol == "ES"
        assert cb._is_equity_index is True
        assert cb._tick_size == Decimal("0.25")

    def test_init_commodity(self):
        """Test initialization for commodity product."""
        cb = CMECircuitBreaker("GC")
        assert cb.symbol == "GC"
        assert cb._is_equity_index is False
        assert cb._commodity_limits is not None

    def test_init_with_custom_tick_size(self):
        """Test initialization with custom tick size."""
        cb = CMECircuitBreaker("ES", tick_size=Decimal("0.50"))
        assert cb._tick_size == Decimal("0.50")

    def test_set_reference_price(self):
        """Test setting reference price."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4500"))
        assert cb.state.reference_price == Decimal("4500")

    def test_overnight_limits_calculated(self):
        """Test overnight limits are calculated from reference."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4000"))
        # ±5%
        assert cb.state.overnight_upper_limit == Decimal("4200")  # 4000 * 1.05
        assert cb.state.overnight_lower_limit == Decimal("3800")  # 4000 * 0.95

    def test_initial_state_is_normal(self):
        """Test initial trading state is NORMAL."""
        cb = CMECircuitBreaker("ES")
        assert cb.state.trading_state == TradingState.NORMAL
        assert cb.is_halted is False


# =============================================================================
# Test Equity Circuit Breakers
# =============================================================================


class TestEquityCircuitBreakers:
    """Test equity index circuit breaker triggers."""

    def test_level_1_triggers_during_rth(self):
        """Test Level 1 triggers during RTH."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4500"))

        # -7% decline = 4185
        level = cb.check_circuit_breaker(
            current_price=Decimal("4185"),
            timestamp_ms=1000000,
            is_rth=True,
        )

        assert level == CircuitBreakerLevel.LEVEL_1
        assert cb.state.trading_state == TradingState.HALTED
        assert cb.is_halted is True

    def test_level_1_does_not_trigger_outside_rth(self):
        """Test Level 1 does NOT trigger outside RTH."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4500"))

        level = cb.check_circuit_breaker(
            current_price=Decimal("4185"),  # -7%
            timestamp_ms=1000000,
            is_rth=False,  # Outside RTH
        )

        assert level == CircuitBreakerLevel.NONE
        assert cb.state.trading_state == TradingState.NORMAL

    def test_level_1_triggers_only_once(self):
        """Test Level 1 triggers only once per day."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4500"))

        # First trigger
        level1 = cb.check_circuit_breaker(Decimal("4185"), 1000000, True)
        assert level1 == CircuitBreakerLevel.LEVEL_1

        # Simulate halt ending
        cb._state.trading_state = TradingState.NORMAL

        # Second attempt - should not re-trigger
        level2 = cb.check_circuit_breaker(Decimal("4185"), 2000000, True)
        assert level2 == CircuitBreakerLevel.NONE

    def test_level_2_triggers_after_level_1(self):
        """Test Level 2 can trigger after Level 1."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4500"))

        # Trigger Level 1
        cb.check_circuit_breaker(Decimal("4185"), 1000000, True)
        cb._state.trading_state = TradingState.NORMAL  # Simulate halt end

        # Trigger Level 2 (-13% = 3915)
        level = cb.check_circuit_breaker(Decimal("3915"), 2000000, True)
        assert level == CircuitBreakerLevel.LEVEL_2

    def test_level_3_triggers_anytime(self):
        """Test Level 3 can trigger even outside RTH."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4500"))

        # -20% = 3600, outside RTH
        level = cb.check_circuit_breaker(Decimal("3600"), 1000000, is_rth=False)
        assert level == CircuitBreakerLevel.LEVEL_3
        assert cb.is_halted is True

    def test_level_3_halt_has_no_end(self):
        """Test Level 3 halt has no end time (day halt)."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4500"))

        cb.check_circuit_breaker(Decimal("3600"), 1000000, True)
        assert cb.state.halt_end_time_ms is None

    def test_halt_end_time_correct(self):
        """Test halt end time is calculated correctly."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4500"))

        cb.check_circuit_breaker(Decimal("4185"), 1000000, True)
        # Level 1 = 15 minutes = 900 seconds = 900,000 ms
        expected_end = 1000000 + (15 * 60 * 1000)
        assert cb.state.halt_end_time_ms == expected_end

    def test_halt_ends_after_duration(self):
        """Test halt ends after specified duration."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4500"))

        cb.check_circuit_breaker(Decimal("4185"), 1000000, True)

        # Check still halted before end
        level = cb.check_circuit_breaker(Decimal("4200"), 1500000, True)
        assert cb.is_halted is True

        # Check after halt ends (15 min + buffer)
        level = cb.check_circuit_breaker(Decimal("4200"), 1000000 + 16 * 60 * 1000, True)
        assert cb.state.trading_state == TradingState.NORMAL

    def test_no_trigger_without_reference(self):
        """Test no trigger if reference price not set."""
        cb = CMECircuitBreaker("ES")
        # No reference price set

        level = cb.check_circuit_breaker(Decimal("100"), 1000000, True)
        assert level == CircuitBreakerLevel.NONE

    def test_circuit_breaker_event_recorded(self):
        """Test circuit breaker event is recorded."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4500"))

        cb.check_circuit_breaker(Decimal("4185"), 1000000, True)

        assert len(cb.events) == 1
        event = cb.events[0]
        assert event.level == CircuitBreakerLevel.LEVEL_1
        assert event.trigger_price == Decimal("4185")
        assert event.reference_price == Decimal("4500")


# =============================================================================
# Test Overnight Limits
# =============================================================================


class TestOvernightLimits:
    """Test overnight (ETH) price limits."""

    def test_within_overnight_limits(self):
        """Test price within overnight limits."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4000"))

        is_within, violation = cb.check_overnight_limit(
            price=Decimal("4100"),  # +2.5%
            timestamp_ms=1000000,
            is_overnight=True,
        )

        assert is_within is True
        assert violation == LimitViolationType.NONE

    def test_overnight_limit_up(self):
        """Test overnight limit up violation."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4000"))

        is_within, violation = cb.check_overnight_limit(
            price=Decimal("4300"),  # +7.5% > 5%
            timestamp_ms=1000000,
            is_overnight=True,
        )

        assert is_within is False
        assert violation == LimitViolationType.LIMIT_UP

    def test_overnight_limit_down(self):
        """Test overnight limit down violation."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4000"))

        is_within, violation = cb.check_overnight_limit(
            price=Decimal("3700"),  # -7.5% < -5%
            timestamp_ms=1000000,
            is_overnight=True,
        )

        assert is_within is False
        assert violation == LimitViolationType.LIMIT_DOWN

    def test_overnight_limits_not_applied_during_rth(self):
        """Test overnight limits not checked during RTH."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4000"))

        # Large move, but is_overnight=False
        is_within, violation = cb.check_overnight_limit(
            price=Decimal("5000"),
            timestamp_ms=1000000,
            is_overnight=False,
        )

        assert is_within is True

    def test_overnight_limits_only_for_equity(self):
        """Test overnight limits only apply to equity index."""
        cb = CMECircuitBreaker("GC")  # Gold - commodity
        cb.set_reference_price(Decimal("2000"))

        is_within, violation = cb.check_overnight_limit(
            price=Decimal("3000"),  # +50%
            timestamp_ms=1000000,
            is_overnight=True,
        )

        assert is_within is True  # No limits for commodities

    def test_price_limit_event_recorded(self):
        """Test price limit event is recorded."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4000"))

        cb.check_overnight_limit(Decimal("4300"), 1000000, True)

        assert len(cb.state.price_limit_events) == 1
        event = cb.state.price_limit_events[0]
        assert event.limit_type == LimitViolationType.LIMIT_UP


# =============================================================================
# Test Commodity Price Limits
# =============================================================================


class TestCommodityPriceLimits:
    """Test commodity daily price limits."""

    def test_within_initial_limits(self):
        """Test price within initial commodity limits."""
        cb = CMECircuitBreaker("GC")
        cb.set_reference_price(Decimal("2000"))

        # GC initial limit is 5%
        is_within, lower, upper, violation = cb.check_commodity_limit(
            price=Decimal("2050"),  # +2.5%
            timestamp_ms=1000000,
        )

        assert is_within is True
        assert lower == Decimal("1900")  # -5%
        assert upper == Decimal("2100")  # +5%
        assert violation == LimitViolationType.NONE

    def test_commodity_limit_up(self):
        """Test commodity limit up violation."""
        cb = CMECircuitBreaker("GC")
        cb.set_reference_price(Decimal("2000"))

        is_within, lower, upper, violation = cb.check_commodity_limit(
            price=Decimal("2200"),  # +10% > 5%
            timestamp_ms=1000000,
        )

        assert is_within is False
        assert violation == LimitViolationType.LIMIT_UP
        assert cb.state.trading_state == TradingState.LIMIT_UP

    def test_commodity_limit_down(self):
        """Test commodity limit down violation."""
        cb = CMECircuitBreaker("CL")
        cb.set_reference_price(Decimal("80"))

        # CL initial limit is 7%
        is_within, _, _, violation = cb.check_commodity_limit(
            price=Decimal("70"),  # -12.5% < -7%
            timestamp_ms=1000000,
        )

        assert is_within is False
        assert violation == LimitViolationType.LIMIT_DOWN

    def test_limit_expansion(self):
        """Test commodity limit expansion mechanism."""
        cb = CMECircuitBreaker("GC")
        cb.set_reference_price(Decimal("2000"))

        assert cb.state.limit_status == PriceLimitStatus.INITIAL

        # Expand once
        result = cb.expand_commodity_limit()
        assert result is True
        assert cb.state.limit_status == PriceLimitStatus.EXPANDED_1

        # Check new limits (7.5%)
        _, lower, upper, _ = cb.check_commodity_limit(Decimal("2000"), 1000000)
        assert lower == Decimal("1850")  # 2000 * 0.925
        assert upper == Decimal("2150")  # 2000 * 1.075

    def test_limit_expansion_sequence(self):
        """Test full limit expansion sequence."""
        cb = CMECircuitBreaker("GC")
        cb.set_reference_price(Decimal("2000"))

        assert cb.expand_commodity_limit() is True  # INITIAL -> EXPANDED_1
        assert cb.expand_commodity_limit() is True  # EXPANDED_1 -> EXPANDED_2
        assert cb.expand_commodity_limit() is True  # EXPANDED_2 -> MAX_EXPANDED
        assert cb.expand_commodity_limit() is False  # Already at max

    def test_no_limits_for_unknown_commodity(self):
        """Test no limits for products without configured limits."""
        cb = CMECircuitBreaker("ZN")  # Bond - not in COMMODITY_LIMITS
        cb.set_reference_price(Decimal("115"))

        is_within, lower, upper, violation = cb.check_commodity_limit(
            price=Decimal("200"),
            timestamp_ms=1000000,
        )

        assert is_within is True


# =============================================================================
# Test Velocity Logic
# =============================================================================


class TestVelocityLogic:
    """Test velocity logic (fat-finger protection)."""

    def test_no_trigger_on_first_price(self):
        """Test no trigger on first price update."""
        cb = CMECircuitBreaker("ES")

        triggered = cb.check_velocity_logic(Decimal("4500"), 1000000)
        assert triggered is False

    def test_no_trigger_for_slow_movement(self):
        """Test no trigger for slow price movement."""
        cb = CMECircuitBreaker("ES")

        # First price
        cb.check_velocity_logic(Decimal("4500"), 1000000)

        # Slow movement (1 tick over 1 second)
        triggered = cb.check_velocity_logic(Decimal("4500.25"), 2000000)
        assert triggered is False

    def test_trigger_for_rapid_movement(self):
        """Test trigger for rapid price movement."""
        cb = CMECircuitBreaker("ES")

        # First price
        cb.check_velocity_logic(Decimal("4500"), 1000000)

        # Rapid movement: 20 points = 80 ticks in 100ms = 800 ticks/sec
        # ES threshold is 12 ticks/sec
        triggered = cb.check_velocity_logic(Decimal("4520"), 1000100)
        assert triggered is True
        assert cb.state.trading_state == TradingState.VELOCITY_PAUSE

    def test_velocity_pause_duration(self):
        """Test velocity pause lasts correct duration."""
        cb = CMECircuitBreaker("ES")

        cb.check_velocity_logic(Decimal("4500"), 1000000)
        cb.check_velocity_logic(Decimal("4520"), 1000100)  # Triggers

        # Should still be paused
        assert cb.state.velocity_pause_until_ms == 1000100 + VELOCITY_PAUSE_MS

    def test_velocity_pause_ends(self):
        """Test velocity pause ends after duration."""
        cb = CMECircuitBreaker("ES")

        cb.check_velocity_logic(Decimal("4500"), 1000000)
        cb.check_velocity_logic(Decimal("4520"), 1000100)  # Triggers

        # Check after pause ends
        triggered = cb.check_velocity_logic(Decimal("4521"), 1000100 + VELOCITY_PAUSE_MS + 100)
        assert triggered is False
        assert cb.state.trading_state == TradingState.NORMAL

    def test_still_paused_if_within_duration(self):
        """Test returns True if still within pause period."""
        cb = CMECircuitBreaker("ES")

        cb.check_velocity_logic(Decimal("4500"), 1000000)
        cb.check_velocity_logic(Decimal("4520"), 1000100)  # Triggers

        # Check during pause
        triggered = cb.check_velocity_logic(Decimal("4521"), 1000500)
        assert triggered is True  # Still paused

    def test_velocity_event_recorded(self):
        """Test velocity event is recorded."""
        cb = CMECircuitBreaker("ES")

        cb.check_velocity_logic(Decimal("4500"), 1000000)
        cb.check_velocity_logic(Decimal("4520"), 1000100)

        assert len(cb.state.velocity_events) == 1
        event = cb.state.velocity_events[0]
        assert event.symbol == "ES"
        assert event.threshold == 12  # ES threshold


# =============================================================================
# Test can_trade Method
# =============================================================================


class TestCanTrade:
    """Test can_trade comprehensive check."""

    def test_can_trade_normal_state(self):
        """Test trading allowed in normal state."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4500"))

        can, reason = cb.can_trade(1000000)
        assert can is True
        assert reason == "OK"

    def test_cannot_trade_when_halted(self):
        """Test trading blocked when halted."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4500"))

        # Trigger Level 3
        cb.check_circuit_breaker(Decimal("3600"), 1000000, True)

        can, reason = cb.can_trade(1000000)
        assert can is False
        assert "Level 3" in reason

    def test_cannot_trade_during_velocity_pause(self):
        """Test trading blocked during velocity pause."""
        cb = CMECircuitBreaker("ES")

        cb.check_velocity_logic(Decimal("4500"), 1000000)
        cb.check_velocity_logic(Decimal("4520"), 1000100)  # Triggers

        can, reason = cb.can_trade(1000500)
        assert can is False
        assert "Velocity pause" in reason

    def test_cannot_trade_at_overnight_limit(self):
        """Test trading blocked at overnight limit."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4000"))

        can, reason = cb.can_trade(
            timestamp_ms=1000000,
            price=Decimal("4300"),  # Above 5% limit
            is_overnight=True,
        )
        assert can is False
        assert "LIMIT_UP" in reason

    def test_cannot_trade_at_commodity_limit(self):
        """Test trading blocked at commodity limit."""
        cb = CMECircuitBreaker("GC")
        cb.set_reference_price(Decimal("2000"))

        can, reason = cb.can_trade(
            timestamp_ms=1000000,
            price=Decimal("2200"),  # Above 5% limit
        )
        assert can is False
        assert "LIMIT_UP" in reason


# =============================================================================
# Test Daily Reset
# =============================================================================


class TestDailyReset:
    """Test daily state reset."""

    def test_reset_clears_triggered_levels(self):
        """Test reset clears triggered circuit breaker levels."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4500"))

        # Trigger Level 1
        cb.check_circuit_breaker(Decimal("4185"), 1000000, True)
        assert CircuitBreakerLevel.LEVEL_1 in cb.state.triggered_levels

        # Reset
        cb.reset_daily()
        assert len(cb.state.triggered_levels) == 0

    def test_reset_clears_trading_state(self):
        """Test reset returns to normal trading state."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4500"))

        cb.check_circuit_breaker(Decimal("3600"), 1000000, True)  # Level 3
        cb.reset_daily()

        assert cb.state.trading_state == TradingState.NORMAL

    def test_reset_clears_limit_expansion(self):
        """Test reset clears commodity limit expansion."""
        cb = CMECircuitBreaker("GC")
        cb.set_reference_price(Decimal("2000"))

        cb.expand_commodity_limit()
        cb.expand_commodity_limit()
        assert cb.state.limit_status == PriceLimitStatus.EXPANDED_2

        cb.reset_daily()
        assert cb.state.limit_status == PriceLimitStatus.INITIAL

    def test_reset_clears_events(self):
        """Test reset clears event history."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4500"))

        cb.check_circuit_breaker(Decimal("4185"), 1000000, True)
        assert len(cb.events) == 1

        cb.reset_daily()
        assert len(cb.events) == 0


# =============================================================================
# Test get_current_limits
# =============================================================================


class TestGetCurrentLimits:
    """Test get_current_limits method."""

    def test_equity_index_limits_info(self):
        """Test limit info for equity index."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4000"))

        limits = cb.get_current_limits()

        assert limits["symbol"] == "ES"
        assert limits["reference_price"] == 4000.0
        assert limits["overnight_upper"] == 4200.0  # +5%
        assert limits["overnight_lower"] == 3800.0  # -5%
        assert limits["cb_levels_triggered"] == []

    def test_commodity_limits_info(self):
        """Test limit info for commodity."""
        cb = CMECircuitBreaker("GC")
        cb.set_reference_price(Decimal("2000"))

        limits = cb.get_current_limits()

        assert limits["symbol"] == "GC"
        assert limits["limit_status"] == "INITIAL"
        assert limits["limit_pct"] == 0.05
        assert limits["upper_limit"] == 2100.0
        assert limits["lower_limit"] == 1900.0

    def test_limits_after_expansion(self):
        """Test limits info after expansion."""
        cb = CMECircuitBreaker("GC")
        cb.set_reference_price(Decimal("2000"))
        cb.expand_commodity_limit()

        limits = cb.get_current_limits()
        assert limits["limit_status"] == "EXPANDED_1"
        assert limits["limit_pct"] == 0.075  # 7.5%


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_circuit_breaker_simple(self):
        """Test basic factory function."""
        cb = create_circuit_breaker("ES")
        assert cb.symbol == "ES"
        assert cb._is_equity_index is True

    def test_create_circuit_breaker_with_reference(self):
        """Test factory with reference price."""
        cb = create_circuit_breaker("ES", reference_price=Decimal("4500"))
        assert cb.state.reference_price == Decimal("4500")

    def test_is_equity_index_product(self):
        """Test equity index product detection."""
        assert is_equity_index_product("ES") is True
        assert is_equity_index_product("NQ") is True
        assert is_equity_index_product("GC") is False
        assert is_equity_index_product("CL") is False

    def test_is_commodity_with_limits(self):
        """Test commodity with limits detection."""
        assert is_commodity_with_limits("GC") is True
        assert is_commodity_with_limits("CL") is True
        assert is_commodity_with_limits("ES") is False  # Equity
        assert is_commodity_with_limits("ZN") is False  # No limits

    def test_get_commodity_limits(self):
        """Test getting commodity limits."""
        limits = get_commodity_limits("GC")
        assert limits is not None
        assert limits.initial == Decimal("0.05")

        limits = get_commodity_limits("ES")
        assert limits is None

    def test_get_velocity_threshold(self):
        """Test getting velocity threshold."""
        assert get_velocity_threshold("ES") == 12
        assert get_velocity_threshold("CL") == 50
        assert get_velocity_threshold("UNKNOWN") == 30  # Default


# =============================================================================
# Test CircuitBreakerManager
# =============================================================================


class TestCircuitBreakerManager:
    """Test multi-product circuit breaker manager."""

    def test_add_product(self):
        """Test adding products to manager."""
        manager = CircuitBreakerManager()

        cb = manager.add_product("ES", Decimal("4500"))

        assert "ES" in manager.products
        assert cb.symbol == "ES"

    def test_get_breaker(self):
        """Test getting breaker by symbol."""
        manager = CircuitBreakerManager()
        manager.add_product("ES")
        manager.add_product("NQ")

        es_cb = manager.get_breaker("ES")
        assert es_cb is not None
        assert es_cb.symbol == "ES"

        unknown = manager.get_breaker("UNKNOWN")
        assert unknown is None

    def test_set_reference_prices(self):
        """Test setting multiple reference prices."""
        manager = CircuitBreakerManager()
        manager.add_product("ES")
        manager.add_product("NQ")

        manager.set_reference_prices(
            {
                "ES": Decimal("4500"),
                "NQ": Decimal("15000"),
            }
        )

        assert manager.get_breaker("ES").state.reference_price == Decimal("4500")
        assert manager.get_breaker("NQ").state.reference_price == Decimal("15000")

    def test_check_all_normal(self):
        """Test check_all when all normal."""
        manager = CircuitBreakerManager()
        manager.add_product("ES", Decimal("4500"))
        manager.add_product("GC", Decimal("2000"))

        status = manager.check_all(
            timestamp_ms=1000000,
            prices={"ES": Decimal("4400"), "GC": Decimal("2050")},
        )

        assert status["can_trade"] is True
        assert status["reason"] == "OK"

    def test_check_all_with_circuit_breaker(self):
        """Test check_all when circuit breaker triggered."""
        manager = CircuitBreakerManager()
        manager.add_product("ES", Decimal("4500"))
        manager.add_product("GC", Decimal("2000"))

        status = manager.check_all(
            timestamp_ms=1000000,
            prices={"ES": Decimal("3600"), "GC": Decimal("2050")},  # ES -20%
            is_rth=True,
        )

        assert status["can_trade"] is False
        assert "ES" in status["reason"]
        assert "Level 3" in status["reason"]

    def test_check_all_product_status(self):
        """Test check_all returns per-product status."""
        manager = CircuitBreakerManager()
        manager.add_product("ES", Decimal("4500"))

        status = manager.check_all(
            timestamp_ms=1000000,
            prices={"ES": Decimal("4400")},
        )

        assert "products" in status
        assert "ES" in status["products"]
        assert status["products"]["ES"]["can_trade"] is True

    def test_reset_all_daily(self):
        """Test resetting all circuit breakers."""
        manager = CircuitBreakerManager()
        manager.add_product("ES", Decimal("4500"))
        manager.add_product("NQ", Decimal("15000"))

        # Trigger circuit breakers
        es_cb = manager.get_breaker("ES")
        es_cb.check_circuit_breaker(Decimal("4185"), 1000000, True)

        manager.reset_all_daily()

        assert es_cb.state.trading_state == TradingState.NORMAL
        assert len(es_cb.state.triggered_levels) == 0


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_case_insensitive_symbol(self):
        """Test symbol is case insensitive."""
        cb = CMECircuitBreaker("es")
        assert cb.symbol == "ES"

    def test_none_reference_price_handling(self):
        """Test behavior with no reference price."""
        cb = CMECircuitBreaker("ES")

        # Circuit breaker check
        level = cb.check_circuit_breaker(Decimal("100"), 1000000, True)
        assert level == CircuitBreakerLevel.NONE

        # Overnight limit check
        is_within, _ = cb.check_overnight_limit(Decimal("100"), 1000000, True)
        assert is_within is True

    def test_zero_time_delta_velocity(self):
        """Test velocity calculation with zero time delta."""
        cb = CMECircuitBreaker("ES")

        cb.check_velocity_logic(Decimal("4500"), 1000000)
        # Same timestamp
        triggered = cb.check_velocity_logic(Decimal("4520"), 1000000)
        # Should handle gracefully (min time delta)
        assert triggered is True  # Huge velocity -> triggered

    def test_boundary_prices(self):
        """Test prices exactly at limits."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4000"))

        # Exactly at overnight upper limit
        is_within, _ = cb.check_overnight_limit(Decimal("4200"), 1000000, True)
        assert is_within is True  # At limit is OK

        # Just above
        is_within, violation = cb.check_overnight_limit(Decimal("4200.01"), 1000000, True)
        assert is_within is False
        assert violation == LimitViolationType.LIMIT_UP

    def test_default_tick_size_fallback(self):
        """Test default tick size for unknown product."""
        cb = CMECircuitBreaker("UNKNOWN")
        assert cb._tick_size == Decimal("0.01")  # Default

    def test_get_halt_duration(self):
        """Test get_halt_duration helper."""
        cb = CMECircuitBreaker("ES")

        assert cb.get_halt_duration(CircuitBreakerLevel.LEVEL_1) == 15 * 60
        assert cb.get_halt_duration(CircuitBreakerLevel.LEVEL_3) is None


# =============================================================================
# Test Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_flash_crash_scenario(self):
        """Test flash crash with multiple triggers."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4500"))

        # Market opens normally
        can_trade, _ = cb.can_trade(1000000)
        assert can_trade is True

        # Rapid decline triggers Level 1
        cb.check_circuit_breaker(Decimal("4185"), 1000000, True)
        assert cb.is_halted is True

        # Wait for halt to end (15 min)
        halt_end = cb.state.halt_end_time_ms + 1000
        cb.check_circuit_breaker(Decimal("4100"), halt_end, True)

        # Further decline triggers Level 2
        cb.check_circuit_breaker(Decimal("3915"), halt_end + 60000, True)
        assert cb.state.current_level == CircuitBreakerLevel.LEVEL_2

    def test_overnight_trading_scenario(self):
        """Test overnight trading with limits."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4000"))

        # Normal overnight trading
        can_trade, _ = cb.can_trade(
            timestamp_ms=1000000,
            price=Decimal("4100"),
            is_overnight=True,
        )
        assert can_trade is True

        # Price approaches limit
        can_trade, reason = cb.can_trade(
            timestamp_ms=1000000,
            price=Decimal("4250"),  # Above 5%
            is_overnight=True,
        )
        assert can_trade is False
        assert "LIMIT_UP" in reason

    def test_commodity_limit_expansion_scenario(self):
        """Test commodity limit expansion during volatile day."""
        cb = CMECircuitBreaker("GC")
        cb.set_reference_price(Decimal("2000"))

        # Initial limit hit
        is_within, _, upper, _ = cb.check_commodity_limit(Decimal("2100"), 1000000)
        assert is_within is True

        is_within, _, upper, _ = cb.check_commodity_limit(Decimal("2150"), 1000000)
        assert is_within is False

        # Expand limits
        cb.expand_commodity_limit()

        # Now 2150 is within limits
        is_within, _, upper, _ = cb.check_commodity_limit(Decimal("2140"), 2000000)
        assert is_within is True

    def test_velocity_during_halt(self):
        """Test velocity logic during circuit breaker halt."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4500"))

        # Trigger circuit breaker
        cb.check_circuit_breaker(Decimal("4185"), 1000000, True)
        assert cb.is_halted is True

        # Trading blocked regardless of velocity
        can_trade, _ = cb.can_trade(1001000)
        assert can_trade is False

    def test_manager_portfolio_status(self):
        """Test manager tracks multiple products."""
        manager = CircuitBreakerManager()
        manager.add_product("ES", Decimal("4500"))
        manager.add_product("NQ", Decimal("15000"))
        manager.add_product("GC", Decimal("2000"))

        # Check normal status
        status = manager.check_all(
            timestamp_ms=1000000,
            prices={
                "ES": Decimal("4400"),
                "NQ": Decimal("14000"),
                "GC": Decimal("2050"),
            },
            is_rth=True,
        )

        assert status["can_trade"] is True
        assert len(status["products"]) == 3


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestCoverageEdgeCases:
    """Tests for edge cases to achieve 100% coverage."""

    def test_non_equity_index_returns_none(self):
        """Test circuit breaker check on non-equity index product returns NONE."""
        # GC (gold) is not equity index
        cb = CMECircuitBreaker("GC")
        cb.set_reference_price(Decimal("2000"))

        # Should return NONE immediately since not equity index
        level = cb.check_circuit_breaker(Decimal("1800"), 1000000, True)
        assert level == CircuitBreakerLevel.NONE

    def test_commodity_limits_no_reference_price(self):
        """Test commodity limits with no reference price returns early."""
        cb = CMECircuitBreaker("CL")
        # Don't set reference price

        is_within, lower, upper, violation = cb.check_commodity_limit(Decimal("70"), 1000000)
        # Should return early with defaults
        assert is_within is True
        assert violation == LimitViolationType.NONE

    def test_commodity_limits_expanded_2(self):
        """Test commodity limits at EXPANDED_2 status."""
        cb = CMECircuitBreaker("CL")
        cb.set_reference_price(Decimal("80"))

        # Expand twice to reach EXPANDED_2
        cb.expand_commodity_limit()  # -> EXPANDED_1
        cb.expand_commodity_limit()  # -> EXPANDED_2

        # Check status reflects EXPANDED_2
        limits = cb.get_current_limits()
        assert limits["limit_status"] == "EXPANDED_2"

        # CL expanded_2 = 14%, so limits are [68.8, 91.2]
        # Price of 90 is within expanded limits
        is_within, lower, upper, violation = cb.check_commodity_limit(Decimal("90"), 1000000)
        assert is_within is True

    def test_commodity_limits_max_expansion(self):
        """Test commodity limits at MAX expansion status."""
        cb = CMECircuitBreaker("CL")
        cb.set_reference_price(Decimal("80"))

        # Expand three times to reach max
        cb.expand_commodity_limit()  # -> EXPANDED_1
        cb.expand_commodity_limit()  # -> EXPANDED_2
        success = cb.expand_commodity_limit()  # -> MAX_EXPANDED
        assert success is True

        # Fourth expansion should fail
        success = cb.expand_commodity_limit()
        assert success is False

        limits = cb.get_current_limits()
        assert limits["limit_status"] == "MAX_EXPANDED"

    def test_expand_commodity_limit_without_limits(self):
        """Test expand_commodity_limit on product without commodity limits."""
        # ES is equity index, has no commodity limits
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4500"))

        # Should return False since no commodity limits
        success = cb.expand_commodity_limit()
        assert success is False

    def test_halt_ended_transition(self):
        """Test trading state transition when halt ends."""
        cb = CMECircuitBreaker("ES")
        cb.set_reference_price(Decimal("4500"))

        # Trigger Level 1 circuit breaker
        cb.check_circuit_breaker(Decimal("4185"), 1000000, True)
        assert cb.is_halted is True

        # Wait for halt to end (15 minutes = 900000 ms)
        after_halt_time = 1000000 + 900001

        # Check can_trade after halt ends
        can_trade, reason = cb.can_trade(after_halt_time)
        # After halt ends, should be able to trade
        assert can_trade is True or "halted" not in reason.lower()

    def test_get_current_limits_expanded_states(self):
        """Test get_current_limits returns correct limit percentages for expanded states."""
        cb = CMECircuitBreaker("NG")
        cb.set_reference_price(Decimal("3.0"))

        # INITIAL state
        limits = cb.get_current_limits()
        assert "limit_status" in limits
        assert limits["limit_status"] == "INITIAL"

        # EXPANDED_1
        cb.expand_commodity_limit()
        limits = cb.get_current_limits()
        assert limits["limit_status"] == "EXPANDED_1"

        # EXPANDED_2
        cb.expand_commodity_limit()
        limits = cb.get_current_limits()
        assert limits["limit_status"] == "EXPANDED_2"
        assert "limit_pct" in limits
        assert limits["limit_pct"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
