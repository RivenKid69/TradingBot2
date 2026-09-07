# -*- coding: utf-8 -*-
"""
impl_circuit_breaker.py
CME circuit breaker and price limit simulation.

This module implements multiple CME price protection mechanisms used in
production trading. These mechanisms are critical for realistic backtesting
and risk management.

Price Protection Mechanisms:
===========================

1. EQUITY INDEX CIRCUIT BREAKERS (Rule 80B)
   Triggered by S&P 500 decline from previous day's closing value.
   - Level 1: -7% → 15 min halt (RTH only, 9:30-15:25 ET, once per day)
   - Level 2: -13% → 15 min halt (RTH only, once per day)
   - Level 3: -20% → trading halted for remainder of day

   Notes:
   - Reference price is previous NYSE closing auction
   - Circuit breakers DO NOT apply after 3:25 PM ET
   - Only Level 3 triggers outside RTH

2. OVERNIGHT LIMIT UP/LIMIT DOWN (Equity Index)
   During Extended Trading Hours (ETH):
   - Limit: ±5% from reference price (previous settlement)
   - Orders beyond limit are rejected, no trading halt
   - Reference resets at 6:00 PM ET each day

3. COMMODITY DAILY PRICE LIMITS
   Product-specific price limits with expansion mechanism:
   - Initial limits: ±5-10% depending on product
   - After initial limit hit, limits can expand (2x, then 3x)
   - No trading halt, but price cannot trade beyond limit
   - Limits reset at start of next trading day

4. VELOCITY LOGIC
   Detects rapid price movements that may indicate:
   - Fat finger errors
   - Stop-loss cascades
   - Flash crashes
   Triggers brief protective pause (1-2 seconds)

5. STOP SPIKE LOGIC
   Prevents stop-loss cascade by detecting:
   - Multiple stop orders triggering simultaneously
   - Price gaps that would trigger many stops
   Adds brief delay to stop executions

References:
==========
- CME Rule 80B: https://www.cmegroup.com/education/articles-and-reports/understanding-stock-index-futures-circuit-breakers.html
- NYSE Circuit Breakers: https://www.nyse.com/markets/circuit-breakers
- CME Limit Up/Limit Down: https://www.cmegroup.com/trading/equity-index/
- CME Velocity Logic: https://www.cmegroup.com/market-data/files/CME_Globex_Velocity_Logic.pdf
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

from impl_span_margin import ProductGroup, PRODUCT_GROUPS

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class CircuitBreakerLevel(int, Enum):
    """Circuit breaker levels for equity index."""

    NONE = 0
    LEVEL_1 = 1  # -7%
    LEVEL_2 = 2  # -13%
    LEVEL_3 = 3  # -20%


class TradingState(str, Enum):
    """Trading state for circuit breakers."""

    NORMAL = "NORMAL"
    HALTED = "HALTED"
    LIMIT_UP = "LIMIT_UP"
    LIMIT_DOWN = "LIMIT_DOWN"
    VELOCITY_PAUSE = "VELOCITY_PAUSE"
    RESTRICTED = "RESTRICTED"  # Can trade but with restrictions


class PriceLimitStatus(str, Enum):
    """Status of commodity price limit."""

    INITIAL = "INITIAL"
    EXPANDED_1 = "EXPANDED_1"
    EXPANDED_2 = "EXPANDED_2"
    MAX_EXPANDED = "MAX_EXPANDED"


class LimitViolationType(str, Enum):
    """Type of price limit violation."""

    NONE = "NONE"
    LIMIT_UP = "LIMIT_UP"
    LIMIT_DOWN = "LIMIT_DOWN"


# =============================================================================
# Circuit Breaker Configuration
# =============================================================================

# Equity index circuit breaker thresholds (percentage decline from reference)
EQUITY_CB_THRESHOLDS: Dict[CircuitBreakerLevel, Decimal] = {
    CircuitBreakerLevel.LEVEL_1: Decimal("-0.07"),  # -7%
    CircuitBreakerLevel.LEVEL_2: Decimal("-0.13"),  # -13%
    CircuitBreakerLevel.LEVEL_3: Decimal("-0.20"),  # -20%
}

# Halt durations in seconds
EQUITY_CB_HALT_DURATIONS: Dict[CircuitBreakerLevel, Optional[int]] = {
    CircuitBreakerLevel.LEVEL_1: 15 * 60,  # 15 minutes
    CircuitBreakerLevel.LEVEL_2: 15 * 60,  # 15 minutes
    CircuitBreakerLevel.LEVEL_3: None,  # Remainder of day
}

# Products that use equity circuit breakers
EQUITY_CB_PRODUCTS: Set[str] = {"ES", "NQ", "YM", "RTY", "MES", "MNQ", "MYM", "M2K"}

# Overnight price limits (% from reference)
OVERNIGHT_LIMITS: Dict[str, Decimal] = {
    "ES": Decimal("0.05"),  # ±5%
    "NQ": Decimal("0.05"),
    "YM": Decimal("0.05"),
    "RTY": Decimal("0.05"),
    "MES": Decimal("0.05"),
    "MNQ": Decimal("0.05"),
    "MYM": Decimal("0.05"),
    "M2K": Decimal("0.05"),
}


@dataclass(frozen=True)
class CommodityPriceLimits:
    """
    Commodity price limit configuration.

    Attributes:
        initial: Initial daily limit (±%)
        expanded_1: First expansion (±%)
        expanded_2: Second expansion (±%)
        max_expansion: Maximum expansion (±%)
        expansion_trigger_minutes: Minutes at limit before expansion
    """

    initial: Decimal
    expanded_1: Decimal
    expanded_2: Decimal
    max_expansion: Decimal = Decimal("0.20")  # Default max 20%
    expansion_trigger_minutes: int = 2


# Commodity daily price limits
COMMODITY_LIMITS: Dict[str, CommodityPriceLimits] = {
    # Metals (COMEX)
    "GC": CommodityPriceLimits(
        initial=Decimal("0.05"),
        expanded_1=Decimal("0.075"),
        expanded_2=Decimal("0.10"),
    ),
    "SI": CommodityPriceLimits(
        initial=Decimal("0.07"),
        expanded_1=Decimal("0.105"),
        expanded_2=Decimal("0.14"),
    ),
    "HG": CommodityPriceLimits(
        initial=Decimal("0.07"),
        expanded_1=Decimal("0.105"),
        expanded_2=Decimal("0.14"),
    ),
    "MGC": CommodityPriceLimits(
        initial=Decimal("0.05"),
        expanded_1=Decimal("0.075"),
        expanded_2=Decimal("0.10"),
    ),
    # Energy (NYMEX)
    "CL": CommodityPriceLimits(
        initial=Decimal("0.07"),
        expanded_1=Decimal("0.105"),
        expanded_2=Decimal("0.14"),
    ),
    "NG": CommodityPriceLimits(
        initial=Decimal("0.10"),
        expanded_1=Decimal("0.15"),
        expanded_2=Decimal("0.20"),
    ),
    "MCL": CommodityPriceLimits(
        initial=Decimal("0.07"),
        expanded_1=Decimal("0.105"),
        expanded_2=Decimal("0.14"),
    ),
    "RB": CommodityPriceLimits(
        initial=Decimal("0.07"),
        expanded_1=Decimal("0.105"),
        expanded_2=Decimal("0.14"),
    ),
    "HO": CommodityPriceLimits(
        initial=Decimal("0.07"),
        expanded_1=Decimal("0.105"),
        expanded_2=Decimal("0.14"),
    ),
}

# Velocity logic thresholds (ticks per second)
VELOCITY_THRESHOLDS: Dict[str, int] = {
    "ES": 12,  # 12 ticks/sec = 3 points/sec
    "NQ": 20,  # 20 ticks/sec = 5 points/sec
    "YM": 20,  # 20 ticks/sec = 20 points/sec
    "RTY": 20,  # 20 ticks/sec = 2 points/sec
    "GC": 30,  # 30 ticks/sec = 3 points/sec
    "CL": 50,  # 50 ticks/sec = 0.50/sec
    "NG": 100,  # 100 ticks/sec
    "6E": 50,  # 50 ticks/sec
    "ZN": 30,  # 30 ticks/sec
}

# Velocity pause duration
VELOCITY_PAUSE_MS = 2000  # 2 seconds


# =============================================================================
# Data Classes
# =============================================================================


class CircuitBreakerEvent(NamedTuple):
    """Record of a circuit breaker event."""

    timestamp_ms: int
    level: CircuitBreakerLevel
    trigger_price: Decimal
    reference_price: Decimal
    change_pct: Decimal
    halt_duration_sec: Optional[int]
    resume_time_ms: Optional[int]


class PriceLimitEvent(NamedTuple):
    """Record of a price limit event."""

    timestamp_ms: int
    symbol: str
    limit_type: LimitViolationType
    limit_price: Decimal
    attempted_price: Decimal
    status: PriceLimitStatus


class VelocityEvent(NamedTuple):
    """Record of a velocity logic event."""

    timestamp_ms: int
    symbol: str
    velocity_ticks_per_sec: float
    threshold: int
    pause_until_ms: int


@dataclass
class CircuitBreakerState:
    """
    Current state of circuit breaker system.

    Attributes:
        trading_state: Current trading state
        current_level: Current circuit breaker level
        triggered_levels: Levels already triggered today
        halt_end_time_ms: Timestamp when halt ends (None if day halt)
        reference_price: Reference price for calculations
        overnight_upper_limit: Upper overnight limit price
        overnight_lower_limit: Lower overnight limit price
        limit_status: Current commodity limit expansion status
        velocity_pause_until_ms: Velocity pause end time
        events: List of circuit breaker events
    """

    trading_state: TradingState = TradingState.NORMAL
    current_level: CircuitBreakerLevel = CircuitBreakerLevel.NONE
    triggered_levels: Set[CircuitBreakerLevel] = field(default_factory=set)
    halt_end_time_ms: Optional[int] = None
    reference_price: Optional[Decimal] = None
    overnight_upper_limit: Optional[Decimal] = None
    overnight_lower_limit: Optional[Decimal] = None
    limit_status: PriceLimitStatus = PriceLimitStatus.INITIAL
    velocity_pause_until_ms: int = 0
    events: List[CircuitBreakerEvent] = field(default_factory=list)
    price_limit_events: List[PriceLimitEvent] = field(default_factory=list)
    velocity_events: List[VelocityEvent] = field(default_factory=list)


# =============================================================================
# Circuit Breaker Engine
# =============================================================================


class CMECircuitBreaker:
    """
    CME circuit breaker and price limit simulation.

    This class implements multiple price protection mechanisms:
    1. Equity index circuit breakers (Rule 80B)
    2. Overnight limit up/down
    3. Commodity daily price limits with expansion
    4. Velocity logic for fat-finger protection

    Example Usage:
        >>> cb = CMECircuitBreaker("ES")
        >>> cb.set_reference_price(Decimal("4500"))

        # Check for circuit breaker
        >>> result = cb.check_circuit_breaker(
        ...     current_price=Decimal("4185"),  # -7%
        ...     timestamp_ms=1000000,
        ...     is_rth=True,
        ... )
        >>> print(result)  # CircuitBreakerLevel.LEVEL_1

        # Check velocity logic
        >>> triggered = cb.check_velocity_logic(
        ...     price=Decimal("4180"),
        ...     timestamp_ms=1000100,
        ... )

    RTH Hours (when Level 1/2 can trigger):
        - Start: 9:30 AM ET
        - End: 3:25 PM ET
        - Level 1/2 only trigger ONCE per day during RTH
        - Level 3 can trigger anytime, halts for day
    """

    def __init__(
        self,
        symbol: str,
        tick_size: Optional[Decimal] = None,
    ) -> None:
        """
        Initialize circuit breaker engine.

        Args:
            symbol: Product symbol (ES, NQ, GC, CL, etc.)
            tick_size: Tick size (auto-detected if not provided)
        """
        self._symbol = symbol.upper()
        self._state = CircuitBreakerState()

        # Detect tick size
        if tick_size is not None:
            self._tick_size = tick_size
        else:
            self._tick_size = self._get_default_tick_size()

        # Track last price for velocity logic
        self._last_price: Optional[Decimal] = None
        self._last_price_ts_ms: int = 0

        # Determine product type
        self._is_equity_index = self._symbol in EQUITY_CB_PRODUCTS
        self._commodity_limits = COMMODITY_LIMITS.get(self._symbol)

    def _get_default_tick_size(self) -> Decimal:
        """Get default tick size for the product."""
        tick_sizes = {
            "ES": Decimal("0.25"),
            "NQ": Decimal("0.25"),
            "YM": Decimal("1.0"),
            "RTY": Decimal("0.10"),
            "MES": Decimal("0.25"),
            "MNQ": Decimal("0.25"),
            "GC": Decimal("0.10"),
            "SI": Decimal("0.005"),
            "HG": Decimal("0.0005"),
            "CL": Decimal("0.01"),
            "NG": Decimal("0.001"),
            "6E": Decimal("0.00005"),
            "ZN": Decimal("0.015625"),
        }
        return tick_sizes.get(self._symbol, Decimal("0.01"))

    def set_reference_price(self, price: Decimal) -> None:
        """
        Set reference price for circuit breaker calculations.

        For equity index: This is typically the previous NYSE closing price.
        For commodities: This is the previous day's settlement price.

        Args:
            price: Reference price
        """
        self._state.reference_price = price

        # Calculate overnight limits for equity index
        if self._is_equity_index:
            limit_pct = OVERNIGHT_LIMITS.get(self._symbol, Decimal("0.05"))
            self._state.overnight_upper_limit = price * (1 + limit_pct)
            self._state.overnight_lower_limit = price * (1 - limit_pct)

    def check_circuit_breaker(
        self,
        current_price: Decimal,
        timestamp_ms: int,
        is_rth: bool = True,
    ) -> CircuitBreakerLevel:
        """
        Check if equity index circuit breaker is triggered.

        Args:
            current_price: Current market price
            timestamp_ms: Current timestamp in milliseconds
            is_rth: True if during Regular Trading Hours (9:30-15:25 ET)

        Returns:
            Circuit breaker level triggered (NONE if no trigger)

        Notes:
            - Level 1 & 2 only trigger during RTH and only once per day
            - Level 3 can trigger anytime and halts trading for the day
            - Returns NONE if reference price not set
        """
        # Only applies to equity index
        if not self._is_equity_index:
            return CircuitBreakerLevel.NONE

        if self._state.reference_price is None:
            return CircuitBreakerLevel.NONE

        # Check if currently halted
        if self._state.trading_state == TradingState.HALTED:
            if (
                self._state.halt_end_time_ms is not None
                and timestamp_ms >= self._state.halt_end_time_ms
            ):
                # Halt period ended
                self._state.trading_state = TradingState.NORMAL
            else:
                return self._state.current_level

        # Calculate price change percentage
        change_pct = (current_price - self._state.reference_price) / self._state.reference_price

        # Check levels from highest to lowest
        for level in [
            CircuitBreakerLevel.LEVEL_3,
            CircuitBreakerLevel.LEVEL_2,
            CircuitBreakerLevel.LEVEL_1,
        ]:
            threshold = EQUITY_CB_THRESHOLDS[level]

            if change_pct <= threshold:
                # Level 1 & 2 only trigger during RTH
                if level in (CircuitBreakerLevel.LEVEL_1, CircuitBreakerLevel.LEVEL_2):
                    if not is_rth:
                        continue
                    # Only trigger once per day
                    if level in self._state.triggered_levels:
                        continue

                # Trigger circuit breaker
                self._state.triggered_levels.add(level)
                self._state.current_level = level
                self._state.trading_state = TradingState.HALTED

                # Calculate halt end time
                halt_duration = EQUITY_CB_HALT_DURATIONS.get(level)
                if halt_duration is not None:
                    self._state.halt_end_time_ms = timestamp_ms + (halt_duration * 1000)
                else:
                    self._state.halt_end_time_ms = None  # Day halt

                # Record event
                event = CircuitBreakerEvent(
                    timestamp_ms=timestamp_ms,
                    level=level,
                    trigger_price=current_price,
                    reference_price=self._state.reference_price,
                    change_pct=change_pct,
                    halt_duration_sec=halt_duration,
                    resume_time_ms=self._state.halt_end_time_ms,
                )
                self._state.events.append(event)

                logger.warning(
                    f"Circuit breaker triggered: {self._symbol} Level {level.value} "
                    f"at {change_pct:.2%} decline"
                )

                return level

        return CircuitBreakerLevel.NONE

    def check_overnight_limit(
        self,
        price: Decimal,
        timestamp_ms: int,
        is_overnight: bool = True,
    ) -> Tuple[bool, LimitViolationType]:
        """
        Check overnight price limit (limit up/down for equity index).

        Args:
            price: Proposed trade or order price
            timestamp_ms: Current timestamp
            is_overnight: True if during Extended Trading Hours

        Returns:
            (is_within_limit, violation_type)
            violation_type is LIMIT_UP, LIMIT_DOWN, or NONE
        """
        if not is_overnight or not self._is_equity_index:
            return (True, LimitViolationType.NONE)

        if self._state.overnight_upper_limit is None or self._state.overnight_lower_limit is None:
            return (True, LimitViolationType.NONE)

        if price > self._state.overnight_upper_limit:
            event = PriceLimitEvent(
                timestamp_ms=timestamp_ms,
                symbol=self._symbol,
                limit_type=LimitViolationType.LIMIT_UP,
                limit_price=self._state.overnight_upper_limit,
                attempted_price=price,
                status=self._state.limit_status,
            )
            self._state.price_limit_events.append(event)
            return (False, LimitViolationType.LIMIT_UP)

        if price < self._state.overnight_lower_limit:
            event = PriceLimitEvent(
                timestamp_ms=timestamp_ms,
                symbol=self._symbol,
                limit_type=LimitViolationType.LIMIT_DOWN,
                limit_price=self._state.overnight_lower_limit,
                attempted_price=price,
                status=self._state.limit_status,
            )
            self._state.price_limit_events.append(event)
            return (False, LimitViolationType.LIMIT_DOWN)

        return (True, LimitViolationType.NONE)

    def check_commodity_limit(
        self,
        price: Decimal,
        timestamp_ms: int,
    ) -> Tuple[bool, Decimal, Decimal, LimitViolationType]:
        """
        Check commodity daily price limit.

        Args:
            price: Current or proposed price
            timestamp_ms: Current timestamp

        Returns:
            (is_within_limit, lower_bound, upper_bound, violation_type)
        """
        if self._commodity_limits is None:
            # No limits for this product
            return (True, Decimal("0"), Decimal("inf"), LimitViolationType.NONE)

        if self._state.reference_price is None:
            return (True, Decimal("0"), Decimal("inf"), LimitViolationType.NONE)

        # Get current limit percentage based on expansion status
        limits = self._commodity_limits
        if self._state.limit_status == PriceLimitStatus.INITIAL:
            limit_pct = limits.initial
        elif self._state.limit_status == PriceLimitStatus.EXPANDED_1:
            limit_pct = limits.expanded_1
        elif self._state.limit_status == PriceLimitStatus.EXPANDED_2:
            limit_pct = limits.expanded_2
        else:
            limit_pct = limits.max_expansion

        upper_bound = self._state.reference_price * (1 + limit_pct)
        lower_bound = self._state.reference_price * (1 - limit_pct)

        violation = LimitViolationType.NONE
        is_within = True

        if price > upper_bound:
            is_within = False
            violation = LimitViolationType.LIMIT_UP
            self._state.trading_state = TradingState.LIMIT_UP
        elif price < lower_bound:
            is_within = False
            violation = LimitViolationType.LIMIT_DOWN
            self._state.trading_state = TradingState.LIMIT_DOWN
        else:
            if self._state.trading_state in (TradingState.LIMIT_UP, TradingState.LIMIT_DOWN):
                self._state.trading_state = TradingState.NORMAL

        if not is_within:
            event = PriceLimitEvent(
                timestamp_ms=timestamp_ms,
                symbol=self._symbol,
                limit_type=violation,
                limit_price=(
                    upper_bound if violation == LimitViolationType.LIMIT_UP else lower_bound
                ),
                attempted_price=price,
                status=self._state.limit_status,
            )
            self._state.price_limit_events.append(event)

        return (is_within, lower_bound, upper_bound, violation)

    def expand_commodity_limit(self) -> bool:
        """
        Expand commodity price limit after hitting initial limit.

        Called when price has been at limit for expansion_trigger_minutes.

        Returns:
            True if successfully expanded, False if already at max
        """
        if self._commodity_limits is None:
            return False

        if self._state.limit_status == PriceLimitStatus.INITIAL:
            self._state.limit_status = PriceLimitStatus.EXPANDED_1
            logger.info(f"{self._symbol}: Limit expanded to level 1")
            return True
        elif self._state.limit_status == PriceLimitStatus.EXPANDED_1:
            self._state.limit_status = PriceLimitStatus.EXPANDED_2
            logger.info(f"{self._symbol}: Limit expanded to level 2")
            return True
        elif self._state.limit_status == PriceLimitStatus.EXPANDED_2:
            self._state.limit_status = PriceLimitStatus.MAX_EXPANDED
            logger.info(f"{self._symbol}: Limit expanded to maximum")
            return True
        else:
            logger.warning(f"{self._symbol}: Already at maximum limit expansion")
            return False

    def check_velocity_logic(
        self,
        price: Decimal,
        timestamp_ms: int,
    ) -> bool:
        """
        Check if velocity logic should trigger protective pause.

        Velocity logic detects rapid price movements that may indicate
        fat-finger errors or cascade effects.

        Args:
            price: Current price
            timestamp_ms: Current timestamp in milliseconds

        Returns:
            True if velocity pause triggered (should pause trading)
        """
        # Check if already in pause
        if timestamp_ms < self._state.velocity_pause_until_ms:
            return True

        # Need previous price to calculate velocity
        if self._last_price is None:
            self._last_price = price
            self._last_price_ts_ms = timestamp_ms
            return False

        # Calculate velocity in ticks per second
        price_move = abs(price - self._last_price)
        ticks_moved = price_move / self._tick_size
        time_delta_sec = max(0.001, (timestamp_ms - self._last_price_ts_ms) / 1000)
        velocity = float(ticks_moved / Decimal(str(time_delta_sec)))

        # Update last price
        self._last_price = price
        self._last_price_ts_ms = timestamp_ms

        # Check against threshold
        threshold = VELOCITY_THRESHOLDS.get(self._symbol, 30)

        if velocity > threshold:
            # Trigger velocity pause
            self._state.velocity_pause_until_ms = timestamp_ms + VELOCITY_PAUSE_MS
            self._state.trading_state = TradingState.VELOCITY_PAUSE

            event = VelocityEvent(
                timestamp_ms=timestamp_ms,
                symbol=self._symbol,
                velocity_ticks_per_sec=velocity,
                threshold=threshold,
                pause_until_ms=self._state.velocity_pause_until_ms,
            )
            self._state.velocity_events.append(event)

            logger.warning(
                f"Velocity logic triggered: {self._symbol} "
                f"{velocity:.1f} ticks/sec > {threshold} threshold"
            )
            return True

        # Check if pause period ended
        if self._state.trading_state == TradingState.VELOCITY_PAUSE:
            if timestamp_ms >= self._state.velocity_pause_until_ms:
                self._state.trading_state = TradingState.NORMAL

        return False

    def can_trade(
        self,
        timestamp_ms: int,
        price: Optional[Decimal] = None,
        is_rth: bool = True,
        is_overnight: bool = False,
    ) -> Tuple[bool, str]:
        """
        Check if trading is allowed at current moment.

        Args:
            timestamp_ms: Current timestamp
            price: Price to check (for limit checks)
            is_rth: True if Regular Trading Hours
            is_overnight: True if Extended Trading Hours

        Returns:
            (can_trade, reason)
        """
        # Check halt state
        if self._state.trading_state == TradingState.HALTED:
            if self._state.halt_end_time_ms is None:
                return (False, "Trading halted for remainder of day (Level 3)")
            elif timestamp_ms < self._state.halt_end_time_ms:
                remaining = (self._state.halt_end_time_ms - timestamp_ms) / 1000
                return (False, f"Trading halted, {remaining:.0f}s remaining")
            else:
                # Halt ended
                self._state.trading_state = TradingState.NORMAL

        # Check velocity pause
        if timestamp_ms < self._state.velocity_pause_until_ms:
            remaining = self._state.velocity_pause_until_ms - timestamp_ms
            return (False, f"Velocity pause, {remaining}ms remaining")

        # Check price limits if price provided
        if price is not None:
            # Overnight limit check
            if is_overnight and self._is_equity_index:
                is_within, violation = self.check_overnight_limit(price, timestamp_ms, True)
                if not is_within:
                    return (False, f"Price violates overnight {violation.value} limit")

            # Commodity limit check
            if self._commodity_limits is not None:
                is_within, lower, upper, violation = self.check_commodity_limit(price, timestamp_ms)
                if not is_within:
                    return (
                        False,
                        f"Price violates {violation.value} limit ({lower:.2f} - {upper:.2f})",
                    )

        return (True, "OK")

    def reset_daily(self) -> None:
        """
        Reset circuit breaker state for new trading day.

        Call this at start of each trading day to:
        - Clear triggered circuit breaker levels
        - Reset commodity price limit expansion
        - Clear velocity pause
        """
        self._state = CircuitBreakerState()
        self._last_price = None
        self._last_price_ts_ms = 0
        logger.info(f"{self._symbol}: Circuit breaker state reset for new day")

    def get_halt_duration(self, level: CircuitBreakerLevel) -> Optional[int]:
        """
        Get halt duration in seconds for a circuit breaker level.

        Args:
            level: Circuit breaker level

        Returns:
            Halt duration in seconds, or None for day halt
        """
        return EQUITY_CB_HALT_DURATIONS.get(level)

    def get_current_limits(self) -> Dict[str, Any]:
        """
        Get current price limits.

        Returns:
            Dictionary with limit information
        """
        result: Dict[str, Any] = {
            "symbol": self._symbol,
            "reference_price": (
                float(self._state.reference_price) if self._state.reference_price else None
            ),
            "trading_state": self._state.trading_state.value,
        }

        if self._is_equity_index:
            result["overnight_upper"] = (
                float(self._state.overnight_upper_limit)
                if self._state.overnight_upper_limit
                else None
            )
            result["overnight_lower"] = (
                float(self._state.overnight_lower_limit)
                if self._state.overnight_lower_limit
                else None
            )
            result["cb_levels_triggered"] = [l.value for l in self._state.triggered_levels]

        if self._commodity_limits is not None:
            limits = self._commodity_limits
            ref = self._state.reference_price or Decimal("0")

            if self._state.limit_status == PriceLimitStatus.INITIAL:
                pct = limits.initial
            elif self._state.limit_status == PriceLimitStatus.EXPANDED_1:
                pct = limits.expanded_1
            elif self._state.limit_status == PriceLimitStatus.EXPANDED_2:
                pct = limits.expanded_2
            else:
                pct = limits.max_expansion

            result["limit_status"] = self._state.limit_status.value
            result["limit_pct"] = float(pct)
            result["upper_limit"] = float(ref * (1 + pct)) if ref else None
            result["lower_limit"] = float(ref * (1 - pct)) if ref else None

        return result

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state

    @property
    def symbol(self) -> str:
        """Get product symbol."""
        return self._symbol

    @property
    def is_halted(self) -> bool:
        """Check if trading is currently halted."""
        return self._state.trading_state == TradingState.HALTED

    @property
    def events(self) -> List[CircuitBreakerEvent]:
        """Get list of circuit breaker events."""
        return self._state.events.copy()


# =============================================================================
# Factory Functions
# =============================================================================


def create_circuit_breaker(
    symbol: str,
    reference_price: Optional[Decimal] = None,
) -> CMECircuitBreaker:
    """
    Create circuit breaker engine for a symbol.

    Args:
        symbol: Product symbol
        reference_price: Initial reference price (optional)

    Returns:
        Configured CMECircuitBreaker
    """
    cb = CMECircuitBreaker(symbol)
    if reference_price is not None:
        cb.set_reference_price(reference_price)
    return cb


def get_circuit_breaker_level(
    change_pct: Decimal,
) -> CircuitBreakerLevel:
    """
    Get circuit breaker level for a given percentage change.

    Args:
        change_pct: Price change as decimal (e.g., -0.07 for -7%)

    Returns:
        Circuit breaker level
    """
    for level in [
        CircuitBreakerLevel.LEVEL_3,
        CircuitBreakerLevel.LEVEL_2,
        CircuitBreakerLevel.LEVEL_1,
    ]:
        if change_pct <= EQUITY_CB_THRESHOLDS[level]:
            return level
    return CircuitBreakerLevel.NONE


def is_equity_index_product(symbol: str) -> bool:
    """
    Check if a symbol is an equity index product.

    Args:
        symbol: Product symbol

    Returns:
        True if equity index (uses Rule 80B circuit breakers)
    """
    return symbol.upper() in EQUITY_CB_PRODUCTS


def is_commodity_with_limits(symbol: str) -> bool:
    """
    Check if a symbol has commodity price limits.

    Args:
        symbol: Product symbol

    Returns:
        True if commodity with daily price limits
    """
    return symbol.upper() in COMMODITY_LIMITS


def get_commodity_limits(symbol: str) -> Optional[CommodityPriceLimits]:
    """
    Get commodity price limit configuration.

    Args:
        symbol: Product symbol

    Returns:
        CommodityPriceLimits or None if not a commodity
    """
    return COMMODITY_LIMITS.get(symbol.upper())


def get_velocity_threshold(symbol: str) -> int:
    """
    Get velocity logic threshold for a symbol.

    Args:
        symbol: Product symbol

    Returns:
        Threshold in ticks per second
    """
    return VELOCITY_THRESHOLDS.get(symbol.upper(), 30)


# =============================================================================
# Multi-Product Circuit Breaker Manager
# =============================================================================


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers across products.

    Provides centralized control for portfolio-level circuit breaker
    monitoring and cross-product limit tracking.

    Example:
        >>> manager = CircuitBreakerManager()
        >>> manager.add_product("ES", reference_price=Decimal("4500"))
        >>> manager.add_product("NQ", reference_price=Decimal("15000"))
        >>>
        >>> status = manager.check_all(timestamp_ms=1000000, is_rth=True)
        >>> if not status["can_trade"]:
        ...     print(f"Trading restricted: {status['reason']}")
    """

    def __init__(self) -> None:
        """Initialize circuit breaker manager."""
        self._breakers: Dict[str, CMECircuitBreaker] = {}

    def add_product(
        self,
        symbol: str,
        reference_price: Optional[Decimal] = None,
    ) -> CMECircuitBreaker:
        """
        Add a product to manage.

        Args:
            symbol: Product symbol
            reference_price: Reference price for calculations

        Returns:
            The created circuit breaker
        """
        cb = create_circuit_breaker(symbol, reference_price)
        self._breakers[symbol.upper()] = cb
        return cb

    def get_breaker(self, symbol: str) -> Optional[CMECircuitBreaker]:
        """Get circuit breaker for a symbol."""
        return self._breakers.get(symbol.upper())

    def set_reference_prices(self, prices: Dict[str, Decimal]) -> None:
        """
        Set reference prices for multiple products.

        Args:
            prices: Map of symbol to reference price
        """
        for symbol, price in prices.items():
            cb = self._breakers.get(symbol.upper())
            if cb is not None:
                cb.set_reference_price(price)

    def check_all(
        self,
        timestamp_ms: int,
        prices: Optional[Dict[str, Decimal]] = None,
        is_rth: bool = True,
        is_overnight: bool = False,
    ) -> Dict[str, Any]:
        """
        Check all circuit breakers.

        Args:
            timestamp_ms: Current timestamp
            prices: Current prices by symbol
            is_rth: True if RTH
            is_overnight: True if ETH

        Returns:
            Dictionary with overall status and per-product details
        """
        prices = prices or {}
        can_trade = True
        reasons: List[str] = []
        product_status: Dict[str, Dict[str, Any]] = {}

        for symbol, cb in self._breakers.items():
            price = prices.get(symbol)

            # Check circuit breaker
            if cb._is_equity_index:
                if price is not None:
                    level = cb.check_circuit_breaker(price, timestamp_ms, is_rth)
                    if level != CircuitBreakerLevel.NONE:
                        can_trade = False
                        reasons.append(f"{symbol}: CB Level {level.value}")

            # Check if can trade
            product_can_trade, reason = cb.can_trade(
                timestamp_ms=timestamp_ms,
                price=price,
                is_rth=is_rth,
                is_overnight=is_overnight,
            )

            product_status[symbol] = {
                "can_trade": product_can_trade,
                "reason": reason,
                "state": cb.state.trading_state.value,
                "limits": cb.get_current_limits(),
            }

            if not product_can_trade:
                can_trade = False
                reasons.append(f"{symbol}: {reason}")

        return {
            "can_trade": can_trade,
            "reason": "; ".join(reasons) if reasons else "OK",
            "products": product_status,
        }

    def reset_all_daily(self) -> None:
        """Reset all circuit breakers for new trading day."""
        for cb in self._breakers.values():
            cb.reset_daily()

    @property
    def products(self) -> List[str]:
        """Get list of managed products."""
        return list(self._breakers.keys())
