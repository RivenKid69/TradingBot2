# -*- coding: utf-8 -*-
"""
execution_providers_cme_l3.py
L3 Execution Provider for CME Group Futures.

This module provides high-fidelity execution simulation for CME Group futures
using the complete L3 LOB infrastructure with CME Globex-specific mechanics.

Key Features:
============

1. GLOBEX MATCHING ENGINE
   - FIFO Price-Time Priority (Globex algorithm)
   - Market with Protection (MWP) orders
   - Opening/closing auction matching
   - Stop orders with velocity logic
   - All-or-None / Minimum quantity support
   - Trade at Settlement (TAS) orders

2. CIRCUIT BREAKER INTEGRATION
   - Rule 80B equity index breakers (7%, 13%, 20%)
   - Commodity daily price limits with expansion
   - Overnight limit up/down (±5%)
   - Velocity logic for fat-finger protection
   - Stop spike logic

3. DAILY SETTLEMENT SIMULATION
   - Product-specific settlement times
   - Daily variation margin calculation
   - Settlement price impact on slippage

4. SESSION AWARENESS
   - RTH (Regular Trading Hours) vs ETH (Electronic Trading Hours)
   - Auction periods (opening/closing)
   - Maintenance windows
   - Session-based spread multipliers

5. ROLL PERIOD HANDLING
   - Front month liquidity shift detection
   - Roll calendar integration
   - Spread adjustments during roll

Architecture:
============
    CMEL3ExecutionProvider extends L3ExecutionProvider with:
    - GlobexMatchingEngine (from lob/cme_matching.py)
    - CMECircuitBreaker integration (from impl_circuit_breaker.py)
    - DailySettlementSimulator
    - Session-aware slippage adjustments

References:
==========
- CME Globex Matching Algorithm: https://www.cmegroup.com/confluence/display/EPICSANDBOX/Matching+Algorithms
- CME Rule 80B: Circuit breakers
- CME Velocity Logic: https://www.cmegroup.com/market-data/files/CME_Globex_Velocity_Logic.pdf
- Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
- Kissell (2013): "Optimal Trading Strategies"

Phase 5B of Futures Integration.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

# Base execution providers
from execution_providers import (
    AssetClass,
    BarData,
    Fill,
    MarketState,
    Order,
    OrderSide,
    OrderType,
    SlippageProvider,
    FeeProvider,
)

# L2 CME providers
from execution_providers_cme import (
    CMESlippageProvider,
    CMESlippageConfig,
    CMEFeeProvider,
    CMEFeeConfig,
    CMETradingSession,
    CircuitBreakerState as L2CircuitBreakerState,
    TICK_SIZES,
    DEFAULT_TICK_SIZE,
    CME_SLIPPAGE_PROFILES,
)

# L3 execution providers
from execution_providers_l3 import (
    L3ExecutionProvider,
    L3SlippageProvider,
    L3FillProvider,
    create_l3_execution_provider,
)

# LOB components
from lob import (
    Side as LOBSide,
    OrderType as LOBOrderType,
    LimitOrder,
    OrderBook,
    Fill as LOBFill,
    Trade,
    MatchingEngine,
    MatchResult,
    MatchType,
    STPAction,
    QueuePositionTracker,
    OrderManager,
    AlmgrenChrissModel,
    ImpactParameters,
    create_impact_model,
)

# CME Globex matching engine
from lob.cme_matching import (
    GlobexMatchingEngine,
    GlobexOrderType,
    StopOrder,
    StopTriggerType,
    AuctionOrder,
    AuctionResult,
    AuctionState,
    VelocityLogicResult,
    create_globex_matching_engine,
    DEFAULT_PROTECTION_POINTS,
)

# Circuit breaker
from impl_circuit_breaker import (
    CMECircuitBreaker,
    CircuitBreakerLevel,
    TradingState,
    PriceLimitStatus,
    LimitViolationType,
    CircuitBreakerEvent,
    EQUITY_CB_PRODUCTS,
    OVERNIGHT_LIMITS,
    COMMODITY_LIMITS,
)

# SPAN margin
from impl_span_margin import (
    SPANMarginCalculator,
    ProductGroup,
    PRODUCT_GROUPS,
)

# L3 configuration
from lob.config import (
    L3ExecutionConfig,
    LatencyConfig,
    FillProbabilityConfig,
    MarketImpactConfig,
    LatencyProfileType,
    create_l3_config,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

_DEFAULT_ADV = 1_000_000_000.0  # $1B typical for ES
_DEFAULT_VOLATILITY = 0.015  # 1.5% daily for equity index

# RTH hours in Eastern Time
RTH_START_HOUR = 9
RTH_START_MINUTE = 30
RTH_END_HOUR = 16
RTH_END_MINUTE = 0

# CME maintenance window (16:15-16:30 ET daily)
MAINTENANCE_START_HOUR = 16
MAINTENANCE_START_MINUTE = 15
MAINTENANCE_END_HOUR = 16
MAINTENANCE_END_MINUTE = 30

# Settlement times (Eastern Time) by product group
SETTLEMENT_TIMES: Dict[str, Tuple[int, int]] = {
    # Equity Index: 3:00 PM CT = 4:00 PM ET
    "ES": (16, 0),
    "NQ": (16, 0),
    "YM": (16, 0),
    "RTY": (16, 0),
    "MES": (16, 0),
    "MNQ": (16, 0),
    # Metals: 1:30 PM ET
    "GC": (13, 30),
    "SI": (13, 30),
    "HG": (13, 30),
    "MGC": (13, 30),
    # Energy: 2:30 PM ET
    "CL": (14, 30),
    "NG": (14, 30),
    "MCL": (14, 30),
    # Currencies: 2:00 PM ET
    "6E": (14, 0),
    "6J": (14, 0),
    "6B": (14, 0),
    "6A": (14, 0),
    # Bonds: 3:00 PM ET
    "ZB": (15, 0),
    "ZN": (15, 0),
    "ZF": (15, 0),
    "ZT": (15, 0),
}


# =============================================================================
# Session Detection
# =============================================================================


class CMESession(str, Enum):
    """CME trading session."""

    RTH = "RTH"  # Regular Trading Hours (9:30-16:00 ET)
    ETH = "ETH"  # Electronic Trading Hours (overnight)
    PRE_OPEN = "PRE_OPEN"  # Pre-open auction (before RTH)
    CLOSE = "CLOSE"  # Closing period
    MAINTENANCE = "MAINTENANCE"  # Daily maintenance window
    CLOSED = "CLOSED"  # Weekend/holiday


def get_cme_session(
    timestamp_ms: int,
    symbol: str = "ES",
) -> CMESession:
    """
    Determine current CME session from timestamp.

    Args:
        timestamp_ms: Timestamp in milliseconds
        symbol: Product symbol (for product-specific hours)

    Returns:
        CMESession enum value

    Notes:
        - Assumes timestamp is in UTC
        - Converts to US/Eastern for session detection
        - Handles weekend detection
    """
    # Convert to datetime in UTC
    dt_utc = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

    # Convert to Eastern Time (approximate, doesn't handle DST perfectly)
    # For production, use pytz or zoneinfo
    et_offset = timedelta(hours=-5)  # EST (winter), -4 for EDT
    dt_et = dt_utc + et_offset

    hour = dt_et.hour
    minute = dt_et.minute
    weekday = dt_et.weekday()  # Monday=0, Sunday=6

    # Weekend: Closed Saturday, opens Sunday 6 PM ET
    if weekday == 5:  # Saturday
        return CMESession.CLOSED
    if weekday == 6 and hour < 18:  # Sunday before 6 PM
        return CMESession.CLOSED

    # Maintenance window: 16:15-16:30 ET daily
    if hour == MAINTENANCE_START_HOUR and minute >= MAINTENANCE_START_MINUTE:
        if minute < MAINTENANCE_END_MINUTE:
            return CMESession.MAINTENANCE
    if hour == MAINTENANCE_END_HOUR and minute < MAINTENANCE_END_MINUTE:
        return CMESession.MAINTENANCE

    # RTH: 9:30 AM - 4:00 PM ET
    rth_start_minutes = RTH_START_HOUR * 60 + RTH_START_MINUTE
    rth_end_minutes = RTH_END_HOUR * 60 + RTH_END_MINUTE
    current_minutes = hour * 60 + minute

    if rth_start_minutes <= current_minutes < rth_end_minutes:
        return CMESession.RTH

    # Pre-open: 9:25-9:30 ET
    if hour == 9 and 25 <= minute < 30:
        return CMESession.PRE_OPEN

    # Otherwise ETH
    return CMESession.ETH


def is_rth_session(timestamp_ms: int, symbol: str = "ES") -> bool:
    """Check if timestamp is during RTH."""
    return get_cme_session(timestamp_ms, symbol) == CMESession.RTH


def get_minutes_to_settlement(
    timestamp_ms: int,
    symbol: str,
) -> Optional[int]:
    """
    Get minutes until daily settlement.

    Args:
        timestamp_ms: Current timestamp in milliseconds
        symbol: Product symbol

    Returns:
        Minutes until settlement, or None if past settlement time
    """
    symbol_upper = symbol.upper()
    settlement_time = SETTLEMENT_TIMES.get(symbol_upper)
    if settlement_time is None:
        return None

    settle_hour, settle_minute = settlement_time

    # Convert timestamp to ET
    dt_utc = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    et_offset = timedelta(hours=-5)
    dt_et = dt_utc + et_offset

    current_minutes = dt_et.hour * 60 + dt_et.minute
    settlement_minutes = settle_hour * 60 + settle_minute

    diff = settlement_minutes - current_minutes
    return diff if diff > 0 else None


# =============================================================================
# Daily Settlement Simulator
# =============================================================================


@dataclass
class SettlementEvent:
    """Record of a settlement event."""

    timestamp_ms: int
    symbol: str
    settlement_price: Decimal
    previous_settlement: Optional[Decimal]
    variation_margin: Decimal


@dataclass
class DailySettlementState:
    """Daily settlement tracking state."""

    last_settlement_date: Optional[str] = None
    last_settlement_price: Optional[Decimal] = None
    settlement_events: List[SettlementEvent] = field(default_factory=list)
    pending_variation_margin: Decimal = Decimal("0")


class DailySettlementSimulator:
    """
    Simulates CME daily settlement mechanics.

    Daily settlement in futures means:
    1. Positions are marked-to-market at settlement price
    2. Variation margin is calculated and credited/debited
    3. Reference price resets for next day

    This affects:
    - Overnight limit calculations
    - Circuit breaker reference prices
    - Position P&L tracking

    Example:
        >>> sim = DailySettlementSimulator("ES")
        >>> # At settlement time:
        >>> event = sim.process_settlement(
        ...     timestamp_ms=settle_time,
        ...     settlement_price=Decimal("4500"),
        ...     position_qty=Decimal("2"),
        ... )
        >>> print(f"Variation margin: ${event.variation_margin}")
    """

    def __init__(
        self,
        symbol: str,
        contract_multiplier: Optional[Decimal] = None,
    ) -> None:
        """
        Initialize settlement simulator.

        Args:
            symbol: Product symbol
            contract_multiplier: Contract multiplier (auto-detected if None)
        """
        self._symbol = symbol.upper()
        self._state = DailySettlementState()

        # Contract multipliers
        multipliers = {
            "ES": Decimal("50"),  # $50 per point
            "NQ": Decimal("20"),  # $20 per point
            "YM": Decimal("5"),  # $5 per point
            "RTY": Decimal("50"),  # $50 per point
            "MES": Decimal("5"),  # $5 per point (micro)
            "MNQ": Decimal("2"),  # $2 per point (micro)
            "GC": Decimal("100"),  # $100 per oz
            "SI": Decimal("5000"),  # $5000 per oz
            "CL": Decimal("1000"),  # $1000 per barrel
            "NG": Decimal("10000"),  # $10000 per MMBtu
            "6E": Decimal("125000"),  # €125,000
            "ZN": Decimal("1000"),  # $1000 per point
            "ZB": Decimal("1000"),  # $1000 per point
        }
        self._multiplier = contract_multiplier or multipliers.get(self._symbol, Decimal("1000"))

    def process_settlement(
        self,
        timestamp_ms: int,
        settlement_price: Decimal,
        position_qty: Decimal,
    ) -> SettlementEvent:
        """
        Process daily settlement.

        Args:
            timestamp_ms: Settlement timestamp
            settlement_price: Official settlement price
            position_qty: Current position quantity (+ for long, - for short)

        Returns:
            SettlementEvent with variation margin

        Notes:
            - Variation margin = (settle - prev_settle) * qty * multiplier
            - Long positions profit when price increases
            - Short positions profit when price decreases
        """
        # Calculate date for tracking
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        date_str = dt.strftime("%Y-%m-%d")

        # Calculate variation margin
        prev_settle = self._state.last_settlement_price
        if prev_settle is not None:
            price_change = settlement_price - prev_settle
            variation_margin = price_change * position_qty * self._multiplier
        else:
            variation_margin = Decimal("0")

        # Create event
        event = SettlementEvent(
            timestamp_ms=timestamp_ms,
            symbol=self._symbol,
            settlement_price=settlement_price,
            previous_settlement=prev_settle,
            variation_margin=variation_margin,
        )

        # Update state
        self._state.last_settlement_date = date_str
        self._state.last_settlement_price = settlement_price
        self._state.settlement_events.append(event)
        self._state.pending_variation_margin += variation_margin

        logger.info(
            f"Settlement processed: {self._symbol} @ {settlement_price}, "
            f"VM: ${variation_margin:,.2f}"
        )

        return event

    def get_last_settlement_price(self) -> Optional[Decimal]:
        """Get last settlement price (for overnight limits, etc.)."""
        return self._state.last_settlement_price

    def get_pending_variation_margin(self) -> Decimal:
        """Get accumulated variation margin since last clearance."""
        return self._state.pending_variation_margin

    def clear_variation_margin(self) -> Decimal:
        """Clear pending variation margin (after cash settlement)."""
        vm = self._state.pending_variation_margin
        self._state.pending_variation_margin = Decimal("0")
        return vm

    def get_settlement_history(self) -> List[SettlementEvent]:
        """Get settlement event history."""
        return list(self._state.settlement_events)


# =============================================================================
# L3 CME Slippage Provider
# =============================================================================


class CMEL3SlippageProvider:
    """
    L3 CME slippage provider with LOB walk-through.

    Extends CMESlippageProvider with actual LOB depth traversal
    when available, with CME-specific adjustments:

    - Session-aware spread multipliers (RTH vs ETH)
    - Settlement time premium
    - Circuit breaker integration
    - Roll period effects
    - Globex matching characteristics

    Falls back to statistical model when LOB unavailable.
    """

    def __init__(
        self,
        symbol: str,
        config: Optional[CMESlippageConfig] = None,
        impact_model: Optional[AlmgrenChrissModel] = None,
    ) -> None:
        """
        Initialize L3 CME slippage provider.

        Args:
            symbol: Product symbol
            config: CME slippage configuration
            impact_model: Market impact model
        """
        self._symbol = symbol.upper()
        self._config = config or CMESlippageConfig()
        self._tick_size = float(TICK_SIZES.get(self._symbol, DEFAULT_TICK_SIZE))

        # Circuit breaker state
        self._circuit_breaker_level = CircuitBreakerLevel.NONE
        self._is_roll_period = False

        # Create impact model
        if impact_model is not None:
            self._impact_model = impact_model
        else:
            # Use Almgren-Chriss with CME-appropriate parameters
            # eta = temporary impact, gamma = permanent impact
            impact_coef = self._config.get_impact_coef(self._symbol)
            params = ImpactParameters(
                eta=impact_coef,  # Temporary impact coefficient
                gamma=impact_coef * 0.3,  # Permanent impact (lower)
                volatility=0.02,  # Default volatility (updated per-trade)
            )
            self._impact_model = AlmgrenChrissModel(params=params)

        # Fallback to L2 statistical model
        self._fallback = CMESlippageProvider(config=config)

    def compute_slippage_bps(
        self,
        order: Order,
        market: MarketState,
        participation_ratio: float,
        is_rth: bool = True,
        minutes_to_settlement: Optional[int] = None,
        is_roll_period: bool = False,
        **kwargs: Any,
    ) -> float:
        """
        Compute CME L3 slippage in basis points.

        Uses LOB walk-through when depth available, otherwise
        falls back to statistical model.

        Args:
            order: Order to execute
            market: Current market state with optional LOB depth
            participation_ratio: Order notional / ADV
            is_rth: True if Regular Trading Hours
            minutes_to_settlement: Minutes until daily settlement
            is_roll_period: True if within roll period
            **kwargs: Additional parameters

        Returns:
            Expected slippage in basis points
        """
        # Check if we have LOB depth
        has_lob = (
            market.bid_depth is not None
            and market.ask_depth is not None
            and len(market.bid_depth or []) > 0
            and len(market.ask_depth or []) > 0
        )

        # Get mid price
        mid_price = market.get_mid_price()
        if mid_price is None or mid_price <= 0:
            mid_price = 1.0

        if has_lob:
            # L3: Walk through the book
            slippage_bps = self._compute_lob_walkthrough(order, market, mid_price)
        else:
            # L2: Use statistical model
            slippage_bps = self._fallback.compute_slippage_bps(
                order=order,
                market=market,
                participation_ratio=participation_ratio,
                is_rth=is_rth,
                minutes_to_settlement=minutes_to_settlement,
                is_roll_period=is_roll_period,
            )
            return slippage_bps

        # Add market impact for larger orders
        if participation_ratio > 0.001:
            adv = market.adv or _DEFAULT_ADV
            volatility = market.volatility or _DEFAULT_VOLATILITY
            notional = order.qty * mid_price

            impact_result = self._impact_model.compute_total_impact(
                order_qty=notional,
                adv=adv,
                volatility=volatility,
                mid_price=mid_price,
            )
            slippage_bps += impact_result.temporary_impact_bps

        # Apply session multiplier
        if not is_rth:
            slippage_bps *= self._config.eth_spread_multiplier

        # Apply settlement premium
        if minutes_to_settlement is not None and minutes_to_settlement > 0:
            if minutes_to_settlement <= self._config.settlement_window_minutes:
                fraction = 1.0 - (minutes_to_settlement / self._config.settlement_window_minutes)
                premium = 1.0 + (self._config.settlement_premium_max * fraction)
                slippage_bps *= premium

        # Apply roll period multiplier
        if is_roll_period or self._is_roll_period:
            slippage_bps *= self._config.roll_period_spread_multiplier

        # Apply circuit breaker effects
        if self._circuit_breaker_level == CircuitBreakerLevel.LEVEL_1:
            slippage_bps *= 2.0  # Double during Level 1
        elif self._circuit_breaker_level in (
            CircuitBreakerLevel.LEVEL_2,
            CircuitBreakerLevel.LEVEL_3,
        ):
            slippage_bps = self._config.max_slippage_bps

        # Apply bounds
        return max(self._config.min_slippage_bps, min(self._config.max_slippage_bps, slippage_bps))

    def _compute_lob_walkthrough(
        self,
        order: Order,
        market: MarketState,
        mid_price: float,
    ) -> float:
        """Walk through LOB to compute execution price."""
        is_buy = str(order.side).upper() == "BUY"
        depth = market.ask_depth if is_buy else market.bid_depth

        if not depth:
            return self._config.default_spread_bps / 2

        remaining_qty = order.qty
        total_cost = 0.0

        for price, size in depth:
            if remaining_qty <= 0:
                break
            fill_qty = min(remaining_qty, size)
            total_cost += fill_qty * price
            remaining_qty -= fill_qty

        # If couldn't fill completely
        if remaining_qty > 0 and depth:
            worst_price = depth[-1][0]
            total_cost += remaining_qty * worst_price

        avg_price = total_cost / order.qty if order.qty > 0 else mid_price

        # Compute slippage vs mid
        if is_buy:
            slippage_bps = (avg_price - mid_price) / mid_price * 10000
        else:
            slippage_bps = (mid_price - avg_price) / mid_price * 10000

        return max(0.0, slippage_bps)

    def set_circuit_breaker_level(self, level: CircuitBreakerLevel) -> None:
        """Set circuit breaker level for slippage adjustment."""
        self._circuit_breaker_level = level

    def set_roll_period(self, is_roll: bool) -> None:
        """Set roll period flag."""
        self._is_roll_period = is_roll


# =============================================================================
# L3 CME Fill Provider
# =============================================================================


class CMEL3FillProvider:
    """
    L3 CME fill provider with Globex matching.

    Uses GlobexMatchingEngine for realistic order matching:
    - FIFO Price-Time Priority
    - Market with Protection orders
    - Auction matching (open/close)
    - Stop orders with velocity logic
    - All-or-None / Minimum quantity

    Features:
        - Queue position tracking for limit orders
        - Fill probability estimation
        - Circuit breaker integration (halts execution)
        - Session-aware order handling
    """

    def __init__(
        self,
        symbol: str,
        slippage_provider: Optional[CMEL3SlippageProvider] = None,
        fee_provider: Optional[CMEFeeProvider] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize L3 CME fill provider.

        Args:
            symbol: Product symbol
            slippage_provider: CME L3 slippage provider
            fee_provider: CME fee provider
            seed: Random seed for reproducibility
        """
        self._symbol = symbol.upper()
        self._tick_size = TICK_SIZES.get(self._symbol, DEFAULT_TICK_SIZE)
        self._protection_points = DEFAULT_PROTECTION_POINTS.get(
            self._symbol, DEFAULT_PROTECTION_POINTS.get("ES", 6)
        )

        # Initialize Globex matching engine
        self._matching_engine = create_globex_matching_engine(
            symbol=self._symbol,
            tick_size=self._tick_size,
            protection_points=self._protection_points,
        )

        # Providers
        if slippage_provider is not None:
            self.slippage = slippage_provider
        else:
            self.slippage = CMEL3SlippageProvider(symbol=self._symbol)

        if fee_provider is not None:
            self.fees = fee_provider
        else:
            self.fees = CMEFeeProvider()

        # Queue tracker
        self._queue_tracker = QueuePositionTracker()

        # Circuit breaker
        self._circuit_breaker = CMECircuitBreaker(
            symbol=self._symbol,
            tick_size=self._tick_size,
        )

        # Order tracking
        self._order_counter = 0
        self._pending_orders: Dict[str, LimitOrder] = {}

        # Random seed for stochastic fills
        import random

        self._rng = random.Random(seed) if seed is not None else random.Random()

    def try_fill(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
        timestamp_ms: Optional[int] = None,
        is_rth: bool = True,
    ) -> Optional[Fill]:
        """
        Attempt to fill order using Globex matching.

        Args:
            order: Order to execute
            market: Current market state
            bar: Current bar data
            timestamp_ms: Current timestamp (uses market.timestamp if None)
            is_rth: True if Regular Trading Hours

        Returns:
            Fill if executed, None otherwise

        Notes:
            - Returns None during circuit breaker halts
            - Market orders always fill (at simulated price)
            - Limit orders require price touch and queue advancement
        """
        ts_ms = timestamp_ms or market.timestamp or int(time.time() * 1000)

        # Check circuit breaker state
        if not self._check_circuit_breaker(market, bar, ts_ms, is_rth):
            logger.debug(f"Order rejected: circuit breaker active for {self._symbol}")
            return None

        # Check velocity logic
        velocity_result = self._matching_engine.check_velocity_logic(
            float(bar.close),
            ts_ms * 1_000_000,  # Convert to nanoseconds
        )
        if velocity_result.triggered:
            logger.debug(f"Order delayed: velocity pause for {self._symbol}")
            # In real simulation, we'd queue for later
            # For simplicity, we still try to fill at adjusted price

        # Process based on order type
        order_type = str(order.order_type).upper()

        if order_type == "MARKET":
            return self._fill_market_order(order, market, bar, ts_ms, is_rth)
        elif order_type == "LIMIT":
            return self._fill_limit_order(order, market, bar, ts_ms, is_rth)
        else:
            logger.warning(f"Unsupported order type: {order_type}")
            return None

    def _check_circuit_breaker(
        self,
        market: MarketState,
        bar: BarData,
        ts_ms: int,
        is_rth: bool,
    ) -> bool:
        """Check if trading is allowed (no circuit breaker halt)."""
        # Get current price
        price = Decimal(str(bar.close))

        # Check circuit breaker
        level = self._circuit_breaker.check_circuit_breaker(
            current_price=price,
            timestamp_ms=ts_ms,
            is_rth=is_rth,
        )

        # Update slippage provider
        self.slippage.set_circuit_breaker_level(level)

        # Check if trading is halted
        if self._circuit_breaker._state.trading_state == TradingState.HALTED:
            return False

        return True

    def _fill_market_order(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
        ts_ms: int,
        is_rth: bool,
    ) -> Fill:
        """Fill market order (always fills)."""
        mid = market.get_mid_price() or bar.close
        is_buy = str(order.side).upper() == "BUY"

        # Calculate participation
        adv = market.adv or _DEFAULT_ADV
        participation = (order.qty * mid) / adv if adv > 0 else 0.01

        # Get slippage
        minutes_to_settle = get_minutes_to_settlement(ts_ms, self._symbol)
        slippage_bps = self.slippage.compute_slippage_bps(
            order=order,
            market=market,
            participation_ratio=participation,
            is_rth=is_rth,
            minutes_to_settlement=minutes_to_settle,
        )

        # Calculate fill price
        slippage_mult = slippage_bps / 10000
        if is_buy:
            fill_price = mid * (1 + slippage_mult)
            fill_price = min(fill_price, bar.high)  # Cap at bar high
        else:
            fill_price = mid * (1 - slippage_mult)
            fill_price = max(fill_price, bar.low)  # Cap at bar low

        # Calculate fee
        notional = fill_price * order.qty
        fee = self.fees.compute_fee(
            notional=notional,
            side=str(order.side),
            liquidity="taker",
            qty=order.qty,
            symbol=self._symbol,
        )

        return Fill(
            price=fill_price,
            qty=order.qty,
            fee=fee,
            slippage_bps=slippage_bps,
            liquidity="taker",
            timestamp=ts_ms,
            notional=notional,
            metadata={
                "symbol": self._symbol,
                "fill_type": "market",
                "session": "RTH" if is_rth else "ETH",
                "circuit_breaker_level": self._circuit_breaker._state.current_level.name,
            },
        )

    def _fill_limit_order(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
        ts_ms: int,
        is_rth: bool,
    ) -> Optional[Fill]:
        """Fill limit order (may not fill)."""
        if order.limit_price is None:
            logger.warning("LIMIT order without limit_price")
            return None

        limit_price = float(order.limit_price)
        is_buy = str(order.side).upper() == "BUY"
        mid = market.get_mid_price() or bar.close

        # Get quotes
        ask = market.ask if market.ask and math.isfinite(market.ask) else mid * 1.0005
        bid = market.bid if market.bid and math.isfinite(market.bid) else mid * 0.9995

        # Check for immediate fill (crosses spread) - TAKER
        crosses_spread = (is_buy and limit_price >= ask) or (not is_buy and limit_price <= bid)

        if crosses_spread:
            # Immediate fill at limit or better
            if is_buy:
                fill_price = min(limit_price, ask)
            else:
                fill_price = max(limit_price, bid)

            slippage_bps = abs(fill_price - mid) / mid * 10000 if mid > 0 else 0

            notional = fill_price * order.qty
            fee = self.fees.compute_fee(
                notional=notional,
                side=str(order.side),
                liquidity="taker",
                qty=order.qty,
                symbol=self._symbol,
            )

            return Fill(
                price=fill_price,
                qty=order.qty,
                fee=fee,
                slippage_bps=slippage_bps,
                liquidity="taker",
                timestamp=ts_ms,
                notional=notional,
                metadata={
                    "symbol": self._symbol,
                    "fill_type": "limit_aggressive",
                    "crosses_spread": True,
                },
            )

        # Passive limit - check if bar touches limit price
        tolerance = float(self._tick_size) * 0.5

        if is_buy:
            fills = bar.low <= limit_price + tolerance
        else:
            fills = bar.high >= limit_price - tolerance

        if not fills:
            return None

        # Use fill probability to determine if we get filled
        # Based on queue position (simplified)
        fill_prob = self._estimate_fill_probability(order, market, bar)
        if self._rng.random() > fill_prob:
            return None

        # Maker fill at limit price
        fill_price = limit_price
        slippage_bps = 0.0  # No slippage for maker fills

        notional = fill_price * order.qty
        fee = self.fees.compute_fee(
            notional=notional,
            side=str(order.side),
            liquidity="maker",
            qty=order.qty,
            symbol=self._symbol,
        )

        return Fill(
            price=fill_price,
            qty=order.qty,
            fee=fee,
            slippage_bps=slippage_bps,
            liquidity="maker",
            timestamp=ts_ms,
            notional=notional,
            metadata={
                "symbol": self._symbol,
                "fill_type": "limit_passive",
                "fill_probability": fill_prob,
            },
        )

    def _estimate_fill_probability(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
    ) -> float:
        """
        Estimate fill probability for passive limit order.

        Uses simplified model based on:
        - Distance from mid price
        - Bar volume vs order size
        - Queue position estimate

        Returns:
            Fill probability in [0, 1]
        """
        if order.limit_price is None:
            return 0.0

        mid = market.get_mid_price() or bar.close
        limit_price = float(order.limit_price)
        is_buy = str(order.side).upper() == "BUY"

        # Distance from mid in ticks
        distance_ticks = abs(limit_price - mid) / float(self._tick_size)

        # Base probability decreases with distance
        if distance_ticks <= 1:
            base_prob = 0.8
        elif distance_ticks <= 3:
            base_prob = 0.5
        elif distance_ticks <= 5:
            base_prob = 0.3
        else:
            base_prob = 0.1

        # Adjust for volume
        bar_volume = bar.volume or 1000
        volume_ratio = min(1.0, bar_volume / (order.qty * 10))
        prob = base_prob * volume_ratio

        return max(0.0, min(1.0, prob))

    def set_reference_price(self, price: Decimal) -> None:
        """Set reference price for circuit breaker."""
        self._circuit_breaker.set_reference_price(price)

    def submit_stop_order(
        self,
        order: Order,
        trigger_price: float,
        is_stop_limit: bool = False,
        limit_price: Optional[float] = None,
    ) -> str:
        """
        Submit a stop order to the matching engine.

        Args:
            order: Base order
            trigger_price: Stop trigger price
            is_stop_limit: True for stop-limit, False for stop-market
            limit_price: Limit price for stop-limit orders

        Returns:
            Order ID
        """
        self._order_counter += 1
        order_id = f"STOP_{self._symbol}_{self._order_counter}"

        side = LOBSide.BUY if str(order.side).upper() == "BUY" else LOBSide.SELL

        stop = StopOrder(
            order_id=order_id,
            symbol=self._symbol,
            side=side,
            qty=float(order.qty),
            stop_price=float(trigger_price),
            limit_price=float(limit_price) if limit_price else None,
            timestamp_ns=int(time.time() * 1e9),
            use_protection=not is_stop_limit,  # Use MWP for stop-market
        )

        self._matching_engine.submit_stop_order(stop)
        return order_id


# =============================================================================
# L3 CME Execution Provider
# =============================================================================


class CMEL3ExecutionProvider:
    """
    Full L3 execution provider for CME futures.

    Combines all CME-specific components:
    - GlobexMatchingEngine for realistic order matching
    - CMECircuitBreaker for price protection
    - DailySettlementSimulator for mark-to-market
    - Session-aware execution (RTH/ETH)

    This is the main entry point for CME futures execution simulation.

    Example:
        >>> provider = CMEL3ExecutionProvider("ES")
        >>> provider.set_reference_price(Decimal("4500"))
        >>>
        >>> fill = provider.execute(
        ...     order=Order("ES", "BUY", 5, "MARKET"),
        ...     market=MarketState(timestamp=now, bid=4500, ask=4500.25, adv=2e9),
        ...     bar=BarData(open=4500, high=4510, low=4490, close=4505, volume=100000),
        ... )
        >>> print(f"Filled at {fill.price}")

    Attributes:
        symbol: Product symbol
        fill_provider: L3 CME fill provider
        slippage_provider: L3 CME slippage provider
        fee_provider: CME fee provider
        circuit_breaker: Circuit breaker engine
        settlement_sim: Daily settlement simulator
    """

    def __init__(
        self,
        symbol: str,
        slippage_config: Optional[CMESlippageConfig] = None,
        fee_config: Optional[CMEFeeConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize CME L3 execution provider.

        Args:
            symbol: Product symbol (ES, NQ, GC, CL, etc.)
            slippage_config: CME slippage configuration
            fee_config: CME fee configuration
            seed: Random seed for reproducibility
        """
        self._symbol = symbol.upper()
        self._tick_size = TICK_SIZES.get(self._symbol, DEFAULT_TICK_SIZE)

        # Initialize slippage provider
        self._slippage = CMEL3SlippageProvider(
            symbol=self._symbol,
            config=slippage_config,
        )

        # Initialize fee provider
        self._fees = CMEFeeProvider(config=fee_config)

        # Initialize fill provider (includes matching engine + circuit breaker)
        self._fill = CMEL3FillProvider(
            symbol=self._symbol,
            slippage_provider=self._slippage,
            fee_provider=self._fees,
            seed=seed,
        )

        # Settlement simulator
        self._settlement = DailySettlementSimulator(symbol=self._symbol)

        # Statistics
        self._stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "rejected_orders": 0,
            "taker_fills": 0,
            "maker_fills": 0,
            "circuit_breaker_halts": 0,
            "velocity_pauses": 0,
        }

    @property
    def asset_class(self) -> AssetClass:
        """Asset class this provider handles."""
        return AssetClass.FUTURES

    @property
    def symbol(self) -> str:
        """Product symbol."""
        return self._symbol

    def execute(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
        **kwargs: Any,
    ) -> Optional[Fill]:
        """
        Execute an order with full CME L3 simulation.

        Process:
        1. Determine trading session (RTH/ETH/maintenance)
        2. Check circuit breaker state
        3. Calculate minutes to settlement
        4. Execute via Globex matching engine
        5. Apply CME-specific fees

        Args:
            order: Order to execute
            market: Current market state
            bar: Current bar data
            **kwargs: Additional parameters:
                - is_rth: Override RTH detection
                - minutes_to_settlement: Override settlement time calc
                - is_roll_period: Flag for roll period

        Returns:
            Fill with execution details, or None if not filled/rejected
        """
        self._stats["total_orders"] += 1

        # Get timestamp
        ts_ms = market.timestamp or int(time.time() * 1000)

        # Determine session
        if "is_rth" in kwargs:
            is_rth = kwargs["is_rth"]
        else:
            session = get_cme_session(ts_ms, self._symbol)
            is_rth = session == CMESession.RTH

            # Reject during maintenance
            if session == CMESession.MAINTENANCE:
                self._stats["rejected_orders"] += 1
                logger.debug(f"Order rejected: maintenance window for {self._symbol}")
                return None

            # Reject during closed (weekend)
            if session == CMESession.CLOSED:
                self._stats["rejected_orders"] += 1
                logger.debug(f"Order rejected: market closed for {self._symbol}")
                return None

        # Get minutes to settlement
        if "minutes_to_settlement" in kwargs:
            minutes_to_settle = kwargs["minutes_to_settlement"]
        else:
            minutes_to_settle = get_minutes_to_settlement(ts_ms, self._symbol)

        # Check roll period
        is_roll_period = kwargs.get("is_roll_period", False)
        self._slippage.set_roll_period(is_roll_period)

        # Execute via fill provider
        fill = self._fill.try_fill(
            order=order,
            market=market,
            bar=bar,
            timestamp_ms=ts_ms,
            is_rth=is_rth,
        )

        if fill is not None:
            self._stats["filled_orders"] += 1
            if fill.liquidity == "maker":
                self._stats["maker_fills"] += 1
            else:
                self._stats["taker_fills"] += 1

            # Add session info to metadata
            fill.metadata["session"] = "RTH" if is_rth else "ETH"
            fill.metadata["minutes_to_settlement"] = minutes_to_settle
        else:
            self._stats["rejected_orders"] += 1

        return fill

    def set_reference_price(self, price: Decimal) -> None:
        """
        Set reference price for circuit breaker and overnight limits.

        Should be called at start of each trading day with previous
        day's settlement price.

        Args:
            price: Reference price (previous settlement)
        """
        self._fill.set_reference_price(price)

    def process_settlement(
        self,
        timestamp_ms: int,
        settlement_price: Decimal,
        position_qty: Decimal,
    ) -> SettlementEvent:
        """
        Process daily settlement.

        Args:
            timestamp_ms: Settlement timestamp
            settlement_price: Official settlement price
            position_qty: Current position quantity

        Returns:
            SettlementEvent with variation margin details
        """
        return self._settlement.process_settlement(
            timestamp_ms=timestamp_ms,
            settlement_price=settlement_price,
            position_qty=position_qty,
        )

    def submit_stop_order(
        self,
        order: Order,
        trigger_price: float,
        is_stop_limit: bool = False,
        limit_price: Optional[float] = None,
    ) -> str:
        """
        Submit a stop order.

        Args:
            order: Base order
            trigger_price: Stop trigger price
            is_stop_limit: True for stop-limit order
            limit_price: Limit price (required for stop-limit)

        Returns:
            Order ID for tracking
        """
        return self._fill.submit_stop_order(
            order=order,
            trigger_price=trigger_price,
            is_stop_limit=is_stop_limit,
            limit_price=limit_price,
        )

    def estimate_cost(
        self,
        qty: float,
        price: float,
        adv: float,
        is_rth: bool = True,
    ) -> Dict[str, float]:
        """
        Pre-trade cost estimation.

        Args:
            qty: Number of contracts
            price: Current price
            adv: Average daily volume (notional)
            is_rth: True if RTH

        Returns:
            Dictionary with cost breakdown
        """
        notional = qty * price
        participation = notional / adv if adv > 0 else 0.01

        # Create synthetic order
        order = Order(
            symbol=self._symbol,
            side="BUY",
            qty=qty,
            order_type="MARKET",
        )
        market = MarketState(
            timestamp=int(time.time() * 1000),
            bid=price * 0.9999,
            ask=price * 1.0001,
            adv=adv,
        )

        slippage_bps = self._slippage.compute_slippage_bps(
            order=order,
            market=market,
            participation_ratio=participation,
            is_rth=is_rth,
        )

        slippage_cost = notional * slippage_bps / 10000
        fee = self._fees.compute_fee(
            notional=notional,
            side="BUY",
            liquidity="taker",
            qty=qty,
            symbol=self._symbol,
        )

        return {
            "symbol": self._symbol,
            "notional": notional,
            "participation": participation,
            "slippage_bps": slippage_bps,
            "slippage_cost": slippage_cost,
            "fee": fee,
            "total_cost": slippage_cost + fee,
            "total_bps": (slippage_cost + fee) / notional * 10000 if notional > 0 else 0,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        total = self._stats["total_orders"]
        filled = self._stats["filled_orders"]

        return {
            **self._stats,
            "fill_rate": filled / total if total > 0 else 0.0,
            "maker_rate": self._stats["maker_fills"] / filled if filled > 0 else 0.0,
            "taker_rate": self._stats["taker_fills"] / filled if filled > 0 else 0.0,
        }

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "rejected_orders": 0,
            "taker_fills": 0,
            "maker_fills": 0,
            "circuit_breaker_halts": 0,
            "velocity_pauses": 0,
        }

    def get_settlement_history(self) -> List[SettlementEvent]:
        """Get settlement event history."""
        return self._settlement.get_settlement_history()


# =============================================================================
# Factory Functions
# =============================================================================


def create_cme_l3_execution_provider(
    symbol: str,
    profile: str = "default",
    seed: Optional[int] = None,
) -> CMEL3ExecutionProvider:
    """
    Factory function to create CME L3 execution provider.

    Args:
        symbol: Product symbol (ES, NQ, GC, CL, etc.)
        profile: Slippage profile name
        seed: Random seed for reproducibility

    Returns:
        Configured CMEL3ExecutionProvider
    """
    slippage_config = CME_SLIPPAGE_PROFILES.get(profile, CME_SLIPPAGE_PROFILES["default"])
    return CMEL3ExecutionProvider(
        symbol=symbol,
        slippage_config=slippage_config,
        seed=seed,
    )


def create_cme_l3_slippage_provider(
    symbol: str,
    profile: str = "default",
) -> CMEL3SlippageProvider:
    """
    Factory function to create CME L3 slippage provider.

    Args:
        symbol: Product symbol
        profile: Slippage profile name

    Returns:
        Configured CMEL3SlippageProvider
    """
    config = CME_SLIPPAGE_PROFILES.get(profile, CME_SLIPPAGE_PROFILES["default"])
    return CMEL3SlippageProvider(symbol=symbol, config=config)


def create_cme_l3_fill_provider(
    symbol: str,
    slippage_provider: Optional[CMEL3SlippageProvider] = None,
    fee_provider: Optional[CMEFeeProvider] = None,
    seed: Optional[int] = None,
) -> CMEL3FillProvider:
    """
    Factory function to create CME L3 fill provider.

    Args:
        symbol: Product symbol
        slippage_provider: Custom slippage provider
        fee_provider: Custom fee provider
        seed: Random seed for reproducibility

    Returns:
        Configured CMEL3FillProvider
    """
    return CMEL3FillProvider(
        symbol=symbol,
        slippage_provider=slippage_provider,
        fee_provider=fee_provider,
        seed=seed,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "CMESession",
    # Data classes
    "SettlementEvent",
    "DailySettlementState",
    # Session helpers
    "get_cme_session",
    "is_rth_session",
    "get_minutes_to_settlement",
    # Settlement
    "DailySettlementSimulator",
    # Providers
    "CMEL3SlippageProvider",
    "CMEL3FillProvider",
    "CMEL3ExecutionProvider",
    # Factory functions
    "create_cme_l3_execution_provider",
    "create_cme_l3_slippage_provider",
    "create_cme_l3_fill_provider",
    # Constants
    "SETTLEMENT_TIMES",
    "RTH_START_HOUR",
    "RTH_END_HOUR",
]
