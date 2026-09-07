# -*- coding: utf-8 -*-
"""
CME Globex Matching Engine Emulator.

This module extends the base FIFO matching engine with CME Globex-specific
matching rules and order types that are critical for realistic CME futures
simulation.

CME Globex-Specific Features:
============================

1. MARKET WITH PROTECTION (MWP)
   - Market orders with implicit price protection
   - Buy MWP: Limit = Best Ask + Protection Points × Tick Size
   - Sell MWP: Limit = Best Bid - Protection Points × Tick Size
   - Protection points vary by product (typically 10-30 ticks)
   - Prevents sweeping the book during thin liquidity

2. OPENING/CLOSING AUCTIONS
   - Opening auction at session start (6:00 PM ET for ETH, 9:30 AM for RTH)
   - Closing auction during final minutes of RTH
   - Equilibrium price calculation maximizing matched volume
   - Imbalance indicator publication

3. STOP ORDER HANDLING
   - Stop Market: Converts to MWP when triggered
   - Stop Limit: Converts to limit order when triggered
   - Triggering based on last trade price (not bid/ask)
   - Integration with Velocity Logic (brief pause on rapid triggers)

4. MINIMUM QUANTITY ORDERS
   - All-or-None (AON): Fill all qty or nothing
   - Minimum Quantity: Fill at least min_qty or cancel

5. ICE BERG (ICEBERG) ORDERS
   - Show only display_qty at a time
   - When display_qty filled, replenish from hidden reserve
   - Loses time priority on replenishment

6. TRADE AT SETTLEMENT (TAS)
   - Orders to trade at daily settlement price
   - Executes during settlement window

Stage 5B of Futures Integration (v1.0)

References:
    - CME Globex Matching Algorithm
    - CME Market with Protection: https://www.cmegroup.com/trading/trading-on-globex/
    - CME Price Banding: https://www.cmegroup.com/trading/equity-index/
    - NYSE Auction Mechanisms (similar concepts)
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, IntEnum
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from lob.data_structures import (
    Fill,
    LimitOrder,
    OrderBook,
    OrderType,
    PriceLevel,
    Side,
    Trade,
)
from lob.matching_engine import (
    MatchingEngine,
    MatchResult,
    MatchType,
    OrderStatus,
    STPAction,
    STPResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Configuration
# =============================================================================

# Default protection points for Market with Protection (in ticks)
DEFAULT_PROTECTION_POINTS: Dict[str, int] = {
    # Equity Index - tight protection for liquid products
    "ES": 10,  # 10 ticks = 2.5 points = $125 per contract
    "NQ": 15,  # 15 ticks = 3.75 points
    "YM": 20,  # 20 points
    "RTY": 20,  # 20 ticks = 2.0 points
    "MES": 10,
    "MNQ": 15,
    # Metals - wider for less liquid
    "GC": 30,  # 30 ticks = $3
    "SI": 40,  # 40 ticks
    "HG": 40,
    # Energy
    "CL": 50,  # 50 ticks = $0.50
    "NG": 100,  # 100 ticks = $0.10 (volatile)
    # Currencies
    "6E": 20,
    "6J": 30,
    "6B": 20,
    # Bonds
    "ZB": 20,  # 20 ticks
    "ZN": 15,
    "ZF": 15,
}

DEFAULT_PROTECTION_POINTS_FALLBACK = 20


# =============================================================================
# Enums
# =============================================================================


class GlobexOrderType(str, Enum):
    """Extended order types for CME Globex."""

    LIMIT = "LIMIT"
    MARKET = "MARKET"
    MARKET_WITH_PROTECTION = "MWP"  # Market with price protection
    STOP_MARKET = "STOP_MKT"
    STOP_LIMIT = "STOP_LMT"
    STOP_WITH_PROTECTION = "STOP_MWP"
    TRADE_AT_SETTLEMENT = "TAS"
    ALL_OR_NONE = "AON"


class AuctionState(str, Enum):
    """Auction state for opening/closing."""

    NOT_IN_AUCTION = "NOT_IN_AUCTION"
    COLLECTING = "COLLECTING"  # Accepting orders, no matching
    UNCROSSING = "UNCROSSING"  # Calculating equilibrium
    EXECUTING = "EXECUTING"  # Matching at equilibrium price
    COMPLETED = "COMPLETED"


class StopTriggerType(str, Enum):
    """Stop order trigger condition."""

    LAST_TRADE = "LAST_TRADE"  # CME default: trigger on last trade
    BID = "BID"  # Trigger on bid touch
    ASK = "ASK"  # Trigger on ask touch
    MID = "MID"  # Trigger on mid-price


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class StopOrder:
    """
    Stop order waiting to be triggered.

    Attributes:
        order_id: Unique order identifier
        symbol: Trading symbol
        side: BUY or SELL
        qty: Order quantity
        stop_price: Trigger price
        limit_price: Limit price for stop-limit (None for stop-market)
        trigger_type: Condition for triggering
        use_protection: If True, converts to MWP instead of market
        protection_points: Number of ticks for MWP (if applicable)
        timestamp_ns: Order submission time
        account_id: Account for STP
    """

    order_id: str
    symbol: str
    side: Side
    qty: float
    stop_price: float
    limit_price: Optional[float] = None
    trigger_type: StopTriggerType = StopTriggerType.LAST_TRADE
    use_protection: bool = True
    protection_points: Optional[int] = None
    timestamp_ns: int = 0
    account_id: Optional[str] = None

    @property
    def is_stop_limit(self) -> bool:
        """Check if this is a stop-limit order."""
        return self.limit_price is not None

    def should_trigger(
        self,
        last_trade_price: Optional[float] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
    ) -> bool:
        """
        Check if stop order should trigger.

        For BUY stops: trigger when price >= stop_price
        For SELL stops: trigger when price <= stop_price
        """
        trigger_price: Optional[float] = None

        if self.trigger_type == StopTriggerType.LAST_TRADE:
            trigger_price = last_trade_price
        elif self.trigger_type == StopTriggerType.BID:
            trigger_price = bid
        elif self.trigger_type == StopTriggerType.ASK:
            trigger_price = ask
        elif self.trigger_type == StopTriggerType.MID and bid and ask:
            trigger_price = (bid + ask) / 2

        if trigger_price is None:
            return False

        if self.side == Side.BUY:
            return trigger_price >= self.stop_price
        else:  # SELL
            return trigger_price <= self.stop_price


@dataclass
class AuctionOrder:
    """Order participating in auction."""

    order_id: str
    side: Side
    qty: float
    price: Optional[float]  # None for market orders
    timestamp_ns: int
    account_id: Optional[str] = None
    remaining_qty: float = field(init=False)

    def __post_init__(self) -> None:
        self.remaining_qty = self.qty


@dataclass
class AuctionResult:
    """
    Result of auction price calculation.

    Attributes:
        equilibrium_price: Calculated clearing price
        matched_volume: Total volume that can be matched
        buy_imbalance: Buy qty > sell qty at equilibrium
        sell_imbalance: Sell qty > buy qty at equilibrium
        imbalance_direction: Net order imbalance direction
        fills: List of fills from auction
    """

    equilibrium_price: float
    matched_volume: float
    buy_imbalance: float = 0.0
    sell_imbalance: float = 0.0
    imbalance_direction: str = "NONE"
    fills: List[Fill] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "AuctionResult":
        """Create empty auction result."""
        return cls(
            equilibrium_price=0.0,
            matched_volume=0.0,
            imbalance_direction="NONE",
        )


class VelocityLogicResult(NamedTuple):
    """Result of velocity logic check."""

    triggered: bool
    pause_duration_ms: int
    reason: str


# =============================================================================
# Globex Matching Engine
# =============================================================================


class GlobexMatchingEngine(MatchingEngine):
    """
    CME Globex-style matching engine with enhanced order types.

    Extends base FIFO matching engine with:
    - Market with Protection (MWP) orders
    - Opening/closing auction matching
    - Stop order handling with velocity logic
    - Minimum quantity / All-or-None orders
    - Trade at Settlement (TAS) orders

    Example Usage:
        >>> engine = GlobexMatchingEngine(
        ...     symbol="ES",
        ...     tick_size=0.25,
        ...     protection_points=10,
        ... )
        >>>
        >>> # Submit Market with Protection order
        >>> result = engine.match_with_protection(
        ...     order=LimitOrder(...),
        ...     protection_points=10,
        ... )
        >>>
        >>> # Run opening auction
        >>> engine.start_auction(AuctionState.COLLECTING)
        >>> engine.add_auction_order(order1)
        >>> engine.add_auction_order(order2)
        >>> auction_result = engine.execute_auction()

    References:
        CME Globex Matching Algorithm
        CME Market with Protection specification
    """

    def __init__(
        self,
        symbol: str,
        tick_size: float = 0.01,
        protection_points: Optional[int] = None,
        enable_velocity_logic: bool = True,
        velocity_threshold_ticks: int = 50,
        velocity_pause_ms: int = 2000,
        enable_stop_spike_logic: bool = True,
        stop_spike_threshold: int = 5,
        stop_spike_delay_ms: int = 500,
        stp_action: STPAction = STPAction.CANCEL_NEWEST,
        account_id: Optional[str] = None,
    ) -> None:
        """
        Initialize Globex matching engine.

        Args:
            symbol: Trading symbol (ES, NQ, GC, etc.)
            tick_size: Minimum price increment
            protection_points: MWP protection in ticks (uses product default if None)
            enable_velocity_logic: Enable velocity logic protection
            velocity_threshold_ticks: Price move threshold for velocity logic
            velocity_pause_ms: Pause duration when velocity logic triggers
            enable_stop_spike_logic: Enable stop cascade protection
            stop_spike_threshold: Number of stops to trigger protection
            stop_spike_delay_ms: Delay to add to stop executions
            stp_action: Self-trade prevention action
            account_id: Default account ID for orders
        """
        super().__init__(
            stp_action=stp_action,
            enable_stp=True,
        )

        self._symbol = symbol
        self._tick_size = tick_size
        self._account_id = account_id

        # Internal order book
        self._order_book = OrderBook()

        # Protection points for MWP
        if protection_points is not None:
            self._protection_points = protection_points
        else:
            self._protection_points = DEFAULT_PROTECTION_POINTS.get(
                symbol.upper(),
                DEFAULT_PROTECTION_POINTS_FALLBACK,
            )

        # Velocity logic
        self._enable_velocity_logic = enable_velocity_logic
        self._velocity_threshold_ticks = velocity_threshold_ticks
        self._velocity_pause_ms = velocity_pause_ms
        self._last_trade_price: Optional[float] = None
        self._velocity_pause_until_ns: int = 0

        # Stop spike logic
        self._enable_stop_spike_logic = enable_stop_spike_logic
        self._stop_spike_threshold = stop_spike_threshold
        self._stop_spike_delay_ms = stop_spike_delay_ms
        self._pending_stop_count = 0
        self._stop_executions_in_window: Deque[int] = deque(maxlen=100)

        # Stop orders
        self._stop_orders: Dict[str, StopOrder] = {}

        # Auction state
        self._auction_state = AuctionState.NOT_IN_AUCTION
        self._auction_orders: List[AuctionOrder] = []

        # TAS orders
        self._tas_orders: List[AuctionOrder] = []
        self._settlement_price: Optional[float] = None

        # Statistics
        self._mwp_order_count = 0
        self._stop_trigger_count = 0
        self._auction_count = 0
        self._velocity_trigger_count = 0

    # =========================================================================
    # Core Order Book Operations
    # =========================================================================

    def add_order(self, order: LimitOrder) -> MatchResult:
        """
        Add a limit order to the book.

        If the order is aggressive (crosses the spread), it will be matched first.
        Any resting portion is added to the order book.

        Args:
            order: LimitOrder to add

        Returns:
            MatchResult with fills and resting order info
        """
        return self.match(order)

    def match(self, order: LimitOrder) -> MatchResult:
        """
        Match a limit order against the internal order book.

        Args:
            order: Order to match

        Returns:
            MatchResult with fills and resting order
        """
        result = self.match_limit_order(order, self._order_book)

        # Add resting order to book if any
        if result.resting_order is not None:
            self._order_book.add_limit_order(result.resting_order)

        return result

    def get_order_book(self) -> OrderBook:
        """Get the internal order book."""
        return self._order_book

    def get_best_bid(self) -> Optional[float]:
        """Get the best bid price."""
        return self._order_book.best_bid

    def get_best_ask(self) -> Optional[float]:
        """Get the best ask price."""
        return self._order_book.best_ask

    def get_spread(self) -> Optional[float]:
        """Get the current bid-ask spread."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is not None and ask is not None:
            return ask - bid
        return None

    def clear(self) -> None:
        """Clear the order book and all pending orders."""
        self._order_book = OrderBook()
        self._stop_orders.clear()
        self._auction_orders.clear()
        self._tas_orders.clear()
        self._last_trade_price = None
        self._velocity_pause_until_ns = 0
        self._pending_stop_count = 0
        self._stop_executions_in_window.clear()

    # =========================================================================
    # Market with Protection (MWP)
    # =========================================================================

    def match_with_protection(
        self,
        order: LimitOrder,
        protection_points: Optional[int] = None,
    ) -> MatchResult:
        """
        Match Market with Protection order.

        MWP orders have an implicit limit price:
        - BUY: best_ask + protection_points × tick_size
        - SELL: best_bid - protection_points × tick_size

        This prevents sweeping the entire order book in thin markets.

        Args:
            order: Order to match (treated as market)
            protection_points: Override protection points (uses default if None)

        Returns:
            MatchResult with fills

        Notes:
            - If no liquidity, order is rejected (not rested)
            - Unfilled portion is cancelled
        """
        if protection_points is None:
            protection_points = self._protection_points

        # Get current best prices
        best_bid = self._order_book.best_bid
        best_ask = self._order_book.best_ask

        if order.side == Side.BUY:
            if best_ask is None:
                # No liquidity - reject
                return MatchResult(
                    fills=[],
                    total_filled_qty=0.0,
                    is_complete=False,
                    match_type=MatchType.TAKER,
                )
            limit_price = best_ask + (protection_points * self._tick_size)
        else:
            if best_bid is None:
                return MatchResult(
                    fills=[],
                    total_filled_qty=0.0,
                    is_complete=False,
                    match_type=MatchType.TAKER,
                )
            limit_price = best_bid - (protection_points * self._tick_size)

        # Create protected limit order
        protected_order = LimitOrder(
            order_id=order.order_id,
            price=limit_price,
            qty=order.qty,
            remaining_qty=order.remaining_qty,
            timestamp_ns=order.timestamp_ns,
            side=order.side,
            order_type=OrderType.LIMIT,
        )

        # Match as limit order
        result = self.match(protected_order)

        # MWP unfilled portion is cancelled, not rested
        if result.resting_order is not None:
            result = MatchResult(
                fills=result.fills,
                resting_order=None,  # Cancel unfilled
                cancelled_orders=result.cancelled_orders + [result.resting_order],
                total_filled_qty=result.total_filled_qty,
                avg_fill_price=result.avg_fill_price,
                is_complete=False,
                match_type=result.match_type,
            )

        self._mwp_order_count += 1
        return result

    def get_protection_limit(
        self,
        side: Side,
        protection_points: Optional[int] = None,
    ) -> Optional[float]:
        """
        Calculate MWP protection limit price.

        Args:
            side: Order side
            protection_points: Override protection points

        Returns:
            Protection limit price or None if no liquidity
        """
        if protection_points is None:
            protection_points = self._protection_points

        best_bid = self._order_book.best_bid
        best_ask = self._order_book.best_ask

        if side == Side.BUY:
            if best_ask is None:
                return None
            return best_ask + (protection_points * self._tick_size)
        else:
            if best_bid is None:
                return None
            return best_bid - (protection_points * self._tick_size)

    # =========================================================================
    # Stop Orders
    # =========================================================================

    def submit_stop_order(self, stop_order: StopOrder) -> bool:
        """
        Submit stop order to engine.

        Stop orders are held until triggered, then converted to:
        - Market with Protection (if use_protection=True)
        - Market order (if use_protection=False)
        - Limit order (if limit_price is set)

        Args:
            stop_order: Stop order to submit

        Returns:
            True if accepted, False if rejected
        """
        if stop_order.order_id in self._stop_orders:
            logger.warning(f"Duplicate stop order ID: {stop_order.order_id}")
            return False

        self._stop_orders[stop_order.order_id] = stop_order
        return True

    def cancel_stop_order(self, order_id: str) -> bool:
        """
        Cancel pending stop order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled, False if not found
        """
        if order_id in self._stop_orders:
            del self._stop_orders[order_id]
            return True
        return False

    def check_stops(
        self,
        last_trade_price: Optional[float] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        timestamp_ns: int = 0,
    ) -> List[MatchResult]:
        """Alias for check_stop_triggers for convenience."""
        return self.check_stop_triggers(last_trade_price, bid, ask, timestamp_ns)

    def check_stop_triggers(
        self,
        last_trade_price: Optional[float] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        timestamp_ns: int = 0,
    ) -> List[MatchResult]:
        """
        Check and execute triggered stop orders.

        Args:
            last_trade_price: Last trade price
            bid: Current best bid
            ask: Current best ask
            timestamp_ns: Current timestamp

        Returns:
            List of match results from triggered stops
        """
        # Check velocity logic
        if self._enable_velocity_logic and timestamp_ns < self._velocity_pause_until_ns:
            return []  # Still in velocity pause

        # Update last trade price for velocity logic
        if last_trade_price is not None and self._last_trade_price is not None:
            velocity_result = self._check_velocity_logic(last_trade_price, timestamp_ns)
            if velocity_result.triggered:
                self._velocity_pause_until_ns = timestamp_ns + (
                    velocity_result.pause_duration_ms * 1_000_000
                )
                self._velocity_trigger_count += 1
                return []  # Pause trading

        self._last_trade_price = last_trade_price

        # Find triggered stops
        triggered_ids: List[str] = []
        for order_id, stop in self._stop_orders.items():
            if stop.should_trigger(
                last_trade_price=last_trade_price,
                bid=bid,
                ask=ask,
            ):
                triggered_ids.append(order_id)

        if not triggered_ids:
            return []

        # Check stop spike logic
        if self._enable_stop_spike_logic:
            self._stop_executions_in_window.append(timestamp_ns)
            recent_count = len(self._stop_executions_in_window)
            if recent_count >= self._stop_spike_threshold:
                # Delay stop executions
                logger.debug(f"Stop spike logic triggered: {recent_count} stops in window")
                # Still execute but with delay in real system
                # For simulation, we just log

        # Execute triggered stops
        results: List[MatchResult] = []
        for order_id in triggered_ids:
            stop = self._stop_orders.pop(order_id)
            result = self._execute_triggered_stop(stop, timestamp_ns)
            results.append(result)
            self._stop_trigger_count += 1

        return results

    def _execute_triggered_stop(
        self,
        stop: StopOrder,
        timestamp_ns: int,
    ) -> MatchResult:
        """Execute a triggered stop order."""
        if stop.is_stop_limit:
            # Convert to limit order
            limit_order = LimitOrder(
                order_id=stop.order_id,
                price=stop.limit_price,  # type: ignore
                qty=stop.qty,
                remaining_qty=stop.qty,
                timestamp_ns=timestamp_ns,
                side=stop.side,
                order_type=OrderType.LIMIT,
            )
            return self.match(limit_order)
        elif stop.use_protection:
            # Convert to MWP
            order = LimitOrder(
                order_id=stop.order_id,
                price=0.0,  # Will be calculated
                qty=stop.qty,
                remaining_qty=stop.qty,
                timestamp_ns=timestamp_ns,
                side=stop.side,
                order_type=OrderType.MARKET,
            )
            return self.match_with_protection(
                order=order,
                protection_points=stop.protection_points,
            )
        else:
            # Convert to market order
            order = LimitOrder(
                order_id=stop.order_id,
                price=0.0,
                qty=stop.qty,
                remaining_qty=stop.qty,
                timestamp_ns=timestamp_ns,
                side=stop.side,
                order_type=OrderType.MARKET,
            )
            return self.match(order)

    def check_velocity_logic(
        self,
        current_price: float,
        timestamp_ns: int,
    ) -> VelocityLogicResult:
        """
        Public method to check velocity logic.

        Args:
            current_price: Current market price
            timestamp_ns: Current timestamp in nanoseconds

        Returns:
            VelocityLogicResult with triggered status and pause info
        """
        return self._check_velocity_logic_internal(current_price, timestamp_ns)

    def _check_velocity_logic(
        self,
        current_price: float,
        timestamp_ns: int,
    ) -> VelocityLogicResult:
        """Alias for internal velocity logic check."""
        return self._check_velocity_logic_internal(current_price, timestamp_ns)

    def _check_velocity_logic_internal(
        self,
        current_price: float,
        timestamp_ns: int,
    ) -> VelocityLogicResult:
        """
        Check if velocity logic should trigger.

        Velocity logic detects rapid price movements and triggers
        a brief trading pause.
        """
        if self._last_trade_price is None:
            return VelocityLogicResult(False, 0, "")

        price_change = abs(current_price - self._last_trade_price)
        change_in_ticks = price_change / self._tick_size

        if change_in_ticks >= self._velocity_threshold_ticks:
            return VelocityLogicResult(
                triggered=True,
                pause_duration_ms=self._velocity_pause_ms,
                reason=f"Price move of {change_in_ticks:.0f} ticks exceeds threshold",
            )

        return VelocityLogicResult(False, 0, "")

    # =========================================================================
    # Opening/Closing Auctions
    # =========================================================================

    def start_auction(self, auction_type: AuctionState) -> None:
        """
        Start an auction period.

        Args:
            auction_type: Type of auction (COLLECTING for start)
        """
        if self._auction_state != AuctionState.NOT_IN_AUCTION:
            logger.warning(f"Cannot start auction: already in state {self._auction_state}")
            return

        self._auction_state = auction_type
        self._auction_orders.clear()
        logger.debug(f"Auction started: {auction_type}")

    def add_auction_order(self, order: AuctionOrder) -> bool:
        """
        Add order to current auction.

        Args:
            order: Order to add to auction

        Returns:
            True if added, False if not in auction mode
        """
        if self._auction_state != AuctionState.COLLECTING:
            return False

        self._auction_orders.append(order)
        return True

    def calculate_auction_price(self) -> AuctionResult:
        """
        Calculate equilibrium price for auction.

        Uses maximum volume matching algorithm:
        1. Build cumulative buy curve (highest to lowest price)
        2. Build cumulative sell curve (lowest to highest price)
        3. Find price that maximizes matched volume
        4. Handle imbalances at equilibrium

        Returns:
            AuctionResult with equilibrium price and expected volume
        """
        if not self._auction_orders:
            return AuctionResult.empty()

        # Separate buy and sell orders
        buys = [o for o in self._auction_orders if o.side == Side.BUY]
        sells = [o for o in self._auction_orders if o.side == Side.SELL]

        if not buys or not sells:
            return AuctionResult.empty()

        # Get all prices (including market orders)
        prices: Set[float] = set()
        for order in self._auction_orders:
            if order.price is not None:
                prices.add(order.price)

        if not prices:
            return AuctionResult.empty()

        prices_sorted = sorted(prices)

        # Calculate cumulative quantities at each price
        best_price = 0.0
        best_volume = 0.0
        buy_imb = 0.0
        sell_imb = 0.0

        for price in prices_sorted:
            # Buy volume: all buys at price or higher + market buys
            buy_qty = sum(o.remaining_qty for o in buys if o.price is None or o.price >= price)
            # Sell volume: all sells at price or lower + market sells
            sell_qty = sum(o.remaining_qty for o in sells if o.price is None or o.price <= price)

            matched = min(buy_qty, sell_qty)
            if matched > best_volume:
                best_volume = matched
                best_price = price
                buy_imb = buy_qty - matched
                sell_imb = sell_qty - matched

        # Determine imbalance direction
        if buy_imb > sell_imb:
            imb_dir = "BUY"
        elif sell_imb > buy_imb:
            imb_dir = "SELL"
        else:
            imb_dir = "NONE"

        return AuctionResult(
            equilibrium_price=best_price,
            matched_volume=best_volume,
            buy_imbalance=buy_imb,
            sell_imbalance=sell_imb,
            imbalance_direction=imb_dir,
        )

    def execute_auction(self) -> AuctionResult:
        """
        Execute the auction and match orders.

        Changes state from COLLECTING to EXECUTING, calculates
        equilibrium price, and matches all crossing orders.

        Returns:
            AuctionResult with fills
        """
        if self._auction_state != AuctionState.COLLECTING:
            return AuctionResult.empty()

        self._auction_state = AuctionState.UNCROSSING
        result = self.calculate_auction_price()

        if result.equilibrium_price <= 0 or result.matched_volume <= 0:
            self._auction_state = AuctionState.NOT_IN_AUCTION
            return AuctionResult.empty()

        self._auction_state = AuctionState.EXECUTING

        # Match orders at equilibrium price
        fills: List[Fill] = []
        buys = sorted(
            [o for o in self._auction_orders if o.side == Side.BUY],
            key=lambda o: (-(o.price or float("inf")), o.timestamp_ns),
        )
        sells = sorted(
            [o for o in self._auction_orders if o.side == Side.SELL],
            key=lambda o: (o.price or 0, o.timestamp_ns),
        )

        # Pro-rata matching at equilibrium
        remaining_volume = result.matched_volume

        for buy in buys:
            if remaining_volume <= 0:
                break
            if buy.price is not None and buy.price < result.equilibrium_price:
                continue

            fill_qty = min(buy.remaining_qty, remaining_volume)
            if fill_qty > 0:
                fill = Fill(
                    order_id=buy.order_id,
                    price=result.equilibrium_price,
                    qty=fill_qty,
                    aggressor_side=buy.side,
                    timestamp_ns=int(time.time_ns()),
                    resting_order_id=buy.order_id,
                    trade_type="auction",
                    is_maker=True,
                )
                fills.append(fill)
                buy.remaining_qty -= fill_qty
                remaining_volume -= fill_qty

        result.fills = fills
        self._auction_state = AuctionState.COMPLETED
        self._auction_count += 1

        # Reset for next auction
        self._auction_orders.clear()
        self._auction_state = AuctionState.NOT_IN_AUCTION

        return result

    def get_auction_state(self) -> AuctionState:
        """Get current auction state."""
        return self._auction_state

    def get_indicative_auction_price(self) -> Optional[float]:
        """
        Get indicative auction price without executing.

        Returns:
            Indicative equilibrium price or None
        """
        if self._auction_state != AuctionState.COLLECTING:
            return None

        result = self.calculate_auction_price()
        return result.equilibrium_price if result.matched_volume > 0 else None

    # =========================================================================
    # Trade at Settlement (TAS)
    # =========================================================================

    def submit_tas_order(self, order: AuctionOrder) -> bool:
        """
        Submit Trade at Settlement order.

        TAS orders execute at the daily settlement price.

        Args:
            order: TAS order to submit

        Returns:
            True if accepted
        """
        self._tas_orders.append(order)
        return True

    def set_settlement_price(self, price: float) -> None:
        """
        Set settlement price for TAS matching.

        Args:
            price: Daily settlement price
        """
        self._settlement_price = price

    def execute_tas_orders(self) -> List[Fill]:
        """
        Execute all TAS orders at settlement price.

        Returns:
            List of fills from TAS matching
        """
        if self._settlement_price is None:
            logger.warning("Cannot execute TAS: no settlement price set")
            return []

        if not self._tas_orders:
            return []

        # Separate buys and sells
        buys = [o for o in self._tas_orders if o.side == Side.BUY]
        sells = [o for o in self._tas_orders if o.side == Side.SELL]

        # Match at settlement price
        buy_qty = sum(o.remaining_qty for o in buys)
        sell_qty = sum(o.remaining_qty for o in sells)
        matched_qty = min(buy_qty, sell_qty)

        fills: List[Fill] = []

        if matched_qty > 0:
            # Allocate fills pro-rata
            remaining = matched_qty
            for order in buys + sells:
                if remaining <= 0:
                    break
                fill_qty = min(order.remaining_qty, remaining)
                if fill_qty > 0:
                    fill = Fill(
                        order_id=order.order_id,
                        price=self._settlement_price,
                        qty=fill_qty,
                        aggressor_side=order.side,
                        timestamp_ns=int(time.time_ns()),
                        trade_type="TAS",
                        is_maker=True,
                    )
                    fills.append(fill)
                    remaining -= fill_qty
                    order.remaining_qty -= fill_qty

        # Clear matched orders
        self._tas_orders = [o for o in self._tas_orders if o.remaining_qty > 0]

        return fills

    # =========================================================================
    # Minimum Quantity / All-or-None
    # =========================================================================

    def match_with_minimum_qty(
        self,
        order: LimitOrder,
        min_qty: float,
    ) -> MatchResult:
        """
        Match order with minimum quantity requirement.

        Order is only filled if at least min_qty can be executed.

        Args:
            order: Order to match
            min_qty: Minimum fill quantity required

        Returns:
            MatchResult with fills or empty if min_qty not met
        """
        # First, check if min_qty can be filled
        available_qty = self._calculate_available_qty(order)

        if available_qty < min_qty:
            # Cannot meet minimum - reject
            return MatchResult(
                fills=[],
                total_filled_qty=0.0,
                is_complete=False,
            )

        # Execute normally
        return self.match(order)

    def match_all_or_none(self, order: LimitOrder) -> MatchResult:
        """
        Match All-or-None order.

        Order must be completely filled or cancelled.

        Args:
            order: AON order to match

        Returns:
            MatchResult with fills or empty if cannot fill completely
        """
        return self.match_with_minimum_qty(order, order.remaining_qty)

    def _calculate_available_qty(self, order: LimitOrder) -> float:
        """
        Calculate available quantity for order without executing.

        Args:
            order: Order to check

        Returns:
            Available quantity that could be filled
        """
        available = 0.0

        if order.side == Side.BUY:
            # Walk through ask levels
            for level in self._order_book.iter_ask_levels():
                if order.order_type == OrderType.LIMIT and level.price > order.price:
                    break
                available += level.total_qty
        else:
            # Walk through bid levels
            for level in self._order_book.iter_bid_levels():
                if order.order_type == OrderType.LIMIT and level.price < order.price:
                    break
                available += level.total_qty

        return available

    # =========================================================================
    # Statistics and Diagnostics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        base_stats = super().get_statistics() if hasattr(super(), "get_statistics") else {}
        return {
            **base_stats,
            "symbol": self._symbol,
            "protection_points": self._protection_points,
            "mwp_order_count": self._mwp_order_count,
            "stop_trigger_count": self._stop_trigger_count,
            "auction_count": self._auction_count,
            "velocity_trigger_count": self._velocity_trigger_count,
            "pending_stops": len(self._stop_orders),
            "pending_tas_orders": len(self._tas_orders),
            "velocity_logic_enabled": self._enable_velocity_logic,
            "stop_spike_logic_enabled": self._enable_stop_spike_logic,
        }

    def reset_statistics(self) -> None:
        """Reset engine statistics."""
        if hasattr(super(), "reset_statistics"):
            super().reset_statistics()
        self._mwp_order_count = 0
        self._stop_trigger_count = 0
        self._auction_count = 0
        self._velocity_trigger_count = 0

    @property
    def symbol(self) -> str:
        """Get trading symbol."""
        return self._symbol

    @property
    def tick_size(self) -> float:
        """Get tick size."""
        return self._tick_size

    @property
    def protection_points(self) -> int:
        """Get default protection points."""
        return self._protection_points


# =============================================================================
# Factory Functions
# =============================================================================


def create_globex_matching_engine(
    symbol: str,
    tick_size: Optional[float] = None,
    protection_points: Optional[int] = None,
    profile: str = "default",
    **kwargs: Any,
) -> GlobexMatchingEngine:
    """
    Create a Globex matching engine for a symbol.

    Args:
        symbol: Trading symbol
        tick_size: Override tick size (uses product default if None)
        protection_points: Override protection points (uses product default if None)
        profile: Configuration profile
        **kwargs: Additional engine parameters

    Returns:
        Configured GlobexMatchingEngine
    """
    from execution_providers_cme import TICK_SIZES, DEFAULT_TICK_SIZE

    if tick_size is None:
        tick_size = float(TICK_SIZES.get(symbol.upper(), DEFAULT_TICK_SIZE))

    if protection_points is None:
        protection_points = DEFAULT_PROTECTION_POINTS.get(
            symbol.upper(),
            DEFAULT_PROTECTION_POINTS_FALLBACK,
        )

    # Apply profile settings
    if profile == "conservative":
        kwargs.setdefault("velocity_threshold_ticks", 30)
        kwargs.setdefault("velocity_pause_ms", 3000)
    elif profile == "aggressive":
        kwargs.setdefault("velocity_threshold_ticks", 80)
        kwargs.setdefault("velocity_pause_ms", 1000)

    return GlobexMatchingEngine(
        symbol=symbol,
        tick_size=tick_size,
        protection_points=protection_points,
        **kwargs,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Main class
    "GlobexMatchingEngine",
    # Order types
    "GlobexOrderType",
    "StopOrder",
    "StopTriggerType",
    "AuctionOrder",
    # Results
    "AuctionResult",
    "AuctionState",
    "VelocityLogicResult",
    # Factory
    "create_globex_matching_engine",
    # Constants
    "DEFAULT_PROTECTION_POINTS",
]
