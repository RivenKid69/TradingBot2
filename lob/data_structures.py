"""
L3 LOB Data Structures for Equity Market Simulation.

This module provides efficient data structures for maintaining order book state:
- LimitOrder: Individual order with queue position tracking
- PriceLevel: Single price level with FIFO queue
- OrderBook: Full two-sided order book with efficient operations

Performance targets:
- O(1) order insert/cancel via HashMap + LinkedList
- O(log n) price level lookup via SortedDict
- O(k) walk_book where k = number of levels consumed
- <1μs per message update

Note:
    This is a PURE PYTHON implementation for equity L3 simulation.
    Crypto continues to use Cython LOB (fast_lob.pyx).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Callable,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

from sortedcontainers import SortedDict

# ==============================================================================
# Enums
# ==============================================================================


class Side(IntEnum):
    """Order side enumeration."""

    BUY = 1
    SELL = -1

    @classmethod
    def from_string(cls, s: str) -> "Side":
        """Parse side from string."""
        s_upper = s.upper()
        if s_upper in ("BUY", "B", "BID", "1"):
            return cls.BUY
        elif s_upper in ("SELL", "S", "ASK", "OFFER", "-1"):
            return cls.SELL
        raise ValueError(f"Unknown side: {s}")


class OrderType(IntEnum):
    """Order type enumeration."""

    LIMIT = 1
    MARKET = 2
    ICEBERG = 3
    HIDDEN = 4

    @classmethod
    def from_string(cls, s: str) -> "OrderType":
        """Parse order type from string."""
        s_upper = s.upper()
        if s_upper in ("LIMIT", "LMT", "L"):
            return cls.LIMIT
        elif s_upper in ("MARKET", "MKT", "M"):
            return cls.MARKET
        elif s_upper in ("ICEBERG", "ICE", "I"):
            return cls.ICEBERG
        elif s_upper in ("HIDDEN", "HID", "H"):
            return cls.HIDDEN
        raise ValueError(f"Unknown order type: {s}")


# ==============================================================================
# Order Data Structures
# ==============================================================================


@dataclass
class LimitOrder:
    """
    Individual limit order with queue position tracking.

    Attributes:
        order_id: Unique identifier for this order
        price: Limit price (in price ticks or float)
        qty: Original order quantity
        remaining_qty: Remaining unfilled quantity
        hidden_qty: Hidden/reserve quantity (for iceberg orders)
        display_qty: Visible portion of order
        timestamp_ns: Order submission timestamp in nanoseconds
        is_own: Whether this is our order vs market order
        queue_position: Position in FIFO queue at this price level
        side: BUY or SELL
        order_type: LIMIT, ICEBERG, HIDDEN
        participant_id: Optional participant/firm identifier

    Note:
        For regular limit orders: hidden_qty = 0, display_qty = remaining_qty
        For iceberg orders: display_qty = visible portion, hidden_qty = reserve
        For hidden orders: display_qty = 0, hidden_qty = remaining_qty
    """

    order_id: str
    price: float
    qty: float
    remaining_qty: float
    timestamp_ns: int
    side: Side

    # Iceberg/hidden order support
    hidden_qty: float = 0.0
    display_qty: float = 0.0
    _original_display_qty: float = 0.0  # Stored for iceberg replenishment

    # Queue tracking
    is_own: bool = False
    queue_position: int = 0

    # Order classification
    order_type: OrderType = OrderType.LIMIT
    participant_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Initialize display_qty if not set."""
        if self.order_type == OrderType.HIDDEN:
            # Hidden order - nothing visible, all hidden
            self.display_qty = 0.0
            self.hidden_qty = self.remaining_qty
        elif self.display_qty == 0.0 and self.hidden_qty == 0.0:
            # Regular limit order - all visible
            self.display_qty = self.remaining_qty

        # Store original display size for iceberg replenishment
        if self._original_display_qty == 0.0 and self.display_qty > 0:
            self._original_display_qty = self.display_qty

    @property
    def visible_qty(self) -> float:
        """Return visible quantity in the book."""
        return self.display_qty

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.remaining_qty <= 0.0

    @property
    def is_hidden(self) -> bool:
        """Check if order is hidden type."""
        return self.order_type == OrderType.HIDDEN

    @property
    def is_iceberg(self) -> bool:
        """Check if order is iceberg type."""
        return self.order_type == OrderType.ICEBERG

    def fill(self, fill_qty: float) -> float:
        """
        Fill order by given quantity.

        For iceberg orders, replenishes display_qty from hidden_qty.

        Returns:
            Actual filled quantity (may be less if insufficient remaining)
        """
        actual_fill = min(fill_qty, self.remaining_qty)

        if actual_fill <= 0:
            return 0.0

        # Reduce remaining
        self.remaining_qty -= actual_fill

        # Handle display/hidden for iceberg
        if self.order_type == OrderType.ICEBERG:
            # First consume display_qty
            display_consumed = min(actual_fill, self.display_qty)
            self.display_qty -= display_consumed
            remaining_fill = actual_fill - display_consumed

            # Then consume hidden_qty if needed
            if remaining_fill > 0:
                self.hidden_qty -= remaining_fill

            # Replenish display from hidden
            # FIX: Must reduce hidden_qty by replenished amount to maintain invariant:
            # remaining_qty = display_qty + hidden_qty
            if self.display_qty == 0 and self.hidden_qty > 0:
                replenish = min(self.hidden_qty, self._original_display_size())
                self.display_qty = replenish
                self.hidden_qty -= replenish  # FIX: Reduce hidden_qty
        else:
            # Regular limit or hidden - just reduce display
            self.display_qty = max(0.0, self.display_qty - actual_fill)

        return actual_fill

    def _original_display_size(self) -> float:
        """Get original display size for iceberg replenishment."""
        # Use stored value if available, otherwise estimate
        if self._original_display_qty > 0:
            return self._original_display_qty
        # Fallback heuristic: use qty - hidden_qty as original display
        return max(1.0, self.qty - self.hidden_qty)

    def clone(self) -> "LimitOrder":
        """Create a deep copy of this order."""
        cloned = LimitOrder(
            order_id=self.order_id,
            price=self.price,
            qty=self.qty,
            remaining_qty=self.remaining_qty,
            timestamp_ns=self.timestamp_ns,
            side=self.side,
            hidden_qty=self.hidden_qty,
            display_qty=self.display_qty,
            _original_display_qty=self._original_display_qty,
            is_own=self.is_own,
            queue_position=self.queue_position,
            order_type=self.order_type,
            participant_id=self.participant_id,
        )
        return cloned


@dataclass
class PriceLevel:
    """
    Single price level in the order book.

    Maintains a FIFO queue of orders at this price with O(1) operations
    for adding to back and removing from front.

    Attributes:
        price: Price of this level
        orders: FIFO queue of orders (deque for O(1) operations)
        total_visible_qty: Sum of visible quantities
        total_hidden_qty: Sum of hidden quantities (iceberg reserves)
        order_count: Number of orders at this level
    """

    price: float
    orders: Deque[LimitOrder] = field(default_factory=deque)
    total_visible_qty: float = 0.0
    total_hidden_qty: float = 0.0

    # Index for O(1) order lookup by ID
    _order_index: Dict[str, LimitOrder] = field(default_factory=dict)

    @property
    def order_count(self) -> int:
        """Number of orders at this price level."""
        return len(self.orders)

    @property
    def total_qty(self) -> float:
        """Total quantity (visible + hidden) at this level."""
        return self.total_visible_qty + self.total_hidden_qty

    @property
    def is_empty(self) -> bool:
        """Check if this level has no orders."""
        return len(self.orders) == 0

    def add_order(self, order: LimitOrder) -> int:
        """
        Add order to back of FIFO queue.

        Returns:
            Queue position (0-indexed from front)
        """
        queue_pos = len(self.orders)
        order.queue_position = queue_pos

        self.orders.append(order)
        self._order_index[order.order_id] = order

        self.total_visible_qty += order.display_qty
        self.total_hidden_qty += order.hidden_qty

        return queue_pos

    def remove_order(self, order_id: str) -> Optional[LimitOrder]:
        """
        Remove order by ID from this level.

        Updates queue positions for remaining orders.

        Returns:
            Removed order or None if not found
        """
        if order_id not in self._order_index:
            return None

        order = self._order_index.pop(order_id)

        # Remove from deque (O(n) but necessary)
        try:
            self.orders.remove(order)
        except ValueError:
            return None

        # Update totals
        self.total_visible_qty -= order.display_qty
        self.total_hidden_qty -= order.hidden_qty

        # Recompute queue positions
        self._recompute_queue_positions()

        return order

    def get_order(self, order_id: str) -> Optional[LimitOrder]:
        """Get order by ID (O(1) lookup)."""
        return self._order_index.get(order_id)

    def peek_front(self) -> Optional[LimitOrder]:
        """Peek at front order without removing."""
        return self.orders[0] if self.orders else None

    def pop_front(self) -> Optional[LimitOrder]:
        """Remove and return front order (highest priority)."""
        if not self.orders:
            return None

        order = self.orders.popleft()
        self._order_index.pop(order.order_id, None)

        self.total_visible_qty -= order.display_qty
        self.total_hidden_qty -= order.hidden_qty

        # Recompute queue positions
        self._recompute_queue_positions()

        return order

    def fill_qty(self, qty: float) -> Tuple[float, List["Trade"]]:
        """
        Fill quantity from this level (FIFO order).

        Returns:
            Tuple of (filled_qty, list of trades)
        """
        filled = 0.0
        trades: List[Trade] = []

        while qty > 0 and self.orders:
            order = self.orders[0]

            # Calculate fill amount
            fill_amount = min(qty, order.remaining_qty)

            if fill_amount > 0:
                # Record trade
                trade = Trade(
                    price=self.price,
                    qty=fill_amount,
                    maker_order_id=order.order_id,
                    maker_is_own=order.is_own,
                    timestamp_ns=time.time_ns(),
                )
                trades.append(trade)

                # Fill order
                order.fill(fill_amount)
                filled += fill_amount
                qty -= fill_amount

                # Update level totals
                self.total_visible_qty = max(0, self.total_visible_qty - fill_amount)

            # Remove fully filled orders
            if order.is_filled:
                self.orders.popleft()
                self._order_index.pop(order.order_id, None)

        self._recompute_queue_positions()
        return filled, trades

    def _recompute_queue_positions(self) -> None:
        """Recompute queue positions after order removal."""
        for i, order in enumerate(self.orders):
            order.queue_position = i

    def get_queue_position(self, order_id: str) -> Optional[int]:
        """Get queue position for order (O(1))."""
        order = self._order_index.get(order_id)
        return order.queue_position if order else None

    def iter_orders(self) -> Iterator[LimitOrder]:
        """Iterate orders in FIFO order."""
        return iter(self.orders)

    def clone(self) -> "PriceLevel":
        """Create a deep copy of this price level."""
        new_level = PriceLevel(price=self.price)
        for order in self.orders:
            new_level.add_order(order.clone())
        return new_level


# ==============================================================================
# Trade / Fill Structures
# ==============================================================================


@dataclass
class Trade:
    """
    Record of a single trade execution.

    Attributes:
        price: Execution price
        qty: Executed quantity
        maker_order_id: ID of the maker (passive) order
        taker_order_id: ID of the taker (aggressive) order
        maker_is_own: Whether maker was our order
        taker_is_own: Whether taker was our order
        timestamp_ns: Execution timestamp in nanoseconds
        aggressor_side: Side of the aggressor (taker)
    """

    price: float
    qty: float
    maker_order_id: str
    timestamp_ns: int
    maker_is_own: bool = False
    taker_is_own: bool = False
    taker_order_id: Optional[str] = None
    aggressor_side: Optional[Side] = None

    @property
    def notional(self) -> float:
        """Trade notional value."""
        return self.price * self.qty


@dataclass
class Fill:
    """
    Aggregated fill result for an order.

    Attributes:
        order_id: ID of the order that was filled
        total_qty: Total quantity filled
        avg_price: Volume-weighted average fill price
        trades: List of individual trades
        remaining_qty: Quantity left unfilled
        is_complete: Whether order was fully filled
    """

    order_id: str
    total_qty: float
    avg_price: float
    trades: List[Trade]
    remaining_qty: float = 0.0
    is_complete: bool = False

    @classmethod
    def from_trades(
        cls,
        order_id: str,
        original_qty: float,
        trades: List[Trade],
    ) -> "Fill":
        """Create Fill from list of trades."""
        if not trades:
            return cls(
                order_id=order_id,
                total_qty=0.0,
                avg_price=0.0,
                trades=[],
                remaining_qty=original_qty,
                is_complete=False,
            )

        total_qty = sum(t.qty for t in trades)
        total_notional = sum(t.notional for t in trades)
        avg_price = total_notional / total_qty if total_qty > 0 else 0.0
        remaining = original_qty - total_qty

        return cls(
            order_id=order_id,
            total_qty=total_qty,
            avg_price=avg_price,
            trades=trades,
            remaining_qty=remaining,
            is_complete=remaining <= 0,
        )

    @property
    def notional(self) -> float:
        """Total notional value of fill."""
        return self.avg_price * self.total_qty


# ==============================================================================
# Order Book
# ==============================================================================


class OrderBook:
    """
    Full two-sided order book with efficient operations.

    Supports:
    - O(log n) insert/delete at price levels via SortedDict
    - O(1) order lookup via HashMap
    - O(1) best bid/ask access
    - Queue position tracking for limit orders
    - Market-by-Order (MBO) and Market-by-Price (MBP) views
    - Iceberg and hidden order simulation

    Attributes:
        symbol: Trading symbol
        tick_size: Minimum price increment
        lot_size: Minimum quantity increment
    """

    def __init__(
        self,
        symbol: str = "",
        tick_size: float = 0.01,
        lot_size: float = 1.0,
        enforce_tick_size: bool = False,
        enforce_lot_size: bool = False,
        auto_round_price: bool = True,
    ) -> None:
        """
        Initialize order book.

        Args:
            symbol: Trading symbol
            tick_size: Minimum price increment (default 0.01 for US equities)
            lot_size: Minimum quantity increment (default 1.0)
            enforce_tick_size: If True, validate and optionally round prices to tick
                FIX (2025-11-28): Added for US market structure compliance (Rule 612)
            enforce_lot_size: If True, validate order quantity against lot size
                FIX (2025-11-28): Added for US market structure compliance (Rule 600)
            auto_round_price: If True and enforce_tick_size=True, automatically round
                prices to nearest valid tick. If False, reject invalid prices.
        """
        self.symbol = symbol
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.enforce_tick_size = enforce_tick_size
        self.enforce_lot_size = enforce_lot_size
        self.auto_round_price = auto_round_price

        # Price levels - SortedDict for O(log n) price operations
        # Bids: descending order (highest first) - use negative keys
        # Asks: ascending order (lowest first)
        self._bids: SortedDict[float, PriceLevel] = SortedDict()
        self._asks: SortedDict[float, PriceLevel] = SortedDict()

        # Order index for O(1) lookup by order_id
        self._orders: Dict[str, Tuple[LimitOrder, Side]] = {}

        # Statistics
        self._order_count = 0
        self._message_count = 0

        # Sequence number for message ordering
        self._sequence: int = 0

    # ==========================================================================
    # Best Bid/Ask Properties
    # ==========================================================================

    @property
    def best_bid(self) -> Optional[float]:
        """Best (highest) bid price."""
        if not self._bids:
            return None
        # Keys are negative, so first key (most negative) is highest price
        return -self._bids.peekitem(0)[0]

    @property
    def best_ask(self) -> Optional[float]:
        """Best (lowest) ask price."""
        if not self._asks:
            return None
        return self._asks.peekitem(0)[0]

    @property
    def best_bid_qty(self) -> float:
        """Total quantity at best bid."""
        if not self._bids:
            return 0.0
        return self._bids.peekitem(0)[1].total_visible_qty

    @property
    def best_ask_qty(self) -> float:
        """Total quantity at best ask."""
        if not self._asks:
            return 0.0
        return self._asks.peekitem(0)[1].total_visible_qty

    @property
    def mid_price(self) -> Optional[float]:
        """Mid price between best bid and ask."""
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return bid or ask
        return (bid + ask) / 2.0

    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread in price units."""
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return None
        return ask - bid

    @property
    def spread_bps(self) -> Optional[float]:
        """Bid-ask spread in basis points."""
        spread = self.spread
        mid = self.mid_price
        if spread is None or mid is None or mid == 0:
            return None
        return (spread / mid) * 10000.0

    @property
    def is_crossed(self) -> bool:
        """Check if book is crossed (best bid >= best ask)."""
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return False
        return bid >= ask

    @property
    def num_bid_levels(self) -> int:
        """Number of distinct bid price levels."""
        return len(self._bids)

    @property
    def num_ask_levels(self) -> int:
        """Number of distinct ask price levels."""
        return len(self._asks)

    @property
    def num_levels(self) -> int:
        """Total number of price levels (bids + asks)."""
        return len(self._bids) + len(self._asks)

    # ==========================================================================
    # Order Operations
    # ==========================================================================

    def add_limit_order(self, order: LimitOrder) -> int:
        """
        Add limit order to the book.

        Args:
            order: LimitOrder to add

        Returns:
            Queue position at the price level

        Raises:
            ValueError: If order quantity invalid, duplicate order_id,
                or invalid tick size (when enforce_tick_size=True and auto_round=False)

        FIX (2025-11-28): Added tick size and lot size validation for US market structure.
        Reference: docs/PLATFORM_REFERENCE.md → Issue #7 "L3 LOB: US Market Structure"
        """
        # Validate order quantity
        if order.remaining_qty <= 0:
            raise ValueError("Order quantity must be positive")

        # FIX (2025-11-28): Lot size validation
        if self.enforce_lot_size:
            if order.remaining_qty < self.lot_size:
                raise ValueError(
                    f"Order quantity {order.remaining_qty} below minimum lot size {self.lot_size}"
                )

        # Check for duplicate
        if order.order_id in self._orders:
            raise ValueError(f"Duplicate order_id: {order.order_id}")

        # FIX (2025-11-28): Tick size validation and adjustment
        effective_price = order.price
        if self.enforce_tick_size:
            effective_price = self._validate_and_round_tick(order.price)
            if effective_price != order.price:
                # Update order price (modifying in place)
                order.price = effective_price

        # Get or create price level
        if order.side == Side.BUY:
            key = -effective_price  # Negative for descending sort
            levels = self._bids
        else:
            key = effective_price
            levels = self._asks

        if key not in levels:
            levels[key] = PriceLevel(price=effective_price)

        level = levels[key]
        queue_pos = level.add_order(order)

        # Index order
        self._orders[order.order_id] = (order, order.side)
        self._order_count += 1
        self._sequence += 1

        return queue_pos

    def _validate_and_round_tick(self, price: float) -> float:
        """
        Validate and optionally round price to valid tick increment.

        FIX (2025-11-28): Added for US market structure compliance (Rule 612).

        Args:
            price: Price to validate

        Returns:
            Validated/rounded price

        Raises:
            ValueError: If price invalid and auto_round_price=False
        """
        if self.tick_size <= 0:
            return price  # No tick enforcement

        # Check if price is on valid tick
        remainder = abs(price % self.tick_size)
        tolerance = self.tick_size * 1e-9

        if remainder < tolerance or abs(remainder - self.tick_size) < tolerance:
            return price  # Already on valid tick

        # Price not on valid tick
        if self.auto_round_price:
            # Round to nearest tick
            rounded = round(price / self.tick_size) * self.tick_size
            return rounded
        else:
            raise ValueError(
                f"Price {price} not on valid tick (tick_size={self.tick_size}). "
                f"Set auto_round_price=True to automatically round."
            )

    def is_valid_tick(self, price: float) -> bool:
        """
        Check if price is on a valid tick increment.

        FIX (2025-11-28): Added for US market structure validation.

        Args:
            price: Price to check

        Returns:
            True if price is on valid tick
        """
        if self.tick_size <= 0:
            return True

        remainder = abs(price % self.tick_size)
        tolerance = self.tick_size * 1e-9

        return remainder < tolerance or abs(remainder - self.tick_size) < tolerance

    def round_to_tick(self, price: float, direction: str = "nearest") -> float:
        """
        Round price to valid tick increment.

        FIX (2025-11-28): Added for US market structure compliance.

        Args:
            price: Price to round
            direction: "nearest", "up", or "down"

        Returns:
            Rounded price
        """
        if self.tick_size <= 0:
            return price

        if direction == "nearest":
            return round(price / self.tick_size) * self.tick_size
        elif direction == "down":
            import math
            return math.floor(price / self.tick_size) * self.tick_size
        elif direction == "up":
            import math
            return math.ceil(price / self.tick_size) * self.tick_size
        else:
            raise ValueError(f"Invalid direction: {direction}")

    def cancel_order(self, order_id: str) -> Optional[LimitOrder]:
        """
        Cancel (remove) order by ID.

        Args:
            order_id: ID of order to cancel

        Returns:
            Cancelled order or None if not found
        """
        if order_id not in self._orders:
            return None

        order, side = self._orders.pop(order_id)

        # Get price level
        if side == Side.BUY:
            key = -order.price
            levels = self._bids
        else:
            key = order.price
            levels = self._asks

        if key not in levels:
            return None

        level = levels[key]
        removed = level.remove_order(order_id)

        # Remove empty price level
        if level.is_empty:
            del levels[key]

        if removed:
            self._order_count -= 1
            self._sequence += 1

        return removed

    def modify_order(
        self,
        order_id: str,
        new_qty: Optional[float] = None,
        new_price: Optional[float] = None,
    ) -> Optional[LimitOrder]:
        """
        Modify existing order (cancel + replace).

        Price change: loses queue priority (cancel + new order)
        Quantity decrease: maintains queue priority
        Quantity increase: loses queue priority

        Args:
            order_id: ID of order to modify
            new_qty: New quantity (optional)
            new_price: New price (optional)

        Returns:
            Modified order or None if not found
        """
        if order_id not in self._orders:
            return None

        old_order, side = self._orders[order_id]

        # Determine if we lose queue priority
        loses_priority = False
        if new_price is not None and new_price != old_order.price:
            loses_priority = True
        if new_qty is not None and new_qty > old_order.remaining_qty:
            loses_priority = True

        if loses_priority:
            # Cancel + new order
            self.cancel_order(order_id)

            new_order = LimitOrder(
                order_id=order_id,
                price=new_price if new_price else old_order.price,
                qty=new_qty if new_qty else old_order.qty,
                remaining_qty=new_qty if new_qty else old_order.remaining_qty,
                timestamp_ns=time.time_ns(),
                side=side,
                hidden_qty=old_order.hidden_qty,
                display_qty=old_order.display_qty if new_qty is None else new_qty,
                is_own=old_order.is_own,
                order_type=old_order.order_type,
                participant_id=old_order.participant_id,
            )
            self.add_limit_order(new_order)
            return new_order
        else:
            # In-place modification (qty decrease)
            if new_qty is not None:
                delta = old_order.remaining_qty - new_qty
                old_order.remaining_qty = new_qty
                old_order.display_qty = max(0, old_order.display_qty - delta)

                # Update level totals
                if side == Side.BUY:
                    key = -old_order.price
                    levels = self._bids
                else:
                    key = old_order.price
                    levels = self._asks

                if key in levels:
                    levels[key].total_visible_qty -= delta

            self._sequence += 1
            return old_order

    def get_order(self, order_id: str) -> Optional[LimitOrder]:
        """Get order by ID (O(1))."""
        if order_id not in self._orders:
            return None
        return self._orders[order_id][0]

    def get_queue_position(self, order_id: str) -> Optional[int]:
        """Get queue position for order."""
        order = self.get_order(order_id)
        if order is None:
            return None
        return order.queue_position

    def contains_order(self, order_id: str) -> bool:
        """Check if order exists."""
        return order_id in self._orders

    # ==========================================================================
    # Market Order Execution
    # ==========================================================================

    def execute_market_order(
        self,
        side: Side,
        qty: float,
        taker_order_id: Optional[str] = None,
        taker_is_own: bool = False,
    ) -> Fill:
        """
        Execute market order against resting liquidity.

        Args:
            side: BUY or SELL
            qty: Quantity to execute
            taker_order_id: ID of the taker order
            taker_is_own: Whether taker is our order

        Returns:
            Fill with execution details
        """
        if qty <= 0:
            return Fill(
                order_id=taker_order_id or "",
                total_qty=0.0,
                avg_price=0.0,
                trades=[],
                remaining_qty=qty,
                is_complete=False,
            )

        # Select opposite side
        if side == Side.BUY:
            levels = self._asks
            sign = 1  # Ascending prices
        else:
            levels = self._bids
            sign = -1  # We stored negative keys, so reverse iteration

        remaining = qty
        all_trades: List[Trade] = []
        levels_to_remove: List[float] = []

        # Walk through price levels
        for key in list(levels.keys()):
            if remaining <= 0:
                break

            level = levels[key]
            filled_at_level, trades = level.fill_qty(remaining)

            # Tag trades with taker info
            for t in trades:
                t.taker_order_id = taker_order_id
                t.taker_is_own = taker_is_own
                t.aggressor_side = side

            all_trades.extend(trades)
            remaining -= filled_at_level

            # Remove empty levels
            if level.is_empty:
                levels_to_remove.append(key)

            # Update order index for filled orders
            for t in trades:
                if t.maker_order_id in self._orders:
                    order, _ = self._orders[t.maker_order_id]
                    if order.is_filled:
                        del self._orders[t.maker_order_id]
                        self._order_count -= 1

        # Clean up empty levels
        for key in levels_to_remove:
            if key in levels:
                del levels[key]

        self._sequence += 1

        return Fill.from_trades(
            order_id=taker_order_id or "",
            original_qty=qty,
            trades=all_trades,
        )

    # ==========================================================================
    # Book Depth / Walk Operations
    # ==========================================================================

    def get_depth(self, n_levels: int = 10) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Get top N price levels on each side.

        Args:
            n_levels: Number of levels to return

        Returns:
            Tuple of (bids, asks) where each is list of (price, qty) tuples
            Bids are sorted descending (best first)
            Asks are sorted ascending (best first)
        """
        bids: List[Tuple[float, float]] = []
        asks: List[Tuple[float, float]] = []

        # Bids - keys are negative, so iterate in order
        for i, (key, level) in enumerate(self._bids.items()):
            if i >= n_levels:
                break
            bids.append((-key, level.total_visible_qty))

        # Asks - keys are positive, iterate in order
        for i, (key, level) in enumerate(self._asks.items()):
            if i >= n_levels:
                break
            asks.append((key, level.total_visible_qty))

        return bids, asks

    def get_depth_with_orders(
        self, n_levels: int = 10
    ) -> Tuple[List[Tuple[float, float, int]], List[Tuple[float, float, int]]]:
        """
        Get top N price levels with order counts.

        Returns:
            Tuple of (bids, asks) where each is list of (price, qty, order_count)
        """
        bids: List[Tuple[float, float, int]] = []
        asks: List[Tuple[float, float, int]] = []

        for i, (key, level) in enumerate(self._bids.items()):
            if i >= n_levels:
                break
            bids.append((-key, level.total_visible_qty, level.order_count))

        for i, (key, level) in enumerate(self._asks.items()):
            if i >= n_levels:
                break
            asks.append((key, level.total_visible_qty, level.order_count))

        return bids, asks

    def get_bid_levels(self, limit: int = 100) -> List[PriceLevel]:
        """
        Get bid price levels as PriceLevel objects.

        Args:
            limit: Maximum number of levels to return

        Returns:
            List of PriceLevel objects, sorted by price descending (best first)
        """
        levels: List[PriceLevel] = []
        for i, (key, level) in enumerate(self._bids.items()):
            if i >= limit:
                break
            levels.append(level)
        return levels

    def get_ask_levels(self, limit: int = 100) -> List[PriceLevel]:
        """
        Get ask price levels as PriceLevel objects.

        Args:
            limit: Maximum number of levels to return

        Returns:
            List of PriceLevel objects, sorted by price ascending (best first)
        """
        levels: List[PriceLevel] = []
        for i, (key, level) in enumerate(self._asks.items()):
            if i >= limit:
                break
            levels.append(level)
        return levels

    def walk_book(
        self,
        side: Side,
        qty: float,
    ) -> Tuple[float, float, List[Tuple[float, float]]]:
        """
        Simulate walking the book to fill quantity (without executing).

        Args:
            side: BUY (walk asks) or SELL (walk bids)
            qty: Quantity to fill

        Returns:
            Tuple of (avg_price, total_filled, fills_by_level)
            fills_by_level is list of (price, qty) at each level
        """
        if qty <= 0:
            return 0.0, 0.0, []

        # Select opposite side
        levels = self._asks if side == Side.BUY else self._bids

        remaining = qty
        fills: List[Tuple[float, float]] = []
        total_notional = 0.0
        total_filled = 0.0

        for key, level in levels.items():
            if remaining <= 0:
                break

            price = key if side == Side.BUY else -key
            available = level.total_visible_qty + level.total_hidden_qty
            fill_at_level = min(remaining, available)

            if fill_at_level > 0:
                fills.append((price, fill_at_level))
                total_notional += price * fill_at_level
                total_filled += fill_at_level
                remaining -= fill_at_level

        avg_price = total_notional / total_filled if total_filled > 0 else 0.0

        return avg_price, total_filled, fills

    def get_vwap(self, side: Side, qty: float) -> Optional[float]:
        """
        Get VWAP for filling given quantity.

        Args:
            side: BUY or SELL
            qty: Quantity to fill

        Returns:
            Volume-weighted average price or None if insufficient liquidity
        """
        avg_price, total_filled, _ = self.walk_book(side, qty)
        if total_filled < qty:
            return None  # Insufficient liquidity
        return avg_price

    def get_total_liquidity(self, side: Side, price_range_bps: float = 100.0) -> float:
        """
        Get total visible liquidity within price range.

        Args:
            side: Which side to measure (BUY = bids, SELL = asks)
            price_range_bps: Price range in basis points from best

        Returns:
            Total visible quantity
        """
        if side == Side.BUY:
            best = self.best_bid
            levels = self._bids
            if best is None:
                return 0.0
            min_price = best * (1 - price_range_bps / 10000)
        else:
            best = self.best_ask
            levels = self._asks
            if best is None:
                return 0.0
            max_price = best * (1 + price_range_bps / 10000)

        total = 0.0
        for key, level in levels.items():
            price = -key if side == Side.BUY else key
            if side == Side.BUY and price < min_price:
                break
            if side == Side.SELL and price > max_price:
                break
            total += level.total_visible_qty

        return total

    # ==========================================================================
    # MBO / MBP Views
    # ==========================================================================

    def get_mbo_snapshot(self, side: Side, n_orders: int = 100) -> List[LimitOrder]:
        """
        Get Market-by-Order (MBO) snapshot.

        Args:
            side: BUY or SELL
            n_orders: Maximum orders to return

        Returns:
            List of orders sorted by price-time priority
        """
        levels = self._bids if side == Side.BUY else self._asks
        orders: List[LimitOrder] = []

        for key, level in levels.items():
            for order in level.iter_orders():
                orders.append(order)
                if len(orders) >= n_orders:
                    return orders

        return orders

    def get_mbp_snapshot(self, n_levels: int = 10) -> Dict[str, List[Dict]]:
        """
        Get Market-by-Price (MBP) snapshot.

        Returns:
            Dict with 'bids' and 'asks' lists of {price, qty, orders} dicts
        """
        bids, asks = self.get_depth_with_orders(n_levels)

        return {
            "bids": [{"price": p, "qty": q, "orders": n} for p, q, n in bids],
            "asks": [{"price": p, "qty": q, "orders": n} for p, q, n in asks],
        }

    # ==========================================================================
    # Book State Management
    # ==========================================================================

    def clear(self) -> None:
        """Clear all orders from the book."""
        self._bids.clear()
        self._asks.clear()
        self._orders.clear()
        self._order_count = 0
        self._sequence += 1

    def clone(self) -> "OrderBook":
        """Create a deep copy of the order book."""
        new_book = OrderBook(
            symbol=self.symbol,
            tick_size=self.tick_size,
            lot_size=self.lot_size,
        )

        # Clone bids
        for key, level in self._bids.items():
            new_book._bids[key] = level.clone()
            # Update order index
            for order in new_book._bids[key].iter_orders():
                new_book._orders[order.order_id] = (order, Side.BUY)

        # Clone asks
        for key, level in self._asks.items():
            new_book._asks[key] = level.clone()
            for order in new_book._asks[key].iter_orders():
                new_book._orders[order.order_id] = (order, Side.SELL)

        new_book._order_count = self._order_count
        new_book._sequence = self._sequence

        return new_book

    def swap(self, other: "OrderBook") -> None:
        """Swap state with another order book."""
        self._bids, other._bids = other._bids, self._bids
        self._asks, other._asks = other._asks, self._asks
        self._orders, other._orders = other._orders, self._orders
        self._order_count, other._order_count = other._order_count, self._order_count
        self._sequence, other._sequence = other._sequence, self._sequence

    @property
    def order_count(self) -> int:
        """Total number of orders in book."""
        return self._order_count

    @property
    def sequence(self) -> int:
        """Current sequence number."""
        return self._sequence

    def __len__(self) -> int:
        """Number of orders in book."""
        return self._order_count

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OrderBook(symbol={self.symbol!r}, "
            f"bid={self.best_bid}, ask={self.best_ask}, "
            f"orders={self._order_count})"
        )
