"""
Ring Buffer Order Book for Memory-Efficient LOB Simulation.

This module provides a fixed-depth order book that uses ring buffers
to bound memory usage. Instead of storing unlimited price levels,
it keeps only the top N levels and aggregates the rest.

Key Features:
    - O(1) top-of-book access
    - O(log N) price level operations
    - Fixed memory footprint regardless of market depth
    - Aggregated "rest of book" buckets for deep liquidity
    - Compatible with existing OrderBook interface

Memory Savings:
    - Full LOB: O(all_levels) = unbounded
    - Ring Buffer LOB: O(N) where N = max_depth

Reference:
    Phase 0.5 of OPTIONS_INTEGRATION_PLAN.md
    Ring Buffer pattern from operating systems I/O scheduling

Performance Targets:
    - Order add/cancel: < 10 μs
    - Best bid/ask access: < 1 μs
    - Memory per LOB: < 100 KB for 100 levels
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from sortedcontainers import SortedDict

from lob.data_structures import (
    Fill,
    LimitOrder,
    OrderType,
    PriceLevel,
    Side,
    Trade,
)


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass
class AggregatedLevel:
    """
    Aggregated liquidity beyond visible depth.

    Represents the sum of all orders beyond the top N levels.
    Used for market impact estimation without storing every level.

    Attributes:
        total_qty: Total quantity in aggregated levels
        num_levels: Number of price levels aggregated
        num_orders: Total number of orders aggregated
        price_range: (min_price, max_price) of aggregated levels
        weighted_price: Volume-weighted average price
    """

    total_qty: float = 0.0
    num_levels: int = 0
    num_orders: int = 0
    price_range: Tuple[float, float] = (0.0, 0.0)
    weighted_price: float = 0.0

    @property
    def is_empty(self) -> bool:
        """Check if aggregated level is empty."""
        return self.total_qty <= 0.0

    # Aliases for test API compatibility
    @property
    def level_count(self) -> int:
        """Alias for num_levels."""
        return self.num_levels

    @property
    def total_quantity(self) -> float:
        """Alias for total_qty."""
        return self.total_qty

    def add_level(self, price: float, qty: float, order_count: int = 1) -> None:
        """Add a price level to the aggregate."""
        if qty <= 0:
            return

        # Update price range
        if self.num_levels == 0:
            self.price_range = (price, price)
        else:
            min_p, max_p = self.price_range
            self.price_range = (min(min_p, price), max(max_p, price))

        # Update weighted price
        old_total = self.total_qty
        self.total_qty += qty
        if self.total_qty > 0:
            self.weighted_price = (old_total * self.weighted_price + qty * price) / self.total_qty

        self.num_levels += 1
        self.num_orders += order_count

    def remove_level(self, price: float, qty: float, order_count: int = 1) -> None:
        """Remove a price level from the aggregate (approximate)."""
        if qty <= 0:
            return

        self.total_qty = max(0.0, self.total_qty - qty)
        self.num_levels = max(0, self.num_levels - 1)
        self.num_orders = max(0, self.num_orders - order_count)

        # Note: weighted_price becomes less accurate after removal
        # Full recalculation would require storing all prices

    def clear(self) -> None:
        """Clear the aggregated level."""
        self.total_qty = 0.0
        self.num_levels = 0
        self.num_orders = 0
        self.price_range = (0.0, 0.0)
        self.weighted_price = 0.0


@dataclass
class BookLevel:
    """
    Single price level in the ring buffer book.

    Tracks orders at a specific price with FIFO queue semantics.
    """

    price: float
    orders: Deque[LimitOrder] = field(default_factory=deque)

    @property
    def total_qty(self) -> float:
        """Total quantity at this level."""
        return sum(o.remaining_qty for o in self.orders)

    @property
    def visible_qty(self) -> float:
        """Visible quantity (excluding hidden orders)."""
        return sum(o.visible_qty for o in self.orders)

    @property
    def order_count(self) -> int:
        """Number of orders at this level."""
        return len(self.orders)

    @property
    def is_empty(self) -> bool:
        """Check if level has no orders."""
        return len(self.orders) == 0

    def add_order(self, order: LimitOrder) -> None:
        """Add order to the end of the queue (FIFO)."""
        self.orders.append(order)

    def remove_order(self, order_id: str) -> Optional[LimitOrder]:
        """Remove order by ID. Returns removed order or None."""
        for i, order in enumerate(self.orders):
            if order.order_id == order_id:
                del self.orders[i]
                return order
        return None

    def get_front_order(self) -> Optional[LimitOrder]:
        """Get first order in queue (highest priority)."""
        return self.orders[0] if self.orders else None

    def pop_front_order(self) -> Optional[LimitOrder]:
        """Remove and return first order in queue."""
        return self.orders.popleft() if self.orders else None

    def to_price_level(self) -> PriceLevel:
        """Convert to PriceLevel for compatibility."""
        return PriceLevel(
            price=self.price,
            orders=list(self.orders),
        )


@dataclass
class SnapshotLevel:
    """
    Price level in a snapshot with simple price/quantity access.

    Used for test API compatibility.
    """

    price: Union[float, Decimal]
    quantity: Union[float, Decimal]
    order_count: int = 1

    @classmethod
    def from_book_level(cls, level: BookLevel) -> "SnapshotLevel":
        """Create from BookLevel."""
        return cls(
            price=Decimal(str(level.price)),
            quantity=Decimal(str(level.total_qty)),
            order_count=level.order_count,
        )


@dataclass
class BookSnapshot:
    """
    Snapshot of ring buffer book state.

    Used for serialization and analysis.
    """

    timestamp_ns: int
    symbol: str
    bid_levels: List[BookLevel]
    ask_levels: List[BookLevel]
    bid_aggregate: AggregatedLevel
    ask_aggregate: AggregatedLevel
    total_bid_qty: float
    total_ask_qty: float

    # Cached snapshot levels for API compatibility
    _bids_cache: Optional[List[SnapshotLevel]] = field(default=None, repr=False)
    _asks_cache: Optional[List[SnapshotLevel]] = field(default=None, repr=False)

    @property
    def bids(self) -> List[SnapshotLevel]:
        """Get bid levels as SnapshotLevel list (API compatibility)."""
        if self._bids_cache is None:
            self._bids_cache = [SnapshotLevel.from_book_level(l) for l in self.bid_levels]
        return self._bids_cache

    @property
    def asks(self) -> List[SnapshotLevel]:
        """Get ask levels as SnapshotLevel list (API compatibility)."""
        if self._asks_cache is None:
            self._asks_cache = [SnapshotLevel.from_book_level(l) for l in self.ask_levels]
        return self._asks_cache

    @property
    def aggregated_bid(self) -> AggregatedLevel:
        """Alias for bid_aggregate."""
        return self.bid_aggregate

    @property
    def aggregated_ask(self) -> AggregatedLevel:
        """Alias for ask_aggregate."""
        return self.ask_aggregate


@dataclass
class BookStatistics:
    """Statistics for a ring buffer order book."""

    num_bid_levels: int = 0
    num_ask_levels: int = 0
    total_bid_qty: float = 0.0
    total_ask_qty: float = 0.0
    visible_bid_qty: float = 0.0
    visible_ask_qty: float = 0.0
    bid_aggregate_qty: float = 0.0
    ask_aggregate_qty: float = 0.0
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    spread: Optional[float] = None
    mid_price: Optional[float] = None
    total_orders: int = 0
    imbalance: float = 0.0  # (bid - ask) / (bid + ask)

    # Aliases for test API compatibility
    @property
    def bid_levels(self) -> int:
        """Alias for num_bid_levels."""
        return self.num_bid_levels

    @property
    def ask_levels(self) -> int:
        """Alias for num_ask_levels."""
        return self.num_ask_levels


# ==============================================================================
# Ring Buffer Order Book
# ==============================================================================


class RingBufferOrderBook:
    """
    Memory-efficient order book with fixed depth.

    Instead of storing unlimited price levels, this order book
    keeps only the top N levels (closest to mid price) and
    aggregates everything beyond into "rest of book" buckets.

    Memory Model:
        - Visible levels: SortedDict[price -> BookLevel]
        - Capped at max_depth levels per side
        - Beyond max_depth: Aggregated into AggregatedLevel
        - Orders by ID: Dict[order_id -> (price, side)]

    Usage:
        book = RingBufferOrderBook(symbol="AAPL_241220_C_200", max_depth=100)

        # Add orders
        book.add_order(order)

        # Get best bid/ask
        bid = book.get_best_bid()
        ask = book.get_best_ask()

        # Walk the book for market order simulation
        fills = book.walk_book(Side.BUY, qty=100.0)

    Thread Safety:
        This class is NOT thread-safe. External synchronization required.
    """

    def __init__(
        self,
        symbol: str,
        max_depth: int = 100,
        tick_size: Optional[float] = None,
    ):
        """
        Initialize ring buffer order book.

        Args:
            symbol: Symbol/series identifier
            max_depth: Maximum price levels to store per side
            tick_size: Price tick size for rounding (optional)

        Raises:
            ValueError: If max_depth < 1
        """
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")

        self._symbol = symbol
        self._max_depth = max_depth
        self._tick_size = tick_size

        # Price levels stored in sorted containers
        # Bids: highest price first (descending)
        # Asks: lowest price first (ascending)
        self._bid_levels: SortedDict[float, BookLevel] = SortedDict(lambda x: -x)
        self._ask_levels: SortedDict[float, BookLevel] = SortedDict()

        # Aggregated beyond-depth liquidity
        self._bid_aggregate = AggregatedLevel()
        self._ask_aggregate = AggregatedLevel()

        # Order lookup by ID
        self._orders_by_id: Dict[str, Tuple[float, Side]] = {}

        # Counters
        self._total_orders = 0
        self._last_update_ns = 0

    # ==========================================================================
    # Properties
    # ==========================================================================

    @property
    def symbol(self) -> str:
        """Get symbol."""
        return self._symbol

    @property
    def max_depth(self) -> int:
        """Get maximum depth per side."""
        return self._max_depth

    @property
    def num_bid_levels(self) -> int:
        """Number of bid levels in visible book."""
        return len(self._bid_levels)

    @property
    def num_ask_levels(self) -> int:
        """Number of ask levels in visible book."""
        return len(self._ask_levels)

    @property
    def num_orders(self) -> int:
        """Total number of orders in book."""
        return len(self._orders_by_id)

    @property
    def is_empty(self) -> bool:
        """Check if book has no orders."""
        return len(self._orders_by_id) == 0

    # ==========================================================================
    # Best Bid/Ask
    # ==========================================================================

    def get_best_bid(self) -> Optional[float]:
        """Get best (highest) bid price."""
        if not self._bid_levels:
            return None
        return self._bid_levels.peekitem(0)[0]

    def get_best_ask(self) -> Optional[float]:
        """Get best (lowest) ask price."""
        if not self._ask_levels:
            return None
        return self._ask_levels.peekitem(0)[0]

    def get_best_bid_qty(self) -> float:
        """Get quantity at best bid."""
        if not self._bid_levels:
            return 0.0
        level = self._bid_levels.peekitem(0)[1]
        return level.total_qty

    def get_best_ask_qty(self) -> float:
        """Get quantity at best ask."""
        if not self._ask_levels:
            return 0.0
        level = self._ask_levels.peekitem(0)[1]
        return level.total_qty

    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is None or ask is None:
            return None
        return ask - bid

    def get_mid_price(self) -> Optional[float]:
        """Get mid price."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def get_imbalance(self) -> float:
        """
        Get order book imbalance.

        Imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)

        Positive values indicate more bids (buying pressure).
        Negative values indicate more asks (selling pressure).

        Returns:
            Imbalance in range [-1, 1], or 0 if no liquidity
        """
        bid_qty = self.get_total_bid_liquidity()
        ask_qty = self.get_total_ask_liquidity()
        total = bid_qty + ask_qty
        if total == 0:
            return 0.0
        return (bid_qty - ask_qty) / total

    def get_weighted_mid_price(self) -> Optional[Union[float, Decimal]]:
        """
        Get volume-weighted mid price.

        Weighted by quantities at best bid/ask.
        If more quantity at ask, weighted mid is closer to bid.
        If more quantity at bid, weighted mid is closer to ask.

        Returns:
            Weighted mid price, or None if no bid/ask
        """
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is None or ask is None:
            return None

        bid_qty = self.get_best_bid_qty()
        ask_qty = self.get_best_ask_qty()
        total_qty = bid_qty + ask_qty

        if total_qty == 0:
            return Decimal(str((bid + ask) / 2.0))

        # Weight by opposite side - more ask qty means closer to bid
        weighted = (bid * ask_qty + ask * bid_qty) / total_qty
        return Decimal(str(weighted))

    def get_depth_at_price(
        self,
        side: Side,
        price: Union[float, Decimal],
    ) -> Decimal:
        """
        Get total quantity at a specific price level.

        Args:
            side: BUY for bids, SELL for asks
            price: Price level to query

        Returns:
            Total quantity at that price, or 0 if no orders
        """
        price_f = float(price) if isinstance(price, Decimal) else price

        if side == Side.BUY:
            levels = self._bid_levels
        else:
            levels = self._ask_levels

        if price_f in levels:
            return Decimal(str(levels[price_f].total_qty))
        return Decimal("0")

    def get_vwap(
        self,
        side: Side,
        qty: Union[float, Decimal],
    ) -> Optional[Decimal]:
        """
        Calculate volume-weighted average price for a given quantity.

        Simulates market order execution to determine average fill price.

        Args:
            side: BUY (walks asks) or SELL (walks bids)
            qty: Quantity to fill

        Returns:
            Volume-weighted average price, or None if insufficient liquidity
        """
        qty_f = float(qty) if isinstance(qty, Decimal) else qty
        fills = self.walk_book(side, qty_f, include_aggregate=True)

        if not fills:
            return None

        total_value = sum(price * fill_qty for price, fill_qty in fills)
        total_qty = sum(fill_qty for _, fill_qty in fills)

        if total_qty == 0:
            return None

        return Decimal(str(total_value / total_qty))

    # ==========================================================================
    # Order Management
    # ==========================================================================

    def add_order(
        self,
        order: Optional[LimitOrder] = None,
        *,
        side: Optional[Side] = None,
        price: Optional[Union[float, Decimal]] = None,
        qty: Optional[Union[float, Decimal]] = None,
        order_id: Optional[str] = None,
    ) -> bool:
        """
        Add order to the book.

        Can be called with a LimitOrder object or with individual parameters:
            book.add_order(limit_order)
            book.add_order(side=Side.BUY, price=100.0, qty=50, order_id="o1")

        If adding this order would exceed max_depth, the order
        at the worst price is pushed to the aggregate bucket.

        Args:
            order: LimitOrder to add (or None if using kwargs)
            side: Order side (BUY or SELL) - required if order is None
            price: Order price - required if order is None
            qty: Order quantity - required if order is None
            order_id: Order ID - required if order is None

        Returns:
            True if added to visible book, False if aggregated

        Raises:
            ValueError: If order already exists or required params missing
        """
        # Handle kwargs-style call
        if order is None:
            if side is None or price is None or qty is None or order_id is None:
                raise ValueError("Either provide 'order' or all of: side, price, qty, order_id")

            # Convert Decimal to float for internal use
            price_float = float(price) if isinstance(price, Decimal) else price
            qty_float = float(qty) if isinstance(qty, Decimal) else qty

            order = LimitOrder(
                order_id=order_id,
                price=price_float,
                qty=qty_float,
                remaining_qty=qty_float,
                timestamp_ns=time.time_ns(),
                side=side,
                order_type=OrderType.LIMIT,
            )

        if order.order_id in self._orders_by_id:
            raise ValueError(f"Order already exists: {order.order_id}")

        order_price = order.price
        order_side = order.side

        # Select appropriate side
        if order_side == Side.BUY:
            levels = self._bid_levels
            aggregate = self._bid_aggregate
        else:
            levels = self._ask_levels
            aggregate = self._ask_aggregate

        # Check if we need to aggregate instead of add
        if len(levels) >= self._max_depth:
            if not self._should_add_to_visible(order_price, order_side, levels):
                # Aggregate this order
                aggregate.add_level(order_price, order.remaining_qty, 1)
                self._orders_by_id[order.order_id] = (order_price, order_side)
                self._total_orders += 1
                return False

            # Need to evict worst level to aggregate
            self._evict_worst_level(order_side)

        # Add to visible book
        if order_price not in levels:
            levels[order_price] = BookLevel(price=order_price)

        levels[order_price].add_order(order)
        self._orders_by_id[order.order_id] = (order_price, order_side)
        self._total_orders += 1
        self._last_update_ns = order.timestamp_ns

        return True

    def remove_order(self, order_id: str) -> Optional[LimitOrder]:
        """
        Remove order from the book.

        Args:
            order_id: ID of order to remove

        Returns:
            Removed order, or None if not found
        """
        if order_id not in self._orders_by_id:
            return None

        price, side = self._orders_by_id[order_id]

        if side == Side.BUY:
            levels = self._bid_levels
        else:
            levels = self._ask_levels

        if price not in levels:
            # Order might be in aggregate (not tracked individually)
            del self._orders_by_id[order_id]
            return None

        level = levels[price]
        order = level.remove_order(order_id)

        if order:
            # Clean up empty level
            if level.is_empty:
                del levels[price]

            del self._orders_by_id[order_id]
            self._total_orders -= 1

        return order

    def modify_order(
        self,
        order_id: str,
        new_qty: Optional[float] = None,
        new_price: Optional[float] = None,
    ) -> Optional[LimitOrder]:
        """
        Modify an existing order.

        Note: Price change causes order to lose queue priority.

        Args:
            order_id: ID of order to modify
            new_qty: New quantity (optional)
            new_price: New price (optional, causes re-queue)

        Returns:
            Modified order, or None if not found
        """
        if order_id not in self._orders_by_id:
            return None

        price, side = self._orders_by_id[order_id]

        if side == Side.BUY:
            levels = self._bid_levels
        else:
            levels = self._ask_levels

        if price not in levels:
            return None

        level = levels[price]

        # Find order in level
        for order in level.orders:
            if order.order_id == order_id:
                if new_price is not None and new_price != price:
                    # Price change - remove and re-add
                    self.remove_order(order_id)
                    order.price = new_price
                    if new_qty is not None:
                        order.remaining_qty = new_qty
                        order.qty = new_qty
                    order.timestamp_ns = time.time_ns()
                    self.add_order(order)
                    return order
                elif new_qty is not None:
                    # Qty change only - update in place
                    order.remaining_qty = new_qty
                    order.qty = new_qty
                    return order
                else:
                    return order

        return None

    def has_order(self, order_id: str) -> bool:
        """Check if order exists in book."""
        return order_id in self._orders_by_id

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel order by ID.

        Alias for remove_order that returns bool instead of order.

        Args:
            order_id: ID of order to cancel

        Returns:
            True if order was found and cancelled, False otherwise
        """
        return self.remove_order(order_id) is not None

    def get_order(self, order_id: str) -> Optional[LimitOrder]:
        """Get order by ID without removing."""
        if order_id not in self._orders_by_id:
            return None

        price, side = self._orders_by_id[order_id]

        if side == Side.BUY:
            levels = self._bid_levels
        else:
            levels = self._ask_levels

        if price not in levels:
            return None

        for order in levels[price].orders:
            if order.order_id == order_id:
                return order

        return None

    # ==========================================================================
    # Book Walking
    # ==========================================================================

    def walk_book(
        self,
        side: Side,
        qty: float,
        include_aggregate: bool = True,
    ) -> List[Tuple[float, float]]:
        """
        Walk the book to simulate market order execution.

        Returns list of (price, qty) tuples representing
        each level consumed.

        Args:
            side: Direction of walk (BUY walks asks, SELL walks bids)
            qty: Total quantity to fill
            include_aggregate: Include aggregate bucket if needed

        Returns:
            List of (price, qty) tuples
        """
        fills: List[Tuple[float, float]] = []
        remaining = qty

        # Select opposite side (buying hits asks, selling hits bids)
        if side == Side.BUY:
            levels = self._ask_levels
            aggregate = self._ask_aggregate
        else:
            levels = self._bid_levels
            aggregate = self._bid_aggregate

        # Walk visible levels
        for price, level in levels.items():
            if remaining <= 0:
                break

            level_qty = level.total_qty
            fill_qty = min(remaining, level_qty)

            if fill_qty > 0:
                fills.append((price, fill_qty))
                remaining -= fill_qty

        # Include aggregate if needed and available
        if include_aggregate and remaining > 0 and not aggregate.is_empty:
            fill_qty = min(remaining, aggregate.total_qty)
            if fill_qty > 0:
                fills.append((aggregate.weighted_price, fill_qty))
                remaining -= fill_qty

        return fills

    def get_depth(
        self,
        side: Side,
        num_levels: int = 10,
    ) -> List[Tuple[float, float]]:
        """
        Get book depth (price, qty) pairs.

        Args:
            side: BUY for bids, SELL for asks
            num_levels: Number of levels to return

        Returns:
            List of (price, qty) tuples
        """
        if side == Side.BUY:
            levels = self._bid_levels
        else:
            levels = self._ask_levels

        depth = []
        for i, (price, level) in enumerate(levels.items()):
            if i >= num_levels:
                break
            depth.append((price, level.total_qty))

        return depth

    def get_bid_levels(self, limit: Optional[int] = None) -> List[PriceLevel]:
        """Get bid levels as PriceLevel objects."""
        result = []
        for i, (price, level) in enumerate(self._bid_levels.items()):
            if limit and i >= limit:
                break
            result.append(level.to_price_level())
        return result

    def get_ask_levels(self, limit: Optional[int] = None) -> List[PriceLevel]:
        """Get ask levels as PriceLevel objects."""
        result = []
        for i, (price, level) in enumerate(self._ask_levels.items()):
            if limit and i >= limit:
                break
            result.append(level.to_price_level())
        return result

    # ==========================================================================
    # Aggregate Access
    # ==========================================================================

    def get_bid_aggregate(self) -> AggregatedLevel:
        """Get aggregated bid liquidity beyond visible depth."""
        return self._bid_aggregate

    def get_ask_aggregate(self) -> AggregatedLevel:
        """Get aggregated ask liquidity beyond visible depth."""
        return self._ask_aggregate

    def get_total_bid_liquidity(self) -> float:
        """Get total bid liquidity (visible + aggregate)."""
        visible = sum(float(level.total_qty) for level in self._bid_levels.values())
        return visible + self._bid_aggregate.total_qty

    def get_total_ask_liquidity(self) -> float:
        """Get total ask liquidity (visible + aggregate)."""
        visible = sum(float(level.total_qty) for level in self._ask_levels.values())
        return visible + self._ask_aggregate.total_qty

    # ==========================================================================
    # Statistics
    # ==========================================================================

    def get_statistics(self) -> BookStatistics:
        """Get comprehensive book statistics."""
        bid_qty = sum(float(level.total_qty) for level in self._bid_levels.values())
        ask_qty = sum(float(level.total_qty) for level in self._ask_levels.values())
        visible_bid = sum(float(level.visible_qty) for level in self._bid_levels.values())
        visible_ask = sum(float(level.visible_qty) for level in self._ask_levels.values())

        total_bid = bid_qty + self._bid_aggregate.total_qty
        total_ask = ask_qty + self._ask_aggregate.total_qty

        # Calculate imbalance
        total = total_bid + total_ask
        imbalance = (total_bid - total_ask) / total if total > 0 else 0.0

        return BookStatistics(
            num_bid_levels=len(self._bid_levels),
            num_ask_levels=len(self._ask_levels),
            total_bid_qty=total_bid,
            total_ask_qty=total_ask,
            visible_bid_qty=visible_bid,
            visible_ask_qty=visible_ask,
            bid_aggregate_qty=self._bid_aggregate.total_qty,
            ask_aggregate_qty=self._ask_aggregate.total_qty,
            best_bid=self.get_best_bid(),
            best_ask=self.get_best_ask(),
            spread=self.get_spread(),
            mid_price=self.get_mid_price(),
            total_orders=len(self._orders_by_id),
            imbalance=imbalance,
        )

    def get_snapshot(self) -> BookSnapshot:
        """Get full book snapshot for serialization."""
        return BookSnapshot(
            timestamp_ns=self._last_update_ns,
            symbol=self._symbol,
            bid_levels=list(self._bid_levels.values()),
            ask_levels=list(self._ask_levels.values()),
            bid_aggregate=self._bid_aggregate,
            ask_aggregate=self._ask_aggregate,
            total_bid_qty=self.get_total_bid_liquidity(),
            total_ask_qty=self.get_total_ask_liquidity(),
        )

    def get_memory_estimate_bytes(self) -> int:
        """Estimate memory usage in bytes."""
        # Rough estimate: each order ~200 bytes, each level ~100 bytes
        num_orders = len(self._orders_by_id)
        num_levels = len(self._bid_levels) + len(self._ask_levels)
        return num_orders * 200 + num_levels * 100 + 500  # +500 overhead

    # ==========================================================================
    # Clear and Reset
    # ==========================================================================

    def clear(self) -> None:
        """Clear all orders from the book."""
        self._bid_levels.clear()
        self._ask_levels.clear()
        self._bid_aggregate.clear()
        self._ask_aggregate.clear()
        self._orders_by_id.clear()
        self._total_orders = 0

    # ==========================================================================
    # Internal Methods
    # ==========================================================================

    def _should_add_to_visible(
        self,
        price: float,
        side: Side,
        levels: SortedDict,
    ) -> bool:
        """Check if price should be added to visible book or aggregated."""
        if len(levels) < self._max_depth:
            return True

        # Get worst visible price
        worst_price, _ = levels.peekitem(-1)  # Last item is worst

        if side == Side.BUY:
            # For bids, higher price is better
            return price > worst_price
        else:
            # For asks, lower price is better
            return price < worst_price

    def _evict_worst_level(self, side: Side) -> None:
        """Evict worst visible level to aggregate."""
        if side == Side.BUY:
            levels = self._bid_levels
            aggregate = self._bid_aggregate
        else:
            levels = self._ask_levels
            aggregate = self._ask_aggregate

        if not levels:
            return

        # Get and remove worst level
        worst_price, worst_level = levels.popitem(-1)

        # Add to aggregate
        aggregate.add_level(
            price=worst_price,
            qty=worst_level.total_qty,
            order_count=worst_level.order_count,
        )

    # ==========================================================================
    # Compatibility with OrderBook
    # ==========================================================================

    def to_order_book(self) -> "RingBufferOrderBook":
        """Return self for compatibility (already is an order book)."""
        return self


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_ring_buffer_book(
    symbol: Optional[str] = None,
    max_depth: int = 100,
    tick_size: Optional[Union[float, Decimal]] = None,
) -> RingBufferOrderBook:
    """
    Create a ring buffer order book.

    Args:
        symbol: Symbol/series identifier (optional, defaults to "BOOK")
        max_depth: Maximum price levels per side
        tick_size: Price tick size (optional, can be float or Decimal)

    Returns:
        Configured RingBufferOrderBook
    """
    # Convert tick_size from Decimal if needed
    tick_size_f: Optional[float] = None
    if tick_size is not None:
        tick_size_f = float(tick_size) if isinstance(tick_size, Decimal) else tick_size

    return RingBufferOrderBook(
        symbol=symbol or "BOOK",
        max_depth=max_depth,
        tick_size=tick_size_f,
    )


def create_options_book(
    symbol: Optional[str] = None,
    underlying: Optional[str] = None,
    expiry: Optional[str] = None,
    option_type: Optional[str] = None,
    strike: Optional[float] = None,
    max_depth: int = 100,
) -> RingBufferOrderBook:
    """
    Create a ring buffer order book for an option series.

    Can be called with either:
        create_options_book(symbol="AAPL_241220_C_200", max_depth=20)
        create_options_book(underlying="AAPL", expiry="241220", option_type="C", strike=200.0)

    Args:
        symbol: Full option symbol (e.g., "AAPL_241220_C_200")
        underlying: Underlying symbol (e.g., "AAPL")
        expiry: Expiration date (YYMMDD)
        option_type: "C" for call, "P" for put
        strike: Strike price
        max_depth: Maximum price levels

    Returns:
        Configured RingBufferOrderBook for the option series
    """
    # If symbol provided directly, use it
    if symbol is not None:
        return RingBufferOrderBook(
            symbol=symbol,
            max_depth=max_depth,
        )

    # Otherwise, construct from components
    if underlying is None or expiry is None or option_type is None or strike is None:
        raise ValueError(
            "Either 'symbol' or all of (underlying, expiry, option_type, strike) must be provided"
        )

    constructed_symbol = f"{underlying}_{expiry}_{option_type}_{strike:g}"
    return RingBufferOrderBook(
        symbol=constructed_symbol,
        max_depth=max_depth,
    )


# ==============================================================================
# Exports
# ==============================================================================

__all__ = [
    # Data classes
    "AggregatedLevel",
    "BookLevel",
    "BookSnapshot",
    "BookStatistics",
    "SnapshotLevel",
    # Main class
    "RingBufferOrderBook",
    # Factory functions
    "create_ring_buffer_book",
    "create_options_book",
]
