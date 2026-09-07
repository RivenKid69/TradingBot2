"""
Order Lifecycle Manager for L3 LOB Simulation.

Provides unified interface for order management combining:
- OrderBook state management
- MatchingEngine for execution
- QueuePositionTracker for position tracking

Features:
- Order lifecycle: NEW -> PARTIALLY_FILLED -> FILLED/CANCELLED
- Automatic queue position tracking
- Event callbacks for fills, cancels, position updates
- Order modification with priority handling
- Order statistics and reporting

This module serves as the main entry point for order management in
equity L3 simulation, integrating all LOB components.

Performance Target: <10us per order operation
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
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
    STPAction,
)
from lob.queue_tracker import (
    FillProbability,
    QueuePositionTracker,
    QueueState,
)


# ==============================================================================
# Enums and Constants
# ==============================================================================


class OrderLifecycleState(IntEnum):
    """Order lifecycle states."""

    PENDING = 0  # Order created but not submitted
    NEW = 1  # Order accepted, in book
    PARTIALLY_FILLED = 2  # Order partially executed
    FILLED = 3  # Order fully executed
    CANCELLED = 4  # Order cancelled
    REJECTED = 5  # Order rejected
    EXPIRED = 6  # Order expired (time-in-force)


class TimeInForce(IntEnum):
    """Time-in-force options."""

    DAY = 1  # Good for day
    GTC = 2  # Good till cancelled
    IOC = 3  # Immediate or cancel
    FOK = 4  # Fill or kill
    GTD = 5  # Good till date


class OrderEventType(IntEnum):
    """Order event types."""

    SUBMITTED = 1
    ACCEPTED = 2
    REJECTED = 3
    PARTIALLY_FILLED = 4
    FILLED = 5
    CANCELLED = 6
    MODIFIED = 7
    EXPIRED = 8


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass
class ManagedOrder:
    """
    Order with full lifecycle tracking.

    Attributes:
        order: Underlying LimitOrder
        state: Current lifecycle state
        original_qty: Original order quantity
        filled_qty: Total quantity filled
        cancelled_qty: Quantity cancelled
        avg_fill_price: Volume-weighted average fill price
        fills: List of fills
        time_in_force: Time-in-force setting
        created_ns: Creation timestamp
        last_update_ns: Last update timestamp
        client_order_id: Client-provided order ID
        internal_id: Internal tracking ID
        tags: Custom metadata tags
    """

    order: LimitOrder
    state: OrderLifecycleState = OrderLifecycleState.PENDING
    original_qty: float = 0.0
    filled_qty: float = 0.0
    cancelled_qty: float = 0.0
    avg_fill_price: float = 0.0
    fills: List[Fill] = field(default_factory=list)
    time_in_force: TimeInForce = TimeInForce.DAY
    created_ns: int = 0
    last_update_ns: int = 0
    client_order_id: Optional[str] = None
    internal_id: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.original_qty == 0.0:
            self.original_qty = self.order.qty
        if self.created_ns == 0:
            self.created_ns = time.time_ns()
        if not self.internal_id:
            self.internal_id = str(uuid.uuid4())

    @property
    def remaining_qty(self) -> float:
        """Remaining quantity to fill."""
        return max(0.0, self.original_qty - self.filled_qty - self.cancelled_qty)

    @property
    def is_active(self) -> bool:
        """Check if order is still active (can be filled)."""
        return self.state in (
            OrderLifecycleState.NEW,
            OrderLifecycleState.PARTIALLY_FILLED,
        )

    @property
    def is_done(self) -> bool:
        """Check if order is terminal state."""
        return self.state in (
            OrderLifecycleState.FILLED,
            OrderLifecycleState.CANCELLED,
            OrderLifecycleState.REJECTED,
            OrderLifecycleState.EXPIRED,
        )

    @property
    def fill_rate(self) -> float:
        """Fill rate (filled / original)."""
        if self.original_qty <= 0:
            return 0.0
        return self.filled_qty / self.original_qty

    def add_fill(self, fill: Fill) -> None:
        """Add fill to order."""
        self.fills.append(fill)

        # Update filled qty and avg price
        old_notional = self.avg_fill_price * self.filled_qty
        new_notional = fill.avg_price * fill.total_qty
        total_notional = old_notional + new_notional
        total_qty = self.filled_qty + fill.total_qty

        self.filled_qty = total_qty
        self.avg_fill_price = total_notional / total_qty if total_qty > 0 else 0.0
        self.last_update_ns = time.time_ns()

        # Update state
        if self.remaining_qty <= 0:
            self.state = OrderLifecycleState.FILLED
        else:
            self.state = OrderLifecycleState.PARTIALLY_FILLED


@dataclass
class OrderEvent:
    """
    Order lifecycle event for reporting.

    Attributes:
        event_type: Type of event
        order_id: Order ID
        timestamp_ns: Event timestamp
        fill: Fill details (if applicable)
        reason: Reason string (for reject/cancel)
        old_state: Previous state
        new_state: New state
    """

    event_type: OrderEventType
    order_id: str
    timestamp_ns: int
    fill: Optional[Fill] = None
    reason: Optional[str] = None
    old_state: Optional[OrderLifecycleState] = None
    new_state: Optional[OrderLifecycleState] = None


@dataclass
class OrderManagerStats:
    """Statistics from order manager."""

    total_orders: int = 0
    active_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0
    total_fills: int = 0
    total_volume_filled: float = 0.0
    total_volume_cancelled: float = 0.0
    avg_fill_rate: float = 0.0


# ==============================================================================
# Order Manager
# ==============================================================================


class OrderManager:
    """
    Unified Order Lifecycle Manager.

    Integrates OrderBook, MatchingEngine, and QueuePositionTracker
    to provide complete order management for L3 simulation.

    Features:
    - Order submission with automatic matching
    - Queue position tracking
    - Order modification with priority handling
    - Fill and cancel callbacks
    - Order statistics and reporting

    Usage:
        manager = OrderManager(symbol="AAPL")

        # Submit order
        result = manager.submit_order(
            side=Side.BUY,
            price=100.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
        )

        # Check queue position
        state = manager.get_queue_state(result.order_id)

        # Cancel order
        manager.cancel_order(result.order_id)
    """

    def __init__(
        self,
        symbol: str = "",
        tick_size: float = 0.01,
        lot_size: float = 1.0,
        enable_stp: bool = True,
        stp_action: STPAction = STPAction.CANCEL_NEWEST,
        participant_id: Optional[str] = None,
        on_fill: Optional[Callable[[ManagedOrder, Fill], None]] = None,
        on_cancel: Optional[Callable[[ManagedOrder], None]] = None,
        on_event: Optional[Callable[[OrderEvent], None]] = None,
    ) -> None:
        """
        Initialize order manager.

        Args:
            symbol: Trading symbol
            tick_size: Minimum price increment
            lot_size: Minimum quantity increment
            enable_stp: Enable self-trade prevention
            stp_action: STP action mode
            participant_id: Our participant ID for STP
            on_fill: Callback for fills
            on_cancel: Callback for cancellations
            on_event: Callback for all events
        """
        self._symbol = symbol
        self._tick_size = tick_size
        self._lot_size = lot_size
        self._participant_id = participant_id

        # Core components
        self._order_book = OrderBook(
            symbol=symbol,
            tick_size=tick_size,
            lot_size=lot_size,
        )

        self._matching_engine = MatchingEngine(
            stp_action=stp_action,
            enable_stp=enable_stp,
            on_trade=self._handle_trade,
        )

        self._queue_tracker = QueuePositionTracker(
            on_position_update=self._handle_position_update,
        )

        # Order tracking
        self._orders: Dict[str, ManagedOrder] = {}
        self._client_id_map: Dict[str, str] = {}  # client_id -> internal_id

        # Callbacks
        self._on_fill = on_fill
        self._on_cancel = on_cancel
        self._on_event = on_event

        # Statistics
        self._stats = OrderManagerStats()

    # ==========================================================================
    # Order Submission
    # ==========================================================================

    def submit_order(
        self,
        side: Side,
        price: float,
        qty: float,
        order_type: OrderType = OrderType.LIMIT,
        time_in_force: TimeInForce = TimeInForce.DAY,
        client_order_id: Optional[str] = None,
        is_own: bool = True,
        participant_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> ManagedOrder:
        """
        Submit a new order.

        Args:
            side: BUY or SELL
            price: Limit price (ignored for MARKET orders)
            qty: Order quantity
            order_type: LIMIT, MARKET, etc.
            time_in_force: Time-in-force setting
            client_order_id: Client-provided order ID
            is_own: Whether this is our order
            participant_id: Participant ID for STP
            tags: Custom metadata

        Returns:
            ManagedOrder with execution details
        """
        # Generate order ID
        internal_id = str(uuid.uuid4())
        order_id = client_order_id or internal_id

        # Validate order
        validation_error = self._validate_order(side, price, qty, order_type)
        if validation_error:
            return self._create_rejected_order(order_id, side, price, qty, validation_error)

        # Create LimitOrder
        limit_order = LimitOrder(
            order_id=order_id,
            price=self._round_price(price),
            qty=qty,
            remaining_qty=qty,
            timestamp_ns=time.time_ns(),
            side=side,
            is_own=is_own,
            order_type=order_type,
            participant_id=participant_id or self._participant_id,
        )

        # Create ManagedOrder
        managed = ManagedOrder(
            order=limit_order,
            state=OrderLifecycleState.PENDING,
            original_qty=qty,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
            internal_id=internal_id,
            tags=tags or {},
        )

        # Track order - store by order_id (which equals client_order_id if provided)
        self._orders[order_id] = managed
        if client_order_id and client_order_id != internal_id:
            # FIX: Map client_id -> order_id (not internal_id) for correct lookup
            self._client_id_map[client_order_id] = order_id
        self._stats.total_orders += 1

        # Handle by order type
        if order_type == OrderType.MARKET:
            return self._execute_market_order(managed)
        else:
            return self._execute_limit_order(managed, time_in_force)

    def _execute_market_order(
        self,
        managed: ManagedOrder,
    ) -> ManagedOrder:
        """Execute market order."""
        result = self._matching_engine.match_market_order(
            side=managed.order.side,
            qty=managed.order.remaining_qty,
            order_book=self._order_book,
            taker_order_id=managed.order.order_id,
            taker_participant_id=managed.order.participant_id,
        )

        # Update managed order with fills
        for fill in result.fills:
            managed.add_fill(fill)
            self._stats.total_fills += 1
            self._stats.total_volume_filled += fill.total_qty

            if self._on_fill:
                self._on_fill(managed, fill)

        # Update state
        if result.is_complete:
            managed.state = OrderLifecycleState.FILLED
            self._stats.filled_orders += 1
        else:
            # Partial fill - cancel remaining for market order (insufficient liquidity)
            # FIX: was result.total_filled_qty - managed.original_qty (negative!)
            managed.cancelled_qty = managed.original_qty - result.total_filled_qty
            managed.state = OrderLifecycleState.FILLED
            self._stats.filled_orders += 1
            self._stats.total_volume_cancelled += managed.cancelled_qty

        self._emit_event(
            OrderEvent(
                event_type=(
                    OrderEventType.FILLED if result.is_complete else OrderEventType.PARTIALLY_FILLED
                ),
                order_id=managed.order.order_id,
                timestamp_ns=time.time_ns(),
            )
        )

        return managed

    def _execute_limit_order(
        self,
        managed: ManagedOrder,
        time_in_force: TimeInForce,
    ) -> ManagedOrder:
        """Execute limit order (match aggressive, add passive to book)."""
        # For FOK orders, first check if we can fill completely
        if time_in_force == TimeInForce.FOK:
            # Simulate to check available liquidity at limit price
            can_fill = self._can_fill_completely(
                side=managed.order.side,
                price=managed.order.price,
                qty=managed.order.remaining_qty,
            )
            if not can_fill:
                # Cancel entire order without any execution
                managed.cancelled_qty = managed.original_qty
                managed.state = OrderLifecycleState.CANCELLED
                self._stats.cancelled_orders += 1
                return managed

        # Match against book
        result = self._matching_engine.match_limit_order(
            order=managed.order,
            order_book=self._order_book,
        )

        # Process fills
        for fill in result.fills:
            managed.add_fill(fill)
            self._stats.total_fills += 1
            self._stats.total_volume_filled += fill.total_qty

            if self._on_fill:
                self._on_fill(managed, fill)

        # Handle IOC (Immediate-Or-Cancel)
        if time_in_force == TimeInForce.IOC:
            if result.resting_order:
                # Cancel unfilled portion - IOC doesn't rest in book
                managed.cancelled_qty = result.resting_order.remaining_qty
                self._stats.total_volume_cancelled += managed.cancelled_qty
                # FIX: IOC with partial fill should be CANCELLED (not FILLED)
                # Per FIX protocol: OrdStatus=4 (Canceled) with CumQty > 0
                # Use CANCELLED state - filled_qty tracks what was executed
                managed.state = OrderLifecycleState.CANCELLED
                self._stats.cancelled_orders += 1
                return managed

        # Add resting portion to book
        if result.resting_order and result.resting_order.remaining_qty > 0:
            # Get level qty before adding
            level_qty_before = self._get_level_qty(
                result.resting_order.side,
                result.resting_order.price,
            )

            # Add to book
            queue_pos = self._order_book.add_limit_order(result.resting_order)
            managed.order = result.resting_order
            managed.state = (
                OrderLifecycleState.PARTIALLY_FILLED
                if managed.filled_qty > 0
                else OrderLifecycleState.NEW
            )

            self._stats.active_orders += 1

            # Track queue position
            self._queue_tracker.add_order(
                result.resting_order,
                level_qty_before=level_qty_before,
            )

            self._emit_event(
                OrderEvent(
                    event_type=OrderEventType.ACCEPTED,
                    order_id=managed.order.order_id,
                    timestamp_ns=time.time_ns(),
                )
            )
        else:
            # Fully filled
            managed.state = OrderLifecycleState.FILLED
            self._stats.filled_orders += 1

            self._emit_event(
                OrderEvent(
                    event_type=OrderEventType.FILLED,
                    order_id=managed.order.order_id,
                    timestamp_ns=time.time_ns(),
                )
            )

        return managed

    # ==========================================================================
    # Order Cancellation
    # ==========================================================================

    def cancel_order(
        self,
        order_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel
            reason: Optional cancellation reason

        Returns:
            True if order was cancelled, False if not found or already done
        """
        managed = self._orders.get(order_id)
        if managed is None:
            return False

        if not managed.is_active:
            return False

        # Remove from book
        cancelled_order = self._order_book.cancel_order(order_id)

        if cancelled_order:
            managed.cancelled_qty = cancelled_order.remaining_qty
            self._stats.total_volume_cancelled += cancelled_order.remaining_qty

        # Update state
        old_state = managed.state
        if managed.filled_qty > 0:
            managed.state = OrderLifecycleState.PARTIALLY_FILLED
        else:
            managed.state = OrderLifecycleState.CANCELLED

        self._stats.active_orders -= 1
        self._stats.cancelled_orders += 1

        # Remove from queue tracker
        self._queue_tracker.remove_order(order_id)

        # Callbacks
        if self._on_cancel:
            self._on_cancel(managed)

        self._emit_event(
            OrderEvent(
                event_type=OrderEventType.CANCELLED,
                order_id=order_id,
                timestamp_ns=time.time_ns(),
                reason=reason,
                old_state=old_state,
                new_state=managed.state,
            )
        )

        return True

    def cancel_all_orders(
        self,
        side: Optional[Side] = None,
    ) -> int:
        """
        Cancel all active orders.

        Args:
            side: Optional side filter

        Returns:
            Number of orders cancelled
        """
        cancelled_count = 0

        for order_id, managed in list(self._orders.items()):
            if not managed.is_active:
                continue
            if side is not None and managed.order.side != side:
                continue

            if self.cancel_order(order_id):
                cancelled_count += 1

        return cancelled_count

    # ==========================================================================
    # Order Modification
    # ==========================================================================

    def modify_order(
        self,
        order_id: str,
        new_qty: Optional[float] = None,
        new_price: Optional[float] = None,
    ) -> Optional[ManagedOrder]:
        """
        Modify an existing order.

        Price changes and quantity increases lose queue priority.
        Quantity decreases maintain queue priority.

        Args:
            order_id: Order ID to modify
            new_qty: New quantity (optional)
            new_price: New price (optional)

        Returns:
            Modified ManagedOrder or None if not found
        """
        managed = self._orders.get(order_id)
        if managed is None:
            return None

        if not managed.is_active:
            return None

        old_state = managed.state

        # Modify in book
        modified_order = self._order_book.modify_order(
            order_id=order_id,
            new_qty=new_qty,
            new_price=new_price,
        )

        if modified_order:
            managed.order = modified_order
            managed.last_update_ns = time.time_ns()

            # Update queue tracking if price changed
            if new_price is not None and new_price != managed.order.price:
                self._queue_tracker.remove_order(order_id)
                level_qty = self._get_level_qty(
                    modified_order.side,
                    modified_order.price,
                )
                self._queue_tracker.add_order(
                    modified_order,
                    level_qty_before=level_qty - modified_order.remaining_qty,
                )

            self._emit_event(
                OrderEvent(
                    event_type=OrderEventType.MODIFIED,
                    order_id=order_id,
                    timestamp_ns=time.time_ns(),
                    old_state=old_state,
                    new_state=managed.state,
                )
            )

        return managed

    # ==========================================================================
    # Order Query
    # ==========================================================================

    def get_order(self, order_id: str) -> Optional[ManagedOrder]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_order_by_client_id(
        self,
        client_order_id: str,
    ) -> Optional[ManagedOrder]:
        """Get order by client order ID."""
        internal_id = self._client_id_map.get(client_order_id)
        if internal_id:
            return self._orders.get(internal_id)
        return self._orders.get(client_order_id)

    def get_active_orders(
        self,
        side: Optional[Side] = None,
    ) -> List[ManagedOrder]:
        """Get all active orders."""
        result = []
        for managed in self._orders.values():
            if not managed.is_active:
                continue
            if side is not None and managed.order.side != side:
                continue
            result.append(managed)
        return result

    def get_queue_state(
        self,
        order_id: str,
    ) -> Optional[QueueState]:
        """Get queue position state for an order."""
        return self._queue_tracker.get_state(order_id)

    def get_fill_probability(
        self,
        order_id: str,
        volume_per_second: float = 100.0,
        time_horizon_sec: float = 60.0,
    ) -> FillProbability:
        """Get fill probability for an order."""
        managed = self._orders.get(order_id)
        if managed is None:
            return FillProbability(order_id=order_id, prob_fill=0.0)

        return self._queue_tracker.estimate_fill_probability(
            order_id=order_id,
            volume_per_second=volume_per_second,
            time_horizon_sec=time_horizon_sec,
            order_qty=managed.remaining_qty,
        )

    # ==========================================================================
    # Book Access
    # ==========================================================================

    @property
    def order_book(self) -> OrderBook:
        """Get underlying order book."""
        return self._order_book

    @property
    def matching_engine(self) -> MatchingEngine:
        """Get matching engine."""
        return self._matching_engine

    @property
    def queue_tracker(self) -> QueuePositionTracker:
        """Get queue tracker."""
        return self._queue_tracker

    def get_best_bid(self) -> Optional[float]:
        """Get best bid price."""
        return self._order_book.best_bid

    def get_best_ask(self) -> Optional[float]:
        """Get best ask price."""
        return self._order_book.best_ask

    def get_mid_price(self) -> Optional[float]:
        """Get mid price."""
        return self._order_book.mid_price

    def get_spread_bps(self) -> Optional[float]:
        """Get spread in basis points."""
        return self._order_book.spread_bps

    def get_depth(
        self,
        n_levels: int = 10,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Get order book depth."""
        return self._order_book.get_depth(n_levels)

    # ==========================================================================
    # Internal Helpers
    # ==========================================================================

    def _validate_order(
        self,
        side: Side,
        price: float,
        qty: float,
        order_type: OrderType,
    ) -> Optional[str]:
        """Validate order parameters."""
        if qty <= 0:
            return "Quantity must be positive"

        if order_type != OrderType.MARKET and price <= 0:
            return "Price must be positive for limit orders"

        if qty % self._lot_size != 0:
            # Allow non-round lots but warn
            pass

        return None

    def _round_price(self, price: float) -> float:
        """Round price to tick size."""
        return round(price / self._tick_size) * self._tick_size

    def _get_level_qty(self, side: Side, price: float) -> float:
        """Get total quantity at a price level."""
        if side == Side.BUY:
            key = -price
            levels = self._order_book._bids
        else:
            key = price
            levels = self._order_book._asks

        if key not in levels:
            return 0.0
        return levels[key].total_visible_qty

    def _can_fill_completely(
        self,
        side: Side,
        price: float,
        qty: float,
    ) -> bool:
        """Check if order can be completely filled at limit price or better."""
        # Walk the book to see if we can fill completely
        if side == Side.BUY:
            # Can buy at or below our limit price
            levels = self._order_book._asks
            get_price = lambda key: key
            price_ok = lambda p: p <= price
        else:
            # Can sell at or above our limit price
            levels = self._order_book._bids
            get_price = lambda key: -key
            price_ok = lambda p: p >= price

        available = 0.0
        for key in levels.keys():
            level_price = get_price(key)
            if not price_ok(level_price):
                break
            level = levels[key]
            available += level.total_visible_qty + level.total_hidden_qty
            if available >= qty:
                return True

        return available >= qty

    def _create_rejected_order(
        self,
        order_id: str,
        side: Side,
        price: float,
        qty: float,
        reason: str,
    ) -> ManagedOrder:
        """Create rejected order."""
        order = LimitOrder(
            order_id=order_id,
            price=price,
            qty=qty,
            remaining_qty=qty,
            timestamp_ns=time.time_ns(),
            side=side,
        )

        managed = ManagedOrder(
            order=order,
            state=OrderLifecycleState.REJECTED,
            original_qty=qty,
        )

        self._orders[order_id] = managed
        self._stats.rejected_orders += 1

        self._emit_event(
            OrderEvent(
                event_type=OrderEventType.REJECTED,
                order_id=order_id,
                timestamp_ns=time.time_ns(),
                reason=reason,
            )
        )

        return managed

    def _handle_trade(self, trade: Trade) -> None:
        """Handle trade from matching engine."""
        # Update queue positions for executions at this price
        self._queue_tracker.update_on_execution(
            executed_qty=trade.qty,
            at_price=trade.price,
        )

        # Update ManagedOrder for maker (passive) order
        maker_id = trade.maker_order_id
        if maker_id and maker_id in self._orders:
            maker_managed = self._orders[maker_id]

            # Create fill for maker
            maker_fill = Fill(
                order_id=maker_id,
                total_qty=trade.qty,
                avg_price=trade.price,
                trades=[trade],
                remaining_qty=maker_managed.remaining_qty - trade.qty,
                is_complete=(maker_managed.remaining_qty - trade.qty) <= 0,
            )
            maker_managed.add_fill(maker_fill)
            self._stats.total_fills += 1
            self._stats.total_volume_filled += trade.qty

            # Update state
            if maker_managed.remaining_qty <= 0:
                maker_managed.state = OrderLifecycleState.FILLED
                self._stats.active_orders -= 1
                self._stats.filled_orders += 1
                self._queue_tracker.remove_order(maker_id)

            # Callback
            if self._on_fill:
                self._on_fill(maker_managed, maker_fill)

    def _handle_position_update(
        self,
        order_id: str,
        state: QueueState,
    ) -> None:
        """Handle queue position update."""
        # Can add logging or additional callbacks here
        pass

    def _emit_event(self, event: OrderEvent) -> None:
        """Emit order event."""
        if self._on_event:
            self._on_event(event)

    # ==========================================================================
    # Statistics
    # ==========================================================================

    def get_statistics(self) -> OrderManagerStats:
        """Get order manager statistics."""
        # Update active orders count
        self._stats.active_orders = sum(1 for m in self._orders.values() if m.is_active)

        # Calculate average fill rate
        filled_orders = [m for m in self._orders.values() if m.filled_qty > 0]
        if filled_orders:
            self._stats.avg_fill_rate = sum(m.fill_rate for m in filled_orders) / len(filled_orders)

        return self._stats

    def reset_statistics(self) -> None:
        """Reset statistics."""
        self._stats = OrderManagerStats()
        self._matching_engine.reset_statistics()

    def clear(self) -> None:
        """Clear all orders and reset state."""
        self._orders.clear()
        self._client_id_map.clear()
        self._order_book.clear()
        self._queue_tracker.clear()
        self.reset_statistics()


# ==============================================================================
# Factory Function
# ==============================================================================


def create_order_manager(
    symbol: str = "",
    **kwargs,
) -> OrderManager:
    """
    Create order manager instance.

    Args:
        symbol: Trading symbol
        **kwargs: Additional OrderManager arguments

    Returns:
        OrderManager instance
    """
    return OrderManager(symbol=symbol, **kwargs)
