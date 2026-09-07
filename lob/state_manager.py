"""
LOB State Manager for efficient order book updates.

Manages order book state with:
- Incremental updates from LOBSTER/ITCH messages
- Snapshot reconstruction for cold start
- Memory-efficient storage
- State persistence for simulation checkpoints

This module provides the bridge between raw market data messages
and the OrderBook data structure.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from lob.data_structures import (
    Fill,
    LimitOrder,
    OrderBook,
    OrderType,
    PriceLevel,
    Side,
    Trade,
)
from lob.parsers import (
    ITCHAddOrder,
    ITCHMessage,
    ITCHOrderCancel,
    ITCHOrderDelete,
    ITCHOrderExecuted,
    ITCHOrderReplace,
    LOBSTERMessage,
    LOBSTEREventType,
)


# ==============================================================================
# Message Types
# ==============================================================================


class MessageType(IntEnum):
    """Standardized message types for state updates."""

    ADD = 1
    MODIFY = 2
    DELETE = 3
    EXECUTE = 4
    TRADE = 5
    SNAPSHOT = 6


@dataclass
class LOBMessage:
    """
    Standardized LOB message for state updates.

    This is a unified message format that can be created from
    LOBSTER, ITCH, or other source formats.
    """

    msg_type: MessageType
    timestamp_ns: int
    order_id: str
    side: Side
    price: float
    qty: float

    # Optional fields
    execute_qty: float = 0.0
    new_order_id: Optional[str] = None  # For MODIFY (replace)
    trade_id: Optional[str] = None
    is_hidden: bool = False

    @classmethod
    def from_lobster(cls, msg: LOBSTERMessage) -> "LOBMessage":
        """Create from LOBSTER message."""
        if msg.event_type == LOBSTEREventType.ADD:
            msg_type = MessageType.ADD
        elif msg.event_type == LOBSTEREventType.MODIFY:
            msg_type = MessageType.MODIFY
        elif msg.event_type == LOBSTEREventType.DELETE:
            msg_type = MessageType.DELETE
        elif msg.event_type in (LOBSTEREventType.EXECUTE, LOBSTEREventType.HIDDEN_EXECUTE):
            msg_type = MessageType.EXECUTE
        elif msg.event_type == LOBSTEREventType.CROSS:
            msg_type = MessageType.TRADE
        else:
            msg_type = MessageType.ADD  # Default

        return cls(
            msg_type=msg_type,
            timestamp_ns=msg.timestamp_ns,
            order_id=str(msg.order_id),
            side=msg.side,
            price=msg.price,
            qty=float(msg.size),
            execute_qty=float(msg.size) if msg_type == MessageType.EXECUTE else 0.0,
            is_hidden=msg.event_type == LOBSTEREventType.HIDDEN_EXECUTE,
        )

    @classmethod
    def from_itch_add(cls, msg: ITCHAddOrder) -> "LOBMessage":
        """Create from ITCH Add Order."""
        return cls(
            msg_type=MessageType.ADD,
            timestamp_ns=msg.timestamp_ns,
            order_id=str(msg.order_ref),
            side=Side.BUY if msg.side == "B" else Side.SELL,
            price=msg.price,
            qty=float(msg.shares),
        )

    @classmethod
    def from_itch_execute(cls, msg: ITCHOrderExecuted) -> "LOBMessage":
        """Create from ITCH Order Executed."""
        return cls(
            msg_type=MessageType.EXECUTE,
            timestamp_ns=msg.timestamp_ns,
            order_id=str(msg.order_ref),
            side=Side.BUY,  # Side unknown from execute message
            price=msg.price or 0.0,
            qty=0.0,  # Original qty unknown
            execute_qty=float(msg.shares),
        )

    @classmethod
    def from_itch_cancel(cls, msg: ITCHOrderCancel) -> "LOBMessage":
        """Create from ITCH Order Cancel."""
        return cls(
            msg_type=MessageType.MODIFY,  # Partial cancel = modify
            timestamp_ns=msg.timestamp_ns,
            order_id=str(msg.order_ref),
            side=Side.BUY,  # Side unknown
            price=0.0,  # Price unknown
            qty=-float(msg.cancelled_shares),  # Negative = reduce qty
        )

    @classmethod
    def from_itch_delete(cls, msg: ITCHOrderDelete) -> "LOBMessage":
        """Create from ITCH Order Delete."""
        return cls(
            msg_type=MessageType.DELETE,
            timestamp_ns=msg.timestamp_ns,
            order_id=str(msg.order_ref),
            side=Side.BUY,  # Side unknown
            price=0.0,
            qty=0.0,
        )

    @classmethod
    def from_itch_replace(cls, msg: ITCHOrderReplace) -> "LOBMessage":
        """Create from ITCH Order Replace."""
        return cls(
            msg_type=MessageType.MODIFY,
            timestamp_ns=msg.timestamp_ns,
            order_id=str(msg.original_ref),
            side=Side.BUY,  # Side unknown
            price=msg.price,
            qty=float(msg.shares),
            new_order_id=str(msg.new_ref),
        )


@dataclass
class LOBSnapshot:
    """
    Full order book snapshot for reconstruction.

    Can be used for:
    - Cold start initialization
    - Simulation checkpoints
    - State persistence
    """

    timestamp_ns: int
    symbol: str
    bids: List[Tuple[float, float]]  # List of (price, qty)
    asks: List[Tuple[float, float]]
    sequence: int = 0

    # Optional: full order details for MBO reconstruction
    bid_orders: Optional[List[Dict[str, Any]]] = None
    ask_orders: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp_ns": self.timestamp_ns,
            "symbol": self.symbol,
            "bids": self.bids,
            "asks": self.asks,
            "sequence": self.sequence,
            "bid_orders": self.bid_orders,
            "ask_orders": self.ask_orders,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LOBSnapshot":
        """Create from dictionary."""
        return cls(
            timestamp_ns=data["timestamp_ns"],
            symbol=data["symbol"],
            bids=[(p, q) for p, q in data["bids"]],
            asks=[(p, q) for p, q in data["asks"]],
            sequence=data.get("sequence", 0),
            bid_orders=data.get("bid_orders"),
            ask_orders=data.get("ask_orders"),
        )

    @classmethod
    def from_orderbook(cls, book: OrderBook, timestamp_ns: Optional[int] = None) -> "LOBSnapshot":
        """Create snapshot from OrderBook."""
        bids, asks = book.get_depth(n_levels=100)

        # Extract full order details if needed
        bid_orders = []
        ask_orders = []

        for order in book.get_mbo_snapshot(Side.BUY, n_orders=1000):
            bid_orders.append(
                {
                    "order_id": order.order_id,
                    "price": order.price,
                    "qty": order.remaining_qty,
                    "timestamp_ns": order.timestamp_ns,
                    "hidden_qty": order.hidden_qty,
                }
            )

        for order in book.get_mbo_snapshot(Side.SELL, n_orders=1000):
            ask_orders.append(
                {
                    "order_id": order.order_id,
                    "price": order.price,
                    "qty": order.remaining_qty,
                    "timestamp_ns": order.timestamp_ns,
                    "hidden_qty": order.hidden_qty,
                }
            )

        return cls(
            timestamp_ns=timestamp_ns or time.time_ns(),
            symbol=book.symbol,
            bids=list(bids),
            asks=list(asks),
            sequence=book.sequence,
            bid_orders=bid_orders,
            ask_orders=ask_orders,
        )


# ==============================================================================
# State Manager
# ==============================================================================


class LOBStateManager:
    """
    Manages order book state with efficient updates.

    Features:
    - Incremental message processing
    - Snapshot reconstruction
    - Statistics tracking
    - Callback hooks for events

    Usage:
        manager = LOBStateManager(symbol="AAPL")

        # Process messages
        for msg in parser.parse_file("messages.csv"):
            manager.apply_message(LOBMessage.from_lobster(msg))

        # Get current state
        book = manager.orderbook
        print(f"Mid: {book.mid_price}, Spread: {book.spread_bps} bps")
    """

    def __init__(
        self,
        symbol: str = "",
        tick_size: float = 0.01,
        lot_size: float = 1.0,
        on_trade: Optional[Callable[[Trade], None]] = None,
        on_level_change: Optional[Callable[[Side, float, float], None]] = None,
    ) -> None:
        """
        Initialize state manager.

        Args:
            symbol: Trading symbol
            tick_size: Minimum price increment
            lot_size: Minimum quantity increment
            on_trade: Callback for trade events (trade: Trade)
            on_level_change: Callback for level changes (side, price, new_qty)
        """
        self._book = OrderBook(symbol=symbol, tick_size=tick_size, lot_size=lot_size)

        # Callbacks
        self._on_trade = on_trade
        self._on_level_change = on_level_change

        # Statistics
        self._message_count = 0
        self._add_count = 0
        self._delete_count = 0
        self._execute_count = 0
        self._modify_count = 0
        self._trade_count = 0
        self._last_timestamp_ns = 0

        # Order tracking for ITCH (doesn't include side in execute/delete)
        self._order_sides: Dict[str, Side] = {}
        self._order_prices: Dict[str, float] = {}

    @property
    def orderbook(self) -> OrderBook:
        """Get current order book state."""
        return self._book

    @property
    def message_count(self) -> int:
        """Total messages processed."""
        return self._message_count

    @property
    def last_timestamp_ns(self) -> int:
        """Timestamp of last processed message."""
        return self._last_timestamp_ns

    def apply_message(self, msg: LOBMessage) -> Optional[Trade]:
        """
        Process single LOB message and update state.

        Args:
            msg: Standardized LOB message

        Returns:
            Trade if execution occurred, None otherwise
        """
        self._message_count += 1
        self._last_timestamp_ns = msg.timestamp_ns

        if msg.msg_type == MessageType.ADD:
            return self._handle_add(msg)
        elif msg.msg_type == MessageType.MODIFY:
            return self._handle_modify(msg)
        elif msg.msg_type == MessageType.DELETE:
            return self._handle_delete(msg)
        elif msg.msg_type == MessageType.EXECUTE:
            return self._handle_execute(msg)
        elif msg.msg_type == MessageType.TRADE:
            return self._handle_trade(msg)

        return None

    def apply_lobster_message(self, msg: LOBSTERMessage) -> Optional[Trade]:
        """
        Process LOBSTER message directly.

        Args:
            msg: LOBSTER message

        Returns:
            Trade if execution occurred
        """
        return self.apply_message(LOBMessage.from_lobster(msg))

    def apply_itch_message(self, msg: ITCHMessage) -> Optional[Trade]:
        """
        Process ITCH message directly.

        Args:
            msg: ITCH message

        Returns:
            Trade if execution occurred
        """
        if isinstance(msg, ITCHAddOrder):
            lob_msg = LOBMessage.from_itch_add(msg)
        elif isinstance(msg, ITCHOrderExecuted):
            lob_msg = LOBMessage.from_itch_execute(msg)
        elif isinstance(msg, ITCHOrderCancel):
            lob_msg = LOBMessage.from_itch_cancel(msg)
        elif isinstance(msg, ITCHOrderDelete):
            lob_msg = LOBMessage.from_itch_delete(msg)
        elif isinstance(msg, ITCHOrderReplace):
            lob_msg = LOBMessage.from_itch_replace(msg)
        else:
            return None

        return self.apply_message(lob_msg)

    def _handle_add(self, msg: LOBMessage) -> None:
        """Handle ADD message."""
        self._add_count += 1

        order = LimitOrder(
            order_id=msg.order_id,
            price=msg.price,
            qty=msg.qty,
            remaining_qty=msg.qty,
            timestamp_ns=msg.timestamp_ns,
            side=msg.side,
            order_type=OrderType.HIDDEN if msg.is_hidden else OrderType.LIMIT,
        )

        # Track order info for ITCH
        self._order_sides[msg.order_id] = msg.side
        self._order_prices[msg.order_id] = msg.price

        self._book.add_limit_order(order)

        if self._on_level_change:
            self._on_level_change(
                msg.side,
                msg.price,
                self._get_level_qty(msg.side, msg.price),
            )

        return None

    def _handle_modify(self, msg: LOBMessage) -> None:
        """Handle MODIFY message (cancel + replace)."""
        self._modify_count += 1

        # Get original order info
        side = self._order_sides.get(msg.order_id, msg.side)
        old_price = self._order_prices.get(msg.order_id, msg.price)

        if msg.new_order_id:
            # Replace: cancel old, add new
            self._book.cancel_order(msg.order_id)

            new_order = LimitOrder(
                order_id=msg.new_order_id,
                price=msg.price,
                qty=msg.qty,
                remaining_qty=msg.qty,
                timestamp_ns=msg.timestamp_ns,
                side=side,
            )
            self._book.add_limit_order(new_order)

            # Update tracking
            del self._order_sides[msg.order_id]
            del self._order_prices[msg.order_id]
            self._order_sides[msg.new_order_id] = side
            self._order_prices[msg.new_order_id] = msg.price

        elif msg.qty < 0:
            # Partial cancel (qty is reduction amount)
            order = self._book.get_order(msg.order_id)
            if order:
                new_qty = max(0, order.remaining_qty + msg.qty)
                self._book.modify_order(msg.order_id, new_qty=new_qty)
        else:
            # Full modify
            self._book.modify_order(
                msg.order_id,
                new_qty=msg.qty if msg.qty > 0 else None,
                new_price=msg.price if msg.price > 0 else None,
            )

        return None

    def _handle_delete(self, msg: LOBMessage) -> None:
        """Handle DELETE message."""
        self._delete_count += 1

        side = self._order_sides.get(msg.order_id, msg.side)
        price = self._order_prices.get(msg.order_id, msg.price)

        self._book.cancel_order(msg.order_id)

        # Clean up tracking
        self._order_sides.pop(msg.order_id, None)
        self._order_prices.pop(msg.order_id, None)

        if self._on_level_change and price > 0:
            self._on_level_change(
                side,
                price,
                self._get_level_qty(side, price),
            )

        return None

    def _handle_execute(self, msg: LOBMessage) -> Optional[Trade]:
        """Handle EXECUTE message."""
        self._execute_count += 1

        order = self._book.get_order(msg.order_id)
        if not order:
            return None

        side = order.side
        price = msg.price if msg.price > 0 else order.price

        # Create trade
        trade = Trade(
            price=price,
            qty=msg.execute_qty,
            maker_order_id=msg.order_id,
            timestamp_ns=msg.timestamp_ns,
            aggressor_side=Side.SELL if side == Side.BUY else Side.BUY,
        )
        self._trade_count += 1

        # Update order in book
        order.fill(msg.execute_qty)
        if order.is_filled:
            self._book.cancel_order(msg.order_id)
            self._order_sides.pop(msg.order_id, None)
            self._order_prices.pop(msg.order_id, None)

        if self._on_trade:
            self._on_trade(trade)

        if self._on_level_change:
            self._on_level_change(
                side,
                price,
                self._get_level_qty(side, price),
            )

        return trade

    def _handle_trade(self, msg: LOBMessage) -> Optional[Trade]:
        """Handle TRADE message (cross/auction)."""
        self._trade_count += 1

        trade = Trade(
            price=msg.price,
            qty=msg.qty,
            maker_order_id="",
            timestamp_ns=msg.timestamp_ns,
            trade_id=msg.trade_id,
        )

        if self._on_trade:
            self._on_trade(trade)

        return trade

    def _get_level_qty(self, side: Side, price: float) -> float:
        """Get total quantity at price level."""
        if side == Side.BUY:
            key = -price
            levels = self._book._bids
        else:
            key = price
            levels = self._book._asks

        if key not in levels:
            return 0.0
        return levels[key].total_visible_qty

    # ==========================================================================
    # Snapshot Operations
    # ==========================================================================

    def reconstruct_from_snapshot(self, snapshot: LOBSnapshot) -> None:
        """
        Initialize order book from snapshot.

        Args:
            snapshot: LOB snapshot to restore
        """
        # Clear current state
        self._book.clear()
        self._order_sides.clear()
        self._order_prices.clear()

        # Update symbol
        self._book.symbol = snapshot.symbol

        if snapshot.bid_orders and snapshot.ask_orders:
            # Full MBO reconstruction
            for order_data in snapshot.bid_orders:
                order = LimitOrder(
                    order_id=order_data["order_id"],
                    price=order_data["price"],
                    qty=order_data["qty"],
                    remaining_qty=order_data["qty"],
                    timestamp_ns=order_data.get("timestamp_ns", snapshot.timestamp_ns),
                    side=Side.BUY,
                    hidden_qty=order_data.get("hidden_qty", 0.0),
                )
                self._book.add_limit_order(order)
                self._order_sides[order.order_id] = Side.BUY
                self._order_prices[order.order_id] = order.price

            for order_data in snapshot.ask_orders:
                order = LimitOrder(
                    order_id=order_data["order_id"],
                    price=order_data["price"],
                    qty=order_data["qty"],
                    remaining_qty=order_data["qty"],
                    timestamp_ns=order_data.get("timestamp_ns", snapshot.timestamp_ns),
                    side=Side.SELL,
                    hidden_qty=order_data.get("hidden_qty", 0.0),
                )
                self._book.add_limit_order(order)
                self._order_sides[order.order_id] = Side.SELL
                self._order_prices[order.order_id] = order.price
        else:
            # MBP reconstruction (create synthetic orders)
            for i, (price, qty) in enumerate(snapshot.bids):
                order = LimitOrder(
                    order_id=f"bid_{i}_{price}",
                    price=price,
                    qty=qty,
                    remaining_qty=qty,
                    timestamp_ns=snapshot.timestamp_ns,
                    side=Side.BUY,
                )
                self._book.add_limit_order(order)

            for i, (price, qty) in enumerate(snapshot.asks):
                order = LimitOrder(
                    order_id=f"ask_{i}_{price}",
                    price=price,
                    qty=qty,
                    remaining_qty=qty,
                    timestamp_ns=snapshot.timestamp_ns,
                    side=Side.SELL,
                )
                self._book.add_limit_order(order)

        self._last_timestamp_ns = snapshot.timestamp_ns

    def create_snapshot(self) -> LOBSnapshot:
        """
        Create snapshot of current state.

        Returns:
            LOBSnapshot
        """
        return LOBSnapshot.from_orderbook(self._book, self._last_timestamp_ns)

    def save_snapshot(self, filepath: Union[str, Path]) -> None:
        """Save snapshot to JSON file."""
        snapshot = self.create_snapshot()
        with open(filepath, "w") as f:
            json.dump(snapshot.to_dict(), f)

    def load_snapshot(self, filepath: Union[str, Path]) -> None:
        """Load and restore snapshot from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        snapshot = LOBSnapshot.from_dict(data)
        self.reconstruct_from_snapshot(snapshot)

    # ==========================================================================
    # Queue Position Tracking
    # ==========================================================================

    def get_queue_position(self, order_id: str) -> Optional[int]:
        """
        Get current queue position for order.

        Args:
            order_id: Order ID

        Returns:
            Queue position (0 = front) or None if not found
        """
        return self._book.get_queue_position(order_id)

    def estimate_fill_probability(
        self,
        order_id: str,
        volume_per_second: float = 1000.0,
        time_horizon_sec: float = 60.0,
    ) -> float:
        """
        Estimate probability of order being filled.

        Simple model based on queue position and volume rate.

        Args:
            order_id: Order ID
            volume_per_second: Expected volume rate at this price
            time_horizon_sec: Time horizon for estimation

        Returns:
            Estimated fill probability [0, 1]
        """
        pos = self.get_queue_position(order_id)
        if pos is None:
            return 0.0

        order = self._book.get_order(order_id)
        if order is None:
            return 0.0

        # Calculate queue ahead
        side = order.side
        price = order.price

        if side == Side.BUY:
            key = -price
            levels = self._book._bids
        else:
            key = price
            levels = self._book._asks

        if key not in levels:
            return 0.0

        level = levels[key]
        queue_ahead = 0.0
        for o in level.iter_orders():
            if o.order_id == order_id:
                break
            queue_ahead += o.remaining_qty

        # Estimate fill probability
        expected_volume = volume_per_second * time_horizon_sec
        if expected_volume <= 0:
            return 0.0

        # Simple model: prob = min(1, expected_volume / queue_ahead)
        if queue_ahead <= 0:
            return 1.0  # Front of queue

        return min(1.0, expected_volume / (queue_ahead + order.remaining_qty))

    # ==========================================================================
    # Statistics
    # ==========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "message_count": self._message_count,
            "add_count": self._add_count,
            "delete_count": self._delete_count,
            "execute_count": self._execute_count,
            "modify_count": self._modify_count,
            "trade_count": self._trade_count,
            "order_count": self._book.order_count,
            "bid_levels": len(self._book._bids),
            "ask_levels": len(self._book._asks),
            "best_bid": self._book.best_bid,
            "best_ask": self._book.best_ask,
            "spread_bps": self._book.spread_bps,
            "last_timestamp_ns": self._last_timestamp_ns,
        }

    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self._message_count = 0
        self._add_count = 0
        self._delete_count = 0
        self._execute_count = 0
        self._modify_count = 0
        self._trade_count = 0


# ==============================================================================
# Utility Functions
# ==============================================================================


def build_orderbook_from_lobster(
    message_file: Union[str, Path],
    orderbook_file: Optional[Union[str, Path]] = None,
    symbol: str = "",
    price_multiplier: float = 0.0001,
    max_messages: Optional[int] = None,
) -> LOBStateManager:
    """
    Build order book from LOBSTER files.

    Args:
        message_file: Path to LOBSTER message file
        orderbook_file: Optional path to initial orderbook snapshot
        symbol: Trading symbol
        price_multiplier: Price multiplier for LOBSTER format
        max_messages: Maximum messages to process

    Returns:
        LOBStateManager with reconstructed order book
    """
    from lob.parsers import LOBSTERParser

    manager = LOBStateManager(symbol=symbol)
    parser = LOBSTERParser(price_multiplier=price_multiplier)

    # Load initial snapshot if provided
    if orderbook_file:
        asks, bids = parser.parse_orderbook_file(orderbook_file)
        snapshot = LOBSnapshot(
            timestamp_ns=0,
            symbol=symbol,
            bids=[(p, float(q)) for p, q in bids],
            asks=[(p, float(q)) for p, q in asks],
        )
        manager.reconstruct_from_snapshot(snapshot)

    # Process messages
    for msg in parser.parse_file(message_file, max_messages=max_messages):
        manager.apply_lobster_message(msg)

    return manager
