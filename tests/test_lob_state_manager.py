"""
Tests for LOB State Manager.

Tests cover:
- Message application (ADD, MODIFY, DELETE, EXECUTE)
- Snapshot creation and restoration
- Queue position tracking
- Statistics tracking
- ITCH message handling
"""

import json
import tempfile
import pytest

pytest.importorskip("sortedcontainers")

from lob.data_structures import (
    LimitOrder,
    OrderBook,
    Side,
    Trade,
)
from lob.state_manager import (
    LOBStateManager,
    LOBSnapshot,
    LOBMessage,
    MessageType,
)
from lob.parsers import (
    LOBSTERMessage,
    LOBSTEREventType,
)


# ==============================================================================
# LOBMessage Tests
# ==============================================================================


class TestLOBMessage:
    """Tests for LOBMessage class."""

    def test_from_lobster_add(self):
        """Test creating LOBMessage from LOBSTER ADD."""
        lobster = LOBSTERMessage(
            timestamp_sec=34200.5,
            event_type=LOBSTEREventType.ADD,
            order_id=12345,
            size=100,
            price=100.0,
            direction=1,
        )

        msg = LOBMessage.from_lobster(lobster)

        assert msg.msg_type == MessageType.ADD
        assert msg.order_id == "12345"
        assert msg.side == Side.BUY
        assert msg.price == 100.0
        assert msg.qty == 100.0

    def test_from_lobster_delete(self):
        """Test creating LOBMessage from LOBSTER DELETE."""
        lobster = LOBSTERMessage(
            timestamp_sec=34200.5,
            event_type=LOBSTEREventType.DELETE,
            order_id=12345,
            size=100,
            price=100.0,
            direction=-1,
        )

        msg = LOBMessage.from_lobster(lobster)

        assert msg.msg_type == MessageType.DELETE
        assert msg.side == Side.SELL

    def test_from_lobster_execute(self):
        """Test creating LOBMessage from LOBSTER EXECUTE."""
        lobster = LOBSTERMessage(
            timestamp_sec=34200.5,
            event_type=LOBSTEREventType.EXECUTE,
            order_id=12345,
            size=50,
            price=100.0,
            direction=1,
        )

        msg = LOBMessage.from_lobster(lobster)

        assert msg.msg_type == MessageType.EXECUTE
        assert msg.execute_qty == 50.0
        assert not msg.is_hidden

    def test_from_lobster_hidden_execute(self):
        """Test creating LOBMessage from LOBSTER HIDDEN EXECUTE."""
        lobster = LOBSTERMessage(
            timestamp_sec=34200.5,
            event_type=LOBSTEREventType.HIDDEN_EXECUTE,
            order_id=12345,
            size=50,
            price=100.0,
            direction=1,
        )

        msg = LOBMessage.from_lobster(lobster)

        assert msg.msg_type == MessageType.EXECUTE
        assert msg.is_hidden


# ==============================================================================
# LOBSnapshot Tests
# ==============================================================================


class TestLOBSnapshot:
    """Tests for LOBSnapshot class."""

    def test_create_snapshot(self):
        """Test creating snapshot."""
        snapshot = LOBSnapshot(
            timestamp_ns=1000000000,
            symbol="AAPL",
            bids=[(100.0, 100.0), (99.0, 200.0)],
            asks=[(101.0, 150.0), (102.0, 250.0)],
            sequence=42,
        )

        assert snapshot.symbol == "AAPL"
        assert len(snapshot.bids) == 2
        assert len(snapshot.asks) == 2

    def test_to_dict(self):
        """Test serialization to dict."""
        snapshot = LOBSnapshot(
            timestamp_ns=1000000000,
            symbol="AAPL",
            bids=[(100.0, 100.0)],
            asks=[(101.0, 150.0)],
        )

        d = snapshot.to_dict()

        assert d["symbol"] == "AAPL"
        assert d["timestamp_ns"] == 1000000000
        assert d["bids"] == [(100.0, 100.0)]

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "timestamp_ns": 1000000000,
            "symbol": "AAPL",
            "bids": [(100.0, 100.0)],
            "asks": [(101.0, 150.0)],
            "sequence": 1,
        }

        snapshot = LOBSnapshot.from_dict(d)

        assert snapshot.symbol == "AAPL"
        assert snapshot.bids[0] == (100.0, 100.0)

    def test_from_orderbook(self):
        """Test creating snapshot from OrderBook."""
        book = OrderBook(symbol="AAPL")

        book.add_limit_order(
            LimitOrder(
                order_id="bid_1",
                price=100.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000,
                side=Side.BUY,
            )
        )
        book.add_limit_order(
            LimitOrder(
                order_id="ask_1",
                price=101.0,
                qty=150.0,
                remaining_qty=150.0,
                timestamp_ns=1000,
                side=Side.SELL,
            )
        )

        snapshot = LOBSnapshot.from_orderbook(book)

        assert snapshot.symbol == "AAPL"
        assert len(snapshot.bids) == 1
        assert len(snapshot.asks) == 1
        assert snapshot.bids[0] == (100.0, 100.0)
        assert snapshot.asks[0] == (101.0, 150.0)


# ==============================================================================
# LOBStateManager Tests
# ==============================================================================


class TestLOBStateManager:
    """Tests for LOBStateManager class."""

    def test_create_manager(self):
        """Test creating state manager."""
        manager = LOBStateManager(symbol="AAPL")

        assert manager.orderbook.symbol == "AAPL"
        assert manager.message_count == 0

    def test_apply_add_message(self):
        """Test applying ADD message."""
        manager = LOBStateManager()

        msg = LOBMessage(
            msg_type=MessageType.ADD,
            timestamp_ns=1000000000,
            order_id="order_1",
            side=Side.BUY,
            price=100.0,
            qty=100.0,
        )

        manager.apply_message(msg)

        book = manager.orderbook
        assert book.best_bid == 100.0
        assert book.contains_order("order_1")

    def test_apply_delete_message(self):
        """Test applying DELETE message."""
        manager = LOBStateManager()

        # First add
        manager.apply_message(
            LOBMessage(
                msg_type=MessageType.ADD,
                timestamp_ns=1000000000,
                order_id="order_1",
                side=Side.BUY,
                price=100.0,
                qty=100.0,
            )
        )

        # Then delete
        manager.apply_message(
            LOBMessage(
                msg_type=MessageType.DELETE,
                timestamp_ns=1000000001,
                order_id="order_1",
                side=Side.BUY,
                price=100.0,
                qty=0.0,
            )
        )

        book = manager.orderbook
        assert not book.contains_order("order_1")
        assert book.best_bid is None

    def test_apply_execute_message(self):
        """Test applying EXECUTE message."""
        manager = LOBStateManager()

        # Add order
        manager.apply_message(
            LOBMessage(
                msg_type=MessageType.ADD,
                timestamp_ns=1000000000,
                order_id="order_1",
                side=Side.BUY,
                price=100.0,
                qty=100.0,
            )
        )

        # Execute part
        trade = manager.apply_message(
            LOBMessage(
                msg_type=MessageType.EXECUTE,
                timestamp_ns=1000000001,
                order_id="order_1",
                side=Side.BUY,
                price=100.0,
                qty=0.0,
                execute_qty=30.0,
            )
        )

        assert trade is not None
        assert trade.qty == 30.0

        order = manager.orderbook.get_order("order_1")
        assert order.remaining_qty == 70.0

    def test_apply_execute_full(self):
        """Test executing full order removes it."""
        manager = LOBStateManager()

        manager.apply_message(
            LOBMessage(
                msg_type=MessageType.ADD,
                timestamp_ns=1000000000,
                order_id="order_1",
                side=Side.BUY,
                price=100.0,
                qty=100.0,
            )
        )

        manager.apply_message(
            LOBMessage(
                msg_type=MessageType.EXECUTE,
                timestamp_ns=1000000001,
                order_id="order_1",
                side=Side.BUY,
                price=100.0,
                qty=0.0,
                execute_qty=100.0,
            )
        )

        assert not manager.orderbook.contains_order("order_1")

    def test_apply_modify_message(self):
        """Test applying MODIFY message."""
        manager = LOBStateManager()

        manager.apply_message(
            LOBMessage(
                msg_type=MessageType.ADD,
                timestamp_ns=1000000000,
                order_id="order_1",
                side=Side.BUY,
                price=100.0,
                qty=100.0,
            )
        )

        # Modify with new order ID (replace)
        manager.apply_message(
            LOBMessage(
                msg_type=MessageType.MODIFY,
                timestamp_ns=1000000001,
                order_id="order_1",
                side=Side.BUY,
                price=101.0,
                qty=150.0,
                new_order_id="order_2",
            )
        )

        assert not manager.orderbook.contains_order("order_1")
        assert manager.orderbook.contains_order("order_2")
        assert manager.orderbook.best_bid == 101.0

    def test_apply_lobster_message(self):
        """Test applying LOBSTER message directly."""
        manager = LOBStateManager()

        lobster = LOBSTERMessage(
            timestamp_sec=34200.5,
            event_type=LOBSTEREventType.ADD,
            order_id=12345,
            size=100,
            price=100.0,
            direction=1,
        )

        manager.apply_lobster_message(lobster)

        assert manager.orderbook.contains_order("12345")

    def test_reconstruct_from_snapshot_mbp(self):
        """Test reconstruction from MBP snapshot."""
        manager = LOBStateManager(symbol="AAPL")

        snapshot = LOBSnapshot(
            timestamp_ns=1000000000,
            symbol="AAPL",
            bids=[(100.0, 100.0), (99.0, 200.0)],
            asks=[(101.0, 150.0), (102.0, 250.0)],
        )

        manager.reconstruct_from_snapshot(snapshot)

        book = manager.orderbook
        assert book.best_bid == 100.0
        assert book.best_ask == 101.0
        assert book.best_bid_qty == 100.0

    def test_reconstruct_from_snapshot_mbo(self):
        """Test reconstruction from MBO snapshot."""
        manager = LOBStateManager(symbol="AAPL")

        snapshot = LOBSnapshot(
            timestamp_ns=1000000000,
            symbol="AAPL",
            bids=[(100.0, 100.0)],
            asks=[(101.0, 150.0)],
            bid_orders=[
                {"order_id": "bid_1", "price": 100.0, "qty": 60.0},
                {"order_id": "bid_2", "price": 100.0, "qty": 40.0},
            ],
            ask_orders=[
                {"order_id": "ask_1", "price": 101.0, "qty": 150.0},
            ],
        )

        manager.reconstruct_from_snapshot(snapshot)

        book = manager.orderbook
        assert book.contains_order("bid_1")
        assert book.contains_order("bid_2")
        assert book.best_bid_qty == 100.0  # 60 + 40

    def test_create_snapshot(self):
        """Test snapshot creation."""
        manager = LOBStateManager(symbol="AAPL")

        manager.apply_message(
            LOBMessage(
                msg_type=MessageType.ADD,
                timestamp_ns=1000000000,
                order_id="bid_1",
                side=Side.BUY,
                price=100.0,
                qty=100.0,
            )
        )

        snapshot = manager.create_snapshot()

        assert snapshot.symbol == "AAPL"
        assert len(snapshot.bids) == 1
        assert snapshot.bids[0] == (100.0, 100.0)

    def test_save_load_snapshot(self):
        """Test saving and loading snapshot."""
        manager1 = LOBStateManager(symbol="AAPL")

        manager1.apply_message(
            LOBMessage(
                msg_type=MessageType.ADD,
                timestamp_ns=1000000000,
                order_id="bid_1",
                side=Side.BUY,
                price=100.0,
                qty=100.0,
            )
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        manager1.save_snapshot(filepath)

        # Load into new manager
        manager2 = LOBStateManager()
        manager2.load_snapshot(filepath)

        assert manager2.orderbook.best_bid == 100.0

    def test_get_queue_position(self):
        """Test queue position tracking."""
        manager = LOBStateManager()

        # Add multiple orders at same price
        for i in range(3):
            manager.apply_message(
                LOBMessage(
                    msg_type=MessageType.ADD,
                    timestamp_ns=1000000000 + i,
                    order_id=f"order_{i}",
                    side=Side.BUY,
                    price=100.0,
                    qty=100.0,
                )
            )

        assert manager.get_queue_position("order_0") == 0
        assert manager.get_queue_position("order_1") == 1
        assert manager.get_queue_position("order_2") == 2
        assert manager.get_queue_position("fake") is None

    def test_estimate_fill_probability(self):
        """Test fill probability estimation."""
        manager = LOBStateManager()

        # Add orders at back of queue
        for i in range(5):
            manager.apply_message(
                LOBMessage(
                    msg_type=MessageType.ADD,
                    timestamp_ns=1000000000 + i,
                    order_id=f"order_{i}",
                    side=Side.BUY,
                    price=100.0,
                    qty=100.0,
                )
            )

        # Front of queue with low volume - still high probability (nothing ahead)
        prob_front = manager.estimate_fill_probability(
            "order_0",
            volume_per_second=1.0,  # Low volume
            time_horizon_sec=60.0,
        )
        # Queue ahead = 0, so probability is capped at 1.0
        assert prob_front == 1.0

        # Back of queue with low volume - lower probability
        # Queue ahead = 400 (4 * 100), expected_volume = 60 (1 * 60)
        # prob = min(1, 60 / (400 + 100)) = min(1, 60/500) = 0.12
        prob_back = manager.estimate_fill_probability(
            "order_4",
            volume_per_second=1.0,
            time_horizon_sec=60.0,
        )
        assert prob_back < prob_front
        assert 0.1 < prob_back < 0.2  # ~0.12

    def test_statistics(self):
        """Test statistics tracking."""
        manager = LOBStateManager()

        manager.apply_message(
            LOBMessage(
                msg_type=MessageType.ADD,
                timestamp_ns=1000,
                order_id="order_1",
                side=Side.BUY,
                price=100.0,
                qty=100.0,
            )
        )

        manager.apply_message(
            LOBMessage(
                msg_type=MessageType.EXECUTE,
                timestamp_ns=1001,
                order_id="order_1",
                side=Side.BUY,
                price=100.0,
                qty=0.0,
                execute_qty=50.0,
            )
        )

        stats = manager.get_statistics()

        assert stats["message_count"] == 2
        assert stats["add_count"] == 1
        assert stats["execute_count"] == 1
        assert stats["trade_count"] == 1

    def test_reset_statistics(self):
        """Test statistics reset."""
        manager = LOBStateManager()

        manager.apply_message(
            LOBMessage(
                msg_type=MessageType.ADD,
                timestamp_ns=1000,
                order_id="order_1",
                side=Side.BUY,
                price=100.0,
                qty=100.0,
            )
        )

        manager.reset_statistics()

        assert manager.message_count == 0

    def test_trade_callback(self):
        """Test trade callback."""
        trades_received = []

        def on_trade(trade: Trade):
            trades_received.append(trade)

        manager = LOBStateManager(on_trade=on_trade)

        manager.apply_message(
            LOBMessage(
                msg_type=MessageType.ADD,
                timestamp_ns=1000,
                order_id="order_1",
                side=Side.BUY,
                price=100.0,
                qty=100.0,
            )
        )

        manager.apply_message(
            LOBMessage(
                msg_type=MessageType.EXECUTE,
                timestamp_ns=1001,
                order_id="order_1",
                side=Side.BUY,
                price=100.0,
                qty=0.0,
                execute_qty=50.0,
            )
        )

        assert len(trades_received) == 1
        assert trades_received[0].qty == 50.0

    def test_level_change_callback(self):
        """Test level change callback."""
        level_changes = []

        def on_level_change(side: Side, price: float, qty: float):
            level_changes.append((side, price, qty))

        manager = LOBStateManager(on_level_change=on_level_change)

        manager.apply_message(
            LOBMessage(
                msg_type=MessageType.ADD,
                timestamp_ns=1000,
                order_id="order_1",
                side=Side.BUY,
                price=100.0,
                qty=100.0,
            )
        )

        assert len(level_changes) == 1
        assert level_changes[0] == (Side.BUY, 100.0, 100.0)


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestStateManagerIntegration:
    """Integration tests for state manager."""

    def test_full_message_sequence(self):
        """Test processing full message sequence."""
        manager = LOBStateManager(symbol="AAPL")

        # Build order book
        messages = [
            LOBMessage(MessageType.ADD, 1000, "bid_1", Side.BUY, 100.0, 100.0),
            LOBMessage(MessageType.ADD, 1001, "bid_2", Side.BUY, 99.0, 200.0),
            LOBMessage(MessageType.ADD, 1002, "ask_1", Side.SELL, 101.0, 150.0),
            LOBMessage(MessageType.ADD, 1003, "ask_2", Side.SELL, 102.0, 250.0),
        ]

        for msg in messages:
            manager.apply_message(msg)

        book = manager.orderbook
        assert book.best_bid == 100.0
        assert book.best_ask == 101.0
        assert book.spread == 1.0

        # Execute against ask
        manager.apply_message(
            LOBMessage(MessageType.EXECUTE, 1004, "ask_1", Side.SELL, 101.0, 0.0, execute_qty=100.0)
        )

        assert book.best_ask == 101.0  # Still 50 left
        assert book.get_order("ask_1").remaining_qty == 50.0

        # Delete bid
        manager.apply_message(LOBMessage(MessageType.DELETE, 1005, "bid_1", Side.BUY, 100.0, 0.0))

        assert book.best_bid == 99.0

    def test_order_replace_sequence(self):
        """Test order replace (modify with new ID)."""
        manager = LOBStateManager()

        # Add original order
        manager.apply_message(LOBMessage(MessageType.ADD, 1000, "order_1", Side.BUY, 100.0, 100.0))

        # Replace with new price and ID
        msg = LOBMessage(MessageType.MODIFY, 1001, "order_1", Side.BUY, 101.0, 150.0)
        msg.new_order_id = "order_2"
        manager.apply_message(msg)

        book = manager.orderbook
        assert not book.contains_order("order_1")
        assert book.contains_order("order_2")
        assert book.get_order("order_2").price == 101.0
        assert book.get_order("order_2").remaining_qty == 150.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
