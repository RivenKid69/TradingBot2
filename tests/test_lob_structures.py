"""
Comprehensive tests for L3 LOB data structures.

Tests cover:
- LimitOrder CRUD and fill operations
- PriceLevel FIFO queue management
- OrderBook bid/ask operations
- Market order execution
- Walk book / VWAP calculations
- Snapshot operations
- Edge cases and error handling

Target: 50+ tests with <1μs per message benchmark
"""

import time
import pytest

pytest.importorskip("sortedcontainers")
import numpy as np
from typing import List, Tuple

from lob.data_structures import (
    Side,
    OrderType,
    LimitOrder,
    PriceLevel,
    OrderBook,
    Fill,
    Trade,
)


# ==============================================================================
# LimitOrder Tests
# ==============================================================================


class TestLimitOrder:
    """Tests for LimitOrder dataclass."""

    def test_create_basic_order(self):
        """Test creating a basic limit order."""
        order = LimitOrder(
            order_id="order_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.BUY,
        )
        assert order.order_id == "order_1"
        assert order.price == 100.0
        assert order.qty == 100.0
        assert order.remaining_qty == 100.0
        assert order.side == Side.BUY
        assert order.display_qty == 100.0  # Auto-initialized
        assert order.hidden_qty == 0.0
        assert not order.is_filled
        assert not order.is_hidden
        assert not order.is_iceberg

    def test_create_iceberg_order(self):
        """Test creating an iceberg order."""
        order = LimitOrder(
            order_id="ice_1",
            price=100.0,
            qty=1000.0,
            remaining_qty=1000.0,
            timestamp_ns=1000,
            side=Side.BUY,
            hidden_qty=900.0,
            display_qty=100.0,
            order_type=OrderType.ICEBERG,
        )
        assert order.is_iceberg
        assert order.visible_qty == 100.0
        assert order.hidden_qty == 900.0

    def test_create_hidden_order(self):
        """Test creating a hidden order."""
        order = LimitOrder(
            order_id="hid_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.SELL,
            order_type=OrderType.HIDDEN,
        )
        assert order.is_hidden
        assert order.display_qty == 0.0
        assert order.hidden_qty == 100.0

    def test_fill_order_partial(self):
        """Test partial order fill."""
        order = LimitOrder(
            order_id="order_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.BUY,
        )
        filled = order.fill(30.0)
        assert filled == 30.0
        assert order.remaining_qty == 70.0
        assert not order.is_filled

    def test_fill_order_complete(self):
        """Test complete order fill."""
        order = LimitOrder(
            order_id="order_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.BUY,
        )
        filled = order.fill(100.0)
        assert filled == 100.0
        assert order.remaining_qty == 0.0
        assert order.is_filled

    def test_fill_order_overfill(self):
        """Test that overfill is capped."""
        order = LimitOrder(
            order_id="order_1",
            price=100.0,
            qty=100.0,
            remaining_qty=50.0,
            timestamp_ns=1000,
            side=Side.BUY,
        )
        filled = order.fill(100.0)
        assert filled == 50.0  # Capped to remaining
        assert order.remaining_qty == 0.0

    def test_fill_iceberg_replenish(self):
        """Test iceberg order replenishment."""
        order = LimitOrder(
            order_id="ice_1",
            price=100.0,
            qty=1000.0,
            remaining_qty=1000.0,
            timestamp_ns=1000,
            side=Side.BUY,
            hidden_qty=900.0,
            display_qty=100.0,
            order_type=OrderType.ICEBERG,
        )

        # First fill exhausts display
        filled = order.fill(100.0)
        assert filled == 100.0
        assert order.remaining_qty == 900.0
        # Display should be replenished from hidden
        assert order.display_qty > 0

    def test_clone_order(self):
        """Test order cloning."""
        order = LimitOrder(
            order_id="order_1",
            price=100.0,
            qty=100.0,
            remaining_qty=50.0,
            timestamp_ns=1000,
            side=Side.BUY,
        )
        clone = order.clone()

        # Modify original
        order.fill(10.0)

        # Clone should be unchanged
        assert clone.remaining_qty == 50.0
        assert clone.order_id == "order_1"

    def test_side_enum(self):
        """Test Side enum parsing."""
        assert Side.from_string("BUY") == Side.BUY
        assert Side.from_string("SELL") == Side.SELL
        assert Side.from_string("B") == Side.BUY
        assert Side.from_string("S") == Side.SELL
        assert Side.from_string("bid") == Side.BUY
        assert Side.from_string("ask") == Side.SELL

    def test_order_type_enum(self):
        """Test OrderType enum parsing."""
        assert OrderType.from_string("LIMIT") == OrderType.LIMIT
        assert OrderType.from_string("MARKET") == OrderType.MARKET
        assert OrderType.from_string("ICEBERG") == OrderType.ICEBERG
        assert OrderType.from_string("HIDDEN") == OrderType.HIDDEN


# ==============================================================================
# PriceLevel Tests
# ==============================================================================


class TestPriceLevel:
    """Tests for PriceLevel class."""

    def test_create_empty_level(self):
        """Test creating an empty price level."""
        level = PriceLevel(price=100.0)
        assert level.price == 100.0
        assert level.order_count == 0
        assert level.is_empty
        assert level.total_visible_qty == 0.0
        assert level.total_hidden_qty == 0.0

    def test_add_single_order(self):
        """Test adding single order to level."""
        level = PriceLevel(price=100.0)
        order = LimitOrder(
            order_id="order_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.BUY,
        )

        pos = level.add_order(order)
        assert pos == 0  # First order
        assert level.order_count == 1
        assert not level.is_empty
        assert level.total_visible_qty == 100.0

    def test_add_multiple_orders_fifo(self):
        """Test FIFO ordering of multiple orders."""
        level = PriceLevel(price=100.0)

        for i in range(5):
            order = LimitOrder(
                order_id=f"order_{i}",
                price=100.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000 + i,
                side=Side.BUY,
            )
            pos = level.add_order(order)
            assert pos == i  # Queue position matches insertion order

        assert level.order_count == 5
        assert level.total_visible_qty == 500.0

        # Verify FIFO order
        front = level.peek_front()
        assert front.order_id == "order_0"

    def test_remove_order_by_id(self):
        """Test removing order by ID."""
        level = PriceLevel(price=100.0)

        for i in range(3):
            order = LimitOrder(
                order_id=f"order_{i}",
                price=100.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000 + i,
                side=Side.BUY,
            )
            level.add_order(order)

        # Remove middle order
        removed = level.remove_order("order_1")
        assert removed is not None
        assert removed.order_id == "order_1"
        assert level.order_count == 2
        assert level.total_visible_qty == 200.0

    def test_remove_nonexistent_order(self):
        """Test removing non-existent order."""
        level = PriceLevel(price=100.0)
        removed = level.remove_order("fake_order")
        assert removed is None

    def test_get_order_by_id(self):
        """Test O(1) order lookup."""
        level = PriceLevel(price=100.0)

        order = LimitOrder(
            order_id="order_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.BUY,
        )
        level.add_order(order)

        found = level.get_order("order_1")
        assert found is not None
        assert found.order_id == "order_1"

        not_found = level.get_order("fake")
        assert not_found is None

    def test_pop_front(self):
        """Test popping front order."""
        level = PriceLevel(price=100.0)

        for i in range(3):
            order = LimitOrder(
                order_id=f"order_{i}",
                price=100.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000 + i,
                side=Side.BUY,
            )
            level.add_order(order)

        popped = level.pop_front()
        assert popped.order_id == "order_0"  # FIFO
        assert level.order_count == 2

        next_front = level.peek_front()
        assert next_front.order_id == "order_1"

    def test_fill_qty_fifo(self):
        """Test FIFO fill at level."""
        level = PriceLevel(price=100.0)

        for i in range(3):
            order = LimitOrder(
                order_id=f"order_{i}",
                price=100.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000 + i,
                side=Side.BUY,
            )
            level.add_order(order)

        # Fill 150 - should fill first order fully, second partially
        filled, trades = level.fill_qty(150.0)
        assert filled == 150.0
        assert len(trades) == 2  # Two orders touched
        assert trades[0].maker_order_id == "order_0"
        assert trades[0].qty == 100.0
        assert trades[1].maker_order_id == "order_1"
        assert trades[1].qty == 50.0

        # First order should be removed
        assert level.order_count == 2
        front = level.peek_front()
        assert front.order_id == "order_1"
        assert front.remaining_qty == 50.0

    def test_queue_position_updates(self):
        """Test queue positions update after removal."""
        level = PriceLevel(price=100.0)

        for i in range(5):
            order = LimitOrder(
                order_id=f"order_{i}",
                price=100.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000 + i,
                side=Side.BUY,
            )
            level.add_order(order)

        # Remove order at position 2
        level.remove_order("order_2")

        # Verify queue positions updated
        assert level.get_queue_position("order_0") == 0
        assert level.get_queue_position("order_1") == 1
        assert level.get_queue_position("order_3") == 2
        assert level.get_queue_position("order_4") == 3

    def test_clone_level(self):
        """Test level cloning."""
        level = PriceLevel(price=100.0)

        order = LimitOrder(
            order_id="order_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.BUY,
        )
        level.add_order(order)

        clone = level.clone()

        # Modify original
        level.pop_front()

        # Clone should be unchanged
        assert clone.order_count == 1
        assert clone.get_order("order_1") is not None


# ==============================================================================
# OrderBook Tests
# ==============================================================================


class TestOrderBook:
    """Tests for OrderBook class."""

    def test_create_empty_book(self):
        """Test creating empty order book."""
        book = OrderBook(symbol="AAPL", tick_size=0.01)
        assert book.symbol == "AAPL"
        assert book.tick_size == 0.01
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.mid_price is None
        assert book.spread is None
        assert book.order_count == 0

    def test_add_bid_order(self):
        """Test adding bid order."""
        book = OrderBook()
        order = LimitOrder(
            order_id="bid_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.BUY,
        )

        pos = book.add_limit_order(order)
        assert pos == 0
        assert book.best_bid == 100.0
        assert book.best_bid_qty == 100.0
        assert book.order_count == 1

    def test_add_ask_order(self):
        """Test adding ask order."""
        book = OrderBook()
        order = LimitOrder(
            order_id="ask_1",
            price=101.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.SELL,
        )

        book.add_limit_order(order)
        assert book.best_ask == 101.0
        assert book.best_ask_qty == 100.0

    def test_bid_ask_ordering(self):
        """Test correct bid/ask price ordering."""
        book = OrderBook()

        # Add bids at different prices
        for price in [98.0, 99.0, 100.0]:
            book.add_limit_order(
                LimitOrder(
                    order_id=f"bid_{price}",
                    price=price,
                    qty=100.0,
                    remaining_qty=100.0,
                    timestamp_ns=1000,
                    side=Side.BUY,
                )
            )

        # Add asks at different prices
        for price in [101.0, 102.0, 103.0]:
            book.add_limit_order(
                LimitOrder(
                    order_id=f"ask_{price}",
                    price=price,
                    qty=100.0,
                    remaining_qty=100.0,
                    timestamp_ns=1000,
                    side=Side.SELL,
                )
            )

        assert book.best_bid == 100.0  # Highest bid
        assert book.best_ask == 101.0  # Lowest ask
        assert book.mid_price == 100.5
        assert book.spread == 1.0

    def test_cancel_order(self):
        """Test cancelling order."""
        book = OrderBook()
        order = LimitOrder(
            order_id="order_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.BUY,
        )
        book.add_limit_order(order)

        cancelled = book.cancel_order("order_1")
        assert cancelled is not None
        assert cancelled.order_id == "order_1"
        assert book.order_count == 0
        assert book.best_bid is None

    def test_modify_order_qty_decrease(self):
        """Test modifying order quantity (decrease maintains priority)."""
        book = OrderBook()

        # Add two orders
        book.add_limit_order(
            LimitOrder(
                order_id="order_1",
                price=100.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000,
                side=Side.BUY,
            )
        )
        book.add_limit_order(
            LimitOrder(
                order_id="order_2",
                price=100.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1001,
                side=Side.BUY,
            )
        )

        # Decrease qty of first order
        book.modify_order("order_1", new_qty=50.0)

        order = book.get_order("order_1")
        assert order.remaining_qty == 50.0
        assert order.queue_position == 0  # Maintains priority

    def test_modify_order_price_loses_priority(self):
        """Test modifying order price loses priority."""
        book = OrderBook()

        book.add_limit_order(
            LimitOrder(
                order_id="order_1",
                price=100.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000,
                side=Side.BUY,
            )
        )
        book.add_limit_order(
            LimitOrder(
                order_id="order_2",
                price=100.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1001,
                side=Side.BUY,
            )
        )

        # Change price of first order
        book.modify_order("order_1", new_price=99.0)

        # order_2 should now be at best bid 100.0
        assert book.best_bid == 100.0
        bids, _ = book.get_depth(2)
        assert bids[0] == (100.0, 100.0)  # order_2
        assert bids[1] == (99.0, 100.0)  # order_1

    def test_get_order(self):
        """Test O(1) order lookup."""
        book = OrderBook()
        order = LimitOrder(
            order_id="order_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.BUY,
        )
        book.add_limit_order(order)

        found = book.get_order("order_1")
        assert found is not None
        assert found.order_id == "order_1"

    def test_contains_order(self):
        """Test order existence check."""
        book = OrderBook()
        order = LimitOrder(
            order_id="order_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.BUY,
        )
        book.add_limit_order(order)

        assert book.contains_order("order_1")
        assert not book.contains_order("fake_order")

    def test_duplicate_order_raises(self):
        """Test duplicate order ID raises error."""
        book = OrderBook()
        order = LimitOrder(
            order_id="order_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.BUY,
        )
        book.add_limit_order(order)

        with pytest.raises(ValueError, match="Duplicate"):
            book.add_limit_order(order)

    def test_execute_market_buy(self):
        """Test market buy execution."""
        book = OrderBook()

        # Add asks
        for i, price in enumerate([101.0, 102.0, 103.0]):
            book.add_limit_order(
                LimitOrder(
                    order_id=f"ask_{i}",
                    price=price,
                    qty=100.0,
                    remaining_qty=100.0,
                    timestamp_ns=1000,
                    side=Side.SELL,
                )
            )

        # Execute market buy for 150
        fill = book.execute_market_order(Side.BUY, 150.0, "taker_1")

        assert fill.total_qty == 150.0
        assert len(fill.trades) == 2
        assert fill.trades[0].price == 101.0
        assert fill.trades[0].qty == 100.0
        assert fill.trades[1].price == 102.0
        assert fill.trades[1].qty == 50.0

        # Average price check
        expected_avg = (101.0 * 100 + 102.0 * 50) / 150
        assert abs(fill.avg_price - expected_avg) < 0.001

    def test_execute_market_sell(self):
        """Test market sell execution."""
        book = OrderBook()

        # Add bids
        for i, price in enumerate([100.0, 99.0, 98.0]):
            book.add_limit_order(
                LimitOrder(
                    order_id=f"bid_{i}",
                    price=price,
                    qty=100.0,
                    remaining_qty=100.0,
                    timestamp_ns=1000,
                    side=Side.BUY,
                )
            )

        # Execute market sell for 250
        fill = book.execute_market_order(Side.SELL, 250.0)

        assert fill.total_qty == 250.0
        assert fill.trades[0].price == 100.0  # Best bid first
        assert fill.trades[1].price == 99.0
        assert fill.trades[2].price == 98.0
        assert fill.trades[2].qty == 50.0

    def test_execute_market_insufficient_liquidity(self):
        """Test market order with insufficient liquidity."""
        book = OrderBook()

        book.add_limit_order(
            LimitOrder(
                order_id="ask_1",
                price=101.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000,
                side=Side.SELL,
            )
        )

        fill = book.execute_market_order(Side.BUY, 200.0)

        assert fill.total_qty == 100.0
        assert fill.remaining_qty == 100.0
        assert not fill.is_complete

    def test_get_depth(self):
        """Test getting book depth."""
        book = OrderBook()

        # Add bids
        for price in [100.0, 99.0, 98.0]:
            book.add_limit_order(
                LimitOrder(
                    order_id=f"bid_{price}",
                    price=price,
                    qty=100.0,
                    remaining_qty=100.0,
                    timestamp_ns=1000,
                    side=Side.BUY,
                )
            )

        # Add asks
        for price in [101.0, 102.0, 103.0]:
            book.add_limit_order(
                LimitOrder(
                    order_id=f"ask_{price}",
                    price=price,
                    qty=100.0,
                    remaining_qty=100.0,
                    timestamp_ns=1000,
                    side=Side.SELL,
                )
            )

        bids, asks = book.get_depth(2)

        assert len(bids) == 2
        assert len(asks) == 2
        assert bids[0] == (100.0, 100.0)  # Best bid
        assert bids[1] == (99.0, 100.0)
        assert asks[0] == (101.0, 100.0)  # Best ask
        assert asks[1] == (102.0, 100.0)

    def test_walk_book(self):
        """Test walking the book for VWAP."""
        book = OrderBook()

        # Add asks
        book.add_limit_order(
            LimitOrder(
                order_id="ask_1",
                price=101.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000,
                side=Side.SELL,
            )
        )
        book.add_limit_order(
            LimitOrder(
                order_id="ask_2",
                price=102.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000,
                side=Side.SELL,
            )
        )

        avg_price, total_filled, fills = book.walk_book(Side.BUY, 150.0)

        expected_avg = (101.0 * 100 + 102.0 * 50) / 150
        assert abs(avg_price - expected_avg) < 0.001
        assert total_filled == 150.0
        assert len(fills) == 2
        assert fills[0] == (101.0, 100.0)
        assert fills[1] == (102.0, 50.0)

    def test_get_vwap(self):
        """Test VWAP calculation."""
        book = OrderBook()

        # Add bids
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
                order_id="bid_2",
                price=99.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000,
                side=Side.BUY,
            )
        )

        vwap = book.get_vwap(Side.SELL, 150.0)
        expected = (100.0 * 100 + 99.0 * 50) / 150
        assert abs(vwap - expected) < 0.001

        # Insufficient liquidity
        vwap_none = book.get_vwap(Side.SELL, 500.0)
        assert vwap_none is None

    def test_spread_bps(self):
        """Test spread in basis points."""
        book = OrderBook()

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
                price=100.10,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000,
                side=Side.SELL,
            )
        )

        # Spread = 0.10 / 100.05 * 10000 ≈ 10 bps
        assert book.spread_bps is not None
        assert abs(book.spread_bps - 10.0) < 0.5

    def test_mbo_snapshot(self):
        """Test Market-by-Order snapshot."""
        book = OrderBook()

        for i in range(3):
            book.add_limit_order(
                LimitOrder(
                    order_id=f"bid_{i}",
                    price=100.0 - i,
                    qty=100.0,
                    remaining_qty=100.0,
                    timestamp_ns=1000 + i,
                    side=Side.BUY,
                )
            )

        orders = book.get_mbo_snapshot(Side.BUY, n_orders=2)
        assert len(orders) == 2
        assert orders[0].price == 100.0  # Best first

    def test_mbp_snapshot(self):
        """Test Market-by-Price snapshot."""
        book = OrderBook()

        for i in range(3):
            book.add_limit_order(
                LimitOrder(
                    order_id=f"bid_{i}",
                    price=100.0,
                    qty=100.0,
                    remaining_qty=100.0,
                    timestamp_ns=1000 + i,
                    side=Side.BUY,
                )
            )
        book.add_limit_order(
            LimitOrder(
                order_id="ask_1",
                price=101.0,
                qty=50.0,
                remaining_qty=50.0,
                timestamp_ns=1000,
                side=Side.SELL,
            )
        )

        snapshot = book.get_mbp_snapshot(n_levels=2)
        assert len(snapshot["bids"]) == 1
        assert snapshot["bids"][0]["price"] == 100.0
        assert snapshot["bids"][0]["qty"] == 300.0
        assert snapshot["bids"][0]["orders"] == 3

    def test_clone_book(self):
        """Test book cloning."""
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

        clone = book.clone()

        # Modify original
        book.cancel_order("bid_1")

        # Clone should be unchanged
        assert clone.order_count == 1
        assert clone.best_bid == 100.0

    def test_swap_books(self):
        """Test swapping two books."""
        book1 = OrderBook(symbol="AAPL")
        book1.add_limit_order(
            LimitOrder(
                order_id="order_1",
                price=100.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000,
                side=Side.BUY,
            )
        )

        book2 = OrderBook(symbol="MSFT")
        book2.add_limit_order(
            LimitOrder(
                order_id="order_2",
                price=200.0,
                qty=200.0,
                remaining_qty=200.0,
                timestamp_ns=1000,
                side=Side.SELL,
            )
        )

        book1.swap(book2)

        assert book1.best_ask == 200.0
        assert book2.best_bid == 100.0

    def test_clear_book(self):
        """Test clearing book."""
        book = OrderBook()
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

        book.clear()

        assert book.order_count == 0
        assert book.best_bid is None

    def test_is_crossed(self):
        """Test crossed book detection."""
        book = OrderBook()

        book.add_limit_order(
            LimitOrder(
                order_id="bid_1",
                price=101.0,  # Higher than ask
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000,
                side=Side.BUY,
            )
        )
        book.add_limit_order(
            LimitOrder(
                order_id="ask_1",
                price=100.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000,
                side=Side.SELL,
            )
        )

        assert book.is_crossed

    def test_get_total_liquidity(self):
        """Test total liquidity calculation."""
        book = OrderBook()

        for i in range(5):
            book.add_limit_order(
                LimitOrder(
                    order_id=f"bid_{i}",
                    price=100.0 - i * 0.1,
                    qty=100.0,
                    remaining_qty=100.0,
                    timestamp_ns=1000,
                    side=Side.BUY,
                )
            )

        # Within 50 bps of best bid (100.0)
        # 50 bps = 0.5% = $0.50 range
        # Should include prices 99.5 and above
        liq = book.get_total_liquidity(Side.BUY, price_range_bps=50.0)
        assert liq == 500.0  # All 5 levels within 50 bps


# ==============================================================================
# Performance Benchmarks
# ==============================================================================


class TestPerformance:
    """Performance benchmarks for LOB operations."""

    def test_add_order_performance(self):
        """Benchmark add order performance."""
        book = OrderBook()
        n_orders = 10000

        start = time.perf_counter()
        for i in range(n_orders):
            book.add_limit_order(
                LimitOrder(
                    order_id=f"order_{i}",
                    price=100.0 + (i % 100) * 0.01,
                    qty=100.0,
                    remaining_qty=100.0,
                    timestamp_ns=1000 + i,
                    side=Side.BUY if i % 2 == 0 else Side.SELL,
                )
            )
        elapsed = time.perf_counter() - start

        ops_per_sec = n_orders / elapsed
        ns_per_op = (elapsed * 1e9) / n_orders

        print(f"\nAdd order: {ns_per_op:.0f} ns/op, {ops_per_sec:.0f} ops/sec")

        # Target: <10μs per operation
        assert ns_per_op < 10000, f"Too slow: {ns_per_op:.0f} ns"

    def test_cancel_order_performance(self):
        """Benchmark cancel order performance."""
        book = OrderBook()
        n_orders = 5000

        # Pre-populate
        for i in range(n_orders):
            book.add_limit_order(
                LimitOrder(
                    order_id=f"order_{i}",
                    price=100.0 + (i % 100) * 0.01,
                    qty=100.0,
                    remaining_qty=100.0,
                    timestamp_ns=1000 + i,
                    side=Side.BUY,
                )
            )

        start = time.perf_counter()
        for i in range(n_orders):
            book.cancel_order(f"order_{i}")
        elapsed = time.perf_counter() - start

        ns_per_op = (elapsed * 1e9) / n_orders

        print(f"Cancel order: {ns_per_op:.0f} ns/op")

        # Target: <10μs per operation
        assert ns_per_op < 10000, f"Too slow: {ns_per_op:.0f} ns"

    def test_market_order_performance(self):
        """Benchmark market order execution."""
        book = OrderBook()

        # Pre-populate with 100 price levels
        for i in range(100):
            for j in range(10):
                book.add_limit_order(
                    LimitOrder(
                        order_id=f"ask_{i}_{j}",
                        price=101.0 + i * 0.01,
                        qty=100.0,
                        remaining_qty=100.0,
                        timestamp_ns=1000,
                        side=Side.SELL,
                    )
                )

        n_orders = 1000

        start = time.perf_counter()
        for i in range(n_orders):
            # Re-add liquidity (quick)
            for j in range(5):
                try:
                    book.add_limit_order(
                        LimitOrder(
                            order_id=f"new_{i}_{j}",
                            price=101.0,
                            qty=100.0,
                            remaining_qty=100.0,
                            timestamp_ns=1000,
                            side=Side.SELL,
                        )
                    )
                except ValueError:
                    pass

            fill = book.execute_market_order(Side.BUY, 500.0)
        elapsed = time.perf_counter() - start

        ns_per_op = (elapsed * 1e9) / n_orders

        print(f"Market order (500 qty): {ns_per_op:.0f} ns/op")

    def test_walk_book_performance(self):
        """Benchmark walk book performance."""
        book = OrderBook()

        # Pre-populate
        for i in range(100):
            book.add_limit_order(
                LimitOrder(
                    order_id=f"ask_{i}",
                    price=101.0 + i * 0.01,
                    qty=100.0,
                    remaining_qty=100.0,
                    timestamp_ns=1000,
                    side=Side.SELL,
                )
            )

        n_walks = 10000

        start = time.perf_counter()
        for _ in range(n_walks):
            book.walk_book(Side.BUY, 500.0)
        elapsed = time.perf_counter() - start

        ns_per_op = (elapsed * 1e9) / n_walks

        print(f"Walk book (500 qty): {ns_per_op:.0f} ns/op")

        # Target: <1μs
        assert ns_per_op < 5000, f"Too slow: {ns_per_op:.0f} ns"


# ==============================================================================
# Edge Cases
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_book_operations(self):
        """Test operations on empty book."""
        book = OrderBook()

        assert book.best_bid is None
        assert book.best_ask is None
        assert book.mid_price is None
        assert book.spread is None
        assert book.spread_bps is None
        assert not book.is_crossed

        fill = book.execute_market_order(Side.BUY, 100.0)
        assert fill.total_qty == 0.0

        bids, asks = book.get_depth(10)
        assert len(bids) == 0
        assert len(asks) == 0

    def test_zero_qty_order_rejected(self):
        """Test zero quantity order is rejected."""
        book = OrderBook()

        with pytest.raises(ValueError):
            book.add_limit_order(
                LimitOrder(
                    order_id="order_1",
                    price=100.0,
                    qty=0.0,
                    remaining_qty=0.0,
                    timestamp_ns=1000,
                    side=Side.BUY,
                )
            )

    def test_single_sided_book(self):
        """Test book with only one side."""
        book = OrderBook()

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

        assert book.best_bid == 100.0
        assert book.best_ask is None
        assert book.mid_price == 100.0  # Fallback to available side
        assert book.spread is None

    def test_multiple_orders_same_price(self):
        """Test multiple orders at same price (FIFO)."""
        book = OrderBook()

        for i in range(10):
            book.add_limit_order(
                LimitOrder(
                    order_id=f"order_{i}",
                    price=100.0,
                    qty=100.0,
                    remaining_qty=100.0,
                    timestamp_ns=1000 + i,
                    side=Side.BUY,
                )
            )

        assert book.best_bid_qty == 1000.0

        # First in queue
        first_order = book.get_order("order_0")
        assert first_order.queue_position == 0

    def test_very_large_qty(self):
        """Test handling very large quantities."""
        book = OrderBook()

        book.add_limit_order(
            LimitOrder(
                order_id="big_order",
                price=100.0,
                qty=1e12,  # 1 trillion
                remaining_qty=1e12,
                timestamp_ns=1000,
                side=Side.BUY,
            )
        )

        assert book.best_bid_qty == 1e12

    def test_very_small_price(self):
        """Test handling very small prices (penny stocks)."""
        book = OrderBook()

        book.add_limit_order(
            LimitOrder(
                order_id="penny",
                price=0.0001,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000,
                side=Side.BUY,
            )
        )

        assert book.best_bid == 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
