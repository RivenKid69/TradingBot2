"""
Comprehensive tests for L3 LOB Matching Engine (Stage 2).

Tests cover:
- MatchingEngine FIFO matching
- Market/Limit order execution
- Self-Trade Prevention (STP)
- Partial fill handling
- QueuePositionTracker MBP/MBO estimation
- Fill probability estimation
- OrderManager lifecycle
- Performance benchmarks

Target: 40+ tests with <10us per match operation
"""

import time
import math
import pytest
pytest.importorskip("sortedcontainers")
from typing import List, Optional

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
    ProRataMatchingEngine,
    STPAction,
    STPResult,
    create_matching_engine,
)
from lob.queue_tracker import (
    FillProbability,
    LevelStatistics,
    PositionEstimationMethod,
    QueuePositionTracker,
    QueueState,
    create_queue_tracker,
)
from lob.order_manager import (
    ManagedOrder,
    OrderEvent,
    OrderEventType,
    OrderLifecycleState,
    OrderManager,
    TimeInForce,
    create_order_manager,
)


# ==============================================================================
# MatchingEngine Tests
# ==============================================================================


class TestMatchingEngine:
    """Tests for FIFO Matching Engine."""

    def test_create_engine(self):
        """Test creating matching engine."""
        engine = MatchingEngine()
        assert engine is not None
        assert engine.match_count == 0
        assert engine.trade_count == 0

    def test_create_engine_with_stp(self):
        """Test creating engine with STP options."""
        engine = MatchingEngine(
            stp_action=STPAction.CANCEL_OLDEST,
            enable_stp=True,
        )
        assert engine._stp_action == STPAction.CANCEL_OLDEST
        assert engine._enable_stp is True

    def test_match_market_buy_empty_book(self):
        """Test market buy against empty book."""
        engine = MatchingEngine()
        book = OrderBook()

        result = engine.match_market_order(Side.BUY, 100.0, book)

        assert result.total_filled_qty == 0.0
        assert len(result.fills) == 0
        assert not result.is_complete

    def test_match_market_buy_single_level(self):
        """Test market buy against single ask level."""
        engine = MatchingEngine()
        book = OrderBook()

        # Add asks
        book.add_limit_order(LimitOrder(
            order_id="ask_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.SELL,
        ))

        result = engine.match_market_order(Side.BUY, 50.0, book)

        assert result.total_filled_qty == 50.0
        assert result.is_complete
        assert result.avg_fill_price == 100.0
        assert len(result.fills) == 1

    def test_match_market_buy_multiple_levels(self):
        """Test market buy walking through multiple levels."""
        engine = MatchingEngine()
        book = OrderBook()

        # Add asks at different prices
        for i, price in enumerate([101.0, 102.0, 103.0]):
            book.add_limit_order(LimitOrder(
                order_id=f"ask_{i}",
                price=price,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000 + i,
                side=Side.SELL,
            ))

        result = engine.match_market_order(Side.BUY, 250.0, book)

        assert result.total_filled_qty == 250.0
        assert result.is_complete
        # VWAP = (101*100 + 102*100 + 103*50) / 250
        expected_vwap = (101.0 * 100 + 102.0 * 100 + 103.0 * 50) / 250.0
        assert abs(result.avg_fill_price - expected_vwap) < 0.001

    def test_match_market_sell(self):
        """Test market sell against bids."""
        engine = MatchingEngine()
        book = OrderBook()

        # Add bids
        book.add_limit_order(LimitOrder(
            order_id="bid_1",
            price=99.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.BUY,
        ))

        result = engine.match_market_order(Side.SELL, 50.0, book)

        assert result.total_filled_qty == 50.0
        assert result.avg_fill_price == 99.0

    def test_match_market_insufficient_liquidity(self):
        """Test market order with insufficient liquidity."""
        engine = MatchingEngine()
        book = OrderBook()

        book.add_limit_order(LimitOrder(
            order_id="ask_1",
            price=100.0,
            qty=50.0,
            remaining_qty=50.0,
            timestamp_ns=1000,
            side=Side.SELL,
        ))

        result = engine.match_market_order(Side.BUY, 100.0, book)

        assert result.total_filled_qty == 50.0
        assert not result.is_complete

    def test_match_limit_passive(self):
        """Test passive limit order (no cross)."""
        engine = MatchingEngine()
        book = OrderBook()

        # Add ask
        book.add_limit_order(LimitOrder(
            order_id="ask_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.SELL,
        ))

        # Submit bid below ask - passive
        bid = LimitOrder(
            order_id="bid_1",
            price=99.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=2000,
            side=Side.BUY,
        )

        result = engine.match_limit_order(bid, book)

        assert result.total_filled_qty == 0.0
        assert result.resting_order is not None
        assert result.match_type == MatchType.MAKER

    def test_match_limit_aggressive(self):
        """Test aggressive limit order (crosses spread)."""
        engine = MatchingEngine()
        book = OrderBook()

        # Add ask
        book.add_limit_order(LimitOrder(
            order_id="ask_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.SELL,
        ))

        # Submit bid at/above ask - aggressive
        bid = LimitOrder(
            order_id="bid_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=2000,
            side=Side.BUY,
        )

        result = engine.match_limit_order(bid, book)

        assert result.total_filled_qty == 100.0
        assert result.is_complete
        assert result.resting_order is None

    def test_match_limit_partial_fill_and_rest(self):
        """Test limit order partial fill with resting portion."""
        engine = MatchingEngine()
        book = OrderBook()

        # Add small ask
        book.add_limit_order(LimitOrder(
            order_id="ask_1",
            price=100.0,
            qty=50.0,
            remaining_qty=50.0,
            timestamp_ns=1000,
            side=Side.SELL,
        ))

        # Submit larger bid
        bid = LimitOrder(
            order_id="bid_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=2000,
            side=Side.BUY,
        )

        result = engine.match_limit_order(bid, book)

        assert result.total_filled_qty == 50.0
        assert not result.is_complete
        assert result.resting_order is not None
        assert result.resting_order.remaining_qty == 50.0

    def test_fifo_order_priority(self):
        """Test FIFO priority - earlier orders fill first."""
        engine = MatchingEngine()
        book = OrderBook()

        # Add multiple orders at same price
        for i in range(5):
            book.add_limit_order(LimitOrder(
                order_id=f"ask_{i}",
                price=100.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000 + i,
                side=Side.SELL,
            ))

        result = engine.match_market_order(Side.BUY, 150.0, book)

        # Check FIFO: ask_0 fully filled, ask_1 partially filled
        trades = result.fills[0].trades
        assert trades[0].maker_order_id == "ask_0"
        assert trades[0].qty == 100.0
        assert trades[1].maker_order_id == "ask_1"
        assert trades[1].qty == 50.0

    def test_stp_cancel_newest(self):
        """Test STP with CANCEL_NEWEST action."""
        engine = MatchingEngine(
            stp_action=STPAction.CANCEL_NEWEST,
            enable_stp=True,
        )
        book = OrderBook()

        # Add resting order from participant "FIRM_A"
        book.add_limit_order(LimitOrder(
            order_id="ask_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.SELL,
            participant_id="FIRM_A",
        ))

        # Same participant tries to buy - should trigger STP
        result = engine.match_market_order(
            Side.BUY, 100.0, book,
            taker_order_id="buy_1",
            taker_participant_id="FIRM_A",
        )

        # Incoming order should be cancelled (no fill)
        assert result.total_filled_qty == 0.0
        assert engine.stp_count == 1

    def test_stp_cancel_oldest(self):
        """Test STP with CANCEL_OLDEST action."""
        engine = MatchingEngine(
            stp_action=STPAction.CANCEL_OLDEST,
            enable_stp=True,
        )
        book = OrderBook()

        # Add resting orders from same participant
        book.add_limit_order(LimitOrder(
            order_id="ask_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.SELL,
            participant_id="FIRM_A",
        ))
        book.add_limit_order(LimitOrder(
            order_id="ask_2",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1001,
            side=Side.SELL,
            participant_id="FIRM_B",  # Different participant
        ))

        result = engine.match_market_order(
            Side.BUY, 200.0, book,
            taker_order_id="buy_1",
            taker_participant_id="FIRM_A",
        )

        # ask_1 should be cancelled (STP), ask_2 should be filled
        assert len(result.cancelled_orders) >= 1
        assert result.total_filled_qty == 100.0  # Only ask_2

    def test_stp_different_participants_no_trigger(self):
        """Test that STP doesn't trigger for different participants."""
        engine = MatchingEngine(
            stp_action=STPAction.CANCEL_NEWEST,
            enable_stp=True,
        )
        book = OrderBook()

        book.add_limit_order(LimitOrder(
            order_id="ask_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.SELL,
            participant_id="FIRM_A",
        ))

        result = engine.match_market_order(
            Side.BUY, 100.0, book,
            taker_order_id="buy_1",
            taker_participant_id="FIRM_B",  # Different
        )

        assert result.total_filled_qty == 100.0
        assert engine.stp_count == 0

    def test_trade_callback(self):
        """Test trade callback is invoked."""
        trades_received: List[Trade] = []

        def on_trade(trade: Trade):
            trades_received.append(trade)

        engine = MatchingEngine(on_trade=on_trade)
        book = OrderBook()

        book.add_limit_order(LimitOrder(
            order_id="ask_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.SELL,
        ))

        engine.match_market_order(Side.BUY, 50.0, book)

        assert len(trades_received) == 1
        assert trades_received[0].qty == 50.0

    def test_simulate_market_order(self):
        """Test market order simulation (without execution)."""
        engine = MatchingEngine()
        book = OrderBook()

        for i, price in enumerate([100.0, 101.0, 102.0]):
            book.add_limit_order(LimitOrder(
                order_id=f"ask_{i}",
                price=price,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000,
                side=Side.SELL,
            ))

        # Simulate without executing
        avg_price, total_filled, fills = engine.simulate_market_order(
            Side.BUY, 200.0, book
        )

        assert total_filled == 200.0
        # Book should still have all orders
        assert book.order_count == 3

    def test_estimate_market_impact(self):
        """Test market impact estimation."""
        engine = MatchingEngine()
        book = OrderBook()

        # Add symmetric book
        book.add_limit_order(LimitOrder(
            order_id="bid_1",
            price=99.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.BUY,
        ))
        book.add_limit_order(LimitOrder(
            order_id="ask_1",
            price=101.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.SELL,
        ))

        impact = engine.estimate_market_impact(Side.BUY, 100.0, book)

        assert impact is not None
        assert impact > 0  # Some positive impact
        # Impact should be (101 - 100) / 100 * 10000 = 100 bps
        assert abs(impact - 100.0) < 10.0

    def test_statistics_tracking(self):
        """Test statistics are properly tracked."""
        engine = MatchingEngine()
        book = OrderBook()

        book.add_limit_order(LimitOrder(
            order_id="ask_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.SELL,
        ))

        engine.match_market_order(Side.BUY, 100.0, book)

        assert engine.match_count == 1
        assert engine.trade_count == 1

        engine.reset_statistics()
        assert engine.match_count == 0

    def test_factory_function(self):
        """Test factory function for creating engines."""
        fifo = create_matching_engine("fifo")
        assert isinstance(fifo, MatchingEngine)

        pro_rata = create_matching_engine("pro_rata")
        assert isinstance(pro_rata, ProRataMatchingEngine)

        with pytest.raises(ValueError):
            create_matching_engine("unknown")


class TestProRataMatchingEngine:
    """Tests for Pro-Rata Matching Engine."""

    def test_pro_rata_allocation(self):
        """Test pro-rata allocation at single level."""
        engine = ProRataMatchingEngine(min_allocation=1.0)
        book = OrderBook()

        # Add two orders at same price
        book.add_limit_order(LimitOrder(
            order_id="ask_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.SELL,
        ))
        book.add_limit_order(LimitOrder(
            order_id="ask_2",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1001,
            side=Side.SELL,
        ))

        result = engine.match_market_order(Side.BUY, 100.0, book)

        # Each should get 50% pro-rata
        assert result.total_filled_qty == 100.0
        trades = result.fills[0].trades
        # Both orders should receive fills
        order_fills = {t.maker_order_id: t.qty for t in trades}
        assert "ask_1" in order_fills
        assert "ask_2" in order_fills


# ==============================================================================
# QueuePositionTracker Tests
# ==============================================================================


class TestQueuePositionTracker:
    """Tests for Queue Position Tracker."""

    def test_create_tracker(self):
        """Test creating queue tracker."""
        tracker = QueuePositionTracker()
        assert tracker is not None
        assert tracker.tracked_count == 0

    def test_add_order_mbp(self):
        """Test adding order with MBP estimation."""
        tracker = QueuePositionTracker()

        order = LimitOrder(
            order_id="order_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.BUY,
        )

        state = tracker.add_order(order, level_qty_before=500.0)

        assert state.order_id == "order_1"
        assert state.qty_ahead == 500.0
        assert state.method == PositionEstimationMethod.MBP_PESSIMISTIC

    def test_add_order_mbo(self):
        """Test adding order with MBO estimation (exact)."""
        tracker = QueuePositionTracker()

        order = LimitOrder(
            order_id="order_3",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=3000,
            side=Side.BUY,
        )

        orders_ahead = [
            LimitOrder(
                order_id=f"order_{i}",
                price=100.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000 + i,
                side=Side.BUY,
            )
            for i in range(2)
        ]

        state = tracker.add_order(order, level_qty_before=200.0, orders_ahead=orders_ahead)

        assert state.estimated_position == 2
        assert state.qty_ahead == 200.0
        assert state.method == PositionEstimationMethod.MBO

    def test_remove_order(self):
        """Test removing tracked order."""
        tracker = QueuePositionTracker()

        order = LimitOrder(
            order_id="order_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.BUY,
        )

        tracker.add_order(order, level_qty_before=500.0)
        assert tracker.tracked_count == 1

        state = tracker.remove_order("order_1")
        assert state is not None
        assert tracker.tracked_count == 0

    def test_update_on_execution(self):
        """Test position update on execution."""
        tracker = QueuePositionTracker()

        order = LimitOrder(
            order_id="order_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.BUY,
        )

        tracker.add_order(order, level_qty_before=500.0)

        # Execute 200 at our price
        updated = tracker.update_on_execution(
            executed_qty=200.0,
            at_price=100.0,
        )

        assert "order_1" in updated
        state = tracker.get_state("order_1")
        assert state.qty_ahead == 300.0  # 500 - 200

    def test_update_on_execution_different_price(self):
        """Test that execution at different price doesn't update."""
        tracker = QueuePositionTracker()

        order = LimitOrder(
            order_id="order_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.BUY,
        )

        tracker.add_order(order, level_qty_before=500.0)

        # Execute at different price
        updated = tracker.update_on_execution(
            executed_qty=200.0,
            at_price=99.0,
        )

        assert "order_1" not in updated
        state = tracker.get_state("order_1")
        assert state.qty_ahead == 500.0  # Unchanged

    def test_update_on_cancel_ahead(self):
        """Test probabilistic update on cancellation."""
        tracker = QueuePositionTracker()

        order = LimitOrder(
            order_id="order_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.BUY,
        )

        tracker.add_order(order, level_qty_before=500.0)

        # Cancel 100 with 50% probability ahead
        tracker.update_on_cancel_ahead(
            cancelled_qty=100.0,
            at_price=100.0,
            at_side=Side.BUY,
            probability=0.5,
        )

        state = tracker.get_state("order_1")
        assert state.qty_ahead == 450.0  # 500 - 100*0.5

    def test_fill_probability_front_of_queue(self):
        """Test fill probability at front of queue."""
        tracker = QueuePositionTracker()

        order = LimitOrder(
            order_id="order_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.BUY,
        )

        # No one ahead
        tracker.add_order(order, level_qty_before=0.0)

        prob = tracker.estimate_fill_probability(
            "order_1",
            volume_per_second=100.0,
            time_horizon_sec=60.0,
        )

        assert prob.prob_fill == 0.95  # High probability at front

    def test_fill_probability_back_of_queue(self):
        """Test fill probability at back of queue."""
        tracker = QueuePositionTracker()

        order = LimitOrder(
            order_id="order_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.BUY,
        )

        # Large queue ahead
        tracker.add_order(order, level_qty_before=10000.0)

        prob = tracker.estimate_fill_probability(
            "order_1",
            volume_per_second=10.0,  # Low volume
            time_horizon_sec=60.0,
        )

        assert prob.prob_fill < 0.1  # Low probability

    def test_factory_function(self):
        """Test factory function for queue tracker."""
        tracker = create_queue_tracker("mbo")
        assert tracker._default_method == PositionEstimationMethod.MBO

        tracker = create_queue_tracker("mbp_pessimistic")
        assert tracker._default_method == PositionEstimationMethod.MBP_PESSIMISTIC

    def test_estimate_position_mbp(self):
        """Test MBP position estimation."""
        tracker = QueuePositionTracker()

        order = LimitOrder(
            order_id="order_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.BUY,
        )

        position = tracker.estimate_position_mbp(
            order,
            level_qty_before=500.0,
            level_qty_after=600.0,
        )

        # Position = qty_before / avg_order_size (100)
        assert position == 5

    def test_get_all_states(self):
        """Test getting all tracked states."""
        tracker = QueuePositionTracker()

        for i in range(3):
            order = LimitOrder(
                order_id=f"order_{i}",
                price=100.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000 + i,
                side=Side.BUY,
            )
            tracker.add_order(order, level_qty_before=i * 100.0)

        states = tracker.get_all_states()
        assert len(states) == 3


# ==============================================================================
# OrderManager Tests
# ==============================================================================


class TestOrderManager:
    """Tests for Order Lifecycle Manager."""

    def test_create_manager(self):
        """Test creating order manager."""
        manager = OrderManager(symbol="AAPL")
        assert manager._symbol == "AAPL"
        assert manager.order_book is not None
        assert manager.matching_engine is not None

    def test_submit_limit_order_passive(self):
        """Test submitting passive limit order."""
        manager = OrderManager()

        # Submit bid below any asks
        result = manager.submit_order(
            side=Side.BUY,
            price=99.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
        )

        assert result.state == OrderLifecycleState.NEW
        assert result.filled_qty == 0.0
        assert manager.get_best_bid() == 99.0

    def test_submit_limit_order_aggressive(self):
        """Test submitting aggressive limit order."""
        manager = OrderManager()

        # First add ask
        manager.submit_order(
            side=Side.SELL,
            price=100.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
        )

        # Submit bid at ask price - aggressive
        result = manager.submit_order(
            side=Side.BUY,
            price=100.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
        )

        assert result.state == OrderLifecycleState.FILLED
        assert result.filled_qty == 100.0
        assert result.avg_fill_price == 100.0

    def test_submit_market_order(self):
        """Test submitting market order."""
        manager = OrderManager()

        # Add liquidity
        manager.submit_order(
            side=Side.SELL,
            price=100.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
        )

        # Submit market buy
        result = manager.submit_order(
            side=Side.BUY,
            price=0.0,  # Ignored for market
            qty=50.0,
            order_type=OrderType.MARKET,
        )

        assert result.state == OrderLifecycleState.FILLED
        assert result.filled_qty == 50.0

    def test_cancel_order(self):
        """Test cancelling order."""
        manager = OrderManager()

        order = manager.submit_order(
            side=Side.BUY,
            price=99.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
        )

        success = manager.cancel_order(order.order.order_id)

        assert success
        assert order.state == OrderLifecycleState.CANCELLED
        assert manager.get_best_bid() is None

    def test_cancel_nonexistent_order(self):
        """Test cancelling non-existent order."""
        manager = OrderManager()

        success = manager.cancel_order("fake_order")
        assert not success

    def test_modify_order_qty_decrease(self):
        """Test modifying order quantity decrease."""
        manager = OrderManager()

        order = manager.submit_order(
            side=Side.BUY,
            price=99.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
        )

        modified = manager.modify_order(
            order.order.order_id,
            new_qty=50.0,
        )

        assert modified is not None
        assert modified.order.remaining_qty == 50.0

    def test_ioc_time_in_force(self):
        """Test Immediate-Or-Cancel orders."""
        manager = OrderManager()

        # Add partial liquidity
        manager.submit_order(
            side=Side.SELL,
            price=100.0,
            qty=50.0,
            order_type=OrderType.LIMIT,
        )

        # IOC for 100 - should fill 50, cancel rest
        result = manager.submit_order(
            side=Side.BUY,
            price=100.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.IOC,
        )

        assert result.filled_qty == 50.0
        assert result.cancelled_qty == 50.0
        # Per FIX protocol: IOC with partial fill is CANCELLED (OrdStatus=4)
        assert result.state == OrderLifecycleState.CANCELLED

    def test_fok_time_in_force_success(self):
        """Test Fill-Or-Kill success case."""
        manager = OrderManager()

        # Add enough liquidity
        manager.submit_order(
            side=Side.SELL,
            price=100.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
        )

        # FOK for 100 - should fill completely
        result = manager.submit_order(
            side=Side.BUY,
            price=100.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.FOK,
        )

        assert result.state == OrderLifecycleState.FILLED
        assert result.filled_qty == 100.0

    def test_fok_time_in_force_failure(self):
        """Test Fill-Or-Kill failure case."""
        manager = OrderManager()

        # Add partial liquidity
        manager.submit_order(
            side=Side.SELL,
            price=100.0,
            qty=50.0,
            order_type=OrderType.LIMIT,
        )

        # FOK for 100 - should cancel entirely
        result = manager.submit_order(
            side=Side.BUY,
            price=100.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.FOK,
        )

        assert result.state == OrderLifecycleState.CANCELLED
        assert result.filled_qty == 0.0

    def test_order_callbacks(self):
        """Test fill and cancel callbacks."""
        fills_received: List[Fill] = []
        cancels_received: List[ManagedOrder] = []

        def on_fill(order: ManagedOrder, fill: Fill):
            fills_received.append(fill)

        def on_cancel(order: ManagedOrder):
            cancels_received.append(order)

        manager = OrderManager(
            on_fill=on_fill,
            on_cancel=on_cancel,
        )

        # Add liquidity
        ask = manager.submit_order(
            side=Side.SELL,
            price=100.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
        )

        # Execute fill
        manager.submit_order(
            side=Side.BUY,
            price=100.0,
            qty=50.0,
            order_type=OrderType.LIMIT,
        )

        # Both maker (ask) and taker (buy) receive fill callbacks
        assert len(fills_received) == 2

        # Verify both fills are present
        maker_fill = [f for f in fills_received if f.order_id == ask.order.order_id]
        taker_fill = [f for f in fills_received if f.order_id != ask.order.order_id]
        assert len(maker_fill) == 1
        assert len(taker_fill) == 1
        assert maker_fill[0].total_qty == 50.0
        assert taker_fill[0].total_qty == 50.0

        # Cancel remaining
        manager.cancel_order(ask.order.order_id)
        assert len(cancels_received) == 1

    def test_event_callback(self):
        """Test event callback."""
        events: List[OrderEvent] = []

        def on_event(event: OrderEvent):
            events.append(event)

        manager = OrderManager(on_event=on_event)

        manager.submit_order(
            side=Side.BUY,
            price=99.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
        )

        assert len(events) >= 1
        assert events[0].event_type == OrderEventType.ACCEPTED

    def test_get_queue_state(self):
        """Test getting queue state for order."""
        manager = OrderManager()

        order = manager.submit_order(
            side=Side.BUY,
            price=99.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
        )

        state = manager.get_queue_state(order.order.order_id)

        assert state is not None
        assert state.price == 99.0

    def test_get_fill_probability(self):
        """Test fill probability estimation."""
        manager = OrderManager()

        order = manager.submit_order(
            side=Side.BUY,
            price=99.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
        )

        prob = manager.get_fill_probability(
            order.order.order_id,
            volume_per_second=100.0,
        )

        assert prob.prob_fill > 0.0

    def test_get_active_orders(self):
        """Test getting active orders."""
        manager = OrderManager()

        for i in range(3):
            manager.submit_order(
                side=Side.BUY,
                price=99.0 - i,
                qty=100.0,
                order_type=OrderType.LIMIT,
            )

        active = manager.get_active_orders()
        assert len(active) == 3

        active_bids = manager.get_active_orders(side=Side.BUY)
        assert len(active_bids) == 3

    def test_cancel_all_orders(self):
        """Test cancelling all orders."""
        manager = OrderManager()

        for i in range(3):
            manager.submit_order(
                side=Side.BUY,
                price=99.0 - i,
                qty=100.0,
                order_type=OrderType.LIMIT,
            )

        cancelled = manager.cancel_all_orders()
        assert cancelled == 3
        assert len(manager.get_active_orders()) == 0

    def test_statistics(self):
        """Test order statistics."""
        manager = OrderManager()

        # Submit and fill some orders
        manager.submit_order(
            side=Side.SELL,
            price=100.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
        )
        manager.submit_order(
            side=Side.BUY,
            price=100.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
        )

        stats = manager.get_statistics()

        assert stats.total_orders >= 2
        assert stats.total_volume_filled > 0

    def test_factory_function(self):
        """Test factory function."""
        manager = create_order_manager(symbol="MSFT")
        assert manager._symbol == "MSFT"


# ==============================================================================
# Performance Benchmarks
# ==============================================================================


class TestPerformanceBenchmarks:
    """Performance benchmarks for matching operations."""

    def test_market_order_performance(self):
        """Benchmark market order matching."""
        engine = MatchingEngine()
        book = OrderBook()

        # Pre-populate with 100 price levels
        for i in range(100):
            book.add_limit_order(LimitOrder(
                order_id=f"ask_{i}",
                price=100.0 + i * 0.01,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000,
                side=Side.SELL,
            ))

        n_orders = 1000

        start = time.perf_counter()
        for i in range(n_orders):
            # Simulate small order
            engine.simulate_market_order(Side.BUY, 100.0, book)
        elapsed = time.perf_counter() - start

        ns_per_op = (elapsed * 1e9) / n_orders

        print(f"\nMarket order simulation: {ns_per_op:.0f} ns/op")

        # Target: <10us per match
        assert ns_per_op < 10000, f"Too slow: {ns_per_op:.0f} ns"

    def test_limit_order_matching_performance(self):
        """Benchmark limit order matching."""
        engine = MatchingEngine()

        n_orders = 1000
        total_time = 0.0

        for i in range(n_orders):
            book = OrderBook()

            # Add some asks
            for j in range(10):
                book.add_limit_order(LimitOrder(
                    order_id=f"ask_{j}",
                    price=100.0 + j * 0.01,
                    qty=100.0,
                    remaining_qty=100.0,
                    timestamp_ns=1000,
                    side=Side.SELL,
                ))

            bid = LimitOrder(
                order_id=f"bid_{i}",
                price=100.05,  # Cross 6 levels
                qty=500.0,
                remaining_qty=500.0,
                timestamp_ns=2000,
                side=Side.BUY,
            )

            start = time.perf_counter()
            engine.match_limit_order(bid, book)
            total_time += time.perf_counter() - start

        ns_per_op = (total_time * 1e9) / n_orders

        print(f"Limit order matching: {ns_per_op:.0f} ns/op")

        # Target: <50us per match (relaxed for Python implementation)
        # Production C++/Cython would be <10us
        assert ns_per_op < 50000, f"Too slow: {ns_per_op:.0f} ns"

    def test_queue_position_update_performance(self):
        """Benchmark queue position updates."""
        tracker = QueuePositionTracker()

        # Track many orders
        for i in range(100):
            order = LimitOrder(
                order_id=f"order_{i}",
                price=100.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000 + i,
                side=Side.BUY,
            )
            tracker.add_order(order, level_qty_before=i * 100.0)

        n_updates = 10000

        start = time.perf_counter()
        for i in range(n_updates):
            tracker.update_on_execution(10.0, 100.0)
        elapsed = time.perf_counter() - start

        ns_per_op = (elapsed * 1e9) / n_updates

        print(f"Queue position update: {ns_per_op:.0f} ns/op")

        # Target: <500us for pure Python (optimize with Cython later)
        assert ns_per_op < 500000, f"Too slow: {ns_per_op:.0f} ns"


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests combining all components."""

    def test_full_order_lifecycle(self):
        """Test complete order lifecycle."""
        manager = OrderManager(symbol="AAPL")

        # 1. Submit passive order
        order1 = manager.submit_order(
            side=Side.BUY,
            price=99.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
            client_order_id="client_1",
        )

        assert order1.state == OrderLifecycleState.NEW
        assert manager.get_best_bid() == 99.0

        # 2. Check queue position
        state = manager.get_queue_state(order1.order.order_id)
        assert state is not None

        # 3. Submit another order at same price
        order2 = manager.submit_order(
            side=Side.BUY,
            price=99.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
        )

        # 4. Submit sell that fills both
        sell = manager.submit_order(
            side=Side.SELL,
            price=99.0,
            qty=200.0,
            order_type=OrderType.LIMIT,
        )

        assert sell.filled_qty == 200.0
        assert order1.state == OrderLifecycleState.FILLED
        assert order2.state == OrderLifecycleState.FILLED

    def test_market_making_scenario(self):
        """Test market making with two-sided quotes."""
        manager = OrderManager(symbol="AAPL")

        # Post two-sided quotes
        bid = manager.submit_order(
            side=Side.BUY,
            price=99.95,
            qty=1000.0,
            order_type=OrderType.LIMIT,
        )

        ask = manager.submit_order(
            side=Side.SELL,
            price=100.05,
            qty=1000.0,
            order_type=OrderType.LIMIT,
        )

        assert manager.get_spread_bps() is not None
        assert manager.get_spread_bps() == pytest.approx(10.0, rel=0.1)

        # Aggressive buyer lifts ask
        buyer = manager.submit_order(
            side=Side.BUY,
            price=100.05,
            qty=500.0,
            order_type=OrderType.LIMIT,
        )

        assert buyer.filled_qty == 500.0
        assert ask.filled_qty == 500.0
        assert ask.state == OrderLifecycleState.PARTIALLY_FILLED

    def test_crypto_path_not_affected(self):
        """Verify crypto code paths are not affected."""
        # This test ensures the LOB module doesn't interfere with crypto
        # Crypto uses fast_lob.pyx, not this Python LOB

        # Just verify imports work and types are compatible
        from lob import Side, OrderBook, LimitOrder

        book = OrderBook(symbol="BTC-USDT")
        order = LimitOrder(
            order_id="btc_1",
            price=50000.0,
            qty=0.1,
            remaining_qty=0.1,
            timestamp_ns=1000,
            side=Side.BUY,
        )
        book.add_limit_order(order)

        assert book.best_bid == 50000.0
        assert book.symbol == "BTC-USDT"


# ==============================================================================
# Bug Fix Tests (2025-11-27)
# ==============================================================================


class TestBugFixes:
    """Tests verifying bug fixes from code review."""

    def test_market_order_partial_fill_cancelled_qty(self):
        """
        Test market order with insufficient liquidity has correct cancelled_qty.

        BUG FIX: cancelled_qty was calculated as (filled - original) which was negative.
        CORRECT: cancelled_qty = original - filled
        """
        manager = OrderManager()

        # Add partial liquidity
        manager.submit_order(
            side=Side.SELL,
            price=100.0,
            qty=30.0,
            order_type=OrderType.LIMIT,
        )

        # Submit market order for more than available
        result = manager.submit_order(
            side=Side.BUY,
            price=0.0,
            qty=100.0,
            order_type=OrderType.MARKET,
        )

        # Verify correct cancelled_qty (should be positive, not negative)
        assert result.filled_qty == 30.0
        assert result.cancelled_qty == 70.0  # 100 - 30 = 70
        assert result.cancelled_qty >= 0, "cancelled_qty must be non-negative"
        assert result.state == OrderLifecycleState.FILLED

    def test_ioc_partial_fill_state_cancelled(self):
        """
        Test IOC with partial fill has CANCELLED state (not FILLED).

        BUG FIX: IOC with partial fill was marked FILLED.
        CORRECT: Per FIX protocol, IOC with partial fill is CANCELLED (OrdStatus=4).
        """
        manager = OrderManager()

        # Add partial liquidity
        manager.submit_order(
            side=Side.SELL,
            price=100.0,
            qty=50.0,
            order_type=OrderType.LIMIT,
        )

        # IOC for more than available
        result = manager.submit_order(
            side=Side.BUY,
            price=100.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.IOC,
        )

        assert result.filled_qty == 50.0
        assert result.cancelled_qty == 50.0
        # FIX: State should be CANCELLED, not FILLED
        assert result.state == OrderLifecycleState.CANCELLED
        # Verify order is done (not active)
        assert result.is_done
        assert not result.is_active

    def test_ioc_full_fill_state_filled(self):
        """Test IOC with full fill has FILLED state."""
        manager = OrderManager()

        # Add enough liquidity
        manager.submit_order(
            side=Side.SELL,
            price=100.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
        )

        # IOC that fills completely
        result = manager.submit_order(
            side=Side.BUY,
            price=100.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.IOC,
        )

        assert result.filled_qty == 100.0
        assert result.cancelled_qty == 0.0
        assert result.state == OrderLifecycleState.FILLED

    def test_client_id_mapping_consistency(self):
        """
        Test client order ID mapping works correctly.

        BUG FIX: _client_id_map stored internal_id but lookup used client_order_id.
        CORRECT: Map client_id -> order_id for correct lookup.
        """
        manager = OrderManager()

        # Submit order with client_order_id
        order = manager.submit_order(
            side=Side.BUY,
            price=99.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
            client_order_id="my_client_order_123",
        )

        assert order.client_order_id == "my_client_order_123"

        # Lookup by client_order_id should work
        found = manager.get_order_by_client_id("my_client_order_123")
        assert found is not None
        assert found.client_order_id == "my_client_order_123"
        assert found.order.order_id == order.order.order_id

        # Lookup by order_id should also work
        found2 = manager.get_order(order.order.order_id)
        assert found2 is not None
        assert found2 is found

    def test_client_id_without_explicit_id(self):
        """Test order without client_order_id still works."""
        manager = OrderManager()

        # Submit order without client_order_id
        order = manager.submit_order(
            side=Side.BUY,
            price=99.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
        )

        assert order.client_order_id is None

        # Lookup by order_id should work
        found = manager.get_order(order.order.order_id)
        assert found is not None
        assert found.order.order_id == order.order.order_id


# ==============================================================================
# Iceberg and Hidden Order Tests
# ==============================================================================


class TestIcebergAndHiddenOrders:
    """Tests for iceberg and hidden order matching."""

    def test_iceberg_order_partial_visible(self):
        """Test iceberg order shows only display_qty."""
        book = OrderBook()

        # Add iceberg order with 100 visible, 400 hidden
        iceberg = LimitOrder(
            order_id="iceberg_1",
            price=100.0,
            qty=500.0,
            remaining_qty=500.0,
            timestamp_ns=1000,
            side=Side.SELL,
            display_qty=100.0,
            hidden_qty=400.0,
            order_type=OrderType.ICEBERG,
        )
        book.add_limit_order(iceberg)

        # Visible qty should be 100
        assert book.best_ask_qty == 100.0

        # But walk_book sees total (visible + hidden)
        avg_price, total_filled, _ = book.walk_book(Side.BUY, 500.0)
        assert total_filled == 500.0

    def test_iceberg_order_fill_replenish(self):
        """Test iceberg order replenishes display from hidden after fill."""
        iceberg = LimitOrder(
            order_id="iceberg_1",
            price=100.0,
            qty=500.0,
            remaining_qty=500.0,
            timestamp_ns=1000,
            side=Side.SELL,
            display_qty=100.0,
            hidden_qty=400.0,
            order_type=OrderType.ICEBERG,
        )

        # Fill 100 (all visible)
        filled = iceberg.fill(100.0)
        assert filled == 100.0
        assert iceberg.remaining_qty == 400.0

        # Display should replenish from hidden
        # Original display was 100, so replenish up to 100
        assert iceberg.display_qty == 100.0  # Replenished
        assert iceberg.hidden_qty == 300.0   # Reduced

    def test_iceberg_order_matching(self):
        """Test matching engine handles iceberg orders correctly."""
        engine = MatchingEngine()
        book = OrderBook()

        # Add iceberg ask
        iceberg = LimitOrder(
            order_id="iceberg_1",
            price=100.0,
            qty=500.0,
            remaining_qty=500.0,
            timestamp_ns=1000,
            side=Side.SELL,
            display_qty=100.0,
            hidden_qty=400.0,
            order_type=OrderType.ICEBERG,
        )
        book.add_limit_order(iceberg)

        # Market buy should fill entire iceberg
        result = engine.match_market_order(Side.BUY, 500.0, book)

        assert result.total_filled_qty == 500.0
        assert result.is_complete

    def test_hidden_order_not_visible(self):
        """Test hidden order has 0 visible quantity."""
        book = OrderBook()

        # Add hidden order
        hidden = LimitOrder(
            order_id="hidden_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.SELL,
            order_type=OrderType.HIDDEN,
        )
        book.add_limit_order(hidden)

        # Visible qty should be 0
        assert book.best_ask_qty == 0.0

        # But the order exists and can be matched
        avg_price, total_filled, _ = book.walk_book(Side.BUY, 100.0)
        assert total_filled == 100.0

    def test_hidden_order_matching(self):
        """Test matching engine handles hidden orders."""
        engine = MatchingEngine()
        book = OrderBook()

        # Add hidden ask
        hidden = LimitOrder(
            order_id="hidden_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.SELL,
            order_type=OrderType.HIDDEN,
        )
        book.add_limit_order(hidden)

        # Market buy should fill hidden order
        result = engine.match_market_order(Side.BUY, 100.0, book)

        assert result.total_filled_qty == 100.0
        assert result.is_complete

    def test_visible_orders_match_before_hidden(self):
        """Test visible orders have priority over hidden at same price."""
        engine = MatchingEngine()
        book = OrderBook()

        # Add hidden order first (earlier timestamp)
        hidden = LimitOrder(
            order_id="hidden_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1000,
            side=Side.SELL,
            order_type=OrderType.HIDDEN,
        )
        book.add_limit_order(hidden)

        # Add visible order second
        visible = LimitOrder(
            order_id="visible_1",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=2000,
            side=Side.SELL,
            order_type=OrderType.LIMIT,
        )
        book.add_limit_order(visible)

        # Market buy 100 - FIFO means hidden fills first (it arrived first)
        # Note: In real exchanges, visible often has priority over hidden.
        # Our implementation uses strict FIFO.
        result = engine.match_market_order(Side.BUY, 100.0, book)

        assert result.total_filled_qty == 100.0
        # First trade should be from hidden (arrived first - FIFO)
        assert result.fills[0].trades[0].maker_order_id == "hidden_1"


# ==============================================================================
# Additional Edge Case Tests
# ==============================================================================


class TestEdgeCases:
    """Additional edge case tests."""

    def test_zero_quantity_order_rejected(self):
        """Test zero quantity order is rejected."""
        manager = OrderManager()

        result = manager.submit_order(
            side=Side.BUY,
            price=100.0,
            qty=0.0,
            order_type=OrderType.LIMIT,
        )

        assert result.state == OrderLifecycleState.REJECTED

    def test_negative_quantity_order_rejected(self):
        """Test negative quantity order is rejected."""
        manager = OrderManager()

        result = manager.submit_order(
            side=Side.BUY,
            price=100.0,
            qty=-100.0,
            order_type=OrderType.LIMIT,
        )

        assert result.state == OrderLifecycleState.REJECTED

    def test_zero_price_limit_order_rejected(self):
        """Test zero price limit order is rejected."""
        manager = OrderManager()

        result = manager.submit_order(
            side=Side.BUY,
            price=0.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
        )

        assert result.state == OrderLifecycleState.REJECTED

    def test_statistics_cancelled_volume_tracked(self):
        """Test that cancelled volume is tracked in statistics."""
        manager = OrderManager()

        # Add partial liquidity
        manager.submit_order(
            side=Side.SELL,
            price=100.0,
            qty=30.0,
            order_type=OrderType.LIMIT,
        )

        # Market order with partial fill
        manager.submit_order(
            side=Side.BUY,
            price=0.0,
            qty=100.0,
            order_type=OrderType.MARKET,
        )

        stats = manager.get_statistics()
        assert stats.total_volume_cancelled >= 70.0

    def test_modify_order_qty_increase_loses_priority(self):
        """
        Test quantity increase modification loses queue priority.

        FIFO rule: When qty is increased, order loses priority and goes to back.
        This is verified by submitting a sell that fills the first order in queue.
        """
        manager = OrderManager()

        # Submit two orders at same price - order1 first, order2 second
        order1 = manager.submit_order(
            side=Side.BUY,
            price=99.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
        )
        order2 = manager.submit_order(
            side=Side.BUY,
            price=99.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
        )

        # Before modification: order1 is at front, order2 is second
        # Submit small sell to verify order1 fills first
        small_sell = manager.submit_order(
            side=Side.SELL,
            price=99.0,
            qty=10.0,
            order_type=OrderType.LIMIT,
        )
        assert small_sell.filled_qty == 10.0
        # order1 should be partially filled (it was at front)
        assert order1.filled_qty == 10.0
        assert order2.filled_qty == 0.0

        # Modify order1 qty UP (increase) - should lose priority, move to back
        # Note: order1 has remaining 90, increasing to 150 means adding 60 more
        modified = manager.modify_order(order1.order.order_id, new_qty=150.0)
        assert modified is not None

        # Now order2 should be at front (order1 lost priority and moved to back)
        # Submit another sell to verify order2 fills first
        sell = manager.submit_order(
            side=Side.SELL,
            price=99.0,
            qty=100.0,
            order_type=OrderType.LIMIT,
        )

        # order2 should be filled first (was at front after order1 lost priority)
        assert sell.filled_qty == 100.0
        assert order2.state == OrderLifecycleState.FILLED
        assert order2.filled_qty == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
