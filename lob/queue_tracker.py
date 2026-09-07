"""
Queue Position Tracker for Limit Order Book.

Tracks and estimates queue position for our orders using:
- MBP (Market-by-Price) estimation: Pessimistic, uses aggregate level data
- MBO (Market-by-Order) estimation: Exact, uses individual order data
- Probabilistic model for cancellations ahead of us
- Fill probability estimation based on queue position

Reference:
    Erik Rigtorp's method for estimating order queue position
    https://rigtorp.se/2013/06/08/estimating-order-queue-position.html

    Columbia University LOB paper on fill probability estimation
    "Limit Order Book Dynamics and Order Sizes" - Cont et al.

Performance Target: <1us per position update
"""

from __future__ import annotations

import math
import time
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
    PriceLevel,
    Side,
    Trade,
)


# ==============================================================================
# Data Structures
# ==============================================================================


class PositionEstimationMethod(IntEnum):
    """Method used for position estimation."""

    MBO = 1  # Market-by-Order (exact)
    MBP_PESSIMISTIC = 2  # Market-by-Price (pessimistic)
    MBP_OPTIMISTIC = 3  # Market-by-Price (optimistic)
    PROBABILISTIC = 4  # Probabilistic model


@dataclass
class QueueState:
    """
    State of an order's queue position.

    Attributes:
        order_id: Order identifier
        price: Price level
        side: Order side
        estimated_position: Estimated queue position (0 = front)
        qty_ahead: Estimated quantity ahead in queue
        total_level_qty: Total quantity at this level
        confidence: Confidence in estimation [0, 1]
        method: Method used for estimation
        last_update_ns: Timestamp of last update
    """

    order_id: str
    price: float
    side: Side
    estimated_position: int = 0
    qty_ahead: float = 0.0
    total_level_qty: float = 0.0
    confidence: float = 1.0
    method: PositionEstimationMethod = PositionEstimationMethod.MBO
    last_update_ns: int = 0

    @property
    def position_pct(self) -> float:
        """Position as percentage of queue (0% = front, 100% = back)."""
        if self.total_level_qty <= 0:
            return 0.0
        return (self.qty_ahead / self.total_level_qty) * 100.0


@dataclass
class FillProbability:
    """
    Fill probability estimation for an order.

    Attributes:
        order_id: Order identifier
        prob_fill: Probability of fill in time horizon [0, 1]
        prob_partial: Probability of partial fill
        expected_fill_qty: Expected fill quantity
        expected_wait_time_sec: Expected time to fill
        time_horizon_sec: Time horizon for estimation
    """

    order_id: str
    prob_fill: float = 0.0
    prob_partial: float = 0.0
    expected_fill_qty: float = 0.0
    expected_wait_time_sec: float = float("inf")
    time_horizon_sec: float = 60.0


@dataclass
class LevelStatistics:
    """
    Statistics for a price level used in fill probability estimation.

    Attributes:
        price: Price level
        avg_arrival_rate: Average order arrival rate (orders/sec)
        avg_cancellation_rate: Average cancellation rate (orders/sec)
        avg_execution_rate: Average execution rate (qty/sec)
        avg_order_size: Average order size
    """

    price: float
    avg_arrival_rate: float = 1.0  # orders per second
    avg_cancellation_rate: float = 0.1  # orders per second
    avg_execution_rate: float = 10.0  # shares per second
    avg_order_size: float = 100.0


# ==============================================================================
# Queue Position Tracker
# ==============================================================================


class QueuePositionTracker:
    """
    Tracks queue position for our orders.

    Supports two estimation methods:
    1. MBP (Market-by-Price): Uses aggregate level data, pessimistic
    2. MBO (Market-by-Order): Uses individual order data, exact

    Usage:
        tracker = QueuePositionTracker()

        # Track our order
        tracker.add_order(order, level_qty_before=500.0)

        # Update on executions
        tracker.update_on_execution(executed_qty=100.0, at_price=100.0)

        # Get fill probability
        prob = tracker.estimate_fill_probability(order_id, volume_rate=1000.0)

    Reference:
        Erik Rigtorp's method:
        https://rigtorp.se/2013/06/08/estimating-order-queue-position.html
    """

    def __init__(
        self,
        default_method: PositionEstimationMethod = PositionEstimationMethod.MBP_PESSIMISTIC,
        on_position_update: Optional[Callable[[str, QueueState], None]] = None,
    ) -> None:
        """
        Initialize queue position tracker.

        Args:
            default_method: Default estimation method
            on_position_update: Callback for position updates
        """
        self._default_method = default_method
        self._on_position_update = on_position_update

        # Tracked orders: order_id -> QueueState
        self._tracked_orders: Dict[str, QueueState] = {}

        # Level statistics for fill probability
        self._level_stats: Dict[Tuple[Side, float], LevelStatistics] = {}

        # Historical execution data for calibration
        self._execution_history: List[Tuple[int, float, float]] = []  # (ts, price, qty)

    # ==========================================================================
    # Order Tracking
    # ==========================================================================

    def add_order(
        self,
        order: LimitOrder,
        level_qty_before: float,
        orders_ahead: Optional[List[LimitOrder]] = None,
    ) -> QueueState:
        """
        Start tracking a new order.

        Args:
            order: LimitOrder to track
            level_qty_before: Total quantity at level before our order
            orders_ahead: Optional list of orders ahead (for MBO)

        Returns:
            QueueState for the order
        """
        if orders_ahead is not None:
            # MBO: Exact position
            position = len(orders_ahead)
            qty_ahead = sum(o.remaining_qty for o in orders_ahead)
            method = PositionEstimationMethod.MBO
            confidence = 1.0
        else:
            # MBP: Pessimistic estimate (we're at the end)
            position = -1  # Unknown
            qty_ahead = level_qty_before
            method = PositionEstimationMethod.MBP_PESSIMISTIC
            confidence = 0.7

        state = QueueState(
            order_id=order.order_id,
            price=order.price,
            side=order.side,
            estimated_position=position,
            qty_ahead=qty_ahead,
            total_level_qty=qty_ahead + order.remaining_qty,
            confidence=confidence,
            method=method,
            last_update_ns=time.time_ns(),
        )

        self._tracked_orders[order.order_id] = state

        if self._on_position_update:
            self._on_position_update(order.order_id, state)

        return state

    def remove_order(self, order_id: str) -> Optional[QueueState]:
        """
        Stop tracking an order.

        Args:
            order_id: Order ID to stop tracking

        Returns:
            Final QueueState or None if not found
        """
        return self._tracked_orders.pop(order_id, None)

    def get_state(self, order_id: str) -> Optional[QueueState]:
        """Get current queue state for an order."""
        return self._tracked_orders.get(order_id)

    # ==========================================================================
    # Position Estimation
    # ==========================================================================

    def estimate_position_mbp(
        self,
        order: LimitOrder,
        level_qty_before: float,
        level_qty_after: float,
    ) -> int:
        """
        Estimate queue position from MBP data (pessimistic).

        MBP estimation is pessimistic because:
        - We don't know exact order positions
        - Assume our order is at the back of the queue
        - Only advance on executions at our price

        Args:
            order: Our limit order
            level_qty_before: Level quantity before add
            level_qty_after: Level quantity after add

        Returns:
            Estimated position (0 = front)
        """
        # Position estimate: qty ahead / avg order size
        # Pessimistic: assume all qty before us is ahead
        avg_order_size = 100.0  # Default assumption
        estimated_position = int(level_qty_before / avg_order_size)

        return estimated_position

    def estimate_position_mbo(
        self,
        order: LimitOrder,
        orders_ahead: List[LimitOrder],
    ) -> int:
        """
        Estimate queue position from MBO data (exact).

        MBO provides exact position by counting orders ahead.

        Args:
            order: Our limit order
            orders_ahead: List of orders ahead of us

        Returns:
            Exact position (0 = front)
        """
        return len(orders_ahead)

    def estimate_qty_ahead(
        self,
        order_id: str,
        orders_at_level: Optional[List[LimitOrder]] = None,
    ) -> float:
        """
        Estimate quantity ahead of our order in the queue.

        Args:
            order_id: Our order ID
            orders_at_level: Optional list of all orders at level (for MBO)

        Returns:
            Estimated quantity ahead
        """
        state = self._tracked_orders.get(order_id)
        if state is None:
            return 0.0

        if orders_at_level is not None:
            # MBO: Exact calculation
            qty_ahead = 0.0
            found_our_order = False
            for order in orders_at_level:
                if order.order_id == order_id:
                    found_our_order = True
                    break
                qty_ahead += order.remaining_qty
            return qty_ahead if found_our_order else state.qty_ahead

        return state.qty_ahead

    # ==========================================================================
    # Position Updates
    # ==========================================================================

    def update_on_execution(
        self,
        executed_qty: float,
        at_price: float,
        at_side: Optional[Side] = None,
    ) -> List[str]:
        """
        Update queue positions when executions occur at a price.

        When executions happen at our price level, we advance in the queue.

        Args:
            executed_qty: Quantity executed
            at_price: Execution price
            at_side: Side where execution occurred (optional)

        Returns:
            List of order IDs that were updated
        """
        updated_orders: List[str] = []
        now = time.time_ns()

        # Record execution for calibration
        self._execution_history.append((now, at_price, executed_qty))
        self._trim_history()

        for order_id, state in self._tracked_orders.items():
            # Only update if execution is at our price
            if abs(state.price - at_price) > 1e-9:
                continue

            # Skip if side doesn't match (if provided)
            if at_side is not None and state.side != at_side:
                continue

            # Advance position: reduce qty ahead
            old_qty_ahead = state.qty_ahead
            state.qty_ahead = max(0.0, state.qty_ahead - executed_qty)

            # Update position estimate
            if state.qty_ahead == 0:
                state.estimated_position = 0  # Front of queue

            state.last_update_ns = now
            updated_orders.append(order_id)

            if self._on_position_update:
                self._on_position_update(order_id, state)

        return updated_orders

    def update_on_cancel_ahead(
        self,
        cancelled_qty: float,
        at_price: float,
        at_side: Side,
        probability: float = 0.5,
    ) -> List[str]:
        """
        Update positions when cancellation occurs ahead of us.

        For MBP data, we don't know if the cancellation was ahead or behind.
        We use a probabilistic model based on queue position.

        Args:
            cancelled_qty: Quantity cancelled
            at_price: Price level
            at_side: Side of cancellation
            probability: Probability cancellation was ahead of us

        Returns:
            List of order IDs that were updated
        """
        updated_orders: List[str] = []
        now = time.time_ns()

        for order_id, state in self._tracked_orders.items():
            if abs(state.price - at_price) > 1e-9:
                continue
            if state.side != at_side:
                continue

            # Probabilistic adjustment
            expected_reduction = cancelled_qty * probability
            state.qty_ahead = max(0.0, state.qty_ahead - expected_reduction)

            # Reduce confidence due to uncertainty
            state.confidence *= 0.95
            state.method = PositionEstimationMethod.PROBABILISTIC
            state.last_update_ns = now

            updated_orders.append(order_id)

            if self._on_position_update:
                self._on_position_update(order_id, state)

        return updated_orders

    def update_on_level_change(
        self,
        side: Side,
        price: float,
        new_total_qty: float,
    ) -> List[str]:
        """
        Update tracking when level total changes.

        Args:
            side: Side of the level
            price: Price level
            new_total_qty: New total quantity at level

        Returns:
            List of order IDs that were updated
        """
        updated_orders: List[str] = []
        now = time.time_ns()

        for order_id, state in self._tracked_orders.items():
            if abs(state.price - price) > 1e-9:
                continue
            if state.side != side:
                continue

            state.total_level_qty = new_total_qty
            state.last_update_ns = now
            updated_orders.append(order_id)

        return updated_orders

    # ==========================================================================
    # Fill Probability Estimation
    # ==========================================================================

    def estimate_fill_probability(
        self,
        order_id: str,
        volume_per_second: float = 100.0,
        time_horizon_sec: float = 60.0,
        order_qty: Optional[float] = None,
    ) -> FillProbability:
        """
        Estimate probability of order being filled.

        Uses a simple model based on:
        - Queue position (qty ahead)
        - Expected volume rate at this price
        - Time horizon

        More sophisticated models can use:
        - Poisson arrival processes for trades
        - Intensity models based on price distance from mid
        - Historical fill rates

        Args:
            order_id: Order to estimate for
            volume_per_second: Expected volume rate at this price
            time_horizon_sec: Time horizon for estimation
            order_qty: Order quantity (if not tracked)

        Returns:
            FillProbability with estimates
        """
        state = self._tracked_orders.get(order_id)
        if state is None:
            return FillProbability(
                order_id=order_id,
                prob_fill=0.0,
                time_horizon_sec=time_horizon_sec,
            )

        qty_ahead = state.qty_ahead

        # Simple model: exponential fill probability
        # P(fill) = 1 - exp(-volume_rate * time / qty_ahead)
        if qty_ahead <= 0:
            # Front of queue - high probability
            prob_fill = 0.95
            expected_wait = 1.0 / max(0.01, volume_per_second)
        else:
            expected_volume = volume_per_second * time_horizon_sec
            lambda_param = expected_volume / (qty_ahead + 1)

            # Poisson-based fill probability
            prob_fill = 1.0 - math.exp(-lambda_param)
            prob_fill = max(0.0, min(1.0, prob_fill))

            # Expected wait time (queue / rate)
            expected_wait = qty_ahead / max(0.01, volume_per_second)

        # Partial fill probability (higher than full fill)
        prob_partial = min(1.0, prob_fill * 1.5)

        return FillProbability(
            order_id=order_id,
            prob_fill=prob_fill,
            prob_partial=prob_partial,
            expected_fill_qty=prob_fill * (order_qty or 0.0),
            expected_wait_time_sec=expected_wait,
            time_horizon_sec=time_horizon_sec,
        )

    def estimate_fill_probability_advanced(
        self,
        order_id: str,
        order_book: OrderBook,
        historical_trades: Optional[List[Trade]] = None,
        time_horizon_sec: float = 60.0,
    ) -> FillProbability:
        """
        Advanced fill probability estimation using book state.

        Uses:
        - Distance from mid price (intensity decreases with distance)
        - Historical trade frequency at this level
        - Imbalance of book (predicts direction of trades)

        Args:
            order_id: Order to estimate for
            order_book: Current order book state
            historical_trades: Recent trades for calibration
            time_horizon_sec: Time horizon

        Returns:
            FillProbability with estimates
        """
        state = self._tracked_orders.get(order_id)
        if state is None:
            return FillProbability(order_id=order_id, prob_fill=0.0)

        # Base rate from historical trades
        base_rate = self._estimate_rate_at_level(state.side, state.price, historical_trades)

        # Distance from mid price factor
        mid = order_book.mid_price
        if mid is not None and mid > 0:
            distance_pct = abs(state.price - mid) / mid
            # Exponential decay with distance
            distance_factor = math.exp(-distance_pct * 100)
        else:
            distance_factor = 0.5

        # Book imbalance factor
        imbalance = self._calculate_book_imbalance(order_book, state.side)

        # Combined rate
        adjusted_rate = base_rate * distance_factor * (1 + imbalance)

        # Fill probability
        if state.qty_ahead <= 0:
            prob_fill = 0.95
        else:
            expected_volume = adjusted_rate * time_horizon_sec
            lambda_param = expected_volume / (state.qty_ahead + 1)
            prob_fill = 1.0 - math.exp(-lambda_param)

        return FillProbability(
            order_id=order_id,
            prob_fill=max(0.0, min(1.0, prob_fill)),
            prob_partial=min(1.0, prob_fill * 1.3),
            expected_wait_time_sec=state.qty_ahead / max(0.01, adjusted_rate),
            time_horizon_sec=time_horizon_sec,
        )

    def _estimate_rate_at_level(
        self,
        side: Side,
        price: float,
        historical_trades: Optional[List[Trade]] = None,
    ) -> float:
        """Estimate execution rate at a price level."""
        # Use level statistics if available
        key = (side, price)
        if key in self._level_stats:
            return self._level_stats[key].avg_execution_rate

        # Use execution history
        if self._execution_history:
            recent_qty = sum(qty for ts, p, qty in self._execution_history if abs(p - price) < 0.01)
            time_span = (self._execution_history[-1][0] - self._execution_history[0][0]) / 1e9
            if time_span > 0:
                return recent_qty / time_span

        # Default rate
        return 10.0  # shares per second

    def _calculate_book_imbalance(
        self,
        order_book: OrderBook,
        side: Side,
    ) -> float:
        """
        Calculate book imbalance.

        Positive imbalance for BUY side means more buying pressure.

        Args:
            order_book: Current order book
            side: Side to calculate imbalance for

        Returns:
            Imbalance factor [-1, 1]
        """
        bid_qty = order_book.best_bid_qty
        ask_qty = order_book.best_ask_qty

        total = bid_qty + ask_qty
        if total <= 0:
            return 0.0

        # Imbalance = (bid - ask) / (bid + ask)
        imbalance = (bid_qty - ask_qty) / total

        # For BUY orders, positive imbalance means more likely to fill
        # (buying pressure pushes price up, lifting asks)
        if side == Side.BUY:
            return imbalance
        else:
            return -imbalance

    def _trim_history(self, max_age_sec: float = 3600.0) -> None:
        """Trim old execution history."""
        if not self._execution_history:
            return

        cutoff = time.time_ns() - int(max_age_sec * 1e9)
        self._execution_history = [
            (ts, p, q) for ts, p, q in self._execution_history if ts >= cutoff
        ]

    # ==========================================================================
    # Level Statistics Management
    # ==========================================================================

    def update_level_statistics(
        self,
        side: Side,
        price: float,
        stats: LevelStatistics,
    ) -> None:
        """Update statistics for a price level."""
        self._level_stats[(side, price)] = stats

    def get_level_statistics(
        self,
        side: Side,
        price: float,
    ) -> Optional[LevelStatistics]:
        """Get statistics for a price level."""
        return self._level_stats.get((side, price))

    # ==========================================================================
    # Bulk Operations
    # ==========================================================================

    def get_all_states(self) -> Dict[str, QueueState]:
        """Get all tracked order states."""
        return dict(self._tracked_orders)

    def get_orders_at_price(
        self,
        side: Side,
        price: float,
    ) -> List[QueueState]:
        """Get all tracked orders at a specific price."""
        return [
            state
            for state in self._tracked_orders.values()
            if state.side == side and abs(state.price - price) < 1e-9
        ]

    def clear(self) -> None:
        """Clear all tracked orders."""
        self._tracked_orders.clear()
        self._execution_history.clear()

    @property
    def tracked_count(self) -> int:
        """Number of orders being tracked."""
        return len(self._tracked_orders)


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_queue_tracker(
    method: str = "mbp_pessimistic",
    **kwargs,
) -> QueuePositionTracker:
    """
    Create queue position tracker.

    Args:
        method: Estimation method ("mbo", "mbp_pessimistic", "mbp_optimistic")
        **kwargs: Additional arguments

    Returns:
        QueuePositionTracker instance
    """
    method_map = {
        "mbo": PositionEstimationMethod.MBO,
        "mbp_pessimistic": PositionEstimationMethod.MBP_PESSIMISTIC,
        "mbp_optimistic": PositionEstimationMethod.MBP_OPTIMISTIC,
        "probabilistic": PositionEstimationMethod.PROBABILISTIC,
    }

    estimation_method = method_map.get(
        method.lower(),
        PositionEstimationMethod.MBP_PESSIMISTIC,
    )

    return QueuePositionTracker(default_method=estimation_method, **kwargs)
