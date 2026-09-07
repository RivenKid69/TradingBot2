"""
Queue Value Model for Limit Order Book.

This module provides models for computing the value of holding a queue position
in a limit order book, enabling intelligent order management decisions.

The value of a queue position depends on:
    1. Expected profit if filled (spread capture)
    2. Probability of fill before adverse price move
    3. Adverse selection cost (information asymmetry)
    4. Opportunity cost of capital tied up

Key formula:
    V(position, spread) = spread/2 * P(fill before adverse move) - E[loss | adverse]

References:
    - Moallemi & Yuan (2017): "A Model for Queue Position Valuation..."
      https://moallemi.com/ciamac/papers/queue-value-2016.pdf
    - Cont & de Larrard (2013): "Price dynamics in a Markovian limit order market"
    - Cartea & Jaimungal (2015): "Optimal Execution with Limit Orders"

Performance Target: <50μs per value computation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

from lob.data_structures import (
    LimitOrder,
    OrderBook,
    Side,
)
from lob.fill_probability import (
    FillProbabilityModel,
    FillProbabilityResult,
    LOBState,
    QueueReactiveModel,
)
from lob.queue_tracker import (
    QueueState,
)


# ==============================================================================
# Data Structures
# ==============================================================================


class OrderDecision(IntEnum):
    """Decision for order management."""

    HOLD = 1  # Keep order, queue position valuable
    CANCEL = 2  # Cancel order, queue position worthless
    REPRICE = 3  # Move to better price (aggressive)
    INCREASE_SIZE = 4  # Add to position
    REDUCE_SIZE = 5  # Partial cancel


@dataclass
class QueueValueResult:
    """
    Result of queue value computation.

    Attributes:
        order_id: Order identifier
        queue_value: Value of current queue position ($)
        expected_profit: Expected profit if filled
        adverse_selection_cost: Expected cost from adverse selection
        opportunity_cost: Cost of capital tied up
        fill_probability: Probability of fill in horizon
        decision: Recommended action
        confidence: Model confidence [0, 1]
        breakeven_spread_bps: Spread needed to make position profitable
        details: Additional model-specific details
    """

    order_id: str
    queue_value: float = 0.0
    expected_profit: float = 0.0
    adverse_selection_cost: float = 0.0
    opportunity_cost: float = 0.0
    fill_probability: float = 0.0
    decision: OrderDecision = OrderDecision.HOLD
    confidence: float = 1.0
    breakeven_spread_bps: float = 0.0
    details: Dict[str, float] = field(default_factory=dict)


@dataclass
class AdverseSelectionParams:
    """
    Parameters for adverse selection model.

    Attributes:
        informed_trader_fraction: Fraction of informed traders (PIN-like)
        information_value_bps: Value of private info in basis points
        adverse_move_probability: P(adverse move | trade)
        adverse_move_size_bps: Expected size of adverse move in bps
    """

    informed_trader_fraction: float = 0.2  # 20% informed traders
    information_value_bps: float = 10.0  # 10 bps info value
    adverse_move_probability: float = 0.3  # 30% chance of adverse move
    adverse_move_size_bps: float = 5.0  # 5 bps adverse move


@dataclass
class QueueValueConfig:
    """
    Configuration for queue value model.

    Attributes:
        time_horizon_sec: Time horizon for value estimation
        risk_aversion: Risk aversion coefficient (0 = neutral, >0 = averse)
        opportunity_cost_rate: Annual opportunity cost rate
        min_profitable_value: Minimum value to recommend HOLD
        adverse_selection: Adverse selection parameters
    """

    time_horizon_sec: float = 60.0
    risk_aversion: float = 0.0
    opportunity_cost_rate: float = 0.02  # 2% annual
    min_profitable_value: float = 0.0
    adverse_selection: AdverseSelectionParams = field(default_factory=AdverseSelectionParams)


# ==============================================================================
# Queue Value Model
# ==============================================================================


class QueueValueModel:
    """
    Model for computing the value of holding a queue position.

    The queue value represents the expected profit from keeping an order
    in the book versus canceling it. A positive value indicates the order
    should be held; negative value suggests cancellation.

    Value Components:
        1. Spread Capture: Half-spread profit if filled
        2. Fill Probability: Likelihood of execution
        3. Adverse Selection: Loss from trading with informed traders
        4. Opportunity Cost: Time value of money tied up

    Formula:
        V = P(fill) * spread/2 - P(fill) * E[adverse_selection] - opportunity_cost

    Reference:
        Moallemi & Yuan (2017): "A Model for Queue Position Valuation"
        https://moallemi.com/ciamac/papers/queue-value-2016.pdf
    """

    def __init__(
        self,
        fill_model: Optional[FillProbabilityModel] = None,
        config: Optional[QueueValueConfig] = None,
    ) -> None:
        """
        Initialize queue value model.

        Args:
            fill_model: Fill probability model (default: QueueReactiveModel)
            config: Configuration parameters
        """
        self._fill_model = fill_model or QueueReactiveModel()
        self._config = config or QueueValueConfig()

    def compute_queue_value(
        self,
        order: LimitOrder,
        market_state: LOBState,
        queue_state: Optional[QueueState] = None,
    ) -> QueueValueResult:
        """
        Compute the value of holding the current queue position.

        Args:
            order: The limit order
            market_state: Current market state
            queue_state: Optional queue state (computed if not provided)

        Returns:
            QueueValueResult with value and decision
        """
        # Get queue position info
        if queue_state is not None:
            qty_ahead = queue_state.qty_ahead
            position = queue_state.estimated_position
        else:
            qty_ahead = 0.0
            position = 0

        # Compute fill probability
        fill_result = self._fill_model.compute_fill_probability(
            queue_position=position,
            qty_ahead=qty_ahead,
            order_qty=order.remaining_qty,
            time_horizon_sec=self._config.time_horizon_sec,
            market_state=market_state,
        )
        prob_fill = fill_result.prob_fill

        # Spread capture profit (half spread)
        spread_bps = market_state.spread_bps if market_state.spread_bps > 0 else 2.0
        half_spread_bps = spread_bps / 2.0
        notional = (
            order.remaining_qty * market_state.mid_price if market_state.mid_price > 0 else 0.0
        )
        spread_profit = notional * half_spread_bps / 10000.0

        # Expected profit if filled
        expected_profit = prob_fill * spread_profit

        # Adverse selection cost
        adverse_cost = self._compute_adverse_selection_cost(
            order, market_state, prob_fill, notional
        )

        # Opportunity cost
        opportunity_cost = self._compute_opportunity_cost(
            notional, fill_result.expected_wait_time_sec
        )

        # Queue value = expected profit - costs
        queue_value = expected_profit - adverse_cost - opportunity_cost

        # Risk adjustment
        if self._config.risk_aversion > 0:
            # Reduce value for uncertain outcomes
            uncertainty = prob_fill * (1 - prob_fill)
            risk_penalty = self._config.risk_aversion * uncertainty * spread_profit
            queue_value -= risk_penalty

        # Compute breakeven spread
        if prob_fill > 0 and notional > 0:
            min_spread_needed = (adverse_cost + opportunity_cost) / (prob_fill * notional) * 10000
            breakeven_spread_bps = min_spread_needed * 2  # Full spread from half-spread
        else:
            breakeven_spread_bps = float("inf")

        # Decision logic
        decision = self._make_decision(queue_value, prob_fill, spread_bps, breakeven_spread_bps)

        # Confidence based on fill model confidence and data quality
        confidence = fill_result.confidence * 0.9  # Slightly reduce for additional model

        return QueueValueResult(
            order_id=order.order_id,
            queue_value=queue_value,
            expected_profit=expected_profit,
            adverse_selection_cost=adverse_cost,
            opportunity_cost=opportunity_cost,
            fill_probability=prob_fill,
            decision=decision,
            confidence=confidence,
            breakeven_spread_bps=breakeven_spread_bps,
            details={
                "spread_bps": spread_bps,
                "half_spread_profit": spread_profit,
                "notional": notional,
                "expected_wait_sec": fill_result.expected_wait_time_sec,
                "qty_ahead": qty_ahead,
            },
        )

    def _compute_adverse_selection_cost(
        self,
        order: LimitOrder,
        market_state: LOBState,
        prob_fill: float,
        notional: float,
    ) -> float:
        """
        Compute expected adverse selection cost.

        Adverse selection occurs when we trade with informed traders who
        have information about future price moves.

        Formula:
            E[AS] = P(fill) * P(informed | trade) * E[adverse_move]
        """
        params = self._config.adverse_selection

        # Probability of adverse selection given fill
        prob_adverse = params.adverse_move_probability * params.informed_trader_fraction

        # Expected loss from adverse move
        adverse_loss = notional * params.adverse_move_size_bps / 10000.0

        # Conditional on fill
        adverse_cost = prob_fill * prob_adverse * adverse_loss

        return adverse_cost

    def _compute_opportunity_cost(
        self,
        notional: float,
        expected_wait_sec: float,
    ) -> float:
        """
        Compute opportunity cost of capital tied up.

        Uses annual opportunity cost rate converted to the expected
        holding period.
        """
        if expected_wait_sec <= 0 or expected_wait_sec == float("inf"):
            return 0.0

        # Convert expected wait to fraction of year
        wait_fraction = expected_wait_sec / (365.25 * 24 * 3600)

        # Apply opportunity cost rate
        opportunity_cost = notional * self._config.opportunity_cost_rate * wait_fraction

        return opportunity_cost

    def _make_decision(
        self,
        queue_value: float,
        prob_fill: float,
        current_spread_bps: float,
        breakeven_spread_bps: float,
    ) -> OrderDecision:
        """
        Make order management decision based on queue value.

        Decision rules:
            - HOLD: Value positive and above threshold
            - CANCEL: Value negative or very low fill probability
            - REPRICE: Could be more profitable at different price
        """
        min_value = self._config.min_profitable_value

        # Cancel if value is negative
        if queue_value < min_value:
            # Check if repricing could help
            if current_spread_bps > breakeven_spread_bps:
                # Current spread is sufficient, issue is queue position
                return OrderDecision.CANCEL
            else:
                # Might be profitable at wider spread (more aggressive)
                return OrderDecision.REPRICE

        # Cancel if fill probability is very low
        if prob_fill < 0.01:  # < 1% chance
            return OrderDecision.CANCEL

        # Hold if value is positive
        return OrderDecision.HOLD

    def should_cancel(
        self,
        order: LimitOrder,
        market_state: LOBState,
        queue_state: Optional[QueueState] = None,
    ) -> bool:
        """
        Check if this order should be cancelled (not repriced).

        Returns True only for CANCEL decision. For orders that should be
        repriced but not cancelled, this returns False.

        Use should_modify() if you want to check for either CANCEL or REPRICE.
        """
        result = self.compute_queue_value(order, market_state, queue_state)
        return result.decision == OrderDecision.CANCEL

    def should_reprice(
        self,
        order: LimitOrder,
        market_state: LOBState,
        queue_state: Optional[QueueState] = None,
    ) -> bool:
        """
        Check if this order should be repriced (moved to better price).

        Returns True only for REPRICE decision.
        """
        result = self.compute_queue_value(order, market_state, queue_state)
        return result.decision == OrderDecision.REPRICE

    def should_modify(
        self,
        order: LimitOrder,
        market_state: LOBState,
        queue_state: Optional[QueueState] = None,
    ) -> bool:
        """
        Check if this order should be modified (cancelled or repriced).

        Convenience method for checking if any action should be taken.
        Use should_cancel() or should_reprice() for specific decisions.
        """
        result = self.compute_queue_value(order, market_state, queue_state)
        return result.decision in (OrderDecision.CANCEL, OrderDecision.REPRICE)


# ==============================================================================
# Enhanced Queue Value Models
# ==============================================================================


class DynamicQueueValueModel(QueueValueModel):
    """
    Queue value model with dynamic parameter adjustment.

    Adjusts adverse selection and opportunity cost parameters based on
    market conditions (volatility, volume, time of day).
    """

    def __init__(
        self,
        fill_model: Optional[FillProbabilityModel] = None,
        config: Optional[QueueValueConfig] = None,
        volatility_adjustment: float = 1.0,  # Scale adverse selection with vol
    ) -> None:
        super().__init__(fill_model, config)
        self._volatility_adjustment = volatility_adjustment

    def _compute_adverse_selection_cost(
        self,
        order: LimitOrder,
        market_state: LOBState,
        prob_fill: float,
        notional: float,
    ) -> float:
        """Compute adverse selection with volatility adjustment."""
        base_cost = super()._compute_adverse_selection_cost(
            order, market_state, prob_fill, notional
        )

        # Scale by volatility relative to reference
        reference_vol = 0.02  # 2% reference
        if market_state.volatility > 0:
            vol_scale = market_state.volatility / reference_vol
            vol_scale = max(0.5, min(2.0, vol_scale))  # Bound
            return base_cost * vol_scale**self._volatility_adjustment

        return base_cost


class SpreadDecompositionModel(QueueValueModel):
    """
    Queue value model using spread decomposition.

    Decomposes the bid-ask spread into:
        1. Order processing cost
        2. Inventory risk premium
        3. Adverse selection component

    Reference:
        Glosten & Harris (1988): "Estimating the Components of the Bid-Ask Spread"
    """

    def __init__(
        self,
        fill_model: Optional[FillProbabilityModel] = None,
        config: Optional[QueueValueConfig] = None,
        processing_cost_bps: float = 0.5,  # Fixed processing cost
        inventory_risk_bps: float = 1.0,  # Per-unit inventory risk
        adverse_selection_pct: float = 0.3,  # % of spread from AS
    ) -> None:
        super().__init__(fill_model, config)
        self._processing_cost_bps = processing_cost_bps
        self._inventory_risk_bps = inventory_risk_bps
        self._adverse_selection_pct = adverse_selection_pct

    def _compute_spread_components(
        self,
        spread_bps: float,
        position_size: float,
    ) -> Tuple[float, float, float]:
        """
        Decompose spread into components.

        Returns:
            Tuple of (processing, inventory, adverse_selection) in bps
        """
        # Processing cost is fixed
        processing = self._processing_cost_bps

        # Inventory risk scales with position
        inventory = self._inventory_risk_bps * abs(position_size) / 1000.0

        # Adverse selection is fraction of remaining spread
        remaining = max(0, spread_bps - processing - inventory)
        adverse = remaining * self._adverse_selection_pct

        return processing, inventory, adverse

    def compute_queue_value(
        self,
        order: LimitOrder,
        market_state: LOBState,
        queue_state: Optional[QueueState] = None,
    ) -> QueueValueResult:
        """Compute value with spread decomposition."""
        # Get base result
        result = super().compute_queue_value(order, market_state, queue_state)

        # Decompose spread
        spread_bps = market_state.spread_bps if market_state.spread_bps > 0 else 2.0
        processing, inventory, adverse = self._compute_spread_components(
            spread_bps, order.remaining_qty
        )

        # Update details
        result.details["spread_processing_bps"] = processing
        result.details["spread_inventory_bps"] = inventory
        result.details["spread_adverse_bps"] = adverse

        # Recalculate expected profit using only processing + inventory
        # (adverse selection is a cost, not profit)
        profit_spread_bps = processing + inventory
        notional = result.details.get("notional", 0)
        revised_profit = result.fill_probability * notional * profit_spread_bps / 10000.0

        # Adjust queue value
        result.queue_value = (
            revised_profit - result.adverse_selection_cost - result.opportunity_cost
        )

        return result


# ==============================================================================
# Queue Value Tracker
# ==============================================================================


class QueueValueTracker:
    """
    Tracks queue values for multiple orders over time.

    Provides:
        - Continuous value monitoring
        - Automated cancel recommendations
        - Value history for analysis
    """

    def __init__(
        self,
        model: Optional[QueueValueModel] = None,
        auto_cancel_threshold: float = -0.01,  # Cancel if value < -1 cent
    ) -> None:
        """
        Initialize queue value tracker.

        Args:
            model: Queue value model
            auto_cancel_threshold: Threshold for auto-cancel recommendation
        """
        self._model = model or QueueValueModel()
        self._auto_cancel_threshold = auto_cancel_threshold

        # Tracked orders: order_id -> (order, queue_state)
        self._orders: Dict[str, Tuple[LimitOrder, Optional[QueueState]]] = {}

        # Value history: order_id -> [(timestamp, value)]
        self._value_history: Dict[str, List[Tuple[int, float]]] = {}

        # Cancel recommendations
        self._cancel_recommendations: Dict[str, QueueValueResult] = {}

    def track_order(
        self,
        order: LimitOrder,
        queue_state: Optional[QueueState] = None,
    ) -> None:
        """Start tracking an order."""
        self._orders[order.order_id] = (order, queue_state)
        self._value_history[order.order_id] = []

    def untrack_order(self, order_id: str) -> Optional[LimitOrder]:
        """Stop tracking an order."""
        if order_id in self._orders:
            order, _ = self._orders.pop(order_id)
            self._value_history.pop(order_id, None)
            self._cancel_recommendations.pop(order_id, None)
            return order
        return None

    def update_queue_state(
        self,
        order_id: str,
        queue_state: QueueState,
    ) -> None:
        """Update queue state for an order."""
        if order_id in self._orders:
            order, _ = self._orders[order_id]
            self._orders[order_id] = (order, queue_state)

    def update_values(
        self,
        market_state: LOBState,
    ) -> List[str]:
        """
        Update values for all tracked orders.

        Returns:
            List of order IDs recommended for cancellation
        """
        import time

        cancel_recommendations: List[str] = []
        timestamp = time.time_ns()

        for order_id, (order, queue_state) in self._orders.items():
            # Compute current value
            result = self._model.compute_queue_value(order, market_state, queue_state)

            # Record history
            self._value_history[order_id].append((timestamp, result.queue_value))

            # Check for cancel
            if result.queue_value < self._auto_cancel_threshold:
                cancel_recommendations.append(order_id)
                self._cancel_recommendations[order_id] = result

        return cancel_recommendations

    def get_value(
        self,
        order_id: str,
        market_state: LOBState,
    ) -> Optional[QueueValueResult]:
        """Get current queue value for an order."""
        if order_id not in self._orders:
            return None

        order, queue_state = self._orders[order_id]
        return self._model.compute_queue_value(order, market_state, queue_state)

    def get_value_history(
        self,
        order_id: str,
    ) -> List[Tuple[int, float]]:
        """Get value history for an order."""
        return self._value_history.get(order_id, [])

    def get_orders_to_cancel(self) -> Dict[str, QueueValueResult]:
        """Get orders recommended for cancellation."""
        return dict(self._cancel_recommendations)

    @property
    def tracked_count(self) -> int:
        """Number of orders being tracked."""
        return len(self._orders)


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_queue_value_model(
    model_type: str = "standard",
    fill_model: Optional[FillProbabilityModel] = None,
    config: Optional[QueueValueConfig] = None,
    **kwargs,
) -> QueueValueModel:
    """
    Factory function to create queue value models.

    Args:
        model_type: "standard", "dynamic", or "spread_decomposition"
        fill_model: Fill probability model
        config: Queue value configuration
        **kwargs: Model-specific parameters

    Returns:
        QueueValueModel instance
    """
    model_type_lower = model_type.lower()

    if model_type_lower == "dynamic":
        return DynamicQueueValueModel(fill_model, config, **kwargs)
    elif model_type_lower in ("spread", "spread_decomposition", "glosten_harris"):
        return SpreadDecompositionModel(fill_model, config, **kwargs)
    else:
        return QueueValueModel(fill_model, config)


def compute_order_queue_value(
    order: LimitOrder,
    order_book: OrderBook,
    queue_state: Optional[QueueState] = None,
    model: Optional[QueueValueModel] = None,
) -> QueueValueResult:
    """
    Convenience function to compute queue value for an order.

    Args:
        order: The limit order
        order_book: Current order book
        queue_state: Optional queue state
        model: Queue value model (default: standard)

    Returns:
        QueueValueResult
    """
    if model is None:
        model = QueueValueModel()

    market_state = LOBState.from_order_book(order_book)

    return model.compute_queue_value(order, market_state, queue_state)


# ==============================================================================
# Queue Improvement Estimator
# ==============================================================================


class QueueImprovementEstimator:
    """
    Estimates the improvement in queue position over time.

    Models how queue position improves due to:
        1. Executions ahead (fills at our price)
        2. Cancellations ahead
        3. Time decay of stale orders

    This helps estimate when an order might reach the front of the queue.
    """

    def __init__(
        self,
        execution_rate: float = 100.0,  # qty/sec executed
        cancel_rate: float = 0.1,  # fraction of queue/sec cancelled
        avg_order_size: float = 100.0,
    ) -> None:
        """
        Initialize estimator.

        Args:
            execution_rate: Expected execution rate (qty/sec)
            cancel_rate: Expected cancellation rate (fraction/sec)
            avg_order_size: Average order size at the level
        """
        self._execution_rate = execution_rate
        self._cancel_rate = cancel_rate
        self._avg_order_size = avg_order_size

    def estimate_time_to_front(
        self,
        qty_ahead: float,
        market_state: Optional[LOBState] = None,
    ) -> float:
        """
        Estimate time until order reaches front of queue.

        Args:
            qty_ahead: Quantity ahead in queue
            market_state: Optional market state for rate adjustment

        Returns:
            Expected time in seconds
        """
        if qty_ahead <= 0:
            return 0.0

        # Effective depletion rate = executions + cancellations
        exec_rate = self._execution_rate
        if market_state and market_state.volume_rate > 0:
            exec_rate = market_state.volume_rate

        cancel_depletion = qty_ahead * self._cancel_rate
        total_rate = exec_rate + cancel_depletion

        if total_rate <= 0:
            return float("inf")

        return qty_ahead / total_rate

    def estimate_position_at_time(
        self,
        current_qty_ahead: float,
        time_sec: float,
        market_state: Optional[LOBState] = None,
    ) -> float:
        """
        Estimate queue position after given time.

        Args:
            current_qty_ahead: Current quantity ahead
            time_sec: Time horizon
            market_state: Optional market state

        Returns:
            Expected quantity ahead after time_sec
        """
        if current_qty_ahead <= 0:
            return 0.0

        exec_rate = self._execution_rate
        if market_state and market_state.volume_rate > 0:
            exec_rate = market_state.volume_rate

        # Expected position = current - executions - cancellations
        executed = exec_rate * time_sec
        cancelled = current_qty_ahead * self._cancel_rate * time_sec

        remaining = current_qty_ahead - executed - cancelled

        return max(0.0, remaining)

    def compute_position_distribution(
        self,
        current_qty_ahead: float,
        time_sec: float,
        n_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Compute distribution of queue position after time.

        Provides percentiles for risk analysis.

        Returns:
            Dict with "mean", "p10", "p50", "p90" estimates
        """
        mean = self.estimate_position_at_time(current_qty_ahead, time_sec)

        # Simple variance estimate (Poisson-like)
        exec_rate = self._execution_rate
        variance = exec_rate * time_sec  # Poisson variance = rate * time

        std = math.sqrt(variance) if variance > 0 else 0

        return {
            "mean": mean,
            "std": std,
            "p10": max(0, mean - 1.28 * std),
            "p50": mean,
            "p90": mean + 1.28 * std,
        }
