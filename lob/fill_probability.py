"""
Fill Probability Models for Limit Order Book.

This module provides probabilistic models for estimating the likelihood
of limit order fills based on queue position, market state, and historical data.

Models:
    1. AnalyticalPoissonModel: Classical Poisson-based fill probability
       P(fill in T) = 1 - exp(-λT / position)

    2. QueueReactiveModel: Intensity depends on queue state (Huang et al.)
       λ_i = f(q_i, spread, volatility, imbalance)

    3. HistoricalRateModel: Calibrated rates from historical fill data

References:
    - "A Deep Learning Approach to Estimating Fill Probabilities"
      https://business.columbia.edu/sites/default/files-efs/citation_file_upload/deep-lob-2021.pdf
    - Cont, Stoikov & Talreja (2010): "A Stochastic Model for Order Book Dynamics"
    - Huang, Lehalle & Rosenbaum (2015): Queue-Reactive Model

Performance Target: <10μs per probability estimation
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from lob.data_structures import (
    LimitOrder,
    OrderBook,
    Side,
)
from lob.queue_tracker import (
    LevelStatistics,
    QueueState,
)


# ==============================================================================
# Data Structures
# ==============================================================================


class FillProbabilityModelType(IntEnum):
    """Fill probability model type enumeration."""

    POISSON = 1  # Classical Poisson process
    QUEUE_REACTIVE = 2  # Queue-reactive intensity (Huang et al.)
    HISTORICAL = 3  # Calibrated historical rates
    HAWKES = 4  # Hawkes process (self-exciting)
    DEEP_LOB = 5  # Deep learning model (placeholder)


@dataclass
class LOBState:
    """
    Snapshot of limit order book state for fill probability estimation.

    This provides a unified interface for accessing LOB features needed
    by fill probability models.

    Attributes:
        timestamp_ns: Current timestamp in nanoseconds
        mid_price: Mid-market price
        spread: Bid-ask spread
        spread_bps: Spread in basis points
        bid_depth: Total bid liquidity (qty)
        ask_depth: Total ask liquidity (qty)
        bid_levels: Number of bid price levels
        ask_levels: Number of ask price levels
        imbalance: Book imbalance [-1, 1]
        volatility: Recent price volatility (if available)
        trade_rate: Recent trade arrival rate (trades/sec)
        volume_rate: Recent volume rate (qty/sec)
        order_side: Side of the order being evaluated (optional, for model-specific use)
        order_price: Price of the order being evaluated (optional, for historical lookups)
    """

    timestamp_ns: int = 0
    mid_price: float = 0.0
    spread: float = 0.0
    spread_bps: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    bid_levels: int = 0
    ask_levels: int = 0
    imbalance: float = 0.0
    volatility: float = 0.0
    trade_rate: float = 0.0
    volume_rate: float = 100.0  # Default: 100 shares/sec
    order_side: Optional[Side] = None  # Side of order being evaluated
    order_price: Optional[float] = None  # Price of order being evaluated

    @classmethod
    def from_order_book(
        cls,
        book: OrderBook,
        volatility: float = 0.0,
        trade_rate: float = 0.0,
        volume_rate: float = 100.0,
        timestamp_ns: Optional[int] = None,
        order_side: Optional[Side] = None,
        order_price: Optional[float] = None,
    ) -> "LOBState":
        """Create LOBState from OrderBook instance.

        Args:
            book: OrderBook instance
            volatility: Recent price volatility
            trade_rate: Trade arrival rate (trades/sec)
            volume_rate: Volume rate (qty/sec)
            timestamp_ns: Timestamp in nanoseconds
            order_side: Side of the order being evaluated (for model-specific lookups)
            order_price: Price of the order being evaluated (for historical rate lookups)
        """
        import time

        mid = book.mid_price or 0.0
        spread = book.spread or 0.0
        spread_bps = book.spread_bps or 0.0

        # Calculate depth (top 10 levels)
        bids, asks = book.get_depth(n_levels=10)
        bid_depth = sum(qty for _, qty in bids)
        ask_depth = sum(qty for _, qty in asks)

        # Calculate imbalance
        total = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total if total > 0 else 0.0

        return cls(
            timestamp_ns=timestamp_ns or time.time_ns(),
            mid_price=mid,
            spread=spread,
            spread_bps=spread_bps,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            bid_levels=len(bids),
            ask_levels=len(asks),
            imbalance=imbalance,
            volatility=volatility,
            trade_rate=trade_rate,
            volume_rate=volume_rate,
            order_side=order_side,
            order_price=order_price,
        )


@dataclass
class FillProbabilityResult:
    """
    Result of fill probability estimation.

    Attributes:
        order_id: Order identifier
        prob_fill: Probability of complete fill [0, 1]
        prob_partial: Probability of any fill (partial or complete)
        expected_fill_qty: Expected quantity to be filled
        expected_fill_pct: Expected fill percentage
        expected_wait_time_sec: Expected time to fill (seconds)
        time_horizon_sec: Time horizon used for estimation
        confidence: Model confidence [0, 1]
        model_type: Model used for estimation
        details: Additional model-specific details
    """

    order_id: str
    prob_fill: float = 0.0
    prob_partial: float = 0.0
    expected_fill_qty: float = 0.0
    expected_fill_pct: float = 0.0
    expected_wait_time_sec: float = float("inf")
    time_horizon_sec: float = 60.0
    confidence: float = 1.0
    model_type: FillProbabilityModelType = FillProbabilityModelType.POISSON
    details: Dict[str, float] = field(default_factory=dict)


# ==============================================================================
# Abstract Base Class
# ==============================================================================


class FillProbabilityModel(ABC):
    """
    Abstract base class for fill probability models.

    Subclasses implement specific probabilistic models for estimating
    the likelihood of limit order fills.
    """

    @property
    @abstractmethod
    def model_type(self) -> FillProbabilityModelType:
        """Return model type identifier."""
        pass

    @abstractmethod
    def compute_fill_probability(
        self,
        queue_position: int,
        qty_ahead: float,
        order_qty: float,
        time_horizon_sec: float,
        market_state: LOBState,
    ) -> FillProbabilityResult:
        """
        Compute fill probability for given queue position.

        Args:
            queue_position: Position in queue (0 = front)
            qty_ahead: Quantity ahead in queue
            order_qty: Our order quantity
            time_horizon_sec: Time horizon for estimation
            market_state: Current market state

        Returns:
            FillProbabilityResult with estimates
        """
        pass

    @abstractmethod
    def compute_expected_fill_time(
        self,
        queue_position: int,
        qty_ahead: float,
        market_state: LOBState,
    ) -> float:
        """
        Compute expected time to fill.

        Args:
            queue_position: Position in queue (0 = front)
            qty_ahead: Quantity ahead in queue
            market_state: Current market state

        Returns:
            Expected fill time in seconds
        """
        pass

    def compute_for_queue_state(
        self,
        queue_state: QueueState,
        order_qty: float,
        time_horizon_sec: float,
        market_state: LOBState,
    ) -> FillProbabilityResult:
        """
        Compute fill probability from QueueState.

        Convenience method that extracts parameters from QueueState.
        """
        return self.compute_fill_probability(
            queue_position=queue_state.estimated_position,
            qty_ahead=queue_state.qty_ahead,
            order_qty=order_qty,
            time_horizon_sec=time_horizon_sec,
            market_state=market_state,
        )


# ==============================================================================
# Analytical Poisson Model
# ==============================================================================


class AnalyticalPoissonModel(FillProbabilityModel):
    """
    Classical Poisson-based fill probability model.

    Assumes market order arrivals follow a Poisson process with rate λ.

    Model:
        P(fill in T) = 1 - exp(-λ * T / (qty_ahead + 1))
        E[fill_time] = (qty_ahead + 1) / λ

    Where:
        λ = arrival_rate (expected volume per second at this price)
        T = time horizon
        qty_ahead = quantity ahead of our order in queue

    This is the simplest model and serves as a baseline for more
    sophisticated approaches.

    Reference:
        Cont, Stoikov & Talreja (2010): "A Stochastic Model for Order Book Dynamics"
    """

    def __init__(
        self,
        default_arrival_rate: float = 100.0,  # shares/sec
        front_of_queue_prob: float = 0.95,
        min_arrival_rate: float = 0.01,
    ) -> None:
        """
        Initialize Poisson model.

        Args:
            default_arrival_rate: Default volume arrival rate (qty/sec)
            front_of_queue_prob: Probability when at front of queue
            min_arrival_rate: Minimum arrival rate to prevent division issues
        """
        self._default_arrival_rate = default_arrival_rate
        self._front_of_queue_prob = front_of_queue_prob
        self._min_arrival_rate = min_arrival_rate

    @property
    def model_type(self) -> FillProbabilityModelType:
        return FillProbabilityModelType.POISSON

    def compute_fill_probability(
        self,
        queue_position: int,
        qty_ahead: float,
        order_qty: float,
        time_horizon_sec: float,
        market_state: LOBState,
    ) -> FillProbabilityResult:
        """
        Compute fill probability using Poisson model.

        P(fill) = 1 - exp(-λT / (qty_ahead + 1))
        """
        # Get arrival rate from market state or use default
        arrival_rate = max(
            self._min_arrival_rate,
            (
                market_state.volume_rate
                if market_state.volume_rate > 0
                else self._default_arrival_rate
            ),
        )

        # Handle front of queue case
        if qty_ahead <= 0:
            prob_fill = self._front_of_queue_prob
            expected_wait = order_qty / arrival_rate
        else:
            # Poisson fill probability
            # λ_param represents expected "queue positions consumed"
            expected_volume = arrival_rate * time_horizon_sec
            lambda_param = expected_volume / (qty_ahead + order_qty)

            # P(fill) = P(volume >= qty_ahead + order_qty)
            prob_fill = 1.0 - math.exp(-lambda_param)
            prob_fill = max(0.0, min(1.0, prob_fill))

            # Expected wait time
            expected_wait = (qty_ahead + order_qty) / arrival_rate

        # Partial fill probability (any fill)
        # More likely than complete fill
        partial_lambda = arrival_rate * time_horizon_sec / max(1.0, qty_ahead + 1)
        prob_partial = 1.0 - math.exp(-partial_lambda)
        prob_partial = max(prob_fill, prob_partial)  # At least as likely as complete

        # Expected fill quantity
        expected_fill_qty = prob_fill * order_qty + (prob_partial - prob_fill) * order_qty * 0.5
        expected_fill_pct = expected_fill_qty / order_qty if order_qty > 0 else 0.0

        return FillProbabilityResult(
            order_id="",  # Caller sets this
            prob_fill=prob_fill,
            prob_partial=prob_partial,
            expected_fill_qty=expected_fill_qty,
            expected_fill_pct=expected_fill_pct,
            expected_wait_time_sec=expected_wait,
            time_horizon_sec=time_horizon_sec,
            confidence=0.7,  # Poisson is simple model
            model_type=self.model_type,
            details={
                "arrival_rate": arrival_rate,
                "qty_ahead": qty_ahead,
                "lambda_param": lambda_param if qty_ahead > 0 else 0.0,
            },
        )

    def compute_expected_fill_time(
        self,
        queue_position: int,
        qty_ahead: float,
        market_state: LOBState,
    ) -> float:
        """
        Compute expected fill time.

        E[T] = (qty_ahead + 1) / λ
        """
        arrival_rate = max(
            self._min_arrival_rate,
            (
                market_state.volume_rate
                if market_state.volume_rate > 0
                else self._default_arrival_rate
            ),
        )

        return (qty_ahead + 1) / arrival_rate


# ==============================================================================
# Queue-Reactive Model
# ==============================================================================


class QueueReactiveModel(FillProbabilityModel):
    """
    Queue-Reactive fill probability model.

    Models the arrival intensity as a function of queue state:
        λ_i = λ_0 * f(queue_size) * g(spread) * h(volatility) * k(imbalance)

    Components:
        f(q) = 1 / (1 + α * q)  -- Larger queues attract fewer orders
        g(s) = 1 + β * (s - s_ref) / s_ref  -- Wider spreads increase maker activity
        h(σ) = 1 - γ * σ / σ_ref  -- Higher volatility reduces maker activity
        k(imb) = 1 + δ * imb  -- Imbalance affects direction

    Reference:
        Huang, Lehalle & Rosenbaum (2015):
        "Simulating and analyzing order book data: The queue-reactive model"
    """

    def __init__(
        self,
        base_rate: float = 100.0,  # λ_0: base arrival rate (qty/sec)
        queue_decay_alpha: float = 0.01,  # α: queue size impact
        spread_sensitivity_beta: float = 0.5,  # β: spread impact
        volatility_sensitivity_gamma: float = 0.3,  # γ: volatility impact
        imbalance_sensitivity_delta: float = 0.2,  # δ: imbalance impact
        reference_spread_bps: float = 5.0,  # s_ref: reference spread
        reference_volatility: float = 0.02,  # σ_ref: reference volatility (2%)
        min_rate: float = 1.0,  # Minimum adjusted rate
        max_rate: float = 1000.0,  # Maximum adjusted rate
    ) -> None:
        """
        Initialize Queue-Reactive model.

        Args:
            base_rate: Base arrival rate (qty/sec)
            queue_decay_alpha: Queue size decay coefficient
            spread_sensitivity_beta: Spread sensitivity coefficient
            volatility_sensitivity_gamma: Volatility sensitivity coefficient
            imbalance_sensitivity_delta: Imbalance sensitivity coefficient
            reference_spread_bps: Reference spread in basis points
            reference_volatility: Reference volatility (decimal)
            min_rate: Minimum allowed rate
            max_rate: Maximum allowed rate
        """
        self._base_rate = base_rate
        self._queue_decay_alpha = queue_decay_alpha
        self._spread_sensitivity_beta = spread_sensitivity_beta
        self._volatility_sensitivity_gamma = volatility_sensitivity_gamma
        self._imbalance_sensitivity_delta = imbalance_sensitivity_delta
        self._reference_spread_bps = reference_spread_bps
        self._reference_volatility = reference_volatility
        self._min_rate = min_rate
        self._max_rate = max_rate

    @property
    def model_type(self) -> FillProbabilityModelType:
        return FillProbabilityModelType.QUEUE_REACTIVE

    def compute_adjusted_rate(
        self,
        qty_ahead: float,
        market_state: LOBState,
        side: Optional[Side] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute adjusted arrival rate based on market state.

        Returns:
            Tuple of (adjusted_rate, component_details)
        """
        # Queue size factor: f(q) = 1 / (1 + α * q)
        queue_factor = 1.0 / (1.0 + self._queue_decay_alpha * qty_ahead)

        # Spread factor: g(s) = 1 + β * (s - s_ref) / s_ref
        spread_bps = (
            market_state.spread_bps if market_state.spread_bps > 0 else self._reference_spread_bps
        )
        spread_factor = 1.0 + self._spread_sensitivity_beta * (
            (spread_bps - self._reference_spread_bps) / self._reference_spread_bps
        )
        spread_factor = max(0.5, min(2.0, spread_factor))  # Bound factor

        # Volatility factor: h(σ) = 1 - γ * σ / σ_ref
        vol = market_state.volatility if market_state.volatility > 0 else self._reference_volatility
        vol_factor = 1.0 - self._volatility_sensitivity_gamma * (vol / self._reference_volatility)
        vol_factor = max(0.3, min(1.5, vol_factor))  # Bound factor

        # Imbalance factor: k(imb) = 1 + δ * imb
        # Positive imbalance (more bids) increases fill prob for asks
        imb = market_state.imbalance
        if side == Side.SELL:
            imb_factor = 1.0 + self._imbalance_sensitivity_delta * imb
        elif side == Side.BUY:
            imb_factor = 1.0 - self._imbalance_sensitivity_delta * imb
        else:
            imb_factor = 1.0
        imb_factor = max(0.5, min(1.5, imb_factor))

        # Combine factors
        adjusted_rate = self._base_rate * queue_factor * spread_factor * vol_factor * imb_factor
        adjusted_rate = max(self._min_rate, min(self._max_rate, adjusted_rate))

        details = {
            "base_rate": self._base_rate,
            "queue_factor": queue_factor,
            "spread_factor": spread_factor,
            "vol_factor": vol_factor,
            "imb_factor": imb_factor,
            "adjusted_rate": adjusted_rate,
        }

        return adjusted_rate, details

    def compute_fill_probability(
        self,
        queue_position: int,
        qty_ahead: float,
        order_qty: float,
        time_horizon_sec: float,
        market_state: LOBState,
    ) -> FillProbabilityResult:
        """
        Compute fill probability using Queue-Reactive model.
        """
        # Compute adjusted rate
        arrival_rate, details = self.compute_adjusted_rate(qty_ahead, market_state)

        # Apply Poisson formula with adjusted rate
        if qty_ahead <= 0:
            prob_fill = 0.95
            expected_wait = order_qty / arrival_rate
        else:
            expected_volume = arrival_rate * time_horizon_sec
            lambda_param = expected_volume / (qty_ahead + order_qty)

            prob_fill = 1.0 - math.exp(-lambda_param)
            prob_fill = max(0.0, min(1.0, prob_fill))

            expected_wait = (qty_ahead + order_qty) / arrival_rate

        # Partial fill
        partial_lambda = arrival_rate * time_horizon_sec / max(1.0, qty_ahead + 1)
        prob_partial = 1.0 - math.exp(-partial_lambda)
        prob_partial = max(prob_fill, prob_partial)

        expected_fill_qty = prob_fill * order_qty
        expected_fill_pct = expected_fill_qty / order_qty if order_qty > 0 else 0.0

        return FillProbabilityResult(
            order_id="",
            prob_fill=prob_fill,
            prob_partial=prob_partial,
            expected_fill_qty=expected_fill_qty,
            expected_fill_pct=expected_fill_pct,
            expected_wait_time_sec=expected_wait,
            time_horizon_sec=time_horizon_sec,
            confidence=0.8,  # Better than simple Poisson
            model_type=self.model_type,
            details=details,
        )

    def compute_expected_fill_time(
        self,
        queue_position: int,
        qty_ahead: float,
        market_state: LOBState,
    ) -> float:
        """Compute expected fill time with adjusted rate."""
        arrival_rate, _ = self.compute_adjusted_rate(qty_ahead, market_state)
        return (qty_ahead + 1) / arrival_rate


# ==============================================================================
# Historical Rate Model
# ==============================================================================


@dataclass
class HistoricalFillRate:
    """
    Historical fill rate statistics for a price level.

    Attributes:
        price: Price level
        side: Order side
        avg_fill_rate: Average fill rate (qty/sec)
        fill_rate_std: Standard deviation of fill rate
        avg_time_to_fill: Average time to fill (seconds)
        fill_count: Number of historical fills used
        observation_period_sec: Period over which rates were computed
    """

    price: float
    side: Side
    avg_fill_rate: float = 10.0
    fill_rate_std: float = 5.0
    avg_time_to_fill: float = 30.0
    fill_count: int = 0
    observation_period_sec: float = 3600.0


class HistoricalRateModel(FillProbabilityModel):
    """
    Fill probability model using calibrated historical rates.

    Uses actual historical fill data to estimate rates rather than
    theoretical models. This can provide more accurate estimates
    for specific symbols/levels where sufficient data is available.

    The model maintains a dictionary of HistoricalFillRate objects
    keyed by (price, side) and falls back to defaults when no
    historical data is available.
    """

    def __init__(
        self,
        default_fill_rate: float = 50.0,  # qty/sec
        distance_decay_rate: float = 50.0,  # decay per 1% from mid
        min_confidence: float = 0.5,  # confidence with no historical data
    ) -> None:
        """
        Initialize Historical Rate model.

        Args:
            default_fill_rate: Default rate when no historical data
            distance_decay_rate: Rate decay factor for price distance
            min_confidence: Minimum confidence level
        """
        self._default_fill_rate = default_fill_rate
        self._distance_decay_rate = distance_decay_rate
        self._min_confidence = min_confidence

        # Historical rates by (price, side)
        self._historical_rates: Dict[Tuple[float, Side], HistoricalFillRate] = {}

        # Global statistics (fallback)
        self._global_avg_rate: float = default_fill_rate
        self._global_fill_count: int = 0

    @property
    def model_type(self) -> FillProbabilityModelType:
        return FillProbabilityModelType.HISTORICAL

    def add_historical_rate(self, rate: HistoricalFillRate) -> None:
        """Add historical rate data for a price level."""
        key = (rate.price, rate.side)
        self._historical_rates[key] = rate

        # Update global average
        total_weight = self._global_fill_count + rate.fill_count
        if total_weight > 0:
            self._global_avg_rate = (
                self._global_avg_rate * self._global_fill_count
                + rate.avg_fill_rate * rate.fill_count
            ) / total_weight
            self._global_fill_count = total_weight

    def get_rate_at_price(
        self,
        price: float,
        side: Side,
        mid_price: float,
    ) -> Tuple[float, float]:
        """
        Get fill rate at price level.

        Returns:
            Tuple of (fill_rate, confidence)
        """
        key = (price, side)

        if key in self._historical_rates:
            hist = self._historical_rates[key]
            # Higher confidence with more data
            confidence = min(1.0, 0.5 + hist.fill_count / 100.0)
            return hist.avg_fill_rate, confidence

        # No exact match - use distance-adjusted global rate
        if mid_price > 0:
            distance_pct = abs(price - mid_price) / mid_price * 100
            decay = math.exp(-distance_pct * self._distance_decay_rate / 100)
            rate = self._global_avg_rate * decay
        else:
            rate = self._default_fill_rate

        return rate, self._min_confidence

    def compute_fill_probability(
        self,
        queue_position: int,
        qty_ahead: float,
        order_qty: float,
        time_horizon_sec: float,
        market_state: LOBState,
    ) -> FillProbabilityResult:
        """Compute fill probability using historical rates."""
        # Get rate using order info from market_state if available
        # Otherwise use imbalance-based heuristic for side
        price = (
            market_state.order_price
            if market_state.order_price is not None
            else market_state.mid_price
        )

        if market_state.order_side is not None:
            side = market_state.order_side
        else:
            # Heuristic: infer side from imbalance
            # Positive imbalance (more bids) suggests more likely to be a SELL order
            side = Side.SELL if market_state.imbalance > 0 else Side.BUY

        fill_rate, confidence = self.get_rate_at_price(
            price=price,
            side=side,
            mid_price=market_state.mid_price,
        )

        if qty_ahead <= 0:
            prob_fill = 0.95
            expected_wait = order_qty / fill_rate
        else:
            expected_volume = fill_rate * time_horizon_sec
            lambda_param = expected_volume / (qty_ahead + order_qty)

            prob_fill = 1.0 - math.exp(-lambda_param)
            prob_fill = max(0.0, min(1.0, prob_fill))

            expected_wait = (qty_ahead + order_qty) / fill_rate

        prob_partial = min(1.0, prob_fill * 1.3)
        expected_fill_qty = prob_fill * order_qty

        return FillProbabilityResult(
            order_id="",
            prob_fill=prob_fill,
            prob_partial=prob_partial,
            expected_fill_qty=expected_fill_qty,
            expected_fill_pct=expected_fill_qty / order_qty if order_qty > 0 else 0.0,
            expected_wait_time_sec=expected_wait,
            time_horizon_sec=time_horizon_sec,
            confidence=confidence,
            model_type=self.model_type,
            details={
                "fill_rate": fill_rate,
                "data_points": self._global_fill_count,
            },
        )

    def compute_expected_fill_time(
        self,
        queue_position: int,
        qty_ahead: float,
        market_state: LOBState,
    ) -> float:
        """Compute expected fill time."""
        # Use order info from market_state if available
        price = (
            market_state.order_price
            if market_state.order_price is not None
            else market_state.mid_price
        )

        if market_state.order_side is not None:
            side = market_state.order_side
        else:
            side = Side.SELL if market_state.imbalance > 0 else Side.BUY

        fill_rate, _ = self.get_rate_at_price(
            price=price,
            side=side,
            mid_price=market_state.mid_price,
        )
        return (qty_ahead + 1) / fill_rate


# ==============================================================================
# Distance-Based Model
# ==============================================================================


class DistanceBasedModel(FillProbabilityModel):
    """
    Fill probability model based on distance from mid price.

    Intensity decays exponentially with distance from mid:
        λ(d) = λ_0 * exp(-κ * d)

    Where:
        d = |price - mid| / mid (fractional distance)
        κ = distance decay coefficient

    This model is useful when queue position is unknown but
    the order's price level is known.
    """

    def __init__(
        self,
        base_rate: float = 500.0,  # Very high at best bid/ask
        distance_decay_kappa: float = 100.0,  # Decay coefficient
        tick_adjustment: bool = True,  # Adjust for tick distance
    ) -> None:
        """
        Initialize Distance-Based model.

        Args:
            base_rate: Base arrival rate at BBO
            distance_decay_kappa: Distance decay coefficient
            tick_adjustment: Whether to adjust for tick size
        """
        self._base_rate = base_rate
        self._distance_decay_kappa = distance_decay_kappa
        self._tick_adjustment = tick_adjustment

    @property
    def model_type(self) -> FillProbabilityModelType:
        return FillProbabilityModelType.QUEUE_REACTIVE  # Similar class

    def compute_rate_at_distance(
        self,
        price: float,
        mid_price: float,
        spread_bps: float,
    ) -> float:
        """
        Compute arrival rate based on distance from mid.

        At best bid/ask (half spread away): full base rate
        Further away: exponential decay
        """
        if mid_price <= 0:
            return self._base_rate

        # Distance in percentage terms
        distance_pct = abs(price - mid_price) / mid_price * 100

        # Half spread is the "neutral" point
        half_spread_pct = spread_bps / 200  # Convert bps to % / 2

        # Decay starts from half spread
        excess_distance = max(0, distance_pct - half_spread_pct)
        decay = math.exp(-self._distance_decay_kappa * excess_distance / 100)

        return self._base_rate * decay

    def compute_fill_probability(
        self,
        queue_position: int,
        qty_ahead: float,
        order_qty: float,
        time_horizon_sec: float,
        market_state: LOBState,
    ) -> FillProbabilityResult:
        """Compute fill probability with distance decay."""
        # Assume order is at best bid/ask when distance not specified
        rate = self.compute_rate_at_distance(
            price=market_state.mid_price,
            mid_price=market_state.mid_price,
            spread_bps=market_state.spread_bps,
        )

        if qty_ahead <= 0:
            prob_fill = 0.95
            expected_wait = order_qty / rate
        else:
            expected_volume = rate * time_horizon_sec
            lambda_param = expected_volume / (qty_ahead + order_qty)

            prob_fill = 1.0 - math.exp(-lambda_param)
            expected_wait = (qty_ahead + order_qty) / rate

        return FillProbabilityResult(
            order_id="",
            prob_fill=max(0.0, min(1.0, prob_fill)),
            prob_partial=min(1.0, prob_fill * 1.3),
            expected_fill_qty=prob_fill * order_qty,
            expected_fill_pct=prob_fill,
            expected_wait_time_sec=expected_wait,
            time_horizon_sec=time_horizon_sec,
            confidence=0.75,
            model_type=self.model_type,
            details={"rate": rate},
        )

    def compute_expected_fill_time(
        self,
        queue_position: int,
        qty_ahead: float,
        market_state: LOBState,
    ) -> float:
        """Compute expected fill time."""
        rate = self.compute_rate_at_distance(
            price=market_state.mid_price,
            mid_price=market_state.mid_price,
            spread_bps=market_state.spread_bps,
        )
        return (qty_ahead + 1) / rate


# ==============================================================================
# Composite Model
# ==============================================================================


class CompositeFillProbabilityModel(FillProbabilityModel):
    """
    Composite model that combines multiple fill probability models.

    Uses weighted averaging of predictions from multiple models,
    with weights based on model confidence and configurable factors.

    This allows leveraging the strengths of different models:
    - Historical for well-observed price levels
    - Queue-reactive for dynamic market conditions
    - Distance-based for far-from-mid orders
    """

    def __init__(
        self,
        models: Optional[List[Tuple[FillProbabilityModel, float]]] = None,
    ) -> None:
        """
        Initialize composite model.

        Args:
            models: List of (model, weight) tuples
        """
        if models is None:
            # Default ensemble
            self._models = [
                (AnalyticalPoissonModel(), 0.3),
                (QueueReactiveModel(), 0.5),
                (DistanceBasedModel(), 0.2),
            ]
        else:
            self._models = models

        # Normalize weights
        total_weight = sum(w for _, w in self._models)
        if total_weight > 0:
            self._models = [(m, w / total_weight) for m, w in self._models]

    @property
    def model_type(self) -> FillProbabilityModelType:
        return FillProbabilityModelType.QUEUE_REACTIVE  # Mixed

    def add_model(self, model: FillProbabilityModel, weight: float) -> None:
        """Add a model to the ensemble."""
        self._models.append((model, weight))
        # Renormalize
        total_weight = sum(w for _, w in self._models)
        if total_weight > 0:
            self._models = [(m, w / total_weight) for m, w in self._models]

    def compute_fill_probability(
        self,
        queue_position: int,
        qty_ahead: float,
        order_qty: float,
        time_horizon_sec: float,
        market_state: LOBState,
    ) -> FillProbabilityResult:
        """Compute weighted average of model predictions."""
        weighted_prob_fill = 0.0
        weighted_prob_partial = 0.0
        weighted_wait_time = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0

        all_details: Dict[str, float] = {}

        for model, weight in self._models:
            result = model.compute_fill_probability(
                queue_position, qty_ahead, order_qty, time_horizon_sec, market_state
            )

            # Weight by both configured weight and model confidence
            effective_weight = weight * result.confidence

            weighted_prob_fill += result.prob_fill * effective_weight
            weighted_prob_partial += result.prob_partial * effective_weight
            weighted_wait_time += result.expected_wait_time_sec * effective_weight
            weighted_confidence += result.confidence * weight
            total_weight += effective_weight

            # Collect details
            all_details[f"{model.model_type.name}_prob"] = result.prob_fill
            all_details[f"{model.model_type.name}_weight"] = effective_weight

        if total_weight > 0:
            prob_fill = weighted_prob_fill / total_weight
            prob_partial = weighted_prob_partial / total_weight
            expected_wait = weighted_wait_time / total_weight
        else:
            prob_fill = 0.0
            prob_partial = 0.0
            expected_wait = float("inf")

        return FillProbabilityResult(
            order_id="",
            prob_fill=prob_fill,
            prob_partial=prob_partial,
            expected_fill_qty=prob_fill * order_qty,
            expected_fill_pct=prob_fill,
            expected_wait_time_sec=expected_wait,
            time_horizon_sec=time_horizon_sec,
            confidence=weighted_confidence,
            model_type=self.model_type,
            details=all_details,
        )

    def compute_expected_fill_time(
        self,
        queue_position: int,
        qty_ahead: float,
        market_state: LOBState,
    ) -> float:
        """Compute weighted average expected fill time."""
        weighted_time = 0.0
        total_weight = 0.0

        for model, weight in self._models:
            time = model.compute_expected_fill_time(queue_position, qty_ahead, market_state)
            weighted_time += time * weight
            total_weight += weight

        return weighted_time / total_weight if total_weight > 0 else float("inf")


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_fill_probability_model(
    model_type: Union[str, FillProbabilityModelType] = "queue_reactive",
    **kwargs,
) -> FillProbabilityModel:
    """
    Factory function to create fill probability models.

    Args:
        model_type: Model type name or enum
        **kwargs: Model-specific parameters

    Returns:
        FillProbabilityModel instance

    Example:
        model = create_fill_probability_model("queue_reactive", base_rate=200.0)
    """
    if isinstance(model_type, str):
        model_type_map = {
            "poisson": FillProbabilityModelType.POISSON,
            "queue_reactive": FillProbabilityModelType.QUEUE_REACTIVE,
            "historical": FillProbabilityModelType.HISTORICAL,
            "distance": FillProbabilityModelType.QUEUE_REACTIVE,  # Distance is a variant
            "composite": FillProbabilityModelType.QUEUE_REACTIVE,
        }
        model_enum = model_type_map.get(model_type.lower(), FillProbabilityModelType.POISSON)
    else:
        model_enum = model_type

    if model_type == "composite":
        return CompositeFillProbabilityModel(**kwargs)
    elif model_type == "distance":
        return DistanceBasedModel(**kwargs)
    elif model_enum == FillProbabilityModelType.POISSON:
        return AnalyticalPoissonModel(**kwargs)
    elif model_enum == FillProbabilityModelType.QUEUE_REACTIVE:
        return QueueReactiveModel(**kwargs)
    elif model_enum == FillProbabilityModelType.HISTORICAL:
        return HistoricalRateModel(**kwargs)
    else:
        return AnalyticalPoissonModel(**kwargs)


# ==============================================================================
# Integration with QueuePositionTracker
# ==============================================================================


def compute_fill_probability_for_order(
    queue_state: QueueState,
    order: LimitOrder,
    order_book: OrderBook,
    model: Optional[FillProbabilityModel] = None,
    time_horizon_sec: float = 60.0,
) -> FillProbabilityResult:
    """
    Convenience function to compute fill probability for an order.

    Args:
        queue_state: Current queue state from tracker
        order: The limit order
        order_book: Current order book
        model: Fill probability model (default: QueueReactiveModel)
        time_horizon_sec: Time horizon for estimation

    Returns:
        FillProbabilityResult
    """
    if model is None:
        model = QueueReactiveModel()

    # Include order info in market state for models that need it (e.g., HistoricalRateModel)
    market_state = LOBState.from_order_book(
        order_book,
        order_side=order.side,
        order_price=order.price,
    )

    result = model.compute_for_queue_state(
        queue_state=queue_state,
        order_qty=order.remaining_qty,
        time_horizon_sec=time_horizon_sec,
        market_state=market_state,
    )

    result.order_id = order.order_id
    return result
