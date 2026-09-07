# -*- coding: utf-8 -*-
"""
Unified L3 Calibration Pipeline.

Combines all calibration subsystems for L3 LOB simulation:
1. Market Impact Calibration (η, γ, τ, β from Almgren-Chriss/Gatheral)
2. Fill Probability Calibration (λ, α, β from Queue-Reactive model)
3. Latency Distribution Fitting (μ, σ for log-normal distributions)
4. Queue Dynamics Calibration (arrival rates, cancel rates)

The pipeline produces an L3Config suitable for use with L3ExecutionProvider.

Usage:
    pipeline = L3CalibrationPipeline()

    # Add historical data
    for trade in historical_trades:
        pipeline.add_trade(trade)
    for order in historical_orders:
        pipeline.add_order(order)

    # Run calibration
    config = pipeline.run_full_calibration()

    # Use with L3 execution provider
    provider = L3ExecutionProvider(config=config)

References:
    - Almgren et al. (2005): Impact calibration
    - Huang et al. (2015): Queue-reactive model
    - Cont, Stoikov & Talreja (2010): Poisson fill rates
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from lob.calibration import (
    AdverseSelectionCalibrator,
    CalibrationPipeline as FillProbCalibrationPipeline,
    CalibrationResult as FillCalibrationResult,
    HistoricalRateCalibrator,
    OrderRecord,
    PoissonRateCalibrator,
    QueueReactiveCalibrator,
    TradeRecord,
)
from lob.config import (
    FillProbabilityConfig,
    FillProbabilityModelType,
    HiddenLiquidityConfig,
    ImpactModelType,
    L3ExecutionConfig,
    LatencyComponentConfig,
    LatencyConfig,
    LatencyDistributionType,
    LatencyProfileType,
    MarketImpactConfig,
    QueueTrackingConfig,
)
from lob.data_structures import Side
from lob.fill_probability import (
    FillProbabilityModelType as FillModelType,
)
from lob.impact_calibration import (
    AlmgrenChrissCalibrator,
    CalibrationDataset as ImpactCalibrationDataset,
    CalibrationResult as ImpactCalibrationResult,
    GatheralDecayCalibrator,
    ImpactCalibrationPipeline,
    KyleLambdaCalibrator,
    RollingImpactCalibrator,
    TradeObservation,
)
from lob.market_impact import ImpactModelType as ImpactType

logger = logging.getLogger(__name__)


# ==============================================================================
# Data Structures
# ==============================================================================


class CalibrationDataType(Enum):
    """Type of calibration data."""

    TRADE = "trade"
    ORDER = "order"
    LATENCY = "latency"
    QUOTE = "quote"


@dataclass
class LatencyObservation:
    """
    Single latency observation for calibration.

    Attributes:
        timestamp_ns: Observation timestamp
        latency_type: Type of latency (feed, order, exchange, fill)
        latency_us: Latency in microseconds
        symbol: Optional symbol
        side: Optional order side
    """

    timestamp_ns: int
    latency_type: str  # "feed", "order", "exchange", "fill", "round_trip"
    latency_us: float
    symbol: str = ""
    side: Optional[Side] = None
    order_type: str = "market"


@dataclass
class QuoteObservation:
    """
    Quote observation for spread and depth calibration.

    Attributes:
        timestamp_ns: Observation timestamp
        bid_price: Best bid price
        ask_price: Best ask price
        bid_size: Best bid size
        ask_size: Best ask size
        spread_bps: Spread in basis points
    """

    timestamp_ns: int
    bid_price: float
    ask_price: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    spread_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.spread_bps == 0.0 and self.bid_price > 0:
            self.spread_bps = (self.ask_price - self.bid_price) / self.bid_price * 10000


@dataclass
class LatencyDistributionParams:
    """
    Parameters for latency distribution.

    Supports log-normal fitting (most common for latency).
    """

    distribution: LatencyDistributionType = LatencyDistributionType.LOGNORMAL
    mean_us: float = 100.0
    std_us: float = 30.0
    min_us: float = 1.0
    max_us: float = 100_000.0
    # Log-normal specific
    mu: float = 0.0  # log(mean) - sigma^2/2
    sigma: float = 0.0  # sqrt(log(1 + var/mean^2))
    # Pareto specific
    pareto_alpha: float = 2.5
    pareto_xmin_us: float = 10.0

    @classmethod
    def fit_lognormal(cls, samples: Sequence[float]) -> "LatencyDistributionParams":
        """
        Fit log-normal distribution to latency samples.

        Args:
            samples: Latency samples in microseconds

        Returns:
            Fitted LatencyDistributionParams
        """
        if len(samples) < 2:
            return cls()

        # Filter outliers (>99.9 percentile)
        arr = np.array([s for s in samples if s > 0])
        if len(arr) < 2:
            return cls()

        p999 = np.percentile(arr, 99.9)
        arr = arr[arr <= p999]

        # Compute mean and std
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))

        # Log-normal parameters
        # For log-normal: E[X] = exp(mu + sigma^2/2)
        # Var[X] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
        if mean_val > 0 and std_val > 0:
            cv2 = (std_val / mean_val) ** 2
            sigma = math.sqrt(math.log(1 + cv2))
            mu = math.log(mean_val) - sigma**2 / 2
        else:
            mu = math.log(max(mean_val, 1.0))
            sigma = 0.5

        return cls(
            distribution=LatencyDistributionType.LOGNORMAL,
            mean_us=mean_val,
            std_us=std_val,
            min_us=min_val,
            max_us=max_val,
            mu=mu,
            sigma=sigma,
        )

    @classmethod
    def fit_pareto(cls, samples: Sequence[float]) -> "LatencyDistributionParams":
        """
        Fit Pareto distribution for heavy-tailed latency.

        Args:
            samples: Latency samples in microseconds

        Returns:
            Fitted LatencyDistributionParams
        """
        if len(samples) < 2:
            return cls(distribution=LatencyDistributionType.PARETO)

        arr = np.array([s for s in samples if s > 0])
        if len(arr) < 2:
            return cls(distribution=LatencyDistributionType.PARETO)

        # Estimate x_min as median
        xmin = float(np.median(arr))

        # MLE for alpha: alpha = n / sum(log(x/xmin)) for x >= xmin
        valid = arr[arr >= xmin]
        if len(valid) < 2:
            alpha = 2.5
        else:
            log_sum = np.sum(np.log(valid / xmin))
            # Protect against division by zero when all values equal xmin
            if log_sum < 1e-10:
                alpha = 2.5  # Default for degenerate case
            else:
                alpha = len(valid) / log_sum
            alpha = max(1.1, min(alpha, 10.0))  # Bound alpha

        return cls(
            distribution=LatencyDistributionType.PARETO,
            mean_us=float(np.mean(arr)),
            std_us=float(np.std(arr)),
            min_us=float(np.min(arr)),
            max_us=float(np.max(arr)),
            pareto_alpha=alpha,
            pareto_xmin_us=xmin,
        )


@dataclass
class LatencyCalibrationResult:
    """Result of latency calibration."""

    feed_latency: LatencyDistributionParams = field(default_factory=LatencyDistributionParams)
    order_latency: LatencyDistributionParams = field(default_factory=LatencyDistributionParams)
    exchange_latency: LatencyDistributionParams = field(default_factory=LatencyDistributionParams)
    fill_latency: LatencyDistributionParams = field(default_factory=LatencyDistributionParams)
    n_samples: int = 0
    time_range_sec: float = 0.0


@dataclass
class QueueDynamicsResult:
    """Result of queue dynamics calibration."""

    avg_arrival_rate: float = 100.0  # Orders per second
    avg_cancel_rate: float = 50.0  # Cancels per second
    fill_rate_by_level: Dict[int, float] = field(default_factory=dict)  # Level -> fills/sec
    avg_queue_position_at_fill: float = 5.0
    avg_time_in_queue_sec: float = 30.0
    n_samples: int = 0


@dataclass
class L3CalibrationResult:
    """
    Complete result of L3 calibration pipeline.

    Contains all calibrated parameters and quality metrics.
    """

    # Impact calibration
    impact_result: Optional[ImpactCalibrationResult] = None

    # Fill probability calibration
    fill_prob_result: Optional[FillCalibrationResult] = None

    # Latency calibration
    latency_result: Optional[LatencyCalibrationResult] = None

    # Queue dynamics
    queue_dynamics: Optional[QueueDynamicsResult] = None

    # Overall metrics
    n_trades: int = 0
    n_orders: int = 0
    data_duration_sec: float = 0.0
    calibration_quality: str = "unknown"  # "low", "medium", "high"

    # Confidence intervals (95%)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def get_impact_params(self) -> Dict[str, float]:
        """Get calibrated impact parameters."""
        if self.impact_result:
            return self.impact_result.parameters.copy()
        return {}

    def get_fill_prob_params(self) -> Dict[str, float]:
        """Get calibrated fill probability parameters."""
        if self.fill_prob_result:
            return self.fill_prob_result.parameters.copy()
        return {}


# ==============================================================================
# Latency Calibrator
# ==============================================================================


class LatencyCalibrator:
    """
    Calibrates latency distribution parameters from historical data.

    Fits log-normal or Pareto distributions to observed latencies.
    """

    def __init__(
        self,
        distribution_type: LatencyDistributionType = LatencyDistributionType.LOGNORMAL,
    ) -> None:
        """
        Initialize latency calibrator.

        Args:
            distribution_type: Target distribution type
        """
        self._distribution_type = distribution_type
        self._observations: Dict[str, List[float]] = {
            "feed": [],
            "order": [],
            "exchange": [],
            "fill": [],
            "round_trip": [],
        }

    def add_observation(self, obs: LatencyObservation) -> None:
        """Add a latency observation."""
        key = obs.latency_type.lower()
        if key in self._observations:
            self._observations[key].append(obs.latency_us)

    def add_observations(self, observations: Sequence[LatencyObservation]) -> None:
        """Add multiple latency observations."""
        for obs in observations:
            self.add_observation(obs)

    def fit(self) -> LatencyCalibrationResult:
        """
        Fit latency distributions from observations.

        Returns:
            LatencyCalibrationResult with fitted parameters
        """
        result = LatencyCalibrationResult()
        total_samples = 0

        # Fit each latency type
        if self._observations["feed"]:
            result.feed_latency = self._fit_distribution(self._observations["feed"])
            total_samples += len(self._observations["feed"])

        if self._observations["order"]:
            result.order_latency = self._fit_distribution(self._observations["order"])
            total_samples += len(self._observations["order"])

        if self._observations["exchange"]:
            result.exchange_latency = self._fit_distribution(self._observations["exchange"])
            total_samples += len(self._observations["exchange"])

        if self._observations["fill"]:
            result.fill_latency = self._fit_distribution(self._observations["fill"])
            total_samples += len(self._observations["fill"])

        result.n_samples = total_samples

        return result

    def _fit_distribution(self, samples: List[float]) -> LatencyDistributionParams:
        """Fit distribution to samples."""
        if self._distribution_type == LatencyDistributionType.PARETO:
            return LatencyDistributionParams.fit_pareto(samples)
        else:
            return LatencyDistributionParams.fit_lognormal(samples)

    def clear(self) -> None:
        """Clear all observations."""
        for key in self._observations:
            self._observations[key].clear()


# ==============================================================================
# Queue Dynamics Calibrator
# ==============================================================================


class QueueDynamicsCalibrator:
    """
    Calibrates queue dynamics parameters.

    Estimates:
    - Order arrival rates by price level
    - Cancel rates
    - Fill rates by queue position
    - Average time in queue
    """

    def __init__(self) -> None:
        self._orders: List[OrderRecord] = []
        self._trades: List[TradeRecord] = []
        self._queue_fills: List[Tuple[int, float, float]] = []  # (position, qty, time_in_queue)

    def add_order(self, order: OrderRecord) -> None:
        """Add order record."""
        self._orders.append(order)

    def add_trade(self, trade: TradeRecord) -> None:
        """Add trade record."""
        self._trades.append(trade)

    def add_queue_fill(
        self,
        queue_position: int,
        fill_qty: float,
        time_in_queue_sec: float,
    ) -> None:
        """Add queue fill observation."""
        self._queue_fills.append((queue_position, fill_qty, time_in_queue_sec))

    def fit(self) -> QueueDynamicsResult:
        """
        Fit queue dynamics parameters.

        Returns:
            QueueDynamicsResult with calibrated parameters
        """
        result = QueueDynamicsResult()

        # Calculate time range
        all_times = [o.timestamp_ns for o in self._orders] + [t.timestamp_ns for t in self._trades]
        if len(all_times) >= 2:
            time_range_sec = (max(all_times) - min(all_times)) / 1e9
        else:
            time_range_sec = 1.0

        # Arrival rate: orders per second
        add_orders = [o for o in self._orders if o.event_type == "ADD"]
        result.avg_arrival_rate = len(add_orders) / max(time_range_sec, 1.0)

        # Cancel rate: cancels per second
        cancel_orders = [o for o in self._orders if o.event_type == "CANCEL"]
        result.avg_cancel_rate = len(cancel_orders) / max(time_range_sec, 1.0)

        # Fill rate by queue position
        if self._queue_fills:
            # Group by position
            position_fills: Dict[int, List[float]] = {}
            position_times: Dict[int, List[float]] = {}

            for pos, qty, time_in_queue in self._queue_fills:
                bucket = min(pos // 5, 10) * 5  # Bucket positions
                if bucket not in position_fills:
                    position_fills[bucket] = []
                    position_times[bucket] = []
                position_fills[bucket].append(qty)
                position_times[bucket].append(time_in_queue)

            for bucket, fills in position_fills.items():
                times = position_times[bucket]
                if times:
                    avg_time = sum(times) / len(times)
                    result.fill_rate_by_level[bucket] = (
                        sum(fills) / sum(times) if sum(times) > 0 else 0
                    )

            # Average queue position at fill
            positions = [pos for pos, _, _ in self._queue_fills]
            result.avg_queue_position_at_fill = (
                sum(positions) / len(positions) if positions else 5.0
            )

            # Average time in queue
            times = [t for _, _, t in self._queue_fills if t > 0]
            result.avg_time_in_queue_sec = sum(times) / len(times) if times else 30.0

        result.n_samples = len(self._orders) + len(self._trades) + len(self._queue_fills)

        return result

    def clear(self) -> None:
        """Clear all data."""
        self._orders.clear()
        self._trades.clear()
        self._queue_fills.clear()


# ==============================================================================
# L3 Calibration Pipeline
# ==============================================================================


class L3CalibrationPipeline:
    """
    Unified calibration pipeline for L3 LOB simulation.

    Calibrates all model parameters from historical data:
    - Market impact coefficients (η, γ, τ)
    - Fill probability parameters (λ)
    - Latency distributions
    - Queue dynamics

    Produces L3ExecutionConfig ready for simulation.
    """

    def __init__(
        self,
        symbol: str = "",
        asset_class: str = "equity",
    ) -> None:
        """
        Initialize L3 calibration pipeline.

        Args:
            symbol: Trading symbol
            asset_class: "equity" or "crypto"
        """
        self._symbol = symbol
        self._asset_class = asset_class

        # Sub-calibrators
        self._impact_pipeline = ImpactCalibrationPipeline()
        self._fill_prob_pipeline = FillProbCalibrationPipeline()
        self._latency_calibrator = LatencyCalibrator()
        self._queue_calibrator = QueueDynamicsCalibrator()

        # Raw data storage
        self._trade_observations: List[TradeObservation] = []
        self._trade_records: List[TradeRecord] = []
        self._order_records: List[OrderRecord] = []
        self._latency_observations: List[LatencyObservation] = []
        self._quote_observations: List[QuoteObservation] = []

        # Computed values
        self._avg_adv: float = 10_000_000.0
        self._avg_volatility: float = 0.02

    @property
    def symbol(self) -> str:
        """Get symbol."""
        return self._symbol

    @property
    def n_trades(self) -> int:
        """Number of trade records."""
        return len(self._trade_observations)

    @property
    def n_orders(self) -> int:
        """Number of order records."""
        return len(self._order_records)

    def add_trade(
        self,
        timestamp_ms: int,
        price: float,
        qty: float,
        side: int,
        pre_trade_mid: Optional[float] = None,
        post_trade_mid: Optional[float] = None,
        adv: Optional[float] = None,
        volatility: Optional[float] = None,
    ) -> None:
        """
        Add a trade observation.

        Args:
            timestamp_ms: Trade timestamp in milliseconds
            price: Trade price
            qty: Trade quantity
            side: Trade side (+1 = buy, -1 = sell)
            pre_trade_mid: Mid price before trade
            post_trade_mid: Mid price after trade
            adv: Average daily volume
            volatility: Volatility estimate
        """
        obs = TradeObservation(
            timestamp_ms=timestamp_ms,
            price=price,
            qty=qty,
            side=side,
            adv=adv or self._avg_adv,
            volatility=volatility or self._avg_volatility,
            pre_trade_mid=pre_trade_mid,
            post_trade_mid=post_trade_mid,
        )
        self._trade_observations.append(obs)

        # Also add to fill probability pipeline
        trade_side = Side.BUY if side > 0 else Side.SELL
        trade_record = TradeRecord(
            timestamp_ns=timestamp_ms * 1_000_000,
            price=price,
            qty=qty,
            side=trade_side,
        )
        self._trade_records.append(trade_record)
        self._fill_prob_pipeline.add_trade(trade_record)

    def add_order(
        self,
        timestamp_ns: int,
        price: float,
        qty: float,
        side: Side,
        event_type: str,
        fill_qty: float = 0.0,
        queue_position: Optional[int] = None,
    ) -> None:
        """
        Add an order observation.

        Args:
            timestamp_ns: Order timestamp in nanoseconds
            price: Order price
            qty: Order quantity
            side: Order side
            event_type: "ADD", "CANCEL", "MODIFY", or "FILL"
            fill_qty: Filled quantity (for FILL events)
            queue_position: Queue position at submission
        """
        record = OrderRecord(
            timestamp_ns=timestamp_ns,
            price=price,
            qty=qty,
            side=side,
            event_type=event_type,
            fill_qty=fill_qty,
            queue_position=queue_position,
        )
        self._order_records.append(record)
        self._fill_prob_pipeline.add_order(record)
        self._queue_calibrator.add_order(record)

    def add_latency_observation(
        self,
        timestamp_ns: int,
        latency_type: str,
        latency_us: float,
        symbol: str = "",
    ) -> None:
        """
        Add a latency observation.

        Args:
            timestamp_ns: Observation timestamp
            latency_type: "feed", "order", "exchange", "fill", or "round_trip"
            latency_us: Latency in microseconds
            symbol: Optional symbol
        """
        obs = LatencyObservation(
            timestamp_ns=timestamp_ns,
            latency_type=latency_type,
            latency_us=latency_us,
            symbol=symbol or self._symbol,
        )
        self._latency_observations.append(obs)
        self._latency_calibrator.add_observation(obs)

    def add_quote_observation(
        self,
        timestamp_ns: int,
        bid_price: float,
        ask_price: float,
        bid_size: float = 0.0,
        ask_size: float = 0.0,
    ) -> None:
        """
        Add a quote observation for spread calibration.

        Args:
            timestamp_ns: Quote timestamp
            bid_price: Best bid price
            ask_price: Best ask price
            bid_size: Best bid size
            ask_size: Best ask size
        """
        obs = QuoteObservation(
            timestamp_ns=timestamp_ns,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
        )
        self._quote_observations.append(obs)

    def set_market_params(
        self,
        avg_adv: float,
        avg_volatility: float,
    ) -> None:
        """
        Set market parameters for impact calibration.

        Args:
            avg_adv: Average daily volume
            avg_volatility: Average volatility
        """
        self._avg_adv = avg_adv
        self._avg_volatility = avg_volatility

    def calibrate_impact(
        self,
        trades: Optional[Sequence[Dict[str, Any]]] = None,
        quotes: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> ImpactCalibrationResult:
        """
        Calibrate market impact parameters.

        Uses regression-based calibration for Almgren-Chriss model.

        Args:
            trades: Optional trade data (if not using add_trade)
            quotes: Optional quote data (for mid price)

        Returns:
            ImpactCalibrationResult with calibrated parameters
        """
        # Build dataset from observations
        if trades:
            for t in trades:
                self.add_trade(
                    timestamp_ms=t.get("timestamp_ms", 0),
                    price=t.get("price", 0.0),
                    qty=t.get("qty", 0.0),
                    side=t.get("side", 1),
                    pre_trade_mid=t.get("pre_mid"),
                    post_trade_mid=t.get("post_mid"),
                )

        dataset = ImpactCalibrationDataset(
            observations=self._trade_observations,
            symbol=self._symbol,
            avg_adv=self._avg_adv,
            avg_volatility=self._avg_volatility,
        )

        # Run calibration
        results = self._impact_pipeline.calibrate_all(dataset)

        # Return best model (Almgren-Chriss by default)
        if ImpactType.ALMGREN_CHRISS in results:
            return results[ImpactType.ALMGREN_CHRISS]
        elif results:
            return next(iter(results.values()))

        # Return default if no data
        return ImpactCalibrationResult(
            model_type=ImpactType.ALMGREN_CHRISS,
            parameters={"eta": 0.05, "gamma": 0.03, "delta": 0.5},
        )

    def calibrate_fill_probability(
        self,
        orders: Optional[Sequence[Dict[str, Any]]] = None,
        fills: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> FillCalibrationResult:
        """
        Calibrate fill probability parameters.

        Uses MLE for Poisson and Queue-Reactive models.

        Args:
            orders: Optional order data
            fills: Optional fill data

        Returns:
            FillCalibrationResult with calibrated parameters
        """
        # Add additional data if provided
        if orders:
            for o in orders:
                self.add_order(
                    timestamp_ns=o.get("timestamp_ns", 0),
                    price=o.get("price", 0.0),
                    qty=o.get("qty", 0.0),
                    side=Side.BUY if o.get("side", 1) > 0 else Side.SELL,
                    event_type=o.get("event_type", "ADD"),
                    fill_qty=o.get("fill_qty", 0.0),
                    queue_position=o.get("queue_position"),
                )

        # Run calibration
        results = self._fill_prob_pipeline.run_calibration()

        # Return queue-reactive result by default
        if "queue_reactive" in results:
            return results["queue_reactive"]
        elif "poisson" in results:
            return results["poisson"]

        return FillCalibrationResult(
            model_type=FillModelType.QUEUE_REACTIVE,
            parameters={"base_rate": 100.0, "queue_decay_alpha": 0.01},
            metrics={},
            n_samples=0,
            time_range_sec=0.0,
        )

    def calibrate_latency(self) -> LatencyCalibrationResult:
        """
        Calibrate latency distribution parameters.

        Returns:
            LatencyCalibrationResult with fitted distributions
        """
        return self._latency_calibrator.fit()

    def calibrate_queue_dynamics(self) -> QueueDynamicsResult:
        """
        Calibrate queue dynamics parameters.

        Returns:
            QueueDynamicsResult with calibrated parameters
        """
        return self._queue_calibrator.fit()

    def run_full_calibration(
        self,
        data_path: Optional[str] = None,
    ) -> L3ExecutionConfig:
        """
        Run complete calibration pipeline.

        Calibrates all parameters and returns ready-to-use L3ExecutionConfig.

        Args:
            data_path: Optional path to calibration data file

        Returns:
            L3ExecutionConfig with calibrated parameters
        """
        # Load data from file if provided
        if data_path:
            self._load_calibration_data(data_path)

        # Run all calibrations
        impact_result = self.calibrate_impact()
        fill_result = self.calibrate_fill_probability()
        latency_result = self.calibrate_latency()
        queue_result = self.calibrate_queue_dynamics()

        # Build L3ExecutionConfig from results
        config = self._build_config(
            impact_result,
            fill_result,
            latency_result,
            queue_result,
        )

        return config

    def get_calibration_result(self) -> L3CalibrationResult:
        """
        Get detailed calibration results.

        Returns:
            L3CalibrationResult with all calibration details
        """
        # Run calibrations
        impact_result = self.calibrate_impact()
        fill_result = self.calibrate_fill_probability()
        latency_result = self.calibrate_latency()
        queue_result = self.calibrate_queue_dynamics()

        # Compute data duration
        all_times = [o.timestamp_ms for o in self._trade_observations]
        if len(all_times) >= 2:
            duration_sec = (max(all_times) - min(all_times)) / 1000.0
        else:
            duration_sec = 0.0

        # Assess calibration quality
        if self.n_trades >= 1000 and self.n_orders >= 500:
            quality = "high"
        elif self.n_trades >= 100 or self.n_orders >= 100:
            quality = "medium"
        else:
            quality = "low"

        # Compute confidence intervals (95% CI)
        confidence_intervals = self._compute_confidence_intervals(impact_result, latency_result)

        return L3CalibrationResult(
            impact_result=impact_result,
            fill_prob_result=fill_result,
            latency_result=latency_result,
            queue_dynamics=queue_result,
            n_trades=self.n_trades,
            n_orders=self.n_orders,
            data_duration_sec=duration_sec,
            calibration_quality=quality,
            confidence_intervals=confidence_intervals,
        )

    def _compute_confidence_intervals(
        self,
        impact_result: Optional[ImpactCalibrationResult],
        latency_result: Optional[LatencyCalibrationResult],
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute 95% confidence intervals for calibrated parameters.

        Uses bootstrap or analytical methods depending on sample size.

        Args:
            impact_result: Impact calibration result
            latency_result: Latency calibration result

        Returns:
            Dictionary mapping parameter names to (lower, upper) bounds
        """
        ci: Dict[str, Tuple[float, float]] = {}
        n = self.n_trades

        if n < 10:
            return ci  # Not enough data for meaningful CI

        # Z-score for 95% CI
        z = 1.96

        # Impact parameter CIs (using standard errors from regression)
        if impact_result and impact_result.parameters:
            # For OLS regression, SE = RMSE / sqrt(n)
            se_factor = (impact_result.rmse / math.sqrt(n)) if n > 0 else 0.1

            for param, value in impact_result.parameters.items():
                # Approximate SE as fraction of parameter value
                # More accurate would use Hessian from MLE
                se = max(abs(value) * 0.1, se_factor)
                ci[f"impact_{param}"] = (value - z * se, value + z * se)

        # Latency parameter CIs (using log-normal CI formula)
        if latency_result and latency_result.n_samples > 0:
            n_lat = latency_result.n_samples

            for name, params in [
                ("feed", latency_result.feed_latency),
                ("order", latency_result.order_latency),
                ("exchange", latency_result.exchange_latency),
                ("fill", latency_result.fill_latency),
            ]:
                if params.mean_us > 0 and params.std_us > 0:
                    # CI for mean of log-normal: use log-transformed CI
                    se_mean = params.std_us / math.sqrt(max(1, n_lat // 4))
                    ci[f"latency_{name}_mean_us"] = (
                        max(0, params.mean_us - z * se_mean),
                        params.mean_us + z * se_mean,
                    )

        return ci

    def _load_calibration_data(self, path: str) -> None:
        """Load calibration data from file."""
        import json
        from pathlib import Path

        file_path = Path(path)
        if not file_path.exists():
            logger.warning(f"Calibration data file not found: {path}")
            return

        with open(file_path, "r") as f:
            data = json.load(f)

        # Load trades
        for t in data.get("trades", []):
            self.add_trade(
                timestamp_ms=t.get("timestamp_ms", 0),
                price=t.get("price", 0.0),
                qty=t.get("qty", 0.0),
                side=t.get("side", 1),
                pre_trade_mid=t.get("pre_mid"),
                post_trade_mid=t.get("post_mid"),
            )

        # Load orders
        for o in data.get("orders", []):
            self.add_order(
                timestamp_ns=o.get("timestamp_ns", 0),
                price=o.get("price", 0.0),
                qty=o.get("qty", 0.0),
                side=Side.BUY if o.get("side", 1) > 0 else Side.SELL,
                event_type=o.get("event_type", "ADD"),
                fill_qty=o.get("fill_qty", 0.0),
            )

        # Load latency observations
        for l in data.get("latencies", []):
            self.add_latency_observation(
                timestamp_ns=l.get("timestamp_ns", 0),
                latency_type=l.get("type", "order"),
                latency_us=l.get("latency_us", 100.0),
            )

        # Load market params
        if "market_params" in data:
            self.set_market_params(
                avg_adv=data["market_params"].get("avg_adv", 10_000_000.0),
                avg_volatility=data["market_params"].get("avg_volatility", 0.02),
            )

    def _build_config(
        self,
        impact_result: ImpactCalibrationResult,
        fill_result: FillCalibrationResult,
        latency_result: LatencyCalibrationResult,
        queue_result: QueueDynamicsResult,
    ) -> L3ExecutionConfig:
        """Build L3ExecutionConfig from calibration results."""
        # Impact config
        impact_params = impact_result.parameters
        impact_config = MarketImpactConfig(
            enabled=True,
            model=ImpactModelType.ALMGREN_CHRISS,
            eta=impact_params.get("eta", 0.05),
            gamma=impact_params.get("gamma", 0.03),
            delta=impact_params.get("delta", 0.5),
            tau_ms=impact_params.get("tau_ms", 30000.0),
            apply_to_lob=True,
        )

        # Fill probability config
        fill_params = fill_result.parameters
        fill_config = FillProbabilityConfig(
            enabled=True,
            model=FillProbabilityModelType.QUEUE_REACTIVE,
            base_rate=fill_params.get("arrival_rate", fill_params.get("base_rate", 100.0)),
            queue_decay_alpha=fill_params.get("queue_decay_alpha", 0.01),
            spread_sensitivity_beta=fill_params.get("spread_sensitivity_beta", 0.5),
        )

        # Latency config
        latency_config = LatencyConfig(
            enabled=len(self._latency_observations) > 0,
            profile=LatencyProfileType.INSTITUTIONAL,
            feed_latency=self._latency_params_to_config(latency_result.feed_latency),
            order_latency=self._latency_params_to_config(latency_result.order_latency),
            exchange_latency=self._latency_params_to_config(latency_result.exchange_latency),
            fill_latency=self._latency_params_to_config(latency_result.fill_latency),
        )

        # Queue tracking config
        queue_config = QueueTrackingConfig(
            enabled=True,
            estimation_method="probabilistic",
        )

        # Build full config
        if self._asset_class == "crypto":
            base_config = L3ExecutionConfig.for_crypto()
        else:
            base_config = L3ExecutionConfig.for_equity()

        # Override with calibrated values
        base_config.market_impact = impact_config
        base_config.fill_probability = fill_config
        base_config.latency = latency_config
        base_config.queue_tracking = queue_config

        return base_config

    def _latency_params_to_config(
        self,
        params: LatencyDistributionParams,
    ) -> LatencyComponentConfig:
        """Convert LatencyDistributionParams to LatencyComponentConfig."""
        return LatencyComponentConfig(
            enabled=True,
            distribution=params.distribution,
            mean_us=params.mean_us,
            std_us=params.std_us,
            min_us=params.min_us,
            max_us=params.max_us,
            pareto_alpha=params.pareto_alpha,
            pareto_xmin_us=params.pareto_xmin_us,
        )

    def clear(self) -> None:
        """Clear all calibration data."""
        self._trade_observations.clear()
        self._trade_records.clear()
        self._order_records.clear()
        self._latency_observations.clear()
        self._quote_observations.clear()
        self._fill_prob_pipeline.clear()
        self._latency_calibrator.clear()
        self._queue_calibrator.clear()


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_calibration_pipeline(
    symbol: str = "",
    asset_class: str = "equity",
) -> L3CalibrationPipeline:
    """
    Create L3 calibration pipeline.

    Args:
        symbol: Trading symbol
        asset_class: "equity" or "crypto"

    Returns:
        L3CalibrationPipeline instance
    """
    return L3CalibrationPipeline(symbol=symbol, asset_class=asset_class)


def calibrate_from_dataframe(
    df: Any,  # pandas DataFrame
    symbol: str = "",
    asset_class: str = "equity",
    price_col: str = "price",
    qty_col: str = "qty",
    side_col: str = "side",
    timestamp_col: str = "timestamp",
) -> L3ExecutionConfig:
    """
    Calibrate L3 parameters from pandas DataFrame.

    Args:
        df: DataFrame with trade data
        symbol: Trading symbol
        asset_class: "equity" or "crypto"
        price_col: Price column name
        qty_col: Quantity column name
        side_col: Side column name
        timestamp_col: Timestamp column name

    Returns:
        Calibrated L3ExecutionConfig
    """
    pipeline = L3CalibrationPipeline(symbol=symbol, asset_class=asset_class)

    for _, row in df.iterrows():
        pipeline.add_trade(
            timestamp_ms=int(row.get(timestamp_col, 0)),
            price=float(row.get(price_col, 0.0)),
            qty=float(row.get(qty_col, 0.0)),
            side=int(row.get(side_col, 1)),
        )

    return pipeline.run_full_calibration()


def calibrate_from_alpaca(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    feed: str = "iex",
) -> L3ExecutionConfig:
    """
    Calibrate L3 parameters using Alpaca historical data.

    This function fetches historical quotes and trades from Alpaca
    and runs the full calibration pipeline for equity assets.

    Args:
        symbol: Stock symbol (e.g., "AAPL")
        start_date: Start date (ISO format, e.g., "2025-01-01")
        end_date: End date (ISO format)
        api_key: Alpaca API key (uses env var if not provided)
        api_secret: Alpaca API secret (uses env var if not provided)
        feed: Data feed ("iex" or "sip")

    Returns:
        Calibrated L3ExecutionConfig for the equity

    Example:
        config = calibrate_from_alpaca(
            "AAPL",
            start_date="2025-01-01",
            end_date="2025-01-31"
        )
        provider = L3ExecutionProvider(config=config)
    """
    import os

    from lob.data_adapters import AlpacaL2Adapter

    # Create adapter with credentials
    config = {
        "api_key": api_key or os.environ.get("ALPACA_API_KEY", ""),
        "api_secret": api_secret or os.environ.get("ALPACA_API_SECRET", ""),
        "feed": feed,
    }

    adapter = AlpacaL2Adapter(symbol=symbol, config=config)

    # Get calibration data from adapter
    cal_data = adapter.to_calibration_pipeline_data(symbol, start_date, end_date)

    if "error" in cal_data:
        logger.warning(f"Failed to get calibration data for {symbol}: {cal_data.get('error')}")
        # Return default equity config
        return L3ExecutionConfig.for_equity()

    # Create pipeline and add data
    pipeline = L3CalibrationPipeline(symbol=symbol, asset_class="equity")

    # Set market params
    market_params = cal_data.get("market_params", {})
    pipeline.set_market_params(
        avg_adv=market_params.get("avg_adv", 10_000_000),
        avg_volatility=market_params.get("avg_volatility", 0.02),
    )

    # Add trades
    for trade in cal_data.get("trades", []):
        pipeline.add_trade(
            timestamp_ms=trade.get("timestamp_ms", 0),
            price=trade.get("price", 0.0),
            qty=trade.get("qty", 0.0),
            side=trade.get("side", 1),
            pre_trade_mid=trade.get("pre_mid"),
            post_trade_mid=trade.get("post_mid"),
        )

    # Run calibration
    return pipeline.run_full_calibration()


def create_equity_calibration_pipeline(
    symbol: str,
    avg_adv: float = 10_000_000,
    avg_volatility: float = 0.02,
) -> L3CalibrationPipeline:
    """
    Create a calibration pipeline pre-configured for equity assets.

    This function creates a pipeline with equity-specific defaults:
    - Lower default spread (2 bps vs 5 bps for crypto)
    - Lower impact coefficients
    - Equity-typical latency profiles

    Args:
        symbol: Stock symbol
        avg_adv: Average daily volume (default: 10M shares)
        avg_volatility: Average volatility (default: 2%)

    Returns:
        L3CalibrationPipeline ready for data loading
    """
    pipeline = L3CalibrationPipeline(symbol=symbol, asset_class="equity")
    pipeline.set_market_params(avg_adv=avg_adv, avg_volatility=avg_volatility)
    return pipeline
