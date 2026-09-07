"""
Calibration Pipeline for LOB Fill Probability Models.

This module provides tools for fitting fill probability model parameters
from historical LOB data:
    1. Execution rate estimation (Poisson λ parameter)
    2. Queue-reactive model coefficient fitting
    3. Adverse selection parameter estimation
    4. Cross-validation for model selection

Supported Data Formats:
    - LOBSTER message files
    - Trade and order CSVs
    - Custom DataFrame formats

References:
    - Cont, Stoikov & Talreja (2010): Poisson model calibration
    - Huang et al. (2015): Queue-reactive intensity estimation
    - Glosten & Harris (1988): Adverse selection estimation

Performance Target: O(n) calibration where n = number of historical events
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
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
    Side,
    Trade,
)
from lob.fill_probability import (
    AnalyticalPoissonModel,
    FillProbabilityModel,
    FillProbabilityModelType,
    HistoricalFillRate,
    HistoricalRateModel,
    LOBState,
    QueueReactiveModel,
)
from lob.queue_tracker import (
    LevelStatistics,
)


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass
class TradeRecord:
    """
    Historical trade record for calibration.

    Attributes:
        timestamp_ns: Trade timestamp in nanoseconds
        price: Trade price
        qty: Trade quantity
        side: Aggressor side
        maker_queue_position: Maker's queue position (if known)
        time_in_queue_sec: Time maker was in queue (if known)
    """

    timestamp_ns: int
    price: float
    qty: float
    side: Side
    maker_queue_position: Optional[int] = None
    time_in_queue_sec: Optional[float] = None


@dataclass
class OrderRecord:
    """
    Historical order record for calibration.

    Attributes:
        timestamp_ns: Order submission time
        price: Limit price
        qty: Order quantity
        side: Order side
        event_type: ADD, CANCEL, MODIFY, or FILL
        fill_qty: Filled quantity (for FILL events)
        queue_position: Queue position at submission
    """

    timestamp_ns: int
    price: float
    qty: float
    side: Side
    event_type: str  # "ADD", "CANCEL", "MODIFY", "FILL"
    fill_qty: float = 0.0
    queue_position: Optional[int] = None


@dataclass
class CalibrationResult:
    """
    Result of model calibration.

    Attributes:
        model_type: Type of model calibrated
        parameters: Fitted parameter values
        metrics: Calibration quality metrics
        n_samples: Number of samples used
        time_range_sec: Time range of data
        details: Additional details
    """

    model_type: FillProbabilityModelType
    parameters: Dict[str, float]
    metrics: Dict[str, float]
    n_samples: int
    time_range_sec: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossValidationResult:
    """
    Result of cross-validation.

    Attributes:
        n_folds: Number of folds
        train_scores: Training set scores per fold
        test_scores: Test set scores per fold
        mean_train_score: Mean training score
        mean_test_score: Mean test score
        std_test_score: Std of test scores
        best_params: Best parameters found
    """

    n_folds: int
    train_scores: List[float]
    test_scores: List[float]
    mean_train_score: float
    mean_test_score: float
    std_test_score: float
    best_params: Dict[str, float] = field(default_factory=dict)


# ==============================================================================
# Base Calibrator
# ==============================================================================


class BaseCalibrator:
    """
    Base class for model calibrators.

    Provides common functionality for:
        - Data preprocessing
        - Metric computation
        - Cross-validation
    """

    def __init__(self) -> None:
        self._trades: List[TradeRecord] = []
        self._orders: List[OrderRecord] = []
        self._is_fitted: bool = False

    def add_trade(self, trade: TradeRecord) -> None:
        """Add a trade record for calibration."""
        self._trades.append(trade)

    def add_trades(self, trades: List[TradeRecord]) -> None:
        """Add multiple trade records."""
        self._trades.extend(trades)

    def add_order(self, order: OrderRecord) -> None:
        """Add an order record for calibration."""
        self._orders.append(order)

    def add_orders(self, orders: List[OrderRecord]) -> None:
        """Add multiple order records."""
        self._orders.extend(orders)

    def clear(self) -> None:
        """Clear all data."""
        self._trades.clear()
        self._orders.clear()
        self._is_fitted = False

    @property
    def n_trades(self) -> int:
        """Number of trade records."""
        return len(self._trades)

    @property
    def n_orders(self) -> int:
        """Number of order records."""
        return len(self._orders)

    def _compute_time_range(self) -> float:
        """Compute time range of data in seconds."""
        all_times = [t.timestamp_ns for t in self._trades] + [o.timestamp_ns for o in self._orders]
        if not all_times:
            return 0.0
        return (max(all_times) - min(all_times)) / 1e9


# ==============================================================================
# Poisson Rate Calibrator
# ==============================================================================


class PoissonRateCalibrator(BaseCalibrator):
    """
    Calibrates Poisson arrival rate parameters.

    Estimates:
        - λ: Volume arrival rate (qty/sec)
        - Per-level rates for different price levels

    Method:
        Maximum Likelihood Estimation (MLE) for Poisson rate:
        λ_MLE = N / T where N = total volume, T = time period
    """

    def __init__(self) -> None:
        super().__init__()
        self._level_rates: Dict[Tuple[float, Side], float] = {}
        self._global_rate: float = 0.0

    def fit(self) -> CalibrationResult:
        """
        Fit Poisson rate from trade data.

        Returns:
            CalibrationResult with fitted parameters
        """
        if not self._trades:
            return CalibrationResult(
                model_type=FillProbabilityModelType.POISSON,
                parameters={"arrival_rate": 100.0},  # Default
                metrics={"n_samples": 0},
                n_samples=0,
                time_range_sec=0.0,
            )

        # Compute time range
        time_range = self._compute_time_range()
        if time_range <= 0:
            time_range = 1.0  # Prevent division by zero

        # Global rate: total volume / time
        total_volume = sum(t.qty for t in self._trades)
        self._global_rate = total_volume / time_range

        # Per-level rates
        level_volumes: Dict[Tuple[float, Side], float] = {}
        for trade in self._trades:
            key = (trade.price, trade.side)
            level_volumes[key] = level_volumes.get(key, 0.0) + trade.qty

        for key, volume in level_volumes.items():
            self._level_rates[key] = volume / time_range

        # Compute metrics
        metrics = self._compute_metrics(time_range)

        self._is_fitted = True

        return CalibrationResult(
            model_type=FillProbabilityModelType.POISSON,
            parameters={
                "arrival_rate": self._global_rate,
                "n_price_levels": len(self._level_rates),
            },
            metrics=metrics,
            n_samples=len(self._trades),
            time_range_sec=time_range,
            details={"level_rates": dict(self._level_rates)},
        )

    def _compute_metrics(self, time_range: float) -> Dict[str, float]:
        """Compute calibration quality metrics."""
        if not self._trades:
            return {}

        # Compute inter-arrival times
        sorted_trades = sorted(self._trades, key=lambda t: t.timestamp_ns)
        inter_arrivals = []
        for i in range(1, len(sorted_trades)):
            dt = (sorted_trades[i].timestamp_ns - sorted_trades[i - 1].timestamp_ns) / 1e9
            if dt > 0:
                inter_arrivals.append(dt)

        if not inter_arrivals:
            return {"arrival_rate": self._global_rate}

        # For Poisson process, inter-arrival times should be exponential
        # Mean should equal 1/λ
        mean_inter_arrival = sum(inter_arrivals) / len(inter_arrivals)
        implied_rate = 1.0 / mean_inter_arrival if mean_inter_arrival > 0 else 0.0

        # Coefficient of variation (should be ~1 for exponential)
        std_inter_arrival = math.sqrt(
            sum((x - mean_inter_arrival) ** 2 for x in inter_arrivals) / len(inter_arrivals)
        )
        cv = std_inter_arrival / mean_inter_arrival if mean_inter_arrival > 0 else 0.0

        return {
            "arrival_rate": self._global_rate,
            "implied_rate_from_inter_arrivals": implied_rate,
            "cv_inter_arrivals": cv,
            "n_trades": len(self._trades),
            "time_range_sec": time_range,
        }

    def get_model(self) -> AnalyticalPoissonModel:
        """Get calibrated Poisson model."""
        return AnalyticalPoissonModel(
            default_arrival_rate=self._global_rate if self._is_fitted else 100.0
        )

    def get_rate_at_level(self, price: float, side: Side) -> float:
        """Get calibrated rate at a specific price level."""
        key = (price, side)
        return self._level_rates.get(key, self._global_rate)


# ==============================================================================
# Queue-Reactive Model Calibrator
# ==============================================================================


class QueueReactiveCalibrator(BaseCalibrator):
    """
    Calibrates Queue-Reactive model parameters.

    Estimates:
        - α (queue_decay): Queue size impact
        - β (spread_sensitivity): Spread impact
        - γ (volatility_sensitivity): Volatility impact
        - δ (imbalance_sensitivity): Imbalance impact

    Method:
        Grid search or gradient descent to minimize prediction error.
    """

    def __init__(
        self,
        param_grid: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        """
        Initialize calibrator.

        Args:
            param_grid: Grid of parameter values to search
        """
        super().__init__()

        # Default parameter grid
        self._param_grid = param_grid or {
            "queue_decay_alpha": [0.001, 0.01, 0.05, 0.1],
            "spread_sensitivity_beta": [0.1, 0.3, 0.5, 0.7],
            "volatility_sensitivity_gamma": [0.1, 0.2, 0.3, 0.5],
            "imbalance_sensitivity_delta": [0.1, 0.2, 0.3],
        }

        self._best_params: Dict[str, float] = {}
        self._base_rate: float = 100.0

        # Additional state data for fitting
        self._state_data: List[Dict[str, float]] = []

    def add_state_observation(
        self,
        timestamp_ns: int,
        queue_size: float,
        spread_bps: float,
        volatility: float,
        imbalance: float,
        executed_qty: float,
    ) -> None:
        """
        Add a market state observation for calibration.

        Args:
            timestamp_ns: Observation time
            queue_size: Queue size at observation
            spread_bps: Spread in basis points
            volatility: Volatility measure
            imbalance: Book imbalance [-1, 1]
            executed_qty: Volume executed after observation
        """
        self._state_data.append(
            {
                "timestamp_ns": timestamp_ns,
                "queue_size": queue_size,
                "spread_bps": spread_bps,
                "volatility": volatility,
                "imbalance": imbalance,
                "executed_qty": executed_qty,
            }
        )

    def fit(
        self,
        method: str = "grid_search",
    ) -> CalibrationResult:
        """
        Fit Queue-Reactive model parameters.

        Args:
            method: "grid_search" or "mle"

        Returns:
            CalibrationResult with fitted parameters
        """
        # First estimate base rate from trades
        if self._trades:
            time_range = self._compute_time_range()
            if time_range > 0:
                total_volume = sum(t.qty for t in self._trades)
                self._base_rate = total_volume / time_range

        # If we have state observations, fit coefficients
        if self._state_data:
            if method == "grid_search":
                self._best_params = self._grid_search_fit()
            else:
                self._best_params = self._mle_fit()
        else:
            # Use defaults
            self._best_params = {
                "base_rate": self._base_rate,
                "queue_decay_alpha": 0.01,
                "spread_sensitivity_beta": 0.5,
                "volatility_sensitivity_gamma": 0.3,
                "imbalance_sensitivity_delta": 0.2,
            }

        self._is_fitted = True

        return CalibrationResult(
            model_type=FillProbabilityModelType.QUEUE_REACTIVE,
            parameters=self._best_params,
            metrics={"n_observations": len(self._state_data)},
            n_samples=len(self._trades) + len(self._state_data),
            time_range_sec=self._compute_time_range(),
        )

    def _grid_search_fit(self) -> Dict[str, float]:
        """Fit parameters via grid search."""
        best_error = float("inf")
        best_params: Dict[str, float] = {}

        # Generate all parameter combinations
        from itertools import product

        param_names = list(self._param_grid.keys())
        param_values = list(self._param_grid.values())

        for values in product(*param_values):
            params = dict(zip(param_names, values))
            params["base_rate"] = self._base_rate

            # Compute prediction error
            error = self._compute_prediction_error(params)

            if error < best_error:
                best_error = error
                best_params = params.copy()

        return best_params

    def _mle_fit(self) -> Dict[str, float]:
        """Fit parameters via Maximum Likelihood Estimation."""
        # Simplified MLE using moment matching
        # For production, would use scipy.optimize

        params = {
            "base_rate": self._base_rate,
            "queue_decay_alpha": 0.01,
            "spread_sensitivity_beta": 0.5,
            "volatility_sensitivity_gamma": 0.3,
            "imbalance_sensitivity_delta": 0.2,
        }

        if not self._state_data:
            return params

        # Estimate α from queue size correlation
        queue_sizes = [s["queue_size"] for s in self._state_data]
        volumes = [s["executed_qty"] for s in self._state_data]

        if queue_sizes and volumes:
            # Negative correlation between queue size and execution rate
            mean_q = sum(queue_sizes) / len(queue_sizes)
            mean_v = sum(volumes) / len(volumes)

            cov = sum((q - mean_q) * (v - mean_v) for q, v in zip(queue_sizes, volumes))
            var_q = sum((q - mean_q) ** 2 for q in queue_sizes)

            if var_q > 0:
                # α = -cov(q, v) / (var(q) * mean(v))
                alpha = -cov / (var_q * mean_v) if mean_v > 0 else 0.01
                params["queue_decay_alpha"] = max(0.001, min(0.1, alpha))

        return params

    def _compute_prediction_error(self, params: Dict[str, float]) -> float:
        """Compute prediction error for given parameters."""
        if not self._state_data:
            return float("inf")

        model = QueueReactiveModel(
            base_rate=params.get("base_rate", self._base_rate),
            queue_decay_alpha=params.get("queue_decay_alpha", 0.01),
            spread_sensitivity_beta=params.get("spread_sensitivity_beta", 0.5),
            volatility_sensitivity_gamma=params.get("volatility_sensitivity_gamma", 0.3),
            imbalance_sensitivity_delta=params.get("imbalance_sensitivity_delta", 0.2),
        )

        total_error = 0.0
        for obs in self._state_data:
            # Create market state
            market_state = LOBState(
                spread_bps=obs["spread_bps"],
                volatility=obs["volatility"],
                imbalance=obs["imbalance"],
            )

            # Predict rate
            predicted_rate, _ = model.compute_adjusted_rate(
                qty_ahead=obs["queue_size"],
                market_state=market_state,
            )

            # Compare to actual (assuming 1 second observation window)
            actual_rate = obs["executed_qty"]
            error = (predicted_rate - actual_rate) ** 2
            total_error += error

        return total_error / len(self._state_data)

    def get_model(self) -> QueueReactiveModel:
        """Get calibrated Queue-Reactive model."""
        return QueueReactiveModel(
            base_rate=self._best_params.get("base_rate", self._base_rate),
            queue_decay_alpha=self._best_params.get("queue_decay_alpha", 0.01),
            spread_sensitivity_beta=self._best_params.get("spread_sensitivity_beta", 0.5),
            volatility_sensitivity_gamma=self._best_params.get("volatility_sensitivity_gamma", 0.3),
            imbalance_sensitivity_delta=self._best_params.get("imbalance_sensitivity_delta", 0.2),
        )


# ==============================================================================
# Adverse Selection Calibrator
# ==============================================================================


class AdverseSelectionCalibrator(BaseCalibrator):
    """
    Calibrates adverse selection parameters.

    Estimates:
        - Informed trader fraction (PIN-like)
        - Average adverse move size
        - Probability of adverse move given fill

    Method:
        Roll decomposition or sequential trade analysis
    """

    def __init__(self) -> None:
        super().__init__()
        self._price_changes: List[Tuple[int, float]] = []  # (direction, magnitude)
        self._informed_fraction: float = 0.2  # Default
        self._adverse_move_bps: float = 5.0  # Default

    def add_price_observation(
        self,
        pre_trade_mid: float,
        post_trade_mid: float,
        trade_side: Side,
        time_delta_sec: float = 1.0,
    ) -> None:
        """
        Add a price observation for adverse selection estimation.

        Args:
            pre_trade_mid: Mid price before trade
            post_trade_mid: Mid price after trade (e.g., 1 second later)
            trade_side: Side of the trade (BUY or SELL)
            time_delta_sec: Time between observations
        """
        if pre_trade_mid <= 0:
            return

        # Price change in bps
        change_bps = (post_trade_mid - pre_trade_mid) / pre_trade_mid * 10000

        # Adverse move: price moves against the maker
        # If taker was BUY, adverse for maker (seller) is price going UP
        # If taker was SELL, adverse for maker (buyer) is price going DOWN
        direction = 1 if trade_side == Side.BUY else -1
        adverse_change = change_bps * direction

        self._price_changes.append((direction, adverse_change))

    def fit(self) -> CalibrationResult:
        """
        Fit adverse selection parameters.

        Uses the Roll model decomposition:
            - Informed trades move price in their direction
            - Uninformed trades have random price impact
        """
        if not self._price_changes:
            return CalibrationResult(
                model_type=FillProbabilityModelType.HISTORICAL,
                parameters={
                    "informed_fraction": 0.2,
                    "adverse_move_bps": 5.0,
                    "adverse_probability": 0.3,
                },
                metrics={},
                n_samples=0,
                time_range_sec=0.0,
            )

        # Count adverse moves
        adverse_moves = [change for _, change in self._price_changes if change > 0]
        favorable_moves = [change for _, change in self._price_changes if change < 0]

        n_total = len(self._price_changes)
        n_adverse = len(adverse_moves)

        # Probability of adverse move
        prob_adverse = n_adverse / n_total if n_total > 0 else 0.3

        # Average adverse move size
        avg_adverse_bps = sum(adverse_moves) / n_adverse if adverse_moves else 5.0

        # Estimate informed fraction using Roll decomposition
        # Informed traders: trades followed by consistent price moves
        # PIN = P(informed) ≈ (adverse - favorable) / total
        informed_signal = n_adverse - len(favorable_moves)
        self._informed_fraction = max(
            0.05, min(0.5, informed_signal / n_total if n_total > 0 else 0.2)
        )

        self._adverse_move_bps = avg_adverse_bps

        self._is_fitted = True

        return CalibrationResult(
            model_type=FillProbabilityModelType.HISTORICAL,
            parameters={
                "informed_fraction": self._informed_fraction,
                "adverse_move_bps": self._adverse_move_bps,
                "adverse_probability": prob_adverse,
            },
            metrics={
                "n_observations": n_total,
                "n_adverse": n_adverse,
                "n_favorable": len(favorable_moves),
            },
            n_samples=n_total,
            time_range_sec=self._compute_time_range(),
        )


# ==============================================================================
# Historical Rate Calibrator
# ==============================================================================


class HistoricalRateCalibrator(BaseCalibrator):
    """
    Calibrates historical fill rates per price level.

    Creates a lookup table of fill rates that can be used
    by HistoricalRateModel for accurate per-level estimates.
    """

    def __init__(
        self,
        bucket_size_bps: float = 5.0,  # Price bucket size
    ) -> None:
        """
        Initialize calibrator.

        Args:
            bucket_size_bps: Size of price buckets in basis points
        """
        super().__init__()
        self._bucket_size_bps = bucket_size_bps

        # Fill rates by (bucket, side)
        self._fill_data: Dict[Tuple[int, Side], List[Tuple[float, float]]] = {}

    def _price_to_bucket(self, price: float, mid_price: float) -> int:
        """Convert price to bucket index (distance from mid in bps)."""
        if mid_price <= 0:
            return 0
        distance_bps = abs(price - mid_price) / mid_price * 10000
        return int(distance_bps / self._bucket_size_bps)

    def add_fill_observation(
        self,
        price: float,
        mid_price: float,
        side: Side,
        fill_qty: float,
        time_in_queue_sec: float,
    ) -> None:
        """
        Add a fill observation for calibration.

        Args:
            price: Fill price
            mid_price: Mid price at time of fill
            side: Order side
            fill_qty: Quantity filled
            time_in_queue_sec: Time order was in queue before fill
        """
        bucket = self._price_to_bucket(price, mid_price)
        key = (bucket, side)

        if key not in self._fill_data:
            self._fill_data[key] = []

        # Store (qty, time) for rate calculation
        self._fill_data[key].append((fill_qty, time_in_queue_sec))

    def fit(self) -> CalibrationResult:
        """
        Fit historical fill rates.

        Returns:
            CalibrationResult with per-level rates
        """
        rates: Dict[Tuple[int, Side], HistoricalFillRate] = {}

        for key, data in self._fill_data.items():
            bucket, side = key

            if not data:
                continue

            # Calculate average fill rate
            total_qty = sum(qty for qty, _ in data)
            total_time = sum(time for _, time in data if time > 0)

            if total_time > 0:
                avg_rate = total_qty / total_time
            else:
                avg_rate = 10.0  # Default

            # Calculate standard deviation
            times = [time for _, time in data if time > 0]
            if times:
                mean_time = total_time / len(times)
                var_time = sum((t - mean_time) ** 2 for t in times) / len(times)
                std_rate = math.sqrt(var_time) * avg_rate / mean_time if mean_time > 0 else 5.0
            else:
                std_rate = 5.0

            rates[key] = HistoricalFillRate(
                price=bucket * self._bucket_size_bps,  # Store as distance in bps
                side=side,
                avg_fill_rate=avg_rate,
                fill_rate_std=std_rate,
                avg_time_to_fill=total_time / len(data) if data else 30.0,
                fill_count=len(data),
            )

        self._is_fitted = True

        return CalibrationResult(
            model_type=FillProbabilityModelType.HISTORICAL,
            parameters={"n_buckets": len(rates)},
            metrics={
                "total_fills": sum(len(d) for d in self._fill_data.values()),
                "n_price_buckets": len(rates),
            },
            n_samples=sum(len(d) for d in self._fill_data.values()),
            time_range_sec=self._compute_time_range(),
            details={"rates": {str(k): v for k, v in rates.items()}},
        )

    def get_model(self) -> HistoricalRateModel:
        """Get calibrated Historical Rate model."""
        model = HistoricalRateModel()

        for key, data in self._fill_data.items():
            bucket, side = key

            if not data:
                continue

            total_qty = sum(qty for qty, _ in data)
            total_time = sum(time for _, time in data if time > 0)

            if total_time > 0:
                rate = HistoricalFillRate(
                    price=bucket * self._bucket_size_bps,
                    side=side,
                    avg_fill_rate=total_qty / total_time,
                    fill_count=len(data),
                )
                model.add_historical_rate(rate)

        return model


# ==============================================================================
# Calibration Pipeline
# ==============================================================================


class CalibrationPipeline:
    """
    Complete calibration pipeline for fill probability models.

    Orchestrates:
        1. Data loading and preprocessing
        2. Model-specific calibration
        3. Cross-validation
        4. Model selection and combination
    """

    def __init__(self) -> None:
        self._poisson_calibrator = PoissonRateCalibrator()
        self._qr_calibrator = QueueReactiveCalibrator()
        self._adverse_calibrator = AdverseSelectionCalibrator()
        self._historical_calibrator = HistoricalRateCalibrator()

        self._calibration_results: Dict[str, CalibrationResult] = {}

    def add_trade(self, trade: TradeRecord) -> None:
        """Add trade to all calibrators."""
        self._poisson_calibrator.add_trade(trade)
        self._qr_calibrator.add_trade(trade)

    def add_trades(self, trades: List[TradeRecord]) -> None:
        """Add multiple trades."""
        for trade in trades:
            self.add_trade(trade)

    def add_order(self, order: OrderRecord) -> None:
        """Add order to relevant calibrators."""
        self._poisson_calibrator.add_order(order)
        self._qr_calibrator.add_order(order)

    def add_market_state(
        self,
        timestamp_ns: int,
        queue_size: float,
        spread_bps: float,
        volatility: float,
        imbalance: float,
        executed_qty: float,
    ) -> None:
        """Add market state observation."""
        self._qr_calibrator.add_state_observation(
            timestamp_ns, queue_size, spread_bps, volatility, imbalance, executed_qty
        )

    def add_price_observation(
        self,
        pre_trade_mid: float,
        post_trade_mid: float,
        trade_side: Side,
    ) -> None:
        """Add price observation for adverse selection."""
        self._adverse_calibrator.add_price_observation(pre_trade_mid, post_trade_mid, trade_side)

    def add_fill_observation(
        self,
        price: float,
        mid_price: float,
        side: Side,
        fill_qty: float,
        time_in_queue_sec: float,
    ) -> None:
        """Add fill observation for historical rate model."""
        self._historical_calibrator.add_fill_observation(
            price, mid_price, side, fill_qty, time_in_queue_sec
        )

    def run_calibration(self) -> Dict[str, CalibrationResult]:
        """
        Run full calibration pipeline.

        Returns:
            Dict of model name -> CalibrationResult
        """
        # Run each calibrator
        self._calibration_results["poisson"] = self._poisson_calibrator.fit()
        self._calibration_results["queue_reactive"] = self._qr_calibrator.fit()
        self._calibration_results["adverse_selection"] = self._adverse_calibrator.fit()
        self._calibration_results["historical"] = self._historical_calibrator.fit()

        return self._calibration_results

    def get_best_model(
        self,
        model_type: str = "queue_reactive",
    ) -> FillProbabilityModel:
        """
        Get the best calibrated model.

        Args:
            model_type: "poisson", "queue_reactive", or "historical"

        Returns:
            Calibrated FillProbabilityModel
        """
        if model_type == "poisson":
            return self._poisson_calibrator.get_model()
        elif model_type == "queue_reactive":
            return self._qr_calibrator.get_model()
        elif model_type == "historical":
            return self._historical_calibrator.get_model()
        else:
            # Default to queue-reactive
            return self._qr_calibrator.get_model()

    def cross_validate(
        self,
        n_folds: int = 5,
        model_type: str = "queue_reactive",
    ) -> CrossValidationResult:
        """
        Cross-validate model on historical data.

        Args:
            n_folds: Number of folds
            model_type: Model to validate

        Returns:
            CrossValidationResult
        """
        # Get all data
        all_trades = self._poisson_calibrator._trades[:]

        if len(all_trades) < n_folds:
            return CrossValidationResult(
                n_folds=n_folds,
                train_scores=[0.0],
                test_scores=[0.0],
                mean_train_score=0.0,
                mean_test_score=0.0,
                std_test_score=0.0,
            )

        # Sort by time
        all_trades.sort(key=lambda t: t.timestamp_ns)

        # Split into folds
        fold_size = len(all_trades) // n_folds
        train_scores: List[float] = []
        test_scores: List[float] = []

        for fold in range(n_folds):
            # Train on all but this fold
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < n_folds - 1 else len(all_trades)

            train_data = all_trades[:test_start] + all_trades[test_end:]
            test_data = all_trades[test_start:test_end]

            # Train
            calibrator = PoissonRateCalibrator()
            calibrator.add_trades(train_data)
            result = calibrator.fit()

            # Evaluate on train
            train_score = self._evaluate_model(
                calibrator.get_model(),
                train_data,
            )
            train_scores.append(train_score)

            # Evaluate on test
            test_score = self._evaluate_model(
                calibrator.get_model(),
                test_data,
            )
            test_scores.append(test_score)

        mean_test = sum(test_scores) / len(test_scores) if test_scores else 0.0
        std_test = (
            math.sqrt(sum((s - mean_test) ** 2 for s in test_scores) / len(test_scores))
            if test_scores
            else 0.0
        )

        return CrossValidationResult(
            n_folds=n_folds,
            train_scores=train_scores,
            test_scores=test_scores,
            mean_train_score=sum(train_scores) / len(train_scores) if train_scores else 0.0,
            mean_test_score=mean_test,
            std_test_score=std_test,
        )

    def _evaluate_model(
        self,
        model: FillProbabilityModel,
        trades: List[TradeRecord],
    ) -> float:
        """
        Evaluate model prediction accuracy.

        Returns log-likelihood score (higher is better).
        """
        if not trades:
            return 0.0

        # Compute average volume rate from trades data
        if len(trades) >= 2:
            time_span = (trades[-1].timestamp_ns - trades[0].timestamp_ns) / 1e9
            total_volume = sum(t.qty for t in trades)
            avg_volume_rate = total_volume / max(time_span, 1.0)
        else:
            avg_volume_rate = 100.0  # Default fallback

        # Compute average time in queue for time horizon estimation
        times_in_queue = [t.time_in_queue_sec for t in trades if t.time_in_queue_sec is not None]
        avg_time_horizon = sum(times_in_queue) / len(times_in_queue) if times_in_queue else 60.0

        log_likelihood = 0.0

        for trade in trades:
            # Create LOBState with actual trade info for accurate model evaluation
            market_state = LOBState(
                volume_rate=avg_volume_rate,
                mid_price=trade.price,  # Approximate mid as trade price
                order_side=trade.side,
                order_price=trade.price,
            )

            # Use actual time in queue if available, otherwise average
            time_horizon = (
                trade.time_in_queue_sec if trade.time_in_queue_sec is not None else avg_time_horizon
            )

            # Predict fill probability
            result = model.compute_fill_probability(
                queue_position=trade.maker_queue_position or 0,
                qty_ahead=0.0,  # Unknown
                order_qty=trade.qty,
                time_horizon_sec=time_horizon,
                market_state=market_state,
            )

            # Log-likelihood: log(p) if filled (which it was)
            p = max(0.001, result.prob_fill)  # Avoid log(0)
            log_likelihood += math.log(p)

        return log_likelihood / len(trades)  # Normalized

    def clear(self) -> None:
        """Clear all data from all calibrators."""
        self._poisson_calibrator.clear()
        self._qr_calibrator.clear()
        self._adverse_calibrator.clear()
        self._historical_calibrator.clear()
        self._calibration_results.clear()


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_calibration_pipeline() -> CalibrationPipeline:
    """Create a new calibration pipeline."""
    return CalibrationPipeline()


def calibrate_from_trades(
    trades: List[TradeRecord],
    model_type: str = "queue_reactive",
) -> FillProbabilityModel:
    """
    Quick calibration from trade list.

    Args:
        trades: List of historical trades
        model_type: Model to calibrate

    Returns:
        Calibrated model
    """
    pipeline = CalibrationPipeline()
    pipeline.add_trades(trades)
    pipeline.run_calibration()
    return pipeline.get_best_model(model_type)


def estimate_arrival_rate(
    trades: List[TradeRecord],
) -> float:
    """
    Quick estimation of arrival rate from trades.

    Args:
        trades: List of historical trades

    Returns:
        Estimated arrival rate (qty/sec)
    """
    if not trades:
        return 100.0  # Default

    calibrator = PoissonRateCalibrator()
    calibrator.add_trades(trades)
    result = calibrator.fit()

    return result.parameters.get("arrival_rate", 100.0)
