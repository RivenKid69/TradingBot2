# -*- coding: utf-8 -*-
"""
impl_conformal.py
Conformal prediction implementations.

Layer: IMPL (depends on core_conformal, numpy, torch)

References:
- CQR: Romano et al., "Conformalized Quantile Regression" (2019)
  https://arxiv.org/abs/1905.03222
- EnbPI: Xu & Xie, "Conformal Prediction Interval for Dynamic Time-Series" (ICML 2021)
- ACI: Gibbs & Candes, "Adaptive Conformal Inference Under Distribution Shift" (2021)
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from core_conformal import (
    CalibrationConfig,
    CalibrationResult,
    ConformalConfig,
    ConformalMethod,
    CVaRBounds,
    CVaRBoundsConfig,
    EscalationAction,
    EscalationConfig,
    EscalationLevel,
    PredictionInterval,
    RiskIntegrationConfig,
    TimeSeriesConfig,
    UncertaintyState,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================


def _now_ms() -> int:
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)


def _compute_quantile_with_correction(
    values: np.ndarray,
    alpha: float,
    n: int,
) -> float:
    """
    Compute quantile with finite-sample correction.

    The correction ensures valid coverage for finite samples.

    Args:
        values: Array of values to compute quantile from
        alpha: Desired coverage level (e.g., 0.90 for 90% coverage)
        n: Sample size

    Returns:
        Corrected quantile value
    """
    # Finite-sample correction: use (1-α)(1 + 1/n) quantile
    quantile_level = min(1.0, alpha * (1.0 + 1.0 / n))
    return float(np.quantile(values, quantile_level))


# =============================================================================
# CQR Calibrator
# =============================================================================


class CQRCalibrator:
    """
    Conformalized Quantile Regression calibrator.

    Reference: Romano et al., "Conformalized Quantile Regression" (2019)
    https://arxiv.org/abs/1905.03222

    Key insight: Instead of using raw quantile predictions, we calibrate
    the residuals to achieve guaranteed coverage.

    Algorithm:
    1. On calibration set: compute conformity scores
       E_i = max(q_lo_i - Y_i, Y_i - q_hi_i)
    2. Find Q_{1-α}(E) = (1-α)-quantile of conformity scores
    3. At inference: [q_lo - Q, q_hi + Q] has coverage >= 1-α

    The conformity score measures how much the true value exceeds the
    prediction interval. Positive when Y is outside [q_lo, q_hi].
    """

    def __init__(self, config: ConformalConfig):
        """
        Initialize CQR calibrator.

        Args:
            config: Conformal prediction configuration
        """
        self.config = config
        self._residuals: Optional[np.ndarray] = None
        self._calibration_quantile: Optional[float] = None
        self._calibration_samples: int = 0
        self._empirical_coverage: Optional[float] = None
        self._lock = threading.Lock()

    def calibrate(
        self,
        predicted_lower: np.ndarray,
        predicted_upper: np.ndarray,
        true_values: np.ndarray,
    ) -> CalibrationResult:
        """
        Calibrate using CQR method.

        Args:
            predicted_lower: Lower quantile predictions (e.g., 5th percentile)
            predicted_upper: Upper quantile predictions (e.g., 95th percentile)
            true_values: Actual observed values

        Returns:
            CalibrationResult with calibration statistics
        """
        n = len(true_values)

        if n < self.config.min_calibration_samples:
            logger.warning(
                f"Insufficient calibration samples: {n} < " f"{self.config.min_calibration_samples}"
            )
            return CalibrationResult(
                success=False,
                samples_used=n,
                method=ConformalMethod.CQR,
                error_message=f"Insufficient samples: {n} < {self.config.min_calibration_samples}",
                timestamp_ms=_now_ms(),
            )

        # Validate inputs
        if len(predicted_lower) != n or len(predicted_upper) != n:
            return CalibrationResult(
                success=False,
                samples_used=0,
                method=ConformalMethod.CQR,
                error_message="Array length mismatch",
                timestamp_ms=_now_ms(),
            )

        # CQR conformity scores: E_i = max(q_lo_i - Y_i, Y_i - q_hi_i)
        # Positive when Y is outside [q_lo, q_hi]
        scores = np.maximum(predicted_lower - true_values, true_values - predicted_upper)

        # Compute (1-α)(1 + 1/n) quantile for finite-sample correction
        alpha = 1.0 - self.config.coverage_target
        quantile_level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n))
        calibration_quantile = float(np.quantile(scores, quantile_level))

        # Compute empirical coverage for validation
        # After calibration, check what coverage we achieve on calibration set
        adjusted_lower = predicted_lower - calibration_quantile
        adjusted_upper = predicted_upper + calibration_quantile
        in_interval = (adjusted_lower <= true_values) & (true_values <= adjusted_upper)
        empirical_coverage = float(np.mean(in_interval))

        with self._lock:
            self._residuals = scores.copy()
            self._calibration_quantile = calibration_quantile
            self._calibration_samples = n
            self._empirical_coverage = empirical_coverage

        logger.info(
            f"CQR calibrated: Q_{quantile_level:.3f} = {calibration_quantile:.4f}, "
            f"empirical coverage = {empirical_coverage:.2%}, n = {n}"
        )

        return CalibrationResult(
            success=True,
            samples_used=n,
            calibration_quantile=calibration_quantile,
            empirical_coverage=empirical_coverage,
            method=ConformalMethod.CQR,
            timestamp_ms=_now_ms(),
        )

    def predict_interval(
        self,
        predicted_lower: float,
        predicted_upper: float,
        point_estimate: Optional[float] = None,
        timestamp_ms: Optional[int] = None,
    ) -> PredictionInterval:
        """
        Get calibrated prediction interval.

        Args:
            predicted_lower: Lower quantile prediction
            predicted_upper: Upper quantile prediction
            point_estimate: Optional point estimate (defaults to midpoint)
            timestamp_ms: Optional timestamp

        Returns:
            PredictionInterval with calibrated bounds
        """
        if point_estimate is None:
            point_estimate = (predicted_lower + predicted_upper) / 2.0

        ts = timestamp_ms or _now_ms()

        if self._calibration_quantile is None:
            # Not calibrated: return raw quantiles
            return PredictionInterval(
                point_estimate=point_estimate,
                lower_bound=predicted_lower,
                upper_bound=predicted_upper,
                coverage_target=self.config.coverage_target,
                method=ConformalMethod.CQR,
                is_calibrated=False,
                calibration_samples=0,
                timestamp_ms=ts,
            )

        # Apply calibration adjustment
        with self._lock:
            q = self._calibration_quantile
            n = self._calibration_samples

        return PredictionInterval(
            point_estimate=point_estimate,
            lower_bound=predicted_lower - q,
            upper_bound=predicted_upper + q,
            coverage_target=self.config.coverage_target,
            method=ConformalMethod.CQR,
            is_calibrated=True,
            calibration_samples=n,
            timestamp_ms=ts,
        )

    @property
    def is_calibrated(self) -> bool:
        """Check if calibrator has been calibrated."""
        return self._calibration_quantile is not None

    @property
    def calibration_samples(self) -> int:
        """Number of samples used for calibration."""
        return self._calibration_samples

    @property
    def calibration_quantile(self) -> Optional[float]:
        """Get the calibration quantile value."""
        return self._calibration_quantile

    @property
    def empirical_coverage(self) -> Optional[float]:
        """Get empirical coverage from calibration."""
        return self._empirical_coverage

    def get_state(self) -> Dict[str, Any]:
        """Get calibrator state for serialization."""
        with self._lock:
            return {
                "method": "CQR",
                "is_calibrated": self.is_calibrated,
                "calibration_samples": self._calibration_samples,
                "calibration_quantile": self._calibration_quantile,
                "empirical_coverage": self._empirical_coverage,
                "coverage_target": self.config.coverage_target,
            }


# =============================================================================
# EnbPI Calibrator
# =============================================================================


class EnbPICalibrator:
    """
    Ensemble batch Prediction Intervals for time series.

    Reference: Xu & Xie, "Conformal Prediction Interval for Dynamic Time-Series"
    (ICML 2021)

    Key insight: Maintains rolling residuals to handle non-exchangeability
    in time series data. Uses bootstrap aggregation of base predictors.

    Algorithm:
    1. Maintain rolling window of recent residuals
    2. For each new prediction, estimate interval from residual distribution
    3. Update residuals when true value observed (partial_fit)

    This method is particularly suitable for:
    - Non-stationary time series
    - Distribution shift scenarios
    - Online/streaming predictions
    """

    def __init__(self, config: ConformalConfig):
        """
        Initialize EnbPI calibrator.

        Args:
            config: Conformal prediction configuration
        """
        self.config = config
        self._residuals: Deque[float] = deque(maxlen=config.lookback_window)
        self._abs_residuals: Deque[float] = deque(maxlen=config.lookback_window)
        self._timestamps: Deque[int] = deque(maxlen=config.lookback_window)
        self._lock = threading.Lock()
        self._agg_func: Callable[[np.ndarray], float] = (
            np.median if config.time_series.enbpi_agg_func == "median" else np.mean
        )

    def partial_fit(
        self,
        predicted: float,
        true_value: float,
        timestamp_ms: Optional[int] = None,
    ) -> None:
        """
        Update residuals with new observation (online learning).

        Called after each step when true value becomes available.

        Args:
            predicted: Predicted value
            true_value: Actual observed value
            timestamp_ms: Optional timestamp
        """
        residual = true_value - predicted
        ts = timestamp_ms or _now_ms()

        with self._lock:
            self._residuals.append(residual)
            self._abs_residuals.append(abs(residual))
            self._timestamps.append(ts)

    def calibrate(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> CalibrationResult:
        """
        Batch calibration from historical data.

        For EnbPI, this just populates the residual buffer.

        Args:
            predictions: Array of predictions
            true_values: Array of true values
            timestamps: Optional array of timestamps

        Returns:
            CalibrationResult with statistics
        """
        n = len(predictions)

        if n < self.config.min_calibration_samples:
            return CalibrationResult(
                success=False,
                samples_used=n,
                method=ConformalMethod.ENBPI,
                error_message=f"Insufficient samples: {n}",
                timestamp_ms=_now_ms(),
            )

        residuals = true_values - predictions
        ts_array = timestamps if timestamps is not None else np.zeros(n, dtype=int)

        with self._lock:
            self._residuals.clear()
            self._abs_residuals.clear()
            self._timestamps.clear()

            # Keep only the most recent samples up to window size
            start_idx = max(0, n - self.config.lookback_window)
            for i in range(start_idx, n):
                self._residuals.append(float(residuals[i]))
                self._abs_residuals.append(abs(float(residuals[i])))
                self._timestamps.append(int(ts_array[i]))

        # Compute empirical coverage
        alpha = 1.0 - self.config.coverage_target
        lower_q = alpha / 2.0
        upper_q = 1.0 - alpha / 2.0
        q_lo, q_hi = np.quantile(residuals, [lower_q, upper_q])
        in_interval = (predictions + q_lo <= true_values) & (true_values <= predictions + q_hi)
        empirical_coverage = float(np.mean(in_interval))

        logger.info(
            f"EnbPI calibrated with {len(self._residuals)} residuals, "
            f"empirical coverage = {empirical_coverage:.2%}"
        )

        return CalibrationResult(
            success=True,
            samples_used=len(self._residuals),
            empirical_coverage=empirical_coverage,
            method=ConformalMethod.ENBPI,
            timestamp_ms=_now_ms(),
        )

    def predict_interval(
        self,
        point_estimate: float,
        coverage: Optional[float] = None,
        timestamp_ms: Optional[int] = None,
    ) -> PredictionInterval:
        """
        Get prediction interval using rolling residuals.

        Args:
            point_estimate: Point prediction
            coverage: Optional coverage level (defaults to config)
            timestamp_ms: Optional timestamp

        Returns:
            PredictionInterval with calibrated bounds
        """
        coverage = coverage or self.config.coverage_target
        alpha = 1.0 - coverage
        ts = timestamp_ms or _now_ms()

        with self._lock:
            n = len(self._residuals)

            if n < self.config.min_calibration_samples:
                # Not enough data: return point estimate with no bounds
                return PredictionInterval(
                    point_estimate=point_estimate,
                    lower_bound=point_estimate,
                    upper_bound=point_estimate,
                    coverage_target=coverage,
                    method=ConformalMethod.ENBPI,
                    is_calibrated=False,
                    calibration_samples=n,
                    timestamp_ms=ts,
                )

            residuals = np.array(self._residuals)

        # Symmetric interval from residual quantiles
        lower_q = alpha / 2.0
        upper_q = 1.0 - alpha / 2.0

        # Finite-sample correction
        lower_q_adj = max(0.0, lower_q * (n + 1) / n - 1.0 / n)
        upper_q_adj = min(1.0, upper_q * (n + 1) / n + 1.0 / n)

        q_lo, q_hi = np.quantile(residuals, [lower_q_adj, upper_q_adj])

        return PredictionInterval(
            point_estimate=point_estimate,
            lower_bound=point_estimate + q_lo,
            upper_bound=point_estimate + q_hi,
            coverage_target=coverage,
            method=ConformalMethod.ENBPI,
            is_calibrated=True,
            calibration_samples=n,
            timestamp_ms=ts,
        )

    @property
    def is_calibrated(self) -> bool:
        """Check if we have enough samples for calibration."""
        return len(self._residuals) >= self.config.min_calibration_samples

    @property
    def calibration_samples(self) -> int:
        """Number of samples in the residual buffer."""
        return len(self._residuals)

    def get_state(self) -> Dict[str, Any]:
        """Get calibrator state for serialization."""
        with self._lock:
            return {
                "method": "EnbPI",
                "is_calibrated": self.is_calibrated,
                "calibration_samples": len(self._residuals),
                "window_size": self.config.lookback_window,
                "agg_func": self.config.time_series.enbpi_agg_func,
            }


# =============================================================================
# ACI Calibrator
# =============================================================================


class ACICalibrator:
    """
    Adaptive Conformal Inference.

    Reference: Gibbs & Candes, "Adaptive Conformal Inference Under
    Distribution Shift" (2021)

    Key insight: Dynamically adjusts quantile threshold based on recent
    coverage performance. If we're under-covering, widen intervals.

    Algorithm:
    1. Track empirical coverage in rolling window
    2. If coverage < target: increase α_t (widen intervals)
    3. If coverage > target: decrease α_t (narrow intervals)

    Update rule: α_{t+1} = α_t + γ(α - 1{Y_t ∈ C_t})

    This method is particularly suitable for:
    - Distribution shift scenarios
    - Adaptive online learning
    - When coverage must be maintained under non-stationarity
    """

    def __init__(self, config: ConformalConfig):
        """
        Initialize ACI calibrator.

        Args:
            config: Conformal prediction configuration
        """
        self.config = config
        self._alpha_t: float = 1.0 - config.coverage_target  # Current alpha
        self._gamma: float = config.aci_gamma  # Adaptation rate
        self._coverage_history: Deque[bool] = deque(maxlen=config.lookback_window)
        self._residuals: Deque[float] = deque(maxlen=config.lookback_window)
        self._timestamps: Deque[int] = deque(maxlen=config.lookback_window)
        self._lock = threading.Lock()

        # Bounds on alpha for stability
        self._alpha_min = 0.01
        self._alpha_max = 0.50

    def update(
        self,
        predicted_interval: PredictionInterval,
        true_value: float,
        timestamp_ms: Optional[int] = None,
    ) -> None:
        """
        Update ACI state with new observation.

        Adjusts alpha based on whether true value was in interval.

        Args:
            predicted_interval: The prediction interval used
            true_value: Actual observed value
            timestamp_ms: Optional timestamp
        """
        covered = predicted_interval.contains(true_value)
        residual = abs(true_value - predicted_interval.point_estimate)
        ts = timestamp_ms or _now_ms()

        with self._lock:
            self._coverage_history.append(covered)
            self._residuals.append(residual)
            self._timestamps.append(ts)

            # ACI update rule: α_{t+1} = α_t + γ(α - 1{Y_t ∈ C_t})
            # If we missed (covered=False), increase alpha (widen)
            # If we hit (covered=True), decrease alpha (narrow)
            target_alpha = 1.0 - self.config.coverage_target
            err = target_alpha - (0.0 if covered else 1.0)
            self._alpha_t = np.clip(
                self._alpha_t + self._gamma * err,
                self._alpha_min,
                self._alpha_max,
            )

    def partial_fit(
        self,
        predicted: float,
        true_value: float,
        timestamp_ms: Optional[int] = None,
    ) -> None:
        """
        Update with new observation (online learning).

        For ACI, this requires creating an interval first.

        Args:
            predicted: Predicted value
            true_value: Actual observed value
            timestamp_ms: Optional timestamp
        """
        # Create interval from current state
        interval = self.predict_interval(predicted, timestamp_ms=timestamp_ms)
        # Update based on coverage
        self.update(interval, true_value, timestamp_ms)

    def calibrate(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> CalibrationResult:
        """
        Batch calibration from historical data.

        For ACI, this runs the adaptive update over historical data.

        Args:
            predictions: Array of predictions
            true_values: Array of true values
            timestamps: Optional array of timestamps

        Returns:
            CalibrationResult with statistics
        """
        n = len(predictions)

        if n < self.config.min_calibration_samples:
            return CalibrationResult(
                success=False,
                samples_used=n,
                method=ConformalMethod.ACI,
                error_message=f"Insufficient samples: {n}",
                timestamp_ms=_now_ms(),
            )

        ts_array = timestamps if timestamps is not None else np.zeros(n, dtype=int)

        # Reset state
        with self._lock:
            self._alpha_t = 1.0 - self.config.coverage_target
            self._coverage_history.clear()
            self._residuals.clear()
            self._timestamps.clear()

        # Run through historical data
        for i in range(n):
            self.partial_fit(float(predictions[i]), float(true_values[i]), int(ts_array[i]))

        empirical_coverage = self.empirical_coverage

        logger.info(
            f"ACI calibrated: current α = {self._alpha_t:.4f}, "
            f"empirical coverage = {empirical_coverage:.2%}"
        )

        return CalibrationResult(
            success=True,
            samples_used=len(self._residuals),
            calibration_quantile=1.0 - self._alpha_t,
            empirical_coverage=empirical_coverage,
            method=ConformalMethod.ACI,
            timestamp_ms=_now_ms(),
        )

    def predict_interval(
        self,
        point_estimate: float,
        coverage: Optional[float] = None,
        timestamp_ms: Optional[int] = None,
    ) -> PredictionInterval:
        """
        Get adaptively-calibrated prediction interval.

        Args:
            point_estimate: Point prediction
            coverage: Ignored for ACI (uses adaptive alpha)
            timestamp_ms: Optional timestamp

        Returns:
            PredictionInterval with adaptive bounds
        """
        ts = timestamp_ms or _now_ms()

        with self._lock:
            n = len(self._residuals)

            if n < self.config.min_calibration_samples:
                return PredictionInterval(
                    point_estimate=point_estimate,
                    lower_bound=point_estimate,
                    upper_bound=point_estimate,
                    coverage_target=self.config.coverage_target,
                    method=ConformalMethod.ACI,
                    is_calibrated=False,
                    calibration_samples=n,
                    timestamp_ms=ts,
                )

            residuals = np.array(self._residuals)
            current_alpha = self._alpha_t

        # Use current adaptive alpha for symmetric interval
        quantile_level = 1.0 - current_alpha / 2.0
        q = float(np.quantile(np.abs(residuals), quantile_level))

        return PredictionInterval(
            point_estimate=point_estimate,
            lower_bound=point_estimate - q,
            upper_bound=point_estimate + q,
            coverage_target=1.0 - current_alpha,
            method=ConformalMethod.ACI,
            is_calibrated=True,
            calibration_samples=n,
            timestamp_ms=ts,
        )

    @property
    def current_alpha(self) -> float:
        """Get current adaptive alpha value."""
        return self._alpha_t

    @property
    def current_coverage_target(self) -> float:
        """Get current coverage target (1 - alpha)."""
        return 1.0 - self._alpha_t

    @property
    def empirical_coverage(self) -> float:
        """Get empirical coverage from recent history."""
        if not self._coverage_history:
            return 0.0
        return sum(self._coverage_history) / len(self._coverage_history)

    @property
    def is_calibrated(self) -> bool:
        """Check if we have enough samples."""
        return len(self._residuals) >= self.config.min_calibration_samples

    @property
    def calibration_samples(self) -> int:
        """Number of samples in history."""
        return len(self._residuals)

    def get_state(self) -> Dict[str, Any]:
        """Get calibrator state for serialization."""
        with self._lock:
            return {
                "method": "ACI",
                "is_calibrated": self.is_calibrated,
                "calibration_samples": len(self._residuals),
                "current_alpha": self._alpha_t,
                "current_coverage_target": 1.0 - self._alpha_t,
                "empirical_coverage": self.empirical_coverage,
                "gamma": self._gamma,
            }


# =============================================================================
# Naive Calibrator (Baseline)
# =============================================================================


class NaiveCalibrator:
    """
    Naive percentile-based calibrator (baseline, no guarantees).

    This is a simple baseline that uses percentiles of historical residuals
    without any conformal calibration. It does NOT provide any coverage
    guarantees and should only be used for comparison purposes.
    """

    def __init__(self, config: ConformalConfig):
        """
        Initialize naive calibrator.

        Args:
            config: Conformal prediction configuration
        """
        self.config = config
        self._residuals: List[float] = []
        self._lock = threading.Lock()

    def calibrate(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray,
        **kwargs: Any,
    ) -> CalibrationResult:
        """Store residuals for percentile estimation."""
        n = len(predictions)
        residuals = true_values - predictions

        with self._lock:
            self._residuals = list(residuals)

        return CalibrationResult(
            success=n > 0,
            samples_used=n,
            method=ConformalMethod.NAIVE,
            timestamp_ms=_now_ms(),
        )

    def partial_fit(
        self,
        predicted: float,
        true_value: float,
        timestamp_ms: Optional[int] = None,
    ) -> None:
        """Add new residual."""
        with self._lock:
            self._residuals.append(true_value - predicted)
            # Keep only recent residuals
            if len(self._residuals) > self.config.lookback_window:
                self._residuals = self._residuals[-self.config.lookback_window :]

    def predict_interval(
        self,
        point_estimate: float,
        coverage: Optional[float] = None,
        timestamp_ms: Optional[int] = None,
    ) -> PredictionInterval:
        """Get interval using simple percentiles (no guarantees)."""
        coverage = coverage or self.config.coverage_target
        alpha = 1.0 - coverage
        ts = timestamp_ms or _now_ms()

        with self._lock:
            n = len(self._residuals)

            if n < 2:
                return PredictionInterval(
                    point_estimate=point_estimate,
                    lower_bound=point_estimate,
                    upper_bound=point_estimate,
                    coverage_target=coverage,
                    method=ConformalMethod.NAIVE,
                    is_calibrated=False,
                    calibration_samples=n,
                    timestamp_ms=ts,
                )

            residuals = np.array(self._residuals)

        lower_q = alpha / 2.0
        upper_q = 1.0 - alpha / 2.0
        q_lo, q_hi = np.quantile(residuals, [lower_q, upper_q])

        return PredictionInterval(
            point_estimate=point_estimate,
            lower_bound=point_estimate + q_lo,
            upper_bound=point_estimate + q_hi,
            coverage_target=coverage,
            method=ConformalMethod.NAIVE,
            is_calibrated=True,
            calibration_samples=n,
            timestamp_ms=ts,
        )

    @property
    def is_calibrated(self) -> bool:
        return len(self._residuals) >= 2

    @property
    def calibration_samples(self) -> int:
        return len(self._residuals)


# =============================================================================
# Conformal CVaR Estimator
# =============================================================================


class ConformalCVaREstimator:
    """
    CVaR estimation with conformal prediction bounds.

    Wraps the existing CVaR computation to add uncertainty bounds.
    Uses the CQR calibrator to provide bounds on quantile estimates.
    """

    def __init__(
        self,
        config: ConformalConfig,
        calibrator: Optional[CQRCalibrator] = None,
    ):
        """
        Initialize CVaR estimator with conformal bounds.

        Args:
            config: Conformal prediction configuration
            calibrator: Optional pre-configured CQR calibrator
        """
        self.config = config
        self._calibrator = calibrator or CQRCalibrator(config)

    def compute_cvar_with_bounds(
        self,
        quantiles: np.ndarray,
        alpha: float = 0.05,
    ) -> CVaRBounds:
        """
        Compute CVaR with conformal bounds.

        Args:
            quantiles: Array of shape [num_quantiles] with quantile values
            alpha: CVaR alpha level (e.g., 0.05 for CVaR_5%)

        Returns:
            CVaRBounds with point estimate and calibrated bounds
        """
        num_quantiles = len(quantiles)
        if num_quantiles == 0:
            return CVaRBounds(
                cvar_point=0.0,
                cvar_lower=0.0,
                cvar_upper=0.0,
                var_point=0.0,
                var_lower=0.0,
                var_upper=0.0,
                alpha=alpha,
                interval=PredictionInterval(
                    point_estimate=0.0,
                    lower_bound=0.0,
                    upper_bound=0.0,
                    coverage_target=self.config.coverage_target,
                    method=ConformalMethod.CQR,
                    is_calibrated=False,
                ),
            )

        # Find VaR (alpha-quantile)
        var_idx = int(alpha * num_quantiles)
        var_idx = min(var_idx, num_quantiles - 1)
        var_point = float(quantiles[var_idx])

        # Compute CVaR as average of quantiles below VaR
        if var_idx > 0:
            cvar_point = float(np.mean(quantiles[:var_idx]))
        else:
            cvar_point = float(quantiles[0])

        # Get prediction intervals for VaR and CVaR
        # Use neighboring quantiles as "predicted lower/upper"
        if var_idx > 0 and var_idx < num_quantiles - 1:
            var_interval = self._calibrator.predict_interval(
                predicted_lower=float(quantiles[var_idx - 1]),
                predicted_upper=float(quantiles[var_idx + 1]),
                point_estimate=var_point,
            )
        else:
            var_interval = self._calibrator.predict_interval(
                predicted_lower=float(quantiles[0]),
                predicted_upper=float(quantiles[-1]),
                point_estimate=var_point,
            )

        # For CVaR bounds, use spread of tail quantiles
        tail_quantiles = quantiles[: max(1, var_idx)]
        cvar_interval = self._calibrator.predict_interval(
            predicted_lower=float(np.min(tail_quantiles)),
            predicted_upper=float(np.max(tail_quantiles)),
            point_estimate=cvar_point,
        )

        return CVaRBounds(
            cvar_point=cvar_point,
            cvar_lower=cvar_interval.lower_bound,
            cvar_upper=cvar_interval.upper_bound,
            var_point=var_point,
            var_lower=var_interval.lower_bound,
            var_upper=var_interval.upper_bound,
            alpha=alpha,
            interval=cvar_interval,
        )

    @property
    def is_calibrated(self) -> bool:
        """Check if underlying calibrator is calibrated."""
        return self._calibrator.is_calibrated


# =============================================================================
# Uncertainty Tracker
# =============================================================================


class UncertaintyTrackerImpl:
    """
    Tracks uncertainty over time for escalation decisions.

    Maintains a history of interval widths and computes percentiles
    to determine when uncertainty is abnormally high.
    """

    def __init__(
        self,
        config: ConformalConfig,
        history_size: int = 1000,
    ):
        """
        Initialize uncertainty tracker.

        Args:
            config: Conformal prediction configuration
            history_size: Maximum size of history to maintain
        """
        self.config = config
        self._interval_widths: Deque[float] = deque(maxlen=history_size)
        self._timestamps: Deque[int] = deque(maxlen=history_size)
        self._warning_threshold: Optional[float] = None
        self._critical_threshold: Optional[float] = None
        self._last_state: Optional[UncertaintyState] = None
        self._lock = threading.Lock()

    def record_interval(
        self,
        interval: PredictionInterval,
        timestamp_ms: Optional[int] = None,
    ) -> UncertaintyState:
        """
        Record new interval and return current uncertainty state.

        Args:
            interval: PredictionInterval to record
            timestamp_ms: Optional timestamp

        Returns:
            UncertaintyState with current uncertainty metrics
        """
        width = interval.interval_width
        ts = timestamp_ms or _now_ms()

        with self._lock:
            self._interval_widths.append(width)
            self._timestamps.append(ts)

            n = len(self._interval_widths)

            if n >= 100:
                widths = np.array(self._interval_widths)
                self._warning_threshold = float(
                    np.percentile(widths, self.config.warning_percentile)
                )
                self._critical_threshold = float(
                    np.percentile(widths, self.config.critical_percentile)
                )
                sorted_widths = np.sort(widths)
                percentile = float(np.searchsorted(sorted_widths, width) / n * 100)
            else:
                percentile = 50.0  # Default until we have enough history

            # Determine escalation level
            if self._critical_threshold and width > self._critical_threshold:
                level = EscalationLevel.CRITICAL
                action = self.config.escalation.action_on_critical
            elif self._warning_threshold and width > self._warning_threshold:
                level = EscalationLevel.WARNING
                action = self.config.escalation.action_on_warning
            else:
                level = EscalationLevel.NORMAL
                action = EscalationAction.LOG

            # Compute position scale
            if self.config.uncertainty_position_scaling:
                baseline = self.config.baseline_interval_width
                if width <= baseline:
                    scale = 1.0
                else:
                    # Linear reduction
                    reduction = min(
                        (width - baseline) / baseline * self.config.max_uncertainty_reduction,
                        self.config.max_uncertainty_reduction,
                    )
                    scale = 1.0 - reduction
            else:
                scale = 1.0

            state = UncertaintyState(
                current_interval_width=width,
                historical_percentile=percentile,
                escalation_level=level,
                recommended_action=action,
                recommended_position_scale=max(0.0, min(1.0, scale)),
                samples_since_calibration=interval.calibration_samples,
                is_calibrated=interval.is_calibrated,
                timestamp_ms=ts,
            )

            self._last_state = state

        return state

    def get_current_state(self) -> UncertaintyState:
        """Get current uncertainty state without recording new interval."""
        with self._lock:
            if self._last_state is not None:
                return self._last_state

        # Return default state if no history
        return UncertaintyState(
            current_interval_width=0.0,
            historical_percentile=50.0,
            escalation_level=EscalationLevel.NORMAL,
            recommended_action=EscalationAction.LOG,
            recommended_position_scale=1.0,
            samples_since_calibration=0,
            is_calibrated=False,
            timestamp_ms=_now_ms(),
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about tracked intervals."""
        with self._lock:
            if not self._interval_widths:
                return {
                    "count": 0,
                    "mean_width": None,
                    "std_width": None,
                    "warning_threshold": None,
                    "critical_threshold": None,
                }

            widths = np.array(self._interval_widths)
            return {
                "count": len(widths),
                "mean_width": float(np.mean(widths)),
                "std_width": float(np.std(widths)),
                "min_width": float(np.min(widths)),
                "max_width": float(np.max(widths)),
                "warning_threshold": self._warning_threshold,
                "critical_threshold": self._critical_threshold,
            }


# =============================================================================
# Factory Functions
# =============================================================================


def create_calibrator(
    config: ConformalConfig,
) -> Union[CQRCalibrator, EnbPICalibrator, ACICalibrator, NaiveCalibrator]:
    """
    Create appropriate calibrator based on config.

    Args:
        config: Conformal prediction configuration

    Returns:
        Calibrator instance matching the configured method

    Raises:
        ValueError: If conformal prediction is disabled or method unknown
    """
    if not config.enabled:
        raise ValueError("Conformal prediction is disabled in config")

    method = config.method

    if method == ConformalMethod.CQR:
        return CQRCalibrator(config)
    elif method == ConformalMethod.ENBPI:
        return EnbPICalibrator(config)
    elif method == ConformalMethod.ACI:
        return ACICalibrator(config)
    elif method == ConformalMethod.NAIVE:
        return NaiveCalibrator(config)
    else:
        raise ValueError(f"Unknown conformal method: {method}")


def create_cvar_estimator(
    config: ConformalConfig,
    calibrator: Optional[CQRCalibrator] = None,
) -> ConformalCVaREstimator:
    """
    Create CVaR estimator with conformal bounds.

    Args:
        config: Conformal prediction configuration
        calibrator: Optional pre-configured calibrator

    Returns:
        ConformalCVaREstimator instance
    """
    return ConformalCVaREstimator(config, calibrator)


def create_uncertainty_tracker(
    config: ConformalConfig,
    history_size: int = 1000,
) -> UncertaintyTrackerImpl:
    """
    Create uncertainty tracker.

    Args:
        config: Conformal prediction configuration
        history_size: Maximum history size

    Returns:
        UncertaintyTrackerImpl instance
    """
    return UncertaintyTrackerImpl(config, history_size)
