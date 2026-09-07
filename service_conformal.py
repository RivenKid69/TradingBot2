# -*- coding: utf-8 -*-
"""
service_conformal.py
Conformal prediction service layer.

Layer: SERVICE (integrates with existing components)

This module provides:
1. ConformalPredictionService - Main service for conformal prediction integration
2. Integration helpers for existing components (distributional_ppo, risk_guard)
3. Calibration management (offline + online)
4. Uncertainty tracking and escalation
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from core_conformal import (
    CalibrationResult,
    ConformalConfig,
    ConformalMethod,
    CVaRBounds,
    EscalationAction,
    EscalationLevel,
    PredictionInterval,
    UncertaintyState,
    create_conformal_config,
    create_disabled_config,
)
from impl_conformal import (
    ACICalibrator,
    ConformalCVaREstimator,
    CQRCalibrator,
    EnbPICalibrator,
    NaiveCalibrator,
    UncertaintyTrackerImpl,
    create_calibrator,
    create_cvar_estimator,
    create_uncertainty_tracker,
)

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)


# =============================================================================
# Conformal Prediction Service
# =============================================================================


class ConformalPredictionService:
    """
    Main service for conformal prediction integration.

    Provides:
    1. Calibration management (offline + online)
    2. Prediction interval generation
    3. CVaR bounds computation
    4. Uncertainty tracking and escalation
    5. Integration hooks for existing components

    Usage:
        # Create service
        config = create_conformal_config(config_dict)
        service = ConformalPredictionService(config)

        # Calibrate (after training)
        service.calibrate(predictions, true_values)

        # Get prediction interval
        interval = service.predict_interval(point_estimate)

        # Get CVaR bounds
        bounds = service.compute_cvar_bounds(quantiles)

        # Get uncertainty state for risk management
        state = service.get_uncertainty_state(interval)
    """

    def __init__(self, config: ConformalConfig):
        """
        Initialize conformal prediction service.

        Args:
            config: Conformal prediction configuration
        """
        self.config = config
        self._enabled = config.enabled

        if not self._enabled:
            logger.info("ConformalPredictionService disabled")
            self._calibrator = None
            self._cvar_estimator = None
            self._uncertainty_tracker = None
            return

        # Initialize components
        self._calibrator = create_calibrator(config)
        self._cvar_estimator = create_cvar_estimator(
            config, self._calibrator if isinstance(self._calibrator, CQRCalibrator) else None
        )
        self._uncertainty_tracker = create_uncertainty_tracker(config)

        # Calibration state
        self._calibration_data: Dict[str, List[float]] = {
            "predictions": [],
            "true_values": [],
            "timestamps": [],
        }
        self._steps_since_calibration = 0
        self._lock = threading.RLock()  # Reentrant lock for nested calibrate() calls

        # Callbacks for escalation
        self._escalation_callbacks: Dict[
            EscalationLevel, List[Callable[[UncertaintyState], None]]
        ] = {
            EscalationLevel.NORMAL: [],
            EscalationLevel.WARNING: [],
            EscalationLevel.CRITICAL: [],
        }

        logger.info(
            f"ConformalPredictionService initialized: method={config.method.name}, "
            f"coverage={config.coverage_target}"
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def is_enabled(self) -> bool:
        """Check if conformal prediction is enabled."""
        return self._enabled

    def predict_interval(
        self,
        point_estimate: float,
        predicted_lower: Optional[float] = None,
        predicted_upper: Optional[float] = None,
        timestamp_ms: Optional[int] = None,
    ) -> PredictionInterval:
        """
        Get prediction interval for a point estimate.

        Args:
            point_estimate: The point prediction
            predicted_lower: Optional lower quantile (for CQR)
            predicted_upper: Optional upper quantile (for CQR)
            timestamp_ms: Optional timestamp

        Returns:
            PredictionInterval with calibrated bounds
        """
        ts = timestamp_ms or _now_ms()

        if not self._enabled:
            return PredictionInterval(
                point_estimate=point_estimate,
                lower_bound=point_estimate,
                upper_bound=point_estimate,
                coverage_target=0.0,
                method=ConformalMethod.NAIVE,
                is_calibrated=False,
                timestamp_ms=ts,
            )

        if isinstance(self._calibrator, CQRCalibrator):
            if predicted_lower is None or predicted_upper is None:
                # Use symmetric interval around point estimate
                default_width = self.config.baseline_interval_width * abs(point_estimate)
                if default_width == 0:
                    default_width = self.config.baseline_interval_width
                predicted_lower = point_estimate - default_width
                predicted_upper = point_estimate + default_width

            return self._calibrator.predict_interval(
                predicted_lower, predicted_upper, point_estimate, ts
            )
        else:
            return self._calibrator.predict_interval(point_estimate, timestamp_ms=ts)

    def compute_cvar_bounds(
        self,
        quantiles: np.ndarray,
        alpha: float = 0.05,
    ) -> CVaRBounds:
        """
        Compute CVaR with conformal bounds.

        Integration point: Call this instead of _cvar_from_quantiles
        when bounds are needed.

        Args:
            quantiles: Array of quantile values
            alpha: CVaR alpha level

        Returns:
            CVaRBounds with point estimate and bounds
        """
        if not self._enabled or not self.config.cvar_bounds_enabled:
            # Return point estimate without bounds
            num_q = len(quantiles)
            if num_q == 0:
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
                        coverage_target=0.0,
                        method=ConformalMethod.NAIVE,
                    ),
                )

            var_idx = int(alpha * num_q)
            var_idx = min(var_idx, num_q - 1)
            cvar = float(np.mean(quantiles[: max(1, var_idx)]))
            var = float(quantiles[var_idx])

            return CVaRBounds(
                cvar_point=cvar,
                cvar_lower=cvar,
                cvar_upper=cvar,
                var_point=var,
                var_lower=var,
                var_upper=var,
                alpha=alpha,
                interval=PredictionInterval(
                    point_estimate=cvar,
                    lower_bound=cvar,
                    upper_bound=cvar,
                    coverage_target=0.0,
                    method=ConformalMethod.NAIVE,
                ),
            )

        return self._cvar_estimator.compute_cvar_with_bounds(quantiles, alpha)

    def get_uncertainty_state(
        self,
        interval: Optional[PredictionInterval] = None,
        point_estimate: Optional[float] = None,
        timestamp_ms: Optional[int] = None,
    ) -> UncertaintyState:
        """
        Get current uncertainty state for decision making.

        Integration point: Call before making trading decisions.

        Args:
            interval: Optional prediction interval to record
            point_estimate: Optional point estimate (will create interval)
            timestamp_ms: Optional timestamp

        Returns:
            UncertaintyState with current metrics and recommendations
        """
        ts = timestamp_ms or _now_ms()

        if not self._enabled or not self.config.escalation_enabled:
            return UncertaintyState(
                current_interval_width=0.0,
                historical_percentile=50.0,
                escalation_level=EscalationLevel.NORMAL,
                recommended_action=EscalationAction.LOG,
                recommended_position_scale=1.0,
                samples_since_calibration=0,
                is_calibrated=False,
                timestamp_ms=ts,
            )

        if interval is None:
            if point_estimate is not None:
                interval = self.predict_interval(point_estimate, timestamp_ms=ts)
            else:
                return self._uncertainty_tracker.get_current_state()

        state = self._uncertainty_tracker.record_interval(interval, ts)

        # Trigger escalation callbacks
        callbacks = self._escalation_callbacks.get(state.escalation_level, [])
        for callback in callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.warning(f"Escalation callback failed: {e}")

        return state

    def record_observation(
        self,
        prediction: float,
        true_value: float,
        timestamp_ms: Optional[int] = None,
    ) -> None:
        """
        Record an observation for online calibration.

        Call this when the true value becomes known (e.g., after trade execution).

        Args:
            prediction: The predicted value
            true_value: The actual observed value
            timestamp_ms: Optional timestamp
        """
        if not self._enabled:
            return

        ts = timestamp_ms or _now_ms()

        with self._lock:
            self._calibration_data["predictions"].append(prediction)
            self._calibration_data["true_values"].append(true_value)
            self._calibration_data["timestamps"].append(float(ts))
            self._steps_since_calibration += 1

            # Online update for EnbPI/ACI
            if isinstance(self._calibrator, (EnbPICalibrator, ACICalibrator)):
                if isinstance(self._calibrator, EnbPICalibrator):
                    self._calibrator.partial_fit(prediction, true_value, ts)
                elif isinstance(self._calibrator, ACICalibrator):
                    self._calibrator.partial_fit(prediction, true_value, ts)

            # Check if recalibration needed (for CQR)
            if (
                self.config.recalibrate_interval > 0
                and self._steps_since_calibration >= self.config.recalibrate_interval
            ):
                self._maybe_recalibrate()

    def calibrate(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray,
        predicted_lower: Optional[np.ndarray] = None,
        predicted_upper: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> CalibrationResult:
        """
        Explicitly calibrate the conformal predictor.

        Call this after training with held-out validation data.

        Args:
            predictions: Array of point predictions
            true_values: Array of true values
            predicted_lower: Optional lower quantile predictions (for CQR)
            predicted_upper: Optional upper quantile predictions (for CQR)
            timestamps: Optional timestamps

        Returns:
            CalibrationResult with calibration statistics
        """
        if not self._enabled:
            return CalibrationResult(
                success=False,
                samples_used=0,
                method=ConformalMethod.NAIVE,
                error_message="Conformal prediction disabled",
                timestamp_ms=_now_ms(),
            )

        n = len(predictions)

        if n < self.config.min_calibration_samples:
            logger.warning(
                f"Insufficient samples for calibration: {n} < "
                f"{self.config.min_calibration_samples}"
            )
            return CalibrationResult(
                success=False,
                samples_used=n,
                method=self.config.method,
                error_message=f"Insufficient samples: {n}",
                timestamp_ms=_now_ms(),
            )

        if isinstance(self._calibrator, CQRCalibrator):
            if predicted_lower is None or predicted_upper is None:
                # Estimate quantiles from residuals
                residuals = true_values - predictions
                q_lo = float(np.percentile(residuals, 5))
                q_hi = float(np.percentile(residuals, 95))
                predicted_lower = predictions + q_lo
                predicted_upper = predictions + q_hi

            result = self._calibrator.calibrate(predicted_lower, predicted_upper, true_values)

        elif isinstance(self._calibrator, (EnbPICalibrator, ACICalibrator)):
            result = self._calibrator.calibrate(predictions, true_values, timestamps)

        elif isinstance(self._calibrator, NaiveCalibrator):
            result = self._calibrator.calibrate(predictions, true_values)

        else:
            return CalibrationResult(
                success=False,
                samples_used=0,
                method=self.config.method,
                error_message=f"Unknown calibrator type",
                timestamp_ms=_now_ms(),
            )

        with self._lock:
            self._steps_since_calibration = 0

        logger.info(f"Calibration complete: {n} samples, success={result.success}")
        return result

    def register_escalation_callback(
        self,
        level: EscalationLevel,
        callback: Callable[[UncertaintyState], None],
    ) -> None:
        """
        Register callback for escalation level.

        Args:
            level: Escalation level to trigger on
            callback: Function to call with UncertaintyState
        """
        if level not in self._escalation_callbacks:
            self._escalation_callbacks[level] = []
        self._escalation_callbacks[level].append(callback)

    def get_position_scale(
        self,
        interval: Optional[PredictionInterval] = None,
        point_estimate: Optional[float] = None,
    ) -> float:
        """
        Get recommended position scale based on uncertainty.

        Integration point: Multiply position size by this factor.

        Args:
            interval: Optional prediction interval
            point_estimate: Optional point estimate (will create interval)

        Returns:
            Position scale factor (0.0 to 1.0)
        """
        if not self._enabled or not self.config.risk_integration_enabled:
            return 1.0

        if interval is None and point_estimate is not None:
            interval = self.predict_interval(point_estimate)

        if interval is None:
            return 1.0

        state = self.get_uncertainty_state(interval)
        return state.recommended_position_scale

    def get_calibrator_state(self) -> Dict[str, Any]:
        """Get calibrator state for serialization."""
        if not self._enabled or self._calibrator is None:
            return {"enabled": False}

        return {
            "enabled": True,
            "method": self.config.method.name,
            "is_calibrated": self._calibrator.is_calibrated,
            "calibration_samples": self._calibrator.calibration_samples,
            "steps_since_calibration": self._steps_since_calibration,
            "calibrator_state": self._calibrator.get_state(),
        }

    def get_uncertainty_statistics(self) -> Dict[str, Any]:
        """Get uncertainty tracking statistics."""
        if not self._enabled or self._uncertainty_tracker is None:
            return {"enabled": False}

        return self._uncertainty_tracker.get_statistics()

    def save_state(self, path: Path) -> None:
        """Save service state to disk."""
        if not self._enabled:
            return

        state = {
            "config": {
                "method": self.config.method.name,
                "coverage_target": self.config.coverage_target,
            },
            "calibrator": self.get_calibrator_state(),
            "uncertainty": self.get_uncertainty_statistics(),
            "timestamp_ms": _now_ms(),
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved conformal state to {path}")

    def _maybe_recalibrate(self) -> None:
        """Recalibrate if enough new data."""
        if not isinstance(self._calibrator, CQRCalibrator):
            return  # EnbPI/ACI do online updates

        data = self._calibration_data
        n = len(data["predictions"])

        if n < self.config.min_calibration_samples:
            return

        # Use recent data for recalibration
        start_idx = max(0, n - self.config.min_calibration_samples * 2)

        predictions = np.array(data["predictions"][start_idx:])
        true_values = np.array(data["true_values"][start_idx:])

        self.calibrate(predictions, true_values)
        self._steps_since_calibration = 0


# =============================================================================
# Integration Helpers
# =============================================================================


def create_conformal_service(config_dict: Dict[str, Any]) -> ConformalPredictionService:
    """
    Create conformal service from config dictionary.

    Args:
        config_dict: Configuration dictionary (from YAML)

    Returns:
        ConformalPredictionService instance
    """
    if not config_dict.get("enabled", True):
        return ConformalPredictionService(create_disabled_config())

    config = create_conformal_config(config_dict)
    return ConformalPredictionService(config)


def wrap_cvar_with_bounds(
    original_cvar_fn: Callable,
    conformal_service: ConformalPredictionService,
) -> Callable:
    """
    Wrap existing CVaR function to add conformal bounds.

    Usage in distributional_ppo.py:
        self._cvar_from_quantiles = wrap_cvar_with_bounds(
            self._cvar_from_quantiles,
            self.conformal_service
        )

    Args:
        original_cvar_fn: Original CVaR computation function
        conformal_service: ConformalPredictionService instance

    Returns:
        Wrapped function that computes CVaR and stores bounds
    """
    if torch is None:
        return original_cvar_fn

    def wrapped(predicted_quantiles: "torch.Tensor") -> "torch.Tensor":
        # Call original for point estimate
        cvar_point = original_cvar_fn(predicted_quantiles)

        if not conformal_service.is_enabled():
            return cvar_point

        # Store bounds in service for later use
        # (actual modification depends on use_bounds_for_gae config)
        quantiles_np = predicted_quantiles.detach().cpu().numpy()
        batch_size = quantiles_np.shape[0]

        for i in range(batch_size):
            try:
                bounds = conformal_service.compute_cvar_bounds(quantiles_np[i])
                # Bounds available via conformal_service.get_uncertainty_state()
            except Exception as e:
                logger.debug(f"Failed to compute CVaR bounds: {e}")

        return cvar_point

    return wrapped


def create_risk_guard_integration(
    conformal_service: ConformalPredictionService,
    base_position_scale: Callable[[], float],
) -> Callable[[], float]:
    """
    Create position scaling function for risk guard integration.

    Usage in risk_guard.py:
        position_scale = create_risk_guard_integration(
            conformal_service,
            lambda: self.cfg.max_abs_position
        )

    Args:
        conformal_service: ConformalPredictionService instance
        base_position_scale: Function returning base position scale

    Returns:
        Function returning uncertainty-adjusted position scale
    """

    def get_scaled_position() -> float:
        base = base_position_scale()

        if not conformal_service.is_enabled():
            return base

        state = conformal_service.get_uncertainty_state()
        return base * state.recommended_position_scale

    return get_scaled_position


def create_escalation_handler(
    on_warning: Optional[Callable[[UncertaintyState], None]] = None,
    on_critical: Optional[Callable[[UncertaintyState], None]] = None,
    reduce_position_callback: Optional[Callable[[float], None]] = None,
    halt_trading_callback: Optional[Callable[[], None]] = None,
) -> Dict[EscalationLevel, Callable[[UncertaintyState], None]]:
    """
    Create escalation handler callbacks.

    Usage:
        handlers = create_escalation_handler(
            on_warning=lambda s: logger.warning(f"Uncertainty warning: {s}"),
            reduce_position_callback=lambda scale: engine.set_position_scale(scale),
            halt_trading_callback=lambda: engine.halt(),
        )

        for level, callback in handlers.items():
            conformal_service.register_escalation_callback(level, callback)

    Args:
        on_warning: Optional custom warning handler
        on_critical: Optional custom critical handler
        reduce_position_callback: Callback to reduce position
        halt_trading_callback: Callback to halt trading

    Returns:
        Dict mapping escalation levels to callbacks
    """

    def default_warning_handler(state: UncertaintyState) -> None:
        logger.warning(
            f"Uncertainty WARNING: width={state.current_interval_width:.4f}, "
            f"percentile={state.historical_percentile:.1f}%, "
            f"recommended_scale={state.recommended_position_scale:.2f}"
        )
        if state.recommended_action == EscalationAction.REDUCE_POSITION:
            if reduce_position_callback:
                reduce_position_callback(state.recommended_position_scale)

    def default_critical_handler(state: UncertaintyState) -> None:
        logger.error(
            f"Uncertainty CRITICAL: width={state.current_interval_width:.4f}, "
            f"percentile={state.historical_percentile:.1f}%, "
            f"recommended_scale={state.recommended_position_scale:.2f}"
        )
        if state.recommended_action == EscalationAction.HALT:
            if halt_trading_callback:
                halt_trading_callback()
        elif state.recommended_action == EscalationAction.REDUCE_POSITION:
            if reduce_position_callback:
                reduce_position_callback(state.recommended_position_scale)

    return {
        EscalationLevel.WARNING: on_warning or default_warning_handler,
        EscalationLevel.CRITICAL: on_critical or default_critical_handler,
    }


# =============================================================================
# Training Integration
# =============================================================================


class TrainingConformalIntegration:
    """
    Integration helper for training loop.

    Handles:
    1. Collecting predictions and values during training
    2. Calibration after training epochs
    3. Logging uncertainty metrics
    """

    def __init__(
        self,
        service: ConformalPredictionService,
        calibration_frequency: int = 1,  # Calibrate every N epochs
    ):
        """
        Initialize training integration.

        Args:
            service: ConformalPredictionService instance
            calibration_frequency: Calibrate every N epochs
        """
        self.service = service
        self.calibration_frequency = calibration_frequency
        self._epoch_predictions: List[float] = []
        self._epoch_true_values: List[float] = []
        self._epoch_count = 0

    def record_batch(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray,
    ) -> None:
        """Record a batch of predictions and true values."""
        self._epoch_predictions.extend(predictions.flatten().tolist())
        self._epoch_true_values.extend(true_values.flatten().tolist())

    def on_epoch_end(self) -> Optional[CalibrationResult]:
        """
        Call at end of each epoch.

        Returns:
            CalibrationResult if calibration was performed, else None
        """
        self._epoch_count += 1
        result = None

        if self.calibration_frequency > 0 and self._epoch_count % self.calibration_frequency == 0:

            if len(self._epoch_predictions) > 0:
                predictions = np.array(self._epoch_predictions)
                true_values = np.array(self._epoch_true_values)

                result = self.service.calibrate(predictions, true_values)

        # Clear epoch data
        self._epoch_predictions = []
        self._epoch_true_values = []

        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics for logging."""
        return {
            "conformal_enabled": self.service.is_enabled(),
            "calibrator_state": self.service.get_calibrator_state(),
            "uncertainty_stats": self.service.get_uncertainty_statistics(),
            "epoch_count": self._epoch_count,
            "pending_samples": len(self._epoch_predictions),
        }


# =============================================================================
# Live Trading Integration
# =============================================================================


class LiveTradingConformalIntegration:
    """
    Integration helper for live trading.

    Handles:
    1. Real-time uncertainty monitoring
    2. Position scaling based on uncertainty
    3. Escalation handling
    """

    def __init__(
        self,
        service: ConformalPredictionService,
        reduce_position_callback: Optional[Callable[[float], None]] = None,
        halt_trading_callback: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize live trading integration.

        Args:
            service: ConformalPredictionService instance
            reduce_position_callback: Callback to reduce position
            halt_trading_callback: Callback to halt trading
        """
        self.service = service

        # Register escalation handlers
        handlers = create_escalation_handler(
            reduce_position_callback=reduce_position_callback,
            halt_trading_callback=halt_trading_callback,
        )

        for level, callback in handlers.items():
            self.service.register_escalation_callback(level, callback)

    def get_position_scale(self, value_estimate: float) -> float:
        """
        Get position scale factor based on current uncertainty.

        Args:
            value_estimate: Current value estimate

        Returns:
            Position scale factor (0.0 to 1.0)
        """
        return self.service.get_position_scale(point_estimate=value_estimate)

    def record_trade_result(
        self,
        predicted_pnl: float,
        actual_pnl: float,
        timestamp_ms: Optional[int] = None,
    ) -> UncertaintyState:
        """
        Record trade result for online calibration.

        Args:
            predicted_pnl: Predicted PnL
            actual_pnl: Actual PnL
            timestamp_ms: Optional timestamp

        Returns:
            Current uncertainty state
        """
        self.service.record_observation(predicted_pnl, actual_pnl, timestamp_ms)

        # Get and return current uncertainty state
        interval = self.service.predict_interval(predicted_pnl, timestamp_ms=timestamp_ms)
        return self.service.get_uncertainty_state(interval)

    def should_reduce_exposure(self) -> Tuple[bool, float]:
        """
        Check if exposure should be reduced.

        Returns:
            Tuple of (should_reduce, scale_factor)
        """
        state = self.service.get_uncertainty_state()

        if state.escalation_level in (EscalationLevel.WARNING, EscalationLevel.CRITICAL):
            return True, state.recommended_position_scale

        return False, 1.0
