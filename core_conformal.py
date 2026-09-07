# -*- coding: utf-8 -*-
"""
core_conformal.py
Core conformal prediction types and protocols.

Layer: CORE (no dependencies except standard library + dataclasses)

Scientific References:
- CQR: Romano et al., "Conformalized Quantile Regression" (2019)
  https://arxiv.org/abs/1905.03222
- EnbPI: Xu & Xie, "Conformal Prediction Interval for Dynamic Time-Series" (ICML 2021)
- ACI: Gibbs & Candes, "Adaptive Conformal Inference Under Distribution Shift" (2021)
- MAPIE: https://mapie.readthedocs.io/en/latest/
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from dataclasses import dataclass, field
from typing import Protocol, Optional, Tuple, Literal, Dict, Any, List, runtime_checkable
from enum import Enum, auto
import math


# =========================
# Enumerations
# =========================


class ConformalMethod(Enum):
    """Supported conformal prediction methods."""

    CQR = auto()  # Conformalized Quantile Regression (Romano et al., 2019)
    ENBPI = auto()  # Ensemble batch Prediction Intervals (Xu & Xie, ICML 2021)
    ACI = auto()  # Adaptive Conformal Inference (Gibbs & Candes, 2021)
    NAIVE = auto()  # Simple percentile-based (baseline, no guarantees)


class EscalationLevel(Enum):
    """Uncertainty escalation levels for risk management."""

    NORMAL = auto()  # Within normal bounds
    WARNING = auto()  # Elevated uncertainty, consider reducing exposure
    CRITICAL = auto()  # Very high uncertainty, take defensive action


class EscalationAction(Enum):
    """Actions to take on escalation."""

    LOG = auto()  # Log the escalation only
    REDUCE_POSITION = auto()  # Reduce position size
    HALT = auto()  # Halt trading
    HUMAN_REVIEW = auto()  # Flag for human review


# =========================
# Configuration
# =========================


@dataclass(frozen=True)
class CalibrationConfig:
    """Calibration-specific configuration."""

    method: ConformalMethod = ConformalMethod.CQR
    coverage_target: float = 0.90
    calibration_fraction: float = 0.15
    min_calibration_samples: int = 500
    recalibrate_interval: int = 1000  # 0 = never auto-recalibrate

    def __post_init__(self):
        if not 0 < self.coverage_target < 1:
            raise ValueError(f"coverage_target must be in (0, 1), got {self.coverage_target}")
        if not 0 < self.calibration_fraction < 1:
            raise ValueError(
                f"calibration_fraction must be in (0, 1), got {self.calibration_fraction}"
            )
        if self.min_calibration_samples < 10:
            raise ValueError(
                f"min_calibration_samples must be >= 10, got {self.min_calibration_samples}"
            )


@dataclass(frozen=True)
class TimeSeriesConfig:
    """Time series specific configuration (for EnbPI/ACI)."""

    enabled: bool = True
    enbpi_agg_func: Literal["mean", "median"] = "mean"
    aci_gamma: float = 0.01  # ACI adaptation rate
    lookback_window: int = 100  # Window for residual estimation

    def __post_init__(self):
        if not 0 < self.aci_gamma < 1:
            raise ValueError(f"aci_gamma must be in (0, 1), got {self.aci_gamma}")
        if self.lookback_window < 10:
            raise ValueError(f"lookback_window must be >= 10, got {self.lookback_window}")


@dataclass(frozen=True)
class CVaRBoundsConfig:
    """CVaR bounds configuration."""

    enabled: bool = True
    use_for_gae: bool = False  # Experimental: use bounds in GAE computation
    log_interval: int = 100  # Log bounds statistics every N steps


@dataclass(frozen=True)
class RiskIntegrationConfig:
    """Risk management integration configuration."""

    enabled: bool = True
    uncertainty_position_scaling: bool = True
    baseline_interval_width: float = 0.1
    max_uncertainty_reduction: float = 0.5  # Max position reduction factor

    def __post_init__(self):
        if self.baseline_interval_width <= 0:
            raise ValueError(
                f"baseline_interval_width must be > 0, got {self.baseline_interval_width}"
            )
        if not 0 <= self.max_uncertainty_reduction <= 1:
            raise ValueError(
                f"max_uncertainty_reduction must be in [0, 1], got {self.max_uncertainty_reduction}"
            )


@dataclass(frozen=True)
class EscalationConfig:
    """Escalation configuration."""

    enabled: bool = True
    warning_percentile: float = 90.0
    critical_percentile: float = 99.0
    action_on_warning: EscalationAction = EscalationAction.LOG
    action_on_critical: EscalationAction = EscalationAction.REDUCE_POSITION

    def __post_init__(self):
        if not 0 < self.warning_percentile < 100:
            raise ValueError(
                f"warning_percentile must be in (0, 100), got {self.warning_percentile}"
            )
        if not 0 < self.critical_percentile < 100:
            raise ValueError(
                f"critical_percentile must be in (0, 100), got {self.critical_percentile}"
            )
        if self.warning_percentile >= self.critical_percentile:
            raise ValueError(
                f"warning_percentile ({self.warning_percentile}) must be < "
                f"critical_percentile ({self.critical_percentile})"
            )


@dataclass(frozen=True)
class TradingSignalsConfig:
    """Trading signals configuration (option pricing style)."""

    enabled: bool = False
    undervalued_threshold: float = 0.0
    overvalued_threshold: float = 0.0


@dataclass(frozen=True)
class PerformanceConfig:
    """Performance optimization configuration."""

    cache_calibration: bool = True
    async_recalibration: bool = True


@dataclass(frozen=True)
class DebugConfig:
    """Debug configuration."""

    log_residuals: bool = False
    save_calibration_data: bool = False
    validate_coverage: bool = True


@dataclass(frozen=True)
class ConformalConfig:
    """
    Complete conformal prediction configuration.

    Master configuration that aggregates all sub-configurations.
    """

    enabled: bool = True

    # Sub-configurations
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    time_series: TimeSeriesConfig = field(default_factory=TimeSeriesConfig)
    cvar_bounds: CVaRBoundsConfig = field(default_factory=CVaRBoundsConfig)
    risk_integration: RiskIntegrationConfig = field(default_factory=RiskIntegrationConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    trading_signals: TradingSignalsConfig = field(default_factory=TradingSignalsConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    # Convenience accessors
    @property
    def method(self) -> ConformalMethod:
        return self.calibration.method

    @property
    def coverage_target(self) -> float:
        return self.calibration.coverage_target

    @property
    def min_calibration_samples(self) -> int:
        return self.calibration.min_calibration_samples

    @property
    def recalibrate_interval(self) -> int:
        return self.calibration.recalibrate_interval

    @property
    def lookback_window(self) -> int:
        return self.time_series.lookback_window

    @property
    def aci_gamma(self) -> float:
        return self.time_series.aci_gamma

    @property
    def cvar_bounds_enabled(self) -> bool:
        return self.cvar_bounds.enabled

    @property
    def risk_integration_enabled(self) -> bool:
        return self.risk_integration.enabled

    @property
    def uncertainty_position_scaling(self) -> bool:
        return self.risk_integration.uncertainty_position_scaling

    @property
    def baseline_interval_width(self) -> float:
        return self.risk_integration.baseline_interval_width

    @property
    def max_uncertainty_reduction(self) -> float:
        return self.risk_integration.max_uncertainty_reduction

    @property
    def escalation_enabled(self) -> bool:
        return self.escalation.enabled

    @property
    def warning_percentile(self) -> float:
        return self.escalation.warning_percentile

    @property
    def critical_percentile(self) -> float:
        return self.escalation.critical_percentile


# =========================
# Data Types
# =========================


@dataclass(frozen=True)
class PredictionInterval:
    """
    Prediction with conformal bounds.

    Represents a point estimate with calibrated lower and upper bounds
    that provide coverage guarantees under the assumptions of the
    conformal prediction method used.
    """

    point_estimate: float
    lower_bound: float
    upper_bound: float
    coverage_target: float
    method: ConformalMethod
    is_calibrated: bool = False
    calibration_samples: int = 0
    timestamp_ms: int = 0

    @property
    def interval_width(self) -> float:
        """Width of the prediction interval."""
        return self.upper_bound - self.lower_bound

    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty (width / |point_estimate|)."""
        if abs(self.point_estimate) < 1e-8:
            return float("inf") if self.interval_width > 0 else 0.0
        return self.interval_width / abs(self.point_estimate)

    @property
    def is_valid(self) -> bool:
        """Check if interval is valid (non-NaN, proper ordering)."""
        return (
            math.isfinite(self.point_estimate)
            and math.isfinite(self.lower_bound)
            and math.isfinite(self.upper_bound)
            and self.lower_bound <= self.upper_bound
        )

    def contains(self, value: float) -> bool:
        """Check if value is within the prediction interval."""
        return self.lower_bound <= value <= self.upper_bound

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "point_estimate": self.point_estimate,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "coverage_target": self.coverage_target,
            "method": self.method.name,
            "is_calibrated": self.is_calibrated,
            "calibration_samples": self.calibration_samples,
            "interval_width": self.interval_width,
            "relative_uncertainty": self.relative_uncertainty,
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass(frozen=True)
class CVaRBounds:
    """
    CVaR estimate with conformal bounds.

    Provides both CVaR and VaR estimates with calibrated bounds,
    enabling risk-aware decision making under uncertainty.
    """

    cvar_point: float  # Original CVaR estimate
    cvar_lower: float  # Conservative bound (more negative)
    cvar_upper: float  # Optimistic bound (less negative)
    var_point: float  # VaR estimate
    var_lower: float
    var_upper: float
    alpha: float  # CVaR alpha level
    interval: PredictionInterval

    @property
    def worst_case_cvar(self) -> float:
        """Most conservative CVaR estimate."""
        return min(self.cvar_point, self.cvar_lower)

    @property
    def best_case_cvar(self) -> float:
        """Most optimistic CVaR estimate."""
        return max(self.cvar_point, self.cvar_upper)

    @property
    def cvar_uncertainty(self) -> float:
        """Uncertainty in CVaR estimate."""
        return abs(self.cvar_upper - self.cvar_lower)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cvar_point": self.cvar_point,
            "cvar_lower": self.cvar_lower,
            "cvar_upper": self.cvar_upper,
            "var_point": self.var_point,
            "var_lower": self.var_lower,
            "var_upper": self.var_upper,
            "alpha": self.alpha,
            "worst_case_cvar": self.worst_case_cvar,
            "cvar_uncertainty": self.cvar_uncertainty,
            "interval": self.interval.to_dict(),
        }


@dataclass(frozen=True)
class UncertaintyState:
    """
    Current uncertainty state for decision making.

    Provides a snapshot of the uncertainty level and recommended
    actions for risk management.
    """

    current_interval_width: float
    historical_percentile: float  # Where current width falls in history
    escalation_level: EscalationLevel
    recommended_action: EscalationAction
    recommended_position_scale: float  # 0.0 to 1.0
    samples_since_calibration: int
    is_calibrated: bool
    timestamp_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "current_interval_width": self.current_interval_width,
            "historical_percentile": self.historical_percentile,
            "escalation_level": self.escalation_level.name,
            "recommended_action": self.recommended_action.name,
            "recommended_position_scale": self.recommended_position_scale,
            "samples_since_calibration": self.samples_since_calibration,
            "is_calibrated": self.is_calibrated,
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass
class CalibrationResult:
    """Result of a calibration run."""

    success: bool
    samples_used: int
    calibration_quantile: Optional[float] = None
    empirical_coverage: Optional[float] = None
    method: ConformalMethod = ConformalMethod.CQR
    error_message: Optional[str] = None
    timestamp_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "samples_used": self.samples_used,
            "calibration_quantile": self.calibration_quantile,
            "empirical_coverage": self.empirical_coverage,
            "method": self.method.name,
            "error_message": self.error_message,
            "timestamp_ms": self.timestamp_ms,
        }


# =========================
# Protocols
# =========================


@runtime_checkable
class ConformalPredictor(Protocol):
    """Protocol for conformal predictors."""

    def calibrate(
        self,
        predictions: "np.ndarray",  # type: ignore
        true_values: "np.ndarray",  # type: ignore
        **kwargs: Any,
    ) -> CalibrationResult:
        """Calibrate using residuals from validation set."""
        ...

    def predict_interval(
        self, point_prediction: float, coverage: Optional[float] = None, **kwargs: Any
    ) -> PredictionInterval:
        """Get prediction interval for a point prediction."""
        ...

    @property
    def is_calibrated(self) -> bool:
        """Check if predictor has been calibrated."""
        ...

    @property
    def calibration_samples(self) -> int:
        """Number of samples used for calibration."""
        ...


@runtime_checkable
class OnlineConformalPredictor(ConformalPredictor, Protocol):
    """Protocol for online conformal predictors (EnbPI, ACI)."""

    def partial_fit(
        self, predicted: float, true_value: float, timestamp_ms: Optional[int] = None
    ) -> None:
        """Update with new observation (online learning)."""
        ...


@runtime_checkable
class ConformalCVaRProvider(Protocol):
    """Protocol for CVaR with conformal bounds."""

    def compute_cvar_with_bounds(
        self, quantiles: "np.ndarray", alpha: float = 0.05  # type: ignore
    ) -> CVaRBounds:
        """Compute CVaR with conformal prediction bounds."""
        ...


@runtime_checkable
class UncertaintyTracker(Protocol):
    """Protocol for tracking uncertainty over time."""

    def record_interval(
        self, interval: PredictionInterval, timestamp_ms: Optional[int] = None
    ) -> UncertaintyState:
        """Record new interval and return current uncertainty state."""
        ...

    def get_current_state(self) -> UncertaintyState:
        """Get current uncertainty state without recording new interval."""
        ...


# =========================
# Factory Functions
# =========================


def create_conformal_config(config_dict: Dict[str, Any]) -> ConformalConfig:
    """
    Create ConformalConfig from dictionary (e.g., from YAML).

    Args:
        config_dict: Dictionary with configuration values

    Returns:
        ConformalConfig instance

    Example:
        config = create_conformal_config({
            "enabled": True,
            "calibration": {
                "method": "cqr",
                "coverage_target": 0.90,
            },
        })
    """
    if not config_dict.get("enabled", True):
        return ConformalConfig(enabled=False)

    # Parse calibration config
    cal_dict = config_dict.get("calibration", {})
    method_str = cal_dict.get("method", "cqr").upper()
    try:
        method = ConformalMethod[method_str]
    except KeyError:
        method = ConformalMethod.CQR

    calibration = CalibrationConfig(
        method=method,
        coverage_target=cal_dict.get("coverage_target", 0.90),
        calibration_fraction=cal_dict.get("calibration_fraction", 0.15),
        min_calibration_samples=cal_dict.get("min_calibration_samples", 500),
        recalibrate_interval=cal_dict.get("recalibrate_interval", 1000),
    )

    # Parse time series config
    ts_dict = config_dict.get("time_series", {})
    time_series = TimeSeriesConfig(
        enabled=ts_dict.get("enabled", True),
        enbpi_agg_func=ts_dict.get("enbpi_agg_func", "mean"),
        aci_gamma=ts_dict.get("aci_gamma", 0.01),
        lookback_window=ts_dict.get("lookback_window", 100),
    )

    # Parse CVaR bounds config
    cvar_dict = config_dict.get("cvar_bounds", {})
    cvar_bounds = CVaRBoundsConfig(
        enabled=cvar_dict.get("enabled", True),
        use_for_gae=cvar_dict.get("use_for_gae", False),
        log_interval=cvar_dict.get("log_interval", 100),
    )

    # Parse risk integration config
    risk_dict = config_dict.get("risk_integration", {})
    risk_integration = RiskIntegrationConfig(
        enabled=risk_dict.get("enabled", True),
        uncertainty_position_scaling=risk_dict.get("uncertainty_position_scaling", True),
        baseline_interval_width=risk_dict.get("baseline_interval_width", 0.1),
        max_uncertainty_reduction=risk_dict.get("max_uncertainty_reduction", 0.5),
    )

    # Parse escalation config
    esc_dict = config_dict.get("escalation", {})
    action_warning_str = esc_dict.get("action_on_warning", "log").upper()
    action_critical_str = esc_dict.get("action_on_critical", "reduce_position").upper()
    try:
        action_warning = EscalationAction[action_warning_str]
    except KeyError:
        action_warning = EscalationAction.LOG
    try:
        action_critical = EscalationAction[action_critical_str]
    except KeyError:
        action_critical = EscalationAction.REDUCE_POSITION

    escalation = EscalationConfig(
        enabled=esc_dict.get("enabled", True),
        warning_percentile=esc_dict.get("warning_percentile", 90.0),
        critical_percentile=esc_dict.get("critical_percentile", 99.0),
        action_on_warning=action_warning,
        action_on_critical=action_critical,
    )

    # Parse trading signals config
    sig_dict = config_dict.get("trading_signals", {})
    trading_signals = TradingSignalsConfig(
        enabled=sig_dict.get("enabled", False),
        undervalued_threshold=sig_dict.get("undervalued_threshold", 0.0),
        overvalued_threshold=sig_dict.get("overvalued_threshold", 0.0),
    )

    # Parse performance config
    perf_dict = config_dict.get("performance", {})
    performance = PerformanceConfig(
        cache_calibration=perf_dict.get("cache_calibration", True),
        async_recalibration=perf_dict.get("async_recalibration", True),
    )

    # Parse debug config
    debug_dict = config_dict.get("debug", {})
    debug = DebugConfig(
        log_residuals=debug_dict.get("log_residuals", False),
        save_calibration_data=debug_dict.get("save_calibration_data", False),
        validate_coverage=debug_dict.get("validate_coverage", True),
    )

    return ConformalConfig(
        enabled=config_dict.get("enabled", True),
        calibration=calibration,
        time_series=time_series,
        cvar_bounds=cvar_bounds,
        risk_integration=risk_integration,
        escalation=escalation,
        trading_signals=trading_signals,
        performance=performance,
        debug=debug,
    )


def create_disabled_config() -> ConformalConfig:
    """Create a disabled conformal config (zero overhead)."""
    return ConformalConfig(enabled=False)


# =========================
# Utility Functions
# =========================


def validate_config(config: ConformalConfig) -> List[str]:
    """
    Validate conformal config and return list of warnings.

    Returns:
        List of warning messages (empty if no warnings)
    """
    warnings = []

    if config.enabled:
        if config.coverage_target > 0.99:
            warnings.append(
                f"Very high coverage_target ({config.coverage_target}) may lead to "
                "very wide intervals"
            )
        if config.min_calibration_samples < 100:
            warnings.append(
                f"Low min_calibration_samples ({config.min_calibration_samples}) may "
                "lead to unstable calibration"
            )
        if config.method == ConformalMethod.NAIVE:
            warnings.append("NAIVE method provides no theoretical coverage guarantees")
        if config.cvar_bounds.use_for_gae:
            warnings.append("use_for_gae is experimental and may affect training stability")

    return warnings
