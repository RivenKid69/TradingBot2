# -*- coding: utf-8 -*-
"""
L3 LOB Simulation Configuration Models.

This module provides Pydantic-based configuration models for all L3 LOB
subsystems, enabling YAML configuration and validation.

Architecture:
    - Pydantic BaseModel for strict validation
    - Factory methods for preset configurations (equity, crypto)
    - YAML loading support via from_yaml() class methods
    - Integration with execution_providers.py factory functions

Stage 7 of L3 LOB Simulation (v7.0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lob.market_impact import ImpactParameters

import os
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

try:
    from pydantic import BaseModel, Field, field_validator, model_validator

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    # Fallback to dataclass-based configs if Pydantic not available
    from dataclasses import dataclass as BaseModel

    Field = lambda default=None, **kw: default  # type: ignore
    field_validator = lambda *a, **kw: lambda f: f  # type: ignore
    model_validator = lambda *a, **kw: lambda f: f  # type: ignore

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# =============================================================================
# Enums
# =============================================================================


class LatencyDistributionType(str, Enum):
    """Latency distribution types."""

    CONSTANT = "constant"
    UNIFORM = "uniform"
    LOGNORMAL = "lognormal"
    PARETO = "pareto"
    GAMMA = "gamma"
    EMPIRICAL = "empirical"


class LatencyProfileType(str, Enum):
    """Pre-configured latency profiles."""

    COLOCATED = "colocated"  # HFT co-location: ~10-50μs
    PROXIMITY = "proximity"  # Proximity hosting: ~100-500μs
    RETAIL = "retail"  # Retail broker: ~1-10ms
    INSTITUTIONAL = "institutional"  # Institutional: ~200μs-2ms
    CUSTOM = "custom"  # Custom configuration


class FillProbabilityModelType(str, Enum):
    """Fill probability model types."""

    POISSON = "poisson"
    QUEUE_REACTIVE = "queue_reactive"
    HISTORICAL = "historical"
    DISTANCE_BASED = "distance_based"


class ImpactModelType(str, Enum):
    """Market impact model types."""

    KYLE = "kyle"
    ALMGREN_CHRISS = "almgren_chriss"
    GATHERAL = "gatheral"
    LINEAR = "linear"


class DecayFunctionType(str, Enum):
    """Impact decay function types."""

    EXPONENTIAL = "exponential"
    POWER_LAW = "power_law"
    LINEAR = "linear"
    NONE = "none"


class QueueEstimationMethod(str, Enum):
    """Queue position estimation methods."""

    PESSIMISTIC = "pessimistic"  # Assume worst case
    PROBABILISTIC = "probabilistic"  # Monte Carlo
    MBO = "mbo"  # Market-by-Order (exact)


class LOBDataSourceType(str, Enum):
    """LOB data source types."""

    LOBSTER = "lobster"
    ITCH = "itch"
    INTERNAL = "internal"  # Internal simulation
    SYNTHETIC = "synthetic"  # Generated data


# =============================================================================
# Latency Configuration
# =============================================================================


class LatencyComponentConfig(BaseModel):
    """Configuration for a single latency component.

    Attributes:
        enabled: Whether this latency component is enabled
        distribution: Distribution type for sampling
        mean_us: Mean latency in microseconds
        std_us: Standard deviation in microseconds
        min_us: Minimum latency floor
        max_us: Maximum latency cap
        spike_prob: Probability of latency spike [0, 1]
        spike_mult: Multiplier during spike events
        pareto_alpha: Shape parameter for Pareto distribution
        pareto_xmin_us: Minimum value for Pareto distribution
    """

    enabled: bool = True
    distribution: LatencyDistributionType = LatencyDistributionType.LOGNORMAL
    mean_us: float = Field(default=100.0, ge=0)
    std_us: float = Field(default=30.0, ge=0)
    min_us: float = Field(default=1.0, ge=0)
    max_us: float = Field(default=100_000.0, ge=0)
    spike_prob: float = Field(default=0.001, ge=0, le=1)
    spike_mult: float = Field(default=10.0, ge=1)
    pareto_alpha: float = Field(default=2.5, gt=0)
    pareto_xmin_us: float = Field(default=10.0, gt=0)

    if HAS_PYDANTIC:

        @model_validator(mode="after")
        def validate_bounds(self) -> "LatencyComponentConfig":
            """Validate max >= min."""
            if self.max_us < self.min_us:
                raise ValueError("max_us must be >= min_us")
            return self


class LatencyConfig(BaseModel):
    """Complete latency simulation configuration.

    Attributes:
        enabled: Master switch for latency simulation
        profile: Pre-configured profile (overrides individual settings)
        feed_latency: Market data feed latency
        order_latency: Order submission latency
        exchange_latency: Exchange processing latency
        fill_latency: Fill notification latency
        time_of_day_adjustment: Enable time-of-day latency scaling
        volatility_adjustment: Enable volatility-based latency scaling
    """

    enabled: bool = False
    profile: LatencyProfileType = LatencyProfileType.INSTITUTIONAL
    feed_latency: LatencyComponentConfig = Field(
        default_factory=lambda: LatencyComponentConfig(mean_us=200.0, std_us=50.0)
    )
    order_latency: LatencyComponentConfig = Field(
        default_factory=lambda: LatencyComponentConfig(mean_us=300.0, std_us=80.0)
    )
    exchange_latency: LatencyComponentConfig = Field(
        default_factory=lambda: LatencyComponentConfig(mean_us=100.0, std_us=30.0)
    )
    fill_latency: LatencyComponentConfig = Field(
        default_factory=lambda: LatencyComponentConfig(mean_us=150.0, std_us=40.0)
    )
    time_of_day_adjustment: bool = True
    volatility_adjustment: bool = True

    @classmethod
    def colocated(cls) -> "LatencyConfig":
        """Create co-located HFT latency profile (~10-50μs)."""
        return cls(
            enabled=True,
            profile=LatencyProfileType.COLOCATED,
            feed_latency=LatencyComponentConfig(mean_us=10.0, std_us=3.0, max_us=100.0),
            order_latency=LatencyComponentConfig(mean_us=15.0, std_us=5.0, max_us=100.0),
            exchange_latency=LatencyComponentConfig(mean_us=5.0, std_us=2.0, max_us=50.0),
            fill_latency=LatencyComponentConfig(mean_us=10.0, std_us=3.0, max_us=100.0),
        )

    @classmethod
    def retail(cls) -> "LatencyConfig":
        """Create retail broker latency profile (~1-10ms)."""
        return cls(
            enabled=True,
            profile=LatencyProfileType.RETAIL,
            feed_latency=LatencyComponentConfig(mean_us=5000.0, std_us=2000.0, max_us=50_000.0),
            order_latency=LatencyComponentConfig(mean_us=10000.0, std_us=5000.0, max_us=100_000.0),
            exchange_latency=LatencyComponentConfig(mean_us=500.0, std_us=200.0, max_us=5000.0),
            fill_latency=LatencyComponentConfig(mean_us=5000.0, std_us=2000.0, max_us=50_000.0),
        )

    @classmethod
    def institutional(cls) -> "LatencyConfig":
        """Create institutional latency profile (~200μs-2ms)."""
        return cls(
            enabled=True,
            profile=LatencyProfileType.INSTITUTIONAL,
            feed_latency=LatencyComponentConfig(mean_us=200.0, std_us=50.0, max_us=2000.0),
            order_latency=LatencyComponentConfig(mean_us=300.0, std_us=80.0, max_us=3000.0),
            exchange_latency=LatencyComponentConfig(mean_us=100.0, std_us=30.0, max_us=1000.0),
            fill_latency=LatencyComponentConfig(mean_us=150.0, std_us=40.0, max_us=1500.0),
        )


# =============================================================================
# Fill Probability Configuration
# =============================================================================


class FillProbabilityConfig(BaseModel):
    """Fill probability model configuration.

    Attributes:
        enabled: Enable fill probability modeling
        model: Model type to use
        base_rate: Base volume rate (qty/sec) for Poisson model
        queue_decay_alpha: Queue size decay factor for queue-reactive model
        spread_sensitivity_beta: Spread sensitivity for queue-reactive model
        imbalance_factor: Book imbalance sensitivity
        volatility_factor: Volatility sensitivity
        confidence_threshold: Minimum confidence for estimates
    """

    enabled: bool = True
    model: FillProbabilityModelType = FillProbabilityModelType.QUEUE_REACTIVE
    base_rate: float = Field(default=100.0, gt=0)
    queue_decay_alpha: float = Field(default=0.01, ge=0)
    spread_sensitivity_beta: float = Field(default=0.5, ge=0)
    imbalance_factor: float = Field(default=0.3, ge=0)
    volatility_factor: float = Field(default=0.2, ge=0)
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)


# =============================================================================
# Queue Value Configuration
# =============================================================================


class QueueValueConfig(BaseModel):
    """Queue value computation configuration (Moallemi & Yuan).

    Attributes:
        enabled: Enable queue value computation
        hold_threshold: Queue value threshold for HOLD decision
        cancel_threshold: Queue value threshold for CANCEL decision
        adverse_selection_lambda: Adverse selection parameter
        time_horizon_sec: Default time horizon for value computation
        discount_rate: Discount rate for future fills
    """

    enabled: bool = True
    hold_threshold: float = Field(default=0.001)
    cancel_threshold: float = Field(default=-0.002)
    adverse_selection_lambda: float = Field(default=0.1, ge=0)
    time_horizon_sec: float = Field(default=60.0, gt=0)
    discount_rate: float = Field(default=0.0, ge=0)


# =============================================================================
# Market Impact Configuration
# =============================================================================


class MarketImpactConfig(BaseModel):
    """Market impact model configuration.

    Attributes:
        enabled: Enable market impact modeling
        model: Impact model type
        eta: Temporary impact coefficient (η)
        gamma: Permanent impact coefficient (γ)
        delta: Impact exponent (δ), typically 0.5 for square-root
        tau_ms: Decay time constant in milliseconds
        beta: Power-law decay exponent
        decay_type: Decay function type
        apply_to_lob: Apply impact effects to LOB state
        momentum_detection: Enable momentum/trend detection
    """

    enabled: bool = True
    model: ImpactModelType = ImpactModelType.ALMGREN_CHRISS
    eta: float = Field(default=0.05, ge=0)  # Temporary
    gamma: float = Field(default=0.03, ge=0)  # Permanent
    delta: float = Field(default=0.5, gt=0)  # Exponent
    tau_ms: float = Field(default=30_000, gt=0)  # Decay time
    beta: float = Field(default=1.5, gt=0)  # Power-law exponent
    decay_type: DecayFunctionType = DecayFunctionType.POWER_LAW
    apply_to_lob: bool = True
    momentum_detection: bool = True

    @classmethod
    def for_equity(cls) -> "MarketImpactConfig":
        """Create impact config for US equities."""
        return cls(
            enabled=True,
            model=ImpactModelType.ALMGREN_CHRISS,
            eta=0.05,
            gamma=0.03,
            delta=0.5,
            tau_ms=30_000,
            beta=1.5,
        )

    @classmethod
    def for_crypto(cls) -> "MarketImpactConfig":
        """Create impact config for crypto markets."""
        return cls(
            enabled=True,
            model=ImpactModelType.ALMGREN_CHRISS,
            eta=0.10,
            gamma=0.05,
            delta=0.5,
            tau_ms=60_000,
            beta=1.2,
        )


# =============================================================================
# Hidden Liquidity Configuration
# =============================================================================


class IcebergConfig(BaseModel):
    """Iceberg order detection configuration.

    Attributes:
        enabled: Enable iceberg detection
        min_refills_to_confirm: Minimum refills to confirm iceberg
        lookback_window_sec: Time window for refill pattern detection
        min_display_size: Minimum display size to track
        hidden_ratio_estimate: Default hidden:display ratio estimate
    """

    enabled: bool = True
    min_refills_to_confirm: int = Field(default=2, ge=1)
    lookback_window_sec: float = Field(default=60.0, gt=0)
    min_display_size: float = Field(default=10.0, ge=0)
    hidden_ratio_estimate: float = Field(default=0.15, ge=0, le=1)


class HiddenLiquidityConfig(BaseModel):
    """Hidden liquidity estimation configuration.

    Attributes:
        enabled: Enable hidden liquidity estimation
        iceberg: Iceberg detection settings
        default_hidden_ratio: Default hidden liquidity ratio
        use_historical_estimates: Use historical data for estimates
    """

    enabled: bool = True
    iceberg: IcebergConfig = Field(default_factory=IcebergConfig)
    default_hidden_ratio: float = Field(default=0.15, ge=0, le=1)
    use_historical_estimates: bool = True


# =============================================================================
# Dark Pool Configuration
# =============================================================================


class DarkPoolVenueConfig(BaseModel):
    """Configuration for a single dark pool venue.

    Attributes:
        venue_id: Unique venue identifier
        venue_type: Venue type (midpoint_cross, etc.)
        enabled: Whether venue is active
        min_order_size: Minimum order size
        max_order_size: Maximum order size (0 = unlimited)
        base_fill_probability: Base probability of fill
        size_penalty_factor: Size impact on fill probability
        info_leakage_probability: Probability of information leakage
        latency_ms: Venue latency in milliseconds
    """

    venue_id: str
    venue_type: str = "midpoint_cross"
    enabled: bool = True
    min_order_size: float = Field(default=100.0, ge=0)
    max_order_size: float = Field(default=0.0, ge=0)  # 0 = unlimited
    base_fill_probability: float = Field(default=0.30, ge=0, le=1)
    size_penalty_factor: float = Field(default=0.5, ge=0)
    info_leakage_probability: float = Field(default=0.10, ge=0, le=1)
    latency_ms: float = Field(default=5.0, ge=0)


class DarkPoolsConfig(BaseModel):
    """Dark pool simulation configuration.

    Attributes:
        enabled: Enable dark pool simulation
        venues: List of dark pool venue configurations
        max_dark_fill_pct: Maximum % of order to fill in dark pools
        routing_strategy: Smart order routing strategy
    """

    enabled: bool = False
    venues: List[DarkPoolVenueConfig] = Field(
        default_factory=lambda: [
            DarkPoolVenueConfig(
                venue_id="sigma_x",
                venue_type="midpoint_cross",
                base_fill_probability=0.30,
            ),
            DarkPoolVenueConfig(
                venue_id="iex_dark",
                venue_type="midpoint_cross",
                min_order_size=50.0,
                base_fill_probability=0.25,
            ),
        ]
    )
    max_dark_fill_pct: float = Field(default=0.5, ge=0, le=1)
    routing_strategy: str = "sequential"  # sequential, parallel, smart


# =============================================================================
# Queue Tracking Configuration
# =============================================================================


class QueueTrackingConfig(BaseModel):
    """Queue position tracking configuration.

    Attributes:
        enabled: Enable queue tracking
        estimation_method: Method for queue position estimation
        use_mbo_when_available: Use MBO data when available
        max_tracked_orders: Maximum orders to track
        cleanup_interval_sec: Interval for cleaning up stale orders
    """

    enabled: bool = True
    estimation_method: QueueEstimationMethod = QueueEstimationMethod.PESSIMISTIC
    use_mbo_when_available: bool = True
    max_tracked_orders: int = Field(default=10_000, ge=1)
    cleanup_interval_sec: float = Field(default=60.0, gt=0)


# =============================================================================
# Event Scheduling Configuration
# =============================================================================


class EventSchedulingConfig(BaseModel):
    """Event scheduler configuration.

    Attributes:
        enabled: Enable event scheduling
        race_condition_buffer_us: Buffer for race condition detection
        max_events_per_step: Maximum events to process per step
        deterministic_ordering: Use deterministic tie-breaking
    """

    enabled: bool = True
    race_condition_buffer_us: float = Field(default=100.0, ge=0)
    max_events_per_step: int = Field(default=1000, ge=1)
    deterministic_ordering: bool = True


# =============================================================================
# LOB Data Source Configuration
# =============================================================================


class LOBDataConfig(BaseModel):
    """LOB data source configuration.

    Attributes:
        source: Data source type
        reconstruction_mode: Full or incremental LOB reconstruction
        max_depth: Maximum depth levels to maintain
        snapshot_interval_ms: Interval between snapshots
        message_path: Path to LOBSTER/ITCH message files
    """

    source: LOBDataSourceType = LOBDataSourceType.INTERNAL
    reconstruction_mode: str = "full"  # full or incremental
    max_depth: int = Field(default=20, ge=1)
    snapshot_interval_ms: int = Field(default=1000, ge=0)
    message_path: Optional[str] = None


# =============================================================================
# Main L3 Configuration
# =============================================================================


class L3ExecutionConfig(BaseModel):
    """
    Complete L3 LOB execution simulation configuration.

    This is the main configuration class that combines all L3 subsystems.

    Attributes:
        enabled: Master switch for L3 simulation
        lob_data: LOB data source configuration
        latency: Latency simulation configuration
        fill_probability: Fill probability model configuration
        queue_value: Queue value computation configuration
        market_impact: Market impact model configuration
        hidden_liquidity: Hidden liquidity estimation configuration
        dark_pools: Dark pool simulation configuration
        queue_tracking: Queue position tracking configuration
        event_scheduling: Event scheduler configuration
    """

    enabled: bool = False
    lob_data: LOBDataConfig = Field(default_factory=LOBDataConfig)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    fill_probability: FillProbabilityConfig = Field(default_factory=FillProbabilityConfig)
    queue_value: QueueValueConfig = Field(default_factory=QueueValueConfig)
    market_impact: MarketImpactConfig = Field(default_factory=MarketImpactConfig)
    hidden_liquidity: HiddenLiquidityConfig = Field(default_factory=HiddenLiquidityConfig)
    dark_pools: DarkPoolsConfig = Field(default_factory=DarkPoolsConfig)
    queue_tracking: QueueTrackingConfig = Field(default_factory=QueueTrackingConfig)
    event_scheduling: EventSchedulingConfig = Field(default_factory=EventSchedulingConfig)

    @classmethod
    def for_equity(cls) -> "L3ExecutionConfig":
        """Create L3 config optimized for US equities."""
        return cls(
            enabled=True,
            latency=LatencyConfig.institutional(),
            market_impact=MarketImpactConfig.for_equity(),
            fill_probability=FillProbabilityConfig(enabled=True),
            queue_value=QueueValueConfig(enabled=True),
            hidden_liquidity=HiddenLiquidityConfig(enabled=True),
            dark_pools=DarkPoolsConfig(enabled=True),
        )

    @classmethod
    def for_crypto(cls) -> "L3ExecutionConfig":
        """Create L3 config optimized for crypto markets.

        Note: Crypto typically uses Cython LOB (fast_lob.pyx) for performance.
        This Python L3 config is for equity-style simulation of crypto.
        """
        return cls(
            enabled=True,
            latency=LatencyConfig(
                enabled=True,
                profile=LatencyProfileType.RETAIL,  # Most crypto users are retail
            ),
            market_impact=MarketImpactConfig.for_crypto(),
            fill_probability=FillProbabilityConfig(enabled=True, base_rate=200.0),
            queue_value=QueueValueConfig(enabled=True),
            hidden_liquidity=HiddenLiquidityConfig(enabled=False),  # Less relevant for crypto
            dark_pools=DarkPoolsConfig(enabled=False),  # No crypto dark pools
        )

    @classmethod
    def minimal(cls) -> "L3ExecutionConfig":
        """Create minimal L3 config (matching engine only)."""
        return cls(
            enabled=True,
            latency=LatencyConfig(enabled=False),
            fill_probability=FillProbabilityConfig(enabled=False),
            queue_value=QueueValueConfig(enabled=False),
            market_impact=MarketImpactConfig(enabled=False),
            hidden_liquidity=HiddenLiquidityConfig(enabled=False),
            dark_pools=DarkPoolsConfig(enabled=False),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "L3ExecutionConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            L3ExecutionConfig instance

        Raises:
            FileNotFoundError: If the config file does not exist
            ImportError: If PyYAML is not installed
        """
        if not HAS_YAML:
            raise ImportError(
                "PyYAML is required for YAML loading. Install with: pip install pyyaml"
            )

        # Validate path exists before opening
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"L3 config file not found: {path}")
        if not config_path.is_file():
            raise ValueError(f"L3 config path is not a file: {path}")

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        # Handle nested 'l3_simulation' or 'execution' key
        if "l3_simulation" in data:
            data = data["l3_simulation"]
        elif "execution" in data:
            data = data["execution"]

        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "L3ExecutionConfig":
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            L3ExecutionConfig instance
        """
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary with enum values serialized as strings
            for YAML compatibility.
        """
        if HAS_PYDANTIC:
            # Use mode='json' to serialize enum values as strings
            # This ensures YAML roundtrip compatibility with safe_load()
            return self.model_dump(mode="json")
        else:
            # Fallback for dataclass
            import dataclasses

            return dataclasses.asdict(self)  # type: ignore

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML file
        """
        if not HAS_YAML:
            raise ImportError(
                "PyYAML is required for YAML saving. Install with: pip install pyyaml"
            )

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @property
    def is_enabled(self) -> bool:
        """Check if any L3 feature is enabled."""
        return self.enabled and any(
            [
                self.latency.enabled,
                self.fill_probability.enabled,
                self.queue_value.enabled,
                self.market_impact.enabled,
                self.hidden_liquidity.enabled,
                self.dark_pools.enabled,
            ]
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_l3_config(
    preset: str = "equity",
    **overrides: Any,
) -> L3ExecutionConfig:
    """
    Factory function to create L3 configuration.

    Args:
        preset: Configuration preset ("equity", "crypto", "minimal", "custom")
        **overrides: Override specific configuration values

    Returns:
        L3ExecutionConfig instance
    """
    preset_lower = preset.lower()

    if preset_lower == "equity":
        config = L3ExecutionConfig.for_equity()
    elif preset_lower == "crypto":
        config = L3ExecutionConfig.for_crypto()
    elif preset_lower == "minimal":
        config = L3ExecutionConfig.minimal()
    else:
        config = L3ExecutionConfig()

    # Apply overrides
    if overrides and HAS_PYDANTIC:
        config_dict = config.model_dump()
        _deep_update(config_dict, overrides)
        config = L3ExecutionConfig(**config_dict)

    return config


def _deep_update(base: Dict, updates: Dict) -> Dict:
    """Deep merge updates into base dictionary."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


# =============================================================================
# Backward Compatibility Helpers
# =============================================================================


def latency_config_to_dataclass(config: LatencyComponentConfig) -> "LatencyConfig":
    """Convert Pydantic LatencyComponentConfig to lob.latency_model.LatencyConfig dataclass."""
    from lob.latency_model import LatencyConfig as DCLatencyConfig, LatencyDistribution

    dist_map = {
        LatencyDistributionType.CONSTANT: LatencyDistribution.CONSTANT,
        LatencyDistributionType.UNIFORM: LatencyDistribution.UNIFORM,
        LatencyDistributionType.LOGNORMAL: LatencyDistribution.LOGNORMAL,
        LatencyDistributionType.PARETO: LatencyDistribution.PARETO,
        LatencyDistributionType.GAMMA: LatencyDistribution.GAMMA,
        LatencyDistributionType.EMPIRICAL: LatencyDistribution.EMPIRICAL,
    }

    return DCLatencyConfig(
        distribution=dist_map.get(config.distribution, LatencyDistribution.LOGNORMAL),
        mean_us=config.mean_us,
        std_us=config.std_us,
        min_us=config.min_us,
        max_us=config.max_us,
        spike_prob=config.spike_prob,
        spike_mult=config.spike_mult,
        pareto_alpha=config.pareto_alpha,
        pareto_xmin_us=config.pareto_xmin_us,
    )


def impact_config_to_parameters(config: MarketImpactConfig) -> "ImpactParameters":
    """Convert Pydantic MarketImpactConfig to lob.market_impact.ImpactParameters dataclass."""
    from lob.market_impact import ImpactParameters

    return ImpactParameters(
        eta=config.eta,
        gamma=config.gamma,
        delta=config.delta,
        tau_ms=config.tau_ms,
        beta=config.beta,
    )


__all__ = [
    # Enums
    "LatencyDistributionType",
    "LatencyProfileType",
    "FillProbabilityModelType",
    "ImpactModelType",
    "DecayFunctionType",
    "QueueEstimationMethod",
    "LOBDataSourceType",
    # Configuration Classes
    "LatencyComponentConfig",
    "LatencyConfig",
    "FillProbabilityConfig",
    "QueueValueConfig",
    "MarketImpactConfig",
    "IcebergConfig",
    "HiddenLiquidityConfig",
    "DarkPoolVenueConfig",
    "DarkPoolsConfig",
    "QueueTrackingConfig",
    "EventSchedulingConfig",
    "LOBDataConfig",
    "L3ExecutionConfig",
    # Factory Functions
    "create_l3_config",
    # Backward Compatibility
    "latency_config_to_dataclass",
    "impact_config_to_parameters",
]
