"""
Realistic latency simulation for L3 LOB.

This module provides high-fidelity latency modeling for market microstructure
simulation, supporting:
- Separate feed/order/exchange/fill latencies
- Log-normal, Pareto, and Gamma distributions
- Time-of-day seasonality adjustments
- Co-location vs retail simulation profiles
- Latency spikes (fat tails)

Reference: hftbacktest latency modeling
https://hftbacktest.readthedocs.io/en/latest/

Timestamp Convention:
    - Internal calculations use MICROSECONDS (us) for precision
    - External API returns NANOSECONDS (ns) for LOB integration
    - Conversion: ns = us * 1000

Architecture:
    This module is SEPARATE from the existing latency.py/impl_latency.py
    which handle execution simulator latency. This module provides
    microsecond-precision latency for L3 order book simulation.

Stage 5 of L3 LOB Simulation (v5.0)
"""

from __future__ import annotations

import math
import random
import threading
import warnings
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# Try to import numpy for better distribution sampling
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class LatencyDistribution(IntEnum):
    """Distribution types for latency sampling."""

    CONSTANT = 0  # Fixed latency (for testing)
    UNIFORM = 1  # Uniform distribution
    LOGNORMAL = 2  # Log-normal (typical for network latency)
    PARETO = 3  # Heavy-tailed (for rare spikes)
    GAMMA = 4  # Gamma distribution (flexible shape)
    EMPIRICAL = 5  # Sample from historical data


class LatencyProfile(IntEnum):
    """Pre-configured latency profiles."""

    COLOCATED = 0  # Co-located HFT: ~10-50us total
    PROXIMITY = 1  # Proximity hosting: ~100-500us
    RETAIL = 2  # Retail broker: ~1-10ms
    INSTITUTIONAL = 3  # Institutional: ~200us-2ms
    CUSTOM = 4  # Custom configuration


@dataclass
class LatencyConfig:
    """Configuration for a single latency component.

    Attributes:
        distribution: Distribution type for sampling
        mean_us: Mean latency in microseconds
        std_us: Standard deviation in microseconds (for log-normal/gamma)
        min_us: Minimum latency floor
        max_us: Maximum latency cap
        spike_prob: Probability of a latency spike [0, 1]
        spike_mult: Multiplier during spike events
        pareto_alpha: Shape parameter for Pareto distribution (higher = lighter tail)
        pareto_xmin_us: Minimum value for Pareto distribution
    """

    distribution: LatencyDistribution = LatencyDistribution.LOGNORMAL
    mean_us: float = 100.0
    std_us: float = 30.0
    min_us: float = 1.0
    max_us: float = 100_000.0  # 100ms cap
    spike_prob: float = 0.001  # 0.1% spike probability
    spike_mult: float = 10.0  # 10x during spikes
    pareto_alpha: float = 2.5  # Pareto shape (higher = lighter tail)
    pareto_xmin_us: float = 10.0  # Pareto minimum

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.mean_us < 0:
            raise ValueError("mean_us must be non-negative")
        if self.std_us < 0:
            raise ValueError("std_us must be non-negative")
        if self.min_us < 0:
            raise ValueError("min_us must be non-negative")
        if self.max_us < self.min_us:
            raise ValueError("max_us must be >= min_us")
        if not 0 <= self.spike_prob <= 1:
            raise ValueError("spike_prob must be in [0, 1]")
        if self.spike_mult < 1:
            raise ValueError("spike_mult must be >= 1")
        if self.pareto_alpha <= 0:
            raise ValueError("pareto_alpha must be positive")
        if self.pareto_xmin_us <= 0:
            raise ValueError("pareto_xmin_us must be positive")

        # Warn about extreme Pareto alpha values
        if self.distribution == LatencyDistribution.PARETO:
            if self.pareto_alpha <= 1:
                warnings.warn(
                    f"Pareto alpha={self.pareto_alpha} <= 1 has undefined mean. "
                    "Results may be extreme. Consider alpha > 2 for stable statistics.",
                    UserWarning,
                    stacklevel=2,
                )
            elif self.pareto_alpha <= 2:
                warnings.warn(
                    f"Pareto alpha={self.pareto_alpha} <= 2 has infinite variance. "
                    "Results may have high variability. Consider alpha > 2.",
                    UserWarning,
                    stacklevel=2,
                )


@dataclass
class LatencySample:
    """Result of a latency sample.

    Attributes:
        latency_ns: Sampled latency in nanoseconds
        latency_us: Sampled latency in microseconds
        is_spike: Whether this was a spike event
        distribution_used: Which distribution generated this sample
        raw_sample_us: Raw sample before bounds/spike adjustment
    """

    latency_ns: int
    latency_us: float
    is_spike: bool = False
    distribution_used: LatencyDistribution = LatencyDistribution.LOGNORMAL
    raw_sample_us: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "latency_ns": self.latency_ns,
            "latency_us": self.latency_us,
            "is_spike": self.is_spike,
            "distribution": self.distribution_used.name,
            "raw_sample_us": self.raw_sample_us,
        }


class LatencySampler:
    """Sampler for a single latency component.

    Thread-safe latency sampling with configurable distributions.
    """

    def __init__(
        self,
        config: LatencyConfig,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize sampler.

        Args:
            config: Latency configuration
            seed: Random seed for reproducibility
        """
        self._config = config
        self._lock = threading.Lock()

        # Initialize RNG
        if seed is not None:
            self._rng = random.Random(seed)
            if HAS_NUMPY:
                self._np_rng = np.random.RandomState(seed)
            else:
                self._np_rng = None
        else:
            self._rng = random.Random()
            if HAS_NUMPY:
                self._np_rng = np.random.RandomState()
            else:
                self._np_rng = None

        # Empirical data storage
        self._empirical_samples: List[float] = []

        # Statistics
        self._sample_count = 0
        self._spike_count = 0
        self._samples: List[float] = []
        self._max_samples = 10000  # Limit memory usage

    @property
    def config(self) -> LatencyConfig:
        """Get current configuration."""
        return self._config

    def update_config(self, config: LatencyConfig) -> None:
        """Update configuration (thread-safe)."""
        with self._lock:
            self._config = config

    def set_empirical_data(self, samples_us: Sequence[float]) -> None:
        """Set empirical samples for EMPIRICAL distribution.

        Args:
            samples_us: Historical latency samples in microseconds
        """
        with self._lock:
            self._empirical_samples = [max(0.0, float(s)) for s in samples_us]

    def _sample_distribution(self, cfg: LatencyConfig) -> float:
        """Sample from configured distribution (internal, not thread-safe)."""
        dist = cfg.distribution

        if dist == LatencyDistribution.CONSTANT:
            return cfg.mean_us

        elif dist == LatencyDistribution.UNIFORM:
            # Uniform around mean +/- std
            low = max(cfg.min_us, cfg.mean_us - cfg.std_us)
            high = cfg.mean_us + cfg.std_us
            return self._rng.uniform(low, high)

        elif dist == LatencyDistribution.LOGNORMAL:
            # Log-normal: mu and sigma are of the underlying normal
            # Convert mean/std to log-space parameters
            if cfg.std_us <= 0:
                return cfg.mean_us

            # For log-normal: E[X] = exp(mu + sigma^2/2)
            # Var[X] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
            # Solving: sigma^2 = ln(1 + (std/mean)^2)
            #          mu = ln(mean) - sigma^2/2
            mean = max(1.0, cfg.mean_us)
            std = cfg.std_us

            cv_sq = (std / mean) ** 2
            sigma_sq = math.log1p(cv_sq)
            sigma = math.sqrt(sigma_sq)
            mu = math.log(mean) - sigma_sq / 2

            if HAS_NUMPY and self._np_rng is not None:
                return float(self._np_rng.lognormal(mu, sigma))
            else:
                # Use inverse transform
                z = self._rng.gauss(0, 1)
                return math.exp(mu + sigma * z)

        elif dist == LatencyDistribution.PARETO:
            # Pareto: P(X > x) = (x_min/x)^alpha for x >= x_min
            alpha = cfg.pareto_alpha
            xmin = cfg.pareto_xmin_us

            # Inverse CDF: x = x_min / (1-u)^(1/alpha)
            u = self._rng.random()
            # Avoid division by zero
            u = max(u, 1e-10)
            return xmin / (u ** (1.0 / alpha))

        elif dist == LatencyDistribution.GAMMA:
            # Gamma distribution
            # mean = k * theta, var = k * theta^2
            # So: k = mean^2 / var, theta = var / mean
            if cfg.std_us <= 0:
                return cfg.mean_us

            mean = max(1.0, cfg.mean_us)
            var = cfg.std_us**2

            k = mean**2 / var  # shape
            theta = var / mean  # scale

            if HAS_NUMPY and self._np_rng is not None:
                return float(self._np_rng.gamma(k, theta))
            else:
                return self._rng.gammavariate(k, theta)

        elif dist == LatencyDistribution.EMPIRICAL:
            if not self._empirical_samples:
                # Fall back to log-normal if no data
                return cfg.mean_us
            return self._rng.choice(self._empirical_samples)

        else:
            return cfg.mean_us

    def sample(self) -> LatencySample:
        """Sample latency (thread-safe).

        Returns:
            LatencySample with latency in nanoseconds and metadata
        """
        with self._lock:
            cfg = self._config

            # Sample from distribution
            raw_us = self._sample_distribution(cfg)

            # Check for spike
            is_spike = self._rng.random() < cfg.spike_prob
            if is_spike:
                raw_us *= cfg.spike_mult
                self._spike_count += 1

            # Apply bounds
            clamped_us = max(cfg.min_us, min(cfg.max_us, raw_us))

            # Convert to nanoseconds
            latency_ns = int(round(clamped_us * 1000))

            # Track statistics
            self._sample_count += 1
            if len(self._samples) < self._max_samples:
                self._samples.append(clamped_us)

            return LatencySample(
                latency_ns=latency_ns,
                latency_us=clamped_us,
                is_spike=is_spike,
                distribution_used=cfg.distribution,
                raw_sample_us=raw_us,
            )

    def sample_ns(self) -> int:
        """Sample and return latency in nanoseconds."""
        return self.sample().latency_ns

    def sample_us(self) -> float:
        """Sample and return latency in microseconds."""
        return self.sample().latency_us

    def stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        with self._lock:
            n = len(self._samples)
            if n == 0:
                return {
                    "count": 0,
                    "spike_rate": 0.0,
                    "mean_us": 0.0,
                    "std_us": 0.0,
                    "p50_us": 0.0,
                    "p95_us": 0.0,
                    "p99_us": 0.0,
                    "min_us": 0.0,
                    "max_us": 0.0,
                }

            sorted_samples = sorted(self._samples)

            def percentile(p: float) -> float:
                if n == 1:
                    return sorted_samples[0]
                k = (n - 1) * p
                f = int(k)
                c = min(f + 1, n - 1)
                if f == c:
                    return sorted_samples[f]
                return sorted_samples[f] + (sorted_samples[c] - sorted_samples[f]) * (k - f)

            mean_val = sum(self._samples) / n
            var_val = sum((x - mean_val) ** 2 for x in self._samples) / n if n > 1 else 0.0

            return {
                "count": self._sample_count,
                "spike_rate": (
                    self._spike_count / self._sample_count if self._sample_count > 0 else 0.0
                ),
                "mean_us": mean_val,
                "std_us": math.sqrt(var_val),
                "p50_us": percentile(0.5),
                "p95_us": percentile(0.95),
                "p99_us": percentile(0.99),
                "min_us": sorted_samples[0],
                "max_us": sorted_samples[-1],
            }

    def reset_stats(self) -> None:
        """Reset statistics."""
        with self._lock:
            self._sample_count = 0
            self._spike_count = 0
            self._samples.clear()


@dataclass
class LatencyModelConfig:
    """Configuration for complete latency model.

    Attributes:
        feed_config: Latency for receiving market data feed
        order_config: Latency for order submission to exchange
        exchange_config: Exchange internal processing time
        fill_config: Latency for receiving fill notifications
        profile: Pre-configured profile (overrides individual configs)
        seasonality_multipliers: 24-element array for hourly adjustments
        volatility_sensitivity: How much volatility affects latency [0, 1]
    """

    feed_config: Optional[LatencyConfig] = None
    order_config: Optional[LatencyConfig] = None
    exchange_config: Optional[LatencyConfig] = None
    fill_config: Optional[LatencyConfig] = None
    profile: LatencyProfile = LatencyProfile.CUSTOM
    seasonality_multipliers: Optional[Sequence[float]] = None
    volatility_sensitivity: float = 0.0
    seed: Optional[int] = None

    @classmethod
    def from_profile(
        cls, profile: LatencyProfile, seed: Optional[int] = None
    ) -> "LatencyModelConfig":
        """Create config from pre-defined profile."""
        if profile == LatencyProfile.COLOCATED:
            # Co-located HFT: ~10-50us
            return cls(
                feed_config=LatencyConfig(
                    distribution=LatencyDistribution.LOGNORMAL,
                    mean_us=10.0,
                    std_us=3.0,
                    min_us=5.0,
                    max_us=100.0,
                    spike_prob=0.0001,
                    spike_mult=5.0,
                ),
                order_config=LatencyConfig(
                    distribution=LatencyDistribution.LOGNORMAL,
                    mean_us=15.0,
                    std_us=5.0,
                    min_us=8.0,
                    max_us=150.0,
                    spike_prob=0.0001,
                    spike_mult=5.0,
                ),
                exchange_config=LatencyConfig(
                    distribution=LatencyDistribution.LOGNORMAL,
                    mean_us=5.0,
                    std_us=2.0,
                    min_us=1.0,
                    max_us=50.0,
                    spike_prob=0.00001,
                    spike_mult=3.0,
                ),
                fill_config=LatencyConfig(
                    distribution=LatencyDistribution.LOGNORMAL,
                    mean_us=12.0,
                    std_us=4.0,
                    min_us=6.0,
                    max_us=120.0,
                    spike_prob=0.0001,
                    spike_mult=5.0,
                ),
                profile=profile,
                seed=seed,
            )

        elif profile == LatencyProfile.PROXIMITY:
            # Proximity hosting: ~100-500us
            return cls(
                feed_config=LatencyConfig(
                    distribution=LatencyDistribution.LOGNORMAL,
                    mean_us=100.0,
                    std_us=30.0,
                    min_us=50.0,
                    max_us=1000.0,
                    spike_prob=0.001,
                    spike_mult=5.0,
                ),
                order_config=LatencyConfig(
                    distribution=LatencyDistribution.LOGNORMAL,
                    mean_us=200.0,
                    std_us=50.0,
                    min_us=100.0,
                    max_us=2000.0,
                    spike_prob=0.001,
                    spike_mult=5.0,
                ),
                exchange_config=LatencyConfig(
                    distribution=LatencyDistribution.LOGNORMAL,
                    mean_us=10.0,
                    std_us=3.0,
                    min_us=5.0,
                    max_us=100.0,
                    spike_prob=0.0001,
                    spike_mult=3.0,
                ),
                fill_config=LatencyConfig(
                    distribution=LatencyDistribution.LOGNORMAL,
                    mean_us=150.0,
                    std_us=40.0,
                    min_us=70.0,
                    max_us=1500.0,
                    spike_prob=0.001,
                    spike_mult=5.0,
                ),
                profile=profile,
                seed=seed,
            )

        elif profile == LatencyProfile.RETAIL:
            # Retail broker: ~1-10ms
            return cls(
                feed_config=LatencyConfig(
                    distribution=LatencyDistribution.LOGNORMAL,
                    mean_us=2000.0,  # 2ms
                    std_us=500.0,
                    min_us=500.0,
                    max_us=20000.0,
                    spike_prob=0.01,
                    spike_mult=5.0,
                ),
                order_config=LatencyConfig(
                    distribution=LatencyDistribution.LOGNORMAL,
                    mean_us=5000.0,  # 5ms
                    std_us=1500.0,
                    min_us=1000.0,
                    max_us=50000.0,
                    spike_prob=0.01,
                    spike_mult=5.0,
                ),
                exchange_config=LatencyConfig(
                    distribution=LatencyDistribution.LOGNORMAL,
                    mean_us=50.0,
                    std_us=15.0,
                    min_us=20.0,
                    max_us=500.0,
                    spike_prob=0.001,
                    spike_mult=3.0,
                ),
                fill_config=LatencyConfig(
                    distribution=LatencyDistribution.LOGNORMAL,
                    mean_us=3000.0,  # 3ms
                    std_us=800.0,
                    min_us=700.0,
                    max_us=30000.0,
                    spike_prob=0.01,
                    spike_mult=5.0,
                ),
                profile=profile,
                seed=seed,
            )

        elif profile == LatencyProfile.INSTITUTIONAL:
            # Institutional: ~200us-2ms
            return cls(
                feed_config=LatencyConfig(
                    distribution=LatencyDistribution.LOGNORMAL,
                    mean_us=300.0,
                    std_us=80.0,
                    min_us=100.0,
                    max_us=3000.0,
                    spike_prob=0.005,
                    spike_mult=5.0,
                ),
                order_config=LatencyConfig(
                    distribution=LatencyDistribution.LOGNORMAL,
                    mean_us=500.0,
                    std_us=150.0,
                    min_us=200.0,
                    max_us=5000.0,
                    spike_prob=0.005,
                    spike_mult=5.0,
                ),
                exchange_config=LatencyConfig(
                    distribution=LatencyDistribution.LOGNORMAL,
                    mean_us=20.0,
                    std_us=5.0,
                    min_us=10.0,
                    max_us=200.0,
                    spike_prob=0.0005,
                    spike_mult=3.0,
                ),
                fill_config=LatencyConfig(
                    distribution=LatencyDistribution.LOGNORMAL,
                    mean_us=400.0,
                    std_us=100.0,
                    min_us=150.0,
                    max_us=4000.0,
                    spike_prob=0.005,
                    spike_mult=5.0,
                ),
                profile=profile,
                seed=seed,
            )

        else:  # CUSTOM - use defaults
            return cls(
                feed_config=LatencyConfig(),
                order_config=LatencyConfig(),
                exchange_config=LatencyConfig(mean_us=10.0, std_us=3.0),
                fill_config=LatencyConfig(),
                profile=LatencyProfile.CUSTOM,
                seed=seed,
            )


class LatencyModel:
    """
    Realistic latency simulation for L3 LOB.

    Models four latency components:
    1. Feed latency: when we receive market data updates
    2. Order latency: when our order reaches the exchange
    3. Exchange latency: internal exchange processing time
    4. Fill latency: when we receive fill notifications

    Total round-trip: order_latency + exchange_latency + fill_latency

    Reference: hftbacktest latency modeling
    https://hftbacktest.readthedocs.io/en/latest/

    Timestamp Convention:
        - Returns NANOSECONDS for LOB integration
        - Internal calculations use MICROSECONDS for precision

    Precision Limitation:
        Internal calculations are performed in microseconds (μs) and then
        converted to nanoseconds (ns) via multiplication by 1000. This means:
        - Minimum effective latency is ~1μs (1000ns)
        - Sub-microsecond precision (< 1μs) is NOT supported
        - For co-located HFT simulation requiring < 1μs precision,
          consider using nanosecond-native calculations

    Thread Safety:
        - All sampling methods are thread-safe
        - Configuration updates are thread-safe

    Example:
        >>> model = LatencyModel.from_profile(LatencyProfile.COLOCATED)
        >>> feed_latency = model.sample_feed_latency()  # nanoseconds
        >>> order_latency = model.sample_order_latency()  # nanoseconds
    """

    def __init__(
        self,
        config: Optional[LatencyModelConfig] = None,
        # Legacy parameters for backward compatibility
        feed_latency_mean_us: float = 100.0,
        feed_latency_std_us: float = 30.0,
        order_latency_mean_us: float = 200.0,
        order_latency_std_us: float = 50.0,
        exchange_latency_us: float = 10.0,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize latency model.

        Args:
            config: Full configuration (overrides legacy params if provided)
            feed_latency_mean_us: Mean feed latency in microseconds (legacy)
            feed_latency_std_us: Std dev of feed latency (legacy)
            order_latency_mean_us: Mean order submission latency (legacy)
            order_latency_std_us: Std dev of order latency (legacy)
            exchange_latency_us: Exchange processing time (legacy)
            seed: Random seed for reproducibility
        """
        self._lock = threading.Lock()

        # Use config if provided, otherwise build from legacy params
        if config is not None:
            self._config = config
        else:
            # Build config from legacy parameters
            self._config = LatencyModelConfig(
                feed_config=LatencyConfig(
                    distribution=LatencyDistribution.LOGNORMAL,
                    mean_us=feed_latency_mean_us,
                    std_us=feed_latency_std_us,
                ),
                order_config=LatencyConfig(
                    distribution=LatencyDistribution.LOGNORMAL,
                    mean_us=order_latency_mean_us,
                    std_us=order_latency_std_us,
                ),
                exchange_config=LatencyConfig(
                    distribution=LatencyDistribution.CONSTANT,
                    mean_us=exchange_latency_us,
                    std_us=0.0,
                ),
                fill_config=LatencyConfig(
                    distribution=LatencyDistribution.LOGNORMAL,
                    mean_us=feed_latency_mean_us,  # Similar to feed
                    std_us=feed_latency_std_us,
                ),
                profile=LatencyProfile.CUSTOM,
                seed=seed,
            )

        # Determine seed
        actual_seed = self._config.seed if config is not None else seed

        # Initialize samplers
        self._feed_sampler = LatencySampler(
            self._config.feed_config or LatencyConfig(),
            seed=actual_seed,
        )
        self._order_sampler = LatencySampler(
            self._config.order_config or LatencyConfig(),
            seed=(actual_seed + 1) if actual_seed is not None else None,
        )
        self._exchange_sampler = LatencySampler(
            self._config.exchange_config or LatencyConfig(mean_us=10.0, std_us=3.0),
            seed=(actual_seed + 2) if actual_seed is not None else None,
        )
        self._fill_sampler = LatencySampler(
            self._config.fill_config or LatencyConfig(),
            seed=(actual_seed + 3) if actual_seed is not None else None,
        )

        # Seasonality
        self._seasonality: Optional[List[float]] = None
        if self._config.seasonality_multipliers is not None:
            self._seasonality = [float(m) for m in self._config.seasonality_multipliers]
            if len(self._seasonality) != 24:
                raise ValueError("seasonality_multipliers must have length 24")

        self._volatility_sensitivity = self._config.volatility_sensitivity
        self._current_volatility_mult = 1.0

    @classmethod
    def from_profile(
        cls,
        profile: LatencyProfile,
        seed: Optional[int] = None,
    ) -> "LatencyModel":
        """Create model from pre-defined profile.

        Args:
            profile: Latency profile (COLOCATED, PROXIMITY, RETAIL, INSTITUTIONAL)
            seed: Random seed for reproducibility

        Returns:
            Configured LatencyModel

        Example:
            >>> model = LatencyModel.from_profile(LatencyProfile.COLOCATED)
        """
        config = LatencyModelConfig.from_profile(profile, seed=seed)
        return cls(config=config)

    @property
    def config(self) -> LatencyModelConfig:
        """Get current configuration."""
        return self._config

    def set_volatility_multiplier(self, mult: float) -> None:
        """Set current volatility multiplier for adaptive latency.

        Higher volatility can increase latency due to:
        - More market data messages
        - Exchange throttling
        - Network congestion

        Args:
            mult: Volatility multiplier (1.0 = normal, >1.0 = elevated)
        """
        with self._lock:
            if mult < 0:
                mult = 1.0
            self._current_volatility_mult = mult

    def _get_seasonality_mult(self, hour: int) -> float:
        """Get seasonality multiplier for given hour (0-23)."""
        if self._seasonality is None:
            return 1.0
        return self._seasonality[hour % 24]

    def _apply_adjustments(self, base_ns: int, hour: Optional[int] = None) -> int:
        """Apply seasonality and volatility adjustments."""
        mult = 1.0

        # Seasonality adjustment
        if hour is not None and self._seasonality is not None:
            mult *= self._get_seasonality_mult(hour)

        # Volatility adjustment
        if self._volatility_sensitivity > 0:
            vol_factor = 1.0 + (self._current_volatility_mult - 1.0) * self._volatility_sensitivity
            mult *= vol_factor

        return int(round(base_ns * mult))

    def sample_feed_latency(self, hour: Optional[int] = None) -> int:
        """Sample feed latency in nanoseconds.

        Feed latency represents the time between an event occurring on the
        exchange and when we receive the market data update.

        Args:
            hour: Current hour (0-23) for seasonality adjustment

        Returns:
            Latency in nanoseconds
        """
        sample = self._feed_sampler.sample()
        return self._apply_adjustments(sample.latency_ns, hour)

    def sample_order_latency(self, hour: Optional[int] = None) -> int:
        """Sample order submission latency in nanoseconds.

        Order latency represents the time for our order to reach the exchange
        after we submit it.

        Args:
            hour: Current hour (0-23) for seasonality adjustment

        Returns:
            Latency in nanoseconds
        """
        sample = self._order_sampler.sample()
        return self._apply_adjustments(sample.latency_ns, hour)

    def sample_exchange_latency(self, hour: Optional[int] = None) -> int:
        """Sample exchange processing time in nanoseconds.

        Exchange latency represents the internal processing time at the
        exchange for matching/validation.

        Args:
            hour: Current hour (0-23) for seasonality adjustment

        Returns:
            Latency in nanoseconds
        """
        sample = self._exchange_sampler.sample()
        return self._apply_adjustments(sample.latency_ns, hour)

    def sample_fill_latency(self, hour: Optional[int] = None) -> int:
        """Sample fill notification latency in nanoseconds.

        Fill latency represents the time between a fill occurring on the
        exchange and when we receive the fill notification.

        Args:
            hour: Current hour (0-23) for seasonality adjustment

        Returns:
            Latency in nanoseconds
        """
        sample = self._fill_sampler.sample()
        return self._apply_adjustments(sample.latency_ns, hour)

    def sample_round_trip(self, hour: Optional[int] = None) -> int:
        """Sample total round-trip latency for order submission in nanoseconds.

        Round-trip = order_latency + exchange_latency + fill_latency

        This represents the total time from submitting an order to receiving
        confirmation that it was executed or placed.

        Args:
            hour: Current hour (0-23) for seasonality adjustment

        Returns:
            Total round-trip latency in nanoseconds
        """
        order_lat = self.sample_order_latency(hour)
        exchange_lat = self.sample_exchange_latency(hour)
        fill_lat = self.sample_fill_latency(hour)
        return order_lat + exchange_lat + fill_lat

    def sample_all(self, hour: Optional[int] = None) -> Dict[str, int]:
        """Sample all latency components.

        Args:
            hour: Current hour (0-23) for seasonality adjustment

        Returns:
            Dictionary with feed_ns, order_ns, exchange_ns, fill_ns, round_trip_ns
        """
        feed = self.sample_feed_latency(hour)
        order = self.sample_order_latency(hour)
        exchange = self.sample_exchange_latency(hour)
        fill = self.sample_fill_latency(hour)

        return {
            "feed_ns": feed,
            "order_ns": order,
            "exchange_ns": exchange,
            "fill_ns": fill,
            "round_trip_ns": order + exchange + fill,
        }

    def stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all latency components."""
        return {
            "feed": self._feed_sampler.stats(),
            "order": self._order_sampler.stats(),
            "exchange": self._exchange_sampler.stats(),
            "fill": self._fill_sampler.stats(),
        }

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self._feed_sampler.reset_stats()
        self._order_sampler.reset_stats()
        self._exchange_sampler.reset_stats()
        self._fill_sampler.reset_stats()

    def set_empirical_data(
        self,
        component: str,
        samples_us: Sequence[float],
    ) -> None:
        """Set empirical samples for a latency component.

        Args:
            component: "feed", "order", "exchange", or "fill"
            samples_us: Historical latency samples in microseconds
        """
        sampler_map = {
            "feed": self._feed_sampler,
            "order": self._order_sampler,
            "exchange": self._exchange_sampler,
            "fill": self._fill_sampler,
        }

        if component not in sampler_map:
            raise ValueError(f"Unknown component: {component}")

        # Update config to empirical
        with self._lock:
            sampler = sampler_map[component]
            old_config = sampler.config
            new_config = LatencyConfig(
                distribution=LatencyDistribution.EMPIRICAL,
                mean_us=old_config.mean_us,
                std_us=old_config.std_us,
                min_us=old_config.min_us,
                max_us=old_config.max_us,
                spike_prob=old_config.spike_prob,
                spike_mult=old_config.spike_mult,
            )
            sampler.update_config(new_config)
            sampler.set_empirical_data(samples_us)


# Factory functions
def create_latency_model(
    profile: Union[str, LatencyProfile] = "institutional",
    seed: Optional[int] = None,
) -> LatencyModel:
    """Create a latency model from profile name.

    Args:
        profile: Profile name ("colocated", "proximity", "retail", "institutional")
                 or LatencyProfile enum
        seed: Random seed for reproducibility

    Returns:
        Configured LatencyModel
    """
    if isinstance(profile, str):
        profile_map = {
            "colocated": LatencyProfile.COLOCATED,
            "colo": LatencyProfile.COLOCATED,
            "proximity": LatencyProfile.PROXIMITY,
            "prox": LatencyProfile.PROXIMITY,
            "retail": LatencyProfile.RETAIL,
            "institutional": LatencyProfile.INSTITUTIONAL,
            "inst": LatencyProfile.INSTITUTIONAL,
            "custom": LatencyProfile.CUSTOM,
        }
        profile = profile_map.get(profile.lower(), LatencyProfile.INSTITUTIONAL)

    return LatencyModel.from_profile(profile, seed=seed)


# Convenience type aliases
LatencyCallback = Callable[[int], int]  # exchange_time_ns -> our_receive_time_ns


__all__ = [
    "LatencyDistribution",
    "LatencyProfile",
    "LatencyConfig",
    "LatencySample",
    "LatencySampler",
    "LatencyModelConfig",
    "LatencyModel",
    "create_latency_model",
    "LatencyCallback",
]
