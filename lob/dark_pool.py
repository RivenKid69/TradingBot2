"""
Dark Pool Simulation for L3 LOB Simulation.

This module provides dark pool execution simulation including:
- Mid-price execution at dark pool venues
- Probabilistic fill modeling based on order characteristics
- Information leakage modeling
- Multiple dark pool venue simulation

Key Features:
- DarkPoolSimulator: Core simulation engine
- DarkPoolVenue: Individual venue characteristics
- DarkPoolFill: Fill result with execution details
- InformationLeakage: Model market impact from dark pool activity

Reference:
    - SEC Rule 606 disclosures on dark pool routing
    - FINRA ATS transparency data
    - Academic literature on dark pool price discovery

Dark Pool Characteristics:
    1. Mid-price execution (no spread cost)
    2. Probabilistic fills based on order size vs venue liquidity
    3. Minimum quantity thresholds
    4. Information leakage to lit markets
    5. Time-of-day liquidity patterns

Note:
    This module is for EQUITY L3 simulation only.
    Crypto uses separate execution paths (no dark pools).
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Callable,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

from lob.data_structures import (
    Fill,
    LimitOrder,
    OrderType,
    Side,
    Trade,
)


# ==============================================================================
# Enums
# ==============================================================================


class DarkPoolVenueType(IntEnum):
    """Types of dark pool venues."""

    MIDPOINT_CROSS = 1  # Pure midpoint crossing (e.g., IEX D-Limit)
    BLOCK_CROSS = 2  # Block-size only (e.g., Liquidnet)
    RETAIL_INTERNALIZATION = 3  # Retail flow internalization
    CONTINUOUS_CROSS = 4  # Continuous matching at midpoint
    AUCTION_CROSS = 5  # Periodic auctions (e.g., opening/closing)


class FillType(IntEnum):
    """Type of dark pool fill."""

    FULL = 1  # Complete fill
    PARTIAL = 2  # Partial fill
    NO_FILL = 3  # No execution
    BLOCKED = 4  # Blocked due to information leakage risk


class LeakageType(IntEnum):
    """Type of information leakage."""

    NONE = 0  # No leakage
    QUOTE_UPDATE = 1  # Lit quotes updated
    TRADE_SIGNAL = 2  # Trade activity detected
    SIZE_INFERENCE = 3  # Order size inferred from patterns
    VENUE_SPECIFIC = 4  # Venue-specific leakage pattern


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass
class DarkPoolConfig:
    """
    Configuration for a dark pool venue.

    Attributes:
        venue_id: Unique venue identifier
        venue_type: Type of dark pool
        min_order_size: Minimum order size for execution
        max_order_size: Maximum order size (0 = unlimited)
        base_fill_probability: Base probability of fill (0-1)
        size_penalty_factor: How much larger orders reduce fill probability
        size_penalty_multiplier: Multiplier for size_ratio in penalty calc (default 10)
            Formula: penalty = 1.0 - min(1.0, size_ratio * size_penalty_factor * size_penalty_multiplier)
        time_of_day_factor: Whether to apply time-of-day adjustments
        info_leakage_probability: Probability of information leakage per attempt
        typical_adv_fraction: Typical volume as fraction of ADV
        latency_ms: Average latency in milliseconds
        partial_fill_min_ratio: Minimum fill ratio for partial fills (default 0.3)
        partial_fill_size_multiplier: Size impact on partial fill ratio (default 5.0)
        partial_fill_max_reduction: Max reduction from size impact (default 0.7)
        impact_size_normalization: Shares for normalizing impact calculation (default 10000)
    """

    venue_id: str
    venue_type: DarkPoolVenueType = DarkPoolVenueType.MIDPOINT_CROSS
    min_order_size: float = 100.0
    max_order_size: float = 0.0  # 0 = unlimited
    base_fill_probability: float = 0.30
    size_penalty_factor: float = 0.5
    size_penalty_multiplier: float = 10.0  # Multiplier in size penalty formula
    time_of_day_factor: bool = True
    info_leakage_probability: float = 0.10
    typical_adv_fraction: float = 0.02  # 2% of ADV
    latency_ms: float = 5.0
    # Partial fill configuration
    partial_fill_min_ratio: float = 0.3  # Min fill ratio for partial fills
    partial_fill_size_multiplier: float = 5.0  # How much size affects fill ratio
    partial_fill_max_reduction: float = 0.7  # Max reduction from size impact
    # Impact calculation
    impact_size_normalization: float = 10000.0  # Shares for normalizing impact
    # History limits (to prevent memory leaks)
    max_history_size: int = 10000  # Maximum entries in fill/leakage history

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.impact_size_normalization <= 0:
            raise ValueError(
                f"impact_size_normalization must be positive, got {self.impact_size_normalization}"
            )
        if self.base_fill_probability < 0 or self.base_fill_probability > 1:
            raise ValueError(
                f"base_fill_probability must be in [0, 1], got {self.base_fill_probability}"
            )
        if self.info_leakage_probability < 0 or self.info_leakage_probability > 1:
            raise ValueError(
                f"info_leakage_probability must be in [0, 1], got {self.info_leakage_probability}"
            )
        if self.min_order_size < 0:
            raise ValueError(f"min_order_size must be non-negative, got {self.min_order_size}")
        if self.max_history_size <= 0:
            raise ValueError(f"max_history_size must be positive, got {self.max_history_size}")


@dataclass
class DarkPoolFill:
    """
    Result of a dark pool execution attempt.

    Attributes:
        order_id: Original order ID
        venue_id: Venue where execution occurred
        fill_type: Type of fill result
        filled_qty: Quantity filled
        fill_price: Execution price (typically midpoint)
        timestamp_ns: Execution timestamp
        latency_ns: Latency from order to fill
        info_leakage: Information leakage that occurred
        remaining_qty: Unfilled quantity
        lit_mid_at_fill: Lit market midpoint at fill time
    """

    order_id: str
    venue_id: str
    fill_type: FillType
    filled_qty: float
    fill_price: float
    timestamp_ns: int
    latency_ns: int = 0
    info_leakage: Optional["InformationLeakage"] = None
    remaining_qty: float = 0.0
    lit_mid_at_fill: float = 0.0

    @property
    def is_filled(self) -> bool:
        """Check if any fill occurred."""
        return self.fill_type in (FillType.FULL, FillType.PARTIAL)

    @property
    def notional(self) -> float:
        """Notional value of fill."""
        return self.filled_qty * self.fill_price

    @property
    def savings_vs_spread(self) -> float:
        """Savings from mid-price execution vs crossing spread."""
        # Approximate: half spread saved
        if self.lit_mid_at_fill > 0 and self.filled_qty > 0:
            return self.filled_qty * self.lit_mid_at_fill * 0.0005  # ~5 bps half spread
        return 0.0


@dataclass
class InformationLeakage:
    """
    Information leakage from dark pool activity.

    Attributes:
        leakage_type: Type of leakage
        timestamp_ns: When leakage occurred
        magnitude: Severity of leakage (0-1)
        detected_side: Inferred order side (if detected)
        detected_size_estimate: Inferred size (if detected)
        market_impact_bps: Estimated market impact in basis points
        description: Human-readable description
    """

    leakage_type: LeakageType
    timestamp_ns: int
    magnitude: float = 0.0
    detected_side: Optional[Side] = None
    detected_size_estimate: float = 0.0
    market_impact_bps: float = 0.0
    description: str = ""


@dataclass
class DarkPoolState:
    """
    Current state of dark pool simulation.

    Attributes:
        total_volume: Total volume executed in dark pools
        total_attempts: Total execution attempts
        total_fills: Number of successful fills
        total_leakage_events: Number of information leakage events
        venue_volumes: Volume by venue
    """

    total_volume: float = 0.0
    total_attempts: int = 0
    total_fills: int = 0
    total_leakage_events: int = 0
    venue_volumes: Dict[str, float] = field(default_factory=dict)

    @property
    def fill_rate(self) -> float:
        """Overall fill rate."""
        return self.total_fills / self.total_attempts if self.total_attempts > 0 else 0.0


# ==============================================================================
# Dark Pool Venue
# ==============================================================================


class DarkPoolVenue:
    """
    Individual dark pool venue simulation.

    Models venue-specific characteristics including:
    - Fill probability based on order characteristics
    - Minimum quantity thresholds
    - Time-of-day liquidity patterns
    - Information leakage modeling
    """

    def __init__(
        self,
        config: DarkPoolConfig,
        rng: Optional[random.Random] = None,
    ) -> None:
        """
        Initialize dark pool venue.

        Args:
            config: Venue configuration
            rng: Random number generator (for reproducibility)
        """
        self._config = config
        self._rng = rng or random.Random()

        # State
        self._total_volume = 0.0
        self._total_attempts = 0
        self._total_fills = 0
        self._daily_volume = 0.0

    @property
    def venue_id(self) -> str:
        """Venue identifier."""
        return self._config.venue_id

    @property
    def venue_type(self) -> DarkPoolVenueType:
        """Venue type."""
        return self._config.venue_type

    def attempt_fill(
        self,
        order: LimitOrder,
        lit_mid_price: float,
        lit_spread: float = 0.0,
        adv: float = 0.0,
        volatility: float = 0.02,
        hour_of_day: int = 12,
    ) -> DarkPoolFill:
        """
        Attempt to fill order in this dark pool venue.

        Args:
            order: The order to execute
            lit_mid_price: Current midpoint in lit market
            lit_spread: Current spread in lit market
            adv: Average daily volume (for size scaling)
            volatility: Current volatility
            hour_of_day: Hour of day (0-23) for time patterns

        Returns:
            DarkPoolFill with execution result
        """
        timestamp_ns = time.time_ns()
        self._total_attempts += 1

        # Check minimum size
        if order.remaining_qty < self._config.min_order_size:
            return DarkPoolFill(
                order_id=order.order_id,
                venue_id=self.venue_id,
                fill_type=FillType.NO_FILL,
                filled_qty=0.0,
                fill_price=0.0,
                timestamp_ns=timestamp_ns,
                remaining_qty=order.remaining_qty,
                lit_mid_at_fill=lit_mid_price,
            )

        # Check maximum size
        if self._config.max_order_size > 0 and order.remaining_qty > self._config.max_order_size:
            # For block venues, reject if too large
            if self._config.venue_type == DarkPoolVenueType.BLOCK_CROSS:
                return DarkPoolFill(
                    order_id=order.order_id,
                    venue_id=self.venue_id,
                    fill_type=FillType.NO_FILL,
                    filled_qty=0.0,
                    fill_price=0.0,
                    timestamp_ns=timestamp_ns,
                    remaining_qty=order.remaining_qty,
                    lit_mid_at_fill=lit_mid_price,
                )

        # Calculate fill probability
        fill_prob = self._calculate_fill_probability(
            order_qty=order.remaining_qty,
            adv=adv,
            volatility=volatility,
            hour_of_day=hour_of_day,
        )

        # Determine if fill occurs
        if self._rng.random() > fill_prob:
            # No fill, but check for information leakage
            leakage = self._check_information_leakage(order, lit_mid_price)
            return DarkPoolFill(
                order_id=order.order_id,
                venue_id=self.venue_id,
                fill_type=FillType.NO_FILL,
                filled_qty=0.0,
                fill_price=0.0,
                timestamp_ns=timestamp_ns,
                info_leakage=leakage,
                remaining_qty=order.remaining_qty,
                lit_mid_at_fill=lit_mid_price,
            )

        # Determine fill quantity (may be partial)
        filled_qty = self._determine_fill_quantity(order.remaining_qty, adv)

        # Determine fill price (typically midpoint)
        fill_price = self._determine_fill_price(
            order=order,
            lit_mid_price=lit_mid_price,
            lit_spread=lit_spread,
        )

        # Calculate latency with exponential noise
        # Base latency in nanoseconds
        base_latency_ns = int(self._config.latency_ms * 1_000_000)
        if base_latency_ns > 0:
            # Add exponential noise (~50% of base latency on average)
            noise_ns = int(self._rng.expovariate(1.0 / base_latency_ns) * 0.5)
            latency_ns = base_latency_ns + noise_ns
        else:
            # Zero latency configured - no noise added
            latency_ns = 0

        # Check for information leakage
        leakage = self._check_information_leakage(order, lit_mid_price)

        # Update state
        self._total_fills += 1
        self._total_volume += filled_qty
        self._daily_volume += filled_qty

        fill_type = FillType.FULL if filled_qty >= order.remaining_qty else FillType.PARTIAL

        return DarkPoolFill(
            order_id=order.order_id,
            venue_id=self.venue_id,
            fill_type=fill_type,
            filled_qty=filled_qty,
            fill_price=fill_price,
            timestamp_ns=timestamp_ns,
            latency_ns=latency_ns,
            info_leakage=leakage,
            remaining_qty=order.remaining_qty - filled_qty,
            lit_mid_at_fill=lit_mid_price,
        )

    def _calculate_fill_probability(
        self,
        order_qty: float,
        adv: float,
        volatility: float,
        hour_of_day: int,
    ) -> float:
        """Calculate probability of fill based on order characteristics."""
        base_prob = self._config.base_fill_probability

        # Size penalty: larger orders have lower fill probability
        # Formula: penalty = 1.0 - min(1.0, size_ratio * factor * multiplier)
        if adv > 0:
            size_ratio = order_qty / adv
            size_penalty = 1.0 - min(
                1.0,
                size_ratio
                * self._config.size_penalty_factor
                * self._config.size_penalty_multiplier,
            )
        else:
            size_penalty = 0.8  # Default penalty if no ADV

        # Time of day adjustment
        if self._config.time_of_day_factor:
            # Higher probability during open/close, lower midday
            if hour_of_day in (9, 10, 15, 16):  # Opening and closing hours
                tod_factor = 1.2
            elif hour_of_day in (12, 13):  # Lunch hours
                tod_factor = 0.7
            else:
                tod_factor = 1.0
        else:
            tod_factor = 1.0

        # Volatility adjustment: higher volatility = more crossing opportunities
        vol_factor = 1.0 + min(0.3, volatility * 5)

        # Combine factors
        final_prob = base_prob * size_penalty * tod_factor * vol_factor

        return min(0.95, max(0.01, final_prob))

    def _determine_fill_quantity(
        self,
        order_qty: float,
        adv: float,
    ) -> float:
        """Determine how much of the order gets filled."""
        # For smaller orders, likely full fill
        if adv > 0:
            size_ratio = order_qty / adv
            if size_ratio < 0.001:  # Very small order (<0.1% ADV)
                return order_qty

            # Larger orders may get partial fills
            if size_ratio > 0.01:  # >1% ADV
                # Partial fill with decreasing probability based on size
                # Uses configurable parameters for min ratio, size multiplier, and max reduction
                min_ratio = self._config.partial_fill_min_ratio
                size_mult = self._config.partial_fill_size_multiplier
                max_reduction = self._config.partial_fill_max_reduction
                fill_ratio = self._rng.uniform(min_ratio, 1.0) * (
                    1.0 - min(max_reduction, size_ratio * size_mult)
                )
                return max(self._config.min_order_size, order_qty * fill_ratio)

        return order_qty

    def _determine_fill_price(
        self,
        order: LimitOrder,
        lit_mid_price: float,
        lit_spread: float,
    ) -> float:
        """Determine execution price."""
        # Most dark pools execute at midpoint
        if self._config.venue_type in (
            DarkPoolVenueType.MIDPOINT_CROSS,
            DarkPoolVenueType.CONTINUOUS_CROSS,
            DarkPoolVenueType.AUCTION_CROSS,
        ):
            return lit_mid_price

        # Block crosses may have slight price improvement
        if self._config.venue_type == DarkPoolVenueType.BLOCK_CROSS:
            # Small improvement from midpoint (1-2 bps)
            improvement = lit_mid_price * 0.0001 * self._rng.uniform(0, 2)
            if order.side == Side.BUY:
                return lit_mid_price - improvement
            else:
                return lit_mid_price + improvement

        # Retail internalization: typically midpoint or better
        if self._config.venue_type == DarkPoolVenueType.RETAIL_INTERNALIZATION:
            # Price improvement of 0.1-0.5 cents typically
            improvement = 0.001 + self._rng.random() * 0.004
            if order.side == Side.BUY:
                return lit_mid_price - improvement
            else:
                return lit_mid_price + improvement

        return lit_mid_price

    def _check_information_leakage(
        self,
        order: LimitOrder,
        lit_mid_price: float,
    ) -> Optional[InformationLeakage]:
        """Check if information leakage occurs."""
        if self._rng.random() > self._config.info_leakage_probability:
            return None

        timestamp_ns = time.time_ns()

        # Determine leakage type based on venue and order
        leakage_types = [
            (LeakageType.QUOTE_UPDATE, 0.4),
            (LeakageType.TRADE_SIGNAL, 0.3),
            (LeakageType.SIZE_INFERENCE, 0.2),
            (LeakageType.VENUE_SPECIFIC, 0.1),
        ]

        r = self._rng.random()
        cumulative = 0.0
        selected_type = LeakageType.QUOTE_UPDATE

        for leak_type, prob in leakage_types:
            cumulative += prob
            if r < cumulative:
                selected_type = leak_type
                break

        # Calculate magnitude (severity)
        magnitude = self._rng.uniform(0.1, 0.5)

        # Estimate market impact using configurable normalization
        # size_factor normalizes order size relative to typical institutional size
        size_factor = order.remaining_qty / self._config.impact_size_normalization
        impact_bps = magnitude * size_factor * 2.0  # ~0.2-1.0 bps typical

        # Create leakage event
        return InformationLeakage(
            leakage_type=selected_type,
            timestamp_ns=timestamp_ns,
            magnitude=magnitude,
            detected_side=order.side if magnitude > 0.3 else None,
            detected_size_estimate=(
                order.remaining_qty * self._rng.uniform(0.5, 1.5) if magnitude > 0.4 else 0.0
            ),
            market_impact_bps=impact_bps,
            description=self._describe_leakage(selected_type, magnitude),
        )

    def _describe_leakage(self, leakage_type: LeakageType, magnitude: float) -> str:
        """Generate human-readable leakage description."""
        severity = "minor" if magnitude < 0.3 else "moderate" if magnitude < 0.6 else "significant"

        descriptions = {
            LeakageType.QUOTE_UPDATE: f"Lit market quotes updated ({severity} impact)",
            LeakageType.TRADE_SIGNAL: f"Trade activity detected in related instruments ({severity})",
            LeakageType.SIZE_INFERENCE: f"Order size partially inferred from venue patterns ({severity})",
            LeakageType.VENUE_SPECIFIC: f"Venue-specific information leakage ({severity})",
        }

        return descriptions.get(leakage_type, f"Unknown leakage type ({severity})")

    def reset_daily_stats(self) -> None:
        """Reset daily statistics."""
        self._daily_volume = 0.0

    def stats(self) -> Dict[str, float]:
        """Get venue statistics."""
        return {
            "venue_id": self.venue_id,
            "total_volume": self._total_volume,
            "total_attempts": float(self._total_attempts),
            "total_fills": float(self._total_fills),
            "fill_rate": (
                self._total_fills / self._total_attempts if self._total_attempts > 0 else 0.0
            ),
            "daily_volume": self._daily_volume,
        }


# ==============================================================================
# Dark Pool Simulator
# ==============================================================================


class DarkPoolSimulator:
    """
    Simulates dark pool execution across multiple venues.

    Features:
    - Multi-venue routing simulation
    - Mid-price execution
    - Probabilistic fill modeling
    - Information leakage tracking
    - Smart order routing (optional)

    Usage:
        # Create simulator with default venues
        simulator = DarkPoolSimulator()

        # Or with custom venues
        venues = [
            DarkPoolConfig(venue_id="IEX", venue_type=DarkPoolVenueType.MIDPOINT_CROSS),
            DarkPoolConfig(venue_id="SIGMA", venue_type=DarkPoolVenueType.BLOCK_CROSS),
        ]
        simulator = DarkPoolSimulator(venues=[DarkPoolVenue(c) for c in venues])

        # Attempt dark pool fill
        fill = simulator.attempt_dark_fill(
            order=my_order,
            lit_mid_price=150.0,
            lit_spread=0.02,
            adv=10_000_000,
        )

        if fill.is_filled:
            print(f"Filled {fill.filled_qty} @ {fill.fill_price}")
    """

    # Default venue configurations (based on real US dark pools)
    DEFAULT_VENUES = [
        DarkPoolConfig(
            venue_id="SIGMA_X",
            venue_type=DarkPoolVenueType.MIDPOINT_CROSS,
            min_order_size=100,
            base_fill_probability=0.35,
            info_leakage_probability=0.08,
        ),
        DarkPoolConfig(
            venue_id="IEX_D",
            venue_type=DarkPoolVenueType.MIDPOINT_CROSS,
            min_order_size=100,
            base_fill_probability=0.25,
            info_leakage_probability=0.05,  # IEX has lower leakage
        ),
        DarkPoolConfig(
            venue_id="LIQUIDNET",
            venue_type=DarkPoolVenueType.BLOCK_CROSS,
            min_order_size=10000,
            max_order_size=0,  # No max for blocks
            base_fill_probability=0.15,
            info_leakage_probability=0.03,
        ),
        DarkPoolConfig(
            venue_id="RETAIL_INT",
            venue_type=DarkPoolVenueType.RETAIL_INTERNALIZATION,
            min_order_size=1,
            max_order_size=5000,
            base_fill_probability=0.60,
            info_leakage_probability=0.15,
        ),
    ]

    def __init__(
        self,
        venues: Optional[List[DarkPoolVenue]] = None,
        enable_smart_routing: bool = True,
        max_leakage_tolerance: float = 0.5,
        seed: Optional[int] = None,
        on_fill: Optional[Callable[[DarkPoolFill], None]] = None,
        on_leakage: Optional[Callable[[InformationLeakage], None]] = None,
        max_history_size: int = 10000,
    ) -> None:
        """
        Initialize dark pool simulator.

        Args:
            venues: List of dark pool venues (uses defaults if not provided)
            enable_smart_routing: Enable smart order routing between venues
            max_leakage_tolerance: Maximum acceptable leakage magnitude (0-1)
            seed: Random seed for reproducibility
            on_fill: Callback when fill occurs
            on_leakage: Callback when leakage detected
            max_history_size: Maximum entries in fill/leakage history (prevents memory leaks)
        """
        self._rng = random.Random(seed)
        self._max_history_size = max_history_size

        # Initialize venues
        if venues:
            self._venues = {v.venue_id: v for v in venues}
        else:
            self._venues = {
                cfg.venue_id: DarkPoolVenue(cfg, self._rng) for cfg in self.DEFAULT_VENUES
            }

        self._enable_smart_routing = enable_smart_routing
        self._max_leakage_tolerance = max_leakage_tolerance

        # Callbacks
        self._on_fill = on_fill
        self._on_leakage = on_leakage

        # State with bounded history (prevents memory leaks in long simulations)
        self._state = DarkPoolState()
        self._leakage_history: Deque[InformationLeakage] = deque(maxlen=max_history_size)
        self._fill_history: Deque[DarkPoolFill] = deque(maxlen=max_history_size)

    def attempt_dark_fill(
        self,
        order: LimitOrder,
        lit_mid_price: float,
        lit_spread: float = 0.0,
        adv: float = 0.0,
        volatility: float = 0.02,
        hour_of_day: int = 12,
        preferred_venue: Optional[str] = None,
    ) -> Optional[DarkPoolFill]:
        """
        Attempt to fill order in dark pools.

        If smart routing is enabled, tries multiple venues in optimal order.
        Otherwise, tries venues in a fixed order.

        Args:
            order: Order to execute
            lit_mid_price: Current lit market midpoint
            lit_spread: Current lit market spread
            adv: Average daily volume
            volatility: Current volatility
            hour_of_day: Hour of day (0-23)
            preferred_venue: Optional preferred venue ID

        Returns:
            DarkPoolFill if any fill occurred, None otherwise
        """
        if not self._venues:
            return None

        self._state.total_attempts += 1

        # Check if we should block due to recent leakage
        if self._should_block_for_leakage():
            return DarkPoolFill(
                order_id=order.order_id,
                venue_id="BLOCKED",
                fill_type=FillType.BLOCKED,
                filled_qty=0.0,
                fill_price=0.0,
                timestamp_ns=time.time_ns(),
                remaining_qty=order.remaining_qty,
                lit_mid_at_fill=lit_mid_price,
            )

        # Determine venue order
        if preferred_venue and preferred_venue in self._venues:
            venue_order = [preferred_venue] + [v for v in self._venues if v != preferred_venue]
        elif self._enable_smart_routing:
            venue_order = self._smart_route_order(order, adv)
        else:
            venue_order = list(self._venues.keys())

        # Try each venue
        for venue_id in venue_order:
            venue = self._venues[venue_id]

            fill = venue.attempt_fill(
                order=order,
                lit_mid_price=lit_mid_price,
                lit_spread=lit_spread,
                adv=adv,
                volatility=volatility,
                hour_of_day=hour_of_day,
            )

            # Track leakage
            if fill.info_leakage:
                self._leakage_history.append(fill.info_leakage)
                self._state.total_leakage_events += 1
                if self._on_leakage:
                    self._on_leakage(fill.info_leakage)

            # If fill occurred, record and return
            if fill.is_filled:
                self._state.total_fills += 1
                self._state.total_volume += fill.filled_qty
                self._state.venue_volumes[venue_id] = (
                    self._state.venue_volumes.get(venue_id, 0.0) + fill.filled_qty
                )
                self._fill_history.append(fill)

                if self._on_fill:
                    self._on_fill(fill)

                return fill

        # No fill at any venue
        return None

    def attempt_fill_with_routing(
        self,
        order: LimitOrder,
        lit_mid_price: float,
        lit_spread: float = 0.0,
        adv: float = 0.0,
        volatility: float = 0.02,
        hour_of_day: int = 12,
        max_attempts: int = 3,
    ) -> List[DarkPoolFill]:
        """
        Attempt fill with smart routing across multiple venues.

        May return multiple fills from different venues.

        Args:
            order: Order to execute
            lit_mid_price: Current lit market midpoint
            lit_spread: Current lit market spread
            adv: Average daily volume
            volatility: Current volatility
            hour_of_day: Hour of day (0-23)
            max_attempts: Maximum number of venues to try

        Returns:
            List of fills (may be empty, partial, or multiple)
        """
        fills: List[DarkPoolFill] = []
        remaining_qty = order.remaining_qty

        venue_order = self._smart_route_order(order, adv)

        for i, venue_id in enumerate(venue_order):
            if i >= max_attempts or remaining_qty <= 0:
                break

            venue = self._venues[venue_id]

            # Create order with remaining quantity
            temp_order = LimitOrder(
                order_id=f"{order.order_id}_part_{i}",
                price=order.price,
                qty=remaining_qty,
                remaining_qty=remaining_qty,
                timestamp_ns=order.timestamp_ns,
                side=order.side,
                order_type=order.order_type,
            )

            fill = venue.attempt_fill(
                order=temp_order,
                lit_mid_price=lit_mid_price,
                lit_spread=lit_spread,
                adv=adv,
                volatility=volatility,
                hour_of_day=hour_of_day,
            )

            if fill.is_filled:
                fills.append(fill)
                remaining_qty -= fill.filled_qty

                # Update state
                self._state.total_fills += 1
                self._state.total_volume += fill.filled_qty

            # Track leakage
            if fill.info_leakage:
                self._leakage_history.append(fill.info_leakage)
                self._state.total_leakage_events += 1

        return fills

    def _smart_route_order(
        self,
        order: LimitOrder,
        adv: float,
    ) -> List[str]:
        """
        Determine optimal venue routing order.

        Considers:
        - Order size vs venue minimums/maximums
        - Historical fill rates
        - Information leakage history
        """
        scores: List[Tuple[str, float]] = []

        for venue_id, venue in self._venues.items():
            config = venue._config

            # Base score from fill probability
            score = config.base_fill_probability

            # Bonus for appropriate size match
            if order.remaining_qty >= config.min_order_size:
                if config.max_order_size == 0 or order.remaining_qty <= config.max_order_size:
                    score *= 1.5
                else:
                    score *= 0.5
            else:
                score *= 0.1  # Order too small

            # Penalty for leakage probability
            score *= 1.0 - config.info_leakage_probability

            # Bonus for block venues on large orders
            if config.venue_type == DarkPoolVenueType.BLOCK_CROSS and adv > 0:
                if order.remaining_qty / adv > 0.001:  # >0.1% of ADV
                    score *= 1.3

            scores.append((venue_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return [venue_id for venue_id, _ in scores]

    def _should_block_for_leakage(self) -> bool:
        """Check if we should block trading due to recent leakage."""
        if not self._leakage_history:
            return False

        # Check recent leakage severity
        recent_window_ns = 60_000_000_000  # 60 seconds
        current_time = time.time_ns()
        cutoff = current_time - recent_window_ns

        # Convert deque to list for slicing (deque doesn't support slice indexing)
        leakage_list = list(self._leakage_history)
        recent_leakage = [l for l in leakage_list[-10:] if l.timestamp_ns > cutoff]

        if not recent_leakage:
            return False

        # Calculate average magnitude
        avg_magnitude = sum(l.magnitude for l in recent_leakage) / len(recent_leakage)

        return avg_magnitude > self._max_leakage_tolerance

    def get_venue(self, venue_id: str) -> Optional[DarkPoolVenue]:
        """Get venue by ID."""
        return self._venues.get(venue_id)

    def get_all_venues(self) -> Iterator[DarkPoolVenue]:
        """Iterate over all venues."""
        return iter(self._venues.values())

    def get_state(self) -> DarkPoolState:
        """Get current simulation state."""
        return self._state

    def get_leakage_history(
        self,
        limit: int = 100,
    ) -> List[InformationLeakage]:
        """Get recent leakage history."""
        # Convert deque slice to list
        history_list = list(self._leakage_history)
        return history_list[-limit:] if limit < len(history_list) else history_list

    def get_fill_history(
        self,
        limit: int = 100,
    ) -> List[DarkPoolFill]:
        """Get recent fill history."""
        # Convert deque slice to list
        history_list = list(self._fill_history)
        return history_list[-limit:] if limit < len(history_list) else history_list

    def estimate_fill_probability(
        self,
        order: LimitOrder,
        adv: float = 0.0,
        volatility: float = 0.02,
        hour_of_day: int = 12,
    ) -> Dict[str, float]:
        """
        Estimate fill probability at each venue.

        Args:
            order: Order to evaluate
            adv: Average daily volume
            volatility: Current volatility
            hour_of_day: Hour of day

        Returns:
            Dict mapping venue_id to estimated fill probability
        """
        probs: Dict[str, float] = {}

        for venue_id, venue in self._venues.items():
            prob = venue._calculate_fill_probability(
                order_qty=order.remaining_qty,
                adv=adv,
                volatility=volatility,
                hour_of_day=hour_of_day,
            )
            probs[venue_id] = prob

        return probs

    def reset_daily_stats(self) -> None:
        """Reset daily statistics for all venues."""
        for venue in self._venues.values():
            venue.reset_daily_stats()

    def clear_history(self) -> None:
        """Clear execution and leakage history."""
        self._leakage_history.clear()
        self._fill_history.clear()

    def stats(self) -> Dict[str, float]:
        """Get simulator statistics."""
        return {
            "total_volume": self._state.total_volume,
            "total_attempts": float(self._state.total_attempts),
            "total_fills": float(self._state.total_fills),
            "fill_rate": self._state.fill_rate,
            "total_leakage_events": float(self._state.total_leakage_events),
            "num_venues": float(len(self._venues)),
        }


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_dark_pool_simulator(
    venue_configs: Optional[List[DarkPoolConfig]] = None,
    enable_smart_routing: bool = True,
    seed: Optional[int] = None,
    on_fill: Optional[Callable[[DarkPoolFill], None]] = None,
    on_leakage: Optional[Callable[[InformationLeakage], None]] = None,
    max_history_size: int = 10000,
) -> DarkPoolSimulator:
    """
    Create a DarkPoolSimulator with configuration.

    Args:
        venue_configs: Optional list of venue configurations
        enable_smart_routing: Enable smart order routing
        seed: Random seed for reproducibility
        on_fill: Fill callback
        on_leakage: Leakage callback
        max_history_size: Maximum entries in fill/leakage history (prevents memory leaks)

    Returns:
        Configured DarkPoolSimulator

    Note:
        When venue_configs is provided, all venues share the same RNG for
        consistent reproducibility. When None, default venues are created
        by the simulator using its internal RNG.
    """
    venues = None
    if venue_configs:
        # Create shared RNG for all custom venues
        rng = random.Random(seed)
        venues = [DarkPoolVenue(cfg, rng) for cfg in venue_configs]
        # Pass seed=None to simulator since venues already have their RNG
        # This avoids creating duplicate RNG state
        return DarkPoolSimulator(
            venues=venues,
            enable_smart_routing=enable_smart_routing,
            seed=None,  # Venues already seeded
            on_fill=on_fill,
            on_leakage=on_leakage,
            max_history_size=max_history_size,
        )

    # No custom venues - let simulator create defaults with the seed
    return DarkPoolSimulator(
        venues=None,
        enable_smart_routing=enable_smart_routing,
        seed=seed,
        on_fill=on_fill,
        on_leakage=on_leakage,
        max_history_size=max_history_size,
    )


def create_default_dark_pool_simulator(
    seed: Optional[int] = None,
    max_history_size: int = 10000,
) -> DarkPoolSimulator:
    """
    Create a DarkPoolSimulator with default US equity venues.

    Args:
        seed: Random seed for reproducibility
        max_history_size: Maximum entries in fill/leakage history (prevents memory leaks)

    Returns:
        DarkPoolSimulator with default venues
    """
    return DarkPoolSimulator(
        venues=None,  # Uses DEFAULT_VENUES
        enable_smart_routing=True,
        seed=seed,
        max_history_size=max_history_size,
    )
