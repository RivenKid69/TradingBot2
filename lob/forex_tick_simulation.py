# -*- coding: utf-8 -*-
"""
lob/forex_tick_simulation.py
Tick-Level Execution Simulation for Forex OTC Markets

Phase 11: Forex Realism Enhancement (2025-11-30)

This module provides high-fidelity tick-by-tick execution simulation for forex,
capturing the microstructure of OTC dealer markets at the tick level.

Key Features:
1. Tick-by-tick price simulation with realistic properties
2. Spread dynamics simulation (time-varying spreads)
3. Execution quality modeling at tick resolution
4. Latency-aware order matching
5. Price impact at tick level
6. Adverse selection modeling

Unlike LOB simulation for exchanges, forex tick simulation models:
- Dealer quote behavior (not order book depth)
- Spread widening/tightening dynamics
- Quote flickering at high frequency
- Last-look execution delays
- Price gaps around news events

Tick Properties (empirical from BIS 2022):
- Major pairs: ~2-5 ticks/second average, up to 100+ during news
- Spread: 0.5-2 pips for majors, varies by session
- Tick size: 0.1 pip (5th decimal for most, 3rd for JPY)

References:
- Goodhart & Figliuoli (1991): "Every minute counts in financial markets"
- Dacorogna et al. (2001): "An Introduction to High-Frequency Finance"
- Chaboud et al. (2014): "Rise of the Machines: Algorithmic Trading in FX"
- BIS (2022): Triennial Central Bank Survey

Author: Sultan Suyunbayev
Date: 2025-11-30
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Tick arrival rates (ticks per second) by session
# Source: BIS 2022, academic studies
TICK_ARRIVAL_RATES: Dict[str, float] = {
    "sydney": 1.5,
    "tokyo": 2.5,
    "london": 4.0,
    "new_york": 3.5,
    "london_ny_overlap": 5.0,
    "tokyo_london_overlap": 3.0,
    "off_hours": 0.5,
    "news_event": 15.0,  # During major news
}

# Spread dynamics parameters (in pips)
SPREAD_PARAMS: Dict[str, Dict[str, float]] = {
    "major": {
        "base": 1.0,
        "volatility_sensitivity": 0.5,
        "inventory_sensitivity": 0.3,
        "news_multiplier": 3.0,
    },
    "minor": {
        "base": 2.0,
        "volatility_sensitivity": 0.7,
        "inventory_sensitivity": 0.4,
        "news_multiplier": 4.0,
    },
    "cross": {
        "base": 3.0,
        "volatility_sensitivity": 0.8,
        "inventory_sensitivity": 0.5,
        "news_multiplier": 5.0,
    },
    "exotic": {
        "base": 25.0,
        "volatility_sensitivity": 1.0,
        "inventory_sensitivity": 0.8,
        "news_multiplier": 8.0,
    },
}

# Price jump parameters
JUMP_PARAMS = {
    "intensity_per_hour": 0.5,  # Average jumps per hour
    "jump_size_std_pips": 5.0,  # Standard deviation of jump size
    "recovery_half_life_sec": 30.0,  # Time for spread to recover
}


# =============================================================================
# Enums
# =============================================================================


class TickType(str, Enum):
    """Type of tick."""

    QUOTE = "quote"  # Dealer quote update
    TRADE = "trade"  # Executed trade
    SPREAD_CHANGE = "spread_change"  # Spread adjustment
    GAP = "gap"  # Price gap (news, weekend)


class ExecutionQuality(str, Enum):
    """Execution quality indicator."""

    EXCELLENT = "excellent"  # Better than expected
    GOOD = "good"  # At or near quote
    FAIR = "fair"  # Minor slippage
    POOR = "poor"  # Significant slippage
    REJECTED = "rejected"  # Order rejected


class MarketCondition(str, Enum):
    """Current market condition."""

    NORMAL = "normal"
    VOLATILE = "volatile"
    NEWS_EVENT = "news_event"
    LOW_LIQUIDITY = "low_liquidity"
    GAP = "gap"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Tick:
    """
    Single market tick.

    Represents one price update from a dealer or execution.

    Attributes:
        timestamp_ns: Timestamp in nanoseconds
        bid: Bid price
        ask: Ask price
        mid: Mid price (computed)
        spread_pips: Spread in pips
        tick_type: Type of tick
        volume: Optional trade volume
        dealer_id: Source dealer (if applicable)
        sequence_num: Tick sequence number
    """

    timestamp_ns: int
    bid: float
    ask: float
    mid: float = field(init=False)
    spread_pips: float = field(init=False)
    tick_type: TickType = TickType.QUOTE
    volume: float = 0.0
    dealer_id: Optional[str] = None
    sequence_num: int = 0
    is_jpy_pair: bool = False

    def __post_init__(self) -> None:
        """Compute derived fields."""
        self.mid = (self.bid + self.ask) / 2.0
        pip_value = 0.01 if self.is_jpy_pair else 0.0001
        self.spread_pips = (self.ask - self.bid) / pip_value

    @property
    def timestamp_ms(self) -> float:
        """Get timestamp in milliseconds."""
        return self.timestamp_ns / 1_000_000.0

    @property
    def timestamp_sec(self) -> float:
        """Get timestamp in seconds."""
        return self.timestamp_ns / 1_000_000_000.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp_ns": self.timestamp_ns,
            "bid": self.bid,
            "ask": self.ask,
            "mid": self.mid,
            "spread_pips": self.spread_pips,
            "tick_type": self.tick_type.value,
            "volume": self.volume,
            "dealer_id": self.dealer_id,
            "sequence_num": self.sequence_num,
        }


@dataclass
class TickExecutionResult:
    """
    Result of tick-level execution attempt.

    Attributes:
        filled: Whether order was filled
        fill_price: Execution price
        fill_timestamp_ns: Execution timestamp
        slippage_pips: Slippage from requested price
        execution_quality: Quality indicator
        latency_ns: Order-to-fill latency
        ticks_to_fill: Number of ticks until fill
        market_moved_pips: Market movement during execution
        rejection_reason: Reason if rejected
    """

    filled: bool
    fill_price: float = 0.0
    fill_timestamp_ns: int = 0
    slippage_pips: float = 0.0
    execution_quality: ExecutionQuality = ExecutionQuality.GOOD
    latency_ns: int = 0
    ticks_to_fill: int = 0
    market_moved_pips: float = 0.0
    rejection_reason: Optional[str] = None

    @property
    def latency_ms(self) -> float:
        """Get latency in milliseconds."""
        return self.latency_ns / 1_000_000.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "filled": self.filled,
            "fill_price": self.fill_price,
            "fill_timestamp_ns": self.fill_timestamp_ns,
            "slippage_pips": self.slippage_pips,
            "execution_quality": self.execution_quality.value,
            "latency_ms": self.latency_ms,
            "ticks_to_fill": self.ticks_to_fill,
            "market_moved_pips": self.market_moved_pips,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class SpreadState:
    """
    Current spread state with dynamics.

    Tracks spread evolution and factors affecting it.
    """

    base_spread_pips: float
    current_spread_pips: float
    volatility_component: float = 0.0
    inventory_component: float = 0.0
    news_component: float = 0.0
    last_update_ns: int = 0
    is_recovering: bool = False
    recovery_target: float = 0.0

    @property
    def total_spread_pips(self) -> float:
        """Get total spread including all components."""
        return max(
            0.1,  # Minimum spread
            self.base_spread_pips
            + self.volatility_component
            + self.inventory_component
            + self.news_component,
        )


@dataclass
class TickSimulationConfig:
    """
    Configuration for tick simulation.

    Attributes:
        pair_type: Currency pair classification
        session: Current trading session
        base_tick_rate: Base ticks per second
        volatility_scale: Volatility multiplier
        spread_widening_enabled: Enable dynamic spreads
        price_impact_enabled: Enable price impact
        jump_enabled: Enable price jumps
        latency_model: Latency distribution type
        seed: Random seed for reproducibility
    """

    pair_type: str = "major"
    session: str = "london"
    base_tick_rate: float = 3.0
    volatility_scale: float = 1.0
    spread_widening_enabled: bool = True
    price_impact_enabled: bool = True
    jump_enabled: bool = True
    latency_model: str = "lognormal"
    mean_latency_ms: float = 50.0
    latency_std_ms: float = 20.0
    max_latency_ms: float = 500.0
    adverse_selection_threshold_pips: float = 0.3
    seed: Optional[int] = None


# =============================================================================
# Tick Generator
# =============================================================================


class TickGenerator:
    """
    Generates realistic tick streams for forex simulation.

    Models tick arrival as a Poisson process with time-varying intensity,
    and price evolution as a jump-diffusion process.

    Tick Arrival Model:
        - Base: Poisson process with session-dependent intensity
        - Clustering: GARCH-like clustering during high activity
        - Gaps: Weekend and holiday gaps

    Price Model:
        - Diffusion: Geometric Brownian Motion with stochastic volatility
        - Jumps: Compound Poisson process for news events
        - Mean reversion: Short-term mean reversion around fair value

    References:
        - Dacorogna et al. (2001): "High-Frequency Finance"
        - Andersen et al. (2007): "Real-Time Price Discovery"
    """

    def __init__(
        self,
        config: Optional[TickSimulationConfig] = None,
        initial_mid: float = 1.0,
    ) -> None:
        """
        Initialize tick generator.

        Args:
            config: Simulation configuration
            initial_mid: Initial mid price
        """
        self.config = config or TickSimulationConfig()
        self._rng = np.random.default_rng(self.config.seed)

        # Price state
        self._current_mid = initial_mid
        self._current_spread_pips = SPREAD_PARAMS[self.config.pair_type]["base"]

        # Volatility state (for stochastic volatility)
        self._current_vol = 0.0001  # Daily vol as decimal
        self._vol_mean = 0.0001
        self._vol_kappa = 0.1  # Mean reversion speed

        # Spread state
        self._spread_state = SpreadState(
            base_spread_pips=self._current_spread_pips,
            current_spread_pips=self._current_spread_pips,
        )

        # Tick tracking
        self._sequence = 0
        self._last_tick_ns = 0
        self._tick_history: Deque[Tick] = deque(maxlen=1000)

        # Jump state
        self._in_jump = False
        self._jump_start_ns = 0
        self._jump_size = 0.0

        # Determine pip value
        self._pip_value = 0.01 if "JPY" in self.config.pair_type.upper() else 0.0001

    def generate_tick(
        self,
        timestamp_ns: Optional[int] = None,
        market_condition: MarketCondition = MarketCondition.NORMAL,
    ) -> Tick:
        """
        Generate a single tick.

        Args:
            timestamp_ns: Tick timestamp (uses current time if None)
            market_condition: Current market condition

        Returns:
            Generated Tick
        """
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()

        # Calculate time delta
        if self._last_tick_ns > 0:
            dt_sec = (timestamp_ns - self._last_tick_ns) / 1e9
        else:
            dt_sec = 0.001  # 1ms default

        dt_sec = max(0.0001, min(1.0, dt_sec))  # Clamp

        # Update volatility (Ornstein-Uhlenbeck process)
        vol_noise = self._rng.normal(0, 0.00001)
        self._current_vol += self._vol_kappa * (self._vol_mean - self._current_vol) * dt_sec
        self._current_vol += vol_noise * math.sqrt(dt_sec)
        self._current_vol = max(0.00001, self._current_vol)

        # Generate price innovation
        scaled_vol = self._current_vol * self.config.volatility_scale
        price_return = self._rng.normal(0, scaled_vol * math.sqrt(dt_sec))

        # Check for jump
        if self.config.jump_enabled and market_condition == MarketCondition.NEWS_EVENT:
            jump_prob = JUMP_PARAMS["intensity_per_hour"] / 3600.0 * dt_sec
            if self._rng.random() < jump_prob:
                jump_size = self._rng.normal(0, JUMP_PARAMS["jump_size_std_pips"])
                price_return += jump_size * self._pip_value
                self._in_jump = True
                self._jump_start_ns = timestamp_ns

        # Update mid price
        self._current_mid *= 1 + price_return

        # Update spread
        self._update_spread(timestamp_ns, market_condition)

        # Calculate bid/ask
        half_spread = self._spread_state.total_spread_pips * self._pip_value / 2.0
        bid = self._current_mid - half_spread
        ask = self._current_mid + half_spread

        # Create tick
        self._sequence += 1
        tick = Tick(
            timestamp_ns=timestamp_ns,
            bid=bid,
            ask=ask,
            tick_type=TickType.QUOTE,
            dealer_id=f"dealer_{self._rng.integers(0, 5)}",
            sequence_num=self._sequence,
            is_jpy_pair="JPY" in self.config.pair_type.upper(),
        )

        self._tick_history.append(tick)
        self._last_tick_ns = timestamp_ns

        return tick

    def generate_tick_stream(
        self,
        duration_sec: float,
        start_timestamp_ns: Optional[int] = None,
    ) -> Generator[Tick, None, None]:
        """
        Generate a stream of ticks over a duration.

        Uses Poisson process for tick arrival times.

        Args:
            duration_sec: Duration in seconds
            start_timestamp_ns: Start timestamp

        Yields:
            Sequence of Tick objects
        """
        if start_timestamp_ns is None:
            start_timestamp_ns = time.time_ns()

        current_ns = start_timestamp_ns
        end_ns = start_timestamp_ns + int(duration_sec * 1e9)

        # Get tick rate for session
        tick_rate = TICK_ARRIVAL_RATES.get(self.config.session, self.config.base_tick_rate)

        while current_ns < end_ns:
            # Inter-arrival time (exponential distribution)
            inter_arrival_sec = self._rng.exponential(1.0 / tick_rate)
            current_ns += int(inter_arrival_sec * 1e9)

            if current_ns >= end_ns:
                break

            yield self.generate_tick(current_ns)

    def _update_spread(
        self,
        timestamp_ns: int,
        market_condition: MarketCondition,
    ) -> None:
        """Update spread state with dynamics."""
        params = SPREAD_PARAMS[self.config.pair_type]

        # Volatility component
        vol_contribution = (
            params["volatility_sensitivity"]
            * (self._current_vol / self._vol_mean - 1.0)
            * params["base"]
        )
        self._spread_state.volatility_component = max(0, vol_contribution)

        # News component
        if market_condition == MarketCondition.NEWS_EVENT:
            self._spread_state.news_component = params["base"] * (params["news_multiplier"] - 1.0)
        elif self._spread_state.news_component > 0:
            # Decay news component
            decay = 0.1  # 10% per update
            self._spread_state.news_component *= 1 - decay
            if self._spread_state.news_component < 0.01:
                self._spread_state.news_component = 0.0

        # Recovery from jump
        if self._in_jump:
            elapsed_sec = (timestamp_ns - self._jump_start_ns) / 1e9
            half_life = JUMP_PARAMS["recovery_half_life_sec"]
            recovery_factor = 1.0 - math.exp(-0.693 * elapsed_sec / half_life)

            if recovery_factor > 0.95:
                self._in_jump = False

        self._spread_state.current_spread_pips = self._spread_state.total_spread_pips
        self._spread_state.last_update_ns = timestamp_ns

    def get_current_state(self) -> Dict[str, Any]:
        """Get current generator state."""
        return {
            "mid": self._current_mid,
            "spread_pips": self._spread_state.total_spread_pips,
            "volatility": self._current_vol,
            "sequence": self._sequence,
            "in_jump": self._in_jump,
        }

    def set_mid_price(self, mid: float) -> None:
        """Set current mid price."""
        self._current_mid = mid


# =============================================================================
# Tick-Level Execution Simulator
# =============================================================================


class TickLevelExecutor:
    """
    Tick-level execution simulator for forex.

    Simulates order execution at tick resolution with:
    - Latency modeling (order submission to fill)
    - Price movement during execution
    - Adverse selection detection
    - Partial fills
    - Slippage calculation

    Execution Flow:
    1. Order submitted → latency delay
    2. During delay, market ticks continue
    3. At execution time, check if price is acceptable
    4. Apply adverse selection filter
    5. Fill at current market price (with possible improvement)

    References:
        - Hasbrouck (2007): "Empirical Market Microstructure"
        - Chaboud et al. (2014): "Rise of the Machines"
    """

    def __init__(
        self,
        tick_generator: TickGenerator,
        config: Optional[TickSimulationConfig] = None,
    ) -> None:
        """
        Initialize tick executor.

        Args:
            tick_generator: Tick generator for price simulation
            config: Simulation configuration
        """
        self._tick_gen = tick_generator
        self.config = config or tick_generator.config
        self._rng = np.random.default_rng(self.config.seed)

        # Execution statistics
        self._stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "rejected_orders": 0,
            "total_slippage_pips": 0.0,
            "avg_latency_ms": 0.0,
        }

    def execute_market_order(
        self,
        is_buy: bool,
        size: float,
        submitted_at_ns: int,
        max_slippage_pips: float = 5.0,
    ) -> TickExecutionResult:
        """
        Execute a market order at tick resolution.

        Simulates the full execution lifecycle:
        1. Order submission
        2. Network/processing latency
        3. Market movement during latency
        4. Adverse selection check
        5. Fill at available price

        Args:
            is_buy: True for buy, False for sell
            size: Order size
            submitted_at_ns: Order submission timestamp
            max_slippage_pips: Maximum acceptable slippage

        Returns:
            TickExecutionResult with fill details
        """
        self._stats["total_orders"] += 1

        # Get initial price
        initial_tick = self._tick_gen.generate_tick(submitted_at_ns)
        initial_price = initial_tick.ask if is_buy else initial_tick.bid

        # Simulate latency
        latency_ns = self._sample_latency_ns()
        execution_ns = submitted_at_ns + latency_ns

        # Generate ticks during latency window
        ticks_during_latency: List[Tick] = []
        current_ns = submitted_at_ns

        while current_ns < execution_ns:
            tick_rate = TICK_ARRIVAL_RATES.get(self.config.session, 3.0)
            inter_arrival = self._rng.exponential(1.0 / tick_rate)
            current_ns += int(inter_arrival * 1e9)

            if current_ns < execution_ns:
                tick = self._tick_gen.generate_tick(current_ns)
                ticks_during_latency.append(tick)

        # Get execution tick
        execution_tick = self._tick_gen.generate_tick(execution_ns)
        execution_price = execution_tick.ask if is_buy else execution_tick.bid

        # Calculate market movement
        pip_value = 0.01 if execution_tick.is_jpy_pair else 0.0001
        market_moved_pips = (execution_price - initial_price) / pip_value
        if not is_buy:
            market_moved_pips = -market_moved_pips

        # Adverse selection check
        # If market moved significantly in our favor, dealer may reject
        adverse_threshold = self.config.adverse_selection_threshold_pips

        if market_moved_pips < -adverse_threshold:
            # Market moved against dealer in our favor
            reject_prob = min(0.8, abs(market_moved_pips) / 5.0)

            if self._rng.random() < reject_prob:
                self._stats["rejected_orders"] += 1
                return TickExecutionResult(
                    filled=False,
                    rejection_reason="adverse_selection",
                    latency_ns=latency_ns,
                    ticks_to_fill=len(ticks_during_latency),
                    market_moved_pips=market_moved_pips,
                )

        # Calculate slippage
        slippage_pips = abs(market_moved_pips)

        # Check max slippage
        if slippage_pips > max_slippage_pips:
            self._stats["rejected_orders"] += 1
            return TickExecutionResult(
                filled=False,
                rejection_reason="max_slippage_exceeded",
                latency_ns=latency_ns,
                ticks_to_fill=len(ticks_during_latency),
                market_moved_pips=market_moved_pips,
                slippage_pips=slippage_pips,
            )

        # Determine execution quality
        if slippage_pips < 0.1:
            quality = ExecutionQuality.EXCELLENT
        elif slippage_pips < 0.5:
            quality = ExecutionQuality.GOOD
        elif slippage_pips < 2.0:
            quality = ExecutionQuality.FAIR
        else:
            quality = ExecutionQuality.POOR

        # Price improvement possibility
        improvement = 0.0
        if self._rng.random() < 0.15:  # 15% chance of improvement
            max_improvement = execution_tick.spread_pips * 0.3
            improvement = self._rng.uniform(0, max_improvement) * pip_value
            if is_buy:
                execution_price -= improvement
            else:
                execution_price += improvement

        # Update statistics
        self._stats["filled_orders"] += 1
        self._stats["total_slippage_pips"] += slippage_pips

        n = self._stats["filled_orders"]
        self._stats["avg_latency_ms"] = (n - 1) / n * self._stats[
            "avg_latency_ms"
        ] + latency_ns / 1_000_000.0 / n

        return TickExecutionResult(
            filled=True,
            fill_price=execution_price,
            fill_timestamp_ns=execution_ns,
            slippage_pips=slippage_pips if market_moved_pips > 0 else 0.0,
            execution_quality=quality,
            latency_ns=latency_ns,
            ticks_to_fill=len(ticks_during_latency),
            market_moved_pips=market_moved_pips,
        )

    def execute_limit_order(
        self,
        is_buy: bool,
        size: float,
        limit_price: float,
        submitted_at_ns: int,
        time_in_force_sec: float = 60.0,
    ) -> TickExecutionResult:
        """
        Execute a limit order at tick resolution.

        Monitors tick stream for fill opportunity within time limit.

        Args:
            is_buy: True for buy, False for sell
            size: Order size
            limit_price: Limit price
            submitted_at_ns: Order submission timestamp
            time_in_force_sec: Order validity duration

        Returns:
            TickExecutionResult with fill details
        """
        self._stats["total_orders"] += 1

        # Determine pip value
        current_tick = self._tick_gen.generate_tick(submitted_at_ns)
        pip_value = 0.01 if current_tick.is_jpy_pair else 0.0001

        # Apply initial latency for order to reach market
        initial_latency_ns = self._sample_latency_ns()
        market_entry_ns = submitted_at_ns + initial_latency_ns
        expiry_ns = market_entry_ns + int(time_in_force_sec * 1e9)

        # Monitor ticks for fill
        current_ns = market_entry_ns
        ticks_checked = 0
        fill_ns: Optional[int] = None
        fill_price: Optional[float] = None

        tick_rate = TICK_ARRIVAL_RATES.get(self.config.session, 3.0)

        while current_ns < expiry_ns:
            # Next tick
            inter_arrival = self._rng.exponential(1.0 / tick_rate)
            current_ns += int(inter_arrival * 1e9)

            if current_ns >= expiry_ns:
                break

            tick = self._tick_gen.generate_tick(current_ns)
            ticks_checked += 1

            # Check fill condition
            if is_buy:
                # Buy limit: fill if ask <= limit_price
                if tick.ask <= limit_price:
                    fill_ns = current_ns
                    fill_price = tick.ask
                    break
            else:
                # Sell limit: fill if bid >= limit_price
                if tick.bid >= limit_price:
                    fill_ns = current_ns
                    fill_price = tick.bid
                    break

        if fill_ns is None or fill_price is None:
            # Order expired without fill
            self._stats["rejected_orders"] += 1
            return TickExecutionResult(
                filled=False,
                rejection_reason="expired",
                latency_ns=initial_latency_ns,
                ticks_to_fill=ticks_checked,
            )

        # Calculate slippage (for limits, could be price improvement)
        if is_buy:
            slippage_pips = (fill_price - limit_price) / pip_value
        else:
            slippage_pips = (limit_price - fill_price) / pip_value

        # Negative slippage = price improvement
        quality = ExecutionQuality.GOOD
        if slippage_pips < -0.1:
            quality = ExecutionQuality.EXCELLENT
        elif slippage_pips > 0.5:
            quality = ExecutionQuality.FAIR

        self._stats["filled_orders"] += 1
        if slippage_pips > 0:
            self._stats["total_slippage_pips"] += slippage_pips

        total_latency_ns = fill_ns - submitted_at_ns

        return TickExecutionResult(
            filled=True,
            fill_price=fill_price,
            fill_timestamp_ns=fill_ns,
            slippage_pips=max(0, slippage_pips),
            execution_quality=quality,
            latency_ns=total_latency_ns,
            ticks_to_fill=ticks_checked,
        )

    def _sample_latency_ns(self) -> int:
        """Sample execution latency from configured distribution."""
        if self.config.latency_model == "lognormal":
            # Log-normal distribution (realistic for network latency)
            mean_log = math.log(self.config.mean_latency_ms)
            std_log = self.config.latency_std_ms / self.config.mean_latency_ms
            latency_ms = self._rng.lognormal(mean_log, std_log)
        elif self.config.latency_model == "exponential":
            latency_ms = self._rng.exponential(self.config.mean_latency_ms)
        else:  # normal
            latency_ms = self._rng.normal(
                self.config.mean_latency_ms,
                self.config.latency_std_ms,
            )

        # Clamp to reasonable range
        latency_ms = max(1.0, min(self.config.max_latency_ms, latency_ms))

        return int(latency_ms * 1_000_000)  # Convert to nanoseconds

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        filled = self._stats["filled_orders"]
        return {
            **self._stats,
            "fill_rate": filled / max(1, self._stats["total_orders"]) * 100,
            "avg_slippage_pips": (self._stats["total_slippage_pips"] / max(1, filled)),
        }

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "rejected_orders": 0,
            "total_slippage_pips": 0.0,
            "avg_latency_ms": 0.0,
        }


# =============================================================================
# Price Impact Model at Tick Level
# =============================================================================


class TickPriceImpact:
    """
    Models price impact at tick resolution.

    In OTC forex markets, price impact is different from exchange markets:
    - No visible order book
    - Impact through dealer inventory management
    - Spreads widen for large orders
    - Impact dissipates differently

    Model:
        temporary_impact = σ × √(Q/ADV) × λ_temp
        permanent_impact = γ × (Q/ADV)

    where:
        σ = volatility
        Q = order size
        ADV = average daily volume
        λ_temp, γ = impact coefficients

    References:
        - Evans & Lyons (2002): "Order Flow and Exchange Rate Dynamics"
        - Osler (2008): "Foreign Exchange Microstructure"
    """

    def __init__(
        self,
        temp_impact_coef: float = 0.1,
        perm_impact_coef: float = 0.02,
        decay_half_life_sec: float = 60.0,
    ) -> None:
        """
        Initialize price impact model.

        Args:
            temp_impact_coef: Temporary impact coefficient
            perm_impact_coef: Permanent impact coefficient
            decay_half_life_sec: Temporary impact decay half-life
        """
        self.temp_coef = temp_impact_coef
        self.perm_coef = perm_impact_coef
        self.decay_half_life = decay_half_life_sec

        # Track recent impacts for decay calculation
        self._impact_history: Deque[Tuple[int, float]] = deque(maxlen=100)

    def calculate_impact(
        self,
        order_size: float,
        adv: float,
        volatility: float,
        is_buy: bool,
    ) -> Tuple[float, float]:
        """
        Calculate price impact for an order.

        Args:
            order_size: Order size in base currency
            adv: Average daily volume
            volatility: Current volatility (daily)
            is_buy: True for buy order

        Returns:
            (temporary_impact_pips, permanent_impact_pips) tuple
            Positive = price increase
        """
        if adv <= 0:
            adv = 1e9  # Default ADV

        participation = order_size / adv

        # Temporary impact (square root model)
        temp_impact = volatility * math.sqrt(participation) * self.temp_coef * 10000

        # Permanent impact (linear model)
        perm_impact = participation * self.perm_coef * 10000

        # Direction
        sign = 1.0 if is_buy else -1.0

        return (temp_impact * sign, perm_impact * sign)

    def get_decayed_impact(
        self,
        timestamp_ns: int,
    ) -> float:
        """
        Get total decayed impact from recent orders.

        Args:
            timestamp_ns: Current timestamp

        Returns:
            Total remaining impact in pips
        """
        total_impact = 0.0
        decay_rate = 0.693 / self.decay_half_life  # ln(2) / half_life

        for impact_ns, impact_pips in self._impact_history:
            elapsed_sec = (timestamp_ns - impact_ns) / 1e9
            remaining = impact_pips * math.exp(-decay_rate * elapsed_sec)
            if abs(remaining) > 0.01:  # Threshold
                total_impact += remaining

        return total_impact

    def record_impact(
        self,
        timestamp_ns: int,
        impact_pips: float,
    ) -> None:
        """Record an impact for decay tracking."""
        self._impact_history.append((timestamp_ns, impact_pips))


# =============================================================================
# Factory Functions
# =============================================================================


def create_tick_simulator(
    pair: str = "EUR_USD",
    session: str = "london",
    initial_price: float = 1.10,
    volatility_scale: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[TickGenerator, TickLevelExecutor]:
    """
    Create a complete tick simulation environment.

    Args:
        pair: Currency pair
        session: Trading session
        initial_price: Initial mid price
        volatility_scale: Volatility multiplier
        seed: Random seed

    Returns:
        (TickGenerator, TickLevelExecutor) tuple
    """
    # Determine pair type
    pair_upper = pair.upper().replace("/", "_")

    if pair_upper in {"EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD"}:
        pair_type = "major"
    elif "TRY" in pair_upper or "ZAR" in pair_upper or "MXN" in pair_upper:
        pair_type = "exotic"
    elif "JPY" in pair_upper and pair_upper != "USD_JPY":
        pair_type = "cross"
    else:
        pair_type = "minor"

    config = TickSimulationConfig(
        pair_type=pair_type,
        session=session,
        volatility_scale=volatility_scale,
        seed=seed,
    )

    generator = TickGenerator(config=config, initial_mid=initial_price)
    executor = TickLevelExecutor(tick_generator=generator, config=config)

    return generator, executor


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "TickType",
    "ExecutionQuality",
    "MarketCondition",
    # Data classes
    "Tick",
    "TickExecutionResult",
    "SpreadState",
    "TickSimulationConfig",
    # Classes
    "TickGenerator",
    "TickLevelExecutor",
    "TickPriceImpact",
    # Factory
    "create_tick_simulator",
    # Constants
    "TICK_ARRIVAL_RATES",
    "SPREAD_PARAMS",
]
