# -*- coding: utf-8 -*-
"""
services/forex_dealer.py
OTC Forex Dealer Quote Simulation

Phase 5: Forex Integration (2025-11-30)

Unlike exchange LOBs (used for crypto/equity L3), forex is OTC with dealer quotes.
This module simulates the dealer market structure:

1. Multi-dealer quote aggregation (like ECN)
2. Last-look rejection simulation
3. Quote flickering (rapid updates)
4. Size-dependent spread widening
5. Latency arbitrage protection
6. Session-dependent liquidity

Key Differences from LOB Simulation:
- NO queue position (no FIFO matching)
- NO price-time priority
- Dealer discretion (last-look)
- Indicative vs firm quotes
- Request-for-quote (RFQ) model for large sizes

Integration with ForexParametricSlippageProvider:
- Provides dealer-level quote simulation
- Last-look rejection adds realistic execution uncertainty
- Size-dependent spread widening complements TCA model

References:
- Oomen (2017): "Last Look" in FX - Journal of Financial Markets
- Hasbrouck & Saar (2013): "Low-latency trading"
- King, Osler, Rime (2012): "Foreign Exchange Market Structure"
- BIS (2022): Triennial Survey - FX Market Structure
- Chaboud et al. (2014): "Rise of the Machines"
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
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Minimum dealer latency (nanoseconds) - 10 microseconds
MIN_DEALER_LATENCY_NS: int = 10_000

# Maximum dealer latency (nanoseconds) - 500 milliseconds
MAX_DEALER_LATENCY_NS: int = 500_000_000

# Default quote validity (milliseconds)
DEFAULT_QUOTE_VALIDITY_MS: int = 200

# Default last-look window (milliseconds)
DEFAULT_LAST_LOOK_MS: int = 200

# Price stability threshold (pips) for adverse selection detection
DEFAULT_ADVERSE_THRESHOLD_PIPS: float = 0.3


# =============================================================================
# Enums
# =============================================================================


class QuoteType(str, Enum):
    """Type of dealer quote."""

    FIRM = "firm"  # Executable quote
    INDICATIVE = "indicative"  # May be rejected
    LAST_LOOK = "last_look"  # Subject to last-look window


class RejectReason(str, Enum):
    """Reason for trade rejection."""

    NONE = "none"
    LAST_LOOK_ADVERSE = "last_look_adverse"
    SIZE_EXCEEDED = "size_exceeded"
    QUOTE_EXPIRED = "quote_expired"
    PRICE_MOVED = "price_moved"
    LATENCY_ARBITRAGE = "latency_arbitrage"
    DEALER_DISCRETION = "dealer_discretion"
    MARKET_CLOSED = "market_closed"


class DealerTier(str, Enum):
    """Dealer tier classification."""

    TIER1 = "tier1"  # Primary liquidity providers (tightest spreads)
    TIER2 = "tier2"  # Secondary dealers (moderate spreads)
    TIER3 = "tier3"  # Retail aggregators (wider spreads)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DealerProfile:
    """
    Individual dealer characteristics.

    Models heterogeneous dealer pool with different characteristics:
    - Primary LPs have tighter spreads, larger capacity, lower rejection rates
    - Tier 2/3 dealers provide depth with varying characteristics

    Attributes:
        dealer_id: Unique dealer identifier
        tier: Dealer tier classification
        spread_factor: Multiplier on base spread (lower = tighter)
        max_size_usd: Maximum single trade size
        last_look_window_ms: Last-look review window in milliseconds
        base_reject_prob: Baseline rejection probability
        adverse_threshold_pips: Price move threshold for adverse selection
        latency_ms: Typical quote/execution latency
        quote_flicker_prob: Probability of quote update per check
        is_primary: Whether this is a primary liquidity provider
        active_hours: Hours UTC when dealer is active (None = 24/7)
    """

    dealer_id: str
    tier: DealerTier = DealerTier.TIER2
    spread_factor: float = 1.0
    max_size_usd: float = 5_000_000.0
    last_look_window_ms: int = DEFAULT_LAST_LOOK_MS
    base_reject_prob: float = 0.05
    adverse_threshold_pips: float = DEFAULT_ADVERSE_THRESHOLD_PIPS
    latency_ms: float = 50.0
    quote_flicker_prob: float = 0.1
    is_primary: bool = False
    active_hours: Optional[Tuple[int, int]] = None  # (start_hour, end_hour) UTC

    def __post_init__(self) -> None:
        """Validate dealer profile parameters."""
        if self.spread_factor <= 0:
            raise ValueError("spread_factor must be positive")
        if self.max_size_usd <= 0:
            raise ValueError("max_size_usd must be positive")
        if not 0 <= self.base_reject_prob <= 1:
            raise ValueError("base_reject_prob must be in [0, 1]")
        if self.last_look_window_ms < 0:
            raise ValueError("last_look_window_ms must be non-negative")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")


@dataclass
class DealerQuote:
    """
    Single dealer quote.

    Represents a two-sided quote from a single dealer with metadata
    about quote type, validity, and sizing.

    Attributes:
        dealer_id: Source dealer identifier
        bid: Bid price
        ask: Ask price
        bid_size_usd: Available bid size in USD
        ask_size_usd: Available ask size in USD
        timestamp_ns: Quote timestamp in nanoseconds
        quote_type: Type of quote (firm, indicative, last_look)
        valid_for_ms: Quote validity window in milliseconds
        last_look_ms: Last-look window for this quote
        sequence_num: Quote sequence number for staleness detection
    """

    dealer_id: str
    bid: float
    ask: float
    bid_size_usd: float
    ask_size_usd: float
    timestamp_ns: int
    quote_type: QuoteType = QuoteType.LAST_LOOK
    valid_for_ms: int = DEFAULT_QUOTE_VALIDITY_MS
    last_look_ms: int = DEFAULT_LAST_LOOK_MS
    sequence_num: int = 0

    @property
    def mid(self) -> float:
        """Get mid price."""
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        """Get absolute spread."""
        return self.ask - self.bid

    def spread_pips(self, is_jpy_pair: bool = False) -> float:
        """
        Get spread in pips.

        Args:
            is_jpy_pair: Whether this is a JPY pair (different pip size)

        Returns:
            Spread in pips
        """
        pip_size = 0.01 if is_jpy_pair else 0.0001
        return self.spread / pip_size

    def is_valid(self, current_time_ns: int) -> bool:
        """
        Check if quote is still valid.

        Args:
            current_time_ns: Current time in nanoseconds

        Returns:
            True if quote is still valid
        """
        elapsed_ms = (current_time_ns - self.timestamp_ns) / 1_000_000
        return elapsed_ms < self.valid_for_ms


@dataclass
class AggregatedQuote:
    """
    Best bid/ask across all dealers (ECN-style aggregation).

    Represents the top-of-book from aggregating multiple dealer quotes,
    similar to an ECN (Electronic Communication Network) view.

    Attributes:
        best_bid: Best (highest) bid price
        best_ask: Best (lowest) ask price
        best_bid_dealer: Dealer providing best bid
        best_ask_dealer: Dealer providing best ask
        total_bid_size: Total available bid size at best price
        total_ask_size: Total available ask size at best price
        dealer_quotes: List of all dealer quotes
        timestamp_ns: Aggregation timestamp
        num_active_dealers: Number of dealers with active quotes
    """

    best_bid: float
    best_ask: float
    best_bid_dealer: str
    best_ask_dealer: str
    total_bid_size: float
    total_ask_size: float
    dealer_quotes: List[DealerQuote]
    timestamp_ns: int
    num_active_dealers: int = 0

    @property
    def mid(self) -> float:
        """Get mid price from best bid/ask."""
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        """Get best spread."""
        return self.best_ask - self.best_bid

    def spread_pips(self, is_jpy_pair: bool = False) -> float:
        """Get spread in pips."""
        pip_size = 0.01 if is_jpy_pair else 0.0001
        return self.spread / pip_size

    def is_crossed(self) -> bool:
        """Check if market is crossed (bid >= ask)."""
        return self.best_bid >= self.best_ask

    def get_depth_at_level(self, level: int, is_bid: bool) -> Tuple[float, float]:
        """
        Get price and size at a specific depth level.

        Args:
            level: Depth level (0 = top of book)
            is_bid: Whether to get bid (True) or ask (False) side

        Returns:
            (price, size) tuple at the level, or (0, 0) if not available
        """
        if is_bid:
            sorted_quotes = sorted(
                [q for q in self.dealer_quotes if q.bid_size_usd > 0], key=lambda q: -q.bid
            )
        else:
            sorted_quotes = sorted(
                [q for q in self.dealer_quotes if q.ask_size_usd > 0], key=lambda q: q.ask
            )

        if level >= len(sorted_quotes):
            return (0.0, 0.0)

        quote = sorted_quotes[level]
        if is_bid:
            return (quote.bid, quote.bid_size_usd)
        return (quote.ask, quote.ask_size_usd)


@dataclass
class ExecutionResult:
    """
    Result of execution attempt.

    Contains all information about the execution outcome including
    fill details, latency, and rejection reason if applicable.

    Attributes:
        filled: Whether the order was filled
        fill_price: Execution price (None if not filled)
        fill_qty: Filled quantity in USD (None if not filled)
        dealer_id: Dealer that filled the order (None if not filled)
        latency_ns: Total execution latency in nanoseconds
        reject_reason: Reason for rejection if not filled
        slippage_pips: Slippage from reference price in pips
        last_look_passed: Whether last-look check passed
        price_improvement: Price improvement from quote (positive = better)
        partial_fill: Whether this was a partial fill
        remaining_qty: Remaining quantity after partial fill
    """

    filled: bool
    fill_price: Optional[float] = None
    fill_qty: Optional[float] = None
    dealer_id: Optional[str] = None
    latency_ns: int = 0
    reject_reason: RejectReason = RejectReason.NONE
    slippage_pips: float = 0.0
    last_look_passed: bool = True
    price_improvement: float = 0.0
    partial_fill: bool = False
    remaining_qty: float = 0.0

    @property
    def latency_ms(self) -> float:
        """Get latency in milliseconds."""
        return self.latency_ns / 1_000_000.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "filled": self.filled,
            "fill_price": self.fill_price,
            "fill_qty": self.fill_qty,
            "dealer_id": self.dealer_id,
            "latency_ms": self.latency_ms,
            "reject_reason": self.reject_reason.value,
            "slippage_pips": self.slippage_pips,
            "last_look_passed": self.last_look_passed,
            "price_improvement": self.price_improvement,
            "partial_fill": self.partial_fill,
            "remaining_qty": self.remaining_qty,
        }


@dataclass
class ForexDealerConfig:
    """
    Configuration for dealer simulation.

    Controls all aspects of the multi-dealer OTC market simulation.

    Attributes:
        num_dealers: Number of dealers in the pool
        base_spread_pips: Base spread in pips (before adjustments)
        last_look_enabled: Whether last-look rejection is active
        size_impact_threshold_usd: Order size threshold for spread widening
        size_impact_factor: Spread widening per $1M above threshold
        session_spread_adjustment: Whether to adjust spreads by session
        latency_variance: Coefficient of variation for latency
        max_slippage_pips: Maximum allowed slippage in pips
        enable_price_improvement: Whether dealers can offer price improvement
        price_improvement_prob: Probability of price improvement
        max_price_improvement_pips: Maximum price improvement in pips
        quote_refresh_interval_ms: Interval for quote updates
        enable_partial_fills: Whether to allow partial fills
        min_quote_size_usd: Minimum quote size offered
        max_history_size: Maximum size of execution history
        adverse_selection_decay_ms: Decay constant for adverse selection memory
    """

    num_dealers: int = 5
    base_spread_pips: float = 1.0
    last_look_enabled: bool = True
    size_impact_threshold_usd: float = 1_000_000.0
    size_impact_factor: float = 0.5
    session_spread_adjustment: bool = True
    latency_variance: float = 0.3
    max_slippage_pips: float = 10.0
    enable_price_improvement: bool = True
    price_improvement_prob: float = 0.15
    max_price_improvement_pips: float = 0.2
    quote_refresh_interval_ms: int = 100
    enable_partial_fills: bool = True
    min_quote_size_usd: float = 10_000.0
    max_history_size: int = 1000
    adverse_selection_decay_ms: float = 5000.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.num_dealers < 1:
            raise ValueError("num_dealers must be at least 1")
        if self.base_spread_pips < 0:
            raise ValueError("base_spread_pips must be non-negative")
        if self.size_impact_threshold_usd <= 0:
            raise ValueError("size_impact_threshold_usd must be positive")
        if self.size_impact_factor < 0:
            raise ValueError("size_impact_factor must be non-negative")
        if self.latency_variance < 0:
            raise ValueError("latency_variance must be non-negative")
        if self.max_slippage_pips <= 0:
            raise ValueError("max_slippage_pips must be positive")


@dataclass
class ExecutionStats:
    """
    Statistics about execution quality.

    Tracks metrics for monitoring and analysis of execution quality.
    """

    total_attempts: int = 0
    total_fills: int = 0
    total_rejections: int = 0
    rejection_by_reason: Dict[str, int] = field(default_factory=dict)
    avg_slippage_pips: float = 0.0
    avg_latency_ms: float = 0.0
    total_volume_usd: float = 0.0
    price_improvements: int = 0
    last_look_rejections: int = 0
    partial_fills: int = 0

    @property
    def fill_rate(self) -> float:
        """Get fill rate as percentage."""
        if self.total_attempts == 0:
            return 0.0
        return (self.total_fills / self.total_attempts) * 100.0


# =============================================================================
# Protocols
# =============================================================================


class SessionLiquidityProvider(Protocol):
    """Protocol for session liquidity factor providers."""

    def get_liquidity_factor(self, timestamp_ms: int) -> float:
        """Get liquidity factor for the given timestamp."""
        ...


class QuoteFlickerSimulator(Protocol):
    """Protocol for quote flicker simulation."""

    def should_update_quote(self, dealer_id: str, elapsed_ms: float) -> bool:
        """Check if dealer should update quote."""
        ...


# =============================================================================
# Main Implementation
# =============================================================================


class ForexDealerSimulator:
    """
    Simulates multi-dealer OTC forex market.

    This is fundamentally different from LOB simulation:
    - Dealers provide quotes (no order book depth in traditional sense)
    - Last-look gives dealers rejection rights
    - No queue position (not FIFO)
    - Price improvement possible
    - Session-dependent liquidity

    Usage:
        simulator = ForexDealerSimulator(config=ForexDealerConfig())

        # Get aggregated quote
        quote = simulator.get_aggregated_quote(
            symbol="EUR_USD",
            mid_price=1.0850,
            session_factor=1.1,
            order_size_usd=500000,
        )

        # Attempt execution
        result = simulator.attempt_execution(
            is_buy=True,
            size_usd=500000,
            quote=quote,
            current_mid=1.0851,
        )

        if result.filled:
            print(f"Filled at {result.fill_price} by {result.dealer_id}")
        else:
            print(f"Rejected: {result.reject_reason}")

    Integration with ForexParametricSlippageProvider:
        The simulator provides additional execution uncertainty via:
        1. Last-look rejection simulation
        2. Size-dependent spread widening
        3. Dealer-level latency simulation

        Use slippage from ForexParametricSlippageProvider for expected cost,
        then use this simulator for realistic execution simulation.

    References:
        - Oomen (2017): "Last Look" - Journal of Financial Markets
        - King et al. (2012): "FX Market Structure"
    """

    def __init__(
        self,
        config: Optional[ForexDealerConfig] = None,
        seed: Optional[int] = None,
        session_provider: Optional[SessionLiquidityProvider] = None,
    ) -> None:
        """
        Initialize forex dealer simulator.

        Args:
            config: Simulation configuration
            seed: Random seed for reproducibility
            session_provider: Optional session liquidity provider
        """
        self.config = config or ForexDealerConfig()
        self._rng = np.random.default_rng(seed)
        self._session_provider = session_provider

        # Create heterogeneous dealer pool
        self._dealers: List[DealerProfile] = self._create_dealers()
        self._dealer_map: Dict[str, DealerProfile] = {d.dealer_id: d for d in self._dealers}

        # Quote state
        self._current_quotes: Dict[str, DealerQuote] = {}
        self._quote_sequence: int = 0

        # Execution history for adaptive behavior
        self._execution_history: Deque[ExecutionResult] = deque(maxlen=self.config.max_history_size)
        self._adverse_selection_memory: Dict[str, float] = {}

        # Statistics
        self._stats = ExecutionStats()

    def _create_dealers(self) -> List[DealerProfile]:
        """
        Create heterogeneous dealer pool.

        Creates a mix of tier 1, 2, and 3 dealers with varying
        characteristics for realistic market simulation.

        Returns:
            List of dealer profiles
        """
        dealers = []
        num = self.config.num_dealers

        # Tier distribution: 20% T1, 40% T2, 40% T3
        num_t1 = max(1, num // 5)
        num_t2 = (num - num_t1) // 2
        num_t3 = num - num_t1 - num_t2

        for i in range(num):
            if i < num_t1:
                # Tier 1: Primary LPs
                tier = DealerTier.TIER1
                spread_factor = 0.7 + self._rng.uniform(-0.1, 0.1)
                max_size = 15_000_000.0 + self._rng.uniform(-3e6, 5e6)
                last_look_ms = int(100 + self._rng.uniform(-20, 50))
                reject_prob = 0.02 + self._rng.uniform(-0.01, 0.02)
                adverse_thresh = 0.15 + self._rng.uniform(-0.03, 0.05)
                latency = 20 + self._rng.uniform(-5, 15)
                is_primary = True
            elif i < num_t1 + num_t2:
                # Tier 2: Secondary dealers
                tier = DealerTier.TIER2
                spread_factor = 1.0 + self._rng.uniform(-0.15, 0.25)
                max_size = 5_000_000.0 + self._rng.uniform(-2e6, 3e6)
                last_look_ms = int(200 + self._rng.uniform(-50, 100))
                reject_prob = 0.05 + self._rng.uniform(-0.02, 0.05)
                adverse_thresh = 0.3 + self._rng.uniform(-0.05, 0.1)
                latency = 50 + self._rng.uniform(-10, 40)
                is_primary = False
            else:
                # Tier 3: Retail aggregators
                tier = DealerTier.TIER3
                spread_factor = 1.3 + self._rng.uniform(-0.1, 0.3)
                max_size = 2_000_000.0 + self._rng.uniform(-500e3, 1e6)
                last_look_ms = int(300 + self._rng.uniform(-50, 150))
                reject_prob = 0.08 + self._rng.uniform(-0.02, 0.07)
                adverse_thresh = 0.5 + self._rng.uniform(-0.1, 0.2)
                latency = 80 + self._rng.uniform(-20, 60)
                is_primary = False

            dealers.append(
                DealerProfile(
                    dealer_id=f"dealer_{i:02d}_{tier.value}",
                    tier=tier,
                    spread_factor=max(0.5, spread_factor),
                    max_size_usd=max(100_000, max_size),
                    last_look_window_ms=max(50, last_look_ms),
                    base_reject_prob=max(0.01, min(0.3, reject_prob)),
                    adverse_threshold_pips=max(0.1, adverse_thresh),
                    latency_ms=max(10, latency),
                    quote_flicker_prob=0.05 + self._rng.uniform(0, 0.15),
                    is_primary=is_primary,
                )
            )

        return dealers

    def get_aggregated_quote(
        self,
        symbol: str,
        mid_price: float,
        session_factor: float = 1.0,
        order_size_usd: float = 100_000.0,
        hour_utc: Optional[int] = None,
    ) -> AggregatedQuote:
        """
        Get aggregated quote from all dealers.

        Simulates ECN-style aggregation of dealer quotes with:
        - Session-adjusted spreads
        - Size-dependent spread widening
        - Dealer-specific characteristics

        Args:
            symbol: Currency pair (e.g., "EUR_USD")
            mid_price: Current mid price
            session_factor: Session liquidity factor (0.5 to 1.5)
            order_size_usd: Indicative order size for spread adjustment
            hour_utc: Current hour UTC for dealer availability check

        Returns:
            AggregatedQuote with best bid/ask and all dealer quotes
        """
        timestamp_ns = time.time_ns()
        quotes: List[DealerQuote] = []

        # Determine pip size based on symbol
        is_jpy = "JPY" in symbol.upper()
        pip_size = 0.01 if is_jpy else 0.0001
        base_spread = self.config.base_spread_pips * pip_size

        num_active = 0

        for dealer in self._dealers:
            # Check dealer availability
            if hour_utc is not None and dealer.active_hours is not None:
                start_h, end_h = dealer.active_hours
                if start_h <= end_h:
                    if not (start_h <= hour_utc < end_h):
                        continue
                else:  # Wraps around midnight
                    if not (hour_utc >= start_h or hour_utc < end_h):
                        continue

            # Compute dealer-specific spread
            spread = base_spread * dealer.spread_factor

            # Session adjustment (wider spreads in low liquidity)
            if self.config.session_spread_adjustment:
                spread = spread / max(0.3, session_factor)

            # Size impact (wider spreads for larger orders)
            if order_size_usd > self.config.size_impact_threshold_usd:
                excess_usd = order_size_usd - self.config.size_impact_threshold_usd
                size_widening = (excess_usd / 1_000_000) * self.config.size_impact_factor * pip_size
                spread += size_widening

            # Add noise for quote heterogeneity
            spread *= 1.0 + self._rng.uniform(-0.1, 0.1)
            spread = max(pip_size * 0.1, spread)  # Minimum spread

            half_spread = spread / 2.0
            bid = mid_price - half_spread
            ask = mid_price + half_spread

            # Available size varies by dealer
            size_mult = self._rng.uniform(0.4, 1.0)
            bid_size = dealer.max_size_usd * size_mult
            ask_size = dealer.max_size_usd * size_mult

            # Determine quote type
            if self.config.last_look_enabled:
                quote_type = QuoteType.LAST_LOOK
            else:
                quote_type = QuoteType.FIRM if dealer.is_primary else QuoteType.INDICATIVE

            self._quote_sequence += 1

            quote = DealerQuote(
                dealer_id=dealer.dealer_id,
                bid=bid,
                ask=ask,
                bid_size_usd=bid_size,
                ask_size_usd=ask_size,
                timestamp_ns=timestamp_ns,
                quote_type=quote_type,
                valid_for_ms=DEFAULT_QUOTE_VALIDITY_MS,
                last_look_ms=dealer.last_look_window_ms,
                sequence_num=self._quote_sequence,
            )
            quotes.append(quote)
            self._current_quotes[dealer.dealer_id] = quote
            num_active += 1

        if not quotes:
            # No dealers available (market closed scenario)
            return AggregatedQuote(
                best_bid=mid_price - base_spread,
                best_ask=mid_price + base_spread,
                best_bid_dealer="none",
                best_ask_dealer="none",
                total_bid_size=0.0,
                total_ask_size=0.0,
                dealer_quotes=[],
                timestamp_ns=timestamp_ns,
                num_active_dealers=0,
            )

        # Aggregate best bid/ask
        best_bid_quote = max(quotes, key=lambda q: q.bid)
        best_ask_quote = min(quotes, key=lambda q: q.ask)

        # Total size at best levels
        total_bid = sum(q.bid_size_usd for q in quotes if q.bid == best_bid_quote.bid)
        total_ask = sum(q.ask_size_usd for q in quotes if q.ask == best_ask_quote.ask)

        return AggregatedQuote(
            best_bid=best_bid_quote.bid,
            best_ask=best_ask_quote.ask,
            best_bid_dealer=best_bid_quote.dealer_id,
            best_ask_dealer=best_ask_quote.dealer_id,
            total_bid_size=total_bid,
            total_ask_size=total_ask,
            dealer_quotes=quotes,
            timestamp_ns=timestamp_ns,
            num_active_dealers=num_active,
        )

    def attempt_execution(
        self,
        is_buy: bool,
        size_usd: float,
        quote: AggregatedQuote,
        current_mid: float,
        symbol: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Attempt to execute against dealer quotes.

        Simulates realistic execution with:
        1. Dealer selection (best price first)
        2. Size availability check
        3. Latency simulation
        4. Last-look decision (adverse selection detection)
        5. Optional price improvement

        Args:
            is_buy: True for buy, False for sell
            size_usd: Order size in USD
            quote: Current aggregated quote
            current_mid: Current mid price (for last-look check)
            symbol: Optional symbol for pip size determination

        Returns:
            ExecutionResult with fill details or rejection reason
        """
        self._stats.total_attempts += 1

        # Check if any dealers are active
        if quote.num_active_dealers == 0:
            return ExecutionResult(
                filled=False,
                reject_reason=RejectReason.MARKET_CLOSED,
            )

        # Determine pip size
        is_jpy = symbol is not None and "JPY" in symbol.upper()
        pip_size = 0.01 if is_jpy else 0.0001

        # Sort dealers by price (best first)
        if is_buy:
            # For buy: sort by ask (lowest first)
            sorted_quotes = sorted(quote.dealer_quotes, key=lambda q: q.ask)
            reference_price = quote.best_ask
        else:
            # For sell: sort by bid (highest first)
            sorted_quotes = sorted(quote.dealer_quotes, key=lambda q: -q.bid)
            reference_price = quote.best_bid

        remaining_size = size_usd
        total_fill_qty = 0.0
        weighted_fill_price = 0.0
        total_latency_ns = 0
        last_dealer_id: Optional[str] = None

        for dealer_quote in sorted_quotes:
            if remaining_size <= 0:
                break

            dealer = self._dealer_map.get(dealer_quote.dealer_id)
            if dealer is None:
                continue

            # Check size availability
            available = dealer_quote.ask_size_usd if is_buy else dealer_quote.bid_size_usd

            if available < self.config.min_quote_size_usd:
                continue

            # Determine fill size for this dealer
            fill_size = min(remaining_size, available)

            # Simulate latency with variance
            latency_mult = 1.0 + self._rng.normal(0, self.config.latency_variance)
            latency_ms = max(1.0, dealer.latency_ms * latency_mult)
            latency_ns = int(latency_ms * 1_000_000)

            # Last-look check
            if self.config.last_look_enabled:
                quote_mid = dealer_quote.mid
                price_move = current_mid - quote_mid
                price_move_pips = abs(price_move) / pip_size

                # Adverse selection check
                is_adverse = (is_buy and current_mid > quote_mid) or (
                    not is_buy and current_mid < quote_mid
                )

                # Higher rejection probability for adverse moves
                reject_prob = dealer.base_reject_prob
                if is_adverse and price_move_pips > dealer.adverse_threshold_pips:
                    # Scale rejection probability with price move
                    adverse_factor = min(
                        0.9, 0.5 + (price_move_pips - dealer.adverse_threshold_pips) * 0.1
                    )
                    reject_prob = max(reject_prob, adverse_factor)

                    # Track adverse selection for this dealer
                    self._adverse_selection_memory[dealer.dealer_id] = time.time() * 1000

                # Latency arbitrage protection
                elapsed_ms = (time.time_ns() - dealer_quote.timestamp_ns) / 1_000_000
                if elapsed_ms > dealer.last_look_window_ms * 0.8:
                    reject_prob = min(0.95, reject_prob + 0.3)

                # Apply rejection
                if self._rng.random() < reject_prob:
                    self._stats.last_look_rejections += 1
                    if is_adverse:
                        continue  # Try next dealer
                    else:
                        # Random rejection
                        if self._rng.random() < 0.5:
                            continue

            # Determine fill price
            base_fill_price = dealer_quote.ask if is_buy else dealer_quote.bid

            # Optional price improvement
            price_improvement = 0.0
            if self.config.enable_price_improvement:
                if self._rng.random() < self.config.price_improvement_prob:
                    max_improvement = self.config.max_price_improvement_pips * pip_size
                    improvement = self._rng.uniform(0, max_improvement)
                    if is_buy:
                        base_fill_price -= improvement
                        price_improvement = improvement
                    else:
                        base_fill_price += improvement
                        price_improvement = improvement
                    self._stats.price_improvements += 1

            # Execute fill
            fill_price = base_fill_price
            weighted_fill_price += fill_price * fill_size
            total_fill_qty += fill_size
            remaining_size -= fill_size
            total_latency_ns += latency_ns
            last_dealer_id = dealer.dealer_id

            if not self.config.enable_partial_fills:
                break  # Single fill only

        # Calculate result
        if total_fill_qty > 0:
            avg_fill_price = weighted_fill_price / total_fill_qty
            slippage = abs(avg_fill_price - reference_price) / pip_size

            # Update statistics
            self._stats.total_fills += 1
            self._stats.total_volume_usd += total_fill_qty

            # Update average slippage (exponential moving average)
            alpha = 0.05
            self._stats.avg_slippage_pips = (
                alpha * slippage + (1 - alpha) * self._stats.avg_slippage_pips
            )
            self._stats.avg_latency_ms = (
                alpha * (total_latency_ns / 1_000_000) + (1 - alpha) * self._stats.avg_latency_ms
            )

            partial = remaining_size > 0
            if partial:
                self._stats.partial_fills += 1

            result = ExecutionResult(
                filled=True,
                fill_price=avg_fill_price,
                fill_qty=total_fill_qty,
                dealer_id=last_dealer_id,
                latency_ns=total_latency_ns,
                slippage_pips=slippage,
                last_look_passed=True,
                price_improvement=price_improvement,
                partial_fill=partial,
                remaining_qty=remaining_size,
            )
            self._execution_history.append(result)
            return result

        # All dealers rejected
        self._stats.total_rejections += 1
        reason = RejectReason.DEALER_DISCRETION

        # Track rejection reason
        self._stats.rejection_by_reason[reason.value] = (
            self._stats.rejection_by_reason.get(reason.value, 0) + 1
        )

        return ExecutionResult(
            filled=False,
            reject_reason=reason,
            latency_ns=total_latency_ns,
            remaining_qty=size_usd,
        )

    def simulate_quote_flicker(
        self,
        symbol: str,
        base_mid: float,
        duration_ms: int,
        tick_interval_ms: int = 10,
    ) -> Iterator[AggregatedQuote]:
        """
        Simulate quote flickering over a time period.

        Generates a sequence of quotes with realistic update patterns
        showing how dealer quotes change rapidly.

        Args:
            symbol: Currency pair
            base_mid: Base mid price
            duration_ms: Duration to simulate in milliseconds
            tick_interval_ms: Interval between ticks

        Yields:
            Sequence of AggregatedQuote objects
        """
        is_jpy = "JPY" in symbol.upper()
        pip_size = 0.01 if is_jpy else 0.0001

        # Volatility for price drift
        tick_vol = 0.1 * pip_size  # 0.1 pip per tick

        current_mid = base_mid
        elapsed_ms = 0

        while elapsed_ms < duration_ms:
            # Random walk for mid price
            current_mid += self._rng.normal(0, tick_vol)

            # Session factor (assume normal hours)
            session_factor = 1.0 + self._rng.uniform(-0.1, 0.1)

            quote = self.get_aggregated_quote(
                symbol=symbol,
                mid_price=current_mid,
                session_factor=session_factor,
            )

            yield quote
            elapsed_ms += tick_interval_ms

    def get_dealer(self, dealer_id: str) -> Optional[DealerProfile]:
        """
        Get dealer profile by ID.

        Args:
            dealer_id: Dealer identifier

        Returns:
            DealerProfile or None if not found
        """
        return self._dealer_map.get(dealer_id)

    def get_all_dealers(self) -> List[DealerProfile]:
        """
        Get all dealer profiles.

        Returns:
            List of all dealer profiles
        """
        return self._dealers.copy()

    def get_stats(self) -> ExecutionStats:
        """
        Get execution statistics.

        Returns:
            Current execution statistics
        """
        return self._stats

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._stats = ExecutionStats()

    def reset_state(self) -> None:
        """Reset all state including quotes and history."""
        self._current_quotes.clear()
        self._execution_history.clear()
        self._adverse_selection_memory.clear()
        self._quote_sequence = 0
        self.reset_stats()

    def get_recent_fill_rate(self, window: int = 100) -> float:
        """
        Get recent fill rate.

        Args:
            window: Number of recent executions to consider

        Returns:
            Fill rate as percentage
        """
        recent = list(self._execution_history)[-window:]
        if not recent:
            return 0.0

        fills = sum(1 for r in recent if r.filled)
        return (fills / len(recent)) * 100.0

    def get_recent_slippage(self, window: int = 100) -> float:
        """
        Get recent average slippage.

        Args:
            window: Number of recent executions to consider

        Returns:
            Average slippage in pips
        """
        recent = [r for r in list(self._execution_history)[-window:] if r.filled]
        if not recent:
            return 0.0

        return sum(r.slippage_pips for r in recent) / len(recent)


# =============================================================================
# Factory Functions
# =============================================================================


def create_forex_dealer_simulator(
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    profile: str = "retail",
) -> ForexDealerSimulator:
    """
    Create forex dealer simulator from configuration.

    Args:
        config: Configuration dictionary
        seed: Random seed for reproducibility
        profile: Configuration profile ("retail", "institutional", "conservative")

    Returns:
        Configured ForexDealerSimulator
    """
    # Profile defaults
    profiles = {
        "retail": {
            "num_dealers": 5,
            "base_spread_pips": 1.2,
            "last_look_enabled": True,
            "size_impact_threshold_usd": 500_000,
        },
        "institutional": {
            "num_dealers": 10,
            "base_spread_pips": 0.3,
            "last_look_enabled": True,
            "size_impact_threshold_usd": 5_000_000,
            "price_improvement_prob": 0.25,
        },
        "conservative": {
            "num_dealers": 3,
            "base_spread_pips": 2.0,
            "last_look_enabled": True,
            "size_impact_threshold_usd": 250_000,
            "max_slippage_pips": 15.0,
        },
    }

    # Merge profile with custom config
    profile_config = profiles.get(profile, profiles["retail"])
    merged_config = {**profile_config, **(config or {})}

    dealer_config = ForexDealerConfig(**merged_config)

    return ForexDealerSimulator(config=dealer_config, seed=seed)


def create_default_dealer_pool() -> List[DealerProfile]:
    """
    Create a default dealer pool for testing.

    Returns:
        List of default dealer profiles
    """
    return [
        DealerProfile(
            dealer_id="tier1_primary",
            tier=DealerTier.TIER1,
            spread_factor=0.7,
            max_size_usd=15_000_000,
            last_look_window_ms=100,
            base_reject_prob=0.02,
            adverse_threshold_pips=0.15,
            latency_ms=20,
            is_primary=True,
        ),
        DealerProfile(
            dealer_id="tier2_dealer_a",
            tier=DealerTier.TIER2,
            spread_factor=1.0,
            max_size_usd=5_000_000,
            last_look_window_ms=200,
            base_reject_prob=0.05,
            adverse_threshold_pips=0.3,
            latency_ms=50,
        ),
        DealerProfile(
            dealer_id="tier2_dealer_b",
            tier=DealerTier.TIER2,
            spread_factor=1.1,
            max_size_usd=4_000_000,
            last_look_window_ms=220,
            base_reject_prob=0.06,
            adverse_threshold_pips=0.35,
            latency_ms=60,
        ),
        DealerProfile(
            dealer_id="tier3_aggregator",
            tier=DealerTier.TIER3,
            spread_factor=1.4,
            max_size_usd=2_000_000,
            last_look_window_ms=300,
            base_reject_prob=0.10,
            adverse_threshold_pips=0.5,
            latency_ms=100,
        ),
    ]


# =============================================================================
# Integration Helpers
# =============================================================================


def combine_with_parametric_slippage(
    parametric_slippage_pips: float,
    execution_result: ExecutionResult,
    weight_execution: float = 0.3,
) -> float:
    """
    Combine parametric slippage estimate with actual execution slippage.

    Useful for blending model-based expectations with simulation results.

    Args:
        parametric_slippage_pips: Slippage from ForexParametricSlippageProvider
        execution_result: Result from ForexDealerSimulator
        weight_execution: Weight given to execution result (0-1)

    Returns:
        Combined slippage estimate in pips
    """
    if not execution_result.filled:
        # Use parametric estimate with penalty for rejection
        return parametric_slippage_pips * 1.5

    # Weighted combination
    return (
        weight_execution * execution_result.slippage_pips
        + (1 - weight_execution) * parametric_slippage_pips
    )


def estimate_rejection_probability(
    size_usd: float,
    session_factor: float,
    volatility_regime: str = "normal",
) -> float:
    """
    Estimate rejection probability for planning.

    Provides a rough estimate of rejection probability based on
    order characteristics for pre-trade analysis.

    Args:
        size_usd: Order size in USD
        session_factor: Session liquidity factor
        volatility_regime: Current volatility regime

    Returns:
        Estimated rejection probability (0-1)
    """
    # Base rejection probability
    base_prob = 0.05

    # Size impact
    if size_usd > 5_000_000:
        base_prob += 0.10 * (size_usd / 10_000_000)

    # Session impact (lower liquidity = higher rejection)
    if session_factor < 0.7:
        base_prob += 0.15
    elif session_factor < 1.0:
        base_prob += 0.05

    # Volatility impact
    vol_mult = {
        "low": 0.8,
        "normal": 1.0,
        "high": 1.5,
        "extreme": 2.5,
    }.get(volatility_regime, 1.0)

    base_prob *= vol_mult

    return min(0.95, max(0.01, base_prob))
