# -*- coding: utf-8 -*-
"""
services/forex_requote.py
Forex Requote Flow Simulation

Phase 11: Forex Realism Enhancement (2025-11-30)

This module simulates the requote flow in forex OTC markets.
Requotes occur when a dealer cannot fill at the requested price
and offers a new (usually worse) price.

Key Features:
1. Requote probability modeling based on market conditions
2. Requote price calculation with spread widening
3. Multi-requote scenarios (up to N requotes)
4. Client acceptance/rejection modeling
5. Statistics tracking for execution quality analysis

Requote Flow:
1. Client sends order at displayed price
2. Dealer receives order (after latency)
3. If price moved unfavorably for dealer → REQUOTE
4. Client receives new price
5. Client accepts or rejects requote
6. If accepted, fill at new price (with additional latency)

When Requotes Occur:
- Market volatility (prices moving fast)
- Large order sizes (liquidity constraints)
- News events (spread widening)
- Last-look period price movement
- Latency arbitrage protection
- Dealer inventory limits reached

Requote Probability Model:
    P(requote) = base_prob
                 × vol_factor          # Volatility effect
                 × size_factor         # Order size effect
                 × session_factor      # Session liquidity effect
                 × spread_factor       # Current spread width effect
                 × movement_factor     # Price movement during latency

References:
- Oomen (2017): "Last Look" - Journal of Financial Markets
- Chaboud et al. (2014): "Rise of the Machines: Algorithmic Trading in FX"
- King, Osler, Rime (2012): "Foreign Exchange Market Structure"
- BIS (2019): "FX Execution Algorithms and Market Functioning"

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

# Base requote probability by client tier
BASE_REQUOTE_PROBS: Dict[str, float] = {
    "retail": 0.08,        # 8% base for retail clients
    "professional": 0.04,  # 4% for professional
    "institutional": 0.02, # 2% for institutional
    "prime": 0.01,        # 1% for prime brokerage
}

# Requote probability multipliers by session
SESSION_REQUOTE_FACTORS: Dict[str, float] = {
    "sydney": 1.3,
    "tokyo": 1.2,
    "london": 0.9,          # Best liquidity
    "new_york": 0.95,
    "london_ny_overlap": 0.8,  # Best execution
    "tokyo_london_overlap": 1.0,
    "off_hours": 1.8,
    "weekend": 10.0,        # Very high if trading allowed
}

# Size thresholds for requote probability (USD notional)
SIZE_THRESHOLDS = {
    "small": 50_000,        # < 50k: low impact
    "medium": 500_000,      # 50k - 500k: moderate impact
    "large": 2_000_000,     # 500k - 2M: significant impact
    "institutional": 10_000_000,  # > 10M: very high impact
}

# Requote price widening factors
REQUOTE_WIDENING: Dict[str, float] = {
    "mild": 0.3,      # 0.3 pips widening
    "moderate": 0.8,  # 0.8 pips widening
    "severe": 2.0,    # 2.0 pips widening
    "extreme": 5.0,   # 5.0 pips widening during events
}

# Maximum requotes before rejection
MAX_REQUOTES_BY_TIER: Dict[str, int] = {
    "retail": 3,
    "professional": 2,
    "institutional": 2,
    "prime": 1,
}


# =============================================================================
# Enums
# =============================================================================

class RequoteReason(str, Enum):
    """Reason for requote."""
    PRICE_MOVED = "price_moved"
    VOLATILITY = "volatility"
    SIZE_EXCEEDED = "size_exceeded"
    LIQUIDITY_LOW = "liquidity_low"
    NEWS_EVENT = "news_event"
    INVENTORY_LIMIT = "inventory_limit"
    LATENCY_ARBITRAGE = "latency_arbitrage"
    SPREAD_WIDENED = "spread_widened"


class RequoteOutcome(str, Enum):
    """Outcome of requote flow."""
    FILLED_AT_ORIGINAL = "filled_at_original"
    FILLED_AT_REQUOTE = "filled_at_requote"
    CLIENT_REJECTED = "client_rejected"
    DEALER_REJECTED = "dealer_rejected"
    EXPIRED = "expired"
    MAX_REQUOTES_REACHED = "max_requotes_reached"


class ClientBehavior(str, Enum):
    """Client behavior profile for requote acceptance."""
    AGGRESSIVE = "aggressive"    # Low acceptance, seeks best price
    NEUTRAL = "neutral"         # Moderate acceptance
    PASSIVE = "passive"         # High acceptance, prioritizes fill
    ALGORITHMIC = "algorithmic" # Rule-based acceptance


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RequoteEvent:
    """
    Single requote event in the flow.

    Attributes:
        timestamp_ns: When requote was issued
        original_price: Price client requested
        requote_price: New price offered by dealer
        price_diff_pips: Difference in pips
        reason: Why requote was issued
        dealer_id: Dealer issuing requote
        latency_ns: Latency from request to requote
        accepted: Whether client accepted
        sequence_num: Requote number in sequence (1, 2, 3...)
    """
    timestamp_ns: int
    original_price: float
    requote_price: float
    price_diff_pips: float
    reason: RequoteReason
    dealer_id: str = "dealer_0"
    latency_ns: int = 0
    accepted: bool = False
    sequence_num: int = 1

    @property
    def latency_ms(self) -> float:
        """Get latency in milliseconds."""
        return self.latency_ns / 1_000_000.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp_ns": self.timestamp_ns,
            "original_price": self.original_price,
            "requote_price": self.requote_price,
            "price_diff_pips": self.price_diff_pips,
            "reason": self.reason.value,
            "dealer_id": self.dealer_id,
            "latency_ms": self.latency_ms,
            "accepted": self.accepted,
            "sequence_num": self.sequence_num,
        }


@dataclass
class RequoteFlowResult:
    """
    Complete result of a requote flow.

    Attributes:
        outcome: Final outcome
        fill_price: Final fill price (if filled)
        fill_timestamp_ns: Fill timestamp
        requotes: List of requote events
        total_requotes: Number of requotes issued
        total_slippage_pips: Total slippage from original
        total_latency_ns: Total time from order to fill/rejection
        original_price: Original requested price
        was_requoted: Whether any requotes occurred
    """
    outcome: RequoteOutcome
    fill_price: Optional[float] = None
    fill_timestamp_ns: int = 0
    requotes: List[RequoteEvent] = field(default_factory=list)
    total_requotes: int = 0
    total_slippage_pips: float = 0.0
    total_latency_ns: int = 0
    original_price: float = 0.0
    was_requoted: bool = False

    @property
    def was_filled(self) -> bool:
        """Check if order was filled."""
        return self.outcome in {
            RequoteOutcome.FILLED_AT_ORIGINAL,
            RequoteOutcome.FILLED_AT_REQUOTE,
        }

    @property
    def total_latency_ms(self) -> float:
        """Get total latency in milliseconds."""
        return self.total_latency_ns / 1_000_000.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "outcome": self.outcome.value,
            "fill_price": self.fill_price,
            "fill_timestamp_ns": self.fill_timestamp_ns,
            "total_requotes": self.total_requotes,
            "total_slippage_pips": self.total_slippage_pips,
            "total_latency_ms": self.total_latency_ms,
            "original_price": self.original_price,
            "was_requoted": self.was_requoted,
            "requotes": [r.to_dict() for r in self.requotes],
        }


@dataclass
class RequoteConfig:
    """
    Configuration for requote simulation.

    Attributes:
        client_tier: Client tier for base probability
        client_behavior: Client acceptance behavior
        base_requote_prob: Override base probability
        max_requotes: Maximum requotes before rejection
        max_total_slippage_pips: Max acceptable slippage
        requote_timeout_ms: Time limit for accepting requote
        use_volatility_scaling: Scale probability by volatility
        use_size_scaling: Scale probability by order size
        acceptance_threshold_pips: Max price diff to auto-accept
        seed: Random seed
    """
    client_tier: str = "retail"
    client_behavior: ClientBehavior = ClientBehavior.NEUTRAL
    base_requote_prob: Optional[float] = None
    max_requotes: Optional[int] = None
    max_total_slippage_pips: float = 5.0
    requote_timeout_ms: float = 5000.0  # 5 seconds
    use_volatility_scaling: bool = True
    use_size_scaling: bool = True
    acceptance_threshold_pips: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Set defaults based on tier."""
        if self.base_requote_prob is None:
            self.base_requote_prob = BASE_REQUOTE_PROBS.get(
                self.client_tier, 0.05
            )
        if self.max_requotes is None:
            self.max_requotes = MAX_REQUOTES_BY_TIER.get(
                self.client_tier, 3
            )


@dataclass
class MarketSnapshot:
    """
    Market state snapshot for requote decision.

    Attributes:
        timestamp_ns: Snapshot time
        bid: Current bid
        ask: Current ask
        spread_pips: Current spread
        volatility: Current volatility (daily %)
        session: Trading session
        is_news_event: Whether news event is active
        tick_rate: Recent tick arrival rate
    """
    timestamp_ns: int
    bid: float
    ask: float
    spread_pips: float
    volatility: float = 0.01  # 1% daily vol default
    session: str = "london"
    is_news_event: bool = False
    tick_rate: float = 3.0  # Ticks per second

    @property
    def mid(self) -> float:
        """Get mid price."""
        return (self.bid + self.ask) / 2.0


# =============================================================================
# Requote Probability Model
# =============================================================================

class RequoteProbabilityModel:
    """
    Models the probability of requotes based on market conditions.

    Combines multiple factors:
    - Base probability by client tier
    - Volatility scaling
    - Order size scaling
    - Session liquidity scaling
    - Price movement during latency
    - Spread width scaling

    References:
        - Oomen (2017): "Last Look" in FX markets
    """

    def __init__(self, config: RequoteConfig) -> None:
        """Initialize probability model."""
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def compute_requote_probability(
        self,
        order_size_usd: float,
        market: MarketSnapshot,
        price_movement_pips: float = 0.0,
        is_adverse: bool = False,
    ) -> Tuple[float, RequoteReason]:
        """
        Compute probability of requote and primary reason.

        Args:
            order_size_usd: Order size in USD
            market: Current market snapshot
            price_movement_pips: Price movement since order submission
            is_adverse: Whether movement is adverse for dealer

        Returns:
            (probability, reason) tuple
        """
        # Start with base probability
        prob = self.config.base_requote_prob or 0.05
        reasons: List[Tuple[RequoteReason, float]] = []

        # Volatility factor
        if self.config.use_volatility_scaling:
            vol_factor = market.volatility / 0.01  # Normalized to 1% vol
            vol_factor = max(0.5, min(3.0, vol_factor))
            if vol_factor > 1.5:
                reasons.append((RequoteReason.VOLATILITY, vol_factor))
            prob *= vol_factor

        # Size factor
        if self.config.use_size_scaling:
            if order_size_usd > SIZE_THRESHOLDS["institutional"]:
                size_factor = 3.0
                reasons.append((RequoteReason.SIZE_EXCEEDED, size_factor))
            elif order_size_usd > SIZE_THRESHOLDS["large"]:
                size_factor = 2.0
                reasons.append((RequoteReason.SIZE_EXCEEDED, size_factor))
            elif order_size_usd > SIZE_THRESHOLDS["medium"]:
                size_factor = 1.3
            else:
                size_factor = 1.0
            prob *= size_factor

        # Session factor
        session_factor = SESSION_REQUOTE_FACTORS.get(market.session, 1.0)
        if session_factor > 1.3:
            reasons.append((RequoteReason.LIQUIDITY_LOW, session_factor))
        prob *= session_factor

        # Price movement factor (key for last-look)
        if abs(price_movement_pips) > 0.1:
            movement_factor = 1.0 + abs(price_movement_pips) * 0.5
            if is_adverse:
                movement_factor *= 2.0  # Double if adverse for dealer
            reasons.append((RequoteReason.PRICE_MOVED, movement_factor))
            prob *= movement_factor

        # Spread width factor
        normal_spread = 1.0  # 1 pip for majors
        spread_factor = market.spread_pips / normal_spread
        if spread_factor > 2.0:
            reasons.append((RequoteReason.SPREAD_WIDENED, spread_factor))
            prob *= min(2.0, spread_factor)

        # News event
        if market.is_news_event:
            news_factor = 3.0
            reasons.append((RequoteReason.NEWS_EVENT, news_factor))
            prob *= news_factor

        # Cap probability
        prob = min(0.95, max(0.0, prob))

        # Determine primary reason
        if not reasons:
            reason = RequoteReason.PRICE_MOVED
        else:
            reason = max(reasons, key=lambda x: x[1])[0]

        return (prob, reason)

    def should_requote(
        self,
        order_size_usd: float,
        market: MarketSnapshot,
        price_movement_pips: float = 0.0,
        is_adverse: bool = False,
    ) -> Tuple[bool, Optional[RequoteReason]]:
        """
        Determine if a requote should occur.

        Args:
            order_size_usd: Order size
            market: Market snapshot
            price_movement_pips: Price movement
            is_adverse: Adverse for dealer

        Returns:
            (should_requote, reason) tuple
        """
        prob, reason = self.compute_requote_probability(
            order_size_usd, market, price_movement_pips, is_adverse
        )

        should = self._rng.random() < prob
        return (should, reason if should else None)


# =============================================================================
# Client Acceptance Model
# =============================================================================

class ClientAcceptanceModel:
    """
    Models client behavior for requote acceptance.

    Different client types have different acceptance patterns:
    - Aggressive: Low acceptance, seeks best execution
    - Neutral: Moderate acceptance based on price diff
    - Passive: High acceptance, prioritizes getting filled
    - Algorithmic: Rule-based with strict thresholds
    """

    # Base acceptance rates by behavior type
    BASE_ACCEPTANCE: Dict[ClientBehavior, float] = {
        ClientBehavior.AGGRESSIVE: 0.40,
        ClientBehavior.NEUTRAL: 0.70,
        ClientBehavior.PASSIVE: 0.90,
        ClientBehavior.ALGORITHMIC: 0.50,
    }

    def __init__(
        self,
        behavior: ClientBehavior = ClientBehavior.NEUTRAL,
        acceptance_threshold_pips: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize acceptance model.

        Args:
            behavior: Client behavior type
            acceptance_threshold_pips: Max acceptable price diff
            seed: Random seed
        """
        self.behavior = behavior
        self.threshold = acceptance_threshold_pips
        self._rng = np.random.default_rng(seed)

    def will_accept_requote(
        self,
        price_diff_pips: float,
        requote_count: int = 1,
        urgency: float = 0.5,
    ) -> bool:
        """
        Determine if client will accept a requote.

        Args:
            price_diff_pips: Price difference in pips
            requote_count: Number of requotes so far
            urgency: Order urgency (0-1, higher = more likely to accept)

        Returns:
            True if client accepts
        """
        # Algorithmic clients use strict threshold
        if self.behavior == ClientBehavior.ALGORITHMIC:
            return abs(price_diff_pips) <= self.threshold

        # Base acceptance rate
        base_rate = self.BASE_ACCEPTANCE[self.behavior]

        # Adjust for price difference
        if abs(price_diff_pips) > self.threshold * 2:
            price_factor = 0.3  # Much worse price
        elif abs(price_diff_pips) > self.threshold:
            price_factor = 0.6  # Worse than threshold
        else:
            price_factor = 1.0  # Acceptable

        # Adjust for requote count (fatigue)
        fatigue_factor = 0.9 ** requote_count

        # Adjust for urgency
        urgency_factor = 0.7 + urgency * 0.6  # 0.7 to 1.3

        # Final probability
        accept_prob = base_rate * price_factor * fatigue_factor * urgency_factor
        accept_prob = min(0.98, max(0.05, accept_prob))

        return self._rng.random() < accept_prob


# =============================================================================
# Requote Flow Simulator
# =============================================================================

class RequoteFlowSimulator:
    """
    Simulates the complete requote flow.

    Handles the full requote lifecycle:
    1. Initial order submission
    2. Latency simulation
    3. Requote decision
    4. Requote price calculation
    5. Client acceptance/rejection
    6. Multi-requote scenarios
    7. Final fill or rejection

    Usage:
        config = RequoteConfig(client_tier="retail")
        simulator = RequoteFlowSimulator(config)

        result = simulator.simulate_requote_flow(
            is_buy=True,
            size_usd=100000,
            requested_price=1.0850,
            market=market_snapshot,
        )

        if result.was_filled:
            print(f"Filled at {result.fill_price}")
        else:
            print(f"Rejected: {result.outcome}")
    """

    def __init__(
        self,
        config: Optional[RequoteConfig] = None,
    ) -> None:
        """
        Initialize requote simulator.

        Args:
            config: Simulation configuration
        """
        self.config = config or RequoteConfig()
        self._rng = np.random.default_rng(self.config.seed)

        # Sub-models
        self._prob_model = RequoteProbabilityModel(self.config)
        self._accept_model = ClientAcceptanceModel(
            behavior=self.config.client_behavior,
            acceptance_threshold_pips=self.config.acceptance_threshold_pips,
            seed=self.config.seed,
        )

        # Statistics
        self._stats = {
            "total_orders": 0,
            "orders_requoted": 0,
            "total_requotes": 0,
            "requotes_accepted": 0,
            "requotes_rejected": 0,
            "filled_at_original": 0,
            "filled_at_requote": 0,
            "client_rejected": 0,
            "dealer_rejected": 0,
            "max_requotes_reached": 0,
            "total_slippage_pips": 0.0,
        }

        # History
        self._history: Deque[RequoteFlowResult] = deque(maxlen=1000)

    def simulate_requote_flow(
        self,
        is_buy: bool,
        size_usd: float,
        requested_price: float,
        market: MarketSnapshot,
        urgency: float = 0.5,
        order_timestamp_ns: Optional[int] = None,
    ) -> RequoteFlowResult:
        """
        Simulate complete requote flow for an order.

        Args:
            is_buy: True for buy order
            size_usd: Order size in USD
            requested_price: Price client requested
            market: Current market snapshot
            urgency: Order urgency (0-1)
            order_timestamp_ns: Order submission time

        Returns:
            RequoteFlowResult with complete flow details
        """
        self._stats["total_orders"] += 1

        if order_timestamp_ns is None:
            order_timestamp_ns = time.time_ns()

        # Pip value
        is_jpy = market.bid > 10  # Simple JPY detection
        pip_value = 0.01 if is_jpy else 0.0001

        current_timestamp_ns = order_timestamp_ns
        current_price = requested_price
        requotes: List[RequoteEvent] = []
        total_slippage = 0.0

        # Initial latency
        latency_ns = self._sample_latency_ns()
        current_timestamp_ns += latency_ns
        total_latency_ns = latency_ns

        # Simulate current market price (may have moved)
        market_price = self._simulate_price_movement(
            market, latency_ns, is_buy, pip_value
        )

        # Calculate price movement
        if is_buy:
            price_movement_pips = (market.ask - requested_price) / pip_value
            is_adverse = market.ask > requested_price
        else:
            price_movement_pips = (requested_price - market.bid) / pip_value
            is_adverse = market.bid < requested_price

        # Check if requote needed
        should_requote, reason = self._prob_model.should_requote(
            order_size_usd=size_usd,
            market=market,
            price_movement_pips=price_movement_pips,
            is_adverse=is_adverse,
        )

        if not should_requote:
            # Fill at original price
            self._stats["filled_at_original"] += 1
            result = RequoteFlowResult(
                outcome=RequoteOutcome.FILLED_AT_ORIGINAL,
                fill_price=requested_price,
                fill_timestamp_ns=current_timestamp_ns,
                original_price=requested_price,
                total_latency_ns=total_latency_ns,
                was_requoted=False,
            )
            self._history.append(result)
            return result

        # Requote flow
        self._stats["orders_requoted"] += 1
        requote_count = 0
        max_requotes = self.config.max_requotes or 3

        while requote_count < max_requotes:
            requote_count += 1
            self._stats["total_requotes"] += 1

            # Calculate requote price
            requote_price = self._calculate_requote_price(
                is_buy=is_buy,
                original_price=current_price,
                market=market,
                reason=reason or RequoteReason.PRICE_MOVED,
                pip_value=pip_value,
            )

            price_diff_pips = abs(requote_price - current_price) / pip_value

            # Record requote event
            requote_latency_ns = self._sample_latency_ns()
            requote_event = RequoteEvent(
                timestamp_ns=current_timestamp_ns,
                original_price=current_price,
                requote_price=requote_price,
                price_diff_pips=price_diff_pips,
                reason=reason or RequoteReason.PRICE_MOVED,
                dealer_id=f"dealer_{self._rng.integers(0, 5)}",
                latency_ns=requote_latency_ns,
                sequence_num=requote_count,
            )

            # Client decision
            client_accepts = self._accept_model.will_accept_requote(
                price_diff_pips=price_diff_pips,
                requote_count=requote_count,
                urgency=urgency,
            )

            requote_event.accepted = client_accepts
            requotes.append(requote_event)

            total_latency_ns += requote_latency_ns
            current_timestamp_ns += requote_latency_ns

            if client_accepts:
                # Client accepts requote
                self._stats["requotes_accepted"] += 1

                # Additional latency for acceptance processing
                accept_latency_ns = self._sample_latency_ns() // 2
                total_latency_ns += accept_latency_ns
                current_timestamp_ns += accept_latency_ns

                # Check if dealer still wants to fill
                # (market may have moved again)
                fill_prob = 0.9 - requote_count * 0.1  # Decreasing fill chance

                if self._rng.random() < fill_prob:
                    # Fill at requote price
                    total_slippage = abs(requote_price - requested_price) / pip_value
                    self._stats["filled_at_requote"] += 1
                    self._stats["total_slippage_pips"] += total_slippage

                    result = RequoteFlowResult(
                        outcome=RequoteOutcome.FILLED_AT_REQUOTE,
                        fill_price=requote_price,
                        fill_timestamp_ns=current_timestamp_ns,
                        requotes=requotes,
                        total_requotes=requote_count,
                        total_slippage_pips=total_slippage,
                        total_latency_ns=total_latency_ns,
                        original_price=requested_price,
                        was_requoted=True,
                    )
                    self._history.append(result)
                    return result

                # Dealer changed mind, issue another requote
                current_price = requote_price
                reason = RequoteReason.PRICE_MOVED
                continue

            else:
                # Client rejects requote
                self._stats["requotes_rejected"] += 1

                # Check max slippage
                if total_slippage + price_diff_pips > self.config.max_total_slippage_pips:
                    self._stats["client_rejected"] += 1
                    result = RequoteFlowResult(
                        outcome=RequoteOutcome.CLIENT_REJECTED,
                        requotes=requotes,
                        total_requotes=requote_count,
                        total_latency_ns=total_latency_ns,
                        original_price=requested_price,
                        was_requoted=True,
                    )
                    self._history.append(result)
                    return result

                # Client might try again with counter-offer
                # Simulate re-attempt with some probability
                retry_prob = 0.3 * (1 - requote_count / max_requotes)
                if self._rng.random() > retry_prob:
                    self._stats["client_rejected"] += 1
                    result = RequoteFlowResult(
                        outcome=RequoteOutcome.CLIENT_REJECTED,
                        requotes=requotes,
                        total_requotes=requote_count,
                        total_latency_ns=total_latency_ns,
                        original_price=requested_price,
                        was_requoted=True,
                    )
                    self._history.append(result)
                    return result

                # Update for next round
                current_price = (requested_price + requote_price) / 2  # Meet in middle
                reason = RequoteReason.PRICE_MOVED

        # Max requotes reached
        self._stats["max_requotes_reached"] += 1
        result = RequoteFlowResult(
            outcome=RequoteOutcome.MAX_REQUOTES_REACHED,
            requotes=requotes,
            total_requotes=requote_count,
            total_latency_ns=total_latency_ns,
            original_price=requested_price,
            was_requoted=True,
        )
        self._history.append(result)
        return result

    def _calculate_requote_price(
        self,
        is_buy: bool,
        original_price: float,
        market: MarketSnapshot,
        reason: RequoteReason,
        pip_value: float,
    ) -> float:
        """Calculate the requote price."""
        # Determine widening based on reason
        if reason == RequoteReason.NEWS_EVENT:
            widening_pips = REQUOTE_WIDENING["extreme"]
        elif reason == RequoteReason.VOLATILITY:
            widening_pips = REQUOTE_WIDENING["severe"]
        elif reason == RequoteReason.SIZE_EXCEEDED:
            widening_pips = REQUOTE_WIDENING["moderate"]
        else:
            widening_pips = REQUOTE_WIDENING["mild"]

        # Add some randomness
        widening_pips *= (0.8 + self._rng.random() * 0.4)

        # Apply widening in direction unfavorable to client
        if is_buy:
            # Buy requote = higher price
            requote_price = original_price + widening_pips * pip_value
            # Don't exceed current ask
            requote_price = min(requote_price, market.ask * 1.001)
        else:
            # Sell requote = lower price
            requote_price = original_price - widening_pips * pip_value
            # Don't go below current bid
            requote_price = max(requote_price, market.bid * 0.999)

        return requote_price

    def _simulate_price_movement(
        self,
        market: MarketSnapshot,
        latency_ns: int,
        is_buy: bool,
        pip_value: float,
    ) -> float:
        """Simulate price movement during latency."""
        latency_sec = latency_ns / 1e9

        # Price can move with volatility
        vol_per_sec = market.volatility / math.sqrt(252 * 24 * 3600)
        price_move = self._rng.normal(0, vol_per_sec * math.sqrt(latency_sec))

        if is_buy:
            return market.ask * (1 + price_move)
        return market.bid * (1 + price_move)

    def _sample_latency_ns(self) -> int:
        """Sample processing latency."""
        # Log-normal latency
        mean_ms = 50.0
        std_ms = 20.0
        latency_ms = self._rng.lognormal(
            math.log(mean_ms),
            std_ms / mean_ms,
        )
        latency_ms = max(5.0, min(500.0, latency_ms))
        return int(latency_ms * 1_000_000)

    def get_stats(self) -> Dict[str, Any]:
        """Get simulation statistics."""
        total = max(1, self._stats["total_orders"])
        requoted = max(1, self._stats["orders_requoted"])
        filled = self._stats["filled_at_original"] + self._stats["filled_at_requote"]

        return {
            **self._stats,
            "requote_rate": self._stats["orders_requoted"] / total * 100,
            "fill_rate": filled / total * 100,
            "requote_acceptance_rate": (
                self._stats["requotes_accepted"] /
                max(1, self._stats["total_requotes"]) * 100
            ),
            "avg_slippage_pips": (
                self._stats["total_slippage_pips"] /
                max(1, self._stats["filled_at_requote"])
            ),
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        for key in self._stats:
            if isinstance(self._stats[key], float):
                self._stats[key] = 0.0
            else:
                self._stats[key] = 0

    def get_recent_results(
        self,
        n: int = 10,
    ) -> List[RequoteFlowResult]:
        """Get recent simulation results."""
        return list(self._history)[-n:]


# =============================================================================
# Factory Functions
# =============================================================================

def create_requote_simulator(
    client_tier: str = "retail",
    behavior: str = "neutral",
    max_slippage_pips: float = 5.0,
    seed: Optional[int] = None,
) -> RequoteFlowSimulator:
    """
    Create a configured requote simulator.

    Args:
        client_tier: Client tier ("retail", "professional", "institutional")
        behavior: Client behavior ("aggressive", "neutral", "passive", "algorithmic")
        max_slippage_pips: Maximum acceptable slippage
        seed: Random seed

    Returns:
        Configured RequoteFlowSimulator
    """
    try:
        client_behavior = ClientBehavior(behavior)
    except ValueError:
        client_behavior = ClientBehavior.NEUTRAL

    config = RequoteConfig(
        client_tier=client_tier,
        client_behavior=client_behavior,
        max_total_slippage_pips=max_slippage_pips,
        seed=seed,
    )

    return RequoteFlowSimulator(config)


def simulate_requote_scenario(
    n_orders: int = 100,
    client_tier: str = "retail",
    session: str = "london",
    volatility: float = 0.01,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run a batch requote simulation scenario.

    Args:
        n_orders: Number of orders to simulate
        client_tier: Client tier
        session: Trading session
        volatility: Market volatility
        seed: Random seed

    Returns:
        Aggregated statistics
    """
    simulator = create_requote_simulator(client_tier=client_tier, seed=seed)
    rng = np.random.default_rng(seed)

    for _ in range(n_orders):
        market = MarketSnapshot(
            timestamp_ns=time.time_ns(),
            bid=1.0850,
            ask=1.0852,
            spread_pips=2.0,
            volatility=volatility,
            session=session,
            is_news_event=rng.random() < 0.05,
        )

        simulator.simulate_requote_flow(
            is_buy=rng.random() < 0.5,
            size_usd=rng.uniform(10_000, 500_000),
            requested_price=market.ask if rng.random() < 0.5 else market.bid,
            market=market,
            urgency=rng.uniform(0.3, 0.8),
        )

    return simulator.get_stats()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "RequoteReason",
    "RequoteOutcome",
    "ClientBehavior",
    # Data classes
    "RequoteEvent",
    "RequoteFlowResult",
    "RequoteConfig",
    "MarketSnapshot",
    # Classes
    "RequoteProbabilityModel",
    "ClientAcceptanceModel",
    "RequoteFlowSimulator",
    # Factory functions
    "create_requote_simulator",
    "simulate_requote_scenario",
    # Constants
    "BASE_REQUOTE_PROBS",
    "SESSION_REQUOTE_FACTORS",
]
