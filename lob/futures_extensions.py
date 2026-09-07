# -*- coding: utf-8 -*-
"""
LOB Futures Extensions for Crypto Perpetual Markets.

This module provides futures-specific extensions to the L3 LOB simulation:

1. Liquidation Order Stream - Real-time liquidation order injection
2. Liquidation Cascade Simulation - Cascading forced selling dynamics
3. Insurance Fund Dynamics - Insurance fund contribution/depletion tracking
4. ADL Queue Simulation - Auto-Deleveraging queue management
5. Funding-Adjusted Queue Dynamics - Queue behavior near funding times

These extensions are designed specifically for crypto perpetual futures
(Binance USDT-M, etc.) where:
- Positions can be liquidated at any time based on mark price
- Liquidation orders are market orders that must be filled
- Insurance fund covers socialized losses
- ADL mechanism triggers when insurance fund is depleted

Stage 5A of Futures Integration (v1.0)

References:
    - Binance Futures Liquidation: https://www.binance.com/en/support/faq/360033525271
    - Binance ADL: https://www.binance.com/en/support/faq/360033525711
    - Zhao et al. (2020): "Liquidation Cascade Effects in Crypto Markets"
    - Cont et al. (2014): "The Price Impact of Order Book Events"
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

from lob.data_structures import (
    Side,
    OrderType,
    LimitOrder,
    PriceLevel,
    OrderBook,
    Fill as LOBFill,
    Trade,
)
from lob.matching_engine import MatchingEngine, MatchResult


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Standard funding times (UTC hours)
FUNDING_TIMES_UTC = [0, 8, 16]

# Funding period in milliseconds (8 hours)
FUNDING_PERIOD_MS = 8 * 3600 * 1000

# Default cascade parameters
DEFAULT_CASCADE_DECAY = 0.7  # Each wave is 70% of previous
DEFAULT_MAX_CASCADE_WAVES = 5
DEFAULT_INSURANCE_FUND_INITIAL = Decimal("1_000_000_000")  # $1B

# ADL ranking thresholds
ADL_RANK_THRESHOLDS = [0.2, 0.4, 0.6, 0.8]  # Score thresholds for ranks 1-5


# =============================================================================
# Data Types
# =============================================================================


class LiquidationType(str, Enum):
    """Type of liquidation."""

    FULL = "full"
    PARTIAL = "partial"
    BANKRUPTCY = "bankruptcy"  # Position goes below 0


class ADLRank(int, Enum):
    """Auto-Deleveraging rank (1-5, 5 = highest priority)."""

    RANK_1 = 1  # Lowest risk
    RANK_2 = 2
    RANK_3 = 3
    RANK_4 = 4
    RANK_5 = 5  # Highest risk - ADL'd first


class CascadePhase(str, Enum):
    """Phase of liquidation cascade."""

    INITIAL = "initial"
    PROPAGATING = "propagating"
    DAMPENING = "dampening"
    COMPLETE = "complete"


@dataclass(frozen=True)
class LiquidationOrderInfo:
    """
    Information about a liquidation order to be injected into the order book.

    Attributes:
        symbol: Contract symbol
        side: Order side (BUY to close short, SELL to close long)
        qty: Quantity to liquidate
        bankruptcy_price: Price at which position becomes bankrupt
        mark_price: Mark price at liquidation time
        timestamp_ms: Liquidation timestamp
        position_entry_price: Original entry price of position
        position_leverage: Position leverage
        is_adl: True if this is an ADL order (not regular liquidation)
        source_account_id: Optional identifier for source account
    """

    symbol: str
    side: str  # "BUY" or "SELL"
    qty: Decimal
    bankruptcy_price: Decimal
    mark_price: Decimal
    timestamp_ms: int
    position_entry_price: Decimal = Decimal("0")
    position_leverage: int = 1
    is_adl: bool = False
    source_account_id: Optional[str] = None

    @property
    def notional(self) -> Decimal:
        """Notional value at bankruptcy price."""
        return self.qty * self.bankruptcy_price

    @property
    def is_long_liquidation(self) -> bool:
        """True if liquidating a long position (selling)."""
        return self.side.upper() == "SELL"

    @property
    def is_short_liquidation(self) -> bool:
        """True if liquidating a short position (buying)."""
        return self.side.upper() == "BUY"


@dataclass
class LiquidationFillResult:
    """
    Result of executing a liquidation order.

    Attributes:
        order_info: Original liquidation order
        fill_price: Actual fill price
        fill_qty: Quantity filled
        slippage_bps: Slippage from bankruptcy price
        insurance_fund_impact: Amount to/from insurance fund
        caused_adl: True if fill depleted insurance fund and triggered ADL
        cascade_triggered: True if fill triggered additional liquidations
    """

    order_info: LiquidationOrderInfo
    fill_price: Decimal
    fill_qty: Decimal
    slippage_bps: float
    insurance_fund_impact: Decimal
    caused_adl: bool = False
    cascade_triggered: bool = False
    timestamp_ms: int = 0

    @property
    def is_filled(self) -> bool:
        """True if order was filled."""
        return self.fill_qty > 0

    @property
    def fill_notional(self) -> Decimal:
        """Notional value at fill price."""
        return self.fill_qty * self.fill_price


@dataclass
class CascadeWave:
    """
    Represents one wave of a liquidation cascade.

    Attributes:
        wave_number: Wave index (0 = initial)
        liquidation_count: Number of liquidations in this wave
        total_qty: Total quantity liquidated
        price_impact_bps: Price impact from this wave
        timestamp_ms: Wave timestamp
    """

    wave_number: int
    liquidation_count: int
    total_qty: Decimal
    price_impact_bps: float
    timestamp_ms: int
    liquidations: List[LiquidationOrderInfo] = field(default_factory=list)


@dataclass
class CascadeResult:
    """
    Result of a complete liquidation cascade simulation.

    Attributes:
        initial_event: The triggering liquidation
        waves: List of cascade waves
        total_liquidations: Total number of positions liquidated
        total_qty_liquidated: Total quantity liquidated across all waves
        total_price_impact_bps: Cumulative price impact
        insurance_fund_depleted: True if insurance fund hit zero
        adl_triggered: True if ADL was triggered
        final_price: Price after cascade completion
        duration_ms: Total cascade duration
    """

    initial_event: LiquidationOrderInfo
    waves: List[CascadeWave] = field(default_factory=list)
    total_liquidations: int = 0
    total_qty_liquidated: Decimal = Decimal("0")
    total_price_impact_bps: float = 0.0
    insurance_fund_depleted: bool = False
    adl_triggered: bool = False
    final_price: Decimal = Decimal("0")
    duration_ms: int = 0

    @property
    def cascade_depth(self) -> int:
        """Number of waves in cascade."""
        return len(self.waves)

    @property
    def phase(self) -> CascadePhase:
        """Current cascade phase."""
        if not self.waves:
            return CascadePhase.INITIAL
        if len(self.waves) < 3:
            return CascadePhase.PROPAGATING
        if self.waves[-1].liquidation_count < self.waves[-2].liquidation_count:
            return CascadePhase.DAMPENING
        return CascadePhase.COMPLETE


class ADLQueueEntry(NamedTuple):
    """Entry in the ADL queue."""

    account_id: str
    symbol: str
    side: str  # Position side ("LONG" or "SHORT")
    qty: Decimal
    pnl_percentile: float
    leverage_percentile: float
    rank: int
    estimated_adl_qty: Decimal


@dataclass
class InsuranceFundState:
    """
    State of the insurance fund.

    Attributes:
        balance: Current fund balance
        high_water_mark: Maximum historical balance
        last_contribution_ms: Timestamp of last contribution
        total_contributions: Cumulative contributions
        total_payouts: Cumulative payouts
        is_depleted: True if balance is zero or negative
    """

    balance: Decimal
    high_water_mark: Decimal
    last_contribution_ms: int = 0
    total_contributions: Decimal = Decimal("0")
    total_payouts: Decimal = Decimal("0")

    @property
    def is_depleted(self) -> bool:
        """Check if fund is depleted."""
        return self.balance <= 0

    @property
    def utilization_pct(self) -> float:
        """Current utilization as percentage of high water mark."""
        if self.high_water_mark <= 0:
            return 100.0
        return float((self.high_water_mark - self.balance) / self.high_water_mark) * 100


@dataclass
class FundingPeriodState:
    """
    State tracking for funding period dynamics.

    Near funding time, queue behavior changes:
    - More aggressive position changes
    - Wider spreads
    - Higher impact
    """

    current_funding_rate: Decimal
    next_funding_time_ms: int
    time_to_funding_ms: int
    is_funding_window: bool  # Within 15 min of funding
    spread_multiplier: float  # Spread widening near funding
    impact_multiplier: float  # Impact increase near funding


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class LiquidationStreamProvider(Protocol):
    """Protocol for liquidation order stream providers."""

    def get_liquidations_up_to(
        self,
        timestamp_ms: int,
    ) -> Iterator[LiquidationOrderInfo]:
        """
        Get liquidation orders up to given timestamp.

        Args:
            timestamp_ms: Timestamp to fetch up to

        Yields:
            LiquidationOrderInfo for each liquidation
        """
        ...

    def peek_next_liquidation(self) -> Optional[LiquidationOrderInfo]:
        """Peek at next liquidation without consuming it."""
        ...


@runtime_checkable
class PositionTracker(Protocol):
    """Protocol for tracking open positions (for cascade simulation)."""

    def get_positions_at_risk(
        self,
        symbol: str,
        current_price: Decimal,
        price_move_bps: float,
    ) -> List[Dict[str, Any]]:
        """
        Get positions that would be liquidated at a given price move.

        Args:
            symbol: Contract symbol
            current_price: Current mark price
            price_move_bps: Additional price move in bps

        Returns:
            List of position dicts with liquidation info
        """
        ...


# =============================================================================
# Liquidation Order Stream
# =============================================================================


class LiquidationOrderStream:
    """
    Manages a stream of liquidation orders for injection into LOB.

    Can be backed by:
    - Historical data (for backtesting)
    - Real-time feed (for live simulation)
    - Generated data (for Monte Carlo)

    Features:
    - Time-ordered delivery
    - Buffering and batching
    - Statistics tracking

    Example:
        >>> stream = LiquidationOrderStream()
        >>> stream.add_historical_data(liquidation_events)
        >>>
        >>> for liq in stream.get_liquidations_up_to(current_ts):
        ...     fill = matching_engine.execute_market_order(liq)
    """

    def __init__(
        self,
        max_buffer_size: int = 10000,
    ):
        """
        Initialize liquidation order stream.

        Args:
            max_buffer_size: Maximum buffered events
        """
        self._buffer: Deque[LiquidationOrderInfo] = deque(maxlen=max_buffer_size)
        self._last_delivered_ts: int = 0
        self._total_delivered: int = 0
        self._total_qty_delivered: Decimal = Decimal("0")

    def add_event(self, event: LiquidationOrderInfo) -> None:
        """Add single liquidation event to stream."""
        self._buffer.append(event)

    def add_events(self, events: Sequence[LiquidationOrderInfo]) -> None:
        """Add multiple liquidation events to stream."""
        for event in events:
            self._buffer.append(event)

    def add_historical_data(
        self,
        data: List[Dict[str, Any]],
    ) -> None:
        """
        Load historical liquidation data.

        Args:
            data: List of dicts with liquidation fields
        """
        for d in data:
            event = LiquidationOrderInfo(
                symbol=str(d.get("symbol", "")),
                side=str(d.get("side", "SELL")),
                qty=Decimal(str(d.get("qty", "0"))),
                bankruptcy_price=Decimal(str(d.get("price", d.get("bankruptcy_price", "0")))),
                mark_price=Decimal(str(d.get("mark_price", d.get("price", "0")))),
                timestamp_ms=int(d.get("timestamp_ms", d.get("time", 0))),
                position_entry_price=Decimal(str(d.get("entry_price", "0"))),
                position_leverage=int(d.get("leverage", 1)),
                is_adl=bool(d.get("is_adl", False)),
            )
            self._buffer.append(event)

        # Sort by timestamp
        self._buffer = deque(
            sorted(self._buffer, key=lambda x: x.timestamp_ms),
            maxlen=self._buffer.maxlen,
        )

    def get_liquidations_up_to(
        self,
        timestamp_ms: int,
    ) -> Iterator[LiquidationOrderInfo]:
        """
        Get liquidation orders up to given timestamp.

        Yields events in timestamp order and removes them from buffer.
        """
        delivered = []

        for event in self._buffer:
            if event.timestamp_ms <= timestamp_ms:
                delivered.append(event)
                self._total_delivered += 1
                self._total_qty_delivered += event.qty
                self._last_delivered_ts = event.timestamp_ms
                yield event
            else:
                break

        # Remove delivered events
        for event in delivered:
            self._buffer.remove(event)

    def peek_next_liquidation(self) -> Optional[LiquidationOrderInfo]:
        """Peek at next liquidation without consuming it."""
        if self._buffer:
            return self._buffer[0]
        return None

    @property
    def pending_count(self) -> int:
        """Number of pending liquidations in buffer."""
        return len(self._buffer)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get stream statistics."""
        return {
            "total_delivered": self._total_delivered,
            "total_qty_delivered": str(self._total_qty_delivered),
            "pending_count": self.pending_count,
            "last_delivered_ts": self._last_delivered_ts,
        }

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()


# =============================================================================
# Liquidation Cascade Simulator
# =============================================================================


class LiquidationCascadeSimulator:
    """
    Simulates cascading liquidations.

    When price moves against leveraged positions:
    1. Positions hit liquidation price
    2. Liquidation orders placed (market orders)
    3. Orders fill, push price further
    4. More positions liquidated (cascade)
    5. Process repeats until dampening

    The cascade model uses:
    - Kyle (1985) price impact model
    - Exponential decay for wave attenuation
    - Insurance fund depletion tracking

    Example:
        >>> simulator = LiquidationCascadeSimulator(cascade_decay=0.7)
        >>>
        >>> # Initial liquidation event
        >>> initial = LiquidationOrderInfo(...)
        >>>
        >>> # Simulate cascade
        >>> result = simulator.simulate_cascade(
        ...     initial_liquidation=initial,
        ...     order_book=order_book,
        ...     matching_engine=engine,
        ...     price_impact_coef=0.1,
        ... )
        >>>
        >>> print(f"Cascade depth: {result.cascade_depth} waves")
        >>> print(f"Total liquidated: {result.total_qty_liquidated}")

    References:
        - Kyle (1985): "Continuous Auctions and Insider Trading"
        - Zhao et al. (2020): "Liquidation Cascade Effects"
    """

    def __init__(
        self,
        cascade_decay: float = DEFAULT_CASCADE_DECAY,
        max_waves: int = DEFAULT_MAX_CASCADE_WAVES,
        price_impact_coef: float = 0.1,
        min_cascade_qty: Decimal = Decimal("0.001"),
    ):
        """
        Initialize cascade simulator.

        Args:
            cascade_decay: Decay factor for each wave (0-1, lower = faster dampening)
            max_waves: Maximum cascade waves to simulate
            price_impact_coef: Kyle lambda coefficient for price impact
            min_cascade_qty: Minimum qty to continue cascade
        """
        if not 0 < cascade_decay <= 1:
            raise ValueError(f"cascade_decay must be in (0, 1], got {cascade_decay}")
        if max_waves < 1:
            raise ValueError(f"max_waves must be >= 1, got {max_waves}")
        if price_impact_coef <= 0:
            raise ValueError(f"price_impact_coef must be > 0, got {price_impact_coef}")

        self._cascade_decay = cascade_decay
        self._max_waves = max_waves
        self._price_impact_coef = price_impact_coef
        self._min_cascade_qty = min_cascade_qty

        # Statistics
        self._total_cascades: int = 0
        self._total_waves: int = 0
        self._deepest_cascade: int = 0

    def simulate_cascade(
        self,
        initial_liquidation: LiquidationOrderInfo,
        order_book: OrderBook,
        matching_engine: Optional[MatchingEngine] = None,
        position_tracker: Optional[PositionTracker] = None,
        volatility: float = 0.02,
        adv: float = 1_000_000_000,
    ) -> CascadeResult:
        """
        Simulate liquidation cascade from initial event.

        Args:
            initial_liquidation: Triggering liquidation
            order_book: Current order book state
            matching_engine: Optional matching engine for fills
            position_tracker: Optional tracker for at-risk positions
            volatility: Current volatility for impact calculation
            adv: Average daily volume

        Returns:
            CascadeResult with complete cascade analysis
        """
        result = CascadeResult(
            initial_event=initial_liquidation,
            waves=[],
            final_price=initial_liquidation.mark_price,
        )

        current_price = initial_liquidation.mark_price
        cumulative_qty = Decimal("0")
        start_ts = initial_liquidation.timestamp_ms

        # Wave 0: Initial liquidation
        initial_wave = CascadeWave(
            wave_number=0,
            liquidation_count=1,
            total_qty=initial_liquidation.qty,
            price_impact_bps=0.0,
            timestamp_ms=start_ts,
            liquidations=[initial_liquidation],
        )
        result.waves.append(initial_wave)
        cumulative_qty += initial_liquidation.qty

        # Subsequent waves
        wave_qty = initial_liquidation.qty

        for wave_num in range(1, self._max_waves + 1):
            # Apply decay
            wave_qty = wave_qty * Decimal(str(self._cascade_decay))

            # Stop if below minimum
            if wave_qty < self._min_cascade_qty:
                break

            # Calculate price impact from this wave
            # Using Almgren-Chriss style impact: Δp = λ × σ × √(Q/V)
            participation = float(wave_qty) / adv if adv > 0 else 0.0
            impact_bps = self._price_impact_coef * volatility * math.sqrt(participation) * 10000

            # Apply directional impact
            if initial_liquidation.is_long_liquidation:
                # Long liquidation = sell pressure = price down
                new_price = current_price * Decimal(str(1 - impact_bps / 10000))
            else:
                # Short liquidation = buy pressure = price up
                new_price = current_price * Decimal(str(1 + impact_bps / 10000))

            # Find additional liquidations at new price (if tracker available)
            new_liquidations = []
            if position_tracker:
                at_risk = position_tracker.get_positions_at_risk(
                    symbol=initial_liquidation.symbol,
                    current_price=current_price,
                    price_move_bps=(
                        impact_bps if initial_liquidation.is_long_liquidation else -impact_bps
                    ),
                )

                for pos in at_risk:
                    new_liquidations.append(
                        LiquidationOrderInfo(
                            symbol=initial_liquidation.symbol,
                            side="SELL" if pos.get("qty", 0) > 0 else "BUY",
                            qty=abs(Decimal(str(pos.get("qty", "0")))),
                            bankruptcy_price=Decimal(str(pos.get("liquidation_price", "0"))),
                            mark_price=new_price,
                            timestamp_ms=start_ts + wave_num * 100,  # 100ms per wave
                        )
                    )

            # Create synthetic liquidation if no tracker
            if not new_liquidations:
                new_liquidations = [
                    LiquidationOrderInfo(
                        symbol=initial_liquidation.symbol,
                        side=initial_liquidation.side,
                        qty=wave_qty,
                        bankruptcy_price=new_price,
                        mark_price=new_price,
                        timestamp_ms=start_ts + wave_num * 100,
                    )
                ]

            wave = CascadeWave(
                wave_number=wave_num,
                liquidation_count=len(new_liquidations),
                total_qty=sum(l.qty for l in new_liquidations),
                price_impact_bps=impact_bps,
                timestamp_ms=start_ts + wave_num * 100,
                liquidations=new_liquidations,
            )
            result.waves.append(wave)

            cumulative_qty += wave.total_qty
            result.total_price_impact_bps += impact_bps
            current_price = new_price

        # Finalize result
        result.total_liquidations = sum(w.liquidation_count for w in result.waves)
        result.total_qty_liquidated = cumulative_qty
        result.final_price = current_price
        result.duration_ms = (result.waves[-1].timestamp_ms - start_ts) if result.waves else 0

        # Update statistics
        self._total_cascades += 1
        self._total_waves += len(result.waves)
        self._deepest_cascade = max(self._deepest_cascade, len(result.waves))

        return result

    def estimate_cascade_impact(
        self,
        initial_qty: Decimal,
        volatility: float,
        adv: float,
    ) -> Dict[str, float]:
        """
        Estimate cascade impact without full simulation.

        Quick estimation for pre-trade analysis.

        Args:
            initial_qty: Initial liquidation quantity
            volatility: Current volatility
            adv: Average daily volume

        Returns:
            Dict with estimated impact metrics
        """
        # Sum of geometric series: initial × (1 - decay^n) / (1 - decay)
        total_qty_factor = (1 - self._cascade_decay**self._max_waves) / (1 - self._cascade_decay)
        estimated_total_qty = float(initial_qty) * total_qty_factor

        # Aggregate impact
        participation = estimated_total_qty / adv if adv > 0 else 0.0
        estimated_impact_bps = (
            self._price_impact_coef * volatility * math.sqrt(participation) * 10000
        )

        return {
            "estimated_waves": min(self._max_waves, 5),
            "estimated_total_qty": estimated_total_qty,
            "estimated_impact_bps": estimated_impact_bps,
            "decay_factor": self._cascade_decay,
        }

    @property
    def stats(self) -> Dict[str, Any]:
        """Get simulator statistics."""
        avg_depth = self._total_waves / self._total_cascades if self._total_cascades > 0 else 0
        return {
            "total_cascades": self._total_cascades,
            "total_waves": self._total_waves,
            "avg_cascade_depth": avg_depth,
            "deepest_cascade": self._deepest_cascade,
            "cascade_decay": self._cascade_decay,
            "max_waves": self._max_waves,
        }


# =============================================================================
# Insurance Fund Manager
# =============================================================================


class InsuranceFundManager:
    """
    Manages insurance fund dynamics.

    The insurance fund:
    - Receives contributions from profitable liquidations
    - Pays out for bankrupt liquidations
    - Triggers ADL when depleted

    Example:
        >>> fund = InsuranceFundManager(initial_balance=1_000_000_000)
        >>>
        >>> # Liquidation with profit (fills above bankruptcy price)
        >>> impact = fund.process_liquidation_fill(
        ...     bankruptcy_price=50000,
        ...     fill_price=50100,
        ...     qty=1.0,
        ... )
        >>> # impact > 0: contribution to fund
        >>>
        >>> # Liquidation at loss (fills below bankruptcy price)
        >>> impact = fund.process_liquidation_fill(
        ...     bankruptcy_price=50000,
        ...     fill_price=49900,
        ...     qty=1.0,
        ... )
        >>> # impact < 0: payout from fund
    """

    def __init__(
        self,
        initial_balance: Decimal = DEFAULT_INSURANCE_FUND_INITIAL,
    ):
        """
        Initialize insurance fund manager.

        Args:
            initial_balance: Starting fund balance
        """
        if initial_balance < 0:
            raise ValueError(f"initial_balance must be >= 0, got {initial_balance}")

        self._state = InsuranceFundState(
            balance=initial_balance,
            high_water_mark=initial_balance,
        )
        self._history: List[Tuple[int, Decimal, str]] = []  # (ts, amount, reason)

    def process_liquidation_fill(
        self,
        bankruptcy_price: Decimal,
        fill_price: Decimal,
        qty: Decimal,
        side: str,
        timestamp_ms: int = 0,
    ) -> Decimal:
        """
        Process liquidation fill and update fund.

        Args:
            bankruptcy_price: Price at which position would be bankrupt
            fill_price: Actual fill price achieved
            qty: Quantity filled
            side: "BUY" (short close) or "SELL" (long close)
            timestamp_ms: Fill timestamp

        Returns:
            Impact on fund (positive = contribution, negative = payout)
        """
        # Calculate difference
        # Long liquidation (SELL): profit if fill > bankruptcy
        # Short liquidation (BUY): profit if fill < bankruptcy
        if side.upper() == "SELL":
            # Long position liquidation
            price_diff = fill_price - bankruptcy_price
        else:
            # Short position liquidation
            price_diff = bankruptcy_price - fill_price

        impact = price_diff * qty

        if impact > 0:
            # Profitable liquidation → fund receives excess
            self._state = InsuranceFundState(
                balance=self._state.balance + impact,
                high_water_mark=max(self._state.high_water_mark, self._state.balance + impact),
                last_contribution_ms=timestamp_ms,
                total_contributions=self._state.total_contributions + impact,
                total_payouts=self._state.total_payouts,
            )
            self._history.append((timestamp_ms, impact, "contribution"))
        elif impact < 0:
            # Loss → fund covers deficit
            payout = abs(impact)
            new_balance = self._state.balance - payout
            self._state = InsuranceFundState(
                balance=max(Decimal("0"), new_balance),  # Can't go below zero
                high_water_mark=self._state.high_water_mark,
                last_contribution_ms=self._state.last_contribution_ms,
                total_contributions=self._state.total_contributions,
                total_payouts=self._state.total_payouts + payout,
            )
            self._history.append((timestamp_ms, -payout, "payout"))

        return impact

    def check_adl_trigger(self) -> bool:
        """
        Check if ADL should be triggered.

        ADL triggers when insurance fund is depleted and cannot cover losses.
        """
        return self._state.is_depleted

    @property
    def balance(self) -> Decimal:
        """Current fund balance."""
        return self._state.balance

    @property
    def state(self) -> InsuranceFundState:
        """Current fund state."""
        return self._state

    def get_utilization_history(
        self,
        lookback_entries: int = 100,
    ) -> List[Tuple[int, float]]:
        """
        Get fund utilization history.

        Returns:
            List of (timestamp, utilization_pct) tuples
        """
        if not self._history:
            return []

        recent = self._history[-lookback_entries:]
        running_balance = self._state.balance

        # Work backwards
        result = []
        for ts, amount, _ in reversed(recent):
            running_balance -= amount
            util = (
                float((self._state.high_water_mark - running_balance) / self._state.high_water_mark)
                * 100
            )
            result.append((ts, util))

        return list(reversed(result))

    def reset(self, new_balance: Optional[Decimal] = None) -> None:
        """Reset fund state."""
        balance = new_balance if new_balance is not None else DEFAULT_INSURANCE_FUND_INITIAL
        self._state = InsuranceFundState(
            balance=balance,
            high_water_mark=balance,
        )
        self._history.clear()


# =============================================================================
# ADL Queue Manager
# =============================================================================


class ADLQueueManager:
    """
    Manages Auto-Deleveraging (ADL) queue.

    When insurance fund is depleted, profitable traders on the opposite side
    are forced to close at bankruptcy price (ADL).

    ADL ranking is based on:
    - PnL percentile (higher profit = higher rank)
    - Leverage percentile (higher leverage = higher rank)

    Score = PnL_percentile × Leverage_percentile
    Rank 5: score >= 0.8 (highest priority for ADL)
    Rank 4: score >= 0.6
    Rank 3: score >= 0.4
    Rank 2: score >= 0.2
    Rank 1: score < 0.2 (lowest risk)

    Example:
        >>> adl = ADLQueueManager()
        >>>
        >>> # Build queue from profitable positions
        >>> adl.build_queue(positions, mark_price)
        >>>
        >>> # Get top candidates for ADL
        >>> candidates = adl.get_adl_candidates(qty_to_fill=10.0)

    References:
        - Binance ADL: https://www.binance.com/en/support/faq/360033525711
    """

    def __init__(self):
        """Initialize ADL queue manager."""
        self._queue: Dict[str, List[ADLQueueEntry]] = {}  # symbol_side -> sorted entries
        self._last_update_ms: int = 0

    def build_queue(
        self,
        positions: List[Dict[str, Any]],
        mark_price: Decimal,
        symbol: str,
        side: str,
    ) -> List[ADLQueueEntry]:
        """
        Build ADL queue from positions.

        Positions on opposite side of liquidation are ranked.

        Args:
            positions: List of position dicts with qty, entry_price, leverage
            mark_price: Current mark price
            symbol: Contract symbol
            side: Side to build queue for ("LONG" or "SHORT")

        Returns:
            Sorted queue entries (highest rank first)
        """
        if not positions:
            return []

        # Calculate PnL and leverage for each position
        scored = []
        for pos in positions:
            qty = Decimal(str(pos.get("qty", "0")))
            entry = Decimal(str(pos.get("entry_price", "0")))
            leverage = int(pos.get("leverage", 1))
            margin = Decimal(str(pos.get("margin", "1")))

            # Skip wrong side positions
            if side == "LONG" and qty <= 0:
                continue
            if side == "SHORT" and qty >= 0:
                continue

            # Calculate PnL ratio
            pnl = (mark_price - entry) * qty
            pnl_ratio = float(pnl / margin) if margin > 0 else 0.0

            scored.append(
                {
                    "account_id": pos.get("account_id", f"acc_{len(scored)}"),
                    "qty": abs(qty),
                    "pnl_ratio": pnl_ratio,
                    "leverage": leverage,
                }
            )

        if not scored:
            return []

        # Calculate percentiles
        pnl_values = sorted([s["pnl_ratio"] for s in scored])
        lev_values = sorted([s["leverage"] for s in scored])

        def percentile_rank(value: float, sorted_list: List[float]) -> float:
            if not sorted_list:
                return 0.0
            # Count values less than or equal
            count = sum(1 for v in sorted_list if v <= value)
            return count / len(sorted_list)

        # Build queue entries
        queue = []
        for s in scored:
            pnl_pct = percentile_rank(s["pnl_ratio"], pnl_values)
            lev_pct = percentile_rank(float(s["leverage"]), [float(v) for v in lev_values])
            score = pnl_pct * lev_pct

            # Determine rank
            rank = 1
            for i, threshold in enumerate(ADL_RANK_THRESHOLDS):
                if score >= threshold:
                    rank = i + 2

            queue.append(
                ADLQueueEntry(
                    account_id=s["account_id"],
                    symbol=symbol,
                    side=side,
                    qty=s["qty"],
                    pnl_percentile=pnl_pct,
                    leverage_percentile=lev_pct,
                    rank=min(rank, 5),
                    estimated_adl_qty=s["qty"],
                )
            )

        # Sort by rank descending
        queue.sort(key=lambda x: (-x.rank, -x.pnl_percentile))

        # Store
        key = f"{symbol}_{side}"
        self._queue[key] = queue
        self._last_update_ms = int(time.time() * 1000)

        return queue

    def get_adl_candidates(
        self,
        symbol: str,
        side: str,
        qty_to_fill: Decimal,
    ) -> List[Tuple[ADLQueueEntry, Decimal]]:
        """
        Get ADL candidates to fill required quantity.

        Returns positions in queue order until qty is satisfied.

        Args:
            symbol: Contract symbol
            side: Position side to ADL
            qty_to_fill: Quantity needed to fill

        Returns:
            List of (entry, qty_to_adl) tuples
        """
        key = f"{symbol}_{side}"
        queue = self._queue.get(key, [])

        candidates = []
        remaining = qty_to_fill

        for entry in queue:
            if remaining <= 0:
                break

            adl_qty = min(remaining, entry.qty)
            candidates.append((entry, adl_qty))
            remaining -= adl_qty

        return candidates

    def get_queue(self, symbol: str, side: str) -> List[ADLQueueEntry]:
        """Get current queue for symbol/side."""
        return self._queue.get(f"{symbol}_{side}", [])

    def clear(self) -> None:
        """Clear all queues."""
        self._queue.clear()


# =============================================================================
# Funding Period Dynamics
# =============================================================================


class FundingPeriodDynamics:
    """
    Models queue dynamics near funding time.

    Near funding settlement:
    - Spreads widen
    - Impact increases
    - Queue positions become more aggressive

    Example:
        >>> dynamics = FundingPeriodDynamics()
        >>> state = dynamics.get_state(current_ts, funding_rate)
        >>>
        >>> if state.is_funding_window:
        ...     adjusted_spread = base_spread * state.spread_multiplier
    """

    # Minutes before funding when effects start
    FUNDING_WINDOW_MINUTES = 15

    # Maximum multipliers
    MAX_SPREAD_MULTIPLIER = 2.0
    MAX_IMPACT_MULTIPLIER = 1.5

    def __init__(
        self,
        funding_times_utc: List[int] = None,
    ):
        """
        Initialize funding dynamics.

        Args:
            funding_times_utc: Funding hours in UTC (default: [0, 8, 16])
        """
        self._funding_times = funding_times_utc or FUNDING_TIMES_UTC

    def get_state(
        self,
        current_ts_ms: int,
        current_funding_rate: Decimal,
    ) -> FundingPeriodState:
        """
        Get current funding period state.

        Args:
            current_ts_ms: Current timestamp
            current_funding_rate: Current funding rate

        Returns:
            FundingPeriodState with multipliers
        """
        from datetime import datetime, timezone, timedelta

        dt = datetime.fromtimestamp(current_ts_ms / 1000, tz=timezone.utc)
        current_hour = dt.hour
        current_minute = dt.minute

        # Find next funding time
        next_funding_hour = None
        for funding_hour in self._funding_times:
            if current_hour < funding_hour:
                next_funding_hour = funding_hour
                break
            elif current_hour == funding_hour and current_minute < 0:
                next_funding_hour = funding_hour
                break

        if next_funding_hour is None:
            # Next funding is tomorrow 00:00
            next_dt = (dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            next_dt = dt.replace(hour=next_funding_hour, minute=0, second=0, microsecond=0)

        next_funding_ms = int(next_dt.timestamp() * 1000)
        time_to_funding_ms = max(0, next_funding_ms - current_ts_ms)
        time_to_funding_min = time_to_funding_ms / (60 * 1000)

        # Determine if in funding window
        is_funding_window = time_to_funding_min <= self.FUNDING_WINDOW_MINUTES

        # Calculate multipliers (linear ramp up as funding approaches)
        if is_funding_window and time_to_funding_min > 0:
            proximity = 1 - (time_to_funding_min / self.FUNDING_WINDOW_MINUTES)
            spread_mult = 1.0 + proximity * (self.MAX_SPREAD_MULTIPLIER - 1.0)
            impact_mult = 1.0 + proximity * (self.MAX_IMPACT_MULTIPLIER - 1.0)
        else:
            spread_mult = 1.0
            impact_mult = 1.0

        return FundingPeriodState(
            current_funding_rate=current_funding_rate,
            next_funding_time_ms=next_funding_ms,
            time_to_funding_ms=time_to_funding_ms,
            is_funding_window=is_funding_window,
            spread_multiplier=spread_mult,
            impact_multiplier=impact_mult,
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_liquidation_stream(
    max_buffer_size: int = 10000,
) -> LiquidationOrderStream:
    """
    Create liquidation order stream.

    Args:
        max_buffer_size: Maximum buffer size

    Returns:
        LiquidationOrderStream instance
    """
    return LiquidationOrderStream(max_buffer_size=max_buffer_size)


def create_cascade_simulator(
    cascade_decay: float = DEFAULT_CASCADE_DECAY,
    max_waves: int = DEFAULT_MAX_CASCADE_WAVES,
    price_impact_coef: float = 0.1,
) -> LiquidationCascadeSimulator:
    """
    Create liquidation cascade simulator.

    Args:
        cascade_decay: Decay factor per wave
        max_waves: Maximum cascade waves
        price_impact_coef: Kyle lambda coefficient

    Returns:
        LiquidationCascadeSimulator instance
    """
    return LiquidationCascadeSimulator(
        cascade_decay=cascade_decay,
        max_waves=max_waves,
        price_impact_coef=price_impact_coef,
    )


def create_insurance_fund(
    initial_balance: Decimal = DEFAULT_INSURANCE_FUND_INITIAL,
) -> InsuranceFundManager:
    """
    Create insurance fund manager.

    Args:
        initial_balance: Starting balance

    Returns:
        InsuranceFundManager instance
    """
    return InsuranceFundManager(initial_balance=initial_balance)


def create_adl_manager() -> ADLQueueManager:
    """
    Create ADL queue manager.

    Returns:
        ADLQueueManager instance
    """
    return ADLQueueManager()


def create_funding_dynamics(
    funding_times_utc: Optional[List[int]] = None,
) -> FundingPeriodDynamics:
    """
    Create funding period dynamics.

    Args:
        funding_times_utc: Funding hours

    Returns:
        FundingPeriodDynamics instance
    """
    return FundingPeriodDynamics(funding_times_utc=funding_times_utc)
