"""
Hidden Liquidity Detection and Modeling for L3 LOB Simulation.

This module provides algorithms for detecting and modeling hidden liquidity,
particularly iceberg orders that show characteristic refill patterns.

Key Features:
- IcebergDetector: Pattern-based detection from execution flow
- HiddenLiquidityEstimator: Estimate total hidden quantity
- IcebergOrder: Data structure for detected icebergs
- RefillPattern: Pattern analysis for iceberg identification

Reference:
    Bookmap iceberg detection methodology
    https://bookmap.com/blog/advanced-order-flow-trading-spotting-hidden-liquidity-iceberg-orders

Iceberg Detection Patterns:
    1. Multiple fills at same price without visible qty change
    2. Visible qty "refills" after each execution
    3. Pattern: visible_size -> fill -> visible_size -> fill -> ...
    4. Total executed quantity exceeds initially visible quantity

Note:
    This module is for EQUITY L3 simulation only.
    Crypto uses separate execution paths (Cython LOB).
"""

from __future__ import annotations

import math
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
    PriceLevel,
    Side,
    Trade,
)


# ==============================================================================
# Enums
# ==============================================================================


class IcebergState(IntEnum):
    """State of iceberg order detection."""

    SUSPECTED = 1  # Initial detection, needs confirmation
    CONFIRMED = 2  # Multiple refills observed
    EXHAUSTED = 3  # Hidden reserve depleted
    CANCELLED = 4  # Order was cancelled


class DetectionConfidence(IntEnum):
    """Confidence level for iceberg detection."""

    LOW = 1  # Single refill observed
    MEDIUM = 2  # Multiple refills with consistent size
    HIGH = 3  # Strong pattern match with statistical significance


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass
class RefillEvent:
    """
    Record of a single iceberg refill event.

    Attributes:
        timestamp_ns: Time of refill detection (nanoseconds)
        filled_qty: Quantity that was filled before refill
        refill_qty: New display quantity after refill
        price: Price level where refill occurred
        pre_level_qty: Visible qty at level before fill
        post_level_qty: Visible qty at level after refill
    """

    timestamp_ns: int
    filled_qty: float
    refill_qty: float
    price: float
    pre_level_qty: float = 0.0
    post_level_qty: float = 0.0


@dataclass
class IcebergOrder:
    """
    Detected iceberg order with tracking state.

    Attributes:
        iceberg_id: Unique identifier for this iceberg
        order_id: Underlying order ID (if known)
        price: Price level of the iceberg
        side: BUY or SELL
        display_size: Estimated display (visible) size per refill
        estimated_hidden_qty: Estimated remaining hidden quantity
        total_executed: Total quantity executed so far
        refill_events: History of refill events
        state: Current detection state
        confidence: Detection confidence level
        first_seen_ns: Timestamp of first detection
        last_update_ns: Timestamp of last update
        participant_id: Optional participant identifier
    """

    iceberg_id: str
    order_id: Optional[str]
    price: float
    side: Side
    display_size: float
    estimated_hidden_qty: float = 0.0
    total_executed: float = 0.0
    refill_events: List[RefillEvent] = field(default_factory=list)
    state: IcebergState = IcebergState.SUSPECTED
    confidence: DetectionConfidence = DetectionConfidence.LOW
    first_seen_ns: int = 0
    last_update_ns: int = 0
    participant_id: Optional[str] = None

    @property
    def refill_count(self) -> int:
        """Number of refills observed."""
        return len(self.refill_events)

    @property
    def estimated_total_size(self) -> float:
        """Estimated total iceberg size (executed + remaining hidden)."""
        return self.total_executed + self.estimated_hidden_qty

    @property
    def is_active(self) -> bool:
        """Check if iceberg is still active."""
        return self.state in (IcebergState.SUSPECTED, IcebergState.CONFIRMED)

    @property
    def avg_refill_size(self) -> float:
        """Average refill size from observed events."""
        if not self.refill_events:
            return self.display_size
        return sum(e.refill_qty for e in self.refill_events) / len(self.refill_events)

    def add_refill(self, event: RefillEvent) -> None:
        """Add a refill event and update state."""
        self.refill_events.append(event)
        self.total_executed += event.filled_qty
        self.last_update_ns = event.timestamp_ns

        # Update confidence based on refill count
        if len(self.refill_events) >= 3:
            self.confidence = DetectionConfidence.HIGH
            self.state = IcebergState.CONFIRMED
        elif len(self.refill_events) >= 2:
            self.confidence = DetectionConfidence.MEDIUM
            self.state = IcebergState.CONFIRMED

    def update_hidden_estimate(self, new_estimate: float) -> None:
        """Update estimated hidden quantity."""
        self.estimated_hidden_qty = max(0.0, new_estimate)
        if self.estimated_hidden_qty <= 0:
            self.state = IcebergState.EXHAUSTED


@dataclass
class LevelSnapshot:
    """Snapshot of price level state for change detection."""

    price: float
    visible_qty: float
    order_count: int
    timestamp_ns: int
    order_ids: Set[str] = field(default_factory=set)


@dataclass
class ExecutionEvent:
    """
    Execution event for iceberg detection.

    Attributes:
        trade: The trade that occurred
        pre_level: Level state before execution
        post_level: Level state after execution
    """

    trade: Trade
    pre_level: Optional[LevelSnapshot] = None
    post_level: Optional[LevelSnapshot] = None


# ==============================================================================
# Iceberg Detector
# ==============================================================================


class IcebergDetector:
    """
    Detects and tracks iceberg orders from execution flow.

    Detection Algorithm:
        1. Monitor executions at each price level
        2. Detect "refill" pattern: visible qty restored after fill
        3. Track refill frequency and consistency
        4. Estimate hidden reserve based on execution patterns

    Reference:
        Bookmap iceberg detection methodology
        https://bookmap.com/blog/advanced-order-flow-trading-spotting-hidden-liquidity-iceberg-orders

    Usage:
        detector = IcebergDetector()

        # Process executions
        for trade in trades:
            pre_snap = detector.take_level_snapshot(level, side)
            # ... execution happens ...
            post_snap = detector.take_level_snapshot(level, side)
            iceberg = detector.process_execution(trade, pre_snap, post_snap)
            if iceberg:
                print(f"Detected iceberg: {iceberg}")

        # Get all active icebergs
        for iceberg in detector.get_active_icebergs():
            print(f"Active iceberg at {iceberg.price}: {iceberg.estimated_hidden_qty}")
    """

    def __init__(
        self,
        min_refill_ratio: float = 0.8,
        max_refill_ratio: float = 1.2,
        min_refills_to_confirm: int = 2,
        lookback_window_ns: int = 60_000_000_000,  # 60 seconds
        min_display_size: float = 1.0,
        decay_factor: float = 0.95,
        initial_hidden_multiplier: float = 3.0,
        on_iceberg_detected: Optional[Callable[[IcebergOrder], None]] = None,
        on_iceberg_exhausted: Optional[Callable[[IcebergOrder], None]] = None,
    ) -> None:
        """
        Initialize iceberg detector.

        Args:
            min_refill_ratio: Minimum refill/display ratio to consider consistent
            max_refill_ratio: Maximum refill/display ratio to consider consistent
            min_refills_to_confirm: Minimum refills to confirm iceberg
            lookback_window_ns: Time window for pattern detection (ns)
            min_display_size: Minimum display size to consider
            decay_factor: Decay factor for hidden estimate (0.95 = conservative)
            initial_hidden_multiplier: Multiplier for initial hidden estimate
                (display_size * multiplier). Default 3.0 means initial estimate
                is 3x display size (assumes ~2x more hidden reserve).
            on_iceberg_detected: Callback when iceberg is first detected
            on_iceberg_exhausted: Callback when iceberg reserve is exhausted
        """
        self._min_refill_ratio = min_refill_ratio
        self._max_refill_ratio = max_refill_ratio
        self._min_refills_to_confirm = min_refills_to_confirm
        self._lookback_window_ns = lookback_window_ns
        self._min_display_size = min_display_size
        self._decay_factor = decay_factor
        self._initial_hidden_multiplier = initial_hidden_multiplier

        # Callbacks
        self._on_iceberg_detected = on_iceberg_detected
        self._on_iceberg_exhausted = on_iceberg_exhausted

        # State tracking
        self._icebergs: Dict[str, IcebergOrder] = {}  # iceberg_id -> IcebergOrder
        self._price_to_iceberg: Dict[Tuple[float, Side], str] = {}  # (price, side) -> iceberg_id
        # Level history stores post-execution snapshots for each (price, side) pair
        self._level_history: Dict[Tuple[float, Side], Deque[LevelSnapshot]] = {}
        self._execution_history: Deque[ExecutionEvent] = deque(maxlen=1000)

        # Statistics
        self._total_detected = 0
        self._total_confirmed = 0
        self._iceberg_counter = 0

    def take_level_snapshot(
        self,
        level: PriceLevel,
        side: Side,
        timestamp_ns: Optional[int] = None,
    ) -> LevelSnapshot:
        """
        Take a snapshot of price level state.

        Args:
            level: Price level to snapshot
            side: BUY or SELL side
            timestamp_ns: Optional timestamp (defaults to current time)

        Returns:
            LevelSnapshot with current state
        """
        ts = timestamp_ns or time.time_ns()
        order_ids = {o.order_id for o in level.iter_orders()}

        return LevelSnapshot(
            price=level.price,
            visible_qty=level.total_visible_qty,
            order_count=level.order_count,
            timestamp_ns=ts,
            order_ids=order_ids,
        )

    def process_execution(
        self,
        trade: Trade,
        pre_level: Optional[LevelSnapshot],
        post_level: Optional[LevelSnapshot],
        side: Optional[Side] = None,
    ) -> Optional[IcebergOrder]:
        """
        Process an execution and detect iceberg patterns.

        This is the main detection method. Call it after each execution
        with snapshots taken before and after the fill.

        Args:
            trade: The trade that occurred
            pre_level: Level snapshot before execution
            post_level: Level snapshot after execution
            side: Side of the resting order (inferred from trade if not provided)

        Returns:
            IcebergOrder if detected or updated, None otherwise
        """
        if pre_level is None or post_level is None:
            return None

        # Infer side from trade (maker side is opposite of aggressor)
        if side is None:
            if trade.aggressor_side:
                side = Side.SELL if trade.aggressor_side == Side.BUY else Side.BUY
            else:
                return None

        # Record execution event
        event = ExecutionEvent(trade=trade, pre_level=pre_level, post_level=post_level)
        self._execution_history.append(event)

        # Update level history (only store post-level to avoid redundancy)
        # post_level of step N equals pre_level of step N+1
        key = (pre_level.price, side)
        if key not in self._level_history:
            self._level_history[key] = deque(maxlen=100)
        self._level_history[key].append(post_level)

        # Check for iceberg pattern: refill after execution
        iceberg = self._detect_iceberg_pattern(trade, pre_level, post_level, side)

        return iceberg

    def _detect_iceberg_pattern(
        self,
        trade: Trade,
        pre_level: LevelSnapshot,
        post_level: LevelSnapshot,
        side: Side,
    ) -> Optional[IcebergOrder]:
        """
        Detect iceberg refill pattern from level changes.

        Pattern: level qty decreases by fill amount, then increases back
        (possibly to same or similar level) indicating hidden reserve.

        Args:
            trade: The executed trade
            pre_level: Level before execution
            post_level: Level after execution
            side: Side of the level

        Returns:
            IcebergOrder if pattern detected
        """
        price = pre_level.price
        fill_qty = trade.qty
        key = (price, side)

        # Calculate expected post-fill qty (if no refill)
        expected_post_qty = pre_level.visible_qty - fill_qty

        # Detect refill: actual post qty > expected (indicating replenishment)
        actual_post_qty = post_level.visible_qty
        refill_detected = actual_post_qty > expected_post_qty + 1e-9

        if not refill_detected:
            # Check if this might still be an iceberg being exhausted
            if key in self._price_to_iceberg:
                iceberg_id = self._price_to_iceberg[key]
                iceberg = self._icebergs.get(iceberg_id)
                if iceberg and iceberg.is_active:
                    # Update with final execution
                    iceberg.total_executed += fill_qty
                    # Check if exhausted (no refill this time)
                    if actual_post_qty < self._min_display_size:
                        self._mark_exhausted(iceberg)
                    return iceberg
            return None

        # Calculate refill amount
        refill_qty = actual_post_qty - expected_post_qty

        # Check if refill is significant enough
        if refill_qty < self._min_display_size:
            return None

        # Check for existing iceberg at this price
        if key in self._price_to_iceberg:
            iceberg_id = self._price_to_iceberg[key]
            iceberg = self._icebergs.get(iceberg_id)
            if iceberg and iceberg.is_active:
                # Update existing iceberg
                return self._update_iceberg(iceberg, trade, refill_qty, pre_level, post_level)

        # Create new iceberg
        return self._create_iceberg(
            trade=trade,
            price=price,
            side=side,
            display_size=refill_qty,
            fill_qty=fill_qty,
            pre_level=pre_level,
            post_level=post_level,
        )

    def _create_iceberg(
        self,
        trade: Trade,
        price: float,
        side: Side,
        display_size: float,
        fill_qty: float,
        pre_level: LevelSnapshot,
        post_level: LevelSnapshot,
    ) -> IcebergOrder:
        """Create a new iceberg order tracking entry."""
        self._iceberg_counter += 1
        iceberg_id = f"iceberg_{self._iceberg_counter}_{trade.timestamp_ns}"

        # Create refill event
        refill_event = RefillEvent(
            timestamp_ns=trade.timestamp_ns,
            filled_qty=fill_qty,
            refill_qty=display_size,
            price=price,
            pre_level_qty=pre_level.visible_qty,
            post_level_qty=post_level.visible_qty,
        )

        # Estimate initial hidden quantity using configurable multiplier
        # Default 3.0 means initial estimate is 3x display size
        initial_hidden_estimate = display_size * self._initial_hidden_multiplier

        iceberg = IcebergOrder(
            iceberg_id=iceberg_id,
            order_id=trade.maker_order_id,
            price=price,
            side=side,
            display_size=display_size,
            estimated_hidden_qty=initial_hidden_estimate,
            total_executed=fill_qty,
            refill_events=[refill_event],
            state=IcebergState.SUSPECTED,
            confidence=DetectionConfidence.LOW,
            first_seen_ns=trade.timestamp_ns,
            last_update_ns=trade.timestamp_ns,
        )

        # Register
        self._icebergs[iceberg_id] = iceberg
        self._price_to_iceberg[(price, side)] = iceberg_id
        self._total_detected += 1

        # Callback
        if self._on_iceberg_detected:
            self._on_iceberg_detected(iceberg)

        return iceberg

    def _update_iceberg(
        self,
        iceberg: IcebergOrder,
        trade: Trade,
        refill_qty: float,
        pre_level: LevelSnapshot,
        post_level: LevelSnapshot,
    ) -> IcebergOrder:
        """Update existing iceberg with new refill event."""
        fill_qty = trade.qty

        # Check refill consistency
        ratio = refill_qty / iceberg.display_size if iceberg.display_size > 0 else 1.0
        is_consistent = self._min_refill_ratio <= ratio <= self._max_refill_ratio

        # Create refill event
        refill_event = RefillEvent(
            timestamp_ns=trade.timestamp_ns,
            filled_qty=fill_qty,
            refill_qty=refill_qty,
            price=iceberg.price,
            pre_level_qty=pre_level.visible_qty,
            post_level_qty=post_level.visible_qty,
        )

        # Capture state BEFORE add_refill (which may change state)
        was_suspected = iceberg.state == IcebergState.SUSPECTED

        # Add refill and update state
        iceberg.add_refill(refill_event)

        # Update display size estimate (exponential moving average)
        if is_consistent:
            iceberg.display_size = 0.8 * iceberg.display_size + 0.2 * refill_qty

        # Update hidden quantity estimate
        # Conservative: decay existing estimate, assume some reserve remains
        self._update_hidden_estimate(iceberg)

        # Check for confirmation (use captured state before add_refill)
        if iceberg.refill_count >= self._min_refills_to_confirm:
            if was_suspected:
                # State was SUSPECTED before, now confirmed
                self._total_confirmed += 1

            # Higher confidence with more refills
            if iceberg.refill_count >= 5:
                iceberg.confidence = DetectionConfidence.HIGH

        return iceberg

    def _update_hidden_estimate(self, iceberg: IcebergOrder) -> None:
        """
        Update hidden quantity estimate based on execution patterns.

        Computes pattern-based estimate without decay. Decay is applied
        separately in estimate_hidden_reserve() to avoid double-decay.

        Note:
            The stored estimated_hidden_qty is the base estimate.
            Use estimate_hidden_reserve() for decay-adjusted values.
        """
        if not iceberg.refill_events:
            return

        # Get recent refill pattern
        recent_refills = iceberg.refill_events[-5:]
        avg_refill = sum(e.refill_qty for e in recent_refills) / len(recent_refills)

        # Check refill consistency
        if len(recent_refills) >= 2:
            refill_variance = sum((e.refill_qty - avg_refill) ** 2 for e in recent_refills) / len(
                recent_refills
            )
            consistency = 1.0 - min(1.0, math.sqrt(refill_variance) / (avg_refill + 1e-9))
        else:
            consistency = 0.5

        # Estimate remaining based on pattern
        # More consistent refills -> likely more hidden reserve
        multiplier = 2.0 + 3.0 * consistency  # 2x to 5x display remaining

        # Store pattern-based estimate (NO decay here - applied in estimate_hidden_reserve)
        # FIX: Removed double-decay by not applying decay_factor here
        pattern_estimate = avg_refill * multiplier
        iceberg.estimated_hidden_qty = max(
            iceberg.estimated_hidden_qty,
            pattern_estimate,
        )

    def _mark_exhausted(self, iceberg: IcebergOrder) -> None:
        """Mark iceberg as exhausted and trigger callback."""
        iceberg.state = IcebergState.EXHAUSTED
        iceberg.estimated_hidden_qty = 0.0

        # Remove from active tracking
        key = (iceberg.price, iceberg.side)
        if key in self._price_to_iceberg:
            del self._price_to_iceberg[key]

        # Callback
        if self._on_iceberg_exhausted:
            self._on_iceberg_exhausted(iceberg)

    def detect_iceberg(
        self,
        executions: List[Trade],
        level_qty_history: List[float],
        price: Optional[float] = None,
        side: Optional[Side] = None,
    ) -> Optional[IcebergOrder]:
        """
        Detect iceberg from execution history at a price level.

        This is a batch detection method for historical analysis.

        Args:
            executions: List of trades at this price level
            level_qty_history: Visible qty at level after each execution
            price: Price level (inferred from first trade if not provided)
            side: Side of the level

        Returns:
            Detected IcebergOrder or None
        """
        if not executions or len(level_qty_history) < len(executions):
            return None

        # Infer price from trades
        if price is None:
            price = executions[0].price

        # Look for refill pattern
        refill_events: List[RefillEvent] = []
        prev_qty = level_qty_history[0] if level_qty_history else 0.0

        for i, trade in enumerate(executions):
            post_qty = level_qty_history[i] if i < len(level_qty_history) else 0.0

            # Expected qty after fill (no refill)
            expected_qty = max(0.0, prev_qty - trade.qty)

            # Detect refill
            if post_qty > expected_qty + 1e-9:
                refill_qty = post_qty - expected_qty
                if refill_qty >= self._min_display_size:
                    refill_events.append(
                        RefillEvent(
                            timestamp_ns=trade.timestamp_ns,
                            filled_qty=trade.qty,
                            refill_qty=refill_qty,
                            price=price,
                            pre_level_qty=prev_qty,
                            post_level_qty=post_qty,
                        )
                    )

            prev_qty = post_qty

        # Check if enough refills detected
        if len(refill_events) < 1:
            return None

        # Create iceberg from pattern
        total_executed = sum(t.qty for t in executions)
        avg_display = (
            sum(e.refill_qty for e in refill_events) / len(refill_events) if refill_events else 0.0
        )

        self._iceberg_counter += 1
        iceberg_id = f"iceberg_batch_{self._iceberg_counter}"

        is_confirmed = len(refill_events) >= 2
        # Use configurable multiplier for initial estimate
        initial_estimate = avg_display * self._initial_hidden_multiplier
        iceberg = IcebergOrder(
            iceberg_id=iceberg_id,
            order_id=executions[0].maker_order_id if executions else None,
            price=price,
            side=side or Side.BUY,
            display_size=avg_display,
            estimated_hidden_qty=initial_estimate,
            total_executed=total_executed,
            refill_events=refill_events,
            state=IcebergState.CONFIRMED if is_confirmed else IcebergState.SUSPECTED,
            confidence=(
                DetectionConfidence.HIGH
                if len(refill_events) >= 3
                else DetectionConfidence.MEDIUM if is_confirmed else DetectionConfidence.LOW
            ),
            first_seen_ns=executions[0].timestamp_ns if executions else 0,
            last_update_ns=executions[-1].timestamp_ns if executions else 0,
        )

        # Register in tracking state (FIX: was missing counter updates)
        self._icebergs[iceberg_id] = iceberg
        actual_side = side or Side.BUY
        self._price_to_iceberg[(price, actual_side)] = iceberg_id
        self._total_detected += 1
        if is_confirmed:
            self._total_confirmed += 1

        # Callback
        if self._on_iceberg_detected:
            self._on_iceberg_detected(iceberg)

        return iceberg

    def estimate_hidden_reserve(
        self,
        iceberg: IcebergOrder,
        current_visible_qty: float = 0.0,
    ) -> float:
        """
        Estimate remaining hidden quantity for an iceberg.

        Uses statistical model based on observed refill patterns.

        Args:
            iceberg: The iceberg order to estimate
            current_visible_qty: Current visible qty at the level

        Returns:
            Estimated remaining hidden quantity
        """
        if not iceberg.is_active:
            return 0.0

        if not iceberg.refill_events:
            # No refill data, use display size heuristic
            return iceberg.display_size * 2.0

        # Calculate average refill and consistency
        refills = [e.refill_qty for e in iceberg.refill_events]
        avg_refill = sum(refills) / len(refills)

        if len(refills) >= 2:
            # More data -> better estimate
            std_refill = math.sqrt(sum((r - avg_refill) ** 2 for r in refills) / len(refills))
            cv = std_refill / (avg_refill + 1e-9)  # Coefficient of variation

            # Lower CV = more consistent = likely more hidden reserve
            consistency_bonus = max(0.0, 1.0 - cv)
            multiplier = 2.0 + 3.0 * consistency_bonus  # 2x to 5x
        else:
            multiplier = 3.0  # Default multiplier

        # Base estimate from pattern
        base_estimate = avg_refill * multiplier

        # Adjust for current visible quantity
        if current_visible_qty > 0:
            # If visible qty is close to display size, reserve likely still exists
            visible_ratio = current_visible_qty / iceberg.display_size
            if visible_ratio >= 0.8:
                base_estimate *= 1.2  # Boost estimate

        # Apply decay based on executions
        decay = self._decay_factor**iceberg.refill_count
        final_estimate = base_estimate * decay

        return max(0.0, final_estimate)

    def get_iceberg(self, iceberg_id: str) -> Optional[IcebergOrder]:
        """Get iceberg by ID."""
        return self._icebergs.get(iceberg_id)

    def get_iceberg_at_price(
        self,
        price: float,
        side: Side,
    ) -> Optional[IcebergOrder]:
        """Get active iceberg at price level."""
        iceberg_id = self._price_to_iceberg.get((price, side))
        if iceberg_id:
            iceberg = self._icebergs.get(iceberg_id)
            if iceberg and iceberg.is_active:
                return iceberg
        return None

    def get_active_icebergs(self) -> Iterator[IcebergOrder]:
        """Iterate over all active icebergs."""
        for iceberg in self._icebergs.values():
            if iceberg.is_active:
                yield iceberg

    def get_all_icebergs(self) -> Iterator[IcebergOrder]:
        """Iterate over all icebergs (including exhausted)."""
        return iter(self._icebergs.values())

    def remove_stale_icebergs(self, current_time_ns: int) -> int:
        """
        Remove icebergs that haven't been updated within lookback window.

        Args:
            current_time_ns: Current timestamp in nanoseconds

        Returns:
            Number of icebergs removed
        """
        stale_ids: List[str] = []
        cutoff = current_time_ns - self._lookback_window_ns

        for iceberg_id, iceberg in self._icebergs.items():
            if iceberg.last_update_ns < cutoff and iceberg.is_active:
                stale_ids.append(iceberg_id)

        for iceberg_id in stale_ids:
            iceberg = self._icebergs[iceberg_id]
            self._mark_exhausted(iceberg)

        return len(stale_ids)

    def clear(self) -> None:
        """Clear all tracking state."""
        self._icebergs.clear()
        self._price_to_iceberg.clear()
        self._level_history.clear()
        self._execution_history.clear()

    def stats(self) -> Dict[str, float]:
        """Get detector statistics."""
        active_count = sum(1 for i in self._icebergs.values() if i.is_active)
        confirmed_active = sum(
            1 for i in self._icebergs.values() if i.is_active and i.state == IcebergState.CONFIRMED
        )
        total_hidden_estimate = sum(
            i.estimated_hidden_qty for i in self._icebergs.values() if i.is_active
        )

        return {
            "total_detected": float(self._total_detected),
            "total_confirmed": float(self._total_confirmed),
            "active_icebergs": float(active_count),
            "confirmed_active": float(confirmed_active),
            "total_hidden_estimate": total_hidden_estimate,
            "avg_refills_per_iceberg": (
                sum(i.refill_count for i in self._icebergs.values()) / len(self._icebergs)
                if self._icebergs
                else 0.0
            ),
        }


# ==============================================================================
# Hidden Liquidity Estimator
# ==============================================================================


class HiddenLiquidityEstimator:
    """
    Estimates total hidden liquidity at each price level.

    Combines iceberg detection with statistical modeling to estimate
    non-visible liquidity in the order book.

    Usage:
        estimator = HiddenLiquidityEstimator(detector)

        # Get estimated hidden liquidity at price level
        hidden = estimator.estimate_hidden_at_level(price=100.0, side=Side.BUY)

        # Get total hidden on one side
        total = estimator.estimate_total_hidden(side=Side.BUY)
    """

    def __init__(
        self,
        iceberg_detector: IcebergDetector,
        hidden_ratio_estimate: float = 0.15,
        confidence_weight: Dict[DetectionConfidence, float] = None,
    ) -> None:
        """
        Initialize hidden liquidity estimator.

        Args:
            iceberg_detector: IcebergDetector instance
            hidden_ratio_estimate: Default hidden/visible ratio for levels without detected icebergs
            confidence_weight: Weight to apply based on detection confidence
        """
        self._detector = iceberg_detector
        self._hidden_ratio = hidden_ratio_estimate
        self._confidence_weight = confidence_weight or {
            DetectionConfidence.LOW: 0.5,
            DetectionConfidence.MEDIUM: 0.75,
            DetectionConfidence.HIGH: 1.0,
        }

    def estimate_hidden_at_level(
        self,
        price: float,
        side: Side,
        visible_qty: float = 0.0,
    ) -> float:
        """
        Estimate hidden liquidity at a price level.

        Args:
            price: Price level
            side: BUY or SELL
            visible_qty: Current visible quantity at level

        Returns:
            Estimated hidden quantity
        """
        # Check for detected iceberg
        iceberg = self._detector.get_iceberg_at_price(price, side)

        if iceberg and iceberg.is_active:
            # Use iceberg estimate with confidence weighting
            weight = self._confidence_weight.get(iceberg.confidence, 0.75)
            iceberg_hidden = self._detector.estimate_hidden_reserve(iceberg, visible_qty)
            return iceberg_hidden * weight

        # Default estimate based on visible quantity
        return visible_qty * self._hidden_ratio

    def estimate_total_hidden(
        self,
        side: Side,
        price_levels: Optional[List[Tuple[float, float]]] = None,
    ) -> float:
        """
        Estimate total hidden liquidity on one side.

        Args:
            side: BUY or SELL
            price_levels: Optional list of (price, visible_qty) tuples

        Returns:
            Total estimated hidden quantity
        """
        # Sum iceberg estimates
        iceberg_hidden = sum(
            self._detector.estimate_hidden_reserve(i)
            for i in self._detector.get_active_icebergs()
            if i.side == side
        )

        # Add default estimates for non-iceberg levels
        if price_levels:
            for price, visible_qty in price_levels:
                iceberg = self._detector.get_iceberg_at_price(price, side)
                if not iceberg or not iceberg.is_active:
                    iceberg_hidden += visible_qty * self._hidden_ratio

        return iceberg_hidden

    def get_hidden_liquidity_map(
        self,
        side: Side,
    ) -> Dict[float, float]:
        """
        Get map of price -> estimated hidden quantity.

        Args:
            side: BUY or SELL

        Returns:
            Dict mapping price to hidden estimate
        """
        result: Dict[float, float] = {}

        for iceberg in self._detector.get_active_icebergs():
            if iceberg.side == side:
                result[iceberg.price] = self._detector.estimate_hidden_reserve(iceberg)

        return result


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_iceberg_detector(
    min_refills_to_confirm: int = 2,
    lookback_window_sec: float = 60.0,
    min_display_size: float = 1.0,
    decay_factor: float = 0.95,
    initial_hidden_multiplier: float = 3.0,
    on_iceberg_detected: Optional[Callable[[IcebergOrder], None]] = None,
    on_iceberg_exhausted: Optional[Callable[[IcebergOrder], None]] = None,
) -> IcebergDetector:
    """
    Create an IcebergDetector with common configuration.

    Args:
        min_refills_to_confirm: Minimum refills to confirm iceberg
        lookback_window_sec: Time window in seconds
        min_display_size: Minimum display size to consider
        decay_factor: Decay factor for hidden estimate (0.95 = conservative)
        initial_hidden_multiplier: Multiplier for initial hidden estimate
            (display_size * multiplier). Default 3.0.
        on_iceberg_detected: Detection callback
        on_iceberg_exhausted: Exhaustion callback

    Returns:
        Configured IcebergDetector
    """
    return IcebergDetector(
        min_refills_to_confirm=min_refills_to_confirm,
        lookback_window_ns=int(lookback_window_sec * 1_000_000_000),
        min_display_size=min_display_size,
        decay_factor=decay_factor,
        initial_hidden_multiplier=initial_hidden_multiplier,
        on_iceberg_detected=on_iceberg_detected,
        on_iceberg_exhausted=on_iceberg_exhausted,
    )


def create_hidden_liquidity_estimator(
    iceberg_detector: Optional[IcebergDetector] = None,
    hidden_ratio: float = 0.15,
) -> HiddenLiquidityEstimator:
    """
    Create a HiddenLiquidityEstimator.

    Args:
        iceberg_detector: Optional detector (creates new one if not provided)
        hidden_ratio: Default hidden/visible ratio

    Returns:
        Configured HiddenLiquidityEstimator
    """
    detector = iceberg_detector or create_iceberg_detector()
    return HiddenLiquidityEstimator(
        iceberg_detector=detector,
        hidden_ratio_estimate=hidden_ratio,
    )
