"""
Event-Driven LOB Coordinator for Options Markets.

This module provides efficient cross-series event propagation for
options chains with O(N log N) complexity instead of naive O(N²).

Problem Statement:
    - N option series × M quote updates per second = O(N×M) per tick
    - For SPY: 960 series × 100 updates = 96,000 operations/sec
    - Naive propagation: every update checks all 960 series

Solution: Strike Bucketing + Selective Propagation
    - Group series by strike buckets (e.g., $5 increments)
    - Only propagate events to nearby strikes (±2 buckets)
    - Same-expiry propagation for term structure effects
    - Result: ~30 series affected per update instead of 960

Complexity Analysis:
    - Naive: O(N²) per tick
    - With bucketing: O(N log N) per tick
    - Memory: O(N) for bucket index

Reference:
    Phase 0.5 of OPTIONS_INTEGRATION_PLAN.md

Performance Targets:
    - Event propagation: < 100 μs per tick
    - Bucket lookup: O(1)
    - Affected series calculation: O(log N)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Event Types
# ==============================================================================


class OptionsEventType(Enum):
    """Types of events that can propagate across series."""

    QUOTE_UPDATE = auto()  # Bid/ask price or size change
    QUOTE = QUOTE_UPDATE  # Alias for QUOTE_UPDATE
    TRADE = auto()  # Trade execution
    UNDERLYING_MOVE = auto()  # Underlying price change
    UNDERLYING_TICK = UNDERLYING_MOVE  # Alias for UNDERLYING_MOVE
    VOLATILITY_CHANGE = auto()  # Implied volatility change
    VOLATILITY = VOLATILITY_CHANGE  # Alias for VOLATILITY_CHANGE
    GREEK_UPDATE = auto()  # Greeks recalculation
    EXPIRY_TICK = auto()  # Time decay tick
    DIVIDEND_ANNOUNCEMENT = auto()  # Dividend affecting options
    EARNINGS_EVENT = auto()  # Earnings affecting IV


class PropagationScope(Enum):
    """Scope of event propagation."""

    LOCAL = auto()  # Only the source series
    NEARBY_STRIKES = auto()  # Source + nearby strikes (uses nearby_buckets config)
    ADJACENT = auto()  # Source + immediately adjacent strikes (±1 bucket only)
    SAME_EXPIRY = auto()  # All series with same expiry
    ALL_EXPIRIES = auto()  # All series (full chain)
    ALL = auto()  # All registered series
    ATM_ONLY = auto()  # Only at-the-money strikes
    CUSTOM = auto()  # Custom filter function


@dataclass
class OptionsQuote:
    """
    Quote data for an option series.

    Supports both old API (bid_price, ask_price, timestamp_ns) and new API
    (bid, ask, timestamp as datetime) for backward compatibility.

    Attributes:
        series_key: Series identifier
        bid_price: Best bid price (old API)
        ask_price: Best ask price (old API)
        bid: Best bid price (new API - preferred)
        ask: Best ask price (new API - preferred)
        bid_size: Size at best bid
        ask_size: Size at best ask
        underlying_price: Current underlying price
        implied_volatility: Implied volatility
        delta: Option delta
        timestamp_ns: Quote timestamp in nanoseconds (old API)
        timestamp: Quote timestamp as datetime (new API - preferred)
    """

    series_key: str
    # Old API fields
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_size: Union[float, Decimal] = 0.0
    ask_size: Union[float, Decimal] = 0.0
    underlying_price: float = 0.0
    implied_volatility: float = 0.0
    delta: float = 0.0
    timestamp_ns: int = 0
    # New API fields (for init kwargs)
    bid: Optional[Union[float, Decimal]] = field(default=None, repr=False)
    ask: Optional[Union[float, Decimal]] = field(default=None, repr=False)
    timestamp: Optional[datetime] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Handle initialization from either API style."""
        # Convert bid_size/ask_size to float for consistency
        self.bid_size = float(self.bid_size) if self.bid_size else 0.0
        self.ask_size = float(self.ask_size) if self.ask_size else 0.0

        # New API: bid/ask take precedence over bid_price/ask_price
        if self.bid is not None:
            self.bid_price = float(self.bid)
        if self.ask is not None:
            self.ask_price = float(self.ask)

        # Handle timestamp conversion
        if self.timestamp is not None and self.timestamp_ns == 0:
            self.timestamp_ns = int(self.timestamp.timestamp() * 1e9)
        elif self.timestamp_ns > 0 and self.timestamp is None:
            object.__setattr__(self, "timestamp", datetime.fromtimestamp(self.timestamp_ns / 1e9))

        # Sync bid/ask back to None-checked values for property access
        if self.bid is None:
            object.__setattr__(self, "bid", self.bid_price)
        if self.ask is None:
            object.__setattr__(self, "ask", self.ask_price)

    @property
    def mid_price(self) -> float:
        """Get mid price."""
        return (self.bid_price + self.ask_price) / 2.0

    @property
    def spread(self) -> float:
        """Get bid-ask spread."""
        return self.ask_price - self.bid_price

    @property
    def strike(self) -> float:
        """Extract strike from series key."""
        parts = self.series_key.split("_")
        if len(parts) >= 4:
            try:
                return float(parts[3])
            except ValueError:
                pass
        return 0.0

    @property
    def expiry(self) -> str:
        """Extract expiry from series key."""
        parts = self.series_key.split("_")
        if len(parts) >= 2:
            return parts[1]
        return ""


@dataclass
class OptionsEvent:
    """
    Event to be propagated across series.

    Supports both old API (source_series, timestamp_ns) and new API
    (series_key, timestamp as datetime) for backward compatibility.

    Attributes:
        event_type: Type of event
        source_series: Series that generated the event (or use 'series_key' alias)
        quote: Quote data (if applicable)
        underlying_price: Underlying price (for price moves)
        volatility_change: Change in IV (for vol events)
        timestamp_ns: Event timestamp (nanoseconds)
        scope: Propagation scope
        custom_filter: Custom filter function (if scope is CUSTOM)
        series_key: Alias for source_series (new API)
        trade_price: Trade price (for TRADE events)
        trade_qty: Trade quantity (for TRADE events)
        implied_vol: Implied volatility (for VOLATILITY events)
        delta: Option delta (for GREEK_UPDATE events)
        gamma: Option gamma (for GREEK_UPDATE events)
        theta: Option theta (for GREEK_UPDATE events)
        vega: Option vega (for GREEK_UPDATE events)
        _timestamp_dt: Internal datetime storage
    """

    event_type: OptionsEventType
    source_series: str = ""
    quote: Optional[OptionsQuote] = None
    underlying_price: Optional[Union[float, Decimal]] = None
    volatility_change: Optional[float] = None
    timestamp_ns: int = 0
    scope: PropagationScope = PropagationScope.NEARBY_STRIKES
    custom_filter: Optional[Callable[[str], bool]] = None
    # New API fields
    series_key: str = ""
    trade_price: Optional[Union[float, Decimal]] = None
    trade_qty: Optional[Union[float, Decimal]] = None
    implied_vol: Optional[Union[float, Decimal]] = None
    delta: Optional[Union[float, Decimal]] = None
    gamma: Optional[Union[float, Decimal]] = None
    theta: Optional[Union[float, Decimal]] = None
    vega: Optional[Union[float, Decimal]] = None
    timestamp: Optional[datetime] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Handle initialization from either API style."""
        # Handle series_key vs source_series
        if self.series_key and not self.source_series:
            self.source_series = self.series_key
        elif self.source_series and not self.series_key:
            self.series_key = self.source_series

        # If no source_series but we have a quote, extract from quote
        if not self.source_series and self.quote is not None:
            self.source_series = self.quote.series_key
            self.series_key = self.quote.series_key

        # Handle timestamp: new API (datetime) takes precedence
        if self.timestamp is not None and self.timestamp_ns == 0:
            self.timestamp_ns = int(self.timestamp.timestamp() * 1e9)
        elif self.timestamp_ns == 0:
            self.timestamp_ns = time.time_ns()
            object.__setattr__(self, "timestamp", datetime.fromtimestamp(self.timestamp_ns / 1e9))
        elif self.timestamp is None and self.timestamp_ns > 0:
            object.__setattr__(self, "timestamp", datetime.fromtimestamp(self.timestamp_ns / 1e9))


@dataclass
class PropagationResult:
    """Result of event propagation."""

    affected_series: List[str]
    propagation_time_ns: int
    num_buckets_checked: int
    source_series: str
    event_type: OptionsEventType

    @property
    def series_affected(self) -> int:
        """Get count of affected series (new API alias)."""
        return len(self.affected_series)


@dataclass
class CoordinatorStats:
    """Statistics for the event coordinator."""

    total_events: int = 0
    total_propagations: int = 0
    avg_affected_series: float = 0.0
    avg_propagation_time_ns: float = 0.0
    peak_affected_series: int = 0
    events_by_type: Dict[OptionsEventType, int] = field(default_factory=dict)
    buckets_count: int = 0
    series_count: int = 0

    @property
    def registered_series(self) -> int:
        """Alias for series_count for compatibility."""
        return self.series_count


# ==============================================================================
# Strike Bucketing
# ==============================================================================


class StrikeBucket:
    """
    Bucket for grouping series by strike price.

    Groups options with similar strikes together for efficient
    event propagation. Default bucket width is $5.

    Example:
        Strikes 100, 101, 102, 103, 104 all go in bucket 100
        Strikes 105, 106, 107, 108, 109 all go in bucket 105
    """

    def __init__(self, bucket_id: int, bucket_width: float = 5.0):
        """
        Initialize strike bucket.

        Args:
            bucket_id: Bucket identifier (strike floor / width)
            bucket_width: Width of each bucket
        """
        self._bucket_id = bucket_id
        self._bucket_width = bucket_width
        self._series: Set[str] = set()
        self._series_by_expiry: Dict[str, Set[str]] = defaultdict(set)

    @property
    def bucket_id(self) -> int:
        """Get bucket ID."""
        return self._bucket_id

    @property
    def strike_range(self) -> Tuple[float, float]:
        """Get (min, max) strike for this bucket."""
        min_strike = self._bucket_id * self._bucket_width
        max_strike = min_strike + self._bucket_width
        return (min_strike, max_strike)

    @property
    def series_count(self) -> int:
        """Get number of series in bucket."""
        return len(self._series)

    def add_series(self, series_key: str, expiry: str) -> None:
        """Add series to bucket."""
        self._series.add(series_key)
        self._series_by_expiry[expiry].add(series_key)

    def remove_series(self, series_key: str, expiry: str) -> None:
        """Remove series from bucket."""
        self._series.discard(series_key)
        if expiry in self._series_by_expiry:
            self._series_by_expiry[expiry].discard(series_key)
            if not self._series_by_expiry[expiry]:
                del self._series_by_expiry[expiry]

    def get_all_series(self) -> Set[str]:
        """Get all series in bucket."""
        return self._series.copy()

    def get_series_by_expiry(self, expiry: str) -> Set[str]:
        """Get series for specific expiry."""
        return self._series_by_expiry.get(expiry, set()).copy()

    def get_expiries(self) -> Set[str]:
        """Get all expiries in bucket."""
        return set(self._series_by_expiry.keys())


# ==============================================================================
# Event-Driven LOB Coordinator
# ==============================================================================


class EventDrivenLOBCoordinator:
    """
    Efficient cross-series event propagation using strike bucketing.

    Groups option series by strike buckets and propagates events
    only to nearby buckets and same-expiry series, reducing
    complexity from O(N²) to O(N log N).

    Usage:
        coordinator = EventDrivenLOBCoordinator(
            bucket_width=5.0,
            nearby_buckets=2,
        )

        # Register series
        coordinator.register_series("AAPL_241220_C_150")
        coordinator.register_series("AAPL_241220_C_155")

        # Propagate event
        event = OptionsEvent(
            event_type=OptionsEventType.QUOTE_UPDATE,
            source_series="AAPL_241220_C_150",
            quote=quote,
        )
        result = coordinator.propagate_event(event)
        print(f"Affected series: {result.affected_series}")

    Thread Safety:
        This class is NOT thread-safe. External synchronization required.
    """

    def __init__(
        self,
        bucket_width: float = 5.0,
        nearby_buckets: int = 2,
        same_expiry_propagation: bool = True,
        on_event: Optional[Callable[[str, OptionsEvent], None]] = None,
        underlying: Optional[str] = None,
        expiry: Optional[str] = None,
        max_propagation_depth: Optional[int] = None,
    ):
        """
        Initialize the event coordinator.

        Args:
            bucket_width: Width of strike buckets (default: $5)
            nearby_buckets: Number of buckets each side to propagate to
            same_expiry_propagation: Propagate to same-expiry series
            on_event: Callback when event is propagated to a series
            underlying: Underlying symbol (for factory-created coordinators)
            expiry: Expiry date (for factory-created coordinators)
            max_propagation_depth: Alias for nearby_buckets (new API)

        Raises:
            ValueError: If bucket_width <= 0 or nearby_buckets < 0
        """
        # Convert bucket_width to float for consistent arithmetic
        bucket_width_float = float(bucket_width)
        if bucket_width_float <= 0:
            raise ValueError("bucket_width must be > 0")
        # Handle max_propagation_depth as alias for nearby_buckets
        if max_propagation_depth is not None:
            nearby_buckets = max_propagation_depth
        if nearby_buckets < 0:
            raise ValueError("nearby_buckets must be >= 0")

        self._bucket_width = bucket_width_float
        self._nearby_buckets = nearby_buckets
        self._same_expiry_propagation = same_expiry_propagation
        self._on_event = on_event
        self._underlying = underlying
        self._expiry = expiry

        # Strike buckets indexed by bucket_id
        self._buckets: Dict[int, StrikeBucket] = {}

        # Series to bucket mapping
        self._series_to_bucket: Dict[str, int] = {}

        # Strike to bucket mapping (for get_bucket_for_strike)
        self._strike_to_bucket: Dict[float, int] = {}

        # Expiry index for cross-strike propagation
        self._expiry_index: Dict[str, Set[str]] = defaultdict(set)

        # Per-series callbacks
        self._per_series_callbacks: Dict[str, List[Callable[[str, OptionsEvent], None]]] = (
            defaultdict(list)
        )

        # Current ATM strike (for ATM_ONLY scope)
        self._atm_strike: Optional[float] = None

        # Statistics
        self._stats = CoordinatorStats()

        logger.info(
            f"EventDrivenLOBCoordinator initialized: "
            f"bucket_width={bucket_width}, nearby_buckets={nearby_buckets}"
        )

    @property
    def underlying(self) -> Optional[str]:
        """Get underlying symbol."""
        return self._underlying

    @property
    def expiry(self) -> Optional[str]:
        """Get expiry date."""
        return self._expiry

    @property
    def bucket_width(self) -> float:
        """Get bucket width."""
        return self._bucket_width

    # ==========================================================================
    # Series Registration
    # ==========================================================================

    def register_series(
        self,
        series_key: str,
        strike: Optional[Union[float, Decimal]] = None,
    ) -> None:
        """
        Register an option series with the coordinator.

        Parses the series key to extract strike and expiry (if strike not provided),
        then adds to appropriate bucket and indices.

        Args:
            series_key: Series key like "AAPL_241220_C_150"
            strike: Strike price (optional, will be parsed from series_key if not provided)

        Raises:
            ValueError: If series key format is invalid and strike not provided
        """
        if series_key in self._series_to_bucket:
            return  # Already registered

        # Parse series key
        parts = series_key.split("_")

        # Get strike from parameter or parse from key
        if strike is not None:
            strike_val = float(strike)
        elif len(parts) >= 4:
            try:
                strike_val = float(parts[3])
            except ValueError:
                raise ValueError(f"Invalid strike in series key: {series_key}")
        else:
            raise ValueError(
                f"Invalid series key format: {series_key}. "
                f"Expected: UNDERLYING_YYMMDD_C/P_STRIKE or provide strike parameter"
            )

        # Get expiry from key or use default
        expiry = parts[1] if len(parts) > 1 else self._expiry or "000000"

        # Calculate bucket ID
        bucket_id = int(strike_val / self._bucket_width)

        # Create bucket if needed
        if bucket_id not in self._buckets:
            self._buckets[bucket_id] = StrikeBucket(
                bucket_id=bucket_id,
                bucket_width=self._bucket_width,
            )

        # Add to bucket
        self._buckets[bucket_id].add_series(series_key, expiry)
        self._series_to_bucket[series_key] = bucket_id
        self._strike_to_bucket[strike_val] = bucket_id

        # Add to expiry index
        self._expiry_index[expiry].add(series_key)

        # Update stats
        self._stats.series_count = len(self._series_to_bucket)
        self._stats.buckets_count = len(self._buckets)

    def unregister_series(self, series_key: str) -> None:
        """
        Unregister an option series.

        Args:
            series_key: Series key to unregister
        """
        if series_key not in self._series_to_bucket:
            return

        # Parse expiry
        parts = series_key.split("_")
        expiry = parts[1] if len(parts) > 1 else ""

        # Remove from bucket
        bucket_id = self._series_to_bucket[series_key]
        if bucket_id in self._buckets:
            self._buckets[bucket_id].remove_series(series_key, expiry)

            # Clean up empty bucket
            if self._buckets[bucket_id].series_count == 0:
                del self._buckets[bucket_id]

        del self._series_to_bucket[series_key]

        # Remove from expiry index
        if expiry in self._expiry_index:
            self._expiry_index[expiry].discard(series_key)
            if not self._expiry_index[expiry]:
                del self._expiry_index[expiry]

        # Update stats
        self._stats.series_count = len(self._series_to_bucket)
        self._stats.buckets_count = len(self._buckets)

    def is_registered(self, series_key: str) -> bool:
        """Check if series is registered."""
        return series_key in self._series_to_bucket

    def get_registered_series(self) -> List[str]:
        """Get all registered series."""
        return list(self._series_to_bucket.keys())

    def get_series_by_expiry(self, expiry: str) -> List[str]:
        """Get all series for a specific expiry."""
        return list(self._expiry_index.get(expiry, set()))

    def get_series_by_strike_range(
        self,
        min_strike: float,
        max_strike: float,
    ) -> List[str]:
        """Get all series within a strike range."""
        min_bucket = int(min_strike / self._bucket_width)
        max_bucket = int(max_strike / self._bucket_width)

        result = []
        for bucket_id in range(min_bucket, max_bucket + 1):
            if bucket_id in self._buckets:
                result.extend(self._buckets[bucket_id].get_all_series())

        return result

    def get_bucket_for_strike(
        self,
        strike: Union[float, Decimal],
    ) -> Optional[int]:
        """
        Get the bucket ID for a given strike price.

        Args:
            strike: Strike price

        Returns:
            Bucket ID or None if no bucket exists for this strike
        """
        strike_val = float(strike)
        bucket_id = int(strike_val / self._bucket_width)
        return bucket_id if bucket_id in self._buckets else None

    def register_callback(
        self,
        series_key: str,
        callback: Callable[[str, OptionsEvent], None],
    ) -> None:
        """
        Register a callback for events affecting a specific series.

        Args:
            series_key: Series key to register callback for
            callback: Callback function(series_key, event)
        """
        self._per_series_callbacks[series_key].append(callback)

    def unregister_callback(
        self,
        series_key: str,
        callback: Optional[Callable[[str, OptionsEvent], None]] = None,
    ) -> None:
        """
        Unregister callbacks for a series.

        Args:
            series_key: Series key
            callback: Specific callback to remove, or None to remove all
        """
        if series_key not in self._per_series_callbacks:
            return

        if callback is None:
            del self._per_series_callbacks[series_key]
        else:
            try:
                self._per_series_callbacks[series_key].remove(callback)
            except ValueError:
                pass

    def set_atm_strike(self, strike: Union[float, Decimal]) -> None:
        """
        Set the current at-the-money strike for ATM_ONLY scope.

        Args:
            strike: ATM strike price
        """
        self._atm_strike = float(strike)

    # ==========================================================================
    # Event Propagation
    # ==========================================================================

    def propagate(
        self,
        event: OptionsEvent,
        scope: Optional[PropagationScope] = None,
    ) -> PropagationResult:
        """
        Propagate an event to affected series (new API with scope override).

        This is a convenience wrapper around propagate_event that allows
        overriding the event's scope.

        Args:
            event: Event to propagate
            scope: Override scope (optional, uses event.scope if not provided)

        Returns:
            PropagationResult with affected series list
        """
        # Update ATM strike from underlying price if provided
        if event.underlying_price is not None:
            self._atm_strike = float(event.underlying_price)

        # Override scope if provided
        if scope is not None:
            # Create a copy with updated scope
            event = OptionsEvent(
                event_type=event.event_type,
                source_series=event.source_series,
                quote=event.quote,
                underlying_price=event.underlying_price,
                volatility_change=event.volatility_change,
                timestamp_ns=event.timestamp_ns,
                scope=scope,
                custom_filter=event.custom_filter,
                series_key=event.series_key,
                trade_price=event.trade_price,
                trade_qty=event.trade_qty,
                implied_vol=event.implied_vol,
                delta=event.delta,
                gamma=event.gamma,
                theta=event.theta,
                vega=event.vega,
                timestamp=event.timestamp,
            )

        return self.propagate_event(event)

    def propagate_batch(
        self,
        events: List[OptionsEvent],
    ) -> List[PropagationResult]:
        """
        Propagate a batch of events.

        Args:
            events: List of events to propagate

        Returns:
            List of PropagationResult objects
        """
        return [self.propagate_event(event) for event in events]

    def propagate_event(self, event: OptionsEvent) -> PropagationResult:
        """
        Propagate an event to affected series.

        Determines which series are affected based on the event's
        scope and source series, then optionally calls the on_event
        callback for each affected series.

        Args:
            event: Event to propagate

        Returns:
            PropagationResult with affected series list
        """
        start_time = time.time_ns()

        # Get affected series based on scope
        affected = self._get_affected_series(event)

        # Track buckets checked for stats
        buckets_checked = self._count_buckets_checked(event)

        # Call global callback
        if self._on_event:
            for series_key in affected:
                try:
                    self._on_event(series_key, event)
                except Exception as e:
                    logger.warning(f"Event callback failed for {series_key}: {e}")

        # Call per-series callbacks
        for series_key in affected:
            if series_key in self._per_series_callbacks:
                for callback in self._per_series_callbacks[series_key]:
                    try:
                        callback(series_key, event)
                    except Exception as e:
                        logger.warning(f"Per-series callback failed for {series_key}: {e}")

        end_time = time.time_ns()
        propagation_time = end_time - start_time
        # Ensure at least 1 ns (Windows timer resolution can be low)
        if propagation_time < 1:
            propagation_time = 1

        # Update stats
        self._update_stats(event, len(affected), propagation_time)

        return PropagationResult(
            affected_series=list(affected),
            propagation_time_ns=propagation_time,
            num_buckets_checked=buckets_checked,
            source_series=event.source_series,
            event_type=event.event_type,
        )

    def propagate_quote_update(
        self,
        source_series: str,
        quote: OptionsQuote,
    ) -> PropagationResult:
        """
        Propagate a quote update event.

        Convenience method for the most common event type.

        Args:
            source_series: Series that was updated
            quote: New quote data

        Returns:
            PropagationResult
        """
        event = OptionsEvent(
            event_type=OptionsEventType.QUOTE_UPDATE,
            source_series=source_series,
            quote=quote,
            scope=PropagationScope.NEARBY_STRIKES,
        )
        return self.propagate_event(event)

    def propagate_underlying_move(
        self,
        underlying: str,
        new_price: float,
    ) -> PropagationResult:
        """
        Propagate an underlying price move to all affected options.

        This affects all series for the underlying across all expiries.

        Args:
            underlying: Underlying symbol
            new_price: New underlying price

        Returns:
            PropagationResult
        """
        # Find all series for this underlying
        affected = set()
        for series_key in self._series_to_bucket:
            if series_key.startswith(underlying + "_"):
                affected.add(series_key)

        start_time = time.time_ns()

        event = OptionsEvent(
            event_type=OptionsEventType.UNDERLYING_MOVE,
            source_series=f"{underlying}_underlying",
            underlying_price=new_price,
            scope=PropagationScope.ALL_EXPIRIES,
        )

        if self._on_event:
            for series_key in affected:
                try:
                    self._on_event(series_key, event)
                except Exception as e:
                    logger.warning(f"Event callback failed for {series_key}: {e}")

        end_time = time.time_ns()
        propagation_time = end_time - start_time
        # Ensure at least 1 ns (Windows timer resolution can be low)
        if propagation_time < 1:
            propagation_time = 1

        self._update_stats(event, len(affected), propagation_time)

        return PropagationResult(
            affected_series=list(affected),
            propagation_time_ns=propagation_time,
            num_buckets_checked=len(self._buckets),
            source_series=event.source_series,
            event_type=event.event_type,
        )

    # ==========================================================================
    # Affected Series Calculation
    # ==========================================================================

    def _get_affected_series(self, event: OptionsEvent) -> Set[str]:
        """Determine which series are affected by an event."""
        if event.scope == PropagationScope.LOCAL:
            return {event.source_series} if self.is_registered(event.source_series) else set()

        elif event.scope == PropagationScope.NEARBY_STRIKES:
            return self._get_nearby_series(event.source_series)

        elif event.scope == PropagationScope.ADJACENT:
            # ADJACENT means ±1 bucket, not full nearby_buckets range
            return self._get_adjacent_series(event.source_series)

        elif event.scope == PropagationScope.SAME_EXPIRY:
            return self._get_same_expiry_series(event.source_series)

        elif event.scope == PropagationScope.ALL_EXPIRIES or event.scope == PropagationScope.ALL:
            # ALL scope: return all registered series if no source, else underlying
            if not event.source_series:
                return set(self._series_to_bucket.keys())
            return self._get_all_underlying_series(event.source_series)

        elif event.scope == PropagationScope.ATM_ONLY:
            return self._get_atm_series()

        elif event.scope == PropagationScope.CUSTOM:
            if event.custom_filter:
                return {s for s in self._series_to_bucket if event.custom_filter(s)}
            return set()

        return set()

    def _get_adjacent_series(self, source_series: str) -> Set[str]:
        """Get series in immediately adjacent strike buckets (±1 bucket only)."""
        if source_series not in self._series_to_bucket:
            return set()

        source_bucket = self._series_to_bucket[source_series]
        affected: Set[str] = set()

        # Include source bucket and ±1 adjacent buckets
        for offset in [-1, 0, 1]:
            bucket_id = source_bucket + offset
            if bucket_id in self._buckets:
                affected.update(self._buckets[bucket_id].get_all_series())

        return affected

    def _get_atm_series(self) -> Set[str]:
        """Get series at or near the current ATM strike."""
        if self._atm_strike is None:
            return set()

        atm_bucket = int(float(self._atm_strike) / self._bucket_width)
        if atm_bucket in self._buckets:
            return set(self._buckets[atm_bucket].get_all_series())
        return set()

    def _get_nearby_series(self, source_series: str) -> Set[str]:
        """Get series in nearby strike buckets."""
        if source_series not in self._series_to_bucket:
            return set()

        source_bucket = self._series_to_bucket[source_series]
        affected: Set[str] = set()

        # Include source bucket and nearby buckets
        for offset in range(-self._nearby_buckets, self._nearby_buckets + 1):
            bucket_id = source_bucket + offset
            if bucket_id in self._buckets:
                affected.update(self._buckets[bucket_id].get_all_series())

        # Optionally include same-expiry series in other buckets
        if self._same_expiry_propagation:
            parts = source_series.split("_")
            if len(parts) > 1:
                expiry = parts[1]
                affected.update(self._expiry_index.get(expiry, set()))

        return affected

    def _get_same_expiry_series(self, source_series: str) -> Set[str]:
        """Get all series with the same expiry."""
        parts = source_series.split("_")
        if len(parts) < 2:
            return {source_series} if self.is_registered(source_series) else set()

        expiry = parts[1]
        return self._expiry_index.get(expiry, set()).copy()

    def _get_all_underlying_series(self, source_series: str) -> Set[str]:
        """Get all series for the underlying."""
        parts = source_series.split("_")
        underlying = parts[0] if parts else ""

        return {s for s in self._series_to_bucket if s.startswith(underlying + "_")}

    def _count_buckets_checked(self, event: OptionsEvent) -> int:
        """Count number of buckets that would be checked."""
        if event.scope == PropagationScope.LOCAL:
            return 1
        elif event.scope == PropagationScope.NEARBY_STRIKES:
            return min(len(self._buckets), 2 * self._nearby_buckets + 1)
        else:
            return len(self._buckets)

    # ==========================================================================
    # Statistics
    # ==========================================================================

    def _update_stats(
        self,
        event: OptionsEvent,
        affected_count: int,
        propagation_time_ns: int,
    ) -> None:
        """Update coordinator statistics."""
        self._stats.total_events += 1
        self._stats.total_propagations += affected_count

        # Update average affected series
        n = self._stats.total_events
        old_avg = self._stats.avg_affected_series
        self._stats.avg_affected_series = (old_avg * (n - 1) + affected_count) / n

        # Update average propagation time
        old_time = self._stats.avg_propagation_time_ns
        self._stats.avg_propagation_time_ns = (old_time * (n - 1) + propagation_time_ns) / n

        # Update peak
        if affected_count > self._stats.peak_affected_series:
            self._stats.peak_affected_series = affected_count

        # Update by-type counts
        if event.event_type not in self._stats.events_by_type:
            self._stats.events_by_type[event.event_type] = 0
        self._stats.events_by_type[event.event_type] += 1

    def get_stats(self) -> CoordinatorStats:
        """Get coordinator statistics."""
        self._stats.buckets_count = len(self._buckets)
        self._stats.series_count = len(self._series_to_bucket)
        return CoordinatorStats(
            total_events=self._stats.total_events,
            total_propagations=self._stats.total_propagations,
            avg_affected_series=self._stats.avg_affected_series,
            avg_propagation_time_ns=self._stats.avg_propagation_time_ns,
            peak_affected_series=self._stats.peak_affected_series,
            events_by_type=self._stats.events_by_type.copy(),
            buckets_count=self._stats.buckets_count,
            series_count=self._stats.series_count,
        )

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = CoordinatorStats()
        self._stats.buckets_count = len(self._buckets)
        self._stats.series_count = len(self._series_to_bucket)

    # ==========================================================================
    # Bulk Operations
    # ==========================================================================

    def register_chain(
        self,
        underlying: str,
        expiries: List[str],
        strikes: List[float],
        include_calls: bool = True,
        include_puts: bool = True,
    ) -> int:
        """
        Register an entire option chain.

        Args:
            underlying: Underlying symbol
            expiries: List of expiry dates (YYMMDD)
            strikes: List of strike prices
            include_calls: Include call options
            include_puts: Include put options

        Returns:
            Number of series registered
        """
        count = 0

        for expiry in expiries:
            for strike in strikes:
                if include_calls:
                    key = f"{underlying}_{expiry}_C_{strike:g}"
                    self.register_series(key)
                    count += 1

                if include_puts:
                    key = f"{underlying}_{expiry}_P_{strike:g}"
                    self.register_series(key)
                    count += 1

        logger.info(f"Registered {count} series for {underlying}")
        return count

    def clear(self) -> None:
        """Clear all registered series."""
        self._buckets.clear()
        self._series_to_bucket.clear()
        self._expiry_index.clear()
        self.reset_stats()


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_event_coordinator(
    underlying: Optional[str] = None,
    bucket_width: float = 5.0,
    nearby_buckets: int = 2,
    same_expiry_propagation: bool = True,
    on_event: Optional[Callable[[str, OptionsEvent], None]] = None,
    expiry: Optional[str] = None,
    max_propagation_depth: Optional[int] = None,
) -> EventDrivenLOBCoordinator:
    """
    Create an event-driven LOB coordinator.

    Args:
        underlying: Underlying symbol (e.g., "SPY")
        bucket_width: Width of strike buckets (default: $5)
        nearby_buckets: Number of buckets each side to propagate to
        same_expiry_propagation: Propagate to same-expiry series
        on_event: Callback when event is propagated
        expiry: Default expiry date (YYMMDD)
        max_propagation_depth: Maximum propagation depth

    Returns:
        Configured EventDrivenLOBCoordinator
    """
    return EventDrivenLOBCoordinator(
        bucket_width=bucket_width,
        nearby_buckets=nearby_buckets,
        same_expiry_propagation=same_expiry_propagation,
        on_event=on_event,
        underlying=underlying,
        expiry=expiry,
        max_propagation_depth=max_propagation_depth,
    )


def create_options_coordinator(
    underlying: str = "SPY",
    expiry: Optional[str] = None,
    atm_strike: float = 450.0,
    strike_step: float = 1.0,
    num_strikes: int = 50,
    expiries: Optional[List[str]] = None,
    bucket_width: float = 5.0,
) -> EventDrivenLOBCoordinator:
    """
    Create a coordinator pre-populated with an options chain.

    Args:
        underlying: Underlying symbol
        expiry: Default expiry (YYMMDD format) - if provided, used for simple form
        atm_strike: At-the-money strike price
        strike_step: Step between strikes
        num_strikes: Number of strikes (each side of ATM)
        expiries: List of expiries (default: 4 weekly)
        bucket_width: Width of strike buckets

    Returns:
        Configured EventDrivenLOBCoordinator with chain registered
    """
    coordinator = create_event_coordinator(
        underlying=underlying,
        expiry=expiry,
        bucket_width=bucket_width,
        nearby_buckets=2,
    )

    # Handle single expiry case (simple API)
    if expiry is not None and expiries is None:
        expiries = [expiry]
    # Default expiries (4 weekly)
    elif expiries is None:
        expiries = ["241220", "241227", "250103", "250110"]

    # Generate strikes
    strikes = []
    for i in range(-num_strikes, num_strikes + 1):
        strikes.append(atm_strike + i * strike_step)

    # Register chain
    coordinator.register_chain(
        underlying=underlying,
        expiries=expiries,
        strikes=strikes,
        include_calls=True,
        include_puts=True,
    )

    return coordinator


# ==============================================================================
# Exports
# ==============================================================================

__all__ = [
    # Enums
    "OptionsEventType",
    "PropagationScope",
    # Data classes
    "OptionsQuote",
    "OptionsEvent",
    "PropagationResult",
    "CoordinatorStats",
    "StrikeBucket",
    # Main class
    "EventDrivenLOBCoordinator",
    # Factory functions
    "create_event_coordinator",
    "create_options_coordinator",
]
