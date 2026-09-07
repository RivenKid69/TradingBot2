"""
Event scheduler for L3 LOB with latency simulation.

This module manages event timing with realistic latency modeling,
ensuring correct ordering of:
1. Market data arrival (with feed latency)
2. Our order submission (with order latency)
3. Fill notifications (with fill latency)

Key Features:
- Priority queue for event ordering by timestamp
- Race condition handling (our order vs market data)
- Support for event batching
- Integration with MatchingEngine and OrderBook

Reference: hftbacktest event handling
https://hftbacktest.readthedocs.io/en/latest/

Timestamp Convention:
    All timestamps use NANOSECONDS for consistency with LOB module.

Stage 5 of L3 LOB Simulation (v5.0)
"""

from __future__ import annotations

import heapq
import threading
import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from lob.latency_model import LatencyModel, LatencyProfile, create_latency_model
from lob.data_structures import LimitOrder, Side, Fill, Trade


class EventType(IntEnum):
    """Types of events in the scheduler."""

    # Market data events (from exchange to us)
    MARKET_DATA_UPDATE = 1  # Price/quantity update
    TRADE_REPORT = 2  # Public trade occurred
    BOOK_SNAPSHOT = 3  # Full book snapshot
    ORDER_ADD = 4  # New order added to book
    ORDER_CANCEL = 5  # Order cancelled
    ORDER_MODIFY = 6  # Order modified

    # Our order events (from us to exchange, then back)
    ORDER_SUBMITTED = 10  # We submitted an order
    ORDER_RECEIVED = 11  # Exchange received our order
    ORDER_ACCEPTED = 12  # Exchange accepted our order
    ORDER_REJECTED = 13  # Exchange rejected our order

    # Fill events
    OUR_FILL = 20  # Our order was filled
    OUR_PARTIAL_FILL = 21  # Our order partially filled

    # Timer events
    TIMER = 30  # Timer callback
    END_OF_DATA = 31  # No more data


@dataclass
class ScheduledEvent:
    """Event in the scheduler priority queue.

    Events are ordered by (timestamp_ns, priority, sequence_id) to ensure
    deterministic ordering when timestamps are equal.

    Ordering rules:
        1. Earlier timestamp_ns wins
        2. For same timestamp: lower priority value wins (0 = highest priority)
        3. For same timestamp AND priority: lower sequence_id wins (FIFO)

    Attributes:
        timestamp_ns: When this event should be processed (our local time)
        sequence_id: Unique sequence number for FIFO ordering
        event_type: Type of event
        exchange_time_ns: Original exchange timestamp
        payload: Event-specific data
        callback: Optional callback to execute
        priority: Lower = higher priority (for same timestamp, default=0)
    """

    timestamp_ns: int
    sequence_id: int
    event_type: EventType
    exchange_time_ns: int
    payload: Any = None
    callback: Optional[Callable[["ScheduledEvent"], None]] = None
    priority: int = 0

    def __post_init__(self) -> None:
        """Validate event."""
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")

    def _sort_key(self) -> Tuple[int, int, int]:
        """Return sort key: (timestamp_ns, priority, sequence_id)."""
        return (self.timestamp_ns, self.priority, self.sequence_id)

    def __lt__(self, other: "ScheduledEvent") -> bool:
        """Compare events for priority queue ordering."""
        if not isinstance(other, ScheduledEvent):
            return NotImplemented
        return self._sort_key() < other._sort_key()

    def __le__(self, other: "ScheduledEvent") -> bool:
        """Compare events for priority queue ordering."""
        if not isinstance(other, ScheduledEvent):
            return NotImplemented
        return self._sort_key() <= other._sort_key()

    def __gt__(self, other: "ScheduledEvent") -> bool:
        """Compare events for priority queue ordering."""
        if not isinstance(other, ScheduledEvent):
            return NotImplemented
        return self._sort_key() > other._sort_key()

    def __ge__(self, other: "ScheduledEvent") -> bool:
        """Compare events for priority queue ordering."""
        if not isinstance(other, ScheduledEvent):
            return NotImplemented
        return self._sort_key() >= other._sort_key()

    @property
    def latency_ns(self) -> int:
        """Get latency (our time - exchange time)."""
        return self.timestamp_ns - self.exchange_time_ns


@dataclass
class OrderSubmission:
    """Our order submission with timing info.

    Attributes:
        order: The order we're submitting
        our_send_time_ns: When we sent the order
        exchange_receive_time_ns: When exchange receives it (computed)
        round_trip_latency_ns: Total round-trip latency
    """

    order: LimitOrder
    our_send_time_ns: int
    exchange_receive_time_ns: int = 0
    round_trip_latency_ns: int = 0


@dataclass
class MarketDataEvent:
    """Market data event payload.

    Attributes:
        symbol: Trading symbol
        exchange_time_ns: When event occurred on exchange
        our_receive_time_ns: When we received it
        bid_price: Best bid price (if applicable)
        ask_price: Best ask price (if applicable)
        bid_qty: Best bid quantity
        ask_qty: Best ask quantity
        last_price: Last trade price (for trade events)
        last_qty: Last trade quantity
        is_snapshot: Whether this is a full snapshot
    """

    symbol: str
    exchange_time_ns: int
    our_receive_time_ns: int = 0
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_qty: float = 0.0
    ask_qty: float = 0.0
    last_price: Optional[float] = None
    last_qty: float = 0.0
    is_snapshot: bool = False


@dataclass
class FillEvent:
    """Fill notification payload.

    Attributes:
        order_id: ID of filled order
        fill: Fill details
        exchange_time_ns: When fill occurred on exchange
        our_receive_time_ns: When we received notification
    """

    order_id: str
    fill: Fill
    exchange_time_ns: int
    our_receive_time_ns: int = 0


@dataclass
class RaceConditionInfo:
    """Information about a detected race condition.

    Attributes:
        our_event: Our order submission event
        market_event: Market data event
        our_exchange_time_ns: When our order reaches exchange
        market_exchange_time_ns: When market event occurred
        race_delta_ns: Time difference (positive = we arrived first)
        description: Human-readable description
    """

    our_event: ScheduledEvent
    market_event: ScheduledEvent
    our_exchange_time_ns: int
    market_exchange_time_ns: int
    race_delta_ns: int
    description: str = ""


# Type for event handlers
EventHandler = Callable[[ScheduledEvent], None]


class EventScheduler:
    """
    Manages event timing with latency for LOB simulation.

    Ensures correct ordering of:
    1. Market data arrival (with feed latency)
    2. Our order submission (with order latency)
    3. Fill notifications (with fill latency)

    Thread Safety:
        - All public methods are thread-safe
        - Uses internal lock for priority queue access

    Race Condition Handling:
        The scheduler can detect and report race conditions where:
        - Our order might arrive at exchange before/after relevant market data
        - Multiple events have ordering ambiguity due to latency

    Example:
        >>> model = LatencyModel.from_profile(LatencyProfile.INSTITUTIONAL)
        >>> scheduler = EventScheduler(latency_model=model)
        >>>
        >>> # Schedule market data
        >>> our_time = scheduler.schedule_market_data(
        ...     MarketDataEvent(symbol="AAPL", exchange_time_ns=1000000),
        ...     exchange_time_ns=1000000,
        ... )
        >>>
        >>> # Schedule our order
        >>> arrival_time = scheduler.schedule_order_arrival(
        ...     order=order,
        ...     our_send_time_ns=1000100,
        ... )
    """

    # Default maximum number of race conditions to track
    DEFAULT_MAX_RACE_CONDITIONS = 10000

    def __init__(
        self,
        latency_model: Optional[LatencyModel] = None,
        profile: Union[str, LatencyProfile] = "institutional",
        seed: Optional[int] = None,
        detect_race_conditions: bool = True,
        on_race_condition: Optional[Callable[[RaceConditionInfo], None]] = None,
        on_event: Optional[EventHandler] = None,
        max_race_conditions: int = DEFAULT_MAX_RACE_CONDITIONS,
    ) -> None:
        """Initialize event scheduler.

        Args:
            latency_model: Latency model to use (creates one if not provided)
            profile: Latency profile if no model provided
            seed: Random seed for latency sampling
            detect_race_conditions: Whether to detect and report race conditions
            on_race_condition: Callback for race condition detection
            on_event: Default callback for event processing
            max_race_conditions: Maximum number of race conditions to track (default 10000).
                                 Oldest race conditions are dropped when limit is reached.
        """
        self._lock = threading.Lock()

        # Initialize latency model
        if latency_model is not None:
            self._latency_model = latency_model
        else:
            self._latency_model = create_latency_model(profile, seed=seed)

        # Priority queue: (timestamp_ns, priority, sequence_id, event)
        self._queue: List[ScheduledEvent] = []
        self._sequence_counter = 0

        # Race condition detection with bounded memory
        self._detect_races = detect_race_conditions
        self._on_race_condition = on_race_condition
        self._pending_orders: Dict[str, OrderSubmission] = {}
        self._max_race_conditions = max(1, max_race_conditions)
        self._race_conditions: Deque[RaceConditionInfo] = deque(maxlen=self._max_race_conditions)

        # Event handling
        self._on_event = on_event
        self._event_handlers: Dict[EventType, List[EventHandler]] = {}

        # Current simulation time
        self._current_time_ns = 0

        # Statistics
        self._total_events = 0
        self._processed_events = 0
        self._detected_races = 0

    @property
    def latency_model(self) -> LatencyModel:
        """Get the latency model."""
        return self._latency_model

    @property
    def current_time_ns(self) -> int:
        """Get current simulation time in nanoseconds."""
        return self._current_time_ns

    @property
    def pending_count(self) -> int:
        """Get number of pending events."""
        with self._lock:
            return len(self._queue)

    def register_handler(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> None:
        """Register event handler for specific event type.

        Args:
            event_type: Type of events to handle
            handler: Callback function
        """
        with self._lock:
            if event_type not in self._event_handlers:
                self._event_handlers[event_type] = []
            self._event_handlers[event_type].append(handler)

    def unregister_handler(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> bool:
        """Unregister event handler.

        Returns:
            True if handler was found and removed
        """
        with self._lock:
            if event_type in self._event_handlers:
                try:
                    self._event_handlers[event_type].remove(handler)
                    return True
                except ValueError:
                    pass
        return False

    def _next_sequence_id(self) -> int:
        """Get next sequence ID (not thread-safe, called within lock)."""
        seq = self._sequence_counter
        self._sequence_counter += 1
        return seq

    def schedule_market_data(
        self,
        event: MarketDataEvent,
        exchange_time_ns: int,
        callback: Optional[EventHandler] = None,
    ) -> int:
        """Schedule market data event with feed latency.

        Args:
            event: Market data event
            exchange_time_ns: When event occurred on exchange
            callback: Optional callback for this event

        Returns:
            our_receive_time_ns: When we'll receive this data
        """
        # Sample feed latency
        feed_latency = self._latency_model.sample_feed_latency()
        our_receive_time = exchange_time_ns + feed_latency

        # Update event
        event.our_receive_time_ns = our_receive_time

        # Determine event type
        if event.is_snapshot:
            event_type = EventType.BOOK_SNAPSHOT
        elif event.last_price is not None:
            event_type = EventType.TRADE_REPORT
        else:
            event_type = EventType.MARKET_DATA_UPDATE

        # Create scheduled event
        with self._lock:
            scheduled = ScheduledEvent(
                timestamp_ns=our_receive_time,
                sequence_id=self._next_sequence_id(),
                event_type=event_type,
                exchange_time_ns=exchange_time_ns,
                payload=event,
                callback=callback,
            )
            heapq.heappush(self._queue, scheduled)
            self._total_events += 1

            # Check for race conditions with pending orders
            if self._detect_races:
                self._check_race_conditions(scheduled)

        return our_receive_time

    def schedule_order_arrival(
        self,
        order: LimitOrder,
        our_send_time_ns: int,
        callback: Optional[EventHandler] = None,
    ) -> int:
        """Schedule our order arrival at exchange.

        Args:
            order: Order to submit
            our_send_time_ns: When we're sending the order
            callback: Optional callback for acceptance event

        Returns:
            exchange_receive_time_ns: When exchange will receive our order
        """
        # Sample order latency (one-way to exchange)
        order_latency = self._latency_model.sample_order_latency()
        exchange_receive_time = our_send_time_ns + order_latency

        # Sample exchange processing time
        exchange_latency = self._latency_model.sample_exchange_latency()

        # Sample fill/confirmation latency (back to us)
        fill_latency = self._latency_model.sample_fill_latency()

        # Total round-trip
        round_trip = order_latency + exchange_latency + fill_latency

        # Track order submission
        submission = OrderSubmission(
            order=order,
            our_send_time_ns=our_send_time_ns,
            exchange_receive_time_ns=exchange_receive_time,
            round_trip_latency_ns=round_trip,
        )

        with self._lock:
            # Schedule submission event
            submit_event = ScheduledEvent(
                timestamp_ns=our_send_time_ns,
                sequence_id=self._next_sequence_id(),
                event_type=EventType.ORDER_SUBMITTED,
                exchange_time_ns=our_send_time_ns,
                payload=submission,
            )
            heapq.heappush(self._queue, submit_event)

            # Schedule received event (when exchange gets it)
            received_event = ScheduledEvent(
                timestamp_ns=exchange_receive_time + exchange_latency,
                sequence_id=self._next_sequence_id(),
                event_type=EventType.ORDER_RECEIVED,
                exchange_time_ns=exchange_receive_time,
                payload=submission,
            )
            heapq.heappush(self._queue, received_event)

            # Schedule accepted event (confirmation back to us)
            confirmation_time = our_send_time_ns + round_trip
            accepted_event = ScheduledEvent(
                timestamp_ns=confirmation_time,
                sequence_id=self._next_sequence_id(),
                event_type=EventType.ORDER_ACCEPTED,
                exchange_time_ns=exchange_receive_time + exchange_latency,
                payload=submission,
                callback=callback,
            )
            heapq.heappush(self._queue, accepted_event)

            self._total_events += 3
            self._pending_orders[order.order_id] = submission

        return exchange_receive_time

    def schedule_fill_notification(
        self,
        fill: Fill,
        exchange_time_ns: int,
        is_partial: bool = False,
        callback: Optional[EventHandler] = None,
    ) -> int:
        """Schedule fill notification with latency.

        Args:
            fill: Fill to notify
            exchange_time_ns: When fill occurred on exchange
            is_partial: Whether this is a partial fill
            callback: Optional callback for this event

        Returns:
            our_receive_time_ns: When we'll receive notification
        """
        # Sample fill notification latency
        fill_latency = self._latency_model.sample_fill_latency()
        our_receive_time = exchange_time_ns + fill_latency

        # Create fill event payload
        fill_event = FillEvent(
            order_id=fill.order_id,
            fill=fill,
            exchange_time_ns=exchange_time_ns,
            our_receive_time_ns=our_receive_time,
        )

        # Determine event type
        event_type = EventType.OUR_PARTIAL_FILL if is_partial else EventType.OUR_FILL

        with self._lock:
            scheduled = ScheduledEvent(
                timestamp_ns=our_receive_time,
                sequence_id=self._next_sequence_id(),
                event_type=event_type,
                exchange_time_ns=exchange_time_ns,
                payload=fill_event,
                callback=callback,
            )
            heapq.heappush(self._queue, scheduled)
            self._total_events += 1

            # Remove from pending if complete fill
            if not is_partial and fill.order_id in self._pending_orders:
                del self._pending_orders[fill.order_id]

        return our_receive_time

    def schedule_timer(
        self,
        trigger_time_ns: int,
        callback: EventHandler,
        payload: Any = None,
    ) -> None:
        """Schedule a timer callback.

        Args:
            trigger_time_ns: When to trigger
            callback: Function to call
            payload: Optional data to pass
        """
        with self._lock:
            scheduled = ScheduledEvent(
                timestamp_ns=trigger_time_ns,
                sequence_id=self._next_sequence_id(),
                event_type=EventType.TIMER,
                exchange_time_ns=trigger_time_ns,
                payload=payload,
                callback=callback,
            )
            heapq.heappush(self._queue, scheduled)
            self._total_events += 1

    def schedule_custom(
        self,
        timestamp_ns: int,
        event_type: EventType,
        exchange_time_ns: int,
        payload: Any = None,
        callback: Optional[EventHandler] = None,
        priority: int = 0,
    ) -> None:
        """Schedule a custom event.

        Args:
            timestamp_ns: When to process this event
            event_type: Type of event
            exchange_time_ns: Original exchange timestamp
            payload: Event data
            callback: Optional callback
            priority: Priority (lower = higher priority)
        """
        with self._lock:
            scheduled = ScheduledEvent(
                timestamp_ns=timestamp_ns,
                sequence_id=self._next_sequence_id(),
                event_type=event_type,
                exchange_time_ns=exchange_time_ns,
                payload=payload,
                callback=callback,
                priority=priority,
            )
            heapq.heappush(self._queue, scheduled)
            self._total_events += 1

    def _check_race_conditions(self, new_event: ScheduledEvent) -> None:
        """Check for race conditions with pending orders (not thread-safe)."""
        if not self._pending_orders:
            return

        for order_id, submission in self._pending_orders.items():
            # Check if market event could affect order execution
            if new_event.event_type in (
                EventType.MARKET_DATA_UPDATE,
                EventType.TRADE_REPORT,
                EventType.ORDER_ADD,
                EventType.ORDER_CANCEL,
            ):
                # Race condition: market event vs our order arrival
                our_arrival = submission.exchange_receive_time_ns
                market_exchange = new_event.exchange_time_ns
                race_delta = our_arrival - market_exchange

                # Race if events are within 1ms of each other at exchange
                if abs(race_delta) < 1_000_000:  # 1ms in nanoseconds
                    race_info = RaceConditionInfo(
                        our_event=ScheduledEvent(
                            timestamp_ns=submission.our_send_time_ns
                            + submission.round_trip_latency_ns,
                            sequence_id=-1,
                            event_type=EventType.ORDER_ACCEPTED,
                            exchange_time_ns=submission.exchange_receive_time_ns,
                            payload=submission,
                        ),
                        market_event=new_event,
                        our_exchange_time_ns=our_arrival,
                        market_exchange_time_ns=market_exchange,
                        race_delta_ns=race_delta,
                        description=(
                            f"Order {order_id} arrival at {our_arrival} ns "
                            f"vs market event at {market_exchange} ns "
                            f"(delta: {race_delta} ns)"
                        ),
                    )
                    self._race_conditions.append(race_info)
                    self._detected_races += 1

                    if self._on_race_condition is not None:
                        self._on_race_condition(race_info)

    def peek(self) -> Optional[ScheduledEvent]:
        """Peek at next event without removing it."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue[0]

    def pop(self) -> Optional[ScheduledEvent]:
        """Pop and return next event.

        Returns:
            Next event or None if queue is empty
        """
        with self._lock:
            if not self._queue:
                return None
            event = heapq.heappop(self._queue)
            self._current_time_ns = event.timestamp_ns
            return event

    def process_next(self) -> Optional[ScheduledEvent]:
        """Process next event and call handlers.

        Returns:
            Processed event or None if queue is empty
        """
        event = self.pop()
        if event is None:
            return None

        self._processed_events += 1

        # Call event-specific callback
        if event.callback is not None:
            event.callback(event)

        # Call registered handlers
        with self._lock:
            handlers = self._event_handlers.get(event.event_type, [])

        for handler in handlers:
            handler(event)

        # Call default handler
        if self._on_event is not None:
            self._on_event(event)

        return event

    def process_until(self, until_time_ns: int) -> List[ScheduledEvent]:
        """Process all events up to a given time.

        Args:
            until_time_ns: Process events with timestamp <= this

        Returns:
            List of processed events
        """
        processed = []
        while True:
            event = self.peek()
            if event is None or event.timestamp_ns > until_time_ns:
                break
            processed.append(self.process_next())
        return processed

    def process_all(self) -> List[ScheduledEvent]:
        """Process all pending events.

        Returns:
            List of all processed events
        """
        processed = []
        while self.pending_count > 0:
            event = self.process_next()
            if event is not None:
                processed.append(event)
        return processed

    def clear(self) -> None:
        """Clear all pending events."""
        with self._lock:
            self._queue.clear()
            self._pending_orders.clear()

    def advance_time(self, new_time_ns: int) -> None:
        """Advance current time without processing events.

        Args:
            new_time_ns: New current time
        """
        with self._lock:
            if new_time_ns > self._current_time_ns:
                self._current_time_ns = new_time_ns

    def get_race_conditions(self) -> List[RaceConditionInfo]:
        """Get detected race conditions."""
        with self._lock:
            return list(self._race_conditions)

    def clear_race_conditions(self) -> None:
        """Clear detected race conditions."""
        with self._lock:
            self._race_conditions.clear()

    def stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        latency_stats = self._latency_model.stats()

        with self._lock:
            return {
                "total_events": self._total_events,
                "processed_events": self._processed_events,
                "pending_events": len(self._queue),
                "pending_orders": len(self._pending_orders),
                "detected_races": self._detected_races,
                "tracked_races": len(self._race_conditions),
                "max_race_conditions": self._max_race_conditions,
                "current_time_ns": self._current_time_ns,
                "latency": latency_stats,
            }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._latency_model.reset_stats()
        with self._lock:
            self._total_events = 0
            self._processed_events = 0
            self._detected_races = 0
            self._race_conditions.clear()

    def __iter__(self) -> Iterator[ScheduledEvent]:
        """Iterate over events (processing each one)."""
        while True:
            event = self.process_next()
            if event is None:
                break
            yield event

    def __len__(self) -> int:
        """Get number of pending events."""
        return self.pending_count


class SimulationClock:
    """
    Simulation clock with latency-aware time tracking.

    Provides a unified view of time for:
    - Our local time (when we see events)
    - Exchange time (when events actually occurred)

    Thread Safety:
        All methods are thread-safe.

    Example:
        >>> clock = SimulationClock()
        >>> clock.set_exchange_time(1000000)  # Exchange is at 1ms
        >>> our_time = clock.exchange_to_local(1000000)  # With feed latency
    """

    def __init__(
        self,
        latency_model: Optional[LatencyModel] = None,
        initial_time_ns: int = 0,
    ) -> None:
        """Initialize simulation clock.

        Args:
            latency_model: Model for latency sampling
            initial_time_ns: Initial time
        """
        self._lock = threading.Lock()
        self._latency_model = latency_model or create_latency_model("institutional")
        self._exchange_time_ns = initial_time_ns
        self._local_time_ns = initial_time_ns

    @property
    def exchange_time_ns(self) -> int:
        """Get current exchange time."""
        with self._lock:
            return self._exchange_time_ns

    @property
    def local_time_ns(self) -> int:
        """Get current local time (with latency offset)."""
        with self._lock:
            return self._local_time_ns

    def set_exchange_time(self, time_ns: int) -> None:
        """Set exchange time (advances local time by feed latency)."""
        feed_latency = self._latency_model.sample_feed_latency()
        with self._lock:
            self._exchange_time_ns = time_ns
            self._local_time_ns = time_ns + feed_latency

    def advance(self, delta_ns: int) -> None:
        """Advance both clocks by delta."""
        with self._lock:
            self._exchange_time_ns += delta_ns
            self._local_time_ns += delta_ns

    def exchange_to_local(self, exchange_time_ns: int) -> int:
        """Convert exchange time to local time with feed latency.

        Args:
            exchange_time_ns: Time on exchange

        Returns:
            Estimated local time when we'd see this
        """
        feed_latency = self._latency_model.sample_feed_latency()
        return exchange_time_ns + feed_latency

    def local_to_exchange(self, local_time_ns: int) -> int:
        """Convert local time to estimated exchange time.

        This is an approximation since we can't know exact latency
        without the actual event.

        Args:
            local_time_ns: Our local time

        Returns:
            Estimated exchange time
        """
        # Use mean feed latency as approximation
        feed_stats = self._latency_model._feed_sampler.config
        mean_latency_ns = int(feed_stats.mean_us * 1000)
        return local_time_ns - mean_latency_ns

    def order_arrival_time(self, send_time_ns: int) -> int:
        """Get estimated exchange arrival time for order sent at send_time.

        Args:
            send_time_ns: When we send the order

        Returns:
            When it arrives at exchange
        """
        order_latency = self._latency_model.sample_order_latency()
        return send_time_ns + order_latency

    def get_round_trip_time(self) -> int:
        """Get sampled round-trip latency for order.

        Returns:
            Round-trip time in nanoseconds
        """
        return self._latency_model.sample_round_trip()


# Factory functions
def create_event_scheduler(
    profile: Union[str, LatencyProfile] = "institutional",
    seed: Optional[int] = None,
    detect_race_conditions: bool = True,
) -> EventScheduler:
    """Create an event scheduler with specified latency profile.

    Args:
        profile: Latency profile name or enum
        seed: Random seed
        detect_race_conditions: Whether to detect races

    Returns:
        Configured EventScheduler
    """
    model = create_latency_model(profile, seed=seed)
    return EventScheduler(
        latency_model=model,
        detect_race_conditions=detect_race_conditions,
    )


def create_simulation_clock(
    profile: Union[str, LatencyProfile] = "institutional",
    seed: Optional[int] = None,
    initial_time_ns: int = 0,
) -> SimulationClock:
    """Create a simulation clock with specified latency profile.

    Args:
        profile: Latency profile name or enum
        seed: Random seed
        initial_time_ns: Initial time

    Returns:
        Configured SimulationClock
    """
    model = create_latency_model(profile, seed=seed)
    return SimulationClock(
        latency_model=model,
        initial_time_ns=initial_time_ns,
    )


__all__ = [
    "EventType",
    "ScheduledEvent",
    "OrderSubmission",
    "MarketDataEvent",
    "FillEvent",
    "RaceConditionInfo",
    "EventHandler",
    "EventScheduler",
    "SimulationClock",
    "create_event_scheduler",
    "create_simulation_clock",
]
