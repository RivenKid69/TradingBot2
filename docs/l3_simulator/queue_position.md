# L3 LOB Simulator - Queue Position Tracking

## Overview

Queue position tracking estimates where an order sits in the execution queue at a price level. This is critical for realistic limit order fill simulation, as queue position determines fill probability.

**Files**:
- `lob/queue_tracker.py` (~766 lines)
- `lob/order_manager.py` (~1,013 lines)

## Queue Position Estimation Methods

### MBO (Market-by-Order) - Exact

When individual order data is available, exact queue position is known.

```python
from lob import QueuePositionTracker, PositionEstimationMethod

tracker = QueuePositionTracker(method=PositionEstimationMethod.MBO)

# Add order with exact position (orders_ahead list)
state = tracker.add_order(
    order,
    orders_ahead=["order_1", "order_2", "order_3"],  # 3 orders ahead
)
print(f"Queue position: {state.position}")  # 4 (1-indexed)
```

### MBP (Market-by-Price) - Estimated

When only price-level aggregated data is available (most common).

```python
tracker = QueuePositionTracker(method=PositionEstimationMethod.MBP)

# Add order with level quantity before our order
state = tracker.add_order(
    order,
    level_qty_before=500.0,  # 500 units ahead of us
)
print(f"Queue position (estimated): {state.position}")
print(f"Quantity ahead: {state.qty_ahead}")  # 500.0
```

### Pessimistic Estimation (Default)

Assumes we're at the back of the queue - most conservative.

```python
tracker = QueuePositionTracker(method=PositionEstimationMethod.PESSIMISTIC)

# Assumes max queue position
state = tracker.add_order(order, level_qty_before=500.0)
# Position = total level qty / avg order size (pessimistic)
```

### Probabilistic Estimation

Uses statistical modeling to estimate position.

```python
tracker = QueuePositionTracker(method=PositionEstimationMethod.PROBABILISTIC)

# Uses arrival rate statistics to estimate position distribution
state = tracker.add_order(order, level_qty_before=500.0)
print(f"Position range: [{state.position_min}, {state.position_max}]")
print(f"Confidence: {state.confidence}")
```

## QueueState

The state object returned by queue tracking operations.

```python
from lob import QueueState

state = tracker.add_order(order, level_qty_before=500.0)

# Basic position info
state.order_id        # Order being tracked
state.position        # Queue position (1 = front)
state.qty_ahead       # Total quantity ahead
state.level_qty       # Total level quantity

# Timestamps
state.added_ns        # When added to queue
state.last_update_ns  # Last position update

# MBP-specific
state.estimated       # True if estimated (MBP)
state.confidence      # Confidence in estimate (0-1)
state.position_min    # Min possible position
state.position_max    # Max possible position
```

## Queue Position Updates

### On Execution (Fill at Level)

```python
# When trades occur at our price level
tracker.on_execution(
    order_id="our_order",
    executed_qty=100.0,
    price=150.0,
)

# Queue position advances
state = tracker.get_state("our_order")
print(f"New position: {state.position}")  # Decreased
print(f"Qty ahead: {state.qty_ahead}")    # Decreased by 100
```

### On Cancellation

```python
# When orders ahead of us cancel
tracker.on_cancellation(
    order_id="other_order",
    cancelled_qty=50.0,
    price=150.0,
)

# Our position improves
state = tracker.get_state("our_order")
```

### On Order Add (Behind Us)

```python
# Orders added behind us don't affect our position
tracker.on_order_added(
    order_id="new_order",
    qty=100.0,
    price=150.0,
    timestamp_ns=...,
)

# Our position unchanged
```

## Fill Probability Calculation

The tracker provides fill probability estimates based on queue position.

```python
# Get fill probability for a time horizon
prob = tracker.estimate_fill_probability(
    order_id="our_order",
    volume_per_second=100.0,  # Expected execution rate
    time_horizon_sec=60.0,    # 1 minute horizon
)

print(f"P(fill in 60s): {prob.probability:.2%}")
print(f"Expected time to fill: {prob.expected_fill_time_sec:.1f}s")
```

### Poisson Fill Model

```python
from lob import FillProbability

# Analytical Poisson model
# P(fill in T) = 1 - exp(-λT / position)
prob = FillProbability.poisson(
    arrival_rate=100.0,      # Volume/second at level
    queue_position=5,        # Our position
    time_horizon_sec=60.0,
)
```

## Order Manager Integration

The OrderManager provides full lifecycle management including queue tracking.

```python
from lob import OrderManager, TimeInForce, Side

manager = OrderManager(symbol="AAPL")

# Submit with automatic queue tracking
managed = manager.submit_order(
    side=Side.BUY,
    price=150.0,
    qty=100.0,
    order_type=OrderType.LIMIT,
    time_in_force=TimeInForce.DAY,
    track_queue=True,  # Enable queue tracking
)

# Access queue state
print(f"Queue position: {managed.queue_state.position}")

# Get fill probability
prob = manager.get_fill_probability("order_id")
```

### Time-in-Force Support

| TIF | Behavior |
|-----|----------|
| `DAY` | Cancel at end of day |
| `GTC` | Good-til-cancelled |
| `IOC` | Immediate-or-cancel |
| `FOK` | Fill-or-kill |

```python
# IOC: Must fill immediately or cancel
managed = manager.submit_order(
    side=Side.BUY,
    price=150.0,
    qty=100.0,
    time_in_force=TimeInForce.IOC,
)

if managed.state == OrderLifecycleState.CANCELLED:
    print("IOC order not filled immediately")
```

## Level Statistics

Track statistics per price level for better estimation.

```python
from lob import LevelStatistics

stats = tracker.get_level_statistics(price=150.0, side=Side.BUY)

print(f"Avg queue length: {stats.avg_queue_length}")
print(f"Avg time in queue: {stats.avg_time_in_queue_sec}s")
print(f"Fill rate: {stats.fill_rate_per_sec}")
print(f"Cancel rate: {stats.cancel_rate_per_sec}")
```

## Factory Functions

```python
from lob import create_queue_tracker, create_order_manager

# Create tracker with specific method
tracker = create_queue_tracker(
    method="mbo",  # or "mbp", "pessimistic", "probabilistic"
    max_tracked_orders=10000,
)

# Create order manager with tracker
manager = create_order_manager(
    symbol="AAPL",
    enable_queue_tracking=True,
    queue_method="mbp",
)
```

## Callbacks

```python
def on_position_change(order_id, old_pos, new_pos, reason):
    print(f"{order_id}: position {old_pos} -> {new_pos} ({reason})")

def on_probable_fill(order_id, probability, time_horizon):
    print(f"{order_id}: {probability:.0%} likely to fill in {time_horizon}s")

tracker = QueuePositionTracker(
    on_position_change=on_position_change,
    on_probable_fill=on_probable_fill,
    probable_fill_threshold=0.90,  # Callback when >90% likely
)
```

## Performance

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Add order | O(1) | Hash table insert |
| Update position | O(1) | Direct update |
| Get state | O(1) | Hash table lookup |
| Fill probability | O(1) | Analytical formula |
| Level statistics | O(1) | Pre-computed |

## Memory Management

```python
# Configure max tracked orders
tracker = QueuePositionTracker(
    max_tracked_orders=10000,
    cleanup_interval_sec=60.0,  # Auto-cleanup old orders
)

# Manual cleanup
tracker.cleanup_completed_orders()

# Clear all
tracker.clear()
```

## Integration Example

```python
from lob import (
    MatchingEngine,
    QueuePositionTracker,
    create_matching_engine,
    create_queue_tracker,
)

# Setup
engine = create_matching_engine("AAPL")
tracker = create_queue_tracker(method="mbp")

# Integrate
def on_order_added(order):
    level_qty = engine.order_book.get_qty_at_price(order.price, order.side)
    tracker.add_order(order, level_qty_before=level_qty - order.remaining_qty)

def on_fill(fill):
    tracker.on_execution(fill.order_id, fill.qty, fill.price)

def on_cancel(order_id, reason):
    tracker.remove_order(order_id)

engine.on_order_added = on_order_added
engine.on_fill = on_fill
engine.on_order_cancelled = on_cancel

# Submit order and track
order = LimitOrder(...)
result = engine.submit_order(order)

if result.added_to_book:
    state = tracker.get_state(order.order_id)
    prob = tracker.estimate_fill_probability(
        order.order_id,
        volume_per_second=100.0,
        time_horizon_sec=60.0,
    )
    print(f"Queue position: {state.position}")
    print(f"Fill probability: {prob.probability:.2%}")
```

## Related Documentation

- [Matching Engine](matching_engine.md) - How orders are matched
- [Fill Probability](../../lob/fill_probability.py) - Probability models
- [Queue Value](../../lob/queue_value.py) - Queue position valuation
