# L3 LOB Simulator - Latency Simulation

## Overview

Realistic latency simulation is critical for modeling HFT and institutional execution. The L3 simulator provides configurable latency profiles with time-of-day and volatility adjustments.

**Files**:

- `lob/latency_model.py` (~997 lines) - Latency distributions
- `lob/event_scheduler.py` (~970 lines) - Event ordering

## Latency Components

| Component | Description | Typical Range |
|-----------|-------------|---------------|
| Feed Latency | Market data reception | 10μs - 10ms |
| Order Latency | Order submission to exchange | 50μs - 50ms |
| Exchange Latency | Exchange processing time | 10μs - 1ms |
| Fill Latency | Fill notification reception | 20μs - 10ms |

```
Time →
[Market Event] --feed_latency--> [We See It]
                                      |
                                 [Decision]
                                      |
                             --order_latency-->
                                              [Exchange Receives]
                                                     |
                                              --exchange_latency-->
                                                                 [Match/Execute]
                                                                        |
                                                              --fill_latency-->
                                                                              [We Know]
```

## Latency Profiles

### Co-located (~10-50 μs)

HFT firms with servers in exchange data centers.

```python
from lob import LatencyModel, LatencyProfile

model = LatencyModel.from_profile(LatencyProfile.COLOCATED)

# Typical values:
# Feed: 10-30 μs
# Order: 15-50 μs
# Exchange: 5-20 μs
# Fill: 10-30 μs
# Round-trip: ~50-150 μs
```

### Proximity (~100-500 μs)

Firms with nearby data center presence.

```python
model = LatencyModel.from_profile(LatencyProfile.PROXIMITY)

# Typical values:
# Feed: 100-300 μs
# Order: 150-400 μs
# Exchange: 20-100 μs
# Fill: 100-200 μs
# Round-trip: ~400-1000 μs
```

### Institutional (~200 μs - 2 ms)

Institutional investors with good connectivity.

```python
model = LatencyModel.from_profile(LatencyProfile.INSTITUTIONAL)

# Typical values:
# Feed: 200-500 μs
# Order: 300-800 μs
# Exchange: 50-150 μs
# Fill: 150-400 μs
# Round-trip: ~700-2000 μs
```

### Retail (~1-10 ms)

Retail investors using broker APIs.

```python
model = LatencyModel.from_profile(LatencyProfile.RETAIL)

# Typical values:
# Feed: 1-5 ms
# Order: 2-10 ms
# Exchange: 100-500 μs
# Fill: 1-3 ms
# Round-trip: ~5-20 ms
```

## Latency Distributions

### Log-normal (Default)

Most realistic for network latencies - heavy right tail.

```python
from lob import LatencyDistribution, LatencyConfig

config = LatencyConfig(
    distribution=LatencyDistribution.LOGNORMAL,
    mean_us=200.0,
    std_us=50.0,
    min_us=50.0,
    max_us=2000.0,
)

model = LatencyModel(feed_latency=config, ...)
```

### Pareto (Heavy Tail)

For environments with occasional extreme delays.

```python
config = LatencyConfig(
    distribution=LatencyDistribution.PARETO,
    mean_us=200.0,
    alpha=3.0,  # Shape parameter (lower = heavier tail)
    min_us=50.0,
    max_us=5000.0,
)
```

### Gamma

Smoother distribution with configurable shape.

```python
config = LatencyConfig(
    distribution=LatencyDistribution.GAMMA,
    mean_us=200.0,
    shape=4.0,  # Shape parameter
    min_us=50.0,
    max_us=2000.0,
)
```

### Constant

Fixed latency for deterministic testing.

```python
config = LatencyConfig(
    distribution=LatencyDistribution.CONSTANT,
    mean_us=100.0,
)
```

## Using the Latency Model

```python
from lob import LatencyModel, create_latency_model

# Create from profile
model = create_latency_model("institutional", seed=42)

# Sample latencies (returns nanoseconds)
feed_lat = model.sample_feed_latency()      # Market data latency
order_lat = model.sample_order_latency()    # Order submission latency
exch_lat = model.sample_exchange_latency()  # Exchange processing
fill_lat = model.sample_fill_latency()      # Fill notification latency

print(f"Feed: {feed_lat/1000:.1f} μs")
print(f"Order: {order_lat/1000:.1f} μs")

# Full round-trip
round_trip = model.sample_round_trip()
print(f"Round-trip: {round_trip/1000:.1f} μs")

# Get statistics
stats = model.stats()
print(f"Feed p50: {stats['feed']['p50_us']:.1f} μs")
print(f"Feed p95: {stats['feed']['p95_us']:.1f} μs")
print(f"Feed p99: {stats['feed']['p99_us']:.1f} μs")
```

## Latency Adjustments

### Time-of-Day

Latency varies throughout the trading day.

```python
model = create_latency_model(
    "institutional",
    time_of_day_adjustment=True,
)

# Higher latency during market open/close
# 9:30-10:00 AM: +30% latency
# 3:30-4:00 PM: +20% latency
# Mid-day: baseline

# Sample with time context
lat = model.sample_feed_latency(hour_of_day=9, minute=35)  # Higher
lat = model.sample_feed_latency(hour_of_day=12, minute=0)  # Baseline
```

### Volatility

Higher volatility increases latency (more market activity).

```python
model = create_latency_model(
    "institutional",
    volatility_adjustment=True,
)

# Sample with volatility context
lat = model.sample_feed_latency(volatility=0.01)  # Low vol - lower latency
lat = model.sample_feed_latency(volatility=0.05)  # High vol - higher latency
```

### Spike Events

Occasional latency spikes (network congestion, etc.).

```python
from lob import LatencyConfig

config = LatencyConfig(
    distribution=LatencyDistribution.LOGNORMAL,
    mean_us=200.0,
    std_us=50.0,
    spike_prob=0.001,    # 0.1% chance of spike
    spike_mult=10.0,     # Spike = 10x normal latency
)
```

## Event Scheduler

Order events in correct timestamp order despite latency.

```python
from lob import EventScheduler, create_event_scheduler, EventType

# Create scheduler
scheduler = create_event_scheduler("institutional", seed=42)

# Schedule market data event
from lob import MarketDataEvent

md_event = MarketDataEvent(
    symbol="AAPL",
    exchange_time_ns=1_000_000,
    bid_price=150.0,
    ask_price=150.05,
    bid_qty=1000.0,
    ask_qty=1200.0,
)
our_receive_time = scheduler.schedule_market_data(md_event, exchange_time_ns=1_000_000)
print(f"We see market data at: {our_receive_time} ns")

# Schedule our order submission
from lob import LimitOrder, Side

order = LimitOrder(
    order_id="order_1",
    price=150.02,
    qty=100.0,
    remaining_qty=100.0,
    timestamp_ns=our_receive_time,  # We decide after seeing data
    side=Side.BUY,
)
arrival_at_exchange = scheduler.schedule_order_arrival(
    order,
    our_send_time_ns=our_receive_time + 1000,  # 1μs decision time
)

# Process events in order
for event in scheduler:
    print(f"Event: {event.event_type.name} at {event.timestamp_ns} ns")
    if event.event_type == EventType.MARKET_DATA:
        # Process market data
        pass
    elif event.event_type == EventType.ORDER_ARRIVAL:
        # Order arrives at exchange
        pass
    elif event.event_type == EventType.FILL:
        # Fill notification
        pass
```

## Race Condition Detection

Detect when event ordering is ambiguous due to latency.

```python
from lob import RaceConditionInfo

# Get race condition info
race_info = scheduler.detect_race_conditions(buffer_us=100.0)

for info in race_info:
    print(f"Race between {info.event1_id} and {info.event2_id}")
    print(f"Time difference: {info.time_diff_ns} ns")
    print(f"Uncertain winner: {info.uncertain}")
```

## Simulation Clock

Track simulation time with latency awareness.

```python
from lob import SimulationClock, create_simulation_clock

clock = create_simulation_clock(start_time_ns=0)

# Advance time
clock.advance_to(1_000_000)

# Get current time
print(f"Current time: {clock.current_time_ns} ns")

# Track pending events
clock.add_pending_event("order_1", arrival_time_ns=1_500_000)
pending = clock.get_pending_events(until_ns=2_000_000)
```

## YAML Configuration

```yaml
# configs/execution_l3.yaml
latency:
  enabled: true
  profile: institutional

  feed_latency:
    enabled: true
    distribution: lognormal
    mean_us: 200.0
    std_us: 50.0
    min_us: 50.0
    max_us: 2000.0
    spike_prob: 0.001
    spike_mult: 10.0

  order_latency:
    enabled: true
    distribution: lognormal
    mean_us: 300.0
    std_us: 80.0

  exchange_latency:
    enabled: true
    distribution: lognormal
    mean_us: 100.0
    std_us: 30.0

  fill_latency:
    enabled: true
    distribution: lognormal
    mean_us: 150.0
    std_us: 40.0

  time_of_day_adjustment: true
  volatility_adjustment: true
```

## Performance

| Operation | Complexity | Typical Latency |
|-----------|------------|-----------------|
| Sample latency | O(1) | <1 μs |
| Schedule event | O(log n) | <10 μs |
| Process event | O(log n) | <10 μs |
| Race detection | O(n²) | <1 ms |

## Deterministic Seeding

For reproducible simulations:

```python
model = create_latency_model("institutional", seed=42)

# Same seed = same latency sequence
lat1 = model.sample_feed_latency()  # e.g., 234.5 μs
lat2 = model.sample_feed_latency()  # e.g., 189.2 μs

# Reset seed
model.reset_seed(42)
lat3 = model.sample_feed_latency()  # 234.5 μs (same as lat1)
```

## Related Documentation

- [Overview](overview.md) - Architecture overview
- [Configuration](configuration.md) - Config reference
- [Event Scheduler](../../lob/event_scheduler.py) - Event ordering
