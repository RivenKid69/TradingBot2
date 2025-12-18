# L3 LOB Simulator - Matching Engine

## Overview

The matching engine implements FIFO Price-Time Priority matching (CME Globex style) with optional Pro-Rata matching for options markets. It includes comprehensive Self-Trade Prevention (STP) functionality.

**File**: `lob/matching_engine.py` (~844 lines)

## Quick Start

```python
from lob import MatchingEngine, OrderBook, LimitOrder, Side, STPAction

# Create matching engine
engine = MatchingEngine(
    symbol="AAPL",
    enable_stp=True,
    stp_action=STPAction.CANCEL_NEWEST,
)

# Submit orders
result = engine.submit_order(order)
print(f"Match type: {result.match_type}")
print(f"Fills: {result.fills}")
```

## Factory Functions

```python
from lob import create_matching_engine

# Default FIFO engine
engine = create_matching_engine("AAPL")

# With STP
engine = create_matching_engine(
    symbol="AAPL",
    enable_stp=True,
    stp_action="cancel_oldest",  # or STPAction.CANCEL_OLDEST
)

# Pro-Rata engine (for options)
engine = create_matching_engine(
    symbol="SPY_OPTION",
    matching_type="pro_rata",
    pro_rata_min_allocation=1.0,
)
```

## Matching Algorithms

### FIFO (Price-Time Priority)

The default matching algorithm used by most equity exchanges.

```
Order Priority:
1. Price (best price first)
2. Time (earliest timestamp first at same price)
```

**Example:**

```python
# Book state:
# Bids: [order1 @ 150.00 (100), order2 @ 150.00 (50), order3 @ 149.95 (200)]
#       order1 arrived before order2

# Incoming SELL MARKET 120 units:
# 1. Match order1: 100 units @ 150.00 (fully filled)
# 2. Match order2: 20 units @ 150.00 (partially filled)
# Result: 120 units filled, order2 has 30 remaining

result = engine.submit_order(LimitOrder(
    order_id="sell_1",
    price=0.0,  # Market order
    qty=120.0,
    remaining_qty=120.0,
    timestamp_ns=...,
    side=Side.SELL,
))
```

### Pro-Rata Matching

Used by some options exchanges (CME options).

```
Pro-Rata allocation:
- Each order at best price gets allocation proportional to its size
- Minimum allocation threshold applies
```

```python
from lob import ProRataMatchingEngine

engine = ProRataMatchingEngine(
    symbol="SPY_OPTION",
    min_allocation=1.0,  # Minimum 1 contract per fill
)

# Book state:
# Bids: [order1 @ 5.00 (100), order2 @ 5.00 (100)]

# Incoming SELL 100 units:
# order1 allocation: 100 * (100/200) = 50
# order2 allocation: 100 * (100/200) = 50
```

## Self-Trade Prevention (STP)

Prevents orders from the same trader from matching against each other.

### STP Actions

| Action | Behavior |
|--------|----------|
| `CANCEL_NEWEST` | Cancel incoming (aggressive) order |
| `CANCEL_OLDEST` | Cancel resting (passive) order |
| `CANCEL_BOTH` | Cancel both orders |
| `DECREMENT_AND_CANCEL` | Reduce quantities, cancel smaller |

```python
from lob import MatchingEngine, STPAction

engine = MatchingEngine(
    symbol="AAPL",
    enable_stp=True,
    stp_action=STPAction.CANCEL_NEWEST,
)

# Both orders from same trader
order1 = LimitOrder(
    order_id="bid_1",
    price=150.0,
    qty=100.0,
    remaining_qty=100.0,
    timestamp_ns=1000,
    side=Side.BUY,
    trader_id="trader_A",  # Same trader
)

order2 = LimitOrder(
    order_id="sell_1",
    price=149.90,
    qty=50.0,
    remaining_qty=50.0,
    timestamp_ns=2000,
    side=Side.SELL,
    trader_id="trader_A",  # Same trader - would match with order1
)

engine.submit_order(order1)
result = engine.submit_order(order2)

# With CANCEL_NEWEST: order2 is cancelled, no match
print(result.stp_triggered)  # True
print(result.stp_cancelled_orders)  # ["sell_1"]
```

### STP Configuration

```python
# Per-engine STP
engine = MatchingEngine(
    symbol="AAPL",
    enable_stp=True,
    stp_action=STPAction.DECREMENT_AND_CANCEL,
)

# Disable STP for specific traders
engine.set_stp_exemption("market_maker_1", exempt=True)
```

## MatchResult

The result returned from order submission.

```python
from lob import MatchResult, MatchType

result = engine.submit_order(order)

# Match type
result.match_type  # MatchType.FULL, PARTIAL, NONE, STP_CANCELLED

# Fills from this order
result.fills  # List[Fill]

# Total filled quantity
result.filled_qty  # float

# Average fill price
result.avg_price  # float or None

# Remaining quantity (if partial or none)
result.remaining_qty  # float

# STP information
result.stp_triggered  # bool
result.stp_cancelled_orders  # List[str]

# Order was added to book (limit order, not fully matched)
result.added_to_book  # bool
```

### MatchType Enum

| Type | Description |
|------|-------------|
| `FULL` | Order completely filled |
| `PARTIAL` | Order partially filled, remainder on book |
| `NONE` | No fill, order added to book |
| `STP_CANCELLED` | Order cancelled due to STP |
| `REJECTED` | Order rejected (validation failure) |

## Order Types

### Limit Order

```python
# Limit order - added to book if not immediately matched
result = engine.submit_order(LimitOrder(
    order_id="order_1",
    price=150.0,  # Limit price
    qty=100.0,
    remaining_qty=100.0,
    timestamp_ns=...,
    side=Side.BUY,
))
```

### Market Order

```python
# Market order - immediate execution only
# Use price=0 for BUY (will match any ask)
# Use price=float('inf') for SELL (will match any bid)
result = engine.submit_order(LimitOrder(
    order_id="mkt_1",
    price=0.0,  # Market order (BUY)
    qty=100.0,
    remaining_qty=100.0,
    timestamp_ns=...,
    side=Side.BUY,
))
```

## Order Management

### Cancel Order

```python
# Cancel by order ID
cancelled = engine.cancel_order("order_123")
print(f"Cancelled: {cancelled}")  # bool

# Bulk cancel
engine.cancel_all_orders(trader_id="trader_A")
```

### Modify Order

```python
# Modify (cancel + replace)
# Note: Loses time priority
new_order = engine.modify_order(
    order_id="order_123",
    new_price=151.0,
    new_qty=150.0,
)
```

## Book State Access

```python
# Access underlying order book
book = engine.order_book

# Best prices
best_bid = engine.best_bid()
best_ask = engine.best_ask()
spread = engine.spread()
mid = engine.mid_price()

# Depth
depth = engine.get_depth(levels=10)
# {"bids": [...], "asks": [...]}

# Order lookup
order = engine.get_order("order_123")

# Statistics
stats = engine.get_statistics()
# {
#     "total_orders": 1000,
#     "total_fills": 500,
#     "total_volume": 50000.0,
#     "avg_spread": 0.05,
#     ...
# }
```

## Event Callbacks

```python
def on_fill(fill):
    print(f"Fill: {fill.order_id} @ {fill.price} x {fill.qty}")

def on_order_added(order):
    print(f"Order added: {order.order_id}")

def on_order_cancelled(order_id, reason):
    print(f"Order cancelled: {order_id}, reason: {reason}")

engine = MatchingEngine(
    symbol="AAPL",
    on_fill=on_fill,
    on_order_added=on_order_added,
    on_order_cancelled=on_order_cancelled,
)
```

## Integration with Queue Tracker

```python
from lob import MatchingEngine, QueuePositionTracker, create_queue_tracker

engine = MatchingEngine(symbol="AAPL")
tracker = create_queue_tracker()

# Track queue position when order added
def on_order_added(order):
    level_qty = engine.order_book.get_qty_at_price(order.price, order.side)
    tracker.add_order(order, level_qty_before=level_qty - order.remaining_qty)

engine.on_order_added = on_order_added

# Update tracker on fills
def on_fill(fill):
    tracker.on_execution(
        order_id=fill.order_id,
        executed_qty=fill.qty,
        price=fill.price,
    )

engine.on_fill = on_fill
```

## Performance

| Operation | Complexity | Typical Latency |
|-----------|------------|-----------------|
| Submit order (no match) | O(log n) | <50 μs |
| Submit order (match) | O(k log n) | <100 μs |
| Cancel order | O(log n) | <20 μs |
| Best bid/ask | O(1) | <1 μs |

Where n = number of price levels, k = number of fills.

## Thread Safety

The matching engine is **NOT thread-safe**. For multi-threaded use:

```python
import threading

lock = threading.Lock()

def submit_order_safe(order):
    with lock:
        return engine.submit_order(order)
```

## Related Documentation

- [Data Structures](data_structures.md) - Core LOB structures
- [Queue Position](queue_position.md) - Position tracking
- [Order Manager](../../lob/order_manager.py) - Order lifecycle management
