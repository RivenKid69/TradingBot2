# L3 LOB Simulator - Data Structures

## Overview

The L3 LOB simulator uses a carefully designed set of data structures to represent order book state efficiently while maintaining O(1) or O(log n) complexity for critical operations.

**File**: `lob/data_structures.py` (~1,150 lines)

## Core Enums

### Side

```python
from lob import Side

Side.BUY   # Bid side
Side.SELL  # Ask side
```

### OrderType

```python
from lob import OrderType

OrderType.LIMIT   # Limit order (price specified)
OrderType.MARKET  # Market order (immediate execution)
```

## LimitOrder

The fundamental order representation.

```python
from lob import LimitOrder, Side

order = LimitOrder(
    order_id="order_123",
    price=150.0,
    qty=100.0,
    remaining_qty=100.0,
    timestamp_ns=1699900000000000000,  # Nanoseconds
    side=Side.BUY,
    trader_id="trader_A",              # Optional, for STP
    is_hidden=False,                   # Hidden order flag
    display_qty=None,                  # Iceberg display quantity
)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `order_id` | str | Unique identifier |
| `price` | float | Limit price |
| `qty` | float | Original quantity |
| `remaining_qty` | float | Unfilled quantity |
| `timestamp_ns` | int | Submission timestamp (nanoseconds) |
| `side` | Side | BUY or SELL |
| `trader_id` | Optional[str] | For self-trade prevention |
| `is_hidden` | bool | Hidden order flag |
| `display_qty` | Optional[float] | Iceberg display size |

### Methods

```python
# Check if order is completely filled
order.is_filled()  # bool

# Check if order is iceberg
order.is_iceberg()  # bool

# Get visible quantity (display_qty if iceberg, else remaining_qty)
order.visible_qty()  # float

# Fill partially
order.fill(qty=50.0, fill_price=150.0, timestamp_ns=..., trade_id="trade_1")
# Returns: Fill object

# Create copy with modifications
new_order = order.copy_with(remaining_qty=50.0)
```

## PriceLevel

A collection of orders at the same price, maintaining FIFO ordering.

```python
from lob import PriceLevel, Side

level = PriceLevel(price=150.0, side=Side.BUY)

# Add orders (FIFO order maintained)
level.add_order(order1)
level.add_order(order2)

# Properties
level.price          # 150.0
level.side           # Side.BUY
level.total_qty      # Sum of all remaining quantities
level.num_orders     # Count of orders
level.is_empty()     # True if no orders

# Access orders
for order in level.orders:
    print(order.order_id)

# Get first order (FIFO front)
first = level.first_order()

# Remove specific order
removed_order = level.remove_order("order_123")

# Match quantity (FIFO)
fills = level.match(qty=75.0, timestamp_ns=..., maker_fee=0.0002)
# Returns: List[Fill]
```

### Matching Behavior

```python
# Level with orders: [order1(50), order2(30), order3(20)]
# Match 75 units:
#   - order1: fully filled (50)
#   - order2: partially filled (25)
#   - order3: untouched
fills = level.match(qty=75.0, ...)
# fills[0].qty = 50.0 (order1)
# fills[1].qty = 25.0 (order2)
```

## OrderBook

Full order book with bid and ask sides.

```python
from lob import OrderBook

book = OrderBook(symbol="AAPL")

# Add limit orders
book.add_limit_order(order)

# Remove order
book.remove_order("order_123")

# Get best prices
best_bid = book.best_bid()   # float or None
best_ask = book.best_ask()   # float or None
spread = book.spread()       # float or None
mid_price = book.mid_price() # float or None

# Get top N levels
bid_levels = book.get_bid_levels(n=5)  # List[PriceLevel]
ask_levels = book.get_ask_levels(n=5)  # List[PriceLevel]

# Get quantity at price
qty = book.get_qty_at_price(price=150.0, side=Side.BUY)

# Get cumulative depth
depth = book.get_cumulative_depth(levels=10)
# Returns: {"bids": [(price, cum_qty), ...], "asks": [...]}

# Execute market order
fills = book.execute_market_order(
    side=Side.BUY,
    qty=100.0,
    timestamp_ns=...,
    taker_fee=0.0003,
)
# Returns: List[Fill]
```

### Internal Structure

```
OrderBook
├── bids: SortedDict[float, PriceLevel]  # Price descending
├── asks: SortedDict[float, PriceLevel]  # Price ascending
└── order_index: Dict[str, LimitOrder]   # O(1) lookup by order_id
```

## Fill

Execution result for a matched order.

```python
from lob import Fill

fill = Fill(
    order_id="order_123",
    trade_id="trade_456",
    price=150.0,
    qty=50.0,
    timestamp_ns=1699900001000000000,
    side=Side.BUY,
    is_maker=True,         # Maker (passive) or taker (aggressive)
    fee=0.015,             # Fee in quote currency
    counterparty_id=None,  # Optional counterparty
)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `order_id` | str | Order that was filled |
| `trade_id` | str | Unique trade identifier |
| `price` | float | Execution price |
| `qty` | float | Filled quantity |
| `timestamp_ns` | int | Fill timestamp |
| `side` | Side | Order side |
| `is_maker` | bool | Maker (True) or taker (False) |
| `fee` | float | Fee charged |
| `counterparty_id` | Optional[str] | Counterparty identifier |

### Methods

```python
# Get notional value
fill.notional()  # price * qty

# Get signed quantity (+BUY, -SELL)
fill.signed_qty()  # qty if BUY else -qty
```

## Trade

A completed trade (may involve multiple fills).

```python
from lob import Trade

trade = Trade(
    trade_id="trade_456",
    price=150.0,
    qty=100.0,
    timestamp_ns=1699900001000000000,
    aggressor_side=Side.BUY,  # Who initiated
    maker_order_id="order_123",
    taker_order_id="order_789",
)
```

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Add limit order | O(log n) | SortedDict insertion |
| Remove order | O(log n) | SortedDict removal |
| Lookup by order_id | O(1) | Hash table |
| Best bid/ask | O(1) | Cached |
| Market order match | O(k) | k = number of levels touched |
| Get level at price | O(log n) | Binary search |

## Memory Layout

```python
# Typical memory usage per component
LimitOrder:  ~200 bytes
PriceLevel:  ~100 bytes + orders
OrderBook:   ~1 KB base + levels + orders

# Example: 1000 orders across 100 levels
# ~200 KB for orders + ~10 KB for levels = ~210 KB
```

## Serialization

```python
# To dictionary
order_dict = order.to_dict()

# From dictionary
order = LimitOrder.from_dict(order_dict)

# OrderBook snapshot
snapshot = book.to_snapshot()
book = OrderBook.from_snapshot(snapshot)
```

## Thread Safety

The data structures are **NOT thread-safe** by design. For multi-threaded scenarios:

```python
import threading

lock = threading.Lock()

with lock:
    book.add_limit_order(order)
```

## Integration with LOBStateManager

For full state management with snapshots and message replay:

```python
from lob import LOBStateManager, MessageType

manager = LOBStateManager(symbol="AAPL")

# Add order via message
manager.process_message(LOBMessage(
    msg_type=MessageType.ADD,
    order_id="order_123",
    price=150.0,
    qty=100.0,
    side=Side.BUY,
    timestamp_ns=...,
))

# Take snapshot
snapshot = manager.create_snapshot()

# Restore from snapshot
manager.restore_from_snapshot(snapshot)
```

## Related Documentation

- [Matching Engine](matching_engine.md) - How orders are matched
- [Queue Position](queue_position.md) - Position tracking
- [State Manager](../../lob/state_manager.py) - Full state management
