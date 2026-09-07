# Deribit Crypto Options Integration

## Overview

This document describes the integration of Deribit (crypto options) into the CustodiaCloud codebase as an **optional digital-assets expansion path**. Deribit offers unique features compared to traditional equity options, primarily **inverse settlement** where P&L is denominated in the underlying cryptocurrency (BTC/ETH) rather than USD.

**Status**: ✅ Internal testing complete (not independently audited) | **Phase**: 2B | **Tests**: 120 internal tests (see CI for current status)

## Table of Contents

1. [Key Differences from US Equity Options](#key-differences-from-us-equity-options)
2. [Inverse Settlement Mechanics](#inverse-settlement-mechanics)
3. [Adapter Architecture](#adapter-architecture)
4. [Market Data Adapter](#market-data-adapter)
5. [Order Execution Adapter](#order-execution-adapter)
6. [Margin Calculation](#margin-calculation)
7. [WebSocket Streaming](#websocket-streaming)
8. [DVOL Index](#dvol-index)
9. [Configuration](#configuration)
10. [Usage Examples](#usage-examples)
11. [Testing](#testing)
12. [References](#references)

---

## Key Differences from US Equity Options

| Aspect | US Equity Options (IB/CBOE) | Deribit Crypto Options |
|--------|----------------------------|------------------------|
| **Settlement** | USD (linear) | Crypto (inverse) |
| **Exercise Style** | American/European | European only |
| **Trading Hours** | NYSE hours | 24/7 |
| **Volatility Index** | VIX (SPX based) | DVOL (30-day IV) |
| **Margin** | USD collateral | Crypto collateral |
| **Strike Increments** | Varies by underlying | BTC: $1000, ETH: $50 |
| **Expiration** | 3rd Friday of month | 08:00 UTC (various cycles) |
| **Min Order Size** | 1 contract | BTC: 0.1, ETH: 1.0 |

---

## Inverse Settlement Mechanics

### Core Concept

In **inverse settlement**, option payoffs are denominated in the underlying cryptocurrency, not USD. This creates unique risk/reward characteristics:

**Call Option Payoff (in crypto)**:

```
payoff_crypto = max(0, S - K) / S
```

**Put Option Payoff (in crypto)**:

```
payoff_crypto = max(0, K - S) / S
```

Where:

- `S` = Spot price at expiration
- `K` = Strike price

### Example: BTC Call Option

**Scenario**: BTC-28MAR25-100000-C (BTC $100,000 Call)

- Strike: $100,000
- Spot at expiration: $120,000

**Inverse Payoff**:

```python
payoff_crypto = max(0, 120000 - 100000) / 120000
payoff_crypto = 20000 / 120000
payoff_crypto = 0.1667 BTC  # ~$20,000 at current spot
```

**USD Equivalent**: `0.1667 × $120,000 = $20,000`

### Example: BTC Put Option

**Scenario**: BTC-28MAR25-50000-P (BTC $50,000 Put)

- Strike: $50,000
- Spot at expiration: $40,000

**Inverse Payoff**:

```python
payoff_crypto = max(0, 50000 - 40000) / 40000
payoff_crypto = 10000 / 40000
payoff_crypto = 0.25 BTC  # ~$10,000 at current spot
```

### "Double-Whammy" Risk

For **short positions**, inverse margining creates amplified risk:

1. If you're short a BTC call and BTC rallies:
   - Your position loses value (call goes ITM)
   - Your BTC collateral loses USD value

2. This creates a **double exposure** to adverse price moves

**Risk Mitigation**:

- Use portfolio margin for hedged positions
- Monitor margin ratios closely during high volatility
- Consider delta-neutral strategies

---

## Adapter Architecture

```
adapters/deribit/
├── __init__.py          # Package exports and documentation
├── options.py           # Market data and order execution adapters
├── margin.py            # Inverse margin calculator
└── websocket.py         # Real-time streaming client
```

### Component Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                    DeribitAPIClient                          │
│  - REST API calls (JSON-RPC 2.0)                            │
│  - Authentication (client_id/secret)                         │
│  - Rate limiting                                             │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ MarketData      │ │ OrderExecution  │ │ WebSocket       │
│ Adapter         │ │ Adapter         │ │ Client          │
│ - Option chains │ │ - Submit orders │ │ - Real-time     │
│ - Quotes/Greeks │ │ - Cancel orders │ │   quotes        │
│ - DVOL index    │ │ - Positions     │ │ - Order updates │
│ - Orderbook     │ │ - Order status  │ │ - Trade alerts  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                 ┌─────────────────────┐
                 │  DeribitMargin      │
                 │  Calculator         │
                 │  - Portfolio margin │
                 │  - Liquidation est. │
                 │  - Max position     │
                 └─────────────────────┘
```

---

## Market Data Adapter

### DeribitOptionsMarketDataAdapter

Provides access to option chains, quotes, Greeks, and the DVOL volatility index.

```python
from adapters.deribit import (
    DeribitOptionsMarketDataAdapter,
    create_deribit_options_market_data_adapter,
)
from datetime import date

# Create adapter (testnet by default)
adapter = create_deribit_options_market_data_adapter(testnet=True)

# Get option chain for BTC March 2025 expiration
chain = adapter.get_option_chain("BTC", date(2025, 3, 28))

for contract in chain:
    print(f"{contract.instrument_name}: {contract.strike} {contract.option_type.value}")

# Get quote with Greeks
quote = adapter.get_option_quote("BTC-28MAR25-100000-C")
print(f"Mark Price: {quote.mark_price}")
print(f"Mark IV: {quote.mark_iv}%")
print(f"Delta: {quote.greeks.delta}")
print(f"Gamma: {quote.greeks.gamma}")
print(f"Theta: {quote.greeks.theta}")
print(f"Vega: {quote.greeks.vega}")

# Get orderbook
orderbook = adapter.get_orderbook("BTC-28MAR25-100000-C")
print(f"Best Bid: {orderbook.best_bid_price} ({orderbook.best_bid_amount})")
print(f"Best Ask: {orderbook.best_ask_price} ({orderbook.best_ask_amount})")

# Get DVOL (Deribit Volatility Index)
dvol = adapter.get_dvol("BTC")
print(f"DVOL: {dvol.value}%")
print(f"High 24h: {dvol.high_24h}%")
print(f"Low 24h: {dvol.low_24h}%")
```

### Instrument Naming Convention

Deribit uses a specific format for option instruments:

```
{UNDERLYING}-{DDMMMYY}-{STRIKE}-{TYPE}
```

Examples:

- `BTC-28MAR25-100000-C` = BTC March 28, 2025 $100,000 Call
- `ETH-28MAR25-5000-P` = ETH March 28, 2025 $5,000 Put

**Parsing and Creating Instrument Names**:

```python
from adapters.deribit import (
    parse_deribit_instrument_name,
    create_deribit_instrument_name,
)
from datetime import date
from decimal import Decimal

# Parse instrument name
parsed = parse_deribit_instrument_name("BTC-28MAR25-100000-C")
# Returns:
# {
#     "underlying": "BTC",
#     "expiration": date(2025, 3, 28),
#     "strike": Decimal("100000"),
#     "option_type": "call"
# }

# Create instrument name
name = create_deribit_instrument_name(
    underlying="BTC",
    expiration=date(2025, 3, 28),
    strike=Decimal("100000"),
    option_type="call"
)
# Returns: "BTC-28MAR25-100000-C"
```

---

## Order Execution Adapter

### DeribitOptionsOrderExecutionAdapter

Handles order submission, cancellation, position queries, and order status.

```python
from adapters.deribit import (
    DeribitOptionsOrderExecutionAdapter,
    create_deribit_options_order_execution_adapter,
    DeribitOrder,
    DeribitOrderType,
    DeribitDirection,
    DeribitTimeInForce,
)
from decimal import Decimal

# Create execution adapter (requires API credentials)
adapter = create_deribit_options_order_execution_adapter(
    client_id="your_client_id",
    client_secret="your_client_secret",
    testnet=True,  # Use testnet for testing
)

# Create a limit order
order = DeribitOrder(
    instrument_name="BTC-28MAR25-100000-C",
    direction=DeribitDirection.BUY,
    amount=Decimal("0.1"),  # 0.1 BTC notional
    order_type=DeribitOrderType.LIMIT,
    price=Decimal("0.05"),  # Price in BTC
    time_in_force=DeribitTimeInForce.GTC,
    label="my_order_001",
)

# Submit order
result = adapter.submit_order(order)
if result and result.success:
    print(f"Order ID: {result.order_id}")
    print(f"State: {result.order_state}")
    print(f"Filled: {result.filled_amount}")

# Cancel order
success = adapter.cancel_order(result.order_id)

# Get all positions
positions = adapter.get_positions("BTC")
for pos in positions:
    print(f"{pos.instrument_name}: {pos.size} @ {pos.average_price}")
    print(f"  Delta: {pos.delta}, Gamma: {pos.gamma}")
    print(f"  Unrealized PnL: {pos.floating_profit_loss} BTC")

# Get open orders
orders = adapter.get_open_orders("BTC")
for order in orders:
    print(f"{order.order_id}: {order.direction} {order.amount} @ {order.price}")
```

### Order Types

| Type | Description | Use Case |
|------|-------------|----------|
| `LIMIT` | Limit order at specified price | Normal trading |
| `MARKET` | Market order, immediate execution | Urgent fills |
| `STOP_LIMIT` | Limit order triggered at stop price | Stop-loss |
| `STOP_MARKET` | Market order triggered at stop price | Guaranteed stop |

### Time in Force

| TIF | Description |
|-----|-------------|
| `GTC` | Good-Till-Cancelled |
| `IOC` | Immediate-Or-Cancel |
| `FOK` | Fill-Or-Kill |

---

## Margin Calculation

### DeribitMarginCalculator

Calculates initial/maintenance margin for positions with support for portfolio margining.

```python
from adapters.deribit import (
    DeribitMarginCalculator,
    create_deribit_margin_calculator,
)
from adapters.deribit.margin import PositionForMargin, MarginMode
from decimal import Decimal

# Create margin calculator
calculator = create_deribit_margin_calculator()

# Define positions
positions = [
    PositionForMargin(
        instrument_name="BTC-28MAR25-100000-C",
        size=Decimal("1.0"),  # Long 1 contract
        mark_price=Decimal("0.05"),
        delta=Decimal("0.45"),
        gamma=Decimal("0.00001"),
        underlying_price=Decimal("95000"),
        is_call=True,
        strike=Decimal("100000"),
    ),
    PositionForMargin(
        instrument_name="BTC-28MAR25-90000-P",
        size=Decimal("-1.0"),  # Short 1 contract (hedge)
        mark_price=Decimal("0.03"),
        delta=Decimal("-0.35"),
        gamma=Decimal("0.00001"),
        underlying_price=Decimal("95000"),
        is_call=False,
        strike=Decimal("90000"),
    ),
]

# Calculate portfolio margin
result = calculator.calculate_portfolio_margin(
    positions=positions,
    margin_balance=Decimal("2.0"),  # 2 BTC margin
)

print(f"Initial Margin: {result.initial_margin} BTC")
print(f"Maintenance Margin: {result.maintenance_margin} BTC")
print(f"Available Margin: {result.available_margin} BTC")
print(f"Margin Ratio: {result.margin_ratio}")
print(f"Margin Call Level: {result.margin_call_level.value}")
print(f"Portfolio Delta: {result.delta_total}")
print(f"Portfolio Gamma: {result.gamma_total}")
```

### Margin Levels

| Level | Margin Ratio | Action |
|-------|--------------|--------|
| `HEALTHY` | > 1.5 | Normal trading |
| `WARNING` | 1.2 - 1.5 | Monitor closely |
| `DANGER` | 1.05 - 1.2 | Reduce positions |
| `LIQUIDATION` | ≤ 1.05 | Imminent liquidation |

### Portfolio Margin Benefits

Deribit offers **portfolio margin** which provides margin offsets for hedged positions:

```python
# Delta-neutral position: Long call + Short put at same strike
# Receives significant margin reduction

# Calculate hedge benefit
delta_net = abs(result.delta_total)  # Near zero for delta-neutral
hedge_benefit = (1.0 - delta_net) * 0.5  # Up to 50% reduction
```

### Liquidation Price Estimation

```python
# Estimate liquidation price for a position
liq_price = calculator.estimate_liquidation_price(
    position=positions[0],
    margin_balance=Decimal("0.5"),  # 0.5 BTC margin
)
print(f"Estimated Liquidation Price: ${liq_price}")
```

### Max Position Size Calculation

```python
# Calculate maximum position size given available margin
max_size = calculator.calculate_max_position_size(
    instrument_name="BTC-28MAR25-100000-C",
    mark_price=Decimal("0.05"),
    delta=Decimal("0.45"),
    underlying_price=Decimal("95000"),
    is_call=True,
    strike=Decimal("100000"),
    available_margin=Decimal("1.0"),  # 1 BTC available
    leverage_target=Decimal("5"),  # 5x leverage
)
print(f"Max Position: {max_size} contracts")
```

---

## WebSocket Streaming

### DeribitWebSocketClient

Provides real-time streaming for quotes, orderbook, trades, and user events.

```python
import asyncio
from adapters.deribit import (
    DeribitWebSocketClient,
    DeribitStreamConfig,
    DeribitSubscription,
    create_deribit_websocket_client,
)

# Callback for ticker updates
def on_ticker(data):
    print(f"Ticker: {data['instrument_name']}")
    print(f"  Mark Price: {data.get('mark_price')}")
    print(f"  Mark IV: {data.get('mark_iv')}")

# Callback for orderbook updates
def on_book(data):
    print(f"Book Update: {data['instrument_name']}")
    print(f"  Bids: {len(data.get('bids', []))}")
    print(f"  Asks: {len(data.get('asks', []))}")

# Callback for user order updates (private)
def on_order(data):
    print(f"Order Update: {data['order_id']}")
    print(f"  State: {data['order_state']}")

async def main():
    # Create WebSocket client
    config = DeribitStreamConfig(
        testnet=True,
        client_id="your_client_id",      # For private channels
        client_secret="your_client_secret",
        reconnect_enabled=True,
    )
    client = create_deribit_websocket_client(config)

    # Connect
    connected = await client.connect()
    if not connected:
        print("Failed to connect")
        return

    # Subscribe to public ticker
    await client.subscribe(
        DeribitSubscription.ticker(
            instrument_name="BTC-28MAR25-100000-C",
            interval="100ms",
            callback=on_ticker,
        )
    )

    # Subscribe to orderbook
    await client.subscribe(
        DeribitSubscription.book(
            instrument_name="BTC-28MAR25-100000-C",
            group="none",
            depth="10",
            callback=on_book,
        )
    )

    # Subscribe to user orders (private, requires auth)
    await client.subscribe(
        DeribitSubscription.user_orders(
            currency="BTC",
            callback=on_order,
        )
    )

    # Run event loop
    try:
        await client.run_forever()
    finally:
        await client.disconnect()

# Run
asyncio.run(main())
```

### Subscription Types

| Channel | Type | Description |
|---------|------|-------------|
| `ticker` | Public | Real-time quotes and Greeks |
| `book` | Public | Orderbook updates |
| `trades` | Public | Trade feed |
| `deribit_price_index` | Public | Index price |
| `deribit_price_ranking` | Public | Price ranking |
| `user.orders` | Private | Order status updates |
| `user.trades` | Private | User trade fills |
| `user.portfolio` | Private | Portfolio changes |

### Connection States

| State | Description |
|-------|-------------|
| `DISCONNECTED` | Not connected |
| `CONNECTING` | Connection in progress |
| `CONNECTED` | Connected, not authenticated |
| `AUTHENTICATED` | Connected and authenticated |
| `RECONNECTING` | Auto-reconnecting |

---

## DVOL Index

The **Deribit Volatility Index (DVOL)** is a 30-day constant maturity implied volatility index, similar to VIX for SPX options.

### Methodology

DVOL is calculated using:

1. Weighted average of near-term and next-term option IVs
2. Interpolation to achieve constant 30-day maturity
3. Only OTM options within specific delta range

### Accessing DVOL

```python
from adapters.deribit import DeribitOptionsMarketDataAdapter

adapter = DeribitOptionsMarketDataAdapter(testnet=True)

# Get current DVOL
dvol = adapter.get_dvol("BTC")
print(f"Current DVOL: {dvol.value}%")
print(f"Underlying Price: ${dvol.underlying_price}")
print(f"Timestamp: {dvol.timestamp}")
print(f"24h High: {dvol.high_24h}%")
print(f"24h Low: {dvol.low_24h}%")

# ETH DVOL also available
eth_dvol = adapter.get_dvol("ETH")
```

### Trading DVOL

DVOL futures are available for trading volatility directly:

```python
# DVOL futures naming: DVOL-{DDMMMYY}
# Example: DVOL-28MAR25

# Get DVOL futures quote
# Note: Use futures adapter for DVOL futures trading
```

---

## Configuration

### Environment Variables

```bash
# API Credentials
DERIBIT_CLIENT_ID=your_client_id
DERIBIT_CLIENT_SECRET=your_client_secret

# Environment
DERIBIT_TESTNET=true  # Use testnet (recommended for testing)

# Rate Limiting
DERIBIT_RATE_LIMIT_REQUESTS=20  # Requests per second
DERIBIT_RATE_LIMIT_BURST=50     # Burst capacity
```

### YAML Configuration

```yaml
# configs/deribit.yaml
deribit:
  testnet: true

  api:
    client_id: "${DERIBIT_CLIENT_ID}"
    client_secret: "${DERIBIT_CLIENT_SECRET}"
    timeout: 30.0

  rate_limit:
    requests_per_second: 20
    burst_capacity: 50

  websocket:
    reconnect_enabled: true
    reconnect_delay_initial: 1.0
    reconnect_delay_max: 60.0
    heartbeat_interval: 30.0

  margin:
    mode: portfolio  # or "standard"
    warning_ratio: 1.5
    danger_ratio: 1.2

  execution:
    default_time_in_force: GTC
    max_slippage_bps: 50
```

### Loading Configuration

```python
from adapters.deribit import (
    create_deribit_options_market_data_adapter,
    create_deribit_options_order_execution_adapter,
    create_deribit_margin_calculator,
    create_deribit_websocket_client,
    DeribitStreamConfig,
)
import yaml
import os

# Load config
with open("configs/deribit.yaml") as f:
    config = yaml.safe_load(f)

# Create adapters from config
market_data = create_deribit_options_market_data_adapter(
    testnet=config["deribit"]["testnet"],
)

order_exec = create_deribit_options_order_execution_adapter(
    client_id=os.environ.get("DERIBIT_CLIENT_ID"),
    client_secret=os.environ.get("DERIBIT_CLIENT_SECRET"),
    testnet=config["deribit"]["testnet"],
)

margin_calc = create_deribit_margin_calculator()

ws_config = DeribitStreamConfig(
    testnet=config["deribit"]["testnet"],
    client_id=os.environ.get("DERIBIT_CLIENT_ID"),
    client_secret=os.environ.get("DERIBIT_CLIENT_SECRET"),
    reconnect_enabled=config["deribit"]["websocket"]["reconnect_enabled"],
)
ws_client = create_deribit_websocket_client(ws_config)
```

---

## Usage Examples

### Example 1: Options Pricing Analysis

```python
from adapters.deribit import (
    create_deribit_options_market_data_adapter,
    parse_deribit_instrument_name,
)
from datetime import date, datetime
from decimal import Decimal

adapter = create_deribit_options_market_data_adapter(testnet=True)

# Get full option chain
chain = adapter.get_option_chain("BTC", date(2025, 3, 28))

# Filter ATM options
underlying_price = chain[0].underlying_price if chain else Decimal("95000")
atm_strike = round(float(underlying_price) / 1000) * 1000

atm_options = [c for c in chain if abs(float(c.strike) - atm_strike) <= 5000]

print(f"ATM Options (Strike ≈ ${atm_strike}):")
for contract in atm_options:
    quote = adapter.get_option_quote(contract.instrument_name)
    if quote:
        print(f"  {contract.instrument_name}")
        print(f"    Mark: {quote.mark_price} BTC, IV: {quote.mark_iv}%")
        print(f"    Delta: {quote.greeks.delta}, Theta: {quote.greeks.theta}")
```

### Example 2: Portfolio Risk Analysis

```python
from adapters.deribit import (
    create_deribit_options_order_execution_adapter,
    create_deribit_margin_calculator,
)
from adapters.deribit.margin import PositionForMargin
from decimal import Decimal
import os

# Initialize
exec_adapter = create_deribit_options_order_execution_adapter(
    client_id=os.environ.get("DERIBIT_CLIENT_ID"),
    client_secret=os.environ.get("DERIBIT_CLIENT_SECRET"),
    testnet=True,
)
margin_calc = create_deribit_margin_calculator()

# Get current positions
positions = exec_adapter.get_positions("BTC")

# Convert to margin format
margin_positions = []
for pos in positions:
    margin_positions.append(PositionForMargin(
        instrument_name=pos.instrument_name,
        size=pos.size,
        mark_price=pos.mark_price,
        delta=pos.delta,
        gamma=pos.gamma,
        underlying_price=pos.underlying_price,
        is_call=pos.option_type == "call",
        strike=pos.strike,
    ))

# Calculate portfolio margin
account = exec_adapter.get_account_summary("BTC")
margin_balance = account.get("equity", Decimal("0"))

result = margin_calc.calculate_portfolio_margin(
    positions=margin_positions,
    margin_balance=margin_balance,
)

print("Portfolio Risk Summary:")
print(f"  Total Positions: {len(margin_positions)}")
print(f"  Portfolio Delta: {result.delta_total:.4f}")
print(f"  Portfolio Gamma: {result.gamma_total:.6f}")
print(f"  Initial Margin: {result.initial_margin:.4f} BTC")
print(f"  Maintenance Margin: {result.maintenance_margin:.4f} BTC")
print(f"  Margin Ratio: {result.margin_ratio:.2f}")
print(f"  Status: {result.margin_call_level.value}")
```

### Example 3: Delta-Neutral Strategy

```python
from adapters.deribit import (
    create_deribit_options_market_data_adapter,
    create_deribit_options_order_execution_adapter,
    DeribitOrder,
    DeribitOrderType,
    DeribitDirection,
    DeribitTimeInForce,
)
from decimal import Decimal
import os

# Initialize
md_adapter = create_deribit_options_market_data_adapter(testnet=True)
exec_adapter = create_deribit_options_order_execution_adapter(
    client_id=os.environ.get("DERIBIT_CLIENT_ID"),
    client_secret=os.environ.get("DERIBIT_CLIENT_SECRET"),
    testnet=True,
)

# Find ATM straddle
underlying_price = Decimal("95000")
atm_strike = Decimal(round(float(underlying_price) / 1000) * 1000)

call_name = f"BTC-28MAR25-{atm_strike}-C"
put_name = f"BTC-28MAR25-{atm_strike}-P"

call_quote = md_adapter.get_option_quote(call_name)
put_quote = md_adapter.get_option_quote(put_name)

print(f"ATM Straddle @ ${atm_strike}:")
print(f"  Call: {call_quote.mark_price} BTC (Δ={call_quote.greeks.delta})")
print(f"  Put: {put_quote.mark_price} BTC (Δ={put_quote.greeks.delta})")

# Combined delta (should be near zero)
net_delta = call_quote.greeks.delta + put_quote.greeks.delta
print(f"  Net Delta: {net_delta:.4f}")

# Place straddle order (buy both legs)
size = Decimal("0.1")  # 0.1 BTC per leg

call_order = DeribitOrder(
    instrument_name=call_name,
    direction=DeribitDirection.BUY,
    amount=size,
    order_type=DeribitOrderType.LIMIT,
    price=call_quote.mark_price * Decimal("1.01"),  # 1% premium
    time_in_force=DeribitTimeInForce.GTC,
    label="straddle_call",
)

put_order = DeribitOrder(
    instrument_name=put_name,
    direction=DeribitDirection.BUY,
    amount=size,
    order_type=DeribitOrderType.LIMIT,
    price=put_quote.mark_price * Decimal("1.01"),
    time_in_force=DeribitTimeInForce.GTC,
    label="straddle_put",
)

# Submit orders
call_result = exec_adapter.submit_order(call_order)
put_result = exec_adapter.submit_order(put_order)

print(f"Orders submitted:")
print(f"  Call: {call_result.order_id if call_result else 'Failed'}")
print(f"  Put: {put_result.order_id if put_result else 'Failed'}")
```

### Example 4: Real-Time Greeks Streaming

```python
import asyncio
from adapters.deribit import (
    create_deribit_websocket_client,
    DeribitStreamConfig,
    DeribitSubscription,
)

class GreeksMonitor:
    def __init__(self):
        self.positions = {}

    def on_ticker(self, data):
        instrument = data["instrument_name"]
        self.positions[instrument] = {
            "mark_price": data.get("mark_price"),
            "mark_iv": data.get("mark_iv"),
            "delta": data.get("greeks", {}).get("delta"),
            "gamma": data.get("greeks", {}).get("gamma"),
            "theta": data.get("greeks", {}).get("theta"),
            "vega": data.get("greeks", {}).get("vega"),
        }

        # Print portfolio Greeks
        total_delta = sum(p.get("delta", 0) or 0 for p in self.positions.values())
        total_gamma = sum(p.get("gamma", 0) or 0 for p in self.positions.values())
        print(f"Portfolio Delta: {total_delta:.4f}, Gamma: {total_gamma:.6f}")

async def main():
    monitor = GreeksMonitor()

    config = DeribitStreamConfig(testnet=True)
    client = create_deribit_websocket_client(config)

    await client.connect()

    # Subscribe to multiple instruments
    instruments = [
        "BTC-28MAR25-100000-C",
        "BTC-28MAR25-100000-P",
        "BTC-28MAR25-95000-C",
        "BTC-28MAR25-95000-P",
    ]

    for inst in instruments:
        await client.subscribe(
            DeribitSubscription.ticker(inst, "100ms", monitor.on_ticker)
        )

    await client.run_forever()

asyncio.run(main())
```

---

## Testing

### Running Tests

```bash
# All Deribit tests (120 tests)
pytest tests/test_deribit_options.py -v

# By category
pytest tests/test_deribit_options.py::TestInstrumentNaming -v
pytest tests/test_deribit_options.py::TestInversePayoffs -v
pytest tests/test_deribit_options.py::TestGreeksValidation -v
pytest tests/test_deribit_options.py::TestQuoteMarketData -v
pytest tests/test_deribit_options.py::TestMarginCalculations -v
pytest tests/test_deribit_options.py::TestOrderExecution -v
pytest tests/test_deribit_options.py::TestWebSocketStreaming -v
pytest tests/test_deribit_options.py::TestIntegration -v
```

### Test Categories

| Category | Tests | Coverage |
|----------|-------|----------|
| Instrument Naming | 12 | Parsing, creation, roundtrip |
| Inverse Payoffs | 20 | Call/put, ITM/OTM/ATM, edge cases |
| Greeks Validation | 15 | Delta, gamma, theta, vega bounds |
| Quote/Market Data | 18 | Quotes, orderbook, DVOL |
| Margin Calculations | 25 | Single/portfolio, liquidation |
| Order Execution | 15 | Orders, states, API params |
| WebSocket | 15 | Subscriptions, reconnection |
| Integration | Variable | End-to-end workflows |

---

## References

### Deribit Documentation

- **API Reference**: https://docs.deribit.com/
- **DVOL Methodology**: https://www.deribit.com/pages/docs/volatility-index
- **Margin Specification**: https://www.deribit.com/pages/docs/options-margin
- **Settlement Procedures**: https://www.deribit.com/pages/docs/settlement

### Academic References

- Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"
- Carr, P., & Wu, L. (2009). "Variance Risk Premiums"
- CBOE (2019). "VIX White Paper" (methodology reference for DVOL)

### Related Documentation

- [Options Core Models](core_models.md) - Options pricing and Greeks
- [Exchange Adapters](exchange_adapters.md) - IB, Polygon adapters
- [Memory Architecture](memory_architecture.md) - Caching strategies

---

**Last Updated**: 2025-12-03
**Version**: 1.0.0
**Status**: ✅ Tested and operational
