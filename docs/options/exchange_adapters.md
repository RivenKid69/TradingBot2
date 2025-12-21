# Options Exchange Adapters (Phase 2)

## Overview

This document describes the exchange adapters for options trading, providing unified access to multiple data sources and execution venues. The architecture enables seamless integration with Interactive Brokers (IB), Theta Data, and Polygon.io for comprehensive options market data and execution.

**Status**: ✅ Internal testing complete (not independently audited)
**Implementation Date**: 2025-12-03
**Reference**: [OPTIONS_INTEGRATION_PLAN.md](../OPTIONS_INTEGRATION_PLAN.md) Phase 2
**Tests**: 165 internal tests (see CI for current status)

## Supported Exchanges

| Exchange | Asset Class | Data | Execution | Protocol | Cost |
|----------|-------------|------|-----------|----------|------|
| **IB TWS** | US Equity Options | ✅ | ✅ | TWS API | Subscription |
| **Theta Data** | US Options (historical+RT) | ✅ | ❌ | REST/WS | $100/mo |
| **Polygon.io** | US Options (historical) | ✅ | ❌ | REST | $200/mo |
| **Alpaca** | US Equity Options | ✅ | ✅ | REST/WS | Free |

### Recommended Data Stack

| Data Need | Primary Source | Backup |
|-----------|---------------|--------|
| Real-time US options | IB TWS API | Theta Data |
| Historical US options | Theta Data | Polygon.io |
| Index options (SPX, VIX) | IB TWS API | -- |
| Commission-free execution | Alpaca | IB TWS |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Options Exchange Adapters                         │
├─────────────────────────────────────────────────────────────────────┤
│  Real-Time Data & Execution    │  Historical Data                   │
│  ├─ adapters/ib/options.py     │  ├─ adapters/theta_data/options.py │
│  ├─ adapters/ib/options_combo.py│  └─ adapters/polygon/options.py   │
│  └─ adapters/alpaca/options_*   │                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Rate Limiting                  │  Registry                          │
│  └─ adapters/ib/options_rate_   │  └─ adapters/registry.py           │
│     limiter.py                  │     - OPTIONS_MARKET_DATA          │
│                                 │     - OPTIONS_ORDER_EXECUTION      │
│                                 │     - OPTIONS_COMBO                │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. IB Options Adapter (`adapters/ib/options.py`)

**File**: [adapters/ib/options.py](../../adapters/ib/options.py)

Full-featured options market data and order execution via Interactive Brokers TWS API.

#### Key Classes

| Class | Description |
|-------|-------------|
| `IBOptionsContract` | Options contract specification with OCC symbology |
| `IBOptionsQuote` | Real-time quote with Greeks from IB |
| `IBOptionsGreeks` | Complete Greeks (Delta, Gamma, Theta, Vega, Rho, IV) |
| `IBOptionsMarketDataAdapter` | Market data: chains, quotes, streaming |
| `IBOptionsOrderExecutionAdapter` | Order execution: market, limit, combos |

#### OCC Symbology Support

```python
from adapters.ib.options import (
    IBOptionsContract,
    occ_to_ib_contract,
    ib_to_occ_symbol,
)

# Parse OCC symbol
contract = IBOptionsContract.from_occ_symbol("AAPL  241220C00200000")
# Result: AAPL Dec 2024 $200 Call

# Create IB Contract object
ib_contract = occ_to_ib_contract("AAPL  241220C00200000")

# Generate OCC symbol from components
occ = IBOptionsContract(
    symbol="AAPL",
    expiry="2024-12-20",
    strike=200.0,
    right="C",
).to_occ_symbol()
# Result: "AAPL  241220C00200000"
```

#### Market Data Usage

```python
from adapters.ib.options import (
    IBOptionsMarketDataAdapter,
    create_ib_options_market_data_adapter,
)

# Create adapter
adapter = create_ib_options_market_data_adapter(
    host="127.0.0.1",
    port=7497,  # TWS paper: 7497, live: 7496
    client_id=1,
)

# Get option chain
chain = adapter.get_option_chain(
    underlying="AAPL",
    expiration=date(2024, 12, 20),
)

# Get real-time quote with Greeks
quote = adapter.get_option_quote(chain[0])
print(f"Bid: {quote.bid}, Ask: {quote.ask}")
print(f"IV: {quote.greeks.implied_volatility:.2%}")
print(f"Delta: {quote.greeks.delta:.4f}")

# Stream quotes (async)
async for quote in adapter.stream_option_quotes_async(chain[:10]):
    print(f"{quote.contract.symbol}: {quote.bid}x{quote.ask}")
```

#### Order Execution Usage

```python
from adapters.ib.options import (
    IBOptionsOrderExecutionAdapter,
    IBOptionsOrder,
    IBOptionsOrderType,
    IBOptionsSide,
    create_ib_options_execution_adapter,
)

# Create adapter
adapter = create_ib_options_execution_adapter(
    host="127.0.0.1",
    port=7497,
    client_id=2,
)

# Submit single-leg order
order = IBOptionsOrder(
    contract=contract,
    side=IBOptionsSide.BUY,
    quantity=10,
    order_type=IBOptionsOrderType.LIMIT,
    limit_price=5.50,
)
result = adapter.submit_option_order(order)
print(f"Order ID: {result.order_id}, Status: {result.status}")

# What-If margin query
margin = adapter.get_what_if_margin(order)
print(f"Initial margin: ${margin.initial_margin:,.2f}")
print(f"Maintenance margin: ${margin.maintenance_margin:,.2f}")
```

### 2. IB Options Rate Limiter (`adapters/ib/options_rate_limiter.py`)

**File**: [adapters/ib/options_rate_limiter.py](../../adapters/ib/options_rate_limiter.py)

Intelligent rate limiting for IB API requests with caching and priority queue.

#### IB Rate Limits

| Limit Type | IB Limit | Safety Margin | Implemented |
|------------|----------|---------------|-------------|
| Option chains | 10/min | 8/min | ✅ |
| Quote requests | 100/sec | 80/sec | ✅ |
| Order submissions | 50/sec | 40/sec | ✅ |
| Concurrent market data | 100 lines | 100 lines | ✅ |

#### Chain Caching

```python
from adapters.ib.options_rate_limiter import (
    OptionsChainCache,
    CachedChain,
)

cache = OptionsChainCache(
    max_chains=100,
    default_ttl_sec=300.0,      # 5-minute TTL for back-months
    front_month_ttl_sec=60.0,   # 1-minute TTL for front-month
)

# Check cache
cached = cache.get("AAPL", date(2024, 12, 20))
if cached is None:
    # Fetch from IB
    chain = ib_adapter.get_option_chain("AAPL", date(2024, 12, 20))
    cache.put("AAPL", date(2024, 12, 20), chain)
```

#### Priority Queue

```python
from adapters.ib.options_rate_limiter import (
    IBOptionsRateLimitManager,
    RequestPriority,
)

manager = IBOptionsRateLimitManager(
    chain_limit_per_min=8,
    quote_limit_per_sec=80,
    order_limit_per_sec=40,
)

# Priority levels (lower = higher priority)
# 0: Order execution (highest)
# 1: Position risk updates
# 2: Front-month chain refresh
# 3: Active underlyings
# 4: Background chain refresh
# 9: Backfill requests (lowest)

# Request with priority
manager.request_chain(
    underlying="SPY",
    expiration=date(2024, 12, 20),
    callback=handle_chain,
    priority=RequestPriority.FRONT_MONTH,
)

# Process queue within rate limits
processed = manager.process_queue()
```

### 3. IB Combo Orders (`adapters/ib/options_combo.py`)

**File**: [adapters/ib/options_combo.py](../../adapters/ib/options_combo.py)

Multi-leg spread and combo order support via IB.

#### Supported Strategies

| Strategy | Legs | Description |
|----------|------|-------------|
| Vertical Spread | 2 | Same expiry, different strikes |
| Calendar Spread | 2 | Same strike, different expiries |
| Diagonal Spread | 2 | Different strike and expiry |
| Straddle | 2 | ATM call + put |
| Strangle | 2 | OTM call + put |
| Iron Condor | 4 | Bull put + bear call spreads |
| Butterfly | 3-4 | Middle strike vs wings |
| Ratio Spread | 2+ | Unequal quantities |

#### Usage

```python
from adapters.ib.options_combo import (
    IBComboOrderBuilder,
    IBComboLeg,
    IBComboStrategy,
    build_vertical_spread,
    build_iron_condor,
)

# Build vertical spread
spread = build_vertical_spread(
    underlying="SPY",
    expiry=date(2024, 12, 20),
    long_strike=500.0,
    short_strike=505.0,
    option_type="C",
    quantity=10,
)

# Build iron condor
condor = build_iron_condor(
    underlying="SPY",
    expiry=date(2024, 12, 20),
    put_long_strike=490.0,
    put_short_strike=495.0,
    call_short_strike=505.0,
    call_long_strike=510.0,
    quantity=5,
)

# Submit combo order
from adapters.ib.options import IBOptionsOrderExecutionAdapter

adapter = create_ib_options_execution_adapter(...)
result = adapter.submit_combo_order(
    legs=spread.legs,
    order_type="LMT",
    limit_price=2.50,  # Net debit/credit
)
```

#### Combo Leg Structure

```python
@dataclass
class IBComboLeg:
    contract: IBOptionsContract
    action: str  # "BUY" or "SELL"
    ratio: int   # Quantity ratio

# Example: Bull Call Spread
legs = [
    IBComboLeg(contract=long_call, action="BUY", ratio=1),
    IBComboLeg(contract=short_call, action="SELL", ratio=1),
]
```

### 4. Theta Data Adapter (`adapters/theta_data/options.py`)

**File**: [adapters/theta_data/options.py](../../adapters/theta_data/options.py)

Cost-effective options data via Theta Data ($100/mo vs OPRA $2,500/mo).

#### Key Features

| Feature | Description |
|---------|-------------|
| Full US options universe | All listed US equity options |
| Historical data | Back to 2013 |
| Real-time quotes | 15-min delay (free) or real-time ($) |
| Greeks included | Delta, Gamma, Theta, Vega, IV |
| EOD snapshots | Full chain snapshots |

#### Data Classes

```python
from adapters.theta_data.options import (
    ThetaDataOptionsAdapter,
    ThetaDataConfig,
    ThetaDataQuote,
    ThetaDataTrade,
    create_theta_data_adapter,
)

@dataclass
class ThetaDataQuote:
    timestamp: datetime
    bid: Decimal
    ask: Decimal
    bid_size: int
    ask_size: int
    underlying_price: Decimal
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    open_interest: int
```

#### Usage

```python
# Create adapter
adapter = create_theta_data_adapter(
    api_key="your_theta_data_key",
    use_delayed=True,  # Free 15-min delayed
)

# Get current option chain
chain = adapter.get_option_chain(
    underlying="AAPL",
    expiration=date(2024, 12, 20),
)

# Get historical quotes
quotes_df = adapter.get_historical_quotes(
    symbol="AAPL",
    expiration=date(2024, 12, 20),
    strike=200.0,
    option_type="C",
    start_date=date(2024, 1, 1),
    end_date=date(2024, 6, 30),
    interval="1min",
)

# Get historical trades
trades_df = adapter.get_historical_trades(
    symbol="AAPL",
    expiration=date(2024, 12, 20),
    strike=200.0,
    option_type="C",
    trade_date=date(2024, 6, 15),
)

# Get EOD chain snapshot
eod_df = adapter.get_eod_chain(
    underlying="SPY",
    date=date(2024, 6, 14),
)
```

### 5. Polygon Options Adapter (`adapters/polygon/options.py`)

**File**: [adapters/polygon/options.py](../../adapters/polygon/options.py)

Historical options data via Polygon.io (2018+).

#### Polygon Ticker Format

```
O:{SYMBOL}{YYMMDD}{C/P}{STRIKE×1000}

Examples:
- O:AAPL241220C00200000  →  AAPL Dec 2024 $200 Call
- O:SPY241115P00450000   →  SPY Nov 2024 $450 Put
```

#### Symbol Conversion

```python
from adapters.polygon.options import (
    polygon_ticker_to_occ,
    occ_to_polygon_ticker,
    parse_polygon_ticker,
)

# Parse Polygon ticker
parsed = parse_polygon_ticker("O:AAPL241220C00200000")
# parsed = PolygonOptionsParsed(
#     underlying="AAPL",
#     expiration=date(2024, 12, 20),
#     option_type=OptionType.CALL,
#     strike=Decimal("200.00"),
# )

# Convert to OCC
occ = polygon_ticker_to_occ("O:AAPL241220C00200000")
# occ = "AAPL  241220C00200000"

# Convert from OCC
polygon = occ_to_polygon_ticker("AAPL  241220C00200000")
# polygon = "O:AAPL241220C00200000"
```

#### Data Classes

```python
from adapters.polygon.options import (
    PolygonOptionsAdapter,
    PolygonOptionsContract,
    PolygonOptionsQuote,
    PolygonOptionsSnapshot,
    create_polygon_options_adapter,
)

@dataclass
class PolygonOptionsSnapshot:
    """Full chain snapshot for a single date."""
    underlying: str
    snapshot_date: date
    contracts: List[PolygonOptionsContract]
    underlying_price: Decimal
    timestamp: datetime
```

#### Usage

```python
# Create adapter
adapter = create_polygon_options_adapter(
    api_key="your_polygon_key",
)

# Get historical chain snapshot
chain = adapter.get_historical_chain(
    underlying="AAPL",
    date=date(2024, 6, 14),
)

# Get historical NBBO quotes
quotes_df = adapter.get_historical_quotes(
    contract_symbol="O:AAPL241220C00200000",
    start_date=date(2024, 1, 1),
    end_date=date(2024, 6, 30),
)

# Get historical trades
trades_df = adapter.get_historical_trades(
    contract_symbol="O:AAPL241220C00200000",
    trade_date=date(2024, 6, 15),
)

# Get options snapshot (current)
snapshot = adapter.get_options_snapshot("AAPL")
```

## Registry Integration

### Adapter Types

Phase 2 adds three new adapter types to the registry:

```python
from adapters.registry import AdapterType

AdapterType.OPTIONS_MARKET_DATA      # Historical chains, quotes, Greeks
AdapterType.OPTIONS_ORDER_EXECUTION  # Single-leg options orders
AdapterType.OPTIONS_COMBO            # Multi-leg spreads, combos (IB)
```

### Factory Functions

```python
from adapters.registry import (
    create_options_market_data_adapter,
    create_options_order_execution_adapter,
    create_options_combo_adapter,
)
from adapters.models import ExchangeVendor

# Create IB options market data adapter
ib_md = create_options_market_data_adapter(
    vendor=ExchangeVendor.IB,
    config={"host": "127.0.0.1", "port": 7497},
)

# Create Theta Data adapter
theta = create_options_market_data_adapter(
    vendor=ExchangeVendor.THETA_DATA,
    config={"api_key": "..."},
)

# Create Polygon adapter
polygon = create_options_market_data_adapter(
    vendor=ExchangeVendor.POLYGON,
    config={"api_key": "..."},
)

# Create IB options order execution adapter
ib_exec = create_options_order_execution_adapter(
    vendor=ExchangeVendor.IB,
    config={"host": "127.0.0.1", "port": 7497},
)

# Create IB combo order adapter
ib_combo = create_options_combo_adapter(
    vendor=ExchangeVendor.IB,
    config={"host": "127.0.0.1", "port": 7497},
)
```

### Automatic Registration

Adapters are automatically registered when their modules are imported:

```python
# adapters/ib/__init__.py registers IB options adapters
# adapters/theta_data/__init__.py registers Theta Data adapter
# adapters/polygon/__init__.py registers Polygon adapter
```

## Configuration

### Environment Variables

```bash
# Interactive Brokers (local TWS/Gateway connection)
# No API key needed - uses local TWS connection

# Theta Data
THETA_DATA_API_KEY=your_theta_data_key

# Polygon.io
POLYGON_API_KEY=your_polygon_key
```

### YAML Configuration

```yaml
# configs/options_adapters.yaml
ib:
  host: "127.0.0.1"
  port: 7497  # Paper: 7497, Live: 7496
  client_id: 1
  timeout: 30.0
  rate_limits:
    chain_per_min: 8
    quote_per_sec: 80
    order_per_sec: 40

theta_data:
  api_key: "${THETA_DATA_API_KEY}"
  use_delayed: true  # Free 15-min delayed
  cache_ttl_sec: 300

polygon:
  api_key: "${POLYGON_API_KEY}"
  timeout: 30.0
```

## Test Coverage

### Test Matrix (165 tests)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| IB Options Rate Limiter | 10 | Throttling, backoff, cache |
| IB Options Adapter | 50 | Chains, quotes, orders, Greeks |
| IB What-If Margin | 15 | Pre-trade margin calculation |
| IB Combo Orders | 20 | Multi-leg spreads, execution |
| Theta Data Adapter | 25 | Chains, historical, EOD |
| Polygon Options | 15 | Historical chains, quotes |
| Registry Integration | 5 | Factory functions, types |
| **Total** | **165** | **100%** |

### Running Tests

```bash
# All Phase 2 tests
pytest tests/test_options_adapters.py -v

# By category
pytest tests/test_options_adapters.py::TestIBOptionsRateLimiter -v
pytest tests/test_options_adapters.py::TestIBOptionsAdapter -v
pytest tests/test_options_adapters.py::TestIBWhatIfMargin -v
pytest tests/test_options_adapters.py::TestIBComboOrders -v
pytest tests/test_options_adapters.py::TestThetaDataAdapter -v
pytest tests/test_options_adapters.py::TestPolygonOptionsAdapter -v
pytest tests/test_options_adapters.py::TestRegistryIntegration -v
```

## Best Practices

### 1. Use Caching for Chain Requests

```python
# IB has strict rate limits (10 chains/min)
# Always use the rate limiter with caching
manager = IBOptionsRateLimitManager()

# Front-month chains refresh every 1 minute
# Back-month chains refresh every 5 minutes
```

### 2. Choose Appropriate Data Source

```python
# Real-time execution: Use IB
ib_adapter = create_options_market_data_adapter(ExchangeVendor.IB)

# Historical backtesting: Use Theta Data (cost-effective)
theta_adapter = create_options_market_data_adapter(ExchangeVendor.THETA_DATA)

# Long-term historical analysis: Use Polygon (2018+)
polygon_adapter = create_options_market_data_adapter(ExchangeVendor.POLYGON)
```

### 3. Handle Rate Limit Errors

```python
from adapters.ib.options_rate_limiter import RateLimitExceeded

try:
    chain = adapter.get_option_chain("SPY", date(2024, 12, 20))
except RateLimitExceeded as e:
    # Wait and retry
    await asyncio.sleep(e.retry_after_seconds)
    chain = adapter.get_option_chain("SPY", date(2024, 12, 20))
```

### 4. Use Priority Queue for Critical Requests

```python
# Order execution always gets highest priority
manager.request_chain(
    underlying="SPY",
    expiration=date(2024, 12, 20),
    callback=execute_order,
    priority=RequestPriority.ORDER,  # Priority 0
)

# Background refresh gets lowest priority
manager.request_chain(
    underlying="SPY",
    expiration=date(2025, 6, 20),
    callback=update_cache,
    priority=RequestPriority.BACKFILL,  # Priority 9
)
```

## Integration with Existing Adapters

### Alpaca Options Integration

The existing `adapters/alpaca/options_execution.py` is already integrated:

```python
from adapters.alpaca.options_execution import (
    AlpacaOptionsExecutionAdapter,
    OptionType,        # Shared enum
    OptionStrategy,    # 11 strategies
    OptionContract,    # OCC symbology
)

# Use existing Alpaca adapter
alpaca = AlpacaOptionsExecutionAdapter(
    api_key="...",
    api_secret="...",
    paper=True,
)

# Submit option order
result = alpaca.submit_option_order(
    contract=contract,
    order_type=OptionOrderType.LIMIT,
    qty=10,
    side="buy",
    limit_price=5.50,
)
```

### LOB Integration

Phase 2 adapters integrate with Phase 0.5 LOB architecture:

```python
from lob.lazy_multi_series import create_lazy_lob_manager

# Feed IB quotes into LOB
manager = create_lazy_lob_manager(max_active_lobs=50)

async for quote in ib_adapter.stream_option_quotes_async(chain):
    series_key = f"{quote.contract.symbol}_{quote.contract.expiry}_{quote.contract.right}_{quote.contract.strike}"
    lob = manager.get_or_create(series_key)
    lob.update_quote(quote.bid, quote.ask, quote.bid_size, quote.ask_size)
```

## References

- OPTIONS_INTEGRATION_PLAN.md Phase 2
- IB TWS API Documentation: https://interactivebrokers.github.io/tws-api/
- Theta Data API: https://www.thetadata.io/
- Polygon.io Options API: https://polygon.io/docs/options/
- OCC Symbology: https://www.theocc.com/

## Changelog

| Date | Change |
|------|--------|
| 2025-12-03 | Initial implementation (Phase 2) |

---

**Last Updated**: 2025-12-03
**Version**: 1.0.0
