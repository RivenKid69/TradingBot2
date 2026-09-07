# Interactive Brokers TWS API Documentation Summary

## Phase 0 Deliverable: API Analysis (CME Track)

**Version**: 1.0
**Date**: 2025-11-30
**Status**: RESEARCH COMPLETE

---

## 1. API Overview

Interactive Brokers (IB) provides access to CME, COMEX, NYMEX, CBOT futures via TWS API.

### Connection Options

| Method | Description | Library |
|--------|-------------|---------|
| TWS API | Native IB API | `ibapi` (official) |
| ib_insync | Async wrapper | `ib_insync` (recommended) |
| IB Gateway | Headless connection | Same API |

### Ports

| Mode | Port | Description |
|------|------|-------------|
| TWS Live | 7496 | Live trading |
| TWS Paper | 7497 | Paper trading |
| Gateway Live | 4001 | Live (headless) |
| Gateway Paper | 4002 | Paper (headless) |

---

## 2. Contract Definitions

### 2.1 Futures Contract Structure

```python
from ibapi.contract import Contract

# E-mini S&P 500
contract = Contract()
contract.symbol = "ES"
contract.secType = "FUT"
contract.exchange = "CME"
contract.currency = "USD"
contract.lastTradeDateOrContractMonth = "202403"  # March 2024

# Gold Futures
contract = Contract()
contract.symbol = "GC"
contract.secType = "FUT"
contract.exchange = "COMEX"
contract.currency = "USD"
contract.lastTradeDateOrContractMonth = "202402"

# Continuous Contract (auto-roll)
contract = Contract()
contract.symbol = "ES"
contract.secType = "CONTFUT"  # Continuous futures
contract.exchange = "CME"
contract.currency = "USD"
```

### 2.2 Contract Details Request

```python
def reqContractDetails(reqId: int, contract: Contract)
```

**Response Fields:**

```python
class ContractDetails:
    contract: Contract
    marketName: str           # "ES" for E-mini S&P
    minTick: float           # 0.25 for ES
    priceMagnifier: int      # 1
    orderTypes: str          # "ACTIVETIM,..."
    validExchanges: str      # "CME,SMART"
    underConId: int          # Underlying contract ID
    longName: str            # "E-mini S&P 500"
    contractMonth: str       # "202403"
    industry: str
    category: str
    subcategory: str
    timeZoneId: str          # "US/Central"
    tradingHours: str        # "20231218:1700-..."
    liquidHours: str         # Regular trading hours
    evRule: str              # Price magnifier rule
    evMultiplier: float      # Contract multiplier (50 for ES)
    mdSizeMultiplier: int
    aggGroup: int
    underSymbol: str         # Underlying symbol
    underSecType: str
    marketRuleIds: str
    realExpirationDate: str  # "20240315"
```

---

## 3. Market Data

### 3.1 Real-time Quotes

```python
def reqMktData(
    reqId: int,
    contract: Contract,
    genericTickList: str,  # "100,101,104,..."
    snapshot: bool,
    regulatorySnapshot: bool,
    mktDataOptions: list
)
```

**Generic Tick Types:**

| Code | Description |
|------|-------------|
| 100 | Option volume |
| 101 | Option open interest |
| 104 | Historical volatility |
| 105 | Average option volume |
| 106 | Implied volatility |
| 162 | Index future premium |
| 165 | Misc stats |
| 221 | Trade count |
| 225 | Volatility |
| 233 | RT volume |
| 293 | Trade rate |
| 294 | Volume rate |
| 411 | RT historical vol |

**Callback:**

```python
def tickPrice(reqId, tickType, price, attrib):
    # tickType: 1=Bid, 2=Ask, 4=Last, 6=High, 7=Low, 9=Close
    pass

def tickSize(reqId, tickType, size):
    # tickType: 0=BidSize, 3=AskSize, 5=LastSize, 8=Volume
    pass
```

### 3.2 Level 2 Market Depth

```python
def reqMktDepth(
    reqId: int,
    contract: Contract,
    numRows: int,           # Number of rows (5-50)
    isSmartDepth: bool,
    mktDepthOptions: list
)
```

**Callback:**

```python
def updateMktDepth(reqId, position, operation, side, price, size):
    # operation: 0=Insert, 1=Update, 2=Delete
    # side: 0=Ask, 1=Bid
    pass
```

### 3.3 Historical Data

```python
def reqHistoricalData(
    reqId: int,
    contract: Contract,
    endDateTime: str,      # "20231218 16:00:00 US/Eastern"
    durationStr: str,      # "1 D", "1 W", "1 M", "1 Y"
    barSizeSetting: str,   # "1 min", "5 mins", "1 hour", "1 day"
    whatToShow: str,       # "TRADES", "MIDPOINT", "BID", "ASK"
    useRTH: int,           # 0=All hours, 1=RTH only
    formatDate: int,       # 1=yyyyMMdd, 2=unix
    keepUpToDate: bool,
    chartOptions: list
)
```

**Bar Sizes:**

- `1 secs`, `5 secs`, `10 secs`, `15 secs`, `30 secs`
- `1 min`, `2 mins`, `3 mins`, `5 mins`, `10 mins`, `15 mins`, `20 mins`, `30 mins`
- `1 hour`, `2 hours`, `3 hours`, `4 hours`, `8 hours`
- `1 day`, `1 week`, `1 month`

### 3.4 Real-Time Bars (5-second)

```python
def reqRealTimeBars(
    reqId: int,
    contract: Contract,
    barSize: int,          # Currently only 5 seconds
    whatToShow: str,       # "TRADES", "MIDPOINT", "BID", "ASK"
    useRTH: bool,
    realTimeBarsOptions: list
)
```

---

## 4. Order Management

### 4.1 Order Submission

```python
from ibapi.order import Order

order = Order()
order.action = "BUY"           # "BUY" or "SELL"
order.orderType = "LMT"        # Order type
order.totalQuantity = 1        # Number of contracts
order.lmtPrice = 4500.00       # Limit price
order.tif = "DAY"              # Time in force

client.placeOrder(orderId, contract, order)
```

### 4.2 Order Types

| Type | Code | Description |
|------|------|-------------|
| Market | MKT | Immediate execution |
| Limit | LMT | Price specified |
| Stop | STP | Trigger at stop price |
| Stop Limit | STP LMT | Stop triggers limit |
| Trailing Stop | TRAIL | Dynamic stop |
| Market on Close | MOC | At market close |
| Limit on Close | LOC | Limit at close |
| Market on Open | MKT ON OPEN | At market open |
| Limit on Open | LMT ON OPEN | Limit at open |

### 4.3 Time in Force

| TIF | Description |
|-----|-------------|
| DAY | Good for day only |
| GTC | Good til cancelled |
| IOC | Immediate or cancel |
| FOK | Fill or kill |
| GTD | Good til date |
| OPG | At the opening |

### 4.4 Bracket Orders

```python
# Parent order
parent = Order()
parent.orderId = parentId
parent.action = "BUY"
parent.orderType = "LMT"
parent.totalQuantity = 1
parent.lmtPrice = 4500.00
parent.transmit = False  # Don't submit yet

# Take profit
takeProfit = Order()
takeProfit.orderId = parentId + 1
takeProfit.action = "SELL"
takeProfit.orderType = "LMT"
takeProfit.totalQuantity = 1
takeProfit.lmtPrice = 4600.00
takeProfit.parentId = parentId
takeProfit.transmit = False

# Stop loss
stopLoss = Order()
stopLoss.orderId = parentId + 2
stopLoss.action = "SELL"
stopLoss.orderType = "STP"
stopLoss.totalQuantity = 1
stopLoss.auxPrice = 4400.00  # Stop price
stopLoss.parentId = parentId
stopLoss.transmit = True  # Submit all
```

---

## 5. Account & Position Data

### 5.1 Account Summary

```python
def reqAccountSummary(
    reqId: int,
    groupName: str,        # "All" for all accounts
    tags: str              # Comma-separated tags
)
```

**Tags:**

- `AccountType` - Individual, Joint, etc.
- `NetLiquidation` - Net liquidation value
- `TotalCashValue` - Cash balance
- `SettledCash` - Settled cash
- `AccruedCash` - Interest accrued
- `BuyingPower` - Margin buying power
- `EquityWithLoanValue` - Equity with loan
- `PreviousDayEquityWithLoanValue`
- `GrossPositionValue` - Gross position value
- `RegTEquity` - Reg T equity
- `RegTMargin` - Reg T margin
- `SMA` - Special memorandum account
- `InitMarginReq` - Initial margin required
- `MaintMarginReq` - Maintenance margin required
- `AvailableFunds` - Available funds
- `ExcessLiquidity` - Excess liquidity
- `Cushion` - Margin cushion %
- `FullInitMarginReq` - Full initial margin
- `FullMaintMarginReq` - Full maintenance margin
- `FullAvailableFunds` - Full available funds
- `FullExcessLiquidity` - Full excess liquidity
- `LookAheadNextChange` - Next margin change time
- `LookAheadInitMarginReq` - Look-ahead initial margin
- `LookAheadMaintMarginReq` - Look-ahead maintenance margin
- `LookAheadAvailableFunds` - Look-ahead available funds
- `LookAheadExcessLiquidity` - Look-ahead excess liquidity
- `HighestSeverity` - Highest severity level
- `DayTradesRemaining` - Day trades left
- `Leverage` - Account leverage

### 5.2 Position Updates

```python
def reqPositions()
```

**Callback:**

```python
def position(account, contract, position, avgCost):
    # position: Signed (positive=long, negative=short)
    # avgCost: Average cost per unit
    pass
```

### 5.3 Account Updates

```python
def reqAccountUpdates(subscribe: bool, accountCode: str)
```

**Callback:**

```python
def updateAccountValue(key, val, currency, accountName):
    pass

def updatePortfolio(contract, position, marketPrice, marketValue,
                    avgCost, unrealizedPNL, realizedPNL, accountName):
    pass
```

---

## 6. Margin Calculation (SPAN)

### 6.1 SPAN Margin Overview

CME uses SPAN (Standard Portfolio ANalysis of Risk) for margin:

- Portfolio-based margin calculation
- Offsets for correlated products
- Scenario analysis for price moves

### 6.2 Margin Requirements

| Contract | Initial Margin* | Maintenance Margin* |
|----------|----------------|---------------------|
| ES (E-mini S&P) | ~$12,000 | ~$10,800 |
| NQ (E-mini Nasdaq) | ~$16,000 | ~$14,400 |
| GC (Gold) | ~$8,000 | ~$7,200 |
| CL (Crude Oil) | ~$8,500 | ~$7,650 |
| 6E (Euro FX) | ~$2,500 | ~$2,250 |

*Margins change frequently based on volatility

### 6.3 Requesting Margin

```python
# Option 1: From account summary
def reqAccountSummary(reqId, "All", "InitMarginReq,MaintMarginReq")

# Option 2: What-if order (preview margin impact)
# Use order.whatIf = True before placing
```

---

## 7. Trading Hours

### 7.1 CME Globex Hours

| Product | Trading Hours (CT) | Maintenance |
|---------|-------------------|-------------|
| ES, NQ, YM, RTY | Sun 5pm - Fri 4pm | 3:15-3:30pm daily |
| GC, SI | Sun 5pm - Fri 4pm | 4:15-4:30pm daily |
| CL, NG | Sun 5pm - Fri 4pm | 4:15-4:30pm daily |
| 6E, 6J, 6B | Sun 5pm - Fri 4pm | 4:15-4:30pm daily |

### 7.2 Settlement Times

| Product | Settlement Time (CT) |
|---------|---------------------|
| Equity Index | 3:00pm (daily) |
| Gold, Silver | 12:30pm (daily) |
| Crude Oil | 1:30pm (daily) |
| Currency | 2:00pm (daily) |

---

## 8. Contract Rollover

### 8.1 Expiration Months

| Product | Active Months |
|---------|--------------|
| ES, NQ | H, M, U, Z (Mar, Jun, Sep, Dec) |
| GC | G, J, M, Q, V, Z (bi-monthly) |
| CL | All months (monthly) |
| 6E | H, M, U, Z (quarterly) |

### 8.2 Standard Roll Dates

- **Index Futures**: Roll ~8 days before expiry (2nd Thursday)
- **Gold**: Roll ~4 days before expiry
- **Crude Oil**: Roll 3-4 days before expiry
- **Currency**: Roll ~2 days before expiry

### 8.3 Continuous Contracts

```python
# Use CONTFUT for auto-rolled continuous data
contract.secType = "CONTFUT"
```

---

## 9. Price Limits & Circuit Breakers

### 9.1 Equity Index Limits

| Level | Trigger | Action |
|-------|---------|--------|
| Level 1 | -7% | 15-min halt |
| Level 2 | -13% | 15-min halt |
| Level 3 | -20% | Market closed |

### 9.2 Limit Up/Limit Down

| Product | Daily Limit |
|---------|-------------|
| ES | 7% / 13% / 20% |
| GC | $150/oz |
| CL | $15/barrel |
| 6E | No limit |

---

## 10. Python Implementation with ib_insync

### 10.1 Basic Connection

```python
from ib_insync import IB, Future

# Connect
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # Paper trading

# Define contract
es = Future('ES', '202403', 'CME')

# Get contract details
details = ib.reqContractDetails(es)

# Request market data
ib.reqMktData(es)

# Stream updates
def onTick(ticker):
    print(f"Bid: {ticker.bid}, Ask: {ticker.ask}")

es_ticker = ib.ticker(es)
es_ticker.updateEvent += onTick

# Historical data
bars = ib.reqHistoricalData(
    es,
    endDateTime='',
    durationStr='1 D',
    barSizeSetting='5 mins',
    whatToShow='TRADES',
    useRTH=False
)
```

### 10.2 Order Management

```python
from ib_insync import MarketOrder, LimitOrder, StopOrder

# Market order
order = MarketOrder('BUY', 1)
trade = ib.placeOrder(es, order)

# Limit order
order = LimitOrder('BUY', 1, 4500.00)
trade = ib.placeOrder(es, order)

# Bracket order
bracket = ib.bracketOrder(
    'BUY', 1, 4500.00,  # Entry
    4600.00,             # Take profit
    4400.00              # Stop loss
)
for order in bracket:
    ib.placeOrder(es, order)
```

---

## 11. Key Differences from Binance

| Aspect | Binance Futures | CME via IB |
|--------|-----------------|------------|
| **Funding** | Every 8h | None (daily settlement) |
| **Margin** | Tiered brackets | SPAN (portfolio) |
| **Hours** | 24/7 | 23/5 with maintenance |
| **Expiration** | Perpetual available | All expire |
| **Max Leverage** | 125x | ~20x |
| **Settlement** | Continuous | Daily at 4pm CT |
| **Position Mode** | Hedge/One-way | Always net |

---

## 12. Required Dependencies

```bash
pip install ib_insync
# or
pip install ibapi  # Official IB API
```

---

## 13. References

- **IB API Docs**: https://interactivebrokers.github.io/tws-api/
- **ib_insync**: https://ib-insync.readthedocs.io/
- **CME Group**: https://www.cmegroup.com/
- **SPAN Margin**: https://www.cmegroup.com/clearing/risk-management/span.html

---

## 14. Next Steps (Phase 3B)

1. Create `adapters/interactive_brokers/__init__.py`
2. Create `adapters/interactive_brokers/market_data.py`
3. Create `adapters/interactive_brokers/order_execution.py`
4. Create `adapters/interactive_brokers/exchange_info.py`
5. Create `adapters/interactive_brokers/margin_calculator.py`
6. Add connection management with auto-reconnect
7. Implement continuous contract handling
