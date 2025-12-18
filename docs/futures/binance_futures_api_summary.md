# Binance Futures API Documentation Summary

## Phase 0 Deliverable: API Analysis (Crypto Track)

**Version**: 1.0
**Date**: 2025-11-30
**Status**: RESEARCH COMPLETE

---

## 1. API Overview

Binance offers two types of futures:
- **USDT-M Futures** (USDT-margined perpetual/delivery) - Primary target
- **COIN-M Futures** (coin-margined delivery) - Secondary target

### Base URLs

| Environment | USDT-M Futures | COIN-M Futures |
|-------------|----------------|----------------|
| Production | `https://fapi.binance.com` | `https://dapi.binance.com` |
| Testnet | `https://testnet.binancefuture.com` | `https://testnet.binancefuture.com` |

### Rate Limits

| Limit Type | Value | Scope |
|------------|-------|-------|
| Request Weight | 2400/min | IP |
| Order Rate | 1200/min | IP |
| Order Rate | 10/sec | Symbol |

---

## 2. Market Data Endpoints (Public)

### 2.1 Exchange Information
```
GET /fapi/v1/exchangeInfo
```
Returns trading rules and symbol information.

**Key Fields per Symbol:**
- `symbol` - Trading pair (e.g., "BTCUSDT")
- `pair` - Underlying pair
- `contractType` - "PERPETUAL" | "CURRENT_QUARTER" | "NEXT_QUARTER"
- `deliveryDate` - Delivery timestamp (0 for perpetual)
- `onboardDate` - When symbol was listed
- `status` - "TRADING" | "SETTLING" | "CLOSE"
- `maintMarginPercent` - Maintenance margin rate
- `requiredMarginPercent` - Initial margin rate
- `baseAsset`, `quoteAsset`, `marginAsset`
- `pricePrecision`, `quantityPrecision`
- `filters` - LOT_SIZE, PRICE_FILTER, MIN_NOTIONAL, etc.

**Existing Code**: `binance_public.py:get_exchange_info()`

### 2.2 Klines (Candlesticks)
```
GET /fapi/v1/klines
GET /fapi/v1/markPriceKlines  # Mark price candles
GET /fapi/v1/indexPriceKlines # Index price candles
GET /fapi/v1/continuousKlines # Continuous contract klines
```

**Parameters:**
- `symbol` (required) - e.g., "BTCUSDT"
- `interval` - 1m, 5m, 15m, 1h, 4h, 1d, 1w, 1M
- `startTime`, `endTime` - Timestamp in ms
- `limit` - Max 1500, default 500

**Existing Code**: `ingest_funding_mark.py:_fetch_all_mark()`

### 2.3 Funding Rate
```
GET /fapi/v1/fundingRate
```
Historical funding rate data.

**Parameters:**
- `symbol` (required)
- `startTime`, `endTime` - Filter range
- `limit` - Max 1000

**Response:**
```json
[
  {
    "symbol": "BTCUSDT",
    "fundingTime": 1698422400000,
    "fundingRate": "0.00010000",
    "markPrice": "34500.00000000"
  }
]
```

**Existing Code**: `ingest_funding_mark.py:_fetch_all_funding()`

### 2.4 Premium Index (Mark Price + Funding)
```
GET /fapi/v1/premiumIndex
```
Real-time mark price and funding rate.

**Response per Symbol:**
```json
{
  "symbol": "BTCUSDT",
  "markPrice": "34523.12345678",
  "indexPrice": "34520.00000000",
  "estimatedSettlePrice": "34521.00000000",
  "lastFundingRate": "0.00010000",
  "interestRate": "0.00010000",
  "nextFundingTime": 1698451200000,
  "time": 1698422412345
}
```

### 2.5 Order Book Depth
```
GET /fapi/v1/depth
```
L2 order book snapshot.

**Parameters:**
- `symbol` (required)
- `limit` - 5, 10, 20, 50, 100, 500, 1000

### 2.6 24hr Ticker
```
GET /fapi/v1/ticker/24hr
```
Rolling 24h statistics.

**Key Fields:**
- `openPrice`, `highPrice`, `lowPrice`, `lastPrice`
- `volume`, `quoteVolume`
- `openInterest` - Current open interest
- `priceChange`, `priceChangePercent`
- `weightedAvgPrice` - VWAP

### 2.7 Open Interest
```
GET /fapi/v1/openInterest
GET /futures/data/openInterestHist  # Historical OI
```

---

## 3. Account & Trade Endpoints (Authenticated)

### 3.1 Account Information
```
GET /fapi/v2/account
```
**Key Fields:**
```json
{
  "totalWalletBalance": "10000.00",
  "totalUnrealizedProfit": "500.00",
  "totalMarginBalance": "10500.00",
  "availableBalance": "8000.00",
  "maxWithdrawAmount": "7500.00",
  "positions": [...],
  "assets": [...]
}
```

### 3.2 Position Information
```
GET /fapi/v2/positionRisk
```
**Response per Position:**
```json
{
  "symbol": "BTCUSDT",
  "positionAmt": "0.100",
  "entryPrice": "34000.00",
  "markPrice": "34523.00",
  "unRealizedProfit": "52.30",
  "liquidationPrice": "30000.00",
  "leverage": "20",
  "maxNotionalValue": "5000000",
  "marginType": "cross",
  "isolatedMargin": "0",
  "isAutoAddMargin": "false",
  "positionSide": "BOTH",
  "notional": "3452.30",
  "isolatedWallet": "0"
}
```

### 3.3 Order Submission
```
POST /fapi/v1/order
```
**Parameters:**
- `symbol` (required)
- `side` - BUY | SELL
- `type` - LIMIT, MARKET, STOP, STOP_MARKET, TAKE_PROFIT, TAKE_PROFIT_MARKET, TRAILING_STOP_MARKET
- `quantity` (required)
- `price` - Required for LIMIT
- `positionSide` - BOTH (one-way) | LONG | SHORT (hedge mode)
- `reduceOnly` - true/false
- `timeInForce` - GTC, IOC, FOK, GTX
- `newClientOrderId` - Custom order ID
- `stopPrice` - For stop orders
- `closePosition` - true to close position
- `activationPrice` - For trailing stop
- `callbackRate` - For trailing stop (1.0 = 1%)
- `workingType` - MARK_PRICE | CONTRACT_PRICE

### 3.4 Leverage Settings
```
POST /fapi/v1/leverage
```
**Parameters:**
- `symbol` (required)
- `leverage` - 1 to 125

**Response:**
```json
{
  "leverage": 20,
  "maxNotionalValue": "5000000",
  "symbol": "BTCUSDT"
}
```

### 3.5 Margin Type
```
POST /fapi/v1/marginType
```
**Parameters:**
- `symbol` (required)
- `marginType` - ISOLATED | CROSSED

### 3.6 Position Mode
```
POST /fapi/v1/positionSide/dual
```
**Parameters:**
- `dualSidePosition` - true (hedge mode) | false (one-way mode)

---

## 4. Risk Management Endpoints

### 4.1 Leverage Brackets
```
GET /fapi/v1/leverageBracket
```
Tiered leverage limits by notional.

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "brackets": [
    {
      "bracket": 1,
      "initialLeverage": 125,
      "notionalCap": 10000,
      "notionalFloor": 0,
      "maintMarginRatio": 0.004,
      "cum": 0
    },
    {
      "bracket": 2,
      "initialLeverage": 100,
      "notionalCap": 50000,
      "notionalFloor": 10000,
      "maintMarginRatio": 0.005,
      "cum": 10
    }
    // ... more brackets
  ]
}
```

### 4.2 ADL Quantile
```
GET /fapi/v1/adlQuantile
```
Auto-deleveraging indicator (1-5, higher = more risk).

### 4.3 Force Orders (Liquidations)
```
GET /fapi/v1/forceOrders
```
User's liquidation history.

### 4.4 Income History
```
GET /fapi/v1/income
```
Funding payments, commissions, realized PnL.

**Income Types:**
- TRANSFER
- WELCOME_BONUS
- REALIZED_PNL
- FUNDING_FEE
- COMMISSION
- INSURANCE_CLEAR
- REFERRAL_KICKBACK
- COMMISSION_REBATE
- API_REBATE

---

## 5. WebSocket Streams

### 5.1 Market Data Streams
```
wss://fstream.binance.com/ws/<streamName>
wss://fstream.binance.com/stream?streams=<streamName1>/<streamName2>
```

**Available Streams:**
| Stream | Description |
|--------|-------------|
| `<symbol>@aggTrade` | Aggregated trades |
| `<symbol>@markPrice@1s` | Mark price (1s or 3s) |
| `<symbol>@kline_<interval>` | Candlestick |
| `<symbol>@miniTicker` | Mini 24hr ticker |
| `<symbol>@ticker` | 24hr ticker |
| `<symbol>@depth<levels>@<speed>` | Order book (5/10/20, 100ms/250ms/500ms) |
| `<symbol>@forceOrder` | Liquidation orders |
| `!forceOrder@arr` | All liquidations |

### 5.2 User Data Stream
```
POST /fapi/v1/listenKey  # Create listen key
PUT /fapi/v1/listenKey   # Keepalive (every 60min)
DELETE /fapi/v1/listenKey # Close
```

**Events:**
- `MARGIN_CALL` - Margin call warning
- `ACCOUNT_UPDATE` - Balance/position changes
- `ORDER_TRADE_UPDATE` - Order/trade updates
- `ACCOUNT_CONFIG_UPDATE` - Leverage/margin mode changes
- `STRATEGY_UPDATE` - Grid/strategy automation updates

---

## 6. Key Concepts for Implementation

### 6.1 Mark Price Calculation
```
Mark Price = Index Price × (1 + Funding Basis)

Funding Basis = Average((Futures Price - Index Price) / Index Price)
              over 8-hour rolling window
```

### 6.2 Liquidation Price (Isolated)
```
Long:
  Liq Price = Entry × (1 - Initial Margin% + Maintenance Margin%)

Short:
  Liq Price = Entry × (1 + Initial Margin% - Maintenance Margin%)
```

### 6.3 Funding Rate
- Paid every 8 hours (00:00, 08:00, 16:00 UTC)
- Positive rate: Longs pay shorts
- Negative rate: Shorts pay longs
- Formula: `Payment = Position Size × Mark Price × Funding Rate`

### 6.4 ADL (Auto-Deleveraging)
Triggered when insurance fund depleted during extreme volatility:
1. Profitable positions ranked by profit and leverage
2. Highest ranked liquidated first against bankrupt positions
3. ADL indicator (1-5) shows queue position

---

## 7. Existing Code Compatibility

| Component | File | Futures Support | Notes |
|-----------|------|-----------------|-------|
| Market Data | `adapters/binance/market_data.py` | Partial | Has `use_futures` flag |
| Public Client | `binance_public.py` | Yes | Supports futures endpoints |
| Funding/Mark | `ingest_funding_mark.py` | Yes | Complete implementation |
| Exchange Info | `adapters/binance/exchange_info.py` | Partial | Needs futures filters |
| Fees | `adapters/binance/fees.py` | Partial | Needs futures fee tiers |

---

## 8. Missing Components (Phase 1+)

1. **Leverage Management**
   - `POST /fapi/v1/leverage` - Set leverage
   - `POST /fapi/v1/marginType` - Set cross/isolated
   - `GET /fapi/v1/leverageBracket` - Get brackets

2. **Position Management**
   - `GET /fapi/v2/positionRisk` - Current positions
   - `POST /fapi/v1/positionSide/dual` - Hedge mode

3. **Risk Data**
   - `GET /fapi/v1/adlQuantile` - ADL indicator
   - `GET /fapi/v1/forceOrders` - Liquidation history

4. **WebSocket Streams**
   - Mark price streams
   - Liquidation streams
   - User data streams (balance/position updates)

---

## 9. References

- **Official Docs**: https://binance-docs.github.io/apidocs/futures/en/
- **Testnet**: https://testnet.binancefuture.com
- **API Status**: https://www.binance.com/en/futures-activity/status

---

## 10. Next Steps (Phase 1)

1. Create `adapters/binance/futures_market_data.py`
2. Create `adapters/binance/futures_exchange_info.py`
3. Create `adapters/binance/futures_order_execution.py`
4. Extend `core_futures.py` with Binance-specific models
5. Add WebSocket support for mark price and liquidations
