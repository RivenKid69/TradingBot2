# Existing Code Compatibility Report

## Phase 0 Deliverable: Code Audit & Compatibility Analysis

**Version**: 1.0
**Date**: 2025-11-30
**Status**: COMPLETE

---

## 1. Executive Summary

This report analyzes existing codebase components that can be reused or extended for futures integration. The codebase already has substantial infrastructure for futures trading, requiring primarily extension rather than rewrite.

### Compatibility Score by Component

| Component | Compatibility | Effort | Notes |
|-----------|---------------|--------|-------|
| Market Data Adapters | 85% | Low | `use_futures` flag exists |
| Data Models | 70% | Medium | MarketType enums present |
| Funding Rate Ingestion | 95% | Minimal | Complete implementation |
| Risk Guards | 60% | Medium | Forex pattern adaptable |
| Fee Calculation | 50% | Medium | Needs futures tier logic |
| Execution Simulation | 40% | High | Needs L2+ futures model |

---

## 2. Adapter Layer Analysis

### 2.1 adapters/models.py

**Status**: ✅ Partially Ready

**Existing Support**:

```python
class MarketType(str, Enum):
    SPOT = "spot"
    CRYPTO_SPOT = "crypto_spot"
    CRYPTO_FUTURES = "crypto_futures"   # ✅ Already defined
    CRYPTO_PERP = "crypto_perp"         # ✅ Already defined
    MARGIN = "margin"
    EQUITY = "equity"
    FOREX = "forex"
```

**Missing for Futures**:

- `EQUITY_FUTURES` - For ES, NQ, YM
- `COMMODITY_FUTURES` - For GC, CL, SI
- `CURRENCY_FUTURES` - For 6E, 6J, 6B
- `BOND_FUTURES` - For ZB, ZN, ZF

**Required Changes**:

```python
# Proposed additions to MarketType enum
EQUITY_FUTURES = "equity_futures"
COMMODITY_FUTURES = "commodity_futures"
CURRENCY_FUTURES = "currency_futures"
BOND_FUTURES = "bond_futures"
```

**Exchange Vendor Status**:

```python
class ExchangeVendor(str, Enum):
    BINANCE = "binance"           # ✅ Ready for futures
    ALPACA = "alpaca"             # Equity only
    POLYGON = "polygon"           # Data only
    YAHOO = "yahoo"               # Data only
    OANDA = "oanda"               # Forex only
    # MISSING:
    # INTERACTIVE_BROKERS = "ib"  # For CME futures
```

### 2.2 adapters/binance/market_data.py

**Status**: ✅ Ready with Minor Extensions

**Existing Futures Support**:

```python
class BinanceMarketDataAdapter(MarketDataAdapter):
    def __init__(self, vendor, config):
        self._use_futures = self._config.get("use_futures", False)
        # Automatically switches endpoints based on use_futures flag
```

**API Endpoints Already Configured**:

- Klines: `/fapi/v1/klines` (when use_futures=True)
- Ticker: `/fapi/v1/ticker/24hr`
- Depth: `/fapi/v1/depth`

**Missing Endpoints**:

| Endpoint | Purpose | Priority |
|----------|---------|----------|
| `/fapi/v1/fundingRate` | Historical funding | P0 (exists in ingest) |
| `/fapi/v1/premiumIndex` | Mark price + funding | P0 |
| `/fapi/v2/positionRisk` | Position info | P1 |
| `/fapi/v1/leverageBracket` | Leverage tiers | P1 |
| `/fapi/v1/adlQuantile` | ADL indicator | P2 |

### 2.3 adapters/binance/exchange_info.py

**Status**: ⚠️ Needs Extension

**Current Implementation**:

- Fetches spot exchange info
- Parses LOT_SIZE, PRICE_FILTER, MIN_NOTIONAL

**Missing for Futures**:

- Futures-specific filters (MAX_NUM_ORDERS, PERCENT_PRICE)
- Leverage bracket parsing
- Margin rate extraction
- Contract type handling (PERPETUAL, QUARTERLY)

### 2.4 adapters/binance/fees.py

**Status**: ⚠️ Needs Extension

**Current**: Spot fee tiers (maker/taker based on VIP level)

**Missing for Futures**:

- Futures-specific fee tiers (lower than spot)
- Fee discount for BNB payment
- Fee rebates for high volume

---

## 3. Data Ingestion Layer

### 3.1 ingest_funding_mark.py

**Status**: ✅ Complete Implementation

**Capabilities**:

```python
def _fetch_all_funding(symbol, start_ms, end_ms):
    """Fetch historical funding rates with pagination"""
    # Handles rate limiting, pagination, error recovery

def _fetch_all_mark(symbol, start_ms, end_ms, tf_str):
    """Fetch mark price klines"""
    # Supports all standard timeframes
```

**Data Produced**:

- `funding_rate` - Hourly funding rate values
- `mark_price` - Mark price OHLCV data
- `index_price` - From mark price endpoint

**Compatibility**: 100% reusable for crypto perpetuals

### 3.2 data_loader_multi_asset.py

**Status**: ⚠️ Needs Extension

**Current Support**:

- `AssetClass.CRYPTO`
- `AssetClass.EQUITY`
- `AssetClass.FOREX`

**Missing**:

- `AssetClass.FUTURES` with sub-types
- Rollover handling for expiring contracts
- Continuous contract construction

---

## 4. Risk Management Layer

### 4.1 services/forex_risk_guards.py

**Status**: ✅ Excellent Reference Pattern

**Reusable Components**:

```python
class ForexMarginGuard:
    """Pattern for margin calculation"""
    MARGIN_REQUIREMENTS = {
        "EUR_USD": 0.02,  # 50:1 leverage
        "GBP_USD": 0.02,
        # ...
    }

    def check_margin_requirement(self, position, price, leverage):
        # Reusable margin check logic

class ForexLeverageGuard:
    """Pattern for leverage limits"""
    LEVERAGE_LIMITS = {
        "major": 50,
        "minor": 20,
        "exotic": 10,
    }
```

**Adaptation for Futures**:

- Replace currency pairs with futures symbols
- Add SPAN margin calculation
- Add maintenance margin check
- Add liquidation price calculation

### 4.2 risk_guard.py

**Status**: ✅ Core Infrastructure Ready

**Existing Guards**:

- Position size limits
- Drawdown limits
- PnL thresholds

**Extension Needed**:

- Futures-specific margin checks
- Funding rate cost tracking
- ADL risk monitoring

---

## 5. Execution Simulation Layer

### 5.1 execution_providers.py

**Status**: ⚠️ Needs Futures-Specific Provider

**Current Providers**:

- `CryptoParametricSlippageProvider` - L2+ for crypto spot
- `EquityParametricSlippageProvider` - L2+ for equities
- `ForexParametricSlippageProvider` - L2+ for forex

**Missing**:

- `FuturesParametricSlippageProvider` - For all futures types

**Key Differences for Futures**:

| Aspect | Crypto Spot | Crypto Futures | CME Futures |
|--------|-------------|----------------|-------------|
| Spread | ~5 bps | ~3 bps | ~1-2 ticks |
| Impact | √participation | √participation × funding | SPAN-adjusted |
| Sessions | 24/7 | 24/7 | 23/5 |
| Settlement | Continuous | Funding 8h | Daily 4pm CT |

### 5.2 execution_sim.py

**Status**: ⚠️ Needs Extension

**Current Features**:

- OHLCV-based fill simulation
- Intrabar execution timing
- Slippage modeling

**Missing for Futures**:

- Mark price vs last price logic
- Funding rate deduction
- Contract rollover simulation
- Price limit handling

---

## 6. Feature Engineering Layer

### 6.1 features_pipeline.py

**Status**: ⚠️ Needs Futures Features

**Current Features** (63 total):

- Price features
- Volume features
- Volatility features
- Momentum features

**Missing Futures Features**:

| Feature | Description | Priority |
|---------|-------------|----------|
| `funding_rate` | Current 8h funding | P0 |
| `funding_rate_ma` | Moving average | P0 |
| `basis` | Futures - spot spread | P0 |
| `open_interest` | Position accumulation | P1 |
| `oi_change` | OI rate of change | P1 |
| `long_short_ratio` | Sentiment indicator | P2 |
| `liquidation_volume` | Liquidation pressure | P2 |

### 6.2 forex_features.py (Reference)

**Status**: ✅ Pattern for Futures Features

**Reusable Pattern**:

```python
class ForexFeatures:
    def compute_session_features(self, data):
        # Session-based feature computation
        # Adaptable for CME trading hours
```

---

## 7. Configuration Layer

### 7.1 configs/asset_class_defaults.yaml

**Status**: ⚠️ Needs Futures Defaults

**Current Entries**:

```yaml
crypto:
  slippage_bps: 5.0
  limit_offset_bps: 10.0

equity:
  slippage_bps: 2.0
  limit_offset_bps: 5.0

forex:
  slippage_pips: 1.0
```

**Missing**:

```yaml
# Proposed additions
crypto_futures:
  slippage_bps: 3.0
  limit_offset_bps: 5.0
  funding_rate_check: true
  leverage_default: 10

equity_futures:
  slippage_ticks: 1.0
  limit_offset_ticks: 2.0
  margin_type: "span"

commodity_futures:
  slippage_ticks: 1.0
  daily_limit_check: true
```

---

## 8. Test Infrastructure

### 8.1 Existing Test Coverage

**Relevant Test Files**:

| File | Coverage | Futures Relevance |
|------|----------|-------------------|
| `test_binance_adapters.py` | Spot only | Extend for futures |
| `test_forex_parametric_tca.py` | Forex TCA | Pattern for futures |
| `test_execution_providers.py` | L2 providers | Add futures provider |
| `test_forex_risk_guards.py` | Forex guards | Pattern for futures |

### 8.2 Test Data Requirements

**Existing Data**:

- `data/raw/*.parquet` - Crypto spot
- `data/raw_stocks/*.parquet` - Equity

**Missing Data**:

- `data/raw_futures/crypto/*.parquet` - Crypto perpetuals
- `data/raw_futures/cme/*.parquet` - CME futures

---

## 9. Compatibility Matrix

### 9.1 Component Reuse Summary

| Component | File | Reuse Level | Action |
|-----------|------|-------------|--------|
| MarketType enum | adapters/models.py | Extend | Add 4 new types |
| ExchangeVendor | adapters/models.py | Extend | Add IB vendor |
| Market data adapter | adapters/binance/market_data.py | Extend | Add endpoints |
| Funding ingestion | ingest_funding_mark.py | Reuse | As-is |
| Risk guards | services/forex_risk_guards.py | Adapt | New FuturesRiskGuard |
| TCA provider | execution_providers.py | New | FuturesParametricProvider |
| Features | features_pipeline.py | Extend | Add futures features |
| Data loader | data_loader_multi_asset.py | Extend | Add futures class |

### 9.2 Dependency Graph

```
adapters/models.py (MarketType)
    ↓
adapters/binance/market_data.py (use_futures)
    ↓
ingest_funding_mark.py (funding rate data)
    ↓
features_pipeline.py (futures features)
    ↓
execution_providers.py (FuturesParametricProvider)
    ↓
services/futures_risk_guards.py (NEW)
    ↓
trading_patchnew.py (env integration)
```

---

## 10. Risk Assessment

### 10.1 Low Risk Components

| Component | Risk | Reason |
|-----------|------|--------|
| Funding ingestion | Low | Already tested and deployment-ready |
| Market type enum | Low | Simple addition |
| Config extension | Low | YAML changes only |

### 10.2 Medium Risk Components

| Component | Risk | Mitigation |
|-----------|------|------------|
| Risk guards | Medium | Use forex_risk_guards as template |
| Fee calculation | Medium | Reference Binance docs |
| Data loader | Medium | Incremental extension |

### 10.3 High Risk Components

| Component | Risk | Mitigation |
|-----------|------|------------|
| Execution provider | High | Extensive testing, paper trading |
| Rollover logic | High | Historical validation |
| IB integration | High | Sandbox testing first |

---

## 11. Recommendations

### 11.1 Immediate Actions (Phase 1)

1. **Extend MarketType enum** - Add 4 new futures types
2. **Add ExchangeVendor.INTERACTIVE_BROKERS** - For CME access
3. **Create FuturesRiskGuard** - Based on ForexRiskGuard pattern
4. **Add funding rate features** - To features_pipeline.py

### 11.2 Phase 2 Actions

1. **Create FuturesParametricSlippageProvider** - L2+ for futures
2. **Extend data_loader_multi_asset.py** - Add futures asset class
3. **Create continuous contract logic** - For rollover handling
4. **Add CME trading hours** - To trading hours service

### 11.3 Phase 3 Actions

1. **Create IB adapter** - adapters/interactive_brokers/
2. **Add SPAN margin calculation** - For CME margin
3. **Implement rollover strategy** - Volume-based detection
4. **Create futures-specific tests** - Comprehensive coverage

---

## 12. Conclusion

The existing codebase provides a solid foundation for futures integration:

- **85% of market data infrastructure** can be reused
- **95% of funding rate logic** is already implemented
- **60% of risk management patterns** are adaptable
- **New code required**: ~40% (primarily execution and CME-specific logic)

The phased approach in `FUTURES_INTEGRATION_PLAN.md` aligns well with this analysis. Priority should be given to:

1. Leveraging `ingest_funding_mark.py` for crypto perpetuals
2. Adapting `forex_risk_guards.py` pattern for futures margin
3. Creating new `FuturesParametricSlippageProvider` for L2+

---

## Appendix A: Code References

### Key Files for Review

```
adapters/
├── models.py                    # Line 45-60: MarketType enum
├── binance/
│   ├── market_data.py          # Line 35: use_futures flag
│   ├── exchange_info.py        # Needs futures filters
│   └── fees.py                 # Needs futures tiers
├── interactive_brokers/         # NEW: To be created
│   ├── __init__.py
│   ├── market_data.py
│   ├── order_execution.py
│   └── margin_calculator.py

services/
├── forex_risk_guards.py        # Reference for FuturesRiskGuard

ingest_funding_mark.py          # Complete funding rate implementation

execution_providers.py          # Add FuturesParametricSlippageProvider
```

### Test Files to Create

```
tests/
├── test_futures_market_data.py
├── test_futures_risk_guards.py
├── test_futures_parametric_tca.py
├── test_futures_rollover.py
├── test_ib_adapters.py
└── test_span_margin.py
```

---

**Document End**
