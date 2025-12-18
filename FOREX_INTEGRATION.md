# Forex Asset Class Integration Plan - L3 Architecture Analysis

**Date**: 2025-11-29  
**Status**: Architectural Design Document  
**Scope**: Comprehensive Forex integration at L3 execution level

## Executive Summary

This document provides a detailed analysis of the CustodiaCloud codebase architecture and a comprehensive integration plan for adding Forex as a new asset class. The platform is built on a layered, adapter-based architecture that supports multiple asset classes (equities-first, with optional digital assets) at both L2 (statistical models) and L3 (full LOB simulation) levels.

**Key Findings:**
1. **Plugin Architecture**: Adapters use registry-based system making new exchanges/asset classes easy to integrate
2. **L3 Execution Ready**: L3 LOB simulation infrastructure supports arbitrary asset classes
3. **Asset Class Abstractions**: Clear boundaries between crypto/equity allow extension
4. **Trading Hours Aware**: Built-in support for non-24/7 markets (essential for Forex)
5. **Fee Computation Framework**: Extensible fee models for different market structures

---

## Part 1: Core Architecture Overview

### 1.1 Layered Architecture

```
script_* (CLI Entry) → service_* (Business Logic) → impl_* (Implementation) 
→ core_* (Domain Models)

NO circular dependencies. Each layer only imports from layers below.
```

### 1.2 Asset Class Definition

Already defined in codebase:
- `adapters/models.py:MarketType.FOREX` ✓ Already exists!
- `execution_providers.py:AssetClass` (need to add FOREX)
- `data_loader_multi_asset.py:AssetClass.FOREX` ✓ Already exists!

---

## Part 2: Adapter Architecture

### 2.1 Registry System

**Location**: `adapters/registry.py`

Two-level mapping:
1. Vendor Level: ExchangeVendor → adapter module  
2. Interface Level: AdapterType → implementation class

### 2.2 Core Adapter Interfaces

| Interface | Purpose |
|-----------|---------|
| MarketDataAdapter | OHLCV bars, ticks, real-time |
| FeeAdapter | Fee computation |
| TradingHoursAdapter | Market hours, holidays |
| OrderExecutionAdapter | Place/cancel orders, positions |
| ExchangeInfoAdapter | Symbol rules, exchange metadata |

### 2.3 Implementation Pattern

Each vendor package contains:
- `market_data.py` - MarketDataAdapter
- `fees.py` - FeeAdapter
- `trading_hours.py` - TradingHoursAdapter
- `exchange_info.py` - ExchangeInfoAdapter
- `__init__.py` - Registration hook

---

## Part 3: Forex-Specific Considerations

### 3.1 Market Characteristics

| Aspect | Forex |
|--------|-------|
| Hours | Sun 5pm - Fri 4pm ET (~24/5) |
| Sessions | Tokyo (5pm-2am), London (2am-11am), NY (8am-5pm) ET |
| Tick Size | 0.0001 pips (majors), 0.01 (JPY pairs) |
| Fee Structure | Spread-based (no explicit commission) |
| Min Trade | 1,000 units (micro: 100 units) |
| Leverage | 50:1 typical, up to 100:1 |
| Liquidity | OTC market (dealer-based) |

### 3.2 Key Integration Points

1. **Trading Hours**: Implement ForexTradingHoursAdapter with 3 sessions
2. **Fee Computation**: Return 0.0 (spread is implicit)
3. **Slippage Model**: Add ForexParametricSlippageProvider (6-8 factors)
4. **Data Loading**: Filter non-trading hours, add FX-specific features
5. **Session Router**: Route orders to best-liquidity session

---

## Part 4: Implementation Roadmap (7 weeks)

### Week 1-2: Foundation

**Changes**:
```python
# adapters/models.py
class ExchangeVendor(str, Enum):
    OANDA = "oanda"  # ← Add
    INTERACTIVE_BROKERS = "interactive_brokers"

# adapters/registry.py
_lazy_modules: Dict[ExchangeVendor, str] = {
    ExchangeVendor.OANDA: "adapters.oanda",  # ← Add

# execution_providers.py  
class AssetClass(enum.Enum):
    FOREX = "forex"  # ← Add

# script_live.py
VENDOR_TO_ASSET_CLASS: Dict[str, str] = {
    "oanda": "forex",  # ← Add
}
```

### Week 2-3: Parametric TCA

Implement `ForexParametricSlippageProvider` with factors:
1. √Participation impact
2. Time-of-day (session liquidity)
3. Volatility regime
4. Correlation decay (exotics)
5. Interest rate carry
6. Economic news impact

**~500 lines, 50+ tests**

### Week 3-4: OANDA Adapter

Create `adapters/oanda/`:
- `market_data.py` (REST + WebSocket)
- `trading_hours.py` (3 sessions)
- `fees.py` (return 0.0)
- `exchange_info.py` (forex symbols)
- `__init__.py` (registration)

**~1200 lines total**

### Week 4: Data Loading

Update `data_loader_multi_asset.py`:
- Filter trading hours
- Add forex features (carry, volatility, risk sentiment)
- Handle weekend gaps
- Validate feature completeness

**~200 lines, 20+ tests**

### Week 5: L3 Execution (Optional Phase 1)

Add L3ExecutionConfig.for_forex() and simplified matching.

**~300 lines**

### Week 6: Training & Backtesting

Create configs:
- `config_train_forex.yaml`
- `config_backtest_forex.yaml`
- `execution_l3_forex.yaml`

Test end-to-end: data → training → backtest

### Week 7: Live Trading

Implement:
- `ForexSessionRouter`
- Paper trading integration
- Position synchronization
- Order execution with session awareness

---

## Part 5: Key Files to Modify

| File | Changes | Priority |
|------|---------|----------|
| `adapters/models.py` | Add OANDA vendor enum | HIGH |
| `adapters/registry.py` | Add lazy loading | HIGH |
| `execution_providers.py` | ForexParametricSlippageProvider | HIGH |
| `script_live.py` | Asset class detection | MEDIUM |
| `data_loader_multi_asset.py` | Forex feature loading | MEDIUM |
| `features_pipeline.py` | FX-specific indicators | MEDIUM |
| `services/session_router.py` | ForexSessionRouter | LOW |

**New Directories**:
- `adapters/oanda/` (~1200 lines)
- `tests/test_forex_*.py` (~1500 lines)
- Config files (~50 lines)

---

## Part 6: Dependencies

### Python Packages
```bash
pip install oanda-v20  # OANDA SDK
```

### Environment Variables
```
OANDA_API_KEY=...
OANDA_ACCOUNT_ID=...
OANDA_ENVIRONMENT=practice|live
```

---

## Part 7: Success Criteria

- [ ] OANDA adapters implement all 5 interfaces
- [ ] ForexParametricSlippageProvider handles 6+ factors
- [ ] Major pairs (EUR/USD, GBP/USD, USD/JPY) trade with 1-5 bps slippage
- [ ] Trading hours correctly filter weekends
- [ ] Model trains on forex data (>100K steps)
- [ ] Paper trading executes orders successfully
- [ ] No look-ahead bias in features

---

## Part 8: File-by-File Checklist

### Create (Weeks 1-4):
- `adapters/oanda/__init__.py` (120 lines)
- `adapters/oanda/market_data.py` (350 lines)
- `adapters/oanda/fees.py` (100 lines)
- `adapters/oanda/trading_hours.py` (250 lines)
- `adapters/oanda/exchange_info.py` (200 lines)
- `tests/test_forex_adapter_oanda.py` (400 lines)
- `tests/test_forex_parametric_tca.py` (600 lines)
- `tests/test_forex_data_loading.py` (300 lines)
- `configs/config_train_forex.yaml`
- `configs/config_backtest_forex.yaml`
- `configs/config_live_oanda.yaml`

### Modify (Weeks 1-4):
- `adapters/models.py` - +5 lines
- `adapters/registry.py` - +3 lines
- `execution_providers.py` - +500 lines (ForexParametricSlippageProvider)
- `data_loader_multi_asset.py` - +200 lines
- `script_live.py` - +10 lines
- `features_pipeline.py` - +150 lines

**Total**: ~5000 LOC to implement complete Forex support

---

## Summary

The codebase is **exceptionally well-designed** for Forex integration:

✓ Plugin-based adapter system (no core changes needed)
✓ L3 LOB simulation ready for any asset class
✓ FOREX already in MarketType enum (designed for this!)
✓ Parametric TCA framework ready to extend
✓ Trading hours abstraction supports any schedule
✓ Config-driven design (YAML-based)

Estimated effort: **6-7 weeks** with 1-2 engineers
