# 🚀 UNIFIED FUTURES INTEGRATION PLAN

## Comprehensive L3-Level Multi-Asset Futures Trading Integration

**Версия**: 2.2
**Дата**: 2025-12-02
**Статус**: IN PROGRESS (Phase 9 Completed)
**Целевой реализм симуляции**: 95%+

---

## 📋 EXECUTIVE SUMMARY

### Цель

Интеграция **всех типов фьючерсов** на уровне L3 с полной симуляцией:

1. **Crypto Futures** (Binance USDT-M Perpetual & Quarterly)
2. **Equity Index Futures** (CME: ES, NQ, YM, RTY)
3. **Commodity Futures** (CME/COMEX: GC, CL, SI)
4. **Currency Futures** (CME: 6E, 6J, 6B)

### Поддерживаемые биржи/брокеры

| Тип | Биржа/Брокер | API | Статус |
|-----|--------------|-----|--------|
| **Crypto** | Binance Futures | REST + WebSocket | 🎯 Primary |
| **Crypto** | Bybit | REST + WebSocket | 📋 Future |
| **Equity Index** | CME via Interactive Brokers | TWS API | 🎯 Primary |
| **Equity Index** | CME via Alpaca | REST | 📋 Alternative |
| **Commodity** | CME via Interactive Brokers | TWS API | 🎯 Primary |
| **Currency** | CME via Interactive Brokers | TWS API | 🎯 Primary |

### Сравнение типов фьючерсов

| Аспект | Crypto Perpetual | Crypto Quarterly | Index Futures (ES) | Commodity (GC) | Currency (6E) |
|--------|------------------|------------------|-------------------|----------------|---------------|
| **Expiration** | Никогда | Mar/Jun/Sep/Dec | Mar/Jun/Sep/Dec | Monthly | Mar/Jun/Sep/Dec |
| **Settlement** | N/A | Cash (USDT) | Cash (USD) | Physical/Cash | Cash (USD) |
| **Funding** | Каждые 8ч | Нет | Нет | Нет | Нет |
| **Basis** | Minimal | Да | Да | Да | Да |
| **Trading Hours** | 24/7 | 24/7 | 23/5* | 23/5* | 23/5* |
| **Max Leverage** | 125x | 125x | 20x | 10x | 50x |
| **Tick Size** | Variable | Variable | 0.25 ($12.50) | 0.10 ($10) | 0.00005 ($6.25) |
| **Contract Size** | 1 unit | 1 unit | $50 × Index | 100 oz | €125,000 |
| **Initial Margin** | 0.8%-50% | 0.8%-50% | ~5-10% | ~5-10% | ~2-3% |

*CME: Sun 6pm - Fri 5pm ET с 15-минутным перерывом 4:15-4:30pm ET

### Ключевые концепции для всех типов

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED FUTURES CONCEPTS                      │
├─────────────────────────────────────────────────────────────────┤
│ ✅ Leverage & Margin          - All futures types               │
│ ✅ Mark Price vs Last Price   - For liquidation calculation     │
│ ✅ Long/Short positions       - Bidirectional trading           │
│ ✅ Contract specifications    - Tick size, multiplier, expiry   │
│ ✅ Settlement mechanics       - Cash vs physical                │
│ ✅ Rollover handling          - Contract expiration             │
├─────────────────────────────────────────────────────────────────┤
│ 🔸 Crypto-specific: Funding rates, Insurance fund, ADL         │
│ 🔸 CME-specific: Settlement times, daily limits, circuit breaks│
│ 🔸 Commodity-specific: Delivery months, storage costs          │
└─────────────────────────────────────────────────────────────────┘
```

### Существующий код для переиспользования

| Компонент | Файл | Применимость | Статус |
|-----------|------|--------------|--------|
| MarketType.CRYPTO_FUTURES | adapters/models.py | Crypto | ✅ Определён |
| crypto_futures defaults | ../configs/asset_class_defaults.yaml | Crypto | ✅ Базовый |
| Funding rate ingestion | ingest_funding_mark.py | Crypto | ✅ Готов |
| Forex leverage guards | services/forex_risk_guards.py | **Все типы** | ✅ Референс |
| Binance futures URL | adapters/binance/market_data.py | Crypto | ✅ Поддержка |
| L3 LOB simulation | lob/*.py | **Все типы** | ✅ Полностью |
| US market structure | lob/us_market_structure.py | CME futures | ✅ Tick size, circuit breakers |
| Equity parametric TCA | execution_providers.py | Index futures | ✅ Impact models |
| Session router | services/session_router.py | CME futures | ✅ Trading hours |

### Новые MarketType для добавления

```python
# adapters/models.py - расширение
class MarketType(str, Enum):
    # Существующие
    CRYPTO_SPOT = "CRYPTO_SPOT"
    CRYPTO_FUTURES = "CRYPTO_FUTURES"
    CRYPTO_PERP = "CRYPTO_PERP"
    EQUITY = "EQUITY"
    EQUITY_OPTIONS = "EQUITY_OPTIONS"
    FOREX = "FOREX"

    # НОВЫЕ для унифицированных фьючерсов
    INDEX_FUTURES = "INDEX_FUTURES"        # ES, NQ, YM, RTY
    COMMODITY_FUTURES = "COMMODITY_FUTURES" # GC, CL, SI, NG
    CURRENCY_FUTURES = "CURRENCY_FUTURES"   # 6E, 6J, 6B, 6A
    BOND_FUTURES = "BOND_FUTURES"          # ZB, ZN, ZF (Treasury)
```

### Новые ExchangeVendor

```python
# adapters/models.py - расширение
class ExchangeVendor(str, Enum):
    # Существующие
    BINANCE = "binance"
    ALPACA = "alpaca"
    OANDA = "oanda"

    # НОВЫЕ для CME фьючерсов
    INTERACTIVE_BROKERS = "interactive_brokers"  # TWS API
    TRADOVATE = "tradovate"                       # Alternative CME
    NINJATRADER = "ninjatrader"                   # Alternative CME
```

---

## 🏗️ УНИФИЦИРОВАННАЯ АРХИТЕКТУРА

### Принцип: Общее ядро + Vendor-специфичные адаптеры

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LAYER 0: UNIFIED CORE MODELS                         │
│  core_futures.py - Универсальные модели для ВСЕХ типов фьючерсов            │
│  ├── FuturesContract (symbol, expiry, multiplier, tick_size, margin_req)    │
│  ├── FuturesPosition (qty, entry, leverage, margin_mode, unrealized_pnl)    │
│  ├── MarginRequirement (initial, maintenance, variation)                    │
│  ├── SettlementInfo (type, date, price, method)                             │
│  └── ContractRollover (from_contract, to_contract, roll_date, adjustment)   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LAYER 1: VENDOR ADAPTERS                             │
│                                                                              │
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐   │
│  │   BINANCE FUTURES   │ │ INTERACTIVE BROKERS │ │     TRADOVATE       │   │
│  │  (Crypto Perpetual) │ │   (CME/COMEX/ICE)   │ │    (CME Alternative)│   │
│  ├─────────────────────┤ ├─────────────────────┤ ├─────────────────────┤   │
│  │ market_data.py      │ │ market_data.py      │ │ market_data.py      │   │
│  │ exchange_info.py    │ │ exchange_info.py    │ │ exchange_info.py    │   │
│  │ order_execution.py  │ │ order_execution.py  │ │ order_execution.py  │   │
│  │ funding_rates.py    │ │ settlement.py       │ │ settlement.py       │   │
│  │ liquidation.py      │ │ margin_req.py       │ │ margin_req.py       │   │
│  └─────────────────────┘ └─────────────────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LAYER 2: UNIFIED IMPLEMENTATION                           │
│                                                                              │
│  impl_futures_margin.py      - Margin calculator (ALL types)                │
│  ├── CryptoMarginCalculator     (Tiered brackets, isolated/cross)           │
│  ├── CMEMarginCalculator        (SPAN margin, performance bonds)            │
│  └── UnifiedMarginInterface     (Common API for all)                        │
│                                                                              │
│  impl_futures_settlement.py  - Settlement & Rollover                        │
│  ├── CryptoSettlement           (Funding payments)                          │
│  ├── CMESettlement              (Daily settlement, expiry)                  │
│  └── RolloverManager            (Contract roll handling)                    │
│                                                                              │
│  impl_futures_liquidation.py - Liquidation engine                           │
│  ├── CryptoLiquidation          (Mark price, insurance fund, ADL)           │
│  └── CMELiquidation             (Margin call → forced close)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LAYER 3: SERVICES                                    │
│                                                                              │
│  services/futures_risk_guards.py                                            │
│  ├── LeverageGuard              - Max leverage enforcement                  │
│  ├── MarginGuard                - Margin ratio monitoring                   │
│  ├── ConcentrationGuard         - Position limits                           │
│  ├── FundingExposureGuard       - Crypto: funding rate risk                 │
│  └── ExpirationGuard            - CME: contract expiry warnings             │
│                                                                              │
│  services/futures_position_manager.py                                       │
│  ├── PositionTracker            - Real-time position state                  │
│  ├── RolloverScheduler          - Auto-roll near expiry                     │
│  └── PnLCalculator              - Mark-to-market P&L                        │
│                                                                              │
│  services/futures_calendar.py                                               │
│  ├── CMETradingCalendar         - Trading hours, holidays                   │
│  ├── ExpirationCalendar         - Contract expiry dates                     │
│  └── RolloverCalendar           - Standard roll dates                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LAYER 4: EXECUTION PROVIDERS                              │
│                                                                              │
│  execution_providers_futures.py                                             │
│  ├── L2: FuturesParametricSlippageProvider                                  │
│  │   ├── CryptoFuturesSlippage  (Funding impact, liquidation cascade)       │
│  │   ├── IndexFuturesSlippage   (ES/NQ: high liquidity model)               │
│  │   └── CommodityFuturesSlippage (GC/CL: seasonal patterns)                │
│  │                                                                          │
│  └── L3: FuturesL3ExecutionProvider                                         │
│      ├── Uses existing lob/matching_engine.py                               │
│      ├── Liquidation order injection                                        │
│      └── Daily settlement simulation                                        │
│                                                                              │
│  futures_features.py - Type-specific features                               │
│  ├── CryptoFuturesFeatures      (funding_rate, oi, basis, liquidations)     │
│  ├── IndexFuturesFeatures       (roll_yield, term_structure, vix_corr)      │
│  └── CommodityFuturesFeatures   (contango, backwardation, seasonality)      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LAYER 5: ENTRY POINTS                                │
│                                                                              │
│  script_futures_backtest.py     - Unified backtest (all futures types)      │
│  script_futures_live.py         - Unified live trading                      │
│  train_model_multi_patch.py     - Extended with futures support             │
│                                                                              │
│  configs/                                                                   │
│  ├── config_train_crypto_futures.yaml                                       │
│  ├── config_train_index_futures.yaml                                        │
│  ├── config_train_commodity_futures.yaml                                    │
│  └── config_live_futures.yaml                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Vendor-Specific Considerations

#### Crypto Futures (Binance)

```
Уникальные особенности:
├── Perpetual contracts (no expiry)
├── Funding rate payments (8h intervals)
├── Insurance fund + ADL mechanism
├── Mark price = TWAP of index
├── Cross/Isolated margin modes
├── Tiered leverage brackets
└── 24/7 trading
```

#### CME Index Futures (ES, NQ)

```
Уникальные особенности:
├── Quarterly expiration (3rd Friday)
├── Daily settlement at 4pm ET
├── SPAN margin methodology
├── Price limits / circuit breakers
├── Cash settlement
├── Micro contracts available (MES, MNQ)
└── 23/5 trading with maintenance windows
```

#### Commodity Futures (GC, CL)

```
Уникальные особенности:
├── Monthly expiration
├── Physical delivery option (most roll before)
├── Contango/Backwardation dynamics
├── Storage cost implications
├── Seasonal patterns (heating oil, natural gas)
└── Different tick values by product
```

#### Currency Futures (6E, 6J)

```
Уникальные особенности:
├── Quarterly expiration
├── Cash settled in USD
├── Inverse relationship to forex spot
├── High leverage (up to 50x)
└── Correlation with forex markets
```

---

## 📅 ФАЗЫ РЕАЛИЗАЦИИ (ОБНОВЛЁННЫЕ)

### Структура фаз

```
┌─────────────────────────────────────────────────────────────────────┐
│  FOUNDATION (Phases 0-2)                                            │
│  ├── Phase 0: Research & API Analysis (All vendors)                │
│  ├── Phase 1: Unified Core Models                                   │
│  └── Phase 2: Unified Margin & Settlement Interfaces                │
├─────────────────────────────────────────────────────────────────────┤
│  CRYPTO FUTURES TRACK (Phases 3A-6A)                                │
│  ├── Phase 3A: Binance Futures Adapters                            │
│  ├── Phase 4A: Funding Rate & Liquidation (Crypto)                 │
│  ├── Phase 5A: L2/L3 Execution (Crypto)                            │
│  └── Phase 6A: Crypto Futures Features & Training                   │
├─────────────────────────────────────────────────────────────────────┤
│  CME FUTURES TRACK (Phases 3B-6B) - Can run in parallel             │
│  ├── Phase 3B: Interactive Brokers Adapters                        │
│  ├── Phase 4B: SPAN Margin & Daily Settlement                      │
│  ├── Phase 5B: L2/L3 Execution (CME)                               │
│  └── Phase 6B: CME Futures Features & Training                      │
├─────────────────────────────────────────────────────────────────────┤
│  INTEGRATION (Phases 7-10)                                          │
│  ├── Phase 7: Unified Risk Management                              │
│  ├── Phase 8: Multi-Futures Training Pipeline                      │
│  ├── Phase 9: Unified Live Trading                                 │
│  └── Phase 10: Validation & Documentation                          │
└─────────────────────────────────────────────────────────────────────┘
```

### Временная линия

| Фаза | Название | Длительность | Зависимости | Статус |
|------|----------|--------------|-------------|--------|
| 0 | Research | 1 week | - | ✅ DONE |
| 1 | Core Models | 1 week | Phase 0 | ✅ DONE |
| 2 | Margin/Settlement Interfaces | 1 week | Phase 1 | ✅ DONE |
| 3A | Binance Adapters | 2 weeks | Phase 2 | ✅ DONE |
| 3B | IB Adapters | 2 weeks | Phase 2 | ✅ DONE |
| 4A | Crypto Funding/Liquidation | 1.5 weeks | Phase 3A | ✅ DONE |
| 4B | CME SPAN/Settlement | 1.5 weeks | Phase 3B | ✅ DONE |
| 5A | Crypto L2/L3 | 1.5 weeks | Phase 4A | ✅ DONE |
| 5B | CME L2/L3 | 1.5 weeks | Phase 4B | ✅ DONE |
| 6A | Crypto Futures Risk | 1 week | Phase 5A | ✅ DONE |
| 6B | CME Futures Risk | 1 week | Phase 5B | ✅ DONE |
| 7 | Unified Risk Management | 1.5 weeks | Phase 6A, 6B | ✅ DONE |
| 8 | Training Pipeline | 2 weeks | Phase 7 | ✅ DONE |
| 9 | Live Trading | 1.5 weeks | Phase 8 | ✅ DONE |
| 10 | Validation | 2 weeks | Phase 9 | 📋 Pending |

**Общая длительность**: ~14-16 недель (с параллельными tracks)

### Completed Phase Summary

| Phase | Date Completed | Key Deliverables | Tests |
|-------|----------------|------------------|-------|
| 3B | 2025-11-30 | IB Adapters, CME Calendar | 205 |
| 4A | 2025-12-02 | FuturesSlippageProvider | 54 |
| 4B | 2025-12-02 | SPAN Margin, Circuit Breakers | 258 |
| 5A | 2025-12-02 | Crypto L3 (Liquidation, ADL, Funding) | 100 |
| 5B | 2025-12-02 | CME L3 (Globex, MWP, Stops, Settlement) | 42 |
| 6A | 2025-12-02 | Crypto Futures Risk Guards (Leverage, Margin, ADL, Funding, Concentration) | 101 |
| 6B | 2025-12-02 | CME Risk Guards (SPAN, Position Limits, Circuit Breaker, Settlement, Rollover) | 130 |
| **7** | **2025-12-02** | **Unified Risk Management (Multi-asset, Auto-delegation, Portfolio-level)** | **116** |
| **8** | **2025-12-02** | **Multi-Futures Training Pipeline (FuturesEnv, Feature Flags, Training Config)** | **131** |

---

## 📦 PHASE 0: RESEARCH & FOUNDATION

### Цели

- Детальное изучение API всех целевых бирж/брокеров
- Анализ существующего кода для переиспользования
- Определение унифицированной архитектуры
- Сбор референсных данных

### Задачи

#### 0.1 Binance Futures API Analysis (Crypto Track)

```
Endpoints to study:
├── Market Data
│   ├── GET /fapi/v1/klines - Candlesticks
│   ├── GET /fapi/v1/markPriceKlines - Mark price candles
│   ├── GET /fapi/v1/depth - Order book
│   ├── GET /fapi/v1/ticker/24hr - 24h stats
│   ├── GET /fapi/v1/fundingRate - Funding history
│   ├── GET /fapi/v1/premiumIndex - Mark price + funding
│   └── WebSocket streams (aggTrade, markPrice, forceOrder)
├── Account/Trade
│   ├── GET /fapi/v2/account - Account info (margin, positions)
│   ├── GET /fapi/v2/positionRisk - Position risk
│   ├── POST /fapi/v1/order - New order
│   ├── POST /fapi/v1/leverage - Set leverage
│   └── POST /fapi/v1/marginType - Set margin mode
└── Risk
    ├── GET /fapi/v1/adlQuantile - ADL indicator
    └── GET /fapi/v1/forceOrders - Liquidation orders
```

#### 0.2 Interactive Brokers TWS API Analysis (CME Track)

```
TWS API Components to study:
├── Market Data
│   ├── reqMktData() - Real-time quotes
│   ├── reqHistoricalData() - Historical bars
│   ├── reqRealTimeBars() - 5-second bars
│   ├── reqMktDepth() - Level 2 order book
│   └── reqContractDetails() - Contract specifications
├── Orders
│   ├── placeOrder() - Submit orders
│   ├── reqOpenOrders() - Open orders
│   ├── reqPositions() - Current positions
│   ├── reqAccountUpdates() - Account state (margin, PnL)
│   └── Order types: LMT, MKT, STP, TRAIL, BRACKET
├── Contract Definitions
│   ├── FUT - Futures contracts
│   ├── CONTFUT - Continuous futures (auto-roll)
│   ├── expiry, multiplier, exchange
│   └── Symbol mapping: ES, NQ, GC, CL, 6E, ZB
└── Risk & Margin
    ├── reqAccountSummary() - Margin requirements
    ├── Initial Margin, Maintenance Margin
    └── Excess Liquidity, Buying Power

Key Differences from Binance:
├── No funding rates (daily settlement instead)
├── SPAN margin instead of tiered brackets
├── Contract expiration (quarterly)
├── Trading hours: 23/5 with maintenance window
├── Different tick sizes per product
└── Physical vs cash settlement options
```

#### 0.3 CME Contract Specifications

```
Index Futures (via IB):
├── ES (E-mini S&P 500)
│   ├── Exchange: CME
│   ├── Multiplier: $50
│   ├── Tick: 0.25 = $12.50
│   ├── Expiry: Mar/Jun/Sep/Dec
│   ├── Settlement: Cash (3rd Friday)
│   └── Margin: ~5-6% initial
├── NQ (E-mini Nasdaq-100)
│   ├── Multiplier: $20
│   ├── Tick: 0.25 = $5.00
│   └── Similar to ES
├── MES/MNQ (Micro E-mini)
│   ├── Multiplier: $5/$2 (1/10th of E-mini)
│   └── Lower capital requirement

Commodity Futures:
├── GC (Gold)
│   ├── Exchange: COMEX
│   ├── Multiplier: 100 oz
│   ├── Tick: 0.10 = $10.00
│   ├── Expiry: Feb/Apr/Jun/Aug/Oct/Dec
│   └── Settlement: Physical (most roll before)
├── CL (Crude Oil)
│   ├── Exchange: NYMEX
│   ├── Multiplier: 1,000 barrels
│   ├── Tick: 0.01 = $10.00
│   └── Monthly expiration

Currency Futures:
├── 6E (Euro FX)
│   ├── Exchange: CME
│   ├── Multiplier: €125,000
│   ├── Tick: 0.00005 = $6.25
│   ├── Expiry: Mar/Jun/Sep/Dec
│   └── Settlement: Cash
├── 6J (Japanese Yen)
│   ├── Multiplier: ¥12,500,000
│   └── Inverse relationship to USD/JPY
```

#### 0.4 Key Concepts Documentation

**Crypto Futures (Binance) Concepts:**

- **Mark Price**: TWAP of index price + funding basis
- **Index Price**: Weighted average from multiple exchanges
- **Funding Rate**: `(Mark Price - Index Price) / Index Price` + premium (каждые 8ч)
- **Liquidation Price**: `Entry * (1 - IM% + MM%)`
- **ADL (Auto-Deleveraging)**: Forced position close when insurance fund depleted

**CME Futures Concepts:**

- **Settlement Price**: Daily settlement at 4:00pm ET (used for margin)
- **SPAN Margin**: Portfolio-based margin (offsets between correlated products)
- **Initial Margin**: Required to open position (~5-10% notional)
- **Maintenance Margin**: Min to keep position (~80% of initial)
- **Variation Margin**: Daily P&L settlement
- **Circuit Breakers**: 7%/13%/20% limit down for indices
- **Roll Date**: Standard roll ~8 days before expiry

#### 0.5 Existing Code Audit

```bash
# Files to review for reuse
adapters/binance/*.py          # Binance integration (crypto futures base)
adapters/alpaca/*.py           # Equity integration (session/hours reference)
ingest_funding_mark.py         # Funding rate ingestion (crypto)
services/forex_risk_guards.py  # Leverage/margin reference (all futures)
services/session_router.py     # Trading hours handling (CME futures)
lob/*.py                       # L3 LOB simulation (reuse for all)
lob/us_market_structure.py     # Tick sizes, circuit breakers (CME)
execution_providers.py         # L2 TCA (extend for futures)
```

### Deliverables Phase 0

- [ ] Binance Futures API documentation summary
- [ ] IB TWS API documentation summary
- [ ] CME contract specifications database
- [ ] Existing code compatibility report
- [ ] Unified architecture decision record (ADR)
- [ ] Test data collection plan (both crypto + CME)

### Tests

```bash
# No code changes in Phase 0 - documentation only
# Verify existing tests still pass
pytest tests/ -v --tb=short

# Verify IB connectivity (optional, requires IB account)
python -c "from ib_insync import IB; ib = IB(); print('IB available')"
```

---

## 📦 PHASE 1: UNIFIED CORE MODELS

### Цели

- Создать унифицированные futures модели для ВСЕХ типов (crypto, index, commodity, currency)
- Абстрагировать vendor-specific детали в адаптерах
- Обеспечить backward compatibility с существующим кодом

### Ключевой принцип

```
Унифицированные модели НЕ зависят от вендора.
Различия между Binance и CME инкапсулированы в адаптерах.
```

### 1.1 Unified Core Models (`core_futures.py`)

```python
# NEW FILE: core_futures.py
"""
Unified futures trading models for ALL futures types.

Supports:
- Crypto Perpetual (Binance USDT-M, Bybit)
- Crypto Quarterly (Binance delivery)
- Index Futures (CME: ES, NQ, YM, RTY)
- Commodity Futures (COMEX: GC, SI; NYMEX: CL, NG)
- Currency Futures (CME: 6E, 6J, 6B, 6A)
- Bond Futures (CBOT: ZB, ZN, ZF)

Design: Immutable dataclasses, Decimal precision, UTC milliseconds.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List

# ============================================================================
# ENUMS (Unified across all futures types)
# ============================================================================

class FuturesType(str, Enum):
    """Unified futures type classification."""
    # Crypto
    CRYPTO_PERPETUAL = "CRYPTO_PERPETUAL"    # No expiration, funding rate
    CRYPTO_QUARTERLY = "CRYPTO_QUARTERLY"    # Quarterly expiry
    # CME/CBOT/COMEX/NYMEX
    INDEX_FUTURES = "INDEX_FUTURES"          # ES, NQ, YM, RTY
    COMMODITY_FUTURES = "COMMODITY_FUTURES"  # GC, CL, SI, NG
    CURRENCY_FUTURES = "CURRENCY_FUTURES"    # 6E, 6J, 6B
    BOND_FUTURES = "BOND_FUTURES"            # ZB, ZN, ZF

class ContractType(str, Enum):
    """Contract expiration type."""
    PERPETUAL = "PERPETUAL"          # No expiration (crypto only)
    CURRENT_MONTH = "CURRENT_MONTH"  # Monthly contracts (CL, NG)
    CURRENT_QUARTER = "CURRENT_QUARTER"
    NEXT_QUARTER = "NEXT_QUARTER"
    BACK_MONTH = "BACK_MONTH"        # Further out months
    CONTINUOUS = "CONTINUOUS"        # Auto-rolling continuous contract

class SettlementType(str, Enum):
    """Settlement method."""
    CASH = "CASH"                    # Cash settlement (ES, NQ, 6E)
    PHYSICAL = "PHYSICAL"            # Physical delivery (GC, CL)
    FUNDING = "FUNDING"              # Funding rate (crypto perpetual)

class MarginMode(str, Enum):
    """Margin mode for position."""
    CROSS = "CROSS"        # Shared margin (Binance cross, IB portfolio)
    ISOLATED = "ISOLATED"  # Per-position margin (Binance isolated)
    SPAN = "SPAN"          # CME SPAN margin (portfolio-based)

class PositionSide(str, Enum):
    """Position side (for hedge mode)."""
    BOTH = "BOTH"    # One-way mode (net position)
    LONG = "LONG"    # Hedge mode long
    SHORT = "SHORT"  # Hedge mode short

class Exchange(str, Enum):
    """Exchange where contract trades."""
    BINANCE = "BINANCE"
    CME = "CME"           # E-mini indices
    COMEX = "COMEX"       # Gold, Silver
    NYMEX = "NYMEX"       # Oil, Natural Gas
    CBOT = "CBOT"         # Treasuries
    ICE = "ICE"           # Brent, Coffee

# ============================================================================
# CONTRACT SPECIFICATION (Works for ALL futures types)
# ============================================================================

@dataclass(frozen=True)
class FuturesContractSpec:
    """
    Unified futures contract specification.

    Works for Crypto (Binance) AND CME futures.
    """
    # Core identification
    symbol: str                           # BTCUSDT, ES, GC, 6E
    futures_type: FuturesType             # Classification
    contract_type: ContractType           # Perpetual, quarterly, monthly
    exchange: Exchange                    # Where it trades

    # Asset identifiers
    base_asset: str                       # BTC, SPX, Gold, EUR
    quote_asset: str                      # USDT, USD
    margin_asset: str                     # USDT, USD

    # Contract sizing
    contract_size: Decimal = Decimal("1")        # Units per contract
    multiplier: Decimal = Decimal("1")           # Price multiplier (ES=$50, GC=100oz)
    tick_size: Decimal = Decimal("0.01")         # Minimum price increment
    tick_value: Decimal = Decimal("0.01")        # Value per tick (ES=$12.50)

    # Position limits
    min_qty: Decimal = Decimal("0.001")
    max_qty: Decimal = Decimal("1000")
    lot_size: Decimal = Decimal("1")             # Order increment

    # Margin requirements
    max_leverage: int = 125                      # Crypto: 125x, ES: ~20x, GC: ~10x
    initial_margin_pct: Decimal = Decimal("5.0")   # % of notional
    maint_margin_pct: Decimal = Decimal("4.0")     # % of notional

    # Settlement
    settlement_type: SettlementType = SettlementType.CASH
    delivery_date: Optional[str] = None          # For expiring contracts (YYYYMMDD)
    last_trading_day: Optional[str] = None       # Last day to trade

    # Fees (vendor-specific, but defaults provided)
    liquidation_fee_pct: Decimal = Decimal("0.5")  # Crypto: 0.5%, CME: N/A

    # Trading hours
    trading_hours: str = "24/7"                  # "24/7", "23/5", "RTH only"

    @property
    def notional_per_contract(self) -> Decimal:
        """Calculate notional value per contract."""
        # For index futures: price * multiplier
        # For crypto: price * contract_size
        return self.multiplier * self.contract_size

    @property
    def is_perpetual(self) -> bool:
        """Check if perpetual (no expiry)."""
        return self.contract_type == ContractType.PERPETUAL

    @property
    def is_crypto(self) -> bool:
        """Check if crypto futures."""
        return self.futures_type in (
            FuturesType.CRYPTO_PERPETUAL,
            FuturesType.CRYPTO_QUARTERLY
        )

    @property
    def uses_funding(self) -> bool:
        """Check if uses funding rate instead of settlement."""
        return self.settlement_type == SettlementType.FUNDING

@dataclass(frozen=True)
class FuturesPosition:
    """Current futures position state."""
    symbol: str
    side: PositionSide
    entry_price: Decimal
    qty: Decimal  # Positive for long, negative for short
    leverage: int
    margin_mode: MarginMode
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    liquidation_price: Decimal = Decimal("0")
    mark_price: Decimal = Decimal("0")
    margin: Decimal = Decimal("0")  # Isolated margin amount
    maint_margin: Decimal = Decimal("0")
    timestamp_ms: int = 0

@dataclass(frozen=True)
class FundingPayment:
    """Funding rate payment record."""
    symbol: str
    timestamp_ms: int
    funding_rate: Decimal  # e.g., 0.0001 = 0.01%
    mark_price: Decimal
    position_qty: Decimal
    payment_amount: Decimal  # Positive = received, negative = paid
    asset: str = "USDT"

@dataclass(frozen=True)
class LiquidationEvent:
    """Liquidation event record."""
    symbol: str
    timestamp_ms: int
    side: str  # "BUY" or "SELL" (closing side)
    qty: Decimal
    price: Decimal
    liquidation_type: str  # "partial" or "full"
    loss_amount: Decimal
    insurance_fund_contribution: Decimal = Decimal("0")

@dataclass(frozen=True)
class FuturesAccountState:
    """Futures account state snapshot."""
    timestamp_ms: int
    total_wallet_balance: Decimal
    total_margin_balance: Decimal
    total_unrealized_pnl: Decimal
    available_balance: Decimal
    total_initial_margin: Decimal
    total_maint_margin: Decimal
    total_position_initial_margin: Decimal
    total_open_order_initial_margin: Decimal
    max_withdraw_amount: Decimal
    positions: Dict[str, FuturesPosition] = field(default_factory=dict)
```

### 1.2 Futures Market Data Adapter

```python
# NEW FILE: adapters/binance/futures_market_data.py
"""
Binance Futures market data adapter.

Extends BinanceMarketDataAdapter with futures-specific endpoints.
"""

from adapters.binance.market_data import BinanceMarketDataAdapter
from adapters.models import ExchangeVendor, MarketType

class BinanceFuturesMarketDataAdapter(BinanceMarketDataAdapter):
    """
    Binance USDT-M Futures market data adapter.

    Features:
    - Mark price candles (markPriceKlines)
    - Funding rate history
    - Open interest
    - Liquidation stream
    - Premium index
    """

    def __init__(self, config: Optional[Mapping] = None):
        super().__init__(
            vendor=ExchangeVendor.BINANCE,
            config={**(config or {}), "use_futures": True}
        )
        self._market_type = MarketType.CRYPTO_FUTURES

    def get_mark_price_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[Bar]:
        """Fetch mark price candlesticks."""
        # Uses /fapi/v1/markPriceKlines
        pass

    def get_funding_rate_history(
        self,
        symbol: str,
        limit: int = 100,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[FundingPayment]:
        """Fetch historical funding rates."""
        pass

    def get_current_funding_rate(self, symbol: str) -> FundingRateInfo:
        """Get current funding rate and next funding time."""
        pass

    def get_open_interest(self, symbol: str) -> OpenInterestInfo:
        """Get current open interest."""
        pass

    def stream_liquidations(
        self,
        symbols: Sequence[str],
    ) -> Iterator[LiquidationEvent]:
        """Stream liquidation orders (forceOrder stream)."""
        pass

    def stream_mark_price(
        self,
        symbols: Sequence[str],
        update_speed: str = "1s",  # "1s" or "3s"
    ) -> Iterator[MarkPriceTick]:
        """Stream real-time mark prices."""
        pass
```

### 1.3 Futures Exchange Info Adapter

```python
# NEW FILE: adapters/binance/futures_exchange_info.py
"""
Binance Futures exchange info adapter.
"""

class BinanceFuturesExchangeInfoAdapter(ExchangeInfoAdapter):
    """
    Futures-specific exchange info.

    Provides:
    - Contract specifications
    - Leverage brackets
    - Maintenance margin rates
    - Trading rules
    """

    def get_contract_spec(self, symbol: str) -> FuturesContractSpec:
        """Get contract specification."""
        pass

    def get_leverage_brackets(self, symbol: str) -> List[LeverageBracket]:
        """
        Get leverage brackets with maintenance margin rates.

        Example:
        [
            {"bracket": 1, "notionalCap": 10000, "maintMarginRatio": 0.0040, "maxLeverage": 125},
            {"bracket": 2, "notionalCap": 50000, "maintMarginRatio": 0.0050, "maxLeverage": 100},
            ...
        ]
        """
        pass

    def get_all_contracts(self) -> List[FuturesContractSpec]:
        """Get all tradeable futures contracts."""
        pass
```

### 1.4 Futures Order Execution Adapter

```python
# NEW FILE: adapters/binance/futures_order_execution.py
"""
Binance Futures order execution adapter.
"""

class BinanceFuturesOrderExecutionAdapter(OrderExecutionAdapter):
    """
    Futures order execution with leverage support.

    Features:
    - Set leverage per symbol
    - Set margin mode (cross/isolated)
    - Submit orders with reduceOnly flag
    - Hedge mode support
    """

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for symbol."""
        pass

    def set_margin_mode(self, symbol: str, mode: MarginMode) -> bool:
        """Set margin mode for symbol."""
        pass

    def submit_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        qty: Decimal,
        price: Optional[Decimal] = None,
        reduce_only: bool = False,
        position_side: PositionSide = PositionSide.BOTH,
        time_in_force: str = "GTC",
    ) -> FuturesOrder:
        """Submit futures order."""
        pass

    def get_position(self, symbol: str) -> Optional[FuturesPosition]:
        """Get current position for symbol."""
        pass

    def get_all_positions(self) -> List[FuturesPosition]:
        """Get all open positions."""
        pass

    def get_account_balance(self) -> FuturesAccountState:
        """Get account state with margin info."""
        pass
```

### 1.5 Registry Updates

```python
# UPDATE: adapters/registry.py
# Add futures adapter registration

from adapters.binance.futures_market_data import BinanceFuturesMarketDataAdapter
from adapters.binance.futures_exchange_info import BinanceFuturesExchangeInfoAdapter
from adapters.binance.futures_order_execution import BinanceFuturesOrderExecutionAdapter

# Register futures adapters
_registry.register(
    vendor=ExchangeVendor.BINANCE,
    adapter_type=AdapterType.MARKET_DATA,
    adapter_class=BinanceFuturesMarketDataAdapter,
    market_type=MarketType.CRYPTO_FUTURES,
    description="Binance USDT-M Futures market data",
)

# Factory function update
def create_market_data_adapter(
    vendor: str,
    config: Optional[Dict] = None,
    market_type: Optional[MarketType] = None,
) -> MarketDataAdapter:
    """Create market data adapter with futures support."""
    v = ExchangeVendor(vendor.lower())

    # Determine market type
    if market_type is None:
        market_type = config.get("market_type") if config else None
        if market_type is None:
            market_type = v.market_type  # Default for vendor

    # Select appropriate adapter
    if v == ExchangeVendor.BINANCE:
        if market_type in (MarketType.CRYPTO_FUTURES, MarketType.CRYPTO_PERP):
            return BinanceFuturesMarketDataAdapter(config)
        return BinanceMarketDataAdapter(config)
    # ... other vendors
```

### Tests for Phase 1

```python
# NEW FILE: tests/test_futures_core_models.py
"""Tests for futures core models."""

import pytest
from decimal import Decimal
from core_futures import (
    ContractType, MarginMode, PositionSide,
    FuturesContractSpec, FuturesPosition, FundingPayment,
)

class TestFuturesContractSpec:
    """Contract specification tests."""

    def test_perpetual_contract_creation(self):
        spec = FuturesContractSpec(
            symbol="BTCUSDT",
            contract_type=ContractType.PERPETUAL,
            base_asset="BTC",
            quote_asset="USDT",
            margin_asset="USDT",
            max_leverage=125,
        )
        assert spec.contract_type == ContractType.PERPETUAL
        assert spec.max_leverage == 125

    def test_quarterly_contract_with_delivery(self):
        spec = FuturesContractSpec(
            symbol="BTCUSDT_240329",
            contract_type=ContractType.CURRENT_QUARTER,
            base_asset="BTC",
            quote_asset="USDT",
            margin_asset="USDT",
            delivery_date="2024-03-29",
        )
        assert spec.delivery_date is not None

class TestFuturesPosition:
    """Position state tests."""

    def test_long_position(self):
        pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("0.1"),
            leverage=20,
            margin_mode=MarginMode.CROSS,
        )
        assert pos.qty > 0

    def test_short_position_negative_qty(self):
        pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            entry_price=Decimal("50000"),
            qty=Decimal("-0.1"),
            leverage=20,
            margin_mode=MarginMode.ISOLATED,
        )
        assert pos.qty < 0

# NEW FILE: tests/test_futures_adapters.py
"""Tests for futures adapters."""

class TestBinanceFuturesMarketDataAdapter:
    """Futures market data adapter tests."""

    @pytest.fixture
    def adapter(self):
        return BinanceFuturesMarketDataAdapter({"use_futures": True})

    def test_adapter_uses_futures_endpoints(self, adapter):
        assert adapter._use_futures is True

    def test_get_mark_price_bars(self, adapter, mocker):
        # Mock API response
        pass

    def test_get_funding_rate_history(self, adapter, mocker):
        pass
```

### Regression Tests

```bash
# Ensure no regressions in existing functionality
pytest tests/test_binance*.py -v
pytest tests/test_execution_providers*.py -v
pytest tests/test_alpaca*.py -v
```

### Deliverables Phase 1

- [ ] `core_futures.py` - Core futures models
- [ ] `adapters/binance/futures_market_data.py`
- [ ] `adapters/binance/futures_exchange_info.py`
- [ ] `adapters/binance/futures_order_execution.py`
- [ ] Updated `adapters/registry.py`
- [ ] `tests/test_futures_core_models.py` (50+ tests)
- [ ] `tests/test_futures_adapters.py` (40+ tests)

---

## 📦 PHASE 2: MARGIN & LIQUIDATION SYSTEM

### Цели

- Реализовать margin calculation engine
- Создать liquidation simulation
- Поддержка cross и isolated margin

### 2.0 Abstract Base for Futures Execution Provider

```python
# NEW FILE: execution_providers_futures_base.py
"""
Abstract base classes for unified futures execution providers.

Design Principles:
1. Single interface for ALL futures types (crypto, index, commodity, currency)
2. Vendor-agnostic execution logic
3. Clear separation between L2 (parametric) and L3 (LOB) simulation
4. Protocol-based dependency injection for testability

References:
- Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
- CME Group (2023): Futures Execution Best Practices
- Binance API (2024): USDT-M Futures Execution Documentation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Protocol, Optional, Dict, Any, List
from datetime import datetime

from core_futures import (
    FuturesContract,
    FuturesPosition,
    MarginMode,
    MarginRequirement,
)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS & DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

class FuturesType(str, Enum):
    """Futures contract type classification."""
    CRYPTO_PERPETUAL = "crypto_perp"       # Binance USDT-M Perpetual
    CRYPTO_QUARTERLY = "crypto_quarterly"  # Binance USDT-M Quarterly
    INDEX_FUTURES = "index"                # ES, NQ, YM, RTY
    COMMODITY_FUTURES = "commodity"        # GC, CL, SI, NG
    CURRENCY_FUTURES = "currency"          # 6E, 6J, 6B, 6A
    BOND_FUTURES = "bond"                  # ZB, ZN, ZF


@dataclass(frozen=True)
class FuturesMarketState:
    """
    Unified market state for futures execution.

    Vendor-agnostic representation of current market conditions.
    """
    timestamp_ms: int
    bid: Decimal
    ask: Decimal
    bid_size: Decimal
    ask_size: Decimal
    mark_price: Decimal           # For liquidation calc
    index_price: Decimal          # Underlying reference
    last_price: Decimal
    volume_24h: Decimal
    open_interest: Decimal
    funding_rate: Optional[Decimal] = None        # Crypto only
    next_funding_time_ms: Optional[int] = None    # Crypto only
    settlement_price: Optional[Decimal] = None    # CME daily settlement
    days_to_expiry: Optional[int] = None          # Quarterly contracts

    @property
    def mid_price(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def spread_bps(self) -> Decimal:
        if self.mid_price == 0:
            return Decimal("0")
        return (self.ask - self.bid) / self.mid_price * 10000


@dataclass(frozen=True)
class FuturesOrder:
    """Unified order representation for all futures types."""
    symbol: str
    side: str              # "BUY" or "SELL"
    order_type: str        # "MARKET", "LIMIT", "STOP", "STOP_MARKET"
    qty: Decimal
    price: Optional[Decimal] = None
    reduce_only: bool = False
    time_in_force: str = "GTC"
    post_only: bool = False
    client_order_id: Optional[str] = None


@dataclass
class FuturesFill:
    """Result of a simulated futures execution."""
    order_id: str
    symbol: str
    side: str
    filled_qty: Decimal
    avg_price: Decimal
    commission: Decimal
    commission_asset: str
    realized_pnl: Decimal
    slippage_bps: Decimal
    timestamp_ms: int
    is_maker: bool
    liquidity: str               # "MAKER" or "TAKER"

    # Futures-specific fields
    margin_impact: Decimal       # Change in margin requirement
    new_position_size: Decimal   # Position after fill
    new_avg_entry: Decimal       # New average entry price


# ═══════════════════════════════════════════════════════════════════════════
# PROTOCOL INTERFACES (Dependency Injection)
# ═══════════════════════════════════════════════════════════════════════════

class FuturesMarginProvider(Protocol):
    """Protocol for margin calculation providers."""

    def calculate_initial_margin(
        self,
        contract: FuturesContract,
        notional: Decimal,
        leverage: int,
    ) -> Decimal:
        """Calculate initial margin requirement."""
        ...

    def calculate_maintenance_margin(
        self,
        contract: FuturesContract,
        notional: Decimal,
    ) -> Decimal:
        """Calculate maintenance margin requirement."""
        ...

    def calculate_liquidation_price(
        self,
        position: FuturesPosition,
        wallet_balance: Decimal,
    ) -> Decimal:
        """Calculate liquidation price for position."""
        ...


class FuturesSlippageProvider(Protocol):
    """Protocol for slippage estimation providers."""

    def estimate_slippage_bps(
        self,
        order: FuturesOrder,
        market: FuturesMarketState,
        participation_rate: Optional[Decimal] = None,
    ) -> Decimal:
        """Estimate execution slippage in basis points."""
        ...


class FuturesFeeProvider(Protocol):
    """Protocol for fee calculation providers."""

    def calculate_fee(
        self,
        notional: Decimal,
        is_maker: bool,
        fee_tier: Optional[str] = None,
    ) -> Decimal:
        """Calculate trading fee."""
        ...


class FuturesFundingProvider(Protocol):
    """Protocol for funding rate providers (crypto only)."""

    def get_current_funding_rate(self, symbol: str) -> Decimal:
        """Get current funding rate."""
        ...

    def calculate_funding_payment(
        self,
        position: FuturesPosition,
        funding_rate: Decimal,
        mark_price: Decimal,
    ) -> Decimal:
        """Calculate funding payment amount."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# ABSTRACT BASE CLASSES
# ═══════════════════════════════════════════════════════════════════════════

class BaseFuturesExecutionProvider(ABC):
    """
    Abstract base class for all futures execution providers.

    Provides unified interface for:
    - Crypto perpetuals (Binance)
    - Crypto quarterly (Binance)
    - Index futures (CME via IB)
    - Commodity futures (CME via IB)
    - Currency futures (CME via IB)

    Subclasses implement vendor-specific execution logic while
    maintaining consistent interface for backtesting and live trading.
    """

    def __init__(
        self,
        futures_type: FuturesType,
        margin_provider: FuturesMarginProvider,
        slippage_provider: FuturesSlippageProvider,
        fee_provider: FuturesFeeProvider,
        funding_provider: Optional[FuturesFundingProvider] = None,
    ):
        self._futures_type = futures_type
        self._margin_provider = margin_provider
        self._slippage_provider = slippage_provider
        self._fee_provider = fee_provider
        self._funding_provider = funding_provider

        # Validate funding provider for crypto
        if futures_type in (FuturesType.CRYPTO_PERPETUAL,) and not funding_provider:
            raise ValueError("FundingProvider required for crypto perpetuals")

    @property
    def futures_type(self) -> FuturesType:
        return self._futures_type

    @abstractmethod
    def execute(
        self,
        order: FuturesOrder,
        market: FuturesMarketState,
        position: Optional[FuturesPosition] = None,
    ) -> FuturesFill:
        """
        Execute order in simulation.

        Args:
            order: Order to execute
            market: Current market state
            position: Current position (if any)

        Returns:
            FuturesFill with execution details
        """
        pass

    @abstractmethod
    def estimate_execution_cost(
        self,
        order: FuturesOrder,
        market: FuturesMarketState,
    ) -> Dict[str, Decimal]:
        """
        Pre-trade cost estimation.

        Returns:
            Dict with keys: 'slippage_bps', 'fee_bps', 'total_cost_bps',
            'impact_cost', 'estimated_fill_price'
        """
        pass

    def get_margin_requirement(
        self,
        contract: FuturesContract,
        qty: Decimal,
        price: Decimal,
        leverage: int,
    ) -> MarginRequirement:
        """Get margin requirement for position."""
        notional = qty * price * contract.multiplier
        initial = self._margin_provider.calculate_initial_margin(
            contract, notional, leverage
        )
        maintenance = self._margin_provider.calculate_maintenance_margin(
            contract, notional
        )
        return MarginRequirement(
            initial=initial,
            maintenance=maintenance,
            variation=Decimal("0"),  # Calculated on daily settlement
        )

    def calculate_pnl(
        self,
        position: FuturesPosition,
        current_price: Decimal,
    ) -> Decimal:
        """Calculate unrealized P&L for position."""
        if position.qty == 0:
            return Decimal("0")

        notional_entry = position.qty * position.entry_price * position.contract.multiplier
        notional_current = position.qty * current_price * position.contract.multiplier

        if position.side == "LONG":
            return notional_current - notional_entry
        else:  # SHORT
            return notional_entry - notional_current


class L2FuturesExecutionProvider(BaseFuturesExecutionProvider):
    """
    L2 parametric execution provider for futures.

    Uses statistical models for slippage estimation:
    - √participation impact model (Almgren-Chriss)
    - Volatility regime adjustments
    - Funding rate stress (crypto)
    - Time-of-day liquidity curves

    Suitable for:
    - Backtesting with realistic cost modeling
    - Strategy development and optimization
    - Production with moderate accuracy requirements (~80-90%)
    """

    def execute(
        self,
        order: FuturesOrder,
        market: FuturesMarketState,
        position: Optional[FuturesPosition] = None,
    ) -> FuturesFill:
        """Execute with L2 parametric slippage model."""
        # Estimate slippage
        slippage_bps = self._slippage_provider.estimate_slippage_bps(order, market)

        # Calculate execution price
        mid = market.mid_price
        slippage_factor = slippage_bps / Decimal("10000")

        if order.side == "BUY":
            exec_price = mid * (1 + slippage_factor)
        else:
            exec_price = mid * (1 - slippage_factor)

        # Calculate fee
        notional = order.qty * exec_price
        is_maker = order.order_type == "LIMIT" and order.post_only
        fee = self._fee_provider.calculate_fee(notional, is_maker)

        # Calculate margin impact
        margin_req = self.get_margin_requirement(
            position.contract if position else FuturesContract.default(order.symbol),
            order.qty,
            exec_price,
            20,  # Default leverage
        )

        # Calculate realized PnL if reducing position
        realized_pnl = Decimal("0")
        new_position_size = order.qty
        new_avg_entry = exec_price

        if position and position.qty != 0:
            if (position.side == "LONG" and order.side == "SELL") or \
               (position.side == "SHORT" and order.side == "BUY"):
                # Reducing position
                close_qty = min(order.qty, abs(position.qty))
                realized_pnl = self.calculate_pnl(
                    FuturesPosition(
                        contract=position.contract,
                        qty=close_qty,
                        entry_price=position.entry_price,
                        side=position.side,
                        leverage=position.leverage,
                        margin_mode=position.margin_mode,
                    ),
                    exec_price,
                )

        return FuturesFill(
            order_id=order.client_order_id or f"SIM_{market.timestamp_ms}",
            symbol=order.symbol,
            side=order.side,
            filled_qty=order.qty,
            avg_price=exec_price,
            commission=fee,
            commission_asset="USDT",  # Configurable
            realized_pnl=realized_pnl,
            slippage_bps=slippage_bps,
            timestamp_ms=market.timestamp_ms,
            is_maker=is_maker,
            liquidity="MAKER" if is_maker else "TAKER",
            margin_impact=margin_req.initial,
            new_position_size=new_position_size,
            new_avg_entry=new_avg_entry,
        )

    def estimate_execution_cost(
        self,
        order: FuturesOrder,
        market: FuturesMarketState,
    ) -> Dict[str, Decimal]:
        """Pre-trade cost estimation."""
        slippage_bps = self._slippage_provider.estimate_slippage_bps(order, market)

        mid = market.mid_price
        slippage_factor = slippage_bps / Decimal("10000")
        estimated_price = mid * (1 + slippage_factor) if order.side == "BUY" else mid * (1 - slippage_factor)

        notional = order.qty * estimated_price
        fee = self._fee_provider.calculate_fee(notional, is_maker=False)
        fee_bps = fee / notional * Decimal("10000")

        return {
            'slippage_bps': slippage_bps,
            'fee_bps': fee_bps,
            'total_cost_bps': slippage_bps + fee_bps,
            'impact_cost': notional * slippage_bps / Decimal("10000"),
            'estimated_fill_price': estimated_price,
        }


class L3FuturesExecutionProvider(BaseFuturesExecutionProvider):
    """
    L3 order book execution provider for futures.

    Uses full LOB simulation:
    - Queue position tracking
    - Market impact modeling (Kyle, Almgren-Chriss, Gatheral)
    - Latency simulation
    - Fill probability models

    Extends existing lob/ module for futures-specific features:
    - Liquidation order injection
    - Funding rate impact
    - Daily settlement simulation (CME)

    Suitable for:
    - High-fidelity backtesting (95%+ accuracy)
    - Market microstructure research
    - HFT strategy development
    """

    def __init__(
        self,
        futures_type: FuturesType,
        margin_provider: FuturesMarginProvider,
        slippage_provider: FuturesSlippageProvider,
        fee_provider: FuturesFeeProvider,
        funding_provider: Optional[FuturesFundingProvider] = None,
        lob_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            futures_type,
            margin_provider,
            slippage_provider,
            fee_provider,
            funding_provider,
        )
        # Initialize LOB components
        self._lob_config = lob_config or {}
        # LOB initialization deferred to avoid circular imports
        self._matching_engine = None
        self._impact_model = None
        self._latency_model = None

    def execute(
        self,
        order: FuturesOrder,
        market: FuturesMarketState,
        position: Optional[FuturesPosition] = None,
    ) -> FuturesFill:
        """Execute with full LOB simulation."""
        # Implementation delegates to lob/matching_engine.py
        # with futures-specific extensions
        raise NotImplementedError("L3 execution requires LOB initialization")

    def estimate_execution_cost(
        self,
        order: FuturesOrder,
        market: FuturesMarketState,
    ) -> Dict[str, Decimal]:
        """Pre-trade cost estimation using LOB state."""
        raise NotImplementedError("L3 estimation requires LOB initialization")


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def create_futures_execution_provider(
    futures_type: FuturesType,
    level: str = "L2",  # "L2" or "L3"
    config: Optional[Dict[str, Any]] = None,
) -> BaseFuturesExecutionProvider:
    """
    Factory function for creating futures execution providers.

    Args:
        futures_type: Type of futures contract
        level: Simulation fidelity ("L2" = parametric, "L3" = LOB)
        config: Provider configuration

    Returns:
        Configured execution provider

    Example:
        >>> provider = create_futures_execution_provider(
        ...     FuturesType.CRYPTO_PERPETUAL,
        ...     level="L2",
        ...     config={"slippage_profile": "binance_futures"},
        ... )
    """
    config = config or {}

    # Create default providers based on futures type
    margin_provider = _create_margin_provider(futures_type, config)
    slippage_provider = _create_slippage_provider(futures_type, config)
    fee_provider = _create_fee_provider(futures_type, config)
    funding_provider = None

    if futures_type in (FuturesType.CRYPTO_PERPETUAL,):
        funding_provider = _create_funding_provider(futures_type, config)

    if level == "L2":
        return L2FuturesExecutionProvider(
            futures_type=futures_type,
            margin_provider=margin_provider,
            slippage_provider=slippage_provider,
            fee_provider=fee_provider,
            funding_provider=funding_provider,
        )
    elif level == "L3":
        return L3FuturesExecutionProvider(
            futures_type=futures_type,
            margin_provider=margin_provider,
            slippage_provider=slippage_provider,
            fee_provider=fee_provider,
            funding_provider=funding_provider,
            lob_config=config.get("lob_config"),
        )
    else:
        raise ValueError(f"Unknown execution level: {level}")


def _create_margin_provider(futures_type: FuturesType, config: Dict) -> FuturesMarginProvider:
    """Create appropriate margin provider for futures type."""
    # Implementation depends on futures type
    # - Crypto: TieredMarginProvider (Binance brackets)
    # - CME: SPANMarginProvider (simplified SPAN)
    raise NotImplementedError("Use concrete implementations")


def _create_slippage_provider(futures_type: FuturesType, config: Dict) -> FuturesSlippageProvider:
    """Create appropriate slippage provider for futures type."""
    raise NotImplementedError("Use concrete implementations")


def _create_fee_provider(futures_type: FuturesType, config: Dict) -> FuturesFeeProvider:
    """Create appropriate fee provider for futures type."""
    raise NotImplementedError("Use concrete implementations")


def _create_funding_provider(futures_type: FuturesType, config: Dict) -> FuturesFundingProvider:
    """Create funding provider for crypto perpetuals."""
    raise NotImplementedError("Use concrete implementations")
```

**Key Design Decisions:**

1. **Protocol-based DI**: Uses Python `Protocol` for dependency injection, enabling easy mocking in tests
2. **Unified Data Models**: `FuturesMarketState`, `FuturesOrder`, `FuturesFill` work for ALL futures types
3. **Clear L2/L3 Separation**: L2 uses parametric models, L3 uses full LOB simulation
4. **Factory Pattern**: `create_futures_execution_provider()` abstracts instantiation complexity
5. **Type Safety**: Full typing with `Decimal` for financial calculations

### 2.1 Margin Calculator (`impl_futures_margin.py`)

```python
# NEW FILE: impl_futures_margin.py
"""
Futures margin calculation engine.

Implements Binance USDT-M margin rules:
- Initial margin = Notional / Leverage
- Maintenance margin = Notional * MMR (tiered)
- Liquidation price calculation

References:
- https://www.binance.com/en/support/faq/leverage-and-margin-of-usd%E2%93%A2-m-futures-360033162192
"""

from decimal import Decimal
from typing import List, Optional, Tuple
from core_futures import FuturesPosition, MarginMode, LeverageBracket

class MarginCalculator:
    """
    Futures margin calculator with tiered maintenance margin.

    Key formulas:
    - Initial Margin (IM) = Notional Value / Leverage
    - Maintenance Margin (MM) = Notional Value * MMR
    - Position Margin = max(IM, MM)

    Liquidation Price (Long):
        LP = Entry Price * (1 - Initial Margin Rate + Maintenance Margin Rate)

    Liquidation Price (Short):
        LP = Entry Price * (1 + Initial Margin Rate - Maintenance Margin Rate)
    """

    def __init__(self, leverage_brackets: List[LeverageBracket]):
        """
        Initialize with leverage brackets.

        Args:
            leverage_brackets: Tiered margin requirements from exchange
        """
        self._brackets = sorted(leverage_brackets, key=lambda x: x.notional_cap)

    def get_maintenance_margin_rate(self, notional: Decimal) -> Decimal:
        """
        Get maintenance margin rate for notional value.

        Tiered system - higher notional = higher MMR.
        """
        for bracket in self._brackets:
            if notional <= bracket.notional_cap:
                return bracket.maint_margin_rate
        return self._brackets[-1].maint_margin_rate

    def calculate_initial_margin(
        self,
        notional: Decimal,
        leverage: int,
    ) -> Decimal:
        """Calculate initial margin requirement."""
        return notional / Decimal(leverage)

    def calculate_maintenance_margin(
        self,
        notional: Decimal,
    ) -> Decimal:
        """Calculate maintenance margin requirement."""
        mmr = self.get_maintenance_margin_rate(notional)
        return notional * mmr

    def calculate_liquidation_price(
        self,
        entry_price: Decimal,
        qty: Decimal,
        leverage: int,
        wallet_balance: Decimal,
        margin_mode: MarginMode,
        isolated_margin: Decimal = Decimal("0"),
    ) -> Decimal:
        """
        Calculate liquidation price.

        For CROSS margin:
            Entire wallet balance is used

        For ISOLATED margin:
            Only isolated margin for position is used
        """
        is_long = qty > 0
        abs_qty = abs(qty)
        notional = entry_price * abs_qty

        mmr = self.get_maintenance_margin_rate(notional)

        if margin_mode == MarginMode.CROSS:
            # Cross margin uses wallet balance
            available_margin = wallet_balance
        else:
            # Isolated uses only position margin
            available_margin = isolated_margin

        # Margin ratio at liquidation
        # MR = (Wallet Balance + Unrealized PnL) / Maintenance Margin
        # At liquidation, MR = 1

        if is_long:
            # Long: liquidation when price drops
            # UPnL = (Mark Price - Entry) * qty
            # At liquidation: available_margin + UPnL = notional * mmr
            # Mark Price = Entry - (available_margin - notional * mmr) / qty
            liq_price = entry_price - (available_margin - notional * mmr) / abs_qty
        else:
            # Short: liquidation when price rises
            # UPnL = (Entry - Mark Price) * |qty|
            liq_price = entry_price + (available_margin - notional * mmr) / abs_qty

        return max(Decimal("0"), liq_price)

    def calculate_margin_ratio(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
        wallet_balance: Decimal,
    ) -> Decimal:
        """
        Calculate current margin ratio.

        MR = (Wallet Balance + Unrealized PnL) / Maintenance Margin
        MR < 1 = Liquidation
        MR < 1.5 = Margin warning
        """
        abs_qty = abs(position.qty)
        notional = mark_price * abs_qty

        # Unrealized PnL
        if position.qty > 0:
            upnl = (mark_price - position.entry_price) * abs_qty
        else:
            upnl = (position.entry_price - mark_price) * abs_qty

        mm = self.calculate_maintenance_margin(notional)

        if mm == 0:
            return Decimal("inf")

        margin_ratio = (wallet_balance + upnl) / mm
        return margin_ratio
```

### 2.2 Liquidation Engine (`impl_futures_liquidation.py`)

```python
# NEW FILE: impl_futures_liquidation.py
"""
Futures liquidation simulation engine.

Simulates liquidation cascade with insurance fund mechanics.

References:
- Binance liquidation: https://www.binance.com/en/support/faq/how-is-the-liquidation-price-calculated-360033525271
"""

from decimal import Decimal
from typing import List, Optional, Tuple
from enum import Enum
from core_futures import FuturesPosition, LiquidationEvent, MarginMode


class LiquidationPriority(str, Enum):
    """
    Liquidation priority for cross-margin accounts.

    When an account's total margin ratio falls below 1.0, positions must be
    liquidated to restore margin. This enum defines the order in which
    positions are selected for liquidation.

    Different exchanges use different strategies:
    - Binance: HIGHEST_LOSS_FIRST (liquidate most unprofitable first)
    - Some exchanges: LOWEST_MARGIN_RATIO (most risky positions first)
    - Others: LARGEST_POSITION (reduce biggest exposure first)

    References:
    - Binance Cross-Margin: https://www.binance.com/en/support/faq/360038685551
    """
    HIGHEST_LOSS_FIRST = "highest_loss"      # Binance default - close losers first
    LOWEST_MARGIN_RATIO = "lowest_ratio"     # Most risky positions first
    OLDEST_POSITION = "oldest"               # FIFO - oldest positions first
    LARGEST_POSITION = "largest"             # Biggest notional first
    HIGHEST_LEVERAGE = "highest_leverage"    # Most leveraged first


class CrossMarginLiquidationOrdering:
    """
    Determines order of position liquidation for cross-margin accounts.

    In cross-margin mode, all positions share the same margin pool. When
    total account margin ratio drops below maintenance, the system must
    decide WHICH positions to liquidate first.

    This affects:
    1. Which positions get closed (user preference may differ from system)
    2. Cascade effects (closing one position may save others)
    3. Overall account survival probability
    """

    def __init__(self, priority: LiquidationPriority = LiquidationPriority.HIGHEST_LOSS_FIRST):
        self._priority = priority

    def order_positions_for_liquidation(
        self,
        positions: List[FuturesPosition],
        mark_prices: Dict[str, Decimal],
    ) -> List[FuturesPosition]:
        """
        Order positions by liquidation priority.

        Args:
            positions: All open positions in cross-margin account
            mark_prices: Current mark prices per symbol

        Returns:
            Positions ordered from first-to-liquidate to last
        """
        def get_pnl(pos: FuturesPosition) -> Decimal:
            mark = mark_prices.get(pos.symbol, pos.entry_price)
            if pos.qty > 0:  # Long
                return (mark - pos.entry_price) * abs(pos.qty)
            else:  # Short
                return (pos.entry_price - mark) * abs(pos.qty)

        def get_notional(pos: FuturesPosition) -> Decimal:
            mark = mark_prices.get(pos.symbol, pos.entry_price)
            return mark * abs(pos.qty)

        if self._priority == LiquidationPriority.HIGHEST_LOSS_FIRST:
            # Sort by PnL ascending (most negative first)
            return sorted(positions, key=get_pnl)

        elif self._priority == LiquidationPriority.LOWEST_MARGIN_RATIO:
            # Would need margin calc per position - simplified to use leverage
            return sorted(positions, key=lambda p: -p.leverage)

        elif self._priority == LiquidationPriority.OLDEST_POSITION:
            # Sort by entry time (requires timestamp on position)
            return sorted(positions, key=lambda p: getattr(p, 'entry_time_ms', 0))

        elif self._priority == LiquidationPriority.LARGEST_POSITION:
            # Sort by notional descending
            return sorted(positions, key=get_notional, reverse=True)

        elif self._priority == LiquidationPriority.HIGHEST_LEVERAGE:
            # Sort by leverage descending
            return sorted(positions, key=lambda p: -p.leverage)

        return positions  # Default: no ordering

    def select_positions_to_liquidate(
        self,
        positions: List[FuturesPosition],
        mark_prices: Dict[str, Decimal],
        target_margin_ratio: Decimal,
        current_balance: Decimal,
        margin_calculator: 'MarginCalculator',
    ) -> List[FuturesPosition]:
        """
        Select minimum set of positions to liquidate to restore margin.

        Greedy algorithm: liquidate in priority order until margin ratio >= target.

        Args:
            positions: All open positions
            mark_prices: Current mark prices
            target_margin_ratio: Target margin ratio to achieve (e.g., 1.5)
            current_balance: Current wallet balance
            margin_calculator: Margin calculator instance

        Returns:
            List of positions to liquidate (subset of input)
        """
        ordered = self.order_positions_for_liquidation(positions, mark_prices)
        to_liquidate = []

        remaining = list(ordered)
        for pos in ordered:
            # Check if we've achieved target
            total_margin = sum(
                margin_calculator.calculate_maintenance_margin(
                    mark_prices.get(p.symbol, p.entry_price) * abs(p.qty)
                )
                for p in remaining
            )
            total_upnl = sum(
                (mark_prices.get(p.symbol, p.entry_price) - p.entry_price) * p.qty
                for p in remaining
            )
            current_ratio = (current_balance + total_upnl) / total_margin if total_margin > 0 else Decimal("inf")

            if current_ratio >= target_margin_ratio:
                break  # Target achieved

            # Mark for liquidation and remove from remaining
            to_liquidate.append(pos)
            remaining.remove(pos)

        return to_liquidate


class LiquidationEngine:
    """
    Simulates futures liquidation mechanics.

    Liquidation process:
    1. Position margin ratio drops below 1 (MM)
    2. Liquidation order placed at bankruptcy price
    3. If filled above bankruptcy → profit to insurance fund
    4. If filled below bankruptcy → ADL or insurance fund covers
    """

    def __init__(
        self,
        insurance_fund_balance: Decimal = Decimal("1000000"),
        liquidation_fee_rate: Decimal = Decimal("0.005"),  # 0.5%
    ):
        self._insurance_fund = insurance_fund_balance
        self._liquidation_fee_rate = liquidation_fee_rate

    def check_liquidation(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
        wallet_balance: Decimal,
        margin_calculator: MarginCalculator,
    ) -> Optional[LiquidationEvent]:
        """
        Check if position should be liquidated.

        Returns:
            LiquidationEvent if liquidation triggered, None otherwise
        """
        margin_ratio = margin_calculator.calculate_margin_ratio(
            position, mark_price, wallet_balance
        )

        if margin_ratio >= Decimal("1"):
            return None  # No liquidation

        # Calculate liquidation parameters
        abs_qty = abs(position.qty)
        notional = mark_price * abs_qty

        # Liquidation fee
        liq_fee = notional * self._liquidation_fee_rate

        # Loss amount
        if position.qty > 0:
            pnl = (mark_price - position.entry_price) * abs_qty
        else:
            pnl = (position.entry_price - mark_price) * abs_qty

        loss = abs(min(Decimal("0"), pnl)) + liq_fee

        # Insurance fund contribution
        insurance_contribution = Decimal("0")
        if pnl < -wallet_balance:
            # Position went bankrupt - insurance fund covers
            insurance_contribution = min(
                self._insurance_fund,
                abs(pnl) - wallet_balance
            )
            self._insurance_fund -= insurance_contribution

        return LiquidationEvent(
            symbol=position.symbol,
            timestamp_ms=0,  # Set by caller
            side="SELL" if position.qty > 0 else "BUY",
            qty=abs_qty,
            price=mark_price,
            liquidation_type="full",
            loss_amount=loss,
            insurance_fund_contribution=insurance_contribution,
        )

    def simulate_partial_liquidation(
        self,
        position: FuturesPosition,
        target_margin_ratio: Decimal = Decimal("2"),
    ) -> Tuple[Decimal, FuturesPosition]:
        """
        Calculate partial liquidation to restore margin ratio.

        Returns:
            (qty_to_liquidate, remaining_position)
        """
        # Partial liquidation reduces position to restore margin
        pass

    def get_adl_ranking(
        self,
        position: FuturesPosition,
        pnl_percentile: Decimal,
        leverage_percentile: Decimal,
    ) -> int:
        """
        Calculate ADL (Auto-Deleveraging) ranking 1-5.

        Higher ranking = higher priority for ADL.
        Based on profit and leverage.
        """
        # ADL ranking formula from Binance
        score = pnl_percentile * leverage_percentile

        if score >= Decimal("0.8"):
            return 5
        elif score >= Decimal("0.6"):
            return 4
        elif score >= Decimal("0.4"):
            return 3
        elif score >= Decimal("0.2"):
            return 2
        return 1


@dataclass
class ADLQueuePosition:
    """
    Position in Auto-Deleveraging (ADL) queue.

    ADL is triggered when insurance fund is insufficient to cover
    liquidation losses. Profitable traders on the opposite side
    are forced to close at bankruptcy price.

    Ranking based on: PnL percentile × Leverage percentile

    Attributes:
        symbol: Contract symbol
        side: Position side (LONG/SHORT)
        rank: ADL rank 1-5 (5 = highest priority for ADL)
        percentile: Position's percentile in ADL queue (0-100)
        margin_ratio: Current margin ratio
        pnl_ratio: PnL as percentage of margin
        estimated_adl_qty: Estimated qty to be ADL'd if triggered

    Reference:
        https://www.binance.com/en/support/faq/360033525711
    """
    symbol: str
    side: Literal["LONG", "SHORT"]
    rank: int  # 1-5, where 5 = highest risk of ADL
    percentile: float  # 0-100, position's rank in ADL queue
    margin_ratio: Decimal
    pnl_ratio: Decimal  # PnL / margin
    estimated_adl_qty: Optional[Decimal] = None

    @property
    def is_high_risk(self) -> bool:
        """True if in top 20% (rank 4-5) - high ADL risk."""
        return self.rank >= 4

    @property
    def risk_level(self) -> str:
        """Human-readable risk level."""
        if self.rank == 5:
            return "CRITICAL"  # Will be ADL'd first
        elif self.rank == 4:
            return "HIGH"
        elif self.rank == 3:
            return "MEDIUM"
        else:
            return "LOW"


class ADLSimulator:
    """
    Simulates Auto-Deleveraging events.

    When liquidation cannot be filled at bankruptcy price and
    insurance fund is depleted, profitable traders are ADL'd.
    """

    def __init__(self, all_positions: List[FuturesPosition]):
        self._positions = all_positions
        self._adl_queue: Dict[str, List[ADLQueuePosition]] = {}

    def build_adl_queue(
        self,
        symbol: str,
        side: Literal["LONG", "SHORT"],
        mark_price: Decimal,
    ) -> List[ADLQueuePosition]:
        """
        Build ADL queue for positions that could be ADL'd.

        ADL targets the opposite side of the liquidated position.
        If a LONG is liquidated, profitable SHORTs are ADL'd.
        """
        # Filter positions on opposite side
        opposite_side = "SHORT" if side == "LONG" else "LONG"
        candidates = [
            p for p in self._positions
            if p.symbol == symbol and (
                (opposite_side == "LONG" and p.qty > 0) or
                (opposite_side == "SHORT" and p.qty < 0)
            )
        ]

        if not candidates:
            return []

        # Calculate PnL and leverage for each
        scored = []
        for pos in candidates:
            pnl = (mark_price - pos.entry_price) * pos.qty
            pnl_pct = float(pnl / pos.margin) if pos.margin > 0 else 0.0
            leverage = float(abs(pos.qty) * mark_price / pos.margin) if pos.margin > 0 else 0.0
            scored.append((pos, pnl_pct, leverage))

        # Calculate percentiles
        pnl_values = sorted([s[1] for s in scored])
        lev_values = sorted([s[2] for s in scored])

        def percentile_rank(value, sorted_list):
            if not sorted_list:
                return 0.0
            idx = sorted_list.index(value)
            return (idx + 1) / len(sorted_list)

        queue = []
        for pos, pnl_pct, leverage in scored:
            pnl_percentile = percentile_rank(pnl_pct, pnl_values)
            lev_percentile = percentile_rank(leverage, lev_values)
            score = pnl_percentile * lev_percentile

            # ADL rank 1-5
            if score >= 0.8:
                rank = 5
            elif score >= 0.6:
                rank = 4
            elif score >= 0.4:
                rank = 3
            elif score >= 0.2:
                rank = 2
            else:
                rank = 1

            queue.append(ADLQueuePosition(
                symbol=pos.symbol,
                side=opposite_side,
                rank=rank,
                percentile=score * 100,
                margin_ratio=pos.margin_ratio if hasattr(pos, 'margin_ratio') else Decimal("0"),
                pnl_ratio=Decimal(str(pnl_pct)),
            ))

        # Sort by rank descending (highest risk first)
        queue.sort(key=lambda x: -x.rank)
        self._adl_queue[f"{symbol}_{opposite_side}"] = queue
        return queue

    def execute_adl(
        self,
        symbol: str,
        side: Literal["LONG", "SHORT"],
        qty_to_adl: Decimal,
        bankruptcy_price: Decimal,
    ) -> List[Tuple[ADLQueuePosition, Decimal]]:
        """
        Execute ADL on positions in queue order.

        Returns:
            List of (position, qty_adl'd) tuples
        """
        key = f"{symbol}_{side}"
        queue = self._adl_queue.get(key, [])

        results = []
        remaining = qty_to_adl

        for adl_pos in queue:
            if remaining <= 0:
                break

            # Find actual position
            pos = next(
                (p for p in self._positions if p.symbol == symbol),
                None
            )
            if not pos:
                continue

            adl_qty = min(remaining, abs(pos.qty))
            results.append((adl_pos, adl_qty))
            remaining -= adl_qty

        return results
```

### 2.3 Leverage Brackets Data

```python
# NEW FILE: data/futures/leverage_brackets.json
# Or fetch dynamically from Binance API

{
  "BTCUSDT": [
    {"bracket": 1, "notionalCap": 50000, "maintMarginRate": 0.004, "maxLeverage": 125},
    {"bracket": 2, "notionalCap": 250000, "maintMarginRate": 0.005, "maxLeverage": 100},
    {"bracket": 3, "notionalCap": 1000000, "maintMarginRate": 0.01, "maxLeverage": 50},
    {"bracket": 4, "notionalCap": 10000000, "maintMarginRate": 0.025, "maxLeverage": 20},
    {"bracket": 5, "notionalCap": 20000000, "maintMarginRate": 0.05, "maxLeverage": 10},
    {"bracket": 6, "notionalCap": 50000000, "maintMarginRate": 0.10, "maxLeverage": 5},
    {"bracket": 7, "notionalCap": 100000000, "maintMarginRate": 0.125, "maxLeverage": 4},
    {"bracket": 8, "notionalCap": 200000000, "maintMarginRate": 0.15, "maxLeverage": 3},
    {"bracket": 9, "notionalCap": 300000000, "maintMarginRate": 0.25, "maxLeverage": 2},
    {"bracket": 10, "notionalCap": 9223372036854776000, "maintMarginRate": 0.50, "maxLeverage": 1}
  ],
  "ETHUSDT": [
    {"bracket": 1, "notionalCap": 10000, "maintMarginRate": 0.005, "maxLeverage": 100},
    // ...
  ]
}
```

### Tests for Phase 2

```python
# NEW FILE: tests/test_futures_margin.py

class TestMarginCalculator:
    """Margin calculation tests."""

    def test_initial_margin_125x(self):
        """125x leverage = 0.8% initial margin."""
        calc = MarginCalculator(BTCUSDT_BRACKETS)
        im = calc.calculate_initial_margin(
            notional=Decimal("10000"),
            leverage=125,
        )
        assert im == Decimal("80")  # $80 for $10,000 position

    def test_maintenance_margin_tiered(self):
        """Higher notional = higher MMR."""
        calc = MarginCalculator(BTCUSDT_BRACKETS)

        # Small position: 0.4% MMR
        mm_small = calc.calculate_maintenance_margin(Decimal("10000"))
        assert mm_small == Decimal("40")

        # Large position: 5% MMR
        mm_large = calc.calculate_maintenance_margin(Decimal("15000000"))
        # Uses bracket 6: 10% MMR
        assert mm_large == Decimal("1500000")

    def test_liquidation_price_long(self):
        """Long liquidation price below entry."""
        calc = MarginCalculator(BTCUSDT_BRACKETS)
        liq_price = calc.calculate_liquidation_price(
            entry_price=Decimal("50000"),
            qty=Decimal("0.1"),  # Long
            leverage=20,
            wallet_balance=Decimal("250"),  # 5% margin
            margin_mode=MarginMode.CROSS,
        )
        assert liq_price < Decimal("50000")

    def test_liquidation_price_short(self):
        """Short liquidation price above entry."""
        calc = MarginCalculator(BTCUSDT_BRACKETS)
        liq_price = calc.calculate_liquidation_price(
            entry_price=Decimal("50000"),
            qty=Decimal("-0.1"),  # Short
            leverage=20,
            wallet_balance=Decimal("250"),
            margin_mode=MarginMode.CROSS,
        )
        assert liq_price > Decimal("50000")

class TestLiquidationEngine:
    """Liquidation simulation tests."""

    def test_no_liquidation_healthy_margin(self):
        """Position with good margin ratio not liquidated."""
        pass

    def test_liquidation_triggers_at_mmr(self):
        """Liquidation when margin ratio < 1."""
        pass

    def test_insurance_fund_covers_bankruptcy(self):
        """Insurance fund used when position goes bankrupt."""
        pass
```

### Deliverables Phase 2

- [ ] `impl_futures_margin.py` - Margin calculator
- [ ] `impl_futures_liquidation.py` - Liquidation engine
- [ ] `data/futures/leverage_brackets.json` - Bracket data
- [ ] `scripts/fetch_leverage_brackets.py` - Auto-fetch script
- [ ] `tests/test_futures_margin.py` (60+ tests)
- [ ] `tests/test_futures_liquidation.py` (40+ tests)

---

## 📦 PHASE 3A: FUNDING RATE MECHANICS (Crypto Track)

### Цели

- Реализовать funding rate tracking для crypto perpetual
- Симуляция funding payments
- Integration с P&L calculation

**Применимость**: Binance USDT-M Perpetual только. CME futures используют daily settlement (Phase 3B).

### 3.1 Funding Rate Tracker (`impl_futures_funding.py`)

```python
# NEW FILE: impl_futures_funding.py
"""
Futures funding rate mechanics.

Perpetual futures use funding to keep price close to spot index.
Funding paid every 8 hours (00:00, 08:00, 16:00 UTC).

Formula:
    Funding Payment = Position Value * Funding Rate

If Funding Rate > 0:
    Longs pay Shorts
If Funding Rate < 0:
    Shorts pay Longs

IMPORTANT: Funding Rate Conventions
------------------------------------
Binance API returns funding rate as a DECIMAL (e.g., 0.0003 = 0.03% = 3 bps).

Conversion helpers:
    rate_decimal = 0.0003           # Raw from API
    rate_percentage = rate_decimal * 100  # 0.03%
    rate_bps = rate_decimal * 10000       # 3 bps

Typical range: -0.375% to +0.375% (clamped by exchange)
Neutral rate: ~0.01% (1 bps) per 8 hours = ~0.03% daily

References:
- Binance funding: https://www.binance.com/en/support/faq/360033525031
- Binance funding rate API: returns decimal, NOT percentage
"""

from decimal import Decimal
from typing import List, Optional, Tuple
from datetime import datetime, timezone
from core_futures import FundingPayment, FuturesPosition

# Standard funding times (UTC)
FUNDING_TIMES_UTC = [0, 8, 16]  # 00:00, 08:00, 16:00

class FundingRateTracker:
    """
    Tracks and simulates funding rate payments.

    Features:
    - Historical funding rate storage
    - Payment calculation
    - Next funding time prediction
    - Funding cost estimation
    """

    def __init__(self, funding_history: Optional[List[FundingPayment]] = None):
        self._history: List[FundingPayment] = funding_history or []
        self._pending_payments: List[FundingPayment] = []

    def add_funding_rate(
        self,
        symbol: str,
        timestamp_ms: int,
        rate: Decimal,
        mark_price: Decimal,
    ) -> None:
        """Add funding rate to history."""
        pass

    def calculate_funding_payment(
        self,
        position: FuturesPosition,
        funding_rate: Decimal,
        mark_price: Decimal,
        timestamp_ms: int,
        entry_time_ms: Optional[int] = None,
        exit_time_ms: Optional[int] = None,
    ) -> FundingPayment:
        """
        Calculate funding payment for position with pro-rata support.

        Payment = Position Value * Funding Rate
        Position Value = Mark Price * |Qty|

        Pro-rata calculation:
        - If position opened/closed near funding timestamp, only charge
          for the portion of the funding period the position was held.
        - This prevents edge cases where a position opened 1 second before
          funding pays full 8-hour funding cost.

        Args:
            position: Futures position
            funding_rate: Funding rate for this period
            mark_price: Mark price at funding time
            timestamp_ms: Funding settlement timestamp
            entry_time_ms: When position was opened (None = before period start)
            exit_time_ms: When position was closed (None = still open)

        Returns:
            FundingPayment with positive = received, negative = paid
        """
        abs_qty = abs(position.qty)
        position_value = mark_price * abs_qty

        # Funding period is 8 hours = 28,800,000 ms
        FUNDING_PERIOD_MS = 8 * 3600 * 1000
        period_start_ms = timestamp_ms - FUNDING_PERIOD_MS

        # Calculate pro-rata factor (0.0 to 1.0)
        prorate_factor = Decimal("1.0")

        if entry_time_ms is not None or exit_time_ms is not None:
            # Determine effective holding period within this funding window
            effective_start = max(period_start_ms, entry_time_ms or 0)
            effective_end = min(timestamp_ms, exit_time_ms or timestamp_ms)

            # Position must have been held during the funding period
            if effective_start >= timestamp_ms:
                # Position opened after funding timestamp - no payment
                prorate_factor = Decimal("0.0")
            elif effective_end <= period_start_ms:
                # Position closed before funding period started - no payment
                prorate_factor = Decimal("0.0")
            else:
                # Calculate fraction of period held
                held_duration_ms = max(0, effective_end - effective_start)
                prorate_factor = Decimal(str(held_duration_ms)) / Decimal(str(FUNDING_PERIOD_MS))
                # Clamp to [0, 1]
                prorate_factor = max(Decimal("0"), min(Decimal("1"), prorate_factor))

        # Payment amount (with pro-rata adjustment)
        payment = position_value * funding_rate * prorate_factor

        # Sign depends on position direction
        if position.qty > 0:  # Long
            # Positive funding = longs pay → negative payment for us
            payment = -payment
        else:  # Short
            # Positive funding = shorts receive → positive payment
            payment = payment

        return FundingPayment(
            symbol=position.symbol,
            timestamp_ms=timestamp_ms,
            funding_rate=funding_rate,
            mark_price=mark_price,
            position_qty=position.qty,
            payment_amount=payment,
            prorate_factor=float(prorate_factor),  # New field for debugging
        )

    def should_apply_funding(
        self,
        position_entry_ms: int,
        position_exit_ms: Optional[int],
        funding_time_ms: int,
    ) -> bool:
        """
        Check if position should receive/pay funding at given time.

        A position is eligible for funding if:
        1. It was open BEFORE the funding timestamp
        2. It was not closed BEFORE the funding timestamp

        Edge case handling:
        - Position opened at exactly funding time: NO funding (need to hold through)
        - Position closed at exactly funding time: YES funding (held through)
        """
        if position_entry_ms >= funding_time_ms:
            return False  # Opened at or after funding - no payment

        if position_exit_ms is not None and position_exit_ms < funding_time_ms:
            return False  # Closed before funding - no payment

        return True

    def get_next_funding_time(self, current_ts_ms: int) -> int:
        """Get next funding settlement time."""
        dt = datetime.fromtimestamp(current_ts_ms / 1000, tz=timezone.utc)
        current_hour = dt.hour

        for funding_hour in FUNDING_TIMES_UTC:
            if current_hour < funding_hour:
                next_dt = dt.replace(hour=funding_hour, minute=0, second=0, microsecond=0)
                return int(next_dt.timestamp() * 1000)

        # Next funding is tomorrow 00:00 UTC
        next_dt = (dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return int(next_dt.timestamp() * 1000)

    def estimate_daily_funding_cost(
        self,
        position: FuturesPosition,
        avg_funding_rate: Decimal,
        mark_price: Decimal,
    ) -> Decimal:
        """
        Estimate daily funding cost.

        3 fundings per day = 3 * payment
        """
        single_payment = self.calculate_funding_payment(
            position, avg_funding_rate, mark_price, 0
        )
        return single_payment.payment_amount * 3

    def get_average_funding_rate(
        self,
        symbol: str,
        lookback_hours: int = 24,
    ) -> Decimal:
        """Get average funding rate over period."""
        pass
```

### 3.2 Funding Integration with Trading Env

```python
# Updates to trading_patchnew.py or new futures_env.py

class FuturesTradingEnv:
    """
    Trading environment with funding rate simulation.
    """

    def _apply_funding_if_due(
        self,
        position: FuturesPosition,
        current_ts_ms: int,
        mark_price: Decimal,
    ) -> Decimal:
        """
        Apply funding payment if funding time passed.

        Returns:
            Funding payment amount (positive = received)
        """
        if not self._is_funding_time(current_ts_ms):
            return Decimal("0")

        # Get funding rate for this timestamp
        funding_rate = self._get_funding_rate(
            position.symbol, current_ts_ms
        )

        payment = self._funding_tracker.calculate_funding_payment(
            position, funding_rate, mark_price, current_ts_ms
        )

        # Record payment
        self._funding_payments.append(payment)

        return payment.payment_amount
```

### Tests for Phase 3

```python
# NEW FILE: tests/test_futures_funding.py

class TestFundingRateTracker:
    """Funding rate tests."""

    def test_long_pays_positive_funding(self):
        """Long position pays when funding > 0."""
        tracker = FundingRateTracker()
        pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("1"),  # 1 BTC long
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )

        payment = tracker.calculate_funding_payment(
            pos,
            funding_rate=Decimal("0.0001"),  # 0.01%
            mark_price=Decimal("50000"),
            timestamp_ms=0,
        )

        # Long pays: 50000 * 0.0001 = $5
        assert payment.payment_amount == Decimal("-5")

    def test_short_receives_positive_funding(self):
        """Short position receives when funding > 0."""
        tracker = FundingRateTracker()
        pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            entry_price=Decimal("50000"),
            qty=Decimal("-1"),  # 1 BTC short
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )

        payment = tracker.calculate_funding_payment(
            pos,
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("50000"),
            timestamp_ms=0,
        )

        # Short receives: 50000 * 0.0001 = $5
        assert payment.payment_amount == Decimal("5")

    def test_next_funding_time_calculation(self):
        """Correctly predicts next funding settlement."""
        tracker = FundingRateTracker()

        # At 07:30 UTC, next is 08:00 UTC
        ts = int(datetime(2024, 1, 1, 7, 30, tzinfo=timezone.utc).timestamp() * 1000)
        next_ts = tracker.get_next_funding_time(ts)

        expected = int(datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc).timestamp() * 1000)
        assert next_ts == expected
```

### Deliverables Phase 3A

- [ ] `impl_futures_funding.py` - Funding rate tracker
- [ ] `services/futures_funding_tracker.py` - Service wrapper
- [ ] `scripts/download_funding_history.py` - Historical data download
- [ ] `tests/test_futures_funding.py` (50+ tests)

---

## 📦 PHASE 3B: INTERACTIVE BROKERS ADAPTERS (CME Track)

### Цели

- Реализовать IB TWS API адаптеры для CME futures
- Поддержка ES, NQ, GC, CL, 6E и других контрактов
- Daily settlement вместо funding rate
- Rollover handling для expiring contracts

**Применимость**: CME, COMEX, NYMEX, CBOT futures через Interactive Brokers.

### 3B.1 IB Market Data Adapter (`adapters/ib/market_data.py`)

```python
# NEW FILE: adapters/ib/market_data.py
"""
Interactive Brokers market data adapter for CME futures.

Uses ib_insync library for TWS API connectivity.
Supports real-time quotes, historical bars, and Level 2 depth.
"""

from typing import List, Optional, Iterator, Sequence
from decimal import Decimal
from ib_insync import IB, Future, ContFuture, util
from core_models import Bar, Tick
from adapters.base import MarketDataAdapter
from adapters.models import ExchangeVendor


class IBRateLimiter:
    """
    Comprehensive IB TWS API rate limiter.

    IB has multiple rate limits that MUST be respected to avoid:
    - Temporary bans (automatic pacing violations)
    - Connection drops
    - "Max messages per second exceeded" errors

    Rate Limits:
    ─────────────────────────────────────────────────────────────────────────────
    | Type                    | Limit                  | Window    | Action     |
    |-------------------------|------------------------|-----------|------------|
    | General messages        | 50 msg/sec             | 1 sec     | Block      |
    | Historical data         | 60 requests            | 10 min    | Pacing     |
    | Identical hist request  | 6 requests             | 10 min    | Cache      |
    | Market data subscribe   | 1 subscription/sec     | 1 sec     | Block      |
    | Market data lines       | 100 concurrent         | N/A       | Hard limit |
    | Scanner subscriptions   | 10 concurrent          | N/A       | Hard limit |
    | Account updates         | 1 request/sec          | 1 sec     | Block      |
    ─────────────────────────────────────────────────────────────────────────────

    Reference:
    - https://interactivebrokers.github.io/tws-api/historical_limitations.html
    - https://interactivebrokers.github.io/tws-api/market_data.html
    """

    # Rate limit constants
    MSG_PER_SEC = 45                    # General: 50 limit, 45 for safety
    HIST_PER_10MIN = 55                 # Historical: 60 limit, 55 for safety
    HIST_IDENTICAL_PER_10MIN = 5        # Identical requests: 6 limit
    SUBSCRIPTION_PER_SEC = 1            # Market data subscribe
    MAX_MARKET_DATA_LINES = 100         # Concurrent market data subscriptions
    MAX_SCANNER_SUBSCRIPTIONS = 10      # Concurrent scanner subscriptions

    def __init__(self):
        self._message_times: List[float] = []
        self._historical_times: List[float] = []
        self._historical_requests: Dict[str, List[float]] = {}  # For identical request tracking
        self._subscription_times: List[float] = []
        self._active_subscriptions: Set[str] = set()
        self._active_scanners: Set[str] = set()
        self._lock = threading.Lock()

    def can_send_message(self) -> bool:
        """Check if general message can be sent."""
        with self._lock:
            now = time.time()
            self._message_times = [t for t in self._message_times if now - t < 1.0]
            return len(self._message_times) < self.MSG_PER_SEC

    def record_message(self) -> None:
        """Record a message being sent."""
        with self._lock:
            self._message_times.append(time.time())

    def wait_for_message_slot(self, timeout: float = 5.0) -> bool:
        """Block until message can be sent or timeout."""
        start = time.time()
        while not self.can_send_message():
            if time.time() - start > timeout:
                return False
            time.sleep(0.02)  # 20ms poll
        self.record_message()
        return True

    def can_request_historical(self, request_key: Optional[str] = None) -> Tuple[bool, str]:
        """
        Check if historical data can be requested.

        Args:
            request_key: Unique key for this request (for identical request tracking)

        Returns:
            (can_request, reason_if_blocked)
        """
        now = time.time()
        window_10min = now - 600  # 10 minutes

        with self._lock:
            # Clean old entries
            self._historical_times = [t for t in self._historical_times if t > window_10min]

            # Check general historical limit
            if len(self._historical_times) >= self.HIST_PER_10MIN:
                wait_time = self._historical_times[0] - window_10min
                return False, f"Historical rate limit: wait {wait_time:.0f}s"

            # Check identical request limit
            if request_key:
                identical = self._historical_requests.get(request_key, [])
                identical = [t for t in identical if t > window_10min]
                self._historical_requests[request_key] = identical

                if len(identical) >= self.HIST_IDENTICAL_PER_10MIN:
                    wait_time = identical[0] - window_10min
                    return False, f"Identical request limit: wait {wait_time:.0f}s"

            return True, ""

    def record_historical_request(self, request_key: Optional[str] = None) -> None:
        """Record historical data request."""
        with self._lock:
            now = time.time()
            self._historical_times.append(now)
            if request_key:
                if request_key not in self._historical_requests:
                    self._historical_requests[request_key] = []
                self._historical_requests[request_key].append(now)

    def can_subscribe_market_data(self, symbol: str) -> Tuple[bool, str]:
        """Check if can subscribe to market data."""
        with self._lock:
            now = time.time()
            self._subscription_times = [t for t in self._subscription_times if now - t < 1.0]

            if len(self._subscription_times) >= self.SUBSCRIPTION_PER_SEC:
                return False, "Subscription rate limit: 1 per second"

            if len(self._active_subscriptions) >= self.MAX_MARKET_DATA_LINES:
                return False, f"Max market data lines reached ({self.MAX_MARKET_DATA_LINES})"

            return True, ""

    def record_subscription(self, symbol: str) -> None:
        """Record market data subscription."""
        with self._lock:
            self._subscription_times.append(time.time())
            self._active_subscriptions.add(symbol)

    def record_unsubscription(self, symbol: str) -> None:
        """Record market data unsubscription."""
        with self._lock:
            self._active_subscriptions.discard(symbol)

    def get_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        with self._lock:
            now = time.time()
            return {
                "messages_this_second": len([t for t in self._message_times if now - t < 1.0]),
                "messages_per_sec_limit": self.MSG_PER_SEC,
                "historical_last_10min": len([t for t in self._historical_times if now - t < 600]),
                "historical_per_10min_limit": self.HIST_PER_10MIN,
                "active_subscriptions": len(self._active_subscriptions),
                "max_subscriptions": self.MAX_MARKET_DATA_LINES,
                "active_scanners": len(self._active_scanners),
                "max_scanners": self.MAX_SCANNER_SUBSCRIPTIONS,
            }


class IBConnectionManager:
    """
    IB TWS connection lifecycle manager designed for production use.

    Handles:
    - Automatic heartbeat every 30 seconds (IB requires activity)
    - Message rate limiting (IB limit: 50 msg/sec, we use 45 for safety)
    - Exponential backoff reconnection
    - Paper vs Live account routing
    - Connection state monitoring

    References:
    - IB TWS API: https://interactivebrokers.github.io/tws-api/
    - ib_insync: https://ib-insync.readthedocs.io/
    """

    HEARTBEAT_INTERVAL_SEC = 30  # IB requires activity every 60s, we use 30 for safety
    MAX_MESSAGES_PER_SEC = 45    # IB limit is 50, leave margin for safety
    RECONNECT_DELAYS = [1, 2, 5, 10, 30, 60, 120]  # Exponential backoff (seconds)
    MAX_RECONNECT_ATTEMPTS = len(RECONNECT_DELAYS)

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,  # 7497=TWS Paper, 7496=TWS Live, 4002=Gateway Paper, 4001=Gateway Live
        client_id: int = 1,
        readonly: bool = False,
        account: Optional[str] = None,
    ):
        self._host = host
        self._port = port
        self._client_id = client_id
        self._readonly = readonly
        self._account = account

        self._ib: Optional[IB] = None
        self._connected = False
        self._reconnect_count = 0
        self._last_heartbeat_ts: float = 0.0
        self._message_count_this_second: int = 0
        self._last_second: float = 0.0

        # Rate limiter
        self._message_times: List[float] = []

    def connect(self, timeout: float = 10.0) -> bool:
        """
        Connect to TWS/Gateway with retry logic.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if connected successfully
        """
        for attempt, delay in enumerate(self.RECONNECT_DELAYS):
            try:
                self._ib = IB()
                self._ib.connect(
                    self._host,
                    self._port,
                    clientId=self._client_id,
                    readonly=self._readonly,
                    account=self._account,
                    timeout=timeout,
                )
                self._connected = True
                self._reconnect_count = 0
                self._last_heartbeat_ts = time.time()

                # Register disconnect handler
                self._ib.disconnectedEvent += self._on_disconnect

                return True

            except Exception as e:
                self._reconnect_count = attempt + 1
                if attempt < len(self.RECONNECT_DELAYS) - 1:
                    time.sleep(delay)
                else:
                    raise ConnectionError(
                        f"Failed to connect after {self.MAX_RECONNECT_ATTEMPTS} attempts: {e}"
                    )

        return False

    def disconnect(self) -> None:
        """Safely disconnect from TWS/Gateway."""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
        self._connected = False

    def _on_disconnect(self) -> None:
        """Handle unexpected disconnection - attempt reconnect."""
        self._connected = False
        try:
            self.connect()
        except ConnectionError:
            pass  # Will be handled by health check

    def send_heartbeat(self) -> None:
        """Send heartbeat to keep connection alive."""
        if not self._connected or not self._ib:
            return

        now = time.time()
        if now - self._last_heartbeat_ts >= self.HEARTBEAT_INTERVAL_SEC:
            # Request current time as heartbeat
            try:
                self._ib.reqCurrentTime()
                self._last_heartbeat_ts = now
            except Exception:
                self._connected = False

    def check_rate_limit(self) -> bool:
        """
        Check if we can send a message without exceeding rate limit.

        Returns:
            True if message can be sent, False if should wait
        """
        now = time.time()

        # Clean old entries (older than 1 second)
        self._message_times = [t for t in self._message_times if now - t < 1.0]

        if len(self._message_times) >= self.MAX_MESSAGES_PER_SEC:
            return False

        self._message_times.append(now)
        return True

    def wait_for_rate_limit(self) -> None:
        """Block until rate limit allows sending."""
        while not self.check_rate_limit():
            time.sleep(0.02)  # 20ms

    @property
    def ib(self) -> IB:
        """Get IB instance, ensuring connected."""
        if not self._connected or not self._ib:
            raise ConnectionError("Not connected to IB")
        self.send_heartbeat()  # Opportunistic heartbeat
        return self._ib

    @property
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected and self._ib is not None and self._ib.isConnected()

    def health_check(self) -> Dict[str, Any]:
        """Return connection health status."""
        return {
            "connected": self.is_connected,
            "host": self._host,
            "port": self._port,
            "client_id": self._client_id,
            "reconnect_count": self._reconnect_count,
            "last_heartbeat": self._last_heartbeat_ts,
            "messages_this_second": len(self._message_times),
        }


class IBMarketDataAdapter(MarketDataAdapter):
    """
    Interactive Brokers market data adapter.

    Configuration:
        host: TWS/Gateway host (default: 127.0.0.1)
        port: TWS port (7497 paper, 7496 live) or Gateway (4002 paper, 4001 live)
        client_id: Unique client ID
        timeout: Connection timeout seconds
        readonly: If True, no trading allowed (safer for data-only usage)
        account: Specific account to use (for multi-account setups)

    Rate Limits:
        - IB allows max 50 messages/second - we enforce 45 for safety
        - Historical data: max 60 requests per 10 minutes (pacing)
        - Market data lines: varies by subscription
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.INTERACTIVE_BROKERS,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)
        self._conn_manager: Optional[IBConnectionManager] = None
        self._host = self._config.get("host", "127.0.0.1")
        self._port = self._config.get("port", 7497)  # Paper trading default
        self._client_id = self._config.get("client_id", 1)
        self._readonly = self._config.get("readonly", True)
        self._account = self._config.get("account")

    def _do_connect(self) -> None:
        """Connect to TWS/Gateway with robust connection management."""
        self._conn_manager = IBConnectionManager(
            host=self._host,
            port=self._port,
            client_id=self._client_id,
            readonly=self._readonly,
            account=self._account,
        )
        self._conn_manager.connect(timeout=self._config.get("timeout", 10.0))

    def _do_disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self._conn_manager:
            self._conn_manager.disconnect()
            self._conn_manager = None

    @property
    def _ib(self) -> IB:
        """Get IB instance with rate limiting."""
        if not self._conn_manager:
            raise ConnectionError("Not connected")
        self._conn_manager.wait_for_rate_limit()
        return self._conn_manager.ib

    def _create_contract(self, symbol: str, use_continuous: bool = True) -> Future:
        """
        Create IB Future contract.

        Args:
            symbol: Contract symbol (ES, NQ, GC, CL, 6E)
            use_continuous: Use continuous contract (auto-roll)
        """
        # Map common symbols to IB contract details
        contract_map = {
            "ES": {"exchange": "CME", "currency": "USD"},
            "NQ": {"exchange": "CME", "currency": "USD"},
            "YM": {"exchange": "CBOT", "currency": "USD"},
            "RTY": {"exchange": "CME", "currency": "USD"},
            "GC": {"exchange": "COMEX", "currency": "USD"},
            "SI": {"exchange": "COMEX", "currency": "USD"},
            "CL": {"exchange": "NYMEX", "currency": "USD"},
            "NG": {"exchange": "NYMEX", "currency": "USD"},
            "6E": {"exchange": "CME", "currency": "USD"},
            "6J": {"exchange": "CME", "currency": "USD"},
            "6B": {"exchange": "CME", "currency": "USD"},
            "ZB": {"exchange": "CBOT", "currency": "USD"},
            "ZN": {"exchange": "CBOT", "currency": "USD"},
        }

        details = contract_map.get(symbol, {"exchange": "CME", "currency": "USD"})

        if use_continuous:
            return ContFuture(symbol, exchange=details["exchange"])
        else:
            # Specific contract (need expiry)
            return Future(symbol, exchange=details["exchange"])

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        *,
        limit: int = 500,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[Bar]:
        """
        Fetch historical bars from IB.

        Note: IB has pacing limits - max 60 requests per 10 minutes.
        """
        contract = self._create_contract(symbol)
        self._ib.qualifyContracts(contract)

        # Convert timeframe to IB format
        bar_size = self._convert_timeframe(timeframe)
        duration = self._calculate_duration(limit, timeframe)

        bars_raw = self._ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1,
        )

        return [self._parse_ib_bar(symbol, b) for b in bars_raw]

    def get_tick(self, symbol: str) -> Optional[Tick]:
        """Get current quote."""
        contract = self._create_contract(symbol)
        self._ib.qualifyContracts(contract)
        ticker = self._ib.reqMktData(contract, snapshot=True)
        self._ib.sleep(1)  # Wait for data

        if ticker.bid is None:
            return None

        return Tick(
            ts=int(util.dt(ticker.time).timestamp() * 1000),
            symbol=symbol,
            price=Decimal(str(ticker.last)) if ticker.last else None,
            bid=Decimal(str(ticker.bid)),
            ask=Decimal(str(ticker.ask)),
        )

    def get_contract_details(self, symbol: str) -> FuturesContractSpec:
        """Get contract specification from IB."""
        contract = self._create_contract(symbol, use_continuous=False)
        details = self._ib.reqContractDetails(contract)

        if not details:
            raise ValueError(f"Contract not found: {symbol}")

        d = details[0]
        return FuturesContractSpec(
            symbol=symbol,
            futures_type=self._infer_futures_type(symbol),
            contract_type=ContractType.CURRENT_QUARTER,
            exchange=Exchange(d.contract.exchange),
            base_asset=d.contract.symbol,
            quote_asset="USD",
            margin_asset="USD",
            multiplier=Decimal(str(d.contract.multiplier or 1)),
            tick_size=Decimal(str(d.minTick)),
        )

    @staticmethod
    def _convert_timeframe(timeframe: str) -> str:
        """Convert timeframe to IB bar size."""
        mapping = {
            "1m": "1 min", "5m": "5 mins", "15m": "15 mins",
            "30m": "30 mins", "1h": "1 hour", "4h": "4 hours",
            "1d": "1 day",
        }
        return mapping.get(timeframe, "1 hour")
```

### 3B.2 IB Order Execution Adapter (`adapters/ib/order_execution.py`)

```python
# NEW FILE: adapters/ib/order_execution.py
"""
Interactive Brokers order execution adapter.

Supports market, limit, stop, and bracket orders for CME futures.
"""

from ib_insync import IB, Future, MarketOrder, LimitOrder, StopOrder
from typing import Optional, List
from decimal import Decimal
from core_futures import FuturesPosition, FuturesOrder

class IBOrderExecutionAdapter:
    """
    IB futures order execution.

    Features:
    - Market, limit, stop orders
    - Bracket orders (entry + take-profit + stop-loss)
    - Position queries
    - Account margin info
    """

    def __init__(self, ib: IB):
        self._ib = ib

    def get_positions(self) -> List[FuturesPosition]:
        """Get all futures positions."""
        positions = []
        for pos in self._ib.positions():
            if isinstance(pos.contract, Future):
                positions.append(FuturesPosition(
                    symbol=pos.contract.symbol,
                    side=PositionSide.LONG if pos.position > 0 else PositionSide.SHORT,
                    entry_price=Decimal(str(pos.avgCost / pos.contract.multiplier)),
                    qty=Decimal(str(abs(pos.position))),
                    leverage=1,  # IB doesn't expose effective leverage
                    margin_mode=MarginMode.SPAN,
                ))
        return positions

    def get_account_margin(self) -> Dict[str, Decimal]:
        """Get account margin info."""
        summary = self._ib.accountSummary()
        margin_info = {}
        for item in summary:
            if item.tag in (
                "InitMarginReq", "MaintMarginReq",
                "AvailableFunds", "ExcessLiquidity", "BuyingPower"
            ):
                margin_info[item.tag] = Decimal(str(item.value))
        return margin_info

    def submit_market_order(
        self,
        symbol: str,
        side: str,
        qty: int,
    ) -> FuturesOrder:
        """Submit market order."""
        contract = self._create_contract(symbol)
        self._ib.qualifyContracts(contract)

        action = "BUY" if side.upper() == "BUY" else "SELL"
        order = MarketOrder(action, qty)

        trade = self._ib.placeOrder(contract, order)
        return self._convert_trade(trade)

    def submit_limit_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: Decimal,
    ) -> FuturesOrder:
        """Submit limit order."""
        contract = self._create_contract(symbol)
        self._ib.qualifyContracts(contract)

        action = "BUY" if side.upper() == "BUY" else "SELL"
        order = LimitOrder(action, qty, float(price))

        trade = self._ib.placeOrder(contract, order)
        return self._convert_trade(trade)

    def get_margin_requirement(
        self,
        symbol: str,
        qty: int,
    ) -> Dict[str, Decimal]:
        """Get margin requirement for order."""
        contract = self._create_contract(symbol)
        self._ib.qualifyContracts(contract)

        order = MarketOrder("BUY", qty)
        what_if = self._ib.whatIfOrder(contract, order)

        return {
            "initial_margin": Decimal(str(what_if.initMarginChange)),
            "maint_margin": Decimal(str(what_if.maintMarginChange)),
            "impact_on_margin": Decimal(str(what_if.equityWithLoanChange)),
        }
```

### 3B.3 CME Settlement & Rollover (`impl_cme_settlement.py`)

```python
# NEW FILE: impl_cme_settlement.py
"""
CME daily settlement and contract rollover.

IMPORTANT: Settlement times vary by product!
- Equity index futures (ES, NQ): 14:30 CT = 15:30 ET
- Agricultural futures: various times
- Currency futures: 14:00 CT = 15:00 ET

Rollover occurs ~8 days before contract expiry.

Reference:
- https://www.cmegroup.com/trading/equity-index/us-index/e-mini-sandp500.html
"""

from decimal import Decimal
from datetime import datetime, date, timedelta
from typing import Optional, Tuple, List

class CMESettlementEngine:
    """
    CME daily settlement simulation.

    Unlike crypto (funding every 8h), CME settles once daily.
    Variation margin is credited/debited to account.

    Settlement times (Central Time → Eastern Time):
    - Equity index (ES, NQ, YM, RTY): 14:30 CT → 15:30 ET
    - Currencies (6E, 6J, 6B): 14:00 CT → 15:00 ET
    - Metals (GC, SI): 13:30 CT → 14:30 ET
    - Energy (CL, NG): 14:30 CT → 15:30 ET
    """

    # Default settlement time (equity index futures)
    # CME uses Central Time, convert to ET (+1 hour)
    SETTLEMENT_TIME_CT = 14  # 2:30pm CT
    SETTLEMENT_TIME_ET = 15  # 3:30pm ET (14:30 CT + 1 hour)
    SETTLEMENT_MINUTE = 30   # :30

    # Product-specific settlement times (hour in ET)
    SETTLEMENT_TIMES_ET: Dict[str, Tuple[int, int]] = {
        # Equity index: 14:30 CT = 15:30 ET
        "ES": (15, 30), "NQ": (15, 30), "YM": (15, 30), "RTY": (15, 30),
        "MES": (15, 30), "MNQ": (15, 30), "MYM": (15, 30), "M2K": (15, 30),
        # Currencies: 14:00 CT = 15:00 ET
        "6E": (15, 0), "6J": (15, 0), "6B": (15, 0), "6A": (15, 0),
        # Metals: 13:30 CT = 14:30 ET
        "GC": (14, 30), "SI": (14, 30), "HG": (14, 30),
        # Energy: 14:30 CT = 15:30 ET
        "CL": (15, 30), "NG": (15, 30),
        # Bonds: 15:00 CT = 16:00 ET
        "ZB": (16, 0), "ZN": (16, 0), "ZT": (16, 0),
    }

    def get_settlement_time_et(self, symbol: str) -> Tuple[int, int]:
        """Get settlement time (hour, minute) in ET for symbol."""
        return self.SETTLEMENT_TIMES_ET.get(
            symbol.upper(),
            (self.SETTLEMENT_TIME_ET, self.SETTLEMENT_MINUTE)
        )

    def __init__(self):
        self._last_settlement_prices: Dict[str, Decimal] = {}

    def calculate_daily_settlement(
        self,
        position: FuturesPosition,
        settlement_price: Decimal,
        contract_spec: FuturesContractSpec,
    ) -> Decimal:
        """
        Calculate daily variation margin.

        Variation = (Settlement - Previous Settlement) * qty * multiplier
        """
        previous = self._last_settlement_prices.get(
            position.symbol, position.entry_price
        )

        price_change = settlement_price - previous
        variation = price_change * abs(position.qty) * contract_spec.multiplier

        # Adjust sign for short positions
        if position.qty < 0:
            variation = -variation

        self._last_settlement_prices[position.symbol] = settlement_price
        return variation

    def is_settlement_time(self, timestamp_ms: int) -> bool:
        """Check if current time is settlement time."""
        dt = datetime.utcfromtimestamp(timestamp_ms / 1000)
        # Convert to ET (UTC-5 or UTC-4 DST)
        et_hour = (dt.hour - 5) % 24
        return et_hour == self.SETTLEMENT_TIME_ET and dt.minute == 0


class ContractRolloverManager:
    """
    Contract rollover management.

    Standard roll: 8 days before expiry (2nd Thursday before 3rd Friday)
    """

    STANDARD_ROLL_DAYS_BEFORE = 8

    def __init__(self, expiration_calendar: Dict[str, List[date]]):
        """
        Args:
            expiration_calendar: Map of symbol to list of expiration dates
        """
        self._calendar = expiration_calendar

    def should_roll(self, symbol: str, current_date: date) -> bool:
        """Check if contract should be rolled."""
        expiry = self._get_current_expiry(symbol)
        if expiry is None:
            return False

        days_to_expiry = (expiry - current_date).days
        return days_to_expiry <= self.STANDARD_ROLL_DAYS_BEFORE

    def get_roll_info(
        self,
        symbol: str,
        current_date: date,
    ) -> Optional[Tuple[str, str, Decimal]]:
        """
        Get rollover information.

        Returns:
            (from_contract, to_contract, roll_adjustment)
            Roll adjustment = front price - back price (for continuous data)
        """
        pass

    def _get_current_expiry(self, symbol: str) -> Optional[date]:
        """Get current front month expiry."""
        expirations = self._calendar.get(symbol, [])
        today = date.today()
        for exp in sorted(expirations):
            if exp > today:
                return exp
        return None
```

### 3B.4 CME Trading Calendar (`services/cme_calendar.py`)

```python
# NEW FILE: services/cme_calendar.py
"""
CME trading calendar and hours.

CME Globex: Sunday 6pm ET - Friday 5pm ET
Daily maintenance: 4:15pm - 4:30pm ET
"""

from datetime import datetime, time, timedelta
from typing import Tuple, Optional
import pytz

ET = pytz.timezone("US/Eastern")

class CMETradingCalendar:
    """
    CME trading hours and holiday calendar.
    """

    # Globex hours (ET)
    GLOBEX_OPEN = time(18, 0)    # Sunday 6pm
    GLOBEX_CLOSE = time(17, 0)   # Friday 5pm
    MAINTENANCE_START = time(16, 15)
    MAINTENANCE_END = time(16, 30)

    # US market holidays (2024-2025)
    HOLIDAYS = {
        "2024-01-01", "2024-01-15", "2024-02-19",
        "2024-05-27", "2024-06-19", "2024-07-04",
        "2024-09-02", "2024-11-28", "2024-12-25",
        # Add 2025...
    }

    def is_trading_hours(self, dt: datetime) -> bool:
        """Check if within trading hours."""
        et_dt = dt.astimezone(ET)
        weekday = et_dt.weekday()
        t = et_dt.time()

        # Saturday: closed
        if weekday == 5:
            return False

        # Sunday: open from 6pm
        if weekday == 6:
            return t >= self.GLOBEX_OPEN

        # Friday: close at 5pm
        if weekday == 4 and t >= self.GLOBEX_CLOSE:
            return False

        # Maintenance window
        if self.MAINTENANCE_START <= t < self.MAINTENANCE_END:
            return False

        return True

    def get_next_open(self, dt: datetime) -> datetime:
        """Get next market open time."""
        pass

    def is_holiday(self, d: date) -> bool:
        """Check if date is a market holiday."""
        return d.isoformat() in self.HOLIDAYS
```

### Tests for Phase 3B

```python
# NEW FILE: tests/test_ib_adapters.py

class TestIBMarketDataAdapter:
    """IB market data tests."""

    def test_contract_creation_es(self):
        """Creates correct ES contract."""
        pass

    def test_historical_bars_fetched(self, mock_ib):
        """Fetches historical data from IB."""
        pass

class TestIBOrderExecutionAdapter:
    """IB order execution tests."""

    def test_market_order_submission(self, mock_ib):
        """Submits market order correctly."""
        pass

    def test_margin_requirement_query(self, mock_ib):
        """Queries margin requirements."""
        pass

class TestCMESettlement:
    """CME settlement tests."""

    def test_daily_variation_margin_long(self):
        """Calculates variation for long position."""
        pass

    def test_daily_variation_margin_short(self):
        """Calculates variation for short position."""
        pass

class TestContractRollover:
    """Rollover tests."""

    def test_roll_date_detection(self):
        """Detects roll date correctly."""
        pass

class TestCMETradingCalendar:
    """Trading hours tests."""

    def test_weekday_trading_hours(self):
        """Mon-Thu within hours."""
        pass

    def test_maintenance_window_closed(self):
        """4:15-4:30 ET closed."""
        pass

    def test_friday_close_at_5pm(self):
        """Friday closes at 5pm ET."""
        pass
```

### Deliverables Phase 3B

- [ ] `adapters/ib/__init__.py` - IB adapter package
- [ ] `adapters/ib/market_data.py` - IB market data adapter
- [ ] `adapters/ib/order_execution.py` - IB order execution
- [ ] `adapters/ib/exchange_info.py` - Contract details
- [ ] `impl_cme_settlement.py` - Daily settlement engine
- [ ] `impl_cme_rollover.py` - Contract rollover manager
- [ ] `services/cme_calendar.py` - Trading calendar
- [ ] `tests/test_ib_adapters.py` (80+ tests)
- [ ] `tests/test_cme_settlement.py` (40+ tests)

### Зависимости

```bash
pip install ib_insync  # IB TWS API wrapper
```

---

## 📦 PHASE 4A: L2 EXECUTION PROVIDER (Crypto Track)

### Цели

- Создать L2 execution provider для futures
- Адаптировать slippage model для futures
- Интеграция с существующей инфраструктурой

### 4.1 Futures Slippage Provider

```python
# NEW FILE: execution_providers_futures.py
"""
Futures execution providers (L2).

Extends crypto parametric TCA with futures-specific factors:
1. Funding rate impact on spread
2. Open interest impact on liquidity
3. Liquidation cascade simulation
4. Mark vs last price execution
"""

from decimal import Decimal
from typing import Optional
from execution_providers import (
    SlippageProvider,
    CryptoParametricSlippageProvider,
    CryptoParametricConfig,
    MarketState,
    Order,
    BarData,
)

class FuturesSlippageProvider(CryptoParametricSlippageProvider):
    """
    Futures-specific slippage model.

    Additional factors beyond crypto spot:
    - Funding rate stress (high funding = directional pressure)
    - Open interest concentration
    - Liquidation cascade risk
    """

    def __init__(
        self,
        config: Optional[FuturesSlippageConfig] = None,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self._funding_impact_sensitivity = 5.0  # Funding rate impact multiplier
        self._oi_concentration_threshold = 0.3  # 30% concentration warning

    def compute_slippage_bps(
        self,
        order: Order,
        market: MarketState,
        participation_ratio: float,
        funding_rate: Optional[float] = None,
        open_interest: Optional[float] = None,
        recent_liquidations: Optional[float] = None,  # Volume of recent liquidations
        **kwargs,
    ) -> float:
        """
        Compute futures slippage with additional factors.

        Slippage = base_slippage * funding_stress * liquidation_cascade
        """
        # Base slippage from crypto model
        base_bps = super().compute_slippage_bps(
            order, market, participation_ratio,
            funding_rate=funding_rate, **kwargs
        )

        # Funding stress factor
        # High positive funding + buying = extra cost (crowded long)
        # High negative funding + selling = extra cost (crowded short)
        funding_stress = 1.0
        if funding_rate is not None:
            is_same_direction = (
                (funding_rate > 0 and order.side == "BUY") or
                (funding_rate < 0 and order.side == "SELL")
            )
            if is_same_direction:
                funding_stress = 1.0 + abs(funding_rate) * self._funding_impact_sensitivity * 10000

        # Liquidation cascade factor
        cascade_factor = 1.0
        if recent_liquidations is not None and market.adv is not None:
            liquidation_ratio = recent_liquidations / market.adv
            if liquidation_ratio > 0.01:  # >1% of ADV is liquidations
                cascade_factor = 1.0 + liquidation_ratio * 5  # Up to 50% extra slippage

        return base_bps * funding_stress * cascade_factor

class FuturesL2ExecutionProvider:
    """
    L2 execution provider for futures.

    Combines:
    - FuturesSlippageProvider
    - FuturesFeeProvider (with funding)
    - OHLCVFillProvider (uses mark price option)
    """

    def __init__(
        self,
        use_mark_price: bool = True,  # Use mark price for execution
        **kwargs,
    ):
        self._use_mark_price = use_mark_price
        self._slippage = FuturesSlippageProvider(**kwargs)
        self._fees = FuturesFeeProvider(**kwargs)
        self._fill = OHLCVFillProvider(**kwargs)

    def execute(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
        funding_rate: Optional[float] = None,
        mark_bar: Optional[BarData] = None,  # Mark price bar
    ) -> Fill:
        """Execute order with futures mechanics."""
        # Use mark price bar if available and configured
        exec_bar = mark_bar if (self._use_mark_price and mark_bar) else bar

        # Compute fill
        fill = self._fill.compute_fill(order, market, exec_bar)

        if not fill.is_filled:
            return fill

        # Apply slippage
        slippage_bps = self._slippage.compute_slippage_bps(
            order, market, fill.participation_ratio,
            funding_rate=funding_rate,
        )

        # Adjust price
        fill = fill.with_slippage(slippage_bps)

        # Compute fees
        fee = self._fees.compute_fee(fill)

        return fill.with_fee(fee)
```

### 4.2 Futures Fee Provider

```python
# Part of execution_providers_futures.py

class FuturesFeeProvider(FeeProvider):
    """
    Futures fee provider.

    Includes:
    - Maker/taker fees (same as spot)
    - Funding payments (tracked separately)
    - Liquidation fees (if applicable)
    """

    def __init__(
        self,
        maker_bps: float = 2.0,
        taker_bps: float = 4.0,
        liquidation_fee_bps: float = 50.0,  # 0.5%
        **kwargs,
    ):
        self._maker_bps = maker_bps
        self._taker_bps = taker_bps
        self._liquidation_fee_bps = liquidation_fee_bps

    def compute_fee(
        self,
        fill: Fill,
        is_liquidation: bool = False,
    ) -> Decimal:
        """Compute fee for fill."""
        if is_liquidation:
            return fill.notional * Decimal(self._liquidation_fee_bps) / 10000

        bps = self._maker_bps if fill.is_maker else self._taker_bps
        return fill.notional * Decimal(bps) / 10000
```

### Tests for Phase 4

```python
# NEW FILE: tests/test_futures_execution_providers.py

class TestFuturesSlippageProvider:
    """Futures slippage tests."""

    def test_funding_stress_increases_slippage(self):
        """High funding rate in same direction increases slippage."""
        provider = FuturesSlippageProvider()

        # Base slippage without funding
        base = provider.compute_slippage_bps(
            Order("BTCUSDT", "BUY", Decimal("0.1"), "MARKET"),
            MarketState(0, Decimal("50000"), Decimal("50001"), adv=1e9),
            participation_ratio=0.001,
        )

        # With high positive funding (crowded long)
        with_funding = provider.compute_slippage_bps(
            Order("BTCUSDT", "BUY", Decimal("0.1"), "MARKET"),
            MarketState(0, Decimal("50000"), Decimal("50001"), adv=1e9),
            participation_ratio=0.001,
            funding_rate=0.001,  # 0.1% = very high
        )

        assert with_funding > base

    def test_liquidation_cascade_impact(self):
        """Liquidation cascade increases slippage."""
        pass

class TestFuturesL2ExecutionProvider:
    """L2 futures provider tests."""

    def test_uses_mark_price_when_configured(self):
        """Provider uses mark price bar for execution."""
        pass

    def test_integration_with_funding(self):
        """Full execution with funding rate tracking."""
        pass
```

### Deliverables Phase 4A

- [ ] `execution_providers_futures.py` - L2 crypto futures providers
- [ ] Updated factory functions in `execution_providers.py`
- [ ] `tests/test_futures_execution_providers.py` (80+ tests)

---

## 📦 PHASE 4B: CME SPAN MARGIN & SLIPPAGE (CME Track)

### Цели

- Реализовать SPAN margin calculation для CME
- Создать slippage модель для index/commodity/currency futures
- Учёт circuit breakers и daily limits

**Применимость**: ES, NQ, GC, CL, 6E и все CME/COMEX/NYMEX контракты.

### 4B.1 SPAN Margin Calculator (`impl_span_margin.py`)

```python
# NEW FILE: impl_span_margin.py
"""
CME SPAN Margin Calculator.

SPAN (Standard Portfolio Analysis of Risk) is a risk-based margin system.
Unlike crypto tiered brackets, SPAN uses portfolio-level risk analysis.

Key differences from crypto:
- Portfolio-based (offsets between correlated positions)
- Scenario-based (16 risk scenarios)
- Separate initial and maintenance margins (~80% of initial)

References:
- CME Group SPAN documentation
"""

from decimal import Decimal
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from core_futures import FuturesPosition, FuturesContractSpec

@dataclass
class SPANMarginRequirement:
    """SPAN margin requirements for a position."""
    initial_margin: Decimal
    maintenance_margin: Decimal
    scanning_risk: Decimal          # Max loss across 16 scenarios
    inter_commodity_spread: Decimal  # Credit for correlated positions
    net_option_value: Decimal = Decimal("0")

class SPANMarginCalculator:
    """
    Simplified SPAN margin calculator (APPROXIMATION).

    ⚠️ IMPORTANT LIMITATIONS:
    This is a SIMPLIFIED approximation of SPAN margin. Full SPAN (CME's official
    Standard Portfolio Analysis of Risk) uses a comprehensive 16-scenario risk
    analysis that we approximate here.

    What Full SPAN Does (NOT implemented here):
    1. 16 price/volatility scenarios: Tests portfolio under ±3σ price moves
       combined with ±33% volatility changes
    2. Inter-commodity spread credits: Complex correlation-based offsets
       between related products (e.g., ES vs NQ, WTI vs Brent)
    3. Intra-commodity spread credits: Calendar spread offsets for different
       delivery months
    4. Delivery month charges: Extra margin for near-expiry contracts
    5. Short option minimum: Floor margin for short options
    6. Net option value: Mark-to-market of option positions

    What This Implementation Does:
    - Uses simplified scanning ranges per product (±6-12% moves)
    - Provides basic inter-commodity credits for common spreads
    - Applies fixed initial/maintenance ratios (initial = 1.25x scanning,
      maintenance = 0.80x initial)

    Accuracy Assessment:
    - For single positions: ~90-95% accurate vs real SPAN
    - For simple spreads (ES/NQ): ~80-90% accurate
    - For complex portfolios: ~60-80% accurate (may underestimate margin)

    For Production Systems:
    - Use CME's official PC-SPAN calculator
    - Use IB's whatIfOrder() for real-time margin queries
    - Integrate with CME SPAN parameter files (updated daily)
    - Consider OpenGamma's Strata for enterprise-grade SPAN

    References:
    - CME SPAN Methodology: https://www.cmegroup.com/clearing/risk-management/span-overview.html
    - SPAN Parameter Files: https://www.cmegroup.com/clearing/risk-management/files-resources.html
    """

    # Default scanning ranges (% of price move)
    SCANNING_RANGES = {
        "ES": 0.08,   # ±8% price move
        "NQ": 0.10,   # ±10% (more volatile)
        "GC": 0.06,   # ±6%
        "CL": 0.12,   # ±12% (high volatility)
        "6E": 0.04,   # ±4% (currency)
        "ZB": 0.03,   # ±3% (bonds)
    }

    # Intra-commodity spread credits (same product, different months)
    SPREAD_CREDITS = {
        "ES": 0.70,  # 70% credit for calendar spreads
        "GC": 0.75,
        "CL": 0.60,  # Lower credit due to contango/backwardation
    }

    # Inter-commodity spread credits (correlated products)
    INTER_COMMODITY_CREDITS = {
        ("ES", "NQ"): 0.40,   # 40% credit for ES/NQ spread
        ("GC", "SI"): 0.30,   # Gold/Silver
        ("CL", "NG"): 0.20,   # Oil/Nat Gas
    }

    def __init__(
        self,
        contract_specs: Dict[str, FuturesContractSpec],
    ):
        self._specs = contract_specs

    def calculate_margin(
        self,
        position: FuturesPosition,
        current_price: Decimal,
    ) -> SPANMarginRequirement:
        """
        Calculate SPAN margin for single position.

        Simplified: margin = notional * scanning_range
        """
        spec = self._specs.get(position.symbol)
        if spec is None:
            raise ValueError(f"Unknown contract: {position.symbol}")

        # Calculate notional
        abs_qty = abs(position.qty)
        notional = current_price * abs_qty * spec.multiplier

        # Scanning risk (max loss under price scenarios)
        scanning_range = self.SCANNING_RANGES.get(position.symbol, 0.08)
        scanning_risk = notional * Decimal(str(scanning_range))

        # Initial margin (typically ~5-10% of notional for index futures)
        # For CME, initial ≈ scanning_risk + additional buffers
        initial_margin = scanning_risk * Decimal("1.25")

        # Maintenance margin (~80% of initial)
        maintenance_margin = initial_margin * Decimal("0.80")

        return SPANMarginRequirement(
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            scanning_risk=scanning_risk,
            inter_commodity_spread=Decimal("0"),
        )

    def calculate_portfolio_margin(
        self,
        positions: List[FuturesPosition],
        prices: Dict[str, Decimal],
    ) -> Tuple[Decimal, Decimal, Dict[str, SPANMarginRequirement]]:
        """
        Calculate SPAN margin for portfolio with spread credits.

        Returns:
            (total_initial, total_maintenance, per_position_margins)
        """
        position_margins = {}
        total_scanning = Decimal("0")

        for pos in positions:
            margin = self.calculate_margin(pos, prices[pos.symbol])
            position_margins[pos.symbol] = margin
            total_scanning += margin.scanning_risk

        # Apply inter-commodity spread credits
        spread_credit = self._calculate_spread_credit(positions, position_margins)

        # Net margin after credits
        net_scanning = max(Decimal("0"), total_scanning - spread_credit)

        total_initial = net_scanning * Decimal("1.25")
        total_maintenance = total_initial * Decimal("0.80")

        return total_initial, total_maintenance, position_margins

    def _calculate_spread_credit(
        self,
        positions: List[FuturesPosition],
        margins: Dict[str, SPANMarginRequirement],
    ) -> Decimal:
        """Calculate spread credits for correlated positions."""
        credit = Decimal("0")

        symbols = [p.symbol for p in positions]

        for (sym1, sym2), credit_pct in self.INTER_COMMODITY_CREDITS.items():
            if sym1 in symbols and sym2 in symbols:
                # Credit = smaller margin * credit_pct
                m1 = margins.get(sym1, SPANMarginRequirement(0,0,0,0)).scanning_risk
                m2 = margins.get(sym2, SPANMarginRequirement(0,0,0,0)).scanning_risk
                credit += min(m1, m2) * Decimal(str(credit_pct))

        return credit
```

### 4B.2 CME Slippage Provider (`execution_providers_cme.py`)

```python
# NEW FILE: execution_providers_cme.py
"""
CME futures slippage provider.

CME index futures (ES, NQ) have high liquidity and tight spreads.
Commodity futures (GC, CL) have seasonal and time-of-day patterns.
Currency futures (6E, 6J) correlate with forex spot markets.
"""

from decimal import Decimal
from typing import Optional
from execution_providers import (
    SlippageProvider,
    EquityParametricSlippageProvider,
    EquityParametricConfig,
    MarketState,
    Order,
)

class CMESlippageConfig:
    """CME-specific slippage configuration."""

    # Default spreads in ticks
    DEFAULT_SPREADS = {
        "ES": Decimal("0.25"),   # 1 tick = $12.50
        "NQ": Decimal("0.25"),   # 1 tick = $5.00
        "YM": Decimal("1.0"),    # 1 tick = $5.00
        "GC": Decimal("0.10"),   # 1 tick = $10.00
        "CL": Decimal("0.01"),   # 1 tick = $10.00
        "6E": Decimal("0.00005"), # 1 tick = $6.25
    }

    # Impact coefficients (similar to equity, but product-specific)
    IMPACT_COEFFICIENTS = {
        "ES": 0.03,   # Very liquid, low impact
        "NQ": 0.04,   # Slightly less liquid than ES
        "GC": 0.06,   # Commodity, moderate impact
        "CL": 0.08,   # More volatile
        "6E": 0.05,   # Currency, moderate
    }

class CMESlippageProvider(EquityParametricSlippageProvider):
    """
    CME futures slippage model.

    Based on EquityParametricSlippageProvider with:
    - Product-specific spreads and impact coefficients
    - Trading hours awareness (Globex vs RTH)
    - Daily settlement time effects
    - Circuit breaker considerations
    """

    def __init__(
        self,
        config: Optional[CMESlippageConfig] = None,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self._cme_config = config or CMESlippageConfig()
        self._circuit_breaker_active = False

    def compute_slippage_bps(
        self,
        order: Order,
        market: MarketState,
        participation_ratio: float,
        is_rth: bool = True,           # Regular Trading Hours
        minutes_to_settlement: Optional[int] = None,
        **kwargs,
    ) -> float:
        """
        Compute CME futures slippage.

        Additional factors:
        - RTH vs ETH (extended hours wider spreads)
        - Settlement time (higher volatility near 4pm ET)
        - Circuit breaker (if active, reject or limit trades)
        """
        if self._circuit_breaker_active:
            return 1000.0  # Effectively block trading

        # Get product-specific parameters
        symbol = order.symbol
        base_spread = float(self._cme_config.DEFAULT_SPREADS.get(
            symbol, Decimal("0.01")
        )) / (market.get_mid_price() or 1) * 10000

        impact_k = self._cme_config.IMPACT_COEFFICIENTS.get(symbol, 0.05)

        # RTH/ETH adjustment
        session_mult = 1.0 if is_rth else 1.5  # 50% wider in ETH

        # Settlement time premium
        settlement_mult = 1.0
        if minutes_to_settlement is not None and minutes_to_settlement < 30:
            settlement_mult = 1.0 + (30 - minutes_to_settlement) / 100  # Up to 30%

        # Base calculation
        base_slippage = base_spread / 2 + impact_k * (participation_ratio ** 0.5) * 10000

        return base_slippage * session_mult * settlement_mult

    def set_circuit_breaker(self, active: bool) -> None:
        """Set circuit breaker state (e.g., limit down)."""
        self._circuit_breaker_active = active


class CMECircuitBreaker:
    """
    CME circuit breaker and price limit simulation.

    This class implements multiple CME price protection mechanisms:

    1. EQUITY INDEX CIRCUIT BREAKERS (Rule 80B):
       - Level 1: -7% → 15 min halt (RTH only, once per day)
       - Level 2: -13% → 15 min halt (RTH only, once per day)
       - Level 3: -20% → trading halted for day
       Note: Levels 1 & 2 only apply during RTH (9:30-15:25 ET)

    2. OVERNIGHT LIMIT UP/LIMIT DOWN (Equity Index Futures):
       - During ETH (after 5pm CT previous day until RTH open)
       - Limit: ±5% from reference price
       - Orders beyond limit rejected, no halt

    3. COMMODITY PRICE LIMITS (GC, CL, NG):
       - Limit Up/Down: ±10% (varies by product)
       - Expanded limits: Can expand 2x after initial limit hit
       - No trading halt, just price ceiling/floor

    4. VELOCITY LOGIC:
       - Detects rapid price moves (potential fat finger)
       - Triggers brief (1-2 second) protective pause
       - Different thresholds per product

    5. STOP SPIKE LOGIC:
       - Prevents stop-loss cascade
       - Protects against price gaps that trigger multiple stops

    References:
    - CME Rule 80B: https://www.cmegroup.com/education/articles-and-reports/understanding-stock-index-futures-circuit-breakers.html
    - Limit Up/Limit Down: https://www.cmegroup.com/trading/equity-index/us-index/e-mini-sandp500_contract_specifications.html
    """

    # Equity Index Circuit Breaker Levels (based on S&P 500 decline from previous close)
    EQUITY_CB_LEVELS = {
        1: Decimal("-0.07"),   # -7%
        2: Decimal("-0.13"),   # -13%
        3: Decimal("-0.20"),   # -20%
    }

    EQUITY_HALT_DURATIONS_SEC = {
        1: 15 * 60,  # 15 minutes
        2: 15 * 60,  # 15 minutes
        3: None,     # Full day halt
    }

    # Overnight Price Limits (ETH only, equity index futures)
    OVERNIGHT_LIMIT_PCT = {
        "ES": Decimal("0.05"),   # ±5%
        "NQ": Decimal("0.05"),
        "YM": Decimal("0.05"),
        "RTY": Decimal("0.05"),
    }

    # Commodity Daily Price Limits
    COMMODITY_LIMITS = {
        "GC": {
            "initial": Decimal("0.05"),      # ±5% initial limit
            "expanded_1": Decimal("0.075"),  # ±7.5% expanded
            "expanded_2": Decimal("0.10"),   # ±10% max
        },
        "CL": {
            "initial": Decimal("0.07"),      # ±7%
            "expanded_1": Decimal("0.105"),  # ±10.5%
            "expanded_2": Decimal("0.14"),   # ±14%
        },
        "NG": {
            "initial": Decimal("0.10"),      # ±10% (more volatile)
            "expanded_1": Decimal("0.15"),
            "expanded_2": Decimal("0.20"),
        },
        "SI": {
            "initial": Decimal("0.07"),
            "expanded_1": Decimal("0.105"),
            "expanded_2": Decimal("0.14"),
        },
    }

    # Velocity Logic Thresholds (price move in ticks per second)
    VELOCITY_THRESHOLDS = {
        "ES": 12,    # 12 ticks/sec = 3 points/sec = $150/contract/sec
        "NQ": 20,    # 20 ticks/sec = 5 points/sec
        "GC": 30,    # 30 ticks/sec = 3 points/sec
        "CL": 50,    # 50 ticks/sec = 0.50/sec
    }

    VELOCITY_PAUSE_DURATION_MS = 2000  # 2 second protective pause

    def __init__(self, symbol: str = "ES"):
        self._symbol = symbol
        self._triggered_levels: Set[int] = set()  # Track triggered CBs for the day
        self._current_limit_expansion: int = 0     # For commodity limit expansion
        self._velocity_pause_until_ms: int = 0
        self._last_price: Optional[Decimal] = None
        self._last_price_ts_ms: int = 0

    def check_circuit_breaker(
        self,
        current_price: Decimal,
        reference_price: Decimal,  # Previous day settlement
        timestamp_ms: int,
        is_rth: bool = True,
    ) -> Optional[int]:
        """
        Check if equity index circuit breaker triggered.

        Args:
            current_price: Current price
            reference_price: Previous day settlement price
            timestamp_ms: Current timestamp
            is_rth: True if during Regular Trading Hours (9:30-15:25 ET)

        Returns:
            Circuit breaker level (1, 2, 3) or None
        """
        # Circuit breakers only apply to equity index futures
        if self._symbol not in ("ES", "NQ", "YM", "RTY"):
            return None

        change_pct = (current_price - reference_price) / reference_price

        for level, threshold in sorted(self.EQUITY_CB_LEVELS.items(), reverse=True):
            if change_pct <= threshold:
                # Level 1 & 2 only trigger during RTH
                if level in (1, 2) and not is_rth:
                    continue

                # Each level only triggers once per day
                if level in self._triggered_levels:
                    continue

                self._triggered_levels.add(level)
                return level

        return None

    def check_overnight_limit(
        self,
        price: Decimal,
        reference_price: Decimal,
        is_overnight: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check overnight price limit (limit up/down).

        Args:
            price: Order or trade price
            reference_price: Previous settlement
            is_overnight: True if in ETH session

        Returns:
            (is_within_limit, violation_type or None)
            violation_type: "LIMIT_UP" or "LIMIT_DOWN"
        """
        if not is_overnight:
            return (True, None)

        limit_pct = self.OVERNIGHT_LIMIT_PCT.get(self._symbol)
        if limit_pct is None:
            return (True, None)

        upper_limit = reference_price * (1 + limit_pct)
        lower_limit = reference_price * (1 - limit_pct)

        if price > upper_limit:
            return (False, "LIMIT_UP")
        if price < lower_limit:
            return (False, "LIMIT_DOWN")

        return (True, None)

    def check_commodity_limit(
        self,
        price: Decimal,
        reference_price: Decimal,
    ) -> Tuple[bool, Decimal, Decimal]:
        """
        Check commodity daily price limit.

        Returns:
            (is_within_limit, lower_bound, upper_bound)
        """
        limits = self.COMMODITY_LIMITS.get(self._symbol)
        if limits is None:
            return (True, Decimal("0"), Decimal("inf"))

        # Get current limit level (initial, expanded_1, expanded_2)
        if self._current_limit_expansion == 0:
            limit_pct = limits["initial"]
        elif self._current_limit_expansion == 1:
            limit_pct = limits["expanded_1"]
        else:
            limit_pct = limits["expanded_2"]

        upper = reference_price * (1 + limit_pct)
        lower = reference_price * (1 - limit_pct)

        is_within = lower <= price <= upper
        return (is_within, lower, upper)

    def expand_commodity_limit(self) -> bool:
        """
        Expand commodity limit after hitting initial limit.

        Returns:
            True if expansion successful, False if already at max
        """
        if self._current_limit_expansion >= 2:
            return False
        self._current_limit_expansion += 1
        return True

    def check_velocity_logic(
        self,
        price: Decimal,
        timestamp_ms: int,
    ) -> bool:
        """
        Check if velocity logic should trigger protective pause.

        Returns:
            True if velocity logic triggered (should pause)
        """
        if timestamp_ms < self._velocity_pause_until_ms:
            return True  # Already in pause

        if self._last_price is None:
            self._last_price = price
            self._last_price_ts_ms = timestamp_ms
            return False

        # Calculate velocity in ticks per second
        tick_size = Decimal("0.25")  # Default for ES
        if self._symbol in ("GC",):
            tick_size = Decimal("0.10")
        elif self._symbol in ("CL",):
            tick_size = Decimal("0.01")

        price_move_ticks = abs(price - self._last_price) / tick_size
        time_delta_sec = max(0.001, (timestamp_ms - self._last_price_ts_ms) / 1000)
        velocity = float(price_move_ticks / Decimal(str(time_delta_sec)))

        threshold = self.VELOCITY_THRESHOLDS.get(self._symbol, 20)

        self._last_price = price
        self._last_price_ts_ms = timestamp_ms

        if velocity > threshold:
            self._velocity_pause_until_ms = timestamp_ms + self.VELOCITY_PAUSE_DURATION_MS
            return True

        return False

    def reset_daily(self) -> None:
        """Reset circuit breakers for new trading day."""
        self._triggered_levels.clear()
        self._current_limit_expansion = 0
        self._velocity_pause_until_ms = 0
        self._last_price = None

    def get_halt_duration(self, level: int) -> Optional[int]:
        """Get halt duration in seconds for a circuit breaker level."""
        return self.EQUITY_HALT_DURATIONS_SEC.get(level)
```

### Tests for Phase 4B

```python
# NEW FILE: tests/test_span_margin.py

class TestSPANMarginCalculator:
    """SPAN margin tests."""

    def test_single_position_margin(self):
        """Calculates margin for single ES position."""
        pass

    def test_portfolio_spread_credit(self):
        """ES/NQ spread gets credit."""
        pass

    def test_maintenance_is_80_percent(self):
        """Maintenance margin ~80% of initial."""
        pass

class TestCMESlippageProvider:
    """CME slippage tests."""

    def test_es_tight_spread(self):
        """ES has tight spread ~0.25 ticks."""
        pass

    def test_eth_wider_spread(self):
        """Extended hours has wider spread."""
        pass

    def test_circuit_breaker_blocks_trading(self):
        """Circuit breaker returns high slippage."""
        pass

class TestCMECircuitBreaker:
    """Circuit breaker tests."""

    def test_level_1_at_7_percent(self):
        """Level 1 triggered at -7%."""
        pass

    def test_level_3_halts_trading(self):
        """Level 3 halts for full day."""
        pass
```

### Deliverables Phase 4B

- [ ] `impl_span_margin.py` - SPAN margin calculator
- [ ] `execution_providers_cme.py` - CME slippage provider
- [ ] `impl_circuit_breaker.py` - Circuit breaker simulation
- [ ] `tests/test_span_margin.py` (50+ tests)
- [ ] `tests/test_cme_slippage.py` (40+ tests)

---

## 📦 PHASE 5A: L3 LOB INTEGRATION (Crypto Track)

### Цели

- Интегрировать L3 LOB с futures mechanics
- Адаптировать queue position для futures
- Liquidation order flow simulation

### 5.1 Futures L3 Execution Provider

```python
# NEW FILE: execution_providers_futures_l3.py
"""
L3 Futures execution provider.

Combines:
- Full LOB simulation
- Liquidation order injection
- Mark price execution reference
- Funding-adjusted queue dynamics
"""

from lob import (
    MatchingEngine,
    QueuePositionTracker,
    AlmgrenChrissModel,
    LatencyModel,
)
from execution_providers_l3 import L3ExecutionProvider

class FuturesL3ExecutionProvider(L3ExecutionProvider):
    """
    L3 execution with futures-specific mechanics.

    Extensions:
    - Liquidation order stream injection
    - Market-wide liquidation cascade
    - Insurance fund dynamics
    """

    def __init__(
        self,
        liquidation_stream: Optional[Iterator[LiquidationEvent]] = None,
        insurance_fund_balance: Decimal = Decimal("1000000"),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._liquidation_stream = liquidation_stream
        self._insurance_fund = insurance_fund_balance
        self._pending_liquidations: List[LiquidationEvent] = []

    def inject_liquidation_orders(
        self,
        current_ts_ms: int,
    ) -> List[Fill]:
        """
        Inject liquidation orders into order book.

        Liquidation orders are market orders that must be filled.
        They can cause price cascades.
        """
        fills = []

        for event in self._get_liquidations_up_to(current_ts_ms):
            # Liquidation order
            liq_order = Order(
                symbol=event.symbol,
                side=event.side,
                qty=event.qty,
                order_type="MARKET",
                is_liquidation=True,
            )

            # Execute through matching engine
            fill = self._matching_engine.match(liq_order)
            fills.append(fill)

            # Impact on order book
            self._apply_liquidation_impact(fill)

        return fills

    def execute(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
        mark_bar: Optional[BarData] = None,
    ) -> Fill:
        """
        Execute order with liquidation cascade awareness.
        """
        # First, process any pending liquidations
        self.inject_liquidation_orders(bar.timestamp_ms)

        # Then execute our order
        return super().execute(order, market, bar)
```

### 5.2 Liquidation Cascade Simulator

```python
# Part of execution_providers_futures_l3.py

class LiquidationCascadeSimulator:
    """
    Simulates cascade liquidations.

    When price moves against leveraged positions:
    1. Positions hit liquidation price
    2. Liquidation orders placed
    3. Orders fill, push price further
    4. More positions liquidated (cascade)

    Uses historical liquidation data to calibrate.
    """

    def __init__(
        self,
        cascade_decay: float = 0.7,  # Each wave is 70% of previous
        max_waves: int = 5,
    ):
        self._cascade_decay = cascade_decay
        self._max_waves = max_waves

    def simulate_cascade(
        self,
        initial_liquidation: LiquidationEvent,
        order_book: OrderBook,
        open_positions: List[FuturesPosition],
    ) -> List[LiquidationEvent]:
        """
        Simulate liquidation cascade from initial event.
        """
        all_events = [initial_liquidation]

        for wave in range(self._max_waves):
            # Execute current liquidation
            new_price = self._execute_liquidation(
                all_events[-1], order_book
            )

            # Find new liquidations at this price
            new_liquidations = self._find_liquidations(
                new_price, open_positions
            )

            if not new_liquidations:
                break

            # Apply cascade decay
            for event in new_liquidations:
                event = event._replace(
                    qty=event.qty * Decimal(str(self._cascade_decay ** wave))
                )
                all_events.append(event)

        return all_events
```

### Tests for Phase 5

```python
# NEW FILE: tests/test_futures_l3_execution.py

class TestFuturesL3ExecutionProvider:
    """L3 futures tests."""

    def test_liquidation_injection(self):
        """Liquidation orders injected into order book."""
        pass

    def test_cascade_simulation(self):
        """Cascade liquidation simulation."""
        pass

    def test_insurance_fund_depletion(self):
        """Insurance fund properly tracks contributions."""
        pass
```

### Deliverables Phase 5A

- [ ] `execution_providers_futures_l3.py` - L3 crypto futures provider
- [ ] `lob/futures_extensions.py` - LOB extensions for crypto futures
- [ ] `tests/test_futures_l3_execution.py` (60+ tests)

---

## 📦 PHASE 5B: L3 LOB INTEGRATION (CME Track)

### Цели

- Интегрировать L3 LOB с CME-specific mechanics
- Circuit breaker simulation
- Daily settlement simulation
- Globex matching engine emulation

### 5B.1 CME L3 Execution Provider

```python
# NEW FILE: execution_providers_cme_l3.py
"""
L3 CME Futures execution provider.

Combines:
- Globex-style FIFO matching
- Circuit breaker halt simulation
- Daily settlement time effects
- RTH/ETH liquidity differences
"""

from datetime import datetime, time as dt_time
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Tuple

from lob import (
    MatchingEngine,
    QueuePositionTracker,
    AlmgrenChrissModel,
    LatencyModel,
)
from execution_providers_l3 import L3ExecutionProvider


class MarketState(str, Enum):
    """CME market state."""
    PRE_OPEN = "PRE_OPEN"
    OPENING_AUCTION = "OPENING_AUCTION"
    CONTINUOUS = "CONTINUOUS"
    HALTED = "HALTED"
    CLOSING_AUCTION = "CLOSING_AUCTION"
    CLOSED = "CLOSED"


class CMEL3ExecutionProvider(L3ExecutionProvider):
    """
    L3 execution with CME Globex-specific mechanics.

    Features:
    - Circuit breaker levels (7%, 13%, 20%)
    - Daily settlement at 4pm ET
    - Globex hours (Sun 6pm - Fri 5pm ET)
    - RTH (9:30-4:00 ET) vs ETH liquidity
    - Opening/Closing auction simulation
    """

    def __init__(
        self,
        product: str,
        multiplier: Decimal,
        tick_size: Decimal,
        daily_settle_time: dt_time = dt_time(16, 0),  # 4pm ET
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._product = product
        self._multiplier = multiplier
        self._tick_size = tick_size
        self._daily_settle_time = daily_settle_time

        # Circuit breaker state
        self._circuit_breaker = CMECircuitBreakerSimulator(product)
        self._halt_until_ts: Optional[int] = None

        # Last settlement price for daily mark
        self._last_settlement_price: Optional[Decimal] = None
        self._last_settlement_ts: Optional[int] = None

    def execute(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
    ) -> Fill:
        """
        Execute order with CME mechanics.
        """
        # Check if market is halted
        if self._is_halted(bar.timestamp_ms):
            return Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                qty=Decimal("0"),
                price=Decimal("0"),
                status="REJECTED",
                reject_reason="MARKET_HALTED",
            )

        # Check circuit breaker
        halt_result = self._circuit_breaker.check_and_trigger(
            current_price=bar.close,
            reference_price=self._last_settlement_price,
            timestamp_ms=bar.timestamp_ms,
        )

        if halt_result.is_triggered:
            self._halt_until_ts = halt_result.resume_time_ms
            return Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                qty=Decimal("0"),
                price=Decimal("0"),
                status="REJECTED",
                reject_reason=f"CIRCUIT_BREAKER_L{halt_result.level}",
            )

        # Get session-adjusted slippage
        session = self._get_trading_session(bar.timestamp_ms)
        if session in (TradingSession.ETH, TradingSession.PRE_OPEN):
            # Wider spreads and higher impact in ETH
            adjusted_slippage = self._adjust_slippage_for_eth(market)
        else:
            adjusted_slippage = market

        # Execute through parent L3 provider
        return super().execute(order, adjusted_slippage, bar)

    def _is_halted(self, timestamp_ms: int) -> bool:
        """Check if market is currently halted."""
        if self._halt_until_ts is None:
            return False
        return timestamp_ms < self._halt_until_ts

    def _get_trading_session(self, timestamp_ms: int) -> TradingSession:
        """Determine current trading session."""
        dt = datetime.utcfromtimestamp(timestamp_ms / 1000)
        hour_et = (dt.hour - 5) % 24  # Approximate ET

        if hour_et >= 9.5 and hour_et < 16:
            return TradingSession.RTH
        elif hour_et >= 16 and hour_et < 17:
            return TradingSession.CLOSING
        elif hour_et >= 18 or hour_et < 9.5:
            return TradingSession.ETH
        else:
            return TradingSession.CLOSED

    def _adjust_slippage_for_eth(
        self,
        market: MarketStateData,
    ) -> MarketStateData:
        """
        Adjust market state for ETH conditions.

        ETH typically has:
        - 2-3x wider spreads
        - 50-70% less depth
        - Higher price impact
        """
        return MarketStateData(
            timestamp=market.timestamp,
            bid=market.bid,
            ask=market.ask,
            spread_bps=market.spread_bps * Decimal("2.5"),  # 2.5x wider
            adv=market.adv * Decimal("0.4"),  # 40% of RTH volume
            depth_bid=market.depth_bid * Decimal("0.5") if market.depth_bid else None,
            depth_ask=market.depth_ask * Decimal("0.5") if market.depth_ask else None,
        )


class TradingSession(str, Enum):
    """CME trading session."""
    PRE_OPEN = "PRE_OPEN"
    RTH = "RTH"  # Regular Trading Hours 9:30-16:00 ET
    ETH = "ETH"  # Extended Trading Hours
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
```

### 5B.2 CME Circuit Breaker Simulator

```python
# Part of execution_providers_cme_l3.py

from dataclasses import dataclass
from typing import NamedTuple


class CircuitBreakerResult(NamedTuple):
    """Result of circuit breaker check."""
    is_triggered: bool
    level: int  # 1, 2, or 3
    resume_time_ms: Optional[int]
    halt_duration_minutes: int


class CMECircuitBreakerSimulator:
    """
    Simulates CME S&P 500 circuit breakers.

    Levels (based on prior day's closing price):
    - Level 1: -7%  → 15-minute halt (RTH only)
    - Level 2: -13% → 15-minute halt (RTH only)
    - Level 3: -20% → Halt for remainder of day

    Note: Halts only apply during RTH (9:30-3:25pm ET).
    After 3:25pm, only Level 3 triggers halt.
    """

    # Circuit breaker levels (percentage decline)
    LEVEL_1_PCT = Decimal("-0.07")
    LEVEL_2_PCT = Decimal("-0.13")
    LEVEL_3_PCT = Decimal("-0.20")

    # Halt durations in minutes
    HALT_DURATION_L1_L2 = 15
    HALT_DURATION_L3 = 999  # Rest of day

    # RTH boundaries for halt applicability
    RTH_START_HOUR = 9.5  # 9:30 AM ET
    RTH_CUTOFF_HOUR = 15.417  # 3:25 PM ET (after this only L3 halts)
    RTH_END_HOUR = 16.0  # 4:00 PM ET

    def __init__(self, product: str):
        self._product = product
        self._triggered_levels: set = set()
        self._last_trigger_ts: Dict[int, int] = {}

    def check_and_trigger(
        self,
        current_price: Decimal,
        reference_price: Optional[Decimal],
        timestamp_ms: int,
    ) -> CircuitBreakerResult:
        """
        Check if circuit breaker should trigger.

        Args:
            current_price: Current market price
            reference_price: Prior day settlement price
            timestamp_ms: Current timestamp

        Returns:
            CircuitBreakerResult with halt info
        """
        if reference_price is None or reference_price <= 0:
            return CircuitBreakerResult(False, 0, None, 0)

        # Calculate decline percentage
        decline_pct = (current_price - reference_price) / reference_price

        # Check if within RTH
        dt = datetime.utcfromtimestamp(timestamp_ms / 1000)
        hour_et = (dt.hour - 5) % 24  # Approximate ET

        is_rth = self.RTH_START_HOUR <= hour_et < self.RTH_END_HOUR
        is_pre_cutoff = hour_et < self.RTH_CUTOFF_HOUR

        # Determine triggered level
        triggered_level = 0

        if decline_pct <= self.LEVEL_3_PCT:
            triggered_level = 3
        elif decline_pct <= self.LEVEL_2_PCT and is_rth and is_pre_cutoff:
            triggered_level = 2
        elif decline_pct <= self.LEVEL_1_PCT and is_rth and is_pre_cutoff:
            triggered_level = 1

        if triggered_level == 0:
            return CircuitBreakerResult(False, 0, None, 0)

        # Check if this level already triggered today
        if triggered_level in self._triggered_levels:
            return CircuitBreakerResult(False, 0, None, 0)

        # Trigger circuit breaker
        self._triggered_levels.add(triggered_level)
        self._last_trigger_ts[triggered_level] = timestamp_ms

        # Calculate halt duration
        if triggered_level == 3:
            halt_minutes = self.HALT_DURATION_L3
        else:
            halt_minutes = self.HALT_DURATION_L1_L2

        resume_time_ms = timestamp_ms + (halt_minutes * 60 * 1000)

        return CircuitBreakerResult(
            is_triggered=True,
            level=triggered_level,
            resume_time_ms=resume_time_ms,
            halt_duration_minutes=halt_minutes,
        )

    def reset_daily(self) -> None:
        """Reset circuit breakers for new trading day."""
        self._triggered_levels.clear()
        self._last_trigger_ts.clear()
```

### 5B.3 Daily Settlement Simulator

```python
# Part of execution_providers_cme_l3.py

class DailySettlementSimulator:
    """
    Simulates CME daily settlement process.

    Settlement occurs at 4:00 PM ET each trading day:
    1. Settlement price determined (usually last trade or weighted average)
    2. All open positions marked to settlement price
    3. Variation margin transferred
    4. P&L realized for daily accounting
    """

    def __init__(
        self,
        settlement_time_et: dt_time = dt_time(16, 0),
    ):
        self._settlement_time = settlement_time_et
        self._last_settlement_date: Optional[date] = None
        self._settlement_prices: Dict[str, Decimal] = {}

    def compute_settlement_price(
        self,
        symbol: str,
        bars: List[BarData],
        timestamp_ms: int,
    ) -> Decimal:
        """
        Compute settlement price.

        CME uses various methods depending on product:
        - Volume-weighted average of last N minutes
        - Last trade if liquid
        - Synthetic from related products
        """
        # Simple: use VWAP of last 30 minutes
        cutoff_ms = timestamp_ms - (30 * 60 * 1000)
        relevant_bars = [b for b in bars if b.timestamp_ms >= cutoff_ms]

        if not relevant_bars:
            return bars[-1].close if bars else Decimal("0")

        total_value = sum(
            b.close * b.volume for b in relevant_bars
        )
        total_volume = sum(b.volume for b in relevant_bars)

        if total_volume > 0:
            return total_value / total_volume

        return relevant_bars[-1].close

    def process_settlement(
        self,
        positions: List[CMEFuturesPosition],
        settlement_price: Decimal,
        timestamp_ms: int,
    ) -> List[SettlementRecord]:
        """
        Process daily settlement for all positions.

        Returns:
            List of settlement records with P&L
        """
        records = []

        for pos in positions:
            # Calculate variation margin
            price_change = settlement_price - pos.last_mark_price
            variation = price_change * pos.qty * pos.multiplier

            record = SettlementRecord(
                symbol=pos.symbol,
                position_qty=pos.qty,
                prev_settle_price=pos.last_mark_price,
                new_settle_price=settlement_price,
                variation_margin=variation,
                timestamp_ms=timestamp_ms,
            )

            records.append(record)

            # Update position mark
            pos.last_mark_price = settlement_price
            pos.last_mark_ts = timestamp_ms

        return records


@dataclass
class SettlementRecord:
    """Record of daily settlement."""
    symbol: str
    position_qty: Decimal
    prev_settle_price: Decimal
    new_settle_price: Decimal
    variation_margin: Decimal
    timestamp_ms: int


@dataclass
class CMEFuturesPosition:
    """CME futures position with settlement tracking."""
    symbol: str
    qty: Decimal
    entry_price: Decimal
    multiplier: Decimal
    last_mark_price: Decimal
    last_mark_ts: int
```

### 5B.4 Globex Matching Engine Emulator

```python
# Part of lob/cme_matching.py

class GlobexMatchingEngine(MatchingEngine):
    """
    CME Globex-style matching engine.

    Features:
    - FIFO Price-Time Priority (standard)
    - Market with Protection orders
    - Stop orders with trigger logic
    - Self-Trade Prevention (STP)
    - Minimum Quantity orders
    """

    def __init__(
        self,
        symbol: str,
        tick_size: Decimal,
        max_price_bands: int = 3,  # Price banding levels
    ):
        super().__init__(symbol=symbol, tick_size=tick_size)
        self._max_price_bands = max_price_bands
        self._indicative_opening_price: Optional[Decimal] = None

    def match_with_protection(
        self,
        order: Order,
        protection_points: int = 10,
    ) -> Fill:
        """
        Match Market with Protection order.

        Market with Protection has price limit:
        - Buy: best ask + protection_points * tick_size
        - Sell: best bid - protection_points * tick_size

        Protects against sweeping the book in thin markets.
        """
        if order.side == "BUY":
            limit_price = self._best_ask + (protection_points * self._tick_size)
        else:
            limit_price = self._best_bid - (protection_points * self._tick_size)

        # Convert to limit order with protection price
        protected_order = Order(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            order_type="LIMIT",
            price=limit_price,
        )

        return self.match(protected_order)

    def calculate_opening_price(
        self,
        orders: List[Order],
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate indicative opening price.

        Uses equilibrium price that maximizes matched volume.

        Returns:
            (indicative_price, matched_volume)
        """
        if not orders:
            return Decimal("0"), Decimal("0")

        # Collect buy and sell orders
        buys = [o for o in orders if o.side == "BUY"]
        sells = [o for o in orders if o.side == "SELL"]

        if not buys or not sells:
            return Decimal("0"), Decimal("0")

        # Build cumulative volume curves
        # ... (auction matching logic)

        # Find equilibrium
        best_price = Decimal("0")
        best_volume = Decimal("0")

        # Simplified: midpoint of overlapping range
        max_buy = max(o.price for o in buys if o.price)
        min_sell = min(o.price for o in sells if o.price)

        if max_buy >= min_sell:
            best_price = (max_buy + min_sell) / 2
            best_volume = min(
                sum(o.qty for o in buys if o.price >= best_price),
                sum(o.qty for o in sells if o.price <= best_price),
            )

        self._indicative_opening_price = best_price
        return best_price, best_volume
```

### Tests for Phase 5B

```python
# NEW FILE: tests/test_cme_l3_execution.py
"""Tests for CME L3 execution provider."""

import pytest
from decimal import Decimal
from datetime import datetime, time as dt_time

from execution_providers_cme_l3 import (
    CMEL3ExecutionProvider,
    CMECircuitBreakerSimulator,
    DailySettlementSimulator,
    TradingSession,
    MarketState,
)


class TestCMECircuitBreaker:
    """Circuit breaker simulation tests."""

    def test_level_1_trigger_at_7_percent(self):
        """Level 1 triggers at -7% decline."""
        cb = CMECircuitBreakerSimulator("ES")

        result = cb.check_and_trigger(
            current_price=Decimal("4650"),
            reference_price=Decimal("5000"),  # -7%
            timestamp_ms=1700000000000,  # RTH
        )

        assert result.is_triggered
        assert result.level == 1
        assert result.halt_duration_minutes == 15

    def test_level_3_halts_rest_of_day(self):
        """Level 3 halts for remainder of day."""
        cb = CMECircuitBreakerSimulator("ES")

        result = cb.check_and_trigger(
            current_price=Decimal("4000"),
            reference_price=Decimal("5000"),  # -20%
            timestamp_ms=1700000000000,
        )

        assert result.is_triggered
        assert result.level == 3
        assert result.halt_duration_minutes == 999

    def test_no_halt_in_eth_for_level_1(self):
        """Level 1 doesn't halt during ETH."""
        cb = CMECircuitBreakerSimulator("ES")

        # Set timestamp to ETH (e.g., 7am ET)
        eth_ts = 1700000000000  # Adjust to ETH hours

        result = cb.check_and_trigger(
            current_price=Decimal("4650"),
            reference_price=Decimal("5000"),
            timestamp_ms=eth_ts,
        )

        # Should not trigger during ETH
        # (actual test would verify hour calculation)

    def test_same_level_not_triggered_twice(self):
        """Same circuit breaker level only triggers once per day."""
        cb = CMECircuitBreakerSimulator("ES")

        # First trigger
        result1 = cb.check_and_trigger(
            current_price=Decimal("4650"),
            reference_price=Decimal("5000"),
            timestamp_ms=1700000000000,
        )
        assert result1.is_triggered

        # Second trigger attempt at same level
        result2 = cb.check_and_trigger(
            current_price=Decimal("4650"),
            reference_price=Decimal("5000"),
            timestamp_ms=1700001000000,
        )
        assert not result2.is_triggered  # Already triggered

    def test_daily_reset(self):
        """Reset allows re-triggering next day."""
        cb = CMECircuitBreakerSimulator("ES")

        # Trigger
        cb.check_and_trigger(
            current_price=Decimal("4650"),
            reference_price=Decimal("5000"),
            timestamp_ms=1700000000000,
        )

        # Reset
        cb.reset_daily()

        # Can trigger again
        result = cb.check_and_trigger(
            current_price=Decimal("4650"),
            reference_price=Decimal("5000"),
            timestamp_ms=1700100000000,
        )
        assert result.is_triggered


class TestDailySettlement:
    """Daily settlement simulation tests."""

    def test_settlement_price_calculation(self):
        """Settlement price computed from VWAP."""
        sim = DailySettlementSimulator()

        bars = [
            BarData(timestamp_ms=1, close=Decimal("5000"), volume=Decimal("100")),
            BarData(timestamp_ms=2, close=Decimal("5010"), volume=Decimal("200")),
            BarData(timestamp_ms=3, close=Decimal("5005"), volume=Decimal("100")),
        ]

        price = sim.compute_settlement_price(
            symbol="ES",
            bars=bars,
            timestamp_ms=1000000,
        )

        # VWAP = (5000*100 + 5010*200 + 5005*100) / 400 = 5006.25
        assert price == Decimal("5006.25")

    def test_variation_margin_calculation(self):
        """Variation margin correctly calculated."""
        sim = DailySettlementSimulator()

        position = CMEFuturesPosition(
            symbol="ES",
            qty=Decimal("2"),
            entry_price=Decimal("5000"),
            multiplier=Decimal("50"),
            last_mark_price=Decimal("4990"),
            last_mark_ts=1000000,
        )

        records = sim.process_settlement(
            positions=[position],
            settlement_price=Decimal("5010"),  # +20 points
            timestamp_ms=2000000,
        )

        # Variation = (5010 - 4990) * 2 * 50 = $2000
        assert len(records) == 1
        assert records[0].variation_margin == Decimal("2000")


class TestCMEL3Provider:
    """CME L3 execution provider tests."""

    @pytest.fixture
    def provider(self):
        return CMEL3ExecutionProvider(
            product="ES",
            multiplier=Decimal("50"),
            tick_size=Decimal("0.25"),
        )

    def test_order_rejected_during_halt(self, provider):
        """Orders rejected when market halted."""
        # Trigger halt
        provider._halt_until_ts = 9999999999999

        order = Order(symbol="ES", side="BUY", qty=Decimal("1"))
        fill = provider.execute(order, market, bar)

        assert fill.status == "REJECTED"
        assert fill.reject_reason == "MARKET_HALTED"

    def test_eth_slippage_adjustment(self, provider):
        """ETH trading has wider spreads."""
        # Test that spread is multiplied for ETH
        pass

    def test_session_detection(self, provider):
        """Trading session correctly detected."""
        # Test RTH vs ETH detection
        pass


class TestGlobexMatching:
    """Globex matching engine tests."""

    def test_market_with_protection(self):
        """Market with protection limits sweep."""
        engine = GlobexMatchingEngine(
            symbol="ES",
            tick_size=Decimal("0.25"),
        )

        # Place some resting orders
        # Then match MWP order
        pass

    def test_opening_auction_price(self):
        """Opening auction calculates equilibrium."""
        engine = GlobexMatchingEngine(
            symbol="ES",
            tick_size=Decimal("0.25"),
        )

        orders = [
            Order(side="BUY", price=Decimal("5000"), qty=Decimal("10")),
            Order(side="BUY", price=Decimal("4995"), qty=Decimal("20")),
            Order(side="SELL", price=Decimal("4998"), qty=Decimal("15")),
            Order(side="SELL", price=Decimal("5002"), qty=Decimal("25")),
        ]

        price, volume = engine.calculate_opening_price(orders)

        # Price should maximize matched volume
        assert price > Decimal("0")
        assert volume > Decimal("0")
```

### Deliverables Phase 5B

- [ ] `execution_providers_cme_l3.py` - L3 CME futures provider
- [ ] `lob/cme_matching.py` - Globex matching engine emulator
- [ ] `tests/test_cme_l3_execution.py` (55+ tests)

---

## 📦 PHASE 6A: RISK MANAGEMENT (Crypto Track)

### Цели

- Создать futures-specific risk guards
- Интеграция с существующим risk_guard.py
- Position sizing с учётом leverage

### 6.1 Futures Risk Guards

```python
# NEW FILE: services/futures_risk_guards.py
"""
Futures-specific risk management guards.

Implements:
1. LeverageGuard - Enforce leverage limits
2. MarginGuard - Monitor margin ratios
3. LiquidationGuard - Pre-liquidation warnings
4. FundingExposureGuard - Funding rate risk management
5. ConcentrationGuard - Position concentration limits
"""

from decimal import Decimal
from typing import Optional, List, Tuple
from core_futures import FuturesPosition, MarginMode
from impl_futures_margin import MarginCalculator
from impl_futures_liquidation import LiquidationEngine

class FuturesLeverageGuard:
    """
    Enforces leverage limits per symbol and account.

    Rules:
    - Max leverage per symbol (from brackets)
    - Max account-wide leverage
    - Gradual leverage reduction at size thresholds
    """

    def __init__(
        self,
        max_account_leverage: int = 20,
        max_symbol_leverage: int = 50,
        concentration_limit: float = 0.5,  # Max 50% in single symbol
    ):
        self._max_account_leverage = max_account_leverage
        self._max_symbol_leverage = max_symbol_leverage
        self._concentration_limit = concentration_limit

    def validate_new_position(
        self,
        proposed_position: FuturesPosition,
        current_positions: List[FuturesPosition],
        account_balance: Decimal,
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Validate if new position is allowed.

        Returns:
            (is_valid, error_message, suggested_leverage)
        """
        # Check symbol leverage
        if proposed_position.leverage > self._max_symbol_leverage:
            return (
                False,
                f"Leverage {proposed_position.leverage}x exceeds max {self._max_symbol_leverage}x",
                self._max_symbol_leverage,
            )

        # Check account-wide leverage
        total_notional = sum(
            abs(p.entry_price * p.qty) for p in current_positions
        )
        new_notional = abs(proposed_position.entry_price * proposed_position.qty)
        total_after = total_notional + new_notional

        account_leverage = total_after / account_balance if account_balance > 0 else 0

        if account_leverage > self._max_account_leverage:
            return (
                False,
                f"Account leverage {account_leverage:.1f}x exceeds max {self._max_account_leverage}x",
                None,
            )

        # Check concentration
        symbol_notional = new_notional
        for p in current_positions:
            if p.symbol == proposed_position.symbol:
                symbol_notional += abs(p.entry_price * p.qty)

        concentration = symbol_notional / total_after if total_after > 0 else 1

        if concentration > self._concentration_limit:
            return (
                False,
                f"Symbol concentration {concentration:.1%} exceeds limit {self._concentration_limit:.1%}",
                None,
            )

        return (True, None, None)

class FuturesMarginGuard:
    """
    Monitors margin ratios and triggers actions.

    Levels:
    - Warning: margin_ratio < 2.0 (200%)
    - Danger: margin_ratio < 1.5 (150%)
    - Critical: margin_ratio < 1.2 (120%)
    - Liquidation: margin_ratio < 1.0 (100%)
    """

    WARNING_LEVEL = Decimal("2.0")
    DANGER_LEVEL = Decimal("1.5")
    CRITICAL_LEVEL = Decimal("1.2")
    LIQUIDATION_LEVEL = Decimal("1.0")

    def __init__(
        self,
        margin_calculator: MarginCalculator,
        auto_reduce_at_danger: bool = True,
        reduce_by_percent: float = 0.25,  # Reduce 25% at danger level
    ):
        self._calculator = margin_calculator
        self._auto_reduce = auto_reduce_at_danger
        self._reduce_percent = reduce_by_percent

    def check_margin_status(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
        wallet_balance: Decimal,
    ) -> MarginStatus:
        """Check margin status for position."""
        ratio = self._calculator.calculate_margin_ratio(
            position, mark_price, wallet_balance
        )

        if ratio < self.LIQUIDATION_LEVEL:
            return MarginStatus.LIQUIDATION
        elif ratio < self.CRITICAL_LEVEL:
            return MarginStatus.CRITICAL
        elif ratio < self.DANGER_LEVEL:
            return MarginStatus.DANGER
        elif ratio < self.WARNING_LEVEL:
            return MarginStatus.WARNING
        return MarginStatus.HEALTHY


# ═══════════════════════════════════════════════════════════════════════════════
# MARGIN CALL NOTIFICATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class MarginCallLevel(str, Enum):
    """Margin call severity levels."""
    WARNING = "warning"       # 150-200% margin ratio
    DANGER = "danger"         # 120-150% margin ratio
    CRITICAL = "critical"     # 100-120% margin ratio
    LIQUIDATION = "liquidation"  # <100% margin ratio


@dataclass
class MarginCallEvent:
    """
    Margin call notification event.

    Emitted when margin ratio crosses a threshold. Used for:
    - User notifications (email, SMS, push)
    - Automated position reduction
    - Audit logging
    - Dashboard alerts
    """
    timestamp_ms: int
    symbol: str
    level: MarginCallLevel
    margin_ratio: Decimal          # Current ratio (e.g., 1.35 = 135%)
    required_margin: Decimal       # Maintenance margin required
    current_margin: Decimal        # Current margin balance
    shortfall: Decimal             # How much to add to reach safe level
    recommended_action: str        # Human-readable recommendation
    position_qty: Decimal          # Current position quantity
    mark_price: Decimal            # Current mark price
    liquidation_price: Decimal     # Estimated liquidation price
    time_to_liquidation_bars: Optional[int] = None  # Estimated bars until liquidation
    auto_action_triggered: bool = False  # True if auto-reduce was triggered

    def __post_init__(self):
        """Calculate shortfall if not provided."""
        if self.shortfall == Decimal("0") and self.required_margin > self.current_margin:
            safe_margin = self.required_margin * Decimal("2.0")  # 200% target
            self.shortfall = safe_margin - self.current_margin

    @property
    def severity_score(self) -> int:
        """Numerical severity for sorting (4 = highest)."""
        return {
            MarginCallLevel.WARNING: 1,
            MarginCallLevel.DANGER: 2,
            MarginCallLevel.CRITICAL: 3,
            MarginCallLevel.LIQUIDATION: 4,
        }.get(self.level, 0)

    @property
    def is_urgent(self) -> bool:
        """True if immediate action required."""
        return self.level in (MarginCallLevel.CRITICAL, MarginCallLevel.LIQUIDATION)

    def to_notification_dict(self) -> Dict[str, Any]:
        """Format for notification systems (email, Telegram, etc.)."""
        return {
            "title": f"⚠️ MARGIN CALL: {self.symbol} - {self.level.value.upper()}",
            "severity": self.level.value,
            "message": self.recommended_action,
            "details": {
                "symbol": self.symbol,
                "margin_ratio": f"{float(self.margin_ratio)*100:.1f}%",
                "shortfall_usd": f"${float(self.shortfall):,.2f}",
                "position_size": str(self.position_qty),
                "mark_price": f"${float(self.mark_price):,.2f}",
                "liquidation_price": f"${float(self.liquidation_price):,.2f}",
            },
            "timestamp": self.timestamp_ms,
            "requires_ack": self.is_urgent,
        }


class MarginCallNotifier:
    """
    Margin call notification and escalation system.

    Features:
    - Multi-channel notifications (callback, log, queue)
    - Escalation ladder (warning → danger → critical)
    - Cooldown to prevent notification spam
    - Audit trail for compliance
    - Auto-acknowledge for resolved margin calls
    """

    def __init__(
        self,
        on_margin_call: Optional[Callable[[MarginCallEvent], None]] = None,
        cooldown_seconds: float = 60.0,  # Min time between same-level notifications
        escalation_speedup: float = 0.5,  # Reduce cooldown for escalating severity
        enable_auto_reduce: bool = False,
        auto_reduce_at_level: MarginCallLevel = MarginCallLevel.DANGER,
        auto_reduce_percent: float = 0.25,  # Reduce position by 25%
    ):
        self._callback = on_margin_call
        self._cooldown_sec = cooldown_seconds
        self._escalation_speedup = escalation_speedup
        self._enable_auto_reduce = enable_auto_reduce
        self._auto_reduce_level = auto_reduce_at_level
        self._auto_reduce_pct = auto_reduce_percent

        # State tracking
        self._last_notification: Dict[str, Tuple[int, MarginCallLevel]] = {}  # symbol -> (ts, level)
        self._active_margin_calls: Dict[str, MarginCallEvent] = {}
        self._notification_history: List[MarginCallEvent] = []
        self._lock = threading.Lock()

    def check_and_notify(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
        wallet_balance: Decimal,
        margin_calculator: 'MarginCalculator',
        timestamp_ms: int,
    ) -> Optional[MarginCallEvent]:
        """
        Check margin status and emit notification if needed.

        Returns:
            MarginCallEvent if notification was sent, None otherwise
        """
        ratio = margin_calculator.calculate_margin_ratio(
            position, mark_price, wallet_balance
        )

        # Determine level
        if ratio < Decimal("1.0"):
            level = MarginCallLevel.LIQUIDATION
        elif ratio < Decimal("1.2"):
            level = MarginCallLevel.CRITICAL
        elif ratio < Decimal("1.5"):
            level = MarginCallLevel.DANGER
        elif ratio < Decimal("2.0"):
            level = MarginCallLevel.WARNING
        else:
            # Margin healthy - clear any active margin call
            self._clear_margin_call(position.symbol)
            return None

        # Check cooldown
        if not self._should_notify(position.symbol, level, timestamp_ms):
            return None

        # Calculate details
        required_margin = margin_calculator.calculate_maintenance_margin(
            mark_price * abs(position.qty)
        )
        liquidation_price = margin_calculator.calculate_liquidation_price(
            position.entry_price,
            position.qty,
            position.leverage,
            wallet_balance,
            position.margin_mode,
        )

        # Build event
        event = MarginCallEvent(
            timestamp_ms=timestamp_ms,
            symbol=position.symbol,
            level=level,
            margin_ratio=ratio,
            required_margin=required_margin,
            current_margin=wallet_balance,
            shortfall=Decimal("0"),  # Calculated in __post_init__
            recommended_action=self._get_recommendation(level, ratio),
            position_qty=position.qty,
            mark_price=mark_price,
            liquidation_price=liquidation_price,
            auto_action_triggered=False,
        )

        # Check if auto-reduce should trigger
        if (
            self._enable_auto_reduce
            and level.value >= self._auto_reduce_level.value
        ):
            event.auto_action_triggered = True
            event.recommended_action += f" [AUTO-REDUCE {self._auto_reduce_pct*100:.0f}% TRIGGERED]"

        # Record and notify
        with self._lock:
            self._last_notification[position.symbol] = (timestamp_ms, level)
            self._active_margin_calls[position.symbol] = event
            self._notification_history.append(event)

        # Invoke callback
        if self._callback:
            try:
                self._callback(event)
            except Exception as e:
                logging.error(f"Margin call callback failed: {e}")

        return event

    def _should_notify(
        self,
        symbol: str,
        level: MarginCallLevel,
        timestamp_ms: int,
    ) -> bool:
        """Check if notification should be sent (respecting cooldown)."""
        with self._lock:
            if symbol not in self._last_notification:
                return True

            last_ts, last_level = self._last_notification[symbol]
            elapsed_sec = (timestamp_ms - last_ts) / 1000.0

            # Escalation = shorter cooldown
            effective_cooldown = self._cooldown_sec
            if level.value > last_level.value:
                effective_cooldown *= self._escalation_speedup

            return elapsed_sec >= effective_cooldown

    def _get_recommendation(self, level: MarginCallLevel, ratio: Decimal) -> str:
        """Generate human-readable recommendation."""
        ratio_pct = float(ratio) * 100

        if level == MarginCallLevel.LIQUIDATION:
            return f"IMMEDIATE ACTION REQUIRED: Margin ratio {ratio_pct:.1f}% - add funds or reduce position NOW to avoid liquidation"
        elif level == MarginCallLevel.CRITICAL:
            return f"URGENT: Margin ratio {ratio_pct:.1f}% - liquidation imminent. Reduce position or add margin immediately"
        elif level == MarginCallLevel.DANGER:
            return f"WARNING: Margin ratio {ratio_pct:.1f}% - consider reducing position size or adding margin"
        else:
            return f"NOTICE: Margin ratio {ratio_pct:.1f}% - monitor closely"

    def _clear_margin_call(self, symbol: str) -> None:
        """Clear active margin call when margin is restored."""
        with self._lock:
            if symbol in self._active_margin_calls:
                del self._active_margin_calls[symbol]

    def get_active_margin_calls(self) -> List[MarginCallEvent]:
        """Get all active margin calls, sorted by severity."""
        with self._lock:
            return sorted(
                self._active_margin_calls.values(),
                key=lambda e: -e.severity_score
            )

    def get_notification_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[MarginCallEvent]:
        """Get notification history for audit/compliance."""
        with self._lock:
            history = self._notification_history
            if symbol:
                history = [e for e in history if e.symbol == symbol]
            return history[-limit:]


class FundingExposureGuard:
    """
    Manages funding rate exposure.

    High funding rates mean holding costs.
    Guard can:
    - Warn on high funding exposure
    - Limit position duration during high funding
    - Force position reduction
    """

    def __init__(
        self,
        max_daily_funding_cost_bps: float = 30,  # 0.3% max daily funding cost
        funding_rate_warning_threshold: float = 0.0005,  # 0.05% per 8h
    ):
        self._max_daily_cost_bps = max_daily_funding_cost_bps
        self._warning_threshold = funding_rate_warning_threshold

    def check_funding_exposure(
        self,
        position: FuturesPosition,
        current_funding_rate: Decimal,
        predicted_funding_rates: List[Decimal],  # Next 3 funding periods
    ) -> FundingExposureStatus:
        """Evaluate funding rate exposure."""
        # Calculate expected daily funding cost
        avg_rate = sum(predicted_funding_rates) / len(predicted_funding_rates)
        daily_cost_bps = abs(float(avg_rate)) * 3 * 10000  # 3 fundings/day

        if daily_cost_bps > self._max_daily_cost_bps:
            return FundingExposureStatus.EXCESSIVE
        elif float(abs(current_funding_rate)) > self._warning_threshold:
            return FundingExposureStatus.WARNING
        return FundingExposureStatus.NORMAL
```

### 6.2 Integration with Risk Guard

```python
# Updates to risk_guard.py

class RiskGuard:
    """Extended RiskGuard with futures support."""

    def __init__(
        self,
        futures_guards: Optional[FuturesRiskGuards] = None,
        **kwargs,
    ):
        # ... existing init
        self._futures_guards = futures_guards

    def validate_trade(
        self,
        trade: Trade,
        market_type: MarketType,
        **kwargs,
    ) -> TradeValidation:
        """Validate trade with asset-class specific guards."""
        # Existing validation
        result = self._base_validate(trade)

        if market_type in (MarketType.CRYPTO_FUTURES, MarketType.CRYPTO_PERP):
            if self._futures_guards:
                futures_result = self._futures_guards.validate(trade, **kwargs)
                result = result.merge(futures_result)

        return result
```

### Tests for Phase 6

```python
# NEW FILE: tests/test_futures_risk_guards.py

class TestFuturesLeverageGuard:
    """Leverage guard tests."""

    def test_blocks_excessive_leverage(self):
        """Rejects leverage above limit."""
        guard = FuturesLeverageGuard(max_symbol_leverage=50)

        position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("1"),
            leverage=100,  # Above limit!
            margin_mode=MarginMode.CROSS,
        )

        valid, error, suggested = guard.validate_new_position(
            position, [], Decimal("10000")
        )

        assert not valid
        assert suggested == 50

    def test_concentration_limit(self):
        """Blocks excessive concentration in single symbol."""
        pass

class TestFuturesMarginGuard:
    """Margin guard tests."""

    def test_warning_level_detection(self):
        """Detects margin warning level."""
        pass

    def test_liquidation_level_action(self):
        """Triggers action at liquidation level."""
        pass
```

### Deliverables Phase 6A

- [ ] `services/futures_risk_guards.py` - Crypto futures risk guards
- [ ] Updated `risk_guard.py` with crypto futures integration
- [ ] `tests/test_futures_risk_guards.py` (80+ tests)

---

## 📦 PHASE 6B: RISK MANAGEMENT (CME Track)

### Цели

- SPAN margin integration с risk guards
- Position limits по CME rules
- Circuit breaker integration
- Daily settlement P&L tracking

### 6B.1 CME Risk Guards

```python
# NEW FILE: services/cme_risk_guards.py
"""
CME-specific risk management guards.

Implements:
1. SPANMarginGuard - Monitor SPAN margin requirements
2. PositionLimitGuard - CME position limit enforcement
3. PriceLimitGuard - Circuit breaker awareness
4. SettlementRiskGuard - Daily settlement risk
5. RolloverGuard - Contract expiry management
"""

from datetime import datetime, date, time as dt_time
from decimal import Decimal
from typing import Optional, List, Dict, Tuple
from enum import Enum

from impl_cme_margin import SPANMarginCalculator
from execution_providers_cme_l3 import CMECircuitBreakerSimulator


class MarginLevel(str, Enum):
    """Margin level states."""
    NORMAL = "NORMAL"
    WARNING = "WARNING"     # < 150% of maintenance
    DANGER = "DANGER"       # < 125% of maintenance
    MARGIN_CALL = "MARGIN_CALL"  # < 100% of maintenance


class SPANMarginGuard:
    """
    Monitors SPAN margin requirements.

    Levels:
    - Normal: margin_ratio >= 1.5
    - Warning: 1.25 <= margin_ratio < 1.5
    - Danger: 1.0 <= margin_ratio < 1.25
    - Margin Call: margin_ratio < 1.0
    """

    WARNING_RATIO = Decimal("1.50")
    DANGER_RATIO = Decimal("1.25")
    MARGIN_CALL_RATIO = Decimal("1.00")

    def __init__(
        self,
        span_calculator: SPANMarginCalculator,
        auto_reduce_at_danger: bool = True,
        reduce_target_ratio: Decimal = Decimal("2.0"),
    ):
        self._span_calc = span_calculator
        self._auto_reduce = auto_reduce_at_danger
        self._reduce_target = reduce_target_ratio

    def check_margin_status(
        self,
        positions: List[CMEFuturesPosition],
        account_equity: Decimal,
    ) -> Tuple[MarginLevel, Decimal, Optional[str]]:
        """
        Check current margin status.

        Returns:
            (margin_level, margin_ratio, action_message)
        """
        if not positions:
            return MarginLevel.NORMAL, Decimal("999"), None

        # Calculate total SPAN margin requirement
        total_margin = self._span_calc.calculate_portfolio_margin(positions)

        if total_margin <= 0:
            return MarginLevel.NORMAL, Decimal("999"), None

        margin_ratio = account_equity / total_margin

        if margin_ratio >= self.WARNING_RATIO:
            return MarginLevel.NORMAL, margin_ratio, None
        elif margin_ratio >= self.DANGER_RATIO:
            return (
                MarginLevel.WARNING,
                margin_ratio,
                f"Margin ratio {margin_ratio:.2%} below warning level {self.WARNING_RATIO:.0%}",
            )
        elif margin_ratio >= self.MARGIN_CALL_RATIO:
            return (
                MarginLevel.DANGER,
                margin_ratio,
                f"DANGER: Margin ratio {margin_ratio:.2%} - reduce position",
            )
        else:
            return (
                MarginLevel.MARGIN_CALL,
                margin_ratio,
                f"MARGIN CALL: Ratio {margin_ratio:.2%} - immediate action required",
            )

    def calculate_safe_position_size(
        self,
        symbol: str,
        current_price: Decimal,
        account_equity: Decimal,
        target_margin_ratio: Decimal = Decimal("3.0"),
    ) -> Decimal:
        """
        Calculate maximum safe position size.

        Returns max quantity that maintains target margin ratio.
        """
        margin_per_contract = self._span_calc.get_initial_margin(
            symbol, current_price
        )

        if margin_per_contract <= 0:
            return Decimal("0")

        max_contracts = (account_equity / target_margin_ratio) / margin_per_contract

        return max_contracts.quantize(Decimal("1"))


class CMEPositionLimitGuard:
    """
    Enforces CME position limits.

    Position limits vary by:
    - Product (ES: 20k, NQ: 10k, GC: 6k)
    - Account type (speculative vs hedge)
    - Month (spot month limits tighter)
    """

    # Speculative position limits (contracts)
    POSITION_LIMITS = {
        "ES": 20000,
        "NQ": 10000,
        "MES": 50000,
        "MNQ": 50000,
        "GC": 6000,
        "SI": 6000,
        "CL": 10000,
        "6E": 10000,
        "6J": 10000,
        "ZB": 25000,
    }

    # Spot month limits (usually tighter)
    SPOT_MONTH_LIMITS = {
        "ES": 20000,  # Same for financials
        "GC": 3000,   # Physical delivery
        "CL": 3000,   # Physical delivery
    }

    def __init__(
        self,
        account_type: str = "speculative",
    ):
        self._account_type = account_type

    def check_position_limit(
        self,
        symbol: str,
        proposed_qty: Decimal,
        current_qty: Decimal,
        is_spot_month: bool = False,
    ) -> Tuple[bool, Optional[str], Optional[Decimal]]:
        """
        Check if proposed position exceeds limits.

        Returns:
            (is_valid, error_message, max_allowed_qty)
        """
        # Get applicable limit
        product = self._extract_product(symbol)

        if is_spot_month and product in self.SPOT_MONTH_LIMITS:
            limit = self.SPOT_MONTH_LIMITS[product]
        else:
            limit = self.POSITION_LIMITS.get(product, 10000)  # Default

        total_qty = abs(current_qty + proposed_qty)

        if total_qty <= limit:
            return True, None, Decimal(str(limit))

        return (
            False,
            f"Position {total_qty} exceeds limit {limit} for {product}",
            Decimal(str(limit)) - abs(current_qty),
        )

    def _extract_product(self, symbol: str) -> str:
        """Extract product code from symbol (e.g., ESH24 -> ES)."""
        # Remove month/year suffix
        for i, c in enumerate(symbol):
            if c.isdigit() or (i >= 2 and c in "FGHJKMNQUVXZ"):
                return symbol[:i]
        return symbol[:2]


class CircuitBreakerAwareGuard:
    """
    Integrates circuit breaker awareness into risk management.

    Actions:
    - Reduce position sizing near limits
    - Halt new orders after circuit break
    - Widen stops during volatile periods
    """

    def __init__(
        self,
        cb_simulator: CMECircuitBreakerSimulator,
        buffer_pct: Decimal = Decimal("0.02"),  # 2% buffer before L1
    ):
        self._cb = cb_simulator
        self._buffer_pct = buffer_pct

    def check_proximity_to_limit(
        self,
        current_price: Decimal,
        reference_price: Decimal,
    ) -> Tuple[int, Decimal, str]:
        """
        Check how close to circuit breaker levels.

        Returns:
            (nearest_level, distance_pct, recommendation)
        """
        if reference_price <= 0:
            return 0, Decimal("1"), "NO_REFERENCE"

        decline_pct = (current_price - reference_price) / reference_price

        # Check distance to each level
        distance_to_l1 = decline_pct - Decimal("-0.07")
        distance_to_l2 = decline_pct - Decimal("-0.13")
        distance_to_l3 = decline_pct - Decimal("-0.20")

        if distance_to_l1 > self._buffer_pct:
            return 0, distance_to_l1, "NORMAL"
        elif distance_to_l1 > 0:
            return 1, distance_to_l1, "APPROACHING_L1"
        elif distance_to_l2 > 0:
            return 1, distance_to_l2, "BETWEEN_L1_L2"
        elif distance_to_l3 > 0:
            return 2, distance_to_l3, "BETWEEN_L2_L3"
        else:
            return 3, distance_to_l3, "BELOW_L3"

    def get_position_scale_factor(
        self,
        proximity_level: int,
        distance_pct: Decimal,
    ) -> Decimal:
        """
        Get position scaling factor based on circuit breaker proximity.
        """
        if proximity_level == 0:
            return Decimal("1.0")  # Full size
        elif proximity_level == 1:
            # Scale down linearly as approaching
            return max(Decimal("0.5"), min(distance_pct / self._buffer_pct, Decimal("1")))
        elif proximity_level == 2:
            return Decimal("0.25")  # Minimal new positions
        else:
            return Decimal("0")  # No new positions


class RolloverGuard:
    """
    Manages contract expiry and rollover risk.

    Actions:
    - Warn approaching expiry
    - Block new positions in expiring contracts
    - Suggest rollover timing
    """

    def __init__(
        self,
        rollover_days_warning: int = 5,
        rollover_days_block: int = 1,
    ):
        self._warning_days = rollover_days_warning
        self._block_days = rollover_days_block

    def check_expiry_status(
        self,
        symbol: str,
        expiry_date: date,
        today: date,
    ) -> Tuple[str, int, Optional[str]]:
        """
        Check contract expiry status.

        Returns:
            (status, days_to_expiry, message)
        """
        days_to_expiry = (expiry_date - today).days

        if days_to_expiry <= 0:
            return "EXPIRED", days_to_expiry, "Contract has expired"
        elif days_to_expiry <= self._block_days:
            return "BLOCKED", days_to_expiry, f"Contract expires in {days_to_expiry}d - no new positions"
        elif days_to_expiry <= self._warning_days:
            return "ROLLOVER_WARNING", days_to_expiry, f"Consider rolling - {days_to_expiry}d to expiry"
        else:
            return "NORMAL", days_to_expiry, None

    def suggest_rollover_contract(
        self,
        current_symbol: str,
        available_contracts: List[str],
    ) -> Optional[str]:
        """
        Suggest next contract to roll into.

        Usually the next quarterly (ESH24 -> ESM24).
        """
        # Parse current contract month/year
        # Find next available contract
        # Return suggestion
        pass
```

### 6B.2 Daily Settlement Risk

```python
# Part of services/cme_risk_guards.py

class SettlementRiskGuard:
    """
    Manages daily settlement risk.

    Tracks:
    - Unrealized P&L approaching settlement
    - Margin requirements post-settlement
    - Cash flow timing
    """

    def __init__(
        self,
        settlement_time_et: dt_time = dt_time(16, 0),
    ):
        self._settlement_time = settlement_time_et

    def check_settlement_impact(
        self,
        positions: List[CMEFuturesPosition],
        current_prices: Dict[str, Decimal],
        account_equity: Decimal,
    ) -> Dict[str, any]:
        """
        Estimate impact of upcoming settlement.

        Returns:
            {
                'total_variation': Decimal,
                'post_settle_equity': Decimal,
                'largest_loss_position': str,
                'margin_impact': str,
            }
        """
        total_variation = Decimal("0")
        largest_loss = Decimal("0")
        largest_loss_symbol = None

        for pos in positions:
            current = current_prices.get(pos.symbol, pos.last_mark_price)
            variation = (current - pos.last_mark_price) * pos.qty * pos.multiplier
            total_variation += variation

            if variation < largest_loss:
                largest_loss = variation
                largest_loss_symbol = pos.symbol

        post_settle_equity = account_equity + total_variation

        return {
            'total_variation': total_variation,
            'post_settle_equity': post_settle_equity,
            'largest_loss_position': largest_loss_symbol,
            'largest_loss_amount': largest_loss,
            'equity_change_pct': total_variation / account_equity if account_equity > 0 else Decimal("0"),
        }

    def should_reduce_before_settlement(
        self,
        settlement_impact: Dict[str, any],
        risk_threshold_pct: Decimal = Decimal("0.05"),
    ) -> Tuple[bool, Optional[str]]:
        """
        Recommend if positions should be reduced before settlement.

        Args:
            settlement_impact: Result from check_settlement_impact
            risk_threshold_pct: Max acceptable equity change
        """
        equity_change = abs(settlement_impact['equity_change_pct'])

        if equity_change > risk_threshold_pct:
            return (
                True,
                f"Large settlement impact {equity_change:.1%} > {risk_threshold_pct:.1%} threshold",
            )

        return False, None
```

### 6B.3 Tests for CME Risk Guards

```python
# NEW FILE: tests/test_cme_risk_guards.py
"""Tests for CME-specific risk guards."""

import pytest
from decimal import Decimal
from datetime import date, time as dt_time

from services.cme_risk_guards import (
    SPANMarginGuard,
    CMEPositionLimitGuard,
    CircuitBreakerAwareGuard,
    RolloverGuard,
    SettlementRiskGuard,
    MarginLevel,
)


class TestSPANMarginGuard:
    """SPAN margin guard tests."""

    def test_normal_margin_level(self):
        """Above warning level = NORMAL."""
        guard = SPANMarginGuard(span_calculator=mock_span)

        level, ratio, msg = guard.check_margin_status(
            positions=[mock_position],
            account_equity=Decimal("100000"),  # High equity
        )

        assert level == MarginLevel.NORMAL
        assert msg is None

    def test_warning_margin_level(self):
        """Below 150% = WARNING."""
        guard = SPANMarginGuard(span_calculator=mock_span)

        level, ratio, msg = guard.check_margin_status(
            positions=[mock_position],
            account_equity=Decimal("15000"),  # 140% of margin
        )

        assert level == MarginLevel.WARNING
        assert "warning" in msg.lower()

    def test_danger_margin_level(self):
        """Below 125% = DANGER."""
        guard = SPANMarginGuard(span_calculator=mock_span)

        level, ratio, msg = guard.check_margin_status(
            positions=[mock_position],
            account_equity=Decimal("12000"),  # 110% of margin
        )

        assert level == MarginLevel.DANGER
        assert "danger" in msg.lower()

    def test_margin_call_level(self):
        """Below 100% = MARGIN_CALL."""
        guard = SPANMarginGuard(span_calculator=mock_span)

        level, ratio, msg = guard.check_margin_status(
            positions=[mock_position],
            account_equity=Decimal("8000"),  # 80% of margin
        )

        assert level == MarginLevel.MARGIN_CALL
        assert "margin call" in msg.lower()


class TestPositionLimitGuard:
    """Position limit guard tests."""

    def test_within_limit(self):
        """Position within CME limits passes."""
        guard = CMEPositionLimitGuard()

        valid, msg, max_qty = guard.check_position_limit(
            symbol="ESH24",
            proposed_qty=Decimal("100"),
            current_qty=Decimal("50"),
            is_spot_month=False,
        )

        assert valid is True
        assert msg is None

    def test_exceeds_limit(self):
        """Position exceeding limit fails."""
        guard = CMEPositionLimitGuard()

        valid, msg, max_qty = guard.check_position_limit(
            symbol="ESH24",
            proposed_qty=Decimal("25000"),
            current_qty=Decimal("0"),
            is_spot_month=False,
        )

        assert valid is False
        assert "exceeds limit" in msg

    def test_spot_month_tighter_limits(self):
        """Spot month has tighter limits for physical delivery."""
        guard = CMEPositionLimitGuard()

        # Within regular limit but exceeds spot month limit
        valid, msg, max_qty = guard.check_position_limit(
            symbol="GCZ24",
            proposed_qty=Decimal("4000"),
            current_qty=Decimal("0"),
            is_spot_month=True,
        )

        assert valid is False  # Exceeds 3000 spot month limit

    def test_product_extraction(self):
        """Product code extracted correctly from symbol."""
        guard = CMEPositionLimitGuard()

        assert guard._extract_product("ESH24") == "ES"
        assert guard._extract_product("GCZ24") == "GC"
        assert guard._extract_product("6EH24") == "6E"


class TestCircuitBreakerAwareGuard:
    """Circuit breaker awareness tests."""

    def test_normal_distance(self):
        """Far from circuit breaker = NORMAL."""
        guard = CircuitBreakerAwareGuard(cb_simulator=mock_cb)

        level, distance, rec = guard.check_proximity_to_limit(
            current_price=Decimal("5000"),
            reference_price=Decimal("5000"),  # No decline
        )

        assert level == 0
        assert rec == "NORMAL"

    def test_approaching_level_1(self):
        """Near -7% decline = APPROACHING_L1."""
        guard = CircuitBreakerAwareGuard(cb_simulator=mock_cb)

        level, distance, rec = guard.check_proximity_to_limit(
            current_price=Decimal("4700"),
            reference_price=Decimal("5000"),  # -6%
        )

        assert level == 1
        assert rec == "APPROACHING_L1"

    def test_position_scale_factor(self):
        """Position scaling reduces near limits."""
        guard = CircuitBreakerAwareGuard(cb_simulator=mock_cb)

        # Normal
        assert guard.get_position_scale_factor(0, Decimal("0.05")) == Decimal("1.0")

        # Approaching L1
        scale = guard.get_position_scale_factor(1, Decimal("0.01"))
        assert scale < Decimal("1.0")

        # Between L2-L3
        assert guard.get_position_scale_factor(2, Decimal("0.02")) == Decimal("0.25")

        # Below L3
        assert guard.get_position_scale_factor(3, Decimal("0")) == Decimal("0")


class TestRolloverGuard:
    """Contract rollover guard tests."""

    def test_normal_expiry(self):
        """Far from expiry = NORMAL."""
        guard = RolloverGuard(rollover_days_warning=5, rollover_days_block=1)

        status, days, msg = guard.check_expiry_status(
            symbol="ESH24",
            expiry_date=date(2024, 3, 15),
            today=date(2024, 3, 1),  # 14 days to expiry
        )

        assert status == "NORMAL"
        assert days == 14
        assert msg is None

    def test_rollover_warning(self):
        """Within warning period = ROLLOVER_WARNING."""
        guard = RolloverGuard(rollover_days_warning=5, rollover_days_block=1)

        status, days, msg = guard.check_expiry_status(
            symbol="ESH24",
            expiry_date=date(2024, 3, 15),
            today=date(2024, 3, 11),  # 4 days to expiry
        )

        assert status == "ROLLOVER_WARNING"
        assert "rolling" in msg.lower()

    def test_blocked_near_expiry(self):
        """Very close to expiry = BLOCKED."""
        guard = RolloverGuard(rollover_days_warning=5, rollover_days_block=1)

        status, days, msg = guard.check_expiry_status(
            symbol="ESH24",
            expiry_date=date(2024, 3, 15),
            today=date(2024, 3, 14),  # 1 day to expiry
        )

        assert status == "BLOCKED"
        assert "no new positions" in msg.lower()


class TestSettlementRiskGuard:
    """Settlement risk guard tests."""

    def test_settlement_impact_calculation(self):
        """Settlement impact correctly calculated."""
        guard = SettlementRiskGuard()

        positions = [
            CMEFuturesPosition(
                symbol="ES",
                qty=Decimal("2"),
                entry_price=Decimal("5000"),
                multiplier=Decimal("50"),
                last_mark_price=Decimal("5000"),
                last_mark_ts=0,
            ),
        ]

        impact = guard.check_settlement_impact(
            positions=positions,
            current_prices={"ES": Decimal("5020")},  # +20 points
            account_equity=Decimal("50000"),
        )

        # Variation = +20 * 2 * 50 = $2000
        assert impact['total_variation'] == Decimal("2000")
        assert impact['post_settle_equity'] == Decimal("52000")

    def test_large_settlement_triggers_warning(self):
        """Large settlement impact recommends reduction."""
        guard = SettlementRiskGuard()

        impact = {
            'equity_change_pct': Decimal("0.10"),  # 10% change
            'total_variation': Decimal("-5000"),
        }

        should_reduce, msg = guard.should_reduce_before_settlement(
            settlement_impact=impact,
            risk_threshold_pct=Decimal("0.05"),
        )

        assert should_reduce is True
        assert "threshold" in msg
```

### Deliverables Phase 6B

- [ ] `services/cme_risk_guards.py` - CME-specific risk guards
- [ ] `impl_cme_settlement.py` - Daily settlement integration
- [ ] `tests/test_cme_risk_guards.py` (70+ tests)

---

## 📦 PHASE 7A: FEATURES & DATA PIPELINE (Crypto Track)

### Цели

- Создать futures-specific features
- Интеграция funding rate в features
- Open interest и liquidation features

### 7.1 Futures Features

```python
# NEW FILE: futures_features.py
"""
Futures-specific feature engineering.

Features unique to futures:
1. Funding rate features (current, predicted, cumulative)
2. Open interest features (change, concentration)
3. Basis features (spot-futures spread)
4. Liquidation features (recent liquidations, cascade risk)
5. Mark-index spread features

CRITICAL: Look-Ahead Bias Prevention
─────────────────────────────────────
ALL features MUST be shifted by 1 period before use in training!

At time t, we can only use data known at t-1. Features computed from
data at time t have look-ahead bias if used to predict actions at t.

Pattern:
    raw_feature = compute_feature(data)
    shifted_feature = raw_feature.shift(1)  # <-- REQUIRED!

This follows the same pattern as features_pipeline.py:339-353.
"""

from decimal import Decimal
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def shift_features_for_lookahead(
    features: pd.DataFrame,
    shift_periods: int = 1,
    exclude_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Shift all features by N periods to prevent look-ahead bias.

    MUST be called after all features are computed, before training.

    Args:
        features: DataFrame with computed features
        shift_periods: Number of periods to shift (default=1)
        exclude_cols: Columns to NOT shift (e.g., timestamps)

    Returns:
        DataFrame with shifted features

    Example:
        >>> features = calculate_funding_features(funding_rates)
        >>> features = shift_features_for_lookahead(features)
        >>> # Now safe for training at time t
    """
    exclude_cols = exclude_cols or []
    shifted = features.copy()

    for col in shifted.columns:
        if col not in exclude_cols:
            shifted[col] = shifted[col].shift(shift_periods)

    return shifted


def calculate_funding_features(
    funding_rates: pd.Series,
    lookback_periods: int = 8,  # 8 fundings = ~2.67 days
    apply_shift: bool = True,   # Auto-apply look-ahead prevention
) -> pd.DataFrame:
    """
    Calculate funding rate features.

    Features:
    - funding_rate_current: Current funding rate
    - funding_rate_sma: SMA of funding rates
    - funding_rate_std: Volatility of funding
    - funding_cumulative_24h: Cumulative funding last 24h
    - funding_direction: Sign consistency (-1 to 1)

    Args:
        funding_rates: Series of funding rates (decimal, e.g., 0.0003 = 3 bps)
        lookback_periods: Number of funding periods for rolling calculations
        apply_shift: If True, auto-apply shift(1) for look-ahead prevention

    IMPORTANT: If apply_shift=False, caller MUST apply shift manually!
    """
    features = pd.DataFrame(index=funding_rates.index)

    features['funding_rate_current'] = funding_rates
    features['funding_rate_sma'] = funding_rates.rolling(lookback_periods).mean()
    features['funding_rate_std'] = funding_rates.rolling(lookback_periods).std()
    features['funding_cumulative_24h'] = funding_rates.rolling(3).sum()  # 3 = 24h

    # Direction consistency
    signs = np.sign(funding_rates)
    features['funding_direction'] = signs.rolling(lookback_periods).mean()

    # Apply look-ahead bias prevention shift
    if apply_shift:
        features = shift_features_for_lookahead(features, shift_periods=1)

    return features

def calculate_open_interest_features(
    open_interest: pd.Series,
    price: pd.Series,
    lookback: int = 20,
    apply_shift: bool = True,   # Auto-apply look-ahead prevention
) -> pd.DataFrame:
    """
    Calculate open interest features.

    Features:
    - oi_change_pct: OI change percentage
    - oi_price_divergence: OI vs price divergence
    - oi_concentration: OI concentration metric

    Args:
        open_interest: Series of open interest values
        price: Series of prices
        lookback: Rolling window for calculations
        apply_shift: If True, auto-apply shift(1) for look-ahead prevention
    """
    features = pd.DataFrame(index=open_interest.index)

    features['oi_change_pct'] = open_interest.pct_change(lookback)

    # OI-Price divergence (rising OI + falling price = bearish signal)
    oi_change = open_interest.pct_change(lookback)
    price_change = price.pct_change(lookback)
    features['oi_price_divergence'] = oi_change - price_change

    # Apply look-ahead bias prevention shift
    if apply_shift:
        features = shift_features_for_lookahead(features, shift_periods=1)

    return features


def calculate_basis_features(
    futures_price: pd.Series,
    spot_price: pd.Series,
    days_to_expiry: Optional[pd.Series] = None,
    apply_shift: bool = True,   # Auto-apply look-ahead prevention
) -> pd.DataFrame:
    """
    Calculate basis features.

    Basis = Futures Price - Spot Price
    Annualized Basis = (Basis / Spot) * (365 / DTE) for quarterly contracts

    For perpetuals, basis approximates funding expectation.

    Args:
        futures_price: Series of futures prices
        spot_price: Series of spot prices
        days_to_expiry: Optional series of days to expiry (for quarterly)
        apply_shift: If True, auto-apply shift(1) for look-ahead prevention
    """
    features = pd.DataFrame(index=futures_price.index)

    basis = futures_price - spot_price
    features['basis'] = basis
    features['basis_pct'] = basis / spot_price * 100

    if days_to_expiry is not None:
        features['basis_annualized'] = (basis / spot_price) * (365 / days_to_expiry) * 100

    # Apply look-ahead bias prevention shift
    if apply_shift:
        features = shift_features_for_lookahead(features, shift_periods=1)

    return features


def compute_basis_features_extended(
    futures_price: pd.Series,
    spot_price: pd.Series,
    funding_rate: Optional[pd.Series] = None,
    days_to_expiry: Optional[pd.Series] = None,
    risk_free_rate: float = 0.05,  # Annual risk-free rate
    storage_cost: float = 0.0,     # For commodities
) -> pd.DataFrame:
    """
    Extended basis features for cash-and-carry arbitrage analysis.

    Implements research-backed metrics for basis trading:
    - Theoretical fair value using cost-of-carry model
    - Basis mispricing signals
    - Roll yield for term structure trading
    - Carry trade opportunity detection

    References:
    - Hull, J.C. (2018): Options, Futures, and Other Derivatives
    - CME Group (2023): Understanding Basis Trading

    Args:
        futures_price: Futures price series
        spot_price: Underlying spot price series
        funding_rate: Funding rate for perpetuals (optional)
        days_to_expiry: Days to contract expiration (optional)
        risk_free_rate: Annualized risk-free rate
        storage_cost: Annualized storage cost (for commodities like GC, CL)

    Returns:
        DataFrame with extended basis features
    """
    features = pd.DataFrame(index=futures_price.index)

    # ─────────────────────────────────────────────────────────────────────
    # 1. BASIC BASIS METRICS
    # ─────────────────────────────────────────────────────────────────────
    basis = futures_price - spot_price
    features['basis_absolute'] = basis
    features['basis_pct'] = basis / spot_price * 100
    features['basis_log'] = np.log(futures_price / spot_price) * 100

    # ─────────────────────────────────────────────────────────────────────
    # 2. THEORETICAL FAIR VALUE (Cost-of-Carry Model)
    # ─────────────────────────────────────────────────────────────────────
    # F_theoretical = S * exp((r + c) * T)
    # where: r = risk-free rate, c = storage cost, T = time to expiry
    if days_to_expiry is not None:
        time_to_expiry = days_to_expiry / 365.0
        cost_of_carry = risk_free_rate + storage_cost
        fair_value = spot_price * np.exp(cost_of_carry * time_to_expiry)
        features['fair_value_theoretical'] = fair_value
        features['mispricing'] = futures_price - fair_value
        features['mispricing_pct'] = (features['mispricing'] / fair_value) * 100

        # Annualized basis for quarterly contracts
        features['basis_annualized'] = (basis / spot_price) * (365 / days_to_expiry.replace(0, np.nan)) * 100

        # Implied yield: what rate does the market imply?
        # F = S * exp(y * T) => y = ln(F/S) / T
        features['implied_yield'] = (np.log(futures_price / spot_price) / time_to_expiry.replace(0, np.nan)) * 100

    # ─────────────────────────────────────────────────────────────────────
    # 3. CARRY TRADE SIGNALS (Contango/Backwardation)
    # ─────────────────────────────────────────────────────────────────────
    features['is_contango'] = (basis > 0).astype(float)
    features['is_backwardation'] = (basis < 0).astype(float)

    # Basis regime (normalized between -1 and 1)
    basis_percentile = basis.rolling(window=252, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )
    features['basis_regime'] = (basis_percentile - 0.5) * 2  # Scale to [-1, 1]

    # ─────────────────────────────────────────────────────────────────────
    # 4. ROLL YIELD FEATURES
    # ─────────────────────────────────────────────────────────────────────
    # Roll yield = (F_near - F_far) / F_near * (365 / days_between)
    # Positive roll yield in backwardation = profit from rolling long position
    if days_to_expiry is not None:
        features['roll_yield_signal'] = -features['basis_annualized']  # Negative basis = positive roll

    # ─────────────────────────────────────────────────────────────────────
    # 5. FUNDING RATE INTEGRATION (Perpetuals)
    # ─────────────────────────────────────────────────────────────────────
    # For perpetuals: basis ≈ expected funding (arbitrage keeps them aligned)
    if funding_rate is not None:
        features['funding_rate'] = funding_rate
        # Annualized funding (3 payments per day × 365 days)
        features['funding_annualized'] = funding_rate * 3 * 365 * 100

        # Basis-funding spread: if basis >> funding, arbitrage opportunity
        features['basis_funding_spread'] = features['basis_pct'] * 3 * 365 - features['funding_annualized']

        # Funding prediction: basis often leads funding
        features['funding_predicted'] = features['basis_pct'].shift(1) / 3  # Approx next 8h funding

    # ─────────────────────────────────────────────────────────────────────
    # 6. MEAN REVERSION SIGNALS
    # ─────────────────────────────────────────────────────────────────────
    # Basis tends to mean-revert around fair value
    basis_zscore = (features['basis_pct'] - features['basis_pct'].rolling(20).mean()) / \
                   features['basis_pct'].rolling(20).std()
    features['basis_zscore'] = basis_zscore.clip(-3, 3)

    # Bollinger bands for basis
    basis_sma = features['basis_pct'].rolling(20).mean()
    basis_std = features['basis_pct'].rolling(20).std()
    features['basis_bb_upper'] = basis_sma + 2 * basis_std
    features['basis_bb_lower'] = basis_sma - 2 * basis_std
    features['basis_bb_signal'] = (features['basis_pct'] - basis_sma) / (2 * basis_std)

    # ─────────────────────────────────────────────────────────────────────
    # 7. CASH-AND-CARRY ARBITRAGE DETECTION
    # ─────────────────────────────────────────────────────────────────────
    if days_to_expiry is not None:
        # Arbitrage profit = |mispricing| - transaction_costs
        # Conservative estimate: 0.1% round-trip for execution
        TRANSACTION_COST_PCT = 0.1

        features['arb_profit_pct'] = features['mispricing_pct'].abs() - TRANSACTION_COST_PCT
        features['arb_opportunity'] = (features['arb_profit_pct'] > 0).astype(float)

        # Direction: positive mispricing = sell futures, buy spot
        features['arb_direction'] = np.sign(features['mispricing_pct'])

    return features


def calculate_term_structure_features(
    front_month_price: pd.Series,
    back_month_price: pd.Series,
    front_dte: pd.Series,
    back_dte: pd.Series,
) -> pd.DataFrame:
    """
    Calculate term structure features for calendar spread trading.

    Used for:
    - Index futures (ES, NQ) quarterly rolls
    - Commodity futures (GC, CL) monthly rolls
    - VIX futures term structure

    Args:
        front_month_price: Near-term contract price
        back_month_price: Next-term contract price
        front_dte: Days to expiry (front)
        back_dte: Days to expiry (back)

    Returns:
        DataFrame with term structure features
    """
    features = pd.DataFrame(index=front_month_price.index)

    # Calendar spread
    spread = back_month_price - front_month_price
    features['calendar_spread'] = spread
    features['calendar_spread_pct'] = spread / front_month_price * 100

    # Annualized roll
    days_between = back_dte - front_dte
    features['roll_annualized'] = (spread / front_month_price) * (365 / days_between.replace(0, np.nan)) * 100

    # Term structure shape
    features['is_contango'] = (spread > 0).astype(float)
    features['is_backwardation'] = (spread < 0).astype(float)

    # Spread volatility
    features['spread_volatility'] = features['calendar_spread_pct'].rolling(20).std()

    # Z-score for mean reversion
    spread_mean = features['calendar_spread_pct'].rolling(60).mean()
    spread_std = features['calendar_spread_pct'].rolling(60).std()
    features['spread_zscore'] = ((features['calendar_spread_pct'] - spread_mean) / spread_std).clip(-3, 3)

    return features


def calculate_liquidation_features(
    liquidation_volume: pd.Series,
    total_volume: pd.Series,
    price: pd.Series,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Calculate liquidation-related features.

    Features:
    - liq_volume_ratio: Liquidation as % of total volume
    - liq_cascade_risk: Indicator of cascade risk
    - liq_price_impact: Price impact from liquidations
    """
    features = pd.DataFrame(index=liquidation_volume.index)

    features['liq_volume_ratio'] = liquidation_volume / total_volume.replace(0, np.nan)

    # Cascade risk: high recent liquidations + volatility
    features['liq_cascade_risk'] = (
        features['liq_volume_ratio'].rolling(lookback).mean() *
        price.pct_change().rolling(lookback).std()
    )

    return features
```

### 7.2 Data Pipeline Updates

```python
# Updates to data_loader_multi_asset.py

def load_futures_data(
    paths: List[str],
    funding_paths: Optional[List[str]] = None,
    mark_price_paths: Optional[List[str]] = None,
    liquidation_paths: Optional[List[str]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int]]:
    """
    Load futures data with all components.

    Merges:
    - OHLCV bars (last price)
    - Mark price bars
    - Funding rate history
    - Liquidation data
    """
    # Load base OHLCV
    base_frames = load_parquet_files(paths)

    # Merge funding rates
    if funding_paths:
        funding_frames = load_parquet_files(funding_paths)
        base_frames = merge_funding_data(base_frames, funding_frames)

    # Merge mark price
    if mark_price_paths:
        mark_frames = load_parquet_files(mark_price_paths)
        base_frames = merge_mark_price(base_frames, mark_frames)

    # Merge liquidations
    if liquidation_paths:
        liq_frames = load_parquet_files(liquidation_paths)
        base_frames = merge_liquidation_data(base_frames, liq_frames)

    return base_frames, compute_obs_shapes(base_frames)
```

### 7.3 Data Download Script

```python
# NEW FILE: scripts/download_futures_data.py
"""
Download Binance Futures historical data.

Downloads:
- OHLCV candlesticks
- Mark price candlesticks
- Funding rate history
- Liquidation events (aggregated)
"""

import argparse
from binance_public import BinancePublicClient
from ingest_funding_mark import _fetch_all_funding, _fetch_all_mark

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--out-dir", default="data/futures")
    args = parser.parse_args()

    client = BinancePublicClient()

    for symbol in args.symbols:
        # Download OHLCV
        download_ohlcv(client, symbol, args)

        # Download mark price
        download_mark_price(client, symbol, args)

        # Download funding
        download_funding(client, symbol, args)

if __name__ == "__main__":
    main()
```

### Tests for Phase 7

```python
# NEW FILE: tests/test_futures_features.py

class TestFundingFeatures:
    """Funding feature tests."""

    def test_funding_cumulative_24h(self):
        """Correctly sums 3 funding periods."""
        pass

    def test_funding_direction_consistency(self):
        """Direction feature in [-1, 1]."""
        pass

class TestOpenInterestFeatures:
    """OI feature tests."""

    def test_oi_price_divergence(self):
        """Divergence shows bullish/bearish signal."""
        pass

class TestBasisFeatures:
    """Basis feature tests."""

    def test_annualized_basis(self):
        """Correctly annualizes basis."""
        pass
```

### Deliverables Phase 7

- [ ] `futures_features.py` - Feature engineering
- [ ] Updated `data_loader_multi_asset.py`
- [ ] `scripts/download_futures_data.py`
- [ ] `tests/test_futures_features.py` (50+ tests)

---

## 📦 PHASE 8: TRAINING INTEGRATION

### Цели

- Интеграция futures в training pipeline
- Position sizing с leverage
- Reward shaping для futures

### 8.1 Futures Trading Environment

```python
# NEW FILE: wrappers/futures_env.py
"""
Futures trading environment wrapper.

Extends TradingEnv with:
- Leverage control action
- Margin tracking
- Funding payment integration
- Liquidation handling
"""

from wrappers.base import TradingEnvWrapper
from core_futures import FuturesPosition, MarginMode

class FuturesTradingEnv(TradingEnvWrapper):
    """
    Futures environment with leverage and margin.

    Action space:
    - Original: position size [-1, 1]
    - Extended: [position_size, leverage_target]

    Observation space:
    - Original features
    - + margin_ratio
    - + funding_rate
    - + liquidation_distance
    """

    def __init__(
        self,
        env,
        initial_leverage: int = 10,
        max_leverage: int = 50,
        margin_mode: MarginMode = MarginMode.CROSS,
        funding_tracker: Optional[FundingRateTracker] = None,
        margin_calculator: Optional[MarginCalculator] = None,
    ):
        super().__init__(env)
        self._leverage = initial_leverage
        self._max_leverage = max_leverage
        self._margin_mode = margin_mode
        self._funding_tracker = funding_tracker
        self._margin_calculator = margin_calculator

        # Position tracking
        self._position: Optional[FuturesPosition] = None
        self._realized_funding: Decimal = Decimal("0")

    def step(self, action):
        """Execute step with futures mechanics."""
        # Parse action (position_size, optional leverage)
        position_target = action[0] if len(action) > 0 else action
        leverage_target = action[1] if len(action) > 1 else self._leverage

        # Execute base environment step
        obs, reward, terminated, truncated, info = self.env.step(position_target)

        # Apply funding if due
        funding_payment = self._apply_funding(info.get("timestamp_ms", 0))

        # Check liquidation
        if self._check_liquidation(info.get("mark_price")):
            terminated = True
            reward = self._liquidation_penalty
            info["liquidated"] = True

        # Augment observation
        obs = self._augment_obs(obs, info)

        # Augment reward with funding
        reward = reward + float(funding_payment)

        info["funding_payment"] = float(funding_payment)
        info["margin_ratio"] = self._get_margin_ratio(info.get("mark_price"))

        return obs, reward, terminated, truncated, info

    def _apply_funding(self, timestamp_ms: int) -> Decimal:
        """Apply funding payment if due."""
        if self._position is None or self._funding_tracker is None:
            return Decimal("0")

        # Check if funding time
        if not self._is_funding_time(timestamp_ms):
            return Decimal("0")

        funding_rate = self._get_current_funding_rate()
        mark_price = self._get_mark_price()

        payment = self._funding_tracker.calculate_funding_payment(
            self._position, funding_rate, mark_price, timestamp_ms
        )

        self._realized_funding += payment.payment_amount
        return payment.payment_amount
```

### 8.2 Training Pipeline Updates

```python
# Updates to train_model_multi_patch.py

def create_futures_env(cfg: Config) -> FuturesTradingEnv:
    """Create futures environment."""
    # Load data
    base_frames, obs_shapes = load_futures_data(
        paths=cfg.data.paths,
        funding_paths=cfg.data.funding_paths,
        mark_price_paths=cfg.data.mark_price_paths,
    )

    # Create base env
    base_env = TradingEnv(
        frames=base_frames,
        config=cfg,
        asset_class="crypto_futures",
    )

    # Wrap with futures mechanics
    env = FuturesTradingEnv(
        base_env,
        initial_leverage=cfg.futures.initial_leverage,
        max_leverage=cfg.futures.max_leverage,
        margin_mode=MarginMode(cfg.futures.margin_mode),
    )

    return env
```

### 8.3 Futures Training Config

```yaml
# NEW FILE: configs/config_train_futures.yaml
mode: train
asset_class: crypto_futures
data_vendor: binance

data:
  paths:
    - "data/futures/*.parquet"
  funding_paths:
    - "data/futures/*_funding.parquet"
  mark_price_paths:
    - "data/futures/*_mark_*.parquet"
  timeframe: "4h"

futures:
  initial_leverage: 10
  max_leverage: 50
  margin_mode: cross
  include_funding_in_reward: true
  liquidation_penalty: -10.0

env:
  session:
    calendar: crypto_24x7

fees:
  structure: maker_taker
  maker_bps: 2.0
  taker_bps: 4.0

slippage:
  profile: crypto_futures
  k: 0.09
```

### 8.4 Unified Futures Config Structure

The following template provides a **unified configuration schema** for ALL futures types.
This enables consistent configuration across crypto perpetuals, index futures, commodities, and currency futures.

```yaml
# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED FUTURES CONFIGURATION TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════════
# NEW FILE: configs/config_futures_unified.yaml
#
# This is the master template for futures trading configuration.
# Supports: crypto_perp, crypto_quarterly, index, commodity, currency futures
#
# Usage:
#   python script_futures_backtest.py --config configs/config_futures_unified.yaml
#   python script_futures_live.py --config configs/config_futures_unified.yaml
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# CORE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
mode: backtest                    # train, backtest, live, eval
futures_type: crypto_perp         # crypto_perp, crypto_quarterly, index, commodity, currency

# Vendor/exchange selection
vendor:
  primary: binance                # binance, interactive_brokers, tradovate
  fallback: null                  # Optional fallback for data

# ─────────────────────────────────────────────────────────────────────────────
# CONTRACT SPECIFICATION (Required for non-crypto)
# ─────────────────────────────────────────────────────────────────────────────
contract:
  symbol: BTCUSDT                 # Trading symbol
  exchange: binance               # Exchange code
  underlying: BTC                 # Underlying asset
  quote_currency: USDT            # Quote currency
  multiplier: 1                   # Contract multiplier (1 for crypto, $50 for ES)
  tick_size: 0.10                 # Minimum price increment
  min_qty: 0.001                  # Minimum order size
  max_qty: 1000                   # Maximum order size
  expiry: null                    # ISO date for quarterly (null for perpetual)
  settlement: cash                # cash, physical

# ─────────────────────────────────────────────────────────────────────────────
# LEVERAGE & MARGIN
# ─────────────────────────────────────────────────────────────────────────────
margin:
  mode: cross                     # cross, isolated
  initial_leverage: 10            # Starting leverage (1-125)
  max_leverage: 50                # Hard cap (exchange limit)
  maintenance_margin_rate: 0.004  # 0.4% for low leverage (tiered for crypto)

  # Tiered brackets (Binance USDT-M style) - crypto only
  brackets:
    - notional_cap: 50000
      max_leverage: 125
      maint_margin_rate: 0.004
    - notional_cap: 250000
      max_leverage: 100
      maint_margin_rate: 0.005
    - notional_cap: 1000000
      max_leverage: 50
      maint_margin_rate: 0.01
    - notional_cap: 10000000
      max_leverage: 20
      maint_margin_rate: 0.025
    - notional_cap: null          # Unlimited
      max_leverage: 10
      maint_margin_rate: 0.05

  # SPAN margin settings (CME only)
  span:
    enabled: false                # Use SPAN approximation
    scanning_range: 0.12          # 12% price range
    volatility_adjustment: true
    inter_month_spread_credit: 0.70  # 70% credit for calendar spreads

# ─────────────────────────────────────────────────────────────────────────────
# FUNDING (Perpetuals only)
# ─────────────────────────────────────────────────────────────────────────────
funding:
  enabled: true                   # Enable funding rate simulation
  interval_hours: 8               # Funding interval (8 for Binance)
  times_utc: ["00:00", "08:00", "16:00"]  # Settlement times
  include_in_reward: true         # Include funding P&L in reward
  clamp_rate_bps: 75              # Max funding rate (0.75%)
  pro_rata_settlement: true       # Calculate pro-rata for partial period

  # Funding risk limits
  limits:
    max_cumulative_24h_bps: 150   # Max 1.5% daily funding exposure
    skip_when_rate_exceeds_bps: 50  # Avoid high funding positions

# ─────────────────────────────────────────────────────────────────────────────
# SETTLEMENT (Quarterly/CME only)
# ─────────────────────────────────────────────────────────────────────────────
settlement:
  enabled: false                  # Enable daily settlement (CME)
  # CME uses 14:30 CT (Central Time) = 15:30 ET for equity index futures
  # Note: Some products have different settlement times (e.g., agricultural)
  # Reference: https://www.cmegroup.com/trading/equity-index/us-index/e-mini-sandp500.html
  time_et: "15:30"                # Daily settlement time (14:30 CT)
  time_ct: "14:30"                # Central Time (CME native timezone)
  variation_margin: true          # Daily P&L settlement
  auto_roll:
    enabled: true                 # Auto-roll before expiry
    days_before_expiry: 8         # Roll 8 days before
    max_roll_cost_bps: 50         # Abort roll if spread > 50bps

# ─────────────────────────────────────────────────────────────────────────────
# LIQUIDATION
# ─────────────────────────────────────────────────────────────────────────────
liquidation:
  penalty_reward: -10.0           # Reward penalty on liquidation
  simulate_cascade: true          # Simulate liquidation cascades
  insurance_fund_deduction: true  # Deduct from insurance fund (crypto)
  adl_simulation: true            # Auto-deleveraging simulation (crypto)

  # Cross-margin priority (when multiple positions)
  cross_margin_priority: highest_loss  # highest_loss, lowest_ratio, oldest, largest

# ─────────────────────────────────────────────────────────────────────────────
# TRADING HOURS & SESSIONS
# ─────────────────────────────────────────────────────────────────────────────
session:
  calendar: crypto_24x7           # crypto_24x7, cme_futures, forex_24x5

  # CME-specific (index, commodity, currency futures)
  cme:
    regular_hours:
      start_et: "09:30"
      end_et: "16:00"
    globex_hours:
      start_et: "18:00"           # Sunday 6pm
      end_et: "17:00"             # Friday 5pm
    maintenance_break:
      start_et: "16:15"
      end_et: "16:30"
    holidays: us_federal           # Holiday calendar

  # Circuit breakers (CME equity index futures)
  circuit_breakers:
    enabled: true
    levels:
      - threshold_pct: -7
        halt_minutes: 15
        applies_to: rth            # Regular trading hours only
      - threshold_pct: -13
        halt_minutes: 15
        applies_to: rth
      - threshold_pct: -20
        halt_minutes: null         # Market closed for day
        applies_to: rth

    # Overnight limits
    overnight_limit_pct: 5         # ±5% from settlement price
    velocity_logic:
      enabled: true
      threshold_ticks: 12          # ES: 12 ticks in short window

# ─────────────────────────────────────────────────────────────────────────────
# FEES
# ─────────────────────────────────────────────────────────────────────────────
fees:
  structure: maker_taker          # maker_taker, flat, exchange_fees

  # Crypto (Binance)
  crypto:
    maker_bps: 2.0
    taker_bps: 4.0
    use_bnb_discount: true

  # CME via IB
  cme:
    commission_per_contract: 2.25  # USD per contract (one-way)
    exchange_fee_per_contract: 1.25
    nfa_fee_per_contract: 0.02
    clearing_fee_per_contract: 0.10

  # Regulatory (CME only)
  regulatory:
    enabled: true
    sec_fee_per_million: 27.80     # Not applicable to futures, but for consistency

# ─────────────────────────────────────────────────────────────────────────────
# SLIPPAGE & EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
execution:
  level: L2                       # L2 (parametric), L3 (LOB simulation)

  slippage:
    profile: crypto_futures       # crypto_futures, index_futures, commodity_futures
    impact_coef_base: 0.09        # Almgren-Chriss k coefficient
    default_spread_bps: 4.0
    min_slippage_bps: 0.5
    max_slippage_bps: 500.0

    # Futures-specific adjustments
    funding_stress_sensitivity: 8.0   # Slippage increase with high funding
    liquidation_cascade_multiplier: 1.5  # Extra slippage during liquidations
    oi_imbalance_sensitivity: 0.3     # Open interest imbalance impact

  # L3 LOB settings (if execution.level == "L3")
  lob:
    latency_profile: institutional    # colocated, institutional, retail
    queue_position_method: mbo        # mbo, mbp
    impact_model: almgren_chriss      # kyle, almgren_chriss, gatheral
    fill_probability_model: queue_reactive

# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
risk:
  # Position limits
  max_position_usd: 1000000       # Max notional per symbol
  max_open_positions: 5           # Max concurrent positions
  max_leverage_used: 0.8          # Max 80% of available leverage

  # Drawdown limits
  max_drawdown_pct: 15            # Max 15% drawdown
  daily_loss_limit_usd: 10000     # Max daily loss
  trailing_stop_pct: null         # Optional trailing stop

  # Funding risk (crypto)
  max_funding_exposure_24h_pct: 2.0  # Max 2% funding exposure per day

  # Concentration
  max_concentration_pct: 50       # Max 50% in single position

  # Kill switch
  kill_switch:
    enabled: true
    triggers:
      - consecutive_losses: 5
      - drawdown_pct: 10
      - daily_loss_usd: 5000

# ─────────────────────────────────────────────────────────────────────────────
# DATA SOURCES
# ─────────────────────────────────────────────────────────────────────────────
data:
  # OHLCV bars
  paths:
    - "data/futures/*.parquet"
  timeframe: "4h"

  # Funding rate history (crypto)
  funding_paths:
    - "data/futures/*_funding.parquet"

  # Mark price (for liquidation simulation)
  mark_price_paths:
    - "data/futures/*_mark_*.parquet"

  # Open interest (optional)
  open_interest_paths:
    - "data/futures/*_oi.parquet"

  # Liquidations (optional, for cascade simulation)
  liquidation_paths:
    - "data/futures/*_liquidations.parquet"

  # Data validation
  validation:
    check_gaps: true
    max_gap_bars: 2
    check_volume_zeros: true
    min_history_bars: 1000

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES (Futures-specific)
# ─────────────────────────────────────────────────────────────────────────────
features:
  # Enable futures-specific features
  include_funding_features: true
  include_basis_features: true
  include_oi_features: true
  include_liquidation_features: true
  include_term_structure: false   # For quarterly/CME

  # Feature configuration
  funding_lookback: 8             # 8 funding periods (~2.67 days)
  oi_lookback: 20                 # 20 bars
  basis_lookback: 20

# ─────────────────────────────────────────────────────────────────────────────
# MODEL (Training mode)
# ─────────────────────────────────────────────────────────────────────────────
model:
  algo: ppo
  policy: RecurrentActorCriticPolicy
  path: "models/futures_ppo.zip"

  params:
    learning_rate: 0.0001
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    ent_coef: 0.001
    vf_coef: 0.5
    max_grad_norm: 0.5
    use_twin_critics: true
    num_quantiles: 21
    cvar_alpha: 0.05

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING & MONITORING
# ─────────────────────────────────────────────────────────────────────────────
logging:
  level: INFO
  tensorboard: true
  tensorboard_dir: "runs/futures"

  # Futures-specific metrics
  track_funding_pnl: true
  track_liquidation_distance: true
  track_margin_usage: true
```

### 8.5 Feature Flags for Gradual Rollout

Feature flags enable safe, incremental deployment of futures functionality.

```python
# NEW FILE: services/futures_feature_flags.py
"""
Feature flags for gradual futures integration rollout.

Enables:
1. Shadow mode testing (run parallel to production without affecting positions)
2. Canary deployment (small % of traffic)
3. Kill switch for rapid rollback
4. A/B testing of execution algorithms

References:
- Martin Fowler: Feature Toggles (Feature Flags)
- LaunchDarkly: Best Practices for Feature Flags
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Callable
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class RolloutStage(str, Enum):
    """Deployment stage for futures features."""
    DISABLED = "disabled"           # Feature completely off
    SHADOW = "shadow"               # Run in parallel, don't affect positions
    CANARY = "canary"               # Small % of traffic (configurable)
    PRODUCTION = "production"       # Full rollout


class FuturesFeature(str, Enum):
    """Individual futures features that can be toggled."""
    # Core features
    PERPETUAL_TRADING = "perpetual_trading"
    QUARTERLY_TRADING = "quarterly_trading"
    INDEX_FUTURES = "index_futures"
    COMMODITY_FUTURES = "commodity_futures"
    CURRENCY_FUTURES = "currency_futures"

    # Margin & Liquidation
    CROSS_MARGIN = "cross_margin"
    ISOLATED_MARGIN = "isolated_margin"
    LIQUIDATION_SIMULATION = "liquidation_simulation"
    ADL_SIMULATION = "adl_simulation"

    # Funding
    FUNDING_RATE_TRACKING = "funding_rate_tracking"
    FUNDING_IN_REWARD = "funding_in_reward"
    PRO_RATA_FUNDING = "pro_rata_funding"

    # Execution
    L2_EXECUTION = "l2_execution"
    L3_EXECUTION = "l3_execution"
    LIQUIDATION_CASCADE_SLIPPAGE = "liquidation_cascade_slippage"

    # Risk
    FUTURES_RISK_GUARDS = "futures_risk_guards"
    LEVERAGE_GUARD = "leverage_guard"
    FUNDING_EXPOSURE_GUARD = "funding_exposure_guard"

    # Data
    FUTURES_FEATURES_PIPELINE = "futures_features_pipeline"
    TERM_STRUCTURE_FEATURES = "term_structure_features"
    BASIS_TRADING_FEATURES = "basis_trading_features"


@dataclass
class FeatureConfig:
    """Configuration for a single feature flag."""
    stage: RolloutStage = RolloutStage.DISABLED
    canary_percentage: float = 0.0      # 0-100, used when stage=CANARY
    allowed_symbols: Optional[list] = None  # If set, only these symbols
    allowed_accounts: Optional[list] = None  # If set, only these accounts
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FuturesFeatureFlags:
    """
    Centralized feature flag management for futures integration.

    Usage:
        flags = FuturesFeatureFlags.load("configs/feature_flags.yaml")

        if flags.is_enabled(FuturesFeature.PERPETUAL_TRADING):
            # Execute perpetual trading logic
            pass

        if flags.should_execute(FuturesFeature.L3_EXECUTION, symbol="BTCUSDT"):
            # Use L3 execution for this symbol
            pass
    """

    features: Dict[FuturesFeature, FeatureConfig] = field(default_factory=dict)
    global_kill_switch: bool = False
    environment: str = "development"  # development, staging, production

    def __post_init__(self):
        # Initialize all features as disabled by default
        for feature in FuturesFeature:
            if feature not in self.features:
                self.features[feature] = FeatureConfig()

    @classmethod
    def load(cls, path: str) -> "FuturesFeatureFlags":
        """Load feature flags from YAML/JSON file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Feature flags file not found: {path}, using defaults")
            return cls()

        with open(path) as f:
            if path.suffix in (".yaml", ".yml"):
                import yaml
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "FuturesFeatureFlags":
        """Parse feature flags from dictionary."""
        flags = cls(
            global_kill_switch=data.get("global_kill_switch", False),
            environment=data.get("environment", "development"),
        )

        for feature_name, config in data.get("features", {}).items():
            try:
                feature = FuturesFeature(feature_name)
                flags.features[feature] = FeatureConfig(
                    stage=RolloutStage(config.get("stage", "disabled")),
                    canary_percentage=config.get("canary_percentage", 0.0),
                    allowed_symbols=config.get("allowed_symbols"),
                    allowed_accounts=config.get("allowed_accounts"),
                    metadata=config.get("metadata", {}),
                )
            except ValueError:
                logger.warning(f"Unknown feature flag: {feature_name}")

        return flags

    def is_enabled(self, feature: FuturesFeature) -> bool:
        """Check if feature is enabled (any stage except DISABLED)."""
        if self.global_kill_switch:
            return False
        return self.features[feature].stage != RolloutStage.DISABLED

    def is_production(self, feature: FuturesFeature) -> bool:
        """Check if feature is in full production rollout."""
        if self.global_kill_switch:
            return False
        return self.features[feature].stage == RolloutStage.PRODUCTION

    def is_shadow_mode(self, feature: FuturesFeature) -> bool:
        """Check if feature is in shadow mode (run but don't affect positions)."""
        return self.features[feature].stage == RolloutStage.SHADOW

    def should_execute(
        self,
        feature: FuturesFeature,
        symbol: Optional[str] = None,
        account_id: Optional[str] = None,
        random_value: Optional[float] = None,  # 0-100 for canary selection
    ) -> bool:
        """
        Determine if feature should execute for given context.

        Args:
            feature: Feature to check
            symbol: Trading symbol (for symbol-specific rollout)
            account_id: Account ID (for account-specific rollout)
            random_value: Random value 0-100 for canary percentage check

        Returns:
            True if feature should execute
        """
        if self.global_kill_switch:
            return False

        config = self.features[feature]

        if config.stage == RolloutStage.DISABLED:
            return False

        if config.stage == RolloutStage.SHADOW:
            # Shadow mode: execute but caller should not affect positions
            return True

        if config.stage == RolloutStage.CANARY:
            # Check canary criteria
            if config.allowed_symbols and symbol not in config.allowed_symbols:
                return False
            if config.allowed_accounts and account_id not in config.allowed_accounts:
                return False
            if random_value is not None:
                return random_value < config.canary_percentage
            return True

        if config.stage == RolloutStage.PRODUCTION:
            return True

        return False

    def get_stage(self, feature: FuturesFeature) -> RolloutStage:
        """Get current rollout stage for feature."""
        return self.features[feature].stage

    def set_stage(self, feature: FuturesFeature, stage: RolloutStage) -> None:
        """Set rollout stage for feature (runtime update)."""
        logger.info(f"Feature {feature.value} stage changed: "
                   f"{self.features[feature].stage.value} -> {stage.value}")
        self.features[feature].stage = stage

    def enable_kill_switch(self) -> None:
        """Emergency kill switch - disable all features."""
        logger.critical("GLOBAL KILL SWITCH ACTIVATED - All futures features disabled")
        self.global_kill_switch = True

    def disable_kill_switch(self) -> None:
        """Re-enable features after kill switch."""
        logger.warning("Global kill switch disabled - Features restored to configured state")
        self.global_kill_switch = False

    def to_dict(self) -> Dict[str, Any]:
        """Export current state as dictionary."""
        return {
            "global_kill_switch": self.global_kill_switch,
            "environment": self.environment,
            "features": {
                f.value: {
                    "stage": self.features[f].stage.value,
                    "canary_percentage": self.features[f].canary_percentage,
                    "allowed_symbols": self.features[f].allowed_symbols,
                    "allowed_accounts": self.features[f].allowed_accounts,
                }
                for f in FuturesFeature
            }
        }

    def save(self, path: str) -> None:
        """Save current state to file."""
        path = Path(path)
        data = self.to_dict()

        with open(path, "w") as f:
            if path.suffix in (".yaml", ".yml"):
                import yaml
                yaml.dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# HELPER DECORATORS
# ═══════════════════════════════════════════════════════════════════════════

def feature_flag(feature: FuturesFeature, fallback: Callable = None):
    """
    Decorator to gate function execution by feature flag.

    Usage:
        @feature_flag(FuturesFeature.L3_EXECUTION)
        def execute_l3(order, market):
            # Only runs if L3_EXECUTION is enabled
            ...

        @feature_flag(FuturesFeature.ADL_SIMULATION, fallback=lambda: None)
        def simulate_adl(positions):
            # Falls back to no-op if disabled
            ...
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Get flags from global or context
            flags = kwargs.pop("_feature_flags", None) or _get_global_flags()

            if flags.should_execute(feature):
                if flags.is_shadow_mode(feature):
                    # Execute but log shadow mode
                    logger.debug(f"[SHADOW] Executing {func.__name__} for {feature.value}")
                return func(*args, **kwargs)
            else:
                if fallback:
                    return fallback()
                return None

        return wrapper
    return decorator


# Global flags instance (set during initialization)
_global_flags: Optional[FuturesFeatureFlags] = None


def init_feature_flags(path: str) -> FuturesFeatureFlags:
    """Initialize global feature flags from file."""
    global _global_flags
    _global_flags = FuturesFeatureFlags.load(path)
    return _global_flags


def _get_global_flags() -> FuturesFeatureFlags:
    """Get global flags instance, creating default if not initialized."""
    global _global_flags
    if _global_flags is None:
        _global_flags = FuturesFeatureFlags()
    return _global_flags
```

**Feature Flags Configuration File:**

```yaml
# NEW FILE: configs/feature_flags_futures.yaml
# Feature flags for gradual futures rollout

global_kill_switch: false
environment: staging              # development, staging, production

features:
  # ─────────────────────────────────────────────────────────────────
  # CORE TRADING (Enable first)
  # ─────────────────────────────────────────────────────────────────
  perpetual_trading:
    stage: production             # Fully rolled out
    canary_percentage: 100

  quarterly_trading:
    stage: canary                 # Limited rollout
    canary_percentage: 25
    allowed_symbols:
      - BTCUSDT_QUARTERLY
      - ETHUSDT_QUARTERLY

  index_futures:
    stage: shadow                 # Testing only
    allowed_symbols:
      - ES
      - NQ

  commodity_futures:
    stage: disabled               # Not yet ready

  currency_futures:
    stage: disabled               # Not yet ready

  # ─────────────────────────────────────────────────────────────────
  # MARGIN & LIQUIDATION
  # ─────────────────────────────────────────────────────────────────
  cross_margin:
    stage: production

  isolated_margin:
    stage: production

  liquidation_simulation:
    stage: production

  adl_simulation:
    stage: canary
    canary_percentage: 50         # 50% of positions

  # ─────────────────────────────────────────────────────────────────
  # FUNDING
  # ─────────────────────────────────────────────────────────────────
  funding_rate_tracking:
    stage: production

  funding_in_reward:
    stage: production

  pro_rata_funding:
    stage: canary                 # New feature, testing
    canary_percentage: 20

  # ─────────────────────────────────────────────────────────────────
  # EXECUTION
  # ─────────────────────────────────────────────────────────────────
  l2_execution:
    stage: production

  l3_execution:
    stage: shadow                 # Running in parallel for comparison
    allowed_symbols:
      - BTCUSDT
      - ETHUSDT

  liquidation_cascade_slippage:
    stage: canary
    canary_percentage: 30

  # ─────────────────────────────────────────────────────────────────
  # RISK
  # ─────────────────────────────────────────────────────────────────
  futures_risk_guards:
    stage: production

  leverage_guard:
    stage: production

  funding_exposure_guard:
    stage: production

  # ─────────────────────────────────────────────────────────────────
  # DATA & FEATURES
  # ─────────────────────────────────────────────────────────────────
  futures_features_pipeline:
    stage: production

  term_structure_features:
    stage: canary
    canary_percentage: 50

  basis_trading_features:
    stage: production
```

**Recommended Rollout Order:**

| Phase | Features | Stage | Duration |
|-------|----------|-------|----------|
| 1 | Core margin, fees, basic execution | Production | Week 1 |
| 2 | Funding tracking, risk guards | Production | Week 2 |
| 3 | Liquidation simulation | Canary 50% | Week 3 |
| 4 | ADL, cascade slippage | Canary 25% | Week 4 |
| 5 | L3 execution | Shadow | Week 5 |
| 6 | L3 execution | Canary 10% | Week 6 |
| 7 | Index/Commodity futures | Shadow | Week 7-8 |
| 8 | Full production | Production | Week 9+ |

### Tests for Phase 8

```python
# NEW FILE: tests/test_futures_training.py

class TestFuturesTradingEnv:
    """Futures env tests."""

    def test_funding_applied_at_correct_times(self):
        """Funding only applied at 00:00, 08:00, 16:00 UTC."""
        pass

    def test_liquidation_terminates_episode(self):
        """Episode ends on liquidation."""
        pass

    def test_leverage_scaling_position_size(self):
        """Leverage correctly scales position."""
        pass

class TestFuturesTraining:
    """Training integration tests."""

    def test_training_runs_without_error(self):
        """Basic training loop completes."""
        pass

    def test_model_learns_funding_avoidance(self):
        """Model learns to avoid high funding positions."""
        pass
```

### Deliverables Phase 8

- [ ] `wrappers/futures_env.py` - Futures environment
- [ ] Updated `train_model_multi_patch.py`
- [ ] `configs/config_train_futures.yaml`
- [ ] `tests/test_futures_training.py` (60+ tests)

---

## 📦 PHASE 9: LIVE TRADING

### Цели

- Live trading с futures
- Position synchronization
- Real-time funding tracking

### 9.1 Futures Live Script

```python
# NEW FILE: script_futures_live.py
"""
Futures live trading entry point.

Features:
- Position sync with exchange
- Real-time funding tracking
- Margin monitoring
- Auto-deleveraging detection
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--leverage", type=int, default=10)
    parser.add_argument("--margin-mode", choices=["cross", "isolated"], default="cross")
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Create adapters
    market_data = BinanceFuturesMarketDataAdapter(cfg.binance)
    order_exec = BinanceFuturesOrderExecutionAdapter(cfg.binance)

    # Set leverage and margin mode
    order_exec.set_leverage(cfg.symbol, args.leverage)
    order_exec.set_margin_mode(cfg.symbol, MarginMode(args.margin_mode))

    # Create live runner
    runner = FuturesLiveRunner(
        market_data=market_data,
        order_execution=order_exec,
        model=load_model(cfg.model_path),
        risk_guards=create_futures_risk_guards(cfg),
    )

    # Run
    runner.run()
```

### 9.2 Position Sync

```python
# NEW FILE: services/futures_position_sync.py
"""
Futures position synchronization.

Syncs local state with exchange state.
"""

class FuturesPositionSynchronizer:
    """
    Synchronizes futures positions with exchange.

    Handles:
    - Position quantity mismatches
    - Entry price drift (due to partial fills)
    - Leverage changes
    - ADL events
    """

    def __init__(
        self,
        order_execution: BinanceFuturesOrderExecutionAdapter,
        sync_interval_sec: float = 10.0,
    ):
        self._adapter = order_execution
        self._interval = sync_interval_sec
        self._local_positions: Dict[str, FuturesPosition] = {}

    def sync(self) -> List[PositionDiff]:
        """
        Synchronize with exchange.

        Returns:
            List of position differences found
        """
        exchange_positions = {
            p.symbol: p for p in self._adapter.get_all_positions()
        }

        diffs = []

        for symbol, local_pos in self._local_positions.items():
            exchange_pos = exchange_positions.get(symbol)

            if exchange_pos is None:
                # Position closed externally (liquidation, ADL)
                diffs.append(PositionDiff(
                    symbol=symbol,
                    diff_type="closed_externally",
                    local_qty=local_pos.qty,
                    exchange_qty=Decimal("0"),
                ))
            elif local_pos.qty != exchange_pos.qty:
                # Quantity mismatch
                diffs.append(PositionDiff(
                    symbol=symbol,
                    diff_type="qty_mismatch",
                    local_qty=local_pos.qty,
                    exchange_qty=exchange_pos.qty,
                ))

        return diffs
```

### Tests for Phase 9

```python
# NEW FILE: tests/test_futures_live_trading.py

class TestFuturesLiveRunner:
    """Live trading tests."""

    def test_position_sync_detects_external_close(self):
        """Detects when position closed by exchange (liquidation)."""
        pass

    def test_funding_payment_handling(self):
        """Real-time funding payments tracked."""
        pass
```

### Deliverables Phase 9

- [x] `script_futures_live.py` - Live trading script
- [x] `services/futures_position_sync.py` - Position sync service
- [x] `services/futures_live_runner.py` - Live trading runner
- [x] `services/futures_funding_tracker.py` - Funding rate tracker
- [x] `services/futures_margin_monitor.py` - Margin monitoring service
- [x] `configs/config_live_futures.yaml` - Live trading configuration
- [x] `tests/test_futures_live_trading.py` (81 tests)
- [x] `tests/test_futures_position_sync.py` (56 tests)
- [x] `tests/test_futures_margin_monitor.py` (49 tests)

---

## 📦 PHASE 10: TESTING & VALIDATION

### Цели

- Comprehensive test suite
- Validation metrics
- Documentation

### 10.1 Validation Metrics

```python
# NEW FILE: tests/test_futures_validation.py
"""
Futures simulation validation.

Target metrics:
- Fill rate: >95%
- Slippage error: <3 bps vs historical
- Funding rate accuracy: >99%
- Liquidation timing: <1 bar delay
- Margin calculation error: <0.1%
"""

class TestFuturesValidation:
    """Validation against historical data."""

    def test_fill_rate(self):
        """Fill rate >95% for limit orders within spread."""
        pass

    def test_slippage_accuracy(self):
        """Simulated slippage within 3 bps of historical."""
        pass

    def test_funding_rate_accuracy(self):
        """Funding payments match historical."""
        pass

    def test_liquidation_timing(self):
        """Liquidation triggered within 1 bar of mark price breach."""
        pass

    def test_margin_calculation_accuracy(self):
        """Margin calculations match exchange."""
        pass
```

### 10.2 Backward Compatibility Tests

```python
# NEW FILE: tests/test_futures_backward_compatibility.py
"""
Ensure crypto spot and equity unchanged.
"""

class TestBackwardCompatibility:
    """No regressions in existing functionality."""

    def test_crypto_spot_unchanged(self):
        """Crypto spot behavior unchanged."""
        pass

    def test_equity_unchanged(self):
        """Equity behavior unchanged."""
        pass

    def test_forex_unchanged(self):
        """Forex behavior unchanged."""
        pass
```

### 10.3 Performance Benchmarks

```python
# NEW FILE: benchmarks/bench_futures_simulation.py
"""
Futures simulation performance benchmarks.

Targets:
- L2 execution: <100μs per fill
- L3 execution: <1ms per fill
- Full day simulation: <10 seconds
"""

def benchmark_l2_execution():
    """L2 execution performance."""
    pass

def benchmark_l3_execution():
    """L3 execution performance."""
    pass
```

### 10.4 Documentation

```
docs/futures/
├── overview.md              # Futures integration overview
├── api_reference.md         # API reference
├── configuration.md         # Configuration guide
├── margin_calculation.md    # Margin mechanics
├── funding_rates.md         # Funding rate handling
├── liquidation.md           # Liquidation simulation
├── deployment.md            # Deployment guide
└── migration_guide.md       # Migration from spot
```

### Deliverables Phase 10

- [ ] `tests/test_futures_validation.py` (100+ tests)
- [ ] `tests/test_futures_backward_compatibility.py` (50+ tests)
- [ ] `benchmarks/bench_futures_simulation.py`
- [ ] Documentation suite in `docs/futures/`
- [ ] `FUTURES_INTEGRATION_REPORT.md` - Final validation report

---

## 📝 PLATFORM REFERENCE INTEGRATION

### Раздел для добавления в `PLATFORM_REFERENCE.md`

При завершении интеграции добавить следующую секцию в `PLATFORM_REFERENCE.md`:

```markdown
## 📈 Futures Integration (Phase 11)

### Обзор

Phase 11 добавляет полную поддержку фьючерсов:

1. **Crypto Futures** (Binance USDT-M Perpetual & Quarterly)
2. **Index Futures** (CME: ES, NQ via Interactive Brokers)
3. **Commodity Futures** (COMEX: GC, CL, SI)
4. **Currency Futures** (CME: 6E, 6J, 6B)

### Quick Reference - Futures

| Задача | Где искать | Тесты |
|--------|------------|-------|
| Futures core models | `core_futures.py` | `pytest tests/test_core_futures.py` |
| Crypto margin calculation | `impl_futures_margin.py` | `pytest tests/test_futures_margin.py` |
| CME SPAN margin | `impl_futures_margin.py::CMEMarginCalculator` | `pytest tests/test_futures_span_margin.py` |
| Funding rate tracking | `impl_futures_funding.py` | `pytest tests/test_futures_funding.py` |
| Liquidation simulation | `impl_futures_liquidation.py` | `pytest tests/test_futures_liquidation.py` |
| ADL simulation | `impl_futures_liquidation.py::ADLSimulator` | `pytest tests/test_futures_adl.py` |
| Futures risk guards | `services/futures_risk_guards.py` | `pytest tests/test_futures_risk_guards.py` |
| IB TWS adapter | `adapters/ib/` | `pytest tests/test_ib_adapters.py` |
| Futures features | `futures_features.py` | `pytest tests/test_futures_features.py` |
| Contract rollover | `services/futures_calendar.py` | `pytest tests/test_futures_calendar.py` |
| Margin call notifications | `services/futures_margin_notifications.py` | `pytest tests/test_margin_notifications.py` |

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `core_futures.py` | Core models (FuturesContract, FuturesPosition, MarginRequirement) |
| `impl_futures_margin.py` | Margin calculators (Crypto tiered, CME SPAN) |
| `impl_futures_liquidation.py` | Liquidation engine + ADL simulator |
| `impl_futures_funding.py` | Funding rate tracking and payments |
| `services/futures_risk_guards.py` | Leverage, margin ratio, funding exposure guards |
| `services/futures_position_manager.py` | Position tracking, rollover, P&L |
| `services/futures_calendar.py` | Trading hours, expirations, roll dates |
| `adapters/binance/futures/` | Binance Futures adapters |
| `adapters/ib/` | Interactive Brokers TWS adapters |
| `execution_providers_futures.py` | L2/L3 futures execution |
| `futures_features.py` | Futures-specific features (funding, basis, term structure) |

### CLI Usage

\`\`\`bash
# Download futures data
python scripts/download_futures_data.py --exchange binance --symbols BTCUSDT ETHUSDT

# Training with futures
python train_model_multi_patch.py --config configs/config_train_futures.yaml

# Backtest futures
python script_backtest.py --config configs/config_backtest_futures.yaml

# Live trading futures (paper)
python script_futures_live.py --config configs/config_live_futures.yaml --paper
\`\`\`

### Feature Flags

Futures features controlled via `configs/feature_flags_futures.yaml`:

\`\`\`yaml
perpetual_trading:
  stage: production
index_futures:
  stage: shadow        # Testing mode
\`\`\`

### Конфигурации

| Файл | Назначение |
|------|------------|
| `config_train_futures.yaml` | Training with futures |
| `config_backtest_futures.yaml` | Backtest futures strategies |
| `config_live_futures.yaml` | Live futures trading |
| `feature_flags_futures.yaml` | Feature flag control |
| `futures_contracts.yaml` | Contract specifications |

### Тестирование

\`\`\`bash
# All futures tests
pytest tests/test_futures*.py -v

# Core models
pytest tests/test_core_futures.py -v

# Margin calculation
pytest tests/test_futures_margin.py tests/test_futures_span_margin.py -v

# Liquidation & ADL
pytest tests/test_futures_liquidation.py tests/test_futures_adl.py -v

# Risk guards
pytest tests/test_futures_risk_guards.py -v

# IB adapters
pytest tests/test_ib_adapters.py -v
\`\`\`

**Покрытие**: 1,035+ тестов
```

### Обновление таблицы Quick Reference

Добавить в существующую таблицу "📍 Быстрый поиск по задачам":

```markdown
| Futures core models | `core_futures.py` | `pytest tests/test_core_futures.py` |
| Crypto margin calculation | `impl_futures_margin.py` | `pytest tests/test_futures_margin.py` |
| CME SPAN margin | `impl_futures_margin.py::CMEMarginCalculator` | `pytest tests/test_futures_span_margin.py` |
| Funding rate tracking | `impl_futures_funding.py` | `pytest tests/test_futures_funding.py` |
| Liquidation simulation | `impl_futures_liquidation.py` | `pytest tests/test_futures_liquidation.py` |
| ADL simulation | `impl_futures_liquidation.py::ADLSimulator` | `pytest tests/test_futures_adl.py` |
| Futures risk guards | `services/futures_risk_guards.py` | `pytest tests/test_futures_risk_guards.py` |
| IB TWS adapter | `adapters/ib/` | `pytest tests/test_ib_adapters.py` |
| Margin call notifications | `services/futures_margin_notifications.py` | `pytest tests/test_margin_notifications.py` |
```

---

## 🧪 INTEGRATION TESTS SPECIFICATION

### Категории интеграционных тестов

#### 1. Cross-Component Integration Tests

```python
# NEW FILE: tests/integration/test_futures_integration.py
"""
End-to-end integration tests for futures trading.

Tests verify that all components work together correctly:
- Data pipeline → Features → Model → Execution → Position management
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from core_futures import FuturesContract, FuturesPosition, MarginMode
from impl_futures_margin import UnifiedMarginCalculator
from impl_futures_liquidation import LiquidationEngine
from impl_futures_funding import FundingManager
from services.futures_risk_guards import FuturesRiskGuardChain
from execution_providers_futures import FuturesL3ExecutionProvider


class TestCryptoFuturesIntegration:
    """End-to-end crypto futures integration tests."""

    @pytest.fixture
    def full_trading_stack(self):
        """Create complete trading stack with all components."""
        return {
            "margin_calc": UnifiedMarginCalculator(exchange="binance"),
            "liquidation": LiquidationEngine(exchange="binance"),
            "funding": FundingManager(),
            "risk_guards": FuturesRiskGuardChain.default_chain(),
            "executor": FuturesL3ExecutionProvider(level="L3"),
        }

    def test_full_trading_cycle(self, full_trading_stack):
        """
        Complete trading cycle:
        1. Open position
        2. Check margin requirements
        3. Apply funding payment
        4. Close position
        5. Verify P&L calculation
        """
        stack = full_trading_stack

        # 1. Open position
        position = FuturesPosition(
            symbol="BTCUSDT",
            qty=Decimal("1.0"),
            entry_price=Decimal("50000"),
            leverage=10,
            margin_mode=MarginMode.ISOLATED,
        )

        # 2. Check margin - should pass
        margin_req = stack["margin_calc"].calculate_margin(position)
        assert margin_req.is_sufficient(wallet_balance=Decimal("10000"))

        # 3. Apply funding
        funding_amount = stack["funding"].calculate_funding(
            position=position,
            funding_rate=Decimal("0.0003"),  # 3 bps
            mark_price=Decimal("50100"),
        )
        assert funding_amount == Decimal("-15.03")  # Paid (long position, positive rate)

        # 4. Mark-to-market
        unrealized_pnl = position.calculate_unrealized_pnl(
            mark_price=Decimal("51000"),
        )
        assert unrealized_pnl == Decimal("1000")  # +$1000 on 1 BTC × $1000 move

        # 5. Close position
        fill = stack["executor"].execute(
            order=Order(symbol="BTCUSDT", side="SELL", qty=Decimal("1.0")),
            market=MarketState(bid=Decimal("50990"), ask=Decimal("51010")),
        )
        realized_pnl = fill.price * fill.qty - position.entry_price * position.qty
        assert realized_pnl > Decimal("0")

    def test_liquidation_cascade_integration(self, full_trading_stack):
        """
        Test liquidation triggers correctly:
        1. Position approaches liquidation price
        2. Liquidation engine detects
        3. Risk guards fire
        4. Position force-closed
        """
        stack = full_trading_stack

        position = FuturesPosition(
            symbol="BTCUSDT",
            qty=Decimal("1.0"),
            entry_price=Decimal("50000"),
            leverage=20,  # High leverage
            margin_mode=MarginMode.ISOLATED,
        )

        # Price drops significantly
        mark_price = Decimal("47800")  # ~4.4% drop

        # Should trigger liquidation check
        liq_result = stack["liquidation"].check_liquidation(
            position=position,
            mark_price=mark_price,
            maintenance_margin_rate=Decimal("0.004"),  # 0.4%
        )

        assert liq_result.is_liquidatable is True
        assert liq_result.reason == "margin_ratio_below_maintenance"

    def test_funding_accumulation_over_time(self, full_trading_stack):
        """Test funding payments accumulate correctly over multiple periods."""
        stack = full_trading_stack

        position = FuturesPosition(
            symbol="ETHUSDT",
            qty=Decimal("10.0"),
            entry_price=Decimal("3000"),
            leverage=5,
        )

        # Simulate 3 funding periods
        funding_rates = [
            Decimal("0.0001"),   # 1 bps
            Decimal("-0.0002"),  # -2 bps (receive)
            Decimal("0.0003"),   # 3 bps
        ]

        total_funding = Decimal("0")
        for rate in funding_rates:
            payment = stack["funding"].calculate_funding(
                position=position,
                funding_rate=rate,
                mark_price=Decimal("3000"),
            )
            total_funding += payment

        # Net: -1 + 2 - 3 = -2 bps = -$6 on $30k position
        assert total_funding == Decimal("-6")


class TestCMEFuturesIntegration:
    """End-to-end CME futures integration tests."""

    def test_span_margin_with_position_change(self):
        """SPAN margin recalculation on position changes."""
        pass

    def test_daily_settlement_integration(self):
        """Daily settlement at 15:30 ET triggers correctly."""
        pass

    def test_contract_rollover_integration(self):
        """Contract rollover preserves position state."""
        pass


class TestCrossExchangeIntegration:
    """Tests for multi-exchange scenarios."""

    def test_crypto_and_cme_simultaneous(self):
        """
        Test running crypto and CME futures simultaneously:
        - Different margin systems
        - Different trading hours
        - Separate risk limits
        """
        pass

    def test_feature_flag_isolation(self):
        """Feature flags correctly isolate exchange behavior."""
        pass
```

#### 2. Data Pipeline Integration Tests

```python
# NEW FILE: tests/integration/test_futures_data_integration.py
"""
Data pipeline integration tests.

Tests verify:
- Data download → Validation → Feature generation → Training
"""

class TestFuturesDataPipeline:
    """Data pipeline integration tests."""

    def test_funding_data_to_features(self):
        """Funding rate data correctly flows to features."""
        # Download → Parse → Validate → Feature extraction
        pass

    def test_mark_price_temporal_alignment(self):
        """Mark price aligns with OHLCV data temporally."""
        pass

    def test_liquidation_data_integration(self):
        """Liquidation cascades appear in order flow features."""
        pass

    def test_feature_shift_prevents_leakage(self):
        """No look-ahead bias in futures features."""
        # Apply shift_features_for_lookahead()
        # Verify features at t don't use data from t+1
        pass
```

#### 3. Training Integration Tests

```python
# NEW FILE: tests/integration/test_futures_training_integration.py
"""
Training pipeline integration tests.
"""

class TestFuturesTrainingIntegration:
    """Training integration tests."""

    def test_training_with_liquidation_episodes(self):
        """Training handles liquidation-terminated episodes."""
        pass

    def test_reward_includes_funding(self):
        """Reward correctly includes funding payments."""
        pass

    def test_model_checkpoint_includes_futures_state(self):
        """Model checkpoints save futures-specific state."""
        pass
```

#### 4. Live Trading Integration Tests

```python
# NEW FILE: tests/integration/test_futures_live_integration.py
"""
Live trading integration tests (paper mode).
"""

class TestFuturesLiveIntegration:
    """Live trading integration tests."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_position_sync_recovery(self):
        """Position sync recovers after disconnect."""
        pass

    @pytest.mark.integration
    def test_margin_call_notification_flow(self):
        """Margin call triggers notification chain."""
        pass

    @pytest.mark.integration
    def test_rate_limiter_under_load(self):
        """Rate limiter prevents API violations under load."""
        pass
```

### Test Configuration

```yaml
# NEW FILE: tests/integration/pytest_integration.ini
[pytest]
markers =
    integration: Integration tests (may require external services)
    slow: Slow tests (> 10 seconds)
    requires_api: Tests requiring live API connection

testpaths =
    tests/integration

timeout = 120
timeout_method = thread

# Run integration tests with: pytest -m integration
# Skip slow tests with: pytest -m "not slow"
```

### Coverage Requirements

| Category | Minimum Coverage | Target Coverage |
|----------|-----------------|-----------------|
| Core Models | 95% | 100% |
| Margin Calculators | 90% | 95% |
| Liquidation Engine | 95% | 100% |
| Risk Guards | 90% | 95% |
| Adapters | 85% | 90% |
| Features Pipeline | 90% | 95% |

---

## 📊 MONITORING DASHBOARDS CONFIGURATION

### Prometheus Metrics

```yaml
# NEW FILE: configs/monitoring/futures_metrics.yaml
# Prometheus metrics configuration for futures trading

metrics:
  # ═══════════════════════════════════════════════════════════════════════════
  # POSITION METRICS
  # ═══════════════════════════════════════════════════════════════════════════
  position_metrics:
    - name: futures_position_value_usd
      type: gauge
      help: "Current position value in USD"
      labels: [symbol, side, exchange]

    - name: futures_position_leverage
      type: gauge
      help: "Current leverage for position"
      labels: [symbol, exchange]

    - name: futures_unrealized_pnl_usd
      type: gauge
      help: "Unrealized P&L in USD"
      labels: [symbol, side]

    - name: futures_margin_ratio
      type: gauge
      help: "Current margin ratio (equity/maintenance)"
      labels: [symbol, margin_mode]

  # ═══════════════════════════════════════════════════════════════════════════
  # FUNDING METRICS
  # ═══════════════════════════════════════════════════════════════════════════
  funding_metrics:
    - name: futures_funding_rate_bps
      type: gauge
      help: "Current funding rate in basis points"
      labels: [symbol]

    - name: futures_funding_payment_usd_total
      type: counter
      help: "Total funding payments (cumulative)"
      labels: [symbol, direction]  # direction: paid/received

    - name: futures_predicted_funding_rate_bps
      type: gauge
      help: "Predicted next funding rate"
      labels: [symbol]

    - name: futures_funding_rate_8h_avg_bps
      type: gauge
      help: "8-hour average funding rate"
      labels: [symbol]

  # ═══════════════════════════════════════════════════════════════════════════
  # LIQUIDATION METRICS
  # ═══════════════════════════════════════════════════════════════════════════
  liquidation_metrics:
    - name: futures_liquidation_price_distance_pct
      type: gauge
      help: "Distance to liquidation price in percent"
      labels: [symbol, side]

    - name: futures_liquidation_events_total
      type: counter
      help: "Total liquidation events"
      labels: [symbol, reason]  # reason: margin_call, mark_price, adl

    - name: futures_adl_queue_rank
      type: gauge
      help: "ADL queue rank (1-5, higher = more risk)"
      labels: [symbol, side]

    - name: futures_margin_call_events_total
      type: counter
      help: "Total margin call events by level"
      labels: [symbol, level]  # level: warning, danger, critical

  # ═══════════════════════════════════════════════════════════════════════════
  # EXECUTION METRICS
  # ═══════════════════════════════════════════════════════════════════════════
  execution_metrics:
    - name: futures_order_fill_rate
      type: gauge
      help: "Order fill rate (0-1)"
      labels: [symbol, order_type, side]

    - name: futures_slippage_bps
      type: histogram
      help: "Execution slippage in basis points"
      labels: [symbol, side, liquidity_role]
      buckets: [0.5, 1, 2, 5, 10, 20, 50, 100]

    - name: futures_fill_latency_ms
      type: histogram
      help: "Order fill latency in milliseconds"
      labels: [symbol, order_type]
      buckets: [10, 50, 100, 500, 1000, 5000]

    - name: futures_fees_paid_usd_total
      type: counter
      help: "Total fees paid in USD"
      labels: [symbol, fee_type]  # fee_type: maker, taker, funding

  # ═══════════════════════════════════════════════════════════════════════════
  # RISK METRICS
  # ═══════════════════════════════════════════════════════════════════════════
  risk_metrics:
    - name: futures_risk_guard_triggers_total
      type: counter
      help: "Risk guard trigger count"
      labels: [guard_type, action]  # action: blocked, warning, passed

    - name: futures_max_leverage_used
      type: gauge
      help: "Maximum leverage used across positions"
      labels: [exchange]

    - name: futures_concentration_ratio
      type: gauge
      help: "Position concentration (largest/total)"
      labels: [exchange]

  # ═══════════════════════════════════════════════════════════════════════════
  # API METRICS (IB + Binance)
  # ═══════════════════════════════════════════════════════════════════════════
  api_metrics:
    - name: futures_api_requests_total
      type: counter
      help: "Total API requests"
      labels: [exchange, endpoint, status]

    - name: futures_api_latency_ms
      type: histogram
      help: "API request latency"
      labels: [exchange, endpoint]
      buckets: [10, 50, 100, 200, 500, 1000, 2000, 5000]

    - name: futures_api_rate_limit_remaining
      type: gauge
      help: "Remaining rate limit quota"
      labels: [exchange, limit_type]

    - name: futures_websocket_reconnects_total
      type: counter
      help: "WebSocket reconnection count"
      labels: [exchange, stream_type]
```

### Grafana Dashboard Configuration

```json
// NEW FILE: configs/monitoring/grafana/futures_dashboard.json
{
  "dashboard": {
    "title": "Futures Trading Dashboard",
    "uid": "futures-main",
    "tags": ["futures", "trading", "production"],
    "refresh": "10s",
    "rows": [
      {
        "title": "Position Overview",
        "panels": [
          {
            "title": "Total Position Value (USD)",
            "type": "stat",
            "targets": [
              {
                "expr": "sum(futures_position_value_usd)",
                "legendFormat": "Total Value"
              }
            ],
            "fieldConfig": {
              "defaults": {
                "unit": "currencyUSD",
                "thresholds": {
                  "mode": "absolute",
                  "steps": [
                    {"color": "green", "value": null},
                    {"color": "yellow", "value": 100000},
                    {"color": "red", "value": 500000}
                  ]
                }
              }
            }
          },
          {
            "title": "Margin Ratio by Symbol",
            "type": "gauge",
            "targets": [
              {
                "expr": "futures_margin_ratio",
                "legendFormat": "{{symbol}}"
              }
            ],
            "fieldConfig": {
              "defaults": {
                "min": 0,
                "max": 10,
                "thresholds": {
                  "steps": [
                    {"color": "red", "value": 0},
                    {"color": "orange", "value": 1.5},
                    {"color": "yellow", "value": 2.0},
                    {"color": "green", "value": 3.0}
                  ]
                }
              }
            }
          },
          {
            "title": "Unrealized P&L",
            "type": "timeseries",
            "targets": [
              {
                "expr": "futures_unrealized_pnl_usd",
                "legendFormat": "{{symbol}} ({{side}})"
              }
            ]
          }
        ]
      },
      {
        "title": "Funding Rates",
        "panels": [
          {
            "title": "Current Funding Rates (bps)",
            "type": "table",
            "targets": [
              {
                "expr": "futures_funding_rate_bps",
                "format": "table",
                "instant": true
              }
            ]
          },
          {
            "title": "Funding Rate History",
            "type": "timeseries",
            "targets": [
              {
                "expr": "futures_funding_rate_bps",
                "legendFormat": "{{symbol}}"
              }
            ],
            "fieldConfig": {
              "defaults": {
                "custom": {
                  "lineWidth": 2,
                  "fillOpacity": 10
                }
              }
            }
          },
          {
            "title": "Cumulative Funding Paid/Received",
            "type": "stat",
            "targets": [
              {
                "expr": "sum(futures_funding_payment_usd_total{direction='paid'})",
                "legendFormat": "Paid"
              },
              {
                "expr": "sum(futures_funding_payment_usd_total{direction='received'})",
                "legendFormat": "Received"
              }
            ]
          }
        ]
      },
      {
        "title": "Risk & Liquidation",
        "panels": [
          {
            "title": "Distance to Liquidation",
            "type": "gauge",
            "targets": [
              {
                "expr": "futures_liquidation_price_distance_pct",
                "legendFormat": "{{symbol}}"
              }
            ],
            "fieldConfig": {
              "defaults": {
                "unit": "percent",
                "min": 0,
                "max": 50,
                "thresholds": {
                  "steps": [
                    {"color": "red", "value": 0},
                    {"color": "orange", "value": 5},
                    {"color": "yellow", "value": 10},
                    {"color": "green", "value": 20}
                  ]
                }
              }
            }
          },
          {
            "title": "ADL Queue Position",
            "type": "bargauge",
            "targets": [
              {
                "expr": "futures_adl_queue_rank",
                "legendFormat": "{{symbol}}"
              }
            ],
            "fieldConfig": {
              "defaults": {
                "min": 1,
                "max": 5,
                "thresholds": {
                  "steps": [
                    {"color": "green", "value": 1},
                    {"color": "yellow", "value": 3},
                    {"color": "orange", "value": 4},
                    {"color": "red", "value": 5}
                  ]
                }
              }
            }
          },
          {
            "title": "Margin Calls (Last 24h)",
            "type": "stat",
            "targets": [
              {
                "expr": "increase(futures_margin_call_events_total[24h])",
                "legendFormat": "{{level}}"
              }
            ]
          },
          {
            "title": "Risk Guard Triggers",
            "type": "timeseries",
            "targets": [
              {
                "expr": "rate(futures_risk_guard_triggers_total[5m])",
                "legendFormat": "{{guard_type}} - {{action}}"
              }
            ]
          }
        ]
      },
      {
        "title": "Execution Quality",
        "panels": [
          {
            "title": "Slippage Distribution (bps)",
            "type": "histogram",
            "targets": [
              {
                "expr": "futures_slippage_bps_bucket",
                "legendFormat": "{{le}} bps"
              }
            ]
          },
          {
            "title": "Fill Rate by Order Type",
            "type": "bargauge",
            "targets": [
              {
                "expr": "futures_order_fill_rate",
                "legendFormat": "{{order_type}}"
              }
            ]
          },
          {
            "title": "Fill Latency p95",
            "type": "stat",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, futures_fill_latency_ms_bucket)",
                "legendFormat": "p95 Latency"
              }
            ],
            "fieldConfig": {
              "defaults": {
                "unit": "ms"
              }
            }
          }
        ]
      },
      {
        "title": "API Health",
        "panels": [
          {
            "title": "API Request Rate",
            "type": "timeseries",
            "targets": [
              {
                "expr": "rate(futures_api_requests_total[1m])",
                "legendFormat": "{{exchange}} - {{endpoint}}"
              }
            ]
          },
          {
            "title": "Rate Limit Remaining",
            "type": "gauge",
            "targets": [
              {
                "expr": "futures_api_rate_limit_remaining",
                "legendFormat": "{{exchange}} - {{limit_type}}"
              }
            ]
          },
          {
            "title": "WebSocket Reconnects (24h)",
            "type": "stat",
            "targets": [
              {
                "expr": "increase(futures_websocket_reconnects_total[24h])",
                "legendFormat": "{{exchange}}"
              }
            ]
          },
          {
            "title": "API Latency p99",
            "type": "timeseries",
            "targets": [
              {
                "expr": "histogram_quantile(0.99, futures_api_latency_ms_bucket)",
                "legendFormat": "{{exchange}}"
              }
            ]
          }
        ]
      }
    ]
  }
}
```

### Alert Rules

```yaml
# NEW FILE: configs/monitoring/alerts/futures_alerts.yaml
# Prometheus alerting rules for futures trading

groups:
  - name: futures_critical
    rules:
      # ─────────────────────────────────────────────────────────────────────────
      # MARGIN ALERTS
      # ─────────────────────────────────────────────────────────────────────────
      - alert: FuturesMarginCritical
        expr: futures_margin_ratio < 1.2
        for: 1m
        labels:
          severity: critical
          team: trading
        annotations:
          summary: "Critical margin ratio on {{ $labels.symbol }}"
          description: "Margin ratio {{ $value }} is below 1.2 (critical threshold)"
          runbook: "https://docs.internal/runbooks/margin-critical"

      - alert: FuturesLiquidationImminent
        expr: futures_liquidation_price_distance_pct < 2
        for: 30s
        labels:
          severity: critical
          team: trading
          page: true
        annotations:
          summary: "Liquidation imminent on {{ $labels.symbol }}"
          description: "Distance to liquidation is {{ $value }}%"
          action: "Immediately reduce position or add margin"

      # ─────────────────────────────────────────────────────────────────────────
      # FUNDING ALERTS
      # ─────────────────────────────────────────────────────────────────────────
      - alert: FuturesHighFundingRate
        expr: abs(futures_funding_rate_bps) > 50
        for: 5m
        labels:
          severity: warning
          team: trading
        annotations:
          summary: "High funding rate on {{ $labels.symbol }}"
          description: "Funding rate is {{ $value }} bps"

      - alert: FuturesFundingExcessivePaid
        expr: increase(futures_funding_payment_usd_total{direction="paid"}[24h]) > 1000
        for: 1m
        labels:
          severity: warning
          team: trading
        annotations:
          summary: "Excessive funding paid on {{ $labels.symbol }}"
          description: "Paid ${{ $value }} in funding over 24h"

      # ─────────────────────────────────────────────────────────────────────────
      # ADL ALERTS
      # ─────────────────────────────────────────────────────────────────────────
      - alert: FuturesADLHighRisk
        expr: futures_adl_queue_rank >= 4
        for: 5m
        labels:
          severity: warning
          team: trading
        annotations:
          summary: "High ADL risk on {{ $labels.symbol }}"
          description: "ADL queue rank is {{ $value }}/5"

      # ─────────────────────────────────────────────────────────────────────────
      # API ALERTS
      # ─────────────────────────────────────────────────────────────────────────
      - alert: FuturesAPIRateLimitLow
        expr: futures_api_rate_limit_remaining < 10
        for: 1m
        labels:
          severity: warning
          team: infrastructure
        annotations:
          summary: "API rate limit nearly exhausted for {{ $labels.exchange }}"
          description: "Only {{ $value }} requests remaining"

      - alert: FuturesAPIHighLatency
        expr: histogram_quantile(0.99, futures_api_latency_ms_bucket) > 2000
        for: 5m
        labels:
          severity: warning
          team: infrastructure
        annotations:
          summary: "High API latency for {{ $labels.exchange }}"
          description: "p99 latency is {{ $value }}ms"

      - alert: FuturesWebSocketDisconnected
        expr: increase(futures_websocket_reconnects_total[5m]) > 3
        for: 1m
        labels:
          severity: warning
          team: infrastructure
        annotations:
          summary: "Frequent WebSocket reconnects for {{ $labels.exchange }}"
          description: "{{ $value }} reconnects in 5 minutes"

      # ─────────────────────────────────────────────────────────────────────────
      # EXECUTION ALERTS
      # ─────────────────────────────────────────────────────────────────────────
      - alert: FuturesHighSlippage
        expr: histogram_quantile(0.95, futures_slippage_bps_bucket) > 10
        for: 10m
        labels:
          severity: warning
          team: trading
        annotations:
          summary: "High slippage on {{ $labels.symbol }}"
          description: "p95 slippage is {{ $value }} bps"

      - alert: FuturesLowFillRate
        expr: futures_order_fill_rate < 0.8
        for: 15m
        labels:
          severity: warning
          team: trading
        annotations:
          summary: "Low fill rate on {{ $labels.symbol }}"
          description: "Fill rate is {{ $value | humanizePercentage }}"
```

### Metrics Collection Implementation

```python
# NEW FILE: services/futures_metrics.py
"""
Prometheus metrics collector for futures trading.

Usage:
    from services.futures_metrics import FuturesMetrics

    metrics = FuturesMetrics()
    metrics.update_position(position)
    metrics.record_funding_payment(symbol, amount)
    metrics.record_fill(fill)
"""

from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry
from typing import Optional
from decimal import Decimal


class FuturesMetrics:
    """Prometheus metrics for futures trading."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()

        # Position metrics
        self.position_value = Gauge(
            "futures_position_value_usd",
            "Current position value in USD",
            ["symbol", "side", "exchange"],
            registry=self.registry,
        )
        self.margin_ratio = Gauge(
            "futures_margin_ratio",
            "Current margin ratio",
            ["symbol", "margin_mode"],
            registry=self.registry,
        )
        self.unrealized_pnl = Gauge(
            "futures_unrealized_pnl_usd",
            "Unrealized P&L in USD",
            ["symbol", "side"],
            registry=self.registry,
        )

        # Funding metrics
        self.funding_rate = Gauge(
            "futures_funding_rate_bps",
            "Current funding rate in basis points",
            ["symbol"],
            registry=self.registry,
        )
        self.funding_payment = Counter(
            "futures_funding_payment_usd_total",
            "Total funding payments",
            ["symbol", "direction"],
            registry=self.registry,
        )

        # Liquidation metrics
        self.liquidation_distance = Gauge(
            "futures_liquidation_price_distance_pct",
            "Distance to liquidation price",
            ["symbol", "side"],
            registry=self.registry,
        )
        self.adl_rank = Gauge(
            "futures_adl_queue_rank",
            "ADL queue rank (1-5)",
            ["symbol", "side"],
            registry=self.registry,
        )
        self.margin_call_events = Counter(
            "futures_margin_call_events_total",
            "Margin call events by level",
            ["symbol", "level"],
            registry=self.registry,
        )

        # Execution metrics
        self.slippage = Histogram(
            "futures_slippage_bps",
            "Execution slippage in basis points",
            ["symbol", "side", "liquidity_role"],
            buckets=[0.5, 1, 2, 5, 10, 20, 50, 100],
            registry=self.registry,
        )
        self.fill_latency = Histogram(
            "futures_fill_latency_ms",
            "Order fill latency",
            ["symbol", "order_type"],
            buckets=[10, 50, 100, 500, 1000, 5000],
            registry=self.registry,
        )

        # API metrics
        self.api_requests = Counter(
            "futures_api_requests_total",
            "Total API requests",
            ["exchange", "endpoint", "status"],
            registry=self.registry,
        )
        self.api_latency = Histogram(
            "futures_api_latency_ms",
            "API request latency",
            ["exchange", "endpoint"],
            buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000],
            registry=self.registry,
        )
        self.rate_limit_remaining = Gauge(
            "futures_api_rate_limit_remaining",
            "Remaining rate limit quota",
            ["exchange", "limit_type"],
            registry=self.registry,
        )

    def update_position(
        self,
        symbol: str,
        side: str,
        exchange: str,
        value_usd: Decimal,
        margin_ratio: Decimal,
        unrealized_pnl: Decimal,
        margin_mode: str = "isolated",
    ) -> None:
        """Update position metrics."""
        self.position_value.labels(
            symbol=symbol, side=side, exchange=exchange
        ).set(float(value_usd))

        self.margin_ratio.labels(
            symbol=symbol, margin_mode=margin_mode
        ).set(float(margin_ratio))

        self.unrealized_pnl.labels(
            symbol=symbol, side=side
        ).set(float(unrealized_pnl))

    def record_funding_payment(
        self,
        symbol: str,
        amount: Decimal,
    ) -> None:
        """Record funding payment."""
        direction = "paid" if amount < 0 else "received"
        self.funding_payment.labels(
            symbol=symbol, direction=direction
        ).inc(abs(float(amount)))

    def update_funding_rate(self, symbol: str, rate_bps: float) -> None:
        """Update current funding rate."""
        self.funding_rate.labels(symbol=symbol).set(rate_bps)

    def update_liquidation_distance(
        self,
        symbol: str,
        side: str,
        distance_pct: float,
    ) -> None:
        """Update distance to liquidation."""
        self.liquidation_distance.labels(
            symbol=symbol, side=side
        ).set(distance_pct)

    def update_adl_rank(self, symbol: str, side: str, rank: int) -> None:
        """Update ADL queue rank."""
        self.adl_rank.labels(symbol=symbol, side=side).set(rank)

    def record_margin_call(self, symbol: str, level: str) -> None:
        """Record margin call event."""
        self.margin_call_events.labels(symbol=symbol, level=level).inc()

    def record_fill(
        self,
        symbol: str,
        side: str,
        order_type: str,
        slippage_bps: float,
        latency_ms: float,
        liquidity_role: str,
    ) -> None:
        """Record order fill metrics."""
        self.slippage.labels(
            symbol=symbol, side=side, liquidity_role=liquidity_role
        ).observe(slippage_bps)

        self.fill_latency.labels(
            symbol=symbol, order_type=order_type
        ).observe(latency_ms)

    def record_api_request(
        self,
        exchange: str,
        endpoint: str,
        status: str,
        latency_ms: float,
    ) -> None:
        """Record API request metrics."""
        self.api_requests.labels(
            exchange=exchange, endpoint=endpoint, status=status
        ).inc()
        self.api_latency.labels(
            exchange=exchange, endpoint=endpoint
        ).observe(latency_ms)

    def update_rate_limit(
        self,
        exchange: str,
        limit_type: str,
        remaining: int,
    ) -> None:
        """Update rate limit remaining."""
        self.rate_limit_remaining.labels(
            exchange=exchange, limit_type=limit_type
        ).set(remaining)
```

---

## 📊 SUMMARY

### Total Deliverables by Track

#### Crypto Track (Binance)

| Phase | Description | Files | Tests |
|-------|-------------|-------|-------|
| 0 | Research | 0 | 0 (research) |
| 1 | Core Models (Unified) | 5 | 90+ |
| 2 | Data Download | 4 | 100+ |
| 3A | Binance Adapters | 3 | 50+ |
| 4A | Margin & Liquidation | 3 | 80+ |
| 5A | L3 LOB Integration | 2 | 60+ |
| 6A | Risk Management | 2 | 80+ |
| 7A | Features Pipeline | 3 | 50+ |
| **Subtotal** | | **22** | **510+** |

#### CME Track (Interactive Brokers)

| Phase | Description | Files | Tests |
|-------|-------------|-------|-------|
| 3B | IB TWS Adapters | 5 | 45+ |
| 4B | SPAN Margin | 3 | 55+ |
| 5B | CME L3 Execution | 2 | 55+ |
| 6B | CME Risk Guards | 2 | 70+ |
| 7B | CME Features | 2 | 40+ |
| **Subtotal** | | **14** | **265+** |

#### Shared Phases

| Phase | Description | Files | Tests |
|-------|-------------|-------|-------|
| 8 | Training Pipeline | 3 | 60+ |
| 9 | Live Trading | 3 | 50+ |
| 10 | Validation | 5+ | 150+ |
| **Subtotal** | | **11+** | **260+** |

### Grand Total

| Metric | Count |
|--------|-------|
| **Total Files** | **47+** |
| **Total Tests** | **1,035+** |
| **New Core Models** | 15+ |
| **New Adapters** | 12+ |
| **New Risk Guards** | 10+ |

### Key Metrics

| Metric | Crypto Target | CME Target |
|--------|---------------|------------|
| Simulation realism | 95%+ | 95%+ |
| Fill rate accuracy | >95% | >95% |
| Slippage error | <3 bps | <2 bps |
| Margin calc error | <0.5% | <0.1% (SPAN) |
| Test coverage | >90% | >90% |
| Backward compat | 100% | 100% |

### Futures Type Coverage

| Type | Exchange | Products | Status |
|------|----------|----------|--------|
| Crypto Perpetual | Binance | BTCUSDT, ETHUSDT, ... | Phase 1-7A |
| Crypto Quarterly | Binance | BTCUSDT_YYMMDD | Phase 1-7A |
| Index Futures | CME/CBOT | ES, NQ, MES, MNQ | Phase 3B-7B |
| Commodity Futures | COMEX/NYMEX | GC, SI, CL, NG | Phase 3B-7B |
| Currency Futures | CME | 6E, 6J, 6B, 6A | Phase 3B-7B |
| Bond Futures | CBOT | ZB, ZN, ZT | Phase 3B-7B |

### Risk Mitigation

1. **Backward compatibility**: All changes isolated, existing tests run on every phase
2. **Incremental delivery**: Each phase deployable independently (Crypto first, then CME)
3. **Feature flags**: Futures disabled by default, opt-in per asset class
4. **Parallel development**: Crypto and CME tracks can be developed concurrently
5. **Extensive testing**: 1,035+ tests across all phases
6. **Reuse existing L3**: Same LOB simulation, adapted for each exchange
7. **Documentation**: Complete docs for maintenance

### Development Timeline (Parallel Tracks)

```
Month 1-2: Phase 0-2 (Research + Core + Data) ─────────────────────────┐
                                                                       │
Month 2-3: ┌─ Phase 3A-4A (Crypto Margin)                              │
           └─ Phase 3B-4B (CME SPAN) ──────────────────────────────────┤
                                                                       │
Month 3-4: ┌─ Phase 5A-6A (Crypto L3 + Risk)                           │
           └─ Phase 5B-6B (CME L3 + Risk) ─────────────────────────────┤
                                                                       │
Month 4-5: ┌─ Phase 7A (Crypto Features)                               │
           └─ Phase 7B (CME Features) ─────────────────────────────────┤
                                                                       │
Month 5-6: Phase 8-10 (Training + Live + Validation) ──────────────────┘
```

---

## 🔗 REFERENCES

### Research Papers

1. Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
2. Kyle (1985): "Continuous Auctions and Insider Trading"
3. Cartea, Jaimungal, Penalva (2015): "Algorithmic and HF Trading"
4. Cont, Kukanov, Stoikov (2014): "The Price Impact of Order Book Events"
5. Gatheral (2010): "No-Dynamic-Arbitrage and Market Impact"

### Exchange Documentation

#### Binance (Crypto Futures)

1. [Binance Futures API](https://binance-docs.github.io/apidocs/futures/en/)
2. [Binance Leverage & Margin](https://www.binance.com/en/support/faq/360033162192)
3. [Binance Liquidation](https://www.binance.com/en/support/faq/360033525271)
4. [Binance Funding Rate](https://www.binance.com/en/support/faq/360033525031)

#### CME Group

5. [CME Globex Documentation](https://www.cmegroup.com/confluence/display/EPICSANDBOX/Globex)
6. [CME SPAN Margin](https://www.cmegroup.com/clearing/risk-management/span-overview.html)
7. [CME Circuit Breakers](https://www.cmegroup.com/education/articles-and-reports/understanding-stock-index-futures-circuit-breakers.html)
8. [CME Contract Specifications](https://www.cmegroup.com/trading/equity-index/us-index/e-mini-sandp500_contract_specifications.html)

#### Interactive Brokers

9. [IB TWS API](https://interactivebrokers.github.io/tws-api/)
10. [ib_insync Library](https://ib-insync.readthedocs.io/)
11. [IB Futures Trading](https://www.interactivebrokers.com/en/trading/futures.php)

### Internal Documentation

1. [L3 LOB Simulation](l3_simulator/)
2. [Forex Integration](FOREX_INTEGRATION_PLAN.md)
3. [Execution Providers](futures/overview.md)
4. [Asset Class Defaults](../configs/asset_class_defaults.yaml)

---

**Дата создания**: 2025-11-30
**Автор**: Sultan Suyunbayev
**Версия плана**: 2.0 (Unified Multi-Asset Futures)
