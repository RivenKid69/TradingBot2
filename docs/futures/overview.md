# Futures Integration Overview

## Introduction

The futures integration extends the platform to support crypto perpetual futures (Binance USDT-M) and CME Group futures (via Interactive Brokers). This document provides a comprehensive overview of the architecture, components, and design decisions.

**CCEA Architecture Note**: In production, live trading runs via the local Agent daemon (`packages.agent.daemon.agentd`). The `script_live.py` entry point referenced in diagrams is for development/testing only. See `docs/CCEA_OVERVIEW.md`.

## Supported Futures Types

| Type | Exchange | Examples | Status |
|------|----------|----------|--------|
| **Crypto Perpetual** | Binance | BTCUSDT, ETHUSDT | Tested and operational |
| **Crypto Quarterly** | Binance | BTCUSDT_240329 | Planned |
| **Equity Index** | CME (via IB) | ES, NQ, YM, RTY | Tested and operational |
| **Commodity** | CME (via IB) | GC, CL, SI, NG | Tested and operational |
| **Currency** | CME (via IB) | 6E, 6J, 6B | Tested and operational |
| **Bonds** | CME (via IB) | ZN, ZB, ZT | Tested and operational |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       User Interface                             │
│         (script_live.py, train_model_multi_patch.py)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Unified Futures Risk Guard                     │
│              (services/unified_futures_risk.py)                  │
│       - Auto asset type detection                                │
│       - Delegation to crypto/CME guards                          │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌─────────────────────┐               ┌─────────────────────┐
│  Crypto Futures     │               │  CME Futures        │
│  ┌───────────────┐  │               │  ┌───────────────┐  │
│  │ L2 Slippage   │  │               │  │ L2 Slippage   │  │
│  │ (funding,OI)  │  │               │  │ (session,CB)  │  │
│  └───────────────┘  │               │  └───────────────┘  │
│  ┌───────────────┐  │               │  ┌───────────────┐  │
│  │ L3 LOB        │  │               │  │ L3 Globex     │  │
│  │ (cascade,ADL) │  │               │  │ (MWP,stops)   │  │
│  └───────────────┘  │               │  └───────────────┘  │
│  ┌───────────────┐  │               │  ┌───────────────┐  │
│  │ Tiered Margin │  │               │  │ SPAN Margin   │  │
│  └───────────────┘  │               │  └───────────────┘  │
│  ┌───────────────┐  │               │  ┌───────────────┐  │
│  │ Risk Guards   │  │               │  │ Risk Guards   │  │
│  │ (leverage,    │  │               │  │ (CB, settle,  │  │
│  │  funding,ADL) │  │               │  │  rollover)    │  │
│  └───────────────┘  │               │  └───────────────┘  │
└─────────────────────┘               └─────────────────────┘
          │                                       │
          └───────────────────┬───────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Core Futures Models                            │
│                    (core_futures.py)                             │
│       FuturesType, FuturesPosition, FuturesContractSpec          │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Core Models (`core_futures.py`)

- **FuturesType**: Enum defining futures types (CRYPTO_PERPETUAL, CRYPTO_QUARTERLY, CME_INDEX, etc.)
- **MarginMode**: Enum for margin modes (CROSS, ISOLATED, SPAN)
- **PositionSide**: Long/Short position enum
- **FuturesContractSpec**: Contract specifications (multiplier, tick size, margin requirements)
- **FuturesPosition**: Position representation with PnL calculation

### 2. Margin Calculation

| System | File | Description |
|--------|------|-------------|
| **Tiered Margin** | `impl_futures_margin.py` | Binance-style leverage brackets |
| **SPAN Margin** | `impl_span_margin.py` | CME's risk-based margin system |

### 3. Execution Providers

| Level | Crypto | CME | Description |
|-------|--------|-----|-------------|
| **L2** | `execution_providers_futures.py` | `execution_providers_cme.py` | Statistical slippage models |
| **L3** | `execution_providers_futures_l3.py` | `execution_providers_cme_l3.py` | Full LOB simulation |

### 4. Risk Management

| Component | Crypto | CME |
|-----------|--------|-----|
| **Leverage Guard** | Max leverage by notional tier | N/A (SPAN-based) |
| **Margin Guard** | Margin ratio monitoring | SPAN margin monitoring |
| **Funding Guard** | Funding rate exposure | N/A |
| **ADL Guard** | ADL queue risk | N/A |
| **Circuit Breaker** | N/A | Rule 80B, velocity logic |
| **Settlement Guard** | N/A | Daily settlement risk |
| **Rollover Guard** | N/A | Contract expiration |

### 5. LOB Extensions

| Feature | Crypto | CME |
|---------|--------|-----|
| **Matching Engine** | `lob/matching_engine.py` | `lob/cme_matching.py` (Globex) |
| **Liquidation Cascade** | `lob/futures_extensions.py` | N/A |
| **Insurance Fund** | `lob/futures_extensions.py` | N/A |
| **ADL Queue** | `lob/futures_extensions.py` | N/A |
| **MWP Orders** | N/A | `lob/cme_matching.py` |
| **Stop Orders** | Standard | Velocity logic |

## Design Principles

### 1. Backward Compatibility

All futures integration is additive. Existing crypto spot, equity, and forex functionality remains unchanged. The `create_execution_provider()` factory automatically routes to the appropriate provider based on asset class.

### 2. Asset Type Detection

The unified risk guard automatically detects asset type from symbol patterns:

| Pattern | Asset Type |
|---------|------------|
| `*USDT`, `*BUSD` | Crypto Perpetual |
| `*_YYMMDD` | Crypto Quarterly |
| `ES`, `NQ`, `YM`, `RTY` | CME Equity Index |
| `GC`, `SI`, `HG` | CME Metal |
| `CL`, `NG`, `RB` | CME Energy |
| `6E`, `6J`, `6B` | CME Currency |
| `ZN`, `ZB`, `ZT` | CME Bond |

### 3. Configurability

All parameters are configurable via YAML files:
- `configs/config_train_futures.yaml` - Training configuration
- `configs/config_live_futures.yaml` - Live trading configuration
- `configs/unified_futures_risk.yaml` - Risk management settings
- `configs/feature_flags_futures.yaml` - Feature rollout control

### 4. Testability

Every component has comprehensive test coverage:
- Unit tests for individual components
- Integration tests for cross-component interactions
- Validation tests against target metrics
- Backward compatibility tests

## Performance Targets

| Component | Target | Metric |
|-----------|--------|--------|
| L2 Execution | <100 μs | P95 latency |
| L3 Execution | <500 μs | P95 latency |
| Margin Calculation | <50 μs | P95 latency |
| Funding Rate | <10 μs | P95 latency |
| Risk Guard Check | <20 μs | P95 latency |

## Validation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Fill Rate | >95% | Percentage of orders successfully filled |
| Slippage Error | <3 bps | Vs historical execution data |
| Funding Accuracy | >99% | Funding payment calculation accuracy |
| Liquidation Timing | <1 bar | Delay from trigger to liquidation |
| Margin Error | <0.1% | Margin calculation accuracy |

## File Structure

```
TradingBot2/
├── core_futures.py                    # Core models and enums
├── impl_futures_margin.py             # Tiered margin calculator
├── impl_span_margin.py                # CME SPAN margin calculator
├── impl_circuit_breaker.py            # Circuit breaker simulation
├── impl_cme_settlement.py             # Daily settlement engine
├── impl_cme_rollover.py               # Contract rollover manager
├── execution_providers_futures.py     # L2 crypto futures execution
├── execution_providers_futures_l3.py  # L3 crypto futures execution
├── execution_providers_cme.py         # L2 CME futures execution
├── execution_providers_cme_l3.py      # L3 CME futures execution
├── services/
│   ├── futures_risk_guards.py         # Crypto futures risk guards
│   ├── cme_risk_guards.py             # CME futures risk guards
│   ├── unified_futures_risk.py        # Unified risk management
│   ├── futures_live_runner.py         # Live trading coordinator
│   ├── futures_position_sync.py       # Position synchronization
│   ├── futures_margin_monitor.py      # Margin monitoring
│   ├── futures_funding_tracker.py     # Funding rate tracking
│   ├── futures_feature_flags.py       # Feature rollout control
│   └── cme_calendar.py                # CME trading calendar
├── wrappers/
│   └── futures_env.py                 # Futures trading env wrapper
├── lob/
│   ├── futures_extensions.py          # Crypto futures LOB extensions
│   └── cme_matching.py                # CME Globex matching engine
├── adapters/
│   └── ib/                            # Interactive Brokers adapters
│       ├── market_data.py
│       ├── order_execution.py
│       └── exchange_info.py
├── configs/
│   ├── config_train_futures.yaml
│   ├── config_live_futures.yaml
│   ├── unified_futures_risk.yaml
│   └── feature_flags_futures.yaml
├── tests/
│   ├── test_futures_*.py              # All futures tests (565+)
│   ├── test_futures_validation.py     # Validation tests (125)
│   └── test_futures_backward_compatibility.py # Compat tests (50+)
├── benchmarks/
│   └── bench_futures_simulation.py    # Performance benchmarks
└── docs/
    └── futures/                       # This documentation
```

## Next Steps

1. Read [Configuration](configuration.md) to understand how to configure futures trading
2. Read [Margin Calculation](margin_calculation.md) for margin system details
3. Read [Funding Rates](funding_rates.md) for crypto perpetual funding mechanics
4. Read [Liquidation](liquidation.md) for liquidation engine details
5. Read [Deployment](deployment.md) for production deployment guide
6. Read [Migration Guide](migration_guide.md) for upgrading existing systems
