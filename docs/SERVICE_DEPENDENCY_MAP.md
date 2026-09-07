# Service Dependency Map

This document describes the service architecture, dependencies, and data flow.

**CCEA Architecture Note**: In production deployments, live trading runs via the local Agent daemon (`packages.agent.daemon.agentd`). The `script_live.py` entry point is retained for development and testing only. See `docs/CCEA_OVERVIEW.md` for the full architecture.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ENTRY POINTS                                    │
│  script_backtest.py    train_model_multi_patch.py    script_live.py         │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────────┐
│                              SERVICES                                        │
│  service_backtest.py    service_train.py    service_signal_runner.py        │
│  service_eval.py        service_calibrate_*.py                              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────────┐
│                           IMPLEMENTATIONS                                    │
│  impl_sim_executor.py   impl_fees.py   impl_slippage.py   impl_latency.py   │
│  impl_offline_data.py   impl_binance_*.py   impl_risk_basic.py              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────────┐
│                               CORE                                           │
│  core_config.py   core_models.py   core_strategy.py   core_events.py        │
│  core_constants.py   core_conformal.py                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Dependencies (STRICT!)

```
┌─────────────┐
│  script_*   │  ← Entry points (CLI)
└──────┬──────┘
       │ uses
       ▼
┌─────────────┐
│  service_*  │  ← Business logic
└──────┬──────┘
       │ uses
       ▼
┌─────────────┐
│  impl_*     │  ← Infrastructure
└──────┬──────┘
       │ uses
       ▼
┌─────────────┐
│  core_*     │  ← Models, contracts (NO DEPENDENCIES)
└─────────────┘

⚠️  NEVER import upward! This causes circular imports.
```

---

## Service Components

### 1. Live Trading Service

```
script_live.py
    │
    ├── service_signal_runner.py
    │       │
    │       ├── Trading Environment (trading_patchnew.py)
    │       │       │
    │       │       ├── Mediator (mediator.py)
    │       │       ├── Feature Pipeline (features_pipeline.py)
    │       │       └── Execution Simulation (execution_sim.py)
    │       │
    │       ├── Policy (distributional_ppo.py + custom_policy_patch1.py)
    │       │
    │       └── Risk Guard (risk_guard.py)
    │
    ├── services/ops_kill_switch.py
    │
    ├── services/healthcheck.py (optional)
    │
    └── Adapters
            │
            ├── adapters/binance/* (crypto)
            │       ├── market_data.py
            │       ├── fees.py
            │       └── exchange_info.py
            │
            └── adapters/alpaca/* (stocks)
                    ├── market_data.py
                    ├── order_execution.py
                    └── exchange_info.py
```

### 2. Training Service

```
train_model_multi_patch.py
    │
    ├── service_train.py
    │       │
    │       ├── Distributional PPO (distributional_ppo.py)
    │       │       │
    │       │       ├── Custom Policy (custom_policy_patch1.py)
    │       │       ├── VGS (variance_gradient_scaler.py)
    │       │       └── AdaptiveUPGD (optimizers/adaptive_upgd.py)
    │       │
    │       └── Vector Environment
    │               │
    │               └── Trading Environment (trading_patchnew.py)
    │
    ├── PBT Scheduler (adversarial/pbt_scheduler.py) [optional]
    │
    └── Data Loader (data_loader_multi_asset.py)
```

### 3. Backtest Service

```
script_backtest.py
    │
    └── service_backtest.py
            │
            ├── Trading Environment (trading_patchnew.py)
            │
            ├── Execution Simulation (execution_sim.py)
            │       │
            │       ├── impl_fees.py
            │       ├── impl_slippage.py
            │       └── impl_latency.py
            │
            └── Offline Data (impl_offline_data.py)
```

---

## External Dependencies

### Exchange APIs

```
┌───────────────────────────────────────────────────────────────┐
│                     EXCHANGE ADAPTERS                          │
├───────────────────┬───────────────────┬───────────────────────┤
│      Binance      │      Alpaca       │      Polygon          │
├───────────────────┼───────────────────┼───────────────────────┤
│ REST API          │ REST API          │ REST API              │
│ - /api/v3/*       │ - /v2/positions   │ - /v2/aggs/*          │
│ - /fapi/v1/*      │ - /v2/orders      │ - /v2/tickers/*       │
│                   │ - /v2/account     │                       │
├───────────────────┼───────────────────┼───────────────────────┤
│ WebSocket         │ WebSocket         │ WebSocket             │
│ - wss://stream.*  │ - wss://stream.*  │ - wss://socket.*      │
├───────────────────┼───────────────────┼───────────────────────┤
│ Rate Limits       │ Rate Limits       │ Rate Limits           │
│ - 1200 req/min    │ - 200 req/min     │ - Varies by plan      │
└───────────────────┴───────────────────┴───────────────────────┘
```

### Data Files

```
data/
├── binance_filters.json     ← Exchange filters (fetch_binance_filters.py)
├── fees/
│   └── fees_by_symbol.json  ← Fee schedules (refresh_fees.py)
├── universe/
│   ├── symbols.json         ← Crypto universe
│   └── alpaca_symbols.json  ← Stock universe
├── latency/
│   └── liquidity_latency_seasonality.json  ← Latency profiles
├── raw_stocks/
│   └── *.parquet            ← Stock historical data
└── train/
    └── *.csv                ← Training data
```

### State Files

```
state/
├── kill_switch.flag         ← Emergency stop flag
├── kill_switch_state.json   ← Kill switch counters
├── ttl_state.json           ← TTL tracking
└── positions.json           ← Position tracking
```

---

## Data Flow

### Live Trading Data Flow

```
                    ┌──────────────────┐
                    │  Exchange API    │
                    └────────┬─────────┘
                             │ market data
                             ▼
                    ┌──────────────────┐
                    │  Market Data     │
                    │  Adapter         │
                    └────────┬─────────┘
                             │ OHLCV bars
                             ▼
┌──────────────────┐ ┌──────────────────┐
│  Feature         │ │  Risk Guard     │
│  Pipeline        │ │                 │
└────────┬─────────┘ └────────┬────────┘
         │ features           │ limits
         ▼                    ▼
┌──────────────────┐ ┌──────────────────┐
│  Policy          │ │  Kill Switch    │
│  (PPO Model)     │ │                 │
└────────┬─────────┘ └────────┬────────┘
         │ action             │ halt?
         ▼                    ▼
┌──────────────────────────────────────┐
│          Signal Runner               │
└────────┬─────────────────────────────┘
         │ order
         ▼
┌──────────────────┐
│  Order Executor  │──────► Exchange
└──────────────────┘
```

### Training Data Flow

```
┌──────────────────┐
│  Data Files      │
│  (CSV/Parquet)   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Data Loader     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────┐
│  Vector Env      │ ──► │  Rollout     │
│  (N parallel)    │     │  Buffer      │
└────────┬─────────┘     └──────┬───────┘
         │                      │
         ▼                      ▼
┌──────────────────┐     ┌──────────────┐
│  Feature         │     │  PPO Update  │
│  Pipeline        │     │  (gradient)  │
└──────────────────┘     └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │  Model       │
                         │  Checkpoint  │
                         └──────────────┘
```

---

## Component Dependencies Matrix

| Component | Depends On | Depended By |
|-----------|------------|-------------|
| **core_config** | - | Everything |
| **core_models** | core_config | impl_*, service_* |
| **impl_fees** | core_* | execution_sim |
| **impl_slippage** | core_* | execution_sim |
| **impl_latency** | core_* | execution_sim |
| **execution_sim** | impl_*, core_* | trading_patchnew |
| **features_pipeline** | core_* | mediator |
| **mediator** | core_*, features_pipeline | trading_patchnew |
| **trading_patchnew** | mediator, execution_sim | service_* |
| **risk_guard** | core_* | service_signal_runner |
| **ops_kill_switch** | - | service_signal_runner |
| **distributional_ppo** | custom_policy_patch1, VGS | service_train |
| **custom_policy_patch1** | - | distributional_ppo |
| **variance_gradient_scaler** | - | distributional_ppo |
| **service_backtest** | trading_patchnew | script_backtest |
| **service_train** | distributional_ppo, trading_patchnew | train_model_multi_patch |
| **service_signal_runner** | trading_patchnew, risk_guard | script_live |

---

## Adapter Registry

```
adapters/
├── registry.py              ← Factory + registration
├── base.py                  ← Abstract base classes
├── models.py                ← Exchange-agnostic models
├── config.py                ← Pydantic configuration
│
├── binance/                 ← Crypto
│   ├── market_data.py       │
│   ├── fees.py              ├── Implements BaseAdapter
│   ├── trading_hours.py     │
│   └── exchange_info.py     │
│
├── alpaca/                  ← US Equities
│   ├── market_data.py       │
│   ├── order_execution.py   ├── Implements BaseAdapter
│   ├── fees.py              │
│   └── exchange_info.py     │
│
├── polygon/                 ← US Equities (data only)
│   ├── market_data.py       │
│   └── exchange_info.py     ├── Implements BaseAdapter
│
└── yahoo/                   ← Indices/Macro
    └── market_data.py       └── Implements BaseAdapter
```

Usage:

```python
from adapters.registry import create_market_data_adapter

# Factory pattern
adapter = create_market_data_adapter("binance")
adapter = create_market_data_adapter("alpaca", {"api_key": "..."})
```

---

## Configuration Hierarchy

```
configs/
├── config_train.yaml        ← Main training config
│   ├── imports: model, env, data settings
│   └── overrides specific values
│
├── config_live.yaml         ← Main live config
│   ├── imports: exchange, risk settings
│   └── overrides for live trading
│
├── Modular configs (can be imported):
│   ├── execution.yaml       ← Execution parameters
│   ├── fees.yaml            ← Fee structures
│   ├── slippage.yaml        ← Slippage profiles
│   ├── risk.yaml            ← Risk limits
│   ├── no_trade.yaml        ← No-trade windows
│   └── exchange.yaml        ← Exchange settings
│
└── examples/                ← Well-documented examples
    ├── example_train_crypto.yaml
    ├── example_live_crypto.yaml
    ├── example_train_stocks.yaml
    └── example_live_stocks.yaml
```

---

## Health Check Points

For monitoring system health, check these components:

| Component | Health Check | Critical? |
|-----------|--------------|-----------|
| Exchange API | Ping endpoint | Yes |
| Market Data | Last update < 60s | Yes |
| Order Executor | No pending orders > 5min | Yes |
| Feature Pipeline | No NaN in features | Yes |
| Risk Guard | Limits not breached | Yes |
| Kill Switch | Not triggered | Yes |
| Clock Sync | Drift < 500ms | Yes |
| Disk Space | > 1GB free | No |
| Memory | < 80% usage | No |
| CPU | < 90% usage | No |

---

## Failure Domains

```
┌─────────────────────────────────────────────────────────────────┐
│                    FAILURE DOMAIN: NETWORK                       │
│  Exchange API, WebSocket, DNS                                   │
│  Impact: No market data, no order execution                     │
│  Recovery: Retry, fallback, graceful degradation                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    FAILURE DOMAIN: COMPUTE                       │
│  CPU, Memory, Disk                                              │
│  Impact: Slow processing, crashes                               │
│  Recovery: Resource limits, auto-restart, scaling               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    FAILURE DOMAIN: DATA                          │
│  Market data, feature computation, model inference              │
│  Impact: Bad signals, wrong positions                           │
│  Recovery: Validation, fallbacks, kill switch                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    FAILURE DOMAIN: STATE                         │
│  Position tracking, order management, PnL                       │
│  Impact: Position mismatch, incorrect risk                      │
│  Recovery: Reconciliation, state sync, manual intervention      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Startup Sequence

```
1. Load configuration
   └── Validate all required fields

2. Initialize logging
   └── Set up secure logging (API key masking)

3. Run doctor checks
   └── Verify environment, dependencies, data files

4. Initialize adapters
   ├── Market data adapter
   ├── Order execution adapter (if live)
   └── Validate API connectivity

5. Initialize components
   ├── Feature pipeline
   ├── Risk guard
   ├── Kill switch
   └── Healthcheck (if enabled)

6. Load model (if applicable)
   └── Verify checkpoint compatibility

7. Initialize trading environment
   └── Load historical data

8. Start main loop
   ├── Market data subscription
   ├── Signal generation
   └── Order execution
```

---

## Shutdown Sequence

```
1. Receive shutdown signal (SIGTERM, SIGINT, or kill switch)

2. Stop accepting new signals
   └── Set shutdown flag

3. Cancel pending orders
   └── Wait for confirmations

4. Close positions (optional, configurable)
   └── Market orders for immediate exit

5. Flush state
   ├── Save position state
   ├── Save metrics
   └── Close file handles

6. Disconnect adapters
   ├── Close WebSocket connections
   └── Cancel pending requests

7. Log shutdown summary
   └── Final P&L, position count, etc.

8. Exit process
   └── Return appropriate exit code
```

---

*Last updated: 2025-12-03*
