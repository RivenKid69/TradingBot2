# AI-Powered Quantitative Research Platform Codebase Comprehensive Structure Analysis

## Project Overview
AI-Powered Quantitative Research Platform is a complex trading bot system written in Python with Rust/C++ integrations. It's a medium-frequency algorithmic trading system with components for simulation, backtesting, live trading, and machine learning model training. The codebase uses a layered architecture with dependency injection patterns.

**Total Python Files**: 292 in root + 119 in subdirectories = ~411 total
**Total Lines of Code**: ~116,894 lines (root level alone)
**Key Technologies**: Pydantic for configuration, async I/O, NumPy/Pandas for data processing, FastAPI/Streamlit for UI, Stable-Baselines3 for RL training

---

## 1. MAIN DIRECTORIES AND PURPOSES

### Core Directories

#### Root Level (292 .py files)
Main working directory containing majority of business logic and utilities.

#### /configs (32+ YAML/JSON files)
- **Purpose**: Configuration management for different run modes
- **Key Files**:
  - `config_sim.yaml` - Simulation configuration
  - `config_live.yaml` - Live trading configuration
  - `config_train.yaml` - Model training configuration
  - `config_eval.yaml` - Strategy evaluation configuration
  - `config_template.yaml` - Template with all possible options
  - `ops.yaml`, `ops.json` - Operational settings
  - `slippage.yaml` - Slippage model configuration
  - `execution.yaml` - Execution parameters
  - `fees.yaml` - Trading fee configuration
  - `risk.yaml` - Risk management settings
  - `no_trade.yaml` - No-trade period definitions
  - `monitoring.yaml` - Monitoring and alerts configuration
  - `seasonality/archive/` - Historical seasonality multipliers

#### /scripts (32 .py files)
**Purpose**: Standalone utility scripts for data processing, building models, and analysis
**Key Scripts**:
- `build_adv.py` - Build Average Daily Volume statistics
- `build_spread_seasonality.py` - Build spread seasonality patterns
- `calibrate_live_slippage.py` - Calibrate slippage models from live trades
- `extract_liquidity_seasonality.py` - Extract hourly/intraday seasonality
- `fetch_binance_filters.py` - Download exchange filters and specifications
- `refresh_universe.py` - Update symbol universe from exchange
- `refresh_fees.py` - Update trading fee information
- `run_seasonality_pipeline.py` - Full seasonality computation pipeline
- `validate_seasonality.py` - Validate seasonality data
- `sim_reality_check.py` - Verify simulation matches reality
- `run_full_cycle.py` - End-to-end data update and inference cycle

#### /services (16 .py files)
**Purpose**: Core business logic services and application features
**Key Services**:
- `monitoring.py` - Metrics, logging, alerting (66KB)
- `rest_budget.py` - REST API rate limiting and budget management (66KB)
- `state_storage.py` - Persistent state management
- `signal_bus.py` - Signal publishing and event distribution
- `metrics.py` - Performance metrics calculation
- `costs.py` - Trading cost tracking and analysis
- `event_bus.py` - Event system infrastructure
- `alerts.py` - Alert generation and notification
- `ops_kill_switch.py` - Emergency trading halt mechanism
- `universe.py` - Symbol management
- `retry.py` - Retry logic for network operations
- `shutdown.py` - Graceful shutdown handling
- `signal_csv_writer.py` - Signal logging to CSV
- `utils_app.py` - Application-level utilities

#### /strategies (3 .py files)
**Purpose**: Trading strategy implementations
**Files**:
- `base.py` - Base strategy class with decision-making protocol
- `momentum.py` - Momentum-based trading strategy

#### /domain (2 .py files)
**Purpose**: Domain models and adapters
**Files**:
- `adapters.py` - Adapter patterns for external integrations
- `__init__.py`

#### /api (3 .py files)
**Purpose**: API integrations
**Files**:
- `spot_signals.py` - Spot market signal formatting (JSON schemas)
- `config.py` - API configuration

#### /adapters (1 .py file)
**Purpose**: Exchange adapters
**Files**:
- `binance_spot_private.py` - Binance spot trading private API wrapper

#### /wrappers (2 .py files)
**Purpose**: Environment and action wrappers
**Files**:
- `action_space.py` - Action space definitions for RL environments

#### /utils (8 .py files)
**Purpose**: Utility functions and helpers
**Key Utils**:
- `time.py` - Time conversion utilities
- `model_io.py` - Model serialization/deserialization
- `time_provider.py` - Mock time provider for testing
- `rate_limiter.py` - Rate limiting utilities
- `prometheus.py` - Prometheus metrics integration
- `moving_average.py` - Moving average calculation

#### /data (16+ files/subdirs)
**Purpose**: Market data, configurations, and datasets
**Structure**:
- `universe/symbols.json` - Symbol universe definition
- `adv/` - Average Daily Volume data
- `fees/fees_by_symbol.json` - Fee structure by symbol
- `latency/liquidity_latency_seasonality.json` - Latency seasonality
- `*.csv` - Sample data files (hist_trades, sim_trades, benchmark_equity)
- `impact_benchmark.json` - Impact benchmark data

#### /docs (25+ .md files + JSON schemas)
**Purpose**: Comprehensive documentation
**Key Documents**:
- `seasonality.md` - Seasonality methodology
- `bar_execution.md` - Bar-based execution details
- `parkinson_volatility.md` - Volatility calculation
- `yang_zhang_volatility.md` - Yang-Zhang volatility estimator
- `no_trade.md` - No-trade period logic
- `pipeline.md` - Decision pipeline architecture
- JSON Schemas: `spot_signal_envelope.schema.json`, `spot_signal_target_weight.schema.json`, `spot_signal_delta_weight.schema.json`

#### /backtest (4 .py files)
**Purpose**: Backtesting infrastructure tests

#### /execution (8 .py files)
**Purpose**: Execution simulator tests and utilities

#### /features (1 .py file)
**Purpose**: Feature engineering pipeline tests

#### /guards (3 .py files)
**Purpose**: Risk management and trade guards tests

#### /pipeline (3 .py files)
**Purpose**: Feature pipeline tests

#### /monitoring (1 .py file)
**Purpose**: Bar execution snapshot tests

#### /tools (2 .py files)
**Purpose**: Gradient checking and utility tools
- `grad_sanity.py` - Gradient sanity checks for model training

#### /tests (150+ .py files)
**Purpose**: Comprehensive test suite covering:
- Environment and wrapper tests
- Signal runner tests
- Execution tests
- Feature engineering tests
- Risk management tests
- Cost and slippage tests
- Model calibration tests

#### /include (3 .h files)
**Purpose**: C++ headers for performance-critical code
- `execevents_types.h` - Execution event types
- `latency_queue.h` - Latency queue implementation

#### /artifacts (subdirectories with training results)
**Purpose**: ML training artifacts and outputs
- `default-run/` - Default training run artifacts
- `training_summary.json` - Training statistics

#### /cache (directories for caching)
**Purpose**: Runtime caching

#### /state (4 JSON files)
**Purpose**: Persistent application state
- `ops_state.json` - Operations state
- `no_trade_state.json` - No-trade restrictions
- `last_bar_seen.json` - Last bar tracking

#### /benchmarks (1 JSON file)
**Purpose**: Benchmark thresholds and KPIs

#### /audits (directory)
**Purpose**: Audit logs and reports

#### /sandbox (directory)
**Purpose**: Sandbox/testing environment

#### /notebooks (directory)
**Purpose**: Jupyter notebooks for analysis

---

## 2. KEY PYTHON MODULES AND NAMING CONVENTIONS

### Module Naming Conventions

The codebase follows a strict layered architecture with prefixes indicating module purpose:

#### `core_` Modules (7 files) - 3,411 lines
Base entities, contracts, and models - no external dependencies
- `core_config.py` (1,382 lines) - Pydantic configuration models, DI component specs, ClockSyncConfig
- `core_models.py` (516 lines) - Domain models: Side, OrderType, TimeInForce, Instrument, Bar, Tick, Order, Position, TradeLogRow
- `core_contracts.py` (141 lines) - Abstract protocols/interfaces: MarketDataSource, Executor, SignalPolicy, RiskGuard
- `core_strategy.py` (~85 lines) - Strategy protocol and deprecated Decision class
- `core_events.py` - Event types and EventType enum
- `core_errors.py` - Custom error types
- `core_constants.py` - Global constants

#### `impl_` Modules (9 files) - 10,443 lines
Concrete infrastructure implementations - depends only on `core_`
- `impl_sim_executor.py` (1,424 lines) - Simulation execution engine
- `impl_bar_executor.py` (1,685 lines) - Bar-based order execution
- `impl_slippage.py` (2,395 lines) - Slippage modeling
- `impl_fees.py` (1,684 lines) - Trading fee handling
- `impl_latency.py` (1,117 lines) - Network latency simulation
- `impl_quantizer.py` (883 lines) - Price/quantity quantization to exchange rules
- `impl_offline_data.py` (294 lines) - CSV/Parquet market data source
- `impl_binance_public.py` (248 lines) - Binance public API wrapper
- `impl_risk_basic.py` (162 lines) - Basic risk guard implementation

#### `service_` Modules (7 files) - 14,763 lines
Higher-level services - depends on `core_` and `impl_`
- `service_signal_runner.py` (9,578 lines) - Main real-time signal generation service
- `service_backtest.py` (2,054 lines) - Backtesting service
- `service_train.py` (218 lines) - ML training service
- `service_eval.py` (339 lines) - Strategy evaluation service
- `service_calibrate_slippage.py` (142 lines) - Slippage calibration service
- `service_calibrate_tcost.py` (263 lines) - Transaction cost calibration
- `service_fetch_exchange_specs.py` (451 lines) - Exchange specification fetcher

#### `script_` Module Naming (7 files)
CLI entry points - can use any layer
- `script_live.py` - Run live trading
- `script_backtest.py` - Run backtesting
- `script_eval.py` - Evaluate strategy
- `script_calibrate_slippage.py` - Calibrate slippage model
- `script_calibrate_tcost.py` - Calibrate transaction costs
- `script_fetch_exchange_specs.py` - Fetch exchange specifications
- `script_compare_runs.py` - Compare multiple run results

### Infrastructure Modules (Standalone, ~60 files)
Key standalone modules without strict prefix:

**Data & Exchange Integration**:
- `binance_public.py` - Binance public API client
- `binance_ws.py` - WebSocket feeds from Binance
- `binance_fee_refresh.py` - Fee refresh mechanism
- `exchangespecs.py` - Exchange specifications handling

**Execution & Pricing**:
- `execution_sim.py` (550KB!) - Complex execution simulator with LOB and intrabar pricing
- `execution_algos.py` - VWAP, POV, and other execution algorithms

**ML Training**:
- `distributional_ppo.py` (444KB) - Distributional PPO algorithm implementation
- `train_model_multi_patch.py` - Multi-worker training orchestrator
- `custom_policy_patch1.py` - Custom policy patches

**Configuration & DI**:
- `config.py` - Environment-specific configurations
- `di_registry.py` - Dependency injection container
- `di_stubs.py` - DI stubs for testing

**Utilities & Helpers**:
- `clock.py` - System clock synchronization with Binance
- `action_proto.py` - Action protocol and legacy compatibility
- `adv_store.py` - Average Daily Volume storage
- `calibration.py` - Calibration utilities
- `compat_shims.py` - Compatibility shims for legacy code
- `dynamic_no_trade_guard.py` - Dynamic no-trade period management
- `drift.py` - Model drift detection

**Data Processing**:
- `build_adv.py`, `build_adv_base.py` - ADV calculation
- `build_training_table.py` - Training data preparation
- `apply_calibrator.py` - Apply calibration models
- `apply_no_trade_mask.py` - Apply no-trade restrictions
- `aggregate_exec_logs.py` - Execution log aggregation
- `agg_klines.py` - Aggregate klines/bars

**Analysis & Validation**:
- `check_drift.py` - Check model drift
- `check_feature_parity.py` - Verify feature consistency
- `data_validation.py` - Data validation rules
- `compare_slippage_curve.py` - Slippage curve comparison
- `coreprice_scale.py` - Price scaling utilities

**Application Layer**:
- `app.py` (180KB) - Streamlit/FastAPI web application
- `event_bus.py` - Event publishing system (also in services/)
- `watchdog_vec_env.py` - Environment watchdog

### Test Modules (150+ files)
Test naming patterns:
- `test_*_*.py` - Focused unit/integration tests
- Test categories cover all layers and modules

---

## 3. MAIN ENTRY POINTS (script_*.py files)

### Primary Entry Points

1. **script_live.py**
   - Launches live trading via `service_signal_runner`
   - Supports runtime parameter overrides
   - Handles signal generation and order placement

2. **script_backtest.py**
   - Runs historical backtest via `service_backtest`
   - Supports multi-regime evaluation
   - Generates performance metrics and PnL

3. **script_eval.py**
   - Evaluates strategy performance
   - Runs reality checks against historical trades
   - Produces equity curves and metrics

4. **script_calibrate_slippage.py**
   - Builds slippage models from execution data
   - Validates models against reality

5. **script_calibrate_tcost.py**
   - Calibrates transaction cost models
   - Uses historical execution logs

6. **script_fetch_exchange_specs.py**
   - Downloads Binance spot/futures specifications
   - Updates filter parameters and commission tiers

7. **script_compare_runs.py**
   - Compares metrics across multiple backtests
   - Generates comparison tables (CSV/stdout)

### Secondary Entry Points

8. **train_model_multi_patch.py**
   - Multi-worker distributed RL training
   - Supports regime-based training splits
   - Manages checkpoints and model artifacts

9. **update_and_infer.py**
   - End-to-end feature update and signal inference
   - Supports scheduled/loop execution
   - Environment variable configuration

10. **scripts/run_full_cycle.py**
    - Complete data ingestion → feature preparation → inference pipeline
    - Supports symbol list and date ranges
    - Wraps orchestrator and infer components

---

## 4. CONFIGURATION FILES AND LOCATIONS

### Primary Configuration Paths

```
/home/user/AI-Powered Quantitative Research Platform/configs/
├── config_sim.yaml          # Simulation run config (20KB)
├── config_live.yaml         # Live trading config (6KB)
├── config_train.yaml        # Training config (17KB)
├── config_eval.yaml         # Evaluation config (3.5KB)
├── config_template.yaml     # Template with all options (20KB)
├── config_train_spot_bar.py # Spot trading config
│
├── execution.yaml           # Execution parameters
├── slippage.yaml            # Slippage model config (9KB)
├── slippage_calibrate.yaml  # Slippage calibration config
├── fees.yaml                # Trading fee structure (7KB)
│
├── risk.yaml                # Risk management settings
├── no_trade.yaml            # No-trade periods (1.8KB)
├── quantizer.yaml           # Price quantization rules
│
├── monitoring.yaml          # Metrics and alerts (1.3KB)
├── state.yaml               # State management config
├── ops.yaml                 # Operations settings (3.4KB)
├── ops.json                 # Operations state
├── rest_budget.yaml         # API rate limiting (506 bytes)
│
├── timing.yaml              # Timing configurations
├── adv.yaml                 # ADV calculation settings
├── offline.yaml             # Offline data processing (3.4KB)
│
├── signal_quality.yaml      # Signal quality filters
├── signals.yaml             # Signal generation settings
│
├── liquidity_latency_seasonality.json     # Latency seasonality data
├── liquidity_seasonality.json             # Symlink to above
├── market_regimes.json                    # Market regime definitions
├── reference_regime_distributions.json    # Reference distributions
│
├── legacy_sim.yaml          # Legacy simulation config
├── legacy_live.yaml         # Legacy live config
├── legacy_eval.yaml         # Legacy eval config
│
└── seasonality/
    └── archive/             # Archived seasonality versions
```

### Configuration Architecture

- **YAML Format**: Primary configuration files
- **JSON Format**: Data files (regimes, distributions, seasonality)
- **Pydantic Models**: Validation and type safety in `core_config.py`
- **DI Components**: Component specifications with dotted paths (e.g., `impl_offline_data:OfflineCSVBarSource`)
- **Environment Variables**: Support for runtime overrides (SYMS, LOOP, SLEEP_MIN, etc.)

---

## 5. PROJECT STRUCTURE MAPPING

### Layered Architecture (as per ARCHITECTURE.md)

```
┌─────────────────────────────────────┐
│ scripts_* / tool scripts            │ ← Entry points, orchestration
├─────────────────────────────────────┤
│ service_* (16 services)             │ ← Business logic services
├─────────────────────────────────────┤
│ impl_* (9 implementations)          │ ← Concrete implementations
├─────────────────────────────────────┤
│ core_* (7 core modules)             │ ← Base models, contracts
└─────────────────────────────────────┘

Additional Layers:
- strategies/ → Trading algorithms using services
- domain/ → Domain adapters and models
- api/ → API integrations
- adapters/ → Exchange adapters
```

### Dependency Flow (Permitted)
```
core_ → impl_ → service_ → strategies → scripts_
```

### Module Categories by Function

**Market Data**:
- impl_offline_data.py - CSV/Parquet loading
- binance_public.py - Binance API
- binance_ws.py - WebSocket feeds
- impl_latency.py - Latency simulation

**Execution**:
- execution_sim.py - Main simulator
- impl_bar_executor.py - Bar-based execution
- impl_sim_executor.py - Simple simulator
- execution_algos.py - VWAP, POV algorithms
- adapters/binance_spot_private.py - Live API

**Pricing & Slippage**:
- impl_slippage.py - Slippage models
- impl_fees.py - Fee calculations
- impl_quantizer.py - Exchange quantization
- coreprice_scale.py - Price scaling

**Risk Management**:
- impl_risk_basic.py - Basic guards
- dynamic_no_trade_guard.py - Dynamic restrictions
- guards/ - Guard implementations

**Training & ML**:
- distributional_ppo.py - RL algorithm
- train_model_multi_patch.py - Training orchestrator
- custom_policy_patch1.py - Policy patches

**State & Persistence**:
- services/state_storage.py - State management
- adv_store.py - ADV caching

**Monitoring & Alerts**:
- services/monitoring.py - Metrics collection
- services/alerts.py - Alert generation
- services/metrics.py - Performance metrics
- services/costs.py - Cost tracking

**Signaling**:
- services/signal_bus.py - Event distribution
- services/signal_csv_writer.py - Signal logging
- api/spot_signals.py - Signal formats

---

## 6. KEY FILE STATISTICS

### Largest Files (by complexity/functionality)
1. **execution_sim.py** - 550KB, 12,993 lines - Main execution simulator
2. **app.py** - 180KB, 4,500+ lines - Web application (Streamlit/FastAPI)
3. **distributional_ppo.py** - 444KB, 9,700+ lines - Distributional PPO RL algorithm
4. **service_signal_runner.py** - 9,578 lines - Core signal generation service
5. **custom_policy_patch1.py** - 63KB - Policy patch for RL models

### Most Complex Services
- service_signal_runner.py (9,578 lines)
- service_backtest.py (2,054 lines)
- impl_slippage.py (2,395 lines)
- impl_bar_executor.py (1,685 lines)
- impl_fees.py (1,684 lines)

### Configuration Complexity
- 32+ YAML configuration files
- 13 JSON data configuration files
- 150+ test configuration scenarios
- Environment variable overrides supported

---

## 7. DATA FLOW ARCHITECTURE

### Signal Generation Pipeline (Live)
```
script_live.py
  ↓
service_signal_runner.py
  ├─ Market data feed (binance_ws.py / binance_public.py)
  ├─ Feature pipeline (feature processing)
  ├─ Policy execution (strategies/)
  ├─ Risk guards (dynamic_no_trade_guard.py, etc.)
  ├─ Order execution (execution_sim.py / adapters/binance_spot_private.py)
  ├─ Signal publishing (services/signal_bus.py)
  └─ Monitoring (services/monitoring.py)
```

### Backtesting Pipeline
```
script_backtest.py
  ↓
service_backtest.py
  ├─ Offline data source (impl_offline_data.py)
  ├─ Feature pipeline
  ├─ Policy execution
  ├─ Risk guards
  ├─ Execution simulator (execution_sim.py)
  ├─ Metrics calculation (services/metrics.py)
  └─ Artifact generation
```

### Training Pipeline
```
train_model_multi_patch.py
  ├─ Data preparation (build_training_table.py, apply_calibrator.py)
  ├─ Multi-worker training (distributional_ppo.py)
  ├─ Checkpoint management
  ├─ Model validation
  └─ Artifact storage
```

---

## 8. CONFIGURATION MANAGEMENT

### Config Loading Flow
1. YAML files in /configs/ parsed by Pydantic models
2. Component specifications loaded via DI (dotted path resolution)
3. Runtime overrides applied via CLI arguments
4. Environment variables merge with file config
5. Resulting config object validated and passed to services

### Example Config Structure (YAML)
```yaml
mode: sim|live|train|eval
components:
  market_data:
    target: impl_offline_data:OfflineCSVBarSource
    params: {paths: [...], timeframe: "1m"}
  executor:
    target: impl_sim_executor:SimExecutor
    params: {...}
  policy:
    target: strategies.momentum:MomentumStrategy
    params: {...}
data:
  timeframe: "1m"
  symbols: ["BTCUSDT", "ETHUSDT"]
execution:
  mode: bar | intrabar
  bar_price: open | close | mid
feature_pipe:
  window: 60
```

---

## 9. DEPENDENCIES AND EXTERNAL INTEGRATIONS

### Market Data Sources
- Binance REST API (binance_public.py)
- Binance WebSocket (binance_ws.py)
- CSV/Parquet files (impl_offline_data.py)

### Infrastructure
- Pydantic (configuration validation)
- FastAPI/Streamlit (web UI)
- Stable-Baselines3 (RL algorithms)
- Pandas/NumPy (data processing)
- asyncio (async I/O)

### Exchange Integration
- Binance Spot/Futures APIs
- Exchange specifications and filter validation
- Real-time quote updates

---

## 10. TEST COVERAGE

**Test Files**: 150+ test modules
**Categories**:
- Unit tests for each module
- Integration tests for services
- End-to-end tests for full pipelines
- Performance/stress tests
- Compatibility tests (legacy code)
- Reality checks against historical data

**Key Test Patterns**:
- `test_service_*.py` - Service tests
- `test_*_integration.py` - Integration tests
- `test_*_smoke.py` - Smoke tests

