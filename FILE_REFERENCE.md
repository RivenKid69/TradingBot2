# TradingBot2 - Complete File Path Reference

## CORE MODULES (core_*.py)
- /home/user/TradingBot2/core_config.py - Configuration models and DI specs (1,382 lines)
- /home/user/TradingBot2/core_models.py - Domain models (Side, Order, Bar, Tick) (516 lines)
- /home/user/TradingBot2/core_contracts.py - Abstract protocols/interfaces (141 lines)
- /home/user/TradingBot2/core_strategy.py - Strategy interface and Decision class (85 lines)
- /home/user/TradingBot2/core_events.py - Event types and enums
- /home/user/TradingBot2/core_errors.py - Custom exception classes
- /home/user/TradingBot2/core_constants.py - Global constants

## IMPLEMENTATION MODULES (impl_*.py)
- /home/user/TradingBot2/impl_sim_executor.py - Basic execution simulator (1,424 lines)
- /home/user/TradingBot2/impl_bar_executor.py - Bar-based order execution (1,685 lines)
- /home/user/TradingBot2/impl_slippage.py - Slippage model calculations (2,395 lines)
- /home/user/TradingBot2/impl_fees.py - Trading fee handling (1,684 lines)
- /home/user/TradingBot2/impl_latency.py - Network latency simulation (1,117 lines)
- /home/user/TradingBot2/impl_quantizer.py - Price/qty quantization (883 lines)
- /home/user/TradingBot2/impl_offline_data.py - CSV/Parquet data loading (294 lines)
- /home/user/TradingBot2/impl_binance_public.py - Binance REST API (248 lines)
- /home/user/TradingBot2/impl_risk_basic.py - Basic risk guards (162 lines)

## SERVICE MODULES (service_*.py)
- /home/user/TradingBot2/service_signal_runner.py - Live signal generation (9,578 lines)
- /home/user/TradingBot2/service_backtest.py - Backtesting service (2,054 lines)
- /home/user/TradingBot2/service_train.py - ML training service (218 lines)
- /home/user/TradingBot2/service_eval.py - Strategy evaluation (339 lines)
- /home/user/TradingBot2/service_calibrate_slippage.py - Slippage calibration (142 lines)
- /home/user/TradingBot2/service_calibrate_tcost.py - Transaction cost calibration (263 lines)
- /home/user/TradingBot2/service_fetch_exchange_specs.py - Fetch Binance specs (451 lines)

## SCRIPT ENTRY POINTS (script_*.py)
- /home/user/TradingBot2/script_live.py - Launch live trading
- /home/user/TradingBot2/script_backtest.py - Run backtesting
- /home/user/TradingBot2/script_eval.py - Evaluate strategy
- /home/user/TradingBot2/script_calibrate_slippage.py - Calibrate slippage
- /home/user/TradingBot2/script_calibrate_tcost.py - Calibrate transaction costs
- /home/user/TradingBot2/script_fetch_exchange_specs.py - Fetch exchange specs
- /home/user/TradingBot2/script_compare_runs.py - Compare backtesting runs

## MAIN INFRASTRUCTURE MODULES
- /home/user/TradingBot2/execution_sim.py - Advanced execution simulator (550KB, 12,993 lines)
- /home/user/TradingBot2/app.py - Web application (Streamlit/FastAPI) (180KB, 4,500+ lines)
- /home/user/TradingBot2/distributional_ppo.py - PPO RL algorithm (444KB, 9,700+ lines)
- /home/user/TradingBot2/train_model_multi_patch.py - Training orchestrator
- /home/user/TradingBot2/binance_public.py - Binance public API client
- /home/user/TradingBot2/binance_ws.py - Binance WebSocket feeds
- /home/user/TradingBot2/binance_fee_refresh.py - Fee refresh mechanism
- /home/user/TradingBot2/exchangespecs.py - Exchange specifications
- /home/user/TradingBot2/execution_algos.py - VWAP, POV algorithms
- /home/user/TradingBot2/clock.py - Clock synchronization with Binance
- /home/user/TradingBot2/action_proto.py - Action protocol & legacy compatibility
- /home/user/TradingBot2/di_registry.py - Dependency injection container
- /home/user/TradingBot2/di_stubs.py - DI stubs for testing
- /home/user/TradingBot2/config.py - Environment configurations

## SERVICES DIRECTORY (/services)
- /home/user/TradingBot2/services/__init__.py
- /home/user/TradingBot2/services/monitoring.py - Metrics & alerts (64KB)
- /home/user/TradingBot2/services/rest_budget.py - REST rate limiting (66KB)
- /home/user/TradingBot2/services/state_storage.py - State persistence (32KB)
- /home/user/TradingBot2/services/signal_bus.py - Signal distribution (11KB)
- /home/user/TradingBot2/services/metrics.py - Performance metrics (16KB)
- /home/user/TradingBot2/services/costs.py - Cost tracking (12KB)
- /home/user/TradingBot2/services/event_bus.py - Event system (12KB)
- /home/user/TradingBot2/services/alerts.py - Alert generation (4KB)
- /home/user/TradingBot2/services/ops_kill_switch.py - Trading halt (7KB)
- /home/user/TradingBot2/services/universe.py - Symbol management (5KB)
- /home/user/TradingBot2/services/retry.py - Retry logic (5KB)
- /home/user/TradingBot2/services/shutdown.py - Shutdown handling (5KB)
- /home/user/TradingBot2/services/signal_csv_writer.py - Signal logging (9KB)
- /home/user/TradingBot2/services/utils_app.py - App utilities (9KB)
- /home/user/TradingBot2/services/utils_config.py - Config utilities (1KB)
- /home/user/TradingBot2/services/utils_sandbox.py - Sandbox utilities (1KB)

## STRATEGIES DIRECTORY (/strategies)
- /home/user/TradingBot2/strategies/__init__.py
- /home/user/TradingBot2/strategies/base.py - Base strategy class (9.5KB)
- /home/user/TradingBot2/strategies/momentum.py - Momentum strategy (7.2KB)

## UTILS DIRECTORY (/utils)
- /home/user/TradingBot2/utils/__init__.py
- /home/user/TradingBot2/utils/time.py - Time utilities
- /home/user/TradingBot2/utils/model_io.py - Model I/O
- /home/user/TradingBot2/utils/time_provider.py - Mock time provider
- /home/user/TradingBot2/utils/rate_limiter.py - Rate limiting
- /home/user/TradingBot2/utils/prometheus.py - Prometheus integration
- /home/user/TradingBot2/utils/moving_average.py - Moving average

## API DIRECTORY (/api)
- /home/user/TradingBot2/api/__init__.py
- /home/user/TradingBot2/api/spot_signals.py - Spot market signals
- /home/user/TradingBot2/api/config.py - API configuration

## DOMAIN DIRECTORY (/domain)
- /home/user/TradingBot2/domain/__init__.py
- /home/user/TradingBot2/domain/adapters.py - Domain adapters

## ADAPTERS DIRECTORY (/adapters)
- /home/user/TradingBot2/adapters/binance_spot_private.py - Binance private API

## WRAPPERS DIRECTORY (/wrappers)
- /home/user/TradingBot2/wrappers/__init__.py
- /home/user/TradingBot2/wrappers/action_space.py - Action space definitions

## CONFIGURATION FILES (/configs)
### Main Configurations
- /home/user/TradingBot2/configs/config_sim.yaml - Simulation config (20KB)
- /home/user/TradingBot2/configs/config_live.yaml - Live trading config (6KB)
- /home/user/TradingBot2/configs/config_train.yaml - Training config (17KB)
- /home/user/TradingBot2/configs/config_eval.yaml - Evaluation config (3.5KB)
- /home/user/TradingBot2/configs/config_template.yaml - Config template (20KB)

### Component Configurations
- /home/user/TradingBot2/configs/execution.yaml - Execution parameters
- /home/user/TradingBot2/configs/slippage.yaml - Slippage model (9KB)
- /home/user/TradingBot2/configs/slippage_calibrate.yaml - Slippage calibration
- /home/user/TradingBot2/configs/fees.yaml - Trading fees (7KB)

### Risk & State
- /home/user/TradingBot2/configs/risk.yaml - Risk management
- /home/user/TradingBot2/configs/no_trade.yaml - No-trade periods (1.8KB)
- /home/user/TradingBot2/configs/quantizer.yaml - Price quantization
- /home/user/TradingBot2/configs/state.yaml - State management

### Operations & Monitoring
- /home/user/TradingBot2/configs/monitoring.yaml - Monitoring & alerts (1.3KB)
- /home/user/TradingBot2/configs/ops.yaml - Operations settings (3.4KB)
- /home/user/TradingBot2/configs/ops.json - Operations state
- /home/user/TradingBot2/configs/rest_budget.yaml - API rate limiting
- /home/user/TradingBot2/configs/timing.yaml - Timing settings
- /home/user/TradingBot2/configs/adv.yaml - ADV calculation
- /home/user/TradingBot2/configs/offline.yaml - Offline processing (3.4KB)

### Data Configuration
- /home/user/TradingBot2/configs/signal_quality.yaml - Signal quality filters
- /home/user/TradingBot2/configs/signals.yaml - Signal generation
- /home/user/TradingBot2/configs/liquidity_latency_seasonality.json - Latency seasonality
- /home/user/TradingBot2/configs/liquidity_seasonality.json - Seasonality (symlink)
- /home/user/TradingBot2/configs/market_regimes.json - Market regimes
- /home/user/TradingBot2/configs/reference_regime_distributions.json - Reference distributions

### Legacy Configurations
- /home/user/TradingBot2/configs/legacy_sim.yaml
- /home/user/TradingBot2/configs/legacy_realtime.yaml
- /home/user/TradingBot2/configs/legacy_sandbox.yaml
- /home/user/TradingBot2/configs/legacy_eval.yaml

## SCRIPTS DIRECTORY (/scripts)
### Data Building
- /home/user/TradingBot2/scripts/build_adv.py - Build ADV data
- /home/user/TradingBot2/scripts/build_adv_base.py - Base ADV calculation
- /home/user/TradingBot2/scripts/build_spread_seasonality.py - Spread seasonality
- /home/user/TradingBot2/scripts/extract_liquidity_seasonality.py - Liquidity seasonality
- /home/user/TradingBot2/scripts/fetch_binance_filters.py - Fetch exchange filters
- /home/user/TradingBot2/scripts/refresh_universe.py - Refresh symbol universe
- /home/user/TradingBot2/scripts/refresh_fees.py - Update fees
- /home/user/TradingBot2/scripts/impact_curve.py - Impact curve analysis

### Calibration & Validation
- /home/user/TradingBot2/scripts/calibrate_dynamic_spread.py - Dynamic spread calibration
- /home/user/TradingBot2/scripts/calibrate_live_slippage.py - Live slippage calibration
- /home/user/TradingBot2/scripts/calibrate_regimes.py - Regime calibration
- /home/user/TradingBot2/scripts/validate_seasonality.py - Validate seasonality
- /home/user/TradingBot2/scripts/validate_regime_distributions.py - Validate regimes
- /home/user/TradingBot2/scripts/verify_fees.py - Verify fee data

### Processing & Analysis
- /home/user/TradingBot2/scripts/run_seasonality_pipeline.py - Seasonality pipeline
- /home/user/TradingBot2/scripts/run_full_cycle.py - Full update cycle
- /home/user/TradingBot2/scripts/sim_reality_check.py - Reality check
- /home/user/TradingBot2/scripts/offline_utils.py - Offline utilities
- /home/user/TradingBot2/scripts/check_reward_clipping_bar_vs_cython.py - Reward clipping check

### Utilities & Monitoring
- /home/user/TradingBot2/scripts/check_pii.py - PII detection
- /home/user/TradingBot2/scripts/reset_kill_switch.py - Reset kill switch
- /home/user/TradingBot2/scripts/edit_multiplier.py - Edit multipliers
- /home/user/TradingBot2/scripts/convert_multipliers.py - Convert multipliers
- /home/user/TradingBot2/scripts/plot_seasonality.py - Plot seasonality
- /home/user/TradingBot2/scripts/seasonality_dashboard.py - Seasonality dashboard
- /home/user/TradingBot2/scripts/compare_seasonality_versions.py - Compare versions
- /home/user/TradingBot2/scripts/smoke_check_action_wrapper.py - Smoke checks
- /home/user/TradingBot2/scripts/chart_hourly_multiplier_metrics.py - Chart metrics
- /home/user/TradingBot2/scripts/cron_update_seasonality.sh - Cron job script

## DATA FILES (/data)
- /home/user/TradingBot2/data/symbols.json - Symbol list
- /home/user/TradingBot2/data/universe/symbols.json - Universe definitions
- /home/user/TradingBot2/data/fees/fees_by_symbol.json - Fee structure
- /home/user/TradingBot2/data/latency/liquidity_latency_no_seasonality.json - Latency data (no seasonality)
- /home/user/TradingBot2/data/latency/liquidity_latency_seasonality.json - Latency seasonality (symlink to configs/liquidity_latency_seasonality.json)
- /home/user/TradingBot2/data/impact_benchmark.json - Impact benchmarks
- /home/user/TradingBot2/data/hist_trades.csv - Historical trades
- /home/user/TradingBot2/data/sim_trades.csv - Simulated trades
- /home/user/TradingBot2/data/benchmark_equity.csv - Benchmark equity
- /home/user/TradingBot2/data/hourly_pattern_trades.csv - Hourly patterns
- /home/user/TradingBot2/data/no_trade_sample.csv - No-trade samples
- /home/user/TradingBot2/data/pipeline_time_split.csv - Train/val split

## DOCUMENTATION FILES (/docs)
### Main Documentation
- /home/user/TradingBot2/docs/seasonality.md - Seasonality methodology (11KB)
- /home/user/TradingBot2/docs/bar_execution.md - Bar execution details (5KB)
- /home/user/TradingBot2/docs/parkinson_volatility.md - Parkinson volatility (12KB)
- /home/user/TradingBot2/docs/yang_zhang_volatility.md - Yang-Zhang volatility (8KB)
- /home/user/TradingBot2/docs/no_trade.md - No-trade period logic (8KB)
- /home/user/TradingBot2/docs/pipeline.md - Decision pipeline (2KB)
- /home/user/TradingBot2/docs/data_degradation.md - Data degradation (1.6KB)
- /home/user/TradingBot2/docs/dynamic_spread.md - Dynamic spread (2.8KB)
- /home/user/TradingBot2/docs/large_orders.md - Large orders (1.9KB)
- /home/user/TradingBot2/docs/moving_average.md - Moving average (470 bytes)
- /home/user/TradingBot2/docs/permissions.md - File permissions (1KB)
- /home/user/TradingBot2/docs/parallel.md - Parallelization (3.5KB)

### Seasonality Documentation
- /home/user/TradingBot2/docs/seasonality_api.md - Seasonality API
- /home/user/TradingBot2/docs/seasonality_example.md - Seasonality examples
- /home/user/TradingBot2/docs/seasonality_process.md - Seasonality process
- /home/user/TradingBot2/docs/seasonality_quickstart.md - Seasonality quickstart
- /home/user/TradingBot2/docs/seasonality_QA.md - Seasonality Q&A
- /home/user/TradingBot2/docs/seasonality_migration.md - Seasonality migration
- /home/user/TradingBot2/docs/seasonality_checklist.md - Seasonality checklist
- /home/user/TradingBot2/docs/seasonality_data_policy.md - Data policy
- /home/user/TradingBot2/docs/seasonality_signoff.md - Sign-off

### JSON Schemas
- /home/user/TradingBot2/docs/spot_signal_envelope.schema.json - Signal envelope
- /home/user/TradingBot2/docs/spot_signal_target_weight.schema.json - Target weight schema
- /home/user/TradingBot2/docs/spot_signal_delta_weight.schema.json - Delta weight schema

## PROJECT DOCUMENTATION (Root Level)
- /home/user/TradingBot2/README.md - Main README (41KB)
- /home/user/TradingBot2/ARCHITECTURE.md - Architecture overview (14KB)
- /home/user/TradingBot2/CHANGELOG.md - Change log (2KB)
- /home/user/TradingBot2/CONTRIBUTING.md - Contributing guide (1.4KB)
- /home/user/TradingBot2/AUDIT_REPORT.md - Audit report (12KB)
- /home/user/TradingBot2/AUDIT_VERIFICATION_REPORT.md - Verification (10KB)
- /home/user/TradingBot2/FEATURE_MAPPING_56.md - Feature mapping (11KB)
- /home/user/TradingBot2/FULL_FEATURES_LIST.md - Features list (9KB)
- /home/user/TradingBot2/FEATURE_AUDIT_REPORT.md - Feature audit (12KB)
- /home/user/TradingBot2/OBSERVATION_MAPPING.md - Observation mapping (12KB)
- /home/user/TradingBot2/VERIFICATION_REPORT.md - Verification report (8KB)
- /home/user/TradingBot2/VERIFICATION_INSTRUCTIONS.md - Verification instructions (6KB)
- /home/user/TradingBot2/TRAINING_PIPELINE_ANALYSIS.md - Training pipeline (15KB)
- /home/user/TradingBot2/TRAINING_METRICS_ANALYSIS.md - Metrics analysis (29KB)
- /home/user/TradingBot2/METRICS_FIXES_SUMMARY.md - Metrics fixes (7KB)
- /home/user/TradingBot2/METRICS_QUICK_REFERENCE.txt - Metrics reference (8KB)
- /home/user/TradingBot2/YANG_ZHANG_FIX_SUMMARY.md - Yang-Zhang fix (6KB)
- /home/user/TradingBot2/GARCH_FEATURE.md - GARCH feature (10KB)
- /home/user/TradingBot2/SIZE_ANALYSIS.md - Size analysis (8KB)
- /home/user/TradingBot2/COMPILATION_REPORT.md - Compilation report (1.4KB)
- /home/user/TradingBot2/ANALYSIS_4H_TIMEFRAME.md - 4H timeframe analysis (29KB)
- /home/user/TradingBot2/DATASET_FIX_README.md - Dataset fix (5.6KB)
- /home/user/TradingBot2/claude.md - Claude notes (26KB)
- /home/user/TradingBot2/CODEBASE_STRUCTURE_ANALYSIS.md - This analysis document

## TEST FILES (Root level)
150+ test files located in /home/user/TradingBot2/ with test_*.py pattern

### Key Test Categories
- test_service_signal_runner_*.py - Service tests (multiple)
- test_service_backtest_*.py - Backtest service tests
- test_service_eval_*.py - Evaluation tests
- test_shared_memory_vec_env_*.py - Environment tests
- test_signal_*.py - Signal handling tests
- test_execution_*.py - Execution tests
- test_slippage_*.py - Slippage model tests
- test_*_cost*.py - Cost calculation tests
- test_train_model_*.py - Training tests
- test_risk_*.py - Risk management tests
- test_no_trade*.py - No-trade logic tests
- test_offline_*.py - Offline data tests
- test_quantizer_*.py - Quantizer tests

## OTHER KEY FILES
- /home/user/TradingBot2/adv_store.py - ADV storage (16KB)
- /home/user/TradingBot2/calibration.py - Calibration utilities (12KB)
- /home/user/TradingBot2/dynamic_no_trade_guard.py - No-trade management (17KB)
- /home/user/TradingBot2/drift.py - Drift detection (10KB)
- /home/user/TradingBot2/compat_shims.py - Compatibility layer (15KB)
- /home/user/TradingBot2/data_validation.py - Data validation (11KB)
- /home/user/TradingBot2/check_imports.py - Import checking
- /home/user/TradingBot2/custom_policy_patch1.py - Policy patches (63KB)
- /home/user/TradingBot2/conftest.py - Pytest configuration
- /home/user/TradingBot2/__init__.py - Package init
- /home/user/TradingBot2/ingest.yaml - Ingestion configuration

## SUPPORTING DIRECTORIES
- /home/user/TradingBot2/include/ - C++ headers (latency_queue.h, execevents_types.h)
- /home/user/TradingBot2/artifacts/ - Training artifacts (default-run/)
- /home/user/TradingBot2/cache/ - Runtime caching
- /home/user/TradingBot2/state/ - Application state (JSON files)
- /home/user/TradingBot2/notebooks/ - Jupyter notebooks
- /home/user/TradingBot2/sandbox/ - Testing environment
- /home/user/TradingBot2/audits/ - Audit logs
- /home/user/TradingBot2/benchmarks/ - Benchmark data

