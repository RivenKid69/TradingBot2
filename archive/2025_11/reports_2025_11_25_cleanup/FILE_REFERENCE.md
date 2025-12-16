# AI-Powered Quantitative Research Platform - Complete File Path Reference

## CORE MODULES (core_*.py)
- /home/user/AI-Powered Quantitative Research Platform/core_config.py - Configuration models and DI specs (1,382 lines)
- /home/user/AI-Powered Quantitative Research Platform/core_models.py - Domain models (Side, Order, Bar, Tick) (516 lines)
- /home/user/AI-Powered Quantitative Research Platform/core_contracts.py - Abstract protocols/interfaces (141 lines)
- /home/user/AI-Powered Quantitative Research Platform/core_strategy.py - Strategy interface and Decision class (85 lines)
- /home/user/AI-Powered Quantitative Research Platform/core_events.py - Event types and enums
- /home/user/AI-Powered Quantitative Research Platform/core_errors.py - Custom exception classes
- /home/user/AI-Powered Quantitative Research Platform/core_constants.py - Global constants

## IMPLEMENTATION MODULES (impl_*.py)
- /home/user/AI-Powered Quantitative Research Platform/impl_sim_executor.py - Basic execution simulator (1,424 lines)
- /home/user/AI-Powered Quantitative Research Platform/impl_bar_executor.py - Bar-based order execution (1,685 lines)
- /home/user/AI-Powered Quantitative Research Platform/impl_slippage.py - Slippage model calculations (2,395 lines)
- /home/user/AI-Powered Quantitative Research Platform/impl_fees.py - Trading fee handling (1,684 lines)
- /home/user/AI-Powered Quantitative Research Platform/impl_latency.py - Network latency simulation (1,117 lines)
- /home/user/AI-Powered Quantitative Research Platform/impl_quantizer.py - Price/qty quantization (883 lines)
- /home/user/AI-Powered Quantitative Research Platform/impl_offline_data.py - CSV/Parquet data loading (294 lines)
- /home/user/AI-Powered Quantitative Research Platform/impl_binance_public.py - Binance REST API (248 lines)
- /home/user/AI-Powered Quantitative Research Platform/impl_risk_basic.py - Basic risk guards (162 lines)

## SERVICE MODULES (service_*.py)
- /home/user/AI-Powered Quantitative Research Platform/service_signal_runner.py - Live signal generation (9,578 lines)
- /home/user/AI-Powered Quantitative Research Platform/service_backtest.py - Backtesting service (2,054 lines)
- /home/user/AI-Powered Quantitative Research Platform/service_train.py - ML training service (218 lines)
- /home/user/AI-Powered Quantitative Research Platform/service_eval.py - Strategy evaluation (339 lines)
- /home/user/AI-Powered Quantitative Research Platform/service_calibrate_slippage.py - Slippage calibration (142 lines)
- /home/user/AI-Powered Quantitative Research Platform/service_calibrate_tcost.py - Transaction cost calibration (263 lines)
- /home/user/AI-Powered Quantitative Research Platform/service_fetch_exchange_specs.py - Fetch Binance specs (451 lines)

## SCRIPT ENTRY POINTS (script_*.py)
- /home/user/AI-Powered Quantitative Research Platform/script_live.py - Launch live trading
- /home/user/AI-Powered Quantitative Research Platform/script_backtest.py - Run backtesting
- /home/user/AI-Powered Quantitative Research Platform/script_eval.py - Evaluate strategy
- /home/user/AI-Powered Quantitative Research Platform/script_calibrate_slippage.py - Calibrate slippage
- /home/user/AI-Powered Quantitative Research Platform/script_calibrate_tcost.py - Calibrate transaction costs
- /home/user/AI-Powered Quantitative Research Platform/script_fetch_exchange_specs.py - Fetch exchange specs
- /home/user/AI-Powered Quantitative Research Platform/script_compare_runs.py - Compare backtesting runs

## MAIN INFRASTRUCTURE MODULES
- /home/user/AI-Powered Quantitative Research Platform/execution_sim.py - Advanced execution simulator (550KB, 12,993 lines)
- /home/user/AI-Powered Quantitative Research Platform/app.py - Web application (Streamlit/FastAPI) (180KB, 4,500+ lines)
- /home/user/AI-Powered Quantitative Research Platform/distributional_ppo.py - PPO RL algorithm (444KB, 9,700+ lines)
- /home/user/AI-Powered Quantitative Research Platform/train_model_multi_patch.py - Training orchestrator
- /home/user/AI-Powered Quantitative Research Platform/binance_public.py - Binance public API client
- /home/user/AI-Powered Quantitative Research Platform/binance_ws.py - Binance WebSocket feeds
- /home/user/AI-Powered Quantitative Research Platform/binance_fee_refresh.py - Fee refresh mechanism
- /home/user/AI-Powered Quantitative Research Platform/exchangespecs.py - Exchange specifications
- /home/user/AI-Powered Quantitative Research Platform/execution_algos.py - VWAP, POV algorithms
- /home/user/AI-Powered Quantitative Research Platform/clock.py - Clock synchronization with Binance
- /home/user/AI-Powered Quantitative Research Platform/action_proto.py - Action protocol & legacy compatibility
- /home/user/AI-Powered Quantitative Research Platform/di_registry.py - Dependency injection container
- /home/user/AI-Powered Quantitative Research Platform/di_stubs.py - DI stubs for testing
- /home/user/AI-Powered Quantitative Research Platform/config.py - Environment configurations

## SERVICES DIRECTORY (/services)
- /home/user/AI-Powered Quantitative Research Platform/services/__init__.py
- /home/user/AI-Powered Quantitative Research Platform/services/monitoring.py - Metrics & alerts (64KB)
- /home/user/AI-Powered Quantitative Research Platform/services/rest_budget.py - REST rate limiting (66KB)
- /home/user/AI-Powered Quantitative Research Platform/services/state_storage.py - State persistence (32KB)
- /home/user/AI-Powered Quantitative Research Platform/services/signal_bus.py - Signal distribution (11KB)
- /home/user/AI-Powered Quantitative Research Platform/services/metrics.py - Performance metrics (16KB)
- /home/user/AI-Powered Quantitative Research Platform/services/costs.py - Cost tracking (12KB)
- /home/user/AI-Powered Quantitative Research Platform/services/event_bus.py - Event system (12KB)
- /home/user/AI-Powered Quantitative Research Platform/services/alerts.py - Alert generation (4KB)
- /home/user/AI-Powered Quantitative Research Platform/services/ops_kill_switch.py - Trading halt (7KB)
- /home/user/AI-Powered Quantitative Research Platform/services/universe.py - Symbol management (5KB)
- /home/user/AI-Powered Quantitative Research Platform/services/retry.py - Retry logic (5KB)
- /home/user/AI-Powered Quantitative Research Platform/services/shutdown.py - Shutdown handling (5KB)
- /home/user/AI-Powered Quantitative Research Platform/services/signal_csv_writer.py - Signal logging (9KB)
- /home/user/AI-Powered Quantitative Research Platform/services/utils_app.py - App utilities (9KB)
- /home/user/AI-Powered Quantitative Research Platform/services/utils_config.py - Config utilities (1KB)
- /home/user/AI-Powered Quantitative Research Platform/services/utils_sandbox.py - Sandbox utilities (1KB)

## STRATEGIES DIRECTORY (/strategies)
- /home/user/AI-Powered Quantitative Research Platform/strategies/__init__.py
- /home/user/AI-Powered Quantitative Research Platform/strategies/base.py - Base strategy class (9.5KB)
- /home/user/AI-Powered Quantitative Research Platform/strategies/momentum.py - Momentum strategy (7.2KB)

## UTILS DIRECTORY (/utils)
- /home/user/AI-Powered Quantitative Research Platform/utils/__init__.py
- /home/user/AI-Powered Quantitative Research Platform/utils/time.py - Time utilities
- /home/user/AI-Powered Quantitative Research Platform/utils/model_io.py - Model I/O
- /home/user/AI-Powered Quantitative Research Platform/utils/time_provider.py - Mock time provider
- /home/user/AI-Powered Quantitative Research Platform/utils/rate_limiter.py - Rate limiting
- /home/user/AI-Powered Quantitative Research Platform/utils/prometheus.py - Prometheus integration
- /home/user/AI-Powered Quantitative Research Platform/utils/moving_average.py - Moving average

## API DIRECTORY (/api)
- /home/user/AI-Powered Quantitative Research Platform/api/__init__.py
- /home/user/AI-Powered Quantitative Research Platform/api/spot_signals.py - Spot market signals
- /home/user/AI-Powered Quantitative Research Platform/api/config.py - API configuration

## DOMAIN DIRECTORY (/domain)
- /home/user/AI-Powered Quantitative Research Platform/domain/__init__.py
- /home/user/AI-Powered Quantitative Research Platform/domain/adapters.py - Domain adapters

## ADAPTERS DIRECTORY (/adapters)
- /home/user/AI-Powered Quantitative Research Platform/adapters/binance_spot_private.py - Binance private API

## WRAPPERS DIRECTORY (/wrappers)
- /home/user/AI-Powered Quantitative Research Platform/wrappers/__init__.py
- /home/user/AI-Powered Quantitative Research Platform/wrappers/action_space.py - Action space definitions

## CONFIGURATION FILES (/configs)
### Main Configurations
- /home/user/AI-Powered Quantitative Research Platform/configs/config_sim.yaml - Simulation config (20KB)
- /home/user/AI-Powered Quantitative Research Platform/configs/config_live.yaml - Live trading config (6KB)
- /home/user/AI-Powered Quantitative Research Platform/configs/config_train.yaml - Training config (17KB)
- /home/user/AI-Powered Quantitative Research Platform/configs/config_eval.yaml - Evaluation config (3.5KB)
- /home/user/AI-Powered Quantitative Research Platform/configs/config_template.yaml - Config template (20KB)

### Component Configurations
- /home/user/AI-Powered Quantitative Research Platform/configs/execution.yaml - Execution parameters
- /home/user/AI-Powered Quantitative Research Platform/configs/slippage.yaml - Slippage model (9KB)
- /home/user/AI-Powered Quantitative Research Platform/configs/slippage_calibrate.yaml - Slippage calibration
- /home/user/AI-Powered Quantitative Research Platform/configs/fees.yaml - Trading fees (7KB)

### Risk & State
- /home/user/AI-Powered Quantitative Research Platform/configs/risk.yaml - Risk management
- /home/user/AI-Powered Quantitative Research Platform/configs/no_trade.yaml - No-trade periods (1.8KB)
- /home/user/AI-Powered Quantitative Research Platform/configs/quantizer.yaml - Price quantization
- /home/user/AI-Powered Quantitative Research Platform/configs/state.yaml - State management

### Operations & Monitoring
- /home/user/AI-Powered Quantitative Research Platform/configs/monitoring.yaml - Monitoring & alerts (1.3KB)
- /home/user/AI-Powered Quantitative Research Platform/configs/ops.yaml - Operations settings (3.4KB)
- /home/user/AI-Powered Quantitative Research Platform/configs/ops.json - Operations state
- /home/user/AI-Powered Quantitative Research Platform/configs/rest_budget.yaml - API rate limiting
- /home/user/AI-Powered Quantitative Research Platform/configs/timing.yaml - Timing settings
- /home/user/AI-Powered Quantitative Research Platform/configs/adv.yaml - ADV calculation
- /home/user/AI-Powered Quantitative Research Platform/configs/offline.yaml - Offline processing (3.4KB)

### Data Configuration
- /home/user/AI-Powered Quantitative Research Platform/configs/signal_quality.yaml - Signal quality filters
- /home/user/AI-Powered Quantitative Research Platform/configs/signals.yaml - Signal generation
- /home/user/AI-Powered Quantitative Research Platform/configs/liquidity_latency_seasonality.json - Latency seasonality
- /home/user/AI-Powered Quantitative Research Platform/configs/liquidity_seasonality.json - Seasonality (symlink)
- /home/user/AI-Powered Quantitative Research Platform/configs/market_regimes.json - Market regimes
- /home/user/AI-Powered Quantitative Research Platform/configs/reference_regime_distributions.json - Reference distributions

### Legacy Configurations
- /home/user/AI-Powered Quantitative Research Platform/configs/legacy_sim.yaml
- /home/user/AI-Powered Quantitative Research Platform/configs/legacy_realtime.yaml
- /home/user/AI-Powered Quantitative Research Platform/configs/legacy_sandbox.yaml
- /home/user/AI-Powered Quantitative Research Platform/configs/legacy_eval.yaml

## SCRIPTS DIRECTORY (/scripts)
### Data Building
- /home/user/AI-Powered Quantitative Research Platform/scripts/build_adv.py - Build ADV data
- /home/user/AI-Powered Quantitative Research Platform/scripts/build_adv_base.py - Base ADV calculation
- /home/user/AI-Powered Quantitative Research Platform/scripts/build_spread_seasonality.py - Spread seasonality
- /home/user/AI-Powered Quantitative Research Platform/scripts/extract_liquidity_seasonality.py - Liquidity seasonality
- /home/user/AI-Powered Quantitative Research Platform/scripts/fetch_binance_filters.py - Fetch exchange filters
- /home/user/AI-Powered Quantitative Research Platform/scripts/refresh_universe.py - Refresh symbol universe
- /home/user/AI-Powered Quantitative Research Platform/scripts/refresh_fees.py - Update fees
- /home/user/AI-Powered Quantitative Research Platform/scripts/impact_curve.py - Impact curve analysis

### Calibration & Validation
- /home/user/AI-Powered Quantitative Research Platform/scripts/calibrate_dynamic_spread.py - Dynamic spread calibration
- /home/user/AI-Powered Quantitative Research Platform/scripts/calibrate_live_slippage.py - Live slippage calibration
- /home/user/AI-Powered Quantitative Research Platform/scripts/calibrate_regimes.py - Regime calibration
- /home/user/AI-Powered Quantitative Research Platform/scripts/validate_seasonality.py - Validate seasonality
- /home/user/AI-Powered Quantitative Research Platform/scripts/validate_regime_distributions.py - Validate regimes
- /home/user/AI-Powered Quantitative Research Platform/scripts/verify_fees.py - Verify fee data

### Processing & Analysis
- /home/user/AI-Powered Quantitative Research Platform/scripts/run_seasonality_pipeline.py - Seasonality pipeline
- /home/user/AI-Powered Quantitative Research Platform/scripts/run_full_cycle.py - Full update cycle
- /home/user/AI-Powered Quantitative Research Platform/scripts/sim_reality_check.py - Reality check
- /home/user/AI-Powered Quantitative Research Platform/scripts/offline_utils.py - Offline utilities
- /home/user/AI-Powered Quantitative Research Platform/scripts/check_reward_clipping_bar_vs_cython.py - Reward clipping check

### Utilities & Monitoring
- /home/user/AI-Powered Quantitative Research Platform/scripts/check_pii.py - PII detection
- /home/user/AI-Powered Quantitative Research Platform/scripts/reset_kill_switch.py - Reset kill switch
- /home/user/AI-Powered Quantitative Research Platform/scripts/edit_multiplier.py - Edit multipliers
- /home/user/AI-Powered Quantitative Research Platform/scripts/convert_multipliers.py - Convert multipliers
- /home/user/AI-Powered Quantitative Research Platform/scripts/plot_seasonality.py - Plot seasonality
- /home/user/AI-Powered Quantitative Research Platform/scripts/seasonality_dashboard.py - Seasonality dashboard
- /home/user/AI-Powered Quantitative Research Platform/scripts/compare_seasonality_versions.py - Compare versions
- /home/user/AI-Powered Quantitative Research Platform/scripts/smoke_check_action_wrapper.py - Smoke checks
- /home/user/AI-Powered Quantitative Research Platform/scripts/chart_hourly_multiplier_metrics.py - Chart metrics
- /home/user/AI-Powered Quantitative Research Platform/scripts/cron_update_seasonality.sh - Cron job script

## DATA FILES (/data)
- /home/user/AI-Powered Quantitative Research Platform/data/symbols.json - Symbol list
- /home/user/AI-Powered Quantitative Research Platform/data/universe/symbols.json - Universe definitions
- /home/user/AI-Powered Quantitative Research Platform/data/fees/fees_by_symbol.json - Fee structure
- /home/user/AI-Powered Quantitative Research Platform/data/latency/liquidity_latency_no_seasonality.json - Latency data (no seasonality)
- /home/user/AI-Powered Quantitative Research Platform/data/latency/liquidity_latency_seasonality.json - Latency seasonality (symlink to configs/liquidity_latency_seasonality.json)
- /home/user/AI-Powered Quantitative Research Platform/data/impact_benchmark.json - Impact benchmarks
- /home/user/AI-Powered Quantitative Research Platform/data/hist_trades.csv - Historical trades
- /home/user/AI-Powered Quantitative Research Platform/data/sim_trades.csv - Simulated trades
- /home/user/AI-Powered Quantitative Research Platform/data/benchmark_equity.csv - Benchmark equity
- /home/user/AI-Powered Quantitative Research Platform/data/hourly_pattern_trades.csv - Hourly patterns
- /home/user/AI-Powered Quantitative Research Platform/data/no_trade_sample.csv - No-trade samples
- /home/user/AI-Powered Quantitative Research Platform/data/pipeline_time_split.csv - Train/val split

## DOCUMENTATION FILES (/docs)
### Main Documentation
- /home/user/AI-Powered Quantitative Research Platform/docs/seasonality.md - Seasonality methodology (11KB)
- /home/user/AI-Powered Quantitative Research Platform/docs/bar_execution.md - Bar execution details (5KB)
- /home/user/AI-Powered Quantitative Research Platform/docs/parkinson_volatility.md - Parkinson volatility (12KB)
- /home/user/AI-Powered Quantitative Research Platform/docs/yang_zhang_volatility.md - Yang-Zhang volatility (8KB)
- /home/user/AI-Powered Quantitative Research Platform/docs/no_trade.md - No-trade period logic (8KB)
- /home/user/AI-Powered Quantitative Research Platform/docs/pipeline.md - Decision pipeline (2KB)
- /home/user/AI-Powered Quantitative Research Platform/docs/data_degradation.md - Data degradation (1.6KB)
- /home/user/AI-Powered Quantitative Research Platform/docs/dynamic_spread.md - Dynamic spread (2.8KB)
- /home/user/AI-Powered Quantitative Research Platform/docs/large_orders.md - Large orders (1.9KB)
- /home/user/AI-Powered Quantitative Research Platform/docs/moving_average.md - Moving average (470 bytes)
- /home/user/AI-Powered Quantitative Research Platform/docs/permissions.md - File permissions (1KB)
- /home/user/AI-Powered Quantitative Research Platform/docs/parallel.md - Parallelization (3.5KB)

### Seasonality Documentation
- /home/user/AI-Powered Quantitative Research Platform/docs/seasonality_api.md - Seasonality API
- /home/user/AI-Powered Quantitative Research Platform/docs/seasonality_example.md - Seasonality examples
- /home/user/AI-Powered Quantitative Research Platform/docs/seasonality_process.md - Seasonality process
- /home/user/AI-Powered Quantitative Research Platform/docs/seasonality_quickstart.md - Seasonality quickstart
- /home/user/AI-Powered Quantitative Research Platform/docs/seasonality_QA.md - Seasonality Q&A
- /home/user/AI-Powered Quantitative Research Platform/docs/seasonality_migration.md - Seasonality migration
- /home/user/AI-Powered Quantitative Research Platform/docs/seasonality_checklist.md - Seasonality checklist
- /home/user/AI-Powered Quantitative Research Platform/docs/seasonality_data_policy.md - Data policy
- /home/user/AI-Powered Quantitative Research Platform/docs/seasonality_signoff.md - Sign-off

### JSON Schemas
- /home/user/AI-Powered Quantitative Research Platform/docs/spot_signal_envelope.schema.json - Signal envelope
- /home/user/AI-Powered Quantitative Research Platform/docs/spot_signal_target_weight.schema.json - Target weight schema
- /home/user/AI-Powered Quantitative Research Platform/docs/spot_signal_delta_weight.schema.json - Delta weight schema

## PROJECT DOCUMENTATION (Root Level)
- /home/user/AI-Powered Quantitative Research Platform/README.md - Main README (41KB)
- /home/user/AI-Powered Quantitative Research Platform/ARCHITECTURE.md - Architecture overview (14KB)
- /home/user/AI-Powered Quantitative Research Platform/CHANGELOG.md - Change log (2KB)
- /home/user/AI-Powered Quantitative Research Platform/CONTRIBUTING.md - Contributing guide (1.4KB)
- /home/user/AI-Powered Quantitative Research Platform/AUDIT_REPORT.md - Audit report (12KB)
- /home/user/AI-Powered Quantitative Research Platform/AUDIT_VERIFICATION_REPORT.md - Verification (10KB)
- /home/user/AI-Powered Quantitative Research Platform/FEATURE_MAPPING_56.md - Feature mapping (11KB)
- /home/user/AI-Powered Quantitative Research Platform/FULL_FEATURES_LIST.md - Features list (9KB)
- /home/user/AI-Powered Quantitative Research Platform/FEATURE_AUDIT_REPORT.md - Feature audit (12KB)
- /home/user/AI-Powered Quantitative Research Platform/OBSERVATION_MAPPING.md - Observation mapping (12KB)
- /home/user/AI-Powered Quantitative Research Platform/VERIFICATION_REPORT.md - Verification report (8KB)
- /home/user/AI-Powered Quantitative Research Platform/VERIFICATION_INSTRUCTIONS.md - Verification instructions (6KB)
- /home/user/AI-Powered Quantitative Research Platform/TRAINING_PIPELINE_ANALYSIS.md - Training pipeline (15KB)
- /home/user/AI-Powered Quantitative Research Platform/TRAINING_METRICS_ANALYSIS.md - Metrics analysis (29KB)
- /home/user/AI-Powered Quantitative Research Platform/METRICS_FIXES_SUMMARY.md - Metrics fixes (7KB)
- /home/user/AI-Powered Quantitative Research Platform/METRICS_QUICK_REFERENCE.txt - Metrics reference (8KB)
- /home/user/AI-Powered Quantitative Research Platform/YANG_ZHANG_FIX_SUMMARY.md - Yang-Zhang fix (6KB)
- /home/user/AI-Powered Quantitative Research Platform/GARCH_FEATURE.md - GARCH feature (10KB)
- /home/user/AI-Powered Quantitative Research Platform/SIZE_ANALYSIS.md - Size analysis (8KB)
- /home/user/AI-Powered Quantitative Research Platform/COMPILATION_REPORT.md - Compilation report (1.4KB)
- /home/user/AI-Powered Quantitative Research Platform/ANALYSIS_4H_TIMEFRAME.md - 4H timeframe analysis (29KB)
- /home/user/AI-Powered Quantitative Research Platform/DATASET_FIX_README.md - Dataset fix (5.6KB)
- /home/user/AI-Powered Quantitative Research Platform/claude.md - Claude notes (26KB)
- /home/user/AI-Powered Quantitative Research Platform/CODEBASE_STRUCTURE_ANALYSIS.md - This analysis document

## TEST FILES (Root level)
150+ test files located in /home/user/AI-Powered Quantitative Research Platform/ with test_*.py pattern

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
- /home/user/AI-Powered Quantitative Research Platform/adv_store.py - ADV storage (16KB)
- /home/user/AI-Powered Quantitative Research Platform/calibration.py - Calibration utilities (12KB)
- /home/user/AI-Powered Quantitative Research Platform/dynamic_no_trade_guard.py - No-trade management (17KB)
- /home/user/AI-Powered Quantitative Research Platform/drift.py - Drift detection (10KB)
- /home/user/AI-Powered Quantitative Research Platform/compat_shims.py - Compatibility layer (15KB)
- /home/user/AI-Powered Quantitative Research Platform/data_validation.py - Data validation (11KB)
- /home/user/AI-Powered Quantitative Research Platform/check_imports.py - Import checking
- /home/user/AI-Powered Quantitative Research Platform/custom_policy_patch1.py - Policy patches (63KB)
- /home/user/AI-Powered Quantitative Research Platform/conftest.py - Pytest configuration
- /home/user/AI-Powered Quantitative Research Platform/__init__.py - Package init
- /home/user/AI-Powered Quantitative Research Platform/ingest.yaml - Ingestion configuration

## SUPPORTING DIRECTORIES
- /home/user/AI-Powered Quantitative Research Platform/include/ - C++ headers (latency_queue.h, execevents_types.h)
- /home/user/AI-Powered Quantitative Research Platform/artifacts/ - Training artifacts (default-run/)
- /home/user/AI-Powered Quantitative Research Platform/cache/ - Runtime caching
- /home/user/AI-Powered Quantitative Research Platform/state/ - Application state (JSON files)
- /home/user/AI-Powered Quantitative Research Platform/notebooks/ - Jupyter notebooks
- /home/user/AI-Powered Quantitative Research Platform/sandbox/ - Testing environment
- /home/user/AI-Powered Quantitative Research Platform/audits/ - Audit logs
- /home/user/AI-Powered Quantitative Research Platform/benchmarks/ - Benchmark data

