# Claude Documentation - CustodiaCloud

> **IMPORTANT DISCLAIMER (Internal Document)**: This is an **internal technical guide for AI assistants**. Status badges like "✅ Tested and operational" and test count claims (e.g., "205/205 pass") refer to **internal CI test results at time of writing** (verify current status via CI run logs with commit SHA). These are **NOT** claims of independent third-party audit, certification, or regulatory compliance. This document is for internal development use only and should not be used for external marketing, investor materials, or committee submissions without appropriate disclaimers.

---

## 🤖 БЫСТРАЯ НАВИГАЦИЯ ДЛЯ AI-АССИСТЕНТОВ

### Критические паттерны работы

**ВСЕГДА НАЧИНАЙТЕ С:**
1. **Изучите слоистую архитектуру** -- `core_` → `impl_` → `service_` → `strategies` → `script_` -- НЕ НАРУШАЙТЕ зависимости!
2. **Используйте Glob/Grep** для поиска файлов, НЕ используйте bash find/grep
3. **Читайте файлы перед изменением** -- НИКОГДА не редактируйте файлы, которые не читали
4. **Проверяйте тесты** -- перед изменением критичной логики найдите и изучите соответствующие тесты

### 🔀 Git Workflow

**Основная ветка: `main`**

- Все коммиты делать в `main`
- Все пуши делать в `main`
- НЕ создавать feature branches без явной просьбы пользователя

```bash
git add . && git commit -m "message" && git push origin main
```

### 📍 Быстрый поиск по задачам

| Задача | Где искать | Команда |
|--------|------------|---------|
| **CCEA архитектура** | `packages/agent/`, `packages/cloud/`, `docs/CCEA_OVERVIEW.md` | `pytest tests/ccea/` |
| **Firm-wide risk** (strategy→desk→firm VaR/CVaR, Euler) | `service_firm_risk.py` | `pytest tests/test_firm_risk.py` |
| **Live P&L ledger** (realized/unrealized, EOD NAV) | `packages/agent/accounting/pnl_ledger.py` | `pytest tests/test_pnl_ledger.py` |
| **Instrument master / symbology** (FIGI/ISIN/CUSIP/OCC) | `services/instrument_master.py` | `pytest tests/test_instrument_master.py` |
| **Trade blotter + cash ledger** (books-and-records) | `packages/agent/accounting/{blotter,books}.py` | `pytest tests/test_books_and_records.py` |
| **Tamper-evident журнал** (hash-chain) | `packages/agent/audit/hash_chain.py`, `packages/agent/reconciliation/journal.py` | `pytest tests/test_journal_tamper_evident.py` |
| **Market-abuse surveillance** (MAR, wired) | `services/algo_integration/market_abuse.py`, `packages/agent/accounting/books.py` | `pytest tests/test_books_and_records.py` |
| **MC VaR / Euler / named scenarios** | `service_pretrade_risk.py` | `pytest tests/test_pretrade_risk_p1.py` |
| **Optimizer config** (sector/factor caps/robust/BL/multi-period) | `service_xs_pipeline.py`, `service_optimizer.py` | `pytest tests/test_optimizer_config_p1.py` |
| **Block-bootstrap CI / CPCV-PBO** | `research/bootstrap.py`, `service_backtest_validation.py` | `pytest tests/test_bootstrap_pbo_p1.py` |
| **IS executor / FIX 35=G / price-collar** | `service_xs_execution.py`, `packages/agent/execution/{engine,fix_protocol}.py` | `pytest tests/test_execution_p1.py` |
| **SOR live submission** (routed_broker_submit) | `packages/agent/execution/{live_factory,smart_order_router}.py` | `pytest tests/test_sor_live_p1.py` |
| **Market-data QC + vendor failover** | `services/market_data_quality.py` | `pytest tests/test_market_data_quality_p1.py` |
| Найти определение класса/функции | Используйте Glob | `*.py` pattern с именем |
| Исправить ошибку в feature | `features/` + `feature_config.py` | `pytest tests/test_features*.py` |
| Изменить логику исполнения | `execution_sim.py`, `execution_providers.py` | `pytest tests/test_execution*.py` |
| Execution providers (L2/L3) | `execution_providers.py` | `pytest tests/test_execution_providers.py` |
| Crypto Parametric TCA | `execution_providers.py` | `pytest tests/test_crypto_parametric_tca.py` |
| Equity Parametric TCA | `execution_providers.py` | `pytest tests/test_equity_parametric_tca.py` |
| Настроить риск-менеджмент | `configs/risk.yaml`, `risk_guard.py` | Проверить `test_risk*.py` |
| **Live enforcement лимитов** (daily-loss/DD/leverage → pre-trade + circuit breaker) | `services/live_risk_limits.py`, `packages/agent/policy/risk_checker.py` | `pytest tests/test_live_risk_enforcement.py` |
| Обновить модель PPO | `distributional_ppo.py` | Проверить все `test_distributional_ppo*.py` |
| Добавить новую метрику | `services/monitoring.py` | Обновить `metrics.json` schema |
| Калибровать параметры | `service_calibrate_*.py` | Запустить соответствующий script |
| Отладить training | `train_model_multi_patch.py` + logs | Проверить `tensorboard` logs |
| Проблемы с данными | `impl_offline_data.py`, `data_validation.py` | Проверить data degradation params |
| Live trading проблемы | `script_live.py` → `service_signal_runner.py` | Проверить ops_kill_switch, state_storage |
| Position sync (Alpaca) | `services/position_sync.py` | `pytest tests/test_phase9_live_trading.py::TestPositionSynchronizer` |
| Extended hours trading | `services/session_router.py` | `pytest tests/test_phase9_live_trading.py::TestSessionRouter` |
| Bracket/OCO orders | `adapters/alpaca/order_execution.py` | `pytest tests/test_phase9_live_trading.py::TestBracketOrderConfig` |
| Скачать stock data | `scripts/download_stock_data.py` | `--symbols GLD IAU SLV --start 2020-01-01` |
| Скачать VIX данные | `scripts/download_stock_data.py` | `--vix --start 2020-01-01` или `--symbols ^VIX` |
| Скачать macro данные | `scripts/download_stock_data.py` | `--macro --start 2020-01-01` (VIX, DXY, Treasury) |
| Yahoo market data | `adapters/yahoo/market_data.py` | Auto-used for ^VIX, DX-Y.NYB, indices |
| Benchmark temporal alignment | `stock_features.py` | `pytest tests/test_benchmark_temporal_alignment.py` |
| Alpaca streaming | `adapters/alpaca/market_data.py` | `stream_bars_async()`, `stream_ticks_async()` |
| L3 LOB matching | `lob/matching_engine.py` | `pytest tests/test_matching_engine.py` |
| Queue position tracking | `lob/queue_tracker.py` | `pytest tests/test_matching_engine.py::TestQueuePositionTracker` |
| Order lifecycle | `lob/order_manager.py` | `pytest tests/test_matching_engine.py::TestOrderManager` |
| Fill probability models | `lob/fill_probability.py` | `pytest tests/test_fill_probability_queue_value.py` |
| Queue value (Moallemi) | `lob/queue_value.py` | `pytest tests/test_fill_probability_queue_value.py::TestQueueValueModel` |
| LOB calibration | `lob/calibration.py` | `pytest tests/test_fill_probability_queue_value.py::TestCalibrationPipeline` |
| Market impact models | `lob/market_impact.py` | `pytest tests/test_market_impact.py::TestAlmgrenChrissModel` |
| Impact effects on LOB | `lob/impact_effects.py` | `pytest tests/test_market_impact.py::TestImpactEffects` |
| Impact calibration | `lob/impact_calibration.py` | `pytest tests/test_market_impact.py::TestImpactCalibration` |
| Latency simulation | `lob/latency_model.py` | `pytest tests/test_lob_latency.py::TestLatencyModel` |
| Event scheduler | `lob/event_scheduler.py` | `pytest tests/test_lob_latency.py::TestEventScheduler` |
| Iceberg detection | `lob/hidden_liquidity.py` | `pytest tests/test_hidden_liquidity_dark_pools.py::TestIcebergDetector` |
| Hidden liquidity | `lob/hidden_liquidity.py` | `pytest tests/test_hidden_liquidity_dark_pools.py::TestHiddenLiquidityEstimator` |
| Dark pool simulation | `lob/dark_pool.py` | `pytest tests/test_hidden_liquidity_dark_pools.py::TestDarkPoolSimulator` |
| L3 execution provider | `execution_providers_l3.py` | `pytest tests/test_execution_providers_l3.py` |
| L3 config models | `lob/config.py` | `pytest tests/test_execution_providers_l3.py::TestL3ExecutionConfig` |
| Conformal prediction | `core_conformal.py`, `impl_conformal.py`, `service_conformal.py` | `pytest tests/test_conformal_prediction.py` |
| Uncertainty bounds | `service_conformal.py` | `pytest tests/test_conformal_prediction.py::TestUncertaintyTracker` |
| CVaR bounds | `impl_conformal.py` | `pytest tests/test_conformal_prediction.py::TestConformalCVaREstimator` |
| Stock features (VIX, RS) | `stock_features.py` | `pytest tests/test_stock_features.py` |
| Stock risk guards | `services/stock_risk_guards.py` | `pytest tests/test_stock_risk_guards.py` |
| Stock universe mgmt | `services/universe_stocks.py` | `pytest tests/test_universe_stocks.py` |
| US market structure | `lob/us_market_structure.py` | `pytest tests/test_us_market_structure.py` |
| Verification tools | `tools/check_*.py`, `tools/verify_*.py` | Run directly with `python tools/<script>.py` |
| Feature parity check | `tools/check_feature_parity.py` | `python tools/check_feature_parity.py` |
| **Forex Parametric TCA** | `execution_providers.py` | `pytest tests/test_forex_parametric_tca.py` |
| Forex features (sessions) | `forex_features.py` | `pytest tests/test_forex_features.py` |
| Forex dealer simulation | `services/forex_dealer.py` | `pytest tests/test_forex_dealer_simulation.py` |
| Forex risk guards | `services/forex_risk_guards.py` | `pytest tests/test_forex_phase6_risk_services.py` |
| Forex session router | `services/forex_session_router.py` | `pytest tests/test_forex_execution_integration.py` |
| Forex config | `services/forex_config.py` | `pytest tests/test_forex_configuration.py` |
| OANDA adapter | `adapters/oanda/*.py` | `pytest tests/test_forex_foundation.py` |
| Forex tick simulation | `lob/forex_tick_simulation.py` | `pytest tests/test_forex_tick_simulation.py` |
| **IB market data** (CME futures) | `adapters/ib/market_data.py` | `pytest tests/test_ib_adapters.py::TestIBMarketDataAdapter` |
| **IB order execution** (CME) | `adapters/ib/order_execution.py` | `pytest tests/test_ib_adapters.py::TestIBOrderExecutionAdapter` |
| **Binance spot order execution** (crypto live/panic) | `adapters/binance/order_execution.py` | `pytest tests/test_binance_spot_execution.py` |
| **Agent daemon config** (standalone launch) | `configs/agent.yaml`, `packages/agent/daemon/__main__.py` | `pytest tests/test_agent_config.py` |
| **Model signature gate in daemon** (Ed25519, fail-closed LIVE) | `packages/agent/daemon/model_gate.py`, `services/model_signature_gate.py` | `pytest tests/test_agent_model_signature.py` |
| **CME settlement** (daily variation) | `impl_cme_settlement.py` | `pytest tests/test_cme_settlement.py::TestCMESettlementEngine` |
| **CME rollover** (contract expiry) | `impl_cme_rollover.py` | `pytest tests/test_cme_settlement.py::TestContractRolloverManager` |
| **CME trading calendar** | `services/cme_calendar.py` | `pytest tests/test_cme_calendar.py::TestCMETradingCalendar` |
| **SPAN margin calculator** | `impl_span_margin.py` | `pytest tests/test_span_margin.py` |
| **CME slippage provider** | `execution_providers_cme.py` | `pytest tests/test_cme_slippage.py` |
| **CME circuit breaker** | `impl_circuit_breaker.py` | `pytest tests/test_circuit_breaker.py` |
| **CME SPAN margin guard** | `services/cme_risk_guards.py` | `pytest tests/test_cme_risk_guards.py::TestSPANMarginGuard` |
| **CME position limits** | `services/cme_risk_guards.py` | `pytest tests/test_cme_risk_guards.py::TestCMEPositionLimitGuard` |
| **CME CB aware guard** | `services/cme_risk_guards.py` | `pytest tests/test_cme_risk_guards.py::TestCircuitBreakerAwareGuard` |
| **CME settlement risk** | `services/cme_risk_guards.py` | `pytest tests/test_cme_risk_guards.py::TestSettlementRiskGuard` |
| **CME rollover guard** | `services/cme_risk_guards.py` | `pytest tests/test_cme_risk_guards.py::TestRolloverGuard` |
| **CME unified risk** | `services/cme_risk_guards.py` | `pytest tests/test_cme_risk_guards.py::TestCMEFuturesRiskGuard` |
| **Unified futures risk** | `services/unified_futures_risk.py` | `pytest tests/test_unified_futures_risk.py` |
| **Asset type detection** | `services/unified_futures_risk.py` | `pytest tests/test_unified_futures_risk.py::TestAssetType` |
| **Portfolio risk mgr** | `services/unified_futures_risk.py` | `pytest tests/test_unified_futures_risk.py::TestPortfolioRiskManager` |
| **Futures LOB extensions** | `lob/futures_extensions.py` | `pytest tests/test_futures_l3_execution.py` |
| **Liquidation cascade** | `lob/futures_extensions.py` | `pytest tests/test_futures_l3_execution.py::TestLiquidationCascadeSimulator` |
| **Insurance fund** | `lob/futures_extensions.py` | `pytest tests/test_futures_l3_execution.py::TestInsuranceFundManager` |
| **ADL queue** | `lob/futures_extensions.py` | `pytest tests/test_futures_l3_execution.py::TestADLQueueManager` |
| **Funding dynamics** | `lob/futures_extensions.py` | `pytest tests/test_futures_l3_execution.py::TestFundingPeriodDynamics` |
| **Futures L3 execution** | `execution_providers_futures_l3.py` | `pytest tests/test_futures_l3_execution.py::TestFuturesL3ExecutionProvider` |
| **CME Globex matching** | `lob/cme_matching.py` | `pytest tests/test_cme_l3_execution.py::TestGlobexMatchingEngineBasic` |
| **CME MWP orders** | `lob/cme_matching.py` | `pytest tests/test_cme_l3_execution.py::TestGlobexMatchingEngineMWP` |
| **CME stop orders** | `lob/cme_matching.py` | `pytest tests/test_cme_l3_execution.py::TestGlobexMatchingEngineStops` |
| **CME L3 execution** | `execution_providers_cme_l3.py` | `pytest tests/test_cme_l3_execution.py::TestCMEL3ExecutionProvider` |
| **CME session detection** | `execution_providers_cme_l3.py` | `pytest tests/test_cme_l3_execution.py::TestSessionDetection` |
| **CME daily settlement** | `execution_providers_cme_l3.py` | `pytest tests/test_cme_l3_execution.py::TestDailySettlementSimulator` |
| **Futures leverage guard** | `services/futures_risk_guards.py` | `pytest tests/test_futures_risk_guards.py::TestFuturesLeverageGuard` |
| **Futures margin guard** | `services/futures_risk_guards.py` | `pytest tests/test_futures_risk_guards.py::TestFuturesMarginGuard` |
| **Margin call notifier** | `services/futures_risk_guards.py` | `pytest tests/test_futures_risk_guards.py::TestMarginCallNotifier` |
| **Funding exposure guard** | `services/futures_risk_guards.py` | `pytest tests/test_futures_risk_guards.py::TestFundingExposureGuard` |
| **Concentration guard** | `services/futures_risk_guards.py` | `pytest tests/test_futures_risk_guards.py::TestConcentrationGuard` |
| **ADL risk guard** | `services/futures_risk_guards.py` | `pytest tests/test_futures_risk_guards.py::TestADLRiskGuard` |
| **Crypto futures risk** | `risk_guard.py` | `pytest tests/test_futures_risk_guards.py::TestCryptoFuturesRiskGuard` |
| **Futures env wrapper** | `wrappers/futures_env.py` | `pytest tests/test_futures_training.py::TestFuturesEnvWrapper` |
| **Futures feature flags** | `services/futures_feature_flags.py` | `pytest tests/test_futures_feature_flags.py` |
| **Futures training config** | `configs/config_train_futures.yaml` | `pytest tests/test_futures_training.py::TestFuturesTrainingConfig` |
| **Futures live runner** | `services/futures_live_runner.py` | `pytest tests/test_futures_live_trading.py::TestFuturesLiveRunner` |
| **Futures position sync** | `services/futures_position_sync.py` | `pytest tests/test_futures_live_trading.py::TestFuturesPositionSynchronizer` |
| **Futures margin monitor** | `services/futures_margin_monitor.py` | `pytest tests/test_futures_live_trading.py::TestFuturesMarginMonitor` |
| **Futures funding tracker** | `services/futures_funding_tracker.py` | `pytest tests/test_futures_live_trading.py::TestFuturesFundingTracker` |
| **Futures live config** | `configs/config_live_futures.yaml` | `pytest tests/test_futures_live_trading.py::TestFuturesLiveConfig` |
| **Futures validation** | `tests/test_futures_validation.py` | `pytest tests/test_futures_validation.py` |
| **Futures backward compat** | `tests/test_futures_backward_compatibility.py` | `pytest tests/test_futures_backward_compatibility.py` |
| **Futures benchmarks** | `benchmarks/bench_futures_simulation.py` | `python benchmarks/bench_futures_simulation.py` |
| **Options pricing (BS/Binomial)** | `impl_pricing.py` | `pytest tests/test_options_core.py::TestBlackScholesPricing` |
| **Options Greeks (12)** | `impl_greeks_vectorized.py` | `pytest tests/test_options_core.py::TestScalarGreeks` |
| **Options batch Greeks** | `impl_greeks_vectorized.py` | `pytest tests/test_options_core.py::TestVectorizedGreeks` |
| **Options IV solver** | `impl_iv_calculation.py` | `pytest tests/test_options_core.py::TestIVCalculation` |
| **Options exercise prob** | `impl_exercise_probability.py` | `pytest tests/test_options_core.py::TestExerciseProbability` |
| **Options LSMC** | `impl_exercise_probability.py` | `pytest tests/test_options_core.py::TestLongstaffSchwartzMC` |
| **Options variance swap** | `impl_pricing.py` | `pytest tests/test_options_core.py::TestVarianceSwap` |
| **Options jump diffusion** | `impl_pricing.py` | `pytest tests/test_options_core.py::TestJumpDiffusionPricing` |
| **Options discrete dividends** | `impl_pricing.py` | `pytest tests/test_options_core.py::TestDiscreteDividends` |
| **IB Options adapter** | `adapters/ib/options.py` | `pytest tests/test_options_adapters.py::TestIBOptionsAdapter` |
| **IB Options rate limiter** | `adapters/ib/options_rate_limiter.py` | `pytest tests/test_options_adapters.py::TestIBOptionsRateLimiter` |
| **IB Options combo orders** | `adapters/ib/options_combo.py` | `pytest tests/test_options_adapters.py::TestIBOptionsCombo` |
| **Theta Data options** | `adapters/theta_data/options.py` | `pytest tests/test_deribit_options.py::TestThetaDataAdapter` |
| **Polygon options** | `adapters/polygon/options.py` | `pytest tests/test_options_adapters.py::TestPolygonOptions` |
| **Deribit options** | `adapters/deribit/options.py` | `pytest tests/test_deribit_options.py::TestDeribitMarketData` |
| **Deribit inverse margin** | `adapters/deribit/margin.py` | `pytest tests/test_deribit_options.py::TestInverseMargin` |
| **Deribit WebSocket** | `adapters/deribit/websocket.py` | `pytest tests/test_deribit_options.py::TestWebSocketClient` |
| **Options registry** | `adapters/registry.py` | `pytest tests/test_options_adapters.py::TestOptionsRegistry` |
| **OCC symbology** | `adapters/ib/options.py` | `pytest tests/test_options_adapters.py::TestOCCSymbology` |

### 🔍 Quick File Reference

| Префикс | Слой | Зависимости | Примеры |
|---------|------|-------------|---------|
| `core_*` | Базовый | Нет | `core_config.py`, `core_models.py`, `core_strategy.py` |
| `impl_*` | Реализация | `core_` | `impl_sim_executor.py`, `impl_fees.py`, `impl_slippage.py` |
| `service_*` | Сервисы | `core_`, `impl_` | `service_backtest.py`, `service_train.py`, `service_eval.py` |
| `strategies/*` | Стратегии | Все предыдущие | `strategies/base.py`, `strategies/momentum.py` |
| `script_*` | CLI точки входа | Все | `script_backtest.py`, `script_live.py`, `script_eval.py` |

### 🚀 Quick Start (5 минут до первого бэктеста)

**Полное руководство**: [QUICK_START.md](QUICK_START.md)

**CLI инструмент**: `python scripts/quickstart.py`

```bash
# Проверить готовность среды
python scripts/quickstart.py check crypto_momentum

# Список доступных пресетов
python scripts/quickstart.py list

# Информация о пресете
python scripts/quickstart.py info equity_swing

# Запустить бэктест
python scripts/quickstart.py run crypto_momentum

# Обучить модель
python scripts/quickstart.py train forex_carry
```

### 📦 Reference Pipelines (Quick Start Configs)

| Пресет | Asset Class | Стратегия | Конфиг | Сложность |
|--------|-------------|-----------|--------|-----------|
| `crypto_momentum` | Crypto Spot | Momentum (BTC, ETH) | [crypto_momentum.yaml](configs/quickstart/crypto_momentum.yaml) | ⭐⭐ Beginner |
| `equity_swing` | US Equity | Mean-Reversion (SPY, AAPL) | [equity_swing.yaml](configs/quickstart/equity_swing.yaml) | ⭐⭐ Beginner |
| `forex_carry` | Forex OTC | Carry + Momentum | [forex_carry.yaml](configs/quickstart/forex_carry.yaml) | ⭐⭐⭐ Intermediate |
| `crypto_perp` | Crypto Futures | Funding Arbitrage | [crypto_perp.yaml](configs/quickstart/crypto_perp.yaml) | ⭐⭐⭐⭐ Advanced |
| `cme_index` | CME Futures | Equity Index Momentum | [cme_index.yaml](configs/quickstart/cme_index.yaml) | ⭐⭐⭐⭐⭐ Expert |

**Быстрый старт для каждого asset class**:

```bash
# 🪙 Crypto Spot (Binance) - Beginner friendly
python scripts/quickstart.py run crypto_momentum

# 📈 US Equity (Alpaca) - Commission-free
python scripts/quickstart.py run equity_swing

# 💱 Forex (OANDA) - 24/5 trading
python scripts/quickstart.py run forex_carry

# 🔮 Crypto Perpetuals (Binance USDT-M) - Advanced
python scripts/quickstart.py run crypto_perp

# 🏛️ CME Futures (Interactive Brokers) - Expert
python scripts/quickstart.py run cme_index
```

### 📁 Project Organization (Updated 2025-11-30)

**ВАЖНО**: Проект реорганизован (commit db9655a). Файлы перемещены:

```
AI-Powered-Quantitative-Research-Platform/
├── tests/              # 654+ test files, 14,000+ test functions
│   ├── test_*.py       # All test files
│   └── conftest.py     # Pytest fixtures
├── tools/              # 34 utility scripts (moved from root)
│   ├── check_*.py      # Validation scripts
│   ├── verify_*.py     # Verification scripts
│   └── analyze_*.py    # Analysis scripts
├── scripts/            # Data fetching scripts
│   ├── download_stock_data.py
│   ├── fetch_binance_filters.py
│   └── fetch_alpaca_universe.py
├── lob/                # L3 LOB simulation modules
├── adapters/           # Exchange adapters (Binance, Alpaca, etc.)
├── services/           # Business logic services
├── strategies/         # Trading strategies
├── configs/            # YAML configuration files
├── docs/               # Documentation and archives
└── *.py                # Core modules (core_, impl_, script_, etc.)
```

**Key directories**:
- `tools/` -- Scripts for verification, debugging, analysis (run directly)
- `tests/` -- All pytest tests (use `pytest tests/`)
- `scripts/` -- Data management scripts

### ⚡ Критические команды

```bash
# Quick Start (5 минут до первого бэктеста)
python scripts/quickstart.py list                # Список пресетов
python scripts/quickstart.py check crypto_momentum  # Проверка среды
python scripts/quickstart.py run crypto_momentum    # Бэктест
python scripts/quickstart.py train equity_swing     # Обучение

# Тестирование
pytest tests/                                    # Все тесты
pytest tests/test_execution*.py -v               # Execution тесты
pytest -k "test_name" -v                         # Конкретный тест

# Бэктест/Eval
python script_backtest.py --config configs/config_sim.yaml
python script_eval.py --config configs/config_eval.yaml --all-profiles

# Обучение (standard)
python train_model_multi_patch.py --config configs/config_train.yaml

# Обучение (PBT + Adversarial)
python train_model_multi_patch.py --config configs/config_pbt_adversarial.yaml

# Обновление данных (Crypto)
python scripts/fetch_binance_filters.py --universe --out data/binance_filters.json
python scripts/refresh_fees.py
python -m services.universe --output data/universe/symbols.json

# Обновление данных (Stocks)
python scripts/fetch_alpaca_universe.py --output data/universe/alpaca_symbols.json --popular
python scripts/download_stock_data.py --symbols GLD IAU SGOL SLV --start 2020-01-01 --timeframe 1h --resample 4h

# Обновление данных (VIX / Macro indicators)
python scripts/download_stock_data.py --vix --start 2020-01-01 --timeframe 1d
python scripts/download_stock_data.py --macro --start 2020-01-01 --timeframe 1d
python scripts/download_stock_data.py --symbols ^VIX DX-Y.NYB ^TNX --start 2020-01-01

# Live Trading (Stocks - Alpaca)
python script_live.py --config configs/config_live_alpaca.yaml
python script_live.py --config configs/config_live_alpaca.yaml --asset-class equity --paper
python script_live.py --config configs/config_live_alpaca.yaml --extended-hours

# Live Trading (Crypto - Binance)
python script_live.py --config configs/config_live.yaml

# Training (Stocks)
python train_model_multi_patch.py --config configs/config_train_stocks.yaml

# Backtest (Stocks)
python script_backtest.py --config configs/config_backtest_stocks.yaml

# Training (Forex)
python train_model_multi_patch.py --config configs/config_train_forex.yaml

# Backtest (Forex)
python script_backtest.py --config configs/config_backtest_forex.yaml

# Live Trading (Forex - OANDA)
python script_live.py --config configs/config_live_forex.yaml --asset-class forex
```

---

## 🏗️ CCEA: Cloud-Controlled Execution Architecture

> **Полная документация**: [docs/CCEA_OVERVIEW.md](docs/CCEA_OVERVIEW.md) · канон: [Design_Doc_CCEA_Cloud.txt](docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt) · тесты: `tests/ccea/` (104 файла)

**Ключевой принцип (НЕ НАРУШАТЬ!):**
```
Cloud = research / build / monitoring / control plane (lifecycle requests)
Agent = secrets + live loop + risk enforce + order creation/sending
```

**Cloud НИКОГДА:** не хранит broker API keys · не генерирует/передаёт ордера · не имеет доступа к trading endpoints · не отправляет order-like payload (side/qty/price).

| Zone | Пакеты | Secrets | Orders |
|------|--------|---------|--------|
| **SHARED** | `packages/shared/`, `core_*`, `impl_*`, simulation, features | No | No |
| **AGENT** | `packages/agent/*`: vault, policy, execution, daemon, approval | **Yes** | **Yes** |
| **CLOUD** | `packages/cloud/*`: control_plane, builder, governance, research | No | No |

**Команды Cloud→Agent**: `REQUEST_START_RUN`, `REQUEST_STOP_RUN`, `REQUEST_PAUSE_RUN`, `REQUEST_UPGRADE_ARTIFACT`, `REQUEST_UPDATE_CONFIG`. Запрещены в payload: `side`, `quantity`, `price`, `order_type`, `target_position`, `intent`, `signal`. TRADING_IMPACTING изменения требуют локального approve; hard caps в Agent нельзя ослабить из Cloud. Подробности (терминология, классификация изменений, policy firewall, CI guardrails, threat model, legal posture) — в CCEA_OVERVIEW.md.

**Запуск:**
```bash
python -m packages.agent.daemon --config configs/agent.yaml            # live execution
python -m packages.agent.daemon --config configs/agent.yaml --dry-run  # validate config & exit
python script_live.py --config configs/config_live.yaml --dry-run     # dev/testing only
```

**CCEA в десктопе (локально, без серверов):** `ccea/desktop_supervisor.py` поднимает
control-plane (SQLite) + agentd + Vault (keychain) + paper-брокер; включается флагом
`RIVEN_ENABLE_CCEA=1` (дефолт в десктопе). Эндпоинты `/api/ccea/{status,paper_order,connect_broker}`,
карточка CCEA на Home и Pro-Dashboard. Real paper-RUN, live-broker через Vault,
Ed25519-подпись (TUF/agent-updates/evidence/registry). Полная справка —
[docs/CCEA_DESKTOP.md](docs/CCEA_DESKTOP.md); упаковка — [desktop/README.md](desktop/README.md).

---

## 🗺️ Multi-Asset Capability Map

> Детальная документация по каждому домену (формулы, API, конфиги, test counts) вынесена из этого файла.
> **Полный справочник**: [docs/PLATFORM_REFERENCE.md](docs/PLATFORM_REFERENCE.md). Где есть выделенный док — он авторитетнее.

| Домен | Статус | Ключевые файлы | Док |
|-------|--------|----------------|-----|
| **Multi-Exchange адаптеры** (Binance/Alpaca/Polygon/Yahoo/OANDA/IB) | ✅ | `adapters/*` | PLATFORM_REFERENCE.md |
| **Stocks** (training/backtest/live, Alpaca) | ✅ | `data_loader_multi_asset.py`, `stock_features.py`, `script_live.py` | [STOCK_TRADING_GUIDE.md](docs/STOCK_TRADING_GUIDE.md) |
| **Execution Providers** (L2/L2+/L3) | ✅ | `execution_providers*.py` | PLATFORM_REFERENCE.md |
| **Crypto / Equity Parametric TCA** (L2+) | ✅ | `execution_providers.py` | PLATFORM_REFERENCE.md |
| **Stock Features & Risk Guards** | ✅ | `stock_features.py`, `services/stock_risk_guards.py` | PLATFORM_REFERENCE.md |
| **L3 LOB Simulation** (matching/queue/impact/latency/dark pools) | ✅ | `lob/*`, `execution_providers_l3.py` | [docs/l3_simulator/](docs/l3_simulator/overview.md) |
| **Forex** (OANDA, OTC dealer sim) | ✅ | `adapters/oanda/*`, `forex_features.py`, `services/forex_*.py` | [FOREX_INTEGRATION_PLAN.md](docs/FOREX_INTEGRATION_PLAN.md) |
| **Futures** (Crypto perp + CME via IB, Phases 3B–10) | ✅ | `adapters/ib/*`, `impl_span_margin.py`, `services/*futures*.py`, `execution_providers_*futures*.py` | [docs/futures/](docs/futures/overview.md) |
| **Options** (BS/Greeks/IV/LSMC; IB/Polygon/Deribit/Theta) | ✅ | `impl_pricing.py`, `impl_greeks_vectorized.py`, `adapters/{ib,polygon,deribit,theta_data}/options.py` | [docs/options/](docs/options/core_models.md) |
| **Cross-sectional platform** (signals→risk→optimizer→portfolio) | 🚧 | `signals/`, `service_optimizer.py`, `service_cross_asset.py` | [CROSS_SECTIONAL_PLATFORM_DESIGN.md](CROSS_SECTIONAL_PLATFORM_DESIGN.md) |
| **Firm-wide risk + books-and-records** (consolidated VaR/CVaR, P&L ledger, blotter/cash, instrument master, MAR surveillance) | ✅ | `service_firm_risk.py`, `packages/agent/accounting/{pnl_ledger,blotter,books}.py`, `services/instrument_master.py`, `packages/agent/audit/hash_chain.py` | [PRO_PIPELINE_GAP_ANALYSIS.md](PRO_PIPELINE_GAP_ANALYSIS.md) |

Полная таблица «задача → файл → команда» — в разделе [Быстрый поиск по задачам](#-быстрый-поиск-по-задачам) выше.

---
## 🛡️ Критические правила (НЕ НАРУШАТЬ!)

1. **ActionProto.volume_frac = TARGET position, НЕ DELTA!**
   - ✅ `next_units = volume_frac * max_position`
   - ❌ `next_units = current_units + volume_frac * max_position` (удвоение!)

2. **Action space bounds: [-1, 1] для policy с LongOnlyActionWrapper**
   - ✅ `LongOnlyActionWrapper.action_space = Box(-1, 1)` -- wrapper сам устанавливает!
   - ✅ Policy использует `tanh` когда `action_space.low < 0`
   - ❌ Wrapper НЕ должен наследовать `action_space` от env (было [0,1] → баг!)

3. **LongOnlyActionWrapper: mapping [-1,1] → [0,1], НЕ clipping**
   - ✅ `mapped = (action + 1.0) / 2.0` -- policy выдаёт [-1,1], wrapper маппит в [0,1]
   - ✅ `-1.0 → 0.0` (exit), `0.0 → 0.5` (50%), `+1.0 → 1.0` (100%)
   - ❌ `clipped = max(0, action)` (теряет reduction сигналы)
   - ❌ Если wrapper наследует [0,1] от env: sigmoid [0,1] → mapping → [0.5,1.0] **минимум 50%!**

4. **LSTM States ДОЛЖНЫ сбрасываться на episode boundaries!**
   - ✅ `self._last_lstm_states = self._reset_lstm_states_for_done_envs(...)`
   - ⚠️ **НЕ УДАЛЯЙТЕ** вызов в distributional_ppo.py:7418-7427!

5. **UPGD utility scaling: min-max normalization**
   - ✅ `normalized = (utility - global_min) / (global_max - global_min + eps)`
   - ❌ `scaled = utility / global_max` (инвертируется при negative!)

6. **Gamma synchronization для reward shaping**
   - ✅ `reward.gamma == model.params.gamma` (оба = 0.99)
   - ⚠️ При изменении одного -- обновите другой!

7. **Technical Indicators инициализация**
   - ✅ **RSI**: SMA(14) для первых gains/losses
   - ✅ **CCI**: SMA(TP) для baseline
   - ✅ **ATR**: SMA variant корректен

---

## 🚨 Troubleshooting (актуальные проблемы)

| Симптом | Причина | Решение |
|---------|---------|---------|
| step() возвращает obs с той же row что reset() | Observation строился из current row, не next | ✅ Фикс 2025-11-25: obs из next_row (Gymnasium семантика) |
| CLOSE_TO_OPEN + SIGNAL_ONLY: look-ahead bias | signal_pos обновлялся немедленно, игнорируя delay | ✅ Фикс 2025-11-25: использует executed_signal_pos |
| info["signal_pos_next"] показывает intent, не actual | В CLOSE_TO_OPEN + signal_only показывал agent_signal_pos | ✅ Фикс 2025-11-25: показывает next_signal_pos + новое поле signal_pos_requested |
| LSTM первый step на zeros | reset() возвращал np.zeros() | ✅ Фикс 2025-11-25: reset() строит obs из row 0 |
| reward=0 при старте эпизода | NaN close в первых rows → _last_reward_price=0 | ✅ Фикс 2025-11-25: fallback на open/scan rows |
| Long-only: позиция всегда ≥50% | Wrapper наследовал [0,1] action_space | ✅ Фикс 2025-11-25: wrapper ставит [-1,1], policy использует tanh |
| Long-only: entropy collapse | Policy не может выразить exit | Переобучить с новым wrapper (tanh вместо sigmoid) |
| PBT deadlock (workers crash) | ready_percentage слишком высокий | `min_ready_members=2`, `ready_check_max_wait=10` |
| Non-monotonic quantiles | NN predictions без sorting | `critic.enforce_monotonicity=true` |
| Value loss не снижается | LSTM states не сбрасываются | Проверьте `_reset_lstm_states_for_done_envs` |
| External features = 0.0 | NaN → 0.0 silent conversion | `log_nan=True` для debugging |
| Градиенты взрываются | UPGD noise слишком высок | Уменьшите `sigma` (0.0005-0.001) |
| `AttributeError` в конфигах | Pydantic V2 API | `model_dump()` вместо `dict()` |
| Feature mismatch | Online/offline паритет | `check_feature_parity.py` |
| PBT state mismatch | VGS не синхронизирован | Проверьте `variance_gradient_scaler.py` state dict |
| step() IndexError при пустом df | Нет защиты от пустого DataFrame | ✅ Фикс 2025-11-25: проверка len(df)==0 в step() |
| signal_pos в obs отстаёт от market data | Obs содержал prev_signal_pos (t), но market data из t+1 | ✅ Фикс 2025-11-26: obs содержит next_signal_pos (t+1) |
| VGS + AdaptiveUPGD: noise 212x amplification | EMA (beta=0.999) слишком медленно адаптируется к VGS scaling | ✅ Фикс 2025-11-26: `instant_noise_scale=True` (default) |
| FG=50 (neutral) treated as missing data | `abs(value-50.0)>0.1` check false negative | ✅ Фикс 2025-11-26: uses `_get_safe_float_with_validity()` |
| UPGDW: inverted weight protection | Only tracked max_util, not min_util | ✅ Фикс 2025-11-26: min-max normalization like AdaptiveUPGD |
| Episode continues with stale data | row_idx clamped to last row instead of truncation | ✅ Фикс 2025-11-26: returns truncated=True when data exhausted |
| cql_beta=0 causes NaN/Inf | No validation for cql_beta divisor | ✅ Фикс 2025-11-26: ValueError if cql_beta <= 0 |
| Twin Critics categorical VF clipping no effect | `_project_distribution` was identity stub | ✅ Фикс 2025-11-26: uses `_project_categorical_distribution` |
| Yang-Zhang volatility inflated ~11% for n=10 | RS component used (n-1) instead of n | ✅ Фикс 2025-11-26: RS now uses n per original formula |
| `_project_categorical_distribution` shape error | 1D atoms not expanded to batch_size | ✅ Фикс 2025-11-26: proper batch expansion |
| Limit order fills missed for high-price assets | Fixed tolerance 1e-12 < machine epsilon at $100k | ✅ Фикс 2025-11-26: `_compute_price_tolerance` с relative tolerance |
| EV≈0, Twin Critics loss +327%, grad norm -82% | VGS alpha=0.1 даёт 91% редукцию градиентов при высокой variance | ✅ Фикс 2025-11-27: VGS v3.2 с `min_scaling_factor=0.1`, `variance_cap=50.0` |
| DarkPoolSimulator memory leak | `_leakage_history`, `_fill_history` росли unbounded | ✅ Фикс 2025-11-27: `deque(maxlen=max_history_size)` |
| DarkPoolConfig division by zero | `impact_size_normalization=0` не валидировался | ✅ Фикс 2025-11-27: `__post_init__` validation |
| DarkPoolSimulator TypeError on deque slice | `_should_block_for_leakage` использовал slice на deque | ✅ Фикс 2025-11-27: convert to list before slicing |
| VIX/SPY/QQQ benchmark temporal misalignment | Positional indexing вместо timestamp merge → look-ahead | ✅ Фикс 2025-11-29: `merge_asof(direction="backward")` |

---

## ✅ Закрытые вопросы и НЕ-БАГИ (читать перед "исправлением"!)

> **Полные записи** (с кодом и обоснованием): [docs/NOT_BUGS_AND_FAQ.md](docs/NOT_BUGS_AND_FAQ.md)

В коде есть паттерны, которые **выглядят как баги** при статическом анализе, но являются **корректными и намеренными**. Также есть вопросы, уже тщательно проанализированные и **закрытые**. Перед тем как "чинить" что-либо в перечисленных ниже областях — прочитайте соответствующую запись в `docs/NOT_BUGS_AND_FAQ.md`. **НЕ переоткрывайте закрытые вопросы и НЕ "исправляйте" намеренные паттерны.**

**Области с намеренными паттернами (НЕ трогать без чтения дока):**
- **PPO/Distributional** (`distributional_ppo.py`): episode_starts off-by-one (SB3), VGS перед grad-clip, CVaR interpolation weight=0.5, CVaR tail_mass/extrapolation, Twin Critics loss averaging без VF-clip, advantage norm ddof=1, CVaR ~16% approx error.
- **TradingEnv** (`trading_patchnew.py`, `mediator.py`): obs из NEXT row (Gymnasium), CLOSE_TO_OPEN+signal_only delayed position, первые 2 step reward≈0, signal_only terminated=False, signal_pos в obs = next_signal_pos, reward clipping НЕ stacked, ratio не clipped в signal_only.
- **Action space** (`wrappers/action_space.py`, `custom_policy_patch1.py`): LongOnlyActionWrapper маппинг [-1,1]→[0,1] (НЕ clipping), policy tanh/sigmoid adaptive activation, 4-sample entropy, GRU vs LSTM пути.
- **Optimizers** (`optimizers/`): UPGDW global_max_util=-inf, AdaptiveUPGD grad_norm_ema=1.0 warmup (instant_noise_scale обходит EMA), VGS `_param_ids` dead code.
- **Execution/Slippage** (`execution_sim.py`, `impl_slippage.py`, `execution_providers.py`): limit maker fill logic, fee на filled price (не double-count), slippage на mid-price (impact term есть), L2 vs L2+/L3 trade-offs (ADV/impact/spread/fills/whale-threshold), latency clamping configurable, нет LOB depth tracking (by design).
- **Features/Risk** (`features_pipeline.py`, `obs_builder.pyx`, `risk_guard.py`, `services/ops_kill_switch.py`, `transformers.py`): синхронный shift всех фич, winsorization до z-score, RSI valid на bar 14, vol_proxy=0.01 fallback, FG=50 vs missing различимы, asymmetric risk buffer, kill-switch crash recovery, boundscheck=False (Cython).

Если сомневаетесь — запись в доке объясняет, почему это корректно, и приводит тесты.

---
## 📊 СТАТУС ПРОЕКТА (2025-12-16)

### ✅ Инженерный статус (внутренний)

Все критические исправления применены. Актуальный статус и полноту покрытия верифицировать запуском тестов в репозитории (например, `pytest`). Alignment/evidence tooling (not audited/certified). CCEA implemented.

#### 🚩 Pro-readiness блокеры P0/P1/P2 — ЗАКРЫТЫ

> Полные записи: [P0_BLOCKERS_CLOSURE.md](P0_BLOCKERS_CLOSURE.md), [P1_BLOCKERS_CLOSURE.md](P1_BLOCKERS_CLOSURE.md), [P2_BLOCKERS_CLOSURE.md](P2_BLOCKERS_CLOSURE.md), эндпойнты/панели — [MVP_DOCUMENTATION.md](MVP_DOCUMENTATION.md) §7.

**P2 (зрелость/масштаб):** каталог сигналов 26→32 (residual mom/seasonality/sentiment/52w/idio-vol/COT) + COT/calendar enrichers (`signals/common_signals.py`, `loaders/altdata_enrich.py`); Feature Store (версии name/asof/hash + online-кэш, `service_feature_store.py`); FIX 4.4 + Smart Order Routing (`packages/agent/execution/fix_protocol.py`, `smart_order_router.py`); TS-DB абстракция ClickHouse/Timescale/parquet (`services/tsdb.py`); автоматизация drift-ретрейн + авто-TCA + e2e GDPR/DORA (`services/automation/`); cross-asset единый портфель (C1, `service_cross_asset.py`) и options greeks-оптимизатор (B5, `service_options_portfolio.py`) — уже были. **43 новых теста.**

| Блокер | Код | MVP-UI | Ключевые файлы |
|--------|-----|--------|----------------|
| **P0-1** Реальный crypto/equity бэктест (вместо синтетики) | ✅ | ✅ кнопки | `tools/xs_crypto_real_sweep.py`, `tools/xs_equity_real_report.py`, `configs/config_xs_*_real.yaml`, `reports/XS_*_REAL_TRUST_REPORT.md` |
| **P0-2** PIT-данные equity (SEC EDGAR free) + index-membership | ✅ | — | `services/edgar_fundamentals.py`, `services/index_membership_loader.py`, `scripts/download_edgar_fundamentals.py` |
| **P0-3** Честность MVP (no mock-as-real) | ✅ | ✅ бейджи | `app.py` (флаги), `index.html` (`showSimBadge`), `tools/check_mvp_honesty.py` |
| **P0-4** Experiment tracking + Model registry (Ed25519) | ✅ | ✅ панель «3b. MLOps» | `core_experiment.py`, `service_experiment_tracking.py`, `tools/experiment_cli.py` |
| **P1-1** Live pre-trade VaR/CVaR/стресс + factor-monitor | ✅ | ✅ панель | `service_pretrade_risk.py` |
| **P1-2** tcost в objective (scipy) + Kelly/vol-target | ✅ | ✅ тумблеры | `service_optimizer.py` |
| **P1-3** Execution-алго TWAP/VWAP/POV (нарезка w*−w₀) | ✅ | ✅ панель | `service_xs_execution.py` |
| **P1-4** Авто-recovery (retry/circuit-breaker/poll/reconcile) | ✅ | ✅ индикатор | `packages/agent/execution/resilience.py` |
| **P1-5** RL-as-signal (μ-вклад + conformal, без volume_frac в xs) | ✅ | ✅ чекбокс | `impl_rl_signal.py`, `service_rl_inference.py` |

Новые REST: `/api/experiments*`, `/api/models*`, `/api/xs/real/run`, `/api/xs/real/analyze`,
`/api/xs/pretrade_risk`, `/api/xs/execution_plan`, `/api/agent/recovery/status`.
Auth: middleware на всех `/api/*` (`RIVEN_API_AUTH_MODE`, default `loopback`).
Тесты P0/P1: `tests/test_experiment_tracking.py`, `test_edgar_fundamentals.py`,
`test_index_membership.py`, `test_pretrade_risk.py`, `test_optimizer_tcost.py`,
`test_xs_execution.py`, `test_execution_resilience.py`, `test_xs_live_p1.py`.

#### 🏛️ Pro-pipeline P0 (institutional gaps) — ЗАКРЫТЫ (2026-06-15)

> Источник: аудит полного пайплайна квант-фонда — [PRO_PIPELINE_GAP_ANALYSIS.md](PRO_PIPELINE_GAP_ANALYSIS.md).
> Все 5 верхнеприоритетных P0 реализованы, протестированы и **интегрированы в MVP** (REST + Pro-карточки).

| # | Блокер | Код | MVP-UI | Ключевые файлы | Тесты |
|---|--------|-----|--------|----------------|-------|
| **PP-1** | Firm-wide иерархический risk-агрегатор (strategy→desk→firm consolidated VaR/CVaR, Euler component/marginal/incremental, diversification) | ✅ | ✅ карточка «Firm-Wide Risk» (Pro) | `service_firm_risk.py` | `test_firm_risk.py` (16) |
| **PP-2** | Live P&L ledger в Agent (realized/unrealized/fees/financing, day/EOD NAV, avg-cost+FIFO, crash-recovery, equity из ledger) | ✅ | ✅ P&L-блок в CCEA-карточках (Home+Pro) | `packages/agent/accounting/pnl_ledger.py` | `test_pnl_ledger.py` (15) |
| **PP-3** | Instrument master / symbology (FIGI/CUSIP/ISIN/SEDOL/OCC, check-digit валидаторы, OCC parse/build, OpenFIGI) | ✅ | ✅ lookup в карточке «Books & Records» | `services/instrument_master.py` | `test_instrument_master.py` (19) |
| **PP-4** | Market-abuse surveillance (MAR) + tamper-evident журнал в live-путь | ✅ | ✅ surveillance+integrity-бейджи | `services/algo_integration/market_abuse.py` (wired), `packages/agent/audit/hash_chain.py`, `journal.py` (`order_audit`) | `test_journal_tamper_evident.py` (4) |
| **PP-5** | Immutable trade blotter + cash ledger (books-and-records, hash-chained) | ✅ | ✅ blotter/cash таблицы | `packages/agent/accounting/blotter.py`, `books.py` | `test_books_and_records.py` (16) |

Новые REST (PP): `/api/firm_risk/{aggregate,demo}`, `/api/agent/pnl/{status,nav_history,eod_close}`,
`/api/instruments/{resolve,search,list,occ_parse}`, `/api/surveillance/market_abuse`,
`/api/agent/{blotter,cash_ledger,journal/integrity}`. Фасад `BooksAndRecords` (`packages/agent/accounting/books.py`)
связывает P&L-ledger + blotter + cash-GL + instrument-master + surveillance единым `on_fill()`/`on_order()`; подключён
в CCEA-супервизор (HMAC-ключ tamper-chains = vault master key). **70 новых тестов.**

#### ⚡ Pro-pipeline P1 (оживление написанного / live-wiring) — ЗАКРЫТЫ (2026-06-15)

> Источник: [PRO_PIPELINE_GAP_ANALYSIS.md](PRO_PIPELINE_GAP_ANALYSIS.md) §8 P1. Реализовано, протестировано, в MVP (REST + Pro-карточки).

| # | Возможность | Код | MVP | Тесты |
|---|-------------|-----|-----|-------|
| **P1-6** | Optimizer config: sector/factor caps, robust, BL-views, multi-period, beta-neutral из YAML | ✅ | `/api/xs/optimize` | `test_optimizer_config_p1.py` (7) |
| **P1-7** | SOR в live submission (routed_broker_submit + dispatch + live-liquidity) | ✅ | карточка «Execution & Data-QA» | `test_sor_live_p1.py` (4) |
| **P1-8** | CPCV/PBO в sweeps + block-bootstrap CI (Politis–Romano) | ✅ | trust_report в backtest | `test_bootstrap_pbo_p1.py` (8) |
| **P1-9** | Monte-Carlo VaR (Gaussian/t) + Euler component/marginal/incremental + named scenarios (2008/2020/…) | ✅ | `/api/xs/pretrade_risk` | `test_pretrade_risk_p1.py` (7) |
| **P1-10** | Live IS (Almgren-Chriss) executor + engine cancel/replace + FIX 35=G + price-collar/fat-finger | ✅ | CCEA-супервизор, `/api/xs/execution_plan` | `test_execution_p1.py` (11) |
| **P1-11** | Data-QA (spike/staleness/frozen/gap/OHLC) + cross-vendor + MarketDataRouter failover | ✅ | карточка «Execution & Data-QA» | `test_market_data_quality_p1.py` (13) |

Ключевые файлы P1: `service_pretrade_risk.py` (MC/Euler/scenarios), `service_xs_pipeline.py`+`service_optimizer.py`
(config-wiring + `MultiPeriodOptimizer.solve`), `research/bootstrap.py`, `service_xs_execution.py` (`_is_profile`),
`packages/agent/execution/{engine.py (PriceCollarConfig, cancel/replace), fix_protocol.py (35=G), live_factory.py
(routed_broker_submit, BrokerLiquidityProvider), smart_order_router.py}`, `services/market_data_quality.py`.
Новые REST (P1): `/api/exec/route` (+dispatch), `/api/data_quality/{check,demo}`; pretrade/optimize/execution_plan
расширены. **50 новых тестов.**

| Компонент | Статус | Тесты |
|-----------|--------|-------|
| Step Observation Timing | ✅ Production | 6/6 |
| Signal Pos in Observation | ✅ Production | 10/10 |
| CLOSE_TO_OPEN Timing | ✅ Production | 5/5 |
| LongOnlyActionWrapper | ✅ Production | 26/26 |
| AdaptiveUPGD Optimizer | ✅ Production | 119/121 |
| UPGDW Optimizer | ✅ Production | 4/4 |
| Twin Critics + VF Clipping | ✅ Production | 49/50 |
| VGS v3.1 | ✅ Production | 7/7 |
| PBT | ✅ Production | 14/14 |
| SA-PPO | ✅ Production | 16/16 |
| Data Leakage Prevention | ✅ Production | 46/47 |
| Technical Indicators | ✅ Production | 11/16 (C++ pending) |
| Fear & Greed Detection | ✅ Production | 13/13 |
| Crypto Parametric TCA | ✅ Production | 84/84 |
| Equity Parametric TCA | ✅ Production | 86/86 |
| Bug Fixes 2025-11-26 | ✅ Production | 22/22 (includes projection+YZ fixes) |
| **Forex Integration** | ✅ Production | 18 test files (Phase 11) |
| Forex Parametric TCA | ✅ Production | In test_forex_parametric_tca.py |
| OANDA Adapter | ✅ Production | In test_forex_foundation.py |
| **CCEA Architecture** | ✅ Production | 104 test files (All Design Doc requirements implemented) |
| CCEA Agent Zone | ✅ Production | packages/agent/ (53 files) |
| CCEA Cloud Zone | ✅ Production | packages/cloud/ (95+ files) |
| CCEA Guardrails | ✅ Production | tests/ccea/guardrails/ |

### ⚠️ Требуется действие

**Переобучите модели**, если они обучены **до 2025-11-26**:
- **UPGDW min-max normalization fix (2025-11-26)** -- weight protection inverted with negative utilities!
- **Fear & Greed detection fix (2025-11-26)** -- FG=50 ошибочно помечался как missing data!
- **signal_pos in observation fix (2025-11-26)** -- obs содержал prev_signal_pos (t), но market data из t+1!
- **step() observation timing fix (2025-11-25)** -- obs был из той же row что reset!
- **CLOSE_TO_OPEN + SIGNAL_ONLY fix (2025-11-25)** -- look-ahead bias в signal position
- **LongOnlyActionWrapper action space fix (2025-11-25)** -- минимальная позиция была 50%!
- Data leakage fix (2025-11-23) + close_orig fix (2025-11-25)
- RSI/CCI initialization fixes (2025-11-24)
- Twin Critics GAE fix (2025-11-21)
- LSTM state reset fix (2025-11-21)
- UPGD negative utility fix (2025-11-21)

---

## 📜 История критических исправлений

> **Примечание**: Все отчёты перемещены в `docs/archive/`. Путь: `docs/archive/reports_2025_11_25_cleanup/root_reports/`

| Дата | Исправление | Влияние |
|------|-------------|---------|
| **2026-07-16** | fix(P0-A/D/E): битые импорты + `configs/agent.yaml` + Ed25519-гейт в демоне | **P0-A**: `packages.shared.models` (TimeFrame→`core_models`, OrderSide/PositionSide→`core_futures`, +4 имени в contracts), `adapters.theta_data` (Bar→`core_models`), `services.compliance` (graceful degrade mifid-архива, `ARCHIVE_AVAILABLE`). **P0-D**: `configs/agent.yaml` (полная схема) + фикс `build_daemon_config` (stale `DegradedModeConfig` поля→реальные, Decimal kill-switch пороги) + smoke `--dry-run` + фикс команды запуска. **P0-E**: `packages/agent/daemon/model_gate.py` + `RunController._verify_model_signature` — тот же `verify_model_artifact`, что у RL-загрузчика, на пути активации артефакта демона, fail-closed для LIVE ДО pickle. MVP: `/api/agent/daemon/config` + карточка Pro Security. 16 новых тестов (agent_config 6 + agent_model_signature 10) + compat-facade (29 pass/9 skip) + live smoke. См. [docs/P0_ADE_CLOSURE_2026-07-16.md](docs/P0_ADE_CLOSURE_2026-07-16.md) |
| **2026-07-16** | feat(crypto): Binance **spot** order-execution адаптер (P0-C, §3.4 — crypto live/panic был невозможен) | Закрыт последний недостающий execution-путь: `adapters/binance/order_execution.py` (`BinanceOrderExecutionAdapter`, spot `/api/v3/*`, HMAC/RestBudgetSession по образцу futures — submit/cancel/status/open-orders/cancel-all/positions-из-балансов/account/last-price + `submit_spot_order` со STOP/TAKE_PROFIT), зарегистрирован для `BINANCE`+`BINANCE_US`. Проводка: panic-halt crypto-ветка теперь реально флэттенит (балансы→синтетические `{asset}USDT`-пары→market-SELL), holdings/close уже звали фабрику. Spot-семантика: long-only, нет leverage/reduceOnly, `avg_entry_price=0` (cost basis честно недоступен). UI различает Binance Spot/Futures в Vault. 17 тестов (вкл. интеграционный panic-flatten) + live smoke; снят lock старого fail-closed поведения. См. [docs/CRYPTO_SPOT_EXECUTION.md](docs/CRYPTO_SPOT_EXECUTION.md) |
| **2026-07-16** | feat(risk): enforcement риск-лимитов Lite в live-контуре (P0-B, §3.6 — «самый опасный» гэп) | Двухуровневая защита: pre-trade RiskChecker с leverage/drawdown/daily-loss/concentration из `lite_limits` + intra-day `LiveRiskMonitor` circuit breaker (day-loss/max-DD → auto-halt kill switch + флэттенинг + отзыв live-мандатов). `packages/agent/policy/risk_checker.py` (новые `LEVERAGE`/`MAX_DRAWDOWN` checks, обратно совместимо), `services/live_risk_limits.py` (loader+builder+monitor, durable peak equity), проводка в `ccea/desktop_supervisor.py` (оба движка + on_fill hook + reload без рестарта + EOD reset). REST `/api/risk/enforcement`, обновлённые `/api/risk/limits` (честный `applied_to_agent`) + `/api/panic_reset`. Lite-карточка «Применение лимитов (live)» (usage-бары + ARMED/BREACHED). 16 тестов + live smoke (armed→pre-trade block→breach→auto-halt→reset). См. [docs/RISK_LIMIT_ENFORCEMENT.md](docs/RISK_LIMIT_ENFORCEMENT.md) |
| **2026-07-16** | feat(trade): ручной ордер-тикет + частичное закрытие позиций (§5.27–28) | `submit_manual_order` (market/limit/stop/stop-limit, TIF, reduce-only) + `close_position(quantity)` partial + `open_orders`/`cancel_order` в `ccea/desktop_supervisor.py`; REST `/api/ccea/order/{submit,cancel}`, `/api/ccea/open_orders`, `/api/portfolio/close` partial; UI-карточки «Ордер-тикет» + «Рабочие ордера» в Lite Portfolio; всё через настоящий Agent OMS (firewall/journal/fill/books). Журнал сделан thread-safe (check_same_thread=False + lock). 25 тестов + live smoke. См. [docs/MANUAL_ORDER_TICKET.md](docs/MANUAL_ORDER_TICKET.md) |
| **2026-07-15** | feat(trade): авто-торговля XS-ребаланса на LIVE-брокере через CCEA operator-approval | `packages/agent/approval/live_trading_authorization.py` (мандат: хеш конфига + брокер + потолок лимитов + TTL + бюджет, hash-chained аудит Agent-зоны, hard-caps); проводка в `ccea/desktop_supervisor.py` (grant/revoke/status, `_ensure_live_engine`, `submit_rebalance_order(allow_live)`, авто-revoke при halt/смене брокера); гейт в `service_xs_rebalance` (precheck→clamp→final check→consume); REST `/api/ccea/live_trading/{request,grant,revoke,status}` (двухшаговая церемония, anti-replay). 43 теста + live smoke. См. [docs/MODEL_SIGNATURE_AND_REBALANCE.md](docs/MODEL_SIGNATURE_AND_REBALANCE.md) §3 |
| **2026-07-15** | feat(governance+trade): Ed25519-гейт подписи моделей в live (§4.7) + регулярный XS-ребаланс (§4.9/P1-C) | `services/model_signature_gate.py` (enforce/warn/off, fail-closed до pickle-десериализации; проводка в `service_rl_inference`) + `service_xs_rebalance.py` (веса→turnover-cap/no-trade-band/концентрация→Intents→CCEA Agent OMS→журнал решений) + `submit_rebalance_order` в supervisor + REST `/api/xs/rebalance/*`, `/api/models/verify_for_live` + планировщик job `xs_rebalance` (боевой). 36 тестов + live smoke (8/8 ордеров через реальный OMS). См. [docs/MODEL_SIGNATURE_AND_REBALANCE.md](docs/MODEL_SIGNATURE_AND_REBALANCE.md) |
| **2026-07-15** | feat(scheduler): планировщик регулярных задач (P0-F гэп-анализа) | `services/scheduler.py` + `configs/scheduler.yaml` + `/api/scheduler/*` + Lite-карточка: anacron catch-up для десктопа, fail-closed research-пайплайн (LeakGuard-пол 8000 не ослабляется), drift→retrain c durable cooldown, авто-EOD отчёт, бэкапы/ротация логов, ретраи+алерты (Telegram/webhook), CCEA-гейт торговых задач (двойной opt-in). 24 теста (`tests/test_scheduler_service.py`) + live smoke. См. [docs/SCHEDULER.md](docs/SCHEDULER.md) |
| **2026-07-15** | fix(lite): закрыты все 24 дефекта аудита Lite Mode 2026-07-14 (L2-001…L2-024) | Fail-closed Emergency Halt (без выдуманных ликвидаций), Quick Start на backend evidence (`/api/workflow/readiness`), контракты Data Manager (run_no_trade/--sandbox_config/--data/price_col), LeakGuard-floor 8000 мс с журналируемым override, SB3 data files в PyInstaller, typed `/api/risk/limits`, честные Gas Guard/VaR/PSI/slippage/HFT-статус, terminal-lifecycle jobs, asset-aware Quant Lab, `configs/config_backtest_futures.yaml`, code_root/data_root split, DEMO-метки. 39 новых тестов + реальный E2E цепочки (`tests/test_lite_mode_audit_2026_07_14.py`, `tests/test_lite_chain_e2e.py`). См. [docs/LITE_MODE_AUDIT_CLOSURE_2026-07-15.md](docs/LITE_MODE_AUDIT_CLOSURE_2026-07-15.md) |
| **2026-06-15** | feat(P1 6–11): optimizer config-wiring (sector/factor caps/robust/BL/multi-period/beta-neutral) + SOR в live-сабмит + CPCV/PBO+bootstrap CI + MC-VaR/Euler/named-scenarios + IS-executor/FIX 35=G/price-collar + data-QA/vendor-failover | `service_pretrade_risk.py`, `service_optimizer.py`, `research/bootstrap.py`, `service_xs_execution.py`, `packages/agent/execution/{engine,fix_protocol,live_factory}.py`, `services/market_data_quality.py`; REST `/api/{exec/route+dispatch,data_quality/*}`; Pro-карточка «Execution & Data-QA»; 50 тестов. См. [PRO_PIPELINE_GAP_ANALYSIS.md](PRO_PIPELINE_GAP_ANALYSIS.md) |
| **2026-06-15** | feat(PP-3/4/5): instrument master (FIGI/CUSIP/ISIN/OCC) + live MAR surveillance + tamper-evident журнал + immutable trade blotter & cash ledger (hash-chained) | `services/instrument_master.py`, `packages/agent/audit/hash_chain.py`, `packages/agent/accounting/{blotter,books}.py`, `journal.py` (`order_audit`); REST `/api/{instruments,surveillance/market_abuse,agent/{blotter,cash_ledger,journal/integrity}}`; Pro-карточка «Books & Records»; 39 тестов. См. [PRO_PIPELINE_GAP_ANALYSIS.md](PRO_PIPELINE_GAP_ANALYSIS.md) |
| **2026-06-15** | feat(PP-1/2): firm-wide иерархический risk-агрегатор (consolidated VaR/CVaR + Euler attribution) + live P&L ledger в Agent (realized/unrealized + EOD NAV) | `service_firm_risk.py`, `packages/agent/accounting/pnl_ledger.py`; REST `/api/firm_risk/*`, `/api/agent/pnl/*`; Pro-карточка «Firm-Wide Risk» + P&L-блок в CCEA; 31 тест. См. [PRO_PIPELINE_GAP_ANALYSIS.md](PRO_PIPELINE_GAP_ANALYSIS.md) |
| **2026-06-15** | feat(ccea-desktop): живой CCEA в десктопе — real paper-RUN (Intent→firewall→journal→fill→PnL), live-broker через Vault, Ed25519 TUF + dev/prod fallback (105 sign-тестов), CCEA-карточка Home+Pro, эндпоинты `/api/ccea/{status,paper_order,connect_broker}`. См. [docs/CCEA_DESKTOP.md](docs/CCEA_DESKTOP.md) |
| **2026-06-15** | feat(desktop): нативное приложение Tauri v2 + Python-сайдкар (Win11+macOS), `desktop/`, `desktop_backend.py`, `packaging/riven_backend.spec`; MVP 1:1, NSIS-инсталлятор. См. [desktop/README.md](desktop/README.md) |
| **2026-06-15** | fix(security/mvp): copilot XSS-escape, terminal RCE-gating (loopback/env), `/api/system_state` enum+lock, loopback-auth без доверия proxy-хедерам, path-traversal realpath-confine, очистка tmp-конфигов |
| **2026-06-14** | feat(P2): сигналы 26→32 + COT/calendar, feature store, FIX+SOR, TS-DB, drift-ретрейн/TCA/GDPR-DORA e2e | зрелость+масштаб; cross-asset(C1)/options-greeks(B5) уже были; 43 теста. См. P2_BLOCKERS_CLOSURE.md |
| **2026-06-14** | feat(P1): live risk + execution + recovery + tcost-opt + RL-signal (код+MVP-UI) | VaR/CVaR/стресс, TWAP/VWAP/POV slices, retry/circuit-breaker, tcost-в-objective+Kelly; 45 тестов. См. P1_BLOCKERS_CLOSURE.md |
| **2026-06-14** | feat(P0): real backtests + EDGAR PIT + MVP honesty + MLOps registry (код+MVP-UI) | реальный edge вместо синтетики, SEC EDGAR PIT-фундаментал, бейджи simulated/demo, Ed25519 model-registry. См. P0_BLOCKERS_CLOSURE.md |
| **2026-06-14** | sec: global API auth (`RIVEN_API_AUTH_MODE`) + codegen injection fix + frontend XSS/escape | закрыт неаутентифицированный RCE-вектор на `/api/*`, инъекция в calibration codegen, XSS в index.html |
| **2025-11-30** | feat(forex): Phase 11 Forex Integration complete | L2+ parametric TCA, OANDA adapter, 18 test files |
| **2025-11-30** | feat(futures): Unified multi-asset futures plan | 1,035+ tests planned for crypto/equity/commodity futures |
| **2025-11-29** | fix(stocks): Benchmark temporal alignment via merge_asof | VIX/SPY/QQQ used positional index → look-ahead bias for equities |
| **2025-11-28** | feat(equity): EquityParametricSlippageProvider | L2+ smart TCA model for US equities, 9 factors, 86 tests |
| **2025-11-28** | feat(crypto): CryptoParametricSlippageProvider | L2+ smart TCA model with 6 factors, 84 tests |
| **2025-11-27** | Stage 6: DarkPoolSimulator memory leak fix | unbounded List → deque(maxlen=N), prevents OOM in long simulations |
| **2025-11-27** | Stage 6: DarkPoolConfig validation | Division by zero prevented with ValueError for invalid params |
| **2025-11-27** | Stage 6: deque slice fix in _should_block_for_leakage | TypeError on deque slicing → convert to list first |
| **2025-11-27** | VGS v3.2: min_scaling_factor + variance_cap | EV≈0, Twin Critics loss +327%, grad norm -82% → VGS не блокирует обучение |
| **2025-11-26** | Twin Critics categorical VF clipping projection fix | `_project_distribution` was identity stub → now uses proper C51 projection |
| **2025-11-26** | Yang-Zhang RS denominator fix | RS used (n-1) instead of n → +11% inflation for n=10 removed |
| **2025-11-26** | `_project_categorical_distribution` batch shape fix | Shape mismatch for 1D atoms with batched probs → properly expands |
| **2025-11-26** | UPGDW min-max normalization fix | Negative utilities no longer invert weight protection |
| **2025-11-26** | Data exhaustion truncation fix | Episode properly ends with truncated=True when data runs out |
| **2025-11-26** | cql_beta validation fix | Division by zero prevented with ValueError for cql_beta <= 0 |
| **2025-11-26** | Mediator dead code removal | Removed unreachable `is None` check (code smell) |
| **2025-11-26** | Fear & Greed detection fix | FG=50 (neutral) correctly detected as valid data, not missing |
| **2025-11-26** | AdaptiveUPGD instant_noise_scale fix | VGS + UPGD noise 212x amplification → 1.0x (constant ratio) |
| **2025-11-26** | signal_pos in observation uses next_signal_pos | Temporal mismatch: market data t+1, position t → теперь оба t+1 |
| **2025-11-26** | Limit order tolerance fix | Fixed 1e-12 < machine epsilon at $100k → relative tolerance |
| **2025-11-25** | Empty DataFrame protection in step() | IndexError при пустом df → graceful termination |
| **2025-11-25** | step() observation from NEXT row (Gymnasium) | Duplicate obs: reset() и step()#1 возвращали одну row |
| **2025-11-25** | CLOSE_TO_OPEN + SIGNAL_ONLY timing | Look-ahead bias: signal_pos игнорировал 1-bar delay |
| **2025-11-25** | info["signal_pos_next"] consistency | Показывал intent вместо actual; добавлен signal_pos_requested |
| **2025-11-25** | reset() returns actual observation (Issue #1) | LSTM получал zeros на первом step эпизода |
| **2025-11-25** | Improved _last_reward_price init (Issue #3) | reward=0 если данные начинались с NaN |
| **2025-11-25** | Removed redundant signal_position update (Issue #2) | Code smell (не влияло на функционал) |
| **2025-11-25** | LongOnlyActionWrapper action space | Минимальная позиция была 50% вместо 0%! |
| **2025-11-25** | Policy adaptive activation (tanh/sigmoid) | Policy теперь адаптируется к action_space |
| **2025-11-25** | close_orig semantic conflict | Data leakage в pipeline |
| **2025-11-24** | Twin Critics loss aggregation | 25% underestimation |
| **2025-11-24** | RSI/CCI initialization | 5-20x error first 150 bars |
| **2025-11-23** | Data leakage (all features) | Look-ahead bias |
| **2025-11-23** | VGS v3.1 E[g²] computation | 10,000x underestimation |
| **2025-11-23** | SA-PPO epsilon + KL | Schedule + 10x faster |
| **2025-11-23** | GAE overflow protection | Float32 overflow |
| **2025-11-22** | PBT deadlock prevention | Indefinite wait |
| **2025-11-22** | Twin Critics VF Clipping | Independent critic updates |
| **2025-11-21** | Twin Critics GAE | min(Q1,Q2) not applied |
| **2025-11-21** | LSTM state reset | Temporal leakage 5-15% |
| **2025-11-21** | UPGD negative utility | Inverted weight protection |
| **2025-11-21** | Action space (3 bugs) | Position doubling |
| **2025-11-20** | Numerical stability (5 bugs) | Gradient explosions |
| **2025-11-20** | Feature engineering (3 bugs) | Volatility bias 1-5% |

---

## О проекте

**CustodiaCloud** — B2B software/ICT платформа для количественных исследований и контролируемого развёртывания в средах клиента. Архитектура **CCEA**: Cloud выполняет research/monitoring/lifecycle-requests; live execution (если используется) выполняется только локальным **Agent** в среде клиента; Cloud не хранит секреты и не отправляет live trading instructions (orders/targets/signals).

### Основные характеристики

- **Язык**: Python 3.12 + Cython + C++
- **RL Framework**: Stable-Baselines3 (Distributional PPO with Twin Critics)
- **Optimizer**: AdaptiveUPGD (default) -- continual learning
- **Gradient Scaling**: VGS v3.2 -- automatic per-layer normalization + anti-blocking
- **Training**: PBT + SA-PPO (adversarial training)
- **Интеграции**: equities-first (MVP); options/futures/FX/digital assets — опциональное расширение
- **Режимы**: обучение/бэктест/симуляция; production live execution — через Agent
- **Канон формулировок**: `docs/DOCUMENTATION_CANON_DESIGN.md` (+ тех. границы CCEA: `archive/root_files/Design Doc CCEA Cloud.txt`)

---

## 🚀 Продвинутые возможности

### Quick Reference: Training Configuration

```yaml
# configs/config_train.yaml
model:
  algo: "ppo"
  optimizer_class: AdaptiveUPGD
  optimizer_kwargs:
    lr: 1.0e-4
    weight_decay: 0.001
    sigma: 0.001                       # CRITICAL для VGS
    beta_utility: 0.999
    beta1: 0.9
    beta2: 0.999

  vgs:
    enabled: true
    accumulation_steps: 4
    warmup_steps: 10
    clip_threshold: 10.0

  params:
    use_twin_critics: true             # Default: enabled
    num_atoms: 21
    v_min: -10.0
    v_max: 10.0
    cvar_alpha: 0.05
    cvar_weight: 0.15
    clip_range_vf: 0.7
    gamma: 0.99                        # Must match reward.gamma!
    gae_lambda: 0.95
    clip_range: 0.10
    ent_coef: 0.001
    vf_coef: 1.8
    max_grad_norm: 0.5
```

### 1. UPGD Optimizer

**Статус**: ✅ Tested and operational | **Default**: Enabled (AdaptiveUPGD)

Continual learning optimizer для предотвращения catastrophic forgetting.

**Варианты**: AdaptiveUPGD (рекомендуется), UPGD, UPGDW

**Документация**: [docs/UPGD_INTEGRATION.md](docs/UPGD_INTEGRATION.md)

### 2. Twin Critics

**Статус**: ✅ Tested and operational | **Default**: Enabled

Две независимые value networks для снижения overestimation bias.

```
[Observation] → [LSTM] → [MLP] → [Critic Head 1] → [Value 1]
                                ↘ [Critic Head 2] → [Value 2]
Target Value = min(Value 1, Value 2)
```

**Документация**: [docs/twin_critics.md](docs/twin_critics.md)

### 3. VGS (Variance Gradient Scaler)

**Статус**: ✅ Tested and operational | **Version**: v3.1

Автоматическое масштабирование градиентов на основе стохастической вариации.

**Важно**: При использовании с UPGD установите `sigma` в диапазоне 0.0005-0.001.

### 4. PBT (Population-Based Training)

**Статус**: ✅ Tested and operational

Эволюционная оптимизация гиперпараметров через популяцию агентов.

```yaml
pbt:
  enabled: true
  population_size: 8
  perturbation_interval: 10
  min_ready_members: 2          # Deadlock prevention
  ready_check_max_wait: 10
```

### 5. SA-PPO (State-Adversarial PPO)

**Статус**: ✅ Tested and operational

Robust training через adversarial perturbations (PGD attack).

```yaml
adversarial:
  enabled: true
  perturbation:
    epsilon: 0.075
    attack_steps: 3
    attack_lr: 0.03
```

### 6. Conformal Prediction

**Статус**: ✅ Tested and operational | **Тесты**: 59 (at documentation time; verify via CI)

Distribution-free uncertainty bounds на CVaR и value estimates.

**Методы**:
- **CQR** (Conformalized Quantile Regression) -- Romano et al., 2019
- **EnbPI** (Ensemble batch Prediction Intervals) -- Xu & Xie, ICML 2021
- **ACI** (Adaptive Conformal Inference) -- Gibbs & Candes, 2021

**Архитектура**:
```
core_conformal.py → impl_conformal.py → service_conformal.py
```

**Конфигурация** (`configs/conformal.yaml`):
```yaml
conformal:
  enabled: true
  calibration:
    method: "cqr"           # cqr, enbpi, aci, naive
    coverage_target: 0.90   # P(Y ∈ interval) ≥ 90%
    min_calibration_samples: 500
    recalibrate_interval: 1000
  cvar_bounds:
    enabled: true
    use_for_gae: false      # Conservative, experimental
  risk_integration:
    enabled: true
    uncertainty_position_scaling: true
    baseline_interval_width: 0.1
    max_uncertainty_reduction: 0.5
  escalation:
    enabled: true
    warning_percentile: 90
    critical_percentile: 99
    action_on_warning: "log"
    action_on_critical: "reduce_position"
```

**Использование**:
```python
from service_conformal import (
    ConformalPredictionService,
    create_conformal_config,
    wrap_cvar_with_bounds,
    create_risk_guard_integration,
)

# 1. Создание сервиса из YAML
config = create_conformal_config(yaml_dict["conformal"])
service = ConformalPredictionService(config)

# 2. Калибровка после training
service.calibrate(predictions, true_values)

# 3. Получение prediction interval
interval = service.predict_interval(point_estimate)
print(f"[{interval.lower_bound:.3f}, {interval.upper_bound:.3f}]")

# 4. CVaR bounds
bounds = service.compute_cvar_bounds(quantiles)
print(f"CVaR worst-case: {bounds.worst_case_cvar:.3f}")

# 5. Position scaling
scale = service.get_position_scale()  # 0.5-1.0 based on uncertainty

# 6. Integration с risk_guard
position_scale_fn = create_risk_guard_integration(service, lambda: 1.0)
```

**Тестирование**:
```bash
pytest tests/test_conformal_prediction.py -v
```

**Референсы**:
- Romano et al. (2019): [CQR](https://arxiv.org/abs/1905.03222)
- Xu & Xie (ICML 2021): EnbPI
- Gibbs & Candes (2021): ACI
- MAPIE: https://mapie.readthedocs.io/

---

## Архитектура проекта

### CCEA (Cloud-Controlled Execution Architecture)

Основная архитектура проекта: **CCEA** - строгое разделение Cloud и Agent.

**Эталонный документ**: [Design_Doc_CCEA_Cloud.txt](docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt)

#### Компоненты CCEA

| Компонент | Путь | Ответственность | Secrets | Orders |
|-----------|------|-----------------|---------|--------|
| **Cloud** | `packages/cloud/` | Research, Builder, Control Plane, Governance | **NEVER** | **NEVER** |
| **Agent** | `packages/agent/` | Vault, Execution, Policy, Daemon | **LOCAL** | **YES** |
| **Shared** | `packages/shared/`, `core_*`, `impl_*` | Safe for both runtimes | N/A | N/A |

#### Key Security Design Commitments

- Cloud **НИКОГДА** не хранит broker API keys
- Cloud **НИКОГДА** не создает, не передает, не исполняет ордера
- Cloud **НИКОГДА** не имеет доступа к trading endpoints
- Agent - **ЕДИНСТВЕННАЯ** зона исполнения
- Telemetry **ВСЕГДА** редактируется перед отправкой в Cloud

#### Legal Posture

| Мы являемся | Мы НЕ являемся |
|-------------|----------------|
| Software Provider / ICT Provider | Investment Adviser |
| Quantitative research & deployment tools | Broker-Dealer |
| Strategy development platform | Custodian |
| Infrastructure for client-controlled execution | Execution Service |

#### Product Modes

| Режим | Cloud | Agent | Use Case |
|-------|-------|-------|----------|
| **Cloud + BYO Agent (B2B)** | Full (IDE, backtest, sim) | Optional | Research + deploy-to-Agent workflows |
| **Enterprise on‑prem/VPC** | Self-hosted option | HSM/KMS, air-gapped | On-prem/VPC deployments |

#### CCEA Terminology (Эталон)

| Термин | Определение |
|--------|------------|
| **Cloud** | SaaS сервисы (research, backtesting, monitoring, control plane) |
| **Agent** | Daemon клиента в его окружении (BYO host / VPS / on-prem) |
| **Strategy** | User code/model, производит Intent |
| **Intent** | High-level intention (target exposure/position), NOT a ready order |
| **Order** | Concrete broker instruction (создается ТОЛЬКО в Agent) |
| **Deployment** | Связь между artifact + config + target Agent |
| **Run** | Конкретное исполнение стратегии на Agent |
| **Command** | Lifecycle request от Cloud к Agent (REQUEST_START, REQUEST_STOP, etc.) |
| **TRADING_IMPACTING** | Класс изменений, требующих local approval |
| **NON_IMPACTING** | Класс изменений, применяемых автоматически |

#### Lifecycle Commands (Cloud → Agent)

| Command | Description | TRADING_IMPACTING |
|---------|-------------|-------------------|
| REQUEST_START_RUN | Запустить стратегию | YES |
| REQUEST_STOP_RUN | Остановить стратегию | NO (safety) |
| REQUEST_PAUSE_RUN | Приостановить | NO (safety) |
| REQUEST_UPDATE_CONFIG | Изменить config | Depends on field |
| REQUEST_UPGRADE_ARTIFACT | Развернуть новую версию | YES |
| REQUEST_ROTATE_AGENT_SESSION | Ротация ключей | YES |
| REQUEST_EXPORT_LOGS | Экспорт логов | YES (data_sensitive) |

#### Threat Model (Design Doc §15)

| Угроза | Митигация |
|--------|-----------|
| RCE in Cloud | Cloud не имеет trading libs, broker APIs |
| Key Exfiltration | Keys designed to remain in Agent; redaction designed as mandatory (verify via architecture review and pen-test) |
| Artifact Tampering | Digest pinning, signature verification, SBOM |
| Cloud Becomes Execution | Schema prohibits order-like payloads |
| Compute Abuse | Sandbox, CPU/RAM/time quotas, egress allowlist |

#### Safe Defaults (Design Doc §15)

*Note: Defaults enforced by design; verify via CI tests.*

| Setting | Default | Override |
|---------|---------|----------|
| Telemetry Redaction | ON | **NO** (mandatory by design) |
| Local Approval for Trading-Impacting | REQUIRED | Only stricter |
| RAW Order Telemetry | OFF | Enterprise opt-in |
| Artifact Signature Verification | REQUIRED | **NO** |
| Auto-Approve | DISABLED | Local policy only |

#### Config Layering (Design Doc §12)

```
Priority (highest wins):
1. LOCAL_HARD_CAPS     ← Agent enforced, immutable
2. LOCAL_POLICY        ← User's local policy
3. ARTIFACT_RISK       ← Strategy's risk_profile_suggested
4. CLOUD_CONFIG        ← Remote configuration
5. DEFAULTS            ← System defaults
```

**Документация CCEA**: [docs/CCEA_OVERVIEW.md](docs/CCEA_OVERVIEW.md), [Design_Doc_CCEA_Cloud.txt](docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt)

### Слои кода

Источник правды по слоям: `ARCHITECTURE.md` (обновляется вместе с кодом).
- Карта слоёв и допустимых зависимостей: см. `ARCHITECTURE.md#слои`.
- Примеры ключевых файлов и конфигураций запусков: см. `ARCHITECTURE.md#конфигурации-запусков`.
- Быстрый ориентир: слои `core_ → impl_ → service_ → strategies → script_`; зависимости строим только слева направо.

---

## Основные компоненты

### 1. Симулятор исполнения

`execution_sim.py` -- симуляция LOB, микроструктура, проскальзывание, комиссии.

Алгоритмы: TWAP, POV, VWAP

### 2. Distributional PPO

`distributional_ppo.py` -- PPO с:
- Distributional value head (quantile regression)
- Twin Critics (default enabled)
- VGS gradient scaling
- AdaptiveUPGD optimizer
- CVaR risk-aware learning

### 3. Features Pipeline

`features_pipeline.py` -- препроцессинг с проверкой паритета.

63 features: price, volume, volatility, momentum, microstructure.

### 4. Риск-менеджмент

`risk_guard.py` -- гварды на позицию/PnL/дроудаун.

`services/ops_kill_switch.py` -- операционный kill switch.

---

## Конфигурации

### Основные конфиги

| Файл | Назначение |
|------|------------|
| `config_train.yaml` | Обучение crypto (standard) |
| `config_train_stocks.yaml` | Обучение stocks (Alpaca) |
| `config_train_signal_only_stocks.yaml` | Signal-only обучение stocks |
| `config_pbt_adversarial.yaml` | PBT + SA-PPO |
| `config_sim.yaml` | Бэктест crypto |
| `config_backtest_stocks.yaml` | Бэктест stocks |
| `config_live.yaml` | Live trading crypto (Binance) |
| `config_live_alpaca.yaml` | Live trading stocks (Alpaca) |
| `config_eval.yaml` | Оценка модели |
| `config_train_forex.yaml` | Обучение forex (OANDA) |
| `config_backtest_forex.yaml` | Бэктест forex |

### Quick Start конфиги (Reference Pipelines)

| Файл | Asset Class | Описание |
|------|-------------|----------|
| `quickstart/crypto_momentum.yaml` | Crypto Spot | Momentum на BTC/ETH (Binance) |
| `quickstart/equity_swing.yaml` | US Equity | Mean-reversion на SPY/AAPL (Alpaca) |
| `quickstart/forex_carry.yaml` | Forex OTC | Carry + Momentum (OANDA) |
| `quickstart/crypto_perp.yaml` | Crypto Futures | Funding Arbitrage (Binance USDT-M) |
| `quickstart/cme_index.yaml` | CME Futures | Equity Index Momentum (IB) |

### Asset Class конфигурация

| Файл | Назначение |
|------|------------|
| `asset_class_defaults.yaml` | Defaults для crypto/equity/forex/futures |
| `forex_defaults.yaml` | Forex-specific defaults (spreads, sessions, leverage) |
| `exchange.yaml` | Exchange adapter configuration |

### Модульные конфиги

| Файл | Назначение |
|------|------------|
| `execution.yaml` | Execution simulation parameters |
| `execution_l3.yaml` | L3 LOB execution configuration |
| `fees.yaml` | Fee structures (maker/taker, regulatory) |
| `slippage.yaml` | Slippage profiles (crypto, equity) |
| `risk.yaml` | Risk limits and guards |
| `no_trade.yaml` | No-trade windows |
| `conformal.yaml` | Conformal prediction settings |
| `signal_quality.yaml` | Signal quality metrics |

---

## CLI Примеры

```bash
# Бэктест
python script_backtest.py --config configs/config_sim.yaml

# Обучение
python train_model_multi_patch.py --config configs/config_train.yaml

# PBT + Adversarial
python train_model_multi_patch.py --config configs/config_pbt_adversarial.yaml

# Live trading
python script_live.py --config configs/config_live.yaml

# Оценка
python script_eval.py --config configs/config_eval.yaml --all-profiles

# Обновление данных
python scripts/fetch_binance_filters.py --universe --out data/binance_filters.json
python scripts/refresh_fees.py
```

---

## Тестирование

```bash
pytest tests/                          # Все тесты
pytest tests/test_twin_critics*.py -v  # Twin Critics
pytest tests/test_upgd*.py -v          # UPGD
pytest tests/test_pbt*.py -v           # PBT
```

### Ключевые тестовые файлы

| Категория | Файлы |
|-----------|-------|
| **Core Unit Tests** | `test_unit_train_model_multi_patch.py` (192 теста), `test_unit_trading_patchnew.py` (114 тестов), `test_unit_custom_policy_patch1.py` (106 тестов) |
| Twin Critics | `test_twin_critics*.py` (49 тестов) |
| UPGD | `test_upgd*.py` (119 тестов) |
| VGS | `test_vgs*.py` (7 тестов) |
| Data Leakage | `test_data_leakage*.py`, `test_close_orig*.py` |
| Indicators | `test_indicator*.py`, `test_rsi_cci*.py` |
| Action Space | `test_critical_action_space_fixes.py`, `test_long_only_action_space_fix.py` (26+21 тестов) |
| LSTM | `test_lstm_episode_boundary_reset.py` |
| Reset Observation | `test_trading_env_reset_observation_fixes.py` (9 тестов) |
| Phase 9 Live Trading | `test_phase9_live_trading.py` (46 тестов) |
| Stock Features | `test_stock_features.py`, `test_benchmark_temporal_alignment.py` |
| Stock Risk Guards | `test_stock_risk_guards.py` |
| US Market Structure | `test_us_market_structure.py` |

---

## Документация

### Основная

- [DOCS_INDEX.md](DOCS_INDEX.md) -- Индекс документации
- [ARCHITECTURE.md](ARCHITECTURE.md) -- Архитектура
- [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) -- Сборка
- [docs/PLATFORM_REFERENCE.md](docs/PLATFORM_REFERENCE.md) -- **Детальный справочник по asset-классам** (вынесен из этого файла): Multi-Exchange, Stocks, Execution Providers, Crypto/Equity TCA, Stock Features, Live Trading
- [docs/NOT_BUGS_AND_FAQ.md](docs/NOT_BUGS_AND_FAQ.md) -- **Закрытые вопросы и НЕ-баги** (вынесено): намеренные паттерны кода + settled questions

### По доменам

- [docs/l3_simulator/overview.md](docs/l3_simulator/overview.md) -- L3 LOB simulation
- [docs/FOREX_INTEGRATION_PLAN.md](docs/FOREX_INTEGRATION_PLAN.md) -- Forex (OANDA)
- [docs/futures/overview.md](docs/futures/overview.md) -- Futures (Crypto perp + CME/IB)
- [docs/options/core_models.md](docs/options/core_models.md) -- Options (pricing/Greeks/adapters)
- [docs/STOCK_TRADING_GUIDE.md](docs/STOCK_TRADING_GUIDE.md) -- Stocks
- [docs/CCEA_OVERVIEW.md](docs/CCEA_OVERVIEW.md) -- CCEA architecture

### Продвинутые возможности

- [docs/UPGD_INTEGRATION.md](docs/UPGD_INTEGRATION.md) -- UPGD Optimizer
- [docs/twin_critics.md](docs/twin_critics.md) -- Twin Critics
- [docs/pipeline.md](docs/pipeline.md) -- Decision pipeline
- [docs/bar_execution.md](docs/bar_execution.md) -- Bar execution

### Бизнес и IP защита

- [docs/business/IP_PROTECTION_STRATEGY.md](docs/business/IP_PROTECTION_STRATEGY.md) -- Стратегия защиты интеллектуальной собственности
- [docs/business/INVESTOR_IP_SUMMARY.md](docs/business/INVESTOR_IP_SUMMARY.md) -- Краткое изложение IP для инвесторов
- [docs/business/COMPETITIVE_MOAT.md](docs/business/COMPETITIVE_MOAT.md) -- Количественный анализ конкурентного moat (Morgan Stanley framework)
- [docs/business/PATENT_CLAIMS_DRAFT.md](docs/business/PATENT_CLAIMS_DRAFT.md) -- Черновик патентных заявок
- [docs/business/TRADE_SECRET_POLICY.md](docs/business/TRADE_SECRET_POLICY.md) -- Политика защиты коммерческой тайны
- [docs/business/OPEN_CORE_BUSINESS_MODEL.md](docs/business/OPEN_CORE_BUSINESS_MODEL.md) -- Модель Open-Core бизнеса
- [docs/business/PUBLIC_SDK_README_TEMPLATE.md](docs/business/PUBLIC_SDK_README_TEMPLATE.md) -- Шаблон README для публичного SDK
- [docs/business/PRICING_DIFFERENTIATION_STRATEGY.md](docs/business/PRICING_DIFFERENTIATION_STRATEGY.md) -- Стратегия ценовой дифференциации по сегментам клиентов
- [docs/business/SALES_CHANNEL_EVOLUTION_STRATEGY.md](docs/business/SALES_CHANNEL_EVOLUTION_STRATEGY.md) -- Стратегия эволюции каналов продаж (от founder-led к multi-channel)

### Кейсы и социальное доказательство

- [docs/business/PROJECTED_CASE_STUDIES.md](docs/business/PROJECTED_CASE_STUDIES.md) -- Проектируемые кейсы клиентов с research-backed метриками
- [docs/business/CUSTOMER_VALUE_FRAMEWORK.md](docs/business/CUSTOMER_VALUE_FRAMEWORK.md) -- Фреймворк расчёта ROI и TCO для клиентов
- [docs/business/BUILD_VS_BUY_ANALYSIS.md](docs/business/BUILD_VS_BUY_ANALYSIS.md) -- Анализ Build vs Buy с COCOMO II методологией
- [docs/business/TESTIMONIAL_ACQUISITION_STRATEGY.md](docs/business/TESTIMONIAL_ACQUISITION_STRATEGY.md) -- Стратегия получения реальных отзывов и кейсов

### Отчёты об исправлениях

**Все отчёты перенесены в архив:**
- Основной архив: `docs/archive/reports_2025_11_25_cleanup/`
- Критические исправления: `docs/archive/reports_2025_11_25_cleanup/root_reports/`
- Верификация: `docs/archive/verification_2025_11/`

---

## Важные переменные окружения

```bash
BINANCE_API_KEY, BINANCE_API_SECRET     # API ключи
TB_FAIL_ON_STALE_FILTERS=1              # Fail при устаревших фильтрах
BINANCE_PUBLIC_FEES_DISABLE_AUTO=1      # Отключить автообновление fees
```

---

## Production Checklist

### Данные и конфигурация
- [ ] Обновлены фильтры (`fetch_binance_filters.py`)
- [ ] Обновлены комиссии (`refresh_fees.py`)
- [ ] Проверены risk limits (`risk.yaml`)
- [ ] Проверены no-trade окна (`no_trade.yaml`)

### ML Модель
- [ ] AdaptiveUPGD настроен
- [ ] VGS enabled, warmup настроен
- [ ] Twin Critics enabled
- [ ] `gamma` синхронизирован (reward = model)
- [ ] **Long-only**: wrapper устанавливает [-1,1], policy использует tanh
- [ ] Model trained after 2025-11-25

### Тестирование
- [ ] `pytest tests/` -- все тесты проходят
- [ ] `python tools/check_feature_parity.py` -- паритет OK
- [ ] `python tools/verify_fixes.py` -- все фиксы работают

### Live Trading
- [ ] API ключи настроены
- [ ] Kill switch протестирован
- [ ] Мониторинг настроен

---

## Заключение

### Золотые правила

1. **Следуйте слоистой архитектуре**
2. **Читайте файлы перед изменением**
3. **Пишите тесты для критичной логики**
4. **Проверяйте feature parity**
5. **Мониторьте метрики**

### Когда что-то идёт не так

1. Проверьте тесты для проблемной области
2. Используйте Glob/Grep для поиска
3. Проверьте конфиги
4. Проверьте слоистую архитектуру
5. Изучите историю исправлений (таблица выше)

---

**Последнее обновление**: 2026-06-15
**Версия документации**: 14.3 (Pro-pipeline P0+P1: firm-risk/P&L-ledger/instrument-master/MAR-surveillance/books-and-records + optimizer-config-wiring/SOR-live/CPCV-PBO-bootstrap/MC-VaR-Euler-scenarios/IS-FIX35G-price-collar/data-QA-failover — см. [PRO_PIPELINE_GAP_ANALYSIS.md](PRO_PIPELINE_GAP_ANALYSIS.md))
**Статус**: ✅ Инженерный статус (верифицировать тестами) | alignment/evidence tooling (not audited/certified) | CCEA implemented + запускается в десктопе (paper-RUN/live-broker/Ed25519-signing) + books-and-records (P&L ledger/blotter/cash GL/surveillance, hash-chained)

> **Примечание о структуре (v14.0):** этот файл намеренно компактный (~64k симв., лимит 150k), т.к. загружается в контекст каждой сессии. Детальная справка по asset-классам вынесена в [docs/PLATFORM_REFERENCE.md](docs/PLATFORM_REFERENCE.md), записи о НЕ-багах — в [docs/NOT_BUGS_AND_FAQ.md](docs/NOT_BUGS_AND_FAQ.md). Doc-версионный changelog убран (история — в git). При добавлении детальной документации кладите её в `docs/`, а здесь оставляйте только краткий указатель.

