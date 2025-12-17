# AI-Powered Quantitative Research Platform

## Общая информация

**Назначение**: Унифицированная платформа для алгоритмической торговли с использованием Reinforcement Learning (designed for production use)

**Статистика проекта**:
- Python файлов: 1,366
- Строк кода: ~340,000
- Тестов: 14,000+
- Покрытие тестами: 97%+

**Поддерживаемые классы активов**:
- Криптовалюты (спот и фьючерсы)
- Акции США
- Форекс
- CME фьючерсы
- Опционы

---

## Архитектура системы

```
Scripts (train_model_multi_patch.py, script_backtest.py, script_live.py)
    |
Services (service_train.py, service_backtest.py, service_signal_runner.py)
    |
Strategies & Policies (distributional_ppo.py)
    |
Implementations (impl_*.py: slippage, fees, latency, margin)
    |
Core Contracts & Models (core_*.py)
```

**Зависимости**: `core_` -> `impl_` -> `service_` -> `strategies` -> `scripts_`

---

## Основные модули

### Core модули (core_*.py)

| Модуль | Описание |
|--------|----------|
| `core_models.py` | Доменные модели: OrderIntent, Instrument, Bar, Position, TradeLogRow |
| `core_config.py` | Pydantic конфигурации и DI компоненты |
| `core_contracts.py` | Протоколы: FeaturePipe, SignalPolicy |
| `core_events.py` | События рынка |
| `core_futures.py` | Модели для фьючерсов |
| `core_options.py` | Греки и ценообразование опционов |
| `core_conformal.py` | Conformal prediction framework |
| `core_errors.py` | Типы исключений |

### Implementations (impl_*.py)

| Модуль | Описание |
|--------|----------|
| `impl_slippage.py` | Market impact и моделирование проскальзывания (L2/L3) |
| `impl_fees.py` | Расчет комиссий |
| `impl_latency.py` | Симуляция сетевой задержки |
| `impl_sim_executor.py` | Симулятор исполнения ордеров |
| `impl_bar_executor.py` | Исполнение на барах |
| `impl_quantizer.py` | Квантование цены/размера |
| `impl_offline_data.py` | Загрузка исторических данных |
| `impl_greeks*.py` | Расчет греков Black-Scholes |
| `impl_iv_calculation.py` | Расчет implied volatility |
| `impl_futures_*.py` | Маржа, фандинг, ликвидация фьючерсов |
| `impl_circuit_breaker.py` | Остановки торговли |

### Сервисы (service_*.py)

| Модуль | Размер | Описание |
|--------|--------|----------|
| `service_train.py` | 17KB | Оркестрация обучения модели |
| `service_backtest.py` | 75KB | Движок бэктестинга |
| `service_signal_runner.py` | 379KB | Исполнение live-торговли |
| `service_eval.py` | - | Оценка стратегий и метрики |
| `service_conformal.py` | - | Conformal prediction сервис |
| `service_calibrate_slippage.py` | - | Калибровка проскальзывания |
| `service_calibrate_tcost.py` | - | Калибровка транзакционных издержек |

### Execution движки

| Модуль | Размер | Описание |
|--------|--------|----------|
| `execution_sim.py` | 570KB | High-fidelity симулятор стакана |
| `execution_providers.py` | 131KB | Multi-asset execution providers (L2) |
| `execution_algos.py` | - | TWAP, POV алгоритмы |
| `execution_providers_*.py` | - | Провайдеры для фьючерсов, опционов, CME |

---

## Machine Learning

### Distributional PPO (distributional_ppo.py - 654KB)

**Архитектура**:
- Quantile regression value head (21-51 атомов)
- Twin critics для снижения overestimation bias
- CVaR risk-aware learning (фокус на худших 5%)
- LSTM state reset на границах эпизодов
- AdaptiveUPGD оптимизатор
- VGS (Variance Gradient Scaler)
- Population-Based Training (PBT)
- Adversarial training (SA-PPO)

**Связанные файлы**:
- `train_model_multi_patch.py` - точка входа для обучения
- `custom_policy_patch1.py` - кастомизации политики
- `variance_gradient_scaler.py` - нормализация градиентов

### Feature Engineering (51+ технических индикаторов)

**Категории признаков**:
- Momentum: SMA, EMA, ROC, RSI, MACD
- Volatility: Bollinger Bands, ATR, std_dev
- Trend: ADX, DI+, DI-
- Volume: OBV, CMF, ADL
- Microstructure: спреды, имбаланс, order flow
- Time-of-day seasonality
- Regime detection: volatility regimes, trend regimes

**Модули**:
- `feature_pipe.py` (39KB) - pipeline признаков
- `features_pipeline.py` (53KB) - transformer-based features
- `transformers.py` (73KB) - трансформеры признаков

---

## Simulation моделирование

### L1 - Constant Model
- Фиксированный спред/комиссия (baseline)

### L2 - Statistical Model (Production Default)
- Almgren-Chriss market impact: √participation impact
- Multi-factor TCA модель:
  - Market cap tier adjustments
  - Time-of-day liquidity curves
  - Volatility regime scaling
  - Order book imbalance
  - Funding rate stress (crypto)
  - Circuit breaker awareness (CME)

### L3 - Full Order Book Simulation
- Реконструкция стакана
- Queue position tracking
- LOB depth analysis

---

## Биржевые адаптеры (/adapters/)

### Базовые модули

| Файл | Описание |
|------|----------|
| `base.py` | Абстрактные базовые классы |
| `models.py` | Exchange-agnostic модели данных |
| `config.py` | Конфигурация адаптеров |
| `registry.py` | Фабрика и регистрация адаптеров |
| `websocket_base.py` | Production async WebSocket |

### Поддерживаемые биржи

| Адаптер | Класс активов | Функционал |
|---------|---------------|------------|
| `binance/` | Crypto spot/futures | Полная интеграция API, 24/7 |
| `alpaca/` | US equities | Paper и live торговля, extended hours |
| `polygon/` | Stock market data | Исторические данные |
| `oanda/` | Forex (OTC) | Real-time спреды, session-aware pricing |
| `ib/` | CME futures, options | Маржа, расчеты |
| `deribit/` | Crypto options | Опционы на криптовалюты |
| `yahoo/` | Historical equity data | Исторические данные акций |
| `theta_data/` | Options data | Данные по опционам |
| `dukascopy/` | Forex data | Исторические forex данные |
| `ig/` | IG Markets | CFD торговля |

---

## Сервисы бизнес-логики (/services/)

### Core сервисы

| Модуль | Описание |
|--------|----------|
| `services/core/` | Risk management, position sync, state management |
| `services/backtest/` | Оркестрация бэктестинга |
| `services/broker/` | Абстракции интеграции с брокерами |
| `services/api/` | REST API для сигналов |
| `services/healthcheck.py` | Мониторинг здоровья системы |
| `services/monitoring.py` (64KB) | Real-time метрики и KPI |

### Рыночные и риск сервисы

| Модуль | Описание |
|--------|----------|
| `services/forex_*.py` (8 модулей) | Forex-специфичные сервисы |
| `services/futures_*.py` (7 модулей) | Маржа, фандинг, риски фьючерсов |
| `services/stock_risk_guards.py` | Риски акций |
| `services/cme_risk_guards.py` | Риски CME фьючерсов |
| `services/pdt_guard.py` | Pattern Day Trader правила |
| `services/portfolio_constraints.py` | Лимиты портфеля |
| `services/corporate_actions.py` | Дивиденды/сплиты |
| `services/earnings_calendar.py` | Календарь отчетностей |
| `services/trading_halts.py` | Остановки торговли |

### Данные и состояние

| Модуль | Описание |
|--------|----------|
| `services/state_storage.py` | Персистентность состояния (JSON/SQLite) |
| `services/universe.py` | Управление universe символов |
| `services/survivorship.py` | Обработка survivorship bias |
| `services/session_router.py` | Маршрутизация торговых сессий |

---

## Regulatory Compliance

### MiFID II (Designed to Align with Requirements - All Phases Implemented)

| Модуль | Требование |
|--------|------------|
| `compliance_clock.py` | Синхронизация часов (RTS 25) |
| `lei_manager.py` | Legal Entity Identifier |
| `gleif_client.py` | GLEIF API интеграция |
| `algorithm_registry.py` | Регистрация алгоритмов (RTS 28) |
| `transaction_report.py` | Trade reporting (RTS 22) |
| `pre_trade_controls.py` | Risk controls (RTS 6) |
| `enhanced_kill_switch.py` | Circuit breaker (Article 12) |
| `audit_trail_writer.py` | Audit trail (5-7 лет) |
| `best_execution.py` | Качество исполнения (Article 27) |
| `tca_compliance.py` | Transaction Cost Analysis |
| `governance.py` | Governance framework |
| `self_assessment.py` | Годовая самооценка |
| `bcp.py` | Business Continuity Plan |
| `nca_notification.py` | NCA уведомления |

### EU AI Act (Designed to Align with Requirements - 1,007 tests)

| Модуль | Описание |
|--------|----------|
| `accuracy_metrics.py` | Отслеживание точности модели |
| `conformity_assessment.py` | Conformity assessment report |
| `cybersecurity.py` | Требования кибербезопасности |
| `data_governance.py` | Data governance framework |
| `data_lineage.py` | Data lineage tracking |
| `explainability.py` | Прозрачность модели |
| `gpai_model_card.py` | GPAI model cards |
| `human_oversight.py` | Human-in-the-loop controls |
| `logging_system.py` | Аудит логирования |
| `qms.py` | Quality Management System |
| `risk_management.py` | Система управления рисками |
| `robustness_testing.py` | Adversarial testing |
| `technical_documentation.py` | Техническая документация |
| `post_market_monitoring.py` | Post-market мониторинг |

### DORA (Designed to Align with Requirements - ~1,015 tests)

| Фаза | Описание |
|------|----------|
| Phase 0 | Proportionality assessment |
| Phase 1 | ICT Risk Management Framework |
| Phase 2 | ICT Incident Management & Reporting |
| Phase 3 | Digital Resilience Testing (TLPT) |
| Phase 4 | Third-Party ICT Risk Management |
| Phase 5 | Information Sharing & Unified Reporting |

---

## Risk Management

### Risk Guards

| Компонент | Описание |
|-----------|----------|
| `risk_guard.py` (79KB) | Enforcement лимитов рисков |
| Position limits | Лимиты позиций |
| Leverage limits | Лимиты плеча |
| Daily loss limits | Дневные лимиты убытков |
| Drawdown controls | Контроль просадки |
| Volatility halt triggers | Триггеры остановки при волатильности |
| No-trade mask | Запрет торговли в определенные периоды |

### Risk Monitoring

- Real-time P&L tracking
- VAR/CVaR расчет
- Liquidity monitoring
- Counterparty risk (DORA)

---

## Data Models

### OrderIntent
```
ts: int (ms UTC)
symbol: str
side: Side (BUY/SELL)
order_type: OrderType (MARKET/LIMIT)
volume_frac: Decimal [-1.0, 1.0]
price_offset_ticks: int
time_in_force: TimeInForce (GTC/IOC/FOK)
client_tag: str
meta: Dict
```

### Instrument
```
symbol: str
base_asset: str
quote_asset: str
tick_size: Decimal
step_size: Decimal
min_notional: Decimal
price_scale: int
qty_scale: int
filters: Dict
```

### Bar
```
ts: int (ms UTC)
symbol: str
open: Decimal
high: Decimal
low: Decimal
close: Decimal
volume_base: Decimal
volume_quote: Decimal
trades: int
```

### Position
```
symbol: str
quantity: Decimal
entry_price: Decimal
mark_price: Decimal
unrealized_pnl: Decimal
realized_pnl: Decimal
```

### TradeLogRow
```
ts: int
run_id: str
symbol: str
side: Side
order_type: OrderType
price: Decimal
quantity: Decimal
fee: Decimal
exec_status: ExecStatus
liquidity: Liquidity
pnl: Decimal
equity: Decimal
```

### MarketType (Enum)
```
CRYPTO_SPOT, CRYPTO_FUTURES, CRYPTO_PERP, CRYPTO_OPTIONS
EQUITY, EQUITY_OPTIONS
FOREX
INDEX_FUTURES, COMMODITY_FUTURES, CURRENCY_FUTURES, BOND_FUTURES
```

---

## Точки входа

### Обучение модели
```bash
python train_model_multi_patch.py --config configs/config_train.yaml
```

### Бэктестинг
```bash
python script_backtest.py --config configs/config_backtest.yaml \
  --offline-config configs/offline.yaml --dataset-split val
```

### Live торговля
```bash
export BINANCE_API_KEY=...
python script_live.py --config configs/my_live.yaml [--dry-run]
```

### Evaluation
```bash
python script_eval.py --config configs/my_eval.yaml --profile balanced
python script_eval.py --config configs/my_eval.yaml --all-profiles
```

### Вспомогательные скрипты
- `scripts/doctor.py` - диагностика окружения
- `scripts/quickstart.py` - быстрый старт с пресетами
- `scripts/prepare_training_data.py` - подготовка данных
- `scripts/fetch_binance_filters.py` - обновление метаданных биржи
- `scripts/build_hourly_seasonality.py` - калибровка сезонности

---

## API Endpoints

### REST API (FastAPI)

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/seasonality` | GET | Seasonality multipliers |
| `/signals` | POST | Генерация сигналов |
| `/health` | GET | Статус системы |
| `/backtest/{run_id}` | GET | Отчет бэктеста |
| `/live/{account_id}/positions` | GET | Позиции |
| `/live/{account_id}/order` | POST | Создание ордера |
| `/compliance/status` | GET | Статус compliance |

### WebSocket
- Market data streams
- Position updates
- Risk alerts
- Compliance notifications

---

## Конфигурация

### Основные конфигурационные файлы

| Файл | Назначение |
|------|------------|
| `configs/config_train.yaml` | Конфигурация обучения |
| `configs/config_backtest.yaml` | Конфигурация бэктеста |
| `configs/config_live.yaml` | Конфигурация live торговли |
| `configs/runtime.yaml` | Runtime настройки |
| `configs/execution.yaml` | Настройки исполнения |
| `configs/exchange.yaml` | Спецификации бирж |
| `configs/compliance/` | MiFID II конфиги |
| `configs/dora/` | DORA конфиги |

### Структура YAML конфигурации
```yaml
data:
  symbols: [...]
  timeframe: "4h"

components:
  market_data:
    target: "impl_offline_data:OfflineBarSource"
  executor:
    target: "execution_sim:ExecutionSimulator"
  policy:
    target: "distributional_ppo:DistributionalPPO"

execution:
  profiles:
    conservative: {slippage_bps: 5}
    balanced: {slippage_bps: 3}
    aggressive: {slippage_bps: 1}

slippage:
  model: "l2_parametric"
  bps: 3

risk:
  max_position: 0.1
  max_leverage: 2.0
  daily_loss_limit_bps: 100

state:
  enabled: true
  backend: "json"
```

---

## Структура директорий

```
/
├── adapters/           # Биржевые адаптеры (11 бирж)
│   ├── binance/
│   ├── alpaca/
│   ├── oanda/
│   ├── ib/
│   └── ...
├── services/           # Бизнес-логика (50+ модулей)
│   ├── core/
│   ├── compliance/     # MiFID II
│   ├── ai_act/         # EU AI Act
│   ├── dora_integration/  # DORA
│   └── ...
├── strategies/         # Торговые стратегии
├── wrappers/           # RL environments
├── data/               # Данные
│   ├── candles/
│   ├── processed/
│   ├── futures/
│   └── forex/
├── configs/            # Конфигурации
├── tests/              # 660+ тестовых файлов
├── docs/               # Документация
├── tools/              # Утилиты разработки
└── scripts/            # Операционные скрипты
```

---

## Тестирование

### Категории тестов

| Категория | Количество | Описание |
|-----------|------------|----------|
| Unit tests | ~8,000 | Модульные тесты |
| Integration | ~2,000 | Интеграционные тесты |
| AI Act | ~1,007 | EU AI Act compliance |
| DORA | ~1,015 | DORA compliance |
| PPO bugs | ~20 | Регрессионные тесты PPO |
| Feature parity | ~100 | Offline vs online features |

### Pytest markers
- `asyncio` - асинхронные тесты
- `slow` - медленные тесты
- `integration` - интеграционные тесты
- `requires_gpu` - GPU-only тесты
- `ppo_bugs` - тесты фиксов PPO
- `phase1-9` - фазы forex регрессии

---

## UI компоненты

### Streamlit Dashboard (app.py - 183KB)
- Dashboard обучения модели
- Визуализация бэктестов
- Монитор live торговли
- UI управления рисками
- Dashboard compliance
- Редактор конфигураций
- Data browser

### Визуализация
- Интерактивные equity curves
- Trade heatmaps
- Распределение P&L
- Feature importance
- Risk analytics

---

## Ключевые утилиты

| Модуль | Размер | Описание |
|--------|--------|----------|
| `mediator.py` | 92KB | Central orchestration |
| `clock.py` | - | System time management |
| `no_trade.py` | 65KB | No-trade windows |
| `quantizer.py` | - | Price/qty quantization |
| `slippage.py` | 55KB | Slippage curves |
| `fees.py` | 44KB | Fee structures |
| `labels.py` | - | ML target generation |
| `calibration.py` | - | Parameter calibration |

---

## Документация (/docs/)

### Core документы
- `GETTING_STARTED.md` - Руководство по установке
- `AI_GUIDE.md` - Инструкции для AI ассистента
- `SERVICE_DEPENDENCY_MAP.md` - Зависимости сервисов

### Regulatory roadmaps
- `MIFID_II_COMPLIANCE_ROADMAP.md`
- `EU_AI_ACT_INTEGRATION_PLAN.md`
- `DORA_INTEGRATION_PLAN.md`

### Integration guides
- `FOREX_INTEGRATION_PLAN.md`
- `FUTURES_INTEGRATION_PLAN.md`
- `OPTIONS_INTEGRATION_PLAN.md`
- `L3_MIGRATION_GUIDE.md`

---

## Production статус

**Designed for production deployment**:
- Crypto spot/futures торговля (Binance)
- US equities (Alpaca, Polygon)
- Forex (OANDA)
- CME futures (Interactive Brokers)
- Options (Deribit, Theta Data)
- Distributional PPO с safety features
- Multi-asset risk management
- MiFID II toolkit (designed to align with requirements)
- EU AI Act toolkit (designed to align with requirements)
- DORA toolkit (designed to align with requirements)

> **Note**: Regulatory compliance status reflects implementation of compliance toolkits designed to align with regulatory requirements. This does not constitute certification or guarantee of compliance. Organizations must conduct their own compliance assessment with qualified legal and regulatory advisors.

**Simulator-to-live parity**: <3% backtest deviation
