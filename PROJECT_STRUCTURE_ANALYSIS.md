# TradingBot2 - Подробный анализ структуры проекта

## 1. ОБЗОР ПРОЕКТА

### Что это за проект?
**TradingBot2** — это сложная система торговки на базе машинного обучения (ML), предназначенная для среднечастотной торговли (medium-frequency trading) на биржи криптовалют (в основном Binance). Проект реализует полный цикл от накопления данных, обучения моделей до исполнения торговых сигналов в реальном времени.

### Основные характеристики:
- **Язык**: Python (основной) + Cython + C++ (оптимизация критичных операций)
- **Объем кода**: ~410 Python файлов, ~116,894 строк кода
- **Архитектура**: Слойная (layered), с dependency injection
- **Основные технологии**: 
  - Reinforcement Learning (Stable-Baselines3, Distributional PPO)
  - FastAPI + Streamlit (UI/API)
  - Pandas + NumPy (обработка данных)
  - Pydantic (конфигурация)
  - Cython (оптимизация)

---

## 2. ОСНОВНАЯ АРХИТЕКТУРА

### 2.1 Слойная архитектура (Layered Architecture)

Проект следует четко определенной слойной архитектуре с направлением зависимостей "вверх":

```
┌─────────────────────────────────────────────────────────┐
│  scripts_*.py, train_model_multi_patch.py              │  <- CLI Entry Points
│  (запускаемые скрипты и утилиты)                       │
├─────────────────────────────────────────────────────────┤
│  strategies/                                            │  <- Trading Strategies
│  (торговые алгоритмы)                                   │
├─────────────────────────────────────────────────────────┤
│  service_*.py, service_train.py                        │  <- Services Layer
│  (бизнес-логика и оркестрация)                         │
├─────────────────────────────────────────────────────────┤
│  impl_*.py                                              │  <- Implementation Layer
│  (конкретные реализации инфраструктуры)                │
├─────────────────────────────────────────────────────────┤
│  core_*.py                                              │  <- Core Layer
│  (базовые контракты, модели, утилиты)                  │
└─────────────────────────────────────────────────────────┘
```

**Правила зависимостей**: Каждый слой может зависеть только от слоёв, расположенных ниже.

---

## 3. ГЛАВНЫЕ КОМПОНЕНТЫ И ИХ НАЗНАЧЕНИЕ

### 3.1 Core Layer (базовые сущности)

| Модуль | Назначение |
|--------|-----------|
| `core_models.py` | Доменные модели: Order, Position, Bar, TradeLogRow, EquityPoint и т.д. |
| `core_contracts.py` | Контракты (интерфейсы) для компонентов системы |
| `core_config.py` | Конфигурация и загрузка параметров запуска |
| `core_strategy.py` | Интерфейс Strategy и Decision (deprecated в пользу SignalPolicy) |
| `core_constants.py` | Глобальные константы |
| `core_errors.py` | Определение исключений |
| `core_events.py` | События системы |

### 3.2 Implementation Layer (реализации)

| Модуль | Назначение |
|--------|-----------|
| `impl_bar_executor.py` | Исполнение торговых команд в режиме баров (bar-level execution) |
| `impl_sim_executor.py` | Симуляция исполнения ордеров с моделированием рынка |
| `impl_offline_data.py` | Загрузка исторических данных из CSV/Parquet |
| `impl_slippage.py` | Модель проскальзывания (slippage) при исполнении |
| `impl_latency.py` | Модель задержек в исполнении (latency) |
| `impl_fees.py` | Расчет торговых комиссий (fees) |
| `impl_quantizer.py` | Квантование размеров ордеров |
| `impl_binance_public.py` | Публичный API Binance (данные рынка) |

### 3.3 Service Layer (бизнес-логика и оркестрация)

| Сервис | Назначение |
|--------|-----------|
| `service_train.py` | Обучение ML моделей на исторических данных |
| `service_backtest.py` | Бэктестирование стратегии на исторических данных |
| `service_signal_runner.py` | **ГЛАВНЫЙ** - исполнение стратегии на живых/исторических данных |
| `service_eval.py` | Вычисление метрик производительности |
| `service_calibrate_slippage.py` | Калибровка параметров модели проскальзывания |
| `service_calibrate_tcost.py` | Калибровка торговых стоимостей (trading costs) |
| `service_fetch_exchange_specs.py` | Загрузка спецификаций биржи (Binance) |

### 3.4 Features & Observations (признаки для ML модели)

| Модуль | Назначение |
|--------|-----------|
| `feature_config.py` | Конфигурация блоков признаков (layout) |
| `feature_pipe.py` | Pipeline для расчета признаков онлайн |
| `obs_builder.pyx` (Cython) | Оптимизированный сборщик наблюдений (observations) |
| `lob_state_cython.pyx` | Состояние книги заявок (LOB) на Cython |
| `make_features.py` | Оффлайн расчет признаков из исторических данных |

**Текущая конфигурация признаков (Nov 2025)**:
- N_FEATURES = 56
- EXT_NORM_DIM = 21 (внешние нормализованные признаки)
- MAX_NUM_TOKENS = 1

Структура вектора признаков (56D):
```
[
  Bar (3): price, volume, rel_volume
  Derived (2): ret_1h, vol_proxy
  Indicators (13): MA, RSI, MACD, momentum, ATR, CCI, OBV
  Microstructure (3): OFI, imbalance, micro_dev
  Agent (6): cash_ratio, position_ratio, vol_imbalance, intensity, spread, fill_ratio
  Metadata (5): is_important, time_since_event, risk_off, fear_greed, fear_indicator
  External (21): CVD, GARCH, Yang-Zhang, returns, volatility, etc.
]
```

### 3.5 Training & RL Models

| Модуль | Назначение |
|--------|-----------|
| `distributional_ppo.py` | **ГЛАВНАЯ** ML модель: Distributional PPO (Policy Gradient) |
| `train_model_multi_patch.py` | **ГЛАВНЫЙ ENTRY POINT** для обучения |
| `custom_policy_patch1.py` | Кастомная политика для PPO |
| `reward.pyx` (Cython) | Оптимизированный расчет reward (вознаграждение) |
| `risk_manager.pyx` (Cython) | Управление риском на Cython |

**Алгоритм обучения**:
- Distributional PPO (улучшенный PPO с CVaR - Conditional Value at Risk)
- Поддержка Optuna hyperparameter optimization
- TimeLimit bootstrap для корректной обработки эпизодов
- VecEnv для параллельных сред

### 3.6 Execution & Simulation

| Модуль | Назначение |
|--------|-----------|
| `execution_sim.py` | Главный симулятор исполнения ордеров (micro-level) |
| `execution_algos.py` | Алгоритмы исполнения (VWAP, POV, etc.) |
| `execlob_book.pyx` | Order book LOB на Cython |
| `fast_lob.pyx` | Быстрый обработчик LOB |
| `fast_market.pyx` | Быстрый рыночный симулятор |
| `micro_sim.pyx` | Микроструктурный симулятор на Cython |

**Режимы исполнения**:
- **Bar mode** (баровый): детерминированное исполнение на уровне баров
- **Intrabar mode** (внутри-баровый): моделирование цен внутри бара (Brownian bridge или reference prices)

### 3.7 Risk Management

| Модуль | Назначение |
|--------|-----------|
| `risk.py` | Основные правила риск-менеджмента |
| `risk_guard.py` | Защита от чрезмерного риска (максимальная позиция, drawdown) |
| `dynamic_no_trade_guard.py` | Динамический запрет на торговлю (no-trade zones) |
| `leakguard.py` | Защита от утечек в конфигурации |
| `impl_risk_basic.py` | Базовые риск-метрики |

### 3.8 Data & Configuration

| Модуль | Назначение |
|--------|-----------|
| `ingest_orchestrator.py` | **ГЛАВНЫЙ** для загрузки данных с Binance |
| `binance_public.py` | Публичный REST API Binance (свечи, торговли) |
| `binance_ws.py` | WebSocket подписка на live данные |
| `binance_fee_refresh.py` | Обновление комиссий из Binance |
| `config.py` | Старая конфигурация (deprecated) |
| `ingest_config.py` | Конфигурация процесса ingest |

### 3.9 Monitoring & Logging

| Модуль | Назначение |
|--------|-----------|
| `sim_logging.py` | Логирование торговых операций (TradeLogRow, EquityPoint) |
| `services/monitoring.py` | Мониторинг метрик, alerting |
| `services/metrics.py` | Расчет метрик: Sharpe, Sortino, MDD, PnL и т.д. |
| `services/signal_bus.py` | Шина событий (event bus) |
| `services/signal_csv_writer.py` | Запись сигналов в CSV |

---

## 4. ДИРЕКТОРИЯ СТРУКТУРА

```
/home/user/TradingBot2/
├── core_*.py                  # Core layer (models, contracts, strategy interface)
├── impl_*.py                  # Implementation layer (executors, data sources, fees, etc)
├── service_*.py              # Service layer (main business logic)
├── script_*.py               # CLI entry points (backtest, eval, live, compare runs)
├── train_model_multi_patch.py # ГЛАВНЫЙ entry point для обучения
│
├── /configs/                  # YAML конфигурации
│   ├── config_sim.yaml       # Конфигурация для симуляции
│   ├── config_live.yaml      # Конфигурация для лайв-торговли
│   ├── config_train.yaml     # Конфигурация для обучения
│   ├── config_eval.yaml      # Конфигурация для оценки метрик
│   └── ...
│
├── /data/                     # Данные и датасеты
│   ├── universe/symbols.json # Список криптовалютных пар
│   ├── adv/                  # Average Daily Volume
│   ├── fees/                 # Комиссии
│   ├── latency/              # Сезонность задержек
│   └── ...
│
├── /services/                # Утилиты-сервисы
│   ├── monitoring.py         # Мониторинг метрик
│   ├── rest_budget.py        # Управление лимитами API
│   ├── state_storage.py      # Сохранение состояния
│   └── ...
│
├── /strategies/              # Торговые стратегии
│   ├── base.py              # Базовая стратегия
│   └── momentum.py           # Momentum стратегия
│
├── /scripts/                # Утилиты для подготовки данных
│   ├── build_adv.py
│   ├── build_seasonality.py
│   └── ...
│
├── /features/               # Feature engineering
├── /execution/              # Execution tests
├── /guards/                 # Risk guard tests
├── /monitoring/             # Monitoring tests
├── /tests/                  # Юнит и интеграционные тесты (150+ файлов)
│
├── Feature pipeline
│   ├── feature_pipe.py      # Pipeline для расчета фич
│   ├── feature_config.py    # Конфигурация фич
│   └── transformers.py      # Трансформеры для фич
│
├── Cython modules (для оптимизации)
│   ├── *.pyx               # Cython файлы
│   ├── *.pxd               # Cython declarations
│   ├── *.c / *.so          # Скомпилированные модули
│   ├── obs_builder.pyx     # Сборщик наблюдений
│   ├── lob_state_cython.pyx # LOB state
│   ├── reward.pyx          # Reward calculation
│   ├── risk_manager.pyx    # Risk management
│   └── ...
│
├── /docs/                   # Документация
│   ├── bar_execution.md    # Баровый режим исполнения
│   ├── pipeline.md         # Pipeline архитектура
│   ├── seasonality.md      # Сезонность
│   └── ...
│
└── /artifacts/              # Результаты запусков (модели, логи)
```

---

## 5. ГЛАВНЫЕ ENTRY POINTS (ТОЧКИ ВХОДА)

### 5.1 Обучение модели
```bash
python train_model_multi_patch.py --config configs/config_train.yaml \
  --regime-config configs/market_regimes.json \
  --liquidity-seasonality data/latency/liquidity_latency_seasonality.json
```
**Главный файл**: `train_model_multi_patch.py`
**Использует**: `service_train.py`, `distributional_ppo.py`

### 5.2 Бэктестирование
```bash
python script_backtest.py --config configs/config_sim.yaml
```
**Главный файл**: `script_backtest.py`
**Использует**: `service_backtest.py`, `execution_sim.py`

### 5.3 Лайв-торговля или инфоренс на исторических данных
```bash
python script_live.py --config configs/config_live.yaml
```
**Главный файл**: `script_live.py`
**Использует**: `service_signal_runner.py`

### 5.4 Расчет метрик
```bash
python script_eval.py --config configs/config_eval.yaml --profile balanced
```
**Главный файл**: `script_eval.py`
**Использует**: `service_eval.py`

### 5.5 Загрузка данных с Binance
```bash
python ingest_orchestrator.py --config configs/ingest.yaml \
  --symbols BTCUSDT,ETHUSDT --interval 1m
```
**Главный файл**: `ingest_orchestrator.py`
**Использует**: `binance_public.py`, `binance_ws.py`

### 5.6 Полный цикл: загрузка + обновление + инфоренс
```bash
python scripts/run_full_cycle.py \
  --symbols BTCUSDT,ETHUSDT \
  --interval 1m,5m,15m \
  --prepare-args "--config configs/config_train.yaml" \
  --infer-args "--config configs/config_live.yaml"
```
**Главный файл**: `scripts/run_full_cycle.py`
**Использует**: `ingest_orchestrator.py`, `update_and_infer.py`

---

## 6. ПОТОК ДАННЫХ И ОБРАБОТКИ

### 6.1 ТРЕНИРОВКА (Training Pipeline)

```
1. Binance API (ingest_orchestrator.py)
   ↓ Загрузка свечей (1m, 5m, 15m, 1h, 4h, 1d)
   
2. Оффлайн feature engineering (make_features.py, prepare_and_run.py)
   ↓ Расчет SMA, RSI, MACD, Yang-Zhang volatility, CVD, GARCH...
   
3. Подготовка датасета (build_training_table.py)
   ↓ Формирование train/val/test splits
   
4. ОБУЧЕНИЕ МОДЕЛИ (train_model_multi_patch.py)
   ↓ Distributional PPO на параллельных средах
   ↓ Optuna hyperparameter optimization (CVaR alpha, learning rate, etc)
   
5. Сохранение модели (artifacts/default-run/)
   ↓ weights, config, normalization statistics
```

### 6.2 БЭКТЕСТИРОВАНИЕ (Backtest Pipeline)

```
1. Загрузка конфигурации (config_sim.yaml)
   ↓ Параметры: симуляция, проскальзывание, комиссии, риск
   
2. Загрузка исторических данных (impl_offline_data.py)
   ↓ CSV/Parquet файлы с ценами
   
3. Инициализация среды (service_backtest.py)
   ↓ Market simulator, Risk manager, Bar executor
   
4. Цикл симуляции бар за баром
   ├─ Расчет признаков (feature_pipe.py)
   ├─ Инференс модели (распределение действий)
   ├─ Рассчет reward
   ├─ Исполнение ордеров (bar executor или intrabar simulator)
   └─ Обновление позиции и equity
   
5. Расчет метрик (service_eval.py)
   ↓ Sharpe, Sortino, MDD, Hit-rate, CVaR, PnL
   
6. Запись логов (sim_logging.py)
   ↓ logs/log_trades_*.csv, logs/report_equity_*.csv
```

### 6.3 ЛАЙВ-ТОРГОВЛЯ ИЛИ ИНФОРЕНС (Signal Runner Pipeline)

```
1. Загрузка модели (artifacts/)
   ↓ weights, normalization stats
   
2. Инициализация канала данных
   ├─ Binance WebSocket (для live)
   ├─ CSV/Parquet (для backtest на исторических)
   
3. Цикл обработки в реальном времени (service_signal_runner.py)
   ├─ Получение новой свечи (bar)
   ├─ Расчет признаков онлайн (obs_builder.pyx)
   ├─ Инференс: модель -> action distribution
   ├─ Sample action с температурой (entropy)
   ├─ Преобразование в OrderIntent
   ├─ Risk guards (проверка позиций, drawdown, no-trade zones)
   ├─ Исполнение через Bar executor или REST API
   └─ Логирование сделок
   
4. Сохранение состояния (state_storage.py)
   ↓ Для восстановления после перезагрузки
```

---

## 7. ОСНОВНЫЕ ФУНКЦИИ И ВОЗМОЖНОСТИ

### 7.1 Торговля (Trading)
- Spot торговля на Binance (BTCUSDT и другие USDT пары)
- Futures торговля (поддержка, но основной фокус на spot)
- Баровый режим исполнения (bar-level execution)
- Внутри-баровый режим (intrabar mode с Brownian bridge или reference prices)
- Исполнение алгоритмов: VWAP, POV, Impact-aware

### 7.2 Машинное обучение (ML)
- **Алгоритм**: Distributional PPO (улучшенный PPO)
- **Поддержка CVaR** (Conditional Value at Risk) для управления хвостовым риском
- **Гиперпараметр optimization** через Optuna
- **Параллельные среды** через VecEnv (shared memory)
- **Рекуррентные политики** (LSTM support)
- **TimeLimit bootstrap** для корректного обучения

### 7.3 Риск-менеджмент (Risk Management)
- Лимиты на максимальную позицию
- Максимальный drawdown
- Dynamic no-trade zones (периоды без торговли в зависимости от условий)
- Kill-switch (аварийная остановка)
- Сезонность (приспособление параметров к часам недели)

### 7.4 Моделирование и Калибровка (Modeling)
- **Проскальзывание** (slippage): линейная модель с зависимостью от размера ордера
- **Задержки** (latency): Гауссова с сезонностью, зависит от времени суток
- **Комиссии** (fees): Binance структура с дисконтами BNB
- **Волатильность**: Yang-Zhang, Parkinson, GARCH
- **Liquidity seasonality**: 168-часовые коэффициенты (0 = пн 00:00 UTC)

### 7.5 Признаки (Features) - 56D вектор
- **Bar features** (3): цена, объем, относительный объем
- **Technical indicators** (13): SMA, RSI, MACD, Momentum, ATR, CCI, OBV
- **Derived** (2): 1h return, volatility proxy
- **Microstructure** (3): OFI proxy, quantity imbalance, microstructure deviation
- **Agent state** (6): cash ratio, position ratio, trade intensity, etc.
- **Metadata** (5): event importance, risk-off flag, fear/greed index
- **External** (21): CVD, GARCH, Yang-Zhang, returns, volatility

### 7.6 Мониторинг и Аналитика (Monitoring)
- **Метрики**: Sharpe ratio, Sortino, Maximum Drawdown, PnL, Hit-rate, CVaR
- **Логирование**: Каждая сделка записывается в CSV
- **Equity tracking**: Дневные/часовые отчеты о капитале
- **Signal quality**: Фильтры на качество сигналов
- **REST budget**: Управление API лимитами Binance

---

## 8. КОНФИГУРАЦИЯ ЧЕРЕЗ YAML

### Пример конфига для тренировки (config_train.yaml)

```yaml
mode: train
run_id: my_experiment

# Data
data:
  symbols:
    - BTCUSDT
    - ETHUSDT
  timeframe: 1m  # базовый интервал
  train_start: 2023-01-01
  train_end: 2024-01-01

# Model
agent:
  algorithm: distributional_ppo
  cvar_alpha: 0.95        # CVaR уровень
  cvar_weight: 0.5        # вес CVaR в loss
  learning_rate: 1e-4
  batch_size: 128
  n_steps: 2048

# Costs
costs:
  taker_fee_bps: 7.5      # 0.075%
  half_spread_bps: 1.5
  impact:
    sqrt_coeff: 15.0
    linear_coeff: 2.5

# Execution
execution:
  mode: bar                # или intrabar
  timeframe_ms: 60000      # 1 минута

# Risk
risk:
  max_position: 1.0        # макс 100% портфеля
  max_drawdown: 0.3        # макс 30% просадка
```

---

## 9. ОСНОВНЫЕ ТЕХНОЛОГИЧЕСКИЕ СТЕКИ

### Python Libraries
- **Reinforcement Learning**: Stable-Baselines3
- **Optimization**: Optuna, SciPy
- **Data Processing**: Pandas, NumPy, Polars
- **Time Series**: statsmodels
- **Configuration**: Pydantic, YAML
- **Web/API**: FastAPI, Streamlit
- **Performance**: Cython, Numba
- **Testing**: pytest, unittest

### Cython Modules (для критичных операций)
- `obs_builder.pyx` - сборка наблюдений
- `lob_state_cython.pyx` - состояние книги заявок
- `reward.pyx` - расчет вознаграждений
- `risk_manager.pyx` - управление риском
- `fast_lob.pyx`, `fast_market.pyx` - быстрые симуляторы

### C++ Extensions
- `cpp_microstructure_generator.cpp` - генератор микроструктуры
- Преобразование в Python через Cython

---

## 10. КЛЮ ЧЕВЫЕ КОНЦЕПЦИИ И ПАТТЕРНЫ

### 10.1 Dependency Injection (DI)
- Все компоненты регистрируются в `di_registry.py`
- Конфигурация связывает интерфейсы с реализациями
- Облегчает тестирование и замену компонентов

### 10.2 Decision Pipeline
```
MarketData -> FeaturePipe -> Model Inference -> 
  -> OrderIntent -> RiskGuards -> BarExecutor/APIExecutor
```

### 10.3 Observation/Action Spaces
- **Observation** (56D): вектор признаков состояния рынка и агента
- **Action**: непрерывное действие [-1, 1] -> в торговый ордер (BUY/SELL, размер)

### 10.4 Bar-Mode vs Intrabar-Mode
- **Bar mode**: детерминированное исполнение по цене close/open
- **Intrabar mode**: моделирование цен внутри бара для реалистичности

### 10.5 Seasonality
- 168-часовые коэффициенты (недельный паттерн)
- Применяются к: задержкам, ликвидности, спреду
- Обновляются из исторических данных

---

## 11. ПРОЦЕСС РАЗРАБОТКИ И РАЗВЕРТЫВАНИЯ

### 11.1 Типичный workflow

```
1. ПОДГОТОВКА ДАННЫХ
   python ingest_orchestrator.py --symbols BTCUSDT,ETHUSDT --interval 1m

2. ОБУЧЕНИЕ МОДЕЛИ
   python train_model_multi_patch.py --config configs/config_train.yaml

3. БЭКТЕСТИРОВАНИЕ
   python script_backtest.py --config configs/config_sim.yaml

4. ОЦЕНКА МЕТРИК
   python script_eval.py --config configs/config_eval.yaml

5. ЛАЙВ ДЕПЛОИМЕНТ (с осторожностью!)
   python script_live.py --config configs/config_live.yaml
```

### 11.2 Версионирование моделей
- Каждый запуск сохраняет модель в `/artifacts/default-run/` или `/artifacts/<run_id>/`
- Содержит: weights, config, normalization stats, metrics

---

## 12. ОСНОВНЫЕ ФАЙЛЫ ПРОЕКТА (ТОП 20)

| Файл | Размер | Назначение |
|------|--------|-----------|
| `service_signal_runner.py` | 386KB | Главный сервис исполнения сигналов |
| `distributional_ppo.py` | 454KB | ML модель (Distributional PPO) |
| `execution_sim.py` | 562KB | Микро-симулятор исполнения |
| `train_model_multi_patch.py` | 220KB | Entry point обучения |
| `lob_state_cython.pyx` | 57KB | Cython LOB state |
| `trading_patchnew.py` | 82KB | Market environment |
| `mediator.py` | 63KB | Медиатор компонентов |
| `impl_slippage.py` | 96KB | Модель проскальзывания |
| `service_backtest.py` | 76KB | Сервис бэктеста |
| `impl_latency.py` | 44KB | Модель задержек |
| `feature_pipe.py` | 35KB | Pipeline признаков |
| `custom_policy_patch1.py` | 63KB | Кастомная PPO политика |
| `impl_bar_executor.py` | 66KB | Bar-mode исполнитель |
| `impl_sim_executor.py` | 57KB | Симулятор исполнения |

---

## 13. МЕТРИКИ И ЭФФЕКТИВНОСТЬ

### Отслеживаемые метрики
- **Sharpe Ratio** - риск-скорректированная доходность
- **Sortino Ratio** - как Sharpe, но только для downside volatility
- **Maximum Drawdown** - максимальная просадка
- **Win Rate** - процент прибыльных сделок
- **PnL** - абсолютная прибыль/убыток
- **CVaR** (Conditional Value at Risk) - ожидаемое значение при плохом сценарии
- **Cumulative Return** - общий возврат

### Логирование
- `logs/log_trades_<run_id>.csv` - каждая сделка
- `logs/report_equity_<run_id>.csv` - equity через время
- `artifacts/<run_id>/metrics.json` - итоговые метрики

---

## 14. ВЫВОДЫ

TradingBot2 — это **комплексная, production-ready торговая система**, построенная на:

1. **Solid Architecture** - четко определенная слойная архитектура
2. **Advanced ML** - Distributional PPO с CVaR и Optuna HPO
3. **Realistic Simulation** - детальное моделирование: slippage, latency, fees, seasonality
4. **Performance** - Cython оптимизация критичных путей
5. **Flexibility** - YAML конфигурация, DI контейнер, множество режимов
6. **Monitoring** - детальное логирование всех сделок и метрик

**Главные компоненты для понимания**:
- `train_model_multi_patch.py` - обучение
- `service_signal_runner.py` - исполнение
- `execution_sim.py` - симуляция
- `distributional_ppo.py` - ML модель
- `feature_pipe.py` - признаки

---

**Дата анализа**: 2025-11-11
**Текущая ветка**: claude/update-documentation-011CV2SMDZCkyAyYPfB2ewet
