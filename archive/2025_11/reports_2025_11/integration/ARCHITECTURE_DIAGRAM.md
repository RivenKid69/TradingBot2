# AI-Powered Quantitative Research Platform - Архитектурная диаграмма

## Общая архитектура системы

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI ENTRY POINTS                            │
├─────────────────────────────────────────────────────────────────────┤
│  train_model_multi_patch.py  script_backtest.py  script_live.py     │
│  script_eval.py              ingest_orchestrator.py                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                         SERVICE LAYER                               │
├─────────────────────────────────────────────────────────────────────┤
│  service_train.py         ← Обучение моделей (Distributional PPO)  │
│  service_backtest.py      ← Бэктестирование на исторических данных │
│  service_signal_runner.py ← ГЛАВНЫЙ - исполнение стратегии         │
│  service_eval.py          ← Расчет метрик (Sharpe, Sortino, etc)   │
│  service_calibrate_*      ← Калибровка параметров                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                    IMPLEMENTATION LAYER                             │
├──────────────────┬──────────────────┬──────────────────────────────┤
│   DATA LAYER     │ EXECUTION LAYER  │  FEATURE LAYER              │
├──────────────────┼──────────────────┼──────────────────────────────┤
│ impl_offline_data│ impl_bar_executor│ feature_pipe.py            │
│ binance_public.py│ impl_sim_executor│ feature_config.py          │
│ binance_ws.py    │ execution_sim.py  │ obs_builder.pyx (Cython)   │
│                  │ execution_algos.py│ lob_state_cython.pyx       │
└──────────────────┼──────────────────┼──────────────────────────────┘
                   │                  │
┌──────────────────▼──────────────────▼──────────────────────────────┐
│          SPECIALIZED IMPLEMENTATION MODULES                        │
├─────────────────────────────────────────────────────────────────────┤
│  impl_slippage.py    ← Модель проскальзывания                      │
│  impl_latency.py     ← Модель задержек                             │
│  impl_fees.py        ← Расчет комиссий                             │
│  impl_quantizer.py   ← Квантование размеров                        │
│  impl_risk_basic.py  ← Базовые риск-метрики                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                         CORE LAYER                                  │
├─────────────────────────────────────────────────────────────────────┤
│  core_models.py      ← Доменные модели (Order, Position, Bar)      │
│  core_contracts.py   ← Интерфейсы (контракты)                      │
│  core_config.py      ← Загрузка конфигурации                       │
│  core_strategy.py    ← Интерфейс Strategy (deprecated)             │
│  core_events.py      ← События системы                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Основной поток данных: Обучение (Training Flow)

```
┌──────────────────┐
│  Binance API     │  ingest_orchestrator.py
│  (BTCUSDT, etc)  │
└────────┬─────────┘
         │ Загрузка свечей (1m, 5m, 15m, 1h)
         ▼
┌──────────────────────────────────────┐
│  Исторические данные                 │  data/klines/
│  (CSV/Parquet с ценами и объемом)   │
└────────┬─────────────────────────────┘
         │
         │ make_features.py
         │ Расчет признаков:
         │ - SMA (5, 15, 60)
         │ - RSI, MACD, Momentum
         │ - Yang-Zhang volatility
         │ - CVD, GARCH
         ▼
┌──────────────────────────────────────┐
│  Датасет с признаками (56D)          │  data/features/
│  [Bar, Indicators, Microstructure]   │
└────────┬─────────────────────────────┘
         │ build_training_table.py
         │ Формирование splits
         ▼
┌──────────────────────────────────────┐
│  Train / Val / Test splits           │  data/train.parquet
└────────┬─────────────────────────────┘
         │
         │ train_model_multi_patch.py
         │ - Distributional PPO
         │ - Optuna HPO (CVaR alpha, learning_rate)
         │ - Параллельные VecEnv среды
         ▼
┌──────────────────────────────────────┐
│  Обученная модель                    │  artifacts/default-run/
│ - weights.pt                         │  - config.yaml
│ - normalization stats                │  - metrics.json
└──────────────────────────────────────┘
```

---

## Основной поток данных: Инфоренс/Лайв-торговля (Inference Flow)

```
┌──────────────────────┐
│  Binance WebSocket   │
│  (live tick data)    │
└────────┬─────────────┘
         │ OR
         │
         ▼
┌──────────────────────────────────────┐
│  CSV/Parquet (исторические данные)  │
│  (для backtest режима)               │
└────────┬─────────────────────────────┘
         │
         │ service_signal_runner.py
         │ Для каждого нового бара:
         │ 1. Получить свечу (bar)
         │
         ▼
┌──────────────────────────────────────┐
│  obs_builder.pyx (Cython)            │
│  Расчет 56D вектора признаков        │
│  (онлайн, быстро)                    │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Загруженная модель                  │
│  (distributional_ppo.py)             │
│  Инференс: obs -> action distribution│
└────────┬─────────────────────────────┘
         │ action ~ distribution (с temperature)
         │
         ▼
┌──────────────────────────────────────┐
│  OrderIntent преобразование          │
│  (action -> BUY/SELL + размер)       │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Risk Guards                         │
│  - dynamic_no_trade_guard.py         │
│  - risk_guard.py                     │
│  - max_position, max_drawdown        │
└────────┬─────────────────────────────┘
         │ ордер прошел проверку?
         │
      YES│ NO (заблокирован)
         │ │
         ▼ ▼
┌──────────────────────┐  ┌──────────────────┐
│  BarExecutor         │  │  Skip order      │
│  (исполнение)        │  │  (логирование)   │
│  OR                  │  └──────────────────┘
│  REST API Executor   │
│  (для лайв-торговли) │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Логирование                         │
│  logs/log_trades_*.csv               │
│  logs/report_equity_*.csv            │
└──────────────────────────────────────┘
```

---

## Архитектура компонентов: Simulation & Backtesting

```
┌─────────────────────────────────────────────────────────┐
│             service_backtest.py                         │
│  Главный оркестратор симуляции                         │
└─────────────┬───────────────────────────────────────────┘
              │
    ┌─────────┴────────────┬──────────────┬──────────────┐
    │                      │              │              │
    ▼                      ▼              ▼              ▼
┌────────────────┐  ┌────────────────┐ ┌────────────┐ ┌───────────┐
│  Market Data   │  │  Feature Pipe  │ │ Model      │ │ Bar       │
│  Source        │  │ (feature_pipe) │ │ Inference  │ │ Executor  │
│ (csv, parquet) │  │ Расчет 56D obs │ │ (PPO)      │ │ (执行)    │
└────────────────┘  └────────────────┘ └────────────┘ └───────────┘
                            │
                            │ observation
                            │
                    ┌───────▼───────┐
                    │  Reward Calc  │
                    │  (reward.pyx) │
                    └───────┬───────┘
                            │
                    ┌───────▼────────┐
                    │ Risk Manager   │
                    │ (risk_*.py)    │
                    └───────┬────────┘
                            │
                    ┌───────▼──────────┐
                    │ Position Update  │
                    │ Equity Update    │
                    └───────┬──────────┘
                            │
            ┌───────────────┴──────────────┐
            │                              │
            ▼                              ▼
    ┌──────────────────┐          ┌──────────────────┐
    │  Execution Sim   │          │  LOB State       │
    │ (execution_sim)  │          │ (lob_state_*.py) │
    │ Микро-уровень    │          │ Моделирование    │
    └──────────────────┘          │ книги заявок     │
                                  └──────────────────┘
```

---

## Архитектура признаков (Features) - 56D Vector

```
┌────────────────────────────────────────────────────────┐
│              OBSERVATION VECTOR (56D)                  │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │ Bar Features (3)                               │   │
│  │ - price                                        │   │
│  │ - log_volume_norm                              │   │
│  │ - rel_volume                                   │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │ Derived Features (2)                           │   │
│  │ - ret_1h (1h return)                           │   │
│  │ - vol_proxy (volatility proxy)                 │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │ Technical Indicators (13)                      │   │
│  │ - SMA5, SMA20, RSI, MACD, MACD_signal         │   │
│  │ - Momentum, ATR, CCI, OBV                     │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │ Microstructure (3)                             │   │
│  │ - OFI proxy                                    │   │
│  │ - Quantity imbalance                           │   │
│  │ - Microstructure deviation                    │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │ Agent State (6)                                │   │
│  │ - cash_ratio                                  │   │
│  │ - position_ratio                              │   │
│  │ - volume_imbalance                            │   │
│  │ - trade_intensity                             │   │
│  │ - realized_spread                             │   │
│  │ - agent_fill_ratio                            │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │ Metadata (5)                                   │   │
│  │ - is_high_importance                           │   │
│  │ - time_since_event                             │   │
│  │ - risk_off_flag                                │   │
│  │ - fear_greed_value                             │   │
│  │ - fear_greed_indicator                         │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │ External Normalized (21)                       │   │
│  │ - CVD (Cumulative Volume Delta)                │   │
│  │ - GARCH volatility                             │   │
│  │ - Yang-Zhang volatility (24h, 168h, 720h)     │   │
│  │ - Returns (5m, 15m, 1h)                        │   │
│  │ - Other market microstructure                  │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │ Token Metadata (3)                             │   │
│  │ - num_tokens_norm                              │   │
│  │ - token_id_norm                                │   │
│  │ - token_onehot[1] (MAX_NUM_TOKENS=1)          │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
└────────────────────────────────────────────────────────┘

TOTAL: 3 + 2 + 13 + 3 + 6 + 5 + 21 + 3 = 56 features
```

---

## Поток конфигурации (Configuration Flow)

```
┌─────────────────────────────────────────┐
│  configs/config_train.yaml              │
│  configs/config_sim.yaml                │
│  configs/config_live.yaml               │
└──────────────┬──────────────────────────┘
               │
               ▼
        ┌─────────────────────┐
        │  core_config.py     │
        │  load_config()      │
        └──────────┬──────────┘
                   │
      ┌────────────┼────────────┐
      │            │            │
      ▼            ▼            ▼
  ┌────────┐  ┌────────┐  ┌─────────┐
  │ Data   │  │ Costs  │  │ Execution
  │ Config │  │ Config │  │ Config
  └────────┘  └────────┘  └─────────┘
      │            │            │
      └────────────┼────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │  di_registry.py          │
        │  Dependency Injection    │
        │  Container               │
        └──────────┬───────────────┘
                   │
      ┌────────────┼────────────────────────┐
      │            │            │           │
      ▼            ▼            ▼           ▼
  ┌────────┐  ┌────────┐  ┌──────────┐  ┌─────────┐
  │ Market │  │Feature │  │ Executor │  │ Risk    │
  │ Sim    │  │ Pipe   │  │ Instance │  │ Manager │
  └────────┘  └────────┘  └──────────┘  └─────────┘
```

---

## Интеграция Cython модулей (для оптимизации)

```
┌─────────────────────────────────────────────────────────┐
│           PYTHON LAYER (Главные сервисы)               │
├─────────────────────────────────────────────────────────┤
│  service_signal_runner.py                              │
│  service_backtest.py                                   │
│  execution_sim.py                                      │
└──────────────────────┬────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
    ┌────────┐  ┌────────────┐  ┌──────────┐
    │        │  │            │  │          │
    │Cython  │  │  Cython    │  │ Cython   │
    │Modules │  │  Modules   │  │ Modules  │
    │        │  │            │  │          │
    └────────┘  └────────────┘  └──────────┘
        │              │              │
        │ obs_builder  │ reward.pyx   │ risk_mgr
        │ .pyx         │ lob_state    │ .pyx
        │ lob_state    │ .pyx         │
        │ _cython.pyx  │ fast_lob.pyx │
        │ fast_market  │ micro_sim    │
        │ .pyx         │ .pyx         │
        └─────────────┬────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
    ┌────────────┐          ┌─────────────┐
    │ C++ Layer  │          │ Compiled    │
    │            │          │ Extensions  │
    │ *.cpp      │          │ (*.so)      │
    │ *.h        │          │             │
    └────────────┘          └─────────────┘

Критичные пути оптимизированы на Cython для скорости:
- Сборка наблюдений (56D vector)
- Расчет reward
- Управление риском
- Состояние LOB (книги заявок)
```

---

## Summary: Главные компоненты для понимания

```
┌─────────────────────────────────────────────────────────┐
│              ГЛАВНЫЕ КОМПОНЕНТЫ                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. ОБУЧЕНИЕ МОДЕЛИ (⭐ Updated 2025-11-21)            │
│     └─ train_model_multi_patch.py                      │
│        └─ distributional_ppo.py (Distributional PPO)  │
│           ├─ Twin Critics (default enabled)           │
│           ├─ LSTM State Reset (CRITICAL FIX)          │
│           ├─ AdaptiveUPGD Optimizer (default)         │
│           ├─ VGS (Variance Gradient Scaler)           │
│           ├─ PBT (Population-Based Training)          │
│           └─ SA-PPO (State-Adversarial PPO)           │
│                                                         │
│  2. ГЛАВНЫЙ ИСПОЛНИТЕЛЬ СИГНАЛОВ                       │
│     └─ service_signal_runner.py                        │
│        ├─ obs_builder.pyx (Cython: расчет признаков)  │
│        ├─ feature_pipe.py (pipeline признаков)        │
│        ├─ risk_guard.py (проверка рисков)             │
│        └─ impl_bar_executor.py (исполнение)           │
│                                                         │
│  3. СИМУЛЯЦИЯ / БЭКТЕСТ                               │
│     └─ service_backtest.py                            │
│        └─ execution_sim.py (микро-симулятор)          │
│           ├─ impl_slippage.py (проскальзывание)       │
│           ├─ impl_latency.py (задержки)               │
│           ├─ impl_fees.py (комиссии)                  │
│           └─ lob_state_cython.pyx (LOB)               │
│                                                         │
│  4. ДАННЫЕ И КОНФИГУРАЦИЯ                             │
│     ├─ ingest_orchestrator.py (загрузка данных)       │
│     ├─ core_config.py (конфигурация)                  │
│     ├─ feature_config.py (конфигурация признаков)     │
│     └─ configs/*.yaml (YAML конфигурации)             │
│                                                         │
│  5. МОНИТОРИНГ И ЛОГИРОВАНИЕ                          │
│     ├─ sim_logging.py (логирование сделок)            │
│     ├─ services/monitoring.py (мониторинг метрик)     │
│     ├─ services/metrics.py (расчет метрик)            │
│     └─ artifacts/ (сохранение моделей и результатов)  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

