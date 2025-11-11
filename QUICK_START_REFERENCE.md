# TradingBot2 - Краткий справочник (Quick Reference)

## 1. ЧТО ЭТО ТАКОЕ?

**TradingBot2** - это ML-бот для среднечастотной торговли на криптовалютах (Binance).
- Язык: Python + Cython + C++
- Объем: ~410 файлов, ~117K строк кода
- Архитектура: Слойная (Core → Impl → Service → Scripts)

---

## 2. ГЛАВНЫЕ ФАЙЛЫ (МЕНЮ)

### ОБУЧЕНИЕ МОДЕЛИ
```bash
python train_model_multi_patch.py --config configs/config_train.yaml
```
📁 **Главный файл**: `train_model_multi_patch.py`
📁 **ML модель**: `distributional_ppo.py` (Distributional PPO с CVaR)
📁 **Гиперпараметры**: Optuna HPO

### БЭКТЕСТИРОВАНИЕ
```bash
python script_backtest.py --config configs/config_sim.yaml
```
📁 **Главный файл**: `script_backtest.py`
📁 **Сервис**: `service_backtest.py`
📁 **Симулятор**: `execution_sim.py`

### ЛАЙВ-ТОРГОВЛЯ / ИНФОРЕНС
```bash
python script_live.py --config configs/config_live.yaml
```
📁 **Главный файл**: `script_live.py`
📁 **Сервис**: `service_signal_runner.py` (ГЛАВНЫЙ)

### РАСЧЕТ МЕТРИК
```bash
python script_eval.py --config configs/config_eval.yaml
```
📁 **Главный файл**: `script_eval.py`
📁 **Метрики**: Sharpe, Sortino, MDD, CVaR, Hit-rate, PnL

### ЗАГРУЗКА ДАННЫХ
```bash
python ingest_orchestrator.py --symbols BTCUSDT,ETHUSDT --interval 1m
```
📁 **Главный файл**: `ingest_orchestrator.py`
📁 **Источники**: `binance_public.py`, `binance_ws.py`

### ПОЛНЫЙ ЦИКЛ
```bash
python scripts/run_full_cycle.py \
  --symbols BTCUSDT,ETHUSDT \
  --interval 1m,5m,15m \
  --prepare-args "--config configs/config_train.yaml" \
  --infer-args "--config configs/config_live.yaml"
```

---

## 3. АРХИТЕКТУРА СЛОЕВ

```
SCRIPTS ← CLI entry points (train_model_multi_patch.py, script_*.py)
  ↑
SERVICES ← Бизнес-логика (service_train.py, service_signal_runner.py)
  ↑
IMPL ← Реализации (impl_bar_executor.py, impl_slippage.py, impl_fees.py)
  ↑
CORE ← Базовые модели (core_models.py, core_config.py)
```

---

## 4. ОСНОВНЫЕ КОМПОНЕНТЫ

| Компонент | Файл | Назначение |
|-----------|------|-----------|
| **Модель** | `distributional_ppo.py` | Distributional PPO (RL) |
| **Обучение** | `train_model_multi_patch.py` | Entry point для обучения |
| **Исполнение** | `service_signal_runner.py` | Главный исполнитель сигналов |
| **Симуляция** | `execution_sim.py` | Микро-симулятор ордеров |
| **Признаки** | `feature_pipe.py` + `obs_builder.pyx` | 56D вектор признаков |
| **Данные** | `ingest_orchestrator.py` | Загрузка с Binance |
| **Конфигурация** | `core_config.py` | YAML конфигурация |
| **Риск** | `risk_guard.py` | Защита от риска |
| **Логирование** | `sim_logging.py` | Логирование сделок |

---

## 5. ПРИЗНАКИ (56D Vector)

```
BAR (3)                  DERIVED (2)           INDICATORS (13)
├─ price                 ├─ ret_1h              ├─ SMA5, SMA20
├─ volume_norm           └─ vol_proxy           ├─ RSI, MACD
└─ rel_volume                                   ├─ Momentum, ATR
                                                ├─ CCI, OBV
                                                └─ ...

MICROSTRUCTURE (3)       AGENT (6)              METADATA (5)
├─ OFI proxy             ├─ cash_ratio          ├─ is_important
├─ imbalance             ├─ position_ratio      ├─ time_since_event
└─ micro_dev             ├─ trade_intensity     ├─ risk_off
                         ├─ realized_spread     ├─ fear_greed_value
                         └─ fill_ratio          └─ fear_indicator

EXTERNAL (21)            TOKEN (3)
├─ CVD                   ├─ num_tokens_norm
├─ GARCH                 ├─ token_id_norm
├─ Yang-Zhang (24h, 168h, 720h)
├─ Returns (5m, 15m, 1h)
└─ ... (другие признаки)

ИТОГО: 3+2+13+3+6+5+21+3 = 56 features
```

---

## 6. КОНФИГУРАЦИЯ (YAML)

### Основные файлы конфигурации:
- `configs/config_train.yaml` - обучение
- `configs/config_sim.yaml` - симуляция/бэктест
- `configs/config_live.yaml` - лайв-торговля
- `configs/config_eval.yaml` - оценка метрик

### Главные параметры:
```yaml
mode: train  # или sim, live, eval
run_id: my_run

# Данные
data:
  symbols: [BTCUSDT, ETHUSDT]
  timeframe: 1m
  train_start: 2023-01-01
  train_end: 2024-01-01

# Модель
agent:
  algorithm: distributional_ppo
  cvar_alpha: 0.95         # CVaR уровень
  learning_rate: 1e-4
  n_steps: 2048

# Стоимость торговли
costs:
  taker_fee_bps: 7.5       # 0.075%
  half_spread_bps: 1.5
  impact:
    sqrt_coeff: 15.0
    linear_coeff: 2.5

# Исполнение
execution:
  mode: bar                 # или intrabar
  timeframe_ms: 60000       # 1 минута

# Риск
risk:
  max_position: 1.0         # макс позиция
  max_drawdown: 0.3         # макс просадка 30%
```

---

## 7. ДИРЕКТОРИИ

```
/data/                    - Данные и датасеты
  /universe/symbols.json  - Список пар
  /adv/                   - Average Daily Volume
  /fees/                  - Комиссии
  /latency/               - Сезонность задержек

/configs/                 - YAML конфигурации
  /config_train.yaml
  /config_sim.yaml
  /config_live.yaml

/services/                - Утилиты-сервисы
  /monitoring.py          - Мониторинг
  /metrics.py             - Метрики
  /state_storage.py       - Сохранение состояния

/strategies/              - Торговые стратегии
  /base.py
  /momentum.py

/artifacts/               - Результаты запусков
  /default-run/           - Модель, логи, метрики

/tests/                   - ~150 юнит/интеграционных тестов
```

---

## 8. ПОТОК ОБУЧЕНИЯ

```
1. Binance API (ingest_orchestrator.py)
   ↓ Загрузка свечей

2. Feature engineering (make_features.py)
   ↓ Расчет SMA, RSI, Yang-Zhang, CVD, GARCH

3. Подготовка датасета (build_training_table.py)
   ↓ Train/Val/Test splits

4. ОБУЧЕНИЕ (train_model_multi_patch.py)
   ↓ Distributional PPO + Optuna HPO

5. Сохранение (artifacts/)
   ↓ weights, config, stats
```

---

## 9. ПОТОК ИНФОРЕНСА

```
1. Binance WebSocket (live) ИЛИ CSV/Parquet (backtest)
   ↓ Новая свеча

2. obs_builder.pyx (Cython)
   ↓ Расчет 56D вектора признаков

3. Модель (distributional_ppo.py)
   ↓ Инференс → action distribution

4. OrderIntent преобразование
   ↓ action → BUY/SELL + размер

5. Risk Guards (risk_guard.py)
   ↓ Проверка позиций, drawdown

6. BarExecutor или REST API
   ↓ Исполнение ордера

7. sim_logging.py
   ↓ Логирование сделки
```

---

## 10. ВАЖНЫЕ МЕТРИКИ

- **Sharpe Ratio** - риск-скорректированная доходность
- **Sortino Ratio** - downside volatility adjusted return
- **Maximum Drawdown (MDD)** - максимальная просадка
- **Win Rate** - % прибыльных сделок
- **PnL** - прибыль/убыток
- **CVaR** - Conditional Value at Risk (хвостовой риск)
- **Cumulative Return** - общий возврат

---

## 11. CYTHON МОДУЛИ (Оптимизация)

```
obs_builder.pyx         - Сборка 56D наблюдений
lob_state_cython.pyx    - LOB (книга заявок)
reward.pyx              - Расчет reward
risk_manager.pyx        - Управление риском
fast_lob.pyx            - Быстрый LOB
fast_market.pyx         - Быстрый рынок
micro_sim.pyx           - Микро-симулятор
```

---

## 12. СОЧЕТАНИЯ КЛАВИШ / ТИПИЧНЫЕ КОМАНДЫ

### Обучение с кастомными параметрами
```bash
python train_model_multi_patch.py \
  --config configs/config_train.yaml \
  --learning-rate 5e-5 \
  --batch-size 256 \
  --cvar-alpha 0.99
```

### Бэктест с другими параметрами
```bash
python script_backtest.py \
  --config configs/config_sim.yaml \
  --execution-mode bar \
  --portfolio-equity-usd 100000 \
  --costs-taker-fee-bps 5.0
```

### Оценка всех профилей исполнения
```bash
python script_eval.py \
  --config configs/config_eval.yaml \
  --all-profiles
```

### Сравнение нескольких запусков
```bash
python script_compare_runs.py \
  run1/metrics.json \
  run2/metrics.json \
  --csv compare.csv
```

---

## 13. ПРОЦЕСС РАЗРАБОТКИ

1. **Модифицировать конфиг** → `configs/config_*.yaml`
2. **Запустить обучение** → `python train_model_multi_patch.py --config ...`
3. **Бэктестировать** → `python script_backtest.py --config ...`
4. **Проверить метрики** → `python script_eval.py --config ...`
5. **Сравнить результаты** → `python script_compare_runs.py run1 run2`
6. **Деплоить** → `python script_live.py --config configs/config_live.yaml`

---

## 14. ЛОГИРОВАНИЕ И РЕЗУЛЬТАТЫ

После каждого запуска создаются:

```
artifacts/default-run/
├── model.pt              - Веса модели
├── config.yaml           - Конфигурация
├── normalization_stats.json
└── metrics.json          - Метрики (Sharpe, Sortino, MDD, etc)

logs/
├── log_trades_<run_id>.csv
│   └── Каждая строка = сделка
│       (ts, symbol, side, price, quantity, fee, pnl, equity)
└── report_equity_<run_id>.csv
    └── Каждая строка = equity snapshot
        (ts, symbol, equity, position, realized_pnl, drawdown)
```

---

## 15. ГЛАВНЫЕ КОМПОНЕНТЫ ДЛЯ ПОНИМАНИЯ

### ТОП-5 файлов для чтения:
1. `service_signal_runner.py` (386KB) - главный исполнитель
2. `distributional_ppo.py` (454KB) - ML модель
3. `execution_sim.py` (562KB) - симулятор
4. `feature_pipe.py` (35KB) - расчет признаков
5. `core_models.py` (19KB) - доменные модели

### Для быстрого старта:
1. Прочитать `README.md` - обзор
2. Изучить `ARCHITECTURE.md` - архитектура слоев
3. Прочитать `PROJECT_STRUCTURE_ANALYSIS.md` - детальный анализ
4. Посмотреть `ARCHITECTURE_DIAGRAM.md` - диаграммы потоков

---

## 16. ПОЛЕЗНЫЕ ССЫЛКИ (в проекте)

- `docs/bar_execution.md` - баровый режим исполнения
- `docs/pipeline.md` - pipeline архитектура
- `docs/seasonality.md` - сезонность
- `docs/universe.md` - управление символами
- `docs/permissions.md` - права доступа

---

## 17. ТИПИЧНЫЕ ОШИБКИ И РЕШЕНИЯ

### Ошибка: "No module named 'obs_builder'"
**Решение**: Cython модули не скомпилированы. Запустить `python setup.py build_ext --inplace`

### Ошибка: "Config not found"
**Решение**: Убедиться, что файл конфига находится в `configs/` директории

### Ошибка: "Symbol not found in universe"
**Решение**: Обновить `data/universe/symbols.json` через `services/universe.py`

### Ошибка: "Normalization stats not found"
**Решение**: Модель не обучена или сохранена неправильно. Пересохранить после обучения.

---

## 18. ТИПОВОЙ WORKFLOW

```bash
# 1. Загрузить свежие данные
python ingest_orchestrator.py --symbols BTCUSDT,ETHUSDT --interval 1m

# 2. Обучить модель
python train_model_multi_patch.py --config configs/config_train.yaml

# 3. Бэктестировать
python script_backtest.py --config configs/config_sim.yaml

# 4. Проверить метрики
python script_eval.py --config configs/config_eval.yaml --all-profiles

# 5. Сравнить с предыдущими запусками
python script_compare_runs.py artifacts/run1 artifacts/run2 --csv summary.csv

# 6. Если результаты хорошие - перейти в лайв
python script_live.py --config configs/config_live.yaml
```

---

**Дата**: 2025-11-11
**Статус**: Активно разрабатывается
**Основной язык**: Python + Cython + C++

