# Claude Documentation - TradingBot2

## О проекте

**TradingBot2** — высокочастотный торговый бот для криптовалют (преимущественно Binance spot), использующий reinforcement learning (PPO) для принятия торговых решений. Проект написан на Python с критичными к производительности модулями на Cython/C++ и включает в себя полноценный симулятор исполнения, модели проскальзывания, задержек и микроструктуры рынка.

### Основные характеристики

- **Язык**: Python 3.12 + Cython + C++
- **RL Framework**: Stable-Baselines3 (Distributional PPO)
- **Биржа**: Binance (Spot/Futures)
- **Режимы работы**: Бэктест, Live trading, Обучение моделей
- **Архитектура**: Слоистая (layered) с dependency injection

## Архитектура проекта

Проект использует **строгую слоистую архитектуру** с префиксами имён файлов:

```
core_ → impl_ → service_ → strategies → script_
```

### Слои (Layers)

#### 1. `core_*` — Базовый слой
Содержит базовые сущности, контракты (protocols), модели и константы. **Не зависит** от других слоёв.

**Ключевые файлы:**
- `core_config.py` — конфигурационные модели (CommonRunConfig, etc.)
- `core_models.py` — TradeLogRow, EquityPoint, Decision
- `core_strategy.py` — Protocol для торговых стратегий
- `core_contracts.py` — Интерфейсы/контракты
- `core_events.py` — События системы
- `core_errors.py` — Кастомные исключения
- `core_constants.py` — Константы (сопоставление Cython и Python)

#### 2. `impl_*` — Слой реализации
Конкретные имплементации инфраструктуры и внешних зависимостей. Зависит **только от `core_`**.

**Ключевые файлы:**
- `impl_sim_executor.py` — Симулятор исполнения заявок
- `impl_fees.py` — Расчёт комиссий (с поддержкой BNB discount)
- `impl_slippage.py` — Модели проскальзывания
- `impl_latency.py` — Модели задержек (сезонные, волатильные)
- `impl_quantizer.py` — Квантование цен/объёмов по биржевым фильтрам
- `impl_offline_data.py` — Чтение исторических данных
- `impl_binance_public.py` — Публичные API Binance
- `impl_bar_executor.py` — Баровый исполнитель
- `impl_risk_basic.py` — Базовый риск-менеджмент

#### 3. `service_*` — Слой сервисов
Объединяет бизнес-логику. Может зависеть от `core_` и `impl_`.

**Ключевые файлы:**
- `service_backtest.py` — Сервис бэктестинга
- `service_train.py` — Сервис обучения моделей
- `service_eval.py` — Оценка моделей с разными профилями исполнения
- `service_signal_runner.py` — Запуск live trading
- `service_calibrate_tcost.py` / `service_calibrate_slippage.py` — Калибровка
- `service_fetch_exchange_specs.py` — Загрузка биржевых спецификаций
- `services/monitoring.py` — Мониторинг и метрики
- `services/rest_budget.py` — REST API rate limiting
- `services/ops_kill_switch.py` — Operational kill switch
- `services/state_storage.py` — Персистентность состояния
- `services/signal_bus.py` — Шина сигналов
- `services/universe.py` — Управление универсом символов

#### 4. `strategies/` — Торговые стратегии
Реализации алгоритмов принятия решений. Могут зависеть от всех предыдущих слоёв.

**Файлы:**
- `strategies/base.py` — Базовый класс Strategy
- `strategies/momentum.py` — Пример стратегии на моментуме

#### 5. `script_*` — CLI точки входа
Запускаемые скрипты. Используют DI контейнер и **не содержат бизнес-логику**.

**Основные скрипты:**
- `script_backtest.py` → ServiceBacktest
- `script_live.py` → ServiceSignalRunner
- `script_eval.py` → ServiceEval (поддержка `--all-profiles`)
- `script_compare_runs.py` → Сравнение метрик
- `script_calibrate_tcost.py`, `script_calibrate_slippage.py` → Калибровка
- `script_fetch_exchange_specs.py` → Загрузка exchange specs
- `train_model_multi_patch.py` → Обучение моделей (основной скрипт)

### Dependency Injection (DI)

Проект использует DI через модуль `di_registry.py`. Компоненты регистрируются и резолвятся динамически из YAML конфигураций.

Пример:
```yaml
components:
  market_data:
    target: impl_offline_data:OfflineCSVBarSource
    params: {paths: ["data/sample.csv"], timeframe: "1m"}
```

## Основные компоненты

### 1. Симулятор исполнения (ExecutionSimulator)

Находится в `execution_sim.py`. Включает:
- Симуляцию LOB (limit order book) через Cython модули
- Микроструктурный генератор (`micro_sim.pyx`, `cpp_microstructure_generator.cpp`)
- Модели проскальзывания (linear, sqrt, калиброванные)
- Учёт комиссий (maker/taker, BNB discount)
- TTL (time-to-live) для лимитных заявок
- TIF: GTC, IOC, FOK
- Алгоритмические исполнители: TWAP, POV, VWAP

### 2. Distributional PPO (`distributional_ppo.py`)

Кастомизированный PPO с:
- Distributional value head (quantile regression)
- Expected Value (EV) reserve sampling для стабилизации обучения
- EV batching с приоритизацией редких событий
- Поддержка sampling mask для no-trade окон
- Отключённый PopArt (ранее использовался, теперь удалён)

### 3. Features Pipeline

- `feature_pipe.py` — Онлайн расчёт признаков
- `features_pipeline.py` — Оффлайн препроцессинг
- `feature_config.py` — Конфигурация фич
- Поддержка проверки паритета через `check_feature_parity.py`

### 4. Риск-менеджмент

- `risk_guard.py` — Гварды на позицию/PnL/дроудаун
- `risk_manager.pyx` — Cython модуль для быстрой проверки
- `dynamic_no_trade_guard.py` — Динамическое блокирование торговли
- `ops_kill_switch` — Операционный kill switch

### 5. No-Trade окна

- `no_trade.py`, `no_trade_config.py` — Управление запрещёнными окнами
- Поддержка funding windows, daily UTC windows, custom intervals
- Утилита: `no-trade-mask` (CLI)

### 6. Latency & Seasonality

- **Latency**: `latency.py`, `impl_latency.py` — моделирование задержек (mean, std, volatility)
- **Seasonality**: `utils_time.py`, `configs/liquidity_latency_seasonality.json`
  - 168 коэффициентов (24ч × 7 дней недели) для ликвидности, спреда, задержек
  - Валидация: `scripts/validate_seasonality.py`
  - Построение: `scripts/build_hourly_seasonality.py`

### 7. Fees & Quantization

- `fees.py`, `impl_fees.py` — Комиссии (BNB discount, maker/taker)
- `quantizer.py`, `impl_quantizer.py` — Квантование по биржевым фильтрам
- Auto-refresh фильтров: `scripts/fetch_binance_filters.py`
- Auto-refresh fees: `scripts/refresh_fees.py`

### 8. Data Degradation

- `data_validation.py` — Моделирование пропусков, задержек, stale data
- Конфиг: `data_degradation` (stale_prob, drop_prob, dropout_prob, max_delay_ms)

### 9. Logging & Metrics

- `sim_logging.py` — Запись логов трейдов и equity
  - `logs/log_trades_<runid>.csv` (TradeLogRow)
  - `logs/report_equity_<runid>.csv` (EquityPoint)
- `services/monitoring.py` — Метрики (Sharpe, Sortino, MDD, CVaR, etc.)
- Агрегация через `aggregate_exec_logs.py`

## Конфигурации (configs/)

### Основные конфиги

- **config_sim.yaml** — Симуляция (бэктест)
- **config_train.yaml** — Обучение модели
- **config_live.yaml** — Live trading
- **config_eval.yaml** — Оценка модели
- **config_template.yaml** — Шаблон конфигурации

### Модульные конфиги (включаются через YAML anchors)

- **execution.yaml** — Параметры исполнения
- **fees.yaml** — Комиссии и округление
- **slippage.yaml** — Модели проскальзывания
- **risk.yaml** — Риск-менеджмент
- **no_trade.yaml** — No-trade окна
- **quantizer.yaml** — Квантование
- **timing.yaml** — Timing профили
- **runtime.yaml** / **runtime_trade.yaml** — Runtime параметры
- **state.yaml** — Персистентность состояния
- **monitoring.yaml** — Мониторинг
- **ops.yaml** / **ops.json** — Operational kill switch
- **rest_budget.yaml** — REST API rate limiting
- **offline.yaml** — Оффлайн datasets, сплиты

### Сезонность и режимы

- **liquidity_latency_seasonality.json** — 168 коэффициентов для ликвидности/латентности
- **market_regimes.json** — Рыночные режимы

## CLI Примеры

### Бэктест
```bash
python script_backtest.py --config configs/config_sim.yaml
```

### Обучение
```bash
python train_model_multi_patch.py \
  --config configs/config_train.yaml \
  --regime-config configs/market_regimes.json \
  --liquidity-seasonality configs/liquidity_latency_seasonality.json
```

### Live trading
```bash
python script_live.py --config configs/config_live.yaml
```

### Оценка модели (все профили)
```bash
python script_eval.py --config configs/config_eval.yaml --all-profiles
```

### Сравнение запусков
```bash
python script_compare_runs.py run1/ run2/ run3/ --csv compare.csv
```

### Обновление символов
```bash
python -m services.universe --output data/universe/symbols.json --liquidity-threshold 1e6
```

### Обновление биржевых фильтров
```bash
python scripts/fetch_binance_filters.py --universe --out data/binance_filters.json
```

### Обновление комиссий
```bash
python scripts/refresh_fees.py
```

### Валидация сезонности
```bash
python scripts/validate_seasonality.py \
  --historical path/to/trades.csv \
  --multipliers data/latency/liquidity_latency_seasonality.json
```

### Проверка реалистичности симуляции
```bash
python scripts/sim_reality_check.py \
  --trades sim_trades.parquet \
  --historical-trades hist_trades.parquet \
  --equity sim_equity.parquet \
  --benchmark bench_equity.parquet \
  --kpi-thresholds benchmarks/sim_kpi_thresholds.json
```

## Cython/C++ модули

### Критичные к производительности компоненты

- **fast_lob.pyx / fast_lob.cpp** — Быстрая LOB
- **lob_state_cython.pyx** — Состояние LOB
- **micro_sim.pyx** — Микроструктурная симуляция
- **marketmarket_simulator_wrapper.pyx** — Обёртка C++ симулятора
- **obs_builder.pyx** — Построение наблюдений
- **reward.pyx** — Расчёт reward
- **risk_manager.pyx** — Риск-менеджмент
- **coreworkspace.pyx** — Рабочее пространство
- **execlob_book.pyx** — LOB для исполнения

### C++ компоненты

- **MarketSimulator.cpp/.h** — Основной симулятор рынка
- **OrderBook.cpp/.h** — Стакан заявок
- **cpp_microstructure_generator.cpp/.h** — Генератор микроструктуры

## Важные паттерны и концепции

### 1. Execution Profiles

Поддерживаются различные профили исполнения (conservative, balanced, aggressive) с разными:
- `slippage_bps` — проскальзывание
- `offset_bps` — смещение лимитной цены
- `ttl` — время жизни заявки (мс)
- `tif` — Time In Force (GTC/IOC/FOK)

### 2. Bar Execution Mode

Режим `execution.mode: bar` позволяет работать с агрегированными баровыми данными вместо tick-by-tick.

Параметры:
- `bar_price: close` — цена исполнения (open/high/low/close)
- `min_rebalance_step: 0.05` — минимальный шаг ребалансировки

Сигналы должны следовать формату [spot signal envelope](docs/bar_execution.md).

### 3. Intrabar Price Models

- **bridge** — Brownian bridge sampling (legacy)
- **reference** — Использование внешнего M1 reference feed для детерминированных fills

Настраивается через `execution.intrabar_price_model` в YAML.

### 4. Large Order Execution

Заявки с notional > `notional_threshold` разбиваются алгоритмически:
- **TWAP** — Time-Weighted Average Price
- **POV** — Percentage of Volume
- **VWAP** — Volume-Weighted Average Price

Параметры POV:
```yaml
pov:
  participation: 0.2       # 20% от наблюдаемого объёма
  child_interval_s: 1      # Интервал между дочерними заявками
  min_child_notional: 1000 # Минимальный размер дочерней заявки
```

### 5. Expected Value (EV) Reserve

Механизм в Distributional PPO для стабилизации обучения:
- Резервирует часть батча для редких/высоко-ценных событий
- Приоритизация через квантили EV
- Настраивается через `ev_reserve_*` параметры в конфиге

### 6. No-Trade Masks

Блокируют торговлю в определённые периоды:
- Funding windows (±5 минут от 00:00/08:00/16:00 UTC)
- Custom intervals (milliseconds)
- Daily UTC windows

Применяется через:
- Конфиг: `no_trade` секция
- Утилита: `no-trade-mask --mode drop/weight`

### 7. Data Degradation

Моделирование реальных проблем с данными:
- `stale_prob` — вероятность повторить предыдущий бар
- `drop_prob` — вероятность пропустить бар
- `dropout_prob` — вероятность задержки
- `max_delay_ms` — максимальная задержка

### 8. Kill Switch

Два типа:
- **Metric kill switch** — останавливает торговлю при плохих метриках
- **Operational kill switch** — останавливает при операционных проблемах

Восстановление:
```bash
python scripts/reset_kill_switch.py
```

## Data Pipeline

### 1. Ingestion (Загрузка данных)

```bash
python scripts/run_full_cycle.py \
  --symbols BTCUSDT,ETHUSDT \
  --interval 1m,5m,15m \
  --start 2024-01-01 --end 2024-12-31
```

Модули:
- `ingest_orchestrator.py` — Оркестратор загрузки
- `ingest_klines.py` — Загрузка свечей
- `ingest_funding_mark.py` — Funding rates и mark prices
- `binance_public.py` — Публичное API Binance

### 2. Preprocessing

```bash
python prepare_and_run.py --config configs/feature_prepare.yaml
```

Модули:
- `prepare_events.py` — Подготовка событий
- `build_adv.py`, `build_adv_base.py` — ADV (Average Daily Volume)
- `make_features.py` — Создание признаков
- `make_prices_from_klines.py` — Извлечение цен из свечей

### 3. Training

```bash
python train_model_multi_patch.py --config configs/config_train.yaml
```

Создаёт модель (PPO policy) в формате Stable-Baselines3.

### 4. Evaluation

```bash
python script_eval.py --config configs/config_eval.yaml
```

Генерирует метрики в `metrics.json`.

### 5. Live Trading

```bash
python script_live.py --config configs/config_live.yaml
```

## Тестирование

Проект содержит **обширный набор тестов** (pytest):

### Категории тестов

- **Execution** — `test_execution_*.py` (детерминизм, профили, правила)
- **Fees** — `test_fees_*.py` (округление, BNB discount)
- **Latency** — `test_latency_*.py` (сезонность, волатильность)
- **Risk** — `test_risk_*.py` (exposure limits, kill switch)
- **Service** — `test_service_*.py` (бэктест, eval, signal runner)
- **No-trade** — `test_no_trade_*.py` (маски, окна)
- **Distributional PPO** — `test_distributional_ppo_*.py` (CVaR, outliers, EV reserve)

### Запуск тестов

```bash
pytest tests/                          # Все тесты
pytest tests/test_execution_sim*.py    # Конкретная категория
pytest -k "test_fees"                  # По ключевому слову
```

## Документация проекта (docs/)

- **moving_average.md** — Скользящие средние
- **pipeline.md** — Конвейер принятия решений
- **bar_execution.md** — Баровый режим исполнения
- **permissions.md** — Роли, владение файлами, права доступа
- **universe.md** — Управление универсом символов
- **large_orders.md** — Алгоритмическое исполнение крупных заявок
- **seasonality.md** — Сезонные множители
- **seasonality_quickstart.md** — Быстрый старт с сезонностью
- **seasonality_QA.md** — QA процесс для сезонности
- **no_trade.md** — No-trade окна
- **parallel.md** — Параллельные окружения и случайность
- **data_degradation.md** — Деградация данных

## Важные переменные окружения

- `TB_FAIL_ON_STALE_FILTERS=1` — Фейлить при устаревших фильтрах
- `BINANCE_PUBLIC_FEES_DISABLE_AUTO=1` — Отключить автообновление комиссий
- `BINANCE_API_KEY`, `BINANCE_API_SECRET` — API ключи Binance
- `BINANCE_FEE_SNAPSHOT_CSV` — Путь к CSV с комиссиями
- `SYMS`, `LOOP`, `SLEEP_MIN` — Для `update_and_infer.py`

## Git & Collaboration

### Branching

Работа ведётся на feature branches с префиксом `claude/`:
```bash
git checkout -b claude/feature-name-SESSION_ID
```

### Commit Messages

Следуйте стилю из `git log`:
- Краткое описание (1-2 предложения)
- Фокус на "why", а не "what"
- Примеры: "Add BNB fee settlement mode", "Fix EV batch prioritization"

### Pull Requests

Создание PR через `gh` CLI:
```bash
gh pr create --title "Feature: ..." --body "## Summary\n- ...\n\n## Test plan\n- ..."
```

## Debugging & Troubleshooting

### 1. Проверка паритета фич
```bash
python check_feature_parity.py --data prices.csv --threshold 1e-6
```

### 2. Проверка PnL
```bash
pytest tests/test_pnl_report_check.py
```

### 3. Проверка drift
```bash
python check_drift.py --baseline baseline.csv --current current.csv
```

### 4. Валидация кривой проскальзывания
```bash
python compare_slippage_curve.py hist.csv sim.csv --tolerance 5
```

### 5. Логи деградации
Ищите в выводе:
- `OfflineCSVBarSource degradation: ...`
- `BinanceWS degradation: ...`
- `LatencyQueue degradation: ...`

## Performance Tips

1. **Используйте Cython модули** — все критичные компоненты уже оптимизированы
2. **Параллельные окружения** — `shared_memory_vec_env.py` для multi-env training
3. **Кэширование REST** — настройте `rest_budget.cache` в `offline.yaml`
4. **Checkpointing** — используйте `checkpoint_path` для длительных запусков
5. **Offline режим** — используйте `--dry-run` для проверки без сетевых запросов

## Частые задачи

### Добавить новый символ
1. Обновите `data/universe/symbols.json`
2. Перезагрузите фильтры: `python scripts/fetch_binance_filters.py --universe`
3. Загрузите исторические данные через `ingest_orchestrator.py`

### Изменить параметры риска
Отредактируйте `configs/risk.yaml` или передайте через CLI:
```bash
python script_backtest.py --config config.yaml --risk.max-position 100
```

### Добавить новую фичу
1. Реализуйте в `features/`
2. Зарегистрируйте в `features/registry.py`
3. Добавьте в `feature_config.py`
4. Проверьте паритет: `check_feature_parity.py`

### Калибровать slippage
```bash
python script_calibrate_slippage.py --config configs/slippage_calibrate.yaml
```

### Создать новую стратегию
1. Создайте файл в `strategies/`
2. Унаследуйте от `BaseStrategy`
3. Реализуйте метод `decide(ctx) -> list[Decision]`
4. Зарегистрируйте в DI (если нужно)

## Ключевые метрики

При анализе результатов обращайте внимание на:

- **Sharpe Ratio** — скорректированная на риск доходность
- **Sortino Ratio** — учитывает только downside volatility
- **MDD (Max Drawdown)** — максимальная просадка
- **CVaR (Conditional Value at Risk)** — средний убыток в худших 5% случаев
- **Hit Rate** — процент прибыльных сделок
- **PnL Total** — суммарная прибыль/убыток
- **Turnover** — оборот
- **Avg Latency** — средняя задержка исполнения

## Production Checklist

Перед запуском в продакшн:

- [ ] Обновлены фильтры (`fetch_binance_filters.py`)
- [ ] Обновлены комиссии (`refresh_fees.py`)
- [ ] Обновлены exchange specs (`script_fetch_exchange_specs.py`)
- [ ] Валидирована сезонность (`validate_seasonality.py`)
- [ ] Настроен kill switch (`ops.yaml`)
- [ ] Настроен мониторинг (`monitoring.yaml`)
- [ ] Настроено сохранение состояния (`state.yaml`)
- [ ] Проверены risk limits (`risk.yaml`)
- [ ] Проверены no-trade окна (`no_trade.yaml`)
- [ ] Проведён sim reality check (`sim_reality_check.py`)
- [ ] Все тесты проходят (`pytest tests/`)

## Полезные ссылки

- **Issues**: `/home/user/TradingBot2/issues/`
- **Benchmarks**: `/home/user/TradingBot2/benchmarks/`
- **Artifacts**: `/home/user/TradingBot2/artifacts/`
- **Data**: `/home/user/TradingBot2/data/`
- **Logs**: `/home/user/TradingBot2/logs/` (автоматически создаются)
- **CHANGELOG**: `CHANGELOG.md`
- **ARCHITECTURE**: `ARCHITECTURE.md`
- **CONTRIBUTING**: `CONTRIBUTING.md`

## Заключение

TradingBot2 — это сложная система с множеством компонентов. При работе с проектом:

1. **Следуйте слоистой архитектуре** — не нарушайте зависимости между слоями
2. **Используйте DI** — регистрируйте компоненты через `di_registry`
3. **Пишите тесты** — особенно для критичной логики
4. **Проверяйте паритет** — онлайн и оффлайн фичи должны совпадать
5. **Мониторьте метрики** — используйте sim_reality_check
6. **Обновляйте конфиги** — фильтры, комиссии, сезонность устаревают

Удачи в разработке!
