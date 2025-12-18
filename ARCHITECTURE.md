# Архитектура проекта

> **Last Updated**: 2025-12-16 | **Version**: 6.1 (Design Doc Compliance Complete)

## Cloud-Controlled Execution Architecture (CCEA)

### Ключевой принцип (не обсуждается)

```
Cloud = research/build/monitoring/control plane (lifecycle requests)
Agent = secrets + live loop + risk enforce + order creation/sending
```

**Cloud НИКОГДА:**
- Не хранит broker API keys
- Не генерирует и не передаёт ордера
- Не имеет доступа к trading endpoints бирж
- Не может отправить order-like payload (side/qty/price)

### Архитектурная диаграмма

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLOUD ZONE                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Research  │  │   Builder   │  │   Control   │  │     Monitoring      │ │
│  │     IDE     │  │   Registry  │  │    Plane    │  │     Telemetry       │ │
│  │ (backtest)  │  │  (signed)   │  │ (lifecycle) │  │    (redacted)       │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                                                              │
│  packages/cloud/: control_plane, builder, enterprise, governance, research  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ REQUEST_START_RUN, REQUEST_STOP_RUN
                                    │ REQUEST_UPGRADE_ARTIFACT, REQUEST_UPDATE_CONFIG
                                    │ (NO order-like payloads)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AGENT ZONE                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Local     │  │   Policy    │  │  Live Loop  │  │    Broker           │ │
│  │   Vault     │  │  Firewall   │  │  Risk Mgmt  │  │   Connector         │ │
│  │ (keychain)  │  │ (hard caps) │  │Intent→Order │  │   (orders)          │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                                                              │
│  packages/agent/: daemon, vault, policy, execution, approval, reconciliation│
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Orders (created & sent locally)
                                    ▼
                              ┌───────────┐
                              │ EXCHANGE  │
                              └───────────┘
```

### Зоны модулей

| Zone | Модули | Secrets | Orders | Build |
|------|--------|---------|--------|-------|
| **SHARED** | core_*, impl_*, simulation, features, training | No | No | Both |
| **AGENT** | order_execution, vault, policy_firewall, live_runner | Yes | Yes | Agent only |
| **CLOUD** | control_plane, builder, governance, research_sandbox | No | No | Cloud only |

### CI Guardrails

1. **no-trading-libs-in-cloud**: Cloud build не содержит order_execution модулей
2. **no-order-payloads-in-schema**: JSON schema запрещает side/qty/price в командах
3. **artifact-signature-required**: Артефакт подписан перед публикацией
4. **redaction-enabled**: Telemetry redaction нельзя отключить
5. **import-boundary-check**: Agent imports запрещены в Cloud

### Canonical Stack (v6.1)

```
┌─────────────────────────────────────────────────────────────┐
│                    CANONICAL STACK                          │
├─────────────────────────────────────────────────────────────┤
│  packages/cloud/*     ← Cloud Control Plane (production)    │
│  packages/agent/*     ← Agent Runtime (production)          │
│  packages/shared/*    ← Shared contracts                    │
├─────────────────────────────────────────────────────────────┤
│  ccea/artifact/*      ← Signer + Verifier (integrated)      │
│  ccea/crypto/*        ← Signatures/keys                     │
│  ccea/guardrails/*    ← CI boundary checks                  │
│  ccea/models/*        ← State machines, protocol            │
├─────────────────────────────────────────────────────────────┤
│  ccea/agent/*         ← DEPRECATED (use packages/agent/*)   │
└─────────────────────────────────────────────────────────────┘
```

### Design Doc Compliance (v6.1)

| Feature | Implementation | Status |
|---------|---------------|--------|
| **Crypto signature verification** | `ccea.artifact.verifier.ArtifactVerifier` integrated into `packages/agent/daemon/preflight.py` | ✅ |
| **REQUEST_UPGRADE_ARTIFACT** | `packages/agent/daemon/agentd.py:_handle_upgrade_artifact()` | ✅ |
| **REQUEST_UPDATE_CONFIG** | `packages/agent/daemon/agentd.py:_handle_update_config()` | ✅ |
| **Manifest format** | JSON canonical (`manifest.json`), YAML legacy supported | ✅ |
| **Unsigned artifact rejection** | By design: unsigned = REJECTED (fail-closed) | ✅ |
| **ccea/agent/* deprecation** | DeprecationWarning emitted on import | ✅ |

---

## Слои кода

В репозитории используется слойная структура. Имена файлов и модулей начинаются с префиксов, отражающих их принадлежность к слою.

## Поддерживаемые рынки

| Рынок | Адаптер | Статус |
|-------|---------|--------|
| Crypto (Binance Spot/Futures) | `adapters/binance/` | ✅ Production |
| US Equities (Alpaca) | `adapters/alpaca/` | ✅ Production |
| US Equities Data (Polygon) | `adapters/polygon/` | ✅ Production |
| Forex (OANDA) | `adapters/oanda/` | ✅ Production |
| CME Futures (IB) | `adapters/ib/` | ✅ Production |
| Crypto Options (Deribit) | `adapters/deribit/` | ✅ Beta |

## Слои

- `core_`: базовые сущности, контракты и модели. Не зависит от других слоёв.
- `impl_`: конкретные реализации инфраструктуры и внешних зависимостей. Допустима зависимость только от `core_`.
- `service_`: сервисы, объединяющие реализацию и бизнес-логику. Может зависеть от `core_` и `impl_`.
- `strategies`: торговые стратегии и алгоритмы. Допускаются зависимости от всех предыдущих слоёв (`service_`, `impl_`, `core_`).
- `scripts_`: запускаемые скрипты и утилиты. Могут использовать код из любых слоёв.

Допустимые направления зависимостей идут снизу вверх:

```
core_ → impl_ → service_ → strategies → scripts_
```

Каждый слой может зависеть только от слоёв, расположенных левее.

**Примечание**: Общий план развития проекта планируется оформить в отдельном документе.

## Слой strategies

В пакете `strategies` располагаются торговые алгоритмы. Они могут
использовать код из слоёв `core_`, `impl_` и `service_`, но не должны
зависеть от других стратегий или скриптов.

Стратегии реализуют протокол [`Strategy`](core_strategy.py) и обычно
наследуются от `BaseStrategy`. Сервисы (`service_*`) получают стратегию
через DI-контейнер (`di_registry`) и взаимодействуют с ней только через
интерфейс `Strategy`.

### Decision

Решение стратегии описывается датаклассом `Decision` со следующими
полями:

- `side` -- "BUY" или "SELL";
- `volume_frac` -- целевая величина заявки в долях позиции (диапазон
  `[-1.0; 1.0]`);
- `price_offset_ticks` -- смещение цены в тиках для лимитных заявок
  (для рыночных равно `0`);
- `tif` -- срок действия заявки (`GTC`, `IOC` или `FOK`);
- `client_tag` -- опциональная строка для пометки действий.

### Пример модуля

```python
# strategies/momentum.py
from core_strategy import Strategy, Decision

class MomentumStrategy(Strategy):
    def decide(self, ctx: dict) -> list[Decision]:
        if ctx["ref_price"] > ctx["features"]["ma"]:
            return [Decision(side="BUY", volume_frac=0.1)]
        return []
```

## Конфигурации запусков

Конфигурации описываются в формате YAML. Для загрузки и валидации
используйте функцию `load_config`:

```yaml
# configs/config_sim.yaml
mode: sim
components:
  market_data:
    target: impl_offline_data:OfflineCSVBarSource
    params: {paths: ["data/sample.csv"], timeframe: "1m"}
  executor:
    target: impl_sim_executor:SimExecutor
    params: {symbol: "BTCUSDT"}
data:
  timeframe: "1m"
```

```python
from core_config import load_config

cfg = load_config("configs/config_sim.yaml")
```

The runner loads the symbol universe from ``data/universe/symbols.json`` by default.
Override it via the ``--symbols`` CLI flag or an explicit ``data.symbols`` field.

Отдельные параметры можно переопределить из командной строки. Например,
так временно изменяются проскальзывание и задержка:

```bash
python train_model_multi_patch.py --config configs/config_train.yaml --slippage.bps 5 --latency.mean_ms 50
```

Те же значения можно указать напрямую в YAML:

```yaml
slippage:
  bps: 5
latency:
  mean_ms: 50
```

### Сохранение состояния

Параметры сохранения промежуточного состояния находятся в файле
`configs/state.yaml`. Он задаёт расположение, тип хранилища и периодичность
создания снапшотов:

```yaml
enabled: false
backend: json
dir: state
path: state/state_store.json
snapshot_interval_s: 60
# snapshot_interval_ms: null
flush_on_event: true
backup_keep: 3
lock_path: state/state.lock
last_processed_per_symbol: false
```

* `enabled` -- включить сохранение состояния.
* `backend` -- тип хранилища (`json` или `sqlite`).
* `dir` -- каталог, в котором будут храниться файлы состояния (создаётся автоматически).
* `path` -- путь к основному файлу с состоянием.
* `snapshot_interval_s` / `snapshot_interval_ms` -- периодичность автосохранения.
* `flush_on_event` -- писать состояние при принудительном сбросе.
* `backup_keep` -- количество резервных копий.
* `lock_path` -- путь к файлу блокировки.
* `last_processed_per_symbol` -- сохранять прогресс по каждому инструменту, если доступно.

### Профили исполнения

Конфигурация может содержать несколько профилей исполнения. Каждый профиль
описан параметрами `slippage_bps`, `offset_bps`, `ttl` (в мс) и `tif`, которые
определяют цену выставления заявки и её поведение во времени.

| Профиль       | `slippage_bps` | `offset_bps` | `ttl`, мс | `tif` | Поведение |
|---------------|----------------|--------------|-----------|-------|-----------|
| `conservative`| 5              | 2            | 5000      | GTC   | Пассивные лимитные заявки, ожидание исполнения |
| `balanced`    | 3              | 0            | 2000      | GTC   | Заявки около середины книги, умеренное ожидание |
| `aggressive`  | 1              | -1           | 500       | IOC   | Кроссует спред и быстро отменяет невыполненные заявки |

Пример описания профилей в YAML:

```yaml
profile: balanced
profiles:
  conservative:
    slippage_bps: 5
    offset_bps: 2
    ttl: 5000
    tif: GTC
  balanced:
    slippage_bps: 3
    offset_bps: 0
    ttl: 2000
    tif: GTC
  aggressive:
    slippage_bps: 1
    offset_bps: -1
    ttl: 500
    tif: IOC
```

Скрипт `script_eval.py` позволяет выбрать профиль через `--profile` или
запустить оценку всех профилей флагом `--all-profiles`. В последнем случае
`ServiceEval` формирует отдельные наборы метрик и отчётов для каждого
профиля. Значения `Sharpe`, `PnL` и другие показатели следует анализировать
по каждому профилю отдельно и сравнивать между ними.

### CLI-скрипты

Несколько вспомогательных скриптов принимают путь к YAML через
флаг `--config` и запускают соответствующие сервисы через `from_config`:

```
# Research / backtesting (Cloud zone)
python train_model_multi_patch.py --config configs/config_train.yaml
python script_backtest.py --config configs/config_sim.yaml
python script_eval.py    --config configs/config_eval.yaml --profile vwap
python script_eval.py    --config configs/config_eval.yaml --all-profiles

# Live trading (Agent zone - CCEA architecture)
# Production: Run Agent daemon locally
python -m packages.agent.daemon.agentd --config configs/agent.yaml

# Development/testing only:
python script_live.py    --config configs/config_live.yaml --dry-run
```

**CCEA Note**: Production live trading runs via the Agent daemon (`packages.agent.daemon.agentd`).
`script_live.py` is retained for development/testing only and should not be used in production
CCEA deployments.

### Сравнение запусков

Для агрегирования результатов нескольких прогонов используйте скрипт
`script_compare_runs.py`. Он принимает список путей к файлам
`metrics.json` или каталогам запусков и формирует таблицу ключевых
метрик:

```bash
python script_compare_runs.py run1/ run2/metrics.json --csv summary.csv
```

В консоль выводятся значения `run_id`, `Sharpe`, `Sortino`, `MDD`, `PnL`,
`Hit-rate`, `CVaR` и других найденных показателей. При указании флага
`--csv` таблица сохраняется в указанный файл.

## CLI-точки входа

Все консольные скрипты используют DI-контейнер и не содержат бизнес-логики. Они
описывают аргументы командной строки и делегируют работу соответствующим
сервисам:

- `train_model_multi_patch.py` -- запускает обучение через `ServiceTrain`.
- `script_backtest.py` -- проводит бэктест через `ServiceBacktest`.
- `script_eval.py` -- рассчитывает метрики через `ServiceEval` (поддерживает `--profile` и `--all-profiles`).
- `script_live.py` -- исполняет стратегию на живых данных через `ServiceSignalRunner`.
- `script_calibrate_tcost.py` -- калибрует параметры T-cost через `ServiceCalibrateTCost`.
- `script_calibrate_slippage.py` -- калибрует проскальзывание через `ServiceCalibrateSlippage`.
- `script_compare_runs.py` -- агрегирует метрики нескольких запусков.

## ServiceTrain

`ServiceTrain` подготавливает датасет и запускает обучение модели.  Он
ожидает реализацию протокола `FeaturePipe`.  Для оффлайн-расчёта фич
используется тот же класс `FeaturePipe`, оборачивающий функцию
`apply_offline_features`.

Пример запуска обучения:

```python
from core_config import CommonRunConfig
from service_train import from_config, TrainConfig

cfg_run = CommonRunConfig(...)
trainer = ...
cfg = TrainConfig(input_path="data/train.parquet")
from_config(cfg_run, trainer=trainer, train_cfg=cfg)
```
## Логи и отчёты

Сервисы автоматически пишут журналы сделок и отчёты по эквити через
класс `LogWriter` из модуля [`sim_logging.py`](sim_logging.py). По умолчанию
создаются два файла.

### `logs/log_trades_<runid>.csv`

- Каждая строка соответствует датаклассу
  [`TradeLogRow`](core_models.py).
- Обязательные колонки: `ts`, `run_id`, `symbol`, `side`, `order_type`,
  `price`, `quantity`, `fee`, `fee_asset`, `exec_status`, `liquidity`,
  `client_order_id`, `order_id`, `trade_id`, `pnl`, а также добавленные
  `mark_price` и `equity`.
- Пример строки:

```csv
1700000000000,sim,BTCUSDT,BUY,LIMIT,30000,0.01,0.0005,USDT,FILLED,TAKER,c1,o1,t1,15.0,30010,1005.0,{}
```

### `logs/report_equity_<runid>.csv`

- Строки соответствуют [`EquityPoint`](core_models.py).
- Обязательные колонки: `ts`, `run_id`, `symbol`, `fee_total`,
  `position_qty`, `realized_pnl`, `unrealized_pnl`, `equity`,
  `mark_price`, `drawdown`, `risk_paused_until_ms`, `risk_events_count`,
  `funding_events_count`, `cash`, `meta`.
- Пример строки:

```csv
1700000000000,sim,BTCUSDT,1.2,0.05,100.0,5.0,105.0,30050,-0.02,0,0,0,,{}
```

Логи формируются и обновляются автоматически во всех сервисах
(`service_*`, `execution_sim`) и могут сохраняться как в CSV, так и в
формате Parquet.

## Сезонность ликвидности, спреда и задержек

Сезонность моделирует систематические изменения параметров в течение 168
часов недели (0 = понедельник 00:00 UTC). Генерация коэффициентов
происходит скриптом `scripts/build_hourly_seasonality.py`, который
рассчитывает средние значения по историческим данным и сохраняет их в
`data/latency/liquidity_latency_seasonality.json` (симлинк на
`configs/liquidity_latency_seasonality.json`).

Функции `load_seasonality` и `load_hourly_seasonality` из `utils_time.py`
читают JSON, проверяют контрольную сумму и ограничивают коэффициенты.
`TradingPatch` и `MarketSimulator` масштабируют ликвидность и спред, а
`LatencyImpl` применяет коэффициенты к задержке; опция
`seasonality_interpolate` включает сглаживание между часами.

Для проверки и визуализации доступны скрипты
`scripts/validate_seasonality.py` и `scripts/plot_seasonality.py`. Все
временные метки должны быть в UTC во избежание ошибок индексации по
часам недели.
=======
## Проверка паритета фич

Для валидации соответствия оффлайн и онлайнового расчёта признаков используйте скрипт `check_feature_parity.py`.

Пример запуска:

```
python check_feature_parity.py --data path/to/prices.csv --threshold 1e-6
```

Скрипт вычисляет признаки обоими способами и сообщает о строках, где абсолютное различие превышает `--threshold`. При отсутствии расхождений выводится подтверждение паритета.

## Multi-Asset Adapters

Проект поддерживает торговлю на нескольких рынках через унифицированную систему адаптеров.

### Структура адаптеров

```
adapters/
├── base.py               # Абстрактные базовые классы
├── models.py             # Exchange-agnostic модели данных
├── registry.py           # Фабрика + регистрация адаптеров
├── config.py             # Pydantic конфигурация
├── websocket_base.py     # Production-grade async WebSocket
├── binance/              # Binance (crypto)
│   ├── market_data.py
│   ├── fees.py
│   ├── trading_hours.py
│   └── exchange_info.py
├── alpaca/               # Alpaca (stocks)
│   ├── market_data.py
│   ├── fees.py
│   ├── trading_hours.py
│   ├── exchange_info.py
│   └── order_execution.py
└── polygon/              # Polygon.io (stocks data)
    ├── market_data.py
    ├── trading_hours.py
    └── exchange_info.py
```

### Execution Providers (L2)

Симуляция исполнения унифицирована через `execution_providers.py`:

| Level | Модель | Описание |
|-------|--------|----------|
| L1 | Constant | Фиксированный spread/fee (placeholder) |
| **L2** | Statistical | √participation impact (Almgren-Chriss) |
| L3 | LOB | Full order book simulation (planned) |

### Live Trading (Phase 9)

Unified entry point: `script_live.py`

```bash
# Crypto (Binance)
python script_live.py --config configs/config_live.yaml

# Stocks (Alpaca)
python script_live.py --config configs/config_live_alpaca.yaml --paper
python script_live.py --config configs/config_live_alpaca.yaml --extended-hours
```

Подробная документация: см. [claude.md](claude.md) (Phase 2-4, 9).

## Regulatory Compliance Layer

Проект реализует **alignment/evidence toolkit**, сопоставленный с требованиями **MiFID II** (фазы 1–7 представлены в roadmap). Это инструменты для поддержки оценок и процессов клиентов; они не являются аудитом, сертификацией или юридическим утверждением “compliance”.

### Структура compliance модулей

```
services/compliance/
├── config.py                    # Compliance configuration
├── compliance_clock.py          # Clock sync (RTS 25)
├── lei_manager.py               # LEI management
├── gleif_client.py              # GLEIF API integration
├── algorithm_registry.py        # Algorithm registration
├── transaction_report.py        # Transaction reporting (RTS 22)
├── reporting_pipeline.py        # Reporting pipeline
├── pre_trade_controls.py        # Pre-trade checks (RTS 6)
├── enhanced_kill_switch.py      # Kill switch (Article 12)
├── realtime_monitor.py          # Real-time monitoring (Article 17)
├── otr_monitor.py               # Order-to-Trade ratio
├── audit_trail_writer.py        # Audit trail writing
├── audit_storage.py             # Audit storage (63KB)
├── audit_models.py              # Audit data models
├── retention_policy.py          # 5-7 years retention
├── best_execution.py            # Best execution (Article 27)
├── tca_compliance.py            # TCA compliance
├── venue_analysis.py            # Venue analysis & SOR
├── execution_quality_report.py  # Execution quality reports
├── governance.py                # Governance framework
├── self_assessment.py           # Annual self-assessment
├── bcp.py                       # Business Continuity Plan
├── conformance_testing.py       # Conformance testing
├── test_scenarios.py            # Test scenarios
├── certification.py             # Certification module
└── nca_notification.py          # NCA notification
```

### MiFID II Alignment Tooling (engineering coverage)

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | LEI, Clock Sync, Algorithm Registration | ✅ Tooling implemented |
| Phase 2 | Transaction Reporting (RTS 22) | ✅ Tooling implemented |
| Phase 3 | Kill Switch, Pre-Trade Controls, Monitoring | ✅ Tooling implemented |
| Phase 4 | Record Keeping, Audit Trail (5-7 years) | ✅ Tooling implemented |
| Phase 5 | Best Execution, TCA, Venue Analysis | ✅ Tooling implemented |
| Phase 6 | Governance, Self-Assessment, BCP | ✅ Tooling implemented |
| Phase 7 | Testing, Certification, NCA Notification | ✅ Tooling implemented |

> **Note**: Status "Implemented" means compliance toolkit is ready for client use. This has not been independently audited or certified. Clients must conduct their own compliance assessment.

Детали: [docs/compliance/MIFID_II_COMPLIANCE_ROADMAP.md](docs/compliance/MIFID_II_COMPLIANCE_ROADMAP.md)

## CCEA Protocol

### Разрешённые команды (Allowlist)

| Command | Direction | Description | Approval Required |
|---------|-----------|-------------|-------------------|
| `REQUEST_START_RUN` | Cloud→Agent | Запуск стратегии | Yes (trading_impacting) |
| `REQUEST_STOP_RUN` | Cloud→Agent | Остановка | No (safety) |
| `REQUEST_PAUSE_RUN` | Cloud→Agent | Пауза | No (safety) |
| `REQUEST_UPGRADE_ARTIFACT` | Cloud→Agent | Обновление артефакта | Yes (trading_impacting) |
| `REQUEST_UPDATE_CONFIG` | Cloud→Agent | Обновление config | Yes (если trading_impacting) |
| `REQUEST_ROTATE_AGENT_SESSION` | Cloud→Agent | Ротация сессии | Yes |
| `REQUEST_EXPORT_LOGS` | Cloud→Agent | Экспорт логов | Yes (data_sensitive) |
| `HEARTBEAT` | Agent→Cloud | Статус агента | No |
| `TELEMETRY` | Agent→Cloud | Телеметрия | No |

### Запрещённые payload поля

JSON payload в командах **НЕ ДОЛЖЕН** содержать:
- `side` (BUY/SELL)
- `quantity`
- `price`
- `order_type`
- `target_position`

## State Machines

### Deployment State Machine

```
                    ┌─────────┐
                    │ CREATED │
                    └────┬────┘
                         │ deploy
                         ▼
                    ┌─────────┐
         ┌─────────│ PENDING │
         │         └────┬────┘
         │              │ agent_enrolled
         │              ▼
         │         ┌─────────┐
         │         │ ENROLLED│
         │         └────┬────┘
         │              │ approve_start
         │              ▼
         │         ┌─────────┐
         │    ┌────│ RUNNING │◀───┐
         │    │    └────┬────┘    │
         │    │ pause   │ stop    │ resume
         │    ▼         │         │
         │ ┌──────┐     │     ┌───┴───┐
         │ │PAUSED│─────┴────▶│STOPPED│
         │ └──────┘           └───────┘
         │                        │
         │    revoke              │ terminate
         ▼                        ▼
    ┌─────────┐            ┌───────────┐
    │ REVOKED │            │ TERMINATED│
    └─────────┘            └───────────┘
```

### Run State Machine

```
                    ┌───────────────┐
                    │ INITIALIZING  │
                    └──────┬────────┘
                           │ preflight_ok
                           ▼
                    ┌───────────┐
         ┌─────────│  RUNNING  │◀────────┐
         │         └─────┬─────┘         │
         │ pause         │ kill_switch   │ resume
         ▼               │               │
    ┌─────────┐          │          ┌────┴────┐
    │ PAUSED  │◀─────────┼─────────▶│ HALTED  │
    └────┬────┘          │          └────┬────┘
         │ stop          │ stop          │ acknowledge
         ▼               ▼               ▼
    ┌───────────────────────────────────────┐
    │              STOPPED                   │
    └───────────────────────────────────────┘
```

## Config Layering

### Приоритет конфигурации (highest to lowest)

1. **Local hard caps** - НИКОГДА не может быть переопределено
2. **Local policy firewall** - локальные ограничения
3. **Artifact manifest risk_profile_suggested** - предлагаемый профиль
4. **Cloud config** (blob by digest) - конфиг с сервера
5. **Defaults** - значения по умолчанию

### Trading-Impacting Changes

Следующие изменения **ВСЕГДА** требуют local approve:

| Category | Fields |
|----------|--------|
| Strategy/Model | `artifact_digest`, `model_version` |
| Universe | `symbols`, `asset_classes` |
| Execution | `execution_params`, `slippage_config` |
| Risk | `risk_limits`, `position_limits` |
| Mode | `paper_mode` → `live_mode` |
| Schedule | `trading_schedule`, `blackout_windows` |
| Account | `broker_account`, `adapter_config` |

## Threat Model

| Threat | Mitigation |
|--------|------------|
| RCE in Cloud | Cloud cannot execute orders, no trading libs |
| Key exfiltration | Keys never leave Agent, redaction mandatory |
| Artifact tampering | Digest pinning + signature verification |
| Cloud becomes execution | No order-like payloads in protocol |
| Abuse of cloud jobs | Sandbox + quotas + egress allowlist |
| Man-in-the-middle | mTLS/signed messages |
| Replay attacks | Idempotency keys + timestamps |
| Privilege escalation | RBAC + tenant isolation |

### Safe Defaults

- **Redaction**: ON (cannot be disabled)
- **Local approval**: REQUIRED for trading_impacting
- **RAW telemetry**: OFF (opt-in, enterprise-only)
- **Remote flatten**: DISABLED (enterprise-only by contract)
- **Silent upgrades**: DISABLED for trading-impacting
- **Auto-approve**: DISABLED (local policy only)
