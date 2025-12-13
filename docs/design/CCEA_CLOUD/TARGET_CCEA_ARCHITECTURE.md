# Target CCEA Architecture

> **Version**: 1.0.0
> **Date**: 2025-12-13
> **Status**: APPROVED

## 1. Архитектурная диаграмма

```
                              ┌─────────────────────────────────────────────────────────────┐
                              │                         CLOUD ZONE                           │
                              │  ┌─────────────────────────────────────────────────────────┐ │
                              │  │                    Control Plane                         │ │
                              │  │   ┌──────────┐  ┌──────────┐  ┌────────────────────┐   │ │
                              │  │   │  API GW  │  │ Scheduler│  │  Command Dispatcher │   │ │
                              │  │   └──────────┘  └──────────┘  └────────────────────┘   │ │
                              │  └─────────────────────────────────────────────────────────┘ │
                              │  ┌─────────────────────────────────────────────────────────┐ │
                              │  │                     Services                             │ │
                              │  │   ┌───────────┐ ┌───────────┐ ┌──────────────────────┐ │ │
                              │  │   │  Builder  │ │  Registry │ │   Telemetry Ingest   │ │ │
                              │  │   └───────────┘ └───────────┘ └──────────────────────┘ │ │
                              │  │   ┌───────────┐ ┌───────────┐ ┌──────────────────────┐ │ │
                              │  │   │ Research  │ │  Backtest │ │      Monitoring      │ │ │
                              │  │   │    IDE    │ │    Sim    │ │     Dashboards       │ │ │
                              │  │   └───────────┘ └───────────┘ └──────────────────────┘ │ │
                              │  └─────────────────────────────────────────────────────────┘ │
                              │  ┌─────────────────────────────────────────────────────────┐ │
                              │  │                     Governance                           │ │
                              │  │   ┌───────────┐ ┌───────────┐ ┌──────────────────────┐ │ │
                              │  │   │   RBAC    │ │ Retention │ │      Residency       │ │ │
                              │  │   └───────────┘ └───────────┘ └──────────────────────┘ │ │
                              │  └─────────────────────────────────────────────────────────┘ │
                              └───────────────────────────┬─────────────────────────────────┘
                                                          │
                                                          │ Lifecycle Requests Only:
                                                          │ REQUEST_START_RUN
                                                          │ REQUEST_STOP_RUN
                                                          │ REQUEST_PAUSE_RUN
                                                          │ REQUEST_UPGRADE_ARTIFACT
                                                          │ REQUEST_UPDATE_CONFIG
                                                          │
                                                          │ NO: orders, intents, targets
                                                          │
                                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                          AGENT ZONE                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              Command Handler + Approval UI                                │  │
│  │   ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────────────────────┐│  │
│  │   │ Command Receiver│  │  Approval Queue  │  │            Local Approve UI             ││  │
│  │   │  (poll-based)   │  │ (trading_impact) │  │   [APPROVE] [REJECT] [VIEW DIFF]       ││  │
│  │   └─────────────────┘  └──────────────────┘  └─────────────────────────────────────────┘│  │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                    Execution Core                                         │  │
│  │   ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────────────────────┐│  │
│  │   │   Live Loop     │  │  Strategy Runner │  │           Risk Manager                  ││  │
│  │   │ (Intent→Order)  │  │   (Artifact)     │  │   (Policy Firewall + Hard Caps)        ││  │
│  │   └─────────────────┘  └──────────────────┘  └─────────────────────────────────────────┘│  │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                    Security Core                                          │  │
│  │   ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────────────────────┐│  │
│  │   │   Local Vault   │  │  Kill Switch     │  │         Reconciliation                  ││  │
│  │   │ (broker keys)   │  │ (emergency halt) │  │   (positions/orders sync)              ││  │
│  │   └─────────────────┘  └──────────────────┘  └─────────────────────────────────────────┘│  │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                  Broker Integration                                       │  │
│  │   ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────────────────────┐│  │
│  │   │ Order Execution │  │ Position Manager │  │           Broker Connector              ││  │
│  │   │  (local only)   │  │  (local state)   │  │   (Binance/Alpaca/OANDA/IB/...)        ││  │
│  │   └─────────────────┘  └──────────────────┘  └─────────────────────────────────────────┘│  │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                          │
                                                          │ Orders (created locally)
                                                          ▼
                                                   ┌──────────────┐
                                                   │   EXCHANGES  │
                                                   │ Binance/IB/  │
                                                   │ Alpaca/OANDA │
                                                   └──────────────┘

┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         SHARED ZONE                                             │
│  ┌────────────────────────┐  ┌────────────────────────┐  ┌──────────────────────────────────┐  │
│  │     Core Models        │  │    Implementations     │  │          Simulation              │  │
│  │  core_models.py        │  │  impl_slippage.py      │  │  execution_sim.py                │  │
│  │  core_config.py        │  │  impl_fees.py          │  │  impl_sim_executor.py            │  │
│  │  core_contracts.py     │  │  impl_latency.py       │  │  impl_bar_executor.py            │  │
│  │  core_events.py        │  │  impl_pricing.py       │  │                                  │  │
│  └────────────────────────┘  └────────────────────────┘  └──────────────────────────────────┘  │
│  ┌────────────────────────┐  ┌────────────────────────┐  ┌──────────────────────────────────┐  │
│  │   Market Data (public) │  │    Feature Pipeline    │  │           Training               │  │
│  │  adapters/*/market_    │  │  features_pipeline.py  │  │  distributional_ppo.py           │  │
│  │  data.py               │  │  transformers.py       │  │  train_model_multi_patch.py      │  │
│  │                        │  │  feature_config.py     │  │                                  │  │
│  └────────────────────────┘  └────────────────────────┘  └──────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 2. Полная таблица модулей по зонам

### 2.1 SHARED Zone (безопасно для Cloud и Agent)

| Модуль | Путь | Описание | Размер |
|--------|------|----------|--------|
| **Core Models** ||||
| `core_models.py` | `/` | Доменные модели: OrderIntent, Order, ExecReport, Position | 516 строк |
| `core_config.py` | `/` | Pydantic конфигурации, DI компоненты | 1,391 строк |
| `core_contracts.py` | `/` | Протоколы: FeaturePipe, SignalPolicy, Strategy | 141 строк |
| `core_events.py` | `/` | События рынка и исполнения | 119 строк |
| `core_errors.py` | `/` | Типы исключений | 109 строк |
| `core_constants.py` | `/` | Глобальные константы | 39 строк |
| `core_strategy.py` | `/` | Базовый интерфейс Strategy | 84 строк |
| `core_futures.py` | `/` | Модели и расчеты для фьючерсов | 1,605 строк |
| `core_options.py` | `/` | Греки и ценообразование опционов | 912 строк |
| `core_conformal.py` | `/` | Conformal prediction framework | 659 строк |
| **Implementations** ||||
| `impl_slippage.py` | `/` | Market impact, проскальзывание | 2,395 строк |
| `impl_fees.py` | `/` | Расчет комиссий | 1,691 строк |
| `impl_latency.py` | `/` | Симуляция сетевой задержки | 1,117 строк |
| `impl_pricing.py` | `/` | Black-Scholes, биномиальные модели | 1,053 строк |
| `impl_quantizer.py` | `/` | Квантование цены/размера | 885 строк |
| `impl_conformal.py` | `/` | Conformal prediction реализация | 1,208 строк |
| `impl_iv_calculation.py` | `/` | Implied volatility | 855 строк |
| `impl_greeks.py` | `/` | Греки Black-Scholes | 956 строк |
| `impl_greeks_vectorized.py` | `/` | Векторизованные греки | 1,117 строк |
| `impl_futures_funding.py` | `/` | Funding rate расчеты | 1,289 строк |
| `impl_futures_liquidation.py` | `/` | Ликвидация позиций (sim) | 986 строк |
| `impl_futures_margin.py` | `/` | Маржа и SPAN margin | 1,068 строк |
| **Simulation** ||||
| `execution_sim.py` | `/` | High-fidelity LOB симулятор | 13,712 строк |
| `impl_sim_executor.py` | `/` | Симулятор исполнения | 1,424 строк |
| `impl_bar_executor.py` | `/` | Исполнение на барах (OHLC) | 1,810 строк |
| **Data & Features** ||||
| `features_pipeline.py` | `/` | Pipeline обработки фичей | 1,220 строк |
| `transformers.py` | `/` | Трансформеры фичей | 1,468 строк |
| `feature_config.py` | `/` | Конфигурация фичей | - |
| `data_loader_multi_asset.py` | `/` | Multi-asset data loader | 1,344 строк |
| `impl_offline_data.py` | `/` | Загрузка исторических данных | - |
| **Training** ||||
| `distributional_ppo.py` | `/` | Distributional PPO agent | 13,232 строк |
| `train_model_multi_patch.py` | `/` | Training orchestrator | 5,579 строк |
| **Market Data Adapters (public only)** ||||
| `adapters/binance/market_data.py` | `/adapters/binance/` | Binance public market data | - |
| `adapters/binance/fees.py` | `/adapters/binance/` | Binance fee structure | - |
| `adapters/alpaca/market_data.py` | `/adapters/alpaca/` | Alpaca public market data | - |
| `adapters/oanda/market_data.py` | `/adapters/oanda/` | OANDA public market data | - |
| `adapters/polygon/market_data.py` | `/adapters/polygon/` | Polygon market data | - |
| `adapters/ib/market_data.py` | `/adapters/ib/` | IB market data | - |
| `adapters/yahoo/market_data.py` | `/adapters/yahoo/` | Yahoo Finance data | - |
| **LOB Simulation** ||||
| `lob/data_structures.py` | `/lob/` | LOB structures | 41KB |
| `lob/calibration_pipeline.py` | `/lob/` | Калибровка LOB | 41KB |
| `lob/market_impact.py` | `/lob/` | Market impact модели | 35KB |

### 2.2 AGENT Zone (только локальное исполнение)

| Модуль | Путь | Описание | Зона |
|--------|------|----------|------|
| **Order Execution** ||||
| `adapters/alpaca/order_execution.py` | `/adapters/alpaca/` | Alpaca order submission | AGENT-ONLY |
| `adapters/alpaca/options_execution.py` | `/adapters/alpaca/` | Alpaca options execution | AGENT-ONLY |
| `adapters/oanda/order_execution.py` | `/adapters/oanda/` | OANDA order submission | AGENT-ONLY |
| `adapters/ib/order_execution.py` | `/adapters/ib/` | IB order submission | AGENT-ONLY |
| `adapters/ib/options_combo.py` | `/adapters/ib/` | IB options combos | AGENT-ONLY |
| **Execution Providers (live mode)** ||||
| `execution_providers.py` | `/` | Multi-asset execution (live) | AGENT-ONLY |
| `execution_providers_futures_l3.py` | `/` | L3 futures execution | AGENT-ONLY |
| `execution_providers_cme_l3.py` | `/` | CME L3 execution | AGENT-ONLY |
| `execution_providers_l3.py` | `/` | L3 spot execution | AGENT-ONLY |
| **Live Runtime** ||||
| `service_signal_runner.py` | `/` | Live trading runtime | AGENT-ONLY |
| **Security** ||||
| `CredentialVault` (planned) | `/services/security/` | Local vault for secrets | AGENT-ONLY |
| **Risk Control** ||||
| `risk_guard.py` | `/` | Risk guardrails, hard caps | AGENT-ONLY |
| `impl_circuit_breaker.py` | `/` | Circuit breakers | AGENT-ONLY |
| **Reconciliation** ||||
| Position/Order reconciliation | (planned) | Sync with broker | AGENT-ONLY |

### 2.3 CLOUD Zone (только исследование и управление)

| Модуль | Путь | Описание | Зона |
|--------|------|----------|------|
| **UI/IDE** ||||
| `app.py` | `/` | FastAPI + Streamlit UI | CLOUD-ONLY |
| **Services (orchestration)** ||||
| `service_backtest.py` | `/` | Backtest orchestration | CLOUD-ONLY |
| `service_train.py` | `/` | Training orchestration | CLOUD-ONLY |
| `service_eval.py` | `/` | Evaluation metrics | CLOUD-ONLY |
| **Control Plane (planned)** ||||
| Lifecycle management | (planned) | Deployments, commands | CLOUD-ONLY |
| Artifact Builder | (planned) | Build pipeline | CLOUD-ONLY |
| Registry | (planned) | Artifact storage | CLOUD-ONLY |
| **Telemetry** ||||
| Monitoring dashboards | (planned) | Health, alerts | CLOUD-ONLY |
| Telemetry ingestion | (planned) | Event collection | CLOUD-ONLY |
| **Governance** ||||
| RBAC | (planned) | Access control | CLOUD-ONLY |
| Data retention | (planned) | Retention policies | CLOUD-ONLY |
| Data residency | (planned) | EU/regional storage | CLOUD-ONLY |

## 3. Запрещённые зависимости для Cloud

### 3.1 Запрещённые импорты

Cloud build **НЕ ДОЛЖЕН** импортировать:

```python
# ЗАПРЕЩЕНО в Cloud build
from adapters.alpaca.order_execution import *
from adapters.alpaca.options_execution import *
from adapters.oanda.order_execution import *
from adapters.ib.order_execution import *
from adapters.ib.options_combo import *

# ЗАПРЕЩЕНО: любые модули с trading/execution
from execution_providers import *  # в live mode
from service_signal_runner import *  # в live mode
```

### 3.2 Запрещённые библиотеки

```
# ЗАПРЕЩЕНО в Cloud dependencies
- Любые private trading SDK
- Любые broker submission libraries
- Любые order management libraries
```

## 4. Граница периметра

### 4.1 Информационные потоки

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLOUD                                       │
│                                                                      │
│   [Artifact Build] ──▶ [Registry] ──▶ [Control Plane]               │
│                                              │                       │
│                                              │ Lifecycle Requests    │
│                                              │ (no secrets/orders)   │
│                                              ▼                       │
└──────────────────────────────────────────────┬──────────────────────┘
                                               │
                    ╔══════════════════════════╧══════════════════════╗
                    ║              SECURITY BOUNDARY                   ║
                    ║  - No broker keys cross this boundary            ║
                    ║  - No order payloads cross this boundary         ║
                    ║  - Only lifecycle commands allowed               ║
                    ╚══════════════════════════╤══════════════════════╝
                                               │
┌──────────────────────────────────────────────┴──────────────────────┐
│                          AGENT                                       │
│                                                                      │
│   [Command Receiver] ──▶ [Approval] ──▶ [Live Loop]                 │
│                                              │                       │
│                              ┌───────────────┼───────────────┐       │
│                              │               │               │       │
│                              ▼               ▼               ▼       │
│                         [Vault]      [Risk Guard]    [Execution]     │
│                              │               │               │       │
│                              └───────────────┴───────────────┘       │
│                                              │                       │
│                                              │ Orders (local only)   │
│                                              ▼                       │
│                                        [Broker API]                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Что пересекает границу

| Direction | Allowed | NOT Allowed |
|-----------|---------|-------------|
| Cloud → Agent | `REQUEST_START_RUN`, `REQUEST_STOP_RUN`, `REQUEST_PAUSE_RUN`, `REQUEST_UPGRADE_ARTIFACT`, `REQUEST_UPDATE_CONFIG`, `REQUEST_ROTATE_AGENT_SESSION`, `REQUEST_EXPORT_LOGS` | Orders, Intents, Signals, Target positions, Secrets |
| Agent → Cloud | `HEARTBEAT`, `TELEMETRY` (redacted), `COMMAND_ACK`, `COMMAND_RESULT`, `APPROVAL_RECORD` | Raw secrets, Unredacted logs, Broker responses |

## 5. Mapping фаз плана на Rollout

| Plan Phase | Design Doc Sections | Status |
|------------|---------------------|--------|
| **Phase 0** | 0-5 (Architecture, Decisions) | IN PROGRESS |
| Phase 1 | Skeleton E2E | Planned |
| Phase 2 | Cloud/Agent/Shared Split | Planned |
| Phase 3 | Strategy API | Planned |
| Phase 4 | Artifact Builder | Planned |
| Phase 5 | Agent Daemon | Planned |
| Phase 6 | Control Plane | Planned |
| Phase 7 | Protocol | Planned |
| Phase 8 | Telemetry | Planned |
| Phase 9 | Enterprise | Planned |
| Phase 10 | Cloud Jobs | Planned |
| Phase 11 | Documentation | Planned |

## 6. Валидация архитектуры

### 6.1 CI Checks

| Check | Command | Description |
|-------|---------|-------------|
| Import boundary | `python -m ccea.guardrails.import_check` | Проверка импортов |
| Dependency audit | `python -m ccea.guardrails.dep_check` | Проверка зависимостей |
| Schema validation | `python -m ccea.guardrails.schema_check` | Проверка JSON схем |

### 6.2 Runtime Checks

| Check | Location | Description |
|-------|----------|-------------|
| Signature verification | Agent | Проверка подписи артефакта |
| Schema version | Agent | Совместимость версий |
| Hard cap enforcement | Agent | Локальные лимиты |
| Redaction | Agent | Фильтрация секретов |

---

**Document Control:**
- Author: CCEA Architecture Team
- Last Updated: 2025-12-13
