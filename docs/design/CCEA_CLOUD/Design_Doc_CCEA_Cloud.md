# Design Doc: CCEA Cloud Architecture

> **Version**: 1.0.0
> **Date**: 2025-12-13
> **Status**: APPROVED
> **SHA256**: 5474499a7347b9e2d532670a7cfd9deed0c6fa6ea58da069179fccab23836908

## 0. Executive Summary

CCEA (Cloud-Controlled Execution Architecture) определяет строгое разделение между Cloud (исследование/мониторинг/управление жизненным циклом) и Agent (исполнение/секреты/риск-контроль). Ключевой принцип: **Cloud никогда не хранит ключи, не имеет кода/доступа к торговым API от имени пользователя и не передаёт live-торговые инструкции/ордера/targets.**

### 0.1 Ключевой принцип (не обсуждается)

```
Cloud = research/build/monitoring/control plane (lifecycle requests)
Agent = secrets + live loop + risk enforce + order creation/sending
```

### 0.2 Гарантии безопасности

1. Cloud **НИКОГДА** не хранит broker API keys
2. Cloud **НИКОГДА** не генерирует и не передаёт ордера
3. Cloud **НИКОГДА** не имеет доступа к trading endpoints бирж
4. Все торговые операции происходят **ТОЛЬКО** в Agent
5. Agent работает **ЛОКАЛЬНО** у пользователя (или в его VPC)

## 1. Архитектурное разделение

### 1.1 Зона SHARED (безопасна для обоих рантаймов)

Код, который может безопасно использоваться и в Cloud, и в Agent:

| Категория | Модули | Описание |
|-----------|--------|----------|
| **Core Models** | `core_models.py`, `core_config.py`, `core_contracts.py`, `core_events.py`, `core_errors.py`, `core_constants.py` | Базовые доменные модели, контракты, типы |
| **Core Domain** | `core_futures.py`, `core_options.py`, `core_conformal.py`, `core_strategy.py` | Доменная логика без исполнения |
| **Implementation** | `impl_slippage.py`, `impl_fees.py`, `impl_latency.py`, `impl_pricing.py`, `impl_quantizer.py` | Математические модели, pricing |
| **Simulation** | `impl_sim_executor.py`, `impl_bar_executor.py`, `execution_sim.py` | Симуляция (без реальных API) |
| **Features** | `features_pipeline.py`, `transformers.py`, `feature_config.py` | Feature engineering |
| **Training** | `distributional_ppo.py`, `train_model_multi_patch.py` | ML training |
| **Data** | `data_loader_*.py`, `impl_offline_data.py` | Data loading (public data) |
| **Market Data Adapters** | `adapters/*/market_data.py`, `adapters/*/fees.py` | Public market data only |

### 1.2 Зона AGENT (только локальное исполнение)

Код, который должен выполняться **ТОЛЬКО** в Agent:

| Категория | Модули | Описание |
|-----------|--------|----------|
| **Order Execution** | `adapters/*/order_execution.py`, `adapters/*/options_execution.py` | Создание и отправка ордеров |
| **Trading Clients** | Private trading clients, broker connectors | Подключение к broker API |
| **Execution Providers** | `execution_providers.py` (live mode) | Live execution |
| **Local Vault** | `CredentialVault`, keychain integration | Хранение секретов |
| **Policy Firewall** | `risk_guard.py`, hard caps | Локальный контроль рисков |
| **Kill Switch** | Local kill switch implementation | Экстренная остановка |
| **Reconciliation** | Position/order reconciliation | Сверка позиций |
| **Live Runner** | `service_signal_runner.py` (live mode) | Live trading loop |

### 1.3 Зона CLOUD (только исследование и управление)

Код, который работает **ТОЛЬКО** в Cloud:

| Категория | Модули | Описание |
|-----------|--------|----------|
| **UI/IDE** | `app.py`, web interface | Пользовательский интерфейс |
| **Backtest/Sim** | `service_backtest.py`, `service_train.py` | Orchestration |
| **Builder/Registry** | Artifact builder, image registry | Сборка артефактов |
| **Control Plane** | Lifecycle management, deployments | Управление lifecycle |
| **Telemetry** | Monitoring dashboards, alerts | Мониторинг |
| **Governance** | RBAC, retention, residency | Управление доступом |

## 2. Запрещённые операции для Cloud

### 2.1 Запрещённые зависимости/импорты

Cloud build **НЕ ДОЛЖЕН** содержать:

```python
# ЗАПРЕЩЕНО в Cloud
- adapters/*/order_execution.py
- adapters/*/options_execution.py
- Любые private trading clients
- Любые broker submission modules
- Модули с доступом к trading endpoints
```

### 2.2 Запрещённые типы сообщений

Протокол Cloud→Agent **НЕ ДОЛЖЕН** содержать:

| Запрещено | Причина |
|-----------|---------|
| `PLACE_ORDER` | Прямое создание ордера |
| `SUBMIT_ORDER` | Отправка ордера |
| `EXECUTE_SIGNAL` | Исполнение сигнала |
| `SET_TARGET_POSITION` | Установка целевой позиции |
| `SEND_INTENT` | Передача торгового намерения |

### 2.3 Запрещённые поля в payload

JSON payload в командах **НЕ ДОЛЖЕН** содержать:

```json
// ЗАПРЕЩЕНО
{
  "side": "BUY|SELL",
  "quantity": <number>,
  "price": <number>,
  "order_type": "MARKET|LIMIT",
  "symbol": "<trading_symbol>",
  "target_position": <number>
}
```

### 2.4 Secret Hygiene

- Cloud **НИКОГДА** не получает broker API keys
- Cloud **НИКОГДА** не получает master key от vault
- Support dumps **НИКОГДА** не содержат секреты
- Телеметрия **ВСЕГДА** проходит redaction

## 3. Продуктовые режимы

### 3.1 Retail Research SaaS (EU-friendly)

```
┌─────────────────────────────────────────────────────┐
│                     CLOUD                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐  │
│  │ Research│  │   Sim   │  │     Monitoring      │  │
│  │   IDE   │  │ Backtest│  │    Dashboards       │  │
│  └─────────┘  └─────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────┘
                        │
                        │ Lifecycle Requests
                        │ (no orders/secrets)
                        ▼
┌─────────────────────────────────────────────────────┐
│              OPTIONAL: BYO Agent                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐  │
│  │  Local  │  │  Risk   │  │     Live Loop       │  │
│  │  Vault  │  │  Guard  │  │   (user-owned)      │  │
│  └─────────┘  └─────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

**Характеристики:**
- Cloud: research, simulation, monitoring
- Agent: опционален, только для live trading
- Secrets: хранятся только у пользователя
- EU-friendly: telemetry в EU region

### 3.2 Retail Live via Local Agent

```
┌─────────────────────────────────────────────────────┐
│                     CLOUD                            │
│  ┌─────────────────────────────────────────────┐    │
│  │         Control Plane + Monitoring          │    │
│  │   - Lifecycle requests (start/stop/pause)   │    │
│  │   - Artifact registry (signed builds)       │    │
│  │   - Telemetry dashboards                    │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
                        │
                        │ REQUEST_START_RUN
                        │ REQUEST_STOP_RUN
                        │ REQUEST_UPGRADE_ARTIFACT
                        ▼
┌─────────────────────────────────────────────────────┐
│              LOCAL AGENT (user machine)              │
│  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐  │
│  │  Vault  │  │ Policy  │  │     Live Loop       │  │
│  │(secrets)│  │Firewall │  │ Intent→Risk→Order   │  │
│  └─────────┘  └─────────┘  └─────────────────────┘  │
│                     │                                │
│                     ▼                                │
│  ┌─────────────────────────────────────────────┐    │
│  │           Broker Connector                   │    │
│  │      (orders created & sent locally)         │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
                   [ EXCHANGE ]
```

**Характеристики:**
- Auto-execution локально
- Cloud: наблюдаемость и lifecycle requests
- Agent: полный контроль над исполнением
- Hard caps: enforce локально, cloud не может поднять

### 3.3 Enterprise Engine (on-prem/VPC/self-hosted)

```
┌─────────────────────────────────────────────────────┐
│            CUSTOMER INFRASTRUCTURE                   │
│  ┌──────────────────────────────────────────────┐   │
│  │               Self-hosted Cloud               │   │
│  │   ┌─────────┐  ┌─────────┐  ┌──────────┐    │   │
│  │   │ Control │  │Registry │  │Monitoring│    │   │
│  │   │  Plane  │  │ (local) │  │(on-prem) │    │   │
│  │   └─────────┘  └─────────┘  └──────────┘    │   │
│  └──────────────────────────────────────────────┘   │
│                        │                             │
│                        ▼                             │
│  ┌──────────────────────────────────────────────┐   │
│  │                Agent Cluster                  │   │
│  │   ┌─────────┐  ┌─────────┐  ┌──────────┐    │   │
│  │   │ HSM/KMS │  │  Risk   │  │Execution │    │   │
│  │   │ (cust.) │  │ Engine  │  │  Nodes   │    │   │
│  │   └─────────┘  └─────────┘  └──────────┘    │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

**Характеристики:**
- Всё в инфраструктуре клиента
- Vendor pack для compliance
- Air-gapped режим поддерживается
- CMK (customer-managed keys) для данных

## 4. Протокол Cloud ↔ Agent

### 4.1 Разрешённые команды (allowlist)

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
| `COMMAND_ACK` | Agent→Cloud | Подтверждение | No |
| `COMMAND_RESULT` | Agent→Cloud | Результат | No |

### 4.2 Аутентификация и подпись

```
┌─────────────────────────────────────────────────────┐
│                  Message Flow                        │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Cloud                              Agent            │
│    │                                  │              │
│    │  ┌─────────────────────────┐    │              │
│    │  │  Signed Command         │    │              │
│    │  │  - idempotency_key      │    │              │
│    │  │  - payload_ref (digest) │    │              │
│    │  │  - signature (server)   │    │              │
│    │──│  - timestamp            │───▶│              │
│    │  └─────────────────────────┘    │              │
│    │                                  │              │
│    │  ┌─────────────────────────┐    │              │
│    │  │  Signed Response        │    │              │
│    │  │  - result               │    │              │
│    │◀─│  │  - signature (agent)    │───│              │
│    │  │  - evidence_hash        │    │              │
│    │  └─────────────────────────┘    │              │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**Варианты аутентификации:**
- **Option A (Enterprise)**: mTLS
- **Option B (Default)**: Signed JWT на device key

## 5. State Machines

### 5.1 Deployment State Machine

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
         │    │         │         │
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

### 5.2 Run State Machine

```
                    ┌───────────┐
                    │ INITIALIZING │
                    └──────┬──────┘
                           │ preflight_ok
                           ▼
                    ┌───────────┐
         ┌─────────│  RUNNING  │◀────────┐
         │         └─────┬─────┘         │
         │               │               │
         │ pause         │ kill_switch   │ resume
         ▼               │               │
    ┌─────────┐          │          ┌────┴────┐
    │ PAUSED  │◀─────────┼─────────▶│ HALTED  │
    └─────────┘          │          └─────────┘
         │               │               │
         │ stop          │ stop          │ acknowledge
         ▼               ▼               ▼
    ┌───────────────────────────────────────┐
    │              STOPPED                   │
    └───────────────────────────────────────┘
```

## 6. Config Layering

### 6.1 Приоритет конфигурации

```
Priority (highest to lowest):
1. Local hard caps (НИКОГДА не может быть переопределено)
2. Local policy firewall
3. Artifact manifest risk_profile_suggested
4. Cloud config (blob by digest)
5. Defaults
```

### 6.2 Trading-Impacting Changes

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

## 7. Threat Model

### 7.1 Threat Vectors

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

### 7.2 Safe Defaults

- Redaction: **ON** (cannot be disabled)
- Local approval: **REQUIRED** for trading_impacting
- RAW telemetry: **OFF** (opt-in, enterprise-only)
- Remote flatten: **DISABLED** (enterprise-only by contract)
- Silent upgrades: **DISABLED** for trading-impacting
- Auto-approve: **DISABLED** (local policy only)

## 8. Telemetry & Privacy

### 8.1 Telemetry Levels

| Level | Description | Default |
|-------|-------------|---------|
| `AGGREGATED` | PnL, win rate, drawdown (no trade details) | **Yes (retail)** |
| `DETAILED_NON_SENSITIVE` | Trade counts, timing, latency | Opt-in |
| `RAW_ORDER_EVENTS` | Full order details | Enterprise-only, opt-in |

### 8.2 Mandatory Redaction

**ВСЕГДА** маскируются:

- Broker API keys
- Account identifiers
- IP addresses
- Environment variables
- Любые секреты (pattern matching)

### 8.3 Data Residency

- EU tenants: EU region by default
- Enterprise: configurable residency
- Local telemetry mode: data stays on-prem

## 9. CI Guardrails

### 9.1 Build-time Checks

| Check | Description | Failure Action |
|-------|-------------|----------------|
| `no-trading-libs-in-cloud` | Cloud build не содержит order_execution | Block build |
| `no-order-payloads-in-schema` | JSON schema не содержит side/qty/price | Block merge |
| `artifact-signature-required` | Артефакт подписан | Block publish |
| `redaction-enabled` | Telemetry redaction включен | Block deploy |
| `import-boundary-check` | No agent imports in cloud | Block build |

### 9.2 Runtime Checks

| Check | Description | Failure Action |
|-------|-------------|----------------|
| `signature-verification` | Agent verifies artifact signature | Reject artifact |
| `schema-version-check` | Compatible schema versions | Reject command |
| `approval-required` | Trading-impacting needs approval | Queue for approval |
| `hard-cap-enforcement` | Local limits enforced | Reject/limit |

## 10. Rollout Plan Mapping

| Design Doc Section | Plan Phase | Description |
|--------------------|------------|-------------|
| 0-5 (Architecture) | Phase 0 | Inventory, decisions, target arch |
| Skeleton | Phase 1 | E2E minimal, basic guardrails |
| Separation | Phase 2 | Cloud/Agent/Shared split |
| Strategy API | Phase 3 | Intent contract, sim/live parity |
| Artifact Builder | Phase 4 | Immutable, signed, manifest |
| Agent Daemon | Phase 5 | Vault, sandbox, policy, reconciliation |
| Control Plane | Phase 6 | Data model, RBAC, trust/revoke |
| Protocol | Phase 7 | State machines, approvals, idempotency |
| Telemetry | Phase 8 | Privacy, GDPR, residency |
| Enterprise | Phase 9 | On-prem, evidence pack, updates |
| Cloud Jobs | Phase 10 | Isolation, anti-abuse |
| Documentation | Phase 11 | Docs, legal, marketing |

## Appendix A: Sequence Diagrams

См. [CCEA_SEQUENCE_DIAGRAMS.md](./CCEA_SEQUENCE_DIAGRAMS.md)

## Appendix B: JSON Schemas

См. [docs/schemas/](../../schemas/)

## Appendix C: Traceability Matrix

См. [CCEA_TRACEABILITY_MATRIX.md](./CCEA_TRACEABILITY_MATRIX.md)

---

**Document Control:**
- Author: CCEA Architecture Team
- Reviewers: Security, Compliance, Engineering
- Approval: Architecture Review Board
