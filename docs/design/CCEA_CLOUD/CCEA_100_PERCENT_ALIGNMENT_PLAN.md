# План доведения проекта до 100% соответствия Design Doc CCEA Cloud (5 фаз)

Дата: 2025-12-16
Статус: ✅ **COMPLETED** - Все фазы реализованы
Источник требований: `Design Doc CCEA Cloud.txt` (в корне репозитория)

## ИТОГОВЫЙ СТАТУС

**100% Design Doc Compliance достигнуто:**
- Все 10 фаз Master Remediation Plan завершены
- 117 тестовых файлов покрывают все требования
- Полная документация: docs/cloud/, docs/agent/, docs/runbooks/
- Enterprise deployment: deploy/helm/ccea-cloud/

См. [CCEA_TRACEABILITY_MATRIX.md](./CCEA_TRACEABILITY_MATRIX.md) для детальной трассируемости.

---

## 0) Нерушимые инварианты (проверяются всегда)

1) **Cloud никогда не хранит секреты** (broker/exchange API keys, secrets, tokens), не имеет механизма принять их в API/БД/логах, и не пишет их в telemetry/support dumps.  
2) **Cloud никогда не генерирует и не передаёт live-trading инструкции**: orders/targets/intents/signals для live.  
3) **Agent — единственная зона**, где: secrets, live decision loop, risk enforcement, order creation/sending, kill switch, reconciliation/journaling.  
4) **Любые TRADING_IMPACTING изменения требуют локального approve на Agent** (по умолчанию). Cloud может инициировать request, но не может “применить” без локального решения.  
5) **Артефакты стратегии immutable + pinned digest + signed + manifest + SBOM. Agent всегда верифицирует.**  
6) **Agent↔Cloud протокол: outbound-only из Agent, команды идемпотентны, есть audit trail.**  
7) **Надёжность и деградация**: reconciliation, safe-halt при неопределённости, cloud-down/network-down режимы, локальные журналы.  
8) **Privacy/GDPR**: минимизация телеметрии, уровни чувствительности, retention+deletion, EU residency по умолчанию для EU.  
9) **Enterprise**: on-prem/VPC режим, evidence pack export, change management и access audit.
10) **Non-goals соблюдены**: не реализуем managed-hosting агента нами, copy-trading/соц-торговлю, “cloud broker integrations с ключами пользователя”, “guaranteed profit/best execution promises” — и это поддержано guardrails/процессом.

## Как читать план

- Каждый пункт: **что сделать** → **где** → **критерий Done** (и иногда **проверка/тест**).  
- Если пункт уже реализован частично, всё равно остаётся в плане как “проверить/довести до operational”.
- Цель: чтобы любой агентный кодер мог идти по пунктам и не “потерять” ни одно требование дизайн-дока.

---

## Фаза 1 — Канонизация архитектуры и устранение дрейфа (Single Source of Truth)

Цель: убрать параллельные стеки (agent/control-plane/contracts), зафиксировать зоны, чтобы дальнейшая реализация не расходилась.

### 1.1 Выбрать канонический стек Agent
- **Сделать**: выбрать один runtime как production:  
  - либо `packages/agent/*` (рекомендуемо, т.к. уже привязан к `packages/cloud/control_plane/*`),  
  - либо `ccea/agent/*` (тогда нужно перепривязать cloud control plane и guardrails).
- **Где**: `ccea/agent/*`, `packages/agent/*`, `docs/agent/*`, `ARCHITECTURE.md`.
- **Done**:
  - Только один стек объявлен “production-grade” в документации и CLI entrypoints.
  - Второй стек помечен как `legacy/experimental` и/или вынесен в `docs/archive` + удалён из сборочных путей.
  - В CI/Makefile нет “двойного” пути, который может случайно собрать/запустить неправильный агент.

### 1.2 Выбрать канонический стек Cloud Control Plane
- **Сделать**: устранить дублирование control-plane слоёв/команд/enum между `ccea/control_plane/*` и `packages/cloud/control_plane/*`.
- **Где**: `ccea/control_plane/*`, `packages/cloud/control_plane/*`, `ccea/contracts/enums.py`.
- **Done**:
  - Единственный источник истины для команд/статусов/протокольных контрактов и DB state machine.
  - Все импорты в cloud-сервисах указывают на “canonical” контракты.

### 1.3 Централизовать контракты протокола/enum/state machines
- **Сделать**:
  - Зафиксировать: schema (`docs/schemas/protocol_messages.schema.json`) — authoritative.
  - Сгенерировать/синхронизировать enum и mapping в коде (если не генерируем — то хотя бы один модуль-контракт + тест, что он совпадает со schema).
- **Где**: `docs/schemas/protocol_messages.schema.json`, `ccea/contracts/enums.py`, cloud/agent types.
- **Done**:
  - Нет расхождений между allowlist команд в schema, в cloud, и в agent.
  - Нет расхождений по статусам/стейт-машинам (deployment/run/command) между DB и протоколом.
  - Есть тест/проверка “schema == code contract”.

### 1.4 Зафиксировать “Cloud build не содержит trading libs” как единственный поставочный путь
- **Сделать**: закрепить, что production cloud-deploy использует только zone-separated artifact (`ccea_cloud`), а не “монолитную” установку.
- **Где**: `tools/build_zone_distributions.py`, `Makefile`, `deploy/*`, `pyproject.toml`.
- **Done**:
  - Документация деплоя cloud указывает только на `dist-cloud` / `artifact-check-cloud`.
  - Любые legacy инструкции (которые тянут брокерские зависимости в cloud) помечены как запрещённые.

### 1.5 Убрать/заблокировать legacy live-раннеры в “production”
- **Сделать**: запретить использование `script_live.py` и аналогичных “legacy live” путей как production execution (оставить только для dev/sim).
- **Где**: `ARCHITECTURE.md`, `README.md`, `script_live.py`, CI/guardrails.
- **Done**:
  - В docs явное правило: production live только через Agent daemon.
  - В CI/packaging исключены пути, которые могли бы привести к “cloud execution”.

---

## Фаза 2 — Cloud Control Plane: operational governance, audit, privacy (не “заглушки”)

Цель: привести cloud к реальному multi-tenant SaaS/control plane из дизайн-дока: RBAC, audit, retention/residency, DSAR, break-glass, запреты на секреты и order-like payload.

### 2.0 RBAC и tenant isolation должны быть не декларацией
- **Сделать**:
  - Все endpoints, которые читают/пишут tenant-scoped данные, должны проверять workspace/org доступ и роли.
  - Включить и реально использовать RLS (Postgres) в production режимах; для dev/test — явные фильтры по `workspace_id`.
- **Где**: `packages/cloud/control_plane/models.py`, `packages/cloud/control_plane/dependencies.py`, все routers/services.
- **Done**:
  - Нельзя получить доступ к данным другого workspace ни через один endpoint.
  - Есть тесты на cross-tenant access denial (RLS/фильтры).

### 2.1 Довести governance endpoints до DB-backed реализации
- **Сделать**: заменить in-memory singleton сервисы governance на DB-backed модели и транзакции (DSAR, retention, residency, break-glass, access audit).
- **Где**: `packages/cloud/control_plane/routers/governance.py`, `packages/cloud/control_plane/models.py`.
- **Done**:
  - DSAR/Retention/Residency/Break-glass создают записи в БД (tenant-scoped), а не держатся в памяти процесса.
  - Любое “break-glass” пишет evidence hash + reason и попадает в `AccessAudit`.

### 2.1A Agent Registry / Trust / Revoke (security ops)
- **Сделать**:
  - Полный жизненный цикл агента: ENROLL → (SUSPEND/REVOKE) → блокировка команд/доступа.
  - Trust-state enforcement на agent-auth endpoints (heartbeat/poll/ack/result/approval): revoked/suspended не могут продолжать.
  - Issue/rotate agent session tokens (security ops) с аудитом.
- **Где**: `packages/cloud/control_plane/models.py`, `packages/cloud/control_plane/routers/agents.py`, `packages/cloud/control_plane/routers/agent_lifecycle.py`, `packages/cloud/control_plane/routers/auth.py`.
- **Done**:
  - Отозванный агент не может поллить команды/слать telemetry/принимать обновления.
  - Все revoke/suspend действия попадают в audit trail.

### 2.2 Реальный AccessAudit для “кто смотрел чувствительное”
- **Сделать**: добавить audit logging на чтение/экспорт/просмотр чувствительных данных (telemetry detailed/raw, approvals, export logs).
- **Где**: `packages/cloud/control_plane/models.py` (`AccessAudit`), routers/services.
- **Done**:
  - Есть единый helper/middleware для записи audit events.
  - Есть тесты, что audit появляется при чувствительных операциях.

### 2.3 Retention/Deletion как механизм (а не только политика)
- **Сделать**: реализовать auto-purge jobs по retention policy и DSAR deletion workflow.
- **Где**: `packages/cloud/governance/retention.py`, scheduler/worker (если есть), `deploy/*`.
- **Done**:
  - Есть периодический job (cron/celery/systemd) для purge.
  - Есть тест/интеграционная проверка на удаление данных по policy.

### 2.4 Data residency enforcement
- **Сделать**: реализовать реальное разделение хранения по регионам/режимам (EU default для EU), и “telemetry stays local” для enterprise/on-prem.
- **Где**: `packages/cloud/governance/residency.py`, `deploy/*`, storage/config.
- **Done**:
  - Residency policy влияет на маршрутизацию/хранение (object store/db region).
  - Для LOCAL_ONLY/air-gapped: ingestion/экспорт наружу выключаем в конфиге и это enforced.

### 2.5 Telemetry ingestion: уровни чувствительности + дефолты + redaction mandatory
- **Сделать**:
  - Зафиксировать default: AGGREGATED (retail).
  - RAW_ORDER_EVENTS только opt-in (и желательно enterprise).
  - Redaction обязательна и неотключаема.
- **Где**: `packages/agent/telemetry/*`, `packages/cloud/control_plane/routers/telemetry.py`, schema/docs.
- **Done**:
  - Cloud rejects telemetry с запрещёнными полями/секретами.
  - Agent не может отправить telemetry без redaction middleware.

### 2.6 Cloud research job isolation/anti-abuse довести до production
- **Сделать**: sandbox isolation + quotas + egress allowlist + abuse detection как рабочий контур, а не “пример”.
- **Где**: `packages/cloud/control_plane/routers/research_jobs.py`, `packages/cloud/research/sandbox/*`, `deploy/*`.
- **Done**:
  - Есть механизм запуска sandboxed jobs (container/VM) с ограничениями и аудитом.
  - Abuse detector реально может остановить job и оставить audit trail.

### 2.7 CI guardrails как “невозможность нарушить модель”
- **Сделать**: расширить/зафиксировать guardrails:
  - no broker clients in cloud deps,
  - запрет order-like payload в командах,
  - запрет “remote shell”/dangerous ops без enterprise режима и строгого аудита.
- **Где**: `.github/workflows/build-and-test.yml`, `ccea/guardrails/*`, `packages/cloud/control_plane/boundary.py`.
- **Done**:
  - Любая попытка добавить order-like команду/поле ломает CI.
  - Любая попытка завезти trading libs в cloud artifact ломает CI (post-build scan).

### 2.8 Запреты “навсегда” как контрольные проверки (не только слова)
- **Сделать**:
  - Статические/CI проверки, что в cloud runtime нет:
    - брокерских submitter’ов/OMS,
    - эндпоинтов/команд с order-like payload,
    - кода “remote shell/remote exec” без enterprise режима и строгого аудита.
- **Где**: `ccea/guardrails/*`, `.github/workflows/build-and-test.yml`, `packages/cloud/control_plane/boundary.py`.
- **Done**:
  - Добавление запрещённых паттернов ломает CI в PR.

---

## Фаза 3 — Артефакты/конфиги: supply chain end-to-end (build → sign → publish → pull → verify)

Цель: гарантировать immutable+signed+verified артефакт и immutable config blobs с digest и diff, как в дизайн-доке.

### 3.1 Привести Artifact Builder к честному формату и публикации
- **Сделать**:
  - Либо реально собирать OCI image (preferred), либо выставлять `format=ZIP_BUNDLE`, если это ZIP.
  - Публикация в registry/object store + digest pinned.
- **Где**: `packages/cloud/builder/artifact_builder.py`, `packages/cloud/builder/registry.py`, `packages/shared/contracts/manifest.py`.
- **Done**:
  - В manifest `format` соответствует реальному артефакту.
  - Артефакт доступен по digest ref, и агент может его получить без “магии”.

### 3.1A Идемпотентность команд на стороне Cloud (стабильные idempotency keys)
- **Сделать**:
  - Для lifecycle-команд генерировать **детерминированные** `idempotency_key` (например: `deployment_id:command_type:artifact_digest:config_digest`), чтобы повторная выдача “того же desired state” не создавала новый “уникальный” command.
  - Гарантировать, что “повтор” команды Cloud→Agent безопасен.
- **Где**: `packages/cloud/control_plane/services/command_service.py`, `packages/cloud/control_plane/routers/commands.py`.
- **Done**:
  - Повторная выдача того же запроса не плодит новые команды и не приводит к дубликатам на агенте.

### 3.2 Подпись артефактов: обязательность и проверка на Agent
- **Сделать**:
  - Cloud подписывает artifact + manifest blob.
  - Agent проверяет: digest, signature, allowlist registries, schema_version совместимость.
- **Где**: `packages/cloud/builder/*`, `ccea/crypto/*`, `packages/agent/daemon/preflight.py`.
- **Done**:
  - Агент отказывается запускать unsigned/invalid-signed.
  - Есть тесты на “reject unsigned / reject bad digest / reject wrong registry”.

### 3.3 SBOM + provenance
- **Сделать**: гарантировать, что SBOM генерится всегда, хранится/доступен, и provenance заполнен (git_sha, dataset_refs, training_run_id, params_hash).
- **Где**: `packages/cloud/builder/artifact_builder.py`, `packages/shared/contracts/manifest.py`, docs/schema.
- **Done**:
  - SBOM ref валиден и доступен.
  - Provenance обязательные поля проверяются.

### 3.4 Immutable Config Blobs (без секретов) + diff для approve
- **Сделать**:
  - Cloud хранит config как content-addressed blob (digest), изменения создают новый blob.
  - Агент может получить blob по digest и показать diff при approve.
- **Где**: `packages/cloud/control_plane/routers/config_blobs.py`, agent config fetcher (новый модуль).
- **Done**:
  - `REQUEST_UPDATE_CONFIG` содержит только digest ref.
  - Агент применяет конфиг только после approve (если TRADING_IMPACTING).
  - Есть механизм diff summary (локально).

### 3.5 “No silent upgrades” для trading-impacting
- **Сделать**: любые upgrade/config changes с change_class=TRADING_IMPACTING требуют approve. Cloud не может “снять” approve флагом.
- **Где**: cloud command issuance + agent enforcement (`packages/agent/daemon/agentd.py`).
- **Done**:
  - Agent fail-closed: если TRADING_IMPACTING без requires_approval — отказ и audit.
  - Cloud UI/API не позволяет создать TRADING_IMPACTING команду без approval_required.
  - `REQUEST_STOP_RUN`/`REQUEST_PAUSE_RUN` классифицированы как safety/operational и **могут** применяться без approve (как снижение риска), но это не даёт Cloud “рычаг” управления торговым поведением (нет сигналов/targets/orders).

---

## Фаза 4 — Agent: реальный live runtime (execution/risk/reconciliation/degraded/telemetry)

Цель: сделать Agent единственной зоной исполнения, с реальным live-loop, risk firewall, journaling, reconciliation, kill switch и автономностью.

### 4.1 Реальное выполнение lifecycle-команд
- **Сделать**:
  - `REQUEST_START_RUN`: загрузить артефакт, верифицировать, применить конфиг, запустить run.
  - `REQUEST_STOP_RUN`/`REQUEST_PAUSE_RUN`: реально управлять run.
  - `REQUEST_UPGRADE_ARTIFACT`: pull+verify+approve+swap.
  - `REQUEST_UPDATE_CONFIG`: fetch blob+diff+approve+apply.
- **Где**: `packages/agent/daemon/agentd.py`, `packages/agent/cloud/client.py`, `packages/agent/runner/live.py`.
- **Done**:
  - Нет placeholder-веток “acknowledged” без эффекта.
  - Все команды идемпотентны (повтор не приводит к повторной торговле/дубликатам).

### 4.2 Local Vault: secrets only local + rotation local
- **Сделать**:
  - Хранить broker creds только локально (keychain/encrypted file), без вывода в логи/telemetry.
  - Ротация ключей — локальная операция.
  - Поддержка нескольких broker accounts локально (явный выбор), без доступа cloud к ключам/деталям.
- **Где**: `packages/agent/vault/*`, `packages/agent/daemon/keychain.py`, `packages/agent/telemetry/redaction.py`.
- **Done**:
  - Невозможно отправить secret в cloud через telemetry/config/commands.
  - Есть CLI/операции для локальной ротации.
  - Есть локальная модель “account selection”, не требующая cloud.

### 4.3 Policy Firewall + Hard Caps (Cloud не может поднять риск)
- **Сделать**:
  - Явное приоритетное layering: local policy > cloud config > suggested in manifest.
  - Cloud не может ослабить hard caps.
- **Где**: `packages/agent/policy/firewall.py`, `packages/agent/policy/hard_caps.py`, config apply path.
- **Done**:
  - Любая попытка cloud поднять риск сверх hard caps → reject + audit + (опционально) halt.

### 4.4 Intent → Orders только локально (и Cloud не участвует)
- **Сделать**: live loop: snapshot → strategy → intent → risk manager → execution engine → broker connector.
- **Где**: `packages/shared/contracts/strategy.py`, `packages/shared/contracts/intent.py`, `packages/agent/execution/engine.py`, broker adapters (agent-only).
- **Done**:
  - Нет пути, по которому Cloud может подсунуть intent/target/order.
  - Стратегии в live получают только локальный snapshot/данные.

### 4.4A Sandbox/permissions enforcement для strategy runner
- **Сделать**:
  - Запуск стратегии в изоляции (process/container) с лимитами CPU/RAM.
  - Enterprise: network egress deny-by-default + allowlist; read-only FS (кроме tmp); запрет произвольных исходящих запросов.
  - Enforcement должен происходить на Agent, а не быть “только в manifest”.
- **Где**: `packages/agent/daemon/sandbox.py`, `packages/shared/contracts/manifest.py` (permissions), runner wiring.
- **Done**:
  - Стратегия не может выйти в сеть/FS вне разрешений; попытки фиксируются и приводят к halt/deny.

### 4.5 Kill switch (локальный) + причины/действия
- **Сделать**: реализовать triggers (loss, broker errors, latency spikes, order spam, divergence, data feed invalid) и действия (cancel, optional flatten по локальной политике, halt run).
- **Где**: `packages/agent/daemon/kill_switch.py`, интеграция с runner/execution.
- **Done**:
  - Kill switch реально останавливает торговлю и оставляет локальный журнал + телеметрию (redacted).
  - Cloud не имеет команды “REMOTE_FLATTEN/FLATTEN_NOW”; flatten допускается только локально (или enterprise-режим по отдельному контракту и с усиленным аудитом).

### 4.6 Reconciliation + idempotency (без дублей при retries/restarts)
- **Сделать**:
  - Детерминированный `client_order_id` для каждого ордера.
  - Персистентный journal/dedup на диске.
  - На старте/периодически: reconcile positions + open orders; при неопределённости — safe halt.
- **Где**: `packages/agent/execution/engine.py`, `packages/agent/reconciliation/*`, `packages/agent/runner/live.py`, daemon wiring.
- **Done**:
  - После рестарта не возникает duplicate orders.
  - Unresolved/unknown status → halt.

### 4.7 Safe-degraded режимы
- **Сделать**:
  - Cloud down: торговля может продолжаться (если локально разрешено), но мониторинг/команды деградируют.
  - Market data loss/качество данных: halt.
  - Broker errors burst/latency: halt.
- **Где**: `packages/agent/daemon/degraded_mode.py`, интеграция с runner/execution/kill switch.
- **Done**:
  - Явные политики деградации и переходы режимов.

### 4.8 Telemetry: уровни + буфер + redaction mandatory
- **Сделать**:
  - Буферизовать телеметрию локально (durable queue), отправлять aggregated по умолчанию.
  - RAW_ORDER_EVENTS только opt-in/enterprise.
  - Redaction неизбежна.
- **Где**: `packages/agent/daemon/telemetry_buffer.py`, `packages/agent/telemetry/redaction.py`, cloud ingestion.
- **Done**:
  - Telemetry не содержит запрещённых полей/секретов, даже при ошибках/stack traces.

### 4.9 Enrollment + Auth + outbound-only transport
- **Сделать**:
  - Enrollment token TTL, public key registration.
  - Agent auth: signed JWT/mTLS (в зависимости от режима), сообщения подписаны/аутентифицированы.
  - Только outbound соединение из Agent.
- **Где**: `packages/cloud/control_plane/routers/auth.py`, `packages/agent/cloud/client.py`, `packages/cloud/control_plane/routers/agent_lifecycle.py`.
- **Done**:
  - Нет inbound портов/требования открывать firewall для Agent.
  - Replay/idempotency защищены (idempotency keys + timestamps + audit).

### 4.10 Local approval UX (обязательный операторский контур)
- **Сделать**:
  - Предоставить локальный интерфейс approve (CLI как минимум; GUI опционально) для TRADING_IMPACTING.
  - UX должен показывать: что меняется (artifact digest/config digest), дифф/summary, affected instruments/universe/risk, локальные hard caps.
- **Где**: `packages/agent/approval/*`, daemon integration, `docs/agent/APPROVALS.md`.
- **Done**:
  - Любой TRADING_IMPACTING request без решения оператора не применяется.
  - Evidence hash/attestation сохраняются локально и репортятся в cloud (без секретов).

### 4.11 Local auto-approve policies (should-have, но для “100%” должны быть)
- **Сделать**:
  - Локальные правила auto-approve с allowlist (workspace/strategy/instruments/change types) — только на Agent и только stricter/explicit.
  - Cloud не может включить auto-approve удалённо и не может расширить allowlist.
- **Где**: `packages/agent/approval/*`, `packages/agent/policy/*`, `docs/agent/APPROVALS.md`.
- **Done**:
  - Auto-approve действует только по локально заданной политике, с аудитом (кто/когда/по какому правилу).
  - Для “опасных” изменений (risk limit raise, broker account change, universe расширение) auto-approve по умолчанию запрещён.

---

## Фаза 5 — Enterprise/On-Prem, Updates, Evidence Pack, Legal/Marketing “не противоречат архитектуре”

Цель: закрыть enterprise-ready требования и синхронизировать “как продаём” с “как устроено”, чтобы не стать execution service/advice.

### 5.1 On-prem/VPC/air-gapped: довести deployment pack до рабочего состояния
- **Сделать**: обеспечить запуск cloud control plane + registry mirror + governance/telemetry локально, без внешней сети.
- **Где**: `deploy/docker/*`, `deploy/helm/*`, `docs/cloud/ENTERPRISE.md`.
- **Done**:
  - Есть воспроизводимый сценарий развертывания (docker-compose и helm).
  - Есть “air-gapped” режим с отключением внешних экспортов/апдейтов.

### 5.2 Evidence pack exporter (auditability)
- **Сделать**: экспортировать (и опционально подписывать) набор evidence:
  - digests/signatures/SBOM,
  - журнал deploys/commands/approvals,
  - причины halt/инциденты,
  - конфиги retention/residency,
  - (опционально) telemetry export по уровню чувствительности.
- **Где**: `packages/cloud/enterprise/evidence_pack.py`, cloud API/CLI для экспорта.
- **Done**:
  - Экспорт воспроизводим и верифицируем (hash/подпись).
  - Экспорт учитывает residency/retention политики.

### 5.3 Signed Agent updates + staged rollout + rollback + enterprise pinning
- **Сделать**: механизм обновлений агента:
  - подписанные релизы,
  - staged rollout,
  - rollback protection,
  - pin versions/change windows в enterprise.
- **Где**: `packages/cloud/enterprise/agent_updates.py`, `packages/cloud/enterprise/tuf_repository.py`, agent updater.
- **Done**:
  - Агент принимает только доверенные обновления.
  - Enterprise может зафиксировать версию и окно изменений.

### 5.4 “Break-glass” доступ: строго аудируемый
- **Сделать**: реализовать break-glass flow с reason/evidence/audit, ограничить scope (telemetry/export) и роли.
- **Где**: governance + RBAC + `AccessAudit`.
- **Done**:
  - Любой break-glass оставляет trace и доступ ограничен временем/ролью.

### 5.5 Legal/Marketing/AUP: слова не противоречат архитектуре
- **Сделать**:
  - Исправить ToS/Privacy/DPA формулировки, исключить “cloud executes orders / stores keys”.
  - Запретить маркетинговые фразы “we trade for you / cloud auto-execution”.
  - AUP: запрет market abuse/order spam и право ограничить cloud compute.
- **Где**: `docs/legal/TERMS_OF_SERVICE.md`, `docs/legal/PRIVACY_POLICY.md`, `docs/legal/DPA_TEMPLATE.md`, `docs/legal/AUP.md`, `docs/legal/ACCEPTABLE_USE_POLICY.md`, investor/marketing docs.
- **Done**:
  - Нет конфликтующих определений/секций в ToS.
  - Privacy/DPA явно описывают: broker creds и execution только в customer-managed Agent.
  - AUP согласован с техограничениями (abuse detection/quotas).
  - Non-goals явно зафиксированы: “мы не хостим агент”, “не копитрейдинг”, “cloud не брокер/не советник”, “нет cloud order routing”.

### 5.6 Incident playbooks/runbooks
- **Сделать**: runbooks для broker errors/latency/data loss, kill switch, revoke/rotation, recovery без доступа к секретам.
- **Где**: `docs/runbooks/*` (создать/дополнить), `docs/OPERATIONS_RUNBOOK.md` (согласовать).
- **Done**:
  - Есть операционные инструкции “как расследовать” при минимизации данных и без доступа к ключам.

---

## Сквозные критерии “100% соответствие” (Definition of Done)

Чтобы считать проект “100% соответствующим Design Doc”, должны выполняться одновременно:

1) **Граница Cloud/Agent не только описана, но и enforced**: схемами + CI + поставочным артефактом + runtime fail-closed.  
2) **Live execution в реальности работает только через Agent**, Cloud не может отправить ни orders, ни intents, ни targets, и не может повысить риск сверх локальных hard caps.  
3) **TRADING_IMPACTING изменения реально требуют local approve**, включая upgrade/config/universe/risk/broker-account/schedule/mode.  
4) **Артефактный цикл supply chain рабочий end-to-end**: build→sign→publish→pull→verify→run, с manifest+SBOM+provenance.  
5) **Идемпотентность и надёжность**: deterministic client_order_id, persistent journal, reconciliation, safe-halt при неопределённости.  
6) **Privacy/GDPR/Residency**: минимизация telemetry, уровни чувствительности, retention+deletion, EU residency default, enterprise local-only режим.  
7) **Enterprise readiness**: on-prem pack + evidence pack + signed updates + change windows/pinning + break-glass audit.  
8) **Docs/ToS/Marketing** не противоречат архитектуре (никаких “cloud executes/stores keys”).
