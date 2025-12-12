План приведения проекта к 100% соответствию Design Doc CCEA Cloud.txt (v2)

Цель: довести проект до полного соответствия Design Doc CCEA Cloud.txt, включая технические guardrails, протокол и безопасность, а также legal/marketing‑часть (ToS/AUP/позиционирование), потому что по Design Doc это часть архитектуры.

Ключевой принцип (не обсуждается): Cloud = research/build/monitoring/control plane (lifecycle requests), Agent = secrets + live loop + risk enforce + order creation/sending. Cloud никогда не хранит ключи, не имеет кода/доступа к торговым API от имени пользователя и не передаёт live‑торговые инструкции/ордера/targets.

Ниже — фазы с конкретными задачами и “done‑критериями”. План закрывает обязательные разделы Design Doc: требования (3–5), модель данных (6), change_class/policy firewall (7), артефакты (8), agent runtime (9), протокол (10), state machines (11), config layering (12), telemetry/privacy/residency (13–14), security (15), enterprise/evidence pack (16), AI Act posture и “не advice” (17–18), CI guardrails (19), rollout (20), open questions (21).

Примечания по трассируемости (иначе “100% соответствие” не проверяемо):
- Design Doc должен быть доступен/версионирован для ревью и CI (например: снапшот в `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt` или ссылка + `sha256` + дата версии).
- Явно зафиксировать мэппинг на Rollout plan (Design Doc 20): Skeleton → Lifecycle+Approvals → Live+Risk → Telemetry+Privacy → Enterprise pack (мэппинг ниже в Done Фазы 0).

Фаза 0. Инвентаризация, решения по Open Questions, целевая архитектура (1–2 недели)

1) Карта текущих модулей по зонам:
- Shared: всё, что безопасно и нужно обоим рантаймам (core_*, impl_*, симуляция/бэктест/тренинг, общие модели/контракты, DI/YAML).
- Agent (live‑only): broker connectors/private trading clients, создание/отправка ордеров, reconciliation, local vault, policy firewall/hard caps, kill switch.
- Cloud (cloud‑only): UI/IDE, backtest/sim/training orchestration, builder/registry/control plane, telemetry/monitoring, governance (RBAC/retention/residency), sandbox для cloud research jobs.

2) “Граница периметра” и запреты Cloud:
- Список запрещённых зависимостей/импортов/библиотек для cloud‑сборки (любые broker submitters, private trading clients).
- Список запрещённых типов сообщений и полей (никаких PLACE_ORDER/SUBMIT_ORDER/EXECUTE_SIGNAL/SET_TARGET_POSITION, и никаких payload, которые можно интерпретировать как готовый ордер).
- Запрет “remote shell/remote code exec” в Agent из Cloud в retail‑режиме; любые enterprise‑диагностические команды только с redaction и строгим аудитом.
- Secret hygiene: зафиксировать, как гарантируем “секреты не попадают в cloud” (secret scanning + redaction + запрет support dumps с секретами).

3) Зафиксировать продуктовые режимы (Design Doc 0.7):
- Retail Research SaaS (EU-friendly): cloud research/sim/monitoring + BYO agent (опционально).
- Retail Live via Local Agent: auto-execution локально, cloud — наблюдаемость и lifecycle requests.
- Enterprise Engine (on‑prem/VPC/self‑hosted): execution внутри инфраструктуры клиента + vendor pack.

4) Закрыть Open Questions из Design Doc (21) документом Decision Log:
- Минимальный sandbox для retail agent: docker‑required или process‑ok (и что делать при отсутствии docker).
- Политика RAW order telemetry: кто может, когда, по умолчанию (рекомендация: enterprise‑only, opt‑in).
- “Flatten позиции”: разрешаем ли как удалённый request (рекомендация: по умолчанию только локально; enterprise — по договору).
- “Remote brain, local finger” (cloud inference): явное решение “не делаем” в этой итерации.
- Data sensitivity классификация (какие поля персональные/чувствительные, что считается trading‑sensitive/IP).
- Что считаем достаточным доказательством “customer‑managed host” в BYO/VPS сценарии (процесс/ToS/онбординг).

5) Threat model и безопасность по умолчанию:
- Threat model (RCE/эксфильтрация ключей/подмена артефакта/“cloud becomes execution service”/abuse облачных джобов).
- Safe defaults, включая “no silent upgrades” для trading‑impacting.

Done:
- Документ “Target CCEA Architecture” (диаграмма Cloud/Agent/Shared + таблица модулей “куда относится”).
- Decision Log по Open Questions (Design Doc 21) + зафиксированные дефолты.
- Черновик JSON схем (manifest + protocol) и список CI guardrails (Design Doc 19).
- Зафиксированная версия Design Doc (снапшот/ссылка + `sha256`) + короткий мэппинг фаз плана на Rollout plan (Design Doc 20).
- Матрица трассируемости “Design Doc требование → план/фаза → код/док/CI‑check” (1 файл, чтобы потом не спорить словами).

Фаза 1. Skeleton end‑to‑end (минимальный вертикальный срез) + базовые guardrails (2–4 недели)

Цель фазы: получить работающий “CCEA‑скелет” до крупной реструктуризации, чтобы валидировать протокол, подпись артефактов и approvals без брокера.

1) Минимальный Cloud control plane:
- Enrollment token (TTL) → enroll → heartbeat → long‑poll commands → ack/result/approval endpoints.
- Хранилище immutable blobs (config/manifest) по digest (без секретов).

2) Минимальный Agent daemon (agentd) + локальный approve:
- agent enroll --token: генерит device keypair локально, регистрирует public key, получает agent_id.
- Outbound‑only: agent делает poll, принимает только allowlisted команды REQUEST_* (на старте: REQUEST_START_RUN/STOP_RUN/PAUSE_RUN/UPGRADE_ARTIFACT/UPDATE_CONFIG; ops: ROTATE_AGENT_SESSION/EXPORT_LOGS — если нужны).
- Local approval UI/CLI: подтверждение TRADING_IMPACTING запросов (start/upgrade/config) и запись evidence_hash.

3) Аутентификация/подпись каждого сообщения Agent↔Cloud (Design Doc 10.2):
- Вариант A (enterprise‑friendly): mTLS.
- Вариант B: signed JWT на device key + server verification.
- Каждое сообщение аутентифицировано/подписано; ротация/отзыв ключа предусмотрены архитектурно (полная реализация — Фаза 6/9).

4) Минимальный Artifact Builder:
- “Hello strategy” артефакт (без брокера): собрать OCI image (предпочтительно) или zip‑bundle.
- Digest‑pinning + подпись (sigstore/cosign или GPG) + минимальный SBOM + manifest (schema_version).
- Agent: pull только по digest и только из allowlist registry; verify digest+signature+schema_version.

5) Базовые CI guardrails (Design Doc 19) включить уже здесь:
- Cloud build не содержит broker trading client libs (статический анализ импортов/зависимостей).
- JSON schema запрещает order‑like payload (side/qty/price и аналоги) в командах.
- Pipeline не публикует артефакт без подписи; agent rejects unsigned.
- Telemetry redaction mandatory (нельзя выключить фичефлагом).

Done:
- E2E: user в cloud создаёт deployment → cloud отправляет REQUEST_START → agent требует local approve → agent тянет подписанный артефакт по digest → запускает run “hello strategy” (без брокера) → репортит state/telemetry.
- Протокол аутентифицирован, команды идемпотентны, order‑like payload невозможен по схеме/CI.

Фаза 2. Жёсткое разделение Cloud/Agent/Shared (3–4 недели)

Реструктурировать репо в 3 пакета/дистрибутива (монорепо, но отдельные сборки):
- packages/shared — безопасные контракты/модели/симуляция/инфра.
- packages/agent — live loop + broker connectors + local vault + policy firewall + approvals.
- packages/cloud — research/backtest/training UI + builder/registry/control plane, без trading‑кода.

Разрезать адаптеры на data‑only vs trading‑only:
- публичные market‑data клиенты остаются в shared/cloud;
- private‑trading клиенты и submitters — только в agent.

Расширить guardrails:
- allowlist зависимостей для cloud (проверка транзитивных deps тоже).
- запрет импорта live‑модулей из cloud на уровне CI (расширить существующий `importlinter.ini` контрактами cloud↔agent).
- сборочная проверка содержимого cloud‑артефакта (wheel/OCI): “в cloud build нет live‑модулей/`order_execution`/private trading clients” (не только import‑уровень).

Done:
- cloud‑образ/сборка физически не содержит trading client libs и не может обратиться к broker trading API (ни прямо, ни транзитивно).
- agent‑live собирается отдельно и содержит всё live‑необходимое.

Фаза 3. Strategy API и сим/лайв паритет (2–3 недели)

1) Единый контракт стратегии = Intent:
- Стратегия возвращает Intent (например, OrderIntent из core_models.py).
- В cloud: Intent → SimExecutionEngine (только симуляция).
- В agent: Intent → local Risk Manager (enforce + hard caps) → Execution Engine → Orders → Broker Connector.

2) Убрать/зафиксировать legacy:
- “Decision/готовые ордера” не используются как контракт для live; миграция стратегий и раннеров.

3) Запрет “signal/intent push” из cloud:
- В cloud API/протоколе отсутствуют сообщения, несущие live‑intent/targets.
- Cloud отправляет только lifecycle requests (REQUEST_START/STOP/PAUSE/UPGRADE/UPDATE_CONFIG и т.п.).

Done:
- Генерация конкретных Orders находится только в agent‑раннере; cloud не умеет и не может “подсунуть” intent/targets в live.

Фаза 4. Artifact Builder: immutable + signed + manifest + SBOM + provenance (4–6 недель)

В cloud‑пакете реализовать builder pipeline:
- OCI image (digest‑pinned) как основной формат; zip/wheel — только как fallback с явными ограничениями.
- manifest.json/yaml по схеме Design Doc: schema_version, entrypoint, runtime, deps lock digest, model refs (digests), data_contract, permissions (fs/network), risk_profile_suggested (только suggested), telemetry_schema_version, change_class, provenance (git_sha, dataset_refs, training_run_id, params_hash).
- Добавить в manifest: live_capabilities (нужен ли broker access/какие sandbox’ы нужны), чтобы агент мог заранее валидировать окружение и UX (Design Doc приложение 2.1).
- SBOM (CycloneDX/SPDX) + ссылка (sbom_ref) в Build/registry.
- Подпись артефакта и manifest blob (sigstore/cosign или GPG).
- Key management (сразу зафиксировать в Decision Log): как храним/ротируем ключи подписи (keyless sigstore vs keyful для enterprise/offline) и что считаем trust root.

Agent verification:
- verify digest + signature + allowlist registry + schema_version совместимость.
- строгий reject: unsigned/unknown registry/unknown schema_version.

Done:
- Без подписи артефакт не публикуется; agent не запускает артефакт без успешной верификации.
- В cloud хранятся только digest/ref/метаданные; никаких секретов.

Фаза 5. Agent Daemon: Local Vault + Sandbox + Policy Firewall + Reconciliation + Safe‑degraded (4–6 недель)

1) Local Vault (secrets):
- Использовать существующий CredentialVault как Local Vault.
- Зафиксировать источник master key (предпочтительно OS keychain; fallback: encrypted file + env var), и что cloud никогда не получает ни master key, ни “backup”.
- Ротация ключей — локальная операция.
- Поддержка нескольких broker accounts (явный выбор локально).
- Гарантии: секреты не попадают в логи/телеметрию (redaction/DLP).

2) Sandbox/изоляция стратегии:
- Базово: стратегия в отдельном процессе/контейнере + лимиты CPU/RAM.
- Enterprise: ro‑fs, egress allowlist, deny‑by‑default outbound network, запрет произвольных исходящих запросов.

3) Policy Firewall / hard caps (Design Doc 7.3):
- Локальные абсолютные верхние границы риска/ограничения типов ордеров/инструментов.
- Cloud не может поднять риски выше hard caps никогда, даже при approve.
- Локальная политика имеет приоритет над cloud config и risk_profile_suggested из manifest.

4) Pre-flight проверки перед стартом/апгрейдом (Design Doc D1):
- verify signature + digest + schema_version (manifest + protocol).
- verify broker connectivity + permissions (без раскрытия секретов в cloud).
- verify local policy firewall/hard caps.
- verify time sync (допустимый drift) и корректность timestamps/idempotency.

5) Kill Switch + halt reasons (Design Doc 9.4):
- Триггеры: max daily loss, broker errors burst, latency spike, order spam, state divergence, data feed invalid.
- Действия: cancel open orders → optional flatten (только если локально разрешено) → halt run; репорт причины в telemetry (с учётом уровня чувствительности).

6) Reconciliation/idempotency:
- Детерминированный client_order_id для ордеров (idempotent).
- На рестарте: fetch open orders/positions → reconcile local journal → если неопределённость → safe halt.
- Никаких дублирующих ордеров из-за retries.

7) Local journal + telemetry buffer:
- Durable очередь событий (sqlite/jsonl), восстановление после рестарта.
- Degraded safe режимы: cloud down / network down / data feed invalid → halt или ограничение по локальной политике.

Done:
- Агент автономно держит live‑loop, хранит ключи, enforce’ит hard caps, восстанавливается и безопасно деградирует без cloud.

Фаза 6. Cloud Control Plane: модель данных, RBAC, trust/revoke, blobs (6–8 недель)

1) Cloud сервис (FastAPI) + БД (Postgres) + multi‑tenant модель данных:
- Org/Workspace/User/Roles/Permissions (+ access audit).
- Strategy/StrategyVersion, Build/Artifact (digest, signature_ref, sbom_ref, provenance, change_class).
- Agent (+ public_key, agent_version, capabilities, trust_state ENROLLED/REVOKED, last_seen).
- AgentEnrollmentToken (TTL).
- Deployment/Run.
- Command (+ idempotency_key, payload_ref digest, requires_approval, status).
- ApprovalRecord (+ evidence_hash/attestation).
- TelemetryEvent/Alert.
- DataRetentionPolicy/AccessAudit (break‑glass events).
- Tenant boundary enforce (минимум: обязательный workspace_id; рекомендовано: Postgres RLS, чтобы “не ошибиться” кодом).

2) Endpoints:
- Create enrollment token (TTL), enroll, heartbeat.
- Poll commands (long‑poll), ack, approval, result.
- Telemetry ingestion.
- Admin: revoke agent/rotate device key/disable deployments.

3) Immutable config blobs:
- Cloud хранит config как blob с digest; любые изменения → новый digest; команды ссылаются на digest.
- Cloud хранит только desired state (не секреты).

Done:
- Минимально рабочий multi‑tenant control plane с revoke/rotation механикой и immutable blobs по digest.

Фаза 7. Протокол/State Machines/Approvals: версии схем, allowlists, идемпотентность (4–6 недель)

1) Протокол и схемы:
- JSON schemas сообщений: HEARTBEAT, POLL_COMMANDS, COMMAND_BATCH, COMMAND_ACK, COMMAND_APPROVAL, COMMAND_RESULT, TELEMETRY.
- schema_versioning + min_supported/max_supported negotiation.
- Allowlist command types; любой новый тип = security review.
- Initial safe list command types (Design Doc приложение E2): REQUEST_START_RUN, REQUEST_STOP_RUN, REQUEST_PAUSE_RUN, REQUEST_UPGRADE_ARTIFACT, REQUEST_UPDATE_CONFIG, REQUEST_ROTATE_AGENT_SESSION, REQUEST_EXPORT_LOGS (с redaction).

2) Запрет “order‑like payload” (Design Doc 10.5/19.2):
- На уровне схемы: нет полей/структур, которые можно интерпретировать как готовый ордер (side/qty/price и аналоги).
- На уровне CI: тест, что схемы не содержат запрещённых полей и запрещённых команд.

3) Idempotency/dedup:
- Dedup по idempotency_key на agent и cloud.
- Команды idempotent: повторная доставка не меняет итоговое состояние.

4) State machines:
- Deployment state и Run state ровно как в Design Doc (11.1–11.2).

5) Approvals:
- TRADING_IMPACTING всегда требует local approve по умолчанию (стратегия/модель/build, universe, execution params, risk limits, schedule, paper↔live, broker account/adapter).
- Agent показывает diff (config blob digest diff, build digest, universe diff, mode change) и применяет только после approve.
- Auto‑approve только через локальную политику (whitelists/thresholds), причём cloud не может включить auto‑approve сам.
- Stop/Pause можно удалённо (safety); “flatten” по умолчанию локально (или enterprise‑режим по договору).

Done:
- Торгово‑значимые изменения невозможны без локального approve/локальной политики; cloud не может отправить order‑like payload; состояния и идемпотентность соответствуют Design Doc.

Фаза 8. Telemetry + Privacy/GDPR + Residency + Access Controls (3–5 недель)

1) Telemetry уровни и дефолты:
- AGGREGATED default (retail).
- DETAILED_NON_SENSITIVE по опции.
- RAW_ORDER_EVENTS только opt‑in и по умолчанию выключено (рекомендация: enterprise‑only).

2) Mandatory redaction + DLP:
- Агент не может отправить telemetry без включённого redaction middleware.
- Запрет логирования env vars; маскирование account identifiers; фильтр типичных секретов.

3) Cloud governance:
- Retention per tenant, auto‑purge, экспорт/удаление (DSAR).
- RBAC на чувствительные данные, AccessAudit.
- Break‑glass только с причиной и аудитом события.

4) Data residency:
- EU region default для EU tenants.
- Enterprise: “telemetry stays local” режим или выборочный экспорт.

5) Monitoring/alerts (Design Doc 4.1/3.1):
- Дашборды health и состояний (agent online/offline, run state, halted reasons).
- Alerts по базовым событиям (kill switch, broker errors burst, data feed invalid, order spam).

Done:
- Телеметрия минимизирована, классифицирована и редактирована; есть управляемое хранение, residency и аудит доступа.

Фаза 9. Enterprise/on‑prem pack + signed agent updates (4–6 недель)

1) Реальный on‑prem/VPC комплект:
- docker‑compose/Helm для cloud‑стека.
- Опциональный registry mirror.
- Air‑gapped сценарий (если требуется enterprise).

2) Evidence pack exporter (Design Doc 16.1):
- digests/signatures/SBOM
- журналы deploy/upgrade/approvals/commands
- halt reasons/incident logs
- экспорт telemetry (по уровню чувствительности)

3) Agent updates (Design Doc 15.2/5.2):
- Подписанные обновления агента.
- Staged rollout + rollback.
- Enterprise: version pinning + change windows, min/max supported schema versions.
- (Рекомендация best practice) Защита от rollback/freeze атак обновлений: подписанные update‑метаданные (например TUF‑подобный подход), а не только подпись бинаря.

Done:
- Можно развернуть on‑prem/VPC контур, обновлять агента подписанно и управляемо, и выгрузить audit/evidence pack.

Фаза 10. Cloud research jobs: изоляция исполнения + anti‑abuse (2–4 недели)

Требование Design Doc (15.3/5.3): пользовательский код/джобы в cloud должны быть изолированы, с квотами и ограничениями egress, иначе cloud‑часть SaaS становится security‑риском.

- Sandbox для research jobs (контейнер/VM), quotas CPU/RAM/time.
- Egress allowlist + запрет майнинга/сканирования/ботнета (abuse detection).
- Tenant isolation на уровне исполнения job.

Done:
- Cloud execution для research jobs безопасно изолирован и не может быть использован как “compute abuse”.

Фаза 11. Полное обновление документации + legal/marketing guardrails (обязательная фаза, 2–3 недели)

1) Продуктовые и технические доки под CCEA:
- README.md: “cloud research, agent execution, keys local, no cloud orders”.
- ARCHITECTURE.md: диаграмма Cloud/Agent/Shared, протокол, state machines, config layering, threat model.
- docs/CCEA_OVERVIEW.md: boundary, threat model, legal posture, product modes.
- docs/cloud/*: control plane API, builder/registry, governance/privacy/residency, research job isolation.
- docs/agent/*: install, local vault, approvals, policies/hard caps, degraded modes, recovery runbooks.
- docs/schemas/*: manifest + protocol JSON schemas (версионирование).
- runbooks: incident/kill‑switch, recovery, safe‑degraded, revoke/rotation procedures.

2) Legal/marketing (Design Doc 17–18):
- Обновить/согласовать Terms of Service, Privacy Policy, AUP, IP clauses: “не investment advice”, “мы не брокер/не кастодиан/не исполняем и не передаём ордера”, “исполнение — в среде пользователя”.
- UI/онбординг guardrails: тексты и дисклеймеры, чтобы продукт не выглядел как advice/execution.

3) Doc audit:
- Обновить или архивировать устаревшие части.

Done:
- Документация, ToS/AUP/позиционирование и реальный код согласованы с Design Doc; CI‑проверки доков зелёные.
