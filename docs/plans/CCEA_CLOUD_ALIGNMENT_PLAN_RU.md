План приведения проекта к соответствию Design Doc CCEA Cloud

Примечание: формулировки про “полное соответствие” в этом плане описывают целевое состояние и критерии готовности; это не заявление о независимой сертификации, аудите или юридически подтверждённой комплаенс‑оценке.

Ниже — фазы с конкретными задачами и “done‑критериями”. Логика опирается на best‑practice CCEA‑модель (cloud research + signed artifacts + outbound control plane + local live execution) и индустриальные стандарты supply‑chain/telemetry/privacy (OCI+digest, Sigstore/cosign, CycloneDX/SPDX SBOM, SLSA provenance, OpenTelemetry, GDPR/DORA).

Фаза 0. Инвентаризация и целевая архитектура (1–2 недели)

Составить точную карту текущих модулей по ролям:
Shared: core_*, impl_*, симуляция/бэктест/тренинг, общие модели и DI.
Live‑only (Agent): service_signal_runner.py, private‑trading части adapters/*, локальное состояние, kill‑switch enforcement.
Cloud‑only: UI/IDE/backtest/training orchestration, builder/registry/control plane (сейчас почти отсутствуют).
Зафиксировать “границу периметра” в виде 1 диаграммы (что где может жить) и списка запрещённых зависимостей для Cloud (broker submitters, private trading clients).
Определить минимальный sandbox для retail Agent (docker‑default, process‑fallback) и требования enterprise (egress allowlist, ro‑fs).
Done: утверждённый документ “Target CCEA Architecture” + таблица модулей “куда переезжает”.

Фаза 1. Жёсткое разделение Cloud/Agent/Shared (3–4 недели)

Реструктурировать репо в 3 пакета/дистрибутива (монорепо, но отдельные сборки):
packages/shared — всё, что безопасно и нужно обоим рантаймам.
packages/agent — live loop + broker connectors + локальные политики.
packages/cloud — research/backtest/training UI и оркестрация без trading‑кода.
Перенести/разрезать адаптеры на data‑only vs trading‑only:
публичные market‑data клиенты остаются в shared/cloud;
private‑trading клиенты и submitters — только в agent.
Ввести CI guardrails:
статическая проверка импортов/зависимостей cloud‑сборки (нет adapters/*private*, broker_*, *_submit_order* и т.п.);
тест “command schemas must not contain price/qty/side”.
Собрать 2 runtime‑образа: cloud-research и agent-live, плюс базовый shared wheel.
Done: cloud‑образ физически не содержит торговых клиентских библиотек; live‑код запускается только через agent‑сборку.

Фаза 2. Strategy API и сим/лайв паритет (2–3 недели)

Утвердить единый контракт стратегии = Intent (у тебя уже есть OrderIntent в core_models.py).
Cloud: Intent → SimExecutionEngine (без реальных ордеров).
Agent: Intent → local Risk Manager → Orders → Broker.
Вычистить устаревшие места, где ещё используются Decision/“готовые ордера”; обновить стратегии/сервисы на OrderIntent.
Запретить передачу live‑intent из cloud: в cloud‑API и протоколе нет “signal/intent push”.
Done: стратегии везде возвращают Intent; генерация конкретных Orders находится только в agent‑раннере.

Фаза 3. Artifact Builder: immutable + signed + manifest + SBOM (4–6 недель)

В cloud‑пакете реализовать builder пайплайн:
упаковка стратегии/моделей в OCI image (digest‑pinned) или zip‑bundle (второй вариант только если контейнер невозможен);
генерация manifest.json по схеме дока (schema_version, entrypoint, runtime, deps lock digest, model refs, permissions, risk_profile_suggested, change_class, provenance);
генерация SBOM (CycloneDX/SPDX) и сохранение ссылки в Build.
Подпись артефактов: Sigstore/cosign (keyless или с ключом) либо GPG; сохранить public key/identity в cloud registry.
Registry слой: разрешённые registry allowlist + pull только по digest.
Agent‑верификация: при запуске строго проверять digest + signature + schema_version + allowlist; unsigned/unknown → reject.
Done: без подписи артефакт не публикуется; агент не запускает артефакт без успешной верификации.

Фаза 4. Agent Daemon + Local Vault + Sandbox (4–6 недель)

Создать agentd (daemon + CLI):
установка на BYO host/VPS/on‑prem;
управление жизненным циклом runs.
Enrollment‑процесс:
agent enroll --token генерит device keypair локально, регистрирует public key в cloud, получает agent_id.
Secrets: использовать существующий CredentialVault как Local Vault, добавить CLI/UI для ввода/ротации; обеспечить redaction.
Runner:
default docker‑sandbox (ограничения CPU/RAM);
process‑mode как fallback;
enterprise‑режимы: ro‑fs, egress allowlist.
Local journal + telemetry buffer: durable очередь (sqlite/jsonl), восстановление после рестарта, fail‑safe halt при неопределённости.
Done: агент автономно держит live‑loop, хранит ключи, восстанавливается и безопасно деградирует без cloud.

Фаза 5. Cloud Control Plane и модель данных (6–8 недель)

Реализовать cloud‑сервис (FastAPI) с БД (Postgres) и сущностями из дока: Org/Workspace/User/Roles, Strategy/Version, Build/Artifact, Agent, Deployment, Run, Command, ApprovalRecord, TelemetryEvent, Alert, AccessAudit, RetentionPolicy.
Endpoint’ы:
create enrollment token (TTL),
enroll/heartbeat,
poll commands (long‑poll),
accept acks/approvals/results,
telemetry ingestion.
Control plane хранит только desired state и immutable config blobs (без секретов).
Done: есть минимально рабочий multi‑tenant control plane, к которому агент ходит outbound‑poll’ом.

Фаза 6. Протокол, state‑machines, idempotency, approvals (4–6 недель)

Зафиксировать JSON‑схемы сообщений (HEARTBEAT, POLL_COMMANDS, COMMAND_BATCH, ACK, APPROVAL, RESULT, TELEMETRY) + schema_versioning.
Реализовать идемпотентность команд и dedup по idempotency_key на agent и cloud.
Deployment/Run state machines ровно как в доке.
Safe list команд cloud→agent: только REQUEST_* lifecycle.
Локальные approvals:
классификация change_class на Build/Config/Universe/Mode;
агент показывает diff и требует approve на TRADING_IMPACTING;
auto‑approve только через локальную политику.
Done: торгово‑значимые изменения невозможны без локального approve; cloud не может отправить order‑like payload.

Фаза 7. Telemetry + Privacy/GDPR + Residency (3–5 недель)

Agent: обязательный redaction middleware (невозможность выключить), уровни чувствительности: AGGREGATED default, RAW opt‑in.
Cloud:
retention per tenant, auto‑purge, экспорт/удаление (DSAR);
RBAC на чувствительные данные, AccessAudit;
break‑glass с причиной.
Data residency: конфиг регионов (EU default для EU tenants), enterprise‑опция “telemetry stays local”.
Done: телеметрия минимизирована, классифицирована, редактирована; есть управляемое хранение и аудит доступа.

Фаза 8. Enterprise/on‑prem pack (4–6 недель)

Довести существующие services/enterprise/onprem до реального комплекта развертывания: docker‑compose/Helm для cloud‑стека, опциональный registry mirror.
Evidence pack exporter: digests/signatures/SBOM, журналы approvals/commands, halt reasons, telemetry exports.
Version pinning + change windows для Agent; min/max supported schema versions.
Done: можно развернуть execution‑контур on‑prem/VPC клиента и выгрузить audit/evidence pack.

Фаза 9. Полное обновление документации (обязательная отдельная фаза, 2–3 недели)

Переписать пользовательские и продуктовые доки под CCEA:
README.md — чёткая формула “cloud research, agent execution, keys local, no cloud orders”.
ARCHITECTURE.md — новая диаграмма Cloud/Agent/Shared, протокол, state machines.
PRODUCT_OVERVIEW.md/PROJECT_OVERVIEW.md/BUSINESS_OVERVIEW.md — позиционирование software provider, без “we trade for you”.
Добавить новые разделы:
docs/CCEA_OVERVIEW.md (boundary, threat model, legal posture).
docs/cloud/* (control plane API, builder, registry, privacy).
docs/agent/* (install, local vault, approvals, policies, degraded modes).
docs/schemas/* (manifest + protocol JSON schemas).
runbooks: incident/kill‑switch, recovery, safe‑degraded.
Провести “doc audit” старых разделов (например про Decision) и либо обновить, либо унести в docs/archive.
Done: документация полностью согласована с Design Doc и реальным кодом, CI‑проверки доков зелёные.
