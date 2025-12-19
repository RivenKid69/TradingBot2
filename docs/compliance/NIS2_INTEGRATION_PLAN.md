# NIS2 Integration Plan
# Directive (EU) 2022/2555 on measures for a high common level of cybersecurity
# План интеграции NIS2 в CustodiaCloud

**Версия документа**: 1.0.0  
**Дата создания**: 2025-12-08  
**Статус проекта**: NOT STARTED (подготовительный этап)  
**Область**: Все сервисы и инфраструктура платформы, включая AI/ML пайплайны, live execution, data ingestion, backtesting, облачный периметр и поставщиков данных/трейдинга  
**Юрисдикция**: ЕС (транспонирование NIS2 в национальное право до 17 октября 2024)  

---

## Research (нормативка, best practices, архитектурный контекст)

- **NIS2 Articles**  
  - **Article 20**: ответственность management body за утверждение и надзор за киберрисками.  
  - **Article 21**: обязательные меры управления киберрисками (a–j: политика безопасности, incident handling, BCP/DR, supply chain security, secure SDLC & vulnerability handling, тестирование эффективности, cyber hygiene & обучение, криптография, HR/asset/access control, MFA/secure comms).  
  - **Article 23**: раннее предупреждение ≤24h, инцидентное уведомление ≤72h, финальный отчет ≤1 месяц (управляем через NCA/CSIRT).  
  - **Annex I/II**: сектора и критерии essential/important entities (cloud/DSP/MSP, финансовые сервисы, цифровая инфраструктура).  
- **Точки соприкосновения с существующими регуляторками**  
  - **DORA**: incident timelines почти идентичны (24h/72h/30d) — нужно единое расписание (реиспользовать `services/dora/cross_regulation.py`).  
  - **EU AI Act**: Article 73 (24h serious incident) → единый процесс уведомлений.  
  - **GDPR**: breach notification (72h) → увязать с NIS2 Article 23.  
  - **NIST CSF 2.0**: уже описан в `docs/CYBERSECURITY_FRAMEWORK.md`; маппинг на Article 21 облегчает доказательство зрелости.  
- **Архитектурные опоры**  
  - Слоистая архитектура (`core_` → `impl_` → `service_` → `strategies` → `scripts_`) из `ARCHITECTURE.md`.  
  - Инцидентные таймлайны и кросс-регуляторный мост уже частично реализованы в `services/dora/cross_regulation.py` и `services/dora/incident_reporting.py`.  
  - Документы по кибербезопасности и восстановлению: `docs/CYBERSECURITY_FRAMEWORK.md`, `docs/RECOVERY_PROCEDURES.md`, `docs/OPERATIONS_RUNBOOK.md`.  
  - Регуляторная позиция как software vendor (см. `docs/REGULATORY_COMPLIANCE_STRATEGY.md`) — нужно подтвердить применимость NIS2 (скорее **important entity** как цифровой сервис / managed service provider).  

---

## Roadmap по фазам (оптимально закрывать по одному промпту Opus 4.5)

| Фаза | Цель | Ключевые артефакты | Тесты (targets; verify via CI) |
|------|------|--------------------|-------------------------------|
| **Phase 0: Scope & Gap** | Подтвердить применимость NIS2, выбрать NCA/CSIRT, снять gap-матрицу | Scope register, NCA/CSIRT контакты, gap-to-Article21 | 18 planned (scope logic, gap parser, classification) |
| **Phase 1: Governance & Policies** | Закрепить ответственность Article 20, обновить политики | RACI, policy pack, board training proof | 22 planned (policy lint, approvals, training evidence) |
| **Phase 2: Core Controls (Art.21 a,f,g,h,i,j)** | Базовые кибермеры: идентичность, доступ, крипто, журналирование, тестирование эффективности | `services/nis2/risk_management.py`, IAM/MFA baseline, logging schema | 40 planned (unit+integration на контрольные списки, IAM/crypto checks) |
| **Phase 3: Incident Handling & Reporting (Art.21b, Art.23)** | Единый инцидентный процесс и уведомления | `services/nis2/incident_reporting.py`, runbook, CSIRT forms | 36 planned (timeline calc, schema validation, dry-run submissions) |
| **Phase 4: Resilience & BCP/DR (Art.21c)** | Восстановление, резилиентность, failover | BCP, DR playbooks, backup/restore proofs | 28 planned (backup drills, RTO/RPO sims, chaos tests) |
| **Phase 5: Supply Chain & Vendor Security (Art.21d,e)** | Управление поставщиками/библиотеками, CVD, secure SDLC | Vendor registry, SBOM, SBoM attestations, contract clauses | 34 planned (SBOM diff, SAST/DAST gates, vendor controls) |
| **Phase 6: Awareness & Continuous Compliance (Art.21g + supervision)** | Обучение, метрики, постоянный мониторинг | Training program, KPI dashboard, audit pack | 18 planned (training completion, KPI calc, audit export) |

> **Note**: Test counts are planned targets. Actual pass rates should be verified via CI. "Planned" indicates roadmap scope, not completion status.

---

# Phase 0: Scope & Gap Analysis

**Цель**: формально подтвердить применимость NIS2, выбрать статус (essential vs important), определить NCA/CSIRT и получить полную gap-матрицу по Article 20/21/23.

**Шаги**:
- Классификация по Annex I/II: проверить, подпадает ли SaaS-провайдер под **digital provider / managed service provider** → предварительно принять **important entity**; оформить обоснование (маркет, оборот, размер, количество клиентов в ЕС).  
- Определить **country of establishment** и соответствующий **NCA + CSIRT** (по месту основного офиса/сервисного хаба).  
- Завести **Scope Register** (entity type, sector, jurisdiction, exemptions) в `docs/compliance/technical_documentation/` с ссылками на клиентские контракты и сервисные описания.  
- Построить **gap-матрицу**: Article 20, Article 21 (a–j), Article 23 vs текущие артефакты (`CYBERSECURITY_FRAMEWORK`, `RECOVERY_PROCEDURES`, `cross_regulation`).  
- Оценить интерфейсы с соседними регуляторками (GDPR 72h, DORA 24h/72h/30d, AI Act 24h) — зафиксировать в единой таблице требований.  
- Зафиксировать **assumptions/constraints** (например, SaaS-only, no asset holding, no retail users).  

**Тесты (100% покрытия фазы)**:
- Unit: классификация entity по Annex I/II (positive/negative cases).  
- Unit: парсер gap-матрицы сверяет наличие артефактов для каждого подпункта Article 21.  
- Integration: выбор NCA/CSIRT на основе страны → проверка обязательности уведомлений и каналов.  
- Evidence: scope register + sign-off от compliance lead.

---

# Phase 1: Governance & Policies (Article 20)

**Цель**: закрепить ответственность management body, обновить политики и RACI, встроить надзор за киберрисками.

**Шаги**:
- Назначить accountable executive (CISO / Head of Security) и описать **RACI** для инцидентов, уязвимостей, BCP/DR, поставщиков.  
- Обновить набор политик (Information Security, Access Control, Crypto, Vendor Mgmt, Secure SDLC, Incident Response, BCP/DR) с явными ссылками на Article 21(a–j).  
- Добавить обязательное **board-level одобрение и ежегодное обучение** по NIS2 для management body (лог аудита).  
- Обновить `docs/OPERATIONS_RUNBOOK.md` и `docs/RECOVERY_PROCEDURES.md` с ролями и escalation chain под NIS2.  
- Добавить в инженерные процессы **policy-as-code** проверки (linting в CI для YAML/MD политик, контроль сроков ревизии).  
- Подготовить **audit pack**: политика, RACI, протокол обучения, контакт NCA/CSIRT, список артефактов.  

**Тесты (100% покрытия фазы)**:
- Unit: lint/consistency чеков политики (версии, владельцы, ссылки на статьи).  
- Unit: RACI completeness (каждая функция инцидент/BCP/vendor имеет owner/delegate).  
- Integration: CI job, который падает при истекшем сроке ревизии политики.  
- Evidence: протокол обучения и approval лог в репозитории.

---

# Phase 2: Core Controls (Article 21 a,f,g,h,i,j)

**Цель**: внедрить/закрепить базовые киберконтроли и их проверяемость на уровне кода/инфры.

**Шаги**:
- **Risk & control framework**: `services/nis2/risk_management.py` с маппингом Article 21(a–j) → контрольные пункты + связка с NIST CSF разделами из `CYBERSECURITY_FRAMEWORK`.  
- **IAM & Access**: enforce MFA для всех админов/CI, запрет shared accounts, rotation secrets; RBAC для сервисов (`core_`/`service_` контракты) + inventory сервисных аккаунтов.  
- **Crypto policy**: каталог алгоритмов/ключевых длин, требования к TLS (min v1.2/1.3), storage encryption, HSM/KMS использование; фиксировать в `config/security/crypto_policy.yaml`.  
- **Logging & audit**: унифицированный лог-скhema (actor, event_type, system, regulation_source) + маршрут в SIEM; минимальный retention ≥ 12 мес для NIS2 аудита.  
- **Effectiveness testing**: периодические self-checks (control health), метрики coverage, автоматизированные проверки шифрования/логирования/МFA включены по дефолту.  
- **Human resources & asset control**: онбординг/оффбординг чек-листы, инвентаризация окружений (prod/stage/dev) и чувствительных данных.  
- **Secure comms**: требования к защищенным каналам (TLS, VPN, SSH hardening), запрет незашифрованного трафика для prod.  

**Тесты (100% покрытия фазы)**:
- Unit: контрольные точки risk registry → все Article 21 пункты имеют owner/evidence.  
- Unit: IAM policy tests (нет wildcard, MFA=ON, key age < threshold).  
- Integration: шифрование в транзите/на диске проверяется автоматическими probes в CI.  
- Integration: лог-события содержат обязательные поля и маршрутизируются в тестовый SIEM эндпоинт.  
- Security QA: проверка secure comms (TLS versions, ciphers) на тестовом стенде.  

---

# Phase 3: Incident Handling & Reporting (Article 21b, Article 23)

**Цель**: единый процесс обработки инцидентов и уведомлений в NCA/CSIRT с синхронизацией DORA/AI Act/GDPR.

**Шаги**:
- Создать модуль `services/nis2/incident_reporting.py` с таймерами 24h/72h/30d и обязательными полями (initial assessment, impact, mitigation, root cause).  
- Расширить `services/dora/cross_regulation.py` для двунаправленного маппинга NIS2 ↔ DORA ↔ AI Act ↔ GDPR (приоритет ближайшего дедлайна).  
- Обновить `docs/OPERATIONS_RUNBOOK.md` и `docs/RECOVERY_PROCEDURES.md` с формами уведомлений, контактами CSIRT, каналами (портал/шлюз/email/телефон) и критериями **significant incident** по NIS2.  
- Настроить автоматические **early-warning** триггеры из мониторинга/alerting (detector → incident SLO breach → форма 24h).  
- Добавить **post-incident review** шаблон с lessons learned и требованиями к превентивным мерам.  
- Увязать GDPR breach (72h) и AI Act serious incident (24h) через единый классификатор инцидентов.  

**Тесты (100% покрытия фазы)**:
- Unit: корректный расчет дедлайнов 24h/72h/30d при разных временах обнаружения/классификации.  
- Integration: dry-run отправка уведомлений (mock CSIRT API/формы), валидация обязательных полей.  
- Integration: alert → incident → уведомление pipeline с контрольными SLA.  
- Tabletop: симуляция major инцидента с таймингами и заполнением форм.  

---

# Phase 4: Resilience & BCP/DR (Article 21c)

**Цель**: обеспечить непрерывность услуг и восстановление с заданными RTO/RPO, подтвердить резервирование и план кризисного управления.

**Шаги**:
- Обновить **BCP/DR** (каталоги критичных функций, RTO/RPO, владельцы) в `docs/RECOVERY_PROCEDURES.md`.  
- Настроить **backup & restore** для критичных данных (маркет-дата, модели, конфиги, журналы) с геораспределением и регулярной проверкой восстановлений.  
- Реализовать **failover playbooks** для основных сервисов (market data ingest, execution, risk guard, storage).  
- Добавить **chaos/DR тесты** на отказ брокера/биржи, потерю региона облака, деградацию сети.  
- Убедиться, что **communication plan** для кризисов (внутренние/внешние) соответствует Article 21(c).  
- Встроить метрики RTO/RPO в мониторинг и ежемесячные отчеты compliance.  

**Тесты (100% покрытия фазы)**:
- Integration: восстановление из резервной копии (периодичность ≥ еженедельно) с проверкой целостности.  
- Chaos: имитация отказа основного провайдера данных/биржи → автоматический failover.  
- Tabletop: кризисное упражнение с коммуникационным планом.  
- Evidence: RTO/RPO фактические vs целевые — автоматически проверяются и фиксируются.  

---

# Phase 5: Supply Chain & Vendor Security (Article 21d,e)

**Цель**: контролировать риски поставщиков/зависимостей, обеспечить secure SDLC, управление уязвимостями и CVD.

**Шаги**:
- Создать **Vendor & Dependency Register** (облако, биржи, брокеры, дата-провайдеры, email/alerting, CI/CD, библиотеки) с оценкой критичности и контрактных мер безопасности.  
- Добавить `services/nis2/vendor_registry.py` + интеграцию с SBOM (CycloneDX) и мониторингом CVE.  
- Ввести **SAST/DAST/SCA/SBOM** пайплайны в CI для Python/Cython/C++ компонентов; блокировать релизы при CVSS ≥ 7 без исключений.  
- Обновить **procurement & contract clauses**: требования к шифрованию, уведомлениям об инцидентах, праву на аудит, срокам исправлений.  
- Организовать **Coordinated Vulnerability Disclosure**: публичная политика, security.txt, внутренний triage SLA, канал связи.  
- Внедрить **secure SDLC** чек-листы (threat modeling для новых сервисов, обязательные код-ревью с безопасностью, секреты вне кода).  

**Тесты (100% покрытия фазы)**:
- Unit: валидация реестра поставщиков (обязательные поля, уровни критичности, сроки ревизии).  
- Integration: SBOM генерация на билд-артефакт, сравнение diff, оповещение при новых CVE.  
- Integration: SAST/DAST/SCA гейты в CI (fail on critical).  
- Dry-run: CVD процесс (получение отчета → triage → фиксация SLA).  
- Evidence: контрактные требования проверены чек-листом.  

---

# Phase 6: Awareness & Continuous Compliance (Article 21g + supervision readiness)

**Цель**: обучить персонал, внедрить постоянный мониторинг метрик и подготовить материалы для проверок NCA.

**Шаги**:
- Запустить **training program**: onboarding + ежегодное обучение по NIS2, фишинг-симуляции, secure coding для инженеров.  
- Определить **KPI/OKR**: coverage Article 21 controls, среднее время реакции на инцидент, процент закрытых уязвимостей в SLA, MFA coverage.  
- Построить **compliance dashboard** (автоэкспорт из CI/SIEM/SBOM) и хранить снапшоты для аудита.  
- Настроить **continuous control monitoring** (CCM): периодические автоматические проверки политик, IAM, шифрования, логирования, RTO/RPO.  
- Подготовить **audit-ready package**: доказательства выполнения фаз, реестр тестов, результаты дригов, экспорт логов по запросу NCA.  
- Планировать **ежеквартальные внутренние аудиты** + **годовую внешнюю проверку**.  

**Тесты (100% покрытия фазы)**:
- Unit: метрики собираются и корректно считаются (например, % MFA-enabled).  
- Integration: дашборд обновляется из источников (CI, SIEM, SBOM) без ручных шагов.  
- Evidence: training completion records required (target: all personnel; verify via LMS/HR records); phishing simulation results and retest records required.  
- Audit rehearsal: mock запрос от NCA → формируется пакет артефактов.  

---

## Alignment с существующей архитектурой и кодовой базой

- Реиспользовать слой `services/dora/*` для инцидентных таймлайнов и unified reporting; расширить его для NIS2 (Phase 3).  
- Встраивать новые проверки в существующие CI jobs (см. `run_*_tests.sh`, `Makefile`) без ломки пайплайнов train/eval/backtest.  
- Использовать текущие конфигурации (`configs/*.yaml`, `core_config.py`) для внедрения параметров безопасности (MFA flags, logging targets, crypto policy).  
- Сервисы ingest/execution должны получать failover-политики из `config` и иметь тестируемые сценарии деградации (Phase 4).  
- SBOM/SAST/DAST должны покрывать Python, Cython и C++ части (`*.pyx`, `*.cpp`, `*.h`).  

---

## Deliverables & Definition of Done (DoD)

- По итогам каждой фазы: обновлённые документы в `docs/compliance/` + реализованные сервисные модули (`services/nis2/*`), конфиги и CI-пайплайны.  
- **Tests**: All phase tests executed and documented (verify via CI artifacts: tabletop/chaos protocols, CI reports, dry-run notification logs). Test completion status subject to current CI run.  
- **Evidence**: ссылки на политики, реестры, лог-файлы SIEM, результаты восстановления, отчёты по CVE, training completion.  
- **Ownership**: назначен владелец каждого контроля и процедура ежегодного пересмотра.  

---

## Recommended Next Step

Стартовать Phase 0: подтвердить классификацию (important entity), выбрать NCA/CSIRT, собрать gap-матрицу Article 21 и забронировать окна для инцидентных и DR упражнений. После — переходить к Phase 1 по governance.
