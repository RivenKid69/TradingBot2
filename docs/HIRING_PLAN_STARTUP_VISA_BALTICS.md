# CustodiaCloud — План найма (Startup Visa / TRP): Латвия • Литва • Эстония

**Prepared for Baltic startup visa / temporary residence permit (TRP) applications**  
**Версия**: 1.0  
**Дата**: 2025-12-29  
**Статус проекта**: pre‑revenue, single‑founder (solo)  

**Canonical positioning / legally safe wording**: `docs/DOCUMENTATION_CANON_DESIGN.md`.

**Non‑legal notice**: этот документ — планирование и committee‑friendly описание. Это **не** юридическая/налоговая консультация. Требования программ и суммы (fees, proof‑of‑funds, salary thresholds) могут меняться; их нужно перепроверять перед подачей по официальным источникам и/или через местного иммиграционного консультанта.

---

## 0) Зачем этот документ

Мне нужен реалистичный, поэтапный план найма, который:

1) соответствует текущему состоянию CustodiaCloud (код/архитектура/стадия пилота),  
2) привязан к измеримым продуктовым вехам (pilot → revenue → scale),  
3) учитывает ожидания стартап‑комитетов (innovation + viability + economic contribution),  
4) содержит локализацию под **Latvia / Lithuania / Estonia** по официальным процессам (ссылки в конце).

Этот документ **фокусируется только на найме**; он дополняет общий комитет‑план: `docs/BUSINESS_PLAN_EU_VISA.md`.

---

## 1) Контекст проекта (на основании репозитория)

### 1.1 Что строим (committee‑friendly)

CustodiaCloud — **B2B** risk‑first quantitative **research & deployment** платформа (equities‑first go‑to‑market). Ключевой дифференциатор — архитектура **CCEA (Cloud‑Controlled Execution Architecture)**:

- **CustodiaCloud Cloud**: research / simulation / monitoring / artifact registry / lifecycle control plane; **designed not to store broker credentials** и **не отправляет live trading instructions (orders/targets/signals)**. Cloud может отправлять lifecycle‑команды и подписанные артефакты на Agent (не торговые инструкции).  
- **CustodiaCloud Agent (customer‑controlled)**: исполняется в среде клиента; локально хранит secrets; применяет risk‑limits/kill switch; взаимодействует с брокером/исполнением **под контролем клиента**.

### 1.2 Текущая стадия (важно для реалистичности найма)

По `BUSINESS_OVERVIEW.md` и `docs/INVESTOR_BRIEF.md`:

- Проект технически “глубокий” (CCEA boundary, agent/cloud packages, тесты, multi‑asset foundation).
- **Pilot program** запланирован (3 месяца, 3–5 компаний), но **ещё не запущен** и **нет подписанных участников** (internal status as of 2025‑12‑19).
- Цель ближайших 6–12 месяцев — repeatable onboarding + первые платные конверсии (не “обещания доходности”).

Следствие: план найма должен быть **бережливым** (lean), с чёткими триггерами по вехам и burn‑rate.

---

## 2) Принципы найма (best practices для solo‑founder)

### 2.1 Основные принципы

1) **Milestone‑driven hiring**: найм “под вехи”, не “под мечты”.
2) **Сначала снять bottleneck’и**: первая волна — то, что ускоряет пилот/ревенью (platform/DevOps, продуктовый full‑stack, customer onboarding).
3) **Founder‑dependency reduction**: документация/процессы/ownership, чтобы риски “single point of failure” снижались с каждым наймом.
4) **Покупать, а не строить**: бухгалтерия/пэйролл/юристы/HR‑операции — аутсорс, пока <10 FTE.
5) **Remote‑friendly, но с “local core”**: для визы/комитета важно показать вклад в экономику страны; базовый план — 2 ключевые роли локально в хост‑стране в первый год (если позволяет runway), при сохранении remote‑опций для узких задач.
6) **Bar‑raiser**: в ранней команде каждый человек должен быть “A‑player/generalist”, иначе стоимость ошибки слишком высока.

### 2.2 Формат занятости (реализм)

- **FTE (штат)**: роли, которые постоянно держат продукт (platform, core product, solutions/onboarding).
- **Part‑time / contractor**: дизайн, контент, легал, рекрутинг‑ассист, временные интеграции.
- **Advisor (equity/retainer)**: B2B fintech sales, procurement/security, EU legal/regulatory counsel (не “команда”, а support).

---

## 3) Целевая орг‑структура (первые 24 месяца)

### 3.1 Минимально жизнеспособная команда (MVT — Minimum Viable Team)

**0–12 месяцев (после релокации/одобрения):**

- Founder/CTO (я): архитектура, core ML/RL, ключевые решения по CCEA, первые пилоты, founder‑led sales.
- **Hire #1 — Platform/DevOps Engineer (локально)**: multi‑tenant cloud инфраструктура, CI/CD, observability, безопасность деплоя.
- **Hire #2 — Product Full‑Stack Engineer (локально)**: dashboard MVP, customer portal, интеграция с Cloud control plane API.
- **Hire #3 — Solutions Engineer / Quant Developer (по триггеру, локально или EU‑remote)**: customer onboarding, интеграции, runbooks, обратная связь → roadmap.
- **Contractor (0.2–0.5 FTE)**: UX/UI (короткими спринтами) + при необходимости part‑time GTM ops (CRM, outreach, контент).

**12–24 месяца (при появлении paid pilots / revenue / follow‑on funding):**

- **Hire #4 — Backend/Cloud Engineer**: масштабирование control plane, multi‑tenant data model, billing hooks (без платежей “в ядре” на ранней стадии).
- **Hire #5 — Sales Lead (Head of Sales / AE‑track)**: повторяемые продажи, enterprise pipeline, партнёрства (после сигналов PMF).

### 3.2 Почему именно эти роли (привязка к продукту)

- В репозитории уже есть “глубина” research/sim/agent‑архитектуры; узкое место ранней коммерциализации — **продуктовая упаковка + надёжный деплой + onboarding**.
- Ранний найм “чистого ML researcher” или “data team” менее приоритетен, пока не закрыт пилотный цикл и не доказан повторяемый спрос.

---

## 4) Поэтапный план найма (0–24 месяца) с вехами и триггерами

> Временные окна ниже предполагают, что заявка/релокация уже начались. В Литве и Латвии есть формальные сроки на регистрацию компании после получения TRP/Startup Visa (см. раздел 7).

### 4.1 Фаза 0 — Подготовка (до/в момент релокации, 0–6 недель)

**Цели**
- Подготовить job scorecards и процессы найма.
- Собрать sourcing‑пулы по трём рынкам (LV/LT/EE) и EU‑remote.
- Выстроить минимальные HR/ops: бухгалтер, payroll provider, шаблоны контрактов, политика ИБ/доступов.

**Выходы (deliverables)**
- 2 готовых JD: Platform/DevOps, Full‑Stack Product.
- Система интервью: 45‑мин скрининг → 90‑мин тех‑интервью → оплачиваемое тест‑задание 4–6 часов → reference check.
- План компенсаций (зарплата + опцион/phantom, если применимо) и probation период.
- Готовый пакет “security + IP + privacy” для найма: NDA/конфиденциальность, IP assignment (в договоре/контракте), базовая политика доступов, минимизация доступа к prod‑секретам.

**Go/No‑Go гейт перед Hire #1 / #2**
- Есть легальный способ оформлять людей в host‑country (или на переходный период — contractor‑формат, если допустимо локально) и выбран payroll provider.
- Есть подтверждённый runway на **9+ месяцев** на “base burn” (зарплаты + налоги работодателя + провайдеры).
- Есть пилот‑пайплайн с конкретными next steps (например, 1 подписанный пилот/LOI или 2+ активных переговорных пилота с согласованными датами/шагами).

### 4.2 Фаза 1 — “Pilot‑ready platform” (0–3 месяца)

**Веха продукта**
- Pilot‑ready MVP (только то, что нужно для пилота): Cloud control plane в production‑like окружении + Dashboard MVP (runs, artifacts, telemetry redaction, audit events).
- Коммерческая готовность: стандартизированный pilot‑agreement/SOW, baseline security ответы для due diligence (в пределах “software provider posture”).
- Спрос: **минимум 1 подписанный пилот (paid или LOI)** *или* 2 LOI/письма намерений с согласованной датой старта.

**Найм**
- **Hire #1 (Platform/DevOps, local core)** — старт после прохождения go/no‑go гейта (ориентир: неделя 4–6).
- **Hire #2 (Full‑Stack Product, local core)** — старт после Hire #1 и подтверждения пилот‑пайплайна (ориентир: неделя 8–10).

**KPI для найма**
- Time‑to‑deploy (Cloud) ≤ 1 день (repeatable)
- Time‑to‑first‑pilot‑run ≤ 2 недели после старта пилота (через customer‑controlled Agent)

### 4.3 Фаза 2 — “Pilot → First Revenue” (3–6 месяцев)

**Веха продукта**
- 1–2 paid pilots, первые invoice’ы.
- Документированные onboarding runbooks + support SLA (lightweight).

**Найм (по триггерам)**
- Если ≥2 пилота активны параллельно → **Hire #3 (Solutions Engineer)**.
- Если нагрузка на CI/CD/infra становится узким местом → расширение DevOps роли (part‑time contractor или второй инженер позже).

**Стоп‑триггеры (чтобы burn не “убежал”)**
- Если пилоты не стартуют/застревают >8–10 недель → freeze новых FTE и фокус на founder‑led пилот/онбординг.
- Если runway < 6 месяцев при текущем burn → freeze найма, пересборка scope и переговоры по funding/выручке.

### 4.4 Фаза 3 — “Repeatable onboarding” (6–12 месяцев)

**Веха продукта**
- 2+ клиентов продлили пилот/перешли на подписку (ранний PMF‑сигнал).
- Стабильность и наблюдаемость: incident response runbooks, uptime target (internal), security baseline.

**Найм**
- **Hire #4 (Backend/Cloud Engineer)** — если multi‑tenant и биллинг/энтitlements начинают тормозить roadmap.
- **Security/Compliance Ops (part‑time)** — если появляется enterprise due diligence (вместо полного FTE на ранней стадии).

### 4.5 Фаза 4 — Scale (12–24 месяца)

**Триггер для Sales Lead**
- Есть repeatable ICP + конверсия пилота в paid (например, ≥30–40% в cohort, или ≥€10k MRR — *illustrative*).

**Найм**
- **Hire #5 (Sales Lead)**: founder‑led sales → процесс → расширение pipeline.
- Далее (по росту): AE/SDR, Customer Success Manager, Data/ML роли (по потребностям клиентов).

---

## 5) План headcount и бюджета (реалистичный, с диапазонами)

> Ниже — **план‑диапазоны**, а не “обещания”. Точная стоимость найма зависит от страны (налоги/соцвзносы), уровня кандидатов и формата (FTE vs contractor).

### 5.1 Допущения по компенсациям (Baltics‑baseline)

- Senior/Strong mid инженеры (Platform/Backend/Full‑stack): **€45k–€75k gross/year**
- Solutions/Quant Dev (customer‑facing инженер): **€45k–€80k gross/year**
- Sales Lead: **€45k–€70k base + variable**, OTE зависит от рынка
- Рекрутинг/онбординг: **€1k–€4k на найм** (direct sourcing), либо 10–20% salary при агентстве (по возможности избегать на ранней стадии)
- Операционные провайдеры (accounting/payroll/legal): **€6k–€18k/year** (аутсорс)

### 5.2 Headcount‑план (консервативный “Base case”)

| Период | Команда (FTE) | Новые наймы | Комментарий |
|---|---:|---:|---|
| 0–3 мес | 3 | 2 | Platform/DevOps + Full‑stack (оба в host‑country; оформление зависит от стадии регистрации/пэйролла) |
| 3–6 мес | 3–4 | 0–1 | Solutions Engineer по триггеру “2+ активных пилота” |
| 6–12 мес | 4–5 | 1–2 | Backend/Cloud + part‑time Security/Ops |
| 12–24 мес | 6–8 | 2–3 | Sales Lead + ещё 1 инженер + CS (по revenue) |

### 5.3 Пример годового фонда оплаты труда (illustrative)

**Year 1 (после релокации):**
- 2 инженера (средний уровень): €90k–€140k
- contractors (UI/UX + ops): €10k–€30k
- итого payroll/contractors: **€100k–€170k** (+ налоги работодателя / overhead по стране)

**Year 2:**
- 4–6 FTE + variable sales: **€220k–€420k** (+ overhead)

> Для визовых комитетов важна связка “финансирование → план найма → вехи”. В `docs/BUSINESS_PLAN_EU_VISA.md` funding ask заложен как €500k–€750k (illustrative) с runway 18–24 месяца; этот headcount соответствует такому порядку.

---

## 6) Процесс найма (чтобы solo‑founder мог реально выполнить план)

### 6.1 Воронка и SLA

1) **Sourcing (неделя 1–3)**: 30–60 релевантных кандидатов/роль (inbound + outbound).
2) **Screen (48 часов)**: короткая проверка fit + мотивация + английский/коммуникация (если нужно).
3) **Tech interview (90 минут)**: реальный кейс по репозиторию/архитектуре (CCEA boundary, Python, infra).
4) **Paid take‑home (4–6 часов)**: маленький PR/issue в отдельной ветке или безопасное упражнение без доступа к секретам.
5) **Reference check (2 контакта)**.
6) **Offer + probation (3–6 месяцев по стране/контракту)**.

**Безопасность, IP и privacy в найме (минимальный стандарт)**
- Take‑home всегда **оплачиваемый**, с синтетическими данными/изолированным заданием; результат не переносится в production без отдельного рефакторинга/код‑ревью.
- Кандидатам не выдаётся доступ к prod‑секретам/аккаунтам; техническая оценка опирается на публичные/обезличенные материалы.
- Договор/контракт включает: конфиденциальность, IP assignment (в пределах применимого права), запрет на вынесение клиентских данных, и правила использования Open Source.

### 6.2 Scorecard (единый стандарт)

Для каждой роли фиксируются:
- Must‑have skills (3–5 пунктов)
- Nice‑to‑have (3–5 пунктов)
- 90‑day outcomes (измеримые)
- “Red flags” (например, игнорирование security boundaries; отсутствие ownership)

### 6.3 Онбординг 30/60/90

**30 дней**: среда, CI, базовые модули, маленькие PR, понимание CCEA boundary.  
**60 дней**: ownership одного компонента (например, control_plane deployments).  
**90 дней**: measurable KPI (снижение time‑to‑deploy, улучшение pilot onboarding).

---

## 7) Country‑specific требования, которые влияют на тайминг найма

Ниже — требования, которые напрямую влияют на *когда* можно нанимать/оформлять, и *какие* доказательства нужно держать готовыми.

### 7.1 Эстония (Startup Estonia)

**Founder eligibility (официальный гайд Startup Estonia):**
- Startup должен быть **technology‑based, innovative and scalable** с глобальным потенциалом роста.
- Нужен **approval / startup code** от **Startup Committee**; решение заявлено “within 10 working days”.
- Требование по средствам: **минимум €800 на каждый месяц пребывания** (proof of funds).
- Для подачи: страхование с покрытием **≥ €30,000** (Schengen) и государственная пошлина **€100** (D‑visa). В материалах Startup Estonia также встречается пошлина **€160** для TRP — подтверждать по типу заявления.

**Hiring‑релевантное (для эстонских компаний, “qualified startup”):**
- Для найма non‑EU сотрудников у стартапов описаны льготы: **нет иммиграционной квоты**, **нет минимального salary requirement**, упрощённая регистрация занятости (формулировки и детали зависят от статуса компании и процесса).

**Как отражаем в плане найма**
- Базовый план: первые 2 FTE нанимаются локально (EE) в первые 3 месяца (platform + product).
- Параллельно допускаем EU‑remote contractors, но core ownership остаётся в EE.

**Sources**:
- https://startupestonia.ee/startup-visa/foreign-founder  
- https://startupestonia.ee/startup

### 7.2 Литва (Startup Visa Lithuania)

**TRP и средства (официальный Startup Visa Lithuania Guidebook):**
- Подача на TRP: через **https://www.migracija.lt/** по пути “I am a start‑up”.
- Proof of subsistence: минимально **€1038/мес (2025)** ⇒ **€12,456 за 12 месяцев** (как минимум на 1 год).
- Health insurance: покрытие **≥ €30,000**, на **≥ 12 месяцев**, действует в ЕС и покрывает repatriation.
- Fees: **€160 (general)** / **€320 (urgent)** + возможный fee VFS.
- После получения TRP: **120 дней на регистрацию компании** и уведомление Startup Lithuania.

**Hiring‑релевантное (из guidebook):**
- Найм локального сотрудника: трудовой договор + уведомление SoDra.
- Для third‑country высококвалифицированных сотрудников описан упрощающий путь **Startup Employee Visa** (детали — по актуальным источникам Startup Lithuania).

**Как отражаем в плане найма**
- Если host‑country = LT: первые 2 FTE планируются как литовские трудовые договоры (после регистрации компании), а до этого — contractor/consulting.
- Тайминг найма привязывается к 120‑дневному сроку регистрации компании.

**Sources**:
- https://startupvisalithuania.com/  
- https://startupvisalithuania.com/wp-content/uploads/2025/08/Startup-Visa-Lithuania-guidebook-2025.08.pdf  
- https://www.migracija.lt/

**Important note (nationality restrictions)**:
- На сайте программы есть уведомление: приём заявлений на визы для граждан РФ/РБ за рубежом **приостановлен** (есть исключения через посредничество МИД Литвы) — проверять применимость к вашему кейсу и актуальность: https://startupvisalithuania.com/

### 7.3 Латвия (LIAA / Startup Visa)

**Startup Visa = Temporary Residence Permit (TRP)** (по LIAA Startup Guide).

Критичные для планирования найма пункты из официального гайда:
- До **5 non‑EU founders** могут получить Startup Visa под одну идею.
- Срок до **3 лет**, но TRP‑карта продлевается ежегодно.
- Proof of subsistence в гайде фигурирует как **€5,160** (годовой объём средств для основного заявителя; в гайде также приведён пример для ребёнка). Конкретный способ подтверждения и актуальность суммы нужно перепроверять на момент подачи.
- После получения Startup Visa есть **3 месяца**, чтобы зарегистрировать стартап и стать board member новой компании.
- Для продления после первого года: проверка статуса стартапа подтверждается либо **qualifying investment**, либо **progress report**. В гайде приведены ориентиры qualifying investment: **€30,000** (venture capital / AIF manager) или **€15,000** (accelerator / business angel) — при условии соответствия Latvian Startup Law.
- В гайде также указано ограничение: стартап — единственное место занятости и нельзя быть board member в других компаниях (для держателя Startup Visa).

**Как отражаем в плане найма**
- Если host‑country = LV: план найма делается “3‑летним” по форме, потому что это прямо запрашивается в латвийских материалах (planned activities, investments, cost structure).
- На 12‑м месяце планируется checkpoint: подготовка progress report и/или привлечение qualifying investment (это влияет на устойчивость найма во 2‑й год).

**Sources**:
- https://startuplatvia.eu/startup-visa/  
- https://investinlatvia.org/assets/upload/liaa-belarus/1101/LIAA_Startup_Guide-c_re_compressed.pdf

---

## 8) Приложение A — Карточки ролей (первые наймы)

### A1) Platform/DevOps Engineer (Hire #1)

**Mission**: сделать деплой Cloud‑компонентов repeatable, secure, observable.

**90‑day outcomes**
- CI/CD для cloud packages + infra‑as‑code baseline
- Monitoring/alerting/runbooks (минимальный набор)
- “Time‑to‑deploy” ≤ 1 день (repeatable)

**Must‑have**
- Kubernetes/Docker, CI/CD (GitHub Actions и аналоги)
- Observability (logs/metrics/tracing)
- Security hygiene (least privilege, secrets management, supply chain)

### A2) Product Full‑Stack Engineer (Hire #2)

**Mission**: собрать pilot‑dashboard, чтобы пилоты проходили onboarding без ручной боли.

**90‑day outcomes**
- Dashboard MVP: runs/artifacts/telemetry (redacted)/audit events
- UX “happy path” для 3–5 пилотных компаний

**Must‑have**
- Web app development (frontend + backend integration)
- Product thinking: ship small, iterate fast

### A3) Solutions Engineer / Quant Developer (Hire #3)

**Mission**: ускорить onboarding клиентов и превратить feedback в roadmap.

**90‑day outcomes**
- 2–3 клиента onboarded with documented runbooks
- Библиотека конфигов/шаблонов и “golden path”

**Must‑have**
- Сильный Python + умение читать сложный код
- Customer‑facing коммуникация (B2B)

### A4) Sales Lead (Hire #5; после триггеров PMF)

**Mission**: превратить founder‑led продажи в повторяемый процесс.

**90‑day outcomes**
- ICP + messaging + pipeline в CRM
- 10–20 квалифицированных встреч/месяц (illustrative)

**Must‑have**
- B2B enterprise/SMB sales (fintech/infra желательно)
- Умение продавать “compliance/procurement friendly” value, не “доходность”

---

## 9) Приложение B — Как использовать документ для подачи

1) Выберите страну‑хост (LV/LT/EE) и оставьте в финальной версии только соответствующий раздел 7.x.  
2) Уточните суммы (proof‑of‑funds, fees) на дату подачи по официальным источникам.  
3) Приложите этот документ как “Hiring Plan / Job Creation Plan” к основному бизнес‑плану (`docs/BUSINESS_PLAN_EU_VISA.md`) и питч‑деку.
