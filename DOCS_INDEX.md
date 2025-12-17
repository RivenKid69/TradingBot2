# AI-Powered Quantitative Research Platform - Documentation Index

> **Navigation Hub** для всей документации проекта
>
> **Canonical Reference:** [Design_Doc_CCEA_Cloud.txt](docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt)

---

## 🚀 Start Here

### For New Users
| Document | Description |
|----------|-------------|
| [PRODUCT_OVERVIEW.md](PRODUCT_OVERVIEW.md) | ⭐ **High-level product overview** - capabilities, architecture, competitive advantages |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | ⭐ **Step-by-step guide** - from zero to first backtest in 30 minutes |
| [QUICK_START.md](QUICK_START.md) | Quick reference for common commands |

### For Investors & Startup Visa Applications
| Document | Description |
|----------|-------------|
| [docs/PRODUCT_OVERVIEW.md](docs/PRODUCT_OVERVIEW.md) | ⭐ **One-pager** - key innovations, competitive differentiation |
| [docs/INNOVATION_STATEMENT.md](docs/INNOVATION_STATEMENT.md) | ⭐ **Full innovation documentation** - technical depth, IP, academic references |
| [docs/INVESTOR_BRIEF.md](docs/INVESTOR_BRIEF.md) | **Investment highlights** - market opportunity, metrics, roadmap |

### For Developers
| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | ⭐ **Master technical reference** - complete API documentation (RU) |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture and design patterns |
| [docs/AI_GUIDE.md](docs/AI_GUIDE.md) | AI agent context and instructions (EN) |

---

## 🏗️ CCEA Cloud Architecture (Canonical)

**Reference:** [Design_Doc_CCEA_Cloud.txt](docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt) - единственный источник истины для архитектуры CCEA.

### Core Architecture
| Document | Description |
|----------|-------------|
| [CCEA Overview](docs/architecture/CCEA_OVERVIEW.md) | ⭐ **Обзор архитектуры** - Cloud/Agent разделение, зоны безопасности |
| [CCEA Protocol](docs/architecture/CCEA_PROTOCOL.md) | Протокол взаимодействия Cloud ↔ Agent |
| [CCEA Data Model](docs/architecture/CCEA_DATA_MODEL.md) | Модели данных и сущности |
| [CCEA State Machine](docs/architecture/CCEA_STATE_MACHINE.md) | Состояния Deployment и Run |

### Security & Privacy
| Document | Description |
|----------|-------------|
| [CCEA Privacy](docs/architecture/CCEA_PRIVACY.md) | Приватность, GDPR, уровни телеметрии |
| [CCEA CI Guardrails](docs/architecture/CCEA_CI_GUARDRAILS.md) | CI проверки безопасности границы |

### Operations & Business
| Document | Description |
|----------|-------------|
| [CCEA Rollout Plan](docs/architecture/CCEA_ROLLOUT_PLAN.md) | Фазы внедрения и Open Questions |
| [Marketing Guidelines](docs/business/CCEA_MARKETING_GUIDELINES.md) | Правила маркетинговых коммуникаций |
| [ToS Guidelines](docs/business/CCEA_TERMS_OF_SERVICE_GUIDELINES.md) | Руководство по Terms of Service |

### Design Documents
| Document | Description |
|----------|-------------|
| [Target Architecture](docs/design/CCEA_CLOUD/TARGET_CCEA_ARCHITECTURE.md) | Целевая архитектура с диаграммами |
| [Cloud README](docs/cloud/README.md) | Cloud компоненты |

**Ключевые принципы CCEA:**
- **Cloud** = Research, Training, Backtest, Artifact Management (no trading)
- **Agent** = Execution, Risk, Vault, Broker Integration (user-controlled)
- **Граница** = Только lifecycle commands, никаких orders/intents/secrets

---

## 📊 Статус проекта (2025-12-17)

**Production Ready** - Все критические исправления применены и протестированы. **14,000+ автоматизированных тестов.**

| Компонент | Статус | Версия |
|-----------|--------|--------|
| AdaptiveUPGD Optimizer | ✅ Production | -- |
| Twin Critics + VF Clipping | ✅ Production | -- |
| VGS | ✅ Production | v3.2 |
| PBT | ✅ Production | -- |
| SA-PPO | ✅ Production | -- |
| Data Leakage Prevention | ✅ Production | -- |
| **Multi-Asset (Stocks)** | ✅ Production | Phase 3 |
| **Execution Providers** | ✅ Production | Phase 4 (L2) |
| **Live Trading Improvements** | ✅ Production | Phase 9 |
| **MiFID II Compliance** | ✅ Toolkit Ready | 7/7 tools implemented |
| **EU AI Act Compliance** | ✅ Toolkit Ready | 4/4 phases complete |
| **DORA Compliance** | ✅ Toolkit Ready | 5/5 phases complete |
| **GDPR Compliance** | ✅ Toolkit Ready | 9/9 phases complete |

> **Note on Compliance Status**: "Toolkit Ready" means we have implemented all necessary tools and controls designed to help clients align with regulatory requirements. These implementations have not undergone independent third-party audit or certification. Clients seeking regulatory compliance should conduct their own compliance assessment or engage qualified legal/compliance advisors.

**⚠️ Переобучите модели**, если они обучены до 2025-11-26.

---

## 📚 Основная документация

### Ключевые файлы (корень проекта)

| Файл | Описание |
|------|----------|
| [CLAUDE.md](CLAUDE.md) | ⭐ **Master reference** - полная документация (RU) |
| [docs/AI_GUIDE.md](docs/AI_GUIDE.md) | 🤖 **AI Agent Guide** - context & instructions (EN) |
| [README.md](README.md) | Обзор, установка, quick start и runbooks (sim/live/debug/release) |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Архитектура системы |
| [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) | Инструкции по сборке |
| [QUICK_START_REFERENCE.md](QUICK_START_REFERENCE.md) | Быстрый старт |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Руководство по контрибуции |
| [CHANGELOG.md](CHANGELOG.md) | История изменений |

---

## 📖 Техническая документация (docs/)

### Компоненты и функции

| Файл | Описание |
|------|----------|
| [docs/pipeline.md](docs/pipeline.md) | Decision pipeline architecture |
| [docs/bar_execution.md](docs/bar_execution.md) | Bar execution mode |
| [docs/eval.md](docs/eval.md) | Model evaluation framework |
| [docs/parallel.md](docs/parallel.md) | Parallel environments |
| [docs/data_degradation.md](docs/data_degradation.md) | Data degradation simulation |
| [docs/permissions.md](docs/permissions.md) | Role-based access control |
| [docs/no_trade.md](docs/no_trade.md) | No-trade windows |
| [docs/universe.md](docs/universe.md) | Trading universe management |

### ML и оптимизаторы

| Файл | Описание |
|------|----------|
| [docs/UPGD_INTEGRATION.md](docs/UPGD_INTEGRATION.md) | ⭐ UPGD optimizer integration |
| [docs/twin_critics.md](docs/twin_critics.md) | ⭐ Twin critics architecture |

### Multi-Asset Support (Phase 2-4, 9)

| Файл | Описание |
|------|----------|
| [CLAUDE.md#multi-exchange-support](CLAUDE.md) | ⭐ Multi-exchange adapters (Binance, Alpaca, Polygon) |
| [CLAUDE.md#stock-training-backtest](CLAUDE.md) | Stock training & backtest pipeline |
| [CLAUDE.md#execution-providers](CLAUDE.md) | Execution providers (L2 simulation) |
| [CLAUDE.md#live-trading-improvements](CLAUDE.md) | Live trading improvements (Phase 9) |

**Supported Assets:**
- **Crypto**: Binance Spot/Futures (24/7)
- **Stocks**: Alpaca/Polygon US Equities (market hours + extended)
- **ETFs**: SPY, QQQ, IWM, GLD, IAU, SGOL, SLV

### Seasonality Framework

| Файл | Описание |
|------|----------|
| [docs/seasonality.md](docs/seasonality.md) | Framework overview |
| [docs/seasonality_quickstart.md](docs/seasonality_quickstart.md) | Quick start guide |
| [docs/seasonality_api.md](docs/seasonality_api.md) | API reference |
| [docs/seasonality_example.md](docs/seasonality_example.md) | Usage examples |
| [docs/seasonality_checklist.md](docs/seasonality_checklist.md) | Deployment checklist |
| [docs/seasonality_QA.md](docs/seasonality_QA.md) | QA process |
| [docs/seasonality_data_policy.md](docs/seasonality_data_policy.md) | Data policy |
| [docs/seasonality_migration.md](docs/seasonality_migration.md) | Migration guide |
| [docs/seasonality_process.md](docs/seasonality_process.md) | Development process |
| [docs/seasonality_signoff.md](docs/seasonality_signoff.md) | Sign-off procedure |

### MiFID II Compliance (Toolkit Implementation Complete)

| Файл | Описание |
|------|----------|
| [docs/compliance/MIFID_II_COMPLIANCE_ROADMAP.md](docs/compliance/MIFID_II_COMPLIANCE_ROADMAP.md) | ⭐ **Master compliance reference** - все 7 фаз |
| [services/compliance/](services/compliance/) | 30+ модулей compliance |

**Реализованные фазы:**
- Phase 1: LEI, Clock Sync (RTS 25), Algorithm Registration
- Phase 2: Transaction Reporting (RTS 22)
- Phase 3: Kill Switch, Pre-Trade Controls, Real-Time Monitoring
- Phase 4: Audit Trail, Record Keeping (5-7 years retention)
- Phase 5: Best Execution, TCA, Venue Analysis
- Phase 6: Governance, Self-Assessment, BCP
- Phase 7: Conformance Testing, Certification, NCA Notification

### EU AI Act Compliance (Toolkit Implementation Complete)

| Файл | Описание |
|------|----------|
| [docs/compliance/EU_AI_ACT_INTEGRATION_PLAN.md](docs/compliance/EU_AI_ACT_INTEGRATION_PLAN.md) | ⭐ **Master AI Act reference** - все 4 фазы |
| [docs/compliance/EU_DECLARATION_OF_CONFORMITY.md](docs/compliance/EU_DECLARATION_OF_CONFORMITY.md) | EU Declaration of Conformity (Article 47) |
| [docs/compliance/INSTRUCTIONS_FOR_USE.md](docs/compliance/INSTRUCTIONS_FOR_USE.md) | Instructions for Use (Article 13) |
| [services/ai_act/](services/ai_act/) | 15+ модулей AI Act compliance |

**Completion Reports:**
| Фаза | Отчёт | Тесты |
|------|-------|-------|
| Phase 1 | [EU_AI_ACT_PHASE1_COMPLETION_REPORT.md](docs/compliance/EU_AI_ACT_PHASE1_COMPLETION_REPORT.md) | 372 |
| Phase 2 | [EU_AI_ACT_PHASE2_COMPLETION_REPORT.md](docs/compliance/EU_AI_ACT_PHASE2_COMPLETION_REPORT.md) | 236 |
| Phase 3 | [EU_AI_ACT_PHASE3_COMPLETION_REPORT.md](docs/compliance/EU_AI_ACT_PHASE3_COMPLETION_REPORT.md) | 318 |
| Phase 4 | [EU_AI_ACT_PHASE4_COMPLETION_REPORT.md](docs/compliance/EU_AI_ACT_PHASE4_COMPLETION_REPORT.md) | 81 |

**Total AI Act Tests: 1,007 passed**

### DORA Compliance (Toolkit Implementation Complete)

| Файл | Описание |
|------|----------|
| [docs/compliance/DORA_INTEGRATION_PLAN.md](docs/compliance/DORA_INTEGRATION_PLAN.md) | ⭐ **Master DORA reference** - все 5 фаз |
| [docs/compliance/dora/proportionality_assessment.md](docs/compliance/dora/proportionality_assessment.md) | Proportionality assessment (Phase 0) |
| [services/dora/](services/dora/) | 20+ модулей DORA compliance |

**Реализованные фазы:**
- Phase 0: Proportionality Assessment & Scope Verification
- Phase 1: ICT Risk Management Framework (Articles 5-16)
- Phase 2: ICT Incident Management & Reporting (Articles 17-23)
- Phase 3: Digital Resilience Testing / TLPT (Articles 24-27)
- Phase 4: Third-Party ICT Risk Management (Articles 28-44)
- Phase 5: Information Sharing, Compliance Dashboard & Unified Reporting

**Total DORA Tests: ~1,015 passed**

### GDPR Compliance (Toolkit Implementation Complete)

| Файл | Описание |
|------|----------|
| [docs/compliance/GDPR_COMPLIANCE_SUMMARY.md](docs/compliance/GDPR_COMPLIANCE_SUMMARY.md) | ⭐ **GDPR Summary** - обзор статуса compliance |
| [docs/compliance/GDPR_CCEA_IMPLEMENTATION_PLAN.md](docs/compliance/GDPR_CCEA_IMPLEMENTATION_PLAN.md) | ⭐ **Master GDPR reference** - все 9 фаз |
| [docs/legal/PRIVACY_POLICY.md](docs/legal/PRIVACY_POLICY.md) | Privacy Policy (v3.0.0) |
| [docs/legal/TERMS_OF_SERVICE.md](docs/legal/TERMS_OF_SERVICE.md) | Terms of Service (v3.0.0) |
| [docs/legal/DPA_TEMPLATE.md](docs/legal/DPA_TEMPLATE.md) | DPA Template (v2.0.0) |
| [packages/cloud/governance/](packages/cloud/governance/) | 15+ модулей GDPR governance |

**Реализованные фазы (CCEA-aligned):**
- Phase 0: Data Mapping, RoPA, Controller/Processor Roles
- Phase 1: Transparency, Privacy Policy, DPA, DSAR SOP
- Phase 2: Data Minimization, Telemetry Contracts, CI Guardrails
- Phase 3: EU-Only Residency Enforcement
- Phase 4: Retention Policies, Auto-Purge, Legal Holds
- Phase 5: DSAR Workflows (Access, Portability, Erasure)
- Phase 6: RBAC, Access Audit, Break-Glass Procedures
- Phase 7: Security Controls (Art. 32), Breach Workflow (Art. 33-34)
- Phase 8: Continuous Compliance, Privacy-by-Design CI Checks
- Phase 9: Enterprise/On-Prem/VPC Posture

**GDPR Key Documents:**

| Документ | Описание |
|----------|----------|
| [GDPR_RISK_SCOPE_MEMO.md](docs/compliance/GDPR_RISK_SCOPE_MEMO.md) | Scope, data map, roles |
| [DSAR_SOP.md](docs/compliance/DSAR_SOP.md) | DSAR procedures |
| [SUBPROCESSORS_REGISTER.md](docs/compliance/SUBPROCESSORS_REGISTER.md) | EU subprocessors list |
| [BREACH_RESPONSE_SOP.md](docs/compliance/BREACH_RESPONSE_SOP.md) | Breach notification workflow |
| [RETENTION_POLICY_SPEC.md](docs/compliance/RETENTION_POLICY_SPEC.md) | Retention policies |
| [CCEA_PRIVACY_GUARANTEES_CHECKLIST.md](docs/compliance/CCEA_PRIVACY_GUARANTEES_CHECKLIST.md) | Privacy guarantees |

---

## 🗄️ Архив документации

**Все исторические отчёты перемещены в `docs/archive/`**

### Структура архива

```
docs/archive/
├── reports_2025_11_27/           # Отчёты 27 ноября (EV analysis, Signal-Only)
├── reports_2025_11_25_cleanup/   # Основные архивированные отчёты
│   ├── root_reports/             # Критические исправления
│   ├── reports/
│   │   ├── analysis/             # Аналитические отчёты
│   │   ├── audits/               # Аудиты
│   │   ├── bugs/                 # Отчёты о багах
│   │   ├── features/             # Feature mappings
│   │   ├── fixes/                # Отчёты об исправлениях
│   │   ├── integration/          # Интеграционные отчёты
│   │   ├── self_review/          # Self-review отчёты
│   │   ├── summaries/            # Сводки
│   │   ├── tests/                # Тестовые отчёты
│   │   ├── twin_critics/         # Twin Critics отчёты
│   │   └── upgd_vgs/             # UPGD/VGS отчёты
│   └── ...
├── reports_2025_11/              # Отчёты ноябрь 2025
├── reports_2025_11_24/           # Отчёты 24 ноября
├── verification_2025_11/         # Верификация ноябрь
├── audits/                       # Исторические аудиты
├── twin_critics/                 # Twin Critics история
├── pbt/                          # PBT история
└── ...
```

### Ключевые архивные отчёты

Критические исправления (см. `docs/archive/reports_2025_11_25_cleanup/root_reports/`):

| Отчёт | Дата | Тема |
|-------|------|------|
| DATA_LEAKAGE_FIX_REPORT_2025_11_23.md | 2025-11-23 | Data leakage prevention |
| SA_PPO_BUG_FIXES_REPORT_2025_11_23.md | 2025-11-23 | SA-PPO fixes |
| GAE_OVERFLOW_PROTECTION_FIX_REPORT.md | 2025-11-23 | GAE overflow protection |
| TWIN_CRITICS_GAE_FIX_REPORT.md | 2025-11-21 | Twin Critics GAE |
| CRITICAL_LSTM_RESET_FIX_REPORT.md | 2025-11-21 | LSTM state reset |
| UPGD_NEGATIVE_UTILITY_FIX_REPORT.md | 2025-11-21 | UPGD negative utility |
| CRITICAL_FIXES_COMPLETE_REPORT.md | 2025-11-21 | Action space fixes |
| CRITICAL_FIXES_5_REPORT.md | 2025-11-20 | Numerical stability |
| CRITICAL_FIXES_REPORT.md | 2025-11-20 | Feature engineering |

---

## 🧪 Тестирование

### Тестовые файлы

```bash
pytest tests/                          # Все тесты
pytest tests/test_twin_critics*.py -v  # Twin Critics
pytest tests/test_upgd*.py -v          # UPGD
pytest tests/test_pbt*.py -v           # PBT
pytest tests/test_data_leakage*.py -v  # Data Leakage
```

### Статистика тестов

| Категория | Тесты |
|-----------|-------|
| **Всего тестов** | **14,000+** |
| MiFID II Compliance | 200+ |
| EU AI Act Compliance | 1,007 |
| DORA Compliance | ~1,015 |
| Twin Critics | 49+ |
| UPGD | 119+ |
| VGS | 7+ |
| Data Leakage | 46+ |
| SA-PPO | 16+ |
| PBT | 14+ |

---

## 🛠️ Инструменты (tools/)

### Основные утилиты

| Инструмент | Описание |
|------------|----------|
| [cleanup_project.py](tools/cleanup_project.py) | Очистка build artifacts, бэкапов, организация отчётов |
| [check_feature_parity.py](tools/check_feature_parity.py) | Проверка паритета features online/offline |
| [check_encoding.py](tools/check_encoding.py) | Проверка encoding issues (CI/CD) |
| [normalize_encoding.py](tools/normalize_encoding.py) | Нормализация Unicode → ASCII-safe |

### Детальная документация

- [tools/README.md](tools/README.md) - Полное описание всех инструментов
- [tools/README_ENCODING.md](tools/README_ENCODING.md) - Encoding tools

### Регулярная поддержка

```bash
# Еженедельная очистка проекта
python tools/cleanup_project.py --dry-run
python tools/cleanup_project.py

# Проверка encoding перед коммитом
python tools/check_encoding.py

# Проверка feature parity после изменений
python tools/check_feature_parity.py
```

---

## 📍 Навигация

| Задача | Куда смотреть |
|--------|---------------|
| Новичок в проекте | [CLAUDE.md](CLAUDE.md) |
| Архитектура | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Быстрый старт | [QUICK_START_REFERENCE.md](QUICK_START_REFERENCE.md) |
| Twin Critics | [docs/twin_critics.md](docs/twin_critics.md) |
| UPGD Optimizer | [docs/UPGD_INTEGRATION.md](docs/UPGD_INTEGRATION.md) |
| Multi-Asset (Stocks) | [CLAUDE.md](CLAUDE.md) (см. Phase 2-4, 9) |
| Live Trading | [CLAUDE.md](CLAUDE.md) (см. Phase 9) |
| Seasonality | [docs/seasonality.md](docs/seasonality.md) |
| MiFID II Compliance | [docs/compliance/MIFID_II_COMPLIANCE_ROADMAP.md](docs/compliance/MIFID_II_COMPLIANCE_ROADMAP.md) |
| EU AI Act Compliance | [docs/compliance/EU_AI_ACT_INTEGRATION_PLAN.md](docs/compliance/EU_AI_ACT_INTEGRATION_PLAN.md) |
| DORA Compliance | [docs/compliance/DORA_INTEGRATION_PLAN.md](docs/compliance/DORA_INTEGRATION_PLAN.md) |
| GDPR Compliance | [docs/compliance/GDPR_COMPLIANCE_SUMMARY.md](docs/compliance/GDPR_COMPLIANCE_SUMMARY.md) |
| Privacy & Legal | [docs/legal/PRIVACY_POLICY.md](docs/legal/PRIVACY_POLICY.md) |
| Исторические отчёты | `docs/archive/` |

---

**Last Updated**: 2025-12-17 (MiFID II Compliance-Ready, EU AI Act Compliance-Ready, DORA Compliance-Ready, GDPR Compliance-Ready, CCEA Architecture Implemented, 14,000+ tests)
**Status**: ✅ Designed for Production
**Version**: 7.1 (GDPR Implementation Complete)
