# DORA Operational Resilience Plan

**Version**: 1.0
**Date**: 2025-12-09
**Status**: Architecture Review & Roadmap

---

## 1. Executive Summary: Repository Analysis

### 1.1 Discovered Components

| Area | Files/Modules | Maturity |
|------|---------------|----------|
| **DORA Services** | `services/dora/` - 40+ modules | HIGH (over-engineered) |
| **DORA Configs** | `configs/dora/`, `config/dora/` | MEDIUM |
| **DORA Tests** | `tests/dora/` - 12+ test files | MEDIUM |
| **Operations Runbook** | [OPERATIONS_RUNBOOK.md](docs/OPERATIONS_RUNBOOK.md) | HIGH |
| **Recovery Procedures** | [RECOVERY_PROCEDURES.md](docs/RECOVERY_PROCEDURES.md) | HIGH |
| **Service Dependency Map** | [SERVICE_DEPENDENCY_MAP.md](docs/SERVICE_DEPENDENCY_MAP.md) | HIGH |
| **Cybersecurity Framework** | [CYBERSECURITY_FRAMEWORK.md](docs/CYBERSECURITY_FRAMEWORK.md) (NIST CSF 2.0) | HIGH |
| **SOC2 Roadmap** | [SOC2_ROADMAP.md](docs/SOC2_ROADMAP.md) | HIGH |
| **Healthcheck** | [services/healthcheck.py](services/healthcheck.py) | MEDIUM |
| **Kill Switch** | [services/ops_kill_switch.py](services/ops_kill_switch.py) | HIGH |
| **Secure Logging** | [services/secure_logging.py](services/secure_logging.py) | MEDIUM |
| **Monitoring** | [services/monitoring.py](services/monitoring.py) | MEDIUM |
| **MiFID II Compliance** | `services/compliance/` - BCP, audit, reporting | HIGH |
| **CI/CD** | [.github/workflows/build-and-test.yml](.github/workflows/build-and-test.yml) | MEDIUM |

### 1.2 Key Findings

**Well Implemented:**
- Kill switch с graceful degradation
- Recovery procedures (10 сценариев)
- Service dependency mapping с failure domains
- Cybersecurity framework (NIST CSF 2.0 Tier 3)
- SOC2 roadmap с детальным планом
- MiFID II compliance modules (BCP, audit trail)

**Over-Engineered (CRITICAL):**
- `services/dora/` содержит полную имплементацию DORA для financial entities
- 40+ модулей реализуют все 5 фаз DORA как будто платформа сама подпадает под регулирование
- Modules like `ctpp_oversight.py`, `register_of_information.py` - НЕ релевантны для ICT provider

**Missing/Weak:**
- Нет чёткого разделения Core vs Enterprise
- Нет формализованных SLA для B2B клиентов
- Backup automation и testing не документированы
- Chaos testing отсутствует
- Incident reporting для клиентов не формализован

---

## 2. Target DORA Posture & Assumptions

### 2.1 Product Position

```
┌─────────────────────────────────────────────────────────────────┐
│                    OUR POSITION IN DORA ECOSYSTEM               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  DORA REGULATED ENTITIES          │  ICT THIRD-PARTY PROVIDERS  │
│  (Article 2(1)(a-t))              │  (Article 2(1)(u))          │
│                                   │                              │
│  Banks, Investment Firms,         │  ┌─────────────────────────┐│
│  Crypto Providers, etc.           │  │   WE ARE HERE           ││
│                                   │  │   (SaaS Platform)       ││
│  ┌──────────────────────────────┐ │  │                         ││
│  │ Our B2B Clients (regulated)  │ │  │ - Not a financial entity││
│  │                              │◄┼──┤ - ICT service provider  ││
│  │ They must comply with DORA   │ │  │ - Support client DORA   ││
│  │ They use our platform        │ │  └─────────────────────────┘│
│  └──────────────────────────────┘ │                              │
│                                   │                              │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Explicit Scope Boundaries

**WE ARE:**
- SaaS platform / ICT service provider for algo/AI trading
- Potential ICT third-party provider for regulated EU clients
- Responsible for operational resilience of OUR platform

**WE ARE NOT:**
- Financial entity under DORA Article 2(1)(a-t)
- Subject to direct DORA compliance requirements
- Responsible for client's overall regulatory compliance

### 2.3 Relevant DORA Areas (Technical)

| DORA Chapter | Relevance | Our Role |
|--------------|-----------|----------|
| **ICT Risk Management (Art. 5-16)** | HIGH | Implement for own platform |
| **Incident Management (Art. 17-23)** | HIGH | Report to clients per SLA |
| **Resilience Testing (Art. 24-27)** | MEDIUM | Test own systems, support client audits |
| **Third-Party Risk (Art. 28-44)** | LOW | We ARE the third-party; document our practices |
| **Information Sharing (Art. 45)** | LOW | Optional threat intelligence |

### 2.4 What We Don't Take On

- Role of "financial entity" under DORA
- Regulatory reporting to NCAs (client's responsibility)
- TLPT coordination (client's responsibility)
- Register of Information submission (client's responsibility)
- Critical Third-Party Provider (CTPP) oversight (we're not designated)

---

## 3. Design Principles

### Principle 1: Operational Resilience by Design

Monitoring, logging, health-checks, graceful degradation встроены в архитектуру, не добавлены поверх.

**Implementation:**
- Health endpoints на всех сервисах
- Structured logging с correlation IDs
- Circuit breakers для external dependencies
- Graceful shutdown sequences

### Principle 2: Clear Separation of Responsibilities

Документация чётко разграничивает:
- Что обеспечивает платформа (availability, monitoring, backups)
- Что остаётся за клиентом (их DORA compliance, regulatory reporting)

**Implementation:**
- Shared Responsibility Matrix в документации
- SLA templates с явными границами
- Client-facing status page

### Principle 3: Core for Everyone, Extended for Enterprise

```
┌─────────────────────────────────────────────────────────────────┐
│                         ENTERPRISE TIER                          │
│  Extended incident reports, SIEM integration, custom SLAs,      │
│  on-prem support, audit artifacts, dedicated support            │
├─────────────────────────────────────────────────────────────────┤
│                           CORE TIER                              │
│  Standard monitoring, logging, backups, incident handling,      │
│  DR procedures, health checks, basic SLA                        │
└─────────────────────────────────────────────────────────────────┘
```

### Principle 4: Evidence-Friendly Architecture

Все процессы имеют артефакты, которые можно показать клиенту/аудитору:
- Structured logs с retention
- Incident reports с timeline
- DR test reports
- Change management records

---

## 4. Classification of Current Components

### 4.A) Core Operational Resilience Candidates

| Component | Location | Status | Notes |
|-----------|----------|--------|-------|
| Kill Switch | [services/ops_kill_switch.py](services/ops_kill_switch.py) | KEEP | Core safety mechanism |
| Healthcheck | [services/healthcheck.py](services/healthcheck.py) | ENHANCE | Add more endpoints |
| Secure Logging | [services/secure_logging.py](services/secure_logging.py) | KEEP | API key masking |
| Monitoring | [services/monitoring.py](services/monitoring.py) | ENHANCE | Add alerting |
| Recovery Procedures | [docs/RECOVERY_PROCEDURES.md](docs/RECOVERY_PROCEDURES.md) | KEEP | 10 scenarios |
| Operations Runbook | [docs/OPERATIONS_RUNBOOK.md](docs/OPERATIONS_RUNBOOK.md) | KEEP | Comprehensive |
| Service Dependency Map | [docs/SERVICE_DEPENDENCY_MAP.md](docs/SERVICE_DEPENDENCY_MAP.md) | KEEP | Architecture clarity |
| CI/CD Pipeline | [.github/workflows/](/.github/workflows/) | ENHANCE | Add security gates |
| Audit Trail | [services/compliance/audit_trail_writer.py](services/compliance/audit_trail_writer.py) | KEEP | Reposition as Core |
| BCP Module | [services/compliance/bcp.py](services/compliance/bcp.py) | EXTRACT | Move useful parts |

### 4.B) Enterprise DORA Support (Retain, Reposition)

| Component | Location | Action |
|-----------|----------|--------|
| Incident Classification | [services/dora/incident_classification.py](services/dora/incident_classification.py) | REPOSITION as Enterprise |
| Incident Reporting | [services/dora/incident_reporting.py](services/dora/incident_reporting.py) | REPOSITION as Enterprise |
| ICT Business Continuity | [services/dora/ict_business_continuity.py](services/dora/ict_business_continuity.py) | EXTRACT useful parts |
| Backup Recovery | [services/dora/backup_recovery.py](services/dora/backup_recovery.py) | EXTRACT useful parts |
| Third-Party Risk | [services/dora/third_party_risk.py](services/dora/third_party_risk.py) | KEEP for self-documentation |
| Contractual Requirements | [services/dora/contractual_requirements.py](services/dora/contractual_requirements.py) | REPOSITION as Enterprise |

### 4.C) Misaligned / Overkill (Refactor or Remove)

| Component | Location | Issue | Action |
|-----------|----------|-------|--------|
| Scope Verification | [services/dora/scope_verification.py](services/dora/scope_verification.py) | Assumes WE are financial entity | REMOVE or document as client tool |
| Function Classification | [services/dora/function_classification.py](services/dora/function_classification.py) | Article 3(22) for financial entities | REPURPOSE for internal criticality |
| Proportionality | [services/dora/proportionality.py](services/dora/proportionality.py) | DORA regime determination | REMOVE |
| Governance | [services/dora/governance.py](services/dora/governance.py) | Article 5 for financial entities | SIMPLIFY to internal governance |
| TLPT | [services/dora/tlpt.py](services/dora/tlpt.py) | Threat-Led Pen Testing | REMOVE (client responsibility) |
| CTPP Oversight | [services/dora/ctpp_oversight.py](services/dora/ctpp_oversight.py) | Critical Third-Party oversight | REMOVE (not applicable) |
| Register of Information | [services/dora/register_of_information.py](services/dora/register_of_information.py) | ROI submission to NCAs | REMOVE (client responsibility) |
| Concentration Risk | [services/dora/concentration_risk.py](services/dora/concentration_risk.py) | Client's concentration analysis | REMOVE |
| NCA Identification | [config/dora/nca_identification.yaml](config/dora/nca_identification.yaml) | Client identifies their NCA | REMOVE |
| Pooled Testing | [services/dora/pooled_testing.py](services/dora/pooled_testing.py) | Joint testing arrangements | REMOVE |
| Supervisory Feedback | [services/dora/supervisory_feedback.py](services/dora/supervisory_feedback.py) | NCA communication | REMOVE |

---

## 5. Target Core Operational Resilience Layer

### 5.1 Monitoring & Alerting

```yaml
core_monitoring:
  metrics:
    - availability_uptime_percent
    - latency_p50_p95_p99:
        - order_submission
        - market_data_fetch
        - strategy_execution
    - error_rates:
        - broker_api_errors
        - internal_errors
    - resource_usage:
        - cpu_percent
        - memory_percent
        - disk_usage
        - queue_depth

  alerting:
    critical:
      - uptime < 99%
      - latency_p99 > 5s
      - error_rate > 5%
    warning:
      - uptime < 99.5%
      - latency_p95 > 2s
      - error_rate > 1%

  dashboards:
    - system_health_overview
    - trading_operations
    - error_analysis
```

### 5.2 Logging & Audit Trail

```yaml
core_logging:
  technical_logs:
    - market_data_events
    - order_lifecycle
    - strategy_signals
    - system_errors
    - security_events

  audit_trail:
    - session_start_stop
    - config_changes
    - user_actions
    - api_key_usage (masked)

  retention:
    technical: 90_days
    audit: 7_years
    security: 1_year

  format: structured_json
  correlation_id: required
```

### 5.3 Backup & Recovery

```yaml
core_backup:
  targets:
    critical:
      - user_configs
      - strategy_definitions
      - trained_models
      - api_credentials (encrypted)
    important:
      - backtest_results
      - training_artifacts

  schedule:
    critical: daily
    important: weekly

  retention:
    critical: 35_days
    important: 90_days

  recovery:
    documented_runbook: true
    smoke_test: monthly
    rto_target: 4_hours
    rpo_target: 24_hours
```

### 5.4 Business Continuity & DR

```yaml
core_dr:
  scenarios:
    - primary_database_loss
    - cloud_region_failure
    - broker_api_outage
    - complete_platform_failure

  objectives:
    rto: 4_hours
    rpo: 1_hour (for trading state)

  procedures:
    - documented in RECOVERY_PROCEDURES.md
    - tested annually

  fallback:
    - graceful_degradation_mode
    - read_only_mode
    - manual_intervention_procedures
```

### 5.5 Incident Management (Core Level)

```yaml
core_incidents:
  classification:
    critical: # Immediate response
      - complete_service_outage
      - data_breach
      - unauthorized_trades
    high: # 1h response
      - partial_outage
      - degraded_performance
      - security_anomaly
    medium: # 4h response
      - minor_feature_unavailable
      - elevated_error_rates
    low: # 24h response
      - cosmetic_issues
      - non_critical_bugs

  process:
    - detect (automated + manual)
    - triage (classification)
    - mitigate (contain damage)
    - resolve (fix root cause)
    - post_mortem (simple format)

  tracking:
    tool: github_issues_or_linear
    post_mortem: required_for_critical_high
```

### 5.6 Change & Release Management

```yaml
core_changes:
  ci_cd:
    - automated_tests_required
    - linting_required
    - security_scan_required
    - code_review_required

  deployment:
    strategy: rolling_or_blue_green
    rollback: documented_procedure
    feature_flags: for_major_changes

  config_management:
    versioned: git
    review_required: true
    audit_log: true
```

---

## 6. Target Enterprise DORA Support Layer

### 6.1 Extended Incident Reporting

```yaml
enterprise_incidents:
  extended_fields:
    - root_cause_analysis
    - remedial_actions
    - service_impact_assessment
    - timeline_with_timestamps
    - affected_clients

  export_formats:
    - pdf_report
    - json_structured
    - client_specific_template

  sla_integration:
    - notification_within_sla
    - report_delivery_within_sla
```

### 6.2 Formalised BCP/DR Artifacts

```yaml
enterprise_bcp:
  documents:
    - business_continuity_plan_template
    - disaster_recovery_plan_template
    - client_specific_customization

  parameters:
    - customizable_rto_rpo
    - client_specific_sla
    - escalation_contacts

  testing:
    - documented_test_results
    - client_participation_option
```

### 6.3 Extended Monitoring & Logging

```yaml
enterprise_monitoring:
  per_client_metrics:
    - usage_statistics
    - error_rates_per_client
    - latency_per_client

  integrations:
    - siem_export (splunk, elk)
    - client_monitoring_webhook
    - custom_alerting_channels
```

### 6.4 Vendor/Supply Chain Documentation

```yaml
enterprise_vendor:
  documentation:
    - upstream_providers_list
    - cloud_provider_certifications
    - data_provider_agreements
    - broker_integration_status

  reports:
    - third_party_summary_report
    - subprocessor_list_gdpr
```

### 6.5 On-Prem/Self-Hosted Support

```yaml
enterprise_onprem:
  artifacts:
    - infrastructure_requirements
    - deployment_guide
    - operational_procedures
    - compliance_checklist

  support:
    - installation_assistance
    - configuration_review
    - security_hardening_guide
```

---

## 7. Cleanup Plan for Misaligned Components

### 7.1 Immediate Removal (Phase 1)

| Module | Action | Reason |
|--------|--------|--------|
| `services/dora/scope_verification.py` | Archive to `archive/dora_client_tools/` | Not applicable to ICT provider |
| `services/dora/proportionality.py` | Archive | Financial entity classification |
| `services/dora/tlpt.py` | Archive | Client responsibility |
| `services/dora/ctpp_oversight.py` | Archive | Not designated as CTPP |
| `services/dora/register_of_information.py` | Archive | Client submits to NCA |
| `services/dora/concentration_risk.py` | Archive | Client's analysis |
| `services/dora/pooled_testing.py` | Archive | Client arrangements |
| `services/dora/supervisory_feedback.py` | Archive | Client-NCA communication |
| `config/dora/nca_identification.yaml` | Archive | Client identifies NCA |
| `config/dora/entity_classification.yaml` | Archive | Financial entity config |

### 7.2 Repurpose (Phase 2)

| Module | Current | Target |
|--------|---------|--------|
| `services/dora/function_classification.py` | DORA Article 3(22) | Internal service criticality classification |
| `services/dora/governance.py` | Financial entity governance | Platform internal governance |
| `services/dora/ict_systems.py` | DORA Article 7 | Internal system inventory |
| `services/dora/detection.py` | DORA Article 10 | Core anomaly detection |
| `services/dora/protection.py` | DORA Article 9 | Core security controls |

### 7.3 Extract & Move to Core (Phase 2)

| Source | Extract | Target |
|--------|---------|--------|
| `services/dora/backup_recovery.py` | Backup automation logic | `services/core/backup.py` |
| `services/dora/ict_business_continuity.py` | RTO/RPO definitions | `services/core/continuity.py` |
| `services/dora/incident_management.py` | Core incident workflow | `services/core/incidents.py` |
| `services/dora/response_recovery.py` | Recovery procedures | `services/core/recovery.py` |

### 7.4 Move to Enterprise Module (Phase 3)

| Source | Target |
|--------|--------|
| `services/dora/incident_classification.py` | `services/enterprise/incident_classification.py` |
| `services/dora/incident_reporting.py` | `services/enterprise/incident_reporting.py` |
| `services/dora/contractual_requirements.py` | `services/enterprise/contractual.py` |
| `services/dora/third_party_risk.py` | `services/enterprise/vendor_documentation.py` |

### 7.5 Documentation Updates

| Document | Action |
|----------|--------|
| `docs/compliance/DORA_INTEGRATION_PLAN.md` | Reframe as "DORA Support for Clients" |
| `README.md` | Clarify platform role (ICT provider, not financial entity) |
| `ARCHITECTURE.md` | Add Core vs Enterprise separation |
| New: `docs/SHARED_RESPONSIBILITY.md` | Platform vs Client responsibilities |
| New: `docs/CLIENT_DORA_SUPPORT.md` | How we help clients with DORA |

---

## 8. Phased Roadmap

### Phase 1: Baseline Operational Hygiene

**Goals:**
- Clean up misaligned DORA components
- Strengthen core monitoring/logging
- Formalize incident management
- Document shared responsibilities

**Work Blocks:**

| Block | Description | Dependencies | Parallel |
|-------|-------------|--------------|----------|
| 1.1 | Archive misaligned DORA modules (7.1) | None | Yes |
| 1.2 | Enhance healthcheck endpoints | None | Yes |
| 1.3 | Implement structured logging with correlation IDs | None | Yes |
| 1.4 | Add basic alerting to monitoring | 1.3 | No |
| 1.5 | Formalize incident classification (Core level) | None | Yes |
| 1.6 | Create SHARED_RESPONSIBILITY.md | None | Yes |
| 1.7 | Update README.md with platform positioning | 1.6 | No |

**Deliverables:**
- Clean `services/dora/` with only relevant modules
- Enhanced healthcheck with `/health`, `/ready`, `/live` endpoints
- Structured logging across all services
- Basic alerting dashboard
- Core incident classification schema
- Shared responsibility documentation

### Phase 2: Core DORA-Aligned Resilience

**Goals:**
- Extract and consolidate Core layer
- Improve DR/BCP procedures
- Enhance metrics and health checks
- Formalize change management

**Work Blocks:**

| Block | Description | Dependencies | Parallel |
|-------|-------------|--------------|----------|
| 2.1 | Create `services/core/` package | Phase 1 | Yes |
| 2.2 | Extract backup logic to Core | 2.1 | No |
| 2.3 | Extract continuity/recovery to Core | 2.1 | No |
| 2.4 | Repurpose useful DORA modules (7.2) | Phase 1 | Yes |
| 2.5 | Implement backup automation with testing | 2.2 | No |
| 2.6 | Add comprehensive metrics (5.1) | Phase 1 | Yes |
| 2.7 | Enhance CI/CD with security gates | Phase 1 | Yes |
| 2.8 | Annual DR test procedure | 2.3 | No |
| 2.9 | Create CLIENT_DORA_SUPPORT.md | Phase 1 | Yes |

**Deliverables:**
- `services/core/` with backup, continuity, incidents, recovery
- Automated backup with monthly smoke tests
- Comprehensive metrics dashboard
- CI/CD with SAST/DAST gates
- DR test runbook and schedule
- Client DORA support documentation

### Phase 3: Enterprise DORA Support

**Goals:**
- Build Enterprise module for regulated clients
- Create client-facing artifacts
- Implement extended reporting
- Support on-prem deployments

**Work Blocks:**

| Block | Description | Dependencies | Parallel |
|-------|-------------|--------------|----------|
| 3.1 | Create `services/enterprise/` package | Phase 2 | Yes |
| 3.2 | Move advanced incident reporting to Enterprise | 3.1 | No |
| 3.3 | Implement extended incident report formats | 3.2 | No |
| 3.4 | Create BCP/DR document templates | Phase 2 | Yes |
| 3.5 | Implement per-client metrics | 3.1 | No |
| 3.6 | Add SIEM export capability | 3.1 | Yes |
| 3.7 | Create vendor documentation package | Phase 2 | Yes |
| 3.8 | Develop on-prem deployment guide | Phase 2 | Yes |
| 3.9 | Create SLA templates with DORA alignment | 3.3 | No |
| 3.10 | Feature flag system for Enterprise features | 3.1 | Yes |

**Deliverables:**
- `services/enterprise/` with all extended features
- PDF/JSON incident report generation
- BCP/DR templates for clients
- Per-client metrics and dashboards
- SIEM integration (Splunk/ELK export)
- Vendor/subprocessor documentation
- On-prem deployment guide
- Enterprise SLA templates
- Feature flag system separating Core/Enterprise

---

## 9. Architecture After Refactoring

```
services/
├── core/                      # Core operational resilience (all users)
│   ├── __init__.py
│   ├── backup.py              # Backup automation
│   ├── continuity.py          # BCP/DR definitions
│   ├── incidents.py           # Core incident management
│   ├── recovery.py            # Recovery procedures
│   ├── monitoring.py          # Enhanced monitoring (from services/)
│   ├── healthcheck.py         # Enhanced healthcheck (from services/)
│   └── logging.py             # Structured logging (from services/)
│
├── enterprise/                # Enterprise DORA support (licensed clients)
│   ├── __init__.py
│   ├── incident_reporting.py  # Extended incident reports
│   ├── incident_classification.py
│   ├── contractual.py         # Contract templates
│   ├── vendor_documentation.py # Third-party documentation
│   ├── metrics_export.py      # SIEM/per-client metrics
│   └── onprem/                # On-prem support
│       ├── deployment_guide.py
│       └── requirements.py
│
├── dora/                      # Remaining DORA utilities (repurposed)
│   ├── __init__.py            # Simplified exports
│   ├── criticality.py         # Internal criticality classification
│   └── systems_inventory.py   # Internal system inventory
│
├── compliance/                # Regulatory compliance (unchanged)
│   └── ...
│
└── ...                        # Other services
```

---

## 10. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Uptime | 99.9% | Monitoring |
| MTTD (Mean Time to Detect) | <15 min | Incident tracking |
| MTTR (Mean Time to Resolve) | <4h for critical | Incident tracking |
| Backup success rate | 100% | Backup logs |
| DR test pass rate | 100% | Annual test |
| Core test coverage | >80% | CI/CD |
| Enterprise features isolated | 100% | Feature flags |

---

## 11. Risk Factors

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Over-aggressive cleanup breaks functionality | Medium | High | Comprehensive testing before removal |
| Enterprise features leak to Core | Low | Medium | Feature flag enforcement |
| Client confusion about responsibilities | Medium | Medium | Clear documentation |
| Underestimating refactoring effort | Medium | Medium | Incremental phases |

---

## Appendix A: Files to Archive

```
archive/dora_client_tools/
├── scope_verification.py
├── proportionality.py
├── tlpt.py
├── ctpp_oversight.py
├── register_of_information.py
├── concentration_risk.py
├── pooled_testing.py
├── supervisory_feedback.py
├── configs/
│   ├── nca_identification.yaml
│   └── entity_classification.yaml
└── README.md (explaining why archived)
```

---

## Appendix B: Reference Documents

| Document | Purpose |
|----------|---------|
| [DORA Regulation](https://eur-lex.europa.eu/eli/reg/2022/2554/oj) | Official text |
| [OPERATIONS_RUNBOOK.md](docs/OPERATIONS_RUNBOOK.md) | Current operations |
| [RECOVERY_PROCEDURES.md](docs/RECOVERY_PROCEDURES.md) | Current recovery |
| [CYBERSECURITY_FRAMEWORK.md](docs/CYBERSECURITY_FRAMEWORK.md) | NIST CSF 2.0 |
| [SOC2_ROADMAP.md](docs/SOC2_ROADMAP.md) | SOC2 certification |

---

**Document Owner**: Platform Team
**Review Cycle**: Quarterly
**Next Review**: Q1 2026
