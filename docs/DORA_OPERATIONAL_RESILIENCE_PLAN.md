# DORA Operational Resilience Plan

**Version**: 2.0
**Date**: 2025-12-09
**Status**: Architecture Review & Roadmap
**Revision**: Critical audit v2.0 — fixed ICT provider obligations

---

## Changelog v2.0

| # | Issue | Fix |
|---|-------|-----|
| 1 | ICT provider role misunderstood | Section 2 rewritten — contractual obligations via Art. 28-30 |
| 2 | exit_strategies.py marked REMOVE | KEEP — required for client exit rights |
| 3 | register_of_information.py marked REMOVE | KEEP as provider_information_package |
| 4 | contractual_requirements.py in Enterprise | Moved to Core — mandatory for EU clients |
| 5 | Audit rights not mentioned | Added Section 5.7 Audit Readiness |
| 6 | concentration_risk fully removed | Added to risk awareness |
| 7 | RTO/RPO contradictions | Clarified with justification |
| 8 | Incident notification timing | Accelerated for client-critical |
| 9 | Subcontracting not covered | Added to Core layer |
| 10 | SOC2-DORA overlap ignored | Added mapping section |
| 11 | Roadmap dependencies wrong | Contractual moved to Phase 1 |
| 12 | Tests not considered | Added test migration plan |

---

## 1. Executive Summary: Repository Analysis

### 1.1 Discovered Components

| Area | Files/Modules | Maturity |
|------|---------------|----------|
| **DORA Services** | `services/dora/` - 40+ modules | HIGH (needs repositioning) |
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

**Needs Repositioning (NOT over-engineered):**
- `services/dora/` содержит модули, которые НУЖНЫ для ICT provider obligations
- Проблема не в избыточности, а в неверном позиционировании (как financial entity вместо ICT provider)
- Многие модули нужно адаптировать, не удалять

**Missing/Weak:**
- Audit readiness для регуляторных проверок
- Subcontractor documentation (AWS, data providers)
- Provider information package для клиентских ROI
- Formal SLA templates с DORA clauses
- SOC2 ↔ DORA control mapping

---

## 2. Target DORA Posture & Assumptions

### 2.1 Product Position

```
┌─────────────────────────────────────────────────────────────────┐
│              OUR POSITION IN DORA ECOSYSTEM (CORRECTED)         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FINANCIAL ENTITIES              ICT THIRD-PARTY PROVIDERS      │
│  (Direct DORA obligations)       (Contractual DORA obligations) │
│                                                                  │
│  ┌────────────────────────┐      ┌────────────────────────────┐ │
│  │ Our B2B Clients        │      │   WE ARE HERE              │ │
│  │ (Banks, Investment     │◄─────┤   (SaaS Platform)          │ │
│  │  Firms, Crypto CASPs)  │      │                            │ │
│  │                        │      │ DORA applies to us         │ │
│  │ Direct DORA scope:     │      │ INDIRECTLY via:            │ │
│  │ Art. 2(1)(a-t)         │      │ - Art. 28 (general)        │ │
│  │                        │      │ - Art. 30 (contractual)    │ │
│  │ They MUST ensure we    │      │ - Client audit rights      │ │
│  │ comply via contracts   │      │ - NCA inspection rights    │ │
│  └────────────────────────┘      └────────────────────────────┘ │
│                                                                  │
│  KEY INSIGHT: We are NOT exempt from DORA.                      │
│  We must comply through contractual arrangements.               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Explicit Scope — CORRECTED

**WE ARE:**
- SaaS platform / ICT service provider for algo/AI trading
- ICT third-party provider for regulated EU clients
- **Subject to DORA via contractual requirements (Art. 28-30)**
- Required to support client audit and NCA inspection rights
- Required to provide exit strategies and data portability

**WE ARE NOT:**
- Financial entity under DORA Article 2(1)(a-t) — no DIRECT NCA reporting
- Designated Critical Third-Party Provider (CTPP) — yet
- Responsible for client's internal DORA compliance program

### 2.3 DORA Obligations for ICT Providers (Art. 28-30)

| DORA Article | Requirement | Our Obligation |
|--------------|-------------|----------------|
| **Art. 28(5)** | Information security standards | Comply with appropriate standards |
| **Art. 28(8)** | Exit strategies | Provide documented exit plan, data portability |
| **Art. 30(2)** | Basic contractual terms | Include in ALL contracts with EU clients |
| **Art. 30(3)** | Extended terms (critical functions) | Audit rights, inspection, locations, subcontracting |
| **Art. 30(3)(e)** | Audit and access rights | Allow client + NCA audits |
| **Art. 29** | Subcontracting chain | Document and disclose subcontractors |

### 2.4 Article 30(2) — Mandatory Contract Clauses

ALL contracts with EU regulated clients MUST include:

```yaml
mandatory_contract_clauses:
  art_30_2_a: "Clear description of all ICT services"
  art_30_2_b: "Locations of data processing and storage"
  art_30_2_c: "Data protection and access provisions"
  art_30_2_d: "Service availability guarantees (SLA)"
  art_30_2_e: "Termination rights and notice periods"
  art_30_2_f: "Cooperation with competent authorities"
  art_30_2_g: "Termination rights for regulatory reasons"
```

### 2.5 Article 30(3) — Additional Requirements for Critical Functions

If client classifies our services as supporting "critical or important function":

```yaml
additional_requirements_critical:
  art_30_3_a: "Full service level descriptions with quantitative targets"
  art_30_3_b: "Notice periods and reporting obligations"
  art_30_3_c: "Business contingency plans"
  art_30_3_d: "ICT security measures participation"
  art_30_3_e: "UNRESTRICTED audit and access rights"
  art_30_3_f: "Exit strategies with transition periods"
  art_30_3_g: "Cooperation in supervisory oversight"
```

### 2.6 What We Don't Take On

- Role of "financial entity" under DORA Article 2(1)(a-t)
- Direct regulatory reporting to NCAs
- TLPT coordination (client's responsibility, but we must cooperate)
- Register of Information submission (client submits, we provide data)
- Client's internal governance and risk management

### 2.7 CTPP Designation Risk

**Current status:** We are NOT designated as Critical Third-Party Provider.

**Risk factors for future designation:**
- High market share among EU financial entities
- Services supporting critical functions for multiple clients
- Limited substitutability

**If designated as CTPP (Art. 31-44):**
- Direct oversight by Lead Overseer (ESA)
- Mandatory operational resilience requirements
- Regular reporting and examinations

**Mitigation:** Monitor client concentration, prepare for potential designation.

---

## 3. Design Principles

### Principle 1: Operational Resilience by Design

Monitoring, logging, health-checks, graceful degradation встроены в архитектуру.

**Implementation:**
- Health endpoints на всех сервисах
- Structured logging с correlation IDs
- Circuit breakers для external dependencies
- Graceful shutdown sequences

### Principle 2: Clear Separation of Responsibilities

Документация чётко разграничивает:
- Что обеспечивает платформа (availability, monitoring, backups, audit support)
- Что остаётся за клиентом (их internal DORA program, NCA reporting)

**Implementation:**
- Shared Responsibility Matrix
- SLA templates с DORA clauses
- Client-facing status page

### Principle 3: Contractual Compliance First

DORA contractual requirements (Art. 30) — это не "Enterprise feature", а базовое требование для работы с EU clients.

**Implementation:**
- Standard contract templates с Art. 30(2) clauses
- Enhanced templates для critical functions (Art. 30(3))
- Audit readiness procedures

### Principle 4: Evidence-Friendly Architecture

Все процессы имеют артефакты для client/auditor/NCA:
- Structured logs с retention
- Incident reports с timeline
- DR test reports
- Change management records
- Subcontractor documentation

### Principle 5: Audit-Ready Operations

Готовность к проверкам клиентами и регуляторами:
- Документированные процедуры
- Access для аудиторов
- Evidence preservation

---

## 4. Classification of Current Components — REVISED

### 4.A) Core Operational Resilience (ALL users)

| Component | Location | Status | Notes |
|-----------|----------|--------|-------|
| Kill Switch | [services/ops_kill_switch.py](services/ops_kill_switch.py) | KEEP | Core safety |
| Healthcheck | [services/healthcheck.py](services/healthcheck.py) | ENHANCE | Add /ready, /live |
| Secure Logging | [services/secure_logging.py](services/secure_logging.py) | KEEP | API key masking |
| Monitoring | [services/monitoring.py](services/monitoring.py) | ENHANCE | Add alerting |
| Recovery Procedures | [docs/RECOVERY_PROCEDURES.md](docs/RECOVERY_PROCEDURES.md) | KEEP | 10 scenarios |
| Operations Runbook | [docs/OPERATIONS_RUNBOOK.md](docs/OPERATIONS_RUNBOOK.md) | KEEP | Comprehensive |
| Service Dependency Map | [docs/SERVICE_DEPENDENCY_MAP.md](docs/SERVICE_DEPENDENCY_MAP.md) | KEEP | Architecture |
| CI/CD Pipeline | [.github/workflows/](/.github/workflows/) | ENHANCE | Security gates |
| Audit Trail | [services/compliance/audit_trail_writer.py](services/compliance/audit_trail_writer.py) | KEEP | Reposition as Core |
| BCP Module | [services/compliance/bcp.py](services/compliance/bcp.py) | KEEP | Core continuity |

### 4.B) Core DORA Contractual (ALL EU clients) — NEW CATEGORY

| Component | Location | Action | Rationale |
|-----------|----------|--------|-----------|
| **Contractual Requirements** | [services/dora/contractual_requirements.py](services/dora/contractual_requirements.py) | **KEEP as Core** | Art. 30(2) mandatory |
| **Exit Strategies** | [services/dora/exit_strategies.py](services/dora/exit_strategies.py) | **KEEP, adapt** | Art. 28(8), Art. 30(3)(f) |
| **Third-Party Risk** | [services/dora/third_party_risk.py](services/dora/third_party_risk.py) | **KEEP, adapt** | Self-documentation |
| **Incident Management** | [services/dora/incident_management.py](services/dora/incident_management.py) | **KEEP** | Client notification |
| **Incident Reporting** | [services/dora/incident_reporting.py](services/dora/incident_reporting.py) | **KEEP** | Client reports |
| **Backup Recovery** | [services/dora/backup_recovery.py](services/dora/backup_recovery.py) | **KEEP** | Art. 30(3)(c) |
| **ICT Business Continuity** | [services/dora/ict_business_continuity.py](services/dora/ict_business_continuity.py) | **KEEP** | Art. 30(3)(c) |

### 4.C) Enterprise DORA Support (Enhanced for regulated clients)

| Component | Location | Action |
|-----------|----------|--------|
| Incident Classification | [services/dora/incident_classification.py](services/dora/incident_classification.py) | Enterprise — extended taxonomy |
| Register of Information | [services/dora/register_of_information.py](services/dora/register_of_information.py) | **ADAPT** → provider_info_package |
| TLPT | [services/dora/tlpt.py](services/dora/tlpt.py) | Enterprise — cooperation support |
| Resilience Testing | [services/dora/resilience_testing.py](services/dora/resilience_testing.py) | Enterprise — joint testing |
| ICT Testing | [services/dora/ict_testing.py](services/dora/ict_testing.py) | Enterprise — test support |

### 4.D) Internal Platform Tools (Repurpose)

| Component | Current | Target |
|-----------|---------|--------|
| `function_classification.py` | DORA Article 3(22) | Internal service criticality |
| `governance.py` | Financial entity governance | Platform internal governance |
| `ict_systems.py` | DORA Article 7 | Internal system inventory |
| `detection.py` | DORA Article 10 | Core anomaly detection |
| `protection.py` | DORA Article 9 | Core security controls |

### 4.E) Archive (Not applicable to ICT provider role)

| Component | Reason |
|-----------|--------|
| `scope_verification.py` | Determines if DORA applies — irrelevant, we know it applies via contracts |
| `proportionality.py` | Financial entity size classification |
| `ctpp_oversight.py` | We're not designated CTPP (keep awareness) |
| `pooled_testing.py` | Client arrangements, not provider |
| `supervisory_feedback.py` | Client-NCA communication |
| `nca_identification.yaml` | Client identifies their NCA |
| `entity_classification.yaml` | Financial entity config |

### 4.F) Concentration Risk — Special Handling

| Component | Action | Rationale |
|-----------|--------|-----------|
| `concentration_risk.py` | **KEEP for awareness** | If we gain market share → CTPP designation risk |

---

## 5. Target Core Operational Resilience Layer

### 5.1 Monitoring & Alerting

```yaml
core_monitoring:
  metrics:
    availability:
      - uptime_percent
      - error_rate_percent
    latency:
      - order_submission_p50_p95_p99
      - market_data_fetch_p50_p95_p99
      - strategy_execution_p50_p95_p99
    broker_integration:
      - connection_status
      - api_error_rate
      - reconnection_count
    resources:
      - cpu_percent
      - memory_percent
      - disk_usage_percent
      - queue_depth

  alerting:
    critical:  # Immediate escalation
      - uptime < 99%
      - latency_p99 > 5s
      - error_rate > 5%
      - broker_connection_lost
    warning:   # 15min response
      - uptime < 99.5%
      - latency_p95 > 2s
      - error_rate > 1%
    info:      # Daily review
      - resource_usage > 70%

  dashboards:
    - system_health_overview
    - trading_operations
    - error_analysis
    - client_sla_tracking  # Per-client metrics
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
    - configuration_changes

  audit_trail:
    - session_start_stop
    - config_changes
    - user_actions
    - api_key_usage (masked)
    - admin_actions
    - data_access_events

  retention:
    technical: 90_days
    audit: 7_years       # Regulatory requirement
    security: 3_years
    incident: 7_years

  format: structured_json
  correlation_id: required
  tamper_protection: hash_chain
```

### 5.3 Backup & Recovery — CORRECTED

```yaml
core_backup:
  targets:
    tier_1_critical:  # RTO: 1h, RPO: 15min
      - live_trading_state
      - open_positions
      - pending_orders
      - active_sessions
    tier_2_important:  # RTO: 4h, RPO: 1h
      - user_configs
      - strategy_definitions
      - trained_models
      - api_credentials (encrypted)
    tier_3_standard:   # RTO: 24h, RPO: 24h
      - backtest_results
      - training_artifacts
      - historical_logs

  schedule:
    tier_1: continuous_replication
    tier_2: every_4_hours
    tier_3: daily

  retention:
    tier_1: 7_days
    tier_2: 35_days
    tier_3: 90_days

  recovery:
    documented_runbook: true
    automated_smoke_test: weekly
    full_dr_test: quarterly
```

### 5.4 Business Continuity & DR — CORRECTED

```yaml
core_dr:
  recovery_objectives:
    trading_services:
      rto: 1_hour       # Must resume trading within 1h
      rpo: 15_minutes   # Max 15min data loss for positions
      justification: "Trading platform - financial impact of downtime"

    backtest_services:
      rto: 4_hours
      rpo: 24_hours
      justification: "Non-time-critical, no live trading impact"

    admin_services:
      rto: 8_hours
      rpo: 24_hours
      justification: "Support functions, can operate degraded"

  scenarios:
    - primary_database_failure
    - cloud_region_failure
    - broker_api_outage
    - complete_platform_failure
    - cyber_attack_recovery

  procedures:
    - documented in RECOVERY_PROCEDURES.md
    - tested quarterly
    - client notification within 30min of DR activation

  fallback:
    - graceful_degradation_mode
    - read_only_mode
    - manual_intervention_procedures
```

### 5.5 Incident Management — CORRECTED

```yaml
core_incidents:
  classification:
    critical:  # Immediate response, client notification <30min
      - complete_service_outage
      - data_breach
      - unauthorized_trades
      - security_compromise
    high:      # 30min response, client notification <1h
      - partial_outage
      - degraded_performance > 30min
      - security_anomaly
    medium:    # 2h response, client notification <4h
      - minor_feature_unavailable
      - elevated_error_rates
      - single_client_impact
    low:       # 8h response, no immediate notification
      - cosmetic_issues
      - non_critical_bugs

  client_notification:
    critical:
      timing: "<30 minutes"
      rationale: "Client needs 3.5h for their DORA reporting (4h deadline)"
    high:
      timing: "<1 hour"
    medium:
      timing: "<4 hours"

  process:
    - detect (automated + manual)
    - classify (severity + client impact)
    - notify (clients per SLA)
    - mitigate (contain damage)
    - resolve (fix root cause)
    - post_mortem (all critical/high)
    - client_report (formal incident report)

  tracking:
    tool: linear_or_jira
    post_mortem: required_for_critical_high
    client_report: required_for_critical_high_medium
```

### 5.6 Change & Release Management

```yaml
core_changes:
  ci_cd:
    - automated_tests: required
    - linting: required
    - security_scan: required (SAST)
    - code_review: required
    - dependency_check: required

  deployment:
    strategy: blue_green
    rollback: automated_one_click
    feature_flags: for_major_changes
    client_notification: for_breaking_changes

  config_management:
    versioned: git
    review_required: true
    audit_log: true
    rollback: supported
```

### 5.7 Audit Readiness — NEW

```yaml
audit_readiness:
  documentation:
    - system_architecture_diagrams
    - data_flow_diagrams
    - security_controls_inventory
    - change_management_records
    - incident_history
    - backup_test_results
    - dr_test_results

  access_provision:
    client_auditors:
      - read_access_to_logs
      - read_access_to_metrics
      - documentation_access
      - interview_availability
    nca_inspectors:
      - same_as_client_plus
      - on_site_inspection_support
      - evidence_preservation

  preparation:
    - audit_request_response_sla: 5_business_days
    - evidence_package_templates: ready
    - designated_audit_contact: defined

  annual_activities:
    - internal_control_testing
    - external_penetration_test
    - dr_test_with_documentation
    - policy_review
```

### 5.8 Subcontractor Documentation — NEW

```yaml
subcontractor_management:
  documentation_required:
    - subcontractor_name_and_lei
    - services_provided
    - data_processing_locations
    - security_certifications
    - subcontractor_chain (if any)

  current_subcontractors:
    cloud_infrastructure:
      provider: "AWS / GCP / Azure"
      services: "Compute, storage, networking"
      locations: "EU (Frankfurt, Dublin)"
      certifications: "SOC2, ISO27001, C5"

    market_data:
      provider: "Polygon, Binance, Alpaca"
      services: "Real-time and historical market data"
      locations: "US, Global"
      certifications: "Varies by provider"

  client_disclosure:
    - subprocessor_list_available_on_request
    - notification_of_changes: 30_days_advance
    - objection_right: per_contract
```

### 5.9 Contractual Compliance — NEW (Core)

```yaml
contractual_compliance:
  standard_contract_template:
    includes:
      - art_30_2_clauses: all
      - sla_definitions: yes
      - termination_rights: yes
      - data_protection: yes
      - audit_rights: basic

  critical_function_addendum:
    includes:
      - art_30_3_clauses: all
      - unrestricted_audit_rights: yes
      - exit_strategy: detailed
      - business_continuity_plan: yes
      - enhanced_sla: yes

  exit_strategy_components:
    - data_export_formats: json_csv_api
    - transition_period: minimum_90_days
    - data_retention_post_termination: 30_days
    - cooperation_commitment: yes
    - no_vendor_lock_in: documented
```

---

## 6. Target Enterprise DORA Support Layer

### 6.1 Extended Incident Reporting

```yaml
enterprise_incidents:
  extended_fields:
    - root_cause_analysis
    - remedial_actions_taken
    - remedial_actions_planned
    - service_impact_assessment
    - timeline_with_timestamps
    - affected_clients_list
    - regulatory_notification_status

  export_formats:
    - pdf_report (branded)
    - json_structured (machine_readable)
    - client_specific_template
    - nca_compatible_format

  sla_integration:
    - notification_timestamp_tracking
    - report_delivery_tracking
    - escalation_automation
```

### 6.2 Provider Information Package (for client ROI)

```yaml
provider_info_package:
  purpose: "Data for client Register of Information submission"

  contents:
    entity_identification:
      - legal_name
      - lei_or_alternative
      - registration_country
      - registration_number

    service_description:
      - ict_services_provided
      - functions_supported
      - criticality_assessment_support

    locations:
      - data_processing_locations
      - data_storage_locations
      - backup_locations

    subcontracting:
      - subcontractor_list
      - subcontracting_chain
      - material_subcontractors

    certifications:
      - soc2_status
      - iso27001_status
      - other_certifications

  delivery:
    - format: structured_json + pdf
    - update_frequency: annual + on_material_change
    - client_portal: available
```

### 6.3 Joint Testing Support

```yaml
enterprise_testing:
  client_testing_support:
    - penetration_test_cooperation
    - vulnerability_assessment_support
    - scenario_based_testing
    - dr_test_participation

  tlpt_cooperation:
    - threat_intelligence_sharing
    - red_team_access_coordination
    - evidence_provision
    - remediation_tracking

  documentation:
    - test_results_sharing (sanitized)
    - remediation_reports
    - improvement_tracking
```

### 6.4 Extended Monitoring & Logging

```yaml
enterprise_monitoring:
  per_client_metrics:
    - usage_statistics
    - error_rates_per_client
    - latency_per_client
    - availability_per_client

  integrations:
    - siem_export: splunk_elk_sentinel
    - client_monitoring_webhook
    - custom_alerting_channels
    - real_time_log_streaming
```

### 6.5 On-Prem/Self-Hosted Support

```yaml
enterprise_onprem:
  artifacts:
    - infrastructure_requirements
    - deployment_guide
    - operational_procedures
    - security_hardening_guide
    - compliance_checklist

  support:
    - installation_assistance
    - configuration_review
    - security_assessment
    - ongoing_maintenance_guidance
```

---

## 7. SOC2 ↔ DORA Control Mapping — NEW

### 7.1 Overlap Analysis

| SOC2 TSC | DORA Article | Overlap | Notes |
|----------|--------------|---------|-------|
| **CC6 (Security)** | Art. 9 (Protection) | HIGH | Access controls, encryption |
| **CC7 (Operations)** | Art. 10 (Detection) | HIGH | Monitoring, anomaly detection |
| **CC7.4 (Incident)** | Art. 17-19 (Incident) | HIGH | Incident management |
| **A1 (Availability)** | Art. 11-12 (Recovery) | HIGH | BCP/DR, backups |
| **PI1 (Processing)** | Art. 7 (ICT Systems) | MEDIUM | System integrity |
| **C1 (Confidentiality)** | Art. 9 (Protection) | HIGH | Data protection |

### 7.2 Efficiency Opportunities

```yaml
soc2_dora_synergy:
  shared_controls:
    - access_management
    - encryption_standards
    - incident_response
    - backup_procedures
    - change_management
    - vulnerability_management

  shared_evidence:
    - access_review_logs
    - incident_reports
    - backup_test_results
    - penetration_test_reports
    - change_records

  timeline_alignment:
    soc2_type2_observation: "Q4 2025 - Q1 2026"
    dora_compliance: "Ongoing from Jan 2025"
    recommendation: "Align evidence collection"
```

---

## 8. Cleanup Plan — REVISED

### 8.1 Archive (Phase 1)

| Module | Reason |
|--------|--------|
| `services/dora/scope_verification.py` | Not applicable — we know DORA applies via contracts |
| `services/dora/proportionality.py` | Financial entity classification |
| `services/dora/pooled_testing.py` | Client-side arrangements |
| `services/dora/supervisory_feedback.py` | Client-NCA communication |
| `config/dora/nca_identification.yaml` | Client identifies their NCA |
| `config/dora/entity_classification.yaml` | Financial entity config |

### 8.2 Adapt (Phase 1-2)

| Module | Current | Target |
|--------|---------|--------|
| `register_of_information.py` | ROI submission | `provider_info_package.py` — data FOR client ROI |
| `exit_strategies.py` | Client exit planning | Provider exit support + data portability |
| `third_party_risk.py` | Client risk assessment | Self-documentation + subcontractor info |
| `contractual_requirements.py` | DORA clauses | Contract templates with Art. 30 clauses |
| `concentration_risk.py` | Client analysis | CTPP designation awareness + monitoring |

### 8.3 Keep (Core)

| Module | Reason |
|--------|--------|
| `incident_management.py` | Core incident handling |
| `incident_reporting.py` | Client notification |
| `backup_recovery.py` | Core backup |
| `ict_business_continuity.py` | Core BCP |
| `detection.py` | Anomaly detection |
| `protection.py` | Security controls |

### 8.4 Test Migration Plan — NEW

```yaml
test_migration:
  phase_1:
    archive_tests:
      - test_dora_phase0_proportionality.py → archive
    keep_tests:
      - test_dora_phase2_incident_management.py
      - test_dora_concentration_risk.py (adapt)
      - test_dora_contractual_requirements.py
      - test_dora_exit_strategies.py

  phase_2:
    new_tests:
      - test_provider_info_package.py
      - test_audit_readiness.py
      - test_subcontractor_documentation.py
      - test_contract_templates.py

  coverage_target: ">80% for Core modules"
```

---

## 9. Phased Roadmap — REVISED

### Phase 1: Contractual Compliance & Baseline (PRIORITY)

**Goals:**
- Enable compliant contracts with EU clients NOW
- Establish audit readiness
- Clean up non-applicable modules

**Work Blocks:**

| Block | Description | Priority |
|-------|-------------|----------|
| 1.1 | Create contract templates with Art. 30(2) clauses | **CRITICAL** |
| 1.2 | Create critical function addendum (Art. 30(3)) | **CRITICAL** |
| 1.3 | Implement audit readiness procedures | **HIGH** |
| 1.4 | Create provider information package | **HIGH** |
| 1.5 | Document subcontractors (AWS, data providers) | **HIGH** |
| 1.6 | Adapt exit_strategies.py for provider role | **HIGH** |
| 1.7 | Archive non-applicable modules | MEDIUM |
| 1.8 | Create SHARED_RESPONSIBILITY.md | MEDIUM |
| 1.9 | Enhance incident notification (<30min critical) | **HIGH** |

**Deliverables:**
- DORA-compliant contract templates
- Audit readiness package
- Provider information package for client ROI
- Subcontractor documentation
- Exit strategy documentation
- Updated incident notification procedures

### Phase 2: Core Operational Resilience

**Goals:**
- Strengthen monitoring/logging/alerting
- Improve DR/BCP with documented RTO/RPO
- Enhance change management

**Work Blocks:**

| Block | Description | Priority |
|-------|-------------|----------|
| 2.1 | Implement tiered backup (15min/1h/24h RPO) | **HIGH** |
| 2.2 | Enhance healthcheck (/health, /ready, /live) | HIGH |
| 2.3 | Implement structured logging with correlation IDs | HIGH |
| 2.4 | Add comprehensive alerting | HIGH |
| 2.5 | Quarterly DR testing with documentation | HIGH |
| 2.6 | CI/CD security gates (SAST/DAST) | MEDIUM |
| 2.7 | SOC2 ↔ DORA control mapping | MEDIUM |
| 2.8 | Create `services/core/` package | MEDIUM |

**Deliverables:**
- Tiered backup system with automated testing
- Enhanced monitoring with SLA tracking
- Structured logging across all services
- Quarterly DR test reports
- SOC2-DORA mapping document

### Phase 3: Enterprise Enhancements

**Goals:**
- Extended reporting for regulated clients
- Joint testing support
- On-prem deployment support

**Work Blocks:**

| Block | Description | Priority |
|-------|-------------|----------|
| 3.1 | Create `services/enterprise/` package | HIGH |
| 3.2 | Extended incident report formats (PDF/JSON) | HIGH |
| 3.3 | Per-client metrics and dashboards | MEDIUM |
| 3.4 | SIEM integration (Splunk/ELK export) | MEDIUM |
| 3.5 | TLPT cooperation procedures | MEDIUM |
| 3.6 | On-prem deployment guide | MEDIUM |
| 3.7 | Enterprise SLA templates | MEDIUM |
| 3.8 | Feature flag system for Enterprise | LOW |

**Deliverables:**
- Extended incident reporting system
- Per-client monitoring
- SIEM integration
- TLPT cooperation documentation
- On-prem deployment package

---

## 10. Architecture After Refactoring

```
services/
├── core/                      # Core operational resilience (all users)
│   ├── __init__.py
│   ├── backup.py              # Tiered backup (from dora/backup_recovery.py)
│   ├── continuity.py          # BCP/DR (from dora/ict_business_continuity.py)
│   ├── incidents.py           # Incident management (from dora/)
│   ├── recovery.py            # Recovery procedures
│   ├── monitoring.py          # Enhanced monitoring
│   ├── healthcheck.py         # Enhanced healthcheck
│   ├── logging.py             # Structured logging
│   └── audit.py               # Audit readiness (NEW)
│
├── dora/                      # DORA contractual compliance (EU clients)
│   ├── __init__.py
│   ├── contractual.py         # Contract templates (Art. 30)
│   ├── exit_strategies.py     # Provider exit support (adapted)
│   ├── provider_info.py       # Provider info for client ROI (NEW)
│   ├── subcontractors.py      # Subcontractor documentation (NEW)
│   ├── incident_reporting.py  # Client incident reports
│   ├── detection.py           # Anomaly detection
│   └── protection.py          # Security controls
│
├── enterprise/                # Enterprise features (licensed clients)
│   ├── __init__.py
│   ├── extended_reporting.py  # PDF/JSON incident reports
│   ├── client_metrics.py      # Per-client monitoring
│   ├── siem_export.py         # SIEM integration
│   ├── tlpt_support.py        # TLPT cooperation
│   └── onprem/                # On-prem support
│       ├── deployment.py
│       └── requirements.py
│
├── compliance/                # Other regulatory (unchanged)
│   └── ...
│
└── archive/                   # Archived modules
    └── dora_not_applicable/
        ├── scope_verification.py
        ├── proportionality.py
        └── ...
```

---

## 11. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Contract compliance | 100% EU contracts have Art. 30 clauses | Contract review |
| Audit readiness | Response within 5 business days | Audit log |
| Uptime | 99.9% | Monitoring |
| MTTD (Mean Time to Detect) | <15 min | Incident tracking |
| MTTR (Mean Time to Resolve) | <1h critical, <4h high | Incident tracking |
| Client notification | <30min critical | Incident tracking |
| Backup success rate | 100% | Backup logs |
| DR test pass rate | 100% quarterly | DR test reports |
| Core test coverage | >80% | CI/CD |

---

## 12. Risk Factors — EXPANDED

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| EU client without proper contract | HIGH | HIGH | Phase 1 contract templates priority |
| NCA inspection unprepared | MEDIUM | HIGH | Audit readiness in Phase 1 |
| CTPP designation | LOW (now) | HIGH | Monitor concentration, prepare |
| Subcontractor incident | MEDIUM | HIGH | Subcontractor documentation, monitoring |
| Aggressive cleanup breaks tests | MEDIUM | MEDIUM | Test migration plan |
| SOC2 and DORA duplicated effort | MEDIUM | MEDIUM | Control mapping, shared evidence |

---

## Appendix A: Files to Archive

```
archive/dora_not_applicable/
├── scope_verification.py
├── proportionality.py
├── pooled_testing.py
├── supervisory_feedback.py
├── configs/
│   ├── nca_identification.yaml
│   └── entity_classification.yaml
├── tests/
│   └── test_dora_phase0_proportionality.py
└── README.md (explaining ICT provider vs financial entity)
```

## Appendix B: Files to Adapt

```
Adaptations:
├── register_of_information.py → provider_info.py
│   Purpose: Generate data FOR client ROI, not submit ROI
│
├── exit_strategies.py → exit_strategies.py (adapted)
│   Purpose: Provider-side exit support, data portability
│
├── third_party_risk.py → subcontractors.py
│   Purpose: Document OUR subcontractors for clients
│
└── concentration_risk.py → ctpp_awareness.py
    Purpose: Monitor our market concentration for CTPP risk
```

## Appendix C: Reference Documents

| Document | Purpose |
|----------|---------|
| [DORA Article 28](https://www.digital-operational-resilience-act.com/Article_28.html) | ICT third-party risk principles |
| [DORA Article 30](https://www.digital-operational-resilience-act.com/Article_30.html) | Contractual requirements |
| [OPERATIONS_RUNBOOK.md](docs/OPERATIONS_RUNBOOK.md) | Current operations |
| [RECOVERY_PROCEDURES.md](docs/RECOVERY_PROCEDURES.md) | Current recovery |
| [CYBERSECURITY_FRAMEWORK.md](docs/CYBERSECURITY_FRAMEWORK.md) | NIST CSF 2.0 |
| [SOC2_ROADMAP.md](docs/SOC2_ROADMAP.md) | SOC2 certification |

---

**Document Owner**: Platform Team
**Review Cycle**: Quarterly
**Next Review**: Q1 2026
