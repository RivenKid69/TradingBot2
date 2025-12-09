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
| **Art. 30(2)(b)** | Subcontracting provisions | Document and disclose subcontractors |
| **Art. 29** | ICT concentration risk (FE obligation) | Monitor our market concentration for CTPP risk |

### 2.4 Article 30(2) — Mandatory Contract Clauses (ALL 9 Subpoints)

ALL contracts with EU regulated clients MUST include these **9 mandatory clauses**:

```yaml
mandatory_contract_clauses:
  # Basic service terms
  art_30_2_a: "Clear and complete description of all ICT services to be provided"
  art_30_2_b: "Locations where data will be processed/stored, including subcontracting conditions"
  art_30_2_c: "Provisions on availability, authenticity, integrity, confidentiality of data"
  art_30_2_d: "Service level descriptions including quantitative/qualitative targets"
  art_30_2_e: "Obligation to provide assistance in case of ICT incidents at no additional cost or at predetermined cost"
  art_30_2_f: "Obligation to participate in financial entity's ICT resilience testing (per Art. 26-27)"
  art_30_2_g: "Obligation to fully cooperate with competent authorities and resolution authorities"

  # CRITICAL: Often-missed clauses
  art_30_2_h: "Termination rights and related minimum notice periods for contract termination"
  art_30_2_i: "Conditions for ICT provider participation in financial entity's security awareness programmes and digital operational resilience training (per Art. 13(6))"
```

**CRITICAL NOTE**: Many contracts miss Art. 30(2)(h) and (i). These are **mandatory** for ALL ICT service contracts, not just critical functions.

**Art. 30(2)(i) Implementation** — Training Participation:
```yaml
training_participation_clause:
  commitment: |
    Provider shall make relevant personnel available to participate in
    Client's ICT security awareness programmes and digital operational
    resilience training as reasonably requested.

  scope:
    - security_awareness_programs: "Annual participation upon request"
    - resilience_training_exercises: "Tabletop exercises, DR drills"
    - joint_incident_simulations: "As agreed in SLA"

  conditions:
    - reasonable_notice: "14 business days minimum"
    - personnel_availability: "Subject to operational needs"
    - remote_participation: "Preferred where feasible"
    - materials_provided: "Client provides training materials"

  limitations:
    - max_time_commitment: "8 hours per quarter per key contact"
    - travel_costs: "Client responsibility if on-site required"
```

### 2.5 Article 30(3) — Additional Requirements for Critical Functions

If client classifies our services as supporting "critical or important function":

```yaml
additional_requirements_critical:
  art_30_3_a: "Full service level descriptions with quantitative targets"
  art_30_3_b: "Notice periods and reporting obligations"
  art_30_3_c: "Business contingency plans including ICT-specific requirements"
  art_30_3_d: "ICT security measures and testing participation"
  art_30_3_e: "UNRESTRICTED audit and access rights for client and their NCA"
  art_30_3_f: "Exit strategies with transition periods"
  art_30_3_g: "Participation in supervisory oversight activities"
```

#### 2.5.1 Detailed Art. 30(3) Implementation

```yaml
art_30_3_detailed_implementation:

  # Art. 30(3)(a) - Full Service Level Descriptions
  sla_targets:
    availability:
      trading_services: "99.9% monthly (43.8 min downtime max)"
      backtest_services: "99.5% monthly"
      api_services: "99.9% monthly"
      measurement: "External monitoring (UptimeRobot/Datadog)"
    performance:
      order_latency_p95: "<500ms"
      market_data_latency_p95: "<200ms"
      api_response_p95: "<1000ms"
    incident_response:
      critical: "15 min acknowledgment, 1h update"
      high: "30 min acknowledgment, 4h update"
      medium: "4h acknowledgment, 24h update"
    reporting:
      frequency: "Monthly SLA report"
      format: "PDF + JSON via client portal"

  # Art. 30(3)(b) - Notice Periods
  notice_periods:
    termination_by_client: "90 days minimum"
    termination_by_provider: "180 days minimum"
    termination_for_cause: "30 days (material breach)"
    service_changes:
      material_changes: "60 days advance notice"
      security_updates: "Immediate if critical"
      pricing_changes: "90 days advance notice"
    reporting_obligations:
      incident_reports: "Within 30 min (critical)"
      monthly_sla_reports: "By 5th business day"
      annual_security_report: "By January 31"

  # Art. 30(3)(c) - Business Contingency Plans
  business_contingency:
    documented_plans:
      - business_continuity_plan
      - disaster_recovery_plan
      - incident_response_plan
      - pandemic_response_plan
    testing_frequency:
      tabletop_exercises: "Quarterly"
      technical_dr_test: "Semi-annually"
      full_failover_test: "Annually"
    client_communication:
      bcp_summary: "Available on request"
      test_results: "Sanitized summary provided"
      recovery_time_objectives: "Documented in SLA"

  # Art. 30(3)(d) - ICT Security Measures
  security_measures_participation:
    provider_obligations:
      - maintain_security_certifications: "SOC2, ISO27001"
      - vulnerability_management: "Weekly scans, critical <24h remediation"
      - penetration_testing: "Annual third-party test"
      - security_awareness: "Quarterly training for all staff"
    client_cooperation:
      - joint_security_reviews: "Annual or upon request"
      - threat_intelligence_sharing: "As relevant"
      - incident_coordination: "Documented escalation paths"

  # Art. 30(3)(e) - UNRESTRICTED Audit Rights (CRITICAL)
  audit_access_rights:
    scope:
      premises_access: "With 5 business days notice"
      systems_access: "Read-only, escorted/supervised"
      personnel_access: "Key contacts for interviews"
      documentation_access: "All relevant policies, logs, reports"
    unrestricted_means:
      - "No cap on audit frequency for cause-based audits"
      - "No unreasonable limitations on scope"
      - "Access to subcontractor audit reports"
      - "Right to use external auditors"
    practical_limits:
      - "Reasonable notice (5 business days, waived for incidents)"
      - "Confidentiality of other clients' data"
      - "No access to proprietary source code unless relevant"
      - "Business hours unless emergency"
    pooled_audit_option:
      available: true
      conditions: "Per Art. 30(4), client may use pooled audits"
      our_support: "Annual third-party audit report provided"

  # Art. 30(3)(f) - Exit Strategies
  exit_strategy:
    transition_period: "Minimum 90 days, up to 180 for complex"
    data_export:
      formats: ["JSON", "CSV", "SQL dump"]
      scope: "All client data, configurations, trained models"
      timeline: "Export available within 5 business days"
    cooperation:
      knowledge_transfer: "Documentation + handover sessions"
      parallel_running: "Support dual operation during transition"
      post_termination_support: "30 days read-only access"
    no_vendor_lock_in:
      - "Standard data formats documented"
      - "API specifications published"
      - "No proprietary data encoding"

  # Art. 30(3)(g) - Supervisory Cooperation
  supervisory_oversight:
    client_nca_access:
      information_requests: "Response within 5 business days"
      on_site_inspection: "Cooperation required"
      personnel_interviews: "Make available as requested"
    cooperation_scope:
      - "Answer questions about services provided"
      - "Provide documentation about controls"
      - "Support client's regulatory examinations"
    limitations:
      - "Only regarding services to that specific client"
      - "Confidentiality of other clients maintained"
      - "Proprietary business info protected where possible"
```

#### 2.5.2 NCA Inspection Protocol

```yaml
nca_inspection_protocol:
  legal_basis: "Art. 30(3)(e) via client contract"

  upon_request:
    acknowledgment: "Within 24 hours"
    scheduling: "Within 5 business days"
    coordination: "Via client's compliance team"

  what_we_provide:
    - security_policies_and_procedures
    - incident_response_documentation
    - audit_logs_for_client_scope
    - business_continuity_plans
    - subcontractor_documentation
    - personnel_for_interviews

  what_we_protect:
    - other_clients_data: "Strict isolation"
    - proprietary_algorithms: "Unless directly relevant"
    - commercial_sensitive_info: "Reasonable protection"

  documentation:
    inspection_log: "Record all access and requests"
    evidence_provided: "Catalog of documents shared"
    follow_up_tracking: "Action items with deadlines"
```

### 2.6 What We Don't Take On

- Role of "financial entity" under DORA Article 2(1)(a-t)
- Direct regulatory reporting to NCAs
- TLPT coordination (client's responsibility, but we must cooperate)
- Register of Information submission (client submits, we provide data)
- Client's internal governance and risk management

### 2.7 CTPP Designation Risk — EXPANDED

**Current status:** We are NOT designated as Critical Third-Party Provider.

#### 2.7.1 Designation Triggers (Art. 31) — CORRECTED

ESAs assess CTPP designation based on **qualitative criteria** (not numeric thresholds):

| Art. 31(2) Criterion | What ESAs Assess | Our Current Status |
|---------------------|------------------|-------------------|
| **(a) Systemic impact** | Impact on financial services stability if we fail | LOW - no systemic dependency yet |
| **(b) Systemic importance of clients** | Whether GSIBs or O-SIIs rely on us | LOW - no GSIB/O-SII clients |
| **(c) Reliance for critical functions** | How many FEs use us for critical functions | LOW - few critical function designations |
| **(d) Substitutability** | Alternatives available, migration barriers | LOW - many alternatives exist |

**Note:** Art. 31 does NOT set numeric thresholds. ESAs use judgment-based assessment through Joint Committee and Oversight Forum.

#### 2.7.2 CTPP Obligations (Art. 33-44)

If designated as CTPP, we would face:

```yaml
ctpp_direct_obligations:
  oversight:
    lead_overseer: "Designated ESA (EBA/ESMA/EIOPA)"
    oversight_fee: "Annual fee based on turnover"
    reporting: "Regular supervisory reporting"

  operational:
    resilience_testing: "Mandatory annual testing"
    incident_reporting: "Direct to Lead Overseer"
    governance: "Board-level accountability"

  inspections:
    on_site: "Lead Overseer can inspect premises"
    access_rights: "Full access to systems and data"
    third_party_audits: "May require external audits"

  enforcement:
    recommendations: "Binding recommendations possible"
    penalties: "Fines up to 1% of global turnover"
    suspension: "Services can be suspended"
```

#### 2.7.3 CTPP Preparedness Checklist

| Preparation Item | Status | Priority |
|------------------|--------|----------|
| Client concentration tracking | ⚠️ Manual | HIGH |
| ESA communication channel | ❌ Not established | MEDIUM |
| Enhanced incident reporting (direct) | ❌ Not implemented | HIGH |
| Governance documentation | ⚠️ Basic | MEDIUM |
| Fee reserve allocation | ❌ Not planned | LOW |
| Substitutability assessment | ❌ Not done | MEDIUM |

#### 2.7.4 Monitoring and Qualitative Triggers — CORRECTED

```yaml
ctpp_monitoring:
  quarterly_review:
    - count_eu_financial_entity_clients
    - identify_gsib_osii_clients       # Global/Other Systemically Important
    - assess_critical_function_designations
    - evaluate_substitutability_landscape
    - review_esa_public_communications

  # Qualitative warning triggers (not numeric thresholds)
  warning_triggers:
    any_gsib_client: true              # Any Global Systemically Important Bank
    any_osii_client: true              # Any Other Systemically Important Institution
    multiple_critical_designations: 3  # 3+ clients designate us as critical
    sector_concentration: "30%"        # >30% clients in single sector
    limited_alternatives: "market_assessment" # Regular assessment needed

  escalation_actions:
    on_warning_trigger:
      - notify_board_immediately
      - engage_external_legal_counsel
      - assess_ctpp_readiness_gaps
      - consider_proactive_esa_engagement
    ongoing:
      - maintain_ctpp_readiness_package
      - annual_substitutability_review
      - track_esa_guidance_updates
```

**Mitigation Strategy:**
- Monitor client composition qualitatively, not just quantity
- If ANY GSIB/O-SII becomes a client → immediately assess CTPP risk
- Maintain documentation as if designation is possible
- Consider voluntary ESA engagement if risk indicators increase

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

#### 5.2.1 7-Year Log Retention Implementation — NEW

```yaml
log_retention_implementation:
  tiered_storage:
    hot_tier:
      storage: "Elasticsearch / OpenSearch"
      retention: "90 days"
      purpose: "Active search and analysis"
      monthly_cost_per_tb: "€150-300"
    warm_tier:
      storage: "S3 Standard / Azure Blob"
      retention: "1 year"
      purpose: "Occasional access for investigations"
      monthly_cost_per_tb: "€20-40"
    cold_tier:
      storage: "S3 Glacier Deep Archive / Azure Archive"
      retention: "6 years (to complete 7-year total)"
      purpose: "Regulatory compliance, rare access"
      monthly_cost_per_tb: "€1-2"

  tamper_protection:
    mechanism: "Hash chain + Object Lock"
    implementation:
      - daily_hash: "SHA-256 hash of day's logs"
      - chain_verification: "Each day's hash includes previous day's hash"
      - immutable_storage: "S3 Object Lock in GOVERNANCE mode"
      - retention_lock: "7-year retention policy enforced"
    verification:
      frequency: "Quarterly integrity check"
      process: "Automated hash chain verification script"
      alert_on_failure: true

  retrieval_sla:
    hot_tier: "Immediate (seconds)"
    warm_tier: "Minutes"
    cold_tier: "12-48 hours (Glacier restore)"
    audit_request_sla: "Evidence available within 24 hours"

  cost_estimate_7_years:
    assumptions:
      audit_log_growth: "50 GB/month"
      compression_ratio: "10:1"
    total_uncompressed: "4.2 TB"
    total_compressed: "420 GB"
    estimated_cost:
      year_1: "€500 (mostly hot/warm)"
      years_2_7: "€50/year (cold storage)"
      total_7_years: "€800-1000"

  implementation_steps:
    - configure_log_lifecycle_policies
    - implement_hash_chain_module
    - enable_s3_object_lock
    - setup_integrity_verification_cron
    - document_retrieval_procedures
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

#### 5.3.1 RTO/RPO Technical Feasibility Assessment — NEW

```yaml
technical_feasibility:
  tier_1_rpo_15min:
    requirement: "Maximum 15 minutes data loss for trading state"
    implementation_options:
      option_a:
        name: "PostgreSQL Streaming Replication"
        description: "Hot standby with synchronous replication"
        achievable_rpo: "Near-zero (seconds)"
        monthly_cost: "€500-1000 (standby instance)"
        complexity: "Medium"
        recommended: true
      option_b:
        name: "AWS RDS Multi-AZ"
        description: "Managed synchronous replication"
        achievable_rpo: "Near-zero"
        monthly_cost: "€300-600 (Multi-AZ premium)"
        complexity: "Low"
        recommended: true
      option_c:
        name: "Point-in-time recovery only"
        description: "WAL archiving every 5 minutes"
        achievable_rpo: "5-15 minutes"
        monthly_cost: "€50-100 (S3 storage)"
        complexity: "Low"
        recommended: false  # Does not meet 15min requirement reliably

  tier_1_rto_1h:
    requirement: "Trading services resume within 1 hour"
    implementation_options:
      option_a:
        name: "Hot Standby with Auto-Failover"
        description: "Pre-provisioned standby, automated DNS/LB failover"
        achievable_rto: "5-15 minutes"
        monthly_cost: "€2000-4000 (standby infrastructure)"
        complexity: "High"
        recommended_for: "Enterprise tier with strict SLA"
      option_b:
        name: "Warm Standby with Manual Failover"
        description: "Standby infrastructure, manual switchover"
        achievable_rto: "30-60 minutes"
        monthly_cost: "€1000-2000"
        complexity: "Medium"
        recommended_for: "Standard tier"
      option_c:
        name: "Cold Recovery from Backup"
        description: "Restore from backup to new infrastructure"
        achievable_rto: "2-4 hours"
        monthly_cost: "€200-500 (backup storage only)"
        complexity: "Low"
        recommended_for: "Non-critical services only"

  recommended_configuration:
    trading_services:
      target: "RTO 1h, RPO 15min"
      implementation: "Option A (Hot Standby) or Option B (Warm Standby)"
      estimated_monthly_cost: "€1500-3000"
      justification: "Critical for financial operations"
    backtest_services:
      target: "RTO 4h, RPO 24h"
      implementation: "Cold Recovery acceptable"
      estimated_monthly_cost: "€200-500"

  infrastructure_requirements:
    multi_region:
      primary: "eu-central-1 (Frankfurt)"
      secondary: "eu-west-1 (Ireland)"
      rationale: "GDPR compliance, geographic redundancy"
    database:
      - postgresql_streaming_replication: true
      - automated_failover: "Patroni or RDS Multi-AZ"
    monitoring:
      - health_checks: "Every 30 seconds"
      - failover_trigger: "3 consecutive failures"
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

### 5.4.1 Incident Notification Operations — NEW

```yaml
incident_notification_operations:
  # =========================================================================
  # ON-CALL STRUCTURE (Required for <30min notification SLA)
  # =========================================================================
  on_call:
    rotation: "Weekly rotation"
    coverage: "24/7/365"
    team_size_minimum: 4  # For sustainable rotation
    escalation_path:
      level_1:
        role: "On-call Engineer"
        response_time: "15 minutes"
        responsibilities:
          - acknowledge_alert
          - initial_triage
          - start_incident_channel
      level_2:
        role: "Engineering Lead"
        response_time: "30 minutes"
        trigger: "Severity >= High OR L1 escalation"
        responsibilities:
          - technical_decision_making
          - client_notification_approval
          - resource_coordination
      level_3:
        role: "CTO / VP Engineering"
        response_time: "60 minutes"
        trigger: "Severity = Critical OR major client impact"
        responsibilities:
          - executive_decisions
          - external_communications
          - regulatory_coordination

  # =========================================================================
  # TOOLING REQUIREMENTS
  # =========================================================================
  tooling:
    alerting_platform:
      options: ["PagerDuty", "Opsgenie", "VictorOps"]
      requirements:
        - mobile_app_push
        - sms_backup
        - phone_escalation
        - schedule_management
      estimated_cost: "€500-1500/month"

    incident_management:
      options: ["PagerDuty + Slack", "Statuspage", "incident.io"]
      requirements:
        - incident_timeline_tracking
        - stakeholder_notifications
        - post_incident_reports

    status_page:
      purpose: "Real-time client communication"
      options: ["Statuspage", "Instatus", "Custom"]
      features:
        - public_status_dashboard
        - incident_updates
        - maintenance_windows
        - subscriber_notifications

  # =========================================================================
  # NOTIFICATION WORKFLOWS
  # =========================================================================
  notification_workflows:
    critical_incident:
      timeline:
        "T+0": "Alert triggered by monitoring"
        "T+5min": "On-call acknowledges"
        "T+10min": "Incident classified and channel opened"
        "T+15min": "Initial client notification drafted"
        "T+20min": "L2 reviews and approves notification"
        "T+30min": "All affected clients notified"
      channels:
        - webhook_to_client_systems
        - email_to_registered_contacts
        - status_page_update
        - optional_sms_for_critical

    high_incident:
      timeline:
        "T+0": "Alert triggered"
        "T+15min": "On-call acknowledges and triages"
        "T+30min": "Incident classified"
        "T+45min": "Client notification drafted"
        "T+60min": "All affected clients notified"

  # =========================================================================
  # STAFFING REQUIREMENTS
  # =========================================================================
  staffing_requirements:
    minimum_for_24_7:
      engineers: 4
      rationale: "4 engineers = 1 week on-call each per month, sustainable"
      compensation: "On-call allowance + incident response overtime"

    alternative_managed_noc:
      description: "Outsourced 24/7 NOC for initial triage"
      cost: "€3000-5000/month"
      limitations:
        - "Initial triage only"
        - "Escalates to internal team for resolution"
        - "May increase response time by 5-10 minutes"
      suitable_for: "Startups without 4+ engineers"

  # =========================================================================
  # SLA TRACKING
  # =========================================================================
  sla_tracking:
    metrics:
      - time_to_acknowledge
      - time_to_classify
      - time_to_notify_clients
      - time_to_resolve
      - time_to_post_incident_report

    targets:
      critical:
        acknowledge: "5 minutes"
        notify_clients: "30 minutes"
        post_incident_report: "24 hours"
      high:
        acknowledge: "15 minutes"
        notify_clients: "60 minutes"
        post_incident_report: "72 hours"

    reporting:
      internal: "Weekly incident metrics review"
      client_facing: "Monthly SLA report"
```

### 5.4.2 Client SLA Tiers — NEW

Different clients require different service levels. Define tiered SLAs with realistic infrastructure backing:

```yaml
client_sla_tiers:
  # =========================================================================
  # STANDARD TIER (Default for all clients)
  # =========================================================================
  standard:
    availability: "99.5%"
    rto: "4 hours"
    rpo: "1 hour"
    incident_notification: "2 hours"
    support_hours: "Business hours (EU timezone)"

    infrastructure:
      deployment: "Single region (EU-WEST-1)"
      database: "Primary + async replica"
      backups: "Every 4 hours"
      monitoring: "5-minute intervals"

    cost_multiplier: 1.0x
    suitable_for: "Non-critical functions, retail traders, prop firms"

  # =========================================================================
  # PROFESSIONAL TIER (For regulated clients with important functions)
  # =========================================================================
  professional:
    availability: "99.9%"
    rto: "1 hour"
    rpo: "15 minutes"
    incident_notification: "30 minutes"
    support_hours: "Extended (06:00-22:00 CET)"

    infrastructure:
      deployment: "Multi-AZ (EU-WEST-1 a/b/c)"
      database: "Primary + sync replica + async DR"
      backups: "Every 15 minutes (continuous for critical)"
      monitoring: "1-minute intervals with auto-alerting"

    cost_multiplier: 2.0x
    suitable_for: "Important functions, asset managers, hedge funds"

  # =========================================================================
  # ENTERPRISE TIER (For clients with critical functions)
  # =========================================================================
  enterprise:
    availability: "99.95%"
    rto: "15 minutes"
    rpo: "5 minutes"
    incident_notification: "15 minutes"
    support_hours: "24/7/365"

    infrastructure:
      deployment: "Multi-region (EU-WEST-1 + EU-CENTRAL-1)"
      database: "Multi-region sync replication"
      backups: "Continuous with point-in-time recovery"
      monitoring: "Real-time with predictive alerting"

    additional:
      - dedicated_instance_option
      - custom_integrations
      - quarterly_resilience_reviews
      - annual_joint_dr_testing

    cost_multiplier: 4.0x
    suitable_for: "Critical functions, banks, CASPs with significant AUM"
```

### 5.4.2 Infrastructure Requirements by Tier

| Tier | Multi-AZ | Multi-Region | Sync Replication | DR Tested | Est. Monthly Cost |
|------|----------|--------------|------------------|-----------|-------------------|
| Standard | ❌ | ❌ | ❌ | Quarterly | €500-2,000 |
| Professional | ✅ | ❌ | Partial | Monthly | €2,000-5,000 |
| Enterprise | ✅ | ✅ | ✅ | Weekly | €5,000-15,000 |

**CRITICAL**: RTO/RPO claims MUST be backed by documented infrastructure and tested procedures.

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

### 5.8 Subcontractor Documentation — EXPANDED

```yaml
subcontractor_management:
  # =========================================================================
  # DOCUMENTATION REQUIREMENTS (per Art. 30(2)(b), CIR 2024/2956 B_99.01)
  # =========================================================================
  documentation_required:
    - subcontractor_name_and_lei
    - services_provided
    - data_processing_locations
    - data_storage_locations
    - security_certifications
    - subcontractor_chain (if any)
    - data_access_scope
    - contract_reference

  # =========================================================================
  # COMPLETE SUBCONTRACTOR INVENTORY (ITS-aligned)
  # =========================================================================
  current_subcontractors:

    # --- TIER 1: Cloud Infrastructure (Critical) ---
    aws:
      legal_name: "Amazon Web Services EMEA SARL"
      lei: "ZXTILKJGXN5HNWDRYU4"  # AWS parent (Amazon.com, Inc)
      subcontractor_type: "cloud_infrastructure"
      chain_level: 1
      services_provided:
        - "Compute (EC2, ECS, Lambda)"
        - "Storage (S3, EBS)"
        - "Database (RDS PostgreSQL)"
        - "Networking (VPC, CloudFront)"
      data_processing_locations: ["EU-WEST-1 (Ireland)", "EU-CENTRAL-1 (Frankfurt)"]
      data_storage_locations: ["EU-WEST-1 (Ireland)"]
      has_data_access: true
      data_types_accessed: ["All platform data (encrypted)"]
      certifications:
        - "SOC 1/2/3"
        - "ISO 27001"
        - "ISO 27017"
        - "ISO 27018"
        - "C5 (Germany)"
        - "GDPR compliant"
      contract_reference: "AWS Enterprise Agreement"
      is_material: true
      supports_critical_functions: true
      substitutability: "medium"  # GCP/Azure available
      last_audit_date: "2024-11-01"
      next_review_date: "2025-11-01"

    # --- TIER 1: Market Data Providers ---
    polygon:
      legal_name: "Polygon.io, Inc."
      lei: "TBD"  # ACTION: Obtain LEI or use EIN
      alternative_id: "EIN: 84-3159622"
      alternative_id_type: "US_EIN"
      subcontractor_type: "data_provider"
      chain_level: 1
      services_provided:
        - "Real-time stock market data"
        - "Historical price data"
        - "Options data"
      data_processing_locations: ["US (New York)"]
      data_storage_locations: ["US"]
      has_data_access: false  # Market data only, no client data
      data_types_accessed: []
      certifications:
        - "SOC 2 Type II"
      contract_reference: "Polygon Enterprise Agreement"
      is_material: false
      supports_critical_functions: false
      substitutability: "easy"  # Many alternatives
      last_audit_date: "N/A"
      next_review_date: "2025-06-01"

    alpaca:
      legal_name: "AlpacaDB, Inc."
      lei: "TBD"  # ACTION: Obtain LEI
      alternative_id: "EIN: 82-1913791"
      alternative_id_type: "US_EIN"
      subcontractor_type: "data_provider"
      chain_level: 1
      services_provided:
        - "Brokerage API (order execution)"
        - "Market data"
        - "Account management"
      data_processing_locations: ["US"]
      data_storage_locations: ["US"]
      has_data_access: true  # Client API keys stored
      data_types_accessed: ["Client brokerage credentials (encrypted)"]
      certifications:
        - "SEC/FINRA registered broker"
        - "SOC 2"
      contract_reference: "Alpaca API Agreement"
      is_material: true
      supports_critical_functions: true  # Trading execution
      substitutability: "medium"
      last_audit_date: "N/A"
      next_review_date: "2025-06-01"

    binance:
      legal_name: "Binance Holdings Limited"
      lei: "TBD"
      subcontractor_type: "data_provider"
      chain_level: 1
      services_provided:
        - "Crypto market data"
        - "Crypto trading API"
      data_processing_locations: ["Global (various jurisdictions)"]
      data_storage_locations: ["Variable"]
      has_data_access: true  # Client API keys
      data_types_accessed: ["Client exchange credentials (encrypted)"]
      certifications:
        - "Variable by jurisdiction"
      contract_reference: "Binance API Terms"
      is_material: true
      supports_critical_functions: true  # Crypto trading
      substitutability: "medium"  # Kraken, Coinbase alternatives
      last_audit_date: "N/A"
      next_review_date: "2025-06-01"
      special_notes: "Regulatory status varies by jurisdiction"

    # --- TIER 2: Monitoring & Operations ---
    datadog:
      legal_name: "Datadog, Inc."
      lei: "549300F6JNO0KRPO1K63"
      subcontractor_type: "monitoring"
      chain_level: 1
      services_provided:
        - "Application monitoring"
        - "Log management"
        - "Alerting"
      data_processing_locations: ["US", "EU (Germany)"]
      data_storage_locations: ["EU (Germany) - configured"]
      has_data_access: true  # Logs may contain metadata
      data_types_accessed: ["System logs", "Performance metrics"]
      certifications:
        - "SOC 2 Type II"
        - "ISO 27001"
        - "GDPR compliant"
      contract_reference: "Datadog Enterprise Agreement"
      is_material: false
      supports_critical_functions: false
      substitutability: "easy"

    sentry:
      legal_name: "Functional Software, Inc. (Sentry)"
      lei: "TBD"
      subcontractor_type: "monitoring"
      chain_level: 1
      services_provided:
        - "Error tracking"
        - "Performance monitoring"
      data_processing_locations: ["US", "EU option available"]
      has_data_access: true  # Error context may contain data
      certifications:
        - "SOC 2 Type II"
        - "GDPR compliant"
      is_material: false
      supports_critical_functions: false

    # --- TIER 2: Authentication ---
    auth0_clerk:
      legal_name: "Auth0 Inc. (Okta) / Clerk Inc."
      lei: "TBD"  # Okta LEI: 549300N8BTFTU58UJ747
      subcontractor_type: "security_services"
      chain_level: 1
      services_provided:
        - "User authentication"
        - "Identity management"
      data_processing_locations: ["US", "EU option"]
      has_data_access: true  # User credentials
      data_types_accessed: ["User emails", "Authentication tokens"]
      certifications:
        - "SOC 2 Type II"
        - "ISO 27001"
      is_material: true
      supports_critical_functions: true  # Authentication is critical

    # --- TIER 2: Payments ---
    stripe:
      legal_name: "Stripe, Inc."
      lei: "549300HZVWQT6W3NQC36"
      subcontractor_type: "payment_services"
      chain_level: 1
      services_provided:
        - "Payment processing"
        - "Subscription management"
      data_processing_locations: ["US", "EU (Ireland)"]
      has_data_access: true  # Payment info
      data_types_accessed: ["Payment card tokens", "Billing info"]
      certifications:
        - "PCI DSS Level 1"
        - "SOC 2 Type II"
        - "ISO 27001"
      is_material: true
      supports_critical_functions: false  # Not trading-critical

  # =========================================================================
  # CLIENT DISCLOSURE
  # =========================================================================
  client_disclosure:
    subprocessor_list:
      availability: "On request via client portal"
      format: "PDF + JSON (ITS B_99.01 compatible)"
      update_frequency: "Quarterly + on material change"
    notification_of_changes:
      advance_notice: "30 days minimum"
      material_changes: "60 days for critical function subcontractors"
      channels: ["Email", "Client portal", "Contract amendment"]
    objection_right:
      standard_contracts: "Notification only"
      critical_function_contracts: "Consent required for changes affecting critical services"

  # =========================================================================
  # ACTION ITEMS
  # =========================================================================
  action_items:
    - action: "Obtain LEI for Polygon.io"
      priority: "MEDIUM"
      deadline: "2025-Q1"
    - action: "Obtain LEI for Alpaca"
      priority: "MEDIUM"
      deadline: "2025-Q1"
    - action: "Configure Sentry EU data residency"
      priority: "HIGH"
      deadline: "2025-01-31"
    - action: "Document Auth0/Clerk selection"
      priority: "MEDIUM"
      deadline: "2025-Q1"
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

### 5.10 Exit Strategy Testing — NEW

Per Art. 28(8), exit strategies must be "comprehensive, documented and... tested where appropriate."

```yaml
exit_strategy_testing:
  # =========================================================================
  # TESTING SCHEDULE
  # =========================================================================
  schedule:
    frequency: "Annual minimum"
    critical_providers: "Semi-annual testing"
    trigger_based:
      - "Material change to provider services"
      - "Provider acquisition/merger"
      - "New critical function designation"
      - "Failed previous test"

  # =========================================================================
  # TEST SCENARIOS
  # =========================================================================
  test_scenarios:
    scenario_1_planned_termination:
      description: "Orderly transition to alternative provider"
      scope:
        - "Data export completeness"
        - "API compatibility with alternative"
        - "Timeline validation"
      test_method: "Tabletop exercise + partial data export"
      frequency: "Annual"

    scenario_2_provider_failure:
      description: "Immediate provider failure requiring rapid transition"
      scope:
        - "Backup data availability"
        - "Alternative provider activation time"
        - "Service continuity during transition"
      test_method: "Simulation with standby environment"
      frequency: "Annual"

    scenario_3_data_export:
      description: "Validate data export functionality"
      scope:
        - "Export all client data"
        - "Verify format compliance (JSON/CSV)"
        - "Validate data integrity"
        - "Measure export time"
      test_method: "Actual export to test environment"
      frequency: "Quarterly"

    scenario_4_api_migration:
      description: "Client migration to alternative platform"
      scope:
        - "API mapping documentation"
        - "Integration test with mock client"
        - "Breaking changes identification"
      test_method: "Technical walkthrough + integration test"
      frequency: "Annual"

  # =========================================================================
  # TEST COMPONENTS
  # =========================================================================
  test_components:
    data_export:
      validation_checks:
        - "All data types exported"
        - "No data corruption (checksum validation)"
        - "Format meets specifications"
        - "Export completes within SLA (5 business days)"
      success_criteria:
        - "100% of client data exportable"
        - "Data integrity verified"
        - "Export time < 24 hours for typical client"

    documentation:
      validation_checks:
        - "Exit procedures documented"
        - "Contact information current"
        - "Alternative providers identified"
        - "Timeline realistic"
      success_criteria:
        - "Documentation complete and current"
        - "Reviewed within last 12 months"

    alternative_providers:
      validation_checks:
        - "At least 1 alternative identified per critical service"
        - "Alternative capable of receiving data"
        - "Commercial terms understood"
      success_criteria:
        - "Alternatives evaluated and scored"
        - "No single point of failure"

  # =========================================================================
  # TEST DOCUMENTATION
  # =========================================================================
  test_documentation:
    required_records:
      - test_date: "ISO 8601 format"
      - test_type: "Scenario reference"
      - participants: "List of involved personnel"
      - test_results: "Pass/Fail with details"
      - identified_gaps: "Issues found"
      - remediation_actions: "Actions to address gaps"
      - remediation_deadline: "Target date for fixes"
      - sign_off: "Approver name and date"

    retention: "7 years (aligned with audit log retention)"
    availability: "Available for client audits"

  # =========================================================================
  # CURRENT TEST STATUS
  # =========================================================================
  current_status:
    last_test_date: "TBD - Not yet conducted"
    next_scheduled_test: "2025-Q1"
    test_results: []
    gaps_identified: []

  # =========================================================================
  # ACTION ITEMS
  # =========================================================================
  action_items:
    - action: "Conduct first exit strategy test"
      priority: "HIGH"
      deadline: "2025-Q1"
      owner: "Operations Team"
    - action: "Document alternative providers for each service"
      priority: "MEDIUM"
      deadline: "2025-Q1"
    - action: "Create data export validation script"
      priority: "MEDIUM"
      deadline: "2025-Q1"
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
