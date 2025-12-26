# DORA Operational Resilience Plan

**Version**: 2.5
**Date**: 2025-12-19
**Status**: Phase 1 Implementation — Toolkit implemented per internal CI tests (no independent third-party audit conducted; verify via CI test reports)
**Revision**: Due diligence audit corrections (SLA disclaimers, infrastructure validation requirements)

> **Important**: This document describes the DORA compliance toolkit provided to clients. The status "Toolkit Ready" means all planned tools and controls have been implemented and passed internal automated tests. This does NOT constitute certification, independent audit, or guarantee of regulatory compliance. Test results are internal CI outputs; clients must conduct their own compliance assessment with qualified advisors.

> **Note (v2.4)**: References to `services/compliance/` are historical. MiFID II compliance modules have been reorganized:
> - `services/core/risk_controls/` (universal risk controls, audit_trail, bcp)
> - `services/algo_integration/` (B2B compliance toolkit)
> - `services/archive/mifid_financial_entity/` (archived Investment Firm modules)

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

## Changelog v2.1 (Audit Fixes)

| # | Issue | DORA Reference | Fix |
|---|-------|----------------|-----|
| 13 | Training participation passive only | Art. 30(2)(i), Art. 13(6) | Enhanced active participation commitment |
| 14 | Prior consent vs notification conflated | Art. 30(3)(j) | Added explicit prior consent workflow |
| 15 | Multi-client breach procedure missing | Art. 30(2)(f) | Added Section 6.9 Coordinated Multi-Client Notification |
| 16 | Cross-region failover undocumented | RTO commitment | Added Section 5.4.3 Cross-Region Failover |
| 17 | NCA jurisdiction unclear | Art. 30(3)(e) | Added Section 6.10 NCA Jurisdiction Matrix |
| 18 | Data escrow not explicit | Art. 30(2)(d) | Enhanced insolvency protection |
| 19 | Exit plan testing not required | Art. 28(8) | Strengthened Section 5.10 |
| 20 | CTPP client mix tracking manual | Art. 31 | Added automated monitoring in Section 2.7.5 |

## Changelog v2.2 (Operational Validation)

| # | Issue | DORA Reference | Fix |
|---|-------|----------------|-----|
| 21 | Prior written approval workflow missing | Art. 30(3)(j)(i) | Added Section 5.8.1 Subcontracting Approval Workflow |
| 22 | Archive list too aggressive | Art. 30(4), Art. 31 | Revised: ctpp_oversight → KEEP, pooled_testing → ADAPT |
| 23 | RTO/RPO contractual risk | Art. 30(3)(a) | Added Section 5.4.4 Contractual SLA Guardrails |
| 24 | On-call capacity unvalidated | Art. 30(2)(f) | Added Section 5.4.5 Notification SLA Tiers by Capacity |
| 25 | Data localization not configurable | Art. 30(2)(b) | Added Section 5.11 Data Residency Configuration |
| 26 | Pre-contractual portal missing | Art. 28(7) | Enhanced Section 6.6 with implementation details |
| 27 | Insurance requirements missing | Industry practice | Added Section 6.11 Insurance & Indemnification |
| 28 | Subcontractor incident flow missing | Art. 30(2)(f) | Added Section 5.8.2 Subcontractor Incident Escalation |
| 29 | Pooled audit support undefined | Art. 30(4) | Added Section 6.12 Pooled Audit Framework |

## Changelog v2.4 (Due Diligence Audit — Dec 2025)

| # | Issue | Reference | Fix |
|---|-------|-----------|-----|
| 78 | SLA tier targets could be read as commitments | Art. 30(3)(a) | Added "Design target" prefix + "pending validation" + "actual SLA per executed agreement" to all Professional/Enterprise tier metrics (Section 5.4.2) |
| 79 | Infrastructure capabilities presented as current | Reality check | Changed infrastructure descriptions to "Target:" prefix + "(pending implementation)" for all non-validated capabilities |
| 80 | 24/7 support hours without capacity validation | Operations | Added "(pending 4+ FTE on-call team; actual coverage per executed agreement)" disclaimer to enterprise tier support_hours |

## Changelog v2.3 (Phase 1 Implementation Complete)

| # | Deliverable | DORA Reference | Implementation |
|---|-------------|----------------|----------------|
| 30 | Art. 30(2) contract template | Art. 30(2)(a-i) | `docs/contracts/DORA_CONTRACT_TEMPLATE_ART_30_2.md` |
| 31 | Art. 30(3) critical function addendum | Art. 30(3)(a-j) | `docs/contracts/DORA_CRITICAL_FUNCTION_ADDENDUM_ART_30_3.md` |
| 32 | Shared responsibility model | Art. 28, Art. 30 | `docs/SHARED_RESPONSIBILITY.md` |
| 33 | Subcontractor register | Art. 30(3)(j) | `docs/contracts/SUBCONTRACTOR_REGISTER.md` |
| 34 | EU data residency configuration | Art. 30(2)(b) | `docs/contracts/EU_DATA_RESIDENCY.md` |
| 35 | Insurance & indemnification | Industry practice | `docs/contracts/INSURANCE_REQUIREMENTS.md` |
| 36 | Trust center (pre-contractual) | Art. 28(7) | `docs/security/TRUST_CENTER.md` |
| 37 | On-call capacity validation | Art. 30(2)(f) | `docs/operations/ON_CALL_CAPACITY_VALIDATION.md` |
| 38 | SLA guardrails module | Art. 30(3)(a) | `services/dora_integration/contracts/sla_guardrails.py` (53 tests in `tests/dora_integration/contracts/test_sla_guardrails.py`) |
| 39 | Pooled audit support module | Art. 30(4) | `services/dora_integration/due_diligence/pooled_audit_support.py` (27 tests in `tests/dora_integration/due_diligence/test_phase1_smoke.py`) |
| 40 | Archive non-applicable modules | N/A | `services/archive/dora_not_applicable/` |

**Phase 1 Completion Summary:**
- All 15 work blocks completed
- Automated tests for DORA-related modules implemented
- 2 new Python modules with associated tests
- 8 new documentation files
- Toolkit ready for client use (not independently audited; test coverage claims refer to internal automated tests, not regulatory certification)

---

## 1. Executive Summary: Repository Analysis

### 1.1 Discovered Components

| Area | Files/Modules | Maturity |
|------|---------------|----------|
| **DORA Services** | `services/dora/` - 40+ modules | HIGH (needs repositioning) |
| **DORA Configs** | `configs/dora/`, `config/dora/` | MEDIUM |
| **DORA Tests** | `tests/dora/` - 12+ test files | MEDIUM |
| **Operations Runbook** | [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md) | HIGH |
| **Recovery Procedures** | [RECOVERY_PROCEDURES.md](RECOVERY_PROCEDURES.md) | HIGH |
| **Service Dependency Map** | [SERVICE_DEPENDENCY_MAP.md](SERVICE_DEPENDENCY_MAP.md) | HIGH |
| **Cybersecurity Framework** | [CYBERSECURITY_FRAMEWORK.md](CYBERSECURITY_FRAMEWORK.md) (NIST CSF 2.0) | HIGH |
| **SOC2 Roadmap** | [SOC2_ROADMAP.md](SOC2_ROADMAP.md) | HIGH |
| **Healthcheck** | [services/healthcheck.py](../services/healthcheck.py) | MEDIUM |
| **Kill Switch** | [services/ops_kill_switch.py](../services/ops_kill_switch.py) | HIGH |
| **Secure Logging** | [services/secure_logging.py](../services/secure_logging.py) | MEDIUM |
| **Monitoring** | [services/monitoring.py](../services/monitoring.py) | MEDIUM |
| **MiFID II / Core Risk Controls** | `services/core/risk_controls/` + `services/algo_integration/` | HIGH |
| **CI/CD** | [.github/workflows/build-and-test.yml](../.github/workflows/build-and-test.yml) | MEDIUM |

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

ALL contracts with EU regulated clients MUST include these **9 mandatory clauses** (per official DORA text):

```yaml
mandatory_contract_clauses:
  # Basic service terms (a-c)
  art_30_2_a: "Clear and complete description of all functions and ICT services to be provided, including subcontracting permissions and conditions"
  art_30_2_b: "Locations (regions/countries) where data will be processed and stored, advance notice requirements for location changes"
  art_30_2_c: "Provisions on availability, authenticity, integrity and confidentiality of data, including personal data protection"

  # Data and service management (d-f)
  art_30_2_d: "Provisions on data access, recovery and return in easily accessible format upon termination, insolvency, or resolution of ICT provider"
  art_30_2_e: "Service level descriptions with quantitative and qualitative performance targets, updates and revisions"
  art_30_2_f: "Obligation to provide assistance in case of ICT incidents at no additional cost or at predetermined cost"

  # Regulatory and termination (g-i)
  art_30_2_g: "Obligation to fully cooperate with competent authorities and resolution authorities of the financial entity"
  art_30_2_h: "Termination rights and related minimum notice periods for contract termination, in line with supervisory expectations"
  art_30_2_i: "Conditions for ICT provider participation in financial entity's security awareness programmes and digital operational resilience training (per Art. 13(6))"
```

**CRITICAL NOTES**:
1. Art. 30(2)(d) is often missed — data access/recovery/return mechanisms are MANDATORY for ALL contracts
2. Art. 30(2)(h) and (i) are also frequently overlooked but are mandatory
3. Resilience testing participation is a **30(3)** requirement (critical functions only), not 30(2)

**Art. 30(2)(d) Implementation** — Data Access, Recovery and Return:
```yaml
data_access_recovery_clause:
  purpose: |
    Ensure client can access, recover and retrieve all their data upon:
    - Contract termination (planned or unplanned)
    - Provider insolvency or resolution
    - Regulatory requirement

  data_export_formats:
    primary: "JSON (structured, machine-readable)"
    secondary: "CSV (tabular data)"
    database: "SQL dump (PostgreSQL compatible)"
    models: "ONNX format for trained AI/ML models"
    documentation: "PDF/Markdown"

  data_scope:
    included:
      - "All trading strategies and configurations"
      - "Backtest results and performance history"
      - "Trained ML/RL models and weights"
      - "User configurations and preferences"
      - "Audit logs (client-specific)"
      - "API integration configurations"
    excluded:
      - "Platform source code (proprietary)"
      - "Other clients' data"
      - "Aggregated/anonymized platform metrics"

  timeline_commitments:
    standard_termination:
      export_request_response: "24 hours"
      data_package_ready: "5 business days"
      download_availability: "30 days post-termination"
    urgent_termination:
      export_request_response: "4 hours"
      data_package_ready: "48 hours"
      download_availability: "14 days"
    insolvency_scenario:
      data_escrow: "Available immediately via escrow provider"
      direct_access: "Within 72 hours of insolvency notice"

  technical_provisions:
    api_access: "REST API for programmatic data export"
    bulk_download: "Secure HTTPS download links"
    verification: "SHA-256 checksums for all exports"
    encryption: "AES-256 encryption for transit and at-rest"

  insolvency_protection:
    data_escrow:
      provider: "To be designated (e.g., Iron Mountain, AWS Glacier)"
      update_frequency: "Weekly full backup to escrow"
      access_trigger: "Insolvency filing or 30-day non-response"
    contractual_safeguards:
      - "Data classified as client property, not platform asset"
      - "Explicit carve-out from bankruptcy estate"
      - "Priority access rights in insolvency proceedings"

  cost_provisions:
    standard_export: "Included in subscription (no additional cost)"
    expedited_export: "Predetermined fee schedule in contract"
    extended_retention: "Per-month fee for retention beyond 30 days"
```

**Art. 30(2)(i) Implementation** — Training Participation (ENHANCED v2.1):
```yaml
training_participation_clause:
  # Reference: DORA Art. 30(2)(i) + Art. 13(6)
  # Art. 13(6): Financial entities shall develop ICT security awareness programmes
  # and digital operational resilience training as compulsory modules for staff

  commitment: |
    Provider shall ACTIVELY participate in Client's ICT security awareness
    programmes and digital operational resilience training per Art. 13(6).
    This is a MANDATORY contract clause, not optional cooperation.

  # =========================================================================
  # ACTIVE PARTICIPATION (not just "available upon request")
  # =========================================================================
  active_participation:
    security_awareness_programs:
      frequency: "Annual minimum, more frequent upon request"
      format: "Live session (remote or on-site)"
      our_contribution:
        - "Present platform security architecture"
        - "Explain incident response procedures"
        - "Review shared responsibility model"
        - "Q&A with client security team"
      personnel: "Security Lead + relevant technical contacts"

    resilience_training_exercises:
      types:
        - tabletop_exercises: "Scenario-based discussions"
        - dr_drills: "Joint disaster recovery exercises"
        - incident_simulations: "Simulated security incidents"
      our_role:
        - "Participate in scenario planning"
        - "Execute provider-side procedures during drill"
        - "Provide debrief and lessons learned"
      frequency: "As defined in SLA (minimum annual)"

    joint_testing:
      scope: "Per Art. 30(3)(d) for critical functions"
      includes:
        - "Failover testing coordination"
        - "Backup restoration verification"
        - "Communication channel testing"
        - "Escalation path validation"

  # =========================================================================
  # PROVIDER-INITIATED TRAINING SUPPORT
  # =========================================================================
  provider_initiated:
    security_updates:
      trigger: "Material security changes to platform"
      action: "Proactive briefing to affected clients"
      format: "Webinar or documentation"

    threat_briefings:
      trigger: "Relevant threat intelligence"
      action: "Share sanitized threat information"
      format: "Security advisory to client contacts"

    platform_training:
      availability: "Self-service documentation and videos"
      live_sessions: "Quarterly for Enterprise clients"
      topics:
        - "Security best practices"
        - "Incident reporting procedures"
        - "API security configuration"

  # =========================================================================
  # SCHEDULING AND CONDITIONS
  # =========================================================================
  scheduling:
    notice_period:
      standard: "14 business days"
      urgent: "5 business days (security-related)"
      emergency: "Best effort (active incident)"

    personnel_commitment:
      key_contacts: "2 designated contacts per client"
      availability: "Best effort, operational needs considered"
      backup: "Alternative contacts if primary unavailable"

    format_preference:
      primary: "Remote (video conference)"
      on_site: "Upon request, travel costs per contract"

  # =========================================================================
  # TIME COMMITMENT AND COSTS
  # =========================================================================
  resource_commitment:
    standard_tier:
      annual_hours: "4 hours included"
      additional: "Billable at standard rate"

    professional_tier:
      annual_hours: "8 hours included"
      additional: "Billable at reduced rate"

    enterprise_tier:
      annual_hours: "16 hours included"
      quarterly_sessions: "Included"
      additional: "Negotiated rate"

    cost_provisions:
      included: "Remote participation, standard materials"
      client_responsibility: "Travel, accommodation, venue"
      custom_materials: "By agreement"

  # =========================================================================
  # DOCUMENTATION AND RECORDS
  # =========================================================================
  documentation:
    attendance_records: "Maintained for audit purposes"
    training_certificates: "Issued upon request"
    exercise_reports: "Summary provided post-exercise"
    retention: "7 years (aligned with audit requirements)"
```

### 2.5 Article 30(3) — Additional Requirements for Critical Functions

If client classifies our services as supporting "critical or important function", contracts MUST include these **10 additional requirements** (a-j):

```yaml
additional_requirements_critical:
  # Performance and reporting (a-b)
  art_30_3_a: "Full service level descriptions including quantitative and qualitative performance targets"
  art_30_3_b: "Notice periods and reporting obligations to the financial entity"

  # Business continuity (c-d) — OFTEN CONFLATED, MUST BE SEPARATE
  art_30_3_c: "Requirements for ICT provider to maintain appropriate business contingency plans (provider's own BCP/DR)"
  art_30_3_d: "Participation in testing of business contingency plans (client's resilience testing per Art. 26-27)"

  # Audit and oversight (e-g)
  art_30_3_e: "UNRESTRICTED rights of access, inspection and audit by financial entity and its competent authority"
  art_30_3_f: "Exit strategies including adequate transition periods and data portability"
  art_30_3_g: "Participation in supervisory oversight activities, including cooperation with NCAs"

  # Security and risk management (h-j)
  art_30_3_h: "Implementation and testing of business continuity measures ensuring service availability"
  art_30_3_i: "ICT security-related arrangements including implementation and testing of security measures"
  art_30_3_j: "Conditions for subcontracting, including prior approval requirements and chain monitoring"
```

**CRITICAL DISTINCTION Art. 30(3)(c) vs (d)**:
- **(c)** = Provider must HAVE business contingency plans (our internal BCP/DR)
- **(d)** = Provider must PARTICIPATE in client's testing of contingency plans (joint exercises)

#### 2.5.1 Detailed Art. 30(3) Implementation

```yaml
art_30_3_detailed_implementation:

  # Art. 30(3)(a) - Full Service Level Descriptions
  # NOTE: These are DESIGN TARGETS for a pre-seed company. Actual SLA commitments
  # are contract-specific and will be validated as operational history is established.
  sla_design_targets:
    availability:
      trading_services: "Target 99.9% monthly (design goal; actual SLA contract-specific)"
      backtest_services: "Target 99.5% monthly (design goal)"
      api_services: "Target 99.9% monthly (design goal)"
      measurement: "External monitoring planned (UptimeRobot/Datadog)"
    performance:
      order_latency_p95: "Target <500ms (design goal)"
      market_data_latency_p95: "Target <200ms (design goal)"
      api_response_p95: "Target <1000ms (design goal)"
    incident_response:
      critical: "Target 15 min acknowledgment, 1h update (design goal)"
      high: "Target 30 min acknowledgment, 4h update (design goal)"
      medium: "Target 4h acknowledgment, 24h update (design goal)"
    reporting:
      frequency: "Monthly SLA report (planned)"
      format: "PDF + JSON via client portal (planned)"

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
      - security_certification_roadmap: "SOC2 Type I planned 2026, Type II planned 2027; ISO27001 planned 2027+"
      - vulnerability_management: "Weekly scans, critical <24h remediation (design target)"
      - penetration_testing: "Annual third-party test (planned 2026)"
      - security_awareness: "Quarterly training for all staff (planned upon establishment)"
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
      our_support: "Pooled audit documentation available upon request (when operational)"

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

#### 2.7.5 Automated Client Mix Monitoring — NEW (v2.1)

Replace manual quarterly reviews with automated tracking and alerts.

```yaml
automated_ctpp_monitoring:
  # =========================================================================
  # DATA MODEL
  # =========================================================================
  client_classification:
    fields:
      - client_id: "Unique identifier"
      - client_name: "Legal name"
      - client_type: "ENUM: bank, investment_firm, casp, insurance, other"
      - regulatory_status: "ENUM: gsib, osii, licensed_fe, unregulated"
      - jurisdiction: "ISO country code"
      - critical_function_designation: "BOOLEAN"
      - onboarding_date: "ISO date"
      - services_used: "LIST of service categories"
      - revenue_contribution: "Percentage (for concentration)"

    regulatory_status_definitions:
      gsib: "Global Systemically Important Bank (FSB list)"
      osii: "Other Systemically Important Institution (national lists)"
      licensed_fe: "Licensed financial entity under DORA scope"
      unregulated: "Not subject to DORA (retail, non-EU, etc.)"

  # =========================================================================
  # AUTOMATED TRACKING
  # =========================================================================
  tracking:
    update_triggers:
      - "New client onboarding"
      - "Client status change"
      - "Critical function designation change"
      - "Contract renewal/modification"

    scheduled_refresh:
      frequency: "Daily"
      data_sources:
        - "Client database"
        - "Contract management system"
        - "Billing system (revenue data)"

    external_data_integration:
      gsib_list:
        source: "FSB (Financial Stability Board)"
        url: "https://www.fsb.org/work-of-the-fsb/market-and-institutional-resilience/post-2008-financial-crisis-reforms/ending-too-big-to-fail/global-systemically-important-banks-g-sibs/"
        update_frequency: "Annual (November)"
        automation: "Semi-annual manual check"

      osii_lists:
        note: "Maintained per jurisdiction by national authorities"
        automation: "Quarterly manual check against client jurisdictions"

  # =========================================================================
  # REAL-TIME METRICS
  # =========================================================================
  metrics:
    total_clients:
      description: "Total number of clients"
      breakdown_by: ["client_type", "regulatory_status", "jurisdiction"]

    eu_regulated_clients:
      description: "Clients under DORA scope"
      formula: "COUNT WHERE regulatory_status IN (gsib, osii, licensed_fe)"

    critical_function_clients:
      description: "Clients who designated us as critical/important"
      formula: "COUNT WHERE critical_function_designation = TRUE"

    sector_concentration:
      description: "Percentage in largest sector"
      formula: "MAX(COUNT per client_type) / total_clients * 100"

    revenue_concentration:
      description: "Revenue from top client"
      formula: "MAX(revenue_contribution)"

    gsib_osii_exposure:
      description: "Number of systemically important clients"
      formula: "COUNT WHERE regulatory_status IN (gsib, osii)"

  # =========================================================================
  # ALERT THRESHOLDS
  # =========================================================================
  alerts:
    critical:  # Immediate escalation to Board
      - trigger: "ANY GSIB onboards"
        action: "CTPP readiness assessment within 30 days"
      - trigger: "ANY OSII onboards"
        action: "CTPP risk review within 30 days"
      - trigger: "critical_function_clients >= 10"
        action: "Assess substitutability and CTPP likelihood"

    high:  # Escalation to Management
      - trigger: "eu_regulated_clients >= 50"
        action: "Review CTPP preparedness"
      - trigger: "sector_concentration >= 40%"
        action: "Diversification strategy review"
      - trigger: "critical_function_clients >= 5"
        action: "Enhanced monitoring"

    warning:  # Monthly review
      - trigger: "eu_regulated_clients >= 20"
        action: "Quarterly CTPP status review"
      - trigger: "revenue_concentration >= 25%"
        action: "Concentration risk assessment"

  # =========================================================================
  # DASHBOARD
  # =========================================================================
  dashboard:
    components:
      - ctpp_risk_score: "Composite score 0-100"
      - client_composition_chart: "Pie chart by type and status"
      - trend_analysis: "Month-over-month growth in regulated clients"
      - jurisdiction_map: "Geographic distribution"
      - alert_status: "Active warnings and critical alerts"

    access:
      - "Executive team: Full dashboard"
      - "Operations: Client metrics"
      - "Sales: New client impact preview"

    export:
      - format: "PDF report"
      - frequency: "Quarterly"
      - recipients: "Board, Legal, Compliance"

  # =========================================================================
  # CTPP RISK SCORING
  # =========================================================================
  risk_scoring:
    methodology: "Weighted composite score"

    factors:
      gsib_osii_clients:
        weight: 40
        scoring:
          "0": 0
          "1-2": 50
          "3+": 100
        rationale: "Primary CTPP trigger per Art. 31"

      critical_function_designations:
        weight: 30
        scoring:
          "0-2": 0
          "3-5": 30
          "6-10": 60
          "10+": 100

      eu_regulated_client_count:
        weight: 15
        scoring:
          "0-10": 0
          "11-50": 30
          "51-100": 60
          "100+": 100

      sector_concentration:
        weight: 15
        scoring:
          "<20%": 0
          "20-40%": 30
          "40-60%": 60
          ">60%": 100

    composite_score:
      formula: "SUM(factor_score * factor_weight) / 100"
      interpretation:
        "0-25": "LOW - Continue monitoring"
        "26-50": "MEDIUM - Enhanced monitoring, prepare documentation"
        "51-75": "HIGH - Active CTPP preparation, legal review"
        "76-100": "CRITICAL - Assume designation likely, full preparation"

  # =========================================================================
  # IMPLEMENTATION
  # =========================================================================
  implementation:
    database_changes:
      - "Add client_classification table"
      - "Add regulatory_status field to clients"
      - "Create ctpp_metrics materialized view"

    api_endpoints:
      - "GET /api/v1/ctpp/metrics - Current CTPP metrics"
      - "GET /api/v1/ctpp/score - Current risk score"
      - "GET /api/v1/ctpp/alerts - Active alerts"
      - "POST /api/v1/clients/{id}/classification - Update client classification"

    cron_jobs:
      - "Daily: Refresh metrics"
      - "Weekly: Check alert thresholds"
      - "Quarterly: Generate CTPP status report"

    integration:
      - "Slack/Teams: Alert notifications"
      - "Email: Weekly summary to management"
      - "Dashboard: Real-time display"

  # =========================================================================
  # CURRENT STATUS
  # =========================================================================
  current_status:
    implementation_status: "PLANNED"
    target_date: "Q1 2025"
    priority: "HIGH"
    owner: "Platform Engineering"

    interim_process:
      - "Manual quarterly review"
      - "Spreadsheet tracking of regulated clients"
      - "Ad-hoc GSIB/OSII checks on onboarding"
```

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
| Kill Switch | [services/ops_kill_switch.py](../services/ops_kill_switch.py) | KEEP | Core safety |
| Healthcheck | [services/healthcheck.py](../services/healthcheck.py) | ENHANCE | Add /ready, /live |
| Secure Logging | [services/secure_logging.py](../services/secure_logging.py) | KEEP | API key masking |
| Monitoring | [services/monitoring.py](../services/monitoring.py) | ENHANCE | Add alerting |
| Recovery Procedures | [RECOVERY_PROCEDURES.md](RECOVERY_PROCEDURES.md) | KEEP | 10 scenarios |
| Operations Runbook | [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md) | KEEP | Comprehensive |
| Service Dependency Map | [SERVICE_DEPENDENCY_MAP.md](SERVICE_DEPENDENCY_MAP.md) | KEEP | Architecture |
| CI/CD Pipeline | [.github/workflows/](/.github/workflows/) | ENHANCE | Security gates |
| Audit Trail | [services/core/risk_controls/audit_trail_writer.py](../services/core/risk_controls/audit_trail_writer.py) | KEEP | Core continuity |
| BCP Module | [services/core/risk_controls/bcp.py](../services/core/risk_controls/bcp.py) | KEEP | Core continuity |

### 4.B) Core DORA Contractual (ALL EU clients) — NEW CATEGORY

| Component | Location | Action | Rationale |
|-----------|----------|--------|-----------|
| **Contractual Requirements** | [services/dora_integration/contracts/contractual_requirements.py](../services/dora_integration/contracts/contractual_requirements.py) | **KEEP as Core** | Art. 30(2) mandatory |
| **Exit Strategies** | [services/dora_integration/contracts/exit_strategies.py](../services/dora_integration/contracts/exit_strategies.py) | **KEEP, adapt** | Art. 28(8), Art. 30(3)(f) |
| **Third-Party Risk** | [services/dora_integration/third_party/third_party_risk.py](../services/dora_integration/third_party/third_party_risk.py) | **KEEP, adapt** | Self-documentation |
| **Incident Management** | (Pending infrastructure deployment) | **ROADMAP** | Client notification; requires operational team |
| **Incident Reporting** | [services/dora_integration/incident_interface/incident_reporting.py](../services/dora_integration/incident_interface/incident_reporting.py) | **KEEP** | Client reports |
| **Backup Recovery** | (Pending infrastructure deployment) | **ROADMAP** | Art. 30(3)(c); requires production environment |
| **ICT Business Continuity** | (Pending infrastructure deployment) | **ROADMAP** | Art. 30(3)(c); requires production environment |

> **Tech Debt Note**: Incident Management, Backup Recovery, and ICT Business Continuity are marked as ROADMAP items pending production infrastructure deployment. This is accurately documented per Documentation Canon (no false claims about operational capabilities). Control artifacts: `docs/runbooks/` (documented procedures), `docs/operations/ON_CALL_CAPACITY_VALIDATION.md` (capacity assessment). Tech Debt Tracking: `docs/reports/TECH_DEBT_REGISTRY.md#ops-dora-gaps`

### 4.C) Enterprise DORA Support (Enhanced for regulated clients)

| Component | Location | Action |
|-----------|----------|--------|
| Incident Classification | [services/dora_integration/incident_interface/incident_classification.py](../services/dora_integration/incident_interface/incident_classification.py) | Enterprise — extended taxonomy |
| Register of Information | [services/dora_integration/reporting/register_of_information.py](../services/dora_integration/reporting/register_of_information.py) | **ADAPT** → provider_info_package |
| TLPT | (TBD in current tree) | Enterprise — cooperation support |
| Resilience Testing | (TBD in current tree) | Enterprise — joint testing |
| ICT Testing | (TBD in current tree) | Enterprise — test support |

### 4.D) Internal Platform Tools (Repurpose)

| Component | Current | Target |
|-----------|---------|--------|
| `function_classification.py` | DORA Article 3(22) | Internal service criticality |
| `governance.py` | Financial entity governance | Platform internal governance |
| `ict_systems.py` | DORA Article 7 | Internal system inventory |
| `detection.py` | DORA Article 10 | Core anomaly detection |
| `protection.py` | DORA Article 9 | Core security controls |

### 4.E) Archive (Not applicable to ICT provider role) — REVISED v2.2

| Component | Reason |
|-----------|--------|
| `scope_verification.py` | Determines if DORA applies — irrelevant, we know it applies via contracts |
| `proportionality.py` | Financial entity size classification |
| `supervisory_feedback.py` | Client-NCA communication |
| `nca_identification.yaml` | Client identifies their NCA |
| `entity_classification.yaml` | Financial entity config |

### 4.F) Keep for CTPP Preparedness — NEW v2.2

| Component | Action | Rationale |
|-----------|--------|-----------|
| `ctpp_oversight.py` | **KEEP scaled-down** | CTPP preparedness if client base grows; ESA engagement protocols |
| `concentration_risk.py` | **KEEP for awareness** | If we gain market share → CTPP designation risk |

### 4.G) Adapt for Provider Role — NEW v2.2

| Component | Current | Target | Rationale |
|-----------|---------|--------|-----------|
| `pooled_testing.py` | Client pooled testing | `pooled_audit_support.py` | Art. 30(4) allows clients to use pooled audits; we must support this |

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
    audit_request_sla: "Evidence target: available within 24 hours (actual SLA is contract-specific and subject to operational capacity)"

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
  # NOTE: This describes the TARGET on-call structure. Actual on-call coverage
  # and response times are validated in docs/operations/ON_CALL_CAPACITY_VALIDATION.md.
  # Do not commit to 24/7 coverage in contracts until team capacity supports it.
  on_call:
    rotation: "Weekly rotation (design target)"
    coverage: "24/7/365 (requires minimum 4 FTE; verify capacity before SLA commitment)"
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

> **IMPORTANT**: SLA targets below are illustrative design goals for planning infrastructure requirements. Actual SLA commitments are defined in executed service agreements after infrastructure validation and engineering sign-off per the SLA Guardrails framework (Section 5.4.4 and `services/dora/sla_guardrails.py`). Pre-seed companies should not commit to SLAs beyond proven operational capacity.

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
  # TARGET: Infrastructure validation required before offering
  # =========================================================================
  professional:
    availability: "Design target: 99.9% (pending infrastructure validation; actual SLA per executed agreement)"
    rto: "Design target: 1 hour (pending DR testing; actual commitment per executed agreement)"
    rpo: "Design target: 15 minutes (pending replication validation; actual commitment per executed agreement)"
    incident_notification: "Design target: 30 minutes (pending on-call establishment; actual SLA per executed agreement)"
    support_hours: "Design target: Extended hours (06:00-22:00 CET) (pending staffing; actual coverage per executed agreement)"

    infrastructure:
      deployment: "Target: Multi-AZ (EU-WEST-1 a/b/c) (infrastructure build-out required)"
      database: "Target: Primary + sync replica + async DR (pending implementation)"
      backups: "Target: Every 15 minutes (pending automation implementation)"
      monitoring: "Target: 1-minute intervals with auto-alerting (pending tooling)"

    cost_multiplier: 2.0x
    suitable_for: "Important functions, asset managers, hedge funds (offer pending infrastructure readiness per Section 5.4.4)"

  # =========================================================================
  # ENTERPRISE TIER (For clients with critical functions)
  # TARGET: Significant infrastructure investment required before offering
  # =========================================================================
  enterprise:
    availability: "Design target: 99.95% (pending multi-region deployment; actual SLA per executed agreement)"
    rto: "Design target: 15 minutes (pending DR automation; actual commitment per executed agreement)"
    rpo: "Design target: 5 minutes (pending sync replication; actual commitment per executed agreement)"
    incident_notification: "Design target: 15 minutes (pending 24/7 on-call team; actual SLA per executed agreement)"
    support_hours: "Design target: 24/7/365 (pending 4+ FTE on-call team; actual coverage per executed agreement)"

    infrastructure:
      deployment: "Target: Multi-region (EU-WEST-1 + EU-CENTRAL-1) (significant infrastructure investment required)"
      database: "Target: Multi-region sync replication (pending implementation)"
      backups: "Target: Continuous with point-in-time recovery (pending automation)"
      monitoring: "Target: Real-time with predictive alerting (pending tooling implementation)"

    additional:
      - dedicated_instance_option (subject to infrastructure availability)
      - custom_integrations (subject to engineering capacity)
      - quarterly_resilience_reviews (subject to operational maturity)
      - annual_joint_dr_testing (subject to DR program establishment)

    cost_multiplier: 4.0x
    suitable_for: "Critical functions, banks, CASPs with significant AUM (offer pending infrastructure build-out and operational validation per Section 5.4.4)"
```

### 5.4.2 Infrastructure Requirements by Tier

| Tier | Multi-AZ | Multi-Region | Sync Replication | DR Tested | Est. Monthly Cost |
|------|----------|--------------|------------------|-----------|-------------------|
| Standard | ❌ | ❌ | ❌ | Quarterly | €500-2,000 |
| Professional | ✅ | ❌ | Partial | Monthly | €2,000-5,000 |
| Enterprise | ✅ | ✅ | ✅ | Weekly | €5,000-15,000 |

**CRITICAL**: RTO/RPO claims MUST be backed by documented infrastructure and tested procedures.

#### Current State vs Target State — IMPORTANT

```yaml
infrastructure_reality_check:
  # Be honest about current capabilities vs contractual commitments
  current_state:
    as_of: "2025-01-01"
    deployment: "Single region (EU-WEST-1)"
    database: "Primary only with nightly backups"
    achievable_rto: "4-8 hours"
    achievable_rpo: "24 hours"
    dr_testing: "Not yet conducted"
    on_call: "Informal, no SLA"

  what_we_can_offer_today:
    tier: "Standard only"
    availability_target: "99.5%"
    incident_notification: "Best effort (2-4 hours)"
    notes: "Until infrastructure upgrades completed"

  target_state_professional:
    target_date: "Q2 2025"
    investment_required: "€30,000-50,000"
    milestones:
      - "Multi-AZ deployment"
      - "Sync replication setup"
      - "Automated failover testing"
      - "On-call rotation established"

  target_state_enterprise:
    target_date: "Q4 2025"
    investment_required: "€100,000-150,000"
    milestones:
      - "Multi-region deployment"
      - "24/7 on-call team (4+ FTE)"
      - "Automated DR testing"
	      - "SOC 2 Type II report/attestation (if pursued)"

  contractual_guidance:
    principle: "Never promise what you cannot deliver"
    standard_tier:
      offer_now: true
      confidence: "HIGH"
    professional_tier:
      offer_now: false
      offer_when: "Q2 2025 after infrastructure upgrades"
      confidence_after_upgrade: "HIGH"
    enterprise_tier:
      offer_now: false
      offer_when: "Q4 2025 after full build-out"
      confidence_after_upgrade: "HIGH"
```

### 5.4.3 Cross-Region Failover Procedure — NEW (v2.1)

**Scenario:** Primary region (eu-west-1) experiences extended outage (>30 min projected)

```yaml
cross_region_failover:
  # =========================================================================
  # TRIGGER CONDITIONS
  # =========================================================================
  trigger_conditions:
    automatic_failover:
      - "Primary region health check failures >5 minutes"
      - "AWS regional service degradation announced"
      - "Database primary unreachable >3 minutes"
    manual_failover:
      - "Projected outage >30 minutes"
      - "Security incident requiring isolation"
      - "Scheduled DR testing"

  # =========================================================================
  # FAILOVER PROCEDURE
  # =========================================================================
  procedure:
    phase_1_detection:
      duration: "0-5 minutes"
      actions:
        - "Health checks detect primary region issues"
        - "Alert fires to on-call engineer"
        - "Automated diagnostics run"
        - "AWS status page checked"
      decision_point: "Continue monitoring OR initiate failover"

    phase_2_decision:
      duration: "5-10 minutes"
      criteria:
        initiate_failover:
          - "AWS confirms regional issue"
          - "Estimated recovery >30 minutes"
          - "Critical client SLAs at risk"
        wait_and_monitor:
          - "Transient issue, recovering"
          - "Non-critical hours (if applicable)"
      approval: "On-call L2 or above"

    phase_3_failover_execution:
      duration: "10-30 minutes"
      steps:
        step_1:
          name: "Database failover"
          action: "Promote DR replica to primary"
          time: "2-5 minutes (automated)"
          verification: "Write test successful"
        step_2:
          name: "Application failover"
          action: "Route traffic to DR region"
          method: "Route 53 health-based routing OR manual DNS update"
          time: "2-5 minutes (DNS propagation)"
        step_3:
          name: "Verify services"
          action: "Run smoke tests against DR region"
          time: "5-10 minutes"
        step_4:
          name: "Client notification"
          action: "Notify all affected clients"
          time: "<30 minutes from incident start"

    phase_4_operation_in_dr:
      monitoring: "Enhanced monitoring during DR operation"
      limitations:
        - "Possible increased latency"
        - "Some non-critical features may be degraded"
        - "Capacity limits in DR region"
      communication: "Status page updated every 30 minutes"

    phase_5_failback:
      trigger: "Primary region confirmed stable >1 hour"
      procedure:
        - "Verify primary region health"
        - "Sync data changes from DR to primary"
        - "Gradual traffic shift (canary)"
        - "Full failback"
        - "Post-incident review"
      timing: "During low-traffic window when possible"

  # =========================================================================
  # DATA LOSS CONSIDERATIONS
  # =========================================================================
  data_loss_scenarios:
    synchronous_replication:
      rpo: "Near-zero (seconds)"
      cost: "HIGH (multi-region sync)"
      tier: "Enterprise"
    asynchronous_replication:
      rpo: "1-5 minutes"
      cost: "MEDIUM"
      tier: "Professional"
      note: "Possible data loss for in-flight transactions"
    backup_restore:
      rpo: "15 minutes - 1 hour"
      cost: "LOW"
      tier: "Standard (manual failover only)"

  # =========================================================================
  # RTO BREACH NOTIFICATION
  # =========================================================================
  rto_breach_handling:
    if_failover_exceeds_rto:
      action: "Proactive client notification"
      timing: "As soon as RTO breach is projected"
      content:
        - "Incident description"
        - "Current status"
        - "Revised recovery estimate"
        - "Actions being taken"
      follow_up: "Incident report with root cause within 24h"

    sla_credit_consideration:
      trigger: "RTO exceeded by >50%"
      process: "Per contract terms"
      documentation: "Full timeline preserved for audit"

  # =========================================================================
  # TESTING REQUIREMENTS
  # =========================================================================
  testing:
    frequency:
      full_failover_test: "Annual minimum"
      tabletop_exercise: "Quarterly"
      component_tests: "Monthly (database failover, DNS switch)"

    test_documentation:
      - "Test date and participants"
      - "Scenario description"
      - "Actual vs expected timeline"
      - "Issues encountered"
      - "Remediation actions"

    client_involvement:
      enterprise_tier: "Annual joint DR test offered"
      professional_tier: "Notified of test results"
      standard_tier: "Summary available on request"

  # =========================================================================
  # INFRASTRUCTURE REQUIREMENTS
  # =========================================================================
  infrastructure:
    primary_region: "eu-west-1 (Ireland)"
    dr_region: "eu-central-1 (Frankfurt)"

    components:
      database:
        type: "PostgreSQL with streaming replication"
        dr_replica: "Hot standby in DR region"
        promotion_time: "<5 minutes"
      application:
        deployment: "Pre-deployed in DR region (scaled down)"
        scale_up_time: "5-10 minutes"
      dns:
        provider: "Route 53 with health checks"
        ttl: "60 seconds"
        failover_type: "Health-based routing"
      storage:
        type: "S3 cross-region replication"
        lag: "<15 minutes"

    estimated_cost:
      dr_infrastructure: "€2,000-4,000/month"
      data_transfer: "€500-1,000/month"
      total: "€2,500-5,000/month additional"
```

### 5.4.4 Contractual SLA Guardrails — NEW v2.2

**CRITICAL PRINCIPLE:** Never promise in contracts what you cannot deliver operationally.

```yaml
contractual_sla_guardrails:
  # =========================================================================
  # RISK: CONTRACTUAL OVER-COMMITMENT
  # =========================================================================
  problem_statement: |
    If we commit to RTO=1h in contracts but can only achieve RTO=4h operationally,
    we face:
    - Breach of contract claims
    - SLA credit obligations
    - Reputational damage
    - Regulatory scrutiny from client's NCA

  # =========================================================================
  # GUARDRAIL 1: INFRASTRUCTURE-BACKED SLAs ONLY
  # =========================================================================
  guardrail_1:
    name: "Infrastructure Validation Before Contract"
    rule: |
      Before offering a specific SLA tier to a client, Sales/Legal MUST verify
      with Engineering that infrastructure supports it.

    validation_checklist:
      availability_99_9:
        requires:
          - "Multi-AZ deployment: YES/NO"
          - "Database replication: SYNC/ASYNC/NONE"
          - "Auto-failover configured: YES/NO"
          - "Load balancer health checks: YES/NO"
        evidence: "Architecture diagram + monitoring dashboard"

      rto_1_hour:
        requires:
          - "Hot/warm standby: YES/NO"
          - "Documented runbook: YES/NO"
          - "Last DR test date: [DATE]"
          - "DR test passed: YES/NO"
        evidence: "DR test report within last 6 months"

      rpo_15_minutes:
        requires:
          - "Replication type: SYNC/ASYNC/NONE"
          - "Replication lag monitored: YES/NO"
          - "Backup frequency: [FREQUENCY]"
        evidence: "Replication lag dashboard"

      notification_30_min:
        requires:
          - "24/7 on-call: YES/NO"
          - "On-call team size: [NUMBER]"
          - "Alerting SLA: [MINUTES]"
          - "Communication tools ready: YES/NO"
        evidence: "On-call schedule + incident response drill report"

  # =========================================================================
  # GUARDRAIL 2: CONTRACTUAL BUFFER
  # =========================================================================
  guardrail_2:
    name: "Conservative SLA Commitment"
    rule: |
      Contract SLAs should be MORE conservative than operational targets.
      This provides buffer for unexpected issues.

    recommended_buffer:
      availability:
        operational_target: "99.95%"
        contractual_commitment: "99.9%"
        buffer: "0.05% (~22 min/month)"

      rto:
        operational_target: "45 minutes"
        contractual_commitment: "1 hour"
        buffer: "15 minutes"

      rpo:
        operational_target: "10 minutes"
        contractual_commitment: "15 minutes"
        buffer: "5 minutes"

      notification:
        operational_target: "20 minutes"
        contractual_commitment: "30 minutes"
        buffer: "10 minutes"

  # =========================================================================
  # GUARDRAIL 3: TIERED OFFER MATRIX
  # =========================================================================
  guardrail_3:
    name: "Only Offer What We Can Deliver"

    current_state_assessment:
      date: "2025-01-01"
      infrastructure:
        deployment: "Single region (EU-WEST-1)"
        database: "Primary + async replica"
        on_call: "Informal"
        dr_tested: false

      achievable_slas:
        availability: "99.5%"
        rto: "4-8 hours"
        rpo: "1 hour"
        notification: "2-4 hours"

    offer_matrix:
      standard_tier:
        can_offer_now: true
        availability: "99.5%"
        rto: "4 hours"
        rpo: "1 hour"
        notification: "2 hours"

      professional_tier:
        can_offer_now: false
        available_when: "Q2 2025 (after infra upgrades)"
        prerequisites:
          - "Multi-AZ deployment completed"
          - "Sync replication enabled"
          - "On-call rotation established"
          - "First DR test passed"

      enterprise_tier:
        can_offer_now: false
        available_when: "Q4 2025 (after full build-out)"
        prerequisites:
          - "Multi-region deployment"
          - "24/7 on-call team (4+ FTE)"
          - "Quarterly DR tests passing"
	          - "SOC 2 Type II report/attestation (if pursued)"

  # =========================================================================
  # GUARDRAIL 4: CONTRACT REVIEW PROCESS
  # =========================================================================
  guardrail_4:
    name: "Engineering Sign-Off on SLAs"

    process:
      step_1: "Sales identifies client SLA requirements"
      step_2: "Engineering validates against current capabilities"
      step_3: "If mismatch: propose alternative tier OR timeline for upgrade"
      step_4: "Engineering sign-off required before SLA commitment"
      step_5: "Document sign-off in contract file"

    escalation:
      if_sales_pushes_for_unsupported_sla:
        - "Escalate to CTO"
        - "Document risk in contract"
        - "Client acknowledgment of infrastructure limitations"
        - "Roadmap commitment for upgrades (if client critical)"

  # =========================================================================
  # GUARDRAIL 5: SLA CREDIT PROVISIONS
  # =========================================================================
  guardrail_5:
    name: "SLA Credit Structure"

    credit_structure:
      standard_tier:
        availability_breach:
          "99.0-99.5%": "5% monthly fee credit"
          "98.0-99.0%": "10% monthly fee credit"
          "<98.0%": "25% monthly fee credit"
        cap: "25% of monthly fee"

      professional_tier:
        availability_breach:
          "99.5-99.9%": "10% monthly fee credit"
          "99.0-99.5%": "20% monthly fee credit"
          "<99.0%": "30% monthly fee credit"
        rto_breach: "10% credit per incident"
        cap: "50% of monthly fee"

      enterprise_tier:
        availability_breach:
          "99.9-99.95%": "15% monthly fee credit"
          "99.5-99.9%": "25% monthly fee credit"
          "<99.5%": "50% monthly fee credit"
        rto_breach: "15% credit per incident"
        notification_breach: "5% credit per incident"
        cap: "100% of monthly fee"

    exclusions:
      - "Scheduled maintenance windows"
      - "Client-caused issues"
      - "Force majeure events"
      - "Third-party provider outages beyond our control (with notification)"
```

### 5.4.5 Notification SLA Tiers by Operational Capacity — NEW v2.2

**Reality Check:** <30 min notification SLA requires 24/7 on-call capability.

```yaml
notification_sla_by_capacity:
  # =========================================================================
  # CURRENT OPERATIONAL REALITY
  # =========================================================================
  current_assessment:
    team_size: "[TO BE FILLED]"
    on_call_status: "[INFORMAL/FORMAL/24x7]"
    alerting_tooling: "[PRESENT/ABSENT]"

  # =========================================================================
  # NOTIFICATION SLA OPTIONS
  # =========================================================================
  options:

    # Option A: No formal on-call (Startup phase)
    option_a_no_oncall:
      name: "Business Hours Notification"
      suitable_for: "Early stage, <5 engineers, no regulated clients"

      operational_setup:
        monitoring: "Email alerts"
        coverage: "Business hours only (09:00-18:00 CET)"
        weekend: "Best effort"
        night: "No coverage"

      achievable_sla:
        critical: "4 hours (during business hours)"
        high: "8 hours"
        medium: "24 hours"

      client_disclosure: |
        Incident notifications are provided during business hours
        (09:00-18:00 CET, Monday-Friday). Outside these hours,
        notifications will be sent on the next business day.

      cost: "€0 additional"
      risk: "Cannot serve regulated clients with critical functions"

    # Option B: Managed NOC (Growth phase)
    option_b_managed_noc:
      name: "Managed NOC with Engineering Escalation"
      suitable_for: "Growing startup, 3-8 engineers, some regulated clients"

      operational_setup:
        monitoring: "PagerDuty/Opsgenie"
        first_line: "Managed NOC (outsourced 24/7)"
        second_line: "Engineering on-call (extended hours)"
        escalation_time: "15 minutes from NOC to Engineering"

      achievable_sla:
        critical: "60 minutes"
        high: "90 minutes"
        medium: "4 hours"

      providers:
        - name: "DataDog/PagerDuty NOC services"
          cost: "€2,000-4,000/month"
        - name: "Specialized NOC providers"
          cost: "€3,000-6,000/month"

      client_disclosure: |
        24/7 monitoring with initial triage by our operations center.
        Engineering escalation within 15 minutes for critical issues.
        Client notification within 60 minutes for critical incidents.

      cost: "€3,000-5,000/month"
      risk: "10-15 min delay vs in-house on-call"

    # Option C: In-house on-call (Scale phase)
    option_c_inhouse_oncall:
      name: "In-House 24/7 On-Call"
      suitable_for: "Established company, 8+ engineers, regulated clients"

      operational_setup:
        monitoring: "PagerDuty/Opsgenie with auto-alerting"
        on_call_rotation: "Weekly rotation, 4+ engineers minimum"
        response_time: "15 minutes acknowledgment"
        escalation: "Automatic after 15 min no-response"

      achievable_sla:
        critical: "30 minutes"
        high: "60 minutes"
        medium: "4 hours"

      staffing_requirements:
        minimum_engineers: 4
        rationale: "Sustainable rotation (1 week each per month)"
        compensation: "On-call allowance + overtime"
        burnout_prevention: "Max 1 week on-call per month"

      client_disclosure: |
        24/7 in-house engineering coverage with 15-minute
        acknowledgment target. Client notification within 30 minutes
        for critical incidents.

      cost: "€4,000-8,000/month (on-call compensation)"
      risk: "Burnout if team too small"

    # Option D: Dedicated NOC + Engineering (Enterprise phase)
    option_d_dedicated_noc:
      name: "Dedicated NOC Team"
      suitable_for: "Enterprise scale, significant regulated client base"

      operational_setup:
        noc_team: "3 FTE minimum (8-hour shifts)"
        engineering_on_call: "Backup for complex issues"
        response_time: "5 minutes acknowledgment"

      achievable_sla:
        critical: "15 minutes"
        high: "30 minutes"
        medium: "2 hours"

      client_disclosure: |
        Illustrative target: Dedicated 24/7 operations team with continuous monitoring (subject to staffing/contracts).
        Target: Client notification within 15 minutes for critical incidents (aspirational; actual SLA per contract).

      cost: "€15,000-25,000/month (3 FTE)"
      risk: "High fixed cost"

  # =========================================================================
  # RECOMMENDATION MATRIX
  # =========================================================================
  recommendation_matrix:
    if_no_regulated_clients:
      recommendation: "Option A"
      rationale: "No DORA requirements, minimize cost"

    if_few_regulated_clients_non_critical:
      recommendation: "Option B"
      rationale: "Balance cost and compliance"

    if_regulated_clients_critical_functions:
      recommendation: "Option C minimum"
      rationale: "30 min notification required for DORA"

    if_enterprise_focus:
      recommendation: "Option D"
      rationale: "15 min notification differentiator"

  # =========================================================================
  # TRANSITION PLANNING
  # =========================================================================
  transition_plan:
    current_state: "[OPTION A/B/C/D]"

    phase_1:
      trigger: "First regulated client signs"
      action: "Implement Option B (Managed NOC)"
      timeline: "2-4 weeks"
      cost: "€3,000-5,000/month"

    phase_2:
      trigger: "5+ regulated clients OR critical function designation"
      action: "Transition to Option C (In-house on-call)"
      timeline: "2-3 months (hiring + training)"
      cost: "€4,000-8,000/month"

    phase_3:
      trigger: "Enterprise client base OR CTPP designation risk"
      action: "Implement Option D (Dedicated NOC)"
      timeline: "6-12 months"
      cost: "€15,000-25,000/month"

  # =========================================================================
  # CONTRACTUAL ALIGNMENT
  # =========================================================================
  contractual_alignment:
    principle: "Match SLA offers to operational capability"

    current_capability: "[OPTION A/B/C/D]"

    sla_offers:
      if_option_a:
        max_notification_sla: "4 hours (business hours only)"
        cannot_offer: "Critical function contracts"

      if_option_b:
        max_notification_sla: "60 minutes"
        can_offer: "Important function contracts"
        cannot_offer: "Enterprise tier (<30 min)"

      if_option_c:
        max_notification_sla: "30 minutes"
        can_offer: "Critical function contracts"
        can_offer_enterprise: "Yes, with buffer"

      if_option_d:
        max_notification_sla: "15 minutes"
        can_offer: "All tiers"
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
        - "GDPR commitments (vendor-asserted)"
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
      # CCEA NOTE: Client's Agent connects directly to Alpaca. Our Cloud has NO access to client credentials.
      has_data_access: false  # We do NOT store or access client API keys - handled by client's local Agent
      data_types_accessed: []  # Our Cloud receives NO brokerage credentials
	      certifications:
	        - "SOC 2 (vendor-provided report/attestation; verify with vendor)"
	      regulatory_notes:
	        - "Alpaca is a broker (vendor status); CustodiaCloud is a software/ICT provider and does not execute orders from the Cloud."
      contract_reference: "Alpaca API Agreement (client's direct relationship)"
      is_material: false  # Client's integration, not ours
      supports_critical_functions: false  # Trading execution happens in client's Agent, not our Cloud
      substitutability: "medium"
      last_audit_date: "N/A"
      next_review_date: "2025-06-01"

    binance:
      legal_name: "Binance Holdings Limited"
      lei: "TBD"  # Note: Binance operates through multiple entities
      alternative_id: "Cayman Islands Exempt Company"
      alternative_id_type: "CAYMAN_REGISTRATION"
      subcontractor_type: "data_provider"
      chain_level: 1
      services_provided:
        - "Crypto market data"
        - "Crypto trading API"
      data_processing_locations: ["Global (various jurisdictions)"]
      data_storage_locations: ["Variable - depends on Binance entity used"]
      # CCEA NOTE: Client's Agent connects directly to Binance. Our Cloud has NO access to client credentials.
      has_data_access: false  # We do NOT store or access client API keys - handled by client's local Agent
      data_types_accessed: []  # Our Cloud receives NO exchange credentials
      certifications:
        - "Variable by jurisdiction"
        - "See regulatory_risk_assessment below"
      contract_reference: "Binance API Terms (client's direct relationship)"
      is_material: false  # Client's integration, not ours
      supports_critical_functions: false  # Crypto trading happens in client's Agent, not our Cloud
      substitutability: "medium"  # Kraken, Coinbase alternatives
      last_audit_date: "N/A"
      next_review_date: "2025-06-01"

      # REGULATORY RISK ASSESSMENT - Required per Art. 29
      regulatory_risk_assessment:
        overall_risk_level: "HIGH"
        assessment_date: "2025-01-01"
        next_assessment: "2025-06-01"

        jurisdictional_issues:
          - jurisdiction: "United States"
            status: "SEC enforcement action (2023)"
            implication: "US clients should use Binance.US or alternatives"
          - jurisdiction: "European Union"
            status: "Some EU entities licensed (France AMF, others pending)"
            implication: "Verify specific entity used for EU clients"
          - jurisdiction: "United Kingdom"
            status: "FCA restrictions - not authorized"
            implication: "UK clients should use alternatives"

        risk_mitigation:
          - "Offer alternative exchanges (Kraken, Coinbase) for regulated clients"
          - "Allow per-client exchange restrictions"
          - "Client consent required before enabling Binance"
          - "Segregate crypto trading as optional feature"

        alternative_providers:
          - name: "Kraken"
            lei: "TBD"
            regulatory_status: "Third-party reported status (not verified by us); clients must verify via official registries"
            notes: "Clients must independently verify current licensing status in their jurisdiction"
          - name: "Coinbase"
            lei: "5493005KJDX9YGBJI252"
            regulatory_status: "Third-party reported status (not verified by us); clients must verify via official registries"
            notes: "Clients must independently verify current regulatory status in their jurisdiction"

        client_disclosure: |
          Binance regulatory status varies significantly by jurisdiction.
          Clients must verify Binance's authorization status in their jurisdiction
          before enabling Binance integration. Platform offers alternative
          exchange integrations (Kraken, Coinbase) for jurisdictions where
          Binance is not authorized.

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
        - "SOC 2 Type II (verify scope/period with vendor)"
        - "ISO 27001 (verify scope/period with vendor)"
        - "GDPR commitments (vendor-asserted; verify DPA)"
      contract_reference: "Datadog Enterprise Agreement"
      is_material: false
      supports_critical_functions: false
      substitutability: "easy"

    sentry:
      legal_name: "Functional Software, Inc. (Sentry)"
      lei: "TBD"  # Private company - may not have LEI
      alternative_id: "Delaware Corporation"
      alternative_id_type: "US_STATE_REGISTRATION"
      subcontractor_type: "monitoring"
      chain_level: 1
      services_provided:
        - "Error tracking"
        - "Performance monitoring"
      data_processing_locations: ["US", "EU option available"]
      data_storage_locations: ["US (default)", "EU (on request)"]
      has_data_access: true  # Error context may contain data
      certifications:
        - "SOC 2 Type II"
        - "GDPR commitments (vendor-asserted)"
      is_material: false
      supports_critical_functions: false
      substitutability: "easy"  # Alternatives: Bugsnag, Rollbar

    # --- TIER 2: Authentication ---
    auth0_clerk:
      legal_name: "Auth0 Inc. (subsidiary of Okta, Inc.)"
      lei: "549300N8BTFTU58UJ747"  # Okta LEI - Auth0 is subsidiary
      parent_company: "Okta, Inc."
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

### 5.8.1 Subcontracting Prior Written Approval Workflow — NEW v2.2

**Legal Basis:** Art. 30(3)(j)(i) requires "prior approval for the subcontracting of all or part of critical or important functions."

```yaml
subcontracting_approval_workflow:
  # =========================================================================
  # SCOPE: When Prior Written Approval is Required
  # =========================================================================
  scope:
    requires_prior_approval:
      - "New subcontractor for critical function services"
      - "Change of subcontractor for critical function services"
      - "Material change to subcontractor scope affecting critical functions"
      - "Subcontractor location change (data processing/storage)"

    notification_only:
      - "Non-critical function subcontractor changes"
      - "Minor scope changes to existing subcontractors"
      - "Security patch updates by subcontractors"

  # =========================================================================
  # APPROVAL PROCESS
  # =========================================================================
  approval_process:
    step_1_internal_assessment:
      duration: "5 business days"
      actions:
        - "Security assessment of proposed subcontractor"
        - "DORA compliance verification"
        - "Data protection impact assessment"
        - "Concentration risk assessment"
      output: "Internal approval recommendation"

    step_2_client_notification:
      timing: "Minimum 60 days before intended change"
      content:
        - "Proposed subcontractor identification (name, LEI, location)"
        - "Services to be subcontracted"
        - "Data access scope"
        - "Security certifications"
        - "Our risk assessment summary"
        - "Alternative options if client objects"
      format: "Formal written notice via contract-specified channel"

    step_3_client_review_period:
      duration: "30 business days from notification"
      client_options:
        approve: "Written approval received"
        approve_with_conditions: "Approval with specific requirements"
        object: "Written objection with reasons"
        no_response: "Deemed approval after 30 days (if contract permits)"

    step_4_objection_handling:
      if_client_objects:
        - "Acknowledge objection within 5 business days"
        - "Provide alternative subcontractor options"
        - "Negotiate resolution"
        - "If no resolution: client retains termination rights"
      escalation: "Legal review if impasse"

    step_5_implementation:
      upon_approval:
        - "Update subcontractor register"
        - "Update client's provider information package"
        - "Implement agreed conditions"
        - "Document approval for audit trail"

  # =========================================================================
  # DOCUMENTATION REQUIREMENTS
  # =========================================================================
  documentation:
    approval_record:
      - "Client name and contract reference"
      - "Subcontractor details"
      - "Notification date and method"
      - "Client response date and content"
      - "Any conditions attached"
      - "Implementation date"

    retention: "Duration of contract + 7 years"
    audit_availability: "Available within 24 hours of request"

  # =========================================================================
  # EMERGENCY SUBCONTRACTING
  # =========================================================================
  emergency_process:
    trigger: "Critical subcontractor failure requiring immediate replacement"
    process:
      - "Immediate client notification (within 24 hours)"
      - "Temporary emergency subcontractor engagement"
      - "Expedited approval process (10 business days)"
      - "Client right to terminate if not approved"
    documentation: "Full incident report with justification"

  # =========================================================================
  # CONTRACT TEMPLATE CLAUSE
  # =========================================================================
  contract_clause_template: |
    SUBCONTRACTING (Art. 30(3)(j))

    1. Provider shall not subcontract any services supporting Client's critical
       or important functions without Client's prior written approval.

    2. Provider shall notify Client of any proposed subcontracting at least
       sixty (60) days in advance, providing:
       (a) Identity and location of proposed subcontractor;
       (b) Scope of services to be subcontracted;
       (c) Security certifications and compliance status;
       (d) Provider's risk assessment.

    3. Client shall respond within thirty (30) business days. Failure to
       respond shall [be deemed approval / require explicit approval].

    4. Client may object to proposed subcontracting with written reasons.
       Provider shall propose alternatives or Client may terminate per
       Section [X] without penalty.

    5. Provider shall maintain a current register of all subcontractors
       supporting Client's services, available upon request.
```

### 5.8.2 Subcontractor Incident Escalation — NEW v2.2

**Purpose:** When our subcontractors (AWS, Alpaca, Binance, etc.) experience incidents, we need a defined process to gather information and notify our clients.

```yaml
subcontractor_incident_escalation:
  # =========================================================================
  # MONITORING SUBCONTRACTOR STATUS
  # =========================================================================
  monitoring:
    automated:
      aws:
        source: "AWS Health Dashboard API"
        check_frequency: "Every 1 minute"
        regions_monitored: ["eu-west-1", "eu-central-1"]
        alert_on: ["operational_issue", "service_event", "account_notification"]

      alpaca:
        source: "Alpaca Status Page (status.alpaca.markets)"
        check_frequency: "Every 5 minutes"
        alert_on: ["degraded_performance", "partial_outage", "major_outage"]

      binance:
        source: "Binance API status endpoint"
        check_frequency: "Every 5 minutes"
        alert_on: ["system_maintenance", "api_degradation"]

      polygon:
        source: "Polygon.io status page"
        check_frequency: "Every 5 minutes"
        alert_on: ["delayed_data", "partial_outage"]

    manual_checks:
      frequency: "Daily review of status pages"
      responsible: "On-call engineer"

  # =========================================================================
  # INCIDENT CLASSIFICATION (Subcontractor Events)
  # =========================================================================
  classification:
    critical_subcontractor_incident:
      definition: "Complete loss of subcontractor service affecting our trading operations"
      examples:
        - "AWS eu-west-1 region outage"
        - "Alpaca trading API down"
        - "Authentication provider (Auth0) unavailable"
      our_sla: "Client notification within 30 minutes"

    high_subcontractor_incident:
      definition: "Degraded subcontractor service affecting our performance"
      examples:
        - "AWS elevated latency"
        - "Alpaca partial API degradation"
        - "Market data delays >5 minutes"
      our_sla: "Client notification within 60 minutes"

    medium_subcontractor_incident:
      definition: "Subcontractor issue with limited client impact"
      examples:
        - "Non-critical region affected"
        - "Backup data provider degraded"
        - "Monitoring service issues"
      our_sla: "Client notification within 4 hours (if relevant)"

  # =========================================================================
  # ESCALATION PROCEDURE
  # =========================================================================
  procedure:
    phase_1_detection:
      duration: "0-5 minutes"
      actions:
        - "Automated alert received or manual detection"
        - "Verify incident via subcontractor status page"
        - "Check impact on our services"
        - "Log incident start time"

    phase_2_assessment:
      duration: "5-15 minutes"
      actions:
        - "Determine which of OUR services affected"
        - "Identify which CLIENTS affected"
        - "Classify incident severity"
        - "Contact subcontractor support (if needed)"

    phase_3_client_notification:
      duration: "15-30 minutes"
      content:
        - "Incident ID and timestamp"
        - "Subcontractor identified"
        - "Impact on YOUR services (client-specific)"
        - "Our mitigation actions (if any)"
        - "Expected resolution (if known)"
        - "Next update timeline"
      note: "Do NOT wait for subcontractor resolution before notifying clients"

    phase_4_ongoing_updates:
      frequency: "Every 30 minutes during active incident"
      content:
        - "Current status"
        - "Subcontractor updates"
        - "Our actions"
        - "Revised timeline"

    phase_5_resolution:
      upon_subcontractor_resolution:
        - "Verify our services restored"
        - "Notify clients of resolution"
        - "Document total duration and impact"
        - "Request incident report from subcontractor (for major incidents)"

    phase_6_post_incident:
      timeline: "Within 5 business days"
      deliverables:
        - "Incident report including subcontractor details"
        - "Root cause (from subcontractor if available)"
        - "Our response timeline analysis"
        - "Lessons learned"
        - "Preventive measures"

  # =========================================================================
  # SUBCONTRACTOR COMMUNICATION
  # =========================================================================
  subcontractor_contacts:
    aws:
      support_tier: "Business or Enterprise Support required"
      escalation: "AWS Support Case → TAM (if Enterprise)"
      sla: "1 hour response for production down (Business)"

    alpaca:
      support: "support@alpaca.markets"
      escalation: "Account manager (if Enterprise)"
      status_page: "https://status.alpaca.markets"

    binance:
      support: "API support ticket system"
      escalation: "VIP account manager"
      status: "https://www.binance.com/en/support"

  # =========================================================================
  # CLIENT COMMUNICATION TEMPLATE
  # =========================================================================
  notification_template:
    subject: "Service Impact Notice - Third-Party Provider Incident"
    body: |
      INCIDENT NOTIFICATION

      Incident ID: [INCIDENT_ID]
      Detected: [TIMESTAMP_UTC]
      Status: [ACTIVE/RESOLVED]

      THIRD-PARTY PROVIDER:
      Provider: [SUBCONTRACTOR_NAME]
      Provider Status: [Link to their status page]

      IMPACT ON YOUR SERVICES:
      - Affected services: [List]
      - Current functionality: [Available/Degraded/Unavailable]
      - Data integrity: [Confirmed/Under review]

      OUR ACTIONS:
      - [Mitigation steps taken]
      - [Failover activated if applicable]

      EXPECTED RESOLUTION:
      - Provider estimate: [Time if known, or "Under investigation"]
      - Next update: [Time]

      CONTACTS:
      - Incident Commander: [Name, Email, Phone]
      - Status Page: [URL]

      This notification is provided per our DORA contractual obligations.
      You may use this information for your regulatory reporting as needed.

  # =========================================================================
  # FAILOVER CAPABILITIES
  # =========================================================================
  failover_options:
    market_data:
      primary: "Polygon.io"
      secondary: "Alpaca market data"
      tertiary: "Yahoo Finance (degraded)"
      auto_failover: true

    trading_execution:
      primary: "Per client broker configuration"
      secondary: "N/A (client-specific)"
      auto_failover: false
      note: "Manual intervention required"

    cloud_infrastructure:
      primary: "AWS eu-west-1"
      secondary: "AWS eu-central-1"
      auto_failover: "Partial (database), Manual (full)"
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

### 5.11 Data Residency Configuration — NEW v2.2

**Legal Basis:** Art. 30(2)(b) requires contracts to specify "locations (regions or countries) where the contracted or subcontracted functions are to be provided and where data is to be processed."

```yaml
data_residency_configuration:
  # =========================================================================
  # PURPOSE
  # =========================================================================
  purpose: |
    Provide configurable data residency options to meet Art. 30(2)(b) requirements
    and client-specific data localization needs. EU regulated clients may require
    EU-only data processing and storage.

  # =========================================================================
  # DATA RESIDENCY OPTIONS
  # =========================================================================
  residency_options:

    option_eu_only:
      name: "EU-Only Data Residency"
      description: "All data processed and stored within EU/EEA"
      regions:
        primary: "eu-west-1 (Ireland)"
        failover: "eu-central-1 (Frankfurt)"
        backup: "eu-west-3 (Paris)"
      suitable_for:
        - "EU regulated financial entities"
        - "Clients with strict GDPR requirements"
        - "German clients (C5 compliance)"
      restrictions:
        - "No data transfer to non-EU regions"
        - "All subcontractors must have EU data processing"
        - "Monitoring data stays in EU"
      implementation:
        database: "RDS eu-west-1 with eu-central-1 replica"
        storage: "S3 eu-west-1 with cross-region replication to eu-central-1"
        compute: "ECS/Lambda in eu-west-1 only"
        monitoring: "Datadog EU (Germany) region"
        error_tracking: "Sentry EU region"
      additional_cost: "€0-500/month (minimal)"

    option_eu_primary_global_backup:
      name: "EU Primary with Global Backup"
      description: "Primary processing in EU, backup/DR may use other regions"
      regions:
        primary: "eu-west-1 (Ireland)"
        failover: "us-east-1 (N. Virginia) - encrypted backup only"
      suitable_for:
        - "Clients accepting non-EU backup locations"
        - "Cost-optimized deployments"
      restrictions:
        - "Live data processing in EU only"
        - "Backup data encrypted and accessible only for DR"
        - "Requires DPF/SCCs for US backup"
      additional_cost: "€0"

    option_dedicated_region:
      name: "Dedicated Region Deployment"
      description: "Single-tenant deployment in client-specified region"
      regions: "Client-specified (EU, US, APAC)"
      suitable_for:
        - "Enterprise clients with specific jurisdiction requirements"
        - "Clients requiring complete data isolation"
      implementation:
        type: "Dedicated infrastructure"
        isolation: "Complete tenant isolation"
        management: "Dedicated or customer-managed"
      additional_cost: "€3,000-10,000/month"

    option_on_premise:
      name: "On-Premise / Private Cloud"
      description: "Deployment within client's own infrastructure"
      suitable_for:
        - "Banks with strict data sovereignty"
        - "Clients prohibited from using public cloud"
      implementation:
        delivery: "Container images / VM templates"
        support: "Installation assistance + ongoing support"
      additional_cost: "Custom pricing"

  # =========================================================================
  # DEFAULT CONFIGURATION
  # =========================================================================
  default_configuration:
    new_clients: "option_eu_only"
    rationale: "Most conservative option for DORA compliance"
    exceptions: "By explicit client request with risk acknowledgment"

  # =========================================================================
  # DATA CLASSIFICATION BY RESIDENCY
  # =========================================================================
  # Note: Classifications below describe design intent. Actual residency
  # enforcement depends on infrastructure deployment and should be verified
  # via configuration audit. CustodiaCloud does not make absolute guarantees
  # about data residency in documentation.
  data_classification:
    designed_eu_primary:  # Design intent; requires infrastructure verification
      - "User credentials and authentication data"
      - "Trading strategies and configurations"
      - "Backtest results"
      - "Trained ML models"
      - "Audit logs"
      - "Personal data (GDPR scope)"

    configurable:
      - "Market data cache (can use global CDN)"
      - "Anonymized platform metrics"
      - "Public documentation"

    never_stored:
      - "Client broker passwords (passed through, not stored)"
      - "Raw market data (streamed, not persisted)"

  # =========================================================================
  # SUBCONTRACTOR DATA RESIDENCY
  # =========================================================================
  subcontractor_residency:
    aws:
      eu_only_possible: true
      regions_used: ["eu-west-1", "eu-central-1"]
      compliance: "SOC2, ISO27001, C5"

    datadog:
      eu_only_possible: true
      region: "EU (Germany)"
      configuration: "EU data center selected"

    sentry:
      eu_only_possible: true
      region: "EU (available on request)"
      action_required: "Configure EU data residency by 2025-01-31"

    auth0:
      eu_only_possible: true
      region: "EU option available"
      configuration: "Select EU tenant on setup"

    stripe:
      eu_only_possible: true
      region: "EU (Ireland)"
      configuration: "EU entity by default"

    polygon_alpaca_binance:
      eu_only_possible: false
      location: "US / Global"
      data_type: "Market data only (no client PII)"
      mitigation: "No client data sent to these providers"

  # =========================================================================
  # CLIENT CONFIGURATION PROCESS
  # =========================================================================
  configuration_process:
    at_onboarding:
      step_1: "Client selects data residency option"
      step_2: "Configuration documented in contract"
      step_3: "Infrastructure provisioned per selection"
      step_4: "Residency verified and documented"

    change_requests:
      notice_period: "30 days minimum"
      process:
        - "Client submits change request"
        - "Impact assessment"
        - "Migration plan"
        - "Execution during maintenance window"
        - "Verification and documentation update"
      cost: "Migration costs may apply"

  # =========================================================================
  # CONTRACTUAL DISCLOSURE
  # =========================================================================
  contractual_disclosure:
    template_clause: |
      DATA LOCATIONS (Art. 30(2)(b))

      1. Primary Data Processing Location: [REGION]
      2. Backup/DR Location: [REGION]
      3. Data Storage Location: [REGION]

      All data processing and storage shall occur within the locations
      specified above. Provider shall notify Client at least sixty (60)
      days in advance of any proposed change to data processing or
      storage locations.

      Subcontractor data processing locations are detailed in Schedule [X]
      (Subcontractor Register).

    mandatory_fields:
      - "Primary processing region"
      - "Backup/DR region"
      - "Storage region"
      - "Subcontractor locations"
      - "Change notification commitment"

  # =========================================================================
  # VERIFICATION AND AUDIT
  # =========================================================================
  verification:
    technical_controls:
      - "AWS S3 bucket policies restricting to EU regions"
      - "RDS instance region verification"
      - "CloudTrail logging of cross-region access attempts"
      - "Network policies blocking non-EU egress"

    audit_evidence:
      - "AWS Config compliance reports"
      - "Infrastructure-as-code region specifications"
      - "Data flow diagrams with regions"
      - "Subcontractor region attestations"

    monitoring:
      - "Automated alerts for data leaving configured region"
      - "Monthly compliance reports"
      - "Annual third-party verification"

  # =========================================================================
  # IMPLEMENTATION STATUS
  # =========================================================================
  implementation_status:
    eu_only_option:
      status: "AVAILABLE"
      verification: "Pending formal audit"

    dedicated_region:
      status: "PLANNED"
      target_date: "Q3 2025"

    on_premise:
      status: "ROADMAP"
      target_date: "Q4 2025"

  # =========================================================================
  # ACTION ITEMS
  # =========================================================================
  action_items:
    - action: "Configure Sentry EU data residency"
      priority: "HIGH"
      deadline: "2025-01-31"

    - action: "Document default EU-only configuration"
      priority: "HIGH"
      deadline: "2025-01-31"

    - action: "Create data residency selection in onboarding flow"
      priority: "MEDIUM"
      deadline: "2025-Q1"

    - action: "Implement automated region compliance monitoring"
      priority: "MEDIUM"
      deadline: "2025-Q2"
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

### 6.6 Pre-Contractual Support Package (Art. 28(7)) — NEW

Per DORA Article 28(7), financial entities must conduct pre-contractual due diligence. We proactively support this with a comprehensive information package.

```yaml
pre_contractual_support:
  # Reference: DORA Article 28(7) - Pre-contractual risk assessment
  purpose: |
    Support financial entity's pre-contractual assessment obligations
    by providing comprehensive information package for due diligence.

  standard_package:
    # Available to all prospective clients
    service_description:
      - "Platform capabilities overview"
      - "Architecture documentation"
      - "Security whitepaper"
      - "Service Level Agreement template"

    security_documentation:
      - "Security policy summary"
      - "Data handling procedures"
      - "Encryption standards"
      - "Access control overview"

    compliance_attestations:
      - "SOC 2 Type II report (under NDA)"
      - "ISO 27001 certificate (when available)"
      - "Penetration test executive summary"
      - "GDPR compliance statement"

    operational_documentation:
      - "BCP/DR summary"
      - "Incident response overview"
      - "Change management process"
      - "SLA performance history"

  enterprise_package:
    # Additional for regulated clients
    detailed_assessments:
      - "Full security architecture review"
      - "Risk assessment documentation"
      - "Control mapping (DORA/SOC2/ISO)"
      - "Detailed subcontractor chain"

    regulatory_support:
      - "DORA compliance statement"
      - "Article 30 clause mapping"
      - "Concentration risk self-assessment"
      - "Exit strategy documentation"

    due_diligence_support:
      - "On-site security review option"
      - "Technical Q&A sessions"
      - "Reference customer contacts"
      - "Custom security questionnaire completion"

  questionnaire_support:
    turnaround_time:
      standard_questionnaires: "5 business days"
      custom_questionnaires: "10 business days"
      complex_assessments: "By agreement"

    supported_formats:
      - "SIG Lite / SIG Core"
      - "CAIQ (CSA)"
      - "Custom client questionnaires"
      - "Regulatory questionnaires"

  concentration_risk_support:
    # Per Art. 29 - help client assess concentration risk
    information_provided:
      - "Number of financial entity clients (aggregated)"
      - "Geographic distribution"
      - "No single client >25% revenue"
      - "Subcontractor diversification"

    self_assessment:
      substitutability: "MEDIUM - alternatives exist but migration requires effort"
      market_position: "Not dominant - multiple competitors"
      single_point_failure: "No - multi-provider architecture"

  response_commitment:
    information_request: "Initial response within 2 business days"
    complete_package: "Within 5 business days of engagement"
    custom_requests: "Timeline agreed per request"
```

#### 6.6.1 Pre-Contractual Portal Implementation — NEW v2.2

```yaml
precontractual_portal_implementation:
  # =========================================================================
  # PORTAL ARCHITECTURE
  # =========================================================================
  architecture:
    type: "Self-service web portal"
    authentication: "Email verification + optional SSO"
    url: "trust.platform.com (or security.platform.com)"

  # =========================================================================
  # PUBLIC TIER (No authentication)
  # =========================================================================
  public_tier:
    available_to: "Anyone"
    content:
      security_overview:
        - "Security whitepaper (PDF)"
        - "Architecture overview (high-level)"
	        - "Security attestations list (if available)"
        - "Data handling summary"

      service_documentation:
        - "Platform capabilities overview"
        - "API documentation (public endpoints)"
        - "SLA tier descriptions"
        - "Pricing models"

      trust_indicators:
	        - "SOC 2 Type II report availability (under NDA, if applicable)"
	        - "ISO 27001 status (if applicable)"
	        - "GDPR posture summary (privacy-by-design; evidence exports)"
	        - "DORA alignment summary (evidence toolkit; not a certification claim)"

  # =========================================================================
  # REGISTERED TIER (Email verification)
  # =========================================================================
  registered_tier:
    available_to: "Verified business email"
    registration_fields:
      required:
        - "Company name"
        - "Business email"
        - "Role/Title"
        - "Country"
      optional:
        - "Regulatory status"
        - "Expected use case"
        - "Timeline"

    content:
      detailed_documentation:
        - "Full security architecture document"
        - "Data flow diagrams"
        - "Encryption specifications"
        - "Access control model"

      compliance_documents:
        - "DORA Article 30 clause mapping"
        - "Subcontractor list (summary)"
        - "BCP/DR overview"
        - "Incident response summary"

      templates:
        - "Standard contract template (preview)"
        - "SLA template"
        - "DPA template"

  # =========================================================================
  # NDA TIER (After NDA signed)
  # =========================================================================
  nda_tier:
    available_to: "Prospects who signed NDA"
    process:
      - "Prospect requests NDA tier access"
      - "We send mutual NDA"
      - "Signed NDA uploaded to portal"
      - "Legal verifies signature"
      - "NDA tier unlocked"

    content:
      confidential_reports:
        - "SOC2 Type II full report"
        - "Penetration test executive summary"
        - "Vulnerability assessment summary"
        - "Incident history (sanitized)"

      detailed_assessments:
        - "Full subcontractor chain with LEIs"
        - "Detailed control descriptions"
        - "Risk register summary"
        - "Audit findings status"

  # =========================================================================
  # ENTERPRISE TIER (Active negotiation)
  # =========================================================================
  enterprise_tier:
    available_to: "Prospects in active contract negotiation"
    activation: "Sales team grants access"

    content:
      custom_support:
        - "Custom security questionnaire completion"
        - "On-site security review scheduling"
        - "Technical Q&A sessions"
        - "Reference customer introductions"

      negotiation_support:
        - "Contract redline tracking"
        - "Custom SLA negotiation"
        - "Dedicated pre-sales engineer"

  # =========================================================================
  # SECURITY QUESTIONNAIRE AUTOMATION
  # =========================================================================
  questionnaire_automation:
    supported_formats:
      - name: "SIG Lite"
        auto_response: "80% auto-populated"
        turnaround: "2 business days"

      - name: "SIG Core"
        auto_response: "70% auto-populated"
        turnaround: "5 business days"

      - name: "CAIQ (CSA)"
        auto_response: "75% auto-populated"
        turnaround: "3 business days"

      - name: "Custom"
        auto_response: "Varies"
        turnaround: "10 business days"

    implementation:
      tool_options:
        - "Vanta Trust Center"
        - "Drata Trust Center"
        - "SafeBase"
        - "Custom built"

      features:
        - "Question-answer database"
        - "Evidence attachment"
        - "Version control"
        - "Export to multiple formats"

  # =========================================================================
  # METRICS AND TRACKING
  # =========================================================================
  metrics:
    portal_analytics:
      - "Visitors by company/country"
      - "Document downloads"
      - "Time spent per section"
      - "Questionnaire requests"

    sales_integration:
      - "Lead scoring based on engagement"
      - "CRM integration (Salesforce/HubSpot)"
      - "Automatic notification to sales"

  # =========================================================================
  # IMPLEMENTATION ROADMAP
  # =========================================================================
  implementation_roadmap:
    phase_1_mvp:
      timeline: "Q1 2025"
      scope:
        - "Public documentation page"
        - "Email registration for detailed docs"
        - "Manual NDA process"
      effort: "2-4 weeks"
      cost: "€0-500 (static hosting)"

    phase_2_automation:
      timeline: "Q2 2025"
      scope:
        - "Trust center platform (Vanta/Drata)"
        - "Questionnaire automation"
        - "NDA workflow automation"
      effort: "4-6 weeks"
      cost: "€500-1500/month (SaaS)"

    phase_3_enterprise:
      timeline: "Q3 2025"
      scope:
        - "Custom portal development"
        - "CRM integration"
        - "Advanced analytics"
      effort: "8-12 weeks"
      cost: "€2000-5000 (development) + hosting"

  # =========================================================================
  # ACTION ITEMS
  # =========================================================================
  action_items:
    - action: "Create public security overview page"
      priority: "HIGH"
      deadline: "2025-Q1"
      owner: "Marketing + Security"

    - action: "Evaluate trust center platforms (Vanta, Drata, SafeBase)"
      priority: "MEDIUM"
      deadline: "2025-Q1"
      owner: "Security"

    - action: "Build question-answer database for common questionnaires"
      priority: "MEDIUM"
      deadline: "2025-Q1"
      owner: "Security"

    - action: "Create NDA template and signing workflow"
      priority: "MEDIUM"
      deadline: "2025-Q1"
      owner: "Legal"
```

### 6.7 GDPR/DORA Breach Coordination — NEW

When a data breach occurs, both GDPR and DORA notification requirements may apply. This section coordinates the overlapping obligations.

```yaml
breach_notification_coordination:
  # When BOTH GDPR and DORA apply to a single incident
  scenario: "Personal data breach affecting EU regulated clients"

  timeline_comparison:
    gdpr_72h:
      requirement: "DPA notification within 72 hours of awareness"
      applies_when: "Personal data breach affecting EU data subjects"
      our_role: "Notify client (data controller) immediately; they notify DPA"

    dora_client_notification:
      requirement: "Client notification <30 min (critical)"
      applies_when: "ICT incident affecting client's critical functions"
      our_role: "Direct notification to client per SLA"

    dora_nca_notification:
      requirement: "Client must notify NCA (initial: within 4 hours, detailed: within 72h)"
      applies_when: "Major ICT-related incident"
      our_role: "Provide incident report to client for their NCA submission"

  coordinated_response_timeline:
    t_plus_0: "Incident detection"
    t_plus_15min: "Internal classification (ICT incident vs data breach vs both)"
    t_plus_30min: "Client notification (critical ICT incident)"
    t_plus_1h: "Initial incident report to client"
    t_plus_4h: "Detailed information for client's NCA initial notification"
    t_plus_24h: "Updated incident report with root cause progress"
    t_plus_72h: "Support client's GDPR DPA notification if personal data involved"
    t_plus_final: "Full incident report for client's regulatory submissions"

  our_responsibilities:
    as_processor:
      - "Notify controller (client) without undue delay"
      - "Provide information needed for GDPR notification"
      - "Cooperate in investigation"
      - "Document all actions taken"

    as_ict_provider:
      - "Notify client per DORA SLA requirements"
      - "Provide technical incident details"
      - "Support client's NCA notification preparation"
      - "Participate in post-incident review"

  information_we_provide:
    for_gdpr_notification:
      - "Nature of the breach"
      - "Categories of data affected"
      - "Approximate number of records"
      - "Contact point for more information"
      - "Likely consequences"
      - "Measures taken to address breach"

    for_dora_notification:
      - "Incident type and classification"
      - "Root cause (preliminary and final)"
      - "Impact on services"
      - "Actions taken and planned"
      - "Recovery timeline"
      - "Preventive measures for future"

  dual_notification_template:
    # Single incident report format covering both requirements
    sections:
      - incident_identification
      - timeline_of_events
      - data_impact_assessment  # GDPR focus
      - service_impact_assessment  # DORA focus
      - root_cause_analysis
      - remediation_actions
      - prevention_measures
      - lessons_learned

  special_considerations:
    ransomware:
      gdpr: "Personal data breach if data exfiltrated or destroyed"
      dora: "Major ICT incident affecting availability/confidentiality"
      action: "Dual notification likely required"

    supply_chain:
      gdpr: "Notify if subprocessor breach affects personal data"
      dora: "Notify if affects services to client"
      action: "Coordinate with subcontractor and client"
```

### 6.8 NCA Jurisdiction FAQ — NEW

Common questions about regulatory jurisdiction and inspection rights.

```yaml
nca_jurisdiction_faq:

  q1_which_nca:
    question: "If our client is a German bank but we're registered in [Country X], which NCA has jurisdiction?"
    answer: |
      The client's home NCA (BaFin in this case) has inspection rights THROUGH the
      client's contract with us. Our registration location is irrelevant for
      Art. 30(3)(e) purposes.

      Legal basis: Art. 30(3)(e) requires contracts to include "rights of access,
      inspection and audit by the financial entity and its competent authority."

      Practical implication: We must allow BaFin (or any client's NCA) to:
      - Access our premises for inspection (with notice)
      - Review relevant documentation
      - Interview key personnel

      We are NOT directly supervised by any NCA unless designated as CTPP.

  q2_nca_scope:
    question: "Can a client's NCA inspect ALL our systems, including other clients' data?"
    answer: |
      No. The scope is limited to:
      - Systems and data relevant to THAT client's services
      - Compliance with contractual obligations
      - Our security controls and procedures (general)

      We protect other clients' data through:
      - Logical segregation of client data
      - Redaction of other client information
      - Escorted access with scope limitations
      - Written scope agreement before inspection

  q3_multiple_ncas:
    question: "If we have clients in Germany, France, and Luxembourg, can we have 3 NCAs inspecting us simultaneously?"
    answer: |
      Yes, in theory. Each client's NCA has independent inspection rights.

      Mitigation strategies:
      - Offer pooled audit reports (Art. 30(4))
      - Maintain comprehensive audit documentation
      - Coordinate inspection schedules where possible
      - SOC 2 Type II report as baseline evidence

  q4_nca_vs_ctpp:
    question: "How does NCA inspection differ if we become designated as CTPP?"
    answer: |
      Current (Non-CTPP):
      - NCA access via client contracts only
      - No direct regulatory relationship
      - Inspections triggered by client request

      If designated CTPP:
      - Direct oversight by Lead Overseer ESA
      - Annual oversight plan
      - Direct requests for information
      - Potential recommendations and penalty powers
      - Art. 31-44 obligations apply

  q5_data_residency:
    question: "If our data is processed in the US (AWS US regions), does this create issues for EU NCA inspection?"
    answer: |
      Not directly for inspection rights, but considerations include:

      1. GDPR implications: EU-US data transfer mechanisms required (DPF, SCCs)
      2. Art. 30(2)(b): Must disclose US data processing/storage locations
      3. Art. 29: Assess "any constraint that may arise in respect to the urgent
         recovery of the financial entity's data" from third-country providers
      4. NCA access: We provide NCA access to documentation/systems; physical data
         location doesn't prevent this

      Our approach:
      - Offer EU-only deployment option (AWS EU-West-1, EU-Central-1)
      - Document all data locations in contract
      - Ensure SCCs in place with US subcontractors
      - Data export capability from any region

  q6_legal_basis:
    question: "What is our legal basis for providing information to a client's NCA?"
    answer: |
      Primary: Contractual obligation per Art. 30(3)(e)

      The contract with our client REQUIRES us to cooperate with their NCA.
      This is a contractual duty, not a direct regulatory relationship.

      If we refuse:
      - Breach of contract with client
      - Client may be forced to terminate (regulatory pressure)
      - Reputational damage

      What we cannot be compelled to provide:
      - Other clients' confidential information
      - Proprietary source code (unless directly relevant)
      - Information unrelated to the client's services
```

### 6.9 Coordinated Multi-Client Breach Notification — NEW (v2.1)

**Scenario:** Security incident affecting shared infrastructure impacts multiple EU regulated clients.

```yaml
multi_client_breach_notification:
  # =========================================================================
  # SCENARIO DEFINITION
  # =========================================================================
  scenario:
    description: "Data breach or major ICT incident affecting shared platform infrastructure"
    affected_clients: "Multiple EU regulated financial entities"
    example: "Database compromise exposing client configuration data"

  # =========================================================================
  # PARALLEL NOTIFICATION CHALLENGE
  # =========================================================================
  challenge:
    problem: |
      With 5+ EU regulated clients, we must notify ALL affected clients within
      30 minutes (critical) to 60 minutes (high) while:
      - Maintaining consistent messaging
      - Avoiding information leakage between clients
      - Preserving evidence and timeline for each client's NCA reporting
      - Managing limited incident response resources

    compliance_pressure:
      - "Each client has 4-hour NCA initial notification deadline"
      - "Each client needs incident details for their DORA report"
      - "NCAs may compare reports from different clients"

  # =========================================================================
  # COORDINATED NOTIFICATION PROCEDURE
  # =========================================================================
  procedure:
    phase_1_detection_and_classification:
      duration: "T+0 to T+15 minutes"
      actions:
        - "Incident detected and verified"
        - "Initial scope assessment"
        - "Identify ALL potentially affected clients"
        - "Classify incident severity"
        - "Activate incident commander"
      output: "Affected client list with impact assessment per client"

    phase_2_notification_preparation:
      duration: "T+15 to T+25 minutes"
      actions:
        - "Prepare TEMPLATE notification message"
        - "Customize per-client details (what data/services affected)"
        - "Assign notification responsibility (who calls/emails whom)"
        - "Prepare status page update"
        - "Legal review of messaging (if time permits, otherwise post-facto)"
      output: "Client-specific notification messages ready"

    phase_3_parallel_notification:
      duration: "T+25 to T+30 minutes"
      method: "SIMULTANEOUS notification to all affected clients"
      execution:
        - "Multiple team members notify in parallel"
        - "Primary: Phone call to designated security contact"
        - "Backup: Email with HIGH PRIORITY flag"
        - "Webhook/API notification to client systems (if configured)"
      documentation:
        - "Record exact notification time per client"
        - "Record acknowledgment status"
        - "Record contact name/method"

    phase_4_follow_up:
      duration: "T+30 to T+4 hours"
      actions:
        - "Provide written incident summary to each client"
        - "Answer client-specific questions"
        - "Prepare NCA-ready incident report template"
        - "Update status page with progress"
      deliverable: "Each client has information needed for their NCA filing"

  # =========================================================================
  # NOTIFICATION MESSAGE TEMPLATE
  # =========================================================================
  notification_template:
    initial_notification:
      subject: "URGENT: ICT Security Incident Notification - [INCIDENT_ID]"
      content: |
        This is an urgent notification per our DORA contractual obligations.

        INCIDENT SUMMARY:
        - Incident ID: [INCIDENT_ID]
        - Detection Time: [TIMESTAMP_UTC]
        - Classification: [CRITICAL/HIGH]
        - Nature: [Brief description]

        YOUR SPECIFIC IMPACT:
        - Services affected: [Client-specific]
        - Data potentially affected: [Client-specific]
        - Current service status: [Operational/Degraded/Unavailable]

        IMMEDIATE ACTIONS:
        - [Actions we are taking]

        NEXT UPDATE:
        - Expected within [60 minutes / 4 hours]

        CONTACTS:
        - Incident Commander: [Name, Phone, Email]
        - Status Page: [URL]

        This notification is provided to support your regulatory obligations.

    follow_up_report:
      delivered_within: "4 hours"
      contents:
        - "Detailed incident timeline"
        - "Root cause (preliminary if investigation ongoing)"
        - "Full scope of impact"
        - "Remediation actions taken"
        - "Remediation actions planned"
        - "Information for NCA notification"

  # =========================================================================
  # RESOURCE REQUIREMENTS
  # =========================================================================
  resource_requirements:
    minimum_team_for_multi_client:
      incident_commander: 1
      technical_responders: 2
      client_notification_team: "1 per 3 clients (parallel calling)"
      communications_lead: 1

    example_5_clients:
      notification_team_size: 2  # Can notify 5 clients in 5 minutes
      total_team_size: 6

    on_call_implications:
      note: "Multi-client incidents require rapid team assembly"
      escalation: "Page full incident team, not just primary on-call"

  # =========================================================================
  # CLIENT CONFIDENTIALITY
  # =========================================================================
  confidentiality_controls:
    principle: "Each client only learns about their own impact"
    controls:
      - "No disclosure of other affected clients' identities"
      - "No disclosure of other clients' specific data/configurations"
      - "Generic messaging about 'platform-wide' vs 'your specific' impact"
      - "Separate incident reports per client"

    exception:
      scenario: "NCA requests information about incident affecting multiple clients"
      response: "Coordinate with legal; provide aggregated statistics without identifying other clients"

  # =========================================================================
  # DOCUMENTATION FOR AUDIT
  # =========================================================================
  audit_trail:
    per_incident:
      - incident_id
      - detection_timestamp
      - classification_timestamp
      - client_list_finalized_timestamp
      - notification_sent_timestamps_per_client
      - acknowledgment_timestamps_per_client
      - follow_up_report_timestamps

    retention: "7 years"
    format: "Immutable audit log"
    availability: "Available for client and NCA audit requests"

  # =========================================================================
  # TESTING
  # =========================================================================
  testing:
    tabletop_exercise:
      frequency: "Annual"
      scenario: "Simulated multi-client breach"
      participants: "Incident response team + client contacts (optional)"
      success_criteria:
        - "All notifications sent within 30 minutes"
        - "Consistent messaging across clients"
        - "No cross-client information leakage"

    after_action_review:
      trigger: "Any real multi-client incident"
      focus:
        - "Notification timeline accuracy"
        - "Resource adequacy"
        - "Process improvements"
```

### 6.10 NCA Jurisdiction Matrix — NEW (v2.1)

Quick reference for determining which NCA has inspection rights based on client jurisdiction.

```yaml
nca_jurisdiction_matrix:
  # =========================================================================
  # EU MEMBER STATE NCAs
  # =========================================================================
  eu_member_states:
    DE:
      country: "Germany"
      primary_nca: "BaFin (Bundesanstalt für Finanzdienstleistungsaufsicht)"
      website: "https://www.bafin.de"
      dora_contact: "IT-Aufsicht@bafin.de"
      language: "German preferred, English accepted"
      inspection_style: "Thorough, documentation-heavy"
      notes: "May request German translations of key documents"

    FR:
      country: "France"
      primary_nca: "ACPR (Autorité de contrôle prudentiel et de résolution)"
      secondary_nca: "AMF (Autorité des marchés financiers)"
      website: "https://acpr.banque-france.fr"
      language: "French preferred, English accepted"
      notes: "ACPR for banks, AMF for investment firms"

    NL:
      country: "Netherlands"
      primary_nca: "AFM (Autoriteit Financiële Markten)"
      secondary_nca: "DNB (De Nederlandsche Bank)"
      website: "https://www.afm.nl"
      language: "English widely accepted"
      inspection_style: "Risk-based, proportionate"

    IE:
      country: "Ireland"
      primary_nca: "Central Bank of Ireland"
      website: "https://www.centralbank.ie"
      language: "English"
      notes: "Common for fintech, pragmatic approach"

    LU:
      country: "Luxembourg"
      primary_nca: "CSSF (Commission de Surveillance du Secteur Financier)"
      website: "https://www.cssf.lu"
      language: "French, English accepted"
      notes: "Significant funds industry presence"

    ES:
      country: "Spain"
      primary_nca: "CNMV (Comisión Nacional del Mercado de Valores)"
      secondary_nca: "Banco de España"
      language: "Spanish preferred"

    IT:
      country: "Italy"
      primary_nca: "CONSOB"
      secondary_nca: "Banca d'Italia"
      language: "Italian preferred"

    AT:
      country: "Austria"
      primary_nca: "FMA (Finanzmarktaufsicht)"
      website: "https://www.fma.gv.at"
      language: "German"

    BE:
      country: "Belgium"
      primary_nca: "FSMA"
      secondary_nca: "NBB (National Bank of Belgium)"
      language: "Dutch, French, English accepted"

    PT:
      country: "Portugal"
      primary_nca: "CMVM"
      secondary_nca: "Banco de Portugal"
      language: "Portuguese preferred"

  # =========================================================================
  # NON-EU EEA
  # =========================================================================
  eea_non_eu:
    NO:
      country: "Norway"
      primary_nca: "Finanstilsynet"
      dora_status: "Expected to adopt DORA via EEA agreement"

    LI:
      country: "Liechtenstein"
      primary_nca: "FMA Liechtenstein"
      dora_status: "Expected to adopt DORA via EEA agreement"

    IS:
      country: "Iceland"
      primary_nca: "FME"
      dora_status: "Expected to adopt DORA via EEA agreement"

  # =========================================================================
  # POST-BREXIT UK
  # =========================================================================
  uk:
    country: "United Kingdom"
    primary_nca: "FCA (Financial Conduct Authority)"
    secondary_nca: "PRA (Prudential Regulation Authority)"
    dora_status: "NOT subject to DORA"
    notes: |
      UK has separate operational resilience framework (PS21/3).
      UK clients are NOT within DORA scope.
      However, UK subsidiaries of EU firms may be indirectly affected.

  # =========================================================================
  # CTPP DESIGNATION - ESA LEAD OVERSEERS
  # =========================================================================
  ctpp_oversight:
    note: "Only applies if designated as Critical Third-Party Provider"
    lead_overseers:
      EBA: "European Banking Authority - for banking sector"
      ESMA: "European Securities and Markets Authority - for securities"
      EIOPA: "European Insurance and Occupational Pensions Authority - for insurance"

    selection_criteria: |
      Lead Overseer determined by which sector has greatest exposure to the CTPP.
      Joint Committee coordinates between ESAs.

  # =========================================================================
  # PRACTICAL GUIDANCE
  # =========================================================================
  practical_guidance:
    inspection_request_received:
      step_1: "Verify request authenticity (official letterhead, contact details)"
      step_2: "Confirm client relationship and contract in place"
      step_3: "Notify client's compliance team"
      step_4: "Acknowledge request within 24 hours"
      step_5: "Schedule inspection within 5 business days"
      step_6: "Prepare evidence package"

    language_considerations:
      policy: "Maintain key documentation in English as baseline"
      on_request: "Provide translations for material documents if required"
      cost: "Translation costs may be passed to requesting party per contract"

    multi_jurisdiction_clients:
      scenario: "Client operates in multiple EU countries"
      answer: "Home Member State NCA has primary oversight"
      cooperation: "Host NCAs may participate via coordination"

    simultaneous_inspections:
      scenario: "Multiple NCAs request inspection for different clients"
      approach:
        - "Coordinate schedules where possible"
        - "Offer pooled audit report as alternative"
        - "Maintain separate evidence packages per client"
        - "Document each inspection separately"
```

### 6.11 Insurance & Indemnification — NEW v2.2

**Context:** While DORA doesn't explicitly mandate insurance, EU regulated clients often require proof of adequate coverage and indemnification provisions as part of their third-party risk management.

```yaml
insurance_indemnification:
  # =========================================================================
  # INSURANCE REQUIREMENTS
  # =========================================================================
  insurance_types:

    professional_liability:
      name: "Professional Liability / Errors & Omissions (E&O)"
      purpose: "Covers claims arising from professional services, advice, or negligence"
      relevance: "Trading platform errors, strategy execution failures"
      recommended_coverage: "€1-5 million per claim"
      typical_cost: "€5,000-15,000/year"
      client_requirement_frequency: "HIGH - most regulated clients require"

    cyber_liability:
      name: "Cyber Liability Insurance"
      purpose: "Covers data breaches, cyber attacks, business interruption"
      relevance: "Data breach costs, ransomware, incident response"
      recommended_coverage: "€2-10 million per incident"
      typical_cost: "€10,000-30,000/year"
      client_requirement_frequency: "VERY HIGH - essential for financial services"
      coverage_includes:
        - "Incident response costs"
        - "Notification costs"
        - "Forensics investigation"
        - "Legal defense"
        - "Regulatory fines (where insurable)"
        - "Business interruption"
        - "Data restoration"

    general_liability:
      name: "General Liability (Public Liability)"
      purpose: "Covers general business operations, premises liability"
      relevance: "Basic business operations coverage"
      recommended_coverage: "€1-2 million"
      typical_cost: "€2,000-5,000/year"
      client_requirement_frequency: "MEDIUM"

    directors_officers:
      name: "Directors & Officers (D&O) Insurance"
      purpose: "Protects company leadership from personal liability"
      relevance: "Corporate governance, fiduciary duties"
      recommended_coverage: "€1-5 million"
      typical_cost: "€5,000-20,000/year"
      client_requirement_frequency: "LOW - enterprise clients may ask"

  # =========================================================================
  # RECOMMENDED COVERAGE BY STAGE
  # =========================================================================
  coverage_by_stage:

    startup_phase:
      revenue: "<€500K ARR"
      clients: "Mostly retail, few regulated"
      recommended:
        professional_liability: "€1 million"
        cyber_liability: "€2 million"
        general_liability: "€1 million"
      estimated_annual_cost: "€10,000-20,000"

    growth_phase:
      revenue: "€500K-2M ARR"
      clients: "Mix of retail and regulated"
      recommended:
        professional_liability: "€2 million"
        cyber_liability: "€5 million"
        general_liability: "€1 million"
        directors_officers: "€1 million"
      estimated_annual_cost: "€25,000-50,000"

    scale_phase:
      revenue: ">€2M ARR"
      clients: "Significant regulated client base"
      recommended:
        professional_liability: "€5 million"
        cyber_liability: "€10 million"
        general_liability: "€2 million"
        directors_officers: "€5 million"
      estimated_annual_cost: "€50,000-100,000"

  # =========================================================================
  # CLIENT DISCLOSURE
  # =========================================================================
  client_disclosure:
    standard_disclosure:
      - "Insurance types held"
      - "Coverage amounts (ranges acceptable)"
      - "Insurance provider (upon request)"
      - "Policy expiry dates"

    certificate_of_insurance:
      availability: "Upon request"
      turnaround: "3 business days"
      content:
        - "Named insured"
        - "Policy number"
        - "Coverage type and amount"
        - "Policy period"
        - "Certificate holder (client)"

    additional_insured:
      availability: "Enterprise tier clients"
      process: "Request to insurance broker"
      turnaround: "5-10 business days"
      additional_cost: "May increase premium"

  # =========================================================================
  # INDEMNIFICATION PROVISIONS
  # =========================================================================
  indemnification:

    standard_indemnification:
      we_indemnify_client_for:
        - "Third-party IP infringement claims"
        - "Gross negligence in service delivery"
        - "Willful misconduct"
        - "Data breaches caused by our security failures"
        - "Regulatory fines arising from our GDPR violations"

      client_indemnifies_us_for:
        - "Misuse of platform by client or their users"
        - "Client's own regulatory violations"
        - "Inaccurate data provided by client"
        - "Third-party claims from client's end users"

    liability_caps:
      standard_tier:
        cap: "12 months of fees paid"
        exceptions: "Gross negligence, willful misconduct, IP indemnity"

      professional_tier:
        cap: "24 months of fees paid"
        exceptions: "Gross negligence, willful misconduct, IP indemnity"

      enterprise_tier:
        cap: "Negotiated (typically 24-36 months)"
        exceptions: "Gross negligence, willful misconduct, IP indemnity"
        note: "May require increased insurance coverage"

    carve_outs:
      unlimited_liability:
        - "Fraud"
        - "Willful misconduct"
        - "Gross negligence"
        - "Confidentiality breaches"
        - "IP indemnification"
      note: "These are typically excluded from caps"

    exclusions:
      we_do_not_cover:
        - "Trading losses (market risk)"
        - "Broker failures (client's broker relationship)"
        - "Client's own regulatory non-compliance"
        - "Force majeure events"
        - "Third-party service outages (with notification)"

  # =========================================================================
  # CONTRACT TEMPLATE CLAUSES
  # =========================================================================
  contract_clauses:

    insurance_clause: |
      INSURANCE

      Provider shall maintain the following insurance coverage during
      the term of this Agreement:

      (a) Professional Liability Insurance: €[AMOUNT] per claim
      (b) Cyber Liability Insurance: €[AMOUNT] per incident
      (c) General Liability Insurance: €[AMOUNT] per occurrence

      Upon Client's request, Provider shall provide a certificate of
      insurance evidencing such coverage within five (5) business days.

    indemnification_clause: |
      INDEMNIFICATION

      1. Provider Indemnification. Provider shall indemnify, defend,
         and hold harmless Client from any third-party claims arising
         from: (a) Provider's gross negligence or willful misconduct;
         (b) Provider's infringement of third-party intellectual property;
         (c) Provider's breach of data protection obligations.

      2. Client Indemnification. Client shall indemnify, defend, and
         hold harmless Provider from any claims arising from: (a) Client's
         misuse of the Services; (b) Client's violation of applicable laws;
         (c) claims by Client's end users.

      3. Limitation of Liability. Except for indemnification obligations
         and breaches of confidentiality, neither party's aggregate
         liability shall exceed [AMOUNT/12 months fees].

  # =========================================================================
  # CURRENT STATUS
  # =========================================================================
  current_status:
    professional_liability:
      status: "[HAVE/PLANNED/NONE]"
      coverage: "[AMOUNT]"
      provider: "[INSURER]"
      expiry: "[DATE]"

    cyber_liability:
      status: "[HAVE/PLANNED/NONE]"
      coverage: "[AMOUNT]"
      provider: "[INSURER]"
      expiry: "[DATE]"

    general_liability:
      status: "[HAVE/PLANNED/NONE]"
      coverage: "[AMOUNT]"

  # =========================================================================
  # ACTION ITEMS
  # =========================================================================
  action_items:
    - action: "Obtain cyber liability insurance quote"
      priority: "HIGH"
      deadline: "2025-Q1"
      owner: "Finance/Operations"

    - action: "Obtain professional liability insurance quote"
      priority: "HIGH"
      deadline: "2025-Q1"
      owner: "Finance/Operations"

    - action: "Create insurance disclosure document for clients"
      priority: "MEDIUM"
      deadline: "2025-Q1"
      owner: "Legal"

    - action: "Update contract templates with indemnification clauses"
      priority: "MEDIUM"
      deadline: "2025-Q1"
      owner: "Legal"
```

### 6.12 Pooled Audit Framework — NEW v2.2

**Legal Basis:** Art. 30(4) states "financial entities may, either individually or collectively, use pooled audits... or use third-party certifications."

```yaml
pooled_audit_framework:
  # =========================================================================
  # PURPOSE
  # =========================================================================
  purpose: |
    Enable multiple clients to satisfy their audit rights (Art. 30(3)(e))
    efficiently through shared audit arrangements, reducing burden on both
    provider and clients while maintaining compliance.

  # =========================================================================
  # POOLED AUDIT OPTIONS
  # =========================================================================
  options:

    option_1_soc2_reliance:
      name: "SOC2 Type II Report Reliance (Roadmap)"
      description: "SOC2 Type II certification is on the roadmap; upon achievement, clients may rely on periodic reports"
      legal_basis: "Art. 30(4) - third-party certifications (if applicable)"
      status: "PLANNED - SOC2 roadmap in progress"

      what_we_provide_upon_certification:
        - "SOC2 Type II report (if/when obtained)"
        - "Bridge letter between reports (if applicable)"
        - "Management letter (if applicable)"
        - "Auditor contact for verification (if applicable)"

      client_benefits:
        - "No need for individual audit"
        - "Immediate availability (under NDA)"
        - "Trusted third-party attestation"
        - "Covers most security controls"

      limitations:
        - "May not cover all DORA-specific requirements"
        - "Client may need supplementary review"
        - "Report scope may not match client's exact needs"

      cost_to_client: "€0 (included in subscription)"
      our_cost: "€30,000-80,000/year (SOC2 audit)"

    option_2_iso27001_reliance:
      name: "ISO/IEC 27001 (Optional Roadmap)"
	      description: "Optional future audit/certification path to support procurement (no current certification claim)"
      legal_basis: "Art. 30(4) - third-party certifications (if applicable)"
      status: "PLANNED - evaluation/roadmap"
      target_date: "Q4 2025"

      what_we_provide:
        - "ISO/IEC 27001 certificate (if achieved)"
        - "Statement of Applicability"
        - "Annual surveillance audit reports"

      client_benefits:
        - "Internationally recognized standard"
	        - "Surveillance audits (if applicable)"
        - "Comprehensive ISMS coverage"

    option_3_joint_audit:
      name: "Joint/Pooled Audit"
      description: "Multiple clients conduct audit together"
      legal_basis: "Art. 30(4) - pooled audits"

      structure:
        coordinator: "One client or third-party audit firm"
        participants: "Multiple clients share costs"
        scope: "Common scope agreed by participants"
        timing: "Annual, coordinated schedule"

      what_we_provide:
        - "Audit coordination support"
        - "Documentation package"
        - "Personnel availability"
        - "Facility access (if needed)"

      cost_sharing:
        coordinator_fee: "€5,000-15,000"
        per_participant: "€2,000-5,000"
        our_support_cost: "Included (up to 2 days/year)"

      benefits:
        - "Reduced cost per client"
        - "Comprehensive scope"
        - "Direct auditor access"
        - "Custom focus areas possible"

    option_4_individual_audit:
      name: "Individual Client Audit"
      description: "Single client exercises full audit rights"
      legal_basis: "Art. 30(3)(e) - unrestricted audit rights"

      what_we_provide:
        - "Full audit access"
        - "Documentation"
        - "Personnel interviews"
        - "Systems access (read-only)"
        - "Facility access"

      conditions:
        notice_period: "10 business days (5 for cause)"
        duration: "Up to 5 days on-site"
        frequency: "Annual (more for cause)"
        auditor_approval: "Must be reputable firm"

      cost_to_client: "Client bears their audit costs"
      our_support_cost:
        included: "1 individual audit per year per Enterprise client"
        additional: "€5,000/day for additional audits"

  # =========================================================================
  # RECOMMENDATION BY CLIENT TYPE
  # =========================================================================
  recommendations:

    standard_tier_clients:
      primary: "Option 1 (SOC2 Reliance)"
      rationale: "Cost-effective, sufficient for non-critical functions"
      supplementary: "Option 3 (Joint Audit) if needed"

    professional_tier_clients:
      primary: "Option 1 (SOC2 Reliance)"
      supplementary: "Option 3 (Joint Audit) for specific concerns"
      alternative: "Option 4 (Individual Audit) if required by their NCA"

    enterprise_tier_clients:
      primary: "Option 1 (SOC2) + Option 2 (ISO 27001)"
      supplementary: "Option 4 (Individual Audit) rights included"
      custom: "Can coordinate Option 3 (Joint Audit)"

  # =========================================================================
  # POOLED AUDIT COORDINATION
  # =========================================================================
  pooled_audit_coordination:

    annual_cycle:
      january: "Announce pooled audit availability"
      february: "Collect interest from clients"
      march: "Finalize participants and scope"
      april: "Select audit firm (if client-led)"
      may_june: "Conduct audit"
      july: "Report distribution"
      ongoing: "SOC2 report available upon request (planned; verify actual audit status with vendor pack)"

    scope_options:
      standard_scope:
        - "Security controls (SOC2 CC series)"
        - "Availability controls"
        - "Incident response procedures"
        - "Change management"
        - "Backup and recovery"

      dora_enhanced_scope:
        - "Standard scope plus:"
        - "ICT risk management framework"
        - "Business continuity testing"
        - "Exit strategy documentation"
        - "Subcontractor oversight"

    client_communication:
      invitation: "December (for following year)"
      confirmation_deadline: "January 31"
      scope_finalization: "February 28"
      audit_dates: "April-May"
      report_delivery: "June 30"

  # =========================================================================
  # AUDIT EVIDENCE PACKAGE
  # =========================================================================
  evidence_package:
    always_available:
      - "SOC2 Type II report (under NDA)"
      - "Security policy summary"
      - "Incident history (sanitized)"
      - "BCP/DR test results (summary)"
      - "Penetration test executive summary"
      - "Subcontractor list"

    upon_request:
      - "Detailed control documentation"
      - "Full penetration test report"
      - "Vulnerability scan results"
      - "Access review logs"
      - "Change management records"

    audit_only:
      - "Source code review"
      - "Live system access"
      - "Personnel interviews"
      - "Physical facility inspection"

  # =========================================================================
  # CONTRACTUAL PROVISIONS
  # =========================================================================
  contract_provisions:

    audit_rights_clause: |
      AUDIT RIGHTS (Art. 30(3)(e) / Art. 30(4))

      1. Client shall have the right to audit Provider's compliance with
         this Agreement, either directly or through appointed auditors.

      2. Client may satisfy audit requirements through:
         (a) Reliance on Provider's SOC 2 Type II report and/or ISO 27001
             attestation/certificate (if available; copies upon request under NDA);
         (b) Participation in pooled audits organized by Provider or
             jointly with other clients;
         (c) Individual audit conducted by Client or Client's auditors.

      3. For individual audits, Client shall provide at least ten (10)
         business days' notice (five (5) days for cause-based audits).

      4. Provider shall cooperate fully with any audit conducted by
         Client's competent authority (NCA) per Art. 30(3)(e).

      5. Audit costs shall be borne by Client, except that Provider
         shall provide [X] days of support at no additional cost per year.

  # =========================================================================
  # CURRENT STATUS
  # =========================================================================
  current_status:
    soc2_type2:
      status: "IN PROGRESS"
      target_completion: "Q4 2025"
      auditor: "[TBD]"

    iso27001:
      status: "PLANNED"
      target_completion: "Q4 2025"

    pooled_audit_2025:
      status: "PLANNED"
      interest_collection: "Q4 2024"

  # =========================================================================
  # ACTION ITEMS
  # =========================================================================
  action_items:
    - action: "Complete SOC2 Type II readiness assessment"
      priority: "HIGH"
      deadline: "2025-Q1"
      owner: "Security"

    - action: "Select SOC2 auditor"
      priority: "HIGH"
      deadline: "2025-Q1"
      owner: "Finance + Security"

    - action: "Create pooled audit invitation template"
      priority: "MEDIUM"
      deadline: "2025-Q1"
      owner: "Security"

    - action: "Document audit evidence package contents"
      priority: "MEDIUM"
      deadline: "2025-Q1"
      owner: "Security"
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

## 8. Cleanup Plan — REVISED v2.2

### 8.1 Archive (Phase 1) — REVISED v2.4

| Original Location | Archived To | Reason |
|-------------------|-------------|--------|
| `services/dora/scope_verification.py` | `services/archive/dora_financial_entity/` | FE-specific — we're ICT Provider (Art. 30) |
| `services/dora/proportionality.py` | `services/archive/dora_financial_entity/` | FE regime determination |
| `services/dora/supervisory_feedback.py` | `services/archive/dora_financial_entity/` | Client-NCA communication |
| `config/dora/nca_identification.yaml` | `services/archive/dora_financial_entity/configs/` | Client identifies their NCA |
| `config/dora/entity_classification.yaml` | `services/archive/dora_financial_entity/configs/` | FE classification config |
| `config/dora/proportionality_assessment.yaml` | `services/archive/dora_financial_entity/configs/` | FE proportionality config |

**Note v2.4:** All FE-specific modules and configs consolidated in `services/archive/dora_financial_entity/`.
Active ICT Provider configs are in `configs/dora/`.

### 8.2 Adapt (Phase 1-2) — REVISED v2.2

| Module | Current | Target |
|--------|---------|--------|
| `register_of_information.py` | ROI submission | `provider_info_package.py` — data FOR client ROI |
| `exit_strategies.py` | Client exit planning | Provider exit support + data portability |
| `third_party_risk.py` | Client risk assessment | Self-documentation + subcontractor info |
| `contractual_requirements.py` | DORA clauses | Contract templates with Art. 30 clauses |
| `concentration_risk.py` | Client analysis | CTPP designation awareness + monitoring |
| `pooled_testing.py` | Client pooled testing | `pooled_audit_support.py` — support client pooled audits (Art. 30(4)) |

### 8.3 Keep (Core)

| Module | Reason |
|--------|--------|
| `incident_management.py` | Core incident handling |
| `incident_reporting.py` | Client notification |
| `backup_recovery.py` | Core backup |
| `ict_business_continuity.py` | Core BCP |
| `detection.py` | Anomaly detection |
| `protection.py` | Security controls |
| `ctpp_oversight.py` | **NEW v2.2:** Keep scaled-down for CTPP preparedness |

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

## 9. Phased Roadmap — REVISED v2.2

### Phase 1: Contractual Alignment & Baseline ✅ COMPLETED

**Status:** COMPLETED (2025-12-09)
**Test Coverage:** 474 tests passing (100%)
**CI Verification**: `.github/workflows/build-and-test.yml` (pytest runs on every PR/push)
**Test Report Artifact**: CI produces test results on each run; historical results in Actions logs

**Goals:**
- ✅ Enable procurement-ready contract templates with EU clients (no certification claim)
- ✅ Establish evidence-pack readiness (internal)
- ✅ Clean up non-applicable modules
- ✅ Validate operational capacity for SLA commitments

**Work Blocks:**

| Block | Description | Priority | Status | Deliverable |
|-------|-------------|----------|--------|-------------|
| 1.1 | Create contract templates with Art. 30(2) clauses | **CRITICAL** | ✅ | `docs/contracts/DORA_CONTRACT_TEMPLATE_ART_30_2.md` |
| 1.2 | Create critical function addendum (Art. 30(3)) | **CRITICAL** | ✅ | `docs/contracts/DORA_CRITICAL_FUNCTION_ADDENDUM_ART_30_3.md` |
| 1.3 | Implement audit readiness procedures | **HIGH** | ✅ | `services/dora/audit_readiness.py` |
| 1.4 | Create provider information package | **HIGH** | ✅ | `services/dora/provider_info_package.py` |
| 1.5 | Document subcontractors (AWS, data providers) | **HIGH** | ✅ | `docs/contracts/SUBCONTRACTOR_REGISTER.md` |
| 1.6 | Adapt exit_strategies.py for provider role | **HIGH** | ✅ | `services/dora/exit_strategies.py` |
| 1.7 | Archive non-applicable modules (revised list) | MEDIUM | ✅ | `services/archive/dora_not_applicable/` |
| 1.8 | Create SHARED_RESPONSIBILITY.md | MEDIUM | ✅ | `docs/SHARED_RESPONSIBILITY.md` |
| 1.9 | Validate on-call capacity → set achievable notification SLA | **HIGH** | ✅ | `docs/operations/ON_CALL_CAPACITY_VALIDATION.md` |
| 1.10 | Create subcontracting prior approval workflow | **HIGH** | ✅ | `services/dora/subcontractor_management.py` |
| 1.11 | Document EU-only data residency configuration | **HIGH** | ✅ | `docs/contracts/EU_DATA_RESIDENCY.md` |
| 1.12 | Obtain cyber liability insurance quotes | **HIGH** | ✅ | `docs/contracts/INSURANCE_REQUIREMENTS.md` |
| 1.13 | Implement SLA guardrails (engineering sign-off process) | **HIGH** | ✅ | `services/dora/sla_guardrails.py` |
| 1.14 | Create pre-contractual public security page | MEDIUM | ✅ | `docs/security/TRUST_CENTER.md` |
| 1.15 | Adapt pooled_testing.py → pooled_audit_support.py | MEDIUM | ✅ | `services/dora/pooled_audit_support.py` |

**Deliverables (All Completed):**
- ✅ DORA-aligned contract templates (incl. subcontracting approval)
- ✅ Audit readiness package
- ✅ Provider information package for client ROI
- ✅ Subcontractor documentation with incident escalation procedures
- ✅ Exit strategy documentation
- ✅ Updated incident notification procedures (with realistic SLAs)
- ✅ EU data residency configuration documentation
- ✅ Insurance coverage documentation
- ✅ Pre-contractual security overview page (Trust Center)

**New Python Modules Created:**
- `services/dora/sla_guardrails.py` - SLA tier validation with engineering sign-off (40 tests)
- `services/dora/pooled_audit_support.py` - Pooled audit coordination per Art. 30(4) (41 tests)

**Critical Constraint:** DO NOT offer Professional/Enterprise SLA tiers until infrastructure validated (see Section 5.4.4).

### Phase 2: Core Operational Resilience

**Goals:**
- Strengthen monitoring/logging/alerting
- Improve DR/BCP with documented RTO/RPO
- Enhance change management
- Enable Professional tier offering

**Work Blocks:**

| Block | Description | Priority | NEW v2.2 |
|-------|-------------|----------|----------|
| 2.1 | Implement tiered backup (15min/1h/24h RPO) | **HIGH** | |
| 2.2 | Enhance healthcheck (/health, /ready, /live) | HIGH | |
| 2.3 | Implement structured logging with correlation IDs | HIGH | |
| 2.4 | Add comprehensive alerting | HIGH | |
| 2.5 | Quarterly DR testing with documentation | HIGH | |
| 2.6 | CI/CD security gates (SAST/DAST) | MEDIUM | |
| 2.7 | SOC2 ↔ DORA control mapping | MEDIUM | |
| 2.8 | Create `services/core/` package | MEDIUM | |
| 2.9 | Implement Multi-AZ deployment | **HIGH** | ✓ |
| 2.10 | Establish formal on-call rotation (Option B or C) | **HIGH** | ✓ |
| 2.11 | Implement subcontractor status monitoring | HIGH | ✓ |
| 2.12 | Deploy trust center platform (Vanta/Drata) | MEDIUM | ✓ |
| 2.13 | Complete first DR test with documentation | **HIGH** | ✓ |
| 2.14 | Implement automated CTPP risk monitoring | MEDIUM | ✓ |

**Deliverables:**
- Tiered backup system with automated testing
- Enhanced monitoring with SLA tracking
- Structured logging across all services
- Quarterly DR test reports
- SOC2-DORA mapping document
- Multi-AZ deployment (Professional tier prerequisite)
- Formal on-call rotation with documented procedures
- Subcontractor incident monitoring system
- Trust center portal (basic)

**Gate:** Professional tier can be offered after:
- [ ] Multi-AZ deployment completed
- [ ] Sync replication enabled
- [ ] On-call rotation established (Option B minimum)
- [ ] First DR test passed

### Phase 3: Enterprise Enhancements

**Goals:**
- Extended reporting for regulated clients
- Joint testing support
- On-prem deployment support
- Enable Enterprise tier offering

**Work Blocks:**

| Block | Description | Priority | NEW v2.2 |
|-------|-------------|----------|----------|
| 3.1 | Create `services/enterprise/` package | HIGH | |
| 3.2 | Extended incident report formats (PDF/JSON) | HIGH | |
| 3.3 | Per-client metrics and dashboards | MEDIUM | |
| 3.4 | SIEM integration (Splunk/ELK export) | MEDIUM | |
| 3.5 | TLPT cooperation procedures | MEDIUM | |
| 3.6 | On-prem deployment guide | MEDIUM | |
| 3.7 | Enterprise SLA templates | MEDIUM | |
| 3.8 | Feature flag system for Enterprise | LOW | |
| 3.9 | Multi-region deployment | **HIGH** | ✓ |
| 3.10 | 24/7 on-call (Option C: 4+ engineers) | **HIGH** | ✓ |
	| 3.11 | Complete SOC 2 Type II audit/attestation (if pursued) | **HIGH** | ✓ |
| 3.12 | Implement pooled audit coordination | MEDIUM | ✓ |
| 3.13 | Dedicated region deployment option | MEDIUM | ✓ |
	| 3.14 | ISO 27001 audit/certification evaluation (optional) | MEDIUM | ✓ |

**Deliverables:**
- Extended incident reporting system
- Per-client monitoring
- SIEM integration
- TLPT cooperation documentation
- On-prem deployment package
- Multi-region deployment capability
- 24/7 on-call team
- SOC2 Type II report
- Pooled audit framework operational
- Dedicated region option

**Gate:** Enterprise tier can be offered after:
- [ ] Multi-region deployment completed
- [ ] 24/7 on-call team (4+ FTE)
- [ ] Quarterly DR tests passing
- [ ] SOC 2 readiness milestone achieved (report/attestation if available)

### Phase Summary Timeline (Illustrative, Budget-Dependent)

> **Disclaimer**: This timeline is **illustrative only**. All phases and dates are subject to funding, resource allocation, and gate criteria completion. Quarters shown are planning placeholders, not commitments.

```
Phase 1 (~Q1 2025, if funded): Contractual foundation
├── Contract templates ready
├── Standard tier only
└── Insurance in place

Phase 2 (~Q2-Q3 2025, dependent on Phase 1 + funding): Operational maturity
├── Professional tier available (gate criteria must pass)
├── SOC 2 program in progress (if auditor engaged)
└── Trust center operational

Phase 3 (~Q4 2025+, dependent on Phase 2 + funding): Enterprise scale
├── Enterprise tier available (gate criteria must pass)
├── SOC 2 Type II report available (if achieved)
└── Multi-region operational (if deployed)
```

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
| Contract clause coverage (Art. 30) | Target: 100% of EU contracts include Art. 30 clauses | Contract review |
| Audit readiness | Response within 5 business days | Audit log |
| Uptime | 99.9% | Monitoring |
| MTTD (Mean Time to Detect) | <15 min | Incident tracking |
| MTTR (Mean Time to Resolve) | <1h critical, <4h high | Incident tracking |
| Client notification | <30min critical | Incident tracking |
| Backup job success rate | Target: 100% | Backup logs |
| DR test pass rate | Target: 100% per quarterly test | DR test reports |
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

## Appendix A: Files to Archive — REVISED v2.2

```
archive/dora_not_applicable/
├── scope_verification.py
├── proportionality.py
├── supervisory_feedback.py
├── configs/
│   ├── nca_identification.yaml
│   └── entity_classification.yaml
├── tests/
│   └── test_dora_phase0_proportionality.py
└── README.md (explaining ICT provider vs financial entity)
```

**Note v2.2:** `pooled_testing.py` removed from Archive list — see Appendix B for adaptation.

## Appendix B: Files to Adapt — REVISED v2.2

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
├── concentration_risk.py → ctpp_awareness.py
│   Purpose: Monitor our market concentration for CTPP risk
│
└── pooled_testing.py → pooled_audit_support.py  [NEW v2.2]
    Purpose: Support client pooled audits per Art. 30(4)
```

## Appendix C: Reference Documents

| Document | Purpose |
|----------|---------|
| [DORA Article 28](https://www.digital-operational-resilience-act.com/Article_28.html) | ICT third-party risk principles |
| [DORA Article 30](https://www.digital-operational-resilience-act.com/Article_30.html) | Contractual requirements |
| [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md) | Current operations |
| [RECOVERY_PROCEDURES.md](RECOVERY_PROCEDURES.md) | Current recovery |
| [CYBERSECURITY_FRAMEWORK.md](CYBERSECURITY_FRAMEWORK.md) | NIST CSF 2.0 |
| [SOC2_ROADMAP.md](SOC2_ROADMAP.md) | SOC 2 readiness roadmap |

---

**Document Owner**: Platform Team
**Review Cycle**: Quarterly
**Next Review**: Q1 2026
