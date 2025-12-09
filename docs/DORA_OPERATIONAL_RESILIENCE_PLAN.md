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
      - "SOC2 Type II certification"

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
      has_data_access: true  # Client API keys
      data_types_accessed: ["Client exchange credentials (encrypted)"]
      certifications:
        - "Variable by jurisdiction"
        - "See regulatory_risk_assessment below"
      contract_reference: "Binance API Terms"
      is_material: true
      supports_critical_functions: true  # Crypto trading
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
            regulatory_status: "Licensed in multiple EU jurisdictions"
            notes: "Recommended for EU-regulated clients"
          - name: "Coinbase"
            lei: "5493005KJDX9YGBJI252"
            regulatory_status: "US SEC registered, EU licenses"
            notes: "Recommended for US and EU clients"

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
        - "SOC 2 Type II"
        - "ISO 27001"
        - "GDPR compliant"
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
        - "GDPR compliant"
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
