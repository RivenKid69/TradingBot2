# GDPR Integration Plan

## AI-Powered Quantitative Research Platform

**Regulation**: GDPR (EU) 2016/679 - General Data Protection Regulation
**Version**: 1.0
**Date**: December 2024
**Status**: Implementation Ready

---

## Executive Summary

This document provides a phased implementation plan for full GDPR compliance integration into the AI-Powered Quantitative Research Platform. The plan leverages existing compliance infrastructure (MiFID II, DORA, EU AI Act) and follows established architectural patterns.

### Scope of Application

The platform processes:
- **Financial Market Data**: OHLCV, order books, trades (non-personal)
- **User Credentials**: API keys, exchange credentials (sensitive)
- **Audit Logs**: Trading activity, compliance records (may contain personal data)
- **Configuration Data**: User preferences, settings

### Key GDPR Articles to Implement

| Article | Description | Priority | Notes |
|---------|-------------|----------|-------|
| **4** | Definitions (controller, processor, personal data) | Critical | Foundation for all processing |
| **5-6** | Processing Principles & Lawful Basis | Critical | Core compliance |
| **7** | Consent Management | High | Where consent is lawful basis |
| **8** | Child's consent (information society services) | Low | Platform is 18+ only; verify at signup |
| **9** | Special Categories of Personal Data | Critical | Biometric 2FA considerations |
| **10** | Criminal convictions data | Low | Not applicable to trading platform |
| **11** | Processing not requiring identification | Medium | **NEW** - For pseudonymized data handling |
| **12-14** | Transparency & Information Notices | Critical | Layered privacy notices |
| **15-22** | Data Subject Rights (DSAR) | Critical | Full rights implementation |
| **22** | Automated Decision-Making & Profiling | Critical | See detailed classification above |
| **23** | Restrictions (financial services exemptions) | **High** | **CRITICAL** - MiFID II legal basis |
| **24** | Responsibility of the controller | **High** | **NEW** - Accountability framework |
| **25** | Privacy by Design & Default | High | Technical & organizational measures |
| **26** | Joint Controllers | Medium | If applicable |
| **27** | Representatives of non-EU controllers | Low | EU incorporated - N/A |
| **28-29** | Processor & Sub-processor Requirements | Critical | Exchange APIs, cloud providers |
| **30** | Records of Processing Activities (ROPA) | Critical | Mandatory documentation |
| **31** | Cooperation with Supervisory Authority | Medium | SA interaction protocols |
| **32** | Security of Processing | High | Aligns with DORA ICT security |
| **33-34** | Data Breach Notification | Critical | 72h SA, high-risk DS notification |
| **35-36** | DPIA & Prior Consultation | High | Algo trading requires DPIA |
| **37-39** | Data Protection Officer | Medium | DPO tools and interface |
| **40-43** | Codes of conduct & Certification | Low | Future roadmap |
| **44-49** | International Data Transfers | High | SCCs, adequacy, TIAs |
| **77-84** | Remedies, Liability & Penalties | Medium | Liability framework |

**Articles Explicitly Not Implemented (with justification):**

| Article | Reason | Risk |
|---------|--------|------|
| Art. 8 | Platform restricted to 18+; age verification at signup | Low - document in ROPA |
| Art. 10 | No criminal conviction data processed | None |
| Art. 27 | Company EU incorporated | None |
| Art. 40-43 | No certification sought currently | Low - future enhancement |

---

## Architecture Integration

### Directory Structure

```
services/
  gdpr/
    __init__.py                    # Module exports

    # Phase 0: Core Definitions & Processor Framework
    definitions.py                 # Article 4 GDPR definitions
    processor_management.py        # Article 28 processor contracts
    sub_processor_registry.py      # Sub-processor tracking
    joint_controller.py            # Article 26 joint controller agreements
    dpa_generator.py               # Data Processing Agreement templates

    # Phase 1: Foundation
    config.py                      # GDPR configuration
    legal_basis.py                 # Article 6 lawful basis management
    processing_principles.py       # Article 5 principles enforcement
    special_categories.py          # Article 9 special category handling
    accountability.py              # Article 24 controller responsibility (NEW)
    restrictions.py                # Article 23 restrictions framework (NEW)

    # Phase 2a: Consent & Transparency
    consent_manager.py             # Article 7 consent management
    transparency_notices.py        # Articles 12-14 privacy notices
    information_provision.py       # Layered notice approach

    # Phase 2b: Data Subject Rights
    data_subject_rights.py         # Rights framework (Articles 15-22)
    dsar_handler.py                # Data Subject Access Requests
    erasure_manager.py             # Right to Erasure (Article 17)
    portability_manager.py         # Data Portability (Article 20)
    automated_decisions.py         # Article 22 automated decision-making
    restriction_manager.py         # Article 18 restriction of processing
    objection_handler.py           # Article 21 right to object
    no_identification_handler.py   # Article 11 pseudonymized data (NEW)

    # Phase 3: ROPA & Documentation
    ropa.py                        # Records of Processing Activities
    processing_registry.py         # Processing operations registry
    data_mapping.py                # Personal data mapping
    sa_cooperation.py              # Article 31 supervisory authority cooperation

    # Phase 4: Privacy Engineering
    privacy_by_design.py           # Article 25 PbD controls
    data_minimization.py           # Data minimization enforcement
    pseudonymization.py            # Pseudonymization utilities
    retention_manager.py           # Data retention policies (GDPR-MiFID aligned)
    auto_erasure_scheduler.py      # Automatic erasure after retention (NEW)

    # Phase 5: Breach Management
    breach_detection.py            # Breach detection mechanisms
    breach_notification.py         # Articles 33-34 notification
    breach_assessment.py           # Risk assessment for breaches
    incident_response.py           # GDPR-specific incident response

    # Phase 6: DPIA & Governance
    dpia.py                        # Data Protection Impact Assessment
    prior_consultation.py          # Article 36 prior consultation
    dpo_interface.py               # DPO tools and interface
    international_transfers.py     # Articles 44-49 transfers
    uk_adequacy_contingency.py     # UK adequacy sunset handling (NEW)
    compliance_dashboard.py        # GDPR compliance overview
    liability_framework.py         # Articles 77-84 remedies & liability

tests/
  gdpr/
    test_gdpr_phase0_core_processor.py
    test_gdpr_phase1_foundation.py
    test_gdpr_phase2a_consent_transparency.py
    test_gdpr_phase2b_data_subject_rights.py
    test_gdpr_phase3_ropa.py
    test_gdpr_phase4_privacy_engineering.py
    test_gdpr_phase5_breach_management.py
    test_gdpr_phase6_dpia_governance.py
```

### Integration Points

```
Existing Module              → GDPR Integration
─────────────────────────────────────────────────
services/compliance/         → Audit trail, retention
services/dora/               → Incident management, breach
services/ai_act/             → Data governance, logging
services/secure_logging.py   → PII masking, secure logs
adapters/                    → Data flow tracking
```

---

## Phase 0: Core Definitions & Processor Framework

**Estimated Complexity**: Medium
**Dependencies**: None
**Test Coverage Target**: 100%

### 0.1 Objectives

Establish foundational GDPR infrastructure including:
- Article 4 definitions and role classification
- Article 28 processor management
- Article 26 joint controller agreements
- Data Processing Agreement (DPA) generation

### 0.2 Components to Implement

#### 0.2.1 GDPRDefinitions (definitions.py)

Article 4 key definitions mapped to platform context:

```
Enum GDPRRole:
    CONTROLLER = "controller"           # Determines purposes and means
    PROCESSOR = "processor"             # Processes on behalf of controller
    JOINT_CONTROLLER = "joint_controller"  # Joint determination
    SUB_PROCESSOR = "sub_processor"     # Processor's processor

Dataclass PersonalDataCategory:
    category_id: str
    name: str
    description: str
    examples: List[str]
    is_special_category: bool           # Article 9
    requires_explicit_consent: bool
    retention_requirements: Dict[str, int]

# Platform-specific personal data categories:
PLATFORM_DATA_CATEGORIES = {
    "user_identifiers": PersonalDataCategory(
        name="User Identifiers",
        examples=["user_id", "email", "username"],
        is_special_category=False
    ),
    "authentication_data": PersonalDataCategory(
        name="Authentication Data",
        examples=["hashed_passwords", "2fa_secrets", "session_tokens"],
        is_special_category=False
    ),
    "api_credentials": PersonalDataCategory(
        name="API Credentials",
        examples=["exchange_api_keys", "encrypted_secrets"],
        is_special_category=False
    ),
    "trading_activity": PersonalDataCategory(
        name="Trading Activity",
        examples=["orders", "trades", "positions"],
        is_special_category=False
    ),
    "ip_addresses": PersonalDataCategory(
        name="Network Identifiers",
        examples=["ip_address", "device_fingerprint"],
        is_special_category=False
    ),
    "financial_data": PersonalDataCategory(
        name="Financial Data",
        examples=["portfolio_value", "pnl", "account_balance"],
        is_special_category=False
    ),
}
```

#### 0.2.2 ProcessorManagement (processor_management.py)

Article 28 compliant processor management:

```
Enum ProcessorStatus:
    PENDING_ASSESSMENT = "pending_assessment"
    APPROVED = "approved"
    CONDITIONAL = "conditional"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"

Dataclass ProcessorAssessment:
    assessment_id: str
    processor_name: str
    assessment_date: datetime

    # Article 28(1) sufficient guarantees
    technical_measures: List[str]
    organizational_measures: List[str]
    security_certifications: List[str]  # ISO 27001, SOC 2, etc.
    gdpr_compliance_evidence: Dict[str, Any]

    # Assessment results
    risk_level: str  # low, medium, high
    gaps_identified: List[str]
    remediation_required: List[str]
    recommendation: str
    assessor: str

Dataclass ProcessorRecord:
    processor_id: str
    legal_name: str
    trading_name: str
    registration_number: str
    country: str
    contact_dpo: str

    # Processing details
    processing_purposes: List[str]
    data_categories: List[str]
    data_subjects: List[str]

    # Contract details
    dpa_signed_date: datetime
    dpa_version: str
    contract_expiry: Optional[datetime]

    # Sub-processors
    sub_processors_authorized: bool  # General or specific
    sub_processors: List[str]

    # Status
    status: ProcessorStatus
    last_audit_date: Optional[datetime]
    next_audit_date: datetime

    # Article 28(3) mandatory provisions
    documented_instructions: bool
    confidentiality_obligations: bool
    security_measures_implemented: bool
    sub_processor_conditions_met: bool
    dsar_assistance_capability: bool
    compliance_assistance_capability: bool
    deletion_return_commitment: bool
    audit_rights_granted: bool

Class ProcessorManager:
    - assess_processor(processor_info: Dict) -> ProcessorAssessment
    - register_processor(record: ProcessorRecord) -> str
    - update_processor_status(processor_id: str, status: ProcessorStatus)
    - add_sub_processor(processor_id: str, sub_processor: ProcessorRecord)
    - notify_sub_processor_change(processor_id: str, change: Dict)
    - schedule_audit(processor_id: str, audit_date: datetime)
    - suspend_processor(processor_id: str, reason: str)
    - get_processor_list() -> List[ProcessorRecord]
    - generate_processor_report() -> ProcessorReport
```

#### 0.2.3 SubProcessorRegistry (sub_processor_registry.py)

Article 28(2)(4) sub-processor tracking:

```
Dataclass SubProcessorNotification:
    notification_id: str
    processor_id: str
    sub_processor_name: str
    notification_type: str  # "addition", "change", "removal"
    notified_at: datetime
    objection_deadline: datetime
    objection_received: bool
    objection_reason: Optional[str]
    resolution: Optional[str]

Class SubProcessorRegistry:
    - register_sub_processor(processor_id: str, sub_processor: ProcessorRecord)
    - process_notification(notification: SubProcessorNotification)
    - object_to_sub_processor(notification_id: str, reason: str)
    - get_sub_processor_chain(processor_id: str) -> List[ProcessorRecord]
    - verify_sub_processor_contracts(processor_id: str) -> VerificationResult
```

#### 0.2.4 JointControllerAgreement (joint_controller.py)

Article 26 joint controller management:

```
Dataclass JointControllerArrangement:
    arrangement_id: str
    controllers: List[ControllerInfo]

    # Article 26(1) essence of arrangement
    joint_purposes: List[str]
    joint_means: List[str]

    # Responsibility allocation
    responsibility_matrix: Dict[str, Dict[str, str]]
    dsar_contact_point: str
    breach_notification_lead: str
    dpia_responsibility: str

    # Data subject information (Article 26(2))
    transparency_arrangement: str
    contact_point_for_subjects: str

    # Agreement details
    agreement_date: datetime
    agreement_version: str
    review_date: datetime

Class JointControllerManager:
    - create_arrangement(arrangement: JointControllerArrangement) -> str
    - allocate_responsibility(arrangement_id: str, task: str, controller: str)
    - handle_dsar(arrangement_id: str, request: DSARRequest)
    - coordinate_breach_response(arrangement_id: str, breach: BreachIncident)
```

#### 0.2.5 DPAGenerator (dpa_generator.py)

Data Processing Agreement template generation:

```
Enum DPAClauseType:
    SUBJECT_MATTER = "subject_matter"
    DURATION = "duration"
    NATURE_PURPOSE = "nature_purpose"
    DATA_TYPES = "data_types"
    SUBJECT_CATEGORIES = "subject_categories"
    CONTROLLER_OBLIGATIONS = "controller_obligations"
    PROCESSOR_OBLIGATIONS = "processor_obligations"
    SECURITY_MEASURES = "security_measures"
    SUB_PROCESSING = "sub_processing"
    INTERNATIONAL_TRANSFERS = "international_transfers"
    AUDIT_RIGHTS = "audit_rights"
    TERMINATION = "termination"

Dataclass DPATemplate:
    template_id: str
    version: str
    clauses: Dict[DPAClauseType, str]
    scc_module: Optional[str]  # For international transfers
    jurisdiction: str
    language: str

Class DPAGenerator:
    - generate_dpa(processor: ProcessorRecord, template: DPATemplate) -> Document
    - validate_dpa_completeness(dpa: Document) -> ValidationResult
    - check_article_28_compliance(dpa: Document) -> ComplianceResult
    - append_scc(dpa: Document, scc_module: str) -> Document
    - get_template(template_type: str) -> DPATemplate
```

### 0.3 Platform-Specific Processor Mapping

Pre-configured processor relationships for the platform:

| Processor Type | Examples | Data Categories | Article 28 Requirements |
|---------------|----------|-----------------|------------------------|
| Exchange APIs | Binance, Coinbase, Kraken | Trading data, API keys | Full DPA, security audit |
| Cloud Provider | AWS, GCP, Azure | All data at rest | DPA + SCCs if non-EU |
| Market Data | Bloomberg, Reuters | Market data (non-personal) | Limited DPA |
| Analytics | Internal ML pipeline | Aggregated metrics | Internal processing policy |

### 0.4 Test Specifications

```
test_gdpr_phase0_core_processor.py:
├── test_definitions/
│   ├── test_role_classification
│   ├── test_personal_data_categories
│   ├── test_special_category_identification
│   └── test_platform_data_mapping
├── test_processor_management/
│   ├── test_processor_assessment
│   ├── test_processor_registration
│   ├── test_processor_status_update
│   ├── test_processor_suspension
│   ├── test_article_28_compliance_check
│   ├── test_audit_scheduling
│   └── test_processor_report_generation
├── test_sub_processor/
│   ├── test_sub_processor_registration
│   ├── test_notification_workflow
│   ├── test_objection_handling
│   ├── test_chain_verification
│   └── test_contract_cascade
├── test_joint_controller/
│   ├── test_arrangement_creation
│   ├── test_responsibility_allocation
│   ├── test_dsar_coordination
│   └── test_breach_coordination
├── test_dpa_generator/
│   ├── test_dpa_generation
│   ├── test_completeness_validation
│   ├── test_article_28_clause_inclusion
│   ├── test_scc_appendix
│   └── test_multi_language_support
└── test_integration/
    ├── test_exchange_processor_setup
    ├── test_cloud_provider_assessment
    └── test_full_processor_onboarding_workflow
```

**Expected test count**: ~70-90 tests

---

## Phase 1: Foundation & Legal Framework

**Estimated Complexity**: Medium
**Dependencies**: Phase 0
**Test Coverage Target**: 100%

### 1.1 Objectives

Establish the core GDPR framework including:
- Processing principles enforcement (Article 5)
- Lawful basis management (Article 6)
- Special categories handling (Article 9)
- Configuration and base infrastructure

### 1.2 Components to Implement

#### 1.2.1 GDPRConfig (config.py)

```
Configuration parameters:
- dpo_contact: DPO contact information
- supervisory_authority: Relevant SA details
- retention_periods: Per data category
- lawful_basis_defaults: Default legal bases
- consent_expiry_days: Consent validity period
- breach_notification_deadline_hours: 72
- dsar_response_deadline_days: 30
- data_subject_categories: List of categories
- special_category_data_enabled: bool
- international_transfer_mechanisms: List
```

#### 1.2.2 ProcessingPrinciples (processing_principles.py)

Article 5 principles to enforce:

| Principle | Implementation |
|-----------|----------------|
| **Lawfulness** | Verify legal basis before processing |
| **Fairness** | Transparency checks |
| **Transparency** | Logging and disclosure |
| **Purpose Limitation** | Purpose registry and validation |
| **Data Minimization** | Field-level necessity checks |
| **Accuracy** | Data quality validation hooks |
| **Storage Limitation** | Retention policy enforcement |
| **Integrity/Confidentiality** | Security controls verification |
| **Accountability** | Audit trail integration |

Key classes:
- `ProcessingPurpose`: Purpose definition with legal basis
- `PrincipleChecker`: Validates processing against principles
- `PrincipleViolation`: Violation record structure
- `ProcessingPrinciplesEnforcer`: Main enforcement engine

#### 1.2.3 LegalBasisManager (legal_basis.py)

Article 6 lawful bases:

```
Enum LawfulBasis:
    CONSENT = "consent"                    # Article 6(1)(a)
    CONTRACT = "contract"                  # Article 6(1)(b)
    LEGAL_OBLIGATION = "legal_obligation"  # Article 6(1)(c)
    VITAL_INTERESTS = "vital_interests"    # Article 6(1)(d)
    PUBLIC_TASK = "public_task"            # Article 6(1)(e)
    LEGITIMATE_INTEREST = "legitimate_interest"  # Article 6(1)(f)
```

Key classes:
- `LegalBasisRecord`: Documents legal basis for processing
- `LegitimateInterestAssessment`: LIA documentation
- `LegalBasisManager`: Manages and validates legal bases

#### 1.2.4 SpecialCategoriesHandler (special_categories.py)

Article 9 special category data handling:

```
Enum SpecialCategoryType:
    RACIAL_ETHNIC_ORIGIN = "racial_ethnic_origin"
    POLITICAL_OPINIONS = "political_opinions"
    RELIGIOUS_BELIEFS = "religious_beliefs"
    PHILOSOPHICAL_BELIEFS = "philosophical_beliefs"
    TRADE_UNION_MEMBERSHIP = "trade_union_membership"
    GENETIC_DATA = "genetic_data"
    BIOMETRIC_DATA = "biometric_data"  # For identification purposes
    HEALTH_DATA = "health_data"
    SEX_LIFE_ORIENTATION = "sex_life_orientation"

Enum Article9Exception:
    EXPLICIT_CONSENT = "explicit_consent"           # Article 9(2)(a)
    EMPLOYMENT_LAW = "employment_law"               # Article 9(2)(b)
    VITAL_INTERESTS = "vital_interests"             # Article 9(2)(c)
    LEGITIMATE_ACTIVITIES = "legitimate_activities" # Article 9(2)(d)
    MANIFESTLY_PUBLIC = "manifestly_public"         # Article 9(2)(e)
    LEGAL_CLAIMS = "legal_claims"                   # Article 9(2)(f)
    SUBSTANTIAL_PUBLIC_INTEREST = "public_interest" # Article 9(2)(g)
    HEALTHCARE = "healthcare"                       # Article 9(2)(h)
    PUBLIC_HEALTH = "public_health"                 # Article 9(2)(i)
    ARCHIVING_RESEARCH = "archiving_research"       # Article 9(2)(j)

Dataclass SpecialCategoryProcessing:
    processing_id: str
    data_category: SpecialCategoryType
    exception_relied_upon: Article9Exception
    exception_justification: str
    safeguards_implemented: List[str]
    explicit_consent_reference: Optional[str]
    member_state_law_reference: Optional[str]
    dpia_required: bool
    dpia_reference: Optional[str]
    approved_by: str
    approval_date: datetime

Class SpecialCategoriesHandler:
    - detect_special_category(data: Dict) -> List[SpecialCategoryType]
    - validate_processing_lawfulness(
          category: SpecialCategoryType,
          exception: Article9Exception,
          context: Dict
      ) -> ValidationResult
    - register_processing(processing: SpecialCategoryProcessing) -> str
    - get_explicit_consent(data_subject_id: str, category: SpecialCategoryType) -> ConsentRecord
    - check_safeguards(processing_id: str) -> SafeguardAssessment
    - block_processing_without_exception(category: SpecialCategoryType) -> None
```

Platform-specific special category considerations:

| Potential Source | Special Category | Applicability | Handling |
|-----------------|------------------|---------------|----------|
| Biometric 2FA | Biometric data | Only if fingerprint/face used | Explicit consent required |
| News Sentiment | Political opinions | If analyzing political news | Aggregate only, no individual linking |
| Health Sector Trading | Health data | Indirect through sector exposure | Market data only, no personal health |
| ESG Analysis | Various | If analyzing employee data | Use aggregated public data only |

**Important**: For algorithmic trading platforms, special category data should be **avoided by design**. If unavoidable, explicit consent and DPIA are mandatory.

#### 1.2.4 AccountabilityFramework (accountability.py) - NEW

**Article 24 - Responsibility of the Controller**

Per [GDPR Article 24](https://gdpr-info.eu/art-24-gdpr/), the controller must implement appropriate technical and organizational measures to ensure and demonstrate compliance.

```
Dataclass ComplianceEvidence:
    evidence_id: str
    article_reference: str
    measure_type: str  # "technical", "organizational", "policy"
    measure_description: str
    implementation_date: datetime
    last_review_date: datetime
    next_review_date: datetime
    responsible_party: str
    evidence_documentation: List[str]  # Links to policies, configs, logs
    effectiveness_assessment: str

Dataclass PolicyRecord:
    policy_id: str
    policy_name: str
    policy_type: str  # "data_protection", "security", "retention", "breach"
    version: str
    effective_date: datetime
    review_schedule_months: int
    owner: str
    approved_by: str
    document_url: str

Class AccountabilityFramework:
    """
    Article 24 compliance - demonstrate processing compliance.

    Implements the 'accountability principle' from Article 5(2):
    'The controller shall be responsible for, and be able to
    demonstrate compliance with, paragraph 1 ('accountability').'
    """

    # Evidence management
    - register_compliance_measure(measure: ComplianceEvidence) -> str
    - update_measure_effectiveness(evidence_id: str, assessment: str)
    - schedule_measure_review(evidence_id: str, review_date: datetime)
    - get_evidence_for_article(article: str) -> List[ComplianceEvidence]

    # Policy management
    - register_policy(policy: PolicyRecord) -> str
    - review_policy(policy_id: str, reviewer: str, outcome: str)
    - get_policies_due_for_review() -> List[PolicyRecord]

    # Demonstration of compliance
    - generate_accountability_report() -> AccountabilityReport
    - prepare_for_audit(scope: List[str]) -> AuditPackage
    - demonstrate_article_5_compliance() -> ComplianceStatement

    # Technical measures verification
    - verify_technical_measures() -> TechnicalMeasureAudit
    - verify_organizational_measures() -> OrgMeasureAudit
```

**Platform-Specific Accountability Measures:**

| Article | Measure Type | Evidence | Review Frequency |
|---------|--------------|----------|------------------|
| Art. 5 | Policy | Processing principles policy | Annual |
| Art. 6 | Technical | Legal basis documentation in code | Per feature |
| Art. 7 | Technical | Consent management system logs | Continuous |
| Art. 25 | Technical | Privacy by design checklist | Per release |
| Art. 30 | Documentation | ROPA exports | Quarterly |
| Art. 32 | Technical + Policy | Security controls + policies | Annual |

#### 1.2.5 RestrictionsFramework (restrictions.py) - NEW

**Article 23 - Restrictions**

Per [GDPR Article 23](https://gdpr-info.eu/art-23-gdpr/), EU or Member State law may restrict data subject rights and controller obligations for important objectives including financial interests and regulatory compliance.

```
Enum RestrictionType:
    NATIONAL_SECURITY = "national_security"          # Art. 23(1)(a)
    DEFENCE = "defence"                              # Art. 23(1)(b)
    PUBLIC_SECURITY = "public_security"              # Art. 23(1)(c)
    CRIMINAL_PREVENTION = "criminal_prevention"      # Art. 23(1)(d)
    PUBLIC_INTEREST = "public_interest"              # Art. 23(1)(e)
    JUDICIAL_PROTECTION = "judicial_protection"      # Art. 23(1)(f)
    REGULATORY_ENFORCEMENT = "regulatory_enforcement" # Art. 23(1)(h)
    RIGHTS_PROTECTION = "rights_protection"          # Art. 23(1)(i)
    CIVIL_CLAIMS = "civil_claims"                    # Art. 23(1)(j)

Dataclass LegalRestriction:
    restriction_id: str
    legal_basis: str  # e.g., "MiFID II Article 16", "MAR Article 11"
    restriction_type: RestrictionType
    articles_restricted: List[str]  # e.g., ["Art. 15", "Art. 17"]
    scope_of_restriction: str
    duration: Optional[str]  # e.g., "7 years retention period"
    documentation: str

Dataclass RestrictionApplication:
    application_id: str
    restriction_id: str
    data_subject_id: str
    request_type: str  # "erasure", "access", etc.
    request_date: datetime
    restriction_applied: bool
    justification: str
    notified_data_subject: bool
    notification_content: str
    review_date: Optional[datetime]  # When restriction expires

Class RestrictionsFramework:
    """
    Article 23 - Manage legal restrictions on GDPR rights.

    Critical for financial services where MiFID II, MAR, and other
    regulations require retention that restricts GDPR erasure rights.
    """

    # Restriction registry
    - register_legal_restriction(restriction: LegalRestriction) -> str
    - get_applicable_restrictions(request_type: str, context: Dict) -> List[LegalRestriction]

    # Application
    - apply_restriction(
          data_subject_id: str,
          request_type: str,
          restriction: LegalRestriction
      ) -> RestrictionApplication

    # Notification (Art. 23(2)(h) - inform data subject of restriction)
    - notify_restriction(application_id: str) -> NotificationResult

    # Review and expiry
    - schedule_restriction_expiry(application_id: str, expiry: datetime)
    - process_expired_restrictions() -> List[str]  # Returns released request IDs
```

**Financial Services Restrictions (Pre-Configured):**

| Regulation | Restricts | Duration | GDPR Articles Affected |
|------------|-----------|----------|----------------------|
| MiFID II Art. 16(7) | Erasure of transaction records | 5-7 years | Art. 17 (erasure) |
| MiFIR Art. 25 | Erasure of trading data | 5 years | Art. 17 (erasure) |
| MAR Art. 11 | Access during investigation | Variable | Art. 15 (access) |
| MAD Art. 16 | Deletion of suspicious activity reports | 5 years | Art. 17 (erasure) |
| AMLD Art. 40 | AML records | 5 years | Art. 17 (erasure) |

**Integration with Erasure Manager:**

```python
# In erasure_manager.py
class ErasureManager:
    def process_erasure_request(self, request: ErasureRequest) -> ErasureDecision:
        # Check for applicable restrictions
        restrictions = self.restrictions_framework.get_applicable_restrictions(
            request_type="erasure",
            context={"data_categories": request.data_categories}
        )

        if restrictions:
            # Apply restriction, schedule future erasure
            for restriction in restrictions:
                self.restrictions_framework.apply_restriction(
                    data_subject_id=request.data_subject_id,
                    request_type="erasure",
                    restriction=restriction
                )

            # Calculate when ALL restrictions expire
            latest_expiry = max(r.duration for r in restrictions)

            return ErasureDecision(
                action="RESTRICTED",
                reason=f"Legal obligation: {restrictions[0].legal_basis}",
                article_23_applied=True,
                scheduled_erasure_date=latest_expiry,
                pseudonymize_now=True,  # GDPR minimization still applies!
                data_subject_notified=True
            )
```

### 1.3 Implementation Requirements

1. **Integration with existing audit system**
   - Reuse `services/compliance/audit_trail_writer.py`
   - Extend audit models for GDPR-specific events

2. **Configuration hierarchy**
   - YAML configuration file: `configs/gdpr/gdpr_config.yaml`
   - Environment variable overrides
   - Pydantic validation

3. **Logging requirements**
   - All principle checks logged
   - Legal basis decisions recorded
   - Integration with secure logging

### 1.4 Test Specifications

```
test_gdpr_phase1_foundation.py:
├── test_config/
│   ├── test_config_loading
│   ├── test_config_validation
│   ├── test_config_defaults
│   └── test_config_environment_override
├── test_principles/
│   ├── test_lawfulness_check
│   ├── test_fairness_validation
│   ├── test_purpose_limitation
│   ├── test_data_minimization_check
│   ├── test_accuracy_validation
│   ├── test_storage_limitation
│   ├── test_integrity_confidentiality
│   ├── test_accountability_logging
│   └── test_principle_violation_recording
├── test_legal_basis/
│   ├── test_consent_basis
│   ├── test_contract_basis
│   ├── test_legal_obligation_basis
│   ├── test_legitimate_interest_basis
│   ├── test_lia_documentation
│   ├── test_basis_validation
│   └── test_basis_change_tracking
├── test_special_categories/
│   ├── test_special_category_detection
│   ├── test_article9_exception_validation
│   ├── test_explicit_consent_requirement
│   ├── test_processing_registration
│   ├── test_safeguard_verification
│   ├── test_block_without_exception
│   ├── test_biometric_data_handling
│   ├── test_health_data_avoidance
│   └── test_political_data_aggregation
└── test_integration/
    ├── test_audit_trail_integration
    ├── test_secure_logging_integration
    └── test_special_category_audit_logging
```

**Expected test count**: ~90-110 tests

---

## Phase 2a: Consent & Transparency

**Estimated Complexity**: Medium
**Dependencies**: Phase 1
**Test Coverage Target**: 100%

### 2a.1 Objectives

Implement consent management and transparency requirements:
- Consent management (Article 7)
- Privacy notices and information provision (Articles 12-14)
- Layered transparency approach
- Consent evidence and audit trail

### 2a.2 Components to Implement

#### 2a.2.1 TransparencyNotices (transparency_notices.py)

Articles 12-14 information provision:

```
Enum NoticeType:
    COLLECTION_DIRECT = "direct_collection"      # Article 13
    COLLECTION_INDIRECT = "indirect_collection"  # Article 14
    PROCESSING_CHANGE = "processing_change"
    CONSENT_REQUEST = "consent_request"
    BREACH_NOTIFICATION = "breach_notification"  # Article 34

Enum NoticeLayer:
    SUMMARY = "summary"           # Key points (icons, short text)
    FULL = "full"                 # Complete privacy notice
    JUST_IN_TIME = "just_in_time" # Context-specific pop-ups
    DETAILED = "detailed"         # Legal document

Dataclass PrivacyNotice:
    notice_id: str
    notice_type: NoticeType
    version: str

    # Article 13(1) / Article 14(1) - Mandatory information
    controller_identity: str
    controller_contact: str
    dpo_contact: str
    processing_purposes: List[str]
    legal_basis: List[str]
    legitimate_interests: Optional[str]

    # Article 13(1)(e-f) / Article 14(1)(e-f)
    recipients: List[str]
    third_country_transfers: List[ThirdCountryInfo]

    # Article 13(2) / Article 14(2) - Additional information
    retention_periods: Dict[str, str]
    data_subject_rights: List[str]
    right_to_withdraw_consent: bool
    right_to_lodge_complaint: str
    statutory_contractual_requirement: Optional[str]
    automated_decision_making: Optional[AutomatedDecisionInfo]

    # Article 14 specific
    data_source: Optional[str]                    # Article 14(2)(f)
    data_categories: Optional[List[str]]          # Article 14(1)(d)

    # Metadata
    language: str
    effective_date: datetime
    last_updated: datetime
    layers: Dict[NoticeLayer, str]

Dataclass AutomatedDecisionInfo:
    """Article 13(2)(f) / Article 14(2)(g) - Automated decision information."""
    exists: bool
    meaningful_info_about_logic: str
    significance_and_consequences: str
    human_oversight_available: bool
    how_to_request_intervention: str

Class TransparencyManager:
    - create_notice(notice: PrivacyNotice) -> str
    - update_notice(notice_id: str, updates: Dict) -> PrivacyNotice
    - get_notice_for_context(context: str, layer: NoticeLayer) -> str
    - version_notice(notice_id: str) -> PrivacyNotice
    - track_notice_provision(data_subject_id: str, notice_id: str)
    - generate_layered_notice(notice_id: str) -> Dict[NoticeLayer, str]
    - check_notice_completeness(notice: PrivacyNotice) -> ComplianceResult
    - get_article_14_deadline(data_source: str) -> datetime  # "reasonable period" max 1 month
```

#### 2a.2.2 InformationProvision (information_provision.py)

Timing and delivery of privacy information:

```
Enum ProvisionTiming:
    AT_COLLECTION = "at_collection"         # Article 13 - at time of collection
    REASONABLE_PERIOD = "reasonable_period" # Article 14 - within reasonable period
    FIRST_COMMUNICATION = "first_communication"  # Article 14 - on first contact
    BEFORE_DISCLOSURE = "before_disclosure" # Article 14 - before disclosure to third party

Dataclass InformationProvisionRecord:
    record_id: str
    data_subject_id: str
    notice_id: str
    provision_timing: ProvisionTiming
    provided_at: datetime
    delivery_method: str  # "web", "email", "api", "in_app"
    acknowledged: bool
    acknowledgment_timestamp: Optional[datetime]
    evidence: Dict[str, Any]

Class InformationProvisionManager:
    - provide_notice(data_subject_id: str, notice_id: str, method: str) -> InformationProvisionRecord
    - verify_provision(data_subject_id: str, processing_activity: str) -> bool
    - get_provision_history(data_subject_id: str) -> List[InformationProvisionRecord]
    - check_article_14_compliance(data_subject_id: str, data_source: str) -> ComplianceResult
```

#### 2a.2.3 ConsentManager (consent_manager.py)

Article 7 requirements:

```
Dataclass ConsentRecord:
    consent_id: str
    data_subject_id: str
    purpose: str
    legal_basis: str = "consent"
    granted_at: datetime
    expires_at: Optional[datetime]
    withdrawn_at: Optional[datetime]
    consent_text: str
    consent_version: str
    collection_method: str  # "web_form", "api", "verbal"
    evidence: Dict[str, Any]  # IP, user agent, etc.
    granular_choices: Dict[str, bool]
    is_active: bool
```

Key requirements:
- Double opt-in support
- Granular consent per purpose
- Easy withdrawal (same effort as granting)
- Consent evidence storage
- Version tracking for consent text changes
- **No pre-ticked boxes** (Article 7(2) - clear affirmative act)
- **No bundled consent** (Article 7(2) - distinguishable from other matters)
- **Freely given** (Article 7(4) - no conditionality for service)

### 2a.3 Test Specifications

```
test_gdpr_phase2a_consent_transparency.py:
├── test_transparency/
│   ├── test_privacy_notice_creation
│   ├── test_notice_versioning
│   ├── test_layered_notice_generation
│   ├── test_article_13_completeness
│   ├── test_article_14_completeness
│   ├── test_automated_decision_info
│   ├── test_notice_provision_timing
│   └── test_provision_evidence_storage
├── test_consent/
│   ├── test_consent_creation
│   ├── test_consent_granularity
│   ├── test_consent_withdrawal
│   ├── test_withdrawal_ease_equals_granting
│   ├── test_consent_evidence_storage
│   ├── test_consent_version_tracking
│   ├── test_consent_expiry
│   ├── test_double_opt_in
│   ├── test_no_pre_ticked_boxes
│   ├── test_no_bundled_consent
│   ├── test_freely_given_check
│   └── test_consent_audit_trail
└── test_integration/
    ├── test_notice_consent_linkage
    ├── test_transparency_before_processing
    └── test_consent_record_for_legal_basis
```

**Expected test count**: ~60-70 tests

---

## Phase 2b: Data Subject Rights

**Estimated Complexity**: High
**Dependencies**: Phase 2a
**Test Coverage Target**: 100%

### 2b.1 Objectives

Implement comprehensive data subject rights management:
- Right of access / DSAR (Article 15)
- Right to rectification (Article 16)
- Right to erasure (Article 17)
- Right to restriction (Article 18)
- Right to data portability (Article 20)
- Right to object (Article 21)
- **Automated decision-making rights (Article 22)** - CRITICAL for algorithmic trading

### 2b.2 Components to Implement

#### 2b.2.1 DSARHandler (dsar_handler.py)

Article 15 DSAR workflow:

```
Enum DSARStatus:
    RECEIVED = "received"
    IDENTITY_VERIFICATION = "identity_verification"
    IN_PROGRESS = "in_progress"
    DATA_COLLECTION = "data_collection"
    REVIEW = "review"
    COMPLETED = "completed"
    EXTENDED = "extended"
    REFUSED = "refused"

Dataclass DSARRequest:
    request_id: str
    data_subject_id: str
    requested_at: datetime
    deadline: datetime  # 30 days default
    extended_deadline: Optional[datetime]  # +60 days max
    status: DSARStatus
    identity_verified: bool
    requested_data_types: List[str]
    collected_data: Dict[str, Any]
    response_format: str  # "json", "csv", "pdf"
    delivery_method: str  # "download", "email", "post"
```

Key features:
- Identity verification workflow
- Deadline tracking (30 days + 60 extension)
- Data collection across systems
- Redaction of third-party data
- Response generation in multiple formats

#### 2.2.3 ErasureManager (erasure_manager.py)

Article 17 right to erasure:

```
Enum ErasureStatus:
    REQUESTED = "requested"
    VALIDATING = "validating"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REFUSED = "refused"
    PARTIALLY_COMPLETED = "partially_completed"

Enum ErasureException:
    LEGAL_OBLIGATION = "legal_obligation"
    PUBLIC_INTEREST_ARCHIVING = "public_interest"
    SCIENTIFIC_RESEARCH = "scientific_research"
    LEGAL_CLAIMS = "legal_claims"
    FREEDOM_OF_EXPRESSION = "freedom_expression"
```

Key features:
- Exception handling (e.g., MiFID II 5-7 year retention)
- Cascading deletion across systems
- Third-party notification (Article 17(2))
- Backup system handling
- Deletion verification
- Audit trail preservation (anonymized)

#### 2.2.4 PortabilityManager (portability_manager.py)

Article 20 data portability:

```
Supported formats:
- JSON (structured, machine-readable)
- CSV (tabular data)
- XML (optional)

Dataclass PortabilityRequest:
    request_id: str
    data_subject_id: str
    requested_at: datetime
    format: str
    destination: Optional[str]  # Direct transfer to another controller
    data_categories: List[str]
    status: str
```

Key features:
- Machine-readable format generation
- Direct controller-to-controller transfer support
- Selective data category export
- Metadata inclusion
- **Only applies to**: automated processing AND (consent OR contract basis)

#### 2b.2.5 AutomatedDecisionManager (automated_decisions.py)

**Article 22 - CRITICAL FOR ALGORITHMIC TRADING PLATFORMS**

> ⚠️ **IMPORTANT LEGAL CLARIFICATION**: Article 22 does NOT apply to ALL automated trading decisions.
> Per [EDPB Guidelines on Automated Decision-Making](https://www.edpb.europa.eu/our-work-tools/our-documents/guidelines/automated-decision-making-and-profiling_en),
> Article 22(1) only applies when a decision is:
> 1. **Solely based on automated processing** (no meaningful human involvement), AND
> 2. **Produces legal effects** concerning the data subject OR **similarly significantly affects** them

**Article 22 Applicability for Trading Platform:**

```
Enum DecisionType:
    FULLY_AUTOMATED = "fully_automated"       # Article 22(1) applies
    HUMAN_IN_LOOP = "human_in_loop"           # Article 22(1) may not apply
    PROFILING_ONLY = "profiling_only"         # Not decision, just analysis
    RECOMMENDATION = "recommendation"          # Human makes final decision

Enum Article22Basis:
    CONTRACT_NECESSARY = "contract_necessary"       # Article 22(2)(a)
    UNION_MEMBER_STATE_LAW = "law_authorized"       # Article 22(2)(b)
    EXPLICIT_CONSENT = "explicit_consent"           # Article 22(2)(c)

Enum SignificantEffect:
    LEGAL = "legal"                    # Affects legal rights
    FINANCIAL = "financial"            # Affects financial position
    ACCESS_TO_SERVICES = "access"      # Affects service access
    EMPLOYMENT = "employment"          # Affects employment
    CREDIT = "credit"                  # Affects creditworthiness

Dataclass AutomatedDecision:
    decision_id: str
    data_subject_id: str
    decision_type: DecisionType
    decision_timestamp: datetime

    # Article 22(1) assessment
    is_solely_automated: bool
    produces_legal_effects: bool
    significantly_affects: List[SignificantEffect]
    article_22_applies: bool

    # Legal basis for automated decision (if Article 22 applies)
    legal_basis: Optional[Article22Basis]
    explicit_consent_reference: Optional[str]
    law_authorization_reference: Optional[str]

    # Decision details
    input_data_categories: List[str]
    decision_logic_summary: str
    decision_outcome: str
    confidence_score: Optional[float]
    factors_considered: List[Dict[str, Any]]

    # Article 22(3) safeguards
    human_intervention_available: bool
    intervention_contact: str
    right_to_express_view: bool
    right_to_contest: bool

    # Explainability (Article 13(2)(f), 14(2)(g), 15(1)(h))
    meaningful_information_about_logic: str
    significance_explanation: str
    envisaged_consequences: str

Dataclass HumanInterventionRequest:
    request_id: str
    decision_id: str
    data_subject_id: str
    requested_at: datetime
    reason: str
    data_subject_view: str

    # Processing
    assigned_to: str
    reviewed_at: Optional[datetime]
    review_outcome: str
    original_decision_upheld: bool
    new_decision: Optional[str]
    explanation_to_subject: str

    # Status
    status: str  # "pending", "under_review", "completed"
    response_deadline: datetime  # 1 month per Article 12(3)

Dataclass DecisionContestation:
    contestation_id: str
    decision_id: str
    data_subject_id: str
    submitted_at: datetime
    grounds_for_contestation: str
    supporting_evidence: List[str]

    # Resolution
    reviewed_by: str
    review_findings: str
    decision_changed: bool
    new_decision: Optional[str]
    compensation_offered: Optional[float]
    data_subject_satisfied: Optional[bool]

Class AutomatedDecisionManager:
    """
    Article 22 compliance for algorithmic trading platform.

    IMPORTANT: Every automated trading decision that affects user's
    financial position must be tracked and subject to Article 22 rights.
    """

    - assess_decision_type(processing_activity: str, context: Dict) -> DecisionType
    - check_article_22_applicability(decision: AutomatedDecision) -> bool
    - register_automated_decision(decision: AutomatedDecision) -> str
    - get_decision_explanation(decision_id: str) -> DecisionExplanation

    # Article 22(3) Safeguards
    - request_human_intervention(decision_id: str, request: HumanInterventionRequest) -> str
    - assign_human_reviewer(request_id: str, reviewer: str)
    - complete_human_review(request_id: str, outcome: ReviewOutcome)

    # Right to express view and contest
    - submit_contestation(contestation: DecisionContestation) -> str
    - process_contestation(contestation_id: str) -> ContestationResult
    - express_view(decision_id: str, view: str) -> ViewRecord

    # Explainability (per EDPB Guidelines)
    - generate_meaningful_explanation(decision_id: str) -> str
    - explain_logic_involved(decision_id: str) -> LogicExplanation
    - explain_significance_and_consequences(decision_id: str) -> str

    # Audit and reporting
    - get_automated_decisions(data_subject_id: str) -> List[AutomatedDecision]
    - get_intervention_statistics() -> InterventionStats
    - generate_article_22_compliance_report() -> Report
```

**Platform-Specific Article 22 Classification:**

```
Enum Article22Applicability:
    NOT_APPLICABLE = "not_applicable"     # User-initiated, no significant effect
    REQUIRES_ANALYSIS = "requires_analysis"  # Case-by-case assessment needed
    APPLICABLE = "applicable"             # Full Article 22 safeguards required
```

| Processing Activity | Decision Type | Article 22 Applies | Rationale | Safeguards Required |
|--------------------|---------------|-------------------|-----------|---------------------|
| **User-Initiated Order Execution** | Fully Automated | **NO** | User made the decision; platform only executes | Transparency only |
| **User-Configured Stop-Loss** | Fully Automated | **NO** | User set parameters; platform follows instructions | Transparency only |
| **Market Data Display/Analysis** | N/A | **NO** | No decision about user | None |
| **Algorithmic Strategy Execution** (user-selected) | Fully Automated | **REQUIRES ANALYSIS** | Depends on user control level | Case-by-case |
| **Risk Score → Access Denial** | Profiling + Decision | **YES** | Significantly affects access to services | Full Art. 22(3) safeguards |
| **Risk Score → Limit Reduction** | Profiling + Decision | **YES** | Significantly affects financial capacity | Full Art. 22(3) safeguards |
| **Forced Liquidation (Margin Call)** | Fully Automated | **YES** | Significant financial effect without user consent | Full Art. 22(3) safeguards |
| **Account Approval/Denial** | Fully Automated | **YES** | Legal effect (contract denial) | Human review mandatory |
| **Auto-Termination of Strategy** | Fully Automated | **YES** if significant loss | May significantly affect financially | Case-by-case |
| **Strategy Recommendations** | Recommendation | **NO** | Human makes final decision | Transparency only |
| **Position Sizing** (user-configured params) | Fully Automated | **NO** | User defined parameters | Transparency only |
| **Position Sizing** (platform-determined) | Fully Automated | **YES** | Platform decides without user input | Full Art. 22(3) safeguards |

> **Key Principle**: If the user initiated and parameterized the action, Article 22 generally does NOT apply.
> If the platform autonomously makes a decision that significantly affects the user without their specific instruction, Article 22 DOES apply.

**CRITICAL Implementation Notes:**

1. **Real-time Intervention**: For trading decisions, "human intervention" must be practically available (e.g., ability to override/cancel within execution window)

2. **Meaningful Information**: Per EDPB guidelines, must explain:
   - Categories of data used
   - Why data is relevant
   - How data influences outcome
   - Main factors in decision

3. **Right to Contest**: Must have process to:
   - Review decision manually
   - Potentially reverse/compensate
   - Document reasoning

4. **Integration with EU AI Act**: High-risk AI systems (which algorithmic trading may be) have additional requirements per AI Act Article 14 (human oversight)

### 2b.3 Implementation Requirements

1. **Identity verification**
   - Multi-factor verification for sensitive requests
   - Proportionate to risk
   - Documentation of verification

2. **Timeline management**
   - 30-day response deadline
   - Automatic extension handling (+60 days)
   - Deadline notifications

3. **Cross-system data collection**
   - Integration with all data stores
   - Audit log collection
   - Third-party data handling

4. **Conflict resolution with MiFID II**
   - Document retention requirements (5-7 years) take precedence
   - Partial erasure with anonymization
   - Clear documentation of exceptions

### 2b.4 Test Specifications

```
test_gdpr_phase2b_data_subject_rights.py:
├── test_dsar/
│   ├── test_dsar_creation
│   ├── test_identity_verification
│   ├── test_deadline_calculation
│   ├── test_deadline_extension
│   ├── test_data_collection
│   ├── test_third_party_redaction
│   ├── test_response_generation_json
│   ├── test_response_generation_csv
│   ├── test_secure_delivery
│   └── test_dsar_audit_trail
├── test_erasure/
│   ├── test_erasure_request
│   ├── test_exception_legal_obligation
│   ├── test_exception_mifid_retention
│   ├── test_cascading_deletion
│   ├── test_backup_handling
│   ├── test_third_party_notification
│   ├── test_deletion_verification
│   ├── test_anonymization_fallback
│   └── test_erasure_audit_trail
├── test_portability/
│   ├── test_export_json
│   ├── test_export_csv
│   ├── test_selective_export
│   ├── test_direct_transfer
│   └── test_metadata_inclusion
├── test_rectification/
│   ├── test_rectification_request
│   ├── test_data_update
│   └── test_third_party_notification
├── test_restriction/
│   ├── test_restriction_request
│   ├── test_processing_pause
│   └── test_restriction_lifting
├── test_objection/
│   ├── test_objection_to_processing
│   ├── test_direct_marketing_objection
│   └── test_profiling_objection
├── test_article_22_automated_decisions/
│   ├── test_decision_type_assessment
│   ├── test_article_22_applicability_check
│   ├── test_decision_registration
│   ├── test_human_intervention_request
│   ├── test_intervention_assignment
│   ├── test_intervention_review_completion
│   ├── test_contestation_submission
│   ├── test_contestation_processing
│   ├── test_express_view_right
│   ├── test_meaningful_explanation_generation
│   ├── test_logic_explanation_quality
│   ├── test_significance_consequences
│   ├── test_trading_decision_tracking
│   ├── test_real_time_intervention_trading
│   ├── test_stop_loss_trigger_intervention
│   ├── test_position_sizing_explanation
│   ├── test_article_22_with_special_categories
│   └── test_compliance_report_generation
├── test_edge_cases/
│   ├── test_erasure_with_active_legal_claim
│   ├── test_erasure_during_litigation_hold
│   ├── test_dsar_excessive_requests_handling
│   ├── test_dsar_manifestly_unfounded_rejection
│   ├── test_consent_withdrawal_mid_batch_processing
│   ├── test_erasure_backup_rotation_timing
│   ├── test_erasure_processor_notification_chain
│   ├── test_portability_large_dataset_streaming
│   ├── test_intervention_timeout_escalation
│   ├── test_contested_decision_compensation
│   ├── test_concurrent_dsar_and_erasure
│   ├── test_cross_border_dsar_handling
│   └── test_deceased_person_data_handling
└── test_integration/
    ├── test_rights_dashboard_complete
    ├── test_cross_rights_workflow
    ├── test_audit_trail_completeness
    ├── test_article_22_with_ai_act_alignment
    ├── test_dsar_with_mifid_retention_conflict
    └── test_erasure_with_dora_incident_retention
```

**Expected test count**: ~100-120 tests

---

## Phase 3: Records of Processing Activities (ROPA)

**Estimated Complexity**: Medium
**Dependencies**: Phase 0, Phase 1, Phase 2a, Phase 2b
**Test Coverage Target**: 100%

### 3.1 Objectives

Implement Article 30 compliant Records of Processing Activities:
- Complete processing inventory
- Data mapping across systems
- Processing purpose documentation
- Automated ROPA generation

### 3.2 Components to Implement

#### 3.2.1 ROPA (ropa.py)

Article 30(1) Controller ROPA fields:

```
Dataclass ProcessingActivityRecord:
    activity_id: str
    activity_name: str
    description: str

    # Controller information
    controller_name: str
    controller_contact: str
    joint_controllers: List[str]
    dpo_contact: str

    # Processing details
    purposes: List[str]
    legal_basis: str
    legitimate_interest_description: Optional[str]

    # Data categories
    data_subject_categories: List[str]
    personal_data_categories: List[str]
    special_category_data: List[str]

    # Recipients
    recipient_categories: List[str]
    third_country_transfers: List[ThirdCountryTransfer]

    # Retention
    retention_period: str
    retention_criteria: str

    # Security
    security_measures: List[str]

    # Metadata
    created_at: datetime
    updated_at: datetime
    review_date: datetime
    status: str  # "active", "inactive", "under_review"
```

Article 30(2) Processor ROPA fields:

```
Dataclass ProcessorActivityRecord:
    processor_name: str
    processor_contact: str
    controller_name: str
    controller_contact: str
    dpo_contact: str
    processing_categories: List[str]
    third_country_transfers: List[ThirdCountryTransfer]
    security_measures: List[str]
```

#### 3.2.2 ProcessingRegistry (processing_registry.py)

Central registry for all processing activities:

```
Class ProcessingRegistry:
    - register_activity(activity: ProcessingActivityRecord)
    - update_activity(activity_id: str, updates: Dict)
    - deactivate_activity(activity_id: str)
    - get_activity(activity_id: str) -> ProcessingActivityRecord
    - list_activities(filters: Dict) -> List[ProcessingActivityRecord]
    - export_ropa(format: str) -> bytes  # PDF, Excel, JSON
    - validate_ropa() -> List[ValidationError]
```

#### 3.2.3 DataMapping (data_mapping.py)

Personal data flow mapping:

```
Dataclass DataFlow:
    flow_id: str
    source_system: str
    destination_system: str
    data_categories: List[str]
    purpose: str
    legal_basis: str
    transfer_mechanism: str  # "internal", "processor", "third_country"
    encryption_in_transit: bool
    encryption_at_rest: bool

Class DataMapper:
    - map_system(system_id: str) -> SystemDataMap
    - trace_data_flow(data_category: str) -> List[DataFlow]
    - identify_personal_data(schema: Dict) -> List[PersonalDataField]
    - generate_data_flow_diagram() -> str  # SVG/PNG
```

### 3.3 Platform-Specific Processing Activities

Pre-configured processing activities for the platform:

| Activity | Purpose | Legal Basis | Data Categories | Retention |
|----------|---------|-------------|-----------------|-----------|
| Trading Execution | Contract performance | Contract | Orders, trades | 7 years (MiFID) |
| Risk Management | Legitimate interest | Legitimate Interest | Positions, P&L | 5 years |
| Compliance Monitoring | Legal obligation | Legal Obligation | Audit logs | 7 years (MiFID) |
| Model Training | Legitimate interest | Legitimate Interest | Market data, features | Research period |
| User Authentication | Contract performance | Contract | Credentials, sessions | Account lifetime |
| API Key Management | Contract performance | Contract | API keys | Until rotation |

### 3.4 Implementation Requirements

1. **Auto-discovery**
   - Scan codebase for data processing
   - Integrate with existing audit system
   - Track data flows automatically

2. **Regulatory alignment**
   - Cross-reference with MiFID II requirements
   - Align with DORA incident classification
   - Link to EU AI Act data governance

3. **Export formats**
   - Excel (supervisory authority format)
   - PDF (human-readable)
   - JSON (machine-readable)
   - Integration with DORA Register of Information

### 3.5 Test Specifications

```
test_gdpr_phase3_ropa.py:
├── test_ropa_record/
│   ├── test_controller_ropa_creation
│   ├── test_processor_ropa_creation
│   ├── test_required_fields_validation
│   ├── test_optional_fields
│   ├── test_ropa_update
│   ├── test_ropa_versioning
│   └── test_ropa_review_scheduling
├── test_registry/
│   ├── test_activity_registration
│   ├── test_activity_deactivation
│   ├── test_activity_search
│   ├── test_activity_filtering
│   ├── test_export_excel
│   ├── test_export_pdf
│   ├── test_export_json
│   └── test_ropa_validation
├── test_data_mapping/
│   ├── test_system_mapping
│   ├── test_data_flow_tracing
│   ├── test_personal_data_identification
│   ├── test_flow_diagram_generation
│   └── test_cross_system_mapping
├── test_platform_activities/
│   ├── test_trading_execution_activity
│   ├── test_compliance_monitoring_activity
│   ├── test_model_training_activity
│   └── test_authentication_activity
└── test_regulatory_alignment/
    ├── test_mifid_alignment
    ├── test_dora_roi_integration
    └── test_ai_act_data_governance_link
```

**Expected test count**: ~80-100 tests

---

## Phase 4: Privacy Engineering

**Estimated Complexity**: High
**Dependencies**: Phase 1, Phase 2, Phase 3
**Test Coverage Target**: 100%

### 4.1 Objectives

Implement Privacy by Design and Default (Article 25):
- Data minimization enforcement
- Pseudonymization capabilities
- Retention policy automation
- Privacy-preserving defaults

### 4.2 Components to Implement

#### 4.2.1 PrivacyByDesign (privacy_by_design.py)

Article 25 implementation:

```
Enum PrivacyControl:
    DATA_MINIMIZATION = "data_minimization"
    PURPOSE_LIMITATION = "purpose_limitation"
    STORAGE_LIMITATION = "storage_limitation"
    PSEUDONYMIZATION = "pseudonymization"
    ENCRYPTION = "encryption"
    ACCESS_CONTROL = "access_control"
    AUDIT_LOGGING = "audit_logging"
    DATA_SEGREGATION = "data_segregation"

Dataclass PrivacyAssessment:
    assessment_id: str
    system_or_feature: str
    controls_implemented: List[PrivacyControl]
    controls_gaps: List[PrivacyControl]
    recommendations: List[str]
    risk_level: str
    assessment_date: datetime
    assessor: str

Class PrivacyByDesignChecker:
    - assess_system(system_id: str) -> PrivacyAssessment
    - check_data_minimization(schema: Dict) -> List[Violation]
    - check_purpose_limitation(processing: Dict) -> List[Violation]
    - check_storage_limitation(data_store: str) -> List[Violation]
    - verify_defaults_privacy_preserving(config: Dict) -> bool
```

#### 4.2.2 DataMinimization (data_minimization.py)

```
Dataclass DataField:
    field_name: str
    data_type: str
    is_personal_data: bool
    is_special_category: bool
    necessity: str  # "required", "optional", "excessive"
    purpose: str
    retention_period: str

Class DataMinimizationEnforcer:
    - analyze_schema(schema: Dict) -> List[DataField]
    - flag_excessive_collection(fields: List[DataField]) -> List[str]
    - suggest_minimization(fields: List[DataField]) -> List[Recommendation]
    - enforce_collection_limits(data: Dict, allowed: List[str]) -> Dict
```

#### 4.2.3 Pseudonymization (pseudonymization.py)

```
Enum PseudonymizationMethod:
    TOKENIZATION = "tokenization"
    HASHING = "hashing"
    ENCRYPTION = "encryption"
    MASKING = "masking"
    GENERALIZATION = "generalization"

Dataclass PseudonymizationConfig:
    method: PseudonymizationMethod
    key_management: str  # "internal", "hsm", "kms"
    reversible: bool
    salt_rotation_days: int

Class Pseudonymizer:
    - pseudonymize(data: str, config: PseudonymizationConfig) -> str
    - pseudonymize_batch(records: List[Dict], fields: List[str]) -> List[Dict]
    - depseudonymize(token: str, config: PseudonymizationConfig) -> str
    - rotate_keys() -> None
    - verify_pseudonymization(original: str, pseudonymized: str) -> bool
```

Integration with existing:
- Leverage `services/secure_logging.py` patterns
- Use existing `tests/test_pii_detection.py` detection

#### 4.2.4 RetentionManager (retention_manager.py)

```
Dataclass RetentionPolicy:
    policy_id: str
    data_category: str
    retention_period_days: int
    retention_basis: str  # Legal requirement, business need
    deletion_method: str  # "hard_delete", "anonymize", "archive"
    review_period_days: int
    exceptions: List[str]

Dataclass RetentionJob:
    job_id: str
    policy_id: str
    scheduled_at: datetime
    executed_at: Optional[datetime]
    records_processed: int
    records_deleted: int
    records_anonymized: int
    status: str

Class GDPRRetentionManager:
    - register_policy(policy: RetentionPolicy)
    - schedule_retention_jobs() -> List[RetentionJob]
    - execute_retention(job_id: str) -> RetentionJob
    - check_retention_compliance() -> List[Violation]
    - extend_retention(data_id: str, reason: str, duration: int)
```

Integration with:
- `services/compliance/retention_policy.py` (MiFID II)
- Storage limitation principle enforcement

#### 4.2.5 AutoErasureScheduler (auto_erasure_scheduler.py)

> **Article 17 Compliance**: Automatic erasure trigger mechanism after MiFID II retention periods expire.

```
Enum ErasureTrigger:
    MIFID_EXPIRY = "mifid_expiry"           # 5-7 year retention ended
    CONSENT_WITHDRAWAL = "consent_withdrawal" # User withdrew consent
    PURPOSE_FULFILLED = "purpose_fulfilled"   # Processing purpose completed
    MANUAL_REQUEST = "manual_request"         # Article 17 erasure request

Dataclass ScheduledErasure:
    erasure_id: str              # UUID
    data_subject_id: str         # Pseudonymized subject ID
    data_categories: List[str]   # Categories to erase
    trigger: ErasureTrigger      # What triggered erasure
    original_retention_end: datetime  # When retention period ended
    scheduled_erasure_date: datetime  # When erasure will execute
    grace_period_days: int = 30  # Buffer for legal holds
    legal_hold_active: bool = False  # Pause if litigation pending
    status: str = "pending"      # pending/executing/completed/blocked

Class AutoErasureScheduler:
    - schedule_post_retention_erasure(data_id: str, retention_end: datetime) -> ScheduledErasure
    - check_legal_holds(erasure_id: str) -> bool
    - execute_scheduled_erasure(erasure_id: str) -> ErasureResult
    - apply_grace_period(erasure_id: str, reason: str) -> ScheduledErasure
    - get_pending_erasures() -> List[ScheduledErasure]
    - integrate_with_restrictions(erasure_id: str, restriction: LegalRestriction) -> bool
```

**MiFID II → GDPR Transition Logic:**
```
T+0         Data collected (trading record created)
T+5y        MiFID II minimum retention reached
T+5y+30d    Grace period for audit/legal holds
T+5y+31d    AUTO-ERASURE TRIGGERED (if no holds)
            ├─ Check RestrictionsFramework for Art. 23 blocks
            ├─ Verify no pending litigation
            ├─ Execute pseudonymized deletion
            └─ Log erasure for accountability (Art. 24)
```

Integration with:
- `RestrictionsFramework` for Article 23 legal holds
- `AccountabilityFramework` for erasure evidence logging
- `services/compliance/retention_policy.py` for retention period tracking

### 4.3 Platform-Specific Privacy Controls

| System Component | Privacy Controls | Implementation |
|-----------------|------------------|----------------|
| Audit Trail | Pseudonymization, Encryption | User IDs tokenized |
| Trade Logs | Encryption at rest | AES-256 |
| API Keys | Hashing, Secure storage | bcrypt/argon2 |
| User Sessions | Expiry, Rotation | JWT with short TTL |
| ML Features | Aggregation, Anonymization | No PII in features |
| Logs | PII Detection, Masking | SecureLogFilter |

### 4.4 Test Specifications

```
test_gdpr_phase4_privacy_engineering.py:
├── test_privacy_by_design/
│   ├── test_system_assessment
│   ├── test_control_gap_identification
│   ├── test_privacy_defaults
│   ├── test_privacy_recommendations
│   └── test_risk_level_calculation
├── test_data_minimization/
│   ├── test_schema_analysis
│   ├── test_excessive_collection_flag
│   ├── test_minimization_suggestions
│   ├── test_collection_enforcement
│   └── test_necessity_classification
├── test_pseudonymization/
│   ├── test_tokenization
│   ├── test_hashing
│   ├── test_encryption_pseudonymization
│   ├── test_masking
│   ├── test_batch_pseudonymization
│   ├── test_depseudonymization
│   ├── test_key_rotation
│   ├── test_irreversibility_for_hashing
│   └── test_secure_logging_integration
├── test_retention/
│   ├── test_policy_registration
│   ├── test_job_scheduling
│   ├── test_retention_execution
│   ├── test_hard_deletion
│   ├── test_anonymization_deletion
│   ├── test_archive_deletion
│   ├── test_retention_extension
│   ├── test_mifid_retention_override
│   └── test_compliance_check
├── test_auto_erasure/
│   ├── test_mifid_expiry_trigger
│   ├── test_grace_period_application
│   ├── test_legal_hold_blocks_erasure
│   ├── test_restrictions_framework_integration
│   └── test_accountability_logging
├── test_edge_cases/
│   ├── test_erasure_during_active_investigation
│   ├── test_concurrent_retention_extension_and_erasure
│   ├── test_mifid_7year_vs_5year_retention_conflict
│   ├── test_partial_erasure_with_linked_records
│   └── test_erasure_request_during_grace_period
└── test_integration/
    ├── test_pii_detection_integration
    ├── test_secure_logging_integration
    └── test_mifid_retention_integration
```

**Expected test count**: ~120-140 tests

---

## Phase 5: Data Breach Management

**Estimated Complexity**: High
**Dependencies**: Phase 1, Phase 2, Phase 4
**Test Coverage Target**: 100%

### 5.1 Objectives

Implement Articles 33-34 breach notification:
- Breach detection integration
- Risk assessment for breaches
- 72-hour supervisory authority notification
- Data subject notification (high risk)
- Integration with DORA incident management

### 5.2 Components to Implement

#### 5.2.1 BreachDetection (breach_detection.py)

```
Enum BreachType:
    CONFIDENTIALITY = "confidentiality"  # Unauthorized access
    INTEGRITY = "integrity"               # Data alteration
    AVAILABILITY = "availability"         # Data loss/destruction

Enum BreachSource:
    CYBER_ATTACK = "cyber_attack"
    INSIDER_THREAT = "insider_threat"
    ACCIDENTAL_DISCLOSURE = "accidental_disclosure"
    SYSTEM_FAILURE = "system_failure"
    THIRD_PARTY_BREACH = "third_party_breach"
    LOST_DEVICE = "lost_device"

Dataclass BreachIndicator:
    indicator_id: str
    indicator_type: str
    description: str
    severity: str
    detected_at: datetime
    source_system: str
    evidence: Dict[str, Any]

Class BreachDetector:
    - register_indicator(indicator: BreachIndicator)
    - analyze_indicators() -> List[PotentialBreach]
    - integrate_dora_incidents() -> List[PotentialBreach]
    - check_access_anomalies() -> List[BreachIndicator]
    - check_data_exfiltration() -> List[BreachIndicator]
```

Integration points:
- `services/dora/incident_management.py`
- `services/dora/detection.py`
- Security monitoring systems

#### 5.2.2 BreachAssessment (breach_assessment.py)

Risk assessment per EDPB Guidelines:

```
Dataclass BreachRiskAssessment:
    assessment_id: str
    breach_id: str

    # Data affected
    data_categories: List[str]
    special_category_data: bool
    data_subjects_count: int
    data_subjects_categories: List[str]

    # Impact assessment
    severity_of_consequences: str  # "low", "medium", "high", "very_high"
    likelihood_of_consequences: str
    overall_risk_level: str

    # Specific risks
    identity_theft_risk: bool
    financial_loss_risk: bool
    discrimination_risk: bool
    reputational_damage_risk: bool
    loss_of_confidentiality_risk: bool

    # Notification requirements
    notify_supervisory_authority: bool  # Article 33
    notify_data_subjects: bool          # Article 34
    notification_rationale: str

    # Mitigation
    mitigation_measures: List[str]
    residual_risk: str

Class BreachRiskAssessor:
    - assess_breach(breach: PersonalDataBreach) -> BreachRiskAssessment
    - calculate_severity(breach: PersonalDataBreach) -> str
    - determine_notification_requirement(assessment: BreachRiskAssessment) -> Dict
    - document_no_notification_rationale(breach_id: str, reason: str)
```

#### 5.2.3 BreachNotification (breach_notification.py)

Articles 33-34 notification workflow:

```
Enum NotificationStatus:
    PENDING = "pending"
    DRAFTED = "drafted"
    APPROVED = "approved"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    ADDITIONAL_INFO_REQUESTED = "additional_info"
    CLOSED = "closed"

Dataclass SupervisoryAuthorityNotification:
    notification_id: str
    breach_id: str

    # Article 33(3) required information
    nature_of_breach: str
    data_categories_affected: List[str]
    approximate_data_subjects: int
    dpo_contact: str
    likely_consequences: str
    mitigation_measures: str

    # Timing
    breach_detected_at: datetime
    notification_deadline: datetime  # +72 hours
    submitted_at: Optional[datetime]
    is_phased: bool
    phase_number: int

    # Status
    status: NotificationStatus
    delay_reason: Optional[str]  # If >72 hours

Dataclass DataSubjectNotification:
    notification_id: str
    breach_id: str

    # Article 34(2) required information
    nature_of_breach: str
    dpo_contact: str
    likely_consequences: str
    mitigation_measures: str

    # Delivery
    notification_method: str  # "email", "post", "public_announcement"
    data_subjects_notified: int
    notification_template_id: str

    # Status
    status: NotificationStatus
    delivered_at: Optional[datetime]

Class BreachNotificationManager:
    - create_sa_notification(breach_id: str) -> SupervisoryAuthorityNotification
    - create_ds_notification(breach_id: str) -> DataSubjectNotification
    - check_72h_deadline(breach_id: str) -> DeadlineStatus
    - submit_notification(notification_id: str) -> SubmissionResult
    - add_phased_update(notification_id: str, update: Dict)
    - generate_notification_report(breach_id: str) -> Report
```

#### 5.2.4 IncidentResponse (incident_response.py)

GDPR-specific incident response:

```
Dataclass GDPRIncidentResponsePlan:
    plan_id: str
    version: str

    # Roles
    incident_response_team: List[str]
    dpo_contact: str
    legal_contact: str
    communications_contact: str

    # Procedures
    detection_procedures: List[str]
    containment_procedures: List[str]
    assessment_procedures: List[str]
    notification_procedures: List[str]
    recovery_procedures: List[str]
    lessons_learned_procedures: List[str]

    # Timelines
    initial_response_hours: int  # Target: <4 hours
    assessment_deadline_hours: int  # Target: <24 hours
    sa_notification_hours: int  # Deadline: 72 hours

    # Templates
    notification_templates: Dict[str, str]
    communication_templates: Dict[str, str]

Class GDPRIncidentResponseManager:
    - initiate_response(breach: PersonalDataBreach) -> IncidentResponse
    - execute_containment(incident_id: str, actions: List[str])
    - track_timeline(incident_id: str) -> TimelineStatus
    - escalate_to_dpo(incident_id: str)
    - conduct_post_incident_review(incident_id: str) -> Review
```

### 5.3 Integration with DORA

Map GDPR breach to DORA incident:

```
GDPR Breach                    DORA Incident
─────────────────────────────────────────────────
Detection                   →  ICT Incident Detection
Assessment                  →  Incident Classification
SA Notification (72h)       →  Major Incident Reporting (24h/72h/30d)
DS Notification             →  Client Communication
Post-Incident Review        →  Learning and Evolving
```

### 5.4 Test Specifications

```
test_gdpr_phase5_breach_management.py:
├── test_detection/
│   ├── test_indicator_registration
│   ├── test_indicator_analysis
│   ├── test_dora_incident_integration
│   ├── test_access_anomaly_detection
│   ├── test_exfiltration_detection
│   └── test_breach_correlation
├── test_assessment/
│   ├── test_risk_assessment_creation
│   ├── test_severity_calculation
│   ├── test_special_category_impact
│   ├── test_data_subject_count_impact
│   ├── test_notification_determination
│   ├── test_no_notification_documentation
│   ├── test_mitigation_recommendations
│   └── test_residual_risk_calculation
├── test_notification/
│   ├── test_sa_notification_creation
│   ├── test_ds_notification_creation
│   ├── test_72h_deadline_tracking
│   ├── test_deadline_breach_alert
│   ├── test_phased_notification
│   ├── test_delay_documentation
│   ├── test_notification_submission
│   ├── test_template_generation
│   └── test_notification_audit_trail
├── test_incident_response/
│   ├── test_response_initiation
│   ├── test_containment_execution
│   ├── test_timeline_tracking
│   ├── test_dpo_escalation
│   ├── test_post_incident_review
│   └── test_lessons_learned_integration
├── test_edge_cases/
│   ├── test_dora_classification_before_gdpr_72h
│   ├── test_classification_delayed_past_detection_24h
│   ├── test_dual_trigger_earliest_deadline
│   ├── test_nis2_ai_act_concurrent_notifications
│   ├── test_breach_without_personal_data_dora_only
│   ├── test_personal_data_breach_without_ict_gdpr_only
│   └── test_cross_border_breach_multi_sa_notification
└── test_integration/
    ├── test_dora_incident_mapping
    ├── test_dora_reporting_alignment
    └── test_unified_incident_dashboard
```

**Expected test count**: ~120-140 tests

---

## Phase 6: DPIA & Governance

**Estimated Complexity**: Medium
**Dependencies**: All previous phases
**Test Coverage Target**: 100%

### 6.1 Objectives

Complete GDPR governance framework:
- Data Protection Impact Assessment (Article 35)
- DPO tools and interface (Articles 37-39)
- International data transfers (Articles 44-49)
- Compliance dashboard and reporting

### 6.2 Components to Implement

#### 6.2.1 DPIA (dpia.py)

Article 35 Data Protection Impact Assessment:

```
Enum DPIAStatus:
    SCREENING = "screening"
    IN_PROGRESS = "in_progress"
    DPO_REVIEW = "dpo_review"
    PRIOR_CONSULTATION = "prior_consultation"  # Article 36
    APPROVED = "approved"
    REJECTED = "rejected"

Enum DPIATrigger:
    SYSTEMATIC_EVALUATION = "systematic_evaluation"  # Article 35(3)(a)
    LARGE_SCALE_SPECIAL_DATA = "large_scale_special"  # Article 35(3)(b)
    PUBLIC_MONITORING = "public_monitoring"  # Article 35(3)(c)
    NEW_TECHNOLOGY = "new_technology"
    PROFILING = "profiling"
    AUTOMATED_DECISIONS = "automated_decisions"

Dataclass DPIARecord:
    dpia_id: str
    project_name: str
    project_description: str

    # Screening
    trigger_criteria: List[DPIATrigger]
    dpia_required: bool
    screening_rationale: str

    # Article 35(7) minimum contents
    systematic_description: str
    necessity_assessment: str
    proportionality_assessment: str
    risk_assessment: List[RiskItem]
    mitigation_measures: List[MitigationMeasure]

    # Consultation
    dpo_advice: str
    data_subject_views: str  # Article 35(9)

    # Review
    review_schedule: str
    last_review_date: datetime
    next_review_date: datetime

    # Status
    status: DPIAStatus
    created_at: datetime
    approved_at: Optional[datetime]
    approved_by: Optional[str]

Dataclass RiskItem:
    risk_id: str
    description: str
    likelihood: str
    impact: str
    affected_rights: List[str]
    risk_score: float
    mitigation: str
    residual_risk: str

Class DPIAManager:
    - screen_for_dpia(project: Dict) -> DPIAScreeningResult
    - create_dpia(project_name: str) -> DPIARecord
    - assess_risks(dpia_id: str) -> List[RiskItem]
    - add_mitigation(dpia_id: str, mitigation: MitigationMeasure)
    - submit_for_dpo_review(dpia_id: str)
    - initiate_prior_consultation(dpia_id: str)  # Article 36
    - schedule_review(dpia_id: str, interval_months: int)
    - generate_dpia_report(dpia_id: str, format: str) -> bytes
```

#### 6.2.2 DPOInterface (dpo_interface.py)

Tools for Data Protection Officer:

```
Dataclass DPODashboard:
    # Overview
    active_dpias: int
    pending_dsars: int
    open_breaches: int
    consent_withdrawals_30d: int
    upcoming_deadlines: List[Deadline]

    # Compliance status
    ropa_complete: bool
    policies_current: bool
    training_complete: bool

    # Alerts
    critical_alerts: List[Alert]

Class DPOToolkit:
    - get_dashboard() -> DPODashboard
    - review_dpia(dpia_id: str, decision: str, comments: str)
    - approve_dsar_response(dsar_id: str)
    - advise_on_processing(processing_id: str, advice: str)
    - generate_compliance_report(period: str) -> Report
    - schedule_audit(area: str, date: datetime)
    - manage_training(training_id: str, action: str)
    - communicate_with_sa(message: str, attachments: List)
```

#### 6.2.3 InternationalTransfers (international_transfers.py)

Articles 44-49 transfer mechanisms:

```
Enum TransferMechanism:
    ADEQUACY_DECISION = "adequacy"  # Article 45
    SCCs = "standard_contractual_clauses"  # Article 46(2)(c)
    BCRs = "binding_corporate_rules"  # Article 47
    DEROGATION_CONSENT = "consent"  # Article 49(1)(a)
    DEROGATION_CONTRACT = "contract"  # Article 49(1)(b)
    DEROGATION_PUBLIC_INTEREST = "public_interest"  # Article 49(1)(d)
    DEROGATION_LEGAL_CLAIMS = "legal_claims"  # Article 49(1)(e)

Dataclass ThirdCountryTransfer:
    transfer_id: str
    destination_country: str
    recipient_name: str
    data_categories: List[str]
    transfer_purpose: str
    mechanism: TransferMechanism
    adequacy_decision_reference: Optional[str]
    scc_version: Optional[str]
    supplementary_measures: List[str]
    tia_completed: bool  # Transfer Impact Assessment
    tia_date: Optional[datetime]

Dataclass TransferImpactAssessment:
    assessment_id: str
    transfer_id: str
    destination_country: str

    # Assessment criteria
    legislation_assessment: str
    surveillance_assessment: str
    practical_enforcement: str
    overall_risk: str

    # Supplementary measures
    technical_measures: List[str]
    organizational_measures: List[str]
    contractual_measures: List[str]

    # Conclusion
    transfer_permitted: bool
    conditions: List[str]

Class InternationalTransferManager:
    - register_transfer(transfer: ThirdCountryTransfer)
    - check_adequacy(country: str) -> AdequacyStatus
    - validate_mechanism(transfer_id: str) -> ValidationResult
    - conduct_tia(transfer_id: str) -> TransferImpactAssessment
    - recommend_supplementary_measures(transfer_id: str) -> List[str]
    - suspend_transfer(transfer_id: str, reason: str)
    - generate_transfer_map() -> TransferMap
```

#### 6.2.4 UKAdequacyContingency (uk_adequacy_contingency.py) - NEW

**Critical: UK Adequacy Sunset Handling**

Per [European Commission announcement (July 2025)](https://commission.europa.eu/law/law-topic/data-protection/international-dimension-data-protection/adequacy-decisions_en), the UK adequacy decision expires **27 December 2025**. This module provides automated contingency management.

```
Dataclass UKContingencyStatus:
    adequacy_expiry_date: datetime = datetime(2025, 12, 27)
    preparation_start_date: datetime = datetime(2025, 9, 1)  # Q3 2025
    current_status: str  # "monitoring", "preparing", "contingency_active"
    new_adequacy_adopted: Optional[bool]
    last_ec_communication_date: Optional[datetime]

Dataclass UKProcessor:
    processor_id: str
    processor_name: str
    services_provided: List[str]
    data_categories_transferred: List[str]
    current_mechanism: str  # "adequacy"
    scc_prepared: bool
    scc_document_id: Optional[str]
    tia_completed: bool
    tia_document_id: Optional[str]
    contingency_ready: bool

Dataclass UKContingencyPlan:
    plan_id: str
    created_at: datetime
    uk_processors: List[UKProcessor]
    total_data_subjects_affected: int
    sccs_required: int
    sccs_prepared: int
    tias_required: int
    tias_completed: int
    data_subject_notification_required: bool
    notification_template_id: Optional[str]
    go_live_date: datetime  # When to switch mechanisms

Class UKAdequacyContingency:
    """
    Manages UK adequacy decision sunset and contingency planning.

    Timeline:
    - Q3 2025: Begin preparation (SCCs, TIAs)
    - Q4 2025: Complete all preparations, test mechanisms
    - 27 Dec 2025: If no new adequacy, activate contingency
    """

    SUNSET_DATE = datetime(2025, 12, 27)
    PREPARATION_START = datetime(2025, 9, 1)
    FINAL_PREPARATION = datetime(2025, 11, 1)

    # Status monitoring
    - check_ec_announcements() -> ECAnnouncementStatus
    - get_contingency_status() -> UKContingencyStatus
    - should_start_preparation() -> bool

    # Processor management
    - identify_uk_processors() -> List[UKProcessor]
    - prepare_sccs_for_processor(processor_id: str) -> SCCDocument
    - conduct_uk_tia(processor_id: str) -> TransferImpactAssessment

    # Contingency activation
    - generate_contingency_plan() -> UKContingencyPlan
    - validate_contingency_readiness() -> ReadinessReport
    - activate_contingency() -> ActivationResult

    # Data subject notification (if mechanism changes)
    - prepare_ds_notification() -> NotificationTemplate
    - notify_affected_data_subjects() -> NotificationResult

    # Reporting
    - get_uk_transfer_summary() -> UKTransferSummary
    - generate_dpo_briefing() -> DPOBriefing
```

**Automated Alerts:**

| Trigger | Alert | Action Required |
|---------|-------|-----------------|
| Q3 2025 begins | "UK contingency preparation required" | Start SCC/TIA preparation |
| 60 days before sunset | "UK adequacy expiring soon" | Verify all preparations complete |
| 30 days before sunset | "Final UK preparation check" | Test mechanism switch |
| EC new adequacy adopted | "UK adequacy renewed" | Cancel contingency |
| Sunset with no adequacy | "UK contingency activated" | Switch to SCCs |

Adequacy decisions list (as of December 2024):
- Andorra, Argentina, Canada (PIPEDA commercial orgs), Faroe Islands
- Guernsey, Israel, Isle of Man, Japan, Jersey
- New Zealand, Republic of Korea, Switzerland, Uruguay
- **United Kingdom** (⚠️ SUNSET: 27 Dec 2025 - see contingency below)
- **EU-US Data Privacy Framework** (DPF participants only)
- **European Patent Organisation (EPO)** - NEW in 2024

#### 6.2.4 ComplianceDashboard (compliance_dashboard.py)

Unified GDPR compliance view:

```
Dataclass GDPRComplianceStatus:
    # Article compliance
    article_5_principles: ComplianceLevel
    article_6_legal_basis: ComplianceLevel
    article_7_consent: ComplianceLevel
    article_12_22_data_subject_rights: ComplianceLevel
    article_25_privacy_by_design: ComplianceLevel
    article_30_ropa: ComplianceLevel
    article_32_security: ComplianceLevel
    article_33_34_breach_notification: ComplianceLevel
    article_35_dpia: ComplianceLevel
    article_37_39_dpo: ComplianceLevel
    article_44_49_transfers: ComplianceLevel

    # Overall
    overall_compliance: float  # 0-100%
    critical_gaps: List[str]
    recommendations: List[str]

Class GDPRComplianceDashboard:
    - get_compliance_status() -> GDPRComplianceStatus
    - get_pending_actions() -> List[Action]
    - get_upcoming_deadlines() -> List[Deadline]
    - generate_compliance_report(format: str) -> bytes
    - compare_with_previous(date: datetime) -> ComplianceTrend
    - integrate_with_dora_dashboard() -> UnifiedView
    - integrate_with_mifid_dashboard() -> UnifiedView
```

### 6.3 Platform-Specific DPIAs

Pre-configured DPIA templates:

| Processing Activity | DPIA Required | Trigger |
|--------------------|---------------|---------|
| Algorithmic Trading | Yes | Automated decisions, profiling |
| ML Model Training | Yes | Large-scale, new technology |
| Risk Scoring | Yes | Profiling, automated decisions |
| API Key Storage | Screening | Special security considerations |
| Audit Logging | No | Legal obligation, minimal risk |

### 6.4 Test Specifications

```
test_gdpr_phase6_dpia_governance.py:
├── test_dpia/
│   ├── test_dpia_screening
│   ├── test_dpia_trigger_detection
│   ├── test_dpia_creation
│   ├── test_risk_assessment
│   ├── test_mitigation_tracking
│   ├── test_dpo_review_workflow
│   ├── test_prior_consultation
│   ├── test_dpia_review_scheduling
│   ├── test_report_generation
│   └── test_algorithmic_trading_dpia
├── test_dpo_interface/
│   ├── test_dashboard_metrics
│   ├── test_dpia_review
│   ├── test_dsar_approval
│   ├── test_compliance_reporting
│   ├── test_audit_scheduling
│   ├── test_sa_communication
│   └── test_alert_management
├── test_international_transfers/
│   ├── test_transfer_registration
│   ├── test_adequacy_check
│   ├── test_scc_validation
│   ├── test_tia_completion
│   ├── test_supplementary_measures
│   ├── test_transfer_suspension
│   ├── test_transfer_map_generation
│   └── test_eu_us_dpf_handling
├── test_uk_adequacy_contingency/
│   ├── test_sunset_date_monitoring
│   ├── test_auto_fallback_to_scc
│   ├── test_uk_processor_inventory
│   ├── test_pre_deadline_alert_90_days
│   ├── test_contingency_plan_activation
│   ├── test_extension_scenario_handling
│   └── test_manual_override_mechanism
├── test_edge_cases/
│   ├── test_adequacy_revocation_mid_transfer
│   ├── test_dual_legal_basis_transfers
│   ├── test_onward_transfer_chain_validation
│   ├── test_third_country_government_access_risk
│   └── test_bcr_vs_scc_selection_criteria
├── test_compliance_dashboard/
│   ├── test_status_calculation
│   ├── test_gap_identification
│   ├── test_recommendation_generation
│   ├── test_deadline_tracking
│   ├── test_trend_comparison
│   └── test_report_generation
└── test_cross_regulation/
    ├── test_dora_dashboard_integration
    ├── test_mifid_dashboard_integration
    ├── test_ai_act_alignment
    └── test_unified_compliance_view
```

**Expected test count**: ~115-135 tests

---

## Implementation Timeline

| Phase | Description | Est. Tests | Dependencies |
|-------|-------------|------------|--------------|
| 0 | Core Definitions & Processor Framework | 70-90 | None |
| 1 | Foundation & Legal Framework | 90-110 | Phase 0 |
| 2a | Consent & Transparency | 60-70 | Phase 1 |
| 2b | Data Subject Rights | 100-120 | Phase 2a |
| 3 | ROPA & Documentation | 80-100 | Phases 0, 1, 2a, 2b |
| 4 | Privacy Engineering | 100-120 | Phases 1-3 |
| 5 | Breach Management | 100-120 | Phases 1, 2b, 4 |
| 6 | DPIA & Governance | 90-110 | All previous |

**Total estimated tests**: ~690-840 tests

---

## Cross-Regulation Alignment

### GDPR ↔ MiFID II

**IMPORTANT CLARIFICATION**: MiFID II does NOT "override" GDPR. Both regulations apply concurrently with the following resolution:

| GDPR Requirement | MiFID II Requirement | Resolution Approach |
|------------------|---------------------|---------------------|
| Storage limitation (Art. 5(1)(e)) | 5-7 year retention (MiFIR Art. 25) | MiFID II is **lawful basis** (Art. 6(1)(c) legal obligation), NOT an override. Data minimization still applies. |
| Data minimization | Full transaction records | **Pseudonymize personal identifiers** where not required for regulatory purposes. Retain only minimum necessary. |
| Purpose limitation | Regulatory compliance | Document MiFID II as explicit purpose in ROPA. No repurposing without additional basis. |
| Erasure rights (Art. 17) | Retention obligations | Erasure request acknowledged but **suspended** during MiFID II retention period. Automatic erasure upon expiry. |
| Audit trail | Transaction records | Shared infrastructure, but GDPR audit must log access to personal data specifically. |

**Implementation Pattern:**
```python
class MiFIDGDPRRetentionResolver:
    def resolve_erasure_request(self, request: ErasureRequest) -> ErasureDecision:
        # Check if MiFID II retention applies
        if self.mifid_retention_applies(request.data_category):
            return ErasureDecision(
                action="SUSPEND",
                reason="legal_obligation_mifid_ii",
                article_reference="Article 17(3)(b)",
                scheduled_erasure_date=self.calculate_mifid_expiry(request),
                pseudonymize_now=True  # GDPR minimization still applies!
            )
        else:
            return ErasureDecision(action="ERASE_IMMEDIATELY")
```

### GDPR ↔ DORA

| GDPR Requirement | DORA Requirement | Integration Approach |
|------------------|------------------|---------------------|
| Breach notification (72h) | Initial report (4h), Intermediate (72h), Final (30d) | **Unified incident workflow**: DORA timeline is stricter, so DORA triggers GDPR. Breach affecting personal data = both notifications. |
| Security measures (Art. 32) | ICT Risk Management (Art. 6-15) | Shared security controls. DORA is more prescriptive; implement DORA, map to GDPR. |
| Third-party assessment | Third-party ICT risk (Art. 28-44) | Combined processor/vendor assessment. Article 28 DPA + DORA register entry. |
| DPO role | ICT governance | DPO sits on ICT governance board. Cross-functional incident escalation. |
| Incident logging | Incident register | Unified `IncidentRegistry` with both GDPR and DORA fields. |

**Breach Timeline Coordination:**

> ⚠️ **IMPORTANT**: DORA deadlines are triggered by **classification as major incident**, not by detection.
> Per [Commission Delegated Regulation (EU) 2024/1772](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1772),
> DORA Article 19 requires initial notification within 4 hours of **classification** or 24 hours of **detection**, whichever is earlier.

```
Breach/Incident Timeline (Corrected):
──────────────────────────────────────────────────────────────────

T+0         Breach/Incident DETECTED
            ├─ Start GDPR 72h clock
            ├─ Start internal assessment
            └─ Log in unified incident registry

T+Xh        Incident CLASSIFIED as major (DORA assessment)
            ├─ If X ≤ 20h: DORA deadline = T+24h (from detection)
            └─ If X > 20h: DORA deadline = classification + 4h

T+4h from   DORA Initial Notification (if major ICT incident)
classification ├─ OR T+24h from detection (whichever is EARLIER)
            ├─ If breach involves personal data: flag for GDPR
            └─ Required: incident_id, classification, services_affected

T+24h       Internal GDPR Assessment COMPLETED
            ├─ Determine if personal data breach
            ├─ Assess risk to data subjects
            └─ Decision: notify SA? notify data subjects?

T+72h       GDPR Supervisory Authority Notification (if required)
            ├─ Per Article 33(1): "without undue delay and, where feasible,
            │   not later than 72 hours after having become aware"
            ├─ Document any delay reasons per Art. 33(1)
            └─ DORA intermediate report (can be combined if data breach)

T+30d       Final Reports
            ├─ DORA final report (Article 19)
            ├─ GDPR follow-up to SA if additional info available
            └─ Post-incident review
```

**Critical Distinction:**
- **GDPR 72h** starts from **becoming aware** of breach (detection)
- **DORA 4h** starts from **classification** as major incident (or 24h from detection)
- For incidents involving personal data: BOTH timelines run concurrently

### GDPR ↔ EU AI Act

| GDPR Requirement | EU AI Act Requirement | Integration Approach |
|------------------|----------------------|---------------------|
| Data governance | Art. 10 data quality | `DataGovernanceFramework` serves both. GDPR lawfulness + AI Act quality. |
| DPIA (Art. 35) | Conformity assessment | Combined assessment for high-risk AI. DPIA required if personal data + high-risk AI. |
| Transparency | Art. 13 transparency | Single transparency notice covering both. AI-specific disclosures per AI Act Annex IV. |
| Art. 22 automated decisions | Art. 14 human oversight | **Unified handler**: Every Art. 22 decision in high-risk AI needs Art. 14 oversight. |
| Right to explanation | Art. 13(3)(b)(iii) | Explainability module serves both. GDPR "meaningful information" = AI Act "clear and understandable information". |
| Lawful basis | Art. 10(6) personal data | AI Act explicitly references GDPR for personal data. No conflict, layered compliance. |

**High-Risk AI Systems and GDPR:**
```
Algorithmic Trading Platform Assessment:
─────────────────────────────────────────
EU AI Act Annex III check:
  - Category 5(a): Credit scoring → NO (not credit)
  - Category 5(b): Life insurance → NO
  - Category 8: Financial services affecting access → POSSIBLE

If classified as high-risk:
  → Art. 14 human oversight MANDATORY
  → Art. 22 GDPR applies for automated decisions
  → Combined DPIA + Conformity Assessment
  → Fundamental rights impact assessment
```

### GDPR ↔ NIS2

| GDPR Requirement | NIS2 Requirement | Integration Approach |
|------------------|-----------------|---------------------|
| Security (Art. 32) | Art. 21 security measures | NIS2 is sector-specific, broader scope. Implement NIS2, subset covers GDPR Art. 32. |
| Breach notification | Incident notification (24h) | If personal data involved in NIS2 incident → dual notification. |
| Supply chain | Supply chain security | Processor assessment includes NIS2 supply chain requirements. |

### Cross-Regulation Priority Matrix

When regulations conflict, apply this priority:

| Scenario | Priority | Rationale |
|----------|----------|-----------|
| Notification timelines | **Most urgent** | DORA 4h → GDPR 72h. Follow strictest, satisfy all. |
| Retention periods | **Legal obligation** | MiFID II as lawful basis (Art. 6(1)(c)) with GDPR minimization. |
| Security measures | **Most comprehensive** | DORA/NIS2 supersede Art. 32 specifics, but Art. 32 purpose remains. |
| Human oversight | **Both required** | Art. 22 GDPR + Art. 14 AI Act are complementary, not alternatives. |
| Documentation | **Unified** | Single ROPA with all regulation-specific fields. |

---

## Configuration File Template

```yaml
# configs/gdpr/gdpr_config.yaml

gdpr:
  version: "1.0"

  organization:
    name: "Your Organization"
    controller_type: "controller"  # controller, processor, joint
    dpo:
      name: "Data Protection Officer"
      email: "dpo@organization.com"
      phone: "+1234567890"

  supervisory_authority:
    name: "Data Protection Commission"
    country: "Ireland"
    contact: "info@dataprotection.ie"

  legal_bases:
    defaults:
      trading_execution: "contract"
      compliance_monitoring: "legal_obligation"
      risk_management: "legitimate_interest"
      marketing: "consent"

  retention:
    default_days: 365
    # Note: MiFID II provides LEGAL BASIS (Art. 6(1)(c)), not an override
    # GDPR minimization still applies - pseudonymize where possible
    trading_records_years: 7  # MiFID II legal obligation
    audit_logs_years: 7       # Regulatory requirement
    consent_records_years: 3  # Consent evidence
    dsar_records_years: 3     # DSAR documentation
    # Automatic erasure scheduled upon retention expiry
    auto_erasure_on_expiry: true
    pseudonymize_during_retention: true

  consent:
    expiry_days: 365
    require_double_opt_in: true
    granular_purposes: true

  dsar:
    response_deadline_days: 30
    extension_allowed_days: 60
    identity_verification_required: true

  breach:
    notification_deadline_hours: 72
    auto_escalate_to_dpo: true

  international_transfers:
    default_mechanism: "sccs"
    tia_required: true

  privacy_by_design:
    enforce_minimization: true
    default_pseudonymization: false
    encryption_at_rest: true
    encryption_in_transit: true
```

---

## International Data Transfers: Adequacy Decisions (2024-2025 Update)

### Current Adequacy Decisions

As of December 2024, the European Commission recognizes the following countries/territories as providing adequate protection:

| Country/Territory | Decision Date | Status | Notes |
|------------------|---------------|--------|-------|
| Andorra | 2010 | Active | Reviewed Jan 2024 - adequate |
| Argentina | 2003 | Active | Reviewed Jan 2024 - adequate |
| Canada (PIPEDA) | 2002 | Active | Commercial organizations only |
| Faroe Islands | 2010 | Active | Reviewed Jan 2024 - adequate |
| Guernsey | 2003 | Active | Reviewed Jan 2024 - adequate |
| Israel | 2011 | Active | Reviewed Jan 2024 - adequate |
| Isle of Man | 2004 | Active | Reviewed Jan 2024 - adequate |
| Japan | 2019 | Active | Mutual adequacy |
| Jersey | 2008 | Active | Reviewed Jan 2024 - adequate |
| New Zealand | 2013 | Active | Reviewed Jan 2024 - adequate |
| Republic of Korea | 2022 | Active | |
| Switzerland | 2000 | Active | Reviewed Jan 2024 - adequate |
| **United Kingdom** | 2021 | **SUNSET: 27 Dec 2025** | **ACTION REQUIRED** - monitor EU decision |
| **United States (EU-US DPF)** | 2023 | Active | Reviewed Oct 2024 - DPF participants only |
| Uruguay | 2012 | Active | Reviewed Jan 2024 - adequate |
| European Patent Organisation | 2024 | Active | New in 2024 |

### Critical: UK Adequacy Decision

**⚠️ WARNING**: The UK adequacy decision expires **27 December 2025**.

The European Commission is reviewing UK data protection law (Data Use and Access Act) and will decide whether to adopt a new adequacy decision. If no new decision is adopted:

**Required Contingency Plan:**
1. Prepare Standard Contractual Clauses (SCCs) for UK transfers
2. Conduct Transfer Impact Assessments (TIAs) for UK data flows
3. Identify UK processors and prepare alternative transfer mechanisms
4. Set calendar reminder: **Q3 2025** - finalize UK transfer strategy

### Alternative Transfer Mechanisms

When adequacy decision is absent, use in order of preference:

1. **Standard Contractual Clauses (SCCs)** - Commission Decision 2021/914
   - Module 1: Controller → Controller
   - Module 2: Controller → Processor
   - Module 3: Processor → Processor
   - Module 4: Processor → Controller

2. **Binding Corporate Rules (BCRs)** - for intra-group transfers

3. **Derogations (Art. 49)** - only for occasional, non-repetitive transfers
   - Explicit consent (informed of risks)
   - Contract performance
   - Important public interest
   - Legal claims
   - Vital interests

### Transfer Impact Assessment (TIA) Requirements

Per EDPB Recommendations 01/2020, when using SCCs:

```
TIA Components:
├── Step 1: Know your transfers (data mapping)
├── Step 2: Identify transfer tools used
├── Step 3: Assess third country laws
├── Step 4: Identify supplementary measures
├── Step 5: Implement procedural steps
└── Step 6: Re-evaluate at appropriate intervals
```

---

## References

### Official Sources
- [GDPR Full Text](https://gdpr-info.eu/)
- [EDPB Guidelines & Best Practices](https://www.edpb.europa.eu/our-work-tools/general-guidance/guidelines-recommendations-best-practices_en)
- [EU Adequacy Decisions](https://commission.europa.eu/law/law-topic/data-protection/international-dimension-data-protection/adequacy-decisions_en)
- [Article 30 Guidance - DPC Ireland](https://www.dataprotection.ie/en/dpc-guidance/records-of-processing-article-30-guidance)
- [DPIA Guidance - European Commission](https://commission.europa.eu/law/law-topic/data-protection/rules-business-and-organisations/obligations/when-data-protection-impact-assessment-dpia-required_en)

### EDPB Guidelines 2024-2025

**Critical Guidelines to Implement:**

| Guideline | Date | Relevance to Platform |
|-----------|------|----------------------|
| [Guidelines on Legitimate Interest](https://www.edpb.europa.eu/our-work-tools/our-documents/guidelines/guidelines-12024-processing-personal-data-based_en) | Oct 2024 | Risk management processing basis |
| [Guidelines 02/2024 on Article 48](https://www.edpb.europa.eu/our-work-tools/our-documents/guidelines/guidelines-022024-article-48-gdpr_en) | June 2025 | Foreign authority data requests |
| [Joint DMA-GDPR Guidelines](https://www.edpb.europa.eu/our-work-tools/our-documents/guidelines/joint-guidelines-interplay-between-dma-and-gdpr_en) | 2025 | If platform reaches DMA thresholds |
| [Right of Access CEF Report](https://www.edpb.europa.eu/our-work-tools/our-documents/report/coordinated-enforcement-action-right-access_en) | Jan 2025 | DSAR implementation best practices |
| [Guidelines on Data Breach Notification](https://www.edpb.europa.eu/our-work-tools/our-documents/guidelines/guidelines-92022-personal-data-breach-notification_en) | 2023 (v2.0) | Breach assessment and notification |
| [EDPB Work Programme 2024-2025](https://www.edpb.europa.eu/system/files/2024-10/edpb_work_programme_2024-2025_en.pdf) | Oct 2024 | Strategic priorities |
| [Guidelines on Pseudonymisation](https://www.edpb.europa.eu/our-work-tools/our-documents/guidelines/guidelines-012025-pseudonymisation_en) | 2025 | Privacy engineering techniques |

**2025 EDPB Strategic Priorities:**

| Priority Area | Deadline | Action Required |
|--------------|----------|-----------------|
| **Right to Erasure (CEF 2025)** | H1 2025 | Coordinated enforcement - audit erasure workflows |
| **AI Act Alignment** | Aug 2025 | GDPR-AI Act interplay for high-risk AI systems |
| **Cross-Border Processing** | Ongoing | One-stop-shop mechanism compliance |
| **International Transfers** | Dec 2025 | UK adequacy review, EPO implementation |

**2025 Coordinated Enforcement Focus**: Right to Erasure (Article 17)
- Expect increased scrutiny on erasure request handling
- Prioritize robust erasure workflows and documentation
- Implement AutoErasureScheduler for MiFID II retention expiry
- Document all erasure refusals with legal basis (Art. 17(3))

**Financial Services Specific Focus (2025):**
- Algorithmic trading DPIA requirements
- Cross-regulation reporting (DORA-GDPR-MiFID II)
- High-frequency trading data minimization
- Client profiling transparency obligations

### Implementation Guides
- [GDPR Compliance Checklist - Bitsight](https://www.bitsight.com/learn/gdpr-compliance-checklist)
- [DSAR Implementation - Securiti](https://securiti.ai/blog/dsar-rights-and-compliance/)
- [Breach Notification Guidelines - EDPB](https://www.edpb.europa.eu/system/files/2023-04/edpb_guidelines_202209_personal_data_breach_notification_v2.0_en.pdf)
- [SCCs Implementation Guide - European Commission](https://commission.europa.eu/law/law-topic/data-protection/international-dimension-data-protection/standard-contractual-clauses-scc_en)

### Related Regulations
- [DORA Integration Plan](./DORA_INTEGRATION_PLAN.md)
- [NIS2 Integration Plan](./NIS2_INTEGRATION_PLAN.md)
- [EU AI Act Integration](../services/ai_act/)

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Dec 2024 | AI-Generated | Initial plan |
| 1.1 | Dec 2024 | AI-Generated | Added Phase 0 (Art. 4, 28, 26), Art. 9 Special Categories, Art. 22 for algo trading |
| 1.2 | Dec 2024 | AI-Generated | Split Phase 2 into 2a/2b, fixed MiFID II interpretation, added EDPB 2024-2025 guidelines |
| 1.3 | Dec 2024 | AI-Generated | Added adequacy decisions 2024-2025, UK sunset warning, cross-regulation alignment |
| 1.4 | Dec 2024 | AI-Generated (Audit) | **Comprehensive audit and fixes**: Fixed Art. 22 classification for trading decisions, added missing Arts. 8, 10, 11, 23, 24, 27; corrected DORA-GDPR breach timeline (classification vs detection); added EPO to adequacy list; added UK contingency workflow (Art. 46 fallback); added AccountabilityFramework (Art. 24); added RestrictionsFramework (Art. 23); added AutoErasureScheduler for MiFID II expiry; expanded test specifications with edge cases; updated EDPB 2025 strategic priorities |

---

## Appendix A: GDPR Article Coverage Matrix

| Article | Description | Phase | Implementation Status |
|---------|-------------|-------|----------------------|
| 4 | Definitions | 0 | `definitions.py` |
| 5 | Processing Principles | 1 | `processing_principles.py` |
| 6 | Lawful Basis | 1 | `legal_basis.py` |
| 7 | Consent Conditions | 2a | `consent_manager.py` |
| **8** | **Child Consent** | 2a | `consent_manager.py` |
| 9 | Special Categories | 1 | `special_categories.py` |
| **10** | **Criminal Data** | 1 | `special_categories.py` |
| **11** | **No Identification Required** | 2b | `no_identification_handler.py` |
| 12 | Transparent Communication | 2a | `transparency_notices.py` |
| 13 | Information at Collection | 2a | `information_provision.py` |
| 14 | Information Not From Subject | 2a | `information_provision.py` |
| 15 | Right of Access | 2b | `dsar_handler.py` |
| 16 | Right to Rectification | 2b | `data_subject_rights.py` |
| 17 | Right to Erasure | 2b | `erasure_manager.py`, `auto_erasure_scheduler.py` |
| 18 | Right to Restriction | 2b | `restriction_manager.py` |
| 19 | Notification Obligation | 2b | `data_subject_rights.py` |
| 20 | Right to Portability | 2b | `portability_manager.py` |
| 21 | Right to Object | 2b | `objection_handler.py` |
| 22 | Automated Decisions | 2b | `automated_decisions.py` |
| **23** | **Restrictions** | 2b | `restrictions.py` |
| **24** | **Controller Accountability** | 3 | `accountability.py` |
| 25 | Privacy by Design | 4 | `privacy_by_design.py` |
| 26 | Joint Controllers | 0 | `joint_controller.py` |
| **27** | **EU Representative** | 0 | `definitions.py` |
| 28 | Processor | 0 | `processor_management.py` |
| 30 | ROPA | 3 | `ropa.py` |
| 31 | SA Cooperation | 3 | `sa_cooperation.py` |
| 32 | Security | 4 | `privacy_by_design.py` |
| 33 | Breach Notification SA | 5 | `breach_notification.py` |
| 34 | Breach Notification DS | 5 | `breach_notification.py` |
| 35 | DPIA | 6 | `dpia.py` |
| 36 | Prior Consultation | 6 | `prior_consultation.py` |
| 37-39 | DPO | 6 | `dpo_interface.py` |
| 44-49 | International Transfers | 6 | `international_transfers.py`, `uk_adequacy_contingency.py` |
| 77-84 | Remedies & Liability | 6 | `liability_framework.py` |

> **Note**: Articles in **bold** were added in v1.4 audit. Total article coverage: 40+ articles.

---

*This plan provides a comprehensive roadmap for GDPR compliance integration. Each phase is designed to be implementable in a single development session with complete test coverage. Regular review against EDPB guidelines is recommended.*
