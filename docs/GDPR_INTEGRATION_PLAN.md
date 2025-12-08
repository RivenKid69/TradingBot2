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

| Article | Description | Priority |
|---------|-------------|----------|
| **5-6** | Processing Principles & Lawful Basis | Critical |
| **7** | Consent Management | High |
| **12-22** | Data Subject Rights (DSAR) | Critical |
| **25** | Privacy by Design & Default | High |
| **30** | Records of Processing Activities (ROPA) | Critical |
| **32** | Security of Processing | High |
| **33-34** | Data Breach Notification | Critical |
| **35** | Data Protection Impact Assessment | High |
| **37-39** | Data Protection Officer | Medium |
| **44-49** | International Data Transfers | High |

---

## Architecture Integration

### Directory Structure

```
services/
  gdpr/
    __init__.py                    # Module exports
    # Phase 1: Foundation
    config.py                      # GDPR configuration
    legal_basis.py                 # Article 6 lawful basis management
    processing_principles.py       # Article 5 principles enforcement

    # Phase 2: Data Subject Rights
    data_subject_rights.py         # Rights framework (Articles 12-22)
    consent_manager.py             # Article 7 consent management
    dsar_handler.py                # Data Subject Access Requests
    erasure_manager.py             # Right to Erasure (Article 17)
    portability_manager.py         # Data Portability (Article 20)

    # Phase 3: ROPA & Documentation
    ropa.py                        # Records of Processing Activities
    processing_registry.py         # Processing operations registry
    data_mapping.py                # Personal data mapping

    # Phase 4: Privacy Engineering
    privacy_by_design.py           # Article 25 PbD controls
    data_minimization.py           # Data minimization enforcement
    pseudonymization.py            # Pseudonymization utilities
    retention_manager.py           # Data retention policies

    # Phase 5: Breach Management
    breach_detection.py            # Breach detection mechanisms
    breach_notification.py         # Articles 33-34 notification
    breach_assessment.py           # Risk assessment for breaches
    incident_response.py           # GDPR-specific incident response

    # Phase 6: DPIA & Governance
    dpia.py                        # Data Protection Impact Assessment
    dpo_interface.py               # DPO tools and interface
    international_transfers.py     # Articles 44-49 transfers
    compliance_dashboard.py        # GDPR compliance overview

tests/
  gdpr/
    test_gdpr_phase1_foundation.py
    test_gdpr_phase2_data_subject_rights.py
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

## Phase 1: Foundation & Legal Framework

**Estimated Complexity**: Medium
**Dependencies**: None
**Test Coverage Target**: 100%

### 1.1 Objectives

Establish the core GDPR framework including:
- Processing principles enforcement (Article 5)
- Lawful basis management (Article 6)
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
└── test_integration/
    ├── test_audit_trail_integration
    └── test_secure_logging_integration
```

**Expected test count**: ~80-100 tests

---

## Phase 2: Data Subject Rights

**Estimated Complexity**: High
**Dependencies**: Phase 1
**Test Coverage Target**: 100%

### 2.1 Objectives

Implement comprehensive data subject rights management:
- Consent management (Article 7)
- Right of access / DSAR (Article 15)
- Right to rectification (Article 16)
- Right to erasure (Article 17)
- Right to restriction (Article 18)
- Right to data portability (Article 20)
- Right to object (Article 21)
- Automated decision-making rights (Article 22)

### 2.2 Components to Implement

#### 2.2.1 ConsentManager (consent_manager.py)

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

#### 2.2.2 DSARHandler (dsar_handler.py)

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

### 2.3 Implementation Requirements

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

### 2.4 Test Specifications

```
test_gdpr_phase2_data_subject_rights.py:
├── test_consent/
│   ├── test_consent_creation
│   ├── test_consent_granularity
│   ├── test_consent_withdrawal
│   ├── test_withdrawal_ease (must be as easy as granting)
│   ├── test_consent_evidence_storage
│   ├── test_consent_version_tracking
│   ├── test_consent_expiry
│   ├── test_double_opt_in
│   └── test_consent_audit_trail
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
└── test_objection/
    ├── test_objection_to_processing
    ├── test_direct_marketing_objection
    └── test_profiling_objection
```

**Expected test count**: ~120-150 tests

---

## Phase 3: Records of Processing Activities (ROPA)

**Estimated Complexity**: Medium
**Dependencies**: Phase 1, Phase 2
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
└── test_integration/
    ├── test_pii_detection_integration
    ├── test_secure_logging_integration
    └── test_mifid_retention_integration
```

**Expected test count**: ~100-120 tests

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
└── test_integration/
    ├── test_dora_incident_mapping
    ├── test_dora_reporting_alignment
    └── test_unified_incident_dashboard
```

**Expected test count**: ~100-120 tests

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

Adequacy decisions list (as of 2024):
- Andorra, Argentina, Canada (commercial), Faroe Islands
- Guernsey, Israel, Isle of Man, Japan, Jersey
- New Zealand, Republic of Korea, Switzerland, UK, Uruguay
- EU-US Data Privacy Framework

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

**Expected test count**: ~90-110 tests

---

## Implementation Timeline

| Phase | Description | Est. Tests | Dependencies |
|-------|-------------|------------|--------------|
| 1 | Foundation & Legal Framework | 80-100 | None |
| 2 | Data Subject Rights | 120-150 | Phase 1 |
| 3 | ROPA & Documentation | 80-100 | Phases 1, 2 |
| 4 | Privacy Engineering | 100-120 | Phases 1-3 |
| 5 | Breach Management | 100-120 | Phases 1, 2, 4 |
| 6 | DPIA & Governance | 90-110 | All previous |

**Total estimated tests**: ~570-700 tests

---

## Cross-Regulation Alignment

### GDPR ↔ MiFID II

| GDPR Requirement | MiFID II Alignment |
|------------------|-------------------|
| Retention limits | 5-7 year retention overrides GDPR minimization |
| Audit trail | Shared audit infrastructure |
| Record keeping | ROPA aligned with transaction records |

### GDPR ↔ DORA

| GDPR Requirement | DORA Alignment |
|------------------|----------------|
| Breach notification (72h) | Incident reporting (24h/72h) |
| Security measures | ICT Risk Management Framework |
| Third-party assessment | Third-party ICT risk management |
| DPO role | ICT governance structure |

### GDPR ↔ EU AI Act

| GDPR Requirement | EU AI Act Alignment |
|------------------|---------------------|
| Data governance | Article 10 data governance |
| DPIA | Conformity assessment |
| Transparency | Article 13 transparency |
| Automated decisions | Article 14 human oversight |

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
    trading_records_years: 7  # MiFID II override
    audit_logs_years: 7
    consent_records_years: 3
    dsar_records_years: 3

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

## References

### Official Sources
- [GDPR Full Text](https://gdpr-info.eu/)
- [EDPB Guidelines](https://www.edpb.europa.eu/our-work-tools/general-guidance/guidelines-recommendations-best-practices_en)
- [Article 30 Guidance - DPC Ireland](https://www.dataprotection.ie/en/dpc-guidance/records-of-processing-article-30-guidance)
- [DPIA Guidance - European Commission](https://commission.europa.eu/law/law-topic/data-protection/rules-business-and-organisations/obligations/when-data-protection-impact-assessment-dpia-required_en)

### Implementation Guides
- [GDPR Compliance Checklist - Bitsight](https://www.bitsight.com/learn/gdpr-compliance-checklist)
- [DSAR Implementation - Securiti](https://securiti.ai/blog/dsar-rights-and-compliance/)
- [Breach Notification Guidelines - EDPB](https://www.edpb.europa.eu/system/files/2023-04/edpb_guidelines_202209_personal_data_breach_notification_v2.0_en.pdf)

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Dec 2024 | AI-Generated | Initial plan |

---

*This plan provides a comprehensive roadmap for GDPR compliance integration. Each phase is designed to be implementable in a single development session with complete test coverage.*
