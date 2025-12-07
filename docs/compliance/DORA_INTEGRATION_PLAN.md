# DORA Integration Plan
# Digital Operational Resilience Act (EU Regulation 2022/2554)
# План интеграции в AI-Powered Quantitative Research Platform

**Версия документа**: 1.0.0
**Дата создания**: 2025-12-08
**Целевое соответствие**: Regulation (EU) 2022/2554 (DORA)
**Дата вступления в силу**: 17 января 2025
**Статус проекта**: PLANNING

---

## Executive Summary

Digital Operational Resilience Act (DORA) — регулирование ЕС, устанавливающее единые требования к цифровой операционной устойчивости для финансового сектора. Данный план описывает интеграцию DORA в платформу алгоритмической торговли с учетом:

- Существующей архитектуры EU AI Act compliance (1007+ тестов)
- Существующей инфраструктуры MiFID II compliance (7 фаз)
- 5 ключевых направлений DORA
- Технических стандартов ESAs (RTS/ITS)

### Scope of Application

Платформа попадает под действие DORA как:
- **Инвестиционная фирма** (Article 2(1)(b)) — использование алгоритмической торговли
- **Пользователь ICT-сервисов третьих сторон** — интеграции с Binance, Alpaca, Polygon, OANDA, Interactive Brokers, Deribit

### Synergy with Existing Compliance

| Existing Module | DORA Reuse Potential | Gap Analysis |
|-----------------|---------------------|--------------|
| `services/ai_act/risk_management.py` | High — расширить для ICT risk | Добавить ICT-специфичные категории |
| `services/ai_act/post_market_monitoring.py` | High — incident tracking | Расширить для DORA incident classification |
| `services/ai_act/logging_system.py` | High — audit trail | Добавить ICT event logging |
| `services/ai_act/cybersecurity.py` | High — security measures | Расширить threat detection |
| `configs/compliance/mifid_compliance.yaml` | Medium — BCP, kill switch | Расширить для DORA requirements |
| `adapters/*` | Critical — third-party providers | Создать Register of Information |

---

## DORA Requirements Overview

### 5 Pillars of DORA

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DORA - 5 Key Pillars                                 │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────┤
│   PILLAR 1      │   PILLAR 2      │   PILLAR 3      │   PILLAR 4          │
│   ICT Risk      │   Incident      │   Digital       │   Third-Party       │
│   Management    │   Reporting     │   Resilience    │   Risk Management   │
│   (Art. 5-16)   │   (Art. 17-23)  │   Testing       │   (Art. 28-44)      │
│                 │                 │   (Art. 24-27)  │                     │
├─────────────────┴─────────────────┴─────────────────┴─────────────────────┤
│                           PILLAR 5                                         │
│              Information Sharing (Article 45)                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Implementation Plan

### Phase Overview

| Phase | Название | Articles | Тесты (est.) | Ключевые Deliverables |
|-------|----------|----------|--------------|----------------------|
| **Phase 1** | ICT Risk Management Framework | Art. 5-11 | ~250 | ICT Risk Framework, Governance, Detection/Response |
| **Phase 2** | ICT Incident Management & Reporting | Art. 17-23 | ~200 | Incident Classification, Reporting, Major Incident Handling |
| **Phase 3** | Digital Resilience Testing | Art. 24-27 | ~180 | TLPT Framework, Vulnerability Testing, Penetration Testing |
| **Phase 4** | Third-Party ICT Risk Management | Art. 28-44 | ~220 | Register of Information, Contractual Framework, Exit Strategies |
| **Phase 5** | Information Sharing & Integration | Art. 45 + Final | ~150 | Threat Intelligence, Cross-Regulation Integration |
| **TOTAL** | | | **~1000** | Full DORA Compliance |

---

# Phase 1: ICT Risk Management Framework
## Articles 5-16 Implementation

**Приоритет**: P0 (Critical Path)
**Зависимости**: Существующий `services/ai_act/risk_management.py`

### 1.1 Governance and Control Framework (Article 5)

#### 1.1.1 Management Body Responsibilities

**Требования Article 5(2)**:
- Ultimate responsibility for ICT risk management lies with management body
- Define, approve, oversee implementation of ICT risk management framework
- Approve digital operational resilience strategy

**Реализация**:

**Файл**: `services/dora/governance.py`

```
DORAGovernanceFramework:
├── ManagementBodyOversight
│   ├── approve_ict_risk_framework()
│   ├── review_digital_resilience_strategy()
│   ├── approve_ict_budget()
│   └── oversee_arrangements_with_ict_providers()
├── RoleAssignment
│   ├── ICTRiskOfficer (control function)
│   ├── segregation_of_duties()
│   └── three_lines_of_defence()
├── TrainingRequirements
│   ├── mandatory_ict_training()
│   ├── track_training_completion()
│   └── assess_knowledge_skills()
└── AuditIntegration
    ├── internal_audit_schedule()
    └── audit_findings_tracking()
```

**Ключевые аспекты для реализации**:
- [ ] Создать структуру для управленческого контроля ICT рисков
- [ ] Интеграция с существующим `QualityManagementSystem`
- [ ] Определить роли и ответственность по DORA
- [ ] Программа обязательного обучения ICT рискам (Article 5(4))

#### 1.1.2 Digital Operational Resilience Strategy

**Требования Article 6(8)**:
- Стратегия должна описывать реализацию ICT risk management framework
- Устанавливать risk tolerance level
- Определять clear information security objectives
- Включать KPI и KRM (Key Risk Metrics)

**Файл**: `config/dora/digital_resilience_strategy.yaml`

**Элементы стратегии** (per Article 6(8)):
1. Risk tolerance level для ICT risk
2. Information security objectives с KPIs
3. ICT reference architecture explanation
4. Dependencies on ICT third-party providers
5. Full ICT business continuity policy
6. Communication plan for ICT-related incidents

### 1.2 ICT Risk Management Framework (Article 6)

#### 1.2.1 Framework Structure

**Требования Article 6(1-7)**:
- Documented framework with strategies, policies, procedures
- Minimize impact of ICT risk
- Subject to internal audit

**Интеграция с существующим кодом**:

```python
# Расширение services/ai_act/risk_management.py

class DORAICTRiskCategory(Enum):
    """ICT-specific risk categories per DORA."""
    ICT_AVAILABILITY = "ict_availability"
    ICT_AUTHENTICITY = "ict_authenticity"
    ICT_INTEGRITY = "ict_integrity"
    ICT_CONFIDENTIALITY = "ict_confidentiality"
    ICT_THIRD_PARTY = "ict_third_party"
    ICT_CONCENTRATION = "ict_concentration"
    ICT_INFRASTRUCTURE = "ict_infrastructure"
    ICT_CHANGE_MANAGEMENT = "ict_change_management"
    ICT_DISASTER_RECOVERY = "ict_disaster_recovery"
    CYBER_THREAT = "cyber_threat"

class DORAICTRiskFramework:
    """
    DORA Article 6 compliant ICT Risk Management Framework.

    Extends AIActRiskManager with ICT-specific requirements.
    """
    def __init__(self, ai_act_risk_manager: AIActRiskManager):
        self.ai_act_rm = ai_act_risk_manager
        self.ict_risk_registry = DORAICTRiskRegistry()
        self.control_framework = ControlFramework()
```

#### 1.2.2 ICT Systems, Protocols and Tools (Article 7)

**Требования**:
- Use and maintain updated ICT systems
- Design ICT systems ensuring resilience, continuity, availability
- Proportionate to business needs

**Файл**: `services/dora/ict_systems.py`

**Ключевые компоненты**:
```
ICTSystemsManagement:
├── ICTAssetInventory
│   ├── register_all_ict_assets()
│   ├── classify_criticality()
│   ├── document_dependencies()
│   └── version_tracking()
├── ICTSecurityPolicy
│   ├── network_security_rules()
│   ├── access_control_policies()
│   ├── encryption_requirements()
│   └── patch_management()
├── ICTCapacityManagement
│   ├── performance_monitoring()
│   ├── capacity_planning()
│   └── scalability_provisions()
└── ICTChangeManagement
    ├── change_approval_workflow()
    ├── testing_before_deployment()
    └── rollback_procedures()
```

### 1.3 Identification (Article 8)

**Требования Article 8(1-6)**:
- Identify all sources of ICT risk
- Identify all ICT-supported business functions
- Identify all assets and their criticality
- Map ICT assets to business functions
- Assess cyber threats and vulnerabilities

**Файл**: `services/dora/ict_identification.py`

**Компоненты идентификации**:

| Component | Description | Implementation |
|-----------|-------------|----------------|
| **ICT Asset Register** | All ICT assets with criticality | `ICTAssetRegister` class |
| **Business Function Mapping** | ICT → Business function links | `BusinessFunctionMapper` |
| **ICT Provider Dependencies** | Third-party provider mapping | `ICTProviderDependencyMap` |
| **Vulnerability Assessment** | Regular vulnerability scanning | `VulnerabilityScanner` |
| **Threat Landscape** | Current threats identification | `ThreatLandscapeAnalyzer` |

**Интеграция с адаптерами** (критично):

```python
# Mapping existing adapters to ICT assets
ICT_THIRD_PARTY_PROVIDERS = {
    "binance": {
        "provider_type": "exchange",
        "services": ["market_data", "order_execution", "account_management"],
        "criticality": "CRITICAL",
        "adapter_path": "adapters/binance/",
        "contracts": ["spot", "futures"],
    },
    "alpaca": {
        "provider_type": "broker",
        "services": ["market_data", "order_execution"],
        "criticality": "CRITICAL",
        "adapter_path": "adapters/alpaca/",
    },
    # ... all other adapters
}
```

### 1.4 Protection and Prevention (Article 9)

**Требования**:
- Continuous monitoring and control of ICT systems
- Implementation of ICT security policies
- Mechanisms against intrusion and data misuse

**Расширение**: `services/ai_act/cybersecurity.py`

**Новые компоненты для DORA**:

```
DORAProtectionMeasures:
├── ContinuousMonitoring (extends existing monitoring.py)
│   ├── network_traffic_monitoring()
│   ├── security_event_monitoring()
│   ├── anomaly_detection()
│   └── real_time_alerting()
├── ICTSecurityPolicies
│   ├── access_control_implementation()
│   ├── strong_authentication()
│   ├── data_classification()
│   └── encryption_in_transit_at_rest()
├── PatchManagement
│   ├── vulnerability_tracking()
│   ├── patch_prioritization()
│   ├── deployment_automation()
│   └── verification_testing()
└── NetworkSecurity
    ├── segmentation()
    ├── firewall_rules()
    ├── intrusion_detection()
    └── secure_configuration()
```

### 1.5 Detection (Article 10)

**Требования**:
- Mechanisms to detect anomalous activities
- Multiple layers of control
- Detection of single points of failure

**Файл**: `services/dora/detection.py`

**Ключевые детекторы**:

| Detector | Function | Integration |
|----------|----------|-------------|
| `AnomalyDetector` | Unusual ICT activity | Extend `services/ai_act/human_oversight.py` |
| `PerformanceDegradationDetector` | Performance issues | Extend `services/monitoring.py` |
| `SecurityIncidentDetector` | Security breaches | New component |
| `SinglePointOfFailureDetector` | SPOF identification | New component |
| `ICTIncidentDetector` | ICT incidents | Extend `post_market_monitoring.py` |

### 1.6 Response and Recovery (Article 11)

**Требования**:
- ICT business continuity policy
- ICT response and recovery plans
- Crisis communication plans
- Testing of plans at least yearly

**Интеграция с MiFID II BCP**:

Существующий `configs/compliance/mifid_compliance.yaml` уже содержит:
- Business continuity configuration
- RTO/RPO targets
- BCP scenarios
- Drill requirements

**Расширения для DORA**:

**Файл**: `services/dora/response_recovery.py`

```python
class DORAResponseRecovery:
    """
    Article 11 compliant response and recovery system.

    Extends MiFID II BCP with DORA-specific requirements.
    """

    # Article 11(3) - Dedicated response/recovery plans for each ICT scenario
    ICT_SCENARIOS = [
        "cyber_attack",
        "system_failure",
        "data_corruption",
        "third_party_failure",
        "infrastructure_failure",
        "natural_disaster",
        "power_outage",
        "communication_failure",
    ]

    # Article 11(4) - Crisis management function
    def activate_crisis_management(self, incident: ICTIncident):
        """Activate crisis management with clear procedures."""

    # Article 11(5) - Regular testing yearly
    def execute_annual_bcp_test(self):
        """Execute comprehensive BCP testing."""

    # Article 11(6) - Cyber attack scenarios
    def test_cyber_attack_scenario(self, scenario: CyberAttackScenario):
        """Test response to simulated cyber attacks."""
```

### 1.7 Backup Policies and Recovery (Article 12)

**Требования**:
- Backup policies and procedures
- Restoration and recovery methods
- Backup systems physically and logically segregated
- Redundant ICT capacities

**Интеграция**:

Существующий `services/state_storage.py` и `configs/state.yaml`:
- Snapshot interval
- Backup retention
- Atomic writes

**Расширения для DORA**:

**Файл**: `services/dora/backup_recovery.py`

```
DORABackupSystem:
├── BackupPolicyEngine
│   ├── define_backup_scope()        # Article 12(1)(a)
│   ├── set_backup_frequency()       # Based on data criticality
│   ├── manage_retention()           # Compliance with retention requirements
│   └── verify_backup_integrity()
├── RestorationProcedures
│   ├── define_recovery_procedures() # Article 12(1)(b)
│   ├── segregated_recovery_systems() # Article 12(3)
│   ├── recovery_time_objectives()
│   └── recovery_point_objectives()
├── RedundantCapacities
│   ├── infrastructure_redundancy()  # Article 12(5)
│   ├── geographic_separation()
│   ├── failover_automation()
│   └── capacity_testing()
└── DataIntegrityVerification
    ├── post_recovery_checks()       # Article 12(7)
    ├── reconciliation_procedures()
    └── consistency_validation()
```

### 1.8 Learning and Evolving (Article 13)

**Требования**:
- Gather information on vulnerabilities and threats
- Assess impact of severe disruption
- Post-incident reviews
- Regular testing and updating

**Файл**: `services/dora/learning.py`

```python
class DORALearningSystem:
    """
    Article 13 - Continuous learning and evolution.
    """

    def collect_threat_intelligence(self):
        """Gather information on vulnerabilities and cyber threats."""

    def conduct_post_incident_review(self, incident: DORAIncident):
        """
        Post-incident review per Article 13(2).

        Analyze:
        - Root cause
        - Effectiveness of response
        - Lessons learned
        - Process improvements
        """

    def update_risk_assessment(self, new_threats: List[Threat]):
        """Update ICT risk assessment based on new information."""

    def incorporate_testing_results(self, test_results: TestResults):
        """Incorporate results from resilience testing."""
```

### 1.9 Communication (Article 14)

**Требования**:
- Crisis communication plans
- Internal and external communication procedures
- Disclosure obligations for ICT incidents

**Файл**: `services/dora/communication.py`

```
DORACommuncationPlan:
├── InternalCommunication
│   ├── escalation_procedures()
│   ├── staff_notification()
│   └── management_reporting()
├── ExternalCommunication
│   ├── client_notification()
│   ├── counterparty_notification()
│   └── service_provider_notification()
├── RegulatoryReporting
│   ├── competent_authority_notification()
│   ├── incident_reporting()
│   └── annual_reporting()
└── PublicDisclosure
    ├── public_statement_procedures()
    └── media_handling()
```

### Phase 1 Deliverables Summary

| Deliverable | File/Path | Tests |
|-------------|-----------|-------|
| DORA Governance Framework | `services/dora/governance.py` | ~40 |
| Digital Resilience Strategy Config | `config/dora/digital_resilience_strategy.yaml` | ~10 |
| ICT Risk Management Framework | `services/dora/ict_risk_framework.py` | ~50 |
| ICT Systems Management | `services/dora/ict_systems.py` | ~30 |
| ICT Identification | `services/dora/ict_identification.py` | ~30 |
| Detection System | `services/dora/detection.py` | ~25 |
| Response and Recovery | `services/dora/response_recovery.py` | ~30 |
| Backup System | `services/dora/backup_recovery.py` | ~20 |
| Learning System | `services/dora/learning.py` | ~15 |
| Communication Plan | `services/dora/communication.py` | ~20 |
| **TOTAL** | | **~250** |

### Phase 1 Test Requirements

```bash
# New test files
tests/
├── dora/
│   ├── test_dora_governance.py
│   ├── test_dora_ict_risk_framework.py
│   ├── test_dora_ict_systems.py
│   ├── test_dora_identification.py
│   ├── test_dora_detection.py
│   ├── test_dora_response_recovery.py
│   ├── test_dora_backup.py
│   ├── test_dora_learning.py
│   └── test_dora_communication.py
```

**Критерии завершения Phase 1**:
- [ ] 100% тестовое покрытие всех новых модулей
- [ ] Интеграционные тесты с существующими AI Act модулями
- [ ] Документация обновлена
- [ ] Все 250+ тестов проходят

---

# Phase 2: ICT Incident Management & Reporting
## Articles 17-23 Implementation

**Приоритет**: P0 (Critical - Regulatory Reporting)
**Зависимости**: Phase 1, существующий `post_market_monitoring.py`

### 2.1 ICT-Related Incident Management Process (Article 17)

**Требования**:
- Process to detect, manage, and notify ICT-related incidents
- Early warning indicators
- Allocation of roles and responsibilities

**Расширение**: `services/ai_act/post_market_monitoring.py`

**Файл**: `services/dora/incident_management.py`

```python
class DORAIncidentManagement:
    """
    Article 17 compliant ICT incident management.

    Extends AI Act IncidentTracker with DORA-specific requirements.
    """

    # Incident management process
    INCIDENT_PHASES = [
        "DETECTION",      # Early warning, automated detection
        "RECORDING",      # Record and log incident
        "CLASSIFICATION", # Classify per Article 18 criteria
        "ESCALATION",     # Internal escalation
        "NOTIFICATION",   # Regulatory notification if major
        "INVESTIGATION",  # Root cause analysis
        "RESOLUTION",     # Corrective actions
        "CLOSURE",        # Post-incident review
    ]

    def detect_ict_incident(self, event: ICTEvent) -> Optional[DORAIncident]:
        """Early detection using indicators per Article 17(3)(a)."""

    def classify_incident(self, incident: DORAIncident) -> IncidentClassification:
        """Classify using Article 18 criteria."""

    def initiate_notification_workflow(self, incident: DORAIncident):
        """Initiate reporting workflow for major incidents."""
```

### 2.2 Classification of ICT-Related Incidents (Article 18)

**Критерии классификации** (Commission Delegated Regulation 2024/1772):

| Criterion | Description | Threshold for Major |
|-----------|-------------|---------------------|
| **Clients affected** | Number/relevance of affected clients | Substantial impact |
| **Duration** | Duration of incident | Material duration |
| **Geographic spread** | Geographic scope | Multiple jurisdictions |
| **Data losses** | Data integrity/confidentiality | Any data loss |
| **Criticality of services** | Critical services affected | Critical services impacted |
| **Economic impact** | Direct/indirect costs | Material financial impact |
| **Reputational impact** | Potential reputational damage | Significant damage |

**Файл**: `services/dora/incident_classification.py`

```python
class DORAIncidentClassification:
    """
    Article 18 compliant incident classification.

    Uses RTS criteria from Commission Delegated Regulation 2024/1772.
    """

    @dataclass
    class ClassificationCriteria:
        clients_affected: int
        client_relevance: str  # retail, professional, counterparty
        duration_hours: float
        geographic_spread: List[str]  # country codes
        data_losses: bool
        data_type: Optional[str]  # personal, confidential, public
        critical_services_affected: bool
        critical_service_names: List[str]
        economic_impact_eur: float
        reputational_impact: str  # low, medium, high

    def classify_incident(
        self,
        incident: DORAIncident,
        criteria: ClassificationCriteria
    ) -> str:  # "MAJOR" or "SIGNIFICANT" or "MINOR"
        """
        Classify incident using Article 18 criteria.

        Major Incident Conditions (any 2 or more):
        1. Critical services affected + malicious access = MAJOR
        2. Material impact on 2+ criteria = MAJOR
        3. Recurring incidents (3+ in 3 months, same cause) = MAJOR
        """
```

**Порогі для Major Incidents** (per RTS):

```yaml
# config/dora/incident_thresholds.yaml
incident_classification:
  major_incident_thresholds:
    # If ANY of these are met, classify as MAJOR
    critical_service_breach:
      affected: true
      malicious_access: true

    # If 2+ of these criteria have material impact
    criteria_thresholds:
      clients_affected:
        retail_count: 5000
        professional_count: 100
        counterparty_count: 10

      duration:
        hours: 4

      geographic_spread:
        countries: 2

      data_losses:
        any: true

      economic_impact:
        eur: 100000

      reputational:
        level: "high"

  # Recurring incidents threshold
  recurring_incident:
    count: 3
    period_months: 3
    same_cause: true
```

### 2.3 Reporting of Major ICT-Related Incidents (Article 19)

**Временные рамки репортинга** (per RTS):

| Report Type | Deadline | Content |
|-------------|----------|---------|
| **Initial Notification** | 4 hours after classification, 24 hours after detection | Basic incident info |
| **Intermediate Report** | 72 hours after initial | Detailed analysis, status |
| **Final Report** | 1 month after incident | Root cause, lessons learned |

**Файл**: `services/dora/incident_reporting.py`

```python
class DORAIncidentReporter:
    """
    Article 19 compliant incident reporting to competent authorities.

    Implements RTS/ITS reporting requirements.
    """

    # Report templates per ITS (JC 2024-33)
    REPORT_TEMPLATES = {
        "initial": "ITS_INITIAL_NOTIFICATION",
        "intermediate": "ITS_INTERMEDIATE_REPORT",
        "final": "ITS_FINAL_REPORT",
    }

    # Deadlines in hours
    DEADLINES = {
        "initial_from_classification": 4,
        "initial_from_detection": 24,
        "intermediate": 72,
        "final_days": 30,
    }

    def generate_initial_notification(
        self,
        incident: DORAIncident,
        classification: IncidentClassification
    ) -> InitialNotification:
        """
        Generate initial notification per ITS Annex I.

        Limited fields to avoid burden during active incident:
        - Incident reference
        - Date/time of detection
        - Date/time of classification
        - Member state of occurrence
        - Brief description
        - Services affected
        - Estimated impact
        """

    def generate_intermediate_report(
        self,
        incident: DORAIncident
    ) -> IntermediateReport:
        """
        Generate intermediate report per ITS Annex II.

        Additional fields:
        - Detailed incident description
        - Root cause analysis (preliminary)
        - Actions taken
        - Ongoing activities
        - Updated impact assessment
        """

    def generate_final_report(
        self,
        incident: DORAIncident
    ) -> FinalReport:
        """
        Generate final report per ITS Annex III.

        Complete information:
        - Full root cause analysis
        - Timeline of events
        - Effectiveness of response
        - Lessons learned
        - Remediation measures
        - Prevention measures
        """

    def submit_to_competent_authority(
        self,
        report: Union[InitialNotification, IntermediateReport, FinalReport],
        authority: CompetentAuthority
    ):
        """Submit report to NCA."""
```

### 2.4 Notification of Significant Cyber Threats (Article 19(4))

**Требования**:
- Voluntary notification of significant cyber threats
- Threats relevant to financial system, users, or clients

**Файл**: `services/dora/cyber_threat_notification.py`

```python
class CyberThreatNotification:
    """
    Article 19(4) - Voluntary cyber threat notification.

    Financial entities may notify significant cyber threats
    when deemed relevant to financial system.
    """

    def assess_threat_significance(self, threat: CyberThreat) -> bool:
        """
        Assess if threat is significant per DORA criteria.

        Significant if:
        - Impact on critical/important business functions
        - Impact on other financial institutions
        - Impact on third parties or clients
        """

    def notify_significant_threat(
        self,
        threat: CyberThreat,
        assessment: ThreatAssessment
    ):
        """Submit voluntary threat notification to NCA."""
```

### 2.5 Harmonised Reporting Templates (Article 20)

**ESAs Technical Standards** (JC 2024-33):
- ITS on standard forms, templates, and procedures
- RTS on content of reports

**Файл**: `services/dora/reporting_templates.py`

```python
# Implementation of ITS Annex templates

@dataclass
class ITSInitialNotificationTemplate:
    """ITS Annex I - Initial Notification Template."""
    # Mandatory fields
    incident_reference: str
    reporting_entity_lei: str
    detection_datetime: datetime
    classification_datetime: datetime
    member_states_affected: List[str]
    incident_type: str  # cyber_attack, system_failure, etc.
    brief_description: str
    critical_services_affected: List[str]
    estimated_impact: str
    is_recurring: bool

@dataclass
class ITSIntermediateReportTemplate:
    """ITS Annex II - Intermediate Report Template."""
    # All initial fields plus:
    detailed_description: str
    affected_ict_services: List[str]
    affected_clients_count: int
    geographic_spread: List[str]
    data_compromised: bool
    preliminary_root_cause: str
    actions_taken: List[str]
    ongoing_actions: List[str]
    external_support: bool
    estimated_resolution_time: Optional[datetime]

@dataclass
class ITSFinalReportTemplate:
    """ITS Annex III - Final Report Template."""
    # All intermediate fields plus:
    incident_resolved: bool
    resolution_datetime: datetime
    final_root_cause: str
    root_cause_category: str
    full_timeline: List[TimelineEvent]
    total_duration_hours: float
    total_clients_affected: int
    total_economic_impact_eur: float
    data_loss_details: Optional[str]
    response_effectiveness: str
    lessons_learned: List[str]
    remediation_measures: List[str]
    preventive_measures: List[str]
    follow_up_actions: List[str]
```

### 2.6 Centralised Reporting Hub (Article 21)

**Примечание**: ESAs могут создать централизованный EU hub для репортинга. Пока используется direct NCA reporting.

### 2.7 Supervisory Feedback (Article 22)

**Файл**: `services/dora/supervisory_feedback.py`

```python
class SupervisoryFeedbackHandler:
    """
    Handle feedback from competent authorities on incident reports.
    """

    def receive_feedback(self, feedback: SupervisoryFeedback):
        """Process feedback from NCA."""

    def implement_guidance(self, guidance: NCAguidance):
        """Implement NCA guidance on incident handling."""
```

### 2.8 Operational or Security Incidents at Third-Party Providers (Article 23)

**Файл**: `services/dora/third_party_incidents.py`

```python
class ThirdPartyIncidentHandler:
    """
    Article 23 - Handle incidents at ICT third-party providers.

    Critical for our platform as we depend on multiple exchanges.
    """

    def register_provider_incident(
        self,
        provider: str,  # binance, alpaca, etc.
        incident: ProviderIncident
    ):
        """Record incident at third-party provider."""

    def assess_impact_on_operations(
        self,
        provider: str,
        incident: ProviderIncident
    ) -> ImpactAssessment:
        """Assess how provider incident affects our operations."""

    def activate_contingency(
        self,
        provider: str,
        contingency_plan: ContingencyPlan
    ):
        """Activate contingency plan for provider failure."""
```

### Phase 2 Deliverables Summary

| Deliverable | File/Path | Tests |
|-------------|-----------|-------|
| ICT Incident Management | `services/dora/incident_management.py` | ~50 |
| Incident Classification | `services/dora/incident_classification.py` | ~40 |
| Incident Reporting | `services/dora/incident_reporting.py` | ~40 |
| Cyber Threat Notification | `services/dora/cyber_threat_notification.py` | ~20 |
| Reporting Templates (ITS) | `services/dora/reporting_templates.py` | ~30 |
| Supervisory Feedback | `services/dora/supervisory_feedback.py` | ~10 |
| Third-Party Incidents | `services/dora/third_party_incidents.py` | ~20 |
| **TOTAL** | | **~200** |

### Phase 2 Test Requirements

```bash
tests/dora/
├── test_dora_incident_management.py
├── test_dora_incident_classification.py
├── test_dora_incident_reporting.py
├── test_dora_cyber_threat_notification.py
├── test_dora_reporting_templates.py
├── test_dora_supervisory_feedback.py
└── test_dora_third_party_incidents.py
```

**Критерии завершения Phase 2**:
- [ ] All incident classification scenarios covered
- [ ] ITS templates fully implemented
- [ ] Integration with existing IncidentTracker
- [ ] Mock NCA submission workflow tested
- [ ] All 200+ тестов проходят

---

# Phase 3: Digital Resilience Testing
## Articles 24-27 Implementation

**Приоритет**: P1 (High)
**Зависимости**: Phase 1, Phase 2

### 3.1 General Requirements for Testing (Article 24)

**Требования**:
- Sound and comprehensive digital operational resilience testing programme
- Range of assessments, tests, methodologies
- Proportionate to size and risk profile

**Файл**: `services/dora/resilience_testing.py`

```python
class DORAResilienceTestingProgramme:
    """
    Article 24 compliant digital operational resilience testing.

    Testing programme includes:
    1. Vulnerability assessments and scans
    2. Open source analyses
    3. Network security assessments
    4. Gap analyses
    5. Physical security reviews
    6. Questionnaires and scanning software
    7. Source code reviews (where feasible)
    8. Scenario-based tests
    9. Compatibility testing
    10. Performance testing
    11. End-to-end testing
    12. Penetration testing
    """

    TEST_CATEGORIES = {
        "vulnerability_assessment": {
            "frequency": "quarterly",
            "scope": "all_systems",
            "article": "24(1)(a)",
        },
        "network_security": {
            "frequency": "quarterly",
            "scope": "network_infrastructure",
            "article": "24(1)(a)",
        },
        "penetration_testing": {
            "frequency": "yearly",
            "scope": "critical_systems",
            "article": "24(1)(a)",
        },
        "scenario_based": {
            "frequency": "yearly",
            "scope": "business_continuity",
            "article": "24(1)(a)",
        },
        "source_code_review": {
            "frequency": "per_release",
            "scope": "critical_code",
            "article": "24(1)(a)",
        },
    }

    def create_testing_programme(
        self,
        entity_profile: EntityProfile
    ) -> TestingProgramme:
        """Create risk-based testing programme."""

    def execute_testing_cycle(
        self,
        programme: TestingProgramme,
        cycle: str  # quarterly, yearly
    ) -> TestingReport:
        """Execute scheduled testing cycle."""
```

### 3.2 Testing of ICT Tools and Systems (Article 25)

**Требования**:
- Apply testing programme to all ICT systems
- Risk-based approach
- Independent parties for testing

**Файл**: `services/dora/ict_testing.py`

```python
class ICTSystemTesting:
    """
    Article 25 - Testing of all critical ICT systems.
    """

    def identify_systems_for_testing(self) -> List[ICTSystem]:
        """
        Identify all ICT systems requiring testing.

        Priority based on:
        - Criticality of business function
        - Risk profile
        - Recent changes
        """

    def execute_vulnerability_scan(
        self,
        system: ICTSystem
    ) -> VulnerabilityScanResult:
        """Execute automated vulnerability scanning."""

    def execute_penetration_test(
        self,
        system: ICTSystem,
        scope: PentestScope
    ) -> PenetrationTestResult:
        """Execute penetration testing (yearly minimum)."""

    def validate_third_party_interfaces(
        self,
        provider: str
    ) -> InterfaceTestResult:
        """Test interfaces with third-party ICT providers."""
```

### 3.3 Threat-Led Penetration Testing (TLPT) (Article 26)

**Требования**:
- Advanced testing mimicking real threat actors
- At least every 3 years for significant entities
- Live production systems testing
- Based on TIBER-EU framework

**Применимость к нашей платформе**:
- Проверить с NCA, требуется ли TLPT (зависит от размера/системности)
- Для алгоритмической торговли вероятно требуется

**Файл**: `services/dora/tlpt.py`

```python
class ThreatLedPenetrationTesting:
    """
    Article 26 - TLPT (Threat-Led Penetration Testing).

    Based on TIBER-EU framework.

    TLPT Scope per Article 26:
    1. Cover critical or important functions
    2. Performed on live production systems
    3. Every 3 years (or more if NCA requires)
    4. Must include ICT third-party providers (Article 26(4))
    """

    TLPT_PHASES = {
        "preparation": {
            "activities": [
                "scope_definition",
                "threat_intelligence_gathering",
                "scenario_development",
            ],
            "duration_weeks": 4,
        },
        "threat_intelligence": {
            "activities": [
                "targeted_threat_intelligence",
                "scenario_refinement",
                "attack_plan_development",
            ],
            "duration_weeks": 4,
        },
        "red_team_testing": {
            "activities": [
                "attack_simulation",
                "exploitation_attempts",
                "lateral_movement",
                "objective_achievement",
            ],
            "duration_weeks": 8,
        },
        "closure": {
            "activities": [
                "purple_teaming",  # Required by DORA
                "reporting",
                "remediation_planning",
            ],
            "duration_weeks": 4,
        },
    }

    def plan_tlpt_engagement(
        self,
        scope: TLPTScope,
        threat_intelligence: ThreatIntelligence
    ) -> TLPTEngagementPlan:
        """Plan TLPT engagement per TIBER-EU."""

    def validate_testers(
        self,
        testers: List[TLPTTester]
    ) -> TesterValidationResult:
        """
        Validate testers meet Article 27 requirements:
        - Highest suitability and reputability
        - Technical and organisational capabilities
        - Expertise in threat intelligence
        - Expertise in penetration testing
        - Expertise in red team testing
        - Certified by accreditation body
        """

    def conduct_purple_teaming(
        self,
        red_team_results: RedTeamResults,
        blue_team: BlueTeam
    ) -> PurpleTeamReport:
        """
        Mandatory purple teaming per DORA Article 26(5).

        Blue team (defenders) work with red team to:
        - Review attack techniques
        - Understand vulnerabilities exploited
        - Improve detection capabilities
        - Strengthen defenses
        """

    def generate_tlpt_report(
        self,
        engagement: TLPTEngagement
    ) -> TLPTReport:
        """Generate comprehensive TLPT report."""

    def submit_tlpt_attestation(
        self,
        report: TLPTReport,
        authority: CompetentAuthority
    ):
        """Submit TLPT attestation to NCA per Article 26(6)."""
```

### 3.4 Requirements for Testers (Article 27)

**Требования к тестировщикам TLPT**:

| Requirement | Description |
|-------------|-------------|
| **Suitability** | Highest suitability and reputability |
| **Capabilities** | Technical and organisational |
| **Expertise** | Threat intelligence, penetration testing, red team |
| **Certification** | Certified by accreditation body (CREST, OSCP, etc.) |
| **Insurance** | Professional indemnity insurance |
| **Independence** | No conflicts of interest |

**Internal vs External Testers**:
- Internal: Allowed 2 out of 3 TLPTs
- External: Required for 1 out of 3 TLPTs
- Threat intelligence: Always from external party

**Файл**: `services/dora/tester_management.py`

```python
class TLPTTesterManagement:
    """
    Article 27 - Management of TLPT testers.
    """

    REQUIRED_CERTIFICATIONS = [
        "CREST_CRT",
        "CREST_CCT",
        "OSCP",
        "OSCE",
        "GPEN",
        "GWAPT",
    ]

    def validate_tester_qualifications(
        self,
        tester: TLPTTester
    ) -> QualificationValidation:
        """Validate tester meets Article 27 requirements."""

    def check_conflict_of_interest(
        self,
        tester: TLPTTester
    ) -> ConflictCheck:
        """Check for conflicts of interest."""

    def verify_internal_tester_conditions(
        self,
        internal_tester: InternalTester
    ) -> InternalTesterApproval:
        """
        Verify conditions for using internal testers:
        - Approved by NCA
        - No conflict of interest
        - External threat intelligence provider
        """
```

### 3.5 Pooled Testing (Article 26(3))

**Для third-party providers** (важно для наших адаптеров):

```python
class PooledTLPT:
    """
    Article 26(3) - Pooled TLPT for shared ICT services.

    Allows multiple financial entities to jointly test
    shared third-party ICT service providers.
    """

    def organize_pooled_tlpt(
        self,
        participants: List[FinancialEntity],
        shared_provider: ICTProvider
    ) -> PooledTLPTEngagement:
        """Organize pooled TLPT for shared provider."""
```

### Phase 3 Deliverables Summary

| Deliverable | File/Path | Tests |
|-------------|-----------|-------|
| Resilience Testing Programme | `services/dora/resilience_testing.py` | ~50 |
| ICT System Testing | `services/dora/ict_testing.py` | ~40 |
| TLPT Framework | `services/dora/tlpt.py` | ~50 |
| Tester Management | `services/dora/tester_management.py` | ~20 |
| Pooled Testing | `services/dora/pooled_testing.py` | ~20 |
| **TOTAL** | | **~180** |

### Phase 3 Test Requirements

```bash
tests/dora/
├── test_dora_resilience_testing.py
├── test_dora_ict_testing.py
├── test_dora_tlpt.py
├── test_dora_tester_management.py
└── test_dora_pooled_testing.py
```

**Критерии завершения Phase 3**:
- [ ] Testing programme fully documented
- [ ] Vulnerability scanning integrated
- [ ] TLPT framework ready (engagement planning)
- [ ] Tester qualification checks implemented
- [ ] All 180+ тестов проходят

---

# Phase 4: Third-Party ICT Risk Management
## Articles 28-44 Implementation

**Приоритет**: P0 (Critical - Core Platform Dependency)
**Зависимости**: Phase 1

### 4.1 General Principles (Article 28)

**Требования**:
- Manage ICT third-party risk as integral part of ICT risk framework
- Full responsibility remains with financial entity
- Proportionate approach based on nature and criticality

**Критическая важность для платформы**:
Наша платформа зависит от множества ICT third-party providers:

| Provider | Services | Criticality |
|----------|----------|-------------|
| Binance | Market data, Order execution | **CRITICAL** |
| Alpaca | Market data, Order execution | **CRITICAL** |
| Polygon.io | Market data | HIGH |
| OANDA | Forex trading | **CRITICAL** |
| Interactive Brokers | Futures trading | **CRITICAL** |
| Deribit | Crypto options | HIGH |
| Dukascopy | Forex data | MEDIUM |

**Файл**: `services/dora/third_party_risk.py`

```python
class DORAThirdPartyRiskManagement:
    """
    Article 28 - ICT Third-Party Risk Management.

    Manage risk from all ICT service providers.
    """

    def assess_third_party_risk(
        self,
        provider: ICTProvider
    ) -> ThirdPartyRiskAssessment:
        """
        Comprehensive risk assessment including:
        - Criticality of services
        - Concentration risk
        - Substitutability
        - Provider's resilience
        - Geographic location risks
        """

    def maintain_control(
        self,
        provider: ICTProvider
    ) -> ControlAssessment:
        """
        Ensure we maintain control per Article 28(1)(b):
        - Full responsibility remains with us
        - Adequate oversight capabilities
        - No impediment to supervision
        """
```

### 4.2 Register of Information (Article 28(3))

**Критическое требование DORA** - Реестр всех договорных отношений

**Файл**: `services/dora/register_of_information.py`

**ITS Templates структура** (per JC 2023 85):

```python
@dataclass
class RegisterOfInformationEntry:
    """
    Single entry in Register of Information per ITS.

    Template RT.02.01 - Contractual arrangement level.
    """
    # Contractual arrangement identification
    contractual_arrangement_ref: str
    lei_counterparty: str
    counterparty_name: str

    # Contract details
    contract_type: str  # outsourcing, procurement, intra_group
    start_date: date
    end_date: Optional[date]
    termination_notice_period_days: int

    # Services provided
    services_provided: List[str]
    functions_supported: List[str]
    is_supporting_critical_function: bool

    # Location information
    data_processing_locations: List[str]
    data_storage_locations: List[str]

    # Sub-contracting
    permits_subcontracting: bool
    subcontractors: List[str]

    # Audit rights
    audit_rights_granted: bool
    last_audit_date: Optional[date]

    # Exit strategy
    exit_strategy_documented: bool
    transition_plan_available: bool

class DORARegisterOfInformation:
    """
    Article 28(3) - Register of Information for ICT third-party providers.

    Must be maintained and updated at entity level.
    Report to competent authority yearly (by 30 April per ESA Decision).
    """

    def __init__(self):
        self.entries: Dict[str, RegisterOfInformationEntry] = {}
        self.last_submission_date: Optional[datetime] = None

    def register_provider(
        self,
        provider: ICTProvider,
        contract: ICTContract
    ) -> RegisterOfInformationEntry:
        """Register new ICT third-party provider."""

    def update_entry(
        self,
        entry_id: str,
        updates: Dict[str, Any]
    ):
        """Update existing entry."""

    def classify_criticality(
        self,
        entry_id: str
    ) -> str:  # CRITICAL, IMPORTANT, STANDARD
        """Classify criticality of ICT services."""

    def generate_annual_report(self) -> RegisterReport:
        """
        Generate annual report for NCA.

        Due: 30 April each year (ESA Decision).
        """

    def export_to_its_template(self) -> ITSRegisterTemplate:
        """Export in ITS-compliant format."""
```

**Реализация для наших адаптеров**:

```python
# config/dora/register_of_information.yaml
register_of_information:
  entities:
    - entity_lei: "YOUR_LEI_HERE"
      entity_name: "Your Company Name"

  contractual_arrangements:
    - ref_number: "CA-2025-001"
      provider:
        lei: "BINANCE_LEI"  # Or equivalent identifier
        name: "Binance Holdings Limited"
        country: "MT"  # Malta (European entity)
      contract:
        type: "ICT_SERVICE"
        start_date: "2024-01-01"
        end_date: null  # Ongoing
        notice_period_days: 30
      services:
        - code: "MARKET_DATA"
          description: "Real-time and historical market data"
          critical_function: true
        - code: "ORDER_EXECUTION"
          description: "Order placement and execution"
          critical_function: true
        - code: "ACCOUNT_MANAGEMENT"
          description: "Account balance and position queries"
          critical_function: true
      data:
        processing_locations: ["MT", "SG"]
        storage_locations: ["MT", "SG"]
      subcontracting:
        permitted: true
        subcontractors: []  # Unknown/not disclosed
      audit:
        rights_granted: false  # Per standard terms
        last_audit: null
      exit:
        strategy_documented: true
        transition_plan: true

    - ref_number: "CA-2025-002"
      provider:
        lei: "ALPACA_LEI"
        name: "Alpaca Securities LLC"
        country: "US"
      # ... similar structure
```

### 4.3 Contractual Arrangements (Articles 30)

**Требования Article 30(2)** (basic ICT services):
- Clear description of services
- Locations of data processing
- Service level descriptions
- Assistance in case of incidents
- Termination rights

**Требования Article 30(3)** (critical/important functions):
- All basic requirements plus:
- Full service level agreements (SLAs)
- Notice periods and reporting obligations
- Audit rights for entity and NCA
- Exit strategies and transition support
- Performance targets with remedial actions

**Файл**: `services/dora/contractual_requirements.py`

```python
class DORAContractualRequirements:
    """
    Article 30 - Contractual arrangements with ICT providers.
    """

    # Basic requirements per Article 30(2)
    BASIC_REQUIREMENTS = [
        "clear_service_description",
        "data_processing_locations",
        "data_storage_locations",
        "service_level_descriptions",
        "incident_assistance_obligations",
        "cooperation_with_authorities",
        "termination_rights",
    ]

    # Additional for critical functions per Article 30(3)
    CRITICAL_FUNCTION_REQUIREMENTS = [
        "full_sla_with_targets",
        "notice_periods",
        "reporting_obligations",
        "entity_audit_rights",
        "nca_audit_access",
        "exit_strategy",
        "transition_assistance",
        "performance_remediation",
        "business_continuity",
        "security_measures",
    ]

    def assess_contract_compliance(
        self,
        contract: ICTContract,
        is_critical: bool
    ) -> ContractComplianceReport:
        """Assess if contract meets DORA requirements."""

    def generate_contract_gap_analysis(
        self,
        provider: str
    ) -> ContractGapAnalysis:
        """
        Analyze gaps in existing contract with provider.

        For our exchanges (Binance, Alpaca, etc.):
        - Most use standard terms
        - Audit rights typically limited
        - Exit strategies may need development
        """

    def create_contract_amendment_request(
        self,
        provider: str,
        gaps: List[ContractGap]
    ) -> AmendmentRequest:
        """Create request for contract amendments."""
```

### 4.4 Exit Strategies (Article 28(8))

**Файл**: `services/dora/exit_strategies.py`

```python
class DORAExitStrategy:
    """
    Article 28(8) - Exit strategies for ICT third-party providers.

    Critical for our platform - need exit plans for each exchange.
    """

    def create_exit_plan(
        self,
        provider: str
    ) -> ExitPlan:
        """
        Create exit plan including:
        - Alternative providers identified
        - Data migration procedures
        - Service transition timeline
        - Impact on operations
        - Cost estimates
        """

    def identify_alternatives(
        self,
        provider: str,
        services: List[str]
    ) -> List[AlternativeProvider]:
        """
        Identify alternative providers.

        For example:
        - Binance → Kraken, Coinbase, Bybit
        - Alpaca → Interactive Brokers, TD Ameritrade
        - Polygon → Alpha Vantage, IEX Cloud
        """

    def validate_transition_plan(
        self,
        exit_plan: ExitPlan
    ) -> TransitionValidation:
        """Validate exit/transition plan is feasible."""
```

### 4.5 Concentration Risk (Article 29)

**Файл**: `services/dora/concentration_risk.py`

```python
class DORAConcentrationRisk:
    """
    Article 29 - Preliminary assessment of ICT concentration risk.

    Assess concentration risk at:
    - Entity level
    - Sub-consolidated level
    - Consolidated level
    """

    def assess_concentration_risk(self) -> ConcentrationRiskReport:
        """
        Assess concentration risk across all providers.

        Key questions:
        1. How many critical functions depend on single provider?
        2. What if Binance goes down?
        3. What if API connectivity to all crypto exchanges fails?
        4. Geographic concentration (all providers in same region)?
        """

    def calculate_dependency_metrics(self) -> DependencyMetrics:
        """
        Calculate dependency metrics:
        - % of trades via each provider
        - % of market data from each source
        - Provider redundancy ratio
        """

    def develop_mitigation_measures(
        self,
        concentration_risks: List[ConcentrationRisk]
    ) -> List[MitigationMeasure]:
        """Develop measures to reduce concentration risk."""
```

### 4.6 Oversight Framework for Critical ICT Providers (Articles 31-44)

**Примечание**: ESAs designate Critical Third-Party Providers (CTPPs). Our direct exchanges may not be designated, but cloud providers (AWS, GCP) might be.

**Файл**: `services/dora/ctpp_oversight.py`

```python
class CTPPOversight:
    """
    Handle requirements if using designated Critical Third-Party Providers.

    Check ESA list: https://www.esma.europa.eu/dora
    """

    def check_ctpp_designation(
        self,
        provider: str
    ) -> bool:
        """Check if provider is designated CTPP."""

    def implement_ctpp_requirements(
        self,
        ctpp: CriticalProvider
    ):
        """Implement additional requirements for CTPPs."""
```

### Phase 4 Deliverables Summary

| Deliverable | File/Path | Tests |
|-------------|-----------|-------|
| Third-Party Risk Management | `services/dora/third_party_risk.py` | ~40 |
| Register of Information | `services/dora/register_of_information.py` | ~50 |
| Contractual Requirements | `services/dora/contractual_requirements.py` | ~40 |
| Exit Strategies | `services/dora/exit_strategies.py` | ~30 |
| Concentration Risk | `services/dora/concentration_risk.py` | ~30 |
| CTPP Oversight | `services/dora/ctpp_oversight.py` | ~30 |
| **TOTAL** | | **~220** |

### Phase 4 Configuration

```yaml
# config/dora/third_party_management.yaml
third_party_management:
  register_of_information:
    storage_path: "state/dora/register_of_information"
    annual_reporting:
      due_date_month: 4
      due_date_day: 30
      authority: "YOUR_NCA"

  exit_strategies:
    review_frequency_months: 12
    test_frequency_months: 24

  concentration_risk:
    max_single_provider_critical_functions_pct: 30
    geographic_concentration_limit: 2  # countries

  contract_review:
    review_frequency_months: 12
    critical_contracts_frequency_months: 6
```

### Phase 4 Test Requirements

```bash
tests/dora/
├── test_dora_third_party_risk.py
├── test_dora_register_of_information.py
├── test_dora_contractual_requirements.py
├── test_dora_exit_strategies.py
├── test_dora_concentration_risk.py
└── test_dora_ctpp_oversight.py
```

**Критерии завершения Phase 4**:
- [ ] Register of Information fully populated
- [ ] All provider contracts analyzed
- [ ] Exit strategies documented for all critical providers
- [ ] Concentration risk assessed
- [ ] All 220+ тестов проходят

---

# Phase 5: Information Sharing & Final Integration
## Article 45 + Cross-Regulation Integration

**Приоритет**: P1
**Зависимости**: Phases 1-4

### 5.1 Information Sharing Arrangements (Article 45)

**Требования**:
- Share cyber threat information among trusted communities
- Protect sensitive nature of shared information
- Comply with data protection (GDPR)
- Comply with competition law

**Файл**: `services/dora/information_sharing.py`

```python
class DORAInformationSharing:
    """
    Article 45 - Cyber threat information sharing.

    Participate in trusted information sharing communities.
    """

    # Types of information that can be shared
    SHAREABLE_INFORMATION = [
        "indicators_of_compromise",  # IOCs
        "tactics_techniques_procedures",  # TTPs
        "cybersecurity_alerts",
        "configuration_tools",
    ]

    def join_sharing_community(
        self,
        community: SharingCommunity
    ):
        """
        Join information sharing community.

        Examples: FS-ISAC, CERT-EU, National CSIRTs
        """

    def share_threat_intelligence(
        self,
        threat: CyberThreat,
        community: SharingCommunity
    ):
        """
        Share threat intelligence with community.

        Ensure:
        - Anonymization where needed
        - No business confidential data
        - GDPR compliance for personal data
        """

    def receive_threat_intelligence(
        self,
        intelligence: ThreatIntelligence
    ):
        """Process received threat intelligence."""

    def notify_nca_of_participation(
        self,
        community: SharingCommunity
    ):
        """
        Notify competent authority of community participation.

        Required per Article 45(3).
        """
```

### 5.2 Cross-Regulation Integration

**Интеграция DORA с существующими compliance frameworks**:

| Regulation | Overlap Areas | Integration Points |
|------------|---------------|-------------------|
| **EU AI Act** | Risk management, Logging, Incident handling | Extend existing modules |
| **MiFID II** | BCP, Kill switch, Audit trail | Leverage MiFID II config |
| **GDPR** | Data protection, Breach notification | Data governance alignment |
| **NIS2** | Cybersecurity, Incident reporting | Timing alignment |

**Файл**: `services/dora/cross_regulation.py`

```python
class DORARegulationIntegration:
    """
    Integrate DORA with EU AI Act and MiFID II.
    """

    def align_incident_reporting(self):
        """
        Align DORA incident reporting with:
        - AI Act Article 73 (serious incidents)
        - NIS2 requirements

        Note: DORA 24h/72h timeline aligns with NIS2.
        """

    def integrate_risk_frameworks(self):
        """
        Integrate ICT risk with AI Act risk management:
        - Share risk registry
        - Unified risk assessment
        - Combined reporting
        """

    def align_logging_systems(self):
        """
        Extend AI Act logging for DORA:
        - ICT events
        - Security events
        - Incident logs
        """
```

### 5.3 Final Integration and Orchestration

**Файл**: `services/dora/__init__.py`

```python
"""
DORA Compliance Module for AI-Powered Quantitative Research Platform.

Digital Operational Resilience Act (EU Regulation 2022/2554)

This package provides comprehensive DORA compliance:

Phase 1 - ICT Risk Management Framework (Articles 5-16):
    - governance: Management body oversight, roles
    - ict_risk_framework: ICT risk management
    - ict_systems: ICT asset management
    - ict_identification: Risk identification
    - detection: Anomaly and incident detection
    - response_recovery: BCP and recovery
    - backup_recovery: Backup systems
    - learning: Continuous improvement
    - communication: Crisis communication

Phase 2 - ICT Incident Management (Articles 17-23):
    - incident_management: Incident handling
    - incident_classification: Major incident classification
    - incident_reporting: NCA reporting
    - cyber_threat_notification: Threat notification
    - reporting_templates: ITS templates
    - third_party_incidents: Provider incidents

Phase 3 - Digital Resilience Testing (Articles 24-27):
    - resilience_testing: Testing programme
    - ict_testing: System testing
    - tlpt: Threat-Led Penetration Testing
    - tester_management: TLPT tester requirements

Phase 4 - Third-Party ICT Risk (Articles 28-44):
    - third_party_risk: Provider risk management
    - register_of_information: Article 28(3) register
    - contractual_requirements: Contract compliance
    - exit_strategies: Exit plans
    - concentration_risk: Concentration assessment

Phase 5 - Information Sharing (Article 45):
    - information_sharing: Threat intelligence sharing
    - cross_regulation: EU AI Act, MiFID II integration

Application Date: 17 January 2025
"""

__version__ = "1.0.0"
__dora_compliance_phase__ = 0  # Will increment as phases complete
```

### 5.4 Unified Dashboard and Reporting

**Файл**: `services/dora/compliance_dashboard.py`

```python
class DORAComplianceDashboard:
    """
    Unified DORA compliance monitoring dashboard.
    """

    def get_compliance_status(self) -> ComplianceStatus:
        """Get overall DORA compliance status."""

    def generate_compliance_report(
        self,
        period: str
    ) -> DORAComplianceReport:
        """Generate periodic compliance report."""

    def get_upcoming_deadlines(self) -> List[Deadline]:
        """Get upcoming compliance deadlines."""

    def get_open_issues(self) -> List[ComplianceIssue]:
        """Get open compliance issues."""
```

### Phase 5 Deliverables Summary

| Deliverable | File/Path | Tests |
|-------------|-----------|-------|
| Information Sharing | `services/dora/information_sharing.py` | ~40 |
| Cross-Regulation Integration | `services/dora/cross_regulation.py` | ~30 |
| Compliance Dashboard | `services/dora/compliance_dashboard.py` | ~30 |
| Unified Reporting | `services/dora/unified_reporting.py` | ~25 |
| DORA Module Init | `services/dora/__init__.py` | ~25 |
| **TOTAL** | | **~150** |

---

## Project Directory Structure

```
AI-Powered-Quantitative-Research-Platform/
├── services/
│   ├── ai_act/                    # Existing EU AI Act (1007 tests)
│   │   ├── risk_management.py
│   │   ├── post_market_monitoring.py
│   │   ├── logging_system.py
│   │   ├── cybersecurity.py
│   │   └── ... (15 modules)
│   │
│   └── dora/                      # NEW: DORA compliance (~1000 tests)
│       ├── __init__.py
│       │
│       ├── # Phase 1: ICT Risk Management
│       ├── governance.py
│       ├── ict_risk_framework.py
│       ├── ict_systems.py
│       ├── ict_identification.py
│       ├── detection.py
│       ├── response_recovery.py
│       ├── backup_recovery.py
│       ├── learning.py
│       ├── communication.py
│       │
│       ├── # Phase 2: Incident Management
│       ├── incident_management.py
│       ├── incident_classification.py
│       ├── incident_reporting.py
│       ├── cyber_threat_notification.py
│       ├── reporting_templates.py
│       ├── supervisory_feedback.py
│       ├── third_party_incidents.py
│       │
│       ├── # Phase 3: Resilience Testing
│       ├── resilience_testing.py
│       ├── ict_testing.py
│       ├── tlpt.py
│       ├── tester_management.py
│       ├── pooled_testing.py
│       │
│       ├── # Phase 4: Third-Party Risk
│       ├── third_party_risk.py
│       ├── register_of_information.py
│       ├── contractual_requirements.py
│       ├── exit_strategies.py
│       ├── concentration_risk.py
│       ├── ctpp_oversight.py
│       │
│       └── # Phase 5: Information Sharing
│       ├── information_sharing.py
│       ├── cross_regulation.py
│       ├── compliance_dashboard.py
│       └── unified_reporting.py
│
├── configs/
│   ├── compliance/
│   │   ├── mifid_compliance.yaml  # Existing
│   │   └── ...
│   │
│   └── dora/                      # NEW: DORA configs
│       ├── digital_resilience_strategy.yaml
│       ├── ict_risk_thresholds.yaml
│       ├── incident_thresholds.yaml
│       ├── register_of_information.yaml
│       ├── third_party_management.yaml
│       ├── testing_programme.yaml
│       └── information_sharing.yaml
│
├── docs/
│   └── compliance/
│       ├── EU_AI_ACT_INTEGRATION_PLAN.md   # Existing
│       ├── DORA_INTEGRATION_PLAN.md        # THIS DOCUMENT
│       ├── dora/                            # NEW
│       │   ├── digital_resilience_strategy.md
│       │   ├── ict_business_continuity_policy.md
│       │   ├── exit_strategies/
│       │   ├── register_of_information/
│       │   └── incident_reports/
│       └── ...
│
├── tests/
│   ├── test_ai_act_*.py           # Existing (14 files)
│   │
│   └── dora/                      # NEW: DORA tests (~1000 tests)
│       ├── # Phase 1
│       ├── test_dora_governance.py
│       ├── test_dora_ict_risk_framework.py
│       ├── test_dora_ict_systems.py
│       ├── test_dora_identification.py
│       ├── test_dora_detection.py
│       ├── test_dora_response_recovery.py
│       ├── test_dora_backup.py
│       ├── test_dora_learning.py
│       ├── test_dora_communication.py
│       │
│       ├── # Phase 2
│       ├── test_dora_incident_management.py
│       ├── test_dora_incident_classification.py
│       ├── test_dora_incident_reporting.py
│       ├── test_dora_cyber_threat_notification.py
│       ├── test_dora_reporting_templates.py
│       ├── test_dora_supervisory_feedback.py
│       ├── test_dora_third_party_incidents.py
│       │
│       ├── # Phase 3
│       ├── test_dora_resilience_testing.py
│       ├── test_dora_ict_testing.py
│       ├── test_dora_tlpt.py
│       ├── test_dora_tester_management.py
│       ├── test_dora_pooled_testing.py
│       │
│       ├── # Phase 4
│       ├── test_dora_third_party_risk.py
│       ├── test_dora_register_of_information.py
│       ├── test_dora_contractual_requirements.py
│       ├── test_dora_exit_strategies.py
│       ├── test_dora_concentration_risk.py
│       ├── test_dora_ctpp_oversight.py
│       │
│       └── # Phase 5
│       ├── test_dora_information_sharing.py
│       ├── test_dora_cross_regulation.py
│       ├── test_dora_compliance_dashboard.py
│       └── test_dora_unified_reporting.py
│
└── state/
    └── dora/                      # NEW: DORA runtime state
        ├── register_of_information/
        ├── incidents/
        ├── testing/
        └── reports/
```

---

## Implementation Considerations

### 1. Reuse Strategy

| Existing Component | DORA Reuse | Modification Required |
|-------------------|------------|----------------------|
| `AIActRiskManager` | Extend for ICT risks | Add `DORAICTRiskCategory` |
| `IncidentTracker` | Extend for DORA incidents | Add DORA classification |
| `AIActLogger` | Extend for ICT events | Add ICT event types |
| `AIActCybersecurity` | Extend for DORA | Add continuous monitoring |
| MiFID II BCP config | Leverage directly | Minor extensions |
| Adapters | Document as ICT providers | Create Register of Information |

### 2. Priority Order

```
Phase 4 (Third-Party Risk) → Can start immediately (Register of Information)
Phase 1 (ICT Risk Management) → Core framework
Phase 2 (Incident Reporting) → Regulatory deadline sensitive
Phase 3 (Resilience Testing) → Requires framework first
Phase 5 (Integration) → Final integration
```

**Рекомендация**: Начать с Phase 4 параллельно с Phase 1, так как Register of Information является независимым требованием и имеет deadline (30 April 2025).

### 3. RTS/ITS Compliance

| Standard | Status | Implementation |
|----------|--------|----------------|
| RTS on ICT Risk Management | Final | Phase 1 |
| RTS on Incident Classification | Final (2024/1772) | Phase 2 |
| ITS on Incident Reporting | Final (JC 2024-33) | Phase 2 |
| ITS on Register of Information | Final (JC 2023 85) | Phase 4 |
| RTS on TLPT | Final | Phase 3 |
| RTS on Subcontracting | Final (2025/532) | Phase 4 |

### 4. Testing Strategy

Для каждой фазы:
1. Unit tests для каждого модуля (>95% coverage)
2. Integration tests с существующими модулями
3. End-to-end compliance validation tests
4. Mock NCA submission tests (для incident reporting)

---

## References & Sources

### Official EU Sources
- [DORA Full Text (EUR-Lex)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32022R2554)
- [ESMA DORA Page](https://www.esma.europa.eu/esmas-activities/digital-finance-and-innovation/digital-operational-resilience-act-dora)
- [EBA DORA Technical Standards](https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/operational-resilience)
- [EIOPA DORA Page](https://www.eiopa.europa.eu/digital-operational-resilience-act-dora_en)

### Technical Standards (RTS/ITS)
- [RTS on ICT Risk Management Framework](https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/operational-resilience/regulatory-technical-standards-ict-risk-management-framework-and-simplified-ict-risk-management)
- [RTS/ITS on Incident Reporting (JC 2024-33)](https://www.esma.europa.eu/sites/default/files/2024-07/JC_2024-33_-_Final_report_on_the_draft_RTS_and_ITS_on_incident_reporting.pdf)
- [ITS on Register of Information (JC 2023 85)](https://www.esma.europa.eu/sites/default/files/2024-01/JC_2023_85_-_Final_report_on_draft_ITS_on_Register_of_Information.pdf)
- [RTS on TLPT (JC 2024-29)](https://www.esma.europa.eu/sites/default/files/2024-07/JC_2024-29_-_Final_report_DORA_RTS_on_TLPT.pdf)
- [Commission Delegated Regulation 2024/1772 (Incident Classification)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1772)
- [Commission Delegated Regulation 2025/532 (Subcontracting)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32025R0532)

### DORA Articles Reference
- [Article 5-11: ICT Risk Management](https://www.digital-operational-resilience-act.com/DORA_Articles.html)
- [Article 11: Response and Recovery](https://www.digital-operational-resilience-act.com/Article_11.html)
- [Article 12: Backup Policies](https://www.digital-operational-resilience-act.com/Article_12.html)
- [Article 17-23: Incident Reporting](https://www.digital-operational-resilience-act.com/DORA_Articles.html)
- [Article 24-27: Digital Resilience Testing](https://www.digital-operational-resilience-act.com/DORA_Articles.html)
- [Article 28: Third-Party Risk](https://www.digital-operational-resilience-act.com/Article_28.html)
- [Article 45: Information Sharing](https://www.digital-operational-resilience-act.com/Article_45.html)

### Implementation Guides
- [FS-ISAC DORA Implementation Guidance](https://www.fsisac.com/hubfs/Knowledge/DORA/FSISAC_DORA-ImplementationGuidance.pdf)
- [IBM DORA Overview](https://www.ibm.com/think/topics/digital-operational-resilience-act)
- [IT Governance DORA Guide](https://www.itgovernanceusa.com/eu-digital-operational-resilience-act)

---

## Summary

| Metric | Value |
|--------|-------|
| **Total Phases** | 5 |
| **Total New Modules** | ~30 |
| **Total New Tests** | ~1000 |
| **Integration with AI Act** | High (reuse existing modules) |
| **Integration with MiFID II** | Medium (BCP, audit trail) |
| **Regulatory Deadline** | 17 January 2025 (active) |

**Ключевые риски**:
1. Contractual negotiations с exchanges (Binance, Alpaca, etc.)
2. TLPT требует внешних тестировщиков
3. Register of Information deadline (30 April 2025)

**Следующие шаги**:
1. Начать Phase 4 (Register of Information) — критический deadline
2. Параллельно начать Phase 1 (ICT Risk Management)
3. Документировать все ICT third-party providers
4. Провести gap analysis контрактов с exchanges

---

**Document Version History**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-08 | Claude | Initial comprehensive plan |

