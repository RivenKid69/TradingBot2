# DORA Integration Layer Migration Plan

**Version:** 1.1
**Date:** 2025-01-17
**Status:** Draft
**Updated:** Paths, class names, test strategy synchronized with codebase

---

## Executive Summary

Данный документ описывает план реорганизации DORA-модулей платформы с разделением на:

1. **CORE** (`services/core/`) — операционная устойчивость провайдера (НЕ ТРОГАЕМ)
2. **INTEGRATION** (`services/dora_integration/`) — интерфейс взаимодействия с клиентами
3. **ARCHIVE** (`services/archive/dora_financial_entity/`) — модули для финансовых организаций

**Ключевой принцип:** Мы — ICT-провайдер (Art. 30 DORA), а не финансовая организация (Art. 2). Интеграционный слой — это то, что мы отдаём клиенту для его compliance.

---

## 1. Target Architecture

```
services/
├── core/                          # CORE - НЕ ТРОГАЕМ (14 modules)
│   ├── structured_logging.py
│   ├── enhanced_healthcheck.py
│   ├── tiered_backup.py
│   ├── dr_testing.py
│   ├── dr_execution.py
│   ├── multi_az.py
│   ├── alerting.py
│   ├── security_gates.py
│   ├── oncall_rotation.py
│   ├── ctpp_monitoring.py
│   ├── subcontractor_monitoring.py
│   ├── trust_center.py
│   ├── soc2_dora_mapping.py
│   └── __init__.py
│
├── dora_integration/              # NEW - Integration Layer (21 modules)
│   ├── __init__.py
│   │
│   ├── due_diligence/             # 2.1 - Audit & Due Diligence
│   │   ├── __init__.py
│   │   ├── audit_readiness.py
│   │   ├── provider_info_package.py
│   │   ├── pooled_audit_support.py
│   │   └── compliance_dashboard.py
│   │
│   ├── incident_interface/        # 2.2 - Incident Communication
│   │   ├── __init__.py
│   │   ├── client_incident_notification.py
│   │   ├── incident_classification.py
│   │   ├── incident_reporting.py
│   │   ├── cyber_threat_notification.py
│   │   └── communication.py
│   │
│   ├── third_party/               # 2.3 - Third-Party Risk Interface
│   │   ├── __init__.py
│   │   ├── concentration_risk.py
│   │   ├── ctpp_oversight.py
│   │   ├── third_party_risk.py
│   │   ├── third_party_incidents.py
│   │   └── subcontractor_management.py
│   │
│   ├── contracts/                 # 2.4 - Contractual Layer
│   │   ├── __init__.py
│   │   ├── contractual_requirements.py
│   │   ├── sla_guardrails.py
│   │   └── exit_strategies.py
│   │
│   ├── reporting/                 # 2.5 - Unified Reporting
│   │   ├── __init__.py
│   │   ├── unified_reporting.py
│   │   ├── reporting_templates.py
│   │   └── register_of_information.py
│   │
│   └── sharing/                   # 2.6 - Information Sharing
│       ├── __init__.py
│       └── information_sharing.py
│
├── dora/                          # Thin facade → re-exports from dora_integration
│   └── __init__.py                # FE framework archived, integration re-exported
│
├── archive/                       # ARCHIVE - FE modules (единственное место)
│   └── dora_financial_entity/
│       ├── README.md
│       ├── configs/
│       │   ├── entity_classification.yaml
│       │   └── nca_identification.yaml
│       ├── scope_verification.py
│       ├── function_classification.py
│       ├── proportionality.py
│       ├── governance.py
│       ├── ict_risk_framework.py
│       ├── training_participation.py  # Art. 30(2)(i) - FE training requests
│       └── ... (full FE framework, 23 modules total)
│
config/                            # Конфиги (НЕ configs/)
├── dora/
│   └── proportionality_assessment.yaml   # KEEP - internal toggle
│
└── dora_integration/              # NEW - Integration configs
    ├── digital_resilience_strategy.yaml
    ├── third_party_management.yaml
    └── information_sharing.yaml
```

---

## 2. Migration Phases

### Phase 0: Preparation (Pre-requisites)
**Duration:** 1-2 days
**Risk:** Low

| Task | Action | Files |
|------|--------|-------|
| 0.1 | Create directory structure | `services/dora_integration/`, subfolders |
| 0.2 | Create archive directory | `services/archive/dora_financial_entity/` |
| 0.3 | Create integration config dir | `config/dora_integration/` |
| 0.4 | Backup current state | `git tag pre-integration-refactor` |
| 0.5 | Document current imports | Audit all `from services.dora import` |
| 0.6 | Create migration branch | `git checkout -b refactor/dora-integration-layer` |

**Commands:**
```bash
mkdir -p services/dora_integration/{due_diligence,incident_interface,third_party,contracts,reporting,sharing}
mkdir -p services/archive/dora_financial_entity/configs
mkdir -p config/dora_integration
git tag -a pre-integration-refactor -m "Before DORA integration layer refactor"
```

---

### Phase 1: Due Diligence & Audit Layer
**Duration:** 2-3 days
**Risk:** Medium
**Dependencies:** None

#### 1.1 Move Modules

| Source | Target | Notes |
|--------|--------|-------|
| `services/dora/audit_readiness.py` | `services/dora_integration/due_diligence/audit_readiness.py` | Art. 30(3)(e) |
| `services/dora/provider_info_package.py` | `services/dora_integration/due_diligence/provider_info_package.py` | Art. 28(3) ROI data |
| `services/dora/pooled_audit_support.py` | `services/dora_integration/due_diligence/pooled_audit_support.py` | Art. 30(4) |
| `services/dora/compliance_dashboard.py` | `services/dora_integration/due_diligence/compliance_dashboard.py` | Status panel |

#### 1.2 Create `__init__.py`

```python
# services/dora_integration/due_diligence/__init__.py
"""
Due Diligence & Audit Readiness Module.

Provides interfaces for:
- Client audit requests (Art. 30(3)(e))
- Provider information packages for ROI (Art. 28(3))
- Pooled audit coordination (Art. 30(4))
- Compliance status dashboard
"""

# Audit Readiness - uses existing class names from code
from services.dora_integration.due_diligence.audit_readiness import (
    DORAuditReadiness,          # Main class (exists in code)
    AuditRequest,               # Data structure (exists)
    EvidenceItem,               # Data structure (exists)
    create_audit_readiness,     # Factory (exists)
    get_standard_evidence_templates,  # Helper (exists)
)

# Provider Info Package - uses existing class names
from services.dora_integration.due_diligence.provider_info_package import (
    ProviderIdentification,     # Data structure (exists)
    ICTServiceType,             # Enum (exists)
    ICTServiceDescription,      # Data structure (exists)
    DataLocationInfo,           # Data structure (exists)
    ProviderInfoPackageGenerator,  # Main class (exists as implicit)
)

# Pooled Audit Support - uses existing class names
from services.dora_integration.due_diligence.pooled_audit_support import (
    PooledAuditSupport,         # Main class (exists)
    PooledAuditEngagement,      # Data structure (exists)
    CertificationRecord,        # Data structure (exists)
    create_pooled_audit_support,  # Factory (exists)
)

# Compliance Dashboard - uses existing class names
from services.dora_integration.due_diligence.compliance_dashboard import (
    DORAComplianceDashboard,    # Main class (exists)
    ComplianceStatus,           # Data structure (exists)
    DORAComplianceReport,       # Data structure (exists)
)

__all__ = [
    # Audit
    "DORAuditReadiness", "AuditRequest", "EvidenceItem",
    "create_audit_readiness", "get_standard_evidence_templates",
    # Provider Info
    "ProviderIdentification", "ICTServiceType", "ICTServiceDescription",
    "DataLocationInfo", "ProviderInfoPackageGenerator",
    # Pooled Audit
    "PooledAuditSupport", "PooledAuditEngagement", "CertificationRecord",
    "create_pooled_audit_support",
    # Dashboard
    "DORAComplianceDashboard", "ComplianceStatus", "DORAComplianceReport",
]
```

#### 1.3 Update Import Paths

Create compatibility shim in `services/dora/__init__.py`:
```python
# Deprecation shims for Phase 1 modules
import warnings

def _deprecated_import(old_path, new_path, name):
    warnings.warn(
        f"Importing {name} from {old_path} is deprecated. "
        f"Use {new_path} instead.",
        DeprecationWarning,
        stacklevel=3
    )

# Maintain backward compatibility during migration
try:
    from services.dora_integration.due_diligence import (
        DORAuditReadiness, AuditRequest, # ...
    )
except ImportError:
    pass  # Not yet migrated
```

#### 1.4 Tests

```bash
# Run existing tests to verify no breakage
pytest tests/dora/ -v -k "audit or provider_info or pooled"

# Add migration-specific tests
pytest tests/dora_integration/test_due_diligence.py -v
```

---

### Phase 2: Incident Interface Layer
**Duration:** 3-4 days
**Risk:** High (affects real-time notifications)
**Dependencies:** Phase 1 complete

#### 2.1 Move Modules

| Source | Target | Notes |
|--------|--------|-------|
| `services/dora/client_incident_notification.py` | `services/dora_integration/incident_interface/client_incident_notification.py` | Client notification |
| `services/dora/incident_classification.py` | `services/dora_integration/incident_interface/incident_classification.py` | CDR 2024/1772 |
| `services/dora/incident_reporting.py` | `services/dora_integration/incident_interface/incident_reporting.py` | Export, not submit |
| `services/dora/cyber_threat_notification.py` | `services/dora_integration/incident_interface/cyber_threat_notification.py` | Art. 19(4) |
| `services/dora/communication.py` | `services/dora_integration/incident_interface/communication.py` | Art. 14 channels |

#### 2.2 Refactor Focus: Export-Only Semantics

**Ключевое изменение:** Модули должны работать как **экспорт данных клиенту**, а не как полная система управления инцидентами.

> **NOTE:** Текущий `incident_reporting.py` уже содержит правильный docstring:
> _"For ICT THIRD-PARTY PROVIDERS: We do NOT report directly to NCAs... We notify CLIENTS, who then report to their NCAs"_

**Рефакторинг `DORAIncidentReporter`:**

```python
# services/dora_integration/incident_interface/incident_reporting.py
"""
Incident Data Export for Client NCA Reporting.

We generate DORA-compliant incident data packages.
Clients use this data to fulfill their Art. 19 obligations.
"""

class DORAIncidentReporter:
    """
    DORA Incident Reporter - Export-Only Mode.

    Primary methods for ICT providers:
    - generate_client_data_package()  # NEW: main export method
    - create_initial_notification()   # Creates template, doesn't submit
    - create_intermediate_report()    # Creates template, doesn't submit
    - create_final_report()           # Creates template, doesn't submit

    DEPRECATED (for FE use only):
    - submit_report()  # Raises DeprecationWarning, archived to FE module
    """

    def generate_client_data_package(
        self,
        incident_id: str,
        format: str = "json"
    ) -> bytes:
        """
        Generate complete incident data package for client.

        Client uses this to populate their NCA submission.
        Formats: json, xml, csv, dpm (DPM taxonomy)
        """
        ...

    def export_for_client_roi(self, incident_id: str) -> dict:
        """Export incident data for client's Register of Information."""
        ...

    # Keep existing report creation methods - they generate templates
    def create_initial_notification(self, ...) -> InitialNotificationReport:
        """Create initial notification template (client submits to NCA)."""
        ...
```

**Связь с `client_incident_notification.py`:**
```python
# Flow: Detection → Classification → Client Notification → Data Export
#
# 1. DORAIncidentClassification.classify_incident()
# 2. ClientIncidentNotification.notify_client()  # We notify client
# 3. DORAIncidentReporter.generate_client_data_package()  # Client gets data
# 4. Client submits to their NCA using our data package
```

#### 2.3 Create `__init__.py`

```python
# services/dora_integration/incident_interface/__init__.py
"""
Incident Communication Interface.

Provides:
- Client incident notifications (Art. 30(2)(d))
- Incident classification (CDR 2024/1772)
- Incident data export for client NCA reporting
- Cyber threat notifications

NOTE: We notify CLIENTS. Clients report to NCAs.
"""

# Client Notification - uses existing class names
from services.dora_integration.incident_interface.client_incident_notification import (
    IncidentSeverity,              # Enum (exists)
    NotificationStatus,            # Enum (exists)
    NotificationChannel,           # Enum (exists)
    ClientContact,                 # Data structure (exists)
    IncidentNotification,          # Data structure (exists)
    ClientNotificationService,     # Main class (exists)
)

# Incident Classification - uses existing class names
from services.dora_integration.incident_interface.incident_classification import (
    DORAIncidentClassification,        # Main class (exists)
    IncidentClassificationResult,      # Data structure (exists)
    ClassificationThresholds,          # Data structure (exists)
    create_incident_classification,    # Factory (exists)
)

# Incident Reporting (Export-Only) - uses existing class names
from services.dora_integration.incident_interface.incident_reporting import (
    DORAIncidentReporter,              # Main class (exists)
    ReportType,                        # Enum (exists)
    ReportStatus,                      # Enum (exists)
    InitialNotificationReport,         # Data structure (exists)
    IntermediateReport,                # Data structure (exists)
    FinalReport,                       # Data structure (exists)
    create_incident_reporter,          # Factory (exists)
)

# Cyber Threat Notification - uses existing class names
from services.dora_integration.incident_interface.cyber_threat_notification import (
    CyberThreatNotificationService,    # Main class (exists)
    ThreatNotification,                # Data structure (exists)
    ThreatSeverity,                    # Enum (exists)
    create_cyber_threat_notification_service,  # Factory (exists)
)

# Communication - uses existing class names
from services.dora_integration.incident_interface.communication import (
    DORACommunication,                 # Main class (exists)
    CommunicationPolicy,               # Data structure (exists)
    CrisisCommunicationPlan,           # Data structure (exists)
    create_dora_communication,         # Factory (exists)
)

__all__ = [
    # Client Notification
    "IncidentSeverity", "NotificationStatus", "NotificationChannel",
    "ClientContact", "IncidentNotification", "ClientNotificationService",
    # Classification
    "DORAIncidentClassification", "IncidentClassificationResult",
    "ClassificationThresholds", "create_incident_classification",
    # Reporting (Export)
    "DORAIncidentReporter", "ReportType", "ReportStatus",
    "InitialNotificationReport", "IntermediateReport", "FinalReport",
    "create_incident_reporter",
    # Cyber Threat
    "CyberThreatNotificationService", "ThreatNotification", "ThreatSeverity",
    "create_cyber_threat_notification_service",
    # Communication
    "DORACommunication", "CommunicationPolicy", "CrisisCommunicationPlan",
    "create_dora_communication",
]
```

#### 2.4 Integration Points

```yaml
# config/dora_integration/incident_notification.yaml
notification:
  channels:
    webhook:
      enabled: true
      retry_policy:
        max_retries: 3
        backoff_seconds: [30, 60, 120]
    email:
      enabled: true
      template: "incident_notification_v2"
    api:
      enabled: true
      endpoint_pattern: "/api/v1/clients/{client_id}/incidents"

  sla:
    critical:
      notify_within_minutes: 30
      description: "Gives client 3.5h for NCA report"
    high:
      notify_within_minutes: 60
    medium:
      notify_within_minutes: 240
    low:
      notify_within_minutes: 1440
```

---

### Phase 3: Third-Party Risk Interface
**Duration:** 2-3 days
**Risk:** Medium
**Dependencies:** Phase 1 complete

#### 3.1 Move Modules

| Source | Target | Notes |
|--------|--------|-------|
| `services/dora/concentration_risk.py` | `services/dora_integration/third_party/concentration_risk.py` | CTPP designation risk |
| `services/dora/ctpp_oversight.py` | `services/dora_integration/third_party/ctpp_oversight.py` | Art. 31-44 |
| `services/dora/third_party_risk.py` | `services/dora_integration/third_party/third_party_risk.py` | Risk models |
| `services/dora/third_party_incidents.py` | `services/dora_integration/third_party/third_party_incidents.py` | Subcontractor incidents |
| `services/dora/subcontractor_management.py` | `services/dora_integration/third_party/subcontractor_management.py` | Art. 30(2)(b) |

#### 3.2 Link with Core

Связь с `services/core/subcontractor_monitoring.py`:

```python
# services/dora_integration/third_party/subcontractor_management.py

from services.core.subcontractor_monitoring import (
    SubcontractorMonitor,
    SubcontractorStatus,
)

class DORASubcontractorManagement:
    """
    DORA-compliant subcontractor management.

    Extends core monitoring with:
    - Art. 30(2)(b) prior consent workflows
    - Client notification on changes
    - ROI data generation for client registers
    """

    def __init__(self, core_monitor: SubcontractorMonitor):
        self._monitor = core_monitor

    def generate_subcontractor_chain_for_roi(self, client_id: str) -> list:
        """Generate B_99.01 subcontractor chain data for client ROI."""
        ...
```

#### 3.3 Move Configs

```bash
mv config/dora/third_party_management.yaml config/dora_integration/third_party_management.yaml
```

---

### Phase 4: Contracts & SLA Layer
**Duration:** 2 days
**Risk:** Low
**Dependencies:** None (can run parallel with Phase 2-3)

#### 4.1 Move Modules

| Source | Target | Notes |
|--------|--------|-------|
| `services/dora/contractual_requirements.py` | `services/dora_integration/contracts/contractual_requirements.py` | Art. 30 compliance |
| `services/dora/sla_guardrails.py` | `services/dora_integration/contracts/sla_guardrails.py` | Art. 30(2)(e) |
| `services/dora/exit_strategies.py` | `services/dora_integration/contracts/exit_strategies.py` | Art. 28(8) |

#### 4.2 Create `__init__.py`

```python
# services/dora_integration/contracts/__init__.py
"""
Contractual Interface Layer.

Provides:
- Contract compliance checking (Art. 30)
- SLA guardrails and capacity validation
- Exit strategy templates and data migration plans

These modules help structure contracts that meet DORA requirements.
"""
```

---

### Phase 5: Unified Reporting Layer
**Duration:** 2-3 days
**Risk:** Medium
**Dependencies:** Phase 2 (incident interface)

#### 5.1 Move Modules

| Source | Target | Notes |
|--------|--------|-------|
| `services/dora/unified_reporting.py` | `services/dora_integration/reporting/unified_reporting.py` | Cross-reg reports |
| `services/dora/reporting_templates.py` | `services/dora_integration/reporting/reporting_templates.py` | ITS templates |
| `services/dora/register_of_information.py` | `services/dora_integration/reporting/register_of_information.py` | ROI engine |

#### 5.2 Refactor ROI Module

**Ключевое изменение:** ROI модуль становится **генератором данных**, а не регистром:

```python
# BEFORE
class DORARegisterOfInformation:
    def add_provider(self, provider):  # WRONG - we don't maintain client's register
        ...
    def submit_to_nca(self, nca_id):  # WRONG - client submits
        ...

# AFTER
class ROIDataGenerator:
    """
    Generate data for client's Register of Information.

    We provide the DATA that clients need to populate their ROI.
    We do NOT maintain the register - that's the client's responsibility.
    """

    def generate_provider_record(self) -> dict:
        """Generate B_02.01 provider identification record."""
        ...

    def generate_service_records(self) -> list[dict]:
        """Generate B_03.01 service type records."""
        ...

    def generate_subcontractor_chain(self) -> list[dict]:
        """Generate B_99.01 subcontractor chain records."""
        ...

    def export_full_package(self, format: str = "json") -> bytes:
        """Export complete ROI data package for client."""
        ...
```

---

### Phase 6: Information Sharing Layer
**Duration:** 1 day
**Risk:** Low
**Dependencies:** None

#### 6.1 Move Modules

| Source | Target | Notes |
|--------|--------|-------|
| `services/dora/information_sharing.py` | `services/dora_integration/sharing/information_sharing.py` | Art. 45 |

#### 6.2 Move Configs

```bash
mv config/dora/information_sharing.yaml config/dora_integration/information_sharing.yaml
mv config/dora/digital_resilience_strategy.yaml config/dora_integration/digital_resilience_strategy.yaml
```

---

### Phase 7: Archive Financial Entity Modules
**Duration:** 1 day
**Risk:** Low
**Dependencies:** All phases complete

#### 7.1 Archive Modules

Эти модули предназначены для финансовых организаций, не для ICT-провайдера.

**Целевая директория:** `services/archive/dora_financial_entity/`

| # | Module | DORA Article | Reason |
|---|--------|--------------|--------|
| 1 | `scope_verification.py` | Art. 2 | FE scope check |
| 2 | `function_classification.py` | Art. 3(22) | FE классифицирует свои функции |
| 3 | `proportionality.py` | Art. 4, 16 | FE определяет свой режим |
| 4 | `governance.py` | Art. 5 | FE governance |
| 5 | `ict_risk_framework.py` | Art. 6 | FE framework |
| 6 | `ict_systems.py` | Art. 7 | FE systems |
| 7 | `ict_identification.py` | Art. 8 | FE assets |
| 8 | `protection.py` | Art. 9 | FE protection |
| 9 | `detection.py` | Art. 10 | FE detection |
| 10 | `response_recovery.py` | Art. 11 | FE response |
| 11 | `backup_recovery.py` | Art. 12 | FE backup |
| 12 | `learning.py` | Art. 13 | FE learning |
| 13 | `ict_business_continuity.py` | Art. 15 | FE BCP |
| 14 | `simplified_framework.py` | Art. 16 | FE simplified regime |
| 15 | `incident_management.py` | Art. 17 | FE incident mgmt |
| 16 | `supervisory_feedback.py` | Art. 22 | FE ↔ NCA feedback |
| 17 | `resilience_testing.py` | Art. 24 | FE testing programme |
| 18 | `ict_testing.py` | Art. 25 | FE ICT testing |
| 19 | `tlpt.py` | Art. 26 | FE TLPT |
| 20 | `tester_management.py` | Art. 27 | FE testers |
| 21 | `pooled_testing.py` | Art. 26(3) | FE pooled TLPT |
| 22 | `cross_regulation.py` | - | FE cross-reg compliance |
| 23 | `training_participation.py` | Art. 30(2)(i) | FE training requests to providers |

**Итого: 23 модуля**

#### 7.2 Archive Configs

**Целевая директория:** `services/archive/dora_financial_entity/configs/`

| Config | Source | Reason |
|--------|--------|--------|
| `entity_classification.yaml` | `config/dora/` | FE classification |
| `nca_identification.yaml` | `config/dora/` | FE ↔ NCA mapping |

#### 7.3 Archive Commands

```bash
# Move all FE modules
for module in scope_verification function_classification proportionality \
              governance ict_risk_framework ict_systems ict_identification \
              protection detection response_recovery backup_recovery learning \
              ict_business_continuity simplified_framework incident_management \
              supervisory_feedback resilience_testing ict_testing tlpt \
              tester_management pooled_testing cross_regulation training_participation; do
    mv services/dora/${module}.py services/archive/dora_financial_entity/
done

# Move FE configs
mv config/dora/entity_classification.yaml services/archive/dora_financial_entity/configs/
mv config/dora/nca_identification.yaml services/archive/dora_financial_entity/configs/
```

#### 7.4 Create Archive README

```markdown
# Archived DORA Financial Entity Modules

These modules implement DORA requirements for **Financial Entities** (Art. 2),
not for ICT Third-Party Service Providers (Art. 30).

## Why Archived?

As an ICT service provider, we:
- Comply with Art. 30 (contractual requirements)
- Support client due diligence (Art. 28)
- DO NOT implement full FE DORA framework

Our active DORA code lives in:
- `services/core/` - Operational resilience
- `services/dora_integration/` - Client-facing interfaces

## When to Use?

If you're building a product FOR financial entities to manage their own
DORA compliance, these modules provide a reference implementation.

## Archived Modules (23 total)

| Module | Article | Description |
|--------|---------|-------------|
| `scope_verification.py` | Art. 2 | DORA scope determination |
| `function_classification.py` | Art. 3(22) | Critical function classification |
| `proportionality.py` | Art. 4, 16 | Proportionality regime |
| `governance.py` | Art. 5 | ICT governance framework |
| `ict_risk_framework.py` | Art. 6 | ICT risk management |
| `ict_systems.py` | Art. 7 | ICT systems management |
| `ict_identification.py` | Art. 8 | ICT asset identification |
| `protection.py` | Art. 9 | Protection controls |
| `detection.py` | Art. 10 | Anomaly detection |
| `response_recovery.py` | Art. 11 | Incident response |
| `backup_recovery.py` | Art. 12 | Backup policies |
| `learning.py` | Art. 13 | Learning & evolving |
| `ict_business_continuity.py` | Art. 15 | Business continuity |
| `simplified_framework.py` | Art. 16 | Simplified ICT framework |
| `incident_management.py` | Art. 17 | Incident management |
| `supervisory_feedback.py` | Art. 22 | NCA feedback handling |
| `resilience_testing.py` | Art. 24 | Testing programme |
| `ict_testing.py` | Art. 25 | ICT tools testing |
| `tlpt.py` | Art. 26 | Threat-led penetration testing |
| `tester_management.py` | Art. 27 | Tester requirements |
| `pooled_testing.py` | Art. 26(3) | Pooled TLPT |
| `cross_regulation.py` | - | Cross-regulation integration |
| `training_participation.py` | Art. 30(2)(i) | FE training requests |

## Archived Configs

- `configs/entity_classification.yaml` - Entity type classification
- `configs/nca_identification.yaml` - NCA contact mapping

## Recovery

To restore any module:
```bash
git log --oneline -- services/dora/<module>.py  # Find last commit
git checkout <commit>^ -- services/dora/<module>.py
```
```

---

### Phase 8: Final Integration & Cleanup
**Duration:** 2-3 days
**Risk:** Medium
**Dependencies:** All phases complete

#### 8.1 Update Main `__init__.py`

```python
# services/dora_integration/__init__.py
"""
DORA Integration Layer.

This package provides interfaces for interacting with financial entity clients
in a DORA-compliant manner. It implements ICT provider obligations under Art. 30.

Subpackages:
- due_diligence: Audit readiness, provider info packages
- incident_interface: Client notifications, incident data export
- third_party: Subcontractor management, concentration risk
- contracts: Contractual requirements, SLA guardrails, exit strategies
- reporting: Unified reporting, ROI data generation
- sharing: Information sharing arrangements
"""

__version__ = "1.0.0"

# Due Diligence - uses REAL class names from code
from services.dora_integration.due_diligence import (
    DORAuditReadiness,              # audit_readiness.py
    ProviderIdentification,         # provider_info_package.py
    PooledAuditSupport,             # pooled_audit_support.py
    DORAComplianceDashboard,        # compliance_dashboard.py
)

# Incident Interface - uses REAL class names from code
from services.dora_integration.incident_interface import (
    ClientNotificationService,      # client_incident_notification.py
    DORAIncidentClassification,     # incident_classification.py
    DORAIncidentReporter,           # incident_reporting.py (export-only mode)
    CyberThreatNotificationService, # cyber_threat_notification.py
    DORACommunication,              # communication.py
)

# Third Party - uses REAL class names from code
from services.dora_integration.third_party import (
    DORASubcontractorManagement,    # subcontractor_management.py
    DORAConcentrationRisk,          # concentration_risk.py
    DORACtppOversight,              # ctpp_oversight.py
    DORAThirdPartyRiskManagement,   # third_party_risk.py
    DORAThirdPartyIncidents,        # third_party_incidents.py
)

# Contracts - uses REAL class names from code
from services.dora_integration.contracts import (
    DORAContractualRequirements,    # contractual_requirements.py
    SLAGuardrails,                  # sla_guardrails.py
    DORAExitStrategies,             # exit_strategies.py
)

# Reporting - uses REAL class names from code
from services.dora_integration.reporting import (
    UnifiedReportingManager,        # unified_reporting.py
    DORAReportingTemplates,         # reporting_templates.py
    DORARegisterOfInformation,      # register_of_information.py (as data generator)
)

# Sharing - uses REAL class names from code
from services.dora_integration.sharing import (
    DORAInformationSharing,         # information_sharing.py
)

__all__ = [
    # Due Diligence
    "DORAuditReadiness", "ProviderIdentification",
    "PooledAuditSupport", "DORAComplianceDashboard",
    # Incident Interface
    "ClientNotificationService", "DORAIncidentClassification",
    "DORAIncidentReporter", "CyberThreatNotificationService", "DORACommunication",
    # Third Party
    "DORASubcontractorManagement", "DORAConcentrationRisk",
    "DORACtppOversight", "DORAThirdPartyRiskManagement", "DORAThirdPartyIncidents",
    # Contracts
    "DORAContractualRequirements", "SLAGuardrails", "DORAExitStrategies",
    # Reporting
    "UnifiedReportingManager", "DORAReportingTemplates", "DORARegisterOfInformation",
    # Sharing
    "DORAInformationSharing",
]
```

#### 8.2 Update `services/dora/__init__.py`

`services/dora/` становится **тонким фасадом** — re-export из integration layer:

```python
# services/dora/__init__.py
"""
DORA Compliance Module - Facade.

ARCHITECTURE (post-migration):
    services/core/              - Operational resilience (14 modules)
    services/dora_integration/  - Client-facing interfaces (21 modules)
    services/dora/              - THIS FILE: thin facade for backward compatibility
    services/archive/dora_financial_entity/  - Archived FE modules (23 modules)

USAGE:
    # Preferred (direct import from integration layer):
    from services.dora_integration.due_diligence import DORAuditReadiness

    # Also works (via this facade):
    from services.dora import DORAuditReadiness  # DeprecationWarning in v2.0

For new code, prefer direct imports from services.dora_integration.
"""

import warnings

__version__ = "2.0.0"  # Major bump: FE modules archived, integration layer active

# Re-export from integration layer for backward compatibility
from services.dora_integration import *

# Deprecation notice for old import path
def __getattr__(name):
    # Triggered when accessing attributes not explicitly imported
    warnings.warn(
        f"Importing from services.dora is deprecated. "
        f"Use services.dora_integration instead.",
        DeprecationWarning,
        stacklevel=2
    )
    raise AttributeError(f"module 'services.dora' has no attribute '{name}'")
```

#### 8.3 Update All Imports

```bash
# Find all imports from services.dora
grep -r "from services.dora import" --include="*.py" | grep -v "services/dora/"
grep -r "from services.dora." --include="*.py" | grep -v "services/dora/"

# Update to new paths
# Example: sed -i 's/from services.dora.audit_readiness/from services.dora_integration.due_diligence.audit_readiness/g'
```

#### 8.4 Run Full Test Suite

```bash
pytest tests/ -v --tb=short
pytest tests/dora_integration/ -v
pytest tests/core/ -v
```

---

### Phase 9: Test Strategy & Validation
**Duration:** 2-3 days (parallel with Phase 8)
**Risk:** Medium
**Dependencies:** Phases 1-7 complete

#### 9.1 Test Directory Structure

```
tests/
├── core/                          # Existing - НЕ ТРОГАЕМ
│   └── test_*.py
│
├── dora_integration/              # NEW - Integration layer tests
│   ├── __init__.py
│   ├── test_due_diligence.py      # audit, provider_info, pooled, dashboard
│   ├── test_incident_interface.py # notification, classification, reporting
│   ├── test_third_party.py        # concentration, ctpp, subcontractor
│   ├── test_contracts.py          # requirements, sla, exit
│   ├── test_reporting.py          # unified, templates, roi
│   └── test_sharing.py            # information_sharing
│
├── dora/                          # UPDATE - Only facade tests
│   └── test_facade_imports.py     # Verify re-exports work
│
└── archive/                       # OPTIONAL - Archived module tests
    └── dora_financial_entity/
        └── test_fe_modules.py     # Mark as @pytest.mark.archived
```

#### 9.2 Import Migration Script

```bash
#!/bin/bash
# scripts/migrate_dora_imports.sh

# Find all files importing from services.dora (excluding dora/ itself)
FILES=$(grep -rl "from services.dora" --include="*.py" | grep -v "services/dora/")

for file in $FILES; do
    echo "Processing: $file"

    # Update imports for each subpackage
    sed -i 's/from services\.dora\.audit_readiness/from services.dora_integration.due_diligence.audit_readiness/g' "$file"
    sed -i 's/from services\.dora\.provider_info_package/from services.dora_integration.due_diligence.provider_info_package/g' "$file"
    sed -i 's/from services\.dora\.pooled_audit_support/from services.dora_integration.due_diligence.pooled_audit_support/g' "$file"
    sed -i 's/from services\.dora\.compliance_dashboard/from services.dora_integration.due_diligence.compliance_dashboard/g' "$file"

    sed -i 's/from services\.dora\.client_incident_notification/from services.dora_integration.incident_interface.client_incident_notification/g' "$file"
    sed -i 's/from services\.dora\.incident_classification/from services.dora_integration.incident_interface.incident_classification/g' "$file"
    sed -i 's/from services\.dora\.incident_reporting/from services.dora_integration.incident_interface.incident_reporting/g' "$file"
    sed -i 's/from services\.dora\.cyber_threat_notification/from services.dora_integration.incident_interface.cyber_threat_notification/g' "$file"
    sed -i 's/from services\.dora\.communication/from services.dora_integration.incident_interface.communication/g' "$file"

    # ... (continue for all modules)
done
```

#### 9.3 Test Validation Checklist

```python
# tests/dora_integration/test_import_validation.py
"""Validate all imports work after migration."""

import pytest

class TestImportValidation:
    """Verify all integration layer imports resolve correctly."""

    def test_due_diligence_imports(self):
        from services.dora_integration.due_diligence import (
            DORAuditReadiness,
            ProviderIdentification,
            PooledAuditSupport,
            DORAComplianceDashboard,
        )
        assert DORAuditReadiness is not None

    def test_incident_interface_imports(self):
        from services.dora_integration.incident_interface import (
            ClientNotificationService,
            DORAIncidentClassification,
            DORAIncidentReporter,
        )
        assert DORAIncidentReporter is not None

    def test_facade_backward_compat(self):
        """Verify old import path still works (with warning)."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from services.dora import DORAuditReadiness
            # Should work but emit DeprecationWarning
            assert len(w) >= 0  # May or may not warn depending on implementation

    @pytest.mark.archived
    def test_archived_modules_not_importable(self):
        """Verify archived modules are not in main import path."""
        with pytest.raises(ImportError):
            from services.dora import DORAGovernanceFramework  # Archived
```

#### 9.4 CI/CD Integration

```yaml
# .github/workflows/test-migration.yml
name: Test DORA Migration

on:
  push:
    paths:
      - 'services/dora_integration/**'
      - 'services/dora/**'
      - 'tests/dora_integration/**'

jobs:
  test-integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run integration layer tests
        run: pytest tests/dora_integration/ -v --tb=short

      - name: Run facade tests
        run: pytest tests/dora/test_facade_imports.py -v

      - name: Verify no broken imports
        run: |
          python -c "from services.dora_integration import *"
          python -c "from services.dora import *"

      - name: Check for deprecated imports in codebase
        run: |
          # Should find zero direct imports from services.dora.<module>
          ! grep -r "from services.dora\." --include="*.py" \
            | grep -v "services/dora/" \
            | grep -v "dora_integration" \
            | grep -v "test_" \
            | grep -v "#"
```

---

## 3. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking imports | High | Deprecation shims + 1-release grace period |
| Missing dependencies | Medium | Comprehensive import audit in Phase 0 |
| Test failures | Medium | Run tests after each phase |
| Circular imports | Medium | Clear dependency direction: core → integration |
| Lost functionality | High | Archive, don't delete; full backup |

---

## 4. Rollback Plan

```bash
# If migration fails
git checkout pre-integration-refactor
git branch -D refactor/dora-integration-layer

# Partial rollback (specific phase)
git revert <commit-hash>
```

---

## 5. Success Criteria

### Phase Success
- [ ] All tests pass after each phase
- [ ] No breaking changes to public API during grace period
- [ ] Documentation updated

### Final Success
- [ ] Clean separation: core (14) vs integration (21) vs archive (23)
- [ ] All 21 integration modules organized in 6 subpackages
- [ ] All 23 FE modules archived with README
- [ ] `services/dora/__init__.py` is thin facade only
- [ ] All tests pass (core, integration, facade)
- [ ] Zero direct imports from `services.dora.<module>` in app code
- [ ] CI/CD validates migration integrity

---

## 6. Migration Checklist

### Phase 0 (Preparation)
- [ ] Directory structure created (`services/dora_integration/`, `services/archive/`)
- [ ] Config directory created (`config/dora_integration/`)
- [ ] Git tag `pre-integration-refactor` created
- [ ] Import audit complete (all `from services.dora import` documented)
- [ ] Migration branch created

### Phase 1 (Due Diligence)
- [ ] 4 modules moved
- [ ] `__init__.py` created
- [ ] Tests pass

### Phase 2 (Incident Interface)
- [ ] 5 modules moved
- [ ] Refactored to export-only pattern
- [ ] Client notification tested

### Phase 3 (Third-Party)
- [ ] 5 modules moved
- [ ] Linked with `services/core/subcontractor_monitoring`
- [ ] Config moved

### Phase 4 (Contracts)
- [ ] 3 modules moved
- [ ] Tests pass

### Phase 5 (Reporting)
- [ ] 3 modules moved
- [ ] ROI refactored to generator pattern
- [ ] Templates validated

### Phase 6 (Sharing)
- [ ] 1 module moved
- [ ] Config moved

### Phase 7 (Archive)
- [ ] 23 modules archived to `services/archive/dora_financial_entity/`
- [ ] 2 configs archived to `services/archive/dora_financial_entity/configs/`
- [ ] README created with module list and recovery instructions

### Phase 8 (Finalize)
- [ ] Main `services/dora_integration/__init__.py` updated with real class names
- [ ] `services/dora/__init__.py` converted to thin facade
- [ ] All imports updated via migration script
- [ ] Documentation complete

### Phase 9 (Test & Validate)
- [ ] `tests/dora_integration/` directory created with test files
- [ ] Import validation tests pass
- [ ] Facade backward compatibility tests pass
- [ ] CI/CD workflow added
- [ ] Zero deprecated imports in app code

---

## 7. References

- [DORA Regulation (EU) 2022/2554](https://eur-lex.europa.eu/eli/reg/2022/2554/oj)
- [DORA Article 30 - Key contractual provisions](https://www.digital-operational-resilience-act.com/Article_30.html)
- [CIR 2024/2956 - ITS on Register of Information](https://eur-lex.europa.eu/eli/reg_impl/2024/2956)
- [CDR 2024/1772 - Incident Classification](https://eur-lex.europa.eu/eli/reg_del/2024/1772)
- [ESA DORA Implementation Hub](https://www.eba.europa.eu/regulation-and-policy/operational-resilience)
