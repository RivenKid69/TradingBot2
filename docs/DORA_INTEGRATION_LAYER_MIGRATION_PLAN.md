# DORA Integration Layer Migration Plan

**Version:** 1.0
**Date:** 2025-01-17
**Status:** Draft

---

## Executive Summary

Данный документ описывает план реорганизации DORA-модулей платформы с разделением на:

1. **CORE** (`services/core/`) — операционная устойчивость провайдера (НЕ ТРОГАЕМ)
2. **INTEGRATION** (`services/dora_integration/`) — интерфейс взаимодействия с клиентами
3. **ARCHIVE** (`archive/dora_financial_entity/`) — модули для финансовых организаций

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
├── dora/                          # ОСТАЁТСЯ - Internal DORA modules
│   └── (модули, которые относятся к внутренней работе)
│
configs/
├── dora/                          # Конфиги
│   ├── proportionality_assessment.yaml   # KEEP - internal toggle
│   ├── digital_resilience_strategy.yaml  # MOVE → dora_integration/
│   ├── third_party_management.yaml       # MOVE → dora_integration/
│   └── information_sharing.yaml          # MOVE → dora_integration/
│
├── dora_integration/              # NEW - Integration configs
│   ├── digital_resilience_strategy.yaml
│   ├── third_party_management.yaml
│   └── information_sharing.yaml
│
archive/
└── dora_financial_entity/         # Archive - модули для FE (не ICT provider)
    ├── README.md
    ├── entity_classification.yaml
    ├── nca_identification.yaml
    ├── scope_verification.py
    ├── function_classification.py
    ├── proportionality.py
    ├── governance.py
    ├── ict_risk_framework.py
    └── ... (full FE framework)
```

---

## 2. Migration Phases

### Phase 0: Preparation (Pre-requisites)
**Duration:** 1-2 days
**Risk:** Low

| Task | Action | Files |
|------|--------|-------|
| 0.1 | Create directory structure | `services/dora_integration/`, subfolders |
| 0.2 | Create archive directory | `archive/dora_financial_entity/` |
| 0.3 | Backup current state | `git tag pre-integration-refactor` |
| 0.4 | Document current imports | Audit all `from services.dora import` |
| 0.5 | Create migration branch | `git checkout -b refactor/dora-integration-layer` |

**Commands:**
```bash
mkdir -p services/dora_integration/{due_diligence,incident_interface,third_party,contracts,reporting,sharing}
mkdir -p configs/dora_integration
mkdir -p archive/dora_financial_entity
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

from services.dora_integration.due_diligence.audit_readiness import (
    DORAuditReadiness,
    AuditRequest,
    EvidenceItem,
    create_audit_readiness,
)

from services.dora_integration.due_diligence.provider_info_package import (
    ProviderInfoPackage,
    ProviderIdentification,
    ICTServiceType,
    generate_roi_data_package,
)

from services.dora_integration.due_diligence.pooled_audit_support import (
    PooledAuditSupport,
    PooledAuditEngagement,
    create_pooled_audit_support,
)

from services.dora_integration.due_diligence.compliance_dashboard import (
    DORAComplianceDashboard,
    ComplianceStatus,
)

__all__ = [
    "DORAuditReadiness", "AuditRequest", "EvidenceItem", "create_audit_readiness",
    "ProviderInfoPackage", "ProviderIdentification", "ICTServiceType", "generate_roi_data_package",
    "PooledAuditSupport", "PooledAuditEngagement", "create_pooled_audit_support",
    "DORAComplianceDashboard", "ComplianceStatus",
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

#### 2.2 Refactor Focus

**Ключевое изменение:** Модули должны работать как **экспорт данных клиенту**, а не как полная система управления инцидентами:

```python
# BEFORE (in services/dora/incident_reporting.py)
class DORAIncidentReporter:
    def submit_to_nca(self, report):  # WRONG - we don't submit
        ...

# AFTER (in services/dora_integration/incident_interface/)
class IncidentDataExporter:
    def generate_client_report(self, incident) -> ClientIncidentReport:
        """Generate data package for client's NCA submission."""
        ...

    def export_json(self, incident) -> str:
        """Export incident data in DORA-compliant JSON."""
        ...

    def export_dpm_format(self, incident) -> dict:
        """Export in DPM taxonomy format for client ROI."""
        ...
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

from services.dora_integration.incident_interface.client_incident_notification import (
    ClientIncidentNotification,
    NotificationStatus,
    notify_client,
)

from services.dora_integration.incident_interface.incident_classification import (
    DORAIncidentClassification,
    IncidentClassificationResult,
    classify_incident,
)

from services.dora_integration.incident_interface.incident_reporting import (
    IncidentDataExporter,
    ClientIncidentReport,
    export_incident_data,
)

__all__ = [...]
```

#### 2.4 Integration Points

```yaml
# configs/dora_integration/incident_notification.yaml
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
mv configs/dora/third_party_management.yaml configs/dora_integration/third_party_management.yaml
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

#### 6.2 Move Config

```bash
mv configs/dora/information_sharing.yaml configs/dora_integration/information_sharing.yaml
```

---

### Phase 7: Archive Financial Entity Modules
**Duration:** 1 day
**Risk:** Low
**Dependencies:** All phases complete

#### 7.1 Archive Modules

Эти модули предназначены для финансовых организаций, не для ICT-провайдера:

| Module | Reason |
|--------|--------|
| `scope_verification.py` | Art. 2 scope check — для FE |
| `function_classification.py` | Art. 3(22) — FE классифицирует свои функции |
| `proportionality.py` | Art. 4, 16 — FE определяет свой режим |
| `governance.py` | Art. 5 — FE governance, не наше |
| `ict_risk_framework.py` | Art. 6 — FE framework |
| `ict_systems.py` | Art. 7 — FE systems |
| `ict_identification.py` | Art. 8 — FE assets |
| `protection.py` | Art. 9 — FE protection |
| `detection.py` | Art. 10 — FE detection |
| `response_recovery.py` | Art. 11 — FE response |
| `backup_recovery.py` | Art. 12 — FE backup |
| `learning.py` | Art. 13 — FE learning |
| `ict_business_continuity.py` | Art. 15 — FE BCP |
| `simplified_framework.py` | Art. 16 — FE simplified |
| `incident_management.py` | Art. 17 — FE incident mgmt |
| `supervisory_feedback.py` | Art. 22 — FE ↔ NCA |
| `resilience_testing.py` | Art. 24 — FE testing |
| `ict_testing.py` | Art. 25 — FE testing |
| `tlpt.py` | Art. 26 — FE TLPT |
| `tester_management.py` | Art. 27 — FE testers |
| `pooled_testing.py` | Art. 26(3) — FE pooled |
| `cross_regulation.py` | FE cross-reg compliance |

#### 7.2 Archive Configs

| Config | Reason |
|--------|--------|
| `entity_classification.yaml` | FE classification |
| `nca_identification.yaml` | FE ↔ NCA mapping |

#### 7.3 Create Archive README

```markdown
# Archived DORA Financial Entity Modules

These modules implement DORA requirements for **Financial Entities** (Art. 2),
not for ICT Third-Party Service Providers (Art. 30).

## Why Archived?

As an ICT service provider, we:
- Comply with Art. 30 (contractual requirements)
- Support client due diligence (Art. 28)
- DO NOT implement full FE DORA framework

## When to Use?

If you're building a product FOR financial entities to manage their own
DORA compliance, these modules provide a reference implementation.

## Modules

- `scope_verification.py` - DORA scope determination
- `governance.py` - ICT governance framework
- ...
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

from services.dora_integration.due_diligence import (
    DORAuditReadiness,
    ProviderInfoPackage,
    PooledAuditSupport,
    DORAComplianceDashboard,
)

from services.dora_integration.incident_interface import (
    ClientIncidentNotification,
    DORAIncidentClassification,
    IncidentDataExporter,
)

from services.dora_integration.third_party import (
    DORASubcontractorManagement,
    ConcentrationRiskAssessor,
    CTPPOversightPrep,
)

from services.dora_integration.contracts import (
    ContractualRequirementsChecker,
    SLAGuardrails,
    ExitStrategyManager,
)

from services.dora_integration.reporting import (
    UnifiedReportingManager,
    ROIDataGenerator,
)

from services.dora_integration.sharing import (
    InformationSharingCoordinator,
)

__all__ = [...]
```

#### 8.2 Update `services/dora/__init__.py`

Оставляем только ссылки на core + интеграционный слой:

```python
# services/dora/__init__.py
"""
DORA Compliance Module.

ARCHITECTURE:
- services/core/: Operational resilience (logging, health, backup, DR)
- services/dora_integration/: Client-facing interfaces
- services/dora/: Internal utilities (if any remain)

For client-facing DORA functionality, use services.dora_integration.
"""

# Re-export integration layer for convenience
from services.dora_integration import *

# Internal utilities (if any)
# ...
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
- [ ] Clean separation: core vs integration
- [ ] All 21 integration modules organized
- [ ] All 22+ FE modules archived with README
- [ ] `services/dora/__init__.py` simplified
- [ ] Zero deprecation warnings in production

---

## 6. Migration Checklist

### Phase 0
- [ ] Directory structure created
- [ ] Git tag created
- [ ] Import audit complete

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
- [ ] 22+ modules archived
- [ ] 2 configs archived
- [ ] README created

### Phase 8 (Finalize)
- [ ] Main `__init__.py` updated
- [ ] All imports updated
- [ ] Full test suite passes
- [ ] Documentation complete

---

## 7. References

- [DORA Regulation (EU) 2022/2554](https://eur-lex.europa.eu/eli/reg/2022/2554/oj)
- [DORA Article 30 - Key contractual provisions](https://www.digital-operational-resilience-act.com/Article_30.html)
- [CIR 2024/2956 - ITS on Register of Information](https://eur-lex.europa.eu/eli/reg_impl/2024/2956)
- [CDR 2024/1772 - Incident Classification](https://eur-lex.europa.eu/eli/reg_del/2024/1772)
- [ESA DORA Implementation Hub](https://www.eba.europa.eu/regulation-and-policy/operational-resilience)
