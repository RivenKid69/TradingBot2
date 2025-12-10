# MiFID II Module Migration Plan: ICT Provider Restructure

## Context
- Entity Type: ICT Provider / Software Provider (NOT Investment Firm)
- MiFID II: Does NOT apply directly
- Goal: Separate modules into CORE (all users), INTEGRATION (B2B enterprise), ARCHIVE (FE-specific)

---

## 1. Target Directory Structure

```
services/
├── core/
│   └── risk_controls/           # 🟢 CORE - universal risk controls
│       ├── __init__.py
│       ├── audit_models.py      # was: compliance/audit_models.py
│       ├── audit_storage.py     # was: compliance/audit_storage.py
│       ├── retention_policy.py  # was: compliance/retention_policy.py
│       ├── audit_trail_writer.py
│       ├── compliance_clock.py  # rename: time_sync.py
│       ├── kill_switch.py       # was: enhanced_kill_switch.py
│       ├── pre_trade_controls.py
│       ├── realtime_monitor.py
│       ├── bcp.py
│       └── config.py
│
├── algo_integration/            # 🟡 INTEGRATION - B2B compliance toolkit
│   ├── __init__.py
│   ├── best_execution.py
│   ├── tca_compliance.py
│   ├── venue_analysis.py
│   ├── execution_quality_report.py
│   ├── otr_monitor.py
│   ├── algorithm_registry.py
│   ├── conformance_testing.py
│   └── test_scenarios.py
│
├── archive/
│   └── mifid_financial_entity/  # 🔴 ARCHIVE - FE modules
│       ├── __init__.py
│       ├── lei_manager.py
│       ├── gleif_client.py
│       ├── transaction_report.py
│       ├── arm_client.py
│       ├── reporting_pipeline.py
│       ├── self_assessment.py
│       ├── governance.py
│       ├── compliance_policies.py
│       ├── certification.py
│       └── nca_notification.py
│
├── compliance/                  # FACADE (backward compat)
│   └── __init__.py              # Re-exports with deprecation warnings
```

---

## 2. File Migration Map

### 2.1 CORE Modules (services/core/risk_controls/)

| # | Old Path | New Path | Rename |
|---|----------|----------|--------|
| 1 | compliance/audit_models.py | core/risk_controls/audit_models.py | No |
| 2 | compliance/audit_storage.py | core/risk_controls/audit_storage.py | No |
| 3 | compliance/retention_policy.py | core/risk_controls/retention_policy.py | No |
| 4 | compliance/audit_trail_writer.py | core/risk_controls/audit_trail_writer.py | No |
| 5 | compliance/compliance_clock.py | core/risk_controls/time_sync.py | Yes |
| 6 | compliance/enhanced_kill_switch.py | core/risk_controls/kill_switch.py | Yes |
| 7 | compliance/pre_trade_controls.py | core/risk_controls/pre_trade_controls.py | No |
| 8 | compliance/realtime_monitor.py | core/risk_controls/realtime_monitor.py | No |
| 9 | compliance/bcp.py | core/risk_controls/bcp.py | No |
| 10 | compliance/config.py | core/risk_controls/config.py | No |

### 2.2 INTEGRATION Modules (services/algo_integration/)

| # | Old Path | New Path |
|---|----------|----------|
| 11 | compliance/best_execution.py | algo_integration/best_execution.py |
| 12 | compliance/tca_compliance.py | algo_integration/tca_compliance.py |
| 13 | compliance/venue_analysis.py | algo_integration/venue_analysis.py |
| 14 | compliance/execution_quality_report.py | algo_integration/execution_quality_report.py |
| 15 | compliance/otr_monitor.py | algo_integration/otr_monitor.py |
| 16 | compliance/algorithm_registry.py | algo_integration/algorithm_registry.py |
| 17 | compliance/conformance_testing.py | algo_integration/conformance_testing.py |
| 18 | compliance/test_scenarios.py | algo_integration/test_scenarios.py |

### 2.3 ARCHIVE Modules (services/archive/mifid_financial_entity/)

| # | Old Path | New Path |
|---|----------|----------|
| 19 | compliance/lei_manager.py | archive/mifid_financial_entity/lei_manager.py |
| 20 | compliance/gleif_client.py | archive/mifid_financial_entity/gleif_client.py |
| 21 | compliance/transaction_report.py | archive/mifid_financial_entity/transaction_report.py |
| 22 | compliance/arm_client.py | archive/mifid_financial_entity/arm_client.py |
| 23 | compliance/reporting_pipeline.py | archive/mifid_financial_entity/reporting_pipeline.py |
| 24 | compliance/self_assessment.py | archive/mifid_financial_entity/self_assessment.py |
| 25 | compliance/governance.py | archive/mifid_financial_entity/governance.py |
| 26 | compliance/compliance_policies.py | archive/mifid_financial_entity/compliance_policies.py |
| 27 | compliance/certification.py | archive/mifid_financial_entity/certification.py |
| 28 | compliance/nca_notification.py | archive/mifid_financial_entity/nca_notification.py |

---

## 3. Dependency Graph & Migration Order

```
Phase 1 - No Dependencies (parallel):
  ├── audit_models (CORE)
  ├── compliance_clock -> time_sync (CORE)
  ├── enhanced_kill_switch -> kill_switch (CORE)
  ├── pre_trade_controls (CORE)
  ├── realtime_monitor (CORE)
  ├── bcp (CORE)
  ├── config (CORE)
  ├── best_execution (INTEGRATION)
  ├── tca_compliance (INTEGRATION)
  ├── otr_monitor (INTEGRATION)
  ├── algorithm_registry (INTEGRATION)
  ├── conformance_testing (INTEGRATION)
  ├── lei_manager (ARCHIVE)
  ├── transaction_report (ARCHIVE)
  ├── arm_client (ARCHIVE)
  ├── self_assessment (ARCHIVE)
  ├── governance (ARCHIVE)
  ├── compliance_policies (ARCHIVE)
  └── nca_notification (ARCHIVE)

Phase 2 - Single Dependency:
  ├── audit_storage (CORE) <- audit_models
  ├── gleif_client (ARCHIVE) <- lei_manager
  └── venue_analysis (INTEGRATION) <- best_execution

Phase 3 - Multiple Dependencies:
  ├── retention_policy (CORE) <- audit_models, audit_storage
  ├── test_scenarios (INTEGRATION) <- conformance_testing
  ├── execution_quality_report (INTEGRATION) <- best_execution, venue_analysis, tca_compliance
  ├── reporting_pipeline (ARCHIVE) <- transaction_report, arm_client
  └── certification (ARCHIVE) <- conformance_testing

Phase 4 - Final:
  └── audit_trail_writer (CORE) <- audit_models, audit_storage, retention_policy
```

---

## 4. Import Updates

### 4.1 Internal Imports (within moved modules)

**audit_storage.py:**
```python
# OLD
from services.compliance.audit_models import AuditRecord, AuditEventType
# NEW
from services.core.risk_controls.audit_models import AuditRecord, AuditEventType
```

**retention_policy.py:**
```python
# OLD
from services.compliance.audit_models import AuditRecord
from services.compliance.audit_storage import AuditStorageBackend
# NEW
from services.core.risk_controls.audit_models import AuditRecord
from services.core.risk_controls.audit_storage import AuditStorageBackend
```

**audit_trail_writer.py:**
```python
# OLD
from services.compliance.audit_models import AuditRecord, AuditRecordBuilder
from services.compliance.audit_storage import AuditStorageBackend, create_audit_storage
from services.compliance.retention_policy import RetentionManager
# NEW
from services.core.risk_controls.audit_models import AuditRecord, AuditRecordBuilder
from services.core.risk_controls.audit_storage import AuditStorageBackend, create_audit_storage
from services.core.risk_controls.retention_policy import RetentionManager
```

**venue_analysis.py:**
```python
# OLD
from services.compliance.best_execution import BestExecutionAnalyzer
# NEW
from services.algo_integration.best_execution import BestExecutionAnalyzer
```

**execution_quality_report.py:**
```python
# OLD
from services.compliance.best_execution import BestExecutionAnalyzer
from services.compliance.venue_analysis import VenueAnalyzer
from services.compliance.tca_compliance import TCAComplianceWrapper
# NEW
from services.algo_integration.best_execution import BestExecutionAnalyzer
from services.algo_integration.venue_analysis import VenueAnalyzer
from services.algo_integration.tca_compliance import TCAComplianceWrapper
```

**test_scenarios.py:**
```python
# OLD
from services.compliance.conformance_testing import ConformanceTestRunner
# NEW
from services.algo_integration.conformance_testing import ConformanceTestRunner
```

**gleif_client.py:**
```python
# OLD
from services.compliance.lei_manager import LEIManager
# NEW
from services.archive.mifid_financial_entity.lei_manager import LEIManager
```

**reporting_pipeline.py:**
```python
# OLD
from services.compliance.transaction_report import TransactionReport
from services.compliance.arm_client import ARMClient
# NEW
from services.archive.mifid_financial_entity.transaction_report import TransactionReport
from services.archive.mifid_financial_entity.arm_client import ARMClient
```

**certification.py:**
```python
# OLD
from services.compliance.conformance_testing import ConformanceTestSuite
# NEW
from services.algo_integration.conformance_testing import ConformanceTestSuite
```

---

## 5. Test Migration Map

| Old Test Path | New Test Path |
|---------------|---------------|
| tests/test_mifid_phase4_audit_models.py | tests/core/test_audit_models.py |
| tests/test_mifid_phase4_audit_storage.py | tests/core/test_audit_storage.py |
| tests/test_mifid_phase4_retention_policy.py | tests/core/test_retention_policy.py |
| tests/test_mifid_phase4_audit_trail_writer.py | tests/core/test_audit_trail_writer.py |
| tests/test_mifid_compliance_clock.py | tests/core/test_time_sync.py |
| tests/test_mifid_phase3_enhanced_kill_switch.py | tests/core/test_kill_switch.py |
| tests/test_mifid_phase3_pre_trade_controls.py | tests/core/test_pre_trade_controls.py |
| tests/test_mifid_phase3_realtime_monitor.py | tests/core/test_realtime_monitor.py |
| tests/test_mifid_phase6_bcp.py | tests/core/test_bcp.py |
| tests/test_mifid_compliance_config.py | tests/core/test_config.py |
| tests/test_mifid_phase5_best_execution.py | tests/algo_integration/test_best_execution.py |
| tests/test_mifid_phase5_tca_compliance.py | tests/algo_integration/test_tca_compliance.py |
| tests/test_mifid_phase5_venue_analysis.py | tests/algo_integration/test_venue_analysis.py |
| tests/test_mifid_phase5_execution_quality_report.py | tests/algo_integration/test_execution_quality_report.py |
| tests/test_mifid_phase3_otr_monitor.py | tests/algo_integration/test_otr_monitor.py |
| tests/test_mifid_compliance_registry.py | tests/algo_integration/test_algorithm_registry.py |
| tests/test_mifid_phase7_conformance_testing.py | tests/algo_integration/test_conformance_testing.py |
| tests/test_mifid_phase7_test_scenarios.py | tests/algo_integration/test_test_scenarios.py |
| tests/test_mifid_compliance_lei.py | tests/archive/mifid_fe/test_lei_manager.py |
| tests/test_mifid_compliance_gleif.py | tests/archive/mifid_fe/test_gleif_client.py |
| tests/test_mifid_compliance_transaction_report.py | tests/archive/mifid_fe/test_transaction_report.py |
| tests/test_mifid_compliance_arm_client.py | tests/archive/mifid_fe/test_arm_client.py |
| tests/test_mifid_compliance_reporting_pipeline.py | tests/archive/mifid_fe/test_reporting_pipeline.py |
| tests/test_mifid_phase6_self_assessment.py | tests/archive/mifid_fe/test_self_assessment.py |
| tests/test_mifid_phase6_governance.py | tests/archive/mifid_fe/test_governance.py |
| tests/test_mifid_phase7_certification.py | tests/archive/mifid_fe/test_certification.py |
| tests/test_mifid_phase7_nca_notification.py | tests/archive/mifid_fe/test_nca_notification.py |

---

## 6. Configuration Split

### 6.1 New Config Files

**configs/core/risk_controls.yaml:**
```yaml
# Core Risk Controls Configuration
# For ALL platform users (ICT Provider default settings)

risk_controls:
  enabled: true
  mode: production  # production, testing, disabled

  # Time Synchronization (was: clock)
  time_sync:
    ntp_servers:
      - "time.google.com"
      - "pool.ntp.org"
    max_offset_ms: 100.0
    sync_interval_seconds: 60

  # Kill Switch
  kill_switch:
    enabled: true
    cooldown_seconds: 300.0
    auto_recovery_enabled: false

  # Pre-Trade Controls
  pre_trade_controls:
    enabled: true
    price_collar_pct: 5.0
    max_order_value_eur: 1000000.0
    max_messages_per_second: 100

  # Real-Time Monitoring
  realtime_monitoring:
    enabled: true
    max_alert_delay_seconds: 5.0
    check_interval_seconds: 1.0

  # Audit Trail
  audit_trail:
    enabled: true
    writer_mode: "async"
    retention_years: 5

  # Business Continuity
  bcp:
    enabled: true
    rto_minutes: 60
    rpo_minutes: 15
```

**configs/algo_integration/compliance_toolkit.yaml:**
```yaml
# Algo Integration Compliance Toolkit
# Enterprise addon for B2B clients (financial institutions)

algo_integration:
  enabled: false  # Disabled by default, enable for enterprise

  # Best Execution Analysis
  best_execution:
    enabled: true
    policy_review_frequency_days: 365

  # TCA
  tca:
    enabled: true
    impact_model: "almgren_chriss"

  # Venue Analysis
  venue_analysis:
    enabled: true
    min_samples: 30
    rolling_window_days: 30

  # OTR Monitoring
  otr_monitoring:
    enabled: true
    otr_warning_threshold: 50.0

  # Algorithm Registry
  algorithm_registry:
    enabled: true
    auto_register: true

  # Conformance Testing
  conformance_testing:
    enabled: true
    default_test_environment: "uat"
```

**configs/archive/mifid_financial_entity.yaml:**
```yaml
# MiFID II Financial Entity Modules (ARCHIVED)
# These modules are for Investment Firms under MiFID II
# ICT Providers do NOT need these

_archived: true
_archive_reason: "Not applicable to ICT Providers"
_archive_date: "2025-01-17"

# Reference only - do not load by default
financial_entity:
  lei:
    enabled: false
  transaction_reporting:
    enabled: false
  nca_notification:
    enabled: false
  self_assessment:
    enabled: false
```

---

## 7. __init__.py Files

### 7.1 services/core/risk_controls/__init__.py

```python
"""
Core Risk Controls - Universal trading risk management.

For ALL platform users. ICT Provider baseline functionality.
"""

__version__ = "1.0.0"

# Audit Trail
from services.core.risk_controls.audit_models import (
    AuditEventType, AuditRecord, AuditRecordBuilder,
    create_order_submitted_record, create_risk_event_record,
)
from services.core.risk_controls.audit_storage import (
    AuditStorageBackend, create_audit_storage,
    MemoryAuditStorage, SQLiteAuditStorage,
)
from services.core.risk_controls.retention_policy import (
    RetentionManager, create_retention_manager,
)
from services.core.risk_controls.audit_trail_writer import (
    AuditTrailWriter, create_audit_trail_writer,
)

# Time Synchronization (was: compliance_clock)
from services.core.risk_controls.time_sync import (
    ComplianceClock, create_compliance_clock,
    ClockSyncStatus, ClockDriftSeverity,
)

# Kill Switch (was: enhanced_kill_switch)
from services.core.risk_controls.kill_switch import (
    EnhancedKillSwitch, create_enhanced_kill_switch,
    KillSwitchScope, KillSwitchState,
)

# Pre-Trade Controls
from services.core.risk_controls.pre_trade_controls import (
    PreTradeControls, create_pre_trade_controls,
    PreTradeCheckResult, RejectionReason,
)

# Real-Time Monitoring
from services.core.risk_controls.realtime_monitor import (
    RealTimeMonitor, create_realtime_monitor,
    ComplianceAlert, AlertSeverity,
)

# Business Continuity
from services.core.risk_controls.bcp import (
    BusinessContinuityPlan, create_business_continuity_plan,
)

# Config
from services.core.risk_controls.config import RiskControlsConfig

__all__ = [
    # Audit
    "AuditEventType", "AuditRecord", "AuditRecordBuilder",
    "AuditStorageBackend", "create_audit_storage",
    "RetentionManager", "AuditTrailWriter",
    # Time Sync
    "ComplianceClock", "ClockSyncStatus",
    # Kill Switch
    "EnhancedKillSwitch", "KillSwitchScope",
    # Pre-Trade
    "PreTradeControls", "PreTradeCheckResult",
    # Monitoring
    "RealTimeMonitor", "ComplianceAlert",
    # BCP
    "BusinessContinuityPlan",
    # Config
    "RiskControlsConfig",
]
```

### 7.2 services/algo_integration/__init__.py

```python
"""
Algo Integration - B2B Compliance Toolkit.

Enterprise addon for financial institution clients.
Provides MiFID II Article 27 (Best Execution) and RTS 6 tools.
"""

__version__ = "1.0.0"

# Best Execution (MiFID II Article 27)
from services.algo_integration.best_execution import (
    BestExecutionAnalyzer, BestExecutionPolicy,
    create_best_execution_analyzer,
)

# TCA
from services.algo_integration.tca_compliance import (
    TCAComplianceWrapper, create_tca_wrapper,
)

# Venue Analysis
from services.algo_integration.venue_analysis import (
    VenueAnalyzer, SmartOrderRouter,
    create_venue_analyzer, create_smart_order_router,
)

# Execution Quality Reports
from services.algo_integration.execution_quality_report import (
    ExecutionQualityReportGenerator, create_report_generator,
)

# OTR Monitoring
from services.algo_integration.otr_monitor import (
    OTRMonitor, create_otr_monitor,
)

# Algorithm Registry
from services.algo_integration.algorithm_registry import (
    AlgorithmRegistry, create_algorithm_registry,
)

# Conformance Testing (RTS 6 Article 5)
from services.algo_integration.conformance_testing import (
    ConformanceTestRunner, create_test_runner,
)

# Test Scenarios
from services.algo_integration.test_scenarios import (
    ScenarioExecutor, create_scenario_executor,
)

__all__ = [
    "BestExecutionAnalyzer", "BestExecutionPolicy",
    "TCAComplianceWrapper", "VenueAnalyzer", "SmartOrderRouter",
    "ExecutionQualityReportGenerator", "OTRMonitor",
    "AlgorithmRegistry", "ConformanceTestRunner", "ScenarioExecutor",
]
```

### 7.3 services/archive/mifid_financial_entity/__init__.py

```python
"""
MiFID II Financial Entity Modules (ARCHIVED).

These modules implement MiFID II requirements for INVESTMENT FIRMS:
- LEI Management (Article 26 MiFIR)
- Transaction Reporting (RTS 22)
- NCA Notification (Article 17(2))
- Self-Assessment (RTS 6 Article 9)

NOT APPLICABLE TO ICT PROVIDERS.

Use only if your clients are investment firms needing these tools.
"""

import warnings

__version__ = "1.0.0"
__archived__ = True
__archive_reason__ = "Not applicable to ICT Providers per MiFID II scope"

def _warn_archived():
    warnings.warn(
        "mifid_financial_entity modules are archived. "
        "These are for Investment Firms under MiFID II, not ICT Providers.",
        DeprecationWarning,
        stacklevel=3
    )

# Lazy imports with deprecation warning
def __getattr__(name):
    _warn_archived()
    if name in _EXPORTS:
        return _EXPORTS[name]
    raise AttributeError(f"module has no attribute '{name}'")

_EXPORTS = {}  # Populated on first access
```

### 7.4 services/compliance/__init__.py (FACADE)

```python
"""
DEPRECATED: Use services.core.risk_controls or services.algo_integration.

This module provides backward compatibility during migration.
All imports emit DeprecationWarning.

Migration Guide:
  OLD: from services.compliance import EnhancedKillSwitch
  NEW: from services.core.risk_controls import EnhancedKillSwitch

  OLD: from services.compliance import BestExecutionAnalyzer
  NEW: from services.algo_integration import BestExecutionAnalyzer

  OLD: from services.compliance import LEIManager
  NEW: from services.archive.mifid_financial_entity import LEIManager
"""

import warnings

__version__ = "8.0.0"  # Major version bump for migration
__deprecated__ = True

def _emit_deprecation(old_path: str, new_path: str):
    warnings.warn(
        f"Importing from services.compliance is deprecated. "
        f"Use {new_path} instead.",
        DeprecationWarning,
        stacklevel=3
    )

# CORE re-exports
from services.core.risk_controls import (
    AuditEventType, AuditRecord, AuditRecordBuilder,
    AuditStorageBackend, create_audit_storage,
    RetentionManager, AuditTrailWriter,
    ComplianceClock, ClockSyncStatus,  # time_sync alias
    EnhancedKillSwitch, KillSwitchScope,  # kill_switch alias
    PreTradeControls, PreTradeCheckResult,
    RealTimeMonitor, ComplianceAlert,
    BusinessContinuityPlan,
)

# INTEGRATION re-exports
from services.algo_integration import (
    BestExecutionAnalyzer, BestExecutionPolicy,
    TCAComplianceWrapper, VenueAnalyzer,
    ExecutionQualityReportGenerator, OTRMonitor,
    AlgorithmRegistry, ConformanceTestRunner,
)

# ARCHIVE re-exports (with extra warning)
from services.archive.mifid_financial_entity import (
    LEIManager, GLEIFClient,
    TransactionReport, ARMClient,
    NCANotificationManager,
)

# Emit deprecation on module import
_emit_deprecation("services.compliance", "services.core.risk_controls")
```

---

## 8. Execution Steps

### Phase 1: Setup (Day 1)

```bash
# 1.1 Create branch
git checkout -b refactor/mifid-ict-provider-migration

# 1.2 Create directories
mkdir -p services/core/risk_controls
mkdir -p services/algo_integration
mkdir -p services/archive/mifid_financial_entity
mkdir -p tests/core
mkdir -p tests/algo_integration
mkdir -p tests/archive/mifid_fe
mkdir -p configs/core
mkdir -p configs/algo_integration
mkdir -p configs/archive

# 1.3 Create __init__.py files
touch services/core/__init__.py
touch services/core/risk_controls/__init__.py
touch services/algo_integration/__init__.py
touch services/archive/mifid_financial_entity/__init__.py
touch tests/core/__init__.py
touch tests/algo_integration/__init__.py
touch tests/archive/__init__.py
touch tests/archive/mifid_fe/__init__.py

# 1.4 Verify
pytest --collect-only 2>/dev/null | head -20
```

### Phase 2: CORE Migration (Day 2)

```bash
# 2.1 Copy Phase 1 modules (no dependencies)
cp services/compliance/audit_models.py services/core/risk_controls/
cp services/compliance/compliance_clock.py services/core/risk_controls/time_sync.py
cp services/compliance/enhanced_kill_switch.py services/core/risk_controls/kill_switch.py
cp services/compliance/pre_trade_controls.py services/core/risk_controls/
cp services/compliance/realtime_monitor.py services/core/risk_controls/
cp services/compliance/bcp.py services/core/risk_controls/
cp services/compliance/config.py services/core/risk_controls/

# 2.2 Update imports in copied files (sed commands)
# time_sync.py: no internal deps
# kill_switch.py: no internal deps
# pre_trade_controls.py: no internal deps
# realtime_monitor.py: no internal deps

# 2.3 Copy Phase 2 modules
cp services/compliance/audit_storage.py services/core/risk_controls/

# 2.4 Update audit_storage.py imports
sed -i 's/services\.compliance\.audit_models/services.core.risk_controls.audit_models/g' \
    services/core/risk_controls/audit_storage.py

# 2.5 Copy Phase 3 modules
cp services/compliance/retention_policy.py services/core/risk_controls/

# 2.6 Update retention_policy.py imports
sed -i 's/services\.compliance\.audit_models/services.core.risk_controls.audit_models/g' \
    services/core/risk_controls/retention_policy.py
sed -i 's/services\.compliance\.audit_storage/services.core.risk_controls.audit_storage/g' \
    services/core/risk_controls/retention_policy.py

# 2.7 Copy Phase 4 modules
cp services/compliance/audit_trail_writer.py services/core/risk_controls/

# 2.8 Update audit_trail_writer.py imports
sed -i 's/services\.compliance\.audit_models/services.core.risk_controls.audit_models/g' \
    services/core/risk_controls/audit_trail_writer.py
sed -i 's/services\.compliance\.audit_storage/services.core.risk_controls.audit_storage/g' \
    services/core/risk_controls/audit_trail_writer.py
sed -i 's/services\.compliance\.retention_policy/services.core.risk_controls.retention_policy/g' \
    services/core/risk_controls/audit_trail_writer.py

# 2.9 Verify CORE
python -c "from services.core.risk_controls import EnhancedKillSwitch; print('OK')"
pytest tests/core/ -v --tb=short
```

### Phase 3: INTEGRATION Migration (Day 3)

```bash
# 3.1 Copy Phase 1 modules
cp services/compliance/best_execution.py services/algo_integration/
cp services/compliance/tca_compliance.py services/algo_integration/
cp services/compliance/otr_monitor.py services/algo_integration/
cp services/compliance/algorithm_registry.py services/algo_integration/
cp services/compliance/conformance_testing.py services/algo_integration/

# 3.2 Copy Phase 2 modules
cp services/compliance/venue_analysis.py services/algo_integration/

# 3.3 Update venue_analysis.py imports
sed -i 's/services\.compliance\.best_execution/services.algo_integration.best_execution/g' \
    services/algo_integration/venue_analysis.py

# 3.4 Copy Phase 3 modules
cp services/compliance/test_scenarios.py services/algo_integration/
cp services/compliance/execution_quality_report.py services/algo_integration/

# 3.5 Update imports
sed -i 's/services\.compliance\.conformance_testing/services.algo_integration.conformance_testing/g' \
    services/algo_integration/test_scenarios.py

sed -i 's/services\.compliance\.best_execution/services.algo_integration.best_execution/g' \
    services/algo_integration/execution_quality_report.py
sed -i 's/services\.compliance\.venue_analysis/services.algo_integration.venue_analysis/g' \
    services/algo_integration/execution_quality_report.py
sed -i 's/services\.compliance\.tca_compliance/services.algo_integration.tca_compliance/g' \
    services/algo_integration/execution_quality_report.py

# 3.6 Verify INTEGRATION
python -c "from services.algo_integration import BestExecutionAnalyzer; print('OK')"
pytest tests/algo_integration/ -v --tb=short
```

### Phase 4: ARCHIVE Migration (Day 4)

```bash
# 4.1 Copy all ARCHIVE modules
cp services/compliance/lei_manager.py services/archive/mifid_financial_entity/
cp services/compliance/gleif_client.py services/archive/mifid_financial_entity/
cp services/compliance/transaction_report.py services/archive/mifid_financial_entity/
cp services/compliance/arm_client.py services/archive/mifid_financial_entity/
cp services/compliance/reporting_pipeline.py services/archive/mifid_financial_entity/
cp services/compliance/self_assessment.py services/archive/mifid_financial_entity/
cp services/compliance/governance.py services/archive/mifid_financial_entity/
cp services/compliance/compliance_policies.py services/archive/mifid_financial_entity/
cp services/compliance/certification.py services/archive/mifid_financial_entity/
cp services/compliance/nca_notification.py services/archive/mifid_financial_entity/

# 4.2 Update internal imports
sed -i 's/services\.compliance\.lei_manager/services.archive.mifid_financial_entity.lei_manager/g' \
    services/archive/mifid_financial_entity/gleif_client.py

sed -i 's/services\.compliance\.transaction_report/services.archive.mifid_financial_entity.transaction_report/g' \
    services/archive/mifid_financial_entity/reporting_pipeline.py
sed -i 's/services\.compliance\.arm_client/services.archive.mifid_financial_entity.arm_client/g' \
    services/archive/mifid_financial_entity/reporting_pipeline.py

# certification.py depends on conformance_testing (INTEGRATION)
sed -i 's/services\.compliance\.conformance_testing/services.algo_integration.conformance_testing/g' \
    services/archive/mifid_financial_entity/certification.py

# 4.3 Verify ARCHIVE
python -c "from services.archive.mifid_financial_entity import lei_manager; print('OK')"
```

### Phase 5: Test Migration (Day 5)

```bash
# 5.1 Move CORE tests
mv tests/test_mifid_phase4_audit_models.py tests/core/test_audit_models.py
mv tests/test_mifid_phase4_audit_storage.py tests/core/test_audit_storage.py
mv tests/test_mifid_phase4_retention_policy.py tests/core/test_retention_policy.py
mv tests/test_mifid_phase4_audit_trail_writer.py tests/core/test_audit_trail_writer.py
mv tests/test_mifid_compliance_clock.py tests/core/test_time_sync.py
mv tests/test_mifid_phase3_enhanced_kill_switch.py tests/core/test_kill_switch.py
mv tests/test_mifid_phase3_pre_trade_controls.py tests/core/test_pre_trade_controls.py
mv tests/test_mifid_phase3_realtime_monitor.py tests/core/test_realtime_monitor.py
mv tests/test_mifid_phase6_bcp.py tests/core/test_bcp.py
mv tests/test_mifid_compliance_config.py tests/core/test_config.py

# 5.2 Update test imports (CORE)
find tests/core -name "*.py" -exec sed -i \
    's/services\.compliance\./services.core.risk_controls./g' {} \;

# 5.3 Move INTEGRATION tests
mv tests/test_mifid_phase5_best_execution.py tests/algo_integration/test_best_execution.py
mv tests/test_mifid_phase5_tca_compliance.py tests/algo_integration/test_tca_compliance.py
mv tests/test_mifid_phase5_venue_analysis.py tests/algo_integration/test_venue_analysis.py
mv tests/test_mifid_phase5_execution_quality_report.py tests/algo_integration/test_execution_quality_report.py
mv tests/test_mifid_phase3_otr_monitor.py tests/algo_integration/test_otr_monitor.py
mv tests/test_mifid_compliance_registry.py tests/algo_integration/test_algorithm_registry.py
mv tests/test_mifid_phase7_conformance_testing.py tests/algo_integration/test_conformance_testing.py
mv tests/test_mifid_phase7_test_scenarios.py tests/algo_integration/test_test_scenarios.py

# 5.4 Update test imports (INTEGRATION)
find tests/algo_integration -name "*.py" -exec sed -i \
    's/services\.compliance\./services.algo_integration./g' {} \;

# 5.5 Move ARCHIVE tests
mv tests/test_mifid_compliance_lei.py tests/archive/mifid_fe/test_lei_manager.py
mv tests/test_mifid_compliance_gleif.py tests/archive/mifid_fe/test_gleif_client.py
mv tests/test_mifid_compliance_transaction_report.py tests/archive/mifid_fe/test_transaction_report.py
mv tests/test_mifid_compliance_arm_client.py tests/archive/mifid_fe/test_arm_client.py
mv tests/test_mifid_compliance_reporting_pipeline.py tests/archive/mifid_fe/test_reporting_pipeline.py
mv tests/test_mifid_phase6_self_assessment.py tests/archive/mifid_fe/test_self_assessment.py
mv tests/test_mifid_phase6_governance.py tests/archive/mifid_fe/test_governance.py
mv tests/test_mifid_phase7_certification.py tests/archive/mifid_fe/test_certification.py
mv tests/test_mifid_phase7_nca_notification.py tests/archive/mifid_fe/test_nca_notification.py

# 5.6 Update test imports (ARCHIVE)
find tests/archive/mifid_fe -name "*.py" -exec sed -i \
    's/services\.compliance\./services.archive.mifid_financial_entity./g' {} \;

# 5.7 Run all tests
pytest tests/core tests/algo_integration tests/archive/mifid_fe -v
```

### Phase 6: Facade & Config (Day 6)

```bash
# 6.1 Write __init__.py files (from section 7)
# 6.2 Write config files (from section 6)
# 6.3 Update old services/compliance/__init__.py to be facade

# 6.4 Verify backward compat
python -c "from services.compliance import EnhancedKillSwitch" 2>&1 | grep -i deprecat

# 6.5 Run full test suite
pytest --tb=short
```

### Phase 7: Cleanup (Day 7)

```bash
# 7.1 Delete old files (ONLY after all tests pass)
# DO NOT delete services/compliance/__init__.py (facade)
rm services/compliance/audit_models.py
rm services/compliance/audit_storage.py
# ... etc

# 7.2 Final verification
pytest --tb=short

# 7.3 Commit
git add -A
git commit -m "refactor: migrate MiFID modules for ICT Provider architecture

BREAKING CHANGE: Module locations changed

- CORE (risk_controls): services/core/risk_controls/
- INTEGRATION (B2B toolkit): services/algo_integration/
- ARCHIVE (FE modules): services/archive/mifid_financial_entity/

Migration:
  OLD: from services.compliance import X
  NEW: from services.core.risk_controls import X  # or algo_integration/archive

Backward compat: services.compliance facade emits DeprecationWarning"
```

---

## 9. Validation Checklist

### 9.1 Post-Migration Tests

```bash
# All tests pass
pytest -v --tb=short

# Import verification
python -c "
from services.core.risk_controls import EnhancedKillSwitch
from services.core.risk_controls import AuditTrailWriter
from services.core.risk_controls import PreTradeControls
from services.algo_integration import BestExecutionAnalyzer
from services.algo_integration import OTRMonitor
from services.archive.mifid_financial_entity import LEIManager
print('All imports OK')
"

# Backward compat (should warn)
python -c "
import warnings
warnings.filterwarnings('error')
try:
    from services.compliance import EnhancedKillSwitch
except DeprecationWarning:
    print('Deprecation warning works')
"

# Config loading
python -c "
import yaml
with open('configs/core/risk_controls.yaml') as f:
    cfg = yaml.safe_load(f)
    assert cfg['risk_controls']['enabled'] == True
    print('Config OK')
"
```

### 9.2 Acceptance Criteria

- [ ] All 28 modules migrated to correct locations
- [ ] All tests pass (0 failures)
- [ ] CORE modules importable from services.core.risk_controls
- [ ] INTEGRATION modules importable from services.algo_integration
- [ ] ARCHIVE modules importable from services.archive.mifid_financial_entity
- [ ] Backward compat facade emits DeprecationWarning
- [ ] Config files created and loadable
- [ ] No MiFID/RTS terminology in CORE module names
- [ ] Documentation updated

---

## 10. Usage Examples

### 10.1 CORE - All Users

```python
# Risk controls for any algo trading platform user
from services.core.risk_controls import (
    EnhancedKillSwitch,
    PreTradeControls,
    RealTimeMonitor,
    AuditTrailWriter,
)

# Initialize
kill_switch = EnhancedKillSwitch()
pre_trade = PreTradeControls(max_order_value=1_000_000)
monitor = RealTimeMonitor()

# Use
result = pre_trade.check_order(order)
if not result.passed:
    kill_switch.trigger(reason=result.rejection_reason)
```

### 10.2 INTEGRATION - Enterprise B2B

```python
# Enable for financial institution clients
from services.algo_integration import (
    BestExecutionAnalyzer,
    VenueAnalyzer,
    OTRMonitor,
    ConformanceTestRunner,
)

# Best execution for Article 27 compliance
analyzer = BestExecutionAnalyzer()
report = analyzer.analyze_executions(trades)

# OTR monitoring for RTS 6
otr = OTRMonitor()
otr.record_order(order)
metrics = otr.get_metrics()
```

### 10.3 ARCHIVE - Not Loaded by Default

```python
# Only if client is an Investment Firm needing MiFID II
# DO NOT import by default

if client.is_investment_firm:
    from services.archive.mifid_financial_entity import (
        LEIManager,
        TransactionReport,
        NCANotificationManager,
    )

    lei_mgr = LEIManager()
    report = TransactionReport.build(trade)
```

---

## 11. Documentation Updates

Files to update:
- README.md: Update architecture section
- docs/compliance/: Reorganize per new structure
- docs/api/: Update import paths
- CHANGELOG.md: Document breaking change

Key messaging change:
- OLD: "MiFID II Compliance Module"
- NEW: "Risk Controls & Algo Trading Tools for ICT Providers"
