# MiFID II Module Migration Plan V2: ICT Provider Restructure

## Revision Notes
- **V2**: Fixed based on real code analysis (grep, not assumptions)
- certification.py moved to INTEGRATION (was cross-group dependency)
- config.py split into 3 files (was monolithic)
- gleif_client has NO dependency on lei_manager (table was wrong)
- Atomic steps: module + test together
- Rollback checkpoints after each phase

---

## 1. VERIFIED Dependency Graph

Based on actual `grep "^from services.compliance"` analysis:

```
CORE (10 modules):
  audit_models        -> (none)
  compliance_clock    -> (none)
  enhanced_kill_switch-> (none)
  pre_trade_controls  -> (none)
  realtime_monitor    -> (none)
  bcp                 -> (none)
  config              -> (none) [SPLIT INTO 3]
  audit_storage       -> audit_models
  retention_policy    -> audit_models, audit_storage
  audit_trail_writer  -> audit_models, audit_storage, retention_policy

INTEGRATION (9 modules): [+certification from ARCHIVE]
  best_execution      -> (none)
  tca_compliance      -> (none)
  otr_monitor         -> (none)
  algorithm_registry  -> (none)
  conformance_testing -> (none)
  certification       -> conformance_testing [MOVED FROM ARCHIVE]
  venue_analysis      -> best_execution
  test_scenarios      -> conformance_testing
  execution_quality_report -> best_execution, venue_analysis, tca_compliance

ARCHIVE (9 modules): [certification removed]
  lei_manager         -> (none)
  gleif_client        -> (none) [NOT lei_manager!]
  transaction_report  -> (none)
  arm_client          -> (none)
  self_assessment     -> (none)
  governance          -> (none)
  compliance_policies -> (none)
  nca_notification    -> (none)
  reporting_pipeline  -> transaction_report, arm_client
```

---

## 2. Corrected Module Distribution

| # | Module | LOC | Group | Real Dependencies |
|---|--------|-----|-------|-------------------|
| 1 | audit_models | 920 | CORE | — |
| 2 | audit_storage | 1751 | CORE | audit_models |
| 3 | retention_policy | 918 | CORE | audit_models, audit_storage |
| 4 | audit_trail_writer | 941 | CORE | audit_models, audit_storage, retention_policy |
| 5 | compliance_clock | 647 | CORE | — |
| 6 | enhanced_kill_switch | 939 | CORE | — |
| 7 | pre_trade_controls | 1059 | CORE | — |
| 8 | realtime_monitor | 1199 | CORE | — |
| 9 | bcp | 1528 | CORE | — |
| 10 | config | 371 | CORE | — (split) |
| 11 | best_execution | 1371 | INTEGRATION | — |
| 12 | tca_compliance | 1010 | INTEGRATION | — |
| 13 | venue_analysis | 1092 | INTEGRATION | best_execution |
| 14 | execution_quality_report | 1123 | INTEGRATION | best_execution, venue_analysis, tca_compliance |
| 15 | otr_monitor | 1086 | INTEGRATION | — |
| 16 | algorithm_registry | 789 | INTEGRATION | — |
| 17 | conformance_testing | 1466 | INTEGRATION | — |
| 18 | test_scenarios | 1137 | INTEGRATION | conformance_testing |
| 19 | **certification** | 1080 | **INTEGRATION** | conformance_testing |
| 20 | lei_manager | 661 | ARCHIVE | — |
| 21 | gleif_client | 630 | ARCHIVE | — |
| 22 | transaction_report | 1309 | ARCHIVE | — |
| 23 | arm_client | 1009 | ARCHIVE | — |
| 24 | reporting_pipeline | 986 | ARCHIVE | transaction_report, arm_client |
| 25 | self_assessment | 1405 | ARCHIVE | — |
| 26 | governance | 1244 | ARCHIVE | — |
| 27 | compliance_policies | 1010 | ARCHIVE | — |
| 28 | nca_notification | 1233 | ARCHIVE | — |

**Changes from V1:**
- certification: ARCHIVE → INTEGRATION (fixes cross-group dependency)
- gleif_client: dependency on lei_manager REMOVED (was incorrect)
- config: will be SPLIT into 3 files

---

## 3. Config Split Strategy

Original `config.py` contains:
- `LEIConfig` → ARCHIVE (FE-specific)
- `ClockSyncComplianceConfig` → CORE
- `AlgorithmRegistryConfig` → INTEGRATION
- `PreTradeControlsConfig` → CORE
- `MiFIDIIComplianceConfig` → ARCHIVE (aggregates all, FE-specific)

**New structure:**

```python
# services/core/risk_controls/config.py
class TimeSyncConfig(BaseModel):  # renamed from ClockSyncComplianceConfig
    """Time synchronization configuration."""
    ...

class PreTradeControlsConfig(BaseModel):
    """Pre-trade risk controls configuration."""
    ...

class RiskControlsConfig(BaseModel):
    """Top-level config for core risk controls."""
    enabled: bool = True
    time_sync: TimeSyncConfig
    pre_trade: PreTradeControlsConfig
```

```python
# services/algo_integration/config.py
class AlgorithmRegistryConfig(BaseModel):
    """Algorithm registration configuration."""
    ...

class AlgoIntegrationConfig(BaseModel):
    """Top-level config for algo integration."""
    enabled: bool = False  # Disabled by default
    algorithm_registry: AlgorithmRegistryConfig
```

```python
# services/archive/mifid_financial_entity/config.py
class LEIConfig(BaseModel):
    """LEI configuration (FE-specific)."""
    ...

class MiFIDIIComplianceConfig(BaseModel):
    """Full MiFID II config for Investment Firms."""
    ...
```

---

## 4. Target Directory Structure

```
services/
├── core/
│   └── risk_controls/
│       ├── __init__.py
│       ├── config.py              # TimeSyncConfig, PreTradeControlsConfig
│       ├── audit_models.py
│       ├── audit_storage.py
│       ├── retention_policy.py
│       ├── audit_trail_writer.py
│       ├── time_sync.py           # renamed from compliance_clock
│       ├── kill_switch.py         # renamed from enhanced_kill_switch
│       ├── pre_trade_controls.py
│       ├── realtime_monitor.py
│       └── bcp.py
│
├── algo_integration/
│   ├── __init__.py
│   ├── config.py                  # AlgorithmRegistryConfig
│   ├── best_execution.py
│   ├── tca_compliance.py
│   ├── venue_analysis.py
│   ├── execution_quality_report.py
│   ├── otr_monitor.py
│   ├── algorithm_registry.py
│   ├── conformance_testing.py
│   ├── test_scenarios.py
│   └── certification.py           # MOVED HERE from ARCHIVE
│
├── archive/
│   └── mifid_financial_entity/
│       ├── __init__.py
│       ├── config.py              # LEIConfig, MiFIDIIComplianceConfig
│       ├── lei_manager.py
│       ├── gleif_client.py
│       ├── transaction_report.py
│       ├── arm_client.py
│       ├── reporting_pipeline.py
│       ├── self_assessment.py
│       ├── governance.py
│       ├── compliance_policies.py
│       └── nca_notification.py
│
└── compliance/
    └── __init__.py                # FACADE with deprecation warnings

tests/
├── core/
│   ├── __init__.py
│   └── test_*.py                  # 10 test files
├── algo_integration/
│   ├── __init__.py
│   └── test_*.py                  # 9 test files
└── archive/
    └── mifid_fe/
        ├── __init__.py
        └── test_*.py              # 9 test files
```

---

## 5. Migration Order (Topologically Sorted)

### Wave 1: No Dependencies (parallel safe)
```
CORE:     audit_models, compliance_clock, enhanced_kill_switch,
          pre_trade_controls, realtime_monitor, bcp
INTEG:    best_execution, tca_compliance, otr_monitor,
          algorithm_registry, conformance_testing
ARCHIVE:  lei_manager, gleif_client, transaction_report, arm_client,
          self_assessment, governance, compliance_policies, nca_notification
```

### Wave 2: Single Dependency
```
CORE:     audit_storage (← audit_models)
INTEG:    venue_analysis (← best_execution)
          certification (← conformance_testing)
          test_scenarios (← conformance_testing)
ARCHIVE:  (none)
```

### Wave 3: Multiple Dependencies
```
CORE:     retention_policy (← audit_models, audit_storage)
INTEG:    execution_quality_report (← best_execution, venue_analysis, tca_compliance)
ARCHIVE:  reporting_pipeline (← transaction_report, arm_client)
```

### Wave 4: Final
```
CORE:     audit_trail_writer (← audit_models, audit_storage, retention_policy)
```

### Wave 5: Config Split
```
Split config.py into 3 files after all modules migrated
```

---

## 6. Atomic Migration Steps

Each step = 1 module + 1 test + validation. Rollback checkpoint after each wave.

### PHASE 1: Setup

```bash
# Step 1.1: Create branch
git checkout -b refactor/mifid-ict-provider-migration-v2
git push -u origin refactor/mifid-ict-provider-migration-v2

# Step 1.2: Create directories
mkdir -p services/core/risk_controls
mkdir -p services/algo_integration
mkdir -p services/archive/mifid_financial_entity
mkdir -p tests/core tests/algo_integration tests/archive/mifid_fe

# Step 1.3: Create __init__.py stubs
for dir in services/core services/core/risk_controls services/algo_integration \
           services/archive services/archive/mifid_financial_entity \
           tests/core tests/algo_integration tests/archive tests/archive/mifid_fe; do
    echo '"""Package."""' > "$dir/__init__.py"
done

# Step 1.4: Commit setup
git add -A && git commit -m "chore: create directory structure for migration"

# CHECKPOINT 1
git tag checkpoint-1-setup
```

### PHASE 2: CORE Wave 1 (No Dependencies)

```bash
# --- Module: audit_models ---
# Step 2.1: Copy module
cp services/compliance/audit_models.py services/core/risk_controls/

# Step 2.2: Copy test
cp tests/test_mifid_phase4_audit_models.py tests/core/test_audit_models.py

# Step 2.3: Update test imports
sed -i 's/from services\.compliance\.audit_models/from services.core.risk_controls.audit_models/g' \
    tests/core/test_audit_models.py
sed -i 's/from services\.compliance import/from services.core.risk_controls import/g' \
    tests/core/test_audit_models.py

# Step 2.4: Validate
python -c "from services.core.risk_controls.audit_models import AuditRecord; print('OK')"
pytest tests/core/test_audit_models.py -v --tb=short

# Step 2.5: Commit atomically
git add services/core/risk_controls/audit_models.py tests/core/test_audit_models.py
git commit -m "refactor(core): migrate audit_models with test"

# --- Repeat for each Wave 1 module ---
# compliance_clock -> time_sync.py
cp services/compliance/compliance_clock.py services/core/risk_controls/time_sync.py
cp tests/test_mifid_compliance_clock.py tests/core/test_time_sync.py
sed -i 's/from services\.compliance\.compliance_clock/from services.core.risk_controls.time_sync/g' \
    tests/core/test_time_sync.py
pytest tests/core/test_time_sync.py -v --tb=short
git add services/core/risk_controls/time_sync.py tests/core/test_time_sync.py
git commit -m "refactor(core): migrate compliance_clock as time_sync"

# enhanced_kill_switch -> kill_switch.py
cp services/compliance/enhanced_kill_switch.py services/core/risk_controls/kill_switch.py
cp tests/test_mifid_phase3_enhanced_kill_switch.py tests/core/test_kill_switch.py
sed -i 's/from services\.compliance\.enhanced_kill_switch/from services.core.risk_controls.kill_switch/g' \
    tests/core/test_kill_switch.py
pytest tests/core/test_kill_switch.py -v --tb=short
git add services/core/risk_controls/kill_switch.py tests/core/test_kill_switch.py
git commit -m "refactor(core): migrate enhanced_kill_switch as kill_switch"

# pre_trade_controls
cp services/compliance/pre_trade_controls.py services/core/risk_controls/
cp tests/test_mifid_phase3_pre_trade_controls.py tests/core/test_pre_trade_controls.py
sed -i 's/from services\.compliance\.pre_trade_controls/from services.core.risk_controls.pre_trade_controls/g' \
    tests/core/test_pre_trade_controls.py
pytest tests/core/test_pre_trade_controls.py -v --tb=short
git add services/core/risk_controls/pre_trade_controls.py tests/core/test_pre_trade_controls.py
git commit -m "refactor(core): migrate pre_trade_controls"

# realtime_monitor
cp services/compliance/realtime_monitor.py services/core/risk_controls/
cp tests/test_mifid_phase3_realtime_monitor.py tests/core/test_realtime_monitor.py
sed -i 's/from services\.compliance\.realtime_monitor/from services.core.risk_controls.realtime_monitor/g' \
    tests/core/test_realtime_monitor.py
pytest tests/core/test_realtime_monitor.py -v --tb=short
git add services/core/risk_controls/realtime_monitor.py tests/core/test_realtime_monitor.py
git commit -m "refactor(core): migrate realtime_monitor"

# bcp
cp services/compliance/bcp.py services/core/risk_controls/
cp tests/test_mifid_phase6_bcp.py tests/core/test_bcp.py
sed -i 's/from services\.compliance\.bcp/from services.core.risk_controls.bcp/g' \
    tests/core/test_bcp.py
pytest tests/core/test_bcp.py -v --tb=short
git add services/core/risk_controls/bcp.py tests/core/test_bcp.py
git commit -m "refactor(core): migrate bcp"

# CHECKPOINT 2
git tag checkpoint-2-core-wave1
```

### PHASE 3: INTEGRATION Wave 1 (No Dependencies)

```bash
# best_execution
cp services/compliance/best_execution.py services/algo_integration/
cp tests/test_mifid_phase5_best_execution.py tests/algo_integration/test_best_execution.py
sed -i 's/from services\.compliance\.best_execution/from services.algo_integration.best_execution/g' \
    tests/algo_integration/test_best_execution.py
pytest tests/algo_integration/test_best_execution.py -v --tb=short
git add services/algo_integration/best_execution.py tests/algo_integration/test_best_execution.py
git commit -m "refactor(integration): migrate best_execution"

# tca_compliance
cp services/compliance/tca_compliance.py services/algo_integration/
cp tests/test_mifid_phase5_tca_compliance.py tests/algo_integration/test_tca_compliance.py
sed -i 's/from services\.compliance\.tca_compliance/from services.algo_integration.tca_compliance/g' \
    tests/algo_integration/test_tca_compliance.py
pytest tests/algo_integration/test_tca_compliance.py -v --tb=short
git add services/algo_integration/tca_compliance.py tests/algo_integration/test_tca_compliance.py
git commit -m "refactor(integration): migrate tca_compliance"

# otr_monitor
cp services/compliance/otr_monitor.py services/algo_integration/
cp tests/test_mifid_phase3_otr_monitor.py tests/algo_integration/test_otr_monitor.py
sed -i 's/from services\.compliance\.otr_monitor/from services.algo_integration.otr_monitor/g' \
    tests/algo_integration/test_otr_monitor.py
pytest tests/algo_integration/test_otr_monitor.py -v --tb=short
git add services/algo_integration/otr_monitor.py tests/algo_integration/test_otr_monitor.py
git commit -m "refactor(integration): migrate otr_monitor"

# algorithm_registry
cp services/compliance/algorithm_registry.py services/algo_integration/
cp tests/test_mifid_compliance_registry.py tests/algo_integration/test_algorithm_registry.py
sed -i 's/from services\.compliance\.algorithm_registry/from services.algo_integration.algorithm_registry/g' \
    tests/algo_integration/test_algorithm_registry.py
pytest tests/algo_integration/test_algorithm_registry.py -v --tb=short
git add services/algo_integration/algorithm_registry.py tests/algo_integration/test_algorithm_registry.py
git commit -m "refactor(integration): migrate algorithm_registry"

# conformance_testing
cp services/compliance/conformance_testing.py services/algo_integration/
cp tests/test_mifid_phase7_conformance_testing.py tests/algo_integration/test_conformance_testing.py
sed -i 's/from services\.compliance\.conformance_testing/from services.algo_integration.conformance_testing/g' \
    tests/algo_integration/test_conformance_testing.py
pytest tests/algo_integration/test_conformance_testing.py -v --tb=short
git add services/algo_integration/conformance_testing.py tests/algo_integration/test_conformance_testing.py
git commit -m "refactor(integration): migrate conformance_testing"

# CHECKPOINT 3
git tag checkpoint-3-integration-wave1
```

### PHASE 4: ARCHIVE Wave 1 (No Dependencies)

```bash
# lei_manager
cp services/compliance/lei_manager.py services/archive/mifid_financial_entity/
cp tests/test_mifid_compliance_lei.py tests/archive/mifid_fe/test_lei_manager.py
sed -i 's/from services\.compliance\.lei_manager/from services.archive.mifid_financial_entity.lei_manager/g' \
    tests/archive/mifid_fe/test_lei_manager.py
pytest tests/archive/mifid_fe/test_lei_manager.py -v --tb=short
git add services/archive/mifid_financial_entity/lei_manager.py tests/archive/mifid_fe/test_lei_manager.py
git commit -m "refactor(archive): migrate lei_manager"

# gleif_client (NO dependency on lei_manager!)
cp services/compliance/gleif_client.py services/archive/mifid_financial_entity/
cp tests/test_mifid_compliance_gleif.py tests/archive/mifid_fe/test_gleif_client.py
sed -i 's/from services\.compliance\.gleif_client/from services.archive.mifid_financial_entity.gleif_client/g' \
    tests/archive/mifid_fe/test_gleif_client.py
pytest tests/archive/mifid_fe/test_gleif_client.py -v --tb=short
git add services/archive/mifid_financial_entity/gleif_client.py tests/archive/mifid_fe/test_gleif_client.py
git commit -m "refactor(archive): migrate gleif_client"

# transaction_report
cp services/compliance/transaction_report.py services/archive/mifid_financial_entity/
cp tests/test_mifid_compliance_transaction_report.py tests/archive/mifid_fe/test_transaction_report.py
sed -i 's/from services\.compliance\.transaction_report/from services.archive.mifid_financial_entity.transaction_report/g' \
    tests/archive/mifid_fe/test_transaction_report.py
pytest tests/archive/mifid_fe/test_transaction_report.py -v --tb=short
git add services/archive/mifid_financial_entity/transaction_report.py tests/archive/mifid_fe/test_transaction_report.py
git commit -m "refactor(archive): migrate transaction_report"

# arm_client
cp services/compliance/arm_client.py services/archive/mifid_financial_entity/
cp tests/test_mifid_compliance_arm_client.py tests/archive/mifid_fe/test_arm_client.py
sed -i 's/from services\.compliance\.arm_client/from services.archive.mifid_financial_entity.arm_client/g' \
    tests/archive/mifid_fe/test_arm_client.py
sed -i 's/from services\.compliance\.transaction_report/from services.archive.mifid_financial_entity.transaction_report/g' \
    tests/archive/mifid_fe/test_arm_client.py
pytest tests/archive/mifid_fe/test_arm_client.py -v --tb=short
git add services/archive/mifid_financial_entity/arm_client.py tests/archive/mifid_fe/test_arm_client.py
git commit -m "refactor(archive): migrate arm_client"

# self_assessment
cp services/compliance/self_assessment.py services/archive/mifid_financial_entity/
cp tests/test_mifid_phase6_self_assessment.py tests/archive/mifid_fe/test_self_assessment.py
sed -i 's/from services\.compliance\.self_assessment/from services.archive.mifid_financial_entity.self_assessment/g' \
    tests/archive/mifid_fe/test_self_assessment.py
pytest tests/archive/mifid_fe/test_self_assessment.py -v --tb=short
git add services/archive/mifid_financial_entity/self_assessment.py tests/archive/mifid_fe/test_self_assessment.py
git commit -m "refactor(archive): migrate self_assessment"

# governance
cp services/compliance/governance.py services/archive/mifid_financial_entity/
cp tests/test_mifid_phase6_governance.py tests/archive/mifid_fe/test_governance.py
sed -i 's/from services\.compliance\.governance/from services.archive.mifid_financial_entity.governance/g' \
    tests/archive/mifid_fe/test_governance.py
sed -i 's/from services\.compliance\.compliance_policies/from services.archive.mifid_financial_entity.compliance_policies/g' \
    tests/archive/mifid_fe/test_governance.py
pytest tests/archive/mifid_fe/test_governance.py -v --tb=short
git add services/archive/mifid_financial_entity/governance.py tests/archive/mifid_fe/test_governance.py
git commit -m "refactor(archive): migrate governance"

# compliance_policies
cp services/compliance/compliance_policies.py services/archive/mifid_financial_entity/
# No dedicated test file for compliance_policies
git add services/archive/mifid_financial_entity/compliance_policies.py
git commit -m "refactor(archive): migrate compliance_policies"

# nca_notification
cp services/compliance/nca_notification.py services/archive/mifid_financial_entity/
cp tests/test_mifid_phase7_nca_notification.py tests/archive/mifid_fe/test_nca_notification.py
sed -i 's/from services\.compliance\.nca_notification/from services.archive.mifid_financial_entity.nca_notification/g' \
    tests/archive/mifid_fe/test_nca_notification.py
pytest tests/archive/mifid_fe/test_nca_notification.py -v --tb=short
git add services/archive/mifid_financial_entity/nca_notification.py tests/archive/mifid_fe/test_nca_notification.py
git commit -m "refactor(archive): migrate nca_notification"

# CHECKPOINT 4
git tag checkpoint-4-archive-wave1
```

### PHASE 5: Wave 2 (Single Dependencies)

```bash
# --- CORE: audit_storage (depends on audit_models) ---
cp services/compliance/audit_storage.py services/core/risk_controls/
cp tests/test_mifid_phase4_audit_storage.py tests/core/test_audit_storage.py

# Update module imports
sed -i 's/from services\.compliance\.audit_models/from services.core.risk_controls.audit_models/g' \
    services/core/risk_controls/audit_storage.py

# Update test imports
sed -i 's/from services\.compliance\.audit_models/from services.core.risk_controls.audit_models/g' \
    tests/core/test_audit_storage.py
sed -i 's/from services\.compliance\.audit_storage/from services.core.risk_controls.audit_storage/g' \
    tests/core/test_audit_storage.py

pytest tests/core/test_audit_storage.py -v --tb=short
git add services/core/risk_controls/audit_storage.py tests/core/test_audit_storage.py
git commit -m "refactor(core): migrate audit_storage"

# --- INTEGRATION: venue_analysis (depends on best_execution) ---
cp services/compliance/venue_analysis.py services/algo_integration/
cp tests/test_mifid_phase5_venue_analysis.py tests/algo_integration/test_venue_analysis.py

sed -i 's/from services\.compliance\.best_execution/from services.algo_integration.best_execution/g' \
    services/algo_integration/venue_analysis.py
sed -i 's/from services\.compliance\.venue_analysis/from services.algo_integration.venue_analysis/g' \
    tests/algo_integration/test_venue_analysis.py
sed -i 's/from services\.compliance\.best_execution/from services.algo_integration.best_execution/g' \
    tests/algo_integration/test_venue_analysis.py

pytest tests/algo_integration/test_venue_analysis.py -v --tb=short
git add services/algo_integration/venue_analysis.py tests/algo_integration/test_venue_analysis.py
git commit -m "refactor(integration): migrate venue_analysis"

# --- INTEGRATION: certification (depends on conformance_testing) ---
# NOTE: Moved from ARCHIVE to INTEGRATION!
cp services/compliance/certification.py services/algo_integration/
cp tests/test_mifid_phase7_certification.py tests/algo_integration/test_certification.py

sed -i 's/from services\.compliance\.conformance_testing/from services.algo_integration.conformance_testing/g' \
    services/algo_integration/certification.py
sed -i 's/from services\.compliance\.certification/from services.algo_integration.certification/g' \
    tests/algo_integration/test_certification.py
sed -i 's/from services\.compliance\.conformance_testing/from services.algo_integration.conformance_testing/g' \
    tests/algo_integration/test_certification.py

pytest tests/algo_integration/test_certification.py -v --tb=short
git add services/algo_integration/certification.py tests/algo_integration/test_certification.py
git commit -m "refactor(integration): migrate certification (moved from archive)"

# --- INTEGRATION: test_scenarios (depends on conformance_testing) ---
cp services/compliance/test_scenarios.py services/algo_integration/
cp tests/test_mifid_phase7_test_scenarios.py tests/algo_integration/test_test_scenarios.py

sed -i 's/from services\.compliance\.conformance_testing/from services.algo_integration.conformance_testing/g' \
    services/algo_integration/test_scenarios.py
sed -i 's/from services\.compliance\.test_scenarios/from services.algo_integration.test_scenarios/g' \
    tests/algo_integration/test_test_scenarios.py
sed -i 's/from services\.compliance\.conformance_testing/from services.algo_integration.conformance_testing/g' \
    tests/algo_integration/test_test_scenarios.py

pytest tests/algo_integration/test_test_scenarios.py -v --tb=short
git add services/algo_integration/test_scenarios.py tests/algo_integration/test_test_scenarios.py
git commit -m "refactor(integration): migrate test_scenarios"

# CHECKPOINT 5
git tag checkpoint-5-wave2
```

### PHASE 6: Wave 3 (Multiple Dependencies)

```bash
# --- CORE: retention_policy (depends on audit_models, audit_storage) ---
cp services/compliance/retention_policy.py services/core/risk_controls/
cp tests/test_mifid_phase4_retention_policy.py tests/core/test_retention_policy.py

sed -i 's/from services\.compliance\.audit_models/from services.core.risk_controls.audit_models/g' \
    services/core/risk_controls/retention_policy.py
sed -i 's/from services\.compliance\.audit_storage/from services.core.risk_controls.audit_storage/g' \
    services/core/risk_controls/retention_policy.py

sed -i 's/from services\.compliance\.audit_models/from services.core.risk_controls.audit_models/g' \
    tests/core/test_retention_policy.py
sed -i 's/from services\.compliance\.audit_storage/from services.core.risk_controls.audit_storage/g' \
    tests/core/test_retention_policy.py
sed -i 's/from services\.compliance\.retention_policy/from services.core.risk_controls.retention_policy/g' \
    tests/core/test_retention_policy.py

pytest tests/core/test_retention_policy.py -v --tb=short
git add services/core/risk_controls/retention_policy.py tests/core/test_retention_policy.py
git commit -m "refactor(core): migrate retention_policy"

# --- INTEGRATION: execution_quality_report ---
cp services/compliance/execution_quality_report.py services/algo_integration/
cp tests/test_mifid_phase5_execution_quality_report.py tests/algo_integration/test_execution_quality_report.py

sed -i 's/from services\.compliance\.best_execution/from services.algo_integration.best_execution/g' \
    services/algo_integration/execution_quality_report.py
sed -i 's/from services\.compliance\.venue_analysis/from services.algo_integration.venue_analysis/g' \
    services/algo_integration/execution_quality_report.py
sed -i 's/from services\.compliance\.tca_compliance/from services.algo_integration.tca_compliance/g' \
    services/algo_integration/execution_quality_report.py

sed -i 's/from services\.compliance\./from services.algo_integration./g' \
    tests/algo_integration/test_execution_quality_report.py

pytest tests/algo_integration/test_execution_quality_report.py -v --tb=short
git add services/algo_integration/execution_quality_report.py tests/algo_integration/test_execution_quality_report.py
git commit -m "refactor(integration): migrate execution_quality_report"

# --- ARCHIVE: reporting_pipeline ---
cp services/compliance/reporting_pipeline.py services/archive/mifid_financial_entity/
cp tests/test_mifid_compliance_reporting_pipeline.py tests/archive/mifid_fe/test_reporting_pipeline.py

sed -i 's/from services\.compliance\.transaction_report/from services.archive.mifid_financial_entity.transaction_report/g' \
    services/archive/mifid_financial_entity/reporting_pipeline.py
sed -i 's/from services\.compliance\.arm_client/from services.archive.mifid_financial_entity.arm_client/g' \
    services/archive/mifid_financial_entity/reporting_pipeline.py

sed -i 's/from services\.compliance\./from services.archive.mifid_financial_entity./g' \
    tests/archive/mifid_fe/test_reporting_pipeline.py

pytest tests/archive/mifid_fe/test_reporting_pipeline.py -v --tb=short
git add services/archive/mifid_financial_entity/reporting_pipeline.py tests/archive/mifid_fe/test_reporting_pipeline.py
git commit -m "refactor(archive): migrate reporting_pipeline"

# CHECKPOINT 6
git tag checkpoint-6-wave3
```

### PHASE 7: Wave 4 (Final Module)

```bash
# --- CORE: audit_trail_writer ---
cp services/compliance/audit_trail_writer.py services/core/risk_controls/
cp tests/test_mifid_phase4_audit_trail_writer.py tests/core/test_audit_trail_writer.py

sed -i 's/from services\.compliance\.audit_models/from services.core.risk_controls.audit_models/g' \
    services/core/risk_controls/audit_trail_writer.py
sed -i 's/from services\.compliance\.audit_storage/from services.core.risk_controls.audit_storage/g' \
    services/core/risk_controls/audit_trail_writer.py
sed -i 's/from services\.compliance\.retention_policy/from services.core.risk_controls.retention_policy/g' \
    services/core/risk_controls/audit_trail_writer.py

sed -i 's/from services\.compliance\./from services.core.risk_controls./g' \
    tests/core/test_audit_trail_writer.py

pytest tests/core/test_audit_trail_writer.py -v --tb=short
git add services/core/risk_controls/audit_trail_writer.py tests/core/test_audit_trail_writer.py
git commit -m "refactor(core): migrate audit_trail_writer"

# CHECKPOINT 7
git tag checkpoint-7-wave4
```

### PHASE 8: Config Split

```bash
# Step 8.1: Create CORE config
cat > services/core/risk_controls/config.py << 'PYEOF'
"""Core Risk Controls Configuration."""
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import List
from enum import Enum

class ControlsMode(str, Enum):
    PRODUCTION = "production"
    TESTING = "testing"
    DISABLED = "disabled"

class TimeSyncConfig(BaseModel):
    """Time synchronization configuration (was ClockSyncComplianceConfig)."""
    model_config = ConfigDict(extra="forbid")
    ntp_servers: List[str] = Field(default_factory=lambda: ["time.google.com", "pool.ntp.org"])
    max_offset_ms: float = Field(default=100.0, ge=0.001, le=1000.0)
    sync_interval_seconds: int = Field(default=60, ge=10, le=3600)
    warning_threshold_ms: float = Field(default=50.0, ge=0.0)
    critical_threshold_ms: float = Field(default=100.0, ge=0.0)
    kill_switch_threshold_ms: float = Field(default=1000.0, ge=100.0)

class PreTradeControlsConfig(BaseModel):
    """Pre-trade risk controls configuration."""
    model_config = ConfigDict(extra="forbid")
    price_collar_pct: float = Field(default=5.0, ge=0.1, le=50.0)
    max_order_value_eur: float = Field(default=1_000_000.0, ge=0.0)
    max_order_volume: float = Field(default=10_000.0, ge=0.0)
    max_messages_per_second: int = Field(default=100, ge=1, le=10_000)

class RiskControlsConfig(BaseModel):
    """Top-level config for core risk controls."""
    model_config = ConfigDict(extra="allow")
    enabled: bool = Field(default=True)
    mode: ControlsMode = Field(default=ControlsMode.TESTING)
    time_sync: TimeSyncConfig = Field(default_factory=TimeSyncConfig)
    pre_trade: PreTradeControlsConfig = Field(default_factory=PreTradeControlsConfig)
    audit_log_path: str = Field(default="logs/risk_controls/audit.log")

__all__ = ["ControlsMode", "TimeSyncConfig", "PreTradeControlsConfig", "RiskControlsConfig"]
PYEOF

# Step 8.2: Create INTEGRATION config
cat > services/algo_integration/config.py << 'PYEOF'
"""Algo Integration Configuration."""
from pydantic import BaseModel, Field, ConfigDict

class AlgorithmRegistryConfig(BaseModel):
    """Algorithm registration configuration."""
    model_config = ConfigDict(extra="forbid")
    registry_path: str = Field(default="state/algo_integration/algorithm_registry.json")
    auto_register: bool = Field(default=True)
    require_responsible_person: bool = Field(default=True)
    version_on_modification: bool = Field(default=True)
    firm_name: str = Field(default="")
    contact_email: str = Field(default="")

class AlgoIntegrationConfig(BaseModel):
    """Top-level config for algo integration (B2B toolkit)."""
    model_config = ConfigDict(extra="allow")
    enabled: bool = Field(default=False)  # Disabled by default for ICT Provider
    algorithm_registry: AlgorithmRegistryConfig = Field(default_factory=AlgorithmRegistryConfig)

__all__ = ["AlgorithmRegistryConfig", "AlgoIntegrationConfig"]
PYEOF

# Step 8.3: Create ARCHIVE config (keeps FE-specific stuff)
cat > services/archive/mifid_financial_entity/config.py << 'PYEOF'
"""MiFID II Financial Entity Configuration (ARCHIVED).

These configs are for Investment Firms under MiFID II.
Not applicable to ICT Providers.
"""
import warnings
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import List
from enum import Enum
import re

warnings.warn(
    "mifid_financial_entity.config is for Investment Firms, not ICT Providers",
    DeprecationWarning,
    stacklevel=2
)

class ComplianceMode(str, Enum):
    PRODUCTION = "production"
    TESTING = "testing"
    DISABLED = "disabled"

class LEIConfig(BaseModel):
    """LEI configuration for transaction reporting (FE-specific)."""
    model_config = ConfigDict(extra="forbid")
    own_lei: str = Field(default="", max_length=20)
    gleif_api_url: str = Field(default="https://api.gleif.org/api/v1")
    cache_ttl_hours: int = Field(default=24, ge=1, le=168)
    verify_before_trade: bool = Field(default=True)
    renewal_warning_days: int = Field(default=30, ge=7, le=90)

class MiFIDIIComplianceConfig(BaseModel):
    """Full MiFID II config for Investment Firms (ARCHIVED)."""
    model_config = ConfigDict(extra="allow")
    enabled: bool = Field(default=True)
    mode: ComplianceMode = Field(default=ComplianceMode.TESTING)
    lei: LEIConfig = Field(default_factory=LEIConfig)
    nca_jurisdiction: str = Field(default="")

__all__ = ["ComplianceMode", "LEIConfig", "MiFIDIIComplianceConfig"]
PYEOF

# Step 8.4: Update old config test
cp tests/test_mifid_compliance_config.py tests/core/test_config.py
# Update imports in test to use new config
sed -i 's/from services\.compliance\.config import/from services.core.risk_controls.config import/g' \
    tests/core/test_config.py
sed -i 's/MiFIDIIComplianceConfig/RiskControlsConfig/g' tests/core/test_config.py
sed -i 's/ClockSyncComplianceConfig/TimeSyncConfig/g' tests/core/test_config.py

# Validate
python -c "from services.core.risk_controls.config import RiskControlsConfig; print('CORE OK')"
python -c "from services.algo_integration.config import AlgoIntegrationConfig; print('INTEG OK')"
python -c "from services.archive.mifid_financial_entity.config import LEIConfig; print('ARCHIVE OK')"

git add services/core/risk_controls/config.py \
        services/algo_integration/config.py \
        services/archive/mifid_financial_entity/config.py \
        tests/core/test_config.py
git commit -m "refactor: split config.py into 3 group-specific configs"

# CHECKPOINT 8
git tag checkpoint-8-config-split
```

### PHASE 9: Create __init__.py Files

```bash
# Step 9.1: CORE __init__.py
cat > services/core/risk_controls/__init__.py << 'PYEOF'
"""
Core Risk Controls - Universal trading risk management.

For ALL platform users. ICT Provider baseline functionality.
"""
__version__ = "1.0.0"

from services.core.risk_controls.config import (
    ControlsMode, TimeSyncConfig, PreTradeControlsConfig, RiskControlsConfig,
)
from services.core.risk_controls.audit_models import (
    AuditEventType, AuditRecord, AuditRecordBuilder, AuditRecordPriority,
    AuditRecordStatus, OrderSide, AuditChainStatus,
    create_order_submitted_record, create_order_filled_record,
    create_risk_event_record, create_system_event_record,
)
from services.core.risk_controls.audit_storage import (
    StorageBackendType, StorageState, AuditStorageConfig, StorageMetrics,
    AuditStorageBackend, MemoryAuditStorage, SQLiteAuditStorage,
    FileAuditStorage, create_audit_storage,
)
from services.core.risk_controls.retention_policy import (
    RetentionPeriod, ArchiveStatus, RetentionPolicyConfig,
    RetentionManager, create_retention_manager,
)
from services.core.risk_controls.audit_trail_writer import (
    WriterMode, WriterState, AuditTrailWriterConfig, WriterMetrics,
    AuditTrailWriter, create_audit_trail_writer,
)
from services.core.risk_controls.time_sync import (
    ClockSyncStatus, ClockDriftSeverity, ComplianceClock, create_compliance_clock,
)
from services.core.risk_controls.kill_switch import (
    KillSwitchScope, KillSwitchTriggerReason, KillSwitchState,
    KillSwitchEvent, KillSwitchConfig, EmergencyContact,
    EnhancedKillSwitch, create_enhanced_kill_switch,
)
from services.core.risk_controls.pre_trade_controls import (
    RejectionReason, ControlSeverity, PreTradeCheckResult,
    PreTradeControls, create_pre_trade_controls,
)
from services.core.risk_controls.realtime_monitor import (
    AlertSeverity, AlertCategory, ComplianceAlert, MonitoringThreshold,
    RealTimeMonitorConfig, MonitoringMetrics,
    RealTimeMonitor, create_realtime_monitor,
)
from services.core.risk_controls.bcp import (
    ScenarioCategory, ImpactLevel, LikelihoodLevel, RecoveryStatus, AlertLevel,
    BusinessContinuityPlan, create_business_continuity_plan, get_standard_bcp_scenarios,
)

__all__ = [
    "RiskControlsConfig", "TimeSyncConfig", "PreTradeControlsConfig",
    "AuditEventType", "AuditRecord", "AuditRecordBuilder",
    "AuditStorageBackend", "create_audit_storage",
    "RetentionManager", "create_retention_manager",
    "AuditTrailWriter", "create_audit_trail_writer",
    "ComplianceClock", "create_compliance_clock",
    "EnhancedKillSwitch", "create_enhanced_kill_switch",
    "PreTradeControls", "create_pre_trade_controls",
    "RealTimeMonitor", "create_realtime_monitor",
    "BusinessContinuityPlan", "create_business_continuity_plan",
]
PYEOF

# Step 9.2: INTEGRATION __init__.py
cat > services/algo_integration/__init__.py << 'PYEOF'
"""
Algo Integration - B2B Compliance Toolkit.

Enterprise addon for financial institution clients.
Disabled by default for ICT Provider deployments.
"""
__version__ = "1.0.0"

from services.algo_integration.config import (
    AlgorithmRegistryConfig, AlgoIntegrationConfig,
)
from services.algo_integration.best_execution import (
    ExecutionFactor, AssetClass, VenueType, ExecutionQualityLevel,
    BestExecutionPolicy, BestExecutionAnalyzer,
    create_best_execution_policy, create_best_execution_analyzer,
)
from services.algo_integration.tca_compliance import (
    TCAMetricType, TCABenchmark, ExecutionStrategy,
    TCAComplianceWrapper, create_tca_wrapper,
)
from services.algo_integration.venue_analysis import (
    VenueMetricType, VenueSelectionReason, VenueStatus,
    VenueAnalyzer, SmartOrderRouter,
    create_venue_analyzer, create_smart_order_router,
)
from services.algo_integration.execution_quality_report import (
    ReportPeriod, ReportFormat,
    ExecutionQualityReportGenerator, create_report_generator,
)
from services.algo_integration.otr_monitor import (
    OrderEvent, OTRLevel, OTRMonitor, create_otr_monitor,
)
from services.algo_integration.algorithm_registry import (
    AlgorithmType, AlgorithmStatus, AlgorithmRecord,
    AlgorithmRegistry, create_algorithm_registry,
)
from services.algo_integration.conformance_testing import (
    TestResult, TestCategory, TestPriority, TestEnvironment,
    ConformanceTestSuite, ConformanceTestRunner,
    create_conformance_suite, create_test_runner,
)
from services.algo_integration.test_scenarios import (
    ScenarioType, ScenarioSeverity, ExecutionPhase, ScenarioStatus,
    TestScenario, ScenarioExecutor,
    create_test_scenario, create_scenario_executor,
)
from services.algo_integration.certification import (
    CertificateStatus, CertificateType, DeploymentApproval,
    ConformanceCertificate, CertificateManager,
    create_certificate, create_certificate_manager,
)

__all__ = [
    "AlgoIntegrationConfig", "AlgorithmRegistryConfig",
    "BestExecutionAnalyzer", "BestExecutionPolicy",
    "TCAComplianceWrapper", "VenueAnalyzer", "SmartOrderRouter",
    "ExecutionQualityReportGenerator", "OTRMonitor",
    "AlgorithmRegistry", "ConformanceTestRunner",
    "ScenarioExecutor", "CertificateManager",
]
PYEOF

# Step 9.3: ARCHIVE __init__.py
cat > services/archive/mifid_financial_entity/__init__.py << 'PYEOF'
"""
MiFID II Financial Entity Modules (ARCHIVED).

These modules implement MiFID II requirements for INVESTMENT FIRMS:
- LEI Management (Article 26 MiFIR)
- Transaction Reporting (RTS 22)
- NCA Notification (Article 17(2))

NOT APPLICABLE TO ICT PROVIDERS.
Importing this module emits DeprecationWarning.
"""
import warnings

__version__ = "1.0.0"
__archived__ = True
__archive_reason__ = "Not applicable to ICT Providers per MiFID II scope"

warnings.warn(
    "services.archive.mifid_financial_entity is archived. "
    "These modules are for Investment Firms under MiFID II, not ICT Providers.",
    DeprecationWarning,
    stacklevel=2
)

from services.archive.mifid_financial_entity.config import (
    ComplianceMode, LEIConfig, MiFIDIIComplianceConfig,
)
from services.archive.mifid_financial_entity.lei_manager import (
    LEIRecord, LEIStatus, LEIValidationResult, LEIManager, create_lei_manager,
)
from services.archive.mifid_financial_entity.gleif_client import (
    GLEIFClient, GLEIFResponse, GLEIFError, create_gleif_client,
)
from services.archive.mifid_financial_entity.transaction_report import (
    BuySellIndicator, TradingCapacity, TransactionReport, TransactionReportBuilder,
)
from services.archive.mifid_financial_entity.arm_client import (
    ARMProvider, ARMEnvironment, SubmissionStatus,
    ARMClient, MockARMClient, create_arm_client,
)
from services.archive.mifid_financial_entity.reporting_pipeline import (
    PipelineStatus, TransactionReportingPipeline, create_reporting_pipeline,
)
from services.archive.mifid_financial_entity.self_assessment import (
    AssessmentCategory, ComplianceStatus, AnnualSelfAssessment,
    create_annual_assessment, get_rts6_assessment_template,
)
from services.archive.mifid_financial_entity.governance import (
    PolicyType, PolicyStatus, GovernanceFramework, create_governance_framework,
)
from services.archive.mifid_financial_entity.compliance_policies import (
    create_best_execution_policy, create_all_standard_policies,
)
from services.archive.mifid_financial_entity.nca_notification import (
    NCAJurisdiction, NotificationType, NotificationStatus,
    NCANotification, NCANotificationManager, create_nca_notification_manager,
)

__all__ = [
    "LEIManager", "GLEIFClient", "TransactionReport", "ARMClient",
    "TransactionReportingPipeline", "AnnualSelfAssessment",
    "GovernanceFramework", "NCANotificationManager",
]
PYEOF

# Validate
python -c "from services.core.risk_controls import EnhancedKillSwitch; print('CORE OK')"
python -c "from services.algo_integration import BestExecutionAnalyzer; print('INTEG OK')"
python -c "from services.archive.mifid_financial_entity import LEIManager" 2>&1 | grep -q Deprecation && echo "ARCHIVE WARNING OK"

git add services/core/risk_controls/__init__.py \
        services/algo_integration/__init__.py \
        services/archive/mifid_financial_entity/__init__.py
git commit -m "refactor: add __init__.py with proper exports"

# CHECKPOINT 9
git tag checkpoint-9-init-files
```

### PHASE 10: Create Backward Compatibility Facade

```bash
cat > services/compliance/__init__.py << 'PYEOF'
"""
DEPRECATED: Backward compatibility facade.

Migration Guide:
  services.compliance.X → services.core.risk_controls.X
  services.compliance.Y → services.algo_integration.Y
  services.compliance.Z → services.archive.mifid_financial_entity.Z
"""
import warnings

__version__ = "8.0.0"
__deprecated__ = True

warnings.warn(
    "services.compliance is deprecated. "
    "Use services.core.risk_controls, services.algo_integration, "
    "or services.archive.mifid_financial_entity instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export CORE
from services.core.risk_controls import *

# Re-export INTEGRATION
from services.algo_integration import *

# Re-export ARCHIVE (already warns)
try:
    from services.archive.mifid_financial_entity import *
except Exception:
    pass  # Archive may be disabled
PYEOF

# Validate backward compat
python -c "
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    from services.compliance import EnhancedKillSwitch
    assert len(w) >= 1
    assert 'deprecated' in str(w[0].message).lower()
    print('Backward compat OK with warning')
"

git add services/compliance/__init__.py
git commit -m "refactor: add backward compatibility facade with deprecation"

# CHECKPOINT 10
git tag checkpoint-10-facade
```

### PHASE 11: Full Validation

```bash
# Run ALL tests
pytest tests/core tests/algo_integration tests/archive/mifid_fe -v --tb=short

# Verify import paths
python << 'PYEOF'
print("Testing imports...")

# CORE
from services.core.risk_controls import EnhancedKillSwitch
from services.core.risk_controls import AuditTrailWriter
from services.core.risk_controls import PreTradeControls
from services.core.risk_controls import RealTimeMonitor
from services.core.risk_controls import BusinessContinuityPlan
print("  CORE: OK")

# INTEGRATION
from services.algo_integration import BestExecutionAnalyzer
from services.algo_integration import OTRMonitor
from services.algo_integration import CertificateManager
from services.algo_integration import ConformanceTestRunner
print("  INTEGRATION: OK")

# ARCHIVE (with warning suppressed for test)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from services.archive.mifid_financial_entity import LEIManager
    from services.archive.mifid_financial_entity import TransactionReport
print("  ARCHIVE: OK")

# FACADE (should warn)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    from services.compliance import EnhancedKillSwitch as KS2
    assert any("deprecated" in str(x.message).lower() for x in w)
print("  FACADE: OK (with deprecation warning)")

print("\nAll imports successful!")
PYEOF

# Count migrated files
echo "=== Migration Summary ==="
echo "CORE modules: $(ls services/core/risk_controls/*.py 2>/dev/null | wc -l)"
echo "INTEGRATION modules: $(ls services/algo_integration/*.py 2>/dev/null | wc -l)"
echo "ARCHIVE modules: $(ls services/archive/mifid_financial_entity/*.py 2>/dev/null | wc -l)"
echo "CORE tests: $(ls tests/core/test_*.py 2>/dev/null | wc -l)"
echo "INTEGRATION tests: $(ls tests/algo_integration/test_*.py 2>/dev/null | wc -l)"
echo "ARCHIVE tests: $(ls tests/archive/mifid_fe/test_*.py 2>/dev/null | wc -l)"

# CHECKPOINT 11
git tag checkpoint-11-validated
```

### PHASE 12: Cleanup Old Files

```bash
# ONLY after all tests pass!
# Keep services/compliance/__init__.py (facade)

# Remove old module files
rm services/compliance/audit_models.py
rm services/compliance/audit_storage.py
rm services/compliance/audit_trail_writer.py
rm services/compliance/retention_policy.py
rm services/compliance/compliance_clock.py
rm services/compliance/enhanced_kill_switch.py
rm services/compliance/pre_trade_controls.py
rm services/compliance/realtime_monitor.py
rm services/compliance/bcp.py
rm services/compliance/config.py
rm services/compliance/best_execution.py
rm services/compliance/tca_compliance.py
rm services/compliance/venue_analysis.py
rm services/compliance/execution_quality_report.py
rm services/compliance/otr_monitor.py
rm services/compliance/algorithm_registry.py
rm services/compliance/conformance_testing.py
rm services/compliance/test_scenarios.py
rm services/compliance/certification.py
rm services/compliance/lei_manager.py
rm services/compliance/gleif_client.py
rm services/compliance/transaction_report.py
rm services/compliance/arm_client.py
rm services/compliance/reporting_pipeline.py
rm services/compliance/self_assessment.py
rm services/compliance/governance.py
rm services/compliance/compliance_policies.py
rm services/compliance/nca_notification.py

# Remove old test files
rm tests/test_mifid_*.py

# Final test
pytest tests/core tests/algo_integration tests/archive/mifid_fe -v --tb=short

git add -A
git commit -m "refactor: remove old compliance module files (migrated)

BREAKING CHANGE: services.compliance.* is now deprecated facade

New locations:
- CORE: services.core.risk_controls
- INTEGRATION: services.algo_integration
- ARCHIVE: services.archive.mifid_financial_entity

Old imports still work via facade but emit DeprecationWarning."

# FINAL TAG
git tag v2.0.0-migration-complete
git push origin refactor/mifid-ict-provider-migration-v2 --tags
```

---

## 7. Rollback Procedures

### Rollback to Any Checkpoint

```bash
# List checkpoints
git tag -l "checkpoint-*"

# Rollback to specific checkpoint
git reset --hard checkpoint-5-wave2

# If already pushed, create revert branch
git checkout -b rollback/to-checkpoint-5
git reset --hard checkpoint-5-wave2
git push origin rollback/to-checkpoint-5
```

### Full Rollback (abort migration)

```bash
git checkout main
git branch -D refactor/mifid-ict-provider-migration-v2
```

---

## 8. Validation Checklist

- [ ] All 28 modules accounted for (10 CORE + 9 INTEG + 9 ARCHIVE)
- [ ] certification.py in INTEGRATION (not ARCHIVE)
- [ ] config.py split into 3 files
- [ ] No cross-group dependencies
- [ ] All tests pass: `pytest tests/core tests/algo_integration tests/archive/mifid_fe`
- [ ] CORE imports work: `from services.core.risk_controls import X`
- [ ] INTEGRATION imports work: `from services.algo_integration import X`
- [ ] ARCHIVE imports warn: `from services.archive.mifid_financial_entity import X`
- [ ] Facade works with warning: `from services.compliance import X`
- [ ] 12 checkpoints created
- [ ] Old files removed

---

## 9. Post-Migration Tasks

1. **Update CI/CD**: Change test paths in GitHub Actions / GitLab CI
2. **Update docs**: README.md, API docs
3. **Notify B2B clients**: Deprecation notice with migration guide
4. **Set deprecation deadline**: Remove facade in v3.0.0 (6 months)
