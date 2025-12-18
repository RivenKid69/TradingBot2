# API Reference

## Overview

This platform uses a three-tier architecture to separate concerns and clarify regulatory positioning:

```
services/
├── core/risk_controls/       # Universal risk controls (Always loaded)
├── algo_integration/         # B2B compliance toolkit (Enterprise addon)
└── archive/mifid_financial_entity/  # Investment Firm modules (Archived)
```

## Core Risk Controls

`from services.core.risk_controls import ...`

Universal risk management for **all platform users**. These modules implement essential trading safety features regardless of regulatory status.

| Module | Description | Key Classes |
|--------|-------------|-------------|
| [audit_models](core/README.md#audit_models) | Audit record data models | `AuditRecord`, `AuditRecordBuilder` |
| [audit_storage](core/README.md#audit_storage) | Storage backends (SQLite, File, Memory) | `AuditStorageBackend`, `create_audit_storage` |
| [audit_trail_writer](core/README.md#audit_trail_writer) | Write-once audit trail | `AuditTrailWriter`, `create_audit_trail_writer` |
| [retention_policy](core/README.md#retention_policy) | Data retention management | `RetentionManager`, `create_retention_manager` |
| [time_sync](core/README.md#time_sync) | Clock synchronization | `ComplianceClock`, `create_compliance_clock` |
| [kill_switch](core/README.md#kill_switch) | Emergency stop functionality | `EnhancedKillSwitch`, `create_enhanced_kill_switch` |
| [pre_trade_controls](core/README.md#pre_trade_controls) | Order validation, fat finger protection | `PreTradeControls`, `create_pre_trade_controls` |
| [realtime_monitor](core/README.md#realtime_monitor) | P&L and risk monitoring | `RealTimeMonitor`, `create_realtime_monitor` |
| [bcp](core/README.md#bcp) | Business continuity planning | `BusinessContinuityPlan`, `create_business_continuity_plan` |
| [config](core/README.md#config) | Configuration models | `RiskControlsConfig`, `TimeSyncConfig` |

### Quick Start

```python
from services.core.risk_controls import (
    EnhancedKillSwitch,
    PreTradeControls,
    AuditTrailWriter,
    RealTimeMonitor,
    create_enhanced_kill_switch,
    create_pre_trade_controls,
)

# Create kill switch
kill_switch = create_enhanced_kill_switch(
    algorithm_id="ALGO-001",
    firm_name="My Trading Firm"
)

# Create pre-trade controls
controls = create_pre_trade_controls(
    max_order_value_eur=100_000,
    price_collar_pct=5.0
)

# Start monitoring
kill_switch.start()
```

---

## Algo Integration (B2B Alignment/Evidence Toolkit)

`from services.algo_integration import ...`

MiFID II-related alignment/evidence tooling for **enterprise financial institution clients**. These modules are designed to support client assessments and internal workflows; they are not a certification claim and do not replace legal/compliance review.

| Module | MiFID II Reference | Description | Key Classes |
|--------|-------------------|-------------|-------------|
| [best_execution](integration/README.md#best_execution) | Article 27 | Best execution analysis | `BestExecutionAnalyzer`, `BestExecutionPolicy` |
| [tca_compliance](integration/README.md#tca_compliance) | Article 27 | Transaction cost analysis | `TCAComplianceWrapper`, `create_tca_wrapper` |
| [venue_analysis](integration/README.md#venue_analysis) | Article 27 | Venue performance & SOR | `VenueAnalyzer`, `SmartOrderRouter` |
| [execution_quality_report](integration/README.md#execution_quality_report) | Article 27 | Execution quality reports | `ExecutionQualityReportGenerator` |
| [otr_monitor](integration/README.md#otr_monitor) | RTS 6 | Order-to-trade ratio monitoring | `OTRMonitor`, `create_otr_monitor` |
| [algorithm_registry](integration/README.md#algorithm_registry) | Article 17(2) | Algorithm registration | `AlgorithmRegistry`, `create_algorithm_registry` |
| [conformance_testing](integration/README.md#conformance_testing) | RTS 6 Article 5 | Testing framework | `ConformanceTestRunner`, `create_test_runner` |
| [test_scenarios](integration/README.md#test_scenarios) | RTS 6 Article 5 | Standard test scenarios | `ScenarioExecutor`, `TestScenario` |
| [certification](integration/README.md#certification) | RTS 6 Article 7 | Deployment attestation (internal evidence artifact) | `CertificateManager`, `ConformanceCertificate` |
| [config](integration/README.md#config) | - | Configuration | `AlgoIntegrationConfig` |

### Quick Start

```python
from services.algo_integration import (
    BestExecutionAnalyzer,
    TCAComplianceWrapper,
    AlgorithmRegistry,
    ConformanceTestRunner,
    create_best_execution_analyzer,
    create_tca_wrapper,
    create_algorithm_registry,
)

# Create best execution analyzer
analyzer = create_best_execution_analyzer()

# Create TCA wrapper
tca = create_tca_wrapper()

# Register algorithm
registry = create_algorithm_registry(firm_name="Investment Firm Ltd")
registry.register_algorithm(
    algo_id="VWAP-001",
    name="VWAP Strategy",
    algo_type="VWAP"
)
```

---

## Archive (Financial Entity)

`from services.archive.mifid_financial_entity import ...`

**NOT FOR ICT PROVIDERS** - These modules implement MiFID II requirements specifically for **Investment Firms**. Importing this package emits a `DeprecationWarning`.

| Module | Description | Key Classes |
|--------|-------------|-------------|
| [lei_manager](archive/README.md#lei_manager) | LEI validation (ISO 17442) | `LEIManager`, `create_lei_manager` |
| [gleif_client](archive/README.md#gleif_client) | GLEIF API integration | `GLEIFClient`, `create_gleif_client` |
| [transaction_report](archive/README.md#transaction_report) | RTS 22 transaction reporting | `TransactionReport`, `TransactionReportBuilder` |
| [arm_client](archive/README.md#arm_client) | ARM submission | `ARMClient`, `create_arm_client` |
| [reporting_pipeline](archive/README.md#reporting_pipeline) | T+1 reporting pipeline | `TransactionReportingPipeline` |
| [self_assessment](archive/README.md#self_assessment) | Annual self-assessment | `AnnualSelfAssessment` |
| [governance](archive/README.md#governance) | Policy document management | `GovernanceFramework` |
| [compliance_policies](archive/README.md#compliance_policies) | Policy templates | `create_all_standard_policies` |
| [nca_notification](archive/README.md#nca_notification) | NCA notification | `NCANotificationManager` |
| [config](archive/README.md#config) | Configuration | `MiFIDIIComplianceConfig`, `LEIConfig` |

### Usage Warning

```python
import warnings

# This will emit a DeprecationWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # Suppress if needed
    from services.archive.mifid_financial_entity import LEIManager
```

---

## Backward Compatibility

The old `services.compliance` module is a **deprecated facade** that re-exports all modules from their new locations. It emits a `DeprecationWarning` on import.

```python
# This works but emits DeprecationWarning
from services.compliance import EnhancedKillSwitch

# Recommended: Use new import paths
from services.core.risk_controls import EnhancedKillSwitch
```

### Migration Timeline

| Version | Status |
|---------|--------|
| v5.0.0 | Facade created with deprecation warnings |
| v6.0.0 | Facade will be removed |

---

## Regulatory Context

### Why This Architecture?

This platform is positioned as an **ICT Provider** (Information and Communication Technology Provider) under MiFID II, not as an Investment Firm:

1. **We provide software infrastructure** for algorithmic trading
2. **Users trade through their own broker accounts** - we don't execute trades on their behalf
3. **We don't hold client assets** - no custody or dealing
4. **MiFID II doesn't apply directly to us** - we're a software vendor

### Module Positioning

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR SAAS PLATFORM                       │
│                    (ICT Provider)                           │
├─────────────────────────────────────────────────────────────┤
│  CORE (Always)          │  INTEGRATION (B2B)                │
│  ──────────────         │  ─────────────────                │
│  • Kill Switch          │  • Best Execution (Art. 27)       │
│  • Pre-Trade Controls   │  • TCA Compliance                 │
│  • Audit Trail          │  • Conformance Testing (RTS 6)    │
│  • Time Sync            │  • Algorithm Registry             │
│  • BCP                  │  • Certification                  │
├─────────────────────────────────────────────────────────────┤
│  "We provide the tools, clients use their own accounts"     │
│  "MiFID II doesn't apply to us directly"                    │
│  "Our INTEGRATION module helps clients comply with MiFID"   │
└─────────────────────────────────────────────────────────────┘
```

---

## See Also

- [Migration Plan](../migration/MIFID_ICT_PROVIDER_MIGRATION_PLAN_V3_FINAL.md)
- [MiFID II Compliance Roadmap](../compliance/MIFID_II_COMPLIANCE_ROADMAP.md)
- [DORA Integration Plan](../compliance/DORA_INTEGRATION_PLAN.md)
