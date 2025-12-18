# Changelog

## [6.1.0] - 2025-12-15

### Design Doc CCEA Compliance

- **P0: Real Cryptographic Signature Verification** (2025-12-15)
  - Integrated `ArtifactVerifier` into `preflight.py` for Ed25519 cryptographic verification
  - Replaced presence-only signature check with real crypto verification per Design Doc Phase 4
  - Unsigned artifacts now properly REJECTED (fail-closed security model)
  - Files: `packages/agent/daemon/preflight.py`
  - Tests: `tests/ccea/test_design_doc_compliance.py::TestPreflightVerifierIntegration`

- **P0: REQUEST_UPGRADE_ARTIFACT Handler** (2025-12-15)
  - Implemented full lifecycle command for artifact upgrades
  - Downloads artifact from cloud, verifies SHA-256 digest, performs crypto verification
  - Executes atomic upgrade with rollback capability
  - Files: `packages/agent/daemon/agentd.py` (lines 1061-1162)
  - Tests: `tests/ccea/test_design_doc_compliance.py::TestRequestUpgradeArtifact`

- **P0: REQUEST_UPDATE_CONFIG Handler** (2025-12-15)
  - Implemented configuration update lifecycle command
  - Validates config changes, checks TRADING_IMPACTING flag
  - Applies changes through policy firewall per Design Doc compliance
  - Files: `packages/agent/daemon/agentd.py` (lines 1164-1272)
  - Tests: `tests/ccea/test_design_doc_compliance.py::TestRequestUpdateConfig`

- **P0: Manifest Format Standardization** (2025-12-15)
  - Canonical format: `manifest.json` (JSON)
  - Legacy support: `manifest.yaml` (YAML) with deprecation warnings
  - Added `from_dict()`, `from_json()`, `from_file()` methods to `ArtifactManifest`
  - Files: `packages/agent/daemon/artifact_manager.py`
  - Tests: `tests/ccea/test_design_doc_compliance.py::TestManifestFormatStandardization`

### Canonical Stack Definition

- **packages/* as Production Stack** (2025-12-15)
  - `packages/agent/daemon/` - Production CCEA agent daemon
  - `packages/agent/daemon/preflight.py` - Artifact verification
  - `packages/agent/daemon/agentd.py` - Agent lifecycle management
  - `packages/agent/daemon/artifact_manager.py` - Artifact management

- **ccea/agent/* Deprecated** (2025-12-15)
  - Added deprecation warnings to `ccea/agent/daemon.py`, `runner.py`, `approval.py`
  - Migrate to `packages.agent.daemon.*` modules
  - Tests: `tests/ccea/test_design_doc_compliance.py::TestCCEAAgentDeprecation`

### Test Coverage

- **28 Design Doc Compliance Tests** (2025-12-15)
  - `TestPreflightVerifierIntegration` - 4 tests
  - `TestRequestUpgradeArtifact` - 4 tests
  - `TestRequestUpdateConfig` - 4 tests
  - `TestManifestFormatStandardization` - 4 tests
  - `TestCCEAAgentDeprecation` - 3 tests
  - `TestDesignDocMustHave` - 3 tests
  - `TestStateMachineCompliance` - 3 tests
  - `TestProtocolCompliance` - 3 tests
  - All tests passing (100%)

### Documentation

- Updated `ARCHITECTURE.md` to version 6.1
- Added Canonical Stack section with module hierarchy
- Added Design Doc Compliance table

---

## [5.0.0] - 2025-12-11

### BREAKING CHANGES

- **ICT Provider Architecture Restructure** (2025-12-11)
  - Complete module restructure for ICT Provider regulatory positioning
  - `services.compliance.*` is now a deprecated facade with deprecation warnings
  - New module locations:
    - **CORE**: `services.core.risk_controls` (universal risk controls)
    - **INTEGRATION**: `services.algo_integration` (B2B compliance toolkit)
    - **ARCHIVE**: `services.archive.mifid_financial_entity` (Investment Firm modules)

### Migration Guide

```python
# Old (deprecated - emits DeprecationWarning)
from services.compliance import EnhancedKillSwitch

# New
from services.core.risk_controls import EnhancedKillSwitch
```

| Old Import | New Import |
|------------|------------|
| `services.compliance.audit_models` | `services.core.risk_controls.audit_models` |
| `services.compliance.compliance_clock` | `services.core.risk_controls.time_sync` |
| `services.compliance.enhanced_kill_switch` | `services.core.risk_controls.kill_switch` |
| `services.compliance.best_execution` | `services.algo_integration.best_execution` |
| `services.compliance.lei_manager` | `services.archive.mifid_financial_entity.lei_manager` |

### Added

- **Core Risk Controls Package** (`services.core.risk_controls`)
  - 10 modules: audit_models, audit_storage, audit_trail_writer, retention_policy,
    time_sync, kill_switch, pre_trade_controls, realtime_monitor, bcp, config
  - Universal risk management for all platform users
  - Reference: RTS 6, RTS 25

- **Algo Integration Package** (`services.algo_integration`)
  - 9 modules: best_execution, tca_compliance, venue_analysis, execution_quality_report,
    otr_monitor, algorithm_registry, conformance_testing, test_scenarios, certification
  - B2B compliance toolkit for enterprise financial institution clients
  - Reference: MiFID II Article 27, RTS 6 Article 5

- **Archive Package** (`services.archive.mifid_financial_entity`)
  - 9 modules: lei_manager, gleif_client, transaction_report, arm_client,
    reporting_pipeline, self_assessment, governance, compliance_policies, nca_notification
  - Investment Firm specific modules (NOT for ICT Providers)
  - Emits deprecation warning on import

- **Backward Compatibility Facade** (`services.compliance`)
  - Re-exports all modules from new locations
  - Emits DeprecationWarning on import
  - Will be removed in v6.0.0

- **Split Configuration Files**
  - `services.core.risk_controls.config` - TimeSyncConfig, PreTradeControlsConfig, RiskControlsConfig
  - `services.algo_integration.config` - AlgorithmRegistryConfig, AlgoIntegrationConfig
  - `services.archive.mifid_financial_entity.config` - LEIConfig, MiFIDIIComplianceConfig

### Test Coverage

- **1,612 Migration Tests** (2025-12-11)
  - CORE tests: 12 test files
  - INTEGRATION tests: 12 test files
  - ARCHIVE tests: 10 test files
  - All tests passing (100%)

### Documentation

- Updated README.md with Module Architecture section
- Created API reference documentation (`docs/api/README.md`)
- Added migration guide for B2B clients
- Reference: [MIFID_ICT_PROVIDER_MIGRATION_PLAN_V3_FINAL.md](docs/migration/MIFID_ICT_PROVIDER_MIGRATION_PLAN_V3_FINAL.md)

---

## [4.0.0] - 2025-12-08

### DORA Alignment Toolkit (Implementation Complete)

- **Phase 0: Proportionality Assessment** (2025-12-08)
  - DORA scope verification (Article 2)
  - Function classification (Article 3(22))
  - Proportionality assessment for microenterprises
  - Files: `services/dora/scope_verification.py`, `proportionality.py`, `function_classification.py`
  - Reference: [docs/compliance/dora/proportionality_assessment.md](docs/compliance/dora/proportionality_assessment.md)

- **Phase 1: ICT Risk Management Framework** (2025-12-08)
  - ICT systems identification and classification
  - ICT risk identification and assessment
  - Protection and prevention measures
  - Detection capabilities
  - Response and recovery procedures
  - Business continuity planning
  - Backup and recovery policies
  - Learning and evolving
  - Communication protocols
  - Files: `services/dora/ict_*.py`, `protection.py`, `backup_recovery.py`
  - Tests: `tests/dora/test_dora_*.py`
  - Reference: DORA Articles 5-16

- **Phase 2: ICT Incident Management & Reporting** (2025-12-08)
  - Incident classification (major/minor)
  - Incident reporting to NCAs (4h/24h/72h timelines)
  - Third-party incident monitoring
  - CDR 2025/301 compliance (reporting templates)
  - Weekend/holiday extension rules
  - Files: `services/dora/incident_classification.py`, `incident_reporting.py`, `third_party_incidents.py`
  - Reference: DORA Articles 17-23

- **Phase 3: Digital Resilience Testing** (2025-12-08)
  - ICT testing framework
  - Threat-Led Penetration Testing (TLPT) per RTS 2024/2961
  - Tester management and qualification
  - Purple teaming capabilities
  - Test scenario library
  - Files: `services/dora/ict_testing.py`, `tester_management.py`
  - Tests: `tests/test_dora_phase3_tlpt.py`
  - Reference: DORA Articles 24-27

- **Phase 4: Third-Party ICT Risk Management** (2025-12-08)
  - Third-party risk assessment
  - Concentration risk monitoring (19 CTPPs designated Nov 2025)
  - Contractual requirements (Article 30)
  - Register of Information (ROI)
  - Exit strategies
  - CTPP oversight preparation
  - Files: `services/dora/third_party_risk.py`, `concentration_risk.py`, `exit_strategies.py`, `register_of_information.py`
  - Tests: `tests/dora/test_dora_third_party_risk.py`, `test_dora_concentration_risk.py`
  - Reference: DORA Articles 28-44

- **Phase 5: Information Sharing, Dashboard & Unified Reporting** (2025-12-08)
  - Information sharing framework (Article 45)
  - Cross-regulation integration (MiFID II, EU AI Act)
  - Compliance dashboard with real-time monitoring
  - Unified reporting (DPM 4.0 format, B_xx.xx templates)
  - Country-specific deadline tracking (Germany 11 Apr, France 15 Apr)
  - Files: `services/dora/information_sharing.py`, `cross_regulation.py`, `compliance_dashboard.py`, `unified_reporting.py`
  - Tests: `tests/dora/test_dora_information_sharing.py`, `test_dora_compliance_dashboard.py`, `test_dora_unified_reporting.py`

### Test Coverage

- **~1,015 DORA Tests** (2025-12-08)
  - 18 test files, ~395 test functions
  - All 5 phases fully tested
  - Pass rate: 97%+

---

## [3.0.0] - 2025-12-07

### MiFID II Alignment Toolkit (Implementation Complete)

- **Phase 4: Record Keeping & Audit Trail** (2025-12-06)
  - Implemented audit trail writing & storage (63KB module)
  - 5-7 years retention policy (MiFIR Article 25)
  - Audit data models and validators
  - Files: `services/compliance/audit_trail_writer.py`, `audit_storage.py`, `audit_models.py`, `retention_policy.py`
  - Tests: `tests/test_mifid_phase4_*.py`

- **Phase 5: Best Execution** (2025-12-06)
  - Best execution policy implementation (51KB, Article 27)
  - TCA compliance framework (37KB)
  - Venue analysis & Smart Order Routing
  - Execution quality reporting (40KB)
  - Files: `services/compliance/best_execution.py`, `tca_compliance.py`, `venue_analysis.py`, `execution_quality_report.py`
  - Tests: `tests/test_mifid_phase5_*.py`

- **Phase 6: Governance & Documentation** (2025-12-06)
  - Governance framework (44KB, RTS 6 Articles 3, 9)
  - Annual self-assessment module (53KB)
  - Business Continuity Plan (59KB)
  - Policy documentation system
  - Files: `services/compliance/governance.py`, `self_assessment.py`, `bcp.py`
  - Tests: `tests/test_mifid_phase6_*.py`

- **Phase 7: Testing & Certification** (2025-12-07)
  - Conformance testing framework (51KB, RTS 6 Article 5)
  - Test scenarios library (38KB)
  - Certification module (40KB)
  - NCA notification system (44KB)
  - Files: `services/compliance/conformance_testing.py`, `test_scenarios.py`, `certification.py`, `nca_notification.py`
  - Tests: `tests/test_mifid_phase7_*.py`
  - Reference: [MIFID_II_COMPLIANCE_ROADMAP.md](docs/compliance/MIFID_II_COMPLIANCE_ROADMAP.md)

### Test Coverage

- **14,000+ Automated Tests** (2025-12-07)
  - 654+ test files, 14,000+ test functions
  - MiFID II compliance tests: 200+
  - Pass rate: 97%+
  - All 7 phases fully tested

---

## [2.1.0] - 2025-11-21

### Critical Fixes

- **CRITICAL BUG #4: LSTM States NOT Reset on Episode Boundaries** (2025-11-21)
  - Fixed critical issue where LSTM hidden states persisted across episode boundaries,
    causing temporal leakage between unrelated episodes and degrading value estimation accuracy
  - Root cause: Missing reset logic in `distributional_ppo.py` rollout loop
  - Solution: Added `_reset_lstm_states_for_done_envs()` method and integrated into rollout (lines 7418-7427)
  - Files modified: `distributional_ppo.py`
  - Tests added: `tests/test_lstm_episode_boundary_reset.py` (8 comprehensive tests)
  - Impact: **CRITICAL** - Expected 5-15% improvement in value loss and policy performance
  - Models trained before 2025-11-21: **STRONGLY RECOMMENDED** to retrain for best performance
  - Reference: [CRITICAL_LSTM_RESET_FIX_REPORT.md](archive/2025_11/reports_2025_11_25_cleanup/root_reports/CRITICAL_LSTM_RESET_FIX_REPORT.md)
  - Academic reference: Hausknecht & Stone (2015) "Deep Recurrent Q-Learning for POMDPs"

- **IMPROVEMENT #2: External Features NaN → 0.0 Silent Conversion** (2025-11-21)
  - Improved NaN handling with logging capability for debugging missing data
  - Root cause: `_get_safe_float()` silently converted NaN → 0.0 without warning
  - Solution: Enhanced with `log_nan=True` parameter for debugging (mediator.py:989-1072)
  - Documentation: Enhanced obs_builder.pyx docstring (lines 7-36)
  - Files modified: `mediator.py`, `obs_builder.pyx` (comments)
  - Tests added: `tests/test_nan_handling_external_features.py` (10 tests, 9 passing, 1 skipped)
  - Impact: MEDIUM - Semantic ambiguity remains (missing data = 0.0), but now debuggable
  - Future work: Add validity flags for external features (v2.2+)
  - Reference: [NUMERICAL_ISSUES_FIX_SUMMARY.md](archive/2025_11/reports_2025_11_25_cleanup/root_reports/NUMERICAL_ISSUES_FIX_SUMMARY.md)

- **CRITICAL BUG #1: Sign Convention Mismatch in LongOnlyWrapper** (2025-11-21)
  - Fixed sign convention where negative actions (reduction signals) were clipped to zero
  - Root cause: Direct clipping instead of affine mapping lost reduction information
  - Solution: Use mapping `(action + 1.0) / 2.0` to preserve full [-1,1] signal range
  - Files modified: `wrappers/action_space.py`
  - Tests: Covered in `tests/test_critical_action_space_fixes.py` (21 tests, all passing)
  - Impact: HIGH - Policy can now properly reduce positions
  - Reference: [CRITICAL_FIXES_COMPLETE_REPORT.md](archive/2025_11/reports_2025_11_25_cleanup/root_reports/CRITICAL_FIXES_COMPLETE_REPORT.md#problem-1)

- **CRITICAL BUG #2: Position Semantics DELTA→TARGET** (2025-11-21)
  - Fixed critical position doubling bug where DELTA semantics caused 2x leverage violation
  - Root cause: `ActionProto.volume_frac` was interpreted as DELTA instead of TARGET
  - Solution: Changed semantics to TARGET position (prevents doubling)
  - Files modified: `risk_guard.py`, contract documentation
  - Tests: Covered in `tests/test_critical_action_space_fixes.py`
  - Impact: **CRITICAL** - Prevents position doubling in production (2x leverage violation)
  - Models with old DELTA semantics: **MUST** retrain
  - Reference: [CRITICAL_FIXES_COMPLETE_REPORT.md](archive/2025_11/reports_2025_11_25_cleanup/root_reports/CRITICAL_FIXES_COMPLETE_REPORT.md#problem-2)

- **CRITICAL BUG #3: Action Space Range [0,1] vs [-1,1]** (2025-11-21)
  - Fixed action space mismatch where different components used different bounds
  - Root cause: Inconsistent action space definitions across codebase
  - Solution: Unified to [-1,1] everywhere for architectural consistency
  - Files modified: Various action space components
  - Tests: Covered in `tests/test_critical_action_space_fixes.py`
  - Impact: HIGH - Prevents action clipping errors and improves training stability
  - Reference: [CRITICAL_FIXES_COMPLETE_REPORT.md](archive/2025_11/reports_2025_11_25_cleanup/root_reports/CRITICAL_FIXES_COMPLETE_REPORT.md#problem-3)

### Documentation

- **Documentation Modernization** (2025-11-21)
  - Modernized all core documentation to Version 2.1
  - Updated [claude.md](claude.md) - Main project documentation (v2.0 → v2.1)
  - Completely rewrote [README.md](README.md) - Comprehensive project overview
  - Updated [DOCS_INDEX.md](DOCS_INDEX.md) - Navigation hub with critical fixes
  - Enhanced [distributional_ppo.py](distributional_ppo.py) - Expanded class docstring (1 line → 58 lines)
  - Created [DOCUMENTATION_STATUS.md](archive/2025_11/reports_2025_11/integration/DOCUMENTATION_STATUS.md) - Centralized status tracking (historical)
  - Created [DOCUMENTATION_MODERNIZATION_REPORT.md](archive/2025_11/reports_2025_11/integration/DOCUMENTATION_MODERNIZATION_REPORT.md) - Full modernization report (historical)
  - Impact: +15% average improvement in audience coverage
  - Reference: [DOCUMENTATION_MODERNIZATION_REPORT.md](archive/2025_11/reports_2025_11/integration/DOCUMENTATION_MODERNIZATION_REPORT.md)

### Test Coverage

- **52+ New Tests for Critical Fixes** (2025-11-21)
  - LSTM Episode Reset: 8 tests (all passing)
  - NaN Handling: 10 tests (9 passing, 1 skipped - Cython)
  - Action Space Fixes: 21 tests (all passing)
  - Stale Data Temporal Causality: 3 tests (from 2025-11-20)
  - Cross-Symbol Contamination: 4 tests (from 2025-11-20)
  - Quantile Loss Formula: 11 tests (from 2025-11-20)
  - Total: 57 new regression prevention tests
  - All critical issues now have comprehensive test coverage

### Regression Prevention

- **Added Comprehensive Checklist** (2025-11-21)
  - Created [REGRESSION_PREVENTION_CHECKLIST.md](archive/2025_11/reports_2025_11/verification/REGRESSION_PREVENTION_CHECKLIST.md)
  - Mandatory checklist for developers before modifying critical components
  - Covers: LSTM state management, action space semantics, data integrity
  - Enforces: Running tests, reading fix reports, understanding semantics

## [Unreleased]

### Added
- **Seasonality Support**: Introduced hour-of-week seasonality multipliers to improve simulation fidelity.
  - **Required actions**:
    - Regenerate multipliers with the quick-start script.
    - Validate and update configurations before training or running simulations.
  - **Resources**:
    - [Seasonality overview](docs/seasonality.md)
    - [Quick start guide](docs/seasonality_quickstart.md)
    - [Process checklist](docs/seasonality_checklist.md)
    - [Example notebook](docs/seasonality_example.md)
    - [Migration guide](docs/seasonality_migration.md)
- **Dynamic spread builder**: Added `scripts/build_spread_seasonality.py` for generating
  hour-of-week spread profiles consumed by `slippage.dynamic`. The script
  supports custom output paths, rolling windows and warns when the source
  snapshot exceeds the configured `refresh_warn_days` threshold.
- **Fee settlement & rounding controls**: YAML-конфиги теперь содержат блоки
  `fees.rounding` и `fees.settlement` с безопасными значениями по умолчанию.
  `rounding` умеет использовать `commission_step` из биржевых фильтров и
  таблиц комиссий, а `settlement` описывает расчёт комиссий в альтернативном
  активе (например, BNB) с учётом скидок.
- **Daily turnover caps**: Added configuration fields, runtime enforcement, and
  monitoring visibility for daily USD/BPS turnover limits across per-symbol and
  portfolio aggregates. Includes persistence hooks and targeted pytest coverage
  ensuring partial/deferred execution when caps bind.

### Deprecated
- `LatencyImpl.dump_latency_multipliers` and
  `LatencyImpl.load_latency_multipliers` have been replaced by
  `dump_multipliers` and `load_multipliers`. The old names continue to work but
  emit `DeprecationWarning`. See the migration guide for details.

### Fixed
- **CRITICAL BUG #10: Temporal causality violation in stale data** (2025-11-20)
  - Fixed critical issue where stale bars were returned with PREVIOUS timestamp instead
    of CURRENT timestamp, violating temporal causality and corrupting model training
  - Root cause: `impl_offline_data.py` yielded `prev_bar` directly with old timestamp
  - Solution: Create new `Bar` with current timestamp but stale prices/volume
  - Files modified: `impl_offline_data.py`
  - Tests added: `tests/test_stale_bar_temporal_causality.py` (3 tests)
  - Impact: **CRITICAL** - Models trained with data degradation may have learned incorrect
    temporal patterns. Consider retraining if `stale_prob > 0` was used.
  - Reference: [CRITICAL_FIXES_REPORT.md](archive/2025_11/reports_2025_11_25_cleanup/root_reports/CRITICAL_FIXES_REPORT.md#problem-1-temporal-causality-violation)

- **CRITICAL BUG #11: Cross-symbol contamination in feature normalization** (2025-11-20)
  - Fixed critical issue where `shift()` applied after concatenating all symbols caused
    last row of Symbol1 to leak into first row of Symbol2, corrupting normalization stats
  - Root cause: `features_pipeline.py` applied `shift(1)` to concatenated DataFrame
  - Solution: Apply `shift()` per-symbol BEFORE concat, use `groupby()` in transform
  - Files modified: `features_pipeline.py`
  - Tests added: `tests/test_normalization_cross_symbol_contamination.py` (4 tests)
  - Impact: **CRITICAL** - Multi-symbol models may have contaminated features. Consider
    retraining if multiple symbols were used with normalization.
  - Reference: [CRITICAL_FIXES_REPORT.md](archive/2025_11/reports_2025_11_25_cleanup/root_reports/CRITICAL_FIXES_REPORT.md#problem-2-cross-symbol-contamination)

- **CRITICAL BUG #12: Inverted quantile loss formula** (2025-11-20)
  - Fixed critical mathematical error where quantile loss used `Q - T` instead of correct
    `T - Q` formula from Dabney et al. (2018), inverting asymmetric penalties
  - Root cause: `distributional_ppo.py` defaulted to legacy (incorrect) formula
  - Solution: Changed default to `_use_fixed_quantile_loss_asymmetry = True`
  - Files modified: `distributional_ppo.py`
  - Tests added: `tests/test_quantile_loss_formula_default.py` (3 tests)
  - Tests updated: `tests/test_quantile_loss_with_flag.py` (8 tests, all passing)
  - Impact: **CRITICAL** - Quantile critic models have suboptimal convergence and biased
    CVaR estimates. **STRONGLY RECOMMENDED** to retrain all quantile-based models.
  - Reference: [CRITICAL_FIXES_REPORT.md](archive/2025_11/reports_2025_11_25_cleanup/root_reports/CRITICAL_FIXES_REPORT.md#problem-3-inverted-quantile-loss)
  - Academic reference: Dabney et al. (2018) "Distributional RL with Quantile Regression"

- **Bug #9: VGS parameter tracking after model load** - Fixed critical issue where VGS
  (Variance Gradient Scaler) tracked stale parameter copies instead of actual policy
  parameters after `model.load()`, causing gradient scaling to have no effect on training
  after checkpoint restoration.
  - Root cause: VGS pickled parameter references that became stale after unpickling
  - Solution: Exclude `_parameters` from pickle state and relink via `update_parameters()`
    after load
  - Files modified: `variance_gradient_scaler.py`, `distributional_ppo.py`
  - Impact: Critical for production use of checkpointing with VGS enabled

- Ensured the explained-variance reserve path preserves training masks by
  default so no-trade windows and other zero-weight samples no longer depress
  EV metrics.
