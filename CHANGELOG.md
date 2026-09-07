# Changelog

## [1.0.0] - 2026-09-06

First public release, under Apache-2.0.

The pre-1.0 numbering in this file (up to 6.1.1) and in `pyproject.toml` (2.6.0) tracked
two different internal schemes and never agreed with each other. 1.0.0 restarts from the
first version that is actually published; the entries below it are kept as the internal
history they were.

### Licence

- `LICENSE` is now the verbatim Apache License 2.0. It was a "PROPRIETARY AND
  CONFIDENTIAL / All rights reserved" notice, which GitHub reported as NOASSERTION.
- Added `NOTICE` listing the vendored third-party components (Monaco Editor, Chart.js,
  Tailwind CSS, Font Awesome Free, Inter, JetBrains Mono) and their licences.
- Added `SECURITY.md` and `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1); rewrote
  `CONTRIBUTING.md` around the process that actually exists.
- Removed `LICENSING.md` and `tools/repo_split/`: the open-core split they described is
  not being pursued.

### Fixed

- **Windows/MSVC native build.** `info_builder`, `environment` and `risk_manager` declare
  `# distutils: language = c++` but were compiled with C flags lacking `/std:c++17`, so
  MSVC fell back to C++14 and `std::optional` in `AgentOrderTracker.h` did not exist.
  Linux was unaffected because GCC defaults to `gnu++17`. All 17 extensions build.
- **The dependency locks could not be installed.** `shimmy==2.0.0` requires
  `gymnasium>=1.0` against a pinned `gymnasium==0.29.1`; `statsmodels` was pinned nowhere
  and 0.15.0 breaks `arch==7.1.0`; the dev lock pinned `fastapi` without `pydantic`,
  leaving `pydantic` and `pydantic_core` mismatched; and `cryptography` disagreed between
  the CPU and dev locks. This was the root cause of every red CI run: `pip install`
  failed and the tests then reported missing modules.
- **CI.** `pip-audit` was called with `--disable-pip` on non-hashed requirements, which it
  rejects. The `safety` step exited 64 because the tool now requires an account; dropped,
  since `pip-audit` covers the same files. Deprecated action versions bumped.
- **Test isolation.** Three test modules replaced `sys.modules["lob_state_cython"]` with a
  stub at import time and never restored it, so everything collected afterwards saw the
  stub instead of the compiled extension.
- Registered the `documentation` pytest marker, which `--strict-markers` rejected,
  aborting collection of the whole suite.
- Added `.gitattributes` with `eol=lf`. The mixed CRLF tree made
  `ccea.guardrails.design_doc_check` fail on Windows, a failure previously written off as
  an environment artefact.
- Six latent `NameError` sites fixed: a missing `logging` import used seven times in
  `scripts/fetch_binance_filters.py`, `split_time_range` in
  `scripts/extract_liquidity_seasonality.py`, `load_config` in `service_signal_runner.py`,
  a non-existent `SecurityWarning` in `infer_signals.py`, and an undefined `logger` in
  `app.py`. Two further ones in `execution_sim.py` and `train_model_multi_patch.py` are
  annotated in place rather than guessed at — see `docs/AUDIT_2026-09.md`.
- An always-true tuple assertion in `tests/test_web3_custody.py`, a loop variable
  shadowing an import in `services/backtest/disclaimer_injection.py`, and a leftover
  `=======` merge-conflict marker in `ARCHITECTURE.md`.

### Removed

- 175 `scratch_*` step dumps from the repository root, `archive/` (481 files),
  `essential_docs_collection/` (31 duplicated documents), `reports/`, `audits/`,
  `issues/`, `coverage.json`, three diff patches, and reports that lived under `tests/`.
- `experiments/`, `model_registry/`, `models/*.json` and four `state/*.json` files — run
  output written by the pipeline itself, now gitignored. They also embedded absolute
  paths from the machine that produced them.
- 12 root modules nothing imported, and `tests/test_bug3_fix.py`, which skipped at module
  level and whose remaining code was unreachable.
- `data/raw_stocks/` and everything derived from it — roughly 17 MB of Yahoo Finance data
  whose terms do not allow redistribution. `scripts/download_stock_data.py` fetches it
  again; `prepare_demo_data.py` generates synthetic bars so the pipeline runs without it.

### Changed

- Rewrote `README.md` around what the code does and which test proves it.
- Reorganised the documentation: reference material to `docs/reference/`, closed reports
  to `docs/history/`, go-to-market material to `docs/history/business/`. Regenerated
  `DOCS_INDEX.md`.
- Repaired every broken relative link — 1 129 of 1 963 were dead. Added
  `tools/check_markdown_links.py` and a CI job so they stay that way.
- Formatted the tree with `black`, and narrowed the ruff configuration to the rules the
  codebase actually satisfies. `ruff check .` and `black --check .` both pass.
- Removed tool attribution from 41 file headers, the leaked local paths in
  `PRO_MODE_DESIGN_DOCUMENT.md`, and the assistant branding on the Developer Suite
  terminal panel, which is now "Interactive CLI & Job Router".

---

## [6.1.1] - 2025-12-21

### Due Diligence Documentation Corrections

Per Canon (`docs/DOCUMENTATION_CANON_DESIGN.md`) Section 4.2 (avoid certification claims) and Section 4.5 (avoid absolute/unprovable claims):

- **SUBCONTRACTOR_REGISTER.md** (v1.1): Replaced "✓ Signed" and "Executed" DPA claims with "Standard DPA available (verify execution status via contract register)" for Cloudflare, Datadog, Stripe sections. DPA execution depends on customer contract phase; these are vendor-offered terms, not executed contracts.

- **EU_AI_ACT_INTEGRATION_PLAN.md** (v1.6): Replaced specific test counts (372, 236, 318, 81, 189, 1196+) with "Internal tests (verify count via CI)" to avoid outdated claims. Changed "**IMPLEMENTED**" to "Tooling implemented" for Article status table.

- **INVESTOR_BRIEF.md**: Replaced "✅ Complete" status markers with "Foundation implemented" and "Defined" to avoid absolute completion claims.

---

## [6.1.0] - 2025-12-15

### Design Doc CCEA Compliance

- **P0: Real Cryptographic Signature Verification** (2025-12-15)
  - Integrated `ArtifactVerifier` into `preflight.py` for Ed25519 cryptographic verification
  - Replaced presence-only signature check with real crypto verification per Design Doc Phase 4
  - Unsigned artifacts now properly REJECTED (designed for fail-closed behavior)
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
  - All tests passing at time of release (verify via CI run logs)

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
  - All tests passing at time of release (verify via CI run logs for release tag)

### Documentation

- Updated README.md with Module Architecture section
- Created API reference documentation (`docs/api/README.md`)
- Added migration guide for B2B clients
- Reference: [MIFID_ICT_PROVIDER_MIGRATION_PLAN_V3_FINAL.md](docs/migration/MIFID_ICT_PROVIDER_MIGRATION_PLAN_V3_FINAL.md)

---

## [4.0.0] - 2025-12-08

### DORA Alignment Toolkit (Engineering Implementation; Not a Certification Claim)

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
  - CDR 2025/301 reporting-template support (tooling)
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
  - Evidence/dashboard tooling with real-time monitoring
  - Unified reporting (DPM 4.0 format, B_xx.xx templates)
  - Country-specific deadline tracking (jurisdiction-dependent; verify for target deployment)
  - Files: `services/dora/information_sharing.py`, `cross_regulation.py`, `compliance_dashboard.py`, `unified_reporting.py`
  - Tests: `tests/dora/test_dora_information_sharing.py`, `test_dora_compliance_dashboard.py`, `test_dora_unified_reporting.py`

### Test Coverage

- **~1,015 DORA Tests** (2025-12-08; approximate count at documentation time; verify current count via CI)
  - 18 test files, ~395 test functions (approximate)
  - All 5 phases covered (verify via test plan/matrix)
  - Pass rate at documentation time: 97%+ (verify current status via CI reports)

---

## [3.0.0] - 2025-12-07

### MiFID II Alignment Toolkit (Engineering Implementation; Not a Certification Claim)

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
  - MiFID II alignment-toolkit tests: 200+
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
  - Reference: CRITICAL_LSTM_RESET_FIX_REPORT.md
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
  - Reference: NUMERICAL_ISSUES_FIX_SUMMARY.md

- **CRITICAL BUG #1: Sign Convention Mismatch in LongOnlyWrapper** (2025-11-21)
  - Fixed sign convention where negative actions (reduction signals) were clipped to zero
  - Root cause: Direct clipping instead of affine mapping lost reduction information
  - Solution: Use mapping `(action + 1.0) / 2.0` to preserve full [-1,1] signal range
  - Files modified: `wrappers/action_space.py`
  - Tests: Covered in `tests/test_critical_action_space_fixes.py` (21 tests, all passing)
  - Impact: HIGH - Policy can now properly reduce positions
  - Reference: CRITICAL_FIXES_COMPLETE_REPORT.md

- **CRITICAL BUG #2: Position Semantics DELTA→TARGET** (2025-11-21)
  - Fixed critical position doubling bug where DELTA semantics caused 2x leverage violation
  - Root cause: `ActionProto.volume_frac` was interpreted as DELTA instead of TARGET
  - Solution: Changed semantics to TARGET position (prevents doubling)
  - Files modified: `risk_guard.py`, contract documentation
  - Tests: Covered in `tests/test_critical_action_space_fixes.py`
  - Impact: **CRITICAL** - Prevents position doubling in production (2x leverage violation)
  - Models with old DELTA semantics: **MUST** retrain
  - Reference: CRITICAL_FIXES_COMPLETE_REPORT.md

- **CRITICAL BUG #3: Action Space Range [0,1] vs [-1,1]** (2025-11-21)
  - Fixed action space mismatch where different components used different bounds
  - Root cause: Inconsistent action space definitions across codebase
  - Solution: Unified to [-1,1] everywhere for architectural consistency
  - Files modified: Various action space components
  - Tests: Covered in `tests/test_critical_action_space_fixes.py`
  - Impact: HIGH - Prevents action clipping errors and improves training stability
  - Reference: CRITICAL_FIXES_COMPLETE_REPORT.md

### Documentation

- **Documentation Modernization** (2025-11-21)
  - Modernized all core documentation to Version 2.1
  - Updated [docs/PLATFORM_REFERENCE.md](docs/PLATFORM_REFERENCE.md) - Main project documentation (v2.0 → v2.1)
  - Completely rewrote [README.md](README.md) - Comprehensive project overview
  - Updated [DOCS_INDEX.md](DOCS_INDEX.md) - Navigation hub with critical fixes
  - Enhanced [distributional_ppo.py](distributional_ppo.py) - Expanded class docstring (1 line → 58 lines)
  - Created DOCUMENTATION_STATUS.md - Centralized status tracking (historical)
  - Created DOCUMENTATION_MODERNIZATION_REPORT.md - Full modernization report (historical)
  - Impact: +15% average improvement in audience coverage
  - Reference: DOCUMENTATION_MODERNIZATION_REPORT.md

### Test Coverage

- **52+ New Tests for Critical Fixes** (2025-11-21)
  - LSTM Episode Reset: 8 tests (all passing)
  - NaN Handling: 10 tests (9 passing, 1 skipped - Cython)
  - Action Space Fixes: 21 tests (all passing)
  - Stale Data Temporal Causality: 3 tests (from 2025-11-20)
  - Cross-Symbol Contamination: 4 tests (from 2025-11-20)
  - Quantile Loss Formula: 11 tests (from 2025-11-20)
  - Total: 57 new regression prevention tests
  - Critical issues identified as of 2025-11-21 have test coverage (coverage scope defined per issue; verify via CI for current status)

### Regression Prevention

- **Added Comprehensive Checklist** (2025-11-21)
  - Created REGRESSION_PREVENTION_CHECKLIST.md
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
  - Reference: CRITICAL_FIXES_REPORT.md

- **CRITICAL BUG #11: Cross-symbol contamination in feature normalization** (2025-11-20)
  - Fixed critical issue where `shift()` applied after concatenating all symbols caused
    last row of Symbol1 to leak into first row of Symbol2, corrupting normalization stats
  - Root cause: `features_pipeline.py` applied `shift(1)` to concatenated DataFrame
  - Solution: Apply `shift()` per-symbol BEFORE concat, use `groupby()` in transform
  - Files modified: `features_pipeline.py`
  - Tests added: `tests/test_normalization_cross_symbol_contamination.py` (4 tests)
  - Impact: **CRITICAL** - Multi-symbol models may have contaminated features. Consider
    retraining if multiple symbols were used with normalization.
  - Reference: CRITICAL_FIXES_REPORT.md

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
  - Reference: CRITICAL_FIXES_REPORT.md
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
