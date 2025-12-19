# Technical Debt Registry

**Version**: 1.0
**Date**: 2025-12-19
**Status**: Active
**Canon Reference**: `docs/DOCUMENTATION_CANON_DESIGN.md`

---

## Purpose

This registry tracks all known technical debt items with their control artifacts and status.
Each item follows the Documentation Canon requirement for honest disclosure without absolute claims.

---

## Registry Format

Each entry contains:
- **ID**: Unique identifier for reference
- **Category**: Architecture, Data/ML, Testing/Quality, Reliability/Operations, Security, Docs/Drift, Other
- **Severity**: High, Medium, Low
- **Status**: Controlled (with artifacts) / Open / Closed
- **Control Artifact**: What proves this is managed

---

## Architecture

### arch-train-monolith {#arch-train-monolith}

| Field | Value |
|-------|-------|
| **Location** | `distributional_ppo.py:45-77` |
| **Severity** | High |
| **Description** | Monolithic train() method (~4000 lines) with partial refactoring |
| **Status** | Controlled |
| **Control Artifact** | Header documentation, `tests/COMPREHENSIVE_TEST_REPORT.md`, static analysis via `radon cc` |
| **Metrics** | Cyclomatic complexity tracked, ~85% critical path coverage |

---

## Data/ML

### L1-slippage {#L1-slippage}

| Field | Value |
|-------|-------|
| **Location** | `docs/SIMULATION_LIMITATIONS.md#L1`, `execution_providers.py:LOBSlippageProvider` |
| **Severity** | High |
| **Description** | LOB slippage estimation uses spread-based stub, not order book depth |
| **Status** | Controlled |
| **Control Artifact** | TCA calibration report required before live deployment |
| **Mitigation** | StatisticalSlippageProvider available; conservative multipliers recommended |

### L2-fill {#L2-fill}

| Field | Value |
|-------|-------|
| **Location** | `docs/SIMULATION_LIMITATIONS.md#L2`, `execution_providers.py:LOBFillProvider` |
| **Severity** | High |
| **Description** | LOB fill simulation uses OHLCV fallback, no queue position modeling |
| **Status** | Controlled |
| **Control Artifact** | Fill-rate comparison report required before live deployment |
| **Mitigation** | OHLCV fallback provides conservative baseline |

### L3-impact {#L3-impact}

| Field | Value |
|-------|-------|
| **Location** | `docs/SIMULATION_LIMITATIONS.md#L3` |
| **Severity** | High |
| **Description** | Market impact not implemented (no permanent/temporary decomposition) |
| **Status** | Controlled |
| **Control Artifact** | Market impact validation report required for institutional-size orders |
| **Mitigation** | Conservative slippage estimates include implicit impact; ADV limits recommended |

### L4-tif {#L4-tif}

| Field | Value |
|-------|-------|
| **Location** | `docs/SIMULATION_LIMITATIONS.md#L4`, `OrderBook.cpp:70-79` |
| **Severity** | Medium |
| **Description** | IOC (Immediate-Or-Cancel) behaves as GTC in simulation |
| **Status** | Controlled |
| **Control Artifact** | Conformance test suite (T2b milestone) |
| **Mitigation** | IOC avoidance recommended until T2b implementation |

### quantile-uniform {#quantile-uniform}

| Field | Value |
|-------|-------|
| **Location** | `distributional_ppo.py:3888-3895` |
| **Severity** | Medium |
| **Description** | Quantile critic assumes uniform quantile levels; IQN migration requires validation |
| **Status** | Controlled |
| **Control Artifact** | `tests/test_distributional_ppo_quantile_loss.py` |
| **Mitigation** | Current uniform assumption is validated; IQN is roadmap item |

---

## Testing/Quality

### testing-ppo-coverage {#testing-ppo-coverage}

| Field | Value |
|-------|-------|
| **Location** | `tests/COMPREHENSIVE_TEST_REPORT.md` |
| **Severity** | High |
| **Description** | distributional_ppo.py coverage at 35% baseline (168 functions, 58 tested) |
| **Status** | Controlled |
| **Control Artifact** | `tests/COMPREHENSIVE_TEST_REPORT.md`, CI pytest-cov runs |
| **Tracking** | Priority roadmap in report; critical paths at ~85% coverage |

### testing-tif-conformance {#testing-tif-conformance}

| Field | Value |
|-------|-------|
| **Location** | `OrderBook.cpp:75-79` |
| **Severity** | Medium |
| **Description** | Matching engine conformance tests marked as T2b milestone |
| **Status** | Controlled |
| **Control Artifact** | `tests/cpp/test_orderbook_tif_conformance.cpp` (stub with GTEST_SKIP; T2b milestone) |
| **Tracking** | Linked to IOC implementation in L4-tif |

---

## Reliability/Operations

### ops-dora-gaps {#ops-dora-gaps}

| Field | Value |
|-------|-------|
| **Location** | `docs/DORA_OPERATIONAL_RESILIENCE_PLAN.md:976-981` |
| **Severity** | High |
| **Description** | Incident Management, Backup Recovery, ICT Business Continuity marked as ROADMAP |
| **Status** | Controlled |
| **Control Artifact** | `docs/runbooks/`, `docs/operations/ON_CALL_CAPACITY_VALIDATION.md` |
| **Note** | Pre-revenue stage; components require production infrastructure |

### ops-dr-testing {#ops-dr-testing}

| Field | Value |
|-------|-------|
| **Location** | `docs/security/TRUST_CENTER.md:188-204` |
| **Severity** | High |
| **Description** | DR testing not yet conducted; RTO/RPO unvalidated |
| **Status** | Controlled |
| **Control Artifact** | `docs/runbooks/` (documented procedures pending validation) |
| **Note** | Honest disclosure per Canon; validation requires infrastructure deployment |

### ops-incident-response {#ops-incident-response}

| Field | Value |
|-------|-------|
| **Location** | `docs/security/TRUST_CENTER.md:227-246` |
| **Severity** | Medium |
| **Description** | Incident response limited to business hours; 24/7 coverage pending hiring |
| **Status** | Controlled |
| **Control Artifact** | `docs/operations/ON_CALL_CAPACITY_VALIDATION.md` |
| **Note** | Current capacity honestly disclosed; expansion requires funding |

### ops-metrics-baseline {#ops-metrics-baseline}

| Field | Value |
|-------|-------|
| **Location** | `docs/ENTERPRISE_ADOPTION_RISK_MITIGATION.md:791-797` |
| **Severity** | Medium |
| **Description** | Operational metrics pending customer deployment; no track record yet |
| **Status** | Controlled |
| **Control Artifact** | SLO/SLI dashboard (planned for post-deployment) |
| **Note** | Pre-deployment stage honestly disclosed |

---

## Security

### security-external-audits {#security-external-audits}

| Field | Value |
|-------|-------|
| **Location** | `docs/security/TRUST_CENTER.md:46-56` |
| **Severity** | Medium |
| **Description** | Pen-test and SOC2 audits are roadmap items; not yet conducted |
| **Status** | Controlled |
| **Control Artifact** | `docs/security/SECURITY_ROADMAP.md` |
| **Note** | Roadmap items honestly disclosed; funding-dependent |

---

## Docs/Drift

### docs-ci-coverage-gate {#docs-ci-coverage-gate}

| Field | Value |
|-------|-------|
| **Location** | `docs/design/CCEA_CLOUD/CI_GUARDRAILS.md:30-35` |
| **Severity** | Medium |
| **Description** | PM-005 coverage gate at 80% is a target, not currently enforced |
| **Status** | Controlled |
| **Control Artifact** | CI_GUARDRAILS.md now accurately reflects target vs implemented |
| **Note** | Docs corrected to state "TARGET" per Documentation Canon |

---

## Other

### options-max-profit {#options-max-profit}

| Field | Value |
|-------|-------|
| **Location** | `adapters/ib/options_combo.py:280-300` |
| **Severity** | Low |
| **Description** | get_max_profit() only implemented for IRON_CONDOR; others return None |
| **Status** | Controlled |
| **Control Artifact** | Docstring documents scope limitation |
| **Note** | Returning None is conservative (no false profit estimates) |

---

## Summary Statistics

| Category | High | Medium | Low | Total | All Controlled |
|----------|------|--------|-----|-------|----------------|
| Architecture | 1 | 0 | 0 | 1 | Yes |
| Data/ML | 3 | 2 | 0 | 5 | Yes |
| Testing/Quality | 1 | 1 | 0 | 2 | Yes |
| Reliability/Operations | 2 | 2 | 0 | 4 | Yes |
| Security | 0 | 1 | 0 | 1 | Yes |
| Docs/Drift | 0 | 1 | 0 | 1 | Yes |
| Other | 0 | 0 | 1 | 1 | Yes |
| **TOTAL** | **7** | **7** | **1** | **15** | **Yes** |

---

## Document Control

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-19 | Initial registry with 15 items from tech debt closure |
| 1.1 | 2025-12-19 | Created missing control artifacts: SECURITY_ROADMAP.md, test_orderbook_tif_conformance.cpp stub |

**Review Frequency**: Monthly or upon significant changes
**Owner**: Engineering
**Classification**: Internal

---

*This document follows the Documentation Canon - no absolute claims, honest disclosure of limitations.*
