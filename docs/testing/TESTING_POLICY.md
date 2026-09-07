# Testing Policy

**Version**: 1.0.0
**Date**: 2025-12-21
**Status**: Active
**Canon Reference**: `docs/DOCUMENTATION_CANON_DESIGN.md`

---

## Purpose

This document defines the testing standards, mock usage policy, and test categorization for CustodiaCloud.
It serves as a control artifact for the tech debt item `testing-mock-density`.

---

## Test Categories

### Unit Tests

- **Purpose**: Test individual functions/classes in isolation
- **Mock Usage**: Allowed for external dependencies (APIs, databases, file systems)
- **Location**: `tests/test_*.py` (flat structure)
- **Coverage Target**: Critical paths should have unit test coverage

### Integration Tests

- **Purpose**: Test component interactions
- **Mock Usage**: Minimal; prefer real components where feasible
- **Location**: `tests/integration/` or `tests/*_integration*.py`
- **Coverage Target**: Key workflows should have integration coverage

### End-to-End Tests

- **Purpose**: Test complete user workflows
- **Mock Usage**: None; use real or sandbox environments
- **Location**: `tests/e2e/` or `tests/*_e2e*.py`
- **Coverage Target**: Critical business flows

---

## Mock Usage Policy

### Controlled Tech Debt: testing-mock-density

**Current State**: ~5,580 mock/patch usages across 344 test files (~16 per file average)

**Rationale for High Mock Density**:

1. External API dependencies (Binance, Alpaca, IB, OANDA) require mocking
2. Hardware dependencies (GPU, network) require mocking
3. Time-sensitive tests require deterministic behavior
4. CI environment lacks live broker connections

### Acceptable Mock Usage

| Category | Mock Usage | Rationale |
|----------|------------|-----------|
| External APIs | Always | No live credentials in CI |
| Time/Clock | Always | Deterministic tests |
| File System (temp) | Optional | Prefer real temp files |
| Database | Prefer real SQLite | Fast, deterministic |
| Network I/O | Always | Reliability, speed |
| GPU/Hardware | Always | CI environment |

### Unacceptable Mock Usage

- **Core business logic**: Never mock the function under test
- **Data transformers**: Prefer real implementations
- **Risk controls**: Test with real implementations
- **Trading logic**: Minimal mocking; integration tests required

### Mock Density Monitoring

CI tracks mock usage via code analysis:

```bash
# Count mock usages in tests
grep -r "mock\|patch\|MagicMock" tests/ | wc -l
```

**Threshold Policy**:

- Current: ~5,580 (acceptable for external dependency mocking)
- Alert if growth exceeds 20% per quarter without justification
- New test files should document mock rationale if > 20 mocks

---

## Integration Test Requirements

### Critical Paths Requiring Integration Tests

1. **Training Pipeline**
   - Data loading -> Feature extraction -> Model training
   - Tests: `tests/test_training_*.py`

2. **Backtest Pipeline**
   - Data loading -> Strategy execution -> Report generation
   - Tests: `tests/test_service_backtest*.py`

3. **CCEA Protocol**
   - Cloud -> Agent communication
   - Tests: `tests/ccea/test_e2e.py`

4. **Risk Controls**
   - Kill switch, pre-trade checks, position limits
   - Tests: `tests/core/test_*.py`

---

## Test Quality Metrics

### Tracked in CI

| Metric | Target | Status | Artifact |
|--------|--------|--------|----------|
| Test pass rate | 100% | Enforced | pytest results |
| Skip count | < 100 | Tracked | `skip-report.json` |
| Coverage (critical) | > 80% | Tracked (see note) | `coverage.xml`, `coverage-report.json` |
| Mock density | Stable | Manual review | Code analysis |

> **Coverage Gate Status (PM-005)**: Coverage is tracked and reported as CI artifacts (`coverage.xml`, `coverage-report.json`). The 80% threshold is a target goal; enforcement as a merge-blocking gate is planned when baseline coverage stabilizes above 70%. See `docs/design/CCEA_CLOUD/CI_GUARDRAILS.md` for implementation status. Tech Debt: `docs/reports/TECH_DEBT_REGISTRY.md#docs-ci-coverage-gate`

### Review Triggers

- Skip count > 100: Review and document reasons
- New test file with > 30 mocks: Requires review comment
- Coverage drop > 5%: Review required (automated gate planned per PM-005)

---

## Test Documentation Requirements

### For New Test Files

```python
"""
Test module for [component].

Test Categories:
- Unit tests: [list functions]
- Integration tests: [list if any]

Mock Usage:
- [list mocked dependencies and why]

Tech Debt Reference: [if applicable]
"""
```

---

## Control Artifacts

| Artifact | Location | Purpose |
|----------|----------|---------|
| Skip report | CI: `skip-report.json` | Track skipped tests |
| Coverage report | CI: `coverage-report/` | Track coverage |
| This policy | `docs/testing/TESTING_POLICY.md` | Mock density control |

---

## Tech Debt Reference

| ID | Registry Reference | Status |
|----|-------------------|--------|
| testing-mock-density | `docs/reports/TECH_DEBT_REGISTRY.md#testing-mock-density` | Controlled |

---

## Document Control

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-21 | Initial policy documenting mock usage standards |

**Review Frequency**: Quarterly or upon significant test infrastructure changes
**Owner**: Engineering
**Classification**: Internal

---

*This document follows the Documentation Canon - no absolute claims, honest disclosure of limitations.*
