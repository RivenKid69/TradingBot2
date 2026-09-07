# NumPy 2.x Migration Plan

**Version**: 1.0
**Date**: 2025-12-21
**Status**: Planned
**Tech Debt Tracking**: `docs/reports/TECH_DEBT_REGISTRY.md#dependency-numpy-2x-migration`

---

## Overview

This document outlines the migration strategy from NumPy 1.x to NumPy 2.x for the CustodiaCloud platform.

### Current State

| Aspect | Value |
|--------|-------|
| **Current Constraint** | `numpy>=1.26.0,<2.0.0` |
| **Location** | `pyproject.toml:59` |
| **Reason for Pin** | NumPy 2.0 has breaking changes (see below) |
| **Target Migration** | NumPy 2.1+ (when ecosystem stabilizes) |

---

## NumPy 2.0 Breaking Changes (Relevant to This Codebase)

### 1. Dtype String Representation Changes

```python
# NumPy 1.x
str(np.dtype('float64'))  # 'float64'

# NumPy 2.0
str(np.dtype('float64'))  # 'float64' (unchanged for common types)
```

**Impact**: Low - primarily affects string parsing of dtype representations.

### 2. Array API Standard Alignment

NumPy 2.0 aligns with the Array API standard, affecting:

- `np.bool` → `np.bool_` (already deprecated in 1.x)
- `np.int` → `np.int_` (already deprecated in 1.x)
- `np.float` → `np.float64` (already deprecated in 1.x)

**Impact**: Medium - codebase search shows no usage of deprecated aliases.

### 3. Copy Semantics Changes

```python
# NumPy 1.x
arr.copy()  # Always deep copy

# NumPy 2.0
arr.copy()  # May return view for immutable data
```

**Impact**: Low - our code uses explicit `np.array(..., copy=True)` where needed.

### 4. String/Bytes Dtype Changes

NumPy 2.0 introduces `StringDtype` with variable-length strings.

**Impact**: Low - we primarily use numeric dtypes.

### 5. Removed Functions

| Removed | Replacement |
|---------|-------------|
| `np.mat` | `np.asmatrix` |
| `np.rank` | `np.ndim` |
| `np.PINF`, `np.NINF` | `np.inf`, `-np.inf` |

**Impact**: Low - codebase search shows no usage.

---

## Migration Prerequisites

### 1. Dependency Compatibility Check

Before migration, verify ecosystem compatibility:

| Dependency | NumPy 2.x Support | Status |
|------------|-------------------|--------|
| pandas | 2.2+ supports NumPy 2.x | Ready |
| PyTorch | 2.3+ supports NumPy 2.x | Ready |
| scipy | 1.13+ supports NumPy 2.x | Ready |
| stable-baselines3 | TBD | Check before migration |
| arch | 6.3+ supports NumPy 2.x | Ready |
| Cython extensions | Rebuild required | Action needed |

### 2. Cython Rebuild Requirement

All Cython modules must be rebuilt against NumPy 2.x headers:

- `lob_state_cython.pyx`
- `execlob_book.pyx`
- `coreworkspace.pyx`
- All modules in `lob/`, `optimizers/`

---

## Migration Steps

### Phase 1: Compatibility Testing (Pre-Migration)

1. **Create compatibility branch**

   ```bash
   git checkout -b feature/numpy-2x-compatibility
   ```

2. **Update constraint temporarily**

   ```toml
   # pyproject.toml
   "numpy>=2.0.0,<3.0.0",
   ```

3. **Rebuild Cython extensions**

   ```bash
   make clean && make build
   ```

4. **Run full test suite**

   ```bash
   make test
   ```

5. **Document failures** in `docs/migration/NUMPY_2X_ISSUES.md`

### Phase 2: Code Updates

1. **Replace deprecated aliases** (if any found)

   ```python
   # Before
   np.float  # Deprecated

   # After
   np.float64  # Explicit
   ```

2. **Update dtype string parsing** (if needed)

3. **Review copy semantics** in critical paths

### Phase 3: Dependency Updates

1. Update `pyproject.toml`:

   ```toml
   "numpy>=2.0.0,<3.0.0",
   ```

2. Regenerate lockfiles:

   ```bash
   make lock-cpu lock-gpu
   ```

3. Update CI to test both NumPy 1.x and 2.x (transitional period)

### Phase 4: Validation

1. Full test suite pass
2. Benchmark comparison (performance regression check)
3. Integration tests with all adapters
4. Backtest result comparison (numerical stability)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Numerical precision changes | Low | Medium | Backtest comparison before/after |
| Cython ABI breakage | Medium | High | Full rebuild + test coverage |
| Third-party incompatibility | Low | Medium | Pin problematic deps temporarily |
| Performance regression | Low | Medium | Benchmark suite comparison |

---

## Timeline

| Milestone | Target | Status |
|-----------|--------|--------|
| Compatibility assessment | Q1 2026 | Planned |
| Test suite on NumPy 2.x | Q1 2026 | Planned |
| Code updates | Q2 2026 | Planned |
| Production migration | Q2 2026 | Planned |

**Note**: Timeline is aspirational and depends on ecosystem stability. NumPy 2.x adoption across the ML ecosystem is still maturing.

---

## Monitoring

Post-migration monitoring:

- [ ] No new NumPy deprecation warnings in CI
- [ ] Backtest results within numerical tolerance (1e-6)
- [ ] No performance regression >5%
- [ ] All adapters functional

---

## References

- [NumPy 2.0 Migration Guide](https://numpy.org/doc/stable/numpy_2_0_migration_guide.html)
- [NumPy 2.0 Release Notes](https://numpy.org/doc/stable/release/2.0.0-notes.html)
- [Array API Standard](https://data-apis.org/array-api/latest/)

---

*This document follows the Documentation Canon - targets are aspirational, not commitments.*
