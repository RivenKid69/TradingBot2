# Critical Bugs Prevention Guide

**Date**: 2025-11-20
**Purpose**: Prevent recurrence of critical bugs discovered in November 2025

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Critical Bug Patterns](#critical-bug-patterns)
3. [Prevention Guidelines](#prevention-guidelines)
4. [Code Review Checklist](#code-review-checklist)
5. [Testing Requirements](#testing-requirements)
6. [References](#references)

---

## Overview

This document captures lessons learned from three critical bugs discovered on 2025-11-20:

1. **Temporal Causality Violation** in data simulation
2. **Cross-Symbol Contamination** in feature normalization
3. **Inverted Quantile Loss** in distributional value function

**Full details**: [CRITICAL_FIXES_REPORT.md](../CRITICAL_FIXES_REPORT.md)

---

## Critical Bug Patterns

### Pattern 1: Temporal Causality Violations

**What**: Breaking time-ordered structure of observations

**Example Bug**: Returning stale data with old timestamp instead of current timestamp

**Why Critical**:
- Violates fundamental RL assumption of temporally-ordered observations
- Model learns incorrect temporal dependencies
- Breaks causality chain in decision-making

**How to Prevent**:
```python
# ❌ WRONG: Reusing old object with old timestamp
yield prev_bar  # Has ts from previous time step

# ✅ CORRECT: Create new object with current timestamp
yield Bar(
    ts=current_ts,  # Current time
    symbol=prev_bar.symbol,
    # ... copy other fields
)
```

**Detection**:
- Test that timestamps are monotonically increasing
- Verify timestamp matches expected current time
- Check for any timestamp reuse patterns

**Test Example**:
```python
def test_timestamps_monotonic():
    bars = list(source.stream_bars(symbols, interval_ms))
    timestamps = [bar.ts for bar in bars]
    assert timestamps == sorted(timestamps), "Timestamps must be monotonic"

    # Verify timestamp spacing
    for i in range(1, len(timestamps)):
        diff = timestamps[i] - timestamps[i-1]
        assert diff == interval_ms, f"Expected {interval_ms}ms gap, got {diff}ms"
```

---

### Pattern 2: Cross-Boundary Data Contamination

**What**: Data from one entity leaking into another during batch operations

**Example Bug**: Applying `shift()` after concatenating multiple symbols

**Why Critical**:
- Corrupts statistical properties (mean, std, etc.)
- Violates independence assumptions between entities
- Creates spurious correlations

**How to Prevent**:
```python
# ❌ WRONG: Global operation after concatenation
all_data = pd.concat([df1, df2, df3])
all_data['feature'] = all_data['feature'].shift(1)  # Last row of df1 → First row of df2!

# ✅ CORRECT: Per-entity operation before concatenation
shifted_dfs = []
for df in [df1, df2, df3]:
    df_copy = df.copy()
    df_copy['feature'] = df_copy['feature'].shift(1)  # Contained within entity
    shifted_dfs.append(df_copy)
all_data = pd.concat(shifted_dfs)

# ✅ ALSO CORRECT: Use groupby for transform
all_data['feature'] = all_data.groupby('entity_id')['feature'].shift(1)
```

**Detection**:
- Test boundary conditions explicitly
- Verify first row of each entity has NaN/expected value after shift
- Check statistics are computed correctly per-entity

**Test Example**:
```python
def test_no_cross_entity_contamination():
    # Create distinct data
    df1 = pd.DataFrame({'symbol': ['A', 'A'], 'value': [10, 20]})
    df2 = pd.DataFrame({'symbol': ['B', 'B'], 'value': [100, 200]})

    result = process_multi_symbol([df1, df2])

    # First row of B should NOT contain data from A
    first_b_row = result[result['symbol'] == 'B'].iloc[0]
    assert pd.isna(first_b_row['value_shifted']), \
        "First row of symbol B should have NaN after shift, not data from symbol A"
```

---

### Pattern 3: Mathematical Formula Inversions

**What**: Using incorrect mathematical formulas, especially with asymmetric operations

**Example Bug**: Using `Q - T` instead of `T - Q` in quantile loss

**Why Critical**:
- Inverts asymmetric penalties
- Fundamentally changes optimization objective
- Subtle bug that's hard to detect without domain knowledge

**How to Prevent**:

1. **Document mathematical formulas with references**:
```python
def quantile_huber_loss(targets, predicted, tau, kappa):
    """Quantile regression loss with Huber penalty.

    Formula (Dabney et al. 2018, Equation 10):
        ρ_τ(u) = |τ - I{u < 0}| · L_κ(u)
    where:
        u = T - Q  (target - predicted)  ← CRITICAL: Order matters!
        I{·} = indicator function
        L_κ(·) = Huber loss with threshold κ

    References:
        Dabney et al. (2018) "Distributional Reinforcement Learning
        with Quantile Regression" (QR-DQN paper)
    """
    delta = targets - predicted  # ✅ CORRECT: T - Q
    # NOT: delta = predicted - targets  # ❌ WRONG: Q - T

    # Rest of implementation...
```

2. **Add unit tests with known ground truth**:
```python
def test_quantile_loss_asymmetry():
    """Verify correct asymmetric penalties."""
    tau = 0.25

    # Case 1: Underestimation (Q < T)
    # Should penalize with τ = 0.25
    predicted = torch.tensor([1.0])
    target = torch.tensor([2.0])
    loss_under = quantile_loss(target, predicted, tau)

    # Case 2: Overestimation (Q > T)
    # Should penalize with (1-τ) = 0.75
    predicted = torch.tensor([2.0])
    target = torch.tensor([1.0])
    loss_over = quantile_loss(target, predicted, tau)

    # For low quantile (τ < 0.5), overestimation should be penalized MORE
    assert loss_over > loss_under, \
        "For τ=0.25, overestimation penalty (0.75) > underestimation penalty (0.25)"
```

3. **Cross-reference with academic papers**:
- Always cite original paper in docstring
- Include equation numbers
- Verify implementation matches paper's notation

**Detection**:
- Unit tests with known analytical solutions
- Compare with reference implementations
- Peer review by domain experts
- Gradient checks for optimization objectives

---

## Prevention Guidelines

### 1. Temporal Operations

**ALWAYS**:
- ✅ Maintain strict timestamp ordering
- ✅ Use current timestamp for current observations
- ✅ Test timestamp monotonicity
- ✅ Verify time gaps match expected intervals

**NEVER**:
- ❌ Reuse old objects with old timestamps
- ❌ Mix past and present timestamps
- ❌ Assume timestamps will be "fixed later"

### 2. Batch Operations on Multi-Entity Data

**ALWAYS**:
- ✅ Apply operations per-entity BEFORE concatenation
- ✅ Use `groupby()` for operations on concatenated data
- ✅ Test boundary conditions between entities
- ✅ Verify first row of each entity is independent

**NEVER**:
- ❌ Apply global operations to concatenated multi-entity data
- ❌ Assume pandas operations respect entity boundaries
- ❌ Skip testing boundary conditions

### 3. Mathematical Implementations

**ALWAYS**:
- ✅ Document formulas with academic references
- ✅ Include equation numbers from papers
- ✅ Add unit tests with analytical solutions
- ✅ Verify asymmetric operations have correct sign
- ✅ Cross-check with reference implementations

**NEVER**:
- ❌ Implement formulas from memory
- ❌ Assume "it looks right"
- ❌ Skip unit tests for mathematical functions
- ❌ Ignore sign conventions in papers

### 4. Default Values

**ALWAYS**:
- ✅ Default to mathematically correct behavior
- ✅ Document why defaults are chosen
- ✅ Provide explicit flags to override if needed
- ✅ Warn users about non-default (legacy) behavior

**NEVER**:
- ❌ Default to incorrect behavior "for backward compatibility"
- ❌ Hide correct behavior behind flags
- ❌ Make users opt-in to correctness

---

## Code Review Checklist

When reviewing code that involves:

### ✅ Time-Series Data
- [ ] Are timestamps strictly monotonically increasing?
- [ ] Is current timestamp used for current observations?
- [ ] Are there any timestamp reuse patterns?
- [ ] Do tests verify timestamp ordering?

### ✅ Multi-Entity Operations (symbols, agents, envs, etc.)
- [ ] Are operations applied per-entity before concatenation?
- [ ] If concatenated, is `groupby()` used for transformations?
- [ ] Are boundary conditions tested?
- [ ] Does first row of each entity have correct values?

### ✅ Mathematical Formulas
- [ ] Is there a reference to the original paper?
- [ ] Are equation numbers included?
- [ ] Are signs/orders correct for asymmetric operations?
- [ ] Do unit tests verify against known solutions?
- [ ] Has a domain expert reviewed the math?

### ✅ Statistical Operations (shift, diff, rolling, etc.)
- [ ] Are entity boundaries respected?
- [ ] Is the operation documented with expected behavior?
- [ ] Are edge cases tested (first row, last row, NaN handling)?

### ✅ Default Values
- [ ] Do defaults represent best practices?
- [ ] Is incorrect behavior deprecated, not default?
- [ ] Are legacy options documented with warnings?

---

## Testing Requirements

### Mandatory Tests for Time-Series Code

```python
def test_timestamp_monotonicity():
    """Verify timestamps are strictly increasing."""
    pass

def test_timestamp_spacing():
    """Verify gaps between timestamps match expected interval."""
    pass

def test_current_timestamp_used():
    """Verify current observations use current timestamps."""
    pass
```

### Mandatory Tests for Multi-Entity Code

```python
def test_first_row_independence():
    """Verify first row of each entity is independent."""
    pass

def test_no_cross_entity_leakage():
    """Verify operations don't leak data across entity boundaries."""
    pass

def test_per_entity_statistics():
    """Verify statistics computed correctly per entity."""
    pass
```

### Mandatory Tests for Mathematical Functions

```python
def test_formula_with_analytical_solution():
    """Verify against known analytical solution."""
    pass

def test_asymmetric_penalties():
    """Verify correct asymmetry for asymmetric losses."""
    pass

def test_gradient_correctness():
    """Verify gradients match analytical computation."""
    pass

def test_edge_cases():
    """Verify behavior at boundaries (zero, infinity, etc.)."""
    pass
```

---

## References

### Academic Papers

1. **Dabney et al. (2018)** - "Distributional Reinforcement Learning with Quantile Regression"
   - Quantile loss formula (Equation 10)
   - Critical for distributional RL

2. **Sutton & Barto (2018)** - "Reinforcement Learning: An Introduction"
   - Temporal causality in RL
   - Observation ordering requirements

3. **Koenker & Bassett (1978)** - "Regression Quantiles"
   - Original quantile regression paper
   - Foundation for quantile loss

### Internal Documentation

- [CRITICAL_FIXES_REPORT.md](../CRITICAL_FIXES_REPORT.md) - Detailed analysis of critical bugs
- [CHANGELOG.md](../CHANGELOG.md) - Bug fix history
- [CLAUDE.md](../CLAUDE.md) - Main project documentation

### Test Files

- [tests/test_stale_bar_temporal_causality.py](../tests/test_stale_bar_temporal_causality.py)
- [tests/test_normalization_cross_symbol_contamination.py](../tests/test_normalization_cross_symbol_contamination.py)
- [tests/test_quantile_loss_formula_default.py](../tests/test_quantile_loss_formula_default.py)

---

## Quick Reference: "When to Be Extra Careful"

🚨 **HIGH RISK OPERATIONS** - Require extra scrutiny:

1. **`.shift()`, `.diff()`, `.rolling()` on multi-entity data**
   - Risk: Cross-entity contamination
   - Solution: Apply per-entity or use `groupby()`

2. **Timestamp manipulation in data pipelines**
   - Risk: Temporal causality violation
   - Solution: Always use current timestamp for current data

3. **Mathematical formulas with asymmetric penalties**
   - Risk: Sign inversion, order confusion
   - Solution: Document with paper reference, test asymmetry

4. **Concatenation followed by transformation**
   - Risk: Boundary leakage
   - Solution: Transform first, then concatenate

5. **Default values for correctness flags**
   - Risk: Defaulting to incorrect behavior
   - Solution: Default to correct, flag for legacy

---

## Summary

**Golden Rules**:

1. 🕐 **Temporal Causality**: Current time → current timestamp
2. 🔀 **Entity Boundaries**: Per-entity operations, then concat
3. 📐 **Math Correctness**: Paper reference + unit tests
4. ✅ **Test Boundaries**: First row, last row, edge cases
5. 🎯 **Default to Correct**: Never default to bugs

**When in Doubt**:
- Read the paper
- Write a test
- Ask for review
- Check boundary conditions

---

**Last Updated**: 2025-11-20
**Status**: Active Prevention Guide
**Next Review**: Before major refactoring or when adding similar features
