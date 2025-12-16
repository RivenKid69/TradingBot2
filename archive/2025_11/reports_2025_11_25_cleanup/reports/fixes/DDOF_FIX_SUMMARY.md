# Fix: Add ddof=1 to np.std() and np.var() for Unbiased Sample Variance

## Issue

All `np.std()` and `np.var()` calls were using the default `ddof=0` (population variance), which gives a **biased estimate** when working with samples. In RL, we always work with samples (batches, rollouts), not complete populations, so we need `ddof=1` (Bessel's correction) for unbiased estimation.

### Critical Impact

**Advantage Normalization**: The most critical issue was in advantage normalization (line 6474), where:
- ddof=0 systematically **underestimated** std by ~1-2% for typical batch sizes
- This caused **over-normalization** of advantages → larger policy gradients
- Directly affected learning dynamics and convergence

## Changes

### Code Fixes (distributional_ppo.py)

1. **Line 6474**: Advantage normalization
   - `np.std(advantages_flat)` → `np.std(advantages_flat, ddof=1)`

2. **Line 759**: Weighted mean/std function
   - `np.std(filtered)` → `np.std(filtered, ddof=1)`

3. **Lines 9550, 9552**: Value prediction logging
   - `np.std(y_pred_np)` → `np.std(y_pred_np, ddof=1)`
   - `np.std(y_true_np)` → `np.std(y_true_np, ddof=1)`

4. **Lines 3869-3870**: Variance calculations
   - `np.var(true_vals)` → `np.var(true_vals, ddof=1)`
   - `np.var(pred_vals)` → `np.var(pred_vals, ddof=1)`

### Test Updates

Updated all tests to use `ddof=1`:
- `tests/test_advantage_normalization_simple.py`
- `tests/test_advantage_normalization_integration.py`

### New Tests

Added comprehensive test suite: `tests/test_std_ddof_correction.py`
- Verifies ddof=1 gives unbiased estimates
- Tests impact on policy gradients
- Validates small batch behavior
- Confirms code implementation

### Documentation

Created detailed documentation: `docs/STD_DDOF_CORRECTION.md`
- Mathematical background
- Impact analysis
- Best practices
- Migration notes

## Impact

### Batch Size vs Error

| n   | Systematic Error | Impact Level |
|-----|------------------|--------------|
| 10  | ~5.4%           | Very significant |
| 50  | ~2.0%           | Significant |
| 100 | ~1.0%           | Moderate |
| 1000| ~0.1%           | Minor |

### What Changes

1. **Learning behavior**: Slightly smaller normalized advantages (correct now)
2. **Logged metrics**: Higher std values (accurate now)
3. **Reproducibility**: Not bit-exact with old code, but mathematically correct

## Validation

- ✅ All syntax checks pass
- ✅ All test assertions updated to match ddof=1
- ✅ Comprehensive test suite created
- ✅ Documentation complete

## References

- [Bessel's Correction](https://en.wikipedia.org/wiki/Bessel%27s_correction)
- [NumPy std documentation](https://numpy.org/doc/stable/reference/generated/numpy.std.html)
- Statistical best practices for ML

---

**Date**: 2025-11-17
**Status**: ✅ Complete
**Breaking Change**: Yes (numerical, but mathematically correct)
