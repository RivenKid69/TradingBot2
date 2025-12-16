# UPGD Learning Rate Multiplier Bug Fix

**Date:** 2025-11-20
**Type:** Bug Fix
**Severity:** High
**Impact:** Breaking Change

## Summary

Fixed a critical bug in UPGD and AdaptiveUPGD optimizers where the learning rate was effectively **2x higher** than specified due to an incorrect alpha multiplier (`-2.0*lr` instead of `-1.0*lr`).

## Problem Description

### Bug Location
- **File 1:** `optimizers/upgd.py`, line 154
- **File 2:** `optimizers/adaptive_upgd.py`, line 232

### Root Cause

Both optimizers used `alpha=-2.0 * group["lr"]` in the parameter update step:

```python
# BUGGY CODE (before fix)
p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
    (p.grad.data + noise) * (1 - scaled_utility),
    alpha=-2.0 * group["lr"],  # ❌ WRONG: 2x learning rate
)
```

This resulted in the effective learning rate being **double** what users specified.

### Inconsistency

The bug was inconsistent with:
1. **Documented formula in code comments**: States `param -= lr * grad`, but implements `param -= 2*lr * grad`
2. **Mathematical derivation**: The "rearranged" formula in comments is incorrect
3. **UPGDW implementation**: Correctly uses `alpha=-1.0*lr` (or `-step_size`)

### Evidence

Empirical testing showed:
- **Before fix**: `AdaptiveUPGD_change / UPGDW_change = 2.0000` (exact ratio)
- **After fix**: `AdaptiveUPGD_change / UPGDW_change = 1.0000` (perfect match)

## Fix Applied

###  Changes

**File: `optimizers/upgd.py`**
```python
# Line 151: Updated comment
-# Rearranged: param *= (1 - lr * weight_decay); param -= 2*lr * (grad + noise) * (1 - scaled_utility)
+# Rearranged: param *= (1 - lr * weight_decay); param -= lr * (grad + noise) * (1 - scaled_utility)

# Line 154: Changed alpha from -2.0 to -1.0
p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
    (p.grad.data + noise) * (1 - scaled_utility),
-    alpha=-2.0 * group["lr"],
+    alpha=-1.0 * group["lr"],  # BUGFIX: Changed from -2.0 to -1.0
)
```

**File: `optimizers/adaptive_upgd.py`**
```python
# Line 232: Changed alpha from -2.0 to -1.0
p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
    perturbed_update,
-    alpha=-2.0 * group["lr"],
+    alpha=-1.0 * group["lr"],  # BUGFIX: Changed from -2.0 to -1.0
)
```

## Impact and Migration

### Breaking Change ⚠️

This is a **breaking change** that affects training behavior.

### For New Models
- ✅ No action needed
- Use learning rates as normal (e.g., `lr=1e-4`)

### For Existing Models/Checkpoints

Users have two options:

#### Option 1: Double the Learning Rate (Recommended)
Maintain the same effective learning rate by doubling the lr parameter:

```python
# Before fix (effective lr = 2 * 1e-4 = 2e-4)
optimizer = AdaptiveUPGD(model.parameters(), lr=1e-4)

# After fix (to maintain same effective lr = 2e-4)
optimizer = AdaptiveUPGD(model.parameters(), lr=2e-4)
```

#### Option 2: Retrain with Correct Learning Rate
Use the corrected (lower) effective learning rate:

```python
# After fix (true lr = 1e-4)
optimizer = AdaptiveUPGD(model.parameters(), lr=1e-4)
```

This may require re-tuning hyperparameters, but results in more predictable behavior.

### Behavioral Changes

**Before Fix:**
- Specifying `lr=1e-4` resulted in `effective_lr=2e-4`
- Training was more aggressive than intended
- May have caused instability with standard learning rates

**After Fix:**
- Specifying `lr=1e-4` results in `effective_lr=1e-4` (as expected)
- Consistent with standard optimizer conventions (Adam, AdamW, SGD)
- Consistent with UPGDW implementation

## Testing

### Verification Tests Created

1. **`test_upgd_fix_verification.py`** - Manual verification script
   - ✅ Code inspection confirms `alpha=-1.0*lr`
   - ✅ AdaptiveUPGD vs UPGDW ratio = 1.0 (was 2.0)
   - ✅ All optimizers have similar magnitudes

2. **`tests/test_upgd_lr_regression.py`** - Automated regression test suite
   - ✅ `test_no_2x_multiplier_in_source` - Source code check
   - ✅ `test_adaptive_upgd_matches_upgdw_step_size` - Step size parity
   - ✅ `test_step_size_not_doubled` - Not 2x anymore
   - ✅ `test_upgd_upgdw_similar_magnitudes` - Reasonable magnitudes
   - ✅ `test_learning_rate_scaling` - Correct lr scaling

### Test Results

```
tests/test_upgd_lr_regression.py::TestUPGDLearningRateRegression::test_no_2x_multiplier_in_source PASSED [ 20%]
tests/test_upgd_lr_regression.py::TestUPGDLearningRateRegression::test_adaptive_upgd_matches_upgdw_step_size PASSED [ 40%]
tests/test_upgd_lr_regression.py::TestUPGDLearningRateRegression::test_step_size_not_doubled PASSED [ 60%]
tests/test_upgd_lr_regression.py::TestUPGDLearningRateRegression::test_upgd_upgdw_similar_magnitudes PASSED [ 80%]
tests/test_upgd_lr_regression.py::TestUPGDLearningRateRegression::test_learning_rate_scaling PASSED [100%]

============================== 5 passed in 2.02s ==============================
```

**All tests pass! ✅**

## Mathematical Justification

### Intended Formula (from code comments)
```
param_new = param_old - lr * (grad + noise) * (1 - scaled_utility) - lr * weight_decay * param_old
```

### Correct Rearrangement
```
param_new = param_old * (1 - lr * weight_decay) - lr * (grad + noise) * (1 - scaled_utility)
```

This corresponds to:
```python
p.data.mul_(1 - lr * weight_decay)  # weight decay
p.data.add_((grad + noise) * (1 - scaled_utility), alpha=-1.0 * lr)  # gradient update
```

### Incorrect Implementation (Before Fix)
```python
p.data.add_(..., alpha=-2.0 * lr)  # ❌ Wrong! This doubles the learning rate
```

This would correspond to:
```
param_new = param_old * (1 - lr * weight_decay) - 2*lr * (grad + noise) * (1 - scaled_utility)
```

Which is **NOT** equivalent to the intended formula.

## References

- **Bug Report Investigation**: `test_upgd_lr_multiplier.py`, `test_upgd_lr_bug_simple.py`, `test_upgd_lr_controlled.py`, `test_upgd_final_analysis.py`
- **Original UPGD Paper**: Elsayed & Mahmood (2024), "Utility-based Perturbed Gradient Descent", ICLR 2024
- **Related Issue**: The bug was not present in the original paper, but appeared in our implementation

## Recommendations

1. **For all users**: Update to the fixed version and adjust learning rates accordingly
2. **For active training runs**: Consider restarting with doubled learning rate (Option 1) or accepting the new (lower) effective rate (Option 2)
3. **For future development**: The regression test suite will prevent this bug from reappearing

## Files Modified

- ✏️ `optimizers/upgd.py` (line 151, 154)
- ✏️ `optimizers/adaptive_upgd.py` (line 232)

## Files Created

- ✨ `tests/test_upgd_lr_regression.py` (regression test suite)
- ✨ `test_upgd_fix_verification.py` (manual verification script)
- ✨ `test_upgd_lr_multiplier.py` (investigation script)
- ✨ `test_upgd_lr_bug_simple.py` (simple bug demonstration)
- ✨ `test_upgd_lr_controlled.py` (controlled test conditions)
- ✨ `test_upgd_final_analysis.py` (comprehensive analysis)
- ✨ `docs/reports/fixes/UPGD_LR_MULTIPLIER_FIX.md` (this document)

## Backups

Original files backed up as:
- `optimizers/upgd.py.backup`
- `optimizers/adaptive_upgd.py.backup`

---

**Status**: ✅ Fixed and Tested
**Tests**: ✅ 100% Pass Rate
**Documentation**: ✅ Complete
