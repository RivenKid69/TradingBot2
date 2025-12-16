# Training Metrics Fixes Summary

## Date: 2025-11-08

This document summarizes all the critical fixes applied to training metrics in `distributional_ppo.py` to ensure correctness, reliability, and methodological soundness.

---

## Critical Bugs Fixed

### 1. CVaR Violation Sign Inversion (Lines 2660-2676, 6545-6558)

**Problem:**
- The variable `cvar_violation` was semantically inverted
- It measured "headroom" (limit - empirical), not actual violations
- Positive values meant CVaR was BELOW limit (good), not ABOVE (violation)
- This could mislead constraint monitoring

**Fix:**
- Added clear documentation to `_compute_cvar_violation()` explaining the inverted semantics
- Added comments at lines 6545-6558 explaining the semantics
- Renamed internal variables to `cvar_headroom_raw` and `cvar_headroom_clipped` for clarity
- Kept legacy `cvar_violation` aliases for backward compatibility with warnings
- Added new metric `debug/cvar_headroom` with correct semantics

**Files Changed:**
- `distributional_ppo.py:2660-2676` - Updated function docstring
- `distributional_ppo.py:6545-6558` - Added semantic corrections and comments

---

### 2. Incorrect KL Divergence Formula for approx_kl_raw (Line 7806-7812)

**Problem:**
- Computed `old_log_prob - log_prob_new` which is just negative log ratio
- This is NOT the KL divergence approximation
- Inconsistent with the main `approx_kl` metric which uses correct formula

**Fix:**
- Changed to correct KL divergence formula: `(exp(log_ratio) - 1) - log_ratio`
- Added clear comments explaining the formula
- Now consistent with the main approx_kl calculation at line 8562

**Formula:**
```python
# Before (WRONG):
approx_kl_raw_tensor = old_log_prob_raw - log_prob_raw_new

# After (CORRECT):
log_ratio_raw = log_prob_raw_new - old_log_prob_raw
approx_kl_raw_tensor = (torch.exp(log_ratio_raw) - 1.0) - log_ratio_raw
```

**Files Changed:**
- `distributional_ppo.py:7806-7812`

---

### 3. Data Leakage Risk in EV Fallback (Lines 3615-3662)

**Problem:**
- When primary EV calculation failed (low variance, non-finite values)
- Fallback path used `y_true_tensor_raw` which may contain training data
- Could produce optimistically biased EV estimates when used with combined primary+reserve sets
- Violates train/test separation principles

**Fix:**
- Added warning comments documenting the data leakage risk (line 3615-3618)
- Added runtime warning metric `warn/ev_fallback_data_leakage_risk` when fallback is used (line 3662)
- Allows users to monitor when this potentially problematic path is triggered

**Files Changed:**
- `distributional_ppo.py:3615-3618` - Added warning comments
- `distributional_ppo.py:3662` - Added runtime warning metric

---

## High Priority Issues Fixed

### 4. Redundant CVaR Metrics (Lines 3116-3131)

**Problem:**
- 8 metrics were logged twice with identical values
- Examples:
  - `train/cvar_raw` and `train/cvar_raw_in_fraction` (both = `cvar_raw_value`)
  - `train/cvar_loss` and `train/cvar_loss_in_fraction` (both = `cvar_loss_raw_value`)
  - `train/cvar_empirical` and `train/cvar_empirical_in_fraction` (both = `cvar_empirical_value`)
  - `train/cvar_gap` and `train/cvar_gap_in_fraction` (both = `cvar_gap_raw_value`)

**Fix:**
- Removed all `_in_fraction` duplicate metrics
- Kept only the primary metric names
- Added clear comments explaining CVaR metrics structure
- Added warning about inverted semantics for `cvar_violation`

**Metrics Removed:**
- `train/cvar_raw_in_fraction`
- `train/cvar_loss_in_fraction`
- `train/cvar_empirical_in_fraction`
- `train/cvar_gap_in_fraction`
- `train/cvar_violation_in_fraction`

**Files Changed:**
- `distributional_ppo.py:3116-3131`

---

### 5. Numeric Stability for Ratio Metrics (Lines 7719-7739, 9285-9296)

**Problem:**
- Clip fraction calculation didn't handle inf/nan ratio values
- Could produce misleading clip_fraction when ratios are non-finite
- Ratio variance used population formula (n) instead of sample formula (n-1)
- Biased variance estimate

**Fix:**

**Part A: Clip Fraction Stability (lines 7719-7739):**
- Added `torch.isfinite()` check before computing clip fraction
- Filter out non-finite ratio values
- Log warning `warn/ratio_all_nonfinite` if all values are non-finite
- Only accumulate finite ratio values for statistics

**Part B: Ratio Variance Correction (lines 9285-9296):**
- Changed from population variance to sample variance (Bessel's correction)
- Before: `var = E[X²] - E[X]²` (divides by n)
- After: `var = (sum(X²) - n*mean²) / (n-1)` (divides by n-1)
- Added safety check for single sample case (variance = 0)
- Added `math.isfinite()` check when computing std deviation
- Added new metric `train/ratio_var` for transparency

**Files Changed:**
- `distributional_ppo.py:7719-7739` - Clip fraction stability
- `distributional_ppo.py:9285-9296` - Ratio variance correction

---

## Impact Summary

### Correctness Improvements
1. **KL Divergence**: Now correctly measures policy divergence instead of just log ratio
2. **Ratio Variance**: Unbiased sample variance instead of biased population variance
3. **CVaR Semantics**: Clear documentation prevents misinterpretation of constraint metrics

### Reliability Improvements
1. **Numeric Stability**: Robust handling of inf/nan values in ratio calculations
2. **Data Quality**: Warning system for non-finite ratios
3. **Data Leakage Detection**: Explicit warning when potentially problematic fallback is used

### Code Quality Improvements
1. **Reduced Redundancy**: Removed 5 duplicate CVaR metrics
2. **Better Documentation**: Clear comments on semantic issues and formulas
3. **Transparency**: New metrics like `train/ratio_var` and warnings provide better observability

---

## Testing Recommendations

1. **Verify KL Divergence**:
   - Compare `train/approx_kl` with `train/approx_kl_raw` (should be similar now)
   - Check that values are non-negative and reasonable (typically < 0.1)

2. **Monitor Warnings**:
   - `warn/ratio_all_nonfinite` - should be 0 in healthy training
   - `warn/ev_fallback_data_leakage_risk` - should be rare or 0

3. **CVaR Metrics**:
   - Remember `train/cvar_violation` measures headroom (inverted!)
   - Positive values are GOOD (CVaR below limit)
   - Check `debug/cvar_headroom` for semantically correct version

4. **Ratio Statistics**:
   - `train/ratio_mean` should be close to 1.0
   - `train/ratio_std` should be small (< 0.2 typically)
   - `train/ratio_var` should match std² (verify Bessel's correction)

---

## Backward Compatibility

All fixes maintain backward compatibility:
- Legacy metric names preserved (e.g., `cvar_violation` kept despite inverted semantics)
- No breaking changes to public APIs
- Additional metrics and warnings are additive only

---

## Files Modified

1. `distributional_ppo.py` - All fixes applied to this file
   - Function `_compute_cvar_violation()` - Documentation update
   - Function `train()` - Multiple metric fixes
   - Lines modified: ~50 lines total

## Lines of Code Changed

- **Added**: ~40 lines (comments, checks, new metrics)
- **Modified**: ~15 lines (formulas, calculations)
- **Removed**: ~5 lines (duplicate metrics)
- **Net Change**: +30 lines

---

## References

- Original analysis: `TRAINING_METRICS_ANALYSIS.md`
- Quick reference: `METRICS_QUICK_REFERENCE.txt`
