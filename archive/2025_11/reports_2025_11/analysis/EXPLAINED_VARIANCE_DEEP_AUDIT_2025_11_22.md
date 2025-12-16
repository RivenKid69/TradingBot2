# Explained Variance Deep Audit Report

**Date**: 2025-11-22
**Auditor**: Claude Code (Deep Sequential Analysis)
**Scope**: Complete explained variance pipeline from computation to logging
**Methodology**: Code review + Best practices comparison (SB3, CleanRL) + Edge case analysis

---

## Executive Summary

**AUDIT STATUS**: ✅ **MAJOR BUGS FIXED** | ⚠️ **2 NEW ISSUES FOUND** | 📝 **3 RECOMMENDATIONS**

### Previous Bugs (Already Fixed ✅)

1. ✅ **Bug #1.1** (CRITICAL) - Quantile mode EV used VF-clipped predictions → **FIXED** (line 10876)
2. ✅ **Bug #1.2** (CRITICAL) - Categorical mode EV used VF-clipped predictions → **FIXED** (line 11421)
3. ✅ **Bug #6** (MEDIUM) - Missing epsilon in ratio denominator → **FIXED** (lines 355, 376)

### New Issues Found in This Audit ⚠️

1. ⚠️ **NEW BUG #7** (MEDIUM) - Grouped EV computation missing epsilon protection (line 529)
2. ⚠️ **ISSUE #8** (LOW) - Variance floor too aggressive (1e-8 may be insufficient)

### Recommendations 📝

1. 📝 **Recommendation #1** - Add Twin Critics separate EV logging (Q1 vs Q2)
2. 📝 **Recommendation #2** - Make variance_floor adaptive based on normalization
3. 📝 **Recommendation #3** - Add fallback prevention flag for reserve-only EV

---

## Part 1: Verification of Previous Fixes ✅

### ✅ Bug #1.1: Quantile Mode EV Clipping - **VERIFIED FIXED**

**Location**: `distributional_ppo.py:10876`

**Current Code**:
```python
# BUG FIX #1.1: EV should use UNCLIPPED predictions to measure model's true capability
# Using clipped predictions artificially inflates EV metric
# quantiles_for_loss = unclipped model outputs (line 10468-10470)
quantiles_for_ev = quantiles_for_loss  # ✅ UNCLIPPED
```

**Verification**:
- ✅ `quantiles_for_loss` assigned at line 10527-10529 from `quantiles_fp32` (unclipped)
- ✅ VF clipping happens AFTER this assignment (lines 10800-10872)
- ✅ Separate variable `quantiles_norm_clipped_for_loss` used for VF loss computation
- ✅ Comment clearly documents the fix

**Status**: ✅ **CORRECTLY FIXED**

---

### ✅ Bug #1.2: Categorical Mode EV Clipping - **VERIFIED FIXED**

**Location**: `distributional_ppo.py:11421-11423`

**Current Code**:
```python
# BUG FIX #1.2: EV should use UNCLIPPED predictions to measure model's true capability
# Using clipped predictions artificially inflates EV metric
# mean_values_norm_selected = unclipped model outputs (line 11338-11344)
value_pred_norm_for_ev = (
    mean_values_norm_selected.reshape(-1, 1)  # ✅ UNCLIPPED
)
```

**Verification**:
- ✅ `mean_values_norm_selected` comes from line 11400-11408 (before VF clipping)
- ✅ VF clipping creates separate `mean_values_norm_clipped` variable (line 11354-11371)
- ✅ EV uses the UNCLIPPED `mean_values_norm_selected` NOT the clipped version
- ✅ Comment clearly documents the fix

**Status**: ✅ **CORRECTLY FIXED**

---

### ✅ Bug #6: Epsilon in Ratio Denominator - **VERIFIED FIXED**

**Location**: `distributional_ppo.py:352-358` (weighted case) and `370-379` (unweighted case)

**Current Code (Weighted)**:
```python
# BUG FIX #6: Add epsilon to prevent numerical instability when var_y is very small
# var_y > 0 is checked above, but very small positive values (e.g., 1e-100) can cause overflow
eps = 1e-12  # Standard epsilon for variance ratios
ratio = var_res / (var_y + eps)  # ✅ EPSILON ADDED
if not math.isfinite(ratio):
    return float("nan")
return float(1.0 - ratio)
```

**Current Code (Unweighted)**:
```python
# BUG FIX #6: Add epsilon to prevent numerical instability when var_y is very small
# (Same fix as weighted case above)
eps = 1e-12  # Standard epsilon for variance ratios
ratio = var_res / (var_y + eps)  # ✅ EPSILON ADDED
if not math.isfinite(ratio):
    return float("nan")
return float(1.0 - ratio)
```

**Verification**:
- ✅ Both weighted and unweighted paths have epsilon protection
- ✅ Standard epsilon value `1e-12` used (matches SciPy conventions)
- ✅ Finite check AFTER ratio computation catches any remaining issues
- ✅ Comments clearly document the fix

**Status**: ✅ **CORRECTLY FIXED**

---

## Part 2: New Bugs Found ⚠️

### ⚠️ NEW BUG #7: Grouped EV Computation Missing Epsilon Protection

**Location**: `distributional_ppo.py:529`
**Function**: `compute_grouped_explained_variance()`
**Severity**: **MEDIUM**

**Current Code**:
```python
var_true = _weighted_variance_np(true_group, weights_group)
if not math.isfinite(var_true) or var_true <= variance_floor:
    ev_grouped[key] = float("nan")
    continue
err_group = true_group - pred_group
var_err = _weighted_variance_np(err_group, weights_group)
if not math.isfinite(var_err):
    ev_grouped[key] = float("nan")
    continue
ev_value = float(1.0 - (var_err / var_true))  # ❌ NO EPSILON!
ev_grouped[key] = ev_value
```

**Problem**:
While there's a check `var_true <= variance_floor` at line 521, this does NOT prevent numerical instability when `var_true` is **very small but > variance_floor**.

**Example Edge Case**:
```python
variance_floor = 1e-6
var_true = 1.01e-6  # Just above floor, passes check
var_err = 1e-6

ev_value = 1.0 - (1e-6 / 1.01e-6) = 1.0 - 0.99009... = 0.00990...

# But if var_true is very small due to numerical errors:
var_true = 1e-100  # Underflow but still > 0
var_err = 1e-6
ev_value = 1.0 - (1e-6 / 1e-100) = 1.0 - 1e94 = -1e94  # OVERFLOW!
```

**Impact**:
- **Numerical instability** when grouped variance is very small
- Potential **NaN/Inf** in grouped EV metrics
- Silent failures (no logging when this happens)

**Recommended Fix**:
```python
# BEFORE (BUG):
ev_value = float(1.0 - (var_err / var_true))

# AFTER (FIX):
eps = 1e-12  # Match epsilon used in safe_explained_variance()
ev_value = float(1.0 - (var_err / (var_true + eps)))

# Add safety check
if not math.isfinite(ev_value):
    ev_grouped[key] = float("nan")
    continue
ev_grouped[key] = ev_value
```

**Test Case**:
```python
def test_grouped_ev_small_variance():
    """Test grouped EV with very small group variance"""
    y_true = np.array([1.0, 1.000001, 1.000002, 2.0, 2.000001])
    y_pred = np.array([1.0, 1.000001, 1.000001, 2.0, 2.000001])
    groups = ["A", "A", "A", "B", "B"]

    ev_dict, summary = compute_grouped_explained_variance(
        y_true, y_pred, groups, variance_floor=1e-8
    )

    # Group A has extremely small variance (should not crash)
    assert "A" in ev_dict
    assert math.isfinite(ev_dict["A"]) or math.isnan(ev_dict["A"])
    # Should NOT return -inf or inf
    assert not math.isinf(ev_dict["A"])
```

---

### ⚠️ ISSUE #8: Variance Floor Too Aggressive

**Location**: `distributional_ppo.py:5087`
**Function**: `_compute_explained_variance_metric()`
**Severity**: **LOW**

**Current Code**:
```python
def _compute_explained_variance_metric(
    self,
    ...,
    variance_floor: float = 1e-8,  # ← Very small threshold
    ...,
):
```

**Analysis**:
- Default `variance_floor = 1e-8` is **extremely small**
- This is appropriate for **normalized returns** (mean=0, std=1 → var ≈ 1.0)
- **Problematic for raw returns** where variance could be naturally small

**Example**:
```python
# Case 1: Normalized returns (appropriate)
returns_norm = np.array([0.5, 0.2, -0.3, 0.1])  # var ≈ 0.16 >> 1e-8 ✅

# Case 2: Raw PnL in stable period (problematic)
returns_raw = np.array([100.01, 100.02, 100.00, 100.01])  # var ≈ 7e-5
# var = 7e-5 > 1e-8 BUT this is still very small for raw space
# EV computation may be unreliable due to numerical precision
```

**Recommended Enhancement**:
```python
def _compute_explained_variance_metric(
    self,
    ...,
    variance_floor: Optional[float] = None,  # Allow None for auto
    ...,
):
    # Auto-select variance floor based on return normalization
    if variance_floor is None:
        if self.normalize_returns:
            # Normalized space: tight threshold (var ~ 1.0)
            variance_floor = 1e-4  # Require variance > 0.01% of typical
        else:
            # Raw space: looser threshold (var can vary widely)
            variance_floor = 1e-2  # Require variance > 1% of typical scale

    # ... rest of function
```

**Impact**:
- **Low risk** - current value works for normalized returns (most common case)
- **Potential issue** in raw return mode with low volatility periods
- Not a bug per se, but **suboptimal default** that could be improved

---

## Part 3: Best Practice Comparison

### Comparison with Stable-Baselines3

**SB3 Implementation**:
```python
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    """
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
```

**AI-Powered Quantitative Research Platform Advantages** ✅:
1. ✅ Better numerical stability (float64 promotion)
2. ✅ Finite value filtering (NaN/Inf handling)
3. ✅ Weighted variance support (for importance sampling)
4. ✅ Epsilon protection in ratio (SB3 lacks this!)
5. ✅ Comprehensive edge case handling

**AI-Powered Quantitative Research Platform Complexity** ⚠️:
1. ⚠️ More complex code (harder to maintain)
2. ⚠️ Multiple fallback paths (potential for subtle bugs)
3. ⚠️ Data leakage risk in fallback (acknowledged but not prevented)

### Comparison with CleanRL

**CleanRL Implementation**:
```python
y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
var_y = np.var(y_true)
explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
```

**AI-Powered Quantitative Research Platform Advantages** ✅:
1. ✅ More robust (CleanRL can return `inf` if var_y extremely small)
2. ✅ Weighted variance support
3. ✅ Grouped EV computation (per-group diagnostics)

**Overall Assessment**: AI-Powered Quantitative Research Platform implementation is **more robust** than both SB3 and CleanRL, with only minor issues remaining.

---

## Part 4: Existing Features Working Correctly ✅

### ✅ Data Leakage Warning (Logged)

**Location**: `distributional_ppo.py:5254-5255`

**Current Implementation**:
```python
if math.isfinite(fallback_ev):
    explained_var = float(fallback_ev)
    fallback_used = True
    # ... (omitted) ...
    if record_fallback:
        logger = getattr(self, "logger", None)
        if logger is not None:
            logger.record("train/value_explained_variance_fallback", 1.0)
            # Log warning about potential data leakage in fallback path
            logger.record("warn/ev_fallback_data_leakage_risk", 1.0)  # ✅ WARNING LOGGED
```

**Assessment**: ✅ **WORKING AS DESIGNED**
- Warning is logged when fallback is used
- Users are alerted to potential data leakage
- However, no **prevention mechanism** exists (see Recommendation #3 below)

---

### ✅ EV Unavailability Logging

**Location**: `distributional_ppo.py:5257-5263`

**Current Implementation**:
```python
if explained_var is None:
    # This can only happen if need_fallback=True and fallback failed
    if record_fallback:
        logger = getattr(self, "logger", None)
        if logger is not None and fallback_used is False:
            logger.record("train/value_explained_variance_fallback", 0.0)  # ✅ LOGGED
    return None, y_true_eval, y_pred_eval, metrics
```

**Assessment**: ✅ **WORKING AS DESIGNED**
- EV unavailability is explicitly logged
- Users can monitor `train/value_explained_variance_fallback` metric
- Silent failures are prevented

---

### ✅ Weighted Variance Formula (Correct)

**Location**: `distributional_ppo.py:329-358` (safe_explained_variance)
**Location**: `distributional_ppo.py:418-435` (_weighted_variance_np)

**Current Implementation**:
```python
sum_w = float(np.sum(weights64))
sum_w_sq = float(np.sum(weights64**2))

# Bessel's correction for weighted variance (reliability weights)
denom_raw = sum_w - (sum_w_sq / sum_w if sum_w_sq > 0.0 else 0.0)
denom = max(denom_raw, 1e-12)  # Epsilon safeguard
```

**Mathematical Verification**:
This is the **correct formula** for **reliability weights** (frequentist interpretation):
- Wikipedia reference: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
- Formula: `denom = sum(w) - sum(w²)/sum(w)` (effective sample size)

**Edge Case Testing**:
```python
# Test: Equal weights should reduce to standard Bessel's correction
w = np.array([1.0, 1.0, 1.0, 1.0])
sum_w = 4.0
sum_w_sq = 4.0
denom = 4.0 - (4.0 / 4.0) = 4.0 - 1.0 = 3.0  # ✅ Correct (N-1)
```

**Assessment**: ✅ **MATHEMATICALLY CORRECT**

---

## Part 5: Recommendations 📝

### 📝 Recommendation #1: Twin Critics Separate EV Logging

**Priority**: **MEDIUM** (improves debugging)

**Current State**: Twin Critics use `min(Q1, Q2)` for value estimates, but EV is computed on the combined predictions.

**Recommendation**: Log separate EV for each critic to detect learning imbalances.

**Implementation**:
```python
# Add to distributional_ppo.py in _train_step() after Twin Critics loss computation

if self.use_twin_critics and hasattr(self, '_last_q1_predictions') and hasattr(self, '_last_q2_predictions'):
    # Compute separate EV for Q1 and Q2
    y_true_np = rollout_buffer.returns.flatten()

    q1_pred_np = self._last_q1_predictions.cpu().numpy()
    q2_pred_np = self._last_q2_predictions.cpu().numpy()

    ev_q1 = safe_explained_variance(y_true_np, q1_pred_np)
    ev_q2 = safe_explained_variance(y_true_np, q2_pred_np)

    if math.isfinite(ev_q1):
        self.logger.record("train/explained_variance_q1", ev_q1)
    if math.isfinite(ev_q2):
        self.logger.record("train/explained_variance_q2", ev_q2)

    # Log min/max/diff for diagnostics
    if math.isfinite(ev_q1) and math.isfinite(ev_q2):
        self.logger.record("train/explained_variance_q1_q2_diff", abs(ev_q1 - ev_q2))
        self.logger.record("train/explained_variance_q1_q2_min", min(ev_q1, ev_q2))
```

**Benefits**:
- Detect if one critic is learning faster than the other
- Identify critic divergence early
- Better debugging for Twin Critics training issues

---

### 📝 Recommendation #2: Adaptive Variance Floor

**Priority**: **LOW** (edge case improvement)

**Current State**: Hard-coded `variance_floor = 1e-8` in `_compute_explained_variance_metric()`.

**Recommendation**: Make variance floor adaptive based on return normalization.

**Implementation**:
```python
def _compute_explained_variance_metric(
    self,
    y_true_tensor: Optional[torch.Tensor],
    y_pred_tensor: Optional[torch.Tensor],
    *,
    mask_tensor: Optional[torch.Tensor] = None,
    y_true_tensor_raw: Optional[torch.Tensor] = None,
    variance_floor: Optional[float] = None,  # ← Allow None for auto-select
    record_fallback: bool = True,
    group_keys: Optional[Sequence[str]] = None,
) -> tuple[...]:
    """..."""

    # Auto-select variance floor based on normalization
    if variance_floor is None:
        if self.normalize_returns:
            # Normalized space (mean=0, std=1): tight threshold
            variance_floor = 1e-4  # Require var > 0.01% of typical (std=1 → var=1)
        else:
            # Raw space: looser threshold (var can vary widely)
            # Heuristic: Use 1% of empirical return variance if available
            if hasattr(self, 'ret_rms') and self.ret_rms.var is not None:
                # Use 1% of running variance estimate
                variance_floor = max(1e-2, float(self.ret_rms.var) * 0.01)
            else:
                variance_floor = 1e-2  # Fallback for raw space

    # ... rest of function
```

**Benefits**:
- More appropriate thresholds for different return scales
- Avoids false positives (EV=NaN) in low-volatility periods
- Better numerical stability in raw return mode

---

### 📝 Recommendation #3: Fallback Prevention Flag

**Priority**: **LOW** (data hygiene improvement)

**Current State**: Fallback path uses raw returns which may leak training data (warning logged but not prevented).

**Recommendation**: Add flag to disable fallback for reserve-only EV calculations.

**Implementation**:
```python
def _compute_explained_variance_metric(
    self,
    ...,
    allow_fallback: bool = True,  # ← New parameter
    ...,
):
    """..."""

    # ... (earlier code) ...

    var_y = _weighted_variance_np(y_true_np, weights_np)
    need_fallback = (
        (not math.isfinite(primary_ev))
        or (not math.isfinite(var_y))
        or (var_y <= variance_floor)
    )

    # ... (earlier code) ...

    # DATA LEAKAGE PREVENTION: Optionally disable fallback
    if need_fallback and y_true_tensor_raw is not None and allow_fallback:  # ← Check flag
        # ... (fallback logic) ...
```

**Usage**:
```python
# For primary+reserve combined EV (allow fallback)
ev_combined = self._compute_explained_variance_metric(
    ..., allow_fallback=True
)

# For reserve-only EV (prevent data leakage)
ev_reserve = self._compute_explained_variance_metric(
    ..., allow_fallback=False  # Strict: return NaN if variance too small
)
```

**Benefits**:
- Strict train/test separation for reserve-only metrics
- Prevents optimistic bias in reserve EV estimates
- Maintains flexibility for combined metrics

---

## Part 6: Summary & Action Plan

### ✅ What's Working Well

1. ✅ **All 3 previous critical bugs fixed** (Bug #1.1, #1.2, #6)
2. ✅ **Numerical stability** excellent (float64, finite checks, epsilon protection)
3. ✅ **Weighted variance** mathematically correct (reliability weights formula)
4. ✅ **Data leakage warning** properly logged
5. ✅ **EV unavailability** explicitly logged
6. ✅ **More robust** than SB3 and CleanRL implementations

### ⚠️ Issues Found

1. ⚠️ **NEW BUG #7** (MEDIUM): Grouped EV missing epsilon protection → **FIX RECOMMENDED**
2. ⚠️ **ISSUE #8** (LOW): Variance floor too aggressive → **ENHANCEMENT RECOMMENDED**

### 📝 Recommendations (Optional)

1. 📝 Twin Critics separate EV logging (debugging improvement)
2. 📝 Adaptive variance floor (edge case handling)
3. 📝 Fallback prevention flag (data hygiene)

---

## Action Plan

### Week 1: Fix New Bugs ⚠️

**Priority: HIGH - Fix Bug #7**

1. **Apply fix to line 529** (`compute_grouped_explained_variance`):
   ```python
   # Add epsilon to line 529
   eps = 1e-12
   ev_value = float(1.0 - (var_err / (var_true + eps)))

   # Add safety check
   if not math.isfinite(ev_value):
       ev_grouped[key] = float("nan")
       continue
   ev_grouped[key] = ev_value
   ```

2. **Create regression test**:
   ```python
   def test_grouped_ev_numerical_stability():
       """Test grouped EV with very small variance (edge case)"""
       # Test case provided in Bug #7 section above
       ...
   ```

3. **Run existing tests**:
   ```bash
   pytest tests/test_distributional_ppo*.py -v
   pytest tests/test_explained_variance*.py -v -k "grouped"
   ```

**Priority: LOW - Address Issue #8**

1. **Implement adaptive variance floor** (optional, see Recommendation #2)
2. **Test with raw and normalized returns**

---

### Week 2-3: Implement Recommendations 📝

**Recommendation #1: Twin Critics EV Logging**
- Implement separate Q1/Q2 logging
- Add to monitoring dashboard
- Test with Twin Critics enabled

**Recommendation #2: Adaptive Variance Floor**
- Implement auto-selection logic
- Test edge cases (low volatility, high volatility)
- Document behavior

**Recommendation #3: Fallback Prevention**
- Add `allow_fallback` parameter
- Update reserve-only EV calls
- Document data leakage prevention

---

### Month 1: Documentation & Testing

1. **Enhance docstrings** with mathematical formulas and references
2. **Add comprehensive tests** for edge cases
3. **Update CLAUDE.md** with explained variance best practices
4. **Create monitoring alerts** for EV anomalies

---

## Testing Strategy

### Unit Tests (Regression Prevention)

```python
# tests/test_explained_variance_fixes.py

class TestExplainedVarianceFixes:
    """Test EV bug fixes and edge cases"""

    def test_ev_uses_unclipped_quantile_mode(self):
        """Verify Bug #1.1 fix: EV uses UNCLIPPED quantiles"""
        # Verify quantiles_for_ev == quantiles_for_loss (unclipped)
        # NOT quantiles_norm_clipped_for_loss (clipped)
        pass

    def test_ev_uses_unclipped_categorical_mode(self):
        """Verify Bug #1.2 fix: EV uses UNCLIPPED mean values"""
        # Verify value_pred_norm_for_ev uses mean_values_norm_selected
        # NOT mean_values_norm_clipped_selected
        pass

    def test_epsilon_protection_weighted(self):
        """Verify Bug #6 fix: Epsilon in weighted variance ratio"""
        y_true = np.array([1.0, 1.0000001, 1.0000002])
        y_pred = np.array([1.0, 1.0000001, 1.0000002])
        weights = np.array([1.0, 1.0, 1.0])

        ev = safe_explained_variance(y_true, y_pred, weights)
        # Should return NaN or finite value, NOT crash
        assert math.isnan(ev) or math.isfinite(ev)

    def test_epsilon_protection_unweighted(self):
        """Verify Bug #6 fix: Epsilon in unweighted variance ratio"""
        y_true = np.array([1.0, 1.0000001, 1.0000002])
        y_pred = np.array([1.0, 1.0000001, 1.0000002])

        ev = safe_explained_variance(y_true, y_pred, weights=None)
        # Should return NaN or finite value, NOT crash
        assert math.isnan(ev) or math.isfinite(ev)

    def test_grouped_ev_epsilon_protection(self):
        """Verify NEW Bug #7 fix: Grouped EV epsilon protection"""
        y_true = np.array([1.0, 1.000001, 1.000002, 2.0, 2.000001])
        y_pred = np.array([1.0, 1.000001, 1.000001, 2.0, 2.000001])
        groups = ["A", "A", "A", "B", "B"]

        ev_dict, summary = compute_grouped_explained_variance(
            y_true, y_pred, groups, variance_floor=1e-8
        )

        # Group A has very small variance - should not crash
        assert "A" in ev_dict
        assert math.isfinite(ev_dict["A"]) or math.isnan(ev_dict["A"])
        # Should NOT return -inf or +inf (overflow)
        assert not math.isinf(ev_dict["A"])
```

### Integration Tests

```python
class TestEVInTraining:
    """Test EV computation during actual training"""

    def test_ev_with_vf_clipping(self):
        """EV should be unbiased when VF clipping enabled"""
        # Train with VF clipping, compare EV to baseline
        # Difference should be < 5%
        pass

    def test_twin_critics_ev_separate_logging(self):
        """Verify separate Q1/Q2 EV logging (if implemented)"""
        # Train with Twin Critics
        # Check logger has "train/explained_variance_q1" and "q2" metrics
        pass
```

---

## Verification Checklist

Before closing this audit:

### Bug #7 (Grouped EV Epsilon) ⚠️
- [ ] Line 529 has epsilon in denominator: `var_err / (var_true + eps)`
- [ ] Finite check added after ratio computation
- [ ] Test with very small variance (no crash, no overflow)
- [ ] Existing grouped EV tests still pass

### Issue #8 (Variance Floor) 📝
- [ ] Adaptive variance floor implemented (optional)
- [ ] Tested with normalized and raw returns
- [ ] Documentation updated

### Recommendation #1 (Twin Critics Logging) 📝
- [ ] Separate Q1 and Q2 EV logged (optional)
- [ ] Tested with Twin Critics enabled
- [ ] Added to monitoring dashboard

### Recommendation #2 (Adaptive Floor) 📝
- [ ] Implementation complete (optional)
- [ ] Edge cases tested
- [ ] Documentation updated

### Recommendation #3 (Fallback Prevention) 📝
- [ ] `allow_fallback` parameter added (optional)
- [ ] Reserve-only EV uses strict mode
- [ ] Data leakage prevented

---

## Expected Outcomes

### After Bug #7 Fix
- ✅ Grouped EV computation numerically stable
- ✅ No overflow/underflow in edge cases
- ✅ More reliable per-group diagnostics

### After Recommendations (Optional)
- ✅ Better Twin Critics debugging (separate Q1/Q2 metrics)
- ✅ More appropriate variance thresholds
- ✅ Stricter data hygiene in reserve-only EV

---

## References

1. **Stable-Baselines3 Explained Variance**
   https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/utils.py

2. **CleanRL PPO Implementation**
   https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

3. **Weighted Variance (Reliability Weights)**
   https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights

4. **Bessel's Correction**
   https://en.wikipedia.org/wiki/Bessel%27s_correction

5. **PPO Paper (Schulman et al. 2017)**
   https://arxiv.org/abs/1707.06347

6. **Explained Variance Score (Scikit-learn)**
   https://scikit-learn.org/stable/modules/model_evaluation.html#explained-variance-score

---

## Audit Trail

**Audit Performed By**: Claude Code (Sonnet 4.5)
**Audit Date**: 2025-11-22
**Audit Type**: Deep sequential analysis (full pipeline review)
**Previous Audits**: EXPLAINED_VARIANCE_BUGS_REPORT_2025_11_22.md, EXPLAINED_VARIANCE_AUDIT_REPORT_2025_11_22.md
**Bugs Fixed Since Last Audit**: 3 (Bug #1.1, #1.2, #6)
**New Bugs Found**: 1 (Bug #7)
**New Issues Found**: 1 (Issue #8)
**Recommendations**: 3

**Code Inspected**:
- `distributional_ppo.py` (12000+ lines)
- `safe_explained_variance()` function (lines 286-379)
- `_weighted_variance_np()` function (lines 382-445)
- `compute_grouped_explained_variance()` function (lines 447-562)
- `_compute_explained_variance_metric()` function (lines 5080-5340)
- EV computation in train loop (lines 11980+)

**Tests Reviewed**:
- `test_ev_bugs_direct.py` (direct code inspection)
- `test_explained_variance_audit.py` (functional tests)
- Existing `test_distributional_ppo*.py` tests

**Status**: ✅ **AUDIT COMPLETE** - 1 new bug found, 3 recommendations provided

---

**Report Status**: FINAL
**Last Updated**: 2025-11-22
**Next Review**: After Bug #7 fix applied

---

**END OF REPORT**
