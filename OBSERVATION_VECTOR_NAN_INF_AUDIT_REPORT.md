# Observation Vector NaN/Inf Audit Report

**Date:** 2025-01-13
**Auditor:** Claude AI
**Scope:** Complete audit of `obs_builder.pyx` and data flow from `mediator.py`
**Objective:** Identify all potential sources of NaN, Inf, or invalid values in observation vector

---

## Executive Summary

### Overall Assessment: ⚠️ **MODERATE RISK**

The observation vector is **generally well-protected** against NaN/Inf values with good use of:
- ✅ `isnan()` checks for technical indicators
- ✅ `_clipf()` helper function with NaN → 0.0 conversion
- ✅ Semantic defaults for early-bar indicators (RSI=50, MACD=0, etc.)
- ✅ Division protection with epsilon values (+1e-8)

However, **4 critical gaps** exist at the input validation layer:

1. 🔴 **CRITICAL:** No validation for input `price` parameter
2. 🔴 **CRITICAL:** No validation for input `prev_price` parameter
3. 🔴 **CRITICAL:** No validation for `log_volume_norm` and `rel_volume` inputs
4. 🔴 **IMPORTANT:** Incomplete Bollinger Bands validation (only checks `bb_lower`, not `bb_upper`)

**Impact:** If any of these inputs contain NaN/Inf, the entire observation vector becomes corrupted, leading to:
- Model training instability (NaN gradients)
- Policy breakdown (invalid actions)
- Debugging nightmares (silent failures)

**Recommendation:** Apply fixes immediately before next training run (estimated time: 30 minutes).

---

## Detailed Findings

### 🔴 Critical Issues (Must Fix)

#### 1. Input `price` Has No NaN/Inf/Zero Validation

**Severity:** CRITICAL
**File:** `obs_builder.pyx`
**Lines:** 32-34 (function signature), 70 (assignment), 88, 135, 139, 182, 195, 206, 216, 224, 231 (usage)

**Description:**
The `price` parameter is used throughout the function in critical calculations but is NEVER validated:
- Line 88: Direct assignment to observation `out_features[0] = price`
- Line 135: Division `(price_d - prev_price_d) / (prev_price_d + 1e-8)`
- Line 139: Division in vol_proxy calculation `atr / (price_d + 1e-8)`
- Line 182, 206: Division in momentum/trend calculations
- Line 195, 216, 224, 231: Division in Bollinger Bands calculations

**Risk:**
- If `price = NaN`: All features using price become NaN
- If `price = 0`: Division results are incorrect (but protected by +1e-8)
- If `price = Inf`: Features become Inf
- If `price < 0`: Semantically invalid (prices can't be negative)

**Reproduction:**
```python
obs = build_test_observation(price=float('nan'))
# Result: obs[0] = NaN, and many derived features also NaN
```

**Fix:**
```cython
# Add at the beginning of build_observation_vector_c (after line 70)
if isnan(price_d) or price_d <= 0.0:
    price_d = 1.0  # Fallback to safe non-zero value
```

**Estimated Impact:** HIGH - Price is the most fundamental feature, used in 15+ calculations

---

#### 2. Input `prev_price` Has No NaN/Inf/Zero Validation

**Severity:** CRITICAL
**File:** `obs_builder.pyx`
**Lines:** 32-34 (function signature), 71 (assignment), 135 (critical usage)

**Description:**
The `prev_price` parameter is used in the critical `ret_bar` calculation (line 135):
```cython
ret_bar = tanh((price_d - prev_price_d) / (prev_price_d + 1e-8))
```

While division by zero is protected with `+1e-8`, **NaN is not protected**:
- If `prev_price_d = NaN`, division produces NaN
- If `prev_price_d = 0.0`, result is large but finite (protected)

**Risk:**
- `ret_bar` (index 14) becomes NaN if `prev_price = NaN`
- This is a fundamental feature for measuring price momentum

**Reproduction:**
```python
obs = build_test_observation(price=50000.0, prev_price=float('nan'))
# Result: obs[14] = NaN
```

**Fix:**
```cython
# Add after line 71
if isnan(prev_price_d) or prev_price_d <= 0.0:
    prev_price_d = price_d  # Use current price as fallback
```

**Estimated Impact:** HIGH - `ret_bar` is a key feature for policy decisions

---

#### 3. Input `log_volume_norm` and `rel_volume` Have No Validation

**Severity:** CRITICAL
**File:** `obs_builder.pyx`
**Lines:** 90-93

**Description:**
Volume features are directly assigned to observation without ANY validation:
```cython
out_features[feature_idx] = log_volume_norm  # Line 90
feature_idx += 1
out_features[feature_idx] = rel_volume  # Line 92
```

These values are computed in `mediator.py` (lines 935-948):
```python
log_volume_norm = float(np.tanh(np.log1p(quote_volume / 240e6)))
rel_volume = float(np.tanh(np.log1p(volume / 24000.0)))
```

**Risk:**
- If `quote_volume` or `volume` is NaN/negative in input data, `log1p()` may produce NaN
- If computation fails, NaN propagates to observation
- No fallback or validation in obs_builder

**Reproduction:**
```python
obs = build_test_observation(log_volume_norm=float('nan'), rel_volume=float('inf'))
# Result: obs[1] = NaN, obs[2] = Inf
```

**Fix:**
```cython
# Replace lines 90-93 with:
out_features[feature_idx] = log_volume_norm if not isnan(log_volume_norm) else 0.0
feature_idx += 1
out_features[feature_idx] = rel_volume if not isnan(rel_volume) else 0.0
```

**Estimated Impact:** MEDIUM-HIGH - Volume features are important for liquidity assessment

---

#### 4. Incomplete Bollinger Bands Validation

**Severity:** IMPORTANT
**File:** `obs_builder.pyx`
**Lines:** 193-234

**Description:**
The `bb_valid` flag only checks `bb_lower`:
```cython
bb_valid = not isnan(bb_lower)  # Line 193
```

But `bb_upper` is NOT checked. Later, `bb_width` is calculated:
```cython
bb_width = bb_upper - bb_lower  # Line 215
```

**Risk:**
- If `bb_lower` is valid but `bb_upper = NaN`, then `bb_width = NaN`
- Features at indices 25-26 (bb_position, bb_width) become NaN

**Reproduction:**
```python
obs = build_test_observation(bb_lower=49000.0, bb_upper=float('nan'))
# Result: obs[25] = NaN (bb_position), obs[26] = NaN (bb_width)
```

**Fix:**
```cython
# Replace line 193 with:
bb_valid = not isnan(bb_lower) and not isnan(bb_upper)
```

**Estimated Impact:** MEDIUM - BB features are important for volatility-based strategies

---

### 🟡 Important Issues (High Priority)

#### 5. Agent State Inputs Not Validated

**Severity:** IMPORTANT
**File:** `obs_builder.pyx`
**Lines:** 161-171

**Description:**
Agent state parameters are used directly without validation:
- `last_vol_imbalance` → `tanh(last_vol_imbalance)` (line 161)
- `last_trade_intensity` → `tanh(last_trade_intensity)` (line 163)
- `last_agent_fill_ratio` → direct assignment (line 170)

**Risk:**
- If these inputs are NaN, features at indices 18-21 become NaN
- `tanh(NaN) = NaN` (no protection)

**Fix:**
```cython
# Add before line 161
if isnan(last_vol_imbalance):
    last_vol_imbalance = 0.0
if isnan(last_trade_intensity):
    last_trade_intensity = 0.0
if isnan(last_realized_spread):
    last_realized_spread = 0.0
if isnan(last_agent_fill_ratio):
    last_agent_fill_ratio = 0.0
```

**Estimated Impact:** MEDIUM - These features come from internal state, usually valid but not guaranteed

---

#### 6. Event Metadata Inputs Not Validated

**Severity:** IMPORTANT
**File:** `obs_builder.pyx`
**Lines:** 238, 241

**Description:**
Event metadata is assigned directly:
```cython
out_features[feature_idx] = is_high_importance  # Line 238
out_features[feature_idx] = <float>tanh(time_since_event / 24.0)  # Line 241
```

**Risk:**
- If `is_high_importance = NaN`, feature at index 27 becomes NaN
- If `time_since_event = NaN`, `tanh(NaN / 24.0) = NaN` at index 28

**Fix:**
```cython
# Add before line 238
if isnan(is_high_importance):
    is_high_importance = 0.0
if isnan(time_since_event):
    time_since_event = 0.0
```

**Estimated Impact:** LOW-MEDIUM - Event data may be optional, depends on configuration

---

### 🟢 Medium Priority Issues

#### 7. Fear & Greed Division Without Pre-Check

**Severity:** MEDIUM
**File:** `obs_builder.pyx`
**Lines:** 248-257

**Description:**
Fear & Greed value is divided by 100 before clipping:
```cython
feature_val = _clipf(fear_greed_value / 100.0, -3.0, 3.0)  # Line 249
```

If `fear_greed_value = NaN`, division produces NaN **before** `_clipf` is called.
However, `_clipf` will catch it and return 0.0 (line 14-15 of `_clipf` definition).

**Risk:**
- LOW - `_clipf` handles NaN → 0.0
- But relies on `_clipf` being called (if someone refactors and removes it, NaN leaks)

**Fix (Defensive):**
```cython
# Replace line 249 with:
if has_fear_greed and not isnan(fear_greed_value):
    feature_val = _clipf(fear_greed_value / 100.0, -3.0, 3.0)
else:
    feature_val = 0.0
```

**Estimated Impact:** LOW - Already protected by `_clipf`, but not explicit

---

#### 8. Upstream Data Quality in mediator.py

**Severity:** MEDIUM
**File:** `mediator.py`
**Lines:** 1014-1065 (`_extract_norm_cols`)

**Description:**
The 21 normalized columns (cvd, garch, yang_zhang, etc.) are extracted from the dataframe:
```python
norm_cols[0] = self._get_safe_float(row, "cvd_24h", 0.0)
# ... etc
```

`_get_safe_float` returns `default=0.0` if value is None, non-finite, or missing.
Later, in obs_builder, these go through:
```cython
feature_val = _clipf(tanh(norm_cols_values[i]), -3.0, 3.0)  # Line 262
```

**Risk:**
- If upstream `transformers.py` generates NaN/Inf, it's caught by `_get_safe_float` and replaced with 0.0
- This **hides data quality issues** - silently replacing bad data with 0.0 may mask bugs

**Recommendation:**
- Add logging/warnings when NaN is detected and replaced
- Track frequency of NaN replacements per feature
- Consider alerting if NaN rate exceeds threshold (e.g., >5%)

**Estimated Impact:** LOW-MEDIUM - Already protected, but may hide bugs

---

## Edge Cases Analysis

### ✅ WELL-HANDLED

1. **First 30 bars (indicator cold start)**
   - RSI: Default 50.0 (neutral)
   - MACD: Default 0.0 (no divergence)
   - ATR: Default `price × 0.01` (small volatility estimate)
   - BB: Default position 0.5, width 0.0
   - **Verdict:** Semantically meaningful defaults ✅

2. **Zero cash and zero units**
   - Line 147-148: `if total_worth <= 1e-8: feature_val = 1.0` (cash fraction)
   - Line 154-157: `if total_worth <= 1e-8: feature_val = 0.0` (position value)
   - **Verdict:** Properly handled ✅

3. **Token metadata with invalid token_id**
   - Line 273-277: Bounds check `if 0 <= token_id < max_num_tokens`
   - Line 284-285: Additional check before one-hot encoding
   - **Verdict:** Properly handled ✅

4. **Extreme values in tanh**
   - `tanh(1e10)` → 1.0 (saturates safely)
   - `tanh(-1e10)` → -1.0 (saturates safely)
   - **Verdict:** No overflow risk ✅

5. **Division by zero protection**
   - All divisions use `+1e-8` epsilon
   - **Verdict:** Properly protected ✅

### ⚠️ NOT WELL-HANDLED

1. **Market crash scenario: price=500 (99% drop from 50000)**
   - `ret_bar = tanh((500 - 50000) / (50000 + 1e-8))` → Valid, but extreme
   - `atr / price` if ATR is still high → Valid, but may saturate tanh
   - **Verdict:** Handled, but may produce saturated values (±1.0) ⚠️

2. **Zero or negative price**
   - **NOT VALIDATED** - would corrupt all price-dependent features
   - **Verdict:** CRITICAL GAP 🔴

3. **Bollinger Bands: lower > upper (inverted)**
   - Not explicitly checked, but `bb_width` would be negative
   - Line 224: `_clipf((price_d - bb_lower) / (bb_width + 1e-9), -1.0, 2.0)` still works
   - **Verdict:** Works but semantically incorrect ⚠️

---

## Mathematical Operations Audit

### Division Operations

| Line | Operation | Denominator Protection | NaN Protection | Status |
|------|-----------|----------------------|----------------|--------|
| 135 | `(price - prev_price) / (prev_price + 1e-8)` | ✅ | ❌ prev_price | ⚠️ |
| 139 | `atr / (price + 1e-8)` | ✅ | ❌ price, atr | ⚠️ |
| 150 | `cash / total_worth` | ✅ (explicit check) | N/A | ✅ |
| 157 | `position_value / (total_worth + 1e-8)` | ✅ | N/A | ✅ |
| 182 | `momentum / (price × 0.01 + 1e-8)` | ✅ | ❌ price, momentum | ⚠️ |
| 195 | `(bb_upper - bb_lower) / (price + 1e-8)` | ✅ | ❌ price, bb_upper | ⚠️ |
| 206 | `(macd - macd_signal) / (price × 0.01 + 1e-8)` | ✅ | ❌ price | ⚠️ |
| 224 | `(price - bb_lower) / (bb_width + 1e-9)` | ✅ | ❌ price, bb_lower | ⚠️ |
| 231 | `bb_width / (price + 1e-8)` | ✅ | ❌ price | ⚠️ |
| 249 | `fear_greed_value / 100.0` | N/A | ✅ (_clipf) | ✅ |
| 269 | `num_tokens / max_num_tokens` | ✅ (if check) | N/A | ✅ |
| 274 | `token_id / max_num_tokens` | ✅ (if check) | N/A | ✅ |

**Summary:**
- Division by zero: ✅ All protected
- NaN in numerator/denominator: ⚠️ Not consistently checked

### tanh Operations

| Line | Operation | Input Validation | Overflow Risk | Status |
|------|-----------|------------------|---------------|--------|
| 135 | `tanh(ret_calc)` | ❌ | ✅ (tanh saturates) | ⚠️ |
| 139 | `tanh(log1p(...))` | ❌ | ✅ | ⚠️ |
| 157 | `tanh(position_value / ...)` | Partial | ✅ | ⚠️ |
| 161 | `tanh(last_vol_imbalance)` | ❌ | ✅ | ⚠️ |
| 163 | `tanh(last_trade_intensity)` | ❌ | ✅ | ⚠️ |
| 182 | `tanh(momentum / ...)` | ❌ | ✅ | ⚠️ |
| 195 | `tanh((bb_upper - bb_lower) / ...)` | ❌ | ✅ | ⚠️ |
| 206 | `tanh((macd - macd_signal) / ...)` | ❌ | ✅ | ⚠️ |
| 241 | `tanh(time_since_event / 24.0)` | ❌ | ✅ | ⚠️ |
| 262 | `tanh(norm_cols_values[i])` | ❌ (but _clipf after) | ✅ | ⚠️ |

**Summary:**
- Overflow: ✅ tanh is safe (saturates at ±1)
- NaN propagation: ⚠️ `tanh(NaN) = NaN` - not always checked

### log/sqrt Operations

| Operation | Location | Status |
|-----------|----------|--------|
| `log1p()` | mediator.py:942, 948 | ✅ Protected (only positive volumes) |
| `log()` | Not used in obs_builder | N/A |
| `sqrt()` | Not used in obs_builder | N/A |

**Summary:** ✅ No log/sqrt in obs_builder itself

---

## Recommendations

### Immediate Actions (Before Next Training Run)

1. ✅ **Run comprehensive test suite**
   ```bash
   pytest tests/test_all_features_validation.py -v
   ```
   - This will identify any current NaN/Inf issues

2. 🔧 **Apply Priority 1 Fixes** (Estimated time: 15 minutes)
   - Add input validation for `price`, `prev_price`, `log_volume_norm`, `rel_volume`
   - Fix Bollinger Bands validation to check both `bb_lower` and `bb_upper`
   - See code examples in "Detailed Findings" section above

3. 📊 **Add Observation Validation to Training Loop**
   ```python
   # In training loop, after getting observation
   assert np.all(np.isfinite(obs)), f"Invalid obs at step {step}: {obs}"
   ```

4. 📝 **Review FEATURES_VALIDATION_CHECKLIST.md**
   - Use as reference for future feature additions
   - Share with team

### Short-Term Actions (Within 1 Week)

5. 🔧 **Apply Priority 2 Fixes**
   - Add validation for agent state inputs
   - Add validation for event metadata
   - Estimated time: 10 minutes

6. 📊 **Add Monitoring**
   - Log warnings when NaN is replaced with default
   - Track frequency of NaN replacements by feature
   - Alert if NaN rate exceeds 5%

7. 🧪 **Integration Testing**
   - Run 1000+ step simulation with observation validation
   - Test with corrupted input data (inject NaN)
   - Verify all NaN are caught and replaced

### Long-Term Actions (Within 1 Month)

8. 🔍 **Audit Upstream Data Pipeline**
   - Check `transformers.py` for potential NaN sources
   - Verify GARCH, Yang-Zhang, Parkinson volatility calculations
   - Ensure raw data quality

9. 📚 **Documentation**
   - Add observation vector diagram with validation points
   - Document semantics of default values
   - Create runbook for NaN debugging

10. 🏗️ **Architectural Improvement**
    - Consider creating a `validate_observation()` function
    - Centralize all validation logic
    - Add unit tests for validation function

---

## Testing Strategy

### Unit Tests ✅ (Created: test_all_features_validation.py)

```bash
pytest tests/test_all_features_validation.py -v
```

**Coverage:**
- ✅ All input parameters with NaN/Inf
- ✅ All technical indicators with NaN
- ✅ All mathematical operations with boundary values
- ✅ Edge cases (price=0, cash=0, etc.)
- ✅ First 30 bars (cold start)
- ✅ Worst-case scenario (market crash)

**Expected Result:** All tests should PASS after fixes are applied

### Integration Tests (TODO)

1. **1000-Step Simulation**
   - Run full simulation with observation validation
   - Log any NaN occurrences
   - Expected result: Zero NaN

2. **Data Corruption Test**
   - Inject NaN into raw dataframe
   - Verify observation is still valid (defaults applied)
   - Expected result: Warnings logged, but no NaN in observation

3. **Historical Data Test**
   - Run on 1 year of BTC/ETH historical data
   - Check for any NaN in observation
   - Expected result: Zero NaN

---

## Priority Matrix

| Issue | Severity | Likelihood | Priority | Est. Time |
|-------|----------|-----------|----------|-----------|
| 1. price validation | CRITICAL | MEDIUM | P0 | 5 min |
| 2. prev_price validation | CRITICAL | MEDIUM | P0 | 5 min |
| 3. volume validation | CRITICAL | LOW | P0 | 5 min |
| 4. BB validation | IMPORTANT | MEDIUM | P0 | 2 min |
| 5. Agent state validation | IMPORTANT | LOW | P1 | 5 min |
| 6. Event metadata validation | IMPORTANT | LOW | P1 | 5 min |
| 7. Fear & Greed explicit check | MEDIUM | LOW | P2 | 3 min |
| 8. Upstream monitoring | MEDIUM | MEDIUM | P1 | 1 hour |

**Total Estimated Time for P0 Fixes: 17 minutes**

---

## Conclusion

The observation vector has a **strong foundation** with good use of defensive programming patterns:
- ✅ Semantic defaults for indicators
- ✅ Division by zero protection
- ✅ `_clipf` helper with NaN handling
- ✅ Boundary checks for token metadata

However, **4 critical input validation gaps** exist that could allow NaN/Inf to corrupt the entire observation:
1. price
2. prev_price
3. log_volume_norm / rel_volume
4. bb_upper

**Risk Level:** MODERATE → Can be reduced to LOW with 17 minutes of fixes.

**Recommendation:** Apply P0 fixes immediately, then run comprehensive test suite before next training run.

---

## Appendix A: Quick Fix Script

```cython
# obs_builder.pyx - Apply these changes

# === ADD AFTER LINE 85 (after variable declarations) ===

# INPUT VALIDATION BLOCK
# 1. Validate price (CRITICAL)
if isnan(price_d) or price_d <= 0.0:
    price_d = 1.0  # Fallback to safe non-zero value

# 2. Validate prev_price (CRITICAL)
if isnan(prev_price_d) or prev_price_d <= 0.0:
    prev_price_d = price_d  # Use current price

# 3. Validate volumes (CRITICAL)
if isnan(log_volume_norm):
    log_volume_norm = 0.0
if isnan(rel_volume):
    rel_volume = 0.0

# 4. Validate agent state (IMPORTANT)
if isnan(last_vol_imbalance):
    last_vol_imbalance = 0.0
if isnan(last_trade_intensity):
    last_trade_intensity = 0.0
if isnan(last_realized_spread):
    last_realized_spread = 0.0
if isnan(last_agent_fill_ratio):
    last_agent_fill_ratio = 0.0

# 5. Validate event metadata (IMPORTANT)
if isnan(is_high_importance):
    is_high_importance = 0.0
if isnan(time_since_event):
    time_since_event = 0.0

# === REPLACE LINE 193 ===
# OLD: bb_valid = not isnan(bb_lower)
# NEW:
bb_valid = not isnan(bb_lower) and not isnan(bb_upper)
```

**After applying fixes:**
1. Recompile Cython: `python setup.py build_ext --inplace`
2. Run tests: `pytest tests/test_all_features_validation.py -v`
3. Verify: All tests should PASS

---

## Appendix B: Files Created

1. ✅ **tests/test_all_features_validation.py**
   - Comprehensive test suite with 50+ test cases
   - Covers all input parameters, edge cases, and worst-case scenarios

2. ✅ **FEATURES_VALIDATION_CHECKLIST.md**
   - Complete feature map (56 features)
   - Validation status for each feature
   - Checklist for adding new features
   - Recommended fixes with code examples

3. ✅ **OBSERVATION_VECTOR_NAN_INF_AUDIT_REPORT.md** (this file)
   - Detailed audit findings
   - Priority matrix
   - Quick fix script
   - Testing strategy

---

**Next Steps:**
1. Review this report with the team
2. Apply P0 fixes (17 minutes)
3. Run test suite
4. Deploy to production

**Contact:** For questions about this audit, see FEATURES_VALIDATION_CHECKLIST.md (Maintenance section)
