# Features Validation Checklist

## Overview

This document provides a comprehensive checklist for validating all features in the observation vector to ensure NO NaN, Inf, or invalid values can enter the model under any circumstances.

**Critical Invariant:** Every value in the observation vector MUST be a finite float at all times.

---

## Complete Feature Map (56 features)

### Block 1: Bar Level (indices 0-13)

| Index | Feature Name | Source | Min Period | Default if NaN | Valid Range | Validation Status |
|-------|-------------|--------|------------|----------------|-------------|------------------|
| 0 | `price` | Input param | 0 | **NO DEFAULT** | (0, ∞) | ⚠️ **MISSING** |
| 1 | `log_volume_norm` | Input param | 0 | **NO DEFAULT** | (-∞, ∞) | ⚠️ **MISSING** |
| 2 | `rel_volume` | Input param | 0 | **NO DEFAULT** | (-∞, ∞) | ⚠️ **MISSING** |
| 3 | `ma5` | Indicator (SMA 5) | 5 bars | 0.0 | (0, ∞) | ✅ GOOD |
| 4 | `ma5_valid` | Indicator flag | 5 bars | 0.0 (not ready) | {0.0, 1.0} | ✅ GOOD |
| 5 | `ma20` | Indicator (SMA 20) | 20 bars | 0.0 | (0, ∞) | ✅ GOOD |
| 6 | `ma20_valid` | Indicator flag | 20 bars | 0.0 (not ready) | {0.0, 1.0} | ✅ GOOD |
| 7 | `rsi14` | RSI 14 | 14 bars | 50.0 (neutral) | [0, 100] | ✅ GOOD |
| 8 | `macd` | MACD | ~26 bars | 0.0 (no signal) | (-∞, ∞) | ✅ GOOD |
| 9 | `macd_signal` | MACD Signal | ~26 bars | 0.0 (no signal) | (-∞, ∞) | ✅ GOOD |
| 10 | `momentum` | Momentum | 10 bars | 0.0 (no movement) | (-∞, ∞) | ✅ GOOD |
| 11 | `atr` | ATR | 14 bars | price × 0.01 | (0, ∞) | ✅ GOOD |
| 12 | `cci` | CCI | 20 bars | 0.0 (average) | (-∞, ∞) | ✅ GOOD |
| 13 | `obv` | OBV | 0 | 0.0 | (-∞, ∞) | ✅ GOOD |

**Issues:**
- **CRITICAL:** `price` (index 0) has NO NaN validation - if input is NaN, all calculations fail
- **CRITICAL:** `log_volume_norm` and `rel_volume` have NO NaN validation at obs_builder level

---

### Block 2: Derived Signals (indices 14-15)

| Index | Feature Name | Formula | Dependencies | Default if NaN | Valid Range | Validation Status |
|-------|-------------|---------|--------------|----------------|-------------|------------------|
| 14 | `ret_bar` | `tanh((price - prev_price) / (prev_price + 1e-8))` | price, prev_price | N/A | [-1, 1] | ⚠️ **PARTIAL** |
| 15 | `vol_proxy` | `tanh(log1p(atr / (price + 1e-8)))` | atr, price | N/A | [-1, 1] | ⚠️ **PARTIAL** |

**Issues:**
- **IMPORTANT:** If `prev_price` is NaN, division will produce NaN (only protected from 0)
- **IMPORTANT:** If `price` is NaN, all calculations fail
- `atr` is validated earlier (line 123), but still used in division without re-check

---

### Block 3: Agent State (indices 16-21)

| Index | Feature Name | Formula | Dependencies | Edge Cases | Valid Range | Validation Status |
|-------|-------------|---------|--------------|------------|-------------|------------------|
| 16 | `cash_fraction` | `cash / total_worth` (clipped) | cash, units, price | total_worth ≤ 1e-8 → 1.0 | [0, 1] | ✅ GOOD |
| 17 | `position_value_tanh` | `tanh(position_value / total_worth)` | units, price, cash | total_worth ≤ 1e-8 → 0.0 | [-1, 1] | ✅ GOOD |
| 18 | `vol_imbalance` | `tanh(last_vol_imbalance)` | Input param | None | [-1, 1] | ⚠️ **MISSING** |
| 19 | `trade_intensity` | `tanh(last_trade_intensity)` | Input param | None | [-1, 1] | ⚠️ **MISSING** |
| 20 | `realized_spread` | `_clipf(last_realized_spread, -0.1, 0.1)` | Input param | Uses _clipf (NaN → 0.0) | [-0.1, 0.1] | ✅ GOOD |
| 21 | `agent_fill_ratio` | Direct input | Input param | **NO DEFAULT** | [0, 1] (expected) | ⚠️ **MISSING** |

**Issues:**
- **MEDIUM:** `last_vol_imbalance` and `last_trade_intensity` are not validated before tanh (NaN input → NaN output)
- **MEDIUM:** `last_agent_fill_ratio` has no validation

---

### Block 4: Technical Indicators for 4h (indices 22-24)

| Index | Feature Name | Formula | Dependencies | Default if NaN | Valid Range | Validation Status |
|-------|-------------|---------|--------------|----------------|-------------|------------------|
| 22 | `price_momentum` | `tanh(momentum / (price × 0.01 + 1e-8))` | momentum, price | 0.0 if momentum is NaN | [-1, 1] | ⚠️ **PARTIAL** |
| 23 | `bb_squeeze` | `tanh((bb_upper - bb_lower) / (price + 1e-8))` | bb_upper, bb_lower, price | 0.0 if BB not ready | [-1, 1] | ⚠️ **PARTIAL** |
| 24 | `trend_strength` | `tanh((macd - macd_signal) / (price × 0.01 + 1e-8))` | macd, macd_signal, price | 0.0 if MACD not ready | [-1, 1] | ⚠️ **PARTIAL** |

**Issues:**
- **IMPORTANT:** If `price` is NaN or 0, divisions will fail or produce incorrect results
- **IMPORTANT:** `bb_squeeze` checks only `isnan(bb_lower)`, but not `bb_upper` (line 193)
  - If `bb_lower` is valid but `bb_upper` is NaN, `bb_width` will be NaN

---

### Block 5: Bollinger Bands Context (indices 25-26)

| Index | Feature Name | Formula | Dependencies | Default if NaN/Invalid | Valid Range | Validation Status |
|-------|-------------|---------|--------------|----------------------|-------------|------------------|
| 25 | `bb_position` | `(price - bb_lower) / (bb_width + 1e-9)` | bb_lower, bb_upper, price | 0.5 if !bb_valid or width ≤ min_width | [-1, 2] (clipped) | ⚠️ **PARTIAL** |
| 26 | `bb_width` | `bb_width / (price + 1e-8)` | bb_upper, bb_lower, price | 0.0 if !bb_valid | [0, 10] (clipped) | ⚠️ **PARTIAL** |

**Issues:**
- **IMPORTANT:** `bb_valid` is only checked for `bb_lower`, not `bb_upper` (line 193)
- If `bb_upper` is NaN but `bb_lower` is valid, `bb_width` = NaN → both features become NaN
- **CRITICAL:** If `price` is NaN, division will produce NaN

---

### Block 6: Event Metadata (indices 27-29)

| Index | Feature Name | Source | Default if NaN | Valid Range | Validation Status |
|-------|-------------|--------|----------------|-------------|------------------|
| 27 | `is_high_importance` | Input param | **NO DEFAULT** | [0, 1] (expected) | ⚠️ **MISSING** |
| 28 | `time_since_event` | `tanh(time_since_event / 24.0)` | Input param | **NO DEFAULT** | [-1, 1] | ⚠️ **MISSING** |
| 29 | `risk_off_flag` | Boolean input | N/A | {0.0, 1.0} | ✅ GOOD |

**Issues:**
- **MEDIUM:** `is_high_importance` and `time_since_event` have NO NaN validation
- If input is NaN, it propagates to observation

---

### Block 7: Fear & Greed (indices 30-31)

| Index | Feature Name | Formula | Dependencies | Default if NaN | Valid Range | Validation Status |
|-------|-------------|---------|--------------|----------------|-------------|------------------|
| 30 | `fear_greed_value` | `_clipf(fear_greed_value / 100.0, -3.0, 3.0)` | Input param | 0.0 if !has_fear_greed | [-3, 3] | ⚠️ **PARTIAL** |
| 31 | `has_fear_greed` | Boolean input | N/A | {0.0, 1.0} | ✅ GOOD |

**Issues:**
- **MEDIUM:** If `fear_greed_value` is NaN and `has_fear_greed=True`, division by 100 produces NaN
- `_clipf` will catch it (NaN → 0.0), but relies on _clipf being called

---

### Block 8: External Normalized Columns (indices 32-52)

**21 features from norm_cols_values:**

| Index | Feature Name (4h timeframe) | Source | Period | Default if NaN | Valid Range | Validation Status |
|-------|----------------------------|--------|--------|----------------|-------------|------------------|
| 32 | `cvd_24h` | norm_cols[0] | 6 bars | 0.0 (_clipf) | [-3, 3] | ✅ GOOD |
| 33 | `cvd_7d` | norm_cols[1] | 42 bars | 0.0 (_clipf) | [-3, 3] | ✅ GOOD |
| 34 | `yang_zhang_48h` | norm_cols[2] | 12 bars | 0.0 (_clipf) | [-3, 3] | ✅ GOOD |
| 35 | `yang_zhang_7d` | norm_cols[3] | 42 bars | 0.0 (_clipf) | [-3, 3] | ✅ GOOD |
| 36 | `garch_200h` | norm_cols[4] | 50 bars | 0.0 (_clipf) | [-3, 3] | ✅ GOOD |
| 37 | `garch_14d` | norm_cols[5] | 84 bars | 0.0 (_clipf) | [-3, 3] | ✅ GOOD |
| 38 | `ret_12h` | norm_cols[6] | 3 bars | 0.0 (_clipf) | [-3, 3] | ✅ GOOD |
| 39 | `ret_24h` | norm_cols[7] | 6 bars | 0.0 (_clipf) | [-3, 3] | ✅ GOOD |
| 40 | `ret_4h` | norm_cols[8] | 1 bar | 0.0 (_clipf) | [-3, 3] | ✅ GOOD |
| 41 | `sma_12000` | norm_cols[9] | 50 bars | 0.0 (_clipf) | [-3, 3] | ✅ GOOD |
| 42 | `yang_zhang_30d` | norm_cols[10] | 180 bars | 0.0 (_clipf) | [-3, 3] | ✅ GOOD |
| 43 | `parkinson_48h` | norm_cols[11] | 12 bars | 0.0 (_clipf) | [-3, 3] | ✅ GOOD |
| 44 | `parkinson_7d` | norm_cols[12] | 42 bars | 0.0 (_clipf) | [-3, 3] | ✅ GOOD |
| 45 | `garch_30d` | norm_cols[13] | 180 bars | 0.0 (_clipf) | [-3, 3] | ✅ GOOD |
| 46 | `taker_buy_ratio` | norm_cols[14] | 0 | 0.0 (_clipf) | [-3, 3] | ✅ GOOD |
| 47 | `taker_buy_ratio_sma_24h` | norm_cols[15] | 6 bars | 0.0 (_clipf) | [-3, 3] | ✅ GOOD |
| 48 | `taker_buy_ratio_sma_8h` | norm_cols[16] | 2 bars | 0.0 (_clipf) | [-3, 3] | ✅ GOOD |
| 49 | `taker_buy_ratio_sma_16h` | norm_cols[17] | 4 bars | 0.0 (_clipf) | [-3, 3] | ✅ GOOD |
| 50 | `taker_buy_ratio_momentum_4h` | norm_cols[18] | 1 bar | 0.0 (_clipf) | [-3, 3] | ✅ GOOD |
| 51 | `taker_buy_ratio_momentum_8h` | norm_cols[19] | 2 bars | 0.0 (_clipf) | [-3, 3] | ✅ GOOD |
| 52 | `taker_buy_ratio_momentum_12h` | norm_cols[20] | 3 bars | 0.0 (_clipf) | [-3, 3] | ✅ GOOD |

**Validation:** All norm_cols pass through `tanh()` then `_clipf(val, -3.0, 3.0)` (lines 260-264)
- `_clipf` converts NaN to 0.0, so this block is well-protected
- **DEPENDENCY:** Relies on mediator.py providing valid values

**Issues:**
- ⚠️ **UPSTREAM:** If mediator._extract_norm_cols() provides NaN/Inf, it will be sanitized but may hide data quality issues

---

### Block 9: Token Metadata (indices 53-55, optional)

**Only present if `max_num_tokens > 0`**

| Index | Feature Name | Formula | Default if Invalid | Valid Range | Validation Status |
|-------|-------------|---------|-------------------|-------------|------------------|
| 53 | `num_tokens_normalized` | `_clipf(num_tokens / max_num_tokens, 0, 1)` | 0.0 if out of range | [0, 1] | ✅ GOOD |
| 54 | `token_id_normalized` | `_clipf(token_id / max_num_tokens, 0, 1)` | 0.0 if invalid token_id | [0, 1] | ✅ GOOD |
| 55+ | `one_hot_encoding` | 1.0 at token_id, else 0.0 | All zeros if invalid | {0.0, 1.0} | ✅ GOOD |

**Validation:** Token features are well-protected with boundary checks and _clipf

---

## Critical Issues Summary

### 🔴 CRITICAL (Must Fix Immediately)

1. **No validation for input `price`** (obs_builder.pyx:88, 135, 139, 182, 195, 206, 216, 224, 231)
   - **Location:** obs_builder.pyx, lines 32-34, 70
   - **Risk:** If price is NaN/Inf/≤0, entire observation becomes corrupted
   - **Fix:** Add validation at function entry:
     ```cython
     if isnan(price_d) or price_d <= 0.0:
         price_d = 1.0  # Fallback safe value
     ```

2. **No validation for input `prev_price`** (obs_builder.pyx:135)
   - **Location:** obs_builder.pyx, line 135
   - **Risk:** Used in division: `(price_d - prev_price_d) / (prev_price_d + 1e-8)`
   - **Fix:** Add validation:
     ```cython
     if isnan(prev_price_d) or prev_price_d <= 0.0:
         prev_price_d = price_d  # Use current price as fallback
     ```

3. **No validation for input `log_volume_norm` and `rel_volume`** (obs_builder.pyx:90, 92)
   - **Location:** obs_builder.pyx, lines 90-93
   - **Risk:** Direct assignment to observation without NaN check
   - **Fix:** Add validation:
     ```cython
     out_features[feature_idx] = log_volume_norm if not isnan(log_volume_norm) else 0.0
     ```

4. **Incomplete BB validation** (obs_builder.pyx:193, 215)
   - **Location:** obs_builder.pyx, line 193
   - **Current:** Only checks `isnan(bb_lower)`
   - **Risk:** If bb_lower is valid but bb_upper is NaN, bb_width becomes NaN
   - **Fix:** Change to:
     ```cython
     bb_valid = not isnan(bb_lower) and not isnan(bb_upper)
     ```

---

### 🟡 IMPORTANT (High Priority)

5. **No validation for agent state inputs** (obs_builder.pyx:161-171)
   - `last_vol_imbalance`, `last_trade_intensity`, `last_agent_fill_ratio`
   - **Risk:** NaN inputs produce NaN features
   - **Fix:** Add validation before use

6. **No validation for event metadata inputs** (obs_builder.pyx:238, 241)
   - `is_high_importance`, `time_since_event`
   - **Risk:** NaN inputs propagate to observation
   - **Fix:** Add validation or ensure mediator provides valid values

---

### 🟢 MEDIUM (Should Fix)

7. **Fear & Greed value not validated before division** (obs_builder.pyx:249)
   - **Location:** obs_builder.pyx, line 249
   - **Current:** `_clipf(fear_greed_value / 100.0, -3.0, 3.0)`
   - **Risk:** If fear_greed_value is NaN, division produces NaN before _clipf
   - **Fix:** Check NaN before division or rely on _clipf (which already handles it)

8. **Upstream data quality in mediator.py**
   - **Location:** mediator.py, lines 930-1066
   - **Risk:** If transformers.py generates NaN/Inf for norm_cols, it's hidden by _clipf
   - **Fix:** Add logging/warnings when NaN is detected and replaced

---

## Validation Checklist for New Features

When adding a new feature to the observation vector, ensure:

### 1. Input Validation
- [ ] Check `isnan()` for all float inputs
- [ ] Check `isinf()` for all float inputs
- [ ] Validate range (e.g., price > 0, probabilities in [0,1])
- [ ] Provide semantically meaningful defaults for NaN (not just 0.0)

### 2. Mathematical Operations
- [ ] **Division:** Always add epsilon (e.g., `x / (y + 1e-8)`)
- [ ] **Division:** Check denominator for NaN before operation
- [ ] **log/sqrt:** Ensure input is positive
- [ ] **tanh/exp:** Input is bounded to prevent overflow (tanh is safe, exp needs care)
- [ ] Use `_clipf()` for final clipping (handles NaN → 0.0)

### 3. Conditional Logic
- [ ] Ensure all branches return valid values
- [ ] Handle edge cases (e.g., width=0, empty data)
- [ ] Test with extreme values (0, ∞, -∞, NaN)

### 4. Default Values
- [ ] Choose defaults that are semantically meaningful:
  - RSI: 50.0 (neutral)
  - MACD: 0.0 (no divergence)
  - ATR: price × 0.01 (small volatility estimate)
  - BB position: 0.5 (middle of bands)
- [ ] Avoid blind 0.0 defaults that could be misleading

### 5. Testing
- [ ] Unit test with NaN inputs
- [ ] Unit test with Inf inputs
- [ ] Unit test with zero inputs
- [ ] Unit test with extreme values (1e10, 1e-10)
- [ ] Integration test on first 30 bars (indicators not ready)
- [ ] Integration test in worst-case scenario (market crash)

---

## Recommended Fixes

### Priority 1: Add Input Validation to obs_builder.pyx

```cython
cdef void build_observation_vector_c(
    float price,
    float prev_price,
    float log_volume_norm,
    float rel_volume,
    # ... other params
) noexcept nogil:
    """Populate ``out_features`` with the observation vector without acquiring the GIL."""

    cdef double price_d = price
    cdef double prev_price_d = prev_price

    # === INPUT VALIDATION (ADD THIS BLOCK) ===
    # Validate price (CRITICAL)
    if isnan(price_d) or price_d <= 0.0:
        price_d = 1.0  # Fallback to safe non-zero value

    # Validate prev_price (CRITICAL)
    if isnan(prev_price_d) or prev_price_d <= 0.0:
        prev_price_d = price_d  # Use current price

    # Validate volumes (CRITICAL)
    if isnan(log_volume_norm):
        log_volume_norm = 0.0
    if isnan(rel_volume):
        rel_volume = 0.0

    # === EXISTING CODE ===
    cdef int feature_idx = 0
    # ... rest of function
```

### Priority 2: Fix Bollinger Bands Validation

```cython
# OLD (line 193)
bb_valid = not isnan(bb_lower)

# NEW
bb_valid = not isnan(bb_lower) and not isnan(bb_upper)
```

### Priority 3: Add Validation for Agent State

```cython
# Before line 161
if isnan(last_vol_imbalance):
    last_vol_imbalance = 0.0
if isnan(last_trade_intensity):
    last_trade_intensity = 0.0
if isnan(last_realized_spread):
    last_realized_spread = 0.0
if isnan(last_agent_fill_ratio):
    last_agent_fill_ratio = 0.0
```

### Priority 4: Add Validation for Event Metadata

```cython
# Before line 238
if isnan(is_high_importance):
    is_high_importance = 0.0
if isnan(time_since_event):
    time_since_event = 0.0
```

---

## Testing Strategy

### Unit Tests (test_all_features_validation.py)
- ✅ Test each input parameter with NaN/Inf
- ✅ Test mathematical operations with boundary values
- ✅ Test all indicators with NaN inputs
- ✅ Test edge cases (price=0, cash=0, etc.)

### Integration Tests
- ✅ Test observation on first 30 bars (cold start)
- ✅ Test worst-case scenario (market crash)
- ⚠️ TODO: Test with real historical data
- ⚠️ TODO: Test with simulator over 1000+ steps

### Monitoring
- Add assertions in training loop to catch NaN in observations
- Log warnings when NaN is replaced with default
- Track frequency of NaN occurrences by feature

---

## Maintenance

### When to Re-audit
- [ ] When adding new features to observation vector
- [ ] When modifying technical indicators
- [ ] When changing data preprocessing pipeline
- [ ] After major refactoring of obs_builder or mediator
- [ ] When encountering unexplained NaN in training

### Contacts
- Feature validation owner: [Your Name]
- obs_builder maintainer: [Maintainer Name]
- Last audit: 2025-01-13

---

## Appendix: Helper Functions

### _clipf (obs_builder.pyx:7-20)
```cython
cdef inline float _clipf(double value, double lower, double upper) nogil:
    """
    Clip value to [lower, upper] range with NaN handling.
    CRITICAL: NaN comparisons are always False in C/Cython, so we must check explicitly.
    If value is NaN, we return 0.0 as a safe default to prevent NaN propagation.
    """
    if isnan(value):
        return 0.0
    if value < lower:
        value = lower
    elif value > upper:
        value = upper
    return <float>value
```

**Usage:** Use `_clipf()` for all final feature values that need clipping. It provides NaN protection.

---

## Conclusion

The observation vector is **mostly well-protected** against NaN/Inf, but has **4 critical gaps** in input validation:

1. **price** (no validation)
2. **prev_price** (no validation)
3. **log_volume_norm / rel_volume** (no validation)
4. **bb_upper** (not checked in bb_valid)

**Recommended Action:** Apply Priority 1 and Priority 2 fixes immediately before next training run.
