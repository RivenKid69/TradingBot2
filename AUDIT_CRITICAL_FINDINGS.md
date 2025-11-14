# Data Pipeline Audit: Critical Findings

## Executive Summary
**Status: CRITICAL BUGS FOUND** 🔴

While comprehensive validation tests were created, the audit **UNCOVERED CRITICAL BUGS** in feature generation that **ARE NOT CAUGHT BY TESTS**.

## What Was Checked ✅

### P0: Data Loading (fetch_all_data_patch.py)
- ✅ Required columns validation
- ✅ Timestamp normalization (ms→s)
- ✅ 4h alignment (14400s)
- ✅ Deduplication
- ✅ DataValidator is called in train_model_multi_patch.py:4742

### P2: Asof-Merge (asof_join.py)
- ✅ Basic backward merge
- ✅ Tolerance enforcement
- ✅ Keys comparison bug FIXED (list vs tuple)
- ✅ Tolerance type bug FIXED (Timedelta→int for int64)

### P3: Leak Guard (leakguard.py)
- ✅ Decision time attachment
- ✅ Forward-fill gap validation

### P4: Label Generation (labels.py)
- ✅ Future price lookup
- ✅ Log returns calculation

## Critical Bugs Found 🔴

### BUG #1: Returns Feature Always Zero
**Location:** `transformers.py:606-618`
**Impact:** ALL ret_4h values are 0.0 (corrupted training data!)

```python
# When lookback=1 bar (ret_4h on 4h TF):
window = seq[-1:]  # Current bar only
first = window[0]  # Current price
ret = log(price / first)  # log(current/current) = 0
```

**Consequence:**
- ret_4h (240min / 240min = 1 bar) → **ALWAYS 0**
- Model trains on zero values instead of real returns
- **All 4h return predictions are broken**

### BUG #2: RSI Returns NaN for Zero Loss
**Location:** `transformers.py:620-628`
**Impact:** RSI=NaN during strong trends

```python
if float(st["avg_loss"]) > 0.0:  # ❌ Fails when loss=0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
else:
    rsi = NaN  # Should be 100!
```

**Consequence:**
- RSI=NaN during uptrends (no losses)
- RSI=NaN during downtrends (no gains, reverse issue)
- Model sees NaN instead of extreme RSI values (0 or 100)

## What Was NOT Checked ❌

### 1. Feature Generation Correctness
- ❌ **Returns calculation logic** (found bug!)
- ❌ **RSI edge cases** (found bug!)
- ❌ GARCH convergence (just checked "returns None")
- ❌ Yang-Zhang with missing OHLC bars
- ❌ Taker Buy Ratio with zero volume

### 2. Real Data Testing
- ❌ No tests on actual data/processed/*.feather files
- ❌ No verification of existing trained models
- ❌ No check if bugs affected past training runs

### 3. End-to-End Verification
- ❌ train_model_multi_patch.py is 5000+ lines (not fully analyzed)
- ❌ How NaN features are handled by model unknown
- ❌ Whether DataValidator catches these bugs unknown
- ❌ Impact on existing deployed models unknown

### 4. Online vs Offline Consistency
- ✅ One test exists but **doesn't catch the bugs**
- ❌ No verification for all feature types
- ❌ No test for stateful features (RSI, GARCH)

## Test Coverage Analysis

### Tests Created: 44 total
- test_data_pipeline_validation.py: 28 tests ✅ (all pass)
- test_data_corruption_scenarios.py: 16 tests (12 pass, 4 fail)

### Critical Gap: **Tests Don't Validate Feature Logic**
Tests check:
- ✅ Data structure (columns, types, order)
- ✅ Invalid inputs (NaN, negative, inf)
- ✅ Pipeline flow (load→merge→label)

Tests DON'T check:
- ❌ Feature **VALUES** are correct
- ❌ Returns formula produces non-zero values
- ❌ RSI handles zero gain/loss correctly
- ❌ Features match mathematical definitions

## Root Cause Analysis

### Why Bugs Were Missed

1. **Tests were too structural**
   - Checked "NaN exists" but not "should NaN exist here?"
   - Checked "returns column present" but not "returns value correct"

2. **Synthetic data too simple**
   - Constant price increases → always positive returns
   - Didn't test flat/declining markets → zero loss RSI bug missed

3. **Insufficient domain knowledge validation**
   - Assumed returns formula was correct without verification
   - Didn't validate against financial formulas (Wilder's RSI)

4. **No integration with real training**
   - Tests isolated from actual model training pipeline
   - Can't detect if bugs affect model performance

## Recommendations

### Immediate Actions Required 🚨

1. **FIX BUG #1: Returns calculation**
   ```python
   # Use minimum 2 bars for returns or use last_close
   ret = log(price / st["last_close"]) if st["last_close"] else 0.0
   ```

2. **FIX BUG #2: RSI edge cases**
   ```python
   if avg_loss == 0: rsi = 100.0
   elif avg_gain == 0: rsi = 0.0
   else: rsi = 100.0 - (100.0 / (1.0 + avg_gain/avg_loss))
   ```

3. **RE-TRAIN ALL MODELS**
   - Current models trained on corrupted features
   - Need clean data for valid predictions

### Additional Testing Needed

1. **Value-based tests:**
   - Verify returns match expected log(p1/p0)
   - Verify RSI matches Wilder's formula
   - Test with real market data scenarios

2. **Edge case scenarios:**
   - Flat markets (no change)
   - Strong trends (monotonic up/down)
   - Gaps and missing data
   - Extreme volatility

3. **Integration tests:**
   - Load real feather files
   - Process through full pipeline
   - Compare with known-good outputs

## Conclusion

**The audit process itself was valuable but incomplete:**

✅ **Strengths:**
- Comprehensive structural validation
- Good coverage of data integrity
- Found 2 bugs in asof_join.py

❌ **Weaknesses:**
- Missed critical feature calculation bugs
- Tests gave false sense of security
- No validation of mathematical correctness

**Overall Assessment: FAILED** ❌

The pipeline **DOES inject corrupted data** into training:
- Returns are always zero for 4h windows
- RSI is NaN during strong trends
- Model trains on invalid features

**Recommendation: STOP PRODUCTION USE until bugs are fixed.**
