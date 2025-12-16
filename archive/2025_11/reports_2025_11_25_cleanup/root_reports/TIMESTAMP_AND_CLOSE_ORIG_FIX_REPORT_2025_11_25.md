# Timestamp Consistency and close_orig Enhancement Report - 2025-11-25

## Executive Summary

**Status**: ✅ **FIXED AND VERIFIED**

**Two issues were analyzed and addressed**:
1. **Problem #1**: Loss of original close price (close_orig) after feature pipeline transformation
   - **Status**: ✅ **ENHANCEMENT ADDED** (optional preserve_close_orig parameter)
   - **Classification**: Design choice, not a bug (online inference uses market data directly)

2. **Problem #2**: Timestamp inconsistency between CSV and Parquet data sources
   - **Status**: ✅ **FIXED** (unified to CLOSE TIME convention)
   - **Classification**: Critical bug causing 4-hour offset and preventing data merging

**Test Coverage**:
- Problem #1: 10/10 tests passed (100%)
- Problem #2: 8/8 tests passed (100%)
- **Total**: 18 new tests + verification scripts

---

## Problem #1: close_orig Preservation

### Issue Description

**Reported Problem**: After `transform_df()`, close column is shifted by 1 period (correct for data leakage prevention), but original unshifted close price was not preserved. Downstream components (execution simulator, PnL calculation) might need access to current price $P_t$.

### Analysis

**Root Cause Analysis**:
- `features_pipeline.py` checks for `close_orig` marker to prevent repeated shifting
- BUT: `close_orig` is never **created** by the pipeline itself
- User must create it manually before transformation

**Investigation Results**:
1. Checked how `close_orig` is used in codebase → Only in tests and feature pipeline logic
2. Checked how execution simulator gets current price → Uses `mediator.py` (`last_mtm_price`, `set_ref_price`)
3. Checked how reward calculation works → Uses real-time market data, not transformed features

**Conclusion**: This is **NOT a bug**, but a **design choice**:
- **Training (offline)**: `features_pipeline` shifts all features to prevent data leakage ✅ Correct
- **Inference (online)**: Execution simulator and reward function get current prices directly from market data feeds ✅ Correct
- **Use case for close_orig**: Post-training analysis, debugging, comparison only (optional)

### Solution: Enhancement

Added **optional** `preserve_close_orig` parameter to `FeaturePipeline`:

```python
# Default behavior (no close_orig created)
pipe = FeaturePipeline()

# Enable close_orig preservation for analysis
pipe = FeaturePipeline(preserve_close_orig=True)
df_transformed = pipe.transform_df(df)
# df_transformed now has 'close_orig' column with unshifted prices
```

**Implementation**:
- [features_pipeline.py:182](features_pipeline.py#L182) - Added `preserve_close_orig` parameter to `__init__()`
- [features_pipeline.py:531-534](features_pipeline.py#L531-L534) - Create `close_orig` before shifting if enabled
- [features_pipeline.py:607-619](features_pipeline.py#L607-L619) - Save/load `preserve_close_orig` flag

**Benefits**:
- ✅ Backward compatible (default `False` - no change to existing behavior)
- ✅ Explicit opt-in for analysis use cases
- ✅ Prevents accidental double-shifting (close_orig acts as marker)
- ✅ Documented in docstring with clear use case explanation

### Test Coverage

**New Tests**: [tests/test_close_orig_enhancement.py](tests/test_close_orig_enhancement.py)

```
TestCloseOrigPreservation (5 tests):
  ✅ test_close_orig_created_when_enabled
  ✅ test_close_orig_not_created_when_disabled (default)
  ✅ test_close_orig_preserved_across_multiple_transforms
  ✅ test_close_orig_per_symbol_shift
  ✅ test_close_orig_with_analysis_use_case

TestSaveLoadPreserveCloseOrig (3 tests):
  ✅ test_save_load_with_preserve_close_orig
  ✅ test_save_load_without_preserve_close_orig
  ✅ test_backward_compatibility_legacy_artifacts

TestDocumentationAndUseCases (2 tests):
  ✅ test_use_case_post_training_analysis
  ✅ test_use_case_debugging_data_leakage

TOTAL: 10/10 tests passed (100%)
```

---

## Problem #2: Timestamp Consistency

### Issue Description

**Reported Problem**: Timestamp inconsistency between CSV and Parquet data sources causing 4-hour offset.

**Detailed Analysis**:
- **CSV** (`_read_raw`): Used `timestamp = (close_time // 14400) * 14400` (floor division) → **OPEN TIME** (start of 4h bar)
- **Parquet** (`_normalize_ohlcv`):
  - With `close_time`: `timestamp = close_time` → **CLOSE TIME** (end of bar)
  - With `open_time`: `timestamp = open_time + 14400` → **CLOSE TIME** (end of bar)

**Impact**:
- ❌ CSV timestamp for 00:00-04:00 bar = 0 (OPEN TIME)
- ❌ Parquet timestamp for same bar = 14399 (CLOSE TIME)
- ❌ **4-hour offset** (14399 seconds = 3.999 hours)
- ❌ Cannot merge/concat data from different sources
- ❌ `merge_asof` fails to align rows correctly
- ❌ Features from different sources temporally misaligned
- ❌ Creates look-ahead bias or lag in combined datasets

### Solution: Timestamp Unification

**Fix**: Unified timestamp semantics to **CLOSE TIME** consistently across all data sources.

**Changes**:

#### 1. CSV Path (_read_raw)

**Before** (WRONG):
```python
df["timestamp"] = (df["close_time"] // 14400) * 14400  # OPEN TIME (floor division)
```

**After** (CORRECT):
```python
df["timestamp"] = df["close_time"]  # CLOSE TIME (direct assignment)
```

**File**: [prepare_and_run.py:52-56](prepare_and_run.py#L52-L56)

#### 2. Parquet Path (_normalize_ohlcv)

**No changes needed** - Already uses CLOSE TIME:
- With `close_time`: Uses close_time directly ✅
- With `open_time`: Adds bar duration (open_time + 14400) ✅

**Rationale**:
- Binance `close_time` convention: end of interval minus 1ms (e.g., 14399 for 00:00-04:00 bar)
- Using `close_time` directly aligns with market data semantics
- Allows successful merge/concat operations between CSV and Parquet sources

### Verification

**Test Results**:

```
Before fix:
  CSV timestamp:     0      (OPEN TIME)
  Parquet timestamp: 14399  (CLOSE TIME)
  Difference:        14399 seconds (4.0 hours) ❌ INCONSISTENT

After fix:
  CSV timestamp:     14399  (CLOSE TIME)
  Parquet timestamp: 14399  (CLOSE TIME)
  Difference:        0 seconds (0.000 hours) ✅ CONSISTENT
```

**merge_asof Test**:
```python
df_merged = pd.merge_asof(df_csv, df_additional, on='timestamp', direction='backward')
# ✅ SUCCESS - rows aligned correctly
# ✅ Features from both sources temporally consistent
```

### Test Coverage

**New Tests**: [tests/test_timestamp_consistency_fix.py](tests/test_timestamp_consistency_fix.py)

```
TestTimestampConsistency (6 tests):
  ✅ test_csv_uses_close_time_not_floor
  ✅ test_parquet_close_time_unchanged
  ✅ test_parquet_open_time_consistency
  ✅ test_csv_parquet_timestamp_alignment
  ✅ test_merge_asof_works_after_fix
  ✅ test_no_4hour_offset_in_features

TestBackwardCompatibility (2 tests):
  ✅ test_multiple_bars_processing
  ✅ test_ms_to_seconds_conversion

TOTAL: 8/8 tests passed (100%)
```

---

## Files Modified

### Problem #1 Enhancement

1. **[features_pipeline.py](features_pipeline.py)**:
   - Added `preserve_close_orig` parameter (line 182)
   - Implemented close_orig creation logic (lines 531-534)
   - Updated save/load methods (lines 607-619, 621-643)

### Problem #2 Fix

2. **[prepare_and_run.py](prepare_and_run.py)**:
   - Fixed `_read_raw()` to use close_time directly (lines 52-56)
   - Added documentation for timestamp semantics (lines 32-42)
   - Documented Fear & Greed and Events floor division (lines 286-335)

### New Tests

3. **[tests/test_close_orig_enhancement.py](tests/test_close_orig_enhancement.py)** ⭐ NEW
   - 10 comprehensive tests for close_orig preservation

4. **[tests/test_timestamp_consistency_fix.py](tests/test_timestamp_consistency_fix.py)** ⭐ NEW
   - 8 comprehensive tests for timestamp consistency

5. **[test_reported_issues.py](test_reported_issues.py)** ⭐ NEW
   - Verification script for both issues
   - Updated to reflect fixes (now shows SUCCESS)

---

## Impact Assessment

### Problem #1 (close_orig)

**Impact**: ✅ **Low (Enhancement, not a bug)**

**Affected Systems**: None (online inference uses market data directly)

**Backward Compatibility**: ✅ **Fully maintained**
- Default behavior unchanged (`preserve_close_orig=False`)
- Existing code continues to work
- No model retraining required

**Use Cases Enabled**:
- Post-training analysis (compare predictions vs actual prices)
- Debugging data leakage (verify temporal alignment)
- Research and experimentation

### Problem #2 (Timestamp)

**Impact**: ⚠️ **High (Critical bug fix)**

**Affected Systems**:
- Data preprocessing (`prepare_and_run.py`)
- Any code merging CSV and Parquet data
- Feature engineering pipelines combining multiple data sources

**Backward Compatibility**: ⚠️ **Breaking change for CSV processing**

**Action Required**:
1. **Data reprocessing**: CSV files processed with old version will have OPEN TIME timestamps
2. **Recommendation**: Reprocess all CSV data with new version to ensure CLOSE TIME consistency
3. **Detection**: Check for 4-hour offsets in merged datasets
4. **Timeline**: Non-urgent (only affects new data merging operations)

**Benefits After Fix**:
- ✅ CSV and Parquet data can be merged without temporal misalignment
- ✅ Features from different sources temporally consistent
- ✅ No look-ahead bias or lag in combined datasets
- ✅ Simplified data pipeline (one consistent timestamp convention)

---

## Best Practices

### When to Use preserve_close_orig

**Enable** (`preserve_close_orig=True`) when:
- ✅ Post-training analysis (comparing predictions vs actual prices)
- ✅ Debugging data leakage issues
- ✅ Research experiments requiring both shifted and unshifted prices
- ✅ Offline analysis only

**Disable** (`preserve_close_orig=False`, default) when:
- ✅ Standard ML training pipeline
- ✅ Production inference (online trading)
- ✅ Memory-constrained environments
- ✅ Following best practices (ML pipelines should be self-contained)

### Timestamp Convention

**Standard Practice** (after fix):
- ✅ **CLOSE TIME** for all data sources (CSV, Parquet, online feeds)
- ✅ Binance convention: `close_time` = end of interval minus 1ms
- ✅ Consistent merge/concat operations
- ✅ No temporal misalignment

**Special Cases**:
- Fear & Greed: Daily data, floor to 4h boundaries for alignment (acceptable)
- Economic Events: Precise timestamps, floor to 4h for merge_asof tolerance (acceptable)

---

## References

### Code

- [features_pipeline.py](features_pipeline.py) - Feature preprocessing pipeline
- [prepare_and_run.py](prepare_and_run.py) - Data ingestion and normalization
- [mediator.py](mediator.py) - Online price feeds (execution simulator)

### Tests

- [tests/test_close_orig_enhancement.py](tests/test_close_orig_enhancement.py) - close_orig tests
- [tests/test_timestamp_consistency_fix.py](tests/test_timestamp_consistency_fix.py) - Timestamp tests
- [test_reported_issues.py](test_reported_issues.py) - Original verification script

### Documentation

- [DATA_LEAKAGE_FIX_REPORT_2025_11_23.md](DATA_LEAKAGE_FIX_REPORT_2025_11_23.md) - Related fix (feature shifting)
- [CLAUDE.md](CLAUDE.md) - Project documentation (will be updated)

---

## Summary

**Problem #1: close_orig Enhancement**
- ✅ **ENHANCEMENT ADDED** - Optional `preserve_close_orig` parameter
- ✅ Backward compatible (default `False`)
- ✅ Use case: Post-training analysis and debugging only
- ✅ Online inference unaffected (uses market data directly)
- ✅ 10/10 tests passed

**Problem #2: Timestamp Consistency**
- ✅ **FIXED** - Unified to CLOSE TIME convention
- ✅ CSV now consistent with Parquet (0 second offset)
- ⚠️ Breaking change - requires data reprocessing
- ✅ Enables successful merge/concat operations
- ✅ 8/8 tests passed

**Overall**:
- ✅ **18 new tests** (100% pass rate)
- ✅ Comprehensive documentation
- ✅ Production ready
- ⚠️ Recommend reprocessing CSV data for consistency

---

**Date**: 2025-11-25
**Version**: 1.0
**Status**: ✅ **PRODUCTION READY**
