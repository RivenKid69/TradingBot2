# CRITICAL FIX: close_orig Semantic Inconsistency (2025-11-25)

## Executive Summary

**Status**: ✅ **FIXED** (2025-11-25)
**Severity**: **CRITICAL** - Caused DATA LEAKAGE in training pipeline
**Impact**: Models trained before this fix saw FUTURE information

---

## Problem Description

### Root Cause

The `close_orig` column had **conflicting semantic meanings** between components:

| Component | Location | Semantic Meaning | Action |
|-----------|----------|------------------|--------|
| `fetch_all_data_patch.py` | Lines 124-127 | "Backup copy of original close" | Created `close_orig` as COPY (data NOT shifted) |
| `features_pipeline.py` | Lines 326-330 | "Marker that data ALREADY shifted" | When `close_orig` present → SKIPPED shifting! |
| `trading_patchnew.py` | Lines 305-311 | "Marker that data ALREADY shifted" | When `close_orig` present → SKIPPED shifting! |

### Data Flow Analysis

**BEFORE FIX (BUG):**
```
1. load_all_data() → Creates close_orig as COPY (data NOT shifted)
2. features_pipeline.fit() → Sees close_orig → SKIPS shift ❌ BUG!
3. features_pipeline.transform_dict() → Sees close_orig → SKIPS shift ❌ BUG!
4. TradingEnv → Sees close_orig → SKIPS shift ❌ BUG!
5. Model → Sees UNSHIFTED features (RSI, MACD, etc.) → DATA LEAKAGE! ❌
```

**AFTER FIX:**
```
1. load_all_data() → NO close_orig created ✅
2. features_pipeline.fit() → NO close_orig → SHIFTS all features ✅
3. features_pipeline.transform_dict() → NO close_orig → SHIFTS all features ✅
4. features_pipeline adds _close_shifted marker column ✅
5. TradingEnv → Sees _close_shifted → SKIPS double-shifting ✅
6. Model → Sees SHIFTED features → NO DATA LEAKAGE ✅
```

### Impact

- **All models trained before 2025-11-25** potentially learned from **FUTURE information**
- Technical indicators (RSI, MACD, Bollinger Bands, etc.) were NOT shifted
- Model saw current-period indicators but was supposed to make decisions based on past information
- **Backtest results were INFLATED** (model had unfair advantage)
- **Live trading performance was DEGRADED** (model learned spurious patterns)

---

## Fix Details

### 1. Removed close_orig/close_prev from fetch_all_data_patch.py

**Location**: `fetch_all_data_patch.py:124-153`

**Change**: Removed the following code that created the conflicting `close_orig`:
```python
# REMOVED:
# if "close" in df.columns:
#     df["close_orig"] = df["close"].astype(float)
#     df["close_prev"] = df["close_orig"].shift(1)
```

**Rationale**: This code conflicted with the semantic meaning in `features_pipeline.py`

### 2. Added _close_shifted marker in features_pipeline.py

**Location**: `features_pipeline.py:343-345` (fit) and `features_pipeline.py:550-553` (transform_df)

**Change**: Added column-based marker for TradingEnv compatibility:
```python
# After shifting:
out["_close_shifted"] = True
```

**Rationale**: TradingEnv checks for column markers (`_close_shifted`), not DataFrame attrs

### 3. Updated METADATA_COLUMNS

**Location**: `features_pipeline.py:39-43`

**Change**: Added `_close_shifted` to metadata columns set:
```python
METADATA_COLUMNS = {
    "timestamp", "symbol", "wf_role", "close_orig", "_close_shifted",
}
```

**Rationale**: Prevent `_close_shifted` from being treated as a feature column

### 4. Updated _columns_to_scale exclusion

**Location**: `features_pipeline.py:159-173`

**Change**: Added `_close_shifted` to exclusion set:
```python
exclude = {"timestamp", "_close_shifted"}
```

**Rationale**: Prevent `_close_shifted` from being z-scored

---

## Test Coverage

### New Tests: `tests/test_close_orig_semantic_fix.py`

| Test Class | Test Count | Description |
|------------|------------|-------------|
| `TestFetchAllDataPatchNoCloseOrig` | 1 | Verifies `load_all_data()` no longer creates `close_orig` |
| `TestFeaturesPipelineShiftingWithoutCloseOrig` | 3 | Verifies shifting when `close_orig` absent |
| `TestFeaturesPipelineWithCloseOrig` | 2 | Verifies skip-shift when `close_orig` present |
| `TestTradingEnvCompatibility` | 3 | Verifies `_close_shifted` marker works |
| `TestFullPipelineIntegration` | 2 | **CRITICAL** - Full pipeline no double-shift |
| `TestBackwardCompatibility` | 2 | Verifies backward compatibility |

**Total**: 13 new tests, **100% pass rate**

### Existing Tests (Verified Passing)

- `tests/test_data_leakage_prevention.py`: 17/17 passed ✅
- `tests/test_features_pipeline_fixes_2025_11_21.py`: All passed ✅
- `tests/test_close_orig_enhancement.py`: 11/11 passed ✅

---

## Action Required

### For Models Trained Before 2025-11-25

⚠️ **CRITICAL**: All models trained before this fix **MUST BE RETRAINED**!

Models trained before this fix learned from **future information** due to:
- Unshifted technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Unshifted OHLCV data in training features

**Expected impact after retraining**:
- ❌ Backtest performance will DECREASE (data leak removed)
- ✅ Live trading performance will IMPROVE (models learn genuine patterns)
- ✅ Backtest-live gap will CLOSE dramatically

### For New Training Runs

✅ **No action required** - fix is automatically applied

1. Clear any cached artifacts: `rm -rf artifacts/preproc_pipeline.json`
2. Start fresh training run
3. Features will be properly shifted by 1 period

---

## Technical Details

### Why close_orig Was Originally Created

The original intent in `fetch_all_data_patch.py` was:
```python
# "Не ломаем OHLC: оставляем close как есть, а «прошлый close» кладём отдельно"
# Translation: "Don't break OHLC: keep close as is, put previous close in separate column"
```

This was meant to provide a convenience column `close_prev` for features. However:
1. `close_prev` was never used anywhere in the codebase
2. `close_orig` conflicted with the semantic meaning in `features_pipeline.py`
3. The net effect was DATA LEAKAGE

### Why _close_shifted Marker

TradingEnv (`trading_patchnew.py`) checks for shift markers:
```python
if "close_orig" in self.df.columns:
    self._close_actual = self.df["close_orig"].copy()
elif "close" in self.df.columns and "_close_shifted" not in self.df.columns:
    # Apply shift only once...
    self.df["close"] = self.df["close"].shift(1)
    self.df["_close_shifted"] = True
```

By adding `_close_shifted = True` in `features_pipeline.py`, we signal to TradingEnv:
"Data has already been shifted by features_pipeline - don't double-shift!"

---

## References

- **CLAUDE.md**: DATA_LEAKAGE_FIX_REPORT_2025_11_23.md (previous fix)
- **Test file**: `tests/test_close_orig_semantic_fix.py`
- **Changed files**:
  - `fetch_all_data_patch.py`
  - `features_pipeline.py`

---

## Changelog

| Date | Version | Change |
|------|---------|--------|
| 2025-11-25 | 1.0 | Initial fix - removed close_orig creation, added _close_shifted marker |

---

**Author**: Claude Code
**Date**: 2025-11-25
**Status**: ✅ FIXED AND VERIFIED
