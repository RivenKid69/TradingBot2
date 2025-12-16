# Explained Variance Bugs - Quick Summary

**Date**: 2025-11-22
**Status**: **3 CRITICAL BUGS FOUND** ❌

---

## 🔴 Critical Bugs Confirmed

### 1. Bug #1.1: Quantile Mode EV Uses CLIPPED Predictions
- **Location**: Line 10814
- **Current**: `quantiles_for_ev = quantiles_norm_clipped_for_loss` ❌
- **Should be**: `quantiles_for_ev = quantiles_fp32[valid_indices]` ✅

### 2. Bug #1.2: Categorical Mode EV Uses CLIPPED Predictions
- **Location**: Line 11357
- **Current**: `mean_values_norm_clipped_selected.reshape(-1, 1)` ❌
- **Should be**: `mean_values_norm_selected.reshape(-1, 1)` ✅

### 3. Bug #6: Missing Epsilon in Ratio Denominator
- **Location**: Line 352 (and 370)
- **Current**: `ratio = var_res / var_y` ❌
- **Should be**: `ratio = var_res / (var_y + 1e-12)` ✅

---

## 📊 Impact

- **EV metric is BIASED** when VF clipping is enabled
- Cannot reliably assess model quality
- May hide value function issues or show false improvements

---

## ✅ What's Working

1. **EV Unavailability Logging**: Already implemented ✓
2. **Weighted Variance Formula**: Mathematically correct (reliability weights) ✓
3. **Bessel's Correction**: Properly implemented ✓

---

## 🎯 Immediate Action Required

**Fix Priority 1 Bugs (This Week)**:

```python
# FIX #1.1 - Line 10814 in distributional_ppo.py
# BEFORE:
quantiles_for_ev = quantiles_norm_clipped_for_loss

# AFTER:
if valid_indices is not None:
    quantiles_for_ev = quantiles_fp32[valid_indices]
else:
    quantiles_for_ev = quantiles_fp32
```

```python
# FIX #1.2 - Line 11357 in distributional_ppo.py
# BEFORE:
value_pred_norm_for_ev = (
    mean_values_norm_clipped_selected.reshape(-1, 1)
)

# AFTER:
value_pred_norm_for_ev = (
    mean_values_norm_selected.reshape(-1, 1)
)
```

```python
# FIX #6 - Line 352 and 370 in distributional_ppo.py
# BEFORE:
ratio = var_res / var_y

# AFTER:
ratio = var_res / (var_y + 1e-12)
```

---

## 📝 Recommendations (Optional)

1. **Twin Critics EV Logging**: Log separate EV for Q1 and Q2
2. **Enhanced Documentation**: Add Wikipedia reference for weighted variance

---

## 🧪 Testing

**Run after applying fixes**:
```bash
# Regression tests
pytest tests/test_distributional_ppo*.py -v
pytest tests/test_twin_critics*.py -v

# Audit script
python test_ev_bugs_direct.py
```

---

## 📄 Full Report

See [EXPLAINED_VARIANCE_BUGS_REPORT_2025_11_22.md](EXPLAINED_VARIANCE_BUGS_REPORT_2025_11_22.md) for:
- Detailed impact assessment
- Implementation plan
- Testing strategy
- Verification checklist

---

**Next Steps**: Apply the 3 fixes above, run tests, verify EV is now unbiased.
