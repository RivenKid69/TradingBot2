# Migration Guide: 56 → 62 Features

## Overview
This guide helps you migrate from the 56-feature observation system to the new 62-feature system with validity flags for all technical indicators.

**Breaking Change**: Observation shape changed from `(56,)` to `(62,)`

## What Changed?

### Added: 6 Validity Flags
We added explicit validity flags for the following indicators to eliminate ambiguity during warmup period:

| Indicator | Warmup | Fallback | Index (NEW) | Problem Solved |
|-----------|--------|----------|-------------|----------------|
| rsi14 | 14 bars | 50.0 | 8 | Can now distinguish neutral (50) from no data (50) |
| macd | 26 bars | 0.0 | 10 | Can now distinguish no divergence from no data |
| macd_signal | 35 bars | 0.0 | 12 | Can now distinguish no signal from no data |
| momentum | 10 bars | 0.0 | 14 | Can now distinguish no movement from no data |
| cci | 20 bars | 0.0 | 17 | Can now distinguish average level from no data |
| obv | 1 bar | 0.0 | 19 | Can now distinguish balance from no data |

### Index Shifts
All features after index 7 (rsi14) shifted **+6 positions**:

```
OLD (56)         NEW (62)         Shift
---------        ---------        -----
0-7: same        0-7: same        0
-                8: rsi_valid     NEW
8: macd          9: macd          +1
-                10: macd_valid   NEW
9: macd_signal   11: macd_signal  +2
-                12: macd_sig_val NEW
10: momentum     13: momentum     +3
-                14: momentum_val NEW
11: atr          15: atr          +4
12: cci          16: cci          +4
-                17: cci_valid    NEW
13: obv          18: obv          +5
-                19: obv_valid    NEW
14: ret_bar      20: ret_bar      +6
...              ...              +6
32: external[0]  38: external[0]  +6
52: external[20] 58: external[20] +6
53: token_ratio  59: token_ratio  +6
54: token_id     60: token_id     +6
55: token_oh[0]  61: token_oh[0]  +6
```

---

## Migration Checklist

### ☐ Step 1: Verify Your Environment
```bash
# Check current branch
git status

# Ensure you're on the correct branch
git branch
```

### ☐ Step 2: Recompile Cython Modules
```bash
# Clean old builds
rm -rf build/
rm -f *.so *.c obs_builder.cpp lob_state_cython.cpp

# Rebuild
python setup.py build_ext --inplace

# Verify new size
python -c "from lob_state_cython import _compute_n_features; print(f'N_FEATURES: {_compute_n_features()}')"
# Expected output: N_FEATURES: 62
```

### ☐ Step 3: Update Feature Config
Check that `feature_config.py` returns 62:
```python
from feature_config import N_FEATURES
assert N_FEATURES == 62, f"Expected 62, got {N_FEATURES}"
```

### ☐ Step 4: Delete Old Model Checkpoints
⚠️ **CRITICAL**: Old models are incompatible!

```bash
# Backup old checkpoints (optional)
mkdir -p checkpoints_56_backup/
mv checkpoints/*.pt checkpoints_56_backup/ 2>/dev/null || true

# Or delete entirely
rm -rf checkpoints/*.pt
```

### ☐ Step 5: Update Custom Code
If you have custom code accessing observation indices:

**OLD (56 features):**
```python
# Feature indices (OUTDATED)
ret_bar_idx = 14        # ❌ WRONG in 62-feature system
external_start = 32     # ❌ WRONG in 62-feature system
```

**NEW (62 features):**
```python
# Feature indices (CORRECT)
ret_bar_idx = 20        # ✅ Shifted +6
external_start = 38     # ✅ Shifted +6

# Better: Use symbolic names
from feature_config import FEATURES_LAYOUT
# Find index by name instead of hardcoding
```

### ☐ Step 6: Run Tests
```bash
# Run all tests to verify compatibility
pytest tests/ -v

# Specific test suites
pytest tests/test_full_feature_pipeline_62.py -v
pytest tests/test_technical_indicators_in_obs.py -v
pytest tests/test_mediator_integration.py -v
```

### ☐ Step 7: Verify Observation Shape
```python
import numpy as np
from mediator import Mediator

# Create test environment
env = YourEnvironment(...)
mediator = Mediator(env, event_level=0)

# Build test observation
obs = mediator._build_observation(...)

# Verify shape
assert obs.shape == (62,), f"Expected (62,), got {obs.shape}"
assert np.all(np.isfinite(obs)), "Observation contains NaN/Inf"
```

---

## Example: Warmup Period Observations

### Before (56 features) - AMBIGUOUS ❌
```python
# Bar 10: Not enough data for RSI (requires 14 bars)
obs[7] = 50.0   # rsi14 = 50.0 (fallback)

# Question: Is this neutral RSI or insufficient data?
# Answer: AMBIGUOUS! Cannot tell the difference.
```

### After (62 features) - CLEAR ✅
```python
# Bar 10: Not enough data for RSI (requires 14 bars)
obs[7] = 50.0   # rsi14 = 50.0 (fallback)
obs[8] = 0.0    # rsi_valid = 0.0 (INVALID)

# Bar 20: Enough data, RSI is actually neutral
obs[7] = 50.0   # rsi14 = 50.0 (real value)
obs[8] = 1.0    # rsi_valid = 1.0 (VALID)

# NOW we can distinguish neutral from no data!
```

---

## Common Migration Errors

### Error 1: Shape Mismatch
```
ValueError: observation shape is (56,), expected (62,)
```

**Fix**: Recompile Cython modules (Step 2)

### Error 2: Index Out of Bounds
```
IndexError: index 55 is out of bounds for axis 0 with size 62
```

**Fix**: Update hardcoded indices (Step 5)

### Error 3: Model Loading Fails
```
RuntimeError: Expected observation size 56, got 62
```

**Fix**: Delete old checkpoints and retrain (Step 4)

### Error 4: Test Failures
```
AssertionError: Expected shape (56,), got (62,)
```

**Fix**: All tests already updated. Rerun: `pytest tests/ -v`

---

## FAQ

### Q: Can I load models trained on 56 features?
**A**: No. The observation shape is fundamentally incompatible. You must retrain.

### Q: Will my old data still work?
**A**: Yes. The raw data (OHLCV, indicators) is unchanged. Only the observation vector format changed.

### Q: How do I know if warmup is complete?
**A**: Check validity flags! All should be 1.0 after max(35) bars (MACD signal warmup).

```python
# Check if all validity flags are ready
validity_flags = [obs[4], obs[6], obs[8], obs[10], obs[12], obs[14], obs[17], obs[19]]
all_ready = all(flag == 1.0 for flag in validity_flags)
```

### Q: Do I need to change my reward function?
**A**: No, unless it directly accessed observation indices. Use `state` attributes instead.

### Q: What about visualization code?
**A**: Update any code that plots observation features by index. Use feature names from `FEATURE_MAPPING_62.md`.

---

## Rollback Instructions

If you need to rollback to 56 features:

```bash
# 1. Find the last commit before this change
git log --oneline | grep -B 1 "BREAKING CHANGE: Add validity flags"

# 2. Checkout previous commit
git checkout <commit-hash-before-change>

# 3. Recompile
python setup.py build_ext --inplace

# 4. Verify
python -c "from lob_state_cython import _compute_n_features; print(_compute_n_features())"
# Should output: 56
```

---

## Benefits of New System

### 1. Eliminates Ambiguity
```python
# OLD: Cannot tell if RSI=50 is neutral or missing data
if obs[7] == 50.0:
    # Is this neutral or warmup? Unknown! ❌

# NEW: Clear distinction
if obs[8] == 0.0:  # rsi_valid flag
    # Definitely warmup ✅
elif obs[7] == 50.0:
    # Definitely neutral RSI ✅
```

### 2. Better Model Training
- Model can learn to ignore invalid indicators during warmup
- Reduces noise in early episodes
- More stable training dynamics

### 3. Improved Debugging
```python
# Check which indicators are ready
def check_indicator_readiness(obs):
    indicators = {
        'ma5': obs[4],
        'ma20': obs[6],
        'rsi14': obs[8],
        'macd': obs[10],
        'macd_signal': obs[12],
        'momentum': obs[14],
        'cci': obs[17],
        'obv': obs[19],
    }
    ready = {k: v == 1.0 for k, v in indicators.items()}
    return ready

# Output: {'ma5': True, 'ma20': False, 'rsi14': False, ...}
```

### 4. Warmup Detection
```python
# Detect if still in warmup period
def is_warmup_complete(obs):
    validity_flags = [obs[i] for i in [4, 6, 8, 10, 12, 14, 17, 19]]
    return all(flag == 1.0 for flag in validity_flags)
```

---

## Support

If you encounter issues during migration:

1. Check this guide's troubleshooting section
2. Review `FEATURE_MAPPING_62.md` for index reference
3. Run tests: `pytest tests/ -v --tb=short`
4. Check git history: `git log --oneline --grep="validity"`

---

## Summary

✅ **Must Do**:
- Recompile Cython modules
- Delete old model checkpoints
- Update hardcoded observation indices
- Run tests to verify

⚠️ **Critical**:
- Observation shape: `(56,)` → `(62,)`
- All indices after 7 shifted +6
- Models must be retrained

🎯 **Benefits**:
- Clear warmup detection
- Eliminate ambiguous fallbacks
- Better model training
- Easier debugging
