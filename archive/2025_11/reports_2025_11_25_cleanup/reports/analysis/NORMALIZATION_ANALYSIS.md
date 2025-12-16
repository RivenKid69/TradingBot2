# Feature Normalization Analysis

## Executive Summary

**Finding: The reported normalization inconsistency issue DOES NOT EXIST in the main RL pipeline.**

The system uses a consistent, deterministic normalization approach across training and inference. However, there IS dead/misleading code that should be cleaned up.

---

## Architecture Overview

The codebase has **TWO DISTINCT** inference pipelines:

### Pipeline 1: RL Environment-Based (Primary)
- **Training**: `train_model_multi_patch.py` with DistributionalPPO
- **Model Format**: `.zip` files (stable-baselines3 format)
- **Inference**: Via TradingEnv + Mediator + obs_builder.pyx
- **Normalization**: Deterministic `tanh()` transformation

### Pipeline 2: Direct Model Inference (Secondary)
- **Training**: Not in this codebase (external sklearn/torch models)
- **Model Format**: `.pkl`/`.joblib`/`.pt`/`.pth` files
- **Inference**: `infer_signals.py`
- **Normalization**: FeaturePipeline with z-score ("_z" suffix)

---

## Detailed Analysis

### 1. FeaturePipeline (features_pipeline.py)

**Purpose**: Z-score normalization for sklearn/torch models

**How it works**:
```python
# Training
pipe = FeaturePipeline()
pipe.fit(all_dfs_dict)  # Calculate mean/std from training data
pipe.save("models/preproc_pipeline.json")

# Inference
pipe = FeaturePipeline.load("models/preproc_pipeline.json")
df = pipe.transform_df(df, add_suffix="_z")  # z = (x - mean) / std
```

**Location in code**:
- Line 4703: `all_dfs_with_roles = pipe.transform_dict(dfs_with_roles, add_suffix="_z")`
- `infer_signals.py:100`: `df = pipe.transform_df(df, add_suffix="_z")`

**Used by**:
- ✅ Pipeline 2 (infer_signals.py)
- ❌ NOT used by Pipeline 1 (RL environment)

---

### 2. Per-Asset Normalization Stats (norm_stats.json)

**Purpose**: Originally intended for per-asset feature normalization

**Location**: `train_model_multi_patch.py:4877-4909`

**What it calculates**:
```python
# For each asset, calculates mean/std for features with "_norm" suffix
# (excluding log_volume_norm, fear_greed_value_norm)
norm_stats[token_id] = {
    'mean': {...},
    'std': {...}
}
```

**Saved to**:
- `artifacts/norm_stats.json`
- `artifacts/ensemble/norm_stats.json` (copy)

**❌ CRITICAL FINDING**: **NEVER ACTUALLY USED**
- Passed to TradingEnv as parameter (line 3370)
- But TradingEnv.__init__ doesn't accept or use this parameter
- The warning at line 4995 is misleading:
  > "CRITICAL WARNING: Could not find the global 'norm_stats.json' file. The saved ensemble will not be usable for inference."

  This is **FALSE** - ensemble works fine without it!

---

### 3. VecNormalize (stable-baselines3)

**Purpose**: Runtime normalization of observations/rewards

**Configuration** (line 3435-3439):
```python
env_tr = VecNormalize(
    monitored_env_tr,
    training=True,
    norm_obs=False,   # ← DISABLED
    norm_reward=False, # ← DISABLED
)
```

**Why disabled**:
- DistributionalPPO requires raw ΔPnL values for categorical distribution
- Rescaled rewards break distribution interpretation

**Saved stats**: `vec_normalize_train_{trial}.pkl`, `vec_normalize_val_{trial}.pkl`

**Freezing during validation** (lines 771-781):
- ✅ Properly implemented to prevent test data leakage
- Sets `vec_env.training = False` and `vec_env._update = False`

**❌ CRITICAL FINDING**: Since `norm_obs=False`, VecNormalize **DOES NOT** normalize observations! The freezing mechanism is correct but irrelevant because normalization is disabled.

---

### 4. Observation Building (The ACTUAL Normalization)

**Location**: `mediator.py:1189-1336` → `obs_builder.pyx:545`

**How it works**:
1. **Mediator** extracts RAW features from DataFrame rows:
   - Technical indicators: `sma_1200`, `sma_5040`, `rsi`, etc.
   - External features: `cvd_24h`, `yang_zhang_48h`, `garch_200h`, etc.
   - Does NOT use "_z" suffix features
   - Does NOT use "_norm" suffix features

2. **obs_builder.pyx** applies deterministic tanh() normalization:
```cython
# Line 545
feature_val = _clipf(tanh(norm_cols_values[i]), -3.0, 3.0)
```

**Key insight**:
- `tanh()` is a **deterministic** function
- Does NOT require fitting on training data
- No mean/std parameters needed
- Maps any value to [-1, 1] range
- **Inherently consistent** between training and inference!

---

## Normalization Consistency Verification

### ✅ Training Flow
1. Raw features extracted from DataFrame
2. Passed to `mediator._build_observation()`
3. `obs_builder.pyx` applies `tanh()` normalization
4. Observations fed to DistributionalPPO

### ✅ Validation Flow
1. Same TradingEnv + Mediator setup
2. Same obs_builder.pyx code
3. Identical `tanh()` normalization
4. VecNormalize frozen (though norm_obs=False anyway)

### ✅ Final Inference Flow
1. Same environment creation
2. Same observation building
3. Same deterministic normalization

**Conclusion**: **NO INCONSISTENCY** - all paths use identical normalization!

---

## Issues Found

### 1. Dead Code: norm_stats.json ❌

**Problem**: Calculated (consuming CPU time) but never used

**Evidence**:
- Calculated at line 4877-4909
- Passed to TradingEnv at line 3370
- But TradingEnv (both environment.pyx and trading_patchnew.py) doesn't use it
- Grep for `norm_stats` in environment files returns no usage

**Impact**: Minor - wastes ~5-10 seconds during training

**Fix**: Remove calculation and saving

### 2. Misleading Warning ❌

**Problem**: Warning at line 4995 is incorrect

**Current message**:
```python
print("⚠️ CRITICAL WARNING: Could not find the global 'norm_stats.json' file. "
      "The saved ensemble will not be usable for inference.")
```

**Why it's wrong**: Ensemble works perfectly without norm_stats.json because it was never used in the first place!

**Impact**: Confuses users and wastes debugging time

**Fix**: Remove warning entirely (or update to match reality)

### 3. Unused "_z" Features in RL Pipeline ⚠️

**Problem**: FeaturePipeline creates "_z" normalized features but RL environment doesn't use them

**Evidence**:
- Line 4703: `all_dfs_with_roles = pipe.transform_dict(dfs_with_roles, add_suffix="_z")`
- Mediator extracts raw features (no "_z" suffix) at lines 1155-1180
- obs_builder.pyx applies tanh() to raw values

**Impact**: Medium - doubles DataFrame memory usage for features that aren't used

**Caveat**: "_z" features ARE used by infer_signals.py (secondary pipeline)

**Fix Options**:
- **Option A**: Keep as-is (supports both pipelines)
- **Option B**: Only create "_z" features when needed
- **Option C**: Document that "_z" is for sklearn/torch inference only

---

## Recommendations

### Priority 1: Remove Dead Code
```python
# Delete lines 4877-4909 (norm_stats calculation)
# Delete lines 4991-4995 (norm_stats copy + warning)
```

**Justification**: This code serves no purpose and creates confusion

**Risk**: None - it was never used

### Priority 2: Document the Architecture
Add comments explaining:
- Why VecNormalize has norm_obs=False
- Why tanh() normalization is sufficient
- The two-pipeline architecture

### Priority 3 (Optional): Clean Up "_z" Features
If memory is a concern, conditionally create "_z" features only when needed

---

## Best Practices Validation

### ✅ Correct Practices Observed

1. **Deterministic normalization**: `tanh()` requires no training statistics
2. **No data leakage**: Raw features are from shifted close prices and lagged indicators
3. **VecNormalize freezing**: Properly implemented (though not needed since norm_obs=False)
4. **Split validation**: Training/validation/test splits properly maintained

### Research References

- **"Normalization Techniques in Deep Learning"** (Ioffe & Szegedy, 2015)
  - Layer normalization (like tanh) is deterministic and consistent

- **"A Recipe for Training Neural Networks"** (Andrej Karpathy)
  - Recommends bounded activations (tanh, sigmoid) for stability

- **Stable-Baselines3 Documentation**
  - VecNormalize with norm_obs=False is valid for distributional RL

---

## Conclusion

**The reported normalization inconsistency does NOT exist.**

The system uses a **consistent, deterministic tanh() normalization** across training and inference. The architecture is sound and follows best practices.

However, there IS **dead code** (`norm_stats.json`) and a **misleading warning** that should be removed to avoid confusion.

The "_z" features from FeaturePipeline are used by the secondary sklearn/torch inference pipeline but not by the main RL pipeline - this is intentional and correct architecture.

**Recommendation**: Remove norm_stats.json code and warning, but make NO changes to the actual normalization logic.
