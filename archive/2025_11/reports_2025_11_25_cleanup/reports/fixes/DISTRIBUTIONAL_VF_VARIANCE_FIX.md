# Fix: Distributional VF Clipping Variance Calculation

## Problem Description

When using `distributional_vf_clip_mode="mean_and_variance"`, the code was incorrectly computing `old_variance` from **current predictions** instead of **old distributions** stored in the rollout buffer.

### Affected Code

**Quantile Critic** (distributional_ppo.py:8840-8841, before fix):
```python
# INCORRECT: Uses current quantiles, not old ones from rollout
old_quantiles_centered = quantiles_fp32 - value_pred_norm_full
old_variance = (old_quantiles_centered ** 2).mean(dim=1, keepdim=True)
```

**Categorical Critic** (distributional_ppo.py:8993-8995, before fix):
```python
# INCORRECT: Assumes uniform distribution, not actual old probs
old_atoms_centered_approx = atoms_original - old_mean_norm.squeeze(-1)
old_variance_approx = (old_atoms_centered_approx ** 2).mean()
```

### Why This Was Wrong

VF clipping is designed to limit changes relative to the **old value function** from the rollout buffer. Computing `old_variance` from current predictions defeats the purpose:

1. **Quantile critic**: Used `quantiles_fp32` (current predictions) instead of the actual old quantiles that were stored during rollout collection
2. **Categorical critic**:
   - Used current atoms instead of old distribution
   - Assumed uniform probability distribution (`.mean()`) instead of using the actual old probabilities

This meant the variance constraint was comparing the new distribution's variance against the **wrong baseline**.

## Solution

### 1. Extended Rollout Buffer

Added fields to store distributional information:

**RawRecurrentRolloutBufferSamples** (distributional_ppo.py:600-601):
```python
old_value_quantiles: Optional[torch.Tensor]  # For quantile critic
old_value_probs: Optional[torch.Tensor]      # For categorical critic
```

**RawRecurrentRolloutBuffer.reset()** (distributional_ppo.py:1288-1289):
```python
self.value_quantiles: Optional[np.ndarray] = None  # Shape: [buffer_size, n_envs, n_quantiles]
self.value_probs: Optional[np.ndarray] = None      # Shape: [buffer_size, n_envs, n_atoms]
```

### 2. Updated Buffer Operations

**Modified add() method** (distributional_ppo.py:1303-1304):
- Now accepts `value_quantiles` and `value_probs` parameters
- Stores them in the buffer with lazy initialization

**Modified collect_rollouts()** (distributional_ppo.py:6655-6675):
- Extracts quantiles/probs after forward pass
- Passes them to `rollout_buffer.add()`

### 3. Fixed Variance Calculations

**Quantile Critic** (distributional_ppo.py:8837-8854):
```python
# CORRECT: Use actual old quantiles from rollout buffer
if rollout_data.old_value_quantiles is not None:
    old_quantiles_norm = rollout_data.old_value_quantiles.to(...)
    old_mean_norm = rollout_data.old_values.to(...).unsqueeze(-1)
    old_quantiles_centered = old_quantiles_norm - old_mean_norm
    old_variance = (old_quantiles_centered ** 2).mean(dim=1, keepdim=True)
else:
    # Fallback: rough approximation
    ...
```

**Categorical Critic** (distributional_ppo.py:9061-9081):
```python
# CORRECT: Use actual old probabilities from rollout buffer
if rollout_data.old_value_probs is not None:
    old_probs_norm = rollout_data.old_value_probs.to(...)
    old_mean_norm = rollout_data.old_values.to(...)
    old_atoms_centered = atoms_original - old_mean_norm.squeeze(-1)
    # Proper weighted variance using old probabilities
    old_variance_approx = ((old_atoms_centered ** 2) * old_probs_norm).sum(dim=1, keepdim=True)
else:
    # Fallback: rough approximation
    ...
```

## Impact

### Before Fix
- Variance constraint was comparing against the **wrong reference** (current predictions instead of old distributions)
- For quantile critic: Used current quantile spread instead of old quantile spread
- For categorical critic: Assumed uniform distribution instead of actual probability distribution
- VF clipping behavior was **unpredictable** and **not properly constraining variance changes**

### After Fix
- Variance constraint now correctly compares:
  - **Old variance**: Computed from distributions stored during rollout collection
  - **New variance**: Computed from current predictions
- Properly enforces the constraint: `new_variance ≤ old_variance × factor²`
- Categorical critic now uses **weighted variance** with actual probabilities, not uniform assumption

## Testing

Created comprehensive test suite in `test_vf_variance_calculation.py`:

1. **test_quantile_variance_from_old_quantiles**: Verifies quantile variance uses old quantiles, not current
2. **test_categorical_variance_from_old_probs**: Verifies categorical variance uses old probs with proper weighting
3. **test_variance_constraint_correctness**: Validates constraint enforcement logic
4. **test_rollout_buffer_stores_distributions**: Checks buffer data structure contracts
5. **test_variance_calculation_numerical_stability**: Ensures numerical stability with extreme values

## Memory Impact

**Quantile critic**: `+buffer_size × n_envs × n_quantiles × 4 bytes`
- Example: 128 steps × 8 envs × 51 quantiles × 4 bytes = **~209 KB**

**Categorical critic**: `+buffer_size × n_envs × n_atoms × 4 bytes`
- Example: 128 steps × 8 envs × 51 atoms × 4 bytes = **~209 KB**

This is a negligible increase compared to overall memory usage, and essential for correct VF clipping behavior.

## Backward Compatibility

- Existing code without `distributional_vf_clip_mode="mean_and_variance"` is **unaffected**
- The fix includes fallback logic if old distributions are not available (though this should not occur in normal operation)
- Buffer gracefully handles both quantile and categorical critics (only stores what's needed)

## Related Issues

This fix completes the distributional VF clipping implementation, addressing:
- Conceptual error in variance baseline computation
- Incorrect uniform distribution assumption in categorical critic
- Missing storage of old distributions in rollout buffer

## References

- **PPO Paper**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **Distributional RL**: Bellemare et al., "A Distributional Perspective on Reinforcement Learning" (2017)
- **Previous Fix**: `DISTRIBUTIONAL_VF_CLIPPING_FIX.md` (corrected target clipping)
