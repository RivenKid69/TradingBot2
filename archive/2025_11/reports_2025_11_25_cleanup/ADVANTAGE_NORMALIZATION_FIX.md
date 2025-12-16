# Advantage Normalization Fix

## Summary

Fixed advantage normalization to follow standard PPO practice by normalizing advantages **globally** (once for entire rollout buffer) instead of **per-group** (during training).

## Changes Made

### 1. Added Global Normalization in `collect_rollouts()`

**Location:** `distributional_ppo.py:6466-6481`

After computing advantages with GAE, we now normalize them globally:

```python
# Normalize advantages globally (standard PPO practice)
# This ensures consistent learning signal across all samples and proper gradient accumulation
if self.normalize_advantage and rollout_buffer.advantages is not None:
    advantages_flat = rollout_buffer.advantages.reshape(-1).astype(np.float64)
    adv_mean = float(np.mean(advantages_flat))
    adv_std = float(np.std(advantages_flat))
    adv_std_clamped = max(adv_std, 1e-8)

    # Normalize in-place
    rollout_buffer.advantages = (
        (rollout_buffer.advantages - adv_mean) / adv_std_clamped
    ).astype(np.float32)

    # Log global normalization statistics
    self.logger.record("train/advantages_mean_raw", adv_mean)
    self.logger.record("train/advantages_std_raw", adv_std)
```

### 2. Removed Group-Level Normalization from `train()`

**Location:** `distributional_ppo.py:7692-7778`

Removed the following:
- Collection of per-group advantage statistics
- Computation of `group_adv_mean` and `group_adv_std`
- Application of per-group normalization

Advantages are now used directly from the buffer (already normalized):

```python
# Advantages are already globally normalized in collect_rollouts()
# Just extract the relevant samples based on mask
advantages_flat = advantages.reshape(-1)
if valid_indices is not None:
    advantages_selected = advantages_flat[valid_indices]
else:
    advantages_selected = advantages_flat
```

### 3. Updated Logging

- **Removed:** Per-group `adv_mean` and `adv_std` accumulation during training
- **Added:** Global `advantages_mean_raw` and `advantages_std_raw` in `collect_rollouts()`

## Why This Fix Is Important

### Problem with Group-Level Normalization

#### 1. **Inconsistent Learning Signal**

Same raw advantage values get different normalized values depending on which group they're in:

```python
# Example:
# Group A: advantages = [100, 110, 120] → mean=110, std=8.16
# Group B: advantages = [-5, -6, -5.5] → mean=-5.5, std=0.43

# Raw advantage = 5:
# In Group A: (5 - 110) / 8.16 = -12.87
# In Group B: (5 - (-5.5)) / 0.43 = +24.42
```

Same action quality receives vastly different policy updates!

#### 2. **Bias with Unbalanced Groups**

Groups with different sizes bias the normalization:

```python
# Group 1: 1000 samples, high advantages (mean=50, std=10)
# Group 2: 10 samples, low advantages (mean=-5, std=2)
```

Group 2's normalization is unreliable due to small sample size.

#### 3. **Broken Gradient Accumulation**

Gradient accumulation assumes:
```
∇L_total = ∇L_batch1 + ∇L_batch2 + ... + ∇L_batchN
```

But with per-group normalization, each gradient uses different scales, breaking this equivalence.

#### 4. **Loss of Relative Importance**

Normalization erases meaningful differences:

```python
# Trajectory group A: Successful trades (advantages: +50 to +100)
# Trajectory group B: Failed trades (advantages: -20 to -10)

# With group-level normalization:
# Both normalized to mean=0, std=1 → treated as equally important

# With global normalization:
# Group A: High positive normalized advantages
# Group B: Negative normalized advantages
# Algorithm correctly emphasizes group A
```

### Benefits of Global Normalization

✅ **Consistent learning signal** across all samples
✅ **Correct gradient accumulation** behavior
✅ **Preserved relative importance** of different trajectories
✅ **Alignment with PPO theory** and standard implementations
✅ **Reduced training variance** from arbitrary grouping decisions

## Alignment with Standard Implementations

### OpenAI Baselines PPO2

```python
# After computing advantages
atarg = (atarg - atarg.mean()) / atarg.std()
```

### Stable-Baselines3 PPO

```python
class RolloutBuffer:
    def normalize_advantages(self):
        mean = self.advantages.mean()
        std = self.advantages.std()
        self.advantages = (self.advantages - mean) / (std + 1e-8)

# Called once after computing advantages
buffer.normalize_advantages()
```

Our implementation now follows this exact pattern.

## Testing

### Unit Tests

- `tests/test_advantage_normalization_integration.py`: Verifies global normalization
- `tests/test_advantage_normalization_simple.py`: Standalone tests (no pytest dependency)

### Test Coverage

1. **Basic normalization**: Verifies mean≈0, std≈1
2. **Relative ordering**: Confirms ordering is preserved
3. **Consistency**: Same advantages get same normalized values regardless of grouping
4. **Edge cases**: Handles constant advantages, small std
5. **Implementation verification**: Checks code structure

### Running Tests

```bash
# With pytest (if installed)
pytest tests/test_advantage_normalization_integration.py -v

# Standalone
python3 tests/test_advantage_normalization_simple.py
```

## Migration Notes

### Breaking Changes

None. The change is internal to the algorithm.

### Behavioral Changes

- Advantages are now normalized once per rollout, not per gradient accumulation group
- Training may converge differently (likely more stable)
- Logging: `train/advantages_mean_raw` and `train/advantages_std_raw` now logged in `collect_rollouts()`

### Performance Impact

- **Negligible runtime change**: Normalization moved from `O(n_groups)` to `O(1)` per epoch
- **Memory**: No change
- **Convergence**: Expected to be more stable and consistent with standard PPO

## References

- [PPO Paper](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [OpenAI Baselines PPO2](https://github.com/openai/baselines/blob/master/baselines/ppo2/ppo2.py)
- [Stable-Baselines3 PPO](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py)

## Related Issues

- Original issue: Group-level advantage normalization introduces bias
- Fix PR: Implement global advantage normalization following PPO best practices
