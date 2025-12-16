# Distributional VF Clipping in PPO

## Overview

Value Function (VF) clipping is a key component of PPO that prevents large value function updates. The original PPO paper designed VF clipping for **scalar value functions**:

```
V_clipped = clip(V, old_V - ε, old_V + ε)
L_VF = max((V - V_target)², (V_clipped - V_target)²)
```

However, for **distributional critics** (quantile or categorical), which predict entire distributions rather than scalar values, VF clipping requires careful adaptation.

## The Problem

**Naive Approach (mean_only mode):** Clip only the mean of the distribution and shift the entire distribution:

```python
mean_clipped = clip(mean, old_mean - ε, old_mean + ε)
delta = mean_clipped - mean
quantiles_clipped = quantiles + delta  # Parallel shift
```

**Problem:** Individual quantiles can still violate the clipping bounds!

### Example

```
old_value = 10, ε = 5
Bounds: [5, 15]

New quantiles: [5, 20, 35], mean = 20
After clipping mean to 15: [0, 15, 30]  (shift by -5)

VIOLATION: Quantiles [0, 30] are outside [5, 15]!
```

This is problematic for **CVaR-based risk-sensitive RL**, where tail quantiles (representing risk) must be properly constrained.

## Available Modes

This implementation provides **four modes** to handle distributional VF clipping:

### 1. `None` or `"disable"` (Default, Recommended)

**Behavior:** No VF clipping for distributional critics.

**Rationale:**
- VF clipping for distributional critics lacks theoretical foundation
- The original PPO paper only considers scalar value functions
- Distributional critics are inherently more stable due to richer training signal

**Use when:** You want the safest, most conservative option with no surprises.

```python
model = DistributionalPPO(
    ...,
    clip_range_vf=0.5,  # This will be IGNORED for distributional critics
    distributional_vf_clip_mode=None  # or "disable"
)
```

### 2. `"mean_only"` (Legacy, Not Recommended)

**Behavior:** Clip the mean value, then parallel-shift the entire distribution.

**Algorithm:**
```python
# 1. Clip mean in raw space
mean_clipped = clip(mean, old_mean - ε, old_mean + ε)

# 2. Shift all quantiles by the same delta
delta = mean_clipped - mean
quantiles_clipped = quantiles + delta  # Parallel shift
```

**Limitations:**
- ❌ Individual quantiles can exceed clipping bounds
- ❌ Variance is NOT constrained (distribution can still widen arbitrarily)
- ❌ Tail quantiles (CVaR) can violate bounds

**Use when:** For backward compatibility with older code only.

### 3. `"mean_and_variance"` (Improved, Better than mean_only)

**Behavior:** Clip the mean AND constrain variance growth.

**Algorithm:**
```python
# 1. Clip mean via parallel shift
mean_clipped = clip(mean, old_mean - ε, old_mean + ε)
delta = mean_clipped - mean
quantiles_shifted = quantiles + delta

# 2. Constrain variance
current_variance = var(quantiles_shifted)
old_variance = var(old_quantiles)

# Limit variance ratio to factor² (default: 4x variance)
variance_scale = sqrt(clip(current_variance / old_variance, max=factor²))

# 3. Scale quantiles toward mean if variance too large
quantiles_clipped = mean_clipped + (quantiles_shifted - mean_clipped) * variance_scale
```

**Improvements over mean_only:**
- ✅ Constrains distribution spread (variance)
- ✅ Reduces but doesn't eliminate tail violations
- ✅ Better for risk-sensitive RL than mean_only

**Limitations:**
- ⚠️ Individual quantiles can still exceed bounds (less likely than mean_only)
- ⚠️ Variance constraint is approximate, not strict bounds

**Use when:** You want VF clipping with some variance control, but not strict guarantees.

**Configuration:**
```python
model = DistributionalPPO(
    ...,
    clip_range_vf=0.5,
    distributional_vf_clip_mode="mean_and_variance",
    distributional_vf_clip_variance_factor=2.0  # Allow up to 4x variance growth
)
```

### 4. `"per_quantile"` (Strictest, Closest to Original PPO)

**Behavior:** Clip EACH quantile individually relative to old_value.

**Algorithm:**
```python
# Clip each quantile relative to old_value (not old quantile!)
# This matches the scalar PPO formula most closely
for each quantile q:
    q_clipped = old_value + clip(q - old_value, -ε, +ε)
```

**Guarantees:**
- ✅ ALL quantiles guaranteed within [old_value - ε, old_value + ε]
- ✅ Tail quantiles (CVaR) properly constrained
- ✅ Closest semantic match to scalar VF clipping
- ✅ Most predictable behavior

**Trade-offs:**
- ⚠️ Most aggressive clipping (can reduce learning signal)
- ⚠️ May collapse distribution toward old_value (reduces distributional information)
- ⚠️ Extreme distributions get heavily clipped

**Use when:**
- You need strict bounds guarantees on ALL quantiles
- CVaR-based risk-sensitive RL where tail control is critical
- Debugging value function instability
- You want the most faithful adaptation of scalar VF clipping

**Configuration:**
```python
model = DistributionalPPO(
    ...,
    clip_range_vf=0.5,
    distributional_vf_clip_mode="per_quantile"
)
```

## Comparison Table

| Mode | Mean Clipped | Variance Constrained | All Quantiles Bounded | Complexity |
|------|--------------|---------------------|----------------------|------------|
| `None` / `disable` | ❌ | ❌ | ❌ | Lowest |
| `mean_only` | ✅ | ❌ | ❌ | Low |
| `mean_and_variance` | ✅ | ✅ | ⚠️ (approximately) | Medium |
| `per_quantile` | ✅ | ✅ | ✅ (guaranteed) | Low |

## Categorical Critic Differences

For categorical critics (C51-style), the implementation differs slightly:

- **Quantile critic:** Directly clips quantile values
- **Categorical critic:** Clips atom support ranges, then projects probabilities

For `per_quantile` mode with categorical critics:
```python
# Clip atoms (not probabilities) for each sample
for each sample:
    atoms_clipped = old_value + clip(atoms - old_value, -ε, +ε)
    probs_clipped = project(probs, atoms_clipped -> original_atoms)
```

This ensures the probability mass stays within the clipped atom range.

## Recommendations

### For Most Users
**Use `None` (disable)** - Distributional critics are inherently stable, VF clipping may not be needed.

### For Experimentation
**Use `mean_and_variance`** - Provides a good balance between constraint and flexibility.

### For Risk-Sensitive RL (CVaR-based)
**Use `per_quantile`** - Guarantees tail quantiles respect bounds, critical for CVaR training.

### For Debugging Value Instability
**Try `per_quantile` first**, then relax to `mean_and_variance` if too restrictive.

### Never Use
**`mean_only`** - Only exists for backward compatibility. Use `mean_and_variance` or `per_quantile` instead.

## Implementation Example

```python
from distributional_ppo import DistributionalPPO

# Conservative (recommended for most)
model = DistributionalPPO(
    policy="MlpLstmPolicy",
    env=env,
    clip_range_vf=0.5,
    distributional_vf_clip_mode=None  # Disabled
)

# Balanced
model = DistributionalPPO(
    policy="MlpLstmPolicy",
    env=env,
    clip_range_vf=0.5,
    distributional_vf_clip_mode="mean_and_variance",
    distributional_vf_clip_variance_factor=2.0  # Allow up to 4x variance
)

# Strictest (CVaR training)
model = DistributionalPPO(
    policy="MlpLstmPolicy",
    env=env,
    clip_range_vf=0.5,
    distributional_vf_clip_mode="per_quantile"
)
```

## Testing

See `tests/test_distributional_vf_clip_modes.py` for comprehensive tests demonstrating:
- Bounds violations in `mean_only` mode
- Variance constraint in `mean_and_variance` mode
- Guaranteed bounds in `per_quantile` mode
- Edge cases and batch handling

Run tests:
```bash
pytest tests/test_distributional_vf_clip_modes.py -v
pytest tests/test_per_quantile_vf_clip.py -v
```

## References

- Original PPO paper: Schulman et al. (2017)
- C51 (Categorical): Bellemare et al. (2017)
- QR-DQN (Quantile): Dabney et al. (2018)
- CVaR-PPO: Risk-sensitive reinforcement learning

## See Also

- `distributional_ppo.py` lines 8769-8915 (quantile critic implementation)
- `distributional_ppo.py` lines 9034-9191 (categorical critic implementation)
- `test_vf_clip_quantile_bounds.py` (demonstration of the problem and solution)
