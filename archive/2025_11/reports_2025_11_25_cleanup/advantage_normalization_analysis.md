# Advantage Normalization Analysis and Fix

## Problem Statement

The current implementation uses **group-level advantage normalization** during training, where each gradient accumulation group computes its own mean/std statistics. This deviates from standard PPO practice and introduces several issues.

## Current Implementation

**Location:** `distributional_ppo.py:7712-7829`

```python
# For each microbatch GROUP during training:
group_advantages_concat = torch.cat(group_advantages_for_stats, dim=0)
group_adv_mean = group_advantages_concat.mean()
group_adv_std = group_advantages_concat.std(unbiased=False)

# Apply to each microbatch in the group:
advantages_normalized = (advantages - group_adv_mean) / group_adv_std_clamped
```

## Standard PPO Practice

**Reference implementations:** OpenAI Baselines, Stable-Baselines3

```python
# ONCE after computing all advantages, BEFORE training:
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# Then use these pre-normalized advantages for all epochs and batches
```

## Identified Issues

### 1. Inconsistent Normalization Across Groups

**Problem:** Same raw advantage values get different normalized values depending on which group they're in.

**Example:**
- Group A: advantages = [100, 110, 120] → mean=110, std=8.16
- Group B: advantages = [-5, -6, -5.5] → mean=-5.5, std=0.43
- Raw advantage = 5:
  - In Group A: (5 - 110) / 8.16 = **-12.87**
  - In Group B: (5 - (-5.5)) / 0.43 = **+24.42**

**Impact:** Same action quality receives vastly different policy updates depending on group membership.

### 2. Bias with Unbalanced Groups

**Problem:** Groups with different sizes or distributions bias the statistics.

**Example:**
- Group 1: 1000 samples, high advantages (mean=50, std=10)
- Group 2: 10 samples, low advantages (mean=-5, std=2)

Group 2's normalization is dominated by its small sample size, leading to:
- High variance in normalized values
- Unreliable gradient signals
- Potential training instability

### 3. Gradient Accumulation Issues

**Problem:** Gradients from different groups have different scales, violating proper gradient accumulation.

**Theory:** Gradient accumulation assumes:
```
∇L_total = ∇L_batch1 + ∇L_batch2 + ... + ∇L_batchN
```

But with per-group normalization:
```
∇L_batch1 uses advantages normalized by μ1, σ1
∇L_batch2 uses advantages normalized by μ2, σ2
```

This breaks the mathematical equivalence between:
1. Training on full batch
2. Accumulating gradients from sub-batches

**Result:** Training behavior changes based on how data is grouped, which is arbitrary.

### 4. Loss of Relative Importance

**Problem:** Normalization erases meaningful differences between trajectory groups.

**Example:**
- Trajectory group A: Successful trades (advantages: +50 to +100)
- Trajectory group B: Failed trades (advantages: -20 to -10)

**With group-level normalization:**
- Both groups normalized to mean=0, std=1
- Algorithm treats them as equally important
- Loses signal that group A is genuinely better

**With global normalization:**
- Group A: High positive normalized advantages
- Group B: Negative normalized advantages
- Algorithm correctly emphasizes group A

### 5. Inconsistency Across Epochs

**Problem:** With data shuffling between epochs, same advantages get different normalized values.

**Example:**
- Epoch 1: Sample X in group with mean=10 → normalized as -0.5
- Epoch 2: Sample X in group with mean=5 → normalized as +0.5

**Result:** Inconsistent learning signal across epochs.

## Theoretical Justification for Global Normalization

### PPO Objective

PPO maximizes:
```
L^CLIP(θ) = E_t[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
```

Where `A_t` is the advantage and `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` is the probability ratio.

### Why Normalize Advantages?

1. **Numerical stability:** Prevents extreme advantage values from causing large policy updates
2. **Consistent learning rate:** Keeps advantages in similar range across different environments/tasks
3. **Variance reduction:** Reduces gradient variance

### Why Global Normalization?

The normalization `(A - μ) / σ` is a **linear transformation** that:
1. Centers advantages around 0
2. Scales to unit variance
3. **Preserves relative ordering** of advantages

**Key insight:** We want to preserve relative importance ACROSS ALL SAMPLES in the batch, not just within arbitrary sub-groups.

## Correct Implementation

### Approach 1: Normalize in RolloutBuffer (Recommended)

```python
class RolloutBuffer:
    def normalize_advantages(self):
        """Normalize advantages globally across entire buffer."""
        if self.advantages is None:
            return

        # Flatten to handle (buffer_size, n_envs) shape
        adv_flat = self.advantages.reshape(-1)

        # Global statistics
        mean = adv_flat.mean()
        std = adv_flat.std()
        std_clamped = max(std, 1e-8)

        # Normalize in-place
        self.advantages = (self.advantages - mean) / std_clamped
```

Call once after `compute_returns_and_advantage()`, before training.

### Approach 2: Normalize in train() (Alternative)

```python
def train(self):
    # At the START of train(), before any epochs:
    # Collect all advantages
    all_advantages = []
    for rollout_data in self.rollout_buffer.get(batch_size=None):
        all_advantages.append(rollout_data.advantages)

    # Compute global statistics
    adv_concat = torch.cat(all_advantages)
    adv_mean = adv_concat.mean()
    adv_std = adv_concat.std()

    # Normalize buffer in-place
    self.rollout_buffer.advantages = (
        (self.rollout_buffer.advantages - adv_mean) / (adv_std + 1e-8)
    )

    # Then proceed with normal training epochs
    for epoch in range(self.n_epochs):
        for rollout_data in self.rollout_buffer.get(...):
            # Use already-normalized advantages
```

## Implementation Plan

1. **Add `normalize_advantages()` method** after `_compute_returns_with_time_limits()`
2. **Remove group-level normalization** from training loop
3. **Update tests** to verify global normalization
4. **Verify no performance regression** on existing tasks

## Expected Impact

✅ **Consistent learning signal** across all samples
✅ **Correct gradient accumulation** behavior
✅ **Preserved relative importance** of different trajectories
✅ **Alignment with PPO theory** and standard implementations
✅ **Reduced training variance** from arbitrary grouping decisions

## References

- [OpenAI Baselines PPO2](https://github.com/openai/baselines/blob/master/baselines/ppo2/ppo2.py#L112)
- [Stable-Baselines3 PPO](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py#L547)
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
