# Configuration Updates - Bug Fixes 2025-11-22

## Overview

This document describes new configuration parameters added as part of bug fixes on 2025-11-22.
All parameters have safe defaults and are **fully backward compatible**.

---

## PBT Configuration (Bug Fix #2)

### New Parameters

**File**: `configs/config_pbt_adversarial.yaml` (or any config with PBT section)

```yaml
pbt:
  enabled: true
  population_size: 10
  perturbation_interval: 5

  # NEW PARAMETERS (2025-11-22) - Deadlock Prevention
  min_ready_members: 2          # Minimum viable population for fallback (default: 2)
  ready_check_max_wait: 10      # Max consecutive failed checks before fallback (default: 10)

  # Existing parameters...
  ready_percentage: 0.8         # Prefer 80% members ready (existing)
  exploit_method: "truncation"
  # ... other existing parameters
```

### Parameter Details

#### `min_ready_members` (NEW)
- **Type**: `int`
- **Default**: `2`
- **Range**: `>= 2` (minimum viable population)
- **Purpose**: Fallback minimum when `ready_percentage` cannot be met due to worker crashes
- **Example**:
  - Population = 10, ready_percentage = 0.8 → requires 8 members
  - If only 3 members ready (workers crashed) → normally blocks forever
  - With `min_ready_members = 2`: after `ready_check_max_wait` failures, allows PBT with 3 members

#### `ready_check_max_wait` (NEW)
- **Type**: `int`
- **Default**: `10`
- **Range**: `>= 1`
- **Purpose**: Number of consecutive failed ready checks before activating fallback
- **Example**:
  - First 9 checks: Skip PBT (not enough ready members)
  - 10th check: Activate fallback (if `ready_count >= min_ready_members`)

### Migration

**No migration needed!** Existing configs work without changes:

```yaml
# OLD CONFIG (still works)
pbt:
  enabled: true
  population_size: 10
  ready_percentage: 0.8
  # min_ready_members and ready_check_max_wait use defaults (2, 10)
```

```yaml
# NEW CONFIG (recommended for large populations or unstable workers)
pbt:
  enabled: true
  population_size: 20
  ready_percentage: 0.8        # Prefer 16/20 members
  min_ready_members: 4         # But allow fallback with >= 4 members
  ready_check_max_wait: 15     # Wait 15 consecutive failures before fallback
```

### Monitoring

**New Metric**: `pbt/failed_ready_checks`

```python
# In training loop
stats = scheduler.get_stats()
failed_checks = stats["pbt/failed_ready_checks"]

if failed_checks > 5:
    logger.warning("PBT experiencing consecutive failed ready checks - check worker health")

if failed_checks >= ready_check_max_wait:
    logger.critical("PBT fallback will activate on next check")
```

**Tensorboard**:
- Monitor `pbt/failed_ready_checks` (should be ~0 most of the time)
- Monitor `pbt/ready_members` vs `pbt/population_size` (should be close)

---

## Quantile Critic Configuration (Bug Fix #3)

### New Parameter

**File**: Any config with distributional quantile critic

```yaml
arch_params:
  critic:
    distributional: true
    categorical: false          # Use quantile critic
    num_quantiles: 21
    huber_kappa: 1.0

    # NEW PARAMETER (2025-11-22) - Monotonicity Enforcement
    enforce_monotonicity: false   # Default: rely on quantile regression loss (recommended)
    # enforce_monotonicity: true  # Optional: explicit sorting (for CVaR-critical apps)
```

### Parameter Details

#### `enforce_monotonicity` (NEW)
- **Type**: `bool`
- **Default**: `false`
- **Purpose**: Enforce monotonicity constraint Q(τᵢ) ≤ Q(τⱼ) for τᵢ < τⱼ via sorting
- **How it works**: Applies `torch.sort(quantiles, dim=1)` in forward pass (differentiable!)
- **Trade-off**:
  - ✅ **Pro**: Guarantees valid probability distribution, improves CVaR correctness
  - ❌ **Con**: May slightly reduce expressiveness, small computational overhead

### When to Enable?

**Enable `enforce_monotonicity: true` when:**
- CVaR estimates are critical (tail risk applications)
- Early in training (network outputs may be noisy)
- High environmental stochasticity
- Small `num_quantiles` (< 21)
- Observing non-monotonic quantiles in tensorboard

**Keep default `false` when:**
- Well-established training (quantile regression loss sufficient)
- Large `num_quantiles` (> 51)
- Performance-critical applications (avoid sorting overhead)
- Standard distributional RL (Dabney et al., 2018 does NOT use explicit sorting)

### Migration

**No migration needed!** Existing configs work without changes:

```yaml
# OLD CONFIG (still works)
arch_params:
  critic:
    distributional: true
    categorical: false
    num_quantiles: 21
    huber_kappa: 1.0
    # enforce_monotonicity defaults to false
```

```yaml
# NEW CONFIG (for CVaR-critical applications)
arch_params:
  critic:
    distributional: true
    categorical: false
    num_quantiles: 21
    huber_kappa: 1.0
    enforce_monotonicity: true   # Explicit sorting
```

### Monitoring

**Check monotonicity during development:**

```python
# In debugging/development
quantiles = model.policy.quantile_head(latent)  # [batch, num_quantiles]

for i in range(quantiles.size(0)):
    # Check if monotonic
    is_monotonic = torch.all(quantiles[i, :-1] <= quantiles[i, 1:])

    if not is_monotonic:
        if not model.policy.enforce_quantile_monotonicity:
            logger.debug(f"Non-monotonic quantiles (sample {i}) - this is OK with enforce_monotonicity=False")
        else:
            logger.error(f"Non-monotonic quantiles (sample {i}) - should NOT happen with enforce_monotonicity=True!")
```

**Tensorboard visualization:**
- Plot predicted quantiles over episodes
- Check for crossing lines (non-monotonicity)
- If crossing occurs AND `enforce_monotonicity=true` → bug!

---

## SA-PPO Configuration (Bug Fix #1 - Already Fixed)

### Verification Only

**No new parameters needed!** The fix was already in place. For reference:

```yaml
adversarial:
  enabled: true
  perturbation:
    epsilon: 0.075              # Initial epsilon

  # OPTIONAL: Explicit max_updates (auto-computed if omitted)
  # max_updates: 5000           # Explicitly set (rarely needed)

  # Epsilon schedule (auto-computes max_updates from total_timesteps // n_steps)
  adaptive_epsilon: true
  epsilon_schedule: "linear"
  epsilon_final: 0.05
```

**Automatic Computation**:
```python
# Happens automatically in StateAdversarialPPO.__init__()
if total_timesteps and n_steps:
    max_updates = total_timesteps // n_steps  # e.g., 20M / 2048 = 9765
else:
    max_updates = 10000  # Fallback (NOT 1000!)
```

**Verification**:
```bash
# Verify epsilon schedule works correctly
pytest tests/test_bug_fixes_2025_11_22.py::TestSAPPOEpsilonSchedule -v
```

---

## Configuration Examples

### Minimal Config (Uses All Defaults)

```yaml
# Standard training - no changes needed
model:
  algo: "ppo"
  optimizer_class: AdaptiveUPGD

  vgs:
    enabled: true

  params:
    use_twin_critics: true  # Default

arch_params:
  critic:
    distributional: true
    categorical: false
    num_quantiles: 21
    # enforce_monotonicity defaults to false

# PBT defaults: min_ready_members=2, ready_check_max_wait=10
pbt:
  enabled: false  # Not using PBT
```

### Production Config (Recommended)

```yaml
# Production with PBT + Adversarial
model:
  algo: "ppo"
  optimizer_class: AdaptiveUPGD

  vgs:
    enabled: true

  params:
    use_twin_critics: true
    num_quantiles: 21
    cvar_alpha: 0.05

arch_params:
  critic:
    distributional: true
    categorical: false
    num_quantiles: 21
    huber_kappa: 1.0
    enforce_monotonicity: false  # Standard (rely on quantile regression loss)

pbt:
  enabled: true
  population_size: 10
  ready_percentage: 0.8         # Prefer 80%
  min_ready_members: 2          # Fallback to >= 2 (default)
  ready_check_max_wait: 10      # Fallback after 10 failures (default)

adversarial:
  enabled: true
  adaptive_epsilon: true
  epsilon_schedule: "linear"
  epsilon_final: 0.05
  # max_updates auto-computed from total_timesteps
```

### CVaR-Critical Config

```yaml
# CVaR-critical application (tail risk focus)
arch_params:
  critic:
    distributional: true
    categorical: false
    num_quantiles: 51            # Higher resolution
    huber_kappa: 1.0
    enforce_monotonicity: true   # ✅ ENABLE for guaranteed monotonicity

model:
  params:
    cvar_alpha: 0.01             # Focus on worst 1% tail
    cvar_weight: 0.25            # High weight on CVaR loss
```

### Large Population PBT Config

```yaml
# Large population with higher fallback threshold
pbt:
  enabled: true
  population_size: 20
  ready_percentage: 0.8         # Prefer 16/20 members
  min_ready_members: 4          # Fallback to >= 4 members (20% population)
  ready_check_max_wait: 15      # Wait longer before fallback
```

---

## Validation

### Pre-Production Checklist

```bash
# 1. Verify all bug fix tests pass
pytest tests/test_bug_fixes_2025_11_22.py -v
# Expected: 14/14 passed ✅

# 2. Verify backward compatibility (no config changes needed)
pytest tests/ -v
# All existing tests should still pass

# 3. Verify PBT monitoring (if using PBT)
# Check tensorboard: pbt/failed_ready_checks should be ~0

# 4. Verify quantile monotonicity (if enforce_monotonicity=true)
# Check tensorboard: plot quantiles, no crossing lines
```

### Regression Prevention

**Before modifying related code:**
- [ ] Read [REGRESSION_PREVENTION_CHECKLIST_2025_11_22.md](REGRESSION_PREVENTION_CHECKLIST_2025_11_22.md)
- [ ] Run `pytest tests/test_bug_fixes_2025_11_22.py -v`
- [ ] Verify config parameters still present and defaults correct

---

## References

- **Bug Fixes Report**: [BUG_FIXES_REPORT_2025_11_22.md](../BUG_FIXES_REPORT_2025_11_22.md)
- **Regression Prevention**: [REGRESSION_PREVENTION_CHECKLIST_2025_11_22.md](../REGRESSION_PREVENTION_CHECKLIST_2025_11_22.md)
- **Main Documentation**: [CLAUDE.md](../CLAUDE.md)
- **Test Suite**: [tests/test_bug_fixes_2025_11_22.py](../tests/test_bug_fixes_2025_11_22.py)

---

**Last Updated**: 2025-11-22
**Status**: ✅ Production Ready
**Backward Compatibility**: ✅ Fully Maintained
