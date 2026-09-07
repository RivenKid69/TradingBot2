# UPGD Optimizer Integration Guide

## Overview

This document describes the integration of UPGD (Utility-based Perturbed Gradient Descent) optimizers into CustodiaCloud. UPGD is a continual learning optimizer designed to mitigate catastrophic forgetting and maintain plasticity in deep neural networks.

**IMPORTANT**: As of the latest version, **AdaptiveUPGD is now the default optimizer** for all DistributionalPPO models. This provides improved continual learning performance out of the box.

## Reference

**Paper**: Elsayed, M., & Mahmood, A. R. (2024). Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning. In Proceedings of the 12th International Conference on Learning Representations (ICLR).

**Repository**: https://github.com/mohmdelsayed/upgd

## Recent Fixes (2025-11-27)

- **Learning Rate Multiplier**: Fixed a bug in `AdaptiveUPGD.step` where the perturbed update was scaled by `-2.0 * lr` instead of `-1.0 * lr`. The implementation now correctly uses `-1.0 * lr`, matching the standard UPGD algorithm. **Status: Verified (2025-11-30).**

## Available Optimizers

### 1. UPGD (Basic)

The core UPGD optimizer with utility-based weight protection.

**Algorithm**:

1. Compute utility for each parameter: `u = -grad * param`
2. Track exponential moving average of utility
3. Find global maximum utility across all parameters
4. Scale utility using `sigmoid(utility / global_max)`
5. Apply update: `param -= lr * (grad + noise) * (1 - scaled_utility)`

**Default Parameters**:

- `lr`: 1e-5
- `weight_decay`: 0.001
- `beta_utility`: 0.999 (EMA decay for utility)
- `sigma`: 0.001 (Gaussian noise std)

**Use Case**: Basic continual learning scenarios with standard SGD-like behavior.

### 2. AdaptiveUPGD

Combines UPGD with Adam-style adaptive learning rates for improved performance in deep networks.

**Features**:

- First and second moment estimates (like Adam)
- Utility-based weight protection
- Adaptive learning rate per parameter
- Bias correction for both moments and utility

**Default Parameters**:

- `lr`: 1e-5
- `weight_decay`: 0.001
- `beta_utility`: 0.999
- `beta1`: 0.9 (first moment decay)
- `beta2`: 0.999 (second moment decay)
- `eps`: 1e-8 (numerical stability)
- `sigma`: 0.001

**Use Case**: Recommended for most deep RL applications, especially with PPO.

### 3. UPGDW

UPGD with AdamW-style decoupled weight decay for better regularization.

**Features**:

- Decoupled weight decay (applied to parameters, not gradients)
- Drop-in replacement for AdamW
- Adaptive learning rates
- Utility-based protection

**Default Parameters**:

- `lr`: 1e-4
- `betas`: (0.9, 0.999)
- `eps`: 1e-8
- `weight_decay`: 0.01 (decoupled)
- `sigma`: 0.001

**Use Case**: When you need strong regularization and want a direct AdamW replacement.

## Integration with DistributionalPPO

### Default Usage (AdaptiveUPGD)

**NEW**: AdaptiveUPGD is now the default optimizer. Simply create a model without specifying an optimizer:

```python
from distributional_ppo import DistributionalPPO

# Uses AdaptiveUPGD by default (NEW!)
model = DistributionalPPO(
    "MlpPolicy",
    env,
)

# You can customize AdaptiveUPGD parameters via optimizer_kwargs
model = DistributionalPPO(
    "MlpPolicy",
    env,
    optimizer_kwargs={
        "lr": 3e-4,
        "sigma": 0.001,
        "beta_utility": 0.999,
    },
)
```

### Switching to Other Optimizers

You can still explicitly select other optimizers if needed:

```python
# Use AdamW (previous default)
model = DistributionalPPO(
    "MlpPolicy",
    env,
    optimizer_class="adamw",
)

# Use basic UPGD
model = DistributionalPPO(
    "MlpPolicy",
    env,
    optimizer_class="upgd",
    optimizer_kwargs={"lr": 3e-4, "sigma": 0.001},
)

# Use UPGDW (AdamW-style)
model = DistributionalPPO(
    "MlpPolicy",
    env,
    optimizer_class="upgdw",
)

# Or use direct class reference
from optimizers import AdaptiveUPGD

model = DistributionalPPO(
    "MlpPolicy",
    env,
    optimizer_class=AdaptiveUPGD,
    optimizer_kwargs={"lr": 3e-4},
)
```

### Recommended Configurations

#### For Standard Training (Default AdaptiveUPGD)

```python
# Default configuration (recommended for most use cases)
model = DistributionalPPO(
    "MlpPolicy",
    env,
    # No need to specify optimizer_class - uses AdaptiveUPGD by default
    # Default parameters are already optimized for RL:
    # lr=<from learning_rate parameter>, weight_decay=0.001,
    # beta_utility=0.999, beta1=0.9, beta2=0.999, sigma=0.001
)

# Custom AdaptiveUPGD parameters
model = DistributionalPPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,          # Learning rate
    optimizer_kwargs={
        "weight_decay": 0.001,   # L2 regularization
        "beta_utility": 0.999,   # Utility EMA decay
        "beta1": 0.9,            # First moment
        "beta2": 0.999,          # Second moment
        "sigma": 0.001,          # Perturbation noise
    },
)
```

#### For Strong Regularization (UPGDW)

```python
model = DistributionalPPO(
    "MlpPolicy",
    env,
    optimizer_class="upgdw",
    optimizer_kwargs={
        "lr": 1e-4,
        "weight_decay": 0.01,    # Stronger decoupled weight decay
        "betas": (0.9, 0.999),
        "sigma": 0.001,
    },
)
```

#### For Continual Learning Tasks

```python
model = DistributionalPPO(
    "MlpPolicy",
    env,
    optimizer_class="adaptive_upgd",
    optimizer_kwargs={
        "lr": 3e-4,
        "beta_utility": 0.999,   # High EMA for stable utility estimates
        "sigma": 0.005,          # Higher noise for more exploration
        "weight_decay": 0.0001,  # Light regularization
    },
)
```

## Key Hyperparameters

### Learning Rate (`lr`)

- Controls step size
- Recommended: 1e-4 to 3e-4 for RL tasks
- Lower values for continual learning

### Beta Utility (`beta_utility`)

- EMA decay rate for utility tracking
- Range: [0.9, 0.999]
- Higher values → more stable utility estimates
- Lower values → faster adaptation to new patterns

### Sigma (`sigma`)

- Standard deviation of perturbation noise
- Range: [0.0001, 0.01]
- Higher values → more exploration, less forgetting
- Lower values → less noise, more stable updates

### Weight Decay

- L2 regularization strength
- UPGD/AdaptiveUPGD: coupled (default 0.001)
- UPGDW: decoupled (default 0.01)

## How UPGD Works

### Utility Computation

Utility measures the importance of each weight:

```
utility = -grad * param
```

- Positive utility: weight and gradient have opposite signs
- High utility weights are protected from large updates
- Low utility weights get larger updates + noise

### Protection Mechanism

```
scaled_utility = sigmoid(utility / global_max_utility)
update = (grad + noise) * (1 - scaled_utility)
```

- High utility (important weights): `scaled_utility ≈ 1` → small update
- Low utility (less important): `scaled_utility ≈ 0` → full update + noise

### Noise Injection

Gaussian noise maintains plasticity:

```
noise ~ N(0, sigma²)
```

- Applied more to low-utility weights
- Helps escape local minima
- Prevents loss of plasticity

## Performance Characteristics

### Memory Overhead

- **UPGD**: +1 tensor per parameter (avg_utility)
- **AdaptiveUPGD**: +3 tensors per parameter (utility, m1, m2)
- **UPGDW**: +3 tensors per parameter (utility, exp_avg, exp_avg_sq)

### Computational Overhead

- Two passes over parameters per step:
  1. Compute utilities, find global max
  2. Apply scaled updates
- ~1.5x slower than Adam/AdamW
- Overhead is small compared to forward/backward passes in RL

### Convergence

- May converge slightly slower initially
- Better long-term stability
- Reduced catastrophic forgetting
- Better performance in non-stationary environments

## Monitoring and Debugging

### Check Optimizer State

```python
optimizer = model.policy.optimizer

for p in model.policy.parameters():
    if p in optimizer.state:
        state = optimizer.state[p]
        print(f"Step: {state['step']}")
        print(f"Utility stats: {state['avg_utility'].min():.4f} - {state['avg_utility'].max():.4f}")
```

### Verify Optimizer Type

```python
from optimizers import AdaptiveUPGD

assert isinstance(model.policy.optimizer, AdaptiveUPGD)
```

### Log Optimizer Info

The optimizer class is automatically logged:

```python
model.logger.record("config/optimizer_class", optimizer_name)
```

## Troubleshooting

### Issue: NaN or Inf in parameters

**Solution**: Reduce learning rate or sigma

```python
optimizer_kwargs={"lr": 1e-5, "sigma": 0.0001}
```

### Issue: No learning progress

**Solution**: Increase learning rate or reduce utility EMA

```python
optimizer_kwargs={"lr": 1e-3, "beta_utility": 0.9}
```

### Issue: Catastrophic forgetting

**Solution**: Increase sigma and beta_utility

```python
optimizer_kwargs={"sigma": 0.01, "beta_utility": 0.999}
```

### Issue: Too much noise in learning

**Solution**: Decrease sigma

```python
optimizer_kwargs={"sigma": 0.0001}
```

## Testing

### Unit Tests

Run optimizer unit tests:

```bash
python -m pytest tests/test_upgd_optimizer.py -v
```

### Integration Tests

Run integration tests with DistributionalPPO:

```bash
python -m pytest tests/test_upgd_integration.py -v
```

### Quick Validation

```python
from test_upgd_quick import run_all_tests
run_all_tests()
```

## Default Optimizer Change

**BREAKING CHANGE**: The default optimizer has changed from AdamW to AdaptiveUPGD for improved continual learning performance.

### Migration Guide

#### If you want the old behavior (AdamW)

```python
# Explicitly specify AdamW
model = DistributionalPPO(
    "MlpPolicy",
    env,
    optimizer_class="adamw",
)
```

#### If you want the new behavior (AdaptiveUPGD - now default)

```python
# Simply omit optimizer_class (NEW default)
model = DistributionalPPO(
    "MlpPolicy",
    env,
    # AdaptiveUPGD used automatically
)

# Or be explicit
model = DistributionalPPO(
    "MlpPolicy",
    env,
    optimizer_class="adaptive_upgd",
)
```

#### Using UPGDW as AdamW replacement

```python
# UPGDW provides AdamW-style interface with UPGD benefits
model = DistributionalPPO(
    "MlpPolicy",
    env,
    optimizer_class="upgdw",
    optimizer_kwargs={
        "lr": 1e-4,
        "betas": (0.9, 0.999),
        "weight_decay": 0.01,
    },
)
```

## Best Practices

1. **Use the default AdaptiveUPGD**: Now enabled by default for best all-around performance in RL tasks with continual learning

2. **Tune sigma carefully**: Start with default 0.001, adjust based on task
   - More non-stationarity → higher sigma (e.g., 0.01)
   - More stability needed → lower sigma (e.g., 0.0001)

3. **Use with CVaR optimization**: UPGD complements CVaR constraints well

   ```python
   model = DistributionalPPO(
       ...,
       # optimizer_class not needed - uses AdaptiveUPGD by default
       cvar_use_constraint=True,
       cvar_limit=-1.0,
   )
   ```

4. **Monitor utility statistics**: Track utility distribution to understand weight importance

5. **Leverage continual learning**: UPGD shines in non-stationary environments like trading

6. **Fallback to AdamW if needed**: If UPGD optimizers are not available, the system automatically falls back to AdamW with a warning

## Architecture Details

### File Structure

```
ai-quant-platform/
├── optimizers/
│   ├── __init__.py           # Optimizer exports
│   ├── upgd.py               # Basic UPGD
│   ├── adaptive_upgd.py      # AdaptiveUPGD
│   └── upgdw.py              # UPGDW
├── distributional_ppo.py     # Integration point
├── tests/
│   ├── test_upgd_optimizer.py      # Unit tests
│   └── test_upgd_integration.py    # Integration tests
└── docs/
    └── UPGD_INTEGRATION.md   # This file
```

### Integration Points

1. **DistributionalPPO.**init****: Accept `optimizer_class` and `optimizer_kwargs`
2. **DistributionalPPO._get_optimizer_class**: Map string to optimizer class
3. **DistributionalPPO._get_optimizer_kwargs**: Build kwargs with defaults
4. **DistributionalPPO._setup_model**: Create optimizer instance

## Future Enhancements

Potential improvements:

1. **Layer-wise utility tracking**: Different sigma per layer
2. **Adaptive sigma**: Adjust noise based on training progress
3. **Utility visualization**: Plot utility distributions over time
4. **Utility-aware gradient clipping**: Clip based on utility
5. **Mixed precision support**: FP16 training compatibility

## References

1. Elsayed & Mahmood (2024). Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning. ICLR 2024.
2. Loshchilov & Hutter (2019). Decoupled Weight Decay Regularization. ICLR 2019.
3. Kingma & Ba (2015). Adam: A Method for Stochastic Optimization. ICLR 2015.

## Support

For issues or questions:

- Check existing tests for usage examples
- Review this documentation
- Examine optimizer source code for implementation details
- Consult original UPGD paper for theoretical background
