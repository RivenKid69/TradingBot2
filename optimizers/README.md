# UPGD Optimizers

Utility-based Perturbed Gradient Descent (UPGD) optimizers for continual learning in deep neural networks.

## Quick Start

```python
from optimizers import UPGD, AdaptiveUPGD, UPGDW
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 5),
)

# Basic UPGD
optimizer = UPGD(model.parameters(), lr=1e-4)

# Or AdaptiveUPGD (recommended)
optimizer = AdaptiveUPGD(model.parameters(), lr=3e-4)

# Or UPGDW (AdamW replacement)
optimizer = UPGDW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(batch.x), batch.y)
    loss.backward()
    optimizer.step()
```

## Optimizers

### UPGD

Basic utility-based optimizer with perturbation.

**Parameters**:

- `lr` (float): Learning rate (default: 1e-5)
- `weight_decay` (float): L2 penalty (default: 0.001)
- `beta_utility` (float): EMA decay for utility (default: 0.999)
- `sigma` (float): Noise std (default: 0.001)

### AdaptiveUPGD

UPGD with Adam-style adaptive learning rates.

**Parameters**:

- `lr` (float): Learning rate (default: 1e-5)
- `weight_decay` (float): L2 penalty (default: 0.001)
- `beta_utility` (float): Utility EMA decay (default: 0.999)
- `beta1` (float): First moment decay (default: 0.9)
- `beta2` (float): Second moment decay (default: 0.999)
- `eps` (float): Numerical stability (default: 1e-8)
- `sigma` (float): Noise std (default: 0.001)

### UPGDW

UPGD with decoupled weight decay (AdamW-style).

**Parameters**:

- `lr` (float): Learning rate (default: 1e-4)
- `betas` (Tuple[float, float]): Moment decay rates (default: (0.9, 0.999))
- `eps` (float): Numerical stability (default: 1e-8)
- `weight_decay` (float): Decoupled L2 penalty (default: 0.01)
- `sigma` (float): Noise std (default: 0.001)

## Features

- **Catastrophic forgetting mitigation**: Protects important weights from large updates
- **Plasticity maintenance**: Noise injection prevents loss of learning capability
- **Adaptive learning**: Combines with Adam for better performance
- **Decoupled regularization**: UPGDW offers AdamW-style weight decay

## How It Works

1. **Utility Computation**: `utility = -grad * param`
2. **EMA Tracking**: Smooth utility estimates over time
3. **Protection**: High-utility weights get smaller updates
4. **Perturbation**: Low-utility weights get noise for exploration

## Reference

Elsayed, M., & Mahmood, A. R. (2024). Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning. ICLR 2024.

## See Also

- Full documentation: `docs/UPGD_INTEGRATION.md`
- Unit tests: `tests/test_upgd_optimizer.py`
- Integration tests: `tests/test_upgd_integration.py`
