# AWR (Advantage Weighted Regression) Weighting in BC Loss

## Overview

Behavior Cloning (BC) loss in `DistributionalPPO` uses AWR-style weighting to emphasize high-advantage trajectories when learning from off-policy data.

**Formula**: `weight = exp(A / β)`, where:
- `A`: normalized advantage (mean=0, std=1)
- `β` (`cql_beta`): temperature parameter controlling weight sharpness

## Parameters

### `cql_beta` (default: 5.0)
Temperature parameter that controls how sharply advantages are weighted.

- **Higher β** (e.g., 5.0-10.0): More conservative, "soft" weights close to 1.0
- **Lower β** (e.g., 0.5-2.0): More aggressive, sharper distinction between good/bad trajectories

**Default β=5.0** creates very conservative weights:
- 50th percentile (A=0σ): weight = 1.00
- 95th percentile (A=2σ): weight = 1.49
- 99.7th percentile (A=3σ): weight = 1.82

### Weight Clipping (max_weight: 100.0)
Prevents extreme advantages from dominating training.

**CRITICAL IMPLEMENTATION DETAIL**:
```python
# ✓ CORRECT: Clamp exp_arg BEFORE exp
exp_arg = clamp(A / β, max=log(max_weight))
weights = exp(exp_arg)

# ✗ INCORRECT: Clamp twice (buggy)
exp_arg = clamp(A / β, max=20)  # exp(20) ≈ 485M
weights = exp(exp_arg)
weights = clamp(weights, max=100)  # First clamp was useless!
```

**Why this matters**:
- `exp(20) ≈ 485,000,000 >> max_weight=100`
- First clamp becomes useless, wasting computation
- `log(100) ≈ 4.605` is the mathematically correct threshold

## Weight Distribution

With normalized advantages (mean=0, std=1) and β=5.0:

| Percentile | Advantage | Weight | Interpretation |
|------------|-----------|--------|----------------|
| 16th       | -1σ       | 0.82   | Below average  |
| 50th       | 0σ        | 1.00   | Median         |
| 84th       | +1σ       | 1.22   | Above average  |
| 95th       | +2σ       | 1.49   | Good           |
| 99.7th     | +3σ       | 1.82   | Excellent      |
| 99.99th    | +4σ       | 2.23   | Exceptional    |

**max_weight=100 is triggered at ≥23σ** (statistically impossible)

## Comparison with Standard AWR

**d3rlpy (standard AWR implementation)**:
- β = 1.0 (aggressive)
- max_weight = 20.0

**Our implementation**:
- β = 5.0 (5x more conservative)
- max_weight = 100.0 (5x higher ceiling, rarely reached)

**Net effect**: Despite higher max_weight, our weights are more conservative due to high β.

## Usage Context

AWR weighting is applied when `bc_coef > 0`:

```python
if bc_coef > 0:
    with torch.no_grad():
        exp_arg = torch.clamp(advantages_selected / self.cql_beta, max=math.log(100.0))
        weights = torch.exp(exp_arg)
    policy_loss_bc = (-log_prob_selected * weights).mean()
    policy_loss_bc_weighted = policy_loss_bc * bc_coef
```

**Where**:
- `advantages_selected`: Normalized advantages (mean=0, std=1) from group normalization
- `log_prob_selected`: Log probabilities of actions in rollout buffer
- `bc_coef`: Behavior cloning coefficient (decays during training)

## When to Adjust Parameters

### Increase β (more conservative):
- Advantages are noisy or unreliable
- Want uniform weighting across most trajectories
- Concerned about overfitting to outliers

### Decrease β (more aggressive):
- High-quality advantage estimates
- Want strong differentiation between good/bad trajectories
- Sufficient data to avoid overfitting

### Adjust max_weight:
- Rarely necessary with normalized advantages
- Consider lowering (e.g., to 20.0) if very aggressive β is used
- Current 100.0 is effectively infinite with β=5.0

## References

- **Original paper**: Peng et al. 2019, "Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning"
- **Standard implementation**: d3rlpy.algos.AWR
- **Fix commit**: 354bbe8 - fix: Correct BC loss AWR-style weight clamping logic

## Testing

Comprehensive test coverage in `tests/test_distributional_ppo_awr_weighting.py`:
- Basic weight computation
- Max weight clipping
- Overflow prevention
- Correctness vs. old buggy implementation
- Edge cases (inf, nan, extreme values)
- Gradient flow
- Determinism

Run tests:
```bash
pytest tests/test_distributional_ppo_awr_weighting.py -v
```
