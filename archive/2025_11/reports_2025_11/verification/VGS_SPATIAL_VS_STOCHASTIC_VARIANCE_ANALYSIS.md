# VGS Spatial vs. Stochastic Variance - Detailed Analysis

**Date:** 2025-11-21
**Component:** `variance_gradient_scaler.py`
**Severity:** MEDIUM (Conceptual error affecting stability mechanism)
**Status:** CONFIRMED - FIX REQUIRED

---

## Executive Summary

The current `VarianceGradientScaler` implementation computes **spatial variance** (variance across all parameters at a single timestep), but for gradient scaling and stability, it should compute **stochastic variance** (variance of gradient estimates for each parameter over time). This is a fundamental conceptual error that undermines the VGS mechanism.

---

## Problem Description

### Current Implementation (Spatial Variance)

```python
# variance_gradient_scaler.py:143-174
for param in self._parameters:
    grad = param.grad.data
    grad_values.append(grad.abs().flatten())  # Collect all gradient values

all_grads = torch.cat(grad_values)  # Concatenate ALL gradients into one vector
grad_var = all_grads.var().item()   # SPATIAL VARIANCE - across all parameters
```

**What this measures:**
- Variance of gradient magnitudes **across all parameters** at one timestep
- Heterogeneity of gradient scales across different layers/weights
- Formula: `Var(|g₁|, |g₂|, ..., |gₙ|)` where `gᵢ` are all parameter gradients

### Correct Approach (Stochastic Variance)

**What adaptive optimizers (Adam, RMSprop) use:**
```python
# Adam's second moment (per-parameter)
v_t[i] = beta2 * v_t[i] + (1 - beta2) * g_t[i]²
```

**What this measures:**
- Variance of gradient estimate **for each parameter** over time
- Noise/uncertainty in gradient estimation (stochastic variance)
- Formula: `Var_t(gᵢ)` for each parameter `i` over time `t`

---

## Why This Is a Problem

### 1. **Incorrect Variance Metric**

**Spatial variance** measures:
- ❌ How different are gradient scales across layers?
- ❌ Does early layer have smaller gradients than late layer?

**Stochastic variance** measures:
- ✅ How noisy is the gradient estimate for each parameter?
- ✅ How reliable is the gradient signal over mini-batches?

### 2. **Penalizes Natural Gradient Heterogeneity**

Deep networks naturally have different gradient scales across layers:
- Early layers: often smaller gradients (vanishing gradient)
- Late layers: often larger gradients
- Different parameter types: weights vs biases

**Current VGS incorrectly penalizes this natural heterogeneity**, even when gradients are stable.

### 3. **Misses Actual Stochastic Noise**

The true problem VGS should address:
- High variance in mini-batch gradient estimates (noisy gradients)
- Unstable gradient signals that hurt convergence

**Current VGS doesn't measure this noise at all!**

### 4. **Conflicts with Adaptive Optimizers**

Adam/RMSprop/UPGD already use per-parameter second moments (stochastic variance) to scale learning rates.

**Current VGS with spatial variance:**
- Uses a **global scaling factor** based on cross-parameter heterogeneity
- Can conflict with optimizer's per-parameter adaptation
- Applies same scaling to all parameters, regardless of their individual noise

---

## Research Evidence

### Adam Optimizer (Kingma & Ba, 2015)

Adam uses **per-parameter temporal variance**:
- `v_t[i] = β₂ * v_{t-1}[i] + (1 - β₂) * g_t[i]²`
- Tracks variance **for each parameter** over time
- Enables **per-parameter adaptive learning rates**

### "A Study of Gradient Variance in Deep Learning" (Faghri et al., 2020)

The paper cited in VGS documentation studies:
- **Stochastic variance** of mini-batch gradients
- How gradient variance **increases during training**
- Normalized gradient variance: `Var[g] / (E[g]² + ε)`

**Key finding:** Smaller learning rates correlate with **higher stochastic variance**, not spatial variance.

---

## Impact Assessment

### Current Behavior

1. **Models with heterogeneous gradients are penalized**
   - Even if gradients are stable (low stochastic variance)
   - Natural architectural differences (early vs late layers) trigger scaling

2. **True gradient noise is not measured**
   - High mini-batch variance (actual problem) is invisible
   - VGS cannot stabilize what it cannot see

3. **Suboptimal interaction with UPGD**
   - UPGD uses per-parameter utility-based protection
   - VGS applies global scaling based on wrong metric
   - Potential cancellation of benefits

### Expected Impact After Fix

1. **Correct noise measurement**
   - Per-parameter stochastic variance tracks mini-batch noise
   - VGS can properly stabilize noisy parameters

2. **No false positives**
   - Natural gradient heterogeneity is not penalized
   - Only true stochastic noise triggers scaling

3. **Better integration with UPGD**
   - Both use per-parameter metrics
   - Complementary mechanisms: UPGD protects important weights, VGS stabilizes noisy ones

---

## Proposed Solution

### Design: Per-Parameter Stochastic Variance

Track variance **for each parameter** over time using EMA:

```python
# For each parameter i:
grad_mean_ema[i] = beta * grad_mean_ema[i] + (1 - beta) * |g_t[i]|
grad_sq_ema[i] = beta * grad_sq_ema[i] + (1 - beta) * g_t[i]²
grad_var[i] = grad_sq_ema[i] - grad_mean_ema[i]²  # Variance = E[g²] - E[g]²
```

### Scaling Strategy Options

**Option 1: Per-Parameter Scaling (like Adam)**
```python
scaling_factor[i] = 1.0 / (1.0 + alpha * normalized_var[i])
param.grad[i] *= scaling_factor[i]
```

**Option 2: Global Scaling with Per-Parameter Aggregation**
```python
# Aggregate per-parameter variances
global_var_metric = percentile(normalized_var, 90)  # or mean, median
global_scaling = 1.0 / (1.0 + alpha * global_var_metric)
# Apply uniform scaling (current behavior, but correct metric)
for param in parameters:
    param.grad *= global_scaling
```

**Recommendation:** Start with **Option 2** to maintain backward compatibility (global scaling), but use correct metric.

---

## Implementation Plan

### Phase 1: Add Per-Parameter Stochastic Variance Tracking

1. Add `_param_mean_ema` and `_param_sq_ema` dicts
2. Compute per-parameter EMA of first and second moments
3. Derive variance: `Var[g] = E[g²] - E[g]²`

### Phase 2: Update Scaling Computation

1. Compute per-parameter normalized variance
2. Aggregate to global metric (e.g., 90th percentile)
3. Compute global scaling factor (backward compatible)

### Phase 3: Optional - Add Per-Parameter Scaling Mode

1. Add config flag: `per_parameter_scaling: bool`
2. Implement per-parameter scaling (like Adam)
3. Document trade-offs

### Phase 4: Tests

1. **Unit tests**: Verify stochastic variance computation
2. **Integration tests**: Verify with DistributionalPPO + UPGD
3. **Regression tests**: Compare old vs new behavior
4. **Ablation study**: Measure impact on training stability

---

## Migration Strategy

### Backward Compatibility

**Current models trained with spatial variance VGS:**
- Checkpoints contain `_grad_var_ema` (spatial variance)
- New VGS will use different metric (stochastic variance)

**Options:**
1. **Hard cutover** - New models use stochastic variance, old checkpoints incompatible
2. **Soft migration** - Detect old checkpoints, reset VGS statistics
3. **Parallel tracking** - Track both metrics temporarily for comparison

**Recommendation:** **Option 2 (Soft migration)**
- Detect old VGS state_dict format
- Reset statistics when loading old checkpoint
- Log warning about VGS reset
- Document recommendation to retrain models

### Documentation Updates

1. Update `variance_gradient_scaler.py` docstrings
2. Update `CLAUDE.md` VGS section
3. Add migration guide
4. Update tests documentation

---

## Success Metrics

### Correctness

- ✅ Per-parameter stochastic variance correctly computed
- ✅ Tracks mini-batch gradient noise (not spatial heterogeneity)
- ✅ All tests pass

### Performance

- ✅ Training stability improved (lower variance in value loss)
- ✅ Better interaction with UPGD (no cancellation)
- ✅ No degradation in final performance

### Compatibility

- ✅ Backward compatible checkpoint loading (with reset)
- ✅ Existing configs work with new implementation
- ✅ Documentation updated

---

## Conclusion

**Problem Confirmed:** Current VGS uses wrong variance metric (spatial instead of stochastic).

**Impact:** MEDIUM - VGS mechanism is conceptually incorrect, but system is still functional. Fix will improve training stability and interaction with adaptive optimizers.

**Recommendation:** Implement per-parameter stochastic variance tracking with global aggregation (Option 2) for backward compatibility.

**Priority:** HIGH - Should be fixed before next major training runs to ensure optimal UPGD + VGS interaction.

---

## References

1. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. ICLR.
2. Faghri, F., & Duvenaud, D. (2020). A Study of Gradient Variance in Deep Learning. arXiv:2007.04532.
3. Tieleman, T., & Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural networks for machine learning.

---

**Next Steps:**
1. Review and approve this analysis
2. Implement per-parameter stochastic variance tracking
3. Write comprehensive tests
4. Update documentation
5. Validate on training runs
