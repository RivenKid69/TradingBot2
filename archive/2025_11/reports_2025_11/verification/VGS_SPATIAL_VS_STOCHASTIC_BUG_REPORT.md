# VGS Spatial vs Stochastic Variance Bug Report

**Date**: 2025-11-23
**Severity**: CRITICAL
**Component**: `variance_gradient_scaler.py`
**Status**: CONFIRMED BUG

---

## Executive Summary

The Variance Gradient Scaler (VGS) claims to compute "per-parameter stochastic variance" (variance of gradient estimates over time) but actually computes **spatial variance** (variance across parameter elements at each timestep). This fundamentally breaks the theoretical foundation of the algorithm and may explain training instabilities.

---

## Bug Description

### What the code CLAIMS to do (from documentation):

```python
"""
Implements adaptive gradient scaling based on **per-parameter stochastic variance**.

**v2.0 (2025-11-21): Fixed to use per-parameter stochastic variance instead
of spatial variance.**

Algorithm:
    1. For each parameter, track EMA of first moment (mean) and second moment (mean of squares)
    2. Compute per-parameter variance: Var[g] = E[g²] - E[g]²
    3. Compute per-parameter normalized variance: Var[g] / (E[g]² + ε)
```

###What the code ACTUALLY does:

```python
# variance_gradient_scaler.py:268-289
def update_statistics(self) -> None:
    for i, param in enumerate(self._parameters):
        if param.grad is None:
            continue

        grad = param.grad.data
        # PROBLEM: This computes SPATIAL variance (across elements at ONE timestep)
        grad_abs_mean = grad.abs().mean().item()
        grad_variance = grad.var(unbiased=False).item()  # <- SPATIAL VARIANCE!

        # Update EMA
        self._param_grad_mean_ema[i] = (
            self.beta * self._param_grad_mean_ema[i] +
            (1 - self.beta) * grad_abs_mean
        )
        self._param_grad_sq_ema[i] = (
            self.beta * self._param_grad_sq_ema[i] +
            (1 - self.beta) * grad_variance  # <- EMA of SPATIAL variance!
        )
```

**The Critical Error:**
- `grad.var()` computes variance **across all elements of the gradient tensor** at the **current timestep**
- This is **spatial variance**: `Var_spatial[g] = E_j[(g_j - E_j[g_j])²]` where `j` indexes elements
- **NOT** stochastic variance: `Var_stochastic[g] = E_t[(g_t - E_t[g_t])²]` where `t` indexes time

Then the code tracks EMA of this spatial variance, which is:
```
EMA_of_spatial_variance = β * EMA_old + (1-β) * spatial_variance_current
```

This is **completely different** from true stochastic variance!

---

## Proof of Bug

### Test Case 1: Uniform Constant Gradients
```python
# Apply SAME uniform gradient at every timestep
for step in range(30):
    param.grad = torch.ones(100) * 1.0  # No spatial OR temporal variation
    vgs.step()

variance = vgs.get_normalized_variance()
# Result: 0.000000
```

**Analysis:**
- **Spatial variance**: `torch.var(ones(100)) = 0` ✓ Correct
- **Stochastic variance**: `Var[1.0, 1.0, ..., 1.0] = 0` ✓ Correct
- **Both give same result**, so this test is inconclusive

### Test Case 2: Uniform NOISY Gradients (**SMOKING GUN**)
```python
# Apply uniform gradient with TEMPORAL noise
for step in range(30):
    noise_value = torch.randn(1).item() * 2.0 + 1.0  # Changes over time!
    param.grad = torch.ones(100) * noise_value  # All elements SAME, but changes over time
    vgs.step()

variance = vgs.get_normalized_variance()
# Result: 0.000000  <- BUG DETECTED!
```

**Analysis:**
- **Spatial variance at each timestep**: `torch.var(ones(100) * noise) = 0` (all elements identical)
- **EMA of spatial variance**: `0`
- **TRUE stochastic variance**: Should be `Var[noise_sequence] >> 0` (large temporal variation!)

**CONCLUSION**: VGS computes spatial variance, NOT stochastic variance.

### Test Case 3: Constant Heterogeneous Gradients
```python
# Apply SAME spatially-heterogeneous gradient at every timestep
for step in range(30):
    param.grad = torch.linspace(-1.0, 1.0, 100)  # SAME every time
    vgs.step()

variance = vgs.get_normalized_variance()
# Result: 1.278025  <- High variance!
```

**Analysis:**
- **Spatial variance at each timestep**: `torch.var(linspace(-1, 1)) ≈ 0.33` (non-zero)
- **EMA of spatial variance**: `≈ 0.33`
- **TRUE stochastic variance**: `0` (no temporal variation - gradient is CONSTANT over time!)

**CONCLUSION**: VGS reports high variance even when gradients are temporally constant. This is spatial variance, not stochastic!

---

## Impact

### Theoretical Impact
1. **Broken Algorithm**: VGS is supposed to scale gradients based on temporal noise (stochastic variance). Instead, it scales based on spatial heterogeneity.
2. **Wrong Behavior**:
   - **Case A**: Layer with heterogeneous but stable gradients (e.g., different weights learning different features) → High spatial variance → **Incorrectly** scaled down
   - **Case B**: Layer with uniform but noisy gradients (e.g., high-variance stochastic estimates) → Zero spatial variance → **Not** scaled (should be!)

### Practical Impact
- Training instability may be **worse** than without VGS in some cases
- The feature provides incorrect stabilization signal
- Models trained with VGS may have suboptimal convergence

### Why wasn't this caught?
1. **Documentation mismatch**: Code claims "v2.0 fixed to stochastic variance" but implementation unchanged
2. **Test inadequacy**: Existing tests (e.g., `test_vgs_stochastic_variance.py`) don't properly distinguish spatial from stochastic
3. **Numerical stability excuse**: Comments say "v2.0.1 - numerically stable" uses `E[|g|]` and `Var[g]`, but the `Var[g]` is still computed spatially!

---

## Root Cause

The confusion comes from **two different semantics** of "variance":

### Spatial Variance (what VGS computes)
```python
# At timestep t, compute variance ACROSS parameter elements:
spatial_var_t = torch.var(grad_t)  # Variance across dimension
# Then track EMA over time:
ema_spatial = β * ema_spatial + (1-β) * spatial_var_t
```

### Stochastic Variance (what VGS should compute)
```python
# Track mean and second moment OVER TIME:
ema_mean = β * ema_mean + (1-β) * grad_t.mean()  # E[g]
ema_sq = β * ema_sq + (1-β) * (grad_t**2).mean()  # E[g²]
# Compute temporal variance:
stochastic_var = ema_sq - ema_mean**2  # Var[g] = E[g²] - E[g]²
```

**Current code uses approach 1, should use approach 2!**

---

## Fix Strategy

### Option 1: E[g] and E[g²] Approach (Recommended)
```python
def update_statistics(self) -> None:
    for i, param in enumerate(self._parameters):
        if param.grad is None:
            continue

        grad = param.grad.data

        # Compute current statistics
        grad_mean_current = grad.mean().item()  # Mean (can be negative!)
        grad_sq_current = (grad**2).mean().item()  # Mean of squares (always >= 0)

        # Update EMA
        if self._step_count == 1:
            self._param_grad_mean_ema[i] = grad_mean_current  # E[g]
            self._param_grad_sq_ema[i] = grad_sq_current      # E[g²]
        else:
            self._param_grad_mean_ema[i] = (
                self.beta * self._param_grad_mean_ema[i] +
                (1 - self.beta) * grad_mean_current
            )
            self._param_grad_sq_ema[i] = (
                self.beta * self._param_grad_sq_ema[i] +
                (1 - self.beta) * grad_sq_current
            )

def get_normalized_variance(self) -> float:
    # ...

    # Compute stochastic variance: Var[g] = E[g²] - E[g]²
    mean_corrected = self._param_grad_mean_ema / bias_correction  # E[g]
    sq_corrected = self._param_grad_sq_ema / bias_correction      # E[g²]

    # Var[g] = E[g²] - E[g]² (can be negative due to numerical error!)
    variance = sq_corrected - mean_corrected.pow(2)
    variance = torch.clamp(variance, min=0.0)  # Numerical stability

    # Normalized variance: Var[g] / (E[g]² + ε) for scale invariance
    denominator = torch.clamp(mean_corrected.abs().pow(2), min=1e-12) + self.eps
    normalized_var_per_param = variance / denominator
    # ...
```

### Option 2: Welford's Algorithm (More numerically stable)
Use Welford's online algorithm for variance computation to avoid catastrophic cancellation.

---

## Backward Compatibility

**BREAKING CHANGE**: This fix fundamentally changes what VGS computes.

### Migration Strategy:
1. **Version bump**: v2.0 → v3.0 (major version for breaking change)
2. **Checkpoint migration**: Old checkpoints should warn that statistics will be reset
3. **Retraining recommended**: Models trained with old VGS may benefit from retraining

### Checkpoint Handling:
```python
def load_state_dict(self, state_dict):
    vgs_version = state_dict.get("vgs_version", "1.0")

    if vgs_version in ["1.0", "2.0", "2.0.1"]:
        warnings.warn(
            "\n" + "="*80 + "\n"
            "VGS v3.0 CRITICAL FIX: Stochastic Variance Computation\n"
            "="*80 + "\n"
            "Previous versions (v1.x, v2.x) computed SPATIAL variance (across elements)\n"
            "instead of STOCHASTIC variance (over time). This has been FIXED in v3.0.\n"
            "\n"
            "Per-parameter statistics will be RESET to ensure correct behavior.\n"
            "RECOMMENDATION: Retrain models for optimal performance.\n"
            "="*80,
            UserWarning
        )
        # Reset per-parameter stats
        self._param_grad_mean_ema = None
        self._param_grad_sq_ema = None
```

---

## Testing Strategy

### New Tests Required:
1. **Test uniform noisy gradients** - MUST show non-zero variance
2. **Test heterogeneous constant gradients** - MUST show zero variance
3. **Test temporal vs spatial variance** - Verify correct metric is used
4. **Test numerical stability** - E[g²] - E[g]² edge cases
5. **Test checkpoint migration** - Old → new version

### Test Cases:
```python
def test_uniform_noisy_gradients_nonzero_variance():
    """Uniform gradients with temporal noise should have non-zero stochastic variance."""
    param = nn.Parameter(torch.randn(100))
    vgs = VarianceGradientScaler([param], warmup_steps=5)

    for step in range(30):
        noise = torch.randn(1).item() * 2.0 + 1.0
        param.grad = torch.ones(100) * noise  # Uniform, noisy
        vgs.step()

    variance = vgs.get_normalized_variance()
    assert variance > 0.1, f"Expected non-zero variance for noisy gradients, got {variance}"

def test_heterogeneous_constant_gradients_zero_variance():
    """Heterogeneous but constant gradients should have zero stochastic variance."""
    param = nn.Parameter(torch.randn(100))
    vgs = VarianceGradientScaler([param], warmup_steps=5)

    constant_grad = torch.linspace(-1.0, 1.0, 100)
    for step in range(30):
        param.grad = constant_grad.clone()  # SAME every time
        vgs.step()

    variance = vgs.get_normalized_variance()
    assert variance < 0.01, f"Expected near-zero variance for constant gradients, got {variance}"
```

---

## References

- **Original Paper**: Faghri & Duvenaud (2020). "A Study of Gradient Variance in Deep Learning." arXiv:2007.04532
  - Clearly defines stochastic variance as temporal variance of gradient estimates
- **Kingma & Ba (2015)**: "Adam: A Method for Stochastic Optimization." ICLR.
  - Uses E[g] and E[g²] for second-moment estimation (stochastic variance)

---

## Action Items

1. [ ] Implement fix (Option 1: E[g] and E[g²] approach)
2. [ ] Add comprehensive tests for stochastic variance
3. [ ] Update documentation to clarify v3.0 fix
4. [ ] Add checkpoint migration with warning
5. [ ] Run integration tests with distributional PPO
6. [ ] Update CLAUDE.md to document fix
7. [ ] Consider retraining models with corrected VGS

---

## Appendix: Why This Matters

### Example Scenario
Consider a neural network layer learning:
- Weight W[0, :] learns to detect "edges" → gradients: `[-0.1, -0.2, ..., -0.3]` (heterogeneous, stable)
- Weight W[1, :] learns to detect "colors" → gradients: `[0.5, 0.6, ..., 0.7]` (heterogeneous, stable)

**Old VGS (spatial variance)**:
- Computes high spatial variance (elements differ within each weight vector)
- Scales down gradients → **Incorrectly** slows learning of these stable features

**New VGS (stochastic variance)**:
- Computes low stochastic variance (gradients are temporally stable)
- No scaling applied → **Correctly** allows fast learning of stable features

**Result**: Correct VGS should improve convergence on stable features while still dampening noisy gradients.

---

**Report Author**: Claude Code
**Verification**: See `analyze_upgd_vgs_issues.py`
**Status**: Ready for fix implementation
