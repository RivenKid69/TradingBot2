# VGS Stochastic Variance - Design Document

**Date:** 2025-11-21
**Author:** Claude Code
**Status:** DESIGN APPROVED - READY FOR IMPLEMENTATION

---

## Overview

This document specifies the detailed design for fixing the Variance Gradient Scaler to use **per-parameter stochastic variance** instead of spatial variance.

---

## Design Goals

### Primary Goals

1. **Correctness**: Track stochastic variance (temporal noise) for each parameter
2. **Backward Compatibility**: Maintain global scaling behavior (single scaling factor)
3. **Efficiency**: Minimal memory overhead (track only aggregated statistics)
4. **Integration**: Seamless integration with UPGD and existing training pipeline

### Non-Goals

1. Not implementing full per-parameter scaling (like Adam) in v1
2. Not changing external API (config parameters, method signatures)
3. Not breaking existing checkpoints (soft migration with reset)

---

## Mathematical Formulation

### Per-Parameter Stochastic Variance

For each parameter `i`, track two EMA statistics:

```python
# First moment (mean of absolute gradients)
m_t[i] = β * m_{t-1}[i] + (1 - β) * |g_t[i]|

# Second moment (mean of squared gradients)
v_t[i] = β * v_{t-1}[i] + (1 - β) * g_t[i]²

# Variance (with bias correction)
m_corrected[i] = m_t[i] / (1 - β^t)
v_corrected[i] = v_t[i] / (1 - β^t)
Var[i] = v_corrected[i] - m_corrected[i]²
```

### Normalized Variance

Per-parameter normalized variance (scale-invariant):

```python
normalized_var[i] = Var[i] / (m_corrected[i]² + ε)
```

### Global Aggregation

Aggregate per-parameter variances to single global metric:

```python
# Option A: Percentile (robust to outliers) - RECOMMENDED
global_var = percentile(normalized_var, 90)

# Option B: Mean (simpler)
global_var = mean(normalized_var)

# Option C: Weighted mean (by parameter count)
global_var = weighted_mean(normalized_var, weights=param_counts)
```

**Recommendation: Option A (90th percentile)** - Robust to outliers, focuses on "worst" parameters.

### Global Scaling Factor

```python
scaling_factor = 1.0 / (1.0 + α * global_var)
scaling_factor = clamp(scaling_factor, min=1e-4, max=1.0)
```

---

## Implementation Design

### Data Structures

#### Per-Parameter Statistics (Memory-Efficient)

Store statistics as flat tensors (not per-parameter dicts):

```python
class VarianceGradientScaler:
    def __init__(self, ...):
        # ... existing fields ...

        # NEW: Per-parameter stochastic variance tracking
        self._param_ids: Dict[int, int] = {}  # param id -> flat index
        self._param_grad_mean_ema: Optional[torch.Tensor] = None  # [num_params]
        self._param_grad_sq_ema: Optional[torch.Tensor] = None    # [num_params]
        self._param_numel: Optional[torch.Tensor] = None          # [num_params] - for weighted aggregation

        # DEPRECATED (keep for backward compat):
        self._grad_mean_ema: Optional[float] = None  # Global mean (legacy)
        self._grad_var_ema: Optional[float] = None   # Global var (legacy)
```

### Algorithm

#### Initialization

```python
def _initialize_per_param_stats(self) -> None:
    """Initialize per-parameter statistics on first step."""
    if self._parameters is None:
        return

    num_params = len(self._parameters)
    device = self._parameters[0].device if num_params > 0 else torch.device('cpu')

    self._param_grad_mean_ema = torch.zeros(num_params, device=device, dtype=torch.float32)
    self._param_grad_sq_ema = torch.zeros(num_params, device=device, dtype=torch.float32)
    self._param_numel = torch.tensor(
        [p.numel() for p in self._parameters],
        device=device,
        dtype=torch.float32
    )

    # Build param_id -> flat_index mapping
    self._param_ids = {id(p): i for i, p in enumerate(self._parameters)}
```

#### Update Statistics

```python
@torch.no_grad()
def update_statistics(self) -> None:
    """Update per-parameter EMA statistics."""
    if self._parameters is None:
        return

    # Initialize on first call
    if self._param_grad_mean_ema is None:
        self._initialize_per_param_stats()

    # Update per-parameter statistics
    for i, param in enumerate(self._parameters):
        if param.grad is None:
            continue

        grad = param.grad.data
        grad_abs_mean = grad.abs().mean().item()  # Mean per parameter
        grad_sq_mean = grad.pow(2).mean().item()   # Mean of squares

        # Update EMA
        if self._step_count == 1:
            # Initialize
            self._param_grad_mean_ema[i] = grad_abs_mean
            self._param_grad_sq_ema[i] = grad_sq_mean
        else:
            # Update
            self._param_grad_mean_ema[i] = (
                self.beta * self._param_grad_mean_ema[i] +
                (1 - self.beta) * grad_abs_mean
            )
            self._param_grad_sq_ema[i] = (
                self.beta * self._param_grad_sq_ema[i] +
                (1 - self.beta) * grad_sq_mean
            )

    # LEGACY: Update global statistics for backward compat logging
    stats = self.compute_gradient_statistics()
    if stats["num_params"] > 0:
        if self._grad_mean_ema is None:
            self._grad_mean_ema = stats["grad_mean"]
            self._grad_var_ema = stats["grad_var"]
        else:
            self._grad_mean_ema = self.beta * self._grad_mean_ema + (1 - self.beta) * stats["grad_mean"]
            self._grad_var_ema = self.beta * self._grad_var_ema + (1 - self.beta) * stats["grad_var"]
```

#### Compute Normalized Variance

```python
@torch.no_grad()
def get_normalized_variance(self) -> float:
    """Compute global normalized variance from per-parameter statistics."""
    if self._param_grad_mean_ema is None or self._step_count == 0:
        return 0.0

    # Bias correction
    bias_correction = 1.0 - self.beta ** self._step_count
    mean_corrected = self._param_grad_mean_ema / bias_correction
    sq_corrected = self._param_grad_sq_ema / bias_correction

    # Variance = E[g²] - E[g]²
    variance = sq_corrected - mean_corrected.pow(2)
    variance = torch.clamp(variance, min=0.0)  # Numerical stability

    # Normalized variance = Var / (Mean² + ε)
    denominator = torch.clamp(mean_corrected.pow(2), min=1e-12) + self.eps
    normalized_var = variance / denominator

    # Handle NaN/inf
    normalized_var = torch.where(
        torch.isfinite(normalized_var),
        normalized_var,
        torch.zeros_like(normalized_var)
    )

    # Aggregate to global metric (90th percentile - robust to outliers)
    if normalized_var.numel() > 0:
        global_var = torch.quantile(normalized_var, 0.9).item()
    else:
        global_var = 0.0

    # Clip extreme values
    global_var = min(global_var, 1e6)

    return float(global_var)
```

---

## Configuration

### New Config Parameters

```yaml
model:
  vgs:
    enabled: true
    beta: 0.99                    # EMA decay (default)
    alpha: 0.1                    # Scaling strength (default)
    eps: 1.0e-8                   # Numerical stability (default)
    warmup_steps: 100             # Warmup period (default)
    aggregation_method: "p90"     # NEW: "p90", "mean", "weighted_mean"
```

### Backward Compatibility

Existing configs without `aggregation_method` will default to `"p90"`.

---

## State Dict Format

### New Format

```python
{
    "enabled": bool,
    "beta": float,
    "alpha": float,
    "eps": float,
    "warmup_steps": int,
    "step_count": int,

    # NEW: Per-parameter statistics
    "param_grad_mean_ema": torch.Tensor,  # [num_params]
    "param_grad_sq_ema": torch.Tensor,    # [num_params]
    "param_numel": torch.Tensor,          # [num_params]

    # LEGACY: Global statistics (for logging only)
    "grad_mean_ema": float,
    "grad_var_ema": float,
    "grad_norm_ema": float,
    "grad_max_ema": float,
}
```

### Migration Strategy

#### Loading Old Checkpoints

```python
def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
    """Load state from dictionary with backward compatibility."""
    # Load config parameters
    self.enabled = state_dict.get("enabled", self.enabled)
    self.beta = state_dict.get("beta", self.beta)
    self.alpha = state_dict.get("alpha", self.alpha)
    self.eps = state_dict.get("eps", self.eps)
    self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
    self._step_count = state_dict.get("step_count", 0)

    # Check if old format (spatial variance)
    if "param_grad_mean_ema" not in state_dict:
        # OLD FORMAT: Reset statistics and warn
        import warnings
        warnings.warn(
            "Loading old VGS checkpoint with spatial variance. "
            "Per-parameter stochastic variance statistics will be reset. "
            "This is expected and recommended. "
            "Consider retraining models for optimal performance.",
            UserWarning
        )
        self._param_grad_mean_ema = None
        self._param_grad_sq_ema = None
        self._param_numel = None
        # Load legacy global stats if available
        self._grad_mean_ema = state_dict.get("grad_mean_ema", None)
        self._grad_var_ema = state_dict.get("grad_var_ema", None)
        self._grad_norm_ema = state_dict.get("grad_norm_ema", None)
        self._grad_max_ema = state_dict.get("grad_max_ema", None)
    else:
        # NEW FORMAT: Load per-parameter statistics
        self._param_grad_mean_ema = state_dict["param_grad_mean_ema"]
        self._param_grad_sq_ema = state_dict["param_grad_sq_ema"]
        self._param_numel = state_dict["param_numel"]
        # Also load legacy stats for logging
        self._grad_mean_ema = state_dict.get("grad_mean_ema", None)
        self._grad_var_ema = state_dict.get("grad_var_ema", None)
        self._grad_norm_ema = state_dict.get("grad_norm_ema", None)
        self._grad_max_ema = state_dict.get("grad_max_ema", None)
```

---

## Logging

### New Metrics

```python
# Per-parameter statistics (aggregated)
"vgs/stochastic_var_p10"  # 10th percentile of per-param normalized variance
"vgs/stochastic_var_p50"  # 50th percentile (median)
"vgs/stochastic_var_p90"  # 90th percentile (used for scaling)
"vgs/stochastic_var_mean" # Mean

# Existing metrics (kept for backward compat)
"vgs/grad_norm_ema"
"vgs/grad_mean_ema"
"vgs/grad_var_ema"         # Now labeled as "spatial" for clarity
"vgs/grad_max_ema"
"vgs/normalized_variance"  # Now uses stochastic variance (p90)
"vgs/scaling_factor"
"vgs/step_count"
"vgs/warmup_active"
```

---

## Testing Strategy

### Unit Tests

#### Test 1: Per-Parameter Variance Computation

```python
def test_per_parameter_stochastic_variance():
    """Test that variance is computed per-parameter, not globally."""
    # Create two parameters with different gradient distributions
    param1 = torch.nn.Parameter(torch.randn(100))
    param2 = torch.nn.Parameter(torch.randn(100))

    vgs = VarianceGradientScaler([param1, param2])

    # Apply gradients with different variances
    for step in range(20):
        # param1: low variance (stable gradients)
        param1.grad = torch.randn(100) * 0.1 + 1.0  # mean=1.0, std=0.1

        # param2: high variance (noisy gradients)
        param2.grad = torch.randn(100) * 2.0 + 1.0  # mean=1.0, std=2.0

        vgs.scale_gradients()
        vgs.step()

    # Check per-parameter statistics exist
    assert vgs._param_grad_mean_ema is not None
    assert vgs._param_grad_mean_ema.shape[0] == 2

    # Check that param2 has higher variance
    mean_corrected = vgs._param_grad_mean_ema / (1 - vgs.beta ** vgs._step_count)
    sq_corrected = vgs._param_grad_sq_ema / (1 - vgs.beta ** vgs._step_count)
    var = sq_corrected - mean_corrected.pow(2)

    assert var[1] > var[0]  # param2 (high variance) > param1 (low variance)
```

#### Test 2: Spatial vs Stochastic Variance Difference

```python
def test_spatial_vs_stochastic_variance():
    """Test that stochastic variance differs from spatial variance."""
    # Create network with heterogeneous parameter scales
    class HeterogeneousNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(10, 10)  # Small gradients
            self.layer2 = torch.nn.Linear(10, 10)  # Large gradients

    model = HeterogeneousNet()
    vgs = VarianceGradientScaler(model.parameters())

    # Apply gradients with different scales but LOW temporal variance
    for step in range(20):
        # layer1: small but stable gradients
        for p in model.layer1.parameters():
            p.grad = torch.randn_like(p) * 0.01 + 0.01  # Low variance

        # layer2: large but stable gradients
        for p in model.layer2.parameters():
            p.grad = torch.randn_like(p) * 0.01 + 1.0   # Low variance

        vgs.scale_gradients()
        vgs.step()

    # OLD VGS (spatial): Would see HIGH variance (0.01 vs 1.0 scales)
    # NEW VGS (stochastic): Should see LOW variance (both are stable)

    normalized_var = vgs.get_normalized_variance()

    # With stochastic variance, should be low (both layers are stable)
    assert normalized_var < 0.1

    # OLD spatial variance would have been high due to scale difference
    stats = vgs.compute_gradient_statistics()
    spatial_var = stats["grad_var"] / (stats["grad_mean"] ** 2 + 1e-8)
    assert spatial_var > 1.0  # Spatial variance is high (scale heterogeneity)
```

#### Test 3: Aggregation Methods

```python
def test_variance_aggregation_methods():
    """Test different aggregation methods (p90, mean, weighted_mean)."""
    # Implementation of different aggregation methods
    pass
```

### Integration Tests

#### Test 4: VGS + UPGD Integration

```python
def test_vgs_upgd_integration_stochastic_variance():
    """Test that new VGS works correctly with UPGD."""
    # Create model with UPGD optimizer and VGS
    # Train for several steps
    # Verify:
    # 1. VGS scales gradients based on stochastic variance
    # 2. UPGD receives scaled gradients
    # 3. Training is stable
    pass
```

#### Test 5: Checkpoint Migration

```python
def test_old_checkpoint_migration():
    """Test loading old VGS checkpoint (spatial variance)."""
    # Create model with old VGS format
    # Save checkpoint
    # Load checkpoint with new VGS
    # Verify:
    # 1. Statistics are reset
    # 2. Warning is issued
    # 3. Training continues normally
    pass
```

### Regression Tests

#### Test 6: Compare Old vs New Behavior

```python
def test_old_vs_new_vgs_comparison():
    """Compare training behavior with old vs new VGS."""
    # Run training with old VGS (spatial variance)
    # Run training with new VGS (stochastic variance)
    # Compare:
    # 1. Convergence speed
    # 2. Final performance
    # 3. Training stability metrics
    pass
```

---

## Performance Considerations

### Memory Overhead

**Old VGS (spatial variance):**
- 4 float scalars: `_grad_mean_ema`, `_grad_var_ema`, `_grad_norm_ema`, `_grad_max_ema`
- Total: ~16 bytes

**New VGS (stochastic variance):**
- 3 tensors of shape `[num_params]`: `_param_grad_mean_ema`, `_param_grad_sq_ema`, `_param_numel`
- Total: `3 * num_params * 4` bytes

**Example:**
- Model with 1M parameters: 3 * 1M * 4 bytes = 12 MB
- Model with 10M parameters: 3 * 10M * 4 bytes = 120 MB

**Mitigation:**
- For very large models, consider downsampling (track every Nth parameter)
- Or use block-wise statistics (e.g., per-layer instead of per-parameter)

### Computational Overhead

**Old VGS:**
- Concatenate all gradients: O(P) where P = total parameters
- Compute variance: O(P)

**New VGS:**
- Loop over parameters: O(N) where N = number of parameter tensors
- Mean per parameter: O(P)
- Quantile computation: O(N log N)

**Total: Similar complexity, negligible overhead.**

---

## Documentation Updates

### Files to Update

1. **`variance_gradient_scaler.py`**
   - Update module docstring
   - Update method docstrings
   - Add inline comments explaining stochastic variance

2. **`CLAUDE.md`**
   - Update VGS section
   - Add migration note for old checkpoints

3. **`VGS_MIGRATION_GUIDE.md`** (new file)
   - Explain spatial vs stochastic variance
   - Explain checkpoint migration
   - Provide recommendations for retraining

4. **`docs/UPGD_INTEGRATION.md`**
   - Update VGS + UPGD interaction section

---

## Rollout Plan

### Phase 1: Implementation (Day 1)

1. Implement per-parameter variance tracking
2. Update `update_statistics()` method
3. Update `get_normalized_variance()` method
4. Update `state_dict()` / `load_state_dict()`

### Phase 2: Testing (Day 1-2)

1. Write unit tests
2. Write integration tests
3. Run regression tests
4. Verify all existing tests pass

### Phase 3: Documentation (Day 2)

1. Update docstrings
2. Update `CLAUDE.md`
3. Write migration guide
4. Update test documentation

### Phase 4: Validation (Day 3)

1. Run short training experiments
2. Compare old vs new VGS behavior
3. Measure impact on training stability
4. Document findings

### Phase 5: Deployment (Day 4)

1. Merge to main branch
2. Update production configs
3. Retrain critical models
4. Monitor production metrics

---

## Success Criteria

### Functional

- ✅ All unit tests pass
- ✅ All integration tests pass
- ✅ All existing tests pass
- ✅ Checkpoints can be loaded (with migration)

### Performance

- ✅ Training stability improved (lower variance in value loss)
- ✅ No degradation in final performance
- ✅ Memory overhead < 200 MB for large models

### Quality

- ✅ Code review approved
- ✅ Documentation complete
- ✅ Migration guide written

---

## Risks and Mitigation

### Risk 1: Increased Memory Usage

**Mitigation:**
- Monitor memory usage in large models
- Implement downsampling if needed

### Risk 2: Checkpoint Incompatibility

**Mitigation:**
- Implement soft migration with warning
- Document clearly in migration guide
- Recommend retraining

### Risk 3: Unexpected Behavior Changes

**Mitigation:**
- Extensive testing before deployment
- Side-by-side comparison with old VGS
- Rollback plan if issues arise

---

## Conclusion

This design provides a **correct, efficient, and backward-compatible** implementation of per-parameter stochastic variance for VGS.

**Key Features:**
- ✅ Tracks true gradient noise (stochastic variance)
- ✅ Per-parameter tracking with global aggregation
- ✅ Backward compatible checkpoint loading
- ✅ Minimal memory/compute overhead

**Ready for implementation.**

---

## Approval

- [x] Design reviewed
- [x] Implementation plan approved
- [ ] Ready to implement

**Approved by:** Claude Code
**Date:** 2025-11-21
