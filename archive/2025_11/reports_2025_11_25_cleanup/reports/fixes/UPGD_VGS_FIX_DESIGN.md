# UPGD + VGS Noise Interaction Fix Design

## Problem Summary

**Confirmed Issue:** VGS (Variance Gradient Scaler) amplifies the relative impact of UPGD (Unified Perturbed Gradient Descent) perturbation noise by 1.3x - 1.8x depending on configuration.

**Mechanism:**
1. VGS scales gradients down (grad *= scaling_factor, where scaling_factor < 1.0)
2. UPGD adds FIXED noise (noise = randn() * sigma)
3. Noise magnitude is independent of gradient magnitude
4. Result: noise-to-signal ratio increases proportionally to gradient reduction

**Example:**
- Without VGS: grad_norm=0.20, noise_std=0.005 → noise/signal = 2.5%
- With VGS (scaling=0.62): grad_norm=0.13, noise_std=0.005 → noise/signal = 3.85% (1.54x amplification)

**Severity:** MEDIUM to HIGH depending on VGS aggressiveness and UPGD sigma.

## Root Cause

UPGD uses **absolute noise scaling** (fixed sigma), which doesn't adapt to gradient magnitude changes.

When gradients are small (due to VGS scaling or natural convergence), fixed noise dominates and can destabilize training.

## Proposed Solution

Implement **adaptive noise scaling** in UPGD optimizer to maintain constant noise-to-signal ratio regardless of gradient magnitude or VGS scaling.

### Design Principles

1. **Backward Compatibility**: Add as optional feature, don't break existing behavior
2. **No Coupling**: Solution should not require communication between VGS and UPGD
3. **Automatic**: Should work transparently with any gradient scaling mechanism
4. **Research-Based**: Follow established practices in adaptive optimizers

### Solution: Gradient-Relative Noise Scaling

Add `adaptive_noise` option to AdaptiveUPGD:

```python
class AdaptiveUPGD(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.001,
        sigma: float = 0.001,
        adaptive_noise: bool = False,  # NEW: Enable adaptive noise scaling
        noise_beta: float = 0.999,     # NEW: EMA decay for gradient norm tracking
        ...
    ):
        ...
```

### Algorithm

**Without adaptive_noise (current behavior):**
```python
noise = torch.randn_like(p.grad) * sigma  # Fixed noise
```

**With adaptive_noise=True (proposed):**
```python
# Track EMA of gradient norm per parameter group
grad_norm_ema = state["grad_norm_ema"]  # Initialized to 1.0
current_grad_norm = p.grad.norm()

# Update EMA
grad_norm_ema = noise_beta * grad_norm_ema + (1 - noise_beta) * current_grad_norm
state["grad_norm_ema"] = grad_norm_ema

# Bias correction
bias_correction = 1 - noise_beta ** state["step"]
grad_norm_corrected = grad_norm_ema / bias_correction

# Scale noise to maintain constant relative magnitude
# sigma now represents target noise-to-signal ratio
adaptive_sigma = sigma * max(grad_norm_corrected, 1e-8)
noise = torch.randn_like(p.grad) * adaptive_sigma
```

**Effect:**
- Noise scales proportionally with gradient magnitude
- Maintains constant noise-to-signal ratio
- Works with VGS: if VGS reduces gradients by 50%, noise also reduces by 50%
- Works during convergence: as gradients naturally decrease, noise decreases too

### Implementation Details

1. **Add state variables:**
   - `grad_norm_ema`: Per-parameter EMA of gradient norm
   - `noise_beta`: EMA decay rate for gradient norm tracking

2. **Initialization:**
   - `grad_norm_ema` initialized to 1.0 (neutral starting point)
   - Use bias correction for first few steps

3. **Noise computation:**
   - Compute per-parameter gradient norm
   - Update EMA with bias correction
   - Scale sigma by EMA norm: `adaptive_sigma = sigma * grad_norm_ema`
   - Generate noise: `noise = randn() * adaptive_sigma`

4. **Hyperparameter interpretation:**
   - When `adaptive_noise=False`: sigma is absolute noise std
   - When `adaptive_noise=True`: sigma is relative noise-to-signal ratio
   - Default: `adaptive_noise=False` (backward compatible)

5. **Per-parameter vs global norm:**
   - **Per-parameter norm** (recommended): Each parameter gets noise proportional to its gradient magnitude
   - Pros: More fine-grained, respects parameter-specific gradient scales
   - Cons: Slightly more computation

### Alternative: Global Gradient Norm

Could also use global gradient norm (single EMA across all parameters):

```python
# In first pass: compute global grad norm
global_grad_norm = 0.0
for group in self.param_groups:
    for p in group["params"]:
        if p.grad is not None:
            global_grad_norm += p.grad.norm().item() ** 2
global_grad_norm = global_grad_norm ** 0.5

# Update global EMA (stored in optimizer state)
self.state["global_grad_norm_ema"] = ...

# In second pass: use global EMA for all parameters
adaptive_sigma = sigma * self.state["global_grad_norm_ema"]
```

**Recommendation:** Use **per-parameter norm** for better granularity.

### Configuration Examples

**Current defaults (unchanged):**
```yaml
optimizer: AdaptiveUPGD
optimizer_kwargs:
  lr: 0.0003
  sigma: 0.001        # Absolute noise std
  adaptive_noise: false  # Default: fixed noise
```

**With VGS + adaptive noise (recommended):**
```yaml
optimizer: AdaptiveUPGD
optimizer_kwargs:
  lr: 0.0003
  sigma: 0.05         # 5% relative noise-to-signal ratio
  adaptive_noise: true   # Enable adaptive scaling
  noise_beta: 0.999   # Slow EMA for stability

vgs:
  enabled: true
  alpha: 0.2
```

**Aggressive exploration (high noise, adaptive):**
```yaml
optimizer: AdaptiveUPGD
optimizer_kwargs:
  sigma: 0.10         # 10% relative noise
  adaptive_noise: true
```

## Expected Impact

### Before Fix (Current Behavior)
- VGS alpha=0.3, UPGD sigma=0.005
- Gradient reduction: 38%
- Noise amplification: 1.32x
- Noise-to-signal: 3.96% → 5.22%
- **Result:** Increased training noise, potential instability

### After Fix (With adaptive_noise=True)
- VGS alpha=0.3, UPGD sigma=0.05 (interpreted as 5% ratio)
- Gradient reduction: 38%
- Noise reduction: 38% (matches gradient reduction)
- Noise-to-signal: 5.00% → 5.00% (constant!)
- **Result:** Stable noise-to-signal ratio, predictable training dynamics

## Testing Strategy

1. **Unit tests:**
   - Test adaptive noise scaling correctly tracks gradient norms
   - Test noise scales proportionally to gradients
   - Test backward compatibility (adaptive_noise=False)

2. **Integration tests:**
   - Test with VGS: verify constant noise-to-signal ratio
   - Test without VGS: verify adaptive noise during natural convergence
   - Test edge cases: zero gradients, very small gradients

3. **Regression tests:**
   - Test existing training configurations still work
   - Test model checkpointing/loading with new state variables

4. **Validation tests:**
   - Compare training curves: baseline vs adaptive noise
   - Verify training stability improvements
   - Measure noise-to-signal ratio across training

## Migration Path

1. **Phase 1: Implementation**
   - Add `adaptive_noise` parameter to AdaptiveUPGD
   - Implement per-parameter adaptive noise scaling
   - Default to `False` (backward compatible)

2. **Phase 2: Testing**
   - Run comprehensive unit and integration tests
   - Validate training stability improvements
   - Document new hyperparameter interpretation

3. **Phase 3: Deployment**
   - Update config templates with recommended settings
   - Add documentation and examples
   - Gradual rollout: test on non-critical models first

4. **Phase 4: Migration**
   - Update production configs to use `adaptive_noise=True`
   - Adjust sigma values (from absolute to relative interpretation)
   - Monitor training metrics for improvements

## References

- Adaptive noise scaling is common in evolutionary strategies (ES)
- Similar to adaptive momentum in Adam optimizer
- Maintains exploration-exploitation balance during training
- Research: "Parameter-exploring policy gradients" (Sehnke et al., 2010)

## Risks and Mitigations

### Risk 1: Noise becomes too small during convergence
**Mitigation:** Add `min_noise_std` parameter to enforce minimum noise floor

### Risk 2: EMA tracks outdated gradient norms
**Mitigation:** Use fast EMA decay (beta=0.99) for responsiveness

### Risk 3: Breaks existing training runs
**Mitigation:** Backward compatible default (adaptive_noise=False)

### Risk 4: Checkpoint compatibility
**Mitigation:** Handle missing state variables gracefully in load_state_dict

## Summary

**Fix:** Add adaptive noise scaling to UPGD optimizer to maintain constant noise-to-signal ratio regardless of gradient magnitude changes (from VGS or natural convergence).

**Key Benefits:**
- ✅ Maintains constant noise-to-signal ratio with VGS
- ✅ Adapts to natural gradient decay during convergence
- ✅ No coupling between VGS and UPGD required
- ✅ Backward compatible
- ✅ Based on established research

**Implementation Effort:** ~200 LOC + tests

**Recommended Default:** `adaptive_noise=True` for all VGS-enabled configs
