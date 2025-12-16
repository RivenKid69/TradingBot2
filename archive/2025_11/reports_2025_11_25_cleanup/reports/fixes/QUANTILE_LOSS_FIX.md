# CRITICAL FIX: Quantile Regression Loss Asymmetry

## Summary

Fixed a critical mathematical error in the quantile regression loss implementation that caused inverted asymmetry coefficients.

**⚠️ IMPORTANT: This fix is DISABLED BY DEFAULT for backward compatibility.**

To enable the fix, set `policy.use_fixed_quantile_loss_asymmetry = True`.

## Problem Description

### Location
`distributional_ppo.py:2544-2553` (now fixed at line 2549)

### The Bug

The implementation used **inverted delta**:
```python
# INCORRECT (before fix)
delta = predicted_quantiles - targets  # Q - T (inverted!)
indicator = (delta.detach() < 0.0).float()  # I{Q < T}
loss_per_quantile = torch.abs(tau - indicator) * huber
```

This caused **inverted asymmetry coefficients**:
- When `Q < T` (underestimation): coefficient = `(1 - τ)` ❌ (should be `τ`)
- When `Q ≥ T` (overestimation): coefficient = `τ` ❌ (should be `(1 - τ)`)

### Impact

This bug had severe consequences for distributional RL:

1. **Inverted Learning Objectives**:
   - For `τ < 0.5` quantiles: model learns to **overestimate** instead of **underestimate**
   - For `τ > 0.5` quantiles: model learns to **underestimate** instead of **overestimate**
   - Only `τ = 0.5` (median) was unaffected

2. **Broken CVaR (Conditional Value at Risk)**:
   - CVaR computation relies on correct low quantiles
   - With inverted coefficients, low quantiles were biased upward
   - Risk-averse policies would be trained incorrectly

3. **Incorrect Distributional Predictions**:
   - Return distribution predictions were systematically biased
   - Quantile ordering could be preserved, but values were wrong
   - Policy optimization using distributional value functions was compromised

4. **Training Instability**:
   - Gradient directions were inverted for non-median quantiles
   - This could cause slower convergence or training instabilities

## The Correct Formula

According to **Dabney et al. 2018** ("Distributional Reinforcement Learning with Quantile Regression", AAAI):

### Standard Quantile Loss
```
ρ_τ(u) = u · (τ - I{u < 0})
```
where `u = target - predicted`

### With Huber Smoothing
```
ρ_τ^κ(u) = |τ - I{u < 0}| · L_κ(u)
```
where:
- `L_κ(u)` is the Huber loss with threshold `κ`
- `I{·}` is the indicator function

### Asymmetry Properties

For a `τ`-quantile:
- **Underestimation** (`Q < T`, meaning `u > 0`):
  - `I{u < 0} = 0`
  - Coefficient = `|τ - 0| = τ`
  - Example: For `τ = 0.25`, penalty = `0.25`

- **Overestimation** (`Q ≥ T`, meaning `u ≤ 0`):
  - `I{u < 0} = 1`
  - Coefficient = `|τ - 1| = 1 - τ`
  - Example: For `τ = 0.25`, penalty = `0.75`

This asymmetry ensures:
- Low quantiles (`τ < 0.5`) penalize **overestimation** more → conservative predictions
- High quantiles (`τ > 0.5`) penalize **underestimation** more → aggressive predictions
- Median (`τ = 0.5`) penalizes both equally → unbiased predictions

## How to Enable the Fix

The fix is **disabled by default** to ensure backward compatibility with existing models and experiments.

To enable the corrected formula:

```python
# In your policy configuration or network architecture
policy.use_fixed_quantile_loss_asymmetry = True
```

Or when creating a custom policy:

```python
class CustomQuantilePolicy:
    def __init__(self):
        self.uses_quantile_value_head = True
        self.quantile_huber_kappa = 1.0
        self.use_fixed_quantile_loss_asymmetry = True  # Enable fix
```

**Recommendation:** Enable this fix for **all new training runs**. Only keep it disabled if you need exact reproducibility with old (buggy) models.

## The Fix

### Code Change

```python
# FIXED implementation
delta = targets - predicted_quantiles  # T - Q (CORRECT!)
abs_delta = delta.abs()
huber = torch.where(
    abs_delta <= kappa,
    0.5 * delta.pow(2),
    kappa * (abs_delta - 0.5 * kappa),
)
indicator = (delta.detach() < 0.0).float()  # I{T < Q}
loss_per_quantile = torch.abs(tau - indicator) * huber
```

### Key Change
**Changed one line**: `delta = targets - predicted_quantiles` (was `predicted_quantiles - targets`)

This single-character fix (`-` sign swap) corrects the entire asymmetry.

## Verification

### Mathematical Test

A simple test confirms the fix:

```python
# For τ = 0.25 (25th percentile)
target = 0.0
predicted_under = -1.0  # Underestimation
predicted_over = 1.0    # Overestimation

# CORRECT implementation gives:
# - Underestimation coefficient: 0.25 ✓
# - Overestimation coefficient: 0.75 ✓
# - Ratio (over/under): 3.0 ✓ (matches (1-τ)/τ)

# BUGGY implementation gave:
# - Underestimation coefficient: 0.75 ✗
# - Overestimation coefficient: 0.25 ✗
# - Ratio (over/under): 0.333 ✗ (inverted!)
```

See `verify_quantile_bug.py` for the full verification script.

### Test Coverage

Comprehensive tests in `tests/test_quantile_loss_asymmetry_fix.py`:

1. ✅ **Coefficient correctness**: Verifies exact coefficient values
2. ✅ **Multiple tau values**: Tests asymmetry across `τ ∈ {0.1, 0.25, 0.5, 0.75, 0.9}`
3. ✅ **Median symmetry**: Confirms `τ = 0.5` has equal penalties
4. ✅ **Gradient direction**: Ensures gradients push in correct direction
5. ✅ **Training convergence**: Verifies model learns correct quantile ordering
6. ✅ **Huber threshold**: Tests quadratic/linear transition
7. ✅ **Batch independence**: Confirms samples don't interfere

## References

1. **Dabney, W., Rowland, M., Bellemare, M. G., & Munos, R. (2018).**
   "Distributional Reinforcement Learning with Quantile Regression"
   *AAAI Conference on Artificial Intelligence*
   [arXiv:1710.10044](https://arxiv.org/abs/1710.10044)

2. **Koenker, R., & Bassett Jr, G. (1978).**
   "Regression Quantiles"
   *Econometrica*, 46(1), 33-50

## Related Files

- **Main implementation**: `distributional_ppo.py:2484-2564`
- **Verification script**: `verify_quantile_bug.py`
- **Test suite**: `tests/test_quantile_loss_asymmetry_fix.py`
- **Existing tests** (still valid):
  - `tests/test_distributional_ppo_quantile_loss.py`
  - `tests/test_quantile_huber_kappa.py`
  - `tests/test_quantile_huber_integration.py`

## Migration Notes

### For Existing Models

⚠️ **IMPORTANT**: Models trained with the buggy loss have **inverted quantile semantics**:

- What was labeled as "25th percentile" actually learned to behave like "75th percentile"
- What was labeled as "75th percentile" actually learned to behave like "25th percentile"
- The median (50th percentile) is unaffected

### Recommendations

1. **Retrain all models** using the fixed loss
2. If you must use old models, **invert the quantile interpretation**:
   - Old model's `τ`-quantile ≈ new model's `(1-τ)`-quantile
   - Except for `τ = 0.5` which is unchanged

3. **CVaR policies**: Completely retrain, as CVaR was computed incorrectly

## Changelog

- **2025-11-18**: Fixed critical quantile loss asymmetry bug
  - Changed `delta` calculation to use correct sign
  - Added comprehensive test suite
  - Updated documentation with correct formula reference
