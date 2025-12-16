# Quantile Huber Loss Kappa Normalization Fix

## Summary

Fixed incorrect normalization in Quantile Huber Loss implementation by removing division by `kappa` parameter. The implementation now correctly matches the standard QR-DQN formula from Dabney et al., 2018.

## Problem

### Original Implementation (Incorrect)

```python
# distributional_ppo.py:2483 (before fix)
loss = torch.abs(tau - indicator) * (huber / kappa)
```

### Issue

The division by `kappa` caused incorrect loss scaling:

1. **When κ = 1.0 (default)**: No effect, formula is correct
2. **When κ ≠ 1.0**: Loss magnitude scales incorrectly
   - Quadratic region: Loss becomes `0.5 * δ² / κ` instead of `0.5 * δ²`
   - This makes the loss inversely proportional to κ in the quadratic region
   - Violates the standard Huber loss formulation

### Mathematical Analysis

**Standard QR-DQN Formula (Correct):**
```
ρ^κ_τ(u) = |τ - I{u<0}| · L^κ_H(u)

where L^κ_H(u) = { 0.5 * u²,              if |u| <= κ
                 { κ * (|u| - 0.5 * κ),   if |u| > κ
```

**With Division by κ (Incorrect):**
```
ρ^κ_τ(u) = |τ - I{u<0}| · [L^κ_H(u) / κ]

Effective formula = { 0.5 * u² / κ,      if |u| <= κ
                    { |u| - 0.5 * κ,     if |u| > κ
```

### Consequences

1. **Gradient Scaling**: In the quadratic region, gradients become `∂L/∂u = u/κ` instead of `u`, making them inversely proportional to κ
2. **Loss Balance**: When κ ≠ 1, the relative scale between policy loss and value loss changes unpredictably
3. **Non-Standard Behavior**: Deviates from reference implementations (Stable-Baselines3-Contrib, original papers)
4. **Hidden Bug**: Since default κ = 1.0, the bug had no effect in standard configurations

## Solution

### Fixed Implementation

```python
# distributional_ppo.py:2483 (after fix)
loss = torch.abs(tau - indicator) * huber
```

### Changes

**File: `distributional_ppo.py`**
- **Line 2483**: Removed `/ kappa` division
- Formula now matches standard QR-DQN specification

## Verification

### Test Coverage

Created comprehensive test suite in `tests/test_quantile_huber_kappa.py`:

1. **`test_quantile_huber_loss_kappa_scaling_quadratic_region`**
   - Verifies quadratic region loss is independent of κ
   - Tests κ ∈ {0.5, 1.0, 2.0, 5.0}
   - Expected: `L = 0.5 * |τ - I| * δ²` (independent of κ)

2. **`test_quantile_huber_loss_kappa_scaling_linear_region`**
   - Verifies linear region loss scales proportionally with κ
   - Expected: `L = |τ - I| * κ * (|δ| - 0.5*κ)`

3. **`test_quantile_huber_loss_transition_point`**
   - Ensures continuity at |δ| = κ
   - Both formulas should yield `0.5 * κ²`

4. **`test_quantile_huber_loss_gradient_magnitude_with_kappa`**
   - Verifies gradient scaling in linear region
   - Gradients should scale proportionally with κ

5. **`test_quantile_huber_loss_asymmetric_weighting`**
   - Confirms quantile weighting `|τ - I{u<0}|` works correctly
   - For τ = 0.1: underestimation penalty / overestimation penalty ≈ 9

6. **`test_quantile_huber_loss_matches_reference_implementation`**
   - End-to-end test with realistic inputs
   - Verifies correct behavior across multiple quantiles

7. **`test_quantile_huber_loss_zero_when_perfect_prediction`**
   - Edge case: loss = 0 when predictions match targets

8. **`test_quantile_huber_loss_no_division_by_kappa_regression`**
   - **Regression test**: Explicitly verifies division was removed
   - Compares loss with κ=1.0 vs κ=2.0 in quadratic region
   - They should be identical (confirms no division)

### Verification Script

`verify_huber_fix.py` provides standalone verification:

```bash
python verify_huber_fix.py
```

This script:
- Requires no testing framework (pure Python + PyTorch)
- Runs 4 key tests with detailed output
- Confirms the fix is correct

## Reference Implementations

### Stable-Baselines3-Contrib

**File**: `sb3_contrib/common/utils.py`

```python
def quantile_huber_loss(current_quantiles, target_quantiles, ...):
    pairwise_delta = target_quantiles.unsqueeze(-2) - current_quantiles.unsqueeze(-1)
    abs_pairwise_delta = th.abs(pairwise_delta)
    huber_loss = th.where(
        abs_pairwise_delta > 1,  # κ = 1 (hardcoded)
        abs_pairwise_delta - 0.5,
        pairwise_delta**2 * 0.5
    )
    loss = th.abs(cum_prob - (pairwise_delta.detach() < 0).float()) * huber_loss
    # Note: NO division by kappa
```

**Key observations:**
- Uses κ = 1.0 (hardcoded)
- NO division by kappa
- Authoritative implementation used by RL community

### Original Papers

1. **Dabney et al., 2018** - "Distributional Reinforcement Learning with Quantile Regression" (QR-DQN)
   - Defines standard Huber loss without kappa division
   - Formula: `ρ^κ_τ(u) = |τ - I{u<0}| · L^κ_H(u)`

2. **Dabney et al., 2018** - "Implicit Quantile Networks for Distributional RL" (IQN)
   - Uses same formula as QR-DQN
   - No kappa normalization

3. **Yang et al., 2024** - "A Robust Quantile Huber Loss with Interpretable Parameter Adjustment"
   - This paper DOES use division by k: `[L^k_H(u) / k]`
   - However, this is a **newer, alternative formulation**
   - Not the standard used in RL community

## Impact Analysis

### Backward Compatibility

**No impact on default configurations:**
- Default `huber_kappa = 1.0` everywhere in codebase
- With κ = 1, division has no effect: `huber / 1.0 = huber`
- All existing trained models used κ = 1.0

**Impact if κ is changed:**
- Previously: Changing κ would unexpectedly alter loss scale
- Now: Changing κ has predictable, standard behavior

### Performance

**No change expected:**
- Same computation (one less division operation)
- Default κ = 1.0 means identical numerical results
- Tests confirm loss values match expected formulas

## Related Files

### Modified
- `distributional_ppo.py` (line 2483)

### Added
- `tests/test_quantile_huber_kappa.py` - Comprehensive test suite
- `verify_huber_fix.py` - Standalone verification script
- `QUANTILE_HUBER_LOSS_FIX.md` - This documentation

### Configuration Files (No Changes Needed)
- `train_model_multi_patch.py` - Uses default κ = 1.0
- `custom_policy_patch1.py` - Passes κ to loss function

### Existing Tests (Should Still Pass)
- `tests/test_distributional_ppo_quantile_loss.py` - All use κ = 1.0
- `tests/test_popart_integration.py` - Uses κ = 1.0
- `tests/test_distributional_ppo_clip_range_vf.py` - Uses κ = 1.0

## How to Verify

### 1. Run New Tests

```bash
pytest tests/test_quantile_huber_kappa.py -v
```

Expected: All 8 tests pass

### 2. Run Verification Script

```bash
python verify_huber_fix.py
```

Expected output:
```
✓ PASS: Quadratic independence
✓ PASS: Linear scaling
✓ PASS: Transition continuity
✓ PASS: Asymmetric weighting

✓ ALL TESTS PASSED
```

### 3. Run Existing Test Suite

```bash
pytest tests/test_distributional_ppo_quantile_loss.py -v
```

Expected: All existing tests still pass (κ = 1.0, so no change)

### 4. Integration Tests

```bash
pytest tests/ -k "quantile or distributional" -v
```

Expected: All distributional RL tests pass

## Mathematical Proof of Correctness

### Claim
The standard Huber loss formula is:
```
L^κ_H(u) = { 0.5 * u²,              if |u| <= κ
           { κ * (|u| - 0.5 * κ),   if |u| > κ
```

### Proof of Continuity
At the boundary |u| = κ:

**Quadratic side** (|u| ≤ κ):
```
L_H(κ) = 0.5 * κ²
```

**Linear side** (|u| > κ):
```
L_H(κ) = κ * (κ - 0.5 * κ) = κ * 0.5 * κ = 0.5 * κ²
```

Both sides equal → **continuous** ✓

### Proof of Gradient Continuity
**Quadratic side**:
```
dL/du = u
At u = κ: dL/du = κ
```

**Linear side**:
```
dL/du = κ * sign(u)
At u = κ⁺: dL/du = κ * 1 = κ
```

Both sides equal → **smooth** ✓

### Why Division by κ is Wrong

If we divide by κ:
```
L̃^κ_H(u) = L^κ_H(u) / κ = { 0.5 * u² / κ,    if |u| <= κ
                            { |u| - 0.5 * κ,   if |u| > κ
```

**Gradient in quadratic region**:
```
dL̃/du = u / κ
```

This means:
- When κ = 2: gradient is **half** as large
- When κ = 0.5: gradient is **twice** as large

This is **non-standard** and makes the loss behavior depend on κ in unexpected ways.

## Conclusion

The fix removes an incorrect normalization that:
1. Did not match the standard QR-DQN formula
2. Did not match reference implementations
3. Would cause problems if κ ≠ 1.0
4. Has no effect on existing models (κ = 1.0 everywhere)

The implementation now correctly follows the standard formula from Dabney et al., 2018.

## References

1. Dabney, W., Rowland, M., Bellemare, M. G., & Munos, R. (2018). **Distributional Reinforcement Learning with Quantile Regression**. *AAAI Conference on Artificial Intelligence*. https://arxiv.org/abs/1710.10044

2. Dabney, W., Ostrovski, G., Silver, D., & Munos, R. (2018). **Implicit Quantile Networks for Distributional Reinforcement Learning**. *International Conference on Machine Learning (ICML)*. https://arxiv.org/abs/1806.06923

3. Stable-Baselines3 Team. **QR-DQN Implementation**. https://github.com/Stable-Baselines-Team/stable-baselines3-contrib

4. Yang, G., et al. (2024). **A Robust Quantile Huber Loss with Interpretable Parameter Adjustment in Distributional Reinforcement Learning**. https://arxiv.org/abs/2401.02325

---

**Fix Date**: 2025-11-17
**Author**: Claude
**Status**: Complete
