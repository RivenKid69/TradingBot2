# Bug Fixes Summary

**Date:** 2025-11-20
**Status:** All 3 critical bugs FIXED and verified

---

## Bug #2: optimizer_kwargs['lr'] Ignored (FIXED ✓)

### Problem
When passing custom learning rate via `optimizer_kwargs={'lr': 0.005}`, the value was ignored and default learning rate (3e-4) was always used.

### Root Cause
Two issues:
1. In `CustomActorCriticPolicy.__init__()`, `optimizer_kwargs` was not being passed to the policy
2. In `DistributionalPPO.__init__()`, the optimizer was being recreated after policy initialization, using `lr_schedule(1.0)` instead of the user-provided lr

### Fix Locations

**File: `distributional_ppo.py`**

1. Lines 5599-5640: Inject `optimizer_kwargs` into `policy_kwargs` before super().__init__()
2. Lines 5768-5776: Check if user provided custom lr in optimizer_kwargs and use it for base_lr
3. Lines 5820-5822: Remove 'lr' from optimizer_kwargs before creating optimizer (since lr is set in param_groups)

**File: `custom_policy_patch1.py`**

1. Lines 284-295: Store `_pending_optimizer_kwargs` before super().__init__() and temporarily remove 'lr' to avoid conflict
2. Lines 646-661: Use `_pending_optimizer_kwargs` in `_setup_custom_optimizer()` to respect user-provided lr

### Verification
```bash
python verify_critical_bug_2_lr_override.py
# Result: All 4 test cases passed
```

---

## Bug #1: Twin Critics Tensor Dimension Mismatch (FIXED ✓)

### Problem
When using Twin Critics with categorical value head, dimension mismatch between `target_distribution` and `log_predictions` tensors during loss computation:
- `target_distribution`: shape `[32, 51]` (batch × atoms)
- `log_predictions_1`: shape `[58, 51]` (different batch size!)

Error: `RuntimeError: The size of tensor a (32) must match the size of tensor b (58) at non-singleton dimension 0`

### Root Cause
When `valid_indices` is used to filter samples, `target_distribution_selected` is filtered but `latent_vf` (cached from forward pass) remains the full batch. This causes dimension mismatch when computing value logits.

### Fix Locations

**File: `distributional_ppo.py`**

1. Lines 9472-9476: Select `latent_vf` using `valid_indices` before passing to `_twin_critics_loss()` (categorical mode)
2. Lines 9149-9153: Select `latent_vf` using `valid_indices` before passing to `_twin_critics_loss()` (quantile mode)

### Verification
```bash
python verify_critical_bug_1_twin_critics.py
# Result: Training completed without errors
```

---

## Bug #3: SimpleDummyEnv Missing gymnasium.Env Inheritance (FIXED ✓)

### Problem
`SimpleDummyEnv` in `tests/test_twin_critics_integration.py` did not inherit from `gymnasium.Env`, causing ValueError when wrapping with `DummyVecEnv`:

```
ValueError: The environment is of type <class 'SimpleDummyEnv'>,
not a Gymnasium environment.
```

### Root Cause
Stable-Baselines3 `DummyVecEnv` validates environment type and expects either `gymnasium.Env` or `gym.Env` (old OpenAI Gym). `SimpleDummyEnv` had neither.

### Fix Locations

**File: `tests/test_twin_critics_integration.py`**

1. Line 14: Import `gymnasium`
2. Line 21: Change `class SimpleDummyEnv:` → `class SimpleDummyEnv(gymnasium.Env):`
3. Line 25: Add `super().__init__()` call

### Verification
```bash
python test_bug3_fix.py
# Result: SimpleDummyEnv now properly inherits from gymnasium.Env
```

---

## Files Modified

### Production Code
1. `distributional_ppo.py` - 3 changes for Bug #2, 2 changes for Bug #1
2. `custom_policy_patch1.py` - 2 changes for Bug #2

### Test Code
1. `tests/test_twin_critics_integration.py` - 1 change for Bug #3

---

## Verification Tests Created

1. `verify_critical_bug_1_twin_critics.py` - Isolated test for Bug #1
2. `verify_critical_bug_2_lr_override.py` - Isolated test for Bug #2
3. `verify_critical_bug_3_dummy_env.py` - Isolated test for Bug #3
4. `CRITICAL_BUGS_VERIFICATION_REPORT.md` - Comprehensive verification report

---

## Next Steps

All 3 critical bugs are now fixed and verified. The fixes are ready to be committed:

```bash
git add distributional_ppo.py custom_policy_patch1.py tests/test_twin_critics_integration.py
git commit -m "$(cat <<'EOF'
fix: Fix integration issues with UPGD, PBT, and adversarial training

Fixed 3 critical bugs blocking UPGD Optimizer, Population-Based Training,
Twin Critics, and Variance Gradient Scaling integration:

1. Bug #2 (CRITICAL): optimizer_kwargs['lr'] was ignored
   - User-provided learning rate now correctly applied
   - Fixed optimizer recreation in DistributionalPPO.__init__()
   - Fixed optimizer_kwargs passing to CustomActorCriticPolicy

2. Bug #1 (CRITICAL): Twin Critics tensor dimension mismatch
   - Fixed batch dimension mismatch in _twin_critics_loss()
   - latent_vf now properly filtered by valid_indices
   - Applies to both categorical and quantile modes

3. Bug #3 (MEDIUM): SimpleDummyEnv missing gymnasium.Env inheritance
   - Test environment now properly inherits from gymnasium.Env
   - Fixes ValueError in DummyVecEnv wrapper

All bugs verified with isolated reproducible tests.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```
