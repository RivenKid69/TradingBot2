# VGS Parameter Tracking Bug Fix (Bug #9)

## Problem Summary

After `model.load()`, VGS (Variance Gradient Scaler) tracked **copies** of policy parameters instead of the actual current policy parameters. This caused VGS gradient scaling to have NO EFFECT on training after loading a checkpoint.

### Root Causes

1. **VGS pickled parameter references**: `variance_gradient_scaler.py:__getstate__()` included `_parameters` in pickle state, which became stale copies after unpickling
2. **Early return in setup**: `_setup_dependent_components()` returned early due to `_setup_complete=True` before VGS could be initialized with loaded policy parameters
3. **Generator exhaustion**: Using `self.policy.parameters()` (a generator) multiple times could lead to empty parameter lists

## Fixes Applied

### 1. variance_gradient_scaler.py

**`__getstate__()`** - Do NOT pickle `_parameters`:
```python
state.pop("_parameters", None)  # Will be relinked via update_parameters() after load
```

**`__setstate__()`** - Initialize `_parameters` to None:
```python
if not hasattr(self, "_parameters"):
    self._parameters = None
```

### 2. distributional_ppo.py

**`_setup_dependent_components()`** - Early return if policy not available:
```python
if not hasattr(self, "policy") or self.policy is None:
    logger.info("VGS setup skipped - policy not yet available")
    self._variance_gradient_scaler = None
    return  # Do NOT mark setup as complete
```

**`_setup_dependent_components()`** - Always call `update_parameters()`:
```python
# Create VGS
self._variance_gradient_scaler = VarianceGradientScaler(
    parameters=self.policy.parameters(), ...
)
# Restore state
if vgs_saved_state:
    self._variance_gradient_scaler.load_state_dict(vgs_saved_state)
# CRITICAL: Relink parameters to current policy
self._variance_gradient_scaler.update_parameters(self.policy.parameters())
```

**`load()`** - Reset `_setup_complete` before calling setup:
```python
if isinstance(model, DistributionalPPO):
    model._setup_complete = False  # Force re-initialization
    model._setup_dependent_components()
```

## Test Results

✅ Before fix: VGS tracked **0/21** correct parameters after load
✅ After fix: VGS tracks **21/21** correct parameters after load

## Files Modified

- `variance_gradient_scaler.py`: `__getstate__()`, `__setstate__()`
- `distributional_ppo.py`: `_setup_dependent_components()`, `load()`
