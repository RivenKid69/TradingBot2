# Regression Prevention Checklist - Bug Fixes 2025-11-22

## Overview

This checklist ensures that bug fixes from 2025-11-22 are not accidentally reverted or broken.
**ALWAYS** run this checklist before:
- Modifying PBT scheduler logic
- Changing quantile value head implementation
- Refactoring SA-PPO epsilon schedule
- Making changes to distributional PPO

---

## ✅ Mandatory Tests

### Core Regression Tests
**Run after ANY changes to related code:**

```bash
# Bug fixes verification (14 tests - MUST ALL PASS)
pytest tests/test_bug_fixes_2025_11_22.py -v

# Expected output:
# TestSAPPOEpsilonSchedule (3 tests) ........... PASSED
# TestPBTDeadlockPrevention (4 tests) ......... PASSED
# TestQuantileMonotonicity (6 tests) .......... PASSED
# TestBugFixesIntegration (1 test) ............ PASSED
# 14 passed ✅
```

**If ANY test fails → STOP and investigate before proceeding!**

---

## 🔍 Code Review Checklist

### BUG #1: SA-PPO Epsilon Schedule

**Files to watch**: `adversarial/sa_ppo.py`

**❌ NEVER do this:**
```python
# WRONG: Hardcoded max_updates
max_updates = 1000  # ❌ BAD
epsilon = epsilon_init + (epsilon_final - epsilon_init) * (update_count / 1000)  # ❌ BAD
```

**✅ ALWAYS verify:**
```python
# CORRECT: Compute from total_timesteps
if self.config.max_updates is None:
    if total_timesteps is not None and n_steps is not None:
        self._max_updates = total_timesteps // n_steps  # ✅ GOOD
    else:
        self._max_updates = 10000  # ✅ GOOD (fallback, NOT 1000!)
else:
    self._max_updates = self.config.max_updates  # ✅ GOOD (configurable)

# Use self._max_updates in schedule computation
progress = min(1.0, self._update_count / self._max_updates)  # ✅ GOOD
```

**Checklist:**
- [ ] `max_updates` computed from `total_timesteps // n_steps` when available?
- [ ] Fallback value is **10000** (not 1000)?
- [ ] `max_updates` can be explicitly configured?
- [ ] All epsilon schedule methods use `self._max_updates`?
- [ ] Tests pass: `pytest tests/test_bug_fixes_2025_11_22.py::TestSAPPOEpsilonSchedule -v`

---

### BUG #2: PBT Ready Percentage Deadlock

**Files to watch**: `adversarial/pbt_scheduler.py`

**❌ NEVER do this:**
```python
# WRONG: No timeout or fallback
if ready_count < required_count:
    logger.debug("Not enough ready members")  # ❌ BAD (silent failure)
    return None, member.hyperparams, None
```

**✅ ALWAYS verify:**
```python
# CORRECT: Timeout + fallback mechanism
if ready_count < required_count:
    self._failed_ready_checks += 1  # ✅ GOOD (track failures)

    # Fallback after max_wait
    if self._failed_ready_checks >= self.config.ready_check_max_wait and ready_count >= min_count:
        logger.warning(f"PBT deadlock prevention activated...")  # ✅ GOOD (visible warning)
        self._failed_ready_checks = 0  # ✅ GOOD (reset)
    else:
        # Improved logging
        if self._failed_ready_checks == 1:
            logger.info(...)  # ✅ GOOD (first failure at INFO)
        elif self._failed_ready_checks % 5 == 0:
            logger.warning(...)  # ✅ GOOD (periodic WARNING)
        return None, member.hyperparams, None
else:
    # Reset counter when sufficient members ready
    if self._failed_ready_checks > 0:
        logger.info(f"Resuming normal PBT...")  # ✅ GOOD
        self._failed_ready_checks = 0  # ✅ GOOD
```

**Checklist:**
- [ ] `PBTConfig` has `min_ready_members` (default: 2)?
- [ ] `PBTConfig` has `ready_check_max_wait` (default: 10)?
- [ ] `PBTScheduler.__init__()` initializes `self._failed_ready_checks = 0`?
- [ ] Fallback mechanism activates when `failed_ready_checks >= ready_check_max_wait AND ready_count >= min_ready_members`?
- [ ] Counter increments on each failed check?
- [ ] Counter resets to 0 after fallback OR when sufficient members ready?
- [ ] Logging levels: INFO (first), WARNING (every 5th + fallback)?
- [ ] `get_stats()` returns `pbt/failed_ready_checks` metric?
- [ ] Tests pass: `pytest tests/test_bug_fixes_2025_11_22.py::TestPBTDeadlockPrevention -v`

**Production Monitoring:**
```python
# Monitor this metric in production
stats = scheduler.get_stats()
if stats["pbt/failed_ready_checks"] > 5:
    # ALERT: Investigate worker health!
    pass
```

---

### BUG #3: Quantile Monotonicity Not Enforced

**Files to watch**: `custom_policy_patch1.py` (QuantileValueHead)

**❌ NEVER do this:**
```python
# WRONG: Remove sorting option
class QuantileValueHead(nn.Module):
    def __init__(self, input_dim, num_quantiles, huber_kappa):  # ❌ BAD (no enforce_monotonicity param)
        # ...

    def forward(self, latent):
        quantiles = self.linear(latent)
        # ❌ BAD: No optional sorting
        return quantiles
```

**✅ ALWAYS verify:**
```python
# CORRECT: Optional monotonicity enforcement
class QuantileValueHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_quantiles: int,
        huber_kappa: float,
        enforce_monotonicity: bool = False,  # ✅ GOOD (optional, default False)
    ):
        super().__init__()
        # ... existing init ...
        self.enforce_monotonicity = enforce_monotonicity  # ✅ GOOD (store parameter)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        quantiles = self.linear(latent)

        # ✅ GOOD: Optional sorting (differentiable!)
        if self.enforce_monotonicity:
            quantiles = torch.sort(quantiles, dim=1)[0]

        return quantiles
```

**Checklist:**
- [ ] `QuantileValueHead.__init__()` has `enforce_monotonicity: bool = False` parameter?
- [ ] Default is `False` (backward compatibility)?
- [ ] `self.enforce_monotonicity` stored as instance variable?
- [ ] `forward()` applies `torch.sort()` when `enforce_monotonicity=True`?
- [ ] Sorting uses `dim=1` (per-batch)?
- [ ] Only values extracted: `torch.sort(...)[0]` (not indices)?
- [ ] `CustomActorCriticPolicy` passes `enforce_monotonicity` parameter?
- [ ] Configuration supports `critic.enforce_monotonicity` in YAML?
- [ ] Both `quantile_head` and `quantile_head_2` (Twin Critics) use same parameter?
- [ ] Tests pass: `pytest tests/test_bug_fixes_2025_11_22.py::TestQuantileMonotonicity -v`

**Configuration Example:**
```yaml
arch_params:
  critic:
    distributional: true
    categorical: false
    num_quantiles: 21
    huber_kappa: 1.0
    enforce_monotonicity: false  # ✅ GOOD (default)
    # enforce_monotonicity: true  # Use for CVaR-critical applications
```

---

## 🧪 Integration Testing

**After changes to multiple components:**

```bash
# Run integration test
pytest tests/test_bug_fixes_2025_11_22.py::TestBugFixesIntegration::test_pbt_with_quantile_monotonicity -v

# This tests:
# - PBT scheduler with deadlock prevention
# - Quantile heads with monotonicity enforcement
# - Integration between fixes
```

---

## 📊 Production Monitoring

**Monitor these metrics in production:**

```python
# PBT health check
pbt_stats = scheduler.get_stats()
if pbt_stats["pbt/failed_ready_checks"] > 5:
    # WARNING: Potential worker issues
    logger.warning("PBT experiencing consecutive failed ready checks")

if pbt_stats["pbt/ready_members"] < 0.5 * pbt_stats["pbt/population_size"]:
    # CRITICAL: More than half workers down
    logger.error("PBT population severely degraded")
```

**Tensorboard metrics:**
- `pbt/failed_ready_checks`: Should be 0 most of the time
- `pbt/ready_members`: Should be close to `population_size`
- `sa_ppo/current_epsilon`: Should decrease smoothly over full training

**Quantile monotonicity check (optional):**
```python
# During development/debugging
quantiles = model.policy.quantile_head(latent)
for i in range(quantiles.size(0)):
    is_monotonic = torch.all(quantiles[i, :-1] <= quantiles[i, 1:])
    if not is_monotonic and not enforce_monotonicity:
        logger.debug(f"Non-monotonic quantiles detected (sample {i})")
        # This is OK if enforce_monotonicity=False
```

---

## 🔄 Backward Compatibility Verification

**Before merging changes:**

```bash
# Verify no breaking changes
pytest tests/test_bug_fixes_2025_11_22.py -v  # All tests should pass

# Verify old configs still work (no enforce_monotonicity parameter)
python -c "
from custom_policy_patch1 import QuantileValueHead
head = QuantileValueHead(input_dim=64, num_quantiles=21, huber_kappa=1.0)
assert head.enforce_monotonicity == False  # Default should be False
print('✅ Backward compatibility verified')
"

# Verify old PBT configs still work (no new parameters)
python -c "
from adversarial.pbt_scheduler import PBTConfig
config = PBTConfig(population_size=10)
assert config.min_ready_members == 2  # Default should be 2
assert config.ready_check_max_wait == 10  # Default should be 10
print('✅ PBT backward compatibility verified')
"
```

---

## 📝 Documentation Verification

**Before finalizing:**

- [ ] `BUG_FIXES_REPORT_2025_11_22.md` exists and is up to date?
- [ ] `CLAUDE.md` updated with bug fix summary?
- [ ] `REGRESSION_PREVENTION_CHECKLIST_2025_11_22.md` (this file) exists?
- [ ] Code comments reference bug fix reports where appropriate?
- [ ] Configuration examples show new parameters with defaults?

---

## ⚠️ Red Flags - STOP and Investigate

**If you see ANY of these patterns, STOP and review:**

1. **Hardcoded `max_updates = 1000`** in SA-PPO
   - ❌ This is the old bug!
   - ✅ Should compute from `total_timesteps // n_steps`

2. **No `_failed_ready_checks` tracking** in PBT
   - ❌ Deadlock prevention removed!
   - ✅ Should track consecutive failed checks

3. **No `enforce_monotonicity` parameter** in QuantileValueHead
   - ❌ Optional sorting removed!
   - ✅ Should support optional `torch.sort()`

4. **Silent DEBUG logging** for PBT ready check failures
   - ❌ Makes deadlock invisible!
   - ✅ Should use INFO/WARNING levels

5. **Removing `torch.sort()` call** in QuantileValueHead.forward()
   - ❌ Optional monotonicity enforcement removed!
   - ✅ Should keep conditional sorting

---

## 🚨 Emergency Rollback

**If bugs are reintroduced:**

```bash
# 1. Identify which bug was reintroduced
pytest tests/test_bug_fixes_2025_11_22.py -v

# 2. Check git history for the fix
git log --all --grep="BUG #2" --oneline  # Example for BUG #2

# 3. Review the original fix
git show <commit-hash>

# 4. Compare current code with fix
git diff <commit-hash> -- adversarial/pbt_scheduler.py

# 5. Restore the fix or contact team lead
```

---

## ✅ Final Checklist

**Before pushing changes:**

- [ ] All 14 bug fix tests pass?
- [ ] No hardcoded `max_updates = 1000` in SA-PPO?
- [ ] PBT has `min_ready_members` and `ready_check_max_wait`?
- [ ] PBT tracks `_failed_ready_checks` and implements fallback?
- [ ] QuantileValueHead has `enforce_monotonicity` parameter (default: False)?
- [ ] QuantileValueHead.forward() applies optional `torch.sort()`?
- [ ] Backward compatibility verified (old configs work)?
- [ ] Documentation updated (CLAUDE.md, BUG_FIXES_REPORT)?
- [ ] Code review by teammate (if available)?

**If all checkboxes are ✅ → Safe to merge!**

---

## 📚 References

- **Primary Report**: [BUG_FIXES_REPORT_2025_11_22.md](BUG_FIXES_REPORT_2025_11_22.md)
- **Test Suite**: [tests/test_bug_fixes_2025_11_22.py](tests/test_bug_fixes_2025_11_22.py)
- **Main Documentation**: [CLAUDE.md](CLAUDE.md)

**Research References:**
- SA-PPO: Zhang et al. (2020) "Robust Deep RL against Adversarial Perturbations"
- PBT: Jaderberg et al. (2017) "Population Based Training of Neural Networks"
- Quantile Regression: Koenker & Bassett (1978), Dabney et al. (2018) "Implicit Quantile Networks"

---

**Last Updated**: 2025-11-22
**Maintainer**: AI-Powered Quantitative Research Platform Team
**Status**: ✅ ACTIVE - Use this checklist for all future changes
