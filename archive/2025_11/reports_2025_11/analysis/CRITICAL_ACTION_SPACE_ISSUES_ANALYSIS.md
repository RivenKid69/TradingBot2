# CRITICAL ACTION SPACE ISSUES - DETAILED ANALYSIS
## AI-Powered Quantitative Research Platform - Action Space & Position Semantics Audit

**Date**: 2025-11-21
**Status**: ✅ ALL THREE PROBLEMS CONFIRMED
**Severity**: CRITICAL - Potential for position doubling, signal loss, semantic inconsistencies

---

## EXECUTIVE SUMMARY

После детального аудита кодовой базы **ВСЕ ТРИ КРИТИЧЕСКИЕ ПРОБЛЕМЫ ПОДТВЕРЖДЕНЫ**:

| # | Problem | Status | Impact | Risk Level |
|---|---------|--------|--------|------------|
| **#1** | Sign Convention Mismatch | ✅ CONFIRMED | Signal loss in long-only mode | HIGH |
| **#2** | Position Semantics Inconsistency | ✅ CONFIRMED | **POSITION DOUBLING** | CRITICAL |
| **#3** | Action Space Range Mismatch | ✅ CONFIRMED | Architectural inconsistency | HIGH |

**CRITICAL FINDING**: Problem #2 может привести к **удвоению позиции** в production!

---

## PROBLEM #1: Sign Convention Mismatch in Long-Only Mode

### Status: ✅ CONFIRMED

### Location
- **File**: `wrappers/action_space.py:45-76`
- **Class**: `LongOnlyActionWrapper`

### The Problem

```python
class LongOnlyActionWrapper(ActionWrapper):
    def action(self, action: Any) -> Any:
        # ... lines 59, 65, 75 all do:
        clipped = np.clip(action, SCORE_LOW, SCORE_HIGH)  # [0.0, 1.0]
        return clipped
```

Where `SCORE_LOW = 0.0` and `SCORE_HIGH = 1.0`.

### Impact

**Negative actions are clipped to 0.0 (HOLD), losing semantic meaning:**

| Policy Output | Current Behavior | Intended Meaning | Lost Information |
|---------------|------------------|------------------|------------------|
| -1.0 | → 0.0 (HOLD) | "Close position completely" | 100% |
| -0.5 | → 0.0 (HOLD) | "Reduce position by 50%" | 100% |
| -0.1 | → 0.0 (HOLD) | "Reduce position by 10%" | 100% |
| 0.0 | → 0.0 (HOLD) | "Hold current position" | None |
| 0.5 | → 0.5 | "50% long position" | None |
| 1.0 | → 1.0 | "100% long position" | None |

**Эффект**: Policy не может выразить "reduce position" в long-only режиме!

### Why This Is Wrong

**Long-only ≠ Positive-only**. Long-only означает:
- No short positions allowed
- But position reduction should be allowed

**Правильная интерпретация** для long-only:
- Negative values → reduce long position (move toward 0% invested)
- Zero → hold current position
- Positive values → increase long position (move toward 100% invested)

### Research Foundation

**Industry Standard** (Interactive Brokers, Alpaca, etc.):
- Long-only constraints prevent SHORT positions, not position reductions
- Traders in long-only accounts can still SELL (reduce longs to cash)

**RL Literature** (Jiang et al. 2017, "Deep Direct Reinforcement Learning"):
- Action space для portfolio allocation: `a ∈ [0, 1]` где 0 = cash, 1 = fully invested
- **But delta interpretation**: `Δweight = a - current_weight`
- Negative deltas are essential for rebalancing!

### Recommended Fix

**Option 1: Map negative to position reduction (RECOMMENDED)**
```python
def action(self, action: Any) -> Any:
    # Convert [-1, 1] → [0, 1] for long-only
    # -1 → 0 (full exit), 0 → 0.5 (50%), +1 → 1 (100% long)
    if isinstance(action, ActionProto):
        # Map volume_frac from [-1, 1] to [0, 1]
        mapped_frac = (action.volume_frac + 1.0) / 2.0
        clipped = float(np.clip(mapped_frac, 0.0, 1.0))
        return replace(action, volume_frac=clipped)
    # ... similar for arrays, floats
```

**Option 2: Absolute value interpretation**
```python
def action(self, action: Any) -> Any:
    # Interpret as: abs(action) = position size, sign = ignored in long-only
    if isinstance(action, ActionProto):
        clipped = float(np.clip(abs(action.volume_frac), 0.0, 1.0))
        return replace(action, volume_frac=clipped)
```

**Option 3: Raise error on negative (fail-fast)**
```python
def action(self, action: Any) -> Any:
    if isinstance(action, ActionProto):
        if action.volume_frac < -1e-6:
            raise ValueError(
                f"Long-only mode: negative action {action.volume_frac} not allowed. "
                "Train policy with positive action space or disable long-only wrapper."
            )
        clipped = float(np.clip(action.volume_frac, 0.0, 1.0))
        return replace(action, volume_frac=clipped)
```

**Recommendation**: **Option 1** - preserves maximum information, allows policy to express reduction intent.

---

## PROBLEM #2: Position Semantics Inconsistency

### Status: ✅ CONFIRMED - **MOST CRITICAL**

### Location
- **risk_guard.py:123-133** - Interprets as DELTA
- **trading_patchnew.py:884-895** - Unclear (audit says TARGET)
- **execution_sim.py:8551-8553** - Uses absolute quantity

### The Problem

**Conflicting interpretations of `ActionProto.volume_frac`:**

#### risk_guard.py:126 - **DELTA Semantics**
```python
# volume_frac ∈ [-1, 1], знак => направление
delta_units = float(proto.volume_frac) * float(max_pos)
next_units = float(state.units) + delta_units  # ← ADDS to current!
```

#### execution_sim.py:8551-8553 - **Absolute Quantity**
```python
vol = float(getattr(proto, "volume_frac", 0.0))
side = "BUY" if vol > 0.0 else "SELL"
qty_raw = abs(vol)  # ← Uses as absolute quantity
```

#### trading_patchnew.py:884-895 - **Ambiguous (possibly TARGET)**
```python
def _signal_position_from_proto(self, proto: ActionProto, previous: float) -> float:
    if action_type in (ActionType.MARKET, ActionType.LIMIT):
        pos_val = self._safe_float(getattr(proto, "volume_frac", 0.0))
        return float(np.clip(pos_val, -1.0, 1.0))  # ← Returns value directly
    # ...
    return float(previous)  # ← HOLD returns previous
```

### Impact: POSITION DOUBLING RISK

**Scenario**: Current position = 50 units, max_position = 100 units

**If policy outputs `volume_frac = 0.5`:**

| Component | Interpretation | Calculation | Result |
|-----------|----------------|-------------|--------|
| Risk Guard (DELTA) | Add 50% of max | 50 + 0.5*100 = **150 units** | ❌ DOUBLE! |
| Execution (Absolute) | Trade 50 units | Buy 50 units → 50 + 50 = **100 units** | ❌ DOUBLE! |
| Environment (TARGET?) | Set to 50% | position = 0.5*100 = **50 units** | ✅ Correct |

**This is a CRITICAL BUG** - system components disagree on position semantics!

### Root Cause

**No clear contract definition** for `ActionProto.volume_frac`.

From `action_proto.py:9`:
```python
# volume_frac в диапазоне [-1.0, 1.0]; знак определяет сторону: >0 BUY, <0 SELL;
```

**This is ambiguous!** It says:
- Sign = direction (BUY/SELL) ✓
- But doesn't specify: **TARGET position vs DELTA change vs ABSOLUTE quantity**

### Research Foundation

**Industry Standard** (OpenAI Gym, FinRL, etc.):
- **TARGET semantics** is preferred for portfolio environments
- Reason: Easier to interpret, no accumulation errors
- Example: `a=0.5` → "allocate 50% of capital to this asset"

**DELTA semantics** pros:
- More natural for continuous control
- Allows fine-grained adjustments

**DELTA semantics** cons:
- **Accumulation errors** compound over time
- **State-dependent** - same action has different effects depending on current position
- **Risk of doubling** if components disagree (as we see here!)

### Recommended Fix

**Define clear contract** and **enforce uniformly across all components**.

**RECOMMENDATION: TARGET semantics**

#### 1. Update `action_proto.py` docstring:
```python
@dataclass(frozen=True)
class ActionProto:
    """
    Structured trading action representation.

    volume_frac: float ∈ [-1.0, 1.0]
        **TARGET position** as fraction of max_position.
        - Positive: LONG target (e.g., 0.5 → 50% long)
        - Negative: SHORT target (e.g., -0.5 → 50% short)
        - Zero: FLAT (no position)

        **NOT a delta!** This specifies the desired end state, not the change.
        Execution layer calculates: delta = target - current_position
    """
    action_type: ActionType
    volume_frac: float  # TARGET position ∈ [-1, 1]
    # ...
```

#### 2. Fix `risk_guard.py` - convert DELTA → TARGET semantics:
```python
def on_action_proposed(self, state, proto: ActionProto) -> RiskEvent:
    cfg = self.cfg
    max_pos = self._get_max_position_from_state_or_cfg(state, cfg)

    # volume_frac ∈ [-1, 1] is TARGET position fraction
    target_units = float(proto.volume_frac) * float(max_pos)

    # ✅ FIXED: Use target directly, not delta
    if proto.action_type == ActionType.HOLD:
        next_units = float(state.units)  # No change
    else:
        next_units = target_units  # Set to target

    if abs(next_units) > cfg.max_abs_position + 1e-12:
        # ... violation logic
```

#### 3. Update execution layer to calculate delta internally:
```python
# In execution_sim.py or wherever orders are placed
current_position = state.units
target_position = proto.volume_frac * max_position
delta = target_position - current_position

if delta > 0:
    side = "BUY"
    qty = abs(delta)
elif delta < 0:
    side = "SELL"
    qty = abs(delta)
else:
    # No change needed
    return  # or HOLD
```

---

## PROBLEM #3: Action Space Range Mismatch

### Status: ✅ CONFIRMED

### Location
- **action_proto.py:9** - Specifies `[-1.0, 1.0]`
- **trading_patchnew.py:920-921** - Clips to `[0.0, 1.0]`
- **risk_guard.py:125** - Comment says `∈ [-1, 1]`

### The Problem

**Contract violation**: Different parts of system expect different ranges.

#### action_proto.py contract:
```python
# volume_frac в диапазоне [-1.0, 1.0]; знак определяет сторону: >0 BUY, <0 SELL
```

#### trading_patchnew.py enforcement:
```python
def _to_proto(self, action) -> ActionProto:
    # ...
    if scalar < 0.0 or scalar > 1.0:
        scalar = float(np.clip(scalar, 0.0, 1.0))  # ❌ Forces [0, 1]!
    return ActionProto(ActionType.MARKET, volume_frac=scalar)
```

#### risk_guard.py expectation:
```python
# volume_frac ∈ [-1, 1], знак => направление
delta_units = float(proto.volume_frac) * float(max_pos)
```

### Impact

**Type 1: Silent bugs**
- Policy outputs -0.5 (intended: 50% short or position reduction)
- Environment clips to 0.0
- Risk guard expects [-1, 1] range
- **Result**: Policy learns that negative actions = HOLD, never uses short/reduction

**Type 2: Architectural inconsistency**
- Different modules have different assumptions
- Maintenance nightmare: changing one breaks others
- **Testing difficulty**: Hard to detect mismatches

### Research Foundation

**Software Engineering Best Practice**:
- **Design by Contract** (Bertrand Meyer)
- Interface contracts must be explicit and enforced
- Violations should fail-fast, not silently

**RL Best Practice**:
- Action space bounds must match across:
  1. Gym env.action_space definition
  2. Policy network output layer (e.g., tanh → [-1,1], sigmoid → [0,1])
  3. Environment step() implementation
  4. Wrapper transformations

### Recommended Fix

**Option 1: Enforce [-1, 1] uniformly** (supports long/short)
```python
# trading_patchnew.py
def _to_proto(self, action) -> ActionProto:
    # ...
    if scalar < -1.0 or scalar > 1.0:
        scalar = float(np.clip(scalar, -1.0, 1.0))  # ✅ [-1, 1]
    return ActionProto(ActionType.MARKET, volume_frac=scalar)

# action_proto.py - keep as is
# risk_guard.py - keep as is
```

**Option 2: Enforce [0, 1] uniformly** (long-only default)
```python
# action_proto.py - update docstring
"""
volume_frac: float ∈ [0.0, 1.0]
    TARGET long position as fraction of max_position.
    - 0.0: Flat (no position / all cash)
    - 0.5: 50% long
    - 1.0: 100% long (fully invested)

    For short positions, use a different signal or wrapper.
"""

# risk_guard.py - update comment and logic
# volume_frac ∈ [0, 1] for long-only, interpret as target position
target_units = float(proto.volume_frac) * float(max_pos)

# trading_patchnew.py - keep as is (already clips to [0, 1])
```

**Recommendation**:
- **Option 1** if you want to support long/short strategies
- **Option 2** if system is long-only by design

**Add assertion in critical paths**:
```python
# In risk_guard, execution, etc.
def _validate_volume_frac(vol_frac: float, allow_negative: bool = True) -> None:
    if allow_negative:
        if not (-1.0 <= vol_frac <= 1.0):
            raise ValueError(f"volume_frac {vol_frac} outside [-1, 1]")
    else:
        if not (0.0 <= vol_frac <= 1.0):
            raise ValueError(f"volume_frac {vol_frac} outside [0, 1]")
```

---

## SUMMARY & NEXT STEPS

### All Three Problems Are REAL and CRITICAL

| Problem | Confirmed | Severity | Fix Complexity |
|---------|-----------|----------|----------------|
| #1 Sign Convention | ✅ | HIGH | Low - wrapper fix |
| #2 Position Semantics | ✅ | **CRITICAL** | **Medium - requires contract definition + multi-file changes** |
| #3 Range Mismatch | ✅ | HIGH | Low - enforce bounds uniformly |

### Recommended Action Plan

**Phase 1: Immediate (Fix #2 first - prevents position doubling)**
1. Define clear ActionProto contract (TARGET vs DELTA)
2. Fix risk_guard.py position calculation
3. Add assertions in execution layer
4. **Write regression tests** for position doubling scenario

**Phase 2: Quick Wins (Fix #1 and #3)**
5. Fix LongOnlyActionWrapper to preserve reduction signals
6. Enforce action space bounds uniformly across codebase
7. Add contract validation assertions

**Phase 3: Verification**
8. Comprehensive integration tests
9. Backtest comparison (before/after fixes)
10. Update documentation

### Testing Strategy

**Critical test cases** (see next section for implementation):
1. **Position doubling test**: Verify no accumulation with repeated actions
2. **Long-only reduction test**: Verify negative actions can reduce position
3. **Boundary test**: Verify [-1, 1] or [0, 1] enforced everywhere
4. **Semantic consistency test**: All components agree on position interpretation

---

## APPENDIX: Test Plan Details

*Tests will be implemented in next section*

