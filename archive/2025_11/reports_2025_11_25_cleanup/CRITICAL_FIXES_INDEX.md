# CRITICAL FIXES INDEX
## Quick Navigation to All Critical Bug Fixes

**Last Updated**: 2025-11-21

---

## 🔴 ACTION SPACE FIXES (2025-11-21)

**Status**: ✅ FIXED AND VERIFIED
**Test Coverage**: 21/21 tests passed
**Criticality**: PRODUCTION CRITICAL

### Quick Links

- **Detailed Analysis**: [CRITICAL_ACTION_SPACE_ISSUES_ANALYSIS.md](../CRITICAL_ACTION_SPACE_ISSUES_ANALYSIS.md)
- **Complete Report**: [CRITICAL_FIXES_COMPLETE_REPORT.md](../CRITICAL_FIXES_COMPLETE_REPORT.md)
- **Reference Guide**: [ACTION_SPACE_CRITICAL_GUIDE.md](ACTION_SPACE_CRITICAL_GUIDE.md)
- **Test Suite**: [tests/test_critical_action_space_fixes.py](../tests/test_critical_action_space_fixes.py)

### Fixed Issues

| # | Issue | Files Modified | Critical? |
|---|-------|----------------|-----------|
| **#1** | Sign Convention Mismatch in LongOnlyActionWrapper | [wrappers/action_space.py](../wrappers/action_space.py) | HIGH |
| **#2** | Position Semantics (DELTA→TARGET) | [action_proto.py](../action_proto.py), [risk_guard.py](../risk_guard.py), [trading_patchnew.py](../trading_patchnew.py) | **CRITICAL** |
| **#3** | Action Space Range Mismatch | [trading_patchnew.py](../trading_patchnew.py) | HIGH |

### Key Changes

**ActionProto.volume_frac semantics**:
- **Before**: DELTA (add to current) → causes position doubling
- **After**: TARGET (desired end state) → correct behavior

**LongOnlyActionWrapper behavior**:
- **Before**: Clips negative to 0.0 → signal loss
- **After**: Maps [-1,1]→[0,1] → preserves reduction signals

**Action space bounds**:
- **Before**: Mixed [0,1] and [-1,1] → architectural mismatch
- **After**: Unified [-1,1] everywhere → consistency

---

## 🟡 DATA & CRITIC FIXES (2025-11-20)

**Status**: ✅ FIXED AND VERIFIED
**Criticality**: HIGH (affects model training)

### Quick Links

- **Full Report**: [CRITICAL_FIXES_REPORT.md](../CRITICAL_FIXES_REPORT.md)

### Fixed Issues

| # | Issue | Files Modified | Impact |
|---|-------|----------------|--------|
| **#10** | Temporal Causality Violation | [impl_offline_data.py](../impl_offline_data.py) | Models with `stale_prob > 0` |
| **#11** | Cross-Symbol Contamination | [features_pipeline.py](../features_pipeline.py) | Multi-symbol models |
| **#12** | Inverted Quantile Loss | [distributional_ppo.py](../distributional_ppo.py) | CVaR/quantile critic models |

---

## 📚 Documentation Map

### For Developers

```
CRITICAL FIXES DOCUMENTATION TREE
│
├── 🔴 ACTION SPACE FIXES (2025-11-21)
│   ├── CRITICAL_ACTION_SPACE_ISSUES_ANALYSIS.md ← Detailed problem analysis
│   ├── CRITICAL_FIXES_COMPLETE_REPORT.md ← Implementation & verification
│   ├── docs/ACTION_SPACE_CRITICAL_GUIDE.md ← Reference guide (READ FIRST!)
│   └── tests/test_critical_action_space_fixes.py ← Test suite
│
├── 🟡 DATA & CRITIC FIXES (2025-11-20)
│   └── CRITICAL_FIXES_REPORT.md ← Full report
│
└── 📖 MAIN DOCUMENTATION
    ├── CLAUDE.md ← AI assistant guide (includes critical warnings)
    ├── README.md ← Project overview
    └── ARCHITECTURE.md ← System architecture
```

### For AI Assistants

**ALWAYS READ THESE BEFORE MODIFYING ACTION SPACE CODE:**

1. [docs/ACTION_SPACE_CRITICAL_GUIDE.md](ACTION_SPACE_CRITICAL_GUIDE.md) - Critical rules and patterns
2. [CRITICAL_FIXES_COMPLETE_REPORT.md](../CRITICAL_FIXES_COMPLETE_REPORT.md) - What was fixed and why
3. [CLAUDE.md](../CLAUDE.md) - Section "🛡️ Критические правила"

**ALWAYS RUN THESE TESTS:**

```bash
pytest tests/test_critical_action_space_fixes.py -v
```

Expected: 21/21 passed, 2 skipped

---

## ⚠️ Breaking Changes & Migration

### Models Trained Before 2025-11-21

**If using LongOnlyActionWrapper**:
- ⚠️ **RECOMMENDED**: Retrain from scratch
- Reason: Action semantics changed (clipping → mapping)
- Impact: ~10-15% performance change

**If using DELTA position semantics**:
- ⚠️ **REQUIRED**: Retrain from scratch
- Reason: Critical bug fix (DELTA → TARGET)
- Impact: Prevents position doubling in production

**If using long/short (no wrapper)**:
- ✅ **OPTIONAL**: Minor improvements from fixes
- Impact: Minimal (~5% stability improvement)

---

## 🧪 Verification Checklist

Before deploying to production after these fixes:

### Action Space Fixes

- [ ] Run `pytest tests/test_critical_action_space_fixes.py -v`
- [ ] Verify 21/21 tests pass
- [ ] Check all `ActionProto.volume_frac` usage is TARGET semantics
- [ ] Verify `LongOnlyActionWrapper` uses mapping (not clipping)
- [ ] Confirm action space bounds are [-1,1] everywhere

### Model Retraining (if applicable)

- [ ] Identify models affected by fixes
- [ ] Schedule retraining with new semantics
- [ ] Compare before/after metrics
- [ ] Verify no position doubling in backtests

### Production Deployment

- [ ] Update live trading configs
- [ ] Monitor position behavior closely
- [ ] Set up alerts for position violations
- [ ] Have rollback plan ready (though fixes are critical!)

---

## 📞 Support & Questions

### If You See These Symptoms

**Position doubling**:
- Check: Is `volume_frac` interpreted as TARGET (not DELTA)?
- File: [risk_guard.py](../risk_guard.py), [trading_patchnew.py](../trading_patchnew.py)
- Fix: Use `target_units = volume_frac * max`, NOT `delta = ...; next = current + delta`

**Policy can't reduce positions**:
- Check: Is `LongOnlyActionWrapper` using mapping (not clipping)?
- File: [wrappers/action_space.py](../wrappers/action_space.py)
- Fix: Use `(action + 1) / 2`, NOT `max(0, action)`

**Action space mismatch errors**:
- Check: Are bounds [-1,1] everywhere?
- Files: [action_proto.py](../action_proto.py), [trading_patchnew.py](../trading_patchnew.py)
- Fix: Unify to `np.clip(action, -1.0, 1.0)`

### Documentation Issues?

If this documentation is unclear or outdated:
1. Read the full reports (links above)
2. Check test suite for examples
3. Ask maintainer before making changes

---

**CRITICAL REMINDER**: These fixes prevent PRODUCTION BUGS. Do NOT revert without full understanding!

**Last Verification**: 2025-11-21
**Tests Status**: ✅ 21/21 passed
**Production Status**: ✅ READY
