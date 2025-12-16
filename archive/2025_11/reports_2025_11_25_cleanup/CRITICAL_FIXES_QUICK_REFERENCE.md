# Critical Fixes Quick Reference

**Project**: AI-Powered Quantitative Research Platform
**Last Updated**: 2025-11-20

---

## ⚡ One-Minute Summary

**Three critical bugs fixed on 2025-11-20:**

| Bug | Impact | Fixed | Retraining? |
|-----|--------|-------|-------------|
| **#10** Temporal causality | Stale data had wrong timestamp | ✅ Yes | ⚠️ If `stale_prob > 0` |
| **#11** Cross-symbol contamination | Features leaked between symbols | ✅ Yes | ⚠️ If multi-symbol |
| **#12** Inverted quantile loss | Wrong penalty asymmetry | ✅ Yes | 🔴 **STRONGLY** if quantile critic |

**All fixes are active by default. New training runs are safe.**

---

## 🎯 Do I Need to Retrain?

### Check Your Model

Run this to check if your model is affected:

```python
# Check if you need to retrain
import torch

# Load your model
model = torch.load("path/to/model.zip", map_location="cpu")

# Check 1: Was data degradation used?
config = model.get("config", {})
stale_prob = config.get("data_degradation", {}).get("stale_prob", 0.0)
print(f"Stale prob: {stale_prob}")
if stale_prob > 0:
    print("⚠️  BUG #10: Your model was trained with stale data - CONSIDER RETRAINING")

# Check 2: Multi-symbol training?
symbols = config.get("symbols", [])
print(f"Symbols: {symbols}")
if len(symbols) > 1:
    print("⚠️  BUG #11: Multi-symbol training detected - CONSIDER RETRAINING")

# Check 3: Quantile critic?
policy_kwargs = config.get("policy_kwargs", {})
uses_quantile = policy_kwargs.get("uses_quantile_value_head", False)
print(f"Quantile critic: {uses_quantile}")
if uses_quantile:
    print("🔴 BUG #12: Quantile critic detected - STRONGLY RECOMMEND RETRAINING")
```

---

## 🔧 Quick Fixes Applied

### Fix #10: Temporal Causality ([impl_offline_data.py:139-153](../impl_offline_data.py#L139-L153))

```python
# Before (WRONG):
yield prev_bar  # Has old timestamp!

# After (CORRECT):
stale_bar = Bar(
    ts=ts,  # Current timestamp
    symbol=prev_bar.symbol,
    open=prev_bar.open,
    # ... other fields from prev_bar
)
yield stale_bar
```

### Fix #11: Cross-Symbol Contamination ([features_pipeline.py:160-171](../features_pipeline.py#L160-L171))

```python
# Before (WRONG):
big = pd.concat(frames)
big["close"] = big["close"].shift(1)  # Leaks across symbols!

# After (CORRECT):
shifted_frames = []
for frame in frames:
    frame_copy = frame.copy()
    frame_copy["close"] = frame_copy["close"].shift(1)  # Per-symbol
    shifted_frames.append(frame_copy)
big = pd.concat(shifted_frames)
```

### Fix #12: Quantile Loss ([distributional_ppo.py:5707](../distributional_ppo.py#L5707))

```python
# Before (WRONG): Default was False
self._use_fixed_quantile_loss_asymmetry = bool(
    getattr(self.policy, "use_fixed_quantile_loss_asymmetry", False)  # ❌
)

# After (CORRECT): Default is True
self._use_fixed_quantile_loss_asymmetry = bool(
    getattr(self.policy, "use_fixed_quantile_loss_asymmetry", True)  # ✅
)
```

---

## 🧪 Quick Test

Run this to verify fixes are working:

```bash
# Test all three fixes
cd ai-quant-platform
python -m pytest \
    tests/test_stale_bar_temporal_causality.py \
    tests/test_normalization_cross_symbol_contamination.py \
    tests/test_quantile_loss_formula_default.py \
    -v

# Expected: 10/10 tests passed
```

---

## 📚 Full Documentation

- **Detailed Analysis**: [CRITICAL_FIXES_REPORT.md](../CRITICAL_FIXES_REPORT.md)
- **Prevention Guide**: [CRITICAL_BUGS_PREVENTION.md](CRITICAL_BUGS_PREVENTION.md)
- **Changelog**: [CHANGELOG.md](../CHANGELOG.md)
- **Main Docs**: [CLAUDE.md](../CLAUDE.md)

---

## ❓ FAQ

**Q: I have an existing model. Should I retrain?**

A: Check the table above. If your model used any affected features, consider retraining.

**Q: Are new training runs safe?**

A: Yes! All fixes are active by default.

**Q: Can I use the old behavior?**

A: For quantile loss only, set `policy.use_fixed_quantile_loss_asymmetry = False` (not recommended).

**Q: How do I know if my model performance will improve?**

A: Models with quantile critics will see the biggest improvement. Run eval before/after retraining.

**Q: What about backward compatibility?**

A: All fixes are backward-compatible. Old models will load and run (but may benefit from retraining).

---

## 🚨 Golden Rules (Avoid These Bugs)

1. **Temporal Operations**: Current data → current timestamp
2. **Multi-Entity Operations**: Per-entity first, then concat
3. **Math Formulas**: Document with paper reference + unit tests
4. **Default to Correct**: Never default to buggy behavior

---

**Questions?** See full docs or ask the team!
