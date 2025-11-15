"""
Check: Are the RSI edge case values consistent with codebase style?
Pattern in transformers.py: ALL feature values are wrapped in float()
"""

print("=" * 80)
print("Consistency Check: float() wrapping pattern")
print("=" * 80)

# From code analysis:
patterns = {
    "SMA": "float(sma)",
    "Yang-Zhang": "float(yz_vol) or float('nan')",
    "Parkinson": "float(pk_vol) or float('nan')",
    "GARCH": "float(garch_vol) or float('nan')",
    "taker_buy_ratio": "float(ratio_list[-1]) or float('nan')",
}

print("\nExisting patterns in transformers.py:")
for name, pattern in patterns.items():
    print(f"  {name:20} → {pattern}")

print("\nMy RSI fix:")
print("  Line 635: feats['rsi'] = 100.0  ← ⚠️ NOT wrapped in float()")
print("  Line 638: feats['rsi'] = 0.0    ← ⚠️ NOT wrapped in float()")
print("  Line 641: feats['rsi'] = 50.0   ← ⚠️ NOT wrapped in float()")
print("  Line 645: feats['rsi'] = float(100.0 - ...) ← ✓ Wrapped")
print("  Line 647: feats['rsi'] = float('nan')      ← ✓ Wrapped")

print("\n" + "=" * 80)
print("Issue: INCONSISTENT style")
print("=" * 80)
print("\nShould be:")
print("  feats['rsi'] = float(100.0)  # Consistent with codebase")
print("  feats['rsi'] = float(0.0)")
print("  feats['rsi'] = float(50.0)")

print("\nSeverity: LOW")
print("  ✓ Functionally correct (100.0 is already a float)")
print("  ✗ Style inconsistent with rest of codebase")
print("  → Recommendation: Fix for consistency")

print("\n" + "=" * 80)
print("Checking if this causes any functional issue...")
print("=" * 80)

# Test: Does it matter?
a = 100.0
b = float(100.0)

print(f"\ntype(100.0)       = {type(a)}")
print(f"type(float(100.0)) = {type(b)}")
print(f"a == b             = {a == b}")
print(f"a is b             = {a is b}")

print("\n✓ No functional difference!")
print("But for code consistency, should use float() wrapper.")
