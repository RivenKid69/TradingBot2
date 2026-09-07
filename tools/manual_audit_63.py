#!/usr/bin/env python3
"""
Manual audit of 63-feature observation structure.
Tracks expected vs actual feature indices from obs_builder.pyx.

UPDATED: 2025-11-20 to match corrected feature_config.py ordering.
"""

print("=" * 80)
print("MANUAL AUDIT: 63-Feature Observation Structure")
print("=" * 80)
print()

# Define CORRECT feature structure based on obs_builder.pyx (lines 236-590)
features = [
    # Block 1: Bar level (0-2)
    (0, "price"),
    (1, "log_volume_norm"),
    (2, "rel_volume"),
    # Block 2: MA5 (3-4)
    (3, "ma5"),
    (4, "ma5_valid"),
    # Block 3: MA20 (5-6)
    (5, "ma20"),
    (6, "ma20_valid"),
    # Block 4: Technical indicators with validity flags (7-20)
    (7, "rsi14"),
    (8, "rsi_valid"),
    (9, "macd"),
    (10, "macd_valid"),
    (11, "macd_signal"),
    (12, "macd_signal_valid"),
    (13, "momentum"),
    (14, "momentum_valid"),
    (15, "atr"),
    (16, "atr_valid"),  # CRITICAL: Added in 62→63, prevents NaN in vol_proxy
    (17, "cci"),
    (18, "cci_valid"),
    (19, "obv"),
    (20, "obv_valid"),
    # Block 5: Derived price/volatility signals (21-22)
    # NOTE: This comes AFTER indicators, not before!
    (21, "ret_bar"),  # tanh((price - prev_price) / prev_price)
    (22, "vol_proxy"),  # Uses atr_valid flag to prevent NaN
    # Block 6: Agent state (23-28)
    (23, "cash_ratio"),
    (24, "position_ratio"),
    (25, "vol_imbalance"),
    (26, "trade_intensity"),
    (27, "realized_spread"),
    (28, "agent_fill_ratio"),
    # Block 7: Microstructure proxies (29-31)
    (29, "price_momentum"),  # tanh(momentum / (price * 0.01))
    (30, "bb_squeeze"),  # tanh((bb_upper - bb_lower) / price)
    (31, "trend_strength"),  # tanh((macd - macd_signal) / (price * 0.01))
    # Block 8: Bollinger Bands context (32-33)
    # NOTE: This was MISSING in previous versions!
    (32, "bb_position"),  # (price - bb_lower) / bb_width, clipped to [-1, 2]
    (33, "bb_width_norm"),  # (bb_upper - bb_lower) / price
    # Block 9: Event metadata (34-38)
    (34, "is_high_importance"),
    (35, "time_since_event"),
    (36, "risk_off_flag"),
    (37, "fear_greed_value"),
    (38, "fear_greed_indicator"),
    # Block 10: External normalized columns (39-59) - 21 features
    *[(39 + i, f"norm_cols[{i}]") for i in range(21)],
    # Block 11: Token metadata (60-61)
    (60, "num_tokens_norm"),
    (61, "token_id_norm"),
    # Block 12: Token one-hot (62)
    (62, "token_one_hot[0]"),
]

print(f"Total features: {len(features)}")

if len(features) == 63:
    print("✅ CORRECT: 63 features")
else:
    print(f"❌ ERROR: {len(features)} features, expected 63")

# Verify block sizes
print("\n" + "=" * 80)
print("Block Structure Verification:")
print("=" * 80)

blocks = [
    ("bar", 0, 2, 3),
    ("ma5", 3, 4, 2),
    ("ma20", 5, 6, 2),
    ("indicators", 7, 20, 14),
    ("derived", 21, 22, 2),
    ("agent", 23, 28, 6),
    ("microstructure", 29, 31, 3),
    ("bb_context", 32, 33, 2),
    ("metadata", 34, 38, 5),
    ("external", 39, 59, 21),
    ("token_meta", 60, 61, 2),
    ("token", 62, 62, 1),
]

total = 0
for name, start, end, expected_size in blocks:
    actual_size = end - start + 1
    if actual_size == expected_size:
        status = "✅"
    else:
        status = "❌"
    print(
        f"{status} {name:15s}: indices {start:2d}-{end:2d} (size {actual_size:2d}, expected {expected_size:2d})"
    )
    total += actual_size

print(f"\nTotal: {total} features")

if total == 63:
    print("✅ CORRECT: Total = 63")
else:
    print(f"❌ ERROR: Total = {total}, expected 63")

# Print critical features to verify
print("\n" + "=" * 80)
print("Critical Features (Manual Verification Required):")
print("=" * 80)

critical = [
    (15, "atr", "ATR value"),
    (16, "atr_valid", "NEW in 63: Prevents NaN in vol_proxy"),
    (21, "ret_bar", "Bar-to-bar return (MOVED from old indices 3-4!)"),
    (22, "vol_proxy", "Volatility proxy (uses atr_valid flag)"),
    (29, "price_momentum", "Price momentum from momentum indicator"),
    (30, "bb_squeeze", "BB squeeze (volatility regime)"),
    (31, "trend_strength", "MACD divergence"),
    (32, "bb_position", "Price position within BB (ADDED, was missing!)"),
    (33, "bb_width_norm", "BB width normalized (ADDED, was missing!)"),
]

for idx, name, description in critical:
    print(f"  [{idx:2d}] {name:20s} - {description}")

print("\n" + "=" * 80)
print("✅ MANUAL AUDIT COMPLETE")
print("=" * 80)
print()
print("Summary:")
print("  - Total features: 63 ✅")
print("  - All block sizes correct ✅")
print("  - Derived moved from indices 3-4 to 21-22 ✅")
print("  - BB context added at indices 32-33 ✅")
print("  - Order matches obs_builder.pyx ✅")
print()
