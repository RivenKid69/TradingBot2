#!/usr/bin/env python3
"""
Precise manual count of feature indices based on obs_builder.pyx code structure.
"""

# Start from 0
idx = 0

features_list = []

# Bar level (3 features)
features_list.append((idx, "price"))
idx += 1
features_list.append((idx, "log_volume_norm"))
idx += 1
features_list.append((idx, "rel_volume"))
idx += 1

# MA5 (2 features)
features_list.append((idx, "ma5"))
idx += 1
features_list.append((idx, "ma5_valid"))
idx += 1

# MA20 (2 features)
features_list.append((idx, "ma20"))
idx += 1
features_list.append((idx, "ma20_valid"))
idx += 1

# RSI (2 features)
features_list.append((idx, "rsi14"))
idx += 1
features_list.append((idx, "rsi_valid"))
idx += 1

# MACD (2 features)
features_list.append((idx, "macd"))
idx += 1
features_list.append((idx, "macd_valid"))
idx += 1

# MACD Signal (2 features)
features_list.append((idx, "macd_signal"))
idx += 1
features_list.append((idx, "macd_signal_valid"))
idx += 1

# Momentum (2 features)
features_list.append((idx, "momentum"))
idx += 1
features_list.append((idx, "momentum_valid"))
idx += 1

# ATR (2 features) - NEW in 62→63
features_list.append((idx, "atr"))
idx += 1
features_list.append((idx, "atr_valid"))
idx += 1  # NEW

# CCI (2 features)
features_list.append((idx, "cci"))
idx += 1
features_list.append((idx, "cci_valid"))
idx += 1

# OBV (2 features)
features_list.append((idx, "obv"))
idx += 1
features_list.append((idx, "obv_valid"))
idx += 1

print("=" * 80)
print(f"After indicators block: idx = {idx}")
print("=" * 80)

# Derived (2 features) - based on code review
features_list.append((idx, "ret_bar"))
idx += 1
features_list.append((idx, "vol_proxy"))
idx += 1

print(f"After derived block: idx = {idx}")

# Agent state (6 features)
features_list.append((idx, "cash_ratio"))
idx += 1
features_list.append((idx, "position_ratio"))
idx += 1
features_list.append((idx, "vol_imbalance"))
idx += 1
features_list.append((idx, "trade_intensity"))
idx += 1
features_list.append((idx, "realized_spread"))
idx += 1
features_list.append((idx, "agent_fill_ratio"))
idx += 1

print(f"After agent state block: idx = {idx}")

# Microstructure / Technical 4h (3 features)
features_list.append((idx, "price_momentum"))
idx += 1
features_list.append((idx, "bb_squeeze"))
idx += 1
features_list.append((idx, "trend_strength"))
idx += 1

print(f"After microstructure block: idx = {idx}")

# Bollinger Bands (2 features)
features_list.append((idx, "bb_position"))
idx += 1
features_list.append((idx, "bb_width"))
idx += 1

print(f"After BB block: idx = {idx}")

# Event metadata (3 features)
features_list.append((idx, "is_high_importance"))
idx += 1
features_list.append((idx, "time_since_event"))
idx += 1
features_list.append((idx, "risk_off_flag"))
idx += 1

print(f"After event metadata block: idx = {idx}")

# Fear & Greed (2 features)
features_list.append((idx, "fear_greed_value"))
idx += 1
features_list.append((idx, "fear_greed_indicator"))
idx += 1

print(f"After fear & greed block: idx = {idx}")

# External (21 features in loop)
for i in range(21):
    features_list.append((idx, f"norm_cols[{i}]"))
    idx += 1

print(f"After external block: idx = {idx}")

# Token metadata (2 features)
features_list.append((idx, "token_count_ratio"))
idx += 1
features_list.append((idx, "token_id_norm"))
idx += 1

print(f"After token metadata block: idx = {idx}")

# Token one-hot (1 feature, max_num_tokens=1)
features_list.append((idx, "token_one_hot[0]"))
idx += 1

print(f"Final idx = {idx}")
print("=" * 80)

if idx == 63:
    print("✅ CORRECT: 63 features")
else:
    print(f"❌ ERROR: {idx} features, expected 63")

print("\n" + "=" * 80)
print("Critical Indices:")
print("=" * 80)

critical_indices = [15, 16, 17, 21, 22, 23, 24, 31, 32, 33, 39, 59, 62]

for check_idx in critical_indices:
    if check_idx < len(features_list):
        actual_idx, name = features_list[check_idx]
        print(f"  [{actual_idx:2d}] {name}")
    else:
        print(f"  [{check_idx:2d}] MISSING")

print("\n" + "=" * 80)
print("Full feature list:")
print("=" * 80)

for actual_idx, name in features_list[:40]:
    print(f"  [{actual_idx:2d}] {name}")

if len(features_list) > 40:
    print(f"  ... ({len(features_list) - 40} more)")
