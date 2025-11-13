#!/usr/bin/env python
"""
Test to verify that all 56 features are properly integrated into the observation space.
Updated from 51 to 56 features (added 5 taker_buy_ratio derivatives).
"""
import numpy as np
from feature_config import N_FEATURES, make_layout

# Test 1: Check N_FEATURES
print("=" * 80)
print("Test 1: Check N_FEATURES")
print("=" * 80)

make_layout({'max_num_tokens': 1, 'ext_norm_dim': 21})
print(f"✓ N_FEATURES = {N_FEATURES}")

if N_FEATURES == 56:
    print("✓ PASS: N_FEATURES is 56 as expected (was 51, added 5 features)")
else:
    print(f"✗ FAIL: N_FEATURES is {N_FEATURES}, expected 56")
    exit(1)

# Test 2: Check feature layout
print("\n" + "=" * 80)
print("Test 2: Check feature layout")
print("=" * 80)

from feature_config import FEATURES_LAYOUT

total = 0
for block in FEATURES_LAYOUT:
    size = block['size']
    name = block['name']
    print(f"  {name}: {size}")
    total += size

print(f"\nTotal features: {total}")

if total == 56:
    print("✓ PASS: Feature layout sum is 56 (was 51, added 5 features)")
else:
    print(f"✗ FAIL: Feature layout sum is {total}, expected 56")
    exit(1)

# Test 3: Check observation builder
print("\n" + "=" * 80)
print("Test 3: Check observation builder integration")
print("=" * 80)

try:
    from obs_builder import build_observation_vector

    # Create test observation array
    obs = np.zeros(56, dtype=np.float32)
    norm_cols = np.zeros(21, dtype=np.float32)

    # Fill with test data
    for i in range(21):
        norm_cols[i] = float(i) * 0.1

    # Call obs_builder
    build_observation_vector(
        price=100.0,
        prev_price=99.0,
        log_volume_norm=0.5,
        rel_volume=1.0,
        ma5=100.5,
        ma20=100.2,
        rsi14=50.0,
        macd=0.1,
        macd_signal=0.05,
        momentum=0.2,
        atr=1.5,
        cci=0.0,
        obv=1000.0,
        bb_lower=98.0,
        bb_upper=102.0,
        is_high_importance=0.0,
        time_since_event=0.0,
        fear_greed_value=50.0,
        has_fear_greed=True,
        risk_off_flag=False,
        cash=10000.0,
        units=1.0,
        last_vol_imbalance=0.0,
        last_trade_intensity=0.0,
        last_realized_spread=0.0,
        last_agent_fill_ratio=0.0,
        token_id=0,
        max_num_tokens=1,
        num_tokens=1,
        norm_cols_values=norm_cols,
        out_features=obs
    )

    # Check that norm_cols values are present in observation
    # They should be at positions 32-52 (after Fear & Greed at 30-31)
    norm_cols_start = 32

    print(f"\nChecking norm_cols integration:")
    print(f"  norm_cols positions: {norm_cols_start} to {norm_cols_start + 20}")

    all_filled = True
    for i in range(21):
        obs_value = obs[norm_cols_start + i]
        # Values should be transformed by tanh and clipped
        expected_range = (-3.0, 3.0)

        if obs_value == 0.0 and norm_cols[i] != 0.0:
            print(f"  ✗ Position {norm_cols_start + i} is zero but should contain norm_cols[{i}]")
            all_filled = False
        else:
            print(f"  ✓ Position {norm_cols_start + i} = {obs_value:.4f} (from norm_cols[{i}] = {norm_cols[i]:.4f})")

    if all_filled:
        print("\n✓ PASS: All 21 norm_cols values are integrated into observation (was 16, added 5)")
    else:
        print("\n✗ FAIL: Some norm_cols values are missing")
        exit(1)

    # Check observation size
    non_zero_count = np.count_nonzero(obs)
    print(f"\nObservation statistics:")
    print(f"  Total size: {len(obs)}")
    print(f"  Non-zero values: {non_zero_count}")
    print(f"  Min: {obs.min():.4f}, Max: {obs.max():.4f}")

    print("\n✓ PASS: Observation builder works correctly with 56 features (was 51, added 5)")

except Exception as e:
    print(f"\n✗ FAIL: Error during obs_builder test: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: List all 56 features
print("\n" + "=" * 80)
print("Test 4: Complete list of all 56 features (was 51, added 5)")
print("=" * 80)

feature_list = [
    "0-2: Bar (price, log_volume_norm, rel_volume)",
    "3-4: MA5 (value, valid_flag)",
    "5-6: MA20 (value, valid_flag)",
    "7-13: Technical (rsi, macd, macd_signal, momentum, atr, cci, obv)",
    "14-15: Derived (ret_1h, vol_proxy)",
    "16-21: Agent (cash_ratio, position_ratio, vol_imbalance, trade_intensity, realized_spread, agent_fill_ratio)",
    "22-24: Microstructure (ofi_proxy, qimb, micro_dev)",
    "25-26: Bollinger (bb_position, bb_width)",
    "27-29: Event metadata (is_high_importance, time_since_event, risk_off_flag)",
    "30-31: Fear & Greed (value, indicator)",
    "32-52: External/norm_cols (21 features, was 16, added 5) - Обновлено для 4h таймфрейма:",
    "  32: cvd_24h",
    "  33: cvd_7d  (было cvd_168h)",
    "  34: yang_zhang_48h  (было yang_zhang_24h)",
    "  35: yang_zhang_7d  (было yang_zhang_168h)",
    "  36: garch_7d  (было garch_200h. 42 бара = 10080 мин = 7d, минимум для GARCH на 4h)",
    "  37: garch_14d  (было garch_24h)",
    "  38: ret_12h  (было ret_15m)",
    "  39: ret_24h  (было ret_60m)",
    "  40: ret_4h  (было ret_5m)",
    "  41: sma_12000  (было sma_50. 50 баров = 12000 минут = 200h)",
    "  42: yang_zhang_30d  (было yang_zhang_720h)",
    "  43: parkinson_48h  (было parkinson_24h)",
    "  44: parkinson_7d  (было parkinson_168h)",
    "  45: garch_30d  (было garch_500m)",
    "  46: taker_buy_ratio",
    "  47: taker_buy_ratio_sma_24h",
    "  48: taker_buy_ratio_sma_8h (ADDED, было 6h)",
    "  49: taker_buy_ratio_sma_16h (ADDED, было 12h)",
    "  50: taker_buy_ratio_momentum_4h (ADDED, было 1h)",
    "  51: taker_buy_ratio_momentum_8h (ADDED, было 6h)",
    "  52: taker_buy_ratio_momentum_12h (ADDED, без изменений)",
    "53-54: Token metadata (num_tokens_norm, token_id_norm)",
    "55: Token one-hot (1 slot for max_num_tokens=1)",
]

for feature in feature_list:
    print(f"  {feature}")

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED: 56 features are properly integrated!")
print("=" * 80)
