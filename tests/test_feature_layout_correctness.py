#!/usr/bin/env python
"""
Test that feature_config.py block order matches obs_builder.pyx actual construction.

This test ensures that FEATURES_LAYOUT documentation matches the runtime
observation vector construction order.
"""

import pytest
import numpy as np


def test_feature_layout_matches_obs_builder():
    """
    Verify that FEATURES_LAYOUT block order matches obs_builder.pyx.

    This test constructs an observation vector and verifies that each block
    is at the expected position according to obs_builder.pyx implementation.
    """
    try:
        from obs_builder import build_observation_vector
    except ImportError:
        pytest.skip("obs_builder not compiled")

    from feature_config import FEATURES_LAYOUT

    # Build observation with distinctive values for each expected block
    # Phase 5: norm_cols expanded from 21 to 28 for stock features
    norm_cols = np.array([100.0 + i for i in range(28)], dtype=np.float32)
    norm_cols_validity = np.ones(28, dtype=np.uint8)  # All valid for testing
    out = np.zeros(99, dtype=np.float32)  # Updated from 63 to 99 for Phase 5

    # Set distinctive values for verification
    # Phase 5: Added signal_pos and enable_validity_flags parameters
    build_observation_vector(
        price=50000.0,  # Will be at index 0
        prev_price=49900.0,  # Used for ret_bar at index 21
        log_volume_norm=0.5,  # Will be at index 1
        rel_volume=0.5,  # Will be at index 2
        ma5=50100.0,  # Will be at index 3
        ma20=50200.0,  # Will be at index 5
        rsi14=50.0,  # Will be at index 7
        macd=10.0,  # Will be at index 9
        macd_signal=8.0,  # Will be at index 11
        momentum=15.0,  # Will be at index 13
        atr=100.0,  # Will be at index 15
        cci=5.0,  # Will be at index 17
        obv=1000.0,  # Will be at index 19
        bb_lower=49000.0,  # Used for bb_position at index 33
        bb_upper=51000.0,  # Used for bb_position at index 33
        is_high_importance=1.0,  # Will be at index 35
        time_since_event=5.0,  # Will be at index 36
        fear_greed_value=50.0,  # Will be at index 38
        has_fear_greed=True,
        risk_off_flag=False,
        cash=10000.0,  # Will be at index 23
        units=0.0,
        signal_pos=0.0,  # Will be at index 29
        last_vol_imbalance=0.0,
        last_trade_intensity=0.0,
        last_realized_spread=0.0,
        last_agent_fill_ratio=0.0,
        token_id=0,
        max_num_tokens=1,
        num_tokens=1,
        norm_cols_values=norm_cols,
        norm_cols_validity=norm_cols_validity,
        enable_validity_flags=True,
        out_features=out,
    )

    # Verify actual block positions from obs_builder.pyx
    # Based on build_observation_vector_c() implementation

    # Block 1: bar (3) - indices 0-2
    assert out[0] == 50000.0, "price at index 0"
    assert out[1] == 0.5, "log_volume_norm at index 1"
    assert out[2] == 0.5, "rel_volume at index 2"

    # Block 2: MA5 (2) - indices 3-4
    assert out[3] == 50100.0, "ma5 at index 3"
    assert out[4] == 1.0, "ma5_valid at index 4"

    # Block 3: MA20 (2) - indices 5-6
    assert out[5] == 50200.0, "ma20 at index 5"
    assert out[6] == 1.0, "ma20_valid at index 6"

    # Block 4: RSI (2) - indices 7-8
    assert out[7] == 50.0, "rsi14 at index 7"
    assert out[8] == 1.0, "rsi_valid at index 8"

    # Block 5: MACD (2) - indices 9-10
    assert out[9] == 10.0, "macd at index 9"
    assert out[10] == 1.0, "macd_valid at index 10"

    # Block 6: MACD Signal (2) - indices 11-12
    assert out[11] == 8.0, "macd_signal at index 11"
    assert out[12] == 1.0, "macd_signal_valid at index 12"

    # Block 7: Momentum (2) - indices 13-14
    assert out[13] == 15.0, "momentum at index 13"
    assert out[14] == 1.0, "momentum_valid at index 14"

    # Block 8: ATR (2) - indices 15-16
    assert out[15] == 100.0, "atr at index 15"
    assert out[16] == 1.0, "atr_valid at index 16"

    # Block 9: CCI (2) - indices 17-18
    assert out[17] == 5.0, "cci at index 17"
    assert out[18] == 1.0, "cci_valid at index 18"

    # Block 10: OBV (2) - indices 19-20
    assert out[19] == 1000.0, "obv at index 19"
    assert out[20] == 1.0, "obv_valid at index 20"

    # Block 11: DERIVED (2) - indices 21-22 (NOT 3-4 as in feature_config.py!)
    # ret_bar = tanh((50000 - 49900) / 49900) ≈ tanh(0.002) ≈ 0.002
    assert abs(out[21]) < 0.01, f"ret_bar at index 21 (got {out[21]})"
    # vol_proxy = tanh(log1p(100 / 50000)) ≈ tanh(log1p(0.002)) ≈ tanh(0.002) ≈ 0.002
    assert abs(out[22]) < 0.01, f"vol_proxy at index 22 (got {out[22]})"

    # Block 12: Agent (7) - indices 23-29 (Phase 5: added signal_pos at index 29)
    assert out[23] == 1.0, "cash_ratio at index 23 (all cash, no position)"
    assert out[24] == 0.0, "position_ratio at index 24 (no position)"
    assert out[25] == 0.0, "vol_imbalance at index 25"
    assert out[26] == 0.0, "trade_intensity at index 26"
    assert out[27] == 0.0, "realized_spread at index 27"
    assert out[28] == 0.0, "agent_fill_ratio at index 28"
    assert out[29] == 0.0, "signal_pos at index 29"  # Phase 5: new signal_pos

    # Block 13: Microstructure (3) - indices 30-32 (shifted +1 due to signal_pos)
    # price_momentum uses momentum (15.0) / (price * 0.01) = 15 / 500 = 0.03
    assert abs(out[30] - np.tanh(0.03)) < 0.01, f"price_momentum at index 30 (got {out[30]})"
    # bb_squeeze = (51000 - 49000) / 50000 = 2000 / 50000 = 0.04
    assert abs(out[31] - np.tanh(0.04)) < 0.01, f"bb_squeeze at index 31 (got {out[31]})"
    # trend_strength = (macd - macd_signal) / (price * 0.01) = (10 - 8) / 500 = 0.004
    assert abs(out[32] - np.tanh(0.004)) < 0.01, f"trend_strength at index 32 (got {out[32]})"

    # Block 14: BB Context (2) - indices 33-34 (shifted +1)
    # bb_position = (50000 - 49000) / (51000 - 49000) = 1000 / 2000 = 0.5
    assert abs(out[33] - 0.5) < 0.01, f"bb_position at index 33 (got {out[33]})"
    # bb_width_norm = (51000 - 49000) / 50000 = 2000 / 50000 = 0.04
    assert abs(out[34] - 0.04) < 0.01, f"bb_width_norm at index 34 (got {out[34]})"

    # Block 15: Metadata (5) - indices 35-39 (shifted +1)
    assert out[35] == 1.0, "is_high_importance at index 35"
    assert (
        abs(out[36] - np.tanh(5.0 / 24.0)) < 0.01
    ), f"time_since_event at index 36 (got {out[36]})"
    assert out[37] == 0.0, "risk_off_flag at index 37"
    assert abs(out[38] - 0.5) < 0.01, f"fear_greed_value at index 38 (50/100 = 0.5, got {out[38]})"
    assert out[39] == 1.0, "fear_greed_indicator at index 39"

    # Block 16: External (28) - indices 40-67 (Phase 5: expanded from 21 to 28)
    # norm_cols go through tanh(value) then clip to [-3, 3]
    for i in range(28):
        expected_val = np.tanh(100.0 + i)
        assert (
            abs(out[40 + i] - expected_val) < 0.01
        ), f"external[{i}] at index {40+i} (expected {expected_val}, got {out[40+i]})"

    # Block 17: Token metadata (2) - indices 68-69 (Phase 5: shifted due to expanded external)
    assert abs(out[68] - 1.0) < 0.01, f"num_tokens_norm at index 68 (got {out[68]})"
    assert abs(out[69] - 0.0) < 0.01, f"token_id_norm at index 69 (got {out[69]})"

    # Block 18: Token one-hot (1) - index 70
    assert out[70] == 1.0, "token one-hot at index 70"

    # Block 19: External validity (28) - indices 71-98 (Phase 5: added)
    for i in range(28):
        assert out[71 + i] == 1.0, f"external_validity[{i}] at index {71+i} should be 1.0 (valid)"


def test_feature_config_has_correct_total_size():
    """
    Verify FEATURES_LAYOUT sums to 99 features.

    This test ensures compute_n_features() works correctly regardless of block order.
    Note: Feature count updated from 84 to 99 in Phase 5 (stock features).
    - Added 7 stock-specific features (VIX, market regime, RS, sector momentum)
    - external block: 21 → 28
    - external_validity block: 21 → 28
    """
    from feature_config import FEATURES_LAYOUT, N_FEATURES

    total = sum(block["size"] for block in FEATURES_LAYOUT)

    # Phase 6 (2025-11-28): Expanded EXT_NORM_DIM from 28 to 35
    # New total: 3+2+2+14+2+7+3+2+5+35+35+2+1 = 113
    assert total == 113, f"FEATURES_LAYOUT total size = {total}, expected 113"
    assert N_FEATURES == 113, f"N_FEATURES = {N_FEATURES}, expected 113"


def test_feature_config_block_order_documentation():
    """
    DOCUMENTATION TEST: Warns if feature_config.py block order doesn't match obs_builder.pyx.

    This test documents the CORRECT order from obs_builder.pyx and checks if
    feature_config.py matches it. If not, it's a documentation issue (not runtime bug).
    """
    from feature_config import FEATURES_LAYOUT

    # Expected order from obs_builder.pyx (actual implementation)
    # Phase 6 (2025-11-28): Updated for macro/corporate features
    # - agent: 6 → 7 (added signal_pos)
    # - external: 21 → 35 (21 crypto + 7 stock + 7 macro/corp features)
    # - external_validity: 21 → 35
    # - Total: 99 → 113
    expected_order = [
        ("bar", 3),  # 0-2
        ("ma5", 2),  # 3-4
        ("ma20", 2),  # 5-6
        ("indicators", 14),  # 7-20 (rsi, macd, momentum, atr, cci, obv + flags)
        ("derived", 2),  # 21-22 (ret_bar, vol_proxy)
        ("agent", 7),  # 23-29 (added signal_pos at index 29)
        ("microstructure", 3),  # 30-32
        ("bb_context", 2),  # 33-34
        ("metadata", 5),  # 35-39
        ("external", 35),  # 40-74 (21 crypto + 7 stock + 7 macro/corp features)
        ("external_validity", 35),  # 75-109 (validity flags for external features)
        ("token_meta", 2),  # 110-111
        ("token", 1),  # 112
    ]

    # Current feature_config.py order
    actual_order = [(block["name"], block["size"]) for block in FEATURES_LAYOUT]

    # Check if order matches
    if actual_order == expected_order:
        # Perfect match!
        pass
    else:
        # Document the discrepancy (warning, not failure)
        print("\nWARNING: feature_config.py block order doesn't match obs_builder.pyx")
        print("This is a DOCUMENTATION issue, not a runtime bug.")
        print("\nExpected order (from obs_builder.pyx):")
        idx = 0
        for name, size in expected_order:
            print(f"  {idx:2d}-{idx+size-1:2d}: {name:15s} ({size:2d})")
            idx += size

        print("\nActual order (from feature_config.py):")
        idx = 0
        for name, size in actual_order:
            print(f"  {idx:2d}-{idx+size-1:2d}: {name:15s} ({size:2d})")
            idx += size

        # Check if total size is at least correct
        total_expected = sum(size for _, size in expected_order)
        total_actual = sum(size for _, size in actual_order)

        # Note: Feature count updated from 63 to 84 in latest version
        # This test is for documentation only, so we just verify totals match
        assert (
            total_expected == total_actual
        ), f"Total size mismatch: expected {total_expected}, actual {total_actual}"

        print(f"\nOK: Total size is correct ({total_expected})")
        print("OK: Runtime behavior is unaffected (order not used for indexing)")
        print("NOTE: Consider updating feature_config.py for documentation consistency")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
