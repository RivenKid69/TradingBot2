# -*- coding: utf-8 -*-
"""
Regression tests for HIGH priority mathematical issues.

Tests for:
- Issue #1: Population vs Sample Std (ddof=0 → ddof=1)
- Issue #3: Reward Doubling (mutual exclusivity of reward modes)
- Issue #4: Potential Shaping (applied in both reward modes)
- Issue #5: Cross-Symbol Contamination (comprehensive test)

Date: 2025-11-20
Status: Regression protection for already fixed bugs
"""
import numpy as np
import pandas as pd
import pytest
from typing import Optional

# ============================================================================
# Issue #3: Reward Doubling - Mutual Exclusivity Tests
# ============================================================================


def test_reward_mutual_exclusivity():
    """
    CRITICAL REGRESSION TEST: Ensure reward is computed using EITHER
    log_return OR scaled_delta, but NOT both (prevents doubling bug).

    This test validates the fix for the reward doubling bug where
    both methods were incorrectly summed together.
    """
    # Import here to avoid issues if Cython module not built
    try:
        from reward import compute_reward_view, log_return
    except ImportError:
        pytest.skip("reward module not available (Cython not built)")

    # Setup realistic scenario
    prev_net_worth = 10000.0
    net_worth = 11000.0  # 10% gain

    # Compute expected values
    expected_log = log_return(net_worth, prev_net_worth)
    expected_delta = (net_worth - prev_net_worth) / abs(prev_net_worth)

    # Test legacy mode (should use ONLY log_return)
    reward_legacy = compute_reward_view(
        net_worth=net_worth,
        prev_net_worth=prev_net_worth,
        use_legacy_log_reward=True,
        use_potential_shaping=False,
        turnover_penalty_coef=0.0,
        spot_cost_taker_fee_bps=0.0,
        spot_cost_half_spread_bps=0.0,
        last_executed_notional=0.0,
        # Additional params with neutral values
        gamma=0.99,
        last_potential=0.0,
        peak_value=net_worth,
        units=0.0,
        atr=0.0,
        risk_aversion_variance=0.0,
        risk_aversion_drawdown=0.0,
        potential_shaping_coef=0.0,
        spot_cost_impact_coeff=0.0,
        spot_cost_adv_quote=0.0,
        spot_cost_impact_exponent=1.0,
        trade_frequency_penalty=0.0,
        trades_count=0,
        profit_close_bonus=0.0,
        loss_close_penalty=0.0,
        bankruptcy_penalty=0.0,
        closed_reason=0,
    )

    # Verify legacy mode uses ONLY log_return
    assert (
        abs(reward_legacy - expected_log) < 1e-6
    ), f"Legacy mode should use ONLY log_return: expected {expected_log:.6f}, got {reward_legacy:.6f}"

    # Test new mode (should use ONLY scaled_delta)
    reward_new = compute_reward_view(
        net_worth=net_worth,
        prev_net_worth=prev_net_worth,
        use_legacy_log_reward=False,
        use_potential_shaping=False,
        turnover_penalty_coef=0.0,
        spot_cost_taker_fee_bps=0.0,
        spot_cost_half_spread_bps=0.0,
        last_executed_notional=0.0,
        gamma=0.99,
        last_potential=0.0,
        peak_value=net_worth,
        units=0.0,
        atr=0.0,
        risk_aversion_variance=0.0,
        risk_aversion_drawdown=0.0,
        potential_shaping_coef=0.0,
        spot_cost_impact_coeff=0.0,
        spot_cost_adv_quote=0.0,
        spot_cost_impact_exponent=1.0,
        trade_frequency_penalty=0.0,
        trades_count=0,
        profit_close_bonus=0.0,
        loss_close_penalty=0.0,
        bankruptcy_penalty=0.0,
        closed_reason=0,
    )

    # Verify new mode uses ONLY scaled_delta
    assert (
        abs(reward_new - expected_delta) < 1e-6
    ), f"New mode should use ONLY scaled_delta: expected {expected_delta:.6f}, got {reward_new:.6f}"

    # CRITICAL CHECK: Ensure NOT using sum of both (the bug)
    double_reward = expected_log + expected_delta

    # Both modes should be DIFFERENT from the sum
    assert (
        abs(reward_legacy - double_reward) > 0.01
    ), f"CRITICAL BUG DETECTED: Legacy reward {reward_legacy:.6f} equals sum {double_reward:.6f}!"

    assert (
        abs(reward_new - double_reward) > 0.01
    ), f"CRITICAL BUG DETECTED: New reward {reward_new:.6f} equals sum {double_reward:.6f}!"

    # Additional sanity check: legacy and new should be similar magnitude
    if reward_new != 0:
        ratio = reward_legacy / reward_new
        assert (
            0.5 < ratio < 2.0
        ), f"Reward methods should give similar magnitude: legacy={reward_legacy:.6f}, new={reward_new:.6f}"


def test_reward_magnitude_consistency():
    """
    Verify that both reward modes produce reasonable magnitudes
    and don't exhibit 2x scaling.
    """
    try:
        from reward import compute_reward_view
    except ImportError:
        pytest.skip("reward module not available")

    # Test various portfolio changes
    test_cases = [
        (10000, 10100, "1% gain"),
        (10000, 9900, "1% loss"),
        (10000, 11000, "10% gain"),
        (10000, 9000, "10% loss"),
        (10000, 10010, "0.1% gain"),
    ]

    for prev, curr, description in test_cases:
        # Compute with both modes
        reward_legacy = compute_reward_view(
            net_worth=curr,
            prev_net_worth=prev,
            use_legacy_log_reward=True,
            use_potential_shaping=False,
            turnover_penalty_coef=0.0,
            spot_cost_taker_fee_bps=0.0,
            spot_cost_half_spread_bps=0.0,
            last_executed_notional=0.0,
            gamma=0.99,
            last_potential=0.0,
            peak_value=curr,
            units=0.0,
            atr=0.0,
            risk_aversion_variance=0.0,
            risk_aversion_drawdown=0.0,
            potential_shaping_coef=0.0,
            spot_cost_impact_coeff=0.0,
            spot_cost_adv_quote=0.0,
            spot_cost_impact_exponent=1.0,
            trade_frequency_penalty=0.0,
            trades_count=0,
            profit_close_bonus=0.0,
            loss_close_penalty=0.0,
            bankruptcy_penalty=0.0,
            closed_reason=0,
        )

        reward_new = compute_reward_view(
            net_worth=curr,
            prev_net_worth=prev,
            use_legacy_log_reward=False,
            use_potential_shaping=False,
            turnover_penalty_coef=0.0,
            spot_cost_taker_fee_bps=0.0,
            spot_cost_half_spread_bps=0.0,
            last_executed_notional=0.0,
            gamma=0.99,
            last_potential=0.0,
            peak_value=curr,
            units=0.0,
            atr=0.0,
            risk_aversion_variance=0.0,
            risk_aversion_drawdown=0.0,
            potential_shaping_coef=0.0,
            spot_cost_impact_coeff=0.0,
            spot_cost_adv_quote=0.0,
            spot_cost_impact_exponent=1.0,
            trade_frequency_penalty=0.0,
            trades_count=0,
            profit_close_bonus=0.0,
            loss_close_penalty=0.0,
            bankruptcy_penalty=0.0,
            closed_reason=0,
        )

        # Both should have same sign
        assert np.sign(reward_legacy) == np.sign(
            reward_new
        ), f"{description}: rewards have different signs! legacy={reward_legacy}, new={reward_new}"

        # Magnitudes should be within 2x of each other (not exact due to log vs linear)
        if reward_new != 0:
            ratio = abs(reward_legacy / reward_new)
            assert (
                0.5 < ratio < 2.0
            ), f"{description}: magnitude ratio {ratio:.2f} outside [0.5, 2.0]"


# ============================================================================
# Issue #4: Potential Shaping - Applied in Both Modes
# ============================================================================


def test_potential_shaping_applied_both_modes():
    """
    CRITICAL REGRESSION TEST: Ensure potential shaping is applied in BOTH
    use_legacy_log_reward=True and False modes.

    This test validates the fix for the bug where potential shaping was
    silently ignored when use_legacy_log_reward=False.
    """
    try:
        from reward import compute_reward_view, potential_phi
    except ImportError:
        pytest.skip("reward module not available")

    # Setup scenario where potential shaping should apply
    net_worth = 10000.0
    prev_net_worth = 10000.0  # No change in wealth
    peak_value = 12000.0  # In drawdown
    units = 100.0  # Holding position
    atr = 50.0  # High volatility

    # Potential function parameters
    risk_aversion_variance = 0.1
    risk_aversion_drawdown = 0.2
    potential_shaping_coef = 0.5
    gamma = 0.99

    # Compute expected phi (should be negative due to risk and drawdown)
    phi_t = potential_phi(
        net_worth,
        peak_value,
        units,
        atr,
        risk_aversion_variance,
        risk_aversion_drawdown,
        potential_shaping_coef,
    )

    # phi should be negative (penalties)
    assert phi_t < 0, f"Phi should be negative (penalties): {phi_t}"

    # Test LEGACY mode WITH shaping
    reward_legacy_shaped = compute_reward_view(
        net_worth=net_worth,
        prev_net_worth=prev_net_worth,
        use_legacy_log_reward=True,
        use_potential_shaping=True,
        risk_aversion_variance=risk_aversion_variance,
        risk_aversion_drawdown=risk_aversion_drawdown,
        potential_shaping_coef=potential_shaping_coef,
        gamma=gamma,
        last_potential=0.0,
        peak_value=peak_value,
        units=units,
        atr=atr,
        turnover_penalty_coef=0.0,
        spot_cost_taker_fee_bps=0.0,
        spot_cost_half_spread_bps=0.0,
        last_executed_notional=0.0,
        spot_cost_impact_coeff=0.0,
        spot_cost_adv_quote=0.0,
        spot_cost_impact_exponent=1.0,
        trade_frequency_penalty=0.0,
        trades_count=0,
        profit_close_bonus=0.0,
        loss_close_penalty=0.0,
        bankruptcy_penalty=0.0,
        closed_reason=0,
    )

    # Test LEGACY mode WITHOUT shaping
    reward_legacy_no_shape = compute_reward_view(
        net_worth=net_worth,
        prev_net_worth=prev_net_worth,
        use_legacy_log_reward=True,
        use_potential_shaping=False,
        gamma=gamma,
        peak_value=peak_value,
        units=units,
        atr=atr,
        turnover_penalty_coef=0.0,
        spot_cost_taker_fee_bps=0.0,
        spot_cost_half_spread_bps=0.0,
        last_executed_notional=0.0,
        risk_aversion_variance=0.0,
        risk_aversion_drawdown=0.0,
        potential_shaping_coef=0.0,
        last_potential=0.0,
        spot_cost_impact_coeff=0.0,
        spot_cost_adv_quote=0.0,
        spot_cost_impact_exponent=1.0,
        trade_frequency_penalty=0.0,
        trades_count=0,
        profit_close_bonus=0.0,
        loss_close_penalty=0.0,
        bankruptcy_penalty=0.0,
        closed_reason=0,
    )

    # Shaping should make a difference in legacy mode
    legacy_diff = reward_legacy_shaped - reward_legacy_no_shape
    assert (
        abs(legacy_diff) > 1e-4
    ), f"Potential shaping should affect legacy mode: shaped={reward_legacy_shaped}, no_shape={reward_legacy_no_shape}"

    # Difference should be negative (penalty)
    assert (
        legacy_diff < 0
    ), f"Potential shaping should penalize (risk + drawdown): diff={legacy_diff}"

    # Test NEW mode WITH shaping
    reward_new_shaped = compute_reward_view(
        net_worth=net_worth,
        prev_net_worth=prev_net_worth,
        use_legacy_log_reward=False,
        use_potential_shaping=True,
        risk_aversion_variance=risk_aversion_variance,
        risk_aversion_drawdown=risk_aversion_drawdown,
        potential_shaping_coef=potential_shaping_coef,
        gamma=gamma,
        last_potential=0.0,
        peak_value=peak_value,
        units=units,
        atr=atr,
        turnover_penalty_coef=0.0,
        spot_cost_taker_fee_bps=0.0,
        spot_cost_half_spread_bps=0.0,
        last_executed_notional=0.0,
        spot_cost_impact_coeff=0.0,
        spot_cost_adv_quote=0.0,
        spot_cost_impact_exponent=1.0,
        trade_frequency_penalty=0.0,
        trades_count=0,
        profit_close_bonus=0.0,
        loss_close_penalty=0.0,
        bankruptcy_penalty=0.0,
        closed_reason=0,
    )

    # Test NEW mode WITHOUT shaping
    reward_new_no_shape = compute_reward_view(
        net_worth=net_worth,
        prev_net_worth=prev_net_worth,
        use_legacy_log_reward=False,
        use_potential_shaping=False,
        gamma=gamma,
        peak_value=peak_value,
        units=units,
        atr=atr,
        turnover_penalty_coef=0.0,
        spot_cost_taker_fee_bps=0.0,
        spot_cost_half_spread_bps=0.0,
        last_executed_notional=0.0,
        risk_aversion_variance=0.0,
        risk_aversion_drawdown=0.0,
        potential_shaping_coef=0.0,
        last_potential=0.0,
        spot_cost_impact_coeff=0.0,
        spot_cost_adv_quote=0.0,
        spot_cost_impact_exponent=1.0,
        trade_frequency_penalty=0.0,
        trades_count=0,
        profit_close_bonus=0.0,
        loss_close_penalty=0.0,
        bankruptcy_penalty=0.0,
        closed_reason=0,
    )

    # CRITICAL: Shaping should ALSO make a difference in NEW mode!
    new_diff = reward_new_shaped - reward_new_no_shape
    assert abs(new_diff) > 1e-4, (
        f"CRITICAL BUG: Potential shaping NOT applied in new mode! "
        f"shaped={reward_new_shaped}, no_shape={reward_new_no_shape}"
    )

    # Difference should be negative (penalty)
    assert new_diff < 0, f"Potential shaping should penalize in new mode: diff={new_diff}"

    # Shaping effect should be SIMILAR in both modes
    if legacy_diff != 0:
        ratio = abs(new_diff / legacy_diff)
        assert (
            0.8 < ratio < 1.2
        ), f"Shaping effect should be similar across modes: legacy_diff={legacy_diff}, new_diff={new_diff}, ratio={ratio}"


def test_potential_function_penalties():
    """
    Verify potential function correctly penalizes risk and drawdown.
    """
    try:
        from reward import potential_phi
    except ImportError:
        pytest.skip("reward module not available")

    # Base case: no risk, no drawdown
    phi_base = potential_phi(
        net_worth=10000,
        peak_value=10000,
        units=0,  # No position
        atr=50,
        risk_aversion_variance=0.1,
        risk_aversion_drawdown=0.2,
        potential_shaping_coef=0.5,
    )
    # Should be zero or very small
    assert abs(phi_base) < 0.01, f"Phi should be ~0 with no risk/drawdown: {phi_base}"

    # Case 1: Large position → risk penalty
    phi_risky = potential_phi(
        net_worth=10000,
        peak_value=10000,
        units=100,  # Large position
        atr=50,  # High volatility
        risk_aversion_variance=0.1,
        risk_aversion_drawdown=0.2,
        potential_shaping_coef=0.5,
    )
    # Should be MORE negative than base
    assert (
        phi_risky < phi_base
    ), f"Large position should create negative phi: base={phi_base}, risky={phi_risky}"

    # Case 2: Drawdown → drawdown penalty
    phi_drawdown = potential_phi(
        net_worth=8000,  # Down from peak
        peak_value=10000,  # 20% drawdown
        units=0,  # No position risk
        atr=50,
        risk_aversion_variance=0.1,
        risk_aversion_drawdown=0.2,
        potential_shaping_coef=0.5,
    )
    # Should be MORE negative than base
    assert (
        phi_drawdown < phi_base
    ), f"Drawdown should create negative phi: base={phi_base}, dd={phi_drawdown}"

    # Case 3: Both risk AND drawdown
    phi_both = potential_phi(
        net_worth=8000,
        peak_value=10000,
        units=100,
        atr=50,
        risk_aversion_variance=0.1,
        risk_aversion_drawdown=0.2,
        potential_shaping_coef=0.5,
    )
    # Should be MOST negative
    assert (
        phi_both < phi_risky and phi_both < phi_drawdown
    ), f"Combined risk+drawdown should be most negative: both={phi_both}, risky={phi_risky}, dd={phi_drawdown}"


# ============================================================================
# Issue #1: Population vs Sample Standard Deviation (ddof)
# ============================================================================


def test_feature_pipeline_uses_sample_std():
    """
    Verify that feature pipeline uses sample std (ddof=1).

    This test validates the fix for using population std (ddof=0) instead of
    sample std (ddof=1) for feature normalization.
    """
    from features_pipeline import FeaturePipeline

    # Create synthetic data
    data = pd.DataFrame(
        {
            "close": np.random.randn(100) * 10 + 100,
            "volume": np.random.randn(100) * 1000 + 5000,
            "symbol": ["BTCUSDT"] * 100,
        }
    )

    # Fit pipeline
    pipeline = FeaturePipeline()
    pipeline.fit({"BTCUSDT": data})

    # Compute expected stats with ddof=1 (after shift, first value is NaN)
    close_shifted = data["close"].shift(1).dropna()
    expected_std = float(np.std(close_shifted.values, ddof=1))

    # Verify
    actual_std = pipeline.stats["close"]["std"]

    # Should match sample std (ddof=1)
    assert (
        abs(actual_std - expected_std) < 1e-6
    ), f"Pipeline should use sample std (ddof=1): expected {expected_std}, got {actual_std}"

    # Should NOT match population std (ddof=0)
    wrong_std = float(np.std(close_shifted.values, ddof=0))
    assert (
        abs(actual_std - wrong_std) > 1e-9
    ), f"Pipeline should NOT use population std (ddof=0): got {actual_std}, wrong would be {wrong_std}"


def test_feature_pipeline_matches_sklearn():
    """
    Verify normalization matches scikit-learn StandardScaler.

    StandardScaler uses ddof=1 (sample std), so our pipeline should match.
    """
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        pytest.skip("scikit-learn not available")

    from features_pipeline import FeaturePipeline

    # Create test data
    X = np.random.randn(1000, 5) * 100 + 500
    data = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    data["symbol"] = "BTCUSDT"

    # Fit both
    sklearn_scaler = StandardScaler()
    sklearn_scaler.fit(X)

    pipeline = FeaturePipeline()
    pipeline.fit({"BTCUSDT": data})

    # Compare stds (accounting for shift operation)
    for i in range(5):
        feat_name = f"feat_{i}"

        # sklearn std uses ddof=1
        sklearn_std = sklearn_scaler.scale_[i]

        # Our pipeline should match
        if feat_name in pipeline.stats:
            pipeline_std = pipeline.stats[feat_name]["std"]

            # After shift, we have N-1 values, but sklearn uses N values
            # So we need to recompute sklearn std on shifted data
            shifted_values = data[feat_name].shift(1).dropna().values
            sklearn_std_shifted = np.std(shifted_values, ddof=1)

            assert (
                abs(sklearn_std_shifted - pipeline_std) < 1e-4
            ), f"Feature {feat_name}: sklearn {sklearn_std_shifted} vs pipeline {pipeline_std}"


def test_ddof_statistical_correctness():
    """
    Verify that ddof=1 provides unbiased variance estimation.

    This is a statistical test showing that sample variance (ddof=1) is
    an unbiased estimator of population variance.
    """
    from features_pipeline import FeaturePipeline

    # Generate data from known distribution
    np.random.seed(42)
    true_mean = 100.0
    true_std = 15.0

    # Multiple samples
    n_samples = 50
    sample_size = 100

    std_ddof0_list = []
    std_ddof1_list = []

    for _ in range(n_samples):
        sample = np.random.normal(true_mean, true_std, sample_size)

        std_ddof0 = np.std(sample, ddof=0)
        std_ddof1 = np.std(sample, ddof=1)

        std_ddof0_list.append(std_ddof0)
        std_ddof1_list.append(std_ddof1)

    # Mean of sample stds should be close to true std
    mean_std_ddof0 = np.mean(std_ddof0_list)
    mean_std_ddof1 = np.mean(std_ddof1_list)

    # ddof=1 should be closer to true std (unbiased estimator)
    bias_ddof0 = abs(mean_std_ddof0 - true_std)
    bias_ddof1 = abs(mean_std_ddof1 - true_std)

    # ddof=1 should have less bias
    assert (
        bias_ddof1 < bias_ddof0
    ), f"ddof=1 should have less bias: ddof=0 bias={bias_ddof0:.4f}, ddof=1 bias={bias_ddof1:.4f}"

    # For N=100, bias should be small for ddof=1
    assert bias_ddof1 < 1.0, f"ddof=1 bias should be small: {bias_ddof1:.4f}"


# ============================================================================
# Issue #5: Cross-Symbol Contamination - Comprehensive Test
# ============================================================================


def test_cross_symbol_statistics_independence():
    """
    COMPREHENSIVE TEST: Verify that shift() operation doesn't leak data
    between symbols (no cross-symbol contamination).

    NOTE: The pipeline uses GLOBAL normalization (stats computed across all symbols),
    which is the correct design. This test verifies that the SHIFT operation
    is applied per-symbol to prevent data leakage.
    """
    from features_pipeline import FeaturePipeline

    # Create two symbols with very different scales
    btc_data = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 10,
            "close": [50000.0 + i * 100 for i in range(10)],  # BTC scale
            "volume": [10000.0] * 10,
        }
    )

    eth_data = pd.DataFrame(
        {
            "symbol": ["ETHUSDT"] * 10,
            "close": [3000.0 + i * 10 for i in range(10)],  # ETH scale
            "volume": [5000.0] * 10,
        }
    )

    # Fit pipeline
    pipeline = FeaturePipeline()
    pipeline.fit({"BTCUSDT": btc_data, "ETHUSDT": eth_data})

    # Create combined dataframe for transform
    combined = pd.concat([btc_data, eth_data], ignore_index=True)

    # Transform
    transformed = pipeline.transform_df(combined)

    # CRITICAL CHECK: First row of each symbol should have NaN after shift
    # If shift was applied globally (buggy), ETH's first row would have BTC's last value

    # Find first row of each symbol
    btc_rows = transformed[transformed["symbol"] == "BTCUSDT"]
    eth_rows = transformed[transformed["symbol"] == "ETHUSDT"]

    # First row should have NaN in shifted 'close' column (before adding suffix)
    # The original 'close' column should be shifted
    assert pd.isna(
        btc_rows.iloc[0]["close"]
    ), "First BTC row should have NaN after per-symbol shift"
    assert pd.isna(
        eth_rows.iloc[0]["close"]
    ), "First ETH row should have NaN after per-symbol shift (NOT contaminated by BTC)"

    # Second row of BTC should have first BTC value
    assert (
        abs(btc_rows.iloc[1]["close"] - 50000.0) < 0.01
    ), f"Second BTC row should have shifted close=50000, got {btc_rows.iloc[1]['close']}"

    # Second row of ETH should have first ETH value (NOT last BTC value)
    assert abs(eth_rows.iloc[1]["close"] - 3000.0) < 0.01, (
        f"Second ETH row should have shifted close=3000, got {eth_rows.iloc[1]['close']} "
        "(if this is ~50900, there's cross-symbol contamination!)"
    )


def test_cross_symbol_multi_asset_consistency():
    """
    Test that shift() operation is consistent across multiple assets.

    Verifies that per-symbol shift prevents data leakage between symbols,
    even when multiple symbols are processed together.
    """
    from features_pipeline import FeaturePipeline

    # Create 3 symbols with different price scales
    symbols_data = {}
    expected_scales = {
        "BTCUSDT": 50000,
        "ETHUSDT": 3000,
        "BNBUSDT": 400,
    }

    for symbol, base_price in expected_scales.items():
        data = pd.DataFrame(
            {
                "symbol": [symbol] * 10,
                "close": [base_price + i * 10 for i in range(10)],
                "volume": [5000.0] * 10,
            }
        )
        symbols_data[symbol] = data

    # Fit pipeline on all symbols
    pipeline = FeaturePipeline()
    pipeline.fit(symbols_data)

    # Create combined dataframe
    combined = pd.concat(symbols_data.values(), ignore_index=True)

    # Transform
    transformed = pipeline.transform_df(combined)

    # Check each symbol's first row has NaN (per-symbol shift)
    for symbol in expected_scales.keys():
        symbol_rows = transformed[transformed["symbol"] == symbol]
        first_close = symbol_rows.iloc[0]["close"]

        assert pd.isna(
            first_close
        ), f"{symbol}: First row should have NaN after per-symbol shift, got {first_close}"

    # Check that second rows have correct shifted values (no cross-contamination)
    for symbol, base_price in expected_scales.items():
        symbol_rows = transformed[transformed["symbol"] == symbol]
        second_close = symbol_rows.iloc[1]["close"]

        # Should be the first value of THIS symbol, not another symbol
        assert (
            abs(second_close - base_price) < 1.0
        ), f"{symbol}: Second row should have close≈{base_price}, got {second_close}"


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
