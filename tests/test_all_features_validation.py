"""
Comprehensive test suite for observation vector NaN/Inf validation.

This test suite validates that the observation vector is robust against:
1. NaN inputs from technical indicators (early bars)
2. Inf values from mathematical operations
3. Edge cases (price=0, cash=0, empty data)
4. First 30 bars of simulation (warmup period)

Reference: OBSERVATION_VECTOR_AUDIT_REPORT.md, FEATURES_VALIDATION_CHECKLIST.md
"""

import numpy as np
import pytest
from obs_builder import build_observation_vector, compute_n_features


class TestTechnicalIndicatorsNaNHandling:
    """Test that all technical indicators handle NaN inputs correctly."""

    def test_all_indicators_nan_to_defaults(self):
        """Test that NaN indicators are replaced with appropriate defaults."""
        n_features = compute_n_features([])
        obs = np.zeros(n_features, dtype=np.float32)
        norm_cols = np.zeros(21, dtype=np.float32)

        # All indicators as NaN (simulating early bars)
        build_observation_vector(
            price=100.0,
            prev_price=99.0,
            log_volume_norm=0.5,
            rel_volume=0.5,
            ma5=np.nan,  # NaN
            ma20=np.nan,  # NaN
            rsi14=np.nan,  # NaN
            macd=np.nan,  # NaN
            macd_signal=np.nan,  # NaN
            momentum=np.nan,  # NaN
            atr=np.nan,  # NaN
            cci=np.nan,  # NaN
            obv=np.nan,  # NaN (though OBV should never be NaN)
            bb_lower=np.nan,  # NaN
            bb_upper=np.nan,  # NaN
            is_high_importance=0.0,
            time_since_event=1e9,
            fear_greed_value=50.0,
            has_fear_greed=False,
            risk_off_flag=False,
            cash=10000.0,
            units=0.0,
            last_vol_imbalance=0.0,
            last_trade_intensity=0.0,
            last_realized_spread=0.0,
            last_agent_fill_ratio=0.0,
            token_id=0,
            max_num_tokens=1,
            num_tokens=1,
            norm_cols_values=norm_cols,
            out_features=obs,
        )

        # Verify no NaN or Inf in output
        assert np.all(np.isfinite(obs)), f"Found NaN/Inf in observation with NaN indicators: {obs[~np.isfinite(obs)]}"

        # Verify expected defaults (based on obs_builder.pyx)
        # Index mapping (approximate - may need adjustment):
        # 0: price, 1: log_volume_norm, 2: rel_volume
        # 3: ma5, 4: ma5_valid_flag, 5: ma20, 6: ma20_valid_flag
        # 7: rsi14, 8: macd, 9: macd_signal, 10: momentum, 11: atr, 12: cci, 13: obv

        assert obs[0] == 100.0, "price should be 100.0"
        assert obs[3] == 0.0, "ma5 NaN → 0.0"
        assert obs[4] == 0.0, "ma5_valid should be 0.0 (not valid)"
        assert obs[5] == 0.0, "ma20 NaN → 0.0"
        assert obs[6] == 0.0, "ma20_valid should be 0.0 (not valid)"
        assert obs[7] == 50.0, "rsi14 NaN → 50.0 (neutral)"
        assert obs[8] == 0.0, "macd NaN → 0.0"
        assert obs[9] == 0.0, "macd_signal NaN → 0.0"
        assert obs[10] == 0.0, "momentum NaN → 0.0"
        # ATR default is price * 0.01 = 100.0 * 0.01 = 1.0
        assert obs[11] == pytest.approx(1.0, abs=0.01), f"atr NaN → price*0.01=1.0, got {obs[11]}"
        assert obs[12] == 0.0, "cci NaN → 0.0"
        assert obs[13] == 0.0, "obv NaN → 0.0"

    def test_ma5_valid_flag(self):
        """Test that ma5_valid flag is set correctly."""
        n_features = compute_n_features([])
        obs = np.zeros(n_features, dtype=np.float32)
        norm_cols = np.zeros(21, dtype=np.float32)

        # Test with valid MA5
        build_observation_vector(
            100.0, 99.0, 0.0, 0.0,
            105.0,  # ma5 valid
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            0.0, 1e9, 50.0, False, False,
            10000.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0, 1, 1, norm_cols, obs
        )
        assert obs[3] == 105.0, "ma5 should be 105.0"
        assert obs[4] == 1.0, "ma5_valid should be 1.0 (valid)"

        # Test with NaN MA5
        build_observation_vector(
            100.0, 99.0, 0.0, 0.0,
            np.nan,  # ma5 invalid
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            0.0, 1e9, 50.0, False, False,
            10000.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0, 1, 1, norm_cols, obs
        )
        assert obs[3] == 0.0, "ma5 NaN → 0.0"
        assert obs[4] == 0.0, "ma5_valid should be 0.0 (not valid)"

    def test_ma20_valid_flag(self):
        """Test that ma20_valid flag is set correctly."""
        n_features = compute_n_features([])
        obs = np.zeros(n_features, dtype=np.float32)
        norm_cols = np.zeros(21, dtype=np.float32)

        # Test with valid MA20
        build_observation_vector(
            100.0, 99.0, 0.0, 0.0,
            np.nan,
            102.0,  # ma20 valid
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            0.0, 1e9, 50.0, False, False,
            10000.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0, 1, 1, norm_cols, obs
        )
        assert obs[5] == 102.0, "ma20 should be 102.0"
        assert obs[6] == 1.0, "ma20_valid should be 1.0 (valid)"


class TestMathematicalOperationsSafety:
    """Test that all mathematical operations are protected from NaN/Inf."""

    def test_division_by_zero_protection(self):
        """Test that divisions are protected with epsilon."""
        n_features = compute_n_features([])
        obs = np.zeros(n_features, dtype=np.float32)
        norm_cols = np.zeros(21, dtype=np.float32)

        # prev_price = 0 should not cause division by zero in ret_bar
        build_observation_vector(
            price=100.0,
            prev_price=0.0,  # Edge case: prev_price = 0
            log_volume_norm=0.0,
            rel_volume=0.0,
            ma5=np.nan, ma20=np.nan, rsi14=np.nan,
            macd=np.nan, macd_signal=np.nan, momentum=np.nan,
            atr=np.nan, cci=np.nan, obv=0.0,
            bb_lower=np.nan, bb_upper=np.nan,
            is_high_importance=0.0, time_since_event=1e9,
            fear_greed_value=50.0, has_fear_greed=False, risk_off_flag=False,
            cash=10000.0, units=0.0,
            last_vol_imbalance=0.0, last_trade_intensity=0.0,
            last_realized_spread=0.0, last_agent_fill_ratio=0.0,
            token_id=0, max_num_tokens=1, num_tokens=1,
            norm_cols_values=norm_cols, out_features=obs
        )

        assert np.all(np.isfinite(obs)), "Division by prev_price=0 caused NaN/Inf"

    def test_total_worth_zero(self):
        """Test cash_fraction when total_worth=0."""
        n_features = compute_n_features([])
        obs = np.zeros(n_features, dtype=np.float32)
        norm_cols = np.zeros(21, dtype=np.float32)

        # cash=0, units=0 → total_worth=0
        build_observation_vector(
            100.0, 99.0, 0.0, 0.0,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, np.nan, np.nan,
            0.0, 1e9, 50.0, False, False,
            0.0,  # cash = 0
            0.0,  # units = 0
            0.0, 0.0, 0.0, 0.0,
            0, 1, 1, norm_cols, obs
        )

        assert np.all(np.isfinite(obs)), "total_worth=0 caused NaN/Inf"
        # cash_fraction should be 1.0 (special case: 100% cash when portfolio empty)
        # Index 16 is cash_fraction (approximate)
        assert obs[16] == pytest.approx(1.0, abs=0.01), f"cash_fraction should be 1.0 when total_worth=0, got {obs[16]}"

    def test_tanh_with_nan_inputs(self):
        """Test that tanh operations don't propagate NaN."""
        n_features = compute_n_features([])
        obs = np.zeros(n_features, dtype=np.float32)

        # norm_cols with NaN values
        norm_cols = np.full(21, np.nan, dtype=np.float32)

        build_observation_vector(
            100.0, 99.0, 0.0, 0.0,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, np.nan, np.nan,
            0.0, 1e9, 50.0, False, False,
            10000.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0, 1, 1, norm_cols, obs
        )

        # _clipf should handle NaN → 0.0 for norm_cols
        assert np.all(np.isfinite(obs)), "tanh(NaN) from norm_cols propagated to output"

    def test_log1p_safety(self):
        """Test that log1p is used correctly and doesn't produce NaN."""
        n_features = compute_n_features([])
        obs = np.zeros(n_features, dtype=np.float32)
        norm_cols = np.zeros(21, dtype=np.float32)

        # Negative ATR should not happen, but test defensively
        # Actually, ATR is always >=0, but let's test the vol_proxy calculation
        build_observation_vector(
            100.0, 99.0, 0.0, 0.0,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            10.0,  # ATR = 10.0 (valid)
            np.nan, 0.0, np.nan, np.nan,
            0.0, 1e9, 50.0, False, False,
            10000.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0, 1, 1, norm_cols, obs
        )

        assert np.all(np.isfinite(obs)), "log1p in vol_proxy caused NaN/Inf"


class TestEdgeCases:
    """Test edge cases: first bar, price=0, empty data, etc."""

    def test_first_bar_simulation(self):
        """Test observation on first bar (i=0) where most indicators are NaN."""
        n_features = compute_n_features([])
        obs = np.zeros(n_features, dtype=np.float32)
        norm_cols = np.zeros(21, dtype=np.float32)

        # Simulate first bar: all indicators NaN except OBV
        build_observation_vector(
            price=100.0,
            prev_price=100.0,  # Same as price (no previous bar)
            log_volume_norm=0.0,
            rel_volume=0.0,
            ma5=np.nan, ma20=np.nan, rsi14=np.nan,
            macd=np.nan, macd_signal=np.nan, momentum=np.nan,
            atr=np.nan, cci=np.nan,
            obv=0.0,  # OBV starts at 0
            bb_lower=np.nan, bb_upper=np.nan,
            is_high_importance=0.0, time_since_event=1e9,
            fear_greed_value=50.0, has_fear_greed=False, risk_off_flag=False,
            cash=10000.0, units=0.0,
            last_vol_imbalance=0.0, last_trade_intensity=0.0,
            last_realized_spread=0.0, last_agent_fill_ratio=0.0,
            token_id=0, max_num_tokens=1, num_tokens=1,
            norm_cols_values=norm_cols, out_features=obs
        )

        assert np.all(np.isfinite(obs)), f"First bar produced NaN/Inf: {obs[~np.isfinite(obs)]}"

    def test_first_30_bars(self):
        """Test observation vector on first 30 bars with progressive indicator readiness."""
        n_features = compute_n_features([])
        norm_cols = np.zeros(21, dtype=np.float32)

        for bar_idx in range(30):
            obs = np.zeros(n_features, dtype=np.float32)

            # Simulate progressive indicator readiness
            ma5 = 100.0 if bar_idx >= 4 else np.nan
            ma20 = 100.0 if bar_idx >= 19 else np.nan
            atr = 1.0 if bar_idx >= 13 else np.nan
            rsi14 = 50.0 if bar_idx >= 14 else np.nan
            momentum = 0.0 if bar_idx >= 9 else np.nan
            macd = 0.0 if bar_idx >= 25 else np.nan
            macd_signal = 0.0 if bar_idx >= 34 else np.nan  # Won't be ready in first 30
            cci = 0.0 if bar_idx >= 19 else np.nan
            bb_lower = 98.0 if bar_idx >= 19 else np.nan
            bb_upper = 102.0 if bar_idx >= 19 else np.nan

            build_observation_vector(
                100.0, 99.0, 0.0, 0.0,
                ma5, ma20, rsi14, macd, macd_signal, momentum, atr, cci, float(bar_idx),
                bb_lower, bb_upper,
                0.0, 1e9, 50.0, False, False,
                10000.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0, 1, 1, norm_cols, obs
            )

            assert np.all(np.isfinite(obs)), f"Bar {bar_idx} produced NaN/Inf: {obs[~np.isfinite(obs)]}"

    def test_price_zero_edge_case(self):
        """Test that price=0 doesn't break the observation (though unlikely in production)."""
        n_features = compute_n_features([])
        obs = np.zeros(n_features, dtype=np.float32)
        norm_cols = np.zeros(21, dtype=np.float32)

        build_observation_vector(
            price=0.0,  # Edge case: price = 0
            prev_price=0.0,
            log_volume_norm=0.0,
            rel_volume=0.0,
            ma5=np.nan, ma20=np.nan, rsi14=np.nan,
            macd=np.nan, macd_signal=np.nan, momentum=np.nan,
            atr=np.nan, cci=np.nan, obv=0.0,
            bb_lower=np.nan, bb_upper=np.nan,
            is_high_importance=0.0, time_since_event=1e9,
            fear_greed_value=50.0, has_fear_greed=False, risk_off_flag=False,
            cash=10000.0, units=0.0,
            last_vol_imbalance=0.0, last_trade_intensity=0.0,
            last_realized_spread=0.0, last_agent_fill_ratio=0.0,
            token_id=0, max_num_tokens=1, num_tokens=1,
            norm_cols_values=norm_cols, out_features=obs
        )

        assert np.all(np.isfinite(obs)), "price=0 caused NaN/Inf"

    def test_empty_norm_cols(self):
        """Test with empty norm_cols (should still work)."""
        n_features = compute_n_features([])
        obs = np.zeros(n_features, dtype=np.float32)

        # NOTE: norm_cols is always size 21 in current implementation
        # But test with all zeros
        norm_cols = np.zeros(21, dtype=np.float32)

        build_observation_vector(
            100.0, 99.0, 0.0, 0.0,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, np.nan, np.nan,
            0.0, 1e9, 50.0, False, False,
            10000.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0, 1, 1, norm_cols, obs
        )

        assert np.all(np.isfinite(obs)), "Empty norm_cols caused NaN/Inf"

    def test_no_token_metadata(self):
        """Test with max_num_tokens=0 (no token metadata)."""
        # When max_num_tokens=0, token features are not added
        # This changes the observation size, so we need to handle it
        n_features = compute_n_features([])
        obs = np.zeros(n_features, dtype=np.float32)
        norm_cols = np.zeros(21, dtype=np.float32)

        build_observation_vector(
            100.0, 99.0, 0.0, 0.0,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, np.nan, np.nan,
            0.0, 1e9, 50.0, False, False,
            10000.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0,  # token_id
            0,  # max_num_tokens = 0 (no tokens)
            0,  # num_tokens
            norm_cols, obs
        )

        # Only the first part of obs should be filled, rest should be zeros
        # Check that no NaN/Inf in the filled part
        # The exact cutoff depends on implementation, but at least check all finite
        assert np.all(np.isfinite(obs)), "max_num_tokens=0 caused NaN/Inf"

    def test_extreme_values(self):
        """Test with extreme but valid values."""
        n_features = compute_n_features([])
        obs = np.zeros(n_features, dtype=np.float32)
        norm_cols = np.zeros(21, dtype=np.float32)

        # Extreme values
        build_observation_vector(
            price=1e6,  # Very high price
            prev_price=1.0,  # Very low previous price (huge return)
            log_volume_norm=0.999,  # Near tanh limit
            rel_volume=0.999,
            ma5=1e6, ma20=1e6, rsi14=100.0,  # RSI max
            macd=1000.0, macd_signal=1000.0, momentum=1000.0,
            atr=1000.0, cci=300.0, obv=1e12,
            bb_lower=1e6 - 1000, bb_upper=1e6 + 1000,
            is_high_importance=1.0, time_since_event=0.0,
            fear_greed_value=100.0, has_fear_greed=True, risk_off_flag=False,
            cash=1e9, units=1000.0,
            last_vol_imbalance=10.0, last_trade_intensity=10.0,
            last_realized_spread=0.1, last_agent_fill_ratio=1.0,
            token_id=0, max_num_tokens=1, num_tokens=1,
            norm_cols_values=norm_cols, out_features=obs
        )

        assert np.all(np.isfinite(obs)), "Extreme values caused NaN/Inf"


class TestVulnerabilityVULN01:
    """
    Test for VULN-01: vol_proxy uses unprocessed ATR.

    CRITICAL vulnerability identified in OBSERVATION_VECTOR_AUDIT_REPORT.md:
    vol_proxy = tanh(log1p(atr / (price_d + 1e-8)))
    uses the original `atr` parameter which may be NaN, not the processed version.

    This test will FAIL until VULN-01 is fixed.
    """

    @pytest.mark.xfail(reason="VULN-01: vol_proxy uses unprocessed ATR, will fail until fixed")
    def test_vol_proxy_with_nan_atr(self):
        """Test that vol_proxy doesn't produce NaN when ATR is NaN."""
        n_features = compute_n_features([])
        obs = np.zeros(n_features, dtype=np.float32)
        norm_cols = np.zeros(21, dtype=np.float32)

        build_observation_vector(
            100.0, 99.0, 0.0, 0.0,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            np.nan,  # ATR = NaN (first 14 bars)
            np.nan, 0.0, np.nan, np.nan,
            0.0, 1e9, 50.0, False, False,
            10000.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0, 1, 1, norm_cols, obs
        )

        # Index 15 is vol_proxy (approximate)
        vol_proxy_idx = 15
        assert np.isfinite(obs[vol_proxy_idx]), \
            f"vol_proxy (index {vol_proxy_idx}) is NaN/Inf when ATR is NaN: {obs[vol_proxy_idx]}"

        # All features should be finite
        assert np.all(np.isfinite(obs)), f"NaN/Inf found in observation: {obs[~np.isfinite(obs)]}"


class TestProductionScenarios:
    """Test realistic production scenarios."""

    def test_typical_early_episode(self):
        """Simulate a typical early episode with progressive indicator readiness."""
        n_features = compute_n_features([])
        norm_cols = np.zeros(21, dtype=np.float32)

        prices = [100.0, 101.0, 102.0, 101.5, 103.0, 102.0, 104.0, 103.5, 105.0, 104.0]

        for i, price in enumerate(prices):
            obs = np.zeros(n_features, dtype=np.float32)
            prev_price = prices[i - 1] if i > 0 else price

            # Gradual indicator readiness
            ma5 = np.mean(prices[max(0, i - 4):i + 1]) if i >= 4 else np.nan
            ma20 = np.nan  # Not ready in first 10 bars
            atr = 1.0 if i >= 13 else np.nan
            rsi14 = np.nan
            momentum = prices[i] - prices[i - 9] if i >= 9 else np.nan
            macd = np.nan
            macd_signal = np.nan
            cci = np.nan
            obv = float(i * 1000)  # Incremental OBV
            bb_lower = np.nan
            bb_upper = np.nan

            build_observation_vector(
                price, prev_price, 0.5, 0.5,
                ma5, ma20, rsi14, macd, macd_signal, momentum, atr, cci, obv,
                bb_lower, bb_upper,
                0.0, 1e9, 50.0, False, False,
                10000.0 - i * 100, float(i),  # Buying gradually
                0.0, 0.0, 0.0, 0.0,
                0, 1, 1, norm_cols, obs
            )

            assert np.all(np.isfinite(obs)), \
                f"Episode bar {i} (price={price}) produced NaN/Inf: {obs[~np.isfinite(obs)]}"

    def test_fear_greed_variations(self):
        """Test different Fear & Greed values."""
        n_features = compute_n_features([])
        norm_cols = np.zeros(21, dtype=np.float32)

        for fg_value in [0.0, 25.0, 50.0, 75.0, 100.0]:
            obs = np.zeros(n_features, dtype=np.float32)

            build_observation_vector(
                100.0, 99.0, 0.0, 0.0,
                100.0, 100.0, 50.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 98.0, 102.0,
                0.0, 1e9,
                fg_value,  # Fear & Greed
                True,  # has_fear_greed
                fg_value < 25.0,  # risk_off_flag
                10000.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0, 1, 1, norm_cols, obs
            )

            assert np.all(np.isfinite(obs)), f"Fear & Greed {fg_value} caused NaN/Inf"

    def test_bollinger_squeeze_scenarios(self):
        """Test Bollinger Bands squeeze (very narrow bands)."""
        n_features = compute_n_features([])
        norm_cols = np.zeros(21, dtype=np.float32)

        # Very narrow bands (bb_width ≈ 0)
        price = 100.0
        bb_lower = 99.99
        bb_upper = 100.01

        obs = np.zeros(n_features, dtype=np.float32)

        build_observation_vector(
            price, 99.0, 0.0, 0.0,
            100.0, 100.0, 50.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            bb_lower, bb_upper,
            0.0, 1e9, 50.0, False, False,
            10000.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0, 1, 1, norm_cols, obs
        )

        assert np.all(np.isfinite(obs)), "Bollinger squeeze (narrow bands) caused NaN/Inf"


class TestFinalValidation:
    """Final comprehensive validation."""

    def test_no_nan_inf_in_any_feature(self):
        """
        Comprehensive test: build observation with all possible edge cases
        and verify that NO feature contains NaN or Inf.
        """
        n_features = compute_n_features([])
        norm_cols = np.random.randn(21).astype(np.float32)  # Random norm_cols
        norm_cols[0] = np.nan  # One NaN to test _clipf

        test_cases = [
            # (price, prev_price, indicators, description)
            (100.0, 99.0, "all_valid", "All indicators valid"),
            (100.0, 99.0, "all_nan", "All indicators NaN"),
            (0.0, 0.0, "all_nan", "Price=0, all indicators NaN"),
            (100.0, 0.0, "all_nan", "Prev_price=0, all indicators NaN"),
            (1e-6, 1e-6, "all_nan", "Very small price"),
            (1e6, 1e6, "all_valid", "Very large price"),
        ]

        for price, prev_price, ind_type, description in test_cases:
            obs = np.zeros(n_features, dtype=np.float32)

            if ind_type == "all_valid":
                ma5, ma20, rsi14 = 100.0, 100.0, 50.0
                macd, macd_signal, momentum = 0.0, 0.0, 0.0
                atr, cci, obv = 1.0, 0.0, 0.0
                bb_lower, bb_upper = 98.0, 102.0
            else:  # all_nan
                ma5 = ma20 = rsi14 = np.nan
                macd = macd_signal = momentum = np.nan
                atr = cci = obv = np.nan
                bb_lower = bb_upper = np.nan

            build_observation_vector(
                price, prev_price, 0.0, 0.0,
                ma5, ma20, rsi14, macd, macd_signal, momentum, atr, cci, obv,
                bb_lower, bb_upper,
                0.0, 1e9, 50.0, False, False,
                10000.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0, 1, 1, norm_cols, obs
            )

            assert np.all(np.isfinite(obs)), \
                f"Test case '{description}' produced NaN/Inf: {obs[~np.isfinite(obs)]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
