#!/usr/bin/env python3
"""
COMPREHENSIVE TEST: ATR Validity Flag and vol_proxy NaN Prevention

This test suite verifies the fix for the critical bug where ATR without a validity flag
caused NaN propagation in vol_proxy during warmup period.

Problem identified:
- ATR was the ONLY indicator without a validity flag
- vol_proxy calculation used the raw ATR variable, which could be NaN
- This violated the core guarantee "no NaN in observation vector"

Solution implemented:
- Added atr_valid flag (index 16)
- vol_proxy now checks atr_valid before calculation
- If invalid, uses fallback ATR (price * 0.01) for vol_proxy

Test coverage:
1. ✓ ATR valid → atr_valid = 1.0, vol_proxy computed normally
2. ✓ ATR invalid (NaN) → atr_valid = 0.0, fallback used
3. ✓ vol_proxy is NEVER NaN (even when ATR is NaN)
4. ✓ Correct indices (15: atr, 16: atr_valid, 23: vol_proxy)
5. ✓ Fallback values are semantically correct
6. ✓ Consistency with other indicators (RSI, MACD, etc.)

Research references:
- IEEE 754: NaN propagates through all arithmetic operations
- Wilder (1978): ATR requires 14 bars minimum (EMA smoothing)
- "Defense in Depth" (OWASP): Multiple validation layers
- "Fail-Fast Validation" (Martin Fowler): Validate before use
"""

import numpy as np
import pytest
import math

try:
    from obs_builder import build_observation_vector

    HAVE_OBS_BUILDER = True
except ImportError:
    HAVE_OBS_BUILDER = False
    pytest.skip("obs_builder (Cython module) not available", allow_module_level=True)


class TestATRValidityFlag:
    """Test suite for ATR validity flag and vol_proxy NaN prevention."""

    @pytest.fixture
    def valid_params(self):
        """Base valid parameters for observation building."""
        return {
            "price": 1000.0,
            "prev_price": 1000.0,
            "log_volume_norm": 0.5,
            "rel_volume": 0.5,
            "ma5": 1005.0,
            "ma20": 1010.0,
            "rsi14": 50.0,
            "macd": 2.0,
            "macd_signal": 1.5,
            "momentum": 10.0,
            "atr": 15.0,  # Valid ATR (1.5% volatility)
            "cci": 0.0,
            "obv": 1000.0,
            "bb_lower": 990.0,
            "bb_upper": 1010.0,
            "is_high_importance": 0.0,
            "time_since_event": 1.0,
            "fear_greed_value": 50.0,
            "has_fear_greed": True,
            "risk_off_flag": False,
            "cash": 10000.0,
            "units": 1.0,
            "last_vol_imbalance": 0.0,
            "last_trade_intensity": 0.0,
            "last_realized_spread": 0.0,
            "last_agent_fill_ratio": 1.0,
            "token_id": 0,
            "max_num_tokens": 1,
            "num_tokens": 1,
        }

    def test_atr_valid_when_atr_is_valid(self, valid_params):
        """Test 1: When ATR is valid, atr_valid flag = 1.0 and ATR value is stored."""
        obs = np.zeros(63, dtype=np.float32)
        norm_cols = np.zeros(21, dtype=np.float32)
        valid_params["norm_cols_values"] = norm_cols
        valid_params["out_features"] = obs

        build_observation_vector(**valid_params)

        # Index 15: ATR value
        assert obs[15] == pytest.approx(15.0), f"ATR value should be 15.0, got {obs[15]}"

        # Index 16: atr_valid flag
        assert obs[16] == pytest.approx(
            1.0
        ), f"atr_valid should be 1.0 when ATR is valid, got {obs[16]}"

        print("✓ Test 1 passed: ATR valid → atr_valid = 1.0")

    def test_atr_invalid_when_atr_is_nan(self, valid_params):
        """Test 2: When ATR is NaN, atr_valid flag = 0.0 and fallback is used."""
        valid_params["atr"] = float("nan")  # Simulate warmup period

        obs = np.zeros(63, dtype=np.float32)
        norm_cols = np.zeros(21, dtype=np.float32)
        valid_params["norm_cols_values"] = norm_cols
        valid_params["out_features"] = obs

        build_observation_vector(**valid_params)

        # Index 15: ATR fallback (1% of price)
        expected_fallback = valid_params["price"] * 0.01  # 1000.0 * 0.01 = 10.0
        assert obs[15] == pytest.approx(
            expected_fallback
        ), f"ATR fallback should be {expected_fallback}, got {obs[15]}"

        # Index 16: atr_valid flag
        assert obs[16] == pytest.approx(
            0.0
        ), f"atr_valid should be 0.0 when ATR is NaN, got {obs[16]}"

        print("✓ Test 2 passed: ATR NaN → atr_valid = 0.0, fallback used")

    def test_vol_proxy_not_nan_when_atr_is_nan(self, valid_params):
        """
        Test 3: CRITICAL - vol_proxy must NOT be NaN when ATR is NaN (warmup period).

        This is the PRIMARY bug fix being tested. Before adding atr_valid flag,
        vol_proxy would become NaN during warmup, violating the "no NaN" guarantee.
        """
        valid_params["atr"] = float("nan")  # Simulate warmup period

        obs = np.zeros(63, dtype=np.float32)
        norm_cols = np.zeros(21, dtype=np.float32)
        valid_params["norm_cols_values"] = norm_cols
        valid_params["out_features"] = obs

        build_observation_vector(**valid_params)

        # Index 22: vol_proxy (after ret_bar at index 21)
        vol_proxy = obs[22]

        assert not np.isnan(
            vol_proxy
        ), f"vol_proxy MUST NOT be NaN when ATR is NaN! Got {vol_proxy}"

        assert np.isfinite(vol_proxy), f"vol_proxy must be finite, got {vol_proxy}"

        # Expected: vol_proxy calculated with fallback ATR
        # fallback ATR = 1000.0 * 0.01 = 10.0
        # vol_proxy = tanh(log1p(10.0 / 1000.0)) = tanh(log1p(0.01)) ≈ tanh(0.00995) ≈ 0.00995
        expected_vol_proxy = math.tanh(math.log1p(10.0 / 1000.0))

        assert (
            abs(vol_proxy - expected_vol_proxy) < 0.001
        ), f"vol_proxy should be ~{expected_vol_proxy} (calculated with fallback ATR), got {vol_proxy}"

        print(f"✓ Test 3 passed: vol_proxy = {vol_proxy} (NOT NaN, uses fallback ATR)")

    def test_vol_proxy_calculation_with_valid_atr(self, valid_params):
        """Test 4: When ATR is valid, vol_proxy is calculated with real ATR value."""
        valid_params["atr"] = 15.0  # 1.5% volatility

        obs = np.zeros(63, dtype=np.float32)
        norm_cols = np.zeros(21, dtype=np.float32)
        valid_params["norm_cols_values"] = norm_cols
        valid_params["out_features"] = obs

        build_observation_vector(**valid_params)

        vol_proxy = obs[22]

        # Expected: vol_proxy = tanh(log1p(15.0 / 1000.0)) = tanh(log1p(0.015)) ≈ tanh(0.01489) ≈ 0.01488
        expected_vol_proxy = math.tanh(math.log1p(15.0 / 1000.0))

        assert (
            abs(vol_proxy - expected_vol_proxy) < 0.0001
        ), f"vol_proxy should be ~{expected_vol_proxy} (calculated with real ATR), got {vol_proxy}"

        print(f"✓ Test 4 passed: vol_proxy = {vol_proxy} (calculated with real ATR = 15.0)")

    def test_atr_indices_are_correct(self, valid_params):
        """Test 5: Verify ATR and atr_valid are at correct indices (15 and 16)."""
        obs = np.zeros(63, dtype=np.float32)
        norm_cols = np.zeros(21, dtype=np.float32)
        valid_params["norm_cols_values"] = norm_cols
        valid_params["out_features"] = obs

        build_observation_vector(**valid_params)

        # Indices before ATR (should be unchanged from 62-feature system)
        assert obs[0] == pytest.approx(1000.0), "Index 0 should be price"
        assert obs[7] == pytest.approx(50.0), "Index 7 should be rsi14"
        assert obs[8] == pytest.approx(1.0), "Index 8 should be rsi_valid"
        assert obs[13] == pytest.approx(10.0), "Index 13 should be momentum"
        assert obs[14] == pytest.approx(1.0), "Index 14 should be momentum_valid"

        # ATR and its validity flag
        assert obs[15] == pytest.approx(15.0), "Index 15 should be atr"
        assert obs[16] == pytest.approx(1.0), "Index 16 should be atr_valid"

        # Indices after ATR (shifted by +1 from 62-feature system)
        # In 62-feature: cci was at 16, now it's at 17
        assert obs[17] == pytest.approx(0.0), "Index 17 should be cci"
        assert obs[18] == pytest.approx(1.0), "Index 18 should be cci_valid"

        print("✓ Test 5 passed: All indices are correct in 63-feature system")

    def test_atr_fallback_is_reasonable(self, valid_params):
        """Test 6: Fallback ATR (1% of price) is a reasonable volatility estimate."""
        # Test with different prices
        for price in [100.0, 1000.0, 10000.0, 50000.0]:
            valid_params["price"] = price
            valid_params["prev_price"] = price
            valid_params["atr"] = float("nan")

            obs = np.zeros(63, dtype=np.float32)
            norm_cols = np.zeros(21, dtype=np.float32)
            valid_params["norm_cols_values"] = norm_cols
            valid_params["out_features"] = obs

            build_observation_vector(**valid_params)

            expected_fallback = price * 0.01
            actual_atr = obs[15]

            assert actual_atr == pytest.approx(
                expected_fallback
            ), f"For price={price}, ATR fallback should be {expected_fallback}, got {actual_atr}"

        print("✓ Test 6 passed: ATR fallback is proportional to price (1%)")

    def test_consistency_with_other_indicators(self, valid_params):
        """Test 7: ATR validity flag follows same pattern as other indicators."""
        # Test all indicators with NaN to ensure pattern consistency
        indicators_config = [
            ("ma5", 3, 4),
            ("ma20", 5, 6),
            ("rsi14", 7, 8),
            ("macd", 9, 10),
            ("macd_signal", 11, 12),
            ("momentum", 13, 14),
            ("atr", 15, 16),
            ("cci", 17, 18),
            ("obv", 19, 20),
        ]

        for indicator_name, value_idx, flag_idx in indicators_config:
            # Set all to valid first
            params = valid_params.copy()

            # Make this indicator invalid
            params[indicator_name] = float("nan")

            obs = np.zeros(63, dtype=np.float32)
            norm_cols = np.zeros(21, dtype=np.float32)
            params["norm_cols_values"] = norm_cols
            params["out_features"] = obs

            build_observation_vector(**params)

            # Check that validity flag is 0.0
            assert obs[flag_idx] == pytest.approx(
                0.0
            ), f"{indicator_name}_valid (index {flag_idx}) should be 0.0 when {indicator_name} is NaN"

            # Check that value has fallback (not NaN)
            assert not np.isnan(
                obs[value_idx]
            ), f"{indicator_name} value (index {value_idx}) should NOT be NaN (fallback should be used)"

        print("✓ Test 7 passed: ATR follows consistent pattern with other indicators")

    def test_no_nan_in_entire_observation(self, valid_params):
        """Test 8: Comprehensive check - NO feature in observation should be NaN."""
        # Test during "warmup" period - all indicators are NaN
        warmup_params = valid_params.copy()
        warmup_params["ma5"] = float("nan")
        warmup_params["ma20"] = float("nan")
        warmup_params["rsi14"] = float("nan")
        warmup_params["macd"] = float("nan")
        warmup_params["macd_signal"] = float("nan")
        warmup_params["momentum"] = float("nan")
        warmup_params["atr"] = float("nan")  # CRITICAL: ATR is NaN
        warmup_params["cci"] = float("nan")
        warmup_params["obv"] = float("nan")
        warmup_params["bb_lower"] = float("nan")
        warmup_params["bb_upper"] = float("nan")

        obs = np.zeros(63, dtype=np.float32)
        norm_cols = np.zeros(21, dtype=np.float32)
        warmup_params["norm_cols_values"] = norm_cols
        warmup_params["out_features"] = obs

        build_observation_vector(**warmup_params)

        # Check EVERY feature
        nan_indices = []
        for i in range(63):
            if np.isnan(obs[i]):
                nan_indices.append(i)

        assert (
            len(nan_indices) == 0
        ), f"Found NaN at indices: {nan_indices}. Observation MUST NOT contain NaN!"

        print(f"✓ Test 8 passed: No NaN in entire observation (checked all 63 features)")

    def test_warmup_sequence_simulation(self, valid_params):
        """
        Test 9: Simulate first 20 bars to verify correct warmup behavior.

        This simulates a realistic scenario where indicators become valid progressively:
        - Bar 5: MA5 becomes valid
        - Bar 10: Momentum becomes valid
        - Bar 14: ATR becomes valid (CRITICAL)
        - Bar 15: RSI becomes valid
        - Bar 20: MA20, CCI become valid
        """
        scenarios = [
            # (bar, indicators_valid_dict)
            (0, {}),  # No indicators valid
            (5, {"ma5": True}),
            (10, {"ma5": True, "momentum": True}),
            (14, {"ma5": True, "momentum": True, "atr": True}),  # ATR now valid!
            (15, {"ma5": True, "momentum": True, "atr": True, "rsi14": True}),
            (
                20,
                {
                    "ma5": True,
                    "ma20": True,
                    "momentum": True,
                    "atr": True,
                    "rsi14": True,
                    "cci": True,
                },
            ),
        ]

        for bar, valid_indicators in scenarios:
            params = valid_params.copy()

            # Set indicators based on validity
            for indicator in [
                "ma5",
                "ma20",
                "rsi14",
                "macd",
                "macd_signal",
                "momentum",
                "atr",
                "cci",
                "obv",
            ]:
                if indicator not in valid_indicators:
                    params[indicator] = float("nan")

            # Bollinger Bands
            if bar < 20:
                params["bb_lower"] = float("nan")
                params["bb_upper"] = float("nan")

            obs = np.zeros(63, dtype=np.float32)
            norm_cols = np.zeros(21, dtype=np.float32)
            params["norm_cols_values"] = norm_cols
            params["out_features"] = obs

            build_observation_vector(**params)

            # Critical check: vol_proxy (index 22) must NEVER be NaN
            vol_proxy = obs[22]
            assert not np.isnan(
                vol_proxy
            ), f"Bar {bar}: vol_proxy is NaN! This should NEVER happen."

            # Check ATR validity flag
            atr_valid = obs[16]
            if "atr" in valid_indicators:
                assert atr_valid == 1.0, f"Bar {bar}: atr_valid should be 1.0"
            else:
                assert atr_valid == 0.0, f"Bar {bar}: atr_valid should be 0.0"

        print("✓ Test 9 passed: Warmup sequence simulation - vol_proxy never NaN")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST: ATR Validity Flag & vol_proxy NaN Prevention")
    print("=" * 80 + "\n")

    test_instance = TestATRValidityFlag()
    valid_params_fixture = test_instance.valid_params()

    tests = [
        ("Test 1: ATR valid case", test_instance.test_atr_valid_when_atr_is_valid),
        ("Test 2: ATR invalid (NaN) case", test_instance.test_atr_invalid_when_atr_is_nan),
        (
            "Test 3: vol_proxy NOT NaN (CRITICAL)",
            test_instance.test_vol_proxy_not_nan_when_atr_is_nan,
        ),
        (
            "Test 4: vol_proxy with valid ATR",
            test_instance.test_vol_proxy_calculation_with_valid_atr,
        ),
        ("Test 5: Correct indices", test_instance.test_atr_indices_are_correct),
        ("Test 6: Fallback reasonableness", test_instance.test_atr_fallback_is_reasonable),
        (
            "Test 7: Consistency with other indicators",
            test_instance.test_consistency_with_other_indicators,
        ),
        ("Test 8: No NaN in entire observation", test_instance.test_no_nan_in_entire_observation),
        ("Test 9: Warmup sequence simulation", test_instance.test_warmup_sequence_simulation),
    ]

    for test_name, test_func in tests:
        print(f"\n{test_name}...")
        try:
            test_func(valid_params_fixture)
        except Exception as e:
            print(f"❌ FAILED: {e}")
            raise

    print("\n" + "=" * 80)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 80)
    print("\nConclusion:")
    print("✅ ATR validity flag correctly implemented")
    print("✅ vol_proxy NEVER becomes NaN (even during warmup)")
    print("✅ Consistent pattern with other 6 indicators")
    print("✅ Fallback values are semantically correct")
    print("✅ No NaN in observation vector (core guarantee maintained)")
    print("=" * 80)
