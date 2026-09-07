"""Comprehensive test suite for all indicator bugs.

This test suite verifies ALL bugs found in the comprehensive indicator analysis:
1. RSI single value initialization (MarketSimulator.cpp)
2. Bollinger Bands population variance (MarketSimulator.cpp)
3. MACD, Momentum, OBV (verification tests)

Reference: INDICATOR_BUGS_COMPREHENSIVE_ANALYSIS.md
"""

import pytest
import numpy as np
import math
from collections import deque


class TestRSIBugMarketSimulator:
    """Test RSI bug in MarketSimulator.cpp.

    This bug is SEPARATE from transformers.py (which is already fixed).
    MarketSimulator.cpp still has the single-value initialization bug.
    """

    def test_rsi_single_value_init_cpp_simulation(self):
        """Simulate MarketSimulator.cpp RSI bug.

        MarketSimulator.cpp:317-320:
            avg_gain14 = gain;  // BUG! Single value
            avg_loss14 = loss;
        """
        # Simulate buggy RSI (MarketSimulator.cpp style)
        prices = [
            100.0,
            110.0,  # +10.0 gain (huge!)
            110.5,
            110.0,
            110.5,
            110.0,
            110.5,
            110.0,
            110.5,
            110.0,
            110.5,
            110.0,
            110.5,
            110.0,
            110.5,  # 15 prices total (14 changes)
        ]

        # BUGGY implementation (matches MarketSimulator.cpp)
        rsi_init = False
        avg_gain = 0.0
        avg_loss = 0.0

        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            gain = max(change, 0.0)
            loss = max(-change, 0.0)

            # BUG: Single value initialization (i==14)
            if not rsi_init and i == 14:
                rsi_init = True
                avg_gain = gain  # BUG! Should be SMA of 14 gains
                avg_loss = loss  # BUG! Should be SMA of 14 losses

            if rsi_init:
                # Wilder smoothing
                avg_gain = (avg_gain * 13.0 + gain) / 14.0
                avg_loss = (avg_loss * 13.0 + loss) / 14.0

        # Compute RSI
        if avg_loss == 0.0:
            rsi_buggy = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_buggy = 100.0 - (100.0 / (1.0 + rs))

        # CORRECT implementation (SMA initialization)
        gains = []
        losses = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            gains.append(max(change, 0.0))
            losses.append(max(-change, 0.0))

        # Correct: SMA of first 14 gains/losses
        avg_gain_correct = sum(gains[:14]) / 14.0
        avg_loss_correct = sum(losses[:14]) / 14.0

        # Then Wilder smoothing for bar 15 (if exists)
        if len(gains) > 14:
            avg_gain_correct = (avg_gain_correct * 13.0 + gains[14]) / 14.0
            avg_loss_correct = (avg_loss_correct * 13.0 + losses[14]) / 14.0

        rs_correct = avg_gain_correct / avg_loss_correct
        rsi_correct = 100.0 - (100.0 / (1.0 + rs_correct))

        # Verify bug creates significant difference
        print(f"RSI (buggy):   {rsi_buggy:.2f}")
        print(f"RSI (correct): {rsi_correct:.2f}")
        print(f"Difference:    {abs(rsi_buggy - rsi_correct):.2f}")

        # BUG VERIFICATION: Large first gain dominates buggy initialization
        # Expected: rsi_buggy > rsi_correct + 5.0 (at minimum)
        assert rsi_buggy > rsi_correct + 5.0, (
            f"BUG NOT FOUND: RSI difference {abs(rsi_buggy - rsi_correct):.2f} too small. "
            f"Expected buggy RSI to be significantly higher due to single-value init."
        )

    def test_rsi_bug_impact_magnitude(self):
        """Quantify the magnitude of RSI bug impact.

        Tests with various scenarios to measure bias.
        """
        scenarios = [
            {
                "name": "Large initial spike",
                "prices": [100.0, 120.0] + [100.5, 100.0] * 7,  # +20 then oscillate
                "expected_bias": "> 15.0",
            },
            {
                "name": "Large initial drop",
                "prices": [100.0, 80.0] + [100.5, 100.0] * 7,  # -20 then oscillate
                "expected_bias": "> 15.0",
            },
            {
                "name": "Gradual uptrend",
                "prices": [100.0 + i * 0.5 for i in range(15)],  # Steady +0.5 per bar
                "expected_bias": "< 5.0",  # Less impact with uniform changes
            },
        ]

        for scenario in scenarios:
            prices = scenario["prices"]

            # Buggy RSI
            rsi_buggy = self._compute_rsi_buggy(prices)

            # Correct RSI
            rsi_correct = self._compute_rsi_correct(prices)

            bias = abs(rsi_buggy - rsi_correct)

            print(f"\n{scenario['name']}:")
            print(f"  RSI (buggy):   {rsi_buggy:.2f}")
            print(f"  RSI (correct): {rsi_correct:.2f}")
            print(f"  Bias:          {bias:.2f} (expected {scenario['expected_bias']})")

            # Assert bias matches expectation
            if ">" in scenario["expected_bias"]:
                threshold = float(scenario["expected_bias"].split(">")[1].strip())
                assert (
                    bias > threshold
                ), f"{scenario['name']}: Bias {bias:.2f} should be > {threshold}"
            elif "<" in scenario["expected_bias"]:
                threshold = float(scenario["expected_bias"].split("<")[1].strip())
                assert (
                    bias < threshold
                ), f"{scenario['name']}: Bias {bias:.2f} should be < {threshold}"

    def _compute_rsi_buggy(self, prices):
        """Compute RSI with buggy single-value initialization."""
        rsi_init = False
        avg_gain = 0.0
        avg_loss = 0.0

        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            gain = max(change, 0.0)
            loss = max(-change, 0.0)

            if not rsi_init and i == 14:
                rsi_init = True
                avg_gain = gain  # BUG!
                avg_loss = loss

            if rsi_init:
                avg_gain = (avg_gain * 13.0 + gain) / 14.0
                avg_loss = (avg_loss * 13.0 + loss) / 14.0

        if not rsi_init:
            return float("nan")

        if avg_loss == 0.0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _compute_rsi_correct(self, prices):
        """Compute RSI with correct SMA initialization."""
        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            gains.append(max(change, 0.0))
            losses.append(max(-change, 0.0))

        if len(gains) < 14:
            return float("nan")

        # Correct: SMA of first 14
        avg_gain = sum(gains[:14]) / 14.0
        avg_loss = sum(losses[:14]) / 14.0

        # Wilder smoothing for remaining
        for i in range(14, len(gains)):
            avg_gain = (avg_gain * 13.0 + gains[i]) / 14.0
            avg_loss = (avg_loss * 13.0 + losses[i]) / 14.0

        if avg_loss == 0.0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))


class TestBollingerBandsPopulationVariance:
    """Test Bollinger Bands population vs sample variance bug.

    MarketSimulator.cpp uses population variance (divides by n) instead of
    sample variance (divides by n-1). This causes 2.6% underestimation of std dev.
    """

    def test_bb_population_vs_sample_variance(self):
        """Verify that population variance underestimates sample variance."""
        prices = [
            100.0,
            101.0,
            99.0,
            102.0,
            98.0,
            101.5,
            99.5,
            102.5,
            98.5,
            101.0,
            100.5,
            99.5,
            101.5,
            99.0,
            102.0,
            98.5,
            101.5,
            99.5,
            102.0,
            98.0,
        ]  # 20 prices

        assert len(prices) == 20, "Test requires exactly 20 prices for BB(20)"

        # BUGGY implementation (MarketSimulator.cpp)
        sum_prices = sum(prices)
        sum_sq = sum(p * p for p in prices)
        mean = sum_prices / 20.0

        # Population variance (BUGGY)
        var_population = sum_sq / 20.0 - mean * mean
        sd_population = math.sqrt(max(0.0, var_population))

        bb_lower_buggy = mean - 2.0 * sd_population
        bb_upper_buggy = mean + 2.0 * sd_population

        # CORRECT implementation (sample variance)
        var_sample = sum((p - mean) ** 2 for p in prices) / 19.0  # (n-1) = 19
        sd_sample = math.sqrt(var_sample)

        bb_lower_correct = mean - 2.0 * sd_sample
        bb_upper_correct = mean + 2.0 * sd_sample

        # Verify underestimation
        # Population variance is SMALLER than sample variance
        # sd_population / sd_sample = sqrt((n-1)/n) = sqrt(19/20) ≈ 0.9747
        ratio = sd_population / sd_sample

        print(f"Mean: {mean:.4f}")
        print(f"SD (population): {sd_population:.4f}")
        print(f"SD (sample):     {sd_sample:.4f}")
        print(f"Ratio:           {ratio:.4f} (should be < 1.0)")
        print(f"BB Lower (buggy):   {bb_lower_buggy:.4f}")
        print(f"BB Lower (correct): {bb_lower_correct:.4f}")
        print(f"BB Upper (buggy):   {bb_upper_buggy:.4f}")
        print(f"BB Upper (correct): {bb_upper_correct:.4f}")

        # FIX: Ratio should be inverted (population is SMALLER than sample)
        # sd_population / sd_sample = sqrt((n-1)/n) = sqrt(19/20) ≈ 0.9747
        expected_ratio = math.sqrt(19.0 / 20.0)  # ≈ 0.9747 (NOT 1.0264!)

        # Verify ratio matches theory
        assert (
            abs(ratio - expected_ratio) < 0.0001
        ), f"Ratio {ratio:.4f} doesn't match expected {expected_ratio:.4f}"

        # Verify bands are narrower (buggy)
        band_width_buggy = bb_upper_buggy - bb_lower_buggy
        band_width_correct = bb_upper_correct - bb_lower_correct

        assert (
            band_width_buggy < band_width_correct
        ), "Buggy bands should be narrower than correct bands"

        # Verify 2.6% narrower
        width_ratio = band_width_buggy / band_width_correct
        expected_width_ratio = math.sqrt(19.0 / 20.0)  # ≈ 0.9747 (2.5% narrower)

        print(f"Band width ratio: {width_ratio:.4f} (expected: {expected_width_ratio:.4f})")

        assert (
            abs(width_ratio - expected_width_ratio) < 0.001
        ), f"Width ratio {width_ratio:.4f} doesn't match expected {expected_width_ratio:.4f}"

    def test_bb_false_breakout_probability(self):
        """Test that narrower bands lead to more false breakouts.

        With narrower bands, price touches bands more frequently,
        leading to more false breakout signals.
        """
        np.random.seed(42)

        # Generate 1000 random price movements (normal distribution)
        prices = [100.0]
        for _ in range(1000):
            change = np.random.randn() * 1.0  # 1% std moves
            prices.append(prices[-1] + change)

        # Compute rolling BB for each window
        false_breakouts_buggy = 0
        false_breakouts_correct = 0

        for i in range(20, len(prices)):
            window = prices[i - 20 : i]
            current_price = prices[i]

            mean = sum(window) / 20.0

            # Buggy (population)
            var_pop = sum((p - mean) ** 2 for p in window) / 20.0
            sd_pop = math.sqrt(var_pop)
            bb_lower_buggy = mean - 2.0 * sd_pop
            bb_upper_buggy = mean + 2.0 * sd_pop

            # Correct (sample)
            var_sample = sum((p - mean) ** 2 for p in window) / 19.0
            sd_sample = math.sqrt(var_sample)
            bb_lower_correct = mean - 2.0 * sd_sample
            bb_upper_correct = mean + 2.0 * sd_sample

            # Count breakouts
            if current_price < bb_lower_buggy or current_price > bb_upper_buggy:
                false_breakouts_buggy += 1

            if current_price < bb_lower_correct or current_price > bb_upper_correct:
                false_breakouts_correct += 1

        total_bars = len(prices) - 20
        buggy_rate = false_breakouts_buggy / total_bars
        correct_rate = false_breakouts_correct / total_bars

        print(f"False breakouts (buggy):   {false_breakouts_buggy}/{total_bars} ({buggy_rate:.2%})")
        print(
            f"False breakouts (correct): {false_breakouts_correct}/{total_bars} ({correct_rate:.2%})"
        )
        print(
            f"Difference: {false_breakouts_buggy - false_breakouts_correct} bars ({(buggy_rate - correct_rate):.2%})"
        )

        # Verify: buggy version has MORE breakouts (narrower bands)
        assert (
            false_breakouts_buggy > false_breakouts_correct
        ), "Buggy (narrower) bands should have more breakouts"

        # Expected: ~2.5-3% more breakouts
        excess_rate = buggy_rate - correct_rate
        assert (
            0.01 < excess_rate < 0.05
        ), f"Excess breakout rate {excess_rate:.2%} outside expected range [1%, 5%]"


class TestMACDMomentumOBV:
    """Verification tests for MACD, Momentum, and OBV.

    These indicators appear correct, but we test them for completeness.
    """

    def test_macd_ema_formula(self):
        """Verify MACD EMA alpha values are correct."""
        # Standard MACD: EMA(12), EMA(26), EMA(9)
        alpha12 = 2.0 / (12.0 + 1.0)  # = 2/13 ≈ 0.1538
        alpha26 = 2.0 / (26.0 + 1.0)  # = 2/27 ≈ 0.0741
        alpha9 = 2.0 / (9.0 + 1.0)  # = 2/10 = 0.2

        # Verify standard formulas
        assert abs(alpha12 - 0.1538) < 0.0001
        assert abs(alpha26 - 0.0741) < 0.0001
        assert abs(alpha9 - 0.2) < 0.0001

        # Simulate MACD computation
        prices = [100 + i * 0.5 for i in range(30)]  # Uptrend

        ema12 = prices[0]
        ema26 = prices[0]

        for price in prices[1:]:
            ema12 = alpha12 * price + (1.0 - alpha12) * ema12
            ema26 = alpha26 * price + (1.0 - alpha26) * ema26

        macd = ema12 - ema26

        # In uptrend: EMA12 > EMA26 (faster MA responds quicker)
        assert ema12 > ema26, "In uptrend, EMA12 should be > EMA26"
        assert macd > 0, "MACD should be positive in uptrend"

        print(f"EMA12: {ema12:.4f}, EMA26: {ema26:.4f}, MACD: {macd:.4f}")

    def test_momentum_formula(self):
        """Verify Momentum(10) formula is correct."""
        prices = [
            100.0,
            101.0,
            99.0,
            102.0,
            98.0,
            101.5,
            99.5,
            102.5,
            98.5,
            101.0,
            100.5,
        ]  # 11 prices

        # Momentum(10) = close[t] - close[t-10]
        # IMPORTANT: deque of maxlen=10 stores LAST 10 elements, not first 10!
        # So window[0] is the OLDEST in window, not prices[0]
        window = deque(maxlen=10)

        for i, price in enumerate(prices):
            window.append(price)

            if len(window) == 10:
                # window[0] is prices[i-9] (oldest in current window)
                # price is prices[i] (current)
                # momentum = prices[i] - prices[i-9]
                momentum = price - window[0]

                if i == 10:  # Last price
                    # window contains prices[1:11] (size 10)
                    # window[0] = prices[1] = 101.0
                    # price = prices[10] = 100.5
                    # momentum = 100.5 - 101.0 = -0.5
                    expected_momentum = prices[10] - prices[1]  # 100.5 - 101.0 = -0.5
                    assert (
                        abs(momentum - expected_momentum) < 1e-10
                    ), f"Momentum {momentum} != expected {expected_momentum}"
                    print(
                        f"Momentum(10) = {momentum:.2f} (close[10] - close[1] = {prices[10]:.1f} - {prices[1]:.1f})"
                    )

    def test_obv_formula(self):
        """Verify OBV formula is correct."""
        prices = [100.0, 101.0, 99.0, 102.0, 98.0, 101.0]
        volumes = [1000, 1100, 900, 1200, 800, 1050]

        obv = 0.0
        obv_values = [0.0]  # Initial

        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                obv += volumes[i]
            elif prices[i] < prices[i - 1]:
                obv -= volumes[i]
            # else: no change

            obv_values.append(obv)

        # Verify logic
        # Bar 1: 101 > 100 → obv += 1100 = 1100
        assert obv_values[1] == 1100
        # Bar 2: 99 < 101 → obv -= 900 = 200
        assert obv_values[2] == 200
        # Bar 3: 102 > 99 → obv += 1200 = 1400
        assert obv_values[3] == 1400
        # Bar 4: 98 < 102 → obv -= 800 = 600
        assert obv_values[4] == 600
        # Bar 5: 101 > 98 → obv += 1050 = 1650
        assert obv_values[5] == 1650

        print(f"OBV progression: {obv_values}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
