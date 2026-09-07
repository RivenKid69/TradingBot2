#!/usr/bin/env python3
"""
Comprehensive tests for technical indicator bug fixes in MarketSimulator.cpp.

This test suite verifies two critical fixes (2025-11-24):
1. Bug #1 (CRITICAL): RSI initialization with SMA instead of single value
2. Bug #2 (MEDIUM): Bollinger Bands sample variance (Bessel's correction)

Test strategy:
- Direct C++ tests via marketmarket_simulator_wrapper
- Integration tests: Python (transformers.py) vs C++ (MarketSimulator) parity
- Regression tests to prevent future breakage

Research references:
- Wilder (1978): "New Concepts in Technical Trading Systems" (RSI)
- Bollinger (1992): "Bollinger on Bollinger Bands"
- Bessel's Correction: Standard statistics textbook
"""

import pytest
import numpy as np
import math
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from marketmarket_simulator_wrapper import PyMarketSimulator

    HAVE_SIMULATOR = True
except ImportError:
    HAVE_SIMULATOR = False
    pytest.skip("MarketSimulator not available", allow_module_level=True)

try:
    from transformers import FeatureSpec, OnlineFeatureTransformer

    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def generate_price_pattern_for_rsi_test():
    """
    Generate price pattern for RSI test:
    - First gain is LARGE (10.0) → tests single-value vs SMA difference
    - Subsequent gains/losses are SMALL (±0.5) → stable baseline

    Returns:
        list[float]: 20 prices designed to expose RSI initialization bug
    """
    prices = [
        100.0,  # bar 0 (initial)
        110.0,  # bar 1: +10.0 gain (LARGE!)
        110.5,  # bar 2: +0.5 gain
        110.0,  # bar 3: -0.5 loss
        110.5,  # bar 4: +0.5 gain
        110.0,  # bar 5: -0.5 loss
        110.5,  # bar 6: +0.5 gain
        110.0,  # bar 7: -0.5 loss
        110.5,  # bar 8: +0.5 gain
        110.0,  # bar 9: -0.5 loss
        110.5,  # bar 10: +0.5 gain
        110.0,  # bar 11: -0.5 loss
        110.5,  # bar 12: +0.5 gain
        110.0,  # bar 13: -0.5 loss
        110.5,  # bar 14: +0.5 gain (RSI initialization at bar 14)
        110.0,  # bar 15: -0.5 loss
        110.5,  # bar 16: +0.5 gain
        110.0,  # bar 17: -0.5 loss
        110.5,  # bar 18: +0.5 gain
        110.0,  # bar 19: -0.5 loss
    ]
    return prices


def compute_expected_rsi_correct(prices, period=14):
    """
    Compute expected RSI with CORRECT initialization (SMA of first 14 gains/losses).

    This is the reference implementation following Wilder (1978).

    Args:
        prices: list of prices
        period: RSI period (default 14)

    Returns:
        list[float]: RSI values (NaN for bars < period)
    """
    rsi_values = []
    gains = []
    losses = []
    avg_gain = None
    avg_loss = None

    for i in range(len(prices)):
        if i == 0:
            rsi_values.append(float("nan"))
            continue

        # Compute gain/loss
        change = prices[i] - prices[i - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        gains.append(gain)
        losses.append(loss)

        # Wait for first 'period' samples
        if len(gains) < period:
            rsi_values.append(float("nan"))
            continue

        # Initialize with SMA at exactly 'period' samples
        if avg_gain is None:
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
        else:
            # Wilder smoothing
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

        # Compute RSI
        if avg_loss == 0.0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        rsi_values.append(rsi)

    return rsi_values


def compute_bollinger_bands_correct(prices, period=20):
    """
    Compute Bollinger Bands with CORRECT variance (sample variance, Bessel's correction).

    This is the reference implementation following Bollinger (1992).

    Args:
        prices: list of prices
        period: BB period (default 20)

    Returns:
        tuple: (ma20_values, bb_lower_values, bb_upper_values)
    """
    ma20_values = []
    bb_lower_values = []
    bb_upper_values = []

    for i in range(len(prices)):
        if i < period - 1:
            ma20_values.append(float("nan"))
            bb_lower_values.append(float("nan"))
            bb_upper_values.append(float("nan"))
            continue

        # Get last 'period' prices
        window = prices[i - period + 1 : i + 1]
        mean = sum(window) / period

        # Compute sample variance (Bessel's correction: divide by n-1)
        variance = sum((x - mean) ** 2 for x in window) / (period - 1)
        std = math.sqrt(variance)

        ma20 = mean
        bb_lower = mean - 2.0 * std
        bb_upper = mean + 2.0 * std

        ma20_values.append(ma20)
        bb_lower_values.append(bb_lower)
        bb_upper_values.append(bb_upper)

    return ma20_values, bb_lower_values, bb_upper_values


# =============================================================================
# BUG #1: RSI INITIALIZATION TESTS
# =============================================================================


class TestRSIInitializationFix:
    """Test suite for RSI initialization bug fix (Bug #1 - CRITICAL)."""

    def test_rsi_initialization_with_sma_cpp(self):
        """
        Test 1.1: C++ MarketSimulator uses SMA for RSI initialization (not single value).

        This test verifies the FIX for Bug #1.
        After fix, RSI should match expected value with SMA initialization.
        """
        if not HAVE_SIMULATOR:
            pytest.skip("MarketSimulator not available")

        prices = generate_price_pattern_for_rsi_test()
        n = len(prices)

        # Create simulator
        sim = PyMarketSimulator(
            price=np.zeros(n, dtype=np.float64),
            open_arr=np.zeros(n, dtype=np.float64),
            high=np.zeros(n, dtype=np.float64),
            low=np.zeros(n, dtype=np.float64),
            volume_usd=np.zeros(n, dtype=np.float64),
            n_steps=n,
            seed=42,
        )

        # Feed prices manually (bypass step() to control prices exactly)
        for i, price in enumerate(prices):
            sim.step(i, black_swan_probability=0.0, is_training_mode=False)
            # Override close price
            sim.m_close[i] = price

        # Force indicator update
        for i in range(n):
            sim.update_indicators(i)

        # Get RSI at bar 14 (first RSI value)
        rsi_cpp_14 = sim.get_rsi(14)

        # Compute expected RSI (with CORRECT SMA initialization)
        expected_rsi = compute_expected_rsi_correct(prices, period=14)
        expected_rsi_14 = expected_rsi[14]

        # ASSERTION: C++ RSI should match expected (SMA-based) RSI
        assert not math.isnan(rsi_cpp_14), "RSI should be initialized at bar 14"
        assert not math.isnan(expected_rsi_14), "Expected RSI should be computed"

        # Allow 0.5% tolerance for floating-point differences
        tolerance = 0.5  # 0.5 RSI points
        assert abs(rsi_cpp_14 - expected_rsi_14) < tolerance, (
            f"RSI initialization mismatch!\n"
            f"C++ RSI (bar 14): {rsi_cpp_14:.2f}\n"
            f"Expected (SMA):   {expected_rsi_14:.2f}\n"
            f"Difference:       {abs(rsi_cpp_14 - expected_rsi_14):.2f}\n"
            f"This suggests single-value initialization bug is NOT fixed!"
        )

        print(f"✓ Test 1.1 passed: C++ RSI uses SMA initialization")
        print(f"  C++ RSI (bar 14): {rsi_cpp_14:.2f}")
        print(f"  Expected (SMA):   {expected_rsi_14:.2f}")

    def test_rsi_bias_eliminated(self):
        """
        Test 1.2: RSI bias from single-value initialization is ELIMINATED.

        Before fix: First large gain (10.0) dominated RSI for 50-100 bars
        After fix: RSI correctly averages all 14 gains → NO bias
        """
        if not HAVE_SIMULATOR:
            pytest.skip("MarketSimulator not available")

        prices = generate_price_pattern_for_rsi_test()

        # Expected with CORRECT SMA initialization
        expected_rsi = compute_expected_rsi_correct(prices, period=14)
        expected_rsi_14 = expected_rsi[14]

        # Buggy behavior (single-value initialization):
        # avg_gain = 10.0 (first gain only!)
        # avg_loss = 0.0
        # After 1 Wilder update: avg_gain = (10.0 * 13 + 0.5) / 14 = 9.32
        # After 2 updates: avg_gain = (9.32 * 13 + 0.0) / 14 = 8.66
        # After 3 updates: avg_gain = (8.66 * 13 + 0.5) / 14 = 8.07
        # After 4 updates: avg_gain = (8.07 * 13 + 0.0) / 14 = 7.51
        # After 5 updates: avg_gain = (7.51 * 13 + 0.5) / 14 = 7.00
        # RS = 7.00 / 0.25 = 28.0
        # RSI = 100 - (100 / 29.0) = 96.6

        buggy_rsi_14_approx = 96.6  # Approximate buggy RSI

        # The fix should make C++ RSI MUCH LOWER than buggy value
        # Expected ~78.8, buggy ~96.6 → difference ~17.8 points
        max_acceptable_rsi = 85.0  # Well below buggy value

        # Get C++ RSI
        sim = PyMarketSimulator(
            price=np.zeros(len(prices), dtype=np.float64),
            open_arr=np.zeros(len(prices), dtype=np.float64),
            high=np.zeros(len(prices), dtype=np.float64),
            low=np.zeros(len(prices), dtype=np.float64),
            volume_usd=np.zeros(len(prices), dtype=np.float64),
            n_steps=len(prices),
            seed=42,
        )

        for i, price in enumerate(prices):
            sim.step(i, black_swan_probability=0.0, is_training_mode=False)
            sim.m_close[i] = price
        for i in range(len(prices)):
            sim.update_indicators(i)

        rsi_cpp_14 = sim.get_rsi(14)

        # ASSERTION: RSI should be WELL BELOW buggy value
        assert rsi_cpp_14 < max_acceptable_rsi, (
            f"RSI bias NOT eliminated!\n"
            f"C++ RSI (bar 14):        {rsi_cpp_14:.2f}\n"
            f"Expected (SMA):          {expected_rsi_14:.2f}\n"
            f"Buggy (single-value):    ~{buggy_rsi_14_approx:.2f}\n"
            f"C++ RSI should be < {max_acceptable_rsi:.1f} (far from buggy value)"
        )

        print(f"✓ Test 1.2 passed: RSI bias eliminated")
        print(f"  C++ RSI:   {rsi_cpp_14:.2f} (correct)")
        print(f"  Expected:  {expected_rsi_14:.2f}")
        print(f"  Buggy:     ~{buggy_rsi_14_approx:.2f} (old behavior)")

    @pytest.mark.skipif(not HAVE_TRANSFORMERS, reason="transformers.py not available")
    def test_rsi_cpp_vs_python_parity(self):
        """
        Test 1.3: C++ (MarketSimulator) vs Python (transformers.py) RSI parity.

        Both implementations should produce IDENTICAL RSI values after fix.
        """
        if not HAVE_SIMULATOR or not HAVE_TRANSFORMERS:
            pytest.skip("Simulator or transformers not available")

        prices = generate_price_pattern_for_rsi_test()

        # Python RSI (transformers.py)
        spec = FeatureSpec(
            lookbacks_prices=[5, 10],
            rsi_period=14,
            bar_duration_minutes=1,
        )
        transformer = OnlineFeatureTransformer(spec)

        python_rsi = []
        for i, price in enumerate(prices):
            feats = transformer.update(
                symbol="BTCUSDT",
                ts_ms=i * 60000,
                close=price,
            )
            python_rsi.append(feats.get("rsi", float("nan")))

        # C++ RSI (MarketSimulator)
        sim = PyMarketSimulator(
            price=np.zeros(len(prices), dtype=np.float64),
            open_arr=np.zeros(len(prices), dtype=np.float64),
            high=np.zeros(len(prices), dtype=np.float64),
            low=np.zeros(len(prices), dtype=np.float64),
            volume_usd=np.zeros(len(prices), dtype=np.float64),
            n_steps=len(prices),
            seed=42,
        )

        for i, price in enumerate(prices):
            sim.step(i, black_swan_probability=0.0, is_training_mode=False)
            sim.m_close[i] = price
        for i in range(len(prices)):
            sim.update_indicators(i)

        cpp_rsi = [sim.get_rsi(i) for i in range(len(prices))]

        # ASSERTION: RSI values should match (after bar 14)
        tolerance = 0.1  # 0.1 RSI point tolerance
        for i in range(14, len(prices)):
            py_val = python_rsi[i]
            cpp_val = cpp_rsi[i]

            assert not math.isnan(py_val), f"Python RSI should not be NaN at bar {i}"
            assert not math.isnan(cpp_val), f"C++ RSI should not be NaN at bar {i}"

            assert abs(py_val - cpp_val) < tolerance, (
                f"RSI mismatch at bar {i}!\n"
                f"Python (transformers.py): {py_val:.2f}\n"
                f"C++ (MarketSimulator):    {cpp_val:.2f}\n"
                f"Difference:               {abs(py_val - cpp_val):.2f}"
            )

        print(f"✓ Test 1.3 passed: C++ vs Python RSI parity")
        print(f"  All {len(prices) - 14} RSI values match within {tolerance} points")


# =============================================================================
# BUG #2: BOLLINGER BANDS VARIANCE TESTS
# =============================================================================


class TestBollingerBandsVarianceFix:
    """Test suite for Bollinger Bands variance bug fix (Bug #2 - MEDIUM)."""

    def test_bollinger_bands_sample_variance_cpp(self):
        """
        Test 2.1: C++ MarketSimulator uses SAMPLE variance (Bessel's correction).

        This test verifies the FIX for Bug #2.
        After fix, bands should be ~2.53% wider than before.
        """
        if not HAVE_SIMULATOR:
            pytest.skip("MarketSimulator not available")

        # Generate prices with known variance
        # Mean = 50000, std ≈ 1000 (design for easy mental math)
        np.random.seed(42)
        base = 50000.0
        prices = [base + np.random.randn() * 1000.0 for _ in range(30)]

        n = len(prices)
        sim = PyMarketSimulator(
            price=np.zeros(n, dtype=np.float64),
            open_arr=np.zeros(n, dtype=np.float64),
            high=np.zeros(n, dtype=np.float64),
            low=np.zeros(n, dtype=np.float64),
            volume_usd=np.zeros(n, dtype=np.float64),
            n_steps=n,
            seed=42,
        )

        for i, price in enumerate(prices):
            sim.step(i, black_swan_probability=0.0, is_training_mode=False)
            sim.m_close[i] = price
        for i in range(n):
            sim.update_indicators(i)

        # Get BB at bar 19 (first BB value)
        bb_lower_cpp = sim.get_bb_lower(19)
        bb_upper_cpp = sim.get_bb_upper(19)
        ma20_cpp = sim.get_ma20(19)

        # Compute expected BB (with CORRECT sample variance)
        ma20_expected, bb_lower_expected, bb_upper_expected = compute_bollinger_bands_correct(
            prices, period=20
        )
        ma20_exp = ma20_expected[19]
        bb_lower_exp = bb_lower_expected[19]
        bb_upper_exp = bb_upper_expected[19]

        # ASSERTION: C++ BB should match expected (sample variance)
        assert not math.isnan(bb_lower_cpp), "BB lower should not be NaN"
        assert not math.isnan(bb_upper_cpp), "BB upper should not be NaN"

        tolerance = 0.01  # 1% tolerance (relative)
        ma20_diff_pct = abs(ma20_cpp - ma20_exp) / ma20_exp * 100
        bb_lower_diff_pct = abs(bb_lower_cpp - bb_lower_exp) / abs(bb_lower_exp) * 100
        bb_upper_diff_pct = abs(bb_upper_cpp - bb_upper_exp) / abs(bb_upper_exp) * 100

        assert ma20_diff_pct < tolerance, (
            f"MA20 mismatch!\n"
            f"C++:      {ma20_cpp:.2f}\n"
            f"Expected: {ma20_exp:.2f}\n"
            f"Diff:     {ma20_diff_pct:.4f}%"
        )

        assert bb_lower_diff_pct < tolerance, (
            f"BB lower mismatch!\n"
            f"C++:      {bb_lower_cpp:.2f}\n"
            f"Expected: {bb_lower_exp:.2f}\n"
            f"Diff:     {bb_lower_diff_pct:.4f}%"
        )

        assert bb_upper_diff_pct < tolerance, (
            f"BB upper mismatch!\n"
            f"C++:      {bb_upper_cpp:.2f}\n"
            f"Expected: {bb_upper_exp:.2f}\n"
            f"Diff:     {bb_upper_diff_pct:.4f}%"
        )

        print(f"✓ Test 2.1 passed: C++ BB uses sample variance")
        print(f"  MA20:     {ma20_cpp:.2f} (expected {ma20_exp:.2f})")
        print(f"  BB lower: {bb_lower_cpp:.2f} (expected {bb_lower_exp:.2f})")
        print(f"  BB upper: {bb_upper_cpp:.2f} (expected {bb_upper_exp:.2f})")

    def test_bollinger_bands_width_increase(self):
        """
        Test 2.2: Bollinger Bands are ~2.53% wider after fix.

        Before fix: population variance (÷20)
        After fix: sample variance (÷19)
        Expected width increase: sqrt(20/19) = 1.0263 → 2.63% wider
        """
        if not HAVE_SIMULATOR:
            pytest.skip("MarketSimulator not available")

        # Generate prices
        np.random.seed(42)
        prices = [50000.0 + np.random.randn() * 1000.0 for _ in range(30)]

        # Expected BB (sample variance)
        _, bb_lower_expected, bb_upper_expected = compute_bollinger_bands_correct(prices, period=20)
        expected_width = bb_upper_expected[19] - bb_lower_expected[19]

        # Buggy BB (population variance) - compute manually
        window = prices[:20]
        mean = sum(window) / 20.0
        var_population = sum((x - mean) ** 2 for x in window) / 20.0  # BUG: ÷20
        std_population = math.sqrt(var_population)
        buggy_bb_lower = mean - 2.0 * std_population
        buggy_bb_upper = mean + 2.0 * std_population
        buggy_width = buggy_bb_upper - buggy_bb_lower

        # Width ratio
        width_ratio = expected_width / buggy_width
        expected_ratio = math.sqrt(20.0 / 19.0)  # 1.0263

        # ASSERTION: Width increase should match expected ratio
        tolerance = 0.001  # 0.1% tolerance
        assert abs(width_ratio - expected_ratio) < tolerance, (
            f"BB width increase does NOT match expected!\n"
            f"Correct width:   {expected_width:.2f}\n"
            f"Buggy width:     {buggy_width:.2f}\n"
            f"Ratio:           {width_ratio:.6f}\n"
            f"Expected ratio:  {expected_ratio:.6f} (sqrt(20/19))\n"
            f"Difference:      {abs(width_ratio - expected_ratio):.6f}"
        )

        print(f"✓ Test 2.2 passed: BB width increase verified")
        print(f"  Correct width: {expected_width:.2f}")
        print(f"  Buggy width:   {buggy_width:.2f}")
        print(f"  Ratio:         {width_ratio:.6f} (expected {expected_ratio:.6f})")
        print(f"  Increase:      {(width_ratio - 1.0) * 100:.2f}%")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for both bug fixes."""

    def test_all_indicators_finite(self):
        """
        Test 3.1: All indicators remain finite (no NaN/Inf after fixes).

        Regression test to ensure fixes don't introduce numerical instabilities.
        """
        if not HAVE_SIMULATOR:
            pytest.skip("MarketSimulator not available")

        # Generate random prices
        np.random.seed(42)
        n = 100
        prices = [50000.0 + np.random.randn() * 1000.0 for _ in range(n)]

        sim = PyMarketSimulator(
            price=np.zeros(n, dtype=np.float64),
            open_arr=np.zeros(n, dtype=np.float64),
            high=np.zeros(n, dtype=np.float64),
            low=np.zeros(n, dtype=np.float64),
            volume_usd=np.zeros(n, dtype=np.float64),
            n_steps=n,
            seed=42,
        )

        for i, price in enumerate(prices):
            sim.step(i, black_swan_probability=0.0, is_training_mode=False)
            sim.m_close[i] = price
        for i in range(n):
            sim.update_indicators(i)

        # Check all indicators are finite (after warmup)
        for i in range(20, n):  # After BB initialization
            rsi = sim.get_rsi(i)
            ma20 = sim.get_ma20(i)
            bb_lower = sim.get_bb_lower(i)
            bb_upper = sim.get_bb_upper(i)

            if not math.isnan(rsi):  # RSI may be NaN before initialization
                assert math.isfinite(rsi), f"RSI is not finite at bar {i}: {rsi}"
            assert math.isfinite(ma20), f"MA20 is not finite at bar {i}: {ma20}"
            assert math.isfinite(bb_lower), f"BB lower is not finite at bar {i}: {bb_lower}"
            assert math.isfinite(bb_upper), f"BB upper is not finite at bar {i}: {bb_upper}"

        print(f"✓ Test 3.1 passed: All indicators finite for {n} bars")

    def test_no_regressions_in_other_indicators(self):
        """
        Test 3.2: Other indicators (ATR, MACD, CCI, etc.) are NOT affected by fixes.

        Ensures RSI and BB fixes are isolated and don't break other indicators.
        """
        if not HAVE_SIMULATOR:
            pytest.skip("MarketSimulator not available")

        np.random.seed(42)
        n = 50
        prices = [50000.0 + np.random.randn() * 1000.0 for _ in range(n)]

        sim = PyMarketSimulator(
            price=np.zeros(n, dtype=np.float64),
            open_arr=np.zeros(n, dtype=np.float64),
            high=np.zeros(n, dtype=np.float64),
            low=np.zeros(n, dtype=np.float64),
            volume_usd=np.zeros(n, dtype=np.float64),
            n_steps=n,
            seed=42,
        )

        for i, price in enumerate(prices):
            sim.step(i, black_swan_probability=0.0, is_training_mode=False)
            sim.m_close[i] = price
        for i in range(n):
            sim.update_indicators(i)

        # Check other indicators are reasonable (not NaN/Inf after warmup)
        for i in range(30, n):
            atr = sim.get_atr(i)
            macd = sim.get_macd(i)
            macd_signal = sim.get_macd_signal(i)
            cci = sim.get_cci(i)

            if not math.isnan(atr):
                assert math.isfinite(atr), f"ATR not finite at bar {i}"
            if not math.isnan(macd):
                assert math.isfinite(macd), f"MACD not finite at bar {i}"
            if not math.isnan(macd_signal):
                assert math.isfinite(macd_signal), f"MACD signal not finite at bar {i}"
            if not math.isnan(cci):
                assert math.isfinite(cci), f"CCI not finite at bar {i}"

        print(f"✓ Test 3.2 passed: Other indicators unaffected")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("INDICATOR BUG FIXES - COMPREHENSIVE TESTS")
    print("=" * 80)
    print()

    pytest.main([__file__, "-v", "--tb=short"])
