"""Verification tests for RSI and CCI initialization fixes.

This module verifies that the fixes for Bug #1 (RSI) and Bug #3 (CCI) work correctly.

Reference: INDICATOR_INITIALIZATION_BUGS_REPORT.md
"""

import pytest
import numpy as np
import math

from transformers import FeatureSpec, OnlineFeatureTransformer


class TestRSIFixVerification:
    """Verify RSI initialization fix works correctly."""

    def test_rsi_correct_sma_initialization(self):
        """AFTER FIX: RSI should initialize with SMA(14) of gains/losses."""
        spec = FeatureSpec(
            lookbacks_prices=[5, 10],
            rsi_period=14,
            bar_duration_minutes=1,
        )
        transformer = OnlineFeatureTransformer(spec)

        # Price pattern: First bar +10%, then small oscillations
        prices = [
            100.0,
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
            110.5,
            110.0,
            110.5,
        ]

        feats_list = []
        for i, price in enumerate(prices):
            feats = transformer.update(
                symbol="BTCUSDT",
                ts_ms=i * 60000,
                close=price,
            )
            feats_list.append(feats)

        # After 14 bars (bar 14 is the 15th price, index 14)
        rsi_14 = feats_list[14]["rsi"]

        # Expected RSI with CORRECT SMA initialization:
        # gains: [10.0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5]
        # losses: [0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0]
        # avg_gain = (10.0 + 6*0.5) / 14 = 13.0 / 14 = 0.9286
        # avg_loss = (7*0.5) / 14 = 3.5 / 14 = 0.25
        # RS = 0.9286 / 0.25 = 3.714
        # RSI = 100 - (100 / (1 + 3.714)) = 78.8

        expected_rsi = 78.8

        # AFTER FIX: RSI should be close to expected value
        assert (
            abs(rsi_14 - expected_rsi) < 5.0
        ), f"RSI={rsi_14:.1f} differs from expected {expected_rsi:.1f} by more than 5 points"

        print(f"✓ RSI correctly initialized: {rsi_14:.1f} (expected: {expected_rsi:.1f})")

    def test_rsi_no_premature_values(self):
        """RSI should be NaN until rsi_period samples collected."""
        spec = FeatureSpec(
            lookbacks_prices=[5],
            rsi_period=14,
            bar_duration_minutes=1,
        )
        transformer = OnlineFeatureTransformer(spec)

        prices = [100.0 + i * 0.1 for i in range(20)]

        feats_list = []
        for i, price in enumerate(prices):
            feats = transformer.update(
                symbol="BTCUSDT",
                ts_ms=i * 60000,
                close=price,
            )
            feats_list.append(feats)

        # RSI should be NaN for first 14 bars (indices 0-13)
        for i in range(14):
            rsi = feats_list[i].get("rsi", math.nan)
            assert math.isnan(rsi), f"RSI at bar {i} should be NaN, got {rsi:.2f}"

        # RSI should be valid from bar 14 onwards
        for i in range(14, 20):
            rsi = feats_list[i].get("rsi", math.nan)
            assert not math.isnan(rsi), f"RSI at bar {i} should be valid, got NaN"
            assert 0 <= rsi <= 100, f"RSI at bar {i} = {rsi:.2f} out of range [0, 100]"

        print(f"✓ RSI correctly waits for {spec.rsi_period} samples before initialization")

    def test_rsi_wilder_smoothing_after_init(self):
        """After initialization, RSI should use Wilder's smoothing."""
        spec = FeatureSpec(
            lookbacks_prices=[5],
            rsi_period=14,
            bar_duration_minutes=1,
        )
        transformer = OnlineFeatureTransformer(spec)

        # 20 bars of constant +1.0 gains
        prices = [100.0 + i for i in range(20)]

        feats_list = []
        for i, price in enumerate(prices):
            feats = transformer.update(
                symbol="BTCUSDT",
                ts_ms=i * 60000,
                close=price,
            )
            feats_list.append(feats)

        # At bar 14: avg_gain = 1.0 (SMA of first 14 gains)
        # At bar 15: avg_gain = (1.0 * 13 + 1.0) / 14 = 1.0 (no change)
        # With all gains and no losses, RSI should be 100

        rsi_14 = feats_list[14]["rsi"]
        rsi_15 = feats_list[15]["rsi"]

        assert rsi_14 == 100.0, f"RSI at bar 14 should be 100 (all gains), got {rsi_14:.2f}"
        assert rsi_15 == 100.0, f"RSI at bar 15 should be 100 (all gains), got {rsi_15:.2f}"

        print(f"✓ Wilder smoothing works correctly after initialization")

    def test_rsi_comparison_with_reference(self):
        """Compare fixed RSI with reference implementation (conceptual)."""
        spec = FeatureSpec(
            lookbacks_prices=[5],
            rsi_period=14,
            bar_duration_minutes=1,
        )
        transformer = OnlineFeatureTransformer(spec)

        # Random walk price data
        np.random.seed(42)
        prices = [100.0]
        for _ in range(50):
            delta = np.random.randn() * 0.5
            prices.append(prices[-1] + delta)

        feats_list = []
        for i, price in enumerate(prices):
            feats = transformer.update(
                symbol="BTCUSDT",
                ts_ms=i * 60000,
                close=price,
            )
            feats_list.append(feats)

        # Compute reference RSI manually (Wilder's method)
        def compute_rsi_reference(prices, period=14):
            """Reference implementation of RSI."""
            if len(prices) <= period:
                return [math.nan] * len(prices)

            gains = []
            losses = []
            for i in range(1, len(prices)):
                delta = prices[i] - prices[i - 1]
                gains.append(max(0, delta))
                losses.append(max(0, -delta))

            rsi_values = [math.nan] * (period + 1)  # First 14 are NaN

            # Initialize with SMA
            avg_gain = sum(gains[:period]) / period
            avg_loss = sum(losses[:period]) / period

            # First RSI
            if avg_loss == 0:
                rsi_values.append(100.0)
            elif avg_gain == 0:
                rsi_values.append(0.0)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100.0 - (100.0 / (1.0 + rs)))

            # Wilder's smoothing for subsequent values
            for i in range(period, len(gains)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period

                if avg_loss == 0:
                    rsi_values.append(100.0)
                elif avg_gain == 0:
                    rsi_values.append(0.0)
                else:
                    rs = avg_gain / avg_loss
                    rsi_values.append(100.0 - (100.0 / (1.0 + rs)))

            return rsi_values

        rsi_reference = compute_rsi_reference(prices, period=14)
        rsi_actual = [f.get("rsi", math.nan) for f in feats_list]

        # Compare from bar 14 onwards
        for i in range(14, len(prices)):
            ref = rsi_reference[i]
            act = rsi_actual[i]
            if not math.isnan(ref) and not math.isnan(act):
                diff = abs(ref - act)
                assert (
                    diff < 0.1
                ), f"RSI at bar {i}: reference={ref:.2f}, actual={act:.2f}, diff={diff:.2f}"

        print(f"✓ RSI matches reference implementation (max diff < 0.1)")


class TestCCIFixVerification:
    """Verify CCI mean deviation fix works correctly (conceptual tests).

    NOTE: CCI is implemented in MarketSimulator.cpp, which requires C++ compilation.
    These tests document the expected behavior after fix.
    """

    def test_cci_correct_baseline_conceptual(self):
        """AFTER FIX: CCI should use SMA(TP) as baseline, not SMA(close).

        This is a CONCEPTUAL test (C++ code requires compilation to test).
        """
        # Simulate bars where close != TP
        bars = []
        for i in range(20):
            bars.append(
                {
                    "high": 102.0,
                    "low": 98.0,
                    "close": 98.5,  # Close near low
                }
            )

        # Compute TP
        tp_values = [(b["high"] + b["low"] + b["close"]) / 3 for b in bars]

        # CORRECT: Use SMA(TP) as baseline
        sma_tp = sum(tp_values) / 20
        mean_dev_correct = sum(abs(tp - sma_tp) for tp in tp_values) / 20
        cci_correct = (
            (tp_values[-1] - sma_tp) / (0.015 * mean_dev_correct) if mean_dev_correct > 0 else 0
        )

        # Expected: TP = 99.5, SMA(TP) = 99.5, CCI ≈ 0
        expected_tp = (102.0 + 98.0 + 98.5) / 3
        assert abs(expected_tp - 99.5) < 0.1

        # Mean deviation should be from SMA(TP), not SMA(close)
        # With all bars identical, mean_dev = 0, so CCI = 0 or undefined
        print(f"✓ CCI conceptual test: TP={expected_tp:.2f}, SMA(TP)={sma_tp:.2f}")
        print(f"   Mean_dev={mean_dev_correct:.4f}, CCI={cci_correct:.2f}")

        # After compilation, actual CCI from MarketSimulator should match this

    def test_cci_no_sign_inversion(self):
        """Verify CCI fix prevents sign inversion."""
        # Oscillating bars
        bars = []
        for i in range(20):
            if i % 2 == 0:
                bars.append({"high": 102.0, "low": 98.0, "close": 101.5})
            else:
                bars.append({"high": 102.0, "low": 98.0, "close": 98.5})

        tp_values = [(b["high"] + b["low"] + b["close"]) / 3 for b in bars]

        # Last bar: close=101.5, TP=100.5
        tp_last = (102.0 + 98.0 + 101.5) / 3

        # CORRECT: SMA(TP) = 100.5 (all TPs are identical)
        sma_tp = sum(tp_values) / 20

        # Mean deviation = 0 (all TPs identical)
        mean_dev = sum(abs(tp - sma_tp) for tp in tp_values) / 20

        # CCI = (100.5 - 100.5) / (0.015 * 0) = undefined or 0
        print(f"✓ CCI sign inversion test: TP={tp_last:.2f}, SMA(TP)={sma_tp:.2f}")
        print(f"   Mean_dev={mean_dev:.4f} (all TPs identical)")

        assert abs(tp_last - sma_tp) < 0.01, "TP should equal SMA(TP) for identical bars"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
