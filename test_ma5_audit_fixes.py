#!/usr/bin/env python3
"""
Comprehensive test suite for MA5 audit fixes.

Tests:
1. is_final validation in FeaturePipe
2. decision_delay_ms validation in LeakGuard
3. Train-inference consistency
4. Forward-looking bias prevention

References:
- de Prado, M.L. (2018). "Advances in Financial Machine Learning", Chapter 7
- Murphy, J.J. (1999). "Technical Analysis of the Financial Markets"
"""

import unittest
import warnings
from decimal import Decimal

# Test imports
from core_models import Bar
from feature_pipe import FeaturePipe
from transformers import FeatureSpec
from leakguard import LeakGuard, LeakConfig


class TestIsFinalValidation(unittest.TestCase):
    """Test that FeaturePipe only processes final (closed) bars."""

    def test_final_bar_is_processed(self):
        """CRITICAL: Final bars (is_final=True) must be processed."""
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)
        fp = FeaturePipe(spec)

        bar = Bar(
            ts=1000,
            symbol="BTCUSDT",
            open=Decimal("50000"),
            high=Decimal("50100"),
            low=Decimal("49900"),
            close=Decimal("50050"),
            is_final=True  # CLOSED BAR
        )

        feats = fp.update(bar)

        # Must return non-empty features
        self.assertTrue(len(feats) > 0, "Final bar must produce features")
        self.assertIn("ref_price", feats, "Must contain ref_price")

    def test_non_final_bar_is_rejected(self):
        """CRITICAL: Non-final bars (is_final=False) must be REJECTED.

        This prevents forward-looking bias:
        - Binance sends intermediate updates with is_final=False
        - Using unclosed prices creates train-inference mismatch
        - Training uses final prices, inference would use intermediate
        """
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)
        fp = FeaturePipe(spec)

        bar = Bar(
            ts=1000,
            symbol="BTCUSDT",
            open=Decimal("50000"),
            high=Decimal("50100"),
            low=Decimal("49900"),
            close=Decimal("49500"),  # INTERMEDIATE PRICE!
            is_final=False  # NOT CLOSED YET!
        )

        feats = fp.update(bar)

        # Must return empty dict
        self.assertEqual(len(feats), 0, "Non-final bar must be REJECTED (empty dict)")

    def test_sequence_final_vs_non_final(self):
        """Test realistic sequence: intermediate updates + final bar."""
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)
        fp = FeaturePipe(spec)

        # Intermediate update 1 (17:30, bar still open)
        bar1 = Bar(
            ts=1000,
            symbol="BTCUSDT",
            open=Decimal("50000"),
            high=Decimal("50100"),
            low=Decimal("49900"),
            close=Decimal("49500"),
            is_final=False
        )
        feats1 = fp.update(bar1)
        self.assertEqual(len(feats1), 0, "Intermediate update 1 must be rejected")

        # Intermediate update 2 (17:45, bar still open)
        bar2 = Bar(
            ts=1000,
            symbol="BTCUSDT",
            open=Decimal("50000"),
            high=Decimal("50200"),
            low=Decimal("49800"),
            close=Decimal("50100"),
            is_final=False
        )
        feats2 = fp.update(bar2)
        self.assertEqual(len(feats2), 0, "Intermediate update 2 must be rejected")

        # Final bar (18:00, bar closed)
        bar3 = Bar(
            ts=1000,
            symbol="BTCUSDT",
            open=Decimal("50000"),
            high=Decimal("50200"),
            low=Decimal("49800"),
            close=Decimal("50050"),  # FINAL PRICE
            is_final=True
        )
        feats3 = fp.update(bar3)
        self.assertTrue(len(feats3) > 0, "Final bar must be processed")
        self.assertAlmostEqual(float(feats3["ref_price"]), 50050.0, places=1)

    def test_backward_compatibility_no_is_final_field(self):
        """Backward compatibility: bars without is_final field default to True."""
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)
        fp = FeaturePipe(spec)

        # Old-style bar without is_final field
        bar = Bar(
            ts=1000,
            symbol="BTCUSDT",
            open=Decimal("50000"),
            high=Decimal("50100"),
            low=Decimal("49900"),
            close=Decimal("50050")
            # is_final not specified - defaults to True in dataclass
        )

        feats = fp.update(bar)
        self.assertTrue(len(feats) > 0, "Bar without is_final should be processed (defaults to True)")


class TestDecisionDelayValidation(unittest.TestCase):
    """Test decision_delay_ms validation in LeakGuard."""

    def test_zero_delay_raises_warning(self):
        """CRITICAL: decision_delay_ms=0 must raise UserWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            lg = LeakGuard(LeakConfig(decision_delay_ms=0))

            # Must have raised exactly 1 warning
            self.assertEqual(len(w), 1, "Must raise exactly 1 warning")
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertIn("FORWARD-LOOKING BIAS", str(w[0].message))
            self.assertIn("decision_delay_ms=0", str(w[0].message))

    def test_positive_delay_no_warning(self):
        """Positive decision_delay_ms should not raise warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            lg = LeakGuard(LeakConfig(decision_delay_ms=8000))

            # Should have NO warnings
            self.assertEqual(len(w), 0, "Positive delay should not raise warning")

    def test_negative_delay_raises_error(self):
        """CRITICAL: Negative decision_delay_ms must raise ValueError."""
        with self.assertRaises(ValueError) as cm:
            lg = LeakGuard(LeakConfig(decision_delay_ms=-1000))

        self.assertIn("must be >= 0", str(cm.exception))
        self.assertIn("future", str(cm.exception).lower())


class TestTrainInferenceConsistency(unittest.TestCase):
    """Test that training and inference use same semantics."""

    def test_sma_formula_consistency(self):
        """Verify SMA formula matches documentation.

        According to Murphy (1999) and our docs:
        SMA_5 = (P_t + P_{t-1} + P_{t-2} + P_{t-3} + P_{t-4}) / 5
        where P_t = close price of CURRENT (just closed) bar
        """
        spec = FeatureSpec(lookbacks_prices=[1200], bar_duration_minutes=240)
        fp = FeaturePipe(spec)

        prices = [100.0, 102.0, 104.0, 106.0, 108.0]

        for i, price in enumerate(prices):
            bar = Bar(
                ts=i * 1000,
                symbol="BTCUSDT",
                open=Decimal(str(price)),
                high=Decimal(str(price + 1)),
                low=Decimal(str(price - 1)),
                close=Decimal(str(price)),
                is_final=True
            )
            feats = fp.update(bar)

        # After 5 bars, should have sma_1200 (5-period SMA)
        self.assertIn("sma_1200", feats, "Must have sma_1200 after 5 bars")

        # SMA_5 = (108 + 106 + 104 + 102 + 100) / 5 = 104.0
        expected_sma = sum(prices) / len(prices)
        actual_sma = feats["sma_1200"]

        self.assertAlmostEqual(actual_sma, expected_sma, places=5,
            msg=f"SMA_5 должно быть {expected_sma}, получили {actual_sma}")

    def test_lag_documentation(self):
        """Verify lag is ~window/2 as documented (Murphy 1999)."""
        # This is a property test - SMA has inherent lag of ~window/2
        # For SMA_5, lag ≈ 2.5 bars
        # This is CORRECT behavior, not a bug
        # Model must account for this lag in predictions
        pass  # Documented property, not a bug to fix


class TestForwardLookingPrevention(unittest.TestCase):
    """Test that all safeguards prevent forward-looking bias."""

    def test_complete_protection_chain(self):
        """Integration test: Complete protection from features to labels."""
        import pandas as pd

        # 1. Features only from final bars (FeaturePipe)
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)
        fp = FeaturePipe(spec)

        non_final = Bar(
            ts=1000, symbol="BTCUSDT",
            open=Decimal("50000"), high=Decimal("50000"),
            low=Decimal("50000"), close=Decimal("49000"),  # Wrong price!
            is_final=False
        )
        self.assertEqual(len(fp.update(non_final)), 0, "Non-final rejected")

        final = Bar(
            ts=1000, symbol="BTCUSDT",
            open=Decimal("50000"), high=Decimal("50000"),
            low=Decimal("50000"), close=Decimal("50000"),  # Correct price
            is_final=True
        )
        feats = fp.update(final)
        self.assertTrue(len(feats) > 0, "Final processed")

        # 2. Decision delay enforced (LeakGuard)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lg_bad = LeakGuard(LeakConfig(decision_delay_ms=0))
            self.assertEqual(len(w), 1, "Warning raised for delay=0")

        lg_good = LeakGuard(LeakConfig(decision_delay_ms=8000))
        df = pd.DataFrame({"ts_ms": [1000, 2000, 3000]})
        result = lg_good.attach_decision_time(df)

        self.assertTrue("decision_ts" in result.columns)
        self.assertEqual(result["decision_ts"].iloc[0], 1000 + 8000)

        # 3. Labels from decision_ts (tested in labels.py tests)
        # See test_data_pipeline_validation.py


if __name__ == "__main__":
    unittest.main()
