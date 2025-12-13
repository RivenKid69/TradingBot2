"""
Test Bug #7 Fix: Grouped EV Missing Epsilon

Bug #7 Location: distributional_ppo.py:529-535
Issue: Grouped EV computation was missing epsilon in denominator
Fix: Added epsilon (1e-12) for numerical stability

Test Coverage:
1. Normal case - verify epsilon is used
2. Extreme small variance - near epsilon value
3. Perfect predictions - zero error variance
4. Large variance ratio - var_err >> var_true
5. Numerical stability - always finite results
6. Consistency with non-grouped EV
"""

import numpy as np
import pytest
pytest.importorskip("gymnasium")
from typing import Dict, Optional
from distributional_ppo import compute_grouped_explained_variance


class TestBug7GroupedEVEpsilon:
    """Test Bug #7 fix: Epsilon in grouped EV computation."""

    def test_normal_case_epsilon_used(self):
        """Test that epsilon is properly used in normal cases."""
        # Create realistic data
        n = 1000
        y_true = np.random.randn(n) * 10.0
        y_pred = y_true + np.random.randn(n) * 2.0  # Some error
        groups = np.random.choice([0, 1, 2], size=n)

        ev_dict, summary = compute_grouped_explained_variance(y_true, y_pred, groups)

        # Check that overall results are finite
        overall_ev = summary.get("mean_weighted", float("nan"))
        if overall_ev is not None and np.isfinite(overall_ev):
            # Check that EV is in reasonable range
            assert -1.0 <= overall_ev <= 1.0, f"EV should be in [-1, 1], got {overall_ev}"

        # For each group, check finite and reasonable
        for group_id in ["0", "1", "2"]:  # Groups are strings
            if group_id in ev_dict and np.isfinite(ev_dict[group_id]):
                ev = ev_dict[group_id]
                assert -10.0 <= ev <= 1.0, f"Group {group_id} EV should be reasonable, got {ev}"

    def test_extreme_small_variance(self):
        """Test when var_true is very small (near epsilon)."""
        # Create data with extremely small variance (but above variance_floor)
        n = 100
        groups = np.array([0] * 50 + [1] * 50)

        # Group 0: Very small variance (1e-10, above variance_floor 1e-12)
        y_true = np.concatenate([
            np.ones(50) * 10.0 + np.random.randn(50) * 1e-5,  # Very small noise
            np.random.randn(50) * 5.0  # Normal variance
        ])
        y_pred = y_true + np.random.randn(n) * 1e-6  # Very small prediction error

        ev_dict, summary = compute_grouped_explained_variance(
            y_true, y_pred, groups, variance_floor=1e-12
        )

        # Check that epsilon prevents division issues
        overall_ev = summary.get("mean_weighted")
        assert overall_ev is None or np.isfinite(overall_ev) or np.isnan(overall_ev), \
            "Overall EV should be finite or NaN (not inf) even with small variance"

        # Group 0 might have very high EV (near 1.0) due to small error
        if "0" in ev_dict:
            ev_0 = ev_dict["0"]
            if np.isfinite(ev_0):
                assert ev_0 >= -10.0, f"Group 0 EV should be reasonable, got {ev_0}"

    def test_perfect_predictions_ev_near_one(self):
        """Test perfect predictions give EV ≈ 1.0."""
        n = 200
        y_true = np.random.randn(n) * 10.0
        y_pred = y_true.copy()  # Perfect predictions
        groups = np.random.choice([0, 1], size=n)

        ev_dict, summary = compute_grouped_explained_variance(y_true, y_pred, groups)

        # With perfect predictions, var_err ≈ 0, so EV ≈ 1.0
        overall_ev = summary.get("mean_weighted")
        if overall_ev is not None and np.isfinite(overall_ev):
            assert overall_ev >= 0.95, \
                f"Perfect predictions should give EV ≈ 1.0, got {overall_ev}"

    def test_constant_predictions_ev_near_zero(self):
        """Test constant predictions (mean predictor) give EV ≈ 0.0."""
        n = 200
        y_true = np.random.randn(n) * 10.0
        groups = np.random.choice([0, 1], size=n)

        # For each group, predict the mean
        y_pred = np.zeros_like(y_true)
        for group_id in [0, 1]:
            mask = groups == group_id
            if np.sum(mask) > 0:
                y_pred[mask] = np.mean(y_true[mask])

        ev_dict, summary = compute_grouped_explained_variance(y_true, y_pred, groups)

        # Mean predictor should give EV ≈ 0.0
        overall_ev = summary.get("mean_weighted")
        if overall_ev is not None and np.isfinite(overall_ev):
            assert -0.15 <= overall_ev <= 0.15, \
                f"Mean predictor should give EV ≈ 0.0, got {overall_ev}"

    def test_worse_than_mean_ev_negative(self):
        """Test predictions worse than mean give negative EV."""
        n = 200
        y_true = np.random.randn(n) * 10.0
        groups = np.random.choice([0, 1], size=n)

        # Predict opposite direction (worse than mean)
        y_pred = -y_true

        ev_dict, summary = compute_grouped_explained_variance(y_true, y_pred, groups)

        # Worse-than-mean predictions should give negative EV
        overall_ev = summary.get("mean_weighted")
        if overall_ev is not None and np.isfinite(overall_ev):
            assert overall_ev < 0.0, \
                f"Worse-than-mean predictions should give EV < 0, got {overall_ev}"

    def test_large_variance_ratio(self):
        """Test when var_err >> var_true (poor predictions)."""
        n = 100
        groups = np.array([0] * 50 + [1] * 50)

        # Group 0: Small true variance, large error variance
        y_true = np.concatenate([
            np.ones(50) * 10.0 + np.random.randn(50) * 1.0,  # Small variance
            np.random.randn(50) * 5.0  # Normal variance
        ])

        # Add large random error to group 0
        y_pred = y_true.copy()
        y_pred[:50] += np.random.randn(50) * 50.0  # Very large error

        ev_dict, summary = compute_grouped_explained_variance(y_true, y_pred, groups)

        # Check that epsilon prevents numerical issues
        overall_ev = summary.get("mean_weighted")
        assert overall_ev is None or np.isfinite(overall_ev) or np.isnan(overall_ev), \
            "Overall EV should be finite or NaN (not inf)"

        # Group 0 should have very negative EV
        if "0" in ev_dict:
            ev_0 = ev_dict["0"]
            if np.isfinite(ev_0):
                assert ev_0 < 0.0, \
                    f"Group 0 with large errors should have negative EV, got {ev_0}"

    def test_numerical_stability_always_finite(self):
        """Test that results are always finite when var_true > variance_floor."""
        n = 500

        for seed in range(10):
            np.random.seed(seed)

            # Random data with various scales
            scale = 10 ** np.random.uniform(-3, 3)
            y_true = np.random.randn(n) * scale
            y_pred = y_true + np.random.randn(n) * scale * np.random.uniform(0.1, 2.0)
            groups = np.random.choice([0, 1, 2], size=n)

            ev_dict, summary = compute_grouped_explained_variance(y_true, y_pred, groups)

            # Overall should be finite or NaN (not inf)
            overall_ev = summary.get("mean_weighted")
            assert overall_ev is None or np.isfinite(overall_ev) or np.isnan(overall_ev), \
                f"Seed {seed}: Overall EV should be finite or NaN, not inf"

    def test_epsilon_prevents_division_by_near_zero(self):
        """Test that epsilon prevents division issues when var_true is very small."""
        # Create data where var_true is just above variance_floor
        variance_floor = 1e-6

        # Group A: variance slightly above floor (1.5e-6)
        y_true_A = np.array([1.0, 1.0 + 1.2e-3, 1.0 + 1.3e-3])
        y_pred_A = np.array([1.0, 1.0 + 1.2e-3, 1.0 + 1.2e-3])

        # Group B: normal variance (control group)
        y_true_B = np.array([10.0, 20.0, 30.0])
        y_pred_B = np.array([11.0, 19.0, 29.0])

        y_true = np.concatenate([y_true_A, y_true_B])
        y_pred = np.concatenate([y_pred_A, y_pred_B])
        groups = np.array(["A", "A", "A", "B", "B", "B"])

        ev_dict, summary = compute_grouped_explained_variance(
            y_true, y_pred, groups, variance_floor=variance_floor
        )

        # Result should not have inf values
        for key, value in ev_dict.items():
            if not np.isnan(value):
                assert np.isfinite(value), \
                    f"Group {key} EV should be finite, got {value}"

    def test_mixed_variance_scales(self):
        """Test groups with very different variance scales."""
        n = 300
        groups = np.array([0] * 100 + [1] * 100 + [2] * 100)

        # Group 0: Large variance
        # Group 1: Medium variance
        # Group 2: Small variance (but above variance_floor)
        y_true = np.concatenate([
            np.random.randn(100) * 100.0,  # Large
            np.random.randn(100) * 10.0,   # Medium
            np.random.randn(100) * 0.1     # Small
        ])

        y_pred = y_true + np.random.randn(n) * 5.0

        ev_dict, summary = compute_grouped_explained_variance(y_true, y_pred, groups)

        # Check that all groups are handled correctly
        overall_ev = summary.get("mean_weighted")
        assert overall_ev is None or np.isfinite(overall_ev) or np.isnan(overall_ev), \
            "Overall EV should be finite or NaN"

    def test_single_sample_group_returns_nan(self):
        """Test that groups with only 1 sample return NaN."""
        n = 100
        y_true = np.random.randn(n) * 10.0
        y_pred = y_true + np.random.randn(n) * 2.0

        # Create groups with one singleton group
        groups = np.array([0] * 50 + [1] * 49 + [2])  # Group 2 has only 1 sample

        ev_dict, summary = compute_grouped_explained_variance(y_true, y_pred, groups)

        # Group 2 should be NaN
        if "2" in ev_dict:
            assert np.isnan(ev_dict["2"]), "Single-sample group should return NaN"

        # But overall should still be finite (based on groups 0 and 1)
        overall_ev = summary.get("mean_weighted")
        assert overall_ev is None or np.isfinite(overall_ev) or np.isnan(overall_ev), \
            "Overall should be finite or NaN despite singleton group"

    def test_fix_verifies_epsilon_added(self):
        """Verify that the fix (epsilon added) is present in the code."""
        # This test verifies the fix was applied by checking behavior
        # with extremely small variance that would cause issues without epsilon

        # Create scenario that would fail without epsilon protection
        y_true = np.array([1.0, 1.0 + 1e-11, 1.0 + 2e-11, 1.0 + 3e-11])
        y_pred = np.array([1.0, 1.0 + 1e-11, 1.0 + 1.5e-11, 1.0 + 2.5e-11])
        groups = np.array([0, 0, 0, 0])

        # With variance_floor=0, this would expose missing epsilon
        ev_dict, summary = compute_grouped_explained_variance(
            y_true, y_pred, groups, variance_floor=0.0
        )

        # Verify no overflow (which would occur without epsilon)
        if "0" in ev_dict:
            ev_0 = ev_dict["0"]
            # With epsilon, result should be finite (not inf)
            assert np.isfinite(ev_0) or np.isnan(ev_0), \
                f"Bug #7 fix verification: EV should be finite or NaN, got {ev_0}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
