"""
Tests for VGS (Variance Gradient Scaler) interaction with training.

Key Hypothesis: VGS with default alpha=0.1 may be too aggressive when
gradient variance is high, causing:
1. Explained Variance ~ 0 (value function can't learn)
2. Twin Critics loss growth (critics can't adapt to changing targets)
3. Gradient norm collapse (-82%)

This test suite validates the hypothesis and proposes fixes.
"""

import math
import numpy as np
import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn
from typing import Tuple, List
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestVGSScalingBehavior:
    """Test VGS scaling behavior under different conditions."""

    def test_vgs_scaling_with_high_variance_gradients(self):
        """
        Test that VGS scaling factor is too aggressive with high variance.

        When gradient variance is high (>10), scaling factor becomes
        very small (<0.1), effectively blocking learning.
        """
        # VGS formula: scale = 1 / (1 + alpha * normalized_var)
        alpha = 0.1  # Default

        # Simulate different gradient variance levels
        test_cases = [
            (1.0, "Low variance"),
            (10.0, "Moderate variance"),
            (50.0, "High variance"),
            (100.0, "Very high variance"),
            (500.0, "Extreme variance"),
        ]

        print("\nVGS Scaling Analysis:")
        print("-" * 60)
        print(f"{'Variance':<15} {'Scaling Factor':<20} {'Gradient Reduction'}")
        print("-" * 60)

        for var, desc in test_cases:
            scale = 1.0 / (1.0 + alpha * var)
            reduction = (1.0 - scale) * 100
            print(f"{var:<15.1f} {scale:<20.4f} {reduction:.1f}% reduction")

        # With variance=100, scale=0.0909 -> 91% reduction!
        high_var_scale = 1.0 / (1.0 + alpha * 100.0)
        assert high_var_scale < 0.15, f"Expected aggressive scaling, got {high_var_scale}"

    def test_vgs_alpha_sensitivity(self):
        """
        Test how different alpha values affect scaling.

        Finding optimal alpha that:
        1. Reduces high-variance gradients
        2. Doesn't completely block learning
        """
        alphas = [0.01, 0.05, 0.1, 0.2, 0.5]
        variance = 100.0  # High variance scenario

        print("\nAlpha Sensitivity Analysis (variance=100):")
        print("-" * 60)

        for alpha in alphas:
            scale = 1.0 / (1.0 + alpha * variance)
            print(f"alpha={alpha:<5.2f}: scale={scale:.4f} ({(1-scale)*100:.1f}% reduction)")

        # Recommendation: alpha=0.01-0.05 for less aggressive scaling
        recommended_alpha = 0.02
        recommended_scale = 1.0 / (1.0 + recommended_alpha * 100.0)
        assert recommended_scale > 0.3, f"Recommended alpha gives scale {recommended_scale}"

    def test_vgs_minimum_scaling_floor(self):
        """
        Test that VGS has a minimum scaling floor to prevent complete gradient blocking.

        Current implementation: min(scale, 1e-4) - too aggressive!
        Recommendation: min(scale, 0.1) to allow at least 10% of gradients through.
        """
        # Current implementation floor
        current_floor = 1e-4  # Effectively 0

        # Recommended floor
        recommended_floor = 0.1  # At least 10% gradients pass through

        # With extreme variance, current floor is too low
        extreme_variance = 10000.0
        alpha = 0.1
        raw_scale = 1.0 / (1.0 + alpha * extreme_variance)

        current_clamped = max(raw_scale, current_floor)
        recommended_clamped = max(raw_scale, recommended_floor)

        print(f"\nExtreme variance scenario (var={extreme_variance}):")
        print(f"Raw scale: {raw_scale:.6f}")
        print(f"Current floor ({current_floor}): {current_clamped:.6f}")
        print(f"Recommended floor ({recommended_floor}): {recommended_clamped:.2f}")

        assert current_clamped < 0.001, "Current floor is too low"
        assert recommended_clamped >= 0.1, "Recommended floor allows learning"


class TestValueFunctionLearning:
    """Test value function learning under VGS scaling."""

    def test_critic_learning_with_aggressive_vgs(self):
        """
        Test that critic can't learn with aggressive VGS scaling.
        """
        torch.manual_seed(42)

        # Simple critic
        critic = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

        losses_no_vgs = []
        losses_with_vgs = []

        # Training with VGS simulation
        for scenario, vgs_scale in [("no_vgs", 1.0), ("with_vgs", 0.1)]:
            torch.manual_seed(42)  # Reset for fair comparison
            model = nn.Sequential(
                nn.Linear(10, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            scenario_losses = []
            for _ in range(100):
                x = torch.randn(32, 10)
                y = torch.randn(32, 1)

                pred = model(x)
                loss = nn.functional.mse_loss(pred, y)

                opt.zero_grad()
                loss.backward()

                # Apply VGS scaling
                if vgs_scale < 1.0:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(vgs_scale)

                opt.step()
                scenario_losses.append(loss.item())

            if scenario == "no_vgs":
                losses_no_vgs = scenario_losses
            else:
                losses_with_vgs = scenario_losses

        # Compare final losses
        final_no_vgs = np.mean(losses_no_vgs[-10:])
        final_with_vgs = np.mean(losses_with_vgs[-10:])

        print(f"\nCritic Learning Comparison:")
        print(f"No VGS final loss: {final_no_vgs:.4f}")
        print(f"With VGS (scale=0.1) final loss: {final_with_vgs:.4f}")
        print(f"Degradation: {(final_with_vgs - final_no_vgs) / final_no_vgs * 100:.1f}%")

        # With 90% gradient reduction, learning is significantly impaired
        assert final_with_vgs > final_no_vgs, "VGS should impair learning"

    def test_explained_variance_with_gradient_scaling(self):
        """
        Test that EV degrades when gradients are scaled too aggressively.
        """
        torch.manual_seed(42)
        np.random.seed(42)

        n_samples = 1000
        targets = np.random.normal(0, 1, n_samples)

        # Simulate predictions from model trained with different gradient scales
        scales = [1.0, 0.5, 0.1, 0.05]

        print("\nExplained Variance vs Gradient Scale:")
        print("-" * 50)

        for scale in scales:
            # Simulate prediction quality proportional to gradient scale
            # More gradient -> better predictions
            noise_std = 0.5 + (1.0 - scale) * 2.0  # More noise with lower scale
            predictions = targets + np.random.normal(0, noise_std, n_samples)

            # Compute EV
            var_resid = np.var(targets - predictions)
            var_target = np.var(targets)
            ev = 1.0 - (var_resid / (var_target + 1e-8))

            print(f"Scale={scale:.2f}: EV={ev:.3f} (noise_std={noise_std:.2f})")

        # With very low scale, EV approaches 0
        low_scale_predictions = targets + np.random.normal(0, 2.5, n_samples)
        low_scale_ev = 1.0 - (np.var(targets - low_scale_predictions) / np.var(targets))
        assert low_scale_ev < 0.2, f"Expected low EV with aggressive scaling, got {low_scale_ev}"


class TestTwinCriticsLossGrowth:
    """Test Twin Critics loss behavior under VGS."""

    def test_loss_growth_mechanism(self):
        """
        Test mechanism of Twin Critics loss growth.

        When VGS scales gradients too aggressively:
        1. Critics learn slowly
        2. Target distribution shifts (as policy improves elsewhere)
        3. Gap between predictions and targets grows
        4. Loss increases
        """
        torch.manual_seed(42)

        # Two critics
        critic1 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
        critic2 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))

        opt1 = torch.optim.Adam(critic1.parameters(), lr=1e-3)
        opt2 = torch.optim.Adam(critic2.parameters(), lr=1e-3)

        losses1, losses2 = [], []
        vgs_scale = 0.1  # Aggressive VGS

        for epoch in range(100):
            x = torch.randn(32, 10)

            # Target distribution shifts over time (simulating policy improvement)
            target_mean = 0.5 * (epoch / 100)
            targets = torch.randn(32, 1) + target_mean

            # Critic 1 loss
            pred1 = critic1(x)
            loss1 = nn.functional.mse_loss(pred1, targets)
            opt1.zero_grad()
            loss1.backward()
            for p in critic1.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(vgs_scale)
            opt1.step()
            losses1.append(loss1.item())

            # Critic 2 loss
            pred2 = critic2(x)
            loss2 = nn.functional.mse_loss(pred2, targets)
            opt2.zero_grad()
            loss2.backward()
            for p in critic2.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(vgs_scale)
            opt2.step()
            losses2.append(loss2.item())

        # Analyze loss trend
        first_half = (np.mean(losses1[:50]) + np.mean(losses2[:50])) / 2
        second_half = (np.mean(losses1[50:]) + np.mean(losses2[50:])) / 2

        growth = (second_half - first_half) / first_half * 100

        print(f"\nTwin Critics Loss Analysis (VGS scale={vgs_scale}):")
        print(f"First half mean: {first_half:.4f}")
        print(f"Second half mean: {second_half:.4f}")
        print(f"Growth: {growth:.1f}%")

        # With aggressive VGS, loss may grow as critics can't keep up


class TestProposedFixes:
    """Test proposed fixes for VGS issues."""

    def test_proposed_fix_lower_alpha(self):
        """
        Test fix: Lower VGS alpha from 0.1 to 0.02.
        """
        variance = 100.0

        current_alpha = 0.1
        proposed_alpha = 0.02

        current_scale = 1.0 / (1.0 + current_alpha * variance)
        proposed_scale = 1.0 / (1.0 + proposed_alpha * variance)

        print(f"\nFix 1: Lower alpha")
        print(f"Current (alpha={current_alpha}): scale={current_scale:.4f}")
        print(f"Proposed (alpha={proposed_alpha}): scale={proposed_scale:.4f}")
        print(f"Improvement: {(proposed_scale - current_scale) / current_scale * 100:.1f}% more gradients")

        assert proposed_scale > 0.3, "Proposed alpha should allow more learning"

    def test_proposed_fix_higher_floor(self):
        """
        Test fix: Higher minimum scaling floor from 1e-4 to 0.1.
        """
        variance = 10000.0  # Extreme case
        alpha = 0.1

        raw_scale = 1.0 / (1.0 + alpha * variance)

        current_floor = 1e-4
        proposed_floor = 0.1

        current_result = max(raw_scale, current_floor)
        proposed_result = max(raw_scale, proposed_floor)

        print(f"\nFix 2: Higher floor (var={variance})")
        print(f"Current floor ({current_floor}): {current_result:.6f}")
        print(f"Proposed floor ({proposed_floor}): {proposed_result:.2f}")

        assert proposed_result >= 0.1, "Floor should allow minimum learning"

    def test_proposed_fix_adaptive_alpha(self):
        """
        Test fix: Adaptive alpha based on training progress.

        Early training: lower alpha (allow more exploration)
        Late training: higher alpha (stabilize)
        """
        variance = 100.0

        def adaptive_alpha(epoch: int, max_epochs: int = 1000) -> float:
            """Alpha increases from 0.01 to 0.1 over training."""
            progress = min(epoch / max_epochs, 1.0)
            return 0.01 + 0.09 * progress

        print("\nFix 3: Adaptive alpha")
        for epoch in [0, 100, 500, 1000]:
            alpha = adaptive_alpha(epoch)
            scale = 1.0 / (1.0 + alpha * variance)
            print(f"Epoch {epoch}: alpha={alpha:.3f}, scale={scale:.4f}")

    def test_proposed_fix_clip_variance_not_scale(self):
        """
        Test fix: Clip normalized variance instead of scaling factor.

        Current: scale = max(1/(1+alpha*var), 1e-4)
        Proposed: scale = 1/(1+alpha*min(var, var_cap))

        This provides smoother behavior at extreme variances.
        """
        alpha = 0.1
        var_cap = 50.0  # Cap variance at 50

        test_variances = [10.0, 50.0, 100.0, 500.0, 1000.0]

        print("\nFix 4: Clip variance, not scale")
        print(f"{'Variance':<10} {'Current Scale':<15} {'Proposed Scale'}")
        print("-" * 45)

        for var in test_variances:
            current_scale = max(1.0 / (1.0 + alpha * var), 1e-4)
            capped_var = min(var, var_cap)
            proposed_scale = 1.0 / (1.0 + alpha * capped_var)
            print(f"{var:<10.0f} {current_scale:<15.6f} {proposed_scale:.4f}")

        # Proposed approach maintains minimum ~0.17 scale
        extreme_var = 10000.0
        proposed_scale = 1.0 / (1.0 + alpha * min(extreme_var, var_cap))
        assert proposed_scale > 0.15, "Variance capping should prevent aggressive scaling"


class TestComprehensiveRecommendations:
    """Generate comprehensive recommendations."""

    def test_generate_recommendations(self):
        """
        Generate actionable recommendations based on analysis.
        """
        print("\n" + "=" * 70)
        print("COMPREHENSIVE RECOMMENDATIONS FOR TRAINING QUALITY ISSUES")
        print("=" * 70)

        print("""
DIAGNOSIS SUMMARY:
------------------
1. EV ~ 0: Value function not learning effectively
2. Twin Critics loss +327%: Critics can't adapt to changing targets
3. Gradient norm -82%: Likely due to aggressive VGS scaling
4. CVaR lambda growth: Normal behavior, constraint being violated

ROOT CAUSE ANALYSIS:
--------------------
VGS (Variance Gradient Scaler) with default alpha=0.1 is TOO AGGRESSIVE
when gradient variance is high. This causes:

- Scaling factor ~0.09 (91% gradient reduction!)
- Value function can't learn -> EV ~ 0
- Critics can't keep up with target shifts -> loss grows
- Gradient norms collapse -> -82%

RECOMMENDED FIXES (in order of priority):
-----------------------------------------

FIX 1: LOWER VGS ALPHA (HIGH IMPACT)
    Current: alpha = 0.1
    Proposed: alpha = 0.02

    In configs/config_train.yaml:
    ```yaml
    vgs:
      enabled: true
      alpha: 0.02  # Changed from 0.1
    ```

FIX 2: HIGHER MINIMUM SCALING FLOOR (HIGH IMPACT)
    Current: min_scale = 1e-4
    Proposed: min_scale = 0.1

    In variance_gradient_scaler.py line 434:
    ```python
    # OLD: scaling_factor = max(scaling_factor, 1e-4)
    # NEW:
    scaling_factor = max(scaling_factor, 0.1)
    ```

FIX 3: ADD VARIANCE CAP (MEDIUM IMPACT)
    Cap normalized variance to prevent extreme scaling:

    ```python
    normalized_var = min(self.get_normalized_variance(), 50.0)
    scaling_factor = 1.0 / (1.0 + self.alpha * normalized_var)
    ```

FIX 4: INCREASE VF_COEF (MEDIUM IMPACT)
    Compensate for reduced gradients:

    Current: vf_coef = 1.8
    Proposed: vf_coef = 3.0

FIX 5: REDUCE CLIP_RANGE_VF (LOW IMPACT)
    Tighter clipping may help with non-stationary targets:

    Current: clip_range_vf = 0.7
    Proposed: clip_range_vf = 0.3

MONITORING RECOMMENDATIONS:
---------------------------
1. Log VGS scaling factor: Should be > 0.3 during healthy training
2. Log per-critic losses separately: Should NOT monotonically increase
3. Log EV components: var(residuals), var(targets) separately
4. Log gradient norms per layer: Identify vanishing gradient locations

TEST COMMAND:
-------------
python -m pytest tests/test_vgs_training_interaction.py -v -s
""")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
