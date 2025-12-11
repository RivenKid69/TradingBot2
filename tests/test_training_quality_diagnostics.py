"""
Diagnostic tests for training quality issues:
1. Explained Variance ≈ 0 (Value function not learning)
2. Twin Critics loss growth (+327%)
3. CVaR Lambda growth (constraint violation)
4. Gradient norm collapse (-82%)

These tests validate hypotheses about root causes.
"""

import math
import numpy as np
import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn
from typing import Tuple, Optional
from unittest.mock import MagicMock, patch


# ============================================================================
# HYPOTHESIS 1: Target Distribution Outside Support Bounds
# If targets frequently fall outside [v_min, v_max], C51 loses information
# ============================================================================

class TestTargetDistributionBounds:
    """Test if target clamping causes information loss."""

    def test_target_clamping_frequency_simulation(self):
        """
        Simulate typical signal_only rewards and check clamping frequency.

        In signal_only mode, rewards are log(price_ratio) * position.
        Typical values: log(1.001) ≈ 0.001, log(1.01) ≈ 0.01

        With gamma=0.99 and episode length ~1000:
        Returns can accumulate to sum(gamma^i * r_i) ≈ r * (1-gamma^1000)/(1-gamma) ≈ 100 * r
        So max returns ≈ 100 * 0.01 = 1.0 in normalized space

        But with small rewards, returns stay small.
        """
        # Typical signal_only reward distribution
        np.random.seed(42)
        n_samples = 10000

        # Simulate log returns: mostly small, occasional larger
        base_rewards = np.random.normal(0.0, 0.002, n_samples)  # ~0.2% typical

        # Simulate GAE accumulation with gamma=0.99, lambda=0.95
        gamma = 0.99
        gae_lambda = 0.95

        # Simple GAE simulation
        returns = np.zeros(n_samples)
        last_gae = 0.0
        for i in reversed(range(n_samples)):
            delta = base_rewards[i]  # Simplified: no value bootstrap
            last_gae = delta + gamma * gae_lambda * last_gae
            returns[i] = last_gae

        # Normalize returns
        ret_mean = returns.mean()
        ret_std = max(returns.std(), 1e-6)
        returns_norm = (returns - ret_mean) / ret_std

        # Check what fraction falls outside typical v_min/v_max
        v_min, v_max = -10.0, 10.0
        outside_frac = np.mean((returns_norm < v_min) | (returns_norm > v_max))

        # HYPOTHESIS: If >5% outside bounds, this could cause loss of information
        assert outside_frac < 0.05, (
            f"Too many targets outside bounds: {outside_frac:.2%}. "
            f"Returns range: [{returns_norm.min():.2f}, {returns_norm.max():.2f}]"
        )

    def test_small_return_variance_ev_impact(self):
        """
        Test that small return variance leads to EV ≈ 0.

        When var(returns) is very small, even small prediction errors
        lead to var(residuals) ≈ var(returns), giving EV ≈ 0.
        """
        np.random.seed(42)

        # Simulate small variance returns (typical for signal_only)
        n_samples = 1000
        true_returns = np.random.normal(0.0, 0.01, n_samples)  # Very small variance

        # Predictions with small error
        prediction_noise = np.random.normal(0.0, 0.005, n_samples)
        predictions = true_returns + prediction_noise

        # Compute EV
        var_returns = np.var(true_returns)
        var_residuals = np.var(true_returns - predictions)
        ev = 1.0 - (var_residuals / (var_returns + 1e-8))

        # With var_returns = 0.01^2 = 0.0001 and var_residuals ≈ 0.005^2 = 0.000025
        # EV = 1 - 0.25 = 0.75 (should be good!)
        # But if prediction_noise >> true_returns variance, EV → 0

        # Simulate worse predictions
        bad_predictions = true_returns + np.random.normal(0.0, 0.01, n_samples)
        var_residuals_bad = np.var(true_returns - bad_predictions)
        ev_bad = 1.0 - (var_residuals_bad / (var_returns + 1e-8))

        # When prediction error std ≈ return std, EV ≈ 0
        assert abs(ev_bad) < 0.5, f"Expected EV ≈ 0 with matched noise, got {ev_bad:.3f}"

    def test_value_scale_drift_impact(self):
        """
        Test that value scale drift between snapshot and training causes issues.

        If ret_std_snapshot is captured at time T but training uses data
        with different statistics, normalization will be inconsistent.
        """
        np.random.seed(42)

        # Snapshot statistics (from earlier data)
        snapshot_mean = 0.0
        snapshot_std = 0.01

        # Actual training data statistics (different regime)
        actual_mean = 0.005  # Market regime changed
        actual_std = 0.02    # Volatility doubled

        # Raw returns from actual distribution
        n_samples = 1000
        raw_returns = np.random.normal(actual_mean, actual_std, n_samples)

        # Normalize with snapshot statistics (WRONG)
        normalized_wrong = (raw_returns - snapshot_mean) / snapshot_std

        # Normalize with actual statistics (CORRECT)
        normalized_correct = (raw_returns - actual_mean) / actual_std

        # Check the impact
        # Wrong normalization will have mean ≠ 0 and std ≠ 1
        wrong_mean = np.mean(normalized_wrong)
        wrong_std = np.std(normalized_wrong)

        # This causes targets to be biased and scaled incorrectly
        assert abs(wrong_mean) > 0.1, "Mean should be significantly off"
        assert abs(wrong_std - 1.0) > 0.5, "Std should be significantly off"


# ============================================================================
# HYPOTHESIS 2: Twin Critics Loss Growth Due to Non-Stationary Targets
# ============================================================================

class TestTwinCriticsLossGrowth:
    """Test if non-stationary targets cause loss growth."""

    def test_loss_growth_with_changing_target_distribution(self):
        """
        Simulate how loss evolves when target distribution shifts.

        In RL, targets depend on policy which changes during training.
        If policy improves, target distribution shifts, and old critic
        predictions become less accurate.
        """
        torch.manual_seed(42)

        # Simple critic network
        critic = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

        losses = []

        for epoch in range(100):
            # Generate observations
            obs = torch.randn(64, 10)

            # Target distribution shifts over time (simulating policy improvement)
            # Early: targets ≈ N(0, 1)
            # Late: targets ≈ N(0.5, 0.8) - better policy, higher returns
            target_mean = 0.5 * (epoch / 100)
            target_std = 1.0 - 0.2 * (epoch / 100)
            targets = torch.randn(64, 1) * target_std + target_mean

            # Forward pass
            predictions = critic(obs)
            loss = nn.functional.mse_loss(predictions, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Check if loss grows in the second half
        first_half_mean = np.mean(losses[:50])
        second_half_mean = np.mean(losses[50:])

        # If target distribution shifts faster than critic can adapt, loss grows
        # This is EXPECTED behavior in RL, but shouldn't be +327%
        growth_rate = (second_half_mean - first_half_mean) / (first_half_mean + 1e-8)

        # Record the growth for analysis
        print(f"Loss growth rate: {growth_rate:.2%}")
        print(f"First half mean: {first_half_mean:.4f}, Second half mean: {second_half_mean:.4f}")

    def test_categorical_cross_entropy_with_target_shift(self):
        """
        Test C51 categorical loss behavior with shifting targets.

        Cross-entropy loss: -sum(target_dist * log(pred_dist))
        If target_dist changes, loss can increase even if predictions improve.
        """
        torch.manual_seed(42)

        num_atoms = 21
        v_min, v_max = -10.0, 10.0
        atoms = torch.linspace(v_min, v_max, num_atoms)

        losses = []

        for epoch in range(100):
            # Predicted distribution (softmax output)
            pred_logits = torch.randn(64, num_atoms)
            pred_probs = torch.softmax(pred_logits, dim=1)

            # Target distribution shifts over time
            # Early: centered around atom 10 (value = 0)
            # Late: centered around atom 15 (value = 5)
            target_atom = int(10 + 5 * (epoch / 100))
            target_dist = torch.zeros(64, num_atoms)
            target_dist[:, target_atom] = 1.0

            # Cross-entropy loss
            log_pred = torch.log(pred_probs + 1e-8)
            loss = -(target_dist * log_pred).sum(dim=1).mean()

            losses.append(loss.item())

        # Analyze loss trend
        first_quarter = np.mean(losses[:25])
        last_quarter = np.mean(losses[75:])

        print(f"First quarter loss: {first_quarter:.4f}")
        print(f"Last quarter loss: {last_quarter:.4f}")
        print(f"Growth: {(last_quarter - first_quarter) / first_quarter:.2%}")


# ============================================================================
# HYPOTHESIS 3: CVaR Constraint Violation Pattern
# ============================================================================

class TestCVaRConstraint:
    """Test CVaR constraint behavior."""

    def test_cvar_lambda_bounded_update(self):
        """
        Test that lambda updates are properly bounded to [0, 1].

        User reported 260,000% growth (0.00001 → 0.033).
        This is NOT exponential - it's just 0.033, well within bounds.
        """
        lambda_value = 0.00001
        lr = 0.01

        # Simulate constraint violations
        violations = [0.1, 0.05, 0.02, 0.01, 0.005]  # Decreasing violations

        for violation in violations:
            # Dual ascent update: λ += lr * violation
            candidate = lambda_value + lr * violation
            # Projection to [0, 1]
            lambda_value = max(0.0, min(1.0, candidate))

        # Final lambda should be bounded
        assert 0.0 <= lambda_value <= 1.0

        # With these violations, lambda should be about:
        # 0.00001 + 0.01 * (0.1 + 0.05 + 0.02 + 0.01 + 0.005) = 0.00001 + 0.00185 ≈ 0.00186
        expected = 0.00001 + lr * sum(violations)
        assert abs(lambda_value - expected) < 0.001

    def test_cvar_violation_semantic_correctness(self):
        """
        Test CVaR violation semantics.

        cvar_gap = cvar_limit - cvar_empirical
        - gap > 0: VIOLATION (CVaR below limit, too risky)
        - gap < 0: SATISFIED (CVaR above limit, safe)
        """
        # CVaR limit (threshold)
        cvar_limit = -0.05  # Allow up to 5% loss in tail

        # Case 1: Empirical CVaR worse than limit (violation)
        cvar_empirical_bad = -0.08  # 8% loss in tail
        gap_bad = cvar_limit - cvar_empirical_bad  # -0.05 - (-0.08) = 0.03 > 0
        assert gap_bad > 0, "Should be violation when CVaR worse than limit"

        # Case 2: Empirical CVaR better than limit (satisfied)
        cvar_empirical_good = -0.02  # 2% loss in tail
        gap_good = cvar_limit - cvar_empirical_good  # -0.05 - (-0.02) = -0.03 < 0
        assert gap_good < 0, "Should be satisfied when CVaR better than limit"

    def test_cvar_constraint_term_in_loss(self):
        """
        Test that CVaR constraint term is correctly computed.

        constraint_term = λ * max(0, cvar_limit - cvar_predicted)
        """
        lambda_value = 0.1
        cvar_limit = -0.05

        # Case 1: Violation
        cvar_predicted = -0.08
        gap = cvar_limit - cvar_predicted  # 0.03
        violation = max(0.0, gap)
        constraint_term = lambda_value * violation

        assert constraint_term == pytest.approx(0.003), "Constraint term should penalize violation"

        # Case 2: No violation
        cvar_predicted = -0.02
        gap = cvar_limit - cvar_predicted  # -0.03
        violation = max(0.0, gap)
        constraint_term = lambda_value * violation

        assert constraint_term == 0.0, "No penalty when constraint satisfied"


# ============================================================================
# HYPOTHESIS 4: Gradient Norm Collapse
# ============================================================================

class TestGradientNormCollapse:
    """Test gradient flow and potential vanishing gradients."""

    def test_vgs_scaling_effect(self):
        """
        Test VGS (Variance Gradient Scaler) impact on gradient norms.

        VGS scales gradients by 1/(1 + α * variance).
        If variance is high, gradients get scaled down significantly.
        """
        # Simulate gradient variance across layers
        np.random.seed(42)
        n_params = 1000

        # Gradients with high variance
        gradients_high_var = np.random.normal(0, 10.0, n_params)
        variance_high = np.var(gradients_high_var)

        # VGS scaling factor
        alpha = 0.1
        scale_high_var = 1.0 / (1.0 + alpha * variance_high)

        # Scaled gradients
        scaled_high = gradients_high_var * scale_high_var

        # Check reduction
        original_norm = np.linalg.norm(gradients_high_var)
        scaled_norm = np.linalg.norm(scaled_high)
        reduction = scaled_norm / original_norm

        print(f"Gradient variance: {variance_high:.2f}")
        print(f"VGS scale factor: {scale_high_var:.4f}")
        print(f"Norm reduction: {reduction:.4f}")

        # With variance=100, scale ≈ 1/(1+10) ≈ 0.09
        # This means 91% gradient reduction!
        assert scale_high_var < 0.2, "VGS should significantly scale down high-variance gradients"

    def test_gradient_norm_decay_pattern(self):
        """
        Test if gradient norm decay indicates learning or saturation.

        -82% gradient norm reduction could indicate:
        1. Model converged (good)
        2. Vanishing gradients (bad)
        3. Loss landscape flatness (neutral)
        """
        torch.manual_seed(42)

        # Simple network
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        grad_norms = []
        losses = []

        for epoch in range(200):
            # Simple regression task
            x = torch.randn(32, 10)
            y = torch.randn(32, 1)

            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()

            # Compute gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5

            grad_norms.append(total_norm)
            losses.append(loss.item())

            optimizer.step()

        # Analyze pattern
        initial_norm = np.mean(grad_norms[:10])
        final_norm = np.mean(grad_norms[-10:])
        norm_reduction = (initial_norm - final_norm) / initial_norm

        initial_loss = np.mean(losses[:10])
        final_loss = np.mean(losses[-10:])
        loss_reduction = (initial_loss - final_loss) / initial_loss

        print(f"Gradient norm reduction: {norm_reduction:.2%}")
        print(f"Loss reduction: {loss_reduction:.2%}")

        # If loss decreased significantly but grad norm collapsed,
        # it's likely convergence, not vanishing gradients


# ============================================================================
# HYPOTHESIS 5: EV ≈ 0 Due to Mismatch Between Predictions and Targets
# ============================================================================

class TestExplainedVarianceDiagnostics:
    """Diagnose why EV ≈ 0."""

    def test_ev_with_normalized_vs_raw_space(self):
        """
        Test EV computation in normalized vs raw space.

        EV = 1 - var(residuals) / var(targets)

        If targets are normalized (mean=0, std=1) but predictions
        are in different scale, EV will be wrong.
        """
        np.random.seed(42)
        n = 1000

        # Raw returns
        raw_returns = np.random.normal(0.01, 0.02, n)

        # Normalize
        ret_mean = raw_returns.mean()
        ret_std = raw_returns.std()
        normalized_returns = (raw_returns - ret_mean) / ret_std

        # Perfect predictions in normalized space
        perfect_pred_norm = normalized_returns.copy()

        # EV in normalized space (should be 1.0)
        var_resid_norm = np.var(normalized_returns - perfect_pred_norm)
        var_target_norm = np.var(normalized_returns)
        ev_norm = 1.0 - (var_resid_norm / (var_target_norm + 1e-8))

        assert ev_norm > 0.99, f"Perfect predictions should give EV ≈ 1, got {ev_norm}"

        # Predictions with small noise
        noisy_pred_norm = normalized_returns + np.random.normal(0, 0.1, n)
        var_resid_noisy = np.var(normalized_returns - noisy_pred_norm)
        ev_noisy = 1.0 - (var_resid_noisy / (var_target_norm + 1e-8))

        # With 10% noise std, EV ≈ 1 - 0.01/1.0 = 0.99 (still good!)
        # But if noise std = target std, EV ≈ 0
        high_noise_pred = normalized_returns + np.random.normal(0, 1.0, n)
        var_resid_high = np.var(normalized_returns - high_noise_pred)
        ev_high_noise = 1.0 - (var_resid_high / (var_target_norm + 1e-8))

        assert ev_high_noise < 0.5, f"High noise should give low EV, got {ev_high_noise}"

    def test_ev_with_constant_predictions(self):
        """
        Test EV when predictions are nearly constant (baseline model).

        If model just predicts the mean, EV = 0.
        """
        np.random.seed(42)
        n = 1000

        targets = np.random.normal(0, 1, n)

        # Constant prediction (mean)
        constant_pred = np.full(n, targets.mean())

        var_resid = np.var(targets - constant_pred)
        var_target = np.var(targets)
        ev = 1.0 - (var_resid / (var_target + 1e-8))

        # var_resid ≈ var_target when predicting mean
        assert abs(ev) < 0.1, f"Constant predictions should give EV ≈ 0, got {ev}"

    def test_ev_with_biased_predictions(self):
        """
        Test EV with systematic bias in predictions.
        """
        np.random.seed(42)
        n = 1000

        targets = np.random.normal(0, 1, n)

        # Biased predictions (add constant offset)
        bias = 0.5
        biased_pred = targets + bias

        var_resid = np.var(targets - biased_pred)  # var(bias) = 0
        var_target = np.var(targets)
        ev = 1.0 - (var_resid / (var_target + 1e-8))

        # Bias doesn't affect variance of residuals, so EV = 1.0!
        assert ev > 0.99, f"Biased but correlated predictions still give high EV: {ev}"

        # However, if we compute on raw differences...
        mse = np.mean((targets - biased_pred) ** 2)
        # MSE = bias^2 + var_resid = 0.25 + 0 = 0.25


# ============================================================================
# COMPREHENSIVE DIAGNOSTIC TEST
# ============================================================================

class TestComprehensiveDiagnostics:
    """Run all diagnostics together."""

    def test_identify_root_causes(self):
        """
        Comprehensive test to identify the most likely root cause(s).
        """
        results = {
            "target_bounds_issue": False,
            "value_scale_drift": False,
            "non_stationary_targets": True,  # Expected in RL
            "gradient_vanishing": False,
            "ev_computation_correct": True,
        }

        # Analysis based on symptoms:
        # 1. EV ≈ 0 → Predictions not better than mean
        # 2. Twin Critics loss +327% → Targets changing faster than learning
        # 3. CVaR Lambda +260,000% → From 0.00001 to 0.033, still in [0,1]
        # 4. Gradient norm -82% → Could be convergence or vanishing

        print("\n" + "="*60)
        print("DIAGNOSTIC ANALYSIS")
        print("="*60)

        print("\n1. EXPLAINED VARIANCE ~ 0")
        print("-" * 40)
        print("LIKELY CAUSE: Value function predicting near-constant values")
        print("              OR targets have very low variance (signal_only)")
        print("EVIDENCE: If predictions ≈ mean(targets), EV = 0")

        print("\n2. TWIN CRITICS LOSS +327%")
        print("-" * 40)
        print("LIKELY CAUSE: Non-stationary target distribution")
        print("              As policy improves, target distribution shifts")
        print("EXPECTED: In RL, critic loss can increase temporarily")
        print("CONCERNING IF: Loss keeps growing monotonically")

        print("\n3. CVaR LAMBDA +260,000%")
        print("-" * 40)
        print("NOT A BUG: Final value 0.033 is still << 1.0 (bounded)")
        print("MEANING: Constraint is being violated, lambda increases")
        print("EXPECTED: Lambda grows until constraint is satisfied")

        print("\n4. GRADIENT NORM -82%")
        print("-" * 40)
        print("POSSIBLE CAUSES:")
        print("  a) Model converging (good)")
        print("  b) VGS scaling down high-variance gradients (intended)")
        print("  c) Loss landscape becoming flat")
        print("  d) True vanishing gradients (bad)")
        print("NEED: Check if loss is still decreasing")

        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)

        print("""
1. CHECK VALUE FUNCTION PREDICTIONS:
   - Are predictions nearly constant?
   - Plot histogram of predictions vs targets

2. CHECK TARGET VARIANCE:
   - Is var(targets) very small (< 0.01)?
   - If so, EV ≈ 0 is expected (noise dominates)

3. MONITOR LOSS TRAJECTORY:
   - Is loss monotonically increasing? → Problem
   - Is loss oscillating around a value? → Normal

4. VERIFY GRADIENT FLOW:
   - Check gradients at each layer
   - VGS might be too aggressive

5. CONSIDER HYPERPARAMETERS:
   - Learning rate might be too high/low
   - VF coefficient might need adjustment
   - clip_range_vf might be too restrictive
        """)


# ============================================================================
# SPECIFIC BUG HUNT TESTS
# ============================================================================

class TestSpecificBugHunts:
    """Tests for specific potential bugs."""

    def test_ret_std_snapshot_timing(self):
        """
        Test if ret_std_snapshot is updated at wrong time.

        Bug: If snapshot is taken before new data arrives,
        normalization uses stale statistics.
        """
        # Simulate snapshot timing issue
        old_std = 0.01
        new_actual_std = 0.05  # 5x larger

        # Targets normalized with old std
        targets = np.random.normal(0, new_actual_std, 100)
        normalized_wrong = targets / old_std  # Will be 5x larger than expected!

        # This causes targets to exceed v_min/v_max bounds
        v_min, v_max = -10.0, 10.0
        frac_outside = np.mean((normalized_wrong < v_min) | (normalized_wrong > v_max))

        # If std mismatch is 5x, many targets will be clipped
        print(f"Targets outside [{v_min}, {v_max}]: {frac_outside:.2%}")
        print(f"Target range: [{normalized_wrong.min():.1f}, {normalized_wrong.max():.1f}]")

    def test_c51_projection_correctness(self):
        """
        Test C51 target distribution projection.

        Bug: If projection loses information, loss computation is wrong.
        """
        torch.manual_seed(42)

        num_atoms = 21
        v_min, v_max = -10.0, 10.0
        atoms = torch.linspace(v_min, v_max, num_atoms)
        delta_z = (v_max - v_min) / (num_atoms - 1)

        # Target value
        target_value = 3.7  # Should project between atoms 13 and 14

        # C51 projection
        b = (target_value - v_min) / delta_z
        lower = int(b)
        upper = min(lower + 1, num_atoms - 1)

        # Linear interpolation
        upper_weight = b - lower
        lower_weight = 1.0 - upper_weight

        # Expected atom values
        lower_atom = v_min + lower * delta_z
        upper_atom = v_min + upper * delta_z

        # Reconstructed value
        reconstructed = lower_weight * lower_atom + upper_weight * upper_atom

        # Should be close to original
        assert abs(reconstructed - target_value) < delta_z, (
            f"Projection error: {abs(reconstructed - target_value):.4f} > {delta_z:.4f}"
        )

    def test_twin_critics_independent_clipping(self):
        """
        Test that Twin Critics use independent clipping.

        Bug: If both critics are clipped relative to min(Q1, Q2),
        this violates PPO VF clipping semantics.
        """
        torch.manual_seed(42)

        # Old quantiles from buffer
        old_q1 = torch.randn(10, 21)
        old_q2 = torch.randn(10, 21)

        # Current quantiles (after forward pass)
        current_q1 = old_q1 + torch.randn_like(old_q1) * 0.5
        current_q2 = old_q2 + torch.randn_like(old_q2) * 0.5

        clip_delta = 0.1

        # CORRECT: Independent clipping
        q1_clipped_correct = old_q1 + torch.clamp(
            current_q1 - old_q1, -clip_delta, clip_delta
        )
        q2_clipped_correct = old_q2 + torch.clamp(
            current_q2 - old_q2, -clip_delta, clip_delta
        )

        # WRONG: Clipping relative to min(Q1, Q2)
        old_min = torch.min(old_q1, old_q2)
        q1_clipped_wrong = old_min + torch.clamp(
            current_q1 - old_min, -clip_delta, clip_delta
        )
        q2_clipped_wrong = old_min + torch.clamp(
            current_q2 - old_min, -clip_delta, clip_delta
        )

        # Verify they're different
        diff_q1 = (q1_clipped_correct - q1_clipped_wrong).abs().mean()
        diff_q2 = (q2_clipped_correct - q2_clipped_wrong).abs().mean()

        assert diff_q1 > 0.01 or diff_q2 > 0.01, (
            "Independent clipping should differ from shared clipping"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
