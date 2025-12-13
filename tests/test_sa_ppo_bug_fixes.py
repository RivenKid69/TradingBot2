"""
Comprehensive tests for SA-PPO bug fixes (2025-11-23)

Tests cover:
1. БАГ #1: Hardcoded max_updates fix
2. БАГ #2: KL divergence computation improvement

References:
- adversarial/sa_ppo.py
- БАГ #1: Lines 168-216 (_compute_max_updates)
- БАГ #2: Lines 351-365, 584-603 (analytical KL divergence)
"""

import logging
from unittest.mock import Mock

import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn
from torch import Tensor

from adversarial.sa_ppo import SAPPOConfig, StateAdversarialPPO
from adversarial.state_perturbation import PerturbationConfig


# Mock model for testing
class MockPolicy(nn.Module):
    """Mock policy for testing."""

    def __init__(self, state_dim: int = 10, action_dim: int = 5):
        super().__init__()
        self.fc = nn.Linear(state_dim, action_dim * 2)  # Mean and log_std
        self.action_dim = action_dim

    def get_distribution(self, obs: Tensor):
        """Get Gaussian distribution for actions.

        Returns Independent distribution so that log_prob() returns
        scalar log probabilities (summed over action dimensions).
        """
        output = self.fc(obs)
        mean = output[:, : self.action_dim]
        log_std = output[:, self.action_dim :]
        std = torch.exp(log_std)
        # Use Independent to sum log_probs over action dimensions
        return torch.distributions.Independent(
            torch.distributions.Normal(mean, std), reinterpreted_batch_ndims=1
        )

    def predict_values(self, obs: Tensor) -> Tensor:
        """Predict values (dummy implementation)."""
        return torch.zeros(obs.size(0), 1)


class MockModel:
    """Mock PPO model for testing."""

    def __init__(
        self,
        total_timesteps: int = None,
        n_steps: int = None,
        num_timesteps: int = 0,
    ):
        self.total_timesteps = total_timesteps
        self.n_steps = n_steps
        self.num_timesteps = num_timesteps
        self.policy = MockPolicy()


# ============================================================================
# БАГ #1: Hardcoded max_updates fix tests
# ============================================================================


def test_bug1_max_updates_from_config_override():
    """Test БАГ #1 fix: config.max_updates override has highest priority."""
    config = SAPPOConfig(
        enabled=True,
        adaptive_epsilon=True,
        max_updates=5000,  # Explicit override
    )
    model = MockModel(total_timesteps=100000, n_steps=2048)
    sa_ppo = StateAdversarialPPO(config, model)

    # Should use config override (5000), not computed value (100000 // 2048 = 48)
    assert sa_ppo._max_updates == 5000


def test_bug1_max_updates_computed_from_model():
    """Test БАГ #1 fix: max_updates computed from total_timesteps / n_steps."""
    config = SAPPOConfig(enabled=True, adaptive_epsilon=True)
    model = MockModel(total_timesteps=100000, n_steps=2048)
    sa_ppo = StateAdversarialPPO(config, model)

    # Should compute: 100000 // 2048 = 48
    expected = 100000 // 2048
    assert sa_ppo._max_updates == expected


def test_bug1_max_updates_fallback_from_progress():
    """Test БАГ #1 fix: fallback to inferring from current progress."""
    config = SAPPOConfig(enabled=True, adaptive_epsilon=True)
    # No total_timesteps, but has num_timesteps (current progress)
    model = MockModel(total_timesteps=None, n_steps=2048, num_timesteps=50000)
    sa_ppo = StateAdversarialPPO(config, model)

    # Should estimate: (50000 * 2) // 2048 = 48
    expected = (50000 * 2) // 2048
    assert sa_ppo._max_updates == expected


def test_bug1_max_updates_conservative_default():
    """Test БАГ #1 fix: conservative default (10000) when no info available."""
    config = SAPPOConfig(enabled=True, adaptive_epsilon=True)
    model = MockModel()  # No timestep info
    sa_ppo = StateAdversarialPPO(config, model)

    # Should use conservative default (10000, not old hardcoded 1000)
    assert sa_ppo._max_updates == 10000


def test_bug1_epsilon_schedule_uses_computed_max_updates():
    """Test БАГ #1 fix: epsilon schedule uses computed max_updates."""
    config = SAPPOConfig(
        enabled=True,
        adaptive_epsilon=True,
        epsilon_schedule="linear",
        perturbation=PerturbationConfig(epsilon=0.1),
        epsilon_final=0.01,
    )
    model = MockModel(total_timesteps=100000, n_steps=2048)
    sa_ppo = StateAdversarialPPO(config, model)

    max_updates = 100000 // 2048  # 48

    # Test epsilon schedule at different progress points
    sa_ppo._update_count = 0
    epsilon_0 = sa_ppo._get_current_epsilon()
    assert abs(epsilon_0 - 0.1) < 1e-6  # Initial epsilon

    sa_ppo._update_count = max_updates // 2
    epsilon_half = sa_ppo._get_current_epsilon()
    assert abs(epsilon_half - 0.055) < 1e-3  # Midpoint: (0.1 + 0.01) / 2

    sa_ppo._update_count = max_updates
    epsilon_final = sa_ppo._get_current_epsilon()
    assert abs(epsilon_final - 0.01) < 1e-6  # Final epsilon


def test_bug1_epsilon_schedule_linear():
    """Test БАГ #1 fix: linear epsilon schedule works correctly."""
    config = SAPPOConfig(
        enabled=True,
        adaptive_epsilon=True,
        epsilon_schedule="linear",
        perturbation=PerturbationConfig(epsilon=0.15),
        epsilon_final=0.03,
        max_updates=100,  # Explicit for determinism
    )
    model = MockModel()
    sa_ppo = StateAdversarialPPO(config, model)

    # Test linear interpolation
    sa_ppo._update_count = 0
    assert abs(sa_ppo._get_current_epsilon() - 0.15) < 1e-6

    sa_ppo._update_count = 50
    assert abs(sa_ppo._get_current_epsilon() - 0.09) < 1e-3  # (0.15 + 0.03) / 2

    sa_ppo._update_count = 100
    assert abs(sa_ppo._get_current_epsilon() - 0.03) < 1e-6


def test_bug1_epsilon_schedule_cosine():
    """Test БАГ #1 fix: cosine epsilon schedule works correctly."""
    import math

    config = SAPPOConfig(
        enabled=True,
        adaptive_epsilon=True,
        epsilon_schedule="cosine",
        perturbation=PerturbationConfig(epsilon=0.15),
        epsilon_final=0.03,
        max_updates=100,
    )
    model = MockModel()
    sa_ppo = StateAdversarialPPO(config, model)

    # Test cosine schedule
    sa_ppo._update_count = 0
    epsilon_0 = sa_ppo._get_current_epsilon()
    assert abs(epsilon_0 - 0.15) < 1e-6

    sa_ppo._update_count = 50  # Midpoint
    # Cosine formula: final + 0.5 * (init - final) * (1 + cos(π * 0.5))
    # = 0.03 + 0.5 * (0.15 - 0.03) * (1 + cos(π * 0.5))
    # = 0.03 + 0.5 * 0.12 * (1 + 0) = 0.03 + 0.06 = 0.09
    epsilon_half = sa_ppo._get_current_epsilon()
    assert abs(epsilon_half - 0.09) < 1e-3

    sa_ppo._update_count = 100
    epsilon_final = sa_ppo._get_current_epsilon()
    assert abs(epsilon_final - 0.03) < 1e-6


# ============================================================================
# БАГ #2: KL divergence computation improvement tests
# ============================================================================


def test_bug2_kl_divergence_analytical_used():
    """Test БАГ #2 fix: analytical KL divergence is used when available."""
    config = SAPPOConfig(
        enabled=True,
        robust_kl_coef=0.1,
        warmup_updates=0,
    )
    model = MockModel()
    sa_ppo = StateAdversarialPPO(config, model)
    sa_ppo._adversarial_enabled = True
    sa_ppo._update_count = 10

    batch_size = 32
    state_dim = 10
    action_dim = 5

    states_clean = torch.randn(batch_size, state_dim)
    states_adv = states_clean + torch.randn_like(states_clean) * 0.01  # Small perturbation
    actions = torch.randn(batch_size, action_dim)

    penalty, info = sa_ppo.compute_robust_kl_penalty(states_clean, states_adv, actions)

    # Should use analytical KL divergence for Normal distributions
    assert info["sa_ppo/kl_method"] == "analytical"
    assert "sa_ppo/kl_divergence" in info
    assert penalty > 0  # Should have positive KL divergence


def test_bug2_kl_divergence_symmetry():
    """Test БАГ #2 fix: KL divergence is NOT symmetric (as expected)."""
    config = SAPPOConfig(
        enabled=True,
        robust_kl_coef=0.1,
        warmup_updates=0,
    )
    model = MockModel()
    sa_ppo = StateAdversarialPPO(config, model)
    sa_ppo._adversarial_enabled = True
    sa_ppo._update_count = 10

    batch_size = 32
    state_dim = 10
    action_dim = 5

    states_clean = torch.randn(batch_size, state_dim)
    states_adv = states_clean + torch.randn_like(states_clean) * 0.05
    actions = torch.randn(batch_size, action_dim)

    # KL(clean || adv)
    penalty_1, info_1 = sa_ppo.compute_robust_kl_penalty(states_clean, states_adv, actions)

    # KL(adv || clean) - should be different
    penalty_2, info_2 = sa_ppo.compute_robust_kl_penalty(states_adv, states_clean, actions)

    # KL divergence is NOT symmetric
    assert penalty_1 != penalty_2
    assert info_1["sa_ppo/kl_method"] == "analytical"
    assert info_2["sa_ppo/kl_method"] == "analytical"


def test_bug2_kl_divergence_zero_perturbation():
    """Test БАГ #2 fix: KL divergence is ~0 when no perturbation."""
    config = SAPPOConfig(
        enabled=True,
        robust_kl_coef=0.1,
        warmup_updates=0,
    )
    model = MockModel()
    sa_ppo = StateAdversarialPPO(config, model)
    sa_ppo._adversarial_enabled = True
    sa_ppo._update_count = 10

    batch_size = 32
    state_dim = 10
    action_dim = 5

    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)

    # Same states -> KL divergence should be ~0
    penalty, info = sa_ppo.compute_robust_kl_penalty(states, states, actions)

    assert info["sa_ppo/kl_method"] == "analytical"
    assert abs(info["sa_ppo/kl_divergence"]) < 1e-5  # ~0


def test_bug2_kl_divergence_disabled():
    """Test БАГ #2 fix: KL divergence returns 0 when disabled."""
    config = SAPPOConfig(
        enabled=True,
        robust_kl_coef=0.0,  # Disabled
        warmup_updates=0,
    )
    model = MockModel()
    sa_ppo = StateAdversarialPPO(config, model)
    sa_ppo._adversarial_enabled = True
    sa_ppo._update_count = 10

    batch_size = 32
    state_dim = 10
    action_dim = 5

    states_clean = torch.randn(batch_size, state_dim)
    states_adv = states_clean + torch.randn_like(states_clean) * 0.05
    actions = torch.randn(batch_size, action_dim)

    penalty, info = sa_ppo.compute_robust_kl_penalty(states_clean, states_adv, actions)

    assert penalty == 0.0


def test_bug2_kl_divergence_empty_batch():
    """Test БАГ #2 fix: KL divergence handles empty batch gracefully."""
    config = SAPPOConfig(
        enabled=True,
        robust_kl_coef=0.1,
        warmup_updates=0,
    )
    model = MockModel()
    sa_ppo = StateAdversarialPPO(config, model)
    sa_ppo._adversarial_enabled = True
    sa_ppo._update_count = 10

    state_dim = 10
    action_dim = 5

    states_clean = torch.randn(0, state_dim)  # Empty batch
    states_adv = torch.randn(0, state_dim)
    actions = torch.randn(0, action_dim)

    penalty, info = sa_ppo.compute_robust_kl_penalty(states_clean, states_adv, actions)

    assert penalty == 0.0


def test_bug2_kl_divergence_in_compute_adversarial_loss():
    """Test БАГ #2 fix: KL divergence method is logged in compute_adversarial_loss."""
    config = SAPPOConfig(
        enabled=True,
        robust_kl_coef=0.1,
        warmup_updates=0,
        attack_policy=False,  # Disable attack to avoid perturbation generation issues
        attack_value=False,   # Disable value attack (Mock predict_values has no grad)
        adversarial_ratio=0.5,
    )
    model = MockModel()
    sa_ppo = StateAdversarialPPO(config, model)
    sa_ppo._adversarial_enabled = True
    sa_ppo._update_count = 10

    batch_size = 32
    state_dim = 10
    action_dim = 5

    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)
    advantages = torch.randn(batch_size)
    returns = torch.randn(batch_size)

    # For multi-dimensional actions, log_probs should be sum over action dimensions
    # Generate old_log_probs from the distribution
    with torch.no_grad():
        dist = model.policy.get_distribution(states)
        old_log_probs = dist.log_prob(actions)  # This handles multi-dim correctly

    loss, info = sa_ppo.compute_adversarial_loss(
        states=states,
        actions=actions,
        advantages=advantages,
        returns=returns,
        old_log_probs=old_log_probs,
        clip_range=0.2,
    )

    # Should log KL method
    assert "sa_ppo/kl_method" in info
    # Should be analytical for Normal distributions (or "none" if no adversarial samples)
    assert info["sa_ppo/kl_method"] in ["analytical", "monte_carlo", "none"]


def test_bug2_kl_divergence_magnitude_reasonable():
    """Test БАГ #2 fix: KL divergence magnitude is reasonable for typical perturbations."""
    config = SAPPOConfig(
        enabled=True,
        robust_kl_coef=0.1,
        warmup_updates=0,
    )
    model = MockModel()
    sa_ppo = StateAdversarialPPO(config, model)
    sa_ppo._adversarial_enabled = True
    sa_ppo._update_count = 10

    batch_size = 32
    state_dim = 10
    action_dim = 5

    states_clean = torch.randn(batch_size, state_dim)
    # Small perturbation (epsilon=0.01)
    states_adv = states_clean + torch.randn_like(states_clean) * 0.01
    actions = torch.randn(batch_size, action_dim)

    penalty, info = sa_ppo.compute_robust_kl_penalty(states_clean, states_adv, actions)

    # For small perturbations, KL divergence should be small but non-zero
    kl_div = info["sa_ppo/kl_divergence"]
    assert 0 < kl_div < 1.0  # Reasonable range for small perturbations


# ============================================================================
# Integration tests (both fixes together)
# ============================================================================


def test_integration_both_fixes_work_together():
    """Integration test: Both БАГ #1 and БАГ #2 fixes work together."""
    config = SAPPOConfig(
        enabled=True,
        adaptive_epsilon=True,
        epsilon_schedule="linear",
        robust_kl_coef=0.1,
        warmup_updates=0,
        perturbation=PerturbationConfig(epsilon=0.1),
        epsilon_final=0.01,
    )
    model = MockModel(total_timesteps=100000, n_steps=2048)
    sa_ppo = StateAdversarialPPO(config, model)
    sa_ppo._adversarial_enabled = True
    sa_ppo._update_count = 10

    # БАГ #1: max_updates computed correctly
    expected_max_updates = 100000 // 2048
    assert sa_ppo._max_updates == expected_max_updates

    # БАГ #2: KL divergence uses analytical method
    batch_size = 32
    state_dim = 10
    action_dim = 5

    states_clean = torch.randn(batch_size, state_dim)
    states_adv = states_clean + torch.randn_like(states_clean) * 0.01
    actions = torch.randn(batch_size, action_dim)

    penalty, info = sa_ppo.compute_robust_kl_penalty(states_clean, states_adv, actions)

    assert info["sa_ppo/kl_method"] == "analytical"
    assert penalty > 0


def test_integration_epsilon_schedule_and_kl_over_training():
    """Integration test: Epsilon schedule and KL computation work correctly over training."""
    config = SAPPOConfig(
        enabled=True,
        adaptive_epsilon=True,
        epsilon_schedule="linear",
        robust_kl_coef=0.1,
        warmup_updates=5,
        perturbation=PerturbationConfig(epsilon=0.15),
        epsilon_final=0.03,
        max_updates=100,
    )
    model = MockModel()
    sa_ppo = StateAdversarialPPO(config, model)

    batch_size = 32
    state_dim = 10
    action_dim = 5

    # Track epsilon values
    epsilon_values = []

    # Simulate training updates
    for update in range(1, 101):
        # БАГ #1: Epsilon schedule progresses correctly
        # Check epsilon BEFORE on_update_start
        epsilon_before = sa_ppo._get_current_epsilon()
        epsilon_values.append(epsilon_before)

        sa_ppo.on_update_start()  # Increments _update_count and updates epsilon

        # БАГ #2: KL divergence computation (only after warmup)
        if sa_ppo.is_adversarial_enabled:
            states_clean = torch.randn(batch_size, state_dim)
            states_adv = states_clean + torch.randn_like(states_clean) * 0.01
            actions = torch.randn(batch_size, action_dim)

            penalty, info = sa_ppo.compute_robust_kl_penalty(states_clean, states_adv, actions)

            assert info["sa_ppo/kl_method"] == "analytical"

    # Check epsilon schedule progression
    # At update 1 (before on_update_start), _update_count = 0 -> epsilon = 0.15
    assert abs(epsilon_values[0] - 0.15) < 1e-6  # Initial

    # Check that epsilon decreases over time (БАГ #1: uses computed max_updates, not hardcoded 1000)
    # Should decrease from 0.15 to 0.03 over 100 updates
    assert epsilon_values[0] > epsilon_values[99]  # Should decrease
    assert epsilon_values[0] - epsilon_values[99] > 0.10  # Significant decrease

    # Check monotonic decrease for linear schedule
    for i in range(len(epsilon_values) - 1):
        assert epsilon_values[i] >= epsilon_values[i + 1]  # Should decrease monotonically

    # Check that epsilon approaches final value
    assert abs(epsilon_values[99] - 0.03) < 0.02  # Should be close to final epsilon (0.03)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
