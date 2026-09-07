"""
Comprehensive tests for State Perturbation module.

Tests cover:
- PerturbationConfig validation
- StatePerturbation initialization
- FGSM attack generation
- PGD attack generation
- Perturbation norms and constraints
- State clipping
- Statistics tracking
- Edge cases and error handling

Target: 100% code coverage
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import numpy as np

from adversarial.state_perturbation import (
    PerturbationConfig,
    StatePerturbation,
    create_policy_loss_fn,
    create_value_loss_fn,
)


# Module-level fixtures that can be used by all test classes
@pytest.fixture
def config():
    """Default perturbation configuration."""
    return PerturbationConfig(
        epsilon=0.1,
        attack_steps=3,
        attack_lr=0.03,
        random_init=True,
        norm_type="linf",
    )


@pytest.fixture
def perturbation(config):
    """StatePerturbation instance."""
    return StatePerturbation(config)


@pytest.fixture
def dummy_loss_fn():
    """Dummy loss function for testing."""

    def loss_fn(state):
        # Simple loss: sum of squares
        return (state**2).sum()

    return loss_fn


class TestPerturbationConfig:
    """Tests for PerturbationConfig dataclass."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = PerturbationConfig()
        assert config.epsilon == 0.075
        assert config.attack_steps == 3
        assert config.attack_lr == 0.03
        assert config.random_init is True
        assert config.norm_type == "linf"
        assert config.clip_min is None
        assert config.clip_max is None
        assert config.attack_method == "pgd"

    def test_custom_initialization(self):
        """Test custom configuration values."""
        config = PerturbationConfig(
            epsilon=0.1,
            attack_steps=5,
            attack_lr=0.01,
            random_init=False,
            norm_type="l2",
            clip_min=-1.0,
            clip_max=1.0,
            attack_method="fgsm",
        )
        assert config.epsilon == 0.1
        assert config.attack_steps == 5
        assert config.attack_lr == 0.01
        assert config.random_init is False
        assert config.norm_type == "l2"
        assert config.clip_min == -1.0
        assert config.clip_max == 1.0
        assert config.attack_method == "fgsm"

    def test_validation_negative_epsilon(self):
        """Test that negative epsilon raises ValueError."""
        with pytest.raises(ValueError, match="epsilon must be >= 0"):
            PerturbationConfig(epsilon=-0.1)

    def test_validation_zero_attack_steps(self):
        """Test that zero attack_steps raises ValueError."""
        with pytest.raises(ValueError, match="attack_steps must be >= 1"):
            PerturbationConfig(attack_steps=0)

    def test_validation_negative_attack_lr(self):
        """Test that negative attack_lr raises ValueError."""
        with pytest.raises(ValueError, match="attack_lr must be > 0"):
            PerturbationConfig(attack_lr=-0.01)

    def test_validation_invalid_norm_type(self):
        """Test that invalid norm_type raises ValueError."""
        with pytest.raises(ValueError, match="norm_type must be 'linf' or 'l2'"):
            PerturbationConfig(norm_type="l1")

    def test_validation_invalid_attack_method(self):
        """Test that invalid attack_method raises ValueError."""
        with pytest.raises(ValueError, match="attack_method must be 'pgd' or 'fgsm'"):
            PerturbationConfig(attack_method="invalid")

    def test_validation_clip_min_greater_than_clip_max(self):
        """Test that clip_min >= clip_max raises ValueError."""
        with pytest.raises(ValueError, match="clip_min .* must be < clip_max"):
            PerturbationConfig(clip_min=1.0, clip_max=0.5)


class TestStatePerturbation:
    """Tests for StatePerturbation class."""

    def test_initialization(self, config):
        """Test StatePerturbation initialization."""
        perturbation = StatePerturbation(config)
        assert perturbation.config == config
        assert perturbation._attack_count == 0
        assert perturbation._total_perturbation_norm == 0.0

    def test_reset_stats(self, perturbation):
        """Test statistics reset."""
        # Set some stats
        perturbation._attack_count = 10
        perturbation._total_perturbation_norm = 5.0

        # Reset
        perturbation.reset_stats()

        assert perturbation._attack_count == 0
        assert perturbation._total_perturbation_norm == 0.0

    def test_get_stats_empty(self, perturbation):
        """Test get_stats with no attacks."""
        stats = perturbation.get_stats()
        assert stats["attack_count"] == 0
        assert stats["avg_perturbation_norm"] == 0.0

    def test_get_stats_after_attacks(self, perturbation):
        """Test get_stats after some attacks."""
        perturbation._attack_count = 5
        perturbation._total_perturbation_norm = 2.5

        stats = perturbation.get_stats()
        assert stats["attack_count"] == 5
        assert stats["avg_perturbation_norm"] == 0.5

    def test_fgsm_attack_linf(self, perturbation, dummy_loss_fn):
        """Test FGSM attack with L-infinity norm."""
        state = torch.randn(4, 10)
        delta = perturbation.fgsm_attack(state, dummy_loss_fn)

        # Check shape
        assert delta.shape == state.shape

        # Check norm constraint
        assert torch.abs(delta).max().item() <= perturbation.config.epsilon + 1e-6

        # Check that perturbation is not zero
        assert delta.abs().sum().item() > 0

    def test_fgsm_attack_l2(self, dummy_loss_fn):
        """Test FGSM attack with L2 norm."""
        config = PerturbationConfig(epsilon=0.1, attack_method="fgsm", norm_type="l2")
        perturbation = StatePerturbation(config)

        state = torch.randn(4, 10)
        delta = perturbation.fgsm_attack(state, dummy_loss_fn)

        # Check shape
        assert delta.shape == state.shape

        # Check L2 norm constraint
        l2_norms = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1)
        assert torch.all(l2_norms <= config.epsilon + 1e-4)

    def test_fgsm_attack_with_clipping(self, dummy_loss_fn):
        """Test FGSM attack with state clipping."""
        config = PerturbationConfig(
            epsilon=0.5,
            attack_method="fgsm",
            clip_min=-1.0,
            clip_max=1.0,
        )
        perturbation = StatePerturbation(config)

        state = torch.randn(4, 10)
        delta = perturbation.fgsm_attack(state, dummy_loss_fn)

        # Check that perturbed state is within bounds
        perturbed_state = state + delta
        assert torch.all(perturbed_state >= -1.0 - 1e-6)
        assert torch.all(perturbed_state <= 1.0 + 1e-6)

    def test_pgd_attack_linf(self, perturbation, dummy_loss_fn):
        """Test PGD attack with L-infinity norm."""
        state = torch.randn(4, 10)
        delta = perturbation.pgd_attack(state, dummy_loss_fn)

        # Check shape
        assert delta.shape == state.shape

        # Check norm constraint
        assert torch.abs(delta).max().item() <= perturbation.config.epsilon + 1e-6

        # Check that perturbation is not zero
        assert delta.abs().sum().item() > 0

    def test_pgd_attack_l2(self, dummy_loss_fn):
        """Test PGD attack with L2 norm."""
        config = PerturbationConfig(
            epsilon=0.1,
            attack_steps=5,
            attack_lr=0.03,
            norm_type="l2",
            random_init=True,
        )
        perturbation = StatePerturbation(config)

        state = torch.randn(4, 10)
        delta = perturbation.pgd_attack(state, dummy_loss_fn)

        # Check shape
        assert delta.shape == state.shape

        # Check L2 norm constraint
        l2_norms = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1)
        assert torch.all(l2_norms <= config.epsilon + 1e-4)

    def test_pgd_attack_no_random_init(self, dummy_loss_fn):
        """Test PGD attack without random initialization."""
        config = PerturbationConfig(
            epsilon=0.1,
            attack_steps=3,
            attack_lr=0.03,
            random_init=False,
        )
        perturbation = StatePerturbation(config)

        state = torch.randn(4, 10)
        delta = perturbation.pgd_attack(state, dummy_loss_fn)

        # Should still produce valid perturbation
        assert delta.shape == state.shape
        assert torch.abs(delta).max().item() <= config.epsilon + 1e-6

    def test_pgd_attack_with_clipping(self, dummy_loss_fn):
        """Test PGD attack with state clipping."""
        config = PerturbationConfig(
            epsilon=0.5,
            attack_steps=3,
            attack_lr=0.1,
            clip_min=-1.0,
            clip_max=1.0,
        )
        perturbation = StatePerturbation(config)

        state = torch.randn(4, 10)
        delta = perturbation.pgd_attack(state, dummy_loss_fn)

        # Check that perturbed state is within bounds
        perturbed_state = state + delta
        assert torch.all(perturbed_state >= -1.0 - 1e-6)
        assert torch.all(perturbed_state <= 1.0 + 1e-6)

    def test_generate_perturbation_fgsm(self, dummy_loss_fn):
        """Test generate_perturbation with FGSM method."""
        config = PerturbationConfig(attack_method="fgsm")
        perturbation = StatePerturbation(config)

        state = torch.randn(4, 10)
        delta = perturbation.generate_perturbation(state, dummy_loss_fn)

        assert delta.shape == state.shape
        assert perturbation._attack_count == 1

    def test_generate_perturbation_pgd(self, dummy_loss_fn):
        """Test generate_perturbation with PGD method."""
        config = PerturbationConfig(attack_method="pgd")
        perturbation = StatePerturbation(config)

        state = torch.randn(4, 10)
        delta = perturbation.generate_perturbation(state, dummy_loss_fn)

        assert delta.shape == state.shape
        assert perturbation._attack_count == 1

    def test_generate_perturbation_invalid_method(self, perturbation, dummy_loss_fn):
        """Test generate_perturbation with invalid method."""
        perturbation.config.attack_method = "invalid"

        state = torch.randn(4, 10)
        with pytest.raises(ValueError, match="Unknown attack method"):
            perturbation.generate_perturbation(state, dummy_loss_fn)

    def test_pgd_attack_unsupported_norm(self, dummy_loss_fn):
        """Test PGD attack with unsupported norm type."""
        config = PerturbationConfig(epsilon=0.1, attack_steps=3)
        perturbation = StatePerturbation(config)
        # Manually set invalid norm_type to test error handling
        perturbation.config.norm_type = "l1"

        state = torch.randn(4, 10)
        with pytest.raises(ValueError, match="Unsupported norm_type"):
            perturbation.pgd_attack(state, dummy_loss_fn)

    def test_fgsm_attack_unsupported_norm(self, dummy_loss_fn):
        """Test FGSM attack with unsupported norm type."""
        config = PerturbationConfig(epsilon=0.1, attack_method="fgsm")
        perturbation = StatePerturbation(config)
        # Manually set invalid norm_type to test error handling
        perturbation.config.norm_type = "l1"

        state = torch.randn(4, 10)
        with pytest.raises(ValueError, match="Unsupported norm_type"):
            perturbation.fgsm_attack(state, dummy_loss_fn)

    def test_clip_state_both_bounds(self, perturbation):
        """Test _clip_state with both min and max bounds."""
        perturbation.config.clip_min = -1.0
        perturbation.config.clip_max = 1.0

        state_adv = torch.tensor([[-2.0, 0.5], [0.0, 2.0]])
        state_orig = torch.zeros_like(state_adv)

        clipped = perturbation._clip_state(state_adv, state_orig)

        assert torch.all(clipped >= -1.0)
        assert torch.all(clipped <= 1.0)
        assert torch.allclose(clipped, torch.tensor([[-1.0, 0.5], [0.0, 1.0]]))

    def test_clip_state_min_only(self, perturbation):
        """Test _clip_state with only min bound."""
        perturbation.config.clip_min = 0.0
        perturbation.config.clip_max = None

        state_adv = torch.tensor([[-1.0, 0.5], [0.0, 2.0]])
        state_orig = torch.zeros_like(state_adv)

        clipped = perturbation._clip_state(state_adv, state_orig)

        assert torch.all(clipped >= 0.0)
        assert torch.allclose(clipped, torch.tensor([[0.0, 0.5], [0.0, 2.0]]))

    def test_clip_state_max_only(self, perturbation):
        """Test _clip_state with only max bound."""
        perturbation.config.clip_min = None
        perturbation.config.clip_max = 1.0

        state_adv = torch.tensor([[-1.0, 0.5], [0.0, 2.0]])
        state_orig = torch.zeros_like(state_adv)

        clipped = perturbation._clip_state(state_adv, state_orig)

        assert torch.all(clipped <= 1.0)
        assert torch.allclose(clipped, torch.tensor([[-1.0, 0.5], [0.0, 1.0]]))

    def test_clip_state_no_bounds(self, perturbation):
        """Test _clip_state with no bounds."""
        perturbation.config.clip_min = None
        perturbation.config.clip_max = None

        state_adv = torch.tensor([[-2.0, 0.5], [0.0, 2.0]])
        state_orig = torch.zeros_like(state_adv)

        clipped = perturbation._clip_state(state_adv, state_orig)

        assert torch.allclose(clipped, state_adv)

    def test_update_stats_linf(self, perturbation):
        """Test _update_stats with L-infinity norm."""
        delta = torch.tensor([[0.1, -0.2], [0.05, 0.15]])

        perturbation._update_stats(delta)

        assert perturbation._attack_count == 1
        # Max per sample: [0.2, 0.15], mean = 0.175
        expected_norm = (0.2 + 0.15) / 2
        assert abs(perturbation._total_perturbation_norm - expected_norm) < 1e-6

    def test_update_stats_l2(self):
        """Test _update_stats with L2 norm."""
        config = PerturbationConfig(norm_type="l2")
        perturbation = StatePerturbation(config)

        delta = torch.tensor([[0.3, 0.4], [0.0, 0.0]])  # L2 norm = 0.5 for first row

        perturbation._update_stats(delta)

        assert perturbation._attack_count == 1
        # Average L2 norm across batch
        expected_norm = (0.5 + 0.0) / 2
        assert abs(perturbation._total_perturbation_norm - expected_norm) < 1e-5

    def test_stats_tracking_multiple_attacks(self, perturbation, dummy_loss_fn):
        """Test statistics tracking over multiple attacks."""
        state = torch.randn(4, 10)

        # Perform multiple attacks
        for _ in range(5):
            perturbation.generate_perturbation(state, dummy_loss_fn)

        stats = perturbation.get_stats()
        assert stats["attack_count"] == 5
        assert stats["avg_perturbation_norm"] > 0

    def test_perturbation_increases_loss(self, perturbation):
        """Test that adversarial perturbation increases loss."""
        state = torch.randn(4, 10)

        def loss_fn(s):
            return (s**2).sum()

        # Original loss
        original_loss = loss_fn(state).item()

        # Generate perturbation
        delta = perturbation.generate_perturbation(state, loss_fn)
        perturbed_state = state + delta

        # Perturbed loss
        perturbed_loss = loss_fn(perturbed_state).item()

        # Adversarial perturbation should increase loss
        assert perturbed_loss >= original_loss - 1e-4


class TestLossFunctions:
    """Tests for helper loss functions."""

    def test_create_policy_loss_fn_placeholder(self):
        """Test create_policy_loss_fn (placeholder for integration tests)."""
        # This function is designed for integration with actual models
        # Here we just verify it exists and can be called
        assert callable(create_policy_loss_fn)

    def test_create_value_loss_fn_placeholder(self):
        """Test create_value_loss_fn (placeholder for integration tests)."""
        # This function is designed for integration with actual models
        # Here we just verify it exists and can be called
        assert callable(create_value_loss_fn)


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_zero_epsilon(self, dummy_loss_fn):
        """Test perturbation with epsilon=0."""
        config = PerturbationConfig(epsilon=0.0)
        perturbation = StatePerturbation(config)

        state = torch.randn(4, 10)
        delta = perturbation.generate_perturbation(state, dummy_loss_fn)

        # Delta should be all zeros
        assert torch.allclose(delta, torch.zeros_like(delta))

    def test_single_sample(self, dummy_loss_fn):
        """Test perturbation with single sample (batch_size=1)."""
        config = PerturbationConfig()
        perturbation = StatePerturbation(config)

        state = torch.randn(1, 10)
        delta = perturbation.generate_perturbation(state, dummy_loss_fn)

        assert delta.shape == state.shape

    def test_large_batch(self, dummy_loss_fn):
        """Test perturbation with large batch."""
        config = PerturbationConfig()
        perturbation = StatePerturbation(config)

        state = torch.randn(100, 10)
        delta = perturbation.generate_perturbation(state, dummy_loss_fn)

        assert delta.shape == state.shape

    def test_high_dimensional_state(self, dummy_loss_fn):
        """Test perturbation with high-dimensional state."""
        config = PerturbationConfig()
        perturbation = StatePerturbation(config)

        state = torch.randn(4, 28, 28, 3)  # Image-like state
        delta = perturbation.generate_perturbation(state, dummy_loss_fn)

        assert delta.shape == state.shape

    def test_gradient_computation_disabled_outside_attack(self, perturbation):
        """Test that gradients are properly managed."""
        state = torch.randn(4, 10, requires_grad=False)

        def loss_fn(s):
            return (s**2).sum()

        delta = perturbation.generate_perturbation(state, loss_fn)

        # Delta should not require gradients
        assert not delta.requires_grad
        # Original state should not require gradients
        assert not state.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
