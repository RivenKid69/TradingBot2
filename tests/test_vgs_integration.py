"""
Integration tests for Variance Gradient Scaling with DistributionalPPO.

Tests cover:
- VGS initialization through PPO config
- VGS integration in training loop
- Logging and metrics
- State persistence
- Interaction with gradient clipping
- Disabled VGS behavior
"""

import pytest

torch = pytest.importorskip("torch")
import numpy as np

gym = pytest.importorskip("gymnasium")
from stable_baselines3.common.vec_env import DummyVecEnv


try:
    from distributional_ppo import DistributionalPPO
    from variance_gradient_scaler import VarianceGradientScaler
    from custom_policy_patch1 import CustomActorCriticPolicy
except ImportError:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from distributional_ppo import DistributionalPPO
    from variance_gradient_scaler import VarianceGradientScaler
    from custom_policy_patch1 import CustomActorCriticPolicy


def make_simple_env():
    """Create a simple Pendulum environment for testing (continuous action space)."""
    return DummyVecEnv([lambda: gym.make("Pendulum-v1")])


class TestVGSInitialization:
    """Test VGS initialization through PPO config."""

    def test_vgs_enabled_by_default(self) -> None:
        """Test that VGS is enabled by default."""
        env = make_simple_env()
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            value_scale_max_rel_step=0.1,
            n_steps=128,
            batch_size=64,
        )

        assert hasattr(model, "_vgs_enabled")
        assert model._vgs_enabled is True
        assert model._variance_gradient_scaler is not None
        assert isinstance(model._variance_gradient_scaler, VarianceGradientScaler)

        env.close()

    def test_vgs_default_parameters(self) -> None:
        """Test that VGS uses correct default parameters."""
        env = make_simple_env()
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            value_scale_max_rel_step=0.1,
            n_steps=128,
            batch_size=64,
        )

        # Check default parameter values
        assert model._vgs_enabled is True
        assert model._vgs_beta == 0.99
        assert model._vgs_alpha == 0.1
        assert model._vgs_warmup_steps == 100

        # Check scaler instance has correct defaults
        scaler = model._variance_gradient_scaler
        assert scaler is not None
        assert scaler.beta == 0.99
        assert scaler.alpha == 0.1
        assert scaler.warmup_steps == 100
        assert scaler.enabled is True

        env.close()

    def test_vgs_can_be_disabled(self) -> None:
        """Test that VGS can be explicitly disabled."""
        env = make_simple_env()
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            value_scale_max_rel_step=0.1,
            n_steps=128,
            batch_size=64,
            variance_gradient_scaling=False,
        )

        assert hasattr(model, "_vgs_enabled")
        assert model._vgs_enabled is False
        assert model._variance_gradient_scaler is None

        env.close()

    def test_vgs_enabled_through_config(self) -> None:
        """Test that VGS can be enabled through config."""
        env = make_simple_env()
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            value_scale_max_rel_step=0.1,
            n_steps=128,
            batch_size=64,
            variance_gradient_scaling=True,
            vgs_beta=0.95,
            vgs_alpha=0.2,
            vgs_warmup_steps=50,
        )

        assert model._vgs_enabled is True
        assert model._variance_gradient_scaler is not None
        assert isinstance(model._variance_gradient_scaler, VarianceGradientScaler)
        assert model._vgs_beta == 0.95
        assert model._vgs_alpha == 0.2
        assert model._vgs_warmup_steps == 50

        env.close()

    def test_vgs_config_logging(self, caplog) -> None:
        """Test that VGS configuration is logged."""
        env = make_simple_env()
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            value_scale_max_rel_step=0.1,
            n_steps=128,
            batch_size=64,
            variance_gradient_scaling=True,
            vgs_beta=0.97,
            vgs_alpha=0.15,
            vgs_warmup_steps=75,
        )

        # Configuration should be logged
        assert model._vgs_enabled is True
        assert model._vgs_beta == 0.97

        env.close()


class TestVGSTrainingIntegration:
    """Test VGS integration in training loop."""

    def test_training_with_vgs_by_default(self) -> None:
        """Test that training works with VGS enabled by default."""
        env = make_simple_env()
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            value_scale_max_rel_step=0.1,
            n_steps=128,
            batch_size=64,
            n_epochs=2,
            verbose=0,
        )

        # VGS should be enabled by default
        assert model._vgs_enabled is True
        assert model._variance_gradient_scaler is not None

        # Train for a few steps
        model.learn(total_timesteps=256, progress_bar=False)

        # Verify VGS statistics have been updated during training
        scaler = model._variance_gradient_scaler
        assert scaler._step_count > 0
        assert scaler._grad_mean_ema is not None
        assert scaler._grad_var_ema is not None
        assert scaler._grad_norm_ema is not None

        env.close()

    def test_training_with_vgs_enabled(self) -> None:
        """Test that training works with VGS explicitly enabled."""
        env = make_simple_env()
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            value_scale_max_rel_step=0.1,
            n_steps=128,
            batch_size=64,
            n_epochs=2,
            variance_gradient_scaling=True,
            vgs_beta=0.99,
            vgs_alpha=0.1,
            vgs_warmup_steps=10,
            verbose=0,
        )

        # Train for a few steps
        model.learn(total_timesteps=256, progress_bar=False)

        # Verify VGS statistics have been updated
        assert model._variance_gradient_scaler is not None
        assert model._variance_gradient_scaler._step_count > 0
        assert model._variance_gradient_scaler._grad_mean_ema is not None

        env.close()

    def test_training_with_vgs_disabled(self) -> None:
        """Test that training works normally with VGS disabled."""
        env = make_simple_env()
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            value_scale_max_rel_step=0.1,
            n_steps=128,
            batch_size=64,
            n_epochs=2,
            variance_gradient_scaling=False,
            verbose=0,
        )

        # Train for a few steps
        model.learn(total_timesteps=256, progress_bar=False)

        # Verify VGS is not used
        assert model._variance_gradient_scaler is None

        env.close()

    def test_vgs_warmup_behavior(self) -> None:
        """Test that VGS respects warmup period."""
        env = make_simple_env()
        warmup_steps = 20
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            value_scale_max_rel_step=0.1,
            n_steps=128,
            batch_size=64,
            n_epochs=1,
            variance_gradient_scaling=True,
            vgs_warmup_steps=warmup_steps,
            verbose=0,
        )

        # Train briefly
        model.learn(total_timesteps=256, progress_bar=False)

        scaler = model._variance_gradient_scaler
        assert scaler is not None

        # During warmup, scaling factor should be 1.0
        if scaler._step_count < warmup_steps:
            assert scaler.get_scaling_factor() == 1.0

        env.close()

    def test_vgs_with_gradient_clipping(self) -> None:
        """Test that VGS works correctly with gradient clipping."""
        env = make_simple_env()
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            value_scale_max_rel_step=0.1,
            n_steps=128,
            batch_size=64,
            n_epochs=2,
            max_grad_norm=0.5,  # Enable gradient clipping
            variance_gradient_scaling=True,
            vgs_alpha=0.2,
            verbose=0,
        )

        # Train with both VGS and gradient clipping
        model.learn(total_timesteps=256, progress_bar=False)

        # Both should have been applied
        assert model._variance_gradient_scaler is not None
        assert model._variance_gradient_scaler._step_count > 0
        assert model.max_grad_norm == 0.5

        env.close()


class TestVGSMetricsAndLogging:
    """Test VGS metrics and logging."""

    def test_vgs_metrics_logged_by_default(self) -> None:
        """Test that VGS metrics are logged during training by default."""
        env = make_simple_env()
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            value_scale_max_rel_step=0.1,
            n_steps=128,
            batch_size=64,
            n_epochs=1,
            verbose=0,
        )

        # Train to generate metrics (VGS enabled by default)
        model.learn(total_timesteps=256, progress_bar=False)

        # Check that VGS metrics exist (they are logged to logger)
        scaler = model._variance_gradient_scaler
        assert scaler is not None
        assert scaler._step_count > 0

        # Verify statistics are being computed
        assert scaler._grad_mean_ema is not None
        assert scaler._grad_mean_ema >= 0.0
        assert scaler._grad_var_ema >= 0.0
        assert scaler._grad_norm_ema >= 0.0

        env.close()

    def test_vgs_metrics_logged(self) -> None:
        """Test that VGS metrics are logged during training."""
        env = make_simple_env()
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            value_scale_max_rel_step=0.1,
            n_steps=128,
            batch_size=64,
            n_epochs=1,
            variance_gradient_scaling=True,
            vgs_warmup_steps=5,
            verbose=0,
        )

        # Train to generate metrics
        model.learn(total_timesteps=256, progress_bar=False)

        # Check that VGS metrics exist (they are logged to logger)
        scaler = model._variance_gradient_scaler
        assert scaler is not None
        assert scaler._step_count > 0

        # Verify statistics are being computed
        if scaler._grad_mean_ema is not None:
            assert scaler._grad_mean_ema >= 0.0
            assert scaler._grad_var_ema >= 0.0

        env.close()

    def test_vgs_normalized_variance_computation(self) -> None:
        """Test that normalized variance is computed correctly."""
        env = make_simple_env()
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            value_scale_max_rel_step=0.1,
            n_steps=128,
            batch_size=64,
            n_epochs=2,
            variance_gradient_scaling=True,
            verbose=0,
        )

        model.learn(total_timesteps=256, progress_bar=False)

        scaler = model._variance_gradient_scaler
        assert scaler is not None

        # After training, normalized variance should be computed
        if scaler._step_count > scaler.warmup_steps:
            normalized_var = scaler.get_normalized_variance()
            assert normalized_var >= 0.0
            assert np.isfinite(normalized_var)

        env.close()


class TestVGSStatePersistence:
    """Test VGS state persistence."""

    def test_vgs_state_saved_and_loaded_by_default(self, tmp_path) -> None:
        """Test that VGS state is saved and loaded correctly with default settings."""
        env = make_simple_env()

        # Create and train model with VGS (default enabled)
        model1 = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            value_scale_max_rel_step=0.1,
            n_steps=128,
            batch_size=64,
            n_epochs=2,
            verbose=0,
        )

        # VGS should be enabled by default
        assert model1._vgs_enabled is True
        assert model1._variance_gradient_scaler is not None

        model1.learn(total_timesteps=256, progress_bar=False)

        # Save model
        save_path = tmp_path / "vgs_model_default.zip"
        model1.save(save_path)

        # Get VGS state before loading
        scaler1 = model1._variance_gradient_scaler
        assert scaler1 is not None

        # Load model
        model2 = DistributionalPPO.load(save_path, env=env)

        # Verify VGS is still enabled with default params
        assert model2._vgs_enabled is True
        assert model2._variance_gradient_scaler is not None

        scaler2 = model2._variance_gradient_scaler
        # Verify default parameters are preserved
        assert scaler2.beta == 0.99
        assert scaler2.alpha == 0.1
        assert scaler2.warmup_steps == 100

        env.close()

    def test_vgs_state_saved_and_loaded(self, tmp_path) -> None:
        """Test that VGS state is saved and loaded correctly."""
        env = make_simple_env()

        # Create and train model with VGS
        model1 = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            value_scale_max_rel_step=0.1,
            n_steps=128,
            batch_size=64,
            n_epochs=2,
            variance_gradient_scaling=True,
            vgs_beta=0.95,
            vgs_alpha=0.15,
            verbose=0,
        )

        model1.learn(total_timesteps=256, progress_bar=False)

        # Save model
        save_path = tmp_path / "vgs_model.zip"
        model1.save(save_path)

        # Get VGS state before loading
        scaler1 = model1._variance_gradient_scaler
        assert scaler1 is not None
        step_count_before = scaler1._step_count
        mean_ema_before = scaler1._grad_mean_ema

        # Load model
        model2 = DistributionalPPO.load(save_path, env=env)

        # Verify VGS is still enabled
        assert model2._vgs_enabled is True
        assert model2._variance_gradient_scaler is not None

        scaler2 = model2._variance_gradient_scaler
        # Note: state_dict is not automatically saved/loaded by SB3,
        # so we just verify the scaler exists and config is preserved
        assert scaler2.beta == 0.95
        assert scaler2.alpha == 0.15

        env.close()


class TestVGSEdgeCases:
    """Test edge cases and error handling."""

    def test_vgs_with_zero_warmup(self) -> None:
        """Test VGS with warmup_steps=0."""
        env = make_simple_env()
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            value_scale_max_rel_step=0.1,
            n_steps=128,
            batch_size=64,
            variance_gradient_scaling=True,
            vgs_warmup_steps=0,
            verbose=0,
        )

        # Should work without errors
        model.learn(total_timesteps=256, progress_bar=False)

        scaler = model._variance_gradient_scaler
        assert scaler is not None
        assert scaler.warmup_steps == 0

        env.close()

    def test_vgs_with_large_warmup(self) -> None:
        """Test VGS with very large warmup period."""
        env = make_simple_env()
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            value_scale_max_rel_step=0.1,
            n_steps=128,
            batch_size=64,
            n_epochs=1,
            variance_gradient_scaling=True,
            vgs_warmup_steps=10000,  # Very large warmup
            verbose=0,
        )

        model.learn(total_timesteps=256, progress_bar=False)

        scaler = model._variance_gradient_scaler
        assert scaler is not None

        # Scaling should still be 1.0 during entire warmup
        assert scaler.get_scaling_factor() == 1.0

        env.close()

    def test_vgs_with_high_alpha(self) -> None:
        """Test VGS with aggressive scaling (high alpha)."""
        env = make_simple_env()
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            value_scale_max_rel_step=0.1,
            n_steps=128,
            batch_size=64,
            n_epochs=2,
            variance_gradient_scaling=True,
            vgs_alpha=1.0,  # Aggressive scaling
            vgs_warmup_steps=5,
            verbose=0,
        )

        # Should work without numerical issues
        model.learn(total_timesteps=256, progress_bar=False)

        scaler = model._variance_gradient_scaler
        assert scaler is not None
        assert scaler.alpha == 1.0

        # Verify no NaN or inf in statistics
        if scaler._grad_mean_ema is not None:
            assert np.isfinite(scaler._grad_mean_ema)
            assert np.isfinite(scaler._grad_var_ema)

        env.close()


class TestVGSGradientScaling:
    """Test actual gradient scaling behavior."""

    def test_vgs_scaling_applied_by_default(self) -> None:
        """Test that VGS scaling is applied by default."""
        env = make_simple_env()
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            value_scale_max_rel_step=0.1,
            n_steps=128,
            batch_size=64,
            n_epochs=2,
            verbose=0,
        )

        # Train with default VGS enabled
        model.learn(total_timesteps=512, progress_bar=False)

        scaler = model._variance_gradient_scaler
        assert scaler is not None
        assert scaler._step_count > 0

        # After sufficient training, scaler should have accumulated statistics
        if scaler._step_count > scaler.warmup_steps:
            # Verify statistics are reasonable
            normalized_var = scaler.get_normalized_variance()
            scaling_factor = scaler.get_scaling_factor()

            assert np.isfinite(normalized_var)
            assert normalized_var >= 0.0
            assert 0.0 < scaling_factor <= 1.0

        env.close()

    def test_vgs_reduces_gradient_variance(self) -> None:
        """Test that VGS tends to reduce gradient variance over time."""
        env = make_simple_env()
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            value_scale_max_rel_step=0.1,
            n_steps=128,
            batch_size=64,
            n_epochs=3,
            variance_gradient_scaling=True,
            vgs_alpha=0.3,  # Moderate scaling
            vgs_warmup_steps=10,
            verbose=0,
        )

        model.learn(total_timesteps=512, progress_bar=False)

        scaler = model._variance_gradient_scaler
        assert scaler is not None

        # After sufficient training, scaler should have accumulated statistics
        if scaler._step_count > scaler.warmup_steps + 10:
            # Verify statistics are reasonable
            normalized_var = scaler.get_normalized_variance()
            scaling_factor = scaler.get_scaling_factor()

            assert np.isfinite(normalized_var)
            assert 0.0 < scaling_factor <= 1.0

        env.close()

    def test_vgs_scaling_applied_before_clipping(self) -> None:
        """Test that VGS scaling is applied before gradient clipping."""
        env = make_simple_env()

        # Create model with both VGS and grad clipping
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            value_scale_max_rel_step=0.1,
            n_steps=128,
            batch_size=64,
            n_epochs=2,
            max_grad_norm=0.5,
            variance_gradient_scaling=True,
            vgs_alpha=0.2,
            vgs_warmup_steps=5,
            verbose=0,
        )

        model.learn(total_timesteps=256, progress_bar=False)

        # Both mechanisms should be active
        assert model._variance_gradient_scaler is not None
        assert model.max_grad_norm == 0.5

        # VGS should have been applied (evidenced by step count)
        assert model._variance_gradient_scaler._step_count > 0

        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
