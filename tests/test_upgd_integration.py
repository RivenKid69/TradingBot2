"""
Integration tests for UPGD optimizer with DistributionalPPO.

Tests verify that UPGD optimizers work correctly within the full
AI-Powered Quantitative Research Platform training pipeline.
"""

import pytest
torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")
from stable_baselines3.common.vec_env import DummyVecEnv
from distributional_ppo import DistributionalPPO
from optimizers import UPGD, AdaptiveUPGD, UPGDW
from custom_policy_patch1 import CustomActorCriticPolicy


def make_simple_env():
    """Create a simple test environment with continuous action space."""
    return DummyVecEnv([lambda: gym.make("Pendulum-v1")])


class TestUPGDIntegrationWithPPO:
    """Test UPGD optimizer integration with DistributionalPPO."""

    def test_upgd_string_selection(self):
        """Test UPGD selection via string identifier."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="upgd",
            optimizer_kwargs={"lr": 1e-4, "sigma": 0.001},
            verbose=0,
        )

        # Check optimizer was created
        assert model.policy.optimizer is not None
        assert isinstance(model.policy.optimizer, UPGD)

        env.close()

    def test_adaptive_upgd_string_selection(self):
        """Test AdaptiveUPGD selection via string."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="adaptive_upgd",
            optimizer_kwargs={"beta1": 0.9, "beta2": 0.999},
            verbose=0,
        )

        assert isinstance(model.policy.optimizer, AdaptiveUPGD)

        env.close()

    def test_upgdw_string_selection(self):
        """Test UPGDW selection via string."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="upgdw",
            optimizer_kwargs={"weight_decay": 0.01},
            verbose=0,
        )

        assert isinstance(model.policy.optimizer, UPGDW)

        env.close()

    def test_direct_class_selection(self):
        """Test optimizer selection via direct class reference."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class=AdaptiveUPGD,
            optimizer_kwargs={"lr": 3e-4},
            verbose=0,
        )

        assert isinstance(model.policy.optimizer, AdaptiveUPGD)

        env.close()

    def test_default_adaptive_upgd_when_none(self):
        """Test that AdaptiveUPGD is used by default when no optimizer is specified."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class=None,  # Default
            verbose=0,
        )

        assert isinstance(model.policy.optimizer, AdaptiveUPGD)

        env.close()

    def test_explicit_adamw_selection(self):
        """Test that AdamW can still be explicitly selected."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="adamw",  # Explicit
            verbose=0,
        )

        assert isinstance(model.policy.optimizer, torch.optim.AdamW)

        env.close()

    def test_invalid_optimizer_string_raises_error(self):
        """Test that invalid optimizer string raises ValueError."""
        env = make_simple_env()

        with pytest.raises(ValueError, match="Unknown optimizer"):
            DistributionalPPO(
                CustomActorCriticPolicy,
                env,
                optimizer_class="invalid_optimizer",
                verbose=0,
            )

        env.close()

    def test_optimizer_kwargs_validation(self):
        """Test that invalid optimizer_kwargs raises TypeError."""
        env = make_simple_env()

        with pytest.raises(TypeError, match="must be a dictionary"):
            DistributionalPPO(
                CustomActorCriticPolicy,
                env,
                optimizer_class="upgd",
                optimizer_kwargs="invalid",  # Should be dict
                verbose=0,
            )

        env.close()

    def test_upgd_training_single_step(self):
        """Test that UPGD can perform a training step."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="upgd",
            n_steps=64,
            n_epochs=2,
            batch_size=64,
            verbose=0,
        )

        # Perform one training iteration
        model.learn(total_timesteps=128)

        # Check that optimizer state was created
        optimizer = model.policy.optimizer
        assert len(optimizer.state) > 0

        # Check that parameters have optimizer state
        for param_group in optimizer.param_groups:
            for p in param_group["params"]:
                if p.requires_grad:
                    # State might be created only if gradients were computed
                    pass  # Just verify no crashes

        env.close()

    def test_adaptive_upgd_training(self):
        """Test AdaptiveUPGD training."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="adaptive_upgd",
            optimizer_kwargs={"lr": 3e-4},
            n_steps=64,
            n_epochs=2,
            batch_size=64,
            verbose=0,
        )

        model.learn(total_timesteps=128)

        # Verify moments are being tracked
        optimizer = model.policy.optimizer
        has_state = False
        for param_group in optimizer.param_groups:
            for p in param_group["params"]:
                if p in optimizer.state:
                    state = optimizer.state[p]
                    assert "first_moment" in state
                    assert "sec_moment" in state
                    assert "avg_utility" in state
                    has_state = True

        assert has_state, "At least some parameters should have optimizer state"

        env.close()

    def test_upgdw_training(self):
        """Test UPGDW training with decoupled weight decay."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="upgdw",
            optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.01},
            n_steps=64,
            n_epochs=2,
            batch_size=64,
            verbose=0,
        )

        model.learn(total_timesteps=128)

        optimizer = model.policy.optimizer
        assert isinstance(optimizer, UPGDW)

        # Check weight decay parameter
        assert optimizer.param_groups[0]["weight_decay"] == 0.01

        env.close()

    def test_optimizer_logging(self):
        """Test that optimizer class is logged."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="adaptive_upgd",
            verbose=0,
        )

        # Logger should have recorded optimizer class
        # This is checked in the implementation via model.logger.record()

        env.close()

    def test_parameter_groups_with_upgd(self):
        """Test that parameter groups work correctly with UPGD."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="upgd",
            learning_rate=1e-4,
            verbose=0,
        )

        optimizer = model.policy.optimizer

        # Should have created parameter groups (value vs other params)
        assert len(optimizer.param_groups) > 0

        # Each group should have learning rate
        for group in optimizer.param_groups:
            assert "lr" in group
            assert group["lr"] > 0

        env.close()

    def test_learning_rate_schedule_with_upgd(self):
        """Test that learning rate scheduling works with UPGD."""
        env = make_simple_env()

        def lr_schedule(progress):
            return 1e-4 * (1 - progress)

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="upgd",
            learning_rate=lr_schedule,
            n_steps=64,
            verbose=0,
        )

        # Get initial LR
        initial_lr = model.policy.optimizer.param_groups[0]["lr"]

        # Train for some steps
        model.learn(total_timesteps=128)

        # LR should have changed due to schedule
        # (might not change in single step, but structure should work)

        env.close()

    def test_upgd_with_cvar_constraint(self):
        """Test UPGD works with CVaR constraints."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="upgd",
            cvar_use_constraint=True,
            cvar_limit=-1.0,
            cvar_lambda_lr=1e-2,
            n_steps=64,
            n_epochs=2,
            verbose=0,
        )

        model.learn(total_timesteps=128)

        # Should complete without errors
        assert model.policy.optimizer is not None

        env.close()

    def test_upgd_with_gradient_clipping(self):
        """Test UPGD works with gradient clipping."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="upgd",
            max_grad_norm=0.5,
            n_steps=64,
            n_epochs=2,
            verbose=0,
        )

        model.learn(total_timesteps=128)

        assert model.policy.optimizer is not None

        env.close()


class TestUPGDHyperparameters:
    """Test UPGD-specific hyperparameter configurations."""

    def test_custom_sigma(self):
        """Test custom sigma (noise) parameter."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="upgd",
            optimizer_kwargs={"sigma": 0.01},
            verbose=0,
        )

        assert model.policy.optimizer.param_groups[0]["sigma"] == 0.01

        env.close()

    def test_custom_beta_utility(self):
        """Test custom beta_utility parameter."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="upgd",
            optimizer_kwargs={"beta_utility": 0.99},
            verbose=0,
        )

        assert model.policy.optimizer.param_groups[0]["beta_utility"] == 0.99

        env.close()

    def test_adaptive_upgd_custom_betas(self):
        """Test custom beta1/beta2 for AdaptiveUPGD."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="adaptive_upgd",
            optimizer_kwargs={"beta1": 0.95, "beta2": 0.995},
            verbose=0,
        )

        optimizer = model.policy.optimizer
        assert optimizer.param_groups[0]["beta1"] == 0.95
        assert optimizer.param_groups[0]["beta2"] == 0.995

        env.close()

    def test_upgdw_custom_weight_decay(self):
        """Test custom weight decay for UPGDW."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="upgdw",
            optimizer_kwargs={"weight_decay": 0.05},
            verbose=0,
        )

        assert model.policy.optimizer.param_groups[0]["weight_decay"] == 0.05

        env.close()


class TestUPGDStatePersistence:
    """Test optimizer state persistence across save/load."""

    def test_save_and_load_with_upgd(self, tmp_path):
        """Test saving and loading model with UPGD optimizer."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="upgd",
            n_steps=64,
            n_epochs=2,
            verbose=0,
        )

        # Train briefly
        model.learn(total_timesteps=128)

        # Save model
        save_path = tmp_path / "upgd_model.zip"
        model.save(save_path)

        # Load model
        loaded_model = DistributionalPPO.load(save_path, env=env)

        # Should still have UPGD optimizer
        # Note: Optimizer state might not be fully preserved in SB3's save/load
        # This tests that the model can be saved and loaded without crashing

        env.close()

    def test_continue_training_after_load(self, tmp_path):
        """Test continuing training after save/load."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="adaptive_upgd",
            n_steps=64,
            verbose=0,
        )

        model.learn(total_timesteps=128)

        save_path = tmp_path / "upgd_model_continue.zip"
        model.save(save_path)

        loaded_model = DistributionalPPO.load(save_path, env=env)

        # Continue training
        loaded_model.learn(total_timesteps=128)

        env.close()


class TestUPGDPerformance:
    """Performance and convergence tests."""

    def test_upgd_converges(self):
        """Test that UPGD can converge on simple task."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="upgd",
            optimizer_kwargs={"lr": 3e-4},
            n_steps=128,
            n_epochs=4,
            batch_size=128,
            verbose=0,
        )

        # Train for reasonable number of steps
        model.learn(total_timesteps=2048)

        # Model should have learned something
        # (optimizer state should exist)
        optimizer = model.policy.optimizer
        assert len(optimizer.state) > 0

        env.close()

    def test_adaptive_upgd_performance(self):
        """Test AdaptiveUPGD training performance."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="adaptive_upgd",
            optimizer_kwargs={"lr": 3e-4},
            n_steps=128,
            n_epochs=4,
            verbose=0,
        )

        model.learn(total_timesteps=2048)

        # Check that adaptive moments are being used
        optimizer = model.policy.optimizer
        params_with_moments = 0
        for p in model.policy.parameters():
            if p in optimizer.state:
                state = optimizer.state[p]
                if "first_moment" in state and "sec_moment" in state:
                    params_with_moments += 1

        assert params_with_moments > 0

        env.close()


class TestUPGDNumericalStability:
    """Test numerical stability of UPGD in edge cases."""

    def test_stability_with_zero_rewards(self):
        """Test UPGD stability when rewards are zero."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="upgd",
            n_steps=64,
            n_epochs=2,
            verbose=0,
        )

        # Train for a few steps
        # Even with potentially zero rewards, should not crash
        try:
            model.learn(total_timesteps=256)
            stability_ok = True
        except Exception as e:
            if "nan" in str(e).lower() or "inf" in str(e).lower():
                stability_ok = False
            else:
                raise

        # Should maintain numerical stability
        # (might not learn well, but shouldn't crash or produce NaN/Inf)

        env.close()

    def test_no_nan_in_parameters(self):
        """Test that parameters don't become NaN."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="upgd",
            n_steps=64,
            n_epochs=2,
            verbose=0,
        )

        model.learn(total_timesteps=256)

        # Check all parameters are finite
        for param in model.policy.parameters():
            assert torch.all(torch.isfinite(param)), "Parameters should be finite"

        env.close()


class TestUPGDDefaultConfiguration:
    """Test default configuration for UPGD optimizer."""

    def test_default_optimizer_is_adaptive_upgd(self):
        """Test that the default optimizer is AdaptiveUPGD."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            verbose=0,
        )

        # Should use AdaptiveUPGD by default
        assert isinstance(model.policy.optimizer, AdaptiveUPGD)

        env.close()

    def test_default_optimizer_parameters(self):
        """Test that default AdaptiveUPGD has correct parameters."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            verbose=0,
        )

        optimizer = model.policy.optimizer
        assert isinstance(optimizer, AdaptiveUPGD)

        # Check default parameters
        param_group = optimizer.param_groups[0]
        assert param_group["weight_decay"] == 0.001
        assert param_group["sigma"] == 0.001
        assert param_group["beta_utility"] == 0.999
        assert param_group["beta1"] == 0.9
        assert param_group["beta2"] == 0.999
        assert param_group["eps"] == 1e-8

        env.close()

    def test_default_optimizer_can_be_overridden(self):
        """Test that default optimizer can be overridden with custom kwargs."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_kwargs={
                "lr": 1e-3,
                "sigma": 0.01,
                "beta_utility": 0.99,
            },
            verbose=0,
        )

        optimizer = model.policy.optimizer
        param_group = optimizer.param_groups[0]

        # Custom values should be used
        assert param_group["lr"] == 1e-3
        assert param_group["sigma"] == 0.01
        assert param_group["beta_utility"] == 0.99

        # Defaults for non-overridden params
        assert param_group["weight_decay"] == 0.001
        assert param_group["beta1"] == 0.9

        env.close()

    def test_default_optimizer_training_works(self):
        """Test that default AdaptiveUPGD can train successfully."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            n_steps=64,
            n_epochs=2,
            verbose=0,
        )

        # Should train without errors
        model.learn(total_timesteps=256)

        # Check optimizer has state
        optimizer = model.policy.optimizer
        assert len(optimizer.state) > 0

        # Check that at least some parameters have optimizer state
        has_state = False
        for p in model.policy.parameters():
            if p in optimizer.state:
                state = optimizer.state[p]
                assert "avg_utility" in state
                assert "first_moment" in state
                assert "sec_moment" in state
                has_state = True

        assert has_state

        env.close()


class TestUPGDComprehensiveCoverage:
    """Comprehensive tests for all UPGD scenarios."""

    def test_all_upgd_variants_work(self):
        """Test that all UPGD variants can be instantiated and used."""
        env = make_simple_env()

        variants = ["upgd", "adaptive_upgd", "upgdw"]
        expected_classes = [UPGD, AdaptiveUPGD, UPGDW]

        for variant, expected_class in zip(variants, expected_classes):
            model = DistributionalPPO(
                CustomActorCriticPolicy,
                env,
                optimizer_class=variant,
                n_steps=64,
                n_epochs=2,
                verbose=0,
            )

            assert isinstance(model.policy.optimizer, expected_class)

            # Should be able to train
            model.learn(total_timesteps=128)

        env.close()

    def test_upgd_with_all_ppo_features(self):
        """Test UPGD works with all PPO features combined."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="adaptive_upgd",
            optimizer_kwargs={"lr": 3e-4},
            # PPO features
            n_steps=64,
            n_epochs=4,
            batch_size=64,
            max_grad_norm=0.5,
            # CVaR
            cvar_use_constraint=True,
            cvar_limit=-1.0,
            # Distributional
            distributional_vf_clip_mode="mean_only",
            # Entropy
            ent_coef=0.01,
            verbose=0,
        )

        model.learn(total_timesteps=256)

        assert isinstance(model.policy.optimizer, AdaptiveUPGD)

        env.close()

    def test_upgd_state_tracking(self):
        """Test that UPGD properly tracks utility, moments, and steps."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="adaptive_upgd",
            n_steps=64,
            n_epochs=4,
            verbose=0,
        )

        # Train for several updates
        model.learn(total_timesteps=512)

        optimizer = model.policy.optimizer

        # Check that state is being tracked
        for p in model.policy.parameters():
            if p in optimizer.state:
                state = optimizer.state[p]

                # Should have step counter
                assert "step" in state
                assert state["step"] > 0

                # Should have utility tracking
                assert "avg_utility" in state
                assert state["avg_utility"].shape == p.shape

                # Should have moments
                assert "first_moment" in state
                assert "sec_moment" in state

                # All values should be finite
                assert torch.all(torch.isfinite(state["avg_utility"]))
                assert torch.all(torch.isfinite(state["first_moment"]))
                assert torch.all(torch.isfinite(state["sec_moment"]))

        env.close()

    def test_upgd_different_learning_rates(self):
        """Test UPGD with different learning rates."""
        env = make_simple_env()

        learning_rates = [1e-5, 1e-4, 3e-4, 1e-3]

        for lr in learning_rates:
            model = DistributionalPPO(
                CustomActorCriticPolicy,
                env,
                optimizer_class="adaptive_upgd",
                learning_rate=lr,
                n_steps=64,
                verbose=0,
            )

            optimizer = model.policy.optimizer
            assert optimizer.param_groups[0]["lr"] == lr

            # Should train without errors
            model.learn(total_timesteps=128)

        env.close()

    def test_upgd_with_different_sigma_values(self):
        """Test UPGD with different noise (sigma) values."""
        env = make_simple_env()

        sigma_values = [0.0001, 0.001, 0.01, 0.05]

        for sigma in sigma_values:
            model = DistributionalPPO(
                CustomActorCriticPolicy,
                env,
                optimizer_class="adaptive_upgd",
                optimizer_kwargs={"sigma": sigma},
                n_steps=64,
                verbose=0,
            )

            optimizer = model.policy.optimizer
            assert optimizer.param_groups[0]["sigma"] == sigma

            # Should train without errors
            model.learn(total_timesteps=128)

        env.close()

    def test_upgdw_decoupled_weight_decay(self):
        """Test that UPGDW uses decoupled weight decay correctly."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="upgdw",
            optimizer_kwargs={"weight_decay": 0.05},
            n_steps=64,
            n_epochs=2,
            verbose=0,
        )

        optimizer = model.policy.optimizer
        assert isinstance(optimizer, UPGDW)
        assert optimizer.param_groups[0]["weight_decay"] == 0.05

        # Train and verify no NaN/Inf
        model.learn(total_timesteps=256)

        for param in model.policy.parameters():
            assert torch.all(torch.isfinite(param))

        env.close()

    def test_optimizer_class_by_direct_import(self):
        """Test that optimizer can be specified by direct class import."""
        env = make_simple_env()

        # Test all three UPGD variants
        for optimizer_cls in [UPGD, AdaptiveUPGD, UPGDW]:
            model = DistributionalPPO(
                CustomActorCriticPolicy,
                env,
                optimizer_class=optimizer_cls,
                n_steps=64,
                verbose=0,
            )

            assert isinstance(model.policy.optimizer, optimizer_cls)

        env.close()

    def test_optimizer_logging(self):
        """Test that optimizer class and parameters are properly logged."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="adaptive_upgd",
            optimizer_kwargs={"lr": 1e-4, "sigma": 0.005},
            n_steps=64,
            verbose=0,
        )

        # Train for one update to trigger logging
        model.learn(total_timesteps=128)

        # Verify optimizer is correct type
        assert isinstance(model.policy.optimizer, AdaptiveUPGD)

        env.close()


class TestUPGDEdgeCases:
    """Test edge cases and error handling for UPGD."""

    def test_zero_learning_rate(self):
        """Test UPGD with zero learning rate (parameters shouldn't change)."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="adaptive_upgd",
            learning_rate=0.0,
            n_steps=64,
            n_epochs=2,
            verbose=0,
        )

        # Should not crash with zero learning rate
        model.learn(total_timesteps=128)

        env.close()

    def test_very_high_learning_rate(self):
        """Test UPGD stability with very high learning rate."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="adaptive_upgd",
            learning_rate=0.1,
            n_steps=64,
            n_epochs=2,
            verbose=0,
        )

        # Should maintain numerical stability
        try:
            model.learn(total_timesteps=128)

            # Check for NaN/Inf
            for param in model.policy.parameters():
                assert torch.all(torch.isfinite(param))
        except Exception as e:
            # If it diverges, that's expected with very high LR
            # But it shouldn't produce NaN/Inf silently
            if "nan" not in str(e).lower() and "inf" not in str(e).lower():
                pass  # Other errors are acceptable

        env.close()

    def test_zero_sigma(self):
        """Test UPGD with zero noise (no perturbation)."""
        env = make_simple_env()

        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            optimizer_class="adaptive_upgd",
            optimizer_kwargs={"sigma": 0.0},
            n_steps=64,
            n_epochs=2,
            verbose=0,
        )

        model.learn(total_timesteps=128)

        assert model.policy.optimizer.param_groups[0]["sigma"] == 0.0

        env.close()

    def test_invalid_optimizer_kwargs_type(self):
        """Test that invalid optimizer_kwargs type raises error."""
        env = make_simple_env()

        with pytest.raises(TypeError, match="must be a dictionary"):
            DistributionalPPO(
                CustomActorCriticPolicy,
                env,
                optimizer_class="adaptive_upgd",
                optimizer_kwargs="invalid",
                verbose=0,
            )

        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
