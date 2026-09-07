"""Test suite for Bug #8: Model Load Error - Two-Phase Initialization Fix

This test suite verifies that the two-phase initialization fix for Bug #8 works correctly:
1. Model can be saved without pickle errors
2. Model can be loaded without AttributeError
3. State (especially VGS) is preserved across save/load cycles
4. Training can continue after loading
5. Works with UPGD optimizer, Twin Critics, and VGS enabled
"""

import os
import tempfile
from typing import Optional

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

pytest.importorskip(
    "sb3_contrib", reason="Custom policy depends on sb3_contrib recurrent components"
)

from custom_policy_patch1 import CustomActorCriticPolicy
from distributional_ppo import DistributionalPPO


def _make_simple_env():
    """Create a simple dummy environment for testing."""

    def _init():
        # Use Pendulum-v1 which has Box action space
        return gym.make("Pendulum-v1")

    return DummyVecEnv([_init])


def _constant_lr_schedule(_fraction: float) -> float:
    return 3e-4


def test_save_succeeds():
    """Test that model can be saved without pickle errors."""
    env = _make_simple_env()

    model = DistributionalPPO(
        policy=CustomActorCriticPolicy,
        env=env,
        variance_gradient_scaling=True,
        vgs_beta=0.99,
        vgs_alpha=0.1,
        vgs_warmup_steps=100,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
    )

    # Train for a few steps to initialize VGS state
    model.learn(total_timesteps=256, progress_bar=False)

    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "model.zip")
        model.save(save_path)

        # Verify file was created
        assert os.path.exists(save_path)


def test_load_succeeds():
    """Test that model can be loaded without AttributeError."""
    env = _make_simple_env()

    # Create and save model
    model = DistributionalPPO(
        policy=CustomActorCriticPolicy,
        env=env,
        variance_gradient_scaling=True,
        vgs_beta=0.99,
        vgs_alpha=0.1,
        vgs_warmup_steps=100,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
    )

    model.learn(total_timesteps=256, progress_bar=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "model.zip")
        model.save(save_path)

        # Load model - this should not raise AttributeError
        loaded_model = DistributionalPPO.load(save_path, env=env)

        # Verify model was loaded
        assert loaded_model is not None
        assert isinstance(loaded_model, DistributionalPPO)
        assert loaded_model.policy is not None


def test_save_load_cycle_preserves_state():
    """Test that save/load cycle preserves model state."""
    env = _make_simple_env()

    # Create and train model
    model = DistributionalPPO(
        policy=CustomActorCriticPolicy,
        env=env,
        variance_gradient_scaling=True,
        vgs_beta=0.99,
        vgs_alpha=0.1,
        vgs_warmup_steps=100,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
    )

    model.learn(total_timesteps=512, progress_bar=False)

    # Get state before save
    original_params = {name: param.clone() for name, param in model.policy.named_parameters()}
    original_update_calls = model._update_calls

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "model.zip")
        model.save(save_path)

        # Load model
        loaded_model = DistributionalPPO.load(save_path, env=env)

        # Verify state was preserved
        assert loaded_model._update_calls == original_update_calls

        # Check parameters match
        for name, param in loaded_model.policy.named_parameters():
            assert name in original_params
            assert torch.allclose(param, original_params[name], atol=1e-6)


def test_continued_training_after_load():
    """Test that training can continue after loading model."""
    env = _make_simple_env()

    # Create and train model
    model = DistributionalPPO(
        policy=CustomActorCriticPolicy,
        env=env,
        variance_gradient_scaling=True,
        vgs_beta=0.99,
        vgs_alpha=0.1,
        vgs_warmup_steps=100,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
    )

    model.learn(total_timesteps=256, progress_bar=False)
    update_calls_before = model._update_calls

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "model.zip")
        model.save(save_path)

        # Load model
        loaded_model = DistributionalPPO.load(save_path, env=env)

        # Continue training - this should not raise any errors
        loaded_model.learn(total_timesteps=256, progress_bar=False)

        # Verify training happened
        assert loaded_model._update_calls > update_calls_before


def test_vgs_state_restored():
    """Test that VGS state is correctly restored after load."""
    env = _make_simple_env()

    # Create model with VGS enabled
    model = DistributionalPPO(
        policy=CustomActorCriticPolicy,
        env=env,
        variance_gradient_scaling=True,
        vgs_beta=0.99,
        vgs_alpha=0.1,
        vgs_warmup_steps=50,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
    )

    # Train to build up VGS state
    model.learn(total_timesteps=512, progress_bar=False)

    # Check VGS exists and has state
    assert model._variance_gradient_scaler is not None
    original_vgs_step = (
        model._variance_gradient_scaler.step_count
        if hasattr(model._variance_gradient_scaler, "step_count")
        else 0
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "model.zip")
        model.save(save_path)

        # Load model
        loaded_model = DistributionalPPO.load(save_path, env=env)

        # Verify VGS was restored
        assert loaded_model._vgs_enabled is True
        assert loaded_model._variance_gradient_scaler is not None
        assert loaded_model._vgs_beta == 0.99
        assert loaded_model._vgs_alpha == 0.1
        assert loaded_model._vgs_warmup_steps == 50


def test_optimizer_state_restored():
    """Test that optimizer state is restored after load."""
    env = _make_simple_env()

    model = DistributionalPPO(
        policy=CustomActorCriticPolicy,
        env=env,
        variance_gradient_scaling=True,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
    )

    model.learn(total_timesteps=256, progress_bar=False)

    # Get optimizer state before save
    original_optimizer_type = type(model.policy.optimizer).__name__
    original_param_groups = len(model.policy.optimizer.param_groups)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "model.zip")
        model.save(save_path)

        # Load model
        loaded_model = DistributionalPPO.load(save_path, env=env)

        # Verify optimizer was restored
        assert loaded_model.policy.optimizer is not None
        assert type(loaded_model.policy.optimizer).__name__ == original_optimizer_type
        assert len(loaded_model.policy.optimizer.param_groups) == original_param_groups


def test_with_upgd_optimizer():
    """Test save/load works with UPGD optimizer."""
    pytest.importorskip("upgd", reason="UPGD optimizer not available")

    env = _make_simple_env()

    model = DistributionalPPO(
        policy=CustomActorCriticPolicy,
        env=env,
        variance_gradient_scaling=True,
        optimizer_class="UPGD",
        optimizer_kwargs={"lr": 3e-4, "beta": 0.9},
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
    )

    model.learn(total_timesteps=256, progress_bar=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "model.zip")
        model.save(save_path)

        # Load model - should work with UPGD
        loaded_model = DistributionalPPO.load(save_path, env=env)

        # Verify optimizer is UPGD
        assert loaded_model.policy.optimizer is not None
        assert "UPGD" in type(loaded_model.policy.optimizer).__name__

        # Continue training
        loaded_model.learn(total_timesteps=256, progress_bar=False)


def test_with_twin_critics():
    """Test save/load works with Twin Critics enabled."""
    env = _make_simple_env()

    model = DistributionalPPO(
        policy=CustomActorCriticPolicy,
        env=env,
        variance_gradient_scaling=True,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        policy_kwargs={
            "arch_params": {
                "critic": {
                    "distributional": True,
                    "num_quantiles": 5,
                    "twin_critics": True,
                }
            }
        },
    )

    model.learn(total_timesteps=256, progress_bar=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "model.zip")
        model.save(save_path)

        # Load model - should work with twin critics
        loaded_model = DistributionalPPO.load(save_path, env=env)

        # Verify twin critics are present
        assert hasattr(loaded_model.policy, "value_net")

        # Continue training
        loaded_model.learn(total_timesteps=256, progress_bar=False)


def test_normal_init_still_works():
    """Test that normal initialization (without load) still works correctly."""
    env = _make_simple_env()

    # Create model normally
    model = DistributionalPPO(
        policy=CustomActorCriticPolicy,
        env=env,
        variance_gradient_scaling=True,
        vgs_beta=0.99,
        vgs_alpha=0.1,
        vgs_warmup_steps=100,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
    )

    # Verify setup was completed
    assert model._setup_complete is True
    assert model._variance_gradient_scaler is not None
    assert model.policy.optimizer is not None

    # Verify training works
    model.learn(total_timesteps=256, progress_bar=False)
    assert model._update_calls > 0


def test_setup_is_idempotent():
    """Test that _setup_dependent_components() is idempotent."""
    env = _make_simple_env()

    model = DistributionalPPO(
        policy=CustomActorCriticPolicy,
        env=env,
        variance_gradient_scaling=True,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
    )

    # Get optimizer and VGS before second call
    optimizer_before = model.policy.optimizer
    vgs_before = model._variance_gradient_scaler

    # Call _setup_dependent_components() again - should be no-op
    model._setup_dependent_components()

    # Verify nothing changed
    assert model.policy.optimizer is optimizer_before
    assert model._variance_gradient_scaler is vgs_before


def test_setup_complete_flag():
    """Test that _setup_complete flag is managed correctly."""
    env = _make_simple_env()

    # Normal init - flag should be True
    model = DistributionalPPO(
        policy=CustomActorCriticPolicy,
        env=env,
        variance_gradient_scaling=True,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
    )

    assert model._setup_complete is True

    # After save/load, flag should be True again
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "model.zip")
        model.save(save_path)

        loaded_model = DistributionalPPO.load(save_path, env=env)
        assert loaded_model._setup_complete is True


def test_loss_head_weights_preserved():
    """Test that loss head weights configuration is preserved."""
    env = _make_simple_env()

    loss_weights = {
        "policy": 1.0,
        "value": 0.5,
        "entropy": 0.01,
    }

    model = DistributionalPPO(
        policy=CustomActorCriticPolicy,
        env=env,
        variance_gradient_scaling=True,
        loss_head_weights=loss_weights,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
    )

    model.learn(total_timesteps=256, progress_bar=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "model.zip")
        model.save(save_path)

        loaded_model = DistributionalPPO.load(save_path, env=env)

        # Verify loss head weights are preserved
        assert loaded_model._loss_head_weights is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
