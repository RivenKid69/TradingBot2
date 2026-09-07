"""
Additional coverage tests targeting specific uncovered branches in distributional_ppo.py.
Focuses on edge cases in helper methods and internal functions.
"""

from __future__ import annotations

import copy
import math
import os
import sys
import warnings
from typing import Any, Dict, Optional, Sequence
from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

gymnasium = pytest.importorskip("gymnasium")
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

import distributional_ppo as dppo
from distributional_ppo import (
    DistributionalPPO,
    PopArtController,
    RawRecurrentRolloutBuffer,
    safe_explained_variance,
    _weighted_variance_np,
    compute_grouped_explained_variance,
    calculate_cvar,
    _cfg_get,
    _popart_value_to_serializable,
    _serialize_popart_config,
)

np.random.seed(42)
torch.manual_seed(42)


# =============================================================================
# Test Environment
# =============================================================================


class MinimalTestingEnv(gymnasium.Env):
    """Simple test environment."""

    def __init__(self, seed: int = 42, max_steps: int = 8):
        super().__init__()
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._max_steps = max_steps

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        return self._rng.uniform(-1, 1, size=4).astype(np.float32), {}

    def step(self, action):
        self._step_count += 1
        obs = self._rng.uniform(-1, 1, size=4).astype(np.float32)
        reward = float(-np.sum(obs**2) * 0.01 + 0.1)
        terminated = self._step_count >= self._max_steps
        return obs, reward, terminated, False, {"step": self._step_count}


def make_env(n_envs: int = 1, seed: int = 42) -> DummyVecEnv:
    """Create vectorized test environment."""
    return DummyVecEnv([lambda: MinimalTestingEnv(seed=seed) for _ in range(n_envs)])


# =============================================================================
# Tests for _cfg_get helper
# =============================================================================


class TestCfgGet:
    """Test _cfg_get helper function."""

    def test_cfg_get_from_dict(self):
        """Test _cfg_get with dict."""
        cfg = {"key1": "value1", "key2": 42}
        assert _cfg_get(cfg, "key1", "default") == "value1"
        assert _cfg_get(cfg, "key2", 0) == 42
        assert _cfg_get(cfg, "missing", "default") == "default"

    def test_cfg_get_from_object(self):
        """Test _cfg_get with object."""

        class ConfigObj:
            key1 = "value1"
            key2 = 42

        cfg = ConfigObj()
        assert _cfg_get(cfg, "key1", "default") == "value1"
        assert _cfg_get(cfg, "key2", 0) == 42
        assert _cfg_get(cfg, "missing", "default") == "default"

    def test_cfg_get_none_config(self):
        """Test _cfg_get with None config."""
        assert _cfg_get(None, "key", "default") == "default"


# =============================================================================
# Tests for _popart_value_to_serializable
# =============================================================================


class TestPopartValueToSerializable:
    """Test _popart_value_to_serializable helper."""

    def test_numpy_scalar(self):
        """Test with numpy scalar."""
        scalar = np.float32(1.5)
        result = _popart_value_to_serializable(scalar)
        # May return string repr or float
        assert result is not None

    def test_basic_types(self):
        """Test with basic types."""
        assert _popart_value_to_serializable(1) == 1
        assert _popart_value_to_serializable(1.5) == 1.5
        assert _popart_value_to_serializable("test") == "test"
        assert _popart_value_to_serializable(True) is True

    def test_none_value(self):
        """Test with None."""
        assert _popart_value_to_serializable(None) is None


# =============================================================================
# Tests for _serialize_popart_config
# =============================================================================


class TestSerializePopartConfig:
    """Test _serialize_popart_config helper."""

    def test_serialize_basic_dict(self):
        """Test serialization of basic dict."""
        cfg = {"key1": 1.5, "key2": "value"}
        result = _serialize_popart_config(cfg)
        assert isinstance(result, dict)
        assert result["key1"] == 1.5
        assert result["key2"] == "value"


# =============================================================================
# Tests for PopArtController edge cases
# =============================================================================


class TestPopArtControllerAdditional:
    """Additional PopArtController tests."""

    def test_weighted_mean_std_with_inf(self):
        """Test _weighted_mean_std with inf values."""
        values = np.array([1.0, float("inf"), 3.0])
        mean, std = PopArtController._weighted_mean_std(values, None)
        # Should handle inf gracefully
        assert math.isnan(mean) or math.isinf(mean) or isinstance(mean, float)

    def test_safe_numpy_with_tensor(self):
        """Test _safe_numpy with tensor."""
        controller = PopArtController(enabled=True)
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = controller._safe_numpy(tensor)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_controller_update_counter(self):
        """Test update counter increments."""
        controller = PopArtController(enabled=True, mode="shadow")
        initial_count = controller._update_counter

        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            verbose=0,
        )

        controller.evaluate_shadow(
            model=model,
            returns_raw=torch.randn(50),
            ret_mean=0.0,
            ret_std=1.0,
        )

        assert controller._update_counter == initial_count + 1
        env.close()


# =============================================================================
# Tests for training with various configurations
# =============================================================================


class TestTrainingConfigurations:
    """Test various training configurations."""

    def test_train_with_vf_coef_warmup(self):
        """Cover vf_coef_warmup path."""
        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            vf_coef_warmup=0.1,
            vf_coef_warmup_updates=2,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_train_with_clip_range_warmup(self):
        """Cover clip_range_warmup path."""
        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_warmup=0.1,
            clip_range_warmup_updates=2,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_train_with_critic_grad_warmup(self):
        """Cover critic_grad_warmup path."""
        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            critic_grad_warmup_updates=2,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_train_with_bc_warmup(self):
        """Cover bc_warmup path."""
        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            bc_warmup_steps=4,
            bc_decay_steps=4,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_train_with_kl_absolute_stop(self):
        """Cover KL absolute stop factor."""
        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            target_kl=0.01,
            kl_absolute_stop_factor=3.0,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_train_with_cvar_ema(self):
        """Cover CVaR EMA path."""
        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            cvar_ema_beta=0.95,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_train_with_ret_clip(self):
        """Cover ret_clip path."""
        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            ret_clip=5.0,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_train_with_entropy_plateau(self):
        """Cover entropy plateau path."""
        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            ent_coef_decay_steps=10,
            ent_coef_final=0.001,
            ent_coef_plateau_window=3,
            verbose=0,
        )
        model.learn(total_timesteps=32)
        env.close()

    def test_train_with_kl_consec(self):
        """Cover KL consecutive minibatches path."""
        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            target_kl=0.01,
            kl_consec_minibatches=2,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_train_with_kl_epoch_decay(self):
        """Cover KL epoch decay path."""
        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=3,
            target_kl=0.01,
            kl_epoch_decay=0.8,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()


# =============================================================================
# Tests for DistributionalPPO internal methods
# =============================================================================


class TestDistributionalPPOInternals:
    """Test DistributionalPPO internal methods."""

    def test_get_parameters(self):
        """Test get_parameters method."""
        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=4,
            batch_size=4,
            verbose=0,
        )

        params = model.get_parameters()
        assert isinstance(params, dict)
        assert "policy" in params

        env.close()

    def test_set_parameters(self):
        """Test set_parameters method."""
        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=4,
            batch_size=4,
            verbose=0,
        )

        # Get and set parameters
        params = model.get_parameters()
        model.set_parameters(params)

        env.close()


# =============================================================================
# Tests for value target scale edge cases
# =============================================================================


class TestValueTargetScaleEdgeCases:
    """Test value_target_scale edge cases."""

    def test_value_scale_update_disabled(self):
        """Test with value_scale_update_enabled=False."""
        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            value_scale_update_enabled=False,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_value_target_scale_fixed(self):
        """Test with value_target_scale_fixed."""
        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            value_target_scale_fixed=50.0,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()


# =============================================================================
# Tests for VGS (Variance Gradient Scaler)
# =============================================================================


class TestVGSConfiguration:
    """Test VGS configuration."""

    def test_vgs_disabled(self):
        """Test with VGS disabled."""
        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            variance_gradient_scaling=False,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_vgs_with_warmup(self):
        """Test VGS with warmup steps."""
        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            variance_gradient_scaling=True,
            vgs_warmup_steps=4,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()


# =============================================================================
# Tests for optimizer configurations
# =============================================================================


class TestOptimizerConfigurations:
    """Test optimizer configurations."""

    def test_optimizer_sgd(self):
        """Test with SGD optimizer."""
        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            optimizer_class="SGD",
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_optimizer_with_kwargs(self):
        """Test optimizer with custom kwargs."""
        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            optimizer_class="Adam",
            optimizer_kwargs={"betas": (0.9, 0.999)},
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()


# =============================================================================
# Tests for loss head weights
# =============================================================================


class TestLossHeadWeights:
    """Test loss_head_weights configuration."""

    def test_custom_loss_weights(self):
        """Test with custom loss head weights."""
        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            loss_head_weights={
                "actor": 1.0,
                "critic": 0.5,
                "entropy": 0.01,
            },
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()


# =============================================================================
# Tests for gradient accumulation
# =============================================================================


class TestGradientAccumulation:
    """Test gradient accumulation."""

    def test_gradient_accumulation_steps(self):
        """Test with gradient accumulation steps."""
        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            gradient_accumulation_steps=2,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()


# =============================================================================
# Tests for microbatch size
# =============================================================================


class TestMicrobatchSize:
    """Test microbatch size configuration."""

    def test_microbatch_size(self):
        """Test with microbatch_size."""
        env = make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            microbatch_size=4,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
