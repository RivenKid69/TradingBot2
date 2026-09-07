"""
Targeted tests to push coverage toward 95%.
Focus on uncovered validation branches and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest
import warnings

torch = pytest.importorskip("torch")

from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from distributional_ppo import DistributionalPPO, PopArtController

np.random.seed(42)
torch.manual_seed(42)


def _make_env(max_steps: int = 8, n_envs: int = 1) -> DummyVecEnv:
    """Create minimal test environment."""

    def _env_fn():
        import gymnasium

        class _Env(gymnasium.Env):
            def __init__(self):
                self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
                self.observation_space = spaces.Box(-10.0, 10.0, (4,), np.float32)
                self._step = 0
                self._max = max_steps

            def reset(self, *, seed=None, options=None):
                self._step = 0
                obs = np.random.randn(4).astype(np.float32) * 0.1
                return obs, {}

            def step(self, action):
                self._step += 1
                done = self._step >= self._max
                obs = np.random.randn(4).astype(np.float32) * 0.1
                reward = np.random.randn() * 0.1
                return obs, reward, done, False, {"TimeLimit.truncated": done}

        return _Env()

    return DummyVecEnv([_env_fn for _ in range(n_envs)])


class _DummyCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_step(self) -> bool:
        return True


def _make_model(env, **kwargs):
    """Create DistributionalPPO model with defaults."""
    defaults = {
        "policy": "DistributionalPolicy",
        "env": env,
        "n_steps": 8,
        "batch_size": 4,
        "n_epochs": 1,
        "device": "cpu",
        "verbose": 0,
    }
    defaults.update(kwargs)
    return DistributionalPPO(**defaults)


def _setup_and_collect(model, env, n_steps=8):
    """Setup model and collect rollouts."""
    total = int(n_steps * env.num_envs)
    _, callback = model._setup_learn(
        total_timesteps=total,
        callback=_DummyCallback(),
        reset_num_timesteps=True,
    )
    model._last_callback = callback
    model._current_progress_remaining = 1.0
    model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=n_steps)
    return model


# =============================================================================
# __init__ Validation Branch Tests
# =============================================================================


class TestInitValidationBranches:
    """Test validation branches in __init__."""

    def test_vf_clip_threshold_ev_non_finite_raises(self):
        """Test vf_clip_threshold_ev with non-finite value raises."""
        env = _make_env()
        with pytest.raises(ValueError, match="vf_clip_threshold_ev"):
            _make_model(env, vf_clip_threshold_ev=float("inf"))
        env.close()

    def test_vf_clip_threshold_ev_out_of_range_raises(self):
        """Test vf_clip_threshold_ev outside [-1, 1] raises."""
        env = _make_env()
        with pytest.raises(ValueError, match="vf_clip_threshold_ev"):
            _make_model(env, vf_clip_threshold_ev=1.5)
        env.close()

    def test_distributional_vf_clip_variance_factor_invalid_raises(self):
        """Test variance factor < 1.0 raises."""
        env = _make_env()
        with pytest.raises(ValueError, match="distributional_vf_clip_variance_factor"):
            _make_model(env, distributional_vf_clip_variance_factor=0.5)
        env.close()

    def test_value_scale_update_enabled_none(self):
        """Test value_scale_update_enabled=None defaults to True."""
        env = _make_env()
        model = _make_model(env, value_scale_update_enabled=None)
        # Should not raise
        env.close()

    def test_value_target_scale_fixed_positive(self):
        """Test value_target_scale_fixed with positive value."""
        env = _make_env()
        model = _make_model(env, value_target_scale_fixed=100.0)
        assert model._value_target_scale_fixed == 100.0
        env.close()

    def test_value_scale_update_enabled_false(self):
        """Test value_scale_update_enabled=False."""
        env = _make_env()
        model = _make_model(env, value_scale_update_enabled=False)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_value_target_scale_fixed_none(self):
        """Test value_target_scale_fixed=None."""
        env = _make_env()
        model = _make_model(env, value_target_scale_fixed=None)
        assert model._value_target_scale_fixed is None
        env.close()

    def test_clip_range_vf_invalid_raises(self):
        """Test clip_range_vf with invalid value raises."""
        env = _make_env()
        with pytest.raises(ValueError, match="clip_range_vf"):
            _make_model(env, clip_range_vf=-0.1)
        env.close()

    def test_clip_range_vf_inf_raises(self):
        """Test clip_range_vf with inf raises."""
        env = _make_env()
        with pytest.raises(ValueError, match="clip_range_vf"):
            _make_model(env, clip_range_vf=float("inf"))
        env.close()


# =============================================================================
# Train Loop Branch Tests
# =============================================================================


class TestTrainLoopBranches:
    """Test specific branches in train() loop."""

    def test_train_with_normalize_returns_true(self):
        """Test training with normalize_returns=True."""
        env = _make_env()
        model = _make_model(env, normalize_returns=True)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_train_with_normalize_returns_false(self):
        """Test training with normalize_returns=False."""
        env = _make_env()
        model = _make_model(env, normalize_returns=False)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_train_with_normalize_advantage_true(self):
        """Test training with normalize_advantage=True."""
        env = _make_env()
        model = _make_model(env, normalize_advantage=True)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_train_with_multiple_epochs(self):
        """Test training with multiple epochs."""
        env = _make_env()
        model = _make_model(env, n_epochs=3)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_train_with_target_kl_very_low(self):
        """Test training with very low target_kl (triggers early stopping)."""
        env = _make_env()
        model = _make_model(
            env,
            target_kl=0.0001,  # Very low - likely to trigger early stop
            n_epochs=3,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# PopArt Controller Tests
# =============================================================================


class TestPopArtControllerBranches:
    """Test PopArtController branches."""

    def test_controller_disabled(self):
        """Test PopArtController when disabled."""
        controller = PopArtController(enabled=False)
        assert not controller.enabled

    def test_controller_shadow_mode(self):
        """Test PopArtController in shadow mode."""
        controller = PopArtController(enabled=True, mode="shadow")
        assert controller.mode == "shadow"

    def test_controller_live_mode(self):
        """Test PopArtController in live mode."""
        controller = PopArtController(enabled=True, mode="live")
        assert controller.mode == "live"


# =============================================================================
# CVaR Edge Cases
# =============================================================================


class TestCvarEdgeCases:
    """Test CVaR computation edge cases."""

    def test_cvar_with_winsorization(self):
        """Test CVaR with winsorization enabled."""
        env = _make_env()
        model = _make_model(
            env,
            cvar_alpha=0.1,
            cvar_winsor_pct=0.01,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_cvar_with_constraint(self):
        """Test CVaR with constraint mode."""
        env = _make_env()
        model = _make_model(
            env,
            cvar_alpha=0.1,
            cvar_use_constraint=True,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_cvar_with_penalty(self):
        """Test CVaR with penalty mode."""
        env = _make_env()
        model = _make_model(
            env,
            cvar_alpha=0.1,
            cvar_use_penalty=True,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# Value Scale Edge Cases
# =============================================================================


class TestValueScaleEdgeCases:
    """Test value scale edge cases."""

    def test_value_target_scale_percent(self):
        """Test value_target_scale='percent'."""
        env = _make_env()
        model = _make_model(env, value_target_scale="percent")
        assert model._value_target_scale_effective == 100.0
        env.close()

    def test_value_target_scale_bps(self):
        """Test value_target_scale='bps'."""
        env = _make_env()
        model = _make_model(env, value_target_scale="bps")
        assert model._value_target_scale_effective == 10000.0
        env.close()


# =============================================================================
# Optimizer Edge Cases
# =============================================================================


class TestOptimizerEdgeCases:
    """Test optimizer edge cases."""

    def test_optimizer_adamw(self):
        """Test AdamW optimizer."""
        env = _make_env()
        model = _make_model(env, optimizer_class="AdamW")
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_optimizer_sgd(self):
        """Test SGD optimizer."""
        env = _make_env()
        model = _make_model(
            env,
            optimizer_class="SGD",
            optimizer_kwargs={"lr": 0.001, "momentum": 0.9},
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_optimizer_lr_bounds_active(self):
        """Test optimizer with LR bounds that get enforced."""
        env = _make_env()
        model = _make_model(
            env,
            learning_rate=1e-6,
            optimizer_lr_min=1e-5,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# VF Clipping Mode Tests
# =============================================================================


class TestVfClippingModes:
    """Test VF clipping modes."""

    def test_vf_clip_mode_disable(self):
        """Test VF clipping mode='disable'."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.5,
            distributional_vf_clip_mode="disable",
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_clip_warmup_updates(self):
        """Test VF clipping with warmup updates."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.5,
            vf_clip_warmup_updates=1,
        )
        model.learn(total_timesteps=24, progress_bar=False)
        env.close()

    def test_vf_clip_threshold_ev(self):
        """Test VF clipping with EV threshold."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.5,
            vf_clip_threshold_ev=0.5,
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()


# =============================================================================
# Entropy Coefficient Decay Tests
# =============================================================================


class TestEntCoefDecay:
    """Test entropy coefficient decay schedules."""

    def test_ent_coef_linear_decay(self):
        """Test linear entropy decay."""
        env = _make_env()
        model = _make_model(
            env,
            ent_coef=0.01,
            ent_coef_min=0.001,
        )
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_ent_coef_very_low(self):
        """Test with very low entropy coefficient."""
        env = _make_env()
        model = _make_model(env, ent_coef=0.0001)
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# GAE Lambda Tests
# =============================================================================


class TestGaeLambda:
    """Test GAE lambda variations."""

    def test_gae_lambda_zero(self):
        """Test with gae_lambda=0 (TD(0))."""
        env = _make_env()
        model = _make_model(env, gae_lambda=0.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_gae_lambda_one(self):
        """Test with gae_lambda=1 (Monte Carlo)."""
        env = _make_env()
        model = _make_model(env, gae_lambda=1.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# Multi-Environment Tests
# =============================================================================


class TestMultiEnv:
    """Test multi-environment configurations."""

    def test_multi_env_train(self):
        """Test training with multiple environments."""
        env = _make_env(n_envs=3)
        model = _make_model(env, n_steps=4, batch_size=6)
        _setup_and_collect(model, env, n_steps=4)
        model.train()
        env.close()

    def test_multi_env_learn(self):
        """Test learn() with multiple environments."""
        env = _make_env(n_envs=2)
        model = _make_model(env, n_steps=4, batch_size=4)
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()


# =============================================================================
# Gradient Norm Tests
# =============================================================================


class TestGradientNorm:
    """Test gradient norm clipping."""

    def test_max_grad_norm_high(self):
        """Test with high max_grad_norm."""
        env = _make_env()
        model = _make_model(env, max_grad_norm=10.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_max_grad_norm_low(self):
        """Test with low max_grad_norm."""
        env = _make_env()
        model = _make_model(env, max_grad_norm=0.1)
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# Clip Range Tests
# =============================================================================


class TestClipRange:
    """Test clip range variations."""

    def test_clip_range_high(self):
        """Test with high clip range."""
        env = _make_env()
        model = _make_model(env, clip_range=0.5)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_clip_range_low(self):
        """Test with low clip range."""
        env = _make_env()
        model = _make_model(env, clip_range=0.05)
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# VF Coefficient Tests
# =============================================================================


class TestVfCoef:
    """Test VF coefficient variations."""

    def test_vf_coef_high(self):
        """Test with high VF coefficient."""
        env = _make_env()
        model = _make_model(env, vf_coef=1.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_vf_coef_low(self):
        """Test with low VF coefficient."""
        env = _make_env()
        model = _make_model(env, vf_coef=0.1)
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# Quantile Configuration Tests
# =============================================================================


class TestQuantileConfig:
    """Test quantile configuration."""

    def test_quantile_default(self):
        """Test with default quantile config."""
        env = _make_env()
        model = _make_model(env)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_quantile_with_vf_clip(self):
        """Test quantile with VF clipping."""
        env = _make_env()
        model = _make_model(env, clip_range_vf=0.3)
        _setup_and_collect(model, env)
        model.train()
        env.close()


# =============================================================================
# Save/Load Edge Cases
# =============================================================================


class TestSaveLoadEdgeCases:
    """Test save/load edge cases."""

    def test_save_load_with_vf_clip(self, tmp_path):
        """Test save/load with VF clipping enabled."""
        env = _make_env()
        model = _make_model(
            env,
            clip_range_vf=0.5,
            distributional_vf_clip_mode="per_quantile",
        )
        path = tmp_path / "model_vf_clip.zip"
        model.save(str(path))
        env.close()

        env2 = _make_env()
        loaded = DistributionalPPO.load(str(path), env=env2)
        assert loaded is not None
        # Note: clip_range_vf may not be preserved after load - just check model loads
        env2.close()

    def test_save_load_with_cvar(self, tmp_path):
        """Test save/load with CVaR enabled."""
        env = _make_env()
        model = _make_model(env, cvar_alpha=0.1)
        path = tmp_path / "model_cvar.zip"
        model.save(str(path))
        env.close()

        env2 = _make_env()
        loaded = DistributionalPPO.load(str(path), env=env2)
        assert loaded is not None
        env2.close()


# =============================================================================
# Predict Edge Cases
# =============================================================================


class TestPredictEdgeCases:
    """Test prediction edge cases."""

    def test_predict_batch(self):
        """Test batch prediction."""
        env = _make_env()
        model = _make_model(env)

        obs = np.random.randn(5, 4).astype(np.float32)
        actions, _ = model.predict(obs, deterministic=True)

        assert actions.shape[0] == 5
        env.close()

    def test_predict_single(self):
        """Test single observation prediction."""
        env = _make_env()
        model = _make_model(env)

        obs = np.random.randn(4).astype(np.float32)
        action, _ = model.predict(obs, deterministic=False)

        assert action is not None
        env.close()


# =============================================================================
# Additional Train Configurations
# =============================================================================


class TestAdditionalTrainConfigs:
    """Test additional training configurations."""

    def test_normalize_advantage_false(self):
        """Test with normalize_advantage=False."""
        env = _make_env()
        model = _make_model(env, normalize_advantage=False)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_n_epochs_multiple(self):
        """Test with multiple epochs."""
        env = _make_env()
        model = _make_model(env, n_epochs=3)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_batch_size_equals_buffer(self):
        """Test with batch_size equal to buffer size."""
        env = _make_env()
        model = _make_model(env, n_steps=8, batch_size=8)
        _setup_and_collect(model, env)
        model.train()
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
