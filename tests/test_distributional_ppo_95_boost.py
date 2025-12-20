"""
Intensive coverage tests to boost line coverage to 95%+.
Focus on: train() branches, edge cases in value prediction, VGS, etc.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from distributional_ppo import DistributionalPPO

np.random.seed(42)
torch.manual_seed(42)


def _make_env(max_steps: int = 8, n_envs: int = 1) -> DummyVecEnv:
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


class TestTrainBranchesIntensive:
    """Intensive tests for train() branches."""

    def test_train_with_normalize_returns(self):
        """Cover normalize_returns path in train."""
        env = _make_env()
        model = _make_model(env, normalize_returns=True)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_train_with_clip_range_vf(self):
        """Cover clip_range_vf path."""
        env = _make_env()
        model = _make_model(env, clip_range_vf=0.2)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_train_with_target_kl(self):
        """Cover target_kl early stopping."""
        env = _make_env()
        model = _make_model(env, target_kl=0.001)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_train_with_ent_coef_decay(self):
        """Cover entropy coefficient decay."""
        env = _make_env()
        model = _make_model(env, ent_coef_decay_steps=5)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_train_multiple_epochs(self):
        """Cover multiple epochs."""
        env = _make_env()
        model = _make_model(env, n_epochs=3)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_train_with_gae_lambda(self):
        """Cover different GAE lambda."""
        env = _make_env()
        model = _make_model(env, gae_lambda=0.9)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_train_with_max_grad_norm(self):
        """Cover gradient clipping."""
        env = _make_env()
        model = _make_model(env, max_grad_norm=0.5)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_train_with_vf_coef(self):
        """Cover different vf_coef."""
        env = _make_env()
        model = _make_model(env, vf_coef=0.25)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_train_with_ent_coef(self):
        """Cover different ent_coef."""
        env = _make_env()
        model = _make_model(env, ent_coef=0.05)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestValueScaleBranches:
    """Test value scale configuration branches."""

    def test_value_scale_with_freeze(self):
        """Cover value scale freeze."""
        env = _make_env()
        model = _make_model(
            env,
            value_scale={"freeze_after_updates": 5},
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_value_scale_with_stability(self):
        """Cover value scale stability config."""
        env = _make_env()
        model = _make_model(
            env,
            value_scale={
                "stability": {
                    "min_explained_variance": 0.3,
                    "patience": 2,
                }
            },
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_value_scale_with_window(self):
        """Cover value scale window updates."""
        env = _make_env()
        model = _make_model(
            env,
            value_scale={"window_updates": 10},
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestCvarBranches:
    """Test CVaR configuration branches."""

    def test_cvar_constraint(self):
        """Cover CVaR constraint."""
        env = _make_env()
        model = _make_model(
            env,
            cvar_use_constraint=True,
            cvar_alpha=0.1,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_cvar_penalty(self):
        """Cover CVaR penalty."""
        env = _make_env()
        model = _make_model(
            env,
            cvar_use_penalty=True,
            cvar_alpha=0.1,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestMultiEnvTraining:
    """Test training with multiple environments."""

    def test_train_multi_env(self):
        """Cover multi-env training."""
        env = _make_env(n_envs=2)
        model = _make_model(env, n_steps=4, batch_size=4)
        _setup_and_collect(model, env, n_steps=4)
        model.train()
        env.close()


class TestLearnMethod:
    """Test learn() method."""

    def test_learn_short(self):
        """Cover learn() with short training."""
        env = _make_env()
        model = _make_model(env)
        model.learn(total_timesteps=16, progress_bar=False)
        env.close()

    def test_learn_with_callback(self):
        """Cover learn() with callback."""
        env = _make_env()
        model = _make_model(env)
        model.learn(total_timesteps=16, callback=_DummyCallback(), progress_bar=False)
        env.close()


class TestSaveLoadRoundtrip:
    """Test save/load functionality."""

    def test_save_load(self, tmp_path):
        """Cover save/load roundtrip."""
        env = _make_env()
        model = _make_model(env)
        path = tmp_path / "model.zip"
        model.save(str(path))
        env.close()

        env2 = _make_env()
        loaded = DistributionalPPO.load(str(path), env=env2)
        assert loaded is not None

        # Verify can predict
        obs = np.zeros((1, 4), dtype=np.float32)
        action, _ = loaded.predict(obs)
        assert action is not None
        env2.close()


class TestOptimizerConfigs:
    """Test different optimizer configurations."""

    def test_optimizer_adam(self):
        """Cover Adam optimizer."""
        env = _make_env()
        model = _make_model(env, optimizer_class="Adam")
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_optimizer_with_lr_bounds(self):
        """Cover optimizer with LR bounds."""
        env = _make_env()
        model = _make_model(
            env,
            optimizer_lr_min=1e-6,
            optimizer_lr_max=1e-2,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestValueTargetScale:
    """Test value_target_scale configurations."""

    def test_scale_percent(self):
        """Cover 'percent' scale."""
        env = _make_env()
        model = _make_model(env, value_target_scale="percent")
        assert model.value_target_scale == pytest.approx(100.0)
        env.close()

    def test_scale_bps(self):
        """Cover 'bps' scale."""
        env = _make_env()
        model = _make_model(env, value_target_scale="bps")
        assert model.value_target_scale == pytest.approx(10000.0)
        env.close()

    def test_scale_numeric(self):
        """Cover numeric scale."""
        env = _make_env()
        model = _make_model(env, value_target_scale=50.0)
        assert model.value_target_scale == pytest.approx(50.0)
        env.close()


class TestVfClipWarmup:
    """Test VF clip warmup."""

    def test_vf_clip_warmup(self):
        """Cover VF clip warmup path."""
        env = _make_env()
        model = _make_model(env, vf_clip_warmup_updates=10)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestKLDivergenceBranches:
    """Test KL divergence branches."""

    def test_kl_absolute_stop(self):
        """Cover KL absolute stop."""
        env = _make_env()
        model = _make_model(
            env,
            target_kl=0.0001,
            kl_absolute_stop_factor=1.5,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_kl_ema(self):
        """Cover KL EMA updates."""
        env = _make_env()
        model = _make_model(
            env,
            target_kl=0.01,
            kl_ema_updates=5,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestEntropySchedule:
    """Test entropy schedule."""

    def test_entropy_plateau(self):
        """Cover entropy plateau detection."""
        env = _make_env()
        model = _make_model(
            env,
            ent_coef_decay_steps=5,
            ent_coef_plateau_window=3,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestGetParameters:
    """Test get_parameters and set_parameters."""

    def test_get_set_parameters(self):
        """Cover get/set parameters."""
        env = _make_env()
        model = _make_model(env)
        params = model.get_parameters()
        model.set_parameters(params)
        env.close()

    def test_get_parameters_with_optimizer(self):
        """Cover get parameters with optimizer state."""
        env = _make_env()
        model = _make_model(env)
        params = model.get_parameters(include_optimizer=True)
        model.set_parameters(params)
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
