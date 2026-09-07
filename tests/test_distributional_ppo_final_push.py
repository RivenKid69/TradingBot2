"""
Final push for maximum coverage of distributional_ppo.py.
Targets remaining uncovered branches in core functions.
"""

from __future__ import annotations

import math
import types
from collections import deque
from types import SimpleNamespace
from typing import Any, Optional

import gymnasium
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from distributional_ppo import DistributionalPPO, PopArtController


np.random.seed(42)
torch.manual_seed(42)


def _make_env(max_steps: int = 8) -> DummyVecEnv:
    def _env_fn():
        class _Env(gymnasium.Env):
            def __init__(self):
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
                self.observation_space = spaces.Box(
                    low=-10.0, high=10.0, shape=(4,), dtype=np.float32
                )
                self._step = 0
                self._max = max_steps

            def reset(self, *, seed=None, options=None):
                self._step = 0
                return np.random.randn(4).astype(np.float32), {}

            def step(self, action):
                self._step += 1
                done = self._step >= self._max
                return (
                    np.random.randn(4).astype(np.float32),
                    np.random.randn() * 0.5 - 0.1,  # Variable rewards
                    done,
                    False,
                    {"TimeLimit.truncated": done},
                )

        return _Env()

    return DummyVecEnv([_env_fn])


class _DummyCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_step(self) -> bool:
        return True


def _make_model(env: DummyVecEnv, **overrides) -> DistributionalPPO:
    defaults = {
        "policy": "DistributionalPolicy",
        "env": env,
        "n_steps": 8,
        "batch_size": 4,
        "n_epochs": 1,
        "device": "cpu",
        "verbose": 0,
    }
    defaults.update(overrides)
    return DistributionalPPO(**defaults)


def _setup_and_collect(model: DistributionalPPO, env: DummyVecEnv) -> None:
    total = int(model.n_steps * env.num_envs)
    _, callback = model._setup_learn(
        total_timesteps=total,
        callback=_DummyCallback(),
        reset_num_timesteps=True,
    )
    model._last_callback = callback
    model._current_progress_remaining = 1.0
    model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)


class TestMoreInitBranches:
    """Additional __init__ branch coverage."""

    def test_with_distributional_vf_clip_modes(self):
        """Cover different clip modes."""
        for mode in ["per_quantile", "mean_only", "mean_and_variance"]:
            env = _make_env()
            model = _make_model(
                env,
                distributional_vf_clip_mode=mode,
                clip_range_vf=0.1,
            )
            _setup_and_collect(model, env)
            model.train()
            env.close()

    def test_with_value_target_scale_numeric(self):
        """Cover numeric value_target_scale."""
        env = _make_env()
        model = _make_model(env, value_target_scale=50.0)
        assert model.value_target_scale == pytest.approx(50.0)
        env.close()


class TestMoreTrainBranches:
    """Additional train() branch coverage."""

    def test_train_with_high_clip_range(self):
        """Cover high clip range."""
        env = _make_env()
        model = _make_model(env, clip_range=0.5)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_train_with_low_gamma(self):
        """Cover low gamma."""
        env = _make_env()
        model = _make_model(env, gamma=0.9)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_train_with_high_vf_coef(self):
        """Cover high vf_coef."""
        env = _make_env()
        model = _make_model(env, vf_coef=1.0)
        _setup_and_collect(model, env)
        model.train()
        env.close()

    def test_train_repeated_cycles(self):
        """Cover repeated training cycles."""
        env = _make_env()
        model = _make_model(env)
        for i in range(5):
            _setup_and_collect(model, env)
            model.train()
        env.close()


class TestMoreValueScaleBranches:
    """Additional value scale branch coverage."""

    def test_value_scale_warmup(self):
        """Cover warmup period."""
        env = _make_env()
        model = _make_model(
            env,
            value_scale={"warmup_updates": 3},
        )
        for _ in range(4):
            _setup_and_collect(model, env)
            model.train()
        env.close()

    def test_value_scale_freeze_trigger(self):
        """Cover freeze trigger."""
        env = _make_env()
        model = _make_model(
            env,
            value_scale={"freeze_after_updates": 2},
        )
        for _ in range(5):
            _setup_and_collect(model, env)
            model.train()
        env.close()


class TestMoreCvarBranches:
    """Additional CVaR branch coverage."""

    def test_cvar_alpha_extreme(self):
        """Cover extreme CVaR alpha."""
        env = _make_env()
        model = _make_model(env, cvar_alpha=0.05)
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestMoreEntropyBranches:
    """Additional entropy branch coverage."""

    def test_entropy_with_min(self):
        """Cover entropy minimum."""
        env = _make_env()
        model = _make_model(
            env,
            ent_coef_decay_steps=3,
            ent_coef_min=0.001,
        )
        for _ in range(5):
            _setup_and_collect(model, env)
            model.train()
        env.close()


class TestMoreKlBranches:
    """Additional KL branch coverage."""

    def test_kl_with_ema(self):
        """Cover KL EMA."""
        env = _make_env()
        model = _make_model(
            env,
            target_kl=0.05,
            kl_ema_updates=3,
        )
        for _ in range(4):
            _setup_and_collect(model, env)
            model.train()
        env.close()


class TestMoreReturnNormalization:
    """Additional return normalization coverage."""

    def test_normalize_with_high_variance(self):
        """Cover high variance returns."""
        env = _make_env()
        model = _make_model(env, normalize_returns=True, ret_clip=5.0)
        for _ in range(3):
            _setup_and_collect(model, env)
            model.train()
        env.close()


class TestMoreCollectRollouts:
    """Additional collect_rollouts coverage."""

    def test_collect_multiple_envs(self):
        """Cover multiple environments."""

        def _env_fn():
            class _Env(gymnasium.Env):
                def __init__(self):
                    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
                    self.observation_space = spaces.Box(
                        low=-10.0, high=10.0, shape=(4,), dtype=np.float32
                    )
                    self._step = 0

                def reset(self, *, seed=None, options=None):
                    self._step = 0
                    return np.random.randn(4).astype(np.float32), {}

                def step(self, action):
                    self._step += 1
                    done = self._step >= 8
                    return (
                        np.random.randn(4).astype(np.float32),
                        np.random.randn(),
                        done,
                        False,
                        {"TimeLimit.truncated": done},
                    )

            return _Env()

        env = DummyVecEnv([_env_fn, _env_fn])  # 2 envs
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            device="cpu",
            verbose=0,
        )
        _setup_and_collect(model, env)
        model.train()
        env.close()


class TestMorePredict:
    """Additional predict coverage."""

    def test_predict_batch(self):
        """Cover batch prediction."""
        env = _make_env()
        model = _make_model(env)
        obs = np.random.randn(5, 4).astype(np.float32)
        actions, _ = model.predict(obs, deterministic=True)
        assert actions.shape[0] == 5
        env.close()

    def test_predict_single(self):
        """Cover single observation."""
        env = _make_env()
        model = _make_model(env)
        obs = np.random.randn(4).astype(np.float32)
        action, _ = model.predict(obs, deterministic=False)
        assert action is not None
        env.close()


class TestMoreExplainedVariance:
    """Additional explained variance coverage."""

    def test_ev_after_many_updates(self):
        """Cover EV after many updates."""
        env = _make_env()
        model = _make_model(env)
        for _ in range(10):
            _setup_and_collect(model, env)
            model.train()
        assert model._last_explained_variance is not None
        env.close()


class TestMoreSaveLoad:
    """Additional save/load coverage."""

    def test_save_load_with_params(self, tmp_path):
        """Cover save and load with custom params."""
        env = _make_env()
        model = _make_model(
            env,
            normalize_returns=True,
            cvar_alpha=0.25,
        )
        _setup_and_collect(model, env)
        model.train()

        path = tmp_path / "model.zip"
        model.save(str(path))
        env.close()

        env2 = _make_env()
        loaded = DistributionalPPO.load(str(path), env=env2)
        assert loaded is not None
        env2.close()


class TestMoreMixedScenarios:
    """Combined scenarios for more coverage."""

    def test_full_scenario(self):
        """Cover full training scenario with all features."""
        env = _make_env()
        model = _make_model(
            env,
            normalize_returns=True,
            cvar_alpha=0.2,
            target_kl=0.02,
            clip_range=0.2,
            clip_range_vf=0.2,
            ent_coef_decay_steps=5,
            value_scale={
                "ema_beta": 0.3,
                "max_rel_step": 0.3,
            },
        )
        for _ in range(5):
            _setup_and_collect(model, env)
            model.train()
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
