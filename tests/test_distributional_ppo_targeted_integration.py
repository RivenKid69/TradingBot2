"""
Targeted integration tests to cover remaining branches in distributional_ppo.py.
"""

import math
from typing import Optional

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

gymnasium = pytest.importorskip("gymnasium")
from gymnasium import spaces

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

import distributional_ppo as dppo
from distributional_ppo import DistributionalPPO
from custom_policy_patch1 import CustomActorCriticPolicy


class DummyCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True


class EdgeInfoEnv(gymnasium.Env):
    """Env that emits edge-case info fields for rollout coverage."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed: int = 0, max_steps: int = 3) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._max_steps = max_steps
        self._state = np.zeros(4, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        self._state = self._rng.uniform(-0.5, 0.5, size=4).astype(np.float32)
        return self._state.copy(), {"env_id": "edge", "symbol": "edge"}

    def step(self, action):
        self._step += 1
        action_scalar = float(action[0]) if hasattr(action, "__len__") else float(action)
        self._state = np.clip(
            self._state + 0.05 * action_scalar + 0.01 * self._rng.standard_normal(4),
            -1.0,
            1.0,
        ).astype(np.float32)

        reward = float(0.2 - np.sum(self._state ** 2) * 0.01)
        terminated = self._step >= self._max_steps
        truncated = False

        info = {
            "episode": {"r": reward, "l": self._step, "is_success": reward > 0},
            "reward_used_fraction": reward * 0.4,
        }

        if self._step == 1:
            # Force conversion fallbacks
            info.update(
                {
                    "reward_raw_fraction": "bad",
                    "reward_clip_bound_fraction": "bad",
                    "reward_clip_hard_cap_fraction": "bad",
                    "reward_costs_fraction": "bad",
                    "reward_robust_clip_fraction": "0.05",
                    "equity": "bad",
                }
            )
        else:
            info.update(
                {
                    "reward_raw_fraction": reward * 0.5,
                    "reward_clip_bound_fraction": 0.05,
                    "reward_clip_hard_cap_fraction": 0.05,
                    "reward_costs_fraction": 0.01,
                    "reward_robust_clip_fraction": 0.05,
                    "equity": 100.0 + reward,
                }
            )

        if terminated:
            info["time_limit_truncated"] = True
            info["terminal_observation"] = self._state.copy()

        return self._state.copy(), reward, terminated, truncated, info


class MinimalEnv(gymnasium.Env):
    """Minimal deterministic env for train coverage."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed: int = 0, max_steps: int = 4) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._max_steps = max_steps

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        return self._rng.uniform(-0.5, 0.5, size=4).astype(np.float32), {}

    def step(self, action):
        self._step += 1
        obs = self._rng.uniform(-0.5, 0.5, size=4).astype(np.float32)
        reward = float(0.1)
        terminated = self._step >= self._max_steps
        return obs, reward, terminated, False, {}


def make_vec_env(env_cls, n_envs: int = 1, seed: int = 0, **kwargs) -> DummyVecEnv:
    def _make():
        return env_cls(seed=seed, **kwargs)

    return DummyVecEnv([_make for _ in range(n_envs)])


def setup_and_collect(model: DistributionalPPO, env: DummyVecEnv, n_steps: int) -> None:
    total_timesteps = int(n_steps * env.num_envs)
    _, callback = model._setup_learn(
        total_timesteps=total_timesteps,
        callback=DummyCallback(),
        reset_num_timesteps=True,
    )
    model._last_callback = callback
    model._current_progress_remaining = 1.0
    model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=n_steps)


def make_categorical_model(env: DummyVecEnv, **kwargs) -> DistributionalPPO:
    policy_kwargs = {
        "arch_params": {
            "critic": {
                "distributional": True,
                "categorical": True,
                "use_twin_critics": True,
            },
            "num_atoms": 11,
            "v_min": -1.0,
            "v_max": 1.0,
        }
    }
    base_kwargs = {
        "policy": CustomActorCriticPolicy,
        "env": env,
        "n_steps": 3,
        "batch_size": 3,
        "n_epochs": 1,
        "device": "cpu",
        "verbose": 0,
        "clip_range_vf": 0.2,
        "distributional_vf_clip_mode": "mean_and_variance",
        "normalize_returns": False,
        "policy_kwargs": policy_kwargs,
    }
    base_kwargs.update(kwargs)
    return DistributionalPPO(**base_kwargs)


def make_quantile_model(env: DummyVecEnv, **kwargs) -> DistributionalPPO:
    base_kwargs = {
        "policy": "DistributionalPolicy",
        "env": env,
        "n_steps": 4,
        "batch_size": 4,
        "n_epochs": 1,
        "device": "cpu",
        "verbose": 0,
        "clip_range_vf": 0.2,
        "distributional_vf_clip_mode": "per_quantile",
        "normalize_returns": False,
    }
    base_kwargs.update(kwargs)
    return DistributionalPPO(**base_kwargs)


def test_collect_rollouts_edge_info_categorical(monkeypatch):
    env = make_vec_env(EdgeInfoEnv, n_envs=1, seed=1, max_steps=3)
    model = make_categorical_model(env)

    # Preserve training flag to exercise restore branch
    model.policy.train()
    monkeypatch.setattr(model.policy, "set_training_mode", lambda *_args, **_kwargs: None)

    setup_and_collect(model, env, n_steps=3)

    assert model._last_rollout_clip_cap_fraction is not None
    env.close()


def test_collect_rollouts_non_box_action_branch(monkeypatch):
    env = make_vec_env(MinimalEnv, n_envs=1, seed=2, max_steps=2)
    model = make_quantile_model(env)

    # Force non-Box action branch without breaking env expectations
    model.action_space = spaces.Discrete(2)
    monkeypatch.setattr(model, "_ensure_score_action_space", lambda: None)

    setup_and_collect(model, env, n_steps=2)
    env.close()


def test_train_quantile_twin_critics_vf_clipping(monkeypatch):
    env = make_vec_env(MinimalEnv, n_envs=1, seed=3, max_steps=4)
    model = make_quantile_model(env)

    # Ensure scheduler rebuild and KL scaling paths execute
    def _make_scheduler(optimizer):
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    model.policy.optimizer_scheduler_fn = _make_scheduler
    model.policy.lr_scheduler = None
    model.policy.optimizer_scheduler = None

    model._kl_lr_scale = 0.5
    model._kl_min_lr = 1e-6
    model._optimizer_lr_min = 1e-6
    model._optimizer_lr_max = 1e-2
    model._current_progress_remaining = 1.0
    model._base_lr_schedule = lambda _progress: 1e-3

    model.target_kl = 1e-8
    model.kl_early_stop = True
    model._kl_consec_minibatches = 1

    # Inject mismatched group keys to hit warning paths
    model._extract_group_keys_for_indices = lambda *_args, **_kwargs: ["only_one"]
    def _resolve_keys(_rollout_data, valid_indices, _value_valid_indices):
        return ["only_one"], valid_indices
    model._resolve_group_keys_for_training_batch = _resolve_keys

    # Add dummy critic modules with preset grads for gradient monitoring
    model.policy.value_head_critic1 = nn.Linear(1, 1)
    model.policy.value_head_critic2 = nn.Linear(1, 1)
    for param in model.policy.value_head_critic1.parameters():
        param.grad = torch.ones_like(param)
    for param in model.policy.value_head_critic2.parameters():
        param.grad = torch.ones_like(param)

    setup_and_collect(model, env, n_steps=4)

    # Encourage KL threshold to trigger
    model.rollout_buffer.log_probs = np.full_like(model.rollout_buffer.log_probs, 10.0)

    model.train()
    env.close()


def test_train_categorical_twin_critics_mean_variance():
    env = make_vec_env(MinimalEnv, n_envs=1, seed=4, max_steps=4)
    model = make_categorical_model(env, n_steps=4, batch_size=4)

    setup_and_collect(model, env, n_steps=4)
    model.train()
    env.close()


def test_train_nan_loss_logging():
    env = make_vec_env(MinimalEnv, n_envs=1, seed=5, max_steps=4)
    model = make_quantile_model(env)

    setup_and_collect(model, env, n_steps=4)
    model.rollout_buffer.advantages[:] = np.nan

    model.train()
    env.close()
