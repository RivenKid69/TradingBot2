from __future__ import annotations

import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from distributional_ppo import (
    DistributionalPPO,
    PopArtController,
    PopArtHoldoutBatch,
    PopArtHoldoutEvaluation,
    _ValuePredictionCacheEntry,
    compute_grouped_explained_variance,
    safe_explained_variance,
)
from variance_gradient_scaler import VarianceGradientScaler


class TinyEnv(gym.Env):
    """Minimal gymnasium-compatible env with configurable info payload."""

    def __init__(self, action_space=None, info_fn=None, max_steps: int = 3):
        self.action_space = action_space or spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self._step = 0
        self._max_steps = max_steps
        self._info_fn = info_fn

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._step = 0
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action):
        self._step += 1
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        terminated = self._step >= self._max_steps
        info = {} if self._info_fn is None else self._info_fn(self._step, action, terminated)
        return obs, reward, terminated, False, info


class DummyCallback(BaseCallback):
    def __init__(self, *, stop_after: int | None = None):
        super().__init__()
        self._stop_after = stop_after
        self._count = 0

    def _on_step(self) -> bool:
        if self._stop_after is None:
            return True
        self._count += 1
        return self._count < self._stop_after


def make_vec_env(action_space=None, info_fns=None, max_steps: int = 3) -> DummyVecEnv:
    if info_fns is None:
        info_fns = [None]

    def _make(fn):
        return lambda: TinyEnv(action_space=action_space, info_fn=fn, max_steps=max_steps)

    return DummyVecEnv([_make(fn) for fn in info_fns])


def make_model(env=None, **kwargs) -> DistributionalPPO:
    if env is None:
        env = make_vec_env()
    defaults = {
        "policy": "DistributionalPolicy",
        "env": env,
        "n_steps": 4,
        "batch_size": 2,
        "n_epochs": 1,
        "device": "cpu",
        "verbose": 0,
    }
    defaults.update(kwargs)
    return DistributionalPPO(**defaults)


def setup_and_collect(model: DistributionalPPO, env, *, n_steps: int | None = None) -> bool:
    total = int((n_steps or model.n_steps) * env.num_envs)
    _, callback = model._setup_learn(
        total_timesteps=total,
        callback=DummyCallback(),
        reset_num_timesteps=True,
    )
    model._last_callback = callback
    model._current_progress_remaining = 1.0
    return model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=n_steps or model.n_steps)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"ent_coef_min": -0.1}, "ent_coef_min"),
        ({"entropy_boost_cap": 0.0}, "entropy_boost_cap"),
        ({"kl_epoch_decay": 1.5}, "kl_epoch_decay"),
        ({"kl_ema_alpha": 2.0}, "kl_ema_alpha"),
        ({"kl_exceed_stop_fraction": 2.0}, "kl_exceed_stop_fraction"),
        ({"kl_penalty_beta": float("nan")}, "kl_penalty_beta"),
    ],
)
def test_init_invalid_params_raise(kwargs, match):
    env = make_vec_env()
    with pytest.raises(ValueError, match=match):
        make_model(env=env, **kwargs)


def test_get_optimizer_kwargs_variants():
    model = make_model()
    for name in ("UPGDW", "Adam", "SGD"):
        model._optimizer_class = name
        kwargs = model._get_optimizer_kwargs()
        assert "weight_decay" in kwargs
    model._optimizer_class = "UPGDW"
    kwargs = model._get_optimizer_kwargs()
    assert "betas" in kwargs and "eps" in kwargs


def test_value_scale_helpers():
    model = make_model()
    model._value_target_scale_smoothing_beta = 0.5
    model._value_target_scale_max_change_pct = 0.1
    smoothed = model._smooth_value_target_scale(target=10.0, previous=1.0)
    assert smoothed <= 1.1

    model._value_scale_range_max_rel_step = 0.6
    limited_min, limited_max = model._limit_v_range_step(0.0, 1.0, 10.0, -10.0)
    assert limited_max > limited_min

    model.running_v_min = -1.0
    model.running_v_max = 1.0
    model.v_range_initialized = True
    model.v_range_ema_alpha = 0.5
    model._allow_v_range_shrink = False
    old_min, old_max, new_min, new_max, changed = model._apply_v_range_update(0.5, -0.5)
    assert old_min <= new_min <= old_max
    assert old_min <= new_max <= old_max
    assert isinstance(changed, bool)


def test_update_critic_gradient_block_logs():
    model = make_model()
    model._critic_grad_warmup_updates = 2
    model._critic_grad_block_scale = 1.0
    model._critic_grad_block_logged_state = False
    model._update_critic_gradient_block(0)
    model._update_critic_gradient_block(3)
    assert isinstance(model._critic_grad_blocked, bool)


def test_kl_integral_limit_positive():
    model = make_model()
    model.kl_penalty_ki = 0.5
    model.kl_penalty_beta_min = 0.0
    model.kl_penalty_beta_max = 1.0
    assert model._kl_integral_limit() > 0.0


def test_kl_diag_step_missing_stats():
    model = make_model()

    class DummyDist:
        def __init__(self):
            self._base = torch.distributions.Normal(torch.zeros(1), torch.ones(1))

        def log_prob(self, value):
            return self._base.log_prob(value)

    raw = torch.zeros((4, 1))
    rollout_data = SimpleNamespace(
        actions=raw,
        actions_raw=raw,
        old_log_prob_raw=torch.zeros_like(raw),
    )
    model._kl_diag_step(DummyDist(), rollout_data)


def test_compute_explained_variance_metric_fallback():
    model = make_model()
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_pred = torch.tensor([float("nan"), float("nan"), float("nan"), float("nan")])
    mask = torch.tensor([1.0, 1.0, 0.0, 1.0])
    y_true_raw = torch.tensor([1.0, 2.0])
    result = model._compute_explained_variance_metric(
        y_true,
        y_pred,
        mask_tensor=mask,
        y_true_tensor_raw=y_true_raw,
        allow_fallback=True,
    )
    assert result[0] is None or math.isfinite(result[0]) or math.isnan(result[0])


def test_compute_grouped_explained_variance_edges():
    y_true = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float64)
    y_pred = np.array([np.nan, 2.0, 3.0, np.nan], dtype=np.float64)
    groups = np.array(["a", "a", "b", "b"], dtype=object)
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    ev_grouped, summary = compute_grouped_explained_variance(
        y_true, y_pred, groups, weights=weights
    )
    assert "a" in ev_grouped and "b" in ev_grouped
    assert "mean_unweighted" in summary


def test_popart_shadow_and_live_nan_paths():
    env = make_vec_env()
    model = make_model(env=env)

    def holdout_loader():
        obs = torch.zeros((2, 4), dtype=torch.float32)
        returns = torch.zeros((2, 1), dtype=torch.float32)
        episode_starts = torch.zeros((2, 1), dtype=torch.bool)
        mask = torch.zeros((2, 1), dtype=torch.float32)
        return PopArtHoldoutBatch(obs, returns, episode_starts, None, mask)

    controller = PopArtController(enabled=True, mode="shadow", holdout_loader=holdout_loader, logger=model.logger)
    with pytest.raises(IndexError):
        controller.evaluate_shadow(
            model=model,
            returns_raw=torch.tensor(float("nan")),
            ret_mean=float("nan"),
            ret_std=1.0,
        )

    metrics = controller.evaluate_shadow(
        model=model,
        returns_raw=torch.tensor([float("nan")]),
        ret_mean=float("nan"),
        ret_std=1.0,
    )
    assert metrics is None or metrics.samples >= 0

    controller_live = PopArtController(enabled=True, mode="live", holdout_loader=holdout_loader, logger=model.logger)
    controller_live._last_holdout_eval = PopArtHoldoutEvaluation(
        baseline_raw=torch.zeros((2, 1)),
        candidate_raw=torch.ones((2, 1)),
        target_raw=torch.zeros((2, 1)),
        mask=None,
        ev_before=0.0,
        ev_after=0.0,
        clip_fraction_before=0.0,
        clip_fraction_after=0.0,
    )
    controller_live.apply_live_update(model=model, old_mean=0.0, old_std=1.0, new_mean=0.1, new_std=1.1)
    assert controller_live.apply_count >= 0


def test_refresh_value_prediction_tensors_clip():
    env = make_vec_env()
    model = make_model(env=env)
    obs = torch.zeros((2, 4), dtype=torch.float32)
    entry = _ValuePredictionCacheEntry(
        observations=obs,
        lstm_states=model.policy.recurrent_initial_state,
        episode_starts=torch.zeros((2, 1), dtype=torch.bool),
        valid_indices=torch.tensor([0], dtype=torch.long),
        base_scale=1.0,
        old_values_raw=torch.zeros((2, 1), dtype=torch.float32),
        mask_values=torch.ones((2, 1), dtype=torch.float32),
    )
    ret_mu = torch.tensor(0.0)
    ret_std = torch.tensor(1.0)
    primary_preds, reserve_preds, primary_weights, reserve_weights = model._refresh_value_prediction_tensors(
        primary_cache=[entry],
        primary_predictions=[],
        reserve_cache=[],
        reserve_predictions=[],
        primary_weights=[],
        reserve_weights=[],
        clip_range_vf_value=0.2,
        ret_mu_tensor=ret_mu,
        ret_std_tensor=ret_std,
    )
    assert len(primary_preds) == 1
    assert len(primary_weights) == 1


def test_enforce_optimizer_lr_bounds_and_update():
    model = make_model()
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([param], lr=1e-4)
    model.policy.optimizer = optimizer
    model._optimizer_lr_min = 1e-3
    model._optimizer_lr_max = 1e-2
    model._scheduler_min_lr = 1e-3
    model._kl_lr_scale = 1.0
    model._current_progress_remaining = 0.5

    def bad_schedule(_):
        raise RuntimeError("schedule failure")

    model._base_lr_schedule = bad_schedule
    model._enforce_optimizer_lr_bounds()

    class DummyScheduler:
        def get_last_lr(self):
            return [0.002]

    model.policy.lr_scheduler = DummyScheduler()
    model._update_learning_rate(optimizer)


def test_reset_lstm_states_for_done_envs():
    model = make_model()
    states = (
        torch.zeros((1, 2, 3)),
        torch.zeros((1, 2, 3)),
    )
    init_states = (
        torch.ones((1, 1, 3)),
        torch.ones((1, 1, 3)),
    )
    dones = np.array([True, False])
    updated = model._reset_lstm_states_for_done_envs(states, dones, init_states)
    assert updated is not None
    assert torch.allclose(updated[0][:, 0, :], init_states[0][:, 0, :])


def test_filter_ev_reserve_rows_empty_indices():
    model = make_model()
    rollout_data = SimpleNamespace(sample_indices=torch.full((2, 2), -1))
    target_norm = torch.ones((4, 1))
    target_raw = torch.ones((4, 1))
    weights = torch.ones((4, 1))
    new_norm, new_raw, new_weights, indices = model._filter_ev_reserve_rows(
        rollout_data, target_norm, target_raw, weights, None
    )
    assert new_norm.numel() == 0
    assert new_raw.numel() == 0
    assert indices.numel() == 0
    assert new_weights is None or new_weights.numel() == 0


def test_collect_rollouts_edge_infos_and_vecnormalize():
    def info_mapping(step, action, done):
        return {
            "reward_robust_clip_fraction": "bad",
            "time_limit_truncated": True,
            "terminal_observation": None,
        }

    env = make_vec_env(info_fns=[info_mapping, info_mapping])
    env = VecNormalize(env, norm_reward=False)
    model = make_model(env=env)

    total = int(env.num_envs)
    callback = DummyCallback(stop_after=1)
    _, callback = model._setup_learn(
        total_timesteps=total,
        callback=callback,
        reset_num_timesteps=True,
    )
    ok = model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=1)
    assert ok is False


def test_collect_rollouts_rejects_non_recurrent_buffer():
    env = make_vec_env()
    model = make_model(env=env)
    with pytest.raises(TypeError):
        model.collect_rollouts(env, DummyCallback(), object(), n_rollout_steps=1)


@pytest.mark.parametrize(
    "kwargs, match, exc",
    [
        ({"clip_range_vf": -0.1}, "clip_range_vf", ValueError),
        ({"vf_clip_threshold_ev": 2.0}, "vf_clip_threshold_ev", ValueError),
        ({"distributional_vf_clip_mode": "invalid"}, "distributional_vf_clip_mode", ValueError),
        ({"distributional_vf_clip_variance_factor": 0.5}, "distributional_vf_clip_variance_factor", ValueError),
        ({"value_target_scale_fixed": -1.0}, "value_target_scale_fixed", ValueError),
        ({"optimizer_lr_min": -1.0}, "optimizer_lr_min", ValueError),
        ({"scheduler_min_lr": -1.0}, "scheduler_min_lr", ValueError),
        ({"optimizer_lr_max": 0.0}, "optimizer_lr_max", ValueError),
        ({"optimizer_kwargs": 123}, "optimizer_kwargs", TypeError),
        ({"ppo_clip_range": -0.1}, "ppo_clip_range", ValueError),
        ({"clip_range_warmup": -0.1}, "clip_range_warmup", ValueError),
        ({"vf_bad_explained_floor": -0.1}, "vf_bad_explained_floor", ValueError),
        ({"cvar_activation_threshold": -0.1}, "cvar_activation_threshold", ValueError),
        ({"cvar_activation_hysteresis": -0.1}, "cvar_activation_hysteresis", ValueError),
        ({"cql_beta": 0.0}, "cql_beta", ValueError),
        ({"cvar_alpha": 2.0}, "cvar_alpha", ValueError),
        ({"cvar_cap": 0.0}, "cvar_cap", ValueError),
        ({"cvar_lambda_lr": -0.1}, "cvar_lambda_lr", ValueError),
        ({"cvar_penalty_cap": -0.1}, "cvar_penalty_cap", ValueError),
        ({"cvar_ema_beta": 1.0}, "cvar_ema_beta", ValueError),
        ({"v_range_ema_alpha": 1.5}, "v_range_ema_alpha", ValueError),
    ],
)
def test_init_invalid_params_raise_more(kwargs, match, exc):
    env = make_vec_env()
    with pytest.raises(exc, match=match):
        make_model(env=env, **kwargs)


def test_init_defaults_and_adjustments():
    env = make_vec_env()
    model = make_model(
        env=env,
        optimizer_class=None,
        optimizer_lr_min=1e-2,
        optimizer_lr_max=1e-3,
        scheduler_min_lr=5e-3,
        policy_kwargs=None,
        entropy_boost_factor=0.5,
        target_kl=None,
        kl_ema_alpha=0.5,
        kl_penalty_pid_kp=-1.0,
        kl_penalty_pid_ki=-1.0,
        kl_penalty_pid_kd=-1.0,
        value_scale_stability={"min_ev": 0.5},
        value_scale_target_ema_beta=0.5,
        value_scale_controller_holdout=lambda: None,
    )
    assert model._optimizer_class == "adaptive_upgd"
    assert model._optimizer_lr_max == model._optimizer_lr_min
    assert model._entropy_boost_factor == 1.0
    assert model.target_kl == 0.5
    assert model._kl_ema_alpha == 0.5
    assert model.kl_penalty_kp == 0.0
    assert model.kl_penalty_ki == 0.0
    assert model.kl_penalty_kd == 0.0


def test_init_value_scale_target_beta_invalid():
    env = make_vec_env()
    with pytest.raises(ValueError, match="value_scale.ema_beta"):
        make_model(env=env, value_scale_target_ema_beta=2.0)


def test_collect_rollouts_initializes_lstm_states():
    env = make_vec_env()
    model = make_model(env=env)
    total = int(env.num_envs)
    _, callback = model._setup_learn(
        total_timesteps=total,
        callback=DummyCallback(),
        reset_num_timesteps=True,
    )
    model._last_lstm_states = None
    ok = model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=1)
    assert ok is True
    assert model._last_lstm_states is not None


def test_collect_rollouts_vecnormalize_norm_reward_raises():
    env = make_vec_env()
    vec_env = VecNormalize(env, norm_reward=True)
    model = make_model(env=vec_env)
    total = int(vec_env.num_envs)
    _, callback = model._setup_learn(
        total_timesteps=total,
        callback=DummyCallback(),
        reset_num_timesteps=True,
    )
    with pytest.raises(AssertionError):
        model.collect_rollouts(vec_env, callback, model.rollout_buffer, n_rollout_steps=1)


def test_collect_rollouts_discrete_action_path():
    env = make_vec_env(action_space=spaces.Discrete(2))
    with pytest.raises(NotImplementedError, match="Box action space"):
        make_model(env=env)


def test_collect_rollouts_missing_value_logits_raises():
    env = make_vec_env()
    model = make_model(
        env=env,
        policy_kwargs={
            "arch_params": {"critic": {"distributional": True, "categorical": True}}
        },
    )
    total = int(env.num_envs)
    _, callback = model._setup_learn(
        total_timesteps=total,
        callback=DummyCallback(),
        reset_num_timesteps=True,
    )
    original_forward = model.policy.forward

    def _patched_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        model.policy._last_value_logits = None
        return result

    model.policy.forward = _patched_forward
    with pytest.raises(RuntimeError, match="value logits"):
        model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=1)


def test_collect_rollouts_time_limit_bootstrap():
    def info_mapping(step, action, done):
        info = {}
        if done:
            info["time_limit_truncated"] = True
            info["terminal_observation"] = np.zeros((4,), dtype=np.float32)
        return info

    env = make_vec_env(info_fns=[info_mapping])
    model = make_model(env=env)
    total = int(env.num_envs)
    _, callback = model._setup_learn(
        total_timesteps=total,
        callback=DummyCallback(),
        reset_num_timesteps=True,
    )
    ok = model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=2)
    assert ok is True


def test_twin_critics_vf_clipping_loss_quantile_modes():
    model = make_model()
    latent_dim = int(getattr(model.policy, "lstm_output_dim", 32))
    batch = 4
    latent_vf = torch.zeros((batch, latent_dim))
    targets = torch.zeros((batch, 1))
    num_quantiles = int(getattr(model.policy, "num_quantiles", 8))
    old_q1 = torch.zeros((batch, num_quantiles))
    old_q2 = torch.ones((batch, num_quantiles))
    model._twin_critics_vf_clipping_loss(
        latent_vf,
        targets,
        old_q1,
        old_q2,
        clip_delta=0.2,
        reduction="none",
        mode="mean_only",
        return_full=True,
    )
    model._twin_critics_vf_clipping_loss(
        latent_vf,
        targets,
        old_q1,
        old_q2,
        clip_delta=0.2,
        reduction="none",
        mode="mean_and_variance",
        return_full=True,
    )


def test_twin_critics_vf_clipping_loss_categorical_paths():
    env = make_vec_env()
    model = make_model(
        env=env,
        policy_kwargs={
            "arch_params": {"critic": {"distributional": True, "categorical": True}}
        },
    )
    latent_dim = int(getattr(model.policy, "lstm_output_dim", 32))
    batch = 3
    latent_vf = torch.zeros((batch, latent_dim))
    num_atoms = int(getattr(model.policy, "num_atoms", 5))
    old_probs_1 = torch.full((batch, num_atoms), 1.0 / num_atoms)
    old_probs_2 = torch.full((batch, num_atoms), 1.0 / num_atoms)
    target_distribution = torch.full((batch, num_atoms), 1.0 / num_atoms)
    model._twin_critics_vf_clipping_loss(
        latent_vf,
        targets=None,
        old_quantiles_critic1=None,
        old_quantiles_critic2=None,
        clip_delta=0.2,
        reduction="none",
        old_probs_critic1=old_probs_1,
        old_probs_critic2=old_probs_2,
        target_distribution=target_distribution,
        return_full=True,
    )
    with pytest.raises(ValueError, match="target_distribution"):
        model._twin_critics_vf_clipping_loss(
            latent_vf,
            targets=None,
            old_quantiles_critic1=None,
            old_quantiles_critic2=None,
            clip_delta=0.2,
            reduction="none",
            old_probs_critic1=old_probs_1,
            old_probs_critic2=old_probs_2,
            target_distribution=None,
        )
    with pytest.raises(ValueError, match="old_probs"):
        model._twin_critics_vf_clipping_loss(
            latent_vf,
            targets=None,
            old_quantiles_critic1=None,
            old_quantiles_critic2=None,
            clip_delta=0.2,
            reduction="none",
            old_probs_critic1=None,
            old_probs_critic2=None,
            target_distribution=target_distribution,
        )


def test_setup_dependent_components_restores_vgs_state():
    env = make_vec_env()
    model = make_model(env=env)
    vgs = VarianceGradientScaler(
        parameters=model.policy.parameters(),
        enabled=True,
        beta=0.9,
        alpha=0.1,
        warmup_steps=1,
        logger=model.logger,
    )
    model._vgs_enabled = True
    model._setup_complete = False
    model._variance_gradient_scaler = None
    model._vgs_saved_state_for_restore = vgs.state_dict()
    model._setup_dependent_components()
    assert model._variance_gradient_scaler is not None


def test_finalize_return_stats_paths():
    env = make_vec_env()
    model = make_model(env=env)
    if hasattr(model, "running_v_min"):
        delattr(model, "running_v_min")
    if hasattr(model, "running_v_max"):
        delattr(model, "running_v_max")
    model._value_scale_frozen = True
    model._value_scale_never_freeze = True
    model._value_scale_warmup_buffer = [float("nan"), float("nan")]
    model._value_scale_warmup_buffer_limit = 1
    model._value_scale_warmup_limit = 0
    model._value_scale_update_count = 1
    model._value_scale_min_samples = 1
    model._value_scale_updates_enabled = True
    model._value_clip_limit_unscaled = 1.0
    model.rollout_buffer.returns = np.full_like(model.rollout_buffer.returns, np.nan)
    model._finalize_return_stats()


def test_finalize_return_stats_updates_locked_raises():
    env = make_vec_env()
    model = make_model(env=env)
    model._value_target_scale_fixed = 1.0
    model._value_scale_updates_enabled = True
    model._value_scale_warmup_limit = 0
    model._value_scale_update_count = 1
    model._value_scale_min_samples = 1
    model._value_scale_warmup_buffer = [0.1]
    model.rollout_buffer.returns = np.zeros_like(model.rollout_buffer.returns)
    with pytest.raises(RuntimeError, match="value_scale_update_applied"):
        model._finalize_return_stats()

    model._value_scale_updates_enabled = False
    model._value_scale_prev_effective = model._value_target_scale_effective + 1.0
    with pytest.raises(RuntimeError, match="value_target_scale drift"):
        model._finalize_return_stats()


def test_safe_explained_variance_edge_weights():
    y_true = np.array([1.0, 2.0], dtype=np.float64)
    y_pred = np.array([1.5, 2.5], dtype=np.float64)
    weights = np.array([1e100, 1e100], dtype=np.float64)
    value = safe_explained_variance(y_true, y_pred, weights)
    assert math.isnan(value)


def test_adjust_kl_penalty_updates():
    model = make_model()
    model.target_kl = 0.1
    model.kl_penalty_kp = 0.5
    model.kl_penalty_ki = 0.5
    model.kl_penalty_kd = 0.1
    model.kl_penalty_beta_min = 0.0
    model.kl_penalty_beta_max = 1.0
    model.kl_beta = 0.1
    model._kl_err_int = 0.0
    model._kl_err_prev = 0.0
    model._adjust_kl_penalty(0.2)
    assert model.kl_beta >= 0.0
    assert model._kl_pid_p != 0.0


def test_configure_loss_head_weights_records():
    model = make_model()
    model._configure_loss_head_weights({"a": True, "b": 0.2, "c": "bad"})
    assert model._loss_head_weights == {"a": 1.0, "b": 0.2}


def test_record_value_debug_stats_none_tensor():
    model = make_model()
    model._record_value_debug_stats("none", None)


def test_refresh_value_prediction_tensors_empty_cache():
    model = make_model()
    preds, reserve_preds, masks, reserve_masks = model._refresh_value_prediction_tensors(
        primary_cache=[],
        primary_predictions=[torch.tensor([1.0])],
        reserve_cache=[],
        reserve_predictions=[],
        primary_weights=[None],
        reserve_weights=[],
        clip_range_vf_value=None,
        ret_mu_tensor=torch.tensor(0.0),
        ret_std_tensor=torch.tensor(1.0),
    )
    assert preds
    assert reserve_preds == []
    assert masks == [None]
    assert reserve_masks == []


def test_refresh_value_prediction_tensors_quantile_clipping_and_masks():
    env = make_vec_env()
    model = make_model(env=env)
    model.normalize_returns = False
    model._value_clip_limit_scaled = 0.5
    entry_primary = _ValuePredictionCacheEntry(
        observations=torch.zeros((1, 4), dtype=torch.float32),
        lstm_states=None,
        episode_starts=torch.zeros((1, 1), dtype=torch.float32),
        valid_indices=torch.tensor([0], dtype=torch.int32),
        base_scale=1.0,
        old_values_raw=torch.zeros((1, 1), dtype=torch.float32),
        mask_values=None,
    )
    entry_reserve = _ValuePredictionCacheEntry(
        observations=torch.zeros((1, 4), dtype=torch.float32),
        lstm_states=None,
        episode_starts=torch.zeros((1, 1), dtype=torch.float32),
        valid_indices=None,
        base_scale=1.0,
        old_values_raw=torch.zeros((1, 1), dtype=torch.float32),
        mask_values=torch.zeros((0,), dtype=torch.float32),
    )
    preds, reserve_preds, masks, reserve_masks = model._refresh_value_prediction_tensors(
        primary_cache=[entry_primary],
        primary_predictions=[],
        reserve_cache=[entry_reserve],
        reserve_predictions=[],
        primary_weights=[None],
        reserve_weights=[None],
        clip_range_vf_value=0.1,
        ret_mu_tensor=torch.tensor(0.0),
        ret_std_tensor=torch.tensor(1.0),
    )
    assert len(preds) == 1
    assert len(reserve_preds) == 1
    assert masks[0] is None
    assert reserve_masks[0] is not None


def test_refresh_value_prediction_tensors_categorical_paths():
    env = make_vec_env()
    policy_kwargs = {"arch_params": {"critic": {"distributional": True, "categorical": True}}}
    model = make_model(env=env, policy_kwargs=policy_kwargs)
    model._value_clip_limit_scaled = 0.5
    entry = _ValuePredictionCacheEntry(
        observations=torch.zeros((1, 4), dtype=torch.float32),
        lstm_states=None,
        episode_starts=torch.zeros((1, 1), dtype=torch.float32),
        valid_indices=None,
        base_scale=1.0,
        old_values_raw=torch.zeros((1, 1), dtype=torch.float32),
        mask_values=None,
    )
    model.normalize_returns = True
    model._refresh_value_prediction_tensors(
        primary_cache=[entry],
        primary_predictions=[],
        reserve_cache=[],
        reserve_predictions=[],
        primary_weights=[None],
        reserve_weights=[],
        clip_range_vf_value=0.1,
        ret_mu_tensor=torch.tensor(0.0),
        ret_std_tensor=torch.tensor(1.0),
    )
    model.normalize_returns = False
    model._refresh_value_prediction_tensors(
        primary_cache=[entry],
        primary_predictions=[],
        reserve_cache=[],
        reserve_predictions=[],
        primary_weights=[None],
        reserve_weights=[],
        clip_range_vf_value=0.1,
        ret_mu_tensor=torch.tensor(0.0),
        ret_std_tensor=torch.tensor(1.0),
    )


def test_compute_explained_variance_metric_mask_empty_returns():
    model = make_model()
    y_true = torch.tensor([1.0], dtype=torch.float32)
    y_pred = torch.tensor([1.0], dtype=torch.float32)
    mask = torch.zeros((0,), dtype=torch.float32)
    ev, y_true_flat, y_pred_flat, _ = model._compute_explained_variance_metric(
        y_true, y_pred, mask_tensor=mask
    )
    assert ev is None
    assert y_true_flat is not None
    assert y_pred_flat is not None


def test_compute_explained_variance_metric_empty_without_mask():
    model = make_model()
    y_true = torch.zeros((0,), dtype=torch.float32)
    y_pred = torch.zeros((0,), dtype=torch.float32)
    ev, y_true_flat, y_pred_flat, _ = model._compute_explained_variance_metric(y_true, y_pred)
    assert ev is None
    assert y_true_flat is not None
    assert y_pred_flat is not None


def test_compute_explained_variance_metric_fallback_delta_and_empty_indices():
    model = make_model()
    y_true = torch.tensor([1e-6, 2e-6], dtype=torch.float32)
    y_pred = torch.tensor([1e-6, 2e-6], dtype=torch.float32)
    mask = torch.tensor([0.0, 1.0], dtype=torch.float32)
    y_true_raw = torch.tensor([1e-6], dtype=torch.float32)
    ev, y_true_flat, y_pred_flat, _ = model._compute_explained_variance_metric(
        y_true, y_pred, mask_tensor=mask, y_true_tensor_raw=y_true_raw
    )
    assert ev is not None
    assert y_true_flat is not None
    assert y_pred_flat is not None


def test_safe_explained_variance_large_weight_sum():
    y_true = np.array([1.0, 2.0], dtype=np.float64)
    y_pred = np.array([1.5, 2.5], dtype=np.float64)
    weights = np.array([1e308, 1e308], dtype=np.float64)
    assert math.isnan(safe_explained_variance(y_true, y_pred, weights))


def test_safe_explained_variance_residual_overflow():
    y_true = np.array([1e308, 1e308], dtype=np.float64)
    y_pred = np.array([-1e308, -1e308], dtype=np.float64)
    weights = np.array([1.0, 1.0], dtype=np.float64)
    assert math.isnan(safe_explained_variance(y_true, y_pred, weights))


def test_safe_explained_variance_unweighted_overflow():
    y_true = np.array([1e308, -1e308], dtype=np.float64)
    y_pred = np.array([0.0, 0.0], dtype=np.float64)
    assert math.isnan(safe_explained_variance(y_true, y_pred))


def test_train_twin_critics_vf_clipping_with_costs():
    def info_fn(step, action, terminated):
        return {
            "reward_raw_fraction": 0.1,
            "reward_costs_fraction": 0.02,
            "reward_robust_clip_fraction": "bad",
        }

    env = make_vec_env(info_fns=[info_fn], max_steps=4)
    model = make_model(
        env=env,
        clip_range_vf=0.2,
        normalize_returns=True,
        distributional_vf_clip_mode="per_quantile",
    )
    setup_and_collect(model, env, n_steps=4)
    model.train()


def test_train_single_critic_vf_clipping_mean_and_variance():
    env = make_vec_env(max_steps=4)
    policy_kwargs = {"arch_params": {"critic": {"use_twin_critics": False}}}
    model = make_model(
        env=env,
        clip_range_vf=0.2,
        distributional_vf_clip_mode="mean_and_variance",
        policy_kwargs=policy_kwargs,
    )
    setup_and_collect(model, env, n_steps=4)
    model.train()


def test_train_single_critic_vf_clipping_per_quantile():
    env = make_vec_env(max_steps=4)
    policy_kwargs = {"arch_params": {"critic": {"use_twin_critics": False}}}
    model = make_model(
        env=env,
        clip_range_vf=0.2,
        distributional_vf_clip_mode="per_quantile",
        policy_kwargs=policy_kwargs,
    )
    setup_and_collect(model, env, n_steps=4)
    model.train()


def test_train_categorical_vf_clipping_mean_and_variance():
    env = make_vec_env(max_steps=4)
    policy_kwargs = {"arch_params": {"critic": {"distributional": True, "categorical": True}}}
    model = make_model(
        env=env,
        clip_range_vf=0.2,
        normalize_returns=True,
        distributional_vf_clip_mode="mean_and_variance",
        policy_kwargs=policy_kwargs,
    )
    setup_and_collect(model, env, n_steps=4)
    model.train()


def test_train_categorical_vf_clipping_per_quantile():
    env = make_vec_env(max_steps=4)
    policy_kwargs = {"arch_params": {"critic": {"distributional": True, "categorical": True}}}
    model = make_model(
        env=env,
        clip_range_vf=0.2,
        distributional_vf_clip_mode="per_quantile",
        policy_kwargs=policy_kwargs,
    )
    setup_and_collect(model, env, n_steps=4)
    model.train()


def test_train_kl_early_stop_consecutive():
    env = make_vec_env(max_steps=4)
    model = make_model(env=env)
    setup_and_collect(model, env, n_steps=4)
    model.rollout_buffer.log_probs[:] = 50.0
    model.target_kl = 1e-4
    model.kl_early_stop = True
    model._kl_consec_minibatches = 1
    model._kl_absolute_stop_factor = 0.0
    model.train()


def test_train_nan_log_ratio_branch():
    env = make_vec_env(max_steps=4)
    model = make_model(env=env)
    setup_and_collect(model, env, n_steps=4)

    def _eval_with_nan(obs, actions, lstm_states, episode_starts, actions_raw=None):
        values = torch.zeros((actions.shape[0], 1), device=actions.device)
        log_prob = torch.full((actions.shape[0], 1), float("nan"), device=actions.device)
        entropy = torch.zeros_like(log_prob)
        return values, log_prob, entropy

    model.policy.evaluate_actions = _eval_with_nan
    model.train()


def test_train_sa_ppo_and_weighted_entropy():
    env = make_vec_env(max_steps=4)
    model = make_model(env=env)
    setup_and_collect(model, env, n_steps=4)

    class _SaPpoStub:
        is_adversarial_enabled = True

        def apply_adversarial_augmentation(self, states, actions, advantages, old_log_probs, clip_range):
            mask = torch.zeros(states.shape[0], device=states.device)
            mask[0] = 1.0
            return states, mask, {"debug/sa_ppo_enabled": 1.0}

        def compute_robust_kl_penalty(self, states_clean, states_adv, actions):
            return 0.1, {"debug/sa_ppo_robust_kl": 0.1}

    model.set_sa_ppo_wrapper(_SaPpoStub())
    model.policy.weighted_entropy = lambda dist: dist.entropy()
    model.train()
