import types
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import distributional_ppo as dppo
from distributional_ppo import (
    DistributionalPPO,
    PopArtController,
    PopArtHoldoutBatch,
    PopArtHoldoutEvaluation,
    _ValuePredictionCacheEntry,
)


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    np.random.seed(123)
    torch.manual_seed(123)


class _CaptureLogger:
    def __init__(self) -> None:
        self.records: dict[str, float] = {}

    def record(self, key: str, value: float | int | str, **_: object) -> None:
        if isinstance(value, (float, int)):
            self.records[key] = float(value)

    def dump(self, *_: object, **__: object) -> None:
        return None

    def get_dir(self) -> None:
        return None


class _DummyCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True


class MinimalBoxEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0, max_steps: int = 2, time_limit: bool = False) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._max_steps = max_steps
        self._step_count = 0
        self._time_limit = time_limit

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        obs = self._rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs = self._rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
        reward = float(np.clip(action, -1.0, 1.0).sum() * 0.0)
        terminated = self._step_count >= self._max_steps
        truncated = False
        info = {"step": self._step_count}
        if self._time_limit and terminated:
            info["time_limit_truncated"] = True
            info["terminal_observation"] = obs.copy()
        return obs, reward, terminated, truncated, info


def _make_vec_env(seed: int = 0, max_steps: int = 2, time_limit: bool = False) -> DummyVecEnv:
    return DummyVecEnv(
        [lambda: MinimalBoxEnv(seed=seed, max_steps=max_steps, time_limit=time_limit)]
    )


def _make_model(env: DummyVecEnv, **overrides: object) -> DistributionalPPO:
    kwargs = {
        "policy": "DistributionalPolicy",
        "env": env,
        "n_steps": 8,
        "batch_size": 4,
        "n_epochs": 1,
        "device": "cpu",
        "verbose": 0,
    }
    kwargs.update(overrides)
    return DistributionalPPO(**kwargs)


def _make_holdout_batch(device: torch.device) -> PopArtHoldoutBatch:
    obs = torch.zeros((2, 4), device=device)
    returns_raw = torch.zeros((2, 1), device=device)
    episode_starts = torch.zeros((2,), dtype=torch.bool, device=device)
    return PopArtHoldoutBatch(
        observations=obs,
        returns_raw=returns_raw,
        episode_starts=episode_starts,
        lstm_states=None,
        mask=None,
    )


def _make_holdout_eval(
    *,
    ev_after: float,
    ev_before: float = 0.8,
    baseline: torch.Tensor | None = None,
    candidate: torch.Tensor | None = None,
) -> PopArtHoldoutEvaluation:
    if baseline is None:
        baseline = torch.tensor([[0.5], [0.5]])
    if candidate is None:
        candidate = baseline.clone()
    target = torch.zeros_like(baseline)
    return PopArtHoldoutEvaluation(
        baseline_raw=baseline,
        candidate_raw=candidate,
        target_raw=target,
        mask=None,
        ev_before=float(ev_before),
        ev_after=float(ev_after),
        clip_fraction_before=0.0,
        clip_fraction_after=0.0,
    )


def _make_cell(value):
    return (lambda: value).__closure__[0]


def _build_time_limit_eval(
    algo: DistributionalPPO,
    select_value_states,
    base_reward_scale: float = 1.0,
):
    code = None
    for const in DistributionalPPO.collect_rollouts.__code__.co_consts:
        if (
            isinstance(const, type(DistributionalPPO.collect_rollouts.__code__))
            and const.co_name == "_evaluate_time_limit_value"
        ):
            code = const
            break
    assert code is not None
    return types.FunctionType(
        code,
        DistributionalPPO.collect_rollouts.__globals__,
        name=code.co_name,
        closure=(
            _make_cell(select_value_states),
            _make_cell(base_reward_scale),
            _make_cell(algo),
        ),
    )


class _PolicyStub:
    def __init__(self, obs_to_tensor, predict_values, forward_states):
        self._obs_to_tensor = obs_to_tensor
        self._predict_values = predict_values
        self._forward_states = forward_states

    def obs_to_tensor(self, obs):
        return self._obs_to_tensor(obs)

    def forward(self, obs_tensor, lstm_states, episode_starts):
        return None, None, None, self._forward_states

    def predict_values(self, obs_tensor, states, episode_starts):
        return self._predict_values(obs_tensor, states, episode_starts)


class _PredictPolicyStub:
    def __init__(self, use_quantiles: bool) -> None:
        self.training = True
        self._use_quantiles = use_quantiles

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True

    def value_quantiles(self, obs, states, episode_starts):
        batch = obs.shape[0]
        return torch.linspace(0.1, 0.3, steps=3).repeat(batch, 1)

    def predict_values(self, obs, states, episode_starts):
        batch = obs.shape[0]
        return torch.full((batch, 1), 2.0)


def _setup_and_collect(model: DistributionalPPO, env: DummyVecEnv, n_steps: int) -> None:
    total_timesteps = int(n_steps * env.num_envs)
    _, callback = model._setup_learn(
        total_timesteps=total_timesteps,
        callback=_DummyCallback(),
        reset_num_timesteps=True,
    )
    model._last_callback = callback
    model._current_progress_remaining = 1.0
    model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=n_steps)


def test_popart_shadow_blocked_reasons() -> None:
    device = torch.device("cpu")
    holdout = _make_holdout_batch(device)

    controller = PopArtController(
        enabled=True,
        mode="shadow",
        min_samples=5,
        warmup_updates=0,
        holdout_loader=lambda: holdout,
    )
    controller._evaluate_holdout = types.MethodType(
        lambda *_args, **_kwargs: _make_holdout_eval(ev_after=0.9),
        controller,
    )
    metrics = controller.evaluate_shadow(
        model=SimpleNamespace(policy=SimpleNamespace(device=device)),
        returns_raw=torch.tensor([0.0]),
        ret_mean=0.0,
        ret_std=1.0,
    )
    assert metrics is not None
    assert metrics.blocked_reason == "min_samples"

    controller = PopArtController(
        enabled=True,
        mode="shadow",
        min_samples=1,
        warmup_updates=5,
        holdout_loader=lambda: holdout,
    )
    controller._evaluate_holdout = types.MethodType(
        lambda *_args, **_kwargs: _make_holdout_eval(ev_after=0.9),
        controller,
    )
    metrics = controller.evaluate_shadow(
        model=SimpleNamespace(policy=SimpleNamespace(device=device)),
        returns_raw=torch.tensor([0.0, 1.0]),
        ret_mean=0.0,
        ret_std=1.0,
    )
    assert metrics is not None
    assert metrics.blocked_reason == "warmup"

    controller = PopArtController(
        enabled=True,
        mode="shadow",
        min_samples=1,
        warmup_updates=0,
        ret_std_band=(0.1, 0.2),
        holdout_loader=lambda: holdout,
    )
    controller._evaluate_holdout = types.MethodType(
        lambda *_args, **_kwargs: _make_holdout_eval(ev_after=0.9),
        controller,
    )
    metrics = controller.evaluate_shadow(
        model=SimpleNamespace(policy=SimpleNamespace(device=device)),
        returns_raw=torch.tensor([0.0, 3.0]),
        ret_mean=0.0,
        ret_std=1.0,
    )
    assert metrics is not None
    assert metrics.blocked_reason == "std_band"

    controller = PopArtController(
        enabled=True,
        mode="shadow",
        min_samples=1,
        warmup_updates=0,
        max_rel_step=0.1,
        ret_std_band=(0.5, 2.0),
        holdout_loader=lambda: holdout,
    )
    controller._evaluate_holdout = types.MethodType(
        lambda *_args, **_kwargs: _make_holdout_eval(ev_after=0.9),
        controller,
    )
    metrics = controller.evaluate_shadow(
        model=SimpleNamespace(policy=SimpleNamespace(device=device)),
        returns_raw=torch.tensor([0.0, 3.0]),
        ret_mean=0.0,
        ret_std=1.0,
    )
    assert metrics is not None
    assert metrics.blocked_reason == "rel_step"

    controller = PopArtController(
        enabled=True,
        mode="shadow",
        min_samples=1,
        warmup_updates=0,
        ev_floor=0.5,
        max_rel_step=1.0,
        ret_std_band=(0.5, 2.0),
        holdout_loader=lambda: holdout,
    )
    controller._evaluate_holdout = types.MethodType(
        lambda *_args, **_kwargs: _make_holdout_eval(ev_after=0.1),
        controller,
    )
    metrics = controller.evaluate_shadow(
        model=SimpleNamespace(policy=SimpleNamespace(device=device)),
        returns_raw=torch.tensor([0.0, 1.0]),
        ret_mean=0.0,
        ret_std=1.0,
    )
    assert metrics is not None
    assert metrics.blocked_reason == "ev_floor"

    controller = PopArtController(
        enabled=True,
        mode="shadow",
        min_samples=1,
        warmup_updates=0,
        ev_floor=0.0,
        max_rel_step=1.0,
        ret_std_band=(0.5, 2.0),
        holdout_loader=lambda: holdout,
    )
    controller._ev_reference = 0.8
    controller._evaluate_holdout = types.MethodType(
        lambda *_args, **_kwargs: _make_holdout_eval(ev_after=0.5),
        controller,
    )
    metrics = controller.evaluate_shadow(
        model=SimpleNamespace(policy=SimpleNamespace(device=device)),
        returns_raw=torch.tensor([0.0, 1.0]),
        ret_mean=0.0,
        ret_std=1.0,
    )
    assert metrics is not None
    assert metrics.blocked_reason == "ev_regress"


def test_popart_shadow_to_live_transition() -> None:
    device = torch.device("cpu")
    holdout = _make_holdout_batch(device)
    controller = PopArtController(
        enabled=True,
        mode="shadow",
        min_samples=1,
        warmup_updates=0,
        gate_patience=1,
        ret_std_band=(0.1, 2.0),
        max_rel_step=1.0,
        ev_floor=-1.0,
        holdout_loader=lambda: holdout,
    )
    controller._evaluate_holdout = types.MethodType(
        lambda *_args, **_kwargs: _make_holdout_eval(ev_after=0.9),
        controller,
    )
    metrics = controller.evaluate_shadow(
        model=SimpleNamespace(policy=SimpleNamespace(device=device)),
        returns_raw=torch.tensor([0.0, 1.0]),
        ret_mean=0.0,
        ret_std=0.5,
    )
    assert metrics is not None
    assert metrics.blocked_reason is None
    assert controller.mode == "live"


def test_time_limit_value_invalid_obs() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.device = torch.device("cpu")
    algo.normalize_returns = True
    algo._ret_std_snapshot = 1.0
    algo._ret_mean_snapshot = 0.0
    algo.value_target_scale = 1.0

    policy = _PolicyStub(
        obs_to_tensor=lambda _obs: (_raise(ValueError("bad obs")), None),
        predict_values=lambda *_args: torch.tensor([[1.0]]),
        forward_states=("s",),
    )
    algo.policy = policy

    eval_fn = _build_time_limit_eval(algo, lambda _idx: (torch.zeros(1, 1),))
    assert eval_fn(0, object()) is None


def test_time_limit_value_non_tensor_obs() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.device = torch.device("cpu")
    algo.normalize_returns = True
    algo._ret_std_snapshot = 1.0
    algo._ret_mean_snapshot = 0.0
    algo.value_target_scale = 1.0

    policy = _PolicyStub(
        obs_to_tensor=lambda _obs: (np.zeros((1, 4), dtype=np.float32), None),
        predict_values=lambda *_args: torch.tensor([[1.0]]),
        forward_states=("s",),
    )
    algo.policy = policy

    eval_fn = _build_time_limit_eval(algo, lambda _idx: (torch.zeros(1, 1),))
    assert eval_fn(0, object()) is None


def test_time_limit_value_missing_states() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.device = torch.device("cpu")
    algo.normalize_returns = True
    algo._ret_std_snapshot = 1.0
    algo._ret_mean_snapshot = 0.0
    algo.value_target_scale = 1.0

    policy = _PolicyStub(
        obs_to_tensor=lambda _obs: (torch.zeros((1, 4)), None),
        predict_values=lambda *_args: torch.tensor([[1.0]]),
        forward_states=("s",),
    )
    algo.policy = policy

    eval_fn = _build_time_limit_eval(algo, lambda _idx: None)
    assert eval_fn(0, object()) is None


def test_time_limit_value_predict_none() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.device = torch.device("cpu")
    algo.normalize_returns = True
    algo._ret_std_snapshot = 1.0
    algo._ret_mean_snapshot = 0.0
    algo.value_target_scale = 1.0

    policy = _PolicyStub(
        obs_to_tensor=lambda _obs: (torch.zeros((1, 4)), None),
        predict_values=lambda *_args: None,
        forward_states=("s",),
    )
    algo.policy = policy

    eval_fn = _build_time_limit_eval(algo, lambda _idx: (torch.zeros(1, 1),))
    assert eval_fn(0, object()) is None


def test_time_limit_value_predict_empty() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.device = torch.device("cpu")
    algo.normalize_returns = True
    algo._ret_std_snapshot = 1.0
    algo._ret_mean_snapshot = 0.0
    algo.value_target_scale = 1.0

    policy = _PolicyStub(
        obs_to_tensor=lambda _obs: (torch.zeros((1, 4)), None),
        predict_values=lambda *_args: torch.tensor([]),
        forward_states=("s",),
    )
    algo.policy = policy

    eval_fn = _build_time_limit_eval(algo, lambda _idx: (torch.zeros(1, 1),))
    assert eval_fn(0, object()) is None


def test_time_limit_value_normalize_returns() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.device = torch.device("cpu")
    algo.normalize_returns = True
    algo._ret_std_snapshot = 0.5
    algo._ret_mean_snapshot = 1.0
    algo.value_target_scale = 2.0

    policy = _PolicyStub(
        obs_to_tensor=lambda _obs: (torch.zeros((1, 4)), None),
        predict_values=lambda *_args: torch.tensor([[2.0]]),
        forward_states=("s",),
    )
    algo.policy = policy

    eval_fn = _build_time_limit_eval(algo, lambda _idx: (torch.zeros(1, 1),))
    assert eval_fn(0, object()) == pytest.approx(1.0)


def test_time_limit_value_no_normalize_with_clamps() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.device = torch.device("cpu")
    algo.normalize_returns = False
    algo._ret_std_snapshot = 1.0
    algo._ret_mean_snapshot = 0.0
    algo.value_target_scale = 2.0
    algo._value_target_scale_effective = 2.0
    algo._value_clip_limit_scaled = 0.5
    algo._value_clip_limit_unscaled = 0.4

    policy = _PolicyStub(
        obs_to_tensor=lambda _obs: (torch.zeros((1, 4)), None),
        predict_values=lambda *_args: torch.tensor([[2.0]]),
        forward_states=("s",),
    )
    algo.policy = policy

    eval_fn = _build_time_limit_eval(algo, lambda _idx: (torch.zeros(1, 1),), base_reward_scale=2.0)
    assert eval_fn(0, object()) == pytest.approx(0.2)


def test_refresh_value_prediction_quantile_clipped() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.device = torch.device("cpu")
    algo.normalize_returns = True
    algo._ret_std_snapshot = 2.0
    algo._ret_mean_snapshot = 1.0
    algo._value_norm_clip_min = -2.0
    algo._value_norm_clip_max = 2.0
    algo._value_target_scale_effective = 1.0
    algo._value_clip_limit_scaled = None
    algo._use_quantile_value = True
    algo.policy = _PredictPolicyStub(use_quantiles=True)

    entry = _ValuePredictionCacheEntry(
        observations=torch.zeros((2, 4)),
        lstm_states=(torch.zeros((1, 1, 1)),),
        episode_starts=torch.zeros((2,), dtype=torch.bool),
        valid_indices=torch.tensor([1]),
        base_scale=1.0,
        old_values_raw=torch.tensor([0.0, 0.5]),
        mask_values=torch.tensor([1.0, 0.0]),
    )

    preds, _, masks, _ = algo._refresh_value_prediction_tensors(
        primary_cache=[entry],
        primary_predictions=[],
        reserve_cache=[],
        reserve_predictions=[],
        primary_weights=[None],
        reserve_weights=[],
        clip_range_vf_value=0.5,
        ret_mu_tensor=torch.tensor(1.0),
        ret_std_tensor=torch.tensor(2.0),
    )
    assert preds[0].shape == (1, 1)
    assert masks[0] is not None


def test_refresh_value_prediction_value_clipped_no_norm() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.device = torch.device("cpu")
    algo.normalize_returns = False
    algo._ret_std_snapshot = 1.0
    algo._ret_mean_snapshot = 0.0
    algo._value_norm_clip_min = -2.0
    algo._value_norm_clip_max = 2.0
    algo._value_target_scale_effective = 2.0
    algo._value_clip_limit_scaled = 1.0
    algo._use_quantile_value = False
    algo.policy = _PredictPolicyStub(use_quantiles=False)

    entry = _ValuePredictionCacheEntry(
        observations=torch.zeros((2, 4)),
        lstm_states=None,
        episode_starts=torch.zeros((2,), dtype=torch.bool),
        valid_indices=None,
        base_scale=2.0,
        old_values_raw=torch.tensor([0.0, 0.0]),
        mask_values=None,
    )

    preds, _, masks, _ = algo._refresh_value_prediction_tensors(
        primary_cache=[entry],
        primary_predictions=[],
        reserve_cache=[],
        reserve_predictions=[],
        primary_weights=[torch.tensor([[1.0], [0.0]])],
        reserve_weights=[],
        clip_range_vf_value=None,
        ret_mu_tensor=torch.tensor(0.0),
        ret_std_tensor=torch.tensor(1.0),
    )
    assert preds[0].shape == (2, 1)
    assert masks[0] is not None


def test_twin_critics_vf_clipping_loss_quantile_mean_and_variance() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.normalize_returns = False
    algo.value_target_scale = 1.0
    algo._value_target_scale_base = 1.0
    algo._value_target_scale_effective = 1.0
    algo._value_clip_limit_scaled = 0.5
    algo.distributional_vf_clip_variance_factor = 1.5
    algo._quantile_huber_kappa = 1.0
    algo._value_norm_clip_min = -2.0
    algo._value_norm_clip_max = 2.0

    class _Policy:
        _use_twin_critics = True
        _use_quantile_value_head = True
        _value_type = None
        quantile_levels = torch.linspace(0.1, 0.9, 3)
        num_atoms = 3

        def _get_value_logits(self, _latent):
            return [0.1, 0.2, 0.3]

        def _get_value_logits_2(self, _latent):
            return [0.2, 0.1, 0.0]

    algo.policy = _Policy()

    latent_vf = torch.zeros((2, 3))
    targets = torch.zeros((2, 1))
    old_q1 = torch.zeros((2, 3))
    old_q2 = torch.zeros((2, 3))
    result = algo._twin_critics_vf_clipping_loss(
        latent_vf,
        targets,
        old_q1,
        old_q2,
        clip_delta=0.5,
        mode="mean_and_variance",
        return_full=True,
    )
    assert len(result) == 6


def test_twin_critics_vf_clipping_loss_quantile_normalize_returns() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.normalize_returns = True
    algo.value_target_scale = 1.0
    algo._value_target_scale_base = 1.0
    algo._value_target_scale_effective = 1.0
    algo._ret_rms_effective_mean_tensor = torch.tensor(0.0)
    algo._ret_rms_effective_std_tensor = torch.tensor(1.0)
    algo._ret_mean_snapshot = 0.0
    algo._ret_std_snapshot = 1.0
    algo._quantile_huber_kappa = 1.0
    algo._value_norm_clip_min = -2.0
    algo._value_norm_clip_max = 2.0
    algo.distributional_vf_clip_variance_factor = 1.2

    class _Policy:
        _use_twin_critics = True
        _use_quantile_value_head = True
        _value_type = None
        quantile_levels = torch.linspace(0.1, 0.9, 3)

        def _get_value_logits(self, _latent):
            return torch.zeros((2, 3))

        def _get_value_logits_2(self, _latent):
            return torch.ones((2, 3))

    algo.policy = _Policy()

    latent_vf = torch.zeros((2, 3))
    targets = torch.zeros((2, 1))
    old_q1 = torch.zeros((2, 3))
    old_q2 = torch.zeros((2, 3))
    result = algo._twin_critics_vf_clipping_loss(
        latent_vf,
        targets,
        old_q1,
        old_q2,
        clip_delta=0.1,
        mode="per_quantile",
        return_full=False,
    )
    assert len(result) == 4


def test_twin_critics_vf_clipping_loss_categorical_atoms_generated() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.normalize_returns = True
    algo.value_target_scale = 1.0
    algo._ret_rms_effective_mean_tensor = torch.tensor(0.0)
    algo._ret_rms_effective_std_tensor = torch.tensor(1.0)
    algo._ret_mean_snapshot = 0.0
    algo._ret_std_snapshot = 1.0
    algo._value_norm_clip_min = -2.0
    algo._value_norm_clip_max = 2.0
    algo._value_target_scale_base = 1.0
    algo._value_target_scale_effective = 1.0
    algo.distributional_vf_clip_variance_factor = 1.2

    class _Policy:
        _use_twin_critics = True
        _use_quantile_value_head = True
        _value_type = "categorical"
        atoms = None
        num_atoms = 3

        def _get_value_logits(self, _latent):
            return torch.zeros((2, 3))

        def _get_value_logits_2(self, _latent):
            return torch.ones((2, 3))

    algo.policy = _Policy()

    latent_vf = torch.zeros((2, 3))
    targets = torch.zeros((2, 1))
    old_q1 = torch.zeros((2, 3))
    old_q2 = torch.zeros((2, 3))
    old_probs = torch.full((2, 3), 1.0 / 3.0)
    target_distribution = torch.full((2, 3), 1.0 / 3.0)
    result = algo._twin_critics_vf_clipping_loss(
        latent_vf,
        targets,
        old_q1,
        old_q2,
        clip_delta=0.1,
        old_probs_critic1=old_probs,
        old_probs_critic2=old_probs,
        target_distribution=target_distribution,
        mode="mean_only",
        return_full=False,
    )
    assert len(result) == 4


def test_twin_critics_vf_clipping_loss_invalid_mode_raises() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.normalize_returns = False
    algo.value_target_scale = 1.0
    algo._value_target_scale_base = 1.0
    algo._value_target_scale_effective = 1.0
    algo._value_clip_limit_scaled = None
    algo._quantile_huber_kappa = 1.0
    algo._value_norm_clip_min = -2.0
    algo._value_norm_clip_max = 2.0

    class _Policy:
        _use_twin_critics = True
        _use_quantile_value_head = True
        _value_type = None
        quantile_levels = torch.linspace(0.1, 0.9, 3)

        def _get_value_logits(self, _latent):
            return torch.zeros((2, 3))

        def _get_value_logits_2(self, _latent):
            return torch.ones((2, 3))

    algo.policy = _Policy()

    latent_vf = torch.zeros((2, 3))
    targets = torch.zeros((2, 1))
    old_q1 = torch.zeros((2, 3))
    old_q2 = torch.zeros((2, 3))
    with pytest.raises(ValueError, match="distributional_vf_clip_mode"):
        algo._twin_critics_vf_clipping_loss(
            latent_vf,
            targets,
            old_q1,
            old_q2,
            clip_delta=0.1,
            mode="invalid",
        )


def test_get_optimizer_kwargs_vgs_enabled() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._optimizer_kwargs = {}
    algo._variance_gradient_scaler = SimpleNamespace(enabled=True)

    class AdaptiveUPGD:
        pass

    algo._get_optimizer_class = lambda: AdaptiveUPGD
    kwargs = algo._get_optimizer_kwargs()
    assert kwargs["adaptive_noise"] is True
    assert kwargs["sigma"] == pytest.approx(0.0005)


def test_update_learning_rate_with_external_scheduler() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._current_progress_remaining = 0.5
    algo._base_lr_schedule = lambda progress: 1e-4
    algo._logger = _CaptureLogger()
    algo._enforce_optimizer_lr_bounds = lambda **_: None
    algo.policy = SimpleNamespace(lr_scheduler=object())
    optimizer = SimpleNamespace(param_groups=[{"lr": 1e-4}])
    algo._update_learning_rate(optimizer)
    assert "train/learning_rate" in algo._logger.records


def test_update_learning_rate_no_scheduler() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._current_progress_remaining = 0.5
    algo.lr_schedule = lambda progress: 1e-3
    algo._kl_min_lr = 5e-4
    algo._logger = _CaptureLogger()
    algo._enforce_optimizer_lr_bounds = lambda **_: None
    algo.lr_scheduler = None
    algo.policy = SimpleNamespace(lr_scheduler=None)
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.0, "_lr_scale": 2.0}])
    algo._update_learning_rate(optimizer)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-3)


def test_configure_gradient_accumulation_paths() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.batch_size = 8
    algo._configure_gradient_accumulation(None, None)
    assert algo._microbatch_size == 8
    assert algo._grad_accumulation_steps == 1

    algo._configure_gradient_accumulation(4, None)
    assert algo._microbatch_size == 4
    assert algo._grad_accumulation_steps == 2

    algo._configure_gradient_accumulation(None, 4)
    assert algo._microbatch_size == 2
    assert algo._grad_accumulation_steps == 4

    with pytest.raises(ValueError, match="microbatch_size"):
        algo._configure_gradient_accumulation(3, None)


def test_configure_loss_head_weights_paths() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._logger = _CaptureLogger()
    algo.policy = SimpleNamespace(set_loss_head_weights=lambda *_: None)

    algo._configure_loss_head_weights(None)
    assert algo._loss_head_weights is None

    algo._configure_loss_head_weights({"head_a": True, "head_b": 0.5, "head_c": None})
    assert algo._loss_head_weights["head_a"] == 1.0
    assert algo._loss_head_weights["head_b"] == pytest.approx(0.5)


def test_activate_return_scale_snapshot_branches() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.normalize_returns = False
    algo._ret_mean_value = 0.1
    algo._ret_std_value = 0.2
    algo._value_scale_std_floor = 1e-6
    algo._activate_return_scale_snapshot()
    assert algo._pending_rms is None

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.normalize_returns = True
    algo._ret_mean_value = 0.1
    algo._ret_std_value = 0.2
    algo._value_scale_std_floor = 1e-6
    algo._value_scale_updates_enabled = False
    algo._activate_return_scale_snapshot()
    assert algo._pending_rms is None

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.normalize_returns = True
    algo._ret_mean_value = 0.1
    algo._ret_std_value = 0.2
    algo._value_scale_std_floor = 1e-6
    algo._value_scale_updates_enabled = True
    algo._value_scale_frozen = True
    algo._activate_return_scale_snapshot()
    assert algo._pending_rms is None

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.normalize_returns = True
    algo._ret_mean_value = 0.1
    algo._ret_std_value = 0.2
    algo._value_scale_std_floor = 1e-6
    algo._value_scale_updates_enabled = True
    algo._value_scale_frozen = False
    algo._activate_return_scale_snapshot()
    assert algo._pending_rms is not None


def test_getstate_saves_vgs_state() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._variance_gradient_scaler = SimpleNamespace(state_dict=lambda: {"step": 1})
    algo._setup_complete = True
    state = algo.__getstate__()
    assert state["_vgs_saved_state"] == {"step": 1}
    assert state["_setup_complete"] is False
    assert "_variance_gradient_scaler" not in state


def test_restore_optimizer_state_paths() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._restore_optimizer_state(None)

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.policy = None
    algo._restore_optimizer_state({"state": {}})

    algo = DistributionalPPO.__new__(DistributionalPPO)
    optimizer = SimpleNamespace(load_state_dict=lambda _state: None)
    algo.policy = SimpleNamespace(optimizer=optimizer)
    algo._restore_optimizer_state({"state": {}})


def test_restore_vgs_state_paths() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._variance_gradient_scaler = None
    algo._restore_vgs_state({"step_count": 5})
    assert algo._vgs_saved_state_for_restore["step_count"] == 5

    scaler = SimpleNamespace(load_state_dict=lambda _state: None, _step_count=0)
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._variance_gradient_scaler = scaler
    algo._restore_vgs_state({"step_count": 2})


def test_entropy_schedule_plateau_detection() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.ent_coef_decay_steps = 5
    algo.entropy_plateau_window = 2
    algo.entropy_plateau_tolerance = 0.0
    algo.entropy_plateau_min_updates = 0
    algo._entropy_window = deque(maxlen=2)
    algo._entropy_plateau = False
    algo._entropy_decay_start_update = None
    algo._last_entropy_slope = 0.0
    algo._maybe_update_entropy_schedule(0, 1.0)
    algo._maybe_update_entropy_schedule(1, 1.0)
    assert algo._entropy_plateau is True


def test_set_parameters_roundtrip_with_optimizer_and_vgs() -> None:
    env = _make_vec_env(seed=7)
    model = _make_model(env)
    params = model.get_parameters(include_optimizer=True)
    model.set_parameters(params)
    env.close()


def test_train_branches_kl_cvar_popart_vgs() -> None:
    env = _make_vec_env(seed=11, max_steps=4, time_limit=True)
    model = _make_model(
        env,
        target_kl=1e-8,
        kl_absolute_stop_factor=1.0,
        kl_ema_updates=2,
        kl_ema_alpha=None,
        vf_clip_warmup_updates=10,
        cvar_use_constraint=True,
        cvar_use_predicted_for_dual=True,
        cvar_use_penalty=True,
        ent_coef_decay_steps=1,
        ent_coef_plateau_window=2,
    )
    model._cvar_predicted_last_unit = 0.0
    model._cvar_predicted_last_raw = 0.0
    model._popart_controller = PopArtController(
        enabled=True,
        mode="shadow",
        min_samples=1,
        warmup_updates=0,
        holdout_loader=lambda: _make_holdout_batch(model.policy.device),
    )
    _setup_and_collect(model, env, n_steps=8)
    model.train()
    env.close()


def test_train_branches_cvar_penalty_disabled() -> None:
    env = _make_vec_env(seed=13, max_steps=8)
    model = _make_model(
        env,
        cvar_use_penalty=False,
        ent_coef_decay_steps=0,
        vf_clip_warmup_updates=5,
    )
    _setup_and_collect(model, env, n_steps=8)
    model.train()
    env.close()


def _raise(exc: Exception):
    raise exc
