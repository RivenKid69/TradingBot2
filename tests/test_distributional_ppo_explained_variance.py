import math
import types
from typing import Any

import numpy as np
import pytest
import torch

import test_distributional_ppo_raw_outliers  # noqa: F401  # ensure RL stubs are installed
import distributional_ppo as distributional_ppo_module

from distributional_ppo import (
    DistributionalPPO,
    PopArtController,
    PopArtHoldoutBatch,
    safe_explained_variance,
)


def test_ev_builder_returns_none_when_no_batches() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)

    y_true, y_pred, y_true_raw, weights, group_keys = algo._build_explained_variance_tensors(
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    assert y_true is None
    assert y_pred is None
    assert y_true_raw is None
    assert weights is None
    assert group_keys is None


def test_ev_builder_uses_reserve_pairs_without_length_mismatch() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)

    reserve_true = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32)
    reserve_pred = torch.tensor([[0.0], [0.15], [0.25]], dtype=torch.float32)
    reserve_raw = torch.tensor([[1.0], [1.1], [1.2]], dtype=torch.float32)
    reserve_weights = torch.tensor([[1.0], [0.5], [1.0]], dtype=torch.float32)

    reserve_keys = [["g0", "g1", "g2"]]

    y_true, y_pred, y_true_raw, weights, group_keys = algo._build_explained_variance_tensors(
        [],
        [],
        [],
        [],
        [],
        [reserve_true],
        [reserve_pred],
        [reserve_raw],
        [reserve_weights],
        reserve_keys,
    )

    assert y_true is not None and y_pred is not None
    assert y_true.shape == reserve_true.shape == y_pred.shape
    assert torch.equal(y_true, reserve_true)
    assert torch.equal(y_pred, reserve_pred)

    assert y_true_raw is not None
    assert torch.equal(y_true_raw, reserve_raw)

    assert weights is not None
    assert torch.equal(weights, reserve_weights)

    assert group_keys is not None
    assert group_keys == reserve_keys[0]

    ev = safe_explained_variance(
        y_true.detach().cpu().numpy(),
        y_pred.detach().cpu().numpy(),
        weights.detach().cpu().numpy(),
    )

    assert math.isfinite(ev)


def test_explained_variance_fallback_uses_raw_targets() -> None:
    class _DummyLogger:
        def __init__(self) -> None:
            self.records: dict[str, list[float]] = {}

        def record(self, key: str, value: float) -> None:
            self.records.setdefault(key, []).append(value)

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.normalize_returns = True
    algo._ret_mean_snapshot = 0.75
    algo._ret_std_snapshot = 1.5
    algo.value_target_scale = 1.0
    algo._value_target_scale_effective = 1.0
    algo.logger = _DummyLogger()

    y_true_norm = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32)
    y_pred_norm = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)
    y_true_raw = torch.tensor([[0.5], [1.0], [1.5]], dtype=torch.float32)
    mask = torch.ones_like(y_true_norm)

    ev_value, y_true_eval, y_pred_eval, metrics = algo._compute_explained_variance_metric(
        y_true_norm,
        y_pred_norm,
        mask_tensor=mask,
        y_true_tensor_raw=y_true_raw,
    )

    assert ev_value is not None
    assert math.isfinite(ev_value)
    assert y_true_eval is not None and y_pred_eval is not None
    assert y_true_eval.shape == torch.Size([3])
    assert y_pred_eval.shape == torch.Size([3])
    assert algo.logger.records["train/value_explained_variance_fallback"] == [1.0]

    # Raw fallback should evaluate the metric in the same units as production code.
    y_pred_raw = algo._to_raw_returns(y_pred_norm)
    expected_ev = safe_explained_variance(
        y_true_raw.numpy(),
        y_pred_raw.detach().cpu().numpy(),
        mask.numpy(),
    )
    assert expected_ev == pytest.approx(-3.0)
    assert ev_value == pytest.approx(expected_ev)
    assert metrics["n_samples"] == pytest.approx(3.0)
    expected_bias = float(np.mean((y_true_raw - y_pred_raw).numpy()))
    assert metrics["bias"] == pytest.approx(expected_bias)


def test_explained_variance_fallback_recovers_from_clipped_targets() -> None:
    class _DummyLogger:
        def __init__(self) -> None:
            self.records: dict[str, list[float]] = {}

        def record(self, key: str, value: float) -> None:
            self.records.setdefault(key, []).append(value)

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.normalize_returns = False
    algo.value_target_scale = 1.5
    algo._value_target_scale_effective = 0.75
    algo.logger = _DummyLogger()

    # Normalised targets are clipped to a constant, but raw returns preserve variance.
    y_true_norm = torch.zeros((4, 1), dtype=torch.float32)
    y_pred_norm = torch.tensor([[0.0], [0.5], [1.0], [1.5]], dtype=torch.float32)
    y_true_raw = torch.arange(4, dtype=torch.float32).view(-1, 1)
    mask = torch.ones_like(y_true_norm)

    ev_value, _, _, _ = algo._compute_explained_variance_metric(
        y_true_norm,
        y_pred_norm,
        mask_tensor=mask,
        y_true_tensor_raw=y_true_raw,
    )

    assert ev_value is not None
    assert math.isfinite(ev_value)
    y_pred_raw = algo._to_raw_returns(y_pred_norm)
    expected_ev = safe_explained_variance(
        y_true_raw.numpy(),
        y_pred_raw.detach().cpu().numpy(),
        mask.numpy(),
    )
    assert expected_ev == pytest.approx(1.0)
    assert ev_value == pytest.approx(expected_ev)
    assert algo.logger.records["train/value_explained_variance_fallback"] == [1.0]


def test_explained_variance_metric_retains_primary_path_with_small_variance() -> None:
    class _DummyLogger:
        def __init__(self) -> None:
            self.records: dict[str, list[float]] = {}

        def record(self, key: str, value: float) -> None:
            self.records.setdefault(key, []).append(value)

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.normalize_returns = False
    algo.value_target_scale = 1.0
    algo._value_target_scale_effective = 1.0
    algo.logger = _DummyLogger()

    y_true_norm = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    y_pred_norm = torch.tensor([[0.25], [0.75]], dtype=torch.float32)
    mask = torch.tensor([[1.0e12], [1.0]], dtype=torch.float32)

    ev_value, _, _, _ = algo._compute_explained_variance_metric(
        y_true_norm,
        y_pred_norm,
        mask_tensor=mask,
    )

    assert ev_value is not None
    expected_ev = safe_explained_variance(
        y_true_norm.numpy(),
        y_pred_norm.numpy(),
        mask.numpy(),
    )
    assert math.isfinite(expected_ev)
    assert ev_value == pytest.approx(expected_ev)
    assert "train/value_explained_variance_fallback" not in algo.logger.records


def test_explained_variance_metric_falls_back_with_empty_mask() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)

    y_true_norm = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    y_pred_norm = torch.tensor([[0.5], [1.5]], dtype=torch.float32)
    mask = torch.zeros_like(y_true_norm)

    ev_value, y_true_eval, y_pred_eval, metrics = algo._compute_explained_variance_metric(
        y_true_norm,
        y_pred_norm,
        mask_tensor=mask,
    )

    expected = safe_explained_variance(
        y_true_norm.numpy().reshape(-1),
        y_pred_norm.numpy().reshape(-1),
        None,
    )

    assert ev_value == pytest.approx(expected)
    assert y_true_eval.numel() == y_true_norm.numel()
    assert y_pred_eval.numel() == y_pred_norm.numel()
    assert metrics["n_samples"] == pytest.approx(float(y_true_norm.numel()))


def test_explained_variance_metric_sanitizes_nan_mask() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)

    y_true_norm = torch.tensor([[1.0], [3.0], [2.0]], dtype=torch.float32)
    y_pred_norm = torch.tensor([[0.5], [2.5], [1.5]], dtype=torch.float32)
    mask = torch.tensor([[float("nan")], [0.5], [0.2]], dtype=torch.float32)

    ev_value, y_true_eval, y_pred_eval, _ = algo._compute_explained_variance_metric(
        y_true_norm,
        y_pred_norm,
        mask_tensor=mask,
    )

    expected = safe_explained_variance(
        np.array([3.0, 2.0], dtype=np.float32),
        np.array([2.5, 1.5], dtype=np.float32),
        np.array([0.5, 0.2], dtype=np.float32),
    )

    assert ev_value == pytest.approx(expected)
    assert y_true_eval.numel() == 2
    assert y_pred_eval.numel() == 2


def test_explained_variance_metric_returns_none_with_empty_inputs() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)

    y_true_norm = torch.tensor([], dtype=torch.float32)
    y_pred_norm = torch.tensor([], dtype=torch.float32)

    ev_value, y_true_eval, y_pred_eval, _ = algo._compute_explained_variance_metric(
        y_true_norm,
        y_pred_norm,
    )

    assert ev_value is None
    assert y_true_eval.numel() == 0
    assert y_pred_eval.numel() == 0


def test_explained_variance_metric_returns_none_for_degenerate_variance() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)

    y_true_norm = torch.ones((4, 1), dtype=torch.float32)
    y_pred_norm = torch.zeros_like(y_true_norm)
    mask = torch.ones_like(y_true_norm)

    ev_value, y_true_eval, y_pred_eval, _ = algo._compute_explained_variance_metric(
        y_true_norm,
        y_pred_norm,
        mask_tensor=mask,
    )

    assert ev_value is None
    assert y_true_eval.numel() == 4
    assert y_pred_eval.numel() == 4


def test_explained_variance_logging_marks_availability_when_metric_present() -> None:
    class _DummyLogger:
        def __init__(self) -> None:
            self.records: dict[str, list[float]] = {}

        def record(self, key: str, value: float) -> None:
            self.records.setdefault(key, []).append(float(value))

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.logger = _DummyLogger()

    algo._record_explained_variance_logs(0.25, grouped_mean_unweighted=0.1, grouped_median=0.2)

    assert algo.logger.records["train/explained_variance_available"] == [1.0]
    assert algo.logger.records["train/explained_variance"] == [0.25]
    assert algo.logger.records["train/ev/global"] == [0.25]
    assert algo.logger.records["train/ev/mean_grouped_unweighted"] == [0.1]
    assert algo.logger.records["train/ev/mean_grouped"] == [0.1]
    assert algo.logger.records["train/ev/median_grouped"] == [0.2]


def test_explained_variance_logging_marks_absence_when_metric_missing() -> None:
    class _DummyLogger:
        def __init__(self) -> None:
            self.records: dict[str, list[float]] = {}

        def record(self, key: str, value: float) -> None:
            self.records.setdefault(key, []).append(float(value))

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.logger = _DummyLogger()

    algo._record_explained_variance_logs(None)

    assert algo.logger.records["train/explained_variance_available"] == [0.0]
    assert "train/explained_variance" not in algo.logger.records
    assert "train/ev/global" not in algo.logger.records


def test_quantile_holdout_uses_mean_for_explained_variance() -> None:
    controller = PopArtController(enabled=True)

    quantiles = torch.tensor([[0.0, 1.0], [1.0, 3.0]], dtype=torch.float32)
    holdout = PopArtHoldoutBatch(
        observations=torch.zeros((2, 1), dtype=torch.float32),
        returns_raw=torch.tensor([[2.0], [4.0]], dtype=torch.float32),
        episode_starts=torch.zeros((2, 1), dtype=torch.float32),
        lstm_states=None,
        mask=torch.ones((2, 1), dtype=torch.float32),
    )

    old_mean = 0.5
    old_std = 2.0
    new_mean = 1.0
    new_std = 3.0

    class _DummyPolicy:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.recurrent_initial_state = None
            self.training = False

        def eval(self) -> None:
            self.training = False

        def train(self) -> None:
            self.training = True

    class _DummyModel:
        def __init__(self, quantiles_tensor: torch.Tensor) -> None:
            self.policy = _DummyPolicy()
            self._use_quantile_value = True
            self.normalize_returns = True
            self._ret_mean_snapshot = old_mean
            self._ret_std_snapshot = old_std
            self._quantiles = quantiles_tensor

        def _policy_value_outputs(self, *_: Any, **__: Any) -> torch.Tensor:  # type: ignore[override]
            return self._quantiles

        def _to_raw_returns(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor * self._ret_std_snapshot + self._ret_mean_snapshot

    model = _DummyModel(quantiles)

    eval_result = controller._evaluate_holdout(
        model=model,
        holdout=holdout,
        old_mean=old_mean,
        old_std=old_std,
        new_mean=new_mean,
        new_std=new_std,
    )

    scale = old_std / max(new_std, 1e-6)
    shift = (old_mean - new_mean) / max(new_std, 1e-6)
    candidate_quantiles_norm = quantiles * scale + shift
    candidate_norm_mean = candidate_quantiles_norm.mean(dim=-1, keepdim=True)
    candidate_raw_mean = candidate_norm_mean * new_std + new_mean

    assert eval_result.candidate_raw.shape == torch.Size([2, 1])
    assert torch.allclose(eval_result.candidate_raw, candidate_raw_mean)

    expected_ev = safe_explained_variance(
        holdout.returns_raw.numpy().reshape(-1),
        candidate_raw_mean.numpy().reshape(-1),
        holdout.mask.numpy().reshape(-1),
    )

    assert eval_result.ev_after == pytest.approx(expected_ev)


def test_vf_coef_scales_down_when_explained_variance_is_bad() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._vf_coef_target = 0.8
    algo._vf_bad_explained_scale = 0.4
    algo._vf_bad_explained_floor = 0.2

    # Explained variance below the floor should trigger scaling.
    algo._last_explained_variance = 0.1
    reduced = algo._compute_vf_coef_value(update_index=0)
    assert reduced == pytest.approx(0.32)

    # Healthy variance should keep the base coefficient.
    algo._last_explained_variance = 0.5
    assert algo._compute_vf_coef_value(update_index=0) == pytest.approx(0.8)

    # Scaling should not push the coefficient beneath the configured floor.
    algo._vf_bad_explained_scale = 0.0
    algo._last_explained_variance = -0.5
    assert algo._compute_vf_coef_value(update_index=0) == pytest.approx(0.2)


class _MaskCaptureLogger:
    def __init__(self) -> None:
        self.records: dict[str, list[Any]] = {}

    def record(self, key: str, value: Any, **_: object) -> None:
        if isinstance(value, (int, float)):
            self.records.setdefault(key, []).append(float(value))
        else:
            self.records.setdefault(key, []).append(value)


class _MaskGaussianDistribution:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.distribution = torch.distributions.Normal(mean, std)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1, keepdim=True)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1, keepdim=True)


class _MaskPolicyStub(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(1, 1, bias=False)
        self.value_head = torch.nn.Linear(1, 5, bias=False)
        self.log_std = torch.nn.Parameter(torch.zeros(1))
        self.uses_quantile_value_head = False
        self.quantile_huber_kappa = 1.0
        self.v_min = -2.0
        self.v_max = 2.0
        self.num_atoms = 5
        self.atoms = torch.linspace(self.v_min, self.v_max, steps=self.num_atoms)
        self.device = torch.device("cpu")
        self.last_value_logits = None
        self._last_value_logits = None
        self.last_value_quantiles = None
        self._last_value_quantiles = None
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.optimizer_scheduler = None
        with torch.no_grad():
            self.actor.weight.fill_(0.05)
            self.value_head.weight.copy_(
                torch.tensor(
                    [[0.4], [0.2], [0.0], [-0.2], [-0.4]], dtype=torch.float32
                )
            )

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        lstm_states: object,
        episode_starts: torch.Tensor,
        *,
        actions_raw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del lstm_states, episode_starts, actions_raw
        obs_fp32 = obs.to(dtype=torch.float32)
        mean = self.actor(obs_fp32)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        value_logits = self.value_head(obs_fp32)
        self.last_value_logits = value_logits
        self._last_value_logits = value_logits
        return value_logits.mean(dim=1, keepdim=True), log_prob, entropy

    def get_distribution(
        self,
        obs: torch.Tensor,
        actor_states: object,
        episode_starts: torch.Tensor,
    ) -> _MaskGaussianDistribution:
        del actor_states, episode_starts
        obs_fp32 = obs.to(dtype=torch.float32)
        mean = self.actor(obs_fp32)
        std = torch.exp(self.log_std)
        return _MaskGaussianDistribution(mean, std)

    def predict_values(
        self,
        obs: torch.Tensor,
        lstm_states: object,
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        del lstm_states, episode_starts
        logits = self.value_head(obs.to(dtype=torch.float32))
        probs = torch.softmax(logits, dim=1)
        return (probs * self.atoms.to(device=probs.device, dtype=probs.dtype)).sum(dim=1, keepdim=True)

    def _log_prob_raw_only(self, dist: _MaskGaussianDistribution, actions: torch.Tensor) -> torch.Tensor:
        raw = dist.distribution.log_prob(actions)
        if raw.ndim > 1:
            raw = raw.sum(dim=-1, keepdim=True)
        return raw

    def update_atoms(self, v_min: float, v_max: float) -> None:
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.atoms = torch.linspace(self.v_min, self.v_max, steps=self.num_atoms)


class _MaskRolloutBuffer:
    def __init__(self, mask: torch.Tensor) -> None:
        self._mask = mask.clone()
        base_obs = torch.arange(mask.shape[0], dtype=torch.float32).unsqueeze(-1)
        self._observations = base_obs
        self._actions = torch.zeros_like(base_obs)
        self._advantages = torch.tensor([[0.2], [-0.1], [0.3], [-0.05]], dtype=torch.float32)
        self._returns = torch.tensor([[0.4], [0.3], [0.2], [0.1]], dtype=torch.float32)
        self._old_values = torch.zeros_like(self._returns)
        self._old_log_prob = torch.zeros_like(self._returns)
        self._sample_indices = torch.arange(self._returns.shape[0], dtype=torch.int64)
        self.rewards = self._returns.clone()
        self.returns = self._returns.clone()
        self.buffer_size = int(self._returns.shape[0])
        self.n_envs = 1

    def _make_sample(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            observations=self._observations.clone(),
            actions=self._actions.clone(),
            actions_raw=self._actions.clone(),
            lstm_states=None,
            episode_starts=torch.zeros(self._returns.shape[0], dtype=torch.bool),
            advantages=self._advantages.clone(),
            returns=self._returns.clone(),
            old_values=self._old_values.clone(),
            old_log_prob=self._old_log_prob.clone(),
            old_log_prob_raw=self._old_log_prob.clone(),
            mask=self._mask.clone(),
            sample_indices=self._sample_indices.clone(),
        )

    def get(self, batch_size: int):  # noqa: D401 - simple generator interface
        del batch_size
        yield self._make_sample()


def test_ev_reserve_path_respects_training_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = _MaskCaptureLogger()
    policy = _MaskPolicyStub()
    captured_masks: list[Any] = []

    def _fake_super_init(self, policy: object, env: object, *args: object, **kwargs: object) -> None:
        del env, args, kwargs
        self.logger = logger
        self._logger = logger
        self.policy = policy  # type: ignore[assignment]
        self.device = torch.device("cpu")
        self.policy.to(self.device)
        self.n_steps = 4
        self.n_envs = 1
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.n_epochs = 1
        self.batch_size = 4
        self.lr_schedule = lambda _: 0.001
        self.normalize_returns = False
        self._value_scale_updates_enabled = True
        self.ent_coef = 0.01
        self.action_space = types.SimpleNamespace()
        self.observation_space = types.SimpleNamespace()
        self._grad_accumulation_steps = 1
        self._microbatch_size = 4
        self._critic_ce_normalizer = 1.0
        self._current_progress_remaining = 1.0
        self.max_grad_norm = 0.5
        self.clip_range = 0.2
        self._n_updates = 0
        self._global_update_step = 0
        self._global_step = 0
        self.target_kl = None
        self.kl_early_stop = False
        self.kl_exceed_stop_fraction = 0.0
        self._kl_consec_minibatches = 0

    monkeypatch.setattr(
        distributional_ppo_module.RecurrentPPO,
        "__init__",
        _fake_super_init,
        raising=False,
    )
    monkeypatch.setattr(DistributionalPPO, "_rebuild_scheduler_if_needed", lambda self: None)
    monkeypatch.setattr(DistributionalPPO, "_ensure_score_action_space", lambda self: None)
    monkeypatch.setattr(DistributionalPPO, "_configure_loss_head_weights", lambda self, *a, **k: None)
    monkeypatch.setattr(DistributionalPPO, "_configure_gradient_accumulation", lambda self, **_: None)

    original_compute = DistributionalPPO._compute_explained_variance_metric

    def _capture_metric(self, *args: object, **kwargs: object):
        captured_masks.append(kwargs.get("mask_tensor"))
        return original_compute(self, *args, **kwargs)

    monkeypatch.setattr(DistributionalPPO, "_compute_explained_variance_metric", _capture_metric)

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.logger = logger
    algo._logger = logger

    DistributionalPPO.__init__(
        algo,
        policy=policy,
        env=object(),
        value_scale_max_rel_step=0.1,
    )

    mask = torch.tensor([[1.0], [0.0], [0.4], [0.0]], dtype=torch.float32)
    algo.rollout_buffer = _MaskRolloutBuffer(mask)

    algo.train()

    mask_tensors = [
        tensor for tensor in captured_masks if isinstance(tensor, torch.Tensor) and tensor.numel() > 0
    ]
    assert mask_tensors, "Expected explained variance mask to be captured"
    last_mask = mask_tensors[-1].flatten()
    assert last_mask.numel() == 2
    assert last_mask.tolist() == pytest.approx([1.0, 0.4])
