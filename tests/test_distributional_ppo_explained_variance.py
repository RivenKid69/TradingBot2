import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import test_distributional_ppo_raw_outliers  # noqa: F401  # ensure RL stubs are installed

from distributional_ppo import (
    DistributionalPPO,
    PopArtController,
    PopArtHoldoutBatch,
    safe_explained_variance,
)


def test_ev_builder_returns_none_when_no_batches() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)

    y_true, y_pred, y_true_raw, weights = algo._build_explained_variance_tensors(
        [], [], [], [], [], [], [], []
    )

    assert y_true is None
    assert y_pred is None
    assert y_true_raw is None
    assert weights is None


def test_ev_builder_uses_reserve_pairs_without_length_mismatch() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)

    reserve_true = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32)
    reserve_pred = torch.tensor([[0.0], [0.15], [0.25]], dtype=torch.float32)
    reserve_raw = torch.tensor([[1.0], [1.1], [1.2]], dtype=torch.float32)
    reserve_weights = torch.tensor([[1.0], [0.5], [1.0]], dtype=torch.float32)

    y_true, y_pred, y_true_raw, weights = algo._build_explained_variance_tensors(
        [],
        [],
        [],
        [],
        [reserve_true],
        [reserve_pred],
        [reserve_raw],
        [reserve_weights],
    )

    assert y_true is not None and y_pred is not None
    assert y_true.shape == reserve_true.shape == y_pred.shape
    assert torch.equal(y_true, reserve_true)
    assert torch.equal(y_pred, reserve_pred)

    assert y_true_raw is not None
    assert torch.equal(y_true_raw, reserve_raw)

    assert weights is not None
    assert torch.equal(weights, reserve_weights)

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

    ev_value, y_true_eval, y_pred_eval = algo._compute_explained_variance_metric(
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

    ev_value, _, _ = algo._compute_explained_variance_metric(
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

    ev_value, _, _ = algo._compute_explained_variance_metric(
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


def test_explained_variance_metric_returns_none_with_empty_mask() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)

    y_true_norm = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    y_pred_norm = torch.tensor([[0.5], [1.5]], dtype=torch.float32)
    mask = torch.zeros_like(y_true_norm)

    ev_value, y_true_eval, y_pred_eval = algo._compute_explained_variance_metric(
        y_true_norm,
        y_pred_norm,
        mask_tensor=mask,
    )

    assert ev_value is None
    assert y_true_eval.numel() == 0
    assert y_pred_eval.numel() == 0


def test_explained_variance_metric_returns_none_with_empty_inputs() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)

    y_true_norm = torch.tensor([], dtype=torch.float32)
    y_pred_norm = torch.tensor([], dtype=torch.float32)

    ev_value, y_true_eval, y_pred_eval = algo._compute_explained_variance_metric(
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

    ev_value, y_true_eval, y_pred_eval = algo._compute_explained_variance_metric(
        y_true_norm,
        y_pred_norm,
        mask_tensor=mask,
    )

    assert ev_value is None
    assert y_true_eval.numel() == 4
    assert y_pred_eval.numel() == 4


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

    class _DummyModel:
        def __init__(self, quantiles_tensor: torch.Tensor) -> None:
            self.policy = SimpleNamespace(device=torch.device("cpu"), recurrent_initial_state=None)
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
