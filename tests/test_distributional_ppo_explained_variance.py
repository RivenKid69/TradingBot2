import math
import pytest
import torch

import test_distributional_ppo_raw_outliers  # noqa: F401  # ensure RL stubs are installed

from distributional_ppo import DistributionalPPO, safe_explained_variance


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
    algo._ret_mean_snapshot = 0.0
    algo._ret_std_snapshot = 1.0
    algo.value_target_scale = 1.0
    algo._value_target_scale_effective = 1.0
    algo.logger = _DummyLogger()

    y_true_norm = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32)
    y_pred_norm = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32)
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

    # Raw fallback should mirror variance of unclipped targets.
    expected_ev = safe_explained_variance(
        y_true_raw.numpy(),
        y_pred_norm.numpy(),
        mask.numpy(),
    )
    assert ev_value == pytest.approx(expected_ev)


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
