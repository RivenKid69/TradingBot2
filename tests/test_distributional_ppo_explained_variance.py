import math

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
