import math

import pytest

import test_distributional_ppo_raw_outliers  # noqa: F401  # ensure RL stubs are installed


@pytest.mark.parametrize("target_shape", [(2,), (2, 1), (2, 1, 1)])
def test_quantile_huber_loss_preserves_batch_dimension(target_shape) -> None:
    torch = pytest.importorskip("torch")

    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class _PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self) -> torch.Tensor:  # pragma: no cover - simple deterministic tensor
            return torch.tensor([0.25, 0.75], dtype=torch.float32)

    algo.policy = _PolicyStub()

    predicted = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32, requires_grad=True
    )
    targets = torch.tensor([0.0, 1.0], dtype=torch.float32).reshape(target_shape)

    loss = DistributionalPPO._quantile_huber_loss(algo, predicted, targets)
    loss.backward()

    # With a correctly shaped loss the first sample already matches its
    # target, so its gradient must stay zero regardless of how the targets
    # are reshaped.  If the implementation accidentally broadcasts across
    # the batch dimension we would observe identical non-zero gradients
    # for both rows instead.
    grad_first = predicted.grad[0].abs().max().item()
    grad_second = predicted.grad[1].abs().max().item()

    assert math.isclose(grad_first, 0.0, abs_tol=1e-6)
    assert grad_second > 0.0


def test_quantile_huber_loss_unsqueeze_path_produces_distinct_gradients() -> None:
    torch = pytest.importorskip("torch")

    from distributional_ppo import DistributionalPPO

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._quantile_huber_kappa = 1.0

    class _PolicyStub:
        device = torch.device("cpu")

        @property
        def quantile_levels(self) -> torch.Tensor:  # pragma: no cover - deterministic tensor
            return torch.tensor([0.25, 0.75], dtype=torch.float32)

    algo.policy = _PolicyStub()

    predicted = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32, requires_grad=True
    )
    target_returns_norm_selected = torch.tensor([0.0, 1.0], dtype=torch.float32)
    if target_returns_norm_selected.dim() == 1:  # mirror _train_step behaviour
        target_returns_norm_selected = target_returns_norm_selected.unsqueeze(1)

    loss = DistributionalPPO._quantile_huber_loss(
        algo, predicted, target_returns_norm_selected
    )
    loss.backward()

    grad_first = predicted.grad[0].abs().max().item()
    grad_second = predicted.grad[1].abs().max().item()

    assert not math.isclose(grad_first, grad_second, abs_tol=1e-6)
    assert grad_second > grad_first

