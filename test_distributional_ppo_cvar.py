import math

import numpy as np
import pytest
import torch


pytest.importorskip("sb3_contrib", reason="distributional_ppo depends on sb3_contrib")

from distributional_ppo import calculate_cvar


def _discrete_cvar_reference(probs: np.ndarray, atoms: np.ndarray, alpha: float) -> float:
    order = np.argsort(atoms)
    atoms_sorted = atoms[order]
    probs_sorted = probs[order]

    cumulative = np.cumsum(probs_sorted)
    idx = np.searchsorted(cumulative, alpha, side="left")

    expectation = 0.0
    for pos in range(idx):
        expectation += probs_sorted[pos] * atoms_sorted[pos]

    prev_cum = cumulative[idx - 1] if idx > 0 else 0.0
    weight = alpha - prev_cum
    if idx < atoms_sorted.size and weight > 0.0:
        expectation += weight * atoms_sorted[idx]

    return expectation / alpha


def test_calculate_cvar_matches_reference() -> None:
    probs = torch.tensor(
        [[0.2, 0.3, 0.5], [0.05, 0.05, 0.9]], dtype=torch.float32
    )
    atoms = torch.tensor([-2.0, -1.0, 0.5], dtype=torch.float32)
    alpha = 0.1

    result = calculate_cvar(probs, atoms, alpha).cpu().numpy()

    expected = np.array(
        [_discrete_cvar_reference(p, atoms.numpy(), alpha) for p in probs.numpy()]
    )

    assert np.allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5, math.inf, math.nan])
def test_calculate_cvar_invalid_alpha(alpha: float) -> None:
    probs = torch.tensor([[1.0]], dtype=torch.float32)
    atoms = torch.tensor([0.0], dtype=torch.float32)

    with pytest.raises(ValueError):
        calculate_cvar(probs, atoms, alpha)
