import math
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="distributional_ppo tests require torch")


pytest.importorskip("sb3_contrib", reason="distributional_ppo depends on sb3_contrib")

from collections import deque

from distributional_ppo import DistributionalPPO, calculate_cvar
from stable_baselines3.common.running_mean_std import RunningMeanStd


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


def test_value_scale_snapshot_prevents_mismatch() -> None:
    returns_raw = np.array([100.0, 110.0, 90.0, 105.0], dtype=np.float32)
    snapshot_mean = 0.0
    snapshot_std = 1.0
    returns_tensor = torch.tensor(returns_raw, dtype=torch.float32)

    value_pred_norm = (returns_tensor - snapshot_mean) / snapshot_std
    y_true_tensor = returns_tensor
    mse_snapshot = torch.mean((value_pred_norm * snapshot_std + snapshot_mean - y_true_tensor) ** 2)

    fresh_mean = float(np.mean(returns_raw))
    fresh_std = float(np.std(returns_raw)) + 1e-8
    mse_fresh = torch.mean((value_pred_norm * fresh_std + fresh_mean - y_true_tensor) ** 2)

    assert mse_snapshot.item() == pytest.approx(0.0, abs=1e-6)
    assert mse_fresh.item() > 1e3

    model = DistributionalPPO.__new__(DistributionalPPO)
    model.normalize_returns = True
    model.ret_clip = 5.0
    model.ret_rms = RunningMeanStd(shape=())
    model.ret_rms.mean[...] = snapshot_mean
    model.ret_rms.var[...] = snapshot_std**2
    model.ret_rms.count = 1.0
    model._ret_mean_value = snapshot_mean
    model._ret_std_value = snapshot_std
    model._ret_mean_snapshot = snapshot_mean
    model._ret_std_snapshot = snapshot_std
    model._value_scale_ema_beta = 0.5
    model._value_scale_max_rel_step = 0.25
    model._value_scale_std_floor = 1e-2
    model._value_scale_window_updates = 0
    model._value_scale_recent_stats = deque()
    model._value_scale_stats_initialized = True
    model._value_scale_stats_mean = snapshot_mean
    model._value_scale_stats_second = snapshot_std**2 + snapshot_mean**2
    model._pending_rms = RunningMeanStd(shape=())
    model._pending_rms.update(returns_raw)
    model._pending_ret_mean = snapshot_mean
    model._pending_ret_std = snapshot_std
    model._value_target_scale_effective = float(1.0 / (model.ret_clip * snapshot_std))
    model._value_target_scale_robust = 1.0
    model.running_v_min = -model.ret_clip
    model.running_v_max = model.ret_clip
    model.v_range_initialized = True
    model.policy = types.SimpleNamespace(
        atoms=torch.linspace(-model.ret_clip, model.ret_clip, steps=3),
        update_atoms=lambda *_args, **_kwargs: None,
    )

    class _Recorder:
        def __init__(self) -> None:
            self.records: dict[str, float] = {}

        def record(self, key: str, value: float) -> None:
            self.records[key] = float(value)

    model.logger = _Recorder()

    model._finalize_return_stats()

    assert model._ret_mean_value == pytest.approx(model._ret_mean_snapshot)
    assert model._ret_std_value == pytest.approx(model._ret_std_snapshot)
    assert model._ret_std_value <= snapshot_std * (1.0 + model._value_scale_max_rel_step) + 1e-9
    assert model._ret_std_value >= snapshot_std / (1.0 + model._value_scale_max_rel_step) - 1e-9

    max_mean_delta = max(abs(snapshot_mean), model._ret_std_value, fresh_std) * model._value_scale_max_rel_step
    assert abs(model._ret_mean_value - snapshot_mean) <= max_mean_delta + 1e-8

    records = model.logger.records
    assert records["train/value_scale_mean_next"] == pytest.approx(model._ret_mean_value)
    assert records["train/value_scale_std_next"] == pytest.approx(model._ret_std_value)
