import math
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="distributional_ppo tests require torch")


pytest.importorskip("sb3_contrib", reason="distributional_ppo depends on sb3_contrib")

from collections import deque

from distributional_ppo import DistributionalPPO, calculate_cvar
from utils.model_io import upgrade_quantile_value_state_dict
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


def test_upgrade_quantile_value_state_dict_fallback_duplication() -> None:
    weight = torch.linspace(-1.0, 1.0, steps=4, dtype=torch.float32).view(1, -1)
    bias = torch.tensor([0.25], dtype=torch.float32)
    original = {
        "value_net.weight": weight.clone(),
        "value_net.bias": bias.clone(),
    }

    upgraded = upgrade_quantile_value_state_dict(
        original,
        target_prefix="quantile_head.linear",
        num_quantiles=4,
        fallback_prefixes=("value_net",),
    )

    assert upgraded is not original
    assert upgraded["quantile_head.linear.weight"].shape == (4, weight.shape[1])
    assert upgraded["quantile_head.linear.bias"].shape == (4,)
    repeated_weight = weight.expand(4, -1)
    repeated_bias = bias.expand(4)
    assert torch.allclose(upgraded["quantile_head.linear.weight"], repeated_weight)
    assert torch.allclose(upgraded["quantile_head.linear.bias"], repeated_bias)


def test_upgrade_quantile_value_state_dict_noop_when_already_quantile() -> None:
    weight = torch.randn(8, 3, dtype=torch.float32)
    bias = torch.randn(8, dtype=torch.float32)
    state = {
        "quantile_head.linear.weight": weight,
        "quantile_head.linear.bias": bias,
    }

    upgraded = upgrade_quantile_value_state_dict(
        state,
        target_prefix="quantile_head.linear",
        num_quantiles=8,
        fallback_prefixes=("value_net",),
    )

    assert upgraded is state


def _make_cvar_model() -> DistributionalPPO:
    model = DistributionalPPO.__new__(DistributionalPPO)
    model.cvar_alpha = 0.05
    model.cvar_winsor_pct = 0.0
    model.cvar_ema_beta = 0.9
    return model


def test_cvar_winsor_pct_percent_conversion() -> None:
    model = _make_cvar_model()
    model.cvar_winsor_pct = 0.1
    assert model.cvar_winsor_pct == pytest.approx(0.1)
    assert getattr(model, "_cvar_winsor_fraction", None) == pytest.approx(0.001)

    model.cvar_winsor_pct = 0.001  # legacy fraction -> 0.1%
    assert model.cvar_winsor_pct == pytest.approx(0.1)
    assert getattr(model, "_cvar_winsor_fraction", None) == pytest.approx(0.001)


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_compute_empirical_cvar_scales_linearly(scale: float) -> None:
    model = _make_cvar_model()
    base_rewards = torch.tensor([-1.25, -0.8, 0.4, 0.9, -0.2], dtype=torch.float32)
    _, cvar_base = model._compute_empirical_cvar(base_rewards)
    _, cvar_scaled = model._compute_empirical_cvar(base_rewards * scale)

    assert cvar_scaled.item() == pytest.approx(cvar_base.item() * scale, rel=1e-6)

    _, _, _, _, abs_base = model._compute_cvar_statistics(base_rewards)
    _, _, _, _, abs_scaled = model._compute_cvar_statistics(base_rewards * scale)
    assert abs_scaled.item() == pytest.approx(abs_base.item() * abs(scale), rel=1e-6)


def test_cvar_limit_units_consistent() -> None:
    model = _make_cvar_model()
    model.cvar_alpha = 0.5
    rewards = torch.tensor([-0.02, -0.02, 0.01, 0.01], dtype=torch.float32)
    _, cvar_empirical, _, _, _ = model._compute_cvar_statistics(rewards)
    limit_fraction = -0.02
    assert cvar_empirical.item() == pytest.approx(limit_fraction, abs=1e-6)
    violation = max(0.0, limit_fraction - cvar_empirical.item())
    assert violation == pytest.approx(0.0, abs=1e-6)


def test_cvar_violation_uses_fraction_units() -> None:
    model = _make_cvar_model()
    model.cvar_limit = -0.01

    violation = model._compute_cvar_violation(-0.02)
    assert violation == pytest.approx(0.01)

    no_violation = model._compute_cvar_violation(-0.005)
    assert no_violation == pytest.approx(0.0)


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
