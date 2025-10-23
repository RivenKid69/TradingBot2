import math
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="distributional_ppo tests require torch")


pytest.importorskip("sb3_contrib", reason="distributional_ppo depends on sb3_contrib")

from collections import deque

from distributional_ppo import DistributionalPPO, calculate_cvar, create_sequencers
from utils.model_io import upgrade_quantile_value_state_dict
from stable_baselines3.common.running_mean_std import RunningMeanStd


class _CaptureLogger:
    def __init__(self) -> None:
        self.records: dict[str, float] = {}

    def record(self, key: str, value: float, **_: object) -> None:
        self.records[key] = float(value)


def test_create_sequencers_groups_sequences_and_pads() -> None:
    episode_starts = np.array([0, 0, 1, 0, 0, 0, 1], dtype=bool)
    env_change = np.array([1, 0, 0, 0, 1, 0, 0], dtype=bool)

    seq_start_indices, pad, pad_and_flatten = create_sequencers(
        episode_starts, env_change, device="cpu"
    )

    assert seq_start_indices.tolist() == [0, 2, 4, 6]

    values = np.arange(1, episode_starts.size + 1, dtype=np.int64)
    padded = pad(values)
    expected = np.array([[1, 2], [3, 4], [5, 6], [7, 0]], dtype=np.int64)
    assert np.array_equal(padded, expected)

    mask = pad_and_flatten(np.ones_like(values, dtype=np.float32))
    assert mask.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]


def test_create_sequencers_forces_initial_boundary() -> None:
    episode_starts = np.zeros(4, dtype=bool)
    env_change = np.zeros(4, dtype=bool)

    seq_start_indices, pad, _ = create_sequencers(episode_starts, env_change, device="cpu")

    assert seq_start_indices.tolist() == [0]

    padded = pad(np.array([10, 11, 12, 13], dtype=np.int64))
    assert padded.shape == (1, 4)
    assert padded[0].tolist() == [10, 11, 12, 13]


def test_create_sequencers_squeezes_unit_dimensions() -> None:
    episode_starts = np.zeros((4, 1), dtype=bool)
    env_change = np.zeros((4, 1), dtype=bool)

    seq_start_indices, pad, _ = create_sequencers(episode_starts, env_change, device="cpu")

    assert seq_start_indices.tolist() == [0]

    padded = pad(np.array([[10], [11], [12], [13]], dtype=np.int64))
    assert padded.shape == (1, 4)
    assert padded[0].tolist() == [10, 11, 12, 13]


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


def test_build_support_distribution_saturates_upper_bound() -> None:
    model = DistributionalPPO.__new__(DistributionalPPO)
    num_atoms = 5
    v_min = -1.0
    v_max = 2.0
    delta_z = (v_max - v_min) / float(num_atoms - 1)
    policy = types.SimpleNamespace(v_min=v_min, v_max=v_max, delta_z=delta_z)
    model.policy = policy  # type: ignore[assignment]

    template = torch.zeros((2, num_atoms), dtype=torch.float32)
    returns = torch.tensor([v_max, v_max + 5.0], dtype=torch.float32)

    distribution = model._build_support_distribution(returns, template)

    expected = torch.zeros_like(template)
    expected[:, -1] = 1.0

    assert torch.allclose(distribution, expected)


def _make_cvar_model() -> DistributionalPPO:
    model = DistributionalPPO.__new__(DistributionalPPO)
    model.cvar_alpha = 0.05
    model.cvar_winsor_pct = 0.0
    model.cvar_ema_beta = 0.9
    model._value_scale_std_floor = 1e-6
    model._cvar_weight_target = 0.5
    model._cvar_ramp_updates = 1
    model.cvar_penalty_cap = 0.7
    model.value_target_scale = 1.0
    model._value_target_scale_robust = 1.0
    model.normalize_returns = False
    model._ret_mean_snapshot = 0.0
    model._ret_std_snapshot = 1.0
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


def test_cvar_lambda_stays_bounded() -> None:
    assert DistributionalPPO._bounded_dual_update(0.9, 0.5, 10.0) == pytest.approx(1.0)
    assert DistributionalPPO._bounded_dual_update(0.1, 0.5, -10.0) == pytest.approx(0.0)
    mid = DistributionalPPO._bounded_dual_update(0.4, 0.1, 0.2)
    assert 0.0 <= mid <= 1.0


def test_cvar_penalty_turns_off_when_no_violation() -> None:
    model = _make_cvar_model()
    nominal, raw, active = model._resolve_cvar_penalty_state(0.2, 0.2, 0.0)
    assert not active
    assert nominal == pytest.approx(0.0)
    assert raw == pytest.approx(0.0)

    fallback_nominal, fallback_raw, fallback_active = model._resolve_cvar_penalty_state(0.0, 0.0, 0.25)
    assert fallback_active
    assert fallback_nominal > 0.0
    assert fallback_raw == pytest.approx(fallback_nominal)


def test_cvar_normalised_matches_zscore_reference() -> None:
    model = _make_cvar_model()
    model.normalize_returns = True
    model._ret_mean_snapshot = 0.01
    model._ret_std_snapshot = 0.02
    rewards = torch.tensor([-0.03, -0.015, 0.01, 0.02, -0.005], dtype=torch.float32)

    _, cvar_empirical_tensor, *_ = model._compute_cvar_statistics(rewards)
    cvar_empirical = float(cvar_empirical_tensor.item())
    offset, scale = model._get_cvar_normalization_params()
    cvar_unit = (cvar_empirical - offset) / scale

    zscores = (rewards.numpy() - offset) / scale
    tail_count = max(int(math.ceil(model.cvar_alpha * rewards.numel())), 1)
    expected_unit = float(np.mean(np.sort(zscores)[:tail_count]))

    assert cvar_unit == pytest.approx(expected_unit, rel=1e-6, abs=1e-6)


def test_cvar_scale_logging_and_freeze() -> None:
    model = _make_cvar_model()
    model.normalize_returns = False
    model._value_target_scale_robust = 0.12
    model.value_target_scale = 0.03
    _, scale_initial = model._get_cvar_normalization_params()
    model.value_target_scale = 0.9
    _, scale_after_drift = model._get_cvar_normalization_params()
    assert scale_initial == pytest.approx(0.12)
    assert scale_after_drift == pytest.approx(0.12)

    logger = _CaptureLogger()
    model.logger = logger
    model._record_cvar_logs(
        cvar_raw_value=-0.01,
        cvar_unit_value=-0.01 / scale_after_drift,
        cvar_loss_raw_value=0.01,
        cvar_loss_unit_value=0.01 / scale_after_drift,
        cvar_term_raw_value=0.0,
        cvar_term_unit_value=0.0,
        cvar_empirical_value=-0.01,
        cvar_empirical_unit_value=-0.01 / scale_after_drift,
        cvar_empirical_ema_value=-0.01,
        cvar_violation_raw_value=0.02,
        cvar_violation_raw_unclipped_value=0.02,
        cvar_violation_unit_value=0.02 / scale_after_drift,
        cvar_violation_ema_value=0.02,
        cvar_gap_raw_value=0.02,
        cvar_gap_unit_value=0.02 / scale_after_drift,
        cvar_penalty_active_value=1.0,
        cvar_lambda_value=0.5,
        cvar_scale_value=scale_after_drift,
        cvar_limit_raw_value=-0.02,
        cvar_limit_unit_value=-0.02 / scale_after_drift,
        current_cvar_weight_scaled=0.1,
        current_cvar_weight_nominal=0.1,
        current_cvar_weight_raw=0.1,
        cvar_penalty_cap_value=0.7,
    )

    assert logger.records["train/cvar_scale"] == pytest.approx(scale_after_drift)
    assert logger.records["train/cvar_unit"] == pytest.approx(-0.01 / scale_after_drift)


@pytest.mark.parametrize("normalize_returns", [False, True])
def test_cvar_penalty_active_unit_consistency(normalize_returns: bool) -> None:
    model = _make_cvar_model()
    model.normalize_returns = normalize_returns
    model.cvar_use_penalty = True
    model.cvar_limit = -0.02
    model.cvar_lambda_lr = 0.1

    if normalize_returns:
        model._ret_mean_snapshot = 0.012
        model._ret_std_snapshot = 0.05
        model._value_scale_std_floor = 0.003
    else:
        model._value_target_scale_robust = 0.25
        model.value_target_scale = 0.05

    rewards = torch.tensor([-0.03, -0.015, 0.01, 0.02, -0.005], dtype=torch.float32)
    _, cvar_empirical_tensor, *_ = model._compute_cvar_statistics(rewards)
    cvar_empirical = float(cvar_empirical_tensor.item())

    offset, scale = model._get_cvar_normalization_params()
    assert scale > 0.0
    limit_raw = float(model._get_cvar_limit_raw())
    limit_unit = (limit_raw - offset) / scale

    cvar_unit = (cvar_empirical - offset) / scale
    cvar_loss_unit = -cvar_unit
    cvar_loss_raw = cvar_loss_unit * scale

    cvar_gap_raw = limit_raw - cvar_empirical
    cvar_gap_unit = limit_unit - cvar_unit
    violation_raw = max(cvar_gap_raw, 0.0)
    violation_unit = max(cvar_gap_unit, 0.0)

    penalty_nominal, penalty_raw, penalty_active = model._resolve_cvar_penalty_state(
        0.0, 0.0, violation_unit
    )
    assert penalty_active
    assert 0.0 < penalty_nominal <= model.cvar_penalty_cap
    assert penalty_raw == pytest.approx(penalty_nominal)

    lambda_values = []
    lambda_state = 0.0
    for _ in range(3):
        lambda_state = DistributionalPPO._bounded_dual_update(
            lambda_state, model.cvar_lambda_lr, violation_unit
        )
        lambda_values.append(lambda_state)

    assert lambda_values == sorted(lambda_values)
    assert all(0.0 <= value <= 1.0 for value in lambda_values)

    cvar_term_unit = penalty_raw * cvar_loss_unit
    cvar_term_raw = penalty_raw * cvar_loss_raw

    model.logger = _CaptureLogger()
    model._record_cvar_logs(
        cvar_raw_value=cvar_empirical,
        cvar_unit_value=cvar_unit,
        cvar_loss_raw_value=cvar_loss_raw,
        cvar_loss_unit_value=cvar_loss_unit,
        cvar_term_raw_value=cvar_term_raw,
        cvar_term_unit_value=cvar_term_unit,
        cvar_empirical_value=cvar_empirical,
        cvar_empirical_unit_value=cvar_unit,
        cvar_empirical_ema_value=cvar_empirical,
        cvar_violation_raw_value=violation_raw,
        cvar_violation_raw_unclipped_value=cvar_gap_raw,
        cvar_violation_unit_value=violation_unit,
        cvar_violation_ema_value=violation_unit,
        cvar_gap_raw_value=cvar_gap_raw,
        cvar_gap_unit_value=cvar_gap_unit,
        cvar_penalty_active_value=1.0,
        cvar_lambda_value=lambda_values[-1],
        cvar_scale_value=scale,
        cvar_limit_raw_value=limit_raw,
        cvar_limit_unit_value=limit_unit,
        current_cvar_weight_scaled=penalty_raw,
        current_cvar_weight_nominal=penalty_nominal,
        current_cvar_weight_raw=penalty_raw,
        cvar_penalty_cap_value=model.cvar_penalty_cap,
    )

    records = model.logger.records
    assert records["train/cvar_penalty_active"] == pytest.approx(1.0)
    assert records["train/cvar_loss"] == pytest.approx(cvar_loss_raw)
    assert records["train/cvar_loss_unit"] == pytest.approx(cvar_loss_unit)
    assert records["train/cvar_term_in_fraction"] == pytest.approx(cvar_term_raw)
    assert records["train/cvar_term"] == pytest.approx(cvar_term_unit)
    assert records["train/cvar_scale"] == pytest.approx(scale)
    assert records["train/cvar_loss"] == pytest.approx(records["train/cvar_loss_in_fraction"])
    assert records["train/cvar_loss"] == pytest.approx(records["train/cvar_loss_unit"] * scale)
    assert records["train/cvar_term_in_fraction"] == pytest.approx(
        records["train/cvar_term"] * scale
    )
    assert records["train/cvar_unit"] == pytest.approx(cvar_unit)
    assert records["train/cvar_limit_unit"] == pytest.approx(limit_unit)

    offset_again, scale_again = model._get_cvar_normalization_params()
    assert offset_again == pytest.approx(offset)
    assert scale_again == pytest.approx(scale)

    model.value_target_scale *= 10.0
    offset_drift, scale_drift = model._get_cvar_normalization_params()
    assert offset_drift == pytest.approx(offset)
    assert scale_drift == pytest.approx(scale)

    cvar_unit_rescaled = (cvar_empirical - offset_drift) / scale_drift
    assert cvar_unit_rescaled == pytest.approx(cvar_unit)


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
    model._value_scale_std_floor = 3e-3
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
