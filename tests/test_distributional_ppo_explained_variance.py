import math
import math
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
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


def test_resolve_ev_reserve_mask_respects_flag() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)

    valid_indices = torch.tensor([0, 2], dtype=torch.long)
    mask_values = torch.tensor([0.5, 1.0], dtype=torch.float32)

    algo._ev_reserve_apply_mask = True
    resolved_indices, resolved_mask = algo._resolve_ev_reserve_mask(
        valid_indices,
        mask_values,
    )

    assert resolved_indices is valid_indices
    assert resolved_mask is mask_values

    algo._ev_reserve_apply_mask = False
    resolved_indices, resolved_mask = algo._resolve_ev_reserve_mask(
        valid_indices,
        mask_values,
    )

    assert resolved_indices is valid_indices
    assert resolved_mask is None


def test_resolve_ev_reserve_mask_drops_empty_tensors() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._ev_reserve_apply_mask = True

    empty_indices = torch.zeros(0, dtype=torch.long)
    empty_mask = torch.zeros(0, dtype=torch.float32)

    resolved_indices, resolved_mask = algo._resolve_ev_reserve_mask(
        empty_indices,
        empty_mask,
    )

    assert resolved_indices is None
    assert resolved_mask is None






def test_ev_group_key_from_info_prefers_symbol_and_env_mapping() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)

    info_primary = {"env_id": " shard-0 ", "symbol": "btcusdt"}
    info_fallback = {"environment": "prod", "instrument": "ethusd"}

    assert algo._ev_group_key_from_info(7, info_primary) == "shard-0::BTCUSDT"
    assert algo._ev_group_key_from_info(1, info_fallback) == "prod::ETHUSD"


def test_ev_group_key_from_info_falls_back_to_env_index() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)

    assert algo._ev_group_key_from_info(3, object()) == "env3"
    assert algo._ev_group_key_from_info(0, {"symbol": ""}) == "env0"


def test_resolve_ev_group_keys_from_flat_uses_cached_keys_and_env_defaults() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.rollout_buffer = SimpleNamespace(buffer_size=2, n_envs=2)
    algo._last_rollout_ev_keys = np.array(
        [["vec0::BTC", None], ["", "vec1::SOL"]], dtype=object
    )

    indices = np.array([0, 1, 2, 3], dtype=np.int64)
    result = algo._resolve_ev_group_keys_from_flat(indices)

    assert result == ["vec0::BTC", "env0", "env1", "vec1::SOL"]


def test_resolve_ev_group_keys_from_flat_handles_missing_cache() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._last_rollout_ev_keys = None

    indices = np.array([0, -1, 3], dtype=np.int64)
    result = algo._resolve_ev_group_keys_from_flat(indices)

    assert result == ["env0", "env3"]


def test_resolve_ev_group_keys_from_flat_rejects_out_of_bounds() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.rollout_buffer = SimpleNamespace(buffer_size=1, n_envs=1)
    algo._last_rollout_ev_keys = np.array([["only"]], dtype=object)

    assert algo._resolve_ev_group_keys_from_flat(np.array([2], dtype=np.int64)) == []


def test_extract_group_keys_for_indices_filters_invalid_rows() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.rollout_buffer = SimpleNamespace(buffer_size=4, n_envs=1)
    algo._last_rollout_ev_keys = np.array(["A", "B", "C", "D"], dtype=object)

    rollout_data = SimpleNamespace(
        sample_indices=torch.tensor([0, 3, -1, 1], dtype=torch.long)
    )
    subset = torch.tensor([0, 3], dtype=torch.long)

    result = algo._extract_group_keys_for_indices(rollout_data, subset)

    assert result == ["A", "B"]


def test_extract_group_keys_for_indices_returns_empty_on_bad_indices() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.rollout_buffer = SimpleNamespace(buffer_size=2, n_envs=1)
    algo._last_rollout_ev_keys = np.array(["X", "Y"], dtype=object)

    rollout_data = SimpleNamespace(
        sample_indices=torch.tensor([0, 1], dtype=torch.long)
    )
    subset = torch.tensor([0, 5], dtype=torch.long)

    assert algo._extract_group_keys_for_indices(rollout_data, subset) == []


class _CaptureLogger:
    def __init__(self) -> None:
        self.records: dict[str, list[float]] = {}

    def record(self, key: str, value: float, **_: object) -> None:
        self.records.setdefault(key, []).append(value)


class _PolicyMinimal:
    def __init__(self) -> None:
        self.optimizer = object()

    def set_training_mode(self, _: bool) -> None:
        pass


def _make_algo_for_ev_gate(
    threshold: float,
    latest_ev: float | None,
) -> tuple[DistributionalPPO, _CaptureLogger]:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    logger = _CaptureLogger()
    algo.logger = logger
    algo._logger = logger
    algo.policy = _PolicyMinimal()
    algo.device = torch.device("cpu")
    algo.lr_schedule = lambda _: 0.001
    algo.clip_range = 0.2
    algo.clip_range_vf = 0.5
    algo.n_epochs = 0
    algo.batch_size = 1
    algo._grad_accumulation_steps = 1
    algo._microbatch_size = 1
    algo._critic_grad_block_logged_state = False
    algo._critic_grad_block_scale = 1.0
    algo._critic_grad_blocked = False
    algo._global_update_step = 5
    algo._vf_clip_warmup_updates = 2
    algo._vf_clip_threshold_ev = threshold
    algo._vf_clip_latest_ev = latest_ev
    algo._vf_clip_warmup_logged_complete = False
    algo._compute_clip_range_value = MethodType(lambda self, _: 0.15, algo)
    algo._rebuild_scheduler_if_needed = MethodType(lambda self: None, algo)
    algo._update_learning_rate = MethodType(lambda self, _optimizer: None, algo)
    algo._refresh_kl_base_lrs = MethodType(lambda self: None, algo)
    algo._ensure_score_action_space = MethodType(lambda self: None, algo)

    def _stop(*_: Any, **__: Any) -> None:
        raise RuntimeError("stop after vf_clip logging")

    algo._update_ent_coef = MethodType(_stop, algo)
    return algo, logger


@pytest.mark.parametrize(
    "latest_ev, expected_gate",
    [(None, 1.0), (float("nan"), 1.0), (0.1, 1.0), (0.6, 0.0)],
)
def test_train_logs_ev_gate_based_on_latest_explained_variance(
    latest_ev: float | None, expected_gate: float
) -> None:
    algo, logger = _make_algo_for_ev_gate(threshold=0.25, latest_ev=latest_ev)

    with pytest.raises(RuntimeError, match="stop after vf_clip logging"):
        algo.train()

    records = logger.records
    assert records["train/vf_clip_ev_gate_active"][0] == pytest.approx(expected_gate)
    assert records["train/vf_clip_threshold_ev"][0] == pytest.approx(0.25)
    if latest_ev is not None and math.isfinite(latest_ev):
        assert records["train/vf_clip_last_ev"][0] == pytest.approx(latest_ev)
    else:
        assert "train/vf_clip_last_ev" not in records


def test_explained_variance_reserve_path_applies_mask() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)

    base_true = torch.tensor([[0.0], [1.0], [3.0]], dtype=torch.float32)
    base_pred = torch.tensor([[0.5], [0.75], [2.5]], dtype=torch.float32)
    base_raw = base_true.clone()
    valid_indices = torch.tensor([1, 2], dtype=torch.long)
    mask_values = torch.tensor([0.25, 0.75], dtype=torch.float32)

    algo._ev_reserve_apply_mask = True
    masked_indices, masked_weights = algo._resolve_ev_reserve_mask(
        valid_indices,
        mask_values,
    )
    assert masked_indices is not None
    assert masked_weights is not None

    reserve_true_masked = [base_true[masked_indices]]
    reserve_pred_masked = [base_pred[masked_indices]]
    reserve_raw_masked = [base_raw[masked_indices]]
    reserve_weight_masked = [masked_weights.reshape(-1, 1)]
    reserve_group_keys_masked = [["idx=1", "idx=2"]]

    (
        y_true_masked,
        y_pred_masked,
        y_raw_masked,
        weights_masked,
        _,
    ) = algo._build_explained_variance_tensors(
        [],
        [],
        [],
        [],
        [],
        reserve_true_masked,
        reserve_pred_masked,
        reserve_raw_masked,
        reserve_weight_masked,
        reserve_group_keys_masked,
    )

    masked_ev, _, _, masked_metrics = algo._compute_explained_variance_metric(
        y_true_masked,
        y_pred_masked,
        mask_tensor=weights_masked,
        y_true_tensor_raw=y_raw_masked,
    )

    expected_masked_ev = safe_explained_variance(
        base_true[masked_indices].numpy().reshape(-1),
        base_pred[masked_indices].numpy().reshape(-1),
        mask_values.numpy(),
    )
    assert masked_ev == pytest.approx(expected_masked_ev)
    assert masked_metrics["n_samples"] == pytest.approx(float(mask_values.numel()))

    algo._ev_reserve_apply_mask = False
    unmasked_indices, unmasked_weights = algo._resolve_ev_reserve_mask(
        valid_indices,
        mask_values,
    )
    assert unmasked_indices is None
    assert unmasked_weights is None

    reserve_true_unmasked = [base_true]
    reserve_pred_unmasked = [base_pred]
    reserve_raw_unmasked = [base_raw]
    reserve_weight_unmasked: list[torch.Tensor] = []
    reserve_group_keys_unmasked = [["idx=0", "idx=1", "idx=2"]]

    (
        y_true_unmasked,
        y_pred_unmasked,
        y_raw_unmasked,
        weights_unmasked,
        _,
    ) = algo._build_explained_variance_tensors(
        [],
        [],
        [],
        [],
        [],
        reserve_true_unmasked,
        reserve_pred_unmasked,
        reserve_raw_unmasked,
        reserve_weight_unmasked,
        reserve_group_keys_unmasked,
    )

    unmasked_ev, _, _, unmasked_metrics = algo._compute_explained_variance_metric(
        y_true_unmasked,
        y_pred_unmasked,
        mask_tensor=weights_unmasked,
        y_true_tensor_raw=y_raw_unmasked,
    )

    expected_unmasked_ev = safe_explained_variance(
        base_true.numpy().reshape(-1),
        base_pred.numpy().reshape(-1),
        None,
    )
    assert unmasked_ev == pytest.approx(expected_unmasked_ev)
    assert unmasked_metrics["n_samples"] == pytest.approx(float(base_true.numel()))
    assert masked_ev != pytest.approx(unmasked_ev)


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


def test_ev_builder_with_mixed_masks_falls_back_to_unweighted() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)

    primary_true_batches = [
        torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        torch.tensor([[2.0]], dtype=torch.float32),
    ]
    primary_pred_batches = [
        torch.tensor([[0.5], [0.75]], dtype=torch.float32),
        torch.tensor([[1.5]], dtype=torch.float32),
    ]
    primary_raw_batches = [batch.clone() for batch in primary_true_batches]
    primary_weight_batches = [
        torch.tensor([[0.25], [0.75]], dtype=torch.float32),
        torch.zeros((0, 1), dtype=torch.float32),
    ]
    primary_group_keys = [["g0", "g1"], ["g2"]]

    (
        y_true,
        y_pred,
        y_raw,
        weights,
        _,
    ) = algo._build_explained_variance_tensors(
        primary_true_batches,
        primary_pred_batches,
        primary_raw_batches,
        primary_weight_batches,
        primary_group_keys,
        [],
        [],
        [],
        [],
        [],
    )

    assert y_true is not None and y_pred is not None and y_raw is not None
    assert weights is None

    ev_value, _, _, metrics = algo._compute_explained_variance_metric(
        y_true,
        y_pred,
        mask_tensor=weights,
        y_true_tensor_raw=y_raw,
    )

    expected_ev = safe_explained_variance(
        torch.cat(primary_true_batches).numpy().reshape(-1),
        torch.cat(primary_pred_batches).numpy().reshape(-1),
        None,
    )

    assert ev_value == pytest.approx(expected_ev)
    assert metrics["n_samples"] == pytest.approx(float(sum(t.numel() for t in primary_true_batches)))


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


def test_update_explained_variance_tracking_handles_bad_and_good_values() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    logger = _CaptureLogger()
    algo.logger = logger
    algo._bad_explained_counter = 1
    algo._value_scale_auto_thaw_bad_ev = 2
    algo._value_scale_frozen = True
    algo._value_scale_latest_ret_abs_p95 = 1.5
    algo._value_scale_frame_stable = False
    algo._value_scale_stable_counter = 0
    algo._is_value_scale_frame_stable = MethodType(
        lambda self, _p95, ev: ev >= 0.4,
        algo,
    )

    algo._update_explained_variance_tracking(-0.2)

    assert algo._bad_explained_counter == 2
    assert algo._last_explained_variance == pytest.approx(-0.2)
    assert algo._vf_clip_latest_ev == pytest.approx(-0.2)
    assert not algo._value_scale_frozen
    assert logger.records["train/value_scale_auto_thaw"] == [2.0]
    assert logger.records["train/value_scale_frame_stable"][-1] == 0.0
    assert algo._value_scale_stable_counter == 0

    logger.records = {}
    algo._value_scale_latest_ret_abs_p95 = 2.5

    algo._update_explained_variance_tracking(0.6)

    assert algo._bad_explained_counter == 0
    assert algo._last_explained_variance == pytest.approx(0.6)
    assert algo._vf_clip_latest_ev == pytest.approx(0.6)
    assert algo._value_scale_frame_stable
    assert algo._value_scale_stable_counter == 1
    assert logger.records["train/value_scale_frame_stable"][-1] == 1.0
    assert logger.records["train/value_scale_ret_abs_p95"][-1] == pytest.approx(2.5)


def test_update_explained_variance_tracking_clears_state_when_metric_missing() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    logger = _CaptureLogger()
    algo.logger = logger
    algo._bad_explained_counter = 4
    algo._last_explained_variance = 0.2
    algo._vf_clip_latest_ev = 0.2
    algo._value_scale_latest_ret_abs_p95 = 1.0
    algo._value_scale_frame_stable = True
    algo._value_scale_stable_counter = 5

    algo._update_explained_variance_tracking(None)

    assert algo._bad_explained_counter == 4
    assert algo._last_explained_variance is None
    assert algo._vf_clip_latest_ev is None
    assert not algo._value_scale_frame_stable
    assert algo._value_scale_stable_counter == 0
    assert logger.records["train/value_scale_frame_stable"][-1] == 0.0
    assert logger.records["train/value_scale_ret_abs_p95"][-1] == pytest.approx(1.0)


def test_update_explained_variance_warning_streak_triggers_logging() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    logger = _CaptureLogger()
    algo.logger = logger
    algo._explained_variance_warn_streak = 0

    algo._update_explained_variance_warning_streak(None)
    assert algo._explained_variance_warn_streak == 0

    algo._update_explained_variance_warning_streak(0.2)
    algo._update_explained_variance_warning_streak(0.25)
    algo._update_explained_variance_warning_streak(0.3)

    assert algo._explained_variance_warn_streak == 3
    assert logger.records["warn/explained_variance_low"] == [0.3]

    algo._update_explained_variance_warning_streak(0.6)
    assert algo._explained_variance_warn_streak == 0
    assert logger.records["warn/explained_variance_low"] == [0.3]


def test_compute_entropy_boost_scales_with_bad_explained_variance() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._bad_explained_patience = 1
    algo._entropy_boost_factor = 2.0
    algo._entropy_boost_cap = 3.0

    algo._bad_explained_counter = 1
    assert algo._compute_entropy_boost(0.5) == pytest.approx(0.5)

    algo._bad_explained_counter = 3
    assert algo._compute_entropy_boost(0.5) == pytest.approx(2.0)

    algo._entropy_boost_cap = 1.25
    algo._bad_explained_counter = 5
    assert algo._compute_entropy_boost(0.5) == pytest.approx(1.25)


def test_compute_cvar_weight_respects_prerequisites_and_ramps() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo._cvar_ramp_progress = 5
    algo._last_explained_variance = None
    algo._value_scale_update_count = 0
    algo._value_scale_warmup_updates = 3
    algo._value_scale_requires_stability = False
    algo._value_scale_frame_stable = False
    algo._value_scale_stable_counter = 0
    algo._value_scale_stability_patience = 2
    algo._cvar_activation_threshold = 0.4
    algo._cvar_activation_hysteresis = 0.05
    algo._cvar_ramp_updates = 3
    algo._cvar_weight_target = 0.6

    assert algo._compute_cvar_weight() == pytest.approx(0.0)
    assert algo._cvar_ramp_progress == 0

    algo._last_explained_variance = 0.5
    algo._value_scale_update_count = 1
    assert algo._compute_cvar_weight() == pytest.approx(0.0)
    assert algo._cvar_ramp_progress == 0

    algo._value_scale_update_count = 5
    algo._value_scale_requires_stability = True
    algo._value_scale_frame_stable = False
    assert algo._compute_cvar_weight() == pytest.approx(0.0)
    assert algo._cvar_ramp_progress == 0

    algo._value_scale_frame_stable = True
    algo._value_scale_stable_counter = 1
    assert algo._compute_cvar_weight() == pytest.approx(0.0)
    assert algo._cvar_ramp_progress == 0

    algo._value_scale_stable_counter = 2
    algo._last_explained_variance = 0.3
    assert algo._compute_cvar_weight() == pytest.approx(0.0)
    assert algo._cvar_ramp_progress == 0

    algo._last_explained_variance = 0.5
    weight_step_1 = algo._compute_cvar_weight()
    assert weight_step_1 == pytest.approx(0.6 / 3)
    assert algo._cvar_ramp_progress == 1

    weight_step_2 = algo._compute_cvar_weight()
    assert weight_step_2 == pytest.approx(0.6 * 2 / 3)
    assert algo._cvar_ramp_progress == 2

    weight_step_3 = algo._compute_cvar_weight()
    assert weight_step_3 == pytest.approx(0.6)
    assert algo._cvar_ramp_progress == 3

    weight_step_4 = algo._compute_cvar_weight()
    assert weight_step_4 == pytest.approx(0.6)
    assert algo._cvar_ramp_progress == 3

    algo._cvar_ramp_updates = 0
    algo._cvar_weight_target = 2.0
    final_weight = algo._compute_cvar_weight()
    assert final_weight == pytest.approx(1.0)
    assert algo._cvar_ramp_progress == 0


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
