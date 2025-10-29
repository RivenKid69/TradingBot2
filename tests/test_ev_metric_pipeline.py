"""Integration-style tests for explained variance utilities in DistributionalPPO."""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pytest
import torch


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is not None:
        return module
    module = types.ModuleType(name)
    sys.modules[name] = module
    if "." in name:
        parent_name, attr = name.rsplit(".", 1)
        parent_module = _ensure_module(parent_name)
        setattr(parent_module, attr, module)
    return module


def _install_rl_stubs() -> None:
    if "sb3_contrib" not in sys.modules:
        sb3_contrib = _ensure_module("sb3_contrib")
        sb3_contrib.RecurrentPPO = object
    policies = _ensure_module("sb3_contrib.common.recurrent.policies")
    if not hasattr(policies, "RecurrentActorCriticPolicy"):
        policies.RecurrentActorCriticPolicy = object
    buffers = _ensure_module("sb3_contrib.common.recurrent.buffers")
    if not hasattr(buffers, "RecurrentRolloutBuffer"):
        buffers.RecurrentRolloutBuffer = object
    type_aliases = _ensure_module("sb3_contrib.common.recurrent.type_aliases")
    if not hasattr(type_aliases, "RNNStates"):
        type_aliases.RNNStates = tuple

    callbacks = _ensure_module("stable_baselines3.common.callbacks")
    if not hasattr(callbacks, "BaseCallback"):
        callbacks.BaseCallback = object
    if not hasattr(callbacks, "CallbackList"):
        callbacks.CallbackList = list
    if not hasattr(callbacks, "EvalCallback"):
        callbacks.EvalCallback = object

    vec_env = _ensure_module("stable_baselines3.common.vec_env")
    if not hasattr(vec_env, "VecEnv"):
        vec_env.VecEnv = object
    vec_norm = _ensure_module("stable_baselines3.common.vec_env.vec_normalize")
    if not hasattr(vec_norm, "VecNormalize"):
        vec_norm.VecNormalize = object

    type_aliases_common = _ensure_module("stable_baselines3.common.type_aliases")
    if not hasattr(type_aliases_common, "GymEnv"):
        type_aliases_common.GymEnv = object

    running_mean_std = _ensure_module("stable_baselines3.common.running_mean_std")
    if not hasattr(running_mean_std, "RunningMeanStd"):
        running_mean_std.RunningMeanStd = object

    save_util = _ensure_module("stable_baselines3.common.save_util")
    if not hasattr(save_util, "load_from_zip_file"):
        save_util.load_from_zip_file = lambda *args, **kwargs: None


def _load_distributional_ppo_module():
    module_name = "distributional_ppo_test_stub_pipeline"
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    _install_rl_stubs()
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).resolve().parents[1] / "distributional_ppo.py",
    )
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    sys.modules[module_name] = module
    return module


MODULE = _load_distributional_ppo_module()


class DummyLogger:
    def __init__(self) -> None:
        self.records: dict[str, list[float]] = {}

    def record(self, key: str, value: float) -> None:
        self.records.setdefault(key, []).append(float(value))


@dataclass
class DummyModule:
    """Minimal stub implementing the pieces DistributionalPPO helpers need."""

    normalize_returns: bool = False
    value_target_scale: float = 1.0
    _value_target_scale_effective: float = 1.0
    _ret_mean_snapshot: float = 0.0
    _ret_std_snapshot: float = 1.0
    _raw_scale: float = 1.0
    _raw_shift: float = 0.0
    logger: DummyLogger = field(default_factory=DummyLogger)

    def _to_raw_returns(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self._raw_scale + self._raw_shift


def _call_compute_ev(
    *,
    y_true: Optional[torch.Tensor],
    y_pred: Optional[torch.Tensor],
    mask: Optional[torch.Tensor] = None,
    y_true_raw: Optional[torch.Tensor] = None,
    group_keys: Optional[Sequence[str]] = None,
    instance: Optional[DummyModule] = None,
) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor], dict[str, Any]]:
    dummy = instance or DummyModule()
    method = MODULE.DistributionalPPO._compute_explained_variance_metric
    return method(
        dummy,
        y_true,
        y_pred,
        mask_tensor=mask,
        y_true_tensor_raw=y_true_raw,
        group_keys=group_keys,
    )


def _call_build_ev_tensors(
    *,
    targets_norm: Sequence[torch.Tensor],
    preds_norm: Sequence[torch.Tensor],
    targets_raw: Sequence[torch.Tensor],
    weights: Sequence[torch.Tensor],
    group_keys: Sequence[Sequence[str]],
    reserve_targets_norm: Sequence[torch.Tensor],
    reserve_preds_norm: Sequence[torch.Tensor],
    reserve_targets_raw: Sequence[torch.Tensor],
    reserve_weights: Sequence[torch.Tensor],
    reserve_group_keys: Sequence[Sequence[str]],
):
    method = MODULE.DistributionalPPO._build_explained_variance_tensors
    dummy = DummyModule()
    return method(
        dummy,
        targets_norm,
        preds_norm,
        targets_raw,
        weights,
        group_keys,
        reserve_targets_norm,
        reserve_preds_norm,
        reserve_targets_raw,
        reserve_weights,
        reserve_group_keys,
    )


def test_build_explained_variance_tensors_prefers_primary_batches() -> None:
    y_true_batches = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
    y_pred_batches = [torch.tensor([0.9, 2.1]), torch.tensor([3.2])]
    y_raw_batches = [torch.tensor([10.0, 20.0]), torch.tensor([30.0])]
    mask_batches = [torch.tensor([1.0, 0.5]), torch.tensor([0.25])]
    key_batches = [["A", "B"], ["C"]]

    result = _call_build_ev_tensors(
        targets_norm=y_true_batches,
        preds_norm=y_pred_batches,
        targets_raw=y_raw_batches,
        weights=mask_batches,
        group_keys=key_batches,
        reserve_targets_norm=[torch.tensor([100.0])],
        reserve_preds_norm=[torch.tensor([99.0])],
        reserve_targets_raw=[torch.tensor([1000.0])],
        reserve_weights=[torch.tensor([1.0])],
        reserve_group_keys=[["reserve"]],
    )

    y_true_tensor, y_pred_tensor, y_true_raw, mask_tensor, group_keys = result

    assert y_true_tensor is not None and torch.allclose(
        y_true_tensor.flatten(), torch.tensor([1.0, 2.0, 3.0])
    )
    assert y_pred_tensor is not None and torch.allclose(
        y_pred_tensor.flatten(), torch.tensor([0.9, 2.1, 3.2])
    )
    assert y_true_raw is not None and torch.allclose(
        y_true_raw.flatten(), torch.tensor([10.0, 20.0, 30.0])
    )
    assert mask_tensor is not None and torch.allclose(
        mask_tensor.flatten(), torch.tensor([1.0, 0.5, 0.25])
    )
    assert group_keys == ["A", "B", "C"]


def test_build_explained_variance_tensors_falls_back_to_reserve_batches() -> None:
    result = _call_build_ev_tensors(
        targets_norm=[torch.tensor([])],
        preds_norm=[torch.tensor([])],
        targets_raw=[torch.tensor([])],
        weights=[torch.tensor([])],
        group_keys=[[]],
        reserve_targets_norm=[torch.tensor([5.0, 6.0])],
        reserve_preds_norm=[torch.tensor([5.5, 6.5])],
        reserve_targets_raw=[torch.tensor([50.0, 60.0])],
        reserve_weights=[torch.tensor([0.2, 0.8])],
        reserve_group_keys=[["X", "Y"]],
    )

    y_true_tensor, y_pred_tensor, y_true_raw, mask_tensor, group_keys = result

    assert y_true_tensor is not None and torch.allclose(
        y_true_tensor.flatten(), torch.tensor([5.0, 6.0])
    )
    assert y_pred_tensor is not None and torch.allclose(
        y_pred_tensor.flatten(), torch.tensor([5.5, 6.5])
    )
    assert y_true_raw is not None and torch.allclose(
        y_true_raw.flatten(), torch.tensor([50.0, 60.0])
    )
    assert mask_tensor is not None and torch.allclose(
        mask_tensor.flatten(), torch.tensor([0.2, 0.8])
    )
    assert group_keys == ["X", "Y"]


def test_build_explained_variance_tensors_handles_all_empty_batches() -> None:
    result = _call_build_ev_tensors(
        targets_norm=[torch.tensor([])],
        preds_norm=[torch.tensor([])],
        targets_raw=[torch.tensor([])],
        weights=[torch.tensor([])],
        group_keys=[[]],
        reserve_targets_norm=[torch.tensor([])],
        reserve_preds_norm=[torch.tensor([])],
        reserve_targets_raw=[torch.tensor([])],
        reserve_weights=[torch.tensor([])],
        reserve_group_keys=[[]],
    )

    assert result == (None, None, None, None, None)


def test_compute_explained_variance_metric_basic_path() -> None:
    y_true = torch.tensor([1.0, 2.0, 3.0])
    y_pred = torch.tensor([0.9, 2.1, 2.7])

    ev, true_eval, pred_eval, metrics = _call_compute_ev(y_true=y_true, y_pred=y_pred)

    expected = MODULE.safe_explained_variance(
        y_true.detach().numpy(), y_pred.detach().numpy()
    )

    assert ev == pytest.approx(expected)
    assert true_eval is not None and torch.allclose(true_eval, y_true)
    assert pred_eval is not None and torch.allclose(pred_eval, y_pred)
    assert metrics["n_samples"] == pytest.approx(3.0)
    assert metrics["ev_global"] == pytest.approx(expected)
    assert math.isfinite(metrics["corr"])


def test_compute_explained_variance_metric_fallback_to_raw_records_flag() -> None:
    instance = DummyModule(_raw_scale=2.0, _raw_shift=0.5, logger=DummyLogger())
    y_true = torch.tensor([0.0, 0.0, 0.0])
    y_pred = torch.tensor([-0.1, 0.0, 0.1])
    y_true_raw = torch.tensor([1.0, 2.0, 3.0])

    ev, true_eval, pred_eval, metrics = _call_compute_ev(
        y_true=y_true,
        y_pred=y_pred,
        y_true_raw=y_true_raw,
        instance=instance,
    )

    expected = MODULE.safe_explained_variance(
        y_true_raw.numpy(),
        instance._to_raw_returns(y_pred).detach().numpy(),
    )

    assert ev == pytest.approx(expected)
    assert true_eval is not None and torch.allclose(true_eval, y_true)
    assert pred_eval is not None and torch.allclose(pred_eval, y_pred)
    assert metrics["ev_global"] == pytest.approx(expected)
    assert instance.logger.records["train/value_explained_variance_fallback"] == [1.0]


def test_compute_explained_variance_metric_with_mask_and_groups() -> None:
    instance = DummyModule(logger=DummyLogger())
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_pred = torch.tensor([0.9, 2.2, 2.5, 3.5])
    mask = torch.tensor([1.0, 0.0, 2.0, 3.0])
    groups = ["A", "B", "A", "B"]

    ev, true_eval, pred_eval, metrics = _call_compute_ev(
        y_true=y_true,
        y_pred=y_pred,
        mask=mask,
        group_keys=groups,
        instance=instance,
    )

    assert true_eval is not None
    assert pred_eval is not None
    weights_filtered = mask[mask > 0.0].numpy()
    group_keys_effective = groups[: true_eval.numel()]
    y_true_np = true_eval.numpy()
    y_pred_np = pred_eval.numpy()
    expected_ev = MODULE.safe_explained_variance(y_true_np, y_pred_np, weights_filtered)
    expected_grouped, expected_summary = MODULE.compute_grouped_explained_variance(
        y_true_np,
        y_pred_np,
        group_keys_effective,
        weights=weights_filtered,
    )

    assert ev == pytest.approx(expected_ev)
    assert true_eval.numel() == len(weights_filtered)
    assert pred_eval.numel() == len(weights_filtered)
    assert metrics["n_samples"] == pytest.approx(float(len(weights_filtered)))
    assert metrics["ev_global"] == pytest.approx(expected_ev)
    assert set(metrics["ev_grouped"]) == set(expected_grouped)
    for key, value in expected_grouped.items():
        actual = metrics["ev_grouped"][key]
        if math.isnan(value):
            assert math.isnan(actual)
        else:
            assert actual == pytest.approx(value)
    assert expected_summary["mean_weighted"] is not None
    assert expected_summary["mean_unweighted"] is not None
    assert expected_summary["median"] is not None
    assert metrics["ev_mean_weighted"] == pytest.approx(expected_summary["mean_weighted"])
    assert metrics["ev_mean_unweighted"] == pytest.approx(expected_summary["mean_unweighted"])
    assert metrics["ev_median"] == pytest.approx(expected_summary["median"])
