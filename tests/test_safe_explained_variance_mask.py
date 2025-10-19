import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


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
        type_aliases.RNNStates = object

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
    module_name = "distributional_ppo_test_stub"
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


def test_safe_explained_variance_ignores_padded_entries():
    module = _load_distributional_ppo_module()
    safe_ev = module.safe_explained_variance

    y_true = np.array([1.0, 2.0, 0.0, 0.0], dtype=float)
    y_pred = np.array([0.5, 2.5, 0.0, 0.0], dtype=float)
    mask = np.array([1.0, 1.0, 0.0, 0.0], dtype=float)

    ev_masked = safe_ev(y_true, y_pred, mask)
    ev_truncated = safe_ev(y_true[:2], y_pred[:2])
    ev_unmasked = safe_ev(y_true, y_pred)

    assert np.isclose(ev_masked, ev_truncated, equal_nan=True)
    assert not np.isclose(ev_unmasked, ev_truncated, equal_nan=True)


def test_safe_explained_variance_fractional_mask_weights_positive():
    module = _load_distributional_ppo_module()
    safe_ev = module.safe_explained_variance

    y_true = np.array([0.5, 1.25, 2.0, -0.25], dtype=float)
    y_pred = np.array([0.45, 1.3, 1.9, -0.2], dtype=float)
    mask = np.array([0.2, 0.4, 0.3, 0.1], dtype=float)

    ev_weighted = safe_ev(y_true, y_pred, mask)

    assert np.isfinite(ev_weighted)
    assert ev_weighted > 0.0


def test_safe_explained_variance_uses_unbiased_variance() -> None:
    module = _load_distributional_ppo_module()
    safe_ev = module.safe_explained_variance

    y_true = np.array([1.0, 2.0, 4.0], dtype=float)
    y_pred = np.array([0.8, 1.9, 3.7], dtype=float)

    ev = safe_ev(y_true, y_pred)

    var_true = np.var(y_true, ddof=1)
    var_res = np.var(y_true - y_pred, ddof=1)
    expected = 1.0 - var_res / var_true

    assert ev == pytest.approx(expected)


def test_safe_explained_variance_weighted_unbiased() -> None:
    module = _load_distributional_ppo_module()
    safe_ev = module.safe_explained_variance

    y_true = np.array([0.5, 1.25, 2.0, -0.25], dtype=float)
    y_pred = np.array([0.45, 1.3, 1.9, -0.2], dtype=float)
    weights = np.array([0.2, 0.4, 0.3, 0.1], dtype=float)

    ev = safe_ev(y_true, y_pred, weights)

    sum_w = np.sum(weights)
    mean_true = np.sum(weights * y_true) / sum_w
    sum_w_sq = np.sum(weights**2)
    denom = sum_w - sum_w_sq / sum_w
    var_true = np.sum(weights * (y_true - mean_true) ** 2) / denom
    residual = y_true - y_pred
    residual_mean = np.sum(weights * residual) / sum_w
    var_res = np.sum(weights * (residual - residual_mean) ** 2) / denom
    expected = 1.0 - var_res / var_true

    assert ev == pytest.approx(expected)

    # Regression test: constant offset should yield the same explained variance
    # for masked weighted batches as the unweighted, truncated inputs.
    y_true = np.array([1.0, 2.0, 0.0, 0.0], dtype=float)
    y_pred = y_true - 1.0
    weights = np.array([1.0, 1.0, 0.0, 0.0], dtype=float)

    ev_weighted = safe_ev(y_true, y_pred, weights)
    ev_truncated = safe_ev(y_true[:2], y_pred[:2])

    assert ev_weighted == pytest.approx(ev_truncated)


def test_safe_explained_variance_skewed_weights_returns_nan() -> None:
    module = _load_distributional_ppo_module()
    safe_ev = module.safe_explained_variance

    y_true = np.array([1.0, 1.0 + 1e-8, 1.0 - 1e-8], dtype=float)
    y_pred = np.array([0.0, 1e154, -1e154], dtype=float)
    # Heavily skewed mask weights place almost all mass on the first entry while
    # leaving tiny but non-zero weights on the extreme residuals. This used to
    # produce ``-inf`` due to overflow in the weighted ratio.
    weights = np.array([1.0, 1e-15, 1e-15], dtype=float)

    result = safe_ev(y_true, y_pred, weights)

    assert np.isnan(result)


def test_safe_explained_variance_short_mask_returns_nan() -> None:
    module = _load_distributional_ppo_module()
    safe_ev = module.safe_explained_variance

    y_true = np.array([1.0, 1.0, 2.0], dtype=float)
    y_pred = np.array([1.0, 1.0, 2.0], dtype=float)
    weights = np.array([1.0, 0.0], dtype=float)

    result = safe_ev(y_true, y_pred, weights)

    assert np.isnan(result)
