import sys
import types

from collections import deque

import numpy as np
import pytest

try:  # pragma: no cover - optional dependency for tensor helpers
    import torch
except ModuleNotFoundError:  # pragma: no cover - graceful fallback for test-only usage
    torch = None
torch_is_stub = getattr(torch, "__test_stub__", False)


def _install_rl_stubs() -> None:
    if "sb3_contrib" not in sys.modules:
        sb3_contrib = types.ModuleType("sb3_contrib")

        class _RecurrentPPO:  # pragma: no cover - stub for import
            pass

        sb3_contrib.RecurrentPPO = _RecurrentPPO
        sys.modules["sb3_contrib"] = sb3_contrib

        sb3_common = types.ModuleType("sb3_contrib.common")
        sys.modules["sb3_contrib.common"] = sb3_common

        sb3_recurrent = types.ModuleType("sb3_contrib.common.recurrent")
        sys.modules["sb3_contrib.common.recurrent"] = sb3_recurrent

        policies = types.ModuleType("sb3_contrib.common.recurrent.policies")

        class _RecurrentActorCriticPolicy:  # pragma: no cover - stub for import
            pass

        policies.RecurrentActorCriticPolicy = _RecurrentActorCriticPolicy
        sys.modules["sb3_contrib.common.recurrent.policies"] = policies

        buffers = types.ModuleType("sb3_contrib.common.recurrent.buffers")

        class _RecurrentRolloutBuffer:  # pragma: no cover - stub for import
            pass

        buffers.RecurrentRolloutBuffer = _RecurrentRolloutBuffer
        sys.modules["sb3_contrib.common.recurrent.buffers"] = buffers

        type_aliases = types.ModuleType("sb3_contrib.common.recurrent.type_aliases")
        type_aliases.RNNStates = tuple
        sys.modules["sb3_contrib.common.recurrent.type_aliases"] = type_aliases

    if "stable_baselines3" not in sys.modules:
        sb3 = types.ModuleType("stable_baselines3")
        sys.modules["stable_baselines3"] = sb3

        common = types.ModuleType("stable_baselines3.common")
        sys.modules["stable_baselines3.common"] = common

        callbacks = types.ModuleType("stable_baselines3.common.callbacks")

        class _BaseCallback:  # pragma: no cover - stub for import
            pass

        class _CallbackList:  # pragma: no cover - stub for import
            pass

        class _EvalCallback:  # pragma: no cover - stub for import
            pass

        callbacks.BaseCallback = _BaseCallback
        callbacks.CallbackList = _CallbackList
        callbacks.EvalCallback = _EvalCallback
        sys.modules["stable_baselines3.common.callbacks"] = callbacks

        vec_env = types.ModuleType("stable_baselines3.common.vec_env")

        class _VecEnv:  # pragma: no cover - stub for import
            pass

        vec_env.VecEnv = _VecEnv
        sys.modules["stable_baselines3.common.vec_env"] = vec_env

        vec_norm = types.ModuleType("stable_baselines3.common.vec_env.vec_normalize")

        class _VecNormalize:  # pragma: no cover - stub for import
            pass

        vec_norm.VecNormalize = _VecNormalize
        sys.modules["stable_baselines3.common.vec_env.vec_normalize"] = vec_norm

        type_aliases_common = types.ModuleType("stable_baselines3.common.type_aliases")
        type_aliases_common.GymEnv = object
        sys.modules["stable_baselines3.common.type_aliases"] = type_aliases_common

        running_mean_std = types.ModuleType("stable_baselines3.common.running_mean_std")

        class _RunningMeanStd:  # pragma: no cover - stub for import
            def __init__(self, shape=()):
                self.mean = np.zeros(shape, dtype=float)
                self.var = np.ones(shape, dtype=float)
                self.count = 0.0

            def update(self, array_like) -> None:
                values = np.asarray(array_like, dtype=float)
                if values.size == 0:
                    return
                self.mean = np.mean(values, axis=0)
                self.var = np.var(values, axis=0)
                self.count = float(values.size)

        running_mean_std.RunningMeanStd = _RunningMeanStd
        sys.modules["stable_baselines3.common.running_mean_std"] = running_mean_std

    if "torch" not in sys.modules:
        torch_module = types.ModuleType("torch")

        class _NoGrad:  # pragma: no cover - stub context manager
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

        def _not_impl(*_args, **_kwargs):  # pragma: no cover - guard unexpected usage
            raise NotImplementedError("torch functionality is not available in tests")

        class _TensorAlias:  # pragma: no cover - type placeholder
            pass

        class _Device:  # pragma: no cover - type placeholder
            def __init__(self, *_args, **_kwargs) -> None:
                pass

        torch_module.Tensor = _TensorAlias
        torch_module.device = _Device
        torch_module.float32 = np.float32
        torch_module.__test_stub__ = True
        torch_module.tensor = _not_impl
        torch_module.as_tensor = _not_impl
        torch_module.quantile = _not_impl
        torch_module.arange = _not_impl
        torch_module.full = _not_impl
        torch_module.sum = _not_impl
        torch_module.cumsum = _not_impl
        torch_module.sort = _not_impl
        torch_module.gather = _not_impl
        torch_module.searchsorted = _not_impl
        torch_module.clamp = _not_impl
        torch_module.zeros_like = _not_impl
        torch_module.no_grad = lambda: _NoGrad()
        torch_module.set_float32_matmul_precision = lambda *_args, **_kwargs: None

        nn_module = types.ModuleType("torch.nn")
        functional_module = types.ModuleType("torch.nn.functional")
        functional_module.softmax = _not_impl
        functional_module.mse_loss = _not_impl
        nn_module.functional = functional_module
        torch_module.nn = nn_module

        optim_module = types.ModuleType("torch.optim")
        optim_module.Optimizer = object
        torch_module.optim = optim_module

        sys.modules["torch"] = torch_module
        sys.modules["torch.nn"] = nn_module
        sys.modules["torch.nn.functional"] = functional_module
        sys.modules["torch.optim"] = optim_module

    if "gymnasium" not in sys.modules:
        gymnasium = types.ModuleType("gymnasium")
        spaces = types.ModuleType("gymnasium.spaces")

        class _Space:  # pragma: no cover - stub for import
            pass

        class _Box(_Space):  # pragma: no cover - stub for import
            pass

        class _Discrete(_Space):  # pragma: no cover - stub for import
            pass

        spaces.Box = _Box
        spaces.Discrete = _Discrete
        gymnasium.spaces = spaces
        sys.modules["gymnasium"] = gymnasium
        sys.modules["gymnasium.spaces"] = spaces


_install_rl_stubs()

from distributional_ppo import DistributionalPPO
from stable_baselines3.common.running_mean_std import RunningMeanStd


@pytest.mark.skipif(
    torch is None or torch_is_stub, reason="torch is required for tensor-based checks"
)
def test_value_target_outlier_fractions_detects_raw_outliers() -> None:
    values = torch.tensor([-15.0, -11.0, 0.0, 9.0, 25.0], dtype=torch.float32)
    below, above = DistributionalPPO._value_target_outlier_fractions(values, -10.0, 10.0)

    assert below > 0.0
    assert above > 0.0
    assert below == pytest.approx(2.0 / 5.0)
    assert above == pytest.approx(1.0 / 5.0)


def test_value_scale_handles_outlier_batch_with_smoothing() -> None:
    model = DistributionalPPO.__new__(DistributionalPPO)
    model.normalize_returns = True
    model.ret_clip = 5.0
    model.value_target_scale = 1.0
    model.ret_rms = RunningMeanStd(shape=())
    model.ret_rms.mean[...] = 0.0
    model.ret_rms.var[...] = 1.0
    model.ret_rms.count = 1.0
    model._ret_mean_value = 0.0
    model._ret_std_value = 1.0
    model._ret_mean_snapshot = 0.0
    model._ret_std_snapshot = 1.0
    model._pending_rms = None
    model._pending_ret_mean = None
    model._pending_ret_std = None
    model._value_scale_ema_beta = 0.2
    model._value_scale_max_rel_step = 0.5
    model._value_scale_std_floor = 1e-2
    model._value_scale_window_updates = 0
    model._value_scale_recent_stats = deque()
    model._value_scale_stats_initialized = True
    model._value_scale_stats_mean = 0.0
    model._value_scale_stats_second = 1.0
    model._value_target_scale_effective = 1.0 / (model.ret_clip * model._ret_std_value)
    model._value_target_scale_robust = 1.0
    model.running_v_min = -model.ret_clip
    model.running_v_max = model.ret_clip
    model.v_range_initialized = True
    model.v_range_ema_alpha = 0.1
    model.policy = types.SimpleNamespace(update_atoms=lambda *_args, **_kwargs: None)

    class _Recorder:
        def __init__(self) -> None:
            self.records: dict[str, float] = {}

        def record(self, key: str, value: float) -> None:
            self.records[key] = float(value)

    model.logger = _Recorder()

    base_returns = np.array([-1.0, -0.5, 0.25, 0.5, 1.0], dtype=np.float32)
    scales = [1.0, 1.2, 0.8, 1.0, 25.0, 1.1]

    stds: list[float] = []
    scales_effective: list[float] = []
    spans: list[float] = []
    mse_history: list[float] = []

    prev_std = model._ret_std_snapshot

    for scale in scales:
        returns = base_returns * scale
        model._pending_rms = RunningMeanStd(shape=())
        model._pending_rms.update(returns)

        model._finalize_return_stats()

        std = float(model._ret_std_snapshot)
        mean = float(model._ret_mean_snapshot)
        stds.append(std)
        scales_effective.append(float(model._value_target_scale_effective))
        span = (model.running_v_max - model.running_v_min) * std
        spans.append(span)

        normalized = np.clip((returns - mean) / max(std, 1e-8), -model.ret_clip, model.ret_clip)
        mse_history.append(float(np.mean(normalized**2)))

        assert std <= prev_std * (1.0 + model._value_scale_max_rel_step) + 1e-8
        assert std >= prev_std / (1.0 + model._value_scale_max_rel_step) - 1e-8
        prev_std = std

    assert max(stds) < 3.0
    assert min(scales_effective) > 1.0 / (model.ret_clip * 3.0)
    assert max(spans) < model.ret_clip * 4.0
    assert max(mse_history) < model.ret_clip**2
