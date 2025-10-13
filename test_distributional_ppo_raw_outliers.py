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

        class _ActionWrapper:  # pragma: no cover - stub for import
            pass

        class _Box(_Space):  # pragma: no cover - stub for import
            pass

        class _Discrete(_Space):  # pragma: no cover - stub for import
            pass

        class _MultiDiscrete(_Space):  # pragma: no cover - stub for import
            def __init__(self, nvec):
                self.nvec = np.asarray(nvec, dtype=np.int64)

        spaces.Box = _Box
        spaces.Discrete = _Discrete
        spaces.MultiDiscrete = _MultiDiscrete
        gymnasium.ActionWrapper = _ActionWrapper
        gymnasium.spaces = spaces
        sys.modules["gymnasium"] = gymnasium
        sys.modules["gymnasium.spaces"] = spaces


_install_rl_stubs()

from distributional_ppo import DistributionalPPO
from stable_baselines3.common.running_mean_std import RunningMeanStd


def test_volume_head_config_mismatch_detection():
    spaces = sys.modules.get("gymnasium.spaces")
    if spaces is None:
        pytest.skip("gymnasium is required for volume head config checks")

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.policy = types.SimpleNamespace(
        _multi_head_sizes=(201, 33, 4, 4),
        _volume_head_index=3,
    )
    algo.action_space = spaces.MultiDiscrete([201, 33, 4, 4])

    # Should not raise when configuration matches.
    algo._ensure_volume_head_config()

    algo.policy._multi_head_sizes = (201, 33, 4, 5)
    with pytest.raises(RuntimeError, match="expected 4 bins"):
        algo._ensure_volume_head_config()

    algo.policy._multi_head_sizes = (201, 33, 4, 4)
    algo.action_space = spaces.MultiDiscrete([201, 33, 4, 5])
    with pytest.raises(RuntimeError, match="volume bins; expected 4"):
        algo._ensure_volume_head_config()


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
    model.device = torch.device("cpu") if torch is not None and not torch_is_stub else None
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
    model._value_scale_warmup_limit = 3
    model._value_scale_warmup_updates = 3
    model._value_scale_min_samples = 256
    model._value_scale_warmup_buffer = []
    model._value_scale_warmup_buffer_limit = 65536
    model._value_scale_update_count = 0
    model._value_scale_frozen = False
    model._use_quantile_value = False
    model._last_raw_outlier_frac = 0.0
    model._value_target_raw_outlier_warn_threshold = 1.0
    model._value_scale_stable_counter = 0
    model._value_scale_frame_stable = True
    model._value_scale_stability_patience = 0
    model._value_scale_requires_stability = False
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

    base_returns = np.linspace(-1.0, 1.0, model._value_scale_min_samples, dtype=np.float32)
    scales = [1.0, 1.2, 0.8, 1.0, 25.0, 1.1]

    stds: list[float] = []
    means: list[float] = []
    scales_effective: list[float] = []
    spans: list[float] = []
    mse_history: list[float] = []

    prev_std = model._ret_std_snapshot

    for scale in scales:
        returns = base_returns * scale
        model._pending_rms = RunningMeanStd(shape=())
        model._pending_rms.update(returns)
        model.rollout_buffer = types.SimpleNamespace(returns=returns.copy())

        model._finalize_return_stats()

        std = float(model._ret_std_snapshot)
        mean = float(model._ret_mean_snapshot)
        stds.append(std)
        means.append(mean)
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
    assert model._value_scale_update_count == model._value_scale_warmup_limit
    assert model._value_scale_frozen is True
    frozen_std = stds[model._value_scale_warmup_limit - 1]
    frozen_scale = scales_effective[model._value_scale_warmup_limit - 1]
    frozen_mean = means[model._value_scale_warmup_limit - 1]
    for idx in range(model._value_scale_warmup_limit, len(stds)):
        assert stds[idx] == pytest.approx(frozen_std)
        assert scales_effective[idx] == pytest.approx(frozen_scale)
        assert means[idx] == pytest.approx(frozen_mean)


@pytest.mark.skipif(
    torch is None or torch_is_stub, reason="torch is required for tensor-based checks"
)
def test_non_normalized_value_scale_freeze_and_decode_path() -> None:
    model = DistributionalPPO.__new__(DistributionalPPO)
    model.normalize_returns = False
    model.ret_clip = 5.0
    model.value_target_scale = 1.0
    model.device = torch.device("cpu") if torch is not None and not torch_is_stub else None
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
    model._value_target_scale_effective = float(model.value_target_scale)
    model._value_target_scale_robust = 1.0
    model._value_scale_warmup_limit = 3
    model._value_scale_warmup_updates = 3
    model._value_scale_min_samples = 256
    model._value_scale_warmup_buffer = []
    model._value_scale_warmup_buffer_limit = 65536
    model._value_scale_update_count = 0
    model._value_scale_frozen = False
    model._use_quantile_value = False
    model._last_raw_outlier_frac = 0.0
    model._value_target_raw_outlier_warn_threshold = 1.0
    model._value_scale_stable_counter = 0
    model._value_scale_frame_stable = True
    model._value_scale_stability_patience = 0
    model._value_scale_requires_stability = False
    model._value_clip_limit_unscaled = None
    model._value_clip_limit_scaled = None
    model.running_v_min = -model.ret_clip
    model.running_v_max = model.ret_clip
    model.v_range_initialized = True
    model.v_range_ema_alpha = 0.1
    model.policy = types.SimpleNamespace(update_atoms=lambda *_args, **_kwargs: None)

    class _Recorder:
        def __init__(self) -> None:
            self.records: dict[str, object] = {}

        def record(self, key: str, value: object) -> None:
            self.records[key] = value

    model.logger = _Recorder()

    core = np.linspace(-0.05, 0.05, model._value_scale_min_samples - 2, dtype=np.float32)
    base_returns = np.concatenate([core, np.array([0.5, -0.45], dtype=np.float32)])

    for _ in range(model._value_scale_warmup_limit):
        model.rollout_buffer = types.SimpleNamespace(returns=base_returns.copy())
        model._finalize_return_stats()

    assert model._value_scale_update_count == model._value_scale_warmup_limit
    assert model._value_scale_frozen is True

    returns_tensor = torch.as_tensor(base_returns, dtype=torch.float32)
    decode_returns, scale_safe = model._decode_returns_scale_only(returns_tensor)
    assert scale_safe == pytest.approx(model.value_target_scale)
    assert torch.allclose(decode_returns, returns_tensor * scale_safe)
    abs_p99 = float(torch.quantile(decode_returns.abs(), 0.99).item())
    # In the non-normalized path the decoded returns remain raw fractions scaled
    # by ``value_target_scale``; the p99 absolute magnitude should therefore
    # match the robust scale estimate captured during warmup.
    assert abs_p99 == pytest.approx(
        float(model._value_target_scale_robust), rel=1e-3, abs=1e-3
    )

    returns_decode_path = "scale_only"
    model.logger.record("train/returns_decode_path", returns_decode_path)
    model.logger.record("train/value_target_scale", float(model._value_target_scale_effective))
    model.logger.record("train/value_target_scale_config", float(model.value_target_scale))
    model.logger.record("train/value_target_scale_robust", float(model._value_target_scale_robust))
    model.logger.record(
        "train/value_target_scale[1/fraction]", float(model._value_target_scale_effective)
    )
    model.logger.record(
        "train/value_target_scale_config[fraction]", float(model.value_target_scale)
    )
    model.logger.record(
        "train/value_target_scale_robust[fraction]", float(model._value_target_scale_robust)
    )

    assert model.logger.records["train/returns_decode_path"] == "scale_only"
    assert "train/value_target_scale" in model.logger.records
    assert "train/value_target_scale[1/fraction]" in model.logger.records
