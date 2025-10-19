import sys
import types

import numpy as np
import torch

if "sb3_contrib" not in sys.modules:  # pragma: no cover - тестовый шим
    sb3_contrib = types.ModuleType("sb3_contrib")
    sb3_contrib.__path__ = []
    sb3_contrib.RecurrentPPO = object
    sys.modules["sb3_contrib"] = sb3_contrib

    sb3_common = types.ModuleType("sb3_contrib.common")
    sb3_common.__path__ = []
    sb3_contrib.common = sb3_common  # type: ignore[attr-defined]
    sys.modules["sb3_contrib.common"] = sb3_common

    sb3_recurrent = types.ModuleType("sb3_contrib.common.recurrent")
    sb3_recurrent.__path__ = []
    sb3_common.recurrent = sb3_recurrent  # type: ignore[attr-defined]
    sys.modules["sb3_contrib.common.recurrent"] = sb3_recurrent

    policies_mod = types.ModuleType("sb3_contrib.common.recurrent.policies")
    policies_mod.__path__ = []
    policies_mod.RecurrentActorCriticPolicy = object
    sys.modules["sb3_contrib.common.recurrent.policies"] = policies_mod
    sb3_recurrent.policies = policies_mod  # type: ignore[attr-defined]

    buffers_mod = types.ModuleType("sb3_contrib.common.recurrent.buffers")
    buffers_mod.__path__ = []
    buffers_mod.RecurrentRolloutBuffer = object
    sys.modules["sb3_contrib.common.recurrent.buffers"] = buffers_mod
    sb3_recurrent.buffers = buffers_mod  # type: ignore[attr-defined]

    type_aliases_mod = types.ModuleType("sb3_contrib.common.recurrent.type_aliases")
    type_aliases_mod.__path__ = []
    type_aliases_mod.RNNStates = object
    sys.modules["sb3_contrib.common.recurrent.type_aliases"] = type_aliases_mod
    sb3_recurrent.type_aliases = type_aliases_mod  # type: ignore[attr-defined]

if "stable_baselines3" not in sys.modules:  # pragma: no cover - тестовый шим
    sb3_module = types.ModuleType("stable_baselines3")
    sb3_module.__path__ = []
    sys.modules["stable_baselines3"] = sb3_module

    sb3_common = types.ModuleType("stable_baselines3.common")
    sb3_common.__path__ = []
    sb3_module.common = sb3_common  # type: ignore[attr-defined]
    sys.modules["stable_baselines3.common"] = sb3_common

    callbacks_mod = types.ModuleType("stable_baselines3.common.callbacks")

    class _BaseCallback:  # pragma: no cover - простая заглушка
        pass

    class _CallbackList(_BaseCallback):  # pragma: no cover - простая заглушка
        pass

    class _EvalCallback(_BaseCallback):  # pragma: no cover - простая заглушка
        pass

    callbacks_mod.BaseCallback = _BaseCallback
    callbacks_mod.CallbackList = _CallbackList
    callbacks_mod.EvalCallback = _EvalCallback
    sys.modules["stable_baselines3.common.callbacks"] = callbacks_mod
    sb3_common.callbacks = callbacks_mod  # type: ignore[attr-defined]

    vec_env_mod = types.ModuleType("stable_baselines3.common.vec_env")

    class _VecEnv:  # pragma: no cover - простая заглушка
        pass

    vec_env_mod.VecEnv = _VecEnv
    sys.modules["stable_baselines3.common.vec_env"] = vec_env_mod
    sb3_common.vec_env = vec_env_mod  # type: ignore[attr-defined]

    vec_norm_mod = types.ModuleType("stable_baselines3.common.vec_env.vec_normalize")

    class _VecNormalize(_VecEnv):  # pragma: no cover - простая заглушка
        pass

    vec_norm_mod.VecNormalize = _VecNormalize
    vec_norm_mod.unwrap_vec_normalize = lambda env: None
    sys.modules["stable_baselines3.common.vec_env.vec_normalize"] = vec_norm_mod

    type_aliases_mod = types.ModuleType("stable_baselines3.common.type_aliases")
    type_aliases_mod.GymEnv = object
    sys.modules["stable_baselines3.common.type_aliases"] = type_aliases_mod

    rms_mod = types.ModuleType("stable_baselines3.common.running_mean_std")

    class _RunningMeanStd:  # pragma: no cover - простая заглушка
        def __init__(self, shape=()):
            if isinstance(shape, tuple):
                dims = shape or (1,)
            else:
                dims = (shape,)
            self.mean = np.zeros(dims, dtype=np.float64)
            self.var = np.ones(dims, dtype=np.float64)
            self.count = 1.0

    rms_mod.RunningMeanStd = _RunningMeanStd
    sys.modules["stable_baselines3.common.running_mean_std"] = rms_mod

    save_util_mod = types.ModuleType("stable_baselines3.common.save_util")

    def _load_from_zip_file(*_args, **_kwargs):  # pragma: no cover - простая заглушка
        raise RuntimeError("zip load stub")

    save_util_mod.load_from_zip_file = _load_from_zip_file
    sys.modules["stable_baselines3.common.save_util"] = save_util_mod

from distributional_ppo import _compute_returns_with_time_limits


class _DummyRolloutBuffer:
    def __init__(self) -> None:
        self.rewards = np.array([[1.0]], dtype=np.float32)
        self.values = np.array([[0.5]], dtype=np.float32)
        self.episode_starts = np.array([[False]])
        self.advantages = np.zeros_like(self.rewards)
        self.returns = np.zeros_like(self.rewards)


def test_time_limit_bootstrap_uses_terminal_value() -> None:  # FIX-TEST
    buffer = _DummyRolloutBuffer()
    last_values = torch.zeros(1, dtype=torch.float32)
    dones = np.array([True])
    time_limit_mask = np.array([[True]])
    time_limit_bootstrap = np.array([[2.0]], dtype=np.float32)

    _compute_returns_with_time_limits(
        rollout_buffer=buffer,
        last_values=last_values,
        dones=dones,
        gamma=0.99,
        gae_lambda=0.95,
        time_limit_mask=time_limit_mask,
        time_limit_bootstrap=time_limit_bootstrap,
    )

    expected_advantage = 1.0 + 0.99 * 2.0 - 0.5
    expected_return = expected_advantage + 0.5

    assert np.isclose(buffer.advantages[0, 0], expected_advantage)
    assert np.isclose(buffer.returns[0, 0], expected_return)
