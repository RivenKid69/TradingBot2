import sys
import types

import pytest
import torch


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
                self.mean = 0.0
                self.var = 1.0

        running_mean_std.RunningMeanStd = _RunningMeanStd
        sys.modules["stable_baselines3.common.running_mean_std"] = running_mean_std


_install_rl_stubs()

from distributional_ppo import DistributionalPPO


def test_value_target_outlier_fractions_detects_raw_outliers() -> None:
    values = torch.tensor([-15.0, -11.0, 0.0, 9.0, 25.0], dtype=torch.float32)
    below, above = DistributionalPPO._value_target_outlier_fractions(values, -10.0, 10.0)

    assert below > 0.0
    assert above > 0.0
    assert below == pytest.approx(2.0 / 5.0)
    assert above == pytest.approx(1.0 / 5.0)
