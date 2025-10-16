import sys
import types
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

import test_distributional_ppo_raw_outliers  # noqa: F401  # ensure RL stubs are registered

if "stable_baselines3.common.vec_env.base_vec_env" not in sys.modules:
    base_vec_env = types.ModuleType("stable_baselines3.common.vec_env.base_vec_env")

    class _VecEnvStub:  # pragma: no cover - lightweight placeholder
        pass

    class _CloudpickleWrapperStub:  # pragma: no cover - lightweight placeholder
        def __init__(self, var: object) -> None:
            self.var = var

    base_vec_env.VecEnv = _VecEnvStub
    base_vec_env.CloudpickleWrapper = _CloudpickleWrapperStub
    class _VecEnvWrapperStub(_VecEnvStub):  # pragma: no cover - placeholder wrapper
        def __init__(self, env: object) -> None:
            self.env = env

    base_vec_env.VecEnvWrapper = _VecEnvWrapperStub
    sys.modules["stable_baselines3.common.vec_env.base_vec_env"] = base_vec_env

if "gymnasium.spaces.utils" not in sys.modules:
    spaces_utils = types.ModuleType("gymnasium.spaces.utils")

    def _flatten(_space, x):  # pragma: no cover - placeholder passthrough
        return np.asarray(x)

    def _flatten_space(space):  # pragma: no cover - placeholder passthrough
        return space

    def _unflatten(_space, x):  # pragma: no cover - placeholder passthrough
        return x

    spaces_utils.flatten = _flatten
    spaces_utils.flatten_space = _flatten_space
    spaces_utils.unflatten = _unflatten
    sys.modules["gymnasium.spaces.utils"] = spaces_utils

gymnasium_module = sys.modules.get("gymnasium")
if gymnasium_module is not None and not hasattr(gymnasium_module, "Env"):
    class _EnvBase:  # pragma: no cover - placeholder base env
        pass

    gymnasium_module.Env = _EnvBase
    spaces_module = getattr(gymnasium_module, "spaces", None)
    if spaces_module is not None and not hasattr(spaces_module, "Space"):
        class _SpaceBase:  # pragma: no cover - placeholder space
            pass

        spaces_module.Space = _SpaceBase

if "stable_baselines3.common.policies" not in sys.modules:
    policies_module = types.ModuleType("stable_baselines3.common.policies")

    class _ActorCriticPolicyStub:  # pragma: no cover - placeholder policy
        pass

    policies_module.ActorCriticPolicy = _ActorCriticPolicyStub
    sys.modules["stable_baselines3.common.policies"] = policies_module

type_aliases_module = sys.modules.get("stable_baselines3.common.type_aliases")
if type_aliases_module is None:
    type_aliases_module = types.ModuleType("stable_baselines3.common.type_aliases")
    sys.modules["stable_baselines3.common.type_aliases"] = type_aliases_module
if not hasattr(type_aliases_module, "Schedule"):
    type_aliases_module.Schedule = Callable[[float], float]  # type: ignore[attr-defined]

if "stable_baselines3.common.utils" not in sys.modules:
    utils_module = types.ModuleType("stable_baselines3.common.utils")

    def _zip_strict(*iterables):  # pragma: no cover - simplified placeholder
        return zip(*iterables)

    utils_module.zip_strict = _zip_strict
    sys.modules["stable_baselines3.common.utils"] = utils_module

vec_env_module = sys.modules.get("stable_baselines3.common.vec_env")
if vec_env_module is None:
    vec_env_module = types.ModuleType("stable_baselines3.common.vec_env")
    sys.modules["stable_baselines3.common.vec_env"] = vec_env_module
if not hasattr(vec_env_module, "VecEnv"):
    class _VecEnvBase:  # pragma: no cover - placeholder vec env
        pass

    vec_env_module.VecEnv = _VecEnvBase
if not hasattr(vec_env_module, "DummyVecEnv"):
    class _DummyVecEnvStub(vec_env_module.VecEnv):  # type: ignore[attr-defined]
        pass

    vec_env_module.DummyVecEnv = _DummyVecEnvStub
if not hasattr(vec_env_module, "SubprocVecEnv"):
    class _SubprocVecEnvStub(vec_env_module.VecEnv):  # type: ignore[attr-defined]
        pass

    vec_env_module.SubprocVecEnv = _SubprocVecEnvStub
if not hasattr(vec_env_module, "VecMonitor"):
    class _VecMonitorStub(vec_env_module.VecEnv):  # type: ignore[attr-defined]
        def __init__(self, env):
            self.env = env

    vec_env_module.VecMonitor = _VecMonitorStub

if "stable_baselines3.common.vec_env.vec_monitor" not in sys.modules:
    vec_monitor_module = types.ModuleType("stable_baselines3.common.vec_env.vec_monitor")
    vec_monitor_module.VecMonitor = vec_env_module.VecMonitor
    sys.modules["stable_baselines3.common.vec_env.vec_monitor"] = vec_monitor_module

if "stable_baselines3.common.vec_env.util" not in sys.modules:
    vec_util_module = types.ModuleType("stable_baselines3.common.vec_env.util")

    def _is_vecenv_wrapped(_env, _wrapper):  # pragma: no cover - placeholder
        return False

    vec_util_module.is_vecenv_wrapped = _is_vecenv_wrapped
    sys.modules["stable_baselines3.common.vec_env.util"] = vec_util_module

if "stable_baselines3.common.base_class" not in sys.modules:
    base_class_module = types.ModuleType("stable_baselines3.common.base_class")

    class _BaseAlgorithmStub:  # pragma: no cover - placeholder algorithm
        pass

    base_class_module.BaseAlgorithm = _BaseAlgorithmStub
    sys.modules["stable_baselines3.common.base_class"] = base_class_module

if "stable_baselines3.common.monitor" not in sys.modules:
    monitor_module = types.ModuleType("stable_baselines3.common.monitor")

    class _MonitorStub:  # pragma: no cover - placeholder monitor
        def __init__(self, env, *args, **kwargs) -> None:
            self.env = env

    monitor_module.Monitor = _MonitorStub
    sys.modules["stable_baselines3.common.monitor"] = monitor_module

from distributional_ppo import DistributionalPPO, PopArtHoldoutEvaluation
from train_model_multi_patch import (
    _build_popart_holdout_loader,
    _ensure_model_popart_holdout_loader,
)


class _CaptureLogger:
    def __init__(self) -> None:
        self.records: dict[str, float] = {}

    def record(self, key: str, value, **_: object) -> None:  # pragma: no cover - float cast in tests
        self.records[key] = value


def test_popart_controller_initialises_with_existing_holdout(tmp_path: Path) -> None:
    holdout_path = tmp_path / "popart_holdout.npz"
    obs = np.zeros((32, 3), dtype=np.float32)
    returns = np.linspace(-0.01, 0.01, 32, dtype=np.float32).reshape(-1, 1)
    episode_starts = np.zeros((32,), dtype=np.float32)
    np.savez(holdout_path, obs=obs, returns=returns, episode_starts=episode_starts)

    cfg = {
        "enabled": True,
        "mode": "shadow",
        "ema_beta": 0.99,
        "min_samples": 16,
        "warmup_updates": 0,
        "max_rel_step": 1.0,
        "ev_floor": 0.2,
        "ret_std_band": [0.0, 2.0],
        "gate_patience": 1,
        "replay_path": str(holdout_path),
        "replay_seed": 7,
        "replay_batch_size": 16,
    }

    loader = _build_popart_holdout_loader(cfg)
    assert loader is not None

    batch = loader()
    assert batch is not None
    assert getattr(loader, "fallback_generated", False) is False

    algo = DistributionalPPO.__new__(DistributionalPPO)
    logger = _CaptureLogger()
    algo.logger = logger
    algo._popart_holdout_loader = loader

    algo._initialise_popart_controller(cfg)

    controller = algo._popart_controller
    assert controller is not None

    def _fake_holdout_eval(
        self,
        *,
        model,
        holdout,
        old_mean: float,
        old_std: float,
        new_mean: float,
        new_std: float,
    ) -> PopArtHoldoutEvaluation:
        sample_count = holdout.returns_raw.shape[0]
        zeros = torch.zeros(sample_count, 1)
        return PopArtHoldoutEvaluation(
            baseline_raw=zeros,
            candidate_raw=zeros,
            target_raw=zeros,
            mask=None,
            ev_before=0.6,
            ev_after=0.7,
            clip_fraction_before=0.0,
            clip_fraction_after=0.0,
        )

    controller._evaluate_holdout = types.MethodType(_fake_holdout_eval, controller)

    metrics = controller.evaluate_shadow(
        model=algo,
        returns_raw=torch.ones(16, dtype=torch.float32),
        ret_mean=0.0,
        ret_std=1.0,
    )

    assert metrics is not None
    assert metrics.blocked_reason != "no_holdout"
    assert "shadow_popart/pass" in logger.records
    assert getattr(loader, "fallback_generated", False) is False


def test_ensure_model_popart_holdout_loader_assigns_and_reinitialises() -> None:
    class _LoggerStub:
        def __init__(self) -> None:
            self.records: list[tuple[str, float]] = []

        def record(self, key: str, value: float, **_: object) -> None:  # pragma: no cover - simple stub
            self.records.append((key, float(value)))

    class _LoaderStub:
        def __init__(self) -> None:
            self.materialise_calls = 0

        def ensure_materialized(self) -> None:
            self.materialise_calls += 1

        def __call__(self) -> Optional[Any]:  # pragma: no cover - callable interface for parity with real loader
            return None

    class _AlgoStub:
        def __init__(self) -> None:
            self.calls: list[Any] = []
            self.logger = _LoggerStub()

        def _initialise_popart_controller(self, cfg: Any) -> None:
            self.calls.append(cfg)
            self.logger.record("popart/reinitialised", float(len(self.calls)))

    loader = _LoaderStub()
    algo = _AlgoStub()
    cfg = {"enabled": True, "mode": "shadow"}

    _ensure_model_popart_holdout_loader(algo, loader, cfg)

    assert getattr(algo, "_popart_holdout_loader") is loader
    assert algo.calls == [cfg]
    assert loader.materialise_calls == 1
    assert algo.logger.records == [("popart/reinitialised", 1.0)]

    _ensure_model_popart_holdout_loader(algo, loader, cfg)

    assert algo.calls == [cfg, cfg]
    assert loader.materialise_calls == 2
    assert algo.logger.records[-1] == ("popart/reinitialised", 2.0)
