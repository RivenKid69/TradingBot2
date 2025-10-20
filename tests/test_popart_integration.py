import copy
import sys
import types
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

import pytest

import test_distributional_ppo_raw_outliers  # noqa: F401  # ensure RL stubs are registered

import distributional_ppo as distributional_ppo_module
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

from distributional_ppo import DistributionalPPO
from train_model_multi_patch import (
    _build_popart_holdout_loader,
    _ensure_model_popart_holdout_loader,
)


class _CaptureLogger:
    def __init__(self) -> None:
        self.records: dict[str, float] = {}

    def record(self, key: str, value, **_: object) -> None:  # pragma: no cover - float cast in tests
        self.records[key] = value


def test_popart_holdout_loader_returns_none_even_when_enabled(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING")
    cfg = {"enabled": True, "replay_path": "artifacts/popart_holdout.npz"}

    loader = _build_popart_holdout_loader(cfg)

    assert loader is None
    assert any("PopArt holdout loader requested" in rec.message for rec in caplog.records)


def test_ensure_model_popart_holdout_loader_is_noop(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING")

    class _AlgoStub:
        def __init__(self) -> None:
            self.calls: list[Any] = []
            self.logger = types.SimpleNamespace()

        def _initialise_popart_controller(self, cfg: Any) -> None:  # pragma: no cover - should not run
            self.calls.append(cfg)

    algo = _AlgoStub()
    cfg = {"enabled": True}

    _ensure_model_popart_holdout_loader(algo, None, cfg)

    assert not hasattr(algo, "_popart_holdout_loader")
    assert algo.calls == []
    assert any("PopArt controller configuration" in rec.message for rec in caplog.records)


def test_distributionalppo_initialises_with_popart_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    class _PolicyStub:
        uses_quantile_value_head = False
        quantile_huber_kappa = 1.0

        def named_parameters(self):  # pragma: no cover - empty iterable
            return []

    def _fake_super_init(self, *args: Any, **kwargs: Any) -> None:
        logger = getattr(self, "logger", _CaptureLogger())
        self.logger = logger
        self._logger = logger
        self.policy = _PolicyStub()
        self.device = torch.device("cpu")
        self.n_steps = 1
        self.n_envs = 1
        self.gae_lambda = 0.97
        self.n_epochs = 1
        self.lr_schedule = lambda _: 0.001
        self.normalize_returns = True
        self._value_scale_updates_enabled = True
        self.ent_coef = 0.01

    monkeypatch.setattr(
        distributional_ppo_module.RecurrentPPO,
        "__init__",
        _fake_super_init,
        raising=False,
    )
    monkeypatch.setattr(DistributionalPPO, "_rebuild_scheduler_if_needed", lambda self: None)
    monkeypatch.setattr(DistributionalPPO, "_ensure_score_action_space", lambda self: None)
    monkeypatch.setattr(
        DistributionalPPO,
        "_configure_loss_head_weights",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        DistributionalPPO,
        "_configure_gradient_accumulation",
        lambda self, **kwargs: None,
    )

    def _fake_setup_model(self) -> None:
        pending_cfg: dict[str, Any] = getattr(self, "_popart_cfg_pending", {}) or {}
        if pending_cfg or not getattr(self, "_popart_disabled_logged", False):
            self._initialise_popart_controller(pending_cfg)
        self._popart_cfg_pending = {}

    monkeypatch.setattr(DistributionalPPO, "_setup_model", _fake_setup_model)

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.logger = _CaptureLogger()
    algo._logger = algo.logger

    cfg = {
        "enabled": True,
        "mode": "live",
        "replay_path": "artifacts/popart_holdout.npz",
        "replay_seed": 11,
        "replay_batch_size": 64,
    }

    DistributionalPPO.__init__(
        algo,
        policy=_PolicyStub(),
        env=object(),
        value_scale_controller=cfg,
        value_scale_max_rel_step=0.1,
    )

    assert algo.logger.records.get("config/popart/enabled") == pytest.approx(0.0)
    assert algo.logger.records.get("config/popart/requested_enabled") == pytest.approx(1.0)
    assert algo.logger.records.get("config/popart/replay_path") == ""
    assert algo.logger.records.get("config/popart/replay_seed") == pytest.approx(0.0)
    assert algo.logger.records.get("config/popart/replay_batch_size") == pytest.approx(0.0)
    assert getattr(algo, "_popart_controller", None) is None
    assert getattr(algo, "_popart_cfg_serialized", None) is None


def test_popart_save_load_retains_disabled_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved_payload: dict[str, Any] = {}

    class _PolicyStub:
        uses_quantile_value_head = False
        quantile_huber_kappa = 1.0

        def named_parameters(self):  # pragma: no cover - empty iterable
            return []

    def _fake_super_init(self, *args: Any, **kwargs: Any) -> None:
        logger = getattr(self, "logger", _CaptureLogger())
        self.logger = logger
        self._logger = logger
        self.policy = _PolicyStub()
        self.policy_class = _PolicyStub
        self.device = torch.device("cpu")
        self.observation_space = types.SimpleNamespace()
        self.action_space = types.SimpleNamespace()
        self.n_steps = 1
        self.n_envs = 1
        self.gamma = 0.99
        self.gae_lambda = 0.97
        self.n_epochs = 1
        self.lr_schedule = lambda _progress: 0.001
        self.normalize_returns = True
        self._value_scale_updates_enabled = True
        self.ent_coef = 0.01
        self.policy_kwargs: dict[str, Any] = {}
        self.verbose = 0

    monkeypatch.setattr(
        distributional_ppo_module.RecurrentPPO,
        "__init__",
        _fake_super_init,
        raising=False,
    )

    monkeypatch.setattr(DistributionalPPO, "_rebuild_scheduler_if_needed", lambda self: None)
    monkeypatch.setattr(DistributionalPPO, "_ensure_score_action_space", lambda self: None)
    monkeypatch.setattr(
        DistributionalPPO,
        "_configure_loss_head_weights",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        DistributionalPPO,
        "_configure_gradient_accumulation",
        lambda self, **kwargs: None,
    )

    def _fake_setup_model(self) -> None:
        pending_cfg: dict[str, Any] = getattr(self, "_popart_cfg_pending", {}) or {}
        if pending_cfg or not getattr(self, "_popart_disabled_logged", False):
            self._initialise_popart_controller(pending_cfg)
        self._popart_cfg_pending = {}

    monkeypatch.setattr(DistributionalPPO, "_setup_model", _fake_setup_model)

    def _fake_save(self, path: Path, *args: Any, **kwargs: Any) -> None:
        data = {
            "policy_class": self.policy_class,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy_kwargs": dict(getattr(self, "policy_kwargs", {})),
            "verbose": getattr(self, "verbose", 0),
            "n_envs": getattr(self, "n_envs", 1),
        }
        saved_payload["data"] = data
        saved_payload["params"] = {}
        saved_payload["pytorch_variables"] = None
        Path(path).write_text("stub")

    monkeypatch.setattr(
        distributional_ppo_module.RecurrentPPO,
        "save",
        _fake_save,
        raising=False,
    )

    def _fake_load_from_zip_file(
        path: Path,
        *,
        device: Any = None,
        custom_objects: Optional[dict[str, Any]] = None,
        print_system_info: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any], None]:
        assert saved_payload, "Model data was not saved before load invocation"
        return (
            copy.deepcopy(saved_payload["data"]),
            copy.deepcopy(saved_payload["params"]),
            saved_payload["pytorch_variables"],
        )

    monkeypatch.setattr(
        distributional_ppo_module,
        "load_from_zip_file",
        _fake_load_from_zip_file,
    )

    def _fake_super_load(
        cls,
        path: Path,
        *,
        env: Any = None,
        device: Any = "auto",
        custom_objects: Optional[dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs: Any,
    ) -> DistributionalPPO:
        data, params, pytorch_variables = _fake_load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )
        model = cls(
            policy=data.get("policy_class"),
            env=env,
            device=device,
            _init_setup_model=False,
            **kwargs,
        )
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()
        return model

    monkeypatch.setattr(
        distributional_ppo_module.RecurrentPPO,
        "load",
        classmethod(_fake_super_load),
        raising=False,
    )

    cfg = {
        "enabled": True,
        "mode": "live",
        "ema_beta": 0.9,
        "min_samples": 8,
        "warmup_updates": 0,
        "max_rel_step": 0.5,
        "ev_floor": 0.2,
        "ret_std_band": (0.1, 2.0),
        "gate_patience": 1,
    }

    algo = DistributionalPPO.__new__(DistributionalPPO)
    logger = _CaptureLogger()
    algo.logger = logger
    algo._logger = logger

    DistributionalPPO.__init__(
        algo,
        policy=_PolicyStub(),
        env=object(),
        value_scale_controller=cfg,
        value_scale_max_rel_step=0.5,
    )

    assert logger.records.get("config/popart/enabled") == pytest.approx(0.0)
    assert logger.records.get("config/popart/requested_enabled") == pytest.approx(1.0)

    save_path = tmp_path / "popart_model.zip"
    algo.save(save_path)

    loaded = DistributionalPPO.load(save_path, value_scale_max_rel_step=0.5)

    assert isinstance(loaded, DistributionalPPO)
    loaded_logger = getattr(loaded, "logger", None)
    assert isinstance(loaded_logger, _CaptureLogger)
    assert loaded_logger.records.get("config/popart/enabled") == pytest.approx(0.0)
    assert loaded_logger.records.get("config/popart/requested_enabled") == pytest.approx(0.0)
