import sys
import types

import numpy as np
import pandas as pd
import pytest


def _install_sb3_stub():
    sb3_contrib = sys.modules.get("sb3_contrib")
    if sb3_contrib is None:
        sb3_contrib = types.ModuleType("sb3_contrib")
        sb3_contrib.__path__ = []  # mark as package
        sys.modules["sb3_contrib"] = sb3_contrib
    if not hasattr(sb3_contrib, "RecurrentPPO"):
        class _RecurrentPPO:  # pragma: no cover - placeholder
            pass

        sb3_contrib.RecurrentPPO = _RecurrentPPO

    common = sys.modules.setdefault(
        "sb3_contrib.common", types.ModuleType("sb3_contrib.common")
    )
    common.__path__ = []
    sb3_contrib.common = common  # type: ignore[attr-defined]

    recurrent = sys.modules.setdefault(
        "sb3_contrib.common.recurrent", types.ModuleType("sb3_contrib.common.recurrent")
    )
    recurrent.__path__ = []
    common.recurrent = recurrent  # type: ignore[attr-defined]

    policies = sys.modules.setdefault(
        "sb3_contrib.common.recurrent.policies",
        types.ModuleType("sb3_contrib.common.recurrent.policies"),
    )

    class _DummyPolicy:  # pragma: no cover - simple placeholder
        pass

    policies.RecurrentActorCriticPolicy = getattr(
        policies, "RecurrentActorCriticPolicy", _DummyPolicy
    )
    recurrent.policies = policies  # type: ignore[attr-defined]

    buffers = sys.modules.setdefault(
        "sb3_contrib.common.recurrent.buffers",
        types.ModuleType("sb3_contrib.common.recurrent.buffers"),
    )

    class _DummyBuffer:  # pragma: no cover - simple placeholder
        pass

    buffers.RecurrentRolloutBuffer = getattr(
        buffers, "RecurrentRolloutBuffer", _DummyBuffer
    )
    recurrent.buffers = buffers  # type: ignore[attr-defined]

    type_aliases = sys.modules.setdefault(
        "sb3_contrib.common.recurrent.type_aliases",
        types.ModuleType("sb3_contrib.common.recurrent.type_aliases"),
    )
    type_aliases.RNNStates = getattr(type_aliases, "RNNStates", object)
    recurrent.type_aliases = type_aliases  # type: ignore[attr-defined]


    sb3 = sys.modules.setdefault("stable_baselines3", types.ModuleType("stable_baselines3"))
    sb3.__path__ = []

    common_sb3 = sys.modules.setdefault(
        "stable_baselines3.common", types.ModuleType("stable_baselines3.common")
    )
    common_sb3.__path__ = []
    sb3.common = common_sb3  # type: ignore[attr-defined]

    policies_mod = types.ModuleType("stable_baselines3.common.policies")

    class _ActorCriticPolicy:  # pragma: no cover - placeholder
        pass

    policies_mod.ActorCriticPolicy = _ActorCriticPolicy
    sys.modules["stable_baselines3.common.policies"] = policies_mod
    common_sb3.policies = policies_mod  # type: ignore[attr-defined]

    vec_env_mod = types.ModuleType("stable_baselines3.common.vec_env")
    vec_env_mod.__path__ = []
    vec_env_mod.__spec__ = types.SimpleNamespace(submodule_search_locations=[])
    sys.modules["stable_baselines3.common.vec_env"] = vec_env_mod

    class _VecEnv:  # pragma: no cover - placeholder
        pass

    class _DummyVecEnv(_VecEnv):  # pragma: no cover - placeholder
        pass

    class _SubprocVecEnv(_VecEnv):  # pragma: no cover - placeholder
        pass

    class _VecMonitor:  # pragma: no cover - placeholder
        pass

    vec_env_mod.VecEnv = _VecEnv
    vec_env_mod.DummyVecEnv = _DummyVecEnv
    vec_env_mod.SubprocVecEnv = _SubprocVecEnv
    vec_env_mod.VecMonitor = _VecMonitor
    common_sb3.vec_env = vec_env_mod  # type: ignore[attr-defined]

    base_vec_env_mod = types.ModuleType("stable_baselines3.common.vec_env.base_vec_env")

    class _VecEnvWrapper(_VecEnv):  # pragma: no cover - placeholder
        def __init__(self, env):
            self.env = env

    base_vec_env_mod.VecEnv = _VecEnv
    base_vec_env_mod.VecEnvWrapper = _VecEnvWrapper
    base_vec_env_mod.CloudpickleWrapper = object
    sys.modules["stable_baselines3.common.vec_env.base_vec_env"] = base_vec_env_mod

    vec_monitor_mod = types.ModuleType("stable_baselines3.common.vec_env.vec_monitor")

    class _VecMonitorWrapper(_VecEnv):  # pragma: no cover - placeholder
        pass

    vec_monitor_mod.VecMonitor = _VecMonitorWrapper
    sys.modules["stable_baselines3.common.vec_env.vec_monitor"] = vec_monitor_mod

    vec_util_mod = types.ModuleType("stable_baselines3.common.vec_env.util")

    def _is_vecenv_wrapped(env, wrapper_class):  # pragma: no cover - placeholder
        return isinstance(env, wrapper_class)

    vec_util_mod.is_vecenv_wrapped = _is_vecenv_wrapped
    sys.modules["stable_baselines3.common.vec_env.util"] = vec_util_mod

    vec_norm_mod = types.ModuleType("stable_baselines3.common.vec_env.vec_normalize")
    sys.modules["stable_baselines3.common.vec_env.vec_normalize"] = vec_norm_mod

    class _VecNormalize(_VecEnv):  # pragma: no cover - placeholder
        training = True

    vec_norm_mod.VecNormalize = _VecNormalize
    vec_norm_mod.unwrap_vec_normalize = lambda env: None

    callbacks_mod = types.ModuleType("stable_baselines3.common.callbacks")
    sys.modules["stable_baselines3.common.callbacks"] = callbacks_mod

    class _BaseCallback:  # pragma: no cover - placeholder
        pass

    class _EvalCallback(_BaseCallback):  # pragma: no cover - placeholder
        pass

    class _CallbackList(_BaseCallback):  # pragma: no cover - placeholder
        pass

    callbacks_mod.BaseCallback = _BaseCallback
    callbacks_mod.EvalCallback = _EvalCallback
    callbacks_mod.CallbackList = _CallbackList

    running_mean_std = types.ModuleType("stable_baselines3.common.running_mean_std")
    sys.modules["stable_baselines3.common.running_mean_std"] = running_mean_std

    class _RunningMeanStd:  # pragma: no cover - placeholder
        def __init__(self, shape=()):
            if isinstance(shape, tuple):
                array_shape = shape
            elif isinstance(shape, list):
                array_shape = tuple(shape)
            else:
                array_shape = (shape,)
            if array_shape == (None,) or array_shape == ((),):
                array_shape = ()
            self.mean = np.zeros(array_shape, dtype=np.float64)
            self.var = np.ones(array_shape, dtype=np.float64)
            self.count = 1.0

        def update(self, batch: np.ndarray) -> None:
            values = np.asarray(batch, dtype=np.float64)
            batch_count = values.shape[0]
            if batch_count == 0:
                return
            batch_mean = values.mean(axis=0)
            batch_var = values.var(axis=0)
            delta = batch_mean - self.mean
            total_count = self.count + batch_count
            new_mean = self.mean + delta * (batch_count / total_count)
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            adjustment = (delta**2) * (self.count * batch_count) / total_count
            new_var = (m_a + m_b + adjustment) / total_count
            self.mean = new_mean
            self.var = new_var
            self.count = total_count

    running_mean_std.RunningMeanStd = _RunningMeanStd

    type_aliases_sb3 = types.ModuleType("stable_baselines3.common.type_aliases")
    sys.modules["stable_baselines3.common.type_aliases"] = type_aliases_sb3
    type_aliases_sb3.GymEnv = object
    type_aliases_sb3.Schedule = object

    utils_mod = types.ModuleType("stable_baselines3.common.utils")

    def _zip_strict(*iterables):  # pragma: no cover - placeholder
        return zip(*iterables)

    utils_mod.zip_strict = _zip_strict
    sys.modules["stable_baselines3.common.utils"] = utils_mod

    base_class_mod = types.ModuleType("stable_baselines3.common.base_class")

    class _BaseAlgorithm:  # pragma: no cover - placeholder
        pass

    base_class_mod.BaseAlgorithm = _BaseAlgorithm
    sys.modules["stable_baselines3.common.base_class"] = base_class_mod

    monitor_mod = types.ModuleType("stable_baselines3.common.monitor")

    class _Monitor:  # pragma: no cover - placeholder
        def __init__(self, env, *_args, **_kwargs):
            self.env = env

    monitor_mod.Monitor = _Monitor
    sys.modules["stable_baselines3.common.monitor"] = monitor_mod

_install_sb3_stub()

import train_model_multi_patch as train_script  # noqa: E402  (import after stub)


class _DummyTrial:
    def __init__(self):
        self.suggested: list[str] = []
        self.number = 0

    def suggest_float(self, name, *_args, **_kwargs):
        self.suggested.append(name)
        # return mid value when bounds provided
        if _args:
            low = _args[0]
            high = _args[1] if len(_args) > 1 else low
            return (low + high) / 2 if high is not None else low
        return 0.0

    def suggest_categorical(self, name, choices, **_kwargs):
        self.suggested.append(name)
        return choices[0]

    def suggest_int(self, name, low, high=None, **_kwargs):
        self.suggested.append(name)
        return low if high is None else low


def test_dataset_split_none_skips_offline_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path):
    seasonality = tmp_path / "seasonality.json"
    seasonality.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("mode: train\n", encoding="utf-8")

    called = False

    def fake_resolve(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("resolve_split_bundle should not be called when dataset-split is none")

    def stop_after_config(_path):
        raise RuntimeError("stop after config load")

    monkeypatch.setattr(train_script, "resolve_split_bundle", fake_resolve)
    monkeypatch.setattr(train_script, "load_config", stop_after_config)

    argv = [
        "train_model_multi_patch.py",
        "--config",
        str(config_path),
        "--dataset-split",
        "none",
        "--liquidity-seasonality",
        str(seasonality),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(RuntimeError, match="stop after config load"):
        train_script.main()

    assert called is False


def test_config_params_override_optuna(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cfg = types.SimpleNamespace(
        model=types.SimpleNamespace(
            params={
                "learning_rate": 3.0e-4,
                "gamma": 0.99,
                "gae_lambda": 0.97,
                "clip_range": 0.15,
                "ent_coef": 0.0001,
                "vf_coef": 1.8,
                "max_grad_norm": 0.3,
                "n_steps": 1024,
                "batch_size": 256,
            }
        ),
        algo=types.SimpleNamespace(
            actions={"long_only": True},
            action_wrapper=types.SimpleNamespace(bins_vol=4),
        ),
        execution=types.SimpleNamespace(mode="bar"),
    )

    trial = _DummyTrial()

    class _StopWatchdogVecEnv:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("stop after params")

    monkeypatch.setattr(train_script, "WatchdogVecEnv", _StopWatchdogVecEnv)

    df = pd.DataFrame({"price": [1.0], "ts_ms": [0]})

    with pytest.raises(RuntimeError, match="stop after params"):
        train_script.objective(
            trial,
            cfg,
            total_timesteps=1024,
            train_data_by_token={"BTCUSDT": df},
            train_obs_by_token={},
            val_data_by_token={"BTCUSDT": df},
            val_obs_by_token={},
            test_data_by_token={},
            test_obs_by_token={},
            norm_stats={},
            sim_config={},
            timing_env_kwargs={},
            env_runtime_overrides={},
            leak_guard_kwargs={},
            trials_dir=tmp_path,
            tensorboard_log_dir=None,
            n_envs_override=1,
        )

    overridden_keys = {
        "learning_rate",
        "gamma",
        "gae_lambda",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "n_steps",
        "batch_size",
    }

    assert overridden_keys.isdisjoint(trial.suggested)


def test_invalid_batch_size_config_raises(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cfg = types.SimpleNamespace(
        model=types.SimpleNamespace(
            params={
                "learning_rate": 3.0e-4,
                "gamma": 0.99,
                "gae_lambda": 0.97,
                "clip_range": 0.15,
                "ent_coef": 0.0001,
                "vf_coef": 1.8,
                "max_grad_norm": 0.3,
                "n_steps": 1024,
                "batch_size": 257,
            }
        ),
        algo=types.SimpleNamespace(
            actions={"long_only": True},
            action_wrapper=types.SimpleNamespace(bins_vol=4),
        ),
        execution=types.SimpleNamespace(mode="bar"),
    )

    trial = _DummyTrial()

    class _FailWatchdogVecEnv:
        def __init__(self, *_args, **_kwargs):  # pragma: no cover - should never be called
            raise AssertionError("WatchdogVecEnv should not be constructed for invalid batch_size")

    monkeypatch.setattr(train_script, "WatchdogVecEnv", _FailWatchdogVecEnv)

    df = pd.DataFrame({"price": [1.0], "ts_ms": [0]})

    with pytest.raises(ValueError, match="batch_size.*divide"):
        train_script.objective(
            trial,
            cfg,
            total_timesteps=1024,
            train_data_by_token={"BTCUSDT": df},
            train_obs_by_token={},
            val_data_by_token={"BTCUSDT": df},
            val_obs_by_token={},
            test_data_by_token={},
            test_obs_by_token={},
            norm_stats={},
            sim_config={},
            timing_env_kwargs={},
            env_runtime_overrides={},
            leak_guard_kwargs={},
            trials_dir=tmp_path,
            tensorboard_log_dir=None,
            n_envs_override=1,
        )


def test_scheduler_disabled_uses_constant_lr(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cfg = types.SimpleNamespace(
        model=types.SimpleNamespace(
            params={
                "learning_rate": 3.0e-4,
                "gamma": 0.99,
                "gae_lambda": 0.97,
                "clip_range": 0.15,
                "ent_coef": 0.0001,
                "vf_coef": 1.8,
                "max_grad_norm": 0.3,
                "n_steps": 32,
                "batch_size": 16,
                "n_epochs": 1,
                "v_range_ema_alpha": 0.005,
                "cql_alpha": 0.0,
                "cql_beta": 5.0,
                "cvar_alpha": 0.05,
                "cvar_weight": 0.0,
                "cvar_cap": 0.5,
                "num_atoms": 5,
                "v_min": -1.0,
                "v_max": 1.0,
                "hidden_dim": 32,
                "trade_frequency_penalty": 0.0,
                "turnover_penalty_coef": 0.0,
            }
        ),
        algo=types.SimpleNamespace(
            actions={"long_only": True},
            action_wrapper=types.SimpleNamespace(bins_vol=4),
        ),
        risk=types.SimpleNamespace(
            cvar=types.SimpleNamespace(limit=-0.02, winsor_pct=0.1, ema_beta=0.9)
        ),
        optimization={"scheduler": {"enabled": False}},
        execution=types.SimpleNamespace(mode="bar"),
    )

    trial = _DummyTrial()

    class _StubTradingEnv:
        def __init__(self, *_args, **_kwargs):
            self.action_space = None

    class _StubWatchdogVecEnv:
        def __init__(self, env_fns):
            self.env_fns = env_fns

    class _StubVecMonitor:
        def __init__(self, env):
            self.env = env

    class _StubDummyVecEnv:
        def __init__(self, env_fns):
            self.env_fns = env_fns

    constructed_vecnorm: list[object] = []

    class _StubVecNormalize:
        def __init__(self, env, **_kwargs):
            self.env = env
            self.training = True
            self.norm_reward = _kwargs.get("norm_reward")
            self.clip_reward = _kwargs.get("clip_reward")
            constructed_vecnorm.append(self)

        def save(self, _path):
            return None

        @classmethod
        def load(cls, _path, env):
            inst = cls(env, norm_obs=False, norm_reward=False, clip_reward=None, gamma=1.0)
            inst.training = False
            inst.norm_reward = False
            inst.clip_reward = None
            return inst

    captured_policy_kwargs: dict[str, object] = {}
    captured_algo_kwargs: dict[str, object] = {}

    class _StubAlgo:
        def __init__(self, *_, policy_kwargs=None, **kwargs):
            captured_policy_kwargs.clear()
            if policy_kwargs:
                captured_policy_kwargs.update(policy_kwargs)
            captured_algo_kwargs.clear()
            captured_algo_kwargs.update(kwargs)
            raise RuntimeError("stop before training")

    def _fail_one_cycle_lr(*_args, **_kwargs):
        raise AssertionError("scheduler should be disabled")

    monkeypatch.setattr(train_script, "TradingEnv", _StubTradingEnv)
    monkeypatch.setattr(train_script, "_wrap_action_space_if_needed", lambda env, **_: env)
    monkeypatch.setattr(train_script, "WatchdogVecEnv", _StubWatchdogVecEnv)
    monkeypatch.setattr(train_script, "VecMonitor", _StubVecMonitor)
    monkeypatch.setattr(train_script, "DummyVecEnv", _StubDummyVecEnv)
    monkeypatch.setattr(train_script, "VecNormalize", _StubVecNormalize)
    monkeypatch.setattr(train_script, "save_sidecar_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_script, "check_model_compat", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_get_distributional_ppo", lambda: _StubAlgo)
    monkeypatch.setattr(train_script, "OneCycleLR", _fail_one_cycle_lr)

    df = pd.DataFrame({"price": [1.0], "ts_ms": [0]})

    with pytest.raises(RuntimeError, match="stop before training"):
        train_script.objective(
            trial,
            cfg,
            total_timesteps=32,
            train_data_by_token={"BTCUSDT": df},
            train_obs_by_token={},
            val_data_by_token={"BTCUSDT": df},
            val_obs_by_token={},
            test_data_by_token={},
            test_obs_by_token={},
            norm_stats={},
            sim_config={},
            timing_env_kwargs={},
            env_runtime_overrides={},
            leak_guard_kwargs={},
            trials_dir=tmp_path,
            tensorboard_log_dir=None,
            n_envs_override=1,
        )

    assert constructed_vecnorm, "VecNormalize should have been constructed"
    assert constructed_vecnorm[0].norm_reward is False
    assert captured_algo_kwargs.get("cvar_limit") == pytest.approx(
        cfg.risk.cvar.limit
    )
    assert captured_algo_kwargs.get("cvar_winsor_pct") == pytest.approx(0.1)
    assert captured_algo_kwargs.get("gae_lambda") == pytest.approx(0.97)

    assert "optimizer_scheduler_fn" not in captured_policy_kwargs


def test_n_envs_override_priority(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cfg = types.SimpleNamespace(
        model=types.SimpleNamespace(
            params={
                "learning_rate": 3.0e-4,
                "gamma": 0.99,
                "gae_lambda": 0.97,
                "clip_range": 0.15,
                "ent_coef": 0.0001,
                "vf_coef": 1.8,
                "max_grad_norm": 0.3,
                "n_steps": 32,
                "batch_size": 16,
                "n_epochs": 1,
                "n_envs": 5,
            }
        ),
        algo=types.SimpleNamespace(
            actions={"long_only": True},
            action_wrapper=types.SimpleNamespace(bins_vol=4),
        ),
        execution=types.SimpleNamespace(mode="bar"),
    )

    trial = _DummyTrial()

    observed_env_count: dict[str, int] = {}

    class _RecordingWatchdogVecEnv:
        def __init__(self, env_fns):
            observed_env_count["count"] = len(env_fns)
            raise RuntimeError("stop for n_envs")

    monkeypatch.setattr(train_script, "WatchdogVecEnv", _RecordingWatchdogVecEnv)

    df = pd.DataFrame({"price": [1.0], "ts_ms": [0]})

    with pytest.raises(RuntimeError, match="stop for n_envs"):
        train_script.objective(
            trial,
            cfg,
            total_timesteps=32,
            train_data_by_token={"BTCUSDT": df},
            train_obs_by_token={},
            val_data_by_token={"BTCUSDT": df},
            val_obs_by_token={},
            test_data_by_token={},
            test_obs_by_token={},
            norm_stats={},
            sim_config={},
            timing_env_kwargs={},
            env_runtime_overrides={},
            leak_guard_kwargs={},
            trials_dir=tmp_path,
            tensorboard_log_dir=None,
            n_envs_override=2,
        )

    assert observed_env_count["count"] == 2
