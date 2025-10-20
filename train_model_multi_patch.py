# Имя файла: train_model_multi_patch.py
# ИЗМЕНЕНИЯ (ФАЗА 3 - CVaR):
# 1. Добавлен гиперпараметр `cvar_alpha` в пространство поиска Optuna.
# 2. `cvar_alpha` передается в конструктор агента DistributionalPPO.
# ИЗМЕНЕНИЯ (ФАЗА 6 - Устранение утечки данных в HPO):
# 1. Статистики нормализации из тренировочной среды (env_tr) теперь
#    сохраняются в файл ВНАЧАЛЕ испытания.
# 2. Валидационная среда (env_va) для прунинга теперь создается
#    путем ЗАГРУЗКИ этих сохраненных статистик.
# 3. Это устраняет утечку данных, когда env_va вычисляла статистики
#    на валидационных данных, и обеспечивает корректную оценку модели.
# ИЗМЕНЕНИЯ (ФАЗА 7 - Исправление HPO для CVaR):
# 1. Гиперпараметр `cvar_weight` добавлен в пространство поиска Optuna.
# 2. `cvar_weight` теперь корректно передается в конструктор модели
#    DistributionalPPO, что делает HPO полноценным.
# ИЗМЕНЕНИЯ (ОПТИМИЗАЦИЯ ПРОИЗВОДИТЕЛЬНОСТИ):
# 1. Добавлены настройки PyTorch для ускорения вычислений на GPU.
# 2. Добавлена компиляция модели (`torch.compile`) для значительного ускорения.
# 3. Цикл оценки вынесен в быстрый Cython-модуль.

from __future__ import annotations

import dataclasses
import errno
import importlib
import os
import glob
import json
import pandas as pd
import numpy as np
from datetime import datetime
import shutil
import math
import argparse
import yaml
import sys
import hashlib
import random
import threading
import types
from copy import deepcopy
from functools import lru_cache
from collections.abc import Mapping
import logging
from typing import Any, Callable, Dict, MutableMapping, Optional, Sequence  # FIX
from features_pipeline import FeaturePipeline
from pathlib import Path

from core_config import (
    load_config,
    ExecutionProfile,
    load_timing_profiles,
    resolve_execution_timing,
    TrainConfig,
)


logger = logging.getLogger(__name__)


def _install_torch_intrinsic_stub() -> None:
    """Install lightweight stubs for torch's optional quantization packages.

    Certain environments (notably constrained WSL setups) fail to allocate
    enough memory when Python attempts to scan ``torch.nn.intrinsic`` during
    the import of PyTorch.  The training code does not rely on PyTorch's
    quantization features, so we can provide empty placeholder modules to
    satisfy the import without touching the real package hierarchy.
    """

    package_name = "torch.nn.intrinsic"
    if package_name in sys.modules:
        return

    intrinsic_module = types.ModuleType(package_name)
    intrinsic_module.__package__ = "torch.nn"
    intrinsic_module.__path__ = []  # type: ignore[attr-defined]

    for suffix in ("modules", "qat", "quantized"):
        full_name = f"{package_name}.{suffix}"
        if full_name in sys.modules:
            child_module = sys.modules[full_name]
        else:
            child_module = types.ModuleType(full_name)
            child_module.__package__ = package_name
            child_module.__path__ = []  # type: ignore[attr-defined]
            sys.modules[full_name] = child_module
        setattr(intrinsic_module, suffix, child_module)

    sys.modules[package_name] = intrinsic_module


def _import_torch_with_intrinsic_stub() -> Any:
    try:
        import torch  # type: ignore[no-redef]
        return torch
    except OSError as exc:
        if exc.errno == errno.ENOMEM and "torch/nn/intrinsic" in str(exc):
            logger.warning(
                "PyTorch import failed with ENOMEM while loading quantization "
                "modules; installing lightweight stubs and retrying."
            )
            _install_torch_intrinsic_stub()
            try:
                return importlib.import_module("torch")
            except OSError as retry_exc:
                raise RuntimeError(
                    "PyTorch import keeps failing because the environment "
                    "cannot allocate memory for optional quantization "
                    "modules. Consider installing a CPU-only PyTorch build "
                    "or freeing system memory."
                ) from retry_exc
        raise


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Достаёт поле из dict/pydantic/dataclass/кастомного конфига."""

    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    if hasattr(cfg, key):
        return getattr(cfg, key)
    get = getattr(cfg, "get", None)
    if callable(get):
        try:
            return get(key, default)
        except TypeError:
            try:
                return get(key)
            except Exception:
                pass
    for dump in ("model_dump", "dict"):
        fn = getattr(cfg, dump, None)
        if callable(fn):
            try:
                data = fn()
            except Exception:
                continue
            if isinstance(data, Mapping):
                return data.get(key, default)
            try:
                return dict(data).get(key, default)  # type: ignore[arg-type]
            except Exception:
                continue
    if dataclasses.is_dataclass(cfg):
        try:
            return dataclasses.asdict(cfg).get(key, default)
        except Exception:
            pass
    return default


class _PopArtHoldoutLoaderWrapper:
    """Lazily loads or materialises PopArt holdout batches."""

    def __init__(
        self,
        *,
        path: Path,
        batch_size: int,
        seed: int,
        min_samples: int,
        batch_cls: Any,
    ) -> None:
        self._path = path
        self._batch_size = max(int(batch_size), 1)
        self._seed = int(seed)
        self._min_samples = max(int(min_samples), 1)
        self._batch_cls = batch_cls
        self._logger = logging.getLogger(__name__)
        self._cache: dict[str, Optional[Any]] = {}
        self._env: Optional["VecEnv"] = None
        self._lock = threading.RLock()
        self._fallback_attempted = False
        self.fallback_generated = False
        self._target_samples = max(self._min_samples, self._batch_size, 2048)

    # ------------------------------------------------------------------
    # Public API used by the training pipeline
    # ------------------------------------------------------------------
    def attach_env(self, env: "VecEnv") -> None:
        with self._lock:
            self._env = env

    def ensure_materialized(self) -> None:
        if self._path.exists():
            return
        self._generate_if_missing()

    def __call__(self) -> Optional[Any]:
        with self._lock:
            if "batch" in self._cache:
                return self._cache["batch"]

        self._generate_if_missing()

        with self._lock:
            if not self._path.exists():
                self._logger.warning("PopArt replay file not found at %s", self._path)
                self._cache["batch"] = None
                return None
            try:
                payload = np.load(str(self._path), allow_pickle=False)
            except Exception as exc:
                self._logger.warning(
                    "Failed to load PopArt replay %s: %s", self._path, exc
                )
                self._cache["batch"] = None
                return None

        obs = payload.get("obs")
        if obs is None:
            obs = payload.get("observations")
        if obs is None:
            obs = payload.get("states")
        targets = payload.get("returns")
        if targets is None:
            targets = payload.get("returns_raw")
        if targets is None:
            targets = payload.get("target")
        if obs is None or targets is None:
            self._logger.warning(
                "PopArt replay %s missing 'obs'/'returns' arrays", self._path
            )
            with self._lock:
                self._cache["batch"] = None
            return None

        obs_np = np.asarray(obs, dtype=np.float32)
        returns_np = np.asarray(targets, dtype=np.float32)
        if returns_np.ndim == 1:
            returns_np = returns_np.reshape(-1, 1)
        if obs_np.shape[0] == 0 or returns_np.shape[0] == 0:
            with self._lock:
                self._cache["batch"] = None
            return None
        count = int(min(obs_np.shape[0], returns_np.shape[0]))
        rng = np.random.default_rng(self._seed)
        if self._batch_size >= count:
            indices = np.arange(count)
        else:
            indices = rng.choice(count, size=self._batch_size, replace=False)

        obs_tensor = torch.as_tensor(obs_np[indices], dtype=torch.float32)
        returns_tensor = torch.as_tensor(returns_np[indices], dtype=torch.float32)

        starts = payload.get("episode_starts")
        if starts is None:
            starts = payload.get("episode_start")
        if starts is None:
            starts = payload.get("dones")
        if starts is None:
            starts_np = np.zeros((count,), dtype=np.float32)
        else:
            starts_np = np.asarray(starts, dtype=np.float32).reshape(count)
        starts_tensor = torch.as_tensor(starts_np[indices], dtype=torch.float32)

        mask_arr = payload.get("mask")
        if mask_arr is None:
            mask_arr = payload.get("weights")
        if mask_arr is None:
            mask_tensor: Optional[torch.Tensor] = None
        else:
            mask_tensor = torch.as_tensor(
                np.asarray(mask_arr, dtype=np.float32).reshape(-1)[indices],
                dtype=torch.float32,
            )

        batch = self._batch_cls(
            observations=obs_tensor,
            returns_raw=returns_tensor,
            episode_starts=starts_tensor,
            lstm_states=None,
            mask=mask_tensor,
        )
        with self._lock:
            self._cache["batch"] = batch
        return batch

    # ------------------------------------------------------------------
    # Fallback generation helpers
    # ------------------------------------------------------------------
    def _generate_if_missing(self) -> None:
        with self._lock:
            if self._path.exists() or self.fallback_generated:
                return
            env = self._env
            if env is None:
                if not self._fallback_attempted:
                    self._logger.warning(
                        "PopArt replay %s is missing and no environment is available for fallback generation",
                        self._path,
                    )
                    self._fallback_attempted = True
                return
            self._fallback_attempted = True

        try:
            data = self._collect_holdout(env)
        except Exception as exc:
            self._logger.warning(
                "Failed to auto-generate PopArt replay %s: %s", self._path, exc
            )
            return

        obs_arr = data["obs"]
        sample_count = int(obs_arr.shape[0])

        self._logger.warning(
            "PopArt replay %s not found, generated fallback batch with %d samples",
            self._path,
            sample_count,
        )

        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(self._path),
            obs=data["obs"],
            returns=data["returns"],
            episode_starts=data["episode_starts"],
            mask=data.get("mask"),
        )
        with self._lock:
            self._cache.pop("batch", None)
            self.fallback_generated = True

    def _collect_holdout(self, env: "VecEnv") -> dict[str, np.ndarray]:
        import gym

        rng = np.random.default_rng(self._seed)
        target_samples = int(self._target_samples)

        try:
            env.seed(self._seed)
        except Exception:
            pass

        try:
            obs = env.reset(seed=self._seed)
        except TypeError:
            obs = env.reset()

        num_envs = int(getattr(env, "num_envs", 1))
        try:
            action_space = env.action_space
        except AttributeError:
            action_space = None

        previous_training_flag = getattr(env, "training", None)
        if previous_training_flag is not None:
            env.training = False

        obs_samples: list[np.ndarray] = []
        ret_samples: list[np.ndarray] = []
        start_samples: list[np.ndarray] = []
        episode_starts = np.ones((num_envs,), dtype=np.float32)

        try:
            while len(obs_samples) * num_envs < target_samples:
                obs_np = np.asarray(obs, dtype=np.float32)
                if obs_np.shape[:1] == (num_envs,):
                    obs_np = obs_np.reshape((num_envs,) + obs_np.shape[1:])
                else:
                    obs_np = obs_np.reshape((num_envs,) + obs_np.shape)
                obs_samples.append(obs_np)
                start_samples.append(episode_starts.astype(np.float32))

                actions = self._sample_actions(action_space, num_envs, rng, gym)
                step_result = env.step(actions)
                if isinstance(step_result, tuple) and len(step_result) >= 4:
                    next_obs, rewards, dones, _ = step_result[:4]
                else:  # pragma: no cover - safety for unexpected VecEnv API
                    next_obs = step_result[0]
                    rewards = step_result[1]
                    dones = step_result[2]

                rewards_np = np.asarray(rewards, dtype=np.float32).reshape(num_envs, -1)
                if rewards_np.shape[1] == 0:
                    rewards_np = np.zeros((num_envs, 1), dtype=np.float32)
                ret_samples.append(rewards_np[:, :1])

                episode_starts = np.asarray(dones, dtype=np.float32).reshape(num_envs)
                obs = next_obs

            obs_arr = np.concatenate(obs_samples, axis=0).astype(np.float32)
            returns_arr = np.concatenate(ret_samples, axis=0).astype(np.float32)
            starts_arr = np.concatenate(start_samples, axis=0).astype(np.float32)

            if obs_arr.shape[0] > target_samples:
                obs_arr = obs_arr[:target_samples]
                returns_arr = returns_arr[:target_samples]
                starts_arr = starts_arr[:target_samples]

            mask_arr = np.ones((obs_arr.shape[0], 1), dtype=np.float32)

            return {
                "obs": obs_arr,
                "returns": returns_arr.reshape(-1, 1),
                "episode_starts": starts_arr.reshape(-1),
                "mask": mask_arr,
            }
        finally:
            try:
                env.reset()
            except Exception:
                pass
            if previous_training_flag is not None:
                env.training = previous_training_flag

    @staticmethod
    def _sample_actions(
        action_space: Any,
        num_envs: int,
        rng: np.random.Generator,
        gym_module: Any,
    ) -> np.ndarray:
        if action_space is None:
            return np.zeros((num_envs, 1), dtype=np.float32)

        if gym_module is not None and isinstance(action_space, gym_module.spaces.Box):
            dtype = action_space.dtype if action_space.dtype is not None else np.float32
            zeros = np.zeros((num_envs,) + action_space.shape, dtype=dtype)
            actions = zeros.copy()
            low = np.asarray(action_space.low, dtype=np.float32)
            high = np.asarray(action_space.high, dtype=np.float32)
            low = np.where(np.isfinite(low), low, -1.0)
            high = np.where(np.isfinite(high), high, 1.0)
            for idx in range(num_envs):
                if rng.random() < 0.25:
                    sample = rng.uniform(low=low, high=high).astype(dtype)
                    actions[idx] = sample
            return actions

        if gym_module is not None and isinstance(action_space, gym_module.spaces.Discrete):
            base = int(getattr(action_space, "start", 0))
            actions = np.full((num_envs,), base, dtype=np.int64)
            for idx in range(num_envs):
                if rng.random() < 0.25:
                    actions[idx] = int(rng.integers(action_space.n))
            return actions

        samples = [action_space.sample() for _ in range(num_envs)]
        return np.asarray(samples)


def _build_popart_holdout_loader(
    controller_cfg: Any,
) -> Optional[Callable[[], Optional["PopArtHoldoutBatch"]]]:
    try:
        from distributional_ppo import PopArtHoldoutBatch  # Local import to avoid cycles
    except Exception:  # pragma: no cover - safeguard if training module unavailable
        PopArtHoldoutBatch = None

    if PopArtHoldoutBatch is None:
        return None

    enabled = bool(_cfg_get(controller_cfg, "enabled", False))
    if not enabled:
        return None

    replay_path = _cfg_get(controller_cfg, "replay_path", "artifacts/popart_holdout.npz")
    if replay_path is None:
        return None

    try:
        path = Path(str(replay_path))
    except Exception:
        return None

    batch_size_raw = _cfg_get(controller_cfg, "replay_batch_size", 2048)
    try:
        batch_size = max(int(batch_size_raw), 1)
    except Exception:
        batch_size = 2048
    seed_raw = _cfg_get(controller_cfg, "replay_seed", 17)
    try:
        replay_seed = int(seed_raw)
    except Exception:
        replay_seed = 17
    min_samples_raw = _cfg_get(controller_cfg, "min_samples", 4096)
    try:
        min_samples = int(min_samples_raw)
    except Exception:
        min_samples = 4096

    loader = _PopArtHoldoutLoaderWrapper(
        path=path,
        batch_size=batch_size,
        seed=replay_seed,
        min_samples=min_samples,
        batch_cls=PopArtHoldoutBatch,
    )
    return loader


def _ensure_model_popart_holdout_loader(
    model: Any,
    loader: Optional[Callable[[], Optional["PopArtHoldoutBatch"]]],
    controller_cfg: Any,
) -> None:
    """Ensure the DistributionalPPO instance uses the prepared PopArt loader."""

    enabled = bool(_cfg_get(controller_cfg, "enabled", False))
    if not enabled:
        return

    existing_loader = getattr(model, "_popart_holdout_loader", None)
    if loader is None:
        loader = existing_loader or _build_popart_holdout_loader(controller_cfg)
        if loader is None:
            return

    def _materialize(target_loader: Callable[[], Optional["PopArtHoldoutBatch"]]) -> None:
        ensure_fn = getattr(target_loader, "ensure_materialized", None)
        if not callable(ensure_fn):
            bound = getattr(target_loader, "__self__", None)
            ensure_fn = getattr(bound, "ensure_materialized", None)
        if callable(ensure_fn):
            try:
                ensure_fn()
            except Exception as exc:
                logger.warning("PopArt: holdout materialization failed: %s", exc)

    if existing_loader is not loader:
        try:
            setattr(model, "_popart_holdout_loader", loader)
        except Exception:
            logger.warning("Failed to assign PopArt holdout loader to model", exc_info=True)
            return

    _materialize(loader)

    initialise = getattr(model, "_initialise_popart_controller", None)
    if not callable(initialise):
        return

    try:
        initialise(controller_cfg)
    except Exception:
        logger.warning(
            "Failed to re-initialise PopArt controller with assigned holdout loader",
            exc_info=True,
        )

try:
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv, VecEnv
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import VecNormalize
except ImportError:  # pragma: no cover - test-time fallback stubs
    class VecEnv:  # type: ignore[dead-code]
        pass

    class DummyVecEnv(VecEnv):  # pragma: no cover - placeholder
        pass

    class SubprocVecEnv(VecEnv):  # pragma: no cover - placeholder
        pass

    class VecMonitor(VecEnv):  # pragma: no cover - placeholder
        pass

    class VecNormalize(VecEnv):  # pragma: no cover - placeholder
        training = True

        def __init__(self, *args, **kwargs):
            pass

    class BaseCallback:  # pragma: no cover - placeholder
        pass

    class EvalCallback(BaseCallback):  # pragma: no cover - placeholder
        pass
torch = _import_torch_with_intrinsic_stub()
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.exceptions import TrialPruned
from torch.optim.lr_scheduler import OneCycleLR
import multiprocessing as mp
from leakguard import LeakGuard, LeakConfig


logger = logging.getLogger(__name__)
_ACTION_WRAPPER_CONFIG_LOGGED = False

_SECONDS_PER_YEAR = 365 * 24 * 60 * 60
_DEFAULT_ANNUALIZATION_SQRT = float(np.sqrt(365 * 24))
# BAR volume discretization is fixed to four buckets [0.0, 0.2, 0.6, 1.0].
# Keep the expected count in a dedicated constant so validation logic can
# reference a single source of truth.
EXPECTED_VOLUME_BINS = 4


def _flatten_candidates(values: Any):
    if isinstance(values, (list, tuple, set, np.ndarray)):
        for item in values:
            yield from _flatten_candidates(item)
    elif values is not None:
        yield values


def _coerce_positive_seconds(value: Any, *, assumes_ms: bool = False) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    if assumes_ms:
        numeric /= 1000.0
    if numeric <= 0:
        return None
    return numeric


def _resolve_bar_seconds(source: Any) -> float | None:
    """Attempt to extract bar interval (seconds) from an env or wrapper."""

    visited: set[int] = set()

    def _resolve(obj: Any) -> float | None:
        if obj is None:
            return None
        ident = id(obj)
        if ident in visited:
            return None
        visited.add(ident)

        getter = getattr(obj, "get_bar_interval_seconds", None)
        if callable(getter):
            try:
                sec = _coerce_positive_seconds(getter())
            except Exception:
                sec = None
            if sec is not None:
                return sec

        for attr, assumes_ms in (("bar_interval_seconds", False), ("bar_seconds", False), ("bar_interval_ms", True)):
            if hasattr(obj, attr):
                sec = _coerce_positive_seconds(getattr(obj, attr), assumes_ms=assumes_ms)
                if sec is not None:
                    return sec

        env_method = getattr(obj, "env_method", None)
        if callable(env_method):
            try:
                values = env_method("get_bar_interval_seconds")
            except Exception:
                values = []
            for value in _flatten_candidates(values):
                sec = _coerce_positive_seconds(value)
                if sec is not None:
                    return sec

        get_attr = getattr(obj, "get_attr", None)
        if callable(get_attr):
            try:
                values = get_attr("bar_interval_ms")
            except Exception:
                values = []
            for value in _flatten_candidates(values):
                sec = _coerce_positive_seconds(value, assumes_ms=True)
                if sec is not None:
                    return sec

        for attr in ("venv", "env", "_vec_env", "envs"):
            if not hasattr(obj, attr):
                continue
            child = getattr(obj, attr)
            if isinstance(child, (list, tuple)):
                for item in child:
                    sec = _resolve(item)
                    if sec is not None:
                        return sec
            else:
                sec = _resolve(child)
                if sec is not None:
                    return sec
        return None

    return _resolve(source)


def _annualization_sqrt_from_env(source: Any) -> tuple[float, float | None]:
    bar_seconds = _resolve_bar_seconds(source)
    if bar_seconds is None:
        return _DEFAULT_ANNUALIZATION_SQRT, None
    return float(np.sqrt(_SECONDS_PER_YEAR / float(bar_seconds))), bar_seconds


def _log_annualization(label: str, ann_sqrt: float, bar_seconds: float | None) -> None:
    if bar_seconds is None:
        logger.info("%s: using fallback annualization sqrt factor %.6f (bar interval unknown)", label, ann_sqrt)
    else:
        steps_per_year = _SECONDS_PER_YEAR / float(bar_seconds)
        logger.info(
            "%s: annualization sqrt factor %.6f (bar_seconds=%.6f, steps_per_year=%.2f)",
            label,
            ann_sqrt,
            bar_seconds,
            steps_per_year,
        )


def _value_changed(previous: float | None, current: float | None) -> bool:
    if previous is None or current is None:
        return previous is not None or current is not None
    return not math.isclose(previous, current, rel_tol=1e-9, abs_tol=1e-9)



def _freeze_vecnormalize(vec_env: VecNormalize) -> VecNormalize:
    """Ensure VecNormalize stops updating statistics during evaluation."""

    vec_env.training = False
    for attr in ("_update", "update"):
        if hasattr(vec_env, attr):
            try:
                setattr(vec_env, attr, False)
            except Exception:
                pass
    return vec_env
class AdversarialCallback(BaseCallback):
    """
    Проводит стресс-тесты в специальных рыночных режимах и СОХРАНЯЕТ
    результаты (Sortino Ratio) для каждого режима.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int,
        regimes: list,
        regime_duration: int,
        regime_config_path: str | None = None,
        liquidity_seasonality_path: str | None = None,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.regimes = regimes
        self.regime_duration = regime_duration
        # Словарь для хранения метрик: {'regime_name': sortino_score}
        self.regime_metrics = {}

        if regime_config_path:
            os.environ["MARKET_REGIMES_JSON"] = regime_config_path
        self._liq_seasonality_path = liquidity_seasonality_path
        self._sortino_ann_sqrt: float = _DEFAULT_ANNUALIZATION_SQRT
        self._sortino_bar_seconds: float | None = None

    def _resolve_sortino_factor(self) -> float:
        ann_sqrt, bar_seconds = _annualization_sqrt_from_env(self.eval_env)
        if _value_changed(self._sortino_bar_seconds, bar_seconds) or _value_changed(self._sortino_ann_sqrt, ann_sqrt):
            _log_annualization("AdversarialCallback", ann_sqrt, bar_seconds)
            self._sortino_bar_seconds = bar_seconds
            self._sortino_ann_sqrt = ann_sqrt
        return self._sortino_ann_sqrt

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            print("\n--- Starting Adversarial Regime Stress Tests ---")

            ann_sqrt = self._resolve_sortino_factor()

            for regime in self.regimes:
                print(f"Testing regime: {regime}...")
                # Устанавливаем режим в среде
                self.eval_env.env_method("set_market_regime", regime=regime, duration=self.regime_duration)
                
                # Запускаем оценку в этом режиме
                _rewards, equity_curves = evaluate_policy_custom_cython(
                    self.model,
                    self.eval_env,
                    num_episodes=1 # Один длинный эпизод для каждого режима
                )

                # Считаем Sortino и сохраняем
                all_returns = [pd.Series(c).pct_change().dropna().to_numpy() for c in equity_curves if len(c) > 1]
                flat_returns = np.concatenate(all_returns) if all_returns else np.array([0.0])
                score = sortino_ratio(flat_returns, annualization_sqrt=ann_sqrt)
                self.regime_metrics[regime] = score

                print(f"Regime '{regime}' | Sortino: {score:.4f}")

            # Сбрасываем среду в нормальный режим
            self.eval_env.env_method("set_market_regime", regime='normal', duration=0)
            print("--- Adversarial Tests Finished ---\n")

            # Validate distributions against reference after stress tests
            try:
                dist_metrics, _ = compare_regime_distributions(
                    env=self.eval_env,
                    reference_path="configs/reference_regime_distributions.json",
                    n_steps=self.regime_duration,
                    tolerance=0.2,
                    raise_on_fail=False,
                )
                for r_name, vals in dist_metrics.items():
                    for m_name, val in vals.items():
                        if self.logger is not None:
                            self.logger.record(f"regime_dist/{r_name}/{m_name}", val)
            except Exception as exc:
                print(f"Distribution validation failed: {exc}")
        return True

    def get_regime_metrics(self) -> dict:
        """Возвращает словарь с результатами тестов."""
        return self.regime_metrics
from shared_memory_vec_env import SharedMemoryVecEnv


@lru_cache(maxsize=1)
def _get_distributional_ppo():
    from distributional_ppo import DistributionalPPO

    return DistributionalPPO

# --- ИЗМЕНЕНИЕ: Включаем оптимизации PyTorch ---
# Если GPU и версия CUDA >= 11, включаем высокую точность
if torch.cuda.is_available() and int(torch.version.cuda.split(".")[0]) >= 11:
    torch.set_float32_matmul_precision("high")
# Позволяет cuDNN находить лучшие алгоритмы для текущей конфигурации
torch.backends.cudnn.benchmark = True


from trading_patchnew import TradingEnv, DecisionTiming
from gymnasium import spaces
from wrappers.action_space import LongOnlyActionWrapper, ScoreActionWrapper
from custom_policy_patch1 import CustomActorCriticPolicy
from fetch_all_data_patch import load_all_data
# --- ИЗМЕНЕНИЕ: Импортируем быструю Cython-функцию оценки ---
from evaluate_policy_custom_cython import evaluate_policy_custom_cython
from scripts.validate_regime_distributions import compare_regime_distributions
from data_validation import DataValidator
from utils.model_io import save_sidecar_metadata, check_model_compat
from watchdog_vec_env import WatchdogVecEnv
from scripts.offline_utils import load_offline_payload, resolve_split_bundle

# --- helper to compute SHA256 of liquidity seasonality file ---
def _file_sha256(path: str | None) -> str | None:
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except FileNotFoundError:
        return None


def _wrap_action_space_if_needed(
    env,
    *,
    bins_vol: int | None = None,
    action_overrides: dict[str, object] | None = None,
    long_only: bool = False,
):
    """Ensure the environment exposes a score-based Box(1) action space."""

    global _ACTION_WRAPPER_CONFIG_LOGGED
    if not _ACTION_WRAPPER_CONFIG_LOGGED:
        max_asset_weight = None
        if isinstance(action_overrides, Mapping):
            max_asset_weight = action_overrides.get("max_asset_weight")
        if bins_vol is not None and bins_vol != EXPECTED_VOLUME_BINS:
            logger.warning(
                "Unexpected bins_vol value received: expected %s, got %s",
                EXPECTED_VOLUME_BINS,
                bins_vol,
            )
        logger.info(
            "[action wrapper] long_only=%s, max_asset_weight=%s",
            long_only,
            max_asset_weight,
        )
        _ACTION_WRAPPER_CONFIG_LOGGED = True

    try:
        setattr(env, "_signal_long_only", bool(long_only))
        if bins_vol is not None:
            setattr(env, "bins_vol", int(bins_vol))
    except Exception:
        pass

    wrapped_env = env
    try:
        if long_only:
            wrapped_env = LongOnlyActionWrapper(wrapped_env)
        action_space = getattr(wrapped_env, "action_space", None)
        if not isinstance(action_space, spaces.Box) or tuple(action_space.shape) != (1,):
            wrapped_env = ScoreActionWrapper(wrapped_env)
    except Exception:
        return wrapped_env
    final_space = getattr(wrapped_env, "action_space", None)
    if not isinstance(final_space, spaces.Box) or tuple(final_space.shape) != (1,):
        raise RuntimeError("Score action space wrapper failed to enforce Box(1)")
    return wrapped_env


def _assign_nested(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Assign ``value`` inside ``target`` using ``.``-delimited keys."""

    if not dotted_key:
        return
    parts = dotted_key.split(".")
    cursor: dict[str, Any] = target
    for part in parts[:-1]:
        next_obj = cursor.get(part)
        if not isinstance(next_obj, dict):
            next_obj = {}
            cursor[part] = next_obj
        cursor = next_obj
    cursor[parts[-1]] = value


def _snapshot_model_param_keys(cfg: TrainConfig) -> set[str]:
    keys: set[str] = set()
    try:
        model_cfg = getattr(cfg, "model", None)
        if model_cfg is None:
            return keys
        params_obj = getattr(model_cfg, "params", None)
        if params_obj is None:
            return keys
        if isinstance(params_obj, Mapping):
            keys.update(str(k) for k in params_obj.keys())
            return keys
        if hasattr(params_obj, "__dict__") and isinstance(params_obj.__dict__, Mapping):
            keys.update(str(k) for k in params_obj.__dict__.keys())
        dict_method = getattr(params_obj, "dict", None)
        if callable(dict_method):
            try:
                params_dict = dict_method()
            except TypeError:
                params_dict = None
            if isinstance(params_dict, Mapping):
                keys.update(str(k) for k in params_dict.keys())
    except Exception:
        return keys
    return keys


def _propagate_train_window_alias(
    block: MutableMapping[str, Any], updated_key: str, value: Any
) -> None:
    """Keep ``start_ts``/``train_start_ts`` and ``end_ts``/``train_end_ts`` in sync."""

    if "." in updated_key:
        return

    if updated_key in {"train_start_ts", "start_ts"}:
        block["train_start_ts"] = value
        block["start_ts"] = value
    elif updated_key in {"train_end_ts", "end_ts"}:
        block["train_end_ts"] = value
        block["end_ts"] = value


def _ensure_train_window_aliases(block: MutableMapping[str, Any]) -> None:
    """Normalise legacy aliases for training window bounds."""

    start = block.get("start_ts")
    train_start = block.get("train_start_ts")
    if train_start is None and start is not None:
        block["train_start_ts"] = start
    elif train_start is not None and (start is None or start != train_start):
        block["start_ts"] = train_start

    end = block.get("end_ts")
    train_end = block.get("train_end_ts")
    if train_end is None and end is not None:
        block["train_end_ts"] = end
    elif train_end is not None and (end is None or end != train_end):
        block["end_ts"] = train_end


def _extract_env_runtime_overrides(
    env_block: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], DecisionTiming | None]:
    """Normalise optional environment overrides for runtime construction."""

    runtime_kwargs: dict[str, Any] = {}
    decision_override: DecisionTiming | None = None
    if not isinstance(env_block, Mapping):
        return runtime_kwargs, decision_override

    raw_decision = env_block.get("decision_timing") or env_block.get("decision_mode")
    if isinstance(raw_decision, str):
        key = raw_decision.strip().upper()
        if key in DecisionTiming.__members__:
            decision_override = DecisionTiming[key]

    no_trade_cfg = env_block.get("no_trade") if isinstance(env_block, Mapping) else None
    if isinstance(no_trade_cfg, Mapping):
        if "enabled" in no_trade_cfg:
            runtime_kwargs["no_trade_enabled"] = bool(no_trade_cfg["enabled"])
        if "policy" in no_trade_cfg and no_trade_cfg["policy"] is not None:
            runtime_kwargs["no_trade_policy"] = str(no_trade_cfg["policy"])

    for section in ("session", "liquidity", "spreads", "warmup"):
        payload = env_block.get(section)
        if isinstance(payload, Mapping):
            runtime_kwargs[section] = dict(payload)

    return runtime_kwargs, decision_override


def _coerce_timestamp(value) -> int | None:
    """Normalize timestamp representations to integer seconds."""

    if value is None:
        return None
    if isinstance(value, str):
        txt = value.strip()
        if not txt or txt.lower() == "none":
            return None
        try:
            value = int(txt)
        except ValueError:
            try:
                ts = pd.Timestamp(txt)
            except Exception as exc:  # pragma: no cover - defensive branch
                raise ValueError(f"Unable to parse timestamp string '{value}'.") from exc
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return int(ts.value // 10**9)
    if isinstance(value, pd.Timestamp):
        ts = value
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.value // 10**9)
    if isinstance(value, (np.integer, int)):
        iv = int(value)
        if abs(iv) > 10_000_000_000:  # heuristic: values in ms
            return iv // 1000
        return iv
    if isinstance(value, (np.floating, float)):
        if math.isnan(value):
            return None
        return _coerce_timestamp(int(value))
    raise ValueError(f"Unsupported timestamp value: {value!r}")


def _normalize_interval(item) -> tuple[int | None, int | None]:
    if item is None:
        return (None, None)
    if isinstance(item, dict):
        start = item.get("start_ts")
        if start is None:
            start = item.get("start") or item.get("from")
        end = item.get("end_ts")
        if end is None:
            end = item.get("end") or item.get("to")
    elif isinstance(item, (list, tuple)) and len(item) == 2:
        start, end = item
    else:
        raise TypeError(f"Unsupported interval specification: {item!r}")
    start_ts = _coerce_timestamp(start)
    end_ts = _coerce_timestamp(end)
    if start_ts is not None and end_ts is not None and end_ts < start_ts:
        raise ValueError(f"Invalid interval with start {start_ts} after end {end_ts}")
    return (start_ts, end_ts)


def _extract_offline_split_overrides(
    payload: Mapping[str, Any] | None,
    dataset_key: str,
    *,
    fallback_split: str = "time",
) -> dict[str, list[dict[str, int | None]]]:
    """Extract inline time split overrides from an offline payload."""

    if payload is None:
        return {}

    datasets = payload.get("datasets") if isinstance(payload, Mapping) else None
    dataset_entry = datasets.get(dataset_key) if isinstance(datasets, Mapping) else None

    def _select_split_block(entry: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if not isinstance(entry, Mapping):
            return None
        splits_block = entry.get("splits")
        if isinstance(splits_block, Mapping):
            return splits_block
        # Some payloads embed train/val/test directly at the dataset level.
        direct = {k: entry.get(k) for k in ("train", "val", "test") if entry.get(k) is not None}
        return direct or None

    def _pick_split_entry(block: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if not isinstance(block, Mapping) or not block:
            return None
        if dataset_key in block and isinstance(block[dataset_key], Mapping):
            return block[dataset_key]
        if fallback_split in block and isinstance(block[fallback_split], Mapping):
            return block[fallback_split]
        # As a last resort take the first mapping value.
        for value in block.values():
            if isinstance(value, Mapping):
                return value
        return None

    split_entry = _pick_split_entry(_select_split_block(dataset_entry))
    if split_entry is None:
        # fall back to top-level split registry if provided
        split_entry = _pick_split_entry(
            payload.get("splits") if isinstance(payload, Mapping) else None
        )

    if not isinstance(split_entry, Mapping):
        return {}

    overrides: dict[str, list[dict[str, int | None]]] = {}
    for phase, entries in split_entry.items():
        if phase not in {"train", "val", "test"}:
            continue
        if isinstance(entries, (list, tuple)):
            iterable: Sequence[Any] = entries
        else:
            iterable = (entries,)
        normalized_items: list[dict[str, int | None]] = []
        for item in iterable:
            try:
                start_ts, end_ts = _normalize_interval(item)
            except Exception:
                continue
            normalized_items.append({"start_ts": start_ts, "end_ts": end_ts})
        if normalized_items:
            overrides[phase] = normalized_items
    return overrides


def _load_time_splits(data_cfg) -> tuple[str | None, dict[str, list[tuple[int | None, int | None]]]]:
    """Derive train/val/test time windows from config or an external manifest."""

    splits: dict[str, list[tuple[int | None, int | None]]] = {"train": [], "val": [], "test": []}
    version: str | None = getattr(data_cfg, "split_version", None)

    overrides = getattr(data_cfg, "split_overrides", None)
    if isinstance(overrides, Mapping):
        for phase, entries in overrides.items():
            if phase not in splits:
                splits[phase] = []
            if isinstance(entries, (list, tuple)):
                iterable = entries
            else:
                iterable = [entries]
            splits[phase].extend(_normalize_interval(item) for item in iterable)

    split_path = getattr(data_cfg, "split_path", None)
    if split_path:
        manifest_path = Path(split_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Split manifest not found: {split_path}")
        with open(manifest_path, "r", encoding="utf-8") as fh:
            if manifest_path.suffix.lower() in (".yaml", ".yml"):
                raw = yaml.safe_load(fh) or {}
            else:
                raw = json.load(fh)
        version = raw.get("version") or raw.get("name") or version or manifest_path.stem
        raw_splits = raw.get("splits")
        if raw_splits is None:
            raw_splits = {k: raw.get(k) for k in ("train", "val", "test") if raw.get(k) is not None}
        if raw_splits is None:
            raise ValueError("Split manifest must contain 'splits' or per-phase keys (train/val/test).")
        for phase in ("train", "val", "test"):
            entries = raw_splits.get(phase)
            if entries is None:
                continue
            if isinstance(entries, (list, tuple)):
                iterable = entries
            else:
                iterable = [entries]
            splits[phase].extend(_normalize_interval(item) for item in iterable)

    for phase in ("train", "val", "test"):
        start_attr = getattr(data_cfg, f"{phase}_start_ts", None)
        end_attr = getattr(data_cfg, f"{phase}_end_ts", None)
        if start_attr is not None or end_attr is not None:
            splits[phase].append(_normalize_interval({"start_ts": start_attr, "end_ts": end_attr}))

    if not splits["train"]:
        fallback = _normalize_interval({"start_ts": getattr(data_cfg, "start_ts", None), "end_ts": getattr(data_cfg, "end_ts", None)})
        if fallback != (None, None):
            splits["train"].append(fallback)

    for phase, items in splits.items():
        cleaned = [item for item in items if not (item[0] is None and item[1] is None)]
        cleaned.sort(key=lambda it: it[0] if it[0] is not None else -float("inf"))
        splits[phase] = cleaned

    if not splits["train"]:
        raise ValueError("Training split is empty: provide train_start/train_end or a manifest")

    return version, splits


def _ensure_validation_split_present(
    dfs_with_roles: dict[str, pd.DataFrame],
    intervals: dict[str, list[tuple[int | None, int | None]]],
    timestamp_column: str,
    role_column: str,
) -> None:
    """Abort execution when the validation split yields zero rows."""

    val_rows = 0
    for df in dfs_with_roles.values():
        val_rows += int((df[role_column].astype(str) == "val").sum())
    if val_rows > 0:
        return

    configured = intervals.get("val", [])
    configured_desc = (
        ", ".join(_format_interval(it) for it in configured)
        if configured
        else "(not configured)"
    )
    observed_start, observed_end = _phase_bounds(dfs_with_roles, timestamp_column)
    coverage_desc = f"[{_fmt_ts(observed_start)} .. {_fmt_ts(observed_end)}]"

    msg_lines = [
        "Validation split is empty after applying configured intervals.",
        f"Configured validation intervals: {configured_desc}",
        f"Observed data coverage: {coverage_desc}",
    ]

    if configured:
        overlap_detected = None
        if observed_start is not None and observed_end is not None:
            overlap_detected = False
            for start, end in configured:
                start_cmp = observed_start if start is None else start
                end_cmp = observed_end if end is None else end
                if start_cmp <= observed_end and end_cmp >= observed_start:
                    overlap_detected = True
                    break
        if overlap_detected is False:
            msg_lines.append(
                "Configured validation window does not overlap with available data; "
                "regenerate or refresh the offline dataset."
            )
        else:
            msg_lines.append(
                "Adjust the validation split configuration or refresh the offline dataset "
                "to include the desired range."
            )
    else:
        msg_lines.append(
            "Provide validation split overrides or refresh the offline dataset to include validation data."
        )

    raise SystemExit("\n".join(msg_lines))


def _apply_role_column(
    df: pd.DataFrame,
    intervals: dict[str, list[tuple[int | None, int | None]]],
    timestamp_column: str,
    role_column: str,
) -> tuple[pd.DataFrame, bool]:
    """Annotate ``df`` with walk-forward roles using the provided intervals."""

    if timestamp_column not in df.columns:
        raise KeyError(f"DataFrame is missing timestamp column '{timestamp_column}'")
    ts = pd.to_numeric(df[timestamp_column], errors="coerce")
    roles = pd.Series(np.full(len(df), "none", dtype=object), index=df.index)

    for phase in ("train", "val", "test"):
        masks = []
        for start, end in intervals.get(phase, []):
            cur = pd.Series(True, index=df.index)
            if start is not None:
                cur &= ts >= start
            if end is not None:
                cur &= ts <= end
            masks.append(cur)
        if not masks:
            continue
        phase_mask = masks[0]
        for extra in masks[1:]:
            phase_mask |= extra
        assignable = (roles == "none") & phase_mask
        if assignable.any():
            roles.loc[assignable] = phase

    inferred_test = False
    if not intervals.get("test"):
        leftover = roles == "none"
        if leftover.any():
            roles.loc[leftover] = "test"
            inferred_test = True

    df_out = df.copy()
    df_out[role_column] = roles.values
    return df_out, inferred_test


def _phase_bounds(mapping: dict[str, pd.DataFrame], ts_col: str) -> tuple[int | None, int | None]:
    start: int | None = None
    end: int | None = None
    for df in mapping.values():
        ts = pd.to_numeric(df[ts_col], errors="coerce").dropna()
        if ts.empty:
            continue
        cur_start = int(ts.min())
        cur_end = int(ts.max())
        start = cur_start if start is None or cur_start < start else start
        end = cur_end if end is None or cur_end > end else end
    return start, end


def _fmt_ts(ts: int | None) -> str:
    if ts is None:
        return "None"
    return pd.to_datetime(int(ts), unit="s", utc=True).isoformat()


def _format_interval(interval: tuple[int | None, int | None]) -> str:
    start, end = interval
    return f"[{_fmt_ts(start)} .. {_fmt_ts(end)}]"

# === КОНФИГУРАЦИЯ ИНДИКАТОРОВ (ЕДИНЫЙ ИСТОЧНИК ПРАВДЫ) ===
MA5_WINDOW = 5
MA20_WINDOW = 20
ATR_WINDOW = 14
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MOMENTUM_WINDOW = 10
CCI_WINDOW = 14
BB_WINDOW = 20
OBV_MA_WINDOW = 50


class NanGuardCallback(BaseCallback):
    """
    Проверяет loss и градиенты на NaN/Inf.
    При обнаружении прерывает текущий Optuna-trial.
    """
    def __init__(self, threshold: float = float("inf"), verbose: int = 0):
        super().__init__(verbose)
        self.threshold = threshold       # Можно задать лимит «слишком большой» loss

    def _on_rollout_end(self) -> None:
        # 1) Проверяем loss, если SB3 положил его в локалы
        last_loss = self.locals.get("loss", None)
        if last_loss is not None:
            if (not torch.isfinite(last_loss)) or (torch.abs(last_loss) > self.threshold):
                print("🚨  NaN/Inf обнаружен в loss  —  прерываем trial")
                raise TrialPruned("NaN detected in loss")

        # 2) Проверяем градиенты всех параметров
        for p in self.model.parameters():
            if p.grad is not None and (not torch.isfinite(p.grad).all()):
                print("🚨  NaN/Inf обнаружен в градиентах  —  прерываем trial")
                raise TrialPruned("NaN detected in gradients")

    def _on_step(self) -> bool:
        return True

class SortinoPruningCallback(BaseCallback):
    """
    Кастомный callback для Optuna, который оценивает модель и принимает
    решение о прунинге на основе КОЭФФИЦИЕНТА СОРТИНО, а не среднего вознаграждения.
    """

    def __init__(
        self,
        trial: optuna.Trial,
        eval_env: VecEnv,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self._last_eval_step = 0
        self._ann_sqrt: float = _DEFAULT_ANNUALIZATION_SQRT
        self._bar_seconds: float | None = None

    def _resolve_sortino_factor(self) -> tuple[float, float | None]:
        ann_sqrt, bar_seconds = _annualization_sqrt_from_env(self.eval_env)
        if _value_changed(self._bar_seconds, bar_seconds) or _value_changed(self._ann_sqrt, ann_sqrt):
            _log_annualization("SortinoPruningCallback", ann_sqrt, bar_seconds)
            self._ann_sqrt = ann_sqrt
            self._bar_seconds = bar_seconds
        return self._ann_sqrt, self._bar_seconds

    def _on_step(self) -> bool:
        # Запускаем оценку в заданные интервалы (по числу таймстепов)
        current_step = self.num_timesteps
        if (
            self.eval_freq > 0
            and current_step >= self.eval_freq
            and current_step - self._last_eval_step >= self.eval_freq
        ):
            self._last_eval_step = current_step

            # Используем быструю Cython-функцию для оценки
            rewards, equity_curves = evaluate_policy_custom_cython(
                self.model,
                self.eval_env,
                num_episodes=self.n_eval_episodes
            )

            ann_sqrt, bar_seconds = self._resolve_sortino_factor()

            # Рассчитываем Sortino на основе полученных кривых капитала
            if not equity_curves:
                # Если по какой-то причине оценка не вернула результатов, считаем Sortino равным 0
                current_sortino = 0.0
            else:
                all_returns = [
                    pd.Series(curve).pct_change().dropna().to_numpy()
                    for curve in equity_curves if len(curve) > 1
                ]
                flat_returns = np.concatenate(all_returns) if all_returns else np.array([0.0])
                current_sortino = sortino_ratio(flat_returns, annualization_sqrt=ann_sqrt)

            if self.verbose > 0:
                print(
                    f"Step {current_step}: Pruning check with Sortino Ratio = {current_sortino:.4f}"
                )

            if self.logger is not None:
                if bar_seconds is None:
                    self.logger.record("pruning/bar_seconds", float("nan"))
                    self.logger.record("pruning/sortino_steps_per_year", float("nan"))
                else:
                    steps_per_year = _SECONDS_PER_YEAR / float(bar_seconds)
                    self.logger.record("pruning/bar_seconds", float(bar_seconds))
                    self.logger.record("pruning/sortino_steps_per_year", float(steps_per_year))
                self.logger.record("pruning/sortino_ann_factor", float(ann_sqrt))
                self.logger.record("pruning/sortino_ratio", current_sortino)

            # 1. Сообщаем Optuna о промежуточном результате (теперь это Sortino)
            self.trial.report(current_sortino, current_step)

            # 2. Проверяем, не нужно ли остановить этот trial
            if self.trial.should_prune():
                raise TrialPruned(
                    f"Trial pruned at step {current_step} with Sortino Ratio: {current_sortino:.4f}"
                )

        return True

class ObjectiveScorePruningCallback(BaseCallback):
    """
    Callback для прунинга, использующий полную взвешенную метрику objective_score.
    Работает реже, чем SortinoPruningCallback, так как оценка занимает больше времени.
    """

    def __init__(
        self,
        trial: optuna.Trial,
        eval_env: VecEnv,
        eval_freq: int = 40000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.eval_freq = eval_freq

        # Веса, идентичные финальной оценке
        self.main_weight = 0.5
        self.choppy_weight = 0.3
        self.trend_weight = 0.2
        self.regime_duration = 2_500
        self._last_eval_step = 0
        self._ann_sqrt: float = _DEFAULT_ANNUALIZATION_SQRT
        self._bar_seconds: float | None = None

    def _resolve_sortino_factor(self) -> tuple[float, float | None]:
        ann_sqrt, bar_seconds = _annualization_sqrt_from_env(self.eval_env)
        if _value_changed(self._bar_seconds, bar_seconds) or _value_changed(self._ann_sqrt, ann_sqrt):
            _log_annualization("ObjectiveScorePruningCallback", ann_sqrt, bar_seconds)
            self._bar_seconds = bar_seconds
            self._ann_sqrt = ann_sqrt
        return self._ann_sqrt, self._bar_seconds

    def _on_step(self) -> bool:
        current_step = self.num_timesteps
        if (
            self.eval_freq > 0
            and current_step >= self.eval_freq
            and current_step - self._last_eval_step >= self.eval_freq
        ):
            self._last_eval_step = current_step

            print(
                f"\n--- Step {current_step}: Starting comprehensive pruning check with Objective Score ---"
            )

            regimes_to_evaluate = ['normal', 'choppy_flat', 'strong_trend']
            evaluated_metrics = {}
            ann_sqrt, bar_seconds = self._resolve_sortino_factor()

            try:
                for regime in regimes_to_evaluate:
                    if self.verbose > 0:
                        print(f"Pruning evaluation: testing regime '{regime}'...")

                    # Устанавливаем адверсариальный режим
                    if regime != 'normal':
                        self.eval_env.env_method("set_market_regime", regime=regime, duration=self.regime_duration)

                    # Для прунинга достаточно меньшего числа эпизодов, чем в финале
                    num_episodes = 5  # Всегда использовать 5 эпизодов для более стабильной оценки
                    
                    _rewards, equity_curves = evaluate_policy_custom_cython(
                        self.model, self.eval_env, num_episodes=num_episodes
                    )

                    all_returns = [pd.Series(c).pct_change().dropna().to_numpy() for c in equity_curves if len(c) > 1]
                    flat_returns = np.concatenate(all_returns) if all_returns else np.array([0.0])
                    score = sortino_ratio(flat_returns, annualization_sqrt=ann_sqrt)
                    evaluated_metrics[regime] = score

            finally:
                # КРИТИЧЕСКИ ВАЖНО: всегда сбрасываем среду в нормальный режим,
                # чтобы не влиять на следующий шаг обучения или другие колбэки.
                self.eval_env.env_method("set_market_regime", regime='normal', duration=0)

            # Рассчитываем взвешенную метрику
            main_sortino = evaluated_metrics.get('normal', -1.0)
            choppy_score = evaluated_metrics.get('choppy_flat', -1.0)
            trend_score = evaluated_metrics.get('strong_trend', -1.0)
            
            objective_score = (self.main_weight * main_sortino + 
                               self.choppy_weight * choppy_score + 
                               self.trend_weight * trend_score)

            if self.verbose > 0:
                print(
                    f"Comprehensive pruning check complete. Objective Score: {objective_score:.4f}"
                )
                print(
                    f"Components -> Main: {main_sortino:.4f}, Choppy: {choppy_score:.4f}, Trend: {trend_score:.4f}\n"
                )

            if self.logger is not None:
                self.logger.record("pruning/objective_score", objective_score)
                self.logger.record("pruning/objective_main", main_sortino)
                self.logger.record("pruning/objective_choppy", choppy_score)
                self.logger.record("pruning/objective_trend", trend_score)

            # Сообщаем Optuna о результате и проверяем необходимость прунинга
            self.trial.report(objective_score, current_step)
            if self.trial.should_prune():
                raise TrialPruned(
                    f"Trial pruned at step {current_step} with Objective Score: {objective_score:.4f}"
                )

        return True


class OptimizerLrLoggingCallback(BaseCallback):
    """Логирует фактические значения learning rate оптимизатора и шедулера."""

    def _on_step(self) -> bool:
        if self.logger is None:
            return True

        policy = getattr(self.model, "policy", None)
        if policy is None:
            return True

        algo = getattr(self, "model", None)
        opt_floor = float(getattr(algo, "_optimizer_lr_min", 0.0))
        sch_floor = float(getattr(algo, "_scheduler_min_lr", opt_floor))
        if sch_floor < opt_floor:
            sch_floor = opt_floor

        optimizer = getattr(policy, "optimizer", None)
        opt_lrs: list[float] = []
        if optimizer is not None:
            for group in getattr(optimizer, "param_groups", []):
                lr_value = group.get("lr")
                if lr_value is None:
                    continue
                try:
                    opt_lrs.append(float(lr_value))
                except (TypeError, ValueError):
                    continue

        if opt_lrs:
            self.logger.record("train/optimizer_lr", max(opt_lrs[0], opt_floor))
        else:
            self.logger.record("train/optimizer_lr", None)

        scheduler = getattr(policy, "lr_scheduler", None)
        if scheduler is None:
            scheduler = getattr(policy, "optimizer_scheduler", None)

        if scheduler is not None and hasattr(scheduler, "get_last_lr"):
            try:
                sch_lrs = scheduler.get_last_lr()
            except TypeError:
                sch_lrs = None
            if sch_lrs:
                self.logger.record("train/scheduler_lr", max(float(sch_lrs[0]), sch_floor))
            else:
                self.logger.record("train/scheduler_lr", None)
        else:
            self.logger.record("train/scheduler_lr", None)

        return True


def _resolve_ann_sqrt(annualization_sqrt: float | None) -> float:
    if annualization_sqrt is None:
        return _DEFAULT_ANNUALIZATION_SQRT
    try:
        value = float(annualization_sqrt)
    except (TypeError, ValueError):
        return _DEFAULT_ANNUALIZATION_SQRT
    if not math.isfinite(value) or value <= 0:
        return _DEFAULT_ANNUALIZATION_SQRT
    return value


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    *,
    annualization_sqrt: float | None = None,
) -> float:
    ann = _resolve_ann_sqrt(annualization_sqrt)
    std = np.std(returns)
    return np.mean(returns - risk_free_rate) / (std + 1e-9) * ann


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    *,
    annualization_sqrt: float | None = None,
) -> float:
    ann = _resolve_ann_sqrt(annualization_sqrt)
    downside = returns[returns < risk_free_rate] - risk_free_rate
    downside_count = downside.size
    if downside_count == 0:
        # Если нет убытков, используем стандартное отклонение (как в Шарпе).
        # Это более адекватно оценивает риск, чем возврат константы.
        std = np.std(returns)
        # Предотвращаем деление на ноль, если все доходности одинаковы.
        if std < 1e-9:
            return 0.0
        return np.mean(returns - risk_free_rate) / std * ann

    if downside_count < 20:
        std = np.std(returns)
        if std < 1e-9:
            return 0.0
        return np.mean(returns - risk_free_rate) / std * ann

    downside_std = np.sqrt(np.mean(downside**2))
    # Если выборка практически стационарна, возвращаем 0, чтобы не завышать метрику.
    if downside_std < 1e-9:
        return 0.0

    safe_downside_std = max(downside_std, 1e-9)
    # Эта проверка становится избыточной, если downside_std используется только один раз,
    # но оставим для безопасности.
    return np.mean(returns - risk_free_rate) / safe_downside_std * ann

# --- ИЗМЕНЕНИЕ: Старая Python-функция удалена, так как заменена на Cython-версию ---

def objective(trial: optuna.Trial,
              cfg: TrainConfig,
              total_timesteps: int,
              train_data_by_token: dict,
              train_obs_by_token: dict,
              val_data_by_token: dict,
              val_obs_by_token: dict,
              test_data_by_token: dict,
              test_obs_by_token: dict,
              norm_stats: dict,
              sim_config: dict,
              timing_env_kwargs: dict,
              env_runtime_overrides: Mapping[str, Any],
              leak_guard_kwargs: dict,
              trials_dir: Path,
              tensorboard_log_dir: Path | None,
              n_envs_override: int | None):

    print(f">>> Trial {trial.number+1} with budget={total_timesteps}")

    def _extract_bins_vol_from_cfg(cfg, default=EXPECTED_VOLUME_BINS):
        try:
            aw = getattr(getattr(cfg, "algo", None), "action_wrapper", None)
            val = getattr(aw, "bins_vol", None) if aw is not None else None
            if val is None and hasattr(aw, "__dict__"):
                val = aw.__dict__.get("bins_vol")
            if val is None:
                return int(default)
            coerced = int(val)
            if coerced != EXPECTED_VOLUME_BINS:
                raise ValueError(
                    "BAR volume head requires exactly "
                    f"{EXPECTED_VOLUME_BINS} bins (config requested {coerced})."
                )
            return EXPECTED_VOLUME_BINS
        except Exception as exc:
            raise ValueError("Failed to resolve volume bins from config") from exc

    bins_vol = _extract_bins_vol_from_cfg(cfg, default=EXPECTED_VOLUME_BINS)

    def _resolve_nested(cfg_obj, attr: str):
        if cfg_obj is None:
            return None
        if isinstance(cfg_obj, Mapping):
            return cfg_obj.get(attr)
        try:
            value = getattr(cfg_obj, attr)
        except AttributeError:
            value = None
        if value is not None:
            return value
        for extra_name in ("__dict__", "__pydantic_extra__", "model_extra"):
            try:
                extra = getattr(cfg_obj, extra_name)
            except AttributeError:
                extra = None
            if isinstance(extra, Mapping) and attr in extra:
                return extra.get(attr)
        return None

    def _coerce_bool(value: object) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _extract_action_overrides_from_cfg(cfg_obj) -> tuple[dict[str, object], bool]:
        def _normalise_section(section_obj: Any) -> Mapping[str, Any]:
            if section_obj is None:
                return {}
            if hasattr(section_obj, "dict"):
                try:
                    payload = section_obj.dict()
                except TypeError:
                    payload = None
            else:
                payload = None
            if payload is None:
                if isinstance(section_obj, Mapping):
                    payload = dict(section_obj)
                else:
                    payload = {}
                    for extra_name in ("__dict__", "__pydantic_extra__", "model_extra"):
                        try:
                            extra = getattr(section_obj, extra_name)
                        except AttributeError:
                            extra = None
                        if isinstance(extra, Mapping):
                            payload.update(extra)
            return payload if isinstance(payload, Mapping) else {}

        algo_cfg = _resolve_nested(cfg_obj, "algo")
        actions_payload = _normalise_section(_resolve_nested(algo_cfg, "actions"))
        wrapper_payload = _normalise_section(_resolve_nested(algo_cfg, "action_wrapper"))

        overrides: dict[str, object] = {}
        long_only_flag = False

        def _update_overrides(payload: Mapping[str, Any]) -> None:
            nonlocal long_only_flag
            if not payload:
                return
            if "long_only" in payload:
                long_only_flag = _coerce_bool(payload.get("long_only"))
            if "max_asset_weight" in payload and payload.get("max_asset_weight") is not None:
                value = payload.get("max_asset_weight")
                try:
                    overrides["max_asset_weight"] = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid max_asset_weight value: {value!r}"
                    ) from exc

        _update_overrides(actions_payload)
        _update_overrides(wrapper_payload)

        return overrides, long_only_flag

    action_overrides, long_only_flag = _extract_action_overrides_from_cfg(cfg)

    market_name = str(getattr(cfg, "market", "spot") or "spot").strip().lower()
    if market_name != "spot":
        raise RuntimeError(
            "Score-based training supports only spot market configurations (derivatives disabled)"
        )

    if not long_only_flag:
        raise RuntimeError(
            "Score-based policy requires cfg.algo.actions.long_only = true"
        )


    # ИСПРАВЛЕНО: window_size возвращен в пространство поиска HPO
    def _get_model_param_value(cfg: TrainConfig, key: str):
        model_cfg = getattr(cfg, "model", None)
        if model_cfg is None:
            return None
        sentinel = object()
        try:
            value = getattr(model_cfg, key, sentinel)
        except Exception:
            value = sentinel
        if value is not sentinel:
            return value
        try:
            model_extra = _get_extra_mapping(model_cfg)
        except Exception:
            model_extra = getattr(model_cfg, "__dict__", {}) or {}
        if isinstance(model_extra, Mapping) and key in model_extra:
            return model_extra[key]
        params_obj = getattr(model_cfg, "params", None)
        if params_obj is None:
            return None
        if isinstance(params_obj, Mapping):
            return params_obj.get(key)
        getter = getattr(params_obj, "get", None)
        if callable(getter):
            sentinel = object()
            try:
                value = getter(key, sentinel)
            except TypeError:
                try:
                    value = getter(key)
                except Exception:
                    value = sentinel
            except Exception:
                value = sentinel
            if value is not sentinel:
                return value
        attr_sentinel = object()
        try:
            attr_value = getattr(params_obj, key, attr_sentinel)
        except Exception:
            attr_value = attr_sentinel
        if attr_value is not attr_sentinel:
            return attr_value
        d = getattr(params_obj, "__dict__", None)
        if isinstance(d, Mapping) and key in d:
            return d.get(key)
        dict_method = getattr(params_obj, "dict", None)
        if callable(dict_method):
            try:
                pd = dict_method()
            except TypeError:
                pd = None
            if isinstance(pd, Mapping):
                return pd.get(key)
        return None

    def _has_model_param(cfg: TrainConfig, key: str) -> bool:
        try:
            model_cfg = getattr(cfg, "model", None)
        except Exception:
            return False
        if model_cfg is None:
            return False
        sentinel = object()
        try:
            value = getattr(model_cfg, key, sentinel)
        except Exception:
            value = sentinel
        if value is not sentinel:
            return True
        try:
            model_extra = _get_extra_mapping(model_cfg)
        except Exception:
            model_extra = getattr(model_cfg, "__dict__", {}) or {}
        if isinstance(model_extra, Mapping) and key in model_extra:
            return True
        params_obj = getattr(model_cfg, "params", None)
        if params_obj is None:
            return False
        if isinstance(params_obj, Mapping):
            return key in params_obj
        attr_sentinel = object()
        try:
            attr_value = getattr(params_obj, key, attr_sentinel)
        except Exception:
            attr_value = attr_sentinel
        if attr_value is not attr_sentinel:
            return True
        contains = getattr(params_obj, "__contains__", None)
        if callable(contains):
            try:
                if contains(key):
                    return True
            except Exception:
                pass
        getter = getattr(params_obj, "get", None)
        if callable(getter):
            sentinel = object()
            try:
                value = getter(key, sentinel)
            except TypeError:
                try:
                    value = getter(key)
                except Exception:
                    value = sentinel
            except Exception:
                value = sentinel
            if value is not sentinel:
                return True
        d = getattr(params_obj, "__dict__", None)
        if isinstance(d, Mapping) and key in d:
            return True
        dict_method = getattr(params_obj, "dict", None)
        if callable(dict_method):
            try:
                pd = dict_method()
            except TypeError:
                pd = None
            if isinstance(pd, Mapping):
                return key in pd
        return False

    def _get_extra_mapping(obj):
        for name in ("__pydantic_extra__", "model_extra", "__dict__"):
            extra = getattr(obj, name, None)
            if isinstance(extra, Mapping):
                return extra
        return {}

    def _extract_section_payload(section: Any, keys: Sequence[str]) -> Dict[str, Any]:
        if section is None:
            return {}
        if isinstance(section, Mapping):
            return dict(section)

        payload: Dict[str, Any] = {}
        getter = getattr(section, "dict", None)
        if callable(getter):
            try:
                data = getter()
            except TypeError:
                data = None
            if isinstance(data, Mapping):
                payload.update(dict(data))

        extra = _get_extra_mapping(section)
        if isinstance(extra, Mapping):
            payload.update(dict(extra))

        for key in keys:
            try:
                value = getattr(section, key)
            except Exception:
                value = None
            if value is not None and key not in payload:
                payload[key] = value

        return payload

    def _extract_loss_head_config(
        cfg: TrainConfig,
    ) -> tuple[Optional[Dict[str, float]], Optional[Dict[str, bool]]]:
        loss_masks_cfg = getattr(cfg, "loss_masks", None)
        if loss_masks_cfg is None:
            extra = _get_extra_mapping(cfg)
            loss_masks_cfg = extra.get("loss_masks")

        if hasattr(loss_masks_cfg, "dict"):
            try:
                loss_masks_cfg = loss_masks_cfg.dict()
            except TypeError:
                pass
        if not isinstance(loss_masks_cfg, Mapping):
            return None, None

        include_payload = loss_masks_cfg.get("include_heads")
        if hasattr(include_payload, "dict"):
            try:
                include_payload = include_payload.dict()
            except TypeError:
                pass
        if not isinstance(include_payload, Mapping):
            return None, None

        weights: Dict[str, float] = {}
        include_map: Dict[str, bool] = {}
        for head_name, raw_value in include_payload.items():
            if raw_value is None:
                continue
            if isinstance(raw_value, bool):
                weights[head_name] = 1.0 if raw_value else 0.0
                include_map[head_name] = bool(raw_value)
            else:
                try:
                    weights[head_name] = float(raw_value)
                except (TypeError, ValueError):
                    pass
                else:
                    include_map[head_name] = weights[head_name] != 0.0
        if not include_map:
            include_map = {}
        return (weights or None, include_map or None)

    def _coerce_optional_int(value, key: str):
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(
                f"Invalid value '{value}' for '{key}' (expected integer) in cfg.model.params"
            )
        if isinstance(value, int):
            return int(value)
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid value '{value}' for '{key}' (expected integer) in cfg.model.params"
            )
        return coerced

    def _coerce_optional_float(value, key: str):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid value '{value}' for '{key}' (expected float-compatible) in cfg.model.params"
            )

    def _coerce_optional_bool(value, key: str):
        if value is None:
            return None
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, (int, float)):
            if value == 0 or value == 0.0:
                return False
            if value == 1 or value == 1.0:
                return True
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        raise ValueError(
            f"Invalid value '{value}' for '{key}' (expected boolean) in cfg.model.params"
        )

    learning_rate_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "learning_rate"), "learning_rate"
    )
    optimizer_lr_min_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "optimizer_lr_min"), "optimizer_lr_min"
    )
    scheduler_min_lr_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "scheduler_min_lr"), "scheduler_min_lr"
    )
    # Fallback to top-level model scheduler_min_lr if not found within params
    if scheduler_min_lr_cfg is None:
        try:
            scheduler_min_lr_cfg = float(getattr(cfg.model, "scheduler_min_lr", None))
        except Exception:
            scheduler_min_lr_cfg = None
    optimizer_lr_max_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "optimizer_lr_max"), "optimizer_lr_max"
    )
    gamma_cfg = _coerce_optional_float(_get_model_param_value(cfg, "gamma"), "gamma")
    gae_lambda_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "gae_lambda"), "gae_lambda"
    )
    clip_range_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "clip_range"), "clip_range"
    )
    clip_range_vf_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "clip_range_vf"), "clip_range_vf"
    )
    value_scale_update_enabled_cfg = _coerce_optional_bool(
        _get_model_param_value(cfg, "value_scale_update_enabled"),
        "value_scale_update_enabled",
    )
    value_target_scale_fixed_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "value_target_scale_fixed"),
        "value_target_scale_fixed",
    )
    normalize_returns_cfg = _coerce_optional_bool(
        _get_model_param_value(cfg, "normalize_returns"),
        "normalize_returns",
    )
    value_scale_controller_cfg = _get_model_param_value(cfg, "value_scale_controller")
    popart_holdout_loader = _build_popart_holdout_loader(value_scale_controller_cfg)
    ent_coef_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "ent_coef"), "ent_coef"
    )
    ent_coef_final_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "ent_coef_final"), "ent_coef_final"
    )
    ent_coef_min_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "ent_coef_min"), "ent_coef_min"
    )
    ent_coef_decay_steps_cfg = _coerce_optional_int(
        _get_model_param_value(cfg, "ent_coef_decay_steps"), "ent_coef_decay_steps"
    )
    ent_coef_plateau_window_cfg = _coerce_optional_int(
        _get_model_param_value(cfg, "ent_coef_plateau_window"), "ent_coef_plateau_window"
    )
    ent_coef_plateau_tolerance_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "ent_coef_plateau_tolerance"), "ent_coef_plateau_tolerance"
    )
    ent_coef_plateau_min_updates_cfg = _coerce_optional_int(
        _get_model_param_value(cfg, "ent_coef_plateau_min_updates"), "ent_coef_plateau_min_updates"
    )
    vf_coef_cfg = _coerce_optional_float(_get_model_param_value(cfg, "vf_coef"), "vf_coef")
    max_grad_norm_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "max_grad_norm"), "max_grad_norm"
    )
    n_steps_cfg = _coerce_optional_int(_get_model_param_value(cfg, "n_steps"), "n_steps")
    batch_size_cfg = _coerce_optional_int(
        _get_model_param_value(cfg, "batch_size"), "batch_size"
    )
    microbatch_size_cfg = _coerce_optional_int(
        _get_model_param_value(cfg, "microbatch_size"), "microbatch_size"
    )
    n_epochs_cfg = _coerce_optional_int(
        _get_model_param_value(cfg, "n_epochs"), "n_epochs"
    )
    if n_epochs_cfg is None:
        # Некоторые конфиги (особенно легаси или кастомные CLI-переопределения)
        # могут не содержать ``model.params.n_epochs``. Вместо жёсткого падения
        # позволяем Optuna подобрать разумное значение в допустимом диапазоне.
        n_epochs_cfg = trial.suggest_int("n_epochs", 2, 4)
    target_kl_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "target_kl"), "target_kl"
    )
    kl_early_stop_cfg = _coerce_optional_bool(
        _get_model_param_value(cfg, "kl_early_stop"), "kl_early_stop"
    )
    kl_exceed_stop_fraction_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "kl_exceed_stop_fraction"),
        "kl_exceed_stop_fraction",
    )
    kl_absolute_stop_factor_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "kl_absolute_stop_factor"),
        "kl_absolute_stop_factor",
    )
    seed_cfg = _coerce_optional_int(
        _get_model_param_value(cfg, "seed"), "seed"
    )
    kl_epoch_decay_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "kl_epoch_decay"), "kl_epoch_decay"
    )
    kl_early_stop_use_ema_cfg = None
    kl_ema_updates_cfg = None
    kl_ema_alpha_cfg = None
    kl_consec_minibatches_cfg = None
    kl_absolute_stop_factor_override = None
    kl_exceed_stop_fraction_override = None
    kl_section_raw = _get_model_param_value(cfg, "kl")
    kl_section_map = _extract_section_payload(
        kl_section_raw, ["target_kl", "use_lr_decay", "penalty", "early_stop"]
    )
    kl_use_lr_decay_cfg = None
    if "target_kl" in kl_section_map:
        nested_target_kl = _coerce_optional_float(
            kl_section_map.get("target_kl"), "kl.target_kl"
        )
        if nested_target_kl is not None:
            target_kl_cfg = nested_target_kl
    if "use_lr_decay" in kl_section_map:
        try:
            kl_use_lr_decay_cfg = _coerce_optional_bool(
                kl_section_map.get("use_lr_decay"), "kl.use_lr_decay"
            )
        except ValueError:
            print(
                "Warning: cfg.model.params.kl.use_lr_decay is invalid; ignoring override"
            )
            kl_use_lr_decay_cfg = None

    early_stop_section_map = _extract_section_payload(
        kl_section_map.get("early_stop"),
        [
            "enabled",
            "use_ema",
            "ema_updates",
            "ema_alpha",
            "exceed_stop_fraction",
            "consec_minibatches",
            "absolute_stop_factor",
        ],
    )
    if early_stop_section_map:
        if "enabled" in early_stop_section_map:
            try:
                early_stop_enabled = _coerce_optional_bool(
                    early_stop_section_map.get("enabled"), "kl.early_stop.enabled"
                )
            except ValueError:
                print(
                    "Warning: cfg.model.params.kl.early_stop.enabled is invalid; ignoring override"
                )
                early_stop_enabled = None
            if early_stop_enabled is not None:
                kl_early_stop_cfg = early_stop_enabled
        if "use_ema" in early_stop_section_map:
            try:
                kl_early_stop_use_ema_cfg = _coerce_optional_bool(
                    early_stop_section_map.get("use_ema"), "kl.early_stop.use_ema"
                )
            except ValueError:
                print(
                    "Warning: cfg.model.params.kl.early_stop.use_ema is invalid; ignoring override"
                )
                kl_early_stop_use_ema_cfg = None
        if "ema_updates" in early_stop_section_map:
            try:
                kl_ema_updates_cfg = _coerce_optional_int(
                    early_stop_section_map.get("ema_updates"), "kl.early_stop.ema_updates"
                )
            except ValueError:
                print(
                    "Warning: cfg.model.params.kl.early_stop.ema_updates is invalid; ignoring override"
                )
                kl_ema_updates_cfg = None
        if "ema_alpha" in early_stop_section_map:
            try:
                kl_ema_alpha_cfg = _coerce_optional_float(
                    early_stop_section_map.get("ema_alpha"), "kl.early_stop.ema_alpha"
                )
            except ValueError:
                print(
                    "Warning: cfg.model.params.kl.early_stop.ema_alpha is invalid; ignoring override"
                )
                kl_ema_alpha_cfg = None
        if "exceed_stop_fraction" in early_stop_section_map:
            try:
                kl_exceed_stop_fraction_override = _coerce_optional_float(
                    early_stop_section_map.get("exceed_stop_fraction"),
                    "kl.early_stop.exceed_stop_fraction",
                )
            except ValueError:
                print(
                    "Warning: cfg.model.params.kl.early_stop.exceed_stop_fraction is invalid; ignoring override"
                )
                kl_exceed_stop_fraction_override = None
        if "consec_minibatches" in early_stop_section_map:
            try:
                kl_consec_minibatches_cfg = _coerce_optional_int(
                    early_stop_section_map.get("consec_minibatches"),
                    "kl.early_stop.consec_minibatches",
                )
            except ValueError:
                print(
                    "Warning: cfg.model.params.kl.early_stop.consec_minibatches is invalid; ignoring override"
                )
                kl_consec_minibatches_cfg = None

        if "absolute_stop_factor" in early_stop_section_map:
            try:
                kl_absolute_stop_factor_override = _coerce_optional_float(
                    early_stop_section_map.get("absolute_stop_factor"),
                    "kl.early_stop.absolute_stop_factor",
                )
            except ValueError:
                print(
                    "Warning: cfg.model.params.kl.early_stop.absolute_stop_factor is invalid; ignoring override",
                )
                kl_absolute_stop_factor_override = None

    penalty_section_map = _extract_section_payload(
        kl_section_map.get("penalty"), ["type", "beta_init", "pid"]
    )
    penalty_type_raw = penalty_section_map.get("type")
    kl_penalty_type = str(penalty_type_raw) if penalty_type_raw is not None else "adaptive_beta"
    kl_penalty_beta_init_cfg = _coerce_optional_float(
        penalty_section_map.get("beta_init"), "kl.penalty.beta_init"
    )
    pid_section_map = _extract_section_payload(
        penalty_section_map.get("pid"), ["kp", "ki", "kd", "beta_min", "beta_max"]
    )
    kl_penalty_pid_kp_cfg = _coerce_optional_float(
        pid_section_map.get("kp"), "kl.penalty.pid.kp"
    )
    kl_penalty_pid_ki_cfg = _coerce_optional_float(
        pid_section_map.get("ki"), "kl.penalty.pid.ki"
    )
    kl_penalty_pid_kd_cfg = _coerce_optional_float(
        pid_section_map.get("kd"), "kl.penalty.pid.kd"
    )
    kl_penalty_beta_min_cfg = _coerce_optional_float(
        pid_section_map.get("beta_min"), "kl.penalty.pid.beta_min"
    )
    kl_penalty_beta_max_cfg = _coerce_optional_float(
        pid_section_map.get("beta_max"), "kl.penalty.pid.beta_max"
    )

    if kl_penalty_beta_init_cfg is None:
        kl_penalty_beta_init_cfg = _coerce_optional_float(
            _get_model_param_value(cfg, "kl_penalty_beta"), "kl_penalty_beta"
        )

    trade_frequency_penalty_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "trade_frequency_penalty"),
        "trade_frequency_penalty",
    )
    turnover_penalty_coef_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "turnover_penalty_coef"),
        "turnover_penalty_coef",
    )
    reward_return_clip_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "reward_return_clip"),
        "reward_return_clip",
    )
    turnover_norm_cap_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "turnover_norm_cap"),
        "turnover_norm_cap",
    )
    reward_cap_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "reward_cap"),
        "reward_cap",
    )

    if target_kl_cfg is None:
        fallback_target_kl = 0.08
        print(
            "Warning: cfg.model.params.target_kl is missing; "
            f"defaulting to {fallback_target_kl:.4f}."
        )
        target_kl_cfg = fallback_target_kl
    if kl_early_stop_cfg is None:
        print(
            "Warning: cfg.model.params.kl_early_stop is missing; defaulting to True"
        )
        kl_early_stop_cfg = True
    if kl_exceed_stop_fraction_cfg is None:
        kl_exceed_stop_fraction_cfg = (
            0.30 if kl_early_stop_cfg else 0.0
        )
    if turnover_penalty_coef_cfg is None:
        print(
            "Warning: cfg.model.params.turnover_penalty_coef is missing; defaulting to 0.0"
        )
        turnover_penalty_coef_cfg = 0.0

    if reward_return_clip_cfg is None or not math.isfinite(reward_return_clip_cfg) or reward_return_clip_cfg <= 0.0:
        reward_return_clip_cfg = 10.0
    if turnover_norm_cap_cfg is None or not math.isfinite(turnover_norm_cap_cfg) or turnover_norm_cap_cfg <= 0.0:
        turnover_norm_cap_cfg = 1.0
    if reward_cap_cfg is None or not math.isfinite(reward_cap_cfg) or reward_cap_cfg <= 0.0:
        reward_cap_cfg = 10.0

    reward_clip_defaults = {
        "adaptive": True,
        "atr_window": 14,
        "hard_cap_pct": 2.0,
        "multiplier": 4.0,
    }
    reward_clip_cfg_raw = _get_model_param_value(cfg, "reward_clip")
    if isinstance(reward_clip_cfg_raw, Mapping):
        reward_clip_cfg_map = dict(reward_clip_cfg_raw)
    elif reward_clip_cfg_raw is None:
        reward_clip_cfg_map = {}
    else:
        reward_clip_cfg_map = {
            key: getattr(reward_clip_cfg_raw, key)
            for key in ("adaptive", "atr_window", "hard_cap_pct", "multiplier")
            if hasattr(reward_clip_cfg_raw, key)
        }
    reward_clip_params = dict(reward_clip_defaults)
    adaptive_cfg = reward_clip_cfg_map.get("adaptive")
    if adaptive_cfg is not None:
        reward_clip_params["adaptive"] = bool(adaptive_cfg)
    atr_window_cfg = reward_clip_cfg_map.get("atr_window")
    if atr_window_cfg is not None:
        try:
            reward_clip_params["atr_window"] = max(1, int(atr_window_cfg))
        except (TypeError, ValueError):
            pass
    hard_cap_cfg = reward_clip_cfg_map.get("hard_cap_pct")
    if hard_cap_cfg is not None:
        try:
            value = float(hard_cap_cfg)
            if math.isfinite(value) and value > 0.0:
                reward_clip_params["hard_cap_pct"] = value
        except (TypeError, ValueError):
            pass
    multiplier_cfg = reward_clip_cfg_map.get("multiplier")
    if multiplier_cfg is not None:
        try:
            value = float(multiplier_cfg)
            if math.isfinite(value) and value >= 0.0:
                reward_clip_params["multiplier"] = value
        except (TypeError, ValueError):
            pass
    v_range_ema_alpha_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "v_range_ema_alpha"), "v_range_ema_alpha"
    )
    value_scale_cfg = _get_model_param_value(cfg, "value_scale")
    value_scale_ema_beta_cfg = None
    value_scale_max_rel_step_cfg = None
    value_scale_std_floor_cfg = None
    value_scale_window_updates_cfg = None
    value_scale_warmup_updates_cfg = None
    value_scale_freeze_after_cfg = None
    value_scale_freeze_after_updates_cfg = None
    value_scale_never_freeze_cfg = None
    value_scale_target_scale_ema_beta_cfg = None
    value_scale_max_change_pct_cfg = None
    value_scale_range_max_rel_step_cfg = None
    value_scale_range_max_rel_step_provided = False
    value_scale_stability_cfg: dict[str, Any] | None = None
    value_scale_stability_patience_cfg = None
    if isinstance(value_scale_cfg, Mapping):
        value_scale_ema_beta_cfg = _coerce_optional_float(
            value_scale_cfg.get("ema_beta"), "value_scale.ema_beta"
        )
        value_scale_max_rel_step_cfg = _coerce_optional_float(
            value_scale_cfg.get("max_rel_step"), "value_scale.max_rel_step"
        )
        value_scale_std_floor_cfg = _coerce_optional_float(
            value_scale_cfg.get("std_floor"), "value_scale.std_floor"
        )
        value_scale_window_updates_cfg = _coerce_optional_int(
            value_scale_cfg.get("window_updates"), "value_scale.window_updates"
        )
        value_scale_warmup_updates_cfg = _coerce_optional_int(
            value_scale_cfg.get("warmup_updates"), "value_scale.warmup_updates"
        )
        value_scale_freeze_after_cfg = _coerce_optional_int(
            value_scale_cfg.get("freeze_after"), "value_scale.freeze_after"
        )
        value_scale_freeze_after_updates_cfg = _coerce_optional_int(
            value_scale_cfg.get("freeze_after_updates"),
            "value_scale.freeze_after_updates",
        )
        value_scale_never_freeze_cfg = _coerce_optional_bool(
            value_scale_cfg.get("never_freeze"), "value_scale.never_freeze"
        )
        if "range_max_rel_step" in value_scale_cfg:
            value_scale_range_max_rel_step_provided = True
        value_scale_range_max_rel_step_cfg = _coerce_optional_float(
            value_scale_cfg.get("range_max_rel_step"), "value_scale.range_max_rel_step"
        )
        value_scale_target_scale_ema_beta_cfg = _coerce_optional_float(
            value_scale_cfg.get("target_scale_ema_beta"),
            "value_scale.target_scale_ema_beta",
        )
        value_scale_max_change_pct_cfg = _coerce_optional_float(
            value_scale_cfg.get("max_change_pct"),
            "value_scale.max_change_pct",
        )
        stability_raw = value_scale_cfg.get("stability")
        if isinstance(stability_raw, Mapping):
            stability_dict: dict[str, Any] = {}
            stability_ev = _coerce_optional_float(
                stability_raw.get("min_explained_variance"),
                "value_scale.stability.min_explained_variance",
            )
            if stability_ev is None:
                stability_ev = _coerce_optional_float(
                    stability_raw.get("ev_min"),
                    "value_scale.stability.ev_min",
                )
            if stability_ev is not None:
                stability_dict["min_explained_variance"] = stability_ev
            stability_p95 = _coerce_optional_float(
                stability_raw.get("max_abs_p95"),
                "value_scale.stability.max_abs_p95",
            )
            if stability_p95 is None:
                stability_p95 = _coerce_optional_float(
                    stability_raw.get("ret_abs_p95_max"),
                    "value_scale.stability.ret_abs_p95_max",
                )
            if stability_p95 is not None:
                stability_dict["max_abs_p95"] = stability_p95
            stability_patience = _coerce_optional_int(
                stability_raw.get("patience"), "value_scale.stability.patience"
            )
            if stability_patience is None:
                stability_patience = _coerce_optional_int(
                    stability_raw.get("consecutive"),
                    "value_scale.stability.consecutive",
                )
            value_scale_stability_patience_cfg = stability_patience
            if stability_patience is not None:
                stability_dict["patience"] = stability_patience
            if stability_dict:
                value_scale_stability_cfg = stability_dict
        stability_patience_alias = _coerce_optional_int(
            value_scale_cfg.get("stability_patience"), "value_scale.stability_patience"
        )
        if stability_patience_alias is not None:
            value_scale_stability_patience_cfg = stability_patience_alias
    elif value_scale_cfg is not None:
        value_scale_ema_beta_cfg = _coerce_optional_float(
            getattr(value_scale_cfg, "ema_beta", None), "value_scale.ema_beta"
        )
        value_scale_max_rel_step_cfg = _coerce_optional_float(
            getattr(value_scale_cfg, "max_rel_step", None), "value_scale.max_rel_step"
        )
        value_scale_std_floor_cfg = _coerce_optional_float(
            getattr(value_scale_cfg, "std_floor", None), "value_scale.std_floor"
        )
        value_scale_window_updates_cfg = _coerce_optional_int(
            getattr(value_scale_cfg, "window_updates", None), "value_scale.window_updates"
        )
        value_scale_warmup_updates_cfg = _coerce_optional_int(
            getattr(value_scale_cfg, "warmup_updates", None), "value_scale.warmup_updates"
        )
        value_scale_freeze_after_cfg = _coerce_optional_int(
            getattr(value_scale_cfg, "freeze_after", None), "value_scale.freeze_after"
        )
        value_scale_freeze_after_updates_cfg = _coerce_optional_int(
            getattr(value_scale_cfg, "freeze_after_updates", None),
            "value_scale.freeze_after_updates",
        )
        value_scale_never_freeze_cfg = _coerce_optional_bool(
            getattr(value_scale_cfg, "never_freeze", None),
            "value_scale.never_freeze",
        )
        if hasattr(value_scale_cfg, "range_max_rel_step") or (
            hasattr(value_scale_cfg, "__dict__")
            and "range_max_rel_step" in getattr(value_scale_cfg, "__dict__", {})
        ):
            value_scale_range_max_rel_step_provided = True
        value_scale_range_max_rel_step_cfg = _coerce_optional_float(
            getattr(value_scale_cfg, "range_max_rel_step", None),
            "value_scale.range_max_rel_step",
        )
        value_scale_target_scale_ema_beta_cfg = _coerce_optional_float(
            getattr(value_scale_cfg, "target_scale_ema_beta", None),
            "value_scale.target_scale_ema_beta",
        )
        value_scale_max_change_pct_cfg = _coerce_optional_float(
            getattr(value_scale_cfg, "max_change_pct", None),
            "value_scale.max_change_pct",
        )
        stability_raw = getattr(value_scale_cfg, "stability", None)
        if isinstance(stability_raw, Mapping):
            stability_dict = {}
            stability_ev = _coerce_optional_float(
                stability_raw.get("min_explained_variance"),
                "value_scale.stability.min_explained_variance",
            )
            if stability_ev is None:
                stability_ev = _coerce_optional_float(
                    stability_raw.get("ev_min"),
                    "value_scale.stability.ev_min",
                )
            if stability_ev is not None:
                stability_dict["min_explained_variance"] = stability_ev
            stability_p95 = _coerce_optional_float(
                stability_raw.get("max_abs_p95"),
                "value_scale.stability.max_abs_p95",
            )
            if stability_p95 is None:
                stability_p95 = _coerce_optional_float(
                    stability_raw.get("ret_abs_p95_max"),
                    "value_scale.stability.ret_abs_p95_max",
                )
            if stability_p95 is not None:
                stability_dict["max_abs_p95"] = stability_p95
            stability_patience = _coerce_optional_int(
                stability_raw.get("patience"), "value_scale.stability.patience"
            )
            if stability_patience is None:
                stability_patience = _coerce_optional_int(
                    stability_raw.get("consecutive"),
                    "value_scale.stability.consecutive",
                )
            value_scale_stability_patience_cfg = stability_patience
            if stability_patience is not None:
                stability_dict["patience"] = stability_patience
            if stability_dict:
                value_scale_stability_cfg = stability_dict
        stability_patience_alias = _coerce_optional_int(
            getattr(value_scale_cfg, "stability_patience", None),
            "value_scale.stability_patience",
        )
        if stability_patience_alias is not None:
            value_scale_stability_patience_cfg = stability_patience_alias
    else:
        value_scale_ema_beta_cfg = _coerce_optional_float(
            _get_model_param_value(cfg, "value_scale_ema_beta"), "value_scale_ema_beta"
        )
        value_scale_max_rel_step_cfg = _coerce_optional_float(
            _get_model_param_value(cfg, "value_scale_max_rel_step"), "value_scale_max_rel_step"
        )
        value_scale_std_floor_cfg = _coerce_optional_float(
            _get_model_param_value(cfg, "value_scale_std_floor"), "value_scale_std_floor"
        )
        value_scale_window_updates_cfg = _coerce_optional_int(
            _get_model_param_value(cfg, "value_scale_window_updates"),
            "value_scale_window_updates",
        )
        value_scale_warmup_updates_cfg = _coerce_optional_int(
            _get_model_param_value(cfg, "value_scale_warmup_updates"),
            "value_scale_warmup_updates",
        )
        value_scale_freeze_after_cfg = _coerce_optional_int(
            _get_model_param_value(cfg, "value_scale_freeze_after"),
            "value_scale_freeze_after",
        )
        if _has_model_param(cfg, "value_scale_range_max_rel_step"):
            value_scale_range_max_rel_step_provided = True
        value_scale_range_max_rel_step_cfg = _coerce_optional_float(
            _get_model_param_value(cfg, "value_scale_range_max_rel_step"),
            "value_scale_range_max_rel_step",
        )

    if (
        value_scale_range_max_rel_step_cfg is None
        and not value_scale_range_max_rel_step_provided
    ):
        value_scale_range_max_rel_step_cfg = 0.15
        stability_raw = _get_model_param_value(cfg, "value_scale_stability")
        if isinstance(stability_raw, Mapping):
            stability_dict = {}
            stability_ev = _coerce_optional_float(
                stability_raw.get("min_explained_variance"),
                "value_scale_stability.min_explained_variance",
            )
            if stability_ev is None:
                stability_ev = _coerce_optional_float(
                    stability_raw.get("ev_min"), "value_scale_stability.ev_min"
                )
            if stability_ev is not None:
                stability_dict["min_explained_variance"] = stability_ev
            stability_p95 = _coerce_optional_float(
                stability_raw.get("max_abs_p95"),
                "value_scale_stability.max_abs_p95",
            )
            if stability_p95 is None:
                stability_p95 = _coerce_optional_float(
                    stability_raw.get("ret_abs_p95_max"),
                    "value_scale_stability.ret_abs_p95_max",
                )
            if stability_p95 is not None:
                stability_dict["max_abs_p95"] = stability_p95
            stability_patience = _coerce_optional_int(
                stability_raw.get("patience"), "value_scale_stability.patience"
            )
            if stability_patience is None:
                stability_patience = _coerce_optional_int(
                    stability_raw.get("consecutive"),
                    "value_scale_stability.consecutive",
                )
            value_scale_stability_patience_cfg = stability_patience
            if stability_patience is not None:
                stability_dict["patience"] = stability_patience
            if stability_dict:
                value_scale_stability_cfg = stability_dict
        stability_patience_alias = _coerce_optional_int(
            _get_model_param_value(cfg, "value_scale_stability_patience"),
            "value_scale_stability_patience",
        )
        if stability_patience_alias is not None:
            value_scale_stability_patience_cfg = stability_patience_alias
    bc_warmup_steps_cfg = _coerce_optional_int(
        _get_model_param_value(cfg, "bc_warmup_steps"), "bc_warmup_steps"
    )
    bc_decay_steps_cfg = _coerce_optional_int(
        _get_model_param_value(cfg, "bc_decay_steps"), "bc_decay_steps"
    )
    bc_final_coef_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "bc_final_coef"), "bc_final_coef"
    )
    vf_coef_warmup_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "vf_coef_warmup"), "vf_coef_warmup"
    )
    vf_coef_warmup_updates_cfg = _coerce_optional_int(
        _get_model_param_value(cfg, "vf_coef_warmup_updates"), "vf_coef_warmup_updates"
    )
    vf_bad_explained_scale_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "vf_bad_explained_scale"), "vf_bad_explained_scale"
    )
    vf_bad_explained_floor_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "vf_bad_explained_floor"), "vf_bad_explained_floor"
    )
    bad_explained_patience_cfg = _coerce_optional_int(
        _get_model_param_value(cfg, "bad_explained_patience"), "bad_explained_patience"
    )
    entropy_boost_factor_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "entropy_boost_factor"), "entropy_boost_factor"
    )
    entropy_boost_cap_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "entropy_boost_cap"), "entropy_boost_cap"
    )
    clip_range_warmup_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "clip_range_warmup"), "clip_range_warmup"
    )
    clip_range_warmup_updates_cfg = _coerce_optional_int(
        _get_model_param_value(cfg, "clip_range_warmup_updates"), "clip_range_warmup_updates"
    )
    critic_grad_warmup_updates_cfg = _coerce_optional_int(
        _get_model_param_value(cfg, "critic_grad_warmup_updates"), "critic_grad_warmup_updates"
    )
    cvar_activation_threshold_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "cvar_activation_threshold"), "cvar_activation_threshold"
    )
    cvar_activation_hysteresis_cfg = _coerce_optional_float(
        _get_model_param_value(cfg, "cvar_activation_hysteresis"), "cvar_activation_hysteresis"
    )
    cvar_ramp_updates_cfg = _coerce_optional_int(
        _get_model_param_value(cfg, "cvar_ramp_updates"), "cvar_ramp_updates"
    )

    optimizer_lr_min_value = 1e-5
    if optimizer_lr_min_cfg is not None and math.isfinite(optimizer_lr_min_cfg):
        optimizer_lr_min_value = float(max(0.0, optimizer_lr_min_cfg))

    scheduler_min_lr_value = optimizer_lr_min_value
    if scheduler_min_lr_cfg is not None and math.isfinite(scheduler_min_lr_cfg):
        scheduler_min_lr_value = float(max(0.0, scheduler_min_lr_cfg))
        if scheduler_min_lr_value < optimizer_lr_min_value:
            scheduler_min_lr_value = optimizer_lr_min_value

    optimizer_lr_max_value = float("inf")
    if (
        optimizer_lr_max_cfg is not None
        and math.isfinite(optimizer_lr_max_cfg)
        and optimizer_lr_max_cfg > 0.0
    ):
        optimizer_lr_max_value = float(optimizer_lr_max_cfg)

    if math.isfinite(optimizer_lr_max_value) and optimizer_lr_max_value < optimizer_lr_min_value:
        optimizer_lr_max_value = optimizer_lr_min_value

    params = {
        "window_size": trial.suggest_categorical("window_size", [10, 20, 30]),
        "n_steps": n_steps_cfg if n_steps_cfg is not None else trial.suggest_categorical("n_steps", [512, 1024, 2048]),
        "n_epochs": int(n_epochs_cfg),
        "batch_size": batch_size_cfg if batch_size_cfg is not None else trial.suggest_categorical("batch_size", [64, 128, 256]),
        "ent_coef": ent_coef_cfg if ent_coef_cfg is not None else trial.suggest_float("ent_coef", 5e-5, 5e-3, log=True),
        "ent_coef_final": ent_coef_final_cfg if ent_coef_final_cfg is not None else None,
        "ent_coef_min": ent_coef_min_cfg if ent_coef_min_cfg is not None else 5e-4,
        "ent_coef_decay_steps": ent_coef_decay_steps_cfg if ent_coef_decay_steps_cfg is not None else 0,
        "ent_coef_plateau_window": ent_coef_plateau_window_cfg if ent_coef_plateau_window_cfg is not None else 0,
        "ent_coef_plateau_tolerance": ent_coef_plateau_tolerance_cfg if ent_coef_plateau_tolerance_cfg is not None else 0.0,
        "ent_coef_plateau_min_updates": ent_coef_plateau_min_updates_cfg if ent_coef_plateau_min_updates_cfg is not None else 0,
        "learning_rate": learning_rate_cfg if learning_rate_cfg is not None else trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True),
        "optimizer_lr_min": optimizer_lr_min_value,
        "scheduler_min_lr": scheduler_min_lr_value,
        "optimizer_lr_max": optimizer_lr_max_value,
        "risk_aversion_drawdown": trial.suggest_float("risk_aversion_drawdown", 0.05, 0.3),
        "risk_aversion_variance": trial.suggest_float("risk_aversion_variance", 0.005, 0.01),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "gamma": gamma_cfg if gamma_cfg is not None else trial.suggest_float("gamma", 0.97, 0.995),
        "gae_lambda": gae_lambda_cfg if gae_lambda_cfg is not None else trial.suggest_float("gae_lambda", 0.8, 1.0),
        "clip_range": clip_range_cfg if clip_range_cfg is not None else trial.suggest_float("clip_range", 0.08, 0.12),
        "clip_range_vf": clip_range_vf_cfg,
        "max_grad_norm": max_grad_norm_cfg if max_grad_norm_cfg is not None else trial.suggest_float("max_grad_norm", 0.3, 1.0),
        "target_kl": float(np.clip(target_kl_cfg, 0.02, 1.6)),
        "kl_epoch_decay": kl_epoch_decay_cfg if kl_epoch_decay_cfg is not None else 0.5,
        "kl_penalty_beta_init": (
            kl_penalty_beta_init_cfg if kl_penalty_beta_init_cfg is not None else 0.01
        ),
        "kl_penalty_beta_min": (
            kl_penalty_beta_min_cfg if kl_penalty_beta_min_cfg is not None else 1e-4
        ),
        "kl_penalty_beta_max": (
            kl_penalty_beta_max_cfg if kl_penalty_beta_max_cfg is not None else 1.0
        ),
        "kl_penalty_pid_kp": (
            kl_penalty_pid_kp_cfg if kl_penalty_pid_kp_cfg is not None else 0.5
        ),
        "kl_penalty_pid_ki": (
            kl_penalty_pid_ki_cfg if kl_penalty_pid_ki_cfg is not None else 0.05
        ),
        "kl_penalty_pid_kd": (
            kl_penalty_pid_kd_cfg if kl_penalty_pid_kd_cfg is not None else 0.10
        ),
        "kl_penalty_type": kl_penalty_type,
        "kl_use_lr_decay_requested": bool(kl_use_lr_decay_cfg)
        if kl_use_lr_decay_cfg is not None
        else False,
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "atr_multiplier": trial.suggest_float("atr_multiplier", 1.5, 3.0),
        "trailing_atr_mult": trial.suggest_float("trailing_atr_mult", 1.0, 2.0),
        "tp_atr_mult": trial.suggest_float("tp_atr_mult", 2.0, 4.0),
        "momentum_factor": trial.suggest_float("momentum_factor", 0.1, 0.7),
        "mean_reversion_factor": trial.suggest_float("mean_reversion_factor", 0.2, 0.8),
        "adversarial_factor": trial.suggest_float("adversarial_factor", 0.3, 0.9),
        "vf_coef": (
            vf_coef_cfg if vf_coef_cfg is not None else 1.8
        ),
        "v_range_ema_alpha": v_range_ema_alpha_cfg if v_range_ema_alpha_cfg is not None else trial.suggest_float("v_range_ema_alpha", 0.005, 0.05, log=True),
        "bc_warmup_steps": bc_warmup_steps_cfg if bc_warmup_steps_cfg is not None else 0,
        "bc_decay_steps": bc_decay_steps_cfg if bc_decay_steps_cfg is not None else 0,
        "bc_final_coef": bc_final_coef_cfg if bc_final_coef_cfg is not None else 0.0,
    }

    params["value_scale_update_enabled"] = (
        bool(value_scale_update_enabled_cfg)
        if value_scale_update_enabled_cfg is not None
        else True
    )
    params["value_target_scale_fixed"] = value_target_scale_fixed_cfg
    params["normalize_returns"] = (
        bool(normalize_returns_cfg) if normalize_returns_cfg is not None else True
    )
    params["value_scale_controller"] = value_scale_controller_cfg

    params["value_scale_ema_beta"] = (
        value_scale_ema_beta_cfg if value_scale_ema_beta_cfg is not None else 0.2
    )
    fallback_value_scale_max_rel_step = 0.15
    if (
        value_scale_max_rel_step_cfg is None
        or not math.isfinite(value_scale_max_rel_step_cfg)
        or value_scale_max_rel_step_cfg <= 0.0
    ):
        if value_scale_max_rel_step_cfg is not None:
            print(
                "Warning: cfg.model.params.value_scale.max_rel_step is invalid; "
                f"defaulting to {fallback_value_scale_max_rel_step:.2f}"
            )
        else:
            print(
                "Warning: cfg.model.params.value_scale.max_rel_step is missing; "
                f"defaulting to {fallback_value_scale_max_rel_step:.2f}"
            )
        value_scale_max_rel_step_cfg = fallback_value_scale_max_rel_step
    params["value_scale_max_rel_step"] = float(value_scale_max_rel_step_cfg)
    params["value_scale_std_floor"] = (
        value_scale_std_floor_cfg if value_scale_std_floor_cfg is not None else 3e-3
    )
    params["value_scale_window_updates"] = (
        value_scale_window_updates_cfg
        if value_scale_window_updates_cfg is not None
        else 16
    )
    params["value_scale_warmup_updates"] = (
        value_scale_warmup_updates_cfg
        if value_scale_warmup_updates_cfg is not None
        else 0
    )
    params["value_scale_freeze_after"] = value_scale_freeze_after_cfg
    params["value_scale_freeze_after_updates"] = value_scale_freeze_after_updates_cfg
    params["value_scale_never_freeze"] = value_scale_never_freeze_cfg
    params["value_scale_target_ema_beta"] = (
        value_scale_target_scale_ema_beta_cfg
        if value_scale_target_scale_ema_beta_cfg is not None
        else 0.9
    )
    params["value_scale_max_change_pct"] = value_scale_max_change_pct_cfg
    params["value_scale_range_max_rel_step"] = value_scale_range_max_rel_step_cfg
    params["value_scale_stability_patience"] = value_scale_stability_patience_cfg
    params["value_scale_stability"] = value_scale_stability_cfg
    params["vf_coef_warmup"] = (
        vf_coef_warmup_cfg if _has_model_param(cfg, "vf_coef_warmup") else None
    )
    params["vf_coef_warmup_updates"] = (
        vf_coef_warmup_updates_cfg if _has_model_param(cfg, "vf_coef_warmup_updates") else 0
    )
    params["vf_bad_explained_scale"] = (
        vf_bad_explained_scale_cfg if vf_bad_explained_scale_cfg is not None else 0.5
    )
    params["vf_bad_explained_floor"] = (
        vf_bad_explained_floor_cfg if vf_bad_explained_floor_cfg is not None else 0.02
    )
    params["bad_explained_patience"] = (
        bad_explained_patience_cfg if bad_explained_patience_cfg is not None else 2
    )
    params["entropy_boost_factor"] = (
        entropy_boost_factor_cfg if entropy_boost_factor_cfg is not None else 1.5
    )
    params["entropy_boost_cap"] = entropy_boost_cap_cfg
    params["clip_range_warmup"] = (
        clip_range_warmup_cfg if clip_range_warmup_cfg is not None else max(params["clip_range"], 0.18)
    )
    params["clip_range_warmup_updates"] = (
        clip_range_warmup_updates_cfg if clip_range_warmup_updates_cfg is not None else 12
    )
    params["critic_grad_warmup_updates"] = (
        critic_grad_warmup_updates_cfg if critic_grad_warmup_updates_cfg is not None else 12
    )
    params["cvar_activation_threshold"] = (
        cvar_activation_threshold_cfg if cvar_activation_threshold_cfg is not None else 0.25
    )
    params["cvar_activation_hysteresis"] = (
        cvar_activation_hysteresis_cfg if cvar_activation_hysteresis_cfg is not None else 0.05
    )
    params["cvar_ramp_updates"] = (
        cvar_ramp_updates_cfg if cvar_ramp_updates_cfg is not None else 6
    )

    if kl_exceed_stop_fraction_override is not None:
        kl_exceed_stop_fraction_cfg = kl_exceed_stop_fraction_override
    if kl_absolute_stop_factor_override is not None:
        kl_absolute_stop_factor_cfg = kl_absolute_stop_factor_override

    kl_early_stop_value = (
        True if kl_early_stop_cfg is None else bool(kl_early_stop_cfg)
    )
    params["kl_early_stop"] = kl_early_stop_value

    if kl_exceed_stop_fraction_cfg is not None:
        kl_exceed_stop_fraction_value = float(
            np.clip(kl_exceed_stop_fraction_cfg, 0.0, 1.0)
        )
    elif kl_early_stop_value:
        kl_exceed_stop_fraction_value = 0.30
    else:
        kl_exceed_stop_fraction_value = 0.0
    params["kl_exceed_stop_fraction"] = kl_exceed_stop_fraction_value

    kl_early_stop_use_ema_value = True if kl_early_stop_use_ema_cfg is None else bool(kl_early_stop_use_ema_cfg)
    if kl_ema_updates_cfg is None:
        kl_ema_updates_value = 10
    else:
        kl_ema_updates_value = max(1, int(kl_ema_updates_cfg))
    kl_ema_alpha_value = float(kl_ema_alpha_cfg) if kl_ema_alpha_cfg is not None else None
    if kl_consec_minibatches_cfg is None:
        kl_consec_minibatches_value = 3
    else:
        kl_consec_minibatches_value = max(0, int(kl_consec_minibatches_cfg))

    params["kl_early_stop_use_ema"] = kl_early_stop_use_ema_value
    params["kl_ema_updates"] = kl_ema_updates_value
    params["kl_ema_alpha"] = kl_ema_alpha_value
    params["kl_consec_minibatches"] = kl_consec_minibatches_value

    if (
        kl_absolute_stop_factor_cfg is not None
        and math.isfinite(kl_absolute_stop_factor_cfg)
        and kl_absolute_stop_factor_cfg > 0.0
    ):
        kl_absolute_stop_factor_value: Optional[float] = float(kl_absolute_stop_factor_cfg)
    elif kl_early_stop_value:
        kl_absolute_stop_factor_value = 2.5
    else:
        kl_absolute_stop_factor_value = None

    params["kl_absolute_stop_factor"] = kl_absolute_stop_factor_value

    params.setdefault("target_kl", 0.08)
    params.setdefault("kl_early_stop", True)
    params.setdefault(
        "kl_exceed_stop_fraction",
        0.30 if params.get("kl_early_stop", True) else 0.0,
    )
    params.setdefault("kl_early_stop_use_ema", True)
    params.setdefault("kl_ema_updates", 10)
    params.setdefault("kl_ema_alpha", None)
    params.setdefault("kl_consec_minibatches", 3)
    params.setdefault("kl_absolute_stop_factor", 2.5)
    params.setdefault("kl_penalty_beta_init", 0.01)
    params.setdefault("kl_penalty_beta_min", 1e-4)
    params.setdefault("kl_penalty_beta_max", 1.0)
    params.setdefault("kl_penalty_pid_kp", 0.5)
    params.setdefault("kl_penalty_pid_ki", 0.05)
    params.setdefault("kl_penalty_pid_kd", 0.10)

    params["microbatch_size"] = (
        microbatch_size_cfg if microbatch_size_cfg is not None else params["batch_size"]
    )
    params["seed"] = seed_cfg if seed_cfg is not None else 20240518

    if params["ent_coef_final"] is None:
        params["ent_coef_final"] = params["ent_coef"]

    if trade_frequency_penalty_cfg is not None:
        params["trade_frequency_penalty"] = trade_frequency_penalty_cfg
    else:
        params["trade_frequency_penalty"] = trial.suggest_float(
            "trade_frequency_penalty", 1e-5, 5e-4, log=True
        )

    params["turnover_penalty_coef"] = turnover_penalty_coef_cfg
    params["reward_return_clip"] = float(max(reward_return_clip_cfg, 1e-6))
    params["turnover_norm_cap"] = float(max(turnover_norm_cap_cfg, 1e-12))
    params["reward_cap"] = float(max(reward_cap_cfg, 1e-6))

    cql_alpha_cfg = _coerce_optional_float(_get_model_param_value(cfg, "cql_alpha"), "cql_alpha")
    if cql_alpha_cfg is not None:
        params["cql_alpha"] = cql_alpha_cfg
    else:
        params["cql_alpha"] = trial.suggest_float("cql_alpha", 0.1, 10.0, log=True)

    cql_beta_cfg = _coerce_optional_float(_get_model_param_value(cfg, "cql_beta"), "cql_beta")
    if cql_beta_cfg is not None:
        params["cql_beta"] = cql_beta_cfg
    else:
        params["cql_beta"] = trial.suggest_float("cql_beta", 1.0, 10.0)

    cvar_alpha_cfg = _coerce_optional_float(_get_model_param_value(cfg, "cvar_alpha"), "cvar_alpha")
    if cvar_alpha_cfg is not None:
        params["cvar_alpha"] = cvar_alpha_cfg
    else:
        params["cvar_alpha"] = trial.suggest_float("cvar_alpha", 0.01, 0.20)

    cvar_weight_cfg = _coerce_optional_float(_get_model_param_value(cfg, "cvar_weight"), "cvar_weight")
    if cvar_weight_cfg is not None:
        params["cvar_weight"] = cvar_weight_cfg
    else:
        params["cvar_weight"] = trial.suggest_float("cvar_weight", 0.1, 2.0, log=True)

    cvar_cap_cfg = _coerce_optional_float(_get_model_param_value(cfg, "cvar_cap"), "cvar_cap")
    if cvar_cap_cfg is not None:
        params["cvar_cap"] = cvar_cap_cfg
    else:
        params["cvar_cap"] = trial.suggest_float("cvar_cap", 0.01, 1.0, log=True)

    def _extract_risk_cvar_section(cfg: TrainConfig) -> Dict[str, Any]:
        risk_section = getattr(cfg, "risk", None)
        candidate: Any
        if isinstance(risk_section, Mapping):
            candidate = risk_section.get("cvar")
        else:
            candidate = getattr(risk_section, "cvar", None)
        if candidate is None:
            return {}
        if isinstance(candidate, Mapping):
            return dict(candidate)
        payload: Dict[str, Any] = {}
        for key in (
            "use_constraint",
            "alpha",
            "limit",
            "limit_pct",
            "lambda_lr",
            "use_penalty",
            "penalty_cap",
            "weight",
            "cap",
            "winsor_pct",
            "ema_beta",
        ):
            if hasattr(candidate, key):
                payload[key] = getattr(candidate, key)
        dict_method = getattr(candidate, "dict", None)
        if callable(dict_method):
            try:
                payload.update(dict_method(exclude_none=True))
            except Exception:
                pass
        return payload

    risk_cvar_cfg = _extract_risk_cvar_section(cfg)
    if "alpha" in risk_cvar_cfg and risk_cvar_cfg["alpha"] is not None:
        params["cvar_alpha"] = float(risk_cvar_cfg["alpha"])
    params["cvar_use_constraint"] = bool(risk_cvar_cfg.get("use_constraint", False))
    limit_fraction_value = risk_cvar_cfg.get("limit_pct")
    if limit_fraction_value is not None:
        try:
            limit_fraction_value = float(limit_fraction_value)
        except (TypeError, ValueError):
            limit_fraction_value = None
        else:
            if abs(limit_fraction_value) > 1.0:
                limit_fraction_value /= 100.0
    if limit_fraction_value is None:
        limit_raw_value = risk_cvar_cfg.get("limit")
        if limit_raw_value is not None:
            try:
                candidate_float = float(limit_raw_value)
            except (TypeError, ValueError):
                candidate_float = None
            if candidate_float is not None:
                limit_fraction_value = candidate_float
                if abs(limit_fraction_value) > 1.0:
                    limit_fraction_value /= 100.0
    params["cvar_limit"] = float(
        limit_fraction_value if limit_fraction_value is not None else -0.02
    )
    params["cvar_lambda_lr"] = max(float(risk_cvar_cfg.get("lambda_lr", 1e-2)), 0.0)
    params["cvar_use_penalty"] = bool(risk_cvar_cfg.get("use_penalty", True))
    params["cvar_penalty_cap"] = max(float(risk_cvar_cfg.get("penalty_cap", 0.7)), 0.0)
    if "weight" in risk_cvar_cfg and risk_cvar_cfg["weight"] is not None:
        params["cvar_weight"] = float(risk_cvar_cfg["weight"])
    if "cap" in risk_cvar_cfg and risk_cvar_cfg["cap"] is not None:
        params["cvar_cap"] = float(risk_cvar_cfg["cap"])
    winsor_cfg_value = risk_cvar_cfg.get("winsor_pct", 0.1)
    try:
        winsor_pct_value = float(winsor_cfg_value)
    except (TypeError, ValueError):
        winsor_pct_value = 0.1
    if winsor_pct_value <= 0.005:
        # Backward compatibility for legacy fraction inputs (~0.1% => 0.001)
        winsor_pct_value *= 100.0
    winsor_pct_value = max(winsor_pct_value, 0.0)
    if winsor_pct_value > 50.0:
        winsor_pct_value = 50.0
    params["cvar_winsor_pct"] = winsor_pct_value
    ema_beta_cfg = risk_cvar_cfg.get("ema_beta", 0.9)
    params["cvar_ema_beta"] = float(ema_beta_cfg if ema_beta_cfg is not None else 0.9)
    # 1. Определяем окно самого "медленного" индикатора на основе статических параметров.
    #    Эти параметры передаются в C++ симулятор.
    #    (В данном проекте они жестко заданы в TradingEnv, но для надежности берем их из HPO)
    # 1. Определяем окно самого "медленного" индикатора на основе констант.
    #    Это гарантирует синхронизацию с параметрами среды по умолчанию.
    slowest_window = max(
        params["window_size"],
        MA20_WINDOW,
        MACD_SLOW,
        ATR_WINDOW,
        RSI_WINDOW,
        CCI_WINDOW,
        BB_WINDOW,
        OBV_MA_WINDOW
    )
    
    # 2. Устанавливаем период прогрева с запасом (рекомендуется x2).
    #    Это гарантирует полную стабилизацию даже для вложенных сглаживаний,
    #    как в сигнальной линии MACD.
    warmup_period = slowest_window * 2

    num_atoms_cfg = _coerce_optional_int(_get_model_param_value(cfg, "num_atoms"), "num_atoms")
    v_min_cfg = _coerce_optional_float(_get_model_param_value(cfg, "v_min"), "v_min")
    v_max_cfg = _coerce_optional_float(_get_model_param_value(cfg, "v_max"), "v_max")

    num_atoms = num_atoms_cfg if num_atoms_cfg is not None else 51
    v_min = v_min_cfg if v_min_cfg is not None else -10.0
    v_max = v_max_cfg if v_max_cfg is not None else 10.0

    if num_atoms < 1:
        raise ValueError("Invalid configuration: 'num_atoms' must be >= 1 in cfg.model.params")
    if v_max <= v_min:
        raise ValueError(
            "Invalid configuration: 'v_max' must be greater than 'v_min' in cfg.model.params"
        )

    print(f"[cfg override] value head: num_atoms={num_atoms}, v_min={v_min}, v_max={v_max}")

    policy_arch_params = {
        "hidden_dim": params["hidden_dim"],
        "use_memory": True,
        "num_atoms": num_atoms,
        "v_min": v_min,
        "v_max": v_max,
    }

    critic_arch_params = dict(policy_arch_params.get("critic") or {})
    critic_arch_params.update(
        {
            "distributional": True,
            "num_quantiles": 32,
            "huber_kappa": 1.0,
        }
    )
    policy_arch_params["critic"] = critic_arch_params

    execution_mode: str | None = None
    execution_cfg = getattr(cfg, "execution", None)
    if execution_cfg is not None:
        execution_mode = getattr(execution_cfg, "mode", None)
        if execution_mode is None and isinstance(execution_cfg, Mapping):
            execution_mode = execution_cfg.get("mode")  # type: ignore[index]
    if execution_mode is None:
        execution_blob = getattr(cfg, "__dict__", {}).get("execution")
        if isinstance(execution_blob, Mapping):
            execution_mode = execution_blob.get("mode")
    normalized_exec_mode = str(execution_mode or "").strip().lower()
    if normalized_exec_mode != "bar":
        raise RuntimeError(
            "Score-based policy requires execution.mode='bar' in configuration"
        )
    policy_arch_params["execution_mode"] = normalized_exec_mode

    if not train_data_by_token: raise ValueError("Нет данных для обучения в этом trial.")

    n_envs_cfg = _coerce_optional_int(_get_model_param_value(cfg, "n_envs"), "n_envs")
    n_envs = 8
    if n_envs_cfg is not None:
        n_envs = n_envs_cfg
        print(f"[cfg override] using n_envs={n_envs_cfg} from cfg.model.params")
    if n_envs_override is not None:
        if n_envs_override < 1:
            raise ValueError("Override for n_envs must be >= 1")
        n_envs = n_envs_override
        print(f"[arg override] using n_envs={n_envs_override}")

    try:
        cpu_count = os.cpu_count() or 1
    except Exception:
        cpu_count = 1
    if n_envs > cpu_count:
        print(f"Requested {n_envs} envs exceeds available CPU cores ({cpu_count}); capping to {cpu_count}.")
        n_envs = cpu_count
    if n_envs < 1:
        raise ValueError("Computed n_envs must be at least 1")

    print(f"Запускаем {n_envs} параллельных сред...")

    total_batch_size = params["n_steps"] * n_envs
    batch_size = params["batch_size"]
    if batch_size <= 0:
        raise ValueError("Invalid configuration: 'batch_size' must be a positive integer in cfg.model.params")
    if batch_size > total_batch_size:
        raise ValueError(
            "Invalid configuration: 'batch_size' cannot exceed n_steps * num_envs in cfg.model.params"
        )
    if total_batch_size % batch_size != 0:
        raise ValueError(
            "Invalid configuration: 'batch_size' must evenly divide n_steps * num_envs in cfg.model.params"
        )
    microbatch_size = params["microbatch_size"]
    if microbatch_size <= 0:
        raise ValueError(
            "Invalid configuration: 'microbatch_size' must be a positive integer in cfg.model.params"
        )
    if batch_size % microbatch_size != 0:
        raise ValueError(
            "Invalid configuration: 'microbatch_size' must evenly divide batch_size in cfg.model.params"
        )
    base_seed = int(params["seed"])
    os.environ["PYTHONHASHSEED"] = str(base_seed)
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except AttributeError:
        pass
    leak_guard_train = LeakGuard(LeakConfig(**leak_guard_kwargs))
    leak_guard_val = LeakGuard(LeakConfig(**leak_guard_kwargs))

    train_symbol_items = sorted(train_data_by_token.items())
    if not train_symbol_items:
        raise ValueError("Нет тренировочных символов для создания сред.")

    reward_robust_clip_fraction_value: Optional[float] = None

    def make_env_train(rank: int):
        symbol_idx = rank % len(train_symbol_items)
        symbol, df = train_symbol_items[symbol_idx]

        def _init():
            nonlocal reward_robust_clip_fraction_value
            # Создаем детерминированный seed для каждого воркера на основе базового seed
            unique_seed = base_seed + rank
            env_params = {
                # 1. Параметры, которые подбирает Optuna
                "risk_aversion_drawdown": params["risk_aversion_drawdown"],
                "risk_aversion_variance": params["risk_aversion_variance"],
                "atr_multiplier": params["atr_multiplier"],
                "trailing_atr_mult": params["trailing_atr_mult"],
                "tp_atr_mult": params["tp_atr_mult"],
                "window_size": params.get("window_size", 20),
                "gamma": params["gamma"],
                "trade_frequency_penalty": params["trade_frequency_penalty"],
                "turnover_penalty_coef": params["turnover_penalty_coef"],
                "reward_return_clip": params["reward_return_clip"],
                "turnover_norm_cap": params["turnover_norm_cap"],
                "reward_cap": params["reward_cap"],
                "reward_clip": reward_clip_params,
                "momentum_factor": params["momentum_factor"],
                "mean_reversion_factor": params["mean_reversion_factor"],
                "adversarial_factor": params["adversarial_factor"],

                # 2. Данные, которые передаются в функцию objective
                "norm_stats": norm_stats,

                # 3. Статические параметры и параметры индикаторов
                "mode": "train",
                "reward_shaping": False,
                "warmup_period": warmup_period,
                "ma5_window": MA5_WINDOW,
                "ma20_window": MA20_WINDOW,
                "atr_window": ATR_WINDOW,
                "rsi_window": RSI_WINDOW,
                "macd_fast": MACD_FAST,
                "macd_slow": MACD_SLOW,
                "macd_signal": MACD_SIGNAL,
                "momentum_window": MOMENTUM_WINDOW,
                "cci_window": CCI_WINDOW,
                "bb_window": BB_WINDOW,
                "obv_ma_window": OBV_MA_WINDOW,

            }
            env_params.update(sim_config)
            env_params.update(timing_env_kwargs)
            env_params.update(env_runtime_overrides)

            # Создаем и возвращаем экземпляр среды
            if (
                reward_robust_clip_fraction_value is not None
                and math.isfinite(reward_robust_clip_fraction_value)
                and reward_robust_clip_fraction_value > 0.0
            ):
                env_params["reward_robust_clip_fraction"] = (
                    reward_robust_clip_fraction_value
                )

            env = TradingEnv(
                df,
                **env_params,
                leak_guard=leak_guard_train,
                seed=unique_seed,
            )
            if reward_robust_clip_fraction_value is None:
                candidate_fraction = float(
                    getattr(env, "reward_robust_clip_fraction", 0.0)
                )
                if math.isfinite(candidate_fraction) and candidate_fraction > 0.0:
                    reward_robust_clip_fraction_value = candidate_fraction
            elif hasattr(env, "reward_robust_clip_fraction"):
                env.reward_robust_clip_fraction = float(
                    reward_robust_clip_fraction_value
                )
            setattr(env, "selected_symbol", symbol)
            env = _wrap_action_space_if_needed(
                env,
                bins_vol=bins_vol,
                action_overrides=action_overrides,
                long_only=long_only_flag,
            )
            return env
        return _init

    env_constructors = [make_env_train(rank=i) for i in range(n_envs)]
    train_stats_path = trials_dir / f"vec_normalize_train_{trial.number}.pkl"
    os.makedirs(trials_dir, exist_ok=True)

    base_env_tr = WatchdogVecEnv(env_constructors)
    monitored_env_tr = VecMonitor(base_env_tr)
    env_tr = VecNormalize(
        monitored_env_tr,
        training=True,
        norm_obs=False,
        norm_reward=False,
        clip_reward=None,
        gamma=params["gamma"],
    )

    # Distributional PPO expects access to the raw ΔPnL rewards in order to
    # compute its custom targets.  If VecNormalize were to normalise rewards the
    # algorithm would raise during rollout collection (see
    # ``DistributionalPPO.collect_rollouts``) because the rescaled rewards break
    # the interpretation of the categorical distribution.  Explicitly disabling
    # reward normalisation keeps the environment aligned with that expectation
    # and prevents the training crash observed when Optuna launches trials.
    env_tr.norm_reward = False

    if popart_holdout_loader is not None:
        attach_env = getattr(popart_holdout_loader, "attach_env", None)
        if callable(attach_env):
            attach_env(env_tr)
        ensure_ready = getattr(popart_holdout_loader, "ensure_materialized", None)
        if callable(ensure_ready):
            ensure_ready()

    env_tr.save(str(train_stats_path))
    save_sidecar_metadata(str(train_stats_path), extra={"kind": "vecnorm_stats", "phase": "train"})

    val_symbol_items = sorted(val_data_by_token.items())
    if not val_symbol_items:
        raise ValueError("Нет валидационных символов для создания сред.")

    def _make_val_env_factory(symbol: str, df: pd.DataFrame):
        env_val_params = {
            "norm_stats": norm_stats,
            "window_size": params["window_size"],
            "gamma": params["gamma"],
            "atr_multiplier": params["atr_multiplier"],
            "trailing_atr_mult": params["trailing_atr_mult"],
            "tp_atr_mult": params["tp_atr_mult"],
            "trade_frequency_penalty": params["trade_frequency_penalty"],
            "turnover_penalty_coef": params["turnover_penalty_coef"],
            "reward_return_clip": params["reward_return_clip"],
            "turnover_norm_cap": params["turnover_norm_cap"],
            "reward_cap": params["reward_cap"],
            "reward_clip": reward_clip_params,
            "mode": "val",
            "reward_shaping": False,
            "warmup_period": warmup_period,
            "ma5_window": MA5_WINDOW,
            "ma20_window": MA20_WINDOW,
            "atr_window": ATR_WINDOW,
            "rsi_window": RSI_WINDOW,
            "macd_fast": MACD_FAST,
            "macd_slow": MACD_SLOW,
            "macd_signal": MACD_SIGNAL,
            "momentum_window": MOMENTUM_WINDOW,
            "cci_window": CCI_WINDOW,
            "bb_window": BB_WINDOW,
            "obv_ma_window": OBV_MA_WINDOW
        }
        env_val_params.update(sim_config)
        env_val_params.update(timing_env_kwargs)
        env_val_params.update(env_runtime_overrides)
        if (
            reward_robust_clip_fraction_value is not None
            and math.isfinite(reward_robust_clip_fraction_value)
            and reward_robust_clip_fraction_value > 0.0
        ):
            env_val_params["reward_robust_clip_fraction"] = (
                reward_robust_clip_fraction_value
            )

        env = TradingEnv(
            df,
            **env_val_params,
            leak_guard=leak_guard_val,
            seed=base_seed,
        )
        setattr(env, "selected_symbol", symbol)
        env = _wrap_action_space_if_needed(
            env,
            bins_vol=bins_vol,
            action_overrides=action_overrides,
            long_only=long_only_flag,
        )
        return env

    val_env_fns = [
        lambda symbol=symbol, df=df: _make_val_env_factory(symbol, df)
        for symbol, df in val_symbol_items
    ]
    monitored_env_va = VecMonitor(DummyVecEnv(val_env_fns))
    check_model_compat(str(train_stats_path))
    env_va = _freeze_vecnormalize(VecNormalize.load(str(train_stats_path), monitored_env_va))
    env_va.norm_reward = False
    env_va.clip_reward = None

    val_stats_path = trials_dir / f"vec_normalize_val_{trial.number}.pkl"
    env_va.save(str(val_stats_path))
    save_sidecar_metadata(str(val_stats_path), extra={"kind": "vecnorm_stats", "phase": "val"})

    policy_kwargs = {
        "arch_params": policy_arch_params,
        "optimizer_class": torch.optim.AdamW,
        "optimizer_kwargs": {"weight_decay": params["weight_decay"]},
    }
    # --- Stabilise KL behaviour and optimiser updates -----------------------
    clip_range_value = params.get("clip_range", 0.1)
    clip_range_value = float(clip_range_value)
    clip_range_value = float(np.clip(clip_range_value, 0.08, 0.12))
    params["clip_range"] = clip_range_value

    target_kl_value = params.get("target_kl")
    if target_kl_value is not None:
        target_kl_value = float(target_kl_value)
        params["target_kl"] = float(np.clip(target_kl_value, 0.02, 1.6))

    n_epochs_value = int(params.get("n_epochs", 2))
    params["n_epochs"] = max(min(n_epochs_value, 4), 2)

    # Рассчитываем, сколько раз будет собран полный буфер данных (rollout)

    kl_epoch_decay_value = params.get("kl_epoch_decay", 0.5)
    if isinstance(kl_epoch_decay_value, bool):
        kl_epoch_decay_value = 0.5
    kl_epoch_decay_value = float(kl_epoch_decay_value)
    if not (0.0 < kl_epoch_decay_value <= 1.0):
        kl_epoch_decay_value = 0.5
    params["kl_epoch_decay"] = kl_epoch_decay_value

    try:
        beta_min_value = float(params.get("kl_penalty_beta_min", 1e-4))
    except (TypeError, ValueError):
        beta_min_value = 1e-4
    if not math.isfinite(beta_min_value) or beta_min_value < 0.0:
        beta_min_value = 1e-4

    try:
        beta_max_value = float(params.get("kl_penalty_beta_max", 1.0))
    except (TypeError, ValueError):
        beta_max_value = 1.0
    if not math.isfinite(beta_max_value) or beta_max_value <= 0.0:
        beta_max_value = 1.0
    if beta_max_value < beta_min_value:
        beta_max_value = beta_min_value

    try:
        beta_init_value = float(params.get("kl_penalty_beta_init", 0.01))
    except (TypeError, ValueError):
        beta_init_value = 0.01
    if not math.isfinite(beta_init_value):
        beta_init_value = 0.01
    beta_init_value = min(max(beta_init_value, beta_min_value), beta_max_value)

    def _sanitize_pid(value: Any, default: float) -> float:
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            coerced = default
        if not math.isfinite(coerced) or coerced < 0.0:
            coerced = default
        return coerced

    kp_value = _sanitize_pid(params.get("kl_penalty_pid_kp", 0.5), 0.5)
    ki_value = _sanitize_pid(params.get("kl_penalty_pid_ki", 0.05), 0.05)
    kd_value = _sanitize_pid(params.get("kl_penalty_pid_kd", 0.10), 0.10)

    penalty_type_value = params.get("kl_penalty_type", "adaptive_beta")
    if penalty_type_value is None:
        penalty_type_value = "adaptive_beta"
    else:
        penalty_type_value = str(penalty_type_value)

    params["kl_penalty_beta_min"] = beta_min_value
    params["kl_penalty_beta_max"] = beta_max_value
    params["kl_penalty_beta_init"] = beta_init_value
    params["kl_penalty_pid_kp"] = kp_value
    params["kl_penalty_pid_ki"] = ki_value
    params["kl_penalty_pid_kd"] = kd_value
    params["kl_penalty_type"] = penalty_type_value
    params["kl_use_lr_decay_requested"] = bool(params.get("kl_use_lr_decay_requested", False))

    num_rollouts = math.ceil(total_timesteps / (params["n_steps"] * n_envs))
    
    # Рассчитываем, на сколько мини-батчей делится каждый роллаут
    num_minibatches_per_rollout = total_batch_size // batch_size
    
    # Итоговое количество шагов оптимизатора за все обучение
    total_optimizer_steps = num_rollouts * params["n_epochs"] * num_minibatches_per_rollout

    optimization_cfg = getattr(cfg, "optimization", None)
    scheduler_cfg = None
    if isinstance(optimization_cfg, Mapping):
        scheduler_cfg = optimization_cfg.get("scheduler")
    elif optimization_cfg is not None:
        scheduler_cfg = getattr(optimization_cfg, "scheduler", None)

    scheduler_enabled_value = True
    if isinstance(scheduler_cfg, Mapping):
        scheduler_enabled_value = scheduler_cfg.get("enabled", True)
    elif scheduler_cfg is not None:
        scheduler_enabled_value = getattr(scheduler_cfg, "enabled", True)

    if isinstance(scheduler_enabled_value, str):
        scheduler_enabled = scheduler_enabled_value.strip().lower() not in {"0", "false", "no", "off"}
    elif scheduler_enabled_value is None:
        scheduler_enabled = True
    else:
        scheduler_enabled = bool(scheduler_enabled_value)

    if scheduler_enabled:
        # Создаем lambda-функцию для планировщика
        # SB3 вызовет ее со своим внутренним, правильным оптимизатором
        def scheduler_fn(optimizer):
            return OneCycleLR(
                optimizer=optimizer,
                max_lr=params["learning_rate"] * 3,
                total_steps=total_optimizer_steps,
            )

        # Оборачиваем ее в словарь для передачи в policy_kwargs
        policy_kwargs["optimizer_scheduler_fn"] = scheduler_fn
    DistributionalPPO = _get_distributional_ppo()

    use_torch_compile_cfg = _get_model_param_value(cfg, "use_torch_compile")
    if isinstance(use_torch_compile_cfg, str):
        use_torch_compile = use_torch_compile_cfg.strip().lower() in {"1", "true", "yes", "on"}
    elif use_torch_compile_cfg is None:
        use_torch_compile = False
    else:
        use_torch_compile = bool(use_torch_compile_cfg)

    tb_log_path: Path | None = None
    if tensorboard_log_dir is not None:
        tb_log_path = tensorboard_log_dir / f"trial_{trial.number:03d}"
        tb_log_path.mkdir(parents=True, exist_ok=True)

    loss_head_weights, policy_include_heads = _extract_loss_head_config(cfg)
    if policy_include_heads is not None:
        policy_arch_params.setdefault("include_heads", policy_include_heads)

    assert "kl_exceed_stop_fraction" in params, "Missing KL exceed stop fraction parameter"

    value_scale_kwargs: dict[str, Any] = {
        "ema_beta": params["value_scale_ema_beta"],
        "max_rel_step": params["value_scale_max_rel_step"],
        "std_floor": params["value_scale_std_floor"],
        "window_updates": params["value_scale_window_updates"],
        "warmup_updates": params["value_scale_warmup_updates"],
        "freeze_after": params["value_scale_freeze_after"],
        "range_max_rel_step": params["value_scale_range_max_rel_step"],
    }
    if params["value_scale_freeze_after_updates"] is not None:
        value_scale_kwargs["freeze_after_updates"] = params[
            "value_scale_freeze_after_updates"
        ]
    if params["value_scale_never_freeze"] is not None:
        value_scale_kwargs["never_freeze"] = params["value_scale_never_freeze"]
    if params["value_scale_target_ema_beta"] is not None:
        value_scale_kwargs["target_scale_ema_beta"] = params[
            "value_scale_target_ema_beta"
        ]
    if params["value_scale_max_change_pct"] is not None:
        value_scale_kwargs["max_change_pct"] = params["value_scale_max_change_pct"]
    stability_params = params["value_scale_stability"]
    if stability_params:
        value_scale_kwargs["stability"] = stability_params
    if params["value_scale_stability_patience"] is not None:
        value_scale_kwargs["stability_patience"] = params[
            "value_scale_stability_patience"
        ]

    model = DistributionalPPO(
        use_torch_compile=use_torch_compile,
        v_range_ema_alpha=params["v_range_ema_alpha"],
        policy=CustomActorCriticPolicy,
        env=env_tr,
        cql_alpha=0.0,
        cql_beta=params["cql_beta"],
        cvar_alpha=params["cvar_alpha"],
        vf_coef=params["vf_coef"],
        vf_coef_warmup=params["vf_coef_warmup"],
        vf_coef_warmup_updates=int(params["vf_coef_warmup_updates"]),
        vf_bad_explained_scale=params["vf_bad_explained_scale"],
        vf_bad_explained_floor=params["vf_bad_explained_floor"],
        bad_explained_patience=int(params["bad_explained_patience"]),
        entropy_boost_factor=params["entropy_boost_factor"],
        entropy_boost_cap=params["entropy_boost_cap"],
        cvar_weight=params["cvar_weight"],
        cvar_use_constraint=params["cvar_use_constraint"],
        cvar_limit=params["cvar_limit"],
        cvar_lambda_lr=params["cvar_lambda_lr"],
        cvar_use_penalty=params["cvar_use_penalty"],
        cvar_penalty_cap=params["cvar_penalty_cap"],
        cvar_winsor_pct=params["cvar_winsor_pct"],
        cvar_ema_beta=params["cvar_ema_beta"],
        value_scale=value_scale_kwargs,
        value_scale_controller=params.get("value_scale_controller"),
        value_scale_controller_holdout=popart_holdout_loader,
        clip_range_vf=params["clip_range_vf"],
        value_scale_update_enabled=params["value_scale_update_enabled"],
        value_target_scale_fixed=params["value_target_scale_fixed"],
        normalize_returns=params["normalize_returns"],

        bc_warmup_steps=params["bc_warmup_steps"],
        bc_decay_steps=params["bc_decay_steps"],
        bc_final_coef=0.0,
        ent_coef_final=params["ent_coef_final"],
        ent_coef_min=params["ent_coef_min"],
        ent_coef_decay_steps=params["ent_coef_decay_steps"],
        ent_coef_plateau_window=params["ent_coef_plateau_window"],
        ent_coef_plateau_tolerance=params["ent_coef_plateau_tolerance"],
        ent_coef_plateau_min_updates=params["ent_coef_plateau_min_updates"],

        cvar_cap=params["cvar_cap"],
        cvar_activation_threshold=params["cvar_activation_threshold"],
        cvar_activation_hysteresis=params["cvar_activation_hysteresis"],
        cvar_ramp_updates=int(params["cvar_ramp_updates"]),
        kl_early_stop=params["kl_early_stop"],
        kl_epoch_decay=params["kl_epoch_decay"],
        kl_exceed_stop_fraction=params.get("kl_exceed_stop_fraction"),
        kl_early_stop_use_ema=params.get("kl_early_stop_use_ema", True),
        kl_ema_updates=int(params.get("kl_ema_updates", 10)),
        kl_ema_alpha=params.get("kl_ema_alpha"),
        kl_consec_minibatches=int(params.get("kl_consec_minibatches", 3)),
        kl_absolute_stop_factor=params.get("kl_absolute_stop_factor"),
        kl_penalty_beta=params.get("kl_penalty_beta_init", 0.01),
        kl_penalty_beta_min=params.get("kl_penalty_beta_min", 1e-4),
        kl_penalty_beta_max=params.get("kl_penalty_beta_max", 1.0),
        kl_penalty_pid_kp=params.get("kl_penalty_pid_kp", 0.5),
        kl_penalty_pid_ki=params.get("kl_penalty_pid_ki", 0.05),
        kl_penalty_pid_kd=params.get("kl_penalty_pid_kd", 0.10),

        learning_rate=params["learning_rate"],
        optimizer_lr_min=params["optimizer_lr_min"],
        scheduler_min_lr=params["scheduler_min_lr"],
        optimizer_lr_max=params["optimizer_lr_max"],
        n_steps=params["n_steps"],
        n_epochs=params["n_epochs"],
        batch_size=params["batch_size"],
        microbatch_size=params["microbatch_size"],
        ent_coef=params["ent_coef"],
        gamma=params["gamma"],
        gae_lambda=params["gae_lambda"],
        ppo_clip_range=params["clip_range"],
        clip_range_warmup=params["clip_range_warmup"],
        clip_range_warmup_updates=int(params["clip_range_warmup_updates"]),
        critic_grad_warmup_updates=int(params["critic_grad_warmup_updates"]),
        max_grad_norm=params["max_grad_norm"],
        target_kl=params.get("target_kl", 0.08),
        seed=params["seed"],
        loss_head_weights=loss_head_weights,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tb_log_path) if tb_log_path is not None else None,
        verbose=1
    )

    _ensure_model_popart_holdout_loader(
        model,
        popart_holdout_loader,
        value_scale_controller_cfg,
    )

    controller_enabled = False
    if isinstance(value_scale_controller_cfg, Mapping):
        controller_enabled = bool(value_scale_controller_cfg.get("enabled", False))
    elif value_scale_controller_cfg is not None:
        getter = getattr(value_scale_controller_cfg, "get", None)
        if callable(getter):
            try:
                controller_enabled = bool(getter("enabled", False))
            except TypeError:
                controller_enabled = bool(getter("enabled"))
        elif hasattr(value_scale_controller_cfg, "enabled"):
            controller_enabled = bool(getattr(value_scale_controller_cfg, "enabled"))
    if controller_enabled:
        assert getattr(model, "_popart_controller", None) is not None, "PopArt not initialised"

    kl_penalty_beta_logged = getattr(
        model,
        "kl_beta",
        params.get("kl_penalty_beta_init", 0.01),
    )
    kl_early_stop_logged = bool(
        getattr(model, "kl_early_stop", params.get("kl_early_stop", True))
    )
    target_kl_logged = getattr(model, "target_kl", params.get("target_kl", 0.08))
    if target_kl_logged is None:
        target_kl_logged = params.get("target_kl", 0.08)
    kl_exceed_stop_fraction_default = params.get(
        "kl_exceed_stop_fraction",
        0.30 if params.get("kl_early_stop", True) else 0.0,
    )
    kl_exceed_stop_fraction_logged = getattr(
        model,
        "kl_exceed_stop_fraction",
        kl_exceed_stop_fraction_default,
    )
    kl_absolute_stop_factor_default = params.get(
        "kl_absolute_stop_factor",
        2.5 if params.get("kl_early_stop", True) else None,
    )
    kl_absolute_stop_factor_logged = getattr(
        model,
        "kl_absolute_stop_factor",
        kl_absolute_stop_factor_default,
    )
    logger = getattr(model, "logger", None) or getattr(model, "_logger", None)
    if logger is not None:
        logger.record("config/target_kl", float(target_kl_logged))
        logger.record(
            "config/kl_exceed_stop_fraction", float(kl_exceed_stop_fraction_logged)
        )
        if kl_absolute_stop_factor_logged is not None:
            logger.record(
                "config/kl_absolute_stop_factor",
                float(kl_absolute_stop_factor_logged),
            )
        else:
            logger.record("config/kl_absolute_stop_factor", 0.0)
        logger.record("config/kl_early_stop", 1.0 if kl_early_stop_logged else 0.0)
        logger.record(
            "config/kl_use_ema", 1.0 if params.get("kl_early_stop_use_ema", True) else 0.0
        )
        logger.record("config/kl_ema_updates", float(params.get("kl_ema_updates", 10)))
        if params.get("kl_ema_alpha") is not None:
            logger.record("config/kl_ema_alpha", float(params["kl_ema_alpha"]))
        logger.record(
            "config/kl_consec_minibatches", float(params.get("kl_consec_minibatches", 3))
        )
        logger.record("config/kl_penalty_beta_init", float(params["kl_penalty_beta_init"]))
        logger.record("config/kl_penalty_beta_min", float(params["kl_penalty_beta_min"]))
        logger.record("config/kl_penalty_beta_max", float(params["kl_penalty_beta_max"]))
        logger.record("config/kl_penalty_pid_kp", float(params["kl_penalty_pid_kp"]))
        logger.record("config/kl_penalty_pid_ki", float(params["kl_penalty_pid_ki"]))
        logger.record("config/kl_penalty_pid_kd", float(params["kl_penalty_pid_kd"]))
        logger.record("config/kl_penalty_beta", float(kl_penalty_beta_logged))
        logger.record(
            "config/kl_penalty_type",
            params.get("kl_penalty_type", "adaptive_beta"),
        )
        logger.record(
            "config/kl_use_lr_decay_requested",
            1.0 if params.get("kl_use_lr_decay_requested", False) else 0.0,
        )

    



    lr_logger = OptimizerLrLoggingCallback()
    nan_guard = NanGuardCallback()

    # Быстрый колбэк для раннего отсечения по простой метрике
    sortino_pruner = SortinoPruningCallback(trial, eval_env=env_va, eval_freq=50_000, n_eval_episodes=10)

    # Медленный, но точный колбэк для позднего отсечения по финальной метрике
    objective_pruner = ObjectiveScorePruningCallback(trial, eval_env=env_va, eval_freq=250_000, verbose=1)

    all_callbacks = [lr_logger, nan_guard, sortino_pruner, objective_pruner]

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=all_callbacks,
            progress_bar=True,
            tb_log_name=f"trial_{trial.number:03d}" if tb_log_path is not None else "run"
        )
        ev_metrics = getattr(model, "_last_ev_metrics", None)  # FIX
        if isinstance(ev_metrics, Mapping):  # FIX
            ev_mean_value = ev_metrics.get("ev_mean")  # FIX
            if ev_mean_value is not None and math.isfinite(ev_mean_value):  # FIX
                print(f"[EV] Mean explained variance across groups: {ev_mean_value:.4f}")  # FIX
                model_logger = getattr(model, "logger", None)  # FIX
                if model_logger is not None:  # FIX
                    model_logger.record("train/ev_mean_grouped_final", float(ev_mean_value))  # FIX
            grouped_metrics = ev_metrics.get("ev_grouped") if ev_metrics else None  # FIX
            if isinstance(grouped_metrics, Mapping) and grouped_metrics:  # FIX
                finite_items: list[tuple[str, float]] = []  # FIX
                for key, value in grouped_metrics.items():  # FIX
                    try:  # FIX
                        numeric = float(value)  # FIX
                    except (TypeError, ValueError):  # FIX
                        continue  # FIX
                    if not math.isfinite(numeric):  # FIX
                        continue  # FIX
                    finite_items.append((str(key), numeric))  # FIX
                if finite_items:  # FIX
                    finite_items.sort(key=lambda item: item[1])  # FIX
                    diag_limit = 5  # FIX
                    worst_items = finite_items[:diag_limit]  # FIX
                    best_items = finite_items[-diag_limit:][::-1]  # FIX
                    if worst_items:  # FIX
                        print("[EV] Worst groups:")  # FIX
                        for name, value in worst_items:  # FIX
                            print(f"    {name}: {value:.4f}")  # FIX
                            model_logger = getattr(model, "logger", None)  # FIX
                            if model_logger is not None:  # FIX
                                safe_name = name.replace("/", "_").replace(" ", "_")  # FIX
                                model_logger.record(f"train/ev_grouped/worst/{safe_name}", float(value))  # FIX
                    if best_items:  # FIX
                        print("[EV] Best groups:")  # FIX
                        for name, value in best_items:  # FIX
                            print(f"    {name}: {value:.4f}")  # FIX
                            model_logger = getattr(model, "logger", None)  # FIX
                            if model_logger is not None:  # FIX
                                safe_name = name.replace("/", "_").replace(" ", "_")  # FIX
                                model_logger.record(f"train/ev_grouped/best/{safe_name}", float(value))  # FIX
    finally:
        try:
            env_tr.training = False
            env_tr.save(str(train_stats_path))
        except Exception as exc:
            print(f"Failed to resave training VecNormalize stats: {exc}")
        env_tr.close()
        env_va.close()

    trial_model_path = trials_dir / f"trial_{trial.number}_model.zip"
    model.save(str(trial_model_path))
    value_head_meta = None
    policy_obj = getattr(model, "policy", None)
    metadata_getter = getattr(policy_obj, "value_head_metadata", None)
    if callable(metadata_getter):
        try:
            value_head_meta = metadata_getter()
        except Exception:
            value_head_meta = None
    save_sidecar_metadata(
        str(trial_model_path),
        extra={"kind": "sb3_model", "trial": int(trial.number)},
        value_head=value_head_meta,
    )

    

    print(f"<<< Trial {trial.number+1} finished training, starting unified final evaluation…")

    eval_phase_data = test_data_by_token if test_data_by_token else val_data_by_token
    eval_phase_obs = test_obs_by_token if test_data_by_token else val_obs_by_token
    eval_phase_name = "test" if test_data_by_token else "val"
    if not eval_phase_data:
        raise ValueError("No data available for validation/test evaluation. Check time split configuration.")

    # 1. Определяем все режимы для оценки
    regimes_to_evaluate = ['normal', 'choppy_flat', 'strong_trend']
    final_metrics = {}
    regime_duration = 2_500 

    # 2. Последовательно оцениваем модель в каждом режиме
    for regime in regimes_to_evaluate:
        symbol_equity_curves: list[list[float]] = []
        test_stats_path = trials_dir / f"vec_normalize_test_{trial.number}.pkl"

        for symbol, df in sorted(eval_phase_data.items()):
            def make_final_eval_env(symbol: str = symbol, df: pd.DataFrame = df):
                final_env_params = {
                    "norm_stats": norm_stats, "window_size": params["window_size"],
                    "gamma": params["gamma"], "atr_multiplier": params["atr_multiplier"],
                    "trailing_atr_mult": params["trailing_atr_mult"], "tp_atr_mult": params["tp_atr_mult"],
                    "trade_frequency_penalty": params["trade_frequency_penalty"],
                    "turnover_penalty_coef": params["turnover_penalty_coef"], "mode": eval_phase_name,
                    "reward_return_clip": params["reward_return_clip"],
                    "turnover_norm_cap": params["turnover_norm_cap"],
                    "reward_cap": params["reward_cap"],
                    "reward_clip": reward_clip_params,
                    "reward_shaping": False, "warmup_period": warmup_period,
                    "ma5_window": MA5_WINDOW, "ma20_window": MA20_WINDOW, "atr_window": ATR_WINDOW,
                    "rsi_window": RSI_WINDOW, "macd_fast": MACD_FAST, "macd_slow": MACD_SLOW,
                    "macd_signal": MACD_SIGNAL, "momentum_window": MOMENTUM_WINDOW,
                    "cci_window": CCI_WINDOW, "bb_window": BB_WINDOW, "obv_ma_window": OBV_MA_WINDOW,
                }
                final_env_params.update(sim_config)
                final_env_params.update(timing_env_kwargs)
                final_env_params.update(env_runtime_overrides)
                if (
                    reward_robust_clip_fraction_value is not None
                    and math.isfinite(reward_robust_clip_fraction_value)
                    and reward_robust_clip_fraction_value > 0.0
                ):
                    final_env_params["reward_robust_clip_fraction"] = (
                        reward_robust_clip_fraction_value
                    )
                env = TradingEnv(
                    df,
                    **final_env_params,
                    leak_guard=LeakGuard(LeakConfig(**leak_guard_kwargs)),
                )
                setattr(env, "selected_symbol", symbol)
                env = _wrap_action_space_if_needed(
                    env,
                    bins_vol=bins_vol,
                    action_overrides=action_overrides,
                    long_only=long_only_flag,
                )
                return env

            check_model_compat(str(train_stats_path))
            final_eval_norm = _freeze_vecnormalize(
                VecNormalize.load(
                    str(train_stats_path),
                    DummyVecEnv([make_final_eval_env]),
                )
            )
            final_eval_norm.norm_reward = False
            final_eval_norm.clip_reward = None
            final_eval_env = VecMonitor(final_eval_norm)

            ann_sqrt, bar_seconds = _annualization_sqrt_from_env(final_eval_env)
            _log_annualization(f"Final evaluation ({regime})", ann_sqrt, bar_seconds)
            if bar_seconds is None:
                steps_per_year = None
                print(f"[{regime}] Annualization sqrt factor: {ann_sqrt:.6f} (bar interval unknown)")
            else:
                steps_per_year = _SECONDS_PER_YEAR / float(bar_seconds)
                print(
                    f"[{regime}] Annualization sqrt factor: {ann_sqrt:.6f} "
                    f"(bar_seconds={bar_seconds:.3f}, steps_per_year={steps_per_year:.2f})"
                )
            if model.logger is not None:
                if bar_seconds is None:
                    model.logger.record(f"evaluation/{regime}_bar_seconds", float("nan"))
                    model.logger.record(f"evaluation/{regime}_steps_per_year", float("nan"))
                else:
                    model.logger.record(f"evaluation/{regime}_bar_seconds", float(bar_seconds))
                    model.logger.record(f"evaluation/{regime}_steps_per_year", float(steps_per_year))
                model.logger.record(f"evaluation/{regime}_ann_factor", float(ann_sqrt))

            if regime != 'normal':
                final_eval_env.env_method("set_market_regime", regime=regime, duration=regime_duration)

            _rewards, equity_curves = evaluate_policy_custom_cython(model, final_eval_env, num_episodes=1)
            symbol_equity_curves.extend(equity_curves)

            nt_stats = final_eval_env.env_method("get_no_trade_stats")
            if nt_stats:
                total_steps = sum(s["total_steps"] for s in nt_stats if s)
                blocked_steps = sum(s["blocked_steps"] for s in nt_stats if s)
                ratio = blocked_steps / total_steps if total_steps else 0.0
                print(
                    f"[{symbol}] No-trade blocks: {blocked_steps}/{total_steps} steps ({ratio:.2%})"
                )
                if model.logger is not None:
                    model.logger.record("eval/no_trade_ratio", float(ratio))

            final_eval_env.close()

        if not test_stats_path.exists():
            final_eval_norm.save(str(test_stats_path))
            save_sidecar_metadata(str(test_stats_path), extra={"kind": "vecnorm_stats", "phase": eval_phase_name})

        all_returns = [pd.Series(c).pct_change().dropna().to_numpy() for c in symbol_equity_curves if len(c) > 1]
        flat_returns = np.concatenate(all_returns) if all_returns else np.array([0.0])
        final_metrics[regime] = sortino_ratio(flat_returns, annualization_sqrt=ann_sqrt)

    # --- РАСЧЕТ ИТОГОВОЙ ВЗВЕШЕННОЙ МЕТРИКИ ---
    main_sortino = final_metrics.get('normal', -1.0)
    choppy_score = final_metrics.get('choppy_flat', -1.0)
    trend_score = final_metrics.get('strong_trend', -1.0)

    main_weight = 0.5
    choppy_weight = 0.3
    trend_weight = 0.2
 
    objective_score = (main_weight * main_sortino + choppy_weight * choppy_score + trend_weight * trend_score)

    if model.logger is not None:
        for regime_name, score in final_metrics.items():
            model.logger.record(f"evaluation/{regime_name}_sortino", score)
        model.logger.record("evaluation/final_objective_score", objective_score)
        model.logger.dump(step=model.num_timesteps)

    # Сохраняем все компоненты для анализа
    trial.set_user_attr("main_sortino", main_sortino)
    trial.set_user_attr("choppy_sortino", choppy_score)
    trial.set_user_attr("trend_sortino", trend_score)
    trial.set_user_attr("final_objective", objective_score)

    
    trial.set_user_attr("final_return", 0.0) # Устанавливаем в 0, т.к. корректный расчет усложнен

    print(f"\n[✅ Trial {trial.number}] COMPLETE. Final Weighted Score: {objective_score:.4f}")
    print(f"   Components -> Main Sortino: {main_sortino:.4f}, Choppy: {choppy_score:.4f}, Trend: {trend_score:.4f}\n")

    return objective_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_train.yaml", help="Path to YAML config")
    parser.add_argument(
        "--regime-config",
        default="configs/market_regimes.json",
        help="Path to market regime parameters",
    )
    parser.add_argument(
        "--offline-config",
        default="configs/offline.yaml",
        help="Path to offline dataset configuration with artefact declarations",
    )
    parser.add_argument(
        "--dataset-split",
        default="val",
        help="Dataset split identifier declared in the offline config",
    )
    parser.add_argument(
        "--liquidity-seasonality",
        default=None,
        help="Override path to liquidity seasonality coefficients (defaults to offline bundle)",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        default="tensorboard_logs",
        help="Directory where TensorBoard events will be written (use 'none' to disable)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Override the number of parallel training environments (falls back to config or default 8)",
    )
    args, unknown = parser.parse_known_args()

    raw_tensorboard_dir = (args.tensorboard_log_dir or "").strip()
    if raw_tensorboard_dir.lower() in {"", "none", "null"}:
        tensorboard_log_dir = None
    else:
        tensorboard_log_dir = Path(raw_tensorboard_dir).expanduser()

    os.environ["MARKET_REGIMES_JSON"] = args.regime_config

    split_key = (args.dataset_split or "").strip()
    offline_bundle = None
    offline_payload: Mapping[str, Any] | None = None
    offline_split_overrides: dict[str, list[dict[str, int | None]]] = {}
    if split_key and split_key.lower() not in {"none", "null"}:
        try:
            offline_bundle = resolve_split_bundle(args.offline_config, split_key)
        except FileNotFoundError as exc:
            raise SystemExit(f"Offline config not found: {args.offline_config}") from exc
        except KeyError as exc:
            raise SystemExit(
                f"Dataset split '{split_key}' not found in offline config {args.offline_config}"
            ) from exc
        except ValueError as exc:
            raise SystemExit(f"Failed to resolve offline split '{split_key}': {exc}") from exc
        offline_payload = load_offline_payload(args.offline_config)
        offline_split_overrides = _extract_offline_split_overrides(
            offline_payload, split_key, fallback_split="time"
        )

    seasonality_path = args.liquidity_seasonality
    fees_path: str | None = None
    adv_path: str | None = None
    seasonality_hash: str | None = None
    if offline_bundle is not None:
        if offline_bundle.version:
            print(
                f"Resolved offline dataset split '{offline_bundle.name}' version {offline_bundle.version}"
            )
        else:
            print(f"Resolved offline dataset split '{offline_bundle.name}'")
        seasonality_art = offline_bundle.artifacts.get("seasonality")
        if seasonality_art:
            if seasonality_path is None:
                seasonality_path = seasonality_art.path.as_posix()
            raw_hash = seasonality_art.info.artifact.get("verification_hash")
            if raw_hash:
                seasonality_hash = str(raw_hash)
        fees_art = offline_bundle.artifacts.get("fees")
        if fees_art:
            fees_path = fees_art.path.as_posix()
        adv_art = offline_bundle.artifacts.get("adv")
        if adv_art:
            adv_path = adv_art.path.as_posix()

    if seasonality_path is None:
        seasonality_path = "configs/liquidity_seasonality.json"
    args.liquidity_seasonality = seasonality_path

    if not Path(seasonality_path).exists():
        raise FileNotFoundError(
            f"Liquidity seasonality file not found: {seasonality_path}. Run offline builders first."
        )

    config_path = Path(args.config)
    raw_cfg_loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    if not isinstance(raw_cfg_loaded, Mapping):
        raw_cfg_loaded = {}
    env_payload: dict[str, Any] | None = None
    maybe_env = raw_cfg_loaded.get("env") if isinstance(raw_cfg_loaded, Mapping) else None
    if isinstance(maybe_env, Mapping):
        env_payload = deepcopy(dict(maybe_env))

    cfg = load_config(args.config)
    original_model_param_keys = _snapshot_model_param_keys(cfg)

    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--") and "." in unknown[i]:
            key = unknown[i][2:]
            if i + 1 >= len(unknown):
                raise ValueError(f"Missing value for argument {unknown[i]}")
            block, field = key.split('.', 1)
            value = yaml.safe_load(unknown[i + 1])
            overrides.setdefault(block, {})[field] = value
            i += 2
        else:
            i += 1

    cfg_dict = cfg.dict()
    if not isinstance(cfg_dict, dict):
        cfg_dict = dict(cfg_dict)
    if offline_bundle is not None:
        data_block = cfg_dict.setdefault("data", {})
        if offline_bundle.version:
            data_block["split_version"] = offline_bundle.version
        if offline_split_overrides and not data_block.get("split_overrides"):
            data_block["split_overrides"] = offline_split_overrides
        if adv_path:
            adv_block = cfg_dict.setdefault("adv", {})
            if not adv_block.get("path"):
                adv_block["path"] = adv_path
        if fees_path:
            fees_block = cfg_dict.setdefault("fees", {})
            if not fees_block.get("path"):
                fees_block["path"] = fees_path
    for block, params in overrides.items():
        block_dict = cfg_dict.setdefault(block, {})
        if not isinstance(block_dict, dict):
            block_dict = dict(block_dict)
            cfg_dict[block] = block_dict
        for key, value in params.items():
            if (
                block == "model"
                and key.startswith("params.")
                and key.split(".", 1)[1] == "vf_coef"
                and "vf_coef" in original_model_param_keys
            ):
                continue

            if (
                block == "model"
                and key == "params"
                and isinstance(value, Mapping)
            ):
                existing_params = block_dict.get("params")
                if isinstance(existing_params, Mapping):
                    merged_params = dict(existing_params)
                    merged_params.update(dict(value))
                else:
                    merged_params = dict(value)
                block_dict["params"] = merged_params
                continue

            _assign_nested(block_dict, key, value)
            if block == "data" and isinstance(block_dict, MutableMapping):
                _propagate_train_window_alias(block_dict, key, value)
        if block == "data" and isinstance(block_dict, MutableMapping):
            _ensure_train_window_aliases(block_dict)
        if block == "env":
            if env_payload is None or not isinstance(env_payload, dict):
                env_payload = {}
            for key, value in params.items():
                _assign_nested(env_payload, key, value)
    cfg_dict["liquidity_seasonality_path"] = args.liquidity_seasonality
    cfg_dict["latency_seasonality_path"] = args.liquidity_seasonality
    latency_block = cfg_dict.setdefault("latency", {})
    if not latency_block.get("latency_seasonality_path"):
        latency_block["latency_seasonality_path"] = args.liquidity_seasonality
    data_block_existing = cfg_dict.get("data")
    if isinstance(data_block_existing, MutableMapping):
        _ensure_train_window_aliases(data_block_existing)
    if seasonality_hash:
        cfg_dict["liquidity_seasonality_hash"] = seasonality_hash
    cfg = cfg.__class__.parse_obj(cfg_dict)

    env_payload_candidate: Mapping[str, Any] | None
    if isinstance(env_payload, Mapping):
        env_payload_candidate = env_payload
    elif isinstance(cfg_dict, Mapping):
        maybe_cfg_env = cfg_dict.get("env")
        env_payload_candidate = maybe_cfg_env if isinstance(maybe_cfg_env, Mapping) else None
    else:
        env_payload_candidate = None
    env_runtime_overrides, decision_override = _extract_env_runtime_overrides(
        env_payload_candidate
    )

    timing_defaults, timing_profiles = load_timing_profiles()
    exec_profile = getattr(cfg, "execution_profile", ExecutionProfile.MKT_OPEN_NEXT_H1)
    resolved_timing = resolve_execution_timing(exec_profile, timing_defaults, timing_profiles)
    timing_env_kwargs = {
        "decision_mode": DecisionTiming[resolved_timing.decision_mode],
        "decision_delay_ms": resolved_timing.decision_delay_ms,
        "latency_steps": resolved_timing.latency_steps,
    }
    if decision_override is not None:
        timing_env_kwargs["decision_mode"] = decision_override
    leak_guard_kwargs = {
        "decision_delay_ms": resolved_timing.decision_delay_ms,
        "min_lookback_ms": resolved_timing.min_lookback_ms,
    }

    sim_config = {k: getattr(cfg, k) for k in ("quantizer", "slippage", "fees", "latency", "risk", "no_trade")}
    sim_config["liquidity_seasonality_path"] = args.liquidity_seasonality
    liq_hash = _file_sha256(args.liquidity_seasonality)
    if liq_hash:
        sim_config["liquidity_seasonality_hash"] = liq_hash
        print(f"Liquidity seasonality hash: {liq_hash}")
    else:
        print(f"Warning: could not compute hash for {args.liquidity_seasonality}; seasonality consistency not enforced")

    processed_data_dir = getattr(cfg.data, "processed_dir", "data/processed")
    run_id = getattr(cfg, "run_id", "default-run")
    artifacts_root = Path(getattr(cfg, "artifacts_dir", "models")) / run_id
    trials_dir = artifacts_root / "trials"
    N_ENSEMBLE = int(cfg.model.params.get("n_ensemble", 5))

    if os.path.exists(trials_dir):
        print(f"Очистка старых артефактов из директории '{trials_dir}'...")
        shutil.rmtree(trials_dir)
    os.makedirs(trials_dir, exist_ok=True)

    print("Loading all pre-processed data...")
    all_feather_files = glob.glob(os.path.join(processed_data_dir, "*.feather"))
    if not all_feather_files:
        raise FileNotFoundError(
            f"No .feather files found in {processed_data_dir}. "
            f"Run prepare_advanced_data.py (Fear&Greed), prepare_events.py (macro events), "
            f"incremental_klines.py (1h candles), then prepare_and_run.py (merge/export).",
        )
    all_dfs_dict, all_obs_dict = load_all_data(all_feather_files, synthetic_fraction=0, seed=42)

    split_version, time_splits = _load_time_splits(cfg.data)
    if split_version:
        print(f"Using split version: {split_version}")
    timestamp_column = getattr(cfg.data, "timestamp_column", "timestamp")
    role_column = getattr(cfg.data, "role_column", "wf_role")

    dfs_with_roles: dict[str, pd.DataFrame] = {}
    inferred_test_any = False
    for symbol, df in all_dfs_dict.items():
        annotated, inferred_flag = _apply_role_column(df, time_splits, timestamp_column, role_column)
        dfs_with_roles[symbol] = annotated
        inferred_test_any = inferred_test_any or inferred_flag

    _ensure_validation_split_present(dfs_with_roles, time_splits, timestamp_column, role_column)

    train_intervals = time_splits.get("train", [])
    train_start_candidates = [start for start, _ in train_intervals if start is not None]
    train_end_candidates = [end for _, end in train_intervals if end is not None]
    train_start_ts = min(train_start_candidates) if train_start_candidates else None
    train_end_ts = max(train_end_candidates) if train_end_candidates else None

    PREPROC_PATH = artifacts_root / "preproc_pipeline.json"
    pipe = FeaturePipeline()
    pipe.fit(
        dfs_with_roles,
        train_mask_column=role_column,
        train_mask_values={"train"},
        train_start_ts=train_start_ts,
        train_end_ts=train_end_ts,
        timestamp_column=timestamp_column,
        split_version=split_version,
        train_intervals=train_intervals,
    )
    PREPROC_PATH.parent.mkdir(parents=True, exist_ok=True)
    pipe.save(str(PREPROC_PATH))
    try:
        save_sidecar_metadata(
            str(PREPROC_PATH),
            extra={
                "kind": "feature_pipeline",
                "split_version": split_version,
                "train_start_ts": train_start_ts,
                "train_end_ts": train_end_ts,
            },
        )
    except Exception:
        # Sidecar metadata helper may not be available in all environments.
        pass
    all_dfs_with_roles = pipe.transform_dict(dfs_with_roles, add_suffix="_z")
    print(f"Feature pipeline fitted and saved to {PREPROC_PATH}. Standardized columns *_z added.")
    print("To run inference over processed data, execute: python infer_signals.py")

    # --- Гейт качества данных: строгая валидация OHLCV перед сплитом ---
    _validator = DataValidator()
    for _key, _df in all_dfs_with_roles.items():
        try:
            # frequency=None -> автоопределение внутри валидатора
            _validator.validate(_df, frequency=None)
        except Exception as e:
            raise RuntimeError(f"Data validation failed for asset '{_key}': {e}")
    print("✓ Data validation passed for all assets.")

    def _extract_phase(phase: str) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for sym, df in all_dfs_with_roles.items():
            mask = df[role_column].astype(str) == phase
            phase_df = df.loc[mask].copy()
            if not phase_df.empty:
                out[sym] = phase_df.reset_index(drop=True)
        return out

    train_data_by_token = _extract_phase("train")
    val_data_by_token = _extract_phase("val")
    test_data_by_token = _extract_phase("test")

    def _select_obs(mapping: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
        return {sym: all_obs_dict[sym] for sym in mapping if sym in all_obs_dict}

    train_obs_by_token = _select_obs(train_data_by_token)
    val_obs_by_token = _select_obs(val_data_by_token)
    test_obs_by_token = _select_obs(test_data_by_token)

    unused_rows = {
        sym: int((df[role_column].astype(str) == "none").sum())
        for sym, df in all_dfs_with_roles.items()
        if (df[role_column].astype(str) == "none").any()
    }
    if unused_rows:
        total_unused = sum(unused_rows.values())
        print(
            f"Warning: {total_unused} rows across {len(unused_rows)} symbols were not assigned to train/val/test and will be ignored."
        )

    print("Time-based split summary:")
    for phase, mapping in (
        ("train", train_data_by_token),
        ("val", val_data_by_token),
        ("test", test_data_by_token),
    ):
        intervals = time_splits.get(phase, [])
        interval_desc = ", ".join(_format_interval(it) for it in intervals) if intervals else "(inferred remainder)"
        total_rows = sum(len(df) for df in mapping.values())
        observed_start, observed_end = _phase_bounds(mapping, timestamp_column)
        print(
            f"  {phase}: {len(mapping)} symbols, {total_rows} rows, intervals={interval_desc}, "
            f"observed=[{_fmt_ts(observed_start)} .. {_fmt_ts(observed_end)}]"
        )
    if inferred_test_any and not time_splits.get("test"):
        print("  Note: test split inferred from remaining rows (no explicit interval provided).")

    print("Calculating per-asset normalization stats from the training set...")
    norm_stats = {}

    # Итерируем по каждому активу в ТРЕНИРОВОЧНОМ наборе данных
    for asset_key, train_df in train_data_by_token.items():
        
        # 1. Находим признаки для нормализации в ДАННОМ конкретном активе
        features_to_normalize = [
            col for col in train_df.columns 
            if '_norm' in col and col not in ['log_volume_norm', 'fear_greed_value_norm']
        ]
        
        if not features_to_normalize:
            continue # Пропускаем, если у этого ассета нет таких признаков
            
        # 2. Рассчитываем статистики ТОЛЬКО по данным этого ассета
        mean_stats = train_df[features_to_normalize].mean().to_dict()
        std_stats = train_df[features_to_normalize].std().to_dict()
        
        # 3. Находим ID токена, связанный с этим ассетом
        # (Предполагаем, что один файл = один основной токен)
        if 'token_id' in train_df.columns:
            # Убедимся, что в датафрейме есть данные
            if not train_df.empty:
                token_id = train_df['token_id'].iloc[0]
                
                # 4. Сохраняем индивидуальные статистики для этого токена
                norm_stats[str(token_id)] = {'mean': mean_stats, 'std': std_stats}

    norm_stats_path = artifacts_root / "norm_stats.json"
    with open(norm_stats_path, "w") as f:
        json.dump(norm_stats, f, indent=4)
    print(f"Per-asset normalization stats for {len(norm_stats)} tokens calculated and saved.")

    HPO_TRIALS = 20 # Общее количество испытаний
    HPO_BUDGET_PER_TRIAL = 1_000_000 # Таймстепы для каждого испытания

    print(f"\n===== Starting Unified HPO Process ({HPO_TRIALS} trials) =====")

    sampler = TPESampler(n_startup_trials=5, seed=42)
    pruner = HyperbandPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    # Запускаем оптимизацию на ПОЛНОМ, диверсифицированном наборе данных
    n_envs_override: int | None = args.n_envs
    if n_envs_override is None:
        env_override_var = os.environ.get("TRAIN_NUM_ENVS")
        if env_override_var:
            try:
                n_envs_override = int(env_override_var)
            except ValueError:
                print(
                    f"Environment variable TRAIN_NUM_ENVS={env_override_var!r} is not a valid integer; ignoring override."
                )

    study.optimize(
        lambda t: objective(
            t,
            cfg,
            HPO_BUDGET_PER_TRIAL,
            train_data_by_token,
            train_obs_by_token,
            val_data_by_token,
            val_obs_by_token,
            test_data_by_token,
            test_obs_by_token,
            norm_stats,
            sim_config,
            timing_env_kwargs,
            env_runtime_overrides,
            leak_guard_kwargs,
            trials_dir,
            tensorboard_log_dir,
            n_envs_override,
        ),
        n_trials=HPO_TRIALS,
        n_jobs=1,
    )

    # Сохраняем итоговое исследование
    final_study = study
    if not final_study:
        print("No final study completed. Exiting.")
        return
    # <-- КОНЕЦ БЛОКА ДЛЯ ЗАМЕНЫ -->

    print(f"\nSaving best {N_ENSEMBLE} models from the final stage...")
    ensemble_dir = artifacts_root / "ensemble"
    if os.path.exists(ensemble_dir):
        shutil.rmtree(ensemble_dir)
    os.makedirs(ensemble_dir)

    top_trials = sorted(final_study.trials, key=lambda tr: tr.value or -1e9, reverse=True)[:N_ENSEMBLE]

    ensemble_meta = []
    for i, trial in enumerate(top_trials):
        model_idx = i + 1
        src_model = trials_dir / f"trial_{trial.number}_model.zip"
        src_stats = trials_dir / f"vec_normalize_{trial.number}.pkl"

        if os.path.exists(src_model):
            shutil.copyfile(src_model, ensemble_dir / f"model_{model_idx}.zip")
            if os.path.exists(src_stats):
                shutil.copyfile(src_stats, ensemble_dir / f"vec_normalize_{model_idx}.pkl")

            ensemble_meta.append({"ensemble_index": model_idx, "trial_number": trial.number, "value": trial.value, "params": trial.params})
        else:
            print(f"⚠️ WARNING: Could not find model for trial {trial.number}. Skipping.")
    # Копируем единый файл со статистиками нормализации наблюдений,
    # так как он является неотъемлемой частью всех моделей в ансамбле.
    src_norm_stats = artifacts_root / "norm_stats.json"
    if os.path.exists(src_norm_stats):
        shutil.copyfile(src_norm_stats, ensemble_dir / "norm_stats.json")
    else:
        print(f"⚠️ CRITICAL WARNING: Could not find the global 'norm_stats.json' file. The saved ensemble will not be usable for inference.")
    with open(ensemble_dir / "ensemble_meta.json", "w") as f:
        json.dump(ensemble_meta, f, indent=4)
    print(f"\n✅ Ensemble of {len(ensemble_meta)} models saved to '{ensemble_dir}'. HPO complete.")

    # --- Validation of the best model for reproducibility ---
    best_model_path = ensemble_dir / "model_1.zip"
    best_stats_path = ensemble_dir / "vec_normalize_1.pkl"
    if best_model_path.exists() and best_stats_path.exists():
        print("\nRunning validation on the best ensemble model...")
        best_trial = top_trials[0]

        final_eval_data = test_data_by_token if test_data_by_token else val_data_by_token
        final_eval_obs = test_obs_by_token if test_data_by_token else val_obs_by_token
        final_eval_mode = "test" if test_data_by_token else "val"
        if not final_eval_data:
            print("⚠️ Skipping final validation: evaluation split is empty.")
        else:
            def _make_env_val(symbol: str, df: pd.DataFrame):
                params = best_trial.params
                env_val_params = {
                    "norm_stats": norm_stats,
                    "window_size": params["window_size"],
                    "gamma": params["gamma"],
                    "atr_multiplier": params["atr_multiplier"],
                    "trailing_atr_mult": params["trailing_atr_mult"],
                    "tp_atr_mult": params["tp_atr_mult"],
                    "trade_frequency_penalty": params["trade_frequency_penalty"],
                    "turnover_penalty_coef": params["turnover_penalty_coef"],
                    "reward_return_clip": params["reward_return_clip"],
                    "turnover_norm_cap": params["turnover_norm_cap"],
                    "reward_cap": params["reward_cap"],
                    "reward_clip": params.get("reward_clip", reward_clip_params),
                    "mode": final_eval_mode,
                    "reward_shaping": False,
                    "warmup_period": warmup_period,
                    "ma5_window": MA5_WINDOW,
                    "ma20_window": MA20_WINDOW,
                    "atr_window": ATR_WINDOW,
                    "rsi_window": RSI_WINDOW,
                    "macd_fast": MACD_FAST,
                    "macd_slow": MACD_SLOW,
                    "macd_signal": MACD_SIGNAL,
                    "momentum_window": MOMENTUM_WINDOW,
                    "cci_window": CCI_WINDOW,
                    "bb_window": BB_WINDOW,
                    "obv_ma_window": OBV_MA_WINDOW,
                }
                env_val_params.update(sim_config)
                env_val_params.update(timing_env_kwargs)
                env_val_params.update(env_runtime_overrides)
                if (
                    reward_robust_clip_fraction_value is not None
                    and math.isfinite(reward_robust_clip_fraction_value)
                    and reward_robust_clip_fraction_value > 0.0
                ):
                    env_val_params["reward_robust_clip_fraction"] = (
                        reward_robust_clip_fraction_value
                    )
                env = TradingEnv(
                    df,
                    **env_val_params,
                    leak_guard=LeakGuard(LeakConfig(**leak_guard_kwargs))
                )
                setattr(env, "selected_symbol", symbol)
                env = _wrap_action_space_if_needed(
                    env,
                    bins_vol=bins_vol,
                    action_overrides=action_overrides,
                    long_only=long_only_flag,
                )
                return env

            eval_env_fns: list[Callable[[], TradingEnv]] = []
            for symbol, df in sorted(final_eval_data.items()):
                def _factory(symbol=symbol, df=df):
                    return _make_env_val(symbol, df)

                eval_env_fns.append(_factory)

            monitored_eval_env = VecMonitor(DummyVecEnv(eval_env_fns))
            check_model_compat(str(best_stats_path))
            eval_env = _freeze_vecnormalize(
                VecNormalize.load(str(best_stats_path), monitored_eval_env)
            )
            eval_env.norm_reward = False
            eval_env.clip_reward = None

            DistributionalPPO = _get_distributional_ppo()
            best_model = DistributionalPPO.load(str(best_model_path), env=eval_env)

            ann_sqrt, bar_seconds = _annualization_sqrt_from_env(eval_env)
            _log_annualization("Validation evaluation", ann_sqrt, bar_seconds)
            if bar_seconds is None:
                steps_per_year = None
                print(
                    "[validation] Annualization sqrt factor: "
                    f"{ann_sqrt:.6f} (bar interval unknown)"
                )
            else:
                steps_per_year = _SECONDS_PER_YEAR / float(bar_seconds)
                print(
                    "[validation] Annualization sqrt factor: "
                    f"{ann_sqrt:.6f} (bar_seconds={bar_seconds:.3f}, steps_per_year={steps_per_year:.2f})"
                )

            rewards, equity_curves = evaluate_policy_custom_cython(
                best_model, eval_env, num_episodes=max(1, len(final_eval_data))
            )
            all_returns = [
                pd.Series(curve).pct_change().dropna().to_numpy()
                for curve in equity_curves if len(curve) > 1
            ]
            flat_returns = np.concatenate(all_returns) if all_returns else np.array([0.0])
            sortino = sortino_ratio(flat_returns, annualization_sqrt=ann_sqrt)
            sharpe = sharpe_ratio(flat_returns, annualization_sqrt=ann_sqrt)

            report = {
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "sortino_ratio": float(sortino),
                "sharpe_ratio": float(sharpe),
            }
            with open(ensemble_dir / "validation_report.json", "w") as f:
                json.dump(report, f, indent=4)
            print(
                f"Validation metrics -> Sortino: {sortino:.4f}, Sharpe: {sharpe:.4f}. "
                f"Report saved to '{ensemble_dir / 'validation_report.json'}'"
            )
    else:
        print(
            "⚠️ Could not find best model or normalization stats for validation evaluation."
        )

def _configure_start_method() -> None:
    """Pick a multiprocessing start method suited for the current platform."""

    try:
        current = mp.get_start_method(allow_none=True)
    except Exception:
        current = None

    if current is not None:
        return

    preferred_order: list[str]
    if sys.platform.startswith("win"):
        preferred_order = ["spawn"]
    else:
        preferred_order = ["fork", "forkserver", "spawn"]

    available = set()
    try:
        available = set(mp.get_all_start_methods())
    except Exception:
        pass

    for method in preferred_order:
        if available and method not in available:
            continue
        try:
            mp.set_start_method(method, force=False)
            return
        except RuntimeError:
            continue


if __name__ == "__main__":
    _configure_start_method()

    def _extract_grad_sanity(argv: list[str]) -> str | None:
        for idx, arg in enumerate(argv):
            if arg == "--grad-sanity":
                if idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
                    return argv[idx + 1]
                return "1"
            if arg.startswith("--grad-sanity="):
                value = arg.split("=", 1)[1]
                return value if value else "1"
        return None

    flag_value = _extract_grad_sanity(sys.argv[1:])
    if flag_value is not None:
        os.environ["GRAD_SANITY"] = flag_value

    # --- gradient sanity check (включается флагом окружения) ---
    from runtime_flags import get_bool
    if get_bool("GRAD_SANITY", False):
        from tools.grad_sanity import run_check
        run_check()

    main()
