# -*- coding: utf-8 -*-
"""
service_rl_inference.py
=======================

RL-инференс как cross-sectional **сигнал** (Stage D6) — обученная Distributional-PPO политика
становится ОДНИМ измеримым сигналом (IC рядом с классикой), а НЕ «торгующим агентом».

**ОБУЧЕНИЕ НЕ ТРОГАЕМ.** Адаптер только читает ВЫХОД политики по cross-section панели:
  * ``utility`` — ожидаемая полезность позиции по (ts, symbol): из value-head (``utility='value'``)
    или среднего нижних квантилей распределительного критика (``utility='cvar'``,
    ``expected_utility_from_quantiles``);
  * ``confidence`` — опц. из conformal-ширин (``conformal_confidence_from_widths``); сигнал
    шринкуется ``utility × confidence`` (см. ``impl_rl_signal.RLAlphaSignal``).

**Честно/DI:** запуск произвольного чекпоинта требует МОДЕЛЬНО-СПЕЦИФИЧНОЙ ``obs_fn`` (схема
наблюдений как при обучении) и загрузчика — поэтому мы НЕ выдумываем «универсальный» лоадер.
Подайте ``value_fn``/``quantiles_fn``/``obs_fn`` (DI) или ``model_loader(checkpoint)``. Без них
``available()=False`` → сигнал нейтрален (NaN), пайплайн его пропускает (graceful). torch
импортируется лениво и только в пользовательском загрузчике.

**CCEA:** инференс НЕ создаёт ордера — выдаёт сигнал → μ → веса как любой другой. Артефакт
должен быть подписан (BYO/CCEA-signed). Слой ``service_``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from core_portfolio import Panel
from impl_rl_signal import (
    RLAlphaSignal, conformal_confidence_from_widths, expected_utility_from_quantiles,
)

logger = logging.getLogger(__name__)


def _default_obs_fn(panel: Panel) -> Any:
    """По умолчанию obs = сама панель (пользовательский value_fn/quantiles_fn читает что нужно).

    Реальной модели нужна obs_fn, строящая наблюдения по схеме обучения (BYO).
    """
    return panel


def _default_model_loader(checkpoint: str, device: str) -> Optional[dict]:  # pragma: no cover - модельно-специфично
    """Честный no-op: универсального лоадера нет. Верните dict(value_fn/quantiles_fn/obs_fn) в BYO-загрузчике."""
    logger.warning(
        "RLInferenceAdapter: дефолтный загрузчик не запускает произвольный чекпоинт '%s' — "
        "подайте model_loader или value_fn/quantiles_fn+obs_fn (модельно-специфичная obs-схема).",
        checkpoint,
    )
    return None


def make_panel_obs_fn(
    feature_cols: Optional[list] = None,
    *,
    expected_dim: Optional[int] = None,
    dtype: str = "float32",
) -> Callable[[Panel], "np.ndarray"]:
    """Референсный ``obs_fn``: панель → матрица наблюдений ``[n_rows, n_features]``.

    Если ``feature_cols`` задан — берутся ровно эти колонки в этом порядке (это и есть
    obs-схема обучения, BYO-ответственность пользователя). Иначе — все числовые колонки.
    При ``expected_dim`` валидируется размерность против obs_space модели: несовпадение →
    понятная ошибка (надо подать ``feature_cols`` под схему обучения). NaN/inf → 0.0.
    """

    def obs_fn(panel: Panel) -> "np.ndarray":
        if feature_cols is not None:
            cols = list(feature_cols)
            missing = [c for c in cols if c not in panel.columns]
            if missing:
                raise ValueError(f"obs_fn: в панели нет feature-колонок {missing}")
            frame = panel[cols]
        else:
            frame = panel.select_dtypes(include=[np.number])
        arr = np.ascontiguousarray(frame.to_numpy(dtype=dtype))
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if expected_dim is not None and arr.shape[1] != int(expected_dim):
            raise ValueError(
                f"obs_fn: размерность obs {arr.shape[1]} != obs_dim модели {int(expected_dim)}. "
                "Подайте feature_cols, совпадающие со схемой наблюдений обучения."
            )
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    return obs_fn


def make_sb3_distributional_loader(
    *,
    feature_cols: Optional[list] = None,
    ppo_cls: Any = None,
    signature_policy: Optional[str] = None,
    live: bool = False,
) -> Callable[[str, str], Optional[dict]]:
    """Настоящий загрузчик чекпоинта Distributional-PPO (SB3) → value_fn/quantiles_fn/obs_fn.

    Читает ТОЛЬКО выход критика (eval, no_grad). Каждая строка ``(ts, symbol)`` — независимое
    наблюдение: ``episode_starts=1`` обнуляет LSTM-состояние построчно (snapshot-полезность).
    ``quantiles_fn`` → ``policy.value_quantiles`` (квантильная голова критика);
    ``value_fn`` → ``policy.predict_values`` (min twin-critics — консервативная точечная оценка).
    torch/sb3 импортируются лениво ВНУТРИ загрузчика — модуль грузится без torch. Обучение не трогаем.

    Безопасность: SB3-чекпоинт — это pickle. Перед десериализацией артефакт
    проходит гейт Ed25519-подписи (services/model_signature_gate). Политика:
    ``signature_policy``/env ``RIVEN_MODEL_SIGNATURE_POLICY``; по умолчанию
    ``enforce`` при ``live=True`` (провал = исключение, НЕ нейтральный сигнал —
    торговый контур обязан остановиться, а не молча торговать без модели)
    и ``warn`` для research/backtest.
    """

    def model_loader(checkpoint: str, device: str) -> Optional[dict]:
        # Гейт подписи — ДО импорта torch и до любого чтения pickle-содержимого.
        # В enforce-политике ModelSignatureError намеренно НЕ глотается.
        from services.model_signature_gate import verify_model_artifact
        verify_model_artifact(
            checkpoint, policy=signature_policy, live=live, context="rl-inference",
        )

        try:
            import numpy as _np  # noqa: F401
            import torch
            from sb3_contrib.common.recurrent.type_aliases import RNNStates
        except Exception as exc:  # pragma: no cover - окружение без torch
            logger.warning("RL loader: torch/sb3 недоступны (%s) — сигнал нейтрален.", exc)
            return None

        if ppo_cls is not None:
            _cls = ppo_cls
        else:
            try:
                from distributional_ppo import DistributionalPPO as _cls
            except Exception as exc:  # pragma: no cover
                logger.warning("RL loader: не импортировать DistributionalPPO (%s).", exc)
                return None

        try:
            model = _cls.load(checkpoint, env=None, device=device)
        except Exception as exc:
            logger.warning("RL loader: не загрузить чекпоинт '%s' (%s).", checkpoint, exc)
            return None

        policy = model.policy
        policy.set_training_mode(False)
        dev = next(policy.parameters()).device
        obs_dim = int(np.prod(policy.observation_space.shape))

        def _zero_states(batch: int):
            init = policy.recurrent_initial_state

            def _mk(state_tuple):
                out = []
                for t in state_tuple:
                    shp = list(t.shape)
                    shp[1] = batch  # [n_layers, n_envs->batch, hidden]
                    out.append(torch.zeros(shp, dtype=t.dtype, device=dev))
                return tuple(out)

            return RNNStates(pi=_mk(init.pi), vf=_mk(init.vf))

        def _prep(obs):
            t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=dev)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            return t

        def quantiles_fn(obs):
            with torch.no_grad():
                t = _prep(obs)
                b = int(t.shape[0])
                q = policy.value_quantiles(t, _zero_states(b), torch.ones(b, device=dev))
                return q.detach().cpu().numpy()

        def value_fn(obs):
            with torch.no_grad():
                t = _prep(obs)
                b = int(t.shape[0])
                v = policy.predict_values(t, _zero_states(b), torch.ones(b, device=dev))
                return v.detach().cpu().numpy().reshape(-1)

        obs_fn = make_panel_obs_fn(feature_cols=feature_cols, expected_dim=obs_dim)
        return {
            "value_fn": value_fn,
            "quantiles_fn": quantiles_fn,
            "obs_fn": obs_fn,
            "model": model,
            "value_head": policy.value_head_metadata(),
        }

    return model_loader


class RLInferenceAdapter:
    """Выход Distributional-PPO политики → utility/confidence панели → RLAlphaSignal."""

    def __init__(
        self,
        *,
        checkpoint: Optional[str] = None,
        value_fn: Optional[Callable[[Any], Any]] = None,
        quantiles_fn: Optional[Callable[[Any], Any]] = None,
        obs_fn: Optional[Callable[[Panel], Any]] = None,
        utility: str = "value",                 # 'value' | 'cvar'
        cvar_alpha: float = 0.05,
        widths_fn: Optional[Callable[[Panel], pd.Series]] = None,
        conf_baseline_width: float = 0.1,
        model_loader: Optional[Callable[[str, str], Optional[dict]]] = None,
        device: str = "cpu",
    ) -> None:
        self.checkpoint = checkpoint
        self._value_fn = value_fn
        self._quantiles_fn = quantiles_fn
        self._user_obs_fn = obs_fn is not None  # пользовательская obs_fn имеет приоритет над лоадерной
        self._obs_fn = obs_fn or _default_obs_fn
        self.utility = utility
        self.cvar_alpha = float(cvar_alpha)
        self._widths_fn = widths_fn
        self.conf_baseline_width = float(conf_baseline_width)
        self._model_loader = model_loader or _default_model_loader
        self.device = device
        self._loaded = False

    # ------------------------------------------------------------------
    def _ensure(self) -> None:
        if self._value_fn is not None or self._quantiles_fn is not None:
            return
        if self.checkpoint and not self._loaded:
            self._loaded = True
            try:
                built = self._model_loader(self.checkpoint, self.device)
            except Exception as exc:  # pragma: no cover
                logger.warning("RLInferenceAdapter: загрузчик упал: %s", exc)
                built = None
            if built:
                self._value_fn = built.get("value_fn")
                self._quantiles_fn = built.get("quantiles_fn")
                # obs_fn от лоадера применяем ТОЛЬКО если пользователь не подал свою
                if built.get("obs_fn") and not self._user_obs_fn:
                    self._obs_fn = built["obs_fn"]

    def available(self) -> bool:
        self._ensure()
        return (self._value_fn is not None) or (self._quantiles_fn is not None)

    # ------------------------------------------------------------------
    def utility_panel(self, panel: Panel) -> pd.Series:
        """utility по (ts, symbol). Нет политики → NaN (graceful neutral)."""
        self._ensure()
        if not self.available():
            return pd.Series(np.nan, index=panel.index, name="rl_utility")
        obs = self._obs_fn(panel)
        if self.utility == "cvar" and self._quantiles_fn is not None:
            q = np.asarray(self._quantiles_fn(obs), dtype="float64")
            u = expected_utility_from_quantiles(pd.DataFrame(q, index=panel.index), cvar_alpha=self.cvar_alpha)
        elif self._value_fn is not None:
            v = np.asarray(self._value_fn(obs), dtype="float64").reshape(-1)
            u = pd.Series(v, index=panel.index)
        else:  # только quantiles → ожидание
            q = np.asarray(self._quantiles_fn(obs), dtype="float64")
            u = pd.Series(q.mean(axis=1), index=panel.index)
        return u.rename("rl_utility")

    def confidence_panel(self, panel: Panel) -> Optional[pd.Series]:
        """confidence из conformal-ширин (опц.). None → шринкования нет."""
        if self._widths_fn is None:
            return None
        try:
            widths = self._widths_fn(panel)
        except Exception as exc:  # pragma: no cover
            logger.warning("RLInferenceAdapter: widths_fn упал: %s", exc)
            return None
        if not isinstance(widths, pd.Series):
            return None
        return conformal_confidence_from_widths(widths, baseline_width=self.conf_baseline_width)

    # ------------------------------------------------------------------
    def build_signal(self, name: str = "rl_alpha") -> RLAlphaSignal:
        """Собрать RLAlphaSignal (utility × conformal confidence) для SignalLibrary."""
        conf = self.confidence_panel if self._widths_fn is not None else None
        return RLAlphaSignal(name, utility_source=self.utility_panel, confidence=conf)

    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        *,
        feature_cols: Optional[list] = None,
        utility: str = "value",
        cvar_alpha: float = 0.05,
        obs_fn: Optional[Callable[[Panel], Any]] = None,
        widths_fn: Optional[Callable[[Panel], pd.Series]] = None,
        conf_baseline_width: float = 0.1,
        device: str = "cpu",
        ppo_cls: Any = None,
    ) -> "RLInferenceAdapter":
        """Собрать адаптер с НАСТОЯЩИМ загрузчиком Distributional-PPO чекпоинта.

        ``feature_cols`` задаёт obs-схему обучения (порядок колонок панели). Без torch/чекпоинта
        адаптер остаётся ``available()=False`` → сигнал нейтрален (graceful). Можно подать свою
        ``obs_fn`` (имеет приоритет над референсной из загрузчика).
        """
        loader = make_sb3_distributional_loader(feature_cols=feature_cols, ppo_cls=ppo_cls)
        return cls(
            checkpoint=checkpoint,
            utility=utility,
            cvar_alpha=cvar_alpha,
            obs_fn=obs_fn,
            widths_fn=widths_fn,
            conf_baseline_width=conf_baseline_width,
            model_loader=loader,
            device=device,
        )


__all__ = [
    "RLInferenceAdapter",
    "make_sb3_distributional_loader",
    "make_panel_obs_fn",
]
