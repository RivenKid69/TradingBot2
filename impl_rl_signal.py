# -*- coding: utf-8 -*-
"""
impl_rl_signal.py
=================

``RLAlphaSignal`` (Stage A6) — ключевой рефактор парадигмы: RL (Distributional PPO)
перестаёт быть «торгующим агентом» и становится **одним из сигналов** среди многих,
с измеримым IC, попадающим в Alpha-модель (A6) наравне с классическими факторами.

**Обучение RL не трогаем.** Читаем только ВЫХОД политики через адаптер:
* ``utility_source`` — ожидаемая полезность позиции по (ts, symbol): либо готовая
  Series/панель, либо callable(panel)->Series. Может быть получена из value-head или
  усреднения квантилей распределительного критика.
* ``confidence`` — уверенность в [0,1] (как правило из conformal-предсказания: чем уже
  интервал, тем выше уверенность). Сигнал шринкуется: ``signal = utility × confidence``.

DI-дружелюбно: всё внешнее подаётся как Series/callable, поэтому модуль тестируется без
RL и без сети. ``RLAlphaSignal`` — это ``BaseSignal``, регистрируется в ``SignalLibrary``
как любой сигнал. Слой ``impl_``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd

from core_portfolio import Panel
from service_signals import BaseSignal

UtilitySource = Union[pd.Series, Callable[[Panel], pd.Series]]
ConfidenceSource = Union[None, float, pd.Series, Callable[[Panel], pd.Series]]


def cvar_from_quantiles_np(quantiles: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorized CVaR_α из квантилей — кусочно-линейное интегрирование квантильной функции.

    Численно ЭКВИВАЛЕНТНО ``DistributionalPPO._cvar_from_quantiles`` (обучающая цель), но на
    numpy и батчем по строкам. Предполагает уровни-центры ``τ_i = (i+0.5)/N`` (как
    ``QuantileValueHead``). Это устраняет методологический разрыв: раньше сигнал считал CVaR
    наивным ``mean(нижние-k)`` (~16.5% ошибки на N(0,1)), теперь — как модель (~4.6%).

    Args:
        quantiles: ``[B, N]`` (или ``[N]``) — значения на уровнях τ_i, по возрастанию τ.
        alpha: уровень CVaR в (0, 1].
    Returns:
        ``[B]`` значения CVaR_α.
    """
    q = np.asarray(quantiles, dtype="float64")
    if q.ndim == 1:
        q = q[None, :]
    B, N = q.shape
    if N == 0:
        return np.zeros(B)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("CVaR alpha must be in (0, 1]")
    mass = 1.0 / N
    alpha_idx_float = alpha * N - 0.5

    # α ниже первого центра τ_0 → линейная экстраполяция из q0,q1
    if alpha_idx_float < 0.0:
        if N >= 2:
            q0, q1 = q[:, 0], q[:, 1]
            tau_0, tau_1 = 0.5 / N, 1.5 / N
            slope = (q1 - q0) / (tau_1 - tau_0)
            boundary = q0 + slope * (alpha - tau_0)
            value_at_0 = q0 - slope * tau_0
            return (value_at_0 + boundary) / 2.0
        return q[:, 0]

    alpha_idx = int(math.floor(alpha_idx_float))

    # α за последним центром → все квантили с дробным довесом
    if alpha_idx >= N - 1:
        k_float = alpha * N
        full_mass = int(min(N, math.floor(k_float)))
        frac = float(k_float - full_mass)
        tail_sum = q[:, :full_mass].sum(axis=1) if full_mass > 0 else np.zeros(B)
        partial = q[:, full_mass] * frac if (frac > 1e-8 and full_mass < N) else np.zeros(B)
        expectation = mass * (tail_sum + partial)
        tail_mass = max(alpha, mass * (full_mass + frac))
        return expectation / max(tail_mass, 1e-6)

    # стандартный случай: α внутри [alpha_idx/N, (alpha_idx+1)/N)
    tau_i = (alpha_idx + 0.5) / N
    tau_i_next = (alpha_idx + 1.5) / N
    q_i, q_i_next = q[:, alpha_idx], q[:, alpha_idx + 1]
    weight = (alpha - tau_i) / (tau_i_next - tau_i)
    value_at_alpha = q_i * (1.0 - weight) + q_i_next * weight

    interval_start = alpha_idx / N
    if alpha_idx == 0:
        slope = (q_i_next - q_i) / (tau_i_next - tau_i)
        value_at_start = q_i - slope * tau_i
    else:
        q_i_prev = q[:, alpha_idx - 1]
        tau_i_prev = (alpha_idx - 0.5) / N
        weight_start = (interval_start - tau_i_prev) / (tau_i - tau_i_prev)
        value_at_start = q_i_prev * (1.0 - weight_start) + q_i * weight_start

    full_contribution = mass * q[:, :alpha_idx].sum(axis=1) if alpha_idx > 0 else np.zeros(B)
    partial_mass = alpha - interval_start
    avg_value = (value_at_start + value_at_alpha) / 2.0
    expectation = full_contribution + avg_value * partial_mass
    return expectation / max(alpha, 1e-6)


def expected_utility_from_quantiles(
    quantiles: pd.DataFrame,
    *,
    cvar_alpha: Optional[float] = None,
) -> pd.Series:
    """Ожидаемая полезность из квантилей распределительного критика.

    По умолчанию — среднее по квантилям (ожидание). Если задан ``cvar_alpha`` — CVaR_α через
    то же кусочно-линейное интегрирование, что и обучающая цель модели
    (``cvar_from_quantiles_np`` ≡ ``DistributionalPPO._cvar_from_quantiles``), а НЕ наивный
    ``mean(нижние-k)``. Это согласует риск-aware utility сигнала с CVaR обучения.
    """
    q = quantiles.astype("float64")
    if cvar_alpha is None:
        return q.mean(axis=1)
    # квантили должны идти по возрастанию τ; сортируем по значению на случай немонотонной головы
    sorted_q = np.sort(q.to_numpy(), axis=1)
    cvar = cvar_from_quantiles_np(sorted_q, float(cvar_alpha))
    return pd.Series(cvar, index=q.index)


def conformal_confidence_from_widths(
    widths: pd.Series,
    *,
    baseline_width: float,
    min_conf: float = 0.5,
) -> pd.Series:
    """Уверенность из ширины conformal-интервала: уже интервал → выше уверенность.

    Согласовано с КАНОНИЧЕСКОЙ position-scaling платформы
    (``impl_conformal.UncertaintyTrackerImpl``): ``width <= baseline → conf = 1``; иначе
    линейная редукция ``conf = 1 - min((width-baseline)/baseline · max_red, max_red)``,
    где ``max_red = 1 - min_conf`` — глубина шринка (нижний предел ``conf``).
    Реализовано локально (слой ``impl_`` не зависит от ``service_``), но форма кривой,
    граница (``width=baseline → 1``) и floor совпадают с каноном. NaN/inf ширины → ``conf = 1``.
    """
    w = widths.astype("float64").replace([np.inf, -np.inf], np.nan)
    baseline = max(float(baseline_width), 1e-12)
    max_red = 1.0 - float(min_conf)
    over = ((w - baseline) / baseline).clip(lower=0.0)
    reduction = (over * max_red).clip(upper=max_red)
    conf = (1.0 - reduction).fillna(1.0)
    return conf.clip(lower=float(min_conf), upper=1.0)


class RLAlphaSignal(BaseSignal):
    """RL-выход как cross-sectional сигнал (utility × conformal confidence)."""

    def __init__(
        self,
        name: str = "rl_alpha",
        *,
        utility_source: UtilitySource,
        confidence: ConfidenceSource = None,
    ) -> None:
        self.name = name
        self._utility = utility_source
        self._confidence = confidence

    def _resolve_series(self, source: Any, panel: Panel) -> pd.Series:
        if callable(source) and not isinstance(source, pd.Series):
            out = source(panel)
        else:
            out = source
        if not isinstance(out, pd.Series):
            raise TypeError("utility_source must resolve to a pandas Series")
        return out.reindex(panel.index)

    def _resolve_confidence(self, panel: Panel) -> Union[pd.Series, float]:
        c = self._confidence
        if c is None:
            return 1.0
        if isinstance(c, (int, float)):
            return float(c)
        if callable(c) and not isinstance(c, pd.Series):
            c = c(panel)
        if isinstance(c, pd.Series):
            return c.reindex(panel.index).fillna(1.0).clip(0.0, 1.0)
        raise TypeError("confidence must be None, scalar, Series or callable")

    def compute_panel(self, panel: Panel) -> pd.Series:
        u = self._resolve_series(self._utility, panel).astype("float64")
        c = self._resolve_confidence(panel)
        if isinstance(c, pd.Series):
            out = u * c
        else:
            out = u * float(c)
        return out.rename(self.name)

    # удобные конструкторы
    @classmethod
    def from_value_panel(
        cls,
        value: pd.Series,
        *,
        name: str = "rl_alpha",
        confidence: ConfidenceSource = None,
    ) -> "RLAlphaSignal":
        return cls(name, utility_source=value, confidence=confidence)

    @classmethod
    def from_quantiles(
        cls,
        quantiles: pd.DataFrame,
        *,
        name: str = "rl_alpha",
        cvar_alpha: Optional[float] = None,
        confidence: ConfidenceSource = None,
    ) -> "RLAlphaSignal":
        utility = expected_utility_from_quantiles(quantiles, cvar_alpha=cvar_alpha)
        return cls(name, utility_source=utility, confidence=confidence)


__all__ = [
    "RLAlphaSignal",
    "expected_utility_from_quantiles",
    "cvar_from_quantiles_np",
    "conformal_confidence_from_widths",
]
