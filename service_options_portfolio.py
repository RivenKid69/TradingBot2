# -*- coding: utf-8 -*-
"""
service_options_portfolio.py
============================

Options portfolio constructor в **пространстве греческих** (Stage B5) — отдельная машинерия
от directional ``μ → Σ → w*`` (B1-B4). Опционы — это портфель ЭКСПОЗИЦИЙ по греческим, а не
направленные веса по активам: цель — собрать структуру, которая ХАРВЕСТИТ edge (volatility
risk premium / skew / dispersion / term-structure), будучи **нейтральной** по выбранным
грекам (delta/vega/gamma/…).

Метод (numpy-only, без cvxpy/scipy — консистентно с движком):
  1. альфа по ногам (из ``signals/options_signals``: VRP/skew/dispersion/term) → w0;
  2. **проекция на null-space** матрицы греков G (строки = нейтрализуемые греки):
         w = w0 − G⁺ (G w0),  G⁺ = pinv(G)  ⇒  G w ≈ 0 (точная нейтральность);
  3. масштаб к gross_max, клип по ногам, ре-проекция (восстановить нейтральность);
  4. остаточные греки (delta/gamma/vega/theta/rho) — для отчёта (≈0 у нейтрализуемых).

Переиспользует ``impl_greeks_vectorized.compute_all_greeks_batch`` и ``impl_pricing``.
Слой ``service_``. Достаточно много ног (> числа нейтрализуемых греков) — иначе null-space
тривиален и портфель схлопывается в 0 (это корректно: нельзя быть нейтральным и ненулевым).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from impl_greeks_vectorized import compute_all_greeks_batch

logger = logging.getLogger(__name__)

# Греки, доступные для нейтрализации/отчёта.
ALL_GREEKS = ("delta", "gamma", "vega", "theta", "rho")


@dataclass
class OptionLeg:
    """Кандидат-нога опционного портфеля (per-share греки считаются из параметров)."""

    symbol: str
    spot: float
    strike: float
    time_to_expiry: float           # годы
    iv: float                       # implied vol (доля, напр. 0.25)
    is_call: bool = True
    rate: float = 0.0
    dividend_yield: float = 0.0
    alpha: float = 0.0              # привлекательность (из options-сигналов)
    multiplier: float = 100.0       # контрактный мультипликатор


@dataclass
class GreeksNeutralConstraints:
    """Ограничения greeks-space оптимизации."""

    neutralize: List[str] = field(default_factory=lambda: ["delta", "vega"])
    gross_max: float = 1.0          # суммарный |вес| (в контрактах, нормированный)
    max_position: Optional[float] = None  # клип на ногу
    tol: float = 1e-6               # порог «нейтрально» для отчёта

    def normalized(self) -> List[str]:
        bad = [g for g in self.neutralize if g not in ALL_GREEKS]
        if bad:
            raise ValueError(f"unknown greeks to neutralize: {bad} (valid={ALL_GREEKS})")
        return list(self.neutralize)


@dataclass
class OptionsPortfolio:
    """Результат: веса по ногам + остаточные греки + метрики."""

    weights: pd.Series              # index=symbol, value=контракты (signed)
    net_greeks: Dict[str, float]    # остаточные портфельные греки (нейтрализуемые ≈0)
    gross: float
    objective: float                # альфа·w (захваченный edge)
    neutralized: List[str]
    is_neutral: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "weights": {str(k): float(v) for k, v in self.weights.items()},
            "net_greeks": {k: float(v) for k, v in self.net_greeks.items()},
            "gross": float(self.gross), "objective": float(self.objective),
            "neutralized": list(self.neutralized), "is_neutral": bool(self.is_neutral),
        }


class OptionsPortfolioConstructor:
    """Строит greeks-нейтральный опционный портфель методом null-space проекции."""

    def __init__(self, constraints: Optional[GreeksNeutralConstraints] = None) -> None:
        self.constraints = constraints or GreeksNeutralConstraints()

    @staticmethod
    def _greeks_matrix(legs: Sequence[OptionLeg]) -> Dict[str, np.ndarray]:
        spot = np.array([l.spot for l in legs], dtype="float64")
        strike = np.array([l.strike for l in legs], dtype="float64")
        tte = np.array([max(l.time_to_expiry, 1e-9) for l in legs], dtype="float64")
        rate = np.array([l.rate for l in legs], dtype="float64")
        div = np.array([l.dividend_yield for l in legs], dtype="float64")
        iv = np.array([max(l.iv, 1e-9) for l in legs], dtype="float64")
        is_call = np.array([bool(l.is_call) for l in legs], dtype=bool)
        mult = np.array([l.multiplier for l in legs], dtype="float64")
        g = compute_all_greeks_batch(spot, strike, tte, rate, div, iv, is_call)
        # позиционный грек ноги = per-share грек × мультипликатор
        return {
            "delta": np.asarray(g.delta) * mult,
            "gamma": np.asarray(g.gamma) * mult,
            "vega": np.asarray(g.vega) * mult,
            "theta": np.asarray(g.theta) * mult,
            "rho": np.asarray(g.rho) * mult,
        }

    def construct(self, legs: Sequence[OptionLeg],
                  alpha: Optional[Sequence[float]] = None) -> OptionsPortfolio:
        legs = list(legs)
        symbols = [l.symbol for l in legs]
        n = len(legs)
        if n == 0:
            return OptionsPortfolio(pd.Series(dtype="float64"), {g: 0.0 for g in ALL_GREEKS},
                                    0.0, 0.0, self.constraints.normalized(), True)

        greeks = self._greeks_matrix(legs)
        neutralize = self.constraints.normalized()
        G = np.vstack([greeks[g] for g in neutralize]) if neutralize else np.zeros((0, n))

        a = np.array(list(alpha) if alpha is not None else [l.alpha for l in legs], dtype="float64")
        a = np.nan_to_num(a, nan=0.0)
        # стандартизуем альфу → стартовый w0
        sd = a.std(ddof=0)
        w0 = (a - a.mean()) / (sd if sd > 0 else 1.0)

        w = self._project_null(G, w0)
        w = self._scale_gross(w)
        if self.constraints.max_position is not None:
            w = np.clip(w, -self.constraints.max_position, self.constraints.max_position)
            w = self._project_null(G, w)         # восстановить нейтральность после клипа
            w = self._scale_gross(w)

        net = {g: float(greeks[g] @ w) for g in ALL_GREEKS}
        is_neutral = all(abs(net[g]) <= max(self.constraints.tol, 1e-6) * (1.0 + abs(self.constraints.gross_max))
                         for g in neutralize)
        obj = float(a @ w)
        return OptionsPortfolio(
            weights=pd.Series(w, index=symbols, name="contracts"),
            net_greeks=net, gross=float(np.abs(w).sum()), objective=obj,
            neutralized=neutralize, is_neutral=is_neutral,
        )

    @staticmethod
    def _project_null(G: np.ndarray, w0: np.ndarray) -> np.ndarray:
        """Проекция w0 на null-space строк G: w = w0 − pinv(G) (G w0) ⇒ G w ≈ 0."""
        if G.shape[0] == 0:
            return w0.copy()
        Gpinv = np.linalg.pinv(G)
        return w0 - Gpinv @ (G @ w0)

    def _scale_gross(self, w: np.ndarray) -> np.ndarray:
        gross = float(np.abs(w).sum())
        if gross <= 1e-12:
            return w
        return w * (self.constraints.gross_max / gross)


# ---------------------------------------------------------------------------
# High-level + synthetic
# ---------------------------------------------------------------------------
def construct_options_portfolio(
    legs: Sequence[OptionLeg],
    *,
    neutralize: Optional[Sequence[str]] = None,
    gross_max: float = 1.0,
    max_position: Optional[float] = None,
) -> OptionsPortfolio:
    cons = GreeksNeutralConstraints(
        neutralize=list(neutralize) if neutralize is not None else ["delta", "vega"],
        gross_max=gross_max, max_position=max_position,
    )
    return OptionsPortfolioConstructor(cons).construct(legs)


def synthetic_option_book(
    underlying: str = "SPX",
    *,
    spot: float = 100.0,
    n_strikes: int = 7,
    expiries: Sequence[float] = (0.08, 0.25, 0.5),
    base_iv: float = 0.20,
    skew: float = 0.05,
    seed: int = 31,
) -> List[OptionLeg]:
    """Синтетический опционный «бук» (демо/тесты/no-data): сетка страйк×экспирация, call+put.

    IV с улыбкой/skew; альфа = VRP-прокси (богатые OTM-путы → положительная привлекательность).
    """
    rng = np.random.default_rng(seed)
    strikes = np.linspace(spot * 0.8, spot * 1.2, n_strikes)
    legs: List[OptionLeg] = []
    for T in expiries:
        for K in strikes:
            moneyness = (K - spot) / spot
            for is_call in (True, False):
                iv = base_iv + skew * abs(moneyness) - (skew * 0.5 * moneyness)  # smile + skew
                iv = float(max(iv, 0.05))
                # альфа-прокси: VRP богаче для OTM (выше |moneyness|) + лёгкий шум
                alpha = abs(moneyness) * 0.5 + 0.02 * rng.standard_normal()
                sym = f"{underlying}_{int(round(T*365))}d_{K:.0f}{'C' if is_call else 'P'}"
                legs.append(OptionLeg(symbol=sym, spot=spot, strike=float(K),
                                      time_to_expiry=float(T), iv=iv, is_call=is_call,
                                      alpha=float(alpha)))
    return legs


__all__ = [
    "ALL_GREEKS", "OptionLeg", "GreeksNeutralConstraints", "OptionsPortfolio",
    "OptionsPortfolioConstructor", "construct_options_portfolio", "synthetic_option_book",
]
