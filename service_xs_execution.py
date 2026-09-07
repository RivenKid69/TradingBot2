# -*- coding: utf-8 -*-
"""
service_xs_execution.py
=======================

Портфельный execution-scheduler (P1): целевые веса w_target и текущие w_current →
trade-list (дельты по символам) → нарезка на child-slices алгоритмами TWAP / VWAP / POV
поверх impact-моделей (Almgren-Chriss sqrt-participation). Это «середина» между
optimizer (даёт целевые веса) и Agent-исполнением (создаёт ордера локально, CCEA):
Cloud отдаёт schedule намерений, ордера создаёт Agent.

Импакт-стоимость на slice: half_spread + k*sqrt(участие_слайса); нарезка на N слайсов
снижает участие на слайс -> импакт падает ~1/sqrt(N) (зачем и нужна нарезка). POV
ограничивает участие на слайс -> больше слайсов на крупных заявках.

DI: impact_fn(participation)->bps подменяет дефолтную AC-модель (для реального
lob.market_impact). Single-order алгоритмы — в execution_algos (TWAP/VWAP/POV
executors); здесь — ПОРТФЕЛЬНЫЙ слой над ними. Слой service_.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ChildSlice:
    symbol: str
    side: str  # BUY | SELL
    qty: float  # абсолютное кол-во в этом слайсе
    notional: float  # |qty|*price
    slice_index: int
    n_slices: int
    weight: float  # доля родительской заявки
    est_cost_bps: float  # импакт+спред этого слайса (bps)


@dataclass
class SymbolTrade:
    symbol: str
    side: str
    qty: float  # абсолютное кол-во (родитель)
    notional: float  # |qty|*price
    participation: float  # |notional| / adv
    est_cost_bps: float  # средневзвешенный импакт+спред (bps)
    est_cost: float  # стоимость в валюте
    slices: List[ChildSlice] = field(default_factory=list)


@dataclass
class RebalancePlan:
    algo: str
    n_slices: int
    trades: List[SymbolTrade]
    total_notional: float
    total_est_cost: float
    total_est_cost_bps: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algo": self.algo,
            "n_slices": self.n_slices,
            "total_notional": self.total_notional,
            "total_est_cost": self.total_est_cost,
            "total_est_cost_bps": self.total_est_cost_bps,
            "trades": [
                {
                    "symbol": t.symbol,
                    "side": t.side,
                    "qty": t.qty,
                    "notional": t.notional,
                    "participation": t.participation,
                    "est_cost_bps": t.est_cost_bps,
                    "est_cost": t.est_cost,
                    "n_slices": len(t.slices),
                    "slices": [s.__dict__ for s in t.slices],
                }
                for t in self.trades
            ],
        }


def _ushape_profile(n: int) -> np.ndarray:
    """U-образный внутридневной профиль объёма (открытие/закрытие тяжелее середины)."""
    if n <= 1:
        return np.ones(max(n, 1))
    x = np.linspace(0.0, 1.0, n)
    prof = 1.0 + 1.2 * (np.cos(2.0 * np.pi * x) * 0.5 + 0.5)  # макс на краях
    return prof / prof.sum()


class RebalanceScheduler:
    """target/current веса -> trade-list -> impact-aware child-slices (TWAP/VWAP/POV)."""

    def __init__(
        self,
        *,
        algo: str = "TWAP",
        n_slices: int = 6,
        participation: float = 0.10,  # POV: целевое участие на слайс
        spread_bps: float = 2.0,  # half-spread (bps) добавляется к импакту
        impact_coef: float = 0.1,  # k в AC: bps = k*sqrt(participation)*1e4
        max_pov_slices: int = 50,
        min_trade_notional: float = 0.0,  # игнор мелких дельт
        volume_profile: Optional[Sequence[float]] = None,
        impact_fn: Optional[Callable[[float], float]] = None,  # participation->bps (DI)
        urgency: float = 2.0,  # IS: κ·T (0 → TWAP; больше → быстрее в начале)
    ) -> None:
        self.algo = str(algo).upper()
        self.n_slices = int(n_slices)
        self.participation = float(participation)
        self.spread_bps = float(spread_bps)
        self.impact_coef = float(impact_coef)
        self.max_pov_slices = int(max_pov_slices)
        self.min_trade_notional = float(min_trade_notional)
        self.volume_profile = (
            np.asarray(volume_profile, dtype="float64") if volume_profile is not None else None
        )
        self._impact_fn = impact_fn
        self.urgency = float(urgency)

    # ---- impact (Almgren-Chriss sqrt-participation), bps ----
    def _impact_bps(self, participation: float) -> float:
        if self._impact_fn is not None:
            return float(self._impact_fn(max(0.0, participation)))
        half_spread = 0.5 * self.spread_bps
        return half_spread + self.impact_coef * math.sqrt(max(0.0, participation)) * 1e4

    # ---- Almgren-Chriss IS trade profile (front-loaded) ----
    def _is_profile(self, n: int) -> np.ndarray:
        """Implementation-Shortfall slice weights — Almgren & Chriss (2000).

        Risk-averse optimal trajectory holds x(t)=sinh(κ(T−t))/sinh(κT); the trade in
        interval k is the decrement x_{k−1}−x_k, which **front-loads** execution to cut
        timing risk. κ·T is the dimensionless ``urgency``; →0 recovers TWAP (uniform)."""
        n = max(1, int(n))
        kT = max(0.0, float(self.urgency))
        if kT < 1e-6 or n == 1:
            return np.ones(n) / n
        t = np.linspace(0.0, 1.0, n + 1)  # fractional time grid
        x = np.sinh(kT * (1.0 - t)) / math.sinh(kT)  # remaining holdings (x0=1, xn=0)
        trades = x[:-1] - x[1:]  # per-interval traded fraction
        s = float(trades.sum())
        return trades / s if s > 0 else np.ones(n) / n

    # ---- slice weights per algo ----
    def _slice_weights(self, participation: float) -> np.ndarray:
        algo = self.algo
        if algo in ("IS", "IMPLEMENTATION_SHORTFALL", "ALMGREN_CHRISS", "AC"):
            return self._is_profile(self.n_slices)
        if algo == "POV":
            # столько слайсов, чтобы участие на слайс <= target participation
            if self.participation <= 0:
                k = self.n_slices
            else:
                k = int(
                    min(self.max_pov_slices, max(1, math.ceil(participation / self.participation)))
                )
            return np.ones(k) / k
        if algo == "VWAP":
            prof = (
                self.volume_profile
                if self.volume_profile is not None
                else _ushape_profile(self.n_slices)
            )
            prof = np.asarray(prof, dtype="float64")
            return prof / prof.sum()
        # TWAP (default): равные слайсы
        n = max(1, self.n_slices)
        return np.ones(n) / n

    def build_plan(
        self,
        w_target: pd.Series,
        w_current: Optional[pd.Series],
        prices: pd.Series,
        equity: float,
        *,
        adv: Optional[pd.Series] = None,
    ) -> RebalancePlan:
        """Построить план ребаланса с нарезкой и оценкой импакта."""
        wt = w_target.astype("float64").dropna()
        w0 = (
            (w_current if w_current is not None else pd.Series(0.0, index=wt.index))
            .reindex(wt.index)
            .fillna(0.0)
        )
        px = prices.reindex(wt.index)
        equity = float(equity)

        trades: List[SymbolTrade] = []
        total_notional = 0.0
        total_cost = 0.0

        for sym in wt.index:
            dw = float(wt[sym] - w0[sym])
            price = float(px.get(sym, float("nan")))
            if not math.isfinite(price) or price <= 0:
                continue
            notional = dw * equity
            if abs(notional) < self.min_trade_notional:
                continue
            qty = abs(notional) / price
            side = "BUY" if dw > 0 else "SELL"
            adv_v = (
                float(adv[sym])
                if (adv is not None and sym in adv.index and float(adv[sym]) > 0)
                else 0.0
            )
            participation = (abs(notional) / adv_v) if adv_v > 0 else 0.0

            weights = self._slice_weights(participation)
            n = len(weights)
            slices: List[ChildSlice] = []
            sym_cost = 0.0
            for i, wgt in enumerate(weights):
                slice_notional = abs(notional) * float(wgt)
                slice_part = (slice_notional / adv_v) if adv_v > 0 else 0.0
                cbps = self._impact_bps(slice_part)
                slice_cost = slice_notional * cbps / 1e4
                sym_cost += slice_cost
                slices.append(
                    ChildSlice(
                        symbol=sym,
                        side=side,
                        qty=qty * float(wgt),
                        notional=slice_notional,
                        slice_index=i,
                        n_slices=n,
                        weight=float(wgt),
                        est_cost_bps=cbps,
                    )
                )
            eff_bps = (sym_cost / abs(notional) * 1e4) if abs(notional) > 0 else 0.0
            trades.append(
                SymbolTrade(
                    symbol=sym,
                    side=side,
                    qty=qty,
                    notional=abs(notional),
                    participation=participation,
                    est_cost_bps=eff_bps,
                    est_cost=sym_cost,
                    slices=slices,
                )
            )
            total_notional += abs(notional)
            total_cost += sym_cost

        total_bps = (total_cost / total_notional * 1e4) if total_notional > 0 else 0.0
        return RebalancePlan(
            algo=self.algo,
            n_slices=self.n_slices,
            trades=trades,
            total_notional=total_notional,
            total_est_cost=total_cost,
            total_est_cost_bps=total_bps,
        )


__all__ = ["ChildSlice", "SymbolTrade", "RebalancePlan", "RebalanceScheduler"]
