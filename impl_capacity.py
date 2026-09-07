# -*- coding: utf-8 -*-
"""
impl_capacity.py
================

Capacity-анализ (Stage A9): на каком AUM стратегия деградирует из-за market impact.
Первый вопрос аллокатора после Sharpe — «какая ёмкость?».

Издержки масштабируются с размером через √participation impact (Almgren-Chriss). По
умолчанию используется встроенная √-модель; опционально подключается
``lob.market_impact`` (роадмап). Кривая: AUM → annualized Sharpe (после impact-костов)
и avg cost (bps). ``capacity_aum`` — наибольший AUM, при котором Sharpe ≥ порога.

Слой ``impl_``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class CapacityPoint:
    aum: float
    sharpe: float
    avg_cost_bps: float


def _sqrt_impact_fraction(participation: np.ndarray, impact_coef: float) -> np.ndarray:
    """Impact как доля цены: coef·√participation (Almgren-Chriss temporary impact)."""
    return impact_coef * np.sqrt(np.clip(participation, 0.0, None))


def _ann_sharpe(returns: np.ndarray, ppy: float) -> float:
    r = returns[np.isfinite(returns)]
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=0))
    if sd == 0:
        return float("nan")
    return float(r.mean() / sd * math.sqrt(ppy))


def capacity_curve(
    gross_returns: Sequence[float],
    turnover: Sequence[float],
    *,
    adv_usd: float,
    aum_grid: Sequence[float],
    impact_coef: float = 0.1,
    periods_per_year: float = 252.0,
    sharpe_threshold_frac: float = 0.5,
    impact_model: Optional[Any] = None,
) -> Dict[str, Any]:
    """Кривая ёмкости: AUM → Sharpe (после impact-костов) и avg cost (bps).

    ``gross_returns`` — доходности ДО торговых издержек; ``turnover`` — Σ|Δw| за период
    (доля капитала). ``adv_usd`` — средний дневной оборот инструмента (USD), грубая
    оценка ликвидности юниверса. ``impact_model`` — опц. объект с ``.impact_fraction``.
    """
    g = np.asarray(gross_returns, dtype="float64")
    to = np.asarray(turnover, dtype="float64")
    n = min(len(g), len(to))
    g, to = g[:n], to[:n]
    ppy = float(periods_per_year)

    base_sharpe = _ann_sharpe(g, ppy)  # AUM → 0 (без костов)

    curve: List[CapacityPoint] = []
    for aum in aum_grid:
        traded_usd = to * float(aum)
        participation = traded_usd / max(adv_usd, 1e-9)
        if impact_model is not None and hasattr(impact_model, "impact_fraction"):
            impact_frac = np.asarray(
                [float(impact_model.impact_fraction(p)) for p in participation], dtype="float64"
            )
        else:
            impact_frac = _sqrt_impact_fraction(participation, impact_coef)
        cost = impact_frac * to  # платим impact на торгуемую долю
        net = g - cost
        curve.append(
            CapacityPoint(
                aum=float(aum),
                sharpe=_ann_sharpe(net, ppy),
                avg_cost_bps=float(np.nanmean(cost) * 1e4),
            )
        )

    threshold = (base_sharpe * sharpe_threshold_frac) if np.isfinite(base_sharpe) else float("nan")
    capacity_aum = 0.0
    if np.isfinite(threshold):
        for p in curve:
            if np.isfinite(p.sharpe) and p.sharpe >= threshold:
                capacity_aum = p.aum

    return {
        "base_sharpe": float(base_sharpe),
        "threshold_sharpe": float(threshold),
        "sharpe_threshold_frac": float(sharpe_threshold_frac),
        "capacity_aum": float(capacity_aum),
        "adv_usd": float(adv_usd),
        "impact_coef": float(impact_coef),
        "curve": [p.__dict__ for p in curve],
    }


def capacity_from_result(
    result: Any,
    *,
    adv_usd: float,
    aum_grid: Sequence[float],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Удобная обёртка: capacity из ``XSBacktestResult`` (gross = net + costs)."""
    net = result.returns
    costs = (
        result.costs.reindex(net.index).fillna(0.0) if len(getattr(result, "costs", [])) else 0.0
    )
    gross = (net + costs) if not isinstance(costs, float) else net
    turnover = result.turnover.reindex(net.index).fillna(0.0)
    return capacity_curve(
        gross.to_numpy(), turnover.to_numpy(), adv_usd=adv_usd, aum_grid=aum_grid, **kwargs
    )


__all__ = ["CapacityPoint", "capacity_curve", "capacity_from_result"]
