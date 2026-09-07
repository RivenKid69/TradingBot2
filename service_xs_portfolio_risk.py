# -*- coding: utf-8 -*-
"""
service_xs_portfolio_risk.py
============================

Portfolio-level риск-гарды (Stage A11): pre-trade проверка целевого вектора весов и
trade-list ПЕРЕД отправкой Intent'ов в Agent. Контролируются gross/net экспозиции,
концентрация по имени и сектору, факторные экспозиции (Bᵀw), оборот.

Опционально дополняется ``services.unified_futures_risk.PortfolioRiskManager`` (мягкая
интеграция). Это «облачный» pre-trade слой — Agent дополнительно применяет локальный
hard-cap firewall (CCEA). Слой ``service_``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TOL = 1e-9


@dataclass
class PortfolioRiskLimits:
    gross_max: Optional[float] = None  # Σ|w| ≤ gross_max
    net_max: Optional[float] = None  # |Σw| ≤ net_max
    max_position: Optional[float] = None  # max|w_i| ≤ max_position
    max_sector: Optional[float] = None  # |экспозиция сектора| ≤ max_sector
    sector_map: Optional[Dict[str, str]] = None
    factor_caps: Optional[Dict[str, float]] = None  # |(Bᵀw)_f| ≤ cap
    exposures: Optional[pd.DataFrame] = None  # B: index=symbol, cols=factor
    max_turnover: Optional[float] = None  # Σ|w − w₀| ≤ max_turnover


@dataclass
class RiskDecision:
    approved: bool
    violations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "violations": list(self.violations),
            "metrics": self.metrics,
        }


class PortfolioRiskGuard:
    """Pre-trade portfolio-level guard."""

    def __init__(
        self,
        limits: Optional[PortfolioRiskLimits] = None,
        *,
        strict: bool = True,
        manager: Any = None,
    ) -> None:
        self.limits = limits or PortfolioRiskLimits()
        self.strict = strict
        self._manager = manager  # опц. unified_futures_risk.PortfolioRiskManager

    def check(
        self,
        target_weights: pd.Series,
        current_weights: Optional[pd.Series] = None,
    ) -> RiskDecision:
        L = self.limits
        w = target_weights.astype("float64").dropna()
        viol: List[str] = []
        m: Dict[str, Any] = {}

        gross = float(w.abs().sum())
        net = float(w.sum())
        m["gross"] = gross
        m["net"] = net
        if L.gross_max is not None and gross > L.gross_max + _TOL:
            viol.append(f"gross {gross:.4f} > {L.gross_max}")
        if L.net_max is not None and abs(net) > L.net_max + _TOL:
            viol.append(f"|net| {abs(net):.4f} > {L.net_max}")

        if L.max_position is not None and len(w):
            mx = float(w.abs().max())
            m["max_position"] = mx
            if mx > L.max_position + _TOL:
                worst = w.abs().idxmax()
                viol.append(f"position {worst} {mx:.4f} > {L.max_position}")

        if L.sector_map and L.max_sector is not None and len(w):
            sect = w.groupby(w.index.map(lambda s: L.sector_map.get(s, "UNKNOWN"))).sum()
            m["sector_exposure"] = {str(k): float(v) for k, v in sect.items()}
            for s, v in sect.items():
                if abs(float(v)) > L.max_sector + _TOL:
                    viol.append(f"sector {s} exposure {float(v):.4f} > {L.max_sector}")

        if L.factor_caps and L.exposures is not None and len(w):
            B = L.exposures.reindex(w.index).fillna(0.0)
            fexp = B.mul(w, axis=0).sum()
            m["factor_exposure"] = {str(k): float(v) for k, v in fexp.items()}
            for f, cap in L.factor_caps.items():
                if f in fexp.index and abs(float(fexp[f])) > cap + _TOL:
                    viol.append(f"factor {f} exposure {float(fexp[f]):.4f} > {cap}")

        if L.max_turnover is not None and current_weights is not None:
            w0 = current_weights.reindex(w.index).fillna(0.0)
            to = float((w - w0).abs().sum())
            m["turnover"] = to
            if to > L.max_turnover + _TOL:
                viol.append(f"turnover {to:.4f} > {L.max_turnover}")

        # опциональная мягкая проверка через unified manager
        if self._manager is not None:
            try:  # pragma: no cover - зависит от наличия менеджера
                extra = self._manager.check_portfolio_weights(w)
                if extra:
                    viol.extend([str(x) for x in extra])
            except Exception as exc:  # pragma: no cover
                logger.debug("unified manager check skipped: %s", exc)

        approved = (len(viol) == 0) if self.strict else True
        return RiskDecision(approved=approved, violations=viol, metrics=m)


__all__ = ["PortfolioRiskLimits", "RiskDecision", "PortfolioRiskGuard"]
