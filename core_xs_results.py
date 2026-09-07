# -*- coding: utf-8 -*-
"""
core_xs_results.py
==================

Структуры результатов cross-sectional бэктеста (Stage A8) и расчёт метрик
производительности (Sharpe, max drawdown, turnover, total return, vol).

Слой ``core_`` (без тяжёлых зависимостей).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def compute_metrics(
    returns: pd.Series,
    *,
    periods_per_year: float = 252.0,
    rf_per_period: float = 0.0,
    benchmark: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Метрики по серии доходностей за период (net of costs).

    Adds downside-aware metrics (Sortino, Calmar) and, when ``benchmark`` is
    provided, benchmark-relative analytics (tracking error, information ratio,
    beta, Jensen's alpha) — what allocators ask for after Sharpe.
    """
    r = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) == 0:
        return {
            "n_periods": 0,
            "total_return": float("nan"),
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "calmar": float("nan"),
            "ann_return": float("nan"),
            "ann_vol": float("nan"),
            "max_drawdown": float("nan"),
            "hit_rate": float("nan"),
            "mean_return": float("nan"),
        }
    ppy = float(periods_per_year)
    nav = (1.0 + r).cumprod()
    excess = r - float(rf_per_period)
    mean = float(r.mean())
    std = float(r.std(ddof=0))
    sharpe = (float(excess.mean()) / std * math.sqrt(ppy)) if std > 0 else float("nan")
    # Sortino: downside deviation (only negative excess returns) in the denominator.
    downside = excess.clip(upper=0.0)
    dd_std = float(math.sqrt((downside.pow(2)).mean()))
    sortino = (float(excess.mean()) / dd_std * math.sqrt(ppy)) if dd_std > 0 else float("nan")
    # max drawdown по NAV с ведущей единицей
    full = np.concatenate([[1.0], nav.to_numpy()])
    peak = np.maximum.accumulate(full)
    max_dd = float((full / peak - 1.0).min())
    ann_return = float(mean * ppy)
    # Calmar: annualized return / |max drawdown|.
    calmar = (ann_return / abs(max_dd)) if max_dd < 0 else float("nan")

    out = {
        "n_periods": int(len(r)),
        "total_return": float(nav.iloc[-1] - 1.0),
        "mean_return": mean,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "ann_return": ann_return,
        "ann_vol": float(std * math.sqrt(ppy)),
        "max_drawdown": max_dd,
        "hit_rate": float((r > 0).mean()),
    }

    if benchmark is not None:
        b = pd.Series(benchmark, dtype="float64").replace([np.inf, -np.inf], np.nan)
        aligned = pd.concat([r.rename("r"), b.rename("b")], axis=1, join="inner").dropna()
        if len(aligned) >= 2 and float(aligned["b"].std(ddof=0)) > 0:
            ar, br = aligned["r"], aligned["b"]
            active = ar - br
            te = float(active.std(ddof=0)) * math.sqrt(ppy)  # tracking error
            ir = (
                float(active.mean()) / float(active.std(ddof=0)) * math.sqrt(ppy)
                if float(active.std(ddof=0)) > 0
                else float("nan")
            )  # information ratio
            var_b = float(br.var(ddof=0))
            beta = float(np.cov(ar, br, ddof=0)[0, 1] / var_b) if var_b > 0 else float("nan")
            # Jensen's alpha (annualized): a = (E[r]-rf) - beta*(E[b]-rf)
            alpha_p = float(ar.mean() - rf_per_period) - beta * float(br.mean() - rf_per_period)
            out.update(
                {
                    "benchmark_ann_return": float(br.mean() * ppy),
                    "tracking_error": te,
                    "information_ratio": float(ir),
                    "beta": beta,
                    "alpha": float(alpha_p * ppy),
                }
            )
    return out


@dataclass
class XSBacktestResult:
    """Результат cross-sectional бэктеста."""

    returns: pd.Series  # доходность портфеля за период (net of costs), index=rebalance ts
    weights: pd.DataFrame  # целевые веса: index=rebalance ts, columns=symbol
    turnover: pd.Series  # оборот за период
    costs: pd.Series  # издержки за период
    gross: pd.Series  # gross exposure Σ|w|
    net: pd.Series  # net exposure Σw
    nav: pd.Series  # кривая капитала (cumprod(1+returns))
    metrics: Dict[str, float] = field(default_factory=dict)
    benchmark: Optional[pd.Series] = None  # benchmark return series (equal-weight universe)
    meta: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            **self.metrics,
            "avg_turnover": float(self.turnover.mean()) if len(self.turnover) else float("nan"),
            "avg_gross": float(self.gross.mean()) if len(self.gross) else float("nan"),
            "total_costs": float(self.costs.sum()) if len(self.costs) else 0.0,
            "n_rebalances": int(self.weights.shape[0]),
        }


__all__ = ["compute_metrics", "XSBacktestResult"]
