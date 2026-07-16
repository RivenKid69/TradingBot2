# -*- coding: utf-8 -*-
"""
service_attribution.py
======================

Attribution (Stage A10): разложение реализованного P&L по источникам — для LP-отчётности
и compliance-evidence (MiFID/AI-Act).

* **Factor P&L attribution**: ``r = B·f + u`` ⇒ доходность портфеля
  ``wᵀr = (wᵀB)·f + wᵀu`` = Σ(factor contributions) + specific. Разложение **точное**
  по построению (``u = r − B·f``) — сумма факторных вкладов + specific = полный P&L.
* **Signal attribution**: автономный P&L long-short портфеля по каждому сигналу
  (factor-mimicking), с Sharpe.
* **Brinson** allocation/selection/interaction относительно бенчмарка (tie-out к active).
* **tear_sheet**: сводный JSON для инвесторского отчёта / evidence-экспорта.

Слой ``service_``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from core_portfolio import SYMBOL_LEVEL, TS_LEVEL
from core_xs_results import compute_metrics


# ---------------------------------------------------------------------------
# Factor P&L attribution (точное разложение)
# ---------------------------------------------------------------------------
def factor_attribution(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    exposures: pd.DataFrame,
) -> Dict[str, Any]:
    """Разложить P&L портфеля на факторные вклады + specific.

    ``weights`` / ``asset_returns``: index=rebalance ts, columns=symbol (веса в начале
    периода и реализованная доходность за период). ``exposures`` (B): index=symbol,
    columns=factor. Тождество: Σ factor_contrib + specific = wᵀr (точно).
    """
    factors: List[str] = list(exposures.columns)
    rows: List[Dict[str, Any]] = []
    fac_total = {f: 0.0 for f in factors}
    spec_total = 0.0
    tot_total = 0.0

    for t in weights.index:
        if t not in asset_returns.index:
            continue
        w = weights.loc[t].dropna()
        r = asset_returns.loc[t]
        common = [
            s for s in w.index
            if s in r.index and s in exposures.index
            and np.isfinite(r[s]) and np.isfinite(w[s])
        ]
        if len(common) < len(factors):
            continue
        wv = w.loc[common].to_numpy(dtype="float64")
        rv = r.loc[common].to_numpy(dtype="float64")
        B = exposures.loc[common, factors].to_numpy(dtype="float64")
        f, *_ = np.linalg.lstsq(B, rv, rcond=None)   # факторные доходности периода
        u = rv - B @ f                                # specific (по построению)
        pexp = wv @ B                                 # экспозиция портфеля к факторам
        fac_contrib = pexp * f
        spec = float(wv @ u)
        tot = float(wv @ rv)

        row: Dict[str, Any] = {"ts": int(t)}
        for k, fac in enumerate(factors):
            row[fac] = float(fac_contrib[k])
            fac_total[fac] += float(fac_contrib[k])
        row["specific"] = spec
        row["total"] = tot
        rows.append(row)
        spec_total += spec
        tot_total += tot

    per = pd.DataFrame(rows).set_index("ts") if rows else pd.DataFrame()
    resid = tot_total - (sum(fac_total.values()) + spec_total)
    return {
        "factor_pnl": fac_total,
        "specific_pnl": spec_total,
        "total_pnl": tot_total,
        "tie_out_residual": float(resid),
        "per_period": per,
    }


# ---------------------------------------------------------------------------
# Signal attribution (standalone long-short)
# ---------------------------------------------------------------------------
def signal_attribution(
    signal_panel: pd.DataFrame,
    asset_returns: pd.Series,
    *,
    periods_per_year: float = 252.0,
) -> Dict[str, Dict[str, float]]:
    """P&L автономного long-short портфеля по каждому сигналу (factor-mimicking).

    ``asset_returns`` — MultiIndex (ts, symbol) Series реализованных доходностей,
    выровненных с панелью сигналов.
    """
    def _ls_weights(g: pd.Series) -> pd.Series:
        d = g - g.mean()
        denom = d.abs().sum()
        return d / denom if denom > 0 else d * 0.0

    out: Dict[str, Dict[str, float]] = {}
    for col in signal_panel.columns:
        s = signal_panel[col].replace([np.inf, -np.inf], np.nan)
        w = s.groupby(level=TS_LEVEL, group_keys=False).apply(_ls_weights)
        contrib = (w * asset_returns).dropna()
        per_period = contrib.groupby(level=TS_LEVEL).sum()
        per_period = per_period.replace([np.inf, -np.inf], np.nan).dropna()
        if len(per_period) == 0:
            out[col] = {"total_pnl": float("nan"), "sharpe": float("nan"), "n_periods": 0}
            continue
        sd = float(per_period.std(ddof=0))
        sharpe = (float(per_period.mean()) / sd * math.sqrt(periods_per_year)) if sd > 0 else float("nan")
        out[col] = {
            "total_pnl": float(per_period.sum()),
            "sharpe": float(sharpe),
            "n_periods": int(len(per_period)),
        }
    return out


# ---------------------------------------------------------------------------
# Brinson attribution
# ---------------------------------------------------------------------------
def brinson_attribution(
    port_weights: pd.Series,
    bench_weights: pd.Series,
    port_returns: pd.Series,
    bench_returns: pd.Series,
) -> Dict[str, Any]:
    """Brinson-Hood-Beebower: active return = allocation + selection + interaction.

    Все входы — Series по группам (секторам). Tie-out выполняется при Σw_p = Σw_b.
    """
    groups = sorted(set(port_weights.index) | set(bench_weights.index))
    wp = port_weights.reindex(groups).fillna(0.0)
    wb = bench_weights.reindex(groups).fillna(0.0)
    rp = port_returns.reindex(groups).fillna(0.0)
    rb = bench_returns.reindex(groups).fillna(0.0)

    rb_total = float((wb * rb).sum())
    allocation = (wp - wb) * (rb - rb_total)
    selection = wb * (rp - rb)
    interaction = (wp - wb) * (rp - rb)

    total_active = float((wp * rp).sum() - (wb * rb).sum())
    decomposed = float(allocation.sum() + selection.sum() + interaction.sum())
    return {
        "allocation": allocation.to_dict(),
        "selection": selection.to_dict(),
        "interaction": interaction.to_dict(),
        "allocation_total": float(allocation.sum()),
        "selection_total": float(selection.sum()),
        "interaction_total": float(interaction.sum()),
        "total_active": total_active,
        "tie_out_residual": float(total_active - decomposed),
    }


# ---------------------------------------------------------------------------
# Tear sheet (JSON evidence)
# ---------------------------------------------------------------------------
def tear_sheet(
    returns: pd.Series,
    *,
    periods_per_year: float = 252.0,
    factor: Optional[Dict[str, Any]] = None,
    signal: Optional[Dict[str, Any]] = None,
    brinson: Optional[Dict[str, Any]] = None,
    trust: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Сводный JSON-отчёт для LP / compliance evidence."""
    sheet: Dict[str, Any] = {"metrics": compute_metrics(returns, periods_per_year=periods_per_year)}
    if factor is not None:
        sheet["factor_attribution"] = {
            "factor_pnl": factor.get("factor_pnl"),
            "specific_pnl": factor.get("specific_pnl"),
            "total_pnl": factor.get("total_pnl"),
            "tie_out_residual": factor.get("tie_out_residual"),
        }
    if signal is not None:
        sheet["signal_attribution"] = signal
    if brinson is not None:
        sheet["brinson"] = {
            k: v for k, v in brinson.items() if k.endswith("_total") or k in ("total_active", "tie_out_residual")
        }
    if trust is not None:
        sheet["trust_report"] = trust
    if extra:
        sheet.update(extra)
    return sheet


__all__ = [
    "factor_attribution",
    "signal_attribution",
    "brinson_attribution",
    "tear_sheet",
]
