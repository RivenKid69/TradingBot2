# -*- coding: utf-8 -*-
"""
service_firm_risk.py
====================

Firm-wide **hierarchical** risk aggregator: strategy → desk → firm consolidated
VaR / CVaR with diversification benefit and Euler risk attribution.

Motivation (P0 gap from PRO_PIPELINE_GAP_ANALYSIS.md §4): each book today checks
its own slice in isolation (``service_pretrade_risk`` for XS, ``risk_guard`` /
``*_risk_guards`` per instrument). No component aggregates risk across the XS
portfolio, futures, forex and stock books into one consolidated VaR/CVaR against a
**hierarchical** (strategy→desk→firm) limit framework. This module is that
aggregator.

Academic grounding
------------------
* Coherent risk measures — Artzner, Delbaen, Eber, Heath (1999). ES/CVaR is
  *subadditive*: VaR(firm) ≤ Σ VaR(book); the gap is the **diversification
  benefit**. VaR itself is not coherent in general, so we report both VaR and the
  coherent ES/CVaR.
* CVaR / Expected Shortfall — Rockafellar & Uryasev (2000, 2002). Gaussian closed
  form ES = (φ(z_α)/α)·σ; historical ES = mean of the α-tail losses.
* Euler allocation / component risk — Tasche (1999, 2008), Litterman (1996,
  "Hot Spots & Hedges"). Risk measures positively homogeneous of degree 1 satisfy
  ρ(P) = Σ_i wᵢ ∂ρ/∂wᵢ, so the **component VaR/CVaR of each child sums exactly to
  the parent's** total. Component_c = (e_cᵀ Σ e) / σ · z_α  (parametric);
  Component_c = −E[ pnl_c | pnl_total in the α-tail ]  (historical).
* Incremental VaR — VaR(node) − VaR(node without child): the marginal contribution
  of adding/removing a whole sub-book.

Exposure model
--------------
Every position maps to a *risk unit* (a symbol or a common risk factor) with a
signed **dollar exposure** e = qty·price. Books map into a shared risk-unit space
(Σ or a returns matrix over those units), so cross-asset correlation/netting is
captured — futures, forex, equity and XS positions live in one covariance. The
hierarchy is built by grouping positions by (desk, strategy).

Engines
-------
* parametric (Gaussian) — needs Σ over risk units (return covariance, per horizon).
* historical — needs an aligned returns matrix R (T × units); empirical VaR/CVaR
  and exact tail attribution.

Layer ``service_`` — depends only on numpy/pandas (+ optional scipy via
``service_pretrade_risk`` helpers).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Reuse the validated quantile / pdf helpers (scipy-optional) from the pre-trade
# analyzer so the two risk surfaces stay numerically consistent.
from service_pretrade_risk import _z_alpha, _phi

logger = logging.getLogger(__name__)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class FirmPosition:
    """A single position contributing to firm risk.

    ``exposure`` is signed **dollar** market value (qty·price; long > 0, short < 0).
    ``risk_unit`` is the key into the covariance / returns space (defaults to
    ``symbol`` — set it to a factor name to aggregate cross-asset on factors).
    """

    symbol: str
    exposure: float
    desk: str = "default_desk"
    strategy: str = "default_strategy"
    risk_unit: Optional[str] = None
    sector: Optional[str] = None
    asset_class: Optional[str] = None

    @property
    def unit(self) -> str:
        return str(self.risk_unit or self.symbol)


@dataclass
class HierLimits:
    """Risk limits for a node (any ``None`` field is not checked).

    ``hard`` marks a limit whose breach should halt/block (vs a soft warning).
    """

    var: Optional[float] = None  # consolidated VaR (dollars) ≤ var
    cvar: Optional[float] = None  # consolidated CVaR (dollars) ≤ cvar
    gross: Optional[float] = None  # Σ|exposure| (dollars) ≤ gross
    net: Optional[float] = None  # |Σ exposure| (dollars) ≤ net
    var_pct: Optional[float] = None  # VaR / capital ≤ var_pct (needs capital)
    hard: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class LimitBreach:
    node: str
    metric: str
    value: float
    limit: float
    hard: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node,
            "metric": self.metric,
            "value": self.value,
            "limit": self.limit,
            "hard": self.hard,
            "message": f"{self.node}: {self.metric} {self.value:,.2f} > {self.limit:,.2f}"
            + (" [HARD]" if self.hard else " [soft]"),
        }


@dataclass
class ChildContribution:
    """Euler risk attribution of a child sub-book within its parent."""

    name: str
    standalone_var: float  # VaR if the child were the whole portfolio
    standalone_cvar: float
    component_var: float  # Euler component (Σ components = parent VaR)
    component_cvar: float  # Euler component (Σ components = parent CVaR)
    marginal_var: float  # ∂VaR per $ of net exposure in the child
    incremental_var: float  # VaR(parent) − VaR(parent without child)
    pct_var: float  # component_var / parent VaR
    gross: float
    net: float

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class NodeRisk:
    """Consolidated risk at a hierarchy node (firm / desk / strategy)."""

    name: str
    level: str  # "firm" | "desk" | "strategy"
    var: float
    cvar: float
    vol: float  # σ of P&L (dollars)
    gross: float
    net: float
    n_positions: int
    capital: Optional[float] = None
    var_pct: Optional[float] = None
    cvar_pct: Optional[float] = None
    diversification_benefit: float = 0.0  # Σ child standalone VaR − node VaR
    sector_exposure: Dict[str, float] = field(default_factory=dict)
    factor_exposure: Dict[str, float] = field(default_factory=dict)
    contributions: List[ChildContribution] = field(default_factory=list)
    breaches: List[LimitBreach] = field(default_factory=list)
    children: List["NodeRisk"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level,
            "var": self.var,
            "cvar": self.cvar,
            "vol": self.vol,
            "gross": self.gross,
            "net": self.net,
            "n_positions": self.n_positions,
            "capital": self.capital,
            "var_pct": self.var_pct,
            "cvar_pct": self.cvar_pct,
            "diversification_benefit": self.diversification_benefit,
            "sector_exposure": self.sector_exposure,
            "factor_exposure": self.factor_exposure,
            "contributions": [c.to_dict() for c in self.contributions],
            "breaches": [b.to_dict() for b in self.breaches],
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class FirmRiskReport:
    firm: NodeRisk
    alpha: float
    method: str  # "parametric" | "historical"
    approved: bool  # no HARD breaches anywhere
    breaches: List[LimitBreach]
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "firm": self.firm.to_dict(),
            "alpha": self.alpha,
            "method": self.method,
            "approved": self.approved,
            "breaches": [b.to_dict() for b in self.breaches],
            "metrics": self.metrics,
        }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
class FirmRiskAggregator:
    """Consolidates positions across books into a strategy→desk→firm risk tree.

    Parameters
    ----------
    cov : Σ over risk units (DataFrame index/cols = risk_unit, or ndarray + ``units``).
        Return covariance per the chosen horizon. Required for ``method='parametric'``.
    returns : returns matrix (DataFrame index=time, cols=risk_unit). Required for
        ``method='historical'`` (and enables exact tail attribution).
    exposures : optional factor-loadings B (index=risk_unit, cols=factor) for
        reporting consolidated factor exposures (Bᵀe) at each node.
    alpha : tail probability (0.05 → 95% VaR).
    units : explicit risk-unit order when ``cov`` is an ndarray.
    """

    def __init__(
        self,
        *,
        cov: Optional[Any] = None,
        returns: Optional[pd.DataFrame] = None,
        exposures: Optional[pd.DataFrame] = None,
        alpha: float = 0.05,
        units: Optional[Sequence[str]] = None,
    ) -> None:
        self.alpha = float(alpha)
        self._exposures = exposures
        self._returns = returns
        if isinstance(cov, pd.DataFrame):
            self._cov_df = cov.astype("float64")
        elif cov is not None:
            if units is None:
                raise ValueError("units required when cov is an ndarray")
            arr = np.asarray(cov, dtype="float64")
            self._cov_df = pd.DataFrame(arr, index=list(units), columns=list(units))
        elif returns is not None:
            # Estimate Σ from the supplied returns (sample covariance).
            self._cov_df = returns.astype("float64").cov()
        else:
            raise ValueError("either cov or returns must be provided")

    # ---- exposure vectors -------------------------------------------------
    def _exposure_vector(self, positions: Sequence[FirmPosition], units: List[str]) -> np.ndarray:
        idx = {u: i for i, u in enumerate(units)}
        e = np.zeros(len(units), dtype="float64")
        for p in positions:
            j = idx.get(p.unit)
            if j is not None:
                e[j] += float(p.exposure)
        return e

    def _sigma_for(self, units: List[str]) -> np.ndarray:
        S = self._cov_df.reindex(index=units, columns=units).to_numpy(dtype="float64")
        return np.nan_to_num(S, nan=0.0)

    # ---- core metrics on a dollar-exposure vector -------------------------
    def _parametric(self, e: np.ndarray, S: np.ndarray) -> Tuple[float, float, float]:
        var_pnl = float(e @ S @ e)
        sigma = math.sqrt(max(0.0, var_pnl))
        z = _z_alpha(self.alpha)
        var = z * sigma
        cvar = (_phi(z) / max(self.alpha, 1e-6)) * sigma
        return var, cvar, sigma

    def _historical_pnl(self, e: np.ndarray, units: List[str]) -> Optional[np.ndarray]:
        if self._returns is None:
            return None
        cols = [u for u in units if u in self._returns.columns]
        if not cols:
            return None
        R = self._returns[cols].to_numpy(dtype="float64")
        # align e to cols order
        idx = {u: i for i, u in enumerate(units)}
        ec = np.array([e[idx[u]] for u in cols], dtype="float64")
        pnl = R @ ec
        return pnl[np.isfinite(pnl)]

    def _historical(self, e: np.ndarray, units: List[str]) -> Optional[Tuple[float, float, float]]:
        pnl = self._historical_pnl(e, units)
        if pnl is None or pnl.size == 0:
            return None
        q = float(np.quantile(pnl, self.alpha))
        var = max(0.0, -q)
        tail = pnl[pnl <= q]
        cvar = max(0.0, -float(tail.mean())) if tail.size else var
        sigma = float(np.std(pnl, ddof=1)) if pnl.size > 1 else abs(var)
        return var, cvar, sigma

    def _node_metrics(
        self, e: np.ndarray, units: List[str], method: str
    ) -> Tuple[float, float, float]:
        if method == "historical":
            hist = self._historical(e, units)
            if hist is not None:
                return hist
        return self._parametric(e, self._sigma_for(units))

    # ---- Euler component attribution --------------------------------------
    def _component_parametric(
        self, e_parent: np.ndarray, child_vectors: List[np.ndarray], S: np.ndarray
    ) -> Tuple[List[float], List[float]]:
        """Euler component VaR/CVaR per child (Σ = parent VaR / CVaR exactly)."""
        sigma = math.sqrt(max(0.0, float(e_parent @ S @ e_parent)))
        if sigma <= _EPS:
            return [0.0] * len(child_vectors), [0.0] * len(child_vectors)
        z = _z_alpha(self.alpha)
        es_mult = _phi(z) / max(self.alpha, 1e-6)
        Se = S @ e_parent
        comp_var, comp_cvar = [], []
        for ec in child_vectors:
            cov_term = float(ec @ Se) / sigma  # ∂σ contribution of child
            comp_var.append(z * cov_term)
            comp_cvar.append(es_mult * cov_term)
        return comp_var, comp_cvar

    def _component_historical(
        self, e_parent: np.ndarray, child_vectors: List[np.ndarray], units: List[str]
    ) -> Optional[Tuple[List[float], List[float]]]:
        """Tail-scenario Euler attribution: component_c = −E[pnl_c | total in α-tail].

        Σ components = historical CVaR exactly; component VaR uses a small kernel of
        scenarios around the α-quantile (sums ≈ historical VaR).
        """
        if self._returns is None:
            return None
        cols = [u for u in units if u in self._returns.columns]
        if not cols:
            return None
        R = self._returns[cols].to_numpy(dtype="float64")
        idx = {u: i for i, u in enumerate(units)}
        ep = np.array([e_parent[idx[u]] for u in cols], dtype="float64")
        pnl = R @ ep
        good = np.isfinite(pnl)
        R, pnl = R[good], pnl[good]
        if pnl.size == 0:
            return None
        q = float(np.quantile(pnl, self.alpha))
        tail_mask = pnl <= q
        if not tail_mask.any():
            tail_mask = pnl <= np.quantile(pnl, max(self.alpha, 1.0 / pnl.size))
        # kernel around the quantile for VaR attribution (nearest ~max(1,5%·tail) pts)
        order = np.argsort(np.abs(pnl - q))
        k = max(1, int(0.1 * tail_mask.sum()) or 1)
        var_idx = order[:k]
        comp_var, comp_cvar = [], []
        for ec_full in child_vectors:
            ecc = np.array([ec_full[idx[u]] for u in cols], dtype="float64")
            pnl_c = R @ ecc
            comp_cvar.append(-float(pnl_c[tail_mask].mean()))
            comp_var.append(-float(pnl_c[var_idx].mean()))
        return comp_var, comp_cvar

    # ---- exposure breakdowns ----------------------------------------------
    @staticmethod
    def _gross_net(positions: Sequence[FirmPosition]) -> Tuple[float, float]:
        gross = float(sum(abs(p.exposure) for p in positions))
        net = float(sum(p.exposure for p in positions))
        return gross, net

    @staticmethod
    def _sector_exposure(positions: Sequence[FirmPosition]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for p in positions:
            if p.sector:
                out[p.sector] = out.get(p.sector, 0.0) + float(p.exposure)
        return out

    def _factor_exposure(self, e: np.ndarray, units: List[str]) -> Dict[str, float]:
        if self._exposures is None:
            return {}
        B = self._exposures.reindex(index=units).fillna(0.0)
        fexp = B.mul(pd.Series(e, index=units), axis=0).sum()
        return {str(k): float(v) for k, v in fexp.items()}

    # ---- limit checking ---------------------------------------------------
    @staticmethod
    def _check_limits(node: NodeRisk, limits: Optional[HierLimits]) -> List[LimitBreach]:
        if limits is None:
            return []
        out: List[LimitBreach] = []
        if limits.var is not None and node.var > limits.var + _EPS:
            out.append(LimitBreach(node.name, "VaR", node.var, limits.var, limits.hard))
        if limits.cvar is not None and node.cvar > limits.cvar + _EPS:
            out.append(LimitBreach(node.name, "CVaR", node.cvar, limits.cvar, limits.hard))
        if limits.gross is not None and node.gross > limits.gross + _EPS:
            out.append(LimitBreach(node.name, "gross", node.gross, limits.gross, limits.hard))
        if limits.net is not None and abs(node.net) > limits.net + _EPS:
            out.append(LimitBreach(node.name, "net", abs(node.net), limits.net, limits.hard))
        if (
            limits.var_pct is not None
            and node.var_pct is not None
            and node.var_pct > limits.var_pct + _EPS
        ):
            out.append(LimitBreach(node.name, "VaR%", node.var_pct, limits.var_pct, limits.hard))
        return out

    # ---- public API -------------------------------------------------------
    def aggregate(
        self,
        positions: Sequence[FirmPosition],
        *,
        method: str = "parametric",
        firm_name: str = "FIRM",
        limits: Optional[Dict[str, HierLimits]] = None,
        capital: Optional[Dict[str, float]] = None,
    ) -> FirmRiskReport:
        """Build the firm→desk→strategy risk tree with consolidated VaR/CVaR.

        Parameters
        ----------
        positions : all open positions across every book.
        method : "parametric" (Gaussian) or "historical" (empirical, needs returns).
        limits : {node_name: HierLimits} — node_name is the firm/desk/strategy name.
        capital : {node_name: capital$} for VaR%/CVaR% reporting and var_pct limits.
        """
        positions = [p for p in positions if abs(float(p.exposure)) > 0.0]
        units = sorted({p.unit for p in positions if p.unit in self._cov_df.index})
        limits = limits or {}
        capital = capital or {}
        all_breaches: List[LimitBreach] = []

        if not units or not positions:
            firm = NodeRisk(
                name=firm_name,
                level="firm",
                var=0.0,
                cvar=0.0,
                vol=0.0,
                gross=0.0,
                net=0.0,
                n_positions=0,
                capital=capital.get(firm_name),
            )
            return FirmRiskReport(
                firm=firm,
                alpha=self.alpha,
                method=method,
                approved=True,
                breaches=[],
                metrics={"n_units": 0},
            )

        S = self._sigma_for(units)

        def _build_leaf(name: str, level: str, pos: List[FirmPosition]) -> NodeRisk:
            e = self._exposure_vector(pos, units)
            var, cvar, vol = self._node_metrics(e, units, method)
            gross, net = self._gross_net(pos)
            cap = capital.get(name)
            node = NodeRisk(
                name=name,
                level=level,
                var=var,
                cvar=cvar,
                vol=vol,
                gross=gross,
                net=net,
                n_positions=len(pos),
                capital=cap,
                var_pct=(var / cap if cap else None),
                cvar_pct=(cvar / cap if cap else None),
                sector_exposure=self._sector_exposure(pos),
                factor_exposure=self._factor_exposure(e, units),
            )
            b = self._check_limits(node, limits.get(name))
            node.breaches = b
            all_breaches.extend(b)
            return node

        def _build_parent(
            name: str, level: str, child_groups: Dict[str, List[FirmPosition]], child_level: str
        ) -> NodeRisk:
            children: List[NodeRisk] = []
            child_pos_lists: List[List[FirmPosition]] = []
            for cname, cpos in sorted(child_groups.items()):
                if child_level == "strategy":
                    child = _build_leaf(cname, child_level, cpos)
                else:
                    # group cpos by next level (strategy) under this desk
                    sub: Dict[str, List[FirmPosition]] = {}
                    for p in cpos:
                        sub.setdefault(p.strategy, []).append(p)
                    child = _build_parent(cname, child_level, sub, "strategy")
                children.append(child)
                child_pos_lists.append(cpos)

            all_pos = [p for lst in child_pos_lists for p in lst]
            e_parent = self._exposure_vector(all_pos, units)
            var, cvar, vol = self._node_metrics(e_parent, units, method)
            gross, net = self._gross_net(all_pos)
            cap = capital.get(name)

            # Euler attribution per child sub-book
            child_vectors = [self._exposure_vector(lst, units) for lst in child_pos_lists]
            comp = None
            if method == "historical":
                comp = self._component_historical(e_parent, child_vectors, units)
            if comp is None:
                comp = self._component_parametric(e_parent, child_vectors, S)
            comp_var, comp_cvar = comp

            contributions: List[ChildContribution] = []
            standalone_sum = 0.0
            for ci, child in enumerate(children):
                cv = child_vectors[ci]
                c_gross, c_net = self._gross_net(child_pos_lists[ci])
                # incremental VaR = VaR(parent) − VaR(parent \ child)
                e_wo = e_parent - cv
                var_wo, _, _ = self._node_metrics(e_wo, units, method)
                incremental = var - var_wo
                marginal = (comp_var[ci] / c_net) if abs(c_net) > _EPS else 0.0
                standalone_sum += child.var
                contributions.append(
                    ChildContribution(
                        name=child.name,
                        standalone_var=child.var,
                        standalone_cvar=child.cvar,
                        component_var=comp_var[ci],
                        component_cvar=comp_cvar[ci],
                        marginal_var=marginal,
                        incremental_var=incremental,
                        pct_var=(comp_var[ci] / var if var > _EPS else 0.0),
                        gross=c_gross,
                        net=c_net,
                    )
                )

            node = NodeRisk(
                name=name,
                level=level,
                var=var,
                cvar=cvar,
                vol=vol,
                gross=gross,
                net=net,
                n_positions=len(all_pos),
                capital=cap,
                var_pct=(var / cap if cap else None),
                cvar_pct=(cvar / cap if cap else None),
                diversification_benefit=float(standalone_sum - var),
                sector_exposure=self._sector_exposure(all_pos),
                factor_exposure=self._factor_exposure(e_parent, units),
                contributions=contributions,
                children=children,
            )
            b = self._check_limits(node, limits.get(name))
            node.breaches = b
            all_breaches.extend(b)
            return node

        # group positions by desk
        by_desk: Dict[str, List[FirmPosition]] = {}
        for p in positions:
            by_desk.setdefault(p.desk, []).append(p)

        firm = _build_parent(firm_name, "firm", by_desk, "desk")
        approved = not any(b.hard for b in all_breaches)
        return FirmRiskReport(
            firm=firm,
            alpha=self.alpha,
            method=method,
            approved=approved,
            breaches=all_breaches,
            metrics={
                "n_units": len(units),
                "n_positions": len(positions),
                "n_desks": len(by_desk),
                "diversification_benefit": firm.diversification_benefit,
            },
        )


# ---------------------------------------------------------------------------
# Convenience: build positions from heterogeneous book payloads
# ---------------------------------------------------------------------------
def positions_from_books(books: Dict[str, Any]) -> List[FirmPosition]:
    """Build a flat ``FirmPosition`` list from a nested book payload.

    ``books`` shape (any book/desk maps a strategy -> list of position dicts)::

        {
          "equity_desk": {
            "momentum": [{"symbol": "AAPL", "exposure": 50000, "sector": "tech"}, ...],
            "meanrev":  [{"symbol": "XOM",  "exposure": -20000, "sector": "energy"}],
          },
          "futures_desk": {"trend": [{"symbol": "ES", "exposure": 120000}]},
        }

    Each position dict accepts: symbol, exposure (signed $), risk_unit, sector,
    asset_class. ``desk``/``strategy`` come from the nesting keys.
    """
    out: List[FirmPosition] = []
    for desk, strategies in (books or {}).items():
        if not isinstance(strategies, dict):
            continue
        for strat, plist in strategies.items():
            for pd_ in plist or []:
                if not isinstance(pd_, dict) or "symbol" not in pd_:
                    continue
                out.append(
                    FirmPosition(
                        symbol=str(pd_["symbol"]),
                        exposure=float(pd_.get("exposure", 0.0)),
                        desk=str(desk),
                        strategy=str(strat),
                        risk_unit=pd_.get("risk_unit"),
                        sector=pd_.get("sector"),
                        asset_class=pd_.get("asset_class"),
                    )
                )
    return out


__all__ = [
    "FirmPosition",
    "HierLimits",
    "LimitBreach",
    "ChildContribution",
    "NodeRisk",
    "FirmRiskReport",
    "FirmRiskAggregator",
    "positions_from_books",
]
