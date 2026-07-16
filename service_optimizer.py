# -*- coding: utf-8 -*-
"""
service_optimizer.py
====================

Портфельный оптимизатор (Stage A7): μ + Σ + ограничения → целевые веса **w***.
Реализация контракта ``core_portfolio.PortfolioConstructor``.

Режимы (``objective``):
  * ``equal_weight``     — 1/N (baseline);
  * ``min_variance``     — min wᵀΣw;
  * ``mean_variance``    — max μᵀw − λ·wᵀΣw  (направление ∝ Σ⁻¹μ);
  * ``max_sharpe``       — tangency-портфель (∝ Σ⁻¹μ, нормировка масштабом);
  * ``risk_parity``      — equal risk contribution;
  * ``black_litterman``  — BL-постериор μ, затем mean-variance.

Солвер: при наличии ``cvxpy`` — выпуклая задача с неравенствами; иначе (как сейчас в
окружении) — **аналитическое решение + итеративная проекция** на жёсткие ограничения
(gross/net/box/turnover/long-only). Sector/factor-tilt лимиты точно проецируются только
через cvxpy; в fallback применяется best-effort и пишется предупреждение.

Ограничения берутся из ``OptimizerConstraints`` (мост к
``services.portfolio_constraints``). Слой ``service_``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:  # pragma: no cover - окружение без cvxpy
    import cvxpy as _cp  # type: ignore

    _HAS_CVXPY = True
except Exception:  # pragma: no cover
    _cp = None
    _HAS_CVXPY = False


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------
@dataclass
class OptimizerConstraints:
    """Ограничения оптимизатора (жёсткие проецируемые + tilt-лимиты для cvxpy)."""

    gross_max: Optional[float] = None        # Σ|w| ≤ gross_max
    net_target: Optional[float] = None       # Σw = net_target
    long_only: bool = False                  # w ≥ 0
    max_position: Optional[float] = None      # верхняя граница |w_i|
    min_position: Optional[float] = None      # нижняя граница w_i (если задана)
    max_turnover: Optional[float] = None     # Σ|w − w₀| ≤ max_turnover
    # tilt-лимиты (точно — только cvxpy)
    sector_map: Optional[Dict[str, str]] = None
    sector_caps: Optional[Dict[str, float]] = None
    exposures: Optional[pd.DataFrame] = None  # B: index=symbol, cols=factor
    factor_caps: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------
def _solve_psd(cov: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    tr = float(np.trace(cov)) / max(1, n)
    ridge = 1e-10 * max(tr, 1.0)
    try:
        return np.linalg.solve(cov + ridge * np.eye(n), b)
    except np.linalg.LinAlgError:  # pragma: no cover
        return np.linalg.pinv(cov) @ b


def _risk_parity(cov: np.ndarray, *, iters: int = 4000, tol: float = 1e-12) -> np.ndarray:
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(iters):
        sw = cov @ w
        rc = w * sw
        target = float(np.mean(np.abs(rc)))
        denom = np.where(np.abs(rc) < 1e-15, 1e-15, np.abs(rc))
        w_new = w * np.sqrt(target / denom)   # sqrt-damping → стабильная сходимость к ERC
        w_new = np.maximum(w_new, 1e-15)
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


def black_litterman_mu(
    cov: np.ndarray,
    prior: np.ndarray,
    *,
    P: Optional[np.ndarray] = None,
    Q: Optional[np.ndarray] = None,
    omega: Optional[np.ndarray] = None,
    tau: float = 0.05,
) -> np.ndarray:
    """BL-постериор μ. Без views возвращает prior."""
    if P is None or Q is None:
        return np.asarray(prior, dtype="float64")
    P = np.atleast_2d(P).astype("float64")
    Q = np.asarray(Q, dtype="float64").reshape(-1)
    tau_sigma = tau * cov
    if omega is None:
        omega = np.diag(np.diag(P @ tau_sigma @ P.T))
    inv_tau_sigma = np.linalg.pinv(tau_sigma)
    inv_omega = np.linalg.pinv(omega)
    A = inv_tau_sigma + P.T @ inv_omega @ P
    b = inv_tau_sigma @ prior + P.T @ inv_omega @ Q
    return np.linalg.solve(A, b)


# ---------------------------------------------------------------------------
# Transaction-cost-aware objective + sizing (P1)
# ---------------------------------------------------------------------------
try:  # scipy есть в окружении (cvxpy — нет) → tcost-aware solve через SLSQP
    from scipy.optimize import minimize as _scipy_minimize  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _scipy_minimize = None
    _HAS_SCIPY = False


@dataclass
class TCostModel:
    """Транзакционные косты для целевой функции.

    Два режима:
      * **uniform** (по умолчанию): κ·(linear·Σ|Δw| + quad·ΣΔw²);
      * **per-name √impact** (``sqrt_impact=True``, P2 #16): Almgren-Chriss участие по
        каждому имени — impact_i ∝ k·√(|Δnotional_i| / ADV_i). Cost как доля NAV:
        Σ_i |Δw_i| · k · √(participation_i), participation_i = |Δw_i|·NAV / ADV_i.
        Это реальная per-name ёмкость, а не один κ·Δw² на всех.

    ``adv`` — вектор среднего дневного оборота по именам (USD), выровненный к порядку
    весов; ``nav`` — ноционал портфеля (USD). Передаются в ``cost`` из оптимизатора.
    """
    linear: float = 0.0008
    quad: float = 0.0
    coef: float = 1.0          # общий множитель κ
    sqrt_impact: bool = False
    impact_coef: float = 0.1   # k в √-impact: bps-эквивалент = k·√participation

    def cost(self, delta: np.ndarray, *, adv: Optional[np.ndarray] = None,
             nav: Optional[float] = None) -> float:
        d = np.abs(np.asarray(delta, dtype="float64"))
        linear = self.linear * float(d.sum())
        if self.sqrt_impact and adv is not None and nav:
            adv_v = np.asarray(adv, dtype="float64")
            adv_v = np.where(adv_v > 0.0, adv_v, np.inf)   # unknown ADV → no impact term
            traded_notional = d * float(nav)
            participation = traded_notional / adv_v
            impact = float((self.impact_coef * np.sqrt(participation) * d).sum())
            return float(self.coef * (linear + impact))
        quad = self.quad * float((d * d).sum())
        return float(self.coef * (linear + quad))


@dataclass
class SizingConfig:
    """Сайзинг после оптимизации: vol-targeting или (фракционный) Kelly."""
    method: str = "none"        # none | vol_target | kelly
    target_vol: Optional[float] = None    # для vol_target (σ на период)
    kelly_fraction: float = 0.5           # для kelly (0.5 = half-Kelly)
    max_leverage: Optional[float] = None  # ограничить gross после сайзинга


@dataclass
class RobustConfig:
    """Robust optimization против estimation error в μ (P2 #15).

    * **box**: |μ_i − μ̂_i| ≤ δ_i ⇒ worst-case μᵀw = μ̂ᵀw − Σ δ_i|w_i|
      (δ_i = ``kappa`` · σ_μ,i). Эффект — L1-штраф на веса (шринк к диверсификации).
    * **ellipsoidal**: μ ∈ {μ̂ + Ω^½ u : ‖u‖₂ ≤ κ} ⇒ worst-case = μ̂ᵀw − κ·√(wᵀΩw).
      Ω — ковариация оценки μ (по умолчанию diag(σ_μ²)); если не задана, берётся из Σ.
    """
    enabled: bool = False
    kind: str = "box"                       # box | ellipsoidal
    kappa: float = 1.0                       # размер множества неопределённости
    mu_uncertainty: Optional[np.ndarray] = None   # σ_μ по именам (box) — выровнен к symbols
    omega: Optional[np.ndarray] = None      # Ω для ellipsoidal (N×N); None → diag(σ_μ²) или Σ


def kelly_weights(mu: np.ndarray, cov: np.ndarray, fraction: float = 1.0) -> np.ndarray:
    """Фракционный Kelly: w = fraction · Σ⁻¹μ (рост log-капитала)."""
    return float(fraction) * _solve_psd(np.asarray(cov, dtype="float64"),
                                        np.asarray(mu, dtype="float64"))


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
class PortfolioOptimizer:
    """μ + Σ + ограничения → целевые веса w*."""

    _CONVEX = {"min_variance", "mean_variance", "max_sharpe", "black_litterman"}

    def __init__(
        self,
        *,
        objective: str = "mean_variance",
        risk_aversion: float = 5.0,
        constraints: Optional[OptimizerConstraints] = None,
        use_cvxpy: str = "auto",   # 'auto' | 'never'
        bl_views: Optional[Dict[str, Any]] = None,
        tcost: Optional["TCostModel"] = None,   # tcost(w−w₀) В целевой функции (P1)
        sizing: Optional["SizingConfig"] = None,  # vol-target / Kelly после оптимизации
        robust: Optional["RobustConfig"] = None,  # robust μ-uncertainty (P2 #15)
    ) -> None:
        self.objective = objective
        self.risk_aversion = float(risk_aversion)
        self.constraints = constraints or OptimizerConstraints()
        self.use_cvxpy = use_cvxpy
        self.bl_views = bl_views or {}
        self.tcost = tcost
        self.sizing = sizing
        self.robust = robust

    # ---- PortfolioConstructor contract ----
    def solve(
        self,
        mu: pd.Series,
        cov: Any,
        current_w: Optional[pd.Series] = None,
        constraints: Optional[OptimizerConstraints] = None,
        tcost_model: Any = None,
        *,
        adv: Optional[pd.Series] = None,   # per-name ADV (USD) for √-impact tcost (#16)
        nav: Optional[float] = None,       # portfolio notional (USD) for √-impact tcost
    ) -> pd.Series:
        cons = constraints or self.constraints
        symbols = list(mu.index)
        mu_v = mu.astype("float64").to_numpy()
        Sigma = self._align_cov(cov, symbols)
        w0 = self._align_w0(current_w, symbols)
        adv_v = None
        if adv is not None:
            adv_v = pd.Series(adv, dtype="float64").reindex(symbols).fillna(0.0).to_numpy()

        raw = self._raw_weights(mu_v, Sigma, cons)

        w = None
        # 1) constrained solve (scipy SLSQP): tcost(w−w₀) in the objective AND exact
        #    enforcement of sector/factor caps (cvxpy is absent in this env → scipy).
        #    Runs when a tcost model is set, sector/factor caps are requested, OR robust
        #    optimization is enabled — so none of those are silently ignored.
        _robust_on = bool(self.robust is not None and self.robust.enabled)
        _needs_constrained = bool(cons.sector_caps or cons.factor_caps or _robust_on)
        if (self.tcost is not None or _needs_constrained) and _HAS_SCIPY and self.objective in self._CONVEX:
            try:
                w = self._solve_scipy(mu_v, Sigma, w0, cons, symbols, raw, adv=adv_v, nav=nav)
            except Exception as exc:
                logger.warning("scipy constrained solve failed (%s); falling back", exc)
                w = None
        # 2) cvxpy (если доступен в окружении)
        if w is None and self.use_cvxpy != "never" and _HAS_CVXPY and self.objective in self._CONVEX:
            try:  # pragma: no cover - cvxpy недоступен в текущем окружении
                w = self._solve_cvxpy(mu_v, Sigma, w0, cons)
            except Exception as exc:  # pragma: no cover
                logger.warning("cvxpy solve failed (%s); falling back to analytic", exc)
                w = None
        # 3) аналитика + проекция на жёсткие ограничения
        if w is None:
            w = self._project(raw, cons, w0, symbols)

        # сайзинг (vol-target / Kelly) поверх решения
        if self.sizing is not None and self.sizing.method != "none":
            w = self._apply_sizing(np.asarray(w, dtype="float64"), mu_v, Sigma, cons)

        return pd.Series(w, index=symbols, name="weight")

    # ---- tcost-aware solve (scipy) ----
    def _solve_scipy(self, mu, Sigma, w0, cons, symbols, raw, *, adv=None, nav=None):
        import math as _math
        n = len(mu)
        eps = 1e-6
        la = self.risk_aversion
        tc = self.tcost

        def smooth_l1(x):
            return np.sqrt(x * x + eps * eps)

        # Robust μ-uncertainty (P2 #15): worst-case mean penalty subtracted from return.
        rob = self.robust if (self.robust is not None and self.robust.enabled) else None
        if rob is not None and rob.kind == "box":
            if rob.mu_uncertainty is not None:
                _delta = np.asarray(rob.mu_uncertainty, dtype="float64") * float(rob.kappa)
            else:
                _delta = np.sqrt(np.clip(np.diag(Sigma), 0.0, None)) * float(rob.kappa)
        elif rob is not None and rob.kind == "ellipsoidal":
            if rob.omega is not None:
                _Omega = np.asarray(rob.omega, dtype="float64")
            elif rob.mu_uncertainty is not None:
                _Omega = np.diag(np.asarray(rob.mu_uncertainty, dtype="float64") ** 2)
            else:
                _Omega = np.diag(np.clip(np.diag(Sigma), 0.0, None))

        def _robust_penalty(w):
            if rob is None:
                return 0.0
            if rob.kind == "box":
                return float((_delta * smooth_l1(w)).sum())
            return float(rob.kappa * _math.sqrt(max(0.0, float(w @ _Omega @ w))))

        def objective(w):
            risk = la * float(w @ Sigma @ w)
            ret = float(mu @ w)
            cost = tc.cost(w - w0, adv=adv, nav=nav) if tc is not None else 0.0
            return risk - ret + cost + _robust_penalty(w)

        # box bounds
        if cons.min_position is not None:
            lo = float(cons.min_position)
        elif cons.long_only:
            lo = 0.0
        elif cons.max_position is not None:
            lo = -float(cons.max_position)
        else:
            lo = -10.0
        hi = float(cons.max_position) if cons.max_position is not None else 10.0
        bounds = [(lo, hi)] * n

        ccs: List[Dict[str, Any]] = []
        if cons.net_target is not None:
            nt = float(cons.net_target)
            ccs.append({"type": "eq", "fun": (lambda w, nt=nt: float(np.sum(w) - nt))})
        if cons.gross_max is not None:
            gm = float(cons.gross_max)
            ccs.append({"type": "ineq", "fun": (lambda w, gm=gm: gm - float(smooth_l1(w).sum()))})
        if cons.max_turnover is not None and w0 is not None:
            mt = float(cons.max_turnover)
            ccs.append({"type": "ineq", "fun": (lambda w, mt=mt: mt - float(smooth_l1(w - w0).sum()))})
        if cons.exposures is not None and cons.factor_caps:
            B = cons.exposures.reindex(index=list(symbols)).fillna(0.0)
            for f, cap in cons.factor_caps.items():
                if f in B.columns:
                    bf = B[f].to_numpy(dtype="float64")
                    cap = float(cap)
                    ccs.append({"type": "ineq", "fun": (lambda w, bf=bf, cap=cap: cap - float(bf @ w))})
                    ccs.append({"type": "ineq", "fun": (lambda w, bf=bf, cap=cap: cap + float(bf @ w))})
        # Sector caps: gross sector exposure Σ_{i∈sector}|w_i| ≤ cap (smooth-L1 for SLSQP).
        if cons.sector_map and cons.sector_caps:
            sector_idx = self._sector_index_groups(symbols, cons.sector_map)
            for sec, cap in cons.sector_caps.items():
                idx = sector_idx.get(sec)
                if idx:
                    ia = np.asarray(idx, dtype=int)
                    cap = float(cap)
                    ccs.append({"type": "ineq",
                                "fun": (lambda w, ia=ia, cap=cap: cap - float(smooth_l1(w[ia]).sum()))})

        x0 = self._project(np.asarray(raw, dtype="float64"), cons, w0, symbols)
        res = _scipy_minimize(objective, x0, method="SLSQP", bounds=bounds,
                              constraints=ccs, options={"maxiter": 300, "ftol": 1e-10})
        w = np.asarray(res.x, dtype="float64")
        # гарантия жёстких границ (SLSQP даёт приближённое выполнение)
        w = self._apply_box(w, cons)
        w = self._apply_net(w, cons)
        w = self._apply_gross(w, cons)
        # cleanup: ensure gross sector caps hold even after the box/gross rescale
        w = self._apply_sector_caps(w, cons, symbols)
        return w

    # ---- sizing (vol-target / Kelly) ----
    def _apply_sizing(self, w, mu, Sigma, cons):
        import math as _math
        sc = self.sizing
        w = np.asarray(w, dtype="float64")
        if sc.method == "vol_target" and sc.target_vol:
            cur = _math.sqrt(max(0.0, float(w @ Sigma @ w)))
            if cur > 1e-12:
                w = w * (float(sc.target_vol) / cur)
        elif sc.method == "kelly":
            kw = kelly_weights(mu, Sigma, sc.kelly_fraction)
            target_gross = float(np.abs(kw).sum())
            g = float(np.abs(w).sum())
            if g > 1e-12 and target_gross > 0:
                w = w * (target_gross / g)   # направление оптимизатора, Kelly-плечо
        if sc.max_leverage is not None:
            g = float(np.abs(w).sum())
            if g > float(sc.max_leverage) and g > 0:
                w = w * (float(sc.max_leverage) / g)
        return w

    # ---- raw analytic weights ----
    def _raw_weights(self, mu: np.ndarray, Sigma: np.ndarray, cons: OptimizerConstraints) -> np.ndarray:
        n = len(mu)
        obj = self.objective
        if obj == "equal_weight":
            return np.ones(n) / n
        if obj == "min_variance":
            x = _solve_psd(Sigma, np.ones(n))
            s = x.sum()
            return x / s if abs(s) > 1e-12 else np.ones(n) / n
        if obj == "risk_parity":
            return _risk_parity(Sigma)
        if obj in ("mean_variance", "max_sharpe"):
            return _solve_psd(Sigma, mu) / (2.0 * self.risk_aversion)
        if obj == "black_litterman":
            post = black_litterman_mu(
                Sigma,
                mu,
                P=self.bl_views.get("P"),
                Q=self.bl_views.get("Q"),
                omega=self.bl_views.get("omega"),
                tau=float(self.bl_views.get("tau", 0.05)),
            )
            return _solve_psd(Sigma, post) / (2.0 * self.risk_aversion)
        raise ValueError(f"unknown objective: {obj!r}")

    # ---- projection onto hard constraints (numpy fallback) ----
    def _project(
        self,
        w: np.ndarray,
        cons: OptimizerConstraints,
        w0: np.ndarray,
        symbols: Sequence[str],
    ) -> np.ndarray:
        w = np.asarray(w, dtype="float64").copy()
        n = len(w)
        if cons.factor_caps:
            logger.warning(
                "factor tilt limits are exactly enforced only via the scipy/cvxpy path; "
                "applying best-effort in the numpy fallback."
            )

        for _ in range(100):
            prev = w.copy()
            if cons.long_only:
                w = np.maximum(w, 0.0)
            w = self._apply_box(w, cons)
            w = self._apply_net(w, cons)
            w = self._apply_gross(w, cons)
            # best-effort gross sector-cap enforcement (exact in the scipy path)
            w = self._apply_sector_caps(w, cons, symbols)
            if np.max(np.abs(w - prev)) < 1e-13:
                break

        # turnover — после стабилизации формы
        if cons.max_turnover is not None and w0 is not None:
            trade = w - w0
            to = float(np.abs(trade).sum())
            if to > cons.max_turnover + 1e-12 and to > 0:
                w = w0 + (cons.max_turnover / to) * trade

        # финальные жёсткие границы (гарантия)
        if cons.long_only:
            w = np.maximum(w, 0.0)
        w = self._apply_box(w, cons)
        w = self._apply_gross(w, cons)
        return w

    @staticmethod
    def _apply_box(w: np.ndarray, cons: OptimizerConstraints) -> np.ndarray:
        if cons.max_position is None and cons.min_position is None:
            return w
        if cons.min_position is not None:
            lo = cons.min_position
        elif cons.long_only:
            lo = 0.0
        elif cons.max_position is not None:
            lo = -cons.max_position
        else:
            lo = -np.inf
        hi = cons.max_position if cons.max_position is not None else np.inf
        return np.clip(w, lo, hi)

    @staticmethod
    def _apply_net(w: np.ndarray, cons: OptimizerConstraints) -> np.ndarray:
        if cons.net_target is None:
            return w
        nt = float(cons.net_target)
        s = float(w.sum())
        if abs(nt) < 1e-12:
            return w - w.mean()         # market-neutral: центрируем
        if abs(s) > 1e-12:
            return w * (nt / s)         # масштабируем (сохраняет направление)
        return w

    @staticmethod
    def _apply_gross(w: np.ndarray, cons: OptimizerConstraints) -> np.ndarray:
        if cons.gross_max is None:
            return w
        g = float(np.abs(w).sum())
        if g > cons.gross_max + 1e-12 and g > 0:
            return w * (cons.gross_max / g)
        return w

    @staticmethod
    def _sector_index_groups(symbols: Sequence[str], sector_map: Dict[str, str]) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = {}
        for i, s in enumerate(symbols):
            sec = sector_map.get(str(s))
            if sec is not None:
                groups.setdefault(str(sec), []).append(i)
        return groups

    @staticmethod
    def _apply_sector_caps(
        w: np.ndarray, cons: OptimizerConstraints, symbols: Sequence[str]
    ) -> np.ndarray:
        """Best-effort: scale each sector's weights so gross sector exposure
        Σ_{i∈sector}|w_i| ≤ cap (used in the numpy fallback / SLSQP cleanup)."""
        if not cons.sector_map or not cons.sector_caps:
            return w
        w = np.asarray(w, dtype="float64").copy()
        groups = PortfolioOptimizer._sector_index_groups(symbols, cons.sector_map)
        for sec, cap in cons.sector_caps.items():
            idx = groups.get(str(sec))
            if not idx:
                continue
            cap = float(cap)
            ia = np.asarray(idx, dtype=int)
            gross = float(np.abs(w[ia]).sum())
            if gross > cap + 1e-12 and gross > 0:
                w[ia] = w[ia] * (cap / gross)
        return w

    # ---- cvxpy path (used only if available) ----
    def _solve_cvxpy(self, mu, Sigma, w0, cons):  # pragma: no cover - нет cvxpy
        n = len(mu)
        w = _cp.Variable(n)
        risk = _cp.quad_form(w, _cp.psd_wrap(Sigma))
        if self.objective == "min_variance":
            obj = _cp.Minimize(risk)
        else:
            obj = _cp.Minimize(self.risk_aversion * risk - mu @ w)
        ccs = []
        if cons.net_target is not None:
            ccs.append(_cp.sum(w) == cons.net_target)
        if cons.gross_max is not None:
            ccs.append(_cp.norm1(w) <= cons.gross_max)
        if cons.long_only:
            ccs.append(w >= 0)
        if cons.max_position is not None:
            ccs.append(w <= cons.max_position)
            if not cons.long_only:
                ccs.append(w >= -cons.max_position)
        if cons.max_turnover is not None and w0 is not None:
            ccs.append(_cp.norm1(w - w0) <= cons.max_turnover)
        if cons.exposures is not None and cons.factor_caps:
            B = cons.exposures.reindex(index=list(range(n))).to_numpy()
            for f, cap in cons.factor_caps.items():
                if f in cons.exposures.columns:
                    bf = cons.exposures[f].to_numpy()
                    ccs.append(_cp.abs(bf @ w) <= cap)
        _cp.Problem(obj, ccs).solve()
        if w.value is None:
            raise RuntimeError("cvxpy returned no solution")
        return np.asarray(w.value, dtype="float64")

    # ---- helpers ----
    @staticmethod
    def _align_cov(cov: Any, symbols: Sequence[str]) -> np.ndarray:
        if isinstance(cov, pd.DataFrame):
            return cov.reindex(index=symbols, columns=symbols).to_numpy(dtype="float64")
        return np.asarray(cov, dtype="float64")

    @staticmethod
    def _align_w0(current_w: Optional[pd.Series], symbols: Sequence[str]) -> np.ndarray:
        if current_w is None:
            return np.zeros(len(symbols))
        return current_w.reindex(symbols).fillna(0.0).to_numpy(dtype="float64")


class MultiPeriodOptimizer:
    """Multi-period (horizon) optimization — Gârleanu–Pedersen aim-portfolio (P2 #15).

    Single-period MVO over-trades because it ignores future rebalances. The
    multi-period optimum trades only **partially toward the aim** each period; the
    trade rate falls as transaction costs rise. Given a path of expected returns
    ``mu_path`` [T×N], we compute each period's single-period target (the "aim") and
    move a fraction φ of the way there, projecting onto constraints:

        w_t = (1−φ)·w_{t−1} + φ·aim_t,   then project(w_t)

    φ is either fixed (``trade_rate``) or the GP closed-form rate from risk aversion λ
    and a scalar trade-cost Λ:  φ* = (√(λ²+4λΛ) − λ) / (2Λ)  (φ→1 as Λ→0, slower with cost).
    """

    def __init__(
        self,
        single: "PortfolioOptimizer",
        *,
        trade_rate: Optional[float] = None,
        trade_cost: float = 0.001,
    ) -> None:
        self.single = single
        self.trade_cost = float(trade_cost)
        self.trade_rate = trade_rate

    def _phi(self) -> float:
        if self.trade_rate is not None:
            return float(min(1.0, max(1e-3, self.trade_rate)))
        la = max(1e-9, float(self.single.risk_aversion))
        lam = max(1e-12, float(self.trade_cost))
        phi = (np.sqrt(la * la + 4.0 * la * lam) - la) / (2.0 * lam)
        return float(min(1.0, max(1e-3, phi)))

    def solve_path(
        self,
        mu_path: pd.DataFrame,           # index=period, columns=symbol
        cov: Any,
        current_w: Optional[pd.Series] = None,
        **solve_kwargs: Any,
    ) -> pd.DataFrame:
        symbols = list(mu_path.columns)
        phi = self._phi()
        w_prev = (current_w.reindex(symbols).fillna(0.0) if current_w is not None
                  else pd.Series(0.0, index=symbols))
        rows: Dict[Any, pd.Series] = {}
        for t in mu_path.index:
            mu_t = mu_path.loc[t].astype("float64")
            aim = self.single.solve(mu_t, cov, current_w=w_prev, **solve_kwargs)
            blended = (1.0 - phi) * w_prev.reindex(symbols).fillna(0.0) + phi * aim.reindex(symbols).fillna(0.0)
            # project the blended target back onto hard constraints
            wv = self.single._project(
                blended.to_numpy(dtype="float64"), self.single.constraints,
                w_prev.reindex(symbols).fillna(0.0).to_numpy(dtype="float64"), symbols)
            w_t = pd.Series(wv, index=symbols)
            rows[t] = w_t
            w_prev = w_t
        return pd.DataFrame(rows).T.reindex(columns=symbols)

    def solve(
        self,
        mu: pd.Series,
        cov: Any,
        current_w: Optional[pd.Series] = None,
        constraints: Optional[OptimizerConstraints] = None,
        tcost_model: Any = None,
        **solve_kwargs: Any,
    ) -> pd.Series:
        """One Gârleanu–Pedersen step — drop-in for the single-period optimizer
        contract: aim = single-period target, then move φ of the way and project."""
        symbols = list(mu.index)
        phi = self._phi()
        w0 = (current_w.reindex(symbols).fillna(0.0) if current_w is not None
              else pd.Series(0.0, index=symbols))
        aim = self.single.solve(mu, cov, current_w=w0, constraints=constraints,
                                tcost_model=tcost_model, **solve_kwargs)
        blended = (1.0 - phi) * w0.reindex(symbols).fillna(0.0) + phi * aim.reindex(symbols).fillna(0.0)
        wv = self.single._project(
            blended.to_numpy(dtype="float64"),
            constraints or self.single.constraints,
            w0.reindex(symbols).fillna(0.0).to_numpy(dtype="float64"), symbols)
        return pd.Series(wv, index=symbols, name="weight")

    # delegate attributes the backtest may read off the optimizer
    @property
    def constraints(self):
        return self.single.constraints

    @property
    def objective(self):
        return self.single.objective

    @property
    def trade_rate_used(self) -> float:
        return self._phi()


__all__ = [
    "OptimizerConstraints",
    "PortfolioOptimizer",
    "MultiPeriodOptimizer",
    "RobustConfig",
    "black_litterman_mu",
    "TCostModel",
    "SizingConfig",
    "kelly_weights",
]
