# -*- coding: utf-8 -*-
"""
service_pretrade_risk.py
========================

Live portfolio риск-контур (P1): VaR / CVaR / стресс-тесты / сценарный grid ПЕРЕД
отправкой rebalance + real-time мониторинг факторных экспозиций в течение дня.

Дополняет ``service_xs_portfolio_risk.PortfolioRiskGuard`` (жёсткие лимиты gross/net/
sector/factor/turnover): тот проверяет ЛИМИТЫ, этот — РИСК-МЕТРИКИ и СЦЕНАРИИ.

Состав:
  * ``PreTradeRiskAnalyzer`` — параметрический (Gaussian) и исторический VaR/CVaR,
    factor-экспозиции (Bᵀw), сценарный grid (шок рынка, рост волатильности, сдвиг
    корреляций), pre-trade gate с лимитами.
  * ``FactorExposureMonitor`` — внутридневной стейтфул-монитор экспозиций vs лимиты.

Зависимости: numpy/pandas, scipy.stats (есть в окружении). Слой ``service_``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from scipy.stats import norm as _norm

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _norm = None
    _HAS_SCIPY = False


def _z_alpha(alpha: float) -> float:
    """Квантиль стандартного нормального для хвоста ``alpha`` (напр. 0.05 → 1.645)."""
    a = min(max(float(alpha), 1e-6), 0.5)
    if _HAS_SCIPY:
        return float(-_norm.ppf(a))
    # рациональная аппроксимация Acklam (fallback без scipy)
    p = a
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    t = math.sqrt(-2.0 * math.log(p))
    return t - ((c[2] * t + c[1]) * t + c[0]) / (((d[2] * t + d[1]) * t + d[0]) * t + 1.0)


def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@dataclass
class RiskLimits:
    """Лимиты на риск-метрики (любой None → не проверяется)."""

    var_max: Optional[float] = None  # VaR (доля капитала) ≤ var_max
    cvar_max: Optional[float] = None  # CVaR (доля капитала) ≤ cvar_max
    vol_max: Optional[float] = None  # портфельная σ (период) ≤ vol_max
    factor_caps: Optional[Dict[str, float]] = None  # |(Bᵀw)_f| ≤ cap
    scenario_loss_max: Optional[float] = None  # худший сценарный P&L ≥ -scenario_loss_max


@dataclass
class ScenarioResult:
    name: str
    pnl: float  # P&L портфеля в сценарии (доля капитала)
    var: Optional[float] = None
    detail: Dict[str, Any] = field(default_factory=dict)


# Named historical stress scenarios (P1 #9) — calibrated approximate magnitudes of
# the broad equity drawdown, the volatility blow-up and the correlation tightening
# observed in each episode. Used to stress the SAME portfolio via betas + Σ.
#   market_shock : cumulative broad-market return over the episode
#   vol_mult     : realized-vol multiplier vs the calm regime
#   corr_shift   : pull of pairwise correlations toward +1 (diversification collapse)
NAMED_STRESS_SCENARIOS: Dict[str, Dict[str, float]] = {
    "2008_gfc": {"market_shock": -0.40, "vol_mult": 3.0, "corr_shift": 0.40},
    "2020_covid": {"market_shock": -0.34, "vol_mult": 4.0, "corr_shift": 0.50},
    "2010_flash_crash": {"market_shock": -0.09, "vol_mult": 5.0, "corr_shift": 0.60},
    "2015_chf_unpeg": {"market_shock": -0.03, "vol_mult": 3.0, "corr_shift": 0.30},
    "2018_q4_selloff": {"market_shock": -0.20, "vol_mult": 2.0, "corr_shift": 0.30},
    "2022_rates_shock": {"market_shock": -0.25, "vol_mult": 1.8, "corr_shift": 0.25},
}


@dataclass
class PreTradeRiskReport:
    approved: bool
    var: float
    cvar: float
    vol: float
    factor_exposure: Dict[str, float]
    scenarios: List[ScenarioResult]
    worst_scenario: Optional[ScenarioResult]
    violations: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)
    # P1 #9 additions
    mc_var: Optional[float] = None
    mc_cvar: Optional[float] = None
    mc_dist: Optional[str] = None
    component_var: Dict[str, float] = field(default_factory=dict)
    marginal_var: Dict[str, float] = field(default_factory=dict)
    named_scenarios: List[ScenarioResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "var": self.var,
            "cvar": self.cvar,
            "vol": self.vol,
            "factor_exposure": self.factor_exposure,
            "scenarios": [s.__dict__ for s in self.scenarios],
            "worst_scenario": (self.worst_scenario.__dict__ if self.worst_scenario else None),
            "violations": list(self.violations),
            "metrics": self.metrics,
            "mc_var": self.mc_var,
            "mc_cvar": self.mc_cvar,
            "mc_dist": self.mc_dist,
            "component_var": self.component_var,
            "marginal_var": self.marginal_var,
            "named_scenarios": [s.__dict__ for s in self.named_scenarios],
        }


class PreTradeRiskAnalyzer:
    """VaR/CVaR/стресс/сценарии для целевого вектора весов ПЕРЕД rebalance.

    ``cov`` — Σ (DataFrame index=symbol или ndarray в порядке ``symbols``).
    ``exposures`` — B (index=symbol, cols=factor), опц. для factor-стресса/мониторинга.
    Метрики — на горизонт одного периода Σ (доля капитала). Веса — доли NAV.
    """

    def __init__(
        self,
        cov: Any,
        *,
        exposures: Optional[pd.DataFrame] = None,
        market_factor: str = "market",
    ) -> None:
        self._cov = cov
        self._exposures = exposures
        self.market_factor = market_factor

    # ---- align ----
    def _aligned(self, w: pd.Series):
        w = w.astype("float64").dropna()
        symbols = list(w.index)
        if isinstance(self._cov, pd.DataFrame):
            Sigma = (
                self._cov.reindex(index=symbols, columns=symbols)
                .fillna(0.0)
                .to_numpy(dtype="float64")
            )
        else:
            Sigma = np.asarray(self._cov, dtype="float64")
        wv = w.to_numpy(dtype="float64")
        return wv, Sigma, symbols

    # ---- core metrics ----
    def portfolio_vol(self, w: pd.Series) -> float:
        wv, S, _ = self._aligned(w)
        return float(math.sqrt(max(0.0, float(wv @ S @ wv))))

    def parametric_var(self, w: pd.Series, alpha: float = 0.05, *, mean: float = 0.0) -> float:
        """Gaussian VaR (доля капитала, положительное число = величина потери)."""
        sigma = self.portfolio_vol(w)
        return float(max(0.0, _z_alpha(alpha) * sigma - mean))

    def parametric_cvar(self, w: pd.Series, alpha: float = 0.05, *, mean: float = 0.0) -> float:
        """Gaussian Expected Shortfall: ES = φ(z_α)/α · σ − mean."""
        sigma = self.portfolio_vol(w)
        z = _z_alpha(alpha)
        return float(max(0.0, (_phi(z) / max(alpha, 1e-6)) * sigma - mean))

    def monte_carlo_var_cvar(
        self,
        w: pd.Series,
        alpha: float = 0.05,
        *,
        n_sims: int = 20_000,
        dist: str = "normal",
        dof: int = 5,
        seed: int = 12345,
        mean: float = 0.0,
    ):
        """Monte-Carlo VaR/CVaR by simulating portfolio P&L from N(0, Σ) or a
        Student-t copula (fat tails). Returns (var, cvar). Glasserman (2004)."""
        wv, S, _ = self._aligned(w)
        n = wv.shape[0]
        if n == 0:
            return 0.0, 0.0
        rng = np.random.default_rng(seed)
        # PSD-safe factorization (eigh handles near-singular Σ)
        vals, vecs = np.linalg.eigh((S + S.T) / 2.0)
        vals = np.clip(vals, 0.0, None)
        L = vecs @ np.diag(np.sqrt(vals))
        z = rng.standard_normal((int(n_sims), n))
        if dist == "t":
            # Student-t scale mixture: z / sqrt(chi2_dof/dof), variance-matched
            g = rng.chisquare(dof, size=(int(n_sims), 1)) / dof
            z = z / np.sqrt(g)
            z = z * math.sqrt((dof - 2.0) / dof) if dof > 2 else z  # unit-variance scale
        sims = z @ L.T
        pnl = sims @ wv + mean
        q = float(np.quantile(pnl, alpha))
        var = float(max(0.0, -q))
        tail = pnl[pnl <= q]
        cvar = float(max(0.0, -tail.mean())) if tail.size else var
        return var, cvar

    def component_var(self, w: pd.Series, alpha: float = 0.05):
        """Euler component & marginal VaR (parametric Gaussian). Σ component = VaR.

        marginal_i = z·(Σw)_i/σ ; component_i = w_i·marginal_i (Tasche 2008)."""
        wv, S, symbols = self._aligned(w)
        sigma = float(math.sqrt(max(0.0, float(wv @ S @ wv))))
        if sigma <= 1e-15:
            zero = {s: 0.0 for s in symbols}
            return zero, dict(zero)
        z = _z_alpha(alpha)
        Sw = S @ wv
        marginal = {symbols[i]: float(z * Sw[i] / sigma) for i in range(len(symbols))}
        component = {symbols[i]: float(wv[i] * marginal[symbols[i]]) for i in range(len(symbols))}
        return component, marginal

    def incremental_var(self, w: pd.Series, symbol: str, alpha: float = 0.05) -> float:
        """Incremental VaR of a name: VaR(w) − VaR(w without the name)."""
        full = self.parametric_var(w, alpha)
        w_wo = w.drop(index=[symbol], errors="ignore")
        if not len(w_wo):
            return full
        return float(full - self.parametric_var(w_wo, alpha))

    def named_scenarios(
        self,
        w: pd.Series,
        *,
        alpha: float = 0.05,
        betas: Optional[pd.Series] = None,
        scenarios: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[ScenarioResult]:
        """Apply the named historical-stress library (2008/2020/...) to THIS portfolio.

        Each scenario P&L = (β·w)·market_shock (directional loss) and a stressed VaR
        under the episode's vol-multiplier + correlation-tightening regime."""
        lib = scenarios or NAMED_STRESS_SCENARIOS
        wv, S, symbols = self._aligned(w)
        beta = self._resolve_betas(symbols, betas)
        out: List[ScenarioResult] = []
        for name, p in lib.items():
            mkt = float(p.get("market_shock", -0.1))
            vol_mult = float(p.get("vol_mult", 1.0))
            corr_shift = float(p.get("corr_shift", 0.0))
            mkt_pnl = float((beta * wv).sum() * mkt)
            S_stab = self._shift_correlations((vol_mult**2) * S, corr_shift)
            stressed_var = float(_z_alpha(alpha) * math.sqrt(max(0.0, wv @ S_stab @ wv)))
            out.append(
                ScenarioResult(
                    name,
                    mkt_pnl - stressed_var,
                    var=stressed_var,
                    detail={
                        "market_shock": mkt,
                        "vol_mult": vol_mult,
                        "corr_shift": corr_shift,
                        "directional_pnl": mkt_pnl,
                        "stressed_var": stressed_var,
                    },
                )
            )
        return out

    def historical_var_cvar(self, w: pd.Series, returns: pd.DataFrame, alpha: float = 0.05):
        """Исторический VaR/CVaR из матрицы доходностей активов ``returns`` (index=time, cols=symbol)."""
        cols = [c for c in w.index if c in returns.columns]
        if not cols:
            return 0.0, 0.0
        R = returns[cols].to_numpy(dtype="float64")
        wv = w.reindex(cols).fillna(0.0).to_numpy(dtype="float64")
        pnl = R @ wv
        pnl = pnl[np.isfinite(pnl)]
        if pnl.size == 0:
            return 0.0, 0.0
        q = float(np.quantile(pnl, alpha))
        var = float(max(0.0, -q))
        tail = pnl[pnl <= q]
        cvar = float(max(0.0, -tail.mean())) if tail.size else var
        return var, cvar

    def factor_exposures(self, w: pd.Series) -> Dict[str, float]:
        """Bᵀw — факторные экспозиции портфеля."""
        if self._exposures is None:
            return {}
        B = self._exposures.reindex(index=list(w.index)).fillna(0.0)
        fexp = B.mul(w.reindex(B.index).fillna(0.0), axis=0).sum()
        return {str(k): float(v) for k, v in fexp.items()}

    # ---- scenario grid ----
    def scenario_grid(
        self,
        w: pd.Series,
        *,
        alpha: float = 0.05,
        market_shock: float = -0.10,
        vol_mult: float = 1.5,
        corr_shift: float = 0.2,
        betas: Optional[pd.Series] = None,
    ) -> List[ScenarioResult]:
        """Стресс-grid: рыночный шок, рост волатильности, сдвиг корреляций (+ их комбинация)."""
        wv, S, symbols = self._aligned(w)
        out: List[ScenarioResult] = []

        # 1) рыночный шок −10%: P&L = Σ β_i w_i · shock (β из exposures[market] или =1)
        beta = self._resolve_betas(symbols, betas)
        mkt_pnl = float((beta * wv).sum() * market_shock)
        out.append(
            ScenarioResult(
                f"market_shock_{int(market_shock*100)}pct",
                mkt_pnl,
                detail={"beta_dot_w": float((beta * wv).sum())},
            )
        )

        # 2) рост волатильности ×vol_mult → VaR пересчитывается на масштабированной Σ
        S_vol = (vol_mult**2) * S
        vol_var = float(_z_alpha(alpha) * math.sqrt(max(0.0, wv @ S_vol @ wv)))
        out.append(
            ScenarioResult(
                f"vol_x{vol_mult}",
                -vol_var,
                var=vol_var,
                detail={"stressed_vol": float(math.sqrt(max(0.0, wv @ S_vol @ wv)))},
            )
        )

        # 3) сдвиг корреляций: ρ → ρ + corr_shift·(1−ρ) (к 1), диагональ (дисперсии) сохраняем
        S_corr = self._shift_correlations(S, corr_shift)
        corr_var = float(_z_alpha(alpha) * math.sqrt(max(0.0, wv @ S_corr @ wv)))
        out.append(
            ScenarioResult(
                f"corr_shift_+{corr_shift}",
                -corr_var,
                var=corr_var,
                detail={"stressed_vol": float(math.sqrt(max(0.0, wv @ S_corr @ wv)))},
            )
        )

        # 4) комбинированный «кризис»: шок + рост vol + сдвиг корр
        S_combo = self._shift_correlations((vol_mult**2) * S, corr_shift)
        combo_var = float(_z_alpha(alpha) * math.sqrt(max(0.0, wv @ S_combo @ wv)))
        out.append(
            ScenarioResult(
                "crisis_combo",
                mkt_pnl - combo_var,
                var=combo_var,
                detail={"market_pnl": mkt_pnl, "stressed_var": combo_var},
            )
        )
        return out

    def _resolve_betas(self, symbols: Sequence[str], betas: Optional[pd.Series]) -> np.ndarray:
        if betas is not None:
            return betas.reindex(symbols).fillna(1.0).to_numpy(dtype="float64")
        if self._exposures is not None and self.market_factor in self._exposures.columns:
            return (
                self._exposures[self.market_factor]
                .reindex(symbols)
                .fillna(1.0)
                .to_numpy(dtype="float64")
            )
        return np.ones(len(symbols))

    @staticmethod
    def _shift_correlations(S: np.ndarray, shift: float) -> np.ndarray:
        d = np.sqrt(np.clip(np.diag(S), 1e-18, None))
        denom = np.outer(d, d)
        corr = S / denom
        corr = np.clip(corr, -0.999, 0.999)
        n = corr.shape[0]
        off = ~np.eye(n, dtype=bool)
        corr[off] = corr[off] + float(shift) * (1.0 - corr[off])  # тянем к +1
        np.fill_diagonal(corr, 1.0)
        return corr * denom

    # ---- pre-trade gate ----
    def pretrade_check(
        self,
        w_target: pd.Series,
        *,
        limits: Optional[RiskLimits] = None,
        alpha: float = 0.05,
        returns: Optional[pd.DataFrame] = None,
        scenario_kwargs: Optional[Dict[str, Any]] = None,
        strict: bool = True,
        monte_carlo: bool = True,
        mc_dist: str = "t",
        mc_sims: int = 20_000,
    ) -> PreTradeRiskReport:
        """Полный pre-trade риск-отчёт + вердикт (approved) ПЕРЕД отправкой rebalance.

        Включает (P1 #9): Monte-Carlo VaR/CVaR (Gaussian/Student-t fat-tails), Euler
        component/marginal VaR (атрибуция риска на имена) и именованные историч.
        стресс-сценарии (2008/2020/...)."""
        L = limits or RiskLimits()
        vol = self.portfolio_vol(w_target)
        if returns is not None:
            var, cvar = self.historical_var_cvar(w_target, returns, alpha)
        else:
            var = self.parametric_var(w_target, alpha)
            cvar = self.parametric_cvar(w_target, alpha)
        fexp = self.factor_exposures(w_target)
        scens = self.scenario_grid(w_target, alpha=alpha, **(scenario_kwargs or {}))
        named = self.named_scenarios(
            w_target, alpha=alpha, betas=(scenario_kwargs or {}).get("betas")
        )
        all_scens = scens + named
        worst = min(all_scens, key=lambda s: s.pnl) if all_scens else None
        # Monte-Carlo VaR/CVaR + Euler risk attribution
        mc_var = mc_cvar = None
        if monte_carlo:
            try:
                mc_var, mc_cvar = self.monte_carlo_var_cvar(
                    w_target, alpha, n_sims=mc_sims, dist=mc_dist
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("MC VaR failed: %s", exc)
        try:
            comp, marg = self.component_var(w_target, alpha)
        except Exception:  # pragma: no cover
            comp, marg = {}, {}

        viol: List[str] = []
        if L.var_max is not None and var > L.var_max + 1e-12:
            viol.append(f"VaR {var:.4f} > {L.var_max}")
        if L.cvar_max is not None and cvar > L.cvar_max + 1e-12:
            viol.append(f"CVaR {cvar:.4f} > {L.cvar_max}")
        if L.vol_max is not None and vol > L.vol_max + 1e-12:
            viol.append(f"vol {vol:.4f} > {L.vol_max}")
        if L.factor_caps:
            for f, cap in L.factor_caps.items():
                if f in fexp and abs(fexp[f]) > cap + 1e-12:
                    viol.append(f"factor {f} {fexp[f]:.4f} > {cap}")
        if (
            L.scenario_loss_max is not None
            and worst is not None
            and worst.pnl < -L.scenario_loss_max - 1e-12
        ):
            viol.append(f"scenario '{worst.name}' loss {worst.pnl:.4f} < -{L.scenario_loss_max}")

        approved = (len(viol) == 0) if strict else True
        return PreTradeRiskReport(
            approved=approved,
            var=var,
            cvar=cvar,
            vol=vol,
            factor_exposure=fexp,
            scenarios=scens,
            worst_scenario=worst,
            violations=viol,
            metrics={"alpha": alpha, "n_symbols": int(len(w_target.dropna()))},
            mc_var=mc_var,
            mc_cvar=mc_cvar,
            mc_dist=(mc_dist if monte_carlo else None),
            component_var=comp,
            marginal_var=marg,
            named_scenarios=named,
        )


class FactorExposureMonitor:
    """Внутридневной мониторинг факторных экспозиций vs лимиты (стейтфул).

    На каждый апдейт (новые веса/позиции) считает Bᵀw и сравнивает с ``factor_caps``;
    хранит историю и сигналит о пробоях (для real-time контроля в течение дня).
    """

    def __init__(
        self, exposures: pd.DataFrame, factor_caps: Dict[str, float], *, history: int = 512
    ) -> None:
        self.exposures = exposures
        self.factor_caps = dict(factor_caps)
        self._hist: List[Dict[str, Any]] = []
        self._max_hist = int(history)

    def update(self, weights: pd.Series, *, ts_ms: Optional[int] = None) -> Dict[str, Any]:
        B = self.exposures.reindex(index=list(weights.index)).fillna(0.0)
        fexp = B.mul(weights.reindex(B.index).fillna(0.0), axis=0).sum()
        exposure = {str(k): float(v) for k, v in fexp.items()}
        breaches = [
            f"{f}={exposure.get(f, 0.0):.4f} > {cap}"
            for f, cap in self.factor_caps.items()
            if abs(exposure.get(f, 0.0)) > cap + 1e-12
        ]
        rec = {
            "ts_ms": ts_ms,
            "exposure": exposure,
            "breaches": breaches,
            "within_limits": (len(breaches) == 0),
        }
        self._hist.append(rec)
        if len(self._hist) > self._max_hist:
            self._hist = self._hist[-self._max_hist :]
        if breaches:
            logger.warning("FactorExposureMonitor breaches: %s", breaches)
        return rec

    def history(self) -> List[Dict[str, Any]]:
        return list(self._hist)

    def latest(self) -> Optional[Dict[str, Any]]:
        return self._hist[-1] if self._hist else None


__all__ = [
    "RiskLimits",
    "ScenarioResult",
    "PreTradeRiskReport",
    "PreTradeRiskAnalyzer",
    "FactorExposureMonitor",
]
