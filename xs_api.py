# -*- coding: utf-8 -*-
"""
xs_api.py
=========

Изолированный FastAPI-роутер cross-sectional контура (Stage A12): ``/api/xs/*``.
Подключается в основной ``app.py`` одной строкой ``register_xs_routes(api)`` (аддитивно,
не ломает существующий MVP). Тестируется автономно через свежий ``FastAPI`` + роутер.

Эндпоинты — тонкие обёртки над чистыми функциями движка (pipeline / optimizer /
validation / attribution / live). Слой интеграции.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, HTTPException
from starlette.responses import JSONResponse

from service_xs_pipeline import XSConfig, run_backtest, load_panel, latest_target_weights
from service_optimizer import OptimizerConstraints, PortfolioOptimizer
from service_risk_model import StatRiskModel
from service_backtest_validation import trust_report
from service_attribution import factor_attribution
from service_xs_portfolio_risk import PortfolioRiskGuard, PortfolioRiskLimits
from service_xs_live import CrossSectionalLiveRunner
from impl_universe import StaticUniverse, IndexMembershipUniverse


def _json_finite(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN/+-Inf) with None.

    Backend math over empty/degenerate inputs (e.g. zero-variance returns) can
    yield NaN/Inf. The stdlib JSON encoder emits bare ``NaN``/``Infinity``
    tokens, which are invalid JSON and rejected by strict clients (browsers,
    Go, etc.). Mapping them to ``null`` keeps responses parseable everywhere.
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.floating):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _json_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_finite(v) for v in obj]
    return obj


class SafeJSONResponse(JSONResponse):
    """JSONResponse that scrubs NaN/Inf to null before serialization."""

    def render(self, content: Any) -> bytes:
        return super().render(_json_finite(content))


def make_xs_router() -> APIRouter:
    router = APIRouter(
        prefix="/api/xs",
        tags=["cross-sectional"],
        default_response_class=SafeJSONResponse,
    )

    @router.post("/config")
    def validate_config(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            cfg = XSConfig.model_validate(payload)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"invalid config: {exc}")
        return {"valid": True, "normalized": cfg.model_dump()}

    @router.post("/universe")
    def universe(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        spec = payload.get("universe", {})
        asof_ms = int(payload.get("asof_ms", 0))
        u = StaticUniverse(spec.get("symbols", []))
        return {
            "asof_ms": asof_ms,
            "constituents": list(u.constituents(asof_ms)),
            "survivorship_biased": u.survivorship_biased,
        }

    @router.post("/optimize")
    def optimize(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        mu_d = payload.get("mu", {})
        symbols = list(mu_d.keys())
        if len(symbols) < 1:
            raise HTTPException(status_code=422, detail="mu required")
        mu = pd.Series({s: float(v) for s, v in mu_d.items()})
        cov_in = payload.get("cov")
        if isinstance(cov_in, dict):
            cov = pd.DataFrame(cov_in).reindex(index=symbols, columns=symbols).fillna(0.0)
        else:
            cov = pd.DataFrame(np.eye(len(symbols)), index=symbols, columns=symbols)
        c = payload.get("constraints", {})
        # P1 #6: sector/factor caps + robust + BL + multi-period also reachable here.
        exposures_df = None
        if c.get("exposures"):
            exposures_df = pd.DataFrame(c["exposures"]).T.astype("float64")
        cons = OptimizerConstraints(
            gross_max=c.get("gross_max"), net_target=c.get("net_target"),
            long_only=bool(c.get("long_only", False)), max_position=c.get("max_position"),
            max_turnover=c.get("max_turnover"),
            sector_map=c.get("sector_map"), sector_caps=c.get("sector_caps"),
            exposures=exposures_df, factor_caps=c.get("factor_caps"),
        )
        from service_optimizer import RobustConfig as _RC, MultiPeriodOptimizer as _MPO
        robust = None
        _r = payload.get("robust")
        if _r and _r.get("enabled"):
            mu_unc = _r.get("mu_uncertainty")
            robust = _RC(enabled=True, kind=str(_r.get("kind", "box")),
                         kappa=float(_r.get("kappa", 1.0)),
                         mu_uncertainty=(np.asarray(mu_unc, dtype="float64") if mu_unc is not None else None))
        bl_views = None
        _bl = payload.get("bl_views")
        if _bl and _bl.get("P") is not None and _bl.get("Q") is not None:
            bl_views = {"P": np.asarray(_bl["P"], dtype="float64"), "Q": np.asarray(_bl["Q"], dtype="float64"),
                        "omega": (np.asarray(_bl["omega"], dtype="float64") if _bl.get("omega") is not None else None),
                        "tau": float(_bl.get("tau", 0.05))}
        opt = PortfolioOptimizer(
            objective=payload.get("objective", "mean_variance"),
            risk_aversion=float(payload.get("risk_aversion", 5.0)),
            use_cvxpy="never", constraints=cons, robust=robust, bl_views=bl_views,
        )
        _mp = payload.get("multi_period")
        cur_w = None
        if payload.get("current_w"):
            cur_w = pd.Series({s: float(v) for s, v in payload["current_w"].items()})
        if _mp and _mp.get("enabled"):
            tr = _mp.get("trade_rate")
            opt = _MPO(opt, trade_rate=(float(tr) if tr is not None else None),
                       trade_cost=float(_mp.get("trade_cost", 0.001)))
        w = opt.solve(mu, cov, current_w=cur_w)
        return {"weights": {s: float(v) for s, v in w.items()},
                "gross": float(w.abs().sum()), "net": float(w.sum())}

    @router.post("/risk_model")
    def risk_model(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        returns = payload.get("returns", {})  # {symbol: [r...]}
        wide = pd.DataFrame(returns)
        method = payload.get("method", "ledoit_wolf")
        rm = StatRiskModel(method=method, n_factors=payload.get("n_factors")).fit(wide)
        cov = rm.cov()
        return {"symbols": list(cov.index), "cov": cov.to_dict()}

    @router.post("/trust_report")
    def trust(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        returns = [float(x) for x in payload.get("returns", [])]
        return trust_report(
            returns, n_trials=int(payload.get("n_trials", 1)),
            periods_per_year=float(payload.get("periods_per_year", 252.0)),
        )

    @router.post("/attribution")
    def attribution(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        weights = pd.DataFrame(payload["weights"]).T
        asset_returns = pd.DataFrame(payload["asset_returns"]).T
        exposures = pd.DataFrame(payload["exposures"]).T
        rep = factor_attribution(weights, asset_returns, exposures)
        rep.pop("per_period", None)  # не сериализуем DataFrame
        return rep

    @router.post("/backtest")
    def backtest(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            cfg = XSConfig.model_validate(payload)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"invalid config: {exc}")
        out = run_backtest(cfg)
        res = out["result"]
        # Аддитивно: серии для tear-sheet (эквити/доходности/экспозиция/оборот).
        def _floats(s: Any) -> list:
            return [float(x) for x in np.asarray(s, dtype="float64")]
        series = {
            "ts": [int(t) for t in res.returns.index],
            "returns": _floats(res.returns.to_numpy()),
            "nav": _floats(res.nav.to_numpy()),
            "gross": _floats(res.gross.to_numpy()),
            "net": _floats(res.net.to_numpy()),
            "turnover": _floats(res.turnover.to_numpy()),
        }
        return {
            "summary": out["summary"],
            "trust_report": out["trust_report"],
            "n_rebalances": out["n_rebalances"],
            "series": series,
            "attribution": out.get("attribution"),
            "attribution_ts": out.get("attribution_ts"),
            "factor_attribution": out.get("factor_attribution"),  # P1 #11 (tied to risk model)
            "capacity": out.get("capacity"),                      # P1 #12 (AUM→Sharpe)
            # Honesty flags (P0 #2): mark synthetic results so the UI never shows a
            # fabricated edge as real.
            "data_source": out.get("data_source"),
            "is_synthetic": out.get("is_synthetic"),
            "real_data": out.get("real_data"),
            "warning": out.get("warning"),
        }

    @router.post("/tearsheet")
    def tearsheet(payload: Dict[str, Any] = Body(...)):
        """Rendered LP-grade HTML tear-sheet (P1 #10) — printable to PDF."""
        from starlette.responses import HTMLResponse
        from service_tearsheet import render_html_tearsheet
        try:
            cfg = XSConfig.model_validate(payload)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"invalid config: {exc}")
        out = run_backtest(cfg)
        return HTMLResponse(render_html_tearsheet(out))

    @router.post("/frontier")
    def frontier(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Risk/return frontier: realized (vol, return, Sharpe) по сетке бюджета риска (gross).

        Параметризуется ``gross_max`` (бюджет плеча) — при фиксированном gross mean-variance
        инвариантен к risk_aversion, поэтому фронтир строится по риск-бюджету. С учётом издержек
        (cost_bps·turnover) Sharpe деградирует на высоком плече — это и есть capacity-инсайт.
        """
        try:
            base = XSConfig.model_validate(payload)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"invalid config: {exc}")
        grid = payload.get("gross_grid") or [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
        panel = load_panel(base)  # один раз, переиспользуем для всех точек
        pts = []
        for g in grid:
            cfg = base.model_copy(deep=True)
            cfg.optimizer.gross_max = float(g)
            try:
                out = run_backtest(cfg, panel=panel)
                s = out["summary"]
                pts.append({
                    "gross": float(g),
                    "ann_vol": float(s.get("ann_vol", float("nan"))),
                    "ann_return": float(s.get("ann_return", float("nan"))),
                    "sharpe": float(s.get("sharpe", float("nan"))),
                    "max_drawdown": float(s.get("max_drawdown", float("nan"))),
                })
            except Exception:  # pragma: no cover
                continue
        return {"points": pts}

    @router.post("/weights")
    def weights(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Целевые веса последнего ребаланса (текущая оптимизированная cross-section)."""
        try:
            cfg = XSConfig.model_validate(payload)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"invalid config: {exc}")
        w = latest_target_weights(cfg)
        return {
            "weights": {str(s): float(v) for s, v in w.items()},
            "gross": float(w.abs().sum()) if len(w) else 0.0,
            "net": float(w.sum()) if len(w) else 0.0,
            "n_names": int(len(w)),
        }

    @router.post("/pretrade_risk")
    def pretrade_risk(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Pre-trade VaR/CVaR/стресс-сценарии целевого вектора весов ПЕРЕД ребалансом (P1)."""
        from service_pretrade_risk import PreTradeRiskAnalyzer, RiskLimits
        from service_risk_model import StatRiskModel as _SRM
        from core_portfolio import SYMBOL_LEVEL
        try:
            cfg = XSConfig.model_validate(payload)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"invalid config: {exc}")
        w = latest_target_weights(cfg)
        if not len(w):
            raise HTTPException(status_code=422, detail="no target weights")
        panel = load_panel(cfg)
        close = panel[cfg.backtest.price_col].unstack(level=SYMBOL_LEVEL)
        rets = close.pct_change().fillna(0.0)
        cov = _SRM(method="ledoit_wolf").fit(rets).cov()
        cov = cov.reindex(index=list(w.index), columns=list(w.index)).fillna(0.0)
        lim = payload.get("risk_limits", {}) or {}
        limits = RiskLimits(
            var_max=lim.get("var_max"), cvar_max=lim.get("cvar_max"),
            vol_max=lim.get("vol_max"), scenario_loss_max=lim.get("scenario_loss_max"),
        )
        an = PreTradeRiskAnalyzer(cov)
        rep = an.pretrade_check(w, limits=limits, returns=rets, strict=bool(lim))
        out = rep.to_dict()
        out["weights_gross"] = float(w.abs().sum())
        out["weights_net"] = float(w.sum())
        return out

    @router.post("/execution_plan")
    def execution_plan(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Impact-aware execution-plan (TWAP/VWAP/POV slices) для целевых весов (P1)."""
        from service_xs_execution import RebalanceScheduler
        from core_portfolio import SYMBOL_LEVEL
        try:
            cfg = XSConfig.model_validate(payload)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"invalid config: {exc}")
        w = latest_target_weights(cfg)
        if not len(w):
            raise HTTPException(status_code=422, detail="no target weights")
        panel = load_panel(cfg)
        close = panel[cfg.backtest.price_col].unstack(level=SYMBOL_LEVEL)
        prices = close.iloc[-1]
        adv = None
        if "volume" in panel.columns:
            vol = panel["volume"].unstack(level=SYMBOL_LEVEL)
            adv = (vol * close).tail(20).mean()
        eopt = payload.get("execution", {}) or {}
        sched = RebalanceScheduler(
            algo=str(eopt.get("algo", "TWAP")), n_slices=int(eopt.get("n_slices", 6)),
            participation=float(eopt.get("participation", 0.10)),
            spread_bps=float(eopt.get("spread_bps", 2.0)),
            impact_coef=float(eopt.get("impact_coef", 0.1)),
            urgency=float(eopt.get("urgency", 2.0)),   # P1 #10: IS (Almgren-Chriss) front-loading
        )
        equity = float(payload.get("equity", 1_000_000))
        plan = sched.build_plan(w, None, prices, equity, adv=adv)
        return plan.to_dict()

    @router.post("/signals")
    def signals(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        from impl_signal_diagnostics import signal_report
        from impl_panel import PanelBuilder
        from service_xs_pipeline import build_signal_library
        cfg = XSConfig.model_validate(payload)
        panel = load_panel(cfg)
        lib = build_signal_library(cfg)
        sig_panel = lib.compute(panel)
        fwd = PanelBuilder.add_forward_returns(panel, price_col=cfg.backtest.price_col)["fwd_return"]
        return {
            "signals": list(sig_panel.columns),
            "ic": {c: signal_report(sig_panel[c], fwd) for c in sig_panel.columns},
        }

    @router.post("/options/construct")
    def options_construct(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Greeks-нейтральный опционный портфель (Stage B5, отдельный greeks-оптимизатор).

        payload: {legs:[{symbol,spot,strike,time_to_expiry,iv,is_call,rate,dividend_yield,
        alpha,multiplier}...] | demo:true, neutralize:[..], gross_max, max_position}.
        """
        from service_options_portfolio import (
            OptionLeg, GreeksNeutralConstraints, OptionsPortfolioConstructor, synthetic_option_book,
        )
        legs_in = payload.get("legs")
        chain_in = payload.get("chain")
        if chain_in:
            # Stage D5: реальный бук из option chain (free Deribit/EOD)
            from loaders.options_enrich import OptionsBookLoader
            legs = OptionsBookLoader.chain_to_legs(chain_in, spot=float(payload.get("spot", 100.0)))
        elif not legs_in and payload.get("demo", True):
            legs = synthetic_option_book(spot=float(payload.get("spot", 100.0)))
        else:
            legs = []
            for d in (legs_in or []):
                legs.append(OptionLeg(
                    symbol=str(d["symbol"]), spot=float(d["spot"]), strike=float(d["strike"]),
                    time_to_expiry=float(d["time_to_expiry"]), iv=float(d["iv"]),
                    is_call=bool(d.get("is_call", True)), rate=float(d.get("rate", 0.0)),
                    dividend_yield=float(d.get("dividend_yield", 0.0)),
                    alpha=float(d.get("alpha", 0.0)), multiplier=float(d.get("multiplier", 100.0)),
                ))
        if not legs:
            raise HTTPException(status_code=422, detail="legs or demo required")
        cons = GreeksNeutralConstraints(
            neutralize=list(payload.get("neutralize", ["delta", "vega"])),
            gross_max=float(payload.get("gross_max", 1.0)),
            max_position=payload.get("max_position"),
        )
        port = OptionsPortfolioConstructor(cons).construct(legs)
        d = port.to_dict()
        d["n_legs"] = len(legs)
        return d

    @router.post("/data_quality")
    def data_quality(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Honest Data-Quality отчёт собранной панели (Stage D0): провенанс/pit_quality/coverage."""
        from service_xs_pipeline import data_quality_for_config
        try:
            cfg = XSConfig.model_validate(payload)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"invalid config: {exc}")
        return data_quality_for_config(cfg).to_dict()

    @router.post("/data_trust")
    def data_trust(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Data-Trust gate (Stage D7): PIT-lineage сигналов + trust_verdict + violations."""
        from service_xs_pipeline import data_trust_for_config
        try:
            cfg = XSConfig.model_validate(payload)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"invalid config: {exc}")
        return data_trust_for_config(cfg)

    @router.post("/cross_asset")
    def cross_asset(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Unified cross-asset портфель (Stage C1): объединяет directional-вертикали в один портфель.

        payload: {configs:{name: xsconfig,...} | demo:true, target_vol, class_weighting}.
        demo:true → синтетические пресеты crypto+equity+futures+forex.
        """
        from service_cross_asset import combine_from_configs

        target_vol = float(payload.get("target_vol", 0.10))
        weighting = str(payload.get("class_weighting", "risk_parity"))
        cfgs_in = payload.get("configs")
        if not cfgs_in and payload.get("demo", True):
            def _demo(ac, syms, kind):
                return XSConfig.model_validate({
                    "asset_class": ac,
                    "data": {"source": "synthetic", "symbols": syms, "synthetic_bars": 200, "synthetic_seed": 7},
                    "universe": {"type": "static", "symbols": syms},
                    "signals": [{"name": "m", "kind": kind, "lookback": 40, "transforms": ["zscore"]}],
                    "alpha": {"method": "ic_weighted"}, "risk": {"type": "stat", "method": "ledoit_wolf"},
                    "optimizer": {"objective": "mean_variance", "risk_aversion": 5.0, "gross_max": 1.0,
                                  "net_target": 0.0, "long_only": False, "max_position": 0.4},
                    "backtest": {"rebalance_every": 5, "cov_lookback": 40, "min_cov_obs": 20,
                                 "alpha_refit_every": 5, "cost_bps": 3.0, "price_col": "close",
                                 "periods_per_year": 252},
                })
            configs = {
                "crypto": _demo("crypto", ["BTC", "ETH", "SOL", "BNB"], "crypto_momentum"),
                "equity": _demo("equity", ["AAPL", "MSFT", "NVDA", "XOM"], "equity_momentum"),
                "futures": _demo("futures", ["ES", "NQ", "CL", "GC"], "trend"),
                "forex": _demo("forex", ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY"], "fx_momentum"),
            }
        else:
            configs = {name: XSConfig.model_validate(c) for name, c in (cfgs_in or {}).items()}
        if not configs:
            raise HTTPException(status_code=422, detail="configs or demo required")
        res = combine_from_configs(configs, target_vol=target_vol, class_weighting=weighting)
        return res.to_dict()

    @router.post("/live/rebalance")
    def live_rebalance(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        tw = pd.Series({s: float(v) for s, v in payload.get("target_weights", {}).items()})
        equity = float(payload.get("equity", 1.0))
        lim = payload.get("limits", {})
        guard = PortfolioRiskGuard(PortfolioRiskLimits(
            gross_max=lim.get("gross_max"), net_max=lim.get("net_max"),
            max_position=lim.get("max_position"), max_turnover=lim.get("max_turnover"),
        ))

        # Default: dry-run — Cloud forms Intents (target exposures) only (CCEA).
        # Opt-in execute=true runs a PAPER sim broker end-to-end here purely to
        # demonstrate/validate the XS->Agent->broker bridge; it never touches real
        # broker secrets (those live only in the Agent). Real live execution runs in
        # the Agent via `script_xs_live.py --live`.
        if not bool(payload.get("execute", False)):
            runner = CrossSectionalLiveRunner(risk_guard=guard)  # no agent → dry-run
            res = runner.rebalance(tw, equity, ts_ms=int(payload.get("ts_ms", 0)))
            return res.to_dict()

        # --- paper execution path (lazy agent-zone import; simulated only) ---
        try:
            from packages.agent.broker.adapters.sim import SimBrokerConnector
            from packages.agent.execution.live_factory import build_live_stack
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"agent execution unavailable: {exc}")

        prices = {str(s): float(p) for s, p in (payload.get("prices") or {}).items()}
        if not prices:
            raise HTTPException(
                status_code=422,
                detail="execute=true requires 'prices' {symbol: price} to size paper orders",
            )
        positions = {str(s): float(n) for s, n in (payload.get("positions") or {}).items()}
        n_slices = int(payload.get("n_slices", 1))

        broker = SimBrokerConnector(prices, equity=equity,
                                    fill_ratio=float(payload.get("fill_ratio", 1.0)))
        for sym, qty in positions.items():
            px = prices.get(sym)
            if px:
                broker._positions[sym] = __import__("decimal").Decimal(str(qty / px))
        _clk = [float(payload.get("ts_ms", 0)) / 1000.0 or 0.0]
        stack = build_live_stack(
            broker, n_slices=n_slices, symbols=list(prices),
            min_trade_notional=float(payload.get("min_trade_notional", 1.0)),
            slice_interval_s=float(payload.get("slice_interval_s", 1.0)),
            clock=lambda: _clk[0],
        )
        runner = CrossSectionalLiveRunner(
            risk_guard=guard,
            agent_client=stack["agent_client"],
            position_provider=stack["agent_client"]._position_provider,
        )
        res = runner.rebalance(tw, equity, ts_ms=int(payload.get("ts_ms", 0)))

        # pump the execution to completion (sim fills immediately / over the clock)
        ac = stack["agent_client"]
        pumps = []
        for i in range(int(payload.get("pump_steps", 8))):
            _clk[0] += float(payload.get("slice_interval_s", 1.0))
            summary = ac.pump(now_ts=_clk[0])
            pumps.append(summary)
            if summary.get("complete"):
                break

        out = res.to_dict()
        out["execution"] = {
            "simulated": True,
            "data_source": "paper_sim_broker",
            "orders": [o.to_dict() for o in stack["engine"]._orders_by_client_id.values()],
            "positions": {p.symbol: float(p.market_value) for p in broker.get_positions()},
            "pumps": len(pumps),
            "complete": pumps[-1].get("complete") if pumps else True,
        }
        return out

    return router


def register_xs_routes(app: Any) -> None:
    """Подключить cross-sectional роуты к существующему FastAPI-приложению."""
    app.include_router(make_xs_router())


__all__ = ["make_xs_router", "register_xs_routes"]
