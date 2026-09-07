# -*- coding: utf-8 -*-
"""
Tests for P2 blockers #13–#26.

  #13 real FIX session (logon/order/seqnum/heartbeat over a paired transport)
  #14 multi-venue SOR with live liquidity + real dispatch
  #15 robust (μ-uncertainty) + multi-period optimization
  #16 per-name √-impact transaction-cost capacity
  #17 BARRA-style equity factors built from fundamentals (value/quality/low_vol)
  #18 crypto BTC-beta risk factor
  #19 persistent tick/L2/L3 store
  #20 market-abuse surveillance (spoofing/layering/wash/marking-the-close)
  #21 durable alert store + tamper-evident audit chain
  #22 GBM alpha works without sklearn
  #23 reproducibility: env fingerprint + dirty-tree promotion block
  #24 drift KS/Wasserstein/concept + closed-loop retrain
  #25 average-price allocation + give-up/CMTA + T+1 settlement
  #26 IB / OANDA Agent broker connectors (protocol-complete)
"""

import time
from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest


# --------------------------- #13 FIX session ---------------------------
def test_fix_session_logon_order_seqnum():
    from packages.agent.execution.fix_session import FixSessionEngine, PairedTransport, SessionState

    ta, tb = PairedTransport.pair()
    rx = []
    acc = FixSessionEngine(
        "BROKER",
        "RIVEN",
        transport=tb,
        heartbeat_int=2,
        is_acceptor=True,
        on_app_message=lambda f: rx.append(f),
    )
    ini = FixSessionEngine("RIVEN", "BROKER", transport=ta, heartbeat_int=2)
    acc.connect()
    ini.connect()
    try:
        assert ini.logged_on.wait(5) and acc.logged_on.wait(5)
        assert ini.state == SessionState.ACTIVE
        ini.send_new_order(
            cl_ord_id="C1", symbol="AAPL", side="BUY", qty=100, ord_type="LIMIT", price=150.0
        )
        time.sleep(0.5)
        assert any(f.get("35") == "D" and f.get("11") == "C1" and f.get("38") == "100" for f in rx)
        assert ini.status()["out_seq"] >= 3
    finally:
        ini.disconnect()
        acc.disconnect()


# --------------------------- #14 SOR live + dispatch ---------------------------
def test_sor_live_liquidity_and_dispatch():
    from packages.agent.execution.smart_order_router import SmartOrderRouter, Venue, VenueQuote

    venues = [
        Venue("NYSE", fee_bps=0.5, impact_coef=0.05),
        Venue("NASDAQ", fee_bps=0.6, impact_coef=0.05),
    ]
    quotes = {
        "NYSE": VenueQuote(99.98, 100.02, 5000, 5000),
        "NASDAQ": VenueQuote(99.99, 100.03, 3000, 3000),
    }

    class LP:
        def get_quote(self, venue, symbol):
            return quotes.get(venue)

    sor = SmartOrderRouter(venues)
    route = sor.route_live("AAPL", "BUY", 500_000, LP())
    assert len(route.allocations) >= 2
    sent = []
    disp = sor.dispatch(
        route, lambda v, s, sd, n: (sent.append(v) or {"success": True, "broker_order_id": v})
    )
    assert disp["all_ok"] and len(sent) == len(route.allocations)


# --------------------------- #15 robust + multi-period ---------------------------
def test_robust_and_multiperiod():
    from service_optimizer import (
        PortfolioOptimizer,
        OptimizerConstraints,
        TCostModel,
        RobustConfig,
        MultiPeriodOptimizer,
    )

    syms = [f"S{i}" for i in range(6)]
    mu = pd.Series([0.05] * 6, index=syms)
    Sigma = pd.DataFrame(np.eye(6) * 0.04, index=syms, columns=syms)
    base = PortfolioOptimizer(
        objective="mean_variance",
        risk_aversion=3.0,
        constraints=OptimizerConstraints(gross_max=2.0),
        tcost=TCostModel(linear=0),
    )
    w0 = base.solve(mu, Sigma)
    rob = PortfolioOptimizer(
        objective="mean_variance",
        risk_aversion=3.0,
        constraints=OptimizerConstraints(gross_max=2.0),
        tcost=TCostModel(linear=0),
        robust=RobustConfig(enabled=True, kind="box", kappa=0.3),
    )
    wr = rob.solve(mu, Sigma)
    assert abs(wr).sum() <= abs(w0).sum() + 1e-9  # robust never larger
    mp = MultiPeriodOptimizer(base, trade_cost=5.0)
    assert mp.trade_rate_used < 1.0  # high cost → trade gradually
    path = mp.solve_path(pd.DataFrame({s: [0.05, 0.05, 0.05] for s in syms}), Sigma)
    assert path.shape[0] == 3


def test_per_name_sqrt_impact():
    from service_optimizer import PortfolioOptimizer, OptimizerConstraints, TCostModel

    syms = [f"S{i}" for i in range(8)]
    mu = pd.Series([0.03] * 8, index=syms)
    Sigma = pd.DataFrame(np.eye(8) * 0.05, index=syms, columns=syms)
    adv = pd.Series([1e9] * 4 + [1e5] * 4, index=syms)  # last 4 illiquid
    tc = TCostModel(linear=0.0, sqrt_impact=True, impact_coef=0.3)
    opt = PortfolioOptimizer(
        objective="mean_variance",
        risk_aversion=3.0,
        constraints=OptimizerConstraints(gross_max=1.0),
        tcost=tc,
    )
    w = opt.solve(mu, Sigma, adv=adv, nav=1e7)
    assert sum(abs(w[s]) for s in syms[:4]) > sum(abs(w[s]) for s in syms[4:])


# --------------------------- #17/#18 factors ---------------------------
def test_equity_barra_factors_from_fundamentals():
    from xs_risk.equity_factors import build_equity_exposures

    syms = [f"S{i}" for i in range(8)]
    rw = pd.DataFrame(np.random.default_rng(0).normal(0, 0.02, (80, 8)), columns=syms)
    B = build_equity_exposures(
        rw,
        mcaps={s: 1e9 * (i + 1) for i, s in enumerate(syms)},
        earnings={s: 1e8 * (i + 1) for i, s in enumerate(syms)},
        book={s: 5e8 for s in syms},
        roe={s: 0.1 * (i + 1) for i, s in enumerate(syms)},
        momentum_lookback=30,
    )
    assert {"market_beta", "size", "value", "quality", "momentum", "low_vol"} <= set(B.columns)


def test_crypto_btc_beta_factor():
    from xs_risk.crypto_factors import build_crypto_exposures

    syms = [f"S{i}" for i in range(6)]
    rw = pd.DataFrame(np.random.default_rng(1).normal(0, 0.02, (60, 6)), columns=syms)
    B = build_crypto_exposures(rw, btc_symbol="S0", mcaps={s: 1e9 for s in syms})
    assert "btc_beta" in B.columns


# --------------------------- #19 tick store ---------------------------
def test_tick_store_persist_and_query(tmp_path):
    from lob.tick_store import TickStore, TRADE, DEPTH

    ts = TickStore(root=str(tmp_path), flush_every=3)
    for i in range(10):
        ts.record_trade("AAPL", 1700000000000 + i * 1000, 100 + i * 0.1, 50)
    ts.record_depth("AAPL", 1700000000000, bids=[(100, 5)], asks=[(100.1, 4)])
    ts.flush()
    assert len(ts.query("AAPL", kind=TRADE)) == 10
    assert len(ts.query("AAPL", kind=TRADE, start_ms=1700000000000 + 5000)) == 5
    assert "bid_px_0" in ts.query("AAPL", kind=DEPTH).columns


# --------------------------- #20 market abuse ---------------------------
def test_market_abuse_spoofing_and_wash():
    from services.algo_integration.market_abuse import (
        MarketAbuseMonitor,
        MarketAbuseConfig,
        OrderEvent,
        TradeEvent,
    )

    m = MarketAbuseMonitor(
        MarketAbuseConfig(
            spoof_large_qty=1000, spoof_cancel_ms=5000, spoof_min_events=3, spoof_min_distance_bps=5
        )
    )
    t0 = 1700000000000
    for i in range(4):
        m.record_order(
            OrderEvent(t0 + i * 1000, "AAPL", "ACC1", "BUY", "NEW", 2000, 99.0, f"o{i}", mid=100.0)
        )
        m.record_order(
            OrderEvent(
                t0 + i * 1000 + 500, "AAPL", "ACC1", "BUY", "CANCEL", 2000, 99.0, f"o{i}", mid=100.0
            )
        )
    assert len(m.get_alerts(pattern="spoofing")) >= 1
    m.record_trade(TradeEvent(t0, "MSFT", "ACC2", "BUY", 100, 200.0))
    assert len(m.record_trade(TradeEvent(t0 + 1000, "MSFT", "ACC2", "SELL", 100, 200.05))) >= 1


# --------------------------- #21 durable persistence ---------------------------
def test_durable_alert_store_and_audit_chain(tmp_path):
    from services.core.durable_store import DurableAlertStore, AuditChain

    dbp = str(tmp_path / "a.db")
    s = DurableAlertStore(dbp)
    s.save({"alert_id": "A1", "triggered_at": "t", "severity": "high", "status": "triggered"})
    s.update_status("A1", "resolved")
    assert DurableAlertStore(dbp).load_all()[0]["status"] == "resolved"  # survives "restart"
    chain = AuditChain(str(tmp_path / "c.db"))
    for i in range(5):
        chain.append("order", {"id": i})
    assert chain.verify() is True
    chain._conn.execute("UPDATE audit SET payload='{\"id\":9}' WHERE seq=2")
    chain._conn.commit()
    assert chain.verify() is False  # tamper detected


# --------------------------- #22 GBM without sklearn ---------------------------
def test_gbm_alpha_without_sklearn():
    from impl_gbrt import GradientBoostingRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = X[:, 0] * 2 - X[:, 1]
    gb = GradientBoostingRegressor(n_estimators=40, max_depth=3).fit(X, y)
    assert np.corrcoef(gb.predict(X), y)[0, 1] > 0.9
    from service_alpha import GBMAlpha

    sig = pd.DataFrame(X, columns=[f"s{i}" for i in range(4)])
    mu = GBMAlpha(n_estimators=20).fit(sig, pd.Series(y, index=sig.index)).predict(sig)
    assert mu.shape == (300,)


# --------------------------- #23 reproducibility ---------------------------
def test_env_capture_and_dirty_promote_block(tmp_path):
    from service_experiment_tracking import capture_environment, ModelRegistry
    from core_experiment import Lineage

    env = capture_environment()
    assert env["python_version"] and len(env["package_versions"]) >= 2
    reg = ModelRegistry(root=str(tmp_path / "registry"))
    art = tmp_path / "model.json"
    art.write_text("{}")
    mv = reg.register(
        "m",
        run_id="r1",
        artifact_path=str(art),
        metrics={"s": 1.0},
        lineage=Lineage(git_dirty=True),
    )
    with pytest.raises(ValueError):
        reg.transition("m", mv.version, "production")  # dirty tree → blocked
    reg.transition("m", mv.version, "production", force=True)  # force allowed
    assert reg.get("m", stage="production").version == mv.version


# --------------------------- #24 drift ---------------------------
def test_drift_ks_wasserstein_concept_and_closed_loop():
    from drift import ks_statistic, wasserstein1d, concept_drift

    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 2000)
    b = rng.normal(0.8, 1, 2000)
    assert ks_statistic(a, b) > 0.2 and wasserstein1d(a, b) > 0.5
    assert concept_drift(a, a * 0, b, b * 0)["concept_drift"] is True
    from services.automation.drift_retrain import DriftRetrainScheduler

    sched = DriftRetrainScheduler(psi_threshold=0.25, cooldown_sec=0)
    calls = {"r": 0, "p": 0}
    res = sched.run_closed_loop(
        {"f": {"psi": 0.4}},
        retrain_fn=lambda d: (calls.__setitem__("r", 1) or type("A", (), {"name": "m"})()),
        register_fn=lambda a, d: None,
        verify_fn=lambda a: True,
        promote_fn=lambda a: calls.__setitem__("p", 1),
    )
    assert res["retrained"] and res["promoted"]


# --------------------------- #25 allocation/clearing ---------------------------
def test_average_price_allocation_and_settlement():
    from packages.agent.execution.allocation import ClearingEngine, Fill, SubAccountTarget, GiveUp

    out = ClearingEngine().process_block(
        symbol="AAPL",
        side="BUY",
        fills=[Fill(600, "150.00"), Fill(400, "150.50")],
        targets=[SubAccountTarget("A", 700), SubAccountTarget("B", 300)],
        trade_date=date(2026, 6, 15),
        asset_class="equity",
        give_up=GiveUp("EXEC", "CLEAR", cmta_code="CMTA1"),
    )
    al = out["allocation"]
    assert al["avg_price"] == "150.20"
    assert sum(Decimal(a["qty"]) for a in al["allocations"]) == Decimal("1000")
    assert al["give_up"]["cmta_code"] == "CMTA1"
    assert out["settlement_date"] == "2026-06-16"  # T+1 business day
    assert "A" in out["net_obligations"] and "B" in out["net_obligations"]


# --------------------------- #26 IB/OANDA connectors ---------------------------
class _FakeBackend:
    def __init__(self):
        self.oid = 0
        self.pos = {}

    def place(self, **k):
        self.oid += 1
        q = k["qty"] * (1 if k["side"] == "BUY" else -1)
        self.pos[k["symbol"]] = self.pos.get(k["symbol"], 0) + q
        return {
            "success": True,
            "broker_order_id": f"B{self.oid}",
            "status": "filled",
            "filled_qty": k["qty"],
            "avg_price": 100.0,
        }

    def cancel(self, b):
        return True

    def order(self, b):
        return {"status": "filled", "filled_qty": 1, "avg_price": 100.0}

    def positions(self):
        return [
            {"symbol": s, "qty": v, "avg_price": 100.0, "price": 100.0, "market_value": v * 100.0}
            for s, v in self.pos.items()
        ]

    def account(self):
        return {"equity": 100000, "cash": 50000, "buying_power": 100000}

    def last_price(self, s):
        return 100.0


@pytest.mark.parametrize("which", ["ib", "oanda"])
def test_agent_connectors(which):
    from packages.agent.broker.protocol import BrokerCredentials, OrderRequest, OrderSide, OrderType

    if which == "ib":
        from packages.agent.broker.adapters.ib import IBConnector as Conn
    else:
        from packages.agent.broker.adapters.oanda import OANDAConnector as Conn
    c = Conn(BrokerCredentials(api_key="k", api_secret="s"), backend=_FakeBackend())
    assert c.connect() and c.broker_name == which
    r = c.submit_order(
        OrderRequest(
            client_order_id="X1",
            symbol="SYM",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("2"),
        )
    )
    assert r.success and r.status.value == "filled"
    assert len(c.get_positions()) == 1 and float(c.get_account().equity) == 100000


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
