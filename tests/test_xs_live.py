# -*- coding: utf-8 -*-
"""
Stage A11 tests — live ребаланс (Intents) + portfolio risk guards.

  * build_intents: target_notional = weight×equity; idempotency-ключ детерминирован
  * CCEA: Intent НЕ содержит order-полей (side/qty/price)
  * risk guard блокирует нарушение лимитов (gross/factor/sector/turnover)
  * reconciliation ловит расхождение позиций
  * rebalance end-to-end: approved → Intents отправлены Agent; blocked → не отправлены
"""

from __future__ import annotations

import pandas as pd
import pytest

from service_xs_portfolio_risk import PortfolioRiskGuard, PortfolioRiskLimits
from service_xs_live import (
    CrossSectionalLiveRunner,
    Intent,
    _FORBIDDEN_FIELDS,
)


class MockPositions:
    def __init__(self, positions):
        self._p = positions

    def get_positions(self):
        return dict(self._p)


class MockAgent:
    def __init__(self):
        self.received = []

    def send_intents(self, batch):
        self.received.append(batch)


def _w(d):
    return pd.Series(d, dtype="float64")


# ---------------------------------------------------------------------------
# Intents
# ---------------------------------------------------------------------------
def test_build_intents_notional_and_idempotency():
    r = CrossSectionalLiveRunner()
    batch = r.build_intents(_w({"A": 0.5, "B": -0.3}), 1000.0, ts_ms=123)
    by_sym = {i.symbol: i for i in batch.intents}
    assert by_sym["A"].target_notional == pytest.approx(500.0)
    assert by_sym["B"].target_notional == pytest.approx(-300.0)
    assert len(batch.idempotency_key) == 32
    # детерминизм ключа
    batch2 = r.build_intents(_w({"A": 0.5, "B": -0.3}), 1000.0, ts_ms=123)
    assert batch.idempotency_key == batch2.idempotency_key


def test_ccea_intent_has_no_order_fields():
    assert set(Intent.__dataclass_fields__) == {"symbol", "target_weight", "target_notional"}
    batch = CrossSectionalLiveRunner().build_intents(_w({"A": 1.0}), 100.0, ts_ms=1)
    payload = batch.to_dict()
    for intent in payload["intents"]:
        assert _FORBIDDEN_FIELDS.isdisjoint(intent.keys())  # нет side/qty/price


# ---------------------------------------------------------------------------
# risk guard
# ---------------------------------------------------------------------------
def test_guard_blocks_gross():
    guard = PortfolioRiskGuard(PortfolioRiskLimits(gross_max=1.0))
    d = guard.check(_w({"A": 1.0, "B": -1.0}))  # gross=2.0
    assert not d.approved and any("gross" in v for v in d.violations)
    d2 = guard.check(_w({"A": 0.5, "B": -0.5}))  # gross=1.0
    assert d2.approved


def test_guard_factor_sector_turnover():
    B = pd.DataFrame({"mom": [1.0, 1.0]}, index=["A", "B"])
    fguard = PortfolioRiskGuard(PortfolioRiskLimits(factor_caps={"mom": 0.5}, exposures=B))
    assert not fguard.check(_w({"A": 0.4, "B": 0.4})).approved  # Bᵀw=0.8>0.5

    sguard = PortfolioRiskGuard(
        PortfolioRiskLimits(max_sector=0.5, sector_map={"A": "tech", "B": "tech"})
    )
    assert not sguard.check(_w({"A": 0.4, "B": 0.4})).approved  # sector=0.8>0.5

    tguard = PortfolioRiskGuard(PortfolioRiskLimits(max_turnover=0.1))
    assert not tguard.check(_w({"A": 0.5, "B": -0.5}), _w({"A": 0.0, "B": 0.0})).approved


# ---------------------------------------------------------------------------
# reconciliation
# ---------------------------------------------------------------------------
def test_reconcile_detects_discrepancy():
    r = CrossSectionalLiveRunner(position_provider=MockPositions({"A": 50.0, "B": -30.0}))
    recon = r.reconcile(_w({"A": 0.5, "B": -0.5}), 100.0)  # expected -50 для B, actual -30
    assert recon["in_sync"] is False
    assert "B" in recon["discrepancies"] and "A" not in recon["discrepancies"]


# ---------------------------------------------------------------------------
# rebalance end-to-end
# ---------------------------------------------------------------------------
def test_rebalance_approved_sends_intents():
    agent = MockAgent()
    r = CrossSectionalLiveRunner(
        risk_guard=PortfolioRiskGuard(PortfolioRiskLimits(gross_max=2.0)),
        position_provider=MockPositions({"A": 50.0, "B": -50.0}),
        agent_client=agent,
    )
    res = r.rebalance(_w({"A": 0.5, "B": -0.5}), 100.0, ts_ms=1)
    assert res.approved and res.sent
    assert len(agent.received) == 1
    assert res.batch is not None and len(res.batch.intents) == 2
    assert res.reconciliation["in_sync"] is True  # позиции совпадают с целью


def test_rebalance_blocked_does_not_send():
    agent = MockAgent()
    r = CrossSectionalLiveRunner(
        risk_guard=PortfolioRiskGuard(PortfolioRiskLimits(gross_max=1.0)),
        agent_client=agent,
    )
    res = r.rebalance(_w({"A": 1.0, "B": -1.0}), 100.0, ts_ms=1)  # gross=2.0 > 1.0
    assert not res.approved and not res.sent
    assert res.batch is None
    assert len(agent.received) == 0  # CCEA: ничего не отправлено Agent
