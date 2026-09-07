"""Тесты авторизации авто-торговли на live-брокере (Agent-зона).

Закрывает последний пункт §4.9 (авто-ребаланс на live через CCEA approval).
Проверяются: двухшаговая церемония, привязка к хешу конфига, потолок лимитов,
TTL, бюджет, revoke, tamper-evident аудит, hard-caps, долговечность.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from packages.agent.approval.live_trading_authorization import (
    HARD_MAX_NOTIONAL_PER_REBALANCE,
    HARD_MAX_TTL_SEC,
    HARD_MAX_TURNOVER,
    LimitCeiling,
    LiveTradingAuthorizationStore,
    canonical_config_hash,
)


class Clock:
    def __init__(self, ts=1_000_000.0):
        self.ts = float(ts)

    def __call__(self):
        return self.ts

    def advance(self, s):
        self.ts += s


CFG = {"name": "crypto_alpha", "universe": ["BTC", "ETH"], "optimizer": "mvo"}
CEIL = LimitCeiling(
    max_turnover=0.10, max_notional_per_rebalance=100_000.0, max_orders_per_rebalance=25
)


def make_store(
    tmp_path: Path, clock: Clock, key: bytes = b"audit-key"
) -> LiveTradingAuthorizationStore:
    return LiveTradingAuthorizationStore(
        state_path=str(tmp_path / "auth.json"),
        audit_path=str(tmp_path / "audit.jsonl"),
        audit_key=key,
        time_fn=clock,
    )


def grant(store, clock, *, broker="binance", ttl=3600, token="T", expected="T", **kw):
    return store.grant(
        strategy_id="xs-rebalance",
        config=CFG,
        broker=broker,
        limit_ceiling=CEIL,
        ttl_sec=ttl,
        confirmation_token=token,
        expected_token=expected,
        **kw,
    )


# ------------------------------------------------------------- config hash


def test_config_hash_stable_and_order_independent():
    a = canonical_config_hash({"a": 1, "b": 2})
    b = canonical_config_hash({"b": 2, "a": 1})
    assert a == b and len(a) == 64
    assert canonical_config_hash({"a": 1, "b": 3}) != a  # изменение → другой хеш
    # Готовый hex-digest возвращается как есть.
    assert canonical_config_hash(a) == a


# ------------------------------------------------------------- церемония


def test_grant_requires_matching_token(tmp_path):
    clock = Clock()
    store = make_store(tmp_path, clock)
    bad = grant(store, clock, token="WRONG", expected="RIGHT")
    assert bad["ok"] is False and "токен" in bad["error"]
    good = grant(store, clock, token="RIGHT", expected="RIGHT")
    assert good["ok"] is True and good["authorization"]["active"] is True


def test_sim_paper_rejected(tmp_path):
    clock = Clock()
    store = make_store(tmp_path, clock)
    res = grant(store, clock, broker="sim_paper")
    assert res["ok"] is False and "sim_paper" in res["error"]


# ------------------------------------------------------------- check happy


def test_check_allows_within_ceiling(tmp_path):
    clock = Clock()
    store = make_store(tmp_path, clock)
    grant(store, clock)
    chk = store.check(
        strategy_id="xs-rebalance",
        config=CFG,
        broker="binance",
        turnover=0.08,
        notional=50_000.0,
        n_orders=10,
    )
    assert chk.allowed is True and chk.effective_ceiling.max_turnover == 0.10


def test_precheck_zero_returns_ceiling(tmp_path):
    clock = Clock()
    store = make_store(tmp_path, clock)
    grant(store, clock)
    chk = store.check(
        strategy_id="xs-rebalance",
        config=CFG,
        broker="binance",
        turnover=0.0,
        notional=0.0,
        n_orders=0,
    )
    assert chk.allowed and chk.effective_ceiling is not None


# ------------------------------------------------------------- fail-closed


def test_no_authorization_blocks(tmp_path):
    clock = Clock()
    store = make_store(tmp_path, clock)
    chk = store.check(
        strategy_id="xs-rebalance",
        config=CFG,
        broker="binance",
        turnover=0.01,
        notional=100.0,
        n_orders=1,
    )
    assert chk.allowed is False and "нет активной" in chk.reason


def test_config_change_invalidates_authorization(tmp_path):
    clock = Clock()
    store = make_store(tmp_path, clock)
    grant(store, clock)  # мандат на CFG
    changed = {**CFG, "optimizer": "risk_parity"}  # оператор изменил стратегию
    chk = store.check(
        strategy_id="xs-rebalance",
        config=changed,
        broker="binance",
        turnover=0.01,
        notional=100.0,
        n_orders=1,
    )
    assert chk.allowed is False and "конфиг" in chk.reason


def test_broker_binding(tmp_path):
    clock = Clock()
    store = make_store(tmp_path, clock)
    grant(store, clock, broker="binance")
    chk = store.check(
        strategy_id="xs-rebalance",
        config=CFG,
        broker="oanda",
        turnover=0.01,
        notional=100.0,
        n_orders=1,
    )
    assert chk.allowed is False  # мандат binance ≠ oanda


@pytest.mark.parametrize(
    "field,value,word",
    [
        ("turnover", 0.5, "turnover"),
        ("notional", 500_000.0, "notional"),
        ("n_orders", 100, "orders"),
    ],
)
def test_ceiling_enforced(tmp_path, field, value, word):
    clock = Clock()
    store = make_store(tmp_path, clock)
    grant(store, clock)
    args = dict(turnover=0.01, notional=100.0, n_orders=1)
    args[field] = value
    chk = store.check(strategy_id="xs-rebalance", config=CFG, broker="binance", **args)
    assert chk.allowed is False and word in chk.reason


def test_ttl_expiry(tmp_path):
    clock = Clock()
    store = make_store(tmp_path, clock)
    grant(store, clock, ttl=3600)
    clock.advance(3601)
    chk = store.check(
        strategy_id="xs-rebalance",
        config=CFG,
        broker="binance",
        turnover=0.01,
        notional=100.0,
        n_orders=1,
    )
    assert chk.allowed is False and "нет активной" in chk.reason


# ------------------------------------------------------------- бюджет


def test_notional_budget_exhaustion(tmp_path):
    clock = Clock()
    store = make_store(tmp_path, clock)
    g = grant(store, clock, max_total_notional=150_000.0)
    aid = g["authorization"]["auth_id"]
    # первый ребаланс на $100k проходит и потребляет
    assert store.check(
        strategy_id="xs-rebalance",
        config=CFG,
        broker="binance",
        turnover=0.05,
        notional=100_000.0,
        n_orders=5,
    ).allowed
    store.consume(aid, notional=100_000.0, n_orders=5)
    # второй на $100k превысит бюджет $150k → блок
    chk = store.check(
        strategy_id="xs-rebalance",
        config=CFG,
        broker="binance",
        turnover=0.05,
        notional=100_000.0,
        n_orders=5,
    )
    assert chk.allowed is False and "нотиональный бюджет" in chk.reason


def test_rebalance_count_budget(tmp_path):
    clock = Clock()
    store = make_store(tmp_path, clock)
    g = grant(store, clock, max_rebalances=2)
    aid = g["authorization"]["auth_id"]
    for _ in range(2):
        assert store.check(
            strategy_id="xs-rebalance",
            config=CFG,
            broker="binance",
            turnover=0.01,
            notional=100.0,
            n_orders=1,
        ).allowed
        c = store.consume(aid, notional=100.0, n_orders=1)
    assert c["exhausted"] is True
    # После исчерпания мандат помечен EXHAUSTED → перестаёт быть активным.
    chk = store.check(
        strategy_id="xs-rebalance",
        config=CFG,
        broker="binance",
        turnover=0.01,
        notional=100.0,
        n_orders=1,
    )
    assert chk.allowed is False
    assert store.status()["active"] == []


# ------------------------------------------------------------- revoke


def test_revoke_blocks_immediately(tmp_path):
    clock = Clock()
    store = make_store(tmp_path, clock)
    g = grant(store, clock)
    aid = g["authorization"]["auth_id"]
    store.revoke(aid)
    chk = store.check(
        strategy_id="xs-rebalance",
        config=CFG,
        broker="binance",
        turnover=0.01,
        notional=100.0,
        n_orders=1,
    )
    assert chk.allowed is False


def test_revoke_all(tmp_path):
    clock = Clock()
    store = make_store(tmp_path, clock)
    grant(store, clock, broker="binance")
    grant(store, clock, broker="oanda")
    res = store.revoke_all(reason="halt")
    assert res["revoked"] == 2
    assert store.status()["active"] == []


def test_new_grant_supersedes_old_for_same_strategy_broker(tmp_path):
    clock = Clock()
    store = make_store(tmp_path, clock)
    g1 = grant(store, clock)
    g2 = grant(store, clock)
    assert g1["authorization"]["auth_id"] != g2["authorization"]["auth_id"]
    active = store.status()["active"]
    assert len(active) == 1 and active[0]["auth_id"] == g2["authorization"]["auth_id"]


# ------------------------------------------------------------- hard caps


def test_hard_caps_clamp(tmp_path):
    clock = Clock()
    store = make_store(tmp_path, clock)
    huge = LimitCeiling(
        max_turnover=99.0, max_notional_per_rebalance=1e12, max_orders_per_rebalance=10
    )
    res = store.grant(
        strategy_id="s",
        config=CFG,
        broker="binance",
        limit_ceiling=huge,
        ttl_sec=99 * 24 * 3600,
        confirmation_token="T",
        expected_token="T",
    )
    auth = res["authorization"]
    assert auth["limit_ceiling"]["max_turnover"] == HARD_MAX_TURNOVER
    assert auth["limit_ceiling"]["max_notional_per_rebalance"] == HARD_MAX_NOTIONAL_PER_REBALANCE
    assert auth["seconds_remaining"] <= HARD_MAX_TTL_SEC


# ------------------------------------------------------ аудит + durability


def test_audit_chain_valid_and_records_events(tmp_path):
    clock = Clock()
    store = make_store(tmp_path, clock)
    g = grant(store, clock)
    store.consume(g["authorization"]["auth_id"], notional=100.0, n_orders=1)
    store.revoke(g["authorization"]["auth_id"])
    audit = store.verify_audit()
    assert audit["valid"] is True and audit["n"] >= 3
    events = [
        json.loads(l)["event"]
        for l in Path(store.audit_path).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    assert "GRANT" in events and "CONSUME" in events and "REVOKE" in events


def test_state_survives_reload(tmp_path):
    clock = Clock()
    store = make_store(tmp_path, clock)
    g = grant(store, clock, ttl=3600)
    aid = g["authorization"]["auth_id"]
    store.consume(aid, notional=5_000.0, n_orders=2)
    # «Рестарт»: новый стор из тех же файлов.
    store2 = make_store(tmp_path, clock)
    chk = store2.check(
        strategy_id="xs-rebalance",
        config=CFG,
        broker="binance",
        turnover=0.01,
        notional=100.0,
        n_orders=1,
    )
    assert chk.allowed is True  # мандат пережил рестарт
    active = store2.status()["active"][0]
    assert active["consumed_notional"] == 5_000.0
    assert store2.verify_audit()["valid"] is True  # цепочка аудита продолжена
