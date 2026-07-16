# -*- coding: utf-8 -*-
"""
Stage A3 tests — impl_universe (PIT / survivorship-free состав + ADV-фильтр).

Проверяем:
  * StaticUniverse: список, honest-флаг survivorship_biased=True, контракт
  * IndexMembershipUniverse: PIT-реконструкция состава; делистнутый-позже тикер
    присутствует в прошлом (survivorship-free); is_tradable по делистингу
  * ADVLiquidityFilter: режет неликвид по trailing dollar-volume; сохраняет флаг
  * соответствие core_portfolio.UniverseProvider
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import core_portfolio as cp
from impl_panel import PanelBuilder
from impl_universe import (
    ADVLiquidityFilter,
    IndexMembershipUniverse,
    StaticUniverse,
    date_to_ms,
    ms_to_date,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def test_ms_date_bridge_roundtrip():
    ms = date_to_ms("2020-06-01")
    assert ms_to_date(ms).isoformat() == "2020-06-01"


# ---------------------------------------------------------------------------
# StaticUniverse
# ---------------------------------------------------------------------------
def test_static_universe_contract_and_flag():
    u = StaticUniverse(["AAA", "BBB", "AAA"], name="t")
    assert isinstance(u, cp.UniverseProvider)
    assert u.survivorship_biased is True
    asof = date_to_ms("2024-01-01")
    assert u.constituents(asof) == ["AAA", "BBB"]  # дубликаты убраны, порядок сохранён
    assert u.is_tradable("AAA", asof) is True
    assert u.is_tradable("ZZZ", asof) is False
    assert u.describe()["survivorship_biased"] is True


def test_static_universe_tradable_window():
    lo, hi = date_to_ms("2020-01-01"), date_to_ms("2021-01-01")
    u = StaticUniverse(["AAA"], tradable_from_ms=lo, tradable_to_ms=hi)
    assert u.constituents(date_to_ms("2020-06-01")) == ["AAA"]
    assert u.constituents(date_to_ms("2019-06-01")) == []   # до окна
    assert u.constituents(date_to_ms("2021-06-01")) == []   # после окна


# ---------------------------------------------------------------------------
# IndexMembershipUniverse (PIT, survivorship-free)
# ---------------------------------------------------------------------------
def _membership_universe():
    return IndexMembershipUniverse.from_baseline(
        index="TESTIDX",
        constituents=["AAA", "BBB"],
        asof="2020-01-01",
        changes=[
            {"date": "2021-06-01", "added": ["CCC"], "removed": ["BBB"], "reason": "rebalance"},
        ],
        delistings=[
            {"symbol": "BBB", "delist_date": "2021-06-01", "reason": "acquired"},
        ],
    )


def test_index_membership_pit_reconstruction():
    u = _membership_universe()
    assert isinstance(u, cp.UniverseProvider)
    assert u.survivorship_biased is False

    # 2020: BBB ещё в индексе и торгуем (хотя позже будет делистнут) — survivorship-free
    c2020 = u.constituents(date_to_ms("2020-06-01"))
    assert set(c2020) == {"AAA", "BBB"}
    assert u.is_tradable("BBB", date_to_ms("2020-06-01")) is True

    # 2022: BBB заменён на CCC и делистнут → не в составе и не торгуем
    c2022 = u.constituents(date_to_ms("2022-01-01"))
    assert set(c2022) == {"AAA", "CCC"}
    assert u.is_tradable("BBB", date_to_ms("2022-01-01")) is False


def test_index_membership_no_baseline_returns_empty():
    u = IndexMembershipUniverse("EMPTY")  # без baseline
    assert u.constituents(date_to_ms("2020-01-01")) == []


# ---------------------------------------------------------------------------
# ADVLiquidityFilter
# ---------------------------------------------------------------------------
def _liquidity_panel():
    t0 = 1_700_000_000
    step = 86_400  # дневные бары
    ts = [t0 + i * step for i in range(5)]
    liq = pd.DataFrame(
        {"timestamp": ts, "symbol": "LIQ", "close": [100.0] * 5, "volume": [10_000.0] * 5}
    )   # dollar vol = 1e6
    ilq = pd.DataFrame(
        {"timestamp": ts, "symbol": "ILQ", "close": [10.0] * 5, "volume": [100.0] * 5}
    )   # dollar vol = 1e3
    panel = PanelBuilder.from_frames({"LIQ": liq, "ILQ": ilq})
    return panel, ts


def test_adv_filter_cuts_illiquid_and_keeps_flag():
    panel, ts = _liquidity_panel()
    base = StaticUniverse(["LIQ", "ILQ"], name="base")
    filt = ADVLiquidityFilter(
        base, panel, min_adv=1e5, lookback=3, dollar_volume=True
    )
    asof = ts[-1] * 1000  # последний бар, в мс
    assert filt.constituents(asof) == ["LIQ"]            # ILQ (1e3) < 1e5 → отсечён
    assert filt.is_tradable("LIQ", asof) is True
    assert filt.is_tradable("ILQ", asof) is False
    # honest-флаг базы сохранён
    assert filt.survivorship_biased is True
    # ADV-значения доступны для диагностики
    adv = filt.adv(asof)
    assert adv["LIQ"] == pytest.approx(1e6)
    assert adv["ILQ"] == pytest.approx(1e3)


def test_adv_filter_requires_volume_column():
    panel = PanelBuilder.from_frames(
        {"AAA": pd.DataFrame({"timestamp": [1_700_000_000], "symbol": "AAA", "close": [1.0]})}
    )
    with pytest.raises(ValueError):
        ADVLiquidityFilter(StaticUniverse(["AAA"]), panel, min_adv=1.0)
