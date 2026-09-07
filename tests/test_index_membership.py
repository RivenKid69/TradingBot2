# -*- coding: utf-8 -*-
"""Тесты PIT (survivorship-free) истории членства в индексе (P0: equity)."""

from __future__ import annotations

import pandas as pd
import pytest

from services.index_membership_loader import (
    build_index_membership_universe,
    changes_to_baseline_and_events,
    load_membership_changes,
)

DEMO = "data/universe/sp500_membership_demo.csv"


def _ms(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def test_load_demo_changes():
    df = load_membership_changes(DEMO)
    assert set(df.columns) == {"date", "ticker", "action"}
    assert (df["action"] == "add").all()
    assert "TSLA" in set(df["ticker"])


def test_baseline_and_events_split():
    df = load_membership_changes(DEMO)
    base_date, baseline, events = changes_to_baseline_and_events(df)
    assert base_date == "2019-01-02"
    assert "AAPL" in baseline and "TSLA" not in baseline
    # TSLA добавлен отдельным событием 2020-12-21 (реальное событие S&P 500)
    assert any(e["date"] == "2020-12-21" and "TSLA" in e["added"] for e in events)


def test_pit_constituents_add(tmp_path):
    uni = build_index_membership_universe(DEMO, index="SP500_DEMO")
    before = uni.constituents(_ms("2020-01-01"))
    after = uni.constituents(_ms("2021-01-01"))
    assert "TSLA" not in before  # до добавления — нет (PIT)
    assert "TSLA" in after  # после 2020-12-21 — есть
    assert "AAPL" in before and "AAPL" in after
    assert uni.survivorship_biased is False


def test_pit_constituents_remove(tmp_path):
    p = tmp_path / "changes.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01", "2021-06-01", "2022-01-01"],
            "ticker": ["AAA", "BBB", "CCC", "BBB"],
            "action": ["add", "add", "add", "remove"],
        }
    ).to_csv(p, index=False)
    uni = build_index_membership_universe(str(p), index="T")
    assert set(uni.constituents(_ms("2020-03-01"))) == {"AAA", "BBB"}
    assert set(uni.constituents(_ms("2021-07-01"))) == {"AAA", "BBB", "CCC"}
    after_remove = set(uni.constituents(_ms("2022-02-01")))
    assert "BBB" not in after_remove  # удалён 2022-01-01 (PIT)
    assert after_remove == {"AAA", "CCC"}


def test_build_universe_wiring():
    import yaml
    from service_xs_pipeline import XSConfig, build_universe

    cfg = XSConfig.model_validate(
        {
            "mode": "cross_sectional",
            "asset_class": "equity",
            "data": {"source": "synthetic", "symbols": ["AAPL", "MSFT"]},
            "universe": {
                "type": "index_membership",
                "index": "SP500_DEMO",
                "membership_path": DEMO,
            },
        }
    )
    uni = build_universe(cfg)
    # должен быть survivorship-free IndexMembershipUniverse, а не StaticUniverse
    assert getattr(uni, "survivorship_biased", True) is False
    assert "AAPL" in uni.constituents(_ms("2021-01-01"))
