# -*- coding: utf-8 -*-
"""Тесты SEC EDGAR PIT-фундаментала (P0: equity point-in-time без покупки данных).

Сеть НЕ используется — companyfacts подменяются фейком через DI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core_portfolio import SYMBOL_LEVEL, TS_LEVEL
from core_xs_data import PIT_TRUE
from services.edgar_fundamentals import (
    EdgarFundamentals, build_pit_fundamentals_frame, build_symbol_fundamentals,
)


def _facts():
    def node(unit, series):
        return {"units": {unit: series}}
    return {"facts": {
        "us-gaap": {
            "EarningsPerShareDiluted": node("USD/shares", [
                {"filed": "2023-02-01", "end": "2022-12-31", "accn": "A1", "form": "10-K", "fp": "FY", "val": 6.0},
                {"filed": "2024-02-01", "end": "2023-12-31", "accn": "A2", "form": "10-K", "fp": "FY", "val": 6.5},
            ]),
            "StockholdersEquity": node("USD", [
                {"filed": "2023-02-01", "end": "2022-12-31", "accn": "A1", "val": 100.0},
                {"filed": "2024-02-01", "end": "2023-12-31", "accn": "A2", "val": 120.0},
            ]),
            "NetIncomeLoss": node("USD", [
                {"filed": "2023-02-01", "end": "2022-12-31", "accn": "A1", "val": 10.0},
                {"filed": "2024-02-01", "end": "2023-12-31", "accn": "A2", "val": 12.0},
            ]),
            "NetCashProvidedByUsedInOperatingActivities": node("USD", [
                {"filed": "2023-02-01", "end": "2022-12-31", "accn": "A1", "val": 20.0},
                {"filed": "2024-02-01", "end": "2023-12-31", "accn": "A2", "val": 24.0},
            ]),
            "PaymentsToAcquirePropertyPlantAndEquipment": node("USD", [
                {"filed": "2023-02-01", "end": "2022-12-31", "accn": "A1", "val": 5.0},
                {"filed": "2024-02-01", "end": "2023-12-31", "accn": "A2", "val": 6.0},
            ]),
        },
        "dei": {
            "EntityCommonStockSharesOutstanding": node("shares", [
                {"filed": "2023-02-01", "end": "2022-12-31", "accn": "A1", "val": 10.0},
                {"filed": "2024-02-01", "end": "2023-12-31", "accn": "A2", "val": 10.0},
            ]),
        },
    }}


def _ms(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def test_symbol_fundamentals_derivations():
    df = build_symbol_fundamentals("AAA", _facts())
    assert list(df["symbol"].unique()) == ["AAA"]
    assert len(df) == 2
    a2 = df[df["publish_ts"] == _ms("2024-02-01")].iloc[0]
    assert a2["earnings"] == pytest.approx(6.5)          # EPS diluted
    assert a2["book_value"] == pytest.approx(120.0 / 10)  # BVPS = equity/shares
    assert a2["roe"] == pytest.approx(12.0 / 120.0)       # NI/equity
    assert a2["fcf"] == pytest.approx((24.0 - 6.0) / 10)  # (CFO-CapEx)/shares


def test_publish_ts_is_filing_date_not_period_end():
    df = build_symbol_fundamentals("AAA", _facts())
    # publish_ts должен быть датой ПОДАЧИ (filed), а не концом периода (end)
    assert _ms("2024-02-01") in set(df["publish_ts"])
    assert _ms("2023-12-31") not in set(df["publish_ts"])  # end != publish


def test_build_frame_via_DI():
    frame = build_pit_fundamentals_frame(
        ["AAA"], tickers_fn=lambda: {"AAA": 111}, facts_fn=lambda cik: _facts())
    assert set(["publish_ts", "symbol", "earnings", "book_value", "fcf", "roe"]).issubset(frame.columns)
    assert frame["symbol"].iloc[0] == "AAA"


def test_source_is_pit_true():
    src = EdgarFundamentals(tickers_fn=lambda: {"AAA": 111}, facts_fn=lambda cik: _facts())
    assert src.meta.pit_quality == PIT_TRUE
    out = src.get_fundamentals(["AAA"], ["earnings", "book_value"])
    assert list(out.columns) == ["publish_ts", "symbol", "earnings", "book_value"]


def test_asof_join_no_lookahead():
    """PIT-гарантия: до даты подачи значение недоступно (NaN); после — последнее поданное."""
    from loaders.equity_enrich import make_pit_fundamentals_enricher
    src = EdgarFundamentals(tickers_fn=lambda: {"AAA": 111}, facts_fn=lambda cik: _facts())
    enr = make_pit_fundamentals_enricher(source=src, fields=["earnings"])

    ts = [_ms("2023-01-01"), _ms("2023-06-01"), _ms("2024-06-01")]
    idx = pd.MultiIndex.from_arrays([ts, ["AAA"] * 3], names=[TS_LEVEL, SYMBOL_LEVEL])
    panel = pd.DataFrame({"close": [100.0, 110.0, 120.0]}, index=idx)

    out = enr.enrich(panel)
    earn = out["earnings"].to_numpy()
    assert np.isnan(earn[0])                 # до 2023-02-01 — фундаментал ещё не подан
    assert earn[1] == pytest.approx(6.0)     # после A1 (2023-02-01), до A2
    assert earn[2] == pytest.approx(6.5)     # после A2 (2024-02-01)
