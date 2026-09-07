# -*- coding: utf-8 -*-
"""Dukascopy forex adapter — public bi5 tick feed (closes the stub gap).

The UI offered "Dukascopy (Public ticks)" for forex with no backend (43-line
Phase-0 stub). These tests lock in the real bi5 decode + bar aggregation +
registry wiring, with the HTTP download mocked (no network in CI).
"""

from __future__ import annotations

import lzma
import struct
from datetime import datetime, timezone

import pytest

from adapters.base import MarketDataAdapter
from adapters.models import ExchangeVendor
from adapters.registry import create_market_data_adapter
from adapters.dukascopy.market_data import DukascopyMarketDataAdapter
from core_models import Bar, Tick


def _hour_ms(y, mo, d, h):
    return int(datetime(y, mo, d, h, tzinfo=timezone.utc).timestamp() * 1000)


def _bi5(*records):
    """Build a synthetic LZMA-compressed bi5 payload.
    Each record: (ms_offset, ask_points, bid_points, ask_vol, bid_vol)."""
    raw = b"".join(struct.pack(">IIIff", *r) for r in records)
    return lzma.compress(raw, format=lzma.FORMAT_ALONE)


# ------------------------------------------------------------------ registry


def test_registered_as_market_data():
    a = create_market_data_adapter(ExchangeVendor.DUKASCOPY)
    assert isinstance(a, DukascopyMarketDataAdapter)
    assert isinstance(a, MarketDataAdapter)


def test_registered_by_string_vendor():
    a = create_market_data_adapter("dukascopy")
    assert isinstance(a, DukascopyMarketDataAdapter)


# ------------------------------------------------------------------ url / scaling


def test_bi5_url_month_is_zero_indexed():
    a = DukascopyMarketDataAdapter()
    # January must render as /00/ (famous Dukascopy gotcha)
    url = a._bi5_url("EURUSD", datetime(2024, 1, 15, 9, tzinfo=timezone.utc))
    assert url.endswith("/EURUSD/2024/00/15/09h_ticks.bi5")
    # December → /11/
    url2 = a._bi5_url("EURUSD", datetime(2024, 12, 1, 0, tzinfo=timezone.utc))
    assert "/2024/11/01/00h_ticks.bi5" in url2


def test_point_values():
    a = DukascopyMarketDataAdapter()
    assert a._point_value("EURUSD") == 100000.0
    assert a._point_value("USDJPY") == 1000.0  # JPY pairs: 3 decimals
    assert a._point_value("XAUUSD") == 1000.0  # metals
    a2 = DukascopyMarketDataAdapter(config={"point_values": {"EURUSD": 12345.0}})
    assert a2._point_value("EURUSD") == 12345.0  # override


def test_normalize_symbol():
    a = DukascopyMarketDataAdapter()
    assert a._normalize_symbol("EUR/USD") == "EURUSD"
    assert a._normalize_symbol("eur_usd") == "EURUSD"
    assert a._normalize_symbol("EUR-USD") == "EURUSD"


# ------------------------------------------------------------------ decode


def test_parse_ticks_scales_and_offsets():
    a = DukascopyMarketDataAdapter()
    h = _hour_ms(2024, 1, 15, 9)
    raw = _bi5(
        (1000, 108500, 108490, 1.0, 1.2),
        (61000, 108520, 108505, 0.5, 0.7),
    )
    ticks = a._parse_ticks(raw, h, 100000.0)
    assert len(ticks) == 2
    assert ticks[0][0] == h + 1000  # ms offset applied
    assert abs(ticks[0][1] - 1.08490) < 1e-9  # bid
    assert abs(ticks[0][2] - 1.08500) < 1e-9  # ask


def test_decompress_tolerant_of_garbage():
    a = DukascopyMarketDataAdapter()
    assert a._decompress(b"not-lzma") == b""  # never raises


# ------------------------------------------------------------------ get_bars


def _mock_hour(a, mapping):
    """mapping: {(day, hour): bi5-bytes}; missing hours → None (404/weekend)."""

    def fake(instrument, dt_hour):
        return mapping.get((dt_hour.day, dt_hour.hour))

    a._download_bi5 = fake


def test_get_bars_aggregates_ohlc_and_bidask():
    a = DukascopyMarketDataAdapter()
    h = _hour_ms(2024, 1, 15, 9)
    # three ticks within minute 0 → one 1m bar with real OHLC
    raw = _bi5(
        (5_000, 108500, 108490, 1, 1),  # bid 1.08490 ask 1.08500
        (20_000, 108600, 108590, 1, 1),  # higher
        (50_000, 108400, 108390, 1, 1),  # lower, last
    )
    _mock_hour(a, {(15, 9): raw})
    bars = a.get_bars("EURUSD", "1m", start_ts=h, end_ts=h + 60_000)
    assert len(bars) == 1
    b = bars[0]
    assert isinstance(b, Bar)
    # mid OHLC: open ~1.08495, high ~1.08595, low ~1.08395, close ~1.08395
    assert float(b.open) == pytest.approx(1.08495, abs=1e-6)
    assert float(b.high) == pytest.approx(1.08595, abs=1e-6)
    assert float(b.low) == pytest.approx(1.08395, abs=1e-6)
    assert float(b.close) == pytest.approx(1.08395, abs=1e-6)
    assert b.volume_base == 3  # tick count
    # bid/ask channels preserved
    assert float(b.bid_close) == pytest.approx(1.08390, abs=1e-6)
    assert float(b.ask_close) == pytest.approx(1.08400, abs=1e-6)


def test_get_bars_multiple_minutes_and_limit():
    a = DukascopyMarketDataAdapter()
    h = _hour_ms(2024, 1, 15, 9)
    raw = _bi5(
        (1_000, 108500, 108490, 1, 1),  # minute 0
        (61_000, 108520, 108505, 1, 1),  # minute 1
        (121_000, 108480, 108470, 1, 1),  # minute 2
    )
    _mock_hour(a, {(15, 9): raw})
    bars = a.get_bars("EURUSD", "1m", start_ts=h, end_ts=h + 180_000)
    assert len(bars) == 3
    limited = a.get_bars("EURUSD", "1m", start_ts=h, end_ts=h + 180_000, limit=2)
    assert len(limited) == 2  # last 2


def test_get_bars_weekend_no_data_graceful():
    a = DukascopyMarketDataAdapter()
    _mock_hour(a, {})  # every hour → None
    h = _hour_ms(2024, 1, 14, 3)  # Sunday
    assert a.get_bars("EURUSD", "1h", start_ts=h, end_ts=h + 7200_000) == []


def test_get_bars_bad_timeframe_raises():
    a = DukascopyMarketDataAdapter()
    with pytest.raises(ValueError):
        a.get_bars("EURUSD", "3s", start_ts=0, end_ts=1)


def test_get_tick_returns_latest():
    a = DukascopyMarketDataAdapter()
    now_hour = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    raw = _bi5((1_000, 108500, 108490, 1, 1), (2_000, 108510, 108500, 1, 1))

    def fake(instrument, dt_hour):
        return raw if dt_hour.hour == now_hour.hour and dt_hour.day == now_hour.day else None

    a._download_bi5 = fake
    tick = a.get_tick("EURUSD")
    assert isinstance(tick, Tick)
    assert float(tick.ask) == pytest.approx(1.08510, abs=1e-6)  # last tick
    assert float(tick.bid) == pytest.approx(1.08500, abs=1e-6)
    assert tick.symbol == "EURUSD"


def test_premium_matrix_includes_dukascopy():
    from services.premium_data import vendor_status

    by = {v["vendor"]: v for v in vendor_status()}
    assert "dukascopy" in by
    assert by["dukascopy"]["ready"] is True  # keyless public feed
    assert by["dukascopy"]["ticks"] == "history"
    assert "forex" in by["dukascopy"]["asset_classes"]
