# -*- coding: utf-8 -*-
"""Tests for the instrument master & symbology service (services.instrument_master).

Uses documented, real check-digit examples:
  * ISIN  US0378331005 (Apple) — ISO 6166 worked example.
  * CUSIP 037833100 (Apple).
  * FIGI  BBG000BLNQ16 (IBM) — OMG FIGI worked example.
"""

from __future__ import annotations

from datetime import date

import pytest

from services.instrument_master import (
    InstrumentMaster,
    InstrumentRecord,
    is_valid_isin,
    is_valid_cusip,
    is_valid_sedol,
    is_valid_figi,
    isin_check_digit,
    cusip_check_digit,
    build_occ_symbol,
    parse_occ_symbol,
    get_default_master,
)


# --------------------------------------------------------------------------- ISIN
def test_isin_valid_known():
    assert is_valid_isin("US0378331005")  # Apple
    assert is_valid_isin("US5949181045")  # Microsoft
    assert is_valid_isin("US30231G1022")  # Exxon


def test_isin_check_digit_value():
    assert isin_check_digit("US037833100") == 5


def test_isin_invalid_checkdigit():
    assert not is_valid_isin("US0378331004")  # corrupted check digit
    assert not is_valid_isin("US037833100")  # too short
    assert not is_valid_isin("XX")  # garbage


# --------------------------------------------------------------------------- CUSIP
def test_cusip_valid_known():
    assert is_valid_cusip("037833100")  # Apple
    assert cusip_check_digit("03783310") == 0


def test_cusip_invalid():
    assert not is_valid_cusip("037833101")
    assert not is_valid_cusip("ABC")


# --------------------------------------------------------------------------- FIGI
def test_figi_valid_known():
    assert is_valid_figi("BBG000BLNQ16")  # IBM (OMG worked example)


def test_figi_invalid():
    assert not is_valid_figi("BBG000BLNQ17")  # corrupted check digit
    assert not is_valid_figi("BSG000BLNQ16")  # disallowed BS prefix
    assert not is_valid_figi("XX0000000000")  # no G in position 3


# --------------------------------------------------------------------------- SEDOL
def test_sedol_roundtrip_consistency():
    # build a valid SEDOL by appending the algorithm's own check digit
    from services.instrument_master import is_valid_sedol

    body = "B0YBKJ"
    weights = [1, 3, 1, 7, 3, 9]
    total = sum(
        (int(c) if c.isdigit() else ord(c) - ord("A") + 10) * w for c, w in zip(body, weights)
    )
    chk = (10 - (total % 10)) % 10
    assert is_valid_sedol(body + str(chk))
    assert not is_valid_sedol(body + str((chk + 1) % 10))


# --------------------------------------------------------------------------- OCC
def test_occ_build_and_parse_roundtrip():
    sym = build_occ_symbol("AAPL", date(2026, 1, 16), "C", 150.0)
    assert len(sym) == 21
    assert sym == "AAPL  260116C00150000"
    o = parse_occ_symbol(sym)
    assert o.root == "AAPL" and o.option_type == "C"
    assert o.strike == 150.0 and o.expiry == date(2026, 1, 16)


def test_occ_fractional_strike():
    sym = build_occ_symbol("SPY", date(2025, 12, 19), "P", 612.5)
    assert parse_occ_symbol(sym).strike == 612.5


def test_occ_bad_length():
    with pytest.raises(ValueError):
        parse_occ_symbol("TOO SHORT")


# --------------------------------------------------------------------------- master
def test_resolve_by_ticker_and_alias():
    m = InstrumentMaster(seed=True)
    rec = m.resolve("AAPL")
    assert rec is not None and rec.figi == "BBG000B9XRY4"
    # alias resolves to same instrument
    assert m.resolve("BTC/USDT") is m.resolve("BTCUSDT")


def test_resolve_by_isin_and_cusip():
    m = InstrumentMaster(seed=True)
    assert m.resolve("US0378331005").ticker == "AAPL"
    assert m.resolve("037833100").ticker == "AAPL"


def test_figi_for():
    m = InstrumentMaster(seed=True)
    assert m.figi_for("MSFT") == "BBG000BPH459"
    assert m.figi_for("UNKNOWN_X") is None


def test_register_option():
    m = InstrumentMaster(seed=True)
    rec = m.register_option("AAPL", date(2026, 1, 16), "C", 150.0)
    assert rec.asset_class == "option" and rec.underlying == "AAPL"
    assert m.resolve(rec.occ_symbol) is rec


def test_search():
    m = InstrumentMaster(seed=True)
    res = m.search("apple")
    assert any(r.ticker == "AAPL" for r in res)


def test_persistence_roundtrip(tmp_path):
    m = InstrumentMaster(seed=True)
    p = tmp_path / "instruments.json"
    m.to_json(p)
    m2 = InstrumentMaster(seed=False, mapping_files=[p])
    assert len(m2) == len(m)
    assert m2.resolve("AAPL").figi == "BBG000B9XRY4"


def test_default_master_singleton():
    assert get_default_master() is get_default_master()
    assert get_default_master().resolve("SPY").asset_class == "etf"


def test_unknown_returns_none():
    m = InstrumentMaster(seed=True)
    assert m.resolve("NONEXISTENT") is None
    assert m.resolve("") is None
