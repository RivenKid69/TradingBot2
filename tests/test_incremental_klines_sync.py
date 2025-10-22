from __future__ import annotations

import csv
from pathlib import Path

import incremental_klines


INTERVAL = incremental_klines.INTERVAL_MS


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _make_row(open_idx: int):
    open_time = open_idx * INTERVAL
    close_time = open_time + INTERVAL
    base = [
        open_time,
        "1.0",
        "2.0",
        "0.5",
        "1.5",
        "123.4",
        close_time,
        "567.8",
        "42",
        "10.1",
        "20.2",
        "0",
    ]
    return base


def test_sync_symbol_creates_full_history(monkeypatch, tmp_path: Path):
    symbol = "BTCUSDT"
    now_ms = 10 * INTERVAL
    payload = [_make_row(i) for i in range(10)]

    def fake_get(url, *, params, retries=3, backoff=0.5):  # noqa: ARG001
        assert params["startTime"] == 0
        assert params["limit"] == 10
        return DummyResponse(payload)

    monkeypatch.setattr(incremental_klines, "_get_with_retry", fake_get)
    monkeypatch.setattr(incremental_klines.clock, "now_ms", lambda: now_ms)
    monkeypatch.setattr(incremental_klines, "OUT_DIR", str(tmp_path))

    appended = incremental_klines.sync_symbol(symbol, close_lag_ms=0)

    assert appended == 10
    out_file = tmp_path / f"{symbol}.csv"
    assert out_file.exists()
    rows = out_file.read_text().strip().splitlines()
    # header + 10 rows
    assert len(rows) == 11
    assert rows[0].startswith("open_time")
    assert rows[-1].split(",")[0] == str(9 * INTERVAL)


def test_sync_symbol_appends_missing(monkeypatch, tmp_path: Path):
    symbol = "ETHUSDT"
    now_ms = 10 * INTERVAL

    monkeypatch.setattr(incremental_klines.clock, "now_ms", lambda: now_ms)
    monkeypatch.setattr(incremental_klines, "OUT_DIR", str(tmp_path))

    out_file = tmp_path / f"{symbol}.csv"
    with out_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(incremental_klines.HEADER)
        for idx in range(3):
            row = _make_row(idx) + [symbol]
            writer.writerow(row)

    payload = [_make_row(i) for i in range(3, 10)]

    def fake_get(url, *, params, retries=3, backoff=0.5):  # noqa: ARG001
        assert params["startTime"] == 3 * INTERVAL
        assert params["limit"] == 7
        return DummyResponse(payload)

    monkeypatch.setattr(incremental_klines, "_get_with_retry", fake_get)

    appended = incremental_klines.sync_symbol(symbol, close_lag_ms=0)

    assert appended == 7
    rows = out_file.read_text().strip().splitlines()
    # header + 10 rows total
    assert len(rows) == 11
    assert rows[-1].split(",")[0] == str(9 * INTERVAL)


def test_sync_symbol_returns_zero_when_up_to_date(monkeypatch, tmp_path: Path):
    symbol = "XRPUSDT"
    now_ms = 10 * INTERVAL

    monkeypatch.setattr(incremental_klines.clock, "now_ms", lambda: now_ms)
    monkeypatch.setattr(incremental_klines, "OUT_DIR", str(tmp_path))

    out_file = tmp_path / f"{symbol}.csv"
    with out_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(incremental_klines.HEADER)
        for idx in range(10):
            row = _make_row(idx) + [symbol]
            writer.writerow(row)

    def fake_get(url, *, params, retries=3, backoff=0.5):  # noqa: ARG001
        assert params["startTime"] == 0
        assert params["limit"] == 1
        return DummyResponse([_make_row(0)])

    monkeypatch.setattr(incremental_klines, "_get_with_retry", fake_get)

    appended = incremental_klines.sync_symbol(symbol, close_lag_ms=0)

    assert appended == 0
    rows = out_file.read_text().strip().splitlines()
    assert len(rows) == 11


def test_sync_symbol_rebuilds_truncated_csv(monkeypatch, tmp_path: Path):
    symbol = "LTCUSDT"
    now_ms = 10 * INTERVAL

    monkeypatch.setattr(incremental_klines.clock, "now_ms", lambda: now_ms)
    monkeypatch.setattr(incremental_klines, "OUT_DIR", str(tmp_path))

    out_file = tmp_path / f"{symbol}.csv"
    with out_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(incremental_klines.HEADER)
        for idx in range(7, 10):
            row = _make_row(idx) + [symbol]
            writer.writerow(row)

    payload = [_make_row(i) for i in range(10)]

    def fake_get(url, *, params, retries=3, backoff=0.5):  # noqa: ARG001
        start = params["startTime"]
        limit = params["limit"]
        if start == 0 and limit == 1:
            return DummyResponse([payload[0]])
        assert start == 0
        assert limit == 10
        return DummyResponse(payload)

    monkeypatch.setattr(incremental_klines, "_get_with_retry", fake_get)

    appended = incremental_klines.sync_symbol(symbol, close_lag_ms=0)

    assert appended == 10
    rows = out_file.read_text().strip().splitlines()
    assert len(rows) == 11
    assert rows[1].split(",")[0] == str(0)
    assert rows[-1].split(",")[0] == str(9 * INTERVAL)

