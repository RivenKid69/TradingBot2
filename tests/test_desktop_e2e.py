"""Desktop sidecar end-to-end lifecycle test.

Exercises the same process boundary used by Tauri:
launch -> CCEA ready -> paper fill -> graceful shutdown -> restart.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_sidecar(data_dir: Path, port: int) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.update(
        {
            "RIVEN_DATA_DIR": str(data_dir),
            "RIVEN_ENABLE_CCEA": "1",
            "RIVEN_ENABLE_TERMINAL": "0",
            "RIVEN_API_AUTH_MODE": "loopback",
            "SEASONALITY_API_TOKEN": "desktop-e2e-token",
        }
    )
    frozen_backend = os.environ.get("RIVEN_DESKTOP_BACKEND_EXE")
    command = (
        [frozen_backend, "--port", str(port)]
        if frozen_backend
        else [sys.executable, "desktop_backend.py", "--port", str(port)]
    )
    return subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )


def _wait_json(url: str, predicate, timeout: float = 90.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=3.0)
            if response.is_success:
                last = response.json()
                if predicate(last):
                    return last
        except Exception:
            pass
        time.sleep(0.25)
    raise AssertionError(f"Timed out waiting for {url}; last={last!r}")


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _assert_sqlite_files_unlocked(data_dir: Path) -> None:
    for database in data_dir.rglob("*.db"):
        probe = database.with_suffix(database.suffix + ".lock-probe")
        deadline = time.monotonic() + 10
        while True:
            try:
                database.replace(probe)
                probe.replace(database)
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.25)


def test_desktop_launch_paper_shutdown_restart(tmp_path: Path) -> None:
    data_dir = tmp_path / "desktop-data"
    data_dir.mkdir()

    for launch in range(2):
        port = _free_port()
        base = f"http://127.0.0.1:{port}"
        process = _start_sidecar(data_dir, port)
        try:
            status = _wait_json(
                f"{base}/api/ccea/status",
                lambda value: value.get("state") == "error" or (
                    value.get("state") == "running"
                    and value.get("agent", {}).get("cloud_connected") is True
                    and value.get("agent", {}).get("vault_unlocked") is True
                    and value.get("agent", {}).get("broker_connected") is True
                ),
            )
            assert status["state"] == "running", status
            assert status["enrolled"] is True
            assert status["agent"]["cloud_connected"] is True
            assert status["agent"]["vault_unlocked"] is True
            assert status["agent"]["broker_connected"] is True

            if launch == 1:
                ledger = status["pnl_ledger"]
                btc = next(p for p in ledger["positions"] if p["symbol"] == "BTCUSDT")
                assert btc["quantity"] == 0.1
                assert btc["mark"] == 55_000
                assert ledger["equity"] == 100_500
                assert ledger["unrealized_pnl"] == 500
                assert ledger["day_pnl"] == 0

            page = httpx.get(f"{base}/", timeout=10)
            assert page.status_code == 200
            assert "/assets/tailwindcss.js" in page.text
            assert "cdn.tailwindcss.com" not in page.text
            assert "cdn.jsdelivr.net" not in page.text
            assert "fonts.googleapis.com" not in page.text
            for asset in (
                "tailwindcss.js",
                "chart.umd.min.js",
                "fonts/fonts.css",
                "fontawesome/css/all.min.css",
                "monaco/vs/loader.js",
            ):
                assert httpx.get(f"{base}/assets/{asset}", timeout=10).status_code == 200

            # The desktop data boundary is the writable runtime root, not the
            # repository/PyInstaller bundle where this test itself lives.
            missing_runtime_data = httpx.get(
                f"{base}/api/data/preview",
                params={"path": "data/prices.parquet", "n": 1},
                timeout=10,
            )
            assert missing_runtime_data.status_code == 404
            missing_heal = httpx.post(
                f"{base}/api/data/auto_heal",
                json={"path": "data/prices.parquet", "forward_fill_limit": 5},
                timeout=10,
            )
            assert missing_heal.status_code == 404

            if launch == 0:
                stored = httpx.post(
                    f"{base}/api/ccea/store_credentials",
                    json={"broker": "polygon", "credentials": {"api_key": "e2e-secret"}},
                    timeout=10,
                ).json()
                assert stored["ok"] is True
                assert stored["credentials_in_vault"] is True
                assert "e2e-secret" not in str(stored)

            fake_close = httpx.post(
                f"{base}/api/portfolio/close", json={"symbol": "SPY"}, timeout=10
            )
            assert fake_close.status_code == 404

            trade = httpx.post(
                f"{base}/api/ccea/paper_order",
                json={
                    "symbol": "BTCUSDT",
                    "qty": 0.1,
                    "entry_price": 50_000,
                    "mark_price": 55_000,
                },
                timeout=30,
            ).json()
            assert trade["ok"] is True
            assert trade["order"]["status"] == "filled"
            assert trade["ledger_broker_reconciled"] is True
            assert trade["integrity_ok"] is True

            if launch == 0:
                eod = httpx.post(f"{base}/api/agent/pnl/eod_close", timeout=10).json()
                assert eod["ok"] is True
                assert eod["snapshot"]["nav"] == 100_500
                assert eod["snapshot"]["day_pnl"] == 500
            else:
                synced = httpx.post(f"{base}/api/trades/sync", timeout=10).json()
                assert synced["status"] == "success"
                assert synced["source"] == "agent_books"
                assert synced["synchronized"] >= 2

                closed = httpx.post(
                    f"{base}/api/portfolio/close", json={"symbol": "BTCUSDT"}, timeout=30
                ).json()
                assert closed["status"] == "success"
                assert closed["remaining_qty"] == 0
                portfolio = httpx.get(f"{base}/api/portfolio/holdings", timeout=10).json()
                assert not any(h["symbol"] == "BTCUSDT" for h in portfolio["holdings"])

                # Lite Emergency Halt operates on the same Agent broker: it
                # pauses the daemon, cancels real working orders, and closes an
                # actual paper position without inventing demo liquidations.
                emergency_trade = httpx.post(
                    f"{base}/api/ccea/paper_order",
                    json={"symbol": "ETHUSDT", "qty": 1, "entry_price": 3_000, "mark_price": 3_050},
                    timeout=30,
                ).json()
                assert emergency_trade["ok"] is True
                halted = httpx.post(f"{base}/api/panic_halt", timeout=30).json()
                assert halted["ok"] is True
                assert halted["agent_paused"] is True
                assert halted["positions_seen"] == 1
                assert halted["positions_remaining"] == 0
                assert halted["mode"] == "paper"
                halted_portfolio = httpx.get(f"{base}/api/portfolio/holdings", timeout=10).json()
                assert halted_portfolio["kill_switch_tripped"] is True
                assert halted_portfolio["holdings"] == []

                reset = httpx.post(f"{base}/api/panic_reset", timeout=10).json()
                assert reset["kill_switch_tripped"] is False

            shutdown = httpx.post(f"{base}/api/desktop/shutdown", timeout=30).json()
            assert shutdown == {"ok": True, "state": "stopped"}
            process.wait(timeout=30)
        finally:
            _stop_process(process)

        _assert_sqlite_files_unlocked(data_dir)
