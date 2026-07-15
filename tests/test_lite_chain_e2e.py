"""End-to-end execution of the Lite Data Manager chain (audit L2-003/L2-017).

Unlike allow-list checks, this test EXECUTES the exact worker commands the
backend builds for the Lite UI parameters — as real subprocesses, in an
isolated data directory (separate from the code root, which also exercises
audit L2-016) — and verifies exit code 0 plus a non-empty artifact for every
step:

    features → targets → no-trade mask → walk-forward splits → training table
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("SEASONALITY_API_TOKEN", "test-token-lite-e2e")

from fastapi.testclient import TestClient

import app as app_module
from app import api
from desktop_job_runtime import prepare_python_command, worker_environment

ROOT = Path(__file__).resolve().parents[1]

client = TestClient(api, headers={"X-API-Key": app_module.API_TOKEN})

# The exact parameter payloads the Lite UI sends (see index.html
# triggerLiteFeatures/Targets/NoTrade/Splits/FinalBuild), with the short
# lookbacks preset so ~200 hourly bars are enough.
LITE_CHAIN = [
    ("run_features", {
        "in": "data/prices.parquet", "out": "data/features.parquet",
        "lookbacks": "60,120", "rsi_period": 14, "price_col": "close",
        "yang_zhang_windows": "120", "cvd_windows": "120",
        "parkinson_windows": "120", "garch_windows": "240",
        "taker_buy_ratio_windows": "120", "taker_buy_ratio_momentum": "60",
        "bar_duration_minutes": 60,
    }, "data/features.parquet"),
    ("run_targets", {
        "in": "data/features.parquet", "out": "data/targets.parquet",
        "fees_bps_total": 10, "threshold": 0.0003, "horizon_bars": 5,
    }, "data/targets.parquet"),
    ("run_no_trade", {
        "data": "data/targets.parquet", "out": "data/targets_masked.parquet",
        "timeframe": "1h", "config": "configs/sandbox.yaml",
    }, "data/targets_masked.parquet"),
    ("run_splits", {
        "data": "data/targets.parquet", "n_splits": 3, "train_size_pct": 80,
        "config": "configs/sandbox.yaml",
    }, "data/targets_wf.parquet"),
    ("run_training_table", {
        "base": "data/features.parquet", "prices": "data/prices.parquet",
        "out": "data/training_table.parquet", "price_col": "close",
        "decision_delay_ms": 8000, "label_horizon_ms": 7200000,
    }, "data/training_table.parquet"),
]


def _build_commands(monkeypatch):
    captured = []

    def fake_start_background(cmd, pid_file, log_file):
        captured.append([str(c) for c in cmd])
        return 1

    monkeypatch.setattr(app_module, "start_background", fake_start_background)
    monkeypatch.setattr(app_module, "background_running", lambda _pid: False)

    for job, params, _artifact in LITE_CHAIN:
        res = client.post("/api/run_job", json={"job": job, "params": params})
        assert res.status_code == 200, f"{job}: {res.text}"
    assert len(captured) == len(LITE_CHAIN)
    return captured


def _seed_prices(data_root: Path) -> None:
    rng = np.random.default_rng(7)
    n = 240
    ts = 1_700_000_000_000 + np.arange(n) * 3_600_000  # hourly bars
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({
        "ts_ms": ts,
        "symbol": "BTCUSDT",
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.003, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.003, n))),
        "close": close,
        "price": close,
        "volume": rng.uniform(10, 100, n),
    })
    (data_root / "data").mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_root / "data" / "prices.parquet", index=False)


@pytest.mark.slow
def test_lite_data_chain_runs_end_to_end(monkeypatch, tmp_path):
    commands = _build_commands(monkeypatch)

    data_root = tmp_path / "lite-data-root"
    data_root.mkdir()
    _seed_prices(data_root)
    # Workers read configs relative to CWD. The desktop runtime seeds the whole
    # bundled configs/ directory into the data root (desktop_backend.py), so the
    # test replicates that.
    (data_root / "configs").mkdir()
    for src in (ROOT / "configs").glob("*.yaml"):
        (data_root / "configs" / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    env = worker_environment()
    # Production shape: the desktop server's CWD is the data root, so script
    # resolution against the code root must happen from there (audit L2-016).
    monkeypatch.chdir(data_root)
    for (job, _params, artifact), cmd in zip(LITE_CHAIN, commands):
        resolved = prepare_python_command(cmd)
        # Source mode with a separate data root (audit L2-016): the script path
        # must resolve against the code root even though CWD is the data dir.
        proc = subprocess.run(
            resolved, cwd=str(data_root), env=env,
            capture_output=True, text=True, timeout=600,
        )
        assert proc.returncode == 0, (
            f"{job} failed (exit {proc.returncode})\n"
            f"cmd: {resolved}\nstdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
        )
        out_path = data_root / artifact
        assert out_path.is_file(), f"{job}: artifact {artifact} was not created"
        produced = pd.read_parquet(out_path)
        assert len(produced) > 0, f"{job}: artifact {artifact} is empty"

    # LeakGuard invariant (audit L2-004): decision_ts must trail ts by >= 8000 ms.
    table = pd.read_parquet(data_root / "data" / "training_table.parquet")
    assert "decision_ts" in table.columns
    delays = table["decision_ts"] - table["ts_ms"]
    assert (delays >= 8000).all(), f"unsafe decision delays found: min={delays.min()}"

    # Walk-forward manifest is written next to the data (real evidence).
    manifest = data_root / "logs" / "walkforward" / "walkforward_manifest.json"
    assert manifest.is_file()
