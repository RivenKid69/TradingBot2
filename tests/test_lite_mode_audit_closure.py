"""Regression coverage for the 2026-07-11 Lite Mode audit closures."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from desktop_job_runtime import WORKER_MODULES, prepare_python_command
from packages.agent.broker.adapters._delegating import DelegatingConnector
from packages.agent.broker.protocol import BrokerCredentials, OrderRequest, OrderSide, OrderType
from services.lite_data_repair import repair_prices_file
from services.utils_app import background_running, background_status, start_background


ROOT = Path(__file__).resolve().parents[1]


class _TestConnector(DelegatingConnector):
    _NAME = "test"

    def _build_backend(self):  # pragma: no cover - backend is injected
        raise AssertionError("unexpected backend build")


class _RejectedBackend:
    def connect(self) -> bool:
        return False

    def place(self, **_kwargs):  # pragma: no cover - must never be called
        raise AssertionError("order submission must remain blocked")


def _wait_for_job(pid_file: Path, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = background_status(str(pid_file))
        if status.get("state") not in {"running", "idle"}:
            return status
        time.sleep(0.05)
    raise AssertionError(f"job did not finish: {background_status(str(pid_file))}")


def test_frozen_worker_dispatches_all_lite_job_modules(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    expected = {
        "ingest_orchestrator",
        "make_features",
        "make_costaware_targets",
        "build_training_table",
        "apply_no_trade_mask",
        "make_walkforward_splits",
        "train_model_multi_patch",
        "train_calibrator",
        "apply_calibrator",
        "tune_threshold",
        "drift",
        "script_calibrate_tcost",
        "script_calibrate_slippage",
        "tools.check_feature_parity",
        "scripts.download_stock_data",
        "scripts.download_forex_data",
        "scripts.download_options_data",
        "script_live",
        "script_futures_live",
    }
    assert expected <= WORKER_MODULES
    for module in expected:
        script = module.replace(".", "/") + ".py"
        translated = prepare_python_command([sys.executable, script, "--help"])
        assert translated[0] == sys.executable
        assert translated[1] == "--riven-worker-script"
        assert translated[2] == script


def test_frozen_worker_rejects_unbundled_script(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    with pytest.raises(ValueError, match="not packaged/allowed"):
        prepare_python_command([sys.executable, "unknown_user_script.py"])


@pytest.mark.parametrize("exit_code,expected", [(0, "succeeded"), (7, "failed")])
def test_background_job_persists_real_exit_status(tmp_path, exit_code, expected):
    pid_file = tmp_path / f"job-{exit_code}.pid"
    log_file = tmp_path / f"job-{exit_code}.log"
    start_background(
        [sys.executable, "-c", f"import sys; print('worker'); sys.exit({exit_code})"],
        pid_file=str(pid_file),
        log_file=str(log_file),
    )
    status = _wait_for_job(pid_file)
    assert status["state"] == expected
    assert status["exit_code"] == exit_code
    assert status["running"] is False
    assert "worker" in log_file.read_text(encoding="utf-8")


def test_windows_job_status_keeps_fresh_pid_during_tasklist_startup(monkeypatch, tmp_path):
    pid_file = tmp_path / "fresh.pid"
    pid_file.write_text("424242", encoding="utf-8")
    monkeypatch.setattr("services.utils_app.platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "services.utils_app.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="INFO: No tasks are running"),
    )

    assert background_running(str(pid_file)) is True
    assert pid_file.exists()

    old = time.time() - 5.0
    os.utime(pid_file, (old, old))
    assert background_running(str(pid_file)) is False
    assert not pid_file.exists()


def test_rejected_broker_connection_cannot_submit_order():
    connector = _TestConnector(
        BrokerCredentials(api_key="x", api_secret="y"), backend=_RejectedBackend()
    )
    assert connector.connect() is False
    assert connector.is_connected is False
    result = connector.submit_order(
        OrderRequest(
            client_order_id="must-not-submit",
            symbol="ES",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1,
        )
    )
    assert result.success is False
    assert result.error_message


def test_auto_heal_repairs_atomically_and_reports_remaining_gaps(tmp_path):
    path = tmp_path / "prices.parquet"
    pd.DataFrame(
        {
            "symbol": ["BTC", "BTC", "BTC", "BTC", "BTC"],
            "ts": [1, 2, 2, 3, 4],
            "close": [100.0, np.inf, 101.0, np.nan, np.nan],
        }
    ).to_parquet(path, index=False)
    result = repair_prices_file(path, forward_fill_limit=1)
    repaired = pd.read_parquet(path)
    assert result["duplicates_removed"] == 1
    assert result["infinite_values_replaced"] == 1
    assert result["cells_filled"] == 1
    assert result["missing_after"] == 1
    assert result["complete"] is False
    assert len(repaired) == 4
    assert path.with_suffix(".parquet.preheal.bak").is_file()


def test_lite_ui_audit_handlers_are_unique_and_truthful():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    for name in (
        "triggerLiteEmergencyHalt",
        "resetLiteEmergencyHalt",
        "triggerProEmergencyHalt",
        "resetProEmergencyHalt",
        "startQuantLabBacktest",
        "stopQuantLabBacktest",
        "closeLiteHistoryAuditor",
    ):
        assert len(re.findall(rf"function\s+{name}\s*\(", html)) == 1, name

    assert "function jobSucceeded(status)" in html
    assert "status.state === 'succeeded'" in html
    assert "window.location.protocol" in html
    assert "WalletConnect SDK не установлен" in html
    assert "SpeechRecognition || window.webkitSpeechRecognition" in html
    assert "document.createElement('input')" in html
    assert "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb" not in html
    assert "connectedWeb3Balance = 1.234" not in html


def test_strategy_validation_endpoint_is_declared_read_only():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    block = source[
        source.index("def api_validate_strategy") : source.index(
            '@api.post("/api/strategy/params")'
        )
    ]
    assert "ast.parse" in block and "compile(" in block
    assert "open(" not in block
    assert "atomic_write" not in block
    assert '"written": False' in block


def test_job_status_metadata_is_json_serializable(tmp_path):
    pid_file = tmp_path / "meta.pid"
    log_file = tmp_path / "meta.log"
    start_background([sys.executable, "-c", "pass"], str(pid_file), str(log_file))
    status = _wait_for_job(pid_file)
    json.dumps(status)
