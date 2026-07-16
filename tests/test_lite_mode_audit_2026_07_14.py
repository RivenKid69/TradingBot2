"""Regression coverage for the 2026-07-14 Lite Mode audit closures (L2-001…L2-024).

These tests exercise the actual UI→API→worker contracts instead of only
checking that modules exist (audit L2-017): fail-closed Emergency Halt,
canonical job commands, LeakGuard delay floor, honest telemetry defaults,
typed risk-limit persistence, backend workflow evidence, and the absence of
fabricated success constants in the UI.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("SEASONALITY_API_TOKEN", "test-token-lite-audit")

from fastapi.testclient import TestClient

import app as app_module
from app import api, get_default_config_for_asset
from desktop_job_runtime import code_root, prepare_python_command, worker_environment

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "index.html").read_text(encoding="utf-8")
APP_SRC = (ROOT / "app.py").read_text(encoding="utf-8")
SPEC_SRC = (ROOT / "packaging" / "riven_backend.spec").read_text(encoding="utf-8")

# The global auth middleware only whitelists loopback peers; the TestClient
# peer is "testclient", so authenticate explicitly with the API token.
client = TestClient(api, headers={"X-API-Key": app_module.API_TOKEN})


# ---------------------------------------------------------------------------
# L2-001 — Emergency Halt is fail-closed
# ---------------------------------------------------------------------------

def test_panic_halt_without_backend_reports_unavailable(monkeypatch):
    for var in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "OANDA_API_KEY",
                "OANDA_ACCOUNT_ID", "BINANCE_API_KEY", "BINANCE_API_SECRET"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(app_module, "_CCEA_SUPERVISOR", None, raising=False)

    try:
        res = client.post("/api/panic_halt")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "unavailable"
        assert data["execution_mode"] == "no_execution_backend"
        assert data["orders_cancelled"] == 0
        assert data["positions_liquidated"] == []
        assert data["kill_switch_tripped"] is True
        assert "не отменялся" in data["detail"] or "не закрывалась" in data["detail"]
    finally:
        client.post("/api/panic_reset")


def test_panic_halt_source_has_no_fabricated_results():
    assert "LIQ_MOCK" not in APP_SRC
    assert "error_fallback_simulated" not in APP_SRC
    # The per-asset hardcoded liquidation fixtures must be gone.
    assert '"symbol": "SPY", "qty": 200, "price": 512.4' not in APP_SRC
    assert "Mock Equities Liquidation" not in APP_SRC
    assert "Mock Futures Liquidation" not in APP_SRC


def test_lite_ui_handles_non_success_halt_statuses():
    assert "data.status === 'unavailable'" in HTML
    assert "data.status === 'failed'" in HTML


# ---------------------------------------------------------------------------
# L2-003 / L2-004 — Data Manager contracts and LeakGuard floor
# ---------------------------------------------------------------------------

def _capture_job_cmd(monkeypatch):
    captured = {}

    def fake_start_background(cmd, pid_file, log_file):
        captured["cmd"] = [str(c) for c in cmd]
        captured["pid_file"] = pid_file
        return 4242

    monkeypatch.setattr(app_module, "start_background", fake_start_background)
    monkeypatch.setattr(app_module, "background_running", lambda _pid: False)
    return captured


def test_run_no_trade_contract_and_alias(monkeypatch):
    captured = _capture_job_cmd(monkeypatch)
    res = client.post("/api/run_job", json={
        "job": "run_notrade",  # legacy alias must map to the canonical worker
        "params": {"data": "data/targets.parquet", "out": "data/targets_masked.parquet"},
    })
    assert res.status_code == 200, res.text
    cmd = captured["cmd"]
    assert any(c.endswith("apply_no_trade_mask.py") for c in cmd)
    assert "--sandbox_config" in cmd and "--config" not in cmd
    assert "--data" in cmd and "data/targets.parquet" in cmd
    assert "--out" in cmd and "data/targets_masked.parquet" in cmd
    assert "run_no_trade.pid" in captured["pid_file"]


def test_run_splits_contract_passes_data_and_simple_mode(monkeypatch):
    captured = _capture_job_cmd(monkeypatch)
    res = client.post("/api/run_job", json={
        "job": "run_splits",
        "params": {"data": "data/targets.parquet", "n_splits": 5, "train_size_pct": 80},
    })
    assert res.status_code == 200, res.text
    cmd = captured["cmd"]
    assert any(c.endswith("make_walkforward_splits.py") for c in cmd)
    assert "--data" in cmd and "data/targets.parquet" in cmd
    assert "--n_splits" in cmd and "5" in cmd
    assert "--train_size_pct" in cmd and "80" in cmd


def test_walkforward_worker_accepts_simple_mode_args():
    src = (ROOT / "make_walkforward_splits.py").read_text(encoding="utf-8")
    assert "--n_splits" in src and "--train_size_pct" in src


def test_run_training_table_requires_safe_decision_delay(monkeypatch):
    captured = _capture_job_cmd(monkeypatch)

    res = client.post("/api/run_job", json={
        "job": "run_training_table",
        "params": {"decision_delay_ms": 50},
    })
    assert res.status_code == 400
    assert "8000" in res.json()["detail"]

    res = client.post("/api/run_job", json={
        "job": "run_training_table",
        "params": {"decision_delay_ms": 8000, "price_col": "close"},
    })
    assert res.status_code == 200, res.text
    cmd = captured["cmd"]
    assert "--decision-delay-ms" in cmd and "8000" in cmd
    assert "--price-col" in cmd and "close" in cmd


def test_unsafe_delay_override_is_recorded(monkeypatch, tmp_path):
    _capture_job_cmd(monkeypatch)
    monkeypatch.setattr(app_module, "GLOBAL_LOGS_DIR", str(tmp_path))
    res = client.post("/api/run_job", json={
        "job": "run_training_table",
        "params": {"decision_delay_ms": 50, "unsafe_decision_delay_override": True},
    })
    assert res.status_code == 200, res.text
    manifest = tmp_path / "lite_unsafe_overrides.jsonl"
    assert manifest.is_file()
    entry = json.loads(manifest.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert entry["override"] == "decision_delay_ms"
    assert entry["value"] == 50


def test_lite_ui_sends_canonical_data_manager_params():
    assert "run_notrade" not in HTML
    assert "'run_no_trade'" in HTML
    assert '"decision_delay_ms": 8000' in HTML
    assert '"decision_delay_ms": 50' not in HTML
    assert '"price_col": "close"' in HTML


def test_unsafe_delay_override_requires_strict_truthiness(monkeypatch):
    """Review fix: JSON strings like "false"/"0" must not bypass the floor."""
    _capture_job_cmd(monkeypatch)
    for bogus in ("false", "0", "no", "", None, 0):
        res = client.post("/api/run_job", json={
            "job": "run_training_table",
            "params": {"decision_delay_ms": 50, "unsafe_decision_delay_override": bogus},
        })
        assert res.status_code == 400, f"override={bogus!r} must NOT bypass the 8000ms floor"


def test_risk_limits_reject_non_finite_values(monkeypatch, tmp_path):
    """Review fix: NaN/Infinity must not slip past range validation."""
    path = tmp_path / "risk.yaml"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(app_module, "RISK_LIMITS_CONFIG_PATH", str(path))
    # httpx's json= serializer refuses NaN, so send the raw non-strict JSON a
    # hostile/buggy client could produce; python json.loads accepts NaN.
    for bad in ("NaN", "Infinity"):
        res = client.post(
            "/api/risk/limits",
            content='{"daily_loss_limit_usd": %s}' % bad,
            headers={"Content-Type": "application/json"},
        )
        # Either our finite-check (400) or pydantic strictness (422) — but the
        # value must never be persisted as .nan/.inf.
        assert res.status_code in (400, 422), res.text
        assert ".nan" not in path.read_text(encoding="utf-8")
        assert ".inf" not in path.read_text(encoding="utf-8")


def test_panic_halt_review_fixes_in_source():
    """Review fixes for the real-broker halt path stay in place."""
    # core_models.Order has quantity/ts, not qty — the broken kwargs are gone.
    assert "qty=abs(qty), order_type=" not in APP_SRC
    assert "create_futures_order_execution_adapter(ExchangeVendor.BINANCE_FUTURES" in APP_SRC
    # P0-C closed: Binance spot now HAS an order-execution adapter, so the halt
    # path flattens crypto-spot for real instead of staying fail-closed. The old
    # "no adapter" fail-closed string must be gone, and the spot flatten wired in.
    assert "нет order-execution адаптера" not in APP_SRC
    assert "Spot is long-only" in APP_SRC
    # Adapter results are checked, not assumed.
    assert "cancel_failures" in APP_SRC and "close_failures" in APP_SRC
    # Broker errors in holdings never fabricate live-looking positions.
    assert APP_SRC.count("except Exception as _broker_exc:") == 5
    assert '"broker_error"' in APP_SRC


def test_ui_review_honesty_fixes():
    """Residual fabrications found by the adversarial review stay removed."""
    assert "Inject live mock ticks" not in HTML
    assert "falling back to simulated random walk" not in HTML
    assert "meta.risk_check || 'PASSED'" not in HTML
    assert "meta.logits || [0.33, 0.34, 0.33]" not in HTML
    assert "LEAK_GUARD: PASSED" not in HTML
    assert "PASSED (Art. 9-15 Compliance)" not in HTML
    assert "net_liquidation_value > 100000 ? 'SAFE'" not in HTML
    # Flat eval metrics shape is read directly.
    assert "metrics[activeAssetKey]" not in HTML
    assert "const trMetrics = metrics.trades || {};" in HTML


# ---------------------------------------------------------------------------
# L2-009 (follow-up) — lite-data Drift Monitor widget must reflect REAL PSI,
# never a hardcoded reassuring "stable / 0.000" (the widget was dead: 0 JS
# writes, permanently green while real drift could be severe).
# ---------------------------------------------------------------------------

def test_drift_widget_no_fabricated_stable_default():
    # The hardcoded green "PSI 0.000 / Стабильно" initial values are gone.
    assert 'id="litedata-drift-val" class="text-emerald-400 font-bold font-mono">0.000<' not in HTML
    assert 'id="litedata-drift-status" class="text-[11px] font-semibold text-white uppercase font-mono">Стабильно' not in HTML


def test_drift_widget_is_wired_to_real_source():
    # A real updater exists, reads the honest telemetry drift source, and is
    # invoked on data-manager load + after computing PSI.
    assert "function refreshLiteDataDrift" in HTML
    assert "/api/telemetry/live" in HTML
    # widget ids are actually written now (were 0 JS writes before)
    assert "getElementById('litedata-drift-status')" in HTML
    assert "getElementById('litedata-drift-val')" in HTML
    assert "refreshLiteDataDrift()" in HTML  # wired into load / triggerLitePsi


def test_analytics_status_defaults_not_falsely_green():
    # WS/Broker/PSI must not start as hardcoded green "OK"/"Стабильно" before a
    # real check — they self-correct from /api/telemetry/live, but the initial
    # paint must be neutral.
    assert 'id="lite-status-ws" class="text-emerald-400 font-bold flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span> OK' not in HTML
    assert 'id="lite-status-broker-api" class="text-emerald-400 font-bold flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span> OK' not in HTML
    assert 'id="lite-psi-val" class="font-bold text-emerald-400">0.05 (Стабильно)' not in HTML


def test_telemetry_live_exposes_honest_drift_shape():
    res = client.get("/api/telemetry/live")
    assert res.status_code == 200
    drift = res.json().get("drift", {})
    # the widget depends on these keys; psi_avg is a number or None (no fabrication)
    assert "psi_avg" in drift and "status" in drift
    assert drift["psi_avg"] is None or isinstance(drift["psi_avg"], (int, float))


# ---------------------------------------------------------------------------
# L2-005 — packaged RL training stack ships its data files
# ---------------------------------------------------------------------------

def test_spec_bundles_sb3_data_files_in_research_profile():
    assert "collect_data_files" in SPEC_SRC
    research_block = SPEC_SRC[SPEC_SRC.index('PROFILE != "research"'):]
    assert '"stable_baselines3"' in research_block
    assert '"gymnasium"' in research_block


def test_pyinstaller_collects_sb3_version_txt():
    """The exact file whose absence crashed packaged run_train (audit L2-005)."""
    hooks = pytest.importorskip("PyInstaller.utils.hooks")
    files = hooks.collect_data_files("stable_baselines3")
    assert any(src.endswith("version.txt") for src, _dest in files)


@pytest.mark.skipif(
    os.environ.get("RIVEN_PACKAGED_SMOKE") != "1"
    or not (ROOT / "dist" / "riven-backend.exe").is_file(),
    reason=(
        "packaged smoke runs in the packaging pipeline (RIVEN_PACKAGED_SMOKE=1) "
        "against a research-profile EXE built AFTER the spec fix; the audited "
        "2026-07-14 EXE predates it and reproduces L2-005 by design"
    ),
)
def test_packaged_exe_can_import_sb3():
    import subprocess
    exe = str(ROOT / "dist" / "riven-backend.exe")
    proc = subprocess.run(
        [exe, "--riven-worker-code", "import stable_baselines3; print(stable_baselines3.__version__)"],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# L2-006 — typed risk-limit persistence with read-back
# ---------------------------------------------------------------------------

def test_risk_limits_roundtrip(monkeypatch, tmp_path):
    path = tmp_path / "risk.yaml"
    path.write_text("max_total_notional: null\nmax_total_exposure_pct: null\n", encoding="utf-8")
    monkeypatch.setattr(app_module, "RISK_LIMITS_CONFIG_PATH", str(path))

    payload = {
        "daily_loss_limit_usd": 1500.0,
        "max_drawdown_pct": 12,
        "max_leverage": 3.0,
        "max_concentration_pct": 20,
        "pdt_guard_enabled": True,
        "span_guard_enabled": True,
        "greeks_guard_enabled": False,
    }
    res = client.post("/api/risk/limits", json=payload)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "saved"
    assert body["applied_to_agent"] is False  # honesty: file save ≠ live enforcement
    applied = body["applied"]
    assert applied["daily_loss_limit_usd"] == 1500.0
    assert applied["max_drawdown_pct"] == 12
    assert applied["max_leverage"] == 3.0
    assert applied["max_concentration_pct"] == 20
    assert applied["span_guard_enabled"] is True
    assert applied["greeks_guard_enabled"] is False

    # All fields must actually be on disk, including the exposure mapping.
    on_disk = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert on_disk["max_total_exposure_pct"] == 0.2
    assert on_disk["lite_limits"]["daily_loss_limit_usd"] == 1500.0

    res = client.get("/api/risk/limits")
    assert res.status_code == 200
    assert res.json()["max_leverage"] == 3.0


def test_risk_limits_validation_rejects_nonsense(monkeypatch, tmp_path):
    path = tmp_path / "risk.yaml"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(app_module, "RISK_LIMITS_CONFIG_PATH", str(path))
    assert client.post("/api/risk/limits", json={"max_drawdown_pct": 0}).status_code == 400
    assert client.post("/api/risk/limits", json={"max_leverage": 0.5}).status_code == 400
    assert client.post("/api/risk/limits", json={"daily_loss_limit_usd": -5}).status_code == 400


def test_lite_ui_saves_all_risk_fields_via_typed_endpoint():
    save_fn = HTML[HTML.index("async function saveRiskLimits"):]
    save_fn = save_fn[:save_fn.index("// Web3 Connection Methods")]
    assert "/api/risk/limits" in save_fn
    for field in ("daily_loss_limit_usd", "max_drawdown_pct", "max_leverage",
                  "max_concentration_pct", "pdt_guard_enabled", "span_guard_enabled",
                  "greeks_guard_enabled"):
        assert field in save_fn, field
    assert "max_total_exposure_pct:\\s*" not in save_fn  # no regex string-replace saves


# ---------------------------------------------------------------------------
# L2-007 — Gas Guard is now a REAL on-chain gas oracle + threshold (2026-07-16),
# no longer a fabricated "active" guard NOR a permanent "NOT IMPLEMENTED" stub.
# ---------------------------------------------------------------------------

def test_gas_guard_is_real_not_fabricated_or_stub():
    # Not a fabricated always-green "active" guard...
    assert "Gas Guard Active" not in HTML
    # ...and no longer a dead "NOT IMPLEMENTED" placeholder either — it's wired
    # to the real oracle endpoint with a live verdict.
    assert "Gas Guard — NOT IMPLEMENTED" not in HTML
    assert "/api/web3/gas_guard" in HTML and "function saveGasGuard" in HTML


# ---------------------------------------------------------------------------
# L2-008 — portfolio risk endpoint is honest
# ---------------------------------------------------------------------------

def test_portfolio_risk_summary_never_fakes_values(monkeypatch):
    for var in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "OANDA_API_KEY",
                "OANDA_ACCOUNT_ID", "BINANCE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(app_module, "_CCEA_SUPERVISOR", None, raising=False)

    res = client.get("/api/portfolio/risk_summary")
    assert res.status_code == 200
    body = res.json()
    if not body["available"]:
        assert body["reason"]
        assert body["var_95_usd"] is None
    assert "methodology" in body


def test_ui_has_no_var_interface_formula():
    assert "nlv * 0.0245" not in HTML.lower()
    assert "STABLE (12.4k)" not in HTML
    assert "NEUTRAL (Δ=-0.12)" not in HTML
    assert "/api/portfolio/risk_summary" in HTML


# ---------------------------------------------------------------------------
# L2-009 — telemetry shows absence as absence
# ---------------------------------------------------------------------------

def test_telemetry_defaults_are_no_data():
    assert 'LATEST_TELEMETRY["psi_avg"] = 0.045' not in APP_SRC
    assert '"psi_status": "no_data"' in APP_SRC
    # No fabricated structured-log fills.
    assert "Order execution complete: Bought 100 AAPL" not in APP_SRC


def test_ui_telemetry_has_no_healthy_defaults():
    assert "HFT_NODE_OK" not in HTML
    assert "let slipBps = 0.8" not in HTML
    assert "lastWs = 'NTP: '" not in HTML
    assert "drift.psi_avg || 0.05" not in HTML
    assert "sidebar-node-status" in HTML


# ---------------------------------------------------------------------------
# L2-010 — job lifecycle reaches terminal UI states
# ---------------------------------------------------------------------------

def test_quantlab_lifecycle_has_terminal_states():
    assert "restoreQuantLabTrainButtons" in HTML
    assert "restoreQuantLabBacktestButtons" in HTML
    # Training poller must consult the real job status endpoint.
    train_fn = HTML[HTML.index("async function startQuantLabTraining"):]
    train_fn = train_fn[:train_fn.index("function restoreQuantLabTrainButtons")]
    assert "/api/job/status?job=run_train" in train_fn


# ---------------------------------------------------------------------------
# L2-011 — Quant Lab follows the active asset class
# ---------------------------------------------------------------------------

def test_quantlab_asset_defaults_exist():
    assert "QUANTLAB_ASSET_DEFAULTS" in HTML
    assert "applyQuantLabAssetDefaults" in HTML
    for asset in ("equity", "forex", "futures", "crypto", "options"):
        assert re.search(rf"\b{asset}:\s*{{\s*strategy:", HTML), asset


# ---------------------------------------------------------------------------
# L2-012 / L2-013 — canonical configs that actually exist
# ---------------------------------------------------------------------------

def test_backend_sandbox_configs_exist_for_all_assets():
    for asset in ("equity", "forex", "futures", "crypto", "options"):
        path = get_default_config_for_asset("sandbox", asset)
        assert (ROOT / path).is_file(), f"{asset}: {path} missing"


def test_ui_sandbox_config_paths_exist():
    fn = HTML[HTML.index("function getActiveSandboxConfigPath"):]
    fn = fn[:fn.index("async function resolveActiveSandboxConfigPath")]
    for path in re.findall(r'configs/[\w./-]+\.yaml', fn):
        assert (ROOT / path).is_file(), path


def test_futures_sandbox_is_backtest_not_live():
    assert get_default_config_for_asset("sandbox", "futures") == "configs/config_backtest_futures.yaml"
    cfg = yaml.safe_load((ROOT / "configs" / "config_backtest_futures.yaml").read_text(encoding="utf-8"))
    assert cfg["mode"] == "backtest"
    # No live-broker connectivity keys in the historical sandbox.
    for live_key in ("api_key", "api_secret", "ib_host", "ib_port", "paper_trading"):
        assert live_key not in cfg, live_key
    assert "configs/config_live_futures.yaml" != get_default_config_for_asset("sandbox", "futures")
    assert "config_backtest_futures.yaml" in HTML


# ---------------------------------------------------------------------------
# L2-014 — empty risk log is not proof of health
# ---------------------------------------------------------------------------

def test_risk_log_states_are_distinguished():
    fn = HTML[HTML.index("async function syncRiskLogs"):]
    fn = fn[:fn.index("// --- LITE TRADE HISTORY ---")]
    assert "Все лимиты в норме" not in fn
    assert "подтверждённо пуст" in fn
    assert "недоступен" in fn
    assert "Risk engine" in fn


# ---------------------------------------------------------------------------
# L2-015 — every wired onclick handler resolves to a definition
# ---------------------------------------------------------------------------

def test_all_onclick_handlers_are_defined():
    names = set(re.findall(r'onclick="([A-Za-z_$][\w$]*)\s*\(', HTML))
    names |= set(re.findall(r"onclick='([A-Za-z_$][\w$]*)\s*\(", HTML))
    missing = [
        n for n in sorted(names)
        if not re.search(rf"function\s+{re.escape(n)}\s*\(", HTML)
        and not re.search(rf"window\.{re.escape(n)}\s*=", HTML)
        and not re.search(rf"(?:const|let|var)\s+{re.escape(n)}\s*=", HTML)
    ]
    assert not missing, f"undefined onclick handlers: {missing}"


def test_previously_missing_handlers_now_exist():
    for name in ("hideWorkflowConsole", "autoPopulateAndStartTraining", "pollHealthchecks"):
        assert re.search(rf"function\s+{name}\s*\(", HTML), name


# ---------------------------------------------------------------------------
# L2-016 — source runtime separates code root from data root
# ---------------------------------------------------------------------------

def test_prepare_python_command_resolves_scripts_against_code_root(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.chdir(tmp_path)  # simulates RIVEN_DATA_DIR as CWD
    cmd = prepare_python_command([sys.executable, "make_features.py", "--help"])
    assert os.path.isabs(cmd[1]), cmd
    assert Path(cmd[1]).is_file()
    assert Path(cmd[1]).parent == Path(code_root())


def test_worker_environment_exposes_code_root_on_pythonpath():
    env = worker_environment()
    assert code_root() in env["PYTHONPATH"].split(os.pathsep)
    assert env["PYTHONUNBUFFERED"] == "1"


# ---------------------------------------------------------------------------
# L2-002 — Quick Start readiness comes from backend evidence
# ---------------------------------------------------------------------------

def test_workflow_readiness_endpoint_shape():
    res = client.get("/api/workflow/readiness")
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body["ready_for_trading"], bool)
    assert body["data_source"] == "backend_filesystem_evidence"
    for key in ("prices", "features", "targets", "training_table", "model", "eval_metrics"):
        assert key in body["artifacts"], key
        assert "exists" in body["artifacts"][key]
    assert "run_train" in body["jobs"]


def test_quick_start_ui_uses_backend_evidence():
    assert "/api/workflow/readiness" in HTML
    assert "renderPipelineStatusFromEvidence" in HTML
    # Fabricated success constants must be gone.
    for fabricated in ("12,450", "0 NaN строк (100% OK)", "Успешно (OK)",
                       "0 утечек (Чисто)", "'1.84'", "'+24.5%'"):
        assert fabricated not in HTML, fabricated
    # Trade-readiness can only be claimed with backend confirmation.
    assert HTML.count("ГОТОВО К ТОРГАМ") == 1
    assert "ГОТОВО К ТОРГАМ (ПОДТВЕРЖДЕНО BACKEND)" in HTML


# ---------------------------------------------------------------------------
# L2-018 / L2-019 / L2-020 / L2-021 / L2-022 / L2-023 / L2-024 — UX honesty
# ---------------------------------------------------------------------------

def test_import_date_range_is_dynamic():
    assert 'value="2024-12-31"' not in HTML
    assert "initLiteDataDateRange" in HTML


def test_data_provider_is_not_labeled_broker():
    assert "Active Adapter (Broker)" not in HTML
    assert "Market Data Provider" in HTML
    assert "execution-broker-badge" in HTML


def test_demo_trades_are_labeled_per_row():
    assert "liteHistorySimulated" in HTML
    assert ">DEMO</span>" in HTML


def test_strategy_registry_renamed_to_local_bookmarks():
    assert "Реестр сохраненных количественных стратегий" not in HTML
    assert "Локальные закладки стратегий" in HTML


def test_asset_adapter_selection_is_validate_then_commit():
    assert "commitSystemState" in HTML
    fn = HTML[HTML.index("async function commitSystemState"):]
    fn = fn[:fn.index("async function selectAsset")]
    assert "res.ok" in fn


def test_module_switch_resets_scroll():
    fn = HTML[HTML.index("function switchModule"):]
    fn = fn[:fn.index("function fetchStatus") if "function fetchStatus" in fn else 4000]
    assert "scrollTo" in fn


def test_copilot_intro_is_honest():
    assert "полностью проиндексирован" not in HTML
    assert "прямой доступ ко всем 22 модулям" not in HTML
    assert "rule-based" in HTML
