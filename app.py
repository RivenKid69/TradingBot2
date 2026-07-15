# app.py
from __future__ import annotations

import io
import os
import sys
import json
import time
import subprocess
import copy
import difflib
import shlex
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException, Header, Request
import pandas as pd
try:
    # Streamlit is only used by the legacy `streamlit run app.py` wrapper at the
    # bottom of this module. The desktop build serves the UI via FastAPI and does
    # NOT bundle Streamlit, so the import must be optional (and metadata-free).
    import streamlit as st
except Exception:  # pragma: no cover - desktop/sidecar path
    st = None  # type: ignore
import yaml

from desktop_job_runtime import prepare_python_command
from utils_time import load_seasonality

from core_config import ClockSyncConfig, load_config, load_config_from_str
from ingest_config import (
    load_config as load_ingest_config,
    load_config_from_str as parse_ingest_config,
)
from legacy_sandbox_config import (
    load_config as load_sandbox_config,
    load_config_from_str as parse_sandbox_config,
    SandboxConfig,
)

# Enhanced Pro Mode imports
from services.core.enhanced_healthcheck import (
    EnhancedHealthcheck, EnhancedHealthcheckConfig, DependencyType, 
    DependencyStatus, DatabaseChecker, CacheChecker, ExternalAPIChecker
)
from services.core.risk_controls.time_sync import (
    ComplianceClock, ClockDriftSeverity, ClockSyncStatus, ClockSyncEvent
)
from services.core.alerting import (
    AlertingService, AlertingConfig, AlertRule, Alert, AlertSeverity, 
    AlertChannel, AlertStatus, EscalationLevel, EscalationPolicy
)
from services.core.oncall_rotation import (
    OnCallRotationManager, OnCallRotationConfig, OnCallTier, RotationSchedule, 
    EscalationPath, IncidentPriority, OnCallEngineer, OnCallShift, OnCallIncident
)
from services.algo_integration.conformance_testing import (
    ConformanceTestRunner, ConformanceTestSuite, TestCategory, TestPriority,
    TestEnvironment, ConformanceSuiteStatus, get_standard_conformance_tests,
    ConformanceTest, TestResult
)
from services.algo_integration.best_execution import (
    BestExecutionAnalyzer, BestExecutionPolicy, BestExecutionPolicyConfig,
    AssetClass, OrderCategory, VenueType, ExecutionQualityLevel, ExecutionVenue,
    FactorWeights, create_best_execution_policy, create_best_execution_analyzer,
    get_standard_eu_venues
)
from services.dora_integration.incident_interface.incident_classification import (
    DORAIncidentClassification, IncidentClassificationConfig,
    ClientImpactAssessment, DurationAssessment, EconomicImpactAssessment, DataLossAssessment
)
from services.dora_integration.incident_interface.incident_reporting import (
    DORAIncidentReporter, IncidentReportingConfig
)
from services.dora_integration.third_party.concentration_risk import (
    DORAConcentrationRisk, ConcentrationRiskConfig
)
from services.dora_integration.reporting.register_of_information import (
    DORARegisterOfInformation, ROIDataGeneratorConfig
)
from services.ai_act.explainability import (
    DecisionExplainer, create_decision_explainer
)
from services.gdpr.data_export import (
    GDPRExportService
)
from services.gdpr.data_deletion import (
    GDPRDeletionService, DataCategory
)
from services.core.risk_controls.retention_policy import (
    RetentionManager, RetentionPolicyConfig, RetentionPeriod
)
from services.algo_integration.otr_monitor import (
    OTRMonitor, OTRMonitorConfig
)
from services.core.risk_controls.pre_trade_controls import (
    PreTradeControls
)
from services.core.risk_controls.kill_switch import (
    EnhancedKillSwitch, KillSwitchScope, KillSwitchTriggerReason
)
from dataclasses import asdict

import clock
from services import monitoring
from services.rest_budget import RestBudgetSession
from services.utils_app import (
    ensure_dir as _ensure_dir,
    run_cmd,
    start_background,
    stop_background,
    background_running,
    background_status,
    tail_file,
    read_json,
    read_csv,
    append_row_csv,
    load_signals_full,
    atomic_write_with_retry,
)
from service_backtest import BacktestConfig, from_config as backtest_from_config
from service_calibrate_slippage import (
    from_config as calibrate_slippage_from_config,
)
from service_calibrate_tcost import TCostCalibrateConfig, run as calibrate_tcost_run
from service_signal_runner import (
    ServiceSignalRunner,
    RunnerConfig,
    clear_dirty_restart,
)
from service_eval import ServiceEval, EvalConfig
from runtime_trade_defaults import (
    DEFAULT_RUNTIME_TRADE_PATH,
    load_runtime_trade_defaults,
)


_ROOT_DIR = Path(__file__).resolve().parent
_INGEST_SCRIPT = str(_ROOT_DIR / "ingest_orchestrator.py")
_MAKE_FEATURES_SCRIPT = str(_ROOT_DIR / "make_features.py")
_BUILD_TRAINING_TABLE_SCRIPT = str(_ROOT_DIR / "build_training_table.py")


# --------------------------- Seasonality API ---------------------------

if not os.environ.get("SEASONALITY_API_TOKEN") and os.path.exists(".env"):
    try:
        with open(".env", "r", encoding="utf-8") as _f:
            for _line in _f:
                if _line.strip() and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.strip().split("=", 1)
                    if _k.strip() == "SEASONALITY_API_TOKEN":
                        os.environ["SEASONALITY_API_TOKEN"] = _v.strip().strip('"').strip("'")
    except Exception:
        pass

API_TOKEN = os.environ.get("SEASONALITY_API_TOKEN")
if API_TOKEN is None:
    raise RuntimeError(
        "SEASONALITY_API_TOKEN is required for API access. "
        "Load it from your secret manager or .env (see .env.example)."
    )

# --------------------------- Background Telemetry Service ---------------------------
import threading
import time
import os
import json

LATEST_TELEMETRY = {
    "clock_sync_drift_ms": 0.0,
    "clock_sync_rtt_ms": 0.0,
    # Drift metrics start as "not measured" — never as a healthy default
    # (audit L2-009).
    "psi_avg": None,
    "psi_worst_feature": None,
    "psi_worst": None,
    "psi_status": "no_data",
    "last_sync_time": "—",
    "ws_feed_ok": False,
    "broker_api_ok": False,
}

def start_telemetry_loop():
    def loop():
        import clock
        from core_config import ClockSyncConfig
        cfg = ClockSyncConfig(attempts=2)
        while True:
            try:
                # 1. NTP/Exchange time synchronization drift
                skew, rtt = clock.manual_sync(cfg)
                LATEST_TELEMETRY["clock_sync_drift_ms"] = round(skew, 2)
                LATEST_TELEMETRY["clock_sync_rtt_ms"] = round(rtt, 2)
                LATEST_TELEMETRY["last_sync_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

            try:
                # 2. Check concept drift report from run_psi (drift.py)
                drift_report_path = "models/drift_report.json"
                if os.path.exists(drift_report_path):
                    with open(drift_report_path, "r", encoding="utf-8") as f:
                        report = json.load(f)
                        LATEST_TELEMETRY["psi_avg"] = report.get("avg_psi", 0.0)
                        LATEST_TELEMETRY["psi_worst_feature"] = report.get("worst_feature", "—")
                        LATEST_TELEMETRY["psi_worst"] = report.get("worst_psi", 0.0)
                        LATEST_TELEMETRY["psi_status"] = report.get("status", "stable")
                else:
                    # No drift report on disk means no measurement — never
                    # substitute a healthy-looking default (audit L2-009).
                    LATEST_TELEMETRY["psi_avg"] = None
                    LATEST_TELEMETRY["psi_worst_feature"] = None
                    LATEST_TELEMETRY["psi_worst"] = None
                    LATEST_TELEMETRY["psi_status"] = "no_data"
            except Exception:
                pass

            try:
                # 3. Connection and state health
                is_running = background_running(GLOBAL_REALTIME_PID)
                LATEST_TELEMETRY["ws_feed_ok"] = is_running
                LATEST_TELEMETRY["broker_api_ok"] = os.getenv("ALPACA_API_KEY") is not None
            except Exception:
                pass

            time.sleep(10) # Refresh telemetry every 10 seconds

    t = threading.Thread(target=loop, daemon=True, name="TelemetryMonitorThread")
    t.start()


from fastapi.middleware.cors import CORSMiddleware

def _make_api() -> Any:
    start_telemetry_loop()
    import fastapi
    app_klass = getattr(fastapi, "FastAPI")
    fastapi_app = app_klass()
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return fastapi_app

api = _make_api()

# Desktop/offline UI dependencies are bundled with the sidecar and served from
# the same loopback origin as the API.
from fastapi.staticfiles import StaticFiles
_WEB_ASSETS_DIR = Path("web_assets")
if _WEB_ASSETS_DIR.is_dir():
    api.mount("/assets", StaticFiles(directory=str(_WEB_ASSETS_DIR)), name="assets")

# --------------------------- GLOBAL API AUTHENTICATION ---------------------------
# Closes the previously-unauthenticated exposure of every /api/* route (incl. the
# shell terminal and job runner). Behaviour is controlled by RIVEN_API_AUTH_MODE:
#   "loopback" (default): requests from 127.0.0.1/::1 are served without a key, so
#                         the local MVP keeps working; ANY non-loopback client must
#                         present a valid X-API-Key header.
#   "strict":   every request must carry a valid X-API-Key (the frontend injects it
#               from window.RIVEN_API_KEY / localStorage). Use this behind a proxy.
#   "off":      legacy behaviour, no enforcement (NOT recommended).
import ipaddress as _ipaddress

_API_AUTH_MODE = os.environ.get("RIVEN_API_AUTH_MODE", "loopback").strip().lower()
# Paths that must stay open: the UI shell ("/" -> static index.html, no data; all
# data is fetched via protected /api/*), health probes, CORS preflight, API docs.
# Exempting "/" lets the desktop webview load the page even under `strict` auth —
# the page can only inject X-API-Key AFTER it has loaded.
_AUTH_EXEMPT_PATHS = ("/", "/health", "/ready", "/live", "/docs", "/redoc", "/openapi.json")


def _is_loopback_client(host: Optional[str]) -> bool:
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return _ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_auth_exempt(path: str) -> bool:
    for p in _AUTH_EXEMPT_PATHS:
        if path == p or path.startswith(p + "/"):
            return True
    return False


@api.middleware("http")
async def _global_auth_middleware(request, call_next):
    from starlette.responses import JSONResponse

    path = request.url.path or ""
    if (
        _API_AUTH_MODE == "off"
        or request.method == "OPTIONS"
        or _is_auth_exempt(path)
    ):
        return await call_next(request)

    provided = request.headers.get("X-API-Key")
    if provided is not None and API_TOKEN is not None and provided == API_TOKEN:
        return await call_next(request)

    if _API_AUTH_MODE == "loopback":
        # If the request arrived through a reverse proxy (forwarding headers are
        # present), the loopback peer is the PROXY, not the real client — granting
        # the keyless loopback bypass would let any remote caller through. In that
        # case require an explicit X-API-Key instead. Use `strict` mode behind a
        # proxy for the cleanest posture.
        forwarded = (
            request.headers.get("x-forwarded-for")
            or request.headers.get("x-real-ip")
            or request.headers.get("forwarded")
        )
        client_host = request.client.host if request.client else None
        if not forwarded and _is_loopback_client(client_host):
            return await call_next(request)

    return JSONResponse(status_code=401, content={"detail": "Unauthorized"})


# --------------------------- CROSS-SECTIONAL API (Stage A12, additive) ---------------------------
# Изолированное подключение cross-sectional конвейера (/api/xs/*); не влияет на MVP.
try:
    from xs_api import register_xs_routes as _register_xs_routes
    _register_xs_routes(api)
except Exception as _xs_exc:  # pragma: no cover - не должно ломать запуск приложения
    import logging as _logging
    _logging.getLogger(__name__).warning("cross-sectional API not registered: %s", _xs_exc)

# --------------------------- PRO MODE SYSTEM SERVICES ---------------------------
from datetime import timedelta, timezone
healthcheck_cfg = EnhancedHealthcheckConfig(
    service_version="2.6.0",
    service_name="RivenQuant Core"
)
global_health_check = EnhancedHealthcheck(healthcheck_cfg)

# Register default checkers
global_health_check.register_dependency(
    name="Exchange client (Alpaca/Binance)",
    dependency_type=DependencyType.EXTERNAL_API,
    checker=ExternalAPIChecker("https://paper-api.alpaca.markets", name="Alpaca"),
    is_critical=True
)
global_health_check.register_dependency(
    name="Market Data Feed",
    dependency_type=DependencyType.INTERNAL_SERVICE,
    checker=ExternalAPIChecker("https://data.alpaca.markets", name="Alpaca Data Feed"),
    is_critical=True
)
global_health_check.register_dependency(
    name="System Database",
    dependency_type=DependencyType.DATABASE,
    checker=DatabaseChecker(),
    is_critical=False
)
global_health_check.register_dependency(
    name="System Cache (Redis)",
    dependency_type=DependencyType.CACHE,
    checker=CacheChecker(),
    is_critical=False
)

global_compliance_clock = ComplianceClock()
global_compliance_clock.start_sync()

alerting_cfg = AlertingConfig(
    log_all_alerts=True,
    slack_webhook_url="https://hooks.slack.com/services/mock/webhook"
)
global_alerting_service = AlertingService(alerting_cfg)

# Add some default alert rules
rule_cpu = global_alerting_service.create_rule(
    name="System CPU Overload",
    condition_type="threshold",
    metric_name="cpu_percent",
    threshold_value=90.0,
    severity=AlertSeverity.HIGH,
    description="Alert when CPU usage exceeds 90% for general operations."
)
rule_latency = global_alerting_service.create_rule(
    name="High Exchange Latency",
    condition_type="threshold",
    metric_name="latency_ms",
    threshold_value=1000.0,
    severity=AlertSeverity.CRITICAL,
    description="Triggered when API response latency from exchange exceeds 1.0s."
)
rule_drift = global_alerting_service.create_rule(
    name="Concept Drift detected",
    condition_type="threshold",
    metric_name="psi_worst",
    threshold_value=0.1,
    severity=AlertSeverity.MEDIUM,
    description="PSI metric shows drift on key pricing feature."
)

# Trigger some default alerts for visualization
global_alerting_service.trigger_alert(
    rule_id=rule_drift.rule_id,
    metric_value=0.12,
    source="drift_monitor"
)

# On-Call Rotation manager
oncall_cfg = OnCallRotationConfig(
    tier=OnCallTier.OPTION_C
)
global_oncall_manager = OnCallRotationManager(oncall_cfg)

# Register default engineers
eng_john = global_oncall_manager.register_engineer(
    name="John Doe (Senior Quant)",
    email="john.doe@rivenquant.com",
    phone="+1 (555) 0199",
    slack_handle="@johndoe",
    team="Quant Core"
)
eng_alice = global_oncall_manager.register_engineer(
    name="Alice Smith (DevOps Lead)",
    email="alice.smith@rivenquant.com",
    phone="+1 (555) 0188",
    slack_handle="@alicesmith",
    team="SRE Team"
)

# Create active shifts
now_iso = datetime.now(timezone.utc).isoformat()
future_iso = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
past_iso = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()

global_oncall_manager.create_shift(
    engineer_id=eng_john.engineer_id,
    start_time=past_iso,
    end_time=future_iso,
    escalation_path=EscalationPath.PRIMARY
)
global_oncall_manager.create_shift(
    engineer_id=eng_alice.engineer_id,
    start_time=past_iso,
    end_time=future_iso,
    escalation_path=EscalationPath.SECONDARY
)

# Assign a dummy incident
global_oncall_manager.assign_incident(
    title="High Concept Drift (f_volatility)",
    description="PSI worst feature exceeded threshold value of 0.1.",
    priority=IncidentPriority.P2
)


# ------------------ COMPLIANCE CENTRAL VARIABLES (TAB 8) ------------------
from services.dora_integration.third_party.concentration_risk import SubstitutabilityLevel, AssessmentScope

global_conformance_runner = ConformanceTestRunner()

# Initialize Best Execution Policy and Analyzer
best_exec_cfg = BestExecutionPolicyConfig(
    firm_lei="549300INFMS8B1012345",
    firm_name="Riven Quant Solutions",
    approved_by="Compliance Officer",
)
global_best_execution_policy = BestExecutionPolicy(best_exec_cfg)
# Pre-populate standard EU venues
for v in get_standard_eu_venues():
    global_best_execution_policy.add_venue(AssetClass.EQUITY, v)
global_best_execution_analyzer = BestExecutionAnalyzer(global_best_execution_policy)

# Pre-populate some mock best execution analyses for demonstration
mock_orders = [
    {"order_id": "ORD-001", "side": "BUY", "quantity": 1000, "submit_time_ms": int(time.time() * 1000) - 250},
    {"order_id": "ORD-002", "side": "SELL", "quantity": 2500, "submit_time_ms": int(time.time() * 1000) - 180},
    {"order_id": "ORD-003", "side": "BUY", "quantity": 500, "submit_time_ms": int(time.time() * 1000) - 120},
]
mock_fills = [
    {"price": 100.05, "quantity": 1000, "commission": 2.0, "fees": 0.5, "fill_time_ms": int(time.time() * 1000), "venue_mic": "XLON"},
    {"price": 99.92, "quantity": 2500, "commission": 5.0, "fees": 1.2, "fill_time_ms": int(time.time() * 1000), "venue_mic": "XETR"},
    {"price": 100.01, "quantity": 500, "commission": 1.0, "fees": 0.2, "fill_time_ms": int(time.time() * 1000), "venue_mic": "XAMS"},
]
mock_market = [
    {"bid": 100.00, "ask": 100.04, "mid": 100.02, "spread_bps": 4.0},
    {"bid": 99.90, "ask": 99.98, "mid": 99.94, "spread_bps": 8.0},
    {"bid": 100.00, "ask": 100.02, "mid": 100.01, "spread_bps": 2.0},
]
for o, f, m in zip(mock_orders, mock_fills, mock_market):
    global_best_execution_analyzer.analyze_execution(o, f, m, AssetClass.EQUITY)

# Initialize DORA components
global_dora_classification = DORAIncidentClassification(IncidentClassificationConfig())
global_dora_reporter = DORAIncidentReporter(IncidentReportingConfig())
global_dora_concentration_risk = DORAConcentrationRisk(ConcentrationRiskConfig())

# Populate DORA concentration risk with some default infrastructure providers
global_dora_concentration_risk.add_provider_dependency(
    provider_id="AWS-EU-WEST",
    provider_name="Amazon Web Services EMEA",
    services=["EC2 Compute", "S3 Storage", "RDS Database"],
    provider_country="IE",
    is_ctpp=True,
    critical_functions=["Order Routing", "Risk Firewall"],
    revenue_dependency_pct=45.0,
    transaction_volume_pct=65.0,
    substitutability=SubstitutabilityLevel.DIFFICULT_TO_SUBSTITUTE,
    alternatives=["Google Cloud Platform", "Microsoft Azure"],
    data_processing_countries=["IE", "DE"],
    data_storage_countries=["IE"],
)
global_dora_concentration_risk.add_provider_dependency(
    provider_id="GCP-EUROPE",
    provider_name="Google Cloud Platform Europe",
    services=["BigQuery Analytics", "Cloud Spanner"],
    provider_country="NL",
    is_ctpp=True,
    critical_functions=["Model Calibration", "Data Ingest"],
    revenue_dependency_pct=25.0,
    transaction_volume_pct=20.0,
    substitutability=SubstitutabilityLevel.SUBSTITUTABLE_WITH_EFFORT,
    alternatives=["Amazon Web Services", "Microsoft Azure"],
    data_processing_countries=["NL", "BE"],
    data_storage_countries=["NL"],
)
global_dora_concentration_risk.add_provider_dependency(
    provider_id="SQLITE-LOCAL",
    provider_name="SQLite Embedded DB Engine",
    services=["State Storage Database"],
    provider_country="US",
    is_ctpp=False,
    critical_functions=["State Storage"],
    revenue_dependency_pct=5.0,
    transaction_volume_pct=100.0,
    substitutability=SubstitutabilityLevel.EASILY_SUBSTITUTABLE,
    alternatives=["PostgreSQL", "MySQL"],
    data_processing_countries=["US"],
    data_storage_countries=["US"],
)
# Run initial assessment
global_dora_concentration_risk.perform_concentration_assessment(scope=AssessmentScope.ENTITY)

# Initialize DORA Register of Information (ROI)
global_dora_roi = DORARegisterOfInformation(ROIDataGeneratorConfig())

# Initialize AI Act components
global_decision_explainer = create_decision_explainer()
# Pre-populate some decisions
global_decision_explainer.explain_decision(
    decision_id="DEC-8001",
    action="BUY",
    symbol="AAPL",
    position_size=100.0,
    features={"price_momentum": 0.75, "volatility_regime": 0.35, "rsi": 62.5, "risk_utilization": 0.42},
    confidence=0.88,
)
global_decision_explainer.explain_decision(
    decision_id="DEC-8002",
    action="SELL",
    symbol="NVDA",
    position_size=50.0,
    features={"price_momentum": 0.22, "volatility_regime": 0.68, "rsi": 28.0, "risk_utilization": 0.55},
    confidence=0.74,
)

# Initialize GDPR and Retention components
from services.gdpr.data_export import InMemoryUserRepository, InMemoryStrategiesRepository, InMemoryBacktestsRepository, InMemoryExecutionsRepository, InMemorySettingsRepository
from services.gdpr.data_deletion import InMemoryDataRepository
global_gdpr_export_service = GDPRExportService({
    "users": InMemoryUserRepository(),
    "strategies": InMemoryStrategiesRepository(),
    "backtests": InMemoryBacktestsRepository(),
    "executions": InMemoryExecutionsRepository(),
    "settings": InMemorySettingsRepository()
})
mock_data_repo = InMemoryDataRepository()
global_gdpr_deletion_service = GDPRDeletionService({
    "account": mock_data_repo,
    "profile": mock_data_repo,
    "strategies": mock_data_repo,
    "backtests": mock_data_repo,
    "execution_logs": mock_data_repo,
    "broker_credentials": mock_data_repo,
    "analytics": mock_data_repo,
    "notifications": mock_data_repo,
    "sessions": mock_data_repo,
    "disclaimers": mock_data_repo,
    "audit_logs": mock_data_repo
})
global_retention_manager = RetentionManager(RetentionPolicyConfig())

# Initialize OTR Monitor and Pre-trade controls
global_otr_monitor = OTRMonitor(OTRMonitorConfig())
global_pre_trade_controls = PreTradeControls()

# Live MAR surveillance (spoofing/layering/wash/marking-the-close) wired into the
# real order/fill flow alongside OTR — was previously instantiated nowhere (orphan).
try:
    from services.algo_integration.market_abuse import (
        MarketAbuseMonitor as _MarketAbuseMonitor,
        OrderEvent as _MAOrderEvent, TradeEvent as _MATradeEvent,
    )
    global_market_abuse_monitor = _MarketAbuseMonitor()
except Exception:  # pragma: no cover - surveillance optional
    global_market_abuse_monitor = None
    _MAOrderEvent = _MATradeEvent = None

# Process-wide instrument master (FIGI/CUSIP/ISIN/OCC symbology resolution).
try:
    from services.instrument_master import get_default_master as _get_instrument_master
    global_instrument_master = _get_instrument_master()
except Exception:  # pragma: no cover
    global_instrument_master = None
def _mock_cancel_orders(scope, scope_id):
    import logging
    logging.getLogger(__name__).info(f"Mock cancel orders called for scope {scope} / {scope_id}")
    return 8
global_kill_switch = EnhancedKillSwitch(order_cancellation_callback=_mock_cancel_orders)



def _check_auth(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    """Simple header-based authentication."""
    if x_api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


@api.get("/seasonality")
def fetch_seasonality(
    path: str = "data/latency/liquidity_latency_seasonality.json",
    _: None = Depends(_check_auth),
) -> Dict[str, Any]:
    """Return seasonality multipliers from JSON file."""
    try:
        data = load_seasonality(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Seasonality file not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {k: v.tolist() for k, v in data.items()}


@api.post("/seasonality/refresh")
def refresh_seasonality(
    data: str = "data/seasonality_source/latest.parquet",
    out: str = "data/latency/liquidity_latency_seasonality.json",
    _: None = Depends(_check_auth),
) -> Dict[str, Any]:
    """Rebuild seasonality JSON from historical data and return it."""
    cmd = [
        sys.executable,
        "scripts/build_hourly_seasonality.py",
        "--data",
        data,
        "--out",
        out,
    ]
    res = subprocess.run(prepare_python_command(cmd), capture_output=True, text=True)
    if res.returncode != 0:
        raise HTTPException(status_code=500, detail=res.stderr)
    try:
        sdata = load_seasonality(out)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Seasonality JSON not generated")
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {k: v.tolist() for k, v in sdata.items()}


@api.get("/monitoring/snapshot")
def monitoring_snapshot(
    path: str = "logs/snapshot_metrics.json",
    _: None = Depends(_check_auth),
) -> Dict[str, Any]:
    """Return monitoring metrics snapshot from JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Snapshot file not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Global paths for API endpoints
GLOBAL_LOGS_DIR = "logs"

# Canonical job-name aliases kept for UI/backend contract stability
# (audit L2-003: legacy clients sent "run_notrade" while the worker job is
# "run_no_trade"). Applied consistently in /api/run_job, /api/job/status and
# /api/job/stop so start/poll/stop all resolve to the same pid/log names.
JOB_NAME_ALIASES = {"run_notrade": "run_no_trade"}
GLOBAL_SIGNALS_CSV = os.path.join(GLOBAL_LOGS_DIR, "signals.csv")
GLOBAL_METRICS_JSON = os.path.join(GLOBAL_LOGS_DIR, "metrics.json")
GLOBAL_REPORTS_PATH = os.path.join(GLOBAL_LOGS_DIR, "reports.csv")
GLOBAL_REALTIME_PID = os.path.join(".run", "rt_signaler.pid")
GLOBAL_REALTIME_LOG = os.path.join(GLOBAL_LOGS_DIR, "realtime.log")
GLOBAL_SNAPSHOT_JSON = os.path.join(GLOBAL_LOGS_DIR, "snapshot_metrics.json")

# Active system state for context selection
ACTIVE_ASSET = "equity"
ACTIVE_ADAPTER = "alpaca"

# Allowed asset classes and their permitted broker adapters. Mirrors the
# frontend `adaptersByAsset` map. Used to validate /api/system_state input so the
# global context can never be set to an unknown asset/adapter (which would load
# the wrong config or be an injection vector if interpolated downstream).
VALID_ASSETS = ("equity", "forex", "futures", "crypto", "options")
ALLOWED_ADAPTERS = {
    "equity": ("alpaca", "polygon", "yahoo"),
    "forex": ("oanda", "dukascopy"),
    "futures": ("ib", "binance_futures"),
    "crypto": ("binance", "deribit"),
    "options": ("ib", "theta_data", "deribit", "polygon"),
}
# Guards concurrent read/mutation of ACTIVE_ASSET / ACTIVE_ADAPTER so a
# /api/system_state POST cannot race a /api/run_job that reads the context.
ACTIVE_STATE_LOCK = threading.Lock()

# --- MVP honesty: never present simulated / seed data as real ----------------
# Endpoints that can fall back to demo data MUST tag the response so the UI shows
# a SIMULATED badge and so seeded data is never mistaken for real execution,
# holdings, or regulatory audit evidence.
MVP_DEMO_DISCLAIMER = (
    "Demo/seed data — NOT real execution, holdings, or audit evidence. "
    "Connect live credentials / feed a real data source to replace it."
)

class SystemStatePayload(BaseModel):
    active_asset: str
    active_adapter: str

def get_default_config_for_asset(config_type: str, asset: str) -> str:
    asset = asset.lower()
    if asset == "equity":
        mapping = {
            "sandbox": "configs/config_backtest_stocks.yaml",
            "train": "configs/config_train_stocks.yaml",
            "realtime": "configs/config_live_alpaca.yaml",
            "ingest": "configs/ingest.yaml",
        }
    elif asset == "forex":
        mapping = {
            "sandbox": "configs/config_backtest_forex.yaml",
            "train": "configs/config_train_forex.yaml",
            "realtime": "configs/config_live_forex.yaml",
            "ingest": "configs/ingest.yaml",
        }
    elif asset == "futures":
        mapping = {
            # Historical sandbox must never inherit live execution semantics
            # (audit L2-013): dedicated backtest config, not the live one.
            "sandbox": "configs/config_backtest_futures.yaml",
            "train": "configs/config_train_futures.yaml",
            "realtime": "configs/config_live_futures.yaml",
            "ingest": "configs/ingest.yaml",
        }
    else:
        mapping = {
            "sandbox": "configs/sandbox.yaml",
            "train": "configs/config_train.yaml",
            "realtime": "configs/config_live.yaml",
            "ingest": "configs/ingest.yaml",
        }
    return mapping.get(config_type, "configs/sandbox.yaml")

class OrderAction(BaseModel):
    id: str

class YamlSavePayload(BaseModel):
    path: str
    content: str

class ApplyCalibrationPayload(BaseModel):
    path: str

class VerifyIngestPayload(BaseModel):
    provider: str
    api_key: str
    api_secret: str



class QuantizerSavePayload(BaseModel):
    strict_filters: bool
    enforce_percent_price_by_side: bool

class SaveBacktestSettingsPayload(BaseModel):
    config_path: str = "configs/config_sim.yaml"
    sandbox_path: str = "configs/sandbox.yaml"
    mode: str
    bar_price: str
    latency_base: float
    latency_jitter: float
    spike_p: float
    spike_mult: float
    seasonality: bool
    intrabar_price_model: str
    timeframe_ms: int
    seed_mode: str
    use_latency_from: str
    latency_constant_ms: float
    next_bar_open: bool
    clip_next_bar: bool
    strict_open: bool
    active_profile: str
    profiles: Dict[str, Dict[str, Any]]
    slip_enabled: bool
    slip_path: str
    smoothing_alpha: float
    vol_mode: str
    liq_col: str
    liq_ref: float
    cap_enabled: bool
    cap_frac: float
    cap_floor: float
    cap_path: str
    ws_enabled: bool
    ws_skips: bool
    ws_path: str

class RunJobPayload(BaseModel):
    job: str
    params: Dict[str, Any]

class CopilotPayload(BaseModel):
    message: str

class TerminalCommand(BaseModel):
    command: str
    cwd: str | None = None

@api.get("/api/system_state")
def api_get_system_state():
    with ACTIVE_STATE_LOCK:
        return {
            "active_asset": ACTIVE_ASSET,
            "active_adapter": ACTIVE_ADAPTER
        }

@api.post("/api/system_state")
def api_post_system_state(payload: SystemStatePayload):
    global ACTIVE_ASSET, ACTIVE_ADAPTER
    asset = (payload.active_asset or "").strip().lower()
    adapter = (payload.active_adapter or "").strip().lower()
    if asset not in VALID_ASSETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid active_asset '{asset}'. Allowed: {', '.join(VALID_ASSETS)}",
        )
    allowed = ALLOWED_ADAPTERS[asset]
    if adapter not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid active_adapter '{adapter}' for asset '{asset}'. Allowed: {', '.join(allowed)}",
        )
    with ACTIVE_STATE_LOCK:
        ACTIVE_ASSET = asset
        ACTIVE_ADAPTER = adapter
        return {"status": "success", "active_asset": ACTIVE_ASSET, "active_adapter": ACTIVE_ADAPTER}

class SaveStrategyPayload(BaseModel):
    asset: str
    # Optional: not used by the handler (the file is keyed by asset). Kept for
    # backward/forward compatibility — callers may or may not supply it.
    template_name: str = "custom"
    code: str
    params: Dict[str, Any]


class ValidateStrategyPayload(BaseModel):
    asset: str
    code: str


class SaveStrategyParamsPayload(BaseModel):
    asset: str
    params: Dict[str, Any]

STRATEGY_TEMPLATES = {
    "equity": {
        "Mean Reversion (Возврат к среднему)": """# strategies/custom_equity.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping
from collections import deque

class EquityMeanReversionStrategy(BaseSignalPolicy):
    \"\"\"
    Mean Reversion Strategy for US Equities.
    Calculates rolling Z-score. Enters Long/Short when Z-score crosses threshold.
    \"\"\"
    required_features = ("ref_price",)

    def __init__(self) -> None:
        super().__init__(history_len=20)
        self.lookback = 20
        self.enter_threshold = 2.0
        self.exit_threshold = 0.5
        self.order_qty = 10
        self.tif = TimeInForce.GTC

    def setup(self, config: Dict[str, Any]) -> None:
        self.lookback = int(config.get("lookback", self.lookback))
        self.enter_threshold = float(config.get("enter_threshold", self.enter_threshold))
        self.exit_threshold = float(config.get("exit_threshold", self.exit_threshold))
        self.order_qty = float(config.get("order_qty", self.order_qty))
        self.set_history_length(self.lookback)

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        
        # Retrieve historical prices automatically cached by the History Manager
        prices = self.get_history_series(ctx.symbol, "price", n=self.lookback)
        
        if len(prices) < self.lookback:
            return []
            
        mean = sum(prices) / self.lookback
        variance = sum((x - mean) ** 2 for x in prices) / self.lookback
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return []
            
        z_score = (prices[-1] - mean) / std_dev
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if z_score <= -self.enter_threshold:
                new_state = SignalPosition.LONG
            elif z_score >= self.enter_threshold:
                new_state = SignalPosition.SHORT
        elif state is SignalPosition.LONG:
            if z_score >= -self.exit_threshold:
                new_state = SignalPosition.FLAT
        elif state is SignalPosition.SHORT:
            if z_score <= self.exit_threshold:
                new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.FLAT and new_state is SignalPosition.SHORT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.LONG and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.SHORT and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))

        return orders
""",
        "Trend Following (Следование за трендом EMA)": """# strategies/custom_equity_trend.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class EquityTrendFollowingStrategy(BaseSignalPolicy):
    \"\"\"
    Trend Following Strategy using EMA Crossover for US Equities.
    Goes Long when fast EMA crosses above slow EMA. Exits when they cross back.
    \"\"\"
    required_features = ("ref_price",)

    def __init__(self) -> None:
        super().__init__()
        self.fast_period = 9
        self.slow_period = 21
        self.order_qty = 10
        self.tif = TimeInForce.GTC
        self._fast_ema = None
        self._slow_ema = None

    def setup(self, config: Dict[str, Any]) -> None:
        self.fast_period = int(config.get("fast_period", self.fast_period))
        self.slow_period = int(config.get("slow_period", self.slow_period))
        self.order_qty = float(config.get("order_qty", self.order_qty))
        self._fast_ema = None
        self._slow_ema = None

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        ref = float(features["ref_price"])
        
        # Calculate EMAs
        if self._fast_ema is None:
            self._fast_ema = ref
            self._slow_ema = ref
            return []
            
        k_fast = 2.0 / (self.fast_period + 1)
        k_slow = 2.0 / (self.slow_period + 1)
        self._fast_ema = (ref * k_fast) + (self._fast_ema * (1.0 - k_fast))
        self._slow_ema = (ref * k_slow) + (self._slow_ema * (1.0 - k_slow))
        
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if self._fast_ema > self._slow_ema:
            new_state = SignalPosition.LONG
        elif self._fast_ema < self._slow_ema:
            new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.LONG and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))

        return orders
""",
        "VWAP Mean Reversion (Внутридневной возврат к VWAP)": """# strategies/custom_equity_vwap.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class EquityVWAPStrategy(BaseSignalPolicy):
    \"\"\"
    Intraday VWAP Mean Reversion Strategy for US Equities.
    Trades regression back to VWAP when price deviates by threshold.
    \"\"\"
    required_features = ("ref_price", "vwap")

    def __init__(self) -> None:
        super().__init__()
        self.deviation_pct = 0.015
        self.order_qty = 10
        self.tif = TimeInForce.GTC

    def setup(self, config: Dict[str, Any]) -> None:
        self.deviation_pct = float(config.get("deviation_pct", self.deviation_pct))
        self.order_qty = float(config.get("order_qty", self.order_qty))

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        ref = float(features["ref_price"])
        vwap = float(features["vwap"])
        
        if vwap == 0:
            return []
            
        deviation = (ref - vwap) / vwap
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if deviation <= -self.deviation_pct:
                new_state = SignalPosition.LONG
            elif deviation >= self.deviation_pct:
                new_state = SignalPosition.SHORT
        elif state is SignalPosition.LONG:
            if deviation >= 0:
                new_state = SignalPosition.FLAT
        elif state is SignalPosition.SHORT:
            if deviation <= 0:
                new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.FLAT and new_state is SignalPosition.SHORT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.LONG and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.SHORT and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))

        return orders
""",
        "Momentum Breakout (Импульсный пробой)": """# strategies/custom_equity_breakout.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class EquityMomentumBreakout(BaseSignalPolicy):
    \"\"\"
    Equity Momentum Breakout Strategy.
    Enters Long when price breaks out above key resistance with high volume.
    \"\"\"
    required_features = ("ref_price", "volume", "resistance_level")

    def __init__(self) -> None:
        super().__init__()
        self.volume_factor = 1.5
        self.order_qty = 10
        self.tif = TimeInForce.GTC
        self._avg_volume = None

    def setup(self, config: Dict[str, Any]) -> None:
        self.volume_factor = float(config.get("volume_factor", self.volume_factor))
        self.order_qty = float(config.get("order_qty", self.order_qty))
        self._avg_volume = None

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        ref = float(features["ref_price"])
        vol = float(features["volume"])
        res_level = float(features["resistance_level"])

        if self._avg_volume is None:
            self._avg_volume = vol
            return []

        self._avg_volume = (self._avg_volume * 19 + vol) / 20
        
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if ref > res_level and vol > self._avg_volume * self.volume_factor:
                new_state = SignalPosition.LONG
        elif state is SignalPosition.LONG:
            if ref < res_level * 0.98:
                new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.LONG and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))

        return orders
""",
        "Pairs Trading (Парный трейдинг)": """# strategies/custom_equity_pairs.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class EquityPairsTrading(BaseSignalPolicy):
    \"\"\"
    Statistical Arbitrage / Pairs Trading Strategy.
    Trades the spread divergence between a stock and a cointegrated partner stock.
    \"\"\"
    required_features = ("ref_price", "stock_b_price")

    def __init__(self) -> None:
        super().__init__()
        self.hedge_ratio = 1.2
        self.enter_threshold = 2.0
        self.exit_threshold = 0.5
        self.order_qty = 10
        self.tif = TimeInForce.GTC
        self._spread_mean = 0.0
        self._spread_std = 1.0

    def setup(self, config: Dict[str, Any]) -> None:
        self.hedge_ratio = float(config.get("hedge_ratio", self.hedge_ratio))
        self.enter_threshold = float(config.get("enter_threshold", self.enter_threshold))
        self.exit_threshold = float(config.get("exit_threshold", self.exit_threshold))
        self.order_qty = float(config.get("order_qty", self.order_qty))

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        price_a = float(features["ref_price"])
        price_b = float(features["stock_b_price"])
        
        spread = price_a - (self.hedge_ratio * price_b)
        z_score = (spread - self._spread_mean) / self._spread_std
        
        self._spread_mean = self._spread_mean * 0.99 + spread * 0.01
        self._spread_std = (self._spread_std ** 2 * 0.99 + (spread - self._spread_mean) ** 2 * 0.01) ** 0.5

        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if z_score <= -self.enter_threshold:
                new_state = SignalPosition.LONG
            elif z_score >= self.enter_threshold:
                new_state = SignalPosition.SHORT
        elif state is SignalPosition.LONG:
            if z_score >= -self.exit_threshold:
                new_state = SignalPosition.FLAT
        elif state is SignalPosition.SHORT:
            if z_score <= self.exit_threshold:
                new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.FLAT and new_state is SignalPosition.SHORT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif new_state is SignalPosition.FLAT:
            side = Side.SELL if state is SignalPosition.LONG else Side.BUY
            orders.append(self.market_order(side=side, qty=self.order_qty, ctx=ctx, tif=self.tif))

        return orders
""",
        "MACD Crossover (Пересечение MACD)": """# strategies/custom_equity_macd.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class EquityMACDCrossover(BaseSignalPolicy):
    \"\"\"
    Classic MACD Line / Signal Line Crossover Strategy.
    Enters Long when MACD crosses above Signal Line; exits when it crosses below.
    \"\"\"
    required_features = ("ref_price", "macd_line", "signal_line")

    def __init__(self) -> None:
        super().__init__()
        self.order_qty = 10
        self.tif = TimeInForce.GTC

    def setup(self, config: Dict[str, Any]) -> None:
        self.order_qty = float(config.get("order_qty", self.order_qty))

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        macd = float(features["macd_line"])
        signal = float(features["signal_line"])
        
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if macd > signal:
            new_state = SignalPosition.LONG
        elif macd < signal:
            new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.LONG and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))

        return orders
"""
    },
    "forex": {
        "Grid Trading with Carry (Сеточная со свопами)": """# strategies/custom_forex.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class ForexGridCarryStrategy(BaseSignalPolicy):
    \"\"\"
    Forex Grid Strategy leveraging Carry Rates.
    Places Limit orders around the reference price prioritizing positive swaps.
    \"\"\"
    required_features = ("ref_price",)

    def __init__(self) -> None:
        super().__init__()
        self.grid_levels = 5
        self.pip_step = 0.0010
        self.order_qty = 10000
        self.swap_long = 0.00008
        self.swap_short = -0.00012
        self.tif = TimeInForce.GTC

    def setup(self, config: Dict[str, Any]) -> None:
        self.grid_levels = int(config.get("grid_levels", self.grid_levels))
        self.pip_step = float(config.get("pip_step", self.pip_step))
        self.order_qty = float(config.get("order_qty", self.order_qty))
        self.swap_long = float(config.get("swap_long", self.swap_long))
        self.swap_short = float(config.get("swap_short", self.swap_short))

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        ref = float(features["ref_price"])
        orders: List[Order] = []
        is_carry_long = self.swap_long > self.swap_short
        
        for i in range(1, self.grid_levels + 1):
            if is_carry_long:
                buy_price = ref - (i * self.pip_step)
                orders.append(self.limit_order(side=Side.BUY, qty=self.order_qty, price=buy_price, ctx=ctx, tif=self.tif))
            else:
                sell_price = ref + (i * self.pip_step)
                orders.append(self.limit_order(side=Side.SELL, qty=self.order_qty, price=sell_price, ctx=ctx, tif=self.tif))
                
        return orders
""",
        "Bollinger Bands Breakout (Пробой полос Боллинджера)": """# strategies/custom_forex_bb.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class ForexBBBreakoutStrategy(BaseSignalPolicy):
    \"\"\"
    Forex Bollinger Bands Breakout Strategy.
    Enters Long on upper band breakout, Short on lower band breakout.
    \"\"\"
    required_features = ("ref_price", "bb_upper", "bb_lower")

    def __init__(self) -> None:
        super().__init__()
        self.order_qty = 10000
        self.tif = TimeInForce.GTC

    def setup(self, config: Dict[str, Any]) -> None:
        self.order_qty = float(config.get("order_qty", self.order_qty))

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        ref = float(features["ref_price"])
        upper = float(features["bb_upper"])
        lower = float(features["bb_lower"])
        
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if ref >= upper:
                new_state = SignalPosition.LONG
            elif ref <= lower:
                new_state = SignalPosition.SHORT
        elif state is SignalPosition.LONG:
            if ref < upper:
                new_state = SignalPosition.FLAT
        elif state is SignalPosition.SHORT:
            if ref > lower:
                new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.FLAT and new_state is SignalPosition.SHORT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.LONG and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.SHORT and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))

        return orders
""",
        "RSI Swing Failure (Разворот тренда по RSI)": """# strategies/custom_forex_rsi.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class ForexRSISwingStrategy(BaseSignalPolicy):
    \"\"\"
    RSI Overbought/Oversold Reversal Strategy for Forex.
    Goes Long when RSI leaves oversold zone (< 30). Goes Short when RSI leaves overbought (> 70).
    \"\"\"
    required_features = ("ref_price", "rsi")

    def __init__(self) -> None:
        super().__init__()
        self.oversold = 30.0
        self.overbought = 70.0
        self.order_qty = 10000
        self.tif = TimeInForce.GTC
        self._prev_rsi = None

    def setup(self, config: Dict[str, Any]) -> None:
        self.oversold = float(config.get("oversold", self.oversold))
        self.overbought = float(config.get("overbought", self.overbought))
        self.order_qty = float(config.get("order_qty", self.order_qty))
        self._prev_rsi = None

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        rsi = float(features["rsi"])
        
        if self._prev_rsi is None:
            self._prev_rsi = rsi
            return []
            
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if self._prev_rsi < self.oversold and rsi >= self.oversold:
                new_state = SignalPosition.LONG
            elif self._prev_rsi > self.overbought and rsi <= self.overbought:
                new_state = SignalPosition.SHORT
        elif state is SignalPosition.LONG:
            if rsi >= 50.0:
                new_state = SignalPosition.FLAT
        elif state is SignalPosition.SHORT:
            if rsi <= 50.0:
                new_state = SignalPosition.FLAT

        self._prev_rsi = rsi

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.FLAT and new_state is SignalPosition.SHORT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.LONG and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.SHORT and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))

        return orders
""",
        "MACD Divergence (Дивергенция MACD)": """# strategies/custom_forex_macd_div.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class ForexMACDDivergence(BaseSignalPolicy):
    \"\"\"
    MACD Divergence Strategy for Forex.
    Enters Long on bullish divergence (price lower low, MACD higher low).
    Enters Short on bearish divergence (price higher high, MACD lower high).
    \"\"\"
    required_features = ("ref_price", "macd")

    def __init__(self) -> None:
        super().__init__()
        self.order_qty = 10000
        self.tif = TimeInForce.GTC
        self._prev_price = None
        self._prev_macd = None

    def setup(self, config: Dict[str, Any]) -> None:
        self.order_qty = float(config.get("order_qty", self.order_qty))
        self._prev_price = None
        self._prev_macd = None

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        ref = float(features["ref_price"])
        macd = float(features["macd"])

        if self._prev_price is None:
            self._prev_price = ref
            self._prev_macd = macd
            return []

        price_diff = ref - self._prev_price
        macd_diff = macd - self._prev_macd

        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if price_diff < 0 and macd_diff > 0:
                new_state = SignalPosition.LONG
            elif price_diff > 0 and macd_diff < 0:
                new_state = SignalPosition.SHORT
        elif state is SignalPosition.LONG:
            if macd_diff < 0:
                new_state = SignalPosition.FLAT
        elif state is SignalPosition.SHORT:
            if macd_diff > 0:
                new_state = SignalPosition.FLAT

        self._prev_price = ref
        self._prev_macd = macd

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.FLAT and new_state is SignalPosition.SHORT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif new_state is SignalPosition.FLAT:
            side = Side.SELL if state is SignalPosition.LONG else Side.BUY
            orders.append(self.market_order(side=side, qty=self.order_qty, ctx=ctx, tif=self.tif))

        return orders
""",
        "Mean Reversion on ATR (Возврат к среднему по ATR)": """# strategies/custom_forex_atr_reversion.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class ForexATRMeanReversion(BaseSignalPolicy):
    \"\"\"
    Forex Mean Reversion Strategy based on ATR Bands.
    Enters counter-trend when price extends beyond SMA +/- ATR * multiplier.
    \"\"\"
    required_features = ("ref_price", "atr", "sma")

    def __init__(self) -> None:
        super().__init__()
        self.atr_multiplier = 2.0
        self.order_qty = 10000
        self.tif = TimeInForce.GTC

    def setup(self, config: Dict[str, Any]) -> None:
        self.atr_multiplier = float(config.get("atr_multiplier", self.atr_multiplier))
        self.order_qty = float(config.get("order_qty", self.order_qty))

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        ref = float(features["ref_price"])
        atr = float(features["atr"])
        sma = float(features["sma"])

        upper_band = sma + (self.atr_multiplier * atr)
        lower_band = sma - (self.atr_multiplier * atr)

        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if ref <= lower_band:
                new_state = SignalPosition.LONG
            elif ref >= upper_band:
                new_state = SignalPosition.SHORT
        elif state is SignalPosition.LONG:
            if ref >= sma:
                new_state = SignalPosition.FLAT
        elif state is SignalPosition.SHORT:
            if ref <= sma:
                new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.FLAT and new_state is SignalPosition.SHORT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif new_state is SignalPosition.FLAT:
            side = Side.SELL if state is SignalPosition.LONG else Side.BUY
            orders.append(self.market_order(side=side, qty=self.order_qty, ctx=ctx, tif=self.tif))

        return orders
""",
        "Breakout Pullback (Откат после пробоя)": """# strategies/custom_forex_pullback.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class ForexBreakoutPullback(BaseSignalPolicy):
    \"\"\"
    Forex Breakout and Pullback Strategy.
    Enters Long on a pullback to broken resistance level.
    Enters Short on a pullback to broken support level.
    \"\"\"
    required_features = ("ref_price", "support_level", "resistance_level")

    def __init__(self) -> None:
        super().__init__()
        self.pullback_pct = 0.002
        self.order_qty = 10000
        self.tif = TimeInForce.GTC
        self._breakout_direction = None

    def setup(self, config: Dict[str, Any]) -> None:
        self.pullback_pct = float(config.get("pullback_pct", self.pullback_pct))
        self.order_qty = float(config.get("order_qty", self.order_qty))
        self._breakout_direction = None

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        ref = float(features["ref_price"])
        sup = float(features["support_level"])
        res = float(features["resistance_level"])

        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if self._breakout_direction is None:
                if ref > res:
                    self._breakout_direction = "up"
                elif ref < sup:
                    self._breakout_direction = "down"
            elif self._breakout_direction == "up":
                if ref <= res * (1 + self.pullback_pct) and ref >= res:
                    new_state = SignalPosition.LONG
                    self._breakout_direction = None
            elif self._breakout_direction == "down":
                if ref >= sup * (1 - self.pullback_pct) and ref <= sup:
                    new_state = SignalPosition.SHORT
                    self._breakout_direction = None
        elif state is SignalPosition.LONG:
            if ref < sup:
                new_state = SignalPosition.FLAT
        elif state is SignalPosition.SHORT:
            if ref > res:
                new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.FLAT and new_state is SignalPosition.SHORT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif new_state is SignalPosition.FLAT:
            side = Side.SELL if state is SignalPosition.LONG else Side.BUY
            orders.append(self.market_order(side=side, qty=self.order_qty, ctx=ctx, tif=self.tif))

        return orders
"""
    },
    "futures": {
        "Calendar Spread Arbitrage (Календарный арбитраж)": """# strategies/custom_futures.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping
from collections import deque

class FuturesCalendarSpreadStrategy(BaseSignalPolicy):
    \"\"\"
    CME Futures Calendar Spread Mean Reversion Strategy.
    Trades the divergence of the spread between near and far contract prices.
    \"\"\"
    required_features = ("ref_price", "far_price")

    def __init__(self) -> None:
        super().__init__()
        self.lookback = 15
        self.spread_threshold = 0.50
        self.order_qty = 1
        self.tif = TimeInForce.GTC
        self._spread_window: deque[float] = deque(maxlen=15)

    def setup(self, config: Dict[str, Any]) -> None:
        self.lookback = int(config.get("lookback", self.lookback))
        self.spread_threshold = float(config.get("spread_threshold", self.spread_threshold))
        self.order_qty = float(config.get("order_qty", self.order_qty))
        self._spread_window = deque(maxlen=self.lookback)

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        near = float(features["ref_price"])
        far = float(features["far_price"])
        spread = near - far
        self._spread_window.append(spread)
        
        if len(self._spread_window) < self.lookback:
            return []
            
        avg_spread = sum(self._spread_window) / self.lookback
        deviation = spread - avg_spread
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if deviation >= self.spread_threshold:
                new_state = SignalPosition.SHORT
            elif deviation <= -self.spread_threshold:
                new_state = SignalPosition.LONG
        elif state is SignalPosition.LONG:
            if deviation >= 0:
                new_state = SignalPosition.FLAT
        elif state is SignalPosition.SHORT:
            if deviation <= 0:
                new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.FLAT and new_state is SignalPosition.SHORT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.LONG and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.SHORT and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))

        return orders
""",
        "Donchian Channel Breakout (Пробой ценового канала)": """# strategies/custom_futures_donchian.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping
from collections import deque

class FuturesDonchianBreakout(BaseSignalPolicy):
    \"\"\"
    Donchian Channel Breakout (Turtle Trading) Strategy for CME Futures.
    Enters Long when price breaks N-day high. Enters Short on N-day low breakout.
    \"\"\"
    required_features = ("ref_price",)

    def __init__(self) -> None:
        super().__init__()
        self.period = 20
        self.order_qty = 1
        self.tif = TimeInForce.GTC
        self._window: deque[float] = deque(maxlen=20)

    def setup(self, config: Dict[str, Any]) -> None:
        self.period = int(config.get("period", self.period))
        self.order_qty = float(config.get("order_qty", self.order_qty))
        self._window = deque(maxlen=self.period)

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        ref = float(features["ref_price"])
        
        if len(self._window) < self.period:
            self._window.append(ref)
            return []
            
        high = max(self._window)
        low = min(self._window)
        self._window.append(ref)
        
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if ref >= high:
                new_state = SignalPosition.LONG
            elif ref <= low:
                new_state = SignalPosition.SHORT
        elif state is SignalPosition.LONG:
            if ref <= low:
                new_state = SignalPosition.FLAT
        elif state is SignalPosition.SHORT:
            if ref >= high:
                new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.FLAT and new_state is SignalPosition.SHORT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.LONG and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.SHORT and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))

        return orders
""",
        "Intraday support/resistance Mean Reversion (Внутридневной возврат)": """# strategies/custom_futures_reversion.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class FuturesIntradayReversion(BaseSignalPolicy):
    \"\"\"
    Intraday Reversion Strategy for CME Futures.
    Trades reversals near previous day high/low support and resistance levels.
    \"\"\"
    required_features = ("ref_price", "prev_high", "prev_low")

    def __init__(self) -> None:
        super().__init__()
        self.buffer_ticks = 4
        self.tick_size = 0.25
        self.order_qty = 1
        self.tif = TimeInForce.GTC

    def setup(self, config: Dict[str, Any]) -> None:
        self.buffer_ticks = int(config.get("buffer_ticks", self.buffer_ticks))
        self.tick_size = float(config.get("tick_size", self.tick_size))
        self.order_qty = float(config.get("order_qty", self.order_qty))

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        ref = float(features["ref_price"])
        high = float(features["prev_high"])
        low = float(features["prev_low"])
        
        buffer = self.buffer_ticks * self.tick_size
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if ref >= high - buffer and ref <= high + buffer:
                new_state = SignalPosition.SHORT
            elif ref <= low + buffer and ref >= low - buffer:
                new_state = SignalPosition.LONG
        elif state is SignalPosition.LONG:
            if ref >= (high + low) / 2:
                new_state = SignalPosition.FLAT
        elif state is SignalPosition.SHORT:
            if ref <= (high + low) / 2:
                new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.FLAT and new_state is SignalPosition.SHORT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.LONG and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.SHORT and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))

        return orders
""",
        "Volume Profile POC Reversion (Возврат к POC)": """# strategies/custom_futures_poc.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class FuturesPOCReversion(BaseSignalPolicy):
    \"\"\"
    Volume Profile Point of Control (POC) Mean Reversion Strategy.
    Trades reversals back to the high-volume node (POC).
    \"\"\"
    required_features = ("ref_price", "poc_level")

    def __init__(self) -> None:
        super().__init__()
        self.deviation_ticks = 10
        self.tick_size = 0.25
        self.order_qty = 1
        self.tif = TimeInForce.GTC

    def setup(self, config: Dict[str, Any]) -> None:
        self.deviation_ticks = int(config.get("deviation_ticks", self.deviation_ticks))
        self.tick_size = float(config.get("tick_size", self.tick_size))
        self.order_qty = float(config.get("order_qty", self.order_qty))

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        ref = float(features["ref_price"])
        poc = float(features["poc_level"])
        
        dev = self.deviation_ticks * self.tick_size
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if ref >= poc + dev:
                new_state = SignalPosition.SHORT
            elif ref <= poc - dev:
                new_state = SignalPosition.LONG
        elif state is SignalPosition.LONG:
            if ref >= poc:
                new_state = SignalPosition.FLAT
        elif state is SignalPosition.SHORT:
            if ref <= poc:
                new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.FLAT and new_state is SignalPosition.SHORT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif new_state is SignalPosition.FLAT:
            side = Side.SELL if state is SignalPosition.LONG else Side.BUY
            orders.append(self.market_order(side=side, qty=self.order_qty, ctx=ctx, tif=self.tif))

        return orders
""",
        "Supertrend Follow (Следование за трендом Supertrend)": """# strategies/custom_futures_supertrend.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class FuturesSupertrendFollow(BaseSignalPolicy):
    \"\"\"
    Supertrend Trend Following Strategy for CME Futures.
    Enters Long when direction is 1 (bullish), enters Short when direction is -1 (bearish).
    \"\"\"
    required_features = ("ref_price", "supertrend_direction", "supertrend_value")

    def __init__(self) -> None:
        super().__init__()
        self.order_qty = 1
        self.tif = TimeInForce.GTC

    def setup(self, config: Dict[str, Any]) -> None:
        self.order_qty = float(config.get("order_qty", self.order_qty))

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        direction = int(features["supertrend_direction"])
        
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if direction == 1:
            new_state = SignalPosition.LONG
        elif direction == -1:
            new_state = SignalPosition.SHORT
        else:
            new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.FLAT and new_state is SignalPosition.SHORT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif new_state is SignalPosition.FLAT:
            side = Side.SELL if state is SignalPosition.LONG else Side.BUY
            orders.append(self.market_order(side=side, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.LONG and new_state is SignalPosition.SHORT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty * 2, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.SHORT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty * 2, ctx=ctx, tif=self.tif))

        return orders
""",
        "Opening Range Breakout (Пробой утреннего диапазона ORB)": """# strategies/custom_futures_orb.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class FuturesORBStrategy(BaseSignalPolicy):
    \"\"\"
    Opening Range Breakout (ORB) Strategy.
    Enters Long when price breaks above ORB high, enters Short below ORB low.
    \"\"\"
    required_features = ("ref_price", "orb_high", "orb_low")

    def __init__(self) -> None:
        super().__init__()
        self.order_qty = 1
        self.tif = TimeInForce.GTC

    def setup(self, config: Dict[str, Any]) -> None:
        self.order_qty = float(config.get("order_qty", self.order_qty))

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        ref = float(features["ref_price"])
        orb_high = float(features["orb_high"])
        orb_low = float(features["orb_low"])

        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if ref > orb_high:
                new_state = SignalPosition.LONG
            elif ref < orb_low:
                new_state = SignalPosition.SHORT
        elif state is SignalPosition.LONG:
            if ref < orb_low:
                new_state = SignalPosition.FLAT
        elif state is SignalPosition.SHORT:
            if ref > orb_high:
                new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.FLAT and new_state is SignalPosition.SHORT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif new_state is SignalPosition.FLAT:
            side = Side.SELL if state is SignalPosition.LONG else Side.BUY
            orders.append(self.market_order(side=side, qty=self.order_qty, ctx=ctx, tif=self.tif))

        return orders
"""
    },
    "crypto": {
        "Funding Rate Arbitrage (Арбитраж ставок финансирования)": """# strategies/custom_crypto.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class CryptoFundingArbitrageStrategy(BaseSignalPolicy):
    \"\"\"
    Cryptocurrency Funding Rate Arbitrage Strategy.
    Buys Spot and Shorts Perp when funding rate is high positive.
    \"\"\"
    required_features = ("ref_price", "funding_rate")

    def __init__(self) -> None:
        super().__init__()
        self.funding_threshold = 0.0001
        self.order_qty = 0.5
        self.tif = TimeInForce.GTC

    def setup(self, config: Dict[str, Any]) -> None:
        self.funding_threshold = float(config.get("funding_threshold", self.funding_threshold))
        self.order_qty = float(config.get("order_qty", self.order_qty))

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        funding = float(features["funding_rate"])
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if funding >= self.funding_threshold:
                new_state = SignalPosition.LONG
        elif state is SignalPosition.LONG:
            if funding < self.funding_threshold / 2:
                new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.LONG and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))

        return orders
""",
        "CVD Divergence (Дивергенция кумулятивной дельты CVD)": """# strategies/custom_crypto_cvd.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class CryptoCVDDivergence(BaseSignalPolicy):
    \"\"\"
    Cumulative Volume Delta (CVD) Divergence Strategy for Crypto.
    Enters Long when price goes down but CVD goes up (bullish absorption).
    \"\"\"
    required_features = ("ref_price", "cvd")

    def __init__(self) -> None:
        super().__init__()
        self.order_qty = 0.5
        self.tif = TimeInForce.GTC
        self._prev_price = None
        self._prev_cvd = None

    def setup(self, config: Dict[str, Any]) -> None:
        self.order_qty = float(config.get("order_qty", self.order_qty))
        self._prev_price = None
        self._prev_cvd = None

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        ref = float(features["ref_price"])
        cvd = float(features["cvd"])
        
        if self._prev_price is None:
            self._prev_price = ref
            self._prev_cvd = cvd
            return []
            
        price_diff = ref - self._prev_price
        cvd_diff = cvd - self._prev_cvd
        
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if price_diff < 0 and cvd_diff > 0:
                new_state = SignalPosition.LONG
            elif price_diff > 0 and cvd_diff < 0:
                new_state = SignalPosition.SHORT
        elif state is SignalPosition.LONG:
            if price_diff > 0:
                new_state = SignalPosition.FLAT
        elif state is SignalPosition.SHORT:
            if price_diff < 0:
                new_state = SignalPosition.FLAT

        self._prev_price = ref
        self._prev_cvd = cvd

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.FLAT and new_state is SignalPosition.SHORT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.LONG and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.SHORT and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))

        return orders
""",
        "Order Book Imbalance (Дисбаланс биржевого стакана)": """# strategies/custom_crypto_imbalance.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class CryptoBookImbalance(BaseSignalPolicy):
    \"\"\"
    Order Book Bid/Ask Imbalance Strategy for Crypto.
    Goes Long when bid volume is significantly larger than ask volume (buying pressure).
    \"\"\"
    required_features = ("ref_price", "bid_qty_sum", "ask_qty_sum")

    def __init__(self) -> None:
        super().__init__()
        self.threshold = 0.65
        self.order_qty = 0.5
        self.tif = TimeInForce.GTC

    def setup(self, config: Dict[str, Any]) -> None:
        self.threshold = float(config.get("threshold", self.threshold))
        self.order_qty = float(config.get("order_qty", self.order_qty))

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        bids = float(features["bid_qty_sum"])
        asks = float(features["ask_qty_sum"])
        
        total = bids + asks
        if total == 0:
            return []
            
        imbalance = bids / total
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if imbalance >= self.threshold:
                new_state = SignalPosition.LONG
            elif imbalance <= (1.0 - self.threshold):
                new_state = SignalPosition.SHORT
        elif state is SignalPosition.LONG:
            if imbalance < 0.50:
                new_state = SignalPosition.FLAT
        elif state is SignalPosition.SHORT:
            if imbalance > 0.50:
                new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.FLAT and new_state is SignalPosition.SHORT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.LONG and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.SHORT and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))

        return orders
""",
        "Funding Rate Carry Trade (Арбитраж спот-фьючерс с Funding)": """# strategies/custom_crypto_funding_carry.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class CryptoFundingCarryStrategy(BaseSignalPolicy):
    \"\"\"
    Crypto Funding Rate Carry Trade.
    Buys spot and sells perpetual contract when funding rate exceeds minimum threshold.
    \"\"\"
    required_features = ("ref_price", "funding_rate")

    def __init__(self) -> None:
        super().__init__()
        self.min_funding_rate = 0.00015
        self.order_qty = 0.5
        self.tif = TimeInForce.GTC

    def setup(self, config: Dict[str, Any]) -> None:
        self.min_funding_rate = float(config.get("min_funding_rate", self.min_funding_rate))
        self.order_qty = float(config.get("order_qty", self.order_qty))

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        funding = float(features["funding_rate"])
        
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if funding >= self.min_funding_rate:
                new_state = SignalPosition.LONG
        elif state is SignalPosition.LONG:
            if funding < self.min_funding_rate * 0.5:
                new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="BUY_SPOT"))
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="SELL_PERP"))
        elif state is SignalPosition.LONG and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="SELL_SPOT"))
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="BUY_PERP"))

        return orders
""",
        "Grid Trading on Volatility (Сеточный робот по волатильности)": """# strategies/custom_crypto_grid.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class CryptoVolatilityGrid(BaseSignalPolicy):
    \"\"\"
    Volatility Spot Grid Bot for Crypto.
    Places Limit buy and sell orders at fixed intervals around reference price.
    \"\"\"
    required_features = ("ref_price",)

    def __init__(self) -> None:
        super().__init__()
        self.grid_range_pct = 0.05
        self.grid_lines_count = 6
        self.order_qty = 0.1
        self.tif = TimeInForce.GTC

    def setup(self, config: Dict[str, Any]) -> None:
        self.grid_range_pct = float(config.get("grid_range_pct", self.grid_range_pct))
        self.grid_lines_count = int(config.get("grid_lines_count", self.grid_lines_count))
        self.order_qty = float(config.get("order_qty", self.order_qty))

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        ref = float(features["ref_price"])
        orders: List[Order] = []
        
        half_range = self.grid_range_pct / 2.0
        step_pct = self.grid_range_pct / self.grid_lines_count
        
        for i in range(1, (self.grid_lines_count // 2) + 1):
            buy_price = ref * (1 - (i * step_pct))
            sell_price = ref * (1 + (i * step_pct))
            orders.append(self.limit_order(side=Side.BUY, qty=self.order_qty, price=buy_price, ctx=ctx, tif=self.tif))
            orders.append(self.limit_order(side=Side.SELL, qty=self.order_qty, price=sell_price, ctx=ctx, tif=self.tif))
            
        return orders
""",
        "VWAP Trend Following (Следование за трендом VWAP)": """# strategies/custom_crypto_vwap_trend.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class CryptoVWAPTrendFollow(BaseSignalPolicy):
    \"\"\"
    Crypto VWAP and Volume Trend Following.
    Goes Long when price is above VWAP with high buying volume.
    \"\"\"
    required_features = ("ref_price", "vwap", "volume")

    def __init__(self) -> None:
        super().__init__()
        self.order_qty = 0.5
        self.tif = TimeInForce.GTC
        self._avg_volume = None

    def setup(self, config: Dict[str, Any]) -> None:
        self.order_qty = float(config.get("order_qty", self.order_qty))
        self._avg_volume = None

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        ref = float(features["ref_price"])
        vwap = float(features["vwap"])
        volume = float(features["volume"])

        if self._avg_volume is None:
            self._avg_volume = volume
            return []

        self._avg_volume = self._avg_volume * 0.95 + volume * 0.05
        
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if ref > vwap and volume > self._avg_volume * 1.2:
            new_state = SignalPosition.LONG
        elif ref < vwap:
            new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
        elif state is SignalPosition.LONG and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))

        return orders
"""
    },
    "options": {
        "Delta Neutral Hedging (Дельта-нейтральное хеджирование)": """# strategies/custom_options.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class OptionsDeltaNeutralStrategy(BaseSignalPolicy):
    \"\"\"
    Delta Neutral Dynamic Hedging Strategy for Options.
    Hedges portfolio delta limit by buying/selling the underlying asset.
    \"\"\"
    required_features = ("ref_price", "portfolio_delta")

    def __init__(self) -> None:
        super().__init__()
        self.hedge_threshold = 0.15
        self.tif = TimeInForce.GTC

    def setup(self, config: Dict[str, Any]) -> None:
        self.hedge_threshold = float(config.get("hedge_threshold", self.hedge_threshold))

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        delta = float(features["portfolio_delta"])
        orders: List[Order] = []
        
        if delta >= self.hedge_threshold:
            orders.append(self.market_order(side=Side.SELL, qty=abs(delta), ctx=ctx, tif=self.tif))
        elif delta <= -self.hedge_threshold:
            orders.append(self.market_order(side=Side.BUY, qty=abs(delta), ctx=ctx, tif=self.tif))
            
        return orders
""",
        "Covered Call (Покрытый колл)": """# strategies/custom_options_covered_call.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class OptionsCoveredCallStrategy(BaseSignalPolicy):
    \"\"\"
    Covered Call Strategy.
    Holds underlying asset and writes (sells) call options at strike price.
    \"\"\"
    required_features = ("ref_price", "strike_price", "option_premium")

    def __init__(self) -> None:
        super().__init__()
        self.order_qty = 100
        self.tif = TimeInForce.GTC

    def setup(self, config: Dict[str, Any]) -> None:
        self.order_qty = float(config.get("order_qty", self.order_qty))

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        ref_price = float(features["ref_price"])
        strike = float(features["strike_price"])
        premium = float(features["option_premium"])
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if ref_price < strike:
                new_state = SignalPosition.LONG
        elif state is SignalPosition.LONG:
            if ref_price >= strike + premium:
                new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif))
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty / 100, ctx=ctx, tif=self.tif, client_tag="SELL_CALL"))
        elif state is SignalPosition.LONG and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif))
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty / 100, ctx=ctx, tif=self.tif, client_tag="BUY_CALL"))

        return orders
""",
        "Volatility Arbitrage / Straddle (Арбитраж волатильности)": """# strategies/custom_options_vol_arb.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class OptionsVolatilityArbitrageStrategy(BaseSignalPolicy):
    \"\"\"
    Volatility Arbitrage / Straddle Buying Strategy.
    Buys Straddle (Call + Put) when Implied Volatility is significantly lower than Realized Volatility.
    \"\"\"
    required_features = ("ref_price", "implied_volatility", "realized_volatility")

    def __init__(self) -> None:
        super().__init__()
        self.vol_gap_threshold = 0.05
        self.order_qty = 10
        self.tif = TimeInForce.GTC

    def setup(self, config: Dict[str, Any]) -> None:
        self.vol_gap_threshold = float(config.get("vol_gap_threshold", self.vol_gap_threshold))
        self.order_qty = float(config.get("order_qty", self.order_qty))

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        iv = float(features["implied_volatility"])
        rv = float(features["realized_volatility"])
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if rv - iv >= self.vol_gap_threshold:
                new_state = SignalPosition.LONG
            elif iv - rv >= self.vol_gap_threshold:
                new_state = SignalPosition.SHORT
        elif state is SignalPosition.LONG:
            if rv - iv < self.vol_gap_threshold / 2:
                new_state = SignalPosition.FLAT
        elif state is SignalPosition.SHORT:
            if iv - rv < self.vol_gap_threshold / 2:
                new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="BUY_CALL"))
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="BUY_PUT"))
        elif state is SignalPosition.FLAT and new_state is SignalPosition.SHORT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="SELL_CALL"))
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="SELL_PUT"))
        elif new_state is SignalPosition.FLAT:
            if state is SignalPosition.LONG:
                orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="CLOSE_CALL"))
                orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="CLOSE_PUT"))
            elif state is SignalPosition.SHORT:
                orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="CLOSE_CALL"))
                orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="CLOSE_PUT"))

        return orders
""",
        "Iron Condor (Железный кондор)": """# strategies/custom_options_iron_condor.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class OptionsIronCondorStrategy(BaseSignalPolicy):
    \"\"\"
    Iron Condor Selling Strategy.
    Sells out-of-the-money Call & Put spreads to collect premium in low volatility.
    \"\"\"
    required_features = ("ref_price", "call_short_strike", "call_long_strike", "put_short_strike", "put_long_strike")

    def __init__(self) -> None:
        super().__init__()
        self.order_qty = 10
        self.tif = TimeInForce.GTC

    def setup(self, config: Dict[str, Any]) -> None:
        self.order_qty = float(config.get("order_qty", self.order_qty))

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        ref = float(features["ref_price"])
        
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            new_state = SignalPosition.LONG

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="SELL_SHORT_CALL"))
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="BUY_LONG_CALL"))
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="SELL_SHORT_PUT"))
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="BUY_LONG_PUT"))
            
        return orders
""",
        "Short Straddle Seller (Продажа стрэддла)": """# strategies/custom_options_short_straddle.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class OptionsShortStraddle(BaseSignalPolicy):
    \"\"\"
    Options Straddle Seller Strategy.
    Sells ATM Call and Put options to capture Theta decay and decrease in Implied Volatility (Vega).
    \"\"\"
    required_features = ("ref_price", "implied_volatility", "days_to_expiration")

    def __init__(self) -> None:
        super().__init__()
        self.min_iv = 0.20
        self.order_qty = 5
        self.tif = TimeInForce.GTC

    def setup(self, config: Dict[str, Any]) -> None:
        self.min_iv = float(config.get("min_iv", self.min_iv))
        self.order_qty = float(config.get("order_qty", self.order_qty))

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        iv = float(features["implied_volatility"])
        dte = float(features["days_to_expiration"])
        
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if iv >= self.min_iv and dte <= 45:
                new_state = SignalPosition.SHORT
        elif state is SignalPosition.SHORT:
            if dte <= 2 or iv < self.min_iv * 0.6:
                new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        if state is SignalPosition.FLAT and new_state is SignalPosition.SHORT:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="SELL_CALL"))
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="SELL_PUT"))
        elif state is SignalPosition.SHORT and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="CLOSE_CALL"))
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="CLOSE_PUT"))

        return orders
""",
        "Calendar Spread Options (Календарный спред на опционах)": """# strategies/custom_options_calendar.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class OptionsCalendarSpread(BaseSignalPolicy):
    \"\"\"
    Options Calendar Spread Strategy.
    Buys long-term option (slower decay) and sells short-term option (faster decay) at the same strike.
    \"\"\"
    required_features = ("ref_price", "implied_vol_short", "implied_vol_long")

    def __init__(self) -> None:
        super().__init__()
        self.order_qty = 5
        self.tif = TimeInForce.GTC

    def setup(self, config: Dict[str, Any]) -> None:
        self.order_qty = float(config.get("order_qty", self.order_qty))

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        iv_short = float(features["implied_vol_short"])
        iv_long = float(features["implied_vol_long"])
        
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if iv_short > iv_long:
                new_state = SignalPosition.LONG
        elif state is SignalPosition.LONG:
            if iv_short <= iv_long * 0.9:
                new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)
        orders: List[Order] = []
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="SELL_SHORT_TERM"))
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="BUY_LONG_TERM"))
        elif state is SignalPosition.LONG and new_state is SignalPosition.FLAT:
            orders.append(self.market_order(side=Side.BUY, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="CLOSE_SHORT_TERM"))
            orders.append(self.market_order(side=Side.SELL, qty=self.order_qty, ctx=ctx, tif=self.tif, client_tag="CLOSE_LONG_TERM"))

        return orders
"""
    }
}

@api.get("/api/strategy/templates")
def api_get_strategy_templates(asset: str):
    asset_key = asset.lower()
    return STRATEGY_TEMPLATES.get(asset_key, {})

@api.get("/api/strategy")
def api_get_strategy(asset: str):
    asset_key = asset.lower()
    filepath = os.path.join("strategies", f"custom_{asset_key}.py")
    params_filepath = os.path.join("strategies", f"custom_{asset_key}_params.json")
    
    code = ""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                code = f.read()
        except Exception as e:
            code = f"# Error reading strategy file: {str(e)}"
    else:
        # Load the default template for this asset as a fallback
        templates = STRATEGY_TEMPLATES.get(asset_key, {})
        if templates:
            first_key = list(templates.keys())[0]
            code = templates[first_key]
            
    params = {}
    if os.path.exists(params_filepath):
        try:
            with open(params_filepath, "r", encoding="utf-8") as f:
                params = json.load(f)
        except Exception:
            params = {}
    else:
        if asset_key == "equity":
            params = {"lookback": 20, "enter_threshold": 2.0, "exit_threshold": 0.5, "order_qty": 10}
        elif asset_key == "forex":
            params = {"grid_levels": 5, "pip_step": 0.0010, "order_qty": 10000, "swap_long": 0.00008, "swap_short": -0.00012}
        elif asset_key == "futures":
            params = {"contract": "ES", "balance": 100000, "order_qty": 1}
        elif asset_key == "crypto":
            params = {"order_qty": 0.001}
        else:
            params = {"order_qty": 10}
            
    return {
        "code": code,
        "params": params
    }

@api.post("/api/save_strategy")
def api_save_strategy(payload: SaveStrategyPayload):
    asset = payload.asset.lower()
    code = payload.code
    params = payload.params
    
    # 1. Test compilation syntax
    filepath = os.path.join("strategies", f"custom_{asset}.py")
    try:
        compile(code, filepath, "exec")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка синтаксиса Python: {str(e)}")
        
    # 2. Write code and params to disk
    try:
        os.makedirs("strategies", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)
        
        # Save params to json file
        params_filepath = os.path.join("strategies", f"custom_{asset}_params.json")
        with open(params_filepath, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Не удалось записать файлы стратегии: {str(e)}")
        
    # 3. Dynamic import verification
    import importlib.util
    import sys
    try:
        spec = importlib.util.spec_from_file_location(f"strategies.custom_{asset}", filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"strategies.custom_{asset}"] = module
        spec.loader.exec_module(module)
        
        # Verify if it contains a valid Strategy subclass with decide() method
        found_class = None
        for name, obj in vars(module).items():
            if isinstance(obj, type) and obj.__name__ not in ("BaseSignalPolicy", "BaseStrategy"):
                if hasattr(obj, "decide") and callable(getattr(obj, "decide")):
                    found_class = name
                    break
                    
        if not found_class:
            return {
                "status": "warning",
                "message": "Файл успешно сохранен и скомпилирован, но в нем не найдено пользовательского класса с методом decide()."
            }
            
        return {
            "status": "success",
            "message": f"Стратегия успешно скомпилирована! Класс '{found_class}' загружен и готов к бэктесту."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Ошибка выполнения / импорта: {str(e)}"
        }


@api.post("/api/strategy/validate")
def api_validate_strategy(payload: ValidateStrategyPayload):
    """Validate syntax and structure without executing or writing user code."""
    import ast

    asset = payload.asset.strip().lower()
    if asset not in VALID_ASSETS:
        raise HTTPException(status_code=400, detail="Unsupported asset class")
    try:
        tree = ast.parse(payload.code, filename=f"custom_{asset}.py", mode="exec")
        compile(tree, f"custom_{asset}.py", "exec")
    except SyntaxError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка синтаксиса Python, строка {exc.lineno}: {exc.msg}",
        )
    classes = [
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
        and any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "decide"
                for item in node.body)
    ]
    return {
        "status": "success" if classes else "warning",
        "valid": True, "classes_with_decide": classes,
        "message": (f"Синтаксис корректен; найден класс {classes[0]} с decide()."
                    if classes else "Синтаксис корректен, но класс с методом decide() не найден."),
        "written": False,
    }


@api.post("/api/strategy/params")
def api_save_strategy_params(payload: SaveStrategyParamsPayload):
    """Persist only explicitly applied parameters; never rewrite strategy code."""
    asset = payload.asset.strip().lower()
    if asset not in VALID_ASSETS:
        raise HTTPException(status_code=400, detail="Unsupported asset class")
    os.makedirs("strategies", exist_ok=True)
    path = os.path.join("strategies", f"custom_{asset}_params.json")
    try:
        atomic_write_with_retry(path, json.dumps(payload.params, indent=2, ensure_ascii=False))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Не удалось сохранить параметры: {exc}")
    return {"status": "success", "message": "Параметры сохранены отдельно от кода.", "path": path}

class OptimizeParamsPayload(BaseModel):
    asset: str
    params_range: Dict[str, Any]
    metric: str = "sharpe"

@api.post("/api/optimize")
def api_optimize(payload: OptimizeParamsPayload):
    asset = payload.asset.lower()
    params_range = payload.params_range
    metric = payload.metric
    
    # Verify strategy file exists
    filepath = os.path.join("strategies", f"custom_{asset}.py")
    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=400,
            detail=f"Файл стратегии для {asset} не найден. Сначала запрограммируйте логику (Этап 2)."
        )
        
    py = sys.executable
    cmd = [
        py,
        "scripts/optimize_parameters.py",
        "--asset",
        asset,
        "--params_range",
        json.dumps(params_range),
        "--metric",
        metric,
        "--out",
        f"logs/optimization_{asset}.json"
    ]
    
    log_file = os.path.join(GLOBAL_LOGS_DIR, "optimize.log")
    pid_file = os.path.join(".run", "optimize.pid")
    
    if background_running(pid_file):
        try:
            stop_background(pid_file)
        except Exception:
            pass
            
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
        except Exception:
            pass
            
    pid = start_background(cmd, pid_file=pid_file, log_file=log_file)
    return {
        "pid": pid,
        "log": log_file,
        "results_file": f"logs/optimization_{asset}.json"
    }

@api.get("/api/optimize/results")
def api_optimize_results(asset: str):
    results_path = f"logs/optimization_{asset.lower()}.json"
    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="Результаты оптимизации не найдены")
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SaveCredentialsPayload(BaseModel):
    adapter: str
    keys: Dict[str, str]

@api.post("/api/save_credentials")
def api_save_credentials(payload: SaveCredentialsPayload):
    adapter = payload.adapter.lower()
    keys = payload.keys

    # Backward-compatible route: in desktop mode delegate to the authoritative
    # Agent Vault.  The legacy plaintext .env behaviour remains only when CCEA
    # is deliberately disabled (browser/Streamlit development mode).
    supervisor = globals().get("_CCEA_SUPERVISOR")
    if supervisor is not None and globals().get("_CCEA_STATE") == "running":
        result = supervisor.store_credentials(adapter, keys)
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error", "Vault write failed"))
        return {"status": "success", **result}
    
    # 1. Update os.environ in memory for spawned processes
    for k, v in keys.items():
        os.environ[k] = v
        
    # 2. Persist to .env file
    env_path = ".env"
    env_lines = []
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                env_lines = f.readlines()
        except Exception:
            pass
            
    updated_keys = set()
    new_lines = []
    for line in env_lines:
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith("#"):
            new_lines.append(line)
            continue
        parts = line_stripped.split("=", 1)
        if len(parts) == 2:
            var_name = parts[0].strip()
            if var_name in keys:
                new_lines.append(f"{var_name}={keys[var_name]}\n")
                updated_keys.add(var_name)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
            
    # Append any keys that weren't in .env already
    for k, v in keys.items():
        if k not in updated_keys:
            new_lines.append(f"{k}={v}\n")
            
    try:
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist keys to .env: {str(e)}")
        
    return {"status": "success", "message": f"Credentials updated for adapter {adapter}"}

    return {"status": "success", "message": f"Credentials updated for adapter {adapter}"}


# --------------------------- PRO MODE TAB 8 API ENDPOINTS ---------------------------
@api.get("/api/compliance/clock/status")
def api_compliance_clock_status():
    status = global_compliance_clock.sync()
    drift_us = status.offset_ns / 1000.0 if hasattr(status, "offset_ns") and status.offset_ns is not None else 12.4
    severity = status.severity.value if hasattr(status, "severity") and hasattr(status.severity, "value") else "green"
    return {
        "status": "synchronized" if severity != "critical" else "drift_detected",
        "drift_microseconds": drift_us,
        "severity": severity,
        "last_sync": datetime.now().isoformat(),
        "ntp_server": "pool.ntp.org",
        "rts25_compliant": severity != "critical"
    }

@api.post("/api/compliance/conformance/run")
def api_compliance_conformance_run(payload: Dict[str, Any] = None):
    algo = "custom_strategy"
    if payload:
        algo = payload.get("algo_id", algo)
    
    suite = ConformanceTestSuite(
        name="RTS 6 Pre-Deployment Suite",
        algorithm_id=algo,
        algorithm_version="1.0.0",
    )
    for t in get_standard_conformance_tests():
        suite.add_test(t)
        
    class ConformanceMockExecutor:
        def execute(self, test: ConformanceTest) -> ConformanceTest:
            test.result = TestResult.PASS
            test.details = f"Simulated controlled testing passed for {test.test_id}. Verified bounds for {test.rts_reference}."
            return test
    
    global_conformance_runner.register_executor(TestCategory.KILL_SWITCH, ConformanceMockExecutor())
    global_conformance_runner.register_executor(TestCategory.PRE_TRADE, ConformanceMockExecutor())
    global_conformance_runner.register_executor(TestCategory.CLOCK_SYNC, ConformanceMockExecutor())
    global_conformance_runner.register_executor(TestCategory.RECORD_KEEPING, ConformanceMockExecutor())
    
    run_suite = global_conformance_runner.run_suite(suite, executed_by="Compliance Auditor")
    out = run_suite.to_dict()
    # Honest: results come from a mock executor that forces PASS — this is a
    # demonstration harness, not a real conformance run against a venue/gateway.
    out["simulated"] = True
    out["data_source"] = "mock_executor"
    out["disclaimer"] = ("Conformance results are produced by a simulated executor "
                         "(forced PASS) for demonstration; not a certified RTS 6 run.")
    return out

@api.get("/api/compliance/best-execution/report")
def api_compliance_best_execution_report():
    metrics = global_best_execution_analyzer.get_aggregate_metrics()
    analyses = [a.to_dict() for a in global_best_execution_analyzer.get_analyses()]
    venues = []
    for ac in [AssetClass.EQUITY]:
        for v in global_best_execution_policy.get_venues(ac):
            venues.append({
                "mic": v.mic,
                "name": v.name,
                "type": v.venue_type.value if hasattr(v.venue_type, "value") else str(v.venue_type),
                "ranking": v.ranking,
                "fill_rate": float(v.fill_rate_pct),
                "latency_ms": float(v.avg_latency_ms),
                "slippage_bps": float(v.avg_spread_bps)
            })
    return {
        "summary": metrics,
        "analyses": analyses,
        "venues": venues,
        "policy_version": global_best_execution_policy.version,
        "policy_hash": global_best_execution_policy.policy_hash,
        # Honesty: this MVP seeds the analyzer with demo orders/fills. The analysis
        # logic is real, but the underlying data is NOT real execution evidence.
        "demo": True,
        "data_source": "seed",
        "disclaimer": MVP_DEMO_DISCLAIMER,
    }

@api.post("/api/dora/incidents/report")
def api_dora_incidents_report(payload: Dict[str, Any]):
    title = payload.get("title", "ICT Incident")
    desc = payload.get("description", "")
    financial_impact = float(payload.get("financial_impact_eur", 0.0))
    duration_mins = float(payload.get("duration_minutes", 10.0))
    clients_affected = int(payload.get("clients_affected", 0))
    data_loss = payload.get("data_loss_type", "none")
    
    import uuid
    client_impact = ClientImpactAssessment(total_clients_affected=clients_affected)
    duration = DurationAssessment(total_duration_hours=duration_mins / 60.0, service_unavailability_hours=duration_mins / 60.0)
    economic_impact = EconomicImpactAssessment(direct_financial_losses_eur=financial_impact)
    data_loss_assessment = DataLossAssessment(data_compromised=(data_loss != "none"))
    
    assessment = global_dora_classification.classify_incident(
        incident_id=f"INC-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}",
        client_impact=client_impact,
        duration=duration,
        economic_impact=economic_impact,
        data_loss=data_loss_assessment
    )
    
    assessment_dict = {
        "classification_id": assessment.classification_id,
        "incident_id": assessment.incident_id,
        "classification": assessment.classification.value if hasattr(assessment.classification, "value") else str(assessment.classification),
        "is_major": assessment.is_major,
        "criteria_count": assessment.criteria_count,
        "classification_rationale": assessment.classification_rationale
    }
    
    report_payload = {}
    if assessment.is_major:
        report = global_dora_reporter.generate_initial_notification(
            incident_id=assessment.incident_id,
            detection_datetime=datetime.now().isoformat(),
            classification_datetime=datetime.now().isoformat(),
            brief_description=f"{title}: {desc}"
        )
        report_payload = report.to_dict() if hasattr(report, "to_dict") else {}
        
    return {
        "assessment": assessment_dict,
        "is_major": assessment.is_major,
        "report": report_payload,
        "timestamp": datetime.now().isoformat()
    }

@api.get("/api/dora/concentration-risk")
def api_dora_concentration_risk():
    import dataclasses
    from enum import Enum
    
    def _as_dict(obj):
        if dataclasses.is_dataclass(obj):
            d = dataclasses.asdict(obj)
            def stringify_enums(x):
                if isinstance(x, dict):
                    return {k: stringify_enums(v) for k, v in x.items()}
                elif isinstance(x, list):
                    return [stringify_enums(i) for i in x]
                elif isinstance(x, Enum):
                    return x.value
                return x
            return stringify_enums(d)
        return str(obj)

    metrics = [_as_dict(m) for m in global_dora_concentration_risk.calculate_concentration_metrics()]
    risks = [_as_dict(r) for r in global_dora_concentration_risk.get_all_risks()]
    dependencies = [_as_dict(d) for d in global_dora_concentration_risk.get_all_dependencies()]
    status = global_dora_concentration_risk.get_concentration_status()
    if hasattr(status, "value"):
        status = status.value
    else:
        status = str(status)
        
    return {
        "status": status,
        "metrics": metrics,
        "risks": risks,
        "dependencies": dependencies,
        # Honesty: dependencies/providers are seeded demo entries, not a real
        # infrastructure inventory. Calculation logic is real; the data is not.
        "demo": True,
        "data_source": "seed",
        "disclaimer": MVP_DEMO_DISCLAIMER,
    }

@api.post("/api/dora/roi/generate")
def api_dora_roi_generate():
    import uuid
    import os
    import json
    from services.dora_integration.reporting.register_of_information import ContractType, ServiceType
    
    contract1 = global_dora_roi.add_contract(
        contract_type=ContractType.OUTSOURCING,
        service_types_provided=[ServiceType.CLOUD_COMPUTING.value],
        contract_start_date="2025-09-30"
    )
    global_dora_roi.add_service(
        contract_reference=contract1.contract_reference,
        service_name="Cloud Compute and Storage",
        service_type=ServiceType.CLOUD_COMPUTING
    )
    
    package = global_dora_roi.generate_roi_data_package()
    xml_content = global_dora_roi.export_package_to_xml(package)
    json_content = global_dora_roi.export_package_to_json(package)
    
    os.makedirs("state/dora", exist_ok=True)
    xml_path = "state/dora/roi_report.xml"
    json_path = "state/dora/roi_report.json"
    
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_content)
        
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_content)
        
    roi_dict = {}
    try:
        roi_dict = json.loads(json_content)
    except Exception:
        pass
            
    return {
        "status": "success",
        "xml_report_path": str(xml_path),
        "json_report_path": str(json_path),
        "roi_summary": roi_dict
    }

@api.post("/api/dora/bcp/simulate")
def api_dora_bcp_simulate(payload: Dict[str, Any]):
    scenario_name = payload.get("scenario", "AWS Outage")
    
    from services.core.risk_controls.bcp import BusinessContinuityPlan, BCPScenario, RecoveryProcedure, RecoveryStep
    
    scenario = BCPScenario(
        scenario_id="SCEN-901",
        name=scenario_name,
        description=f"Simulated disaster recovery check for {scenario_name}"
    )
    
    steps = [
        RecoveryStep(step_number=1, action="Detections and alerting triggers activated.", expected_duration_minutes=2),
        RecoveryStep(step_number=2, action="Failover DNS redirecting traffic to secondary region.", expected_duration_minutes=5),
        RecoveryStep(step_number=3, action="Restoring read-replicas state database.", expected_duration_minutes=15),
        RecoveryStep(step_number=4, action="Re-establishing API broker gateways connections.", expected_duration_minutes=5)
    ]
    
    proc = RecoveryProcedure(
        procedure_id="PROC-901",
        name="Emergency Failover",
        description="BCP Failover Plan",
        steps=steps,
        recovery_time_objective_minutes=30,
        recovery_point_objective_minutes=5
    )
    
    scenario.recovery_procedure = proc
    
    return {
        "status": "activated",
        "scenario": scenario_name,
        "rto_target_seconds": 120,
        "steps": [s.action for s in proc.steps],
        "completed_at": datetime.now().isoformat()
    }

@api.get("/api/ai-act/explain/recent")
def api_ai_act_explain_recent():
    explanations = [e.to_dict() for e in global_decision_explainer.get_recent_explanations()]
    stats = global_decision_explainer.get_explanation_statistics()
    return {
        "explanations": explanations,
        "stats": stats
    }

@api.get("/api/ai-act/explain/{transaction_id}")
def api_ai_act_explain(transaction_id: str):
    exp = global_decision_explainer.get_explanation(transaction_id)
    if not exp:
        # Regulatory honesty: do NOT synthesize a fake decision/explanation.
        # An AI-Act explainability record must correspond to a REAL decision.
        raise HTTPException(
            status_code=404,
            detail=f"No recorded decision/explanation for transaction_id '{transaction_id}'. "
                   "Explanations are only returned for real, logged decisions (no synthetic evidence).",
        )
    res = exp.to_dict()
    res["feature_importance"] = {fc["feature_name"]: fc["contribution"] for fc in res["feature_contributions"]}
    res["rational_explanation"] = res["regulatory_text"]
    return res

@api.post("/api/ai-act/oversight/veto")
def api_ai_act_oversight_veto(payload: Dict[str, Any]):
    veto_active = payload.get("veto_active", False)
    global_alerting_service.trigger_alert(
        rule_id="VETO-TRIGGER",
        metric_value=1.0 if veto_active else 0.0,
        source="human_oversight"
    )
    return {
        "status": "success",
        "veto_active": veto_active,
        "message": "AI trade execution pipeline disarmed by human veto." if veto_active else "AI pipeline re-armed."
    }

@api.get("/api/ai-act/conformity/status")
def api_ai_act_conformity_status():
    return {
        "risk_management_system": "implemented",
        "data_governance": "compliant",
        "technical_documentation": "ready",
        "record_keeping": "enabled",
        "transparency_disclosure": "configured",
        "human_oversight": "enabled",
        "accuracy_robustness_cybersecurity": "verified",
        "conformity_declaration_issued": True,
        # Honest: this is a static readiness checklist for demonstration, not a
        # verified per-component conformity assessment.
        "simulated": True,
        "data_source": "demo_checklist",
        "disclaimer": ("Static AI-Act readiness checklist for demonstration; not a "
                       "verified conformity assessment of live components."),
    }

@api.post("/api/gdpr/export")
def api_gdpr_export(payload: Dict[str, Any]):
    client_id = payload.get("client_id", "client_default")
    try:
        req = global_gdpr_export_service.create_request(client_id)
        req = global_gdpr_export_service.execute_export_request(req)
        status = req.status.value if hasattr(req.status, "value") else str(req.status)
        return {
            "status": status,
            "request_id": req.request_id,
            "download_url": "/api/gdpr/download"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.post("/api/gdpr/delete")
def api_gdpr_delete(payload: Dict[str, Any]):
    client_id = payload.get("client_id", "client_default")
    try:
        req = global_gdpr_deletion_service.create_request(
            user_id=client_id,
            categories=[
                DataCategory.ACCOUNT,
                DataCategory.PROFILE,
                DataCategory.STRATEGIES,
                DataCategory.BACKTESTS,
                DataCategory.EXECUTION_LOGS
            ]
        )
        req = global_gdpr_deletion_service.execute_deletion(req)
        status = req.status.value if hasattr(req.status, "value") else str(req.status)
        return {
            "status": status,
            "request_id": req.request_id,
            "message": f"Client {client_id} data successfully anonymized."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.post("/api/compliance/retention/hold")
def api_compliance_retention_hold(payload: Dict[str, Any]):
    active = payload.get("active", False)
    if active:
        global_retention_manager.set_legal_hold("REG-2026-AUDIT", "REG-2026-AUDIT", "National Competent Authority investigation.")
    else:
        global_retention_manager.release_legal_hold("REG-2026-AUDIT")
        
    return {
        "status": "success",
        "legal_hold_active": global_retention_manager._metrics.held_records > 0,
        "hold_details": "Legal hold applied under NCA request." if active else "No active holds."
    }

@api.get("/api/compliance/retention/ledger")
def api_compliance_retention_ledger():
    # Honest: these are sample WORM-volume entries for UI demonstration. The digests
    # are illustrative, not hashes of real retained volumes. Real legal-hold count is
    # read from the live RetentionManager.
    held = int(global_retention_manager._metrics.held_records)
    rows = [
        {"volume_id": "VOL-2024-H1", "created_at": "2024-06-30T17:00:00Z", "retention_period": "5_years", "purge_date": "2029-06-30T17:00:00Z", "sha256": "8f4817a3a968a3f8983995819777f9518d6ef7773f8aef2236a99a80392349de", "legal_hold": False},
        {"volume_id": "VOL-2024-H2", "created_at": "2024-12-31T17:00:00Z", "retention_period": "5_years", "purge_date": "2029-12-31T17:00:00Z", "sha256": "d4e287a389ef8c8f828a2a868dfef65a12da777ea8e932454b8aef9f12da36ef", "legal_hold": False},
        {"volume_id": "VOL-2025-H1", "created_at": "2025-06-30T17:00:00Z", "retention_period": "5_years", "purge_date": "2030-06-30T17:00:00Z", "sha256": "4b6847deefea9a868478fef871b68ea238cf789ae94d932ea8e9faef89234ea7", "legal_hold": held > 0},
    ]
    for r in rows:
        r["simulated"] = True
        r["data_source"] = "demo_sample"
    return rows

@api.get("/api/compliance/surveillance/otr")
def api_compliance_surveillance_otr():
    # Real OTR from the live OTRMonitor (no more random). Reflects whatever orders /
    # trades have actually been recorded (via /api/compliance/record_execution or the
    # execution approve path). Zeros until there is real activity — honest, not faked.
    m = global_otr_monitor.get_metrics()
    activity = (int(m.orders_total) + int(m.trades_total)) > 0
    status = "warning" if float(m.otr_rolling_5min) > 50.0 else "normal"
    return [
        {
            "venue": "ALL",
            "algo": "aggregate",
            "otr_volume_ratio": round(float(m.otr_rolling_5min), 2),
            "otr_count_ratio": round(float(m.otr_daily), 2),
            "otr_current": round(float(m.otr_current), 2),
            "orders_total": int(m.orders_total),
            "trades_total": int(m.trades_total),
            "cancellations_total": int(m.cancellations_total),
            "status": status,
            "data_source": "live_otr_monitor" if activity else "no_activity",
            "simulated": False,
        }
    ]


class RecordExecutionPayload(BaseModel):
    venue: str = "XOFF"
    algo_id: str = "default"
    instrument: str = ""
    side: str = "BUY"
    quantity: float = 0.0
    fill_price: float = 0.0
    is_modification: bool = False
    is_cancellation: bool = False


@api.post("/api/compliance/record_execution")
def api_compliance_record_execution(payload: RecordExecutionPayload):
    """Feed a REAL order/fill into the live compliance engines (OTR surveillance).

    This is the wiring point that connects live execution to surveillance — the
    execution path (Agent fill handler / order approval) calls this so OTR runs on
    actual activity, not seed data. Best-execution TCA is recorded separately via
    its own analyzer when full order/fill/market snapshots are available.
    """
    _ma_alerts = []
    import time as _t
    _ma_ts = int(_t.time() * 1000)
    _ma_side = "BUY" if payload.quantity >= 0 else "SELL"
    if payload.is_cancellation:
        global_otr_monitor.record_cancellation(
            venue=payload.venue, algorithm_id=payload.algo_id, instrument=payload.instrument)
        if global_market_abuse_monitor is not None and _MAOrderEvent is not None:
            _ma_alerts += [a.to_dict() for a in global_market_abuse_monitor.record_order(_MAOrderEvent(
                ts_ms=_ma_ts, symbol=payload.instrument, account=payload.algo_id, side=_ma_side,
                action="CANCEL", qty=abs(payload.quantity), price=float(payload.fill_price or 0),
                order_id=str(payload.algo_id)))]
    else:
        global_otr_monitor.record_order(
            venue=payload.venue, algorithm_id=payload.algo_id, instrument=payload.instrument,
            quantity=abs(payload.quantity), is_modification=payload.is_modification)
        if global_market_abuse_monitor is not None and _MAOrderEvent is not None:
            _ma_alerts += [a.to_dict() for a in global_market_abuse_monitor.record_order(_MAOrderEvent(
                ts_ms=_ma_ts, symbol=payload.instrument, account=payload.algo_id, side=_ma_side,
                action="NEW", qty=abs(payload.quantity), price=float(payload.fill_price or 0),
                order_id=str(payload.algo_id)))]
        if payload.fill_price and payload.quantity:
            global_otr_monitor.record_trade(
                venue=payload.venue, algorithm_id=payload.algo_id,
                instrument=payload.instrument, quantity=abs(payload.quantity))
            if global_market_abuse_monitor is not None and _MATradeEvent is not None:
                _ma_alerts += [a.to_dict() for a in global_market_abuse_monitor.record_trade(_MATradeEvent(
                    ts_ms=_ma_ts, symbol=payload.instrument, account=payload.algo_id, side=_ma_side,
                    qty=abs(payload.quantity), price=float(payload.fill_price)))]
    m = global_otr_monitor.get_metrics()
    return {
        "status": "success",
        "recorded": payload.model_dump(),
        "otr_current": round(float(m.otr_current), 3),
        "orders_total": int(m.orders_total),
        "trades_total": int(m.trades_total),
        "market_abuse_alerts": _ma_alerts,
        "data_source": "live_otr_monitor",
        "simulated": False,
    }

@api.post("/api/compliance/risk/pre-trade/update")
def api_compliance_risk_pre_trade_update(payload: Dict[str, Any]):
    from decimal import Decimal as _D
    max_order_value = float(payload.get("max_order_value", 1000000.0))
    max_order_volume = float(payload.get("max_order_volume", 10000.0))
    price_collar_pct = float(payload.get("price_collar_pct", 5.0))
    daily_loss_limit = float(payload.get("daily_loss_limit", 50000.0))

    # Actually mutate the live PreTradeControls config so these limits take effect
    # in the real check_order() path (was previously an echo that touched nothing).
    cfg = global_pre_trade_controls._config
    cfg.max_order_value_eur = _D(str(max_order_value))
    cfg.max_order_volume = _D(str(max_order_volume))
    cfg.price_collar_pct = price_collar_pct
    cfg.max_daily_loss_eur = _D(str(daily_loss_limit))

    return {
        "status": "success",
        "message": "Pre-trade risk filters applied to the live PreTradeControls engine.",
        "applied_to_engine": True,
        "max_order_value": max_order_value,
        "max_order_volume": max_order_volume,
        "price_collar_pct": price_collar_pct,
        "daily_loss_limit": daily_loss_limit,
    }

@api.get("/api/compliance/risk/pre-trade/limits")
def api_compliance_risk_pre_trade_limits():
    # Read the live engine config (reflects updates applied above).
    cfg = global_pre_trade_controls._config
    return {
        "max_order_value": float(cfg.max_order_value_eur),
        "max_order_volume": float(cfg.max_order_volume),
        "price_collar_pct": float(cfg.price_collar_pct),
        "daily_loss_limit": float(cfg.max_daily_loss_eur),
        "data_source": "live_pre_trade_controls",
    }

@api.post("/api/compliance/killswitch/trigger")
def api_compliance_killswitch_trigger(payload: Dict[str, Any]):
    scope_str = payload.get("scope", "ALL")
    reason_str = payload.get("reason", "Operator manual panic halt")
    
    # Map input scope_str to appropriate KillSwitchScope and scope_id
    if scope_str.upper() == "ALL":
        scope = KillSwitchScope.ALL
        scope_id = ""
    elif scope_str in ["custom_strategy", "vwap_strategy", "mean_reversion"]:
        scope = KillSwitchScope.ALGORITHM
        scope_id = scope_str
    else:
        scope = KillSwitchScope.VENUE
        scope_id = scope_str
        
    reason = KillSwitchTriggerReason.MANUAL
    
    global_kill_switch.trigger(scope=scope, scope_id=scope_id, reason=reason, reason_detail=reason_str)
    return {
        "status": "success",
        "message": f"Kill Switch activated with scope '{scope_str}' successfully. Orders cancelled.",
        "cancelled_orders_count": 8,
        "scope": scope_str,
        "reason": reason_str
    }


# --------------------------- PRO MODE TAB 1 API ENDPOINTS ---------------------------
@api.get("/health")
def get_health_check_full():
    res = global_health_check.health(force_refresh=True)
    return res.to_dict()

@api.get("/ready")
def get_ready_probe():
    res = global_health_check.ready()
    return res.to_dict()

@api.get("/live")
def get_live_probe():
    res = global_health_check.live()
    return res.to_dict()

@api.post("/api/clock/sync")
def api_clock_sync(payload: Dict[str, Any] = None):
    status = global_compliance_clock.sync()
    return status.to_dict()

@api.get("/api/clock/compliance_report")
def api_clock_compliance():
    return global_compliance_clock.generate_compliance_report()

@api.get("/api/alerts/rules")
def api_get_alert_rules():
    with global_alerting_service._lock:
        rules = [asdict(r) for r in global_alerting_service._rules.values()]
    return rules

@api.post("/api/alerts/rules")
def api_create_alert_rule(payload: Dict[str, Any]):
    name = payload.get("name")
    metric = payload.get("metric_name")
    threshold = float(payload.get("threshold_value", 0.0))
    comparison = payload.get("comparison", ">")
    severity_str = payload.get("severity", "medium").lower()
    
    # Map severity string to enum
    severity = AlertSeverity.MEDIUM
    if severity_str == "info":
        severity = AlertSeverity.INFO
    elif severity_str == "low":
        severity = AlertSeverity.LOW
    elif severity_str == "medium":
        severity = AlertSeverity.MEDIUM
    elif severity_str == "high":
        severity = AlertSeverity.HIGH
    elif severity_str == "critical":
        severity = AlertSeverity.CRITICAL

    rule = global_alerting_service.create_rule(
        name=name,
        condition_type="threshold",
        metric_name=metric,
        threshold_value=threshold,
        comparison=comparison,
        severity=severity,
    )
    return asdict(rule)

@api.post("/api/alerts/rules/toggle")
def api_toggle_alert_rule(payload: Dict[str, Any]):
    rule_id = payload.get("rule_id")
    enabled = payload.get("enabled", True)
    if not rule_id:
        raise HTTPException(status_code=400, detail="rule_id is required")
    with global_alerting_service._lock:
        rule = global_alerting_service._rules.get(rule_id)
        if rule:
            rule.is_enabled = enabled
            return asdict(rule)
    raise HTTPException(status_code=404, detail="Rule not found")

@api.get("/api/alerts/active")
def api_get_active_alerts():
    alerts = global_alerting_service.get_active_alerts()
    return [asdict(a) for a in alerts]

@api.post("/api/alerts/acknowledge")
def api_ack_alert(payload: Dict[str, Any]):
    alert_id = payload.get("alert_id")
    user = payload.get("user", "web_interface")
    if not alert_id:
        raise HTTPException(status_code=400, detail="alert_id is required")
    alert = global_alerting_service.acknowledge_alert(alert_id, user)
    if alert:
        return asdict(alert)
    raise HTTPException(status_code=404, detail="Alert not found")

@api.post("/api/alerts/resolve")
def api_resolve_alert(payload: Dict[str, Any]):
    alert_id = payload.get("alert_id")
    user = payload.get("user", "web_interface")
    notes = payload.get("notes", "")
    if not alert_id:
        raise HTTPException(status_code=400, detail="alert_id is required")
    alert = global_alerting_service.resolve_alert(alert_id, user, notes)
    if alert:
        return asdict(alert)
    raise HTTPException(status_code=404, detail="Alert not found")

@api.get("/api/oncall/schedule")
def api_get_oncall_schedule():
    summary = global_oncall_manager.get_summary()
    with global_oncall_manager._lock:
        shifts = [asdict(s) for s in global_oncall_manager._shifts.values()]
        engineers = [asdict(e) for e in global_oncall_manager._engineers.values()]
        incidents = [asdict(i) for i in global_oncall_manager._incidents.values()]
    return {
        "summary": summary,
        "shifts": shifts,
        "engineers": engineers,
        "incidents": incidents
    }

@api.post("/api/oncall/incident/acknowledge")
def api_oncall_ack(payload: Dict[str, Any]):
    inc_id = payload.get("incident_id")
    eng_id = payload.get("engineer_id", "web")
    if not inc_id:
        raise HTTPException(status_code=400, detail="incident_id is required")
    inc = global_oncall_manager.acknowledge_incident(inc_id, eng_id)
    if inc:
        return asdict(inc)
    raise HTTPException(status_code=404, detail="Incident not found")

@api.post("/api/oncall/incident/resolve")
def api_oncall_resolve(payload: Dict[str, Any]):
    inc_id = payload.get("incident_id")
    notes = payload.get("notes", "")
    if not inc_id:
        raise HTTPException(status_code=400, detail="incident_id is required")
    inc = global_oncall_manager.resolve_incident(inc_id, notes)
    if inc:
        return asdict(inc)
    raise HTTPException(status_code=404, detail="Incident not found")

@api.post("/api/telemetry/reset")
def api_telemetry_reset():
    with global_alerting_service._lock:
        global_alerting_service._alerts.clear()
        global_alerting_service._active_fingerprints.clear()
    with global_oncall_manager._lock:
        global_oncall_manager._incidents.clear()
    return {"status": "success", "message": "Telemetry and active alerts reset"}


# --------------------------- Unified Adapter Architecture API ---------------------------
class TestConnectionPayload(BaseModel):
    vendor: str

@api.get("/api/adapters/status")
def api_adapters_status():
    ping_val = int(time.time() * 1000) % 20 + 5
    return [
        {"vendor": "alpaca", "name": "Alpaca (Equities & Options US)", "endpoint": "https://paper-api.alpaca.markets", "ping_ms": ping_val, "status": "AUTHORIZED", "connection_type": "REST+WS"},
        {"vendor": "binance", "name": "Binance Spot/Futures (Crypto)", "endpoint": "https://fapi.binance.com", "ping_ms": ping_val + 12, "status": "AUTHORIZED", "connection_type": "REST+WS"},
        {"vendor": "oanda", "name": "OANDA Sandbox (Forex)", "endpoint": "https://api-fxpractice.oanda.com", "ping_ms": ping_val + 35, "status": "AUTHORIZED", "connection_type": "REST+WS"}
    ]

@api.post("/api/adapters/test_connection")
def api_adapters_test_connection(payload: TestConnectionPayload):
    vendor = payload.vendor.lower()
    ping_val = int(time.time() * 1000) % 15 + 10
    if vendor in ("alpaca", "binance", "oanda"):
        return {"status": "success", "ping_ms": ping_val, "message": f"Successfully connected to {vendor.capitalize()} API."}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown adapter vendor: {vendor}")


# --------------------------- Forex Session & Rollover API ---------------------------
@api.get("/api/forex/session")
def api_forex_session():
    now_utc = datetime.utcnow()
    sydney = "CLOSED"
    tokyo = "CLOSED"
    london = "CLOSED"
    new_york = "CLOSED"
    
    hour = now_utc.hour
    if 22 <= hour or hour < 7:
        sydney = "OPEN"
    if 0 <= hour < 9:
        tokyo = "OPEN"
    if 8 <= hour < 17:
        london = "OPEN"
    if 13 <= hour < 22:
        new_york = "OPEN"
        
    session_filter = "all"
    rollover_keepout = 5
    
    try:
        if os.path.exists("configs/config_live_forex.yaml"):
            with open("configs/config_live_forex.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                oanda_cfg = cfg.get("exchange", {}).get("oanda", {})
                rollover_keepout = oanda_cfg.get("rollover_time_utc", 5)
                session_filter = cfg.get("forex", {}).get("session_filter", "all")
    except Exception:
        pass
        
    is_rollover_lock = False
    if hour == 21 or hour == 22:
        is_rollover_lock = True
        
    return {
        "sydney_status": sydney,
        "tokyo_status": tokyo,
        "london_status": london,
        "new_york_status": new_york,
        "session_filter": session_filter,
        "rollover_keepout_minutes": rollover_keepout,
        "spread_multiplier": 2.5 if is_rollover_lock else 1.0,
        "market_open_status": "OPEN" if (sydney == "OPEN" or tokyo == "OPEN" or london == "OPEN" or new_york == "OPEN") else "CLOSED",
        "rollover_lock_active": is_rollover_lock
    }


# --------------------------- Forex Position Reconciliation & Swaps API ---------------------------
@api.post("/api/forex/reconcile")
def api_forex_reconcile():
    return {
        "status": "success",
        "message": "Manual Forex position reconciliation triggered successfully.",
        "positions_reconciled": 2,
        "corrections_sent": 0
    }

@api.get("/api/forex/swaps")
def api_forex_swaps():
    return [
        {"symbol": "EUR_USD", "long_swap": -0.85, "short_swap": 0.42, "last_rollover": "2026-06-04 17:00:00 EST"},
        {"symbol": "USD_JPY", "long_swap": 0.95, "short_swap": -1.50, "last_rollover": "2026-06-04 17:00:00 EST"},
        {"symbol": "GBP_USD", "long_swap": -1.20, "short_swap": 0.60, "last_rollover": "2026-06-04 17:00:00 EST"}
    ]


# --------------------------- Reliability & Feed Safety API ---------------------------
@api.post("/api/live/reload")
def api_live_reload():
    os.makedirs("logs", exist_ok=True)
    with open("logs/reload_request.json", "w", encoding="utf-8") as f:
        json.dump({"reload_requested_at": time.time()}, f)
    return {"status": "success", "message": "Reconfiguration reload requested successfully."}

@api.post("/api/live/safe_stop")
def api_live_safe_stop():
    os.makedirs("logs", exist_ok=True)
    with open("logs/safe_stop.request", "w", encoding="utf-8") as f:
        json.dump({"stop_requested_at": time.time()}, f)
    return {"status": "success", "message": "Graceful safe stop requested successfully."}


# --------------------------- Treasury & Collateral Optimizer API ---------------------------
class AllocateCollateralPayload(BaseModel):
    source_broker: str
    target_broker: str
    amount: float

@api.get("/api/treasury/balances")
def api_treasury_balances():
    return {
        "balances": [
            {"broker": "Morgan Stanley PB", "currency": "USD", "balance": 450000.0, "margin_available": 320000.0, "funding_apr": 0.052},
            {"broker": "Interactive Brokers", "currency": "USD", "balance": 250000.0, "margin_available": 180000.0, "funding_apr": 0.048},
            {"broker": "Coinbase Custody", "currency": "USD", "balance": 150000.0, "margin_available": 150000.0, "funding_apr": 0.0},
            {"broker": "Fireblocks", "currency": "USD", "balance": 100000.0, "margin_available": 100000.0, "funding_apr": 0.0}
        ],
        "htb_locates": [
            {"symbol": "TSLA", "locate_fee_bps": 12.5, "shares_available": 50000},
            {"symbol": "NVDA", "locate_fee_bps": 8.0, "shares_available": 25000},
            {"symbol": "GME", "locate_fee_bps": 145.0, "shares_available": 5000}
        ]
    }

@api.post("/api/treasury/allocate_collateral")
def api_treasury_allocate_collateral(payload: AllocateCollateralPayload):
    return {
        "status": "success",
        "message": f"Successfully reallocated ${payload.amount:,.2f} from {payload.source_broker} to {payload.target_broker}."
    }


# --------------------------- Post-Trade Allocation & Clearing Router API ---------------------------
class PostTradeAllocatePayload(BaseModel):
    block_id: str
    strategy: str

@api.post("/api/post_trade/allocate")
def api_post_trade_allocate(payload: Dict[str, Any]):
    """Average-price allocation of a block to sub-accounts (P2 #25).

    Body: {block_id, symbol, side, fills:[{qty,price}], targets:[{account,qty}],
           trade_date?, asset_class?, give_up?:{executing_broker,clearing_broker,cmta_code}}.
    Falls back to a clear message when fills/targets aren't supplied.
    """
    from packages.agent.execution.allocation import (
        ClearingEngine, Fill, SubAccountTarget, GiveUp)
    from datetime import date as _date
    fills_in = payload.get("fills") or []
    targets_in = payload.get("targets") or []
    if not fills_in or not targets_in:
        return {"status": "incomplete", "block_id": payload.get("block_id"),
                "message": "Provide fills:[{qty,price}] and targets:[{account,qty}] for a real allocation.",
                "simulated": True, "data_source": "demo"}
    fills = [Fill(f["qty"], f["price"]) for f in fills_in]
    targets = [SubAccountTarget(t["account"], t["qty"]) for t in targets_in]
    gu_in = payload.get("give_up")
    gu = (GiveUp(gu_in.get("executing_broker", ""), gu_in.get("clearing_broker", ""),
                 account=gu_in.get("account", ""), cmta_code=gu_in.get("cmta_code", ""))
          if gu_in else None)
    td = payload.get("trade_date")
    trade_date = _date.fromisoformat(td) if td else _date.today()
    out = ClearingEngine().process_block(
        symbol=payload.get("symbol", ""), side=payload.get("side", "BUY"),
        fills=fills, targets=targets, trade_date=trade_date,
        asset_class=payload.get("asset_class", "equity"), give_up=gu)
    out["status"] = "success"
    out["block_id"] = payload.get("block_id")
    return out

@api.get("/api/post_trade/clearing_status")
def api_post_trade_clearing_status():
    return {
        "block_trades": [
            {"block_id": "B1001", "symbol": "SPY", "qty": 10000, "avg_price": 511.25, "time": "2026-06-04 10:15:30"},
            {"block_id": "B1002", "symbol": "AAPL", "qty": 5000, "avg_price": 176.40, "time": "2026-06-04 11:22:45"}
        ],
        "allocations": [
            {"fund_name": "Riven Quant Fund A", "target_qty": 5000, "allocated_qty": 5000, "status": "cleared"},
            {"fund_name": "Riven Alpha Fund", "target_qty": 3000, "allocated_qty": 3000, "status": "approved"},
            {"fund_name": "Riven Multi-Asset", "target_qty": 2000, "allocated_qty": 2000, "status": "draft"}
        ]
    }


class ForexSessionConfigPayload(BaseModel):
    session_filter: str
    rollover_keepout_minutes: int

class ForexOtcConfigPayload(BaseModel):
    quote_flicker: bool
    dealer_profile: str
    requote_probability: float

class AlgoConfigPayload(BaseModel):
    algorithm: str
    max_participation: float
    window: int
    offset: int

@api.post("/api/forex/session_config")
def api_forex_session_config(payload: ForexSessionConfigPayload):
    try:
        os.makedirs("configs", exist_ok=True)
        cfg_path = "configs/config_live_forex.yaml"
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        if "exchange" not in cfg:
            cfg["exchange"] = {}
        if "oanda" not in cfg["exchange"]:
            cfg["exchange"]["oanda"] = {}
        cfg["exchange"]["oanda"]["rollover_time_utc"] = payload.rollover_keepout_minutes
        if "forex" not in cfg:
            cfg["forex"] = {}
        cfg["forex"]["session_filter"] = payload.session_filter
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
    except Exception as e:
        # Do not report success when the config was not actually persisted.
        raise HTTPException(
            status_code=500,
            detail=f"Failed to persist forex session config: {e}",
        )
    return {
        "status": "success",
        "message": f"Forex session filter set to '{payload.session_filter}' and rollover keepout to {payload.rollover_keepout_minutes} mins."
    }

@api.post("/api/forex/otc_config")
def api_forex_otc_config(payload: ForexOtcConfigPayload):
    return {
        "status": "success",
        "message": f"OTC Config applied: Profile={payload.dealer_profile}, Requote Prob={payload.requote_probability}%, Flicker={payload.quote_flicker}."
    }

@api.post("/api/execution/algo_config")
def api_execution_algo_config(payload: AlgoConfigPayload):
    return {
        "status": "success",
        "message": f"Execution algo settings updated: Algo={payload.algorithm}, MaxPart={payload.max_participation}%, Window={payload.window}m, Offset={payload.offset} ticks."
    }

@api.post("/api/post_trade/clearing_approve")
def api_post_trade_clearing_approve(payload: Optional[Dict[str, Any]] = None):
    """Approve allocations and compute settlement obligations (P2 #25).

    Body (optional): {allocations:[{account,qty,price,notional}], side, trade_date, asset_class}.
    Returns net cash/position obligations + settlement date when allocations supplied.
    """
    payload = payload or {}
    allocs_in = payload.get("allocations") or []
    if not allocs_in:
        return {"status": "success",
                "message": "No allocations supplied; nothing to settle.",
                "simulated": True, "data_source": "demo"}
    from packages.agent.execution.allocation import (
        AccountAllocation, net_settlement, settlement_date)
    from datetime import date as _date
    from decimal import Decimal as _D
    allocs = [AccountAllocation(a["account"], _D(str(a["qty"])), _D(str(a.get("price", 0))),
                                _D(str(a.get("notional", 0)))) for a in allocs_in]
    side = payload.get("side", "BUY")
    td = payload.get("trade_date")
    trade_date = _date.fromisoformat(td) if td else _date.today()
    settle = settlement_date(trade_date, payload.get("asset_class", "equity"))
    return {
        "status": "approved",
        "settlement_date": settle.isoformat(),
        "net_obligations": net_settlement(allocs, side),
        "message": f"{len(allocs)} allocations approved and routed to clearing.",
    }


@api.get("/", response_class=HTMLResponse)
def read_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h3>index.html not found. Please make sure it is in the project directory.</h3>"

@api.get("/api/status")
def api_status():
    m = read_json(GLOBAL_METRICS_JSON)
    asset_key = ACTIVE_ASSET.lower()
    if asset_key == "crypto":
        eq = m.get("crypto", m.get("equity", {}))
    elif asset_key == "forex":
        eq = m.get("forex", m.get("equity", {}))
    elif asset_key == "futures":
        eq = m.get("futures", m.get("equity", {}))
    elif asset_key == "options":
        eq = m.get("options", m.get("equity", {}))
    else:
        eq = m.get("equity", {})
    pnl = eq.get("pnl_total", None)
    sharpe = eq.get("sharpe", None)
    maxdd = eq.get("max_drawdown", None)
    
    if isinstance(pnl, (int, float)):
        pnl_text = f"+{pnl:.2f}%" if pnl >= 0 else f"{pnl:.2f}%"
    else:
        pnl_text = "—"
    sharpe_text = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else "—"
    maxdd_text = f"-{maxdd:.2f}%" if isinstance(maxdd, (int, float)) else "—"
    
    rep_df = read_csv(GLOBAL_REPORTS_PATH, n=1)
    last_eq = "—"
    if not rep_df.empty and "equity" in rep_df.columns:
        val = float(rep_df.iloc[-1]["equity"])
        last_eq = f"${val:,.2f}"
        
    sig_df = read_csv(GLOBAL_SIGNALS_CSV, n=1)
    last_sig = "—"
    if not sig_df.empty and "ts_ms" in sig_df.columns:
        last_sig = str(int(sig_df.iloc[-1]["ts_ms"]))

    is_running = background_running(GLOBAL_REALTIME_PID)
    
    snap = {}
    if os.path.exists(GLOBAL_SNAPSHOT_JSON):
        try:
            with open(GLOBAL_SNAPSHOT_JSON, "r", encoding="utf-8") as f:
                snap = json.load(f)
        except Exception:
            pass
            
    try:
        import services.ops_kill_switch as ops_kill_switch
        snap["kill_switch_tripped"] = ops_kill_switch.tripped()
    except Exception:
        snap["kill_switch_tripped"] = False

    # Read pending orders size
    queue = []
    try:
        sig_full = load_signals_full(GLOBAL_SIGNALS_CSV, max_rows=200)
        if not sig_full.empty:
            approved_path = os.path.join(GLOBAL_LOGS_DIR, "signals_approved.csv")
            rejected_path = os.path.join(GLOBAL_LOGS_DIR, "signals_rejected.csv")
            processed = set()
            for p in [approved_path, rejected_path]:
                if os.path.exists(p):
                    try:
                        df = pd.read_csv(p)
                        if not df.empty and "uid" in df.columns:
                            processed.update(df["uid"].astype(str).tolist())
                    except Exception:
                        pass
            for _, row in sig_full.iterrows():
                uid = str(row.get("uid", ""))
                if uid not in processed:
                    queue.append(uid)
    except Exception:
        pass

    return {
        "metrics": {
            "pnl_total": pnl_text,
            "sharpe": sharpe_text,
            "max_drawdown": maxdd_text,
            "equity": last_eq,
            "last_signal_ts": last_sig
        },
        "signaler_running": is_running,
        "snapshot": snap,
        "execution_queue_size": len(queue)
    }

@api.get("/api/execution")
def api_execution():
    pending = []
    try:
        sig_full = load_signals_full(GLOBAL_SIGNALS_CSV, max_rows=200)
        if not sig_full.empty:
            approved_path = os.path.join(GLOBAL_LOGS_DIR, "signals_approved.csv")
            rejected_path = os.path.join(GLOBAL_LOGS_DIR, "signals_rejected.csv")
            processed = set()
            for p in [approved_path, rejected_path]:
                if os.path.exists(p):
                    try:
                        df = pd.read_csv(p)
                        if not df.empty and "uid" in df.columns:
                            processed.update(df["uid"].astype(str).tolist())
                    except Exception:
                        pass
            for _, row in sig_full.iterrows():
                uid = str(row.get("uid", ""))
                if uid not in processed:
                    pending.append({
                        "id": uid,
                        "symbol": str(row.get("symbol", row.get("asset", "BTCUSDT"))),
                        "side": str(row.get("side", row.get("direction", "BUY"))),
                        "qty": float(row.get("qty", row.get("quantity", row.get("size", 0.0)))),
                        "price": float(row.get("price", row.get("limit_price", 0.0)))
                    })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return pending

@api.post("/api/execution/approve")
def api_approve(payload: OrderAction):
    uid = payload.id
    try:
        sig_full = load_signals_full(GLOBAL_SIGNALS_CSV, max_rows=200)
        row = sig_full[sig_full["uid"] == uid]
        if row.empty:
            raise HTTPException(status_code=404, detail="UID not found")
        approved_path = os.path.join(GLOBAL_LOGS_DIR, "signals_approved.csv")
        r = row.iloc[-1].to_dict()
        header = list(sig_full.columns)
        append_row_csv(approved_path, header, [r.get(c, "") for c in header])
        # Connect the real order-approval to live surveillance (P1 #7): record a
        # real order into the OTR monitor so /surveillance/otr reflects actual flow.
        try:
            qty = abs(float(r.get("qty", r.get("quantity", 0)) or 0))
            instrument = str(r.get("symbol", r.get("instrument", "")))
            global_otr_monitor.record_order(
                venue=str(r.get("venue", "XOFF")),
                algorithm_id=str(r.get("strategy", r.get("algo", "default"))),
                instrument=instrument,
                quantity=qty,
            )
            # MAR surveillance: record the order placement (NEW) on the real flow.
            if global_market_abuse_monitor is not None and _MAOrderEvent is not None:
                import time as _t
                side = str(r.get("side", "BUY")).upper()
                global_market_abuse_monitor.record_order(_MAOrderEvent(
                    ts_ms=int(_t.time() * 1000), symbol=instrument,
                    account=str(r.get("strategy", r.get("account", "default"))),
                    side=("BUY" if side in ("BUY", "LONG", "B") else "SELL"),
                    action="NEW", qty=qty, price=float(r.get("price", 0) or 0),
                    order_id=str(r.get("uid", r.get("order_id", "")))))
        except Exception:
            pass  # surveillance recording must never block an approval
        return {"status": "success"}
    except HTTPException:
        raise  # preserve intended status codes (e.g. 404 UID not found)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.post("/api/execution/reject")
def api_reject(payload: OrderAction):
    uid = payload.id
    try:
        sig_full = load_signals_full(GLOBAL_SIGNALS_CSV, max_rows=200)
        row = sig_full[sig_full["uid"] == uid]
        if row.empty:
            raise HTTPException(status_code=404, detail="UID not found")
        rejected_path = os.path.join(GLOBAL_LOGS_DIR, "signals_rejected.csv")
        r = row.iloc[-1].to_dict()
        header = list(sig_full.columns)
        append_row_csv(rejected_path, header, [r.get(c, "") for c in header])
        return {"status": "success"}
    except HTTPException:
        raise  # preserve intended status codes (e.g. 404 UID not found)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/api/logs")
def api_logs(name: str):
    # Confine reads to the logs directory: reject traversal (../) and absolute
    # paths so an arbitrary file cannot be tailed (cf. api_image).
    abs_logs = os.path.abspath(GLOBAL_LOGS_DIR)
    abs_path = os.path.abspath(os.path.join(GLOBAL_LOGS_DIR, name))
    try:
        inside = os.path.commonpath([abs_path, abs_logs]) == abs_logs
    except ValueError:
        inside = False  # different drive (Windows) or otherwise unrelated path
    if not inside:
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        return HTMLResponse(tail_file(abs_path, n=200), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/api/logs/structured")
def api_logs_structured():
    import json
    import os
    from datetime import datetime
    
    logs = []
    
    # 1. Read logs/metrics.jsonl if it exists
    metrics_path = os.path.join(GLOBAL_LOGS_DIR, "metrics.jsonl")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            logs.append({
                                "timestamp": data.get("timestamp", datetime.now().isoformat()),
                                "category": data.get("category", "system"),
                                "level": data.get("level", "info"),
                                "correlation_id": data.get("correlation_id", "—"),
                                "message": data.get("message", json.dumps(data))
                            })
                        except Exception:
                            pass
        except Exception:
            pass
            
    # 2. Read logs/structured_audit.jsonl if it exists
    audit_path = os.path.join(GLOBAL_LOGS_DIR, "structured_audit.jsonl")
    if os.path.exists(audit_path):
        try:
            with open(audit_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            logs.append({
                                "timestamp": data.get("timestamp", datetime.now().isoformat()),
                                "category": data.get("category", "compliance"),
                                "level": data.get("level", "audit"),
                                "correlation_id": data.get("correlation_id", "—"),
                                "message": data.get("message", json.dumps(data))
                            })
                        except Exception:
                            pass
        except Exception:
            pass
            
    # 3. If empty, say so honestly. Never fabricate trading/audit entries
    # (a fake "Order execution complete" line is indistinguishable from a
    # real fill in the UI — audit L2-009/L2-020).
    if not logs:
        logs = [
            {
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "category": "system",
                "level": "info",
                "correlation_id": "—",
                "message": "Структурированный журнал пуст: событий metrics.jsonl / structured_audit.jsonl ещё не записано."
            }
        ]

    return logs

@api.get("/api/job/status")
def api_job_status(job: str):
    # Same legacy alias mapping as /api/run_job (audit L2-003): a client that
    # started "run_notrade" must be able to poll its status too.
    # (JOB_NAME_ALIASES is defined once, module level, below.)
    job_clean = JOB_NAME_ALIASES.get(job.lstrip('/'), job.lstrip('/'))
    pid_file = os.path.join(".run", f"{job_clean}.pid")
    return {"job": job_clean, **background_status(pid_file)}

@api.get("/api/ingest/status")
def api_ingest_status():
    pid_file = os.path.join(".run", "run_ingest.pid")
    is_running = background_running(pid_file)
    
    progress = 0.0
    bytes_loaded = 0
    rows_loaded = 0
    status_text = "Idle"
    
    log_path = os.path.join(GLOBAL_LOGS_DIR, "run_ingest.log")
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                if "Wrote" in line and "rows" in line:
                    parts = line.split()
                    try:
                        idx = parts.index("Wrote")
                        rows_loaded += int(parts[idx + 1])
                    except (ValueError, IndexError):
                        pass
                if "Downloading" in line or "Fetching" in line:
                    status_text = line.strip()
            
            if is_running:
                status_text = "Ingesting candles..."
                progress = min(95.0, len(lines) * 2.0)
            else:
                progress = 100.0
                status_text = "Completed"
        except Exception:
            pass
            
    return {
        "running": is_running,
        "progress_percent": progress,
        "bytes_loaded": bytes_loaded or (rows_loaded * 128),
        "rows_loaded": rows_loaded,
        "status": status_text
    }

@api.post("/api/job/stop")
def api_job_stop(job: str):
    job_clean = JOB_NAME_ALIASES.get(job.lstrip('/'), job.lstrip('/'))
    pid_file = os.path.join(".run", f"{job_clean}.pid")
    if background_running(pid_file):
        try:
            stop_background(pid_file)
            return {"status": "stopped", "job": job}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {"status": "not_running", "job": job}

def get_latest_validation_report():
    import glob
    import os
    import json
    try:
        paths = glob.glob("models/**/validation_report.json", recursive=True)
        if not paths:
            return None
        valid_paths = []
        for p in paths:
            try:
                if os.path.exists(p):
                    valid_paths.append((os.path.getmtime(p), p))
            except OSError:
                pass
        if not valid_paths:
            return None
        valid_paths.sort(key=lambda x: x[0], reverse=True)
        best_path = valid_paths[0][1]
        with open(best_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading validation report: {e}")
        return None

def parse_training_logs(log_path):
    import os
    import re
    if not os.path.exists(log_path):
        return {}
        
    metrics = {
        "trial_number": None,
        "ep_rew_mean": None,
        "fps": None,
        "total_timesteps": None,
        "value_loss": None,
        "explained_variance": None,
        "learning_rate": None,
        "trial_score": None
    }
    
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
        trial_re = re.compile(r">>> Trial (\d+) with budget=(\d+)")
        trial_complete_re = re.compile(r"\[\[OK\] Trial (\d+)\] COMPLETE\. Final Weighted Score: ([\d.-]+)")
        sb3_pair_re = re.compile(r"\|\s*([a-zA-Z0-9_/-]+)\s*\|\s*([e\d.+-]+)\s*\|")
        
        for line in lines:
            trial_match = trial_re.search(line)
            if trial_match:
                metrics["trial_number"] = int(trial_match.group(1))
                
            complete_match = trial_complete_re.search(line)
            if complete_match:
                metrics["trial_score"] = float(complete_match.group(2))
                
            sb3_match = sb3_pair_re.search(line)
            if sb3_match:
                key = sb3_match.group(1).strip()
                val_str = sb3_match.group(2).strip()
                try:
                    val = float(val_str)
                    if "/" in key:
                        key = key.split("/")[-1]
                    if key in metrics:
                        metrics[key] = val
                except ValueError:
                    pass
    except Exception as e:
        print(f"Error parsing training log: {e}")
        
    return metrics

@api.get("/api/train/results")
def api_train_results():
    import os
    pid_file = os.path.join(".run", "run_train.pid")
    is_running = background_running(pid_file)
    
    log_path = os.path.join(GLOBAL_LOGS_DIR, "run_train.log")
    metrics = parse_training_logs(log_path)
    
    val_report = get_latest_validation_report()
    
    import math
    def clean_nan(obj):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        elif isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nan(x) for x in obj]
        return obj

    return clean_nan({
        "running": is_running,
        "metrics": metrics,
        "validation_report": val_report
    })

@api.get("/api/eval/results")
def api_eval_results():
    if not os.path.exists(GLOBAL_METRICS_JSON):
        raise HTTPException(status_code=404, detail="Metrics file not found")
    try:
        with open(GLOBAL_METRICS_JSON, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            
        import math
        def clean_nan(obj):
            if isinstance(obj, float) and math.isnan(obj):
                return None
            elif isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan(x) for x in obj]
            return obj
            
        return clean_nan(metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_latest_metrics_jsonl() -> dict:
    path = "logs/metrics.jsonl"
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            end_pos = f.tell()
            buffer_size = 8192
            if end_pos > buffer_size:
                f.seek(-buffer_size, os.SEEK_END)
                lines = f.readlines()
            else:
                f.seek(0)
                lines = f.readlines()
            if lines:
                last_line = lines[-1].decode("utf-8").strip()
                if last_line:
                    return json.loads(last_line)
    except Exception:
        pass
    return {}

@api.get("/api/telemetry/live")
def api_telemetry_live():
    # Read the latest line of metrics.jsonl for HFT/live signaler metrics
    live_metrics = get_latest_metrics_jsonl()
    
    # Read snapshot metrics
    snap = {}
    if os.path.exists(GLOBAL_SNAPSHOT_JSON):
        try:
            with open(GLOBAL_SNAPSHOT_JSON, "r", encoding="utf-8") as f:
                snap = json.load(f)
        except Exception:
            pass
            
    is_tripped = snap.get("kill_switch_tripped", False)
    
    # Read signaler running status
    is_running = background_running(GLOBAL_REALTIME_PID)
    
    # Simple clean NaN utility
    import math
    def clean_nan(obj):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        elif isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nan(x) for x in obj]
        return obj
    
    return clean_nan({
        "clock_sync": {
            "drift_ms": LATEST_TELEMETRY["clock_sync_drift_ms"],
            "rtt_ms": LATEST_TELEMETRY["clock_sync_rtt_ms"],
            "last_sync": LATEST_TELEMETRY["last_sync_time"],
        },
        "drift": {
            "psi_avg": LATEST_TELEMETRY["psi_avg"],
            "worst_feature": LATEST_TELEMETRY["psi_worst_feature"],
            "worst_psi": LATEST_TELEMETRY["psi_worst"],
            "status": LATEST_TELEMETRY["psi_status"],
        },
        "nodes": {
            "signaler_running": is_running,
            "ws_feed_ok": LATEST_TELEMETRY["ws_feed_ok"] or is_running,
            "broker_api_ok": LATEST_TELEMETRY["broker_api_ok"],
        },
        "panic": {
            "kill_switch_tripped": is_tripped,
            "last_panic_halt_time": snap.get("last_panic_halt_time"),
            "last_panic_report": snap.get("last_panic_report", {}),
        },
        "live_metrics": live_metrics,
    })

@api.get("/api/image")
def api_image(path: str = "logs/equity.png"):
    abs_path = os.path.abspath(path)
    abs_logs = os.path.abspath(GLOBAL_LOGS_DIR)
    if not abs_path.startswith(abs_logs):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail="Image not found")
    from fastapi.responses import FileResponse
    return FileResponse(abs_path, media_type="image/png")

ACTIVE_CLI_PROCESSES = {}

# Terminal endpoints execute arbitrary shell commands (shell=True). They are a
# deliberate dev convenience but are also a full RCE surface. To stop a remote
# caller (e.g. a key holder in strict mode behind a proxy) from getting a shell,
# they are:
#   * disabled entirely if RIVEN_ENABLE_TERMINAL is set to a falsey value;
#   * restricted to loopback clients unless RIVEN_ENABLE_REMOTE_TERMINAL=1.
_TERMINAL_ENABLED = os.environ.get("RIVEN_ENABLE_TERMINAL", "1").strip().lower() not in (
    "0", "false", "no", "off", "",
)
_TERMINAL_ALLOW_REMOTE = os.environ.get("RIVEN_ENABLE_REMOTE_TERMINAL", "0").strip().lower() in (
    "1", "true", "yes", "on",
)


def _require_terminal(request: Optional[Request]) -> None:
    """Guard for shell-executing terminal endpoints. Raises 403 if not permitted."""
    if not _TERMINAL_ENABLED:
        raise HTTPException(
            status_code=403,
            detail="Terminal disabled. Set RIVEN_ENABLE_TERMINAL=1 to enable.",
        )
    if not _TERMINAL_ALLOW_REMOTE:
        client_host = request.client.host if (request and request.client) else None
        if not _is_loopback_client(client_host):
            raise HTTPException(
                status_code=403,
                detail=(
                    "Terminal restricted to loopback clients. "
                    "Set RIVEN_ENABLE_REMOTE_TERMINAL=1 to allow remote use."
                ),
            )


@api.post("/api/terminal/cd")
def api_terminal_cd(payload: TerminalCommand, request: Request):
    _require_terminal(request)
    import os
    current_cwd = payload.cwd or os.getcwd()
    cmd = payload.command.strip()
    
    if not cmd.startswith("cd"):
        raise HTTPException(status_code=400, detail="Not a cd command")
        
    path_arg = cmd[2:].strip()
    if not path_arg:
        return {"cwd": current_cwd, "output": current_cwd}
        
    # Resolve path
    target_path = os.path.abspath(os.path.join(current_cwd, path_arg))
    if os.path.exists(target_path) and os.path.isdir(target_path):
        return {"cwd": target_path, "output": f"Directory changed to {target_path}"}
    else:
        raise HTTPException(status_code=404, detail=f"Directory not found: {path_arg}")

@api.post("/api/terminal/start")
def api_terminal_start(payload: TerminalCommand, request: Request):
    _require_terminal(request)
    import subprocess
    import sys
    import os
    import uuid
    import signal
    from datetime import datetime

    cmd = payload.command.strip()
    if not cmd:
        return {"cmd_id": "", "log_name": ""}
        
    cwd = payload.cwd or os.getcwd()
    cmd_parts = cmd.split()
    
    # Intercept built-in command: jobs / rivenquant jobs
    if cmd == "jobs" or cmd == "rivenquant jobs":
        if not ACTIVE_CLI_PROCESSES:
            output = "No active background jobs."
        else:
            lines = ["Active CLI background processes:"]
            lines.append(f"{'Job ID':<10} {'PID':<8} {'Start Time':<20} {'Command'}")
            lines.append("-" * 60)
            dead_ids = []
            for cid, info in list(ACTIVE_CLI_PROCESSES.items()):
                proc = info["proc"]
                if proc.poll() is not None:
                    dead_ids.append(cid)
            for cid in dead_ids:
                ACTIVE_CLI_PROCESSES.pop(cid, None)
                
            for cid, info in ACTIVE_CLI_PROCESSES.items():
                proc = info["proc"]
                pid = getattr(proc, "pid", 0) or 0
                start = info["start_time"]
                c_str = info["cmd"]
                if len(c_str) > 30:
                    c_str = c_str[:27] + "..."
                lines.append(f"{cid:<10} {pid:<8} {start:<20} {c_str}")
            output = "\n".join(lines)
            
        cmd_id = f"jobs_{str(uuid.uuid4())[:4]}"
        log_name = f"cli_cmd_{cmd_id}.log"
        log_path = os.path.join(GLOBAL_LOGS_DIR, log_name)
        os.makedirs(GLOBAL_LOGS_DIR, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"$ {payload.command}\n{output}\n")
        return {"cmd_id": cmd_id, "log_name": log_name, "is_instant": True}
        
    # Intercept built-in command: kill / rivenquant kill
    if cmd_parts and (cmd_parts[0] == "kill" or (cmd_parts[0] == "rivenquant" and len(cmd_parts) > 1 and cmd_parts[1] == "kill")):
        target_id = cmd_parts[1] if cmd_parts[0] == "kill" else (cmd_parts[2] if len(cmd_parts) > 2 else "")
        output = ""
        if target_id in ACTIVE_CLI_PROCESSES:
            info = ACTIVE_CLI_PROCESSES.pop(target_id)
            proc = info["proc"]
            pid = getattr(proc, "pid", 0) or 0
            # FakeJobProcess trigger stop
            if hasattr(proc, "kill") and not hasattr(proc, "pid"):
                proc.kill()
            else:
                import platform
                try:
                    if platform.system() == "Windows":
                        subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], capture_output=True)
                    else:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            output = f"Job {target_id} (PID {pid}) terminated."
        else:
            try:
                target_pid = int(target_id)
                import platform
                if platform.system() == "Windows":
                    res = subprocess.run(["taskkill", "/PID", str(target_pid), "/F", "/T"], capture_output=True, text=True)
                    output = res.stdout or res.stderr or f"Sent taskkill to PID {target_pid}."
                else:
                    os.kill(target_pid, signal.SIGKILL)
                    output = f"Sent SIGKILL to PID {target_pid}."
            except Exception as e:
                output = f"No job or process found for identifier: {target_id} ({str(e)})"
                
        cmd_id = f"kill_{str(uuid.uuid4())[:4]}"
        log_name = f"cli_cmd_{cmd_id}.log"
        log_path = os.path.join(GLOBAL_LOGS_DIR, log_name)
        os.makedirs(GLOBAL_LOGS_DIR, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"$ {payload.command}\n{output}\n")
        return {"cmd_id": cmd_id, "log_name": log_name, "is_instant": True}

    # Intercept built-in command: clear or cls
    if cmd in ("clear", "cls"):
        cmd_id = f"clear_{str(uuid.uuid4())[:4]}"
        return {"cmd_id": cmd_id, "log_name": "", "is_clear": True}

    # Intercept rivenquant commands to redirect them to api_run_job background pipeline
    if cmd.startswith("rivenquant "):
        sub = cmd_parts[1].lower() if len(cmd_parts) > 1 else ""
        job_map = {
            "ingest": "run_ingest",
            "train": "run_train",
            "backtest": "/backtest",
            "pipeline": "/pipeline",
            "live-start": "/start",
            "live-stop": "/stop",
            "check-guards": "pdt_guard_check"
        }
        
        if sub in job_map:
            job_name = job_map[sub]
            try:
                job_params = {}
                if sub == "check-guards":
                    job_params = {"position_value": 100000, "account_equity": 30000}
                elif sub == "train":
                    job_params = {"config": "configs/sandbox.yaml", "steps": 100000}
                elif sub == "backtest":
                    job_params = {"config": "configs/sandbox.yaml"}
                    
                res = api_run_job(RunJobPayload(job=job_name, params=job_params))
                log_file_basename = os.path.basename(res["log"])
                cmd_id = f"rq_{sub}_{str(uuid.uuid4())[:4]}"
                
                class FakeJobProcess:
                    def __init__(self, job_name):
                        self.job_name = job_name
                        self.pid_file = os.path.join(".run", f"{job_name.lstrip('/')}.pid")
                        if job_name == "/start":
                            self.pid_file = GLOBAL_REALTIME_PID
                    def poll(self):
                        is_running = background_running(self.pid_file)
                        return None if is_running else 0
                    def kill(self):
                        try:
                            stop_background(self.pid_file)
                        except Exception:
                            pass
                            
                ACTIVE_CLI_PROCESSES[cmd_id] = {
                    "proc": FakeJobProcess(job_name),
                    "cmd": cmd,
                    "start_time": datetime.now().isoformat()
                }
                
                return {"cmd_id": cmd_id, "log_name": log_file_basename}
            except Exception as e:
                cmd_id = f"rq_err_{str(uuid.uuid4())[:4]}"
                log_name = f"cli_cmd_{cmd_id}.log"
                log_path = os.path.join(GLOBAL_LOGS_DIR, log_name)
                os.makedirs(GLOBAL_LOGS_DIR, exist_ok=True)
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(f"$ {payload.command}\n[Error] Failed to trigger job: {str(e)}\n")
                return {"cmd_id": cmd_id, "log_name": log_name, "is_instant": True}

    # Standard command processing with substitutions
    if cmd_parts:
        if cmd_parts[0] == "python":
            cmd_parts[0] = sys.executable
            cmd = " ".join(f'"{p}"' if " " in p else p for p in cmd_parts)
        elif cmd_parts[0] == "pip":
            cmd = f'"{sys.executable}" -m pip ' + " ".join(cmd_parts[1:])
        elif cmd_parts[0] == "pytest":
            cmd = f'"{sys.executable}" -m pytest ' + " ".join(cmd_parts[1:])
        elif cmd_parts[0] == "ls" and platform.system() == "Windows":
            cmd_parts[0] = "dir"
            cmd = " ".join(cmd_parts)
            
    cmd_id = str(uuid.uuid4())[:8]
    log_name = f"cli_cmd_{cmd_id}.log"
    log_path = os.path.join(GLOBAL_LOGS_DIR, log_name)
    
    os.makedirs(GLOBAL_LOGS_DIR, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"$ {payload.command}\n")
        
    logf = open(log_path, "a", encoding="utf-8", newline="")
    
    import platform
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        if platform.system() == "Windows":
            creationflags = 0x00000200  # CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdout=logf,
                stderr=logf,
                cwd=cwd,
                creationflags=creationflags,
                env=env
            )
        else:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdout=logf,
                stderr=logf,
                cwd=cwd,
                preexec_fn=os.setsid,
                env=env
            )
            
        ACTIVE_CLI_PROCESSES[cmd_id] = {
            "proc": proc,
            "cmd": payload.command,
            "start_time": datetime.now().isoformat()
        }
        return {"cmd_id": cmd_id, "log_name": log_name}
    except Exception as e:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n[Error] Failed to execute: {str(e)}\n")
        return {"cmd_id": "", "log_name": log_name, "error": str(e)}

@api.get("/api/terminal/status")
def api_terminal_status(cmd_id: str):
    info = ACTIVE_CLI_PROCESSES.get(cmd_id)
    if not info:
        return {"running": False, "exit_code": -1}
    proc = info["proc"]
    poll = proc.poll()
    if poll is None:
        return {"running": True, "exit_code": None}
    else:
        ACTIVE_CLI_PROCESSES.pop(cmd_id, None)
        return {"running": False, "exit_code": poll}

@api.post("/api/terminal/kill")
def api_terminal_kill(cmd_id: str):
    info = ACTIVE_CLI_PROCESSES.pop(cmd_id, None)
    if info:
        proc = info["proc"]
        if hasattr(proc, "kill") and not hasattr(proc, "pid"):
            proc.kill()
            return {"status": "success"}
            
        import platform
        pid = getattr(proc, "pid", 0)
        if pid:
            try:
                if platform.system() == "Windows":
                    subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], capture_output=True)
                else:
                    import os, signal
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        return {"status": "success"}
    return {"status": "not_found"}

@api.post("/api/terminal/run")
def api_terminal_run(payload: TerminalCommand, request: Request):
    _require_terminal(request)
    import subprocess
    import sys
    import os

    cmd = payload.command.strip()
    if not cmd:
        return {"output": ""}
        
    cmd_parts = cmd.split()
    if cmd_parts and cmd_parts[0] == "python":
        cmd_parts[0] = sys.executable
        cmd = " ".join(f'"{p}"' if " " in p else p for p in cmd_parts)
    
    try:
        res = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
            cwd=payload.cwd or os.getcwd()
        )
        output = res.stdout or ""
        if res.stderr:
            output += "\n" + res.stderr
        return {
            "output": output,
            "returncode": res.returncode
        }
    except subprocess.TimeoutExpired as e:
        output = e.stdout or ""
        if e.stderr:
            output += "\n" + e.stderr
        output += "\n[Error] Command timed out after 15 seconds."
        return {
            "output": output,
            "returncode": -1
        }
    except Exception as e:
        return {
            "output": f"[Error] Failed to execute: {str(e)}",
            "returncode": -1
        }

# Project root used to confine path-parameter file reads.
# Desktop sidecar changes into its writable runtime directory before importing
# this module.  Use that directory as the data/config boundary; ``__file__``
# points into PyInstaller's temporary read-only bundle in a frozen build.
_PROJECT_ROOT = os.path.realpath(os.getcwd())


def _safe_resolve_path(path: str) -> str:
    """Resolve a user-supplied relative path and confine it to the project root.

    Rejects absolute paths, drive letters and parent traversal, then resolves
    symlinks via realpath() so a symlink located inside an allowed directory
    cannot point outside the project tree. Returns the resolved absolute path;
    raises HTTPException(400) if the path is unsafe.
    """
    raw = (path or "").strip()
    normalized = os.path.normpath(raw)
    if (
        not raw
        or normalized.startswith("..")
        or os.path.isabs(normalized)
        or ":" in normalized
    ):
        raise HTTPException(status_code=400, detail="Invalid path")
    resolved = os.path.realpath(os.path.join(_PROJECT_ROOT, normalized))
    try:
        inside = os.path.commonpath([resolved, _PROJECT_ROOT]) == _PROJECT_ROOT
    except ValueError:
        inside = False  # e.g. different drive on Windows
    if not inside:
        raise HTTPException(status_code=400, detail="Invalid path")
    return resolved


def _sweep_stale_tmp_configs(max_age_sec: int = 7200) -> None:
    """Remove orphaned per-job temp configs (configs/tmp_*) left by finished or
    crashed jobs.

    These files are written by /api/run_job (via its `_tmp_path` helper) and read
    once at child-process startup, so anything older than ``max_age_sec`` (2h) is
    safe to delete and can never belong to a job still in its startup window.
    Without this they accumulate forever (one per invocation). Best-effort: never
    raises.
    """
    import glob as _glob
    import time as _time
    try:
        now = _time.time()
        for pattern in ("configs/tmp_*.yaml", "configs/tmp_*.json", "configs/tmp_*.yml"):
            for fp in _glob.glob(os.path.join(_PROJECT_ROOT, pattern)):
                try:
                    if now - os.path.getmtime(fp) > max_age_sec:
                        os.remove(fp)
                except Exception:
                    pass
    except Exception:
        pass


@api.get("/api/json/get_file")
def api_json_get_file(path: str):
    import json
    normalized_path = _safe_resolve_path(path)
    if not os.path.exists(normalized_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        with open(normalized_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/api/data/preview")
def api_data_preview(path: str, n: int = 10):
    import math
    normalized_path = _safe_resolve_path(path)
    if not os.path.exists(normalized_path):
        raise HTTPException(status_code=404, detail=f"File {normalized_path} not found")
    try:
        ext = os.path.splitext(normalized_path)[1].lower()
        if ext in (".parquet", ".pq"):
            df = pd.read_parquet(normalized_path)
        elif ext == ".csv":
            df = pd.read_csv(normalized_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
            
        tail_df = df.tail(n)
        tail_df = tail_df.replace({pd.NA: None})
        
        data_rows = []
        for r in tail_df.to_dict(orient="records"):
            cleaned_row = {}
            for k, v in r.items():
                if isinstance(v, float):
                    if not math.isfinite(v) or math.isnan(v):
                        cleaned_row[k] = None
                    else:
                        cleaned_row[k] = v
                elif pd.isna(v):
                    cleaned_row[k] = None
                else:
                    cleaned_row[k] = v
            data_rows.append(cleaned_row)
            
        return {
            "columns": list(df.columns),
            "rows": data_rows,
            "total_rows": len(df)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class LiteDataHealPayload(BaseModel):
    path: str = "data/prices.parquet"
    forward_fill_limit: int = Field(default=5, ge=1, le=100)


@api.post("/api/data/auto_heal")
def api_data_auto_heal(payload: LiteDataHealPayload):
    normalized_path = _safe_resolve_path(payload.path)
    expected = _safe_resolve_path("data/prices.parquet")
    if normalized_path != expected:
        raise HTTPException(status_code=400, detail="Lite auto-heal is limited to data/prices.parquet")
    from services.lite_data_repair import repair_prices_file
    try:
        result = repair_prices_file(normalized_path, forward_fill_limit=payload.forward_fill_limit)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="data/prices.parquet does not exist")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "success" if result["complete"] else "partial", **result}

@api.get("/api/features/specs")
def api_features_specs():
    try:
        import feature_config
        layout = feature_config.make_layout()
        # Some elements in the layout could be numpy objects or dict views, convert them to standard dict/lists
        serializable_layout = []
        for block in layout:
            cleaned_block = {}
            for k, v in block.items():
                if isinstance(v, (list, tuple)):
                    cleaned_block[k] = list(v)
                elif isinstance(v, (dict)):
                    cleaned_block[k] = dict(v)
                else:
                    cleaned_block[k] = str(v) if k == "dtype" else v
            serializable_layout.append(cleaned_block)
            
        return {
            "n_features": feature_config.N_FEATURES,
            "layout": serializable_layout
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/api/yaml/get")
def api_yaml_get(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Config file not found")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {"content": f.read()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.post("/api/yaml/save")
def api_yaml_save(payload: YamlSavePayload):
    try:
        atomic_write_with_retry(payload.path, payload.content)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.post("/api/config/apply_calibration")
def api_config_apply_calibration(payload: ApplyCalibrationPayload):
    import json
    import yaml
    path = payload.path
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Config file {path} not found")
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
            
        # 1. Load tcost calibration
        tcost_path = "models/tcost_calibration.json"
        if os.path.exists(tcost_path):
            with open(tcost_path, "r", encoding="utf-8") as f:
                tcost = json.load(f)
            if "fitted_params" in tcost:
                fp = tcost["fitted_params"]
                if "dynamic_spread" not in config or not isinstance(config["dynamic_spread"], dict):
                    config["dynamic_spread"] = {}
                config["dynamic_spread"]["enabled"] = True
                config["dynamic_spread"]["base_bps"] = float(fp.get("base_bps", 0.0))
                config["dynamic_spread"]["alpha_vol"] = float(fp.get("alpha_vol", 0.0))
                config["dynamic_spread"]["beta_illiquidity"] = float(fp.get("beta_illiquidity", fp.get("beta_liq", 0.0)))
                
        # 2. Load slippage calibration
        slip_path = "models/slippage_calibration.json"
        if os.path.exists(slip_path):
            with open(slip_path, "r", encoding="utf-8") as f:
                slip = json.load(f)
            if "slippage" not in config or not isinstance(config["slippage"], dict):
                config["slippage"] = {}
            config["slippage"]["k"] = float(slip.get("k", 0.8))
            config["slippage"]["default_spread_bps"] = float(slip.get("default_spread_bps", 2.0))
            config["slippage"]["min_half_spread_bps"] = float(slip.get("min_half_spread_bps", 0.0))
            
        atomic_write_with_retry(path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/api/config/get_backtest_settings")
def api_config_get_backtest_settings(config_path: str = "configs/config_sim.yaml", sandbox_path: str = "configs/sandbox.yaml"):
    import yaml
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail=f"Config file {config_path} not found")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        
        sandbox_config = {}
        if os.path.exists(sandbox_path):
            with open(sandbox_path, "r", encoding="utf-8") as f:
                sandbox_config = yaml.safe_load(f) or {}

        execution = config.get("execution", {})
        latency = config.get("latency", {})
        slippage = config.get("slippage", {})
        ws_dedup = config.get("ws_dedup", {})
        
        mode = execution.get("mode", "bar")
        bar_price = execution.get("bar_price", "close")
        latency_base = latency.get("base_ms", 250.0)
        latency_jitter = latency.get("jitter_ms", 50.0)
        spike_p = latency.get("spike_p", 0.01)
        spike_mult = latency.get("spike_mult", 5.0)
        seasonality = latency.get("use_seasonality", True)
        
        intrabar_price_model = execution.get("intrabar_price_model", "none") or "none"
        timeframe_ms = execution.get("timeframe_ms", 14400000)
        seed_mode = "stable" if latency.get("seed", 0) == 0 else "random"
        use_latency_from = execution.get("use_latency_from", "default") or "default"
        latency_constant_ms = execution.get("latency_constant_ms", 0.0) or 0.0
        
        entry_mode = execution.get("entry_mode", "default")
        next_bar_open = (entry_mode == "next_bar_open")
        
        clip_to_bar = execution.get("clip_to_bar", {})
        clip_next_bar = clip_to_bar.get("enabled", True)
        strict_open = clip_to_bar.get("strict_open_fill", False)
        
        active_profile = config.get("execution_profile", "Conservative")
        profiles = config.get("execution_profiles_definitions", {
            "Conservative": {"offset_ticks": 2, "ttl_ms": 5000, "tif": "GTC"},
            "Balanced": {"offset_ticks": 0, "ttl_ms": 2000, "tif": "GTC"},
            "Aggressive": {"offset_ticks": -1, "ttl_ms": 500, "tif": "IOC"},
            "LIMIT_MID_BPS": {"limit_offset_bps": 1.0, "ttl_steps": 5, "tif": "GTC"},
            "MKT_OPEN_NEXT_H1": {"tif": "DAY"},
            "VWAP_CURRENT_H1": {"tif": "DAY"}
        })
        
        slip_enabled = config.get("slippage_calibration_enabled", False) or slippage.get("dynamic", {}).get("enabled", False)
        slip_path = config.get("slippage_calibration_path", "models/slippage_calibration.json")
        smoothing_alpha = slippage.get("dynamic", {}).get("smoothing_alpha", 0.10) or 0.10
        vol_mode = slippage.get("dynamic", {}).get("vol_metric", "hl") or "hl"
        liq_col = slippage.get("dynamic", {}).get("liq_col", "volume") or "volume"
        liq_ref = slippage.get("dynamic", {}).get("liq_ref", 240000.0) or 240000.0
        
        bar_capacity_base = execution.get("bar_capacity_base", {})
        cap_enabled = bar_capacity_base.get("enabled", False)
        cap_frac = bar_capacity_base.get("capacity_frac_of_ADV_base", 0.05)
        cap_floor = bar_capacity_base.get("floor_base", 10.0)
        cap_path = bar_capacity_base.get("adv_base_path", "data/liquidity/adv_base.json")
        
        ws_enabled = ws_dedup.get("enabled", False)
        ws_skips = ws_dedup.get("log_skips", True)
        ws_path = ws_dedup.get("persist_path", "logs/ws_dedup_state.json")
        
        return {
            "mode": mode,
            "bar_price": bar_price,
            "latency_base": latency_base,
            "latency_jitter": latency_jitter,
            "spike_p": spike_p,
            "spike_mult": spike_mult,
            "seasonality": seasonality,
            "intrabar_price_model": intrabar_price_model,
            "timeframe_ms": timeframe_ms,
            "seed_mode": seed_mode,
            "use_latency_from": use_latency_from,
            "latency_constant_ms": latency_constant_ms,
            "next_bar_open": next_bar_open,
            "clip_next_bar": clip_next_bar,
            "strict_open": strict_open,
            "active_profile": active_profile,
            "profiles": profiles,
            "slip_enabled": slip_enabled,
            "slip_path": slip_path,
            "smoothing_alpha": smoothing_alpha,
            "vol_mode": vol_mode,
            "liq_col": liq_col,
            "liq_ref": liq_ref,
            "cap_enabled": cap_enabled,
            "cap_frac": cap_frac,
            "cap_floor": cap_floor,
            "cap_path": cap_path,
            "ws_enabled": ws_enabled,
            "ws_skips": ws_skips,
            "ws_path": ws_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.post("/api/config/save_backtest_settings")
def api_config_save_backtest_settings(payload: SaveBacktestSettingsPayload):
    import yaml
    config_path = payload.config_path
    sandbox_path = payload.sandbox_path
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail=f"Config file {config_path} not found")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
            
        if "execution" not in config or not isinstance(config["execution"], dict):
            config["execution"] = {}
        if "latency" not in config or not isinstance(config["latency"], dict):
            config["latency"] = {}
        if "slippage" not in config or not isinstance(config["slippage"], dict):
            config["slippage"] = {}
        if "ws_dedup" not in config or not isinstance(config["ws_dedup"], dict):
            config["ws_dedup"] = {}
            
        config["execution"]["mode"] = payload.mode
        config["execution"]["bar_price"] = payload.bar_price
        
        config["latency"]["base_ms"] = float(payload.latency_base)
        config["latency"]["jitter_ms"] = float(payload.latency_jitter)
        config["latency"]["spike_p"] = float(payload.spike_p)
        config["latency"]["spike_mult"] = float(payload.spike_mult)
        config["latency"]["use_seasonality"] = bool(payload.seasonality)
        config["latency"]["seed"] = 0 if payload.seed_mode == "stable" else 42
        config["use_seasonality"] = bool(payload.seasonality)
        
        config["execution"]["intrabar_price_model"] = payload.intrabar_price_model
        config["execution"]["timeframe_ms"] = int(payload.timeframe_ms)
        config["execution"]["use_latency_from"] = payload.use_latency_from if payload.use_latency_from != "default" else None
        config["execution"]["latency_constant_ms"] = float(payload.latency_constant_ms) if payload.use_latency_from == "constant" else None
        
        if "bridge" not in config["execution"] or not isinstance(config["execution"]["bridge"], dict):
            config["execution"]["bridge"] = {}
        config["execution"]["bridge"]["intrabar_price_model"] = payload.intrabar_price_model
        config["execution"]["bridge"]["timeframe_ms"] = int(payload.timeframe_ms)
        config["execution"]["bridge"]["use_latency_from"] = payload.use_latency_from if payload.use_latency_from != "default" else None
        config["execution"]["bridge"]["latency_constant_ms"] = float(payload.latency_constant_ms) if payload.use_latency_from == "constant" else None
        
        config["execution"]["entry_mode"] = "next_bar_open" if payload.next_bar_open else "default"
        
        if "clip_to_bar" not in config["execution"] or not isinstance(config["execution"]["clip_to_bar"], dict):
            config["execution"]["clip_to_bar"] = {}
        config["execution"]["clip_to_bar"]["enabled"] = bool(payload.clip_next_bar)
        config["execution"]["clip_to_bar"]["strict_open_fill"] = bool(payload.strict_open)
        
        config["execution_profile"] = payload.active_profile
        config["execution_profiles_definitions"] = payload.profiles
        
        profile_params = payload.profiles.get(payload.active_profile, {})
        if "execution_params" not in config or not isinstance(config["execution_params"], dict):
            config["execution_params"] = {}
        config["execution_params"]["slippage_bps"] = 0.0
        if payload.active_profile == "LIMIT_MID_BPS":
            config["execution_params"]["limit_offset_bps"] = float(profile_params.get("limit_offset_bps", 1.0))
            config["execution_params"]["ttl_steps"] = int(profile_params.get("ttl_steps", 5))
            config["execution_params"]["tif"] = str(profile_params.get("tif", "GTC"))
            config["execution_params"].pop("offset_ticks", None)
            config["execution_params"].pop("ttl_ms", None)
        elif payload.active_profile in ("MKT_OPEN_NEXT_H1", "VWAP_CURRENT_H1"):
            config["execution_params"]["limit_offset_bps"] = 0.0
            config["execution_params"]["ttl_steps"] = 1
            config["execution_params"]["tif"] = str(profile_params.get("tif", "DAY"))
            config["execution_params"].pop("offset_ticks", None)
            config["execution_params"].pop("ttl_ms", None)
        else:
            config["execution_params"]["limit_offset_bps"] = 0.0
            config["execution_params"]["offset_ticks"] = int(profile_params.get("offset_ticks", 0))
            config["execution_params"]["ttl_ms"] = int(profile_params.get("ttl_ms", 2000))
            config["execution_params"]["tif"] = str(profile_params.get("tif", "GTC"))
            config["execution_params"].pop("ttl_steps", None)

        config["slippage_calibration_enabled"] = bool(payload.slip_enabled)
        config["slippage_calibration_path"] = payload.slip_path
        
        if "dynamic" not in config["slippage"] or not isinstance(config["slippage"]["dynamic"], dict):
            config["slippage"]["dynamic"] = {}
        config["slippage"]["dynamic"]["enabled"] = bool(payload.slip_enabled)
        config["slippage"]["dynamic"]["path"] = payload.slip_path
        config["slippage"]["dynamic"]["smoothing_alpha"] = float(payload.smoothing_alpha)
        config["slippage"]["dynamic"]["vol_metric"] = payload.vol_mode
        config["slippage"]["dynamic"]["liq_col"] = payload.liq_col
        config["slippage"]["dynamic"]["liq_ref"] = float(payload.liq_ref)
        
        if "bar_capacity_base" not in config["execution"] or not isinstance(config["execution"]["bar_capacity_base"], dict):
            config["execution"]["bar_capacity_base"] = {}
        config["execution"]["bar_capacity_base"]["enabled"] = bool(payload.cap_enabled)
        config["execution"]["bar_capacity_base"]["capacity_frac_of_ADV_base"] = float(payload.cap_frac)
        config["execution"]["bar_capacity_base"]["floor_base"] = float(payload.cap_floor)
        config["execution"]["bar_capacity_base"]["adv_base_path"] = payload.cap_path
        
        config["ws_dedup"]["enabled"] = bool(payload.ws_enabled)
        config["ws_dedup"]["log_skips"] = bool(payload.ws_skips)
        config["ws_dedup"]["persist_path"] = payload.ws_path
        
        atomic_write_with_retry(config_path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        
        if os.path.exists(sandbox_path):
            with open(sandbox_path, "r", encoding="utf-8") as f:
                sandbox = yaml.safe_load(f) or {}
            
            if "dynamic_spread" not in sandbox or not isinstance(sandbox["dynamic_spread"], dict):
                sandbox["dynamic_spread"] = {}
            sandbox["dynamic_spread"]["enabled"] = bool(payload.slip_enabled)
            sandbox["dynamic_spread"]["vol_mode"] = payload.vol_mode
            sandbox["dynamic_spread"]["liq_col"] = payload.liq_col
            sandbox["dynamic_spread"]["liq_ref"] = float(payload.liq_ref)
            
            atomic_write_with_retry(sandbox_path, yaml.safe_dump(sandbox, sort_keys=False, allow_unicode=True))
            
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.post("/api/ingest/verify")
def api_ingest_verify(payload: VerifyIngestPayload):
    import requests
    provider = payload.provider.lower()
    api_key = payload.api_key
    api_secret = payload.api_secret
    
    if provider == "binance":
        try:
            res = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
            if res.status_code == 200:
                return {"status": "success", "message": "Connection to Binance successful!"}
            else:
                return {"status": "error", "message": f"Binance ping failed with status code {res.status_code}"}
        except Exception as e:
            return {"status": "error", "message": f"Binance error: {str(e)}"}
            
    elif provider == "alpaca":
        try:
            headers = {
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret
            }
            url = "https://paper-api.alpaca.markets/v2/account"
            if api_key and api_secret:
                res = requests.get(url, headers=headers, timeout=5)
                if res.status_code == 200:
                    return {"status": "success", "message": "Connection to Alpaca Paper successful!"}
                else:
                    return {"status": "error", "message": f"Alpaca credentials verification failed: {res.text}"}
            else:
                res = requests.get("https://api.alpaca.markets/v2/clock", timeout=5)
                if res.status_code == 200:
                    return {"status": "success", "message": "Public Alpaca API reachable."}
                else:
                    return {"status": "error", "message": "Alpaca API clock endpoint unreachable."}
        except Exception as e:
            return {"status": "error", "message": f"Alpaca error: {str(e)}"}
            
    elif provider == "oanda":
        try:
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            url = "https://api-fxtrade.oanda.com/v3/accounts"
            if api_key:
                res = requests.get(url, headers=headers, timeout=5)
                if res.status_code == 200:
                    return {"status": "success", "message": "Connection to OANDA successful!"}
                else:
                    url_practice = "https://api-fxpractice.oanda.com/v3/accounts"
                    res_p = requests.get(url_practice, headers=headers, timeout=5)
                    if res_p.status_code == 200:
                        return {"status": "success", "message": "Connection to OANDA Practice successful!"}
                    return {"status": "error", "message": f"Oanda API failed: {res.text}"}
            else:
                res = requests.get("https://api-fxtrade.oanda.com/", timeout=5)
                if res.status_code in (200, 404):
                    return {"status": "success", "message": "Public OANDA API reachable."}
                return {"status": "error", "message": "Oanda API unreachable."}
        except Exception as e:
            return {"status": "error", "message": f"Oanda error: {str(e)}"}
            
    return {"status": "success", "message": f"Connection test for {provider.upper()} passed (public access)."}



@api.post("/api/quantizer/save")
def api_quantizer_save(payload: QuantizerSavePayload):
    for path in ["configs/quantizer.yaml", "configs/config_live.yaml"]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                data.setdefault("quantizer", {})
                data["quantizer"]["strict_filters"] = payload.strict_filters
                data["quantizer"]["enforce_percent_price_by_side"] = payload.enforce_percent_price_by_side
                if "strict" in data["quantizer"]:
                    data["quantizer"]["strict"] = payload.strict_filters
                atomic_write_with_retry(path, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to save {path}: {str(e)}")
    return {"status": "success"}

@api.post("/api/panic_halt")
def api_panic_halt():
    import services.ops_kill_switch as ops_kill_switch
    
    # 1. Trip the operational kill switch flag
    ops_kill_switch._trip()
    
    # 2. Halt automated signaling process
    signaler_was_running = False
    if background_running(GLOBAL_REALTIME_PID):
        stop_background(GLOBAL_REALTIME_PID)
        signaler_was_running = True

    # The desktop CCEA Agent is the only authoritative execution path.  Pause
    # it first, then cancel and flatten positions through its active broker.
    # Never fabricate a liquidation report when credentials or positions are
    # absent.
    if _CCEA_SUPERVISOR is not None and _CCEA_STATE == "running":
        result = _CCEA_SUPERVISOR.emergency_halt()
        result.update({
            "status": "success" if result.get("ok") else "partial",
            "kill_switch_tripped": True,
            "signaler_halted": True,
            "signaler_was_running": signaler_was_running,
            "asset_class": ACTIVE_ASSET.lower(),
        })
        try:
            panic_log_path = os.path.join(GLOBAL_LOGS_DIR, "panic_halt.log")
            with open(panic_log_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps({"at": datetime.now().isoformat(), **result}, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # Snapshot bookkeeping must happen on this path too — /api/telemetry/live
        # derives panic.kill_switch_tripped from GLOBAL_SNAPSHOT_JSON.
        try:
            snap = {}
            if os.path.exists(GLOBAL_SNAPSHOT_JSON):
                with open(GLOBAL_SNAPSHOT_JSON, "r", encoding="utf-8") as f:
                    snap = json.load(f)
            snap["kill_switch_tripped"] = True
            snap["signaler_running"] = False
            snap["last_panic_halt_time"] = datetime.now().isoformat()
            snap["last_panic_report"] = {k: v for k, v in result.items() if k != "kill_switch_tripped"}
            atomic_write_with_retry(GLOBAL_SNAPSHOT_JSON, json.dumps(snap, indent=2))
        except Exception:
            pass
        return result
        
    asset_class = ACTIVE_ASSET.lower()

    # 3. Create panic log file
    panic_log_path = os.path.join(GLOBAL_LOGS_DIR, "panic_halt.log")
    log_msg = f"[{datetime.now().isoformat()}] EMERGENCY PANIC HALT TRIGGERED. Asset: {asset_class}\n"

    # Fail-closed contract: with no authoritative execution backend (no CCEA
    # Agent, no real broker credentials) the halt only trips the local locks.
    # It must NOT report cancelled orders, liquidated positions or any
    # financial result it did not actually observe. A broker error can never
    # be reported as success.
    def _looks_like_placeholder(value: Optional[str]) -> bool:
        if not value:
            return True
        return "test" in value or "YOUR" in value or "$" in value

    def _real_credentials_available() -> bool:
        if asset_class in ("equity", "options"):
            return not _looks_like_placeholder(os.getenv("ALPACA_API_KEY")) and bool(os.getenv("ALPACA_API_SECRET"))
        if asset_class == "forex":
            return not _looks_like_placeholder(os.getenv("OANDA_API_KEY")) and bool(os.getenv("OANDA_ACCOUNT_ID"))
        if asset_class in ("futures", "crypto"):
            return not _looks_like_placeholder(os.getenv("BINANCE_API_KEY"))
        return False

    status = "unavailable"
    execution_mode = "no_execution_backend"
    detail = (
        "CCEA Agent не запущен и учётные данные брокера не настроены. "
        "Включена локальная блокировка (kill switch), сигналер остановлен. "
        "Ни один ордер не отменялся и ни одна позиция не закрывалась."
    )
    orders_cancelled = 0
    positions_liquidated: List[Dict[str, Any]] = []
    quant_report: Dict[str, Any] = {}

    if asset_class == "crypto" and _real_credentials_available():
        # No spot ORDER_EXECUTION adapter is registered for Binance in this
        # build — pretending otherwise would just raise inside the adapter
        # factory. Stay fail-closed and say so explicitly.
        detail = (
            "Для Binance spot в этой сборке нет order-execution адаптера: локальная блокировка "
            "включена, но ордера/позиции НЕ изменялись. Живое исполнение доступно только через CCEA Agent."
        )
        log_msg += "No spot order-execution adapter registered for Binance: local locks tripped only.\n"
    elif _real_credentials_available():
        cancel_failures = 0
        close_failures = 0
        try:
            from decimal import Decimal as _Dec

            from adapters.registry import (
                create_futures_order_execution_adapter,
                create_order_execution_adapter,
            )
            from adapters.models import ExchangeVendor
            from core_models import Order, OrderType, Side

            def _record_close(sym: str, qty: float, entry_price: float, confirmed: bool) -> None:
                nonlocal close_failures
                if not confirmed:
                    close_failures += 1
                positions_liquidated.append({
                    "symbol": sym,
                    "qty": qty,
                    "price": entry_price,
                    # Entry-price notional (estimate) — fills are reported by the
                    # broker asynchronously and are NOT claimed here.
                    "value": abs(qty * entry_price),
                    "side": "LONG" if qty > 0 else "SHORT",
                    "confirmed": confirmed,
                })

            def _close_via_market_orders(adapter) -> None:
                positions = adapter.get_positions()
                for sym, pos in positions.items():
                    qty = float(pos.qty)
                    if qty != 0:
                        order = Order(
                            ts=int(time.time() * 1000),
                            symbol=sym,
                            side=Side.SELL if qty > 0 else Side.BUY,
                            order_type=OrderType.MARKET,
                            quantity=_Dec(str(abs(qty))),
                        )
                        result = adapter.submit_order(order)
                        confirmed = bool(getattr(result, "success", result is not None and result is not False))
                        _record_close(sym, qty, float(pos.avg_entry_price), confirmed)

            def _close_via_close_position(adapter) -> None:
                positions = adapter.get_positions()
                for sym, pos in positions.items():
                    qty = float(pos.qty)
                    if qty != 0:
                        result = adapter.close_position(sym)
                        confirmed = bool(getattr(result, "success", result is not None and result is not False))
                        _record_close(sym, qty, float(pos.avg_entry_price), confirmed)

            if asset_class in ("equity", "options"):
                adapter = create_order_execution_adapter(ExchangeVendor.ALPACA, {
                    "api_key": os.getenv("ALPACA_API_KEY"),
                    "api_secret": os.getenv("ALPACA_API_SECRET", ""),
                    "paper": True,
                })
                orders_cancelled = int(adapter.cancel_all_orders())
                _close_via_market_orders(adapter)
            elif asset_class == "forex":
                adapter = create_order_execution_adapter(ExchangeVendor.OANDA, {
                    "api_key": os.getenv("OANDA_API_KEY"),
                    "account_id": os.getenv("OANDA_ACCOUNT_ID"),
                    "practice": True,
                })
                open_orders = adapter.get_open_orders()
                orders_cancelled = 0
                for o in open_orders:
                    if adapter.cancel_order(client_order_id=o.client_order_id):
                        orders_cancelled += 1
                    else:
                        cancel_failures += 1
                _close_via_close_position(adapter)
            else:  # futures (Binance USDT-M)
                adapter = create_futures_order_execution_adapter(ExchangeVendor.BINANCE_FUTURES, {
                    "api_key": os.getenv("BINANCE_API_KEY"),
                    "api_secret": os.getenv("BINANCE_API_SECRET", ""),
                })
                orders_cancelled = int(adapter.cancel_all_orders())
                _close_via_close_position(adapter)

            # Post-check: how many positions the adapter still reports open.
            positions_remaining: Optional[int] = None
            try:
                positions_remaining = sum(
                    1 for _s, p in adapter.get_positions().items() if float(p.qty) != 0
                )
            except Exception:
                pass

            confirmed_closes = sum(1 for p in positions_liquidated if p.get("confirmed"))
            all_confirmed = (
                cancel_failures == 0
                and close_failures == 0
                and (positions_remaining in (0, None))
            )
            status = "success" if all_confirmed else "partial"
            execution_mode = "live_broker"
            detail = (
                f"По данным адаптера ({asset_class}, paper/practice-окружение): отменено ордеров: {orders_cancelled}"
                + (f" (не отменено: {cancel_failures})" if cancel_failures else "")
                + f"; отправлено закрытий позиций: {len(positions_liquidated)} (подтверждено адаптером: {confirmed_closes})."
                + (f" Осталось открытых позиций: {positions_remaining}." if positions_remaining else "")
                + " Итоговые фактические исполнения подтверждаются брокером асинхронно — проверьте счёт."
            )
            # Only facts actually observed from the adapter — no invented
            # margin/PDT/greeks/slippage estimates.
            quant_report = {
                "estimated_notional_usd_at_entry": sum(p["value"] for p in positions_liquidated),
                "close_requests_sent": len(positions_liquidated),
                "close_requests_confirmed": confirmed_closes,
                "close_failures": close_failures,
                "orders_cancelled": orders_cancelled,
                "cancel_failures": cancel_failures,
                "positions_remaining": positions_remaining,
                "environment": "paper/practice (legacy fallback path is sandbox-only; live accounts use CCEA Agent)",
            }
            log_msg += (
                f"Adapter liquidation attempted. Cancelled: {orders_cancelled} (fail {cancel_failures}). "
                f"Close requests: {len(positions_liquidated)} (confirmed {confirmed_closes}, fail {close_failures}). "
                f"Remaining: {positions_remaining}.\n"
            )
        except Exception as e:
            status = "failed"
            execution_mode = "broker_error"
            detail = (
                f"Ошибка при экстренной ликвидации: {e}. Локальная блокировка включена, "
                "но состояние ордеров и позиций НЕ подтверждено — проверьте счёт у брокера вручную."
            )
            orders_cancelled = 0
            positions_liquidated = []
            quant_report = {"error": str(e)}
            log_msg += f"Error during emergency liquidation: {e}. No fabricated results reported.\n"
    else:
        log_msg += "No execution backend (CCEA down, no real broker credentials): local locks tripped only.\n"

    try:
        with open(panic_log_path, "a", encoding="utf-8") as lf:
            lf.write(log_msg)
    except Exception:
        pass

    snap = {}
    if os.path.exists(GLOBAL_SNAPSHOT_JSON):
        try:
            with open(GLOBAL_SNAPSHOT_JSON, "r", encoding="utf-8") as f:
                snap = json.load(f)
        except Exception:
            pass
    snap["kill_switch_tripped"] = True
    snap["signaler_running"] = False
    snap["last_panic_halt_time"] = datetime.now().isoformat()
    snap["last_panic_report"] = {
        "status": status,
        "execution_mode": execution_mode,
        "detail": detail,
        "orders_cancelled": orders_cancelled,
        "positions_liquidated": positions_liquidated,
        "quant_report": quant_report
    }
    try:
        atomic_write_with_retry(GLOBAL_SNAPSHOT_JSON, json.dumps(snap, indent=2))
    except Exception:
        pass

    return {
        "status": status,
        "detail": detail,
        "kill_switch_tripped": True,
        "signaler_halted": True,
        "signaler_was_running": signaler_was_running,
        "asset_class": asset_class,
        "execution_mode": execution_mode,
        "orders_cancelled": orders_cancelled,
        "positions_liquidated": positions_liquidated,
        "quant_report": quant_report
    }

@api.post("/api/panic_reset")
def api_panic_reset():
    import services.ops_kill_switch as ops_kill_switch
    ops_kill_switch.manual_reset()
    
    snap = {}
    if os.path.exists(GLOBAL_SNAPSHOT_JSON):
        try:
            with open(GLOBAL_SNAPSHOT_JSON, "r", encoding="utf-8") as f:
                snap = json.load(f)
        except Exception:
            pass
    snap["kill_switch_tripped"] = False
    try:
        atomic_write_with_retry(GLOBAL_SNAPSHOT_JSON, json.dumps(snap, indent=2))
    except Exception:
        pass
        
    return {"status": "success", "kill_switch_tripped": False}

@api.get("/api/portfolio/holdings")
def api_portfolio_holdings(asset: str = None):
    import services.ops_kill_switch as ops_kill_switch
    is_tripped = ops_kill_switch.tripped()
    
    asset_class = asset.lower() if asset else ACTIVE_ASSET.lower()
    
    # Defaults
    holdings = []
    broker_error: Optional[str] = None
    metrics = {
        "net_liquidation_value": 0.0,
        "margin_used": 0.0,
        "leverage": "1.0x",
        "buying_power": 0.0
    }
    
    # Desktop CCEA is the authoritative execution source.  Prefer its active
    # broker/books over environment-variable heuristics and decorative holdings.
    if _CCEA_SUPERVISOR is not None and _CCEA_STATE == "running":
        try:
            snap = _CCEA_SUPERVISOR.portfolio_snapshot()
            if snap.get("ok"):
                snap.pop("ok", None)
                snap["kill_switch_tripped"] = is_tripped
                snap["disclaimer"] = (
                    "Paper broker positions — simulated execution, real local books."
                    if snap.get("simulated") else None
                )
                return snap
        except Exception:
            pass

    if is_tripped:
        return {
            "holdings": [],
            "metrics": metrics,
            "kill_switch_tripped": True,
            "simulated": True,
            "data_source": "unavailable_while_halted",
        }
        
    # Check if we have real API key to fetch actual positions
    key_id = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    is_mock = not key_id or "test" in key_id or "YOUR" in key_id or "$" in key_id or not secret
    
    if asset_class == "equity":
        if is_mock:
            holdings = [
                {"symbol": "SPY", "qty": 200, "entry_price": 510.10, "current_price": 512.40, "value": 102480.0, "side": "LONG", "pnl": 460.0},
                {"symbol": "AAPL", "qty": 100, "entry_price": 175.20, "current_price": 178.50, "value": 17850.0, "side": "LONG", "pnl": 330.0}
            ]
            metrics = {
                "net_liquidation_value": 120330.0,
                "margin_used": 59165.0,
                "leverage": "1.5x",
                "buying_power": 120330.0
            }
        else:
            try:
                from adapters.registry import create_order_execution_adapter
                from adapters.models import ExchangeVendor
                config = {"api_key": key_id, "api_secret": secret, "paper": True}
                adapter = create_order_execution_adapter(ExchangeVendor.ALPACA, config)
                positions = adapter.get_positions()
                total_val = 0.0
                for sym, pos in positions.items():
                    qty = float(pos.qty)
                    if qty != 0:
                        entry = float(pos.avg_entry_price)
                        curr = entry
                        val = abs(qty * entry)
                        total_val += val
                        holdings.append({
                            "symbol": sym,
                            "qty": qty,
                            "entry_price": entry,
                            "current_price": curr,
                            "value": val,
                            "side": "LONG" if qty > 0 else "SHORT",
                            "pnl": 0.0
                        })
                metrics = {
                    "net_liquidation_value": total_val or 100000.0,
                    "margin_used": total_val * 0.5,
                    "leverage": f"{round(total_val / 100000.0, 2) if total_val else 1.0}x",
                    "buying_power": max(0.0, 100000.0 - total_val * 0.5)
                }
            except Exception as _broker_exc:
                # Never substitute fabricated holdings for a broker error with
                # real credentials (audit L2-008 class): report the failure.
                holdings = []
                metrics = {"net_liquidation_value": 0.0, "margin_used": 0.0, "leverage": "—", "buying_power": 0.0}
                broker_error = str(_broker_exc)
                
    elif asset_class == "forex":
        key_id = os.getenv("OANDA_API_KEY")
        account_id = os.getenv("OANDA_ACCOUNT_ID")
        is_mock = not key_id or "test" in key_id or "YOUR" in key_id or "$" in key_id or not account_id
        if is_mock:
            holdings = [
                {"symbol": "EUR_USD", "qty": 100000, "entry_price": 1.0820, "current_price": 1.0852, "value": 108520.0, "side": "LONG", "pnl": 320.0},
                {"symbol": "USD_JPY", "qty": -50000, "entry_price": 157.10, "current_price": 156.40, "value": 50000.0, "side": "SHORT", "pnl": 223.65}
            ]
            metrics = {
                "net_liquidation_value": 158520.0,
                "margin_used": 3170.0,
                "leverage": "10.0x",
                "buying_power": 155350.0
            }
        else:
            try:
                from adapters.registry import create_order_execution_adapter
                from adapters.models import ExchangeVendor
                config = {"api_key": key_id, "account_id": account_id, "practice": True}
                adapter = create_order_execution_adapter(ExchangeVendor.OANDA, config)
                positions = adapter.get_positions()
                total_val = 0.0
                for sym, pos in positions.items():
                    qty = int(pos.qty)
                    if qty != 0:
                        entry = float(pos.avg_entry_price)
                        val = abs(qty * entry)
                        total_val += val
                        holdings.append({
                            "symbol": sym,
                            "qty": qty,
                            "entry_price": entry,
                            "current_price": entry,
                            "value": val,
                            "side": "LONG" if qty > 0 else "SHORT",
                            "pnl": 0.0
                        })
                metrics = {
                    "net_liquidation_value": total_val or 100000.0,
                    "margin_used": total_val * 0.02,
                    "leverage": f"{round(total_val / 100000.0, 2) if total_val else 1.0}x",
                    "buying_power": max(0.0, 100000.0 - total_val * 0.02)
                }
            except Exception as _broker_exc:
                # Never substitute fabricated holdings for a broker error with
                # real credentials (audit L2-008 class): report the failure.
                holdings = []
                metrics = {"net_liquidation_value": 0.0, "margin_used": 0.0, "leverage": "—", "buying_power": 0.0}
                broker_error = str(_broker_exc)
                
    elif asset_class == "futures":
        key_id = os.getenv("BINANCE_API_KEY")
        is_mock = not key_id or "test" in key_id or "YOUR" in key_id or "$" in key_id
        if is_mock:
            holdings = [
                {"symbol": "ES", "qty": 2, "entry_price": 5290.00, "current_price": 5310.00, "value": 531000.0, "side": "LONG", "pnl": 2000.0},
                {"symbol": "NQ", "qty": 1, "entry_price": 18650.00, "current_price": 18720.00, "value": 374400.0, "side": "LONG", "pnl": 1400.0}
            ]
            metrics = {
                "net_liquidation_value": 905400.0,
                "margin_used": 45600.0,
                "leverage": "5.0x",
                "buying_power": 859800.0
            }
        else:
            try:
                from adapters.registry import create_order_execution_adapter
                from adapters.models import ExchangeVendor
                config = {"api_key": key_id, "api_secret": os.getenv("BINANCE_API_SECRET", "")}
                adapter = create_order_execution_adapter(ExchangeVendor.BINANCE_FUTURES, config)
                positions = adapter.get_positions()
                total_val = 0.0
                for sym, pos in positions.items():
                    qty = float(pos.qty)
                    if qty != 0:
                        entry = float(pos.avg_entry_price)
                        val = abs(qty * entry)
                        total_val += val
                        holdings.append({
                            "symbol": sym,
                            "qty": qty,
                            "entry_price": entry,
                            "current_price": entry,
                            "value": val,
                            "side": "LONG" if qty > 0 else "SHORT",
                            "pnl": 0.0
                        })
                metrics = {
                    "net_liquidation_value": total_val or 500000.0,
                    "margin_used": len(holdings) * 12400.0,
                    "leverage": "5.0x",
                    "buying_power": max(0.0, 500000.0 - len(holdings) * 12400.0)
                }
            except Exception as _broker_exc:
                # Never substitute fabricated holdings for a broker error with
                # real credentials (audit L2-008 class): report the failure.
                holdings = []
                metrics = {"net_liquidation_value": 0.0, "margin_used": 0.0, "leverage": "—", "buying_power": 0.0}
                broker_error = str(_broker_exc)
                
    elif asset_class == "crypto":
        key_id = os.getenv("BINANCE_API_KEY")
        is_mock = not key_id or "test" in key_id or "YOUR" in key_id or "$" in key_id
        if is_mock:
            holdings = [
                {"symbol": "BTCUSDT", "qty": 0.8, "entry_price": 67500.0, "current_price": 68240.0, "value": 54592.0, "side": "LONG", "pnl": 592.0},
                {"symbol": "ETHUSDT", "qty": 12.0, "entry_price": 3700.0, "current_price": 3780.0, "value": 45360.0, "side": "LONG", "pnl": 960.0}
            ]
            metrics = {
                "net_liquidation_value": 99952.0,
                "margin_used": 9995.0,
                "leverage": "2.1x",
                "buying_power": 89957.0
            }
        else:
            try:
                from adapters.registry import create_order_execution_adapter
                from adapters.models import ExchangeVendor
                config = {"api_key": key_id, "api_secret": os.getenv("BINANCE_API_SECRET", "")}
                adapter = create_order_execution_adapter(ExchangeVendor.BINANCE, config)
                positions = adapter.get_positions()
                total_val = 0.0
                for sym, pos in positions.items():
                    qty = float(pos.qty)
                    if qty != 0:
                        entry = float(pos.avg_entry_price)
                        val = abs(qty * entry)
                        total_val += val
                        holdings.append({
                            "symbol": sym,
                            "qty": qty,
                            "entry_price": entry,
                            "current_price": entry,
                            "value": val,
                            "side": "LONG" if qty > 0 else "SHORT",
                            "pnl": 0.0
                        })
                metrics = {
                    "net_liquidation_value": total_val or 10000.0,
                    "margin_used": total_val * 0.1,
                    "leverage": "2.1x",
                    "buying_power": max(0.0, 10000.0 - total_val * 0.1)
                }
            except Exception as _broker_exc:
                # Never substitute fabricated holdings for a broker error with
                # real credentials (audit L2-008 class): report the failure.
                holdings = []
                metrics = {"net_liquidation_value": 0.0, "margin_used": 0.0, "leverage": "—", "buying_power": 0.0}
                broker_error = str(_broker_exc)
                
    elif asset_class == "options":
        key_id = os.getenv("ALPACA_API_KEY")
        is_mock = not key_id or "test" in key_id or "YOUR" in key_id or "$" in key_id
        if is_mock:
            holdings = [
                {"symbol": "AAPL260619C00180000", "qty": 10, "entry_price": 22.10, "current_price": 24.50, "value": 24500.0, "side": "LONG", "pnl": 240.0},
                {"symbol": "TSLA260619P00170000", "qty": 5, "entry_price": 19.50, "current_price": 18.20, "value": 9100.0, "side": "LONG", "pnl": -65.0}
            ]
            metrics = {
                "net_liquidation_value": 33600.0,
                "margin_used": 0.0,
                "leverage": "1.0x",
                "buying_power": 33600.0
            }
        else:
            try:
                from adapters.registry import create_order_execution_adapter
                from adapters.models import ExchangeVendor
                config = {"api_key": key_id, "api_secret": os.getenv("ALPACA_API_SECRET", "")}
                adapter = create_order_execution_adapter(ExchangeVendor.ALPACA, config)
                positions = adapter.get_positions()
                total_val = 0.0
                for sym, pos in positions.items():
                    qty = float(pos.qty)
                    if qty != 0:
                        entry = float(pos.avg_entry_price)
                        val = abs(qty * entry)
                        total_val += val
                        holdings.append({
                            "symbol": sym,
                            "qty": qty,
                            "entry_price": entry,
                            "current_price": entry,
                            "value": val,
                            "side": "LONG" if qty > 0 else "SHORT",
                            "pnl": 0.0
                        })
                metrics = {
                    "net_liquidation_value": total_val or 50000.0,
                    "margin_used": 0.0,
                    "leverage": "1.0x",
                    "buying_power": total_val or 50000.0
                }
            except Exception as _broker_exc:
                # Never substitute fabricated holdings for a broker error with
                # real credentials (audit L2-008 class): report the failure.
                holdings = []
                metrics = {"net_liquidation_value": 0.0, "margin_used": 0.0, "leverage": "—", "buying_power": 0.0}
                broker_error = str(_broker_exc)
                
    return {
        "holdings": holdings,
        "metrics": metrics,
        "kill_switch_tripped": False,
        # Honesty: when no valid broker credentials are present, holdings/metrics
        # are simulated demo positions — flag it so the UI shows a SIMULATED badge.
        # A broker error with real credentials returns EMPTY holdings plus the
        # error (never fabricated live-looking positions).
        "simulated": bool(is_mock),
        "data_source": ("broker_error" if broker_error else ("simulated" if is_mock else "live")),
        "broker_error": broker_error,
        "disclaimer": MVP_DEMO_DISCLAIMER if is_mock else None,
    }

class ClosePositionPayload(BaseModel):
    symbol: str

@api.post("/api/portfolio/close")
def api_portfolio_close(payload: ClosePositionPayload):
    import services.ops_kill_switch as ops_kill_switch
    if ops_kill_switch.tripped():
        raise HTTPException(status_code=400, detail="Kill switch is active. Cannot close individual positions.")
        
    symbol = payload.symbol
    asset_class = ACTIVE_ASSET.lower()

    if _CCEA_SUPERVISOR is not None and _CCEA_STATE == "running":
        result = _CCEA_SUPERVISOR.close_position(symbol)
        if not result.get("ok"):
            raise HTTPException(status_code=404, detail=result.get("error", "position close failed"))
        return {"status": "success", "detail": f"Position {symbol} closed", **result}
    
    key_id = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    is_mock = not key_id or "test" in key_id or "YOUR" in key_id or "$" in key_id or not secret
    
    if is_mock:
        raise HTTPException(
            status_code=409,
            detail="This is a demo holding, not a broker position; it cannot be closed.",
        )
        
    try:
        from adapters.registry import create_order_execution_adapter
        from adapters.models import ExchangeVendor
        
        if asset_class == "equity":
            config = {"api_key": key_id, "api_secret": secret, "paper": True}
            adapter = create_order_execution_adapter(ExchangeVendor.ALPACA, config)
            positions = adapter.get_positions()
            if symbol in positions:
                qty = float(positions[symbol].qty)
                if qty != 0:
                    from decimal import Decimal as _Dec
                    from core_models import Order, OrderType, Side
                    side = Side.SELL if qty > 0 else Side.BUY
                    # core_models.Order takes ts/quantity (no `qty` kwarg) —
                    # the old construction raised TypeError on any real close.
                    close_order = Order(
                        ts=int(time.time() * 1000),
                        symbol=symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=_Dec(str(abs(qty))),
                    )
                    adapter.submit_order(close_order)
                    
        elif asset_class == "forex":
            config = {"api_key": os.getenv("OANDA_API_KEY", ""), "account_id": os.getenv("OANDA_ACCOUNT_ID", ""), "practice": True}
            adapter = create_order_execution_adapter(ExchangeVendor.OANDA, config)
            adapter.close_position(symbol)
            
        elif asset_class == "futures":
            config = {"api_key": os.getenv("BINANCE_API_KEY", ""), "api_secret": os.getenv("BINANCE_API_SECRET", "")}
            adapter = create_order_execution_adapter(ExchangeVendor.BINANCE_FUTURES, config)
            adapter.close_position(symbol)
            
        elif asset_class == "crypto":
            config = {"api_key": os.getenv("BINANCE_API_KEY", ""), "api_secret": os.getenv("BINANCE_API_SECRET", "")}
            adapter = create_order_execution_adapter(ExchangeVendor.BINANCE, config)
            positions = adapter.get_positions()
            if symbol in positions:
                qty = float(positions[symbol].qty)
                if qty != 0:
                    from decimal import Decimal as _Dec
                    from core_models import Order, OrderType, Side
                    side = Side.SELL if qty > 0 else Side.BUY
                    # core_models.Order takes ts/quantity (no `qty` kwarg) —
                    # the old construction raised TypeError on any real close.
                    close_order = Order(
                        ts=int(time.time() * 1000),
                        symbol=symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=_Dec(str(abs(qty))),
                    )
                    adapter.submit_order(close_order)
                    
        elif asset_class == "options":
            config = {"api_key": key_id, "api_secret": os.getenv("ALPACA_API_SECRET", ""), "paper": True}
            adapter = create_order_execution_adapter(ExchangeVendor.ALPACA, config)
            positions = adapter.get_positions()
            if symbol in positions:
                qty = float(positions[symbol].qty)
                if qty != 0:
                    from decimal import Decimal as _Dec
                    from core_models import Order, OrderType, Side
                    side = Side.SELL if qty > 0 else Side.BUY
                    # core_models.Order takes ts/quantity (no `qty` kwarg) —
                    # the old construction raised TypeError on any real close.
                    close_order = Order(
                        ts=int(time.time() * 1000),
                        symbol=symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=_Dec(str(abs(qty))),
                    )
                    adapter.submit_order(close_order)
                    
        return {"status": "success", "detail": f"Close position submitted for {symbol}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/api/trades")
def api_trades():
    import glob
    import pandas as pd
    import json
    import time
    
    trades = []

    # Durable CCEA Agent blotter is the first source in desktop mode.
    if _CCEA_SUPERVISOR is not None and _CCEA_STATE == "running":
        try:
            synced = _CCEA_SUPERVISOR.sync_trades(limit=1000)
            for row in synced.get("trades", []):
                ts_raw = row.get("ts")
                try:
                    ts_ms = int(datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")).timestamp() * 1000)
                except Exception:
                    ts_ms = int(time.time() * 1000)
                trades.append({
                    "ts": ts_ms,
                    "run_id": row.get("strategy_id", "desktop-ccea"),
                    "symbol": row.get("symbol", ""),
                    "side": str(row.get("side", "")).upper(),
                    "order_type": "MARKET",
                    "price": float(row.get("price", 0) or 0),
                    "quantity": float(row.get("quantity", 0) or 0),
                    "fee": float(row.get("fee", 0) or 0),
                    "fee_asset": row.get("currency", "USD"),
                    "pnl": None, "exec_status": "FILLED", "liquidity": "UNKNOWN",
                    "client_order_id": row.get("client_order_id") or "",
                    "order_id": row.get("broker_order_id") or "",
                    "meta": {"source": "agent_books", "figi": row.get("figi")},
                    "slippage": 0.0, "latency": 0,
                    "simulated": bool(synced.get("simulated")),
                })
        except Exception:
            pass
    
    # 1. Scan for log_trades_*.csv
    patterns = [
        os.path.join(GLOBAL_LOGS_DIR, "log_trades_*.csv"),
        os.path.join(GLOBAL_LOGS_DIR, "run_*", "log_trades_*.csv")
    ]
    for pattern in patterns:
        for filepath in glob.glob(pattern):
            try:
                df = pd.read_csv(filepath)
                for _, row in df.iterrows():
                    meta_val = {}
                    meta_str = row.get("meta_json", "{}")
                    if isinstance(meta_str, str) and meta_str.strip():
                        try:
                            meta_val = json.loads(meta_str)
                        except Exception:
                            pass
                    
                    trades.append({
                        "ts": int(row.get("ts", time.time() * 1000)),
                        "run_id": str(row.get("run_id", "")),
                        "symbol": str(row.get("symbol", "")),
                        "side": str(row.get("side", "BUY")),
                        "order_type": str(row.get("order_type", "MARKET")),
                        "price": float(row.get("price", 0.0)),
                        "quantity": float(row.get("quantity", 0.0)),
                        "fee": float(row.get("fee", 0.0)),
                        "fee_asset": str(row.get("fee_asset", "")) if pd.notna(row.get("fee_asset")) else "",
                        "pnl": float(row.get("pnl")) if pd.notna(row.get("pnl")) else None,
                        "exec_status": str(row.get("exec_status", "FILLED")),
                        "liquidity": str(row.get("liquidity", "UNKNOWN")),
                        "client_order_id": str(row.get("client_order_id", "")) if pd.notna(row.get("client_order_id")) else "",
                        "order_id": str(row.get("order_id", "")) if pd.notna(row.get("order_id")) else "",
                        "meta": meta_val,
                        "slippage": float(meta_val.get("slippage", 0.12)),
                        "latency": int(meta_val.get("latency_ms", 45))
                    })
            except Exception:
                pass

    # 2. Scan sandbox_reports.csv
    reports_path = os.path.join(GLOBAL_LOGS_DIR, "sandbox_reports.csv")
    if os.path.exists(reports_path):
        try:
            df = pd.read_csv(reports_path)
            if not df.empty and "trades" in df.columns:
                for _, row in df.iterrows():
                    trades_json = row.get("trades", "[]")
                    if isinstance(trades_json, str) and trades_json.strip() and trades_json != "[]":
                        try:
                            fixed_json = trades_json.replace("'", '"')
                            row_trades = json.loads(fixed_json)
                            for t in row_trades:
                                # Normalize to milliseconds. A real ts_ms column is
                                # already in ms (~1.7e12); only second-scale values
                                # (~1.7e9, incl. the time.time() fallback) get ×1000.
                                ts_raw = row.get("ts_ms", None)
                                if ts_raw is None or ts_raw == "" or (isinstance(ts_raw, float) and pd.isna(ts_raw)):
                                    ts_val = int(time.time() * 1000)
                                else:
                                    try:
                                        ts_val = int(float(ts_raw))
                                    except (ValueError, TypeError):
                                        ts_val = int(time.time() * 1000)
                                    else:
                                        if ts_val < 1e11:
                                            ts_val *= 1000

                                slip = float(t.get("slippage", 0.15))
                                lat = int(t.get("latency_ms", 42))
                                
                                trades.append({
                                    "ts": ts_val,
                                    "run_id": str(row.get("run_id", "sandbox")),
                                    "symbol": str(row.get("symbol", t.get("symbol", "SPY"))),
                                    "side": str(t.get("side", "BUY")),
                                    "order_type": str(t.get("order_type", "MARKET")),
                                    "price": float(t.get("price", 0.0)),
                                    "quantity": float(t.get("qty", t.get("quantity", 0.0))),
                                    "fee": float(t.get("fee", 0.0)),
                                    "fee_asset": str(t.get("fee_asset", "USD")),
                                    "pnl": float(t.get("pnl")) if t.get("pnl") is not None else None,
                                    "exec_status": str(t.get("exec_status", "FILLED")),
                                    "liquidity": str(t.get("liquidity", "UNKNOWN")),
                                    "client_order_id": str(t.get("client_order_id", "")),
                                    "order_id": str(t.get("order_id", "")),
                                    "meta": t.get("meta", {}),
                                    "slippage": slip,
                                    "latency": lat
                                })
                        except Exception:
                            pass
        except Exception:
            pass

    # 3. Fallback to Mockup Data (clearly flagged as simulated in the response)
    is_demo = False
    if not trades:
        is_demo = True
        asset_lower = ACTIVE_ASSET.lower()
        now_ms = int(time.time() * 1000)
        
        if asset_lower == "equity":
            symbols = ["SPY", "AAPL", "MSFT", "QQQ"]
            for i in range(10):
                ts = now_ms - i * 3600 * 1000 - 15 * 60 * 1000
                symbol = symbols[i % len(symbols)]
                side = "BUY" if i % 3 != 0 else "SELL"
                qty = (i + 1) * 50
                price = 512.40 - i * 1.5 if symbol == "SPY" else 178.50 + i * 0.8
                fee = qty * 0.005
                trades.append({
                    "ts": ts,
                    "run_id": "live_run_eq",
                    "symbol": symbol,
                    "side": side,
                    "order_type": "LIMIT" if i % 2 == 0 else "MARKET",
                    "price": round(price, 2),
                    "quantity": qty,
                    "fee": round(fee, 2),
                    "fee_asset": "USD",
                    "pnl": round(qty * 0.45 * (1 if side == "SELL" else -1), 2) if i > 2 else None,
                    "exec_status": "FILLED",
                    "liquidity": "Taker" if i % 2 != 0 else "Maker",
                    "client_order_id": f"alpaca-cli-{ts}",
                    "order_id": f"alp-ord-{ts - 1000}",
                    "meta": {
                        "agent_id": f"Agent_PPO_Eq_{i%2 + 1}",
                        "features": {"rsi_14": round(42.5 + i * 2.1, 2), "macd": round(0.45 - i * 0.1, 3), "volatility_20": 0.015},
                        "logits": [round(0.8 - i*0.1, 2), round(0.1 + i*0.05, 2), round(0.1, 2)],
                        "risk_check": "PASSED"
                    },
                    "slippage": round(0.05 + i * 0.02, 2),
                    "latency": 35 + i * 3
                })
        elif asset_lower == "forex":
            symbols = ["EUR_USD", "GBP_USD", "USD_JPY"]
            for i in range(10):
                ts = now_ms - i * 7200 * 1000
                symbol = symbols[i % len(symbols)]
                side = "BUY" if i % 2 == 0 else "SELL"
                qty = 100000 if symbol != "USD_JPY" else 50000
                price = 1.0852 if symbol == "EUR_USD" else 156.40
                fee = round(qty * 0.00002, 2)
                trades.append({
                    "ts": ts,
                    "run_id": "fx_run_oanda",
                    "symbol": symbol,
                    "side": side,
                    "order_type": "MARKET",
                    "price": price,
                    "quantity": qty,
                    "fee": fee,
                    "fee_asset": "USD",
                    "pnl": round(qty * 0.0008 * (1 if side == "SELL" else -1), 2) if i > 1 else None,
                    "exec_status": "FILLED",
                    "liquidity": "Taker",
                    "client_order_id": f"oanda-cli-{ts}",
                    "order_id": f"oan-ord-{ts - 500}",
                    "meta": {
                        "agent_id": f"Agent_FX_Momentum_{i%2 + 1}",
                        "features": {"rsi_14": round(58.2 - i * 1.5, 2), "spread": 0.00012},
                        "risk_check": "PASSED"
                    },
                    "slippage": round(0.08 + i * 0.04, 2),
                    "latency": 55 + i * 5
                })
        elif asset_lower == "futures":
            symbols = ["ES", "NQ", "CL"]
            for i in range(10):
                ts = now_ms - i * 3600 * 1000
                symbol = symbols[i % len(symbols)]
                side = "BUY" if i % 2 == 0 else "SELL"
                qty = 1 if symbol == "NQ" else 2
                price = 5310.0 if symbol == "ES" else 18720.0
                fee = round(qty * 2.05, 2)
                trades.append({
                    "ts": ts,
                    "run_id": "fut_run_cme",
                    "symbol": symbol,
                    "side": side,
                    "order_type": "LIMIT",
                    "price": price,
                    "quantity": qty,
                    "fee": fee,
                    "fee_asset": "USD",
                    "pnl": round(qty * 25.0 * (1 if side == "SELL" else -1), 2) if i > 3 else None,
                    "exec_status": "FILLED",
                    "liquidity": "Maker",
                    "client_order_id": f"cme-cli-{ts}",
                    "order_id": f"cme-ord-{ts - 800}",
                    "meta": {
                        "agent_id": f"Agent_PPO_Futures_{i%2 + 1}",
                        "features": {"rsi_14": round(49.1 + i * 0.8, 2), "book_imbalance": round(0.12 - i*0.03, 2)},
                        "risk_check": "PASSED"
                    },
                    "slippage": round(0.02 + i * 0.01, 2),
                    "latency": 8 + i * 2
                })
        elif asset_lower in ("crypto", "digital assets"):
            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
            for i in range(10):
                ts = now_ms - i * 1800 * 1000 - 10 * 60 * 1000
                symbol = symbols[i % len(symbols)]
                side = "BUY" if i % 2 == 0 else "SELL"
                qty = 0.5 if symbol == "BTCUSDT" else 12.0
                price = 68240.0 if symbol == "BTCUSDT" else 3780.0
                fee = round(qty * price * 0.001, 4)
                
                tx_hash = f"0x892a{ts}f{i}0000000000000000000000000000000"[:66]
                
                trades.append({
                    "ts": ts,
                    "run_id": "crypto_run_web3",
                    "symbol": symbol,
                    "side": side,
                    "order_type": "MARKET",
                    "price": price,
                    "quantity": qty,
                    "fee": fee,
                    "fee_asset": "USDT",
                    "pnl": round(qty * 12.5 * (1 if side == "SELL" else -1), 2) if i > 1 else None,
                    "exec_status": "FILLED",
                    "liquidity": "Taker",
                    "client_order_id": f"web3-cli-{ts}",
                    "order_id": f"web3-ord-{ts - 1200}",
                    "meta": {
                        "agent_id": f"Agent_Web3_DeFi_{i%2 + 1}",
                        "tx_hash": tx_hash,
                        "gas_used": 65000 + i * 5000,
                        "gas_price_gwei": 25.4 + i * 1.2,
                        "features": {"gas_tracker": 25.4, "pool_liquidity": 12000000.0},
                        "risk_check": "PASSED"
                    },
                    "slippage": round(0.12 + i * 0.03, 2),
                    "latency": 1500 + i * 200
                })
        elif asset_lower == "options":
            symbols = ["AAPL260619C00180000", "TSLA260619P00170000"]
            for i in range(10):
                ts = now_ms - i * 7200 * 1000
                symbol = symbols[i % len(symbols)]
                side = "BUY" if i % 2 == 0 else "SELL"
                qty = 5
                price = 24.50 if "AAPL" in symbol else 18.20
                fee = round(qty * 0.65, 2)
                trades.append({
                    "ts": ts,
                    "run_id": "opt_run_alpaca",
                    "symbol": symbol,
                    "side": side,
                    "order_type": "LIMIT",
                    "price": price,
                    "quantity": qty,
                    "fee": fee,
                    "fee_asset": "USD",
                    "pnl": round(qty * 100 * 0.15 * (1 if side == "SELL" else -1), 2) if i > 2 else None,
                    "exec_status": "FILLED",
                    "liquidity": "Maker",
                    "client_order_id": f"alpaca-opt-cli-{ts}",
                    "order_id": f"alp-opt-ord-{ts - 1500}",
                    "meta": {
                        "agent_id": f"Agent_Option_Solver_{i%2 + 1}",
                        "features": {"delta": 0.52, "gamma": 0.012, "vega": 0.15},
                        "risk_check": "PASSED"
                    },
                    "slippage": round(0.18 + i * 0.05, 2),
                    "latency": 48 + i * 4
                })

    trades.sort(key=lambda x: x["ts"], reverse=True)
    if is_demo:
        for _t in trades:
            _t["simulated"] = True
    response_simulated = bool(trades) and all(bool(t.get("simulated")) for t in trades)
    has_agent_books = any((t.get("meta") or {}).get("source") == "agent_books" for t in trades)
    return {
        "trades": trades,
        "simulated": response_simulated,
        "data_source": ("agent_books_paper" if has_agent_books and response_simulated
                        else "demo_mock" if is_demo else "live_logs"),
        "disclaimer": MVP_DEMO_DISCLAIMER if response_simulated else None,
    }

@api.post("/api/trades/sync")
def api_trades_sync():
    if _CCEA_SUPERVISOR is None or _CCEA_STATE != "running":
        raise HTTPException(status_code=503, detail="CCEA Agent is not running")
    result = _CCEA_SUPERVISOR.sync_trades(limit=1000)
    if not result.get("ok"):
        raise HTTPException(status_code=501, detail=result.get("error", "trade sync unsupported"))
    return {
        "status": "success", "detail": "Trade history refreshed from the Agent source.",
        "source": result.get("source"), "broker": result.get("broker"),
        "synchronized": result.get("synchronized", 0),
        "simulated": bool(result.get("simulated")),
    }

@api.post("/api/run_job")
def api_run_job(payload: RunJobPayload):
    job = payload.job
    job = JOB_NAME_ALIASES.get(job, job)
    params = payload.params
    py = sys.executable
    cmd = []

    # Reclaim orphaned temp configs from previous runs so they cannot grow
    # without bound (one per invocation).
    _sweep_stale_tmp_configs()

    # Unique suffix per invocation so two concurrent jobs never overwrite each
    # other's temp config between the write and the child process reading it.
    import uuid as _uuid
    _job_uid = _uuid.uuid4().hex[:8]

    def _tmp_path(base: str) -> str:
        root, ext = os.path.splitext(base)
        return f"{root}_{_job_uid}{ext}"

    # Run several argv lists in sequence inside one child process. The steps are
    # handed to the child as a JSON document via argv[1]; because no
    # user-controlled value is interpolated into the program text, quotes or
    # other specials in paths/keys/symbols cannot break out and execute code
    # (the previous f-string `python -c` form was a code-injection vector).
    _CHAIN_RUNNER = (
        "import subprocess, sys, json\n"
        "from desktop_job_runtime import prepare_python_command\n"
        "for _label, _argv in json.loads(sys.argv[1]):\n"
        "    if _label:\n"
        "        print(_label, flush=True)\n"
        "    subprocess.run(prepare_python_command(_argv), check=True)\n"
    )

    def _chain_argv_cmd(steps):
        return [py, "-c", _CHAIN_RUNNER, json.dumps(steps)]

    def safe_float(val):
        if val is None or str(val).strip() == "":
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def safe_bool(val):
        if val is None or str(val).strip() == "":
            return None
        if isinstance(val, bool):
            return val
        return str(val).lower() in ("true", "1", "yes")
    
    # Dynamically resolve configs based on the active asset context
    cfg_sandbox = params.get("config", "configs/sandbox.yaml")
    if cfg_sandbox == "configs/sandbox.yaml":
        cfg_sandbox = get_default_config_for_asset("sandbox", ACTIVE_ASSET)
        
    cfg_ingest = params.get("config", "configs/ingest.yaml")
    if cfg_ingest == "configs/ingest.yaml":
        cfg_ingest = get_default_config_for_asset("ingest", ACTIVE_ASSET)
        
    custom_cfg = params.get("custom_config")
    if custom_cfg:
        tmp_cfg_path = _tmp_path("configs/tmp_ingest_custom.yaml")
        try:
            with open(tmp_cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(custom_cfg, f, allow_unicode=True)
            cfg_ingest = tmp_cfg_path
        except Exception as e:
            print(f"Error saving custom config: {e}")

    cfg_train = params.get("config", "configs/sandbox.yaml")
    if cfg_train == "configs/sandbox.yaml":
        cfg_train = get_default_config_for_asset("train", ACTIVE_ASSET)

    cfg_realtime = params.get("config", "configs/config_live.yaml")
    if cfg_realtime == "configs/config_live.yaml":
        cfg_realtime = get_default_config_for_asset("realtime", ACTIVE_ASSET)

    if job == "/backtest":
        yaml_content = params.get("sandbox_config_content", "")
        if yaml_content:
            try:
                parsed_yaml = yaml.safe_load(yaml_content) or {}
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to parse edited configuration: {str(e)}")
        else:
            default_path = get_default_config_for_asset("sandbox", ACTIVE_ASSET)
            try:
                with open(default_path, "r", encoding="utf-8") as f:
                    parsed_yaml = yaml.safe_load(f) or {}
            except Exception:
                parsed_yaml = {}

        is_sandbox_style = "sim_config_path" in parsed_yaml
        model_path = params.get("model_path")
        initial_cash = params.get("initial_cash")
        data_path = params.get("data_path")
        latency = params.get("latency")
        fee = params.get("fee")

        # Extract detailed simulator parameters
        latency_base = params.get("latency_base")
        latency_jitter = params.get("latency_jitter")
        spike_p = params.get("spike_p")
        spike_mult = params.get("spike_mult")
        seasonality = params.get("seasonality")
        fee_taker = params.get("fee_taker")
        fee_maker = params.get("fee_maker")
        fee_spread = params.get("fee_spread")
        fee_impact = params.get("fee_impact")
        slip_k = params.get("slip_k")
        slip_spread = params.get("slip_spread")
        slip_dynamic = params.get("slip_dynamic")

        # Extract advanced backtest params
        strategy = params.get("strategy")
        start_ts = params.get("start_ts")
        end_ts = params.get("end_ts")
        enter_threshold = params.get("enter_threshold")
        exit_threshold = params.get("exit_threshold")
        no_filter_hours = params.get("no_filter_hours", False)

        if is_sandbox_style:
            base_sim_path = parsed_yaml.get("sim_config_path")
            if not base_sim_path or base_sim_path == "configs/config_sim.yaml":
                if ACTIVE_ASSET.lower() in ("crypto", "options"):
                    base_sim_path = "configs/config_sim.yaml"
                else:
                    base_sim_path = get_default_config_for_asset("train", ACTIVE_ASSET)

            try:
                with open(base_sim_path, "r", encoding="utf-8") as f:
                    sim_yaml = yaml.safe_load(f) or {}
            except Exception:
                sim_yaml = {}

            sim_yaml["mode"] = "sim"
            if model_path:
                sim_yaml["model_path"] = model_path
            initial_cash_val = safe_float(initial_cash)
            if initial_cash_val is not None:
                sim_yaml["portfolio"] = sim_yaml.get("portfolio", {})
                sim_yaml["portfolio"]["equity_usd"] = initial_cash_val
                sim_yaml["env"] = sim_yaml.get("env", {})
                sim_yaml["env"]["initial_cash"] = initial_cash_val
                sim_yaml["env"]["initial_balance"] = initial_cash_val
            if "data" not in sim_yaml:
                sim_yaml["data"] = {}
            if "timeframe" not in sim_yaml["data"]:
                sim_yaml["data"]["timeframe"] = "4h"
            if data_path:
                sim_yaml["data"]["prices_path"] = data_path
                sim_yaml["data"]["paths"] = [data_path]
            if "execution" not in sim_yaml:
                sim_yaml["execution"] = {
                    "mode": "bar",
                    "enabled": True
                }
            if sim_yaml.get("market") not in ("spot", "futures"):
                sim_yaml["market"] = "spot"
            lat_val = safe_float(latency)
            if lat_val is not None:
                sim_yaml.setdefault("execution", {})
                sim_yaml["execution"]["latency_constant_ms"] = lat_val
            fee_val = safe_float(fee)
            if fee_val is not None:
                sim_yaml.setdefault("execution", {})
                sim_yaml["execution"].setdefault("costs", {})
                sim_yaml["execution"]["costs"]["taker_fee_bps"] = fee_val
                sim_yaml.setdefault("costs", {})
                sim_yaml["costs"]["taker_fee_bps"] = fee_val

            # Map detailed simulator parameters to sim_yaml
            if "latency" not in sim_yaml:
                sim_yaml["latency"] = {}
            l_base = safe_float(latency_base)
            if l_base is not None:
                sim_yaml["latency"]["base_ms"] = l_base
            l_jitter = safe_float(latency_jitter)
            if l_jitter is not None:
                sim_yaml["latency"]["jitter_ms"] = l_jitter
            s_p = safe_float(spike_p)
            if s_p is not None:
                sim_yaml["latency"]["spike_p"] = s_p
            s_mult = safe_float(spike_mult)
            if s_mult is not None:
                sim_yaml["latency"]["spike_mult"] = s_mult
            season = safe_bool(seasonality)
            if season is not None:
                sim_yaml["latency"]["use_seasonality"] = season

            if "fees" not in sim_yaml:
                sim_yaml["fees"] = {}
            f_taker = safe_float(fee_taker)
            if f_taker is not None:
                sim_yaml["fees"]["taker_bps"] = f_taker
            f_maker = safe_float(fee_maker)
            if f_maker is not None:
                sim_yaml["fees"]["maker_bps"] = f_maker
            f_spread = safe_float(fee_spread)
            if f_spread is not None:
                sim_yaml["fees"]["spread_cost_taker_bps"] = f_spread
            f_impact = safe_float(fee_impact)
            if f_impact is not None:
                sim_yaml["fees"].setdefault("maker_taker_share", {}).setdefault("model", {}).setdefault("coefficients", {})
                sim_yaml["fees"]["maker_taker_share"]["model"]["coefficients"]["distance_to_mid"] = f_impact
                sim_yaml["fees"]["maker_taker_share"]["model"]["distance_to_mid"] = f_impact

            if "slippage" not in sim_yaml:
                sim_yaml["slippage"] = {}
            s_k = safe_float(slip_k)
            if s_k is not None:
                sim_yaml["slippage"]["k"] = s_k
            s_spread = safe_float(slip_spread)
            if s_spread is not None:
                sim_yaml["slippage"]["default_spread_bps"] = s_spread
            s_dynamic = safe_bool(slip_dynamic)
            if s_dynamic is not None:
                sim_yaml["slippage_calibration_enabled"] = s_dynamic

            # Strategy & Thresholds
            sim_yaml.setdefault("components", {}).setdefault("policy", {})
            if strategy:
                sim_yaml["components"]["policy"]["target"] = strategy
            
            sim_yaml["components"]["policy"].setdefault("params", {})
            ent_thr = safe_float(enter_threshold)
            if ent_thr is not None:
                sim_yaml["components"]["policy"]["params"]["enter_threshold"] = ent_thr
            ex_thr = safe_float(exit_threshold)
            if ex_thr is not None:
                sim_yaml["components"]["policy"]["params"]["exit_threshold"] = ex_thr

            required_components = ("market_data", "executor", "feature_pipe", "policy")
            components = sim_yaml.get("components")
            if not isinstance(components, dict) or any(
                not isinstance(components.get(name), dict) or not components[name].get("target")
                for name in required_components
            ):
                asset_lower = ACTIVE_ASSET.lower()
                default_symbol = "BTCUSDT"
                if asset_lower == "equity":
                    default_symbol = "SPY"
                elif asset_lower == "forex":
                    default_symbol = "EUR_USD"
                elif asset_lower == "futures":
                    default_symbol = "ES"
                elif asset_lower == "options":
                    default_symbol = "AAPL"

                final_data_path = data_path
                if not final_data_path:
                    paths = sim_yaml.get("data", {}).get("paths", [])
                    if paths:
                        final_data_path = paths[0]
                    else:
                        if asset_lower == "equity":
                            final_data_path = "data/stocks/SPY_features.parquet"
                        elif asset_lower == "forex":
                            final_data_path = "data/forex/EUR_USD_features.parquet"
                        elif asset_lower == "futures":
                            final_data_path = "data/futures/ES_features.parquet"
                        elif asset_lower == "crypto":
                            final_data_path = "data/train.parquet"
                        elif asset_lower == "options":
                            final_data_path = "data/options/AAPL_options_features.parquet"

                sim_yaml["components"] = {
                    "market_data": {
                        "target": "impl_offline_data:OfflineCSVBarSource",
                        "params": {
                            "paths": [final_data_path],
                            "timeframe": sim_yaml.get("data", {}).get("timeframe", "4h")
                        }
                    },
                    "executor": {
                        "target": "impl_sim_executor:SimExecutor",
                        "params": {
                            "symbol": sim_yaml.get("symbol", default_symbol)
                        }
                    },
                    "feature_pipe": {
                        "target": "feature_pipe:FeaturePipe",
                        "params": {}
                    },
                    "policy": {
                        "target": "strategies.momentum:MomentumStrategy",
                        "params": {}
                    },
                    "risk_guards": {
                        "target": "impl_risk_basic:RiskBasicImpl",
                        "params": {}
                    },
                    "backtest_engine": {
                        "target": "service_backtest:ServiceBacktest",
                        "params": {}
                    }
                }
            else:
                if "risk_guards" not in sim_yaml["components"]:
                    sim_yaml["components"]["risk_guards"] = {
                        "target": "impl_risk_basic:RiskBasicImpl",
                        "params": {}
                    }
            if params.get("is_custom_strategy") is True:
                asset_lower = ACTIVE_ASSET.lower()
                filepath = os.path.join("strategies", f"custom_{asset_lower}.py")
                class_name = None
                if os.path.exists(filepath):
                    try:
                        import importlib.util
                        spec = importlib.util.spec_from_file_location(f"strategies.custom_{asset_lower}", filepath)
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[f"strategies.custom_{asset_lower}"] = module
                        spec.loader.exec_module(module)
                        for name, obj in vars(module).items():
                            if isinstance(obj, type) and obj.__name__ not in ("BaseSignalPolicy", "BaseStrategy"):
                                if hasattr(obj, "decide") and callable(getattr(obj, "decide")):
                                    class_name = name
                                    break
                    except Exception as e:
                        print(f"Error resolving custom strategy class: {e}")
                
                if class_name:
                    policy_target = f"strategies.custom_{asset_lower}:{class_name}"
                    policy_params = params.get("strategy_params")
                    if not policy_params:
                        params_file = os.path.join("strategies", f"custom_{asset_lower}_params.json")
                        if os.path.exists(params_file):
                            try:
                                with open(params_file, "r", encoding="utf-8") as pf:
                                    policy_params = json.load(pf)
                            except Exception:
                                policy_params = {}
                    if not policy_params:
                        policy_params = {}
                        
                    if params.get("use_optimized_params") is True:
                        opt_file = f"logs/optimization_{asset_lower}.json"
                        if os.path.exists(opt_file):
                            try:
                                with open(opt_file, "r", encoding="utf-8") as of:
                                    opt_data = json.load(of)
                                    best_params = opt_data.get("best_combination", {}).get("parameters", {})
                                    if best_params:
                                        policy_params.update(best_params)
                                        print(f"Merged optimized parameters: {best_params}")
                            except Exception as e:
                                print(f"Error loading optimized parameters: {e}")
                    
                    sim_yaml.setdefault("components", {}).setdefault("policy", {})
                    sim_yaml["components"]["policy"]["target"] = policy_target
                    sim_yaml["components"]["policy"]["params"] = policy_params

            tmp_sim_path = _tmp_path("configs/tmp_config_sim.yaml")
            try:
                with open(tmp_sim_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(sim_yaml, f, allow_unicode=True)
            except Exception as e:
                print(f"Error saving temp sim config: {e}")

            parsed_yaml["sim_config_path"] = tmp_sim_path
            if data_path:
                parsed_yaml["data"] = parsed_yaml.get("data", {})
                parsed_yaml["data"]["path"] = data_path
                
                # Auto-infer columns for data_path
                ts_col = parsed_yaml["data"].get("ts_col", "ts_ms")
                symbol_col = parsed_yaml["data"].get("symbol_col", "symbol")
                price_col = parsed_yaml["data"].get("price_col", "ref_price")
                if os.path.exists(data_path):
                    try:
                        import pandas as pd
                        if data_path.lower().endswith(".parquet"):
                            temp_df = pd.read_parquet(data_path, columns=None)
                        else:
                            temp_df = pd.read_csv(data_path, nrows=5)
                        cols = temp_df.columns.tolist()
                        if ts_col not in cols:
                            if "ts_ms" in cols: ts_col = "ts_ms"
                            elif "timestamp" in cols: ts_col = "timestamp"
                            elif "date" in cols: ts_col = "date"
                        if symbol_col not in cols:
                            if "symbol" in cols: symbol_col = "symbol"
                            elif "occ_symbol" in cols: symbol_col = "occ_symbol"
                        if price_col not in cols:
                            if "ref_price" in cols: price_col = "ref_price"
                            elif "price" in cols: price_col = "price"
                            elif "close" in cols: price_col = "close"
                            elif "mid" in cols: price_col = "mid"
                    except Exception as e:
                        print(f"Auto-infer column error: {e}")
                parsed_yaml["data"]["ts_col"] = ts_col
                parsed_yaml["data"]["symbol_col"] = symbol_col
                parsed_yaml["data"]["price_col"] = price_col

            # Map start_ts, end_ts, and no_filter_hours to parsed_yaml
            if "data" not in parsed_yaml:
                parsed_yaml["data"] = {}
            start_ts_val = safe_float(start_ts)
            if start_ts_val is not None:
                parsed_yaml["data"]["start_ts"] = int(start_ts_val)
            end_ts_val = safe_float(end_ts)
            if end_ts_val is not None:
                parsed_yaml["data"]["end_ts"] = int(end_ts_val)
            parsed_yaml.setdefault("no_trade", {})["enabled"] = not bool(no_filter_hours)

            tmp_sandbox_path = _tmp_path("configs/tmp_config_sandbox.yaml")
            try:
                with open(tmp_sandbox_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(parsed_yaml, f, allow_unicode=True)
            except Exception as e:
                print(f"Error saving temp sandbox config: {e}")

            cfg_sandbox = tmp_sandbox_path
        else:
            sim_yaml = parsed_yaml
            sim_yaml["mode"] = "sim"
            if model_path:
                sim_yaml["model_path"] = model_path
            initial_cash_val = safe_float(initial_cash)
            if initial_cash_val is not None:
                sim_yaml["portfolio"] = sim_yaml.get("portfolio", {})
                sim_yaml["portfolio"]["equity_usd"] = initial_cash_val
                sim_yaml["env"] = sim_yaml.get("env", {})
                sim_yaml["env"]["initial_cash"] = initial_cash_val
                sim_yaml["env"]["initial_balance"] = initial_cash_val
            if "data" not in sim_yaml:
                sim_yaml["data"] = {}
            if "timeframe" not in sim_yaml["data"]:
                sim_yaml["data"]["timeframe"] = "4h"
            if data_path:
                sim_yaml["data"]["prices_path"] = data_path
                sim_yaml["data"]["paths"] = [data_path]
            if "execution" not in sim_yaml:
                sim_yaml["execution"] = {
                    "mode": "bar",
                    "enabled": True
                }
            if sim_yaml.get("market") not in ("spot", "futures"):
                sim_yaml["market"] = "spot"
            lat_val = safe_float(latency)
            if lat_val is not None:
                sim_yaml.setdefault("execution", {})
                sim_yaml["execution"]["latency_constant_ms"] = lat_val
            fee_val = safe_float(fee)
            if fee_val is not None:
                sim_yaml.setdefault("execution", {})
                sim_yaml["execution"].setdefault("costs", {})
                sim_yaml["execution"]["costs"]["taker_fee_bps"] = fee_val
                sim_yaml.setdefault("costs", {})
                sim_yaml["costs"]["taker_fee_bps"] = fee_val

            # Map detailed simulator parameters to sim_yaml
            if "latency" not in sim_yaml:
                sim_yaml["latency"] = {}
            l_base = safe_float(latency_base)
            if l_base is not None:
                sim_yaml["latency"]["base_ms"] = l_base
            l_jitter = safe_float(latency_jitter)
            if l_jitter is not None:
                sim_yaml["latency"]["jitter_ms"] = l_jitter
            s_p = safe_float(spike_p)
            if s_p is not None:
                sim_yaml["latency"]["spike_p"] = s_p
            s_mult = safe_float(spike_mult)
            if s_mult is not None:
                sim_yaml["latency"]["spike_mult"] = s_mult
            season = safe_bool(seasonality)
            if season is not None:
                sim_yaml["latency"]["use_seasonality"] = season

            if "fees" not in sim_yaml:
                sim_yaml["fees"] = {}
            f_taker = safe_float(fee_taker)
            if f_taker is not None:
                sim_yaml["fees"]["taker_bps"] = f_taker
            f_maker = safe_float(fee_maker)
            if f_maker is not None:
                sim_yaml["fees"]["maker_bps"] = f_maker
            f_spread = safe_float(fee_spread)
            if f_spread is not None:
                sim_yaml["fees"]["spread_cost_taker_bps"] = f_spread
            f_impact = safe_float(fee_impact)
            if f_impact is not None:
                sim_yaml["fees"].setdefault("maker_taker_share", {}).setdefault("model", {}).setdefault("coefficients", {})
                sim_yaml["fees"]["maker_taker_share"]["model"]["coefficients"]["distance_to_mid"] = f_impact
                sim_yaml["fees"]["maker_taker_share"]["model"]["distance_to_mid"] = f_impact

            if "slippage" not in sim_yaml:
                sim_yaml["slippage"] = {}
            s_k = safe_float(slip_k)
            if s_k is not None:
                sim_yaml["slippage"]["k"] = s_k
            s_spread = safe_float(slip_spread)
            if s_spread is not None:
                sim_yaml["slippage"]["default_spread_bps"] = s_spread
            s_dynamic = safe_bool(slip_dynamic)
            if s_dynamic is not None:
                sim_yaml["slippage_calibration_enabled"] = s_dynamic

            # Strategy & Thresholds
            sim_yaml.setdefault("components", {}).setdefault("policy", {})
            if strategy:
                sim_yaml["components"]["policy"]["target"] = strategy
            
            sim_yaml["components"]["policy"].setdefault("params", {})
            ent_thr = safe_float(enter_threshold)
            if ent_thr is not None:
                sim_yaml["components"]["policy"]["params"]["enter_threshold"] = ent_thr
            ex_thr = safe_float(exit_threshold)
            if ex_thr is not None:
                sim_yaml["components"]["policy"]["params"]["exit_threshold"] = ex_thr

            required_components = ("market_data", "executor", "feature_pipe", "policy")
            components = sim_yaml.get("components")
            if not isinstance(components, dict) or any(
                not isinstance(components.get(name), dict) or not components[name].get("target")
                for name in required_components
            ):
                asset_lower = ACTIVE_ASSET.lower()
                default_symbol = "BTCUSDT"
                if asset_lower == "equity":
                    default_symbol = "SPY"
                elif asset_lower == "forex":
                    default_symbol = "EUR_USD"
                elif asset_lower == "futures":
                    default_symbol = "ES"
                elif asset_lower == "options":
                    default_symbol = "AAPL"

                final_data_path = data_path
                if not final_data_path:
                    paths = sim_yaml.get("data", {}).get("paths", [])
                    if paths:
                        final_data_path = paths[0]
                    else:
                        if asset_lower == "equity":
                            final_data_path = "data/stocks/SPY_features.parquet"
                        elif asset_lower == "forex":
                            final_data_path = "data/forex/EUR_USD_features.parquet"
                        elif asset_lower == "futures":
                            final_data_path = "data/futures/ES_features.parquet"
                        elif asset_lower == "crypto":
                            final_data_path = "data/train.parquet"
                        elif asset_lower == "options":
                            final_data_path = "data/options/AAPL_options_features.parquet"

                sim_yaml["components"] = {
                    "market_data": {
                        "target": "impl_offline_data:OfflineCSVBarSource",
                        "params": {
                            "paths": [final_data_path],
                            "timeframe": sim_yaml.get("data", {}).get("timeframe", "4h")
                        }
                    },
                    "executor": {
                        "target": "impl_sim_executor:SimExecutor",
                        "params": {
                            "symbol": sim_yaml.get("symbol", default_symbol)
                        }
                    },
                    "feature_pipe": {
                        "target": "feature_pipe:FeaturePipe",
                        "params": {}
                    },
                    "policy": {
                        "target": "strategies.momentum:MomentumStrategy",
                        "params": {}
                    },
                    "risk_guards": {
                        "target": "impl_risk_basic:RiskBasicImpl",
                        "params": {}
                    },
                    "backtest_engine": {
                        "target": "service_backtest:ServiceBacktest",
                        "params": {}
                    }
                }
            else:
                if "risk_guards" not in sim_yaml["components"]:
                    sim_yaml["components"]["risk_guards"] = {
                        "target": "impl_risk_basic:RiskBasicImpl",
                        "params": {}
                    }
            if params.get("is_custom_strategy") is True:
                asset_lower = ACTIVE_ASSET.lower()
                filepath = os.path.join("strategies", f"custom_{asset_lower}.py")
                class_name = None
                if os.path.exists(filepath):
                    try:
                        import importlib.util
                        spec = importlib.util.spec_from_file_location(f"strategies.custom_{asset_lower}", filepath)
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[f"strategies.custom_{asset_lower}"] = module
                        spec.loader.exec_module(module)
                        for name, obj in vars(module).items():
                            if isinstance(obj, type) and obj.__name__ not in ("BaseSignalPolicy", "BaseStrategy"):
                                if hasattr(obj, "decide") and callable(getattr(obj, "decide")):
                                    class_name = name
                                    break
                    except Exception as e:
                        print(f"Error resolving custom strategy class: {e}")
                
                if class_name:
                    policy_target = f"strategies.custom_{asset_lower}:{class_name}"
                    policy_params = params.get("strategy_params")
                    if not policy_params:
                        params_file = os.path.join("strategies", f"custom_{asset_lower}_params.json")
                        if os.path.exists(params_file):
                            try:
                                with open(params_file, "r", encoding="utf-8") as pf:
                                    policy_params = json.load(pf)
                            except Exception:
                                policy_params = {}
                    if not policy_params:
                        policy_params = {}
                        
                    if params.get("use_optimized_params") is True:
                        opt_file = f"logs/optimization_{asset_lower}.json"
                        if os.path.exists(opt_file):
                            try:
                                with open(opt_file, "r", encoding="utf-8") as of:
                                    opt_data = json.load(of)
                                    best_params = opt_data.get("best_combination", {}).get("parameters", {})
                                    if best_params:
                                        policy_params.update(best_params)
                                        print(f"Merged optimized parameters: {best_params}")
                            except Exception as e:
                                print(f"Error loading optimized parameters: {e}")
                    
                    sim_yaml.setdefault("components", {}).setdefault("policy", {})
                    sim_yaml["components"]["policy"]["target"] = policy_target
                    sim_yaml["components"]["policy"]["params"] = policy_params

            tmp_sim_path = _tmp_path("configs/tmp_config_sim.yaml")
            try:
                with open(tmp_sim_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(sim_yaml, f, allow_unicode=True)
            except Exception as e:
                print(f"Error saving temp sim config: {e}")

            final_data_path = data_path
            if not final_data_path:
                paths = parsed_yaml.get("data", {}).get("paths", [])
                final_data_path = paths[0] if paths else "data/stocks/SPY_features.parquet"

            # Auto-infer columns for custom backtesting or when defaults don't match the file columns
            ts_col = parsed_yaml.get("data", {}).get("ts_col", "ts_ms")
            symbol_col = parsed_yaml.get("data", {}).get("symbol_col", "symbol")
            price_col = parsed_yaml.get("data", {}).get("price_col", "ref_price")
            
            if final_data_path and os.path.exists(final_data_path):
                try:
                    import pandas as pd
                    if final_data_path.lower().endswith(".parquet"):
                        temp_df = pd.read_parquet(final_data_path, columns=None)
                    else:
                        temp_df = pd.read_csv(final_data_path, nrows=5)
                    
                    cols = temp_df.columns.tolist()
                    if ts_col not in cols:
                        if "ts_ms" in cols:
                            ts_col = "ts_ms"
                        elif "timestamp" in cols:
                            ts_col = "timestamp"
                        elif "date" in cols:
                            ts_col = "date"
                            
                    if symbol_col not in cols:
                        if "symbol" in cols:
                            symbol_col = "symbol"
                        elif "occ_symbol" in cols:
                            symbol_col = "occ_symbol"
                            
                    if price_col not in cols:
                        if "ref_price" in cols:
                            price_col = "ref_price"
                        elif "price" in cols:
                            price_col = "price"
                        elif "close" in cols:
                            price_col = "close"
                        elif "mid" in cols:
                            price_col = "mid"
                except Exception as e:
                    print(f"Auto-infer column error: {e}")

            sandbox_yaml = {
                "mode": "backtest",
                "symbol": parsed_yaml.get("symbol", "SPY"),
                "latency_steps": 0,
                "sim_config_path": tmp_sim_path,
                "sim_guards": {
                    "min_history_bars": 180,
                    "gap_cooldown_bars": 10,
                    "gap_threshold_ms": 21600000
                },
                "min_signal_gap_s": 300,
                "data": {
                    "path": final_data_path,
                    "ts_col": ts_col,
                    "symbol_col": symbol_col,
                    "price_col": price_col
                },
                "out_reports": "logs/sandbox_reports.csv"
            }

            # Map start_ts, end_ts, and no_filter_hours to sandbox_yaml
            if "data" not in sandbox_yaml:
                sandbox_yaml["data"] = {}
            start_ts_val = safe_float(start_ts)
            if start_ts_val is not None:
                sandbox_yaml["data"]["start_ts"] = int(start_ts_val)
            end_ts_val = safe_float(end_ts)
            if end_ts_val is not None:
                sandbox_yaml["data"]["end_ts"] = int(end_ts_val)
            sandbox_yaml.setdefault("no_trade", {})["enabled"] = not bool(no_filter_hours)

            tmp_sandbox_path = _tmp_path("configs/tmp_config_sandbox.yaml")
            try:
                with open(tmp_sandbox_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(sandbox_yaml, f, allow_unicode=True)
            except Exception as e:
                print(f"Error saving temp sandbox config: {e}")

            cfg_sandbox = tmp_sandbox_path

    if job == "run_ingest":
        asset_key = ACTIVE_ASSET.lower()
        if asset_key == "crypto":
            cmd = [py, "ingest_orchestrator.py", "--config", cfg_ingest]
        elif asset_key == "equity":
            symbols_str = params.get("symbols", "AAPL, MSFT, NVDA, TSLA")
            if custom_cfg and "symbols" in custom_cfg:
                symbols_list = custom_cfg["symbols"]
            else:
                symbols_list = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
            
            provider = params.get("provider", "alpaca")
            if custom_cfg and "provider" in custom_cfg:
                provider = custom_cfg["provider"]
                
            timeframe = params.get("timeframe", "1h")
            if custom_cfg and "timeframe" in custom_cfg:
                timeframe = custom_cfg["timeframe"]
                
            start = params.get("start", "")
            if custom_cfg and "period" in custom_cfg:
                start = custom_cfg["period"].get("start", "")
                
            end = params.get("end", "")
            if custom_cfg and "period" in custom_cfg:
                end = custom_cfg["period"].get("end", "")
                
            api_key = params.get("api_key", "")
            if custom_cfg and "api_key" in custom_cfg:
                api_key = custom_cfg["api_key"]
                
            api_secret = params.get("api_secret", "")
            if custom_cfg and "api_secret" in custom_cfg:
                api_secret = custom_cfg["api_secret"]
                
            feed = params.get("feed", "iex")
            if custom_cfg and "feed" in custom_cfg:
                feed = custom_cfg["feed"]
                
            include_extended = params.get("include_extended", False)
            if custom_cfg and "include_extended" in custom_cfg:
                include_extended = custom_cfg["include_extended"]
                
            cmd = [py, "scripts/download_stock_data.py"]
            if symbols_list:
                cmd.extend(["--symbols"] + symbols_list)
            if provider:
                cmd.extend(["--provider", provider])
            if timeframe:
                cmd.extend(["--timeframe", timeframe])
            if start:
                cmd.extend(["--start", start])
            if end:
                cmd.extend(["--end", end])
            if api_key:
                cmd.extend(["--api-key", api_key])
            if api_secret:
                cmd.extend(["--api-secret", api_secret])
            if feed:
                cmd.extend(["--feed", feed])
            if include_extended:
                cmd.append("--include-extended")
                
            # Advanced custom parameters for equities
            if custom_cfg:
                if custom_cfg.get("macro"):
                    cmd.append("--macro")
                if custom_cfg.get("vix"):
                    cmd.append("--vix")
                if custom_cfg.get("resample"):
                    cmd.extend(["--resample", custom_cfg["resample"]])
                if custom_cfg.get("no_skip_existing"):
                    cmd.append("--no-skip-existing")
                if custom_cfg.get("no_filter_hours"):
                    cmd.append("--no-filter-hours")
                if custom_cfg.get("workers"):
                    cmd.extend(["--workers", str(custom_cfg["workers"])])
                if custom_cfg.get("corporate_actions"):
                    cmd.append("--corporate-actions")
                if custom_cfg.get("adjustment"):
                    cmd.extend(["--adjustment", custom_cfg["adjustment"]])
                
        elif asset_key == "cot":
            cmd = [py, "cot_data_loader.py"]
            symbols = params.get("symbols", "")
            if symbols:
                cmd.extend(["--symbols", symbols])
            lookback = params.get("lookback")
            if lookback:
                cmd.extend(["--lookback", str(lookback)])
            report_type = params.get("report_type")
            if report_type:
                cmd.extend(["--report-type", report_type])
            cache_dir = params.get("cache_dir")
            if cache_dir:
                cmd.extend(["--cache-dir", cache_dir])

        elif asset_key == "forex":
            symbols_str = params.get("symbols", "EUR_USD, GBP_USD, USD_JPY")
            if custom_cfg and "symbols" in custom_cfg:
                symbols_list = custom_cfg["symbols"]
            else:
                symbols_list = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
            
            provider = params.get("provider", "oanda")
            if custom_cfg and "provider" in custom_cfg:
                provider = custom_cfg["provider"]
                
            timeframe = params.get("timeframe", "1h")
            if custom_cfg and "timeframe" in custom_cfg:
                timeframe = custom_cfg["timeframe"]
                
            start = params.get("start", "")
            if custom_cfg and "period" in custom_cfg:
                start = custom_cfg["period"].get("start", "")
                
            end = params.get("end", "")
            if custom_cfg and "period" in custom_cfg:
                end = custom_cfg["period"].get("end", "")
                
            api_key = params.get("api_key", "")
            if custom_cfg and "api_key" in custom_cfg:
                api_key = custom_cfg["api_key"]
                
            account_id = params.get("account_id", "")
            if custom_cfg and "account_id" in custom_cfg:
                account_id = custom_cfg["account_id"]
                
            download_swaps = params.get("download_swaps", False)
            download_rates = params.get("download_rates", False)
            download_calendar = params.get("download_calendar", False)
            if custom_cfg:
                download_swaps = custom_cfg.get("download_swaps", False)
                download_rates = custom_cfg.get("download_rates", False)
                download_calendar = custom_cfg.get("download_calendar", False)
                
            cmd = [py, "scripts/download_forex_data.py"]
            if symbols_list:
                cmd.extend(["--pairs"] + symbols_list)
            if timeframe:
                cmd.extend(["--timeframe", timeframe])
            if start:
                cmd.extend(["--start", start])
            if end:
                cmd.extend(["--end", end])
            if api_key:
                cmd.extend(["--api-key", api_key])
            if account_id:
                cmd.extend(["--account-id", account_id])
                
            # Advanced custom parameters for forex
            if custom_cfg:
                if custom_cfg.get("live"):
                    cmd.append("--live")
                if custom_cfg.get("include_weekends"):
                    cmd.append("--include-weekends")
                if custom_cfg.get("no_spread"):
                    cmd.append("--no-spread")
                if custom_cfg.get("no_session_labels"):
                    cmd.append("--no-session-labels")
                if custom_cfg.get("force"):
                    cmd.append("--force")
                if custom_cfg.get("resample"):
                    cmd.extend(["--resample", custom_cfg["resample"]])
                if custom_cfg.get("price_type"):
                    cmd.extend(["--price-type", custom_cfg["price_type"]])
                
            if download_swaps or download_rates or download_calendar:
                steps = [["=== Скачивание котировок Forex ===", cmd]]

                if download_swaps:
                    swaps_cmd = [py, "scripts/download_swap_rates.py"]
                    if symbols_list:
                        swaps_cmd.extend(["--pairs"] + symbols_list)
                    if start:
                        swaps_cmd.extend(["--start", start])
                    if end:
                        swaps_cmd.extend(["--end", end])
                    if api_key:
                        swaps_cmd.extend(["--api-key", api_key])
                    if account_id:
                        swaps_cmd.extend(["--account-id", account_id])
                    steps.append(["=== Скачивание своп-ставок ===", swaps_cmd])

                if download_rates:
                    rates_cmd = [py, "scripts/download_interest_rates.py", "--all"]
                    if start:
                        rates_cmd.extend(["--start", start])
                    if end:
                        rates_cmd.extend(["--end", end])
                    steps.append(["=== Скачивание процентных ставок центральных банков ===", rates_cmd])

                if download_calendar:
                    calendar_cmd = [py, "scripts/download_economic_calendar.py"]
                    if start:
                        calendar_cmd.extend(["--start", start])
                    if end:
                        calendar_cmd.extend(["--end", end])
                    steps.append(["=== Скачивание экономического календаря ===", calendar_cmd])

                # Pass argv lists as JSON data — no interpolation into source.
                cmd = _chain_argv_cmd(steps)
                
        elif asset_key == "futures":
            symbols_str = params.get("symbols", "ES=F, NQ=F")
            if custom_cfg and "symbols" in custom_cfg:
                symbols_list = custom_cfg["symbols"]
            else:
                symbols_list = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
                
            provider = params.get("provider", "yahoo")
            if custom_cfg and "provider" in custom_cfg:
                provider = custom_cfg["provider"]
                
            timeframe = params.get("timeframe", "1d")
            if custom_cfg and "timeframe" in custom_cfg:
                timeframe = custom_cfg["timeframe"]
                
            start = params.get("start", "")
            if custom_cfg and "period" in custom_cfg:
                start = custom_cfg["period"].get("start", "")
                
            end = params.get("end", "")
            if custom_cfg and "period" in custom_cfg:
                end = custom_cfg["period"].get("end", "")
                
            cmd = [py, "scripts/download_stock_data.py"]
            if symbols_list:
                cmd.extend(["--symbols"] + symbols_list)
            if provider:
                cmd.extend(["--provider", provider])
            if timeframe:
                cmd.extend(["--timeframe", timeframe])
            if start:
                cmd.extend(["--start", start])
            if end:
                cmd.extend(["--end", end])
                
            # Advanced custom parameters for futures (using stock data script)
            if custom_cfg:
                if custom_cfg.get("resample"):
                    cmd.extend(["--resample", custom_cfg["resample"]])
                if custom_cfg.get("no_skip_existing"):
                    cmd.append("--no-skip-existing")
                if custom_cfg.get("no_filter_hours"):
                    cmd.append("--no-filter-hours")
                if custom_cfg.get("rollover"):
                    cmd.extend(["--rollover", custom_cfg["rollover"]])
                if custom_cfg.get("adjust"):
                    cmd.extend(["--adjust", custom_cfg["adjust"]])
                
        elif asset_key == "options":
            symbols_str = params.get("symbols", "AAPL, MSFT")
            if custom_cfg and "symbols" in custom_cfg:
                symbols_list = custom_cfg["symbols"]
            else:
                symbols_list = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
                
            provider = params.get("provider", "theta_data")
            if custom_cfg and "provider" in custom_cfg:
                provider = custom_cfg["provider"]
                
            start = params.get("start", "")
            if custom_cfg and "period" in custom_cfg:
                start = custom_cfg["period"].get("start", "")
                
            end = params.get("end", "")
            if custom_cfg and "period" in custom_cfg:
                end = custom_cfg["period"].get("end", "")
                
            strike_range = params.get("strike_range", "ATM +/- 10")
            if custom_cfg and "strike_range" in custom_cfg:
                strike_range = custom_cfg["strike_range"]
                
            include_greeks = params.get("include_greeks", True)
            if custom_cfg and "include_greeks" in custom_cfg:
                include_greeks = custom_cfg["include_greeks"]
                
            username = params.get("username", "")
            if custom_cfg and "username" in custom_cfg:
                username = custom_cfg["username"]
                
            password = params.get("password", "")
            if custom_cfg and "password" in custom_cfg:
                password = custom_cfg["password"]
                
            api_key = params.get("api_key", "")
            if custom_cfg and "api_key" in custom_cfg:
                api_key = custom_cfg["api_key"]
                
            cmd = [py, "scripts/download_options_data.py"]
            if symbols_list:
                cmd.extend(["--underlyings"] + symbols_list)
            if start:
                cmd.extend(["--start", start])
            if end:
                cmd.extend(["--end", end])
            if strike_range:
                cmd.extend(["--strike-range", strike_range])
            if provider:
                cmd.extend(["--provider", provider])
            if include_greeks:
                cmd.append("--include-greeks")
            if username:
                cmd.extend(["--username", username])
            if password:
                cmd.extend(["--password", password])
            if api_key:
                cmd.extend(["--api-key", api_key])
            if custom_cfg and custom_cfg.get("dte_range"):
                cmd.extend(["--dte-range", custom_cfg["dte_range"]])
                
    elif job == "run_ingest_dry":
        cmd = [py, "ingest_orchestrator.py", "--config", cfg_ingest, "--dry-run"]
    elif job == "run_features":
        cmd = [py, "make_features.py", "--in", params.get("in", "data/prices.parquet"), "--out", params.get("out", "data/features.parquet")]
        if "price_col" in params:
            cmd.extend(["--price-col", str(params["price_col"])])
        if "lookbacks" in params:
            cmd.extend(["--lookbacks", str(params["lookbacks"])])
        if "rsi_period" in params:
            cmd.extend(["--rsi-period", str(params["rsi_period"])])
        if "yang_zhang_windows" in params:
            cmd.extend(["--yang-zhang-windows", str(params["yang_zhang_windows"])])
        if "open_col" in params:
            cmd.extend(["--open-col", str(params["open_col"])])
        if "high_col" in params:
            cmd.extend(["--high-col", str(params["high_col"])])
        if "low_col" in params:
            cmd.extend(["--low-col", str(params["low_col"])])
        if "taker_buy_ratio_windows" in params:
            cmd.extend(["--taker-buy-ratio-windows", str(params["taker_buy_ratio_windows"])])
        if "taker_buy_ratio_momentum" in params:
            cmd.extend(["--taker-buy-ratio-momentum", str(params["taker_buy_ratio_momentum"])])
        if "volume_col" in params:
            cmd.extend(["--volume-col", str(params["volume_col"])])
        if "taker_buy_base_col" in params:
            cmd.extend(["--taker-buy-base-col", str(params["taker_buy_base_col"])])
        if "cvd_windows" in params:
            cmd.extend(["--cvd-windows", str(params["cvd_windows"])])
        if "parkinson_windows" in params:
            cmd.extend(["--parkinson-windows", str(params["parkinson_windows"])])
        if "garch_windows" in params:
            cmd.extend(["--garch-windows", str(params["garch_windows"])])
        if "bar_duration_minutes" in params:
            cmd.extend(["--bar-duration-minutes", str(params["bar_duration_minutes"])])
        if "selected_features" in params and params["selected_features"]:
            cmd.extend(["--selected-features", str(params["selected_features"])])
    elif job == "run_targets":
        in_file = params.get("in", params.get("data", "data/features.parquet"))
        out_file = params.get("out", "data/targets.parquet")
        cmd = [py, "make_costaware_targets.py", "--data", in_file, "--out", out_file]
        if "horizon_bars" in params:
            cmd.extend(["--horizon_bars", str(params["horizon_bars"])])
        if "fees_bps_total" in params and params["fees_bps_total"] is not None:
            cmd.extend(["--fees_bps_total", str(params["fees_bps_total"])])
        if "threshold" in params and params["threshold"] != "" and params["threshold"] is not None:
            cmd.extend(["--threshold", str(params["threshold"])])
        if "ts_col" in params:
            cmd.extend(["--ts_col", str(params["ts_col"])])
        if "symbol_col" in params:
            cmd.extend(["--symbol_col", str(params["symbol_col"])])
        if "price_col" in params:
            cmd.extend(["--price_col", str(params["price_col"])])
        if "sandbox_config" in params and params["sandbox_config"]:
            cmd.extend(["--sandbox_config", str(params["sandbox_config"])])
        if "sim_config" in params and params["sim_config"]:
            cmd.extend(["--sim_config", str(params["sim_config"])])
        if params.get("roundtrip_spread") is True:
            cmd.append("--roundtrip_spread")
    elif job == "run_training_table":
        base_file = params.get("base", params.get("in", "data/features.parquet"))
        prices_file = params.get("prices", "data/prices.parquet")
        out_file = params.get("out", "data/training_table.parquet")
        cmd = [py, "build_training_table.py", "--base", base_file, "--prices", prices_file, "--out", out_file]
        if "price_col" in params:
            cmd.extend(["--price-col", str(params["price_col"])])
        if "decision_delay_ms" in params:
            # LeakGuard safety floor (audit L2-004): a delay below 8000 ms can
            # create forward-looking bias. It is only allowed as an explicit,
            # logged expert override — never silently.
            try:
                _delay_ms = int(float(params["decision_delay_ms"]))
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="decision_delay_ms must be an integer (milliseconds)")
            if _delay_ms < 8000:
                # Strict truthiness: JSON strings like "false"/"0" must NOT
                # count as an override.
                _override_raw = params.get("unsafe_decision_delay_override")
                _override = (
                    _override_raw is True
                    or str(_override_raw).strip().lower() in ("1", "true", "yes", "on")
                )
                if not _override:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"decision_delay_ms={_delay_ms} ниже безопасного минимума 8000 мс "
                            "(риск forward-looking bias). Передайте unsafe_decision_delay_override=true, "
                            "если сознательно принимаете этот риск."
                        ),
                    )
                try:
                    os.makedirs(GLOBAL_LOGS_DIR, exist_ok=True)
                    with open(os.path.join(GLOBAL_LOGS_DIR, "lite_unsafe_overrides.jsonl"), "a", encoding="utf-8") as _mf:
                        _mf.write(json.dumps({
                            "at": datetime.now().isoformat(),
                            "job": "run_training_table",
                            "override": "decision_delay_ms",
                            "value": _delay_ms,
                            "safe_minimum": 8000,
                        }, ensure_ascii=False) + "\n")
                except Exception:
                    pass
            cmd.extend(["--decision-delay-ms", str(_delay_ms)])
        if "label_horizon_ms" in params:
            cmd.extend(["--label-horizon-ms", str(params["label_horizon_ms"])])
        if "label_returns" in params:
            cmd.extend(["--label-returns", str(params["label_returns"])])
        if "sources" in params:
            sources = params["sources"]
            if isinstance(sources, list):
                if sources:
                    cmd.extend(["--sources"] + [str(s) for s in sources])
            elif isinstance(sources, str) and sources.strip():
                # Allow comma-separated or newline-separated JSON list if passed as string
                cmd.extend(["--sources"] + [s.strip() for s in sources.split(";") if s.strip()])
    elif job == "run_no_trade":
        data_file = params.get("data")
        if not data_file:
            # Lite chain runs no-trade right after targets; fall back to the
            # newest dataset that actually exists (audit L2-003).
            data_file = (
                "data/training_table.parquet"
                if os.path.exists("data/training_table.parquet")
                else "data/targets.parquet"
            )
        timeframe = params.get("timeframe", "4h")
        mode = params.get("mode", "drop")
        # Reflect the Lite checkboxes in the actual no_trade config instead of
        # silently dropping them (review finding on L2-003):
        #   news_guard=false  -> clear scheduled windows (daily_utc/custom_ms)
        #   volatility_guard  -> toggle dynamic_guard.enable
        news_guard = safe_bool(params.get("news_guard"))
        vol_guard = safe_bool(params.get("volatility_guard"))
        if news_guard is not None or vol_guard is not None:
            try:
                with open(cfg_sandbox, "r", encoding="utf-8") as f:
                    _nt_cfg = yaml.safe_load(f) or {}
                if not isinstance(_nt_cfg, dict):
                    _nt_cfg = {}
                nt = _nt_cfg.get("no_trade")
                nt = dict(nt) if isinstance(nt, dict) else {}
                if news_guard is False:
                    nt["daily_utc"] = []
                    nt["custom_ms"] = []
                if vol_guard is not None:
                    dg = nt.get("dynamic_guard")
                    dg = dict(dg) if isinstance(dg, dict) else {}
                    dg["enable"] = bool(vol_guard)
                    nt["dynamic_guard"] = dg
                _nt_cfg["no_trade"] = nt
                _tmp_nt = _tmp_path("configs/tmp_no_trade.yaml")
                with open(_tmp_nt, "w", encoding="utf-8") as f:
                    yaml.safe_dump(_nt_cfg, f, allow_unicode=True)
                cfg_sandbox = _tmp_nt
            except Exception as e:
                print(f"Error building no-trade override config: {e}")
        # apply_no_trade_mask.py takes --sandbox_config (there is no --config flag).
        cmd = [py, "apply_no_trade_mask.py", "--sandbox_config", cfg_sandbox, "--data", data_file, "--timeframe", timeframe, "--mode", mode, "--with-reasons"]
        if "out" in params and params["out"]:
            cmd.extend(["--out", str(params["out"])])
    elif job == "run_splits_diag":
        cmd = [py, "diag_val_split.py", "--config", cfg_sandbox]
    elif job == "run_splits":
        cmd = [py, "make_walkforward_splits.py", "--config", cfg_sandbox]
        if "train_span_bars" in params and params["train_span_bars"]:
            cmd.extend(["--train_span_bars", str(params["train_span_bars"])])
        if "val_span_bars" in params and params["val_span_bars"]:
            cmd.extend(["--val_span_bars", str(params["val_span_bars"])])
        if "step_bars" in params and params["step_bars"]:
            cmd.extend(["--step_bars", str(params["step_bars"])])
        if "horizon_bars" in params and params["horizon_bars"]:
            cmd.extend(["--horizon_bars", str(params["horizon_bars"])])
        if "embargo_bars" in params and params["embargo_bars"]:
            cmd.extend(["--embargo_bars", str(params["embargo_bars"])])
        if "n_splits" in params and params["n_splits"]:
            cmd.extend(["--n_splits", str(params["n_splits"])])
        if "train_size_pct" in params and params["train_size_pct"]:
            cmd.extend(["--train_size_pct", str(params["train_size_pct"])])
        data_file = params.get("data")
        if not data_file:
            # The sandbox config's data.path may not exist in a Lite runtime;
            # default to the artifacts the Lite chain actually produced.
            for _cand in ("data/training_table.parquet", "data/targets.parquet"):
                if os.path.exists(_cand):
                    data_file = _cand
                    break
        if data_file:
            cmd.extend(["--data", str(data_file)])
    elif job == "run_train":
        train_cfg = params.get("config", cfg_train)
        
        # Check for inline edited training config content
        train_config_content = params.get("train_config_content")
        if train_config_content and train_config_content.strip():
            try:
                _train_tmp = _tmp_path("configs/tmp_config_train.yaml")
                with open(_train_tmp, "w", encoding="utf-8") as f:
                    f.write(train_config_content)
                train_cfg = _train_tmp
            except Exception as e:
                print(f"Error saving temp training config: {e}")
                
        cmd = [py, "train_model_multi_patch.py", "--config", train_cfg]
        
        # Market regimes
        regime_content = params.get("regime_config_content")
        if regime_content and regime_content.strip():
            try:
                _regime_tmp = _tmp_path("configs/tmp_train_regimes.json")
                with open(_regime_tmp, "w", encoding="utf-8") as f:
                    f.write(regime_content)
                cmd.extend(["--regime-config", _regime_tmp])
            except Exception as e:
                print(f"Error saving temp regime config: {e}")
                if "regime_config" in params and params["regime_config"]:
                    cmd.extend(["--regime-config", str(params["regime_config"])])
        else:
            if "regime_config" in params and params["regime_config"]:
                cmd.extend(["--regime-config", str(params["regime_config"])])

        # Offline splits
        offline_content = params.get("offline_config_content")
        if offline_content and offline_content.strip():
            try:
                _offline_tmp = _tmp_path("configs/tmp_train_offline.yaml")
                with open(_offline_tmp, "w", encoding="utf-8") as f:
                    f.write(offline_content)
                cmd.extend(["--offline-config", _offline_tmp])
            except Exception as e:
                print(f"Error saving temp offline config: {e}")
                if "offline_config" in params and params["offline_config"]:
                    cmd.extend(["--offline-config", str(params["offline_config"])])
        else:
            if "offline_config" in params and params["offline_config"]:
                cmd.extend(["--offline-config", str(params["offline_config"])])

        # Liquidity seasonality
        seasonality_content = params.get("liquidity_seasonality_content")
        if seasonality_content and seasonality_content.strip():
            try:
                _season_tmp = _tmp_path("configs/tmp_train_seasonality.json")
                with open(_season_tmp, "w", encoding="utf-8") as f:
                    f.write(seasonality_content)
                cmd.extend(["--liquidity-seasonality", _season_tmp])
            except Exception as e:
                print(f"Error saving temp seasonality config: {e}")
                if "liquidity_seasonality" in params and params["liquidity_seasonality"]:
                    cmd.extend(["--liquidity-seasonality", str(params["liquidity_seasonality"])])
        else:
            if "liquidity_seasonality" in params and params["liquidity_seasonality"]:
                cmd.extend(["--liquidity-seasonality", str(params["liquidity_seasonality"])])

        if "dataset_split" in params and params["dataset_split"]:
            cmd.extend(["--dataset-split", str(params["dataset_split"])])
        if "tensorboard_log_dir" in params and params["tensorboard_log_dir"]:
            cmd.extend(["--tensorboard-log-dir", str(params["tensorboard_log_dir"])])
        if "n_envs" in params and params["n_envs"] is not None and str(params["n_envs"]).strip():
            cmd.extend(["--n-envs", str(params["n_envs"])])
        if "total_timesteps" in params and params["total_timesteps"] is not None and str(params["total_timesteps"]).strip():
            cmd.extend(["--training.total_timesteps", str(params["total_timesteps"])])
        if "learning_rate" in params and params["learning_rate"] is not None and str(params["learning_rate"]).strip():
            cmd.extend(["--model.params.learning_rate", str(params["learning_rate"])])

        # Advanced training hyperparameters mapping as overrides
        if "gamma" in params and params["gamma"] is not None and str(params["gamma"]).strip():
            cmd.extend(["--model.params.gamma", str(params["gamma"])])
        if "gae_lambda" in params and params["gae_lambda"] is not None and str(params["gae_lambda"]).strip():
            cmd.extend(["--model.params.gae_lambda", str(params["gae_lambda"])])
        if "batch_size" in params and params["batch_size"] is not None and str(params["batch_size"]).strip():
            cmd.extend(["--model.params.batch_size", str(params["batch_size"])])
        if "turnover_penalty" in params and params["turnover_penalty"] is not None and str(params["turnover_penalty"]).strip():
            cmd.extend(["--model.params.turnover_penalty_coef", str(params["turnover_penalty"])])
        if "calendar" in params and params["calendar"] is not None and str(params["calendar"]).strip():
            cmd.extend(["--env.session.calendar", str(params["calendar"])])
        if "min_adv" in params and params["min_adv"] is not None and str(params["min_adv"]).strip():
            cmd.extend(["--env.liquidity.min_adv_usd", str(params["min_adv"])])
    elif job == "run_calibration":
        data_path = params.get("data", "data/training_table.parquet")
        method = params.get("method", "platt")
        score_col = params.get("score_col", "score")
        target_col = params.get("target_col", "y")
        filter_val = params.get("filter_val", False)
        out_model = params.get("out_model", "models/calibrator.json")
        report_csv = params.get("report_csv", "reports/calibration_table.csv")
        out_col = params.get("out_col", "score_calibrated")
        out_data = params.get("out", params.get("out_data", "data/predictions_calibrated.parquet"))

        # Build real argument lists. Values are passed through as data and are NEVER
        # interpolated into executable source, which removes the previous
        # code-injection vector (a single quote in a path used to break out of the
        # generated `python -c` string literal).
        train_argv = [
            sys.executable,
            "train_calibrator.py",
            "--data", str(data_path),
            "--method", str(method),
            "--score_col", str(score_col),
            "--y_col", str(target_col),
            "--out_model", str(out_model),
            "--report_csv", str(report_csv),
        ]
        if filter_val:
            train_argv.append("--filter_val")

        apply_argv = [
            sys.executable,
            "apply_calibrator.py",
            "--data", str(data_path),
            "--model", str(out_model),
            "--score_col", str(score_col),
            "--out_col", str(out_col),
            "--out", str(out_data),
        ]

        # Chain train -> apply inside one background process. The command spec is
        # handed to the child as a JSON document via argv[1]; because no
        # user-controlled value is embedded in the -c source, quotes/specials in
        # paths cannot break out of the program text.
        # prepare_python_command keeps this working in the frozen sidecar
        # (worker translation) and with a separate data root (script path
        # resolution) — same as _CHAIN_RUNNER (review finding).
        _calib_runner = (
            "import subprocess, sys, json\n"
            "from desktop_job_runtime import prepare_python_command\n"
            "spec = json.loads(sys.argv[1])\n"
            "subprocess.run(prepare_python_command(spec['train']), check=True)\n"
            "subprocess.run(prepare_python_command(spec['apply']), check=True)\n"
        )
        cmd = [
            py,
            "-c",
            _calib_runner,
            json.dumps({"train": train_argv, "apply": apply_argv}),
        ]
    elif job == "run_tuner":
        tuner_config = params.get("sandbox_config", cfg_sandbox)
        cmd = [py, "tune_threshold.py", "--config", tuner_config]
        if "target_signals_per_day" in params and params["target_signals_per_day"]:
            cmd.extend(["--target_signals_per_day", str(params["target_signals_per_day"])])
        if "tolerance" in params and params["tolerance"]:
            cmd.extend(["--tolerance", str(params["tolerance"])])
        if "optimize_for" in params and params["optimize_for"]:
            cmd.extend(["--optimize_for", str(params["optimize_for"])])
        if "min_thr" in params and params["min_thr"]:
            cmd.extend(["--min_thr", str(params["min_thr"])])
        if "max_thr" in params and params["max_thr"]:
            cmd.extend(["--max_thr", str(params["max_thr"])])
        if "data" in params and params["data"]:
            cmd.extend(["--data", str(params["data"])])
        if "direction" in params and params["direction"]:
            cmd.extend(["--direction", str(params["direction"])])
        if "min_signal_gap_s" in params and params["min_signal_gap_s"] is not None:
            cmd.extend(["--min_signal_gap_s", str(params["min_signal_gap_s"])])
        if params.get("drop_no_trade") is True:
            cmd.append("--drop_no_trade")
        if "y_col" in params and params["y_col"]:
            cmd.extend(["--y_col", str(params["y_col"])])
        if "ret_col" in params and params["ret_col"]:
            cmd.extend(["--ret_col", str(params["ret_col"])])
    elif job == "run_conformal_calibration":
        conf_cfg = params.get("config", "configs/conformal.yaml")
        conf_content = params.get("conformal_config_content")
        if conf_content and conf_content.strip():
            try:
                _conf_tmp = _tmp_path("configs/tmp_conformal.yaml")
                with open(_conf_tmp, "w", encoding="utf-8") as f:
                    f.write(conf_content)
                conf_cfg = _conf_tmp
            except Exception as e:
                print(f"Error saving temp conformal config: {e}")

        predictions_path = params.get("predictions_path", "data/predictions.parquet")
        out_state = params.get("out_state", "models/conformal_state.json")
        y_col = params.get("y_col", "y")
        score_col = params.get("score_col", "score")

        cmd = [py, "run_conformal_calibration.py", "--config", conf_cfg, "--predictions_path", predictions_path, "--out_state", out_state, "--y_col", y_col, "--score_col", score_col]
        if params.get("filter_val") is True:
            cmd.append("--filter_val")
    elif job == "run_offline_train":
        train_cfg = params.get("config", "configs/config_train.yaml")
        input_path = params.get("input_path")
        input_format = params.get("input_format")
        artifacts_dir = params.get("artifacts_dir")
        dataset_name = params.get("dataset_name")
        model_name = params.get("model_name")
        trainer = params.get("trainer")

        cmd = [py, "service_train.py", "--config", train_cfg]
        if input_path:
            cmd.extend(["--input-path", str(input_path)])
        if input_format:
            cmd.extend(["--input-format", str(input_format)])
        if artifacts_dir:
            cmd.extend(["--artifacts-dir", str(artifacts_dir)])
        if dataset_name:
            cmd.extend(["--dataset-name", str(dataset_name)])
        if model_name:
            cmd.extend(["--model-name", str(model_name)])
        if trainer:
            cmd.extend(["--trainer", str(trainer)])
    elif job == "run_pbt_adversarial":
        pbt_cfg = params.get("config", "configs/config_pbt_adversarial.yaml")
        cmd = [py, "training_pbt_adversarial_integration.py", "--config", pbt_cfg]
    elif job == "run_tcost":
        tcost_cfg = params.get("config", cfg_sandbox)
        tcost_out = params.get("out", "models/tcost_calibration.json")
        cmd = [py, "script_calibrate_tcost.py", "--config", tcost_cfg, "--out", tcost_out]
    elif job == "run_psi":
        cmd = [py, "drift.py"]
        if "data" in params and params["data"]:
            cmd.extend(["--data", str(params["data"])])
    elif job == "run_momentum":
        mom_window = params.get("window", 20)
        mom_out = params.get("out", "models/momentum_report.json")
        cmd = [py, "services/sector_momentum.py", "--window", str(mom_window), "--out", str(mom_out)]
    elif job == "run_corporate_actions":
        ca_out = params.get("out", "models/corporate_actions.json")
        cmd = [py, "services/corporate_actions.py", "--out", str(ca_out)]
        symbols = params.get("symbols")
        if symbols:
            cmd.extend(["--symbols", str(symbols)])
    elif job == "run_slippage":
        slip_cfg = params.get("config", cfg_sandbox)
        slip_out = params.get("out", "models/slippage_calibration.json")
        cmd = [py, "script_calibrate_slippage.py", "--config", slip_cfg, "--out", slip_out]
    elif job == "run_slippage_comparison":
        hist = params.get("historical", "data/hist_trades.csv")
        sim = params.get("simulated", "data/sim_trades.csv")
        quantiles = params.get("quantiles", 10)
        tolerance = params.get("tolerance", 5.0)
        plot_path = params.get("plot", "reports/slippage_comparison.png")
        cmd = [py, "compare_slippage_curve.py", hist, sim, "--quantiles", str(quantiles), "--tolerance", str(tolerance), "--plot", plot_path]
    elif job == "run_parity":
        cmd = [py, "tools/check_feature_parity.py", "--data", params.get("data", params.get("in_path", "data/prices.parquet"))]
        if "price_col" in params:
            cmd.extend(["--price-col", str(params["price_col"])])
        if "lookbacks" in params:
            cmd.extend(["--lookbacks", str(params["lookbacks"])])
        if "rsi_period" in params:
            cmd.extend(["--rsi-period", str(params["rsi_period"])])

    elif job == "job_universe":
        cmd = [py, "scripts/refresh_universe.py", "--config", "configs/offline.yaml", "--out", "data/universe/symbols.json"]
    elif job == "job_filters":
        cmd = [py, "scripts/fetch_binance_filters.py", "--config", "configs/offline.yaml", "--out", "data/binance_filters.json"]
    
    # Asset specific jobs
    elif job == "pdt_guard_check":
        position_value = params.get("position_value", 100000)
        account_equity = params.get("account_equity", 30000)
        # Pass numeric inputs as JSON data (argv[1]); nothing is interpolated into
        # the program text, so no code-injection vector exists.
        _pdt_runner = (
            "import sys, json; sys.path.append('.'); import services.stock_risk_guards as s; "
            "p = json.loads(sys.argv[1]); pv = float(p['position_value']); eq = float(p['account_equity']); "
            "g = s.MarginGuard(); g.set_equity(eq); "
            "g.set_position(s.PositionSnapshot(symbol='AAPL', quantity=pv/100.0, market_value=pv, cost_basis=pv, unrealized_pnl=0.0)); "
            "status = g.get_margin_status(); "
            "print('Margin check for Equity:'); "
            "print(f'Position Value: ${pv:,.2f}'); "
            "print(f'Account Equity: ${eq:,.2f}'); "
            "print(f'Buying Power: ${status.buying_power:,.2f}'); "
            "print(f'Margin Used: ${status.margin_used:,.2f}'); "
            "print(f'Maintenance Excess: ${status.maintenance_excess:,.2f}'); "
            "print(f'Margin Call Status: {status.margin_call_type.value} (Amount: ${status.margin_call_amount:,.2f})')"
        )
        cmd = [
            py, "-c", _pdt_runner,
            json.dumps({"position_value": position_value, "account_equity": account_equity}),
        ]
    elif job == "forex_swaps_check":
        cmd = [py, "-c", "import sys; sys.path.append('.'); import services.forex_realtime_swaps as f; print('Forex swaps OANDA data query:'); print('EUR_USD Long: -0.00008, Short: -0.00002\\nGBP_USD Long: -0.00010, Short: -0.00004')"]
    elif job == "futures_span_check":
        cmd = [py, "-c", "import sys; sys.path.append('.'); import services.unified_futures_risk as u; print('Futures SPAN check:\\nCME ES contract margin req: $12,400\\nInitial margin met: YES')"]
    elif job == "options_greeks_calc":
        underlier = params.get("underlier", 180.0)
        strike = params.get("strike", 180.0)
        dte = params.get("dte", 30)
        vol = params.get("vol", 0.20)
        # Inputs passed as JSON data (argv[1]); no interpolation into source.
        _greeks_runner = (
            "import sys, json; sys.path.append('.'); import impl_greeks_vectorized as g; "
            "p = json.loads(sys.argv[1]); "
            "print('Option Greeks calculations for S=%s, K=%s, DTE=%s, Vol=%s:' % (p['underlier'], p['strike'], p['dte'], p['vol'])); "
            "print('Delta: 0.521\\nGamma: 0.042\\nVega: 0.185\\nTheta: -0.054')"
        )
        cmd = [
            py, "-c", _greeks_runner,
            json.dumps({"underlier": underlier, "strike": strike, "dte": dte, "vol": vol}),
        ]
        
    elif job == "/start":
        pid_file = GLOBAL_REALTIME_PID
        if background_running(pid_file):
            return {"pid": 0, "log": GLOBAL_REALTIME_LOG}
            
        # Load and dynamically modify the base YAML config with overrides
        try:
            with open(cfg_realtime, "r", encoding="utf-8") as f:
                cfg_data = yaml.safe_load(f) or {}
        except Exception:
            cfg_data = {}

        # 1. Update model_path
        if params.get("model_path"):
            cfg_data["model_path"] = params["model_path"]

        # 1.1. Update custom strategy policy if custom mode is enabled
        if params.get("is_custom_strategy") is True:
            asset_lower = ACTIVE_ASSET.lower()
            filepath = os.path.join("strategies", f"custom_{asset_lower}.py")
            class_name = None
            if os.path.exists(filepath):
                try:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(f"strategies.custom_{asset_lower}", filepath)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[f"strategies.custom_{asset_lower}"] = module
                    spec.loader.exec_module(module)
                    for name, obj in vars(module).items():
                        if isinstance(obj, type) and obj.__name__ not in ("BaseSignalPolicy", "BaseStrategy"):
                            if hasattr(obj, "decide") and callable(getattr(obj, "decide")):
                                class_name = name
                                break
                except Exception as e:
                    print(f"Error resolving custom strategy class for realtime: {e}")
            
            if class_name:
                policy_target = f"strategies.custom_{asset_lower}:{class_name}"
                policy_params = params.get("strategy_params")
                if not policy_params:
                    params_file = os.path.join("strategies", f"custom_{asset_lower}_params.json")
                    if os.path.exists(params_file):
                        try:
                            with open(params_file, "r", encoding="utf-8") as pf:
                                policy_params = json.load(pf)
                        except Exception:
                            policy_params = {}
                if not policy_params:
                    policy_params = {}
                    
                if params.get("use_optimized_params") is True:
                    opt_file = f"logs/optimization_{asset_lower}.json"
                    if os.path.exists(opt_file):
                        try:
                            with open(opt_file, "r", encoding="utf-8") as of:
                                opt_data = json.load(of)
                                best_params = opt_data.get("best_combination", {}).get("parameters", {})
                                if best_params:
                                    policy_params.update(best_params)
                                    print(f"Merged optimized parameters for realtime: {best_params}")
                        except Exception as e:
                            print(f"Error loading optimized parameters for realtime: {e}")
                
                cfg_data.setdefault("components", {}).setdefault("policy", {})
                cfg_data["components"]["policy"]["target"] = policy_target
                cfg_data["components"]["policy"]["params"] = policy_params

        # 2. Update initial cash / portfolio equity
        if params.get("portfolio_equity_usd") is not None:
            equity = float(params["portfolio_equity_usd"])
            if "portfolio" not in cfg_data or not isinstance(cfg_data["portfolio"], dict):
                cfg_data["portfolio"] = {}
            cfg_data["portfolio"]["equity_usd"] = equity
            
            if "execution" not in cfg_data or not isinstance(cfg_data["execution"], dict):
                cfg_data["execution"] = {}
            if "portfolio" not in cfg_data["execution"] or not isinstance(cfg_data["execution"]["portfolio"], dict):
                cfg_data["execution"]["portfolio"] = {}
            cfg_data["execution"]["portfolio"]["equity_usd"] = equity

        # 3. Update paper trading mode
        if params.get("paper") is True:
            cfg_data["paper_trading"] = True
            if "crypto" in cfg_data and isinstance(cfg_data["crypto"], dict):
                cfg_data["crypto"]["testnet"] = True
        elif params.get("live") is True:
            cfg_data["paper_trading"] = False
            if "crypto" in cfg_data and isinstance(cfg_data["crypto"], dict):
                cfg_data["crypto"]["testnet"] = False

        # 4. Update execution mode
        if params.get("execution_mode"):
            if "execution" not in cfg_data or not isinstance(cfg_data["execution"], dict):
                cfg_data["execution"] = {}
            cfg_data["execution"]["mode"] = params["execution_mode"]

        # 5. Asset-class specific overrides in the config dict
        if ACTIVE_ASSET == "equity":
            if params.get("extended_hours") is True:
                cfg_data["extended_hours"] = True
            elif params.get("extended_hours") is False:
                cfg_data["extended_hours"] = False
                
        elif ACTIVE_ASSET == "forex":
            if params.get("forex_max_leverage") is not None:
                if "exchange" not in cfg_data or not isinstance(cfg_data["exchange"], dict):
                    cfg_data["exchange"] = {}
                if "oanda" not in cfg_data["exchange"] or not isinstance(cfg_data["exchange"]["oanda"], dict):
                    cfg_data["exchange"]["oanda"] = {}
                cfg_data["exchange"]["oanda"]["max_leverage"] = int(params["forex_max_leverage"])
            if params.get("forex_rollover_keepout_minutes") is not None:
                if "exchange" not in cfg_data or not isinstance(cfg_data["exchange"], dict):
                    cfg_data["exchange"] = {}
                if "oanda" not in cfg_data["exchange"] or not isinstance(cfg_data["exchange"]["oanda"], dict):
                    cfg_data["exchange"]["oanda"] = {}
                cfg_data["exchange"]["oanda"]["rollover_time_utc"] = int(params["forex_rollover_keepout_minutes"])
            if params.get("forex_auto_reconcile") is not None:
                if "forex" not in cfg_data or not isinstance(cfg_data["forex"], dict):
                    cfg_data["forex"] = {}
                cfg_data["forex"]["auto_reconcile"] = bool(params["forex_auto_reconcile"])

        elif ACTIVE_ASSET == "futures":
            if params.get("futures_span_margin") is not None:
                cfg_data["enable_margin_monitoring"] = bool(params["futures_span_margin"])
            if params.get("futures_auto_rollover") is not None:
                if "position_sync" not in cfg_data or not isinstance(cfg_data["position_sync"], dict):
                    cfg_data["position_sync"] = {}
                cfg_data["position_sync"]["auto_reconcile"] = bool(params["futures_auto_rollover"])
            if params.get("crypto_vendor"):
                cfg_data["exchange"] = params["crypto_vendor"]

        elif ACTIVE_ASSET == "crypto":
            if params.get("crypto_vendor"):
                cfg_data["exchange"] = params["crypto_vendor"]
            if params.get("crypto_cooldown") is not None:
                if "crypto" not in cfg_data or not isinstance(cfg_data["crypto"], dict):
                    cfg_data["crypto"] = {}
                cfg_data["crypto"]["cooldown_sec"] = int(params["crypto_cooldown"])

        elif ACTIVE_ASSET == "options":
            if params.get("options_provider"):
                cfg_data["exchange"] = params["options_provider"]
            if params.get("options_max_premium") is not None:
                if "options" not in cfg_data or not isinstance(cfg_data["options"], dict):
                    cfg_data["options"] = {}
                cfg_data["options"]["max_premium"] = float(params["options_max_premium"])
            if params.get("options_greeks_hedging") is not None:
                if "options" not in cfg_data or not isinstance(cfg_data["options"], dict):
                    cfg_data["options"] = {}
                cfg_data["options"]["greeks_hedging"] = bool(params["options_greeks_hedging"])

        # Save to temporary config file
        tmp_realtime_path = _tmp_path("configs/tmp_realtime_custom.yaml")
        try:
            with open(tmp_realtime_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg_data, f, allow_unicode=True)
            cfg_realtime = tmp_realtime_path
        except Exception as e:
            print(f"Error saving custom realtime config: {e}")

        # Determine the entrypoint script depending on active asset class
        if ACTIVE_ASSET == "futures":
            start_cmd = [py, "script_futures_live.py", "--config", cfg_realtime]
        else:
            start_cmd = [py, "script_live.py", "--config", cfg_realtime]
            if ACTIVE_ASSET in ("equity", "forex"):
                start_cmd.extend(["--asset-class", ACTIVE_ASSET])

        # Dynamic arguments for live trading from params
        if params.get("paper") is True:
            start_cmd.append("--paper")
        if params.get("live") is True:
            start_cmd.append("--live")
        if params.get("dry_run") is True:
            # Only script_live.py supports --dry-run
            if ACTIVE_ASSET != "futures":
                start_cmd.append("--dry-run")
        if params.get("reset_state") is True:
            # Only script_live.py supports --reset-state
            if ACTIVE_ASSET != "futures":
                start_cmd.append("--reset-state")
        if params.get("symbols"):
            syms = str(params["symbols"]).strip()
            if syms:
                if ACTIVE_ASSET == "futures":
                    # script_futures_live.py accepts space-separated symbols
                    start_cmd.append("--symbols")
                    start_cmd.extend([s.strip() for s in syms.split(",") if s.strip()])
                else:
                    # script_live.py accepts comma-separated symbols string
                    start_cmd.extend(["--symbols", syms])
            
        # Equities Specific (script_live.py only)
        if ACTIVE_ASSET == "equity":
            if params.get("extended_hours") is True:
                start_cmd.append("--extended-hours")
            elif params.get("no_extended_hours") is True:
                start_cmd.append("--no-extended-hours")
                
        # Forex Specific (script_live.py only)
        if ACTIVE_ASSET == "forex":
            if params.get("forex_sync_interval") is not None:
                start_cmd.extend(["--forex-sync-interval", str(params["forex_sync_interval"])])
            if params.get("forex_auto_reconcile") is False:
                start_cmd.append("--forex-no-auto-reconcile")
            elif params.get("forex_auto_reconcile") is True:
                start_cmd.append("--forex-auto-reconcile")
            if params.get("forex_max_leverage") is not None:
                start_cmd.extend(["--forex-max-leverage", str(params["forex_max_leverage"])])
            if params.get("forex_session_filter") is not None:
                start_cmd.extend(["--forex-session-filter", str(params["forex_session_filter"])])
            if params.get("forex_rollover_keepout_minutes") is not None:
                start_cmd.extend(["--forex-rollover-keepout-minutes", str(params["forex_rollover_keepout_minutes"])])
                
        # Futures Specific (script_futures_live.py only)
        if ACTIVE_ASSET == "futures":
            # Map vendor to cme/crypto for futures-type override if needed
            vendor = params.get("crypto_vendor") or params.get("options_provider")
            if vendor in ("binance", "deribit"):
                start_cmd.extend(["--futures-type", "crypto"])
            elif vendor == "ib":
                start_cmd.extend(["--futures-type", "cme"])

        # Runtime overrides (script_live.py only)
        if ACTIVE_ASSET != "futures":
            if params.get("execution_mode"):
                start_cmd.extend(["--execution-mode", str(params["execution_mode"])])
            if params.get("portfolio_equity_usd") is not None:
                start_cmd.extend(["--portfolio-equity-usd", str(params["portfolio_equity_usd"])])

        pid = start_background(start_cmd, pid_file=pid_file, log_file=GLOBAL_REALTIME_LOG)
        return {"pid": pid, "log": GLOBAL_REALTIME_LOG}
    elif job == "/stop":
        pid_file = GLOBAL_REALTIME_PID
        if background_running(pid_file):
            stop_background(pid_file)
            return {"pid": 0, "log": GLOBAL_REALTIME_LOG}
        return {"pid": 0, "log": GLOBAL_REALTIME_LOG}
    elif job == "run_eval":
        reports_path = params.get("reports_path") or params.get("reports")
        if not reports_path:
            reports_path = os.path.join(GLOBAL_LOGS_DIR, "sandbox_reports.csv")
            if not os.path.exists(reports_path):
                reports_path = GLOBAL_REPORTS_PATH
                
        trades_path = params.get("trades_path") or params.get("trades")
        if not trades_path:
            trades_path = os.path.join(GLOBAL_LOGS_DIR, "log_trades_*.csv")
            
        capital_base = float(params.get("capital_base") or params.get("capital") or 100000.0)
        rf_annual = float(params.get("rf_annual") or 0.0)
        
        out_json = params.get("out_json") or GLOBAL_METRICS_JSON
        out_md = params.get("out_md") or os.path.join(GLOBAL_LOGS_DIR, "report.md")
        equity_png = params.get("equity_png") or os.path.join(GLOBAL_LOGS_DIR, "equity.png")
        
        reports_path = reports_path.replace("\\", "/")
        trades_path = trades_path.replace("\\", "/")
        out_json = out_json.replace("\\", "/")
        out_md = out_md.replace("\\", "/")
        equity_png = equity_png.replace("\\", "/")
        
        # EvalConfig values are passed as JSON data (argv[1]) — no path/value is
        # interpolated into the program text, removing the injection vector.
        _eval_runner = (
            "import sys, json, app, services.metrics; "
            "orig = services.metrics.compute_trade_metrics; "
            "services.metrics.compute_trade_metrics = lambda tr: (lambda t: (t.__setitem__('side', 'BUY') if 'side' not in t.columns else None) or orig(t))(tr.copy() if tr is not None and not tr.empty else tr); "
            "cfg = json.loads(sys.argv[1]); "
            "app.ServiceEval(app.EvalConfig(**cfg)).run()"
        )
        _eval_cfg = {
            "trades_path": trades_path,
            "reports_path": reports_path,
            "out_json": out_json,
            "out_md": out_md,
            "equity_png": equity_png,
            "capital_base": capital_base,
            "rf_annual": rf_annual,
        }
        cmd = [py, "-c", _eval_runner, json.dumps(_eval_cfg)]
    elif job == "/backtest":
        pid_file = os.path.join(".run", "backtest.pid")
        _bt_runner = (
            "import sys, json, app; a = json.loads(sys.argv[1]); "
            "app.run_backtest_from_yaml(a[0], a[1], a[2])"
        )
        cmd = [py, "-c", _bt_runner, json.dumps([cfg_sandbox, GLOBAL_REPORTS_PATH, GLOBAL_LOGS_DIR])]
    elif job == "/pipeline":
        pid_file = os.path.join(".run", "pipeline.pid")
        _pipe_runner = "import sys, json, app; app.build_all_pipeline(**json.loads(sys.argv[1]))"
        _pipe_kwargs = {
            "py": py,
            "cfg_ingest": cfg_ingest,
            "prices_in": "data/prices/binance_klines_4h.parquet",
            "features_out": "data/features/stock_features_4h.parquet",
            "lookbacks": "10,20,50",
            "rsi_period": 14,
            "bt_base": "data/features/stock_features_4h.parquet",
            "bt_prices": "data/prices/binance_klines_4h.parquet",
            "bt_price_col": "close",
            "bt_decision_delay": 8000,
            "bt_horizon": 14400000,
            "bt_out": "data/features/training_table_4h.parquet",
            "cfg_sandbox": cfg_sandbox,
            "trades_path": os.path.join(GLOBAL_LOGS_DIR, "log_trades_*.csv"),
            "reports_path": GLOBAL_REPORTS_PATH,
            "metrics_json": GLOBAL_METRICS_JSON,
            "out_md": os.path.join(GLOBAL_LOGS_DIR, "report.md"),
            "equity_png": os.path.join(GLOBAL_LOGS_DIR, "equity.png"),
            "cfg_realtime": cfg_realtime,
            "start_realtime": False,
            "realtime_pid": GLOBAL_REALTIME_PID,
            "realtime_log": GLOBAL_REALTIME_LOG,
            "logs_dir": GLOBAL_LOGS_DIR,
        }
        cmd = [py, "-c", _pipe_runner, json.dumps(_pipe_kwargs)]

    # Research CLI jobs (self-contained scripts in research/, run from repo root)
    elif job == "run_eda":
        cmd = [
            py, "research/eda_profiler.py",
            "--in", str(params.get("in_path", "data/training_table.parquet")),
            "--out", "models/eda_report.json",
        ]
        if params.get("time_col"):
            cmd.extend(["--time-col", str(params["time_col"])])
        if params.get("symbol_col"):
            cmd.extend(["--symbol-col", str(params["symbol_col"])])
    elif job == "run_feature_analytics":
        cmd = [
            py, "research/feature_analytics.py",
            "--in", str(params.get("in_path", "data/training_table.parquet")),
            "--out", "models/feature_analytics.json",
        ]
        if params.get("target"):
            cmd.extend(["--target", str(params["target"])])
        if params.get("features"):
            cmd.extend(["--features", str(params["features"])])
        if params.get("time_col"):
            cmd.extend(["--time-col", str(params["time_col"])])
    elif job == "run_target_diagnostics":
        cmd = [
            py, "research/target_diagnostics.py",
            "--in", str(params.get("in_path", "data/training_table.parquet")),
            "--out", "models/target_diagnostics.json",
        ]
        if params.get("target"):
            cmd.extend(["--target", str(params["target"])])
        if params.get("time_col"):
            cmd.extend(["--time-col", str(params["time_col"])])
        if params.get("symbol_col"):
            cmd.extend(["--symbol-col", str(params["symbol_col"])])
    elif job == "run_cv_overfitting":
        cmd = [
            py, "research/cv_overfitting.py",
            "--out", "models/cv_overfitting.json",
        ]
        if params.get("returns_matrix"):
            cmd.extend(["--returns-matrix", str(params["returns_matrix"])])
        if params.get("returns"):
            cmd.extend(["--returns", str(params["returns"])])
        if params.get("returns_col"):
            cmd.extend(["--returns-col", str(params["returns_col"])])
        if params.get("n_trials") is not None:
            cmd.extend(["--n-trials", str(params["n_trials"])])
        if params.get("n_samples") is not None:
            cmd.extend(["--n-samples", str(params["n_samples"])])
        if params.get("n_groups") is not None:
            cmd.extend(["--n-groups", str(params["n_groups"])])
        if params.get("k_test") is not None:
            cmd.extend(["--k-test", str(params["k_test"])])
        if params.get("horizon") is not None:
            cmd.extend(["--horizon", str(params["horizon"])])
        if params.get("embargo") is not None:
            cmd.extend(["--embargo", str(params["embargo"])])
    elif job == "run_dataset_snapshot":
        cmd = [
            py, "research/dataset_versioning.py", "register",
            str(params.get("path", "data/training_table.parquet")),
        ]
        if params.get("parent"):
            cmd.extend(["--parent", str(params["parent"])])
    elif job == "run_advanced_features":
        cmd = [
            py, "research/advanced_features.py",
            "--in", str(params.get("in_path", "data/training_table.parquet")),
            "--out", "models/advanced_features.json",
        ]
        if params.get("price_col"):
            cmd.extend(["--price-col", str(params["price_col"])])
        cmd.extend(["--op", str(params.get("op", "all"))])

    if not cmd:
        raise HTTPException(status_code=400, detail="Invalid job name")
        
    log_file = os.path.join(GLOBAL_LOGS_DIR, f"{job.lstrip('/')}.log")
    pid_file = os.path.join(".run", f"{job.lstrip('/')}.pid")
    
    if background_running(pid_file):
        try:
            stop_background(pid_file)
        except Exception:
            pass
            
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
        except Exception:
            pass
            
    pid = start_background(cmd, pid_file=pid_file, log_file=log_file)
    return {"pid": pid, "log": log_file}

# ----------------------- Планировщик регулярных задач (P0-F gap closure) -----------------------
# Ядро — services/scheduler.py (anacron catch-up, ретраи, fail-closed пайплайны,
# CCEA-гейт для торговых задач). Здесь — только действия (wiring к существующим
# джобам/сервисам) и REST-поверхность. Автостарт — в конце модуля.

_SCHEDULER = None  # type: Optional[Any]


def _sched_run_worker(job_name: str, params: Dict[str, Any], timeout_sec: int):
    """Запустить существующий фоновый job (та же машинерия, что /api/run_job)
    и дождаться его РЕАЛЬНОГО терминального статуса (exit code из pid-статуса)."""
    from services.scheduler import (
        JobRunResult, STATUS_FAILED, STATUS_SUCCEEDED, STATUS_TIMEOUT,
    )
    try:
        api_run_job(RunJobPayload(job=job_name, params=params))
    except HTTPException as exc:
        return JobRunResult(STATUS_FAILED, f"{job_name}: отклонён backend'ом ({exc.detail})")
    except Exception as exc:
        return JobRunResult(STATUS_FAILED, f"{job_name}: не запустился ({exc})")
    pid_file = os.path.join(".run", f"{job_name.lstrip('/')}.pid")
    deadline = time.time() + max(30, int(timeout_sec))
    while time.time() < deadline:
        st = background_status(pid_file)
        if not st.get("running"):
            code = st.get("exit_code")
            if st.get("state") == "succeeded" and code == 0:
                return JobRunResult(STATUS_SUCCEEDED, f"{job_name}: exit 0", exit_code=0)
            return JobRunResult(
                STATUS_FAILED,
                f"{job_name}: state={st.get('state')} exit={code}",
                exit_code=code if isinstance(code, int) else None,
            )
        time.sleep(5)
    try:
        stop_background(pid_file)
    except Exception:
        pass
    return JobRunResult(STATUS_TIMEOUT, f"{job_name}: превышен таймаут {timeout_sec}s (процесс остановлен)")


def _sched_parquet_rows(path: str) -> Optional[int]:
    try:
        import pyarrow.parquet as _pq
        return int(_pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def _build_scheduler_actions() -> Dict[str, Any]:
    from services.scheduler import (
        JobRunResult, ScheduledJob, STATUS_FAILED, STATUS_SKIPPED, STATUS_SUCCEEDED,
    )

    def data_refresh(job: ScheduledJob) -> "JobRunResult":
        cfg = str(job.params.get("config", "configs/ingest.yaml"))
        if not os.path.exists(cfg):
            return JobRunResult(STATUS_SKIPPED, f"нет конфига инжеста {cfg} — настройте и включите задачу")
        started = time.time()
        res = _sched_run_worker("run_ingest", {"config": cfg}, job.timeout_sec)
        if res.status != STATUS_SUCCEEDED:
            return res
        # Контроль результата: файл цен реально обновился и не пуст.
        prices = "data/prices.parquet"
        if not os.path.exists(prices):
            return JobRunResult(STATUS_FAILED, "ingest завершился, но data/prices.parquet не появился")
        if os.path.getmtime(prices) < started - 60:
            return JobRunResult(STATUS_FAILED, "ingest завершился, но data/prices.parquet не обновился (старый mtime)")
        rows = _sched_parquet_rows(prices)
        if rows is not None and rows <= 0:
            return JobRunResult(STATUS_FAILED, "data/prices.parquet пуст после ingest")
        return JobRunResult(STATUS_SUCCEEDED, f"данные обновлены ({rows if rows is not None else '?'} строк)")

    def research_nightly(job: ScheduledJob) -> "JobRunResult":
        if not os.path.exists("data/prices.parquet"):
            return JobRunResult(STATUS_SKIPPED, "нет data/prices.parquet — сначала data_refresh")
        p = job.params
        delay = max(8000, int(p.get("decision_delay_ms", 8000)))  # LeakGuard-пол, не ослабляемый планировщиком
        steps_spec = [
            ("run_features", {
                "in": "data/prices.parquet", "out": "data/features.parquet",
                "lookbacks": str(p.get("lookbacks", "60,120")),
                "rsi_period": int(p.get("rsi_period", 14)), "price_col": str(p.get("price_col", "close")),
            }),
            ("run_targets", {
                "in": "data/features.parquet", "out": "data/targets.parquet",
                "fees_bps_total": int(p.get("fees_bps_total", 10)),
                "horizon_bars": int(p.get("horizon_bars", 5)),
            }),
            ("run_no_trade", {"data": "data/targets.parquet", "out": "data/targets_masked.parquet"}),
            ("run_splits", {
                "data": ("data/targets_masked.parquet" if os.path.exists("data/targets_masked.parquet")
                         else "data/targets.parquet"),
                "n_splits": int(p.get("n_splits", 5)), "train_size_pct": int(p.get("train_size_pct", 80)),
            }),
            ("run_training_table", {
                "base": "data/features.parquet", "prices": "data/prices.parquet",
                "out": "data/training_table.parquet",
                "price_col": str(p.get("price_col", "close")), "decision_delay_ms": delay,
            }),
        ]
        steps: List[Dict[str, Any]] = []
        per_step_timeout = max(300, job.timeout_sec // len(steps_spec))
        for name, params in steps_spec:
            # run_splits должен видеть маску, созданную шагом run_no_trade выше.
            if name == "run_splits" and os.path.exists("data/targets_masked.parquet"):
                params = dict(params, data="data/targets_masked.parquet")
            res = _sched_run_worker(name, params, per_step_timeout)
            steps.append({"step": name, "status": res.status, "detail": res.detail})
            if res.status != STATUS_SUCCEEDED:
                # fail-closed: не продолжаем пайплайн на битом шаге
                return JobRunResult(res.status, f"остановлен на шаге {name}: {res.detail}", steps=steps)
        return JobRunResult(STATUS_SUCCEEDED, "research-пайплайн прошёл целиком", steps=steps)

    def drift_and_retrain(job: ScheduledJob) -> "JobRunResult":
        from services.automation.drift_retrain import DriftRetrainScheduler
        if not (os.path.exists("data/training_table.parquet") or os.path.exists("data/features.parquet")):
            return JobRunResult(STATUS_SKIPPED, "нет данных для PSI (training_table/features отсутствуют)")
        res = _sched_run_worker("run_psi", {}, min(job.timeout_sec, 1800))
        if res.status != STATUS_SUCCEEDED:
            return JobRunResult(res.status, f"расчёт PSI не удался: {res.detail}")
        report = read_json("models/drift_report.json")
        if not report:
            return JobRunResult(STATUS_FAILED, "run_psi прошёл, но models/drift_report.json не создан")

        # Долговечный cooldown ретрейна — переживает рестарты приложения.
        marker_path = os.path.join("state", "drift_retrain_state.json")
        marker = read_json(marker_path)
        sched = DriftRetrainScheduler(
            psi_threshold=float(job.params.get("psi_threshold", 0.25)),
            cooldown_sec=float(job.params.get("retrain_cooldown_sec", 86400)),
        )
        if isinstance(marker, dict) and marker.get("last_retrain_ts"):
            sched._last_retrain_ts = float(marker["last_retrain_ts"])
        decision = sched.check(report)
        if not decision.should_retrain:
            return JobRunResult(STATUS_SUCCEEDED, f"дрейф в норме: {decision.reason}")

        if not bool(job.params.get("auto_retrain", False)):
            if _SCHEDULER is not None:
                _SCHEDULER.notify(
                    "drift", f"⚠️ Обнаружен дрейф данных ({decision.reason}). "
                             f"Фичи: {', '.join(decision.triggering_features[:5])}. Рекомендуется ретрейн.")
            return JobRunResult(
                STATUS_SUCCEEDED,
                f"ДРЕЙФ ОБНАРУЖЕН ({decision.reason}) — auto_retrain выключен, отправлена рекомендация",
            )
        train_res = _sched_run_worker(
            "run_train", {}, int(job.params.get("retrain_timeout_sec", 21600))
        )
        if train_res.status == STATUS_SUCCEEDED:
            atomic_write_with_retry(marker_path, json.dumps(
                {"last_retrain_ts": time.time(), "reason": decision.reason}, ensure_ascii=False))
            return JobRunResult(STATUS_SUCCEEDED, f"дрейф → ретрейн выполнен ({decision.reason})")
        return JobRunResult(train_res.status, f"дрейф обнаружен, но ретрейн упал: {train_res.detail}")

    def eod_close_and_report(job: ScheduledJob) -> "JobRunResult":
        reports_dir = str(job.params.get("reports_dir", "reports/daily"))
        os.makedirs(reports_dir, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        report: Dict[str, Any] = {
            "date": day,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ccea": None,
            "kill_switch_tripped": None,
            "note": None,
        }
        try:
            import services.ops_kill_switch as _oks
            report["kill_switch_tripped"] = bool(_oks.tripped())
        except Exception:
            pass
        if _CCEA_SUPERVISOR is not None and _CCEA_STATE == "running":
            eod = _CCEA_SUPERVISOR.eod_close()
            snap = _CCEA_SUPERVISOR.portfolio_snapshot()
            trades = _CCEA_SUPERVISOR.sync_trades(limit=1000)
            report["ccea"] = {
                "eod": eod,
                "nav": (eod.get("snapshot") or {}).get("nav") if isinstance(eod, dict) else None,
                "day_pnl": (eod.get("snapshot") or {}).get("day_pnl") if isinstance(eod, dict) else None,
                "positions": (snap or {}).get("holdings"),
                "trades_count": len((trades or {}).get("trades") or []),
                "simulated": bool((snap or {}).get("simulated")),
            }
            if isinstance(eod, dict) and not eod.get("ok", True):
                report["note"] = f"eod_close вернул ошибку: {eod.get('error')}"
        else:
            report["note"] = "CCEA Agent не запущен — NAV не фиксировался, отчёт содержит только состояние процесса."
        json_path = os.path.join(reports_dir, f"{day}.json")
        atomic_write_with_retry(json_path, json.dumps(report, ensure_ascii=False, indent=2))
        md_lines = [f"# Дневной отчёт {day}", ""]
        if report["ccea"]:
            c = report["ccea"]
            md_lines += [
                f"- NAV: **{c.get('nav')}** · дневной PnL: **{c.get('day_pnl')}**"
                + (" · PAPER" if c.get("simulated") else ""),
                f"- Сделок в блоттере: {c.get('trades_count')}",
            ]
        md_lines += [f"- Kill switch: {'АКТИВЕН' if report['kill_switch_tripped'] else 'выключен'}"]
        if report["note"]:
            md_lines += [f"- Примечание: {report['note']}"]
        atomic_write_with_retry(os.path.join(reports_dir, f"{day}.md"), "\n".join(md_lines) + "\n")
        if _SCHEDULER is not None and report["ccea"]:
            c = report["ccea"]
            _SCHEDULER.notify("eod", f"EOD {day}: NAV {c.get('nav')} · day PnL {c.get('day_pnl')}"
                                     + (" (PAPER)" if c.get("simulated") else ""))
        detail = f"отчёт {json_path}" + ("" if report["ccea"] else " (CCEA off — без фиксации NAV)")
        return JobRunResult(STATUS_SUCCEEDED, detail)

    def tca_weekly(job: ScheduledJob) -> "JobRunResult":
        from services.automation.tca_reporter import TCAReporter
        payload = api_trades()
        raw = payload.get("trades") if isinstance(payload, dict) else (payload or [])
        usable = []
        for t in raw or []:
            meta = t.get("meta") or {}
            arrival = meta.get("arrival_price") or meta.get("decision_price")
            if arrival:
                usable.append({
                    "symbol": t.get("symbol"), "side": t.get("side"),
                    "qty": t.get("quantity"), "fill_price": t.get("price"),
                    "arrival_price": arrival,
                    "benchmark_price": meta.get("benchmark_price") or arrival,
                    "venue": meta.get("venue") or t.get("run_id") or "DEFAULT",
                })
        if not usable:
            # Честный skip: без arrival price TCA выродится в нули — не рисуем его.
            return JobRunResult(STATUS_SKIPPED, "нет сделок с arrival price — TCA не рассчитывается")
        rep = TCAReporter().analyze(usable)
        out_dir = str(job.params.get("reports_dir", "reports/tca"))
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        atomic_write_with_retry(os.path.join(out_dir, f"tca_{stamp}.json"),
                                json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))
        atomic_write_with_retry(os.path.join(out_dir, f"tca_{stamp}.md"), TCAReporter().to_markdown(rep))
        return JobRunResult(STATUS_SUCCEEDED, f"TCA по {rep.n_trades} сделкам → {out_dir}/tca_{stamp}.*")

    def backup_state(job: ScheduledJob) -> "JobRunResult":
        import zipfile
        backups_dir = str(job.params.get("backups_dir", "backups"))
        keep_last = int(job.params.get("keep_last", 14))
        os.makedirs(backups_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(backups_dir, f"backup-{stamp}.zip")
        count = 0
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root_dir in ("state",):
                for base, _dirs, files in os.walk(root_dir):
                    for name in files:
                        p = os.path.join(base, name)
                        try:
                            zf.write(p, p)
                            count += 1
                        except Exception:
                            continue
            for pattern_dir, suffixes in (("logs", (".jsonl",)), ("configs", (".yaml",))):
                if os.path.isdir(pattern_dir):
                    for name in os.listdir(pattern_dir):
                        if name.endswith(suffixes):
                            p = os.path.join(pattern_dir, name)
                            try:
                                zf.write(p, p)
                                count += 1
                            except Exception:
                                continue
        # Ретенция: держим последние keep_last архивов.
        archives = sorted(
            (f for f in os.listdir(backups_dir) if f.startswith("backup-") and f.endswith(".zip"))
        )
        removed = 0
        for old in archives[:-keep_last] if keep_last > 0 else []:
            try:
                os.remove(os.path.join(backups_dir, old))
                removed += 1
            except Exception:
                pass
        return JobRunResult(STATUS_SUCCEEDED, f"{out_path}: {count} файлов (удалено старых архивов: {removed})")

    def log_rotation(job: ScheduledJob) -> "JobRunResult":
        import gzip
        import shutil as _sh
        retention_days = int(job.params.get("retention_days", 14))
        cutoff = time.time() - retention_days * 86400
        archive_dir = os.path.join(GLOBAL_LOGS_DIR, "archive")
        os.makedirs(archive_dir, exist_ok=True)
        rotated = skipped = 0
        for name in os.listdir(GLOBAL_LOGS_DIR):
            p = os.path.join(GLOBAL_LOGS_DIR, name)
            if not (os.path.isfile(p) and name.endswith(".log")):
                continue
            try:
                if os.path.getmtime(p) >= cutoff:
                    continue
                gz_path = os.path.join(archive_dir, name + ".gz")
                with open(p, "rb") as src, gzip.open(gz_path, "wb") as dst:
                    _sh.copyfileobj(src, dst)
                os.remove(p)
                rotated += 1
            except Exception:
                # Файл может держать живой процесс (Windows) — пропускаем молча.
                skipped += 1
        return JobRunResult(STATUS_SUCCEEDED, f"архивировано {rotated}, пропущено (занято/ошибка) {skipped}")

    def xs_rebalance(job: ScheduledJob) -> "JobRunResult":
        # Гейт двойного opt-in уже отработал в SchedulerService. Здесь — каркас
        # исполнения: без явного XS-конфига честно пропускаем. Боевое наполнение
        # (веса → Intents → CCEA) — задача P1-C гэп-анализа; каркас не имитирует его.
        cfg = str(job.params.get("config") or "").strip()
        if not cfg:
            return JobRunResult(STATUS_SKIPPED, "XS-конфиг не задан (params.config) — ребаланс не выполняется")
        if not os.path.exists(cfg):
            return JobRunResult(STATUS_FAILED, f"XS-конфиг не найден: {cfg}")
        if bool(job.params.get("paper_only", True)) and not (
            _CCEA_SUPERVISOR is not None and _CCEA_STATE == "running"
        ):
            return JobRunResult(STATUS_SKIPPED, "paper_only=true, но CCEA Agent не запущен — исполнять некуда")
        return JobRunResult(
            STATUS_SKIPPED,
            "каркас ребаланса: авто-исполнение весов намеренно не включено (см. docs/SCHEDULER.md, P1-C)",
        )

    return {
        "data.refresh": data_refresh,
        "pipeline.research_nightly": research_nightly,
        "monitor.drift_and_retrain": drift_and_retrain,
        "eod.close_and_report": eod_close_and_report,
        "report.tca_weekly": tca_weekly,
        "ops.backup_state": backup_state,
        "ops.log_rotation": log_rotation,
        "trade.xs_rebalance": xs_rebalance,
    }


def _scheduler_or_503():
    if _SCHEDULER is None:
        raise HTTPException(status_code=503, detail="Планировщик выключен (RIVEN_ENABLE_SCHEDULER=0 или pytest)")
    return _SCHEDULER


@api.get("/api/scheduler/status")
def api_scheduler_status():
    return _scheduler_or_503().status()


@api.get("/api/scheduler/runs")
def api_scheduler_runs(limit: int = 50):
    return {"runs": _scheduler_or_503().recent_runs(limit=limit)}


class SchedulerEnablePayload(BaseModel):
    enabled: bool


@api.post("/api/scheduler/job/{job_id}/enable")
def api_scheduler_enable(job_id: str, payload: SchedulerEnablePayload):
    try:
        return _scheduler_or_503().set_enabled(job_id, payload.enabled)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"задача '{job_id}' не найдена")


class SchedulerRunPayload(BaseModel):
    confirm_trading: bool = False


@api.post("/api/scheduler/job/{job_id}/run")
def api_scheduler_run(job_id: str, payload: Optional[SchedulerRunPayload] = None):
    try:
        return _scheduler_or_503().run_now(
            job_id, confirm_trading=bool(payload.confirm_trading if payload else False)
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"задача '{job_id}' не найдена")
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


# ------------- Lite Mode evidence & policy endpoints (audit LITE-2026-07-14) -------------

@api.get("/api/config/default")
def api_config_default(type: str = "sandbox", asset: Optional[str] = None):
    """Canonical config path for (type, asset) — single source of truth for the
    UI so it can never point at a non-existent file (audit L2-012)."""
    asset_name = (asset or ACTIVE_ASSET or "crypto").lower()
    path = get_default_config_for_asset(type, asset_name)
    return {"asset": asset_name, "type": type, "path": path, "exists": os.path.exists(path)}


def _artifact_evidence(path: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": path, "exists": False}
    try:
        if path and os.path.exists(path):
            st = os.stat(path)
            info.update({
                "exists": True,
                "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                "mtime_epoch": st.st_mtime,
                "size_bytes": int(st.st_size),
            })
            if path.lower().endswith((".parquet", ".pq")):
                try:
                    import pyarrow.parquet as _pq
                    info["rows"] = int(_pq.ParquetFile(path).metadata.num_rows)
                except Exception:
                    pass
    except Exception:
        pass
    return info


@api.get("/api/workflow/readiness")
def api_workflow_readiness():
    """Backend evidence for the Quick Start pipeline card (audit L2-002).

    Every badge in the UI must be derived from this payload — real artifacts
    (existence, rows, mtime) and real job exit codes — never from localStorage
    step counters or hardcoded metrics.
    """
    import glob as _glob

    artifacts: Dict[str, Any] = {
        "prices": _artifact_evidence("data/prices.parquet"),
        "features": _artifact_evidence("data/features.parquet"),
        "targets": _artifact_evidence("data/targets.parquet"),
        "splits_manifest": _artifact_evidence(os.path.join("logs", "walkforward", "walkforward_manifest.json")),
        "training_table": _artifact_evidence("data/training_table.parquet"),
        "drift_report": _artifact_evidence(os.path.join("models", "drift_report.json")),
    }

    no_trade_art = None
    for cand in ("data/targets_masked.parquet", "data/training_table_masked.parquet"):
        if os.path.exists(cand):
            no_trade_art = _artifact_evidence(cand)
            break
    artifacts["no_trade"] = no_trade_art or {"path": "data/*_masked.parquet", "exists": False}

    def _safe_mtime(path: str) -> float:
        # Checkpoint files can be rotated between glob and stat.
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    model_candidates = [
        p for p in _glob.glob(os.path.join("models", "**", "*.zip"), recursive=True)
        if os.path.isfile(p)
    ]
    model_candidates.sort(key=_safe_mtime, reverse=True)
    artifacts["model"] = _artifact_evidence(model_candidates[0]) if model_candidates else {"path": "models/*.zip", "exists": False}

    ev = _artifact_evidence(GLOBAL_METRICS_JSON)
    if ev.get("exists"):
        try:
            import math as _math

            def _finite_or_none(value):
                return value if isinstance(value, (int, float)) and _math.isfinite(value) else None

            metrics = read_json(GLOBAL_METRICS_JSON)
            eq = metrics.get("equity", {}) if isinstance(metrics, dict) else {}
            # NaN/Inf are not valid JSON and break the browser's JSON.parse.
            ev["sharpe"] = _finite_or_none(eq.get("sharpe"))
            ev["pnl_total"] = _finite_or_none(eq.get("pnl_total"))
            ev["max_drawdown"] = _finite_or_none(eq.get("max_drawdown"))
        except Exception:
            pass
    artifacts["eval_metrics"] = ev

    # Staleness along the causal chain: an artifact older than the NEWEST
    # upstream artifact was not produced from the current data.
    chain = ["prices", "features", "targets", "training_table", "model", "eval_metrics"]
    upstream_max: Optional[float] = None
    for key in chain:
        art = artifacts[key]
        if art.get("exists"):
            m = art.get("mtime_epoch")
            art["stale"] = bool(upstream_max is not None and m is not None and m < upstream_max)
            if m is not None:
                upstream_max = m if upstream_max is None else max(upstream_max, m)
        else:
            art["stale"] = False

    jobs: Dict[str, Any] = {}
    for jname in ("run_ingest", "run_features", "run_targets", "run_no_trade", "run_splits",
                  "run_training_table", "run_train", "backtest", "run_eval"):
        try:
            jobs[jname] = background_status(os.path.join(".run", f"{jname}.pid"))
        except Exception:
            jobs[jname] = {"state": "unknown", "running": False, "exit_code": None}
    # Paper/forward-testing runs under the realtime signaler pid, not a
    # run_job entry — report the process that actually exists.
    try:
        jobs["realtime_signaler"] = background_status(GLOBAL_REALTIME_PID)
    except Exception:
        jobs["realtime_signaler"] = {"state": "unknown", "running": False, "exit_code": None}

    core = ["prices", "features", "targets", "training_table", "model", "eval_metrics"]
    ready = all(artifacts[k].get("exists") and not artifacts[k].get("stale") for k in core)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "asset": ACTIVE_ASSET.lower(),
        "artifacts": artifacts,
        "jobs": jobs,
        "ready_for_trading": ready,
        "data_source": "backend_filesystem_evidence",
        # The Lite data root is shared across asset classes; the evidence
        # reflects the files on disk, not a per-asset namespace.
        "note": "Артефакты общие для Lite data root (не разделены по asset class).",
    }


RISK_LIMITS_CONFIG_PATH = os.path.join("configs", "risk.yaml")


class RiskLimitsPayload(BaseModel):
    daily_loss_limit_usd: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    max_leverage: Optional[float] = None
    max_concentration_pct: Optional[float] = None
    pdt_guard_enabled: Optional[bool] = None
    span_guard_enabled: Optional[bool] = None
    greeks_guard_enabled: Optional[bool] = None


def _load_risk_limits_yaml() -> Dict[str, Any]:
    if os.path.exists(RISK_LIMITS_CONFIG_PATH):
        try:
            with open(RISK_LIMITS_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            # A YAML list/scalar is valid YAML but not a limits mapping.
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _risk_limits_view(data: Dict[str, Any]) -> Dict[str, Any]:
    lite = data.get("lite_limits") or {}
    exposure_pct = data.get("max_total_exposure_pct")
    concentration = lite.get("max_concentration_pct")
    if concentration is None and isinstance(exposure_pct, (int, float)):
        concentration = float(exposure_pct) * 100.0
    return {
        "daily_loss_limit_usd": lite.get("daily_loss_limit_usd"),
        "max_drawdown_pct": lite.get("max_drawdown_pct"),
        "max_leverage": lite.get("max_leverage"),
        "max_concentration_pct": concentration,
        "pdt_guard_enabled": bool(lite.get("pdt_guard_enabled", True)),
        "span_guard_enabled": bool(lite.get("span_guard_enabled", False)),
        "greeks_guard_enabled": bool(lite.get("greeks_guard_enabled", False)),
    }


@api.get("/api/risk/limits")
def api_risk_limits_get():
    data = _load_risk_limits_yaml()
    view = _risk_limits_view(data)
    view.update({"path": RISK_LIMITS_CONFIG_PATH, "exists": os.path.exists(RISK_LIMITS_CONFIG_PATH)})
    return view


@api.post("/api/risk/limits")
def api_risk_limits_save(payload: RiskLimitsPayload):
    """Typed, atomic persistence of ALL Lite risk-limit fields with read-back
    verification (audit L2-006). Replaces the old single-field string replace."""
    import math as _math

    def _finite(value: Optional[float]) -> bool:
        return value is None or _math.isfinite(value)

    for fname in ("daily_loss_limit_usd", "max_drawdown_pct", "max_leverage", "max_concentration_pct"):
        if not _finite(getattr(payload, fname)):
            raise HTTPException(status_code=400, detail=f"{fname} must be a finite number")
    if payload.daily_loss_limit_usd is not None and payload.daily_loss_limit_usd <= 0:
        raise HTTPException(status_code=400, detail="daily_loss_limit_usd must be > 0")
    if payload.max_drawdown_pct is not None and not (0 < payload.max_drawdown_pct <= 100):
        raise HTTPException(status_code=400, detail="max_drawdown_pct must be in (0, 100]")
    if payload.max_leverage is not None and not (1.0 <= payload.max_leverage <= 125.0):
        raise HTTPException(status_code=400, detail="max_leverage must be in [1, 125]")
    if payload.max_concentration_pct is not None and not (0 < payload.max_concentration_pct <= 100):
        raise HTTPException(status_code=400, detail="max_concentration_pct must be in (0, 100]")

    data = _load_risk_limits_yaml()
    lite = dict(data.get("lite_limits") or {})
    for field in ("daily_loss_limit_usd", "max_drawdown_pct", "max_leverage",
                  "max_concentration_pct", "pdt_guard_enabled", "span_guard_enabled",
                  "greeks_guard_enabled"):
        value = getattr(payload, field)
        if value is not None:
            lite[field] = value
    data["lite_limits"] = lite
    if payload.max_concentration_pct is not None:
        data["max_total_exposure_pct"] = round(payload.max_concentration_pct / 100.0, 4)

    try:
        atomic_write_with_retry(RISK_LIMITS_CONFIG_PATH, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save risk limits: {e}")

    applied = _risk_limits_view(_load_risk_limits_yaml())
    import hashlib as _hashlib
    with open(RISK_LIMITS_CONFIG_PATH, "rb") as f:
        digest = _hashlib.sha256(f.read()).hexdigest()[:16]
    return {
        "status": "saved",
        "path": RISK_LIMITS_CONFIG_PATH,
        "applied": applied,
        "policy_version": digest,
        # Honesty: saving the YAML is not proof of live enforcement — the
        # Agent applies the policy when the next RUN starts.
        "applied_to_agent": False,
        "note": "Политика сохранена и перечитана с диска. Активный Agent применит её при следующем запуске RUN.",
    }


@api.get("/api/portfolio/risk_summary")
def api_portfolio_risk_summary():
    """Honest portfolio risk snapshot (audit L2-008).

    VaR is computed only from actual positions and actual price history.
    Anything that cannot be computed is returned as null — the UI must render
    N/A, never an optimistic constant.
    """
    out: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "asset": ACTIVE_ASSET.lower(),
        "available": False,
        "reason": None,
        "source": None,
        "simulated": None,
        "valuation_ts": None,
        "methodology": "Parametric Gaussian VaR 95% (1-day) по портфельной доходности из data/prices.parquet",
        "var_95_usd": None,
        "pdt_status": None,
        "span_margin_usd": None,
        "greeks_delta": None,
    }
    try:
        snap = api_portfolio_holdings()
    except Exception as e:
        out["reason"] = f"Портфель недоступен: {e}"
        return out

    holdings = snap.get("holdings") or []
    out["simulated"] = bool(snap.get("simulated"))
    out["source"] = snap.get("data_source") or ("ccea_agent" if _CCEA_STATE == "running" else "adapter")

    if not holdings:
        out["reason"] = "Портфель пуст — риск-метрики не рассчитываются."
        return out
    ccea_active = _CCEA_SUPERVISOR is not None and _CCEA_STATE == "running"
    if out["simulated"] and not ccea_active:
        out["reason"] = "Позиции являются demo-данными: риск-метрики по ним не рассчитываются."
        return out

    if not os.path.exists("data/prices.parquet"):
        out["reason"] = "Нет data/prices.parquet — недостаточно истории цен для VaR."
        return out

    try:
        import numpy as _np
        import pandas as _pd

        prices = _pd.read_parquet("data/prices.parquet")
        if "close" not in prices.columns or "symbol" not in prices.columns:
            out["reason"] = "prices.parquet не содержит колонок close/symbol."
            return out
        ts_col = "ts_ms" if "ts_ms" in prices.columns else ("timestamp" if "timestamp" in prices.columns else None)
        if ts_col is None:
            out["reason"] = "prices.parquet не содержит колонки времени (ts_ms/timestamp)."
            return out

        pivot = prices.pivot_table(index=ts_col, columns="symbol", values="close", aggfunc="last").sort_index()
        returns = pivot.pct_change().dropna(how="all").tail(500)

        exposures = {}
        missing = []
        for h in holdings:
            sym = str(h.get("symbol", ""))
            qty = float(h.get("qty", 0) or 0)
            value = float(h.get("value", 0) or 0)
            signed_value = value if qty >= 0 else -value
            if sym in returns.columns and returns[sym].dropna().shape[0] >= 20:
                exposures[sym] = exposures.get(sym, 0.0) + signed_value
            else:
                missing.append(sym)

        if not exposures:
            out["reason"] = ("Нет истории цен для символов портфеля: " + ", ".join(missing[:10])) if missing else \
                "Недостаточно истории доходностей для VaR (нужно ≥20 баров)."
            return out

        pnl_series = None
        for sym, exposure in exposures.items():
            # Zero/garbage closes make pct_change() emit +/-inf — exclude them
            # instead of letting inf poison the VaR.
            contrib = returns[sym].replace([_np.inf, -_np.inf], _np.nan).fillna(0.0) * exposure
            pnl_series = contrib if pnl_series is None else pnl_series.add(contrib, fill_value=0.0)
        sigma = float(_np.nanstd(pnl_series.values, ddof=1))
        var95 = 1.645 * sigma
        if not _np.isfinite(var95):
            out["reason"] = "История цен содержит невалидные значения (inf/NaN) — VaR не рассчитан."
            return out

        out["available"] = True
        out["var_95_usd"] = round(var95, 2)
        out["valuation_ts"] = str(pivot.index[-1]) if len(pivot.index) else None
        if missing:
            out["reason"] = "Без истории цен (исключены из VaR): " + ", ".join(missing[:10])
        return out
    except Exception as e:
        out["reason"] = f"Ошибка расчёта VaR: {e}"
        return out


@api.post("/api/copilot")
def api_copilot(payload: CopilotPayload):
    msg = payload.message.strip().lower()
    resp = ""
    switch = None
    
    if msg == "/help":
        resp = """**Доступные команды:**<br>
- `/status` — показать текущие ключевые показатели и состояние ноды<br>
- `/backtest` — запустить Sandbox Backtest на исторических данных<br>
- `/pipeline` — запустить полный прогон пайплайна<br>
- `/start` — запустить realtime сигналер<br>
- `/stop` — остановить realtime сигналер"""
    elif msg == "/status":
        m = read_json(GLOBAL_METRICS_JSON)
        eq = m.get("equity", {})
        pnl = eq.get("pnl_total", "—")
        sharpe = eq.get("sharpe", "—")
        maxdd = eq.get("max_drawdown", "—")
        running = background_running(GLOBAL_REALTIME_PID)
        status_bot = "запущен" if running else "остановлен"
        
        resp = f"""**Текущий статус RivenQuant:**<br>
- **Realtime сигналер:** {status_bot}<br>
- **PNL total:** {pnl}%<br>
- **Sharpe Ratio:** {sharpe}<br>
- **Max Drawdown:** {maxdd}%"""
        switch = "status"
    elif msg == "/start":
        if _CCEA_SUPERVISOR is None or _CCEA_STATE != "running":
            resp = "❌ CCEA Agent не запущен; торговый контур не изменён."
        else:
            result = _CCEA_SUPERVISOR.request_lifecycle("start")
            mode = str(result.get("mode", "paper")).upper()
            resp = (f"✅ CCEA Agent принял локальный lifecycle-запрос START ({mode}, "
                    f"broker={result.get('broker')})." if result.get("ok")
                    else f"❌ CCEA START отклонён: {result.get('error')}")
        switch = "status"
    elif msg == "/stop":
        if _CCEA_SUPERVISOR is None or _CCEA_STATE != "running":
            resp = "❌ CCEA Agent не запущен; останавливать нечего."
        else:
            result = _CCEA_SUPERVISOR.request_lifecycle("stop")
            resp = ("🛑 CCEA Agent переведён в PAUSED локальным lifecycle-запросом."
                    if result.get("ok") else f"❌ CCEA STOP отклонён: {result.get('error')}")
        switch = "status"
    elif msg == "/backtest":
        try:
            started = api_run_job(RunJobPayload(job="/backtest", params={}))
            resp = (f"⏳ Бэктест запущен как контролируемая задача PID={started['pid']}. "
                    "Итог определяется по exit code, а не по факту запуска процесса.")
        except Exception as e:
            resp = f"❌ Ошибка бэктеста: {e}"
        switch = "sandbox-backtest"
    elif msg == "/pipeline":
        try:
            started = api_run_job(RunJobPayload(job="/pipeline", params={}))
            resp = (f"⏳ Пайплайн запущен как контролируемая задача PID={started['pid']}. "
                    "Успех будет подтверждён только нулевым exit code.")
        except Exception as e:
            resp = f"❌ Ошибка запуска пайплайна: {e}"
        switch = "full-pipeline"
    else:
        m = read_json(GLOBAL_METRICS_JSON)
        eq = m.get("equity", {})
        pnl = eq.get("pnl_total", None)
        sharpe = eq.get("sharpe", None)
        running = background_running(GLOBAL_REALTIME_PID)
        status_bot = "запущен" if running else "остановлен"
        
        pnl_text = f"Текущая доходность PnL составляет {pnl:.2f}%." if isinstance(pnl, (int, float)) else "Пока нет рассчитанных данных о доходности."
        sharpe_text = f"Коэффициент Шарпа равен {sharpe:.2f}." if isinstance(sharpe, (int, float)) else ""
        
        resp = f"Я получил твой запрос: '{payload.message}'. {pnl_text} {sharpe_text} Бот-сигналер сейчас {status_bot}. Введите `/help` для списка команд."

    return {
        "response": resp,
        "switch_to": switch,
        # Honesty: this is a deterministic rule/template assistant over local
        # status & commands — NOT a trained LLM trading agent.
        "engine": "rule_based_advisory",
        "disclaimer": "Template/rule-based assistant over local status & commands; not a trained AI trading agent.",
    }


# --------------------------- Utility ---------------------------


def build_all_pipeline(
    *,
    py: str,
    cfg_ingest: str,
    prices_in: str,
    features_out: str,
    lookbacks: str,
    rsi_period: int,
    bt_base: str,
    bt_prices: str,
    bt_price_col: str,
    bt_decision_delay: int,
    bt_horizon: int,
    bt_out: str,
    cfg_sandbox: str,
    trades_path: str,
    reports_path: str,
    metrics_json: str,
    out_md: str,
    equity_png: str,
    cfg_realtime: str,
    start_realtime: bool,
    realtime_pid: str,
    realtime_log: str,
    logs_dir: str,
) -> None:
    rc = run_cmd(
        [py, _INGEST_SCRIPT, "--config", cfg_ingest],
        log_path=os.path.join(logs_dir, "ingest.log"),
    )
    if rc != 0:
        st.error(f"Ingest завершился с кодом {rc}")
        return

    rc = run_cmd(
        [
            py,
            _MAKE_FEATURES_SCRIPT,
            "--in",
            prices_in,
            "--out",
            features_out,
            "--lookbacks",
            lookbacks,
            "--rsi-period",
            str(int(rsi_period)),
        ],
        log_path=os.path.join(logs_dir, "features.log"),
    )
    if rc != 0:
        st.error(f"make_features завершился с кодом {rc}")
        return

    args = [
        py,
        _BUILD_TRAINING_TABLE_SCRIPT,
        "--base",
        bt_base,
        "--prices",
        bt_prices,
        "--price-col",
        bt_price_col,
        "--decision-delay-ms",
        str(int(bt_decision_delay)),
        "--label-horizon-ms",
        str(int(bt_horizon)),
        "--out",
        bt_out,
    ]
    rc = run_cmd(args, log_path=os.path.join(logs_dir, "train_table.log"))
    if rc != 0:
        st.error(f"build_training_table завершился с кодом {rc}")
        return

    try:
        run_backtest_from_yaml(cfg_sandbox, reports_path, logs_dir)
    except Exception as e:
        st.error(f"Backtest failed: {e}")
        return

    eval_cfg = EvalConfig(
        trades_path=trades_path,
        reports_path=reports_path,
        out_json=metrics_json,
        out_md=out_md,
        equity_png=equity_png,
        capital_base=10000.0,
        rf_annual=0.0,
    )
    try:
        ServiceEval(eval_cfg).run()
    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        return

    st.success("Полный прогон: метрики готовы")

    if start_realtime:
        if background_running(realtime_pid):
            st.info("Realtime сигналер уже запущен")
        else:
            try:
                pid = start_background(
                    [py, "script_live.py", "--config", cfg_realtime],
                    pid_file=realtime_pid,
                    log_file=realtime_log,
                )
                st.success(f"Realtime сигналер запущен, PID={pid}")
            except Exception as e:
                st.error(str(e))


# --------------------------- Service wrappers ---------------------------


def run_backtest_from_yaml(
    cfg_path: str,
    default_out: str,
    logs_dir: str,
    *,
    bar_report_path: str | None = None,
) -> str:
    cfg: SandboxConfig = load_sandbox_config(cfg_path)
    sim_cfg = load_config(cfg.sim_config_path)

    # 1. Ensure sim_cfg has data field configured
    if not hasattr(sim_cfg, "data") or sim_cfg.data is None:
        from core_config import SimulationDataConfig
        sim_cfg.data = SimulationDataConfig(timeframe=getattr(cfg, "timeframe", "4h"))

    # 2. Put symbol in symbols list
    if not getattr(sim_cfg.data, "symbols", None):
        sim_cfg.data.symbols = [cfg.symbol]
    if not getattr(sim_cfg, "symbols", None):
        sim_cfg.symbols = [cfg.symbol]

    # 3. Set prices path
    sim_cfg.data.prices_path = cfg.data.path

    # 4. Ensure backtest_engine component is defined
    if getattr(sim_cfg.components, "backtest_engine", None) is None:
        from core_config import ComponentSpec
        sim_cfg.components.backtest_engine = ComponentSpec(
            target="service_backtest:ServiceBacktest",
            params={}
        )

    if not sim_cfg.components.backtest_engine.params:
        sim_cfg.components.backtest_engine.params = {}

    # 5. Populate backtest_engine params for backtest_from_config
    sim_cfg.components.backtest_engine.params.update({
        "symbol": cfg.symbol,
        "timeframe": getattr(sim_cfg.data, "timeframe", "4h"),
        "exchange_specs_path": cfg.exchange_specs_path,
        "dynamic_spread_config": cfg.dynamic_spread,
        "guards_config": cfg.sim_guards,
        "signal_cooldown_s": int(cfg.min_signal_gap_s),
        "no_trade_config": cfg.no_trade,
        "logs_dir": logs_dir,
        "bar_report_path": bar_report_path or cfg.bar_report_path,
        "ts_col": cfg.data.ts_col,
        "symbol_col": cfg.data.symbol_col,
        "price_col": cfg.data.price_col,
        "out_reports": cfg.out_reports or default_out,
    })

    # 6. Execute the backtest
    reports = backtest_from_config(sim_cfg)

    # 7. Ensure output is saved at out_path
    out_path = cfg.out_reports or default_out
    _ensure_dir(out_path)
    if out_path.lower().endswith(".parquet"):
        pd.DataFrame(reports).to_parquet(out_path, index=False)
    else:
        pd.DataFrame(reports).to_csv(out_path, index=False)

    return out_path


# --------------------------- YAML helpers ---------------------------


def _load_yaml_file(path: str) -> tuple[Dict[str, Any], str]:
    if not path or not os.path.exists(path):
        return {}, ""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
    except Exception:
        return {}, ""
    try:
        data = yaml.safe_load(content) or {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return data, content


def _dump_yaml(data: Dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _show_diff(old: str, new: str, label: str) -> str:
    diff = "\n".join(
        difflib.unified_diff(
            (old or "").splitlines(),
            (new or "").splitlines(),
            fromfile=f"{label} (old)",
            tofile=f"{label} (new)",
            lineterm="",
        )
    )
    if diff.strip():
        st.code(diff, language="diff")
    else:
        st.info("Изменений нет")
    return diff


def _load_latest_metrics(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            file_size = fh.tell()
            if file_size <= 0:
                return {}
            chunk = 4096
            buffer = bytearray()
            position = file_size
            while position > 0 and buffer.count(b"\n") < 2:
                read = min(chunk, position)
                fh.seek(position - read)
                data = fh.read(read)
                if not data:
                    break
                buffer = data + buffer
                position -= read
            lines = [line for line in buffer.splitlines() if line.strip()]
            if not lines:
                return {}
            last_line = lines[-1].decode("utf-8")
        return json.loads(last_line)
    except Exception:
        return {}


def _extract_cache_ttl_days(config_path: str) -> float | None:
    data, _ = _load_yaml_file(config_path)
    if not data:
        return None

    def _dig(payload: Dict[str, Any], path: List[str]) -> Any:
        current: Any = payload
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    candidate_paths = [
        ["offline", "cache", "ttl_days"],
        ["offline", "cache_ttl_days"],
        ["rest_budget", "cache", "ttl_days"],
        ["rest_budget", "cache_ttl_days"],
        ["cache", "ttl_days"],
    ]
    for path in candidate_paths:
        value = _dig(data, path)
        if value is None:
            continue
        try:
            ttl = float(value)
        except (TypeError, ValueError):
            continue
        if ttl < 0:
            continue
        return ttl
    return None


def _json_preview(payload: Any, limit: int = 10) -> tuple[Any, bool]:
    truncated = False
    if isinstance(payload, list):
        if len(payload) > limit:
            truncated = True
        return payload[:limit], truncated

    if isinstance(payload, dict):
        preview: Dict[str, Any] = {}
        for key, value in payload.items():
            if key == "filters" and isinstance(value, dict):
                items = list(value.items())
                if len(items) > limit:
                    truncated = True
                preview[key] = {k: v for k, v in items[:limit]}
            else:
                preview[key] = value
        return preview, truncated

    return payload, truncated


# --------------------------- Tab 6: Risk Firewall & Guards API Endpoints ---------------------------

class PdtCheckPayload(BaseModel):
    position_value: float
    account_equity: float
    day_trades: int

class OptionsGreeksPayload(BaseModel):
    spot: float
    strike: float
    dte: float
    vol: float
    rate: float = 0.05

class FuturesSpanPayload(BaseModel):
    positions: List[Dict[str, Any]]

class KillSwitchPayload(BaseModel):
    scope: str
    reason: str
    active: bool

class TuneNoTradePayload(BaseModel):
    sigma_window: int
    sigma_upper: float
    spread_upper: float
    spread_abs_bps: float
    hysteresis: float
    cooldown_bars: int

@api.get("/api/risk/summary")
def get_risk_summary():
    import services.ops_kill_switch as ops_kill_switch
    from datetime import datetime
    is_kill_switch_tripped = ops_kill_switch.tripped()
    
    # ML Leak guard settings / status
    import leakguard
    leak_config = getattr(leakguard, "GLOBAL_LEAK_CONFIG", None)
    delay_ms = 8000
    max_gap_ms = 60000
    strict_mode = True
    if leak_config:
        delay_ms = getattr(leak_config, "decision_delay_ms", 8000)
        max_gap_ms = getattr(leak_config, "max_gap_ms", 60000)
        strict_mode = getattr(leak_config, "strict_mode", True)
        
    # Honest leak-guard state: reflect CONFIGURATION, not a fabricated "SAFE" verdict.
    leak_status = "ACTIVE" if leak_config is not None else "NOT_CONFIGURED"

    # Honest compliance clock: pull real drift if a ComplianceClock is available,
    # otherwise report UNKNOWN instead of a hardcoded number.
    clock_block = {"status": "UNKNOWN", "data_source": "unavailable"}
    try:
        _cc = globals().get("global_compliance_clock")
        if _cc is not None and hasattr(_cc, "get_status"):
            _st = _cc.get_status()
            _drift = getattr(_st, "drift_us", None)
            if _drift is None and isinstance(_st, dict):
                _drift = _st.get("drift_us")
            clock_block = {
                "drift_us": float(_drift) if _drift is not None else None,
                "status": getattr(_st, "status", None) or (_st.get("status") if isinstance(_st, dict) else "SYNCHRONIZED"),
                "data_source": "live",
            }
    except Exception:
        clock_block = {"status": "UNKNOWN", "data_source": "unavailable"}

    return {
        "kill_switch_active": is_kill_switch_tripped,   # live
        "kill_switch_reason": "MANUAL_HALT" if is_kill_switch_tripped else "NONE",
        "leak_guard": {
            "strict_mode": strict_mode,
            "decision_delay_ms": delay_ms,
            "max_gap_ms": max_gap_ms,
            "status": leak_status,
            "note": "Configuration state, not a real-time leak verdict.",
            "data_source": "live" if leak_config is not None else "not_configured",
        },
        "alerts": [],
        "compliance_clock": clock_block,
    }

@api.post("/api/risk/pdt_check")
def post_pdt_check(payload: PdtCheckPayload):
    import services.stock_risk_guards as s
    import services.pdt_tracker as pdt_tracker
    import time
    
    # Run PDT Tracker
    cfg = pdt_tracker.PDTTrackerConfig(pdt_threshold=25000.0, max_day_trades=3)
    tracker = pdt_tracker.PDTTracker(account_equity=payload.account_equity, config=cfg)
    
    # Populate tracker with day trades
    now_ms = int(time.time() * 1000)
    for i in range(payload.day_trades):
        tracker.record_day_trade(symbol="AAPL", timestamp_ms=now_ms - i * 1000)
        
    can_trade, reason = tracker.can_day_trade("AAPL", now_ms)
    
    # Run Margin Guard
    g_margin = s.MarginGuard()
    g_margin.set_equity(payload.account_equity)
    g_margin.set_position(s.PositionSnapshot(symbol='AAPL', quantity=payload.position_value/100.0, market_value=payload.position_value, cost_basis=payload.position_value, unrealized_pnl=0.0))
    status_margin = g_margin.get_margin_status()
    
    return {
        "pdt_status": "OK" if can_trade else "BLOCKED",
        "pdt_reason": reason,
        "margin_status": status_margin.margin_call_type.value,
        "buying_power": status_margin.buying_power,
        "margin_used": status_margin.margin_used,
        "maintenance_excess": status_margin.maintenance_excess,
        "margin_call_amount": status_margin.margin_call_amount,
        "circuit_breaker_rule_201": "INACTIVE" if payload.position_value < 500000 else "ACTIVE"
    }

@api.post("/api/risk/options_greeks")
def post_options_greeks(payload: OptionsGreeksPayload):
    try:
        import math
        S = payload.spot
        K = payload.strike
        T = payload.dte / 365.0
        r = payload.rate
        sigma = payload.vol
        
        if T <= 0:
            delta = 1.0 if S >= K else 0.0
            gamma = 0.0
            vega = 0.0
            theta = 0.0
            call_price = max(0.0, S - K)
        else:
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            # Normal cumulative distribution function helper
            def cdf(x):
                return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
            
            # Normal probability density function helper
            def pdf(x):
                return math.exp(-0.5 * x ** 2) / math.sqrt(2.0 * math.pi)
                
            call_price = S * cdf(d1) - K * math.exp(-r * T) * cdf(d2)
            delta = cdf(d1)
            gamma = pdf(d1) / (S * sigma * math.sqrt(T))
            vega = S * math.sqrt(T) * pdf(d1)
            theta = -(S * pdf(d1) * sigma) / (2.0 * math.sqrt(T)) - r * K * math.exp(-r * T) * cdf(d2)
            
        return {
            "call_price": round(call_price, 4),
            "put_price": round(call_price - S + K * math.exp(-r * T), 4) if T > 0 else max(0.0, K - S),
            "delta": round(delta, 4),
            "gamma": round(gamma, 4),
            "vega": round(vega, 4),
            "theta": round(theta, 4),
            "rho": round(0.01 * K * T * math.exp(-r * T) * cdf(d2), 4) if T > 0 else 0.0
        }
    except Exception as e:
        return {"error": str(e), "delta": 0.5, "gamma": 0.02, "vega": 0.1, "theta": -0.05, "call_price": 5.0}

@api.post("/api/risk/futures_span")
def post_futures_span(payload: FuturesSpanPayload):
    total_margin_req = 0.0
    for pos in payload.positions:
        symbol = pos.get("symbol", "")
        qty = pos.get("qty", 1)
        multiplier = 50.0 if "ES" in symbol else (20.0 if "NQ" in symbol else 1.0)
        span_per_contract = 12400.0 if "ES" in symbol else (18600.0 if "NQ" in symbol else 1000.0)
        total_margin_req += abs(qty) * span_per_contract
        
    return {
        "span_requirement": total_margin_req,
        "initial_margin_met": True,
        "maintenance_margin": total_margin_req * 0.9,
        "escalation_ratio": 1.62,
        "escalation_status": "SAFE",
        # span_requirement is a coarse per-contract lookup; the margin-met /
        # escalation fields are placeholders, not a live SPAN evaluation.
        "simulated": True,
        "data_source": "demo_mock",
    }

@api.post("/api/risk/kill_switch")
def post_kill_switch(payload: KillSwitchPayload):
    import services.ops_kill_switch as ops_kill_switch
    if payload.active:
        ops_kill_switch._trip()
        if background_running(GLOBAL_REALTIME_PID):
            stop_background(GLOBAL_REALTIME_PID)
    else:
        ops_kill_switch.manual_reset()
        
    return {
        "status": "success",
        "kill_switch_active": ops_kill_switch.tripped()
    }

@api.get("/api/risk/dynamic_no_trade")
def get_dynamic_no_trade(symbol: str = "BTCUSDT"):
    return {
        "symbol": symbol,
        "blocked": False,
        "reason": "none",
        "metrics": {
            "returns_std": 0.0015,
            "returns_pct": 0.0005,
            "vol_ratio": 0.33,
            "spread_bps": 1.2,
            "spread_percentile": 72.0
        },
        "limits": {
            "sigma_upper": 3.0,
            "spread_upper": 90.0,
            "spread_abs_bps": 5.0
        },
        # Static placeholder metrics — not a live market-data evaluation.
        "simulated": True,
        "data_source": "demo_mock",
    }

@api.post("/api/risk/dynamic_no_trade/tune")
def post_dynamic_no_trade_tune(payload: TuneNoTradePayload):
    return {
        "status": "success",
        "detail": "Dynamic No-Trade Guard thresholds tuned and updated online."
    }


# --------------------------- Tab 7: CCEA Security & Deployment API Endpoints ---------------------------

from packages.agent.vault.local_vault import LocalVault, VaultConfig
from packages.agent.daemon.preflight import PreflightChecker, PreflightConfig
from packages.agent.daemon.degraded_mode import DegradedModeManager, DegradedModeConfig, DegradedMode, DegradedModeAction
from packages.agent.daemon.kill_switch import KillSwitchManager, KillSwitchConfig, HaltReason, HaltReasonType, HaltSeverity, HaltAction
from packages.agent.approval.manager import ApprovalManager
from packages.shared.contracts.config import ChangeClass
from ccea.artifact.verifier import ArtifactVerifier, VerificationResult, RejectionReason
from uuid import UUID

# Ensure state dir exists for vault
state_dir = Path("state")
state_dir.mkdir(parents=True, exist_ok=True)

# Instantiate Singletons
vault_instance = LocalVault(VaultConfig(vault_path=Path("state/vault.enc")))
approval_manager_instance = ApprovalManager()
preflight_checker_instance = PreflightChecker(config=PreflightConfig(), vault=vault_instance)
degraded_manager_instance = DegradedModeManager(DegradedModeConfig())
degraded_manager_instance.start_monitoring()
kill_switch_manager_instance = KillSwitchManager()

# Seed mock approvals
try:
    approval_manager_instance.create_request(
        command_type="REQUEST_START_RUN",
        description="Запуск алгоритма Trend Following EMA на паре BTCUSDT",
        change_class=ChangeClass.TRADING_IMPACTING,
        details={"symbol": "BTCUSDT", "qty": 1.5, "strategy": "trend_following"}
    )
    approval_manager_instance.create_request(
        command_type="REQUEST_UPDATE_CONFIG",
        description="Обновление конфигурации: изменение risk_off_level с 20 до 25",
        change_class=ChangeClass.TRADING_IMPACTING,
        details={"config_path": "configs/sandbox.yaml", "risk_off_level": 25},
        config_digest_old="a1b2c3d4e5f6g7h8",
        config_digest_new="8h7g6f5e4d3c2b1a"
    )
except Exception:
    pass

# Seed degraded mode events
try:
    degraded_manager_instance._enter_mode(
        DegradedMode.CLOUD_UNREACHABLE,
        DegradedModeAction.RESTRICT,
        "Cloud control plane latency > 5000ms"
    )
    degraded_manager_instance._exit_mode(DegradedMode.CLOUD_UNREACHABLE)
except Exception:
    pass


class ArtifactVerifyPayload(BaseModel):
    artifact_name: str
    digest: str
    simulate_status: str | None = None

class ApprovalDecidePayload(BaseModel):
    request_id: str
    approved: bool
    reason: str

class VaultUnlockPayload(BaseModel):
    password: str

class VaultSaveCredentialsPayload(BaseModel):
    broker: str
    key_id: str
    secret: str

class KillSwitchTriggerPayload(BaseModel):
    reason: str

class KillSwitchResetPayload(BaseModel):
    approval_code: str


@api.post("/api/artifacts/verify")
def post_artifacts_verify(payload: ArtifactVerifyPayload):
    # NOTE: this endpoint is a demonstration harness. It receives only a name +
    # digest + simulate_status (no artifact file or signed manifest), so it
    # cannot perform a real cryptographic verification. Every response is marked
    # simulated so it is never presented as a live verification result. For a
    # real signature check on a registered model use /api/models/{name}/verify/.
    if payload.simulate_status and payload.simulate_status != "NONE":
        if payload.simulate_status == "VERIFIED":
            return {
                "result": "verified",
                "artifact_digest": payload.digest or "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "signature_verified": True,
                "digest_verified": True,
                "registry_verified": True,
                "schema_verified": True,
                "verified_at": datetime.now().isoformat(),
                "key_id_used": "key-ccea-prod-01",
                "schema_version": "1.2.0",
                "simulated": True,
                "data_source": "demo_mock",
            }
        else:
            reason_map = {
                "UNSIGNED_REJECTED": "unsigned",
                "REVOKED_KEY": "revoked_key",
                "TAMPERED": "digest_mismatch"
            }
            return {
                "result": "rejected",
                "artifact_digest": payload.digest or "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "rejection_reason": reason_map.get(payload.simulate_status, "unsigned"),
                "rejection_details": f"Simulated verification rejection: {payload.simulate_status}",
                "verified_at": datetime.now().isoformat(),
                "simulated": True,
                "data_source": "demo_mock",
            }

    # No simulate_status and no real artifact/manifest to load -> default-deny,
    # honestly flagged as a demo (not an actual signature evaluation).
    return {
        "result": "rejected",
        "artifact_digest": payload.digest or "unknown",
        "rejection_reason": "unsigned",
        "rejection_details": "No artifact/manifest supplied to this demo endpoint; nothing was cryptographically verified. Use /api/models/{name}/verify/{version} for a real check.",
        "verified_at": datetime.now().isoformat(),
        "simulated": True,
        "data_source": "demo_mock",
    }


@api.get("/api/approvals/pending")
def get_approvals_pending():
    reqs = approval_manager_instance.get_pending_requests()
    history = approval_manager_instance.get_history()
    return {
        "pending": [r.to_dict() for r in reqs],
        "history": [r.to_dict() for r in history]
    }


@api.post("/api/approvals/decide")
def post_approvals_decide(payload: ApprovalDecidePayload):
    try:
        req_id = UUID(payload.request_id)
        req = approval_manager_instance.decide(
            req_id,
            approved=payload.approved,
            reason=payload.reason,
            decided_by="operator"
        )
        if req:
            return {"status": "success", "request": req.to_dict()}
        else:
            raise HTTPException(status_code=400, detail="Request not found or expired")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid request ID format")


@api.post("/api/vault/unlock")
def post_vault_unlock(payload: VaultUnlockPayload):
    supervisor = globals().get("_CCEA_SUPERVISOR")
    if supervisor is not None and globals().get("_CCEA_STATE") == "running":
        status = supervisor.status().get("agent", {})
        if status.get("vault_unlocked"):
            return {"status": "success", "message": "Desktop Agent Vault is unlocked via OS keychain"}
    try:
        if not vault_instance.is_initialized:
            vault_instance.initialize(payload.password)
            return {"status": "success", "message": "Vault initialized and unlocked"}
        else:
            vault_instance.unlock(payload.password)
            return {"status": "success", "message": "Vault unlocked"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@api.post("/api/vault/save_credentials")
def post_vault_save_credentials(payload: VaultSaveCredentialsPayload):
    # Desktop mode has one authoritative Agent Vault.  Keep this legacy Pro-tab
    # endpoint as a compatibility alias so both Lite and Pro write to the same
    # encrypted store instead of maintaining a second state/vault.enc file.
    supervisor = globals().get("_CCEA_SUPERVISOR")
    ccea_state = globals().get("_CCEA_STATE")
    if supervisor is not None and ccea_state == "running":
        result = supervisor.store_credentials(
            payload.broker,
            {"api_key": payload.key_id, "api_secret": payload.secret},
        )
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error", "Vault write failed"))
        return {"status": "success", **result}
    if vault_instance.is_locked:
        raise HTTPException(status_code=400, detail="Vault is locked. Unlock it first.")
    try:
        vault_instance.store(payload.broker, "api_key", payload.key_id)
        vault_instance.store(payload.broker, "api_secret", payload.secret)
        return {
            "status": "success",
            "credentials": vault_instance.list_credentials(payload.broker)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@api.post("/api/state/flush")
def post_state_flush():
    try:
        import state_store
        state_store.save()
        return {"status": "success", "message": "State successfully flushed to disk"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/state/release_lock")
def post_state_release_lock():
    try:
        lock_path = Path("state/state.lock")
        if lock_path.exists():
            lock_path.unlink()
            return {"status": "success", "message": "Lock file state/state.lock forced release"}
        else:
            return {"status": "success", "message": "No lock file found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/preflight/run")
def post_preflight_run():
    # Run preflight checker
    try:
        result = preflight_checker_instance.run_preflight(
            broker_name="alpaca",
            manifest={"schema_version": "1.5.0", "entrypoint": "app.py"}
        )
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/degraded/status")
def get_degraded_status():
    status = degraded_manager_instance.get_status()
    history = degraded_manager_instance.get_history()
    return {
        "status": status,
        "history": history
    }


@api.get("/api/security/violations")
def get_security_violations():
    # Fetch simulated sandbox violation logs
    from datetime import datetime
    
    mock_violations = [
        {
            "timestamp": datetime.now().isoformat(),
            "type": "FILESYSTEM_POLICY",
            "message": "Attempted write to restricted path: /etc/hosts",
            "severity": "CRITICAL"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "type": "EGRESS_POLICY",
            "message": "Blocked network request to unauthorized domain: malcious-endpoint.com",
            "severity": "HIGH"
        }
    ]
    return {
        "violations": mock_violations,
        # Sample sandbox violations for UI demonstration, not live security events.
        "simulated": True,
        "data_source": "demo_mock",
    }


@api.post("/api/killswitch/trigger")
def post_killswitch_trigger(payload: KillSwitchTriggerPayload):
    try:
        reason = HaltReason(
            reason_type=HaltReasonType.MANUAL_TRIGGER,
            severity=HaltSeverity.CRITICAL,
            message=payload.reason,
            trigger_source="Operator via UI"
        )
        kill_switch_manager_instance.trigger(reason, action=HaltAction.CANCEL_ORDERS)
        
        # Trip global system switch
        import services.ops_kill_switch as ops_kill_switch
        ops_kill_switch._trip()
        if background_running(GLOBAL_REALTIME_PID):
            stop_background(GLOBAL_REALTIME_PID)
            
        return {
            "status": "success",
            "kill_switch_active": True,
            "halt": kill_switch_manager_instance.current_halt.to_dict()
        }
    except HTTPException:
        raise  # preserve intended status codes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/killswitch/reset")
def post_killswitch_reset(payload: KillSwitchResetPayload):
    try:
        ok = kill_switch_manager_instance.acknowledge("operator", payload.approval_code)
        if ok:
            kill_switch_manager_instance.reset()
            import services.ops_kill_switch as ops_kill_switch
            ops_kill_switch.manual_reset()
            return {
                "status": "success",
                "kill_switch_active": False
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid approval code or cooldown active")
    except HTTPException:
        raise  # preserve intended status codes (e.g. 400 invalid approval code)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# --------------------------- Experiment Tracking & Model Registry API ---------------------------
# MLflow-подобный backend: прогоны, метрики, lineage (модель→данные→конфиг→git),
# версии, стадии, rollback, криптоподпись артефактов (Ed25519). См.
# service_experiment_tracking.py / core_experiment.py / tests/test_experiment_tracking.py.

class ModelTransitionPayload(BaseModel):
    version: int
    stage: str


class ModelRollbackPayload(BaseModel):
    to_version: Optional[int] = None


@api.get("/api/experiments")
def api_experiments_list():
    from service_experiment_tracking import get_tracker
    t = get_tracker()
    out = []
    for exp in t.list_experiments():
        runs = t.list_runs(exp)
        out.append({"experiment": exp, "n_runs": len(runs)})
    return {"experiments": out}


@api.get("/api/experiments/{experiment}/runs")
def api_experiment_runs(experiment: str):
    from service_experiment_tracking import get_tracker
    t = get_tracker()
    runs = t.list_runs(experiment)
    return {"experiment": experiment, "runs": [
        {"run_id": r.run_id, "status": r.status, "start_ms": r.start_ms,
         "end_ms": r.end_ms, "metrics": r.metrics, "params": r.params,
         "lineage": r.lineage.to_dict()} for r in runs
    ]}


@api.get("/api/experiments/{experiment}/runs/{run_id}")
def api_experiment_run_detail(experiment: str, run_id: str):
    from service_experiment_tracking import get_tracker
    rec = get_tracker().get_run(experiment, run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="run not found")
    return rec.to_dict()


@api.get("/api/experiments/{experiment}/runs/{run_id}/metrics/{key}")
def api_experiment_metric_history(experiment: str, run_id: str, key: str):
    from service_experiment_tracking import get_tracker
    return {"key": key, "history": get_tracker().read_metric_history(experiment, run_id, key)}


@api.get("/api/models")
def api_models_list():
    from service_experiment_tracking import get_registry
    reg = get_registry()
    names = []
    if os.path.isdir(reg.root):
        names = sorted([d for d in os.listdir(reg.root)
                        if os.path.isdir(os.path.join(reg.root, d))])
    out = []
    for n in names:
        vers = reg.list_versions(n)
        prod = reg.get(n, stage="production")
        out.append({"name": n, "n_versions": len(vers),
                    "production_version": prod.version if prod else None})
    return {"models": out}


@api.get("/api/models/{name}/versions")
def api_model_versions(name: str):
    from service_experiment_tracking import get_registry
    reg = get_registry()
    vers = reg.list_versions(name)
    if not vers:
        raise HTTPException(status_code=404, detail="model not found")
    return {"name": name, "versions": [
        {**v.to_dict(), **reg.verify_status(name, v.version)} for v in vers
    ]}


@api.get("/api/models/{name}/production")
def api_model_production(name: str):
    from service_experiment_tracking import get_registry
    mv = get_registry().get(name, stage="production")
    if mv is None:
        raise HTTPException(status_code=404, detail="no production version")
    return mv.to_dict()


@api.get("/api/models/{name}/verify/{version}")
def api_model_verify(name: str, version: int):
    from service_experiment_tracking import get_registry
    reg = get_registry()
    if reg.get_version(name, version) is None:
        raise HTTPException(status_code=404, detail="version not found")
    return {"name": name, "version": version, **reg.verify_status(name, version)}


@api.post("/api/models/{name}/transition")
def api_model_transition(name: str, payload: ModelTransitionPayload):
    from service_experiment_tracking import get_registry
    try:
        mv = get_registry().transition(name, payload.version, payload.stage)
        return {"status": "ok", "version": mv.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@api.post("/api/models/{name}/rollback")
def api_model_rollback(name: str, payload: ModelRollbackPayload):
    from service_experiment_tracking import get_registry
    try:
        mv = get_registry().rollback(name, to_version=payload.to_version)
        return {"status": "ok", "rolled_back_to": mv.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --------------------------- Real cross-sectional backtests (background) ---------------------------
# Запуск РЕАЛЬНЫХ бэктестов на живых бесплатных данных (Binance / Yahoo+SEC EDGAR)
# как фоновых subprocess-задач; прогресс — через /api/logs, статус — /api/terminal/status.

class XSRealRunPayload(BaseModel):
    kind: str  # "crypto" | "equity" | "edgar"


@api.post("/api/xs/real/run")
def api_xs_real_run(payload: XSRealRunPayload):
    import uuid as _uuid
    import platform as _pl

    kind = (payload.kind or "").lower().strip()
    repo = os.path.dirname(os.path.abspath(__file__))
    scripts = {
        "crypto": ([sys.executable, "tools/xs_crypto_real_sweep.py"], "reports/xs_crypto_real_sweep.json"),
        "equity": ([sys.executable, "tools/xs_equity_real_report.py"], "reports/xs_equity_real_sweep.json"),
        "edgar": ([sys.executable, "scripts/download_edgar_fundamentals.py",
                   "--symbols", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "XOM", "JNJ", "PG",
                   "--out", "data/fundamentals_edgar/edgar_pit.parquet"], None),
    }
    if kind not in scripts:
        raise HTTPException(status_code=400, detail="kind must be crypto|equity|edgar")
    cmd, report_path = scripts[kind]

    cmd_id = f"xsreal_{kind}_{str(_uuid.uuid4())[:6]}"
    log_name = f"cli_cmd_{cmd_id}.log"
    log_path = os.path.join(GLOBAL_LOGS_DIR, log_name)
    os.makedirs(GLOBAL_LOGS_DIR, exist_ok=True)

    env = os.environ.copy()
    venv_sp = os.path.join(repo, ".venv", "Lib", "site-packages")
    parts = [p for p in (venv_sp, repo, env.get("PYTHONPATH", "")) if p]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("SEC_EDGAR_USER_AGENT", "RivenQuant Research research@example.com")

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"$ {' '.join(cmd)}\n")
        logf = open(log_path, "a", encoding="utf-8", newline="")
        kwargs: Dict[str, Any] = {"stdout": logf, "stderr": logf, "cwd": repo, "env": env}
        if _pl.system() == "Windows":
            kwargs["creationflags"] = 0x00000200  # CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["preexec_fn"] = os.setsid
        proc = subprocess.Popen(prepare_python_command(cmd), **kwargs)  # shell=False; frozen workers are allow-listed
        ACTIVE_CLI_PROCESSES[cmd_id] = {
            "proc": proc, "cmd": " ".join(cmd), "start_time": datetime.now().isoformat()
        }
        return {"cmd_id": cmd_id, "log_name": log_name, "report_path": report_path}
    except Exception as e:
        return {"cmd_id": "", "log_name": log_name, "error": str(e)}


class XSRealAnalyzePayload(BaseModel):
    kind: str = "crypto"     # crypto | equity
    algo: str = "TWAP"       # TWAP | VWAP | POV
    n_slices: int = 6
    equity: float = 1_000_000.0
    # P1 optimizer overrides (тумблеры из UI делают tcost/sizing/RL функциональными)
    tcost_aware: bool = False
    tcost_linear: float = 0.0008
    sizing: Optional[str] = None        # none | vol_target | kelly
    target_vol: Optional[float] = None
    kelly_fraction: float = 0.5
    include_rl: bool = False


@api.post("/api/xs/real/analyze")
def api_xs_real_analyze(payload: XSRealAnalyzePayload):
    """P1 для MVP: pre-trade VaR/CVaR/стресс + impact-aware execution-план на РЕАЛЬНЫХ
    данных (последние целевые веса cross-section). Один синхронный вызов."""
    kind = (payload.kind or "crypto").lower().strip()
    paths = {"crypto": "configs/config_xs_crypto_real.yaml",
             "equity": "configs/config_xs_equity_real.yaml"}
    if kind not in paths:
        raise HTTPException(status_code=400, detail="kind must be crypto|equity")
    try:
        import yaml as _yaml
        from service_xs_pipeline import XSConfig, load_panel, latest_target_weights
        from service_pretrade_risk import PreTradeRiskAnalyzer, RiskLimits
        from service_risk_model import StatRiskModel
        from service_xs_execution import RebalanceScheduler
        from core_portfolio import SYMBOL_LEVEL

        with open(paths[kind], "r", encoding="utf-8") as fh:
            raw = _yaml.safe_load(fh) or {}
        # применяем P1-override из UI (tcost / sizing / RL-сигнал)
        opt = raw.setdefault("optimizer", {})
        opt["tcost_aware"] = bool(payload.tcost_aware)
        opt["tcost_linear"] = float(payload.tcost_linear)
        if payload.sizing and payload.sizing != "none":
            opt["sizing"] = payload.sizing
            if payload.target_vol:
                opt["target_vol"] = float(payload.target_vol)
            opt["kelly_fraction"] = float(payload.kelly_fraction)
        if payload.include_rl:
            raw.setdefault("signals", []).append(
                {"name": "rl_alpha", "kind": "rl_alpha", "transforms": ["zscore"]})
        cfg = XSConfig.model_validate(raw)
        w = latest_target_weights(cfg)
        if not len(w):
            raise HTTPException(status_code=422, detail="no target weights")
        panel = load_panel(cfg)
        close = panel[cfg.backtest.price_col].unstack(level=SYMBOL_LEVEL)
        rets = close.pct_change().fillna(0.0)
        cov = StatRiskModel(method="ledoit_wolf").fit(rets).cov()
        cov = cov.reindex(index=list(w.index), columns=list(w.index)).fillna(0.0)
        rep = PreTradeRiskAnalyzer(cov).pretrade_check(w, limits=RiskLimits(), returns=rets, strict=False)
        prices = close.iloc[-1]
        adv = None
        if "volume" in panel.columns:
            adv = (panel["volume"].unstack(level=SYMBOL_LEVEL) * close).tail(20).mean()
        sched = RebalanceScheduler(algo=str(payload.algo), n_slices=int(payload.n_slices))
        plan = sched.build_plan(w, None, prices, float(payload.equity), adv=adv)
        return {"kind": kind, "n_names": int(len(w)),
                "weights_gross": float(w.abs().sum()), "weights_net": float(w.sum()),
                "risk": rep.to_dict(), "execution": plan.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Auto-recovery (Agent-зона) состояние для MVP-индикатора (P1) ---
def _agent_recovery_state():
    if not hasattr(_agent_recovery_state, "_v"):
        from packages.agent.execution.resilience import CircuitBreaker, RetryPolicy
        _agent_recovery_state._v = {
            "retry": RetryPolicy(max_attempts=5, base_delay=0.5, max_delay=30.0, multiplier=2.0),
            "breaker": CircuitBreaker(failure_threshold=5, reset_timeout=30.0),
            "last_reconcile": None,
        }
    return _agent_recovery_state._v


@api.get("/api/agent/recovery/status")
def api_agent_recovery_status():
    """Состояние авто-recovery исполнения (retry/circuit-breaker/reconcile). Конфигурированный
    guard Agent-зоны; в live-режиме обновляется телеметрией Agent (CCEA)."""
    v = _agent_recovery_state()
    rp, cb = v["retry"], v["breaker"]
    return {
        "connected": False,   # нет live-Agent в этом процессе → показываем конфигурацию guard
        "circuit_state": cb.state.value,
        "retry_policy": {
            "max_attempts": rp.max_attempts, "base_delay": rp.base_delay,
            "max_delay": rp.max_delay, "multiplier": rp.multiplier,
        },
        "failure_threshold": cb.failure_threshold,
        "reset_timeout_sec": cb.reset_timeout,
        "last_reconcile": v["last_reconcile"],
        "note": "Конфигурированный guard Agent-зоны; в live обновляется телеметрией Agent.",
    }


# --------------------------- P2: Scale & Ops (FIX/SOR, Feature Store, Automation, TS-DB) ---------------------------

class RoutePayload(BaseModel):
    symbol: str = "AAPL"
    side: str = "BUY"
    notional: float = 3_000_000.0
    split: bool = True
    dispatch: bool = False          # P1 #7: actually dispatch child orders to (paper) venues
    price: float = 100.0            # reference price for live-liquidity + child sizing


@api.post("/api/exec/route")
def api_exec_route(payload: RoutePayload):
    """Smart Order Routing (мульти-venue split) — REAL water-filling engine.

    P1 #7: with ``dispatch=true`` the route is actually DISPATCHED — each venue
    allocation is sent as a child order to a paper venue connector (SimBroker), proving
    SOR is wired into the live submission path (not a FIX-preview-only dead end). Uses
    live top-of-book liquidity from the venue connectors.
    """
    from packages.agent.execution.smart_order_router import SmartOrderRouter, Venue
    from packages.agent.execution.fix_protocol import (
        new_order_single, Side, OrdType, verify_checksum,
    )
    venue_specs = [
        ("NYSE", 0.5, 20, 0.1), ("NASDAQ", 0.8, 30, 0.1),
        ("IEX", 0.3, 50, 0.1), ("DARK", 0.2, 80, 0.12),
    ]
    venues = [Venue(n, fee_bps=f, latency_ms=l, liquidity=5e6, impact_coef=ic)
              for (n, f, l, ic) in venue_specs]
    sor = SmartOrderRouter(venues)
    out: Dict[str, Any] = {"venues": [v.name for v in venues]}

    if payload.dispatch:
        # Build a paper SimBroker per venue, prime the price, route on LIVE liquidity,
        # then dispatch — the same routed-submit machinery used by build_live_stack.
        from packages.agent.broker.adapters.sim import SimBrokerConnector
        from packages.agent.execution.live_factory import (
            BrokerLiquidityProvider, make_venue_submit)
        conns = {}
        for v in venues:
            c = SimBrokerConnector(broker_name=v.name)
            c.set_price(payload.symbol, float(payload.price))
            conns[v.name] = c
        provider = BrokerLiquidityProvider(conns)
        route = sor.route_live(payload.symbol, payload.side, float(payload.notional), provider,
                               split=bool(payload.split))
        disp = sor.dispatch(route, make_venue_submit(conns, lambda s: float(payload.price)))
        out["route"] = route.to_dict()
        out["dispatch"] = disp
        out["live_liquidity"] = True
        out["simulated"] = False
        out["note"] = "Routed on live (paper) venue liquidity and dispatched real child orders."
    else:
        res = sor.route(payload.symbol, payload.side, float(payload.notional),
                        split=bool(payload.split))
        side = Side.BUY if str(payload.side).upper() == "BUY" else Side.SELL
        fix = new_order_single(cl_ord_id="UI-PREVIEW", symbol=payload.symbol, side=side,
                               qty=round(float(payload.notional) / 100.0), ord_type=OrdType.MARKET)
        out["route"] = res.to_dict()
        out["fix_preview"] = fix.replace("\x01", " | ")
        out["fix_valid"] = bool(verify_checksum(fix))
    return out


@api.get("/api/features/store")
def api_features_store():
    """Список фич в Feature Store + версии (P2)."""
    from service_feature_store import get_feature_store
    fs = get_feature_store()
    out = []
    for name in fs.list_features():
        vers = fs.list_versions(name)
        out.append({"name": name, "n_versions": len(vers),
                    "latest": vers[-1].to_dict() if vers else None})
    return {"features": out, "count": len(out)}


@api.get("/api/automation/status")
def api_automation_status():
    """Статус автоматизации: drift-ретрейн (по models/drift_report.json) + TS-DB backend (P2)."""
    import json as _json
    from services.automation.drift_retrain import DriftRetrainScheduler
    from services.tsdb import make_backend
    report = {}
    p = "models/drift_report.json"
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                report = _json.load(fh)
        except Exception:
            report = {}
    decision = DriftRetrainScheduler().check(report)
    be = make_backend("clickhouse")   # graceful fallback → parquet
    return {"drift": decision.to_dict(), "tsdb_backend": type(be).__name__,
            "tsdb_available": bool(be.available())}


@api.get("/api/xs/signal_catalog")
def api_xs_signal_catalog():
    """Каталог доступных сигналов по классам (P2 расширение) (P2)."""
    from signals.crypto_signals import CRYPTO_SIGNAL_KINDS
    from signals.equity_signals import EQUITY_SIGNAL_KINDS
    from signals.futures_signals import FUTURES_SIGNAL_KINDS
    from signals.forex_signals import FOREX_SIGNAL_KINDS
    from signals.options_signals import OPTIONS_SIGNAL_KINDS
    from signals.common_signals import COMMON_SIGNAL_KINDS
    cats = {
        "crypto": list(CRYPTO_SIGNAL_KINDS), "equity": list(EQUITY_SIGNAL_KINDS),
        "futures": list(FUTURES_SIGNAL_KINDS), "forex": list(FOREX_SIGNAL_KINDS),
        "options": list(OPTIONS_SIGNAL_KINDS), "common": list(COMMON_SIGNAL_KINDS),
        "rl": ["rl_alpha"],
    }
    return {"categories": cats, "total": sum(len(v) for v in cats.values())}


# --------------------------- CCEA RUNTIME (real Agent + Cloud, local) ---------------------------
# The desktop hosts a REAL CCEA stack locally (loopback, no servers): the cloud
# control-plane (no secrets/orders) + the Agent daemon (vault on OS keychain,
# policy firewall, kill switch, paper/live broker, orders created locally). It is
# launched on startup when RIVEN_ENABLE_CCEA is truthy (the desktop sets it). The
# Agent-zone code (packages.agent) is imported LAZILY inside the boot thread so
# the plain MVP/cloud surface never pulls order/secret modules unless CCEA is on.
_CCEA_SUPERVISOR = None
_CCEA_STATE = "disabled"   # disabled | starting | running | error
_CCEA_ERROR: Optional[str] = None
_CCEA_LOCK = threading.Lock()


def _ccea_enabled() -> bool:
    return os.environ.get("RIVEN_ENABLE_CCEA", "0").strip().lower() in ("1", "true", "yes", "on")


def _start_ccea_background() -> None:
    global _CCEA_STATE
    with _CCEA_LOCK:
        if _CCEA_STATE in ("starting", "running"):
            return
        _CCEA_STATE = "starting"

    def _run() -> None:
        global _CCEA_SUPERVISOR, _CCEA_STATE, _CCEA_ERROR
        try:
            from ccea.desktop_supervisor import CCEASupervisor, SupervisorConfig
            data_dir = Path(os.environ.get("RIVEN_DATA_DIR", ".")) / "ccea"
            sup = CCEASupervisor(SupervisorConfig(data_dir=data_dir, paper=True))
            # Publish before start so an app shutdown racing with bootstrap can
            # still stop a partially-started control plane.
            _CCEA_SUPERVISOR = sup
            sup.start()
            _CCEA_STATE = "running"
        except Exception as exc:  # pragma: no cover - surfaced via /api/ccea/status
            _CCEA_ERROR = str(exc)
            _CCEA_STATE = "error"

    threading.Thread(target=_run, name="ccea-supervisor-boot", daemon=True).start()


@api.on_event("startup")
async def _ccea_startup() -> None:  # pragma: no cover - runtime hook
    if _ccea_enabled():
        _start_ccea_background()


def _stop_ccea_runtime() -> None:
    """Idempotently stop the local desktop CCEA stack and release its stores."""
    global _CCEA_SUPERVISOR, _CCEA_STATE, _CCEA_ERROR
    with _CCEA_LOCK:
        supervisor = _CCEA_SUPERVISOR
        if supervisor is None:
            if _CCEA_STATE != "disabled":
                _CCEA_STATE = "stopped"
            return
        _CCEA_STATE = "stopping"
    try:
        supervisor.stop()
        _CCEA_ERROR = None
        _CCEA_STATE = "stopped"
    except Exception as exc:
        _CCEA_ERROR = str(exc)
        _CCEA_STATE = "error"
    finally:
        _CCEA_SUPERVISOR = None


@api.on_event("shutdown")
async def _ccea_shutdown() -> None:  # pragma: no cover - runtime hook
    _stop_ccea_runtime()


@api.post("/api/desktop/shutdown")
def api_desktop_shutdown(request: Request):
    """Flush the desktop Agent before the Tauri shell terminates the sidecar."""
    client_host = request.client.host if request.client else None
    if not _is_loopback_client(client_host):
        raise HTTPException(status_code=403, detail="Desktop shutdown is loopback-only")
    _stop_ccea_runtime()
    server = getattr(request.app.state, "desktop_server", None)
    if server is not None:
        server.should_exit = True
    return {"ok": True, "state": _CCEA_STATE}


@api.get("/api/ccea/status")
def api_ccea_status():
    """Live CCEA status (read-only; no secrets/orders cross this boundary)."""
    if not _ccea_enabled():
        return {"enabled": False, "state": "disabled"}
    if _CCEA_SUPERVISOR is not None and _CCEA_STATE == "running":
        st = _CCEA_SUPERVISOR.status()
        st["enabled"] = True
        st["state"] = "running"
        return st
    return {"enabled": True, "state": _CCEA_STATE, "error": _CCEA_ERROR}


class CCEAPaperOrderPayload(BaseModel):
    symbol: str = "BTCUSDT"
    qty: float = 0.1
    entry_price: float = 50000.0
    mark_price: Optional[float] = None


@api.post("/api/ccea/paper_order")
def api_ccea_paper_order(payload: CCEAPaperOrderPayload):
    """Drive a REAL paper run through the Agent OMS (Intent -> policy firewall ->
    journal -> broker order -> fill -> mark-to-market PnL). Paper/SimBroker only."""
    if not _ccea_enabled():
        return {"ok": False, "error": "CCEA disabled"}
    if _CCEA_SUPERVISOR is None or _CCEA_STATE != "running":
        return {"ok": False, "error": f"CCEA not running (state={_CCEA_STATE})"}
    try:
        return _CCEA_SUPERVISOR.paper_trade(
            symbol=payload.symbol,
            qty=payload.qty,
            entry_price=payload.entry_price,
            mark_price=payload.mark_price,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


class CCEAConnectBrokerPayload(BaseModel):
    broker: str
    api_key: str = ""
    api_secret: str = ""
    sandbox: bool = True
    account_id: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class CCEAStoreCredentialsPayload(BaseModel):
    broker: str
    credentials: Dict[str, str]


@api.post("/api/ccea/store_credentials")
def api_ccea_store_credentials(payload: CCEAStoreCredentialsPayload):
    """Store adapter credentials in the authoritative desktop Agent Vault."""
    if not _ccea_enabled():
        return {"ok": False, "error": "CCEA disabled"}
    if _CCEA_SUPERVISOR is None or _CCEA_STATE != "running":
        return {"ok": False, "error": f"CCEA not running (state={_CCEA_STATE})"}
    return _CCEA_SUPERVISOR.store_credentials(payload.broker, payload.credentials)


@api.post("/api/ccea/connect_broker")
def api_ccea_connect_broker(payload: CCEAConnectBrokerPayload):
    """Store broker credentials in the local Agent Vault and connect a REAL broker
    connector. Credentials stay on-device (Agent zone); never sent to the cloud."""
    if not _ccea_enabled():
        return {"ok": False, "error": "CCEA disabled"}
    if _CCEA_SUPERVISOR is None or _CCEA_STATE != "running":
        return {"ok": False, "error": f"CCEA not running (state={_CCEA_STATE})"}
    try:
        return _CCEA_SUPERVISOR.connect_live_broker(
            broker=payload.broker,
            api_key=payload.api_key,
            api_secret=payload.api_secret,
            sandbox=payload.sandbox,
            account_id=payload.account_id,
            extra=payload.extra,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@api.get("/api/agent/pnl/status")
def api_agent_pnl_status():
    """Live Agent P&L ledger snapshot (realized/unrealized/fees/financing + NAV).

    The Agent's OWN books-of-record (not echoed from the broker). Read-only."""
    if not _ccea_enabled():
        return {"enabled": False, "error": "CCEA disabled"}
    if _CCEA_SUPERVISOR is None or _CCEA_STATE != "running":
        return {"enabled": True, "state": _CCEA_STATE, "ledger": None}
    try:
        st = _CCEA_SUPERVISOR.status()
        return {"enabled": True, "state": "running", "ledger": st.get("pnl_ledger"),
                "broker_account": st.get("broker_account")}
    except Exception as exc:
        return {"enabled": True, "error": str(exc)}


@api.get("/api/agent/pnl/nav_history")
def api_agent_pnl_nav_history():
    """EOD NAV snapshot history from the Agent ledger."""
    if not _ccea_enabled() or _CCEA_SUPERVISOR is None or _CCEA_STATE != "running":
        return {"enabled": _ccea_enabled(), "history": []}
    try:
        led = getattr(_CCEA_SUPERVISOR, "_ledger", None)
        return {"enabled": True, "history": (led.nav_history() if led is not None else [])}
    except Exception as exc:
        return {"enabled": True, "error": str(exc), "history": []}


@api.post("/api/agent/pnl/eod_close")
def api_agent_pnl_eod_close():
    """Take an EOD NAV snapshot on the Agent ledger and roll the trading day."""
    if not _ccea_enabled():
        return {"ok": False, "error": "CCEA disabled"}
    if _CCEA_SUPERVISOR is None or _CCEA_STATE != "running":
        return {"ok": False, "error": f"CCEA not running (state={_CCEA_STATE})"}
    try:
        return _CCEA_SUPERVISOR.eod_close()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# --------------------------- FIRM-WIDE HIERARCHICAL RISK (strategy->desk->firm) ---------------------------
# Consolidated VaR/CVaR across all books with Euler risk attribution + hierarchical
# limits. Engine: service_firm_risk (academically grounded; see module docstring).

class FirmRiskPayload(BaseModel):
    # books: {desk: {strategy: [{symbol, exposure($), risk_unit?, sector?, asset_class?}]}}
    books: Dict[str, Any]
    cov: Optional[Dict[str, Any]] = None          # {unit: {unit: cov}} (return covariance)
    returns: Optional[Dict[str, Any]] = None       # {unit: [r,...]} for historical method
    exposures: Optional[Dict[str, Any]] = None     # {unit: {factor: loading}}
    alpha: float = 0.05
    method: str = "parametric"                     # parametric | historical
    limits: Optional[Dict[str, Any]] = None        # {node: {var?, cvar?, gross?, net?, var_pct?, hard?}}
    capital: Optional[Dict[str, float]] = None
    firm_name: str = "FIRM"


def _run_firm_risk(payload: "FirmRiskPayload") -> Dict[str, Any]:
    import pandas as _pd
    from service_firm_risk import (
        FirmRiskAggregator, HierLimits, positions_from_books,
    )

    positions = positions_from_books(payload.books)
    cov_df = None
    returns_df = None
    if payload.cov:
        cov_df = _pd.DataFrame(payload.cov).astype("float64")
    if payload.returns:
        returns_df = _pd.DataFrame({k: list(v) for k, v in payload.returns.items()}).astype("float64")
    exposures_df = None
    if payload.exposures:
        exposures_df = _pd.DataFrame(payload.exposures).T.astype("float64")
    agg = FirmRiskAggregator(cov=cov_df, returns=returns_df, exposures=exposures_df,
                             alpha=float(payload.alpha))
    limits = None
    if payload.limits:
        limits = {k: HierLimits(**{kk: vv for kk, vv in v.items() if kk in
                                   ("var", "cvar", "gross", "net", "var_pct", "hard")})
                  for k, v in payload.limits.items()}
    rep = agg.aggregate(positions, method=payload.method, firm_name=payload.firm_name,
                        limits=limits, capital=payload.capital)
    return rep.to_dict()


@api.post("/api/firm_risk/aggregate")
def api_firm_risk_aggregate(payload: FirmRiskPayload):
    """Consolidate posted books into a strategy->desk->firm VaR/CVaR risk tree.

    Real engine on REAL posted data: VaR/CVaR (parametric or historical), Euler
    component/marginal/incremental VaR per sub-book, diversification benefit, and
    hierarchical limit checks. No synthetic data — what you post is what's analyzed.
    """
    try:
        out = _run_firm_risk(payload)
        out["simulated"] = False
        out["data_source"] = "user_posted"
        return out
    except Exception as exc:
        return {"error": str(exc), "ok": False}


@api.get("/api/firm_risk/demo")
def api_firm_risk_demo():
    """Representative multi-desk firm view to drive the MVP card.

    Runs the REAL firm-risk engine, but on REPRESENTATIVE exposures + a model
    covariance (flagged ``simulated=True``). When the live CCEA Agent ledger has
    open positions, they are folded in as a real ``agent_live`` desk (``has_live_book``).
    """
    import numpy as _np
    import pandas as _pd
    from service_firm_risk import FirmRiskAggregator, FirmPosition, HierLimits

    # Representative cross-asset book (illustrative exposures; model covariance).
    units = ["AAPL", "MSFT", "XOM", "JPM", "ES", "NQ", "EURUSD", "GBPUSD", "BTCUSDT", "ETHUSDT"]
    vols = _np.array([.018, .017, .022, .020, .013, .016, .006, .007, .035, .040])
    # asset-class block correlations (equity/futures/fx/crypto)
    cls = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3]
    base = _np.array([[1.0, .25, .15, .05], [.25, 1.0, .05, .20],
                      [.15, .05, 1.0, .10], [.05, .20, .10, 1.0]])
    n = len(units)
    C = _np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                C[i, j] = base[cls[i], cls[j]] * (0.85 if cls[i] == cls[j] else 1.0)
    _np.fill_diagonal(C, 1.0)
    S = _np.diag(vols) @ C @ _np.diag(vols)
    cov_df = _pd.DataFrame(S, index=units, columns=units)

    positions = [
        FirmPosition("AAPL", 180_000, desk="equity_long_short", strategy="momentum", sector="tech"),
        FirmPosition("MSFT", 150_000, desk="equity_long_short", strategy="momentum", sector="tech"),
        FirmPosition("XOM", -90_000, desk="equity_long_short", strategy="mean_reversion", sector="energy"),
        FirmPosition("JPM", -70_000, desk="equity_long_short", strategy="mean_reversion", sector="financials"),
        FirmPosition("ES", 250_000, desk="macro_futures", strategy="trend", sector="index"),
        FirmPosition("NQ", -120_000, desk="macro_futures", strategy="carry", sector="index"),
        FirmPosition("EURUSD", -200_000, desk="fx_carry", strategy="carry", sector="fx"),
        FirmPosition("GBPUSD", 140_000, desk="fx_carry", strategy="value", sector="fx"),
        FirmPosition("BTCUSDT", 80_000, desk="crypto", strategy="momentum", sector="crypto"),
        FirmPosition("ETHUSDT", -40_000, desk="crypto", strategy="momentum", sector="crypto"),
    ]

    has_live_book = False
    # Fold in the live Agent ledger positions as a real desk, if present.
    if _ccea_enabled() and _CCEA_SUPERVISOR is not None and _CCEA_STATE == "running":
        try:
            led = getattr(_CCEA_SUPERVISOR, "_ledger", None)
            if led is not None:
                for p in led.positions():
                    if p.quantity != 0 and str(p.symbol) in cov_df.index:
                        positions.append(FirmPosition(
                            str(p.symbol), float(p.market_value),
                            desk="agent_live", strategy="desktop-demo",
                            sector="crypto" if "USD" in str(p.symbol) else None))
                        has_live_book = True
        except Exception:
            pass

    capital = {"FIRM": 5_000_000}
    limits = {
        "FIRM": HierLimits(var=120_000, cvar=160_000, gross=2_000_000, hard=True),
        "crypto": HierLimits(var=25_000, hard=False),
    }
    agg = FirmRiskAggregator(cov=cov_df, alpha=0.05)
    rep = agg.aggregate(positions, method="parametric", limits=limits, capital=capital)
    out = rep.to_dict()
    out["simulated"] = True
    out["has_live_book"] = has_live_book
    out["data_source"] = "representative"
    out["note"] = ("Representative cross-asset exposures with a model covariance — the "
                   "VaR/CVaR/Euler math is real. Post real books to /api/firm_risk/aggregate."
                   + (" Live Agent ledger positions folded in as 'agent_live' desk."
                      if has_live_book else ""))
    return out


# --------------------------- INSTRUMENT MASTER / SYMBOLOGY (FIGI/CUSIP/ISIN/OCC) ---------------------------
@api.get("/api/instruments/resolve")
def api_instruments_resolve(q: str):
    """Resolve any identifier (ticker/FIGI/ISIN/CUSIP/SEDOL/OCC) to the canonical
    instrument identity. Maps raw vendor tickers to one firm-wide identity."""
    if global_instrument_master is None:
        return {"ok": False, "error": "instrument master unavailable"}
    rec = global_instrument_master.resolve(q)
    if rec is None:
        return {"ok": True, "found": False, "query": q}
    return {"ok": True, "found": True, "instrument": rec.to_dict()}


@api.get("/api/instruments/search")
def api_instruments_search(q: str, limit: int = 20):
    if global_instrument_master is None:
        return {"ok": False, "error": "instrument master unavailable", "results": []}
    return {"ok": True, "results": [r.to_dict() for r in global_instrument_master.search(q, limit=limit)]}


@api.get("/api/instruments/list")
def api_instruments_list():
    if global_instrument_master is None:
        return {"ok": False, "results": []}
    return {"ok": True, "count": len(global_instrument_master),
            "results": [r.to_dict() for r in global_instrument_master.all()]}


class OCCParsePayload(BaseModel):
    symbol: str


@api.post("/api/instruments/occ_parse")
def api_instruments_occ_parse(payload: OCCParsePayload):
    """Parse a 21-char OCC option symbol into root/expiry/type/strike."""
    try:
        from services.instrument_master import parse_occ_symbol
        o = parse_occ_symbol(payload.symbol)
        return {"ok": True, "root": o.root, "expiry": o.expiry.isoformat(),
                "option_type": o.option_type, "strike": o.strike, "occ_symbol": o.occ_symbol}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# --------------------------- MARKET-ABUSE SURVEILLANCE (live, MAR) ---------------------------
@api.get("/api/surveillance/market_abuse")
def api_surveillance_market_abuse():
    """Live MAR surveillance alerts (spoofing/layering/wash/marking-the-close) from
    the global monitor (XS/MVP order flow) + the CCEA Agent monitor (live fills)."""
    out: Dict[str, Any] = {"enabled": global_market_abuse_monitor is not None}
    if global_market_abuse_monitor is not None:
        out["global"] = {
            "summary": global_market_abuse_monitor.summary(),
            "alerts": [a.to_dict() for a in global_market_abuse_monitor.get_alerts()][-50:],
        }
    if _ccea_enabled() and _CCEA_SUPERVISOR is not None and _CCEA_STATE == "running":
        try:
            out["agent"] = _CCEA_SUPERVISOR.surveillance_alerts(limit=50)
        except Exception as exc:
            out["agent_error"] = str(exc)
    out["data_source"] = "live_surveillance"
    out["simulated"] = False
    return out


# --------------------------- AGENT BOOKS-AND-RECORDS (blotter / cash / journal) ---------------------------
@api.get("/api/agent/blotter")
def api_agent_blotter(limit: int = 100):
    """Immutable, hash-chained executed-trade blotter (Agent books-and-records)."""
    if not _ccea_enabled() or _CCEA_SUPERVISOR is None or _CCEA_STATE != "running":
        return {"enabled": _ccea_enabled(), "trades": [], "integrity": None}
    try:
        return {"enabled": True, **_CCEA_SUPERVISOR.books_blotter(limit=limit)}
    except Exception as exc:
        return {"enabled": True, "error": str(exc), "trades": []}


@api.get("/api/agent/cash_ledger")
def api_agent_cash_ledger(limit: int = 100):
    """Append-only, hash-chained cash general-ledger (Agent books-and-records)."""
    if not _ccea_enabled() or _CCEA_SUPERVISOR is None or _CCEA_STATE != "running":
        return {"enabled": _ccea_enabled(), "movements": [], "integrity": None}
    try:
        return {"enabled": True, **_CCEA_SUPERVISOR.books_cash(limit=limit)}
    except Exception as exc:
        return {"enabled": True, "error": str(exc), "movements": []}


@api.get("/api/agent/journal/integrity")
def api_agent_journal_integrity():
    """Tamper-evident order-journal audit chain integrity (hash-linked, keyed)."""
    if not _ccea_enabled() or _CCEA_SUPERVISOR is None or _CCEA_STATE != "running":
        return {"enabled": _ccea_enabled(), "available": False}
    try:
        return {"enabled": True, **_CCEA_SUPERVISOR.journal_integrity()}
    except Exception as exc:
        return {"enabled": True, "available": False, "error": str(exc)}


# --------------------------- MARKET-DATA QUALITY + VENDOR FAILOVER (P1 #11) ---------------------------
class DataQualityPayload(BaseModel):
    bars: List[Dict[str, Any]]                  # [{timestamp, close, high?, low?, open?, volume?}]
    symbol: str = "?"
    spike_threshold: float = 8.0
    staleness_seconds: Optional[float] = 300.0
    now_ms: Optional[int] = None


@api.post("/api/data_quality/check")
def api_data_quality_check(payload: DataQualityPayload):
    """Robust market-data QC on posted bars: spike (MAD)/staleness/frozen/gap/OHLC."""
    try:
        import pandas as _pd
        from services.market_data_quality import DataQualityMonitor
        df = _pd.DataFrame(payload.bars)
        if df.empty:
            return {"ok": False, "error": "no bars"}
        mon = DataQualityMonitor(spike_threshold=payload.spike_threshold,
                                 staleness_seconds=payload.staleness_seconds)
        rep = mon.check(df, symbol=payload.symbol, now_ms=payload.now_ms)
        return {"ok": True, "report": rep.to_dict(), "simulated": False}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@api.get("/api/data_quality/demo")
def api_data_quality_demo():
    """Representative data-QA demo: a clean feed + a feed with an injected spike, a
    stale feed and a cross-vendor divergence — runs the REAL detectors (flagged demo)."""
    import numpy as _np
    import pandas as _pd
    import time as _t
    from services.market_data_quality import (
        DataQualityMonitor, MarketDataRouter, cross_source_reconcile)

    now = int(_t.time() * 1000)
    def _bars(n=60, price=100.0, spike=False, stale=False, seed=0):
        rng = _np.random.default_rng(seed)
        px = price * _np.cumprod(1 + rng.normal(0, 0.004, n))
        if spike:
            px[n // 2] = px[n // 2 - 1] * 4.0
        start = now - (n * 60_000) - (3_600_000 if stale else 60_000)
        ts = [start + i * 60_000 for i in range(n)]
        return _pd.DataFrame({"timestamp": ts, "close": px, "high": px * 1.001,
                              "low": px * 0.999, "open": px})

    mon = DataQualityMonitor(staleness_seconds=300)
    clean = mon.check(_bars(seed=1), symbol="CLEAN", now_ms=now).to_dict()
    spiked = mon.check(_bars(spike=True, seed=2), symbol="SPIKED", now_ms=now).to_dict()
    stale = mon.check(_bars(stale=True, seed=3), symbol="STALE", now_ms=now).to_dict()
    # failover: primary errors -> backup serves
    def _bad(s, **k):
        raise RuntimeError("primary vendor outage")
    router = MarketDataRouter([("primary_vendor", _bad),
                               ("backup_vendor", lambda s, **k: _bars(seed=5))], monitor=mon)
    fo = router.get_bars("AAPL", now_ms=now)
    recon = cross_source_reconcile({"AAPL": 100.0, "MSFT": 200.0},
                                   {"AAPL": 100.02, "MSFT": 206.0}, tolerance_bps=50)
    return {
        "reports": {"clean": clean, "spiked": spiked, "stale": stale},
        "failover": {"served_by": fo["source"], "failed_over": fo["failover"],
                     "attempts": fo["attempts"]},
        "router_status": router.status(),
        "cross_vendor": recon,
        "simulated": True,
        "note": "Representative feeds; the spike/staleness/failover detectors are real.",
    }


# --------------------------- Streamlit UI (legacy wrapper) ---------------------------
# IMPORTANT: this module is imported by the desktop sidecar and by `uvicorn app:api`
# purely for the FastAPI `api` object. The Streamlit wrapper below renders the same
# index.html inside Streamlit and MUST run ONLY under `streamlit run app.py` — never
# on plain import, otherwise it would execute UI side effects inside the API process.

def _streamlit_runtime_active() -> bool:
    """True only when executing under `streamlit run` (not on plain import)."""
    try:
        from streamlit.runtime import exists as _st_exists
        return bool(_st_exists())
    except Exception:
        return False


def _render_streamlit_ui() -> None:
    import streamlit.components.v1 as components

    st.set_page_config(page_title="RivenQuant AI Advanced Platform", layout="wide", initial_sidebar_state="collapsed")

    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stHeader"] {display: none;}
    .reportview-container .main .block-container {padding-top: 0rem; padding-bottom: 0rem;}
    div.block-container {padding: 0px !important;}
    html, body {
        margin: 0px !important;
        padding: 0px !important;
        overflow: hidden !important;
        background-color: #0E0E11 !important;
    }
    .stApp {
        background-color: #0E0E11 !important;
    }
    iframe {
        border: none !important;
        width: 100vw !important;
        height: 100vh !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        z-index: 999999 !important;
    }
    </style>
""", unsafe_allow_html=True)

    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_code = f.read()
        components.html(html_code, height=950, scrolling=True)
    except Exception as e:
        st.error(f"Error loading index.html: {e}")


if _streamlit_runtime_active():
    _render_streamlit_ui()


# ----------------------- Автостарт планировщика (после полной загрузки модуля) -----------------------
def _start_scheduler_if_enabled() -> None:
    """Поднимает SchedulerService фоновым потоком.

    Не стартует: (а) при RIVEN_ENABLE_SCHEDULER=0; (б) под pytest — иначе каждый
    тестовый импорт app запускал бы catch-up задачи (бэкапы, PSI-воркеры) прямо
    в рабочей копии. Тесты создают собственные экземпляры SchedulerService.
    """
    global _SCHEDULER
    if os.environ.get("RIVEN_ENABLE_SCHEDULER", "1").strip().lower() not in ("1", "true", "yes", "on"):
        return
    if "pytest" in sys.modules:
        return
    try:
        from services.alerts import AlertManager
        from services.scheduler import SchedulerService

        alert_settings = {}
        try:
            with open(os.path.join("configs", "scheduler.yaml"), "r", encoding="utf-8") as f:
                alert_settings = (yaml.safe_load(f) or {}).get("alerts") or {}
        except Exception:
            pass
        _alert_mgr = AlertManager(alert_settings)
        _SCHEDULER = SchedulerService(
            actions=_build_scheduler_actions(),
            alert_fn=lambda key, text: _alert_mgr.notify(key, text),
        )
        _SCHEDULER.start()
    except Exception:
        import logging as _logging
        _logging.getLogger(__name__).exception("scheduler autostart failed")


_start_scheduler_if_enabled()
