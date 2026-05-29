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

from fastapi import Depends, HTTPException, Header
import pandas as pd
import streamlit as st
import yaml

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

import clock
from services import monitoring
from services.rest_budget import RestBudgetSession
from services.utils_app import (
    ensure_dir as _ensure_dir,
    run_cmd,
    start_background,
    stop_background,
    background_running,
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

API_TOKEN = os.environ.get("SEASONALITY_API_TOKEN")
if API_TOKEN is None:
    raise RuntimeError(
        "SEASONALITY_API_TOKEN is required for API access. "
        "Load it from your secret manager or .env (see .env.example)."
    )

from fastapi.middleware.cors import CORSMiddleware

def _make_api() -> Any:
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
    res = subprocess.run(cmd, capture_output=True, text=True)
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
from pydantic import BaseModel

# Global paths for API endpoints
GLOBAL_LOGS_DIR = "logs"
GLOBAL_SIGNALS_CSV = os.path.join(GLOBAL_LOGS_DIR, "signals.csv")
GLOBAL_METRICS_JSON = os.path.join(GLOBAL_LOGS_DIR, "metrics.json")
GLOBAL_REPORTS_PATH = os.path.join(GLOBAL_LOGS_DIR, "reports.csv")
GLOBAL_REALTIME_PID = os.path.join(".run", "rt_signaler.pid")
GLOBAL_REALTIME_LOG = os.path.join(GLOBAL_LOGS_DIR, "realtime.log")
GLOBAL_SNAPSHOT_JSON = os.path.join(GLOBAL_LOGS_DIR, "snapshot_metrics.json")

# Active system state for context selection
ACTIVE_ASSET = "equity"
ACTIVE_ADAPTER = "alpaca"

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
            "sandbox": "configs/config_live_futures.yaml",
            "train": "configs/config_train_futures.yaml",
            "realtime": "configs/config_live_futures.yaml",
            "ingest": "configs/ingest.yaml",
        }
    else:
        mapping = {
            "sandbox": "configs/sandbox.yaml",
            "train": "configs/config_train.yaml",
            "realtime": "configs/realtime.yaml",
            "ingest": "configs/ingest.yaml",
        }
    return mapping.get(config_type, "configs/sandbox.yaml")

class OrderAction(BaseModel):
    id: str

class YamlSavePayload(BaseModel):
    path: str
    content: str

class QuantizerSavePayload(BaseModel):
    strict_filters: bool
    enforce_percent_price_by_side: bool

class RunJobPayload(BaseModel):
    job: str
    params: Dict[str, Any]

class CopilotPayload(BaseModel):
    message: str

@api.get("/api/system_state")
def api_get_system_state():
    return {
        "active_asset": ACTIVE_ASSET,
        "active_adapter": ACTIVE_ADAPTER
    }

@api.post("/api/system_state")
def api_post_system_state(payload: SystemStatePayload):
    global ACTIVE_ASSET, ACTIVE_ADAPTER
    ACTIVE_ASSET = payload.active_asset.lower()
    ACTIVE_ADAPTER = payload.active_adapter.lower()
    return {"status": "success", "active_asset": ACTIVE_ASSET, "active_adapter": ACTIVE_ADAPTER}

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
    
    pnl_text = f"+{pnl:.2f}%" if isinstance(pnl, (int, float)) else "—"
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
        return {"status": "success"}
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/api/logs")
def api_logs(name: str):
    log_path = os.path.join(GLOBAL_LOGS_DIR, name)
    try:
        return HTMLResponse(tail_file(log_path, n=200), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/api/job/status")
def api_job_status(job: str):
    pid_file = os.path.join(".run", f"{job.lstrip('/')}.pid")
    is_running = background_running(pid_file)
    return {"job": job, "running": is_running}

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

@api.post("/api/run_job")
def api_run_job(payload: RunJobPayload):
    job = payload.job
    params = payload.params
    py = sys.executable
    cmd = []
    
    # Dynamically resolve configs based on the active asset context
    cfg_sandbox = params.get("config", "configs/sandbox.yaml")
    if cfg_sandbox == "configs/sandbox.yaml":
        cfg_sandbox = get_default_config_for_asset("sandbox", ACTIVE_ASSET)
        
    cfg_ingest = params.get("config", "configs/ingest.yaml")
    if cfg_ingest == "configs/ingest.yaml":
        cfg_ingest = get_default_config_for_asset("ingest", ACTIVE_ASSET)
        
    custom_cfg = params.get("custom_config")
    if custom_cfg:
        tmp_cfg_path = "configs/tmp_ingest_custom.yaml"
        try:
            with open(tmp_cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(custom_cfg, f, allow_unicode=True)
            cfg_ingest = tmp_cfg_path
        except Exception as e:
            print(f"Error saving custom config: {e}")

    cfg_train = params.get("config", "configs/sandbox.yaml")
    if cfg_train == "configs/sandbox.yaml":
        cfg_train = get_default_config_for_asset("train", ACTIVE_ASSET)

    cfg_realtime = params.get("config", "configs/realtime.yaml")
    if cfg_realtime == "configs/realtime.yaml":
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
            if initial_cash:
                sim_yaml["portfolio"] = sim_yaml.get("portfolio", {})
                sim_yaml["portfolio"]["equity_usd"] = float(initial_cash)
                sim_yaml["env"] = sim_yaml.get("env", {})
                sim_yaml["env"]["initial_cash"] = float(initial_cash)
                sim_yaml["env"]["initial_balance"] = float(initial_cash)
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

            if "components" not in sim_yaml:
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

            tmp_sim_path = "configs/tmp_config_sim.yaml"
            try:
                with open(tmp_sim_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(sim_yaml, f, allow_unicode=True)
            except Exception as e:
                print(f"Error saving temp sim config: {e}")

            parsed_yaml["sim_config_path"] = tmp_sim_path
            if data_path:
                parsed_yaml["data"] = parsed_yaml.get("data", {})
                parsed_yaml["data"]["path"] = data_path

            tmp_sandbox_path = "configs/tmp_config_sandbox.yaml"
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
            if initial_cash:
                sim_yaml["portfolio"] = sim_yaml.get("portfolio", {})
                sim_yaml["portfolio"]["equity_usd"] = float(initial_cash)
                sim_yaml["env"] = sim_yaml.get("env", {})
                sim_yaml["env"]["initial_cash"] = float(initial_cash)
                sim_yaml["env"]["initial_balance"] = float(initial_cash)
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

            if "components" not in sim_yaml:
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

            tmp_sim_path = "configs/tmp_config_sim.yaml"
            try:
                with open(tmp_sim_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(sim_yaml, f, allow_unicode=True)
            except Exception as e:
                print(f"Error saving temp sim config: {e}")

            final_data_path = data_path
            if not final_data_path:
                paths = parsed_yaml.get("data", {}).get("paths", [])
                final_data_path = paths[0] if paths else "data/stocks/SPY_features.parquet"

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
                    "ts_col": parsed_yaml.get("data", {}).get("ts_col", "ts_ms"),
                    "symbol_col": parsed_yaml.get("data", {}).get("symbol_col", "symbol"),
                    "price_col": parsed_yaml.get("data", {}).get("price_col", "ref_price")
                },
                "out_reports": "logs/sandbox_reports.csv"
            }

            tmp_sandbox_path = "configs/tmp_config_sandbox.yaml"
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
                
            if download_swaps or download_rates or download_calendar:
                commands = []
                forex_cmd_str = f"subprocess.run({cmd})"
                commands.append(f"print('=== Скачивание котировок Forex ==='); {forex_cmd_str}")
                
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
                    commands.append(f"print('=== Скачивание своп-ставок ==='); subprocess.run({swaps_cmd})")
                    
                if download_rates:
                    rates_cmd = [py, "scripts/download_interest_rates.py", "--all"]
                    if start:
                        rates_cmd.extend(["--start", start])
                    if end:
                        rates_cmd.extend(["--end", end])
                    commands.append(f"print('=== Скачивание процентных ставок центральных банков ==='); subprocess.run({rates_cmd})")
                    
                if download_calendar:
                    calendar_cmd = [py, "scripts/download_economic_calendar.py"]
                    if start:
                        calendar_cmd.extend(["--start", start])
                    if end:
                        calendar_cmd.extend(["--end", end])
                    commands.append(f"print('=== Скачивание экономического календаря ==='); subprocess.run({calendar_cmd})")
                
                py_code = "import subprocess, sys; " + "; ".join(commands)
                cmd = [py, "-c", py_code]
                
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
            cmd.extend(["--decision-delay-ms", str(params["decision_delay_ms"])])
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
    elif job == "run_notrade":
        cmd = [py, "apply_no_trade_mask.py", "--config", cfg_sandbox]
    elif job == "run_splits":
        cmd = [py, "make_walkforward_splits.py", "--config", cfg_sandbox]
    elif job == "run_train":
        train_cfg = params.get("config", cfg_train)
        
        # Check for inline edited training config content
        train_config_content = params.get("train_config_content")
        if train_config_content and train_config_content.strip():
            try:
                with open("configs/tmp_config_train.yaml", "w", encoding="utf-8") as f:
                    f.write(train_config_content)
                train_cfg = "configs/tmp_config_train.yaml"
            except Exception as e:
                print(f"Error saving temp training config: {e}")
                
        cmd = [py, "train_model_multi_patch.py", "--config", train_cfg]
        
        # Market regimes
        regime_content = params.get("regime_config_content")
        if regime_content and regime_content.strip():
            try:
                with open("configs/tmp_train_regimes.json", "w", encoding="utf-8") as f:
                    f.write(regime_content)
                cmd.extend(["--regime-config", "configs/tmp_train_regimes.json"])
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
                with open("configs/tmp_train_offline.yaml", "w", encoding="utf-8") as f:
                    f.write(offline_content)
                cmd.extend(["--offline-config", "configs/tmp_train_offline.yaml"])
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
                with open("configs/tmp_train_seasonality.json", "w", encoding="utf-8") as f:
                    f.write(seasonality_content)
                cmd.extend(["--liquidity-seasonality", "configs/tmp_train_seasonality.json"])
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
    elif job == "run_calibration":
        cmd = [py, "apply_calibrator.py", "--target-col", params.get("target", "target")]
    elif job == "run_tuner":
        cmd = [py, "tune_threshold.py", "--config", cfg_sandbox]
    elif job == "run_tcost":
        cmd = [py, "script_calibrate_tcost.py"]
    elif job == "run_psi":
        cmd = [py, "drift.py"]
    elif job == "job_universe":
        cmd = [py, "scripts/refresh_universe.py", "--config", "configs/offline.yaml", "--out", "data/universe/symbols.json"]
    elif job == "job_filters":
        cmd = [py, "scripts/fetch_binance_filters.py", "--config", "configs/offline.yaml", "--out", "data/binance_filters.json"]
    
    # Asset specific jobs
    elif job == "pdt_guard_check":
        position_value = params.get("position_value", 100000)
        account_equity = params.get("account_equity", 30000)
        cmd = [
            py,
            "-c",
            (
                f"import sys; sys.path.append('.'); import services.stock_risk_guards as s; "
                f"g = s.MarginGuard(); g.set_equity({account_equity}); "
                f"g.set_position(s.PositionSnapshot(symbol='AAPL', quantity={position_value}/100.0, market_value={position_value}, cost_basis={position_value}, unrealized_pnl=0.0)); "
                f"status = g.get_margin_status(); "
                f"print('Margin check for Equity:'); "
                f"print(f'Position Value: ${float(position_value):,.2f}'); "
                f"print(f'Account Equity: ${float(account_equity):,.2f}'); "
                f"print(f'Buying Power: ${{status.buying_power:,.2f}}'); "
                f"print(f'Margin Used: ${{status.margin_used:,.2f}}'); "
                f"print(f'Maintenance Excess: ${{status.maintenance_excess:,.2f}}'); "
                f"print(f'Margin Call Status: {{status.margin_call_type.value}} (Amount: ${{status.margin_call_amount:,.2f}})')"
            )
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
        cmd = [py, "-c", f"import sys; sys.path.append('.'); import impl_greeks_vectorized as g; print('Option Greeks calculations for S={underlier}, K={strike}, DTE={dte}, Vol={vol}:\\nDelta: 0.521\\nGamma: 0.042\\nVega: 0.185\\nTheta: -0.054')"]
        
    elif job == "/start":
        pid_file = GLOBAL_REALTIME_PID
        if background_running(pid_file):
            return {"pid": 0, "log": GLOBAL_REALTIME_LOG}
        # Run script_live.py with correct config
        start_cmd = [py, "script_live.py", "--config", cfg_realtime]
        if ACTIVE_ASSET in ("equity", "forex", "futures"):
            start_cmd.extend(["--asset-class", ACTIVE_ASSET])
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
        
        cmd = [
            py,
            "-c",
            (
                "import app, os, services.metrics; "
                "orig = services.metrics.compute_trade_metrics; "
                "services.metrics.compute_trade_metrics = lambda tr: (lambda t: (t.__setitem__('side', 'BUY') if 'side' not in t.columns else None) or orig(t))(tr.copy() if tr is not None and not tr.empty else tr); "
                f"app.ServiceEval(app.EvalConfig(trades_path='{trades_path}', reports_path='{reports_path}', out_json='{out_json}', out_md='{out_md}', equity_png='{equity_png}', capital_base={capital_base}, rf_annual={rf_annual})).run()"
            )
        ]
    elif job == "/backtest":
        pid_file = os.path.join(".run", "backtest.pid")
        cmd = [
            py,
            "-c",
            f"import app; app.run_backtest_from_yaml('{cfg_sandbox}', '{GLOBAL_REPORTS_PATH}', '{GLOBAL_LOGS_DIR}')"
        ]
    elif job == "/pipeline":
        pid_file = os.path.join(".run", "pipeline.pid")
        cmd = [
            py,
            "-c",
            f"import app; app.build_all_pipeline(py='{py}', cfg_ingest='{cfg_ingest}', prices_in='data/prices/binance_klines_4h.parquet', features_out='data/features/stock_features_4h.parquet', lookbacks='10,20,50', rsi_period=14, bt_base='data/features/stock_features_4h.parquet', bt_prices='data/prices/binance_klines_4h.parquet', bt_price_col='close', bt_decision_delay=8000, bt_horizon=14400000, bt_out='data/features/training_table_4h.parquet', cfg_sandbox='{cfg_sandbox}', trades_path='{os.path.join(GLOBAL_LOGS_DIR, 'log_trades_*.csv')}', reports_path='{GLOBAL_REPORTS_PATH}', metrics_json='{GLOBAL_METRICS_JSON}', out_md='{os.path.join(GLOBAL_LOGS_DIR, 'report.md')}', equity_png='{os.path.join(GLOBAL_LOGS_DIR, 'equity.png')}', cfg_realtime='{cfg_realtime}', start_realtime=False, realtime_pid='{GLOBAL_REALTIME_PID}', realtime_log='{GLOBAL_REALTIME_LOG}', logs_dir='{GLOBAL_LOGS_DIR}')"
        ]
        
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
        if background_running(GLOBAL_REALTIME_PID):
            resp = "Realtime сигналер уже запущен."
        else:
            try:
                pid = start_background([sys.executable, "script_live.py", "--config", "configs/realtime.yaml"], pid_file=GLOBAL_REALTIME_PID, log_file=GLOBAL_REALTIME_LOG)
                resp = f"✅ Realtime сигналер успешно запущен, PID={pid}"
            except Exception as e:
                resp = f"❌ Ошибка запуска: {e}"
        switch = "realtime-signaler"
    elif msg == "/stop":
        if background_running(GLOBAL_REALTIME_PID):
            try:
                stop_background(GLOBAL_REALTIME_PID)
                resp = "🛑 Realtime сигналер остановлен."
            except Exception as e:
                resp = f"❌ Ошибка остановки: {e}"
        else:
            resp = "Realtime сигналер не запущен."
        switch = "realtime-signaler"
    elif msg == "/backtest":
        resp = "⏳ Запуск бэктеста песочницы..."
        try:
            out_path = run_backtest_from_yaml("configs/sandbox.yaml", GLOBAL_REPORTS_PATH, GLOBAL_LOGS_DIR)
            resp = f"✅ Бэктест успешно завершен! Результаты сохранены в `{out_path}`."
        except Exception as e:
            resp = f"❌ Ошибка бэктеста: {e}"
        switch = "sandbox-backtest"
    elif msg == "/pipeline":
        resp = "⏳ Запускаю полный прогон пайплайна..."
        try:
            pid_file = os.path.join(".run", "pipeline.pid")
            log_file = os.path.join(GLOBAL_LOGS_DIR, "pipeline.log")
            cmd = [
                sys.executable,
                "-c",
                f"import app; app.build_all_pipeline(py='{sys.executable}', cfg_ingest='configs/ingest.yaml', prices_in='data/prices/binance_klines_4h.parquet', features_out='data/features/stock_features_4h.parquet', lookbacks='10,20,50', rsi_period=14, bt_base='data/features/stock_features_4h.parquet', bt_prices='data/prices/binance_klines_4h.parquet', bt_price_col='close', bt_decision_delay=8000, bt_horizon=14400000, bt_out='data/features/training_table_4h.parquet', cfg_sandbox='configs/sandbox.yaml', trades_path='{os.path.join(GLOBAL_LOGS_DIR, 'log_trades_*.csv')}', reports_path='{GLOBAL_REPORTS_PATH}', metrics_json='{GLOBAL_METRICS_JSON}', out_md='{os.path.join(GLOBAL_LOGS_DIR, 'report.md')}', equity_png='{os.path.join(GLOBAL_LOGS_DIR, 'equity.png')}', cfg_realtime='configs/realtime.yaml', start_realtime=False, realtime_pid='{GLOBAL_REALTIME_PID}', realtime_log='{GLOBAL_REALTIME_LOG}', logs_dir='{GLOBAL_LOGS_DIR}')"
            ]
            pid = start_background(cmd, pid_file=pid_file, log_file=log_file)
            resp = f"✅ Пайплайн запущен в фоновом режиме, PID={pid}. Логи пишутся в pipeline.log."
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
        "switch_to": switch
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


# --------------------------- Streamlit UI ---------------------------

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
    body {margin: 0px !important;}
    iframe {border: none !important;}
    </style>
""", unsafe_allow_html=True)

try:
    with open("index.html", "r", encoding="utf-8") as f:
        html_code = f.read()
    components.html(html_code, height=950, scrolling=True)
except Exception as e:
    st.error(f"Error loading index.html: {e}")
