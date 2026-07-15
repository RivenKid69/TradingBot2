# -*- coding: utf-8 -*-
import os
import pytest
from fastapi.testclient import TestClient
import yaml

os.environ.setdefault("SEASONALITY_API_TOKEN", "test-token-api-config")

import app as app_module
from app import api

# The global auth middleware only whitelists loopback peers; the TestClient
# peer is "testclient", so authenticate explicitly with the API token.
client = TestClient(api, headers={"X-API-Key": app_module.API_TOKEN})

@pytest.fixture
def temp_configs(tmp_path):
    config_sim_data = {
        "execution": {
            "mode": "bar",
            "bar_price": "close",
            "intrabar_price_model": "bridge",
            "timeframe_ms": 14400000,
            "clip_to_bar": {"enabled": True, "strict_open_fill": False},
            "bar_capacity_base": {
                "enabled": False,
                "capacity_frac_of_ADV_base": 0.05,
                "floor_base": 10.0,
                "adv_base_path": "data/liquidity/adv_base.json"
            }
        },
        "latency": {
            "base_ms": 250.0,
            "jitter_ms": 50.0,
            "spike_p": 0.01,
            "spike_mult": 5.0,
            "use_seasonality": True,
            "seed": 0
        },
        "slippage": {
            "dynamic": {
                "enabled": False,
                "path": "models/slippage_calibration.json",
                "smoothing_alpha": 0.10,
                "vol_metric": "hl",
                "liq_col": "volume",
                "liq_ref": 240000.0
            }
        },
        "ws_dedup": {
            "enabled": False,
            "log_skips": True,
            "persist_path": "logs/ws_dedup_state.json"
        },
        "execution_profile": "Conservative",
        "execution_profiles_definitions": {
            "Conservative": {"offset_ticks": 2, "ttl_ms": 5000, "tif": "GTC"},
            "Balanced": {"offset_ticks": 0, "ttl_ms": 2000, "tif": "GTC"},
            "Aggressive": {"offset_ticks": -1, "ttl_ms": 500, "tif": "IOC"}
        }
    }
    
    sandbox_data = {
        "mode": "backtest",
        "sim_config_path": str(tmp_path / "config_sim.yaml"),
        "dynamic_spread": {
            "enabled": False,
            "vol_mode": "hl",
            "liq_col": "volume",
            "liq_ref": 240000.0
        }
    }
    
    config_sim_path = tmp_path / "config_sim.yaml"
    sandbox_path = tmp_path / "sandbox.yaml"
    
    with open(config_sim_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_sim_data, f)
        
    with open(sandbox_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(sandbox_data, f)
        
    return str(config_sim_path), str(sandbox_path)


def test_get_backtest_settings(temp_configs):
    config_sim_path, sandbox_path = temp_configs
    
    response = client.get(
        f"/api/config/get_backtest_settings?config_path={config_sim_path}&sandbox_path={sandbox_path}"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "bar"
    assert data["latency_base"] == 250.0
    assert data["seasonality"] is True
    assert data["active_profile"] == "Conservative"
    assert "Conservative" in data["profiles"]


def test_save_backtest_settings(temp_configs):
    config_sim_path, sandbox_path = temp_configs
    
    save_payload = {
        "config_path": config_sim_path,
        "sandbox_path": sandbox_path,
        "mode": "order",
        "bar_price": "open",
        "latency_base": 100.0,
        "latency_jitter": 20.0,
        "spike_p": 0.05,
        "spike_mult": 3.0,
        "seasonality": False,
        "intrabar_price_model": "linear",
        "timeframe_ms": 3600000,
        "seed_mode": "random",
        "use_latency_from": "constant",
        "latency_constant_ms": 150.0,
        "next_bar_open": True,
        "clip_next_bar": False,
        "strict_open": True,
        "active_profile": "LIMIT_MID_BPS",
        "profiles": {
            "Conservative": {"offset_ticks": 3, "ttl_ms": 6000, "tif": "GTC"},
            "LIMIT_MID_BPS": {"limit_offset_bps": 2.5, "ttl_steps": 10, "tif": "IOC"}
        },
        "slip_enabled": True,
        "slip_path": "models/custom_slippage.json",
        "smoothing_alpha": 0.25,
        "vol_mode": "ret",
        "liq_col": "number_of_trades",
        "liq_ref": 100000.0,
        "cap_enabled": True,
        "cap_frac": 0.10,
        "cap_floor": 5.0,
        "cap_path": "data/custom_adv.json",
        "ws_enabled": True,
        "ws_skips": False,
        "ws_path": "logs/custom_ws.json"
    }
    
    response = client.post("/api/config/save_backtest_settings", json=save_payload)
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    
    # Reload and assert changes are saved in config_sim.yaml
    with open(config_sim_path, "r", encoding="utf-8") as f:
        saved_config = yaml.safe_load(f)
        
    assert saved_config["execution"]["mode"] == "order"
    assert saved_config["execution"]["bar_price"] == "open"
    assert saved_config["latency"]["base_ms"] == 100.0
    assert saved_config["latency"]["use_seasonality"] is False
    assert saved_config["latency"]["seed"] == 42 # random seed mode
    assert saved_config["execution"]["entry_mode"] == "next_bar_open"
    assert saved_config["execution"]["clip_to_bar"]["enabled"] is False
    assert saved_config["execution"]["clip_to_bar"]["strict_open_fill"] is True
    assert saved_config["execution_profile"] == "LIMIT_MID_BPS"
    assert saved_config["execution_params"]["limit_offset_bps"] == 2.5
    assert saved_config["execution_params"]["ttl_steps"] == 10
    assert saved_config["execution_params"]["tif"] == "IOC"
    assert saved_config["slippage"]["dynamic"]["enabled"] is True
    assert saved_config["slippage"]["dynamic"]["vol_metric"] == "ret"
    assert saved_config["execution"]["bar_capacity_base"]["enabled"] is True
    assert saved_config["execution"]["bar_capacity_base"]["capacity_frac_of_ADV_base"] == 0.10
    assert saved_config["ws_dedup"]["enabled"] is True
    assert saved_config["ws_dedup"]["log_skips"] is False
    
    # Reload and assert changes are saved in sandbox.yaml
    with open(sandbox_path, "r", encoding="utf-8") as f:
        saved_sandbox = yaml.safe_load(f)
    assert saved_sandbox["dynamic_spread"]["enabled"] is True
    assert saved_sandbox["dynamic_spread"]["vol_mode"] == "ret"
    assert saved_sandbox["dynamic_spread"]["liq_col"] == "number_of_trades"
    assert saved_sandbox["dynamic_spread"]["liq_ref"] == 100000.0
