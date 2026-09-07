# tests/test_strategy_compilation.py
import os
import pathlib
import sys
import pytest

# Mock environmental token required by app.py
os.environ["SEASONALITY_API_TOKEN"] = "dummy_test_token"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import api
from fastapi.testclient import TestClient

client = TestClient(api)


def test_get_strategy_templates():
    # Test getting templates for equity
    res = client.get("/api/strategy/templates?asset=equity")
    assert res.status_code == 200
    templates = res.json()
    assert "Mean Reversion (Возврат к среднему)" in templates
    assert "BaseSignalPolicy" in templates["Mean Reversion (Возврат к среднему)"]

    # Test getting templates for invalid asset
    res2 = client.get("/api/strategy/templates?asset=invalid")
    assert res2.status_code == 200
    assert res2.json() == {}


def test_save_strategy_success(tmp_path, monkeypatch):
    # Ensure strategies folder exists
    os.makedirs("strategies", exist_ok=True)

    valid_code = """# custom strategy
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class MockTestStrategy(BaseSignalPolicy):
    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        return []
"""
    payload = {
        "asset": "equity",
        "template_name": "Mean Reversion (Возврат к среднему)",
        "code": valid_code,
        "params": {"lookback": 10, "enter_threshold": 1.5},
    }

    res = client.post("/api/save_strategy", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "MockTestStrategy" in data["message"]

    # Check file exists on disk
    filepath = os.path.join("strategies", "custom_equity.py")
    assert os.path.exists(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    assert "class MockTestStrategy" in content


def test_save_strategy_syntax_error():
    invalid_code = """# invalid syntax python code
class BrokenStrategy(BaseSignalPolicy)
    def decide(self, features, ctx)
        return []
"""
    payload = {
        "asset": "forex",
        "template_name": "Grid Trading",
        "code": invalid_code,
        "params": {},
    }
    res = client.post("/api/save_strategy", json=payload)
    assert res.status_code == 400
    assert "Ошибка синтаксиса" in res.json()["detail"]


def test_save_strategy_no_decide_class():
    code_no_class = """# valid code but no Strategy class
def helper_func():
    return 42
"""
    payload = {"asset": "crypto", "template_name": "Arbitrage", "code": code_no_class, "params": {}}
    res = client.post("/api/save_strategy", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "warning"
    assert "decide" in data["message"]


def test_get_strategy():
    # Fetch strategy for equity (which was saved in test_save_strategy_success)
    res = client.get("/api/strategy?asset=equity")
    assert res.status_code == 200
    data = res.json()
    assert "code" in data
    assert "params" in data
    assert "MockTestStrategy" in data["code"]
    assert data["params"]["lookback"] == 10

    # Fetch strategy for a new asset that doesn't exist on disk (should load fallback)
    res2 = client.get("/api/strategy?asset=options")
    assert res2.status_code == 200
    data2 = res2.json()
    assert "code" in data2
    assert "params" in data2
    assert "options" in data2["code"] or "base" in data2["code"] or "Signal" in data2["code"]
