# -*- coding: utf-8 -*-
"""Gas Guard + Fireblocks MPC + EIP-6963 multi-wallet — closing the three
"honestly disabled / NOT IMPLEMENTED" Lite-portfolio features with REAL
implementations.

- Gas Guard: real on-chain gas oracle (public RPC) + persisted threshold +
  armed/breached verdict + fail-closed preflight.
- Fireblocks: real RS256 JWT-signed API client (works with real creds, honest
  refusal without); private key never copied into our storage.
- EIP-6963: real multi-injected-wallet discovery markup (offline, no SDK).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("SEASONALITY_API_TOKEN", "test-token-web3")
os.environ.setdefault("RIVEN_ENABLE_CCEA", "0")

from fastapi.testclient import TestClient

import app as app_module
from app import api

client = TestClient(api, headers={"X-API-Key": app_module.API_TOKEN})

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "index.html").read_text(encoding="utf-8")


# ============================================================ Gas Guard

class TestGasGuard:
    def _mock(self, gwei):
        return lambda url, payload: {"result": hex(int(gwei * 1e9))}

    def test_armed_below_threshold(self):
        from services.web3 import gas_oracle
        cfg = gas_oracle.GasGuardConfig(enabled=True, threshold_gwei=35.0)
        v = gas_oracle.evaluate(cfg=cfg, fetch_fn=self._mock(20))
        assert v["status"] == "armed" and v["blocked"] is False
        assert v["gas_gwei"] == 20.0 and v["usage_pct"] == pytest.approx(57.1, abs=0.2)

    def test_breached_above_threshold(self):
        from services.web3 import gas_oracle
        cfg = gas_oracle.GasGuardConfig(enabled=True, threshold_gwei=35.0)
        v = gas_oracle.evaluate(cfg=cfg, fetch_fn=self._mock(80))
        assert v["status"] == "breached" and v["blocked"] is True

    def test_disabled_never_blocks(self):
        from services.web3 import gas_oracle
        cfg = gas_oracle.GasGuardConfig(enabled=False, threshold_gwei=35.0)
        v = gas_oracle.evaluate(cfg=cfg, fetch_fn=self._mock(500))
        assert v["status"] == "disabled" and v["blocked"] is False

    def test_preflight_fail_closed_when_gas_unavailable(self):
        from services.web3 import gas_oracle

        def boom(url, payload):
            raise RuntimeError("rpc down")
        cfg = gas_oracle.GasGuardConfig(enabled=True, threshold_gwei=35.0)
        pf = gas_oracle.preflight(cfg=cfg, fetch_fn=boom)
        assert pf["allow"] is False   # enabled + unknown gas → don't send

    def test_preflight_fail_open_when_disabled(self):
        from services.web3 import gas_oracle

        def boom(url, payload):
            raise RuntimeError("rpc down")
        cfg = gas_oracle.GasGuardConfig(enabled=False)
        assert gas_oracle.preflight(cfg=cfg, fetch_fn=boom)["allow"] is True

    def test_config_persist_roundtrip(self, tmp_path):
        from services.web3 import gas_oracle
        p = str(tmp_path / "gg.json")
        gas_oracle.save_config(gas_oracle.GasGuardConfig(enabled=True, threshold_gwei=42.0), p)
        cfg = gas_oracle.load_config(p)
        assert cfg.enabled is True and cfg.threshold_gwei == 42.0

    def test_api_gas_guard_save_and_verdict(self, tmp_path, monkeypatch):
        # isolate the persisted config to a temp path
        import services.web3.gas_oracle as go
        monkeypatch.setattr(go, "CONFIG_PATH", str(tmp_path / "gg.json"))
        monkeypatch.setattr(go, "get_gas_price_gwei",
                            lambda chain="ethereum", **k: {"ok": True, "gas_gwei": 12.0,
                                                           "chain": chain, "rpc": "mock"})
        res = client.post("/api/web3/gas_guard", json={"enabled": True, "threshold_gwei": 30, "chain": "ethereum"})
        assert res.status_code == 200, res.text
        v = res.json()["verdict"]
        assert v["status"] == "armed" and v["gas_gwei"] == 12.0

    def test_api_gas_guard_rejects_bad_threshold(self):
        assert client.post("/api/web3/gas_guard", json={"threshold_gwei": 0}).status_code == 400


# ============================================================ Fireblocks

@pytest.fixture
def rsa_pem():
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key, key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()).decode()


class TestFireblocks:
    def test_not_configured_is_honest(self):
        from services.custody import fireblocks_client as fb
        r = fb.connect(fb.FireblocksConfig())
        assert r["ok"] is False and r["configured"] is False
        assert "не настроен" in r["error"].lower() or "api key" in r["error"].lower()

    def test_real_rs256_jwt_signing(self, rsa_pem):
        import jwt as pyjwt
        from cryptography.hazmat.primitives import serialization
        from services.custody import fireblocks_client as fb
        key, pem = rsa_pem
        cfg = fb.FireblocksConfig(api_key="api-uuid", base_url=fb.SANDBOX_URL)
        c = fb.FireblocksClient(cfg, private_key_pem=pem, request_fn=lambda *a: {})
        token = c._sign_jwt("/v1/vault/accounts_paged?limit=1", "")
        pub = key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        claims = pyjwt.decode(token, pub, algorithms=["RS256"], options={"verify_exp": False})
        assert claims["uri"] == "/v1/vault/accounts_paged?limit=1"
        assert claims["sub"] == "api-uuid"
        assert claims["bodyHash"] == hashlib.sha256(b"").hexdigest()
        assert "nonce" in claims and "iat" in claims and "exp" in claims

    def test_test_connection_parses_vaults(self, rsa_pem):
        from services.custody import fireblocks_client as fb
        _, pem = rsa_pem

        def mock_req(method, url, headers, body):
            assert headers["X-API-Key"] == "api-uuid"
            assert headers["Authorization"].startswith("Bearer ")
            assert "/v1/vault/accounts_paged" in url
            return {"accounts": [{"id": "0", "name": "Main"}], "paging": {}}
        c = fb.FireblocksClient(fb.FireblocksConfig(api_key="api-uuid"),
                                private_key_pem=pem, request_fn=mock_req)
        tc = c.test_connection()
        assert tc["ok"] and tc["vault_accounts_visible"] == 1 and tc["sample_vault"] == "Main"

    def test_connect_surfaces_api_error_honestly(self, rsa_pem):
        from services.custody import fireblocks_client as fb
        _, pem = rsa_pem

        def mock_401(method, url, headers, body):
            raise fb.FireblocksError("Fireblocks API 401: unauthorized")
        r = fb.connect(fb.FireblocksConfig(api_key="k"), private_key_pem=pem, request_fn=mock_401)
        assert r["ok"] is False and "401" in r["error"]

    def test_config_never_stores_raw_private_key(self, tmp_path):
        from services.custody import fireblocks_client as fb
        p = str(tmp_path / "fb.json")
        fb.save_config(fb.FireblocksConfig(api_key="k", private_key_path="/x/secret.key"), p)
        saved = json.loads(Path(p).read_text(encoding="utf-8"))
        assert "private_key_path" in saved
        assert "private_key_pem" not in saved and "private_key" not in saved

    def test_missing_key_file_errors(self, tmp_path):
        from services.custody import fireblocks_client as fb
        with pytest.raises(fb.FireblocksError):
            fb.FireblocksClient(fb.FireblocksConfig(api_key="k",
                                                    private_key_path=str(tmp_path / "nope.key")))

    def test_api_status_and_connect_honest(self):
        assert client.get("/api/custody/fireblocks/status").status_code == 200
        r = client.post("/api/custody/fireblocks/connect", json={"api_key": "", "private_key_path": ""})
        assert r.status_code == 200 and r.json()["ok"] is False


# ============================================================ UI markup

def test_ui_gas_guard_no_longer_not_implemented():
    assert "Gas Guard — NOT IMPLEMENTED" not in HTML
    assert "function saveGasGuard" in HTML and "/api/web3/gas_guard" in HTML
    # slider is enabled (no `disabled` + cursor-not-allowed on it)
    assert 'id="liteport-gas-alarm-slider" min="1" max="200" step="1" value="35" oninput' in HTML


def test_ui_fireblocks_real_connect():
    assert "Fireblocks MPC — не настроен" not in HTML
    assert "function connectFireblocks" in HTML and "/api/custody/fireblocks/connect" in HTML
    assert 'id="liteport-mpc-keypath"' in HTML   # private-key path input added


def test_ui_walletconnect_replaced_with_eip6963():
    assert "WalletConnect — недоступен" not in HTML
    assert "eip6963:announceProvider" in HTML and "eip6963:requestProvider" in HTML
    assert "function connectAnyWallet" in HTML
