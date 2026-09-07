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
        assert pf["allow"] is False  # enabled + unknown gas → don't send

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
        monkeypatch.setattr(
            go,
            "get_gas_price_gwei",
            lambda chain="ethereum", **k: {
                "ok": True,
                "gas_gwei": 12.0,
                "chain": chain,
                "rpc": "mock",
            },
        )
        res = client.post(
            "/api/web3/gas_guard", json={"enabled": True, "threshold_gwei": 30, "chain": "ethereum"}
        )
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
    return (
        key,
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode(),
    )


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
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
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

        c = fb.FireblocksClient(
            fb.FireblocksConfig(api_key="api-uuid"), private_key_pem=pem, request_fn=mock_req
        )
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
            fb.FireblocksClient(
                fb.FireblocksConfig(api_key="k", private_key_path=str(tmp_path / "nope.key"))
            )

    def test_api_status_and_connect_honest(self):
        assert client.get("/api/custody/fireblocks/status").status_code == 200
        r = client.post(
            "/api/custody/fireblocks/connect", json={"api_key": "", "private_key_path": ""}
        )
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
    assert 'id="liteport-mpc-keypath"' in HTML  # private-key path input added


def test_ui_walletconnect_replaced_with_eip6963():
    assert "WalletConnect — недоступен" not in HTML
    assert "eip6963:announceProvider" in HTML and "eip6963:requestProvider" in HTML
    assert "function connectAnyWallet" in HTML


# ============================================================ Fireblocks send/withdraw


class TestFireblocksTransfer:
    def test_validate_transfer(self):
        from services.custody import fireblocks_client as fb

        assert (
            fb.validate_transfer("ETH", "0.1", "ONE_TIME_ADDRESS", address="0x" + "a" * 40) is None
        )
        assert "адрес" in fb.validate_transfer("ETH", "0.1", "ONE_TIME_ADDRESS", address="0xbad")
        assert "amount" in fb.validate_transfer(
            "ETH", "-1", "ONE_TIME_ADDRESS", address="0x" + "a" * 40
        )
        assert "amount" in fb.validate_transfer(
            "ETH", "abc", "ONE_TIME_ADDRESS", address="0x" + "a" * 40
        )
        assert "id" in fb.validate_transfer("ETH", "1", "VAULT_ACCOUNT", dest_id="")
        assert "назначения" in fb.validate_transfer("ETH", "1", "SOMETHING_ELSE")

    def test_build_transfer_payload_string_amount_idempotent(self):
        from services.custody import fireblocks_client as fb

        b = fb.build_transfer_payload(
            asset_id="USDC",
            amount="12.5",
            source_vault_id="0",
            dest_type="ONE_TIME_ADDRESS",
            address="0x" + "b" * 40,
            external_tx_id="riven-abc",
            note="test",
        )
        assert isinstance(b["amount"], str) and b["amount"] == "12.5"  # never float
        assert b["externalTxId"] == "riven-abc"  # idempotency
        assert b["source"] == {"type": "VAULT_ACCOUNT", "id": "0"}
        assert b["destination"]["type"] == "ONE_TIME_ADDRESS"
        assert b["destination"]["oneTimeAddress"]["address"] == "0x" + "b" * 40

    def test_estimate_and_create_transaction_hit_right_endpoints(self, rsa_pem):
        from services.custody import fireblocks_client as fb

        _, pem = rsa_pem
        seen = []

        def mock_req(method, url, headers, body):
            seen.append((method, url))
            if "estimate_fee" in url:
                return {"medium": {"networkFee": "0.0005"}}
            return {"id": "tx-123", "status": "SUBMITTED"}

        c = fb.FireblocksClient(
            fb.FireblocksConfig(api_key="k"), private_key_pem=pem, request_fn=mock_req
        )
        body = fb.build_transfer_payload(
            asset_id="ETH",
            amount="0.1",
            source_vault_id="0",
            dest_type="ONE_TIME_ADDRESS",
            address="0x" + "a" * 40,
            external_tx_id="riven-1",
        )
        est = c.estimate_fee(body)
        tx = c.create_transaction(body)
        assert est["medium"]["networkFee"] == "0.0005"
        assert tx["id"] == "tx-123" and tx["status"] == "SUBMITTED"
        assert any(m == "POST" and "estimate_fee" in u for m, u in seen)
        assert any(m == "POST" and u.endswith("/v1/transactions") for m, u in seen)


def _configure_fb(monkeypatch, tmp_path, rsa_pem):
    """Point the app's fireblocks config at a temp configured vault + mock the API."""
    from services.custody import fireblocks_client as fb

    key, pem = rsa_pem
    keyfile = tmp_path / "fb.key"
    keyfile.write_text(pem, encoding="utf-8")
    cfg = fb.FireblocksConfig(
        api_key="api-uuid", private_key_path=str(keyfile), base_url=fb.SANDBOX_URL
    )
    monkeypatch.setattr(fb, "load_config", lambda *a, **k: cfg)
    return cfg


class TestWithdrawCeremony:
    def _mock_client(self, monkeypatch, create_result=None):
        from services.custody import fireblocks_client as fb

        calls = {"create": 0}

        def fake_request(self, method, path, body=None):
            if "estimate_fee" in path:
                return {"medium": {"networkFee": "0.0004"}}
            if path == "/v1/transactions" and method == "POST":
                calls["create"] += 1
                return create_result or {"id": "tx-xyz", "status": "SUBMITTED"}
            if path.startswith("/v1/transactions/"):
                return {"id": "tx-xyz", "status": "COMPLETED"}
            return {}

        monkeypatch.setattr(fb.FireblocksClient, "_request", fake_request)
        return calls

    def test_preview_honest_when_not_configured(self, monkeypatch):
        from services.custody import fireblocks_client as fb

        monkeypatch.setattr(fb, "load_config", lambda *a, **k: fb.FireblocksConfig())
        r = client.post(
            "/api/custody/fireblocks/withdraw/preview",
            json={
                "asset_id": "ETH",
                "amount": "0.1",
                "source_vault_id": "0",
                "dest_type": "ONE_TIME_ADDRESS",
                "address": "0x" + "a" * 40,
            },
        )
        assert r.status_code == 200 and r.json()["ok"] is False
        assert "не настроен" in r.json()["error"].lower()

    def test_preview_validates(self, monkeypatch, tmp_path, rsa_pem):
        _configure_fb(monkeypatch, tmp_path, rsa_pem)
        self._mock_client(monkeypatch)
        r = client.post(
            "/api/custody/fireblocks/withdraw/preview",
            json={
                "asset_id": "ETH",
                "amount": "0.1",
                "source_vault_id": "0",
                "dest_type": "ONE_TIME_ADDRESS",
                "address": "0xbad",
            },
        )
        assert r.status_code == 400

    def test_full_two_step_flow_and_idempotency(self, monkeypatch, tmp_path, rsa_pem):
        _configure_fb(monkeypatch, tmp_path, rsa_pem)
        # gas guard: not enabled → allow
        import services.web3.gas_oracle as go

        monkeypatch.setattr(go, "load_config", lambda *a, **k: go.GasGuardConfig(enabled=False))
        calls = self._mock_client(monkeypatch)
        # isolate journal
        import app as m

        monkeypatch.setattr(m, "FB_WITHDRAW_JOURNAL", str(tmp_path / "wd.jsonl"))

        prev = client.post(
            "/api/custody/fireblocks/withdraw/preview",
            json={
                "asset_id": "ETH",
                "amount": "0.1",
                "source_vault_id": "0",
                "dest_type": "ONE_TIME_ADDRESS",
                "address": "0x" + "a" * 40,
            },
        ).json()
        assert (
            prev["ok"]
            and prev["confirmation_token"]
            and prev["external_tx_id"].startswith("riven-")
        )
        rid, tok = prev["request_id"], prev["confirmation_token"]

        # submit without confirm → refused
        assert (
            client.post(
                "/api/custody/fireblocks/withdraw/submit",
                json={"request_id": rid, "confirmation_token": tok, "confirm": False},
            ).status_code
            == 400
        )
        # wrong token → 403
        assert (
            client.post(
                "/api/custody/fireblocks/withdraw/submit",
                json={"request_id": rid, "confirmation_token": "nope", "confirm": True},
            ).status_code
            == 403
        )
        # correct submit → sends
        ok = client.post(
            "/api/custody/fireblocks/withdraw/submit",
            json={"request_id": rid, "confirmation_token": tok, "confirm": True},
        )
        assert ok.status_code == 200, ok.text
        body = ok.json()
        assert body["ok"] and body["tx_id"] == "tx-xyz" and body["status"] == "SUBMITTED"
        # single-use token: replay → 404 (anti-replay)
        assert (
            client.post(
                "/api/custody/fireblocks/withdraw/submit",
                json={"request_id": rid, "confirmation_token": tok, "confirm": True},
            ).status_code
            == 404
        )
        assert calls["create"] == 1  # sent exactly once
        # journal recorded the submit
        j = client.get("/api/custody/fireblocks/withdrawals").json()["withdrawals"]
        assert any(w["event"] == "withdraw_submitted" and w["tx_id"] == "tx-xyz" for w in j)

    def test_gas_guard_blocks_submit(self, monkeypatch, tmp_path, rsa_pem):
        _configure_fb(monkeypatch, tmp_path, rsa_pem)
        self._mock_client(monkeypatch)
        import app as m

        monkeypatch.setattr(m, "FB_WITHDRAW_JOURNAL", str(tmp_path / "wd.jsonl"))
        # gas guard enabled + breached → block on submit (fresh check)
        import services.web3.gas_oracle as go

        monkeypatch.setattr(
            go, "load_config", lambda *a, **k: go.GasGuardConfig(enabled=True, threshold_gwei=10.0)
        )
        monkeypatch.setattr(
            go,
            "get_gas_price_gwei",
            lambda chain="ethereum", **k: {"ok": True, "gas_gwei": 99.0, "chain": chain},
        )
        prev = client.post(
            "/api/custody/fireblocks/withdraw/preview",
            json={
                "asset_id": "ETH",
                "amount": "0.1",
                "source_vault_id": "0",
                "dest_type": "ONE_TIME_ADDRESS",
                "address": "0x" + "a" * 40,
            },
        ).json()
        assert prev["gas_guard"]["allow"] is False  # informational at preview
        r = client.post(
            "/api/custody/fireblocks/withdraw/submit",
            json={
                "request_id": prev["request_id"],
                "confirmation_token": prev["confirmation_token"],
                "confirm": True,
            },
        )
        assert r.status_code == 409 and "Gas Guard" in r.json()["detail"]


def test_ui_fireblocks_withdraw_form():
    assert "function previewFireblocksWithdraw" in HTML
    assert "function submitFireblocksWithdraw" in HTML
    assert "/api/custody/fireblocks/withdraw/preview" in HTML
    assert "/api/custody/fireblocks/withdraw/submit" in HTML
    # explicit human confirm before moving real funds
    assert "Отправить реальные средства из Fireblocks-vault" in HTML
