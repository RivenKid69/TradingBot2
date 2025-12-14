# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import httpx
import pytest

from packages.agent.cloud.client import CloudClient, CloudClientConfig, AgentIdentity


def test_cloud_client_enroll_signs_and_parses_response():
    agent_id = uuid4()
    workspace_id = uuid4()
    org_id = uuid4()

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/auth/agent/enroll"
        assert "X-CCEA-Signature" in request.headers
        assert "X-CCEA-Public-Key-Fingerprint" in request.headers

        body = json.loads(request.content.decode("utf-8"))
        captured["body"] = body

        return httpx.Response(
            201,
            json={
                "agent_id": str(agent_id),
                "access_token": "agent.jwt.token",
                "workspace_id": str(workspace_id),
                "org_id": str(org_id),
            },
            request=request,
        )

    client = CloudClient(
        CloudClientConfig(base_url="https://cloud.example", timeout_seconds=5),
        identity=AgentIdentity.generate(),
        transport=httpx.MockTransport(handler),
    )
    res = client.enroll(
        enrollment_token="enroll-token",
        agent_name="agent-1",
        agent_version="1.0.0",
        capabilities=["execute_run"],
    )

    assert res.agent_id == agent_id
    assert res.access_token == "agent.jwt.token"
    assert client.access_token == "agent.jwt.token"
    assert captured["body"]["agent_name"] == "agent-1"
    assert captured["body"]["enrollment_token"] == "enroll-token"
    assert "public_key" in captured["body"]


def test_cloud_client_heartbeat_includes_bearer_auth_and_signature():
    called = {"ok": False}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/agent/heartbeat"
        assert request.headers.get("Authorization") == "Bearer agent.jwt.token"
        assert "X-CCEA-Signature" in request.headers

        called["ok"] = True
        return httpx.Response(
            200,
            json={
                "server_time": datetime.now(timezone.utc).isoformat(),
                "trust_state": "enrolled",
                "pending_commands": 0,
                "next_heartbeat_sec": 30,
            },
            request=request,
        )

    client = CloudClient(
        CloudClientConfig(base_url="https://cloud.example", timeout_seconds=5),
        identity=AgentIdentity.generate(),
        access_token="agent.jwt.token",
        transport=httpx.MockTransport(handler),
    )

    hb = client.heartbeat(agent_version="1.0.0", current_state="RUNNING")
    assert called["ok"] is True
    assert hb.trust_state == "enrolled"

