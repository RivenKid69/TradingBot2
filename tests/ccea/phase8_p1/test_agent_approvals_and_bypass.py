# -*- coding: utf-8 -*-

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
import json

from packages.agent.cloud.client import CloudClient, CloudClientConfig, AgentIdentity
from packages.agent.cloud.types import PendingCommand
from packages.agent.daemon.agentd import AgentDaemon, DaemonConfig


def test_trading_impacting_command_auto_approval_submits_evidence(tmp_path: Path):
    cmd_id = uuid4()
    captured = {"approval": None}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(f"/commands/{cmd_id}/approval"):
            captured["approval"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={"command_id": str(cmd_id), "status": "approved", "approved": True},
                request=request,
            )

        if request.url.path == "/api/v1/agent/heartbeat":
            return httpx.Response(
                200,
                json={
                    "server_time": datetime.now(timezone.utc).isoformat(),
                    "trust_state": "enrolled",
                    "pending_commands": 1,
                    "next_heartbeat_sec": 30,
                },
                request=request,
            )

        if request.url.path.startswith("/api/v1/agent/commands/poll"):
            return httpx.Response(
                200,
                json={
                    "commands": [
                        {
                            "id": str(cmd_id),
                            "status": "pending_approval",
                            "idempotency_key": "idem-1",
                            "command_type": "REQUEST_START_RUN",
                            "payload_ref": "sha256:abc",
                            "change_class": "trading_impacting",
                            "requires_approval": True,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                    ],
                    "has_more": False,
                    "poll_again_after_sec": 5,
                },
                request=request,
            )

        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    cloud_client = CloudClient(
        CloudClientConfig(base_url="https://cloud.example", timeout_seconds=5),
        identity=AgentIdentity.generate(),
        access_token="agent.jwt.token",
        transport=httpx.MockTransport(handler),
        verify_cloud_signatures=False,
    )
    # Mark version as negotiated for test purposes (Design Doc 10.3)
    cloud_client._version_negotiated = True
    cloud_client._negotiated_version = "1.0.0"

    daemon = AgentDaemon(
        DaemonConfig(
            data_dir=tmp_path,
            require_preflight=False,
            enable_telemetry=False,
            cloud_endpoint="https://cloud.example",
            cloud_access_token="agent.jwt.token",
        )
    )
    daemon.initialize()

    # Local policy: auto-approve REQUEST_START_RUN (test-only)
    daemon._approval_manager.add_auto_approve_command("REQUEST_START_RUN")

    # Inject mock client and process polled commands
    daemon._cloud_client = cloud_client
    polled = cloud_client.poll_commands(limit=10)
    daemon._process_cloud_commands(polled.commands)

    assert captured["approval"] is not None
    assert captured["approval"]["approved"] is True
    assert captured["approval"]["evidence_hash"].startswith("sha256:")
    assert captured["approval"]["diff_summary"]["payload_ref"] == "sha256:abc"


def test_manual_local_approval_submission(tmp_path: Path):
    cmd_id = uuid4()
    captured = {"approval": None}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(f"/commands/{cmd_id}/approval"):
            captured["approval"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={"command_id": str(cmd_id), "status": "approved", "approved": True},
                request=request,
            )
        if request.url.path.startswith("/api/v1/agent/commands/poll"):
            return httpx.Response(
                200,
                json={
                    "commands": [
                        {
                            "id": str(cmd_id),
                            "status": "pending_approval",
                            "idempotency_key": "idem-3",
                            "command_type": "REQUEST_UPDATE_CONFIG",
                            "payload_ref": "sha256:def",
                            "change_class": "trading_impacting",
                            "requires_approval": True,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                    ],
                    "has_more": False,
                    "poll_again_after_sec": 5,
                },
                request=request,
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    cloud_client = CloudClient(
        CloudClientConfig(base_url="https://cloud.example", timeout_seconds=5),
        identity=AgentIdentity.generate(),
        access_token="agent.jwt.token",
        transport=httpx.MockTransport(handler),
        verify_cloud_signatures=False,
    )
    # Mark version as negotiated for test purposes (Design Doc 10.3)
    cloud_client._version_negotiated = True
    cloud_client._negotiated_version = "1.0.0"

    daemon = AgentDaemon(
        DaemonConfig(
            data_dir=tmp_path,
            require_preflight=False,
            enable_telemetry=False,
            cloud_endpoint="https://cloud.example",
            cloud_access_token="agent.jwt.token",
        )
    )
    daemon.initialize()
    daemon._cloud_client = cloud_client

    polled = cloud_client.poll_commands(limit=10)
    daemon._process_cloud_commands(polled.commands)
    assert captured["approval"] is None  # no auto-approve

    ok = daemon.decide_cloud_command_approval(str(cmd_id), approved=True, reason="Operator approved")
    assert ok is True
    assert captured["approval"] is not None
    assert captured["approval"]["approved"] is True
    assert captured["approval"]["evidence_hash"].startswith("sha256:")


def test_cloud_cannot_bypass_trading_impacting_approval(tmp_path: Path):
    cmd_id = uuid4()
    captured = {"ack": 0, "result": None}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(f"/commands/{cmd_id}/ack"):
            captured["ack"] += 1
            return httpx.Response(
                200,
                json={
                    "command_id": str(cmd_id),
                    "status": "acknowledged",
                    "acknowledged_at": datetime.now(timezone.utc).isoformat(),
                },
                request=request,
            )
        if request.url.path.endswith(f"/commands/{cmd_id}/result"):
            captured["result"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={"command_id": str(cmd_id), "status": "failed", "executed_at": None},
                request=request,
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    cloud_client = CloudClient(
        CloudClientConfig(base_url="https://cloud.example", timeout_seconds=5),
        identity=AgentIdentity.generate(),
        access_token="agent.jwt.token",
        transport=httpx.MockTransport(handler),
        verify_cloud_signatures=False,
    )
    # Mark version as negotiated for test purposes (Design Doc 10.3)
    cloud_client._version_negotiated = True
    cloud_client._negotiated_version = "1.0.0"

    daemon = AgentDaemon(
        DaemonConfig(
            data_dir=tmp_path,
            require_preflight=False,
            enable_telemetry=False,
            cloud_endpoint="https://cloud.example",
            cloud_access_token="agent.jwt.token",
        )
    )
    daemon.initialize()
    daemon._cloud_client = cloud_client

    # Cloud attempts to bypass: TRADING_IMPACTING but requires_approval=False.
    cmd = PendingCommand(
        id=cmd_id,
        status="approved",
        idempotency_key="idem-2",
        command_type="REQUEST_START_RUN",
        payload_ref="sha256:abc",
        change_class="trading_impacting",
        requires_approval=False,
        created_at=datetime.now(timezone.utc),
    )

    daemon._process_cloud_commands([cmd])

    assert captured["ack"] == 1
    assert captured["result"] is not None
    assert captured["result"]["success"] is False
    assert "Refused" in (captured["result"]["error_message"] or "")
