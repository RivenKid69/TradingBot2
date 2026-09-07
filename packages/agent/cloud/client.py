# -*- coding: utf-8 -*-
"""
CloudClient - Outbound-only lifecycle client for Agent.

Implements the minimal Agent-side lifecycle per CCEA Master Remediation Plan Phase 8:
- Enroll via `/api/v1/auth/agent/enroll`
- Heartbeat via `/api/v1/agent/heartbeat`
- Poll commands via `/api/v1/agent/commands/poll`
- Ack commands via `/api/v1/agent/commands/{id}/ack`
- Submit local approvals via `/api/v1/agent/commands/{id}/approval`
- Submit results via `/api/v1/agent/commands/{id}/result`
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID

import httpx

from ccea.crypto.keys import KeyAlgorithm, generate_keypair, load_private_key
from ccea.crypto.signing import sign_message

from .signature_verifier import (
    CloudSignatureVerifier,
    CloudSignatureVerifierConfig,
    SignatureVerificationError,
)
from .types import (
    AgentEnrollResult,
    AgentHeartbeatResult,
    CommandAckResult,
    CommandPollResult,
    CommandResultAck,
    PendingCommand,
)


class CloudClientError(RuntimeError):
    """Cloud client error."""


@dataclass
class CloudClientConfig:
    base_url: str
    timeout_seconds: int = 30
    user_agent: str = "ccea-agent-cloud-client/1.0"


@dataclass
class AgentIdentity:
    """
    Minimal agent identity for enrollment and signing.

    NOTE: Private key stays local; Cloud receives only public key at enrollment.
    """

    public_key_pem: str
    private_key_pem: str
    algorithm: str = KeyAlgorithm.ED25519.value

    @classmethod
    def generate(cls) -> "AgentIdentity":
        keypair = generate_keypair(KeyAlgorithm.ED25519)
        return cls(
            public_key_pem=keypair.get_public_key_pem(),
            private_key_pem=keypair.get_private_key_pem(),
            algorithm=keypair.algorithm.value,
        )

    def private_key(self):
        return load_private_key(self.private_key_pem)

    def public_key_fingerprint(self) -> str:
        return hashlib.sha256(self.public_key_pem.encode("utf-8")).hexdigest()


def _canonical_json_bytes(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


class CloudClient:
    """
    Synchronous client for Agent threads (heartbeat/poll loops).

    Authentication:
    - Enrollment: no auth
    - All other endpoints: Bearer `access_token`

    Integrity:
    - Adds a request signature header for tamper-evident telemetry/debugging.
      (Cloud may choose to verify using enrolled public key.)
    - Verifies Cloud signatures on incoming commands (Design Doc 10.2)
    """

    def __init__(
        self,
        config: CloudClientConfig,
        *,
        identity: Optional[AgentIdentity] = None,
        access_token: Optional[str] = None,
        transport: Optional[httpx.BaseTransport] = None,
        signature_verifier: Optional[CloudSignatureVerifier] = None,
        verify_cloud_signatures: bool = True,
    ):
        self._config = config
        self._identity = identity or AgentIdentity.generate()
        self._access_token = access_token
        self._last_ok_at: Optional[datetime] = None
        self._last_error: Optional[str] = None

        # Version negotiation (Design Doc 10.3/10.4)
        self._negotiated_version: Optional[str] = None
        self._version_negotiation_required = True

        # Cloud signature verification (Design Doc 10.2)
        self._verify_cloud_signatures = verify_cloud_signatures
        self._signature_verifier = signature_verifier or CloudSignatureVerifier(
            config=CloudSignatureVerifierConfig(
                enforce_verification=verify_cloud_signatures,
                allow_unsigned=not verify_cloud_signatures,
            )
        )

        self._client = httpx.Client(
            base_url=self._config.base_url.rstrip("/"),
            timeout=httpx.Timeout(self._config.timeout_seconds),
            transport=transport,
            headers={"User-Agent": self._config.user_agent},
        )

    @property
    def identity(self) -> AgentIdentity:
        return self._identity

    @property
    def access_token(self) -> Optional[str]:
        return self._access_token

    @access_token.setter
    def access_token(self, token: Optional[str]) -> None:
        self._access_token = token

    @property
    def is_connected(self) -> bool:
        if self._last_ok_at is None:
            return False
        # "Connected" means: last successful request within 2 heartbeat intervals.
        age = datetime.now(timezone.utc) - self._last_ok_at
        return age.total_seconds() <= max(30, self._config.timeout_seconds * 2)

    @property
    def signature_verifier(self) -> CloudSignatureVerifier:
        """Get signature verifier for Cloud messages."""
        return self._signature_verifier

    def set_cloud_public_key(self, public_key_pem: str) -> None:
        """
        Set Cloud's public key for signature verification.

        This should be called after enrollment when Cloud provides its public key.
        Design Doc 10.2: Agent MUST verify Cloud signatures.

        Args:
            public_key_pem: Cloud's public key in PEM format
        """
        self._signature_verifier.set_cloud_public_key(public_key_pem)

    def close(self) -> None:
        self._client.close()

    def _signed_headers(self, body: Optional[bytes]) -> Dict[str, str]:
        """
        Build signed headers for outbound request.

        Design Doc 10.2: bytes-to-sign = body_bytes + b"|" + timestamp_iso
        This ensures timestamp is included in signature for replay protection.
        """
        now = datetime.now(timezone.utc).isoformat()
        payload = body if body is not None else b""
        # Design Doc 10.2: Include timestamp in signed message for replay protection
        message_to_sign = payload + b"|" + now.encode("utf-8")
        signature = sign_message(message_to_sign, self._identity.private_key())
        return {
            "X-CCEA-Timestamp": now,
            "X-CCEA-Signature": signature,
            "X-CCEA-Signature-Alg": self._identity.algorithm,
            # Use consistent header name with middleware
            "X-CCEA-Key-Fingerprint": self._identity.public_key_fingerprint(),
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        auth_required: bool = True,
    ) -> httpx.Response:
        headers: Dict[str, str] = {}
        body_bytes = _canonical_json_bytes(json_body) if json_body is not None else None
        headers.update(self._signed_headers(body_bytes))

        if auth_required:
            if not self._access_token:
                raise CloudClientError("Missing access_token for authenticated request")
            headers["Authorization"] = f"Bearer {self._access_token}"

        try:
            resp = self._client.request(method, path, json=json_body, headers=headers)
            resp.raise_for_status()
            self._last_ok_at = datetime.now(timezone.utc)
            self._last_error = None
            return resp
        except Exception as e:
            self._last_error = str(e)
            raise

    def enroll(
        self,
        *,
        enrollment_token: str,
        agent_name: str,
        agent_version: str,
        capabilities: Optional[list[str]] = None,
        attestation: Optional[str] = None,
    ) -> AgentEnrollResult:
        payload = {
            "enrollment_token": enrollment_token,
            "agent_name": agent_name,
            "public_key": self._identity.public_key_pem,
            "agent_version": agent_version,
            "capabilities": capabilities or [],
            "attestation": attestation,
        }

        resp = self._request(
            "POST",
            "/api/v1/auth/agent/enroll",
            json_body=payload,
            auth_required=False,
        )
        data = resp.json()
        result = AgentEnrollResult(
            agent_id=UUID(str(data["agent_id"])),
            access_token=str(data["access_token"]),
            workspace_id=UUID(str(data["workspace_id"])),
            org_id=UUID(str(data["org_id"])),
        )
        self._access_token = result.access_token

        # Set Cloud's public key for signature verification (Design Doc 10.2)
        cloud_public_key = data.get("cloud_public_key")
        if cloud_public_key:
            self.set_cloud_public_key(cloud_public_key)

        return result

    def heartbeat(
        self,
        *,
        agent_version: str,
        current_state: str,
        last_run_id: Optional[UUID] = None,
        health_metrics: Optional[Dict[str, Any]] = None,
    ) -> AgentHeartbeatResult:
        payload: Dict[str, Any] = {
            "agent_version": agent_version,
            "current_state": current_state,
            "last_run_id": str(last_run_id) if last_run_id else None,
            "health_metrics": health_metrics or {},
        }
        resp = self._request(
            "POST", "/api/v1/agent/heartbeat", json_body=payload, auth_required=True
        )
        data = resp.json()
        return AgentHeartbeatResult(
            server_time=datetime.fromisoformat(data["server_time"]),
            trust_state=str(data["trust_state"]),
            pending_commands=int(data.get("pending_commands", 0)),
            next_heartbeat_sec=int(data.get("next_heartbeat_sec", 60)),
        )

    def negotiate_version(
        self,
        *,
        min_supported: str = "1.0.0",
        max_supported: str = "1.0.0",
        preferred: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Negotiate protocol version with Cloud.

        Design Doc 10.3/10.4: Version MUST be negotiated before commands.

        Args:
            min_supported: Agent's minimum supported version
            max_supported: Agent's maximum supported version
            preferred: Agent's preferred version

        Returns:
            Negotiation result dict

        Raises:
            CloudClientError: If negotiation fails
        """
        payload = {
            "min_supported": min_supported,
            "max_supported": max_supported,
        }
        if preferred:
            payload["preferred"] = preferred

        resp = self._request(
            "POST",
            "/api/v1/agent/negotiate-version",
            json_body=payload,
            auth_required=True,
        )
        data = resp.json()

        # Store negotiated version
        if data.get("status") == "SUCCESS" and data.get("selected_version"):
            self._negotiated_version = data["selected_version"]
            self._version_negotiation_required = False

        return data

    @property
    def negotiated_version(self) -> Optional[str]:
        """Get the negotiated protocol version."""
        return self._negotiated_version

    def _ensure_version_negotiated(self) -> None:
        """
        Ensure version has been negotiated.

        Design Doc 10.3/10.4: Version MUST be negotiated before commands.

        Raises:
            CloudClientError: If version not yet negotiated
        """
        if self._version_negotiation_required and not self._negotiated_version:
            raise CloudClientError(
                "Protocol version not negotiated. Call negotiate_version() first."
            )

    def _request_with_version(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        auth_required: bool = True,
    ) -> httpx.Response:
        """Make request with version header."""
        headers: Dict[str, str] = {}
        body_bytes = _canonical_json_bytes(json_body) if json_body is not None else None
        headers.update(self._signed_headers(body_bytes))

        # Add negotiated version header
        if self._negotiated_version:
            headers["X-Protocol-Version"] = self._negotiated_version

        if auth_required:
            if not self._access_token:
                raise CloudClientError("Missing access_token for authenticated request")
            headers["Authorization"] = f"Bearer {self._access_token}"

        try:
            resp = self._client.request(method, path, json=json_body, headers=headers)
            resp.raise_for_status()
            self._last_ok_at = datetime.now(timezone.utc)
            self._last_error = None
            return resp
        except Exception as e:
            self._last_error = str(e)
            raise

    def poll_commands(self, *, limit: int = 10) -> CommandPollResult:
        """
        Poll for pending commands from Cloud.

        Design Doc 10.2: All commands MUST be signed by Cloud.
        Agent MUST verify signatures before processing.

        Design Doc 10.3/10.4: Version MUST be negotiated before polling.

        Args:
            limit: Maximum number of commands to fetch

        Returns:
            CommandPollResult with verified commands

        Raises:
            SignatureVerificationError: If any command has invalid signature
            CloudClientError: If version not negotiated
        """
        # Design Doc 10.3/10.4: Ensure version negotiated
        self._ensure_version_negotiated()

        resp = self._request_with_version(
            "GET",
            f"/api/v1/agent/commands/poll?limit={int(limit)}",
            auth_required=True,
        )
        data = resp.json()

        # Verify signature on each command (Design Doc 10.2)
        raw_commands = data.get("commands", [])
        verified_commands = []

        for cmd_data in raw_commands:
            # Verify command signature before processing
            if self._verify_cloud_signatures:
                self._signature_verifier.verify_or_raise(cmd_data)

            verified_commands.append(PendingCommand.from_dict(cmd_data))

        return CommandPollResult(
            commands=verified_commands,
            has_more=bool(data.get("has_more", False)),
            poll_again_after_sec=int(data.get("poll_again_after_sec", 60)),
        )

    def acknowledge_command(self, command_id: UUID) -> CommandAckResult:
        """Acknowledge receipt of a command."""
        self._ensure_version_negotiated()
        resp = self._request_with_version(
            "POST",
            f"/api/v1/agent/commands/{command_id}/ack",
            auth_required=True,
        )
        data = resp.json()
        return CommandAckResult(
            command_id=UUID(str(data["command_id"])),
            status=str(data["status"]),
            acknowledged_at=datetime.fromisoformat(data["acknowledged_at"]),
        )

    def submit_local_approval(
        self,
        *,
        command_id: UUID,
        approved: bool,
        evidence_hash: Optional[str] = None,
        attestation: Optional[Dict[str, Any]] = None,
        diff_summary: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit local approval for a command."""
        self._ensure_version_negotiated()
        payload: Dict[str, Any] = {
            "approved": bool(approved),
            "evidence_hash": evidence_hash,
            "attestation": attestation,
            "diff_summary": diff_summary,
            "reason": reason,
        }
        resp = self._request_with_version(
            "POST",
            f"/api/v1/agent/commands/{command_id}/approval",
            json_body=payload,
            auth_required=True,
        )
        return resp.json()

    def submit_command_result(
        self,
        *,
        command_id: UUID,
        success: bool,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> CommandResultAck:
        """Submit command execution result."""
        self._ensure_version_negotiated()
        payload: Dict[str, Any] = {
            "success": bool(success),
            "result": result,
            "error_message": error_message,
        }
        resp = self._request_with_version(
            "POST",
            f"/api/v1/agent/commands/{command_id}/result",
            json_body=payload,
            auth_required=True,
        )
        data = resp.json()
        return CommandResultAck(
            command_id=UUID(str(data["command_id"])),
            status=str(data["status"]),
            executed_at=(
                datetime.fromisoformat(data["executed_at"]) if data.get("executed_at") else None
            ),
        )

    def send_telemetry(
        self,
        events: list[Dict[str, Any]],
    ) -> bool:
        """
        Send telemetry events to Cloud.

        Design Doc Phase 5: Agent telemetry upload to Cloud.
        Uses agent-auth endpoint /api/v1/agent/telemetry.

        Args:
            events: List of telemetry event dictionaries

        Returns:
            True if telemetry was accepted by Cloud

        Raises:
            CloudClientError: If request fails
        """
        if not events:
            return True

        payload = {
            "events": events,
            "batch_size": len(events),
        }

        try:
            resp = self._request(
                "POST",
                "/api/v1/agent/telemetry",
                json_body=payload,
                auth_required=True,
            )
            data = resp.json()
            return data.get("accepted", False)
        except Exception as e:
            self._last_error = str(e)
            return False

    def send_telemetry_batch(
        self,
        events: list[Dict[str, Any]],
        max_batch_size: int = 100,
    ) -> int:
        """
        Send telemetry events in batches.

        Design Doc Phase 5: Batch telemetry upload for efficiency.

        Args:
            events: List of telemetry event dictionaries
            max_batch_size: Maximum events per batch

        Returns:
            Number of events successfully sent
        """
        total_sent = 0

        for i in range(0, len(events), max_batch_size):
            batch = events[i : i + max_batch_size]
            if self.send_telemetry(batch):
                total_sent += len(batch)
            else:
                # Stop on first failure
                break

        return total_sent
