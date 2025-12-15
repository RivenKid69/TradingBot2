# -*- coding: utf-8 -*-
"""
Deterministic Idempotency Key Generator.

Phase 3 (3.1A): Commands must have deterministic idempotency keys based on
their semantic content, not random tokens.

Design Doc:
- Deterministic key: deployment_id:command_type:artifact_digest:config_digest
- Ensures "same desired state" doesn't create duplicate commands
- Safe retry behavior for Cloud→Agent communication

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Dict, Final, FrozenSet, Optional
from uuid import UUID


# Command types that reference artifacts
ARTIFACT_COMMANDS: Final[FrozenSet[str]] = frozenset([
    "REQUEST_START_RUN",
    "REQUEST_UPGRADE_ARTIFACT",
])

# Command types that reference config blobs
CONFIG_COMMANDS: Final[FrozenSet[str]] = frozenset([
    "REQUEST_UPDATE_CONFIG",
    "REQUEST_START_RUN",
])

# Commands that are purely operational (no artifact/config)
OPERATIONAL_COMMANDS: Final[FrozenSet[str]] = frozenset([
    "REQUEST_STOP_RUN",
    "REQUEST_PAUSE_RUN",
    "REQUEST_ROTATE_AGENT_SESSION",
    "REQUEST_EXPORT_LOGS",
])


def generate_deterministic_idempotency_key(
    workspace_id: UUID,
    agent_id: UUID,
    command_type: str,
    deployment_id: Optional[UUID] = None,
    run_id: Optional[UUID] = None,
    artifact_digest: Optional[str] = None,
    config_digest: Optional[str] = None,
    payload_ref: Optional[str] = None,
) -> str:
    """
    Generate a deterministic idempotency key for a command.

    Phase 3 (3.1A): The key is computed from the semantic content of the command,
    so that issuing "the same desired state" twice produces the same key.

    Key format:
        sha256(workspace_id:agent_id:command_type:deployment_id:artifact_digest:config_digest)

    For operational commands (stop/pause), the key includes run_id to ensure
    idempotency within the context of a specific run.

    Args:
        workspace_id: Workspace identifier
        agent_id: Target agent identifier
        command_type: Type of command
        deployment_id: Optional deployment identifier
        run_id: Optional run identifier (for operational commands)
        artifact_digest: Optional artifact digest (for artifact commands)
        config_digest: Optional config blob digest (for config commands)
        payload_ref: Optional payload reference (fallback)

    Returns:
        Deterministic idempotency key (24 char base64url)
    """
    import base64

    # Normalize command type
    cmd_type = command_type.upper().strip()

    # Build key components based on command type
    components = [
        str(workspace_id),
        str(agent_id),
        cmd_type,
    ]

    if cmd_type in ARTIFACT_COMMANDS:
        # For artifact commands: include deployment + artifact digest
        components.append(str(deployment_id) if deployment_id else "")
        components.append(artifact_digest or "")
        if cmd_type == "REQUEST_START_RUN":
            # Start run also depends on config
            components.append(config_digest or "")

    elif cmd_type in CONFIG_COMMANDS:
        # For config commands: include deployment + config digest
        components.append(str(deployment_id) if deployment_id else "")
        components.append(config_digest or "")

    elif cmd_type in OPERATIONAL_COMMANDS:
        # For operational commands: include run_id
        components.append(str(run_id) if run_id else "")
        # For stop/pause, the same command to the same run should be idempotent
        if cmd_type == "REQUEST_ROTATE_AGENT_SESSION":
            # Session rotation needs timestamp bucket to allow new rotations
            # Use hour-level granularity for rate limiting
            components.append(datetime.utcnow().strftime("%Y%m%d%H"))

    else:
        # Unknown command type - use payload_ref as fallback
        if payload_ref:
            components.append(payload_ref)

    # Compute deterministic hash
    key_source = ":".join(components)
    hash_bytes = hashlib.sha256(key_source.encode("utf-8")).digest()

    # Return first 18 bytes as base64url (24 chars)
    return base64.urlsafe_b64encode(hash_bytes[:18]).decode("ascii")


def compute_config_digest(config: Dict[str, Any]) -> str:
    """
    Compute digest of config blob content.

    Used for deterministic idempotency key generation.

    Args:
        config: Configuration dictionary

    Returns:
        SHA256 digest with prefix
    """
    import json

    content = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def validate_idempotency_key_determinism(
    key1: str,
    key2: str,
    command_type: str,
    context: str = "",
) -> bool:
    """
    Validate that two idempotency keys are the same (deterministic).

    Useful for testing/debugging.

    Args:
        key1: First key
        key2: Second key
        command_type: Command type for context
        context: Additional context for error message

    Returns:
        True if keys match

    Raises:
        AssertionError if keys don't match
    """
    if key1 != key2:
        raise AssertionError(
            f"Idempotency keys not deterministic for {command_type}: "
            f"'{key1}' != '{key2}'. {context}"
        )
    return True


class IdempotencyKeyBuilder:
    """
    Builder pattern for constructing deterministic idempotency keys.

    Usage:
        key = (
            IdempotencyKeyBuilder()
            .workspace(workspace_id)
            .agent(agent_id)
            .command("REQUEST_START_RUN")
            .deployment(deployment_id)
            .artifact(artifact_digest)
            .config(config_digest)
            .build()
        )
    """

    def __init__(self):
        self._workspace_id: Optional[UUID] = None
        self._agent_id: Optional[UUID] = None
        self._command_type: Optional[str] = None
        self._deployment_id: Optional[UUID] = None
        self._run_id: Optional[UUID] = None
        self._artifact_digest: Optional[str] = None
        self._config_digest: Optional[str] = None
        self._payload_ref: Optional[str] = None

    def workspace(self, workspace_id: UUID) -> "IdempotencyKeyBuilder":
        """Set workspace ID."""
        self._workspace_id = workspace_id
        return self

    def agent(self, agent_id: UUID) -> "IdempotencyKeyBuilder":
        """Set agent ID."""
        self._agent_id = agent_id
        return self

    def command(self, command_type: str) -> "IdempotencyKeyBuilder":
        """Set command type."""
        self._command_type = command_type
        return self

    def deployment(self, deployment_id: UUID) -> "IdempotencyKeyBuilder":
        """Set deployment ID."""
        self._deployment_id = deployment_id
        return self

    def run(self, run_id: UUID) -> "IdempotencyKeyBuilder":
        """Set run ID."""
        self._run_id = run_id
        return self

    def artifact(self, artifact_digest: str) -> "IdempotencyKeyBuilder":
        """Set artifact digest."""
        self._artifact_digest = artifact_digest
        return self

    def config(self, config_digest: str) -> "IdempotencyKeyBuilder":
        """Set config digest."""
        self._config_digest = config_digest
        return self

    def payload(self, payload_ref: str) -> "IdempotencyKeyBuilder":
        """Set payload reference."""
        self._payload_ref = payload_ref
        return self

    def build(self) -> str:
        """Build the deterministic idempotency key."""
        if not self._workspace_id:
            raise ValueError("workspace_id is required")
        if not self._agent_id:
            raise ValueError("agent_id is required")
        if not self._command_type:
            raise ValueError("command_type is required")

        return generate_deterministic_idempotency_key(
            workspace_id=self._workspace_id,
            agent_id=self._agent_id,
            command_type=self._command_type,
            deployment_id=self._deployment_id,
            run_id=self._run_id,
            artifact_digest=self._artifact_digest,
            config_digest=self._config_digest,
            payload_ref=self._payload_ref,
        )
