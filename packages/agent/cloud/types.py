# -*- coding: utf-8 -*-
"""
Types for the Agent<->Cloud lifecycle client.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID


@dataclass(frozen=True)
class AgentEnrollResult:
    agent_id: UUID
    access_token: str
    workspace_id: UUID
    org_id: UUID


@dataclass(frozen=True)
class AgentHeartbeatResult:
    server_time: datetime
    trust_state: str
    pending_commands: int
    next_heartbeat_sec: int


@dataclass(frozen=True)
class PendingCommand:
    id: UUID
    status: str
    idempotency_key: str
    command_type: str
    payload_ref: str
    change_class: str
    requires_approval: bool
    deployment_id: Optional[UUID] = None
    run_id: Optional[UUID] = None
    expires_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PendingCommand":
        return cls(
            id=UUID(str(data["id"])),
            status=str(data.get("status", "")),
            idempotency_key=str(data["idempotency_key"]),
            command_type=str(data["command_type"]),
            payload_ref=str(data["payload_ref"]),
            change_class=str(data["change_class"]),
            requires_approval=bool(data.get("requires_approval", False)),
            deployment_id=UUID(str(data["deployment_id"])) if data.get("deployment_id") else None,
            run_id=UUID(str(data["run_id"])) if data.get("run_id") else None,
            expires_at=(
                datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
            ),
            created_at=(
                datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
            ),
        )


@dataclass(frozen=True)
class CommandAckResult:
    command_id: UUID
    status: str
    acknowledged_at: datetime


@dataclass(frozen=True)
class CommandResultAck:
    command_id: UUID
    status: str
    executed_at: Optional[datetime] = None


@dataclass(frozen=True)
class CommandPollResult:
    commands: List[PendingCommand]
    has_more: bool
    poll_again_after_sec: int
