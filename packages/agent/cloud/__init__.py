# -*- coding: utf-8 -*-
"""
Agent Cloud Client (AGENT ZONE ONLY).

Outbound-only client for Cloud Control Plane lifecycle:
- enroll
- heartbeat
- poll/ack commands
- submit local approvals
- submit command results

This package MUST NOT be imported by Cloud zone code.
"""

from .client import (
    CloudClient,
    CloudClientConfig,
    CloudClientError,
)

from .types import (
    AgentEnrollResult,
    AgentHeartbeatResult,
    PendingCommand,
)

__all__ = [
    "CloudClient",
    "CloudClientConfig",
    "CloudClientError",
    "AgentEnrollResult",
    "AgentHeartbeatResult",
    "PendingCommand",
]

