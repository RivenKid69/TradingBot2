# -*- coding: utf-8 -*-
"""
CCEA Cloud Control Plane.

Manages agent lifecycle through lifecycle requests.

Key Components:
- CommandDispatcher: Sends lifecycle commands to agents
- TelemetryIngester: Receives and stores telemetry from agents
- EnrollmentService: Agent enrollment and token management
- DeploymentManager: Strategy deployment management

IMPORTANT: Control Plane sends only LIFECYCLE requests.
It NEVER sends order-like payloads or trading instructions.

Allowed Commands:
- REQUEST_START_RUN
- REQUEST_STOP_RUN
- REQUEST_PAUSE_RUN
- REQUEST_UPGRADE_ARTIFACT
- REQUEST_UPDATE_CONFIG
- REQUEST_ROTATE_AGENT_SESSION
- REQUEST_EXPORT_LOGS

Prohibited:
- Any command containing side/qty/price/target_position
- Any command that could be interpreted as trading instruction
"""

from typing import Final, List

ZONE: Final[str] = "cloud"

from .commands import (
    CommandDispatcher,
    Command,
    CommandType,
    CommandStatus,
    CommandResult,
)

from .telemetry_ingester import (
    TelemetryIngester,
    TelemetryStorage,
)

__all__ = [
    "CommandDispatcher",
    "Command",
    "CommandType",
    "CommandStatus",
    "CommandResult",
    "TelemetryIngester",
    "TelemetryStorage",
]
