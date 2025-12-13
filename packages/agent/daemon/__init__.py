# -*- coding: utf-8 -*-
"""
CCEA Agent Daemon Package.

Contains the core daemon (agentd) and supporting components:
- AgentDaemon: Main daemon process with lifecycle management
- Sandbox: Strategy isolation with process/container support
- PreflightChecker: Validation before start/upgrade
- DegradedModeManager: Safe degradation handling
- TelemetryBuffer: Durable telemetry storage
- TimeSyncChecker: Time synchronization verification
- KeychainManager: OS keychain integration for master key

Phase 5 of CCEA Cloud Alignment Plan.
"""

from typing import Final, List

__version__ = "1.0.0"

# Components provided by this package
DAEMON_COMPONENTS: Final[List[str]] = [
    "AgentDaemon",
    "DaemonConfig",
    "DaemonState",
    "Sandbox",
    "SandboxConfig",
    "PreflightChecker",
    "PreflightResult",
    "DegradedModeManager",
    "DegradedMode",
    "TelemetryBuffer",
    "TelemetryEvent",
    "TimeSyncChecker",
    "KeychainManager",
    "HaltReason",
    "KillSwitchManager",
]
