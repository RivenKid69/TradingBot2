# -*- coding: utf-8 -*-
"""
CCEA Agent Daemon.

.. deprecated:: 2.0.0
    This module (`ccea.agent`) is a DEPRECATED compatibility shim.

    **CANONICAL IMPLEMENTATION**: Use `packages.agent` instead.

    Migration::

        # Old (deprecated)
        from ccea.agent import AgentDaemon, AgentConfig

        # New (canonical)
        from packages.agent.daemon.agentd import AgentDaemon
        from packages.agent.daemon.config import AgentConfig

    The `packages.agent` module is the canonical Agent implementation for CCEA.
    This shim will be removed in a future version.

Provides:
- Device keypair generation
- Cloud enrollment
- Command polling (outbound-only)
- Local approval workflow
- Artifact verification and execution
- Telemetry with mandatory redaction

Security (Design Doc):
- Agent holds ALL secrets locally
- Keypair generated locally, only public key shared
- Commands are pulled (outbound-only), never pushed
- Local approval required for TRADING_IMPACTING commands
- Policy firewall enforces hard caps

WI-DEDRIFT-01: This module is marked for deprecation.
Canonical stack: packages/agent/*
"""

from __future__ import annotations

import os
import warnings

# ============================================================================
# CI GUARDRAIL: Block import in CI environments
# ============================================================================
# This guardrail prevents accidental use of deprecated modules in CI.
# Set CCEA_ALLOW_DEPRECATED=true to bypass (for migration testing only).

_IN_CI = os.environ.get("CI", "").lower() in ("true", "1", "yes")
_ALLOW_DEPRECATED = os.environ.get("CCEA_ALLOW_DEPRECATED", "").lower() in ("true", "1", "yes")
_STRICT_DEPRECATION = os.environ.get("CCEA_STRICT_DEPRECATION", "").lower() in ("true", "1", "yes")

if (_IN_CI or _STRICT_DEPRECATION) and not _ALLOW_DEPRECATED:
    raise ImportError(
        "\n"
        "=" * 70 + "\n"
        "DEPRECATED MODULE IMPORT BLOCKED\n"
        "=" * 70 + "\n"
        "ccea.agent is DEPRECATED and cannot be imported in CI.\n\n"
        "Migration:\n"
        "  from ccea.agent.daemon -> from packages.agent.daemon.agentd\n"
        "  from ccea.agent.approval -> from packages.agent.daemon.approval_handler\n"
        "  from ccea.agent.runner -> from packages.agent.execution.engine\n\n"
        "To bypass (migration testing only): CCEA_ALLOW_DEPRECATED=true\n"
        "=" * 70
    )

# Emit warning for non-CI environments
warnings.warn(
    "ccea.agent is deprecated. Use packages.agent instead. "
    "See packages/agent/daemon/agentd.py for canonical implementation.",
    DeprecationWarning,
    stacklevel=2,
)

from ccea.agent.daemon import (
    AgentDaemon,
    AgentConfig,
    AgentState,
)

from ccea.agent.enrollment import (
    AgentEnrollment,
    EnrollmentState,
)

from ccea.agent.command_handler import (
    CommandHandler,
    CommandFilter,
)

from ccea.agent.approval import (
    ApprovalManager,
    ApprovalRequest,
    ApprovalResult,
)

from ccea.agent.runner import (
    ArtifactRunner,
    RunContext,
    RunResult,
)

__all__ = [
    # Daemon
    "AgentDaemon",
    "AgentConfig",
    "AgentState",
    # Enrollment
    "AgentEnrollment",
    "EnrollmentState",
    # Command Handler
    "CommandHandler",
    "CommandFilter",
    # Approval
    "ApprovalManager",
    "ApprovalRequest",
    "ApprovalResult",
    # Runner
    "ArtifactRunner",
    "RunContext",
    "RunResult",
]
