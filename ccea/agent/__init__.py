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

import warnings

warnings.warn(
    "ccea.agent is deprecated. Use packages.agent instead. "
    "See packages/agent/daemon/agentd.py for canonical implementation.",
    DeprecationWarning,
    stacklevel=2
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
