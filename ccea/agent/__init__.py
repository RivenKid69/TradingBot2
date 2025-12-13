# -*- coding: utf-8 -*-
"""
CCEA Agent Daemon.

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
"""

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
