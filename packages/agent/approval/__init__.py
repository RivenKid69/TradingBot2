# -*- coding: utf-8 -*-
"""
CCEA Agent Approval Module.

Local approval workflow for TRADING_IMPACTING changes.
Cloud CANNOT auto-approve trading changes - must be approved locally.

Key Features:
- Local approval UI/CLI
- Evidence hash recording
- Approval history
- Auto-approve policies (local-controlled)
"""

from typing import Final

ZONE: Final[str] = "agent"

from .manager import (
    ApprovalManager,
    ApprovalRequest,
    ApprovalDecision,
    ApprovalStatus,
)

from .evidence import (
    EvidenceRecord,
    compute_evidence_hash,
)

from .cli import (
    ApprovalCLI,
    main as approval_cli_main,
    create_cli_parser,
)

from .persistence import (
    ApprovalPersistence,
    ApprovalPersistenceConfig,
)

from . import metrics as approval_metrics

__all__ = [
    "ApprovalManager",
    "ApprovalRequest",
    "ApprovalDecision",
    "ApprovalStatus",
    "EvidenceRecord",
    "compute_evidence_hash",
    "ApprovalCLI",
    "approval_cli_main",
    "create_cli_parser",
    "ApprovalPersistence",
    "ApprovalPersistenceConfig",
    "approval_metrics",
]
