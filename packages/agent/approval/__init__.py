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

__all__ = [
    "ApprovalManager",
    "ApprovalRequest",
    "ApprovalDecision",
    "ApprovalStatus",
    "EvidenceRecord",
    "compute_evidence_hash",
]
