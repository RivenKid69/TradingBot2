# -*- coding: utf-8 -*-
"""
CCEA Agent Reconciliation Module.

Handles position reconciliation and order journal.
Ensures consistency between local state and broker state.

Key Features:
- Position sync with broker
- Order journal for crash recovery
- Idempotent order handling
- Duplicate prevention
"""

from typing import Final

ZONE: Final[str] = "agent"

from .journal import (
    OrderJournal,
    JournalEntry,
    JournalStatus,
)

from .reconciler import (
    PositionReconciler,
    ReconciliationResult,
    PositionMismatch,
)

__all__ = [
    "OrderJournal",
    "JournalEntry",
    "JournalStatus",
    "PositionReconciler",
    "ReconciliationResult",
    "PositionMismatch",
]
