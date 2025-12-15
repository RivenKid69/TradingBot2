# -*- coding: utf-8 -*-
"""
CCEA Cloud Control Plane Services.

Phase 2 + Phase 7: Unified service layer for:
- Command dispatch and lifecycle management (DB-backed)
- Agent lifecycle operations (poll/ack/result)
- Governance services (DSAR, Retention, Residency, Break-Glass)
- Auto-purge scheduler

CLOUD ZONE ONLY.
"""

from .command_service import (
    CommandService,
    CommandNotFoundError,
    CommandStateError,
    DuplicateCommandError,
)
from .agent_service import (
    AgentService,
    AgentNotFoundError,
    AgentNotEnrolledError,
)
from .governance_service import (
    # Enums
    DSARRequestType,
    DSARStatus,
    RetentionAction,
    BreakGlassReason,
    BreakGlassScope,
    DataRegion,
    ResidencyMode,
    # Data classes
    DSARRequestData,
    ResidencyPolicyData,
    BreakGlassRequestData,
    PurgeResult,
    # Services
    DSARService,
    ResidencyPolicyService,
    RetentionPolicyService,
    BreakGlassService,
    # Constants
    EU_COUNTRIES,
    EU_REGIONS,
    DEFAULT_RETENTION_DAYS,
)
from .purge_scheduler import (
    PurgeJobStatus,
    PurgeDataType,
    PurgeJobResult,
    PurgeSchedulerConfig,
    PurgeSchedulerState,
    PurgeScheduler,
    DSARDeletionWorkflow,
)

__all__ = [
    # Command Service
    "CommandService",
    "CommandNotFoundError",
    "CommandStateError",
    "DuplicateCommandError",
    # Agent Service
    "AgentService",
    "AgentNotFoundError",
    "AgentNotEnrolledError",
    # Governance Enums
    "DSARRequestType",
    "DSARStatus",
    "RetentionAction",
    "BreakGlassReason",
    "BreakGlassScope",
    "DataRegion",
    "ResidencyMode",
    # Governance Data Classes
    "DSARRequestData",
    "ResidencyPolicyData",
    "BreakGlassRequestData",
    "PurgeResult",
    # Governance Services
    "DSARService",
    "ResidencyPolicyService",
    "RetentionPolicyService",
    "BreakGlassService",
    # Governance Constants
    "EU_COUNTRIES",
    "EU_REGIONS",
    "DEFAULT_RETENTION_DAYS",
    # Purge Scheduler
    "PurgeJobStatus",
    "PurgeDataType",
    "PurgeJobResult",
    "PurgeSchedulerConfig",
    "PurgeSchedulerState",
    "PurgeScheduler",
    "DSARDeletionWorkflow",
]
