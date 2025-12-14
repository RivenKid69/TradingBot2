# -*- coding: utf-8 -*-
"""
CCEA Cloud Control Plane.

.. deprecated:: 2.0.0
    This module (`ccea.control_plane`) is a DEPRECATED compatibility shim.

    **CANONICAL IMPLEMENTATION**: Use `packages.cloud.control_plane` instead.

    Migration::

        # Old (deprecated)
        from ccea.control_plane import EnrollmentService, CommandService

        # New (canonical)
        from packages.cloud.control_plane.services import enrollment_service
        from packages.cloud.control_plane.commands import CommandDispatcher

    The `packages.cloud.control_plane` module is the canonical Cloud Control
    Plane implementation for CCEA. This shim will be removed in a future version.

Provides:
- Agent enrollment with TTL tokens
- Heartbeat monitoring
- Command dispatch (long-poll)
- Immutable blob storage
- Approval tracking

Security (Design Doc):
- Cloud NEVER stores broker API keys
- Cloud NEVER generates or sends orders
- Commands are lifecycle-only (REQUEST_*)

WI-DEDRIFT-01: This module is marked for deprecation.
Canonical stack: packages/cloud/control_plane/*
"""

import warnings

warnings.warn(
    "ccea.control_plane is deprecated. Use packages.cloud.control_plane instead. "
    "See packages/cloud/control_plane/ for canonical implementation.",
    DeprecationWarning,
    stacklevel=2
)

from ccea.control_plane.enrollment import (
    EnrollmentService,
    EnrollmentError,
    TokenExpiredError,
    TokenInvalidError,
)

from ccea.control_plane.commands import (
    CommandService,
    CommandDispatcher,
    CommandQueue,
)

from ccea.control_plane.blobs import (
    BlobStore,
    BlobNotFoundError,
)

from ccea.control_plane.heartbeat import (
    HeartbeatService,
    AgentStatus,
)

__all__ = [
    # Enrollment
    "EnrollmentService",
    "EnrollmentError",
    "TokenExpiredError",
    "TokenInvalidError",
    # Commands
    "CommandService",
    "CommandDispatcher",
    "CommandQueue",
    # Blobs
    "BlobStore",
    "BlobNotFoundError",
    # Heartbeat
    "HeartbeatService",
    "AgentStatus",
]
