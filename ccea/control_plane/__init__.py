# -*- coding: utf-8 -*-
"""
CCEA Cloud Control Plane.

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
"""

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
