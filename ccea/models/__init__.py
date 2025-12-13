# -*- coding: utf-8 -*-
"""
CCEA Protocol Models.

Pydantic models for CCEA Cloud-Agent protocol messages.
All models enforce schema constraints from Design Doc.
"""

from ccea.models.protocol import (
    # Base types
    Signature,
    IdempotencyKey,
    Digest,
    AgentId,
    # States
    CommandStatus,
    ApprovalStatus,
    RunState,
    DeploymentState,
    ChangeClass,
    TelemetryLevel,
    # Messages
    HeartbeatMessage,
    PollCommandsMessage,
    CommandBatchMessage,
    CommandAckMessage,
    CommandApprovalMessage,
    CommandResultMessage,
    TelemetryMessage,
    # Commands
    RequestStartRunCommand,
    RequestStopRunCommand,
    RequestPauseRunCommand,
    RequestUpgradeArtifactCommand,
    RequestUpdateConfigCommand,
    RequestRotateAgentSessionCommand,
    RequestExportLogsCommand,
    # Union types
    Command,
    Message,
)

from ccea.models.manifest import (
    ArtifactManifest,
    Entrypoint,
    Runtime,
    ModelRef,
    DataContract,
    Permissions,
    RiskProfileSuggested,
    LiveCapabilities,
    Provenance,
    ManifestSignature,
)

from ccea.models.enrollment import (
    EnrollmentToken,
    EnrollmentRequest,
    EnrollmentResponse,
    AgentInfo,
)

__all__ = [
    # Protocol types
    "Signature",
    "IdempotencyKey",
    "Digest",
    "AgentId",
    "CommandStatus",
    "ApprovalStatus",
    "RunState",
    "DeploymentState",
    "ChangeClass",
    "TelemetryLevel",
    # Messages
    "HeartbeatMessage",
    "PollCommandsMessage",
    "CommandBatchMessage",
    "CommandAckMessage",
    "CommandApprovalMessage",
    "CommandResultMessage",
    "TelemetryMessage",
    # Commands
    "RequestStartRunCommand",
    "RequestStopRunCommand",
    "RequestPauseRunCommand",
    "RequestUpgradeArtifactCommand",
    "RequestUpdateConfigCommand",
    "RequestRotateAgentSessionCommand",
    "RequestExportLogsCommand",
    "Command",
    "Message",
    # Manifest
    "ArtifactManifest",
    "Entrypoint",
    "Runtime",
    "ModelRef",
    "DataContract",
    "Permissions",
    "RiskProfileSuggested",
    "LiveCapabilities",
    "Provenance",
    "ManifestSignature",
    # Enrollment
    "EnrollmentToken",
    "EnrollmentRequest",
    "EnrollmentResponse",
    "AgentInfo",
]
