# -*- coding: utf-8 -*-
"""
Agent Cloud Client (AGENT ZONE ONLY).

Outbound-only client for Cloud Control Plane lifecycle:
- enroll
- heartbeat
- poll/ack commands
- submit local approvals
- submit command results

Security Features (Design Doc 10.2):
- All outbound messages signed by Agent
- All commands from Cloud verified by Agent
- Fail-closed on signature verification failure

This package MUST NOT be imported by Cloud zone code.
"""

from .client import (
    CloudClient,
    CloudClientConfig,
    CloudClientError,
)

from .signature_verifier import (
    CloudSignatureVerifier,
    CloudSignatureVerifierConfig,
    SignatureVerificationError,
    VerificationResult,
    VerifiedCloudMessage,
    create_verifier_from_enrollment,
)

from .types import (
    AgentEnrollResult,
    AgentHeartbeatResult,
    PendingCommand,
)

__all__ = [
    # Client
    "CloudClient",
    "CloudClientConfig",
    "CloudClientError",
    # Signature Verification
    "CloudSignatureVerifier",
    "CloudSignatureVerifierConfig",
    "SignatureVerificationError",
    "VerificationResult",
    "VerifiedCloudMessage",
    "create_verifier_from_enrollment",
    # Types
    "AgentEnrollResult",
    "AgentHeartbeatResult",
    "PendingCommand",
]

