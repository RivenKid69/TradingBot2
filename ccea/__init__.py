# -*- coding: utf-8 -*-
"""
CCEA (Cloud-Controlled Execution Architecture) Module.

Phase 1 Implementation - Skeleton E2E + Basic Guardrails

This module provides:
- Cloud Control Plane: Enrollment, heartbeat, commands, blobs
- Agent Daemon: Enrollment, command handling, approval workflow
- Artifact Builder: Package, manifest, signing, registry
- Telemetry: Collection with mandatory redaction
- Guardrails: CI/CD enforcement, schema validation, protocol validation

Key Principle (non-negotiable):
    Cloud = research/build/monitoring/control plane (lifecycle requests)
    Agent = secrets + live loop + risk enforce + order creation/sending

Cloud NEVER:
    - Stores broker API keys
    - Generates or sends orders
    - Has access to trading endpoints
    - Contains order-like payloads in protocol

Security Guarantees:
    - All messages authenticated/signed
    - Local approval for TRADING_IMPACTING
    - Mandatory telemetry redaction
    - Artifacts signed and digest-pinned
"""

__version__ = "1.0.0"
__author__ = "CCEA Architecture Team"

from typing import Final

# Schema version for this release (must match protocol_messages.schema.json x-schema-version)
SCHEMA_VERSION: Final[str] = "1.0.0"

# Minimum supported schema version (must match protocol_messages.schema.json x-min-supported-version)
MIN_SUPPORTED_SCHEMA_VERSION: Final[str] = "1.0.0"

# Maximum supported schema version (must match protocol_messages.schema.json x-max-supported-version)
MAX_SUPPORTED_SCHEMA_VERSION: Final[str] = "1.0.0"

# Phase 1 implementation flag
PHASE_1_COMPLETE: Final[bool] = True
