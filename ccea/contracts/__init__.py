# -*- coding: utf-8 -*-
"""
CCEA Shared Contracts Module.

Single source of truth for enum values, state machines, and command types
across protocol models, cloud DB, and agent code.

This module:
1. Defines canonical enum values (derived from protocol_messages.schema.json)
2. Provides mapping between protocol and DB representations
3. Exports validation functions for contract testing

Design Doc Reference:
    - All command/message types must be consistent across:
      - docs/schemas/protocol_messages.schema.json (authoritative)
      - ccea/models/protocol.py
      - packages/cloud/control_plane/models.py
      - packages/cloud/control_plane/commands.py

Usage:
    from ccea.contracts import (
        COMMAND_TYPES,
        COMMAND_STATUS,
        DEPLOYMENT_STATE,
        RUN_STATE,
        validate_contract_consistency,
    )
"""

from ccea.contracts.enums import (
    # Command Types
    COMMAND_TYPES,
    CommandTypeContract,
    # Command Status
    COMMAND_STATUS_PROTOCOL,
    COMMAND_STATUS_DB,
    CommandStatusMapping,
    # Deployment State
    DEPLOYMENT_STATE_PROTOCOL,
    DEPLOYMENT_STATE_DB,
    DeploymentStateMapping,
    # Run State
    RUN_STATE_PROTOCOL,
    RUN_STATE_DB,
    RunStateMapping,
    # Message Types
    MESSAGE_TYPES,
    # Approval Status
    APPROVAL_STATUS,
    # Change Class
    CHANGE_CLASS,
    # Telemetry Level
    TELEMETRY_LEVEL_PROTOCOL,
    TELEMETRY_LEVEL_DB,
)

from ccea.contracts.validation import (
    validate_contract_consistency,
    ContractValidationResult,
    extract_schema_enums,
    compare_enum_sets,
)

__all__ = [
    # Command Types
    "COMMAND_TYPES",
    "CommandTypeContract",
    # Command Status
    "COMMAND_STATUS_PROTOCOL",
    "COMMAND_STATUS_DB",
    "CommandStatusMapping",
    # Deployment State
    "DEPLOYMENT_STATE_PROTOCOL",
    "DEPLOYMENT_STATE_DB",
    "DeploymentStateMapping",
    # Run State
    "RUN_STATE_PROTOCOL",
    "RUN_STATE_DB",
    "RunStateMapping",
    # Message Types
    "MESSAGE_TYPES",
    # Approval Status
    "APPROVAL_STATUS",
    # Change Class
    "CHANGE_CLASS",
    # Telemetry Level
    "TELEMETRY_LEVEL_PROTOCOL",
    "TELEMETRY_LEVEL_DB",
    # Validation
    "validate_contract_consistency",
    "ContractValidationResult",
    "extract_schema_enums",
    "compare_enum_sets",
]
