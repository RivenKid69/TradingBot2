# -*- coding: utf-8 -*-
"""
CCEA Guardrails Module.

Provides CI/CD guardrails for enforcing CCEA architectural boundaries:
- Import boundary checking (Cloud/Agent/Shared zones)
- Schema validation (no order-like payloads)
- Protocol allowlist validation
- Artifact signature verification
- Redaction enforcement
"""

from .import_check import (
    check_cloud_imports,
    check_agent_imports,
    get_zone_for_module,
    PROHIBITED_IN_CLOUD,
    PROHIBITED_PACKAGES,
    ZoneType,
)
from .schema_check import (
    validate_manifest_schema,
    validate_protocol_schema,
    check_prohibited_fields,
    PROHIBITED_FIELDS,
    PROHIBITED_VALUES,
)
from .protocol_check import (
    ALLOWED_COMMAND_TYPES,
    ALLOWED_MESSAGE_TYPES,
    check_protocol_changes,
    validate_command_type,
)

__all__ = [
    # Import checking
    "check_cloud_imports",
    "check_agent_imports",
    "get_zone_for_module",
    "PROHIBITED_IN_CLOUD",
    "PROHIBITED_PACKAGES",
    "ZoneType",
    # Schema validation
    "validate_manifest_schema",
    "validate_protocol_schema",
    "check_prohibited_fields",
    "PROHIBITED_FIELDS",
    "PROHIBITED_VALUES",
    # Protocol validation
    "ALLOWED_COMMAND_TYPES",
    "ALLOWED_MESSAGE_TYPES",
    "check_protocol_changes",
    "validate_command_type",
]
