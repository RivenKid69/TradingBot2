# -*- coding: utf-8 -*-
"""
CCEA Guardrails Module.

Provides CI/CD guardrails for enforcing CCEA architectural boundaries:
- Import boundary checking (Cloud/Agent/Shared zones)
- Cloud dependency allowlist with transitive deps checking
- Schema validation (no order-like payloads)
- Protocol allowlist validation
- Artifact signature verification
- Redaction enforcement

Phase 2 Implementation: Hard separation of Cloud/Agent/Shared zones.
"""

from .import_check import (
    check_cloud_imports,
    check_agent_imports,
    get_zone_for_module,
    PROHIBITED_IN_CLOUD,
    PROHIBITED_PACKAGES,
    ZoneType,
)
from .cloud_allowlist import (
    ALLOWED_THIRD_PARTY,
    PROHIBITED_INTERNAL,
    PROHIBITED_PATTERNS,
    STDLIB_MODULES,
    AllowlistCheckResult,
    DependencyViolation,
    TransitiveDependencyChecker,
    is_cloud_allowed,
    is_prohibited_internal,
    is_prohibited_package,
    validate_cloud_build,
    validate_cloud_manifest,
)
from .build_artifact_check import (
    PROHIBITED_CODE_PATTERNS,
    PROHIBITED_IMPORTS,
    PROHIBITED_MODULES,
    ArtifactCheckResult,
    ArtifactViolation,
    scan_directory,
    scan_wheel_artifact,
    verify_cloud_artifact,
    verify_cloud_manifest as verify_artifact_manifest,
    verify_cloud_source,
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
    # Cloud allowlist (Phase 2)
    "ALLOWED_THIRD_PARTY",
    "PROHIBITED_INTERNAL",
    "PROHIBITED_PATTERNS",
    "STDLIB_MODULES",
    "AllowlistCheckResult",
    "DependencyViolation",
    "TransitiveDependencyChecker",
    "is_cloud_allowed",
    "is_prohibited_internal",
    "is_prohibited_package",
    "validate_cloud_build",
    "validate_cloud_manifest",
    # Build artifact check (Phase 2)
    "PROHIBITED_CODE_PATTERNS",
    "PROHIBITED_IMPORTS",
    "PROHIBITED_MODULES",
    "ArtifactCheckResult",
    "ArtifactViolation",
    "scan_directory",
    "scan_wheel_artifact",
    "verify_cloud_artifact",
    "verify_artifact_manifest",
    "verify_cloud_source",
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
