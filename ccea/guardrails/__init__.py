# -*- coding: utf-8 -*-
"""
CCEA Guardrails Module.

Provides CI/CD guardrails for enforcing CCEA architectural boundaries:
- Import boundary checking (Cloud/Agent/Shared zones)
- Schema validation (no order-like payloads)
- Protocol allowlist validation
- Artifact signature verification

Phase 2 Implementation: Hard separation of Cloud/Agent/Shared zones.
"""

from .artifact_check import (
    ArtifactGuardrails,
    CheckResult,
    CheckSeverity,
    GuardrailCheck,
    GuardrailReport,
    run_artifact_guardrails,
)
from .import_check import (
    PROHIBITED_IN_CLOUD,
    PROHIBITED_PACKAGES,
    ZoneType,
    check_agent_imports,
    check_cloud_imports,
    get_zone_for_module,
)
from .protocol_check import (
    ALLOWED_COMMAND_TYPES,
    ALLOWED_MESSAGE_TYPES,
    check_protocol_changes,
    validate_command_type,
)
from .schema_check import (
    PROHIBITED_FIELDS,
    PROHIBITED_VALUES,
    check_prohibited_fields,
    validate_manifest_schema,
    validate_protocol_schema,
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
    # Artifact guardrails (Phase 4)
    "ArtifactGuardrails",
    "GuardrailCheck",
    "GuardrailReport",
    "CheckSeverity",
    "CheckResult",
    "run_artifact_guardrails",
]

# Optional, Cloud-/docs-specific guardrails (present in the monorepo / private repos).
# Public SDK builds may intentionally omit these modules; importing ccea.guardrails should still work.
try:
    from .cloud_allowlist import (  # noqa: F401
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

    __all__.extend(
        [
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
        ]
    )
except ImportError:
    pass

try:
    from .build_artifact_check import (  # noqa: F401
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

    __all__.extend(
        [
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
        ]
    )
except ImportError:
    pass

try:
    from .intent_prohibition import (  # noqa: F401
        IntentProhibitionResult,
        IntentProhibitionViolation,
        check_cloud_package_for_intents,
        check_python_source_for_intent_injection,
    )

    __all__.extend(
        [
            "IntentProhibitionResult",
            "IntentProhibitionViolation",
            "check_cloud_package_for_intents",
            "check_python_source_for_intent_injection",
        ]
    )
except ImportError:
    pass

try:
    from .design_doc_check import (  # noqa: F401
        compute_sha256,
        verify_design_doc_sha,
    )

    __all__.extend(["compute_sha256", "verify_design_doc_sha"])
except ImportError:
    pass

try:
    from .traceability_check import (  # noqa: F401
        TraceabilityCheckResult,
        TraceabilityViolation,
        validate_traceability_matrix,
    )

    __all__.extend(
        ["TraceabilityCheckResult", "TraceabilityViolation", "validate_traceability_matrix"]
    )
except ImportError:
    pass
