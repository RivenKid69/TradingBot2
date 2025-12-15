# -*- coding: utf-8 -*-
"""
Cloud Control Plane Middleware.

Phase 2 Implementation: Operational governance, audit, privacy.

This module provides middleware for:
    - AuditMiddleware: Automatic audit logging for sensitive operations
    - TelemetryValidator: Validation and rejection of prohibited telemetry fields

CLOUD ZONE ONLY.
"""

from .audit_middleware import (
    AuditEntry,
    AuditContext,
    AuditLogger,
    AuditMiddleware,
    audit_scope,
    audit_sensitive_access,
    get_current_audit_context,
    SENSITIVE_RESOURCES_READ,
    SENSITIVE_RESOURCES_WRITE,
    SENSITIVE_ENDPOINTS,
)

from .telemetry_validation import (
    ValidationSeverity,
    ViolationType,
    ValidationViolation,
    ValidationResult,
    TelemetryValidator,
    RedactionEnforcer,
    validate_telemetry_payload,
    assert_no_order_fields,
    PROHIBITED_ORDER_FIELDS,
    PROHIBITED_PII_FIELDS,
    ALLOWED_AGGREGATED_FIELDS,
    ALLOWED_DETAILED_FIELDS,
)

__all__ = [
    # Audit
    "AuditEntry",
    "AuditContext",
    "AuditLogger",
    "AuditMiddleware",
    "audit_scope",
    "audit_sensitive_access",
    "get_current_audit_context",
    "SENSITIVE_RESOURCES_READ",
    "SENSITIVE_RESOURCES_WRITE",
    "SENSITIVE_ENDPOINTS",
    # Telemetry Validation
    "ValidationSeverity",
    "ViolationType",
    "ValidationViolation",
    "ValidationResult",
    "TelemetryValidator",
    "RedactionEnforcer",
    "validate_telemetry_payload",
    "assert_no_order_fields",
    "PROHIBITED_ORDER_FIELDS",
    "PROHIBITED_PII_FIELDS",
    "ALLOWED_AGGREGATED_FIELDS",
    "ALLOWED_DETAILED_FIELDS",
]
