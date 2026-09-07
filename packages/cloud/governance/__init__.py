# -*- coding: utf-8 -*-
"""
Cloud Governance Module.

CCEA Phase 8 Implementation: Privacy/GDPR + Residency + Access Controls

This module provides:
    - DSARService: GDPR Data Subject Access Requests (export/delete)
    - DataResidencyManager: EU/US region handling
    - RetentionService: Auto-purge per tenant
    - BreakGlassController: Emergency access with audit
    - HealthMonitorService: Agent health dashboards
    - AlertRulesEngine: Event-based alerting
    - CustomerManagedKeysService: CMK support for enterprise

Design Doc Reference:
    - Phase 8: "Telemetry + Privacy/GDPR + Residency + Access Controls"
    - Cloud governance (13.3): retention, export/delete, RBAC, audit
    - Data residency (13.4): EU region default for EU tenants
    - Monitoring/alerts (4.1/3.1): health dashboards, alerts

CLOUD ZONE ONLY.
"""

from typing import Final

ZONE: Final[str] = "cloud"

from .dsar import (
    DSARService,
    DSARRequest,
    DSARRequestType,
    DSARStatus,
    DSARResult,
)
from .residency import (
    DataResidencyManager,
    DataRegion,
    ResidencyPolicy,
    ResidencyConfig,
    ResidencyMode,
)
from .retention import (
    RetentionService,
    RetentionPolicy,
    RetentionConfig,
    PurgeResult,
    RetentionAction,
)
from .break_glass import (
    BreakGlassController,
    BreakGlassRequest,
    BreakGlassReason,
    BreakGlassResult,
    BreakGlassScope,
)
from .health_monitor import (
    HealthMonitorService,
    AgentHealth,
    HealthStatus,
    HealthDashboard,
    RunStatus,
)
from .alert_rules import (
    AlertRulesEngine,
    AlertRule,
    AlertCondition,
    AlertAction,
    AlertTrigger,
    AlertSeverity,
)

# CMK requires cryptography, make it optional
try:
    from .cmk import (
        CustomerManagedKeysService,
        CMKConfig,
        KeyInfo,
        EncryptionResult,
    )

    _HAS_CMK = True
except ImportError:
    _HAS_CMK = False
    CustomerManagedKeysService = None
    CMKConfig = None
    KeyInfo = None
    EncryptionResult = None

# Policy Engine
from .policy_engine import (
    PolicyEngine,
    Policy,
    PolicyRule,
    PolicyCondition,
    PolicyType,
    PolicyEffect,
    PolicyStatus,
    PolicyEvaluationContext,
    PolicyEvaluationReport,
    EvaluationResult,
    ComparisonOperator,
    create_default_risk_policy,
    create_default_deployment_policy,
)

# Support Consent (GDPR Phase 1)
from .consent import (
    SupportConsentService,
    ConsentRequest,
    ConsentRecord,
    ConsentType,
    ConsentStatus,
    ConsentScope,
    ConsentVerificationResult,
    get_consent_service,
)

# Telemetry Contract (GDPR Phase 2)
from .telemetry_contract import (
    TelemetryLevelContract,
    TelemetryContractValidator,
    RawOrderEventsGate,
    EnterpriseRawOptIn,
    ContractViolation,
    ContractValidationResult,
    ContractViolationType,
    ViolationSeverity,
    validate_telemetry_contract,
    get_allowed_fields_for_level,
    get_forbidden_fields_for_level,
    AGGREGATED_ALLOWED_FIELDS,
    DETAILED_ALLOWED_FIELDS,
    RAW_ORDER_ALLOWED_FIELDS,
    ALWAYS_FORBIDDEN_FIELDS,
    PII_FIELDS,
    ORDER_LIKE_FIELDS,
)

# EU-Only Residency Drift Check (GDPR Phase 3)
from .residency_drift import (
    EUOnlyDriftChecker,
    DeploymentConfigValidator,
    ResidencyEvidenceExporter,
    ResidencyDriftReport,
    ResidencyConfiguration,
    EndpointCheck,
    SubprocessorCheck,
    DriftCheckViolation,
    DriftCheckStatus,
    ComponentType,
    CheckSeverity,
    check_eu_residency,
    is_eu_region,
    validate_endpoint_eu,
    EU_AWS_REGIONS,
    EU_GCP_REGIONS,
    EU_AZURE_REGIONS,
    ALL_EU_REGIONS,
    NON_EU_REGIONS,
)

# Retention Service with Legal Hold (GDPR Phase 4)
from .retention_service import (
    # Constants
    MIN_RETENTION_DAYS,
    DEFAULT_RETENTION_DAYS,
    MAX_RETENTION_DAYS,
    COMPLIANCE_DATA_CATEGORIES,
    ALL_DATA_CATEGORIES,
    # Enums
    RetentionAction as RetentionActionV2,
    PurgeStatus,
    LegalHoldStatus,
    # Data classes
    RetentionPolicy as RetentionPolicyV2,
    LegalHold,
    PurgeEvent,
    RetentionValidationResult,
    PurgeSchedulerConfig,
    PurgeSchedulerState,
    # Services
    RetentionPolicyRegistry,
    LegalHoldService,
    AutoPurgeScheduler,
)

# DSAR Phase 5: Data Subject Access Requests
from .dsar_phase5 import (
    # Constants
    DSAR_RESPONSE_DEADLINE_DAYS,
    DSAR_EXTENSION_DAYS,
    DSAR_MAX_TOTAL_DAYS,
    DSAR_DATA_CATEGORIES as DSAR_DATA_CATEGORIES_V2,
    DSAR_OUT_OF_SCOPE_CATEGORIES as DSAR_OUT_OF_SCOPE_CATEGORIES_V2,
    CCEA_BOUNDARY_NOTICE as CCEA_BOUNDARY_NOTICE_V2,
    ERASURE_EXEMPTIONS,
    COMPLIANCE_DATA_CATEGORIES as DSAR_COMPLIANCE_CATEGORIES,
    # Enums
    DSARRequestType as DSARRequestTypeV2,
    DSARStatus as DSARStatusV2,
    VerificationMethod,
    VerificationStatus,
    ExportFormat,
    AuditAction as DSARAuditAction,
    # Data classes
    VerificationToken,
    DSARRequest as DSARRequestV2,
    DSARResult as DSARResultV2,
    DSARMetrics,
    AuditEntry as DSARAuditEntry,
    DataCategory,
    # Registry
    DATA_CATEGORIES,
    # Service
    DSARPhase5Service,
)

# GDPR Phase 6: Access Control, Audit, Break-Glass
from .rbac_service import (
    # Constants
    PERMISSION_CACHE_TTL_SECONDS,
    MAX_ROLES_PER_USER,
    MAX_PERMISSIONS_PER_ROLE,
    RESOURCE_SENSITIVITY,
    AUDIT_REQUIRED_RESOURCES,
    # Enums
    Scope,
    ResourceType,
    SensitivityLevel,
    # Data classes
    Permission,
    Role,
    Principal,
    AccessContext,
    AccessDecision,
    # Service
    RBACService,
)

from .access_audit import (
    # Constants
    AUDIT_LOG_RETENTION_DAYS,
    MAX_QUERY_RESULTS,
    # Enums
    AuditAction as AccessAuditAction,
    AuditResult,
    # Data classes
    AuditPrincipal,
    AuditEntry as AccessAuditEntry,
    AuditQuery,
    AuditStats,
    AuditExport,
    # Service
    AccessAuditService,
)

from .break_glass_phase6 import (
    # Constants
    MAX_BREAK_GLASS_DURATION_HOURS,
    DEFAULT_BREAK_GLASS_DURATION_HOURS,
    MIN_REASON_LENGTH as BG_MIN_REASON_LENGTH,
    BREAK_GLASS_COOLDOWN_MINUTES,
    # Enums
    BreakGlassReasonType,
    BreakGlassScope as BreakGlassScopeV2,
    BreakGlassStatus,
    # Data classes
    BreakGlassRequest as BreakGlassRequestV2,
    BreakGlassResult as BreakGlassResultV2,
    BreakGlassStats,
    # Service
    BreakGlassPhase6Service,
)

from .change_management import (
    # Constants
    APPROVAL_EXPIRY_HOURS,
    MIN_APPROVAL_REASON_LENGTH,
    # Enums
    ChangeClass,
    ChangeType,
    ChangeStatus,
    ApprovalType,
    # Data classes
    ChangeEvidence,
    ApprovalRecord,
    ChangeRequest,
    ChangeJournalEntry,
    ChangeStats,
    # Service
    ChangeManagementService,
)

__all__ = [
    "ZONE",
    # DSAR
    "DSARService",
    "DSARRequest",
    "DSARRequestType",
    "DSARStatus",
    "DSARResult",
    # Residency
    "DataResidencyManager",
    "DataRegion",
    "ResidencyPolicy",
    "ResidencyConfig",
    "ResidencyMode",
    # Retention
    "RetentionService",
    "RetentionPolicy",
    "RetentionConfig",
    "PurgeResult",
    "RetentionAction",
    # Break Glass
    "BreakGlassController",
    "BreakGlassRequest",
    "BreakGlassReason",
    "BreakGlassResult",
    "BreakGlassScope",
    # Health Monitor
    "HealthMonitorService",
    "AgentHealth",
    "HealthStatus",
    "HealthDashboard",
    "RunStatus",
    # Alert Rules
    "AlertRulesEngine",
    "AlertRule",
    "AlertCondition",
    "AlertAction",
    "AlertTrigger",
    "AlertSeverity",
    # CMK
    "CustomerManagedKeysService",
    "CMKConfig",
    "KeyInfo",
    "EncryptionResult",
    # Policy Engine
    "PolicyEngine",
    "Policy",
    "PolicyRule",
    "PolicyCondition",
    "PolicyType",
    "PolicyEffect",
    "PolicyStatus",
    "PolicyEvaluationContext",
    "PolicyEvaluationReport",
    "EvaluationResult",
    "ComparisonOperator",
    "create_default_risk_policy",
    "create_default_deployment_policy",
    # Support Consent (GDPR Phase 1)
    "SupportConsentService",
    "ConsentRequest",
    "ConsentRecord",
    "ConsentType",
    "ConsentStatus",
    "ConsentScope",
    "ConsentVerificationResult",
    "get_consent_service",
    # Telemetry Contract (GDPR Phase 2)
    "TelemetryLevelContract",
    "TelemetryContractValidator",
    "RawOrderEventsGate",
    "EnterpriseRawOptIn",
    "ContractViolation",
    "ContractValidationResult",
    "ContractViolationType",
    "ViolationSeverity",
    "validate_telemetry_contract",
    "get_allowed_fields_for_level",
    "get_forbidden_fields_for_level",
    "AGGREGATED_ALLOWED_FIELDS",
    "DETAILED_ALLOWED_FIELDS",
    "RAW_ORDER_ALLOWED_FIELDS",
    "ALWAYS_FORBIDDEN_FIELDS",
    "PII_FIELDS",
    "ORDER_LIKE_FIELDS",
    # EU-Only Residency Drift Check (GDPR Phase 3)
    "EUOnlyDriftChecker",
    "DeploymentConfigValidator",
    "ResidencyEvidenceExporter",
    "ResidencyDriftReport",
    "ResidencyConfiguration",
    "EndpointCheck",
    "SubprocessorCheck",
    "DriftCheckViolation",
    "DriftCheckStatus",
    "ComponentType",
    "CheckSeverity",
    "check_eu_residency",
    "is_eu_region",
    "validate_endpoint_eu",
    "EU_AWS_REGIONS",
    "EU_GCP_REGIONS",
    "EU_AZURE_REGIONS",
    "ALL_EU_REGIONS",
    "NON_EU_REGIONS",
    # Retention Service with Legal Hold (GDPR Phase 4)
    "MIN_RETENTION_DAYS",
    "DEFAULT_RETENTION_DAYS",
    "MAX_RETENTION_DAYS",
    "COMPLIANCE_DATA_CATEGORIES",
    "ALL_DATA_CATEGORIES",
    "RetentionActionV2",
    "PurgeStatus",
    "LegalHoldStatus",
    "RetentionPolicyV2",
    "LegalHold",
    "PurgeEvent",
    "RetentionValidationResult",
    "PurgeSchedulerConfig",
    "PurgeSchedulerState",
    "RetentionPolicyRegistry",
    "LegalHoldService",
    "AutoPurgeScheduler",
    # DSAR Phase 5: Data Subject Access Requests
    "DSAR_RESPONSE_DEADLINE_DAYS",
    "DSAR_EXTENSION_DAYS",
    "DSAR_MAX_TOTAL_DAYS",
    "DSAR_DATA_CATEGORIES_V2",
    "DSAR_OUT_OF_SCOPE_CATEGORIES_V2",
    "CCEA_BOUNDARY_NOTICE_V2",
    "ERASURE_EXEMPTIONS",
    "DSAR_COMPLIANCE_CATEGORIES",
    "DSARRequestTypeV2",
    "DSARStatusV2",
    "VerificationMethod",
    "VerificationStatus",
    "ExportFormat",
    "DSARAuditAction",
    "VerificationToken",
    "DSARRequestV2",
    "DSARResultV2",
    "DSARMetrics",
    "DSARAuditEntry",
    "DataCategory",
    "DATA_CATEGORIES",
    "DSARPhase5Service",
    # GDPR Phase 6: RBAC Service
    "PERMISSION_CACHE_TTL_SECONDS",
    "MAX_ROLES_PER_USER",
    "MAX_PERMISSIONS_PER_ROLE",
    "RESOURCE_SENSITIVITY",
    "AUDIT_REQUIRED_RESOURCES",
    "Scope",
    "ResourceType",
    "SensitivityLevel",
    "Permission",
    "Role",
    "Principal",
    "AccessContext",
    "AccessDecision",
    "RBACService",
    # GDPR Phase 6: Access Audit Service
    "AUDIT_LOG_RETENTION_DAYS",
    "MAX_QUERY_RESULTS",
    "AccessAuditAction",
    "AuditResult",
    "AuditPrincipal",
    "AccessAuditEntry",
    "AuditQuery",
    "AuditStats",
    "AuditExport",
    "AccessAuditService",
    # GDPR Phase 6: Break-Glass Phase 6 Service
    "MAX_BREAK_GLASS_DURATION_HOURS",
    "DEFAULT_BREAK_GLASS_DURATION_HOURS",
    "BG_MIN_REASON_LENGTH",
    "BREAK_GLASS_COOLDOWN_MINUTES",
    "BreakGlassReasonType",
    "BreakGlassScopeV2",
    "BreakGlassStatus",
    "BreakGlassRequestV2",
    "BreakGlassResultV2",
    "BreakGlassStats",
    "BreakGlassPhase6Service",
    # GDPR Phase 6: Change Management Service
    "APPROVAL_EXPIRY_HOURS",
    "MIN_APPROVAL_REASON_LENGTH",
    "ChangeClass",
    "ChangeType",
    "ChangeStatus",
    "ApprovalType",
    "ChangeEvidence",
    "ApprovalRecord",
    "ChangeRequest",
    "ChangeJournalEntry",
    "ChangeStats",
    "ChangeManagementService",
]
