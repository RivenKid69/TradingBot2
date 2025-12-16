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
]
