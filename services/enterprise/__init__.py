# -*- coding: utf-8 -*-
"""
Enterprise Features Module for AI-Powered Quantitative Research Platform.

DORA Phase 3: Enterprise Enhancements per DORA_OPERATIONAL_RESILIENCE_PLAN.md

This package provides enterprise-grade features for regulated financial entity clients:

Core Modules:
    - extended_reporting: Extended incident reporting (PDF/JSON formats) per Art. 19-20
    - client_metrics: Per-client metrics and dashboards
    - siem_export: SIEM integration (Splunk/ELK export)
    - tlpt_support: TLPT cooperation procedures per Art. 26
    - feature_flags: Feature flag system for enterprise features
    - multi_region: Multi-region deployment support
    - on_call_management: 24/7 on-call team management
    - soc2_certification: SOC2 Type II certification framework
    - pooled_audit_coordination: Pooled audit coordination per Art. 30(4)
    - dedicated_region: Dedicated region deployment option
    - iso27001_framework: ISO 27001 certification framework
    - sla_templates: Enterprise SLA templates per Art. 30(3)(a)

On-Prem Support:
    - onprem.deployment: On-premises deployment procedures
    - onprem.requirements: On-premises infrastructure requirements

DORA References:
    - Art. 19: Major ICT-related incident reporting
    - Art. 20: Harmonised reporting content and templates
    - Art. 26: Threat-led penetration testing (TLPT)
    - Art. 30(3)(a): Service level descriptions for critical functions
    - Art. 30(4): Pooled audits

Enterprise Tier Requirements:
    - Multi-region deployment completed
    - 24/7 on-call team (4+ FTE)
    - Quarterly DR tests passing
    - SOC2 Type II certification completed
"""

from __future__ import annotations

__version__ = "1.0.0"
__enterprise_phase__ = 3

# =============================================================================
# Extended Reporting (Block 3.2)
# =============================================================================

from services.enterprise.extended_reporting import (
    # Enums
    ReportFormat,
    ReportTemplate,
    ReportSeverity,
    ReportStatus,
    DeliveryMethod,
    # Data structures
    ReportMetadata,
    IncidentSummary,
    TechnicalDetails,
    ImpactAssessment,
    RemediationPlan,
    ExtendedIncidentReport,
    ReportDelivery,
    ReportingConfig,
    # Main class
    ExtendedReportingService,
    # Factory functions
    create_extended_reporting,
    generate_pdf_report,
    generate_json_report,
)

# =============================================================================
# Client Metrics (Block 3.3)
# =============================================================================

from services.enterprise.client_metrics import (
    # Enums
    MetricType,
    MetricPeriod,
    AlertThreshold,
    DashboardType,
    # Data structures
    ClientMetric,
    MetricDataPoint,
    ClientDashboard,
    AlertRule,
    MetricAlert,
    ClientMetricsConfig,
    # Main class
    ClientMetricsService,
    # Factory functions
    create_client_metrics,
)

# =============================================================================
# SIEM Integration (Block 3.4)
# =============================================================================

from services.enterprise.siem_export import (
    # Enums
    SIEMProvider,
    EventSeverity,
    EventCategory,
    ExportStatus,
    # Data structures
    SIEMConfig,
    SecurityEvent,
    EventBatch,
    ExportResult,
    SIEMConnection,
    # Main class
    SIEMExportService,
    # Factory functions
    create_siem_export,
    export_to_splunk,
    export_to_elk,
)

# =============================================================================
# TLPT Support (Block 3.5)
# =============================================================================

from services.enterprise.tlpt_support import (
    # Enums
    TLPTCooperationType,
    TLPTPhase,
    DocumentationType,
    AccessLevel,
    # Data structures
    TLPTCooperationRequest,
    TLPTDocumentation,
    TLPTAccessGrant,
    TLPTFinding,
    TLPTCooperationReport,
    TLPTConfig,
    # Main class
    TLPTCooperationService,
    # Factory functions
    create_tlpt_cooperation,
)

# =============================================================================
# Feature Flags (Block 3.8)
# =============================================================================

from services.enterprise.feature_flags import (
    # Enums
    FeatureTier,
    FeatureStatus,
    RolloutStrategy,
    # Data structures
    FeatureFlag,
    FeatureGate,
    ClientFeatureAccess,
    FeatureFlagConfig,
    # Main class
    FeatureFlagService,
    # Factory functions
    create_feature_flag_service,
)

# =============================================================================
# Multi-Region Deployment (Block 3.9)
# =============================================================================

from services.enterprise.multi_region import (
    # Enums
    Region,
    RegionStatus,
    ReplicationMode,
    FailoverStatus,
    # Data structures
    RegionConfig,
    RegionHealth,
    ReplicationStatus,
    FailoverPlan,
    RegionDeployment,
    MultiRegionConfig,
    # Main class
    MultiRegionService,
    # Factory functions
    create_multi_region_service,
)

# =============================================================================
# On-Call Management (Block 3.10)
# =============================================================================

from services.enterprise.on_call_management import (
    # Enums
    OnCallTier,
    EscalationLevel,
    ShiftType,
    IncidentPriority,
    # Data structures
    OnCallEngineer,
    OnCallSchedule,
    EscalationPolicy,
    IncidentAssignment,
    OnCallMetrics,
    OnCallConfig,
    # Main class
    OnCallManagementService,
    # Factory functions
    create_on_call_management,
)

# =============================================================================
# SOC2 Certification (Block 3.11)
# =============================================================================

from services.enterprise.soc2_certification import (
    # Enums
    SOC2TrustPrinciple,
    ControlStatus,
    EvidenceType,
    AuditStatus,
    # Data structures
    SOC2Control,
    ControlEvidence,
    AuditFinding,
    RemediationItem,
    SOC2Report,
    SOC2Config,
    # Main class
    SOC2CertificationService,
    # Factory functions
    create_soc2_certification,
)

# =============================================================================
# Pooled Audit Coordination (Block 3.12)
# =============================================================================

from services.enterprise.pooled_audit_coordination import (
    # Enums
    AuditCoordinationStatus,
    ParticipantRole,
    CostAllocationMethod,
    # Data structures
    AuditParticipant,
    AuditSchedule,
    CostAllocation,
    AuditCoordinationPlan,
    CoordinationConfig,
    # Main class
    PooledAuditCoordinationService,
    # Factory functions
    create_pooled_audit_coordination,
)

# =============================================================================
# Dedicated Region (Block 3.13)
# =============================================================================

from services.enterprise.dedicated_region import (
    # Enums
    DedicatedRegionType,
    IsolationLevel,
    ComplianceRegime,
    # Data structures
    DedicatedRegionConfig,
    DataResidencyRequirement,
    IsolationBoundary,
    DedicatedRegionDeployment,
    # Main class
    DedicatedRegionService,
    # Factory functions
    create_dedicated_region,
)

# =============================================================================
# ISO 27001 Framework (Block 3.14)
# =============================================================================

from services.enterprise.iso27001_framework import (
    # Enums
    ISO27001Domain,
    ControlObjective,
    ImplementationStatus,
    # Data structures
    ISO27001Control,
    ControlImplementation,
    RiskAssessment,
    ISO27001Audit,
    CertificationStatus,
    ISO27001Config,
    # Main class
    ISO27001FrameworkService,
    # Factory functions
    create_iso27001_framework,
)

# =============================================================================
# Enterprise SLA Templates (Block 3.7)
# =============================================================================

from services.enterprise.sla_templates import (
    # Enums
    SLACategory,
    SLAMetricType,
    PenaltyType,
    # Data structures
    SLAMetric,
    SLATarget,
    SLAPenalty,
    EnterpriseSLA,
    SLATemplateConfig,
    # Main class
    EnterpriseSLAService,
    # Factory functions
    create_enterprise_sla,
    get_enterprise_sla_templates,
)

# =============================================================================
# On-Prem Deployment (Block 3.6)
# =============================================================================

from services.enterprise.onprem.deployment import (
    # Enums
    DeploymentType,
    DeploymentStatus,
    ComponentType,
    # Data structures
    DeploymentRequirement,
    DeploymentComponent,
    DeploymentChecklist,
    OnPremDeployment,
    DeploymentConfig,
    # Main class
    OnPremDeploymentService,
    # Factory functions
    create_onprem_deployment,
)

from services.enterprise.onprem.requirements import (
    # Enums
    RequirementCategory,
    RequirementPriority,
    ComplianceLevel,
    # Data structures
    HardwareRequirement,
    SoftwareRequirement,
    NetworkRequirement,
    SecurityRequirement,
    OnPremRequirements,
    # Main class
    OnPremRequirementsService,
    # Factory functions
    create_onprem_requirements,
    get_minimum_requirements,
)

# =============================================================================
# __all__ exports
# =============================================================================

__all__ = [
    # Version info
    "__version__",
    "__enterprise_phase__",

    # =========================================================================
    # Extended Reporting
    # =========================================================================
    "ReportFormat",
    "ReportTemplate",
    "ReportSeverity",
    "ReportStatus",
    "DeliveryMethod",
    "ReportMetadata",
    "IncidentSummary",
    "TechnicalDetails",
    "ImpactAssessment",
    "RemediationPlan",
    "ExtendedIncidentReport",
    "ReportDelivery",
    "ReportingConfig",
    "ExtendedReportingService",
    "create_extended_reporting",
    "generate_pdf_report",
    "generate_json_report",

    # =========================================================================
    # Client Metrics
    # =========================================================================
    "MetricType",
    "MetricPeriod",
    "AlertThreshold",
    "DashboardType",
    "ClientMetric",
    "MetricDataPoint",
    "ClientDashboard",
    "AlertRule",
    "MetricAlert",
    "ClientMetricsConfig",
    "ClientMetricsService",
    "create_client_metrics",

    # =========================================================================
    # SIEM Export
    # =========================================================================
    "SIEMProvider",
    "EventSeverity",
    "EventCategory",
    "ExportStatus",
    "SIEMConfig",
    "SecurityEvent",
    "EventBatch",
    "ExportResult",
    "SIEMConnection",
    "SIEMExportService",
    "create_siem_export",
    "export_to_splunk",
    "export_to_elk",

    # =========================================================================
    # TLPT Support
    # =========================================================================
    "TLPTCooperationType",
    "TLPTPhase",
    "DocumentationType",
    "AccessLevel",
    "TLPTCooperationRequest",
    "TLPTDocumentation",
    "TLPTAccessGrant",
    "TLPTFinding",
    "TLPTCooperationReport",
    "TLPTConfig",
    "TLPTCooperationService",
    "create_tlpt_cooperation",

    # =========================================================================
    # Feature Flags
    # =========================================================================
    "FeatureTier",
    "FeatureStatus",
    "RolloutStrategy",
    "FeatureFlag",
    "FeatureGate",
    "ClientFeatureAccess",
    "FeatureFlagConfig",
    "FeatureFlagService",
    "create_feature_flag_service",

    # =========================================================================
    # Multi-Region
    # =========================================================================
    "Region",
    "RegionStatus",
    "ReplicationMode",
    "FailoverStatus",
    "RegionConfig",
    "RegionHealth",
    "ReplicationStatus",
    "FailoverPlan",
    "RegionDeployment",
    "MultiRegionConfig",
    "MultiRegionService",
    "create_multi_region_service",

    # =========================================================================
    # On-Call Management
    # =========================================================================
    "OnCallTier",
    "EscalationLevel",
    "ShiftType",
    "IncidentPriority",
    "OnCallEngineer",
    "OnCallSchedule",
    "EscalationPolicy",
    "IncidentAssignment",
    "OnCallMetrics",
    "OnCallConfig",
    "OnCallManagementService",
    "create_on_call_management",

    # =========================================================================
    # SOC2 Certification
    # =========================================================================
    "SOC2TrustPrinciple",
    "ControlStatus",
    "EvidenceType",
    "AuditStatus",
    "SOC2Control",
    "ControlEvidence",
    "AuditFinding",
    "RemediationItem",
    "SOC2Report",
    "SOC2Config",
    "SOC2CertificationService",
    "create_soc2_certification",

    # =========================================================================
    # Pooled Audit Coordination
    # =========================================================================
    "AuditCoordinationStatus",
    "ParticipantRole",
    "CostAllocationMethod",
    "AuditParticipant",
    "AuditSchedule",
    "CostAllocation",
    "AuditCoordinationPlan",
    "CoordinationConfig",
    "PooledAuditCoordinationService",
    "create_pooled_audit_coordination",

    # =========================================================================
    # Dedicated Region
    # =========================================================================
    "DedicatedRegionType",
    "IsolationLevel",
    "ComplianceRegime",
    "DedicatedRegionConfig",
    "DataResidencyRequirement",
    "IsolationBoundary",
    "DedicatedRegionDeployment",
    "DedicatedRegionService",
    "create_dedicated_region",

    # =========================================================================
    # ISO 27001 Framework
    # =========================================================================
    "ISO27001Domain",
    "ControlObjective",
    "ImplementationStatus",
    "ISO27001Control",
    "ControlImplementation",
    "RiskAssessment",
    "ISO27001Audit",
    "CertificationStatus",
    "ISO27001Config",
    "ISO27001FrameworkService",
    "create_iso27001_framework",

    # =========================================================================
    # Enterprise SLA Templates
    # =========================================================================
    "SLACategory",
    "SLAMetricType",
    "PenaltyType",
    "SLAMetric",
    "SLATarget",
    "SLAPenalty",
    "EnterpriseSLA",
    "SLATemplateConfig",
    "EnterpriseSLAService",
    "create_enterprise_sla",
    "get_enterprise_sla_templates",

    # =========================================================================
    # On-Prem Deployment
    # =========================================================================
    "DeploymentType",
    "DeploymentStatus",
    "ComponentType",
    "DeploymentRequirement",
    "DeploymentComponent",
    "DeploymentChecklist",
    "OnPremDeployment",
    "DeploymentConfig",
    "OnPremDeploymentService",
    "create_onprem_deployment",

    # =========================================================================
    # On-Prem Requirements
    # =========================================================================
    "RequirementCategory",
    "RequirementPriority",
    "ComplianceLevel",
    "HardwareRequirement",
    "SoftwareRequirement",
    "NetworkRequirement",
    "SecurityRequirement",
    "OnPremRequirements",
    "OnPremRequirementsService",
    "create_onprem_requirements",
    "get_minimum_requirements",
]
