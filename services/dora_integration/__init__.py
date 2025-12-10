# -*- coding: utf-8 -*-
"""
DORA Integration Layer.

This package provides interfaces for interacting with financial entity clients
in a DORA-compliant manner. It implements ICT provider obligations under Art. 30.

Architecture Context:
    services/core/              - Operational resilience (14 modules) - NOT TOUCHED
    services/dora_integration/  - Client-facing interfaces (21 modules)
    services/dora/              - Legacy facade for backward compatibility
    services/archive/dora_financial_entity/  - Archived FE modules (23 modules)

Subpackages:
    - due_diligence: Audit readiness, provider info packages (Art. 30(3)(e), Art. 28(3))
    - incident_interface: Client notifications, incident data export (Art. 30(2)(d))
    - third_party: Subcontractor management, concentration risk (Art. 30(2)(b))
    - contracts: Contractual requirements, SLA guardrails, exit strategies (Art. 30)
    - reporting: Unified reporting, ROI data generation (Art. 28(3))
    - sharing: Information sharing arrangements (Art. 45)

Key Principle:
    We are an ICT Third-Party Provider (Art. 30), NOT a Financial Entity (Art. 2).
    This integration layer is what we provide to clients for THEIR compliance.

References:
    - DORA Article 30: https://www.digital-operational-resilience-act.com/Article_30.html
    - CIR 2024/2956: ITS on Register of Information
    - CDR 2024/1772: RTS on Incident Classification

Migration Status:
    Phase 0: Directory structure created (COMPLETE)
    Phase 1: Due Diligence & Audit Layer (COMPLETE)
    Phase 2: Incident Interface Layer (COMPLETE)
    Phase 3-8: Module migration pending
"""

from __future__ import annotations

__version__ = "1.1.0"
__migration_phase__ = 2  # Current migration phase

# =============================================================================
# Phase 1: Due Diligence & Audit Layer (Art. 30(3)(e), Art. 28(3), Art. 30(4))
# =============================================================================

from services.dora_integration.due_diligence import (
    # =========================================================================
    # Audit Readiness (Art. 30(3)(e))
    # =========================================================================
    # SLA Constants
    AUDIT_SLA_ACKNOWLEDGMENT_DAYS,
    AUDIT_SLA_SCHEDULING_DAYS,
    AUDIT_SLA_EVIDENCE_STANDARD_DAYS,
    AUDIT_SLA_EVIDENCE_COMPLEX_DAYS,
    AUDIT_SLA_NCA_RESPONSE_DAYS,
    EVIDENCE_RETENTION_YEARS,
    AUDIT_TYPE_SLAS,
    # Enums
    AuditType,
    AuditScope,
    AuditStatus,
    EvidenceType,
    EvidenceCategory,
    # Data structures
    AuditRequest,
    EvidenceItem,
    AuditFinding,
    EvidenceTemplate,
    AuditReadinessConfig,
    # Main class
    DORAuditReadiness,
    # Factory functions
    create_audit_readiness,
    get_standard_evidence_templates,
    # Multi-Client Incident Coordination (Art. 30(2)(f))
    IncidentNotificationStatus,
    ClientNotificationRecord,
    MultiClientIncident,
    MultiClientIncidentCoordinator,
    create_incident_coordinator,

    # =========================================================================
    # Provider Information Package (Art. 28(3))
    # =========================================================================
    # Enums
    ICTServiceType,
    FunctionCriticality,
    SubstitutabilityLevel,
    DataSensitivity,
    # Data structures
    ProviderIdentification,
    ServiceDescription,
    ICTServiceDescription,
    DataLocation,
    DataLocationInfo,
    SubcontractorInfo,
    CertificationInfo,
    ContractSummary,
    ProviderInfoPackage,
    # Config
    ProviderInfoConfig,
    # Main class
    DORAProviderInfoPackage,
    ProviderInfoPackageGenerator,
    # Factory functions
    create_provider_info_package,

    # =========================================================================
    # Pooled Audit Support (Art. 30(4))
    # =========================================================================
    # Enums
    AuditReportType,
    PooledAuditStatus,
    ParticipationStatus,
    AuditScopeArea,
    FindingSeverity,
    RemediationStatus,
    # Data structures
    CertificationRecord,
    PooledAuditParticipant,
    PooledAuditFinding,
    PooledAuditEngagement,
    AuditReportAccess,
    PooledAuditConfig,
    # Main class
    PooledAuditSupport,
    # Factory functions
    create_pooled_audit_support,
    get_audit_scope_areas,
    get_report_types,

    # =========================================================================
    # Compliance Dashboard
    # =========================================================================
    # Enums
    IssueSeverity,
    IssueStatus,
    DeadlineStatus,
    # Data structures
    ComplianceIssue,
    Deadline,
    ComplianceStatus,
    DORAComplianceReport,
    # Main class
    DORAComplianceDashboard,
)

# =============================================================================
# Phase 2: Incident Interface Layer (Art. 30(2)(d), Art. 18, Art. 19, Art. 14)
# =============================================================================

from services.dora_integration.incident_interface import (
    # =========================================================================
    # Client Incident Notification (Article 30(2)(d))
    # =========================================================================
    # Main Service
    ClientNotificationService,
    DORAClientNotification,
    # Configuration
    ClientNotificationConfig,
    # Enums
    IncidentSeverity,
    NotificationStatus,
    NotificationChannel,
    IncidentCategory,
    # Data Structures
    ClientContact,
    IncidentNotification,
    IncidentUpdate,
    ClientIncident,
    # Factory Functions
    create_client_notification_service,
    create_client_notification_system,
    get_notification_template,

    # =========================================================================
    # Incident Classification (Article 18)
    # =========================================================================
    # Main Service
    DORAIncidentClassification,
    # Configuration
    IncidentClassificationConfig,
    ClassificationThresholds,
    # Enums
    IncidentClassificationType,
    ClientType,
    DataType,
    CriticalServiceType,
    MajorIncidentTrigger,
    ReputationalImpactLevel,
    # Assessment Data Structures
    ClientImpactAssessment,
    DurationAssessment,
    GeographicAssessment,
    DataLossAssessment,
    CriticalServiceAssessment,
    EconomicImpactAssessment,
    ReputationalAssessment,
    RecurringIncidentAssessment,
    MaliciousAccessAssessment,
    # Result
    IncidentClassificationResult,
    # Factory Functions
    create_incident_classification,
    get_default_thresholds,
    get_classification_criteria,
    create_client_impact_assessment,
    create_duration_assessment,
    create_economic_impact_assessment,
    create_data_loss_assessment,
    create_critical_service_assessment,

    # =========================================================================
    # Incident Reporting (Article 19) - Export Templates
    # =========================================================================
    # Main Service
    DORAIncidentReporter,
    # Configuration
    IncidentReportingConfig,
    # Enums
    ReportType,
    ReportStatus,
    IncidentTypeCode,
    RootCauseCategory,
    CompetentAuthorityType,
    # Data Structures
    CompetentAuthority,
    InitialNotificationReport,
    IntermediateReport,
    FinalReport,
    ClientDataPackage,
    ReportSubmission,
    # Factory Functions
    create_incident_reporter,
    get_report_deadlines,
    get_report_types as get_incident_report_types,  # Alias to avoid conflict

    # =========================================================================
    # Cyber Threat Notification (Article 19(4))
    # =========================================================================
    # Main Service
    CyberThreatNotificationService,
    # Configuration
    CyberThreatNotificationConfig,
    # Enums
    ThreatCategory,
    ThreatActorType,
    ThreatSeverity,
    ThreatStatus,
    ThreatSignificance,
    # Data Structures
    ThreatIndicator,
    CyberThreat,
    ThreatSignificanceAssessment,
    ThreatNotification,
    # Factory Functions
    create_cyber_threat_notification_service,
    get_threat_categories,
    get_threat_severities,

    # =========================================================================
    # Crisis Communication (Article 14)
    # =========================================================================
    # Main Service
    DORACommunication,
    # Configuration
    CommunicationConfig,
    # Enums
    CommunicationChannel,
    StakeholderType,
    CommunicationPriority,
    CommunicationStatus,
    CrisisPhase,
    PolicyStatus,
    # Data Structures
    CommunicationContact,
    CommunicationTemplate,
    CommunicationRecord,
    CommunicationPolicy,
    CrisisStatus,
    # Factory Functions
    create_communication_service,
    get_communication_channels,
    get_stakeholder_types,
    get_crisis_phases,
)

# =============================================================================
# __all__ exports
# =============================================================================

__all__ = [
    # Version info
    "__version__",
    "__migration_phase__",

    # =========================================================================
    # Phase 1: Due Diligence & Audit Layer
    # =========================================================================

    # -------------------------------------------------------------------------
    # Audit Readiness (Art. 30(3)(e))
    # -------------------------------------------------------------------------
    # SLA Constants
    "AUDIT_SLA_ACKNOWLEDGMENT_DAYS",
    "AUDIT_SLA_SCHEDULING_DAYS",
    "AUDIT_SLA_EVIDENCE_STANDARD_DAYS",
    "AUDIT_SLA_EVIDENCE_COMPLEX_DAYS",
    "AUDIT_SLA_NCA_RESPONSE_DAYS",
    "EVIDENCE_RETENTION_YEARS",
    "AUDIT_TYPE_SLAS",
    # Enums
    "AuditType",
    "AuditScope",
    "AuditStatus",
    "EvidenceType",
    "EvidenceCategory",
    # Data structures
    "AuditRequest",
    "EvidenceItem",
    "AuditFinding",
    "EvidenceTemplate",
    "AuditReadinessConfig",
    # Main class
    "DORAuditReadiness",
    # Factory functions
    "create_audit_readiness",
    "get_standard_evidence_templates",
    # Multi-Client Incident Coordination
    "IncidentNotificationStatus",
    "ClientNotificationRecord",
    "MultiClientIncident",
    "MultiClientIncidentCoordinator",
    "create_incident_coordinator",

    # -------------------------------------------------------------------------
    # Provider Information Package (Art. 28(3))
    # -------------------------------------------------------------------------
    # Enums
    "ICTServiceType",
    "FunctionCriticality",
    "SubstitutabilityLevel",
    "DataSensitivity",
    # Data structures
    "ProviderIdentification",
    "ServiceDescription",
    "ICTServiceDescription",
    "DataLocation",
    "DataLocationInfo",
    "SubcontractorInfo",
    "CertificationInfo",
    "ContractSummary",
    "ProviderInfoPackage",
    # Config
    "ProviderInfoConfig",
    # Main class
    "DORAProviderInfoPackage",
    "ProviderInfoPackageGenerator",
    # Factory functions
    "create_provider_info_package",

    # -------------------------------------------------------------------------
    # Pooled Audit Support (Art. 30(4))
    # -------------------------------------------------------------------------
    # Enums
    "AuditReportType",
    "PooledAuditStatus",
    "ParticipationStatus",
    "AuditScopeArea",
    "FindingSeverity",
    "RemediationStatus",
    # Data structures
    "CertificationRecord",
    "PooledAuditParticipant",
    "PooledAuditFinding",
    "PooledAuditEngagement",
    "AuditReportAccess",
    "PooledAuditConfig",
    # Main class
    "PooledAuditSupport",
    # Factory functions
    "create_pooled_audit_support",
    "get_audit_scope_areas",
    "get_report_types",

    # -------------------------------------------------------------------------
    # Compliance Dashboard
    # -------------------------------------------------------------------------
    # Enums
    "IssueSeverity",
    "IssueStatus",
    "DeadlineStatus",
    # Data structures
    "ComplianceIssue",
    "Deadline",
    "ComplianceStatus",
    "DORAComplianceReport",
    # Main class
    "DORAComplianceDashboard",

    # =========================================================================
    # Phase 2: Incident Interface Layer
    # =========================================================================

    # -------------------------------------------------------------------------
    # Client Incident Notification (Article 30(2)(d))
    # -------------------------------------------------------------------------
    "ClientNotificationService",
    "DORAClientNotification",
    "ClientNotificationConfig",
    "IncidentSeverity",
    "NotificationStatus",
    "NotificationChannel",
    "IncidentCategory",
    "ClientContact",
    "IncidentNotification",
    "IncidentUpdate",
    "ClientIncident",
    "create_client_notification_service",
    "create_client_notification_system",
    "get_notification_template",

    # -------------------------------------------------------------------------
    # Incident Classification (Article 18)
    # -------------------------------------------------------------------------
    "DORAIncidentClassification",
    "IncidentClassificationConfig",
    "ClassificationThresholds",
    "IncidentClassificationType",
    "ClientType",
    "DataType",
    "CriticalServiceType",
    "MajorIncidentTrigger",
    "ReputationalImpactLevel",
    "ClientImpactAssessment",
    "DurationAssessment",
    "GeographicAssessment",
    "DataLossAssessment",
    "CriticalServiceAssessment",
    "EconomicImpactAssessment",
    "ReputationalAssessment",
    "RecurringIncidentAssessment",
    "MaliciousAccessAssessment",
    "IncidentClassificationResult",
    "create_incident_classification",
    "get_default_thresholds",
    "get_classification_criteria",
    "create_client_impact_assessment",
    "create_duration_assessment",
    "create_economic_impact_assessment",
    "create_data_loss_assessment",
    "create_critical_service_assessment",

    # -------------------------------------------------------------------------
    # Incident Reporting (Article 19) - Export Templates
    # -------------------------------------------------------------------------
    "DORAIncidentReporter",
    "IncidentReportingConfig",
    "ReportType",
    "ReportStatus",
    "IncidentTypeCode",
    "RootCauseCategory",
    "CompetentAuthorityType",
    "CompetentAuthority",
    "InitialNotificationReport",
    "IntermediateReport",
    "FinalReport",
    "ClientDataPackage",
    "ReportSubmission",
    "create_incident_reporter",
    "get_report_deadlines",
    "get_incident_report_types",

    # -------------------------------------------------------------------------
    # Cyber Threat Notification (Article 19(4))
    # -------------------------------------------------------------------------
    "CyberThreatNotificationService",
    "CyberThreatNotificationConfig",
    "ThreatCategory",
    "ThreatActorType",
    "ThreatSeverity",
    "ThreatStatus",
    "ThreatSignificance",
    "ThreatIndicator",
    "CyberThreat",
    "ThreatSignificanceAssessment",
    "ThreatNotification",
    "create_cyber_threat_notification_service",
    "get_threat_categories",
    "get_threat_severities",

    # -------------------------------------------------------------------------
    # Crisis Communication (Article 14)
    # -------------------------------------------------------------------------
    "DORACommunication",
    "CommunicationConfig",
    "CommunicationChannel",
    "StakeholderType",
    "CommunicationPriority",
    "CommunicationStatus",
    "CrisisPhase",
    "PolicyStatus",
    "CommunicationContact",
    "CommunicationTemplate",
    "CommunicationRecord",
    "CommunicationPolicy",
    "CrisisStatus",
    "create_communication_service",
    "get_communication_channels",
    "get_stakeholder_types",
    "get_crisis_phases",
]
