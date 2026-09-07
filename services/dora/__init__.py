# -*- coding: utf-8 -*-
"""
DORA Compliance Module - Thin Facade.

Digital Operational Resilience Act (DORA) - Regulation (EU) 2022/2554.

ARCHITECTURE (Post-Migration Phase 8):
    services/core/                  - Operational resilience (14 modules) - NOT TOUCHED
    services/dora_integration/      - Client-facing interfaces (21 modules) - ACTIVE
    services/dora/                  - THIS FILE: thin facade for backward compatibility
    services/archive/dora_financial_entity/  - Archived FE modules (23 modules)

USAGE:
    # PREFERRED (direct import from integration layer):
    from services.dora_integration.due_diligence import DORAuditReadiness
    from services.dora_integration.incident_interface import DORAIncidentClassification
    from services.dora_integration.contracts import DORAContractualRequirements

    # DEPRECATED (via this facade - emits DeprecationWarning):
    from services.dora import DORAuditReadiness  # Works but deprecated

    # For Financial Entity modules (archived):
    from services.archive.dora_financial_entity import DORAGovernanceFramework

For new code, ALWAYS prefer direct imports from services.dora_integration.

Migration History:
    v1.0.0: Initial DORA implementation
    v2.0.0: FE modules archived, integration layer active, facade created
    v2.1.0: Phase 8 complete - final integration & cleanup

Key Compliance Dates:
    - Application Date: 17 January 2025
    - Register of Information: 30 April 2025 (via NCAs to ESAs)
    - Reference Date for ROI: 31 March 2025

References:
    - DORA Full Text: https://eur-lex.europa.eu/eli/reg/2022/2554/oj
    - DORA Article 30: https://www.digital-operational-resilience-act.com/Article_30.html
    - ESAs Technical Standards: https://www.esma.europa.eu/publications-and-data/dora
"""

from __future__ import annotations

import warnings
from typing import Any

__version__ = "2.1.0"  # Phase 8: Final Integration & Cleanup
__dora_compliance_phase__ = 8

# =============================================================================
# Re-export EVERYTHING from services.dora_integration
# =============================================================================

# Phase 1: Due Diligence & Audit Layer
from services.dora_integration.due_diligence import (
    # Audit Readiness
    AUDIT_SLA_ACKNOWLEDGMENT_DAYS,
    AUDIT_SLA_SCHEDULING_DAYS,
    AUDIT_SLA_EVIDENCE_STANDARD_DAYS,
    AUDIT_SLA_EVIDENCE_COMPLEX_DAYS,
    AUDIT_SLA_NCA_RESPONSE_DAYS,
    EVIDENCE_RETENTION_YEARS,
    AUDIT_TYPE_SLAS,
    AuditType,
    AuditScope,
    AuditStatus,
    EvidenceType,
    EvidenceCategory,
    AuditRequest,
    EvidenceItem,
    AuditFinding,
    EvidenceTemplate,
    AuditReadinessConfig,
    DORAuditReadiness,
    create_audit_readiness,
    get_standard_evidence_templates,
    IncidentNotificationStatus,
    ClientNotificationRecord,
    MultiClientIncident,
    MultiClientIncidentCoordinator,
    create_incident_coordinator,
    # Provider Info Package
    ICTServiceType,
    FunctionCriticality,
    SubstitutabilityLevel,
    DataSensitivity,
    ProviderIdentification,
    ServiceDescription,
    ICTServiceDescription,
    DataLocation,
    DataLocationInfo,
    SubcontractorInfo,
    CertificationInfo,
    ContractSummary,
    ProviderInfoPackage,
    ProviderInfoConfig,
    DORAProviderInfoPackage,
    ProviderInfoPackageGenerator,
    create_provider_info_package,
    # Pooled Audit Support
    AuditReportType,
    PooledAuditStatus,
    ParticipationStatus,
    AuditScopeArea,
    FindingSeverity,
    RemediationStatus,
    CertificationRecord,
    PooledAuditParticipant,
    PooledAuditFinding,
    PooledAuditEngagement,
    AuditReportAccess,
    PooledAuditConfig,
    PooledAuditSupport,
    create_pooled_audit_support,
    get_audit_scope_areas,
    get_report_types,
    # Compliance Dashboard
    IssueSeverity,
    IssueStatus,
    DeadlineStatus,
    ComplianceIssue,
    Deadline,
    ComplianceStatus,
    DORAComplianceReport,
    DORAComplianceDashboard,
)

# Phase 2: Incident Interface Layer
from services.dora_integration.incident_interface import (
    # Client Notification
    ClientNotificationService,
    DORAClientNotification,
    ClientNotificationConfig,
    IncidentSeverity,
    NotificationStatus,
    NotificationChannel,
    IncidentCategory,
    ClientContact,
    IncidentNotification,
    IncidentUpdate,
    ClientIncident,
    create_client_notification_service,
    create_client_notification_system,
    get_notification_template,
    # Incident Classification
    DORAIncidentClassification,
    IncidentClassificationConfig,
    ClassificationThresholds,
    IncidentClassificationType,
    ClientType,
    DataType,
    CriticalServiceType,
    MajorIncidentTrigger,
    ReputationalImpactLevel,
    ClientImpactAssessment,
    DurationAssessment,
    GeographicAssessment,
    DataLossAssessment,
    CriticalServiceAssessment,
    EconomicImpactAssessment,
    ReputationalAssessment,
    RecurringIncidentAssessment,
    MaliciousAccessAssessment,
    IncidentClassificationResult,
    create_incident_classification,
    get_default_thresholds,
    get_classification_criteria,
    create_client_impact_assessment,
    create_duration_assessment,
    create_economic_impact_assessment,
    create_data_loss_assessment,
    create_critical_service_assessment,
    # Incident Reporting
    DORAIncidentReporter,
    IncidentReportingConfig,
    ReportType,
    ReportStatus,
    IncidentTypeCode,
    RootCauseCategory,
    CompetentAuthorityType,
    CompetentAuthority,
    InitialNotificationReport,
    IntermediateReport,
    FinalReport,
    ClientDataPackage,
    ReportSubmission,
    create_incident_reporter,
    get_report_deadlines,
    # Cyber Threat Notification
    CyberThreatNotificationService,
    CyberThreatNotificationConfig,
    ThreatCategory,
    ThreatActorType,
    ThreatSeverity,
    ThreatStatus,
    ThreatSignificance,
    ThreatIndicator,
    CyberThreat,
    ThreatSignificanceAssessment,
    ThreatNotification,
    create_cyber_threat_notification_service,
    get_threat_categories,
    get_threat_severities,
    # Communication
    DORACommunication,
    CommunicationConfig,
    CommunicationChannel,
    StakeholderType,
    CommunicationPriority,
    CommunicationStatus,
    CrisisPhase,
    PolicyStatus,
    CommunicationContact,
    CommunicationTemplate,
    CommunicationRecord,
    CommunicationPolicy,
    CrisisStatus,
    create_communication_service,
    get_communication_channels,
    get_stakeholder_types,
    get_crisis_phases,
)

# Phase 3: Third-Party Risk Interface
from services.dora_integration.third_party import (
    # Concentration Risk
    DORAConcentrationRisk,
    ConcentrationRiskConfig,
    ConcentrationType,
    ConcentrationRiskLevel,
    MitigationStatus,
    AssessmentScope,
    ProviderDependency,
    ConcentrationMetric,
    ConcentrationRisk,
    MitigationMeasure,
    ConcentrationAssessment,
    DependencyMap,
    create_concentration_risk,
    get_concentration_types,
    get_substitutability_levels,
    # CTPP Oversight
    DORACtppOversight,
    CTPPOversightConfig,
    LeadOverseer,
    CTPPStatus,
    OversightRecommendationType,
    RecommendationStatus,
    ComplianceLevel,
    OversightExerciseType,
    CTPPDesignation,
    OversightRecommendation,
    OversightExercise,
    CTPPRiskAssessment,
    CTPPContractRequirement,
    EntityCTPPRelationship,
    DESIGNATED_CTPPS_2025,
    create_ctpp_oversight,
    get_lead_overseers,
    get_designated_ctpps_list,
    get_ctpp_requirements,
    get_ctpp_contract_requirements,
    # Third-Party Risk Management
    DORAThirdPartyRiskManagement,
    ThirdPartyRiskConfig,
    ProviderType,
    ProviderCriticality,
    ServiceCriticality,
    ProviderStatus,
    RiskCategory,
    RiskLevel,
    DueDiligenceStatus,
    AssessmentType,
    TPRSubstitutabilityLevel,
    ICTService,
    ICTProvider,
    ThirdPartyRisk,
    ThirdPartyRiskAssessment,
    DueDiligenceCheck,
    ProviderRelationshipEvent,
    create_third_party_risk_management,
    get_provider_types,
    get_risk_categories,
    get_criticality_levels,
    # Third-Party Incidents
    DORAThirdPartyIncidents,
    ThirdPartyProviderType,
    ThirdPartyCriticality,
    ThirdPartyIncidentType,
    ContractualSLAStatus,
    EscalationLevel,
    ThirdPartyProvider,
    AffectedService,
    SLAAssessment,
    EscalationRecord,
    IncidentMitigationAction,
    ThirdPartyIncident,
    PostIncidentReview,
    create_third_party_incidents,
    # Subcontractor Management
    DORASubcontractorManagement,
    SubcontractorConfig,
    SubcontractorType,
    SubcontractorStatus,
    SubcontractorRiskLevel,
    ChangeType,
    ConsentMode,
    Subcontractor,
    SubcontractorChange,
    ClientSubcontractorPreference,
    SubcontractorRiskAssessment,
    create_subcontractor_management,
)

# Phase 4: Contracts & SLA Layer
from services.dora_integration.contracts import (
    # Contractual Requirements
    DORAContractualRequirements,
    ContractualRequirementsConfig,
    RequirementCategory,
    RequirementType,
    GapSeverity,
    ContractStatus,
    ContractualRequirement,
    ContractProvision,
    ContractAssessment,
    ContractGap,
    ContractAmendment,
    SLADefinition,
    ICTContract,
    TerminationClause,
    create_contractual_requirements,
    get_article_30_requirements,
    get_requirement_types,
    get_basic_requirement_count,
    get_critical_requirement_count,
    get_termination_clause_templates,
    # SLA Guardrails
    SLAGuardrails,
    SLAGuardrailsConfig,
    SLATier,
    CapacityStatus,
    ApprovalStatus,
    InfrastructureRequirement,
    OnCallRequirement,
    SLATierDefinition,
    CapacityValidation,
    SLACommitmentRequest,
    CurrentCapacityState,
    create_sla_guardrails,
    get_sla_tier_definitions,
    get_sla_tiers,
    # Exit Strategies
    DORAExitStrategies,
    ExitStrategiesConfig,
    ExitTrigger,
    ExitPhase,
    ExitPlanStatus,
    TransitionType,
    ReadinessLevel,
    AlternativeProviderStatus,
    AlternativeProvider,
    DataMigrationPlan,
    TransitionTask,
    ExitRisk,
    ExitCostEstimate,
    ExitPlan,
    ExitExecution,
    ExitReadinessAssessment,
    create_exit_strategies,
    get_exit_triggers,
    get_exit_phases,
    get_transition_types,
)

# Phase 5: Unified Reporting Layer
from services.dora_integration.reporting import (
    # Unified Reporting
    ReportChannel,
    PackageFormat,
    ReportDestination,
    ReportValidationResult,
    UnifiedReport,
    SubmissionPackage,
    DeliveryRecord,
    UnifiedReportingConfig,
    UnifiedReportingManager,
    create_unified_reporting_manager,
    create_report_destination,
    get_report_statuses,
    # Reporting Templates
    DataTypeCode,
    ClientTypeCode,
    ServiceTypeCode,
    ResponseEffectivenessCode,
    TemplateExportFormat,
    ITSInitialNotificationTemplate,
    ITSIntermediateReportTemplate,
    ITSFinalReportTemplate,
    TimelineEvent,
    ClientIncidentDataPackage,
    DORAReportingTemplates,
    create_reporting_templates,
    get_incident_type_codes,
    get_data_type_codes,
    get_service_type_codes,
    get_client_type_codes,
    create_timeline_event,
    # Register of Information
    ContractType,
    ServiceType,
    FunctionType,
    ProviderLocationType,
    SubcontractingLevel,
    ExportFormat,
    ContractReferenceData,
    SubcontractorData,
    ServiceRecord,
    ROIDataPackage,
    ROIDataGeneratorConfig,
    DORARegisterOfInformation,
    create_register_of_information,
    create_roi_data_generator,
    get_contract_types,
    get_service_types,
    get_subcontracting_levels,
    get_its_templates_provided,
    get_its_templates_client_provides,
)

# Phase 6: Information Sharing Layer
from services.dora_integration.sharing import (
    SHAREABLE_INFORMATION_TYPES,
    TLP_DEFINITIONS,
    DEFAULT_INTELLIGENCE_RETENTION_DAYS,
    NCA_NOTIFICATION_DEADLINE_DAYS,
    CommunityType,
    SharingChannel,
    TLPLevel,
    MembershipStatus,
    SharingOutcome,
    IntelligenceDirection,
    SanitizationLevel,
    SharingCommunity,
    InformationSharingPolicy,
    CyberThreatIntelligence,
    ThreatIntelligenceRecord,
    SharingAuditRecord,
    NCANotification,
    InformationSharingConfig,
    DORAInformationSharing,
    create_information_sharing,
    get_shareable_information_types,
    get_tlp_definitions,
    get_community_types,
    get_sharing_channels,
    get_tlp_levels,
    create_sharing_community,
    create_cyber_threat,
    create_sharing_policy,
)

# =============================================================================
# Re-export from Archived Financial Entity Modules
# =============================================================================

# These are FE-specific modules (Art. 2, 3-16, 17, 22, 24-27)
# Kept for backward compatibility and reference implementations
from services.archive.dora_financial_entity.scope_verification import (
    DORAEntityType,
    DORAScopeResult,
    ScopeVerification,
    EntityAuthorization,
    DORAScope,
    create_scope_verifier,
    get_entity_type_description,
)

from services.archive.dora_financial_entity.function_classification import (
    ImpairmentType,
    FunctionClassification,
    ThirdPartyProvider as FEThirdPartyProvider,
    FunctionClassifier,
    create_function_classifier,
    get_platform_functions,
    get_ict_providers,
)

from services.archive.dora_financial_entity.proportionality import (
    DORARegime,
    ExemptionType,
    EntityClassification,
    ProportionalityAssessment,
    RegimeExemption,
    ProportionalityAssessor,
    create_proportionality_assessor,
    assess_entity_proportionality,
)

from services.archive.dora_financial_entity.governance import (
    GovernanceRole,
    DefenceLine,
    TrainingStatus,
    GovernanceRoleAssignment,
    ICTTrainingRecord,
    FrameworkApproval,
    ICTBudgetAllocation,
    DORAGovernanceFramework,
    create_governance_framework,
    MANDATORY_TRAINING_TOPICS,
)

from services.archive.dora_financial_entity.cross_regulation import (
    Regulation,
    ReportingRequirement,
    IncidentAlignmentResult,
    RiskFrameworkAlignment,
    LoggingAlignmentResult,
    DORARegulationIntegration,
)

# ICT Risk Management Framework (Article 6)
from services.archive.dora_financial_entity.ict_risk_framework import (
    PolicyCategory,
    ControlDomain,
    ControlType,
    RiskPolicy,
    RiskProcedure,
    ICTControl,
    FrameworkReview,
    ICTRisk,
    DORAICTRiskFramework,
    create_ict_risk_framework,
)

# Protection and Prevention (Article 9)
from services.archive.dora_financial_entity.protection import (
    SecurityControlCategory,
    AccessControlType,
    AuthenticationType,
    EncryptionType,
    NetworkZone,
    SecurityControl,
    AccessPolicy,
    EncryptionStandard,
    NetworkSecurityRule,
    DataProtectionPolicy,
    DORAProtection,
    create_protection,
)

# Detection (Article 10)
from services.archive.dora_financial_entity.detection import (
    AnomalyType,
    AlertSeverity,
    AlertStatus,
    DetectionMethod,
    MonitoringStatus,
    DetectionRule,
    DetectionAlert,
    PerformanceMetric,
    SinglePointOfFailure,
    DORADetection,
    create_detection,
)

# Backup and Recovery (Article 12)
from services.archive.dora_financial_entity.backup_recovery import (
    BackupType,
    BackupFrequency,
    BackupStatus,
    RecoveryTestType,
    RecoveryTestResult,
    LocationType,
    BackupPolicy,
    BackupJob,
    BackupLocation,
    RecoveryTest,
    RestorationProcedure,
    DORABackupRecovery,
    create_backup_recovery,
)

# Learning and Evolving (Article 13)
from services.archive.dora_financial_entity.learning import (
    ReviewType,
    LessonCategory,
    LessonPriority,
    LessonStatus,
    ImprovementType,
    ImprovementStatus as LearningImprovementStatus,
    KnowledgeType,
    PostIncidentReview as LearningPostIncidentReview,
    LessonLearned,
    ImprovementInitiative,
    KnowledgeArticle,
    TrainingNeed,
    TrendAnalysis,
    InformationShare,
    DORALearning,
    create_dora_learning,
)

# ICT Business Continuity (Article 15)
from services.archive.dora_financial_entity.ict_business_continuity import (
    ContinuityStatus,
    CriticalityLevel,
    ImpactCategory,
    ImpactSeverity,
    ScenarioType,
    RecoveryStrategy,
    ICTBusinessContinuityPolicy,
    BusinessImpactAssessment,
    RecoveryObjective,
    ContinuityPlan,
    DisruptionScenario,
    ContinuityTest,
    AlternativeArrangement,
    DORAICTBusinessContinuity,
    create_dora_ict_business_continuity,
)

# Simplified Framework (Article 16)
from services.archive.dora_financial_entity.simplified_framework import (
    EntitySize,
    SimplifiedControlCategory,
    ControlStatus as SimplifiedControlStatus,
    EligibilityCriteria,
    SimplifiedControl,
    SimplifiedRiskAssessment,
    SimplifiedIncident,
    SimplifiedBackup,
    SimplifiedThirdParty,
    SimplifiedTest,
    SimplifiedAwarenessTraining,
    AnnualReview,
    ESSENTIAL_CONTROLS,
    DORASimplifiedFramework,
    create_dora_simplified_framework,
)

# Incident Management (Article 17)
from services.archive.dora_financial_entity.incident_management import (
    ICTEventType,
    IncidentPhase,
    EarlyWarningType,
    ICTEvent,
    DORAIncident,
    EarlyWarningIndicator,
    IncidentAction,
    IncidentManagementConfig,
    DORAIncidentManagement,
    create_incident_management,
)

# Resilience Testing Programme (Article 24)
from services.archive.dora_financial_entity.resilience_testing import (
    TestCategory,
    TestFrequency,
    FindingSeverity as ResilienceFindingSeverity,
    FindingStatus as ResilienceFindingStatus,
    TestScope,
    TestDefinition,
    TestExecution,
    TestFinding,
    TestingProgramme,
    TestingCycle,
    ResilienceTestingConfig,
    DORAResilienceTestingProgramme,
    create_resilience_testing_programme,
)

# ICT Testing (Article 25)
from services.archive.dora_financial_entity.ict_testing import (
    ICTSystemType,
    TestingPriority,
    VulnerabilityStatus,
    RemediationStatus as ICTRemediationStatus,
    ICTSystemProfile,
    SystemTestPlan,
    SystemTest,
    Vulnerability,
    RemediationPlan,
    ThirdPartyInterfaceTest,
    ICTTestingConfig,
    DORAICTSystemTesting,
    create_ict_system_testing,
)

# Threat-Led Penetration Testing (Article 26)
from services.archive.dora_financial_entity.tlpt import (
    TLPTPhase,
    TLPTStatus,
    ThreatActorCapability,
    AttackTechnique,
    AttackOutcome,
    TLPTFindingSeverity,
    FindingCategory,
    TLPTScope,
    ThreatIntelligenceReport,
    RedTeamScenario,
    AttackAction,
    TLPTFinding,
    PurpleTeamSession,
    TLPTEngagement,
    TLPTAttestation,
    TLPTConfig,
    DORAThreadLedPenetrationTesting,
    create_tlpt,
)

# Tester Management (Article 27)
from services.archive.dora_financial_entity.tester_management import (
    TesterRole,
    CertificationCategory,
    QualificationStatus,
    ConflictCheckResult,
    SecurityCertification,
    TesterExpertise,
    ConflictOfInterestDeclaration,
    ProfessionalIndemnityInsurance,
    TLPTTester,
    TesterOrganization,
    TesterQualificationAssessment,
    InternalTesterApproval,
    TesterManagementConfig,
    DORATestermanagement,
    create_tester_management,
)

# Pooled Testing (Article 26(3))
from services.archive.dora_financial_entity.pooled_testing import (
    PooledTestStatus,
    ParticipantRole,
    ParticipantStatus as PooledParticipantStatus,
    CostSharingModel,
    ProviderCriticality as PooledProviderCriticality,
    SharedProvider,
    PooledTestingParticipant,
    PooledTestingScope,
    CostSharingAgreement,
    PooledTestingEngagement,
    PooledTestingResults,
    PooledTestingConfig,
    DORAPooledTesting,
    create_pooled_testing,
)

# Training Participation (Article 30(2)(i))
from services.archive.dora_financial_entity.training_participation import (
    TrainingType,
    ParticipationMode,
    PersonnelRole,
    TrainingCommitment,
    TrainingRequest,
    TrainingSession,
    QuarterlyUsage,
    TrainingParticipationConfig,
    DORATrainingParticipation,
)

# =============================================================================
# Aliases for backward compatibility
# =============================================================================

# Aliases to prevent import errors from old code
get_report_types_incident = get_report_deadlines  # Alias

# Mapping for commonly aliased exports
NotificationStatus_SubMgmt = NotificationStatus  # Alias from subcontractor
RiskLevel_Exit = RiskLevel  # Alias from exit strategies


# =============================================================================
# Deprecation Warning Handler
# =============================================================================


def __getattr__(name: str) -> Any:
    """
    Handle deprecated attribute access.

    This is triggered when accessing attributes not explicitly imported.
    Provides helpful migration guidance.
    """
    # Check if it's a known deprecated import
    deprecated_mappings = {
        # Old names -> new locations
        "ProviderInfoPackageGenerator": "services.dora_integration.due_diligence.DORAProviderInfoPackage",
        "CrisisCommunicationPlan": "services.dora_integration.incident_interface.communication",
    }

    if name in deprecated_mappings:
        warnings.warn(
            f"'{name}' is deprecated. Use {deprecated_mappings[name]} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Try to return the attribute if available
        try:
            return globals()[name]
        except KeyError:
            pass

    raise AttributeError(f"module 'services.dora' has no attribute '{name}'")


# =============================================================================
# __all__ exports - Everything available via `from services.dora import *`
# =============================================================================

__all__ = [
    # Version info
    "__version__",
    "__dora_compliance_phase__",
    # =========================================================================
    # Phase 1: Due Diligence & Audit Layer
    # =========================================================================
    # Audit Readiness
    "AUDIT_SLA_ACKNOWLEDGMENT_DAYS",
    "AUDIT_SLA_SCHEDULING_DAYS",
    "AUDIT_SLA_EVIDENCE_STANDARD_DAYS",
    "AUDIT_SLA_EVIDENCE_COMPLEX_DAYS",
    "AUDIT_SLA_NCA_RESPONSE_DAYS",
    "EVIDENCE_RETENTION_YEARS",
    "AUDIT_TYPE_SLAS",
    "AuditType",
    "AuditScope",
    "AuditStatus",
    "EvidenceType",
    "EvidenceCategory",
    "AuditRequest",
    "EvidenceItem",
    "AuditFinding",
    "EvidenceTemplate",
    "AuditReadinessConfig",
    "DORAuditReadiness",
    "create_audit_readiness",
    "get_standard_evidence_templates",
    "IncidentNotificationStatus",
    "ClientNotificationRecord",
    "MultiClientIncident",
    "MultiClientIncidentCoordinator",
    "create_incident_coordinator",
    # Provider Info Package
    "ICTServiceType",
    "FunctionCriticality",
    "SubstitutabilityLevel",
    "DataSensitivity",
    "ProviderIdentification",
    "ServiceDescription",
    "ICTServiceDescription",
    "DataLocation",
    "DataLocationInfo",
    "SubcontractorInfo",
    "CertificationInfo",
    "ContractSummary",
    "ProviderInfoPackage",
    "ProviderInfoConfig",
    "DORAProviderInfoPackage",
    "ProviderInfoPackageGenerator",
    "create_provider_info_package",
    # Pooled Audit Support
    "AuditReportType",
    "PooledAuditStatus",
    "ParticipationStatus",
    "AuditScopeArea",
    "FindingSeverity",
    "RemediationStatus",
    "CertificationRecord",
    "PooledAuditParticipant",
    "PooledAuditFinding",
    "PooledAuditEngagement",
    "AuditReportAccess",
    "PooledAuditConfig",
    "PooledAuditSupport",
    "create_pooled_audit_support",
    "get_audit_scope_areas",
    "get_report_types",
    # Compliance Dashboard
    "IssueSeverity",
    "IssueStatus",
    "DeadlineStatus",
    "ComplianceIssue",
    "Deadline",
    "ComplianceStatus",
    "DORAComplianceReport",
    "DORAComplianceDashboard",
    # =========================================================================
    # Phase 2: Incident Interface Layer
    # =========================================================================
    # Client Notification
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
    # Incident Classification
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
    # Incident Reporting
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
    # Cyber Threat Notification
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
    # Communication
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
    # =========================================================================
    # Phase 3: Third-Party Risk Interface
    # =========================================================================
    # Concentration Risk
    "DORAConcentrationRisk",
    "ConcentrationRiskConfig",
    "ConcentrationType",
    "ConcentrationRiskLevel",
    "MitigationStatus",
    "AssessmentScope",
    "ProviderDependency",
    "ConcentrationMetric",
    "ConcentrationRisk",
    "MitigationMeasure",
    "ConcentrationAssessment",
    "DependencyMap",
    "create_concentration_risk",
    "get_concentration_types",
    "get_substitutability_levels",
    # CTPP Oversight
    "DORACtppOversight",
    "CTPPOversightConfig",
    "LeadOverseer",
    "CTPPStatus",
    "OversightRecommendationType",
    "RecommendationStatus",
    "ComplianceLevel",
    "OversightExerciseType",
    "CTPPDesignation",
    "OversightRecommendation",
    "OversightExercise",
    "CTPPRiskAssessment",
    "CTPPContractRequirement",
    "EntityCTPPRelationship",
    "DESIGNATED_CTPPS_2025",
    "create_ctpp_oversight",
    "get_lead_overseers",
    "get_designated_ctpps_list",
    "get_ctpp_requirements",
    "get_ctpp_contract_requirements",
    # Third-Party Risk Management
    "DORAThirdPartyRiskManagement",
    "ThirdPartyRiskConfig",
    "ProviderType",
    "ProviderCriticality",
    "ServiceCriticality",
    "ProviderStatus",
    "RiskCategory",
    "RiskLevel",
    "DueDiligenceStatus",
    "AssessmentType",
    "TPRSubstitutabilityLevel",
    "ICTService",
    "ICTProvider",
    "ThirdPartyRisk",
    "ThirdPartyRiskAssessment",
    "DueDiligenceCheck",
    "ProviderRelationshipEvent",
    "create_third_party_risk_management",
    "get_provider_types",
    "get_risk_categories",
    "get_criticality_levels",
    # Third-Party Incidents
    "DORAThirdPartyIncidents",
    "ThirdPartyProviderType",
    "ThirdPartyCriticality",
    "ThirdPartyIncidentType",
    "ContractualSLAStatus",
    "EscalationLevel",
    "ThirdPartyProvider",
    "AffectedService",
    "SLAAssessment",
    "EscalationRecord",
    "IncidentMitigationAction",
    "ThirdPartyIncident",
    "PostIncidentReview",
    "create_third_party_incidents",
    # Subcontractor Management
    "DORASubcontractorManagement",
    "SubcontractorConfig",
    "SubcontractorType",
    "SubcontractorStatus",
    "SubcontractorRiskLevel",
    "ChangeType",
    "ConsentMode",
    "Subcontractor",
    "SubcontractorChange",
    "ClientSubcontractorPreference",
    "SubcontractorRiskAssessment",
    "create_subcontractor_management",
    # =========================================================================
    # Phase 4: Contracts & SLA Layer
    # =========================================================================
    # Contractual Requirements
    "DORAContractualRequirements",
    "ContractualRequirementsConfig",
    "RequirementCategory",
    "RequirementType",
    "GapSeverity",
    "ContractStatus",
    "ContractualRequirement",
    "ContractProvision",
    "ContractAssessment",
    "ContractGap",
    "ContractAmendment",
    "SLADefinition",
    "ICTContract",
    "TerminationClause",
    "create_contractual_requirements",
    "get_article_30_requirements",
    "get_requirement_types",
    "get_basic_requirement_count",
    "get_critical_requirement_count",
    "get_termination_clause_templates",
    # SLA Guardrails
    "SLAGuardrails",
    "SLAGuardrailsConfig",
    "SLATier",
    "CapacityStatus",
    "ApprovalStatus",
    "InfrastructureRequirement",
    "OnCallRequirement",
    "SLATierDefinition",
    "CapacityValidation",
    "SLACommitmentRequest",
    "CurrentCapacityState",
    "create_sla_guardrails",
    "get_sla_tier_definitions",
    "get_sla_tiers",
    # Exit Strategies
    "DORAExitStrategies",
    "ExitStrategiesConfig",
    "ExitTrigger",
    "ExitPhase",
    "ExitPlanStatus",
    "TransitionType",
    "ReadinessLevel",
    "AlternativeProviderStatus",
    "AlternativeProvider",
    "DataMigrationPlan",
    "TransitionTask",
    "ExitRisk",
    "ExitCostEstimate",
    "ExitPlan",
    "ExitExecution",
    "ExitReadinessAssessment",
    "create_exit_strategies",
    "get_exit_triggers",
    "get_exit_phases",
    "get_transition_types",
    # =========================================================================
    # Phase 5: Unified Reporting Layer
    # =========================================================================
    # Unified Reporting
    "ReportChannel",
    "PackageFormat",
    "ReportDestination",
    "ReportValidationResult",
    "UnifiedReport",
    "SubmissionPackage",
    "DeliveryRecord",
    "UnifiedReportingConfig",
    "UnifiedReportingManager",
    "create_unified_reporting_manager",
    "create_report_destination",
    "get_report_statuses",
    # Reporting Templates
    "DataTypeCode",
    "ClientTypeCode",
    "ServiceTypeCode",
    "ResponseEffectivenessCode",
    "TemplateExportFormat",
    "ITSInitialNotificationTemplate",
    "ITSIntermediateReportTemplate",
    "ITSFinalReportTemplate",
    "TimelineEvent",
    "ClientIncidentDataPackage",
    "DORAReportingTemplates",
    "create_reporting_templates",
    "get_incident_type_codes",
    "get_data_type_codes",
    "get_service_type_codes",
    "get_client_type_codes",
    "create_timeline_event",
    # Register of Information
    "ContractType",
    "ServiceType",
    "FunctionType",
    "ProviderLocationType",
    "SubcontractingLevel",
    "ExportFormat",
    "ContractReferenceData",
    "SubcontractorData",
    "ServiceRecord",
    "ROIDataPackage",
    "ROIDataGeneratorConfig",
    "DORARegisterOfInformation",
    "create_register_of_information",
    "create_roi_data_generator",
    "get_contract_types",
    "get_service_types",
    "get_subcontracting_levels",
    "get_its_templates_provided",
    "get_its_templates_client_provides",
    # =========================================================================
    # Phase 6: Information Sharing Layer
    # =========================================================================
    "SHAREABLE_INFORMATION_TYPES",
    "TLP_DEFINITIONS",
    "DEFAULT_INTELLIGENCE_RETENTION_DAYS",
    "NCA_NOTIFICATION_DEADLINE_DAYS",
    "CommunityType",
    "SharingChannel",
    "TLPLevel",
    "MembershipStatus",
    "SharingOutcome",
    "IntelligenceDirection",
    "SanitizationLevel",
    "SharingCommunity",
    "InformationSharingPolicy",
    "CyberThreatIntelligence",
    "ThreatIntelligenceRecord",
    "SharingAuditRecord",
    "NCANotification",
    "InformationSharingConfig",
    "DORAInformationSharing",
    "create_information_sharing",
    "get_shareable_information_types",
    "get_tlp_definitions",
    "get_community_types",
    "get_sharing_channels",
    "get_tlp_levels",
    "create_sharing_community",
    "create_cyber_threat",
    "create_sharing_policy",
    # =========================================================================
    # Archived Financial Entity Modules (for backward compatibility)
    # =========================================================================
    "DORAEntityType",
    "DORAScopeResult",
    "ScopeVerification",
    "EntityAuthorization",
    "DORAScope",
    "create_scope_verifier",
    "get_entity_type_description",
    "ImpairmentType",
    "FunctionClassification",
    "FEThirdPartyProvider",
    "FunctionClassifier",
    "create_function_classifier",
    "get_platform_functions",
    "get_ict_providers",
    "DORARegime",
    "ExemptionType",
    "EntityClassification",
    "ProportionalityAssessment",
    "RegimeExemption",
    "ProportionalityAssessor",
    "create_proportionality_assessor",
    "assess_entity_proportionality",
    "GovernanceRole",
    "DefenceLine",
    "TrainingStatus",
    "GovernanceRoleAssignment",
    "ICTTrainingRecord",
    "FrameworkApproval",
    "ICTBudgetAllocation",
    "DORAGovernanceFramework",
    "create_governance_framework",
    "MANDATORY_TRAINING_TOPICS",
    "Regulation",
    "ReportingRequirement",
    "IncidentAlignmentResult",
    "RiskFrameworkAlignment",
    "LoggingAlignmentResult",
    "DORARegulationIntegration",
]
