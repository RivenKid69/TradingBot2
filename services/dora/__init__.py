# -*- coding: utf-8 -*-
"""
DORA Compliance Module for AI-Powered Quantitative Research Platform.

Digital Operational Resilience Act (DORA) - Regulation (EU) 2022/2554.

This package provides compliance tools for financial entities per DORA requirements:

Phase 0 - Proportionality Assessment (Articles 2, 3(22), 4, 16):
    - scope_verification: DORA Scope Verification (Article 2)
    - function_classification: Critical/Important Function Classification (Article 3(22))
    - proportionality: Entity Classification and Regime Determination (Articles 4, 16)

Phase 1 - ICT Risk Management Framework (Articles 5-16):
    - governance: Management Body Oversight and ICT Governance (Article 5)
    - ict_risk_framework: ICT Risk Management Framework (Article 6)
    - ict_systems: ICT Systems, Protocols and Tools (Article 7)
    - ict_identification: ICT Risk Identification (Article 8)
    - protection: Protection and Prevention (Article 9)
    - detection: Anomaly Detection (Article 10)
    - response_recovery: Response and Recovery (Article 11)
    - backup_recovery: Backup and Recovery (Article 12)
    - learning: Learning and Evolving (Article 13)
    - communication: Crisis Communication (Article 14)
    - ict_business_continuity: ICT Business Continuity (Article 15)
    - simplified_framework: Simplified Framework for Smaller Entities (Article 16)

Phase 2 - ICT Incident Management & Reporting (Articles 17-23):
    - incident_classification: Incident Classification (CDR 2024/1772)
    - incident_management: ICT Incident Management (Article 17)
    - incident_reporting: Major Incident Reporting (Article 19)
    - cyber_threat_notification: Cyber Threat Notification (Article 19a)

Phase 3 - Digital Resilience Testing (Articles 24-27):
    - vulnerability_assessment: Vulnerability Assessments (Article 24)
    - penetration_testing: Penetration Testing (Article 24)
    - tlpt: Threat-Led Penetration Testing (Article 26) - if designated

Phase 4 - Third-Party ICT Risk Management (Articles 28-30):
    - register_of_information: Register of Information (Article 28)
    - third_party_risk: Third-Party Risk Assessment (Article 28)
    - contract_requirements: Contractual Arrangements (Article 30)
    - its_export: ITS Export (CIR 2024/2956)

Phase 5 - Information Sharing & Integration (Article 45):
    - threat_intelligence: Threat Intelligence Sharing
    - cross_regulation: Cross-Regulation Integration (AI Act, MiFID II)

Scope of Application:
    Financial entities subject to DORA per Article 2(1):
    - Investment firms (Article 2(1)(e))
    - Crypto-asset service providers (Article 2(1)(f))
    - Other 19 types of financial entities

Key Compliance Dates:
    - Application Date: 17 January 2025
    - Register of Information: 30 April 2025 (via NCAs to ESAs)
    - Reference Date for ROI: 31 March 2025

Technical Standards:
    - RTS on ICT Risk Management Framework (CDR 2024/1774)
    - RTS on Incident Classification (CDR 2024/1772)
    - RTS on Incident Reporting (CDR 2025/301)
    - ITS on Register of Information (CIR 2024/2956)

References:
    - DORA Full Text: https://eur-lex.europa.eu/eli/reg/2022/2554/oj
    - DORA Article 2: https://www.digital-operational-resilience-act.com/Article_2.html
    - ESAs Technical Standards: https://www.esma.europa.eu/publications-and-data/dora
    - CTPP Designations: https://www.esma.europa.eu/press-news/esma-news/european-supervisory-authorities-designate-critical-ict-third-party-providers
"""

from __future__ import annotations

__version__ = "0.3.0"
__dora_compliance_phase__ = 2  # Current implementation phase

# =============================================================================
# Phase 0 exports (Proportionality Assessment)
# =============================================================================

from services.dora.scope_verification import (
    # Enums
    DORAEntityType,
    DORAScopeResult,
    # Data structures
    ScopeVerification,
    EntityAuthorization,
    # Main class
    DORAScope,
    # Factory functions
    create_scope_verifier,
    get_entity_type_description,
)

from services.dora.function_classification import (
    # Enums
    FunctionCriticality,
    ImpairmentType,
    # Data structures
    FunctionClassification,
    ICTService,
    ThirdPartyProvider,
    # Main class
    FunctionClassifier,
    # Factory functions
    create_function_classifier,
    get_platform_functions,
    get_ict_providers,
)

from services.dora.proportionality import (
    # Enums
    DORARegime,
    ExemptionType,
    # Data structures
    EntityClassification,
    ProportionalityAssessment,
    RegimeExemption,
    # Main class
    ProportionalityAssessor,
    # Factory functions
    create_proportionality_assessor,
    assess_entity_proportionality,
)

# =============================================================================
# Phase 1 exports (ICT Risk Management Framework - Articles 5-16)
# =============================================================================

# Article 5 - Governance
from services.dora.governance import (
    # Enums
    GovernanceRole,
    DefenceLine,
    TrainingStatus,
    ApprovalStatus,
    # Data structures
    GovernanceRoleAssignment,
    ICTTrainingRecord,
    FrameworkApproval,
    AuditFinding,
    ICTBudgetAllocation,
    # Main class
    DORAGovernanceFramework,
    # Factory functions
    create_governance_framework,
    # Constants
    MANDATORY_TRAINING_TOPICS,
)

# Article 6 - ICT Risk Management Framework
from services.dora.ict_risk_framework import (
    # Enums
    PolicyCategory,
    ControlDomain,
    ControlType,
    RiskLevel as ICTRiskLevel,
    # Data structures
    RiskPolicy,
    RiskProcedure,
    ICTControl,
    FrameworkReview,
    ICTRisk,
    # Main class
    DORAICTRiskFramework,
    # Factory functions
    create_ict_risk_framework,
)

# Article 7 - ICT Systems
from services.dora.ict_systems import (
    # Enums
    SystemCriticality,
    SystemType,
    SystemStatus,
    CapacityStatus,
    AutomationLevel,
    # Data structures
    ICTSystem,
    CapacityMetric,
    ReliabilityMetric,
    AutomationCapability,
    SystemUpgrade,
    # Main class
    DORAICTSystemsManager,
    # Factory functions
    create_ict_systems_manager,
)

# Article 8 - ICT Identification
from services.dora.ict_identification import (
    # Enums
    AssetType,
    AssetClassification,
    RiskSourceCategory,
    ThreatCategory,
    VulnerabilitySeverity,
    # Data structures
    ICTAsset,
    RiskSource,
    CyberThreat,
    ICTVulnerability,
    ICTDependency,
    BusinessFunction,
    # Main class
    DORAICTIdentification,
    # Factory functions
    create_ict_identification,
)

# Article 9 - Protection and Prevention
from services.dora.protection import (
    # Enums
    SecurityControlCategory,
    AccessControlType,
    AuthenticationType,
    EncryptionType,
    NetworkZone,
    # Data structures
    SecurityControl,
    AccessPolicy,
    EncryptionStandard,
    NetworkSecurityRule,
    DataProtectionPolicy,
    # Main class
    DORAProtection,
    # Factory functions
    create_protection,
)

# Article 10 - Detection
from services.dora.detection import (
    # Enums
    AnomalyType,
    AlertSeverity,
    AlertStatus,
    DetectionMethod,
    MonitoringStatus,
    # Data structures
    DetectionRule,
    DetectionAlert,
    PerformanceMetric,
    SinglePointOfFailure,
    # Main class
    DORADetection,
    # Factory functions
    create_detection,
)

# Article 11 - Response and Recovery
from services.dora.response_recovery import (
    # Enums
    IncidentSeverity,
    IncidentStatus,
    IncidentCategory,
    EscalationLevel,
    CrisisStatus,
    # Data structures
    ICTIncident,
    ResponseProcedure,
    EscalationRule,
    CrisisEvent as ResponseCrisisEvent,
    RecoveryAction,
    # Main class
    DORAResponseRecovery,
    # Factory functions
    create_response_recovery,
)

# Article 12 - Backup and Recovery
from services.dora.backup_recovery import (
    # Enums
    BackupType,
    BackupFrequency,
    BackupStatus,
    RecoveryTestType,
    RecoveryTestResult,
    LocationType,
    # Data structures
    BackupPolicy,
    BackupJob,
    BackupLocation,
    RecoveryTest,
    RestorationProcedure,
    # Main class
    DORABackupRecovery,
    # Factory functions
    create_backup_recovery,
)

# Article 13 - Learning and Evolving
from services.dora.learning import (
    # Enums
    ReviewType,
    LessonCategory,
    LessonPriority,
    LessonStatus,
    ImprovementType,
    ImprovementStatus,
    KnowledgeType,
    # Data structures
    PostIncidentReview,
    LessonLearned,
    ImprovementInitiative,
    KnowledgeArticle,
    TrainingNeed,
    TrendAnalysis,
    InformationShare,
    # Main class
    DORALearning,
    # Factory functions
    create_dora_learning,
)

# Article 14 - Communication
from services.dora.communication import (
    # Enums
    CommunicationType,
    CommunicationChannel,
    CommunicationPriority,
    CommunicationStatus,
    NotificationType,
    StakeholderType,
    EscalationTrigger,
    # Data structures
    CommunicationPolicy,
    Stakeholder,
    CommunicationTemplate,
    Communication,
    EscalationRule as CommunicationEscalationRule,
    CrisisCommunicationPlan,
    RegulatoryNotification,
    CommunicationLog,
    # Main class
    DORACommunication,
    # Factory functions
    create_dora_communication,
)

# Article 15 - ICT Business Continuity
from services.dora.ict_business_continuity import (
    # Enums
    ContinuityStatus,
    CriticalityLevel,
    ImpactCategory,
    ImpactSeverity,
    TestType as ContinuityTestType,
    TestResult as ContinuityTestResult,
    ScenarioType,
    RecoveryStrategy,
    # Data structures
    ICTBusinessContinuityPolicy,
    BusinessFunction as ContinuityBusinessFunction,
    BusinessImpactAssessment,
    RecoveryObjective,
    ContinuityPlan,
    DisruptionScenario,
    ContinuityTest,
    CrisisEvent as ContinuityCrisisEvent,
    AlternativeArrangement,
    # Main class
    DORAICTBusinessContinuity,
    # Factory functions
    create_dora_ict_business_continuity,
)

# Article 16 - Simplified Framework
from services.dora.simplified_framework import (
    # Enums
    EntitySize,
    SimplifiedControlCategory,
    ControlStatus,
    RiskLevel as SimplifiedRiskLevel,
    IncidentPriority as SimplifiedIncidentPriority,
    IncidentStatus as SimplifiedIncidentStatus,
    TestType as SimplifiedTestType,
    # Data structures
    EligibilityCriteria,
    SimplifiedControl,
    SimplifiedRiskAssessment,
    SimplifiedIncident,
    SimplifiedBackup,
    SimplifiedThirdParty,
    SimplifiedTest,
    SimplifiedAwarenessTraining,
    AnnualReview,
    # Constants
    ESSENTIAL_CONTROLS,
    # Main class
    DORASimplifiedFramework,
    # Factory functions
    create_dora_simplified_framework,
)

# =============================================================================
# Phase 2 exports (ICT Incident Management & Reporting - Articles 17-23)
# =============================================================================

# Article 17 - Incident Management
from services.dora.incident_management import (
    # Enums
    ICTEventType,
    IncidentPhase,
    IncidentPriority as IncidentMgmtPriority,
    IncidentStatus as IncidentMgmtStatus,
    EscalationLevel as IncidentMgmtEscalationLevel,
    EarlyWarningType,
    # Data structures
    ICTEvent,
    DORAIncident,
    EarlyWarningIndicator,
    IncidentAction,
    EscalationRule,
    IncidentManagementConfig,
    # Main class
    DORAIncidentManagement,
    # Factory functions
    create_incident_management,
)

# Article 18 - Incident Classification
from services.dora.incident_classification import (
    # Enums
    IncidentClassificationType,
    ClientType,
    DataType as ClassificationDataType,
    CriticalServiceType,
    MajorIncidentTrigger,
    ReputationalImpactLevel,
    # Data structures
    ClassificationThresholds,
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
    IncidentClassificationConfig,
    # Main class
    DORAIncidentClassification,
    # Factory functions
    create_incident_classification,
)

# Article 19 - Incident Reporting
from services.dora.incident_reporting import (
    # Enums
    ReportType,
    ReportStatus,
    IncidentTypeCode,
    RootCauseCategory,
    CompetentAuthorityType,
    # Data structures
    CompetentAuthority as ReportingCompetentAuthority,
    InitialNotificationReport,
    IntermediateReport,
    FinalReport,
    ReportSubmission,
    IncidentReportingConfig,
    # Main class
    DORAIncidentReporter,
    # Factory functions
    create_incident_reporter,
)

# Article 19(4) - Cyber Threat Notification
from services.dora.cyber_threat_notification import (
    # Enums
    ThreatCategory,
    ThreatSeverity,
    ThreatStatus,
    ThreatActorType,
    ThreatSignificance,
    NotificationStatus as ThreatNotificationStatus,
    # Data structures
    ThreatIndicator,
    CyberThreat,
    ThreatSignificanceAssessment,
    ThreatNotification,
    CyberThreatNotificationConfig,
    # Main class
    CyberThreatNotificationService,
    # Factory functions
    create_cyber_threat_notification_service,
)

# Article 20 - Reporting Templates
from services.dora.reporting_templates import (
    # Enums
    IncidentTypeCode as TemplateIncidentTypeCode,
    DataTypeCode,
    ClientTypeCode,
    ServiceTypeCode,
    ResponseEffectivenessCode,
    # Data structures
    TimelineEvent,
    ITSInitialNotificationTemplate,
    ITSIntermediateReportTemplate,
    ITSFinalReportTemplate,
    # Main class
    DORAReportingTemplates,
    # Factory functions
    create_reporting_templates,
)

# Article 22 - Supervisory Feedback
from services.dora.supervisory_feedback import (
    # Enums
    FeedbackType,
    FeedbackPriority,
    FeedbackStatus,
    CorrectiveActionType,
    ResponseType,
    # Data structures
    CompetentAuthority as FeedbackCompetentAuthority,
    SupervisoryFeedback,
    CorrectiveAction,
    FeedbackResponse,
    FeedbackAuditEntry,
    AnonymisedInsight,
    # Main class
    DORASupervisioryFeedback,
    # Factory functions
    create_supervisory_feedback,
)

# Article 23 - Third-Party Incidents
from services.dora.third_party_incidents import (
    # Enums
    ThirdPartyProviderType,
    ThirdPartyCriticality,
    ThirdPartyIncidentType,
    IncidentSeverity as ThirdPartyIncidentSeverity,
    IncidentStatus as ThirdPartyIncidentStatus,
    ContractualSLAStatus,
    EscalationLevel as ThirdPartyEscalationLevel,
    CommunicationChannel as ThirdPartyCommunicationChannel,
    # Data structures
    ThirdPartyProvider as ThirdPartyProviderRecord,
    AffectedService,
    SLAAssessment,
    CommunicationRecord,
    EscalationRecord,
    MitigationAction,
    ThirdPartyIncident,
    PostIncidentReview as ThirdPartyPostIncidentReview,
    # Main class
    DORAThirdPartyIncidents,
    # Factory functions
    create_third_party_incidents,
)

# =============================================================================
# __all__ exports
# =============================================================================

__all__ = [
    # Version info
    "__version__",
    "__dora_compliance_phase__",

    # =========================================================================
    # Phase 0: Proportionality Assessment
    # =========================================================================

    # Scope Verification (Article 2)
    "DORAEntityType",
    "DORAScopeResult",
    "ScopeVerification",
    "EntityAuthorization",
    "DORAScope",
    "create_scope_verifier",
    "get_entity_type_description",

    # Function Classification (Article 3(22))
    "FunctionCriticality",
    "ImpairmentType",
    "FunctionClassification",
    "ICTService",
    "ThirdPartyProvider",
    "FunctionClassifier",
    "create_function_classifier",
    "get_platform_functions",
    "get_ict_providers",

    # Proportionality (Articles 4, 16)
    "DORARegime",
    "ExemptionType",
    "EntityClassification",
    "ProportionalityAssessment",
    "RegimeExemption",
    "ProportionalityAssessor",
    "create_proportionality_assessor",
    "assess_entity_proportionality",

    # =========================================================================
    # Phase 1: ICT Risk Management Framework (Articles 5-16)
    # =========================================================================

    # Governance (Article 5)
    "GovernanceRole",
    "DefenceLine",
    "TrainingStatus",
    "ApprovalStatus",
    "GovernanceRoleAssignment",
    "ICTTrainingRecord",
    "FrameworkApproval",
    "AuditFinding",
    "ICTBudgetAllocation",
    "DORAGovernanceFramework",
    "create_governance_framework",
    "MANDATORY_TRAINING_TOPICS",

    # ICT Risk Management Framework (Article 6)
    "PolicyCategory",
    "ControlDomain",
    "ControlType",
    "ICTRiskLevel",
    "RiskPolicy",
    "RiskProcedure",
    "ICTControl",
    "FrameworkReview",
    "ICTRisk",
    "DORAICTRiskFramework",
    "create_ict_risk_framework",

    # ICT Systems (Article 7)
    "SystemCriticality",
    "SystemType",
    "SystemStatus",
    "CapacityStatus",
    "AutomationLevel",
    "ICTSystem",
    "CapacityMetric",
    "ReliabilityMetric",
    "AutomationCapability",
    "SystemUpgrade",
    "DORAICTSystemsManager",
    "create_ict_systems_manager",

    # ICT Identification (Article 8)
    "AssetType",
    "AssetClassification",
    "RiskSourceCategory",
    "ThreatCategory",
    "VulnerabilitySeverity",
    "ICTAsset",
    "RiskSource",
    "CyberThreat",
    "ICTVulnerability",
    "ICTDependency",
    "BusinessFunction",
    "DORAICTIdentification",
    "create_ict_identification",

    # Protection and Prevention (Article 9)
    "SecurityControlCategory",
    "AccessControlType",
    "AuthenticationType",
    "EncryptionType",
    "NetworkZone",
    "SecurityControl",
    "AccessPolicy",
    "EncryptionStandard",
    "NetworkSecurityRule",
    "DataProtectionPolicy",
    "DORAProtection",
    "create_protection",

    # Detection (Article 10)
    "AnomalyType",
    "AlertSeverity",
    "AlertStatus",
    "DetectionMethod",
    "MonitoringStatus",
    "DetectionRule",
    "DetectionAlert",
    "PerformanceMetric",
    "SinglePointOfFailure",
    "DORADetection",
    "create_detection",

    # Response and Recovery (Article 11)
    "IncidentSeverity",
    "IncidentStatus",
    "IncidentCategory",
    "EscalationLevel",
    "CrisisStatus",
    "ICTIncident",
    "ResponseProcedure",
    "EscalationRule",
    "ResponseCrisisEvent",
    "RecoveryAction",
    "DORAResponseRecovery",
    "create_response_recovery",

    # Backup and Recovery (Article 12)
    "BackupType",
    "BackupFrequency",
    "BackupStatus",
    "RecoveryTestType",
    "RecoveryTestResult",
    "LocationType",
    "BackupPolicy",
    "BackupJob",
    "BackupLocation",
    "RecoveryTest",
    "RestorationProcedure",
    "DORABackupRecovery",
    "create_backup_recovery",

    # Learning and Evolving (Article 13)
    "ReviewType",
    "LessonCategory",
    "LessonPriority",
    "LessonStatus",
    "ImprovementType",
    "ImprovementStatus",
    "KnowledgeType",
    "PostIncidentReview",
    "LessonLearned",
    "ImprovementInitiative",
    "KnowledgeArticle",
    "TrainingNeed",
    "TrendAnalysis",
    "InformationShare",
    "DORALearning",
    "create_dora_learning",

    # Communication (Article 14)
    "CommunicationType",
    "CommunicationChannel",
    "CommunicationPriority",
    "CommunicationStatus",
    "NotificationType",
    "StakeholderType",
    "EscalationTrigger",
    "CommunicationPolicy",
    "Stakeholder",
    "CommunicationTemplate",
    "Communication",
    "CommunicationEscalationRule",
    "CrisisCommunicationPlan",
    "RegulatoryNotification",
    "CommunicationLog",
    "DORACommunication",
    "create_dora_communication",

    # ICT Business Continuity (Article 15)
    "ContinuityStatus",
    "CriticalityLevel",
    "ImpactCategory",
    "ImpactSeverity",
    "ContinuityTestType",
    "ContinuityTestResult",
    "ScenarioType",
    "RecoveryStrategy",
    "ICTBusinessContinuityPolicy",
    "ContinuityBusinessFunction",
    "BusinessImpactAssessment",
    "RecoveryObjective",
    "ContinuityPlan",
    "DisruptionScenario",
    "ContinuityTest",
    "ContinuityCrisisEvent",
    "AlternativeArrangement",
    "DORAICTBusinessContinuity",
    "create_dora_ict_business_continuity",

    # Simplified Framework (Article 16)
    "EntitySize",
    "SimplifiedControlCategory",
    "ControlStatus",
    "SimplifiedRiskLevel",
    "SimplifiedIncidentPriority",
    "SimplifiedIncidentStatus",
    "SimplifiedTestType",
    "EligibilityCriteria",
    "SimplifiedControl",
    "SimplifiedRiskAssessment",
    "SimplifiedIncident",
    "SimplifiedBackup",
    "SimplifiedThirdParty",
    "SimplifiedTest",
    "SimplifiedAwarenessTraining",
    "AnnualReview",
    "ESSENTIAL_CONTROLS",
    "DORASimplifiedFramework",
    "create_dora_simplified_framework",

    # =========================================================================
    # Phase 2: ICT Incident Management & Reporting (Articles 17-23)
    # =========================================================================

    # Incident Management (Article 17)
    "ICTEventType",
    "IncidentPhase",
    "IncidentMgmtPriority",
    "IncidentMgmtStatus",
    "IncidentMgmtEscalationLevel",
    "EarlyWarningType",
    "ICTEvent",
    "DORAIncident",
    "EarlyWarningIndicator",
    "IncidentAction",
    "EscalationRule",
    "IncidentManagementConfig",
    "DORAIncidentManagement",
    "create_incident_management",

    # Incident Classification (Article 18)
    "IncidentClassificationType",
    "ClientType",
    "ClassificationDataType",
    "CriticalServiceType",
    "MajorIncidentTrigger",
    "ReputationalImpactLevel",
    "ClassificationThresholds",
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
    "IncidentClassificationConfig",
    "DORAIncidentClassification",
    "create_incident_classification",

    # Incident Reporting (Article 19)
    "ReportType",
    "ReportStatus",
    "IncidentTypeCode",
    "RootCauseCategory",
    "CompetentAuthorityType",
    "ReportingCompetentAuthority",
    "InitialNotificationReport",
    "IntermediateReport",
    "FinalReport",
    "ReportSubmission",
    "IncidentReportingConfig",
    "DORAIncidentReporter",
    "create_incident_reporter",

    # Cyber Threat Notification (Article 19(4))
    "ThreatCategory",
    "ThreatSeverity",
    "ThreatStatus",
    "ThreatActorType",
    "ThreatSignificance",
    "ThreatNotificationStatus",
    "ThreatIndicator",
    "CyberThreat",
    "ThreatSignificanceAssessment",
    "ThreatNotification",
    "CyberThreatNotificationConfig",
    "CyberThreatNotificationService",
    "create_cyber_threat_notification_service",

    # Reporting Templates (Article 20)
    "TemplateIncidentTypeCode",
    "DataTypeCode",
    "ClientTypeCode",
    "ServiceTypeCode",
    "ResponseEffectivenessCode",
    "TimelineEvent",
    "ITSInitialNotificationTemplate",
    "ITSIntermediateReportTemplate",
    "ITSFinalReportTemplate",
    "DORAReportingTemplates",
    "create_reporting_templates",

    # Supervisory Feedback (Article 22)
    "FeedbackType",
    "FeedbackPriority",
    "FeedbackStatus",
    "CorrectiveActionType",
    "ResponseType",
    "FeedbackCompetentAuthority",
    "SupervisoryFeedback",
    "CorrectiveAction",
    "FeedbackResponse",
    "FeedbackAuditEntry",
    "AnonymisedInsight",
    "DORASupervisioryFeedback",
    "create_supervisory_feedback",

    # Third-Party Incidents (Article 23)
    "ThirdPartyProviderType",
    "ThirdPartyCriticality",
    "ThirdPartyIncidentType",
    "ThirdPartyIncidentSeverity",
    "ThirdPartyIncidentStatus",
    "ContractualSLAStatus",
    "ThirdPartyEscalationLevel",
    "ThirdPartyCommunicationChannel",
    "ThirdPartyProviderRecord",
    "AffectedService",
    "SLAAssessment",
    "CommunicationRecord",
    "EscalationRecord",
    "MitigationAction",
    "ThirdPartyIncident",
    "ThirdPartyPostIncidentReview",
    "DORAThirdPartyIncidents",
    "create_third_party_incidents",
]
