# -*- coding: utf-8 -*-
"""
Archived DORA Financial Entity Modules.

These modules implement DORA requirements for FINANCIAL ENTITIES (Art. 2),
NOT for ICT Third-Party Service Providers (Art. 30).

IMPORTANT:
    As an ICT service provider, we:
    - Comply with Art. 30 (contractual requirements)
    - Support client due diligence (Art. 28)
    - DO NOT implement full FE DORA framework ourselves

    These modules are archived for reference and potential product development
    for financial entity customers.

Active DORA code lives in:
    - services/core/           - Operational resilience
    - services/dora_integration/ - Client-facing interfaces

When to Use These Modules:
    If building a product FOR financial entities to manage their own
    DORA compliance, these modules provide a reference implementation.

Migration Status: Phase 7 Complete - All FE modules archived
Migration Date: 2025-01-17

Usage:
    # Import with deprecation warning
    from services.archive.dora_financial_entity import DORAScope

    # Or import directly without warning
    from services.archive.dora_financial_entity.scope_verification import DORAScope
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

__version__ = "1.0.0"
__archive_date__ = "2025-01-17"


def _emit_deprecation_warning(module_name: str) -> None:
    """Emit deprecation warning for archived module access."""
    warnings.warn(
        f"Importing from services.archive.dora_financial_entity is accessing "
        f"archived DORA Financial Entity modules. These modules are for "
        f"building products for financial entities, not for ICT provider use. "
        f"Module: {module_name}",
        DeprecationWarning,
        stacklevel=3
    )


# =============================================================================
# Phase 0: Proportionality Assessment (Articles 2-4, 16)
# =============================================================================

# Scope Verification (Article 2)
from services.archive.dora_financial_entity.scope_verification import (
    DORAEntityType,
    DORAScopeResult,
    ScopeVerification,
    EntityAuthorization,
    DORAScope,
    create_scope_verifier,
    get_entity_type_description,
)

# Function Classification (Article 3(22))
from services.archive.dora_financial_entity.function_classification import (
    FunctionCriticality,
    ImpairmentType,
    FunctionClassification,
    ICTService,
    ThirdPartyProvider,
    FunctionClassifier,
    create_function_classifier,
    get_platform_functions,
    get_ict_providers,
)

# Proportionality (Articles 4, 16)
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

# =============================================================================
# Phase 1: ICT Risk Management Framework (Articles 5-16)
# =============================================================================

# Governance (Article 5)
from services.archive.dora_financial_entity.governance import (
    GovernanceRole,
    DefenceLine,
    TrainingStatus,
    ApprovalStatus,
    GovernanceRoleAssignment,
    ICTTrainingRecord,
    FrameworkApproval,
    AuditFinding,
    ICTBudgetAllocation,
    DORAGovernanceFramework,
    create_governance_framework,
    MANDATORY_TRAINING_TOPICS,
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

# ICT Systems (Article 7)
from services.archive.dora_financial_entity.ict_systems import (
    SystemCriticality,
    SystemType,
    SystemStatus,
    CapacityStatus,
    AutomationLevel,
    ICTSystem,
    CapacityMetric,
    ReliabilityMetric,
    AutomationCapability,
    SystemUpgrade,
    DORAICTSystemsManager,
    create_ict_systems_manager,
)

# ICT Identification (Article 8)
from services.archive.dora_financial_entity.ict_identification import (
    AssetType,
    AssetClassification,
    RiskSourceCategory,
    ThreatCategory,
    VulnerabilitySeverity,
    ICTAsset,
    RiskSource,
    CyberThreat,
    ICTVulnerability,
    ICTDependency,
    BusinessFunction,
    DORAICTIdentification,
    create_ict_identification,
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

# Response and Recovery (Article 11)
from services.archive.dora_financial_entity.response_recovery import (
    IncidentSeverity,
    IncidentStatus,
    IncidentCategory,
    EscalationLevel,
    CrisisStatus,
    ICTIncident,
    ResponseProcedure,
    EscalationRule,
    RecoveryAction,
    DORAResponseRecovery,
    create_response_recovery,
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
    ImprovementStatus,
    KnowledgeType,
    PostIncidentReview,
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
    ControlStatus,
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

# =============================================================================
# Phase 2: ICT Incident Management & Reporting (Articles 17-23)
# =============================================================================

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

# Supervisory Feedback (Article 22)
from services.archive.dora_financial_entity.supervisory_feedback import (
    FeedbackType,
    FeedbackPriority,
    FeedbackStatus,
    CorrectiveActionType,
    ResponseType,
    SupervisoryFeedback,
    CorrectiveAction,
    FeedbackResponse,
    FeedbackAuditEntry,
    AnonymisedInsight,
    DORASupervisioryFeedback,
    create_supervisory_feedback,
)

# =============================================================================
# Phase 3: Digital Resilience Testing (Articles 24-27)
# =============================================================================

# Resilience Testing Programme (Article 24)
from services.archive.dora_financial_entity.resilience_testing import (
    TestCategory,
    TestFrequency,
    FindingSeverity,
    FindingStatus,
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
    RemediationStatus,
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
    ParticipantStatus,
    CostSharingModel,
    ProviderCriticality,
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

# =============================================================================
# Phase 5: Information Sharing & Integration
# =============================================================================

# Cross Regulation Integration
from services.archive.dora_financial_entity.cross_regulation import (
    Regulation,
    ReportingRequirement,
    IncidentAlignmentResult,
    RiskFrameworkAlignment,
    LoggingAlignmentResult,
    DORARegulationIntegration,
)

# Training Participation (Article 30(2)(i) - FE requesting training from providers)
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
# __all__ exports
# =============================================================================

__all__ = [
    # Version info
    "__version__",
    "__archive_date__",

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

    # ICT Business Continuity (Article 15)
    "ContinuityStatus",
    "CriticalityLevel",
    "ImpactCategory",
    "ImpactSeverity",
    "ScenarioType",
    "RecoveryStrategy",
    "ICTBusinessContinuityPolicy",
    "BusinessImpactAssessment",
    "RecoveryObjective",
    "ContinuityPlan",
    "DisruptionScenario",
    "ContinuityTest",
    "AlternativeArrangement",
    "DORAICTBusinessContinuity",
    "create_dora_ict_business_continuity",

    # Simplified Framework (Article 16)
    "EntitySize",
    "SimplifiedControlCategory",
    "ControlStatus",
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
    # Phase 2: ICT Incident Management (Articles 17-23)
    # =========================================================================

    # Incident Management (Article 17)
    "ICTEventType",
    "IncidentPhase",
    "EarlyWarningType",
    "ICTEvent",
    "DORAIncident",
    "EarlyWarningIndicator",
    "IncidentAction",
    "IncidentManagementConfig",
    "DORAIncidentManagement",
    "create_incident_management",

    # Supervisory Feedback (Article 22)
    "FeedbackType",
    "FeedbackPriority",
    "FeedbackStatus",
    "CorrectiveActionType",
    "ResponseType",
    "SupervisoryFeedback",
    "CorrectiveAction",
    "FeedbackResponse",
    "FeedbackAuditEntry",
    "AnonymisedInsight",
    "DORASupervisioryFeedback",
    "create_supervisory_feedback",

    # =========================================================================
    # Phase 3: Digital Resilience Testing (Articles 24-27)
    # =========================================================================

    # Resilience Testing (Article 24)
    "TestCategory",
    "TestFrequency",
    "FindingSeverity",
    "FindingStatus",
    "TestScope",
    "TestDefinition",
    "TestExecution",
    "TestFinding",
    "TestingProgramme",
    "TestingCycle",
    "ResilienceTestingConfig",
    "DORAResilienceTestingProgramme",
    "create_resilience_testing_programme",

    # ICT Testing (Article 25)
    "ICTSystemType",
    "TestingPriority",
    "VulnerabilityStatus",
    "RemediationStatus",
    "ICTSystemProfile",
    "SystemTestPlan",
    "SystemTest",
    "Vulnerability",
    "RemediationPlan",
    "ThirdPartyInterfaceTest",
    "ICTTestingConfig",
    "DORAICTSystemTesting",
    "create_ict_system_testing",

    # TLPT (Article 26)
    "TLPTPhase",
    "TLPTStatus",
    "ThreatActorCapability",
    "AttackTechnique",
    "AttackOutcome",
    "TLPTFindingSeverity",
    "FindingCategory",
    "TLPTScope",
    "ThreatIntelligenceReport",
    "RedTeamScenario",
    "AttackAction",
    "TLPTFinding",
    "PurpleTeamSession",
    "TLPTEngagement",
    "TLPTAttestation",
    "TLPTConfig",
    "DORAThreadLedPenetrationTesting",
    "create_tlpt",

    # Tester Management (Article 27)
    "TesterRole",
    "CertificationCategory",
    "QualificationStatus",
    "ConflictCheckResult",
    "SecurityCertification",
    "TesterExpertise",
    "ConflictOfInterestDeclaration",
    "ProfessionalIndemnityInsurance",
    "TLPTTester",
    "TesterOrganization",
    "TesterQualificationAssessment",
    "InternalTesterApproval",
    "TesterManagementConfig",
    "DORATestermanagement",
    "create_tester_management",

    # Pooled Testing (Article 26(3))
    "PooledTestStatus",
    "ParticipantRole",
    "ParticipantStatus",
    "CostSharingModel",
    "ProviderCriticality",
    "SharedProvider",
    "PooledTestingParticipant",
    "PooledTestingScope",
    "CostSharingAgreement",
    "PooledTestingEngagement",
    "PooledTestingResults",
    "PooledTestingConfig",
    "DORAPooledTesting",
    "create_pooled_testing",

    # =========================================================================
    # Phase 5: Information Sharing & Integration
    # =========================================================================

    # Cross Regulation
    "Regulation",
    "ReportingRequirement",
    "IncidentAlignmentResult",
    "RiskFrameworkAlignment",
    "LoggingAlignmentResult",
    "DORARegulationIntegration",

    # Training Participation (FE side - requesting training from providers)
    "TrainingType",
    "ParticipationMode",
    "PersonnelRole",
    "TrainingCommitment",
    "TrainingRequest",
    "TrainingSession",
    "QuarterlyUsage",
    "TrainingParticipationConfig",
    "DORATrainingParticipation",
]
