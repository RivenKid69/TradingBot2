# -*- coding: utf-8 -*-
"""
AI Act Compliance Module for TradingBot2.

EU AI Act (Regulation (EU) 2024/1689) Compliance Implementation.

This package provides compliance tools for AI systems in algorithmic trading.

Classification:
    - Primary: General-Purpose AI Model Provider (GPAI) - Article 53
    - Secondary: Limited Risk AI System - Article 50 (Transparency)
    - Voluntary: HIGH-RISK compliance maintained for enterprise readiness

Note: Algorithmic trading is NOT included in Annex III (HIGH-RISK).
      HIGH-RISK compliance is maintained as overcompliance for enterprise clients.

=== HIGH-RISK Compliance (Voluntary Overcompliance) ===

Phase 1 - Foundation & Risk Management (Articles 9, 14, 15):
    - risk_management: Risk Management System (Article 9)
    - risk_registry: Risk Registry and Tracking
    - human_oversight: Human Oversight System (Article 14)
    - explainability: Decision Explainability Module
    - accuracy_metrics: Accuracy Metrics Declaration (Article 15)
    - robustness_testing: Robustness Testing Framework (Article 15)

Phase 2 - Technical Documentation & Logging (Articles 10, 11, 12):
    - logging_system: AI Act Compliant Logging (Article 12)
    - data_governance: Data Governance Framework (Article 10)
    - data_lineage: Data Lineage Tracking

Phase 3 - Quality Management System (Article 17):
    - qms: Quality Management System
    - testing_framework: Testing Framework (Article 9)
    - cybersecurity: Cybersecurity Measures (Article 15)
    - post_market_monitoring: Post-Market Monitoring (Article 72)

Phase 4 - Conformity Assessment & Deployment (Articles 43, 47, 49):
    - conformity_assessment: Self-Assessment and Checklist (Article 43, Annex VI)
    - EU Declaration of Conformity generation (Article 47)
    - Instructions for Use generation (Article 13)
    - EU Database Registration preparation (Article 49)

=== Software Provider / GPAI Compliance (Mandatory) ===

GPAI Phase 1 - Core Requirements (Articles 50, 53):
    - transparency_disclosure: AI Interaction Disclosure (Article 50)
    - copyright_compliance: Copyright Policy for Training Data (Article 53(1)(c))
    - training_data_summary: Public Training Data Summary (Article 53(1)(d))

GPAI Phase 2 - Documentation & User Acknowledgment (Articles 50, 53):
    - gpai_model_card: GPAI Model Card for Downstream Providers (Article 53(1)(b))
    - user_acknowledgment: User AI Acknowledgment System (Article 50)

Compliance Deadline: August 2, 2026

References:
    - EU AI Act: https://artificialintelligenceact.eu/
    - Article 50 (Transparency): https://artificialintelligenceact.eu/article/50/
    - Article 53 (GPAI Obligations): https://artificialintelligenceact.eu/article/53/
    - Article 9 (Risk Management): https://artificialintelligenceact.eu/article/9/
    - Article 10 (Data Governance): https://artificialintelligenceact.eu/article/10/
    - Article 12 (Record-Keeping): https://artificialintelligenceact.eu/article/12/
    - Article 14 (Human Oversight): https://artificialintelligenceact.eu/article/14/
    - Article 15 (Accuracy, Robustness, Cybersecurity): https://artificialintelligenceact.eu/article/15/
    - Article 43 (Conformity Assessment): https://artificialintelligenceact.eu/article/43/
    - Article 47 (EU Declaration of Conformity): https://artificialintelligenceact.eu/article/47/
    - Article 49 (Registration): https://artificialintelligenceact.eu/article/49/
    - Annex VI (Internal Control): https://artificialintelligenceact.eu/annex/6/
    - DSM Directive 2019/790 (Copyright): https://eur-lex.europa.eu/eli/dir/2019/790/oj
    - GPAI Code of Practice: https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai
"""

from __future__ import annotations

__version__ = "4.2.0"
__ai_act_compliance_phase__ = 4  # Current implementation phase
__gpai_compliance_version__ = "2.0.0"  # Software Provider / GPAI compliance (Phase 2 complete)

# =============================================================================
# Phase 1 exports (Foundation & Risk Management)
# =============================================================================

from services.ai_act.risk_management import (
    AIActRiskCategory,
    AIActRiskSeverity,
    AIActRiskLikelihood,
    RiskIdentification,
    RiskAssessment,
    RiskMitigation,
    AIActRiskManager,
    AIActRiskConfig,
    create_risk_manager,
)

from services.ai_act.risk_registry import (
    RiskEntry,
    RiskStatus,
    RiskRegistry,
    create_risk_registry,
    get_default_trading_risks,
)

from services.ai_act.human_oversight import (
    OversightLevel,
    OversightCapability,
    HumanOversightConfig,
    HumanOversightSystem,
    AnomalyDetector,
    ManualOverrideController,
    AutomationBiasMonitor,
    create_human_oversight_system,
)

from services.ai_act.explainability import (
    ExplanationType,
    FeatureContribution,
    DecisionExplanation,
    CounterfactualExplanation,
    DecisionExplainer,
    create_decision_explainer,
)

from services.ai_act.accuracy_metrics import (
    MetricType,
    AccuracyMetric,
    DeclaredAccuracyMetrics,
    AccuracyMonitor,
    create_accuracy_monitor,
    get_default_trading_metrics,
)

from services.ai_act.robustness_testing import (
    RobustnessTestType,
    RobustnessTestResult,
    RobustnessTestSuite,
    AdversarialTester,
    DistributionShiftTester,
    FailsafeTester,
    create_robustness_test_suite,
)

# =============================================================================
# Phase 2 exports (Technical Documentation & Logging)
# =============================================================================

from services.ai_act.logging_system import (
    # Event types
    AIActLogEventType,
    AIActLogCategory,
    # Data structures
    AIActLogEvent,
    LogSession,
    # Configuration
    AIActLoggingConfig,
    # Main classes
    AIActLogger,
    LogRetentionManager,
    LogIntegrityVerifier,
    AIActEventBus,
    # Factory functions
    create_ai_act_logger,
    create_ai_act_event_bus,
)

from services.ai_act.data_governance import (
    # Enums
    DataQualityDimension,
    BiasType,
    DataGapType,
    DatasetRole,
    # Data structures
    QualityCheckResult,
    BiasCheckResult,
    DataGap,
    DataQualityReport,
    DatasetMetadata,
    # Configuration
    DataGovernanceConfig,
    # Main classes
    DataQualityAssessor,
    BiasDetector,
    DataGapAnalyzer,
    DataValidator,
    DataGovernanceFramework,
    # Factory functions
    create_data_governance_framework,
    create_bias_detector,
)

from services.ai_act.data_lineage import (
    # Enums
    DataNodeType,
    TransformationType,
    # Data structures
    DataNode,
    DataTransformation,
    LineageEdge,
    # Configuration
    DataLineageConfig,
    # Main classes
    LineageGraph,
    DataLineageTracker,
    # Factory functions
    create_data_lineage_tracker,
)

from services.ai_act.technical_documentation import (
    # Enums
    DocumentationSectionType,
    ComplianceStatus,
    ExportFormat,
    # Data structures
    DocumentationMetadata,
    ComplianceEvidence,
    DocumentationSection,
    ChangeRecord,
    # Configuration
    TechnicalDocumentationConfig,
    # Main class
    TechnicalDocumentationGenerator,
    # Factory function
    create_technical_documentation_generator,
)

# =============================================================================
# Phase 3 exports (QMS & Testing)
# =============================================================================

from services.ai_act.qms import (
    # Enums
    QMSElementType,
    ProcedureStatus,
    AuditType,
    AuditStatus,
    FindingSeverity,
    ChangeType,
    ChangeImpact,
    CAPAType,
    CAPAStatus,
    # Data structures
    QMSProcedure,
    AuditFinding,
    QMSAudit,
    DesignReview,
    ChangeRequest,
    CAPARecord,
    ResourceRecord,
    AccountabilityRecord,
    # Configuration
    QMSConfig,
    # Main class
    QualityManagementSystem,
    # Factory functions
    create_qms,
    get_default_qms_procedures,
)

from services.ai_act.testing_framework import (
    # Enums
    TestCategory,
    TestPriority,
    TestStatus,
    MetricType as TestMetricType,
    ComparisonOperator,
    VulnerableGroup,
    # Data structures
    TestMetric,
    TestScenario,
    TestExecution,
    TestSuite,
    RealWorldTestPlan,
    # Configuration
    TestingConfig,
    # Main class
    AIActTestingFramework,
    # Factory functions
    create_testing_framework,
    get_default_test_metrics,
)

from services.ai_act.cybersecurity import (
    # Enums
    ThreatType,
    ThreatSeverity,
    ValidationStatus,
    AccessLevel,
    SecurityEventType,
    # Data structures
    SecurityThreat,
    ValidationResult,
    IntegrityRecord,
    AccessRecord,
    SecurityPolicy,
    SecurityEvent,
    # Configuration
    CybersecurityConfig,
    # Components
    InputValidator,
    ModelIntegrityVerifier,
    AdversarialDetector,
    DataPoisoningDetector,
    AccessControlManager,
    # Main class
    AIActCybersecurity,
    # Factory functions
    create_cybersecurity,
    get_default_security_policies,
)

from services.ai_act.post_market_monitoring import (
    # Enums
    MonitoringMetricType,
    DriftType,
    DriftSeverity,
    IncidentSeverity,
    IncidentStatus,
    FeedbackType,
    AlertPriority,
    # Data structures
    MonitoringMetric,
    DriftDetectionResult,
    Incident,
    Feedback,
    MonitoringAlert,
    PeriodicReport,
    # Configuration
    PostMarketConfig,
    # Components
    PerformanceMonitor,
    IncidentTracker,
    FeedbackCollector,
    # Main class
    PostMarketMonitoringSystem,
    # Factory functions
    create_post_market_monitoring,
    get_default_monitoring_metrics,
)

# =============================================================================
# Phase 4 exports (Conformity Assessment & Deployment)
# =============================================================================

from services.ai_act.conformity_assessment import (
    # Enums
    ConformityStatus,
    ChecklistItemStatus,
    RequirementCategory,
    AssessmentPhase,
    GapSeverity,
    # Data structures
    ChecklistItem,
    ComplianceGap,
    AssessmentReport,
    EUDeclaration,
    InstructionsForUse,
    RegistrationInfo,
    # Configuration
    ConformityAssessmentConfig,
    # Main class
    ConformitySelfAssessment,
    # Factory functions
    create_conformity_assessment,
    get_default_checklist,
)

# =============================================================================
# Software Provider / GPAI Compliance (Articles 50, 53)
# =============================================================================

from services.ai_act.transparency_disclosure import (
    # Enums
    DisclosureType,
    DisclosureContext,
    DisclosureLanguage,
    # Data structures
    AIDisclosure,
    DisclosureRequirement,
    DisclosureAuditRecord,
    # Constants
    DISCLOSURE_REQUIREMENTS,
    # Main class
    TransparencyDisclosureManager,
    SyntheticContentMarker,
    # Factory functions
    create_transparency_manager,
    get_disclosure_requirements,
    validate_disclosure_text,
)

from services.ai_act.copyright_compliance import (
    # Enums
    DataSourceType,
    CopyrightStatus,
    OptOutMechanism,
    # Data structures
    DataSourceRecord,
    OptOutCheck,
    RightsHolderRequest,
    # Constants
    DEFAULT_DATA_SOURCES,
    # Main class
    CopyrightComplianceManager,
    # Factory functions
    create_copyright_manager,
    get_default_data_sources,
    validate_source_record,
)

from services.ai_act.training_data_summary import (
    # Enums
    DataCategory,
    DataQualityLevel,
    # Data structures
    DatasetInfo,
    TrainingDataSummary,
    # Main class
    TrainingDataSummaryManager,
    # Factory functions
    create_default_summary,
    create_summary_manager,
    get_data_categories,
    validate_dataset_info,
)

# =============================================================================
# GPAI Phase 2: Model Card & User Acknowledgment (Articles 50, 53(1)(b))
# =============================================================================

from services.ai_act.gpai_model_card import (
    # Enums
    IntendedUse,
    LimitationType,
    RiskLevel,
    EvaluationDataset,
    # Data structures
    ModelLimitation,
    PerformanceMetric,
    BiasAssessment,
    EthicalConsideration,
    DownstreamRequirement,
    GPAIModelCard,
    # Main class
    ModelCardManager,
    # Factory functions
    create_default_model_card,
    create_model_card_manager,
    get_default_limitations,
    get_default_biases,
    get_default_downstream_requirements,
    validate_model_card,
)

from services.ai_act.user_acknowledgment import (
    # Enums
    AcknowledgmentType,
    AcknowledgmentStatus,
    FeatureCategory,
    # Data structures
    UserAcknowledgment,
    AcknowledgmentAuditRecord,
    # Constants
    ACKNOWLEDGMENT_TEXTS,
    FEATURE_REQUIREMENTS,
    # Main class
    UserAcknowledgmentManager,
    # Factory functions
    create_acknowledgment_manager,
    get_acknowledgment_texts,
    get_feature_requirements as get_acknowledgment_feature_requirements,
    validate_acknowledgment,
    get_acknowledgment_summary,
)

# =============================================================================
# __all__ exports
# =============================================================================

__all__ = [
    # Version info
    "__version__",
    "__ai_act_compliance_phase__",

    # =========================================================================
    # Phase 1: Foundation & Risk Management
    # =========================================================================

    # Risk Management (Article 9)
    "AIActRiskCategory",
    "AIActRiskSeverity",
    "AIActRiskLikelihood",
    "RiskIdentification",
    "RiskAssessment",
    "RiskMitigation",
    "AIActRiskManager",
    "AIActRiskConfig",
    "create_risk_manager",

    # Risk Registry
    "RiskEntry",
    "RiskStatus",
    "RiskRegistry",
    "create_risk_registry",
    "get_default_trading_risks",

    # Human Oversight (Article 14)
    "OversightLevel",
    "OversightCapability",
    "HumanOversightConfig",
    "HumanOversightSystem",
    "AnomalyDetector",
    "ManualOverrideController",
    "AutomationBiasMonitor",
    "create_human_oversight_system",

    # Explainability
    "ExplanationType",
    "FeatureContribution",
    "DecisionExplanation",
    "CounterfactualExplanation",
    "DecisionExplainer",
    "create_decision_explainer",

    # Accuracy Metrics (Article 15)
    "MetricType",
    "AccuracyMetric",
    "DeclaredAccuracyMetrics",
    "AccuracyMonitor",
    "create_accuracy_monitor",
    "get_default_trading_metrics",

    # Robustness Testing (Article 15)
    "RobustnessTestType",
    "RobustnessTestResult",
    "RobustnessTestSuite",
    "AdversarialTester",
    "DistributionShiftTester",
    "FailsafeTester",
    "create_robustness_test_suite",

    # =========================================================================
    # Phase 2: Technical Documentation & Logging
    # =========================================================================

    # Logging System (Article 12)
    "AIActLogEventType",
    "AIActLogCategory",
    "AIActLogEvent",
    "LogSession",
    "AIActLoggingConfig",
    "AIActLogger",
    "LogRetentionManager",
    "LogIntegrityVerifier",
    "AIActEventBus",
    "create_ai_act_logger",
    "create_ai_act_event_bus",

    # Data Governance (Article 10)
    "DataQualityDimension",
    "BiasType",
    "DataGapType",
    "DatasetRole",
    "QualityCheckResult",
    "BiasCheckResult",
    "DataGap",
    "DataQualityReport",
    "DatasetMetadata",
    "DataGovernanceConfig",
    "DataQualityAssessor",
    "BiasDetector",
    "DataGapAnalyzer",
    "DataValidator",
    "DataGovernanceFramework",
    "create_data_governance_framework",
    "create_bias_detector",

    # Data Lineage
    "DataNodeType",
    "TransformationType",
    "DataNode",
    "DataTransformation",
    "LineageEdge",
    "DataLineageConfig",
    "LineageGraph",
    "DataLineageTracker",
    "create_data_lineage_tracker",

    # Technical Documentation (Article 11, Annex IV)
    "DocumentationSectionType",
    "ComplianceStatus",
    "ExportFormat",
    "DocumentationMetadata",
    "ComplianceEvidence",
    "DocumentationSection",
    "ChangeRecord",
    "TechnicalDocumentationConfig",
    "TechnicalDocumentationGenerator",
    "create_technical_documentation_generator",

    # =========================================================================
    # Phase 3: QMS & Testing
    # =========================================================================

    # QMS (Article 17)
    "QMSElementType",
    "ProcedureStatus",
    "AuditType",
    "AuditStatus",
    "FindingSeverity",
    "ChangeType",
    "ChangeImpact",
    "CAPAType",
    "CAPAStatus",
    "QMSProcedure",
    "AuditFinding",
    "QMSAudit",
    "DesignReview",
    "ChangeRequest",
    "CAPARecord",
    "ResourceRecord",
    "AccountabilityRecord",
    "QMSConfig",
    "QualityManagementSystem",
    "create_qms",
    "get_default_qms_procedures",

    # Testing Framework (Article 9)
    "TestCategory",
    "TestPriority",
    "TestStatus",
    "TestMetricType",
    "ComparisonOperator",
    "VulnerableGroup",
    "TestMetric",
    "TestScenario",
    "TestExecution",
    "TestSuite",
    "RealWorldTestPlan",
    "TestingConfig",
    "AIActTestingFramework",
    "create_testing_framework",
    "get_default_test_metrics",

    # Cybersecurity (Article 15(5))
    "ThreatType",
    "ThreatSeverity",
    "ValidationStatus",
    "AccessLevel",
    "SecurityEventType",
    "SecurityThreat",
    "ValidationResult",
    "IntegrityRecord",
    "AccessRecord",
    "SecurityPolicy",
    "SecurityEvent",
    "CybersecurityConfig",
    "InputValidator",
    "ModelIntegrityVerifier",
    "AdversarialDetector",
    "DataPoisoningDetector",
    "AccessControlManager",
    "AIActCybersecurity",
    "create_cybersecurity",
    "get_default_security_policies",

    # Post-Market Monitoring (Article 72)
    "MonitoringMetricType",
    "DriftType",
    "DriftSeverity",
    "IncidentSeverity",
    "IncidentStatus",
    "FeedbackType",
    "AlertPriority",
    "MonitoringMetric",
    "DriftDetectionResult",
    "Incident",
    "Feedback",
    "MonitoringAlert",
    "PeriodicReport",
    "PostMarketConfig",
    "PerformanceMonitor",
    "IncidentTracker",
    "FeedbackCollector",
    "PostMarketMonitoringSystem",
    "create_post_market_monitoring",
    "get_default_monitoring_metrics",

    # =========================================================================
    # Phase 4: Conformity Assessment & Deployment
    # =========================================================================

    # Conformity Assessment (Article 43, Annex VI)
    "ConformityStatus",
    "ChecklistItemStatus",
    "RequirementCategory",
    "AssessmentPhase",
    "GapSeverity",
    "ChecklistItem",
    "ComplianceGap",
    "AssessmentReport",
    "EUDeclaration",
    "InstructionsForUse",
    "RegistrationInfo",
    "ConformityAssessmentConfig",
    "ConformitySelfAssessment",
    "create_conformity_assessment",
    "get_default_checklist",

    # =========================================================================
    # Software Provider / GPAI Compliance (Articles 50, 53)
    # =========================================================================

    # Transparency Disclosure (Article 50)
    "DisclosureType",
    "DisclosureContext",
    "DisclosureLanguage",
    "AIDisclosure",
    "DisclosureRequirement",
    "DisclosureAuditRecord",
    "DISCLOSURE_REQUIREMENTS",
    "TransparencyDisclosureManager",
    "SyntheticContentMarker",
    "create_transparency_manager",
    "get_disclosure_requirements",
    "validate_disclosure_text",

    # Copyright Compliance (Article 53(1)(c))
    "DataSourceType",
    "CopyrightStatus",
    "OptOutMechanism",
    "DataSourceRecord",
    "OptOutCheck",
    "RightsHolderRequest",
    "DEFAULT_DATA_SOURCES",
    "CopyrightComplianceManager",
    "create_copyright_manager",
    "get_default_data_sources",
    "validate_source_record",

    # Training Data Summary (Article 53(1)(d))
    "DataCategory",
    "DataQualityLevel",
    "DatasetInfo",
    "TrainingDataSummary",
    "TrainingDataSummaryManager",
    "create_default_summary",
    "create_summary_manager",
    "get_data_categories",
    "validate_dataset_info",

    # =========================================================================
    # GPAI Phase 2: Model Card & User Acknowledgment
    # =========================================================================

    # GPAI Model Card (Article 53(1)(b))
    "IntendedUse",
    "LimitationType",
    "RiskLevel",
    "EvaluationDataset",
    "ModelLimitation",
    "PerformanceMetric",
    "BiasAssessment",
    "EthicalConsideration",
    "DownstreamRequirement",
    "GPAIModelCard",
    "ModelCardManager",
    "create_default_model_card",
    "create_model_card_manager",
    "get_default_limitations",
    "get_default_biases",
    "get_default_downstream_requirements",
    "validate_model_card",

    # User Acknowledgment (Article 50)
    "AcknowledgmentType",
    "AcknowledgmentStatus",
    "FeatureCategory",
    "UserAcknowledgment",
    "AcknowledgmentAuditRecord",
    "ACKNOWLEDGMENT_TEXTS",
    "FEATURE_REQUIREMENTS",
    "UserAcknowledgmentManager",
    "create_acknowledgment_manager",
    "get_acknowledgment_texts",
    "get_acknowledgment_feature_requirements",
    "validate_acknowledgment",
    "get_acknowledgment_summary",
]
