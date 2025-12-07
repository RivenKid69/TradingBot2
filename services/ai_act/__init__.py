# -*- coding: utf-8 -*-
"""
AI Act Compliance Module for TradingBot2.

EU AI Act (Regulation (EU) 2024/1689) Compliance Implementation.

This package provides compliance tools for high-risk AI systems in algorithmic trading:

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

Classification:
    This system is classified as HIGH-RISK AI SYSTEM per Annex III
    (algorithmic trading in financial services context).

Compliance Deadline: August 2, 2026

References:
    - EU AI Act: https://artificialintelligenceact.eu/
    - Article 9 (Risk Management): https://artificialintelligenceact.eu/article/9/
    - Article 10 (Data Governance): https://artificialintelligenceact.eu/article/10/
    - Article 12 (Record-Keeping): https://artificialintelligenceact.eu/article/12/
    - Article 14 (Human Oversight): https://artificialintelligenceact.eu/article/14/
    - Article 15 (Accuracy, Robustness, Cybersecurity): https://artificialintelligenceact.eu/article/15/
"""

from __future__ import annotations

__version__ = "3.0.0"
__ai_act_compliance_phase__ = 3  # Current implementation phase

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
]
