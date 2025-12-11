# -*- coding: utf-8 -*-
"""
DEPRECATED: Backward Compatibility Facade.

This module provides backward compatibility for code that imports from
services.compliance. All imports will continue to work, but a deprecation
warning will be emitted.

Migration Guide:
    OLD: from services.compliance import X
    NEW: Depends on the module:

    CORE (Risk Controls - Universal):
        from services.core.risk_controls import X
        - audit_models, audit_storage, audit_trail_writer, retention_policy
        - time_sync (was compliance_clock), kill_switch (was enhanced_kill_switch)
        - pre_trade_controls, realtime_monitor, bcp, config

    INTEGRATION (B2B Compliance Toolkit):
        from services.algo_integration import X
        - best_execution, tca_compliance, venue_analysis, execution_quality_report
        - otr_monitor, algorithm_registry, conformance_testing, test_scenarios
        - certification, config

    ARCHIVE (Financial Entity - NOT for ICT Providers):
        from services.archive.mifid_financial_entity import X
        - lei_manager, gleif_client, transaction_report, arm_client
        - reporting_pipeline, self_assessment, governance, compliance_policies
        - nca_notification, config

Deprecation Timeline:
    - v8.0.0: Deprecation warnings added
    - v9.0.0: This facade will be removed

Why the restructure:
    This platform is positioned as an ICT Provider under MiFID II scope.
    The old monolithic compliance module mixed:
    - Universal risk controls (apply to everyone)
    - B2B compliance tools (for enterprise clients)
    - Financial Entity requirements (NOT applicable to ICT Providers)

    The new structure clearly separates these concerns.
"""

from __future__ import annotations

import warnings

__version__ = "8.0.0"
__deprecated__ = True

# =============================================================================
# Emit deprecation warning on import
# =============================================================================
warnings.warn(
    "services.compliance is deprecated. "
    "Use services.core.risk_controls, services.algo_integration, "
    "or services.archive.mifid_financial_entity instead. "
    "See module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

# =============================================================================
# Re-export from CORE (services.core.risk_controls)
# These are universal risk controls for all platform users
# =============================================================================
from services.core.risk_controls import (
    # Version
    # Config
    ControlsMode,
    TimeSyncConfig,
    PreTradeControlsConfig,
    AuditConfig,
    KillSwitchConfig,
    RiskControlsConfig,
    load_risk_controls_config,
    # Audit Models
    AuditEventType,
    AuditRecordPriority,
    AuditRecordStatus,
    OrderSide,
    AuditRecord,
    AuditRecordBuilder,
    AuditChainStatus,
    AuditExportRequest,
    AuditExportResult,
    create_order_submitted_record,
    create_order_filled_record,
    create_risk_event_record,
    create_system_event_record,
    # Audit Storage
    StorageBackendType,
    StorageState,
    AuditStorageConfig,
    StorageMetrics,
    AuditStorageBackend,
    MemoryAuditStorage,
    SQLiteAuditStorage,
    FileAuditStorage,
    create_audit_storage,
    # Retention Policy
    RetentionPeriod,
    ArchiveStatus,
    NCARequestType,
    RetentionPolicyConfig,
    NCARequest,
    RetentionRecord,
    RetentionMetrics,
    ArchiveOperation,
    RetentionManager,
    create_retention_manager,
    # Audit Trail Writer
    WriterMode,
    WriterState,
    AuditTrailWriterConfig,
    WriterMetrics,
    AuditTrailWriter,
    create_audit_trail_writer,
    # Time Sync (was compliance_clock)
    ClockDriftSeverity,
    ClockSyncStatus,
    ClockSyncEvent,
    ComplianceClock,
    create_compliance_clock,
    # Kill Switch (was enhanced_kill_switch)
    KillSwitchScope,
    KillSwitchTriggerReason,
    KillSwitchState,
    KillSwitchEvent,
    KillSwitchDetailedConfig,
    EmergencyContact,
    EnhancedKillSwitch,
    create_enhanced_kill_switch,
    # Pre-Trade Controls
    RejectionReason,
    ControlSeverity,
    PreTradeCheckResult,
    PreTradeDetailedConfig,
    TraderAuthorization,
    MessageRateWindow,
    PreTradeControls,
    create_pre_trade_controls,
    # Real-Time Monitoring
    AlertSeverity,
    AlertCategory,
    ComplianceAlert,
    MonitoringThreshold,
    RealTimeMonitorConfig,
    MonitoringMetrics,
    RealTimeMonitor,
    create_realtime_monitor,
    # BCP
    ScenarioCategory,
    ImpactLevel,
    LikelihoodLevel,
    RecoveryStatus,
    AlertLevel,
    BCPEmergencyContact,
    RecoveryStep,
    RecoveryProcedure,
    BCPScenario,
    BCPIncident,
    BusinessContinuityPlan,
    create_business_continuity_plan,
    load_bcp_from_file,
    save_bcp_to_file,
    get_standard_bcp_scenarios,
)

# =============================================================================
# Re-export from INTEGRATION (services.algo_integration)
# B2B compliance toolkit for enterprise clients
# =============================================================================
from services.algo_integration import (
    # Config
    AlgorithmType,
    ConformanceTestLevel,
    AlgorithmRegistryConfig,
    BestExecutionConfig,
    TCAConfig,
    ConformanceTestingConfig,
    OTRConfig,
    AlgoIntegrationConfig,
    load_algo_integration_config,
    # Best Execution
    ExecutionFactor,
    AssetClass,
    OrderCategory,
    VenueType,
    ExecutionQualityLevel,
    ExecutionVenue,
    FactorWeights,
    ExecutionAnalysis,
    BestExecutionPolicyConfig,
    BestExecutionPolicy,
    BestExecutionAnalyzer,
    create_best_execution_policy,
    create_best_execution_analyzer,
    get_standard_eu_venues,
    # TCA
    TCAMetricType,
    TCABenchmark,
    CostCategory,
    ExecutionStrategy,
    PreTradeEstimate,
    PostTradeAnalysis,
    TCADetailedConfig,
    TCAAggregateMetrics,
    SlippageProvider,
    TCAComplianceWrapper,
    create_tca_wrapper,
    # Venue Analysis
    VenueMetricType,
    VenueSelectionReason,
    VenueStatus,
    VenueExecutionRecord,
    VenuePerformanceMetrics,
    VenueRoutingDecision,
    VenueAnalysisConfig,
    VenueAnalyzer,
    SmartOrderRouter,
    create_venue_analyzer,
    create_smart_order_router,
    # Execution Quality Reports
    ReportPeriod,
    ReportFormat,
    ReportStatus,
    VenueExecutionSummary,
    AssetClassExecutionSummary,
    ExecutionQualityReportMetadata,
    ExecutionQualityReport,
    ReportGeneratorConfig,
    ExecutionQualityReportGenerator,
    create_report_generator,
    # OTR Monitor
    OrderEvent,
    OTRBucket,
    OTRLevel,
    OTRMetrics,
    OTRBreachEvent,
    OTRMonitorConfig,
    PerVenueOTR,
    PerAlgorithmOTR,
    OTRMonitor,
    create_otr_monitor,
    # Algorithm Registry
    AlgoType,
    AlgorithmStatus,
    AlgorithmRiskControl,
    AlgorithmRecord,
    AlgorithmRegistry,
    create_algorithm_registry,
    get_default_algorithm_types,
    # Conformance Testing
    TestResult,
    TestCategory,
    TestPriority,
    TestEnvironment,
    ConformanceSuiteStatus,
    CertificationStatus,
    TestEvidence,
    ConformanceTest,
    ConformanceTestSuite,
    TestExecutorConfig,
    ConformanceTestRunner,
    create_conformance_suite,
    create_test_runner,
    get_standard_conformance_tests,
    # Test Scenarios
    ScenarioType,
    ScenarioSeverity,
    ExecutionPhase,
    ScenarioStatus,
    ScenarioStep,
    TestScenario,
    ScenarioExecutor,
    create_test_scenario,
    create_scenario_executor,
    get_kill_switch_scenarios,
    get_pre_trade_scenarios,
    get_stress_test_scenarios,
    get_business_continuity_scenarios,
    get_all_standard_scenarios,
    # Certification
    CertificateStatus,
    CertificateType,
    DeploymentApproval,
    CertificateCondition,
    ConformanceCertificate,
    CertificateManager,
    create_certificate,
    create_certificate_manager,
)

# =============================================================================
# Re-export from ARCHIVE (services.archive.mifid_financial_entity)
# Financial Entity modules - NOT for ICT Providers
# Import suppresses the archive warning since we're already in deprecated mode
# =============================================================================
import warnings as _warnings
with _warnings.catch_warnings():
    _warnings.simplefilter("ignore", DeprecationWarning)
    from services.archive.mifid_financial_entity import (
        # Config
        ComplianceMode,
        LEIStatus,
        LEIConfig,
        TransactionReportingConfig,
        NCANotificationConfig,
        GovernanceConfig,
        MiFIDIIComplianceConfig,
        load_mifid_compliance_config,
        # LEI Manager
        LEIStatusEnum,
        LEIRecord,
        LEIValidationResult,
        LEIManager,
        create_lei_manager,
        # GLEIF Client
        GLEIFErrorCode,
        GLEIFError,
        GLEIFEntity,
        GLEIFRegistration,
        GLEIFResponse,
        GLEIFClient,
        create_gleif_client,
        # Transaction Report
        BuySellIndicator,
        TradingCapacity,
        IdentifierType,
        InstrumentIdentifierType,
        PriceType,
        QuantityType,
        TransactionType,
        # ReportStatus - already imported from algo_integration
        ISINValidator,
        MICValidator,
        CFIValidator,
        TransactionReportParty,
        TransactionReport,
        TransactionReportBuilder,
        # ARM Client
        ARMProvider,
        ARMEnvironment,
        SubmissionStatus,
        ErrorCode,
        ARMError,
        SubmissionResult,
        BatchSubmissionResult,
        ARMClientConfig,
        ARMClient,
        MockARMClient,
        BloombergBTRLClient,
        FileARMClient,
        create_arm_client,
        # Reporting Pipeline
        PipelineStatus,
        ReportQueuePriority,
        PipelineConfig,
        QueuedReport,
        PipelineMetrics,
        TransactionReportingPipeline,
        create_reporting_pipeline,
        # Self Assessment
        AssessmentCategory,
        ComplianceStatus,
        RemediationPriority,
        AssessmentStatus,
        Evidence,
        RemediationAction,
        SelfAssessmentQuestion,
        AnnualSelfAssessment,
        create_annual_assessment,
        load_assessment_from_file,
        save_assessment_to_file,
        get_rts6_assessment_template,
        # Governance
        PolicyType,
        PolicyStatus,
        ApprovalLevel,
        ReviewFrequency,
        PolicyVersion,
        PolicySection,
        PolicyDocument,
        GovernanceFramework,
        create_governance_framework,
        create_algorithmic_trading_policy,
        create_risk_management_policy,
        create_record_keeping_policy,
        load_framework_from_file,
        save_framework_to_file,
        # Compliance Policies
        create_best_execution_policy,
        create_order_handling_policy,
        create_conflicts_of_interest_policy,
        create_kill_switch_policy,
        create_transaction_reporting_policy,
        create_market_abuse_prevention_policy,
        create_business_continuity_policy,
        create_all_standard_policies,
        # NCA Notification
        NCAJurisdiction,
        NotificationType,
        NotificationStatus,
        AlgorithmCategory,
        NCAContact,
        AlgorithmDescription,
        NCANotification,
        NCANotificationManager,
        create_algorithm_description,
        create_nca_notification_manager,
    )

# =============================================================================
# Backward compatibility aliases
# Some names were changed during migration
# =============================================================================
# compliance_clock -> time_sync
# Enhanced kill switch is still called EnhancedKillSwitch

# Old config names (for compatibility with old code)
ClockSyncComplianceConfig = TimeSyncConfig  # Alias

# =============================================================================
# Public API (comprehensive list for star imports)
# =============================================================================
__all__ = [
    # Version info
    "__version__",
    "__deprecated__",
    # =========================================================================
    # CORE: Risk Controls
    # =========================================================================
    # Config
    "ControlsMode",
    "TimeSyncConfig",
    "ClockSyncComplianceConfig",  # Alias for backward compat
    "PreTradeControlsConfig",
    "AuditConfig",
    "KillSwitchConfig",
    "RiskControlsConfig",
    "load_risk_controls_config",
    # Audit Models
    "AuditEventType",
    "AuditRecordPriority",
    "AuditRecordStatus",
    "OrderSide",
    "AuditRecord",
    "AuditRecordBuilder",
    "AuditChainStatus",
    "AuditExportRequest",
    "AuditExportResult",
    "create_order_submitted_record",
    "create_order_filled_record",
    "create_risk_event_record",
    "create_system_event_record",
    # Audit Storage
    "StorageBackendType",
    "StorageState",
    "AuditStorageConfig",
    "StorageMetrics",
    "AuditStorageBackend",
    "MemoryAuditStorage",
    "SQLiteAuditStorage",
    "FileAuditStorage",
    "create_audit_storage",
    # Retention Policy
    "RetentionPeriod",
    "ArchiveStatus",
    "NCARequestType",
    "RetentionPolicyConfig",
    "NCARequest",
    "RetentionRecord",
    "RetentionMetrics",
    "ArchiveOperation",
    "RetentionManager",
    "create_retention_manager",
    # Audit Trail Writer
    "WriterMode",
    "WriterState",
    "AuditTrailWriterConfig",
    "WriterMetrics",
    "AuditTrailWriter",
    "create_audit_trail_writer",
    # Time Sync
    "ClockDriftSeverity",
    "ClockSyncStatus",
    "ClockSyncEvent",
    "ComplianceClock",
    "create_compliance_clock",
    # Kill Switch
    "KillSwitchScope",
    "KillSwitchTriggerReason",
    "KillSwitchState",
    "KillSwitchEvent",
    "KillSwitchDetailedConfig",
    "EmergencyContact",
    "EnhancedKillSwitch",
    "create_enhanced_kill_switch",
    # Pre-Trade Controls
    "RejectionReason",
    "ControlSeverity",
    "PreTradeCheckResult",
    "PreTradeDetailedConfig",
    "TraderAuthorization",
    "MessageRateWindow",
    "PreTradeControls",
    "create_pre_trade_controls",
    # Real-Time Monitoring
    "AlertSeverity",
    "AlertCategory",
    "ComplianceAlert",
    "MonitoringThreshold",
    "RealTimeMonitorConfig",
    "MonitoringMetrics",
    "RealTimeMonitor",
    "create_realtime_monitor",
    # BCP
    "ScenarioCategory",
    "ImpactLevel",
    "LikelihoodLevel",
    "RecoveryStatus",
    "AlertLevel",
    "BCPEmergencyContact",
    "RecoveryStep",
    "RecoveryProcedure",
    "BCPScenario",
    "BCPIncident",
    "BusinessContinuityPlan",
    "create_business_continuity_plan",
    "load_bcp_from_file",
    "save_bcp_to_file",
    "get_standard_bcp_scenarios",
    # =========================================================================
    # INTEGRATION: B2B Compliance Toolkit
    # =========================================================================
    # Config
    "AlgorithmType",
    "ConformanceTestLevel",
    "AlgorithmRegistryConfig",
    "BestExecutionConfig",
    "TCAConfig",
    "ConformanceTestingConfig",
    "OTRConfig",
    "AlgoIntegrationConfig",
    "load_algo_integration_config",
    # Best Execution
    "ExecutionFactor",
    "AssetClass",
    "OrderCategory",
    "VenueType",
    "ExecutionQualityLevel",
    "ExecutionVenue",
    "FactorWeights",
    "ExecutionAnalysis",
    "BestExecutionPolicyConfig",
    "BestExecutionPolicy",
    "BestExecutionAnalyzer",
    "create_best_execution_policy",
    "create_best_execution_analyzer",
    "get_standard_eu_venues",
    # TCA
    "TCAMetricType",
    "TCABenchmark",
    "CostCategory",
    "ExecutionStrategy",
    "PreTradeEstimate",
    "PostTradeAnalysis",
    "TCADetailedConfig",
    "TCAAggregateMetrics",
    "SlippageProvider",
    "TCAComplianceWrapper",
    "create_tca_wrapper",
    # Venue Analysis
    "VenueMetricType",
    "VenueSelectionReason",
    "VenueStatus",
    "VenueExecutionRecord",
    "VenuePerformanceMetrics",
    "VenueRoutingDecision",
    "VenueAnalysisConfig",
    "VenueAnalyzer",
    "SmartOrderRouter",
    "create_venue_analyzer",
    "create_smart_order_router",
    # Execution Quality Reports
    "ReportPeriod",
    "ReportFormat",
    "ReportStatus",
    "VenueExecutionSummary",
    "AssetClassExecutionSummary",
    "ExecutionQualityReportMetadata",
    "ExecutionQualityReport",
    "ReportGeneratorConfig",
    "ExecutionQualityReportGenerator",
    "create_report_generator",
    # OTR Monitor
    "OrderEvent",
    "OTRBucket",
    "OTRLevel",
    "OTRMetrics",
    "OTRBreachEvent",
    "OTRMonitorConfig",
    "PerVenueOTR",
    "PerAlgorithmOTR",
    "OTRMonitor",
    "create_otr_monitor",
    # Algorithm Registry
    "AlgoType",
    "AlgorithmStatus",
    "AlgorithmRiskControl",
    "AlgorithmRecord",
    "AlgorithmRegistry",
    "create_algorithm_registry",
    "get_default_algorithm_types",
    # Conformance Testing
    "TestResult",
    "TestCategory",
    "TestPriority",
    "TestEnvironment",
    "ConformanceSuiteStatus",
    "CertificationStatus",
    "TestEvidence",
    "ConformanceTest",
    "ConformanceTestSuite",
    "TestExecutorConfig",
    "ConformanceTestRunner",
    "create_conformance_suite",
    "create_test_runner",
    "get_standard_conformance_tests",
    # Test Scenarios
    "ScenarioType",
    "ScenarioSeverity",
    "ExecutionPhase",
    "ScenarioStatus",
    "ScenarioStep",
    "TestScenario",
    "ScenarioExecutor",
    "create_test_scenario",
    "create_scenario_executor",
    "get_kill_switch_scenarios",
    "get_pre_trade_scenarios",
    "get_stress_test_scenarios",
    "get_business_continuity_scenarios",
    "get_all_standard_scenarios",
    # Certification
    "CertificateStatus",
    "CertificateType",
    "DeploymentApproval",
    "CertificateCondition",
    "ConformanceCertificate",
    "CertificateManager",
    "create_certificate",
    "create_certificate_manager",
    # =========================================================================
    # ARCHIVE: Financial Entity (NOT for ICT Providers)
    # =========================================================================
    # Config
    "ComplianceMode",
    "LEIStatus",
    "LEIConfig",
    "TransactionReportingConfig",
    "NCANotificationConfig",
    "GovernanceConfig",
    "MiFIDIIComplianceConfig",
    "load_mifid_compliance_config",
    # LEI Manager
    "LEIStatusEnum",
    "LEIRecord",
    "LEIValidationResult",
    "LEIManager",
    "create_lei_manager",
    # GLEIF Client
    "GLEIFErrorCode",
    "GLEIFError",
    "GLEIFEntity",
    "GLEIFRegistration",
    "GLEIFResponse",
    "GLEIFClient",
    "create_gleif_client",
    # Transaction Report
    "BuySellIndicator",
    "TradingCapacity",
    "IdentifierType",
    "InstrumentIdentifierType",
    "PriceType",
    "QuantityType",
    "TransactionType",
    "ISINValidator",
    "MICValidator",
    "CFIValidator",
    "TransactionReportParty",
    "TransactionReport",
    "TransactionReportBuilder",
    # ARM Client
    "ARMProvider",
    "ARMEnvironment",
    "SubmissionStatus",
    "ErrorCode",
    "ARMError",
    "SubmissionResult",
    "BatchSubmissionResult",
    "ARMClientConfig",
    "ARMClient",
    "MockARMClient",
    "BloombergBTRLClient",
    "FileARMClient",
    "create_arm_client",
    # Reporting Pipeline
    "PipelineStatus",
    "ReportQueuePriority",
    "PipelineConfig",
    "QueuedReport",
    "PipelineMetrics",
    "TransactionReportingPipeline",
    "create_reporting_pipeline",
    # Self Assessment
    "AssessmentCategory",
    "ComplianceStatus",
    "RemediationPriority",
    "AssessmentStatus",
    "Evidence",
    "RemediationAction",
    "SelfAssessmentQuestion",
    "AnnualSelfAssessment",
    "create_annual_assessment",
    "load_assessment_from_file",
    "save_assessment_to_file",
    "get_rts6_assessment_template",
    # Governance
    "PolicyType",
    "PolicyStatus",
    "ApprovalLevel",
    "ReviewFrequency",
    "PolicyVersion",
    "PolicySection",
    "PolicyDocument",
    "GovernanceFramework",
    "create_governance_framework",
    "create_algorithmic_trading_policy",
    "create_risk_management_policy",
    "create_record_keeping_policy",
    "load_framework_from_file",
    "save_framework_to_file",
    # Compliance Policies
    "create_best_execution_policy",
    "create_order_handling_policy",
    "create_conflicts_of_interest_policy",
    "create_kill_switch_policy",
    "create_transaction_reporting_policy",
    "create_market_abuse_prevention_policy",
    "create_business_continuity_policy",
    "create_all_standard_policies",
    # NCA Notification
    "NCAJurisdiction",
    "NotificationType",
    "NotificationStatus",
    "AlgorithmCategory",
    "NCAContact",
    "AlgorithmDescription",
    "NCANotification",
    "NCANotificationManager",
    "create_algorithm_description",
    "create_nca_notification_manager",
]
