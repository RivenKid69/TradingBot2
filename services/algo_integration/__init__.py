# -*- coding: utf-8 -*-
"""
Algo Integration - B2B Compliance Toolkit.

This package provides MiFID II compliance tools for enterprise clients
(Financial Institutions / Investment Firms) using the platform.

These modules are DISABLED by default for ICT Provider deployments.
Enable them only for B2B enterprise clients who need to comply with
MiFID II regulations in their own right.

Modules:
    config: Algo integration configuration
    best_execution: Best Execution analysis (MiFID II Article 27)
    tca_compliance: Transaction Cost Analysis
    venue_analysis: Venue performance analysis and Smart Order Routing
    execution_quality_report: Execution quality reports (RTS 28)
    otr_monitor: Order-to-Trade Ratio monitoring (RTS 6)
    algorithm_registry: Algorithm registration (MiFID II Article 17(2))
    conformance_testing: Algorithm testing framework (RTS 6 Article 5)
    test_scenarios: Standard test scenarios
    certification: Deployment certification (RTS 6 Article 7)

Note:
    Import from this package emits NO warnings - it's a valid B2B toolkit.
    Only the Archive package for Financial Entity modules emits warnings.

References:
    - MiFID II Article 27: Best Execution
    - MiFID II Article 17: Algorithmic Trading requirements
    - RTS 6: Organisational requirements for investment firms
    - RTS 28: Best Execution Reports
"""

__version__ = "1.0.0"

# =============================================================================
# Configuration
# =============================================================================
from services.algo_integration.config import (
    AlgorithmType,
    ConformanceTestLevel,
    AlgorithmRegistryConfig,
    BestExecutionConfig,
    TCAConfig,
    ConformanceTestingConfig,
    OTRConfig,
    AlgoIntegrationConfig,
    load_algo_integration_config,
)

# =============================================================================
# Best Execution (MiFID II Article 27)
# =============================================================================
from services.algo_integration.best_execution import (
    # Enums
    ExecutionFactor,
    AssetClass,
    OrderCategory,
    VenueType,
    ExecutionQualityLevel,
    # Data classes
    ExecutionVenue,
    FactorWeights,
    ExecutionAnalysis,
    BestExecutionPolicyConfig,
    # Main classes
    BestExecutionPolicy,
    BestExecutionAnalyzer,
    # Factory functions
    create_best_execution_policy,
    create_best_execution_analyzer,
    # Utilities
    get_standard_eu_venues,
)

# =============================================================================
# Transaction Cost Analysis
# =============================================================================
from services.algo_integration.tca_compliance import (
    # Enums
    TCAMetricType,
    TCABenchmark,
    CostCategory,
    ExecutionStrategy,
    # Data classes
    PreTradeEstimate,
    PostTradeAnalysis,
    TCAConfig as TCADetailedConfig,  # Alias to avoid conflict
    TCAAggregateMetrics,
    # Protocols
    SlippageProvider,
    # Main class
    TCAComplianceWrapper,
    # Factory
    create_tca_wrapper,
)

# =============================================================================
# Venue Analysis & Smart Order Routing
# =============================================================================
from services.algo_integration.venue_analysis import (
    # Enums
    VenueMetricType,
    VenueSelectionReason,
    VenueStatus,
    # Data classes
    VenueExecutionRecord,
    VenuePerformanceMetrics,
    VenueRoutingDecision,
    VenueAnalysisConfig,
    # Main classes
    VenueAnalyzer,
    SmartOrderRouter,
    # Factory functions
    create_venue_analyzer,
    create_smart_order_router,
)

# =============================================================================
# Execution Quality Reports (RTS 28)
# =============================================================================
from services.algo_integration.execution_quality_report import (
    # Enums
    ReportPeriod,
    ReportFormat,
    ReportStatus,
    # Data classes
    VenueExecutionSummary,
    AssetClassExecutionSummary,
    ExecutionQualityReportMetadata,
    ExecutionQualityReport,
    ReportGeneratorConfig,
    # Main class
    ExecutionQualityReportGenerator,
    # Factory
    create_report_generator,
)

# =============================================================================
# Order-to-Trade Ratio Monitoring (RTS 6)
# =============================================================================
from services.algo_integration.otr_monitor import (
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
)

# =============================================================================
# Algorithm Registry (MiFID II Article 17(2))
# =============================================================================
from services.algo_integration.algorithm_registry import (
    AlgorithmType as AlgoType,  # Alias to avoid conflict
    AlgorithmStatus,
    AlgorithmRiskControl,
    AlgorithmRecord,
    AlgorithmRegistry,
    create_algorithm_registry,
    get_default_algorithm_types,
)

# =============================================================================
# Conformance Testing (RTS 6 Article 5)
# =============================================================================
from services.algo_integration.conformance_testing import (
    # Enums
    TestResult,
    TestCategory,
    TestPriority,
    TestEnvironment,
    ConformanceSuiteStatus,
    CertificationStatus,
    # Data classes
    TestEvidence,
    ConformanceTest,
    ConformanceTestSuite,
    # Runner
    TestExecutorConfig,
    ConformanceTestRunner,
    # Factory functions
    create_conformance_suite,
    create_test_runner,
    get_standard_conformance_tests,
)

# =============================================================================
# Test Scenarios
# =============================================================================
from services.algo_integration.test_scenarios import (
    # Enums
    ScenarioType,
    ScenarioSeverity,
    ExecutionPhase,
    ScenarioStatus,
    # Data classes
    ScenarioStep,
    TestScenario,
    # Executor
    ScenarioExecutor,
    # Factory functions
    create_test_scenario,
    create_scenario_executor,
    # Standard scenarios
    get_kill_switch_scenarios,
    get_pre_trade_scenarios,
    get_stress_test_scenarios,
    get_business_continuity_scenarios,
    get_all_standard_scenarios,
)

# =============================================================================
# Certification (RTS 6 Article 7)
# =============================================================================
from services.algo_integration.certification import (
    # Enums
    CertificateStatus,
    CertificateType,
    DeploymentApproval,
    # Data classes
    CertificateCondition,
    ConformanceCertificate,
    # Manager
    CertificateManager,
    # Factory functions
    create_certificate,
    create_certificate_manager,
)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Version
    "__version__",
    # --- Config ---
    "AlgorithmType",
    "ConformanceTestLevel",
    "AlgorithmRegistryConfig",
    "BestExecutionConfig",
    "TCAConfig",
    "ConformanceTestingConfig",
    "OTRConfig",
    "AlgoIntegrationConfig",
    "load_algo_integration_config",
    # --- Best Execution ---
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
    # --- TCA ---
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
    # --- Venue Analysis ---
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
    # --- Execution Quality Reports ---
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
    # --- OTR Monitor ---
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
    # --- Algorithm Registry ---
    "AlgoType",
    "AlgorithmStatus",
    "AlgorithmRiskControl",
    "AlgorithmRecord",
    "AlgorithmRegistry",
    "create_algorithm_registry",
    "get_default_algorithm_types",
    # --- Conformance Testing ---
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
    # --- Test Scenarios ---
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
    # --- Certification ---
    "CertificateStatus",
    "CertificateType",
    "DeploymentApproval",
    "CertificateCondition",
    "ConformanceCertificate",
    "CertificateManager",
    "create_certificate",
    "create_certificate_manager",
]
