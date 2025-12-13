# -*- coding: utf-8 -*-
"""
Tests for Algo Integration Package __init__.py exports.

Verifies that all public APIs are properly exported and accessible.
"""

import pytest


class TestAlgoIntegrationPackageExports:
    """Test that algo_integration package exports all expected items."""

    def test_version_exported(self):
        """Test version is exported."""
        from services.algo_integration import __version__
        assert __version__ == "1.0.0"

    def test_config_exports(self):
        """Test config module exports."""
        from services.algo_integration import (
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
        assert hasattr(AlgorithmType, "MARKET_MAKING")
        assert callable(load_algo_integration_config)

    def test_best_execution_exports(self):
        """Test best_execution module exports."""
        from services.algo_integration import (
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
        )
        assert hasattr(ExecutionFactor, "PRICE")
        assert hasattr(AssetClass, "EQUITY")
        assert callable(create_best_execution_analyzer)
        assert callable(get_standard_eu_venues)

    def test_tca_compliance_exports(self):
        """Test tca_compliance module exports."""
        from services.algo_integration import (
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
        )
        assert hasattr(TCAMetricType, "VWAP_SLIPPAGE")
        assert hasattr(TCABenchmark, "VWAP")
        assert callable(create_tca_wrapper)

    def test_venue_analysis_exports(self):
        """Test venue_analysis module exports."""
        from services.algo_integration import (
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
        )
        assert hasattr(VenueMetricType, "FILL_RATE")
        assert hasattr(VenueStatus, "ACTIVE")
        assert callable(create_venue_analyzer)
        assert callable(create_smart_order_router)

    def test_execution_quality_report_exports(self):
        """Test execution_quality_report module exports."""
        from services.algo_integration import (
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
        )
        assert hasattr(ReportPeriod, "QUARTERLY")
        assert hasattr(ReportFormat, "JSON")
        assert callable(create_report_generator)

    def test_otr_monitor_exports(self):
        """Test otr_monitor module exports."""
        from services.algo_integration import (
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
        assert hasattr(OTRLevel, "WARNING")
        assert callable(create_otr_monitor)

    def test_algorithm_registry_exports(self):
        """Test algorithm_registry module exports."""
        from services.algo_integration import (
            AlgoType,
            AlgorithmStatus,
            AlgorithmRiskControl,
            AlgorithmRecord,
            AlgorithmRegistry,
            create_algorithm_registry,
            get_default_algorithm_types,
        )
        assert hasattr(AlgorithmStatus, "PRODUCTION")
        assert callable(create_algorithm_registry)
        assert callable(get_default_algorithm_types)

    def test_conformance_testing_exports(self):
        """Test conformance_testing module exports."""
        from services.algo_integration import (
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
        )
        assert hasattr(TestResult, "PASS")
        assert hasattr(TestCategory, "KILL_SWITCH")
        assert callable(create_test_runner)
        assert callable(get_standard_conformance_tests)

    def test_test_scenarios_exports(self):
        """Test test_scenarios module exports."""
        from services.algo_integration import (
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
        )
        assert hasattr(ScenarioType, "FUNCTIONAL")
        assert hasattr(ScenarioSeverity, "CRITICAL")
        assert callable(create_scenario_executor)
        assert callable(get_all_standard_scenarios)

    def test_certification_exports(self):
        """Test certification module exports."""
        from services.algo_integration import (
            CertificateStatus,
            CertificateType,
            DeploymentApproval,
            CertificateCondition,
            ConformanceCertificate,
            CertificateManager,
            create_certificate,
            create_certificate_manager,
        )
        assert hasattr(CertificateStatus, "APPROVED")
        assert hasattr(CertificateType, "INITIAL")
        assert callable(create_certificate)
        assert callable(create_certificate_manager)

    def test_all_exports_in_dunder_all(self):
        """Test that all exports are listed in __all__."""
        import services.algo_integration as algo
        assert hasattr(algo, "__all__")
        # Verify some key exports are in __all__
        expected_exports = [
            "BestExecutionAnalyzer",
            "TCAComplianceWrapper",
            "VenueAnalyzer",
            "OTRMonitor",
            "AlgorithmRegistry",
            "ConformanceTestRunner",
            "CertificateManager",
        ]
        for export in expected_exports:
            assert export in algo.__all__, f"{export} not in __all__"


class TestAlgoIntegrationPackageUsability:
    """Test that algo_integration package exports are usable."""

    def test_create_best_execution_analyzer(self):
        """Test creating best execution analyzer."""
        from services.algo_integration import create_best_execution_analyzer, create_best_execution_policy
        policy = create_best_execution_policy()
        analyzer = create_best_execution_analyzer(policy=policy)
        assert analyzer is not None

    def test_create_otr_monitor(self):
        """Test creating OTR monitor."""
        from services.algo_integration import create_otr_monitor
        monitor = create_otr_monitor()
        assert monitor is not None

    def test_create_algorithm_registry(self):
        """Test creating algorithm registry."""
        from services.algo_integration import create_algorithm_registry
        registry = create_algorithm_registry()
        assert registry is not None

    def test_create_test_runner(self):
        """Test creating conformance test runner."""
        from services.algo_integration import create_test_runner
        runner = create_test_runner()
        assert runner is not None

    def test_create_certificate_manager(self):
        """Test creating certificate manager."""
        from services.algo_integration import create_certificate_manager
        manager = create_certificate_manager()
        assert manager is not None

    def test_get_standard_scenarios(self):
        """Test getting standard scenarios."""
        from services.algo_integration import get_all_standard_scenarios
        scenarios = get_all_standard_scenarios()
        assert len(scenarios) > 0

    def test_no_deprecation_warning(self):
        """Test that algo_integration does NOT emit deprecation warning."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import importlib
            import services.algo_integration
            importlib.reload(services.algo_integration)
            # Should not have deprecation warnings
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            # Allow for some warnings but not about this package being deprecated
            for warning in deprecation_warnings:
                assert "algo_integration" not in str(warning.message).lower()
