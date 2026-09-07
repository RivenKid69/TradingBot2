# -*- coding: utf-8 -*-
"""
Tests for Backward Compatibility Facade (Phase 10).

This module tests that the deprecated services.compliance facade:
1. Emits deprecation warnings on import
2. Successfully re-exports all symbols from:
   - services.core.risk_controls (CORE)
   - services.algo_integration (INTEGRATION)
   - services.archive.mifid_financial_entity (ARCHIVE)
3. Maintains backward compatibility for existing code
"""

import pytest
import warnings
from typing import Any

# The MiFID II Financial-Entity archive is intentionally NOT part of the ICT
# Provider build (see services/compliance/__init__.py). The facade degrades
# gracefully; archive-specific tests skip when the archive is unavailable
# instead of hard-failing on a deliberately-removed module.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from services.compliance import ARCHIVE_AVAILABLE

_archive_only = pytest.mark.skipif(
    not ARCHIVE_AVAILABLE,
    reason="MiFID Financial-Entity archive intentionally absent (ICT Provider build)",
)


class TestFacadeDeprecationWarning:
    """Test that facade emits proper deprecation warnings."""

    def test_import_emits_deprecation_warning(self) -> None:
        """Verify that importing from services.compliance emits DeprecationWarning."""
        import sys

        # Remove from cache to ensure fresh import
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith("services.compliance")]
        for mod in modules_to_remove:
            if mod in sys.modules:
                del sys.modules[mod]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import services.compliance

            # Should have at least one deprecation warning
            deprecation_warnings = [
                x
                for x in w
                if issubclass(x.category, DeprecationWarning)
                and "services.compliance is deprecated" in str(x.message)
            ]
            assert (
                len(deprecation_warnings) >= 1
            ), "Expected deprecation warning for services.compliance"

    def test_deprecation_message_contains_migration_info(self) -> None:
        """Verify deprecation message includes migration guidance."""
        import sys

        modules_to_remove = [k for k in sys.modules.keys() if k.startswith("services.compliance")]
        for mod in modules_to_remove:
            if mod in sys.modules:
                del sys.modules[mod]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import services.compliance

            # Find the deprecation warning
            msg = str(w[0].message)
            assert "services.core.risk_controls" in msg
            assert "services.algo_integration" in msg
            assert "services.archive.mifid_financial_entity" in msg

    def test_version_is_8_0_0(self) -> None:
        """Verify facade version is 8.0.0."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import services.compliance

            assert services.compliance.__version__ == "8.0.0"

    def test_deprecated_flag_is_true(self) -> None:
        """Verify __deprecated__ flag is set."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import services.compliance

            assert services.compliance.__deprecated__ is True


class TestCoreReexports:
    """Test that CORE (risk_controls) symbols are re-exported."""

    @pytest.fixture(autouse=True)
    def suppress_warnings(self) -> Any:
        """Suppress deprecation warnings for these tests."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            yield

    def test_kill_switch_imports(self) -> None:
        """Test EnhancedKillSwitch import."""
        from services.compliance import EnhancedKillSwitch, create_enhanced_kill_switch

        assert EnhancedKillSwitch is not None
        assert callable(create_enhanced_kill_switch)

    def test_audit_imports(self) -> None:
        """Test audit trail imports."""
        from services.compliance import (
            AuditTrailWriter,
            AuditRecord,
            AuditEventType,
            create_audit_trail_writer,
        )

        assert AuditTrailWriter is not None
        assert AuditRecord is not None
        assert AuditEventType is not None

    def test_pre_trade_controls_imports(self) -> None:
        """Test pre-trade controls imports."""
        from services.compliance import (
            PreTradeControls,
            PreTradeCheckResult,
            create_pre_trade_controls,
        )

        assert PreTradeControls is not None
        assert PreTradeCheckResult is not None

    def test_realtime_monitor_imports(self) -> None:
        """Test real-time monitoring imports."""
        from services.compliance import (
            RealTimeMonitor,
            ComplianceAlert,
            AlertSeverity,
        )

        assert RealTimeMonitor is not None
        assert ComplianceAlert is not None

    def test_bcp_imports(self) -> None:
        """Test BCP imports."""
        from services.compliance import (
            BusinessContinuityPlan,
            create_business_continuity_plan,
            get_standard_bcp_scenarios,
        )

        assert BusinessContinuityPlan is not None
        assert callable(create_business_continuity_plan)

    def test_time_sync_imports(self) -> None:
        """Test time sync (compliance_clock) imports."""
        from services.compliance import (
            ComplianceClock,
            ClockSyncStatus,
            create_compliance_clock,
        )

        assert ComplianceClock is not None
        assert ClockSyncStatus is not None

    def test_storage_imports(self) -> None:
        """Test audit storage imports."""
        from services.compliance import (
            MemoryAuditStorage,
            SQLiteAuditStorage,
            FileAuditStorage,
            create_audit_storage,
        )

        assert MemoryAuditStorage is not None
        assert SQLiteAuditStorage is not None

    def test_retention_imports(self) -> None:
        """Test retention policy imports."""
        from services.compliance import (
            RetentionManager,
            RetentionPeriod,
            create_retention_manager,
        )

        assert RetentionManager is not None
        assert RetentionPeriod is not None


class TestIntegrationReexports:
    """Test that INTEGRATION (algo_integration) symbols are re-exported."""

    @pytest.fixture(autouse=True)
    def suppress_warnings(self) -> Any:
        """Suppress deprecation warnings for these tests."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            yield

    def test_best_execution_imports(self) -> None:
        """Test best execution imports."""
        from services.compliance import (
            BestExecutionAnalyzer,
            BestExecutionPolicy,
            create_best_execution_analyzer,
        )

        assert BestExecutionAnalyzer is not None
        assert BestExecutionPolicy is not None

    def test_tca_imports(self) -> None:
        """Test TCA imports."""
        from services.compliance import (
            TCAComplianceWrapper,
            TCAMetricType,
            create_tca_wrapper,
        )

        assert TCAComplianceWrapper is not None
        assert TCAMetricType is not None

    def test_venue_analysis_imports(self) -> None:
        """Test venue analysis imports."""
        from services.compliance import (
            VenueAnalyzer,
            SmartOrderRouter,
            VenueMetricType,
        )

        assert VenueAnalyzer is not None
        assert SmartOrderRouter is not None

    def test_otr_monitor_imports(self) -> None:
        """Test OTR monitor imports."""
        from services.compliance import (
            OTRMonitor,
            OTRLevel,
            create_otr_monitor,
        )

        assert OTRMonitor is not None
        assert OTRLevel is not None

    def test_algorithm_registry_imports(self) -> None:
        """Test algorithm registry imports."""
        from services.compliance import (
            AlgorithmRegistry,
            AlgorithmRecord,
            create_algorithm_registry,
        )

        assert AlgorithmRegistry is not None
        assert AlgorithmRecord is not None

    def test_conformance_testing_imports(self) -> None:
        """Test conformance testing imports."""
        from services.compliance import (
            ConformanceTestRunner,
            ConformanceTestSuite,
            TestResult,
        )

        assert ConformanceTestRunner is not None
        assert ConformanceTestSuite is not None

    def test_test_scenarios_imports(self) -> None:
        """Test scenarios imports."""
        from services.compliance import (
            ScenarioExecutor,
            TestScenario,
            get_all_standard_scenarios,
        )

        assert ScenarioExecutor is not None
        assert TestScenario is not None

    def test_certification_imports(self) -> None:
        """Test certification imports."""
        from services.compliance import (
            CertificateManager,
            ConformanceCertificate,
            create_certificate_manager,
        )

        assert CertificateManager is not None
        assert ConformanceCertificate is not None

    def test_execution_quality_report_imports(self) -> None:
        """Test execution quality report imports."""
        from services.compliance import (
            ExecutionQualityReportGenerator,
            ExecutionQualityReport,
            create_report_generator,
        )

        assert ExecutionQualityReportGenerator is not None
        assert ExecutionQualityReport is not None


@_archive_only
class TestArchiveReexports:
    """Test that ARCHIVE (mifid_financial_entity) symbols are re-exported."""

    @pytest.fixture(autouse=True)
    def suppress_warnings(self) -> Any:
        """Suppress deprecation warnings for these tests."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            yield

    def test_lei_manager_imports(self) -> None:
        """Test LEI manager imports."""
        from services.compliance import (
            LEIManager,
            LEIRecord,
            create_lei_manager,
        )

        assert LEIManager is not None
        assert LEIRecord is not None

    def test_gleif_client_imports(self) -> None:
        """Test GLEIF client imports."""
        from services.compliance import (
            GLEIFClient,
            GLEIFResponse,
            create_gleif_client,
        )

        assert GLEIFClient is not None
        assert GLEIFResponse is not None

    def test_transaction_report_imports(self) -> None:
        """Test transaction report imports."""
        from services.compliance import (
            TransactionReport,
            TransactionReportBuilder,
            BuySellIndicator,
        )

        assert TransactionReport is not None
        assert TransactionReportBuilder is not None

    def test_arm_client_imports(self) -> None:
        """Test ARM client imports."""
        from services.compliance import (
            ARMClient,
            MockARMClient,
            create_arm_client,
        )

        assert ARMClient is not None
        assert MockARMClient is not None

    def test_reporting_pipeline_imports(self) -> None:
        """Test reporting pipeline imports."""
        from services.compliance import (
            TransactionReportingPipeline,
            PipelineStatus,
            create_reporting_pipeline,
        )

        assert TransactionReportingPipeline is not None
        assert PipelineStatus is not None

    def test_self_assessment_imports(self) -> None:
        """Test self assessment imports."""
        from services.compliance import (
            AnnualSelfAssessment,
            create_annual_assessment,
            get_rts6_assessment_template,
        )

        assert AnnualSelfAssessment is not None
        assert callable(create_annual_assessment)

    def test_governance_imports(self) -> None:
        """Test governance imports."""
        from services.compliance import (
            GovernanceFramework,
            PolicyDocument,
            create_governance_framework,
        )

        assert GovernanceFramework is not None
        assert PolicyDocument is not None

    def test_nca_notification_imports(self) -> None:
        """Test NCA notification imports."""
        from services.compliance import (
            NCANotificationManager,
            NCANotification,
            create_nca_notification_manager,
        )

        assert NCANotificationManager is not None
        assert NCANotification is not None


class TestBackwardCompatibilityAliases:
    """Test backward compatibility aliases."""

    @pytest.fixture(autouse=True)
    def suppress_warnings(self) -> Any:
        """Suppress deprecation warnings for these tests."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            yield

    def test_clock_sync_compliance_config_alias(self) -> None:
        """Test ClockSyncComplianceConfig alias exists."""
        from services.compliance import ClockSyncComplianceConfig, TimeSyncConfig

        assert ClockSyncComplianceConfig is TimeSyncConfig


class TestNewImportPaths:
    """Test that new import paths work correctly (without deprecation)."""

    def test_core_imports_no_warning(self) -> None:
        """Verify core imports don't produce deprecation warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from services.core.risk_controls import EnhancedKillSwitch

            # Check no deprecation warnings from core
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert (
                len(deprecation_warnings) == 0
            ), "Core imports should not produce deprecation warnings"

    def test_integration_imports_no_warning(self) -> None:
        """Verify integration imports don't produce deprecation warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from services.algo_integration import BestExecutionAnalyzer

            # Check no deprecation warnings from integration
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert (
                len(deprecation_warnings) == 0
            ), "Integration imports should not produce deprecation warnings"

    @_archive_only
    def test_archive_imports_emit_warning(self) -> None:
        """Verify archive imports emit deprecation warnings."""
        import sys

        # Clear module cache
        modules_to_remove = [
            k for k in sys.modules.keys() if k.startswith("services.archive.mifid_financial_entity")
        ]
        for mod in modules_to_remove:
            if mod in sys.modules:
                del sys.modules[mod]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from services.archive.mifid_financial_entity import LEIManager

            # Archive should produce deprecation warning
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert (
                len(deprecation_warnings) >= 1
            ), "Archive imports should produce deprecation warnings"


class TestStarImport:
    """Test that star imports work correctly."""

    @pytest.fixture(autouse=True)
    def suppress_warnings(self) -> Any:
        """Suppress deprecation warnings for these tests."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            yield

    def test_all_contains_key_exports(self) -> None:
        """Verify __all__ contains key exports."""
        from services.compliance import __all__

        key_exports = [
            # CORE
            "EnhancedKillSwitch",
            "AuditTrailWriter",
            "PreTradeControls",
            "RealTimeMonitor",
            "BusinessContinuityPlan",
            "ComplianceClock",
            # INTEGRATION
            "BestExecutionAnalyzer",
            "OTRMonitor",
            "AlgorithmRegistry",
            "ConformanceTestRunner",
            "CertificateManager",
        ]
        if ARCHIVE_AVAILABLE:
            # ARCHIVE symbols only when the MiFID Financial-Entity archive ships.
            key_exports += [
                "LEIManager",
                "GLEIFClient",
                "TransactionReport",
                "ARMClient",
                "NCANotificationManager",
            ]

        for export in key_exports:
            assert export in __all__, f"{export} should be in __all__"


class TestFunctionalBackwardCompatibility:
    """Test that actual functionality works through facade."""

    @pytest.fixture(autouse=True)
    def suppress_warnings(self) -> Any:
        """Suppress deprecation warnings for these tests."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            yield

    def test_can_create_audit_record(self) -> None:
        """Test creating audit record through facade."""
        from decimal import Decimal
        from services.compliance import (
            AuditRecord,
            AuditEventType,
            OrderSide,
            create_order_submitted_record,
        )

        record = create_order_submitted_record(
            firm_lei="549300EXAMPLE00001",
            order_id="TEST-001",
            instrument_isin="US0378331005",
            side=OrderSide.BUY,
            quantity=Decimal("100.0"),
            price=Decimal("50.0"),
            algorithm_id="ALG-001",
        )

        assert record is not None
        assert record.order_id == "TEST-001"
        assert record.event_type == AuditEventType.ORDER_SUBMITTED

    def test_can_instantiate_memory_storage(self) -> None:
        """Test instantiating memory storage through facade."""
        from services.compliance import MemoryAuditStorage

        storage = MemoryAuditStorage()
        assert storage is not None

    def test_can_create_pre_trade_check_result(self) -> None:
        """Test creating pre-trade check result through facade."""
        from services.compliance import PreTradeCheckResult, ControlSeverity

        result = PreTradeCheckResult(
            allowed=True,
            rejection_reason=None,
            severity=ControlSeverity.INFO,
            message="All checks passed",
        )

        assert result.allowed is True
        assert result.message == "All checks passed"


class TestMigrationPath:
    """Test migration from old to new import paths."""

    def test_same_classes_from_old_and_new_paths(self) -> None:
        """Verify classes are identical from old and new paths."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            # Import from old path (facade)
            from services.compliance import EnhancedKillSwitch as OldKillSwitch
            from services.compliance import AuditRecord as OldAuditRecord

            # Import from new path
            from services.core.risk_controls import EnhancedKillSwitch as NewKillSwitch
            from services.core.risk_controls import AuditRecord as NewAuditRecord

            # Should be the exact same class
            assert OldKillSwitch is NewKillSwitch
            assert OldAuditRecord is NewAuditRecord
