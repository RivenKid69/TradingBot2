# -*- coding: utf-8 -*-
"""
Tests for Archive Financial Entity Package __init__.py exports.

Verifies that all public APIs are properly exported and accessible,
and that appropriate deprecation warnings are emitted.
"""

import pytest
import warnings


class TestArchivePackageExports:
    """Test that archive package exports all expected items."""

    def test_version_exported(self):
        """Test version is exported."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from services.archive.mifid_financial_entity import __version__
            assert __version__ == "1.0.0"

    def test_archive_status_exported(self):
        """Test archive status is exported."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from services.archive.mifid_financial_entity import __archived__, __archive_reason__
            assert __archived__ is True
            assert "ICT Provider" in __archive_reason__

    def test_config_exports(self):
        """Test config module exports."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from services.archive.mifid_financial_entity import (
                ComplianceMode,
                LEIStatus,
                LEIConfig,
                TransactionReportingConfig,
                NCANotificationConfig,
                GovernanceConfig,
                MiFIDIIComplianceConfig,
                load_mifid_compliance_config,
            )
            assert hasattr(ComplianceMode, "PRODUCTION")
            assert callable(load_mifid_compliance_config)

    def test_lei_manager_exports(self):
        """Test lei_manager module exports."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from services.archive.mifid_financial_entity import (
                LEIStatusEnum,
                LEIRecord,
                LEIValidationResult,
                LEIManager,
                create_lei_manager,
            )
            assert hasattr(LEIStatusEnum, "ISSUED")
            assert callable(create_lei_manager)

    def test_gleif_client_exports(self):
        """Test gleif_client module exports."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from services.archive.mifid_financial_entity import (
                GLEIFErrorCode,
                GLEIFError,
                GLEIFEntity,
                GLEIFRegistration,
                GLEIFResponse,
                GLEIFClient,
                create_gleif_client,
            )
            assert callable(create_gleif_client)

    def test_transaction_report_exports(self):
        """Test transaction_report module exports."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from services.archive.mifid_financial_entity import (
                BuySellIndicator,
                TradingCapacity,
                IdentifierType,
                InstrumentIdentifierType,
                PriceType,
                QuantityType,
                TransactionType,
                ReportStatus,
                ISINValidator,
                MICValidator,
                CFIValidator,
                TransactionReportParty,
                TransactionReport,
                TransactionReportBuilder,
            )
            assert hasattr(BuySellIndicator, "BUY")
            assert hasattr(TradingCapacity, "DEAL")

    def test_arm_client_exports(self):
        """Test arm_client module exports."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from services.archive.mifid_financial_entity import (
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
            )
            assert hasattr(ARMProvider, "BLOOMBERG_BTRL")
            assert callable(create_arm_client)

    def test_reporting_pipeline_exports(self):
        """Test reporting_pipeline module exports."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from services.archive.mifid_financial_entity import (
                PipelineStatus,
                ReportQueuePriority,
                PipelineConfig,
                QueuedReport,
                PipelineMetrics,
                TransactionReportingPipeline,
                create_reporting_pipeline,
            )
            assert hasattr(PipelineStatus, "RUNNING")
            assert callable(create_reporting_pipeline)

    def test_self_assessment_exports(self):
        """Test self_assessment module exports."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from services.archive.mifid_financial_entity import (
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
            )
            assert hasattr(AssessmentCategory, "RISK_CONTROLS")
            assert callable(create_annual_assessment)
            assert callable(get_rts6_assessment_template)

    def test_governance_exports(self):
        """Test governance module exports."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from services.archive.mifid_financial_entity import (
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
            )
            assert hasattr(PolicyType, "ALGORITHMIC_TRADING")
            assert callable(create_governance_framework)

    def test_compliance_policies_exports(self):
        """Test compliance_policies module exports."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from services.archive.mifid_financial_entity import (
                create_best_execution_policy,
                create_order_handling_policy,
                create_conflicts_of_interest_policy,
                create_kill_switch_policy,
                create_transaction_reporting_policy,
                create_market_abuse_prevention_policy,
                create_business_continuity_policy,
                create_all_standard_policies,
            )
            assert callable(create_best_execution_policy)
            assert callable(create_all_standard_policies)

    def test_nca_notification_exports(self):
        """Test nca_notification module exports."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from services.archive.mifid_financial_entity import (
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
            assert hasattr(NCAJurisdiction, "FCA")
            assert callable(create_nca_notification_manager)

    def test_all_exports_in_dunder_all(self):
        """Test that all exports are listed in __all__."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import services.archive.mifid_financial_entity as archive
            assert hasattr(archive, "__all__")
            # Verify some key exports are in __all__
            expected_exports = [
                "LEIManager",
                "TransactionReport",
                "ARMClient",
                "TransactionReportingPipeline",
                "NCANotificationManager",
                "GovernanceFramework",
            ]
            for export in expected_exports:
                assert export in archive.__all__, f"{export} not in __all__"


class TestArchivePackageDeprecationWarning:
    """Test that archive package emits deprecation warning."""

    def test_deprecation_warning_on_import(self):
        """Test that importing archive package emits deprecation warning."""
        import importlib
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Force reimport
            import services.archive.mifid_financial_entity
            importlib.reload(services.archive.mifid_financial_entity)

            # Check for deprecation warning
            deprecation_warnings = [
                x for x in w
                if issubclass(x.category, DeprecationWarning)
                and "archived" in str(x.message).lower()
            ]
            assert len(deprecation_warnings) >= 1, "Expected deprecation warning"
            assert "ICT Provider" in str(deprecation_warnings[0].message) or \
                   "Investment Firm" in str(deprecation_warnings[0].message)

    def test_warning_mentions_scope(self):
        """Test that warning mentions correct scope."""
        import importlib
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import services.archive.mifid_financial_entity
            importlib.reload(services.archive.mifid_financial_entity)

            # Check warning content
            for warning in w:
                if issubclass(warning.category, DeprecationWarning):
                    msg = str(warning.message)
                    # Should mention it's for Investment Firms
                    if "archived" in msg.lower():
                        assert "Investment Firm" in msg or "ICT Provider" in msg


class TestArchivePackageUsability:
    """Test that archive package exports are usable (with warnings suppressed)."""

    def test_create_lei_manager(self):
        """Test creating LEI manager."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from services.archive.mifid_financial_entity import create_lei_manager
            manager = create_lei_manager()
            assert manager is not None

    def test_create_arm_client(self):
        """Test creating ARM client."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from services.archive.mifid_financial_entity import create_arm_client, ARMClientConfig
            config = ARMClientConfig()
            client = create_arm_client(config=config)
            assert client is not None

    def test_create_reporting_pipeline(self):
        """Test creating reporting pipeline."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from services.archive.mifid_financial_entity import create_reporting_pipeline
            pipeline = create_reporting_pipeline(lei="TEST_LEI_12345678901")
            assert pipeline is not None

    def test_create_governance_framework(self):
        """Test creating governance framework."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from services.archive.mifid_financial_entity import create_governance_framework
            framework = create_governance_framework(
                firm_lei="TEST_LEI_12345678901",
                firm_name="Test Firm",
                governance_owner="Test Owner"
            )
            assert framework is not None

    def test_get_rts6_assessment_template(self):
        """Test getting RTS 6 assessment template."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from services.archive.mifid_financial_entity import get_rts6_assessment_template
            template = get_rts6_assessment_template()
            assert len(template) > 0

    def test_create_nca_notification_manager(self):
        """Test creating NCA notification manager."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from services.archive.mifid_financial_entity import create_nca_notification_manager
            manager = create_nca_notification_manager()
            assert manager is not None
