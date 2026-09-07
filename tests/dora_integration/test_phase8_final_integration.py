# -*- coding: utf-8 -*-
"""
Phase 8 Final Integration & Cleanup Tests.

Validates that Phase 8 of the DORA Integration Layer migration is complete:
- Main __init__.py updated with all real class names
- services/dora/__init__.py converted to thin facade
- All imports updated via migration script
- Documentation complete

Test Coverage:
    - Integration layer completeness
    - Facade backward compatibility
    - All 21 modules accessible
    - All 6 subpackages functioning
    - Version and migration phase validation
"""

import sys
from pathlib import Path
from typing import List, Set

import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestPhase8IntegrationLayerCompleteness:
    """Test that the integration layer is complete."""

    def test_version_is_2_0_0(self) -> None:
        """Verify integration layer version is 2.0.0."""
        from services.dora_integration import __version__

        assert __version__ == "2.0.0", f"Expected version 2.0.0, got {__version__}"

    def test_migration_phase_is_8(self) -> None:
        """Verify migration phase is 8."""
        from services.dora_integration import __migration_phase__

        assert __migration_phase__ == 8, f"Expected phase 8, got {__migration_phase__}"

    def test_all_subpackages_importable(self) -> None:
        """Verify all 6 subpackages can be imported."""
        subpackages = [
            "due_diligence",
            "incident_interface",
            "third_party",
            "contracts",
            "reporting",
            "sharing",
        ]

        for subpackage in subpackages:
            try:
                module = __import__(
                    f"services.dora_integration.{subpackage}", fromlist=[subpackage]
                )
                assert hasattr(module, "__all__"), f"{subpackage} missing __all__"
                assert len(module.__all__) > 0, f"{subpackage} has empty __all__"
            except ImportError as e:
                pytest.fail(f"Failed to import {subpackage}: {e}")

    def test_main_init_exports_count(self) -> None:
        """Verify main __init__.py exports all required symbols."""
        from services import dora_integration

        # Should have substantial exports from all phases
        expected_min_exports = 150  # Conservative estimate
        actual_exports = len(dora_integration.__all__)

        assert (
            actual_exports >= expected_min_exports
        ), f"Expected at least {expected_min_exports} exports, got {actual_exports}"


class TestPhase8DueDiligenceLayer:
    """Test Phase 1: Due Diligence & Audit Layer completeness."""

    def test_audit_readiness_exports(self) -> None:
        """Verify audit_readiness module exports."""
        from services.dora_integration.due_diligence import (
            DORAuditReadiness,
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
            create_audit_readiness,
            get_standard_evidence_templates,
        )

        assert DORAuditReadiness is not None
        assert callable(create_audit_readiness)

    def test_provider_info_package_exports(self) -> None:
        """Verify provider_info_package module exports."""
        from services.dora_integration.due_diligence import (
            DORAProviderInfoPackage,
            ProviderInfoPackageGenerator,
            ProviderIdentification,
            ICTServiceType,
            FunctionCriticality,
            SubstitutabilityLevel,
            DataSensitivity,
            ICTServiceDescription,
            DataLocationInfo,
            ProviderInfoPackage,
            create_provider_info_package,
        )

        assert DORAProviderInfoPackage is not None
        assert ProviderInfoPackageGenerator is not None

    def test_pooled_audit_support_exports(self) -> None:
        """Verify pooled_audit_support module exports."""
        from services.dora_integration.due_diligence import (
            PooledAuditSupport,
            PooledAuditEngagement,
            CertificationRecord,
            PooledAuditParticipant,
            AuditReportType,
            PooledAuditStatus,
            create_pooled_audit_support,
            get_audit_scope_areas,
        )

        assert PooledAuditSupport is not None
        assert callable(create_pooled_audit_support)

    def test_compliance_dashboard_exports(self) -> None:
        """Verify compliance_dashboard module exports."""
        from services.dora_integration.due_diligence import (
            DORAComplianceDashboard,
            ComplianceStatus,
            DORAComplianceReport,
            ComplianceIssue,
            IssueSeverity,
            IssueStatus,
            DeadlineStatus,
        )

        assert DORAComplianceDashboard is not None


class TestPhase8IncidentInterfaceLayer:
    """Test Phase 2: Incident Interface Layer completeness."""

    def test_client_notification_exports(self) -> None:
        """Verify client_incident_notification module exports."""
        from services.dora_integration.incident_interface import (
            ClientNotificationService,
            DORAClientNotification,
            IncidentSeverity,
            NotificationStatus,
            NotificationChannel,
            IncidentCategory,
            ClientContact,
            IncidentNotification,
            create_client_notification_service,
        )

        assert ClientNotificationService is not None
        assert DORAClientNotification is not None

    def test_incident_classification_exports(self) -> None:
        """Verify incident_classification module exports."""
        from services.dora_integration.incident_interface import (
            DORAIncidentClassification,
            IncidentClassificationResult,
            ClassificationThresholds,
            IncidentClassificationType,
            ClientType,
            DataType,
            MajorIncidentTrigger,
            create_incident_classification,
            get_default_thresholds,
        )

        assert DORAIncidentClassification is not None
        assert callable(create_incident_classification)

    def test_incident_reporting_exports(self) -> None:
        """Verify incident_reporting module exports."""
        from services.dora_integration.incident_interface import (
            DORAIncidentReporter,
            ReportType,
            ReportStatus,
            InitialNotificationReport,
            IntermediateReport,
            FinalReport,
            create_incident_reporter,
            get_report_deadlines,
        )

        assert DORAIncidentReporter is not None
        assert callable(create_incident_reporter)

    def test_cyber_threat_notification_exports(self) -> None:
        """Verify cyber_threat_notification module exports."""
        from services.dora_integration.incident_interface import (
            CyberThreatNotificationService,
            ThreatCategory,
            ThreatActorType,
            ThreatSeverity,
            ThreatNotification,
            create_cyber_threat_notification_service,
        )

        assert CyberThreatNotificationService is not None

    def test_communication_exports(self) -> None:
        """Verify communication module exports."""
        from services.dora_integration.incident_interface import (
            DORACommunication,
            CommunicationChannel,
            StakeholderType,
            CommunicationPriority,
            CommunicationPolicy,
            CrisisStatus,
            create_communication_service,
        )

        assert DORACommunication is not None


class TestPhase8ThirdPartyRiskLayer:
    """Test Phase 3: Third-Party Risk Interface Layer completeness."""

    def test_concentration_risk_exports(self) -> None:
        """Verify concentration_risk module exports."""
        from services.dora_integration.third_party import (
            DORAConcentrationRisk,
            ConcentrationType,
            ConcentrationRiskLevel,
            MitigationStatus,
            ProviderDependency,
            ConcentrationAssessment,
            create_concentration_risk,
        )

        assert DORAConcentrationRisk is not None

    def test_ctpp_oversight_exports(self) -> None:
        """Verify ctpp_oversight module exports."""
        from services.dora_integration.third_party import (
            DORACtppOversight,
            LeadOverseer,
            CTPPStatus,
            CTPPDesignation,
            OversightRecommendation,
            DESIGNATED_CTPPS_2025,
            create_ctpp_oversight,
            get_designated_ctpps_list,
        )

        assert DORACtppOversight is not None
        assert isinstance(DESIGNATED_CTPPS_2025, list)

    def test_third_party_risk_exports(self) -> None:
        """Verify third_party_risk module exports."""
        from services.dora_integration.third_party import (
            DORAThirdPartyRiskManagement,
            ProviderType,
            ProviderCriticality,
            RiskCategory,
            RiskLevel,
            ICTProvider,
            ThirdPartyRiskAssessment,
            create_third_party_risk_management,
        )

        assert DORAThirdPartyRiskManagement is not None

    def test_third_party_incidents_exports(self) -> None:
        """Verify third_party_incidents module exports."""
        from services.dora_integration.third_party import (
            DORAThirdPartyIncidents,
            ThirdPartyProviderType,
            ThirdPartyIncidentType,
            ThirdPartyIncident,
            PostIncidentReview,
            create_third_party_incidents,
        )

        assert DORAThirdPartyIncidents is not None

    def test_subcontractor_management_exports(self) -> None:
        """Verify subcontractor_management module exports."""
        from services.dora_integration.third_party import (
            DORASubcontractorManagement,
            SubcontractorType,
            SubcontractorStatus,
            Subcontractor,
            SubcontractorChange,
            create_subcontractor_management,
        )

        assert DORASubcontractorManagement is not None


class TestPhase8ContractsLayer:
    """Test Phase 4: Contracts & SLA Layer completeness."""

    def test_contractual_requirements_exports(self) -> None:
        """Verify contractual_requirements module exports."""
        from services.dora_integration.contracts import (
            DORAContractualRequirements,
            RequirementCategory,
            RequirementType,
            ComplianceStatus,
            ContractualRequirement,
            ContractAssessment,
            ICTContract,
            create_contractual_requirements,
            get_article_30_requirements,
        )

        assert DORAContractualRequirements is not None
        assert callable(get_article_30_requirements)

    def test_sla_guardrails_exports(self) -> None:
        """Verify sla_guardrails module exports."""
        from services.dora_integration.contracts import (
            SLAGuardrails,
            SLATier,
            CapacityStatus,
            ApprovalStatus,
            SLATierDefinition,
            CapacityValidation,
            create_sla_guardrails,
            get_sla_tier_definitions,
        )

        assert SLAGuardrails is not None

    def test_exit_strategies_exports(self) -> None:
        """Verify exit_strategies module exports."""
        from services.dora_integration.contracts import (
            DORAExitStrategies,
            ExitTrigger,
            ExitPhase,
            ExitPlanStatus,
            TransitionType,
            ExitPlan,
            ExitExecution,
            create_exit_strategies,
            get_exit_triggers,
        )

        assert DORAExitStrategies is not None


class TestPhase8ReportingLayer:
    """Test Phase 5: Unified Reporting Layer completeness."""

    def test_unified_reporting_exports(self) -> None:
        """Verify unified_reporting module exports."""
        from services.dora_integration.reporting import (
            UnifiedReportingManager,
            ReportChannel,
            PackageFormat,
            UnifiedReport,
            SubmissionPackage,
            create_unified_reporting_manager,
        )

        assert UnifiedReportingManager is not None

    def test_reporting_templates_exports(self) -> None:
        """Verify reporting_templates module exports."""
        from services.dora_integration.reporting import (
            DORAReportingTemplates,
            IncidentTypeCode,
            DataTypeCode,
            ITSInitialNotificationTemplate,
            ITSIntermediateReportTemplate,
            ITSFinalReportTemplate,
            create_reporting_templates,
        )

        assert DORAReportingTemplates is not None

    def test_register_of_information_exports(self) -> None:
        """Verify register_of_information module exports."""
        from services.dora_integration.reporting import (
            DORARegisterOfInformation,
            ContractType,
            ServiceType,
            FunctionType,
            ROIDataPackage,
            create_register_of_information,
            create_roi_data_generator,
        )

        assert DORARegisterOfInformation is not None
        assert callable(create_roi_data_generator)


class TestPhase8SharingLayer:
    """Test Phase 6: Information Sharing Layer completeness."""

    def test_information_sharing_exports(self) -> None:
        """Verify information_sharing module exports."""
        from services.dora_integration.sharing import (
            DORAInformationSharing,
            CommunityType,
            SharingChannel,
            TLPLevel,
            MembershipStatus,
            SharingOutcome,
            SharingCommunity,
            InformationSharingPolicy,
            CyberThreatIntelligence,
            create_information_sharing,
            SHAREABLE_INFORMATION_TYPES,
            TLP_DEFINITIONS,
        )

        assert DORAInformationSharing is not None
        assert isinstance(SHAREABLE_INFORMATION_TYPES, (list, set, frozenset))
        assert isinstance(TLP_DEFINITIONS, dict)


class TestPhase8FacadeIntegrity:
    """Test that services.dora facade is correctly configured."""

    def test_facade_version(self) -> None:
        """Verify facade version matches integration layer."""
        from services import dora

        assert dora.__version__ == "2.1.0", f"Expected 2.1.0, got {dora.__version__}"

    def test_facade_reexports_integration(self) -> None:
        """Verify facade re-exports from integration layer."""
        from services import dora

        # Test key exports are available
        assert hasattr(dora, "DORAuditReadiness")
        assert hasattr(dora, "DORAIncidentClassification")
        assert hasattr(dora, "DORAContractualRequirements")
        assert hasattr(dora, "DORAInformationSharing")

    def test_facade_has_deprecation_handler(self) -> None:
        """Verify facade has __getattr__ for deprecation warnings."""
        from services import dora

        # Check __getattr__ exists
        assert hasattr(dora, "__getattr__")


class TestPhase8ArchiveAccessibility:
    """Test that archived modules are accessible."""

    def test_archive_fe_modules_importable(self) -> None:
        """Verify archived FE modules can be imported."""
        from services.archive.dora_financial_entity import (
            DORAScope,
            FunctionClassifier,
            ProportionalityAssessor,
            DORAGovernanceFramework,
        )

        assert DORAScope is not None
        assert FunctionClassifier is not None

    def test_archive_fe_via_facade(self) -> None:
        """Verify archived FE modules accessible via facade."""
        from services.dora import (
            DORAEntityType,
            DORAScopeResult,
            DORARegime,
            GovernanceRole,
        )

        assert DORAEntityType is not None
        assert DORARegime is not None


class TestPhase8ModuleCount:
    """Test that all expected modules are present."""

    def test_integration_module_count(self) -> None:
        """Verify 21 modules in integration layer."""
        expected_modules = {
            "due_diligence": [
                "audit_readiness",
                "provider_info_package",
                "pooled_audit_support",
                "compliance_dashboard",
            ],
            "incident_interface": [
                "client_incident_notification",
                "incident_classification",
                "incident_reporting",
                "cyber_threat_notification",
                "communication",
            ],
            "third_party": [
                "concentration_risk",
                "ctpp_oversight",
                "third_party_risk",
                "third_party_incidents",
                "subcontractor_management",
            ],
            "contracts": ["contractual_requirements", "sla_guardrails", "exit_strategies"],
            "reporting": ["unified_reporting", "reporting_templates", "register_of_information"],
            "sharing": ["information_sharing"],
        }

        total_modules = sum(len(modules) for modules in expected_modules.values())
        assert total_modules == 21, f"Expected 21 modules, got {total_modules}"

        # Verify each module file exists
        base_path = PROJECT_ROOT / "services" / "dora_integration"

        for subpackage, modules in expected_modules.items():
            subpackage_path = base_path / subpackage
            for module in modules:
                module_path = subpackage_path / f"{module}.py"
                assert module_path.exists(), f"Module not found: {module_path}"


class TestPhase8DirectoryCleanup:
    """Test that services/dora/ is properly cleaned up."""

    def test_dora_directory_is_facade_only(self) -> None:
        """Verify services/dora/ only contains __init__.py."""
        dora_path = PROJECT_ROOT / "services" / "dora"

        # Get all .py files (excluding __pycache__)
        py_files = [f for f in dora_path.glob("*.py") if f.name != "__init__.py"]

        assert (
            len(py_files) == 0
        ), f"services/dora/ should only have __init__.py, found: {[f.name for f in py_files]}"

    def test_no_duplicate_modules(self) -> None:
        """Verify no duplicate modules exist in services/dora/."""
        dora_path = PROJECT_ROOT / "services" / "dora"

        # Should only have __init__.py and __pycache__
        contents = list(dora_path.iterdir())
        allowed = {"__init__.py", "__pycache__"}

        for item in contents:
            assert item.name in allowed, f"Unexpected item in services/dora/: {item.name}"


class TestPhase8ConfigurationComplete:
    """Test that configuration is properly set up."""

    def test_integration_config_exists(self) -> None:
        """Verify config/dora_integration/ directory exists."""
        config_path = PROJECT_ROOT / "config" / "dora_integration"
        assert config_path.exists(), f"Config directory not found: {config_path}"

    def test_required_configs_present(self) -> None:
        """Verify required config files exist."""
        config_path = PROJECT_ROOT / "config" / "dora_integration"

        # Check for README at minimum
        readme = config_path / "README.md"
        assert readme.exists(), "README.md missing from config/dora_integration/"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
