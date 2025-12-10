# -*- coding: utf-8 -*-
"""
Facade Import Tests for DORA Integration Layer.

Validates that the services.dora facade correctly re-exports
everything from services.dora_integration for backward compatibility.

Test Coverage:
    - All facade re-exports work
    - Deprecation warnings are properly configured
    - Archived FE modules accessible via facade
    - Version and phase info correct
"""

import sys
import warnings
from pathlib import Path
from typing import Any, List

import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestFacadeBasics:
    """Test basic facade functionality."""

    def test_facade_version(self) -> None:
        """Test facade version is correct."""
        from services import dora
        assert dora.__version__ == "2.1.0"

    def test_facade_compliance_phase(self) -> None:
        """Test facade compliance phase."""
        from services import dora
        assert dora.__dora_compliance_phase__ == 8

    def test_facade_has_all_list(self) -> None:
        """Test facade has __all__ list."""
        from services import dora
        assert hasattr(dora, "__all__")
        assert len(dora.__all__) > 100  # Should have many exports

    def test_facade_has_getattr(self) -> None:
        """Test facade has __getattr__ for deprecation."""
        from services import dora
        assert hasattr(dora, "__getattr__")


class TestFacadeDueDiligenceReexports:
    """Test facade re-exports from due_diligence."""

    def test_audit_readiness_reexport(self) -> None:
        """Test audit_readiness classes via facade."""
        from services.dora import (
            DORAuditReadiness,
            AuditType,
            AuditScope,
            AuditStatus,
            AuditRequest,
            EvidenceItem,
            create_audit_readiness,
        )

        assert DORAuditReadiness is not None
        assert callable(create_audit_readiness)

    def test_audit_constants_reexport(self) -> None:
        """Test audit constants via facade."""
        from services.dora import (
            AUDIT_SLA_ACKNOWLEDGMENT_DAYS,
            AUDIT_SLA_SCHEDULING_DAYS,
            AUDIT_TYPE_SLAS,
            EVIDENCE_RETENTION_YEARS,
        )

        assert isinstance(AUDIT_SLA_ACKNOWLEDGMENT_DAYS, int)
        assert isinstance(AUDIT_TYPE_SLAS, dict)

    def test_provider_info_reexport(self) -> None:
        """Test provider_info_package via facade."""
        from services.dora import (
            DORAProviderInfoPackage,
            ProviderInfoPackageGenerator,
            ICTServiceType,
            ProviderIdentification,
            create_provider_info_package,
        )

        assert DORAProviderInfoPackage is not None
        assert ProviderInfoPackageGenerator is not None

    def test_pooled_audit_reexport(self) -> None:
        """Test pooled_audit_support via facade."""
        from services.dora import (
            PooledAuditSupport,
            PooledAuditEngagement,
            CertificationRecord,
            create_pooled_audit_support,
            get_audit_scope_areas,
        )

        assert PooledAuditSupport is not None
        areas = get_audit_scope_areas()
        assert isinstance(areas, list)

    def test_compliance_dashboard_reexport(self) -> None:
        """Test compliance_dashboard via facade."""
        from services.dora import (
            DORAComplianceDashboard,
            ComplianceStatus,
            IssueSeverity,
            DeadlineStatus,
        )

        assert DORAComplianceDashboard is not None

    def test_multi_client_incident_reexport(self) -> None:
        """Test multi-client incident via facade."""
        from services.dora import (
            MultiClientIncidentCoordinator,
            MultiClientIncident,
            create_incident_coordinator,
        )

        assert MultiClientIncidentCoordinator is not None


class TestFacadeIncidentInterfaceReexports:
    """Test facade re-exports from incident_interface."""

    def test_client_notification_reexport(self) -> None:
        """Test client_notification via facade."""
        from services.dora import (
            ClientNotificationService,
            DORAClientNotification,
            IncidentSeverity,
            NotificationStatus,
            NotificationChannel,
            create_client_notification_service,
        )

        assert ClientNotificationService is not None
        assert DORAClientNotification is not None

    def test_incident_classification_reexport(self) -> None:
        """Test incident_classification via facade."""
        from services.dora import (
            DORAIncidentClassification,
            IncidentClassificationResult,
            ClassificationThresholds,
            create_incident_classification,
            get_default_thresholds,
        )

        assert DORAIncidentClassification is not None
        thresholds = get_default_thresholds()
        assert thresholds is not None

    def test_incident_reporting_reexport(self) -> None:
        """Test incident_reporting via facade."""
        from services.dora import (
            DORAIncidentReporter,
            ReportType,
            ReportStatus,
            InitialNotificationReport,
            IntermediateReport,
            FinalReport,
            create_incident_reporter,
        )

        assert DORAIncidentReporter is not None
        assert hasattr(ReportType, "INITIAL_NOTIFICATION")

    def test_cyber_threat_reexport(self) -> None:
        """Test cyber_threat_notification via facade."""
        from services.dora import (
            CyberThreatNotificationService,
            ThreatCategory,
            ThreatSeverity,
            create_cyber_threat_notification_service,
        )

        assert CyberThreatNotificationService is not None

    def test_communication_reexport(self) -> None:
        """Test communication via facade."""
        from services.dora import (
            DORACommunication,
            CommunicationChannel,
            StakeholderType,
            CommunicationPolicy,
            create_communication_service,
        )

        assert DORACommunication is not None


class TestFacadeThirdPartyReexports:
    """Test facade re-exports from third_party."""

    def test_concentration_risk_reexport(self) -> None:
        """Test concentration_risk via facade."""
        from services.dora import (
            DORAConcentrationRisk,
            ConcentrationType,
            ConcentrationRiskLevel,
            create_concentration_risk,
        )

        assert DORAConcentrationRisk is not None

    def test_ctpp_oversight_reexport(self) -> None:
        """Test ctpp_oversight via facade."""
        from services.dora import (
            DORACtppOversight,
            LeadOverseer,
            CTPPStatus,
            DESIGNATED_CTPPS_2025,
            create_ctpp_oversight,
            get_designated_ctpps_list,
        )

        assert DORACtppOversight is not None
        assert isinstance(DESIGNATED_CTPPS_2025, list)

    def test_third_party_risk_reexport(self) -> None:
        """Test third_party_risk via facade."""
        from services.dora import (
            DORAThirdPartyRiskManagement,
            ProviderType,
            RiskCategory,
            RiskLevel,
            create_third_party_risk_management,
        )

        assert DORAThirdPartyRiskManagement is not None

    def test_third_party_incidents_reexport(self) -> None:
        """Test third_party_incidents via facade."""
        from services.dora import (
            DORAThirdPartyIncidents,
            ThirdPartyIncident,
            create_third_party_incidents,
        )

        assert DORAThirdPartyIncidents is not None

    def test_subcontractor_management_reexport(self) -> None:
        """Test subcontractor_management via facade."""
        from services.dora import (
            DORASubcontractorManagement,
            SubcontractorType,
            Subcontractor,
            create_subcontractor_management,
        )

        assert DORASubcontractorManagement is not None


class TestFacadeContractsReexports:
    """Test facade re-exports from contracts."""

    def test_contractual_requirements_reexport(self) -> None:
        """Test contractual_requirements via facade."""
        from services.dora import (
            DORAContractualRequirements,
            RequirementCategory,
            ContractAssessment,
            create_contractual_requirements,
            get_article_30_requirements,
        )

        assert DORAContractualRequirements is not None
        reqs = get_article_30_requirements()
        assert isinstance(reqs, list)

    def test_sla_guardrails_reexport(self) -> None:
        """Test sla_guardrails via facade."""
        from services.dora import (
            SLAGuardrails,
            SLATier,
            CapacityStatus,
            create_sla_guardrails,
            get_sla_tiers,
        )

        assert SLAGuardrails is not None
        tiers = get_sla_tiers()
        assert isinstance(tiers, list)

    def test_exit_strategies_reexport(self) -> None:
        """Test exit_strategies via facade."""
        from services.dora import (
            DORAExitStrategies,
            ExitTrigger,
            ExitPhase,
            ExitPlan,
            create_exit_strategies,
        )

        assert DORAExitStrategies is not None


class TestFacadeReportingReexports:
    """Test facade re-exports from reporting."""

    def test_unified_reporting_reexport(self) -> None:
        """Test unified_reporting via facade."""
        from services.dora import (
            UnifiedReportingManager,
            ReportChannel,
            PackageFormat,
            create_unified_reporting_manager,
        )

        assert UnifiedReportingManager is not None

    def test_reporting_templates_reexport(self) -> None:
        """Test reporting_templates via facade."""
        from services.dora import (
            DORAReportingTemplates,
            DataTypeCode,
            ITSInitialNotificationTemplate,
            create_reporting_templates,
        )

        assert DORAReportingTemplates is not None

    def test_roi_reexport(self) -> None:
        """Test register_of_information via facade."""
        from services.dora import (
            DORARegisterOfInformation,
            ContractType,
            ServiceType,
            ROIDataPackage,
            create_register_of_information,
        )

        assert DORARegisterOfInformation is not None


class TestFacadeSharingReexports:
    """Test facade re-exports from sharing."""

    def test_information_sharing_reexport(self) -> None:
        """Test information_sharing via facade."""
        from services.dora import (
            DORAInformationSharing,
            CommunityType,
            SharingChannel,
            TLPLevel,
            SHAREABLE_INFORMATION_TYPES,
            TLP_DEFINITIONS,
            create_information_sharing,
        )

        assert DORAInformationSharing is not None
        assert isinstance(SHAREABLE_INFORMATION_TYPES, (list, set, frozenset))
        assert isinstance(TLP_DEFINITIONS, dict)


class TestFacadeArchivedModulesReexports:
    """Test facade re-exports archived FE modules."""

    def test_scope_verification_reexport(self) -> None:
        """Test scope_verification via facade."""
        from services.dora import (
            DORAEntityType,
            DORAScopeResult,
            ScopeVerification,
            DORAScope,
            create_scope_verifier,
        )

        assert DORAScope is not None
        assert DORAEntityType is not None

    def test_function_classification_reexport(self) -> None:
        """Test function_classification via facade."""
        from services.dora import (
            ImpairmentType,
            FunctionClassification,
            FunctionClassifier,
            create_function_classifier,
        )

        assert FunctionClassifier is not None

    def test_proportionality_reexport(self) -> None:
        """Test proportionality via facade."""
        from services.dora import (
            DORARegime,
            ExemptionType,
            ProportionalityAssessor,
            assess_entity_proportionality,
        )

        assert ProportionalityAssessor is not None
        assert callable(assess_entity_proportionality)

    def test_governance_reexport(self) -> None:
        """Test governance via facade."""
        from services.dora import (
            DORAGovernanceFramework,
            GovernanceRole,
            DefenceLine,
            MANDATORY_TRAINING_TOPICS,
            create_governance_framework,
        )

        assert DORAGovernanceFramework is not None

    def test_cross_regulation_reexport(self) -> None:
        """Test cross_regulation via facade."""
        from services.dora import (
            Regulation,
            DORARegulationIntegration,
        )

        assert DORARegulationIntegration is not None


class TestFacadeCompatibilityWithIntegrationLayer:
    """Test that facade exports match integration layer."""

    def test_same_classes_different_paths(self) -> None:
        """Test same classes accessible from both paths."""
        from services.dora import DORAuditReadiness as FacadeClass
        from services.dora_integration.due_diligence import DORAuditReadiness as IntegrationClass

        assert FacadeClass is IntegrationClass

    def test_same_enums_different_paths(self) -> None:
        """Test same enums accessible from both paths."""
        from services.dora import AuditType as FacadeEnum
        from services.dora_integration.due_diligence import AuditType as IntegrationEnum

        assert FacadeEnum is IntegrationEnum

    def test_same_constants_different_paths(self) -> None:
        """Test same constants accessible from both paths."""
        from services.dora import AUDIT_SLA_ACKNOWLEDGMENT_DAYS as FacadeConst
        from services.dora_integration.due_diligence import AUDIT_SLA_ACKNOWLEDGMENT_DAYS as IntegrationConst

        assert FacadeConst == IntegrationConst

    def test_same_factories_different_paths(self) -> None:
        """Test same factory functions accessible from both paths."""
        from services.dora import create_audit_readiness as facade_factory
        from services.dora_integration.due_diligence import create_audit_readiness as integration_factory

        assert facade_factory is integration_factory


class TestFacadeAliases:
    """Test facade aliases for backward compatibility."""

    def test_get_report_types_incident_alias(self) -> None:
        """Test get_report_types_incident alias."""
        from services.dora import get_report_types_incident

        assert callable(get_report_types_incident)


class TestFacadeDeprecationWarnings:
    """Test that deprecation warnings work correctly."""

    def test_deprecated_mapping_exists(self) -> None:
        """Test that deprecated mappings are defined."""
        from services import dora

        # Check __getattr__ exists for handling deprecations
        assert callable(dora.__getattr__)

    def test_unknown_attribute_raises_error(self) -> None:
        """Test that unknown attributes raise AttributeError."""
        from services import dora

        with pytest.raises(AttributeError):
            _ = dora.NonExistentClass


class TestFacadeDocumentation:
    """Test facade documentation is complete."""

    def test_facade_has_docstring(self) -> None:
        """Test facade module has docstring."""
        from services import dora

        assert dora.__doc__ is not None
        assert "DORA" in dora.__doc__
        assert "facade" in dora.__doc__.lower() or "Facade" in dora.__doc__

    def test_facade_mentions_integration_layer(self) -> None:
        """Test facade docstring mentions integration layer."""
        from services import dora

        assert "dora_integration" in dora.__doc__


class TestFacadeNoDirectModules:
    """Test that facade doesn't have direct module files."""

    def test_facade_directory_clean(self) -> None:
        """Test services/dora/ only has __init__.py."""
        dora_path = PROJECT_ROOT / "services" / "dora"

        # Get all .py files
        py_files = list(dora_path.glob("*.py"))
        py_file_names = [f.name for f in py_files]

        # Should only have __init__.py
        assert py_file_names == ["__init__.py"], (
            f"Expected only __init__.py, found: {py_file_names}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
