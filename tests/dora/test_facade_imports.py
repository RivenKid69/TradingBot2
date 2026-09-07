# -*- coding: utf-8 -*-
"""
DORA Facade Backward Compatibility Tests.

Validates that the thin facade in services/dora/ correctly re-exports
all components from services/dora_integration/ for backward compatibility.

These tests ensure that existing code using `from services.dora import X`
continues to work after the Phase 8 migration.

Test Coverage:
    - Facade version and phase
    - Re-exports from all integration layer subpackages
    - Archived Financial Entity module re-exports
    - Identity of re-exported classes (same object reference)
    - All __all__ exports accessible
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestFacadeVersion:
    """Test facade version information."""

    def test_facade_version(self) -> None:
        """Verify facade version is 2.1.0."""
        from services.dora import __version__
        assert __version__ == "2.1.0"

    def test_facade_compliance_phase(self) -> None:
        """Verify facade compliance phase is 8."""
        from services.dora import __dora_compliance_phase__
        assert __dora_compliance_phase__ == 8


class TestFacadeDueDiligenceReExports:
    """Test facade re-exports from due_diligence subpackage."""

    def test_audit_readiness_reexport(self) -> None:
        """Test DORAuditReadiness is re-exported."""
        from services.dora import DORAuditReadiness
        from services.dora_integration.due_diligence import DORAuditReadiness as Direct

        assert DORAuditReadiness is Direct, "Should be same class object"

    def test_provider_info_package_reexport(self) -> None:
        """Test DORAProviderInfoPackage is re-exported."""
        from services.dora import DORAProviderInfoPackage
        from services.dora_integration.due_diligence import DORAProviderInfoPackage as Direct

        assert DORAProviderInfoPackage is Direct

    def test_pooled_audit_support_reexport(self) -> None:
        """Test PooledAuditSupport is re-exported."""
        from services.dora import PooledAuditSupport
        from services.dora_integration.due_diligence import PooledAuditSupport as Direct

        assert PooledAuditSupport is Direct

    def test_compliance_dashboard_reexport(self) -> None:
        """Test DORAComplianceDashboard is re-exported."""
        from services.dora import DORAComplianceDashboard
        from services.dora_integration.due_diligence import DORAComplianceDashboard as Direct

        assert DORAComplianceDashboard is Direct

    def test_audit_constants_reexport(self) -> None:
        """Test audit SLA constants are re-exported."""
        from services.dora import (
            AUDIT_SLA_ACKNOWLEDGMENT_DAYS,
            AUDIT_SLA_SCHEDULING_DAYS,
            EVIDENCE_RETENTION_YEARS,
        )
        from services.dora_integration.due_diligence import (
            AUDIT_SLA_ACKNOWLEDGMENT_DAYS as Direct1,
            AUDIT_SLA_SCHEDULING_DAYS as Direct2,
            EVIDENCE_RETENTION_YEARS as Direct3,
        )

        assert AUDIT_SLA_ACKNOWLEDGMENT_DAYS == Direct1
        assert AUDIT_SLA_SCHEDULING_DAYS == Direct2
        assert EVIDENCE_RETENTION_YEARS == Direct3


class TestFacadeIncidentInterfaceReExports:
    """Test facade re-exports from incident_interface subpackage."""

    def test_client_notification_service_reexport(self) -> None:
        """Test ClientNotificationService is re-exported."""
        from services.dora import ClientNotificationService
        from services.dora_integration.incident_interface import ClientNotificationService as Direct

        assert ClientNotificationService is Direct

    def test_incident_classification_reexport(self) -> None:
        """Test DORAIncidentClassification is re-exported."""
        from services.dora import DORAIncidentClassification
        from services.dora_integration.incident_interface import DORAIncidentClassification as Direct

        assert DORAIncidentClassification is Direct

    def test_incident_reporter_reexport(self) -> None:
        """Test DORAIncidentReporter is re-exported."""
        from services.dora import DORAIncidentReporter
        from services.dora_integration.incident_interface import DORAIncidentReporter as Direct

        assert DORAIncidentReporter is Direct

    def test_cyber_threat_notification_reexport(self) -> None:
        """Test CyberThreatNotificationService is re-exported."""
        from services.dora import CyberThreatNotificationService
        from services.dora_integration.incident_interface import CyberThreatNotificationService as Direct

        assert CyberThreatNotificationService is Direct

    def test_communication_reexport(self) -> None:
        """Test DORACommunication is re-exported."""
        from services.dora import DORACommunication
        from services.dora_integration.incident_interface import DORACommunication as Direct

        assert DORACommunication is Direct


class TestFacadeThirdPartyReExports:
    """Test facade re-exports from third_party subpackage."""

    def test_concentration_risk_reexport(self) -> None:
        """Test DORAConcentrationRisk is re-exported."""
        from services.dora import DORAConcentrationRisk
        from services.dora_integration.third_party import DORAConcentrationRisk as Direct

        assert DORAConcentrationRisk is Direct

    def test_ctpp_oversight_reexport(self) -> None:
        """Test DORACtppOversight is re-exported."""
        from services.dora import DORACtppOversight
        from services.dora_integration.third_party import DORACtppOversight as Direct

        assert DORACtppOversight is Direct

    def test_third_party_risk_management_reexport(self) -> None:
        """Test DORAThirdPartyRiskManagement is re-exported."""
        from services.dora import DORAThirdPartyRiskManagement
        from services.dora_integration.third_party import DORAThirdPartyRiskManagement as Direct

        assert DORAThirdPartyRiskManagement is Direct

    def test_subcontractor_management_reexport(self) -> None:
        """Test DORASubcontractorManagement is re-exported."""
        from services.dora import DORASubcontractorManagement
        from services.dora_integration.third_party import DORASubcontractorManagement as Direct

        assert DORASubcontractorManagement is Direct


class TestFacadeContractsReExports:
    """Test facade re-exports from contracts subpackage."""

    def test_contractual_requirements_reexport(self) -> None:
        """Test DORAContractualRequirements is re-exported."""
        from services.dora import DORAContractualRequirements
        from services.dora_integration.contracts import DORAContractualRequirements as Direct

        assert DORAContractualRequirements is Direct

    def test_sla_guardrails_reexport(self) -> None:
        """Test SLAGuardrails is re-exported."""
        from services.dora import SLAGuardrails
        from services.dora_integration.contracts import SLAGuardrails as Direct

        assert SLAGuardrails is Direct

    def test_exit_strategies_reexport(self) -> None:
        """Test DORAExitStrategies is re-exported."""
        from services.dora import DORAExitStrategies
        from services.dora_integration.contracts import DORAExitStrategies as Direct

        assert DORAExitStrategies is Direct


class TestFacadeReportingReExports:
    """Test facade re-exports from reporting subpackage."""

    def test_unified_reporting_manager_reexport(self) -> None:
        """Test UnifiedReportingManager is re-exported."""
        from services.dora import UnifiedReportingManager
        from services.dora_integration.reporting import UnifiedReportingManager as Direct

        assert UnifiedReportingManager is Direct

    def test_reporting_templates_reexport(self) -> None:
        """Test DORAReportingTemplates is re-exported."""
        from services.dora import DORAReportingTemplates
        from services.dora_integration.reporting import DORAReportingTemplates as Direct

        assert DORAReportingTemplates is Direct

    def test_register_of_information_reexport(self) -> None:
        """Test DORARegisterOfInformation is re-exported."""
        from services.dora import DORARegisterOfInformation
        from services.dora_integration.reporting import DORARegisterOfInformation as Direct

        assert DORARegisterOfInformation is Direct


class TestFacadeSharingReExports:
    """Test facade re-exports from sharing subpackage."""

    def test_information_sharing_reexport(self) -> None:
        """Test DORAInformationSharing is re-exported."""
        from services.dora import DORAInformationSharing
        from services.dora_integration.sharing import DORAInformationSharing as Direct

        assert DORAInformationSharing is Direct

    def test_tlp_level_reexport(self) -> None:
        """Test TLPLevel enum is re-exported."""
        from services.dora import TLPLevel
        from services.dora_integration.sharing import TLPLevel as Direct

        assert TLPLevel is Direct

    def test_sharing_constants_reexport(self) -> None:
        """Test sharing constants are re-exported."""
        from services.dora import (
            SHAREABLE_INFORMATION_TYPES,
            TLP_DEFINITIONS,
            DEFAULT_INTELLIGENCE_RETENTION_DAYS,
        )
        from services.dora_integration.sharing import (
            SHAREABLE_INFORMATION_TYPES as Direct1,
            TLP_DEFINITIONS as Direct2,
            DEFAULT_INTELLIGENCE_RETENTION_DAYS as Direct3,
        )

        assert SHAREABLE_INFORMATION_TYPES is Direct1
        assert TLP_DEFINITIONS is Direct2
        assert DEFAULT_INTELLIGENCE_RETENTION_DAYS == Direct3


class TestFacadeArchivedFEModulesReExports:
    """Test facade re-exports from archived Financial Entity modules."""

    def test_scope_verification_reexport(self) -> None:
        """Test DORAScope is re-exported from archive."""
        from services.dora import DORAScope
        from services.archive.dora_financial_entity.scope_verification import DORAScope as Direct

        assert DORAScope is Direct

    def test_function_classifier_reexport(self) -> None:
        """Test FunctionClassifier is re-exported from archive."""
        from services.dora import FunctionClassifier
        from services.archive.dora_financial_entity.function_classification import FunctionClassifier as Direct

        assert FunctionClassifier is Direct

    def test_governance_framework_reexport(self) -> None:
        """Test DORAGovernanceFramework is re-exported from archive."""
        from services.dora import DORAGovernanceFramework
        from services.archive.dora_financial_entity.governance import DORAGovernanceFramework as Direct

        assert DORAGovernanceFramework is Direct

    def test_proportionality_assessor_reexport(self) -> None:
        """Test ProportionalityAssessor is re-exported from archive."""
        from services.dora import ProportionalityAssessor
        from services.archive.dora_financial_entity.proportionality import ProportionalityAssessor as Direct

        assert ProportionalityAssessor is Direct

    def test_regulation_integration_reexport(self) -> None:
        """Test DORARegulationIntegration is re-exported from archive."""
        from services.dora import DORARegulationIntegration
        from services.archive.dora_financial_entity.cross_regulation import DORARegulationIntegration as Direct

        assert DORARegulationIntegration is Direct


class TestFacadeAllExportsAccessible:
    """Test that all items in __all__ can be imported."""

    def test_all_exports_importable(self) -> None:
        """Verify all __all__ exports can be accessed."""
        from services import dora

        failed_imports = []
        for name in dora.__all__:
            if not hasattr(dora, name):
                failed_imports.append(name)

        assert not failed_imports, f"Failed to access: {failed_imports}"

    def test_all_exports_count(self) -> None:
        """Verify __all__ has substantial exports."""
        from services.dora import __all__
        # Should have 250+ exports (integration layer + archived FE modules)
        assert len(__all__) >= 250, f"Expected 250+ exports, got {len(__all__)}"


class TestFacadeFactoryFunctions:
    """Test factory functions are correctly re-exported."""

    def test_audit_readiness_factory(self) -> None:
        """Test create_audit_readiness factory."""
        from services.dora import create_audit_readiness
        assert callable(create_audit_readiness)

    def test_incident_classification_factory(self) -> None:
        """Test create_incident_classification factory."""
        from services.dora import create_incident_classification
        assert callable(create_incident_classification)

    def test_contractual_requirements_factory(self) -> None:
        """Test create_contractual_requirements factory."""
        from services.dora import create_contractual_requirements
        assert callable(create_contractual_requirements)

    def test_information_sharing_factory(self) -> None:
        """Test create_information_sharing factory."""
        from services.dora import create_information_sharing
        assert callable(create_information_sharing)

    def test_scope_verifier_factory(self) -> None:
        """Test create_scope_verifier factory (archived)."""
        from services.dora import create_scope_verifier
        assert callable(create_scope_verifier)

    def test_governance_framework_factory(self) -> None:
        """Test create_governance_framework factory (archived)."""
        from services.dora import create_governance_framework
        assert callable(create_governance_framework)


class TestFacadeEnumsReExport:
    """Test that enums are correctly re-exported with all members."""

    def test_audit_type_enum(self) -> None:
        """Test AuditType enum members."""
        from services.dora import AuditType

        assert hasattr(AuditType, 'CLIENT_OPERATIONAL')
        assert hasattr(AuditType, 'NCA_INSPECTION')
        assert hasattr(AuditType, 'CERTIFICATION')

    def test_incident_severity_enum(self) -> None:
        """Test IncidentSeverity enum members."""
        from services.dora import IncidentSeverity

        assert hasattr(IncidentSeverity, 'CRITICAL')
        assert hasattr(IncidentSeverity, 'HIGH')
        assert hasattr(IncidentSeverity, 'MEDIUM')
        assert hasattr(IncidentSeverity, 'LOW')

    def test_sla_tier_enum(self) -> None:
        """Test SLATier enum members."""
        from services.dora import SLATier

        assert hasattr(SLATier, 'STANDARD')
        assert hasattr(SLATier, 'PROFESSIONAL')
        assert hasattr(SLATier, 'ENTERPRISE')
        assert hasattr(SLATier, 'CRITICAL')


class TestFacadeBackwardCompatibility:
    """Test backward compatibility for existing code patterns."""

    def test_old_import_pattern_1(self) -> None:
        """Test: from services.dora import DORAScope, FunctionClassifier."""
        from services.dora import DORAScope, FunctionClassifier
        assert DORAScope is not None
        assert FunctionClassifier is not None

    def test_old_import_pattern_2(self) -> None:
        """Test: from services.dora import DORAuditReadiness."""
        from services.dora import DORAuditReadiness
        assert DORAuditReadiness is not None

    def test_old_import_pattern_3(self) -> None:
        """Test: from services.dora import DORAIncidentClassification."""
        from services.dora import DORAIncidentClassification
        assert DORAIncidentClassification is not None

    def test_module_access_pattern(self) -> None:
        """Test: import services.dora; services.dora.DORAScope."""
        import services.dora
        assert services.dora.DORAScope is not None
        assert services.dora.FunctionClassifier is not None

    def test_alias_backward_compat(self) -> None:
        """Test aliased exports for backward compatibility."""
        from services.dora import ProviderInfoPackageGenerator, DORAProviderInfoPackage
        # Should be same class
        assert ProviderInfoPackageGenerator is DORAProviderInfoPackage


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
