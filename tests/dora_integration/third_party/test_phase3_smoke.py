# -*- coding: utf-8 -*-
"""
Smoke Tests for DORA Phase 3 - Third-Party Risk Interface.

These tests verify that all Phase 3 modules can be imported,
instantiated, and basic operations work correctly.
"""

import pytest
import tempfile
from pathlib import Path


class TestPhase3ModuleImports:
    """Test all Phase 3 modules can be imported."""

    def test_concentration_risk_imports(self):
        """Test concentration_risk module imports."""
        from services.dora_integration.third_party.concentration_risk import (
            DORAConcentrationRisk,
            ConcentrationRiskConfig,
            ConcentrationType,
            RiskLevel,
            SubstitutabilityLevel,
        )
        assert DORAConcentrationRisk is not None
        assert ConcentrationRiskConfig is not None

    def test_ctpp_oversight_imports(self):
        """Test ctpp_oversight module imports."""
        from services.dora_integration.third_party.ctpp_oversight import (
            DORACtppOversight,
            CTPPOversightConfig,
            LeadOverseer,
            CTPPStatus,
        )
        assert DORACtppOversight is not None
        assert CTPPOversightConfig is not None

    def test_third_party_risk_imports(self):
        """Test third_party_risk module imports."""
        from services.dora_integration.third_party.third_party_risk import (
            DORAThirdPartyRiskManagement,
            ThirdPartyRiskConfig,
            ProviderType,
            ProviderCriticality,
        )
        assert DORAThirdPartyRiskManagement is not None
        assert ThirdPartyRiskConfig is not None

    def test_third_party_incidents_imports(self):
        """Test third_party_incidents module imports."""
        from services.dora_integration.third_party.third_party_incidents import (
            DORAThirdPartyIncidents,
            ThirdPartyProviderType,
            ThirdPartyCriticality,
        )
        assert DORAThirdPartyIncidents is not None

    def test_subcontractor_management_imports(self):
        """Test subcontractor_management module imports."""
        from services.dora_integration.third_party.subcontractor_management import (
            DORASubcontractorManagement,
            SubcontractorConfig,
            SubcontractorType,
        )
        assert DORASubcontractorManagement is not None
        assert SubcontractorConfig is not None


class TestPhase3ModuleInstantiation:
    """Test all Phase 3 modules can be instantiated."""

    def test_concentration_risk_instantiation(self):
        """Test DORAConcentrationRisk instantiation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from services.dora_integration.third_party.concentration_risk import (
                DORAConcentrationRisk,
                ConcentrationRiskConfig,
            )
            config = ConcentrationRiskConfig(log_path=tmpdir)
            manager = DORAConcentrationRisk(config)
            assert manager is not None

    def test_ctpp_oversight_instantiation(self):
        """Test DORACtppOversight instantiation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from services.dora_integration.third_party.ctpp_oversight import (
                DORACtppOversight,
                CTPPOversightConfig,
            )
            config = CTPPOversightConfig(log_path=tmpdir)
            manager = DORACtppOversight(config)
            assert manager is not None

    def test_third_party_risk_instantiation(self):
        """Test DORAThirdPartyRiskManagement instantiation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from services.dora_integration.third_party.third_party_risk import (
                DORAThirdPartyRiskManagement,
                ThirdPartyRiskConfig,
            )
            config = ThirdPartyRiskConfig(log_path=tmpdir)
            manager = DORAThirdPartyRiskManagement(config)
            assert manager is not None

    def test_third_party_incidents_instantiation(self):
        """Test DORAThirdPartyIncidents instantiation."""
        from services.dora_integration.third_party.third_party_incidents import (
            DORAThirdPartyIncidents,
        )
        manager = DORAThirdPartyIncidents(
            entity_id="TEST-001",
            entity_name="Test Entity",
        )
        assert manager is not None

    def test_subcontractor_management_instantiation(self):
        """Test DORASubcontractorManagement instantiation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from services.dora_integration.third_party.subcontractor_management import (
                DORASubcontractorManagement,
                SubcontractorConfig,
            )
            config = SubcontractorConfig(log_path=tmpdir)
            manager = DORASubcontractorManagement(config)
            assert manager is not None


class TestConcentrationRiskBasicOperations:
    """Test basic operations for concentration risk."""

    @pytest.fixture
    def manager(self):
        """Create concentration risk manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from services.dora_integration.third_party.concentration_risk import (
                DORAConcentrationRisk,
                ConcentrationRiskConfig,
            )
            config = ConcentrationRiskConfig(log_path=tmpdir)
            yield DORAConcentrationRisk(config)

    def test_add_provider_dependency(self, manager):
        """Test adding provider dependency."""
        from services.dora_integration.third_party.concentration_risk import (
            SubstitutabilityLevel,
        )
        dep = manager.add_provider_dependency(
            provider_id="PRV-001",
            provider_name="Test Provider",
            services=["compute", "storage"],
            substitutability=SubstitutabilityLevel.SUBSTITUTABLE_WITH_EFFORT,
        )
        assert dep is not None
        assert dep.provider_name == "Test Provider"

    def test_get_provider_dependency(self, manager):
        """Test getting provider dependency."""
        manager.add_provider_dependency(
            provider_id="PRV-001",
            provider_name="Test Provider",
            services=["compute"],
        )
        dep = manager.get_provider_dependency("PRV-001")
        assert dep is not None

    def test_calculate_metrics(self, manager):
        """Test calculating concentration metrics."""
        manager.add_provider_dependency(
            provider_id="PRV-001",
            provider_name="Provider 1",
            services=["compute"],
        )
        metrics = manager.calculate_concentration_metrics()
        assert isinstance(metrics, list)

    def test_identify_risks(self, manager):
        """Test identifying concentration risks."""
        manager.add_provider_dependency(
            provider_id="PRV-001",
            provider_name="Provider 1",
            services=["compute"],
            critical_functions=["order_execution"],
        )
        risks = manager.identify_concentration_risks()
        assert isinstance(risks, list)


class TestSubcontractorManagementBasicOperations:
    """Test basic operations for subcontractor management."""

    @pytest.fixture
    def manager(self):
        """Create subcontractor management manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from services.dora_integration.third_party.subcontractor_management import (
                DORASubcontractorManagement,
                SubcontractorConfig,
            )
            config = SubcontractorConfig(log_path=tmpdir)
            yield DORASubcontractorManagement(config)

    def test_get_all_subcontractors(self, manager):
        """Test getting all subcontractors."""
        subs = manager.get_all_subcontractors()
        assert isinstance(subs, list)
        # Should have pre-initialized subcontractors
        assert len(subs) >= 3

    def test_register_subcontractor(self, manager):
        """Test registering new subcontractor."""
        from services.dora_integration.third_party.subcontractor_management import (
            SubcontractorType,
        )
        sub = manager.register_subcontractor(
            name="New Provider",
            subcontractor_type=SubcontractorType.DATA_PROVIDER,
            services_provided=["market_data"],
        )
        assert sub is not None
        assert sub.subcontractor_name == "New Provider"

    def test_record_change(self, manager):
        """Test recording subcontractor change."""
        from services.dora_integration.third_party.subcontractor_management import (
            ChangeType,
        )
        subs = manager.get_all_subcontractors()
        if subs:
            change = manager.record_change(
                subcontractor_id=subs[0].subcontractor_id,
                change_type=ChangeType.SERVICE_CHANGE,
                change_summary="Test change",
            )
            assert change is not None

    def test_export_for_client_roi(self, manager):
        """Test ROI export."""
        export = manager.export_for_client_roi()
        assert "subcontractors" in export
        assert "its_template" in export


class TestCTPPOversightBasicOperations:
    """Test basic operations for CTPP oversight."""

    @pytest.fixture
    def manager(self):
        """Create CTPP oversight manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from services.dora_integration.third_party.ctpp_oversight import (
                DORACtppOversight,
                CTPPOversightConfig,
            )
            config = CTPPOversightConfig(log_path=tmpdir)
            yield DORACtppOversight(config)

    def test_manager_has_methods(self, manager):
        """Test manager has expected methods."""
        # Check manager has key methods
        assert hasattr(manager, 'get_ctpp_designation')
        assert hasattr(manager, 'get_all_designated_ctpps')

    def test_get_all_designations(self, manager):
        """Test getting all designations."""
        designations = manager.get_all_designated_ctpps()
        assert isinstance(designations, list)


class TestThirdPartyRiskBasicOperations:
    """Test basic operations for third-party risk."""

    @pytest.fixture
    def manager(self):
        """Create third-party risk manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from services.dora_integration.third_party.third_party_risk import (
                DORAThirdPartyRiskManagement,
                ThirdPartyRiskConfig,
            )
            config = ThirdPartyRiskConfig(log_path=tmpdir)
            yield DORAThirdPartyRiskManagement(config)

    def test_manager_has_methods(self, manager):
        """Test manager has expected methods."""
        assert hasattr(manager, 'get_provider')
        assert hasattr(manager, 'get_critical_providers')
        assert hasattr(manager, 'get_providers_by_type')


class TestThirdPartyIncidentsBasicOperations:
    """Test basic operations for third-party incidents."""

    @pytest.fixture
    def manager(self):
        """Create third-party incidents manager."""
        from services.dora_integration.third_party.third_party_incidents import (
            DORAThirdPartyIncidents,
        )
        yield DORAThirdPartyIncidents(
            entity_id="TEST-001",
            entity_name="Test Entity",
        )

    def test_manager_has_methods(self, manager):
        """Test manager has expected methods."""
        assert hasattr(manager, 'get_incident')
        assert hasattr(manager, 'get_active_incidents')
        assert hasattr(manager, 'escalate_incident')


class TestMainPackageExports:
    """Test main package exports all Phase 3 modules."""

    def test_all_managers_exported(self):
        """Test all managers are exported from main package."""
        from services.dora_integration import (
            DORAConcentrationRisk,
            DORACtppOversight,
            DORAThirdPartyRiskManagement,
            DORAThirdPartyIncidents,
            DORASubcontractorManagement,
        )
        assert DORAConcentrationRisk is not None
        assert DORACtppOversight is not None
        assert DORAThirdPartyRiskManagement is not None
        assert DORAThirdPartyIncidents is not None
        assert DORASubcontractorManagement is not None

    def test_all_configs_exported(self):
        """Test all configs are exported from main package."""
        from services.dora_integration import (
            ConcentrationRiskConfig,
            CTPPOversightConfig,
            ThirdPartyRiskConfig,
            SubcontractorConfig,
        )
        assert ConcentrationRiskConfig is not None
        assert CTPPOversightConfig is not None
        assert ThirdPartyRiskConfig is not None
        assert SubcontractorConfig is not None

    def test_factory_functions_exported(self):
        """Test factory functions are exported."""
        from services.dora_integration import (
            create_concentration_risk,
            create_ctpp_oversight,
            create_third_party_risk_management,
            create_third_party_incidents,
            create_subcontractor_management,
        )
        assert create_concentration_risk is not None
        assert create_ctpp_oversight is not None
        assert create_third_party_risk_management is not None
        assert create_third_party_incidents is not None
        assert create_subcontractor_management is not None


class TestBackwardCompatibility:
    """Test backward compatibility with previous phases."""

    def test_phase1_still_works(self):
        """Test Phase 1 imports still work."""
        from services.dora_integration import (
            DORAuditReadiness,
            DORAProviderInfoPackage,
            PooledAuditSupport,
        )
        assert DORAuditReadiness is not None
        assert DORAProviderInfoPackage is not None
        assert PooledAuditSupport is not None

    def test_phase2_still_works(self):
        """Test Phase 2 imports still work."""
        from services.dora_integration import (
            ClientNotificationService,
            DORAIncidentClassification,
            DORAIncidentReporter,
            DORACommunication,
        )
        assert ClientNotificationService is not None
        assert DORAIncidentClassification is not None
        assert DORAIncidentReporter is not None
        assert DORACommunication is not None

    def test_version_updated(self):
        """Test version is at least 1.2.0 (Phase 3+)."""
        from services.dora_integration import __version__, __migration_phase__

        # Version should be at least 1.2.0 (Phase 3 or later)
        # With Phase 8 complete, version is 2.0.0
        major, minor, patch = map(int, __version__.split("."))
        assert major >= 2 or (major >= 1 and minor >= 2)
        # Migration phase should be at least 3 (Phase 3 or later)
        assert __migration_phase__ >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
