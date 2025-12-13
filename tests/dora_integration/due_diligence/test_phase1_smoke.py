# -*- coding: utf-8 -*-
"""
Smoke tests for DORA Integration Layer Phase 1: Due Diligence & Audit Layer.

These tests verify:
1. All imports work correctly from the new location
2. Backward compatibility with services.dora imports
3. Basic functionality of key classes

Coverage: Smoke tests for Phase 1 migration verification
"""

import pytest
from datetime import datetime, timedelta, timezone


class TestPhase1ModuleImports:
    """Test that all Phase 1 modules can be imported from new location."""

    def test_import_audit_readiness_from_due_diligence(self):
        """Test importing audit_readiness from due_diligence."""
        from services.dora_integration.due_diligence.audit_readiness import (
            AUDIT_SLA_ACKNOWLEDGMENT_DAYS,
            AUDIT_SLA_SCHEDULING_DAYS,
            AUDIT_TYPE_SLAS,
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
            DORAuditReadiness,
            create_audit_readiness,
            get_standard_evidence_templates,
            IncidentNotificationStatus,
            ClientNotificationRecord,
            MultiClientIncident,
            MultiClientIncidentCoordinator,
            create_incident_coordinator,
        )
        assert DORAuditReadiness is not None
        assert AuditType is not None

    def test_import_provider_info_package_from_due_diligence(self):
        """Test importing provider_info_package from due_diligence."""
        from services.dora_integration.due_diligence.provider_info_package import (
            ICTServiceType,
            FunctionCriticality,
            SubstitutabilityLevel,
            DataSensitivity,
            ProviderIdentification,
            ServiceDescription,
            DataLocation,
            SubcontractorInfo,
            CertificationInfo,
            ContractSummary,
            ProviderInfoPackage,
            ProviderInfoConfig,
            DORAProviderInfoPackage,
            create_provider_info_package,
        )
        assert DORAProviderInfoPackage is not None
        assert ICTServiceType is not None

    def test_import_pooled_audit_support_from_due_diligence(self):
        """Test importing pooled_audit_support from due_diligence."""
        from services.dora_integration.due_diligence.pooled_audit_support import (
            AuditReportType,
            PooledAuditStatus,
            ParticipationStatus,
            AuditScopeArea,
            FindingSeverity,
            RemediationStatus,
            CertificationRecord,
            PooledAuditParticipant,
            AuditFinding,
            PooledAuditEngagement,
            AuditReportAccess,
            PooledAuditConfig,
            PooledAuditSupport,
            create_pooled_audit_support,
            get_audit_scope_areas,
            get_report_types,
        )
        assert PooledAuditSupport is not None
        assert AuditReportType is not None

    def test_import_compliance_dashboard_from_due_diligence(self):
        """Test importing compliance_dashboard from due_diligence."""
        from services.dora_integration.due_diligence.compliance_dashboard import (
            IssueSeverity,
            IssueStatus,
            DeadlineStatus,
            ComplianceIssue,
            Deadline,
            ComplianceStatus,
            DORAComplianceReport,
            DORAComplianceDashboard,
        )
        assert DORAComplianceDashboard is not None
        assert IssueSeverity is not None


class TestPhase1DoraIntegrationImports:
    """Test imports from dora_integration package."""

    def test_import_all_from_dora_integration(self):
        """Test importing all Phase 1 exports from dora_integration."""
        from services.dora_integration import (
            # Audit Readiness
            DORAuditReadiness,
            AuditType,
            AuditStatus,
            create_audit_readiness,
            # Provider Info
            DORAProviderInfoPackage,
            ICTServiceType,
            create_provider_info_package,
            # Pooled Audit
            PooledAuditSupport,
            AuditReportType,
            create_pooled_audit_support,
            # Compliance Dashboard
            DORAComplianceDashboard,
            IssueSeverity,
        )
        assert all([
            DORAuditReadiness,
            DORAProviderInfoPackage,
            PooledAuditSupport,
            DORAComplianceDashboard,
        ])


class TestPhase1BackwardCompatibility:
    """Test backward compatibility with services.dora imports."""

    def test_audit_readiness_backward_compat(self):
        """Test audit_readiness imports from services.dora."""
        from services.dora import (
            DORAuditReadiness,
            AuditType,
            AuditStatus,
            EvidenceType,
            create_audit_readiness,
        )
        service = create_audit_readiness()
        assert isinstance(service, DORAuditReadiness)

    def test_pooled_audit_support_backward_compat(self):
        """Test pooled_audit_support imports from services.dora."""
        from services.dora import (
            PooledAuditSupport,
            AuditReportType,
            PooledAuditStatus,
            create_pooled_audit_support,
        )
        service = create_pooled_audit_support()
        assert isinstance(service, PooledAuditSupport)

    def test_compliance_dashboard_backward_compat(self):
        """Test compliance_dashboard imports from services.dora."""
        from services.dora import (
            DORAComplianceDashboard,
            IssueSeverity,
            DeadlineStatus,
            ComplianceIssue,
            Deadline,
        )
        dashboard = DORAComplianceDashboard(current_phase=4)
        assert dashboard is not None

    def test_incident_coordinator_backward_compat(self):
        """Test incident coordinator imports from services.dora."""
        from services.dora import (
            MultiClientIncidentCoordinator,
            create_incident_coordinator,
        )
        coordinator = create_incident_coordinator()
        assert isinstance(coordinator, MultiClientIncidentCoordinator)


class TestPhase1BasicFunctionality:
    """Test basic functionality of Phase 1 modules."""

    def test_audit_readiness_create_request(self):
        """Test creating audit request."""
        from services.dora_integration.due_diligence import (
            DORAuditReadiness,
            AuditType,
            AuditScope,
        )
        service = DORAuditReadiness()
        request = service.create_audit_request(
            requesting_entity="TEST-BANK-001",
            requesting_entity_type="FINANCIAL_ENTITY",
            audit_type=AuditType.CLIENT_OPERATIONAL,
            audit_scope=AuditScope.FULL,
            audit_title="Annual Operational Audit 2025",
        )
        assert request.request_id
        assert request.requesting_entity == "TEST-BANK-001"

    def test_pooled_audit_support_register_certification(self):
        """Test registering certification."""
        from services.dora_integration.due_diligence import (
            PooledAuditSupport,
            AuditReportType,
            AuditScopeArea,
        )
        service = PooledAuditSupport()
        cert = service.register_certification(
            certification_type=AuditReportType.ISO27001,
            certifying_body="TUV Rheinland",
            issue_date="2024-03-15",
            expiry_date="2027-03-14",
            scope_areas=[AuditScopeArea.ICT_SECURITY],
            scope_description="ISMS for ICT services",
        )
        assert cert.certification_id

    def test_compliance_dashboard_basic_operations(self):
        """Test basic compliance dashboard operations."""
        from services.dora_integration.due_diligence import (
            DORAComplianceDashboard,
            ComplianceIssue,
            Deadline,
            IssueSeverity,
            DeadlineStatus,
        )
        dashboard = DORAComplianceDashboard(current_phase=4, target_phase=5)

        # Add deadline
        deadline = dashboard.register_deadline(
            Deadline(
                name="ROI Submission",
                due_date=datetime(2025, 4, 30, tzinfo=timezone.utc),
                regulation="DORA",
                status=DeadlineStatus.UPCOMING,
            )
        )
        assert deadline.name == "ROI Submission"

        # Add issue
        issue = dashboard.add_issue(
            ComplianceIssue(
                description="Complete ROI template",
                severity=IssueSeverity.HIGH,
                owner="Compliance Team",
                due_date=datetime.now(timezone.utc) + timedelta(days=30),
            )
        )
        assert issue.issue_id

        # Get status
        status = dashboard.get_compliance_status()
        assert status.current_phase == 4

    def test_incident_coordinator_create_incident(self):
        """Test creating incident with coordinator."""
        from services.dora_integration.due_diligence import (
            create_incident_coordinator,
        )
        coordinator = create_incident_coordinator()
        incident = coordinator.create_incident(
            incident_type="SERVICE_DEGRADATION",
            severity="MEDIUM",
            summary="API response times elevated",
            affected_clients=["CLIENT-001", "CLIENT-002"],
            details="Performance issues affecting analytics platform",
        )
        assert incident.incident_id

    def test_provider_info_package_generation(self):
        """Test provider info package generation."""
        from services.dora_integration.due_diligence import (
            DORAProviderInfoPackage,
            ServiceDescription,
            ICTServiceType,
        )
        service = DORAProviderInfoPackage()
        # Add service to package using ServiceDescription object
        svc_desc = ServiceDescription(
            service_id="SVC-001",
            service_name="Test Analytics",
            service_type=ICTServiceType.DATA_ANALYTICS,
            service_description="AI-powered analytics platform",
        )
        service.add_service(svc_desc)
        package = service.generate_package(
            client_id="CLIENT-001",
            client_name="Test Financial Entity"
        )
        assert package.package_id


class TestPhase1ClassInstantiation:
    """Test that all major classes can be instantiated."""

    def test_dora_audit_readiness_instantiation(self):
        """Test DORAuditReadiness instantiation."""
        from services.dora_integration.due_diligence import DORAuditReadiness
        service = DORAuditReadiness()
        assert service is not None
        assert service.config is not None

    def test_pooled_audit_support_instantiation(self):
        """Test PooledAuditSupport instantiation."""
        from services.dora_integration.due_diligence import PooledAuditSupport
        service = PooledAuditSupport()
        assert service is not None

    def test_compliance_dashboard_instantiation(self):
        """Test DORAComplianceDashboard instantiation."""
        from services.dora_integration.due_diligence import DORAComplianceDashboard
        dashboard = DORAComplianceDashboard(current_phase=3)
        assert dashboard is not None
        assert dashboard.current_phase == 3

    def test_provider_info_package_instantiation(self):
        """Test DORAProviderInfoPackage instantiation."""
        from services.dora_integration.due_diligence import DORAProviderInfoPackage
        service = DORAProviderInfoPackage()
        assert service is not None

    def test_incident_coordinator_instantiation(self):
        """Test MultiClientIncidentCoordinator instantiation."""
        from services.dora_integration.due_diligence import create_incident_coordinator
        coordinator = create_incident_coordinator()
        assert coordinator is not None


class TestPhase1FactoryFunctions:
    """Test factory functions work correctly."""

    def test_create_audit_readiness_factory(self):
        """Test create_audit_readiness factory function."""
        from services.dora_integration.due_diligence import create_audit_readiness
        service = create_audit_readiness()
        assert service is not None

    def test_create_pooled_audit_support_factory(self):
        """Test create_pooled_audit_support factory function."""
        from services.dora_integration.due_diligence import create_pooled_audit_support
        service = create_pooled_audit_support()
        assert service is not None

    def test_create_provider_info_package_factory(self):
        """Test create_provider_info_package factory function."""
        from services.dora_integration.due_diligence import create_provider_info_package
        service = create_provider_info_package()
        assert service is not None

    def test_create_incident_coordinator_factory(self):
        """Test create_incident_coordinator factory function."""
        from services.dora_integration.due_diligence import create_incident_coordinator
        coordinator = create_incident_coordinator()
        assert coordinator is not None

    def test_get_standard_evidence_templates_factory(self):
        """Test get_standard_evidence_templates factory function."""
        from services.dora_integration.due_diligence import get_standard_evidence_templates
        templates = get_standard_evidence_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0


class TestPhase1Aliases:
    """Test that aliases are correctly set up."""

    def test_service_description_alias(self):
        """Test ICTServiceDescription alias."""
        from services.dora_integration.due_diligence import (
            ServiceDescription,
            ICTServiceDescription,
        )
        assert ServiceDescription is ICTServiceDescription

    def test_data_location_alias(self):
        """Test DataLocationInfo alias."""
        from services.dora_integration.due_diligence import (
            DataLocation,
            DataLocationInfo,
        )
        assert DataLocation is DataLocationInfo

    def test_provider_generator_alias(self):
        """Test ProviderInfoPackageGenerator alias."""
        from services.dora_integration.due_diligence import (
            DORAProviderInfoPackage,
            ProviderInfoPackageGenerator,
        )
        assert DORAProviderInfoPackage is ProviderInfoPackageGenerator
