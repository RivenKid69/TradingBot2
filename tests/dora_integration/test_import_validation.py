# -*- coding: utf-8 -*-
"""
Import Validation Tests for DORA Integration Layer.

Validates that all imports work correctly after the Phase 8 migration.
Tests both direct imports from dora_integration and facade imports from dora.

Test Coverage:
    - All dora_integration subpackage imports
    - All class and function imports
    - Enum imports
    - Constant imports
    - Factory function accessibility
    - Cross-module dependencies
"""

import sys
from pathlib import Path
from typing import Any, List, Type

import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestDueDiligenceImports:
    """Test all imports from due_diligence subpackage."""

    def test_audit_readiness_classes(self) -> None:
        """Test audit_readiness class imports."""
        from services.dora_integration.due_diligence import (
            DORAuditReadiness,
            AuditRequest,
            EvidenceItem,
            AuditFinding,
            EvidenceTemplate,
            AuditReadinessConfig,
        )

        # Verify they are classes
        assert isinstance(DORAuditReadiness, type)
        assert isinstance(AuditRequest, type)
        assert isinstance(EvidenceItem, type)
        assert isinstance(AuditFinding, type)

    def test_audit_readiness_enums(self) -> None:
        """Test audit_readiness enum imports."""
        from services.dora_integration.due_diligence import (
            AuditType,
            AuditScope,
            AuditStatus,
            EvidenceType,
            EvidenceCategory,
        )

        # Verify they are enums with values
        assert hasattr(AuditType, "CLIENT_OPERATIONAL")
        assert hasattr(AuditScope, "FULL")
        assert hasattr(AuditStatus, "SCHEDULED")

    def test_audit_readiness_constants(self) -> None:
        """Test audit_readiness constant imports."""
        from services.dora_integration.due_diligence import (
            AUDIT_SLA_ACKNOWLEDGMENT_DAYS,
            AUDIT_SLA_SCHEDULING_DAYS,
            AUDIT_SLA_EVIDENCE_STANDARD_DAYS,
            AUDIT_SLA_EVIDENCE_COMPLEX_DAYS,
            AUDIT_SLA_NCA_RESPONSE_DAYS,
            EVIDENCE_RETENTION_YEARS,
            AUDIT_TYPE_SLAS,
        )

        assert isinstance(AUDIT_SLA_ACKNOWLEDGMENT_DAYS, int)
        assert isinstance(AUDIT_TYPE_SLAS, dict)

    def test_audit_readiness_factories(self) -> None:
        """Test audit_readiness factory function imports."""
        from services.dora_integration.due_diligence import (
            create_audit_readiness,
            get_standard_evidence_templates,
        )

        assert callable(create_audit_readiness)
        assert callable(get_standard_evidence_templates)

        # Test factory creates instance
        instance = create_audit_readiness()
        assert isinstance(instance, type(instance))

    def test_provider_info_package_imports(self) -> None:
        """Test provider_info_package imports."""
        from services.dora_integration.due_diligence import (
            DORAProviderInfoPackage,
            ProviderInfoPackageGenerator,
            ICTServiceType,
            FunctionCriticality,
            SubstitutabilityLevel,
            DataSensitivity,
            ProviderIdentification,
            ServiceDescription,
            ICTServiceDescription,
            DataLocation,
            DataLocationInfo,
            SubcontractorInfo,
            CertificationInfo,
            ContractSummary,
            ProviderInfoPackage,
            ProviderInfoConfig,
            create_provider_info_package,
        )

        assert DORAProviderInfoPackage is not None
        assert callable(create_provider_info_package)

    def test_pooled_audit_support_imports(self) -> None:
        """Test pooled_audit_support imports."""
        from services.dora_integration.due_diligence import (
            PooledAuditSupport,
            AuditReportType,
            PooledAuditStatus,
            ParticipationStatus,
            AuditScopeArea,
            FindingSeverity,
            RemediationStatus,
            CertificationRecord,
            PooledAuditParticipant,
            PooledAuditFinding,
            PooledAuditEngagement,
            AuditReportAccess,
            PooledAuditConfig,
            create_pooled_audit_support,
            get_audit_scope_areas,
            get_report_types,
        )

        assert PooledAuditSupport is not None
        areas = get_audit_scope_areas()
        assert isinstance(areas, list)

    def test_compliance_dashboard_imports(self) -> None:
        """Test compliance_dashboard imports."""
        from services.dora_integration.due_diligence import (
            DORAComplianceDashboard,
            IssueSeverity,
            IssueStatus,
            DeadlineStatus,
            ComplianceIssue,
            Deadline,
            ComplianceStatus,
            DORAComplianceReport,
        )

        assert DORAComplianceDashboard is not None
        assert hasattr(IssueSeverity, "CRITICAL")

    def test_multi_client_incident_imports(self) -> None:
        """Test multi-client incident coordination imports."""
        from services.dora_integration.due_diligence import (
            IncidentNotificationStatus,
            ClientNotificationRecord,
            MultiClientIncident,
            MultiClientIncidentCoordinator,
            create_incident_coordinator,
        )

        assert MultiClientIncidentCoordinator is not None
        assert callable(create_incident_coordinator)


class TestIncidentInterfaceImports:
    """Test all imports from incident_interface subpackage."""

    def test_client_notification_imports(self) -> None:
        """Test client_incident_notification imports."""
        from services.dora_integration.incident_interface import (
            ClientNotificationService,
            DORAClientNotification,
            ClientNotificationConfig,
            IncidentSeverity,
            NotificationStatus,
            NotificationChannel,
            IncidentCategory,
            ClientContact,
            IncidentNotification,
            IncidentUpdate,
            ClientIncident,
            create_client_notification_service,
            create_client_notification_system,
            get_notification_template,
        )

        assert ClientNotificationService is not None
        assert DORAClientNotification is not None
        assert callable(create_client_notification_service)

    def test_incident_classification_imports(self) -> None:
        """Test incident_classification imports."""
        from services.dora_integration.incident_interface import (
            DORAIncidentClassification,
            IncidentClassificationConfig,
            ClassificationThresholds,
            IncidentClassificationType,
            ClientType,
            DataType,
            CriticalServiceType,
            MajorIncidentTrigger,
            ReputationalImpactLevel,
            ClientImpactAssessment,
            DurationAssessment,
            GeographicAssessment,
            DataLossAssessment,
            CriticalServiceAssessment,
            EconomicImpactAssessment,
            ReputationalAssessment,
            RecurringIncidentAssessment,
            MaliciousAccessAssessment,
            IncidentClassificationResult,
            create_incident_classification,
            get_default_thresholds,
            get_classification_criteria,
        )

        assert DORAIncidentClassification is not None
        thresholds = get_default_thresholds()
        assert thresholds is not None

    def test_incident_reporting_imports(self) -> None:
        """Test incident_reporting imports."""
        from services.dora_integration.incident_interface import (
            DORAIncidentReporter,
            IncidentReportingConfig,
            ReportType,
            ReportStatus,
            IncidentTypeCode,
            RootCauseCategory,
            CompetentAuthorityType,
            CompetentAuthority,
            InitialNotificationReport,
            IntermediateReport,
            FinalReport,
            ClientDataPackage,
            ReportSubmission,
            create_incident_reporter,
            get_report_deadlines,
        )

        assert DORAIncidentReporter is not None
        assert hasattr(ReportType, "INITIAL_NOTIFICATION")
        deadlines = get_report_deadlines()
        assert isinstance(deadlines, dict)

    def test_cyber_threat_notification_imports(self) -> None:
        """Test cyber_threat_notification imports."""
        from services.dora_integration.incident_interface import (
            CyberThreatNotificationService,
            CyberThreatNotificationConfig,
            ThreatCategory,
            ThreatActorType,
            ThreatSeverity,
            ThreatStatus,
            ThreatSignificance,
            ThreatIndicator,
            CyberThreat,
            ThreatSignificanceAssessment,
            ThreatNotification,
            create_cyber_threat_notification_service,
            get_threat_categories,
            get_threat_severities,
        )

        assert CyberThreatNotificationService is not None
        categories = get_threat_categories()
        assert isinstance(categories, list)

    def test_communication_imports(self) -> None:
        """Test communication imports."""
        from services.dora_integration.incident_interface import (
            DORACommunication,
            CommunicationConfig,
            CommunicationChannel,
            StakeholderType,
            CommunicationPriority,
            CommunicationStatus,
            CrisisPhase,
            PolicyStatus,
            CommunicationContact,
            CommunicationTemplate,
            CommunicationRecord,
            CommunicationPolicy,
            CrisisStatus,
            create_communication_service,
            get_communication_channels,
            get_stakeholder_types,
            get_crisis_phases,
        )

        assert DORACommunication is not None
        channels = get_communication_channels()
        assert isinstance(channels, list)


class TestThirdPartyImports:
    """Test all imports from third_party subpackage."""

    def test_concentration_risk_imports(self) -> None:
        """Test concentration_risk imports."""
        from services.dora_integration.third_party import (
            DORAConcentrationRisk,
            ConcentrationRiskConfig,
            ConcentrationType,
            ConcentrationRiskLevel,
            MitigationStatus,
            AssessmentScope,
            ProviderDependency,
            ConcentrationMetric,
            ConcentrationRisk,
            MitigationMeasure,
            ConcentrationAssessment,
            DependencyMap,
            create_concentration_risk,
            get_concentration_types,
            get_substitutability_levels,
        )

        assert DORAConcentrationRisk is not None
        types = get_concentration_types()
        assert isinstance(types, list)

    def test_ctpp_oversight_imports(self) -> None:
        """Test ctpp_oversight imports."""
        from services.dora_integration.third_party import (
            DORACtppOversight,
            CTPPOversightConfig,
            LeadOverseer,
            CTPPStatus,
            OversightRecommendationType,
            RecommendationStatus,
            ComplianceLevel,
            OversightExerciseType,
            CTPPDesignation,
            OversightRecommendation,
            OversightExercise,
            CTPPRiskAssessment,
            CTPPContractRequirement,
            EntityCTPPRelationship,
            DESIGNATED_CTPPS_2025,
            create_ctpp_oversight,
            get_lead_overseers,
            get_designated_ctpps_list,
            get_ctpp_requirements,
            get_ctpp_contract_requirements,
        )

        assert DORACtppOversight is not None
        assert isinstance(DESIGNATED_CTPPS_2025, list)
        ctpps = get_designated_ctpps_list()
        assert isinstance(ctpps, list)

    def test_third_party_risk_imports(self) -> None:
        """Test third_party_risk imports."""
        from services.dora_integration.third_party import (
            DORAThirdPartyRiskManagement,
            ThirdPartyRiskConfig,
            ProviderType,
            ProviderCriticality,
            ServiceCriticality,
            ProviderStatus,
            RiskCategory,
            RiskLevel,
            DueDiligenceStatus,
            AssessmentType,
            TPRSubstitutabilityLevel,
            ICTService,
            ICTProvider,
            ThirdPartyRisk,
            ThirdPartyRiskAssessment,
            DueDiligenceCheck,
            ProviderRelationshipEvent,
            create_third_party_risk_management,
            get_provider_types,
            get_risk_categories,
            get_criticality_levels,
        )

        assert DORAThirdPartyRiskManagement is not None
        risk_cats = get_risk_categories()
        assert isinstance(risk_cats, list)

    def test_third_party_incidents_imports(self) -> None:
        """Test third_party_incidents imports."""
        from services.dora_integration.third_party import (
            DORAThirdPartyIncidents,
            ThirdPartyProviderType,
            ThirdPartyCriticality,
            ThirdPartyIncidentType,
            ContractualSLAStatus,
            EscalationLevel,
            ThirdPartyProvider,
            AffectedService,
            SLAAssessment,
            EscalationRecord,
            IncidentMitigationAction,
            ThirdPartyIncident,
            PostIncidentReview,
            create_third_party_incidents,
        )

        assert DORAThirdPartyIncidents is not None

    def test_subcontractor_management_imports(self) -> None:
        """Test subcontractor_management imports."""
        from services.dora_integration.third_party import (
            DORASubcontractorManagement,
            SubcontractorConfig,
            SubcontractorType,
            SubcontractorStatus,
            SubcontractorRiskLevel,
            ChangeType,
            ConsentMode,
            Subcontractor,
            SubcontractorChange,
            ClientSubcontractorPreference,
            SubcontractorRiskAssessment,
            create_subcontractor_management,
        )

        assert DORASubcontractorManagement is not None


class TestContractsImports:
    """Test all imports from contracts subpackage."""

    def test_contractual_requirements_imports(self) -> None:
        """Test contractual_requirements imports."""
        from services.dora_integration.contracts import (
            DORAContractualRequirements,
            ContractualRequirementsConfig,
            RequirementCategory,
            RequirementType,
            ComplianceStatus,
            GapSeverity,
            RemediationStatus,
            ContractStatus,
            ContractualRequirement,
            ContractProvision,
            ContractAssessment,
            ContractGap,
            ContractAmendment,
            SLADefinition,
            ICTContract,
            TerminationClause,
            create_contractual_requirements,
            get_article_30_requirements,
            get_requirement_types,
            get_basic_requirement_count,
            get_critical_requirement_count,
            get_termination_clause_templates,
        )

        assert DORAContractualRequirements is not None
        requirements = get_article_30_requirements()
        assert isinstance(requirements, list)

    def test_sla_guardrails_imports(self) -> None:
        """Test sla_guardrails imports."""
        from services.dora_integration.contracts import (
            SLAGuardrails,
            SLAGuardrailsConfig,
            SLATier,
            CapacityStatus,
            ApprovalStatus,
            InfrastructureRequirement,
            OnCallRequirement,
            SLATierDefinition,
            CapacityValidation,
            SLACommitmentRequest,
            CurrentCapacityState,
            create_sla_guardrails,
            get_sla_tier_definitions,
            get_sla_tiers,
        )

        assert SLAGuardrails is not None
        tiers = get_sla_tiers()
        assert isinstance(tiers, list)

    def test_exit_strategies_imports(self) -> None:
        """Test exit_strategies imports."""
        from services.dora_integration.contracts import (
            DORAExitStrategies,
            ExitStrategiesConfig,
            ExitTrigger,
            ExitPhase,
            ExitPlanStatus,
            TransitionType,
            ReadinessLevel,
            AlternativeProviderStatus,
            AlternativeProvider,
            DataMigrationPlan,
            TransitionTask,
            ExitRisk,
            ExitCostEstimate,
            ExitPlan,
            ExitExecution,
            ExitReadinessAssessment,
            create_exit_strategies,
            get_exit_triggers,
            get_exit_phases,
            get_transition_types,
        )

        assert DORAExitStrategies is not None
        triggers = get_exit_triggers()
        assert isinstance(triggers, list)


class TestReportingImports:
    """Test all imports from reporting subpackage."""

    def test_unified_reporting_imports(self) -> None:
        """Test unified_reporting imports."""
        from services.dora_integration.reporting import (
            UnifiedReportingManager,
            ReportChannel,
            PackageFormat,
            ReportDestination,
            ReportValidationResult,
            UnifiedReport,
            SubmissionPackage,
            DeliveryRecord,
            UnifiedReportingConfig,
            create_unified_reporting_manager,
            create_report_destination,
            get_report_statuses,
        )

        assert UnifiedReportingManager is not None
        statuses = get_report_statuses()
        assert isinstance(statuses, list)

    def test_reporting_templates_imports(self) -> None:
        """Test reporting_templates imports."""
        from services.dora_integration.reporting import (
            DORAReportingTemplates,
            IncidentTypeCode,
            DataTypeCode,
            ClientTypeCode,
            ServiceTypeCode,
            ResponseEffectivenessCode,
            TemplateExportFormat,
            ITSInitialNotificationTemplate,
            ITSIntermediateReportTemplate,
            ITSFinalReportTemplate,
            TimelineEvent,
            ClientIncidentDataPackage,
            create_reporting_templates,
            get_incident_type_codes,
            get_data_type_codes,
            get_service_type_codes,
            get_client_type_codes,
            create_timeline_event,
        )

        assert DORAReportingTemplates is not None
        codes = get_incident_type_codes()
        # Codes can be dict or list depending on implementation
        assert isinstance(codes, (list, dict))

    def test_register_of_information_imports(self) -> None:
        """Test register_of_information imports."""
        from services.dora_integration.reporting import (
            DORARegisterOfInformation,
            ContractType,
            ServiceType,
            FunctionType,
            ProviderLocationType,
            ExportFormat,
            ContractReferenceData,
            SubcontractorData,
            ServiceRecord,
            ROIDataPackage,
            ROIDataGeneratorConfig,
            create_register_of_information,
            create_roi_data_generator,
            get_contract_types,
            get_service_types,
            get_subcontracting_levels,
            get_its_templates_provided,
            get_its_templates_client_provides,
        )

        assert DORARegisterOfInformation is not None
        contract_types = get_contract_types()
        assert isinstance(contract_types, list)


class TestSharingImports:
    """Test all imports from sharing subpackage."""

    def test_information_sharing_imports(self) -> None:
        """Test information_sharing imports."""
        from services.dora_integration.sharing import (
            DORAInformationSharing,
            SHAREABLE_INFORMATION_TYPES,
            TLP_DEFINITIONS,
            DEFAULT_INTELLIGENCE_RETENTION_DAYS,
            NCA_NOTIFICATION_DEADLINE_DAYS,
            CommunityType,
            SharingChannel,
            TLPLevel,
            MembershipStatus,
            SharingOutcome,
            IntelligenceDirection,
            SanitizationLevel,
            SharingCommunity,
            InformationSharingPolicy,
            CyberThreatIntelligence,
            ThreatIntelligenceRecord,
            SharingAuditRecord,
            NCANotification,
            InformationSharingConfig,
            create_information_sharing,
            get_shareable_information_types,
            get_tlp_definitions,
            get_community_types,
            get_sharing_channels,
            get_tlp_levels,
            create_sharing_community,
            create_cyber_threat,
            create_sharing_policy,
        )

        assert DORAInformationSharing is not None
        # SHAREABLE_INFORMATION_TYPES can be set or list
        assert isinstance(SHAREABLE_INFORMATION_TYPES, (list, set))
        assert isinstance(TLP_DEFINITIONS, dict)
        assert isinstance(DEFAULT_INTELLIGENCE_RETENTION_DAYS, int)


class TestMainPackageImports:
    """Test imports from main dora_integration package."""

    def test_version_imports(self) -> None:
        """Test version info imports."""
        from services.dora_integration import __version__, __migration_phase__

        assert __version__ == "2.0.0"
        assert __migration_phase__ == 8

    def test_main_package_exports_all_phases(self) -> None:
        """Test main package exports items from all phases."""
        from services import dora_integration

        # Phase 1 exports
        assert hasattr(dora_integration, "DORAuditReadiness")
        assert hasattr(dora_integration, "PooledAuditSupport")

        # Phase 2 exports
        assert hasattr(dora_integration, "DORAIncidentClassification")
        assert hasattr(dora_integration, "DORACommunication")

        # Phase 3 exports
        assert hasattr(dora_integration, "DORAConcentrationRisk")
        assert hasattr(dora_integration, "DORAThirdPartyRiskManagement")

        # Phase 4 exports
        assert hasattr(dora_integration, "DORAContractualRequirements")
        assert hasattr(dora_integration, "SLAGuardrails")

        # Phase 5 exports
        assert hasattr(dora_integration, "UnifiedReportingManager")
        assert hasattr(dora_integration, "DORARegisterOfInformation")

        # Phase 6 exports
        assert hasattr(dora_integration, "DORAInformationSharing")
        assert hasattr(dora_integration, "TLPLevel")


class TestCrossModuleDependencies:
    """Test that cross-module imports work correctly."""

    def test_classification_to_notification_flow(self) -> None:
        """Test incident classification to notification flow."""
        from services.dora_integration.incident_interface import (
            DORAIncidentClassification,
            ClientNotificationService,
            IncidentSeverity,
        )

        # Both should be usable together
        classifier = DORAIncidentClassification()
        notifier = ClientNotificationService()

        assert classifier is not None
        assert notifier is not None

    def test_reporting_to_roi_flow(self) -> None:
        """Test reporting to ROI data generation flow."""
        from services.dora_integration.reporting import (
            UnifiedReportingManager,
            DORARegisterOfInformation,
        )

        reporter = UnifiedReportingManager()
        roi = DORARegisterOfInformation()

        assert reporter is not None
        assert roi is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
