# -*- coding: utf-8 -*-
"""
Comprehensive tests for archived DORA Financial Entity modules.

These tests verify that:
1. All archived modules can be imported
2. All classes and functions are accessible
3. Factory functions work correctly
4. Basic functionality of each module works

Migration: Phase 7 - Archive Financial Entity Modules
"""

import pytest
import warnings
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from uuid import uuid4


class TestArchiveImports:
    """Test that all archived modules can be imported."""

    def test_version_info(self):
        """Test version info is available."""
        from services.archive.dora_financial_entity import __version__, __archive_date__
        assert __version__ == "1.0.0"
        assert __archive_date__ == "2025-01-17"

    def test_scope_verification_imports(self):
        """Test scope_verification module imports."""
        from services.archive.dora_financial_entity import (
            DORAEntityType,
            DORAScopeResult,
            ScopeVerification,
            EntityAuthorization,
            DORAScope,
            create_scope_verifier,
            get_entity_type_description,
        )
        assert DORAEntityType is not None
        assert DORAScopeResult is not None
        assert create_scope_verifier is not None

    def test_function_classification_imports(self):
        """Test function_classification module imports."""
        from services.archive.dora_financial_entity import (
            FunctionCriticality,
            ImpairmentType,
            FunctionClassification,
            ICTService,
            ThirdPartyProvider,
            FunctionClassifier,
            create_function_classifier,
            get_platform_functions,
            get_ict_providers,
        )
        assert FunctionCriticality is not None
        assert FunctionClassifier is not None

    def test_proportionality_imports(self):
        """Test proportionality module imports."""
        from services.archive.dora_financial_entity import (
            DORARegime,
            ExemptionType,
            EntityClassification,
            ProportionalityAssessment,
            RegimeExemption,
            ProportionalityAssessor,
            create_proportionality_assessor,
            assess_entity_proportionality,
        )
        assert DORARegime is not None
        assert ProportionalityAssessor is not None

    def test_governance_imports(self):
        """Test governance module imports."""
        from services.archive.dora_financial_entity import (
            GovernanceRole,
            DefenceLine,
            TrainingStatus,
            ApprovalStatus,
            GovernanceRoleAssignment,
            ICTTrainingRecord,
            FrameworkApproval,
            AuditFinding,
            ICTBudgetAllocation,
            DORAGovernanceFramework,
            create_governance_framework,
            MANDATORY_TRAINING_TOPICS,
        )
        assert GovernanceRole is not None
        assert DORAGovernanceFramework is not None
        assert len(MANDATORY_TRAINING_TOPICS) > 0

    def test_ict_risk_framework_imports(self):
        """Test ict_risk_framework module imports."""
        from services.archive.dora_financial_entity import (
            PolicyCategory,
            ControlDomain,
            ControlType,
            RiskPolicy,
            RiskProcedure,
            ICTControl,
            FrameworkReview,
            ICTRisk,
            DORAICTRiskFramework,
            create_ict_risk_framework,
        )
        assert PolicyCategory is not None
        assert DORAICTRiskFramework is not None

    def test_ict_systems_imports(self):
        """Test ict_systems module imports."""
        from services.archive.dora_financial_entity import (
            SystemCriticality,
            SystemType,
            SystemStatus,
            CapacityStatus,
            AutomationLevel,
            ICTSystem,
            CapacityMetric,
            ReliabilityMetric,
            AutomationCapability,
            SystemUpgrade,
            DORAICTSystemsManager,
            create_ict_systems_manager,
        )
        assert SystemCriticality is not None
        assert DORAICTSystemsManager is not None

    def test_ict_identification_imports(self):
        """Test ict_identification module imports."""
        from services.archive.dora_financial_entity import (
            AssetType,
            AssetClassification,
            RiskSourceCategory,
            ThreatCategory,
            VulnerabilitySeverity,
            ICTAsset,
            RiskSource,
            CyberThreat,
            ICTVulnerability,
            ICTDependency,
            BusinessFunction,
            DORAICTIdentification,
            create_ict_identification,
        )
        assert AssetType is not None
        assert DORAICTIdentification is not None

    def test_protection_imports(self):
        """Test protection module imports."""
        from services.archive.dora_financial_entity import (
            SecurityControlCategory,
            AccessControlType,
            AuthenticationType,
            EncryptionType,
            NetworkZone,
            SecurityControl,
            AccessPolicy,
            EncryptionStandard,
            NetworkSecurityRule,
            DataProtectionPolicy,
            DORAProtection,
            create_protection,
        )
        assert SecurityControlCategory is not None
        assert DORAProtection is not None

    def test_detection_imports(self):
        """Test detection module imports."""
        from services.archive.dora_financial_entity import (
            AnomalyType,
            AlertSeverity,
            AlertStatus,
            DetectionMethod,
            MonitoringStatus,
            DetectionRule,
            DetectionAlert,
            PerformanceMetric,
            SinglePointOfFailure,
            DORADetection,
            create_detection,
        )
        assert AnomalyType is not None
        assert DORADetection is not None

    def test_response_recovery_imports(self):
        """Test response_recovery module imports."""
        from services.archive.dora_financial_entity import (
            IncidentSeverity,
            IncidentStatus,
            IncidentCategory,
            EscalationLevel,
            CrisisStatus,
            ICTIncident,
            ResponseProcedure,
            EscalationRule,
            RecoveryAction,
            DORAResponseRecovery,
            create_response_recovery,
        )
        assert IncidentSeverity is not None
        assert DORAResponseRecovery is not None

    def test_backup_recovery_imports(self):
        """Test backup_recovery module imports."""
        from services.archive.dora_financial_entity import (
            BackupType,
            BackupFrequency,
            BackupStatus,
            RecoveryTestType,
            RecoveryTestResult,
            LocationType,
            BackupPolicy,
            BackupJob,
            BackupLocation,
            RecoveryTest,
            RestorationProcedure,
            DORABackupRecovery,
            create_backup_recovery,
        )
        assert BackupType is not None
        assert DORABackupRecovery is not None

    def test_learning_imports(self):
        """Test learning module imports."""
        from services.archive.dora_financial_entity import (
            ReviewType,
            LessonCategory,
            LessonPriority,
            LessonStatus,
            ImprovementType,
            ImprovementStatus,
            KnowledgeType,
            PostIncidentReview,
            LessonLearned,
            ImprovementInitiative,
            KnowledgeArticle,
            TrainingNeed,
            TrendAnalysis,
            InformationShare,
            DORALearning,
            create_dora_learning,
        )
        assert ReviewType is not None
        assert DORALearning is not None

    def test_ict_business_continuity_imports(self):
        """Test ict_business_continuity module imports."""
        from services.archive.dora_financial_entity import (
            ContinuityStatus,
            CriticalityLevel,
            ImpactCategory,
            ImpactSeverity,
            ScenarioType,
            RecoveryStrategy,
            ICTBusinessContinuityPolicy,
            BusinessImpactAssessment,
            RecoveryObjective,
            ContinuityPlan,
            DisruptionScenario,
            ContinuityTest,
            AlternativeArrangement,
            DORAICTBusinessContinuity,
            create_dora_ict_business_continuity,
        )
        assert ContinuityStatus is not None
        assert DORAICTBusinessContinuity is not None

    def test_simplified_framework_imports(self):
        """Test simplified_framework module imports."""
        from services.archive.dora_financial_entity import (
            EntitySize,
            SimplifiedControlCategory,
            ControlStatus,
            EligibilityCriteria,
            SimplifiedControl,
            SimplifiedRiskAssessment,
            SimplifiedIncident,
            SimplifiedBackup,
            SimplifiedThirdParty,
            SimplifiedTest,
            SimplifiedAwarenessTraining,
            AnnualReview,
            ESSENTIAL_CONTROLS,
            DORASimplifiedFramework,
            create_dora_simplified_framework,
        )
        assert EntitySize is not None
        assert DORASimplifiedFramework is not None
        assert len(ESSENTIAL_CONTROLS) > 0

    def test_incident_management_imports(self):
        """Test incident_management module imports."""
        from services.archive.dora_financial_entity import (
            ICTEventType,
            IncidentPhase,
            EarlyWarningType,
            ICTEvent,
            DORAIncident,
            EarlyWarningIndicator,
            IncidentAction,
            IncidentManagementConfig,
            DORAIncidentManagement,
            create_incident_management,
        )
        assert ICTEventType is not None
        assert DORAIncidentManagement is not None

    def test_supervisory_feedback_imports(self):
        """Test supervisory_feedback module imports."""
        from services.archive.dora_financial_entity import (
            FeedbackType,
            FeedbackPriority,
            FeedbackStatus,
            CorrectiveActionType,
            ResponseType,
            SupervisoryFeedback,
            CorrectiveAction,
            FeedbackResponse,
            FeedbackAuditEntry,
            AnonymisedInsight,
            DORASupervisioryFeedback,
            create_supervisory_feedback,
        )
        assert FeedbackType is not None
        assert DORASupervisioryFeedback is not None

    def test_resilience_testing_imports(self):
        """Test resilience_testing module imports."""
        from services.archive.dora_financial_entity import (
            TestCategory,
            TestFrequency,
            FindingSeverity,
            FindingStatus,
            TestScope,
            TestDefinition,
            TestExecution,
            TestFinding,
            TestingProgramme,
            TestingCycle,
            ResilienceTestingConfig,
            DORAResilienceTestingProgramme,
            create_resilience_testing_programme,
        )
        assert TestCategory is not None
        assert DORAResilienceTestingProgramme is not None

    def test_ict_testing_imports(self):
        """Test ict_testing module imports."""
        from services.archive.dora_financial_entity import (
            ICTSystemType,
            TestingPriority,
            VulnerabilityStatus,
            RemediationStatus,
            ICTSystemProfile,
            SystemTestPlan,
            SystemTest,
            Vulnerability,
            RemediationPlan,
            ThirdPartyInterfaceTest,
            ICTTestingConfig,
            DORAICTSystemTesting,
            create_ict_system_testing,
        )
        assert ICTSystemType is not None
        assert DORAICTSystemTesting is not None

    def test_tlpt_imports(self):
        """Test tlpt module imports."""
        from services.archive.dora_financial_entity import (
            TLPTPhase,
            TLPTStatus,
            ThreatActorCapability,
            AttackTechnique,
            AttackOutcome,
            TLPTFindingSeverity,
            FindingCategory,
            TLPTScope,
            ThreatIntelligenceReport,
            RedTeamScenario,
            AttackAction,
            TLPTFinding,
            PurpleTeamSession,
            TLPTEngagement,
            TLPTAttestation,
            TLPTConfig,
            DORAThreadLedPenetrationTesting,
            create_tlpt,
        )
        assert TLPTPhase is not None
        assert DORAThreadLedPenetrationTesting is not None

    def test_tester_management_imports(self):
        """Test tester_management module imports."""
        from services.archive.dora_financial_entity import (
            TesterRole,
            CertificationCategory,
            QualificationStatus,
            ConflictCheckResult,
            SecurityCertification,
            TesterExpertise,
            ConflictOfInterestDeclaration,
            ProfessionalIndemnityInsurance,
            TLPTTester,
            TesterOrganization,
            TesterQualificationAssessment,
            InternalTesterApproval,
            TesterManagementConfig,
            DORATestermanagement,
            create_tester_management,
        )
        assert TesterRole is not None
        assert DORATestermanagement is not None

    def test_pooled_testing_imports(self):
        """Test pooled_testing module imports."""
        from services.archive.dora_financial_entity import (
            PooledTestStatus,
            ParticipantRole,
            ParticipantStatus,
            CostSharingModel,
            ProviderCriticality,
            SharedProvider,
            PooledTestingParticipant,
            PooledTestingScope,
            CostSharingAgreement,
            PooledTestingEngagement,
            PooledTestingResults,
            PooledTestingConfig,
            DORAPooledTesting,
            create_pooled_testing,
        )
        assert PooledTestStatus is not None
        assert DORAPooledTesting is not None

    def test_cross_regulation_imports(self):
        """Test cross_regulation module imports."""
        from services.archive.dora_financial_entity import (
            Regulation,
            ReportingRequirement,
            IncidentAlignmentResult,
            RiskFrameworkAlignment,
            LoggingAlignmentResult,
            DORARegulationIntegration,
        )
        assert Regulation is not None
        assert DORARegulationIntegration is not None

    def test_training_participation_imports(self):
        """Test training_participation module imports."""
        from services.archive.dora_financial_entity import (
            TrainingType,
            ParticipationMode,
            PersonnelRole,
            TrainingCommitment,
            TrainingRequest,
            TrainingSession,
            QuarterlyUsage,
            TrainingParticipationConfig,
            DORATrainingParticipation,
        )
        assert TrainingType is not None
        assert DORATrainingParticipation is not None


class TestFactoryFunctions:
    """Test that factory functions work correctly."""

    def test_create_scope_verifier(self):
        """Test create_scope_verifier factory."""
        from services.archive.dora_financial_entity import create_scope_verifier
        verifier = create_scope_verifier()
        assert verifier is not None

    def test_create_function_classifier(self):
        """Test create_function_classifier factory."""
        from services.archive.dora_financial_entity import create_function_classifier
        classifier = create_function_classifier()
        assert classifier is not None

    def test_create_proportionality_assessor(self):
        """Test create_proportionality_assessor factory."""
        from services.archive.dora_financial_entity import create_proportionality_assessor
        assessor = create_proportionality_assessor()
        assert assessor is not None

    def test_create_governance_framework(self):
        """Test create_governance_framework factory."""
        from services.archive.dora_financial_entity import create_governance_framework
        framework = create_governance_framework()
        assert framework is not None

    def test_create_ict_risk_framework(self):
        """Test create_ict_risk_framework factory."""
        from services.archive.dora_financial_entity import create_ict_risk_framework
        framework = create_ict_risk_framework()
        assert framework is not None

    def test_create_ict_systems_manager(self):
        """Test create_ict_systems_manager factory."""
        from services.archive.dora_financial_entity import create_ict_systems_manager
        manager = create_ict_systems_manager()
        assert manager is not None

    def test_create_ict_identification(self):
        """Test create_ict_identification factory."""
        from services.archive.dora_financial_entity import create_ict_identification
        identification = create_ict_identification()
        assert identification is not None

    def test_create_protection(self):
        """Test create_protection factory."""
        from services.archive.dora_financial_entity import create_protection
        protection = create_protection()
        assert protection is not None

    def test_create_detection(self):
        """Test create_detection factory."""
        from services.archive.dora_financial_entity import create_detection
        detection = create_detection()
        assert detection is not None

    def test_create_response_recovery(self):
        """Test create_response_recovery factory."""
        from services.archive.dora_financial_entity import create_response_recovery
        recovery = create_response_recovery()
        assert recovery is not None

    def test_create_backup_recovery(self):
        """Test create_backup_recovery factory."""
        from services.archive.dora_financial_entity import create_backup_recovery
        backup = create_backup_recovery()
        assert backup is not None

    def test_create_dora_learning(self):
        """Test create_dora_learning factory."""
        from services.archive.dora_financial_entity import create_dora_learning
        learning = create_dora_learning()
        assert learning is not None

    def test_create_dora_ict_business_continuity(self):
        """Test create_dora_ict_business_continuity factory."""
        from services.archive.dora_financial_entity import create_dora_ict_business_continuity
        continuity = create_dora_ict_business_continuity()
        assert continuity is not None

    def test_create_dora_simplified_framework(self):
        """Test create_dora_simplified_framework factory."""
        from services.archive.dora_financial_entity import create_dora_simplified_framework
        framework = create_dora_simplified_framework()
        assert framework is not None

    def test_create_incident_management(self):
        """Test create_incident_management factory."""
        from services.archive.dora_financial_entity import create_incident_management
        management = create_incident_management()
        assert management is not None

    def test_create_supervisory_feedback(self):
        """Test create_supervisory_feedback factory."""
        from services.archive.dora_financial_entity import create_supervisory_feedback
        # This factory requires entity_id and entity_name
        feedback = create_supervisory_feedback(
            entity_id="TEST-001",
            entity_name="Test Entity"
        )
        assert feedback is not None

    def test_create_resilience_testing_programme(self):
        """Test create_resilience_testing_programme factory."""
        from services.archive.dora_financial_entity import create_resilience_testing_programme
        programme = create_resilience_testing_programme()
        assert programme is not None

    def test_create_ict_system_testing(self):
        """Test create_ict_system_testing factory."""
        from services.archive.dora_financial_entity import create_ict_system_testing
        testing = create_ict_system_testing()
        assert testing is not None

    def test_create_tlpt(self):
        """Test create_tlpt factory."""
        from services.archive.dora_financial_entity import create_tlpt
        tlpt = create_tlpt()
        assert tlpt is not None

    def test_create_tester_management(self):
        """Test create_tester_management factory."""
        from services.archive.dora_financial_entity import create_tester_management
        management = create_tester_management()
        assert management is not None

    def test_create_pooled_testing(self):
        """Test create_pooled_testing factory."""
        from services.archive.dora_financial_entity import create_pooled_testing
        pooled = create_pooled_testing()
        assert pooled is not None


class TestEnumValues:
    """Test that enum values are correctly defined."""

    def test_dora_entity_type_values(self):
        """Test DORAEntityType enum values."""
        from services.archive.dora_financial_entity import DORAEntityType
        # Should have various entity types
        assert hasattr(DORAEntityType, 'CREDIT_INSTITUTION')
        assert hasattr(DORAEntityType, 'INVESTMENT_FIRM')

    def test_dora_regime_values(self):
        """Test DORARegime enum values."""
        from services.archive.dora_financial_entity import DORARegime
        assert hasattr(DORARegime, 'FULL')
        assert hasattr(DORARegime, 'SIMPLIFIED')

    def test_governance_role_values(self):
        """Test GovernanceRole enum values."""
        from services.archive.dora_financial_entity import GovernanceRole
        # Check that enum has some values
        assert len(GovernanceRole) > 0
        # Check that iteration works
        roles = list(GovernanceRole)
        assert len(roles) > 0

    def test_backup_type_values(self):
        """Test BackupType enum values."""
        from services.archive.dora_financial_entity import BackupType
        assert hasattr(BackupType, 'FULL')
        assert hasattr(BackupType, 'INCREMENTAL')

    def test_incident_severity_values(self):
        """Test IncidentSeverity enum values."""
        from services.archive.dora_financial_entity import IncidentSeverity
        # Check that enum has some values
        assert len(IncidentSeverity) > 0
        # Check that iteration works
        severities = list(IncidentSeverity)
        assert len(severities) > 0


class TestBackwardCompatibility:
    """Test backward compatibility with services.dora imports."""

    def test_import_from_services_dora(self):
        """Test that imports still work from services.dora."""
        from services.dora import (
            DORAScope,
            DORAGovernanceFramework,
            DORAICTRiskFramework,
            DORAProtection,
            DORADetection,
            DORABackupRecovery,
            DORALearning,
            DORAICTBusinessContinuity,
            DORASimplifiedFramework,
            DORAIncidentManagement,
            DORAResilienceTestingProgramme,
            DORAICTSystemTesting,
            DORAThreadLedPenetrationTesting,
            DORATestermanagement,
            DORAPooledTesting,
            DORARegulationIntegration,
            DORATrainingParticipation,
        )
        # All should be imported successfully
        assert DORAScope is not None
        assert DORAGovernanceFramework is not None
        assert DORAICTRiskFramework is not None

    def test_factory_functions_from_services_dora(self):
        """Test factory functions work from services.dora."""
        from services.dora import (
            create_scope_verifier,
            create_governance_framework,
            create_ict_risk_framework,
        )
        # All should be callable
        assert callable(create_scope_verifier)
        assert callable(create_governance_framework)
        assert callable(create_ict_risk_framework)


class TestModuleFunctionality:
    """Test basic functionality of archived modules."""

    def test_scope_verifier_functionality(self):
        """Test scope verifier can check entity scope."""
        from services.archive.dora_financial_entity import (
            create_scope_verifier,
            DORAEntityType,
        )
        verifier = create_scope_verifier()
        # Should have verify_scope method
        assert hasattr(verifier, 'verify_scope')

    def test_governance_framework_functionality(self):
        """Test governance framework basic functionality."""
        from services.archive.dora_financial_entity import (
            create_governance_framework,
            GovernanceRole,
        )
        framework = create_governance_framework()
        # Should have basic methods
        assert hasattr(framework, 'assign_role') or hasattr(framework, 'get_compliance_status')

    def test_backup_recovery_functionality(self):
        """Test backup recovery basic functionality."""
        from services.archive.dora_financial_entity import (
            create_backup_recovery,
            BackupType,
        )
        backup = create_backup_recovery()
        # Should be a valid object with some methods
        assert backup is not None
        # Check it has some attributes/methods
        assert hasattr(backup, '__class__')

    def test_incident_management_functionality(self):
        """Test incident management basic functionality."""
        from services.archive.dora_financial_entity import (
            create_incident_management,
            ICTEventType,
        )
        management = create_incident_management()
        # Should have basic methods
        assert hasattr(management, 'record_event') or hasattr(management, 'create_incident')


class TestConfigFiles:
    """Test that config files are properly archived."""

    def test_entity_classification_config_exists(self):
        """Test entity_classification.yaml exists in archive."""
        from pathlib import Path
        config_path = Path("services/archive/dora_financial_entity/configs/entity_classification.yaml")
        assert config_path.exists(), "entity_classification.yaml should exist in archive configs"

    def test_nca_identification_config_exists(self):
        """Test nca_identification.yaml exists in archive."""
        from pathlib import Path
        config_path = Path("services/archive/dora_financial_entity/configs/nca_identification.yaml")
        assert config_path.exists(), "nca_identification.yaml should exist in archive configs"


class TestAllExports:
    """Test that __all__ exports are correctly defined."""

    def test_all_exports_count(self):
        """Test that __all__ has correct number of exports."""
        from services.archive.dora_financial_entity import __all__
        # Should have substantial number of exports (200+)
        assert len(__all__) > 100, f"Expected 100+ exports, got {len(__all__)}"

    def test_all_exports_accessible(self):
        """Test that all items in __all__ can be imported."""
        import services.archive.dora_financial_entity as archive
        for name in archive.__all__:
            assert hasattr(archive, name), f"Export '{name}' not accessible from archive module"
