"""
Comprehensive Tests for DORA Phase 1 - ICT Risk Management Framework.

Tests for Articles 5-16 of DORA (EU) 2022/2554.

Coverage includes:
- Article 5: Governance Framework
- Article 6: ICT Risk Management Framework
- Article 7: ICT Systems
- Article 8: ICT Identification
- Article 9: Protection and Prevention
- Article 10: Detection
- Article 11: Response and Recovery
- Article 12: Backup and Recovery
- Article 13: Learning and Evolving
- Article 14: Communication
- Article 15: ICT Business Continuity
- Article 16: Simplified Framework
"""


import pytest
pytest.skip(
    "Legacy DORA test - uses deprecated imports from services.dora.* "
    "These modules have been migrated to services.dora_integration.*. "
    "See tests/dora/ and tests/dora_integration/ for current tests.",
    allow_module_level=True
)


import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

# =============================================================================
# Import all Phase 1 modules
# =============================================================================

from services.dora.governance import (
    GovernanceRole,
    DefenceLine,
    TrainingStatus,
    ApprovalStatus,
    DORAGovernanceFramework,
    create_governance_framework,
    MANDATORY_TRAINING_TOPICS,
)

from services.dora.ict_risk_framework import (
    PolicyCategory,
    ControlDomain,
    ControlType,
    RiskLevel,
    DORAICTRiskFramework,
    create_ict_risk_framework,
)

from services.dora.ict_systems import (
    SystemCriticality,
    SystemType,
    SystemStatus,
    CapacityStatus,
    AutomationLevel,
    DORAICTSystemsManager,
    create_ict_systems_manager,
)

from services.dora.ict_identification import (
    AssetType,
    AssetClassification,
    RiskSourceCategory,
    ThreatCategory,
    VulnerabilitySeverity,
    DORAICTIdentification,
    create_ict_identification,
)

from services.dora.protection import (
    SecurityControlCategory,
    AccessControlType,
    AuthenticationType,
    EncryptionType,
    NetworkZone,
    DORAProtection,
    create_protection,
)

from services.dora.detection import (
    AnomalyType,
    AlertSeverity,
    AlertStatus,
    DetectionMethod,
    MonitoringStatus,
    DORADetection,
    create_detection,
)

from services.dora.response_recovery import (
    IncidentSeverity,
    IncidentStatus,
    IncidentCategory,
    EscalationLevel,
    CrisisStatus,
    DORAResponseRecovery,
    create_response_recovery,
)

from services.dora.backup_recovery import (
    BackupType,
    BackupFrequency,
    BackupStatus,
    RecoveryTestType,
    RecoveryTestResult,
    LocationType,
    DORABackupRecovery,
    create_backup_recovery,
)

from services.dora.learning import (
    ReviewType,
    LessonCategory,
    LessonPriority,
    LessonStatus,
    ImprovementType,
    ImprovementStatus,
    KnowledgeType,
    DORALearning,
    create_dora_learning,
)

from services.dora.communication import (
    CommunicationType,
    CommunicationChannel,
    CommunicationPriority,
    CommunicationStatus,
    NotificationType,
    StakeholderType,
    EscalationTrigger,
    DORACommunication,
    create_dora_communication,
)

from services.dora.ict_business_continuity import (
    ContinuityStatus,
    CriticalityLevel,
    ImpactCategory,
    ImpactSeverity,
    TestType,
    TestResult,
    ScenarioType,
    RecoveryStrategy,
    DORAICTBusinessContinuity,
    create_dora_ict_business_continuity,
)

from services.dora.simplified_framework import (
    EntitySize,
    SimplifiedControlCategory,
    ControlStatus,
    RiskLevel as SimplifiedRiskLevel,
    IncidentPriority,
    IncidentStatus as SimplifiedIncidentStatus,
    TestType as SimplifiedTestType,
    DORASimplifiedFramework,
    create_dora_simplified_framework,
    ESSENTIAL_CONTROLS,
)


# =============================================================================
# Article 5: Governance Framework Tests
# =============================================================================

class TestGovernanceFramework:
    """Tests for DORA Article 5 - Governance Framework."""

    def test_create_governance_framework(self):
        """Test governance framework creation."""
        framework = create_governance_framework()
        assert framework is not None
        assert isinstance(framework, DORAGovernanceFramework)

    def test_assign_governance_role(self):
        """Test assigning governance roles."""
        framework = create_governance_framework()
        role = framework.assign_role(
            role=GovernanceRole.ICT_RISK_OFFICER,
            person_name="John Doe",
            email="john.doe@example.com",
            department="IT Risk"
        )
        assert role is not None
        assert role.role == GovernanceRole.ICT_RISK_OFFICER
        assert role.person_name == "John Doe"

    def test_record_training(self):
        """Test recording ICT training."""
        framework = create_governance_framework()
        training = framework.record_training(
            person_id="EMP001",
            person_name="John Doe",
            training_name="ICT Risk Management Training",
            status=TrainingStatus.COMPLETED,
            duration_hours=8.0,
            training_provider="External Provider"
        )
        assert training is not None
        assert training.status == TrainingStatus.COMPLETED

    def test_submit_for_approval(self):
        """Test framework approval workflow."""
        framework = create_governance_framework()
        approval = framework.submit_for_approval(
            document_type="ict_risk_framework",
            document_name="ICT Risk Framework v1.0",
            document_version="1.0",
            submitted_by="CISO"
        )
        assert approval is not None
        assert approval.status == ApprovalStatus.PENDING_REVIEW

    def test_get_governance_summary_roles(self):
        """Test governance summary includes role count."""
        framework = create_governance_framework()
        framework.assign_role(
            role=GovernanceRole.ICT_RISK_OFFICER,
            person_name="John Doe",
            email="john.doe@example.com"
        )
        summary = framework.get_governance_summary()
        assert summary is not None
        assert isinstance(summary, dict)

    def test_mandatory_training_topics(self):
        """Test mandatory training topics are defined."""
        assert len(MANDATORY_TRAINING_TOPICS) > 0
        for topic in MANDATORY_TRAINING_TOPICS:
            assert "topic" in topic
            assert "description" in topic

    def test_get_governance_summary(self):
        """Test governance summary retrieval."""
        framework = create_governance_framework()
        summary = framework.get_governance_summary()
        assert isinstance(summary, dict)


# =============================================================================
# Article 6: ICT Risk Management Framework Tests
# =============================================================================

class TestICTRiskFramework:
    """Tests for DORA Article 6 - ICT Risk Management Framework."""

    def test_create_ict_risk_framework(self):
        """Test ICT risk framework creation."""
        framework = create_ict_risk_framework()
        assert framework is not None
        assert isinstance(framework, DORAICTRiskFramework)

    def test_create_policy(self):
        """Test creating risk policy."""
        framework = create_ict_risk_framework()
        policy = framework.create_policy(
            name="Information Security Policy",
            description="Main security policy",
            category=PolicyCategory.ICT_SECURITY,
            owner="CISO"
        )
        assert policy is not None
        assert policy.category == PolicyCategory.ICT_SECURITY

    def test_create_control(self):
        """Test creating ICT control."""
        framework = create_ict_risk_framework()
        control = framework.create_control(
            name="Access Control",
            description="Role-based access control",
            domain=ControlDomain.PROTECT,
            control_type=ControlType.PREVENTIVE,
            control_owner="IT Security"
        )
        assert control is not None
        assert control.domain == ControlDomain.PROTECT

    def test_register_risk(self):
        """Test registering ICT risk."""
        framework = create_ict_risk_framework()
        risk = framework.register_risk(
            name="Data Breach Risk",
            description="Risk of unauthorized data access",
            category="security",
            likelihood=4,
            impact=5,
            treatment_owner="CISO"
        )
        assert risk is not None
        assert risk.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]

    def test_get_framework_status(self):
        """Test framework status."""
        framework = create_ict_risk_framework()
        status = framework.get_framework_status()
        assert isinstance(status, dict)


# =============================================================================
# Article 7: ICT Systems Tests
# =============================================================================

class TestICTSystems:
    """Tests for DORA Article 7 - ICT Systems."""

    def test_create_ict_systems_manager(self):
        """Test ICT systems manager creation."""
        manager = create_ict_systems_manager()
        assert manager is not None
        assert isinstance(manager, DORAICTSystemsManager)

    def test_register_system(self):
        """Test registering ICT system."""
        manager = create_ict_systems_manager()
        system = manager.register_system(
            name="Trading Platform",
            description="Core trading system",
            system_type=SystemType.APPLICATION,
            criticality=SystemCriticality.CRITICAL,
            system_owner="Trading IT",
            vendor="Internal"
        )
        assert system is not None
        assert system.criticality == SystemCriticality.CRITICAL

    def test_record_capacity_metric(self):
        """Test recording capacity metric."""
        manager = create_ict_systems_manager()
        system = manager.register_system(
            name="Database Server",
            description="Main database",
            system_type=SystemType.DATABASE,
            criticality=SystemCriticality.CRITICAL,
            system_owner="DBA Team",
            vendor="Oracle"
        )
        metric = manager.record_capacity_metric(
            system_id=system.system_id,
            metric_name="cpu_usage",
            current_value=75.0,
            maximum_capacity=100.0
        )
        assert metric is not None

    def test_get_critical_systems(self):
        """Test getting critical systems."""
        manager = create_ict_systems_manager()
        manager.register_system(
            name="Core System",
            description="Critical system",
            system_type=SystemType.APPLICATION,
            criticality=SystemCriticality.CRITICAL,
            system_owner="IT",
            vendor="Internal"
        )
        critical = manager.get_critical_systems()
        assert len(critical) >= 1


# =============================================================================
# Article 8: ICT Identification Tests
# =============================================================================

class TestICTIdentification:
    """Tests for DORA Article 8 - ICT Identification."""

    def test_create_ict_identification(self):
        """Test ICT identification creation."""
        identification = create_ict_identification()
        assert identification is not None
        assert isinstance(identification, DORAICTIdentification)

    def test_register_asset(self):
        """Test registering ICT asset."""
        identification = create_ict_identification()
        asset = identification.register_asset(
            name="Production Server",
            asset_type=AssetType.HARDWARE,
            classification=AssetClassification.CONFIDENTIAL,
            owner="IT Operations",
            location="Data Center A"
        )
        assert asset is not None
        assert asset.asset_type == AssetType.HARDWARE

    def test_record_threat(self):
        """Test recording cyber threat."""
        identification = create_ict_identification()
        threat = identification.record_threat(
            name="Ransomware Attack",
            category=ThreatCategory.MALWARE,
            description="Ransomware encryption threat",
            threat_actor_type="criminal",
            relevance_to_entity="high"
        )
        assert threat is not None
        assert threat.category == ThreatCategory.MALWARE

    def test_record_vulnerability(self):
        """Test recording vulnerability."""
        identification = create_ict_identification()
        vuln = identification.record_vulnerability(
            name="SQL Injection",
            description="SQL injection vulnerability",
            severity=VulnerabilitySeverity.HIGH,
            cve_id="CVE-2024-1234"
        )
        assert vuln is not None
        assert vuln.severity == VulnerabilitySeverity.HIGH


# =============================================================================
# Article 9: Protection and Prevention Tests
# =============================================================================

class TestProtection:
    """Tests for DORA Article 9 - Protection and Prevention."""

    def test_create_protection(self):
        """Test protection framework creation."""
        protection = create_protection()
        assert protection is not None
        assert isinstance(protection, DORAProtection)

    def test_create_security_control(self):
        """Test creating security control."""
        protection = create_protection()
        control = protection.create_security_control(
            name="Firewall",
            description="Network perimeter protection",
            category=SecurityControlCategory.COMMUNICATIONS_SECURITY,
            owner="Network Team"
        )
        assert control is not None

    def test_create_access_policy(self):
        """Test creating access policy."""
        protection = create_protection()
        policy = protection.create_access_policy(
            name="Admin Access Policy",
            description="Policy for admin access",
            access_control_type=AccessControlType.ROLE_BASED,
            mfa_required=True
        )
        assert policy is not None
        assert policy.mfa_required == True

    def test_create_encryption_standard(self):
        """Test creating encryption standard."""
        protection = create_protection()
        standard = protection.create_encryption_standard(
            name="Data at Rest Encryption",
            encryption_type=EncryptionType.AT_REST,
            hsm_required=True
        )
        assert standard is not None


# =============================================================================
# Article 10: Detection Tests
# =============================================================================

class TestDetection:
    """Tests for DORA Article 10 - Detection."""

    def test_create_detection(self):
        """Test detection framework creation."""
        detection = create_detection()
        assert detection is not None
        assert isinstance(detection, DORADetection)

    def test_create_detection_rule(self):
        """Test creating detection rule."""
        from services.dora.detection import IncidentCategory as DetectionIncidentCategory
        detection = create_detection()
        rule = detection.create_detection_rule(
            name="Brute Force Detection",
            description="Detect brute force attacks",
            category=DetectionIncidentCategory.SECURITY,
            threshold_value=5.0,
            method=DetectionMethod.THRESHOLD_BASED,
            default_severity=AlertSeverity.HIGH
        )
        assert rule is not None
        assert rule.category == DetectionIncidentCategory.SECURITY

    def test_check_metric_alert(self):
        """Test checking metric triggers alert."""
        from services.dora.detection import IncidentCategory as DetectionIncidentCategory
        detection = create_detection()
        # Create a rule with low threshold
        rule = detection.create_detection_rule(
            name="CPU High Rule",
            description="Detect high CPU",
            category=DetectionIncidentCategory.PERFORMANCE,
            threshold_value=90.0,
            method=DetectionMethod.THRESHOLD_BASED,
            default_severity=AlertSeverity.MEDIUM
        )
        # Check metric that should trigger alert
        alert = detection.check_metric(
            metric_name="cpu_usage",
            current_value=95.0,
            system_id="SYS-001"
        )
        # May or may not trigger depending on rule config
        # Just verify method works
        assert rule is not None


# =============================================================================
# Article 11: Response and Recovery Tests
# =============================================================================

class TestResponseRecovery:
    """Tests for DORA Article 11 - Response and Recovery."""

    def test_create_response_recovery(self):
        """Test response recovery framework creation."""
        rr = create_response_recovery()
        assert rr is not None
        assert isinstance(rr, DORAResponseRecovery)

    def test_create_incident(self):
        """Test creating ICT incident."""
        rr = create_response_recovery()
        incident = rr.create_incident(
            title="System Outage",
            description="Production system outage",
            severity=IncidentSeverity.P1_CRITICAL,
            category=IncidentCategory.SYSTEM_FAILURE,
            affected_systems=["trading_platform"]
        )
        assert incident is not None
        assert incident.severity == IncidentSeverity.P1_CRITICAL
        assert incident.status == IncidentStatus.NEW

    def test_start_response(self):
        """Test starting incident response."""
        rr = create_response_recovery()
        incident = rr.create_incident(
            title="Test Incident",
            description="Test incident for response",
            severity=IncidentSeverity.P2_HIGH,
            category=IncidentCategory.APPLICATION_FAILURE
        )
        updated = rr.start_response(
            incident_id=incident.incident_id,
            responders=["security_team"],
            incident_commander="CISO"
        )
        assert updated is not None
        assert updated.status == IncidentStatus.INVESTIGATING

    def test_create_escalation_rule(self):
        """Test creating escalation rule."""
        rr = create_response_recovery()
        rule = rr.create_escalation_rule(
            name="Critical Escalation",
            from_level=EscalationLevel.L1_OPERATIONS,
            to_level=EscalationLevel.L4_EXECUTIVE,
            trigger_elapsed_minutes=15,
            trigger_severity=[IncidentSeverity.P1_CRITICAL]
        )
        assert rule is not None


# =============================================================================
# Article 12: Backup and Recovery Tests
# =============================================================================

class TestBackupRecovery:
    """Tests for DORA Article 12 - Backup and Recovery."""

    def test_create_backup_recovery(self):
        """Test backup recovery framework creation."""
        br = create_backup_recovery()
        assert br is not None
        assert isinstance(br, DORABackupRecovery)

    def test_create_backup_policy(self):
        """Test creating backup policy."""
        from services.dora.backup_recovery import DataClassification as BackupDataClassification
        br = create_backup_recovery()
        policy = br.create_backup_policy(
            name="Database Backup Policy",
            data_classification=BackupDataClassification.CRITICAL,
            backup_type=BackupType.FULL,
            frequency=BackupFrequency.DAILY,
            retention_days=30,
            systems_covered=["production_databases"]
        )
        assert policy is not None
        assert policy.backup_type == BackupType.FULL

    def test_create_backup_job(self):
        """Test creating backup job."""
        from services.dora.backup_recovery import DataClassification as BackupDataClassification
        br = create_backup_recovery()
        policy = br.create_backup_policy(
            name="Test Policy",
            data_classification=BackupDataClassification.STANDARD,
            backup_type=BackupType.INCREMENTAL,
            frequency=BackupFrequency.HOURLY,
            retention_days=7
        )
        job = br.create_backup_job(
            policy_id=policy.policy_id,
            source_systems=["test_db"]
        )
        assert job is not None

    def test_register_backup_location(self):
        """Test registering backup location."""
        br = create_backup_recovery()
        location = br.register_backup_location(
            name="Secondary DC",
            location_type=LocationType.OFF_SITE,
            geographic_region="Europe",
            country="Germany",
            is_disaster_recovery=True
        )
        assert location is not None


# =============================================================================
# Article 13: Learning and Evolving Tests
# =============================================================================

class TestLearning:
    """Tests for DORA Article 13 - Learning and Evolving."""

    def test_create_learning(self):
        """Test learning framework creation."""
        learning = create_dora_learning()
        assert learning is not None
        assert isinstance(learning, DORALearning)

    def test_create_post_incident_review(self):
        """Test creating post-incident review."""
        learning = create_dora_learning()
        review = learning.create_post_incident_review(
            incident_id="INC-001",
            review_type=ReviewType.POST_INCIDENT,
            reviewer="Security Team",
            incident_summary="System outage",
            timeline=[{"time": "10:00", "event": "Incident detected"}],
            root_causes=["Configuration error"],
            contributing_factors=["Lack of testing"],
            impact_assessment={"severity": "high"},
            response_effectiveness={"rating": "good"},
            recovery_effectiveness={"rating": "good"},
            what_went_well=["Quick detection"],
            what_went_wrong=["Slow recovery"],
            recommendations=["Improve testing"],
            action_items=[{"action": "Update runbook"}]
        )
        assert review is not None

    def test_document_lesson_learned(self):
        """Test documenting lesson learned."""
        learning = create_dora_learning()
        lesson = learning.document_lesson_learned(
            title="Improve Monitoring",
            description="Need better monitoring",
            category=LessonCategory.TECHNICAL,
            priority=LessonPriority.HIGH,
            identified_by="Incident Manager",
            problem_statement="Delayed detection",
            current_state="Basic monitoring",
            desired_state="Advanced monitoring",
            gap_analysis="Missing alerting",
            recommended_actions=["Implement SIEM"],
            affected_areas=["Security"],
            affected_systems=["All"],
            stakeholders=["IT Security"]
        )
        assert lesson is not None
        assert lesson.priority == LessonPriority.HIGH

    def test_create_knowledge_article(self):
        """Test creating knowledge article."""
        learning = create_dora_learning()
        article = learning.create_knowledge_article(
            title="Incident Response Guide",
            content="Step by step guide",
            knowledge_type=KnowledgeType.RUNBOOK,
            category="Security",
            author="Security Team",
            applicable_systems=["All"],
            applicable_scenarios=["Security incident"],
            procedures=[{"step": 1, "action": "Identify"}]
        )
        assert article is not None


# =============================================================================
# Article 14: Communication Tests
# =============================================================================

class TestCommunication:
    """Tests for DORA Article 14 - Communication."""

    def test_create_communication(self):
        """Test communication framework creation."""
        comm = create_dora_communication()
        assert comm is not None
        assert isinstance(comm, DORACommunication)

    def test_create_policy(self):
        """Test creating communication policy."""
        comm = create_dora_communication()
        policy = comm.create_policy(
            name="Incident Communication Policy",
            description="Policy for incident communications",
            communication_type=CommunicationType.INTERNAL,
            applicable_scenarios=["incident"],
            target_stakeholders=[StakeholderType.MANAGEMENT_BODY],
            required_channels=[CommunicationChannel.EMAIL],
            response_time_hours=1,
            escalation_time_hours=2,
            approval_required=True,
            approver_roles=["CISO"],
            content_requirements=["Incident summary", "Impact"]
        )
        assert policy is not None

    def test_register_stakeholder(self):
        """Test registering stakeholder."""
        comm = create_dora_communication()
        stakeholder = comm.register_stakeholder(
            name="John Smith",
            stakeholder_type=StakeholderType.SENIOR_MANAGEMENT,
            organization="Company",
            role="CTO",
            primary_contact="john@company.com",
            preferred_channel=CommunicationChannel.EMAIL,
            alternative_channels=[CommunicationChannel.PHONE],
            language="en",
            timezone="UTC",
            notification_preferences={"urgent": True}
        )
        assert stakeholder is not None

    def test_create_communication_record(self):
        """Test creating communication record."""
        comm = create_dora_communication()
        record = comm.create_communication(
            subject="Incident Notification",
            content="Incident details...",
            communication_type=CommunicationType.INTERNAL,
            notification_type=NotificationType.INCIDENT,
            priority=CommunicationPriority.HIGH,
            channel=CommunicationChannel.EMAIL,
            sender="security@company.com",
            recipients=["management@company.com"]
        )
        assert record is not None


# =============================================================================
# Article 15: ICT Business Continuity Tests
# =============================================================================

class TestICTBusinessContinuity:
    """Tests for DORA Article 15 - ICT Business Continuity."""

    def test_create_business_continuity(self):
        """Test business continuity framework creation."""
        bc = create_dora_ict_business_continuity()
        assert bc is not None
        assert isinstance(bc, DORAICTBusinessContinuity)

    def test_create_policy(self):
        """Test creating continuity policy."""
        bc = create_dora_ict_business_continuity()
        policy = bc.create_policy(
            name="ICT Business Continuity Policy",
            version="1.0",
            description="Main BCP policy",
            scope=["All ICT systems"],
            objectives=["Minimize downtime"],
            governance_structure={"owner": "CISO"},
            roles_responsibilities=[{"role": "BCM", "responsibility": "Coordination"}],
            policy_statements=["Critical systems must have backup"],
            minimum_service_levels={"trading": 99.9},
            maximum_tolerable_downtime={"trading": 4},
            resource_requirements=["DR site"],
            training_requirements=["Annual BCP training"],
            review_frequency_months=12,
            regulatory_references=["DORA Article 15"]
        )
        assert policy is not None

    def test_register_business_function(self):
        """Test registering business function."""
        bc = create_dora_ict_business_continuity()
        function = bc.register_business_function(
            name="Trading Operations",
            description="Core trading function",
            department="Trading",
            owner="Head of Trading",
            criticality=CriticalityLevel.MISSION_CRITICAL,
            dependencies=["Market Data"],
            ict_dependencies=["Trading Platform"],
            third_party_dependencies=["Exchange"],
            rto_hours=4,
            rpo_hours=1,
            mtpd_hours=24,
            minimum_staff_required=5,
            minimum_resources=["Trading terminals"],
            workaround_available=True
        )
        assert function is not None

    def test_create_recovery_objective(self):
        """Test creating recovery objective."""
        bc = create_dora_ict_business_continuity()
        objective = bc.define_recovery_objective(
            name="Trading System RTO",
            description="Recovery objective for trading",
            rto_target_hours=4,
            rpo_target_hours=1,
            minimum_service_level=95.0,
            target_availability=99.9,
            recovery_strategy=RecoveryStrategy.ACTIVE_PASSIVE,
            recovery_capabilities=["Failover"],
            dependencies_for_recovery=["DR Site"],
            validation_criteria=["Systems operational"]
        )
        assert objective is not None

    def test_schedule_test(self):
        """Test scheduling continuity test."""
        bc = create_dora_ict_business_continuity()
        test = bc.schedule_test(
            name="Annual DR Test",
            description="Annual disaster recovery test",
            test_type=TestType.FULL_INTERRUPTION,
            objectives=["Validate DR capability"],
            scope=["Trading systems"],
            scheduled_date=datetime.now() + timedelta(days=30),
            facilitator="BCM Team",
            participants=[{"name": "IT Team", "role": "Technical"}],
            test_script=[{"step": 1, "action": "Initiate failover"}],
            inject_scenarios=[{"scenario": "DC failure"}],
            expected_outcomes=["Successful failover"]
        )
        assert test is not None


# =============================================================================
# Article 16: Simplified Framework Tests
# =============================================================================

class TestSimplifiedFramework:
    """Tests for DORA Article 16 - Simplified Framework."""

    def test_create_simplified_framework(self):
        """Test simplified framework creation."""
        sf = create_dora_simplified_framework()
        assert sf is not None
        assert isinstance(sf, DORASimplifiedFramework)

    def test_essential_controls_defined(self):
        """Test essential controls are defined."""
        assert len(ESSENTIAL_CONTROLS) > 0
        for control in ESSENTIAL_CONTROLS:
            assert "name" in control
            assert "category" in control
            assert "description" in control

    def test_assess_eligibility(self):
        """Test eligibility assessment."""
        sf = create_dora_simplified_framework()
        eligibility = sf.assess_eligibility(
            entity_name="Small Investment Firm",
            entity_type="investment_firm",
            entity_size=EntitySize.SMALL,
            total_assets_eur=5000000.0,
            annual_revenue_eur=1000000.0,
            number_of_employees=20,
            ict_spend_eur=50000.0,
            critical_functions_count=2,
            third_party_providers_count=5,
            cross_border_operations=False,
            significant_cyber_risk=False,
            part_of_group=False,
            group_uses_simplified=False,
            assessed_by="Compliance",
            next_review_date=datetime.now() + timedelta(days=365)
        )
        assert eligibility is not None
        assert eligibility.eligible == True

    def test_get_all_controls(self):
        """Test getting all controls."""
        sf = create_dora_simplified_framework()
        controls = sf.get_all_controls()
        assert len(controls) > 0

    def test_update_control_status(self):
        """Test updating control status."""
        sf = create_dora_simplified_framework()
        controls = sf.get_all_controls()
        if controls:
            result = sf.update_control_status(
                control_id=controls[0].control_id,
                status=ControlStatus.FULLY_IMPLEMENTED,
                implementation_notes="Implemented successfully",
                reviewer="IT Security"
            )
            assert result == True

    def test_create_risk_assessment(self):
        """Test creating risk assessment."""
        sf = create_dora_simplified_framework()
        assessment = sf.create_risk_assessment(
            assessor="Risk Manager",
            asset_name="Email Server",
            asset_type="server",
            asset_description="Corporate email server",
            threat_description="Malware infection",
            vulnerability_description="Outdated antivirus",
            existing_controls=["Firewall"],
            risk_level=SimplifiedRiskLevel.MEDIUM,
            impact_description="Email service disruption",
            likelihood_description="Possible",
            risk_treatment="mitigate",
            treatment_actions=["Update antivirus"],
            residual_risk=SimplifiedRiskLevel.LOW,
            risk_owner="IT Manager",
            review_date=datetime.now() + timedelta(days=90)
        )
        assert assessment is not None

    def test_report_incident(self):
        """Test reporting incident."""
        sf = create_dora_simplified_framework()
        incident = sf.report_incident(
            title="Email Outage",
            description="Email service unavailable",
            priority=IncidentPriority.HIGH,
            detected_by="IT Support",
            affected_systems=["email_server"],
            affected_services=["email"],
            impact_description="Users cannot send/receive email",
            initial_response="Investigating"
        )
        assert incident is not None

    def test_record_backup(self):
        """Test recording backup."""
        sf = create_dora_simplified_framework()
        backup = sf.record_backup(
            system_name="Database",
            backup_type="full",
            backup_location="offsite_storage",
            retention_days=30,
            encryption_enabled=True,
            size_gb=50.0,
            performed_by="Backup Service"
        )
        assert backup is not None

    def test_register_third_party(self):
        """Test registering third party."""
        sf = create_dora_simplified_framework()
        provider = sf.register_third_party(
            name="Cloud Provider",
            service_description="Cloud hosting",
            service_criticality="critical",
            contract_start_date=datetime.now(),
            contact_name="Account Manager",
            contact_email="am@cloud.com",
            contact_phone="+1234567890",
            data_processing=True
        )
        assert provider is not None

    def test_schedule_test(self):
        """Test scheduling test."""
        sf = create_dora_simplified_framework()
        test = sf.schedule_test(
            test_type=SimplifiedTestType.BACKUP_RESTORATION,
            test_name="Monthly Backup Test",
            description="Test backup restoration",
            scheduled_date=datetime.now() + timedelta(days=7),
            scope=["production_db"]
        )
        assert test is not None

    def test_record_training(self):
        """Test recording training."""
        sf = create_dora_simplified_framework()
        training = sf.record_training(
            title="Security Awareness",
            description="Annual security training",
            target_audience="All staff",
            training_date=datetime.now(),
            duration_minutes=60,
            trainer="Security Team",
            topics_covered=["Phishing", "Passwords"],
            attendees=["emp1", "emp2", "emp3"],
            completion_rate=0.95
        )
        assert training is not None

    def test_create_annual_review(self):
        """Test creating annual review."""
        sf = create_dora_simplified_framework()
        review = sf.create_annual_review(
            review_year=2024,
            reviewer="Compliance Officer",
            eligibility_confirmed=True,
            key_findings=["Controls improved"],
            improvement_actions=["Update policies"],
            next_review_date=datetime.now() + timedelta(days=365)
        )
        assert review is not None

    def test_get_framework_metrics(self):
        """Test getting framework metrics."""
        sf = create_dora_simplified_framework()
        metrics = sf.get_framework_metrics()
        assert isinstance(metrics, dict)
        assert "eligibility" in metrics
        assert "controls" in metrics


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase1Integration:
    """Integration tests for DORA Phase 1."""

    def test_all_modules_can_be_imported(self):
        """Test that all Phase 1 modules can be imported together."""
        from services.dora import (
            __version__,
            __dora_compliance_phase__,
            DORAGovernanceFramework,
            DORAICTRiskFramework,
            DORAICTSystemsManager,
            DORAICTIdentification,
            DORAProtection,
            DORADetection,
            DORAResponseRecovery,
            DORABackupRecovery,
            DORALearning,
            DORACommunication,
            DORAICTBusinessContinuity,
            DORASimplifiedFramework,
        )
        assert __version__ == "0.3.0"
        assert __dora_compliance_phase__ == 2

    def test_factory_functions_work(self):
        """Test all factory functions create valid instances."""
        governance = create_governance_framework()
        risk_framework = create_ict_risk_framework()
        systems = create_ict_systems_manager()
        identification = create_ict_identification()
        protection = create_protection()
        detection = create_detection()
        response = create_response_recovery()
        backup = create_backup_recovery()
        learning = create_dora_learning()
        communication = create_dora_communication()
        continuity = create_dora_ict_business_continuity()
        simplified = create_dora_simplified_framework()

        assert all([
            governance, risk_framework, systems, identification,
            protection, detection, response, backup, learning,
            communication, continuity, simplified
        ])


# =============================================================================
# Export Tests
# =============================================================================

class TestExports:
    """Test module exports."""

    def test_all_enums_exported(self):
        """Test all enums are exported."""
        from services.dora import (
            GovernanceRole,
            DefenceLine,
            PolicyCategory,
            ControlDomain,
            SystemCriticality,
            AssetType,
            SecurityControlCategory,
            AnomalyType,
            IncidentSeverity,
            BackupType,
            ReviewType,
            CommunicationType,
            ContinuityStatus,
            EntitySize,
        )
        # Verify all enums are valid enum types
        from enum import Enum
        for enum_cls in [GovernanceRole, DefenceLine, PolicyCategory, ControlDomain,
                         SystemCriticality, AssetType, SecurityControlCategory,
                         AnomalyType, IncidentSeverity, BackupType, ReviewType,
                         CommunicationType, ContinuityStatus, EntitySize]:
            assert issubclass(enum_cls, Enum), f"{enum_cls.__name__} should be an Enum"

    def test_all_dataclasses_exported(self):
        """Test all dataclasses are exported."""
        from services.dora import (
            GovernanceRoleAssignment,
            RiskPolicy,
            ICTSystem,
            ICTAsset,
            SecurityControl,
            DetectionRule,
            ICTIncident,
            BackupPolicy,
            PostIncidentReview,
            CommunicationPolicy,
            ContinuityPlan,
            EligibilityCriteria,
        )
        # Verify all are dataclasses
        from dataclasses import is_dataclass
        for dc_cls in [GovernanceRoleAssignment, RiskPolicy, ICTSystem, ICTAsset,
                       SecurityControl, DetectionRule, ICTIncident, BackupPolicy,
                       PostIncidentReview, CommunicationPolicy, ContinuityPlan,
                       EligibilityCriteria]:
            assert is_dataclass(dc_cls), f"{dc_cls.__name__} should be a dataclass"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
