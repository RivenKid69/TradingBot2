# -*- coding: utf-8 -*-
"""
Comprehensive Tests for DORA Phase 2 - ICT Incident Management & Reporting.

Tests for Articles 17-23 of DORA (EU) 2022/2554.

Coverage includes:
- Article 17: Incident Management Process
- Article 18: Incident Classification
- Article 19: Incident Reporting
- Article 19(4): Cyber Threat Notification
- Article 20: Reporting Templates
- Article 22: Supervisory Feedback
- Article 23: Third-Party Incidents

References:
- DORA (EU) 2022/2554
- CDR 2024/1772: RTS on incident classification
- CDR 2025/301: RTS on incident reporting
- CIR 2025/302: ITS on reporting templates
"""


import pytest
pytest.skip(
    "Legacy DORA test - uses deprecated imports from services.dora.* "
    "These modules have been migrated to services.dora_integration.*. "
    "See tests/dora/ and tests/dora_integration/ for current tests.",
    allow_module_level=True
)


import pytest
from datetime import datetime, timedelta, timezone

# =============================================================================
# Import all Phase 2 modules
# =============================================================================

from services.dora.incident_management import (
    ICTEventType,
    IncidentPhase,
    IncidentPriority,
    IncidentStatus,
    EscalationLevel,
    EarlyWarningType,
    ICTEvent,
    DORAIncident,
    EarlyWarningIndicator,
    IncidentAction,
    EscalationRule,
    IncidentManagementConfig,
    DORAIncidentManagement,
    create_incident_management,
)

from services.dora.incident_classification import (
    IncidentClassificationType,
    ClientType,
    DataType,
    CriticalServiceType,
    MajorIncidentTrigger,
    ReputationalImpactLevel,
    ClassificationThresholds,
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
    IncidentClassificationConfig,
    DORAIncidentClassification,
    create_incident_classification,
)

from services.dora.incident_reporting import (
    ReportType,
    ReportStatus,
    IncidentTypeCode,
    RootCauseCategory,
    CompetentAuthorityType,
    CompetentAuthority,
    InitialNotificationReport,
    IntermediateReport,
    FinalReport,
    ReportSubmission,
    IncidentReportingConfig,
    DORAIncidentReporter,
    create_incident_reporter,
)

from services.dora.cyber_threat_notification import (
    ThreatCategory,
    ThreatSeverity,
    ThreatStatus,
    ThreatActorType,
    ThreatSignificance,
    NotificationStatus,
    ThreatIndicator,
    CyberThreat,
    ThreatSignificanceAssessment,
    ThreatNotification,
    CyberThreatNotificationConfig,
    CyberThreatNotificationService,
    create_cyber_threat_notification_service,
)

from services.dora.reporting_templates import (
    IncidentTypeCode as TemplateIncidentTypeCode,
    DataTypeCode,
    ClientTypeCode,
    ServiceTypeCode,
    ResponseEffectivenessCode,
    TimelineEvent,
    ITSInitialNotificationTemplate,
    ITSIntermediateReportTemplate,
    ITSFinalReportTemplate,
    DORAReportingTemplates,
    create_reporting_templates,
)

from services.dora.supervisory_feedback import (
    FeedbackType,
    FeedbackPriority,
    FeedbackStatus,
    CorrectiveActionType,
    ResponseType,
    CompetentAuthority as FeedbackCompetentAuthority,
    SupervisoryFeedback,
    CorrectiveAction,
    FeedbackResponse,
    FeedbackAuditEntry,
    AnonymisedInsight,
    DORASupervisioryFeedback,
    create_supervisory_feedback,
)

from services.dora.third_party_incidents import (
    ThirdPartyProviderType,
    ThirdPartyCriticality,
    ThirdPartyIncidentType,
    IncidentSeverity as ThirdPartyIncidentSeverity,
    IncidentStatus as ThirdPartyIncidentStatus,
    ContractualSLAStatus,
    EscalationLevel as ThirdPartyEscalationLevel,
    CommunicationChannel as ThirdPartyCommunicationChannel,
    ThirdPartyProvider,
    AffectedService,
    SLAAssessment,
    CommunicationRecord,
    EscalationRecord,
    MitigationAction,
    ThirdPartyIncident,
    PostIncidentReview,
    DORAThirdPartyIncidents,
    create_third_party_incidents,
)


# =============================================================================
# Article 17: Incident Management Tests
# =============================================================================

class TestIncidentManagementCreation:
    """Tests for DORAIncidentManagement creation and initialization."""

    def test_create_incident_management_default(self):
        """Test incident management creation with defaults."""
        manager = create_incident_management()
        assert manager is not None
        assert isinstance(manager, DORAIncidentManagement)

    def test_create_incident_management_with_config(self):
        """Test incident management creation with custom config."""
        config = IncidentManagementConfig()
        manager = create_incident_management(config)
        assert manager is not None
        assert manager.config is not None

    def test_incident_management_has_required_methods(self):
        """Test incident management has required methods."""
        manager = create_incident_management()
        assert hasattr(manager, 'create_incident')
        assert hasattr(manager, 'classify_incident')
        assert hasattr(manager, 'escalate_incident')


class TestICTEventTypes:
    """Tests for ICT event type enumeration."""

    def test_system_failure_type(self):
        """Test system failure event type."""
        assert ICTEventType.SYSTEM_FAILURE.value == "system_failure"

    def test_security_breach_type(self):
        """Test security breach event type."""
        assert ICTEventType.SECURITY_BREACH.value == "security_breach"

    def test_data_breach_type(self):
        """Test data breach event type."""
        assert ICTEventType.DATA_BREACH.value == "data_breach"

    def test_network_disruption_type(self):
        """Test network disruption event type."""
        assert ICTEventType.NETWORK_DISRUPTION.value == "network_disruption"

    def test_malware_detection_type(self):
        """Test malware detection event type."""
        assert ICTEventType.MALWARE_DETECTION.value == "malware_detection"

    def test_event_types_enumerable(self):
        """Test event types are enumerable."""
        event_types = list(ICTEventType)
        assert len(event_types) >= 5


class TestIncidentPhases:
    """Tests for incident management phases."""

    def test_detection_phase(self):
        """Test detection phase exists."""
        assert IncidentPhase.DETECTION.value == "detection"

    def test_recording_phase(self):
        """Test recording phase exists."""
        assert IncidentPhase.RECORDING.value == "recording"

    def test_classification_phase(self):
        """Test classification phase exists."""
        assert IncidentPhase.CLASSIFICATION.value == "classification"

    def test_escalation_phase(self):
        """Test escalation phase exists."""
        assert IncidentPhase.ESCALATION.value == "escalation"

    def test_notification_phase(self):
        """Test notification phase exists."""
        assert IncidentPhase.NOTIFICATION.value == "notification"

    def test_investigation_phase(self):
        """Test investigation phase exists."""
        assert IncidentPhase.INVESTIGATION.value == "investigation"

    def test_resolution_phase(self):
        """Test resolution phase exists."""
        assert IncidentPhase.RESOLUTION.value == "resolution"

    def test_closure_phase(self):
        """Test closure phase exists."""
        assert IncidentPhase.CLOSURE.value == "closure"


class TestIncidentPriority:
    """Tests for incident priority levels."""

    def test_p1_critical(self):
        """Test P1 critical priority."""
        assert IncidentPriority.P1_CRITICAL.value == "P1"

    def test_p2_high(self):
        """Test P2 high priority."""
        assert IncidentPriority.P2_HIGH.value == "P2"

    def test_p3_medium(self):
        """Test P3 medium priority."""
        assert IncidentPriority.P3_MEDIUM.value == "P3"

    def test_p4_low(self):
        """Test P4 low priority."""
        assert IncidentPriority.P4_LOW.value == "P4"


class TestIncidentStatus:
    """Tests for incident status tracking."""

    def test_new_status(self):
        """Test new status."""
        assert IncidentStatus.NEW.value == "new"

    def test_detected_status(self):
        """Test detected status."""
        assert IncidentStatus.DETECTED.value == "detected"

    def test_classified_status(self):
        """Test classified status."""
        assert IncidentStatus.CLASSIFIED.value == "classified"

    def test_resolved_status(self):
        """Test resolved status."""
        assert IncidentStatus.RESOLVED.value == "resolved"

    def test_closed_status(self):
        """Test closed status."""
        assert IncidentStatus.CLOSED.value == "closed"


class TestEscalationLevel:
    """Tests for escalation levels per Article 17(3)(d)."""

    def test_l1_operational(self):
        """Test L1 operational level."""
        assert EscalationLevel.L1_OPERATIONAL.value == "L1"

    def test_l2_technical(self):
        """Test L2 technical level."""
        assert EscalationLevel.L2_TECHNICAL.value == "L2"

    def test_l3_management(self):
        """Test L3 management level."""
        assert EscalationLevel.L3_MANAGEMENT.value == "L3"

    def test_l4_executive(self):
        """Test L4 executive level."""
        assert EscalationLevel.L4_EXECUTIVE.value == "L4"


class TestICTEvent:
    """Tests for ICT event data structure."""

    def test_create_ict_event_default(self):
        """Test creating an ICT event with defaults."""
        event = ICTEvent()
        assert event.event_id is not None
        assert event.event_id.startswith("EVT-")

    def test_create_ict_event_with_type(self):
        """Test creating an ICT event with type."""
        event = ICTEvent(event_type=ICTEventType.SYSTEM_FAILURE)
        assert event.event_type == ICTEventType.SYSTEM_FAILURE

    def test_event_auto_generates_id(self):
        """Test event auto-generates unique ID."""
        event1 = ICTEvent()
        event2 = ICTEvent()
        assert event1.event_id != event2.event_id

    def test_event_auto_generates_timestamp(self):
        """Test event auto-generates timestamp."""
        event = ICTEvent()
        assert event.detected_at is not None
        assert len(event.detected_at) > 0


class TestDORAIncident:
    """Tests for DORA incident data structure."""

    def test_create_incident_default(self):
        """Test creating a DORA incident with defaults."""
        incident = DORAIncident()
        assert incident.incident_id is not None
        assert incident.incident_id.startswith("INC-")

    def test_incident_default_status(self):
        """Test incident default status is NEW."""
        incident = DORAIncident()
        assert incident.status == IncidentStatus.NEW

    def test_incident_default_priority(self):
        """Test incident default priority is P3."""
        incident = DORAIncident()
        assert incident.priority == IncidentPriority.P3_MEDIUM

    def test_incident_with_title(self):
        """Test creating incident with title."""
        incident = DORAIncident(title="Test Incident")
        assert incident.title == "Test Incident"

    def test_incident_has_timestamps(self):
        """Test incident has proper timestamps."""
        incident = DORAIncident()
        assert incident.created_at is not None


class TestEarlyWarningIndicator:
    """Tests for early warning indicators per Article 17(3)(a)."""

    def test_create_early_warning_indicator(self):
        """Test creating early warning indicator."""
        ewi = EarlyWarningIndicator(
            name="CPU Utilization",
            monitored_system="Trading Server",
            monitored_metric="cpu_percent"
        )
        assert ewi.name == "CPU Utilization"
        assert ewi.indicator_id.startswith("EWI-")

    def test_early_warning_types(self):
        """Test early warning types."""
        assert EarlyWarningType.PERFORMANCE_DEGRADATION is not None
        assert EarlyWarningType.ANOMALOUS_BEHAVIOR is not None


# =============================================================================
# Article 18: Incident Classification Tests
# =============================================================================

class TestIncidentClassificationCreation:
    """Tests for DORAIncidentClassification creation."""

    def test_create_classification_default(self):
        """Test classification service creation."""
        classifier = create_incident_classification()
        assert classifier is not None
        assert isinstance(classifier, DORAIncidentClassification)

    def test_create_classification_with_config(self):
        """Test classification with custom config."""
        config = IncidentClassificationConfig()
        classifier = create_incident_classification(config)
        assert classifier is not None

    def test_classifier_has_classify_method(self):
        """Test classifier has classify_incident method."""
        classifier = create_incident_classification()
        assert hasattr(classifier, 'classify_incident')


class TestClassificationThresholds:
    """Tests for CDR 2024/1772 classification thresholds."""

    def test_retail_client_threshold(self):
        """Test default retail client threshold."""
        thresholds = ClassificationThresholds()
        assert thresholds.retail_client_count == 5000

    def test_professional_client_threshold(self):
        """Test default professional client threshold."""
        thresholds = ClassificationThresholds()
        assert thresholds.professional_client_count == 100

    def test_duration_threshold(self):
        """Test default duration threshold."""
        thresholds = ClassificationThresholds()
        assert thresholds.duration_hours == 4.0

    def test_economic_impact_threshold(self):
        """Test default economic impact threshold."""
        thresholds = ClassificationThresholds()
        assert thresholds.economic_impact_eur == 100000.0

    def test_countries_affected_threshold(self):
        """Test default countries affected threshold."""
        thresholds = ClassificationThresholds()
        assert thresholds.countries_affected == 2


class TestClientImpactAssessment:
    """Tests for client impact assessment per CDR 2024/1772 Article 2."""

    def test_create_assessment_default(self):
        """Test creating assessment with defaults."""
        assessment = ClientImpactAssessment()
        assert assessment.retail_clients_affected == 0

    def test_client_impact_exceeds_threshold(self):
        """Test client impact threshold detection."""
        assessment = ClientImpactAssessment(
            retail_clients_affected=6000
        )
        assert assessment.exceeds_threshold == True

    def test_client_impact_below_threshold(self):
        """Test client impact below threshold."""
        assessment = ClientImpactAssessment(
            retail_clients_affected=1000
        )
        assert assessment.exceeds_threshold == False


class TestDurationAssessment:
    """Tests for duration assessment per CDR 2024/1772 Article 3."""

    def test_create_duration_assessment(self):
        """Test creating duration assessment."""
        assessment = DurationAssessment(
            total_duration_hours=5.0
        )
        assert assessment.total_duration_hours == 5.0

    def test_duration_exceeds_threshold(self):
        """Test duration exceeds 4-hour threshold."""
        assessment = DurationAssessment(
            total_duration_hours=6.0
        )
        assert assessment.exceeds_threshold == True

    def test_duration_below_threshold(self):
        """Test duration below threshold."""
        assessment = DurationAssessment(
            total_duration_hours=2.0
        )
        assert assessment.exceeds_threshold == False


class TestGeographicAssessment:
    """Tests for geographic assessment per CDR 2024/1772 Article 4."""

    def test_create_geographic_assessment(self):
        """Test creating geographic assessment."""
        assessment = GeographicAssessment(
            member_states_affected=["DE", "FR", "IT"]
        )
        assert len(assessment.member_states_affected) == 3

    def test_geographic_exceeds_threshold(self):
        """Test geographic threshold exceeded."""
        assessment = GeographicAssessment(
            member_states_affected=["DE", "FR"]
        )
        assert assessment.exceeds_threshold == True

    def test_geographic_below_threshold(self):
        """Test geographic below threshold."""
        assessment = GeographicAssessment(
            member_states_affected=["DE"]
        )
        assert assessment.exceeds_threshold == False


class TestDataLossAssessment:
    """Tests for data loss assessment per CDR 2024/1772 Article 5."""

    def test_create_data_loss_assessment(self):
        """Test creating data loss assessment."""
        assessment = DataLossAssessment(
            data_compromised=True,
            records_affected=1000
        )
        assert assessment.data_compromised == True

    def test_data_loss_is_material(self):
        """Test data loss materiality."""
        assessment = DataLossAssessment(
            data_compromised=True
        )
        assert assessment.is_material == True


class TestEconomicImpactAssessment:
    """Tests for economic impact assessment per CDR 2024/1772 Article 7."""

    def test_create_economic_assessment(self):
        """Test creating economic impact assessment."""
        assessment = EconomicImpactAssessment(
            direct_financial_losses_eur=50000.0,
            remediation_costs_eur=30000.0
        )
        assert assessment.direct_financial_losses_eur == 50000.0


class TestMajorIncidentTriggers:
    """Tests for major incident triggers per CDR 2024/1772."""

    def test_client_impact_trigger(self):
        """Test client impact trigger."""
        assert MajorIncidentTrigger.CLIENT_IMPACT_THRESHOLD.value == "client_impact_threshold"

    def test_duration_trigger(self):
        """Test duration trigger."""
        assert MajorIncidentTrigger.DURATION_THRESHOLD.value == "duration_threshold"

    def test_geographic_trigger(self):
        """Test geographic spread trigger."""
        assert MajorIncidentTrigger.GEOGRAPHIC_SPREAD_THRESHOLD.value == "geographic_spread_threshold"

    def test_data_breach_trigger(self):
        """Test data breach trigger."""
        assert MajorIncidentTrigger.DATA_BREACH.value == "data_breach"

    def test_critical_service_trigger(self):
        """Test critical service breach trigger."""
        assert MajorIncidentTrigger.CRITICAL_SERVICE_BREACH.value == "critical_service_breach"

    def test_economic_impact_trigger(self):
        """Test economic impact trigger."""
        assert MajorIncidentTrigger.ECONOMIC_IMPACT_THRESHOLD.value == "economic_impact_threshold"


class TestIncidentClassificationTypes:
    """Tests for incident classification types."""

    def test_major_classification(self):
        """Test major incident classification."""
        assert IncidentClassificationType.MAJOR.value == "major"

    def test_significant_classification(self):
        """Test significant incident classification."""
        assert IncidentClassificationType.SIGNIFICANT.value == "significant"

    def test_minor_classification(self):
        """Test minor incident classification."""
        assert IncidentClassificationType.MINOR.value == "minor"


# =============================================================================
# Article 19: Incident Reporting Tests
# =============================================================================

class TestIncidentReporterCreation:
    """Tests for DORAIncidentReporter creation."""

    def test_create_reporter_default(self):
        """Test reporter creation."""
        reporter = create_incident_reporter()
        assert reporter is not None
        assert isinstance(reporter, DORAIncidentReporter)

    def test_create_reporter_with_config(self):
        """Test reporter with custom config."""
        config = IncidentReportingConfig(
            entity_name="Test Entity",
            entity_lei="LEI123456789012345678"
        )
        reporter = create_incident_reporter(config)
        assert reporter is not None


class TestReportTypes:
    """Tests for report types per CDR 2025/301."""

    def test_initial_report_type(self):
        """Test initial notification type."""
        assert ReportType.INITIAL_NOTIFICATION.value == "initial_notification"

    def test_intermediate_report_type(self):
        """Test intermediate report type."""
        assert ReportType.INTERMEDIATE_REPORT.value == "intermediate_report"

    def test_final_report_type(self):
        """Test final report type."""
        assert ReportType.FINAL_REPORT.value == "final_report"


class TestReportStatus:
    """Tests for report status tracking."""

    def test_draft_status(self):
        """Test draft status."""
        assert ReportStatus.DRAFT.value == "draft"

    def test_submitted_status(self):
        """Test submitted status."""
        assert ReportStatus.SUBMITTED.value == "submitted"

    def test_acknowledged_status(self):
        """Test acknowledged status."""
        assert ReportStatus.ACKNOWLEDGED.value == "acknowledged"


class TestCompetentAuthority:
    """Tests for competent authority data structure."""

    def test_create_competent_authority(self):
        """Test creating competent authority."""
        authority = CompetentAuthority(
            authority_id="AUTH001",
            name="BaFin",
            authority_type=CompetentAuthorityType.NCA_PRIMARY,
            country_code="DE"
        )
        assert authority.name == "BaFin"
        assert authority.country_code == "DE"

    def test_authority_types(self):
        """Test authority types."""
        assert CompetentAuthorityType.NCA_PRIMARY is not None
        assert CompetentAuthorityType.ESA is not None


class TestIncidentTypeCode:
    """Tests for incident type codes."""

    def test_system_failure_code(self):
        """Test system failure code."""
        assert IncidentTypeCode.SYSTEM_FAILURE.value == "SYSF"

    def test_cyber_attack_code(self):
        """Test cyber attack code."""
        assert IncidentTypeCode.CYBER_ATTACK.value == "CYBA"


class TestRootCauseCategory:
    """Tests for root cause categories."""

    def test_accidental_internal_category(self):
        """Test accidental internal category."""
        assert RootCauseCategory.ACCIDENTAL_INTERNAL is not None

    def test_system_software_category(self):
        """Test system software category."""
        assert RootCauseCategory.SYSTEM_SOFTWARE is not None


# =============================================================================
# Article 19(4): Cyber Threat Notification Tests
# =============================================================================

class TestCyberThreatServiceCreation:
    """Tests for CyberThreatNotificationService creation."""

    def test_create_service_default(self):
        """Test service creation."""
        service = create_cyber_threat_notification_service()
        assert service is not None
        assert isinstance(service, CyberThreatNotificationService)

    def test_create_service_with_config(self):
        """Test service with custom config."""
        config = CyberThreatNotificationConfig(
            entity_name="Test Entity"
        )
        service = create_cyber_threat_notification_service(config)
        assert service is not None


class TestThreatCategories:
    """Tests for threat category enumeration."""

    def test_malware_category(self):
        """Test malware category."""
        assert ThreatCategory.MALWARE.value == "malware"

    def test_ransomware_category(self):
        """Test ransomware category."""
        assert ThreatCategory.RANSOMWARE.value == "ransomware"

    def test_phishing_category(self):
        """Test phishing category."""
        assert ThreatCategory.PHISHING.value == "phishing"

    def test_denial_of_service_category(self):
        """Test denial of service category."""
        assert ThreatCategory.DENIAL_OF_SERVICE.value == "denial_of_service"

    def test_apt_category(self):
        """Test APT category."""
        assert ThreatCategory.APT.value == "apt"


class TestThreatSeverity:
    """Tests for threat severity levels."""

    def test_critical_severity(self):
        """Test critical severity."""
        assert ThreatSeverity.CRITICAL.value == "critical"

    def test_high_severity(self):
        """Test high severity."""
        assert ThreatSeverity.HIGH.value == "high"

    def test_medium_severity(self):
        """Test medium severity."""
        assert ThreatSeverity.MEDIUM.value == "medium"

    def test_low_severity(self):
        """Test low severity."""
        assert ThreatSeverity.LOW.value == "low"


class TestThreatIndicator:
    """Tests for threat indicator data structure."""

    def test_create_indicator(self):
        """Test creating threat indicator."""
        indicator = ThreatIndicator(
            indicator_type="ip_address",
            indicator_value="192.168.1.100"
        )
        assert indicator.indicator_type == "ip_address"
        assert indicator.indicator_value == "192.168.1.100"


class TestCyberThreat:
    """Tests for cyber threat data structure."""

    def test_create_cyber_threat(self):
        """Test creating cyber threat."""
        threat = CyberThreat(
            category=ThreatCategory.MALWARE,
            severity=ThreatSeverity.HIGH,
            title="Banking Trojan Detected"
        )
        assert threat.category == ThreatCategory.MALWARE
        assert threat.severity == ThreatSeverity.HIGH


class TestThreatActorTypes:
    """Tests for threat actor types."""

    def test_nation_state_actor(self):
        """Test nation state actor."""
        assert ThreatActorType.NATION_STATE is not None

    def test_cybercriminal_actor(self):
        """Test cybercriminal actor."""
        assert ThreatActorType.CYBERCRIMINAL is not None


class TestThreatSignificance:
    """Tests for threat significance levels."""

    def test_significant_significance(self):
        """Test significant level."""
        assert ThreatSignificance.SIGNIFICANT is not None

    def test_potentially_significant(self):
        """Test potentially significant level."""
        assert ThreatSignificance.POTENTIALLY_SIGNIFICANT is not None


# =============================================================================
# Article 20: Reporting Templates Tests
# =============================================================================

class TestReportingTemplatesCreation:
    """Tests for DORAReportingTemplates creation."""

    def test_create_templates_default(self):
        """Test templates service creation."""
        templates = create_reporting_templates()
        assert templates is not None
        assert isinstance(templates, DORAReportingTemplates)

    def test_create_templates_with_entity(self):
        """Test templates with entity info."""
        templates = create_reporting_templates(
            entity_lei="549300EXAMPLE0000",
            entity_name="Test Entity",
            entity_type="credit_institution",
            entity_country="DE"
        )
        assert templates.entity_lei == "549300EXAMPLE0000"


class TestTemplateIncidentTypeCodes:
    """Tests for ITS incident type codes."""

    def test_cyber_attack_code(self):
        """Test cyber attack code."""
        assert TemplateIncidentTypeCode.CYBA.value == "CYBA"

    def test_system_failure_code(self):
        """Test system failure code."""
        assert TemplateIncidentTypeCode.SYSF.value == "SYSF"

    def test_external_event_code(self):
        """Test external event code."""
        assert TemplateIncidentTypeCode.EXTE.value == "EXTE"

    def test_third_party_failure_code(self):
        """Test third party failure code."""
        assert TemplateIncidentTypeCode.TPFA.value == "TPFA"


class TestDataTypeCodes:
    """Tests for ITS data type codes."""

    def test_personal_data_code(self):
        """Test personal data code."""
        assert DataTypeCode.PERS.value == "PERS"

    def test_financial_data_code(self):
        """Test financial data code."""
        assert DataTypeCode.FINA.value == "FINA"

    def test_confidential_data_code(self):
        """Test confidential data code."""
        assert DataTypeCode.CONF.value == "CONF"


class TestClientTypeCodes:
    """Tests for ITS client type codes."""

    def test_retail_code(self):
        """Test retail client code."""
        assert ClientTypeCode.RETA.value == "RETA"

    def test_professional_code(self):
        """Test professional client code."""
        assert ClientTypeCode.PROF.value == "PROF"

    def test_institutional_code(self):
        """Test institutional client code."""
        assert ClientTypeCode.INST.value == "INST"


class TestServiceTypeCodes:
    """Tests for ITS service type codes."""

    def test_order_execution_code(self):
        """Test order execution code."""
        assert ServiceTypeCode.OREX.value == "OREX"

    def test_payment_code(self):
        """Test payment code."""
        assert ServiceTypeCode.PAYM.value == "PAYM"

    def test_settlement_code(self):
        """Test settlement code."""
        assert ServiceTypeCode.SETT.value == "SETT"


class TestTimelineEvent:
    """Tests for timeline event structure."""

    def test_create_timeline_event(self):
        """Test creating timeline event."""
        event = TimelineEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="detection",
            description="Incident detected"
        )
        assert event.event_type == "detection"
        assert event.event_id.startswith("EVT-")


class TestITSTemplates:
    """Tests for ITS template structures."""

    def test_initial_template_auto_reference(self):
        """Test initial template auto-generates reference."""
        template = ITSInitialNotificationTemplate()
        assert template.report_reference.startswith("INIT-")

    def test_initial_template_report_type(self):
        """Test initial template report type."""
        template = ITSInitialNotificationTemplate()
        assert template.report_type == "INIT"

    def test_intermediate_template_auto_reference(self):
        """Test intermediate template auto-generates reference."""
        template = ITSIntermediateReportTemplate()
        assert template.report_reference.startswith("INTM-")

    def test_intermediate_template_report_type(self):
        """Test intermediate template report type."""
        template = ITSIntermediateReportTemplate()
        assert template.report_type == "INTM"

    def test_final_template_auto_reference(self):
        """Test final template auto-generates reference."""
        template = ITSFinalReportTemplate()
        assert template.report_reference.startswith("FINL-")

    def test_final_template_report_type(self):
        """Test final template report type."""
        template = ITSFinalReportTemplate()
        assert template.report_type == "FINL"


class TestTemplateValidation:
    """Tests for template validation."""

    def test_initial_template_validation_missing_fields(self):
        """Test initial template validation with missing fields."""
        template = ITSInitialNotificationTemplate()
        is_valid, errors = template.validate()
        assert is_valid == False
        assert len(errors) > 0

    def test_initial_template_validation_complete(self):
        """Test initial template validation with complete fields."""
        template = ITSInitialNotificationTemplate(
            reporting_entity_lei="549300EXAMPLE0000",
            reporting_entity_name="Test Entity",
            incident_reference="INC-001",
            detection_datetime="2025-01-15T10:00:00Z",
            classification_datetime="2025-01-15T10:30:00Z",
            brief_description="Test incident",
            contact_person_email="test@example.com",
            member_states_affected=["DE"]
        )
        is_valid, errors = template.validate()
        assert is_valid == True


class TestTemplateExport:
    """Tests for template export functionality."""

    def test_export_to_json(self):
        """Test export to JSON."""
        templates = create_reporting_templates()
        initial = templates.create_initial_notification(
            incident_reference="INC-001",
            detection_datetime="2025-01-15T10:00:00Z",
            classification_datetime="2025-01-15T10:30:00Z",
            brief_description="Test incident"
        )
        json_str = templates.export_to_json(initial)
        assert isinstance(json_str, str)
        assert "INC-001" in json_str

    def test_export_to_dict(self):
        """Test export to dictionary."""
        templates = create_reporting_templates()
        initial = templates.create_initial_notification(
            incident_reference="INC-001",
            detection_datetime="2025-01-15T10:00:00Z",
            classification_datetime="2025-01-15T10:30:00Z",
            brief_description="Test incident"
        )
        data = templates.export_to_dict(initial)
        assert isinstance(data, dict)
        assert data["incident_reference"] == "INC-001"


# =============================================================================
# Article 22: Supervisory Feedback Tests
# =============================================================================

class TestSupervisoryFeedbackCreation:
    """Tests for DORASupervisioryFeedback creation."""

    def test_create_feedback_handler(self):
        """Test supervisory feedback handler creation."""
        handler = create_supervisory_feedback(
            entity_id="ENTITY001",
            entity_name="Test Financial Entity"
        )
        assert handler is not None
        assert isinstance(handler, DORASupervisioryFeedback)

    def test_feedback_handler_entity_info(self):
        """Test feedback handler stores entity info."""
        handler = create_supervisory_feedback(
            entity_id="ENTITY001",
            entity_name="Test Financial Entity"
        )
        assert handler.entity_id == "ENTITY001"
        assert handler.entity_name == "Test Financial Entity"


class TestFeedbackTypes:
    """Tests for feedback type enumeration."""

    def test_acknowledgement_type(self):
        """Test acknowledgement feedback type."""
        assert FeedbackType.ACKNOWLEDGEMENT.value == "acknowledgement"

    def test_clarification_request_type(self):
        """Test clarification request type."""
        assert FeedbackType.CLARIFICATION_REQUEST.value == "clarification_request"

    def test_corrective_action_type(self):
        """Test corrective action required type."""
        assert FeedbackType.CORRECTIVE_ACTION_REQUIRED.value == "corrective_action_required"

    def test_guidance_type(self):
        """Test guidance type."""
        assert FeedbackType.GUIDANCE.value == "guidance"

    def test_warning_type(self):
        """Test warning type."""
        assert FeedbackType.WARNING.value == "warning"


class TestFeedbackPriority:
    """Tests for feedback priority levels."""

    def test_critical_priority(self):
        """Test critical priority."""
        assert FeedbackPriority.CRITICAL.value == "critical"

    def test_high_priority(self):
        """Test high priority."""
        assert FeedbackPriority.HIGH.value == "high"

    def test_medium_priority(self):
        """Test medium priority."""
        assert FeedbackPriority.MEDIUM.value == "medium"

    def test_low_priority(self):
        """Test low priority."""
        assert FeedbackPriority.LOW.value == "low"


class TestFeedbackStatus:
    """Tests for feedback status tracking."""

    def test_received_status(self):
        """Test received status."""
        assert FeedbackStatus.RECEIVED.value == "received"

    def test_acknowledged_status(self):
        """Test acknowledged status."""
        assert FeedbackStatus.ACKNOWLEDGED.value == "acknowledged"

    def test_under_review_status(self):
        """Test under review status."""
        assert FeedbackStatus.UNDER_REVIEW.value == "under_review"

    def test_resolved_status(self):
        """Test resolved status."""
        assert FeedbackStatus.RESOLVED.value == "resolved"

    def test_closed_status(self):
        """Test closed status."""
        assert FeedbackStatus.CLOSED.value == "closed"


class TestCorrectiveActionTypes:
    """Tests for corrective action types."""

    def test_process_improvement(self):
        """Test process improvement action."""
        assert CorrectiveActionType.PROCESS_IMPROVEMENT.value == "process_improvement"

    def test_documentation_update(self):
        """Test documentation update action."""
        assert CorrectiveActionType.DOCUMENTATION_UPDATE.value == "documentation_update"

    def test_training_required(self):
        """Test training required action."""
        assert CorrectiveActionType.TRAINING_REQUIRED.value == "training_required"

    def test_technical_remediation(self):
        """Test technical remediation action."""
        assert CorrectiveActionType.TECHNICAL_REMEDIATION.value == "technical_remediation"


class TestFeedbackReception:
    """Tests for receiving supervisory feedback."""

    def test_receive_feedback(self):
        """Test receiving supervisory feedback."""
        handler = create_supervisory_feedback(
            entity_id="ENTITY001",
            entity_name="Test Financial Entity"
        )

        authority = FeedbackCompetentAuthority(
            authority_id="AUTH001",
            name="BaFin",
            country_code="DE",
            contact_email="feedback@bafin.de"
        )

        feedback = handler.receive_feedback(
            incident_id="INC001",
            report_id="RPT001",
            authority=authority,
            feedback_type=FeedbackType.CLARIFICATION_REQUEST,
            priority=FeedbackPriority.HIGH,
            subject="Clarification Required",
            content="Please provide additional details"
        )

        assert feedback is not None
        assert feedback.feedback_type == FeedbackType.CLARIFICATION_REQUEST

    def test_feedback_auto_acknowledge(self):
        """Test feedback auto-acknowledgement."""
        handler = create_supervisory_feedback(
            entity_id="ENTITY001",
            entity_name="Test Entity",
            auto_acknowledge=True
        )

        authority = FeedbackCompetentAuthority(
            authority_id="AUTH001",
            name="BaFin",
            country_code="DE",
            contact_email="test@bafin.de"
        )

        feedback = handler.receive_feedback(
            incident_id="INC001",
            report_id="RPT001",
            authority=authority,
            feedback_type=FeedbackType.ACKNOWLEDGEMENT,
            priority=FeedbackPriority.LOW,
            subject="Acknowledgement",
            content="Report received"
        )

        assert feedback.status == FeedbackStatus.ACKNOWLEDGED


class TestCorrectiveActions:
    """Tests for corrective action management."""

    def test_create_corrective_action(self):
        """Test creating corrective action."""
        handler = create_supervisory_feedback(
            entity_id="ENTITY001",
            entity_name="Test Entity"
        )

        authority = FeedbackCompetentAuthority(
            authority_id="AUTH001",
            name="BaFin",
            country_code="DE",
            contact_email="test@bafin.de"
        )

        feedback = handler.receive_feedback(
            incident_id="INC001",
            report_id="RPT001",
            authority=authority,
            feedback_type=FeedbackType.CORRECTIVE_ACTION_REQUIRED,
            priority=FeedbackPriority.HIGH,
            subject="Corrective Action",
            content="Please implement improvements"
        )

        action = handler.create_corrective_action(
            feedback_id=feedback.feedback_id,
            action_type=CorrectiveActionType.PROCESS_IMPROVEMENT,
            description="Improve incident detection",
            assigned_to="IT Security Team",
            deadline=datetime.utcnow() + timedelta(days=30)
        )

        assert action is not None
        assert action.action_type == CorrectiveActionType.PROCESS_IMPROVEMENT


# =============================================================================
# Article 23: Third-Party Incidents Tests
# =============================================================================

class TestThirdPartyIncidentsCreation:
    """Tests for DORAThirdPartyIncidents creation."""

    def test_create_manager(self):
        """Test third-party incidents manager creation."""
        manager = create_third_party_incidents(
            entity_id="ENTITY001",
            entity_name="Test Financial Entity"
        )
        assert manager is not None
        assert isinstance(manager, DORAThirdPartyIncidents)

    def test_manager_entity_info(self):
        """Test manager stores entity info."""
        manager = create_third_party_incidents(
            entity_id="ENTITY001",
            entity_name="Test Financial Entity"
        )
        assert manager.entity_id == "ENTITY001"


class TestThirdPartyProviderTypes:
    """Tests for third-party provider type enumeration."""

    def test_cloud_provider_type(self):
        """Test cloud service provider type."""
        assert ThirdPartyProviderType.CLOUD_SERVICE_PROVIDER.value == "cloud_service_provider"

    def test_software_vendor_type(self):
        """Test software vendor type."""
        assert ThirdPartyProviderType.SOFTWARE_VENDOR.value == "software_vendor"

    def test_data_center_type(self):
        """Test data center type."""
        assert ThirdPartyProviderType.DATA_CENTER.value == "data_center"

    def test_network_provider_type(self):
        """Test network provider type."""
        assert ThirdPartyProviderType.NETWORK_PROVIDER.value == "network_provider"

    def test_payment_processor_type(self):
        """Test payment processor type."""
        assert ThirdPartyProviderType.PAYMENT_PROCESSOR.value == "payment_processor"


class TestThirdPartyCriticality:
    """Tests for third-party criticality levels per DORA Article 28."""

    def test_critical_level(self):
        """Test critical level."""
        assert ThirdPartyCriticality.CRITICAL.value == "critical"

    def test_important_level(self):
        """Test important level."""
        assert ThirdPartyCriticality.IMPORTANT.value == "important"

    def test_standard_level(self):
        """Test standard level."""
        assert ThirdPartyCriticality.STANDARD.value == "standard"

    def test_non_critical_level(self):
        """Test non-critical level."""
        assert ThirdPartyCriticality.NON_CRITICAL.value == "non_critical"


class TestThirdPartyIncidentTypes:
    """Tests for third-party incident types."""

    def test_service_outage_type(self):
        """Test service outage type."""
        assert ThirdPartyIncidentType.SERVICE_OUTAGE.value == "service_outage"

    def test_security_breach_type(self):
        """Test security breach type."""
        assert ThirdPartyIncidentType.SECURITY_BREACH.value == "security_breach"

    def test_data_breach_type(self):
        """Test data breach type."""
        assert ThirdPartyIncidentType.DATA_BREACH.value == "data_breach"

    def test_supply_chain_attack_type(self):
        """Test supply chain attack type."""
        assert ThirdPartyIncidentType.SUPPLY_CHAIN_ATTACK.value == "supply_chain_attack"

    def test_fourth_party_incident_type(self):
        """Test fourth party incident type."""
        assert ThirdPartyIncidentType.FOURTH_PARTY_INCIDENT.value == "fourth_party_incident"


class TestContractualSLAStatus:
    """Tests for SLA compliance status."""

    def test_compliant_status(self):
        """Test compliant status."""
        assert ContractualSLAStatus.COMPLIANT.value == "compliant"

    def test_at_risk_status(self):
        """Test at risk status."""
        assert ContractualSLAStatus.AT_RISK.value == "at_risk"

    def test_breached_status(self):
        """Test breached status."""
        assert ContractualSLAStatus.BREACHED.value == "breached"


class TestThirdPartyProvider:
    """Tests for third-party provider data structure."""

    def test_create_provider(self):
        """Test creating a third-party provider."""
        provider = ThirdPartyProvider(
            provider_id="PROV001",
            name="Cloud Provider Inc",
            provider_type=ThirdPartyProviderType.CLOUD_SERVICE_PROVIDER,
            criticality=ThirdPartyCriticality.CRITICAL,
            services_provided=["Cloud Hosting", "Database Services"],
            contract_reference="CONTRACT-001",
            primary_contact_name="John Doe",
            primary_contact_email="john@provider.com",
            primary_contact_phone="+1-555-0123"
        )
        assert provider.name == "Cloud Provider Inc"
        assert provider.criticality == ThirdPartyCriticality.CRITICAL

    def test_critical_provider_flag(self):
        """Test critical/important provider flag per Article 28."""
        provider = ThirdPartyProvider(
            provider_id="PROV001",
            name="Critical Provider",
            provider_type=ThirdPartyProviderType.SOFTWARE_VENDOR,
            criticality=ThirdPartyCriticality.CRITICAL,
            services_provided=["Core Banking"],
            contract_reference="CONTRACT-001",
            primary_contact_name="Contact",
            primary_contact_email="contact@provider.com",
            primary_contact_phone="+1-555-0000"
        )
        assert provider.is_critical_or_important == True

    def test_non_critical_provider_flag(self):
        """Test non-critical provider flag."""
        provider = ThirdPartyProvider(
            provider_id="PROV001",
            name="Standard Provider",
            provider_type=ThirdPartyProviderType.SOFTWARE_VENDOR,
            criticality=ThirdPartyCriticality.STANDARD,
            services_provided=["Reporting"],
            contract_reference="CONTRACT-001",
            primary_contact_name="Contact",
            primary_contact_email="contact@provider.com",
            primary_contact_phone="+1-555-0000"
        )
        assert provider.is_critical_or_important == False


class TestProviderRegistration:
    """Tests for third-party provider registration."""

    def test_register_provider(self):
        """Test registering a third-party provider."""
        manager = create_third_party_incidents(
            entity_id="ENTITY001",
            entity_name="Test Financial Entity"
        )

        provider = manager.register_provider(
            name="Cloud Provider",
            provider_type=ThirdPartyProviderType.CLOUD_SERVICE_PROVIDER,
            criticality=ThirdPartyCriticality.CRITICAL,
            services_provided=["Cloud Hosting"],
            contract_reference="CONTRACT-001",
            primary_contact_name="Contact",
            primary_contact_email="contact@provider.com",
            primary_contact_phone="+1-555-0000"
        )

        assert provider is not None
        assert provider.provider_type == ThirdPartyProviderType.CLOUD_SERVICE_PROVIDER


class TestThirdPartyIncidentReporting:
    """Tests for third-party incident reporting."""

    def test_report_incident(self):
        """Test reporting a third-party incident."""
        manager = create_third_party_incidents(
            entity_id="ENTITY001",
            entity_name="Test Financial Entity"
        )

        provider = manager.register_provider(
            name="Cloud Provider",
            provider_type=ThirdPartyProviderType.CLOUD_SERVICE_PROVIDER,
            criticality=ThirdPartyCriticality.CRITICAL,
            services_provided=["Cloud Hosting"],
            contract_reference="CONTRACT-001",
            primary_contact_name="Contact",
            primary_contact_email="contact@provider.com",
            primary_contact_phone="+1-555-0000"
        )

        incident = manager.report_incident(
            provider_id=provider.provider_id,
            incident_type=ThirdPartyIncidentType.SERVICE_OUTAGE,
            severity=ThirdPartyIncidentSeverity.HIGH,
            title="Cloud Service Outage",
            description="Cloud hosting service unavailable"
        )

        assert incident is not None
        assert incident.incident_type == ThirdPartyIncidentType.SERVICE_OUTAGE

    def test_incident_requires_notification_for_critical_provider(self):
        """Test incident notification requirement for critical provider."""
        manager = create_third_party_incidents(
            entity_id="ENTITY001",
            entity_name="Test Entity"
        )

        provider = manager.register_provider(
            name="Critical Cloud",
            provider_type=ThirdPartyProviderType.CLOUD_SERVICE_PROVIDER,
            criticality=ThirdPartyCriticality.CRITICAL,
            services_provided=["Core Services"],
            contract_reference="CONTRACT-001",
            primary_contact_name="Contact",
            primary_contact_email="contact@provider.com",
            primary_contact_phone="+1-555-0000"
        )

        incident = manager.report_incident(
            provider_id=provider.provider_id,
            incident_type=ThirdPartyIncidentType.SERVICE_OUTAGE,
            severity=ThirdPartyIncidentSeverity.CRITICAL,
            title="Critical Outage",
            description="Complete service failure"
        )

        # Critical severity + Critical provider = major incident
        assert incident.is_major_incident == True


class TestThirdPartyEscalation:
    """Tests for third-party incident escalation."""

    def test_escalation_levels(self):
        """Test escalation levels exist."""
        assert ThirdPartyEscalationLevel.OPERATIONAL.value == "operational"
        assert ThirdPartyEscalationLevel.MANAGEMENT.value == "management"
        assert ThirdPartyEscalationLevel.EXECUTIVE.value == "executive"
        assert ThirdPartyEscalationLevel.REGULATORY.value == "regulatory"


class TestCommunicationChannels:
    """Tests for communication channels."""

    def test_email_channel(self):
        """Test email channel."""
        assert ThirdPartyCommunicationChannel.EMAIL.value == "email"

    def test_phone_channel(self):
        """Test phone channel."""
        assert ThirdPartyCommunicationChannel.PHONE.value == "phone"

    def test_incident_portal_channel(self):
        """Test incident portal channel."""
        assert ThirdPartyCommunicationChannel.INCIDENT_PORTAL.value == "incident_portal"


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase2ModuleInstantiation:
    """Integration tests for Phase 2 module instantiation."""

    def test_incident_management_instantiation(self):
        """Test incident management instantiation."""
        manager = create_incident_management()
        assert manager is not None

    def test_incident_classification_instantiation(self):
        """Test incident classification instantiation."""
        classifier = create_incident_classification()
        assert classifier is not None

    def test_incident_reporter_instantiation(self):
        """Test incident reporter instantiation."""
        reporter = create_incident_reporter()
        assert reporter is not None

    def test_cyber_threat_service_instantiation(self):
        """Test cyber threat service instantiation."""
        service = create_cyber_threat_notification_service()
        assert service is not None

    def test_reporting_templates_instantiation(self):
        """Test reporting templates instantiation."""
        templates = create_reporting_templates()
        assert templates is not None

    def test_supervisory_feedback_instantiation(self):
        """Test supervisory feedback instantiation."""
        handler = create_supervisory_feedback("E1", "Test Entity")
        assert handler is not None

    def test_third_party_incidents_instantiation(self):
        """Test third-party incidents instantiation."""
        manager = create_third_party_incidents("E1", "Test Entity")
        assert manager is not None


class TestModuleImports:
    """Test that all Phase 2 modules can be imported correctly."""

    def test_import_from_dora_package(self):
        """Test imports from main dora package."""
        from services.dora import (
            # Phase 2 - Incident Management
            DORAIncidentManagement,
            create_incident_management,
            # Phase 2 - Classification
            DORAIncidentClassification,
            create_incident_classification,
            # Phase 2 - Reporting
            DORAIncidentReporter,
            create_incident_reporter,
            # Phase 2 - Cyber Threats
            CyberThreatNotificationService,
            create_cyber_threat_notification_service,
            # Phase 2 - Templates
            DORAReportingTemplates,
            create_reporting_templates,
            # Phase 2 - Feedback
            DORASupervisioryFeedback,
            create_supervisory_feedback,
            # Phase 2 - Third Party
            DORAThirdPartyIncidents,
            create_third_party_incidents,
        )

        assert DORAIncidentManagement is not None
        assert DORAIncidentClassification is not None
        assert DORAIncidentReporter is not None
        assert CyberThreatNotificationService is not None
        assert DORAReportingTemplates is not None
        assert DORASupervisioryFeedback is not None
        assert DORAThirdPartyIncidents is not None

    def test_dora_compliance_phase(self):
        """Test DORA compliance phase is set to 2."""
        from services.dora import __dora_compliance_phase__
        assert __dora_compliance_phase__ == 2

    def test_dora_version(self):
        """Test DORA version is 0.3.0."""
        from services.dora import __version__
        assert __version__ == "0.3.0"


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_incident_lifecycle(self):
        """Test complete incident lifecycle."""
        # Create incident manager
        manager = create_incident_management()

        # Create incident
        incident = manager.create_incident(
            incident_type=ICTEventType.SYSTEM_FAILURE,
            title="Database Outage",
            description="Primary database server unresponsive"
        )

        assert incident is not None
        # create_incident automatically moves to RECORDED status
        assert incident.status == IncidentStatus.RECORDED

    def test_reporting_workflow(self):
        """Test reporting workflow."""
        templates = create_reporting_templates(
            entity_lei="549300EXAMPLE0000",
            entity_name="Test Bank"
        )

        # Create initial notification
        initial = templates.create_initial_notification(
            incident_reference="INC-001",
            detection_datetime="2025-01-15T10:00:00Z",
            classification_datetime="2025-01-15T10:30:00Z",
            brief_description="System failure affecting trading"
        )

        assert initial is not None
        assert initial.incident_reference == "INC-001"

    def test_third_party_incident_workflow(self):
        """Test third-party incident workflow."""
        manager = create_third_party_incidents(
            entity_id="E001",
            entity_name="Test Bank"
        )

        # Register provider
        provider = manager.register_provider(
            name="Cloud Provider",
            provider_type=ThirdPartyProviderType.CLOUD_SERVICE_PROVIDER,
            criticality=ThirdPartyCriticality.CRITICAL,
            services_provided=["Hosting"],
            contract_reference="C001",
            primary_contact_name="John",
            primary_contact_email="john@cloud.com",
            primary_contact_phone="+1-555-0000"
        )

        # Report incident
        incident = manager.report_incident(
            provider_id=provider.provider_id,
            incident_type=ThirdPartyIncidentType.SERVICE_OUTAGE,
            severity=ThirdPartyIncidentSeverity.HIGH,
            title="Service Outage",
            description="Cloud service unavailable"
        )

        # Get metrics
        metrics = manager.get_metrics()

        assert metrics["total_incidents"] == 1
        assert metrics["active_incidents"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
