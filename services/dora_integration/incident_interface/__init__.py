# -*- coding: utf-8 -*-
"""
Incident Communication Interface - DORA Integration Layer Phase 2.

Provides comprehensive incident communication capabilities for ICT providers:
- Client incident notifications (Article 30(2)(d))
- Incident classification (CDR 2024/1772)
- Incident data export for client NCA reporting
- Cyber threat notifications (Article 19(4))
- Crisis communication management (Article 14)

CRITICAL DISTINCTION:
    We notify CLIENTS. Clients report to NCAs.
    We are ICT providers, NOT financial entities.
    We do NOT submit directly to NCAs (that's client's obligation).

Flow:
    1. DORAIncidentClassification.classify_incident()
    2. ClientNotificationService.notify_client()  # We notify client
    3. DORAIncidentReporter.generate_client_data_package()  # Client gets data
    4. Client submits to their NCA using our data package

Exported Classes:
    Primary Services:
        - ClientNotificationService: Client notification system (Art. 30(2)(d))
        - DORAIncidentClassification: Incident classifier (Art. 18)
        - DORAIncidentReporter: Report template generator (Art. 19)
        - CyberThreatNotificationService: Threat notifications (Art. 19(4))
        - DORACommunication: Crisis communication management (Art. 14)

    Enums and Data Types:
        - IncidentSeverity, NotificationStatus: Notification enums
        - IncidentClassificationType, MajorIncidentTrigger: Classification enums
        - ReportType, ReportStatus: Reporting enums
        - ThreatCategory, ThreatSeverity: Threat enums
        - CommunicationChannel, StakeholderType: Communication enums

    Configuration Classes:
        - ClientNotificationConfig
        - IncidentClassificationConfig
        - IncidentReportingConfig
        - CyberThreatNotificationConfig
        - CommunicationConfig

    Data Structures:
        - ClientContact, IncidentNotification, IncidentUpdate
        - ClassificationThresholds, IncidentClassificationResult
        - InitialNotificationReport, IntermediateReport, FinalReport
        - ClientDataPackage
        - CyberThreat, ThreatSignificanceAssessment, ThreatNotification
        - CommunicationPolicy, CommunicationRecord, CrisisStatus

References:
    - DORA Article 14: Crisis communication
    - DORA Article 18: Incident classification
    - DORA Article 19: Incident reporting
    - DORA Article 30(2)(d): Incident notification obligations
    - CDR 2024/1772: RTS on Incident Classification
    - CDR 2025/301: RTS on Incident Reporting

Migration Status: Phase 2 - Complete
"""

from __future__ import annotations

# =============================================================================
# Client Incident Notification (Article 30(2)(d))
# =============================================================================

from services.dora_integration.incident_interface.client_incident_notification import (
    # Main Service
    ClientNotificationService,
    DORAClientNotification,  # Alias for backward compatibility
    # Configuration
    ClientNotificationConfig,
    # Enums
    IncidentSeverity,
    NotificationStatus,
    NotificationChannel,
    IncidentCategory,
    # Data Structures
    ClientContact,
    IncidentNotification,
    IncidentUpdate,
    ClientIncident,
    # Factory Functions
    create_client_notification_service,
    create_client_notification_system,
    # Template Helper
    get_notification_template,
)

# =============================================================================
# Incident Classification (Article 18)
# =============================================================================

from services.dora_integration.incident_interface.incident_classification import (
    # Main Service
    DORAIncidentClassification,
    # Configuration
    IncidentClassificationConfig,
    ClassificationThresholds,
    # Enums
    IncidentClassificationType,
    ClientType,
    DataType,
    CriticalServiceType,
    MajorIncidentTrigger,
    ReputationalImpactLevel,
    # Assessment Data Structures
    ClientImpactAssessment,
    DurationAssessment,
    GeographicAssessment,
    DataLossAssessment,
    CriticalServiceAssessment,
    EconomicImpactAssessment,
    ReputationalAssessment,
    RecurringIncidentAssessment,
    MaliciousAccessAssessment,
    # Result
    IncidentClassificationResult,
    # Factory Functions
    create_incident_classification,
    get_default_thresholds,
    get_classification_criteria,
    create_client_impact_assessment,
    create_duration_assessment,
    create_economic_impact_assessment,
    create_data_loss_assessment,
    create_critical_service_assessment,
)

# =============================================================================
# Incident Reporting (Article 19) - Export Templates
# =============================================================================

from services.dora_integration.incident_interface.incident_reporting import (
    # Main Service
    DORAIncidentReporter,
    # Configuration
    IncidentReportingConfig,
    # Enums
    ReportType,
    ReportStatus,
    IncidentTypeCode,
    RootCauseCategory,
    CompetentAuthorityType,
    # Data Structures
    CompetentAuthority,
    InitialNotificationReport,
    IntermediateReport,
    FinalReport,
    ClientDataPackage,
    ReportSubmission,
    # Factory Functions
    create_incident_reporter,
    get_report_deadlines,
    get_report_types,
)

# =============================================================================
# Cyber Threat Notification (Article 19(4))
# =============================================================================

from services.dora_integration.incident_interface.cyber_threat_notification import (
    # Main Service
    CyberThreatNotificationService,
    # Configuration
    CyberThreatNotificationConfig,
    # Enums
    ThreatCategory,
    ThreatActorType,
    ThreatSeverity,
    ThreatStatus,
    ThreatSignificance,
    # Data Structures
    ThreatIndicator,
    CyberThreat,
    ThreatSignificanceAssessment,
    ThreatNotification,
    # Factory Functions
    create_cyber_threat_notification_service,
    get_threat_categories,
    get_threat_severities,
)

# =============================================================================
# Crisis Communication (Article 14)
# =============================================================================

from services.dora_integration.incident_interface.communication import (
    # Main Service
    DORACommunication,
    # Configuration
    CommunicationConfig,
    # Enums
    CommunicationChannel,
    StakeholderType,
    CommunicationPriority,
    CommunicationStatus,
    CrisisPhase,
    PolicyStatus,
    # Data Structures
    CommunicationContact,
    CommunicationTemplate,
    CommunicationRecord,
    CommunicationPolicy,
    CrisisStatus,
    # Factory Functions
    create_communication_service,
    get_communication_channels,
    get_stakeholder_types,
    get_crisis_phases,
)


# =============================================================================
# Module Exports
# =============================================================================

__all__: list[str] = [
    # -------------------------------------------------------------------------
    # Client Incident Notification (Article 30(2)(d))
    # -------------------------------------------------------------------------
    "ClientNotificationService",
    "DORAClientNotification",
    "ClientNotificationConfig",
    "IncidentSeverity",
    "NotificationStatus",
    "NotificationChannel",
    "IncidentCategory",
    "ClientContact",
    "IncidentNotification",
    "IncidentUpdate",
    "ClientIncident",
    "create_client_notification_service",
    "create_client_notification_system",
    "get_notification_template",

    # -------------------------------------------------------------------------
    # Incident Classification (Article 18)
    # -------------------------------------------------------------------------
    "DORAIncidentClassification",
    "IncidentClassificationConfig",
    "ClassificationThresholds",
    "IncidentClassificationType",
    "ClientType",
    "DataType",
    "CriticalServiceType",
    "MajorIncidentTrigger",
    "ReputationalImpactLevel",
    "ClientImpactAssessment",
    "DurationAssessment",
    "GeographicAssessment",
    "DataLossAssessment",
    "CriticalServiceAssessment",
    "EconomicImpactAssessment",
    "ReputationalAssessment",
    "RecurringIncidentAssessment",
    "MaliciousAccessAssessment",
    "IncidentClassificationResult",
    "create_incident_classification",
    "get_default_thresholds",
    "get_classification_criteria",
    "create_client_impact_assessment",
    "create_duration_assessment",
    "create_economic_impact_assessment",
    "create_data_loss_assessment",
    "create_critical_service_assessment",

    # -------------------------------------------------------------------------
    # Incident Reporting (Article 19) - Export Templates
    # -------------------------------------------------------------------------
    "DORAIncidentReporter",
    "IncidentReportingConfig",
    "ReportType",
    "ReportStatus",
    "IncidentTypeCode",
    "RootCauseCategory",
    "CompetentAuthorityType",
    "CompetentAuthority",
    "InitialNotificationReport",
    "IntermediateReport",
    "FinalReport",
    "ClientDataPackage",
    "ReportSubmission",
    "create_incident_reporter",
    "get_report_deadlines",
    "get_report_types",

    # -------------------------------------------------------------------------
    # Cyber Threat Notification (Article 19(4))
    # -------------------------------------------------------------------------
    "CyberThreatNotificationService",
    "CyberThreatNotificationConfig",
    "ThreatCategory",
    "ThreatActorType",
    "ThreatSeverity",
    "ThreatStatus",
    "ThreatSignificance",
    "ThreatIndicator",
    "CyberThreat",
    "ThreatSignificanceAssessment",
    "ThreatNotification",
    "create_cyber_threat_notification_service",
    "get_threat_categories",
    "get_threat_severities",

    # -------------------------------------------------------------------------
    # Crisis Communication (Article 14)
    # -------------------------------------------------------------------------
    "DORACommunication",
    "CommunicationConfig",
    "CommunicationChannel",
    "StakeholderType",
    "CommunicationPriority",
    "CommunicationStatus",
    "CrisisPhase",
    "PolicyStatus",
    "CommunicationContact",
    "CommunicationTemplate",
    "CommunicationRecord",
    "CommunicationPolicy",
    "CrisisStatus",
    "create_communication_service",
    "get_communication_channels",
    "get_stakeholder_types",
    "get_crisis_phases",
]
