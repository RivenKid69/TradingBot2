# -*- coding: utf-8 -*-
"""
Unified Reporting Layer - DORA Integration Layer Phase 5.

Provides cross-regulatory report generation and data packages for ICT service
provider clients to meet their regulatory submission requirements.

CRITICAL DISTINCTION - ICT Provider Role:
    We GENERATE data packages for client reports.
    We DO NOT maintain client registers.
    We DO NOT submit to NCAs.

    Financial entities submit to their NCAs using our data packages.
    We track delivery status, not NCA submission.

This layer provides:

1. UnifiedReportingManager:
   - Cross-regulatory report aggregation
   - Multi-channel client delivery
   - Report lifecycle management

2. DORAReportingTemplates:
   - ITS-compliant incident reporting templates (CDR 2025/301)
   - Initial, Intermediate, Final report pre-population
   - Template validation and export

3. DORARegisterOfInformation (ROI Data Generator):
   - ROI data packages for client submissions (Art. 28(3))
   - Provider identification (B_03.01)
   - Service records (B_06.01)
   - Subcontractor chains (B_04.01)
   - Contract reference data (B_02.01)

ROI Data Generation (NOT register maintenance):
    Clients maintain their Register of Information per Art. 28(3).
    We provide them with structured data packages to populate their ROI:
    - B_02.01: Contract reference data
    - B_03.01: Provider identification (OUR data)
    - B_04.01: Subcontractor chains (OUR subcontractors)
    - B_06.01: ICT service records (OUR services)

    Data clients provide themselves:
    - B_01.01: Entity maintaining register
    - B_01.02: Branch information
    - B_02.02: Contractual arrangement functions
    - B_05.01: Entity making use of ICT services
    - B_99.01: Totals

References:
    - DORA Article 28(3): Register of information requirement
    - DORA Article 19: Incident reporting via financial entities
    - DORA Article 20: Harmonised reporting templates
    - CIR 2024/2956: ITS on Register of Information templates
    - CDR 2025/301: RTS on incident reporting content and templates

Migration Status: Phase 5 - Complete
"""

from __future__ import annotations

# =============================================================================
# Unified Reporting Manager
# =============================================================================

from services.dora_integration.reporting.unified_reporting import (
    # Enums
    ReportType,
    ReportStatus,
    ReportChannel,
    PackageFormat,
    ClientType,
    # Data structures
    ReportDestination,
    ReportValidationResult,
    UnifiedReport,
    SubmissionPackage,
    DeliveryRecord,
    UnifiedReportingConfig,
    # Main class
    UnifiedReportingManager,
    # Factory functions
    create_unified_reporting_manager,
    create_report_destination,
    get_report_types,
    get_report_statuses,
)

# =============================================================================
# Reporting Templates
# =============================================================================

from services.dora_integration.reporting.reporting_templates import (
    # Enums
    IncidentTypeCode,
    DataTypeCode,
    ClientTypeCode,
    ServiceTypeCode,
    ResponseEffectivenessCode,
    TemplateExportFormat,
    # Data structures
    ITSInitialNotificationTemplate,
    ITSIntermediateReportTemplate,
    ITSFinalReportTemplate,
    TimelineEvent,
    ClientIncidentDataPackage,
    # Main class
    DORAReportingTemplates,
    # Factory functions
    create_reporting_templates,
    get_incident_type_codes,
    get_data_type_codes,
    get_service_type_codes,
    get_client_type_codes,
    create_timeline_event,
)

# =============================================================================
# Register of Information (ROI Data Generator)
# =============================================================================

from services.dora_integration.reporting.register_of_information import (
    # Enums
    ContractType,
    ServiceType,
    FunctionType,
    DataLocation,
    ProviderLocationType,
    SubcontractingLevel,
    ExportFormat,
    # Data structures
    ProviderIdentification,
    ContractReferenceData,
    SubcontractorData,
    ServiceRecord,
    ROIDataPackage,
    ROIDataGeneratorConfig,
    # Main class
    DORARegisterOfInformation,
    # Factory functions
    create_register_of_information,
    create_roi_data_generator,
    get_contract_types,
    get_service_types,
    get_subcontracting_levels,
    get_its_templates_provided,
    get_its_templates_client_provides,
)

# =============================================================================
# __all__ exports
# =============================================================================

__all__ = [
    # =========================================================================
    # Unified Reporting Manager
    # =========================================================================

    # Enums
    "ReportType",
    "ReportStatus",
    "ReportChannel",
    "PackageFormat",
    "ClientType",

    # Data structures
    "ReportDestination",
    "ReportValidationResult",
    "UnifiedReport",
    "SubmissionPackage",
    "DeliveryRecord",
    "UnifiedReportingConfig",

    # Main class
    "UnifiedReportingManager",

    # Factory functions
    "create_unified_reporting_manager",
    "create_report_destination",
    "get_report_types",
    "get_report_statuses",

    # =========================================================================
    # Reporting Templates
    # =========================================================================

    # Enums
    "IncidentTypeCode",
    "DataTypeCode",
    "ClientTypeCode",
    "ServiceTypeCode",
    "ResponseEffectivenessCode",
    "TemplateExportFormat",

    # Data structures
    "ITSInitialNotificationTemplate",
    "ITSIntermediateReportTemplate",
    "ITSFinalReportTemplate",
    "TimelineEvent",
    "ClientIncidentDataPackage",

    # Main class
    "DORAReportingTemplates",

    # Factory functions
    "create_reporting_templates",
    "get_incident_type_codes",
    "get_data_type_codes",
    "get_service_type_codes",
    "get_client_type_codes",
    "create_timeline_event",

    # =========================================================================
    # Register of Information (ROI Data Generator)
    # =========================================================================

    # Enums
    "ContractType",
    "ServiceType",
    "FunctionType",
    "DataLocation",
    "ProviderLocationType",
    "SubcontractingLevel",
    "ExportFormat",

    # Data structures
    "ProviderIdentification",
    "ContractReferenceData",
    "SubcontractorData",
    "ServiceRecord",
    "ROIDataPackage",
    "ROIDataGeneratorConfig",

    # Main class
    "DORARegisterOfInformation",

    # Factory functions
    "create_register_of_information",
    "create_roi_data_generator",
    "get_contract_types",
    "get_service_types",
    "get_subcontracting_levels",
    "get_its_templates_provided",
    "get_its_templates_client_provides",
]
