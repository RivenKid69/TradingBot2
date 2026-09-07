# -*- coding: utf-8 -*-
"""
Due Diligence & Audit Readiness Module.

Provides interfaces for:
- Client audit requests (Art. 30(3)(e))
- Provider information packages for ROI (Art. 28(3))
- Pooled audit coordination (Art. 30(4))
- Compliance status dashboard

DORA Context:
    Financial entities have the right to audit their ICT providers.
    We facilitate this by maintaining audit readiness and providing
    structured information packages.

Modules:
    - audit_readiness.py: Audit support and evidence management
    - provider_info_package.py: ROI data generation for clients
    - pooled_audit_support.py: Multi-client audit coordination
    - compliance_dashboard.py: Real-time compliance status

References:
    - DORA Article 30(3)(d): Financial entity audit rights
    - DORA Article 30(3)(e): NCA access and inspection rights
    - DORA Article 30(4): Pooled audit arrangements
    - CIR 2024/2956: ITS on Register of Information templates

Migration Status: Phase 1 - Complete
"""

from __future__ import annotations

# =============================================================================
# Audit Readiness (Art. 30(3)(e) - Audit and Access Rights)
# =============================================================================

from services.dora_integration.due_diligence.audit_readiness import (
    # SLA Constants
    AUDIT_SLA_ACKNOWLEDGMENT_DAYS,
    AUDIT_SLA_SCHEDULING_DAYS,
    AUDIT_SLA_EVIDENCE_STANDARD_DAYS,
    AUDIT_SLA_EVIDENCE_COMPLEX_DAYS,
    AUDIT_SLA_NCA_RESPONSE_DAYS,
    EVIDENCE_RETENTION_YEARS,
    AUDIT_TYPE_SLAS,
    # Enums
    AuditType,
    AuditScope,
    AuditStatus,
    EvidenceType,
    EvidenceCategory,
    # Data structures
    AuditRequest,
    EvidenceItem,
    AuditFinding,
    EvidenceTemplate,
    AuditReadinessConfig,
    # Main class
    DORAuditReadiness,
    # Factory functions
    create_audit_readiness,
    get_standard_evidence_templates,
    # Multi-Client Incident Coordination (Art. 30(2)(f))
    IncidentNotificationStatus,
    ClientNotificationRecord,
    MultiClientIncident,
    MultiClientIncidentCoordinator,
    create_incident_coordinator,
)

# =============================================================================
# Provider Information Package (Art. 28(3) - ROI Data Generation)
# =============================================================================

from services.dora_integration.due_diligence.provider_info_package import (
    # Enums
    ICTServiceType,
    FunctionCriticality,
    SubstitutabilityLevel,
    DataSensitivity,
    # Data structures - Provider Identification (B_02.01)
    ProviderIdentification,
    # Data structures - Services (B_03.01)
    ServiceDescription,
    # Data structures - Data Locations (B_04.01)
    DataLocation,
    # Data structures - Subcontractors (B_99.01)
    SubcontractorInfo,
    # Data structures - Certifications
    CertificationInfo,
    # Data structures - Contract
    ContractSummary,
    # Data structures - Package
    ProviderInfoPackage,
    # Config
    ProviderInfoConfig,
    # Main class
    DORAProviderInfoPackage,
    # Factory functions
    create_provider_info_package,
)

# =============================================================================
# Pooled Audit Support (Art. 30(4) - Pooled Audits)
# =============================================================================

from services.dora_integration.due_diligence.pooled_audit_support import (
    # Enums
    AuditReportType,
    PooledAuditStatus,
    ParticipationStatus,
    AuditScopeArea,
    FindingSeverity,
    RemediationStatus,
    # Data structures
    CertificationRecord,
    PooledAuditParticipant,
    AuditFinding as PooledAuditFinding,
    PooledAuditEngagement,
    AuditReportAccess,
    PooledAuditConfig,
    # Main class
    PooledAuditSupport,
    # Factory functions
    create_pooled_audit_support,
    get_audit_scope_areas,
    get_report_types,
)

# =============================================================================
# Compliance Dashboard
# =============================================================================

from services.dora_integration.due_diligence.compliance_dashboard import (
    # Enums
    IssueSeverity,
    IssueStatus,
    DeadlineStatus,
    # Data structures
    ComplianceIssue,
    Deadline,
    ComplianceStatus,
    DORAComplianceReport,
    # Main class
    DORAComplianceDashboard,
)

# =============================================================================
# Re-export aliases for backward compatibility
# =============================================================================

# Alias for backward compatibility with provider_info_package naming
ICTServiceDescription = ServiceDescription
DataLocationInfo = DataLocation
ProviderInfoPackageGenerator = DORAProviderInfoPackage

# =============================================================================
# __all__ exports
# =============================================================================

__all__ = [
    # =========================================================================
    # Audit Readiness (Art. 30(3)(e))
    # =========================================================================
    # SLA Constants
    "AUDIT_SLA_ACKNOWLEDGMENT_DAYS",
    "AUDIT_SLA_SCHEDULING_DAYS",
    "AUDIT_SLA_EVIDENCE_STANDARD_DAYS",
    "AUDIT_SLA_EVIDENCE_COMPLEX_DAYS",
    "AUDIT_SLA_NCA_RESPONSE_DAYS",
    "EVIDENCE_RETENTION_YEARS",
    "AUDIT_TYPE_SLAS",
    # Enums
    "AuditType",
    "AuditScope",
    "AuditStatus",
    "EvidenceType",
    "EvidenceCategory",
    # Data structures
    "AuditRequest",
    "EvidenceItem",
    "AuditFinding",
    "EvidenceTemplate",
    "AuditReadinessConfig",
    # Main class
    "DORAuditReadiness",
    # Factory functions
    "create_audit_readiness",
    "get_standard_evidence_templates",
    # Multi-Client Incident Coordination
    "IncidentNotificationStatus",
    "ClientNotificationRecord",
    "MultiClientIncident",
    "MultiClientIncidentCoordinator",
    "create_incident_coordinator",
    # =========================================================================
    # Provider Information Package (Art. 28(3))
    # =========================================================================
    # Enums
    "ICTServiceType",
    "FunctionCriticality",
    "SubstitutabilityLevel",
    "DataSensitivity",
    # Data structures
    "ProviderIdentification",
    "ServiceDescription",
    "ICTServiceDescription",  # Alias
    "DataLocation",
    "DataLocationInfo",  # Alias
    "SubcontractorInfo",
    "CertificationInfo",
    "ContractSummary",
    "ProviderInfoPackage",
    # Config
    "ProviderInfoConfig",
    # Main class
    "DORAProviderInfoPackage",
    "ProviderInfoPackageGenerator",  # Alias
    # Factory functions
    "create_provider_info_package",
    # =========================================================================
    # Pooled Audit Support (Art. 30(4))
    # =========================================================================
    # Enums
    "AuditReportType",
    "PooledAuditStatus",
    "ParticipationStatus",
    "AuditScopeArea",
    "FindingSeverity",
    "RemediationStatus",
    # Data structures
    "CertificationRecord",
    "PooledAuditParticipant",
    "PooledAuditFinding",
    "PooledAuditEngagement",
    "AuditReportAccess",
    "PooledAuditConfig",
    # Main class
    "PooledAuditSupport",
    # Factory functions
    "create_pooled_audit_support",
    "get_audit_scope_areas",
    "get_report_types",
    # =========================================================================
    # Compliance Dashboard
    # =========================================================================
    # Enums
    "IssueSeverity",
    "IssueStatus",
    "DeadlineStatus",
    # Data structures
    "ComplianceIssue",
    "Deadline",
    "ComplianceStatus",
    "DORAComplianceReport",
    # Main class
    "DORAComplianceDashboard",
]
