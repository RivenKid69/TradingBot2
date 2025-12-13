# -*- coding: utf-8 -*-
"""
Third-Party Risk Interface - DORA Phase 3 Module.

Provides comprehensive third-party ICT risk management per DORA requirements:
- Concentration risk assessment (Article 29)
- CTPP (Critical Third-Party Provider) oversight (Articles 31-44)
- Third-party risk management (Article 28)
- Third-party incident coordination (Article 23)
- Subcontractor management with client consent (Article 30(2)(b))

DORA Context:
    As an ICT provider, we may have our own subcontractors.
    We must:
    - Disclose our subcontractor chain to clients
    - Obtain prior consent for material changes
    - Monitor our own concentration risk
    - Prepare for potential CTPP designation
    - Manage third-party risk as integral part of ICT risk framework
    - Track and coordinate third-party incidents

Modules:
    - concentration_risk.py: Article 29 concentration risk assessment
    - ctpp_oversight.py: Articles 31-44 CTPP oversight framework
    - third_party_risk.py: Article 28 third-party risk management
    - third_party_incidents.py: Article 23 third-party incident handling
    - subcontractor_management.py: Article 30 consent workflows

Links with Core:
    services/core/subcontractor_monitoring.py provides operational monitoring.
    This module adds DORA-specific compliance layer.

References:
    - DORA Article 23: Operational or security payment-related incidents
    - DORA Article 28: ICT third-party risk
    - DORA Article 29: ICT concentration risk
    - DORA Article 30: Contractual arrangements
    - DORA Articles 31-44: CTPP oversight framework
    - CDR 2024/1773: RTS on CTPP designation criteria
    - CDR 2024/1774: RTS on ICT Risk Management Chapter V
    - CIR 2024/2956: ITS on Register of Information

Migration Status: Phase 3 - Complete
"""

from __future__ import annotations

# =============================================================================
# Concentration Risk (Article 29)
# =============================================================================
from .concentration_risk import (
    # Main class
    DORAConcentrationRisk,
    # Configuration
    ConcentrationRiskConfig,
    # Enums
    ConcentrationType,
    RiskLevel as ConcentrationRiskLevel,
    MitigationStatus,
    AssessmentScope,
    SubstitutabilityLevel,
    # Data classes
    ProviderDependency,
    ConcentrationMetric,
    ConcentrationRisk,
    MitigationMeasure,
    ConcentrationAssessment,
    DependencyMap,
    # Factory functions
    create_concentration_risk,
    get_concentration_types,
    get_substitutability_levels,
)

# =============================================================================
# CTPP Oversight (Articles 31-44)
# =============================================================================
from .ctpp_oversight import (
    # Main class
    DORACtppOversight,
    # Configuration
    CTPPOversightConfig,
    # Enums
    LeadOverseer,
    CTPPStatus,
    OversightRecommendationType,
    RecommendationStatus,
    ComplianceLevel,
    OversightExerciseType,
    # Data classes
    CTPPDesignation,
    OversightRecommendation,
    OversightExercise,
    CTPPRiskAssessment,
    CTPPContractRequirement,
    EntityCTPPRelationship,
    # Constants
    DESIGNATED_CTPPS_2025,
    # Factory functions
    create_ctpp_oversight,
    get_lead_overseers,
    get_designated_ctpps_list,
    get_ctpp_requirements,
    get_ctpp_contract_requirements,
)

# =============================================================================
# Third-Party Risk Management (Article 28)
# =============================================================================
from .third_party_risk import (
    # Main class
    DORAThirdPartyRiskManagement,
    # Configuration
    ThirdPartyRiskConfig,
    # Enums
    ProviderType,
    ProviderCriticality,
    ServiceCriticality,
    ProviderStatus,
    RiskCategory,
    RiskLevel,
    DueDiligenceStatus,
    AssessmentType,
    SubstitutabilityLevel as TPRSubstitutabilityLevel,
    # Data classes
    ICTService,
    ICTProvider,
    ThirdPartyRisk,
    ThirdPartyRiskAssessment,
    DueDiligenceCheck,
    ProviderRelationshipEvent,
    # Factory functions
    create_third_party_risk_management,
    get_provider_types,
    get_risk_categories,
    get_criticality_levels,
)

# =============================================================================
# Third-Party Incidents (Article 23)
# =============================================================================
from .third_party_incidents import (
    # Main class
    DORAThirdPartyIncidents,
    # Enums
    ThirdPartyProviderType,
    ThirdPartyCriticality,
    ThirdPartyIncidentType,
    IncidentSeverity,
    IncidentStatus,
    ContractualSLAStatus,
    EscalationLevel,
    CommunicationChannel,
    # Data classes
    ThirdPartyProvider,
    AffectedService,
    SLAAssessment,
    CommunicationRecord,
    EscalationRecord,
    MitigationAction as IncidentMitigationAction,
    ThirdPartyIncident,
    PostIncidentReview,
    # Factory functions
    create_third_party_incidents,
)

# =============================================================================
# Subcontractor Management (Article 30)
# =============================================================================
from .subcontractor_management import (
    # Main class
    DORASubcontractorManagement,
    # Configuration
    SubcontractorConfig,
    # Enums
    SubcontractorType,
    SubcontractorStatus,
    RiskLevel as SubcontractorRiskLevel,
    ChangeType,
    NotificationStatus,
    ConsentMode,
    # Data classes
    Subcontractor,
    SubcontractorChange,
    ClientSubcontractorPreference,
    SubcontractorRiskAssessment,
    # Factory functions
    create_subcontractor_management,
)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # =========================================================================
    # Concentration Risk (Article 29)
    # =========================================================================
    "DORAConcentrationRisk",
    "ConcentrationRiskConfig",
    "ConcentrationType",
    "ConcentrationRiskLevel",
    "MitigationStatus",
    "AssessmentScope",
    "SubstitutabilityLevel",
    "ProviderDependency",
    "ConcentrationMetric",
    "ConcentrationRisk",
    "MitigationMeasure",
    "ConcentrationAssessment",
    "DependencyMap",
    "create_concentration_risk",
    "get_concentration_types",
    "get_substitutability_levels",
    # =========================================================================
    # CTPP Oversight (Articles 31-44)
    # =========================================================================
    "DORACtppOversight",
    "CTPPOversightConfig",
    "LeadOverseer",
    "CTPPStatus",
    "OversightRecommendationType",
    "RecommendationStatus",
    "ComplianceLevel",
    "OversightExerciseType",
    "CTPPDesignation",
    "OversightRecommendation",
    "OversightExercise",
    "CTPPRiskAssessment",
    "CTPPContractRequirement",
    "EntityCTPPRelationship",
    "DESIGNATED_CTPPS_2025",
    "create_ctpp_oversight",
    "get_lead_overseers",
    "get_designated_ctpps_list",
    "get_ctpp_requirements",
    "get_ctpp_contract_requirements",
    # =========================================================================
    # Third-Party Risk Management (Article 28)
    # =========================================================================
    "DORAThirdPartyRiskManagement",
    "ThirdPartyRiskConfig",
    "ProviderType",
    "ProviderCriticality",
    "ServiceCriticality",
    "ProviderStatus",
    "RiskCategory",
    "RiskLevel",
    "DueDiligenceStatus",
    "AssessmentType",
    "TPRSubstitutabilityLevel",
    "ICTService",
    "ICTProvider",
    "ThirdPartyRisk",
    "ThirdPartyRiskAssessment",
    "DueDiligenceCheck",
    "ProviderRelationshipEvent",
    "create_third_party_risk_management",
    "get_provider_types",
    "get_risk_categories",
    "get_criticality_levels",
    # =========================================================================
    # Third-Party Incidents (Article 23)
    # =========================================================================
    "DORAThirdPartyIncidents",
    "ThirdPartyProviderType",
    "ThirdPartyCriticality",
    "ThirdPartyIncidentType",
    "IncidentSeverity",
    "IncidentStatus",
    "ContractualSLAStatus",
    "EscalationLevel",
    "CommunicationChannel",
    "ThirdPartyProvider",
    "AffectedService",
    "SLAAssessment",
    "CommunicationRecord",
    "EscalationRecord",
    "IncidentMitigationAction",
    "ThirdPartyIncident",
    "PostIncidentReview",
    "create_third_party_incidents",
    # =========================================================================
    # Subcontractor Management (Article 30)
    # =========================================================================
    "DORASubcontractorManagement",
    "SubcontractorConfig",
    "SubcontractorType",
    "SubcontractorStatus",
    "SubcontractorRiskLevel",
    "ChangeType",
    "NotificationStatus",
    "ConsentMode",
    "Subcontractor",
    "SubcontractorChange",
    "ClientSubcontractorPreference",
    "SubcontractorRiskAssessment",
    "create_subcontractor_management",
]
