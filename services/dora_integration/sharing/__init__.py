# -*- coding: utf-8 -*-
"""
Information Sharing Layer (Art. 45).

Provides comprehensive cyber threat information sharing capabilities
	per DORA Article 45, enabling:
	    - Participation in trusted information sharing communities
	    - GDPR- and competition-law-aligned intelligence exchange
	    - Traffic Light Protocol (TLP) enforcement
	    - Sanitization and anonymization of shared data
	    - NCA notification of community participation
	    - Audit trails to support compliance evidence

DORA Context:
    Article 45(1) allows financial entities to exchange cyber threat
    intelligence within trusted communities of financial entities and
    ICT third-party service providers.

    As an ICT provider (Art. 30), we facilitate information sharing by:
    - Providing secure channels for threat intelligence exchange
    - Enforcing sharing policies and TLP controls
    - Supporting client NCA notification requirements
    - Maintaining comprehensive audit trails

Art. 45 Key Requirements:
    - Voluntary participation in trusted communities (Art. 45(1))
    - Protection of confidential business information (Art. 45(2)(a))
    - Protection of personal data (GDPR) (Art. 45(2)(b))
    - Competition law compliance (Art. 45(2)(c))
    - NCA notification of participation (Art. 45(3))
    - Appropriate safeguards for sensitive information (Art. 45(4))

Module Exports:
    - DORAInformationSharing: Main service class
    - SharingCommunity: Community metadata
    - InformationSharingPolicy: Policy controls
    - CyberThreatIntelligence: Threat payload
    - ThreatIntelligenceRecord: Sharing record
    - SharingAuditRecord: Audit trail entry
    - NCANotification: NCA notification record
    - Factory functions for easy instantiation

References:
    - DORA Article 45: https://www.digital-operational-resilience-act.com/Article_45.html
    - ESAs Final Report on Information Sharing Arrangements
    - Traffic Light Protocol (TLP) 2.0 - FIRST
    - STIX 2.1 / TAXII 2.1 standards

Migration Status: Phase 6 - COMPLETE
"""

from __future__ import annotations

# =============================================================================
# Constants
# =============================================================================

from services.dora_integration.sharing.information_sharing import (
    # Shareable information types per Art. 45
    SHAREABLE_INFORMATION_TYPES,
    # TLP definitions
    TLP_DEFINITIONS,
    # Retention defaults
    DEFAULT_INTELLIGENCE_RETENTION_DAYS,
    # NCA notification deadline
    NCA_NOTIFICATION_DEADLINE_DAYS,
)

# =============================================================================
# Enums
# =============================================================================

from services.dora_integration.sharing.information_sharing import (
    # Community classification
    CommunityType,
    # Sharing channels
    SharingChannel,
    # TLP levels
    TLPLevel,
    # Membership states
    MembershipStatus,
    # Sharing outcomes
    SharingOutcome,
    # Intelligence direction
    IntelligenceDirection,
    # Threat severity
    ThreatSeverity,
    # Sanitization levels
    SanitizationLevel,
)

# =============================================================================
# Data Structures
# =============================================================================

from services.dora_integration.sharing.information_sharing import (
    # Community metadata
    SharingCommunity,
    # Policy controls
    InformationSharingPolicy,
    # Threat intelligence payload
    CyberThreatIntelligence,
    # Sharing records
    ThreatIntelligenceRecord,
    # Audit records
    SharingAuditRecord,
    # NCA notifications
    NCANotification,
    # Service configuration
    InformationSharingConfig,
)

# =============================================================================
# Main Service Class
# =============================================================================

from services.dora_integration.sharing.information_sharing import (
    DORAInformationSharing,
)

# =============================================================================
# Factory Functions
# =============================================================================

from services.dora_integration.sharing.information_sharing import (
    # Main factory
    create_information_sharing,
    # Helper factories
    get_shareable_information_types,
    get_tlp_definitions,
    get_community_types,
    get_sharing_channels,
    get_tlp_levels,
    create_sharing_community,
    create_cyber_threat,
    create_sharing_policy,
)

# =============================================================================
# __all__ exports
# =============================================================================

__all__ = [
    # =========================================================================
    # Constants
    # =========================================================================
    "SHAREABLE_INFORMATION_TYPES",
    "TLP_DEFINITIONS",
    "DEFAULT_INTELLIGENCE_RETENTION_DAYS",
    "NCA_NOTIFICATION_DEADLINE_DAYS",

    # =========================================================================
    # Enums
    # =========================================================================
    "CommunityType",
    "SharingChannel",
    "TLPLevel",
    "MembershipStatus",
    "SharingOutcome",
    "IntelligenceDirection",
    "ThreatSeverity",
    "SanitizationLevel",

    # =========================================================================
    # Data Structures
    # =========================================================================
    "SharingCommunity",
    "InformationSharingPolicy",
    "CyberThreatIntelligence",
    "ThreatIntelligenceRecord",
    "SharingAuditRecord",
    "NCANotification",
    "InformationSharingConfig",

    # =========================================================================
    # Main Service Class
    # =========================================================================
    "DORAInformationSharing",

    # =========================================================================
    # Factory Functions
    # =========================================================================
    "create_information_sharing",
    "get_shareable_information_types",
    "get_tlp_definitions",
    "get_community_types",
    "get_sharing_channels",
    "get_tlp_levels",
    "create_sharing_community",
    "create_cyber_threat",
    "create_sharing_policy",
]
