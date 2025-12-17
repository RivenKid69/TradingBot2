# -*- coding: utf-8 -*-
"""
DORA Article 45 - Information Sharing.

Implements cyber threat information sharing controls per Regulation (EU)
2022/2554 Article 45, including:
    - Participation in trusted information sharing communities
    - GDPR and competition law safeguards
    - Anonymisation and sanitisation of shared intelligence
    - NCA notification of participation in sharing arrangements
    - Traffic Light Protocol (TLP) compliance
    - STIX/TAXII integration support

This module supports ICT Third-Party Providers in facilitating information
sharing between their financial entity clients, in compliance with DORA
requirements.

ICT Provider Context:
    As an ICT provider (Art. 30), we:
    - Facilitate secure information sharing channels for clients
    - Provide threat intelligence feeds from our operational monitoring
    - Coordinate vulnerability disclosures affecting multiple clients
    - Maintain audit trails for all sharing activities
    - Support GDPR and competition law compliance

References:
    - DORA Article 45: Cyber threat information sharing
    - Regulation (EU) 2022/2554, Art. 45(1)-(4)
    - ESAs Final Report on Information Sharing Arrangements
    - Traffic Light Protocol (TLP) 2.0 - FIRST
    - STIX 2.1 / TAXII 2.1 standards
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# Constants per DORA Article 45
# =============================================================================

# Shareable information types per Article 45 guidance
SHAREABLE_INFORMATION_TYPES: Set[str] = {
    "indicators_of_compromise",
    "tactics_techniques_procedures",
    "cybersecurity_alerts",
    "configuration_tools",
    "threat_actor_profiles",
    "vulnerability_disclosures",
    "malware_signatures",
    "network_artifacts",
    "attack_patterns",
    "remediation_guidance",
}

# TLP 2.0 definitions per FIRST standard
TLP_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "TLP:RED": {
        "description": "For named recipients only",
        "sharing_scope": "No further sharing",
        "color_hex": "#FF0033",
    },
    "TLP:AMBER+STRICT": {
        "description": "Limited disclosure, recipients organization only",
        "sharing_scope": "Organization only",
        "color_hex": "#FFC000",
    },
    "TLP:AMBER": {
        "description": "Limited disclosure, recipients organization and clients",
        "sharing_scope": "Organization and clients",
        "color_hex": "#FFC000",
    },
    "TLP:GREEN": {
        "description": "Limited disclosure, community only",
        "sharing_scope": "Community",
        "color_hex": "#33FF00",
    },
    "TLP:CLEAR": {
        "description": "Unlimited disclosure",
        "sharing_scope": "Public",
        "color_hex": "#FFFFFF",
    },
}

# Maximum intelligence retention per data minimisation (GDPR Art. 5(1)(e))
DEFAULT_INTELLIGENCE_RETENTION_DAYS: int = 365

# Art. 45(3) - NCA notification requirement within days of joining
NCA_NOTIFICATION_DEADLINE_DAYS: int = 30


# =============================================================================
# Enums
# =============================================================================

class CommunityType(Enum):
    """
    Trusted community classification per Art. 45(1).

    Financial entities may participate in cyber threat information
    sharing arrangements within trusted communities.
    """
    FS_ISAC = "fs_isac"  # Financial Services - ISAC
    CERT = "cert"  # Computer Emergency Response Team
    CSIRT = "csirt"  # Computer Security Incident Response Team
    PUBLIC_PRIVATE_PARTNERSHIP = "public_private_partnership"
    PRIVATE_EXCHANGE = "private_exchange"
    SECTOR_ISAC = "sector_isac"  # Other sector ISACs
    GOVERNMENT_SHARING = "government_sharing"  # Government sharing programs
    VENDOR_SHARING = "vendor_sharing"  # Vendor-specific programs


class SharingChannel(Enum):
    """
    Permitted sharing channels for threat intelligence.
    """
    API = "api"  # REST/GraphQL API
    EMAIL = "email"  # Encrypted email
    PORTAL = "portal"  # Secure web portal
    SECURE_FTP = "secure_ftp"  # SFTP/FTPS
    STIX_TAXII = "stix_taxii"  # STIX/TAXII feed
    MISP = "misp"  # MISP platform
    WEBHOOK = "webhook"  # Real-time webhook


class TLPLevel(Enum):
    """
    Traffic Light Protocol 2.0 sensitivity levels.

    TLP is a framework for classifying sensitive information
    and controlling its distribution.
    """
    TLP_RED = "tlp_red"  # Named recipients only
    TLP_AMBER_STRICT = "tlp_amber_strict"  # Organization only
    TLP_AMBER = "tlp_amber"  # Organization + clients
    TLP_GREEN = "tlp_green"  # Community
    TLP_CLEAR = "tlp_clear"  # Public


class MembershipStatus(Enum):
    """
    Membership lifecycle states for sharing communities.
    """
    PENDING = "pending"  # Application submitted
    ACTIVE = "active"  # Full membership
    SUSPENDED = "suspended"  # Temporarily suspended
    REVOKED = "revoked"  # Membership terminated
    EXITED = "exited"  # Voluntary exit


class SharingOutcome(Enum):
    """
    Result of a sharing attempt.
    """
    SUCCESS = "success"  # Shared successfully
    SANITISED = "sanitised"  # Shared after sanitisation
    BLOCKED_POLICY = "blocked_policy"  # Blocked by policy
    BLOCKED_TLP = "blocked_tlp"  # Blocked by TLP restrictions
    BLOCKED_GDPR = "blocked_gdpr"  # Blocked for GDPR concerns
    BLOCKED_COMPETITION = "blocked_competition"  # Competition law concern
    FAILED_TECHNICAL = "failed_technical"  # Technical failure


class IntelligenceDirection(Enum):
    """
    Direction of intelligence flow.
    """
    OUTBOUND = "outbound"  # We share to community
    INBOUND = "inbound"  # We receive from community
    BIDIRECTIONAL = "bidirectional"  # Both directions


class ThreatSeverity(Enum):
    """
    Threat severity classification.
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class SanitizationLevel(Enum):
    """
    Level of data sanitization applied.
    """
    NONE = "none"  # No sanitization
    MINIMAL = "minimal"  # Keywords only
    MODERATE = "moderate"  # Keywords + IPs + domains
    AGGRESSIVE = "aggressive"  # Full anonymization
    CUSTOM = "custom"  # Custom rules applied


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SharingCommunity:
    """
    Information sharing community metadata per Art. 45(1).

    Attributes:
        name: Community name
        community_type: Type of sharing community
        country: Country/jurisdiction
        contact_email: Primary contact
        community_id: Unique identifier
        trust_level: Trust score (0-100)
        requires_anonymization: Whether to always anonymize
        allowed_channels: Permitted sharing channels
        membership_status: Current membership status
        joined_at: Date joined
        nca_notified: Whether NCA was notified of participation
        nca_notification_date: Date of NCA notification
        data_protection_agreement: DPA in place
        notes: Additional notes
    """
    name: str
    community_type: CommunityType
    country: str
    contact_email: str
    community_id: str = ""
    trust_level: int = 50
    requires_anonymization: bool = True
    allowed_channels: List[SharingChannel] = field(default_factory=lambda: [SharingChannel.PORTAL])
    membership_status: MembershipStatus = MembershipStatus.PENDING
    joined_at: Optional[datetime] = None
    nca_notified: bool = False
    nca_notification_date: Optional[datetime] = None
    data_protection_agreement: bool = False
    compliance_certifications: List[str] = field(default_factory=list)
    direction: IntelligenceDirection = IntelligenceDirection.BIDIRECTIONAL
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.community_id:
            self.community_id = f"COMM-{uuid4().hex[:10].upper()}"
        self.trust_level = max(0, min(100, self.trust_level))
        if self.joined_at and self.membership_status == MembershipStatus.PENDING:
            self.membership_status = MembershipStatus.ACTIVE

    def is_active(self) -> bool:
        """Check if membership is active."""
        return self.membership_status == MembershipStatus.ACTIVE

    def requires_nca_notification(self) -> bool:
        """Check if NCA notification is required per Art. 45(3)."""
        return self.is_active() and not self.nca_notified

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "community_type": self.community_type.value,
            "country": self.country,
            "contact_email": self.contact_email,
            "community_id": self.community_id,
            "trust_level": self.trust_level,
            "requires_anonymization": self.requires_anonymization,
            "allowed_channels": [ch.value for ch in self.allowed_channels],
            "membership_status": self.membership_status.value,
            "joined_at": self.joined_at.isoformat() if self.joined_at else None,
            "nca_notified": self.nca_notified,
            "nca_notification_date": self.nca_notification_date.isoformat() if self.nca_notification_date else None,
            "data_protection_agreement": self.data_protection_agreement,
            "compliance_certifications": self.compliance_certifications,
            "direction": self.direction.value,
            "notes": self.notes,
        }


@dataclass
class InformationSharingPolicy:
    """
    Policy controls applied to outgoing intelligence per Art. 45(2).

    Article 45(2) requirements:
    - Protect confidential business information
    - Protect personal data (GDPR)
    - Respect competition law boundaries
    - Maintain adequate cybersecurity safeguards

    Attributes:
        policy_id: Unique policy identifier
        allowed_information_types: Types of information that can be shared
        restricted_keywords: Keywords to redact
        require_gdpr_review: Require GDPR compliance check
        require_competition_review: Require competition law check
        allowed_tlp_levels: Allowed TLP sensitivity levels
        default_tlp: Default TLP level for outgoing
        default_channel: Default sharing channel
        sanitization_level: Default sanitization level
        auto_sanitize: Automatically sanitize before sharing
        pii_patterns: Regex patterns for PII detection
    """
    policy_id: str = ""
    allowed_information_types: Set[str] = field(default_factory=lambda: set(SHAREABLE_INFORMATION_TYPES))
    restricted_keywords: Set[str] = field(default_factory=lambda: {
        "client_name", "client_id", "account_number", "pricing",
        "trade_id", "transaction_id", "internal_ip", "employee_name",
        "salary", "revenue", "profit", "ssn", "passport"
    })
    require_gdpr_review: bool = True
    require_competition_review: bool = True
    allowed_tlp_levels: Set[TLPLevel] = field(default_factory=lambda: {
        TLPLevel.TLP_AMBER, TLPLevel.TLP_GREEN, TLPLevel.TLP_CLEAR
    })
    default_tlp: TLPLevel = TLPLevel.TLP_AMBER
    default_channel: SharingChannel = SharingChannel.PORTAL
    sanitization_level: SanitizationLevel = SanitizationLevel.MODERATE
    auto_sanitize: bool = True
    pii_patterns: List[str] = field(default_factory=lambda: [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b",  # SSN
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b",  # Credit card
        r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",  # IPv4 (internal)
    ])
    retention_days: int = DEFAULT_INTELLIGENCE_RETENTION_DAYS
    version: str = "1.0"
    effective_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.policy_id:
            self.policy_id = f"POL-{uuid4().hex[:8].upper()}"

    def is_shareable(self, threat: "CyberThreatIntelligence") -> Tuple[bool, List[str]]:
        """
        Check if threat is shareable under policy per Art. 45(2).

        Args:
            threat: The threat intelligence to evaluate

        Returns:
            Tuple of (is_shareable, list_of_reasons_if_blocked)
        """
        reasons: List[str] = []

        # Check information type
        if not threat.information_types.issubset(self.allowed_information_types):
            disallowed = threat.information_types - self.allowed_information_types
            reasons.append(f"information_type_not_allowed:{','.join(disallowed)}")

        # Check TLP level
        if threat.tlp_level not in self.allowed_tlp_levels:
            reasons.append(f"tlp_level_not_allowed:{threat.tlp_level.value}")

        # Check restricted keywords
        text_to_check = f"{threat.title} {threat.description}"
        for keyword in self.restricted_keywords:
            if keyword.lower() in text_to_check.lower():
                reasons.append(f"restricted_keyword:{keyword}")

        # Check personal data (GDPR)
        if threat.contains_personal_data and self.require_gdpr_review:
            reasons.append("gdpr_review_required")

        # Check client/commercial data (competition law)
        if threat.contains_client_data and self.require_competition_review:
            reasons.append("competition_review_required")

        # Check PII patterns
        for pattern in self.pii_patterns:
            if re.search(pattern, text_to_check, re.IGNORECASE):
                reasons.append("pii_pattern_detected")
                break

        return len(reasons) == 0, reasons

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "policy_id": self.policy_id,
            "allowed_information_types": list(self.allowed_information_types),
            "restricted_keywords": list(self.restricted_keywords),
            "require_gdpr_review": self.require_gdpr_review,
            "require_competition_review": self.require_competition_review,
            "allowed_tlp_levels": [t.value for t in self.allowed_tlp_levels],
            "default_tlp": self.default_tlp.value,
            "default_channel": self.default_channel.value,
            "sanitization_level": self.sanitization_level.value,
            "auto_sanitize": self.auto_sanitize,
            "retention_days": self.retention_days,
            "version": self.version,
            "effective_date": self.effective_date.isoformat(),
        }


@dataclass
class CyberThreatIntelligence:
    """
    Cyber threat intelligence payload per Art. 45(1).

    Represents shareable threat information including indicators,
    TTPs, and associated metadata.

    Attributes:
        title: Threat title/name
        description: Detailed description
        information_types: Types of information contained
        indicators_of_compromise: List of IOCs
        ttps: MITRE ATT&CK TTPs
        severity: Threat severity
        tlp_level: TLP classification
        contains_personal_data: GDPR flag
        contains_client_data: Competition law flag
        source: Intelligence source
        confidence: Confidence score (0-100)
        threat_id: Unique identifier
        created_at: Creation timestamp
        expires_at: Expiration timestamp
        related_threats: Related threat IDs
        tags: Classification tags
    """
    title: str
    description: str
    information_types: Set[str] = field(default_factory=set)
    indicators_of_compromise: List[str] = field(default_factory=list)
    ttps: List[str] = field(default_factory=list)
    severity: ThreatSeverity = ThreatSeverity.MEDIUM
    tlp_level: TLPLevel = TLPLevel.TLP_AMBER
    contains_personal_data: bool = False
    contains_client_data: bool = False
    source: str = "internal_detection"
    confidence: int = 75
    threat_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    related_threats: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    stix_id: Optional[str] = None
    mitre_attack_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.threat_id:
            self.threat_id = f"THR-{uuid4().hex[:8].upper()}"
        if not self.information_types:
            self.information_types = {"indicators_of_compromise"}
        self.confidence = max(0, min(100, self.confidence))
        if not self.expires_at:
            self.expires_at = self.created_at + timedelta(days=DEFAULT_INTELLIGENCE_RETENTION_DAYS)

    def is_expired(self) -> bool:
        """Check if intelligence has expired."""
        if self.expires_at:
            return datetime.now(timezone.utc) > self.expires_at
        return False

    def get_hash(self) -> str:
        """Generate content hash for deduplication."""
        content = f"{self.title}:{self.description}:{sorted(self.indicators_of_compromise)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "threat_id": self.threat_id,
            "title": self.title,
            "description": self.description,
            "information_types": list(self.information_types),
            "indicators_of_compromise": self.indicators_of_compromise,
            "ttps": self.ttps,
            "severity": self.severity.value,
            "tlp_level": self.tlp_level.value,
            "contains_personal_data": self.contains_personal_data,
            "contains_client_data": self.contains_client_data,
            "source": self.source,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "related_threats": self.related_threats,
            "tags": self.tags,
            "stix_id": self.stix_id,
            "mitre_attack_ids": self.mitre_attack_ids,
        }


@dataclass
class ThreatIntelligenceRecord:
    """
    Shared or received intelligence record.

    Tracks the sharing event with full audit trail.
    """
    threat: CyberThreatIntelligence
    community_id: str
    channel: SharingChannel
    sanitized: bool
    outcome: SharingOutcome
    direction: IntelligenceDirection
    record_id: str = ""
    shared_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    policy_checks: List[str] = field(default_factory=list)
    sanitization_applied: SanitizationLevel = SanitizationLevel.NONE
    original_threat_hash: Optional[str] = None
    acknowledgment_received: bool = False
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.record_id:
            self.record_id = f"INTEL-{uuid4().hex[:10].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "record_id": self.record_id,
            "threat": self.threat.to_dict(),
            "community_id": self.community_id,
            "channel": self.channel.value,
            "sanitized": self.sanitized,
            "outcome": self.outcome.value,
            "direction": self.direction.value,
            "shared_at": self.shared_at.isoformat(),
            "policy_checks": self.policy_checks,
            "sanitization_applied": self.sanitization_applied.value,
            "original_threat_hash": self.original_threat_hash,
            "acknowledgment_received": self.acknowledgment_received,
            "error_message": self.error_message,
        }


@dataclass
class SharingAuditRecord:
    """
    Audit entry for sharing events per Art. 45(4).

    Article 45(4) requires maintaining appropriate records
    of sharing activities.
    """
    record_id: str
    threat_id: str
    community_id: str
    direction: IntelligenceDirection
    outcome: SharingOutcome
    channel: SharingChannel
    sanitized: bool
    sanitization_level: SanitizationLevel
    policy_checks: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    operator_id: Optional[str] = None
    approval_id: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.record_id:
            self.record_id = f"AUDIT-{uuid4().hex[:10].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "record_id": self.record_id,
            "threat_id": self.threat_id,
            "community_id": self.community_id,
            "direction": self.direction.value,
            "outcome": self.outcome.value,
            "channel": self.channel.value,
            "sanitized": self.sanitized,
            "sanitization_level": self.sanitization_level.value,
            "policy_checks": self.policy_checks,
            "created_at": self.created_at.isoformat(),
            "operator_id": self.operator_id,
            "approval_id": self.approval_id,
            "error": self.error,
        }


@dataclass
class NCANotification:
    """
    NCA notification record per Art. 45(3).

    Financial entities must inform their NCA about participation
    in information sharing arrangements.
    """
    notification_id: str = ""
    community_id: str = ""
    community_name: str = ""
    community_type: CommunityType = CommunityType.PRIVATE_EXCHANGE
    country: str = ""
    participation_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notification_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    nca_reference: str = ""
    dpo_contact: str = ""
    acknowledgment_received: bool = False
    acknowledgment_date: Optional[datetime] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.notification_id:
            self.notification_id = f"NCA-{uuid4().hex[:8].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "notification_id": self.notification_id,
            "community_id": self.community_id,
            "community_name": self.community_name,
            "community_type": self.community_type.value,
            "country": self.country,
            "participation_date": self.participation_date.isoformat(),
            "notification_date": self.notification_date.isoformat(),
            "nca_reference": self.nca_reference,
            "dpo_contact": self.dpo_contact,
            "acknowledgment_received": self.acknowledgment_received,
            "acknowledgment_date": self.acknowledgment_date.isoformat() if self.acknowledgment_date else None,
            "notes": self.notes,
        }


@dataclass
class InformationSharingConfig:
    """
    Configuration for the Information Sharing service.
    """
    provider_name: str = "AI Research Platform"
    provider_lei: str = ""
    gdpr_officer_email: str = "dpo@platform.example"
    nca_contact_email: str = "nca@supervisor.example"
    default_policy: Optional[InformationSharingPolicy] = None
    enable_auto_sanitization: bool = True
    enable_stix_export: bool = True
    enable_misp_integration: bool = False
    intelligence_retention_days: int = DEFAULT_INTELLIGENCE_RETENTION_DAYS
    require_dpa_for_sharing: bool = True
    log_all_sharing_attempts: bool = True


# =============================================================================
# Main Service Class
# =============================================================================

class DORAInformationSharing:
    """
    DORA Article 45 Information Sharing Service.

    Provides comprehensive threat intelligence sharing capabilities
    with full DORA compliance, including:
    - Community management and membership tracking
    - Policy-based sharing controls
    - GDPR and competition law safeguards
    - TLP enforcement
    - Sanitization and anonymization
    - NCA notification support
    - Full audit trails

    ICT Provider Perspective:
        As an ICT provider serving financial entities, we facilitate
        information sharing by:
        1. Providing secure channels for intelligence exchange
        2. Enforcing sharing policies and TLP controls
        3. Maintaining audit trails for compliance
        4. Supporting client NCA notification requirements

    Example:
        >>> config = InformationSharingConfig(
        ...     provider_name="My Platform",
        ...     gdpr_officer_email="dpo@myplatform.com"
        ... )
        >>> sharing = DORAInformationSharing(config=config)
        >>> community = SharingCommunity(
        ...     name="FS-ISAC",
        ...     community_type=CommunityType.FS_ISAC,
        ...     country="US",
        ...     contact_email="contact@fs-isac.com"
        ... )
        >>> sharing.join_community(community)
        >>> threat = CyberThreatIntelligence(
        ...     title="Phishing Campaign",
        ...     description="Targeting financial institutions"
        ... )
        >>> result = sharing.share_threat_intelligence(threat, community)
    """

    def __init__(
        self,
        config: Optional[InformationSharingConfig] = None,
        policy: Optional[InformationSharingPolicy] = None,
    ) -> None:
        """
        Initialize the Information Sharing service.

        Args:
            config: Service configuration
            policy: Default sharing policy
        """
        self.config = config or InformationSharingConfig()
        self.policy = policy or self.config.default_policy or InformationSharingPolicy()

        # Storage
        self._communities: Dict[str, SharingCommunity] = {}
        self._audit_records: List[SharingAuditRecord] = []
        self._intelligence_records: Dict[str, ThreatIntelligenceRecord] = {}
        self._received_intelligence: Dict[str, CyberThreatIntelligence] = {}
        self._nca_notifications: Dict[str, NCANotification] = {}
        self._threat_hashes: Set[str] = set()  # Deduplication

        logger.info("DORAInformationSharing initialized with policy %s", self.policy.policy_id)

    # =========================================================================
    # Community Management
    # =========================================================================

    def register_community(self, community: SharingCommunity) -> SharingCommunity:
        """
        Register a sharing community.

        Args:
            community: Community to register

        Returns:
            Registered community with ID
        """
        self._communities[community.community_id] = community
        logger.info("Registered community %s (%s)", community.name, community.community_id)
        return community

    def join_community(
        self,
        community: SharingCommunity,
        notify_nca: bool = True,
    ) -> SharingCommunity:
        """
        Join a sharing community per Art. 45(1).

        Participation in information sharing arrangements is voluntary
        but must be notified to the relevant NCA per Art. 45(3).

        Args:
            community: Community to join
            notify_nca: Whether to prepare NCA notification

        Returns:
            Updated community record
        """
        community.membership_status = MembershipStatus.ACTIVE
        community.joined_at = datetime.now(timezone.utc)
        self.register_community(community)

        if notify_nca and community.requires_nca_notification():
            self._prepare_nca_notification(community)

        logger.info("Joined community %s", community.name)
        return community

    def exit_community(
        self,
        community_id: str,
        reason: str = "voluntary_exit",
    ) -> Optional[SharingCommunity]:
        """
        Exit a sharing community.

        Args:
            community_id: ID of community to exit
            reason: Exit reason

        Returns:
            Updated community or None if not found
        """
        community = self._communities.get(community_id)
        if community:
            community.membership_status = MembershipStatus.EXITED
            community.notes = f"{community.notes} Exit reason: {reason}"
            logger.info("Exited community %s: %s", community.name, reason)
        return community

    def get_community(self, community_id: str) -> Optional[SharingCommunity]:
        """Get community by ID."""
        return self._communities.get(community_id)

    def list_communities(
        self,
        status: Optional[MembershipStatus] = None,
    ) -> List[SharingCommunity]:
        """
        List all registered communities.

        Args:
            status: Filter by membership status

        Returns:
            List of communities
        """
        communities = list(self._communities.values())
        if status:
            communities = [c for c in communities if c.membership_status == status]
        return communities

    def get_active_communities(self) -> List[SharingCommunity]:
        """Get all active communities."""
        return self.list_communities(status=MembershipStatus.ACTIVE)

    # =========================================================================
    # Sharing Workflow
    # =========================================================================

    def share_threat_intelligence(
        self,
        threat: CyberThreatIntelligence,
        community: SharingCommunity,
        channel: Optional[SharingChannel] = None,
        force_sanitize: bool = False,
        operator_id: Optional[str] = None,
    ) -> ThreatIntelligenceRecord:
        """
        Share threat intelligence with a community per Art. 45.

        Implements full policy checking, sanitization, and audit logging.

        Args:
            threat: Threat intelligence to share
            community: Target community
            channel: Override default channel
            force_sanitize: Force sanitization regardless of policy
            operator_id: ID of operator initiating share

        Returns:
            Intelligence record with outcome

        Raises:
            ValueError: If community not joined or channel not allowed
        """
        # Validate community membership
        self._validate_community_membership(community)

        # Determine channel
        chosen_channel = channel or self.policy.default_channel
        self._validate_channel(community, chosen_channel)

        # Store original hash for audit
        original_hash = threat.get_hash()

        # Policy check
        is_shareable, policy_reasons = self.policy.is_shareable(threat)

        sanitized = False
        sanitization_level = SanitizationLevel.NONE
        outcome = SharingOutcome.SUCCESS
        error_message: Optional[str] = None
        threat_to_share = threat

        if not is_shareable:
            # Check if we can sanitize and proceed
            if self._can_sanitize_and_share(policy_reasons):
                threat_to_share = self.sanitize_threat(threat)
                sanitized = True
                sanitization_level = self.policy.sanitization_level
                outcome = SharingOutcome.SANITISED
                logger.info("Threat %s sanitized before sharing", threat.threat_id)
            else:
                # Block sharing
                outcome = self._determine_block_outcome(policy_reasons)
                error_message = f"Sharing blocked: {', '.join(policy_reasons)}"
                logger.warning("Sharing blocked for %s: %s", threat.threat_id, policy_reasons)

                # Create audit record for blocked attempt
                audit = SharingAuditRecord(
                    record_id="",
                    threat_id=threat.threat_id,
                    community_id=community.community_id,
                    direction=IntelligenceDirection.OUTBOUND,
                    outcome=outcome,
                    channel=chosen_channel,
                    sanitized=False,
                    sanitization_level=SanitizationLevel.NONE,
                    policy_checks=policy_reasons,
                    operator_id=operator_id,
                    error=error_message,
                )
                self._audit_records.append(audit)

                # Return blocked record
                return ThreatIntelligenceRecord(
                    threat=threat,
                    community_id=community.community_id,
                    channel=chosen_channel,
                    sanitized=False,
                    outcome=outcome,
                    direction=IntelligenceDirection.OUTBOUND,
                    policy_checks=policy_reasons,
                    original_threat_hash=original_hash,
                    error_message=error_message,
                )

        # Apply community-required anonymization
        if community.requires_anonymization and not sanitized:
            threat_to_share = self.sanitize_threat(threat_to_share)
            sanitized = True
            sanitization_level = self.policy.sanitization_level
            if outcome == SharingOutcome.SUCCESS:
                outcome = SharingOutcome.SANITISED

        # Force sanitize if requested
        if force_sanitize and not sanitized:
            threat_to_share = self.sanitize_threat(threat_to_share)
            sanitized = True
            sanitization_level = self.policy.sanitization_level

        # Create intelligence record
        intel_record = ThreatIntelligenceRecord(
            threat=threat_to_share,
            community_id=community.community_id,
            channel=chosen_channel,
            sanitized=sanitized,
            outcome=outcome,
            direction=IntelligenceDirection.OUTBOUND,
            policy_checks=policy_reasons,
            sanitization_applied=sanitization_level,
            original_threat_hash=original_hash,
        )

        # Store record
        self._intelligence_records[intel_record.record_id] = intel_record

        # Create audit record
        audit = SharingAuditRecord(
            record_id="",
            threat_id=threat_to_share.threat_id,
            community_id=community.community_id,
            direction=IntelligenceDirection.OUTBOUND,
            outcome=outcome,
            channel=chosen_channel,
            sanitized=sanitized,
            sanitization_level=sanitization_level,
            policy_checks=policy_reasons,
            operator_id=operator_id,
        )
        self._audit_records.append(audit)

        logger.info(
            "Shared threat %s with %s via %s (outcome: %s)",
            threat_to_share.threat_id,
            community.name,
            chosen_channel.value,
            outcome.value,
        )

        return intel_record

    def receive_threat_intelligence(
        self,
        threat: CyberThreatIntelligence,
        community_id: str,
        channel: SharingChannel = SharingChannel.PORTAL,
    ) -> bool:
        """
        Receive and process threat intelligence from a community.

        Implements deduplication and validation.

        Args:
            threat: Received threat intelligence
            community_id: Source community ID
            channel: Receiving channel

        Returns:
            True if successfully processed, False if duplicate
        """
        # Check for duplicates
        threat_hash = threat.get_hash()
        if threat_hash in self._threat_hashes:
            logger.debug("Duplicate threat %s ignored", threat.threat_id)
            return False

        # Validate community exists
        community = self._communities.get(community_id)
        if not community:
            logger.warning("Received threat from unknown community %s", community_id)

        # Store threat
        self._received_intelligence[threat.threat_id] = threat
        self._threat_hashes.add(threat_hash)

        # Create audit record
        audit = SharingAuditRecord(
            record_id="",
            threat_id=threat.threat_id,
            community_id=community_id,
            direction=IntelligenceDirection.INBOUND,
            outcome=SharingOutcome.SUCCESS,
            channel=channel,
            sanitized=False,
            sanitization_level=SanitizationLevel.NONE,
        )
        self._audit_records.append(audit)

        logger.info("Received threat %s from community %s", threat.threat_id, community_id)
        return True

    # =========================================================================
    # Sanitization
    # =========================================================================

    def sanitize_threat(
        self,
        threat: CyberThreatIntelligence,
        level: Optional[SanitizationLevel] = None,
    ) -> CyberThreatIntelligence:
        """
        Create a sanitized copy of threat intelligence per Art. 45(2).

        Removes or redacts sensitive information to protect:
        - Personal data (GDPR)
        - Confidential business information
        - Client identifying information

        Args:
            threat: Original threat intelligence
            level: Sanitization level (defaults to policy setting)

        Returns:
            Sanitized copy of threat intelligence
        """
        sanitization_level = level or self.policy.sanitization_level

        # Sanitize text fields
        sanitized_description = self._sanitize_text(threat.description, sanitization_level)
        sanitized_iocs = [
            self._sanitize_text(ioc, sanitization_level)
            for ioc in threat.indicators_of_compromise
        ]

        # Create sanitized copy
        return CyberThreatIntelligence(
            title=threat.title,
            description=sanitized_description,
            information_types=set(threat.information_types),
            indicators_of_compromise=sanitized_iocs,
            ttps=list(threat.ttps),
            severity=threat.severity,
            tlp_level=threat.tlp_level,
            contains_personal_data=False,  # Marked as cleaned
            contains_client_data=False,
            source=threat.source,
            confidence=threat.confidence,
            threat_id=threat.threat_id,
            created_at=threat.created_at,
            expires_at=threat.expires_at,
            related_threats=list(threat.related_threats),
            tags=list(threat.tags),
            stix_id=threat.stix_id,
            mitre_attack_ids=list(threat.mitre_attack_ids),
        )

    def _sanitize_text(self, text: str, level: SanitizationLevel) -> str:
        """Apply sanitization rules to text."""
        result = text

        # Always redact restricted keywords
        for keyword in self.policy.restricted_keywords:
            result = re.sub(
                rf"\b{re.escape(keyword)}\b",
                "[REDACTED]",
                result,
                flags=re.IGNORECASE,
            )

        if level in {SanitizationLevel.MODERATE, SanitizationLevel.AGGRESSIVE}:
            # Redact PII patterns
            for pattern in self.policy.pii_patterns:
                result = re.sub(pattern, "[REDACTED]", result, flags=re.IGNORECASE)

        if level == SanitizationLevel.AGGRESSIVE:
            # Additional aggressive sanitization
            # Redact potential internal hostnames
            result = re.sub(r"\b[a-z0-9-]+\.(internal|local|corp)\b", "[REDACTED]", result, flags=re.IGNORECASE)
            # Redact file paths
            result = re.sub(r"[A-Za-z]:\\[\w\\]+", "[REDACTED]", result)
            result = re.sub(r"/[\w/]+", "[REDACTED]", result)

        return result

    # =========================================================================
    # NCA Notification per Art. 45(3)
    # =========================================================================

    def _prepare_nca_notification(self, community: SharingCommunity) -> NCANotification:
        """Prepare NCA notification for community participation."""
        notification = NCANotification(
            community_id=community.community_id,
            community_name=community.name,
            community_type=community.community_type,
            country=community.country,
            participation_date=community.joined_at or datetime.now(timezone.utc),
            dpo_contact=self.config.gdpr_officer_email,
        )
        self._nca_notifications[notification.notification_id] = notification
        return notification

    def notify_nca_of_participation(
        self,
        community: SharingCommunity,
    ) -> NCANotification:
        """
        Prepare and return NCA notification payload per Art. 45(3).

        Financial entities must inform their NCA about participation
        in information sharing arrangements.

        Args:
            community: Community to notify about

        Returns:
            NCA notification record
        """
        # Check if notification already exists
        for notification in self._nca_notifications.values():
            if notification.community_id == community.community_id:
                return notification

        notification = self._prepare_nca_notification(community)

        # Mark community as notified
        community.nca_notified = True
        community.nca_notification_date = datetime.now(timezone.utc)

        logger.info(
            "NCA notification prepared for community %s (%s)",
            community.name,
            notification.notification_id,
        )
        return notification

    def get_pending_nca_notifications(self) -> List[NCANotification]:
        """Get all NCA notifications pending acknowledgment."""
        return [
            n for n in self._nca_notifications.values()
            if not n.acknowledgment_received
        ]

    def acknowledge_nca_notification(
        self,
        notification_id: str,
        nca_reference: str,
    ) -> Optional[NCANotification]:
        """
        Record NCA acknowledgment of notification.

        Args:
            notification_id: Notification ID
            nca_reference: NCA reference number

        Returns:
            Updated notification or None if not found
        """
        notification = self._nca_notifications.get(notification_id)
        if notification:
            notification.acknowledgment_received = True
            notification.acknowledgment_date = datetime.now(timezone.utc)
            notification.nca_reference = nca_reference
            logger.info("NCA acknowledgment received: %s", nca_reference)
        return notification

    # =========================================================================
    # Audit & Compliance
    # =========================================================================

    def get_sharing_audit_log(
        self,
        community_id: Optional[str] = None,
        direction: Optional[IntelligenceDirection] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[SharingAuditRecord]:
        """
        Get audit log for sharing events per Art. 45(4).

        Args:
            community_id: Filter by community
            direction: Filter by direction
            start_date: Filter from date
            end_date: Filter to date

        Returns:
            Filtered list of audit records
        """
        records = list(self._audit_records)

        if community_id:
            records = [r for r in records if r.community_id == community_id]

        if direction:
            records = [r for r in records if r.direction == direction]

        if start_date:
            records = [r for r in records if r.created_at >= start_date]

        if end_date:
            records = [r for r in records if r.created_at <= end_date]

        return records

    def get_sharing_statistics(self) -> Dict[str, Any]:
        """
        Get sharing statistics for compliance reporting.

        Returns:
            Dictionary with sharing statistics
        """
        outbound = [r for r in self._audit_records if r.direction == IntelligenceDirection.OUTBOUND]
        inbound = [r for r in self._audit_records if r.direction == IntelligenceDirection.INBOUND]

        successful_out = [r for r in outbound if r.outcome in {SharingOutcome.SUCCESS, SharingOutcome.SANITISED}]
        blocked_out = [r for r in outbound if r.outcome not in {SharingOutcome.SUCCESS, SharingOutcome.SANITISED}]

        return {
            "total_sharing_events": len(self._audit_records),
            "outbound_shares": len(outbound),
            "outbound_successful": len(successful_out),
            "outbound_blocked": len(blocked_out),
            "outbound_sanitized": len([r for r in outbound if r.sanitized]),
            "inbound_received": len(inbound),
            "active_communities": len(self.get_active_communities()),
            "total_communities": len(self._communities),
            "received_intelligence_count": len(self._received_intelligence),
            "pending_nca_notifications": len(self.get_pending_nca_notifications()),
        }

    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate Art. 45 compliance report.

        Returns:
            Comprehensive compliance report
        """
        stats = self.get_sharing_statistics()
        communities = self.list_communities()

        return {
            "report_id": f"ART45-{uuid4().hex[:8].upper()}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "policy_version": self.policy.version,
            "statistics": stats,
            "communities": [c.to_dict() for c in communities],
            "nca_notifications": [n.to_dict() for n in self._nca_notifications.values()],
            "compliance_status": {
                "all_communities_nca_notified": all(
                    c.nca_notified for c in communities if c.is_active()
                ),
                "dpa_in_place": all(
                    c.data_protection_agreement for c in communities if c.is_active()
                ),
                "policy_enforced": True,
            },
        }

    # =========================================================================
    # Data Retention per GDPR Art. 5(1)(e)
    # =========================================================================

    def purge_stale_intelligence(
        self,
        max_age_days: Optional[int] = None,
    ) -> int:
        """
        Remove intelligence older than retention period per GDPR.

        Args:
            max_age_days: Override default retention period

        Returns:
            Number of records purged
        """
        retention_days = max_age_days or self.policy.retention_days
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

        # Purge received intelligence
        keys_to_remove = [
            key for key, intel in self._received_intelligence.items()
            if intel.created_at < cutoff
        ]
        for key in keys_to_remove:
            del self._received_intelligence[key]

        # Purge intelligence records
        records_to_remove = [
            key for key, record in self._intelligence_records.items()
            if record.shared_at < cutoff
        ]
        for key in records_to_remove:
            del self._intelligence_records[key]

        total_purged = len(keys_to_remove) + len(records_to_remove)
        if total_purged > 0:
            logger.info("Purged %d stale intelligence records", total_purged)

        return total_purged

    # =========================================================================
    # Export Functions
    # =========================================================================

    def export_to_stix(self, threat: CyberThreatIntelligence) -> Dict[str, Any]:
        """
        Export threat intelligence in STIX 2.1 format.

        Args:
            threat: Threat to export

        Returns:
            STIX 2.1 bundle dictionary
        """
        stix_indicator = {
            "type": "indicator",
            "spec_version": "2.1",
            "id": threat.stix_id or f"indicator--{uuid4()}",
            "created": threat.created_at.isoformat(),
            "modified": datetime.now(timezone.utc).isoformat(),
            "name": threat.title,
            "description": threat.description,
            "indicator_types": list(threat.information_types),
            "pattern_type": "stix",
            "pattern": self._generate_stix_pattern(threat),
            "valid_from": threat.created_at.isoformat(),
            "valid_until": threat.expires_at.isoformat() if threat.expires_at else None,
            "confidence": threat.confidence,
            "labels": threat.tags,
        }

        if threat.tlp_level:
            stix_indicator["object_marking_refs"] = [
                self._get_tlp_marking_ref(threat.tlp_level)
            ]

        return {
            "type": "bundle",
            "id": f"bundle--{uuid4()}",
            "objects": [stix_indicator],
        }

    def _generate_stix_pattern(self, threat: CyberThreatIntelligence) -> str:
        """Generate STIX pattern from IOCs."""
        patterns = []
        for ioc in threat.indicators_of_compromise[:5]:  # Limit patterns
            if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", ioc):
                patterns.append(f"[ipv4-addr:value = '{ioc}']")
            elif re.match(r"^[a-f0-9]{32}$", ioc, re.IGNORECASE):
                patterns.append(f"[file:hashes.MD5 = '{ioc}']")
            elif re.match(r"^[a-f0-9]{64}$", ioc, re.IGNORECASE):
                patterns.append(f"[file:hashes.'SHA-256' = '{ioc}']")
            else:
                patterns.append(f"[domain-name:value = '{ioc}']")

        return " OR ".join(patterns) if patterns else "[ipv4-addr:value = '0.0.0.0']"

    def _get_tlp_marking_ref(self, tlp: TLPLevel) -> str:
        """Get STIX TLP marking definition reference."""
        tlp_refs = {
            TLPLevel.TLP_RED: "marking-definition--5e57c739-391a-4eb3-b6be-7d15ca92d5ed",
            TLPLevel.TLP_AMBER_STRICT: "marking-definition--826578e1-40ad-459f-bc73-ede076f81f37",
            TLPLevel.TLP_AMBER: "marking-definition--f88d31f6-486f-44da-b317-01333bde0b82",
            TLPLevel.TLP_GREEN: "marking-definition--34098fce-860f-48ae-8e50-ebd3cc5e41da",
            TLPLevel.TLP_CLEAR: "marking-definition--94868c89-83c2-464b-929b-a1a8aa3c8487",
        }
        return tlp_refs.get(tlp, tlp_refs[TLPLevel.TLP_CLEAR])

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _validate_community_membership(self, community: SharingCommunity) -> None:
        """Validate community is joined."""
        if community.membership_status != MembershipStatus.ACTIVE:
            raise ValueError(f"Community {community.name} membership not active")

    def _validate_channel(self, community: SharingCommunity, channel: SharingChannel) -> None:
        """Validate channel is allowed for community."""
        if channel not in community.allowed_channels:
            raise ValueError(
                f"Channel {channel.value} not permitted for {community.name}. "
                f"Allowed: {[ch.value for ch in community.allowed_channels]}"
            )

    def _can_sanitize_and_share(self, reasons: List[str]) -> bool:
        """Determine if we can sanitize and proceed with sharing."""
        sanitizable_reasons = {
            "gdpr_review_required",
            "competition_review_required",
            "restricted_keyword",
            "pii_pattern_detected",
        }
        # Can sanitize if all reasons are sanitizable
        return all(
            any(r.startswith(sr) for sr in sanitizable_reasons)
            for r in reasons
        )

    def _determine_block_outcome(self, reasons: List[str]) -> SharingOutcome:
        """Determine blocking outcome based on reasons."""
        for reason in reasons:
            if "gdpr" in reason.lower():
                return SharingOutcome.BLOCKED_GDPR
            if "competition" in reason.lower():
                return SharingOutcome.BLOCKED_COMPETITION
            if "tlp" in reason.lower():
                return SharingOutcome.BLOCKED_TLP
        return SharingOutcome.BLOCKED_POLICY


# =============================================================================
# Factory Functions
# =============================================================================

def create_information_sharing(
    config: Optional[InformationSharingConfig] = None,
    policy: Optional[InformationSharingPolicy] = None,
) -> DORAInformationSharing:
    """
    Factory function to create DORAInformationSharing instance.

    Args:
        config: Service configuration
        policy: Sharing policy

    Returns:
        Configured DORAInformationSharing instance
    """
    return DORAInformationSharing(config=config, policy=policy)


def get_shareable_information_types() -> Set[str]:
    """Get set of shareable information types per Art. 45."""
    return set(SHAREABLE_INFORMATION_TYPES)


def get_tlp_definitions() -> Dict[str, Dict[str, str]]:
    """Get TLP 2.0 definitions."""
    return dict(TLP_DEFINITIONS)


def get_community_types() -> List[str]:
    """Get available community types."""
    return [t.value for t in CommunityType]


def get_sharing_channels() -> List[str]:
    """Get available sharing channels."""
    return [c.value for c in SharingChannel]


def get_tlp_levels() -> List[str]:
    """Get available TLP levels."""
    return [t.value for t in TLPLevel]


def create_sharing_community(
    name: str,
    community_type: CommunityType,
    country: str,
    contact_email: str,
    trust_level: int = 50,
    requires_anonymization: bool = True,
    channels: Optional[List[SharingChannel]] = None,
) -> SharingCommunity:
    """
    Factory function to create a SharingCommunity.

    Args:
        name: Community name
        community_type: Type of community
        country: Country/jurisdiction
        contact_email: Contact email
        trust_level: Trust score (0-100)
        requires_anonymization: Whether to always anonymize
        channels: Allowed sharing channels

    Returns:
        Configured SharingCommunity instance
    """
    return SharingCommunity(
        name=name,
        community_type=community_type,
        country=country,
        contact_email=contact_email,
        trust_level=trust_level,
        requires_anonymization=requires_anonymization,
        allowed_channels=channels or [SharingChannel.PORTAL],
    )


def create_cyber_threat(
    title: str,
    description: str,
    severity: ThreatSeverity = ThreatSeverity.MEDIUM,
    tlp_level: TLPLevel = TLPLevel.TLP_AMBER,
    iocs: Optional[List[str]] = None,
    ttps: Optional[List[str]] = None,
    info_types: Optional[Set[str]] = None,
) -> CyberThreatIntelligence:
    """
    Factory function to create CyberThreatIntelligence.

    Args:
        title: Threat title
        description: Threat description
        severity: Threat severity
        tlp_level: TLP classification
        iocs: Indicators of compromise
        ttps: Tactics, techniques, procedures
        info_types: Information types

    Returns:
        Configured CyberThreatIntelligence instance
    """
    return CyberThreatIntelligence(
        title=title,
        description=description,
        severity=severity,
        tlp_level=tlp_level,
        indicators_of_compromise=iocs or [],
        ttps=ttps or [],
        information_types=info_types or {"indicators_of_compromise"},
    )


def create_sharing_policy(
    allowed_types: Optional[Set[str]] = None,
    restricted_keywords: Optional[Set[str]] = None,
    default_tlp: TLPLevel = TLPLevel.TLP_AMBER,
    require_gdpr_review: bool = True,
    require_competition_review: bool = True,
    auto_sanitize: bool = True,
) -> InformationSharingPolicy:
    """
    Factory function to create InformationSharingPolicy.

    Args:
        allowed_types: Allowed information types
        restricted_keywords: Keywords to restrict
        default_tlp: Default TLP level
        require_gdpr_review: Require GDPR review
        require_competition_review: Require competition review
        auto_sanitize: Enable auto-sanitization

    Returns:
        Configured InformationSharingPolicy instance
    """
    policy = InformationSharingPolicy(
        default_tlp=default_tlp,
        require_gdpr_review=require_gdpr_review,
        require_competition_review=require_competition_review,
        auto_sanitize=auto_sanitize,
    )
    if allowed_types:
        policy.allowed_information_types = allowed_types
    if restricted_keywords:
        policy.restricted_keywords = restricted_keywords
    return policy
