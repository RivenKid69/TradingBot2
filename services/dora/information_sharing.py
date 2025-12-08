# -*- coding: utf-8 -*-
"""
DORA Article 45 - Information Sharing.

Implements cyber threat information sharing controls per Regulation (EU)
2022/2554 Article 45, including:
    - Participation in trusted information sharing communities
    - GDPR and competition law safeguards
    - Anonymisation and sanitisation of shared intelligence
    - NCA notification of participation in sharing arrangements

This module is intentionally self-contained to enable deterministic testing and
alignment with the integration blueprint in docs/compliance/DORA_INTEGRATION_PLAN.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


# Shareable information types per Article 45 guidance
SHAREABLE_INFORMATION = {
    "indicators_of_compromise",
    "tactics_techniques_procedures",
    "cybersecurity_alerts",
    "configuration_tools",
    "threat_actor_profiles",
}


class CommunityType(Enum):
    """Trusted community classification."""
    FS_ISAC = "fs_isac"
    CERT = "cert"
    CSIRT = "csirt"
    PUBLIC_PRIVATE_PARTNERSHIP = "public_private_partnership"
    PRIVATE_EXCHANGE = "private_exchange"


class SharingChannel(Enum):
    """Permitted sharing channels."""
    API = "api"
    EMAIL = "email"
    PORTAL = "portal"
    SECURE_FTP = "secure_ftp"


class SharingSensitivity(Enum):
    """Traffic Light Protocol style sensitivity."""
    TLP_RED = "tlp_red"
    TLP_AMBER = "tlp_amber"
    TLP_GREEN = "tlp_green"
    TLP_CLEAR = "tlp_clear"


class MembershipStatus(Enum):
    """Membership lifecycle."""
    PENDING = "pending"
    JOINED = "joined"
    SUSPENDED = "suspended"
    EXITED = "exited"


class SharingOutcome(Enum):
    """Result of a sharing attempt."""
    SUCCESS = "success"
    BLOCKED = "blocked"
    SANITISED = "sanitised"


@dataclass
class SharingCommunity:
    """Information sharing community metadata."""
    name: str
    community_type: CommunityType
    country: str
    contact_email: str
    community_id: str = ""
    trust_level: int = 50  # 0-100 scale
    requires_anonymization: bool = True
    allowed_channels: List[SharingChannel] = field(default_factory=lambda: [SharingChannel.PORTAL])
    membership_status: MembershipStatus = MembershipStatus.PENDING
    joined_at: Optional[datetime] = None
    notes: str = ""

    def __post_init__(self):
        if not self.community_id:
            self.community_id = f"COMM-{uuid4().hex[:10].upper()}"
        self.trust_level = max(0, min(100, self.trust_level))
        if self.joined_at and self.membership_status == MembershipStatus.PENDING:
            self.membership_status = MembershipStatus.JOINED


@dataclass
class InformationSharingPolicy:
    """Policy controls applied to outgoing intelligence."""
    allowed_information_types: Set[str] = field(default_factory=lambda: set(SHAREABLE_INFORMATION))
    restricted_keywords: Set[str] = field(default_factory=lambda: {"client_name", "pricing", "trade_id"})
    require_gdpr_review: bool = True
    require_competition_review: bool = True
    allowed_sensitivity: SharingSensitivity = SharingSensitivity.TLP_AMBER
    default_channel: SharingChannel = SharingChannel.PORTAL

    def is_shareable(self, threat: "CyberThreat") -> Tuple[bool, List[str]]:
        """Check if threat is shareable under policy."""
        reasons: List[str] = []
        if not threat.information_types.issubset(self.allowed_information_types):
            reasons.append("information_type_not_allowed")
        if any(keyword.lower() in threat.description.lower() for keyword in self.restricted_keywords):
            reasons.append("restricted_keyword_present")
        if threat.contains_personal_data and self.require_gdpr_review:
            reasons.append("gdpr_review_required")
        if threat.contains_client_data and self.require_competition_review:
            reasons.append("competition_review_required")
        return len(reasons) == 0, reasons


@dataclass
class CyberThreat:
    """Threat intelligence payload."""
    title: str
    description: str
    information_types: Set[str]
    indicators_of_compromise: List[str] = field(default_factory=list)
    ttps: List[str] = field(default_factory=list)
    severity: str = "medium"
    contains_personal_data: bool = False
    contains_client_data: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    threat_id: str = ""
    source: str = "internal_detection"

    def __post_init__(self):
        if not self.threat_id:
            self.threat_id = f"THR-{uuid4().hex[:8].upper()}"
        if not self.information_types:
            self.information_types = set(SHAREABLE_INFORMATION)


@dataclass
class ThreatIntelligence:
    """Shared or received intelligence record."""
    threat: CyberThreat
    community_id: str
    channel: SharingChannel
    sanitized: bool
    shared_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    received: bool = False
    policy_notes: List[str] = field(default_factory=list)
    outcome: SharingOutcome = SharingOutcome.SUCCESS
    record_id: str = ""

    def __post_init__(self):
        if not self.record_id:
            self.record_id = f"INTEL-{uuid4().hex[:10].upper()}"


@dataclass
class ThreatSharingRecord:
    """Audit entry for sharing events."""
    record_id: str
    threat_id: str
    community_id: str
    sanitized: bool
    channel: SharingChannel
    status: SharingOutcome
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    policy_checks: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def __post_init__(self):
        if not self.record_id:
            self.record_id = f"SHARE-{uuid4().hex[:10].upper()}"


class DORAInformationSharing:
    """
    Threat intelligence sharing orchestrator per Article 45.
    """

    def __init__(
        self,
        policy: Optional[InformationSharingPolicy] = None,
        gdpr_officer: str = "",
        nca_contact: str = "",
    ):
        self.policy = policy or InformationSharingPolicy()
        self.gdpr_officer = gdpr_officer or "dpo@platform.test"
        self.nca_contact = nca_contact or "nca@supervisor.test"
        self.communities: Dict[str, SharingCommunity] = {}
        self.shared_records: List[ThreatSharingRecord] = []
        self.received_intelligence: Dict[str, ThreatIntelligence] = {}

    # ------------------------------------------------------------------ #
    # Community Management
    # ------------------------------------------------------------------ #
    def register_community(self, community: SharingCommunity) -> SharingCommunity:
        """Register community metadata."""
        self.communities[community.community_id] = community
        return community

    def join_sharing_community(self, community: SharingCommunity) -> SharingCommunity:
        """Mark membership as joined and store community."""
        community.membership_status = MembershipStatus.JOINED
        community.joined_at = community.joined_at or datetime.now(timezone.utc)
        return self.register_community(community)

    # ------------------------------------------------------------------ #
    # Sharing Workflow
    # ------------------------------------------------------------------ #
    def _sanitize_text(self, text: str) -> str:
        """Redact restricted keywords."""
        clean_text = text
        for keyword in self.policy.restricted_keywords:
            clean_text = clean_text.replace(keyword, "[REDACTED]")
            clean_text = clean_text.replace(keyword.upper(), "[REDACTED]")
        return clean_text

    def sanitize_threat(self, threat: CyberThreat) -> CyberThreat:
        """Create a sanitized copy of the threat intelligence."""
        sanitized_iocs = [self._sanitize_text(ioc) for ioc in threat.indicators_of_compromise]
        sanitized_description = self._sanitize_text(threat.description)

        sanitized_threat = CyberThreat(
            title=threat.title,
            description=sanitized_description,
            information_types=set(threat.information_types),
            indicators_of_compromise=sanitized_iocs,
            ttps=list(threat.ttps),
            severity=threat.severity,
            contains_personal_data=False,
            contains_client_data=False,
            created_at=threat.created_at,
            threat_id=threat.threat_id,
            source=threat.source,
        )
        logger.debug("Sanitized threat %s", sanitized_threat.threat_id)
        return sanitized_threat

    def _validate_channel(self, community: SharingCommunity, channel: SharingChannel) -> None:
        if channel not in community.allowed_channels:
            raise ValueError(f"Channel {channel.value} not permitted for {community.name}")

    def _ensure_joined(self, community: SharingCommunity) -> None:
        if community.membership_status != MembershipStatus.JOINED:
            raise ValueError(f"Community {community.name} not joined")

    def share_threat_intelligence(
        self,
        threat: CyberThreat,
        community: SharingCommunity,
        channel: Optional[SharingChannel] = None,
    ) -> ThreatIntelligence:
        """
        Share threat intelligence with a community with policy enforcement.
        """
        self._ensure_joined(community)
        chosen_channel = channel or self.policy.default_channel
        self._validate_channel(community, chosen_channel)

        allowed, reasons = self.policy.is_shareable(threat)
        sanitized = False

        if not allowed:
            if any(reason in {"gdpr_review_required", "competition_review_required", "restricted_keyword_present"} for reason in reasons):
                threat_to_send = self.sanitize_threat(threat)
                sanitized = True
            else:
                record = ThreatSharingRecord(
                    record_id="",
                    threat_id=threat.threat_id,
                    community_id=community.community_id,
                    sanitized=False,
                    channel=chosen_channel,
                    status=SharingOutcome.BLOCKED,
                    policy_checks=reasons,
                    error="policy_violation",
                )
                self.shared_records.append(record)
                logger.warning("Sharing blocked for %s due to %s", threat.threat_id, reasons)
                raise ValueError(f"Threat not shareable: {','.join(reasons)}")
        else:
            threat_to_send = threat

        if community.requires_anonymization and not sanitized:
            threat_to_send = self.sanitize_threat(threat_to_send)
            sanitized = True

        intel = ThreatIntelligence(
            threat=threat_to_send,
            community_id=community.community_id,
            channel=chosen_channel,
            sanitized=sanitized,
            policy_notes=reasons,
            outcome=SharingOutcome.SANITISED if sanitized else SharingOutcome.SUCCESS,
        )

        record = ThreatSharingRecord(
            record_id="",
            threat_id=threat_to_send.threat_id,
            community_id=community.community_id,
            sanitized=sanitized,
            channel=chosen_channel,
            status=intel.outcome,
            policy_checks=reasons,
        )

        self.shared_records.append(record)
        logger.info("Shared threat %s with %s via %s", threat_to_send.threat_id, community.name, chosen_channel.value)
        return intel

    # ------------------------------------------------------------------ #
    # Receiving workflow
    # ------------------------------------------------------------------ #
    def receive_threat_intelligence(self, intelligence: ThreatIntelligence) -> bool:
        """Store received intelligence if not already processed."""
        key = f"{intelligence.threat.threat_id}:{intelligence.community_id}"
        if key in self.received_intelligence:
            logger.debug("Duplicate intelligence %s ignored", key)
            return False
        intelligence.received = True
        self.received_intelligence[key] = intelligence
        logger.info("Received intelligence %s from %s", intelligence.threat.threat_id, intelligence.community_id)
        return True

    # ------------------------------------------------------------------ #
    # Compliance Support
    # ------------------------------------------------------------------ #
    def notify_nca_of_participation(self, community: SharingCommunity) -> Dict[str, str]:
        """Prepare NCA notification payload per Article 45(3)."""
        payload = {
            "community_name": community.name,
            "community_type": community.community_type.value,
            "country": community.country,
            "notified_at": datetime.now(timezone.utc).isoformat(),
            "nca_contact": self.nca_contact,
            "gdpr_officer": self.gdpr_officer,
        }
        logger.debug("Prepared NCA notification: %s", payload)
        return payload

    def get_sharing_audit_log(self) -> List[ThreatSharingRecord]:
        """Return audit log for all sharing events."""
        return list(self.shared_records)

    def purge_stale_intelligence(self, max_age_days: int = 365) -> int:
        """
        Remove intelligence older than max_age_days to comply with data minimisation.
        Returns number of records removed.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        keys_to_remove = [key for key, intel in self.received_intelligence.items() if intel.shared_at < cutoff]
        for key in keys_to_remove:
            del self.received_intelligence[key]
        logger.debug("Purged %d stale intelligence items", len(keys_to_remove))
        return len(keys_to_remove)

