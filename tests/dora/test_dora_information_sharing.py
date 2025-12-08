# -*- coding: utf-8 -*-
"""
Tests for DORA Article 45 information sharing module.
"""

from datetime import datetime, timedelta, timezone

import pytest

from services.dora.information_sharing import (
    CommunityType,
    CyberThreat,
    DORAInformationSharing,
    InformationSharingPolicy,
    MembershipStatus,
    SharingChannel,
    SharingCommunity,
    SharingOutcome,
    SHAREABLE_INFORMATION,
    ThreatIntelligence,
)


@pytest.fixture
def policy():
    return InformationSharingPolicy(
        allowed_information_types=set(SHAREABLE_INFORMATION),
        restricted_keywords={"client_name", "sensitive"},
        require_gdpr_review=True,
        require_competition_review=True,
    )


@pytest.fixture
def community():
    return SharingCommunity(
        name="FS-ISAC",
        community_type=CommunityType.FS_ISAC,
        country="US",
        contact_email="contact@fsisac.test",
        trust_level=90,
        requires_anonymization=False,
        allowed_channels=[SharingChannel.PORTAL, SharingChannel.API],
        membership_status=MembershipStatus.JOINED,
    )


@pytest.fixture
def sharing(policy):
    return DORAInformationSharing(policy=policy, gdpr_officer="dpo@test", nca_contact="nca@test")


@pytest.fixture
def base_threat():
    return CyberThreat(
        title="Malicious IOC",
        description="IOC observed on perimeter",
        information_types={"indicators_of_compromise"},
        indicators_of_compromise=["1.1.1.1"],
        ttps=["T1040"],
        severity="high",
        contains_personal_data=False,
        contains_client_data=False,
    )


def test_join_and_register_community(sharing, community):
    sharing.join_sharing_community(community)
    assert community.community_id in sharing.communities
    assert sharing.communities[community.community_id].membership_status is MembershipStatus.JOINED


def test_share_without_sanitization(sharing, community, base_threat):
    sharing.join_sharing_community(community)
    intel = sharing.share_threat_intelligence(base_threat, community, channel=SharingChannel.PORTAL)
    assert intel.sanitized is False
    assert intel.outcome is SharingOutcome.SUCCESS
    assert sharing.shared_records[-1].status is SharingOutcome.SUCCESS


def test_share_requires_anonymization(sharing, base_threat):
    community = SharingCommunity(
        name="CERT",
        community_type=CommunityType.CERT,
        country="DE",
        contact_email="cert@test",
        trust_level=75,
        requires_anonymization=True,
        allowed_channels=[SharingChannel.PORTAL],
        membership_status=MembershipStatus.JOINED,
    )
    sharing.join_sharing_community(community)
    intel = sharing.share_threat_intelligence(base_threat, community)
    assert intel.sanitized is True


def test_share_triggers_sanitization(sharing, community, base_threat):
    sharing.join_sharing_community(community)
    sensitive = CyberThreat(
        title="Contains client_name",
        description="client_name leaked via misconfiguration",
        information_types={"indicators_of_compromise"},
        indicators_of_compromise=["client_name@corp.test"],
        ttps=["T1190"],
        severity="critical",
        contains_personal_data=True,
        contains_client_data=True,
    )
    intel = sharing.share_threat_intelligence(sensitive, community, channel=SharingChannel.API)
    assert intel.sanitized is True
    assert "[REDACTED]" in intel.threat.description
    assert intel.outcome is SharingOutcome.SANITISED


def test_blocked_sharing_for_disallowed_type(sharing, community):
    sharing.join_sharing_community(community)
    non_shareable = CyberThreat(
        title="Non shareable payload",
        description="benign description",
        information_types={"proprietary_trading_logic"},
        indicators_of_compromise=[],
        ttps=[],
        severity="low",
    )
    with pytest.raises(ValueError):
        sharing.share_threat_intelligence(non_shareable, community, channel=SharingChannel.PORTAL)
    assert sharing.shared_records[-1].status is SharingOutcome.BLOCKED


def test_disallowed_channel_raises(sharing, community, base_threat):
    sharing.join_sharing_community(community)
    with pytest.raises(ValueError):
        sharing.share_threat_intelligence(base_threat, community, channel=SharingChannel.SECURE_FTP)


def test_receive_intelligence_and_dedup(sharing, community, base_threat):
    sharing.join_sharing_community(community)
    intel = sharing.share_threat_intelligence(base_threat, community)
    received_first = sharing.receive_threat_intelligence(intel)
    received_second = sharing.receive_threat_intelligence(intel)
    assert received_first is True
    assert received_second is False


def test_purge_stale_intelligence(sharing, community, base_threat):
    sharing.join_sharing_community(community)
    intel = sharing.share_threat_intelligence(base_threat, community)
    intel.shared_at = datetime.now(timezone.utc) - timedelta(days=400)
    sharing.receive_threat_intelligence(intel)
    removed = sharing.purge_stale_intelligence(max_age_days=365)
    assert removed == 1


def test_notify_nca_payload(sharing, community):
    sharing.join_sharing_community(community)
    payload = sharing.notify_nca_of_participation(community)
    assert payload["community_name"] == "FS-ISAC"
    assert payload["nca_contact"] == "nca@test"
