# -*- coding: utf-8 -*-
"""
Comprehensive tests for DORA Article 45 - Information Sharing.

Tests the DORAInformationSharing service class and all related
components for full DORA compliance.

Test Categories:
    1. Constants and Enums
    2. Data Structures
    3. Policy Configuration
    4. Community Management
    5. Threat Intelligence Sharing
    6. Sanitization
    7. NCA Notification
    8. Audit Trail
    9. STIX Export
    10. Factory Functions
    11. Edge Cases and Error Handling

Coverage Target: 100%
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
import re

# Import from sharing module
from services.dora_integration.sharing import (
    # Constants
    SHAREABLE_INFORMATION_TYPES,
    TLP_DEFINITIONS,
    DEFAULT_INTELLIGENCE_RETENTION_DAYS,
    NCA_NOTIFICATION_DEADLINE_DAYS,
    # Enums
    CommunityType,
    SharingChannel,
    TLPLevel,
    MembershipStatus,
    SharingOutcome,
    IntelligenceDirection,
    ThreatSeverity,
    SanitizationLevel,
    # Data Structures
    SharingCommunity,
    InformationSharingPolicy,
    CyberThreatIntelligence,
    ThreatIntelligenceRecord,
    SharingAuditRecord,
    NCANotification,
    InformationSharingConfig,
    # Main class
    DORAInformationSharing,
    # Factory functions
    create_information_sharing,
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
# Test Fixtures
# =============================================================================

@pytest.fixture
def sharing_config():
    """Create a test sharing configuration."""
    return InformationSharingConfig(
        provider_name="Test Platform",
        provider_lei="TEST12345678901234",
        gdpr_officer_email="dpo@test.example",
        nca_contact_email="nca@test.example",
    )


@pytest.fixture
def sharing_policy():
    """Create a test sharing policy."""
    return InformationSharingPolicy(
        policy_id="POL-TEST-001",
        default_tlp=TLPLevel.TLP_AMBER,
        require_gdpr_review=True,
        require_competition_review=True,
        auto_sanitize=True,
    )


@pytest.fixture
def sharing_service(sharing_config, sharing_policy):
    """Create a configured sharing service."""
    return DORAInformationSharing(
        config=sharing_config,
        policy=sharing_policy,
    )


@pytest.fixture
def test_community():
    """Create a test sharing community."""
    return SharingCommunity(
        name="Test ISAC",
        community_type=CommunityType.FS_ISAC,
        country="EU",
        contact_email="contact@test-isac.example",
        trust_level=75,
        requires_anonymization=True,
        allowed_channels=[SharingChannel.PORTAL, SharingChannel.API],
    )


@pytest.fixture
def active_community(test_community):
    """Create an active community ready for sharing."""
    test_community.membership_status = MembershipStatus.ACTIVE
    test_community.joined_at = datetime.now(timezone.utc)
    return test_community


@pytest.fixture
def test_threat():
    """Create a test threat intelligence payload."""
    return CyberThreatIntelligence(
        title="Test Phishing Campaign",
        description="Targeting financial institutions with credential harvesting",
        information_types={"indicators_of_compromise", "attack_patterns"},
        indicators_of_compromise=["192.168.1.100", "malware.example.com", "abc123def456"],
        ttps=["T1566.001", "T1059.001"],
        severity=ThreatSeverity.HIGH,
        tlp_level=TLPLevel.TLP_AMBER,
        contains_personal_data=False,
        contains_client_data=False,
        source="internal_detection",
        confidence=85,
    )


@pytest.fixture
def sensitive_threat():
    """Create a threat with sensitive data requiring sanitization."""
    return CyberThreatIntelligence(
        title="Internal Data Exposure",
        description="Found client_name data in logs with account_number exposed. Contact john@company.com for details.",
        information_types={"indicators_of_compromise"},
        indicators_of_compromise=["10.0.0.50", "internal.corp.local"],
        severity=ThreatSeverity.CRITICAL,
        tlp_level=TLPLevel.TLP_AMBER,
        contains_personal_data=True,
        contains_client_data=True,
        source="internal_audit",
    )


# =============================================================================
# Test 1: Constants and Enums
# =============================================================================

class TestConstants:
    """Test module constants."""

    def test_shareable_information_types_defined(self):
        """Test that shareable information types are defined."""
        assert isinstance(SHAREABLE_INFORMATION_TYPES, set)
        assert len(SHAREABLE_INFORMATION_TYPES) >= 10
        assert "indicators_of_compromise" in SHAREABLE_INFORMATION_TYPES
        assert "tactics_techniques_procedures" in SHAREABLE_INFORMATION_TYPES
        assert "cybersecurity_alerts" in SHAREABLE_INFORMATION_TYPES

    def test_tlp_definitions_complete(self):
        """Test TLP definitions are complete."""
        assert isinstance(TLP_DEFINITIONS, dict)
        assert "TLP:RED" in TLP_DEFINITIONS
        assert "TLP:AMBER" in TLP_DEFINITIONS
        assert "TLP:GREEN" in TLP_DEFINITIONS
        assert "TLP:CLEAR" in TLP_DEFINITIONS

        for tlp_name, definition in TLP_DEFINITIONS.items():
            assert "description" in definition
            assert "sharing_scope" in definition
            assert "color_hex" in definition

    def test_retention_days_defined(self):
        """Test retention days constant."""
        assert DEFAULT_INTELLIGENCE_RETENTION_DAYS == 365

    def test_nca_notification_deadline(self):
        """Test NCA notification deadline constant."""
        assert NCA_NOTIFICATION_DEADLINE_DAYS == 30


class TestEnums:
    """Test enum definitions."""

    def test_community_type_values(self):
        """Test CommunityType enum values."""
        assert CommunityType.FS_ISAC.value == "fs_isac"
        assert CommunityType.CERT.value == "cert"
        assert CommunityType.CSIRT.value == "csirt"
        assert CommunityType.PUBLIC_PRIVATE_PARTNERSHIP.value == "public_private_partnership"
        assert CommunityType.PRIVATE_EXCHANGE.value == "private_exchange"

    def test_sharing_channel_values(self):
        """Test SharingChannel enum values."""
        assert SharingChannel.API.value == "api"
        assert SharingChannel.EMAIL.value == "email"
        assert SharingChannel.PORTAL.value == "portal"
        assert SharingChannel.STIX_TAXII.value == "stix_taxii"
        assert SharingChannel.MISP.value == "misp"

    def test_tlp_level_values(self):
        """Test TLPLevel enum values."""
        assert TLPLevel.TLP_RED.value == "tlp_red"
        assert TLPLevel.TLP_AMBER_STRICT.value == "tlp_amber_strict"
        assert TLPLevel.TLP_AMBER.value == "tlp_amber"
        assert TLPLevel.TLP_GREEN.value == "tlp_green"
        assert TLPLevel.TLP_CLEAR.value == "tlp_clear"

    def test_membership_status_values(self):
        """Test MembershipStatus enum values."""
        assert MembershipStatus.PENDING.value == "pending"
        assert MembershipStatus.ACTIVE.value == "active"
        assert MembershipStatus.SUSPENDED.value == "suspended"
        assert MembershipStatus.EXITED.value == "exited"

    def test_sharing_outcome_values(self):
        """Test SharingOutcome enum values."""
        assert SharingOutcome.SUCCESS.value == "success"
        assert SharingOutcome.SANITISED.value == "sanitised"
        assert SharingOutcome.BLOCKED_POLICY.value == "blocked_policy"
        assert SharingOutcome.BLOCKED_GDPR.value == "blocked_gdpr"

    def test_threat_severity_values(self):
        """Test ThreatSeverity enum values."""
        assert ThreatSeverity.CRITICAL.value == "critical"
        assert ThreatSeverity.HIGH.value == "high"
        assert ThreatSeverity.MEDIUM.value == "medium"
        assert ThreatSeverity.LOW.value == "low"

    def test_sanitization_level_values(self):
        """Test SanitizationLevel enum values."""
        assert SanitizationLevel.NONE.value == "none"
        assert SanitizationLevel.MINIMAL.value == "minimal"
        assert SanitizationLevel.MODERATE.value == "moderate"
        assert SanitizationLevel.AGGRESSIVE.value == "aggressive"


# =============================================================================
# Test 2: Data Structures
# =============================================================================

class TestSharingCommunity:
    """Test SharingCommunity dataclass."""

    def test_community_creation(self, test_community):
        """Test basic community creation."""
        assert test_community.name == "Test ISAC"
        assert test_community.community_type == CommunityType.FS_ISAC
        assert test_community.country == "EU"
        assert test_community.trust_level == 75

    def test_community_auto_id_generation(self):
        """Test automatic community ID generation."""
        community = SharingCommunity(
            name="Auto ID Test",
            community_type=CommunityType.CERT,
            country="US",
            contact_email="test@example.com",
        )
        assert community.community_id.startswith("COMM-")
        assert len(community.community_id) == 15  # COMM- + 10 chars

    def test_trust_level_clamping(self):
        """Test trust level is clamped to 0-100."""
        community = SharingCommunity(
            name="High Trust",
            community_type=CommunityType.CERT,
            country="US",
            contact_email="test@example.com",
            trust_level=150,  # Should be clamped to 100
        )
        assert community.trust_level == 100

        community_low = SharingCommunity(
            name="Low Trust",
            community_type=CommunityType.CERT,
            country="US",
            contact_email="test@example.com",
            trust_level=-10,  # Should be clamped to 0
        )
        assert community_low.trust_level == 0

    def test_is_active_method(self, active_community):
        """Test is_active method."""
        # Create a fresh pending community for testing
        pending_community = SharingCommunity(
            name="Pending Community",
            community_type=CommunityType.CERT,
            country="US",
            contact_email="pending@test.example",
        )
        assert not pending_community.is_active()  # Pending
        assert active_community.is_active()

    def test_requires_nca_notification(self, active_community):
        """Test requires_nca_notification method."""
        assert active_community.requires_nca_notification()
        active_community.nca_notified = True
        assert not active_community.requires_nca_notification()

    def test_community_to_dict(self, test_community):
        """Test community serialization."""
        data = test_community.to_dict()
        assert data["name"] == "Test ISAC"
        assert data["community_type"] == "fs_isac"
        assert data["country"] == "EU"
        assert "community_id" in data
        assert "allowed_channels" in data


class TestInformationSharingPolicy:
    """Test InformationSharingPolicy dataclass."""

    def test_policy_creation(self, sharing_policy):
        """Test policy creation with defaults."""
        assert sharing_policy.policy_id == "POL-TEST-001"
        assert sharing_policy.default_tlp == TLPLevel.TLP_AMBER
        assert sharing_policy.require_gdpr_review is True
        assert sharing_policy.auto_sanitize is True

    def test_policy_auto_id_generation(self):
        """Test automatic policy ID generation."""
        policy = InformationSharingPolicy()
        assert policy.policy_id.startswith("POL-")

    def test_is_shareable_allowed(self, sharing_policy, test_threat):
        """Test is_shareable for allowed threat."""
        is_shareable, reasons = sharing_policy.is_shareable(test_threat)
        assert is_shareable is True
        assert len(reasons) == 0

    def test_is_shareable_blocked_info_type(self, sharing_policy):
        """Test is_shareable blocks unknown info types."""
        threat = CyberThreatIntelligence(
            title="Test",
            description="Test description",
            information_types={"unknown_type"},
        )
        is_shareable, reasons = sharing_policy.is_shareable(threat)
        assert is_shareable is False
        assert any("information_type_not_allowed" in r for r in reasons)

    def test_is_shareable_restricted_keyword(self, sharing_policy):
        """Test is_shareable detects restricted keywords."""
        threat = CyberThreatIntelligence(
            title="Test",
            description="Found client_name data exposed",
            information_types={"indicators_of_compromise"},
        )
        is_shareable, reasons = sharing_policy.is_shareable(threat)
        assert is_shareable is False
        assert any("restricted_keyword" in r for r in reasons)

    def test_is_shareable_gdpr_flag(self, sharing_policy, sensitive_threat):
        """Test is_shareable detects personal data."""
        is_shareable, reasons = sharing_policy.is_shareable(sensitive_threat)
        assert is_shareable is False
        assert "gdpr_review_required" in reasons

    def test_is_shareable_competition_flag(self, sharing_policy, sensitive_threat):
        """Test is_shareable detects client data."""
        is_shareable, reasons = sharing_policy.is_shareable(sensitive_threat)
        assert "competition_review_required" in reasons

    def test_policy_to_dict(self, sharing_policy):
        """Test policy serialization."""
        data = sharing_policy.to_dict()
        assert data["policy_id"] == "POL-TEST-001"
        assert data["default_tlp"] == "tlp_amber"
        assert "allowed_information_types" in data


class TestCyberThreatIntelligence:
    """Test CyberThreatIntelligence dataclass."""

    def test_threat_creation(self, test_threat):
        """Test threat creation."""
        assert test_threat.title == "Test Phishing Campaign"
        assert test_threat.severity == ThreatSeverity.HIGH
        assert test_threat.confidence == 85

    def test_threat_auto_id_generation(self):
        """Test automatic threat ID generation."""
        threat = CyberThreatIntelligence(
            title="Auto ID Test",
            description="Test",
        )
        assert threat.threat_id.startswith("THR-")

    def test_threat_confidence_clamping(self):
        """Test confidence is clamped to 0-100."""
        threat = CyberThreatIntelligence(
            title="High Confidence",
            description="Test",
            confidence=150,
        )
        assert threat.confidence == 100

    def test_threat_default_info_types(self):
        """Test default information types."""
        threat = CyberThreatIntelligence(
            title="Default Info Types",
            description="Test",
        )
        assert "indicators_of_compromise" in threat.information_types

    def test_threat_expiration(self):
        """Test threat expiration calculation."""
        threat = CyberThreatIntelligence(
            title="Expiration Test",
            description="Test",
        )
        assert threat.expires_at is not None
        assert threat.expires_at > threat.created_at

    def test_is_expired(self):
        """Test is_expired method."""
        threat = CyberThreatIntelligence(
            title="Expired Test",
            description="Test",
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        assert threat.is_expired() is True

        fresh_threat = CyberThreatIntelligence(
            title="Fresh Test",
            description="Test",
        )
        assert fresh_threat.is_expired() is False

    def test_get_hash(self, test_threat):
        """Test content hash generation."""
        hash1 = test_threat.get_hash()
        assert isinstance(hash1, str)
        assert len(hash1) == 16

        # Same content should produce same hash
        threat2 = CyberThreatIntelligence(
            title=test_threat.title,
            description=test_threat.description,
            indicators_of_compromise=test_threat.indicators_of_compromise.copy(),
        )
        assert threat2.get_hash() == hash1

    def test_threat_to_dict(self, test_threat):
        """Test threat serialization."""
        data = test_threat.to_dict()
        assert data["title"] == "Test Phishing Campaign"
        assert data["severity"] == "high"
        assert "indicators_of_compromise" in data


class TestThreatIntelligenceRecord:
    """Test ThreatIntelligenceRecord dataclass."""

    def test_record_creation(self, test_threat, active_community):
        """Test record creation."""
        record = ThreatIntelligenceRecord(
            threat=test_threat,
            community_id=active_community.community_id,
            channel=SharingChannel.PORTAL,
            sanitized=False,
            outcome=SharingOutcome.SUCCESS,
            direction=IntelligenceDirection.OUTBOUND,
        )
        assert record.record_id.startswith("INTEL-")
        assert record.sanitized is False
        assert record.outcome == SharingOutcome.SUCCESS

    def test_record_to_dict(self, test_threat, active_community):
        """Test record serialization."""
        record = ThreatIntelligenceRecord(
            threat=test_threat,
            community_id=active_community.community_id,
            channel=SharingChannel.PORTAL,
            sanitized=True,
            outcome=SharingOutcome.SANITISED,
            direction=IntelligenceDirection.OUTBOUND,
        )
        data = record.to_dict()
        assert "record_id" in data
        assert "threat" in data
        assert data["sanitized"] is True


class TestSharingAuditRecord:
    """Test SharingAuditRecord dataclass."""

    def test_audit_record_creation(self, test_threat, active_community):
        """Test audit record creation."""
        audit = SharingAuditRecord(
            record_id="",
            threat_id=test_threat.threat_id,
            community_id=active_community.community_id,
            direction=IntelligenceDirection.OUTBOUND,
            outcome=SharingOutcome.SUCCESS,
            channel=SharingChannel.PORTAL,
            sanitized=False,
            sanitization_level=SanitizationLevel.NONE,
        )
        assert audit.record_id.startswith("AUDIT-")

    def test_audit_record_to_dict(self, test_threat, active_community):
        """Test audit record serialization."""
        audit = SharingAuditRecord(
            record_id="",
            threat_id=test_threat.threat_id,
            community_id=active_community.community_id,
            direction=IntelligenceDirection.OUTBOUND,
            outcome=SharingOutcome.BLOCKED_POLICY,
            channel=SharingChannel.PORTAL,
            sanitized=False,
            sanitization_level=SanitizationLevel.NONE,
            error="policy_violation",
        )
        data = audit.to_dict()
        assert data["outcome"] == "blocked_policy"
        assert data["error"] == "policy_violation"


class TestNCANotification:
    """Test NCANotification dataclass."""

    def test_notification_creation(self, active_community):
        """Test NCA notification creation."""
        notification = NCANotification(
            community_id=active_community.community_id,
            community_name=active_community.name,
            community_type=active_community.community_type,
            country=active_community.country,
        )
        assert notification.notification_id.startswith("NCA-")
        assert notification.acknowledgment_received is False

    def test_notification_to_dict(self, active_community):
        """Test notification serialization."""
        notification = NCANotification(
            community_id=active_community.community_id,
            community_name=active_community.name,
            community_type=active_community.community_type,
            country=active_community.country,
            dpo_contact="dpo@test.example",
        )
        data = notification.to_dict()
        assert data["community_name"] == "Test ISAC"
        assert data["dpo_contact"] == "dpo@test.example"


# =============================================================================
# Test 3: DORAInformationSharing Service
# =============================================================================

class TestDORAInformationSharingInit:
    """Test DORAInformationSharing initialization."""

    def test_default_initialization(self):
        """Test service initialization with defaults."""
        service = DORAInformationSharing()
        assert service.config is not None
        assert service.policy is not None

    def test_custom_initialization(self, sharing_config, sharing_policy):
        """Test service initialization with custom config."""
        service = DORAInformationSharing(
            config=sharing_config,
            policy=sharing_policy,
        )
        assert service.config.provider_name == "Test Platform"
        assert service.policy.policy_id == "POL-TEST-001"


class TestCommunityManagement:
    """Test community management methods."""

    def test_register_community(self, sharing_service, test_community):
        """Test community registration."""
        result = sharing_service.register_community(test_community)
        assert result.community_id == test_community.community_id
        assert sharing_service.get_community(test_community.community_id) is not None

    def test_join_community(self, sharing_service, test_community):
        """Test joining a community."""
        result = sharing_service.join_community(test_community)
        assert result.membership_status == MembershipStatus.ACTIVE
        assert result.joined_at is not None

    def test_exit_community(self, sharing_service, test_community):
        """Test exiting a community."""
        sharing_service.join_community(test_community)
        result = sharing_service.exit_community(
            test_community.community_id,
            reason="test_exit"
        )
        assert result.membership_status == MembershipStatus.EXITED
        assert "test_exit" in result.notes

    def test_exit_nonexistent_community(self, sharing_service):
        """Test exiting a non-existent community."""
        result = sharing_service.exit_community("NON-EXISTENT")
        assert result is None

    def test_get_community(self, sharing_service, test_community):
        """Test getting community by ID."""
        sharing_service.register_community(test_community)
        result = sharing_service.get_community(test_community.community_id)
        assert result is not None
        assert result.name == "Test ISAC"

    def test_list_communities(self, sharing_service, test_community):
        """Test listing all communities."""
        sharing_service.join_community(test_community)
        communities = sharing_service.list_communities()
        assert len(communities) == 1

    def test_list_communities_by_status(self, sharing_service, test_community):
        """Test listing communities by status."""
        sharing_service.join_community(test_community)
        active = sharing_service.list_communities(status=MembershipStatus.ACTIVE)
        pending = sharing_service.list_communities(status=MembershipStatus.PENDING)
        assert len(active) == 1
        assert len(pending) == 0

    def test_get_active_communities(self, sharing_service, test_community):
        """Test getting active communities."""
        sharing_service.join_community(test_community)
        active = sharing_service.get_active_communities()
        assert len(active) == 1


class TestThreatIntelligenceSharing:
    """Test threat intelligence sharing methods."""

    def test_share_threat_success(self, sharing_service, active_community, test_threat):
        """Test successful threat sharing."""
        sharing_service.register_community(active_community)
        result = sharing_service.share_threat_intelligence(
            test_threat,
            active_community,
            channel=SharingChannel.PORTAL,
        )
        assert result.outcome in {SharingOutcome.SUCCESS, SharingOutcome.SANITISED}

    def test_share_threat_with_sanitization(self, sharing_service, active_community, sensitive_threat):
        """Test threat sharing with automatic sanitization."""
        sharing_service.register_community(active_community)
        result = sharing_service.share_threat_intelligence(
            sensitive_threat,
            active_community,
        )
        assert result.sanitized is True
        assert result.outcome == SharingOutcome.SANITISED

    def test_share_threat_blocked_info_type(self, sharing_service, active_community):
        """Test sharing blocked due to info type."""
        sharing_service.register_community(active_community)

        threat = CyberThreatIntelligence(
            title="Blocked Type",
            description="Test",
            information_types={"proprietary_algorithms"},  # Not allowed
            tlp_level=TLPLevel.TLP_AMBER,
        )

        result = sharing_service.share_threat_intelligence(threat, active_community)
        assert result.outcome in {
            SharingOutcome.BLOCKED_POLICY,
            SharingOutcome.BLOCKED_GDPR,
            SharingOutcome.BLOCKED_COMPETITION,
        }

    def test_share_threat_not_joined(self, sharing_service, test_community, test_threat):
        """Test sharing to non-joined community raises error."""
        sharing_service.register_community(test_community)
        with pytest.raises(ValueError, match="membership not active"):
            sharing_service.share_threat_intelligence(test_threat, test_community)

    def test_share_threat_invalid_channel(self, sharing_service, active_community, test_threat):
        """Test sharing with invalid channel raises error."""
        active_community.allowed_channels = [SharingChannel.EMAIL]
        sharing_service.register_community(active_community)

        with pytest.raises(ValueError, match="not permitted"):
            sharing_service.share_threat_intelligence(
                test_threat,
                active_community,
                channel=SharingChannel.STIX_TAXII,
            )

    def test_share_threat_force_sanitize(self, sharing_service, active_community, test_threat):
        """Test force sanitization option."""
        active_community.requires_anonymization = False
        sharing_service.register_community(active_community)

        result = sharing_service.share_threat_intelligence(
            test_threat,
            active_community,
            force_sanitize=True,
        )
        assert result.sanitized is True

    def test_receive_threat_intelligence(self, sharing_service, test_threat, active_community):
        """Test receiving threat intelligence."""
        sharing_service.register_community(active_community)
        result = sharing_service.receive_threat_intelligence(
            test_threat,
            active_community.community_id,
        )
        assert result is True

    def test_receive_duplicate_threat(self, sharing_service, test_threat, active_community):
        """Test duplicate threat detection."""
        sharing_service.register_community(active_community)
        sharing_service.receive_threat_intelligence(test_threat, active_community.community_id)

        # Try to receive same threat again
        result = sharing_service.receive_threat_intelligence(
            test_threat,
            active_community.community_id,
        )
        assert result is False

    def test_receive_from_unknown_community(self, sharing_service, test_threat):
        """Test receiving from unknown community logs warning."""
        result = sharing_service.receive_threat_intelligence(
            test_threat,
            "UNKNOWN-COMMUNITY",
        )
        assert result is True  # Still processes the threat


class TestSanitization:
    """Test sanitization methods."""

    def test_sanitize_threat(self, sharing_service, sensitive_threat):
        """Test threat sanitization."""
        sanitized = sharing_service.sanitize_threat(sensitive_threat)
        assert sanitized.contains_personal_data is False
        assert sanitized.contains_client_data is False

    def test_sanitize_removes_keywords(self, sharing_service):
        """Test sanitization removes restricted keywords."""
        threat = CyberThreatIntelligence(
            title="Keyword Test",
            description="Found client_name and account_number in logs",
            information_types={"indicators_of_compromise"},
        )
        sanitized = sharing_service.sanitize_threat(threat)
        assert "client_name" not in sanitized.description.lower()
        assert "[REDACTED]" in sanitized.description

    def test_sanitize_pii_patterns(self, sharing_service):
        """Test sanitization removes PII patterns."""
        threat = CyberThreatIntelligence(
            title="PII Test",
            description="Contact email: test@example.com, IP: 192.168.1.1",
            information_types={"indicators_of_compromise"},
        )
        sanitized = sharing_service.sanitize_threat(
            threat,
            level=SanitizationLevel.MODERATE,
        )
        assert "test@example.com" not in sanitized.description

    def test_sanitize_aggressive_level(self, sharing_service):
        """Test aggressive sanitization."""
        threat = CyberThreatIntelligence(
            title="Aggressive Test",
            description="Path: /home/user/data and host server.internal.local",
            indicators_of_compromise=["internal.corp.local"],
            information_types={"indicators_of_compromise"},
        )
        sanitized = sharing_service.sanitize_threat(
            threat,
            level=SanitizationLevel.AGGRESSIVE,
        )
        assert "internal.local" not in sanitized.description.lower() or "[REDACTED]" in sanitized.description


class TestNCANotification:
    """Test NCA notification methods."""

    def test_notify_nca_of_participation(self, sharing_service, active_community):
        """Test NCA notification preparation."""
        sharing_service.register_community(active_community)
        notification = sharing_service.notify_nca_of_participation(active_community)

        assert notification.community_id == active_community.community_id
        assert notification.community_name == active_community.name
        assert active_community.nca_notified is True

    def test_get_pending_nca_notifications(self, sharing_service, active_community):
        """Test getting pending NCA notifications."""
        sharing_service.register_community(active_community)
        sharing_service.notify_nca_of_participation(active_community)

        pending = sharing_service.get_pending_nca_notifications()
        assert len(pending) == 1

    def test_acknowledge_nca_notification(self, sharing_service, active_community):
        """Test acknowledging NCA notification."""
        sharing_service.register_community(active_community)
        notification = sharing_service.notify_nca_of_participation(active_community)

        result = sharing_service.acknowledge_nca_notification(
            notification.notification_id,
            "NCA-REF-12345",
        )

        assert result.acknowledgment_received is True
        assert result.nca_reference == "NCA-REF-12345"

    def test_acknowledge_nonexistent_notification(self, sharing_service):
        """Test acknowledging non-existent notification."""
        result = sharing_service.acknowledge_nca_notification(
            "NON-EXISTENT",
            "NCA-REF-12345",
        )
        assert result is None


class TestAuditAndCompliance:
    """Test audit and compliance methods."""

    def test_get_sharing_audit_log(self, sharing_service, active_community, test_threat):
        """Test getting audit log."""
        sharing_service.register_community(active_community)
        sharing_service.share_threat_intelligence(test_threat, active_community)

        audit_log = sharing_service.get_sharing_audit_log()
        assert len(audit_log) >= 1

    def test_audit_log_filtering_by_community(self, sharing_service, active_community, test_threat):
        """Test audit log filtering by community."""
        sharing_service.register_community(active_community)
        sharing_service.share_threat_intelligence(test_threat, active_community)

        audit_log = sharing_service.get_sharing_audit_log(
            community_id=active_community.community_id
        )
        assert all(r.community_id == active_community.community_id for r in audit_log)

    def test_audit_log_filtering_by_direction(self, sharing_service, active_community, test_threat):
        """Test audit log filtering by direction."""
        sharing_service.register_community(active_community)
        sharing_service.share_threat_intelligence(test_threat, active_community)

        outbound_log = sharing_service.get_sharing_audit_log(
            direction=IntelligenceDirection.OUTBOUND
        )
        assert all(r.direction == IntelligenceDirection.OUTBOUND for r in outbound_log)

    def test_audit_log_filtering_by_date(self, sharing_service, active_community, test_threat):
        """Test audit log filtering by date range."""
        sharing_service.register_community(active_community)
        sharing_service.share_threat_intelligence(test_threat, active_community)

        start_date = datetime.now(timezone.utc) - timedelta(hours=1)
        end_date = datetime.now(timezone.utc) + timedelta(hours=1)

        audit_log = sharing_service.get_sharing_audit_log(
            start_date=start_date,
            end_date=end_date,
        )
        assert len(audit_log) >= 1

    def test_get_sharing_statistics(self, sharing_service, active_community, test_threat):
        """Test getting sharing statistics."""
        sharing_service.register_community(active_community)
        sharing_service.share_threat_intelligence(test_threat, active_community)
        sharing_service.receive_threat_intelligence(test_threat, active_community.community_id)

        stats = sharing_service.get_sharing_statistics()
        assert "total_sharing_events" in stats
        assert "outbound_shares" in stats
        assert "inbound_received" in stats
        assert "active_communities" in stats

    def test_generate_compliance_report(self, sharing_service, active_community, test_threat):
        """Test compliance report generation."""
        sharing_service.register_community(active_community)
        sharing_service.share_threat_intelligence(test_threat, active_community)

        report = sharing_service.generate_compliance_report()
        assert "report_id" in report
        assert "statistics" in report
        assert "communities" in report
        assert "compliance_status" in report


class TestDataRetention:
    """Test data retention methods."""

    def test_purge_stale_intelligence(self, sharing_service, active_community):
        """Test purging stale intelligence."""
        # Add old intelligence
        old_threat = CyberThreatIntelligence(
            title="Old Threat",
            description="Test",
            created_at=datetime.now(timezone.utc) - timedelta(days=400),
        )
        sharing_service.receive_threat_intelligence(old_threat, active_community.community_id)

        # Purge with default retention
        purged = sharing_service.purge_stale_intelligence()
        assert purged >= 1

    def test_purge_with_custom_retention(self, sharing_service, active_community):
        """Test purging with custom retention period."""
        # Add recent intelligence
        recent_threat = CyberThreatIntelligence(
            title="Recent Threat",
            description="Test",
        )
        sharing_service.receive_threat_intelligence(recent_threat, active_community.community_id)

        # Purge with very short retention should purge nothing (too recent)
        purged = sharing_service.purge_stale_intelligence(max_age_days=1)
        # Recent should not be purged
        assert sharing_service._received_intelligence  # Still has data


class TestSTIXExport:
    """Test STIX export functionality."""

    def test_export_to_stix(self, sharing_service, test_threat):
        """Test STIX export."""
        stix_bundle = sharing_service.export_to_stix(test_threat)

        assert stix_bundle["type"] == "bundle"
        assert "objects" in stix_bundle
        assert len(stix_bundle["objects"]) >= 1

        indicator = stix_bundle["objects"][0]
        assert indicator["type"] == "indicator"
        assert indicator["spec_version"] == "2.1"

    def test_stix_export_with_tlp(self, sharing_service, test_threat):
        """Test STIX export includes TLP markings."""
        stix_bundle = sharing_service.export_to_stix(test_threat)
        indicator = stix_bundle["objects"][0]

        assert "object_marking_refs" in indicator

    def test_stix_pattern_generation(self, sharing_service, test_threat):
        """Test STIX pattern is generated."""
        stix_bundle = sharing_service.export_to_stix(test_threat)
        indicator = stix_bundle["objects"][0]

        assert "pattern" in indicator
        assert indicator["pattern_type"] == "stix"


# =============================================================================
# Test 4: Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_information_sharing(self, sharing_config):
        """Test create_information_sharing factory."""
        service = create_information_sharing(config=sharing_config)
        assert isinstance(service, DORAInformationSharing)
        assert service.config.provider_name == "Test Platform"

    def test_get_shareable_information_types(self):
        """Test get_shareable_information_types."""
        types = get_shareable_information_types()
        assert isinstance(types, set)
        assert "indicators_of_compromise" in types

    def test_get_tlp_definitions(self):
        """Test get_tlp_definitions."""
        definitions = get_tlp_definitions()
        assert isinstance(definitions, dict)
        assert "TLP:RED" in definitions

    def test_get_community_types(self):
        """Test get_community_types."""
        types = get_community_types()
        assert isinstance(types, list)
        assert "fs_isac" in types

    def test_get_sharing_channels(self):
        """Test get_sharing_channels."""
        channels = get_sharing_channels()
        assert isinstance(channels, list)
        assert "portal" in channels

    def test_get_tlp_levels(self):
        """Test get_tlp_levels."""
        levels = get_tlp_levels()
        assert isinstance(levels, list)
        assert "tlp_amber" in levels

    def test_create_sharing_community(self):
        """Test create_sharing_community factory."""
        community = create_sharing_community(
            name="Factory Test ISAC",
            community_type=CommunityType.FS_ISAC,
            country="US",
            contact_email="test@factory.example",
            trust_level=80,
        )
        assert community.name == "Factory Test ISAC"
        assert community.trust_level == 80

    def test_create_cyber_threat(self):
        """Test create_cyber_threat factory."""
        threat = create_cyber_threat(
            title="Factory Threat",
            description="Test description",
            severity=ThreatSeverity.CRITICAL,
            iocs=["192.168.1.1", "malware.com"],
        )
        assert threat.title == "Factory Threat"
        assert threat.severity == ThreatSeverity.CRITICAL
        assert len(threat.indicators_of_compromise) == 2

    def test_create_sharing_policy(self):
        """Test create_sharing_policy factory."""
        policy = create_sharing_policy(
            default_tlp=TLPLevel.TLP_GREEN,
            require_gdpr_review=False,
        )
        assert policy.default_tlp == TLPLevel.TLP_GREEN
        assert policy.require_gdpr_review is False


# =============================================================================
# Test 5: Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_threat_description(self, sharing_service, active_community):
        """Test handling threat with empty description."""
        sharing_service.register_community(active_community)
        threat = CyberThreatIntelligence(
            title="Empty Description",
            description="",
            information_types={"indicators_of_compromise"},
        )
        result = sharing_service.share_threat_intelligence(threat, active_community)
        assert result is not None

    def test_empty_iocs(self, sharing_service, active_community):
        """Test handling threat with no IOCs."""
        sharing_service.register_community(active_community)
        threat = CyberThreatIntelligence(
            title="No IOCs",
            description="Test",
            indicators_of_compromise=[],
        )
        result = sharing_service.share_threat_intelligence(threat, active_community)
        assert result is not None

    def test_very_long_description(self, sharing_service, active_community):
        """Test handling threat with very long description."""
        sharing_service.register_community(active_community)
        threat = CyberThreatIntelligence(
            title="Long Description",
            description="A" * 10000,
            information_types={"indicators_of_compromise"},
        )
        result = sharing_service.share_threat_intelligence(threat, active_community)
        assert result is not None

    def test_special_characters_in_keywords(self, sharing_service, active_community):
        """Test sanitization handles special characters."""
        sharing_service.register_community(active_community)
        threat = CyberThreatIntelligence(
            title="Special Chars",
            description="Found client_name (test) with [brackets] and $pecial chars",
            information_types={"indicators_of_compromise"},
        )
        sanitized = sharing_service.sanitize_threat(threat)
        assert "[REDACTED]" in sanitized.description

    def test_multiple_communities(self, sharing_service, test_threat):
        """Test sharing to multiple communities."""
        communities = []
        for i in range(3):
            community = create_sharing_community(
                name=f"Test Community {i}",
                community_type=CommunityType.PRIVATE_EXCHANGE,
                country="EU",
                contact_email=f"test{i}@example.com",
            )
            community.membership_status = MembershipStatus.ACTIVE
            community.joined_at = datetime.now(timezone.utc)
            sharing_service.register_community(community)
            communities.append(community)

        results = []
        for community in communities:
            result = sharing_service.share_threat_intelligence(test_threat, community)
            results.append(result)

        assert len(results) == 3
        # Check audit log has entries for all shares
        audit_log = sharing_service.get_sharing_audit_log()
        assert len(audit_log) >= 3

    def test_concurrent_operations(self, sharing_service, active_community, test_threat):
        """Test that operations don't interfere with each other."""
        sharing_service.register_community(active_community)

        # Share and receive in sequence
        sharing_service.share_threat_intelligence(test_threat, active_community)

        other_threat = create_cyber_threat(
            title="Other Threat",
            description="Different threat",
        )
        sharing_service.receive_threat_intelligence(other_threat, active_community.community_id)

        stats = sharing_service.get_sharing_statistics()
        assert stats["outbound_shares"] >= 1
        assert stats["inbound_received"] >= 1


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_sharing_workflow(self, sharing_service, test_community, test_threat):
        """Test complete sharing workflow."""
        # 1. Join community
        joined = sharing_service.join_community(test_community)
        assert joined.is_active()

        # 2. Prepare NCA notification
        notification = sharing_service.notify_nca_of_participation(joined)
        assert notification is not None

        # 3. Share threat
        result = sharing_service.share_threat_intelligence(test_threat, joined)
        assert result.outcome in {SharingOutcome.SUCCESS, SharingOutcome.SANITISED}

        # 4. Check audit log
        audit = sharing_service.get_sharing_audit_log()
        assert len(audit) >= 1

        # 5. Generate compliance report
        report = sharing_service.generate_compliance_report()
        assert report["statistics"]["outbound_shares"] >= 1

    def test_receiving_and_processing_workflow(self, sharing_service, active_community):
        """Test receiving and processing intelligence workflow."""
        sharing_service.register_community(active_community)

        # 1. Receive multiple threats
        threats = []
        for i in range(5):
            threat = create_cyber_threat(
                title=f"Received Threat {i}",
                description=f"Description {i}",
            )
            sharing_service.receive_threat_intelligence(threat, active_community.community_id)
            threats.append(threat)

        # 2. Check statistics
        stats = sharing_service.get_sharing_statistics()
        assert stats["inbound_received"] >= 5

        # 3. Verify audit trail
        audit = sharing_service.get_sharing_audit_log(
            direction=IntelligenceDirection.INBOUND
        )
        assert len(audit) >= 5


# =============================================================================
# Test 6: Import Validation
# =============================================================================

class TestImportValidation:
    """Test that all imports work correctly."""

    def test_import_from_sharing_module(self):
        """Test imports from sharing module."""
        from services.dora_integration.sharing import (
            DORAInformationSharing,
            SharingCommunity,
            CyberThreatIntelligence,
        )
        assert DORAInformationSharing is not None
        assert SharingCommunity is not None
        assert CyberThreatIntelligence is not None

    def test_import_from_dora_integration(self):
        """Test imports from dora_integration root."""
        from services.dora_integration import (
            DORAInformationSharing,
            SharingCommunity,
            CyberThreatIntelligence,
            create_information_sharing,
        )
        assert DORAInformationSharing is not None
        assert create_information_sharing is not None

    def test_all_exports_available(self):
        """Test all __all__ exports are available."""
        from services.dora_integration import sharing

        for name in sharing.__all__:
            assert hasattr(sharing, name), f"Missing export: {name}"
