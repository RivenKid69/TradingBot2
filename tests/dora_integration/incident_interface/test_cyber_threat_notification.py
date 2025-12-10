# -*- coding: utf-8 -*-
"""
Tests for Cyber Threat Notification Module (Article 19(4)).

Tests cover:
- Threat recording and management
- Significance assessment
- Notification workflow
- Indicator management
- Statistics and export
"""

import pytest
from datetime import datetime, timezone

from services.dora_integration.incident_interface.cyber_threat_notification import (
    CyberThreatNotificationService,
    CyberThreatNotificationConfig,
    ThreatCategory,
    ThreatActorType,
    ThreatSeverity,
    ThreatStatus,
    ThreatSignificance,
    ThreatIndicator,
    CyberThreat,
    ThreatSignificanceAssessment,
    ThreatNotification,
    create_cyber_threat_notification_service,
    get_threat_categories,
    get_threat_severities,
)


class TestCyberThreatNotificationService:
    """Test suite for CyberThreatNotificationService."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return CyberThreatNotificationConfig(
            entity_lei="549300TEST000000",
            entity_name="Test ICT Provider",
            auto_assess_significance=False,  # Disable for predictable tests
            require_approval_for_notification=True,
            log_all_threats=False,
        )

    @pytest.fixture
    def service(self, config):
        """Create service instance."""
        return CyberThreatNotificationService(config)

    # =========================================================================
    # Threat Management Tests
    # =========================================================================

    def test_record_threat(self, service):
        """Test recording a threat."""
        threat = service.record_threat(
            title="New Ransomware Campaign",
            description="Ransomware targeting financial sector",
            category=ThreatCategory.RANSOMWARE,
            severity=ThreatSeverity.HIGH,
            threat_actor_type=ThreatActorType.CYBERCRIMINAL,
            attack_vector="Phishing email",
            targeted_sectors=["financial"],
            potential_impact="Service disruption",
            source="internal",
        )

        assert threat.threat_id.startswith("THR-")
        assert threat.title == "New Ransomware Campaign"
        assert threat.category == ThreatCategory.RANSOMWARE
        assert threat.severity == ThreatSeverity.HIGH

    def test_record_threat_with_indicators(self, service):
        """Test recording threat with IoCs."""
        indicator = ThreatIndicator(
            indicator_type="ip",
            indicator_value="192.0.2.1",
            description="C2 server",
            confidence="high",
        )

        threat = service.record_threat(
            title="APT Activity",
            description="APT activity detected",
            category=ThreatCategory.APT,
            severity=ThreatSeverity.CRITICAL,
            indicators=[indicator],
        )

        assert len(threat.indicators) == 1
        assert threat.indicators[0].indicator_value == "192.0.2.1"

    def test_update_threat(self, service):
        """Test updating a threat."""
        threat = service.record_threat(
            title="Original Title",
            description="Original",
            category=ThreatCategory.MALWARE,
            severity=ThreatSeverity.MEDIUM,
        )

        updated = service.update_threat(
            threat.threat_id,
            title="Updated Title",
            status=ThreatStatus.CONFIRMED,
        )

        assert updated.title == "Updated Title"
        assert updated.status == ThreatStatus.CONFIRMED

    def test_add_indicator(self, service):
        """Test adding indicator to threat."""
        threat = service.record_threat(
            title="Test Threat",
            description="Test",
            category=ThreatCategory.MALWARE,
            severity=ThreatSeverity.LOW,
        )

        indicator = ThreatIndicator(
            indicator_type="hash",
            indicator_value="abc123",
            description="Malware hash",
        )

        updated = service.add_indicator(threat.threat_id, indicator)
        assert len(updated.indicators) == 1

    def test_get_threat(self, service):
        """Test retrieving threat."""
        threat = service.record_threat(
            title="Retrievable Threat",
            description="Test",
            category=ThreatCategory.PHISHING,
            severity=ThreatSeverity.MEDIUM,
        )

        retrieved = service.get_threat(threat.threat_id)
        assert retrieved is not None
        assert retrieved.title == "Retrievable Threat"

    def test_get_active_threats(self, service):
        """Test getting active threats."""
        service.record_threat(
            title="Active Threat",
            description="Active",
            category=ThreatCategory.MALWARE,
            severity=ThreatSeverity.HIGH,
        )

        resolved = service.record_threat(
            title="Resolved Threat",
            description="Resolved",
            category=ThreatCategory.MALWARE,
            severity=ThreatSeverity.LOW,
        )
        service.update_threat(resolved.threat_id, status=ThreatStatus.RESOLVED)

        active = service.get_active_threats()
        assert any(t.title == "Active Threat" for t in active)

    def test_get_threats_by_severity(self, service):
        """Test filtering threats by severity."""
        service.record_threat(
            title="Critical Threat",
            description="Critical",
            category=ThreatCategory.APT,
            severity=ThreatSeverity.CRITICAL,
        )

        service.record_threat(
            title="Low Threat",
            description="Low",
            category=ThreatCategory.SPAM,
            severity=ThreatSeverity.LOW,
        )

        critical = service.get_threats_by_severity(ThreatSeverity.CRITICAL)
        assert len(critical) == 1
        assert critical[0].title == "Critical Threat"

    def test_get_threats_by_category(self, service):
        """Test filtering threats by category."""
        service.record_threat(
            title="Ransomware Threat",
            description="Ransomware",
            category=ThreatCategory.RANSOMWARE,
            severity=ThreatSeverity.HIGH,
        )

        ransomware = service.get_threats_by_category(ThreatCategory.RANSOMWARE)
        assert len(ransomware) == 1

    # =========================================================================
    # Significance Assessment Tests
    # =========================================================================

    def test_assess_significance_not_significant(self, service):
        """Test assessment for non-significant threat."""
        threat = service.record_threat(
            title="Minor Threat",
            description="Minor",
            category=ThreatCategory.SPAM,
            severity=ThreatSeverity.LOW,
        )

        assessment = service.assess_significance(
            threat.threat_id,
            impacts_critical_functions=False,
            exploitation_likelihood="low",
        )

        assert assessment.significance == ThreatSignificance.NOT_SIGNIFICANT
        assert assessment.recommend_notification is False

    def test_assess_significance_significant(self, service):
        """Test assessment for significant threat."""
        threat = service.record_threat(
            title="Significant Threat",
            description="Significant",
            category=ThreatCategory.RANSOMWARE,
            severity=ThreatSeverity.HIGH,
        )

        # Need enough factors to score >= 60 for SIGNIFICANT:
        # - impacts_critical_functions: 15
        # - impacts_other_financial_entities: 10
        # - impacts_clients: 5
        # - exploitation_likelihood="high": 10
        # - active_exploitation_observed: 5
        # - severity HIGH: 7
        # - potential_affected_entities > 10: +5
        # Total: 57 (still POTENTIALLY_SIGNIFICANT)
        # Need: potential_affected_entities > 100: +10 = 62
        assessment = service.assess_significance(
            threat.threat_id,
            impacts_critical_functions=True,
            impacts_other_financial_entities=True,
            impacts_clients=True,
            exploitation_likelihood="high",
            active_exploitation_observed=True,
            potential_affected_entities=150,  # Added for > 100 threshold
        )

        assert assessment.significance in (
            ThreatSignificance.SIGNIFICANT,
            ThreatSignificance.HIGHLY_SIGNIFICANT,
        )
        assert assessment.recommend_notification is True

    def test_assess_significance_highly_significant(self, service):
        """Test assessment for highly significant threat."""
        threat = service.record_threat(
            title="Critical Threat",
            description="Critical",
            category=ThreatCategory.APT,
            severity=ThreatSeverity.CRITICAL,
        )

        assessment = service.assess_significance(
            threat.threat_id,
            impacts_financial_system_stability=True,
            impacts_critical_functions=True,
            impacts_other_financial_entities=True,
            impacts_clients=True,
            potential_affected_entities=200,
            exploitation_likelihood="very_high",
            active_exploitation_observed=True,
        )

        assert assessment.significance == ThreatSignificance.HIGHLY_SIGNIFICANT
        assert assessment.notification_urgency == "urgent"

    def test_get_assessment(self, service):
        """Test retrieving assessment."""
        threat = service.record_threat(
            title="Assessment Test",
            description="Test",
            category=ThreatCategory.MALWARE,
            severity=ThreatSeverity.MEDIUM,
        )

        assessment = service.assess_significance(threat.threat_id)

        retrieved = service.get_assessment(assessment.assessment_id)
        assert retrieved is not None

    def test_get_assessment_for_threat(self, service):
        """Test getting assessment by threat ID."""
        threat = service.record_threat(
            title="Threat Assessment Test",
            description="Test",
            category=ThreatCategory.MALWARE,
            severity=ThreatSeverity.MEDIUM,
        )

        service.assess_significance(threat.threat_id)

        retrieved = service.get_assessment_for_threat(threat.threat_id)
        assert retrieved is not None

    # =========================================================================
    # Notification Workflow Tests
    # =========================================================================

    def test_create_notification(self, service):
        """Test creating notification."""
        threat = service.record_threat(
            title="Notification Test",
            description="Testing notifications",
            category=ThreatCategory.RANSOMWARE,
            severity=ThreatSeverity.HIGH,
        )

        notification = service.create_notification(
            threat.threat_id,
            summary="Summary of threat",
            detailed_description="Detailed description",
            mitigations_applied=["Blocked IPs"],
            recommended_mitigations=["Update AV signatures"],
            contact_person_name="John Smith",
        )

        assert notification.notification_id.startswith("TNOTIF-")
        assert notification.threat_id == threat.threat_id
        assert notification.summary == "Summary of threat"

    def test_approve_notification(self, service):
        """Test approving notification."""
        threat = service.record_threat(
            title="Approval Test",
            description="Test",
            category=ThreatCategory.MALWARE,
            severity=ThreatSeverity.HIGH,
        )

        notification = service.create_notification(threat.threat_id)

        approved = service.approve_notification(
            notification.notification_id,
            approved_by="Security Officer",
        )

        from services.dora_integration.incident_interface.cyber_threat_notification import (
            NotificationStatus,
        )
        assert approved.status == NotificationStatus.APPROVED_FOR_NOTIFICATION

    def test_decline_notification(self, service):
        """Test declining notification."""
        threat = service.record_threat(
            title="Decline Test",
            description="Test",
            category=ThreatCategory.MALWARE,
            severity=ThreatSeverity.LOW,
        )

        notification = service.create_notification(threat.threat_id)

        declined = service.decline_notification(
            notification.notification_id,
            declined_by="Security Officer",
            reason="Not significant enough",
        )

        from services.dora_integration.incident_interface.cyber_threat_notification import (
            NotificationStatus,
        )
        assert declined.status == NotificationStatus.NOTIFICATION_DECLINED

    def test_submit_notification(self, service):
        """Test submitting notification."""
        threat = service.record_threat(
            title="Submit Test",
            description="Test",
            category=ThreatCategory.RANSOMWARE,
            severity=ThreatSeverity.CRITICAL,
        )

        notification = service.create_notification(threat.threat_id)
        service.approve_notification(notification.notification_id, "Approver")

        submitted = service.submit_notification(
            notification.notification_id,
            authority_ids=["NCA-DE"],
            submitted_by="Security Team",
        )

        from services.dora_integration.incident_interface.cyber_threat_notification import (
            NotificationStatus,
        )
        assert submitted.status == NotificationStatus.NOTIFIED
        assert submitted.submitted_at is not None

    def test_get_pending_notifications(self, service):
        """Test getting pending notifications."""
        threat = service.record_threat(
            title="Pending Test",
            description="Test",
            category=ThreatCategory.MALWARE,
            severity=ThreatSeverity.HIGH,
        )

        service.create_notification(threat.threat_id)

        pending = service.get_pending_notifications()
        assert len(pending) >= 1

    # =========================================================================
    # Indicator Management Tests
    # =========================================================================

    def test_create_indicator(self, service):
        """Test creating standalone indicator."""
        indicator = service.create_indicator(
            indicator_type="domain",
            indicator_value="malicious.example.com",
            description="Known C2 domain",
            threat_types=[ThreatCategory.MALWARE],
            confidence="high",
            source="threat_intel",
        )

        assert indicator.indicator_id.startswith("IOC-")
        assert indicator.indicator_type == "domain"

    def test_search_indicators(self, service):
        """Test searching indicators."""
        service.create_indicator(
            indicator_type="ip",
            indicator_value="192.0.2.100",
            description="Malicious IP",
        )

        service.create_indicator(
            indicator_type="domain",
            indicator_value="evil.com",
            description="Malicious domain",
        )

        results = service.search_indicators(indicator_type="ip")
        assert len(results) == 1
        assert results[0].indicator_value == "192.0.2.100"

    def test_search_indicators_by_value(self, service):
        """Test searching indicators by value."""
        service.create_indicator(
            indicator_type="ip",
            indicator_value="192.0.2.50",
            description="Test",
        )

        results = service.search_indicators(indicator_value="192.0.2")
        assert len(results) >= 1

    # =========================================================================
    # Statistics and Export Tests
    # =========================================================================

    def test_get_threat_statistics(self, service):
        """Test threat statistics."""
        service.record_threat(
            title="Stats Threat 1",
            description="Test",
            category=ThreatCategory.RANSOMWARE,
            severity=ThreatSeverity.HIGH,
        )

        service.record_threat(
            title="Stats Threat 2",
            description="Test",
            category=ThreatCategory.PHISHING,
            severity=ThreatSeverity.MEDIUM,
        )

        stats = service.get_threat_statistics()

        assert stats["total_threats"] >= 2
        assert "by_category" in stats
        assert "by_severity" in stats

    def test_export_threat(self, service):
        """Test exporting threat data."""
        threat = service.record_threat(
            title="Export Test",
            description="Test",
            category=ThreatCategory.MALWARE,
            severity=ThreatSeverity.HIGH,
        )

        export = service.export_threat(threat.threat_id)

        assert export["article_reference"] == "Article 19(4)"
        assert "threat" in export


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_cyber_threat_notification_service(self):
        """Test service factory."""
        service = create_cyber_threat_notification_service()
        assert isinstance(service, CyberThreatNotificationService)

    def test_get_threat_categories(self):
        """Test categories list."""
        categories = get_threat_categories()
        assert ThreatCategory.RANSOMWARE in categories
        assert ThreatCategory.MALWARE in categories

    def test_get_threat_severities(self):
        """Test severities list."""
        severities = get_threat_severities()
        assert ThreatSeverity.CRITICAL in severities
        assert ThreatSeverity.LOW in severities


class TestDataStructures:
    """Test data structures."""

    def test_threat_indicator_auto_id(self):
        """Test ThreatIndicator auto ID."""
        indicator = ThreatIndicator()
        assert indicator.indicator_id.startswith("IOC-")
        assert indicator.first_seen is not None

    def test_cyber_threat_auto_id(self):
        """Test CyberThreat auto ID."""
        threat = CyberThreat()
        assert threat.threat_id.startswith("THR-")
        assert threat.detected_at is not None

    def test_threat_significance_assessment_auto_id(self):
        """Test ThreatSignificanceAssessment auto ID."""
        assessment = ThreatSignificanceAssessment()
        assert assessment.assessment_id.startswith("TSA-")

    def test_threat_notification_auto_id(self):
        """Test ThreatNotification auto ID."""
        notification = ThreatNotification()
        assert notification.notification_id.startswith("TNOTIF-")
