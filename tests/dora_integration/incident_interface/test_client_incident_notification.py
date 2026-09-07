# -*- coding: utf-8 -*-
"""
Tests for Client Incident Notification Module (Article 30(2)(d)).

Tests cover:
- Client registration and management
- Incident creation and management
- Notification sending and delivery
- SLA compliance tracking
- Update and resolution notifications
- Audit report generation
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from services.dora_integration.incident_interface.client_incident_notification import (
    ClientNotificationService,
    DORAClientNotification,
    ClientNotificationConfig,
    IncidentSeverity,
    NotificationStatus,
    NotificationChannel,
    IncidentCategory,
    ClientContact,
    IncidentNotification,
    IncidentUpdate,
    ClientIncident,
    create_client_notification_service,
    create_client_notification_system,
    get_notification_template,
)


class TestClientNotificationService:
    """Test suite for ClientNotificationService."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ClientNotificationConfig(
            critical_sla_minutes=30,
            high_sla_minutes=60,
            medium_sla_minutes=240,
            low_sla_minutes=1440,
            max_delivery_attempts=3,
            retry_interval_seconds=60,
            log_all_notifications=False,
        )

    @pytest.fixture
    def service(self, config):
        """Create service instance."""
        return ClientNotificationService(config)

    # =========================================================================
    # Client Management Tests
    # =========================================================================

    def test_register_client_basic(self, service):
        """Test basic client registration."""
        contact = service.register_client(
            client_id="CLIENT-001",
            client_name="Test Client",
            email="test@example.com",
        )

        assert contact.client_id == "CLIENT-001"
        assert contact.client_name == "Test Client"
        assert contact.email == "test@example.com"
        assert contact.is_active is True
        assert contact.preferred_channel == NotificationChannel.EMAIL

    def test_register_client_full_details(self, service):
        """Test client registration with all details."""
        contact = service.register_client(
            client_id="CLIENT-002",
            client_name="Full Details Client",
            email="full@example.com",
            webhook_url="https://webhook.example.com",
            api_endpoint="https://api.example.com/notify",
            phone="+1234567890",
            preferred_channel=NotificationChannel.WEBHOOK,
            is_critical_function=True,
            notification_sla_minutes=15,
        )

        assert contact.client_id == "CLIENT-002"
        assert contact.webhook_url == "https://webhook.example.com"
        assert contact.preferred_channel == NotificationChannel.WEBHOOK
        assert contact.is_critical_function is True
        assert contact.notification_sla_minutes == 15

    def test_get_client(self, service):
        """Test retrieving a registered client."""
        service.register_client(
            client_id="CLIENT-003",
            client_name="Retrievable Client",
            email="retrieve@example.com",
        )

        contact = service.get_client("CLIENT-003")
        assert contact is not None
        assert contact.client_name == "Retrievable Client"

    def test_get_client_not_found(self, service):
        """Test retrieving non-existent client."""
        contact = service.get_client("NONEXISTENT")
        assert contact is None

    def test_get_all_clients(self, service):
        """Test getting all clients."""
        service.register_client("C1", "Client 1", "c1@test.com")
        service.register_client("C2", "Client 2", "c2@test.com")

        clients = service.get_all_clients()
        assert len(clients) == 2

    def test_update_client(self, service):
        """Test updating client information."""
        service.register_client("CLIENT-004", "Original Name", "orig@test.com")

        updated = service.update_client(
            "CLIENT-004",
            email="updated@test.com",
            is_critical_function=True,
        )

        assert updated is not None
        assert updated.email == "updated@test.com"
        assert updated.is_critical_function is True

    # =========================================================================
    # Incident Management Tests
    # =========================================================================

    def test_create_incident(self, service):
        """Test incident creation."""
        incident = service.create_incident(
            title="Database Outage",
            description="Primary database server unavailable",
            severity=IncidentSeverity.HIGH,
            category=IncidentCategory.INFRASTRUCTURE_FAILURE,
            services_affected=["trading", "reporting"],
        )

        assert incident.incident_id.startswith("INC-")
        assert incident.title == "Database Outage"
        assert incident.severity == IncidentSeverity.HIGH
        assert incident.category == IncidentCategory.INFRASTRUCTURE_FAILURE
        assert len(incident.services_affected) == 2
        assert incident.status == "classified"

    def test_create_incident_with_affected_clients(self, service):
        """Test incident creation with specific affected clients."""
        incident = service.create_incident(
            title="API Gateway Issue",
            description="API gateway latency increased",
            severity=IncidentSeverity.MEDIUM,
            clients_affected=["CLIENT-A", "CLIENT-B"],
        )

        assert len(incident.clients_affected) == 2
        assert "CLIENT-A" in incident.clients_affected

    def test_get_incident(self, service):
        """Test retrieving an incident."""
        created = service.create_incident(
            title="Test Incident",
            description="Test description",
            severity=IncidentSeverity.LOW,
        )

        retrieved = service.get_incident(created.incident_id)
        assert retrieved is not None
        assert retrieved.title == "Test Incident"

    def test_update_incident_status(self, service):
        """Test updating incident status."""
        incident = service.create_incident(
            title="Status Update Test",
            description="Testing status updates",
            severity=IncidentSeverity.MEDIUM,
        )

        updated = service.update_incident_status(
            incident.incident_id,
            status="mitigating",
        )

        assert updated.status == "mitigating"
        assert updated.mitigated_at is not None

    def test_update_incident_to_resolved(self, service):
        """Test resolving an incident."""
        incident = service.create_incident(
            title="Resolution Test",
            description="Testing resolution",
            severity=IncidentSeverity.HIGH,
        )

        resolved = service.update_incident_status(
            incident.incident_id,
            status="resolved",
            root_cause="Hardware failure",
            resolution_summary="Replaced faulty component",
        )

        assert resolved.status == "resolved"
        assert resolved.resolved_at is not None
        assert resolved.root_cause == "Hardware failure"

    # =========================================================================
    # Notification Tests
    # =========================================================================

    def test_notify_affected_clients(self, service):
        """Test sending notifications to affected clients."""
        # Register clients
        service.register_client("C1", "Client 1", "c1@test.com")
        service.register_client("C2", "Client 2", "c2@test.com")

        # Create incident
        incident = service.create_incident(
            title="Service Disruption",
            description="Services are degraded",
            severity=IncidentSeverity.HIGH,
        )

        # Send notifications
        notifications = service.notify_affected_clients(
            incident.incident_id,
            estimated_impact="Trading functionality affected",
            client_action_required="No action required",
        )

        assert len(notifications) == 2
        for notif in notifications:
            assert notif.status == NotificationStatus.SENT
            assert notif.notification_sent_at is not None

    def test_notification_sla_tracking(self, service):
        """Test SLA tracking for notifications."""
        service.register_client(
            "SLA-CLIENT",
            "SLA Test Client",
            "sla@test.com",
            is_critical_function=True,
            notification_sla_minutes=30,
        )

        incident = service.create_incident(
            title="SLA Test",
            description="Testing SLA",
            severity=IncidentSeverity.CRITICAL,
        )

        notifications = service.notify_affected_clients(incident.incident_id)

        assert len(notifications) == 1
        notif = notifications[0]
        assert notif.sla_deadline is not None
        # Sent immediately, so SLA should be met
        assert notif.sla_met is True

    def test_notification_severity_filter(self, service):
        """Test notification filtering by severity preference."""
        service.register_client(
            "FILTER-CLIENT",
            "Filter Test Client",
            "filter@test.com",
        )
        # Default severity filter is ["critical", "high", "medium"]

        # Low severity incident should not trigger notification
        incident = service.create_incident(
            title="Low Priority",
            description="Low priority issue",
            severity=IncidentSeverity.LOW,
        )

        notifications = service.notify_affected_clients(incident.incident_id)
        # Client's severity filter doesn't include "low"
        assert len(notifications) == 0

    # =========================================================================
    # Update and Resolution Tests
    # =========================================================================

    def test_send_incident_update(self, service):
        """Test sending incident updates."""
        service.register_client("UPDATE-CLIENT", "Update Client", "update@test.com")

        incident = service.create_incident(
            title="Update Test",
            description="Testing updates",
            severity=IncidentSeverity.HIGH,
        )

        service.notify_affected_clients(incident.incident_id)

        updates = service.send_incident_update(
            incident.incident_id,
            update_title="Progress Update",
            update_description="Issue is being investigated",
            update_type="status_change",
            new_status="mitigating",
        )

        assert len(updates) == 1
        assert updates[0].update_type == "status_change"
        assert updates[0].new_status == "mitigating"

    def test_send_resolution_notification(self, service):
        """Test sending resolution notifications."""
        service.register_client("RES-CLIENT", "Resolution Client", "res@test.com")

        incident = service.create_incident(
            title="Resolution Test",
            description="Testing resolution notifications",
            severity=IncidentSeverity.HIGH,
        )

        service.notify_affected_clients(incident.incident_id)

        resolution_updates = service.send_resolution_notification(
            incident.incident_id,
            root_cause="Configuration error",
            remediation_actions=["Corrected configuration", "Restarted services"],
            prevention_measures=["Added automated config validation"],
        )

        assert len(resolution_updates) == 1
        incident_updated = service.get_incident(incident.incident_id)
        assert incident_updated.status == "resolved"

    def test_record_acknowledgment(self, service):
        """Test recording client acknowledgment."""
        service.register_client("ACK-CLIENT", "Ack Client", "ack@test.com")

        incident = service.create_incident(
            title="Ack Test",
            description="Testing acknowledgment",
            severity=IncidentSeverity.HIGH,
        )

        notifications = service.notify_affected_clients(incident.incident_id)
        notif_id = notifications[0].notification_id

        acked = service.record_acknowledgment(
            notif_id,
            acknowledged_by="John Smith",
            response="Understood, monitoring situation",
        )

        assert acked.status == NotificationStatus.ACKNOWLEDGED
        assert acked.acknowledged_by == "John Smith"
        assert acked.client_response == "Understood, monitoring situation"

    # =========================================================================
    # Reporting Tests
    # =========================================================================

    def test_get_notification_summary(self, service):
        """Test getting notification summary."""
        service.register_client("SUM-CLIENT", "Summary Client", "sum@test.com")

        incident = service.create_incident(
            title="Summary Test",
            description="Testing summary",
            severity=IncidentSeverity.HIGH,
        )

        service.notify_affected_clients(incident.incident_id)

        summary = service.get_notification_summary()

        assert summary["total_notifications"] == 1
        assert summary["by_severity"]["high"] == 1
        assert summary["active_incidents"] == 1

    def test_get_client_notification_history(self, service):
        """Test getting client notification history."""
        service.register_client("HIST-CLIENT", "History Client", "hist@test.com")

        for i in range(3):
            incident = service.create_incident(
                title=f"Incident {i}",
                description=f"Description {i}",
                severity=IncidentSeverity.HIGH,
            )
            service.notify_affected_clients(incident.incident_id)

        history = service.get_client_notification_history("HIST-CLIENT")
        assert len(history) == 3

    def test_generate_audit_report(self, service):
        """Test generating audit report."""
        service.register_client("AUDIT-CLIENT", "Audit Client", "audit@test.com")

        incident = service.create_incident(
            title="Audit Test",
            description="Testing audit",
            severity=IncidentSeverity.CRITICAL,
        )

        service.notify_affected_clients(incident.incident_id)

        report = service.generate_audit_report(client_id="AUDIT-CLIENT")

        assert "report_generated" in report
        assert len(report["notifications"]) == 1
        assert report["summary"]["total_notifications"] == 1

    def test_export_incident_data_for_client(self, service):
        """Test exporting incident data for client ROI."""
        service.register_client("EXPORT-CLIENT", "Export Client", "export@test.com")

        incident = service.create_incident(
            title="Export Test",
            description="Testing export",
            severity=IncidentSeverity.HIGH,
        )

        service.notify_affected_clients(incident.incident_id)

        export = service.export_incident_data_for_client(
            incident.incident_id,
            "EXPORT-CLIENT",
        )

        assert export["export_type"] == "client_roi_data"
        assert export["incident"]["incident_id"] == incident.incident_id


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_client_notification_service(self):
        """Test service factory function."""
        service = create_client_notification_service()
        assert isinstance(service, ClientNotificationService)

    def test_create_client_notification_system_legacy(self):
        """Test legacy factory function."""
        service = create_client_notification_system()
        assert isinstance(service, ClientNotificationService)


class TestNotificationTemplates:
    """Test notification templates."""

    def test_get_notification_template_critical(self):
        """Test critical severity template."""
        template = get_notification_template(IncidentSeverity.CRITICAL)
        assert "[CRITICAL]" in template["subject"]
        assert template["priority"] == "urgent"

    def test_get_notification_template_high(self):
        """Test high severity template."""
        template = get_notification_template(IncidentSeverity.HIGH)
        assert "[HIGH]" in template["subject"]
        assert template["priority"] == "high"

    def test_get_notification_template_medium(self):
        """Test medium severity template."""
        template = get_notification_template(IncidentSeverity.MEDIUM)
        assert "[MEDIUM]" in template["subject"]
        assert template["priority"] == "normal"

    def test_get_notification_template_low(self):
        """Test low severity template."""
        template = get_notification_template(IncidentSeverity.LOW)
        assert "[LOW]" in template["subject"]
        assert template["priority"] == "low"


class TestDataStructures:
    """Test data structures."""

    def test_client_contact_defaults(self):
        """Test ClientContact default values."""
        contact = ClientContact(client_id="C1", client_name="Test")
        assert contact.is_active is True
        assert contact.preferred_channel == NotificationChannel.EMAIL
        assert len(contact.severity_filter) == 3

    def test_incident_notification_auto_id(self):
        """Test IncidentNotification auto ID generation."""
        notif = IncidentNotification()
        assert notif.notification_id.startswith("NTF-")
        assert notif.notification_created_at is not None

    def test_client_incident_auto_id(self):
        """Test ClientIncident auto ID generation."""
        incident = ClientIncident()
        assert incident.incident_id.startswith("INC-")
        assert incident.detected_at is not None


class TestBackwardCompatibility:
    """Test backward compatibility."""

    def test_dora_client_notification_alias(self):
        """Test DORAClientNotification alias."""
        service = DORAClientNotification()
        assert isinstance(service, ClientNotificationService)
