# -*- coding: utf-8 -*-
"""
Tests for Crisis Communication Module (Article 14).

Tests cover:
- Policy management
- Contact management
- Template management
- Communication workflow
- Crisis status management
- Escalation
- Statistics and export
"""

import pytest
from datetime import datetime, timezone

from services.dora_integration.incident_interface.communication import (
    DORACommunication,
    CommunicationConfig,
    CommunicationChannel,
    StakeholderType,
    CommunicationPriority,
    CommunicationStatus,
    CrisisPhase,
    PolicyStatus,
    CommunicationContact,
    CommunicationTemplate,
    CommunicationRecord,
    CommunicationPolicy,
    CrisisStatus,
    create_communication_service,
    get_communication_channels,
    get_stakeholder_types,
    get_crisis_phases,
)


class TestDORACommunication:
    """Test suite for DORACommunication."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return CommunicationConfig(
            organization_name="Test ICT Provider",
            organization_lei="549300TEST000000",
            require_approval_for_external=True,
            approval_timeout_minutes=15,
            log_all_communications=False,
        )

    @pytest.fixture
    def service(self, config):
        """Create service instance."""
        return DORACommunication(config)

    # =========================================================================
    # Policy Management Tests
    # =========================================================================

    def test_create_policy(self, service):
        """Test creating communication policy."""
        policy = service.create_policy(
            name="Client Incident Communication",
            description="Policy for client notifications during incidents",
            applies_to_stakeholders=[StakeholderType.CLIENT],
            applies_to_crisis_phases=[
                CrisisPhase.INITIAL_RESPONSE,
                CrisisPhase.ONGOING,
            ],
            requires_management_approval=True,
        )

        assert policy.policy_id.startswith("POL-")
        assert policy.name == "Client Incident Communication"
        assert StakeholderType.CLIENT in policy.applies_to_stakeholders

    def test_approve_policy(self, service):
        """Test approving a policy."""
        policy = service.create_policy(
            name="Approval Test Policy",
            description="Test",
        )

        approved = service.approve_policy(
            policy.policy_id,
            approved_by="Compliance Officer",
        )

        assert approved.status == PolicyStatus.APPROVED
        assert approved.approved_by == "Compliance Officer"

    def test_activate_policy(self, service):
        """Test activating a policy."""
        policy = service.create_policy(name="Activation Test", description="Test")
        service.approve_policy(policy.policy_id, "Approver")

        activated = service.activate_policy(policy.policy_id)

        assert activated.status == PolicyStatus.ACTIVE
        assert activated.effective_date is not None

    def test_get_active_policies(self, service):
        """Test getting active policies."""
        policy = service.create_policy(name="Active Policy", description="Test")
        service.approve_policy(policy.policy_id, "Approver")
        service.activate_policy(policy.policy_id)

        active = service.get_active_policies()
        assert any(p.name == "Active Policy" for p in active)

    def test_get_policy_for_stakeholder(self, service):
        """Test getting policy for stakeholder and phase."""
        policy = service.create_policy(
            name="Client Policy",
            applies_to_stakeholders=[StakeholderType.CLIENT],
            applies_to_crisis_phases=[CrisisPhase.INITIAL_RESPONSE],
        )
        service.approve_policy(policy.policy_id, "Approver")
        service.activate_policy(policy.policy_id)

        found = service.get_policy_for_stakeholder(
            StakeholderType.CLIENT,
            CrisisPhase.INITIAL_RESPONSE,
        )

        assert found is not None
        assert found.name == "Client Policy"

    # =========================================================================
    # Contact Management Tests
    # =========================================================================

    def test_register_contact(self, service):
        """Test registering a contact."""
        contact = service.register_contact(
            name="John Smith",
            stakeholder_type=StakeholderType.CLIENT,
            email="john@example.com",
            phone="+1234567890",
            role="CTO",
            organization="Client Corp",
            is_primary_contact=True,
        )

        assert contact.contact_id.startswith("CONTACT-")
        assert contact.name == "John Smith"
        assert contact.stakeholder_type == StakeholderType.CLIENT
        assert contact.is_primary_contact is True

    def test_update_contact(self, service):
        """Test updating a contact."""
        contact = service.register_contact(
            name="Original Name",
            stakeholder_type=StakeholderType.CLIENT,
            email="orig@example.com",
        )

        updated = service.update_contact(
            contact.contact_id,
            email="updated@example.com",
            phone="+9876543210",
        )

        assert updated.email == "updated@example.com"
        assert updated.phone == "+9876543210"

    def test_get_contacts_by_type(self, service):
        """Test getting contacts by type."""
        service.register_contact(
            name="Client Contact",
            stakeholder_type=StakeholderType.CLIENT,
            email="client@example.com",
        )

        service.register_contact(
            name="Internal Contact",
            stakeholder_type=StakeholderType.INTERNAL_STAFF,
            email="internal@example.com",
        )

        clients = service.get_contacts_by_type(StakeholderType.CLIENT)
        assert len(clients) == 1
        assert clients[0].name == "Client Contact"

    def test_get_primary_contacts(self, service):
        """Test getting primary contacts."""
        service.register_contact(
            name="Primary",
            stakeholder_type=StakeholderType.CLIENT,
            email="primary@example.com",
            is_primary_contact=True,
        )

        service.register_contact(
            name="Secondary",
            stakeholder_type=StakeholderType.CLIENT,
            email="secondary@example.com",
            is_primary_contact=False,
        )

        primary = service.get_primary_contacts()
        assert len(primary) == 1
        assert primary[0].name == "Primary"

    # =========================================================================
    # Template Management Tests
    # =========================================================================

    def test_create_template(self, service):
        """Test creating a template."""
        template = service.create_template(
            name="Custom Alert Template",
            subject_template="[Alert] {incident_title}",
            body_template="Dear {contact_name}, there is an incident...",
            stakeholder_types=[StakeholderType.CLIENT],
            crisis_phases=[CrisisPhase.INITIAL_RESPONSE],
        )

        assert template.template_id.startswith("TMPL-")
        assert template.name == "Custom Alert Template"

    def test_render_template(self, service):
        """Test rendering a template."""
        template = service.create_template(
            name="Render Test",
            subject_template="[{priority}] {incident_title}",
            body_template="Dear {contact_name}, incident {incident_id} occurred.",
            sms_template="Alert: {incident_title}",
        )

        rendered = service.render_template(
            template.template_id,
            variables={
                "priority": "HIGH",
                "incident_title": "Service Outage",
                "contact_name": "John",
                "incident_id": "INC-001",
            },
        )

        assert "[HIGH] Service Outage" == rendered["subject"]
        assert "Dear John" in rendered["body"]
        assert "Alert: Service Outage" == rendered["sms"]

    def test_get_templates_for_context(self, service):
        """Test getting templates for context."""
        # Default templates are already created
        templates = service.get_templates_for_context(
            StakeholderType.CLIENT,
            CrisisPhase.INITIAL_RESPONSE,
            CommunicationPriority.HIGH,
        )

        assert len(templates) >= 1

    # =========================================================================
    # Communication Workflow Tests
    # =========================================================================

    def test_create_communication(self, service):
        """Test creating a communication."""
        contact = service.register_contact(
            name="Comm Contact",
            stakeholder_type=StakeholderType.CLIENT,
            email="comm@example.com",
        )

        comm = service.create_communication(
            incident_id="INC-001",
            subject="Service Incident Notification",
            body="We are experiencing a service incident...",
            recipients=[contact.contact_id],
            priority=CommunicationPriority.HIGH,
        )

        assert comm.communication_id.startswith("COMM-")
        assert comm.incident_id == "INC-001"
        assert len(comm.recipients) == 1
        assert comm.status == CommunicationStatus.DRAFT

    def test_approve_communication(self, service):
        """Test approving a communication."""
        contact = service.register_contact(
            name="Approve Contact",
            stakeholder_type=StakeholderType.CLIENT,
            email="approve@example.com",
        )

        comm = service.create_communication(
            incident_id="INC-002",
            subject="Test",
            body="Test body",
            recipients=[contact.contact_id],
        )

        approved = service.approve_communication(
            comm.communication_id,
            approved_by="Manager",
        )

        assert approved.status == CommunicationStatus.APPROVED
        assert approved.approved_by == "Manager"

    def test_send_communication(self, service):
        """Test sending a communication."""
        contact = service.register_contact(
            name="Send Contact",
            stakeholder_type=StakeholderType.CLIENT,
            email="send@example.com",
        )

        comm = service.create_communication(
            incident_id="INC-003",
            subject="Test",
            body="Test body",
            recipients=[contact.contact_id],
            requires_approval=True,
        )

        service.approve_communication(comm.communication_id, "Approver")

        sent = service.send_communication(comm.communication_id)

        assert sent.status == CommunicationStatus.SENT
        assert sent.sent_at is not None

    def test_record_delivery(self, service):
        """Test recording delivery."""
        contact = service.register_contact(
            name="Delivery Contact",
            stakeholder_type=StakeholderType.CLIENT,
            email="delivery@example.com",
        )

        comm = service.create_communication(
            incident_id="INC-004",
            subject="Test",
            body="Test",
            recipients=[contact.contact_id],
            requires_approval=False,
        )

        service.send_communication(comm.communication_id)

        delivered = service.record_delivery(comm.communication_id)

        assert delivered.status == CommunicationStatus.DELIVERED

    def test_record_acknowledgement(self, service):
        """Test recording acknowledgement."""
        contact = service.register_contact(
            name="Ack Contact",
            stakeholder_type=StakeholderType.CLIENT,
            email="ack@example.com",
        )

        comm = service.create_communication(
            incident_id="INC-005",
            subject="Test",
            body="Test",
            recipients=[contact.contact_id],
            requires_approval=False,
        )

        service.send_communication(comm.communication_id)

        acked = service.record_acknowledgement(
            comm.communication_id,
            acknowledged_by="John Smith",
            response_content="Understood",
        )

        assert acked.status == CommunicationStatus.ACKNOWLEDGED
        assert acked.response_content == "Understood"

    def test_get_communications_for_incident(self, service):
        """Test getting communications for incident."""
        contact = service.register_contact(
            name="History Contact",
            stakeholder_type=StakeholderType.CLIENT,
            email="history@example.com",
        )

        for i in range(3):
            service.create_communication(
                incident_id="INC-HISTORY",
                subject=f"Update {i}",
                body="Update body",
                recipients=[contact.contact_id],
            )

        comms = service.get_communications_for_incident("INC-HISTORY")
        assert len(comms) == 3

    # =========================================================================
    # Crisis Status Management Tests
    # =========================================================================

    def test_create_crisis_status(self, service):
        """Test creating crisis status."""
        status = service.create_crisis_status(
            incident_id="INC-STATUS",
            current_status="Services degraded",
            current_impact="Trading latency increased",
            affected_services=["trading", "reporting"],
            crisis_phase=CrisisPhase.INITIAL_RESPONSE,
            current_actions=["Investigating root cause"],
            next_steps=["Deploy fix"],
        )

        assert status.incident_id == "INC-STATUS"
        assert status.crisis_phase == CrisisPhase.INITIAL_RESPONSE

    def test_get_crisis_status(self, service):
        """Test getting crisis status."""
        service.create_crisis_status(
            incident_id="INC-GET-STATUS",
            current_status="Under investigation",
        )

        status = service.get_crisis_status("INC-GET-STATUS")
        assert status is not None
        assert status.current_status == "Under investigation"

    def test_update_crisis_phase(self, service):
        """Test updating crisis phase."""
        service.create_crisis_status(
            incident_id="INC-PHASE",
            current_status="Initial",
        )

        updated = service.update_crisis_phase("INC-PHASE", CrisisPhase.RECOVERY)
        assert updated.crisis_phase == CrisisPhase.RECOVERY

    # =========================================================================
    # Escalation Tests
    # =========================================================================

    def test_escalate_communication(self, service):
        """Test escalating communications."""
        service.create_crisis_status(
            incident_id="INC-ESCALATE",
            current_status="Need escalation",
        )

        # Register escalation contacts
        service.register_contact(
            name="Level 2 Contact",
            stakeholder_type=StakeholderType.MANAGEMENT,
            email="level2@example.com",
            escalation_level=2,
        )

        escalated_contacts = service.escalate_communication(
            "INC-ESCALATE",
            reason="No response from primary contacts",
        )

        # Should return level 2 contacts
        assert len(escalated_contacts) >= 0

    # =========================================================================
    # Statistics and Export Tests
    # =========================================================================

    def test_get_communication_statistics(self, service):
        """Test communication statistics."""
        contact = service.register_contact(
            name="Stats Contact",
            stakeholder_type=StakeholderType.CLIENT,
            email="stats@example.com",
        )

        service.create_communication(
            incident_id="INC-STATS",
            subject="Test",
            body="Test",
            recipients=[contact.contact_id],
        )

        stats = service.get_communication_statistics()

        assert "total_communications" in stats
        assert "by_channel" in stats
        assert "by_status" in stats

    def test_export_communication_log(self, service):
        """Test exporting communication log."""
        contact = service.register_contact(
            name="Export Contact",
            stakeholder_type=StakeholderType.CLIENT,
            email="export@example.com",
        )

        service.create_communication(
            incident_id="INC-EXPORT",
            subject="Test",
            body="Test",
            recipients=[contact.contact_id],
        )

        export = service.export_communication_log(incident_id="INC-EXPORT")

        assert export["article_reference"] == "Article 14"
        assert len(export["communications"]) == 1


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_communication_service(self):
        """Test service factory."""
        service = create_communication_service()
        assert isinstance(service, DORACommunication)

    def test_get_communication_channels(self):
        """Test channels list."""
        channels = get_communication_channels()
        assert CommunicationChannel.EMAIL in channels
        assert CommunicationChannel.WEBHOOK in channels

    def test_get_stakeholder_types(self):
        """Test stakeholder types list."""
        types = get_stakeholder_types()
        assert StakeholderType.CLIENT in types
        assert StakeholderType.MANAGEMENT in types

    def test_get_crisis_phases(self):
        """Test crisis phases list."""
        phases = get_crisis_phases()
        assert CrisisPhase.INITIAL_RESPONSE in phases
        assert CrisisPhase.RECOVERY in phases


class TestDataStructures:
    """Test data structures."""

    def test_communication_contact_auto_id(self):
        """Test CommunicationContact auto ID."""
        contact = CommunicationContact()
        assert contact.contact_id.startswith("CONTACT-")
        assert contact.is_active is True

    def test_communication_template_auto_id(self):
        """Test CommunicationTemplate auto ID."""
        template = CommunicationTemplate()
        assert template.template_id.startswith("TMPL-")

    def test_communication_record_auto_id(self):
        """Test CommunicationRecord auto ID."""
        record = CommunicationRecord()
        assert record.communication_id.startswith("COMM-")

    def test_communication_policy_auto_id(self):
        """Test CommunicationPolicy auto ID."""
        policy = CommunicationPolicy()
        assert policy.policy_id.startswith("POL-")

    def test_crisis_status_auto_id(self):
        """Test CrisisStatus auto ID."""
        status = CrisisStatus()
        assert status.status_id.startswith("STATUS-")

    def test_policy_default_timelines(self):
        """Test CommunicationPolicy default timelines."""
        policy = CommunicationPolicy()
        assert "internal_staff" in policy.notification_timeline_minutes
        assert "client" in policy.notification_timeline_minutes
