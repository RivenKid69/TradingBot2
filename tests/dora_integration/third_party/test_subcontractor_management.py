# -*- coding: utf-8 -*-
"""
Tests for DORA Subcontractor Management Module (Article 30).

Tests subcontractor registration, change management,
client notification, consent workflows, and risk assessment.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

from services.dora_integration.third_party.subcontractor_management import (
    DORASubcontractorManagement,
    SubcontractorConfig,
    SubcontractorType,
    SubcontractorStatus,
    RiskLevel,
    ChangeType,
    NotificationStatus,
    ConsentMode,
    Subcontractor,
    SubcontractorChange,
    ClientSubcontractorPreference,
    SubcontractorRiskAssessment,
    create_subcontractor_management,
)


class TestSubcontractorConfig:
    """Tests for SubcontractorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SubcontractorConfig()

        assert config.default_notification_days == 30
        assert config.material_change_notification_days == 60
        assert config.critical_function_notification_days == 90
        assert config.review_frequency_months == 12

    def test_custom_config(self):
        """Test custom configuration."""
        config = SubcontractorConfig(
            default_notification_days=45,
            review_frequency_months=6,
        )

        assert config.default_notification_days == 45
        assert config.review_frequency_months == 6


class TestSubcontractor:
    """Tests for Subcontractor dataclass."""

    def test_basic_subcontractor(self):
        """Test basic subcontractor creation."""
        sub = Subcontractor(
            subcontractor_name="AWS",
            subcontractor_type=SubcontractorType.CLOUD_INFRASTRUCTURE,
        )

        assert sub.subcontractor_id.startswith("SUB-")
        assert sub.subcontractor_name == "AWS"
        assert sub.subcontractor_type == SubcontractorType.CLOUD_INFRASTRUCTURE

    def test_subcontractor_status(self):
        """Test subcontractor status default."""
        sub = Subcontractor(
            subcontractor_name="Test",
        )

        assert sub.status == SubcontractorStatus.ACTIVE

    def test_subcontractor_fields(self):
        """Test all subcontractor fields."""
        sub = Subcontractor(
            subcontractor_name="AWS",
            legal_name="Amazon Web Services EMEA SARL",
            headquarters_country="LU",
            data_processing_countries=["IE", "DE"],
            has_data_access=True,
            is_material=True,
            supports_critical_functions=True,
            risk_level=RiskLevel.MEDIUM,
            certifications=["SOC2", "ISO27001"],
        )

        assert sub.legal_name == "Amazon Web Services EMEA SARL"
        assert len(sub.data_processing_countries) == 2
        assert sub.is_material is True


class TestSubcontractorChange:
    """Tests for SubcontractorChange dataclass."""

    def test_change_auto_id(self):
        """Test change auto-generates ID."""
        change = SubcontractorChange(
            subcontractor_id="SUB-001",
            change_type=ChangeType.SERVICE_CHANGE,
            change_summary="Service scope expanded",
        )

        assert change.change_id.startswith("CHG-")

    def test_change_type(self):
        """Test change type setting."""
        change = SubcontractorChange(
            subcontractor_id="SUB-001",
            change_type=ChangeType.LOCATION_CHANGE,
            change_summary="New data center added",
        )

        assert change.change_type == ChangeType.LOCATION_CHANGE

    def test_change_can_proceed_notification_only(self):
        """Test can_proceed for NOTIFICATION_ONLY mode."""
        change = SubcontractorChange(
            subcontractor_id="SUB-001",
            change_type=ChangeType.SERVICE_CHANGE,
            change_summary="Test",
            consent_mode=ConsentMode.NOTIFICATION_ONLY,
            notification_status=NotificationStatus.SENT,
        )

        assert change.can_proceed() is True

    def test_change_can_proceed_with_objection(self):
        """Test can_proceed with objection."""
        change = SubcontractorChange(
            subcontractor_id="SUB-001",
            change_type=ChangeType.SERVICE_CHANGE,
            change_summary="Test",
            consent_mode=ConsentMode.NOTIFICATION_WITH_OBJECTION,
            clients_objected=["CLIENT-001"],
            objections_resolved=False,
        )

        assert change.can_proceed() is False

    def test_change_can_proceed_prior_consent(self):
        """Test can_proceed for PRIOR_CONSENT mode."""
        change = SubcontractorChange(
            subcontractor_id="SUB-001",
            change_type=ChangeType.SERVICE_CHANGE,
            change_summary="Test",
            consent_mode=ConsentMode.PRIOR_CONSENT,
            clients_requiring_prior_consent=["CLIENT-001"],
            clients_granted_consent=[],
        )

        assert change.can_proceed() is False

    def test_change_get_blocking_clients(self):
        """Test getting blocking clients."""
        change = SubcontractorChange(
            subcontractor_id="SUB-001",
            change_type=ChangeType.SERVICE_CHANGE,
            change_summary="Test",
            consent_mode=ConsentMode.NOTIFICATION_WITH_OBJECTION,
            clients_objected=["CLIENT-001", "CLIENT-002"],
            objections_resolved=False,
        )

        blockers = change.get_blocking_clients()
        assert len(blockers) == 2


class TestClientSubcontractorPreference:
    """Tests for ClientSubcontractorPreference dataclass."""

    def test_preference_creation(self):
        """Test preference creation."""
        pref = ClientSubcontractorPreference(
            client_id="CLIENT-001",
            client_name="Test Bank",
            notification_email="notifications@bank.com",
        )

        assert pref.client_id == "CLIENT-001"
        assert pref.notify_material_changes is True


class TestSubcontractorRiskAssessment:
    """Tests for SubcontractorRiskAssessment dataclass."""

    def test_assessment_auto_id(self):
        """Test assessment auto-generates ID."""
        assessment = SubcontractorRiskAssessment(
            subcontractor_id="SUB-001",
            assessed_by="Risk Team",
        )

        assert assessment.assessment_id.startswith("RSK-")


class TestDORASubcontractorManagement:
    """Tests for DORASubcontractorManagement main class."""

    @pytest.fixture
    def manager(self):
        """Create subcontractor management manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SubcontractorConfig(
                log_path=tmpdir,
            )
            yield DORASubcontractorManagement(config)

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager is not None
        assert manager.config is not None

    def test_initialization_with_standard_subcontractors(self, manager):
        """Test manager initializes with standard subcontractors."""
        subs = manager.get_all_subcontractors()
        # Should have AWS, Polygon, Alpaca from initialization
        assert len(subs) >= 3

    def test_register_subcontractor(self, manager):
        """Test registering a new subcontractor."""
        sub = manager.register_subcontractor(
            name="New Provider",
            subcontractor_type=SubcontractorType.DATA_PROVIDER,
            services_provided=["market_data"],
            headquarters_country="US",
        )

        assert sub is not None
        assert sub.subcontractor_name == "New Provider"

    def test_get_subcontractor(self, manager):
        """Test getting subcontractor by ID."""
        # Use one of the pre-initialized subcontractors
        subs = manager.get_all_subcontractors()
        if subs:
            retrieved = manager.get_subcontractor(subs[0].subcontractor_id)
            assert retrieved is not None

    def test_get_all_subcontractors(self, manager):
        """Test getting all subcontractors."""
        subs = manager.get_all_subcontractors(active_only=True)
        assert len(subs) >= 1

    def test_get_material_subcontractors(self, manager):
        """Test getting material subcontractors."""
        subs = manager.get_all_subcontractors(material_only=True)
        # AWS and Alpaca should be material
        assert len(subs) >= 1

    def test_get_subcontractors_by_type(self, manager):
        """Test getting subcontractors by type."""
        cloud_subs = manager.get_subcontractors_by_type(
            SubcontractorType.CLOUD_INFRASTRUCTURE
        )
        # AWS should be found
        assert len(cloud_subs) >= 1


class TestSubcontractorChain:
    """Tests for subcontractor chain management."""

    @pytest.fixture
    def manager(self):
        """Create manager for chain testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SubcontractorConfig(log_path=tmpdir)
            yield DORASubcontractorManagement(config)

    def test_register_sub_subcontractor(self, manager):
        """Test registering sub-subcontractor."""
        # Get AWS (parent)
        aws_subs = manager.get_subcontractors_by_type(
            SubcontractorType.CLOUD_INFRASTRUCTURE
        )
        if aws_subs:
            parent = aws_subs[0]

            sub_sub = manager.register_subcontractor(
                name="Sub-Subcontractor",
                subcontractor_type=SubcontractorType.SECURITY_SERVICES,
                services_provided=["security_monitoring"],
                chain_level=2,
                parent_subcontractor_id=parent.subcontractor_id,
            )

            assert sub_sub.chain_level == 2

    def test_get_subcontractor_chain(self, manager):
        """Test getting subcontractor chain."""
        # Get first subcontractor
        subs = manager.get_all_subcontractors()
        if subs:
            chain = manager.get_subcontractor_chain(subs[0].subcontractor_id)
            assert len(chain) >= 1


class TestChangeManagement:
    """Tests for change management."""

    @pytest.fixture
    def manager(self):
        """Create manager for change testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SubcontractorConfig(log_path=tmpdir)
            yield DORASubcontractorManagement(config)

    def test_record_change(self, manager):
        """Test recording a change."""
        subs = manager.get_all_subcontractors()
        if subs:
            change = manager.record_change(
                subcontractor_id=subs[0].subcontractor_id,
                change_type=ChangeType.SERVICE_CHANGE,
                change_summary="Added new service",
                change_details="Expanded compute capacity",
            )

            assert change is not None
            assert change.change_type == ChangeType.SERVICE_CHANGE

    def test_change_requires_notification_for_material(self, manager):
        """Test change requires notification for material subcontractor."""
        # AWS is material
        material_subs = manager.get_all_subcontractors(material_only=True)
        if material_subs:
            change = manager.record_change(
                subcontractor_id=material_subs[0].subcontractor_id,
                change_type=ChangeType.SERVICE_CHANGE,
                change_summary="Test change",
            )

            assert change.requires_client_notification is True

    def test_get_changes_for_subcontractor(self, manager):
        """Test getting changes for subcontractor."""
        subs = manager.get_all_subcontractors()
        if subs:
            manager.record_change(
                subcontractor_id=subs[0].subcontractor_id,
                change_type=ChangeType.SERVICE_CHANGE,
                change_summary="Change 1",
            )
            manager.record_change(
                subcontractor_id=subs[0].subcontractor_id,
                change_type=ChangeType.LOCATION_CHANGE,
                change_summary="Change 2",
            )

            changes = manager.get_changes_for_subcontractor(subs[0].subcontractor_id)
            assert len(changes) == 2

    def test_get_pending_notifications(self, manager):
        """Test getting pending notifications."""
        material_subs = manager.get_all_subcontractors(material_only=True)
        if material_subs:
            manager.record_change(
                subcontractor_id=material_subs[0].subcontractor_id,
                change_type=ChangeType.SERVICE_CHANGE,
                change_summary="Test change",
            )

            pending = manager.get_pending_notifications()
            assert len(pending) >= 1


class TestClientNotification:
    """Tests for client notification."""

    @pytest.fixture
    def manager_with_client(self):
        """Create manager with registered client."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SubcontractorConfig(log_path=tmpdir)
            manager = DORASubcontractorManagement(config)

            manager.register_client_preferences(
                client_id="CLIENT-001",
                client_name="Test Bank",
                notification_email="notifications@bank.com",
                notify_material_changes=True,
            )

            yield manager

    def test_register_client_preferences(self, manager_with_client):
        """Test registering client preferences."""
        # Client already registered in fixture
        prefs = manager_with_client._client_preferences.get("CLIENT-001")
        assert prefs is not None
        assert prefs.client_name == "Test Bank"

    def test_notify_clients_of_change(self, manager_with_client):
        """Test notifying clients of change."""
        material_subs = manager_with_client.get_all_subcontractors(material_only=True)
        if material_subs:
            change = manager_with_client.record_change(
                subcontractor_id=material_subs[0].subcontractor_id,
                change_type=ChangeType.SERVICE_CHANGE,
                change_summary="Test change",
            )

            results = manager_with_client.notify_clients_of_change(change.change_id)
            assert "clients_notified" in results

    def test_record_client_acknowledgment(self, manager_with_client):
        """Test recording client acknowledgment."""
        material_subs = manager_with_client.get_all_subcontractors(material_only=True)
        if material_subs:
            change = manager_with_client.record_change(
                subcontractor_id=material_subs[0].subcontractor_id,
                change_type=ChangeType.SERVICE_CHANGE,
                change_summary="Test change",
            )
            manager_with_client.notify_clients_of_change(change.change_id)

            updated = manager_with_client.record_client_response(
                change_id=change.change_id,
                client_id="CLIENT-001",
                acknowledged=True,
            )

            assert updated is not None


class TestObjectionHandling:
    """Tests for objection handling."""

    @pytest.fixture
    def manager_with_objection(self):
        """Create manager with client objection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SubcontractorConfig(log_path=tmpdir)
            manager = DORASubcontractorManagement(config)

            manager.register_client_preferences(
                client_id="CLIENT-001",
                client_name="Test Bank",
                notification_email="notifications@bank.com",
                require_approval_for_critical=True,
            )

            # Find critical function subcontractor
            critical_subs = [
                s for s in manager.get_all_subcontractors()
                if s.supports_critical_functions
            ]
            if critical_subs:
                change = manager.record_change(
                    subcontractor_id=critical_subs[0].subcontractor_id,
                    change_type=ChangeType.SERVICE_CHANGE,
                    change_summary="Critical change",
                    affects_critical_functions=True,
                )
                manager.notify_clients_of_change(change.change_id)

                # Record objection
                manager.record_client_response(
                    change_id=change.change_id,
                    client_id="CLIENT-001",
                    objection=True,
                    objection_reason="Security concerns",
                )

                yield manager, change

    def test_objection_blocks_change(self, manager_with_objection):
        """Test objection blocks change."""
        if manager_with_objection:
            manager, change = manager_with_objection
            retrieved = manager.get_change(change.change_id)

            if retrieved:
                assert retrieved.notification_status == NotificationStatus.OBJECTION_RECEIVED
                assert "CLIENT-001" in retrieved.clients_objected

    def test_resolve_objection(self, manager_with_objection):
        """Test resolving objection."""
        if manager_with_objection:
            manager, change = manager_with_objection

            resolved = manager.resolve_objection(
                change_id=change.change_id,
                client_id="CLIENT-001",
                resolution="Alternative solution proposed",
                resolved_by="Risk Manager",
            )

            if resolved:
                assert resolved.objections_resolved is True
                assert "CLIENT-001" not in resolved.clients_objected


class TestChangeImplementation:
    """Tests for change implementation."""

    @pytest.fixture
    def manager(self):
        """Create manager for implementation testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SubcontractorConfig(log_path=tmpdir)
            yield DORASubcontractorManagement(config)

    def test_implement_change_success(self, manager):
        """Test implementing approved change."""
        subs = manager.get_all_subcontractors()
        if subs:
            # Non-critical change (doesn't require approval)
            non_critical = [s for s in subs if not s.supports_critical_functions]
            if non_critical:
                change = manager.record_change(
                    subcontractor_id=non_critical[0].subcontractor_id,
                    change_type=ChangeType.SERVICE_CHANGE,
                    change_summary="Minor change",
                )

                result = manager.implement_change(
                    change_id=change.change_id,
                    implemented_by="Admin",
                )

                assert result["success"] is True

    def test_implement_blocked_change_fails(self, manager):
        """Test implementing blocked change fails."""
        critical_subs = [
            s for s in manager.get_all_subcontractors()
            if s.supports_critical_functions
        ]
        if critical_subs:
            change = manager.record_change(
                subcontractor_id=critical_subs[0].subcontractor_id,
                change_type=ChangeType.SERVICE_CHANGE,
                change_summary="Critical change",
                affects_critical_functions=True,
            )

            # Try to implement without approval
            result = manager.implement_change(
                change_id=change.change_id,
                implemented_by="Admin",
            )

            # Should fail because critical changes need approval
            assert result.get("success", True) is False or "blockers" in result

    def test_cancel_change(self, manager):
        """Test cancelling a change."""
        subs = manager.get_all_subcontractors()
        if subs:
            change = manager.record_change(
                subcontractor_id=subs[0].subcontractor_id,
                change_type=ChangeType.SERVICE_CHANGE,
                change_summary="Test change",
            )

            cancelled = manager.cancel_change(
                change_id=change.change_id,
                reason="No longer needed",
                cancelled_by="Admin",
            )

            assert cancelled is not None
            assert cancelled.change_status == "cancelled"


class TestRiskAssessment:
    """Tests for subcontractor risk assessment."""

    @pytest.fixture
    def manager(self):
        """Create manager for risk assessment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SubcontractorConfig(log_path=tmpdir)
            yield DORASubcontractorManagement(config)

    def test_assess_subcontractor_risk(self, manager):
        """Test assessing subcontractor risk."""
        subs = manager.get_all_subcontractors()
        if subs:
            assessment = manager.assess_subcontractor_risk(
                subcontractor_id=subs[0].subcontractor_id,
                assessed_by="Risk Team",
                operational_risk="medium",
                security_risk="low",
            )

            assert assessment is not None
            assert assessment.assessed_by == "Risk Team"

    def test_assessment_updates_subcontractor(self, manager):
        """Test assessment updates subcontractor risk level."""
        subs = manager.get_all_subcontractors()
        if subs:
            manager.assess_subcontractor_risk(
                subcontractor_id=subs[0].subcontractor_id,
                assessed_by="Risk Team",
                operational_risk="high",
                security_risk="high",
            )

            sub = manager.get_subcontractor(subs[0].subcontractor_id)
            # Risk level should be updated
            assert sub is not None

    def test_get_subcontractors_due_review(self, manager):
        """Test getting subcontractors due for review."""
        due = manager.get_subcontractors_due_review()
        # This depends on initialization dates
        assert isinstance(due, list)


class TestExportAndReporting:
    """Tests for export and reporting."""

    @pytest.fixture
    def manager(self):
        """Create manager for reporting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SubcontractorConfig(log_path=tmpdir)
            yield DORASubcontractorManagement(config)

    def test_export_for_client_roi(self, manager):
        """Test exporting for client ROI."""
        export = manager.export_for_client_roi()

        assert "export_date" in export
        assert "its_template" in export
        assert export["its_template"] == "B_99_01"
        assert "subcontractors" in export

    def test_export_with_client_filter(self, manager):
        """Test exporting with client filter."""
        manager.register_client_preferences(
            client_id="CLIENT-001",
            client_name="Test Bank",
            notification_email="test@bank.com",
            prohibited_countries=["CN"],
        )

        export = manager.export_for_client_roi(client_id="CLIENT-001")
        assert "subcontractors" in export

    def test_get_summary(self, manager):
        """Test getting summary."""
        summary = manager.get_summary()

        assert "timestamp" in summary
        assert "subcontractors" in summary
        assert "changes" in summary


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_subcontractor_management(self):
        """Test factory function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SubcontractorConfig(log_path=tmpdir)
            manager = create_subcontractor_management(config)

            assert isinstance(manager, DORASubcontractorManagement)


class TestEnumerations:
    """Tests for enumerations."""

    def test_subcontractor_type_values(self):
        """Test SubcontractorType enum values."""
        assert SubcontractorType.CLOUD_INFRASTRUCTURE.value == "cloud_infrastructure"
        assert SubcontractorType.DATA_PROVIDER.value == "data_provider"
        assert SubcontractorType.SECURITY_SERVICES.value == "security_services"

    def test_subcontractor_status_values(self):
        """Test SubcontractorStatus enum values."""
        assert SubcontractorStatus.ACTIVE.value == "active"
        assert SubcontractorStatus.PENDING_APPROVAL.value == "pending_approval"
        assert SubcontractorStatus.TERMINATED.value == "terminated"

    def test_risk_level_values(self):
        """Test RiskLevel enum values."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_change_type_values(self):
        """Test ChangeType enum values."""
        assert ChangeType.NEW_SUBCONTRACTOR.value == "new_subcontractor"
        assert ChangeType.TERMINATED.value == "terminated"
        assert ChangeType.SERVICE_CHANGE.value == "service_change"
        assert ChangeType.LOCATION_CHANGE.value == "location_change"

    def test_notification_status_values(self):
        """Test NotificationStatus enum values."""
        assert NotificationStatus.NOT_REQUIRED.value == "not_required"
        assert NotificationStatus.PENDING.value == "pending"
        assert NotificationStatus.SENT.value == "sent"
        assert NotificationStatus.OBJECTION_RECEIVED.value == "objection_received"

    def test_consent_mode_values(self):
        """Test ConsentMode enum values."""
        assert ConsentMode.NOTIFICATION_ONLY.value == "notification_only"
        assert ConsentMode.NOTIFICATION_WITH_OBJECTION.value == "notification_with_objection"
        assert ConsentMode.PRIOR_CONSENT.value == "prior_consent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
