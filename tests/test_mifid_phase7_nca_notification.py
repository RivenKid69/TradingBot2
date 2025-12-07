# -*- coding: utf-8 -*-
"""
Tests for MiFID II Phase 7 - NCA Notification (Article 17(2)).

Comprehensive test coverage for:
- AlgorithmDescription dataclass
- NCANotification functionality
- NCANotificationManager operations
- Document generation
"""

import json
import tempfile
import time
import pytest
from datetime import datetime, date, timedelta
from pathlib import Path

from services.compliance.nca_notification import (
    # Enums
    NCAJurisdiction,
    NotificationType,
    NotificationStatus,
    AlgorithmCategory,
    # Data classes
    NCAContact,
    AlgorithmDescription,
    NCANotification,
    # Manager
    NCANotificationManager,
    # Factory functions
    create_algorithm_description,
    create_nca_notification_manager,
)


# =============================================================================
# Test Enums
# =============================================================================


class TestEnums:
    """Test enum definitions."""

    def test_nca_jurisdiction_values(self):
        """Test NCAJurisdiction enum has major NCAs."""
        assert NCAJurisdiction.FCA.value == "fca"
        assert NCAJurisdiction.BAFIN.value == "bafin"
        assert NCAJurisdiction.AMF.value == "amf"
        assert NCAJurisdiction.CONSOB.value == "consob"
        assert NCAJurisdiction.ESMA.value == "esma"

    def test_notification_type_values(self):
        """Test NotificationType enum values."""
        assert NotificationType.INITIAL.value == "initial"
        assert NotificationType.NEW_ALGORITHM.value == "new_algorithm"
        assert NotificationType.MATERIAL_CHANGE.value == "material_change"
        assert NotificationType.DECOMMISSION.value == "decommission"
        assert NotificationType.INCIDENT.value == "incident"

    def test_notification_status_values(self):
        """Test NotificationStatus enum values."""
        assert NotificationStatus.DRAFT.value == "draft"
        assert NotificationStatus.PENDING_REVIEW.value == "pending_review"
        assert NotificationStatus.APPROVED.value == "approved"
        assert NotificationStatus.SUBMITTED.value == "submitted"
        assert NotificationStatus.ACKNOWLEDGED.value == "acknowledged"
        assert NotificationStatus.COMPLETED.value == "completed"

    def test_algorithm_category_values(self):
        """Test AlgorithmCategory enum values."""
        assert AlgorithmCategory.EXECUTION.value == "execution"
        assert AlgorithmCategory.MARKET_MAKING.value == "market_making"
        assert AlgorithmCategory.HFT.value == "hft"
        assert AlgorithmCategory.SMART_ORDER_ROUTING.value == "smart_order_routing"


# =============================================================================
# Test NCAContact
# =============================================================================


class TestNCAContact:
    """Test NCAContact dataclass."""

    def test_create_contact(self):
        """Test creating NCA contact."""
        contact = NCAContact(
            jurisdiction=NCAJurisdiction.FCA,
            authority_name="Financial Conduct Authority",
            department="Market Supervision",
            email="algo@fca.org.uk",
        )
        assert contact.jurisdiction == NCAJurisdiction.FCA
        assert contact.authority_name == "Financial Conduct Authority"

    def test_contact_to_dict(self):
        """Test contact serialization."""
        contact = NCAContact(
            jurisdiction=NCAJurisdiction.BAFIN,
            authority_name="BaFin",
            email="algo@bafin.de",
        )
        data = contact.to_dict()

        assert data["jurisdiction"] == "bafin"
        assert data["authority_name"] == "BaFin"

    def test_contact_from_dict(self):
        """Test contact deserialization."""
        data = {
            "jurisdiction": "amf",
            "authority_name": "AMF",
            "email": "algo@amf.fr",
        }
        contact = NCAContact.from_dict(data)

        assert contact.jurisdiction == NCAJurisdiction.AMF


# =============================================================================
# Test AlgorithmDescription
# =============================================================================


class TestAlgorithmDescription:
    """Test AlgorithmDescription dataclass."""

    def test_create_algorithm_description(self):
        """Test creating algorithm description."""
        desc = AlgorithmDescription(
            algorithm_id="ALGO-001",
            algorithm_name="VWAP Execution",
            version="1.0.0",
            category=AlgorithmCategory.EXECUTION,
            description="Volume Weighted Average Price execution algorithm",
            strategy_summary="Executes orders to minimize market impact",
            asset_classes=["equity"],
            trading_venues=["XLON", "XETR"],
            responsible_person="John Doe",
            responsible_email="john@firm.com",
        )

        assert desc.algorithm_id == "ALGO-001"
        assert desc.category == AlgorithmCategory.EXECUTION
        assert len(desc.trading_venues) == 2

    def test_algorithm_description_hft(self):
        """Test HFT algorithm description."""
        desc = AlgorithmDescription(
            algorithm_id="HFT-001",
            algorithm_name="HFT Strategy",
            category=AlgorithmCategory.HFT,
            is_hft=True,
            hft_strategy_type="market_making",
            messages_per_second=1000,
        )

        assert desc.is_hft
        assert desc.messages_per_second == 1000

    def test_algorithm_description_to_dict(self):
        """Test algorithm description serialization."""
        desc = AlgorithmDescription(
            algorithm_id="ALGO-001",
            algorithm_name="Test Algo",
            version="1.0.0",
            category=AlgorithmCategory.EXECUTION,
            max_order_value_eur=1_000_000,
            deployment_date=date.today(),
        )
        data = desc.to_dict()

        assert data["algorithm_id"] == "ALGO-001"
        assert data["category"] == "execution"
        assert data["max_order_value_eur"] == 1_000_000

    def test_algorithm_description_from_dict(self):
        """Test algorithm description deserialization."""
        data = {
            "algorithm_id": "ALGO-002",
            "algorithm_name": "From Dict",
            "version": "2.0.0",
            "category": "market_making",
            "trading_venues": ["XPAR"],
            "is_hft": False,
            "deployment_date": date.today().isoformat(),
        }
        desc = AlgorithmDescription.from_dict(data)

        assert desc.algorithm_id == "ALGO-002"
        assert desc.category == AlgorithmCategory.MARKET_MAKING
        assert desc.deployment_date == date.today()


# =============================================================================
# Test NCANotification
# =============================================================================


class TestNCANotification:
    """Test NCANotification dataclass."""

    @pytest.fixture
    def sample_notification(self) -> NCANotification:
        """Create sample notification."""
        notification = NCANotification(
            notification_type=NotificationType.INITIAL,
            nca_jurisdiction=NCAJurisdiction.FCA,
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Trading Ltd",
            firm_address="123 Trading Street, London",
            authorized_person="John Smith",
            authorized_email="john@test.com",
            authorized_phone="+44 123 456 7890",
            summary="Initial notification of algorithmic trading",
        )

        # Add algorithm
        notification.add_algorithm(AlgorithmDescription(
            algorithm_id="ALGO-001",
            algorithm_name="VWAP Execution",
            version="1.0.0",
            category=AlgorithmCategory.EXECUTION,
            trading_venues=["XLON"],
            responsible_person="Jane Doe",
            responsible_email="jane@test.com",
        ))

        return notification

    def test_create_notification(self, sample_notification):
        """Test creating notification."""
        assert sample_notification.notification_id
        assert sample_notification.notification_reference  # Auto-generated
        assert sample_notification.status == NotificationStatus.DRAFT

    def test_notification_reference_format(self, sample_notification):
        """Test notification reference format."""
        # Should be NCA-FCA-YYYYMMDD-XXXX
        ref = sample_notification.notification_reference
        assert ref.startswith("NCA-FCA-")
        parts = ref.split("-")
        assert len(parts) == 4

    def test_prepare(self, sample_notification):
        """Test preparing notification."""
        sample_notification.prepare("compliance_officer")

        assert sample_notification.status == NotificationStatus.PENDING_REVIEW
        assert sample_notification.prepared_by == "compliance_officer"
        assert sample_notification.prepared_date == date.today()

    def test_review(self, sample_notification):
        """Test reviewing notification."""
        sample_notification.review("senior_reviewer")

        assert sample_notification.reviewed_by == "senior_reviewer"
        assert sample_notification.reviewed_date == date.today()

    def test_approve(self, sample_notification):
        """Test approving notification."""
        sample_notification.approve("compliance_head")

        assert sample_notification.status == NotificationStatus.APPROVED
        assert sample_notification.approved_by == "compliance_head"
        assert sample_notification.approved_date == date.today()

    def test_submit(self, sample_notification):
        """Test submitting notification."""
        sample_notification.submit("NCA-REF-12345")

        assert sample_notification.status == NotificationStatus.SUBMITTED
        assert sample_notification.nca_reference == "NCA-REF-12345"
        assert sample_notification.submission_date == date.today()

    def test_acknowledge(self, sample_notification):
        """Test acknowledging notification."""
        sample_notification.acknowledge("NCA-ACK-001", "Acknowledged")

        assert sample_notification.status == NotificationStatus.ACKNOWLEDGED
        assert sample_notification.acknowledgment_date == date.today()
        assert sample_notification.nca_response == "Acknowledged"

    def test_complete(self, sample_notification):
        """Test completing notification."""
        sample_notification.complete()
        assert sample_notification.status == NotificationStatus.COMPLETED

    def test_reject(self, sample_notification):
        """Test rejecting notification."""
        sample_notification.reject("Missing information")
        assert sample_notification.status == NotificationStatus.REJECTED
        assert sample_notification.nca_response == "Missing information"

    def test_add_algorithm(self, sample_notification):
        """Test adding algorithm."""
        initial_count = len(sample_notification.algorithms)
        sample_notification.add_algorithm(AlgorithmDescription(
            algorithm_id="ALGO-002",
            algorithm_name="New Algo",
        ))

        assert len(sample_notification.algorithms) == initial_count + 1

    def test_remove_algorithm(self, sample_notification):
        """Test removing algorithm."""
        initial_count = len(sample_notification.algorithms)
        result = sample_notification.remove_algorithm("ALGO-001")

        assert result is True
        assert len(sample_notification.algorithms) == initial_count - 1

        # Remove non-existent
        result = sample_notification.remove_algorithm("NON-EXISTENT")
        assert result is False

    def test_get_algorithm(self, sample_notification):
        """Test getting algorithm by ID."""
        algo = sample_notification.get_algorithm("ALGO-001")
        assert algo is not None
        assert algo.algorithm_name == "VWAP Execution"

        algo = sample_notification.get_algorithm("NON-EXISTENT")
        assert algo is None

    def test_generate_notification_document(self, sample_notification):
        """Test document generation."""
        sample_notification.approve("approver")
        doc = sample_notification.generate_notification_document()

        assert "NCA NOTIFICATION" in doc
        assert "MiFID II Article 17(2)" in doc
        assert sample_notification.notification_reference in doc
        assert sample_notification.firm_name in doc
        assert sample_notification.firm_lei in doc
        assert "ALGO-001" in doc

    def test_generate_xml_notification(self, sample_notification):
        """Test XML document generation."""
        sample_notification.approve("approver")
        xml = sample_notification.generate_xml_notification()

        assert '<?xml version="1.0"' in xml
        assert "<AlgoTradingNotification" in xml
        assert f"<LEI>{sample_notification.firm_lei}</LEI>" in xml
        assert "<AlgorithmID>ALGO-001</AlgorithmID>" in xml

    def test_to_dict(self, sample_notification):
        """Test serialization."""
        data = sample_notification.to_dict()

        assert data["notification_type"] == "initial"
        assert data["nca_jurisdiction"] == "fca"
        assert data["firm_lei"] == "5493001KJTIIGC8Y1R12"
        assert len(data["algorithms"]) == 1

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "notification_id": "notif-123",
            "notification_reference": "NCA-FCA-20241207-TEST",
            "notification_type": "new_algorithm",
            "nca_jurisdiction": "bafin",
            "firm_lei": "123456",
            "firm_name": "Test Firm",
            "status": "approved",
            "algorithms": [
                {"algorithm_id": "ALGO-001", "category": "execution"}
            ],
        }

        notification = NCANotification.from_dict(data)

        assert notification.notification_id == "notif-123"
        assert notification.notification_type == NotificationType.NEW_ALGORITHM
        assert notification.nca_jurisdiction == NCAJurisdiction.BAFIN
        assert notification.status == NotificationStatus.APPROVED
        assert len(notification.algorithms) == 1

    def test_json_roundtrip(self, sample_notification):
        """Test JSON serialization roundtrip."""
        json_str = sample_notification.to_json()
        restored = NCANotification.from_dict(json.loads(json_str))

        assert restored.notification_id == sample_notification.notification_id
        assert restored.firm_name == sample_notification.firm_name
        assert len(restored.algorithms) == len(sample_notification.algorithms)


# =============================================================================
# Test NCANotificationManager
# =============================================================================


class TestNCANotificationManager:
    """Test NCANotificationManager."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def manager(self, temp_storage) -> NCANotificationManager:
        """Create notification manager with temp storage."""
        return NCANotificationManager(
            storage_path=temp_storage,
            default_jurisdiction=NCAJurisdiction.FCA,
        )

    def test_create_manager(self, temp_storage):
        """Test creating notification manager."""
        manager = NCANotificationManager(storage_path=temp_storage)
        assert Path(temp_storage).exists()
        assert manager.default_jurisdiction == NCAJurisdiction.FCA

    def test_create_notification(self, manager):
        """Test creating notification."""
        notification = manager.create_notification(
            notification_type=NotificationType.INITIAL,
            firm_lei="123456",
            firm_name="Test Firm",
            authorized_person="John Doe",
        )

        assert notification is not None
        assert notification.notification_type == NotificationType.INITIAL
        assert notification.firm_lei == "123456"

    def test_create_initial_notification(self, manager):
        """Test creating initial notification."""
        algos = [
            AlgorithmDescription(
                algorithm_id="ALGO-001",
                algorithm_name="Algo 1",
                category=AlgorithmCategory.EXECUTION,
            ),
            AlgorithmDescription(
                algorithm_id="ALGO-002",
                algorithm_name="Algo 2",
                category=AlgorithmCategory.MARKET_MAKING,
            ),
        ]

        notification = manager.create_initial_notification(
            firm_lei="123456",
            firm_name="Test Firm",
            algorithms=algos,
        )

        assert notification.notification_type == NotificationType.INITIAL
        assert len(notification.algorithms) == 2
        assert "Initial notification" in notification.summary

    def test_create_new_algorithm_notification(self, manager):
        """Test creating new algorithm notification."""
        algo = AlgorithmDescription(
            algorithm_id="ALGO-003",
            algorithm_name="New Algorithm",
            category=AlgorithmCategory.EXECUTION,
        )

        notification = manager.create_new_algorithm_notification(
            firm_lei="123456",
            firm_name="Test Firm",
            algorithm=algo,
            effective_date=date.today() + timedelta(days=7),
        )

        assert notification.notification_type == NotificationType.NEW_ALGORITHM
        assert len(notification.algorithms) == 1
        assert notification.effective_date == date.today() + timedelta(days=7)

    def test_create_material_change_notification(self, manager):
        """Test creating material change notification."""
        algo = AlgorithmDescription(
            algorithm_id="ALGO-001",
            algorithm_name="Updated Algorithm",
            version="2.0.0",
            category=AlgorithmCategory.EXECUTION,
        )

        notification = manager.create_material_change_notification(
            firm_lei="123456",
            firm_name="Test Firm",
            algorithm=algo,
            change_description="Updated risk limits and trading logic",
        )

        assert notification.notification_type == NotificationType.MATERIAL_CHANGE
        assert notification.change_description == "Updated risk limits and trading logic"

    def test_get_notification(self, manager):
        """Test getting notification by ID."""
        notification = manager.create_notification(
            notification_type=NotificationType.INITIAL,
            firm_lei="123",
            firm_name="Test",
        )

        found = manager.get_notification(notification.notification_id)
        assert found is not None

    def test_get_notification_by_reference(self, manager):
        """Test getting notification by reference."""
        notification = manager.create_notification(
            notification_type=NotificationType.INITIAL,
            firm_lei="123",
            firm_name="Test",
        )

        found = manager.get_notification_by_reference(notification.notification_reference)
        assert found is not None

    def test_get_notifications_by_status(self, manager):
        """Test filtering by status."""
        notif1 = manager.create_notification(
            notification_type=NotificationType.INITIAL,
            firm_lei="123",
            firm_name="Test",
        )

        notif2 = manager.create_notification(
            notification_type=NotificationType.NEW_ALGORITHM,
            firm_lei="123",
            firm_name="Test",
        )
        notif2.approve("approver")

        drafts = manager.get_notifications_by_status(NotificationStatus.DRAFT)
        approved = manager.get_notifications_by_status(NotificationStatus.APPROVED)

        assert len(drafts) == 1
        assert len(approved) == 1

    def test_get_pending_notifications(self, manager):
        """Test getting pending notifications."""
        notif1 = manager.create_notification(
            notification_type=NotificationType.INITIAL,
            firm_lei="123",
            firm_name="Test",
        )  # Draft

        notif2 = manager.create_notification(
            notification_type=NotificationType.NEW_ALGORITHM,
            firm_lei="123",
            firm_name="Test",
        )
        notif2.complete()  # Completed

        pending = manager.get_pending_notifications()
        assert len(pending) == 1

    def test_save_and_load_notification(self, manager):
        """Test notification persistence."""
        notification = manager.create_notification(
            notification_type=NotificationType.INITIAL,
            firm_lei="123",
            firm_name="Test Firm",
        )
        notification.add_algorithm(AlgorithmDescription(
            algorithm_id="ALGO-001",
            algorithm_name="Test Algo",
        ))
        notification.approve("approver")

        manager.save_notification(notification)

        # Create new manager and load
        new_manager = NCANotificationManager(storage_path=manager.storage_path)
        loaded = new_manager.load_notification(notification.notification_id)

        assert loaded is not None
        assert loaded.notification_id == notification.notification_id
        assert loaded.status == NotificationStatus.APPROVED
        assert len(loaded.algorithms) == 1

    def test_get_notification_summary(self, manager):
        """Test notification summary."""
        manager.create_notification(
            notification_type=NotificationType.INITIAL,
            firm_lei="123",
            firm_name="Test",
        )
        notif2 = manager.create_notification(
            notification_type=NotificationType.NEW_ALGORITHM,
            firm_lei="123",
            firm_name="Test",
        )
        notif2.approve("approver")

        summary = manager.get_notification_summary()

        assert summary["total_notifications"] == 2
        assert summary["status_counts"]["draft"] == 1
        assert summary["status_counts"]["approved"] == 1


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_algorithm_description_factory(self):
        """Test create_algorithm_description factory."""
        desc = create_algorithm_description(
            algorithm_id="ALGO-001",
            algorithm_name="Test Algorithm",
            version="1.0.0",
            category=AlgorithmCategory.EXECUTION,
            description="Test description",
            strategy_summary="Test strategy",
            responsible_person="John Doe",
            responsible_email="john@test.com",
            trading_venues=["XLON", "XETR"],
            asset_classes=["equity"],
            max_order_value_eur=500_000,
            is_hft=False,
        )

        assert desc.algorithm_id == "ALGO-001"
        assert desc.category == AlgorithmCategory.EXECUTION
        assert desc.max_order_value_eur == 500_000
        assert desc.kill_switch_enabled is True  # Default
        assert len(desc.trading_venues) == 2

    def test_create_nca_notification_manager_factory(self, tmp_path):
        """Test create_nca_notification_manager factory."""
        manager = create_nca_notification_manager(
            storage_path=str(tmp_path),
            default_jurisdiction=NCAJurisdiction.BAFIN,
        )

        assert isinstance(manager, NCANotificationManager)
        assert manager.default_jurisdiction == NCAJurisdiction.BAFIN


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests."""

    def test_full_notification_workflow(self, tmp_path):
        """Test complete NCA notification workflow."""
        # 1. Create manager
        manager = create_nca_notification_manager(
            storage_path=str(tmp_path),
            default_jurisdiction=NCAJurisdiction.FCA,
        )

        # 2. Create algorithm description
        algo = create_algorithm_description(
            algorithm_id="VWAP-001",
            algorithm_name="VWAP Execution Algorithm",
            version="1.0.0",
            category=AlgorithmCategory.EXECUTION,
            description="Volume Weighted Average Price execution algorithm",
            strategy_summary="Minimizes market impact by executing over time",
            responsible_person="Jane Smith",
            responsible_email="jane@firm.com",
            trading_venues=["XLON", "XETR", "XPAR"],
            asset_classes=["equity"],
        )

        # 3. Create initial notification
        notification = manager.create_initial_notification(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Algorithmic Trading Ltd",
            algorithms=[algo],
            jurisdiction=NCAJurisdiction.FCA,
        )

        # 4. Prepare notification
        notification.prepare("compliance_analyst")
        assert notification.status == NotificationStatus.PENDING_REVIEW

        # 5. Review and approve
        notification.review("senior_compliance")
        notification.approve("compliance_head")
        assert notification.status == NotificationStatus.APPROVED

        # 6. Generate documents
        doc = notification.generate_notification_document()
        xml = notification.generate_xml_notification()

        assert "VWAP Execution Algorithm" in doc
        assert "VWAP-001" in xml

        # 7. Submit to NCA
        notification.submit("FCA-2024-001234")
        assert notification.status == NotificationStatus.SUBMITTED

        # 8. NCA acknowledges
        notification.acknowledge("FCA-ACK-001", "Notification received")
        assert notification.status == NotificationStatus.ACKNOWLEDGED

        # 9. Complete process
        notification.complete()
        assert notification.status == NotificationStatus.COMPLETED

        # 10. Save and verify
        manager.save_notification(notification)
        loaded = manager.load_notification(notification.notification_id)
        assert loaded.status == NotificationStatus.COMPLETED

    def test_material_change_workflow(self, tmp_path):
        """Test material change notification workflow."""
        manager = create_nca_notification_manager(storage_path=str(tmp_path))

        # Updated algorithm
        algo = create_algorithm_description(
            algorithm_id="VWAP-001",
            algorithm_name="VWAP Execution Algorithm",
            version="2.0.0",  # Updated version
            category=AlgorithmCategory.EXECUTION,
            description="Updated VWAP with enhanced risk controls",
            strategy_summary="Minimizes market impact with improved limits",
            responsible_person="Jane Smith",
            responsible_email="jane@firm.com",
        )
        algo.last_material_change = date.today()

        # Create material change notification
        notification = manager.create_material_change_notification(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Algorithmic Trading Ltd",
            algorithm=algo,
            change_description="Updated risk limits from EUR 1M to EUR 500K. "
                              "Added new price collar validation.",
            effective_date=date.today() + timedelta(days=7),
        )

        assert notification.notification_type == NotificationType.MATERIAL_CHANGE
        assert notification.effective_date == date.today() + timedelta(days=7)
        assert "risk limits" in notification.change_description.lower()

        # Generate document
        doc = notification.generate_notification_document()
        assert "MATERIAL CHANGE" in doc.upper() or "material change" in doc.lower()
