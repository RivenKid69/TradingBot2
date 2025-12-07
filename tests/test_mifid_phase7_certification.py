# -*- coding: utf-8 -*-
"""
Tests for MiFID II Phase 7 - Certification (RTS 6 Article 5/7).

Comprehensive test coverage for:
- CertificateCondition dataclass
- ConformanceCertificate functionality
- CertificateManager operations
- Certificate lifecycle management
"""

import json
import tempfile
import time
import pytest
from datetime import datetime, date, timedelta
from pathlib import Path

from services.compliance.certification import (
    # Enums
    CertificateStatus,
    CertificateType,
    DeploymentApproval,
    # Data classes
    CertificateCondition,
    ConformanceCertificate,
    # Manager
    CertificateManager,
    # Factory functions
    create_certificate,
    create_certificate_manager,
)
from services.compliance.conformance_testing import (
    ConformanceTestSuite,
    ConformanceSuiteStatus,
    TestEnvironment,
    TestResult,
    ConformanceTest,
    TestPriority,
    TestCategory,
)


# =============================================================================
# Test Enums
# =============================================================================


class TestEnums:
    """Test enum definitions."""

    def test_certificate_status_values(self):
        """Test CertificateStatus enum values."""
        assert CertificateStatus.DRAFT.value == "draft"
        assert CertificateStatus.PENDING_APPROVAL.value == "pending_approval"
        assert CertificateStatus.APPROVED.value == "approved"
        assert CertificateStatus.EXPIRED.value == "expired"
        assert CertificateStatus.REVOKED.value == "revoked"
        assert CertificateStatus.SUPERSEDED.value == "superseded"

    def test_certificate_type_values(self):
        """Test CertificateType enum values."""
        assert CertificateType.INITIAL.value == "initial"
        assert CertificateType.UPDATE.value == "update"
        assert CertificateType.RECERTIFICATION.value == "recertification"
        assert CertificateType.EMERGENCY.value == "emergency"

    def test_deployment_approval_values(self):
        """Test DeploymentApproval enum values."""
        assert DeploymentApproval.NOT_REQUESTED.value == "not_requested"
        assert DeploymentApproval.PENDING.value == "pending"
        assert DeploymentApproval.APPROVED.value == "approved"
        assert DeploymentApproval.REJECTED.value == "rejected"
        assert DeploymentApproval.CONDITIONAL.value == "conditional"


# =============================================================================
# Test CertificateCondition
# =============================================================================


class TestCertificateCondition:
    """Test CertificateCondition dataclass."""

    def test_create_condition(self):
        """Test creating a condition."""
        condition = CertificateCondition(
            description="Review test warnings",
            requirement="Document resolution of all warnings",
            deadline=date.today() + timedelta(days=14),
        )
        assert condition.condition_id
        assert condition.description == "Review test warnings"
        assert not condition.satisfied

    def test_is_overdue(self):
        """Test overdue detection."""
        # Not overdue - deadline in future
        future_cond = CertificateCondition(
            deadline=date.today() + timedelta(days=7),
        )
        assert not future_cond.is_overdue()

        # Overdue - deadline in past
        past_cond = CertificateCondition(
            deadline=date.today() - timedelta(days=1),
        )
        assert past_cond.is_overdue()

        # Satisfied - never overdue
        satisfied_cond = CertificateCondition(
            deadline=date.today() - timedelta(days=1),
            satisfied=True,
        )
        assert not satisfied_cond.is_overdue()

        # No deadline - never overdue
        no_deadline = CertificateCondition()
        assert not no_deadline.is_overdue()

    def test_condition_to_dict(self):
        """Test condition serialization."""
        condition = CertificateCondition(
            description="Test condition",
            requirement="Do something",
            deadline=date.today(),
            satisfied=True,
            satisfied_by="tester",
        )
        data = condition.to_dict()

        assert data["description"] == "Test condition"
        assert data["satisfied"] is True
        assert data["satisfied_by"] == "tester"

    def test_condition_from_dict(self):
        """Test condition deserialization."""
        data = {
            "condition_id": "cond-123",
            "description": "From dict condition",
            "requirement": "Requirement text",
            "deadline": date.today().isoformat(),
            "satisfied": False,
        }
        condition = CertificateCondition.from_dict(data)

        assert condition.condition_id == "cond-123"
        assert condition.description == "From dict condition"
        assert not condition.satisfied


# =============================================================================
# Test ConformanceCertificate
# =============================================================================


class TestConformanceCertificate:
    """Test ConformanceCertificate dataclass."""

    @pytest.fixture
    def sample_certificate(self) -> ConformanceCertificate:
        """Create sample certificate."""
        cert = ConformanceCertificate(
            certificate_type=CertificateType.INITIAL,
            algorithm_id="ALGO-001",
            algorithm_version="1.0.0",
            algorithm_name="Test Algorithm",
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm Ltd",
            test_suite_id="suite-123",
            test_environment=TestEnvironment.UAT,
            test_pass_rate=95.0,
            critical_tests_passed=10,
            critical_tests_total=10,
        )
        return cert

    def test_create_certificate(self, sample_certificate):
        """Test creating a certificate."""
        assert sample_certificate.certificate_id
        assert sample_certificate.certificate_number  # Auto-generated
        assert sample_certificate.algorithm_id == "ALGO-001"
        assert sample_certificate.status == CertificateStatus.DRAFT

    def test_certificate_number_format(self, sample_certificate):
        """Test certificate number format."""
        # Should be CERT-YYYYMMDD-XXXX
        assert sample_certificate.certificate_number.startswith("CERT-")
        parts = sample_certificate.certificate_number.split("-")
        assert len(parts) == 3
        assert len(parts[1]) == 8  # YYYYMMDD

    def test_is_valid(self, sample_certificate):
        """Test validity check."""
        # Draft certificate not valid
        assert not sample_certificate.is_valid()

        # Approved certificate with valid dates
        sample_certificate.status = CertificateStatus.APPROVED
        sample_certificate.valid_from = date.today() - timedelta(days=1)
        sample_certificate.valid_until = date.today() + timedelta(days=30)
        assert sample_certificate.is_valid()

        # Expired certificate
        sample_certificate.valid_until = date.today() - timedelta(days=1)
        assert not sample_certificate.is_valid()

    def test_is_expired(self, sample_certificate):
        """Test expiry check."""
        sample_certificate.valid_until = date.today() - timedelta(days=1)
        assert sample_certificate.is_expired()

        sample_certificate.valid_until = date.today() + timedelta(days=30)
        assert not sample_certificate.is_expired()

    def test_days_until_expiry(self, sample_certificate):
        """Test days until expiry calculation."""
        sample_certificate.valid_until = date.today() + timedelta(days=30)
        assert sample_certificate.days_until_expiry() == 30

        sample_certificate.valid_until = None
        assert sample_certificate.days_until_expiry() is None

    def test_submit_for_approval(self, sample_certificate):
        """Test submitting for approval."""
        sample_certificate.submit_for_approval("compliance_officer")

        assert sample_certificate.status == CertificateStatus.PENDING_APPROVAL
        assert sample_certificate.issued_by == "compliance_officer"
        assert sample_certificate.issued_date == date.today()

    def test_approve(self, sample_certificate):
        """Test approving certificate."""
        sample_certificate.approve("senior_manager", "All checks passed")

        assert sample_certificate.status == CertificateStatus.APPROVED
        assert sample_certificate.approved_by == "senior_manager"
        assert sample_certificate.approval_date == date.today()
        assert sample_certificate.valid_from == date.today()
        assert sample_certificate.valid_until is not None

    def test_reject(self, sample_certificate):
        """Test rejecting certificate."""
        sample_certificate.status = CertificateStatus.PENDING_APPROVAL
        sample_certificate.reject("reviewer", "Missing documentation")

        assert sample_certificate.status == CertificateStatus.DRAFT
        assert "Rejected" in sample_certificate.approval_notes

    def test_revoke(self, sample_certificate):
        """Test revoking certificate."""
        sample_certificate.status = CertificateStatus.APPROVED
        sample_certificate.revoke("compliance_head", "Issue discovered")

        assert sample_certificate.status == CertificateStatus.REVOKED
        assert sample_certificate.revoked_by == "compliance_head"
        assert sample_certificate.revocation_date == date.today()
        assert sample_certificate.revocation_reason == "Issue discovered"

    def test_supersede(self, sample_certificate):
        """Test superseding certificate."""
        sample_certificate.status = CertificateStatus.APPROVED
        sample_certificate.supersede("new-cert-id")

        assert sample_certificate.status == CertificateStatus.SUPERSEDED
        assert "new-cert-id" in sample_certificate.approval_notes

    def test_expire(self, sample_certificate):
        """Test expiring certificate."""
        sample_certificate.status = CertificateStatus.APPROVED
        sample_certificate.expire()

        assert sample_certificate.status == CertificateStatus.EXPIRED

    def test_add_condition(self, sample_certificate):
        """Test adding condition."""
        condition = CertificateCondition(description="Test condition")
        sample_certificate.add_condition(condition)

        assert len(sample_certificate.conditions) == 1

    def test_get_unsatisfied_conditions(self, sample_certificate):
        """Test getting unsatisfied conditions."""
        sample_certificate.add_condition(CertificateCondition(
            description="Unsatisfied",
            satisfied=False,
        ))
        sample_certificate.add_condition(CertificateCondition(
            description="Satisfied",
            satisfied=True,
        ))

        unsatisfied = sample_certificate.get_unsatisfied_conditions()
        assert len(unsatisfied) == 1
        assert unsatisfied[0].description == "Unsatisfied"

    def test_satisfy_condition(self, sample_certificate):
        """Test satisfying a condition."""
        condition = CertificateCondition(description="To satisfy")
        sample_certificate.add_condition(condition)

        result = sample_certificate.satisfy_condition(
            condition.condition_id,
            "tester",
            "Resolved"
        )

        assert result is True
        assert condition.satisfied is True
        assert condition.satisfied_by == "tester"

    def test_all_conditions_satisfied(self, sample_certificate):
        """Test all conditions satisfied check."""
        # No conditions - all satisfied
        assert sample_certificate.all_conditions_satisfied()

        # Add unsatisfied condition
        sample_certificate.add_condition(CertificateCondition(satisfied=False))
        assert not sample_certificate.all_conditions_satisfied()

        # Satisfy it
        for cond in sample_certificate.conditions:
            cond.satisfied = True
        assert sample_certificate.all_conditions_satisfied()

    def test_compute_hash(self, sample_certificate):
        """Test hash computation."""
        hash1 = sample_certificate.compute_hash()
        assert len(hash1) == 64  # SHA-256

        # Same data should produce same hash
        hash2 = sample_certificate.compute_hash()
        assert hash1 == hash2

        # Different data should produce different hash
        sample_certificate.algorithm_version = "2.0.0"
        hash3 = sample_certificate.compute_hash()
        assert hash1 != hash3

    def test_generate_certificate_document(self, sample_certificate):
        """Test certificate document generation."""
        sample_certificate.approve("approver")
        doc = sample_certificate.generate_certificate_document()

        assert "CONFORMANCE CERTIFICATE" in doc
        assert sample_certificate.certificate_number in doc
        assert sample_certificate.algorithm_id in doc
        assert sample_certificate.firm_name in doc

    def test_generate_deployment_approval(self, sample_certificate):
        """Test deployment approval generation."""
        # Not approved - error
        doc = sample_certificate.generate_deployment_approval()
        assert "ERROR" in doc

        # Approved and valid
        sample_certificate.approve("approver")
        doc = sample_certificate.generate_deployment_approval()
        assert "DEPLOYMENT AUTHORIZED" in doc

    def test_to_dict(self, sample_certificate):
        """Test serialization."""
        data = sample_certificate.to_dict()

        assert data["algorithm_id"] == "ALGO-001"
        assert data["algorithm_version"] == "1.0.0"
        assert data["certificate_type"] == "initial"
        assert data["status"] == "draft"

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "certificate_id": "cert-123",
            "certificate_number": "CERT-20241207-TEST",
            "certificate_type": "update",
            "algorithm_id": "ALGO-002",
            "algorithm_version": "2.0.0",
            "firm_lei": "1234567890123456",
            "status": "approved",
            "test_pass_rate": 98.5,
            "valid_from": date.today().isoformat(),
            "valid_until": (date.today() + timedelta(days=365)).isoformat(),
        }

        cert = ConformanceCertificate.from_dict(data)

        assert cert.certificate_id == "cert-123"
        assert cert.certificate_type == CertificateType.UPDATE
        assert cert.status == CertificateStatus.APPROVED
        assert cert.test_pass_rate == 98.5

    def test_json_roundtrip(self, sample_certificate):
        """Test JSON serialization roundtrip."""
        json_str = sample_certificate.to_json()
        restored = ConformanceCertificate.from_json(json_str)

        assert restored.certificate_id == sample_certificate.certificate_id
        assert restored.algorithm_id == sample_certificate.algorithm_id


# =============================================================================
# Test CertificateManager
# =============================================================================


class TestCertificateManager:
    """Test CertificateManager."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def manager(self, temp_storage) -> CertificateManager:
        """Create certificate manager with temp storage."""
        return CertificateManager(
            storage_path=temp_storage,
            default_validity_days=365,
            expiry_warning_days=30,
        )

    @pytest.fixture
    def passed_suite(self) -> ConformanceTestSuite:
        """Create passed test suite."""
        suite = ConformanceTestSuite(
            algorithm_id="ALGO-001",
            algorithm_version="1.0.0",
            environment=TestEnvironment.UAT,
        )
        suite.add_test(ConformanceTest(
            test_id="CT-001",
            priority=TestPriority.CRITICAL,
            result=TestResult.PASS,
        ))
        suite.add_test(ConformanceTest(
            test_id="CT-002",
            priority=TestPriority.HIGH,
            result=TestResult.PASS,
        ))
        suite.status = ConformanceSuiteStatus.PASSED
        return suite

    def test_create_manager(self, temp_storage):
        """Test creating certificate manager."""
        manager = CertificateManager(storage_path=temp_storage)
        assert Path(temp_storage).exists()

    def test_create_certificate_from_suite(self, manager, passed_suite):
        """Test creating certificate from test suite."""
        cert = manager.create_certificate_from_suite(
            suite=passed_suite,
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            algorithm_name="Test Algorithm",
            issued_by="tester",
        )

        assert cert is not None
        assert cert.algorithm_id == "ALGO-001"
        assert cert.test_pass_rate == 100.0
        assert cert.critical_tests_passed == 1

    def test_create_certificate_from_failed_suite(self, manager):
        """Test cannot create certificate from failed suite."""
        suite = ConformanceTestSuite(status=ConformanceSuiteStatus.FAILED)

        cert = manager.create_certificate_from_suite(
            suite=suite,
            firm_lei="123",
            firm_name="Test",
        )

        assert cert is None

    def test_get_certificate(self, manager, passed_suite):
        """Test getting certificate by ID."""
        cert = manager.create_certificate_from_suite(
            suite=passed_suite,
            firm_lei="123",
            firm_name="Test",
        )

        found = manager.get_certificate(cert.certificate_id)
        assert found is not None
        assert found.certificate_id == cert.certificate_id

    def test_get_certificate_by_number(self, manager, passed_suite):
        """Test getting certificate by number."""
        cert = manager.create_certificate_from_suite(
            suite=passed_suite,
            firm_lei="123",
            firm_name="Test",
        )

        found = manager.get_certificate_by_number(cert.certificate_number)
        assert found is not None

    def test_get_certificates_for_algorithm(self, manager, passed_suite):
        """Test getting certificates for algorithm."""
        cert1 = manager.create_certificate_from_suite(
            suite=passed_suite,
            firm_lei="123",
            firm_name="Test",
        )
        cert1.approve("approver")

        # Modify version for second certificate
        passed_suite.algorithm_version = "1.1.0"
        cert2 = manager.create_certificate_from_suite(
            suite=passed_suite,
            firm_lei="123",
            firm_name="Test",
        )
        cert2.approve("approver")

        certs = manager.get_certificates_for_algorithm("ALGO-001")
        assert len(certs) == 2

    def test_get_valid_certificate(self, manager, passed_suite):
        """Test getting valid certificate for version."""
        cert = manager.create_certificate_from_suite(
            suite=passed_suite,
            firm_lei="123",
            firm_name="Test",
        )
        cert.approve("approver")

        found = manager.get_valid_certificate("ALGO-001", "1.0.0")
        assert found is not None
        assert found.algorithm_version == "1.0.0"

    def test_get_expiring_certificates(self, manager, passed_suite):
        """Test getting expiring certificates."""
        cert = manager.create_certificate_from_suite(
            suite=passed_suite,
            firm_lei="123",
            firm_name="Test",
        )
        cert.approve("approver")
        # Set to expire in 20 days
        cert.valid_until = date.today() + timedelta(days=20)

        expiring = manager.get_expiring_certificates(days=30)
        assert len(expiring) == 1

    def test_approve_certificate(self, manager, passed_suite):
        """Test approving certificate via manager."""
        cert = manager.create_certificate_from_suite(
            suite=passed_suite,
            firm_lei="123",
            firm_name="Test",
        )
        cert.submit_for_approval("issuer")

        result = manager.approve_certificate(
            cert.certificate_id,
            "approver",
            "Looks good",
        )

        assert result is True
        assert cert.status == CertificateStatus.APPROVED

    def test_revoke_certificate(self, manager, passed_suite):
        """Test revoking certificate via manager."""
        cert = manager.create_certificate_from_suite(
            suite=passed_suite,
            firm_lei="123",
            firm_name="Test",
        )
        cert.approve("approver")

        result = manager.revoke_certificate(
            cert.certificate_id,
            "compliance",
            "Issue found",
        )

        assert result is True
        assert cert.status == CertificateStatus.REVOKED

    def test_can_deploy(self, manager, passed_suite):
        """Test deployment authorization check."""
        # No certificate
        can_deploy, reason = manager.can_deploy("ALGO-001", "1.0.0")
        assert not can_deploy
        assert "No valid certificate" in reason

        # Create and approve certificate
        cert = manager.create_certificate_from_suite(
            suite=passed_suite,
            firm_lei="123",
            firm_name="Test",
        )
        cert.approve("approver")

        can_deploy, reason = manager.can_deploy("ALGO-001", "1.0.0")
        assert can_deploy
        assert "authorized" in reason.lower()

    def test_check_and_expire_certificates(self, manager, passed_suite):
        """Test automatic expiration check."""
        cert = manager.create_certificate_from_suite(
            suite=passed_suite,
            firm_lei="123",
            firm_name="Test",
        )
        cert.approve("approver")
        # Set already expired
        cert.valid_until = date.today() - timedelta(days=1)

        expired = manager.check_and_expire_certificates()
        assert len(expired) == 1
        assert cert.status == CertificateStatus.EXPIRED

    def test_save_and_load_certificate(self, manager, passed_suite):
        """Test certificate persistence."""
        cert = manager.create_certificate_from_suite(
            suite=passed_suite,
            firm_lei="123",
            firm_name="Test",
        )
        cert.approve("approver")

        manager.save_certificate(cert)

        # Create new manager and load
        new_manager = CertificateManager(storage_path=manager.storage_path)
        loaded = new_manager.load_certificate(cert.certificate_id)

        assert loaded is not None
        assert loaded.certificate_id == cert.certificate_id
        assert loaded.status == CertificateStatus.APPROVED

    def test_get_certificate_summary(self, manager, passed_suite):
        """Test certificate summary generation."""
        cert = manager.create_certificate_from_suite(
            suite=passed_suite,
            firm_lei="123",
            firm_name="Test",
        )
        cert.approve("approver")

        summary = manager.get_certificate_summary()

        assert summary["total_certificates"] == 1
        assert summary["status_counts"]["approved"] == 1


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_certificate(self):
        """Test create_certificate factory."""
        cert = create_certificate(
            algorithm_id="ALGO-001",
            algorithm_version="1.0.0",
            firm_lei="123456",
            firm_name="Test Firm",
            test_suite_id="suite-123",
            test_pass_rate=95.0,
            critical_passed=10,
            critical_total=10,
            certificate_type=CertificateType.UPDATE,
            validity_days=180,
        )

        assert cert.algorithm_id == "ALGO-001"
        assert cert.certificate_type == CertificateType.UPDATE
        assert cert.validity_days == 180
        assert cert.test_pass_rate == 95.0

    def test_create_certificate_manager(self, tmp_path):
        """Test create_certificate_manager factory."""
        manager = create_certificate_manager(
            storage_path=str(tmp_path),
            default_validity_days=730,
        )

        assert isinstance(manager, CertificateManager)
        assert manager.default_validity_days == 730


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests."""

    def test_full_certification_workflow(self, tmp_path):
        """Test complete certification workflow."""
        # 1. Create manager
        manager = create_certificate_manager(storage_path=str(tmp_path))

        # 2. Create passed test suite
        suite = ConformanceTestSuite(
            algorithm_id="ALGO-001",
            algorithm_version="1.0.0",
        )
        suite.add_test(ConformanceTest(
            test_id="CT-001",
            category=TestCategory.KILL_SWITCH,
            priority=TestPriority.CRITICAL,
            result=TestResult.PASS,
        ))
        suite.status = ConformanceSuiteStatus.PASSED

        # 3. Create certificate
        cert = manager.create_certificate_from_suite(
            suite=suite,
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Trading Ltd",
            algorithm_name="Execution Algorithm",
            issued_by="qa_team",
        )

        assert cert is not None

        # 4. Submit for approval
        manager.submit_for_approval(cert.certificate_id, "compliance_officer")
        assert cert.status == CertificateStatus.PENDING_APPROVAL

        # 5. Approve
        manager.approve_certificate(
            cert.certificate_id,
            "senior_compliance",
            "All requirements met",
        )
        assert cert.status == CertificateStatus.APPROVED

        # 6. Check deployment authorization
        can_deploy, reason = manager.can_deploy("ALGO-001", "1.0.0")
        assert can_deploy

        # 7. Generate deployment approval document
        doc = cert.generate_deployment_approval()
        assert "DEPLOYMENT AUTHORIZED" in doc

        # 8. Save for audit
        manager.save_certificate(cert)

        # 9. Verify can be loaded
        loaded = manager.load_certificate(cert.certificate_id)
        assert loaded.status == CertificateStatus.APPROVED
