# -*- coding: utf-8 -*-
"""
Tests for Approval Workflow - Phase 4.10

Tests cover:
- Approval request creation
- Approval/denial decisions
- Auto-approval for non-TRADING_IMPACTING
- Evidence generation
- Timeout handling
"""

from datetime import datetime, timedelta
from uuid import UUID

import pytest

from packages.shared.contracts.config import ChangeClass
from packages.agent.approval.manager import (
    ApprovalManager,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalDecision,
)
from packages.agent.approval.evidence import (
    compute_evidence_hash,
    EvidenceRecord,
    EvidenceStore,
    EvidenceChain,
    create_evidence_record,
)


class TestApprovalRequest:
    """Test ApprovalRequest dataclass."""

    def test_creation_defaults(self):
        """Test request creation with defaults."""
        request = ApprovalRequest(
            command_type="START_RUN",
            description="Start strategy",
        )

        assert request.command_type == "START_RUN"
        assert request.status == ApprovalStatus.PENDING
        assert request.change_class == ChangeClass.TRADING_IMPACTING
        assert request.request_id is not None

    def test_is_expired_false_initially(self):
        """Test request is not expired initially."""
        request = ApprovalRequest(
            expires_at=datetime.utcnow() + timedelta(minutes=5)
        )

        assert request.is_expired is False

    def test_is_expired_true_after_timeout(self):
        """Test request is expired after timeout."""
        request = ApprovalRequest(
            expires_at=datetime.utcnow() - timedelta(minutes=1)
        )

        assert request.is_expired is True

    def test_to_dict(self):
        """Test serialization."""
        request = ApprovalRequest(
            command_type="UPDATE_CONFIG",
            description="Update config",
            change_class=ChangeClass.OPERATIONAL,
        )

        data = request.to_dict()

        assert data["command_type"] == "UPDATE_CONFIG"
        assert data["description"] == "Update config"
        assert data["change_class"] == "operational"


class TestApprovalManager:
    """Test ApprovalManager operations."""

    @pytest.fixture
    def manager(self):
        """Create approval manager."""
        return ApprovalManager(approval_timeout_seconds=60)

    def test_create_request(self, manager):
        """Test creating approval request."""
        request = manager.create_request(
            command_type="START_RUN",
            description="Start momentum strategy",
            change_class=ChangeClass.TRADING_IMPACTING,
        )

        assert request.status == ApprovalStatus.PENDING
        assert request.command_type == "START_RUN"

    def test_auto_approve_operational(self, manager):
        """Test OPERATIONAL changes auto-approved."""
        # auto_approve_operational=True by default
        request = manager.create_request(
            command_type="UPDATE_LOG_LEVEL",
            description="Update log level",
            change_class=ChangeClass.OPERATIONAL,
        )

        assert request.status == ApprovalStatus.APPROVED
        assert "Auto-approved" in request.decision_reason

    def test_auto_approve_informational(self, manager):
        """Test INFORMATIONAL changes auto-approved."""
        request = manager.create_request(
            command_type="GET_STATUS",
            description="Get status",
            change_class=ChangeClass.INFORMATIONAL,
        )

        assert request.status == ApprovalStatus.APPROVED

    def test_trading_impacting_requires_manual(self, manager):
        """Test TRADING_IMPACTING requires manual approval."""
        request = manager.create_request(
            command_type="START_RUN",
            description="Start trading",
            change_class=ChangeClass.TRADING_IMPACTING,
        )

        assert request.status == ApprovalStatus.PENDING

    def test_approve_request(self, manager):
        """Test approving a request."""
        request = manager.create_request(
            command_type="START_RUN",
            description="Start trading",
            change_class=ChangeClass.TRADING_IMPACTING,
        )

        success = manager.approve(
            request.request_id,
            reason="Approved by admin",
            decided_by="admin",
        )

        assert success is True

        # Check history
        history = manager.get_history(limit=1)
        assert history[0].status == ApprovalStatus.APPROVED

    def test_deny_request(self, manager):
        """Test denying a request."""
        request = manager.create_request(
            command_type="START_RUN",
            description="Start trading",
            change_class=ChangeClass.TRADING_IMPACTING,
        )

        success = manager.deny(
            request.request_id,
            reason="Risk too high",
            decided_by="admin",
        )

        assert success is True

        history = manager.get_history(limit=1)
        assert history[0].status == ApprovalStatus.DENIED

    def test_get_pending_requests(self, manager):
        """Test getting pending requests."""
        manager.create_request("CMD1", "Desc 1", ChangeClass.TRADING_IMPACTING)
        manager.create_request("CMD2", "Desc 2", ChangeClass.TRADING_IMPACTING)

        pending = manager.get_pending_requests()

        assert len(pending) == 2

    def test_expired_request_cannot_be_approved(self, manager):
        """Test expired request cannot be approved."""
        request = ApprovalRequest(
            command_type="OLD_CMD",
            description="Old command",
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        manager._requests[request.request_id] = request

        success = manager.approve(request.request_id)

        assert success is False

    def test_decide_returns_request(self, manager):
        """Test decide method returns updated request."""
        request = manager.create_request(
            command_type="START_RUN",
            description="Start trading",
            change_class=ChangeClass.TRADING_IMPACTING,
        )

        result = manager.decide(
            request.request_id,
            approved=True,
            reason="OK",
        )

        assert result is not None
        assert result.status == ApprovalStatus.APPROVED

    def test_add_auto_approve_command(self, manager):
        """Test adding command to auto-approve whitelist."""
        manager.add_auto_approve_command("SPECIAL_CMD")

        request = manager.create_request(
            command_type="SPECIAL_CMD",
            description="Special command",
            change_class=ChangeClass.TRADING_IMPACTING,
        )

        assert request.status == ApprovalStatus.APPROVED


class TestEvidenceRecord:
    """Test EvidenceRecord and hash computation."""

    def test_compute_evidence_hash(self):
        """Test evidence hash computation."""
        data = {"key": "value", "number": 123}

        hash1 = compute_evidence_hash(data)
        hash2 = compute_evidence_hash(data)

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_create_evidence_record(self):
        """Test creating evidence record."""
        record = create_evidence_record(
            request_id="req-123",
            command_type="START_RUN",
            approved=True,
            decided_by="admin",
            config_hash="config-abc",
            reason="Approved for production",
        )

        assert record.request_id == "req-123"
        assert record.decision == "approved"
        assert record.evidence_hash is not None

    def test_verify_hash(self):
        """Test evidence hash verification."""
        record = create_evidence_record(
            request_id="req-123",
            command_type="START_RUN",
            approved=True,
            decided_by="admin",
        )

        assert record.verify_hash() is True

        # Tamper with record
        record.decision = "denied"
        assert record.verify_hash() is False


class TestEvidenceStore:
    """Test EvidenceStore operations."""

    @pytest.fixture
    def store(self):
        """Create evidence store."""
        return EvidenceStore()

    def test_add_and_get(self, store):
        """Test adding and retrieving records."""
        record = create_evidence_record(
            request_id="req-1",
            command_type="CMD",
            approved=True,
            decided_by="user",
        )

        store.add(record)
        retrieved = store.get(record.record_id)

        assert retrieved is not None
        assert retrieved.request_id == "req-1"

    def test_get_by_request(self, store):
        """Test getting record by request ID."""
        record = create_evidence_record(
            request_id="unique-req",
            command_type="CMD",
            approved=True,
            decided_by="user",
        )

        store.add(record)
        retrieved = store.get_by_request("unique-req")

        assert retrieved is not None
        assert retrieved.request_id == "unique-req"

    def test_verify_all(self, store):
        """Test verifying all records."""
        record1 = create_evidence_record("req-1", "CMD1", True, "user")
        record2 = create_evidence_record("req-2", "CMD2", False, "user")

        store.add(record1)
        store.add(record2)

        is_valid, invalid_ids = store.verify_all()

        assert is_valid is True
        assert invalid_ids == []

    def test_export_audit_log(self, store):
        """Test exporting audit log."""
        record = create_evidence_record("req-1", "CMD", True, "admin")
        store.add(record)

        log = store.export_audit_log()

        assert "APPROVED" in log
        assert "CMD" in log


class TestEvidenceChain:
    """Test EvidenceChain for linked audit trail."""

    @pytest.fixture
    def chain(self):
        """Create evidence chain."""
        return EvidenceChain()

    def test_add_creates_chain_link(self, chain):
        """Test adding records creates chain links."""
        record1 = create_evidence_record("req-1", "CMD1", True, "user")
        record2 = create_evidence_record("req-2", "CMD2", True, "user")

        chain.add(record1)
        hash1 = record1.evidence_hash

        chain.add(record2)

        # Record 2 should reference record 1's hash
        assert record2.metadata.get("previous_hash") == hash1

    def test_verify_chain_integrity(self, chain):
        """Test chain integrity verification."""
        record1 = create_evidence_record("req-1", "CMD1", True, "user")
        record2 = create_evidence_record("req-2", "CMD2", True, "user")

        chain.add(record1)
        chain.add(record2)

        is_valid, errors = chain.verify_chain()

        assert is_valid is True
        assert errors == []

    def test_get_latest(self, chain):
        """Test getting latest record."""
        record1 = create_evidence_record("req-1", "CMD1", True, "user")
        record2 = create_evidence_record("req-2", "CMD2", True, "user")

        chain.add(record1)
        chain.add(record2)

        latest = chain.get_latest()

        assert latest.request_id == "req-2"

    def test_export_and_import(self, chain):
        """Test exporting and importing chain."""
        record1 = create_evidence_record("req-1", "CMD1", True, "user")
        record2 = create_evidence_record("req-2", "CMD2", True, "user")

        chain.add(record1)
        chain.add(record2)

        exported = chain.export()

        new_chain = EvidenceChain()
        success = new_chain.import_chain(exported)

        assert success is True
        assert len(new_chain) == 2
