# -*- coding: utf-8 -*-
"""
Tests for CCEA Agent Daemon.

Tests:
- Command handler
- Approval manager
- Command filtering
"""

import pytest
from datetime import datetime, timedelta
import tempfile
from pathlib import Path

from ccea.agent.command_handler import (
    CommandHandler,
    CommandFilter,
    LocalCommandStatus,
    ALLOWED_COMMANDS,
    APPROVAL_REQUIRED,
    SAFETY_COMMANDS,
)
from ccea.agent.approval import (
    ApprovalManager,
    ApprovalStatus,
    AutoApprovalPolicy,
)
from ccea.models.protocol import CommandType, ChangeClass


class TestCommandFilter:
    """Tests for CommandFilter."""

    def test_allowed_commands(self):
        """Test that allowed commands pass filter."""
        filter = CommandFilter()

        for cmd_type in ALLOWED_COMMANDS:
            command = {
                "command_type": cmd_type,
                "deployment_id": "deploy_123",
                "idempotency_key": "idem_key_123456789012",
            }
            allowed, reason = filter.is_allowed(command)
            assert allowed is True, f"{cmd_type} should be allowed"

    def test_unknown_command_rejected(self):
        """Test that unknown commands are rejected."""
        filter = CommandFilter()

        command = {
            "command_type": "REQUEST_UNKNOWN_ACTION",
            "idempotency_key": "idem_key_123456789012",
        }

        allowed, reason = filter.is_allowed(command)
        assert allowed is False

    def test_prohibited_fields_rejected(self):
        """Test that order-like fields are rejected."""
        filter = CommandFilter()

        # Command with prohibited 'side' field
        command = {
            "command_type": "REQUEST_START_RUN",
            "deployment_id": "deploy_123",
            "idempotency_key": "idem_key_123456789012",
            "side": "BUY",  # PROHIBITED
        }

        allowed, reason = filter.is_allowed(command)
        assert allowed is False
        assert "side" in reason

    def test_nested_prohibited_fields(self):
        """Test detection of nested prohibited fields."""
        filter = CommandFilter()

        command = {
            "command_type": "REQUEST_START_RUN",
            "deployment_id": "deploy_123",
            "idempotency_key": "idem_key_123456789012",
            "config": {
                "nested": {
                    "price": 100.0,  # PROHIBITED nested
                }
            }
        }

        allowed, reason = filter.is_allowed(command)
        assert allowed is False

    def test_quantity_field_rejected(self):
        """Test quantity field is rejected."""
        filter = CommandFilter()

        command = {
            "command_type": "REQUEST_UPDATE_CONFIG",
            "deployment_id": "deploy_123",
            "idempotency_key": "idem_key_123456789012",
            "quantity": 100,  # PROHIBITED
        }

        allowed, reason = filter.is_allowed(command)
        assert allowed is False

    def test_order_type_field_rejected(self):
        """Test order_type field is rejected."""
        filter = CommandFilter()

        command = {
            "command_type": "REQUEST_START_RUN",
            "deployment_id": "deploy_123",
            "idempotency_key": "idem_key_123456789012",
            "order_type": "MARKET",  # PROHIBITED
        }

        allowed, reason = filter.is_allowed(command)
        assert allowed is False


class TestCommandHandler:
    """Tests for CommandHandler."""

    def test_handle_allowed_command(self):
        """Test handling allowed command."""
        handler = CommandHandler()

        command = {
            "command_id": "cmd_123",
            "command_type": "REQUEST_STOP_RUN",
            "deployment_id": "deploy_123",
            "idempotency_key": "idem_key_123456789012",
        }

        record = handler.handle_command(command)

        assert record.status == LocalCommandStatus.APPROVED  # Safety command
        assert record.command_type == "REQUEST_STOP_RUN"

    def test_handle_trading_impacting_requires_approval(self):
        """Test that trading impacting commands require approval."""
        handler = CommandHandler()

        command = {
            "command_id": "cmd_start_123",
            "command_type": "REQUEST_START_RUN",
            "deployment_id": "deploy_123",
            "artifact_digest": "sha256:test1234567890123456789012345678901234567890123456789012345678",
            "idempotency_key": "start_idem_key_12345678",
            "requires_approval": True,
        }

        record = handler.handle_command(command)

        assert record.status == LocalCommandStatus.AWAITING_APPROVAL
        assert record.requires_approval is True

    def test_handle_filtered_command(self):
        """Test handling of filtered command."""
        handler = CommandHandler()

        command = {
            "command_id": "cmd_bad",
            "command_type": "PLACE_ORDER",  # Not in allowlist
            "idempotency_key": "bad_idem_key_123456789",
        }

        record = handler.handle_command(command)

        assert record.status == LocalCommandStatus.FILTERED
        assert record.filter_reason is not None

    def test_command_idempotency(self):
        """Test command idempotency."""
        handler = CommandHandler()

        command = {
            "command_id": "cmd_idem",
            "command_type": "REQUEST_STOP_RUN",
            "deployment_id": "deploy_123",
            "idempotency_key": "unique_idem_key_123456",
        }

        record1 = handler.handle_command(command)
        record2 = handler.handle_command(command)

        assert record1.command_id == record2.command_id

    def test_approve_command(self):
        """Test command approval."""
        handler = CommandHandler()

        command = {
            "command_id": "cmd_approve",
            "command_type": "REQUEST_START_RUN",
            "deployment_id": "deploy_123",
            "artifact_digest": "sha256:approve123456789012345678901234567890123456789012345678901234",
            "idempotency_key": "approve_idem_key_12345",
            "requires_approval": True,
        }

        record = handler.handle_command(command)
        assert record.status == LocalCommandStatus.AWAITING_APPROVAL

        result = handler.approve_command("cmd_approve", "sha256:evidence")
        assert result is True

        updated = handler.get_command("cmd_approve")
        assert updated.status == LocalCommandStatus.APPROVED

    def test_reject_command(self):
        """Test command rejection."""
        handler = CommandHandler()

        command = {
            "command_id": "cmd_reject",
            "command_type": "REQUEST_START_RUN",
            "deployment_id": "deploy_123",
            "artifact_digest": "sha256:reject123456789012345678901234567890123456789012345678901234",
            "idempotency_key": "reject_idem_key_12345",
            "requires_approval": True,
        }

        record = handler.handle_command(command)
        result = handler.reject_command("cmd_reject", "Not approved by user")

        assert result is True
        updated = handler.get_command("cmd_reject")
        assert updated.status == LocalCommandStatus.REJECTED

    def test_execute_safety_command(self):
        """Test executing safety command."""
        handler = CommandHandler()

        command = {
            "command_id": "cmd_exec",
            "command_type": "REQUEST_STOP_RUN",
            "deployment_id": "deploy_123",
            "idempotency_key": "exec_idem_key_12345678",
        }

        record = handler.handle_command(command)
        result = handler.execute_command("cmd_exec")

        assert result is not None
        assert "stopped" in str(result) or "action" in result


class TestApprovalManager:
    """Tests for ApprovalManager."""

    def test_create_approval_request(self):
        """Test creating approval request."""
        manager = ApprovalManager(timeout_seconds=300)

        request = manager.create_request(
            command_id="cmd_123",
            command_type="REQUEST_START_RUN",
            deployment_id="deploy_123",
            artifact_digest="sha256:artifact1234567890123456789012345678901234567890123456789012",
        )

        assert request.status == ApprovalStatus.PENDING
        assert request.command_id == "cmd_123"

    def test_approve_request(self):
        """Test approving request."""
        manager = ApprovalManager()

        request = manager.create_request(
            command_id="cmd_approve_test",
            command_type="REQUEST_UPGRADE_ARTIFACT",
        )

        result = manager.approve(request.request_id, approver="test_user")

        assert result is not None
        assert result.approved is True
        assert result.evidence_hash is not None

    def test_reject_request(self):
        """Test rejecting request."""
        manager = ApprovalManager()

        request = manager.create_request(
            command_id="cmd_reject_test",
            command_type="REQUEST_START_RUN",
        )

        result = manager.reject(
            request.request_id,
            approver="test_user",
            reason="Risk too high",
        )

        assert result is not None
        assert result.approved is False
        assert result.reason == "Risk too high"

    def test_get_pending_requests(self):
        """Test getting pending requests."""
        manager = ApprovalManager()

        manager.create_request("cmd_1", "REQUEST_START_RUN")
        manager.create_request("cmd_2", "REQUEST_UPGRADE_ARTIFACT")

        pending = manager.get_pending()
        assert len(pending) == 2

    def test_approval_timeout(self):
        """Test approval request timeout."""
        manager = ApprovalManager(timeout_seconds=0)

        request = manager.create_request(
            command_id="cmd_timeout",
            command_type="REQUEST_START_RUN",
        )

        # Manually set expired
        request.expires_at = datetime.utcnow() - timedelta(seconds=1)

        # Should not be in pending anymore
        pending = manager.get_pending()
        timeout_request = next(
            (r for r in pending if r.request_id == request.request_id),
            None
        )

        # Expired requests should be removed or marked
        if timeout_request:
            assert timeout_request.is_expired() is True

    def test_auto_approval_disabled_by_default(self):
        """Test auto-approval is disabled by default."""
        manager = ApprovalManager()

        request = manager.create_request(
            command_id="cmd_auto",
            command_type="REQUEST_START_RUN",
        )

        # Should not be auto-approved
        assert request.status == ApprovalStatus.PENDING

    def test_auto_approval_policy(self):
        """Test auto-approval with policy."""
        policy = AutoApprovalPolicy(
            enabled=True,
            allowed_command_types={"REQUEST_UPDATE_CONFIG"},
            require_previous_approval=False,
        )

        manager = ApprovalManager(auto_policy=policy)

        # This command should be auto-approved
        request = manager.create_request(
            command_id="cmd_policy_auto",
            command_type="REQUEST_UPDATE_CONFIG",
        )

        assert request.status == ApprovalStatus.AUTO_APPROVED
        assert request.approver == "auto_policy"

    def test_auto_approval_respects_command_type(self):
        """Test auto-approval respects command type filter."""
        policy = AutoApprovalPolicy(
            enabled=True,
            allowed_command_types={"REQUEST_UPDATE_CONFIG"},
            require_previous_approval=False,
        )

        manager = ApprovalManager(auto_policy=policy)

        # START_RUN should NOT be auto-approved
        request = manager.create_request(
            command_id="cmd_not_auto",
            command_type="REQUEST_START_RUN",
        )

        assert request.status == ApprovalStatus.PENDING

    def test_approval_history(self):
        """Test approval history."""
        manager = ApprovalManager()

        request = manager.create_request("cmd_history", "REQUEST_START_RUN")
        manager.approve(request.request_id)

        history = manager.get_history()
        assert len(history) == 1
        assert history[0].command_id == "cmd_history"


class TestApprovalRequirements:
    """Tests for approval requirements per command type."""

    def test_start_run_requires_approval(self):
        """Test REQUEST_START_RUN requires approval."""
        assert "REQUEST_START_RUN" in APPROVAL_REQUIRED

    def test_upgrade_artifact_requires_approval(self):
        """Test REQUEST_UPGRADE_ARTIFACT requires approval."""
        assert "REQUEST_UPGRADE_ARTIFACT" in APPROVAL_REQUIRED

    def test_stop_run_no_approval(self):
        """Test REQUEST_STOP_RUN doesn't require approval."""
        assert "REQUEST_STOP_RUN" in SAFETY_COMMANDS
        assert "REQUEST_STOP_RUN" not in APPROVAL_REQUIRED

    def test_pause_run_no_approval(self):
        """Test REQUEST_PAUSE_RUN doesn't require approval."""
        assert "REQUEST_PAUSE_RUN" in SAFETY_COMMANDS

    def test_export_logs_requires_approval(self):
        """Test REQUEST_EXPORT_LOGS requires approval (data sensitive)."""
        assert "REQUEST_EXPORT_LOGS" in APPROVAL_REQUIRED
