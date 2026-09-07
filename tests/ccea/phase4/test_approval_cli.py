# -*- coding: utf-8 -*-
"""
Tests for Approval CLI - Phase 4.10

Comprehensive tests for the local approval CLI tool.
Tests cover:
- CLI initialization and configuration
- List pending requests
- Show request details
- Approve/deny operations
- History display
- Interactive mode
- Argument parsing
- Color formatting
- Error handling
"""

import sys
from datetime import datetime, timedelta
from io import StringIO
from typing import List, Optional
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from packages.shared.contracts.config import ChangeClass
from packages.agent.approval.manager import (
    ApprovalManager,
    ApprovalRequest,
    ApprovalStatus,
)
from packages.agent.approval.cli import (
    ApprovalCLI,
    Colors,
    _color,
    _format_time,
    _format_status,
    create_cli_parser,
    main,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_manager():
    """Create a mock approval manager."""
    manager = MagicMock(spec=ApprovalManager)
    manager.get_pending_requests.return_value = []
    manager.get_history.return_value = []
    return manager


@pytest.fixture
def approval_manager():
    """Create a real approval manager."""
    return ApprovalManager(approval_timeout_seconds=300)


@pytest.fixture
def pending_request():
    """Create a pending approval request."""
    return ApprovalRequest(
        request_id=uuid4(),
        command_type="REQUEST_START_RUN",
        description="Start momentum strategy",
        change_class=ChangeClass.TRADING_IMPACTING,
        status=ApprovalStatus.PENDING,
        expires_at=datetime.utcnow() + timedelta(minutes=30),
        config_digest_old="sha256:abc123" + "0" * 56,
        config_digest_new="sha256:def456" + "0" * 56,
        diff_summary="+max_position: 1000\n-max_position: 500",
    )


@pytest.fixture
def approved_request():
    """Create an approved request."""
    return ApprovalRequest(
        request_id=uuid4(),
        command_type="REQUEST_UPDATE_CONFIG",
        description="Update risk limits",
        change_class=ChangeClass.TRADING_IMPACTING,
        status=ApprovalStatus.APPROVED,
        decided_at=datetime.utcnow() - timedelta(hours=1),
        decision_reason="Approved by admin",
    )


@pytest.fixture
def expired_request():
    """Create an expired request."""
    return ApprovalRequest(
        request_id=uuid4(),
        command_type="REQUEST_START_RUN",
        description="Old request",
        change_class=ChangeClass.TRADING_IMPACTING,
        status=ApprovalStatus.PENDING,
        expires_at=datetime.utcnow() - timedelta(hours=1),
    )


@pytest.fixture
def cli(mock_manager):
    """Create CLI instance with mock manager."""
    return ApprovalCLI(mock_manager)


# ============================================================================
# Test Helper Functions
# ============================================================================


class TestColorHelpers:
    """Test color formatting helpers."""

    def test_color_with_tty(self):
        """Test color is applied when terminal supports it."""
        with patch.object(sys.stdout, "isatty", return_value=True):
            result = _color("test", Colors.RED)
            assert Colors.RED in result
            assert Colors.RESET in result
            assert "test" in result

    def test_color_without_tty(self):
        """Test color is not applied when terminal doesn't support it."""
        with patch.object(sys.stdout, "isatty", return_value=False):
            result = _color("test", Colors.RED)
            assert result == "test"
            assert Colors.RED not in result

    def test_format_time_just_now(self):
        """Test time formatting for recent times."""
        dt = datetime.utcnow() - timedelta(seconds=30)
        result = _format_time(dt)
        assert result == "just now"

    def test_format_time_minutes_ago(self):
        """Test time formatting for minutes ago."""
        dt = datetime.utcnow() - timedelta(minutes=15)
        result = _format_time(dt)
        assert "15m ago" in result

    def test_format_time_hours_ago(self):
        """Test time formatting for hours ago."""
        dt = datetime.utcnow() - timedelta(hours=3)
        result = _format_time(dt)
        assert "3h ago" in result

    def test_format_time_days_ago(self):
        """Test time formatting for days ago."""
        dt = datetime.utcnow() - timedelta(days=2)
        result = _format_time(dt)
        # Should show date format
        assert "-" in result  # ISO format date separator

    def test_format_time_none(self):
        """Test time formatting for None."""
        result = _format_time(None)
        assert result == "N/A"

    def test_format_status_pending(self):
        """Test status formatting for pending."""
        with patch.object(sys.stdout, "isatty", return_value=True):
            result = _format_status("pending")
            assert "PENDING" in result
            assert Colors.YELLOW in result

    def test_format_status_approved(self):
        """Test status formatting for approved."""
        with patch.object(sys.stdout, "isatty", return_value=True):
            result = _format_status("approved")
            assert "APPROVED" in result
            assert Colors.GREEN in result

    def test_format_status_denied(self):
        """Test status formatting for denied."""
        with patch.object(sys.stdout, "isatty", return_value=True):
            result = _format_status("denied")
            assert "DENIED" in result
            assert Colors.RED in result

    def test_format_status_expired(self):
        """Test status formatting for expired."""
        with patch.object(sys.stdout, "isatty", return_value=True):
            result = _format_status("expired")
            assert "EXPIRED" in result
            assert Colors.DIM in result

    def test_format_status_unknown(self):
        """Test status formatting for unknown status."""
        with patch.object(sys.stdout, "isatty", return_value=False):
            result = _format_status("unknown_status")
            assert "UNKNOWN_STATUS" in result


# ============================================================================
# Test ApprovalCLI Class
# ============================================================================


class TestApprovalCLIInit:
    """Test ApprovalCLI initialization."""

    def test_init_with_manager(self, mock_manager):
        """Test CLI initialization with manager."""
        cli = ApprovalCLI(mock_manager)
        assert cli._manager == mock_manager
        assert cli._config_manager is None
        assert cli._verbose is False

    def test_init_with_config_manager(self, mock_manager):
        """Test CLI initialization with config manager."""
        config_mgr = MagicMock()
        cli = ApprovalCLI(mock_manager, config_manager=config_mgr, verbose=True)
        assert cli._config_manager == config_mgr
        assert cli._verbose is True


class TestApprovalCLIListPending:
    """Test list_pending functionality."""

    def test_list_pending_empty(self, cli, mock_manager, capsys):
        """Test listing when no pending requests."""
        mock_manager.get_pending_requests.return_value = []

        result = cli.list_pending()

        assert result == []
        captured = capsys.readouterr()
        assert "No pending requests" in captured.out

    def test_list_pending_with_requests(self, cli, mock_manager, pending_request, capsys):
        """Test listing with pending requests."""
        mock_manager.get_pending_requests.return_value = [pending_request]

        result = cli.list_pending()

        assert len(result) == 1
        captured = capsys.readouterr()
        assert "REQUEST_START_RUN" in captured.out
        assert "trading_impacting" in captured.out  # ChangeClass.value is lowercase
        assert "Total: 1 pending request(s)" in captured.out

    def test_list_pending_multiple(self, cli, mock_manager, pending_request, capsys):
        """Test listing multiple pending requests."""
        req2 = ApprovalRequest(
            command_type="REQUEST_UPDATE_CONFIG",
            description="Update config",
            change_class=ChangeClass.TRADING_IMPACTING,
        )
        mock_manager.get_pending_requests.return_value = [pending_request, req2]

        result = cli.list_pending()

        assert len(result) == 2
        captured = capsys.readouterr()
        assert "Total: 2 pending request(s)" in captured.out


class TestApprovalCLIShowRequest:
    """Test show_request functionality."""

    def test_show_request_found(self, cli, mock_manager, pending_request, capsys):
        """Test showing existing request details."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]

        result = cli.show_request(str(pending_request.request_id))

        assert result is not None
        assert result["command_type"] == "REQUEST_START_RUN"
        captured = capsys.readouterr()
        assert "Approval Request Details" in captured.out
        assert "REQUEST_START_RUN" in captured.out

    def test_show_request_not_found(self, cli, mock_manager, capsys):
        """Test showing non-existent request."""
        mock_manager.get_request.return_value = None
        mock_manager.get_pending_requests.return_value = []

        result = cli.show_request("nonexistent-id")

        assert result is None
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_show_request_with_diff(self, cli, mock_manager, pending_request, capsys):
        """Test showing request with diff summary."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]

        result = cli.show_request(str(pending_request.request_id))

        captured = capsys.readouterr()
        assert "Diff Summary" in captured.out
        assert "max_position" in captured.out

    def test_show_request_with_artifact_digest(self, cli, mock_manager, capsys):
        """Test showing request with artifact digest."""
        request = ApprovalRequest(
            command_type="REQUEST_UPGRADE_ARTIFACT",
            description="Upgrade strategy",
            artifact_digest="sha256:" + "a" * 64,
        )
        mock_manager.get_request.return_value = request
        mock_manager.get_pending_requests.return_value = [request]

        cli.show_request(str(request.request_id))

        captured = capsys.readouterr()
        assert "Artifact" in captured.out

    def test_show_request_with_details(self, cli, mock_manager, capsys):
        """Test showing request with additional details."""
        request = ApprovalRequest(
            command_type="REQUEST_START_RUN",
            description="Start",
            details={"key1": "value1", "key2": "value2"},
        )
        mock_manager.get_request.return_value = request
        mock_manager.get_pending_requests.return_value = [request]

        cli.show_request(str(request.request_id))

        captured = capsys.readouterr()
        assert "Details" in captured.out


class TestApprovalCLIApprove:
    """Test approve functionality."""

    def test_approve_success(self, cli, mock_manager, pending_request, capsys):
        """Test successful approval."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]
        mock_manager.approve.return_value = True

        result = cli.approve(
            str(pending_request.request_id),
            reason="Approved for production",
            user="admin",
        )

        assert result is True
        mock_manager.approve.assert_called_once()
        captured = capsys.readouterr()
        assert "APPROVED" in captured.out

    def test_approve_with_reason(self, cli, mock_manager, pending_request, capsys):
        """Test approval with reason shown."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]
        mock_manager.approve.return_value = True

        cli.approve(str(pending_request.request_id), reason="Risk review passed")

        captured = capsys.readouterr()
        assert "Risk review passed" in captured.out

    def test_approve_not_found(self, cli, mock_manager, capsys):
        """Test approval of non-existent request."""
        mock_manager.get_request.return_value = None
        mock_manager.get_pending_requests.return_value = []

        result = cli.approve("nonexistent-id")

        assert result is False
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_approve_failed(self, cli, mock_manager, pending_request, capsys):
        """Test failed approval."""
        pending_request._is_expired = False
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]
        mock_manager.approve.return_value = False

        result = cli.approve(str(pending_request.request_id))

        assert result is False
        captured = capsys.readouterr()
        assert "Failed to approve" in captured.out

    def test_approve_expired_request(self, cli, mock_manager, expired_request, capsys):
        """Test approval of expired request."""
        mock_manager.get_request.return_value = expired_request
        mock_manager.get_pending_requests.return_value = [expired_request]
        mock_manager.approve.return_value = False

        result = cli.approve(str(expired_request.request_id))

        assert result is False
        captured = capsys.readouterr()
        assert "expired" in captured.out.lower() or "Failed" in captured.out


class TestApprovalCLIDeny:
    """Test deny functionality."""

    def test_deny_success(self, cli, mock_manager, pending_request, capsys):
        """Test successful denial."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]
        mock_manager.deny.return_value = True

        result = cli.deny(
            str(pending_request.request_id),
            reason="Risk limits exceeded",
            user="admin",
        )

        assert result is True
        mock_manager.deny.assert_called_once()
        captured = capsys.readouterr()
        assert "DENIED" in captured.out

    def test_deny_with_reason(self, cli, mock_manager, pending_request, capsys):
        """Test denial with reason shown."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]
        mock_manager.deny.return_value = True

        cli.deny(str(pending_request.request_id), reason="Compliance issue")

        captured = capsys.readouterr()
        assert "Compliance issue" in captured.out

    def test_deny_not_found(self, cli, mock_manager, capsys):
        """Test denial of non-existent request."""
        mock_manager.get_request.return_value = None
        mock_manager.get_pending_requests.return_value = []

        result = cli.deny("nonexistent-id")

        assert result is False
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_deny_failed(self, cli, mock_manager, pending_request, capsys):
        """Test failed denial."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]
        mock_manager.deny.return_value = False

        result = cli.deny(str(pending_request.request_id))

        assert result is False
        captured = capsys.readouterr()
        assert "Failed to deny" in captured.out


class TestApprovalCLIShowHistory:
    """Test show_history functionality."""

    def test_show_history_empty(self, cli, mock_manager, capsys):
        """Test showing empty history."""
        mock_manager.get_history.return_value = []

        result = cli.show_history()

        assert result == []
        captured = capsys.readouterr()
        assert "No history available" in captured.out

    def test_show_history_with_entries(self, cli, mock_manager, approved_request, capsys):
        """Test showing history with entries."""
        mock_manager.get_history.return_value = [approved_request]

        result = cli.show_history(limit=20)

        assert len(result) == 1
        mock_manager.get_history.assert_called_once_with(limit=20)
        captured = capsys.readouterr()
        assert "Approval History" in captured.out

    def test_show_history_with_reason(self, cli, mock_manager, approved_request, capsys):
        """Test showing history with decision reason."""
        mock_manager.get_history.return_value = [approved_request]

        cli.show_history()

        captured = capsys.readouterr()
        assert "Approved by admin" in captured.out


class TestApprovalCLIFindRequest:
    """Test _find_request helper."""

    def test_find_request_by_full_uuid(self, cli, mock_manager, pending_request):
        """Test finding request by full UUID."""
        mock_manager.get_request.return_value = pending_request

        result = cli._find_request(str(pending_request.request_id))

        assert result == pending_request
        mock_manager.get_request.assert_called_once()

    def test_find_request_by_prefix(self, cli, mock_manager, pending_request):
        """Test finding request by UUID prefix."""
        # get_request returns None for prefix, fall through to prefix match
        mock_manager.get_request.return_value = None
        mock_manager.get_pending_requests.return_value = [pending_request]

        prefix = str(pending_request.request_id)[:8]
        result = cli._find_request(prefix)

        assert result == pending_request

    def test_find_request_not_found(self, cli, mock_manager):
        """Test request not found."""
        mock_manager.get_request.return_value = None
        mock_manager.get_pending_requests.return_value = []

        result = cli._find_request("nonexistent")

        assert result is None


class TestApprovalCLIInteractive:
    """Test interactive functionality."""

    def test_interactive_approve_yes(self, cli, mock_manager, pending_request, capsys):
        """Test interactive approval with 'yes'."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]
        mock_manager.approve.return_value = True

        with patch("builtins.input", side_effect=["yes", "Manual approval"]):
            result = cli.interactive_approve(str(pending_request.request_id))

        assert result is True

    def test_interactive_approve_no(self, cli, mock_manager, pending_request, capsys):
        """Test interactive denial with 'no'."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]
        mock_manager.deny.return_value = True

        with patch("builtins.input", side_effect=["no", "Risk too high"]):
            result = cli.interactive_approve(str(pending_request.request_id))

        assert result is True  # deny was successful

    def test_interactive_approve_cancel(self, cli, mock_manager, pending_request, capsys):
        """Test interactive cancellation."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]

        with patch("builtins.input", return_value="maybe"):
            result = cli.interactive_approve(str(pending_request.request_id))

        assert result is False
        captured = capsys.readouterr()
        assert "Cancelled" in captured.out

    def test_interactive_approve_not_found(self, cli, mock_manager, capsys):
        """Test interactive approve with non-existent request."""
        mock_manager.get_request.return_value = None
        mock_manager.get_pending_requests.return_value = []

        result = cli.interactive_approve("nonexistent")

        assert result is False


class TestApprovalCLIRunInteractive:
    """Test run_interactive functionality."""

    def test_run_interactive_quit(self, cli, capsys):
        """Test interactive mode quit command."""
        with patch("builtins.input", side_effect=["quit"]):
            cli.run_interactive()

        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_run_interactive_exit(self, cli, capsys):
        """Test interactive mode exit command."""
        with patch("builtins.input", side_effect=["exit"]):
            cli.run_interactive()

        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_run_interactive_help(self, cli, capsys):
        """Test interactive mode help command."""
        with patch("builtins.input", side_effect=["help", "quit"]):
            cli.run_interactive()

        captured = capsys.readouterr()
        assert "Available Commands" in captured.out
        assert "list" in captured.out
        assert "approve" in captured.out

    def test_run_interactive_list(self, cli, mock_manager, capsys):
        """Test interactive mode list command."""
        mock_manager.get_pending_requests.return_value = []

        with patch("builtins.input", side_effect=["list", "quit"]):
            cli.run_interactive()

        captured = capsys.readouterr()
        assert "Pending Approval Requests" in captured.out

    def test_run_interactive_show(self, cli, mock_manager, pending_request, capsys):
        """Test interactive mode show command."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]

        with patch("builtins.input", side_effect=[f"show {pending_request.request_id}", "quit"]):
            cli.run_interactive()

        captured = capsys.readouterr()
        assert "Approval Request Details" in captured.out

    def test_run_interactive_approve_command(self, cli, mock_manager, pending_request, capsys):
        """Test interactive mode approve command."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]
        mock_manager.approve.return_value = True

        with patch(
            "builtins.input",
            side_effect=[f"approve {pending_request.request_id} test reason", "quit"],
        ):
            cli.run_interactive()

        mock_manager.approve.assert_called_once()

    def test_run_interactive_deny_command(self, cli, mock_manager, pending_request, capsys):
        """Test interactive mode deny command."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]
        mock_manager.deny.return_value = True

        with patch("builtins.input", side_effect=[f"deny {pending_request.request_id}", "quit"]):
            cli.run_interactive()

        mock_manager.deny.assert_called_once()

    def test_run_interactive_history(self, cli, mock_manager, capsys):
        """Test interactive mode history command."""
        mock_manager.get_history.return_value = []

        with patch("builtins.input", side_effect=["history", "quit"]):
            cli.run_interactive()

        captured = capsys.readouterr()
        assert "Approval History" in captured.out

    def test_run_interactive_history_with_limit(self, cli, mock_manager, capsys):
        """Test interactive mode history command with limit."""
        mock_manager.get_history.return_value = []

        with patch("builtins.input", side_effect=["history 5", "quit"]):
            cli.run_interactive()

        mock_manager.get_history.assert_called_with(limit=5)

    def test_run_interactive_decide(self, cli, mock_manager, pending_request, capsys):
        """Test interactive mode decide command."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]
        mock_manager.approve.return_value = True

        with patch(
            "builtins.input",
            side_effect=[f"decide {pending_request.request_id}", "yes", "test", "quit"],
        ):
            cli.run_interactive()

        mock_manager.approve.assert_called_once()

    def test_run_interactive_unknown_command(self, cli, capsys):
        """Test interactive mode unknown command."""
        with patch("builtins.input", side_effect=["unknown_cmd", "quit"]):
            cli.run_interactive()

        captured = capsys.readouterr()
        assert "Unknown command" in captured.out

    def test_run_interactive_empty_input(self, cli, capsys):
        """Test interactive mode with empty input."""
        with patch("builtins.input", side_effect=["", "quit"]):
            cli.run_interactive()

        # Should not crash, just continue
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_run_interactive_keyboard_interrupt(self, cli, capsys):
        """Test interactive mode with keyboard interrupt."""
        with patch("builtins.input", side_effect=KeyboardInterrupt()):
            cli.run_interactive()

        captured = capsys.readouterr()
        assert "Exiting" in captured.out

    def test_run_interactive_eof(self, cli, capsys):
        """Test interactive mode with EOF."""
        with patch("builtins.input", side_effect=EOFError()):
            cli.run_interactive()

        captured = capsys.readouterr()
        assert "Exiting" in captured.out

    def test_run_interactive_ls_alias(self, cli, mock_manager, capsys):
        """Test interactive mode 'ls' alias for list."""
        mock_manager.get_pending_requests.return_value = []

        with patch("builtins.input", side_effect=["ls", "quit"]):
            cli.run_interactive()

        captured = capsys.readouterr()
        assert "Pending Approval Requests" in captured.out

    def test_run_interactive_q_alias(self, cli, capsys):
        """Test interactive mode 'q' alias for quit."""
        with patch("builtins.input", side_effect=["q"]):
            cli.run_interactive()

        captured = capsys.readouterr()
        assert "Goodbye" in captured.out


class TestPrintHelp:
    """Test _print_help functionality."""

    def test_print_help_content(self, cli, capsys):
        """Test help message content."""
        cli._print_help()

        captured = capsys.readouterr()
        assert "Available Commands" in captured.out
        assert "list" in captured.out
        assert "show" in captured.out
        assert "approve" in captured.out
        assert "deny" in captured.out
        assert "decide" in captured.out
        assert "history" in captured.out
        assert "quit" in captured.out
        assert "Examples" in captured.out


# ============================================================================
# Test Argument Parser
# ============================================================================


class TestCreateCLIParser:
    """Test argument parser creation."""

    def test_parser_creation(self):
        """Test parser is created successfully."""
        parser = create_cli_parser()
        assert parser is not None
        assert parser.prog == "ccea-approve"

    def test_parser_list_command(self):
        """Test parser handles list command."""
        parser = create_cli_parser()
        args = parser.parse_args(["list"])
        assert args.command == "list"

    def test_parser_list_verbose(self):
        """Test parser handles list with verbose flag."""
        parser = create_cli_parser()
        args = parser.parse_args(["list", "-v"])
        assert args.command == "list"
        assert args.verbose is True

    def test_parser_show_command(self):
        """Test parser handles show command."""
        parser = create_cli_parser()
        args = parser.parse_args(["show", "abc123"])
        assert args.command == "show"
        assert args.request_id == "abc123"

    def test_parser_approve_command(self):
        """Test parser handles approve command."""
        parser = create_cli_parser()
        args = parser.parse_args(["approve", "abc123", "-r", "approved"])
        assert args.command == "approve"
        assert args.request_id == "abc123"
        assert args.reason == "approved"

    def test_parser_approve_with_user(self):
        """Test parser handles approve with user."""
        parser = create_cli_parser()
        args = parser.parse_args(["approve", "abc123", "-u", "admin"])
        assert args.user == "admin"

    def test_parser_deny_command(self):
        """Test parser handles deny command."""
        parser = create_cli_parser()
        args = parser.parse_args(["deny", "abc123", "-r", "rejected"])
        assert args.command == "deny"
        assert args.request_id == "abc123"
        assert args.reason == "rejected"

    def test_parser_history_command(self):
        """Test parser handles history command."""
        parser = create_cli_parser()
        args = parser.parse_args(["history", "-n", "10"])
        assert args.command == "history"
        assert args.limit == 10

    def test_parser_history_default_limit(self):
        """Test parser uses default history limit."""
        parser = create_cli_parser()
        args = parser.parse_args(["history"])
        assert args.limit == 20

    def test_parser_interactive_command(self):
        """Test parser handles interactive command."""
        parser = create_cli_parser()
        args = parser.parse_args(["interactive"])
        assert args.command == "interactive"

    def test_parser_no_command(self):
        """Test parser with no command."""
        parser = create_cli_parser()
        args = parser.parse_args([])
        assert args.command is None


# ============================================================================
# Test Main Function
# ============================================================================


class TestMainFunction:
    """Test main entry point."""

    def test_main_list(self, mock_manager, capsys):
        """Test main with list command."""
        mock_manager.get_pending_requests.return_value = []

        exit_code = main(mock_manager, ["list"])

        assert exit_code == 0

    def test_main_show_found(self, mock_manager, pending_request, capsys):
        """Test main with show command - found."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]

        exit_code = main(mock_manager, ["show", str(pending_request.request_id)])

        assert exit_code == 0

    def test_main_show_not_found(self, mock_manager, capsys):
        """Test main with show command - not found."""
        mock_manager.get_request.return_value = None
        mock_manager.get_pending_requests.return_value = []

        exit_code = main(mock_manager, ["show", "nonexistent"])

        assert exit_code == 1

    def test_main_approve_success(self, mock_manager, pending_request, capsys):
        """Test main with approve command - success."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]
        mock_manager.approve.return_value = True

        exit_code = main(mock_manager, ["approve", str(pending_request.request_id)])

        assert exit_code == 0

    def test_main_approve_failure(self, mock_manager, pending_request, capsys):
        """Test main with approve command - failure."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]
        mock_manager.approve.return_value = False

        exit_code = main(mock_manager, ["approve", str(pending_request.request_id)])

        assert exit_code == 1

    def test_main_deny_success(self, mock_manager, pending_request, capsys):
        """Test main with deny command - success."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]
        mock_manager.deny.return_value = True

        exit_code = main(mock_manager, ["deny", str(pending_request.request_id)])

        assert exit_code == 0

    def test_main_deny_failure(self, mock_manager, pending_request, capsys):
        """Test main with deny command - failure."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]
        mock_manager.deny.return_value = False

        exit_code = main(mock_manager, ["deny", str(pending_request.request_id)])

        assert exit_code == 1

    def test_main_history(self, mock_manager, capsys):
        """Test main with history command."""
        mock_manager.get_history.return_value = []

        exit_code = main(mock_manager, ["history"])

        assert exit_code == 0

    def test_main_interactive(self, mock_manager, capsys):
        """Test main with interactive command."""
        with patch("builtins.input", side_effect=["quit"]):
            exit_code = main(mock_manager, ["interactive"])

        assert exit_code == 0

    def test_main_no_command_starts_interactive(self, mock_manager, capsys):
        """Test main with no command starts interactive."""
        with patch("builtins.input", side_effect=["quit"]):
            exit_code = main(mock_manager, [])

        assert exit_code == 0

    def test_main_with_reason_and_user(self, mock_manager, pending_request, capsys):
        """Test main with reason and user arguments."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]
        mock_manager.approve.return_value = True

        exit_code = main(
            mock_manager,
            [
                "approve",
                str(pending_request.request_id),
                "-r",
                "ready for production review",
                "-u",
                "admin_user",
            ],
        )

        assert exit_code == 0
        mock_manager.approve.assert_called_once()
        call_kwargs = mock_manager.approve.call_args
        assert call_kwargs[1]["reason"] == "ready for production review"
        assert call_kwargs[1]["decided_by"] == "admin_user"


# ============================================================================
# Test Colors Class
# ============================================================================


class TestColorsClass:
    """Test Colors class constants."""

    def test_colors_defined(self):
        """Test all color constants are defined."""
        assert Colors.RESET == "\033[0m"
        assert Colors.RED == "\033[91m"
        assert Colors.GREEN == "\033[92m"
        assert Colors.YELLOW == "\033[93m"
        assert Colors.BLUE == "\033[94m"
        assert Colors.CYAN == "\033[96m"
        assert Colors.BOLD == "\033[1m"
        assert Colors.DIM == "\033[2m"


# ============================================================================
# Integration Tests
# ============================================================================


class TestCLIIntegration:
    """Integration tests with real ApprovalManager."""

    def test_full_workflow(self, approval_manager, capsys):
        """Test full approval workflow through CLI."""
        cli = ApprovalCLI(approval_manager)

        # Create a request
        request = approval_manager.create_request(
            command_type="REQUEST_START_RUN",
            description="Start momentum strategy",
            change_class=ChangeClass.TRADING_IMPACTING,
        )

        # List pending
        pending = cli.list_pending()
        assert len(pending) == 1

        # Show request
        details = cli.show_request(str(request.request_id))
        assert details is not None

        # Approve
        success = cli.approve(
            str(request.request_id), reason="Ready for production review", user="admin"
        )
        assert success is True

        # Verify in history
        history = cli.show_history()
        assert len(history) == 1
        assert history[0]["status"] == "approved"

    def test_deny_workflow(self, approval_manager, capsys):
        """Test denial workflow through CLI."""
        cli = ApprovalCLI(approval_manager)

        # Create a request
        request = approval_manager.create_request(
            command_type="REQUEST_UPDATE_CONFIG",
            description="Update risk limits",
            change_class=ChangeClass.TRADING_IMPACTING,
        )

        # Deny
        success = cli.deny(str(request.request_id), reason="Risk too high", user="compliance")
        assert success is True

        # Verify in history
        history = cli.show_history()
        assert len(history) == 1
        assert history[0]["status"] == "denied"

    def test_prefix_matching(self, approval_manager, capsys):
        """Test request lookup by UUID prefix."""
        cli = ApprovalCLI(approval_manager)

        # Create a request
        request = approval_manager.create_request(
            command_type="REQUEST_START_RUN",
            description="Test",
            change_class=ChangeClass.TRADING_IMPACTING,
        )

        # Find by prefix
        prefix = str(request.request_id)[:8]
        found = cli._find_request(prefix)
        assert found is not None
        assert found.request_id == request.request_id

    def test_operational_auto_approved(self, approval_manager, capsys):
        """Test OPERATIONAL changes are auto-approved."""
        cli = ApprovalCLI(approval_manager)

        # Create operational request
        request = approval_manager.create_request(
            command_type="UPDATE_LOG_LEVEL",
            description="Update logging",
            change_class=ChangeClass.OPERATIONAL,
        )

        # Should be auto-approved
        assert request.status == ApprovalStatus.APPROVED

        # No pending requests
        pending = cli.list_pending()
        assert len(pending) == 0


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_request_with_none_fields(self, cli, mock_manager, capsys):
        """Test showing request with None optional fields."""
        request = ApprovalRequest(
            command_type="REQUEST_START_RUN",
            description=None,  # None description
        )
        mock_manager.get_request.return_value = request
        mock_manager.get_pending_requests.return_value = [request]

        result = cli.show_request(str(request.request_id))

        assert result is not None
        captured = capsys.readouterr()
        assert "N/A" in captured.out

    def test_show_request_expired_indicator(self, cli, mock_manager, expired_request, capsys):
        """Test expired indicator in request summary."""
        mock_manager.get_pending_requests.return_value = [expired_request]

        cli.list_pending()

        captured = capsys.readouterr()
        assert "EXPIRED" in captured.out

    def test_empty_reason_approve(self, cli, mock_manager, pending_request, capsys):
        """Test approval with empty reason."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]
        mock_manager.approve.return_value = True

        result = cli.approve(str(pending_request.request_id), reason="")

        assert result is True
        # Reason line should not appear
        captured = capsys.readouterr()
        assert "Reason:" not in captured.out

    def test_special_characters_in_reason(self, cli, mock_manager, pending_request, capsys):
        """Test approval with special characters in reason."""
        mock_manager.get_request.return_value = pending_request
        mock_manager.get_pending_requests.return_value = [pending_request]
        mock_manager.approve.return_value = True

        result = cli.approve(
            str(pending_request.request_id), reason="Approved: <script>alert('xss')</script>"
        )

        assert result is True
