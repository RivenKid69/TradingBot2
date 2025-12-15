# -*- coding: utf-8 -*-
"""
Local Approval CLI - Command-line interface for approval workflow.

AGENT ZONE ONLY.

Provides interactive approval for TRADING_IMPACTING changes:
- View pending approval requests
- Show config/artifact diffs
- Approve or deny changes
- Set auto-approve policies

Design Doc Phase 4.10:
- Local approval UX: agent показывает diff конфигурации с текущей
- Стратегию можно approve/deny через локальный CLI
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from packages.agent.approval.manager import ApprovalManager, ApprovalRequest


# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


def _color(text: str, color: str) -> str:
    """Apply color to text if terminal supports it."""
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.RESET}"
    return text


def _format_time(dt: datetime) -> str:
    """Format datetime for display."""
    if dt is None:
        return "N/A"
    delta = datetime.utcnow() - dt
    if delta < timedelta(minutes=1):
        return "just now"
    if delta < timedelta(hours=1):
        return f"{int(delta.total_seconds() / 60)}m ago"
    if delta < timedelta(days=1):
        return f"{int(delta.total_seconds() / 3600)}h ago"
    return dt.strftime("%Y-%m-%d %H:%M")


def _format_status(status: str) -> str:
    """Format status with color."""
    colors = {
        "pending": Colors.YELLOW,
        "approved": Colors.GREEN,
        "denied": Colors.RED,
        "expired": Colors.DIM,
        "cancelled": Colors.DIM,
    }
    return _color(status.upper(), colors.get(status.lower(), Colors.RESET))


class ApprovalCLI:
    """
    Interactive CLI for local approval workflow.

    Usage:
        cli = ApprovalCLI(approval_manager)
        cli.run()  # Interactive mode

        # Or programmatic
        cli.list_pending()
        cli.approve(request_id, reason="Approved by admin")
    """

    def __init__(
        self,
        manager: "ApprovalManager",
        config_manager: Optional[Any] = None,
        verbose: bool = False,
    ):
        """
        Initialize CLI.

        Args:
            manager: ApprovalManager instance
            config_manager: ConfigManager for diff display
            verbose: Verbose output mode
        """
        self._manager = manager
        self._config_manager = config_manager
        self._verbose = verbose

    def list_pending(self) -> List[Dict[str, Any]]:
        """
        List all pending approval requests.

        Returns:
            List of pending request summaries
        """
        requests = self._manager.get_pending_requests()

        print()
        print(_color("=== Pending Approval Requests ===", Colors.BOLD))
        print()

        if not requests:
            print(_color("  No pending requests", Colors.DIM))
            return []

        results = []
        for i, req in enumerate(requests, 1):
            self._print_request_summary(i, req)
            results.append(req.to_dict())

        print()
        print(f"Total: {len(requests)} pending request(s)")
        return results

    def _print_request_summary(self, index: int, req: "ApprovalRequest") -> None:
        """Print single request summary."""
        print(f"  [{index}] {_color(str(req.request_id)[:8], Colors.CYAN)}...")
        print(f"      Command: {_color(req.command_type, Colors.BOLD)}")
        print(f"      Class:   {req.change_class.value}")
        print(f"      Status:  {_format_status(req.status.value)}")
        print(f"      Created: {_format_time(req.created_at)}")

        if req.expires_at:
            remaining = req.expires_at - datetime.utcnow()
            if remaining.total_seconds() > 0:
                mins = int(remaining.total_seconds() / 60)
                print(f"      Expires: in {mins}m")
            else:
                print(f"      Expires: {_color('EXPIRED', Colors.RED)}")

        if req.description:
            print(f"      Desc:    {req.description}")

        if req.diff_summary:
            print(f"      Changes: {req.diff_summary}")

        print()

    def show_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Show detailed request information.

        Args:
            request_id: Full or partial request ID

        Returns:
            Request details or None
        """
        req = self._find_request(request_id)
        if not req:
            print(_color(f"Request not found: {request_id}", Colors.RED))
            return None

        print()
        print(_color("=== Approval Request Details ===", Colors.BOLD))
        print()
        print(f"  Request ID:    {req.request_id}")
        print(f"  Command Type:  {_color(req.command_type, Colors.BOLD)}")
        print(f"  Change Class:  {req.change_class.value}")
        print(f"  Status:        {_format_status(req.status.value)}")
        print(f"  Created:       {req.created_at.isoformat()}")

        if req.expires_at:
            print(f"  Expires:       {req.expires_at.isoformat()}")

        print(f"  Description:   {req.description or 'N/A'}")
        print()

        # Config digests
        if req.config_digest_old or req.config_digest_new:
            print(_color("  Config Changes:", Colors.CYAN))
            if req.config_digest_old:
                print(f"    Old: {req.config_digest_old[:32]}...")
            if req.config_digest_new:
                print(f"    New: {req.config_digest_new[:32]}...")
            print()

        # Artifact digest
        if req.artifact_digest:
            print(_color("  Artifact:", Colors.CYAN))
            print(f"    Digest: {req.artifact_digest[:32]}...")
            print()

        # Additional details
        if req.details:
            print(_color("  Details:", Colors.CYAN))
            for key, value in req.details.items():
                print(f"    {key}: {value}")
            print()

        # Show diff if available
        if req.diff_summary:
            print(_color("  Diff Summary:", Colors.CYAN))
            print()
            for line in req.diff_summary.split("\n"):
                if line.startswith("+"):
                    print(f"    {_color(line, Colors.GREEN)}")
                elif line.startswith("-"):
                    print(f"    {_color(line, Colors.RED)}")
                else:
                    print(f"    {line}")
            print()

        return req.to_dict()

    def approve(
        self,
        request_id: str,
        reason: str = "",
        user: str = "local_user",
    ) -> bool:
        """
        Approve a pending request.

        Args:
            request_id: Full or partial request ID
            reason: Reason for approval
            user: Approving user identifier

        Returns:
            True if approved successfully
        """
        req = self._find_request(request_id)
        if not req:
            print(_color(f"Request not found: {request_id}", Colors.RED))
            return False

        success = self._manager.approve(req.request_id, reason=reason, decided_by=user)

        if success:
            print()
            print(_color("✓ Request APPROVED", Colors.GREEN))
            print(f"  ID: {req.request_id}")
            if reason:
                print(f"  Reason: {reason}")
            print()
        else:
            print(_color("✗ Failed to approve request", Colors.RED))
            if req.is_expired:
                print("  Request has expired")
            else:
                print("  Request may have already been decided")

        return success

    def deny(
        self,
        request_id: str,
        reason: str = "",
        user: str = "local_user",
    ) -> bool:
        """
        Deny a pending request.

        Args:
            request_id: Full or partial request ID
            reason: Reason for denial
            user: Denying user identifier

        Returns:
            True if denied successfully
        """
        req = self._find_request(request_id)
        if not req:
            print(_color(f"Request not found: {request_id}", Colors.RED))
            return False

        success = self._manager.deny(req.request_id, reason=reason, decided_by=user)

        if success:
            print()
            print(_color("✗ Request DENIED", Colors.RED))
            print(f"  ID: {req.request_id}")
            if reason:
                print(f"  Reason: {reason}")
            print()
        else:
            print(_color("✗ Failed to deny request", Colors.RED))

        return success

    def show_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Show approval history.

        Args:
            limit: Max entries to show

        Returns:
            List of historical decisions
        """
        history = self._manager.get_history(limit=limit)

        print()
        print(_color("=== Approval History ===", Colors.BOLD))
        print()

        if not history:
            print(_color("  No history available", Colors.DIM))
            return []

        for req in history:
            status_str = _format_status(req.status.value)
            time_str = _format_time(req.decided_at or req.created_at)

            print(f"  {status_str} {req.command_type}")
            print(f"    ID: {str(req.request_id)[:8]}...")
            print(f"    Decided: {time_str}")
            if req.decision_reason:
                print(f"    Reason: {req.decision_reason}")
            print()

        return [r.to_dict() for r in history]

    def _find_request(self, request_id: str) -> Optional["ApprovalRequest"]:
        """Find request by full or partial ID."""
        # Try exact match
        try:
            uuid = UUID(request_id)
            return self._manager.get_request(uuid)
        except (ValueError, TypeError):
            pass

        # Try prefix match
        pending = self._manager.get_pending_requests()
        for req in pending:
            if str(req.request_id).startswith(request_id):
                return req

        return None

    def interactive_approve(self, request_id: str) -> bool:
        """
        Interactive approval with confirmation.

        Args:
            request_id: Request ID to approve

        Returns:
            True if approved
        """
        req = self._find_request(request_id)
        if not req:
            print(_color(f"Request not found: {request_id}", Colors.RED))
            return False

        # Show details
        self.show_request(request_id)

        # Confirm
        print(_color("Do you want to APPROVE this request?", Colors.YELLOW))
        response = input("Type 'yes' to approve, 'no' to deny: ").strip().lower()

        if response == "yes":
            reason = input("Reason (optional): ").strip()
            return self.approve(request_id, reason=reason)
        elif response == "no":
            reason = input("Reason for denial: ").strip()
            return self.deny(request_id, reason=reason)
        else:
            print("Cancelled")
            return False

    def run_interactive(self) -> None:
        """Run interactive approval session."""
        print()
        print(_color("CCEA Local Approval CLI", Colors.BOLD))
        print(_color("Type 'help' for commands, 'quit' to exit", Colors.DIM))
        print()

        while True:
            try:
                cmd = input(_color("approval> ", Colors.CYAN)).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not cmd:
                continue

            parts = cmd.split()
            command = parts[0].lower()
            args = parts[1:]

            if command in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            elif command in ("help", "h", "?"):
                self._print_help()

            elif command in ("list", "ls", "pending"):
                self.list_pending()

            elif command in ("show", "view", "s") and args:
                self.show_request(args[0])

            elif command in ("approve", "a", "yes") and args:
                reason = " ".join(args[1:]) if len(args) > 1 else ""
                self.approve(args[0], reason=reason)

            elif command in ("deny", "d", "no") and args:
                reason = " ".join(args[1:]) if len(args) > 1 else ""
                self.deny(args[0], reason=reason)

            elif command in ("history", "hist"):
                limit = int(args[0]) if args else 20
                self.show_history(limit=limit)

            elif command in ("decide",) and args:
                self.interactive_approve(args[0])

            else:
                print(_color(f"Unknown command: {command}", Colors.RED))
                print("Type 'help' for available commands")

    def _print_help(self) -> None:
        """Print help message."""
        print()
        print(_color("Available Commands:", Colors.BOLD))
        print()
        print("  list, ls        List pending approval requests")
        print("  show <id>       Show detailed request info")
        print("  approve <id>    Approve a request")
        print("  deny <id>       Deny a request")
        print("  decide <id>     Interactive approve/deny")
        print("  history [n]     Show approval history")
        print("  help            Show this help")
        print("  quit            Exit CLI")
        print()
        print(_color("Examples:", Colors.DIM))
        print("  approve abc123 'Approved for production'")
        print("  deny abc123 'Risk limits exceeded'")
        print()


def create_cli_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog="ccea-approve",
        description="CCEA Local Approval CLI",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List pending requests")
    list_parser.add_argument("-v", "--verbose", action="store_true")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show request details")
    show_parser.add_argument("request_id", help="Request ID (full or partial)")

    # Approve command
    approve_parser = subparsers.add_parser("approve", help="Approve request")
    approve_parser.add_argument("request_id", help="Request ID")
    approve_parser.add_argument("-r", "--reason", default="", help="Approval reason")
    approve_parser.add_argument("-u", "--user", default="local_user", help="Approving user")

    # Deny command
    deny_parser = subparsers.add_parser("deny", help="Deny request")
    deny_parser.add_argument("request_id", help="Request ID")
    deny_parser.add_argument("-r", "--reason", default="", help="Denial reason")
    deny_parser.add_argument("-u", "--user", default="local_user", help="Denying user")

    # History command
    history_parser = subparsers.add_parser("history", help="Show approval history")
    history_parser.add_argument("-n", "--limit", type=int, default=20, help="Max entries")

    # Interactive command
    subparsers.add_parser("interactive", help="Run interactive session")

    return parser


def main(manager: "ApprovalManager", args: Optional[List[str]] = None) -> int:
    """
    Main entry point for CLI.

    Args:
        manager: ApprovalManager instance
        args: Command line arguments (default: sys.argv)

    Returns:
        Exit code
    """
    parser = create_cli_parser()
    parsed = parser.parse_args(args)

    cli = ApprovalCLI(manager)

    if parsed.command == "list":
        cli.list_pending()
        return 0

    elif parsed.command == "show":
        result = cli.show_request(parsed.request_id)
        return 0 if result else 1

    elif parsed.command == "approve":
        success = cli.approve(parsed.request_id, reason=parsed.reason, user=parsed.user)
        return 0 if success else 1

    elif parsed.command == "deny":
        success = cli.deny(parsed.request_id, reason=parsed.reason, user=parsed.user)
        return 0 if success else 1

    elif parsed.command == "history":
        cli.show_history(limit=parsed.limit)
        return 0

    elif parsed.command == "interactive" or parsed.command is None:
        cli.run_interactive()
        return 0

    else:
        parser.print_help()
        return 1
