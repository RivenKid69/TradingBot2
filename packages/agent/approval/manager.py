# -*- coding: utf-8 -*-
"""
Approval Manager - Local approval workflow.

AGENT ZONE ONLY.

Handles approval requests for TRADING_IMPACTING changes.
Cloud cannot bypass local approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Final, List, Optional, Set
from uuid import UUID, uuid4

from packages.shared.contracts.config import ChangeClass


class ApprovalStatus(str, Enum):
    """Status of approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class ApprovalRequest:
    """
    Request for local approval.
    """

    request_id: UUID = field(default_factory=uuid4)
    change_class: ChangeClass = ChangeClass.TRADING_IMPACTING
    command_type: str = ""  # e.g., REQUEST_START_RUN
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    # What's changing
    config_digest_old: Optional[str] = None
    config_digest_new: Optional[str] = None
    artifact_digest: Optional[str] = None
    diff_summary: Optional[str] = None

    # Status
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    decided_at: Optional[datetime] = None

    # Evidence
    evidence_hash: Optional[str] = None
    decision_reason: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        """Check if request has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": str(self.request_id),
            "change_class": self.change_class.value,
            "command_type": self.command_type,
            "description": self.description,
            "details": self.details,
            "config_digest_old": self.config_digest_old,
            "config_digest_new": self.config_digest_new,
            "artifact_digest": self.artifact_digest,
            "diff_summary": self.diff_summary,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "evidence_hash": self.evidence_hash,
            "decision_reason": self.decision_reason,
            "is_expired": self.is_expired,
        }


@dataclass
class ApprovalDecision:
    """
    Decision on an approval request.
    """

    approved: bool
    reason: str = ""
    evidence_hash: str = ""
    decided_by: str = "local_user"
    decided_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "approved": self.approved,
            "reason": self.reason,
            "evidence_hash": self.evidence_hash,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at.isoformat(),
        }


# Type for approval callback
ApprovalCallbackFn = Callable[[ApprovalRequest], None]


class ApprovalManager:
    """
    Manages local approval workflow.

    Cloud cannot bypass this - all TRADING_IMPACTING changes
    must be approved locally.

    Usage:
        manager = ApprovalManager()

        # Create request
        request = manager.create_request(
            command_type="REQUEST_START_RUN",
            description="Start momentum strategy",
            details={"strategy_id": "momentum_btc"},
        )

        # Wait for approval
        if manager.wait_for_approval(request.request_id, timeout=60):
            execute_command()
        else:
            reject_command()
    """

    def __init__(
        self,
        auto_approve_operational: bool = True,
        approval_timeout_seconds: int = 300,
        on_approval_requested: Optional[ApprovalCallbackFn] = None,
    ):
        """
        Initialize approval manager.

        Args:
            auto_approve_operational: Auto-approve OPERATIONAL changes
            approval_timeout_seconds: Timeout for approval requests
            on_approval_requested: Callback when approval requested
        """
        self._auto_approve_operational = auto_approve_operational
        self._timeout_seconds = approval_timeout_seconds
        self._on_approval_requested = on_approval_requested

        # Request storage
        self._requests: Dict[UUID, ApprovalRequest] = {}
        self._history: List[ApprovalRequest] = []

        # Auto-approve whitelist (LOCAL only)
        self._auto_approve_commands: Set[str] = set()

    def create_request(
        self,
        command_type: str,
        description: str,
        change_class: ChangeClass = ChangeClass.TRADING_IMPACTING,
        details: Optional[Dict[str, Any]] = None,
        config_digest_old: Optional[str] = None,
        config_digest_new: Optional[str] = None,
        artifact_digest: Optional[str] = None,
    ) -> ApprovalRequest:
        """
        Create approval request.

        Args:
            command_type: Type of command
            description: Human-readable description
            change_class: Classification of change
            details: Additional details
            config_digest_old: Old config digest
            config_digest_new: New config digest
            artifact_digest: Artifact digest if applicable

        Returns:
            Created approval request
        """
        # Check if auto-approve
        if self._should_auto_approve(command_type, change_class):
            from .evidence import compute_evidence_hash

            request = ApprovalRequest(
                command_type=command_type,
                description=description,
                change_class=change_class,
                details=details or {},
                config_digest_old=config_digest_old,
                config_digest_new=config_digest_new,
                artifact_digest=artifact_digest,
                status=ApprovalStatus.APPROVED,
                decided_at=datetime.utcnow(),
                evidence_hash=compute_evidence_hash({
                    "command_type": command_type,
                    "auto_approved": True,
                }),
                decision_reason="Auto-approved by local policy",
            )
            self._history.append(request)
            return request

        # Create pending request
        from datetime import timedelta

        request = ApprovalRequest(
            command_type=command_type,
            description=description,
            change_class=change_class,
            details=details or {},
            config_digest_old=config_digest_old,
            config_digest_new=config_digest_new,
            artifact_digest=artifact_digest,
            expires_at=datetime.utcnow() + timedelta(seconds=self._timeout_seconds),
        )

        # Generate diff summary
        if config_digest_old and config_digest_new:
            request.diff_summary = f"Config change: {config_digest_old[:8]}... -> {config_digest_new[:8]}..."

        self._requests[request.request_id] = request

        # Notify callback
        if self._on_approval_requested:
            self._on_approval_requested(request)

        return request

    def approve(
        self,
        request_id: UUID,
        reason: str = "",
        decided_by: str = "local_user",
    ) -> bool:
        """
        Approve a request.

        Args:
            request_id: Request ID
            reason: Reason for approval
            decided_by: Who approved

        Returns:
            True if approved, False if not found or expired
        """
        request = self._requests.get(request_id)
        if not request:
            return False

        if request.is_expired:
            request.status = ApprovalStatus.EXPIRED
            return False

        if request.status != ApprovalStatus.PENDING:
            return False

        from .evidence import compute_evidence_hash

        request.status = ApprovalStatus.APPROVED
        request.decided_at = datetime.utcnow()
        request.decision_reason = reason
        request.evidence_hash = compute_evidence_hash({
            "request_id": str(request_id),
            "command_type": request.command_type,
            "config_digest_new": request.config_digest_new,
            "artifact_digest": request.artifact_digest,
            "decided_by": decided_by,
            "decided_at": request.decided_at.isoformat(),
        })

        # Move to history
        self._history.append(request)
        del self._requests[request_id]

        return True

    def deny(
        self,
        request_id: UUID,
        reason: str = "",
        decided_by: str = "local_user",
    ) -> bool:
        """
        Deny a request.

        Args:
            request_id: Request ID
            reason: Reason for denial
            decided_by: Who denied

        Returns:
            True if denied, False if not found
        """
        request = self._requests.get(request_id)
        if not request:
            return False

        if request.status != ApprovalStatus.PENDING:
            return False

        request.status = ApprovalStatus.DENIED
        request.decided_at = datetime.utcnow()
        request.decision_reason = reason

        # Move to history
        self._history.append(request)
        del self._requests[request_id]

        return True

    def get_pending_requests(self) -> List[ApprovalRequest]:
        """Get all pending requests."""
        # Check for expired
        for request in list(self._requests.values()):
            if request.is_expired:
                request.status = ApprovalStatus.EXPIRED
                self._history.append(request)
                del self._requests[request.request_id]

        return list(self._requests.values())

    def get_request(self, request_id: UUID) -> Optional[ApprovalRequest]:
        """Get request by ID."""
        return self._requests.get(request_id)

    def get_history(self, limit: int = 100) -> List[ApprovalRequest]:
        """Get approval history."""
        return self._history[-limit:]

    def add_auto_approve_command(self, command_type: str) -> None:
        """Add command type to auto-approve whitelist."""
        self._auto_approve_commands.add(command_type)

    def remove_auto_approve_command(self, command_type: str) -> None:
        """Remove command type from auto-approve whitelist."""
        self._auto_approve_commands.discard(command_type)

    def _should_auto_approve(
        self,
        command_type: str,
        change_class: ChangeClass,
    ) -> bool:
        """Check if request should be auto-approved."""
        # TRADING_IMPACTING never auto-approved by default
        if change_class == ChangeClass.TRADING_IMPACTING:
            # Check whitelist
            return command_type in self._auto_approve_commands

        # OPERATIONAL auto-approved if setting enabled
        if change_class == ChangeClass.OPERATIONAL:
            return self._auto_approve_operational

        # INFORMATIONAL always auto-approved
        return True
