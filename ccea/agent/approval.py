# -*- coding: utf-8 -*-
"""
CCEA Agent Approval Manager.

Handles local approval workflow for TRADING_IMPACTING commands:
- Approval request queue
- Local approval UI/CLI integration
- Evidence recording
- Auto-approval policies

Security:
- Cloud CANNOT bypass local approval
- All approvals recorded with evidence hash
- Auto-approve only through local policy
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
import threading

from ccea.crypto.digest import compute_json_digest


logger = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    """Approval status."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    TIMEOUT = "TIMEOUT"
    AUTO_APPROVED = "AUTO_APPROVED"


@dataclass
class ApprovalRequest:
    """
    Approval request for a command.

    Contains all information needed for user to make decision.
    """
    request_id: str
    command_id: str
    command_type: str
    deployment_id: Optional[str] = None
    artifact_digest: Optional[str] = None
    config_digest: Optional[str] = None
    change_class: Optional[str] = None
    diff_summary: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    approver: Optional[str] = None
    approved_at: Optional[datetime] = None
    evidence_hash: Optional[str] = None
    rejection_reason: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if approval request has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display/storage."""
        return {
            "request_id": self.request_id,
            "command_id": self.command_id,
            "command_type": self.command_type,
            "deployment_id": self.deployment_id,
            "artifact_digest": self.artifact_digest,
            "config_digest": self.config_digest,
            "change_class": self.change_class,
            "diff_summary": self.diff_summary,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status.value,
        }


@dataclass
class ApprovalResult:
    """Result of approval decision."""
    request_id: str
    command_id: str
    approved: bool
    evidence_hash: str
    approver: str
    approved_at: datetime = field(default_factory=datetime.utcnow)
    reason: Optional[str] = None


# ============================================================================
# Auto-Approval Policy
# ============================================================================

@dataclass
class AutoApprovalPolicy:
    """
    Local auto-approval policy.

    IMPORTANT: Cloud CANNOT set or modify this policy.
    Only local configuration can enable auto-approval.
    """
    enabled: bool = False
    # Allowed command types for auto-approval
    allowed_command_types: Set[str] = field(default_factory=set)
    # Allowed deployment IDs (empty = all)
    allowed_deployments: Set[str] = field(default_factory=set)
    # Max artifact size for auto-approval (0 = no limit)
    max_artifact_size_mb: float = 0
    # Require previous manual approval of same artifact
    require_previous_approval: bool = True
    # Previously approved artifacts
    approved_artifacts: Set[str] = field(default_factory=set)


# ============================================================================
# Approval Manager
# ============================================================================

ApprovalCallback = Callable[[ApprovalRequest], None]


class ApprovalManager:
    """
    Local approval workflow manager.

    Handles:
    - Creating approval requests
    - Storing pending approvals
    - Processing approval decisions
    - Recording evidence
    - Auto-approval policies (local only)
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        timeout_seconds: int = 300,  # 5 minutes
        auto_policy: Optional[AutoApprovalPolicy] = None,
    ):
        """
        Initialize approval manager.

        Args:
            data_dir: Directory for approval records
            timeout_seconds: Approval timeout
            auto_policy: Local auto-approval policy
        """
        self.data_dir = Path(data_dir) if data_dir else None
        if self.data_dir:
            self.data_dir.mkdir(parents=True, exist_ok=True)

        self.timeout_seconds = timeout_seconds
        self.auto_policy = auto_policy or AutoApprovalPolicy()

        self._pending: Dict[str, ApprovalRequest] = {}
        self._history: List[ApprovalResult] = []
        self._callbacks: List[ApprovalCallback] = []
        self._lock = threading.Lock()

    def add_callback(self, callback: ApprovalCallback) -> None:
        """Add callback for new approval requests."""
        self._callbacks.append(callback)

    def create_request(
        self,
        command_id: str,
        command_type: str,
        deployment_id: Optional[str] = None,
        artifact_digest: Optional[str] = None,
        config_digest: Optional[str] = None,
        change_class: Optional[str] = None,
        diff_summary: Optional[Dict[str, Any]] = None,
    ) -> ApprovalRequest:
        """
        Create new approval request.

        Args:
            command_id: Command requiring approval
            command_type: Type of command
            deployment_id: Target deployment
            artifact_digest: Artifact being deployed
            config_digest: Config being applied
            change_class: Change classification
            diff_summary: Summary of changes

        Returns:
            ApprovalRequest (may be auto-approved)
        """
        request_id = f"approval_{command_id}"
        expires_at = datetime.utcnow() + timedelta(seconds=self.timeout_seconds)

        request = ApprovalRequest(
            request_id=request_id,
            command_id=command_id,
            command_type=command_type,
            deployment_id=deployment_id,
            artifact_digest=artifact_digest,
            config_digest=config_digest,
            change_class=change_class,
            diff_summary=diff_summary,
            expires_at=expires_at,
        )

        # Check auto-approval policy
        if self._check_auto_approval(request):
            request.status = ApprovalStatus.AUTO_APPROVED
            request.approver = "auto_policy"
            request.approved_at = datetime.utcnow()
            request.evidence_hash = self._compute_evidence_hash(request)

            logger.info(
                "Command auto-approved by local policy",
                extra={"command_id": command_id, "command_type": command_type}
            )
        else:
            with self._lock:
                self._pending[request_id] = request

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(request)
                except Exception as e:
                    logger.exception("Approval callback error")

            logger.info(
                "Approval request created",
                extra={"request_id": request_id, "command_type": command_type}
            )

        return request

    def approve(
        self,
        request_id: str,
        approver: str = "local_user",
        reason: Optional[str] = None,
    ) -> Optional[ApprovalResult]:
        """
        Approve a request.

        Args:
            request_id: Request to approve
            approver: Who approved
            reason: Optional reason

        Returns:
            ApprovalResult or None if not found
        """
        with self._lock:
            request = self._pending.get(request_id)
            if request is None:
                return None

            if request.is_expired():
                request.status = ApprovalStatus.TIMEOUT
                del self._pending[request_id]
                return None

            # Update request
            request.status = ApprovalStatus.APPROVED
            request.approver = approver
            request.approved_at = datetime.utcnow()
            request.evidence_hash = self._compute_evidence_hash(request)

            # Create result
            result = ApprovalResult(
                request_id=request_id,
                command_id=request.command_id,
                approved=True,
                evidence_hash=request.evidence_hash,
                approver=approver,
                reason=reason,
            )

            # Move to history
            self._history.append(result)
            del self._pending[request_id]

            # Update auto-approval policy if artifact
            if request.artifact_digest and self.auto_policy.require_previous_approval:
                self.auto_policy.approved_artifacts.add(request.artifact_digest)

        logger.info(
            "Request approved",
            extra={"request_id": request_id, "approver": approver}
        )

        return result

    def reject(
        self,
        request_id: str,
        approver: str = "local_user",
        reason: Optional[str] = None,
    ) -> Optional[ApprovalResult]:
        """
        Reject a request.

        Args:
            request_id: Request to reject
            approver: Who rejected
            reason: Rejection reason

        Returns:
            ApprovalResult or None if not found
        """
        with self._lock:
            request = self._pending.get(request_id)
            if request is None:
                return None

            # Update request
            request.status = ApprovalStatus.REJECTED
            request.approver = approver
            request.approved_at = datetime.utcnow()
            request.rejection_reason = reason
            request.evidence_hash = self._compute_evidence_hash(request)

            # Create result
            result = ApprovalResult(
                request_id=request_id,
                command_id=request.command_id,
                approved=False,
                evidence_hash=request.evidence_hash,
                approver=approver,
                reason=reason,
            )

            # Move to history
            self._history.append(result)
            del self._pending[request_id]

        logger.info(
            "Request rejected",
            extra={"request_id": request_id, "reason": reason}
        )

        return result

    def get_pending(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        with self._lock:
            # Check for expired
            expired = [
                r for r in self._pending.values()
                if r.is_expired()
            ]
            for r in expired:
                r.status = ApprovalStatus.TIMEOUT
                del self._pending[r.request_id]

            return list(self._pending.values())

    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get request by ID."""
        with self._lock:
            return self._pending.get(request_id)

    def get_history(self, limit: int = 100) -> List[ApprovalResult]:
        """Get approval history."""
        with self._lock:
            return self._history[-limit:]

    def _check_auto_approval(self, request: ApprovalRequest) -> bool:
        """Check if request can be auto-approved."""
        if not self.auto_policy.enabled:
            return False

        # Check command type
        if (self.auto_policy.allowed_command_types and
                request.command_type not in self.auto_policy.allowed_command_types):
            return False

        # Check deployment
        if (self.auto_policy.allowed_deployments and
                request.deployment_id not in self.auto_policy.allowed_deployments):
            return False

        # Check if artifact was previously approved
        if self.auto_policy.require_previous_approval:
            if request.artifact_digest:
                if request.artifact_digest not in self.auto_policy.approved_artifacts:
                    return False

        return True

    def _compute_evidence_hash(self, request: ApprovalRequest) -> str:
        """Compute evidence hash for approval."""
        evidence = {
            "request_id": request.request_id,
            "command_id": request.command_id,
            "command_type": request.command_type,
            "artifact_digest": request.artifact_digest,
            "config_digest": request.config_digest,
            "status": request.status.value,
            "approver": request.approver,
            "approved_at": request.approved_at.isoformat() if request.approved_at else None,
        }
        return compute_json_digest(evidence)


# ============================================================================
# CLI Approval Interface
# ============================================================================

class CLIApprovalInterface:
    """
    Simple CLI interface for approvals.

    Provides interactive approval prompt.
    """

    def __init__(self, manager: ApprovalManager):
        """Initialize CLI interface."""
        self.manager = manager

    def show_pending(self) -> None:
        """Show pending approvals."""
        pending = self.manager.get_pending()

        if not pending:
            print("No pending approvals")
            return

        print(f"\n{'='*60}")
        print(f"PENDING APPROVALS ({len(pending)})")
        print(f"{'='*60}\n")

        for req in pending:
            self._print_request(req)
            print()

    def _print_request(self, req: ApprovalRequest) -> None:
        """Print approval request details."""
        print(f"Request ID: {req.request_id}")
        print(f"Command Type: {req.command_type}")
        print(f"Change Class: {req.change_class or 'N/A'}")

        if req.deployment_id:
            print(f"Deployment: {req.deployment_id}")
        if req.artifact_digest:
            print(f"Artifact: {req.artifact_digest[:32]}...")
        if req.config_digest:
            print(f"Config: {req.config_digest[:32]}...")

        if req.diff_summary:
            print("Changes:")
            for key, value in req.diff_summary.items():
                print(f"  {key}: {value}")

        expires_in = (req.expires_at - datetime.utcnow()).total_seconds() if req.expires_at else 0
        print(f"Expires in: {int(expires_in)} seconds")

    def prompt_approval(self, request_id: str) -> bool:
        """
        Prompt user for approval decision.

        Args:
            request_id: Request to approve/reject

        Returns:
            True if approved, False if rejected
        """
        req = self.manager.get_request(request_id)
        if req is None:
            print(f"Request not found: {request_id}")
            return False

        print("\n" + "="*60)
        print("APPROVAL REQUIRED")
        print("="*60 + "\n")

        self._print_request(req)

        print("\n" + "-"*60)
        response = input("Approve this command? [y/N]: ").strip().lower()

        if response in ("y", "yes"):
            result = self.manager.approve(request_id)
            if result:
                print(f"\nApproved! Evidence hash: {result.evidence_hash[:16]}...")
                return True
        else:
            reason = input("Rejection reason (optional): ").strip()
            self.manager.reject(request_id, reason=reason or None)
            print("\nRejected.")

        return False
