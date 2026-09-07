# -*- coding: utf-8 -*-
"""
GDPR Phase 6: Change Management Service.

Implements change management for TRADING_IMPACTING and other classified changes:
- Change classification
- Local approval workflow
- Diff/evidence hashes
- Change journal
- Full audit trail

Design Doc Reference:
    - Phase 6: "Change management: TRADING_IMPACTING always requires local approval"
    - Design Doc 6.2, 12.2: "Structured approval evidence with immutable blob references"
    - Design Doc 9.4, 11.1: "TRADING_IMPACTING changes require local approval"

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Set, TYPE_CHECKING
from uuid import uuid4

from .access_audit import (
    AccessAuditService,
    AuditAction,
    AuditPrincipal,
    AuditResult,
)
from .rbac_service import ResourceType, SensitivityLevel


logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Approval expiry (hours)
APPROVAL_EXPIRY_HOURS: Final[int] = 24

# Minimum reason length for approvals
MIN_APPROVAL_REASON_LENGTH: Final[int] = 10

# Maximum pending changes per workspace
MAX_PENDING_CHANGES_PER_WORKSPACE: Final[int] = 100


# ============================================================================
# Enums
# ============================================================================


class ChangeClass(str, Enum):
    """Change classification per Design Doc."""

    OPERATIONAL = "operational"
    TRADING_IMPACTING = "trading_impacting"
    SECURITY_SENSITIVE = "security_sensitive"
    DATA_SENSITIVE = "data_sensitive"


class ChangeType(str, Enum):
    """Types of changes."""

    DEPLOY = "deploy"
    UPGRADE = "upgrade"
    CONFIG_UPDATE = "config_update"
    ROLLBACK = "rollback"
    PARAMETER_CHANGE = "parameter_change"
    RISK_LIMIT_CHANGE = "risk_limit_change"
    STRATEGY_UPDATE = "strategy_update"
    AGENT_UPDATE = "agent_update"
    COMMAND = "command"
    OTHER = "other"


class ChangeStatus(str, Enum):
    """Status of a change request."""

    PENDING = "pending"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class ApprovalType(str, Enum):
    """Type of approval."""

    LOCAL = "local"  # User approval on agent
    CLOUD = "cloud"  # Cloud admin approval
    AUTOMATIC = "automatic"  # Auto-approved (non-impacting)
    BREAK_GLASS = "break_glass"


# ============================================================================
# Classification Rules
# ============================================================================

# Change types that are always TRADING_IMPACTING
TRADING_IMPACTING_CHANGE_TYPES: Set[ChangeType] = {
    ChangeType.DEPLOY,
    ChangeType.UPGRADE,
    ChangeType.ROLLBACK,
    ChangeType.PARAMETER_CHANGE,
    ChangeType.RISK_LIMIT_CHANGE,
    ChangeType.STRATEGY_UPDATE,
}

# Change types that are SECURITY_SENSITIVE
SECURITY_SENSITIVE_CHANGE_TYPES: Set[ChangeType] = {
    ChangeType.AGENT_UPDATE,
}

# Classification requirements
CLASSIFICATION_REQUIREMENTS: Dict[ChangeClass, Dict[str, Any]] = {
    ChangeClass.OPERATIONAL: {
        "requires_approval": False,
        "requires_local_approval": False,
        "requires_diff": False,
        "requires_evidence": False,
    },
    ChangeClass.TRADING_IMPACTING: {
        "requires_approval": True,
        "requires_local_approval": True,
        "requires_diff": True,
        "requires_evidence": True,
    },
    ChangeClass.SECURITY_SENSITIVE: {
        "requires_approval": True,
        "requires_local_approval": False,
        "requires_diff": True,
        "requires_evidence": True,
        "requires_security_review": True,
    },
    ChangeClass.DATA_SENSITIVE: {
        "requires_approval": True,
        "requires_local_approval": False,
        "requires_diff": True,
        "requires_evidence": True,
        "requires_dpo_review": True,
    },
}


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class ChangeEvidence:
    """
    Immutable evidence for change approval.

    Per Design Doc 6.2, 12.2: Evidence includes blob digests for audit verification.
    """

    # Config blob at approval time
    config_blob_digest: Optional[str] = None

    # Artifact manifest digest
    manifest_digest: Optional[str] = None

    # State digests
    previous_state_digest: Optional[str] = None
    new_state_digest: Optional[str] = None

    # Diff information
    diff_summary: Dict[str, Any] = field(default_factory=dict)
    diff_hash: Optional[str] = None

    # Display hash (what was shown to approver)
    display_hash: Optional[str] = None

    # Additional evidence
    additional: Dict[str, str] = field(default_factory=dict)

    def compute_evidence_hash(self) -> str:
        """Compute overall evidence hash."""
        content = json.dumps(
            {
                "config_blob_digest": self.config_blob_digest,
                "manifest_digest": self.manifest_digest,
                "previous_state_digest": self.previous_state_digest,
                "new_state_digest": self.new_state_digest,
                "diff_hash": self.diff_hash,
            },
            sort_keys=True,
        )
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_blob_digest": self.config_blob_digest,
            "manifest_digest": self.manifest_digest,
            "previous_state_digest": self.previous_state_digest,
            "new_state_digest": self.new_state_digest,
            "diff_summary": self.diff_summary,
            "diff_hash": self.diff_hash,
            "display_hash": self.display_hash,
            "additional": self.additional,
            "evidence_hash": self.compute_evidence_hash(),
        }


@dataclass
class ApprovalRecord:
    """
    Record of approval decision.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    change_id: str = ""

    # Approval decision
    approved: bool = False
    approval_type: ApprovalType = ApprovalType.LOCAL
    approved_by: str = ""
    approved_by_email: Optional[str] = None
    approved_at: Optional[datetime] = None

    # User acknowledgment
    user_acknowledged: bool = False
    acknowledgment_timestamp: Optional[datetime] = None

    # Reason
    reason: str = ""
    rejection_reason: Optional[str] = None

    # Evidence at approval time
    evidence: ChangeEvidence = field(default_factory=ChangeEvidence)

    # Attestation (what was shown and acknowledged)
    attestation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "change_id": self.change_id,
            "approved": self.approved,
            "approval_type": self.approval_type.value,
            "approved_by": self.approved_by,
            "approved_by_email": self.approved_by_email,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "user_acknowledged": self.user_acknowledged,
            "acknowledgment_timestamp": (
                self.acknowledgment_timestamp.isoformat() if self.acknowledgment_timestamp else None
            ),
            "reason": self.reason,
            "rejection_reason": self.rejection_reason,
            "evidence": self.evidence.to_dict(),
            "attestation": self.attestation,
        }


@dataclass
class ChangeRequest:
    """
    Change request record.

    Tracks the full lifecycle of a change.
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    organization_id: str = ""

    # Change details
    change_type: ChangeType = ChangeType.OTHER
    change_class: ChangeClass = ChangeClass.OPERATIONAL
    description: str = ""
    ticket_id: Optional[str] = None

    # Requester
    requester_id: str = ""
    requester_email: str = ""

    # Target
    target_type: str = ""  # deployment, agent, strategy, etc.
    target_id: Optional[str] = None
    target_name: Optional[str] = None

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Status
    status: ChangeStatus = ChangeStatus.PENDING

    # Evidence
    evidence: ChangeEvidence = field(default_factory=ChangeEvidence)

    # Approval
    approval_record: Optional[ApprovalRecord] = None
    requires_local_approval: bool = False

    # Command reference (if applicable)
    command_id: Optional[str] = None

    # Execution result
    execution_result: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    # Rollback info
    rollback_available: bool = False
    rollback_to_digest: Optional[str] = None
    rolled_back_at: Optional[datetime] = None
    rolled_back_by: Optional[str] = None

    def __post_init__(self):
        """Set expiry and requirements based on classification."""
        # Set expiry if not set
        if not self.expires_at:
            self.expires_at = self.created_at + timedelta(hours=APPROVAL_EXPIRY_HOURS)

        # Set requirements based on classification
        requirements = CLASSIFICATION_REQUIREMENTS.get(self.change_class, {})
        self.requires_local_approval = requirements.get("requires_local_approval", False)

    @property
    def is_pending(self) -> bool:
        return self.status in (ChangeStatus.PENDING, ChangeStatus.AWAITING_APPROVAL)

    @property
    def is_expired(self) -> bool:
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return True
        return self.status == ChangeStatus.EXPIRED

    @property
    def is_approved(self) -> bool:
        return self.status == ChangeStatus.APPROVED

    @property
    def requires_approval(self) -> bool:
        requirements = CLASSIFICATION_REQUIREMENTS.get(self.change_class, {})
        return requirements.get("requires_approval", False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "organization_id": self.organization_id,
            "change": {
                "type": self.change_type.value,
                "class": self.change_class.value,
                "description": self.description,
                "ticket_id": self.ticket_id,
            },
            "requester": {
                "id": self.requester_id,
                "email": self.requester_email,
            },
            "target": {
                "type": self.target_type,
                "id": self.target_id,
                "name": self.target_name,
            },
            "timing": {
                "created_at": self.created_at.isoformat(),
                "expires_at": self.expires_at.isoformat() if self.expires_at else None,
                "executed_at": self.executed_at.isoformat() if self.executed_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            },
            "status": self.status.value,
            "evidence": self.evidence.to_dict(),
            "approval": self.approval_record.to_dict() if self.approval_record else None,
            "requires_local_approval": self.requires_local_approval,
            "requires_approval": self.requires_approval,
            "command_id": self.command_id,
            "execution_result": self.execution_result,
            "error_message": self.error_message,
            "rollback": {
                "available": self.rollback_available,
                "to_digest": self.rollback_to_digest,
                "rolled_back_at": self.rolled_back_at.isoformat() if self.rolled_back_at else None,
                "rolled_back_by": self.rolled_back_by,
            },
            "is_pending": self.is_pending,
            "is_expired": self.is_expired,
            "is_approved": self.is_approved,
        }


@dataclass
class ChangeJournalEntry:
    """
    Entry in the change journal.

    Provides exportable record of changes for evidence pack.
    """

    change_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    workspace_id: str = ""
    organization_id: str = ""

    # Change summary
    change_type: str = ""
    change_class: str = ""
    description: str = ""

    # Actors
    requester_id: str = ""
    approver_id: Optional[str] = None

    # Evidence
    artifact_digest: Optional[str] = None
    config_digest: Optional[str] = None
    approval_record_id: Optional[str] = None

    # Status
    status: str = ""

    # Integrity
    integrity_hash: str = ""

    def __post_init__(self):
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps(
            {
                "change_id": self.change_id,
                "timestamp": self.timestamp.isoformat(),
                "workspace_id": self.workspace_id,
                "change_type": self.change_type,
                "change_class": self.change_class,
                "requester_id": self.requester_id,
                "approver_id": self.approver_id,
                "artifact_digest": self.artifact_digest,
                "config_digest": self.config_digest,
                "status": self.status,
            },
            sort_keys=True,
        )
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "change_id": self.change_id,
            "timestamp": self.timestamp.isoformat(),
            "workspace_id": self.workspace_id,
            "organization_id": self.organization_id,
            "change_type": self.change_type,
            "change_class": self.change_class,
            "description": self.description,
            "requester_id": self.requester_id,
            "approver_id": self.approver_id,
            "artifact_digest": self.artifact_digest,
            "config_digest": self.config_digest,
            "approval_record_id": self.approval_record_id,
            "status": self.status,
            "integrity_hash": self.integrity_hash,
        }


@dataclass
class ChangeStats:
    """Change management statistics."""

    total_changes: int = 0
    by_status: Dict[str, int] = field(default_factory=dict)
    by_class: Dict[str, int] = field(default_factory=dict)
    by_type: Dict[str, int] = field(default_factory=dict)
    pending_count: int = 0
    approved_count: int = 0
    rejected_count: int = 0
    rolled_back_count: int = 0
    trading_impacting_count: int = 0
    avg_approval_time_minutes: float = 0.0
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_changes": self.total_changes,
            "by_status": self.by_status,
            "by_class": self.by_class,
            "by_type": self.by_type,
            "pending_count": self.pending_count,
            "approved_count": self.approved_count,
            "rejected_count": self.rejected_count,
            "rolled_back_count": self.rolled_back_count,
            "trading_impacting_count": self.trading_impacting_count,
            "avg_approval_time_minutes": self.avg_approval_time_minutes,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
        }


# ============================================================================
# Change Management Service
# ============================================================================


class ChangeManagementService:
    """
    Change Management Service.

    Handles change classification, approval workflow, and change journal:
    - Classify changes by impact
    - Enforce approval requirements
    - Track evidence and diffs
    - Maintain change journal for audit

    Usage:
        service = ChangeManagementService(audit_service=audit_service)

        # Classify change
        change_class = service.classify_change(
            change_type=ChangeType.DEPLOY,
            target_type="deployment",
        )

        # Create change request
        change = service.create_change(
            workspace_id="ws-123",
            change_type=ChangeType.DEPLOY,
            description="Deploy strategy v1.1.0",
            requester_id="user-456",
            target_type="deployment",
            target_id="dep-789",
            evidence=ChangeEvidence(
                config_blob_digest="sha256:...",
                manifest_digest="sha256:...",
            ),
        )

        # Approve (local approval)
        approval = service.approve_change(
            change_id=change.id,
            approver_id="user-456",
            approval_type=ApprovalType.LOCAL,
            user_acknowledged=True,
            reason="Approved deployment of strategy v1.1.0",
        )

        # Export journal
        journal = service.export_journal(workspace_id="ws-123")
    """

    def __init__(
        self,
        audit_service: Optional[AccessAuditService] = None,
        on_change_created: Optional[Callable[[ChangeRequest], None]] = None,
        on_change_approved: Optional[Callable[[ChangeRequest], None]] = None,
        on_change_executed: Optional[Callable[[ChangeRequest], None]] = None,
    ):
        """
        Initialize Change Management Service.

        Args:
            audit_service: Access audit service
            on_change_created: Callback when change is created
            on_change_approved: Callback when change is approved
            on_change_executed: Callback when change is executed
        """
        self._audit_service = audit_service
        self._on_change_created = on_change_created
        self._on_change_approved = on_change_approved
        self._on_change_executed = on_change_executed

        # Storage
        self._changes: Dict[str, ChangeRequest] = {}
        self._approvals: Dict[str, ApprovalRecord] = {}
        self._journal: List[ChangeJournalEntry] = []
        self._changes_by_workspace: Dict[str, List[str]] = {}

        self._lock = threading.Lock()

    # ========================================================================
    # Classification
    # ========================================================================

    def classify_change(
        self,
        change_type: ChangeType,
        target_type: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ChangeClass:
        """
        Classify a change by impact.

        Args:
            change_type: Type of change
            target_type: Target resource type
            parameters: Change parameters (for detailed classification)

        Returns:
            ChangeClass
        """
        # Check if change type is inherently TRADING_IMPACTING
        if change_type in TRADING_IMPACTING_CHANGE_TYPES:
            return ChangeClass.TRADING_IMPACTING

        # Check if change type is SECURITY_SENSITIVE
        if change_type in SECURITY_SENSITIVE_CHANGE_TYPES:
            return ChangeClass.SECURITY_SENSITIVE

        # Check target type
        if target_type in ("deployment", "run", "strategy", "agent"):
            return ChangeClass.TRADING_IMPACTING

        # Check parameters for sensitive changes
        if parameters:
            # Risk limit changes are always trading impacting
            if "risk_limit" in parameters or "max_loss" in parameters:
                return ChangeClass.TRADING_IMPACTING

            # PII-related changes are data sensitive
            if any(k in parameters for k in ("pii", "personal_data", "user_data")):
                return ChangeClass.DATA_SENSITIVE

        # Default to OPERATIONAL
        return ChangeClass.OPERATIONAL

    def get_classification_requirements(
        self,
        change_class: ChangeClass,
    ) -> Dict[str, Any]:
        """Get requirements for a change classification."""
        return CLASSIFICATION_REQUIREMENTS.get(change_class, {})

    # ========================================================================
    # Change Request Management
    # ========================================================================

    def create_change(
        self,
        workspace_id: str,
        change_type: ChangeType,
        description: str,
        requester_id: str,
        requester_email: str,
        target_type: str,
        organization_id: str = "",
        target_id: Optional[str] = None,
        target_name: Optional[str] = None,
        ticket_id: Optional[str] = None,
        evidence: Optional[ChangeEvidence] = None,
        command_id: Optional[str] = None,
        change_class: Optional[ChangeClass] = None,
    ) -> ChangeRequest:
        """
        Create a change request.

        Args:
            workspace_id: Workspace context
            change_type: Type of change
            description: Change description
            requester_id: Who is requesting
            requester_email: Requester email
            target_type: Target resource type
            organization_id: Organization ID
            target_id: Target resource ID
            target_name: Target name
            ticket_id: Related ticket ID
            evidence: Change evidence
            command_id: Related command ID
            change_class: Override classification

        Returns:
            ChangeRequest
        """
        # Classify if not provided
        if change_class is None:
            change_class = self.classify_change(change_type, target_type)

        # Check pending limit
        with self._lock:
            ws_changes = self._changes_by_workspace.get(workspace_id, [])
            pending_count = sum(
                1 for cid in ws_changes if self._changes.get(cid) and self._changes[cid].is_pending
            )
            if pending_count >= MAX_PENDING_CHANGES_PER_WORKSPACE:
                raise ValueError(
                    f"Maximum {MAX_PENDING_CHANGES_PER_WORKSPACE} pending changes per workspace"
                )

        # Create change request
        change = ChangeRequest(
            workspace_id=workspace_id,
            organization_id=organization_id,
            change_type=change_type,
            change_class=change_class,
            description=description,
            requester_id=requester_id,
            requester_email=requester_email,
            target_type=target_type,
            target_id=target_id,
            target_name=target_name,
            ticket_id=ticket_id,
            evidence=evidence or ChangeEvidence(),
            command_id=command_id,
        )

        # Set status based on requirements
        if change.requires_approval:
            change.status = ChangeStatus.AWAITING_APPROVAL
        else:
            change.status = ChangeStatus.PENDING

        with self._lock:
            self._changes[change.id] = change

            # Index by workspace
            if workspace_id not in self._changes_by_workspace:
                self._changes_by_workspace[workspace_id] = []
            self._changes_by_workspace[workspace_id].append(change.id)

        # Audit
        self._audit_change_created(change)

        # Callback
        if self._on_change_created:
            try:
                self._on_change_created(change)
            except Exception as e:
                logger.error(f"on_change_created callback error: {e}")

        logger.info(
            f"Change created: {change.id} type={change_type.value} "
            f"class={change_class.value} workspace={workspace_id}"
        )

        return change

    def approve_change(
        self,
        change_id: str,
        approver_id: str,
        approval_type: ApprovalType,
        approver_email: Optional[str] = None,
        user_acknowledged: bool = False,
        reason: str = "",
        evidence_override: Optional[ChangeEvidence] = None,
    ) -> Optional[ApprovalRecord]:
        """
        Approve a change request.

        Args:
            change_id: Change to approve
            approver_id: Approver ID
            approval_type: Type of approval
            approver_email: Approver email
            user_acknowledged: Whether user acknowledged the change
            reason: Approval reason
            evidence_override: Override evidence at approval time

        Returns:
            ApprovalRecord if successful
        """
        with self._lock:
            change = self._changes.get(change_id)
            if not change:
                logger.error(f"Change not found: {change_id}")
                return None

            if change.status not in (ChangeStatus.PENDING, ChangeStatus.AWAITING_APPROVAL):
                logger.error(f"Change cannot be approved: {change_id} status={change.status}")
                return None

            if change.is_expired:
                change.status = ChangeStatus.EXPIRED
                logger.error(f"Change expired: {change_id}")
                return None

            # Check local approval requirement
            if change.requires_local_approval:
                if (
                    approval_type != ApprovalType.LOCAL
                    and approval_type != ApprovalType.BREAK_GLASS
                ):
                    logger.error(f"Change requires local approval: {change_id}")
                    return None

            # Validate reason
            if change.change_class == ChangeClass.TRADING_IMPACTING:
                if len(reason.strip()) < MIN_APPROVAL_REASON_LENGTH:
                    logger.error(
                        f"TRADING_IMPACTING change requires reason >= {MIN_APPROVAL_REASON_LENGTH} chars"
                    )
                    return None

                # User acknowledgment required for trading impacting
                if not user_acknowledged and approval_type == ApprovalType.LOCAL:
                    logger.error(f"User acknowledgment required for TRADING_IMPACTING change")
                    return None

            # Create approval record
            now = datetime.now(timezone.utc)
            evidence = evidence_override or change.evidence

            approval = ApprovalRecord(
                change_id=change_id,
                approved=True,
                approval_type=approval_type,
                approved_by=approver_id,
                approved_by_email=approver_email,
                approved_at=now,
                user_acknowledged=user_acknowledged,
                acknowledgment_timestamp=now if user_acknowledged else None,
                reason=reason,
                evidence=evidence,
                attestation={
                    "displayed_diff": evidence.diff_summary,
                    "display_hash": evidence.display_hash,
                    "user_confirmed": user_acknowledged,
                    "timestamp": now.isoformat(),
                },
            )

            # Update change
            change.status = ChangeStatus.APPROVED
            change.approval_record = approval
            change.evidence = evidence

            self._approvals[approval.id] = approval

            # Add to journal
            self._add_journal_entry(change, "approved")

        # Audit
        self._audit_change_approved(change, approval)

        # Callback
        if self._on_change_approved:
            try:
                self._on_change_approved(change)
            except Exception as e:
                logger.error(f"on_change_approved callback error: {e}")

        logger.info(f"Change approved: {change_id} by {approver_id} type={approval_type.value}")

        return approval

    def reject_change(
        self,
        change_id: str,
        rejector_id: str,
        rejector_email: Optional[str] = None,
        reason: str = "",
    ) -> bool:
        """
        Reject a change request.

        Args:
            change_id: Change to reject
            rejector_id: Rejector ID
            rejector_email: Rejector email
            reason: Rejection reason

        Returns:
            True if rejected
        """
        with self._lock:
            change = self._changes.get(change_id)
            if not change:
                return False

            if change.status not in (ChangeStatus.PENDING, ChangeStatus.AWAITING_APPROVAL):
                return False

            change.status = ChangeStatus.REJECTED

            # Create rejection record
            approval = ApprovalRecord(
                change_id=change_id,
                approved=False,
                approval_type=ApprovalType.LOCAL,
                approved_by=rejector_id,
                approved_by_email=rejector_email,
                approved_at=datetime.now(timezone.utc),
                rejection_reason=reason,
            )
            change.approval_record = approval
            self._approvals[approval.id] = approval

            # Add to journal
            self._add_journal_entry(change, "rejected")

        # Audit
        self._audit_change_rejected(change, rejector_id, reason)

        logger.info(f"Change rejected: {change_id} by {rejector_id}")

        return True

    def execute_change(
        self,
        change_id: str,
        executed_by: str,
        execution_result: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Mark a change as executing/executed.

        Args:
            change_id: Change to execute
            executed_by: Executor ID
            execution_result: Execution result

        Returns:
            True if executed
        """
        with self._lock:
            change = self._changes.get(change_id)
            if not change:
                return False

            if change.status != ChangeStatus.APPROVED:
                return False

            change.status = ChangeStatus.EXECUTING
            change.executed_at = datetime.now(timezone.utc)
            change.execution_result = execution_result or {}

        # Audit
        self._audit_change_executed(change, executed_by)

        # Callback
        if self._on_change_executed:
            try:
                self._on_change_executed(change)
            except Exception as e:
                logger.error(f"on_change_executed callback error: {e}")

        logger.info(f"Change executing: {change_id}")

        return True

    def complete_change(
        self,
        change_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
        rollback_available: bool = False,
        rollback_to_digest: Optional[str] = None,
    ) -> bool:
        """
        Complete a change.

        Args:
            change_id: Change to complete
            success: Whether change succeeded
            error_message: Error message if failed
            rollback_available: Whether rollback is available
            rollback_to_digest: Digest to rollback to

        Returns:
            True if completed
        """
        with self._lock:
            change = self._changes.get(change_id)
            if not change:
                return False

            if change.status != ChangeStatus.EXECUTING:
                return False

            change.completed_at = datetime.now(timezone.utc)
            change.status = ChangeStatus.COMPLETED if success else ChangeStatus.FAILED
            change.error_message = error_message
            change.rollback_available = rollback_available
            change.rollback_to_digest = rollback_to_digest

            # Add to journal
            status = "completed" if success else "failed"
            self._add_journal_entry(change, status)

        logger.info(f"Change completed: {change_id} success={success}")

        return True

    def rollback_change(
        self,
        change_id: str,
        rolledback_by: str,
        reason: str = "",
    ) -> bool:
        """
        Rollback a completed change.

        Args:
            change_id: Change to rollback
            rolledback_by: User performing rollback
            reason: Rollback reason

        Returns:
            True if rollback initiated
        """
        with self._lock:
            change = self._changes.get(change_id)
            if not change:
                return False

            if change.status != ChangeStatus.COMPLETED:
                return False

            if not change.rollback_available:
                return False

            change.status = ChangeStatus.ROLLED_BACK
            change.rolled_back_at = datetime.now(timezone.utc)
            change.rolled_back_by = rolledback_by

            # Add to journal
            self._add_journal_entry(change, "rolled_back")

        # Audit
        self._audit_change_rollback(change, rolledback_by, reason)

        logger.info(f"Change rolled back: {change_id} by {rolledback_by}")

        return True

    def cancel_change(
        self,
        change_id: str,
        cancelled_by: str,
    ) -> bool:
        """Cancel a pending change."""
        with self._lock:
            change = self._changes.get(change_id)
            if not change:
                return False

            if not change.is_pending:
                return False

            change.status = ChangeStatus.CANCELLED

        logger.info(f"Change cancelled: {change_id}")

        return True

    # ========================================================================
    # Queries
    # ========================================================================

    def get_change(self, change_id: str) -> Optional[ChangeRequest]:
        """Get change by ID."""
        with self._lock:
            return self._changes.get(change_id)

    def get_pending_changes(
        self,
        workspace_id: Optional[str] = None,
    ) -> List[ChangeRequest]:
        """Get pending changes."""
        with self._lock:
            if workspace_id:
                change_ids = self._changes_by_workspace.get(workspace_id, [])
                changes = [self._changes[cid] for cid in change_ids if cid in self._changes]
            else:
                changes = list(self._changes.values())

            return [c for c in changes if c.is_pending]

    def get_changes_by_workspace(
        self,
        workspace_id: str,
        status: Optional[ChangeStatus] = None,
        change_class: Optional[ChangeClass] = None,
        limit: int = 100,
    ) -> List[ChangeRequest]:
        """Get changes for a workspace."""
        with self._lock:
            change_ids = self._changes_by_workspace.get(workspace_id, [])
            changes = [self._changes[cid] for cid in change_ids if cid in self._changes]

            if status:
                changes = [c for c in changes if c.status == status]

            if change_class:
                changes = [c for c in changes if c.change_class == change_class]

            return sorted(changes, key=lambda c: c.created_at, reverse=True)[:limit]

    def get_approval_record(self, approval_id: str) -> Optional[ApprovalRecord]:
        """Get approval record by ID."""
        with self._lock:
            return self._approvals.get(approval_id)

    # ========================================================================
    # Journal
    # ========================================================================

    def _add_journal_entry(
        self,
        change: ChangeRequest,
        status: str,
    ) -> None:
        """Add entry to change journal."""
        entry = ChangeJournalEntry(
            change_id=change.id,
            workspace_id=change.workspace_id,
            organization_id=change.organization_id,
            change_type=change.change_type.value,
            change_class=change.change_class.value,
            description=change.description,
            requester_id=change.requester_id,
            approver_id=change.approval_record.approved_by if change.approval_record else None,
            artifact_digest=change.evidence.manifest_digest,
            config_digest=change.evidence.config_blob_digest,
            approval_record_id=change.approval_record.id if change.approval_record else None,
            status=status,
        )
        self._journal.append(entry)

    def get_journal(
        self,
        workspace_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[ChangeJournalEntry]:
        """Get change journal entries."""
        with self._lock:
            entries = list(self._journal)

            if workspace_id:
                entries = [e for e in entries if e.workspace_id == workspace_id]

            if start_time:
                entries = [e for e in entries if e.timestamp >= start_time]

            if end_time:
                entries = [e for e in entries if e.timestamp <= end_time]

            return sorted(entries, key=lambda e: e.timestamp, reverse=True)[:limit]

    def export_journal(
        self,
        workspace_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Export change journal for evidence pack.

        Args:
            workspace_id: Workspace to export
            start_time: Start of period
            end_time: End of period

        Returns:
            Exportable journal dict
        """
        entries = self.get_journal(workspace_id, start_time, end_time)

        export = {
            "export_metadata": {
                "export_id": str(uuid4()),
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "workspace_id": workspace_id,
                "export_type": "change_journal",
                "period_start": start_time.isoformat() if start_time else None,
                "period_end": end_time.isoformat() if end_time else None,
                "entry_count": len(entries),
            },
            "changes": [e.to_dict() for e in entries],
        }

        # Compute integrity
        content = json.dumps(export["changes"], sort_keys=True, default=str)
        export["integrity"] = {
            "checksum": f"sha256:{hashlib.sha256(content.encode()).hexdigest()}",
        }

        return export

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_stats(
        self,
        workspace_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> ChangeStats:
        """Get change management statistics."""
        with self._lock:
            if workspace_id:
                change_ids = self._changes_by_workspace.get(workspace_id, [])
                changes = [self._changes[cid] for cid in change_ids if cid in self._changes]
            else:
                changes = list(self._changes.values())

            if start_time:
                changes = [c for c in changes if c.created_at >= start_time]

            if end_time:
                changes = [c for c in changes if c.created_at <= end_time]

            stats = ChangeStats(
                total_changes=len(changes),
                period_start=start_time,
                period_end=end_time or datetime.now(timezone.utc),
            )

            approval_times = []

            for c in changes:
                # By status
                status_key = c.status.value
                stats.by_status[status_key] = stats.by_status.get(status_key, 0) + 1

                # By class
                class_key = c.change_class.value
                stats.by_class[class_key] = stats.by_class.get(class_key, 0) + 1

                # By type
                type_key = c.change_type.value
                stats.by_type[type_key] = stats.by_type.get(type_key, 0) + 1

                # Counts
                if c.is_pending:
                    stats.pending_count += 1
                if c.status == ChangeStatus.APPROVED or c.approval_record:
                    stats.approved_count += 1
                if c.status == ChangeStatus.REJECTED:
                    stats.rejected_count += 1
                if c.status == ChangeStatus.ROLLED_BACK:
                    stats.rolled_back_count += 1
                if c.change_class == ChangeClass.TRADING_IMPACTING:
                    stats.trading_impacting_count += 1

                # Approval time
                if c.approval_record and c.approval_record.approved_at:
                    delta = (c.approval_record.approved_at - c.created_at).total_seconds() / 60
                    approval_times.append(delta)

            if approval_times:
                stats.avg_approval_time_minutes = sum(approval_times) / len(approval_times)

            return stats

    # ========================================================================
    # Maintenance
    # ========================================================================

    def cleanup_expired(self) -> int:
        """Clean up expired changes."""
        count = 0
        now = datetime.now(timezone.utc)

        with self._lock:
            for change in self._changes.values():
                if change.is_pending and change.expires_at and now > change.expires_at:
                    change.status = ChangeStatus.EXPIRED
                    count += 1

        if count > 0:
            logger.info(f"Cleaned up {count} expired changes")

        return count

    # ========================================================================
    # Audit Logging
    # ========================================================================

    def _audit_change_created(self, change: ChangeRequest) -> None:
        """Audit change creation."""
        if not self._audit_service:
            return

        self._audit_service.log_access(
            workspace_id=change.workspace_id,
            principal=AuditPrincipal(
                id=change.requester_id,
                type="user",
                email=change.requester_email,
            ),
            action=AuditAction.CHANGE_REQUEST,
            resource_type=change.target_type,
            resource_id=change.target_id,
            resource_name=change.target_name,
            result=AuditResult.SUCCESS,
            sensitivity_level=SensitivityLevel.SENSITIVE,
            context={
                "change_id": change.id,
                "change_type": change.change_type.value,
                "change_class": change.change_class.value,
                "requires_approval": change.requires_approval,
                "requires_local_approval": change.requires_local_approval,
            },
        )

    def _audit_change_approved(
        self,
        change: ChangeRequest,
        approval: ApprovalRecord,
    ) -> None:
        """Audit change approval."""
        if not self._audit_service:
            return

        self._audit_service.log_access(
            workspace_id=change.workspace_id,
            principal=AuditPrincipal(
                id=approval.approved_by,
                type="user",
                email=approval.approved_by_email,
            ),
            action=AuditAction.CHANGE_APPROVE,
            resource_type=change.target_type,
            resource_id=change.target_id,
            result=AuditResult.SUCCESS,
            sensitivity_level=SensitivityLevel.CRITICAL,
            context={
                "change_id": change.id,
                "approval_id": approval.id,
                "approval_type": approval.approval_type.value,
                "user_acknowledged": approval.user_acknowledged,
                "evidence_hash": approval.evidence.compute_evidence_hash(),
            },
        )

    def _audit_change_rejected(
        self,
        change: ChangeRequest,
        rejector_id: str,
        reason: str,
    ) -> None:
        """Audit change rejection."""
        if not self._audit_service:
            return

        self._audit_service.log_access(
            workspace_id=change.workspace_id,
            principal=AuditPrincipal(id=rejector_id, type="user"),
            action=AuditAction.CHANGE_REJECT,
            resource_type=change.target_type,
            resource_id=change.target_id,
            result=AuditResult.SUCCESS,
            sensitivity_level=SensitivityLevel.SENSITIVE,
            context={
                "change_id": change.id,
                "rejection_reason": reason,
            },
        )

    def _audit_change_executed(
        self,
        change: ChangeRequest,
        executed_by: str,
    ) -> None:
        """Audit change execution."""
        if not self._audit_service:
            return

        self._audit_service.log_access(
            workspace_id=change.workspace_id,
            principal=AuditPrincipal(id=executed_by, type="user"),
            action=AuditAction.CHANGE_DEPLOY,
            resource_type=change.target_type,
            resource_id=change.target_id,
            result=AuditResult.SUCCESS,
            sensitivity_level=SensitivityLevel.CRITICAL,
            context={
                "change_id": change.id,
                "change_type": change.change_type.value,
            },
        )

    def _audit_change_rollback(
        self,
        change: ChangeRequest,
        rolledback_by: str,
        reason: str,
    ) -> None:
        """Audit change rollback."""
        if not self._audit_service:
            return

        self._audit_service.log_access(
            workspace_id=change.workspace_id,
            principal=AuditPrincipal(id=rolledback_by, type="user"),
            action=AuditAction.CHANGE_ROLLBACK,
            resource_type=change.target_type,
            resource_id=change.target_id,
            result=AuditResult.SUCCESS,
            sensitivity_level=SensitivityLevel.CRITICAL,
            context={
                "change_id": change.id,
                "rollback_reason": reason,
                "rollback_to_digest": change.rollback_to_digest,
            },
        )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Constants
    "APPROVAL_EXPIRY_HOURS",
    "MIN_APPROVAL_REASON_LENGTH",
    # Enums
    "ChangeClass",
    "ChangeType",
    "ChangeStatus",
    "ApprovalType",
    # Data classes
    "ChangeEvidence",
    "ApprovalRecord",
    "ChangeRequest",
    "ChangeJournalEntry",
    "ChangeStats",
    # Service
    "ChangeManagementService",
]
